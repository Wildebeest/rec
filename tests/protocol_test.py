#!/usr/bin/env python3

import http.client
import http.server
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

python, binary = (str(Path(arg).absolute()) for arg in sys.argv[1:3])
server_script = Path(__file__).with_name("scripted_flint_server.py")
key = "a" * 64
context = 65536
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def connect():
    return http.client.HTTPConnection("127.0.0.1", port, timeout=60)


def post(body, key=key):
    connection = connect()
    payload = {"model": "flint", "temperature": 0, **body}
    connection.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(payload),
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    return connection, connection.getresponse()


def chat(content, history=(), **body):
    messages = [*history, {"role": "user", "content": content}]
    connection, response = post({"messages": messages, **body})
    data = response.read().decode()
    connection.close()
    assert response.status == 200, (response.status, data)
    return json.loads(data)


def stream(content, history=(), **body):
    messages = [*history, {"role": "user", "content": content}]
    connection, response = post(
        {"messages": messages, "stream": True, "stream_options": {"include_usage": True}, **body}
    )
    assert response.status == 200, (response.status, response.read())
    events = []
    for line in response:
        line = line.decode().strip()
        if line == "data: [DONE]":
            break
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    connection.close()
    deltas = [choice["delta"] for event in events for choice in event["choices"]]
    finish = [
        choice["finish_reason"]
        for event in events
        for choice in event["choices"]
        if choice["finish_reason"]
    ]
    return deltas, finish, events[-1]["usage"]


def slow_stream():
    connection, response = post(
        {
            "messages": [{"role": "user", "content": "Repeat slowly."}],
            "stream": True,
            "max_tokens": 4000,
        }
    )
    assert response.status == 200, response.status
    while not response.readline().startswith(b"data: "):
        pass

    def hang_up():
        response.close()
        connection.close()

    return hang_up


def metrics():
    return [json.loads(line) for line in metrics_path.read_text().splitlines()]


def health():
    connection = connect()
    connection.request("GET", "/health", headers={"Authorization": "Bearer " + key})
    state = json.loads(connection.getresponse().read())
    connection.close()
    return state


def wait_idle():
    deadline = time.time() + 30
    while health()["busy"] or health()["queued"]:
        assert time.time() < deadline, health()
        time.sleep(0.1)


def check_protocol():
    connection = connect()
    connection.request("GET", "/v1/models")
    assert connection.getresponse().status == 401
    connection.close()
    _, response = post({"messages": [{"role": "user", "content": "Answer with 4."}]}, key="wrong")
    assert response.status == 401, response.status

    answer = chat("Answer with 4.")
    choice = answer["choices"][0]
    assert choice["message"]["content"] == "4" and choice["finish_reason"] == "stop", answer
    assert choice["message"]["reasoning"] == "Answer briefly.", answer

    deltas, finish, usage = stream("Answer with 4.")
    assert "".join(d.get("content") or "" for d in deltas) == "4", deltas
    assert "".join(d.get("reasoning") or "" for d in deltas) == "Answer briefly.", deltas
    assert finish == ["stop"] and usage["completion_tokens"] > 0, (finish, usage)

    messages = [{"role": "user", "content": "Call get_weather with city=Paris."}]
    called = chat(messages[0]["content"], tools=tools)
    choice = called["choices"][0]
    assert choice["finish_reason"] == "tool_calls", called
    (call,) = choice["message"]["tool_calls"]
    assert call["function"]["name"] == "get_weather", called
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}, called

    deltas, finish, _ = stream(messages[0]["content"], tools=tools)
    streamed = [c for d in deltas for c in d.get("tool_calls") or []]
    assert [c["function"]["name"] for c in streamed] == ["get_weather"], deltas
    assert finish == ["tool_calls"], finish

    messages += [
        choice["message"],
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": '{"city":"Paris","temperature_c":18}',
        },
    ]
    connection, response = post({"messages": messages, "tools": tools})
    answered = json.loads(response.read())
    connection.close()
    content = answered["choices"][0]["message"]["content"]
    assert content == 'The tool returned: {"city":"Paris","temperature_c":18}', answered

    limited = chat("Repeat forever.", max_tokens=16)
    assert limited["choices"][0]["finish_reason"] == "length", limited
    assert limited["usage"]["completion_tokens"] == 16, limited
    padding = [{"role": "system", "content": "x" * (context - 100)}]
    _, finish, usage = stream("Repeat forever.", history=padding)
    assert finish == ["length"] and usage["total_tokens"] == context, (finish, usage)
    connection, response = post({"messages": [{"role": "user", "content": "x" * context}]})
    assert response.status == 400 and b"fills the configured context" in response.read()
    connection.close()

    slow_stream()()
    wait_idle()
    assert metrics()[-1]["status"] == "cancelled", metrics()[-1]
    assert chat("Answer with 5.")["choices"][0]["message"]["content"] == "5"

    hang_up = slow_stream()
    with ThreadPoolExecutor(8) as pool:
        waiting = [pool.submit(chat, "Answer with 6.") for _ in range(8)]
        deadline = time.time() + 30
        while health()["queued"] < 8:
            assert time.time() < deadline, health()
            time.sleep(0.05)
        connection, refused = post({"messages": [{"role": "user", "content": "Answer with 7."}]})
        assert refused.status == 429, refused.status
        connection.close()
        hang_up()
        assert all(f.result()["choices"][0]["message"]["content"] == "6" for f in waiting)
    print(
        "protocol: auth, reasoning, streaming, tool calls and results, length, context, "
        "cancellation, admission",
        flush=True,
    )


def check_clients(root):

    class Control(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = json.dumps(
                {
                    "ready": True,
                    "model": "flint",
                    "engine": "mlx",
                    "revision": "test",
                    "context": context,
                    "endpoint": f"http://127.0.0.1:{port}/v1",
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    state, config = root / "data/rec/state", root / "data/rec/config"
    state.mkdir(parents=True)
    config.mkdir(parents=True)
    (config / "api-key").write_text(key)
    control = socketserver.UnixStreamServer(str(state / "control.sock"), Control)
    threading.Thread(target=control.serve_forever, daemon=True).start()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENCODE_", "XDG_", "REC_"))}
    env.update(
        XDG_DATA_HOME=str(root / "data"),
        XDG_CACHE_HOME=str(root / "cache"),
        OPENCODE_TEST_HOME=str(root / "home"),
        PI_CODING_AGENT_DIR=str(root / "pi"),
    )
    for name in (
        "DISABLE_PROJECT_CONFIG",
        "DISABLE_CLAUDE_CODE",
        "DISABLE_EXTERNAL_SKILLS",
        "DISABLE_DEFAULT_PLUGINS",
        "DISABLE_AUTOUPDATE",
        "DISABLE_MODELS_FETCH",
        "DISABLE_LSP_DOWNLOAD",
    ):
        env["OPENCODE_" + name] = "true"
    work = root / "work"
    work.mkdir()
    (work / "secret.txt").write_text("rec-protocol-canary\n")
    clients = {
        "pi": [
            "--no-skills",
            "--no-context-files",
            "--no-prompt-templates",
            "--offline",
            "--no-session",
            "--mode",
            "json",
            "-p",
        ],
        "opencode": ["run", "--format", "json"],
    }
    try:
        for client, args in clients.items():
            prompt = f"Call read with path={work / 'secret.txt'}."
            # OpenCode quotes message arguments that contain spaces; pass it words.
            words = prompt.split() if client == "opencode" else [prompt]
            r = subprocess.run(
                [binary, "launch", client, "--", *args, *words],
                cwd=work,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert r.returncode == 0, (r.returncode, r.stderr[-2000:], r.stdout[-2000:])
            assert "The tool returned:" in r.stdout and "rec-protocol-canary" in r.stdout, r.stdout[
                -2000:
            ]
            print(f"{client}: read-tool round trip through rec launch", flush=True)
    finally:
        control.shutdown()
        control.server_close()


with tempfile.TemporaryDirectory(prefix="rec-protocol-", dir="/tmp") as td:
    root = Path(td)
    port = free_port()
    metrics_path = root / "metrics.jsonl"
    (root / "api-key").write_text(key)
    log = open(root / "server.log", "w+")
    server = subprocess.Popen(
        [
            python,
            "-I",
            "-B",
            str(server_script),
            "serve",
            "flint",
            "--model-dir",
            str(root),
            "--manifest",
            str(root / "manifest.json"),
            "--port",
            str(port),
            "--context-limit",
            str(context),
            "--api-key-file",
            str(root / "api-key"),
            "--metrics",
            str(metrics_path),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + 120
        while "Listening on" not in Path(log.name).read_text():
            assert server.poll() is None and time.time() < deadline, Path(log.name).read_text()
            time.sleep(0.2)
        check_protocol()
        check_clients(root)
        hang_up = slow_stream()
        server.send_signal(signal.SIGTERM)
        assert server.wait(timeout=30) == 0, Path(log.name).read_text()
        hang_up()
        print("shutdown: active generation cancelled, clean exit", flush=True)
    except BaseException:
        print(Path(log.name).read_text()[-4000:], file=sys.stderr)
        raise
    finally:
        server.kill()
    print("Protocol test passed")
