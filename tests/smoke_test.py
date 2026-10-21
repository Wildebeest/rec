#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

binary = str(Path(sys.argv[1]).resolve())


def chat(endpoint, key, **body):
    request = urllib.request.Request(
        endpoint + "/chat/completions",
        data=json.dumps(
            {
                "model": "smoke",
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 16,
                **body,
            }
        ).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode()


def serve(env, log, *model):
    server = subprocess.Popen(
        [binary, "serve", *model, "--port", "11599"], env=env, stdout=log, stderr=subprocess.STDOUT
    )
    deadline = time.time() + 600
    while True:
        assert server.poll() is None, Path(log.name).read_text()
        status = subprocess.run([binary, "status"], env=env, capture_output=True, text=True)
        if status.returncode == 0 and json.loads(status.stdout)["ready"]:
            return server, json.loads(status.stdout)
        assert time.time() < deadline, Path(log.name).read_text()
        time.sleep(1)


with tempfile.TemporaryDirectory(prefix="rec-smoke-", dir="/tmp") as td:
    root = Path(td)
    # The smoke model is public; a stale Hugging Face token must not affect it.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XDG_", "REC_", "HF_"))}
    env["HF_HOME"] = str(root / "hf")
    env["XDG_DATA_HOME"] = str(root / "data")
    wrong = subprocess.run([binary, "serve", "flint"], env=env, capture_output=True, text=True)
    assert wrong.returncode != 0 and 'contains "smoke", not "flint"' in wrong.stderr, wrong.stderr
    for attempt, model in (("download", ["smoke"]), ("cached", [])):
        with open(root / f"{attempt}.log", "w+") as log:
            server, status = serve(env, log, *model)
            try:
                assert status["model"] == "smoke" and status["context"] == 2048, status
                key = subprocess.check_output([binary, "auth", "token"], env=env, text=True).strip()
                greedy = json.loads(chat(status["endpoint"], key, temperature=0))
                assert greedy["choices"][0]["message"]["content"].strip(), greedy
                streamed = chat(
                    status["endpoint"], key, temperature=1, top_p=0.95, top_k=20, stream=True
                )
                assert streamed.rstrip().endswith("data: [DONE]"), streamed
                try:
                    chat(status["endpoint"], "wrong")
                    raise AssertionError("wrong API key accepted")
                except urllib.error.HTTPError as error:
                    assert error.code == 401
                subprocess.run([binary, "stop"], env=env, check=True, capture_output=True)
                assert server.wait(timeout=30) == 0, Path(log.name).read_text()
            finally:
                server.kill()
            output = Path(log.name).read_text()
            assert ("✓ Downloaded" in output) == (attempt == "download"), output
            print(
                f"{attempt}: ready, authenticated, greedy and sampled generation, stop", flush=True
            )
    print("Smoke test passed")
