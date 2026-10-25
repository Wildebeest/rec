import errno
import hmac
import json
import logging
import select
import socket
import threading
from http.server import ThreadingHTTPServer

from .engine import Sampling


class RequestError(ValueError):
    pass


def make_server(
    engine,
    address,
    metrics_path,
    *,
    defaults=Sampling(),
    api_key=None,
    grammar_mode="flint",
    sleep_idle_seconds: float = 0.0,
):
    from mlx_lm.server import APIHandler

    from .scheduler import Generator

    admission = threading.BoundedSemaphore(9)
    ignored_fields = set()
    ignored_fields_lock = threading.Lock()

    class Handler(APIHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(
                generator, *args, system_fingerprint=engine.identity["model_id"], **kwargs
            )

        def reply(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authenticated(self):
            if api_key is not None and not hmac.compare_digest(
                self.headers.get("Authorization", "").encode(),
                ("Bearer " + api_key).encode(),
            ):
                self.reply(401, {"error": "Invalid or missing bearer token"})
                return False
            return True

        def do_GET(self):
            if not self.authenticated():
                return
            if self.path == "/health":
                stopping = generator.stopping.is_set()
                self.reply(
                    503 if stopping else 200,
                    {
                        "status": "stopping" if stopping else "ready",
                        "busy": generator.busy.is_set(),
                        "queued": generator.requests.qsize(),
                        "sleeping": bool(getattr(engine, "sleeping", False)),
                    },
                )
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": "flint", "object": "model"}]})
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if not self.authenticated():
                return
            if self.path == "/sleep":
                if generator.stopping.is_set():
                    self.reply(503, {"error": "The server is stopping"})
                    return
                result = generator.sleep_if_idle()
                self.reply(409 if result["busy"] else 200, result)
                return
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                if not 0 < int(self.headers["Content-Length"]) <= 8 * 1024 * 1024:
                    raise ValueError("Request body must be between 1 byte and 8 MiB")
            except (ValueError, TypeError) as error:
                self.reply(400, {"error": str(error)})
                return
            if generator.stopping.is_set():
                self.reply(503, {"error": "The server is stopping"})
                return
            if not admission.acquire(blocking=False):
                self.reply(
                    429, {"error": "The request queue is full; retry after a generation finishes"}
                )
                return
            try:
                super().do_POST()
            except RequestError as error:
                self.reply(400, {"error": str(error)})
            except (BrokenPipeError, ConnectionResetError):
                logging.info("Client disconnected; cancelling its generation")
            finally:
                admission.release()

        def is_disconnected(self):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
            except ValueError:
                # The handler can close its socket while the generation worker checks it.
                return True
            except OSError as error:
                if error.errno not in (errno.EBADF, errno.ECONNRESET):
                    raise
                return True

        def _set_completion_headers(self, status):
            # mlx-lm maps failures before generation to 404, including a stopped worker.
            if status == 404 and generator.stopping.is_set():
                status = 503
            super()._set_completion_headers(status)

        def generate_response(self, *args, **kwargs):
            response = super().generate_response(*args, **kwargs)
            for choice in response["choices"]:
                if "logprobs" in choice:
                    for entry in choice["logprobs"]["content"]:
                        entry.update(
                            token=engine.tokenizer.decode([entry["id"]], skip_special_tokens=False),
                            bytes=None,
                            top_logprobs=[],
                        )
            return response

        def validate_model_parameters(self):
            try:
                self.validate_request()
            except (ValueError, TypeError, KeyError) as error:
                raise RequestError(str(error)) from error

        def drop_unknown_fields(self):
            allowed = {
                "model",
                "messages",
                "tools",
                "tool_choice",
                "stream",
                "stream_options",
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "max_completion_tokens",
                "seed",
                "stop",
                "frequency_penalty",
                "repetition_penalty",
                "presence_penalty",
                "logprobs",
                "min_p",
                "store",
                "grammar",
                "parallel_tool_calls",
            }
            unknown = set(self.body) - allowed
            for field in unknown:
                del self.body[field]
            with ignored_fields_lock:
                for field in sorted(unknown - ignored_fields):
                    logging.warning("Ignoring unsupported request field %r", field)
                ignored_fields.update(unknown)
            self.requested_draft_model = "default_model"
            self.num_draft_tokens = self.response_generator.cli_args.num_draft_tokens
            self.adapter = None
            self.repetition_context_size = 20
            self.presence_context_size = 20
            self.frequency_context_size = 20
            self.xtc_probability = 0.0
            self.xtc_threshold = 0.0
            self.logit_bias = None
            self.top_logprobs = -1
            self.chat_template_kwargs = None

        def validate_request(self):
            self.drop_unknown_fields()
            super().validate_model_parameters()
            if self.stream and self.logprobs:
                raise ValueError("Token logprobs require stream=false")
            if self.requested_model != "flint":
                raise ValueError("The served model is flint")
            for key in ("frequency_penalty", "min_p"):
                if self.body.get(key, 0) != 0:
                    raise ValueError(f"Nonzero {key} is not supported by this decoder")
            if self.body.get("repetition_penalty", 1) != 1:
                raise ValueError("Only neutral repetition_penalty=1 is supported")
            if self.body.get("store", False) is not False:
                raise ValueError("Stored-completion retrieval is not supported; use store=false")
            stop = self.body.get("stop")
            if stop is not None:
                stops = [stop] if isinstance(stop, str) else stop
                if not isinstance(stops, list) or any(
                    not isinstance(word, str) or not word for word in stops
                ):
                    raise ValueError("stop must be a nonempty string or a list of nonempty strings")
            choice = self.body.get("tool_choice", "auto")
            if choice not in ("auto", "none"):
                raise ValueError(
                    "tool_choice currently supports auto and none; forced calls are not implemented"
                )
            if choice == "none":
                self.body.pop("tools", None)
            mode = self.body["grammar"] if "grammar" in self.body else grammar_mode
            if mode not in ("none", "flint"):
                raise ValueError("grammar must be none or flint")
            if mode == "flint" and self.body.get("stop"):
                raise ValueError("Flint grammar cannot be combined with custom stop strings")
            if (
                "parallel_tool_calls" in self.body
                and type(self.body["parallel_tool_calls"]) is not bool
            ):
                raise ValueError("parallel_tool_calls must be a boolean")
            if self.body.get("parallel_tool_calls") is False:
                raise ValueError("parallel_tool_calls=false is not supported")
            messages = self.body["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a nonempty list")
            for message in messages:
                if message["role"] not in ("system", "developer", "user", "assistant", "tool"):
                    raise ValueError("Unsupported message role")
            if self.stream_options is not None and self.stream_options != {"include_usage": True}:
                raise ValueError("stream_options must be {include_usage: true}")

        def handle_completion(self, request, stop_words):
            from mlx_lm.server import process_message_content

            from .engine import ContextLimitError

            try:
                for message in request.messages:
                    if "reasoning" in message:
                        if "reasoning_content" in message:
                            raise ValueError("Use one reasoning field per message")
                        message["reasoning_content"] = message.pop("reasoning")
                process_message_content(request.messages)
                mode = self.body["grammar"] if "grammar" in self.body else grammar_mode
                request.constraint = None
                if mode == "flint":
                    try:
                        request.constraint = engine.constraints.tools(request.tools)
                    except RuntimeError as error:
                        raise ValueError(f"Grammar compilation failed: {error}") from error
                ids = engine.prompt(request.messages, tools=request.tools)
                remaining = engine.context_limit - len(ids)
                if remaining < 1:
                    raise ContextLimitError("The prompt fills the configured context limit")
                explicit_maximum = self.body.get("max_completion_tokens")
                if explicit_maximum is None:
                    explicit_maximum = self.body.get("max_tokens")
                self.max_tokens = (
                    remaining if explicit_maximum is None else min(self.max_tokens, remaining)
                )
                Sampling(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    presence_penalty=self.presence_penalty,
                    max_tokens=self.max_tokens,
                    seed=defaults.seed if self.seed is None else self.seed,
                )
            except (ValueError, TypeError, KeyError) as error:
                self.reply(400, {"error": str(error)})
                return
            request.prompt_tokens = ids
            request.is_disconnected = self.is_disconnected
            request.generation_context = None
            try:
                super().handle_completion(request, stop_words)
            finally:
                # Headers can fail before mlx-lm enters its streaming cleanup block.
                if request.generation_context is not None:
                    request.generation_context.stop()

    if address[0] in ("0.0.0.0", "::"):
        class DualStackHTTPServer(ThreadingHTTPServer):
            address_family = socket.AF_INET6

            def server_bind(self):
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                return super().server_bind()

        server = DualStackHTTPServer(("::", address[1]), Handler)
    else:
        server = ThreadingHTTPServer(address, Handler)
    generator = Generator(
        engine, metrics_path, defaults, sleep_idle_seconds=sleep_idle_seconds
    )
    server.generator = generator
    return server
