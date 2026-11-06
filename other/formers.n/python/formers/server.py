"""OpenAI-compatible HTTP API for a formers.n model."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import threading
import time
import uuid

from . import Model

logger = logging.getLogger(__name__)


class _ChatSession:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()
        self.chat = None
        self.messages = []
        self.settings = None
        self.last_action = "not started"

    def _start(self, settings, replay, reason):
        self.last_action = "restarted (%s)" % reason
        if self.chat is not None:
            self.chat.close()
        self.chat = self.model.chat(max_tokens=settings[0], temperature=settings[1],
                                    stderr_callback=lambda line: logger.info("formers-cli: %s", line))
        self.settings = settings
        for message in replay:
            if message.get("role") == "user":
                self.chat.send(message.get("content", ""))

    def complete(self, messages, max_tokens, temperature):
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty array")
        if any(not isinstance(message, dict) or not isinstance(message.get("role"), str)
               for message in messages):
            raise ValueError("each message must have a role")
        last = messages[-1]
        if last["role"] != "user":
            raise ValueError("the last message must be from the user")
        content = last.get("content", "")
        if not isinstance(content, str):
            raise ValueError("user message content must be a string")

        settings = (max_tokens, temperature)
        prefix_matches = (len(messages) == len(self.messages) + 1 and
                          messages[:-1] == self.messages and last["role"] == "user")
        if self.chat is None:
            self._start(settings, messages[:-1], "first request")
        elif not prefix_matches:
            self._start(settings, messages[:-1], "history mismatch")
        elif settings != self.settings:
            self._start(settings, messages[:-1], "generation settings changed")
        else:
            self.last_action = "reused"
        result = self.chat.send(content)
        self.messages = [dict(message) for message in messages] + [
            {"role": "assistant", "content": result}
        ]
        return result

    def close(self):
        with self.lock:
            if self.chat is not None:
                self.chat.close()
                self.chat = None


def _error(message, error_type="invalid_request_error"):
    return {"error": {"message": str(message), "type": error_type, "param": None, "code": None}}


def _error_message(exc):
    message = str(exc)
    if "formers-cli" not in message:
        return message
    tagged_lines = [line.strip() for line in message.splitlines() if "[formers.n]" in line]
    error_lines = [line for line in tagged_lines if "error" in line.lower()]
    if error_lines:
        return error_lines[-1]
    if tagged_lines:
        return tagged_lines[-1]
    return message


def create_server(model, host="127.0.0.1", port=8000):
    """Create a server instance, primarily for embedding and tests."""
    session = _ChatSession(model)

    class Handler(BaseHTTPRequestHandler):
        server_version = "formers.n/1.0"

        def handle_one_request(self):
            self._request_started = time.monotonic()
            self._response_status = None
            self._chat_log_details = ""
            try:
                super().handle_one_request()
            finally:
                method = getattr(self, "command", "?") or "?"
                path = getattr(self, "path", "?") or "?"
                status = self._response_status or 0
                duration = time.monotonic() - self._request_started
                logger.info("%s %s status=%s duration=%.3fs%s", method, path, status,
                            duration, self._chat_log_details)

        def send_response(self, code, message=None):
            self._response_status = code
            super().send_response(code, message)

        def _json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/v1/models":
                self._json(404, _error("The requested endpoint was not found.", "not_found_error"))
                return
            model_id = str(getattr(model, "models_dir", "formers-model"))
            self._json(200, {"object": "list", "data": [{
                "id": model_id, "object": "model", "created": 0, "owned_by": "formers"
            }]})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._json(404, _error("The requested endpoint was not found.", "not_found_error"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request body must be a JSON object")
                messages = request.get("messages")
                max_tokens = request.get("max_tokens", getattr(model, "max_tokens", 512))
                temperature = request.get("temperature", 0.7)
                if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
                    raise ValueError("max_tokens must be a positive integer")
                if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature < 0:
                    raise ValueError("temperature must be a non-negative number")
                stream = request.get("stream", False)
                if not isinstance(stream, bool):
                    raise ValueError("stream must be a boolean")
                self._chat_log_details = " stream=%s messages=%s" % (
                    "yes" if stream else "no", len(messages) if isinstance(messages, list) else 0)
                with session.lock:
                    content = session.complete(messages, max_tokens, temperature)
                    action = session.last_action
                completion_id = "chatcmpl-" + uuid.uuid4().hex
                created = int(time.time())
                model_name = request.get("model") or str(getattr(model, "models_dir", "formers-model"))
                if stream:
                    self._stream(content, completion_id, created, model_name)
                else:
                    self._json(200, {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    })
                self._chat_log_details += " chat=%s reply_chars=%s" % (action, len(content))
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                logger.exception("Request failed: %s", exc)
                if self._chat_log_details:
                    self._chat_log_details += " chat=%s" % session.last_action
                self._json(400, _error(exc))
            except Exception as exc:
                logger.exception("Request failed: %s", exc)
                if self._chat_log_details:
                    self._chat_log_details += " chat=%s" % session.last_action
                self._json(500, _error(_error_message(exc), "server_error"))

        def _stream(self, content, completion_id, created, model_name):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            def event(chunk):
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode("utf-8"))
                self.wfile.flush()

            event({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                   "model": model_name, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                        "finish_reason": None}]})
            if content:
                event({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                       "model": model_name, "choices": [{"index": 0, "delta": {"content": content},
                                                            "finish_reason": None}]})
            event({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                   "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.chat_session = session
    return server


def serve(model, host="127.0.0.1", port=8000):
    """Serve the OpenAI-compatible API until interrupted."""
    server = create_server(model, host, port)
    models_dir = os.path.abspath(model.models_dir)
    backend = model.options.get("--backend", "auto")
    logger.info("models dir: %s", models_dir)
    logger.info("backend: %s", backend)
    logger.info("cli path: %s", os.path.abspath(model.cli_path))
    bound_host, bound_port = server.server_address[:2]
    logger.info("listening on http://%s:%s", bound_host, bound_port)
    logger.info("the first chat request loads the model")
    try:
        server.serve_forever()
    finally:
        server.chat_session.close()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Serve a formers model with an OpenAI-compatible API.")
    parser.add_argument("--models", required=True, help="Model directory")
    parser.add_argument("--backend", default="auto", help="Execution backend")
    parser.add_argument("--cli", help="Path to formers-cli")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--max-tokens", type=int, default=512, help="Default generation limit")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Log warnings and errors only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else
                        logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not os.path.isdir(args.models):
        parser.error("models directory does not exist: %s" % os.path.abspath(args.models))
    serve(Model(args.models, backend=args.backend, cli_path=args.cli, max_tokens=args.max_tokens),
          host=args.host, port=args.port)


if __name__ == "__main__":
    main()
