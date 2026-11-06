import json
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from http.client import HTTPConnection
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from formers import Model
from formers.server import create_server, main


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cli = Path(self.temp.name) / "formers-cli"
        self.cli.write_text("""#!/usr/bin/env python3
import sys
args = sys.argv[1:]
if args[args.index('--models') + 1] == 'bad-model':
    print('[formers.n] loading model', file=sys.stderr, flush=True)
    print('[formers.n] error: model load failed', file=sys.stderr, flush=True)
    sys.exit(1)
if '-cnv' not in args:
    print('answer:' + args[args.index('-p') + 1], end='', flush=True)
else:
    while True:
        print('\\n> ', end='', flush=True)
        lines = []
        for line in sys.stdin:
            msg = line.rstrip('\\r\\n')
            if msg.endswith('\\\\'):
                lines.append(msg[:-1])
                continue
            lines.append(msg)
            break
        else:
            break
        msg = '\\n'.join(lines).strip()
        if msg in ('/exit', '/quit', '/bye'):
            break
        print('reply:' + msg + '\\n', end='', flush=True)
""")
        self.cli.chmod(0o755)
        model = Model("test-model", cli_path=self.cli, max_tokens=32)
        self.server = create_server(model, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.chat_session.close()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method, path, body=None):
        connection = HTTPConnection("127.0.0.1", self.port)
        encoded = None if body is None else (body if isinstance(body, bytes) else json.dumps(body))
        connection.request(method, path, body=encoded, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = response.read()
        result = response.status, response.getheader("Content-Type"), payload
        connection.close()
        return result

    def test_non_stream_completion(self):
        status, content_type, payload = self.request("POST", "/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 19, "temperature": 0.2,
        })
        result = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["choices"][0]["message"], {"role": "assistant", "content": "reply:hello"})
        self.assertEqual(result["usage"], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    def test_stream_completion(self):
        status, content_type, payload = self.request("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello stream"}], "stream": True,
        })
        body = payload.decode("utf-8")
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", content_type)
        self.assertIn('"object": "chat.completion.chunk"', body)
        self.assertIn('"content": "reply:hello stream"', body)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    def test_models(self):
        status, _, payload = self.request("GET", "/v1/models")
        result = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(result["data"][0]["id"], "test-model")

    def test_unknown_path(self):
        status, _, payload = self.request("GET", "/unknown")
        self.assertEqual(status, 404)
        self.assertIn("error", json.loads(payload))

    def test_bad_json(self):
        status, _, payload = self.request("POST", "/v1/chat/completions", b"{")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(payload))

    def test_cli_error_body_contains_only_last_error_line(self):
        server = create_server(Model("bad-model", cli_path=self.cli), "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_address[1])
            connection.request("POST", "/v1/chat/completions", body=json.dumps({
                "messages": [{"role": "user", "content": "hello"}],
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            result = json.loads(response.read())
            connection.close()
            self.assertEqual(response.status, 500)
            self.assertEqual(result["error"]["message"], "[formers.n] error: model load failed")
        finally:
            server.shutdown()
            server.chat_session.close()
            server.server_close()
            thread.join(timeout=2)


class StartupValidationTests(unittest.TestCase):
    def test_main_rejects_missing_models_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stderr = StringIO()
            missing = Path(temp_dir) / "missing"
            with patch("sys.argv", ["formers.server", "--models", str(missing)]), \
                    redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("models directory does not exist:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
