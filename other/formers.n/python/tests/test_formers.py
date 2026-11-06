import os
import tempfile
import unittest
from pathlib import Path

from formers import Model


class FormersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cli = Path(self.temp.name) / "formers-cli"
        self.cli.write_text("""#!/usr/bin/env python3
import sys
args = sys.argv[1:]
if '-cnv' not in args:
    if '--fail' in args:
        print('runtime issue', file=sys.stderr)
        sys.exit(1)
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

    def tearDown(self):
        self.temp.cleanup()

    def test_generate_and_stream(self):
        model = Model("models", cli_path=self.cli, backend="cpu", fir_f32=True)
        self.assertEqual(model.generate("hello"), "answer:hello")
        self.assertEqual(list(model.generate("world", stream=True)), ["answer:world"])

    def test_chat_keeps_process_for_turns(self):
        with Model("models", cli_path=self.cli).chat() as chat:
            self.assertEqual(chat.send("first"), "reply:first")
            self.assertEqual(chat.send("second"), "reply:second")
        with self.assertRaises(RuntimeError):
            chat.send("closed")

    def test_chat_sends_multiline_message_with_continuations(self):
        with Model("models", cli_path=self.cli).chat() as chat:
            self.assertEqual(chat.send("first line\nsecond line\nthird line"),
                             "reply:first line\nsecond line\nthird line")

    def test_chat_passes_max_tokens(self):
        with Model("models", cli_path=self.cli, max_tokens=37).chat() as chat:
            self.assertIn(["-n", "37"], [chat._process.args[i:i + 2]
                                           for i in range(len(chat._process.args) - 1)])

    def test_cli_failure_includes_stderr(self):
        model = Model("models", cli_path=self.cli, fail=True)
        with self.assertRaisesRegex(RuntimeError, "runtime issue"):
            model.generate("x")


if __name__ == "__main__":
    unittest.main()
