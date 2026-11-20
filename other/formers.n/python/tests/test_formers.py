import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from formers import Model, check_fir


class _BatchGroup:
    def __init__(self):
        self.call = None

    def generate_batch(self, cli_path, base_args, prompts, **kwargs):
        self.call = (cli_path, base_args, prompts, kwargs)
        if kwargs["token_ids"]:
            return [prompt.replace(",", " ") for prompt in prompts]
        return ["answer:" + prompt for prompt in prompts]


class FormersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cli = Path(self.temp.name) / "formers-cli"
        self.cli.write_text("""#!/usr/bin/env python3
import os
import sys
args = sys.argv[1:]
if '--detect-only' in args:
    print('arch: fir', file=sys.stderr)
    sys.exit(0)
if '--fir-check' in args:
    print('checked:' + args[args.index('--fir-check') + 1])
    sys.exit(0)
if '-cnv' not in args:
    if '--fail' in args:
        print('runtime issue', file=sys.stderr)
        sys.exit(1)
    if '--fir' in args:
        for flag in ('--model', '--lora', '--config', '--tokenizer', '--fir'):
            print(flag + '=' + args[args.index(flag) + 1], file=sys.stderr)
        if os.getenv('BYBY_CODESTRAL_TOKEN_IDS') == '1':
            print('[codestral-tokens] 781 1117', file=sys.stderr)
        if 'BYBY_FIR_DECODE_REPLAY' not in os.environ:
            print('decode replay: default', file=sys.stderr)
    if '--prompt-ids' in args:
        assert args.count('--prompt-ids') == 1
        print(args[args.index('--prompt-ids') + 1].replace(',', ' '))
    else:
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

    def test_codestral_fir_uses_original_adapter_and_reports_diagnostics(self):
        model = Model(
            "models", cli_path=self.cli, backend="vulkan", model="base.nimq",
            lora="original adapter.bin", config="lifted/config.json", tokenizer="hf",
            fir="lifted/model.fir",
            env={"BYBY_CODESTRAL_TOKEN_IDS": 1, "BYBY_FIR_DECODE_REPLAY": None},
        )
        messages = []
        with patch.dict(os.environ, {"BYBY_FIR_DECODE_REPLAY": "0"}):
            text = model.generate("a prompt\nwith a '$'", temperature=0,
                                  stderr_callback=messages.append)
            self.assertEqual(os.environ["BYBY_FIR_DECODE_REPLAY"], "0")
        self.assertEqual(text, "answer:a prompt\nwith a '$'")
        self.assertEqual(messages, [
            "--model=base.nimq", "--lora=original adapter.bin", "--config=lifted/config.json",
            "--tokenizer=hf", "--fir=lifted/model.fir", "[codestral-tokens] 781 1117",
            "decode replay: default",
        ])

    def test_generate_ids_replaces_default_prompt_ids(self):
        model = Model("models", cli_path=self.cli, prompt_ids="0,1")
        self.assertEqual(model.generate_ids([5, 17]), [5, 17])
        with self.assertRaises(ValueError):
            model.generate_ids([])

    def test_generate_batch_preserves_order_and_model_options(self):
        model = Model("models", cli_path=self.cli, backend="vulkan", env={"MODEL_BATCH_ENV": "kept"})
        group = _BatchGroup()
        self.assertEqual(model.generate_batch(["first", "second", "third"], group=group,
                                              ratios=[1, 2], max_tokens=9, temperature=0),
                         ["answer:first", "answer:second", "answer:third"])
        self.assertEqual(group.call[2], ["first", "second", "third"])
        self.assertEqual(group.call[1], ["--models", "models", "--backend", "vulkan"])
        self.assertEqual(group.call[3]["ratios"], [1, 2])

    def test_generate_ids_batch_uses_ids_mode_and_empty_is_noop(self):
        model = Model("models", cli_path=self.cli, prompt_ids="99")
        group = _BatchGroup()
        self.assertEqual(model.generate_ids_batch([[4, 8], [2]], group=group), [[4, 8], [2]])
        self.assertEqual(group.call[2], ["4,8", "2"])
        self.assertEqual(group.call[1], ["--models", "models", "--backend", "auto"])
        self.assertEqual(model.generate_ids_batch([], group=group), [])

    def test_inspection_commands(self):
        self.assertEqual(Model("models", cli_path=self.cli).detect(), "arch: fir")
        self.assertEqual(check_fir(Path("lifted/model.fir"), cli_path=self.cli),
                         "checked:lifted/model.fir")

    def test_lift_fir_records_the_native_cli_with_original_weights_and_adapter(self):
        lifter = self.cli.parent / "firlift" / "firlift"
        lifter.parent.mkdir()
        lifter.write_text("""#!/usr/bin/env python3
import json
import sys
print(json.dumps(sys.argv[1:]), flush=True)
print('recorded in memory: upload data 0 bytes', file=sys.stderr, flush=True)
""")
        lifter.chmod(0o755)
        model = Model("models", cli_path=self.cli, backend="vulkan", lora="original adapter.bin",
                      fir="lifted/model.fir", config="lifted/config.json", prompt_ids="0,1")
        logs = []
        path = model.lift_fir("prompt", "lifted", config="hf/config.json", weights="base.nimq",
                              context=512, log_callback=logs.append)
        command = json.loads(logs[0])
        run = command.index("--run")
        self.assertEqual(command[:run], ["--config", "hf/config.json", "--weights", "base.nimq",
                                         "--out", "lifted", "--report", "--context", "512"])
        self.assertEqual(command[run + 1:], [
            str(self.cli), "--models", "models", "--backend", "vulkan", "--lora", "original adapter.bin",
            "--config", "hf/config.json", "-m", "base.nimq", "-p", "prompt", "-n", "4", "--temp", "0",
            "--repeat-penalty", "1.0", "--no-progress",
        ])
        self.assertEqual(logs[1], "recorded in memory: upload data 0 bytes")
        self.assertEqual(path, Path("lifted/model.fir"))

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
