"""Python interface to the formers.n command line program."""

import os
import shlex
import subprocess
import threading
from pathlib import Path


def _resolve_cli(cli_path):
    candidate = cli_path or os.environ.get("FORMERS_CLI")
    if candidate:
        return str(candidate)
    return str(Path(__file__).resolve().parents[2] / "formers-cli")


class Model:
    def __init__(self, models_dir, backend="auto", cli_path=None, max_tokens=512, **opts):
        self.models_dir = str(models_dir)
        self.cli_path = _resolve_cli(cli_path)
        self.max_tokens = max_tokens
        self.options = {"--backend": backend}
        self.options.update(opts)

    def _base_args(self):
        args = [self.cli_path, "--models", self.models_dir]
        for key, value in self.options.items():
            flag = key if str(key).startswith("-") else "--" + str(key).replace("_", "-")
            if value is False or value is None:
                continue
            if value is True:
                args.append(flag)
            else:
                args.extend((flag, str(value)))
        return args

    def generate(self, prompt, max_tokens=128, temperature=0.8, stream=False):
        args = self._base_args() + ["-p", str(prompt), "-n", str(max_tokens), "--temp", str(temperature)]
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        errors = []
        def collect_errors():
            errors.extend(process.stderr.readlines())
            process.stderr.close()

        thread = threading.Thread(target=collect_errors, daemon=True)
        thread.start()
        if stream:
            def chunks():
                try:
                    for line in process.stdout:
                        yield line
                finally:
                    process.stdout.close()
                    code = process.wait()
                    thread.join()
                    if code:
                        raise RuntimeError("formers-cli failed: " + "".join(errors).strip())
            return chunks()
        output = process.stdout.read()
        process.stdout.close()
        code = process.wait()
        thread.join()
        if code:
            raise RuntimeError("formers-cli failed: " + "".join(errors).strip())
        return output.rstrip("\n")

    def chat(self, max_tokens=None, temperature=None, stderr_callback=None):
        return Chat(self, max_tokens=max_tokens, temperature=temperature,
                    stderr_callback=stderr_callback)


class Chat:
    def __init__(self, model, max_tokens=None, temperature=None, stderr_callback=None):
        args = model._base_args() + ["-cnv", "-n", str(model.max_tokens if max_tokens is None else max_tokens), "--no-progress"]
        if temperature is not None:
            args.extend(("--temp", str(temperature)))
        self._process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, bufsize=1)
        self._errors = []
        def collect_errors():
            for line in self._process.stderr:
                self._errors.append(line)
                if stderr_callback is not None:
                    stderr_callback(line.rstrip("\r\n"))
            self._process.stderr.close()

        self._stderr_thread = threading.Thread(target=collect_errors, daemon=True)
        self._stderr_thread.start()
        self._closed = False
        # The CLI prints a blank line followed by its initial prompt.
        self._read_until_prompt()

    def _read_until_prompt(self):
        output = []
        while True:
            char = self._process.stdout.read(1)
            if not char:
                if self._process.poll() is not None:
                    self._stderr_thread.join()
                    raise RuntimeError("formers-cli chat ended: " + "".join(self._errors).strip())
                continue
            if char == ">":
                if self._process.stdout.read(1) == " ":
                    return "".join(output).strip("\r\n")
            output.append(char)

    def send(self, message):
        if self._closed:
            raise RuntimeError("chat is closed")
        lines = str(message).splitlines() or [""]
        if len(lines) > 1:
            self._process.stdin.write("\n".join(line + "\\" for line in lines[:-1]) + "\n")
        self._process.stdin.write(lines[-1] + "\n")
        self._process.stdin.flush()
        return self._read_until_prompt()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            self._process.stdin.write("/exit\n")
            self._process.stdin.flush()
            self._process.wait(timeout=10)
        self._stderr_thread.join(timeout=1)
        for pipe in (self._process.stdin, self._process.stdout, self._process.stderr):
            if not pipe.closed:
                pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
