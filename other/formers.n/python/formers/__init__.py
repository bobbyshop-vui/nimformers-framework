"""Python interface to the formers.n command line program."""

import os
import shutil
import subprocess
import threading
from operator import index
from pathlib import Path


def _resolve_cli(cli_path):
    candidate = cli_path or os.environ.get("FORMERS_CLI")
    if candidate:
        return str(candidate)
    return str(Path(__file__).resolve().parents[2] / "formers-cli")


def _check_exit(code, details, command="formers-cli"):
    if code:
        raise RuntimeError(f"{command} failed: " + details.strip())


def check_fir(path, *, cli_path=None, fir_f32=False):
    """Validate the FIR file and return the CLI summary without loading weights."""
    args = [_resolve_cli(cli_path), "--fir-check", str(path)]
    if fir_f32:
        args.append("--fir-f32")
    result = subprocess.run(args, capture_output=True, text=True)
    _check_exit(result.returncode, result.stderr)
    return result.stdout.rstrip("\n")


class Model:
    """Run a model with the formers CLI and forward model/runtime options."""

    def __init__(self, models_dir, backend="auto", cli_path=None, max_tokens=512,
                 *, env=None, **opts):
        self.models_dir = str(models_dir)
        self.cli_path = _resolve_cli(cli_path)
        self.max_tokens = max_tokens
        self.env = dict(env or ())
        self.options = {"--backend": backend}
        self.options.update(opts)

    def _subprocess_env(self):
        if not self.env:
            return None
        environ = os.environ.copy()
        for key, value in self.env.items():
            if value is None:
                environ.pop(key, None)
            else:
                environ[key] = str(value)
        return environ

    def _base_args(self, *, exclude=()):
        args = [self.cli_path, "--models", self.models_dir]
        for key, value in self.options.items():
            flag = key if str(key).startswith("-") else "--" + str(key).replace("_", "-")
            if flag in exclude or value is False or value is None:
                continue
            if value is True:
                args.append(flag)
            else:
                args.extend((flag, str(value)))
        return args

    def detect(self):
        """Return the architecture detection report without running inference."""
        result = subprocess.run(self._base_args() + ["--detect-only"],
                                capture_output=True, text=True, env=self._subprocess_env())
        _check_exit(result.returncode, result.stderr)
        return result.stderr.rstrip("\n")

    def lift_fir(self, prompt, out_dir, *, config, weights, context=None,
                 trace_tokens=4, firlift_path=None, log_callback=None):
        """Run the native recorder via ``firlift --run`` and return ``model.fir``."""
        environ = self._subprocess_env()
        search_path = os.pathsep.join(os.get_exec_path(environ))
        cli = Path(shutil.which(self.cli_path, path=search_path) or self.cli_path)
        lifter = firlift_path or cli.parent / "firlift" / "firlift"
        args = [str(lifter), "--config", str(config), "--weights", str(weights),
                "--out", str(out_dir), "--report"]
        if context is not None:
            args.extend(("--context", str(context)))
        native = self._base_args(exclude=("--fir", "--prompt-ids", "--config", "--model", "-m"))
        native.extend(("--config", str(config), "-m", str(weights), "-p", str(prompt),
                       "-n", str(trace_tokens), "--temp", "0", "--repeat-penalty", "1.0", "--no-progress"))
        result = subprocess.run(args + ["--run"] + native, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=environ)
        if log_callback is not None:
            for line in result.stdout.splitlines():
                log_callback(line)
        _check_exit(result.returncode, result.stdout, command="firlift")
        return Path(out_dir) / "model.fir"

    def generate(self, prompt, max_tokens=128, temperature=0.8, stream=False,
                 *, stderr_callback=None):
        """Return CLI text or stream its lines. The callback receives diagnostic lines."""
        args = self._base_args() + ["-p", str(prompt), "-n", str(max_tokens), "--temp", str(temperature)]
        return self._generate(args, stream, stderr_callback)

    def generate_ids(self, prompt_ids, max_tokens=128, temperature=0.8, *, stderr_callback=None):
        """Pass token IDs to a FIR graph and return the generated token IDs."""
        encoded = ",".join(str(index(token_id)) for token_id in prompt_ids)
        if not encoded:
            raise ValueError("prompt_ids must contain at least one token")
        args = self._base_args(exclude=("--prompt-ids",)) + [
            "--prompt-ids", encoded, "-n", str(max_tokens), "--temp", str(temperature),
        ]
        output = self._generate(args, False, stderr_callback)
        try:
            return [int(token) for token in output.split()]
        except ValueError as exc:
            raise RuntimeError("formers-cli did not return space-separated token IDs") from exc

    def _generate(self, args, stream, stderr_callback):
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, bufsize=1, env=self._subprocess_env())
        errors = []
        callback_errors = []

        def collect_errors():
            with process.stderr:
                for line in process.stderr:
                    errors.append(line)
                    if stderr_callback is not None and not callback_errors:
                        try:
                            stderr_callback(line.rstrip("\r\n"))
                        except Exception as exc:
                            callback_errors.append(exc)

        thread = threading.Thread(target=collect_errors, daemon=True)
        thread.start()

        def finish():
            process.stdout.close()
            code = process.wait()
            thread.join()
            _check_exit(code, "".join(errors))
            if callback_errors:
                raise callback_errors[0]

        if stream:
            def chunks():
                try:
                    yield from process.stdout
                finally:
                    finish()
            return chunks()
        try:
            return process.stdout.read().rstrip("\n")
        finally:
            finish()

    def chat(self, max_tokens=None, temperature=None, stderr_callback=None):
        """Open a persistent CLI conversation."""
        return Chat(self, max_tokens=max_tokens, temperature=temperature,
                    stderr_callback=stderr_callback)


class Chat:
    def __init__(self, model, max_tokens=None, temperature=None, stderr_callback=None):
        args = model._base_args() + ["-cnv", "-n", str(model.max_tokens if max_tokens is None else max_tokens), "--no-progress"]
        if temperature is not None:
            args.extend(("--temp", str(temperature)))
        self._process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, bufsize=1,
                                         env=model._subprocess_env())
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
        try:
            self._read_until_prompt()
        except RuntimeError:
            self.close()
            raise

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
