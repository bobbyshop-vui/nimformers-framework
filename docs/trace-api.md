# Trace API (`nf.trace`)

[English](trace-api.md) | [Tiếng Việt](trace-api.vi.md)

A **trace** records what a model run did on the GPU: the buffers, the micro-VM programs, every
dispatch, and the host inputs of each step. The FIR lifter (`firlift`) turns a trace into a
`model.fir`, which `formers-cli` then runs without the original code. `nf.trace` is the one API for
starting and stopping the recorder from Python, Nim, BybyLang and C.

```text
your model code ──nf.trace──▶ run.fprobe (+ run.fprobe.data) ──firlift──▶ model.fir ──formers-cli──▶ tokens
```

| Language | Entry point | Defined in |
| --- | --- | --- |
| Python | `nimformers.trace(...)`, `trace_active()`, `trace_file()`, `trace_stop()` | `nimformers/tracing.py` |
| Nim | `traceStart`, `traceStop`, `withTrace`, `traceStep`, `traceActive`, `traceFile` | `trace.nim` (also re-exported by `import nimformers`) |
| BybyLang | `import nim "trace"`, then the Nim names above | `trace.nim` |
| C | `nimf_trace_start`, `nimf_trace_stop`, `nimf_trace_active`, `nimf_trace_file`, `nimf_probe_step` | `include/nimformers.h` |

The recorder is process-wide: one trace at a time. Starting a trace while another is active stops the
previous one first.

## What to pass

| Parameter (Python / Nim) | Default | Meaning |
| --- | --- | --- |
| `path` | required | Trace file. Its directory must exist. |
| `steps` | `""` (everything) | Which host steps to record: comma-separated `decode:N` and `prefill:N`, for example `"decode:2"` or `"prefill:8,decode:2"`. A step is an open `step` marker (see [Step markers](#step-markers)); `prefill:N` counts tokens, `decode:N` counts steps. The lifter needs **at least two decode steps** to tell state that repeats from state that changes. |
| `data` | `True` | Also write the exact bytes of every host upload to `<path>.data`. Needed for weights, rope tables and any buffer that is not in a checkpoint. With `False` only fingerprints are stored and a checkpoint must supply the weights. |
| `data_max` / `dataMax` | `0` (unlimited) | With `data`, skip uploads larger than this many bytes. A skipped upload keeps its fingerprint, so a checkpoint can still match it. |

`<path>.data` is created when the first upload is captured. A trace of a model with no host uploads
inside a recorded step has no sidecar even with `data=True`.

> **Warning: `data=True` can fill your disk.** The sidecar is a byte-exact copy of every host upload
> inside the recorded steps, and a model driver that uploads its weights inside a step writes **the
> whole model** into it, in the dtype it uploaded. With the default `data_max=0` (unlimited) a
> 1.1 B-parameter model uploaded as f32 produced a 4.5 GB `trace.data`, and a 0.5 B-parameter model
> produced 2.1 GB. A 7 B model would write about 28 GB, and a model with several recorded steps that
> re-upload data writes it again for each step. The file is written while the model runs, so a full
> disk stops the run half way and can leave the machine unusable until space is freed.
>
> - Always set `data_max` (for example `4_000_000`) so large uploads are skipped and only small
>   tables such as rope cos/sin are kept. Skipped uploads keep their fingerprint, and the weights
>   then come from the checkpoint.
> - Check free space first, and delete `<path>.data` when the lift is done; it is a build artifact.
> - Never commit it: keep `trace.data` and `*.fprobe.data` in `.gitignore`.
> - The environment variable `BYBY_FIR_PROBE_DATA=1` has the same effect, and
>   `BYBY_FIR_PROBE_DATA_MAX` is its limit. Setting `BYBY_FIR_PROBE_DATA=1` without a limit is the
>   dangerous combination.

Invalid input raises (`NimformersError` in Python, `ValueError`/`IOError` in Nim, a non-zero status in
C) and leaves no trace active: a missing directory, a path containing a line break, a `steps` entry
that is not `decode:N`/`prefill:N`, or a negative count.

## Python

```python
import nimformers as nf

with nf.Context() as ctx:
    model = MyModel(ctx)                       # builds buffers, compiles kernels
    with nf.trace("run.fprobe", steps="decode:2"):
        for pos, tok in enumerate([1, 100]):
            model.step(tok)                    # Session.step() marks each step
```

| API | Meaning |
| --- | --- |
| `nf.trace(path, steps="", data=True, data_max=0)` | Starts recording and returns a `Trace`. Works as a context manager or with `.stop()`. |
| `Trace.stop()` | Flush the trace and close the sidecar. Calling it twice is harmless. |
| `Trace.active` | `True` while this recording is the active one. |
| `Trace.path`, `Trace.steps`, `Trace.data`, `Trace.data_max` | The arguments it was started with. |
| `Trace.data_path` | `path + ".data"`, or `None` when `data=False`. |
| `nf.trace_active()` | Any trace active in this process. |
| `nf.trace_file()` | Path of this process's trace, `""` if none was started. |
| `nf.trace_stop()` | Stop whatever is active. |

**Order does not matter in Python.** Kernels compiled before the recording started carry no bytecode in
it. `nf.trace` therefore compiles every kernel of every live `Session` again right after starting, so
the usual order (build the model, then `with nf.trace(...)`) produces a liftable trace.

Marking steps is done by `Session.step()` and `Session.set_step_inputs()`
(see [Python and C ABI](python-api.md#resident-buffers-and-sessions)). A step must write its token and
position scalars through `set_step_inputs` so the lifter can bind them to the runner's `token` and
`position` inputs.

## Nim

```nim
import nimformers          # re-exports trace.nim; `import trace` also works

withTrace("run.fprobe", "decode:2"):
  for pos, tok in prompt:
    traceStep(tsDecode, pos, pos, tok):
      runOneToken(tok)
```

| Symbol | Meaning |
| --- | --- |
| `traceStart(path, steps = "", data = true, dataMax = 0)` | Start recording. Raises `IOError`/`ValueError` on bad input. |
| `traceStop()` | Flush and close. Safe to call twice. Restores `BYBY_FIR_PROBE_DATA` and `BYBY_FIR_PROBE_DATA_MAX`. |
| `traceActive(): bool`, `traceFile(): string` | State queries. |
| `withTrace(path, body)`, `withTrace(path, steps, body)`, `withTrace(path, steps, data, dataMax, body)` | Start, run `body`, always stop. |
| `traceBegin(kind, index, position, token, tokens = 1)` / `traceEnd()` | Open and close one step. `kind` is `tsPrefill` or `tsDecode`. No-op when no trace is active. |
| `traceStep(kind, index, position, token): body` | `traceBegin` + `body` + `traceEnd` (closes on exceptions). |

In Nim the recorder only sees kernels compiled **after** the trace started. Call `traceStart` before
the model is built, or use the environment variables below.

## BybyLang

```bybylang
import nim "std/os"
import nim "../../trace"

traceStart(getTempDir() / "run.fprobe", "decode:2", false, 0)
# ... run the model ...
traceStop()
```

BybyLang passes `import nim` modules and calls straight through, so every Nim symbol above is
available, with the same start-before-compile rule. `test/bybylang_test/trace_test.bybylang` is the
runnable example (`make bybylang-test`).

## C

```c
nimf_trace_start("run.fprobe", "decode:2", 1, 0);   /* path, steps (NULL = all), data, data_max */
nimf_probe_step(1, 1, pos, pos, token);              /* begin=1, kind 1 = decode, index, position, token */
/* ... dispatch the step ... */
nimf_probe_step(0, 0, 0, 0, 0);                      /* end the step */
nimf_trace_stop();
```

| Function | Returns |
| --- | --- |
| `int32_t nimf_trace_start(const char* path, const char* steps, int32_t data, int64_t data_max)` | `0`, else `-1` with `nimf_last_error()` |
| `int32_t nimf_trace_stop(void)` | `0` (also when nothing was active) |
| `int32_t nimf_trace_active(void)` | `1` or `0` |
| `const char* nimf_trace_file(void)` | Path, valid until the next call |
| `int32_t nimf_probe_step(int32_t begin, int32_t kind, int64_t index, int64_t position, int64_t token)` | `0`, else `-1`. `kind`: `0` prefill, `1` decode. No-op when no trace is active. |

Like Nim, C callers must start the trace before compiling kernels.

## Environment variables

Setting these before the process starts is equivalent to calling the API at startup, and is the way
to trace a program you do not control (for example the Nim decode loops):

| Variable | Equivalent |
| --- | --- |
| `BYBY_FIR_PROBE=<path>` | `traceStart(path)` when the library loads. |
| `BYBY_FIR_PROBE_STEPS=decode:2` | `steps`. |
| `BYBY_FIR_PROBE_DATA=1` | `data=True`. |
| `BYBY_FIR_PROBE_DATA_MAX=<bytes>` | `data_max`. |
| `BYBY_ONE_DISPATCH` | Must be unset/`0` while tracing: the one-dispatch fast path is opaque and the trace records a note that it cannot be lifted. |

`traceStart` sets the two `DATA` variables for the duration of the trace and restores them in
`traceStop`.

## Step markers

A trace is only liftable if the recorder knows where each token step begins and ends and which
buffer bytes are the step's token and position. The Nim decode loops write these records themselves.
A Python (or C) driver writes them through:

- `Session.step(token, position, kind="decode")` / `nimf_probe_step`: opens and closes a `step`.
- `Session.set_step_inputs(buffer, token_offset, position_offset, token, position)` /
  `nimf_sess_set_step_inputs`: writes the two scalars and annotates them with the `token` and
  `position` roles.

## Trace file

A text file (`fprobe 2`) with `buf` (buffers), `prog` (programs), `step`/`end step`, `op` and `disp`
(dispatches with their program and bindings), `barrier`, `role`/`upload` records, and the program
words at the end. It is written as a complete snapshot via `<path>.tmp` and a rename, so an
interrupted run leaves the previous complete snapshot. Programs that were not recorded appear as
`note kind=missing_words` and `prog=none` on a dispatch; such a trace cannot be lifted.

## From trace to a running model

```sh
# 1. record two decode steps (here with the environment variables; `nf.trace` does the same in code)
BYBY_FIR_PROBE=run.fprobe BYBY_FIR_PROBE_STEPS=decode:2 BYBY_FIR_PROBE_DATA=1 \
    python examples/python/forge_vm.py models/forge-500m-chat --tokens 1,100,200 --new 0
# 2. lift
other/formers.n/firlift/firlift --trace run.fprobe --config model_config.json \
    --out models/forge-500m-chat-fir --report
# 3. check without weights, then run
other/formers.n/formers-cli --fir-check models/forge-500m-chat-fir/model.fir
other/formers.n/formers-cli --models models/forge-500m-chat-fir -m model.safetensors \
    --prompt-ids 1,100,200 -n 8 --temp 0
```

`firlift` options and the lifting rules are in [FIR script lifting](fir-script-lift.md); the file
format is in [FIR](fir.md). A model with no usable `tokenizer.json` is driven with `--prompt-ids` and
prints token ids.

## Limits

- One recorder per process; the recorder is not re-entrant across threads (the library serializes it).
- `steps` is a count of steps to **keep**; steps beyond it still run but are not recorded.
- Position-dependent program generation or per-step shape changes need a lowering rule; the lifter
  reports what it cannot do rather than guessing.
