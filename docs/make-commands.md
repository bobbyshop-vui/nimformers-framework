
# Make command guide

[English](make-commands.md) | [Tiếng Việt](make-commands.vi.md)

Run these commands from the `nimformers-framework` repository root, where the
`Makefile` lives. Default dataset and checkpoint paths are relative to this directory.


## Requirements

Put Nim and a C compiler on `PATH`. Runtime execution needs Metal on macOS or a
Vulkan loader/driver on a supported system. Check your tools:

```sh
nim --version
make --version
```

Download targets use `curl`; `bybylang-test` and `mamba-pipeline` use Bash.
The Python API needs Python >= 3.9 and NumPy; tests also need pytest.
Codestral training/conversion paths import `nimpy`; install it with
`nimble install nimpy` if the compiler reports that module missing. PyTorch
loading paths also need Python and the corresponding packages; see
[LoRA fine-tuning](finetune-lora-api.md).

Framework backends are `auto`, `metal`, and `vulkan`. `cuda` and `opencl` are
compatibility names redirected to Vulkan; they do not enable separate
CUDA/OpenCL runtimes. CPU model execution is no longer supported.


## Start with Spectron

```sh
make build
./test/spectron/test_nimformer
```

`make build` runs `nim check`, then `nim c`, on
`test/spectron/test_nimformer.nim`. The executable is
`test/spectron/test_nimformer`. Building does not train; executing the binary
starts corpus loading, tokenizer loading/training, and model training.

Combine building and running:

```sh
make run
```

Select Metal explicitly on macOS:

```sh
make run-metal
```

See the [Spectron guide](spectron-guide.md) for configuration, data, loss, and resume.


## Spectron and backend targets

- `make` or `make all`: runs `build`, then `run`, so **it starts training**.
  `run` checks/compiles again; use `make build` to compile only.
- `make build`: checks and compiles the trainer; the default backend is `auto`.
- `make run`: checks, compiles, and runs the trainer with `auto`.
- `make run-metal`: checks/compiles with `-d:backend=metal`, then trains.
  Outside macOS, it only prints that Metal is unsupported.
- `make run-opencl`: checks/compiles with backend name `opencl`, then runs
  through Vulkan under the current mapping.
- `make run-cuda`: the corresponding legacy `cuda` name, redirected to Vulkan.
- `make run-cpu`: the old target remains, but the executable fails when
  creating a CPU backend. Use Metal/Vulkan/auto.
- `make build-spectron`: checks and compiles the **chat program**, producing
  `./test_inferencespectron` at the repository root.
- `make run-spectron`: builds chat and runs the trained Spectron checkpoint.
  Enter prompts at `>>>`; use `exit` or `quit` to stop.

Despite its name, `test_nimformer` is the trainer. `test_inferencespectron` is
inference. The two binaries have different output locations.

The Spectron trainer and chat currently do not read CLI options for steps,
batch size, prompts, or checkpoint paths. `make run ARGS='--steps 100'` does
not change trainer configuration. `run-spectron` forwards `ARGS`, but the
binary does not read them yet.


## Download data and models

```sh
make download-datasets
make download-models
make download-codestral-tokenizer
```

- `download-datasets`: downloads `python_code_dataset_1.jsonl` through
  `python_code_dataset_5.jsonl` and `nim_python_code_dataset_mega.jsonl` into
  `test/spectron/`. It **does not download** `databricks-dolly-15k.jsonl`.
- `download-models`: downloads DeepSeek `.nimq`/config/tokenizer and Codestral
  base, adapter, config/tokenizer into `test/*/models/`. It also writes
  `jamba_parallel_lora.bin.hf_tokenizer_dir`.
- `download-codestral-tokenizer`: downloads only Codestral config/tokenizer
  assets into `INFER_LORA_DEFAULT_HFDIR`, useful when weights already exist.

Download targets use `curl` retry/resume. `download-models` fetches large
models; training Spectron does not require those DeepSeek/Codestral models.

## DeepSeek inference

```sh
make build-nim-inference
./nim_inference test/deepseek-inference/models \
  test/deepseek-inference/models/model.nimq auto \
  'Write a Python Fibonacci function.' 128 0.7 0.9 \
  'You are an AI programming assistant.'
```

`build-nim-inference` produces `./nim_inference` without running the model.
To build and run:

```sh
make infer-deepseek ARGS='test/deepseek-inference/models test/deepseek-inference/models/model.nimq auto "Write a Python Fibonacci function." 128 0.7 0.9 "You are an AI programming assistant."'
```

Argument order: `modelDir`, `nimqPath`, `backend`, `prompt`, `maxNewTokens`,
`temperature`, `topP`, `systemText`. Omitting `ARGS` uses defaults. Put prompts
in double quotes inside `ARGS` to keep them as one argument. See the
[LLaMA inference API](llama-inference-api.md).

## Codestral/Mamba LoRA

```sh
make build-infer-lora
make infer-lora BYBY_INFER_BACKEND=metal
```

`build-infer-lora` produces `./infer_lora_bin`, using `codestralInference` to
exclude PyTorch/Nimpy loaders from inference. `infer-lora` builds and runs.
Use `BYBY_INFER_BACKEND=vulkan` or `auto` as appropriate.

```sh
make infer-lora ARGS='test/codestral/models/jamba_parallel_lora.bin "[INST] Write a Python function to add two numbers.[/INST]" 0.7 test/codestral/models/hf test/codestral/models/codestral_mamba_only.nimq'
```

Argument order: `checkpoint`, `prompt`, `temperature`, `hfDir`, `nimqPath`.
The adapter checkpoint still needs base weights and a tokenizer; the loader
uses checkpoint metadata/sidecars. CLI `hfDir` and `nimqPath` configure the
base-Mamba fallback path. An explicitly supplied invalid checkpoint produces
an error; fallback is not a way to bypass an incorrect path. The example
prompt ends at `[/INST]`; the CLI trims trailing whitespace by default.

`make infer-mamba` is an alias of `infer-lora`, not a separate program.

### Fine-tune

```sh
make build-finetune
make finetune-mamba ARGS='codestral - test/codestral/models/hf nimq test/codestral/models/codestral_mamba_only.nimq test/codestral/models/my_lora.bin'
```

`build-finetune` produces `./finetune_lora_bin`. `finetune-mamba` builds and
trains. Argument order:

1. `initMode`: usually `codestral`, or `graft` for the older checkpoint pipeline.
2. `oldCheckpoint`: used with `initMode=graft`; `-` retains the default.
3. `hfDir`: config/tokenizer directory.
4. `hfSource`: `nimq` or `safetensors`.
5. `nimqPath`: quantized base when `hfSource=nimq`.
6. `outCheckpointPath`: output LoRA checkpoint.
7. `scratchDir`: optional temporary activation directory.

This CLI accepts `-` to retain a positional default. Some environment
variables override CLI values, including `BYBY_CHECKPOINT_PATH` and
`BYBY_SCRATCH_DIR`; `BYBY_SEQ_LEN` changes the Codestral trainer's sequence
length. They do not change Spectron's `CFG_*`. See
[fine-tuning and resume](finetune-lora-api.md).


### Automatic pipeline

```sh
SKIP_TO_FINETUNE=1 make mamba-pipeline
```

This calls `test/codestral/finetunemamba.sh`: quantize if needed, fine-tune,
then infer. It skips quantization if the base `.nimq` exists or
`SKIP_TO_FINETUNE=1`; the latter still requires the base file. Main variables:
`HF_DIR`, `OLD_MAMBA_NIMQ`, `LORA_CHECKPOINT`, `INFER_PROMPT`, and
`INFER_TEMPERATURE`. The script sets its own compile commands rather than
using all root Makefile `NIMFLAGS`.


### Graft an older Jamba checkpoint

```sh
make build-graft
make graft ARGS='path/to/mamba.bin MMAM path/to/grafted.bin'
```

`build-graft` produces `./graft_attention_bin`. The `MMAM` example assumes
three existing `M` blocks and inserts one attention `A` block. The pattern
must match the checkpoint architecture; see the [Jamba API](jamba-model-api.md).

The Makefile currently defaults `GRAFT_DEFAULT_OLD`/`GRAFT_DEFAULT_OUT` to
`.nimq`, but this branch of `graft_attention.nim` rejects `.nimq`. Therefore
bare `make graft` is not a working command for the default Codestral base.
Pass a valid Jamba `.bin` through `ARGS`, or use `make finetune-mamba` to
train directly from the Codestral `.nimq` base.


### Checkpoint conversion tools

```sh
make build-compact-lora
./compact_lora_checkpoint path/to/legacy_checkpoint.bin path/to/compact.nimq

make build-migrate-lora
./migrate_lora_json_to_bin path/to/legacy_checkpoint.bin path/to/migrated.bin
```

Both targets only build root-level executables. The compact tool reads
`<input>.lora.json` and writes compact weights plus metadata; it is not a
compressor for arbitrary `.bin` files. The migration tool converts older
JSON checkpoints to binary. Use a separate output as shown to retain the
original. See [checkpoint formats](checkpoint-formats.md).


## Python, C, and BybyLang

```sh
make build-for-python
make python-example
make python-test
make c-example
make bybylang-test
```

- `build-for-python`: checks/builds `lib_c_api.nim` into `libnimformers.dylib`
  on macOS, `libnimformers.so` on Linux, or `nimformers.dll` on Windows.
  It does not install the Python package.
- `python-example`: builds the shared library and runs `examples/python/basic.py`.
- `python-test`: builds the shared library and runs pytest on `test/python_test/`.
- `c-example`: builds the shared library, compiles `examples/c/smoke.c`, and
  runs `examples/c/smoke`. The recipe uses Unix-style linker options.
- `bybylang-test`: builds/runs the native DSL and C ABI scripts. Its runner
  uses the vendored compiler and compiles `lib_c_api.nim` into the AOT test;
  no Python package or prebuilt shared library is needed.

Use the active virtualenv interpreter:

```sh
make python-example PYTHON=python
make python-test PYTEST='python -m pytest' PYTEST_FLAGS=-q
```

See the [Python guide](python-guide.md), [Python API](python-api.md),
[BybyLang guide](bybylang-guide.md), and [BybyLang API](bybylang-api.md).


## Vulkan shaders

```sh
make shaders
```

Requires `glslc` on `PATH`. This compiles `.comp` files under
`vendor/bybylang/backends/vulkan/kernels/` into `.spv` when missing or older
than their source. The recipe still checks timestamps with `make -B shaders`;
`-B` does not bypass the timestamp condition inside the shell loop.


## Preview commands, select a compiler, and clean

```sh
make -n run-metal
make build NIM=/path/to/nim
```

`make -n` previews recipes without starting training. Command-line
`NIMFLAGS` **replaces all** default flags; retain the project's `--path`
options when overriding it. Build targets are `.PHONY`, so calling them runs
the compiler even if the binary exists. The compiler can reuse its cache.

```sh
make clean
```

This removes binaries listed in `CLEAN_BINS`, the shared library,
`nimcache/`, and root-level `model_*.nimq`. The current list does not include
`test/spectron/test_nimformer`, whose binary is next to its source, and it
does not reset `test/spectron/models/finetune.nimq` or the tokenizer. To
start fresh Spectron training, see the checkpoint section of the
[Spectron guide](spectron-guide.md).
