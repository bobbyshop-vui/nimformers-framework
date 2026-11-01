# Nimformer Framework

[English](readme.md) | [Tiếng Việt](readme.vi.md)

Nimformer is a Nim framework for transformer inference and training with Metal and Vulkan GPU backends. The API references, usage guides, file formats, and BybyLang GPU programming docs are in [`docs/`](docs/README.md).

Start with the [documentation index](docs/README.md), then see [Getting Started](docs/getting-started.md) for a minimal program.

Practical guides:

- [Make commands](docs/make-commands.md): build versus run, binary outputs,
  required data, `ARGS`, backends, and cleanup.
- [Train/chat Spectron with `test_nimformer`](docs/spectron-guide.md): configuration,
  JSONL, BPE, checkpoints/resume, loss, and a Nim import example.
- [Python from source to API](docs/python-guide.md): wrapper/shared-library setup,
  NumPy, quantization, NIMQ, micro-VM, and troubleshooting.
- [BybyLang walkthrough](docs/bybylang-guide.md): compiler, DSL/AOT, sessions,
  micro-kernels, and the C ABI.
- [formers.n](#run-deepseek-or-codestral-with-formersn-formers-cli): one llama.cpp-style
  `formers-cli` for DeepSeek/Llama and Codestral (Mamba-2, LoRA), single prompt or chat.

Model-specific programs and data live under `test/spectron/`, `test/codestral/`, and `test/deepseek-inference/`. Model assets live in each project's `models/` folder. The FP4 GEMV micro-kernel modules live under `kernelfortest/`; shared runtime modules remain at the project root.


## Python and C ABI

The core framework is available as a shared library with a stable C ABI
(`lib_c_api.nim`). It exposes backend selection, owned tensors, every math /
neural-network kernel, int8/int4/fp8/fp4/custom-float quantization, NIMQ
state dicts, the micro-VM bytecode builder and dispatch, resident weights,
tokenizers, and Llama inference.

The `cpu` backend runs the same API, sessions, resident weights, and
micro-kernels without a GPU. It uses runtime-dispatched SSE4/AVX2/AVX-512/
NEON kernels with a small-op SSE policy (AVX frequency offset) and L3-aware
matmul blocking; see [`docs/cpu-backend.md`](docs/cpu-backend.md).

Build it and run the Python tests:

```sh
make build-for-python
make python-test
```

```python
import numpy as np
from nimformers import Context

with Context() as ctx:                 # auto -> Metal/Vulkan
    a = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(8, 16).astype(np.float32)
    print(ctx.matmul(a, b).shape)
    print(ctx.softmax(a).sum(axis=-1))
```

The Python package lives in `nimformers/`, the C header in
`include/nimformers.h`, examples in `examples/python/` and `examples/c/`, and
the full reference in [`docs/python-api.md`](docs/python-api.md).

The same ABI can be imported from the vendored BybyLang compiler:

```sh
make bybylang-test     # test/bybylang_test/*.bybylang
```

Measure FP32 matmul GFLOPS on CPU, Metal, and Vulkan:

```sh
python3 benchmarks/bench_gflops.py --backends cpu,metal,vulkan

# 8096^3 tiled mode (avoids the iGPU watchdog for one huge dispatch)
python3 benchmarks/bench_gflops.py --gpu-8096
```


## AI model co-authors and acknowledgements

The following model developers are acknowledged. Their GitHub handles are linked for reference:

- **Mistral AI** ([@mistralai](https://github.com/mistralai)): Devstral, Codestral, Mistral Large
- **RWKV** ([@RWKV](https://github.com/RWKV)): RWKV-3
- **Anthropic** ([@anthropics](https://github.com/anthropics), [@claude](https://github.com/claude)): Claude Sonnet 4.6, Claude Sonnet 5, Claude Sonnet 5.5, Claude Opus 4.6, Claude Opus 5, Claude Opus 5.5
- **Google DeepMind** ([@google-deepmind](https://github.com/google-deepmind)): Gemini 3.8 Flash, Gemini 3.7 Flash
- **Moonshot AI** ([@MoonshotAI](https://github.com/MoonshotAI)): Kimi K3
- **Allen Institute for AI (Ai2)** ([@allenai](https://github.com/allenai)): SERA-8B
- **NVIDIA** ([@nvidia](https://github.com/nvidia)): Nemotron 3 Ultra
- **DeepSeek** ([@deepseek-ai](https://github.com/deepseek-ai)): DeepSeek V4 Pro, DeepSeek V4.1 Flash
- **Meta** ([@facebookresearch](https://github.com/facebookresearch), [@meta-llama](https://github.com/meta-llama)): Muse Spark, Llama 3.2
- **Alibaba / Qwen team** ([@QwenLM](https://github.com/QwenLM)): Qwen-Coder-Next
- **TinyLlama** ([@TinyLlama](https://github.com/TinyLlama)): TinyLlama 1.1B
- **OpenAI** ([@openai](https://github.com/openai)): GPT-6 Luna, GPT-5.6 Luna, GPT-6.1 Sol
- **xAI** ([@xai-org](https://github.com/xai-org)): Grok 4.6, Grok 4.7

This is an AI model and developer acknowledgement list; it does not replace individual human authorship or imply endorsement by the developers.


## Hugging Face models and inference

These repositories contain the model files used by the inference targets in this project:

| Repository | Contents | Make target |
| --- | --- | --- |
| [DeepSeek Coder 6.7B NimQ](https://huggingface.co/bobbydeveloper2014/deepseek-coder-6.7b-nimq) | `model.nimq`, `config.json`, and tokenizer files for `test/deepseek-inference/nim_inference.nim` | `infer-deepseek`, `formers` |
| [Codestral v0.1 7B NimQ](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-7b-nimq) | Codestral base weights, model config, and Tekken/HF tokenizer files | `infer-lora` base, `formers` |
| [Codestral v0.1 QLoRA LinearAttention](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-nimq-qlora-linearattention) | `jamba_parallel_lora.bin` adapter checkpoint; it uses the Codestral NimQ base above | `infer-lora` adapter, `formers` |


### Inference benchmarks

Prefill and decode tok/s for each model on a Colab Tesla T4 (Vulkan) and a
MacBook Pro with Intel Iris Plus 640 (Metal and Vulkan/MoltenVK), with the
test machines and the commands used:

| Model | Benchmark |
| --- | --- |
| DeepSeek Coder 6.7B (`nim_inference`) | [`test/deepseek-inference/benchmark.md`](test/deepseek-inference/benchmark.md) |
| Codestral Mamba + QLoRA (`infer_lora`) | [`test/codestral/benchmark.md`](test/codestral/benchmark.md) |


### Download the models

The Hugging Face repositories are public. From the repository root, run this target to download all DeepSeek and Codestral assets into the paths used by the inference targets:

```sh
make download-models
```

The target creates the model directories under `test/` as needed, downloads the base weights, tokenizer/config files and adapter, then writes the Codestral tokenizer-directory sidecar used by `infer-lora`. It uses `curl` with retries and resume support.

If Codestral weights are already present but `newHfTokenizer` reports missing
files, download only its tokenizer/config assets:

```sh
make download-codestral-tokenizer
```

Codestral inference reads `tokenizer.model.v3` (or `tokenizer.model`) directly
with the pure Nim tokenizer in `test/codestral/sentencepiece_native.nim`. It uses
the model's BPE/Unigram rules, normalization and byte fallback. No Python tokenizer
packages or SentencePiece library installation is required. `make build-infer-lora`
builds inference with `-d:codestralInference`, excluding the PyTorch/Nimpy loaders.


### Run DeepSeek Coder with `nim_inference`

`infer-deepseek` builds `test/deepseek-inference/nim_inference.nim`. The first argument is the local model directory (for `config.json` and tokenizer files), the second is the `.nimq` weights file, and the remaining arguments are backend, prompt, token limit, temperature, top-p, and system prompt.

```sh
make infer-deepseek ARGS='./test/deepseek-inference/models ./test/deepseek-inference/models/model.nimq auto "Write a Python function for Fibonacci." 128 0.7 0.9 "You are an AI programming assistant."'
```

Use `metal`, `vulkan`, or `auto` as the backend argument when supported on the machine.

`nim_inference` loads the model config and tokenizer with Nim's standard library.
It reads the local DeepSeek byte-level BPE vocabulary, merge rules, added tokens
and chat template directly; Nimpy, Transformers and Python tokenizer packages are
not required. Build it with `make build-nim-inference`.

Pass `--no-progress` to hide the live tok/s line. Speeds:
[`test/deepseek-inference/benchmark.md`](test/deepseek-inference/benchmark.md).


### Run Codestral with the QLoRA adapter

The adapter file does not contain the base Mamba weights. `infer_lora` reads the base `.nimq` and uses the tokenizer directory recorded in the checkpoint sidecar. `make download-models` writes the default tokenizer sidecar. Run inference from the project root:

```sh
make infer-lora \
  ARGS='test/codestral/models/jamba_parallel_lora.bin "[INST] Write a Python function for Fibonacci.[/INST]" 0.7 test/codestral/models/hf test/codestral/models/codestral_mamba_only.nimq'
```

`infer-lora` accepts `<checkpoint.bin> <prompt> <temperature> [hfDir] [nimqPath]`. The binary metadata selects the base `.nimq`; the sidecar points the tokenizer loader at the current base-model directory. The last two CLI arguments configure the base-Mamba fallback path. `infer-mamba` is an alias for `infer-lora`. `make infer-lora` defaults to `auto`, preserving automatic backend selection on macOS and other platforms. It checks the selected device type: dGPU weights go to device-local VRAM; iGPU weights use host-visible/shared memory. To override the backend explicitly, use `make infer-lora BYBY_INFER_BACKEND=metal` or `make infer-lora BYBY_INFER_BACKEND=vulkan`.

Speeds: [`test/codestral/benchmark.md`](test/codestral/benchmark.md).


### Run DeepSeek or Codestral with formers.n (`formers-cli`)

formers.n (`other/formers.n/`) is a llama.cpp-style command-line runner written in
BybyLang on top of the nimformers blocks. Point it at a model directory: it detects the
architecture from `config.json` and the `.nimq` header, then runs the matching path with
one binary and one set of flags:

| Detected | Path | Example directory |
| --- | --- | --- |
| Llama family (`LlamaForCausalLM`, DeepSeek Coder) | `nim_inference` transformer blocks | `test/deepseek-inference/models` |
| Mamba-2 (Codestral Mamba) | `infer_lora` base-Mamba path | a directory without a LoRA `.bin` |
| Mamba-2 + LoRA / Linear++ | `infer_lora` checkpoint path | `test/codestral/models` |

#### Build

```sh
make formers-cli
```

This builds the BybyLang compiler in `vendor/bybylang` when needed, compiles
`other/formers.n/src/formers_cli.bybylang` ahead of time to Nim and writes
`other/formers.n/formers-cli`. `make build-formers` is the same target. Only Nim and
a C compiler are required; the GPU backends are the same Metal/Vulkan backends as the
other inference targets.

#### Quick start

Download the models first (`make download-models`), then from the repository root:

```sh
# Codestral + QLoRA, multi-turn chat
other/formers.n/formers-cli --models test/codestral/models -cnv -n 256

# DeepSeek Coder, one prompt
other/formers.n/formers-cli --models test/deepseek-inference/models \
  -p "Write a Python function for Fibonacci." -n 128

# Only print what was detected, without loading the weights
other/formers.n/formers-cli --models test/codestral/models --detect-only
```

The same through make (`make formers` builds first):

```sh
make formers                                              # chat with test/codestral/models
make formers FORMERS_MODELS=test/deepseek-inference/models
make formers ARGS='--models test/codestral/models -p "Write quicksort in Python." -n 200'
```

#### What `--models` must contain

| File | Found as | Override |
| --- | --- | --- |
| weights | the only `.nimq`; with several, the one named in the LoRA `nimqPath` metadata, else `model.nimq` | `-m FILE` |
| config | `config.json` in the directory or a subdirectory (Codestral keeps it in `hf/`) | `--config FILE` |
| tokenizer | a directory with `tokenizer.json` or `tokenizer.model` | `--tokenizer DIR` |
| LoRA (Mamba-2) | the only `.bin` with a `NIMBIN1`/`NIMBIN4` header; several is an error | `--lora FILE`, `--no-lora` |

`--detect-only` prints the files, the architecture, the route and the model sizes, and
says why a model is not supported (for example Mamba-1 or an unknown `model_type`).

#### Single prompt and chat

- Without `-cnv`, `-p TEXT` (or `-f FILE`) is wrapped in the model's chat template and
  generated once; `--system TEXT` sets the system prompt and `--raw-prompt` sends the text
  as is.
- `-cnv` (or `-i`) starts a chat: the context (KV cache for Llama, recurrent state for
  Mamba-2) is kept between turns, `-n` limits each reply and `-p` becomes the system
  prompt. Type `/exit` (or `/quit`, `/bye`) to leave, end a line with `\` to continue on
  the next one, press Ctrl+C to stop a reply and Ctrl+C at the prompt to quit.
- `-c N` sets the Llama chat context size (default 1024 tokens).
- In chat each reply ends with its `prompt eval` and `eval` tok/s; a single Llama prompt
  ends with the `Speed stats` block (prefill, decode, total).

Before the first prompt a warmup compiles the GPU pipelines and prints
`ready: warmup ...`; `--no-warmup` skips it, and the first reply then pays the compile
time. For Mamba-2, prompts shorter than 129 tokens run through the decode kernels already
compiled by the warmup; longer prompts send whole 128-token chunks
(`BYBY_PREFILL_BATCH_SIZE`) through the parallel prefill. On a Vulkan dGPU with tensor
cores (`prefill=tensor-cores-only` in the `--verbose` device line) the rest of a prompt
also prefills in 16-token chunks, later chat turns prefill too (carrying the recurrent
state in), and the warmup compiles the 16-token kernels; every prefill GEMM then runs on
the tensor cores, never on a scalar kernel. `BYBY_VULKAN_TENSOR_CORES=tune` lets the
scalar kernels compete again, `0` turns the tensor cores off.

#### LoRA merge

A Mamba-2 LoRA is merged into the resident FP4-KM weights on the GPU at startup
(`infer_lora.mergeResidentLoraIfEnabled`, the same merge as `make infer-lora`). The decode
kernels then skip the A/B adapter work, and the output matches a merged checkpoint. The
merge adds a one-time pass to startup.

- `--no-merge-lora` (or `BYBY_MERGE_LORA=0`) keeps the adapters separate and fused in the
  kernels.
- On by default, the merge steps aside by itself when `BYBY_ONE_DISPATCH=1`,
  `BYBY_MODEL_MMAP=0` or the process has no memory headroom (`BYBY_MEMORY_LIMIT_MB`,
  default 7000), and inference continues with the separate adapters. Set
  `BYBY_MERGE_LORA=inplace` by hand to make those cases errors instead.

#### Logs

formers-cli is quiet by default. The loader, LoRA, JIT, Metal and kernel-tuning logs go to
`/dev/null`; only the loading line, `ready`, the generated text, the statistics and errors
are shown. `-v`/`--verbose` shows every log line. An error in quiet mode prints its message
and suggests rerunning with `--verbose`.

#### Main flags

| Flag | Meaning | Default |
| --- | --- | --- |
| `-n, --n-predict N` | tokens to generate (per reply in chat) | 128 |
| `--temp T` | temperature, `0` = greedy | 0.7 |
| `--top-p P` | nucleus sampling, `1` = off | 0.9 |
| `--min-p P` | min-p, `0` = off | 0.05 |
| `--repeat-penalty R`, `--repeat-last-n N` | repetition penalty and its window | 1.0 (off), 64 |
| `-s, --seed N` | sampling seed | random |
| `--backend NAME` | `auto`, `metal`, `vulkan`, `cuda` (= `vulkan`) | `auto` |
| `-ngl N` | layers on the GPU; all layers run on the GPU, a smaller value is rejected | all |
| `--no-progress` | no tok/s in the terminal window title | |

`formers-cli --help` lists every flag. Exit codes: 0 ok, 1 runtime error, 2 bad options,
3 unsupported model.

#### Make targets

| Target | What it does |
| --- | --- |
| `make formers-cli` / `make build-formers` | build `other/formers.n/formers-cli` |
| `make formers` | build, then chat with `FORMERS_MODELS` (default `test/codestral/models`); `ARGS='...'` replaces every argument |
| `make formers-test` | architecture-detection test, no model load and no GPU |
| `make formers-smoke` | builds a tiny random Llama and runs it end to end on the GPU |
| `make formers-clean` | removes the formers.n build and binary (`make clean` does too) |

More detail (source layout, the detection rules, what is not done yet):
[`other/formers.n/README.md`](other/formers.n/README.md).


### Train Spectron with `test/spectron/test_nimformer.nim`

`test/spectron/test_nimformer.nim` is the Spectron training program. `make run` compiles and runs it; it reads `test/spectron/databricks-dolly-15k.jsonl` and all six Python/Nim code JSONLs in `test/spectron/`, trains or loads its byte-level BPE tokenizer, trains the model, and saves the checkpoint and tokenizer under `test/spectron/models/`. It scans each code JSONL and keeps a reservoir sample of up to 4,000 valid examples per file, so it uses all six datasets without loading their combined 1.3 GB into memory. The BPE training slice samples from each source. Change the `CFG_*` constants near the top of `test/spectron/test_nimformer.nim` to adjust the datasets, sample limit, model size, step count, batch, sequence length, or learning rate.


#### Download the Python and Nim code datasets

The five Python code shards are published in [bobbydeveloper2014/python-code](https://huggingface.co/datasets/bobbydeveloper2014/python-code). The additional Nim/Python dataset is in [bobbydeveloper2014/mega-nim-python-code](https://huggingface.co/datasets/bobbydeveloper2014/mega-nim-python-code). From the repository root, run the dataset target to save all six files into `test/spectron/`, where the training program expects them:

```sh
make download-datasets
```

The target creates `test/spectron/` as needed and uses `curl` with retries and resume support. The five Python shards are about 250 MB each. The training code samples from each file, so it does not need to load the full dataset into memory.

```sh
make run
```

`make run` compiles and trains with the default `auto` backend, 6,000 steps, batch 32, and sequence length 128. It writes `test/spectron/models/spectron_bpe_tokenizer.json` when no tokenizer exists, saves `test/spectron/models/finetune.nimq` every five steps, and resumes from an existing checkpoint. The checkpoint keeps model weights and step count, but Adam moments restart when resuming. On macOS, `make run-metal` compiles with the Metal backend.


### Run the trained Spectron checkpoint

After training, `make run-spectron` builds `test/spectron/test_inferencespectron.nim`, the separate interactive inference program. It loads the checkpoint and tokenizer from `test/spectron/models/`; it does not train the model:

```sh
make run-spectron
```

Type a prompt at `>>>`; enter `exit` or `quit` to stop. The three Hugging Face model repositories above are for DeepSeek/Codestral inference, not the local Spectron training demo.
