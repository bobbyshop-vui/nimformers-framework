# Nimformer Framework

[English](readme.md) | [Tiếng Việt](readme.vi.md)

> **formers.n** (`other/formers.n/`) is the ready-to-use runner of this framework: one
> llama.cpp-style `formers-cli` that detects the model in a folder and runs DeepSeek/Llama,
> Codestral (Mamba-2, LoRA) or Mamba-3 SISO 187M/1.5B. Build it with `make formers-cli`;
> usage is in [Run with formers.n](#run-deepseek-or-codestral-with-formersn-formers-cli) and
> [`other/formers.n/README.md`](other/formers.n/README.md). New: **FIR**, a model-side IR
> (`--fir`) that lets a model name its own micro-kernels; see [FIR](docs/fir.md).

**Benchmarks** (prefill/decode tok/s on Tesla T4 and Intel Iris Plus 640):
[DeepSeek Coder 6.7B](test/deepseek-inference/benchmark.md) ·
[Codestral Mamba + QLoRA](test/codestral/benchmark.md) ·
[Mamba-3 SISO + LoRA](test/mamba3/benchmark.md) ·
[GPT-2 125M via FIR](models/gpt2-125m/benchmark.md) ·
[TinyLlama](models/tinyllama/benchmark.md)

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
  `formers-cli` for DeepSeek/Llama, Codestral (Mamba-2, LoRA), and Mamba-3 SISO
  187M/1.5B. Mamba-3 uses single-prompt generation; Llama and Mamba-2 also support chat.

- [FIR (formers IR)](docs/fir.md): a thin model graph file that names micro-kernels,
  backend kernels and dimensions; `--fir` runs Mamba-3 resident inference through it.

Model-specific programs and data live under `test/spectron/`, `test/codestral/`, `test/mamba3/`, and `test/deepseek-inference/`. Model assets live in each project's `models/` folder. The FP4 GEMV micro-kernel modules live under `kernelfortest/`; shared runtime modules remain at the project root.


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
- **Anthropic** ([@anthropics](https://github.com/anthropics), [@claude](https://github.com/claude)): Claude Sonnet 4.6, Claude Sonnet 5, Claude Sonnet 5.5, Claude Opus 4.6, Claude Opus 5, Claude Opus 5.5, Claude Fable 5
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

| Repository | Contents | GitHub code | Make target |
| --- | --- | --- | --- |
| [DeepSeek Coder 6.7B NimQ](https://huggingface.co/bobbydeveloper2014/deepseek-coder-6.7b-nimq) | `model.nimq`, `config.json`, and tokenizer files | [DeepSeek inference](https://github.com/bobbyshop-vui/nimformers-framework/tree/main/test/deepseek-inference) | `infer-deepseek`, `formers` |
| [Codestral v0.1 7B NimQ](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-7b-nimq) | Codestral base weights, model config, and Tekken/HF tokenizer files | [Codestral inference](https://github.com/bobbyshop-vui/nimformers-framework/tree/main/test/codestral) | `infer-lora` base, `formers` |
| [Codestral v0.1 QLoRA LinearAttention](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-nimq-qlora-linearattention) | `jamba_parallel_lora.bin` adapter; uses the Codestral NimQ base above | [Codestral LoRA](https://github.com/bobbyshop-vui/nimformers-framework/tree/main/test/codestral) | `infer-lora` adapter, `formers` |
| [Mamba-3 SISO 187M NimQ + LoRA](https://huggingface.co/bobbydeveloper2014/mamba-siso-175m-nimq) | NimQ weights, root-level config/tokenizer, and gated LoRA checkpoint | [Mamba-3](https://github.com/bobbyshop-vui/nimformers-framework/tree/main/test/mamba3) | `infer-lora`, `formers` |
| [Mamba-3 SISO 1.5B NimQ + LoRA](https://huggingface.co/bobbydeveloper2014/mamba-siso-1.5b-nimq) | NimQ weights, root-level config/tokenizer, and gated LoRA checkpoint | [Mamba-3](https://github.com/bobbyshop-vui/nimformers-framework/tree/main/test/mamba3) | `infer-lora`, `formers` |
| [forge-500m-chat](https://huggingface.co/bobbydeveloper2014/forge-500m-chat) | Weights, tokenizer, and model code (`models/forge-500m-chat`) | [forge-500m-chat](https://github.com/bobbyshop-vui/nimformers-framework/tree/main/models/forge-500m-chat) | `formers`, `download-fir-models` |
| [forge-500m-chat-fir](https://huggingface.co/bobbydeveloper2014/forge-500m-chat-fir) | `model.fir`, `config.json`, weights, and tokenizer (`models/forge-500m-chat-fir`) | [forge-500m-chat-fir](https://github.com/bobbyshop-vui/nimformers-framework/tree/main/models/forge-500m-chat-fir) | `formers`, `download-fir-models` |
| [gpt2-125m](https://huggingface.co/bobbydeveloper2014/gpt2-125m) | `model.safetensors`, `model.fir`, and tokenizer files (`models/gpt2-125m`) | [gpt2-125m](https://github.com/bobbyshop-vui/nimformers-framework/tree/main/models/gpt2-125m) | `formers`, `download-fir-models` |
| [huihui-moe-fir](https://huggingface.co/bobbydeveloper2014/huihui-moe-fir) | `model.fir`, weights, and tokenizer files (`models/huihui-moe-fir`) | [huihui-moe-fir](https://github.com/bobbyshop-vui/nimformers-framework/tree/main/models/huihui-moe-fir) | `formers`, `download-fir-models` |

`make download-models` fetches all of the above into those paths (`make download-fir-models` fetches only the FIR ones; requires the `hf` CLI from `pip install -U huggingface_hub`).

To make Hugging Face tools pick up your token from the local ignored environment file, load it in the shell before using `hf` or `huggingface_hub`:

```sh
set -a
source .env.huggingface
set +a
```


### Inference benchmarks

Prefill and decode tok/s for each model on a Colab Tesla T4 (Vulkan) and a
MacBook Pro with Intel Iris Plus 640 (Metal and Vulkan/MoltenVK), with the
test machines and the commands used:

| Model | Benchmark |
| --- | --- |
| DeepSeek Coder 6.7B (`nim_inference`) | [`test/deepseek-inference/benchmark.md`](test/deepseek-inference/benchmark.md) |
| Codestral Mamba + QLoRA (`infer_lora`) | [`test/codestral/benchmark.md`](test/codestral/benchmark.md) |
| Mamba-3 SISO 187M + LoRA (`test/mamba3/infer_lora_bin`) | [187M benchmark](test/mamba3/benchmark.md#mamba-3-siso-187m) |
| Mamba-3 SISO 1.5B + gated LoRA (`test/mamba3/infer_lora_bin`) | [1.5B benchmark](test/mamba3/benchmark.md#mamba-3-siso-15b) |
| GPT-2 125M through FIR (`formers-cli`) | [`models/gpt2-125m/benchmark.md`](models/gpt2-125m/benchmark.md) |
| TinyLlama (`formers-cli`) | [`models/tinyllama/benchmark.md`](models/tinyllama/benchmark.md) |

Latest recorded **Mamba-3 1.5B** run on Intel Iris Plus 640 / Metal: prefill
**9.88 tok/s** (18 tokens, 1.822 s, including 0.039 s kernel setup), decode
**4.89 tok/s** (64 tokens, 13.084 s), and 14.906 s combined.

### Mamba-3 SISO: separate 187M and 1.5B models

The Mamba-3 pipeline supports two separate bases through the same trainer and
inference binaries. The smaller model is **187M**; there is no 125M base in the
current local setup. Both variants use FP4-KM base weights and a learned gated
Mamba-3 + Linear++ branch (`G`) at the last layer.

- **187M:** 12 layers, hidden size 768. Config/tokenizer directory:
  `test/mamba3/models/mamba3-siso-187m/`; base: `test/mamba3/models/mamba3_siso_187m.nimq`.
  LoRA checkpoint: `test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin`; scratch:
  `test/mamba3/models/runs/mamba3-siso-187m-gated/scratch/`. Pattern: 11 `R` blocks + one `G`.
- **1.5B:** 24 layers, hidden size 2048. Config/tokenizer directory:
  `test/mamba3/models/mamba3-siso-1.5b/`; base:
  `test/mamba3/models/mamba3_siso_1.5b.nimq`. LoRA checkpoint:
  `test/mamba3/models/runs/mamba3-siso-1.5b-gated/lora-g1.bin`; scratch:
  `test/mamba3/models/runs/mamba3-siso-1.5b-gated/scratch-g1/`. Pattern: 23 `R` blocks + one `G`.

Each checkpoint records its own base path and topology. Keep checkpoints and
resume state separate between variants. Build the shared inference binary and
choose the checkpoint explicitly:

```sh
make -f test/mamba3/Makefile infer

# 187M
BYBY_METAL_JIT_LOG=0 BYBY_ONE_DISPATCH=0 BYBY_MAX_NEW_TOKENS=64 \
test/mamba3/infer_lora_bin test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
  'Write a Python function to check if a string is a palindrome' 0.7

# 1.5B
BYBY_METAL_JIT_LOG=0 BYBY_ONE_DISPATCH=0 BYBY_MAX_NEW_TOKENS=64 \
test/mamba3/infer_lora_bin test/mamba3/models/runs/mamba3-siso-1.5b-gated/lora-g1.bin \
  'Write a Python function to check if a string is a palindrome' 0.7
```

The final argument is temperature; `BYBY_MAX_NEW_TOKENS` sets the output limit.
See [the Mamba-3 guide](test/mamba3/README.md) for separate finetune/resume commands
and setup. Root `make infer-lora` / `make finetune-mamba` select the Codestral
pipeline; the commands above select Mamba-3.

#### Run either Mamba-3 model through formers.n

`formers-cli` routes both variants to the same `test/mamba3` loader, generation
loop and optimized GPU runtime. Build it, then select the base, config,
tokenizer and adapter explicitly because `models/` contains both variants:

```sh
make build-formers

# 187M: 11 R + G
BYBY_METAL_JIT_LOG=0 BYBY_ONE_DISPATCH=0 other/formers.n/formers-cli \
  --models test/mamba3/models \
  -m test/mamba3/models/mamba3_siso_187m.nimq \
  --config test/mamba3/models/mamba3-siso-187m/config.json \
  --tokenizer test/mamba3/models/mamba3-siso-187m \
  --lora test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
  -p 'Write a Python function to check if a string is a palindrome' \
  -n 64 --temp 0.7 --backend metal

# 1.5B: 23 R + G
BYBY_METAL_JIT_LOG=0 BYBY_ONE_DISPATCH=0 other/formers.n/formers-cli \
  --models test/mamba3/models \
  -m test/mamba3/models/mamba3_siso_1.5b.nimq \
  --config test/mamba3/models/mamba3-siso-1.5b/config.json \
  --tokenizer test/mamba3/models/mamba3-siso-1.5b \
  --lora test/mamba3/models/runs/mamba3-siso-1.5b-gated/lora-g1.bin \
  -p 'Write a Python function to check if a string is a palindrome' \
  -n 64 --temp 0.7 --backend metal
```

Add `--detect-only` to inspect the selected architecture and checkpoint without
loading weights onto the GPU. Replace `--lora <path>` with `--no-lora` for the
base model. Add `--verbose` for Mamba-3 prefill/decode statistics, retaining
`BYBY_METAL_JIT_LOG=0` to hide routine Metal JIT logs. Use `--backend vulkan` on
a Vulkan setup. Add `-cnv` for chat; the prompt comes from the model folder's chat template.

### Download the models

All five Hugging Face repositories above are public. From the repository root, run this target to download their weights, adapters, configs, and tokenizers into the paths used by inference and formers.n:

```sh
make download-models
```

The target creates the model directories under `test/` as needed and uses `curl` with retries and resume support. It also fetches each Mamba-3 config/tokenizer into its own directory so the 187M and 1.5B configs do not overwrite each other, then installs the LoRA tokenizer sidecars at the paths recorded by their checkpoints.

| Hugging Face repository | Downloaded files |
| --- | --- |
| [DeepSeek Coder 6.7B NimQ](https://huggingface.co/bobbydeveloper2014/deepseek-coder-6.7b-nimq) | `test/deepseek-inference/models/` |
| [Codestral v0.1 7B NimQ](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-7b-nimq) | `test/codestral/models/` and `test/codestral/models/hf/` |
| [Codestral QLoRA LinearAttention](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-nimq-qlora-linearattention) | `test/codestral/models/jamba_parallel_lora.bin` and its tokenizer sidecar |
| [Mamba-3 SISO 187M NimQ + LoRA](https://huggingface.co/bobbydeveloper2014/mamba-siso-175m-nimq) | base `test/mamba3/models/`; config/tokenizer `test/mamba3/models/mamba3-siso-187m/`; adapter `test/mamba3/models/runs/mamba3-siso-187m-gated/` |
| [Mamba-3 SISO 1.5B NimQ + LoRA](https://huggingface.co/bobbydeveloper2014/mamba-siso-1.5b-nimq) | base `test/mamba3/models/`; config/tokenizer `test/mamba3/models/mamba3-siso-1.5b/`; adapter `test/mamba3/models/runs/mamba3-siso-1.5b-gated/` |
| [forge-500m-chat](https://huggingface.co/bobbydeveloper2014/forge-500m-chat) | `models/forge-500m-chat/` |
| [forge-500m-chat-fir](https://huggingface.co/bobbydeveloper2014/forge-500m-chat-fir) | `models/forge-500m-chat-fir/` |
| [gpt2-125m](https://huggingface.co/bobbydeveloper2014/gpt2-125m) | `models/gpt2-125m/` |
| [huihui-moe-fir](https://huggingface.co/bobbydeveloper2014/huihui-moe-fir) | `models/huihui-moe-fir/` |

The last four are FIR models. Each repo is the full folder (weights, tokenizer, `model.fir`) and is fetched with the `hf` CLI (`pip install -U huggingface_hub`); `make download-fir-models` downloads only these.

All assets are public, so downloading them does not require `HF_TOKEN` or a Hugging Face login. The `HF_TOKEN` environment setup above is only needed for private or gated repositories and write operations.

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
  ARGS='test/codestral/models/jamba_parallel_lora.bin "Write a Python function to check if a string is a palindrome" 0.7 test/codestral/models/hf test/codestral/models/codestral_mamba_only.nimq'
```

The prompt is wrapped in the Mistral v3 instruct template by default:
`Write a Python function to check if a string is a palindrome` is sent as
`[INST] Write a Python function to check if a string is a palindrome[/INST]`
(the format the adapter was trained on). A prompt that already starts with
`[INST]` is sent unchanged; `BYBY_RAW_PROMPT=1` turns the wrapping off. Without
a prompt argument, that palindrome prompt is the default.

`infer-lora` accepts `<checkpoint.bin> <prompt> <temperature> [hfDir] [nimqPath]`. The binary metadata selects the base `.nimq`; the sidecar points the tokenizer loader at the current base-model directory. The last two CLI arguments configure the base-Mamba fallback path. `infer-mamba` is an alias for `infer-lora`. `make infer-lora` defaults to `auto`, preserving automatic backend selection on macOS and other platforms. It checks the selected device type: dGPU weights go to device-local VRAM; iGPU weights use host-visible/shared memory. To override the backend explicitly, use `make infer-lora BYBY_INFER_BACKEND=metal` or `make infer-lora BYBY_INFER_BACKEND=vulkan`.

Speeds: [`test/codestral/benchmark.md`](test/codestral/benchmark.md).

#### Inference flags (`infer_lora`)

Every flag is an environment variable, set on the `make infer-lora` line or
before the binary (`BYBY_INFER_BACKEND=vulkan ./test/codestral/infer_lora_bin ...`).
An explicit value always wins over the default. Background on the prefill
routes: [`test/codestral/INFERENCE.md`](test/codestral/INFERENCE.md).

Prompt, sampling and output:

| Flag | Default | Meaning |
| --- | --- | --- |
| `BYBY_RAW_PROMPT` | `0` | `1` sends the prompt as is, without the `[INST] ...[/INST]` template |
| `BYBY_KEEP_TRAILING_WS` | `0` | `1` keeps trailing whitespace of the prompt (otherwise stripped: it would add a lone `▁` token) |
| `BYBY_MAX_NEW_TOKENS` | `60` | tokens to generate |
| `BYBY_TOP_P` | `0.9` | nucleus sampling |
| `BYBY_MIN_P` | `0.05` | min-p sampling |
| `BYBY_REPEAT_PENALTY` | `1.1` | repetition penalty |
| `BYBY_REPEAT_WINDOW` | `64` | tokens the repetition penalty looks back over |
| `BYBY_ALLOW_SPECIAL` | `0` | `1` lets the sampler pick control/special tokens |
| `BYBY_PROGRESS` | `auto` | `0` turns off the live tok/s line (on only when stdout/stderr are a TTY) |
| `BYBY_MACOS_HIGH_PRIORITY` | `1` | macOS: run the inference thread at user-interactive QoS; `0` keeps the inherited QoS |

Backend and memory:

| Flag | Default | Meaning |
| --- | --- | --- |
| `BYBY_INFER_BACKEND` | `auto` | `auto`, `metal`, `vulkan` |
| `BYBY_MODEL_MMAP` | `1` | map the `.nimq` instead of reading it into RAM (needed by the in-place merge) |
| `BYBY_MERGE_LORA` | `inplace` | merge LoRA into the FP4 weights on the GPU; `1` = `inplace`, `0` keeps separate adapters |
| `BYBY_MERGE_LORA_TRY_FP4` | `0` | `1` lets the guarded merge try requantizing into 4-bit (kept only within 0.1% RMS delta error) |
| `BYBY_MERGE_SCRATCH_MIB` | `64` | GPU scratch of the in-place merge (at most 64) |
| `BYBY_MEMORY_LIMIT_MB` | `7000` | process RSS guard of the in-place merge |
| `BYBY_KEEP_ZERO_LORA` | `0` | `1` keeps all-zero adapters (dropped by default: exact zero delta) |
| `BYBY_APPLY_LMHEAD_LORA` | off | `1` applies a LoRA adapter on `lm_head` |
| `BYBY_ONE_DISPATCH` | `0` | `1` runs decode as one persistent mega-kernel (Metal experiment) |
| `BYBY_OD_TG` | `16` | threadgroups of the one-dispatch kernel (must all be resident) |
| `BYBY_MVM_MAX_SCRATCH_MB` | auto | Metal: cap on one micro-VM scratch buffer |
| `BYBY_VM_GROUPS` | `1` | `0` uploads/reads each micro-VM group separately (old path) |
| `BYBY_RMS_TG` | `256` | threadgroup size of the RMSNorm kernels |
| `BYBY_METAL_JIT_ARCHIVE` | `~/Library/Caches/nimformers/metal-jit.binarchive` | Metal pipeline binary archive |
| `BYBY_VULKAN_DEVICE` | best GPU (dGPU, NVIDIA first) | Vulkan device index |
| `BYBY_VULKAN_DEVICE_LOCAL` | `auto` | weights in VRAM on a dGPU (`auto`/`1`), `0` keeps them host-visible; an iGPU always uses shared memory |
| `BYBY_VULKAN_SCRATCH` | `auto` | scratch buffers: `auto` (VRAM on a dGPU), `device`, `host` |
| `BYBY_VULKAN_MMAP_IMPORT` | `1` | iGPU: import the mmap'ed weights zero-copy (`VK_EXT_external_memory_host`) |
| `BYBY_VULKAN_POOL_MB` | `2048` | cap on the pooled Vulkan buffers |
| `BYBY_VULKAN_TENSOR_CORES` | `1` | `0` disables `KHR_cooperative_matrix` (FP16/FP8 tensor cores) |
| `BYBY_VULKAN_SPLIT_K` | `auto` | split-K slices of the tensor-core GEMM (`1` = off) |
| `BYBY_VULKAN_MAX_TG` | device max (≤1024) | cap on Vulkan workgroup size |
| `BYBY_VULKAN_JIT_DIR` | temp dir | SPIR-V cache directory (point it at persistent storage) |
| `BYBY_VULKAN_DISPATCH_CACHE` | `1` | `0` recompiles shaders on every dispatch (diagnostics) |
| `BYBY_VULKAN_STRICT` | `0` | `1` fails on an out-of-range buffer offset instead of clamping |

Prefill:

| Flag | Default | Meaning |
| --- | --- | --- |
| `BYBY_PREFILL_RESIDENT` | `1` | the GPU-resident prefill (all layers on the GPU, states carried GPU-to-GPU) |
| `BYBY_PREFILL_PARALLEL` | `1` | `0` disables the sequence-parallel prefill (token by token) |
| `BYBY_PREFILL_PAR_MIN_T` | `2` (resident), `32` | shortest prompt that takes the parallel prefill |
| `BYBY_PREFILL_BATCH_SIZE` | `128` | prompt tokens per prefill chunk (`2..128`) |
| `BYBY_PREFILL_LAYERS_PER_COMMIT` | `8` Vulkan, `4` Metal | layers per command buffer |
| `BYBY_SSD` | `auto` | Mamba2 scan as the chunked SSD/FLA pass: tensor cores, else FP32 tiles; `scalar` skips tensor cores, `0` uses the scan kernels (Vulkan) |
| `BYBY_SSD_CHUNK` | `64` | SSD chunk length (power of two, 16..256) |
| `BYBY_SSD_MIN_T` | `0` | prompts shorter than this use the scan instead of SSD |
| `BYBY_SSD_STATE_MB` | `1024` | cap on the SSD chunk-state scratch; the chunk doubles until it fits |
| `BYBY_PREFILL_SCAN` | `auto` | scan route when SSD is off: `auto` (benchmarked), `serial`, `parallel`, `lanes`, `coalesced` |
| `BYBY_PREFILL_SCAN_LANES` | `1` | `0` drops the register lane scan from the `auto` candidates |
| `BYBY_PREFILL_SCAN_AUTOTUNE_MAX_MIB` | `512` | largest probe arena the scan autotuner allocates |
| `BYBY_MAMBA_SCAN_PARALLEL` | `1` Metal, `0` Vulkan | parallel scan in the non-resident path |
| `BYBY_LP_SCAN_SERIAL` | off | `1` forces the serial Linear++ scan |
| `BYBY_PREFILL_LP_LANES` | `1` | `0` disables the Linear++ lane scan |
| `BYBY_PREFILL_GEMM` | `auto` | `auto`, `m8` (packed FP4 tiles), `mps` (expanded FP32 + MPS, Apple Silicon) |
| `BYBY_PREFILL_MPS` | `1` | `0` never considers MPS |
| `BYBY_PREFILL_FP4_KERNEL` | `auto` | `m4` forces the 4-row FP4 tile |
| `BYBY_PREFILL_2COL` | `1` | `0` selects the one-column FP4 tile |
| `BYBY_PREFILL_TAIL4` | `1` | `0` pads the last rows to 8 instead of a 4-row tail tile |
| `BYBY_PREFILL_UNROLL` | `2` | `1` loads one weight word per iteration in the FP4 tile |
| `BYBY_FP4_PAD_M` | `1` | `0` turns off padding odd row counts to the FP4 tile |
| `BYBY_PREFILL_FP4_TILED`, `BYBY_PREFILL_FP32_TILED` | `1` | `0` skips the tiled tensor-core GEMM (parity-checked first) |
| `BYBY_PREFILL_PREWARM_PACKED` | `0` | `1` also precompiles the packed FP4 tiles when tensor cores are used |
| `BYBY_PREFILL_LORA_DOWN` | `1` | `0` disables the fused LoRA down-projection kernel |
| `BYBY_PREFILL_SPARSE` | `1` | `0` disables the sparse prefill in the hybrid block |
| `BYBY_PREFILL_KEEP_SCRATCH` | `0` | `1` keeps the prefill arena after prefill (throughput experiments) |
| `BYBY_SCAN_NCHUNK` | `16` | command buffers the backward scan is split into (watchdog) |

Decode and tuning:

| Flag | Default | Meaning |
| --- | --- | --- |
| `BYBY_GEMV_TG` | `auto` | GEMV threadgroup width (`auto` = tuned, `max`, or a number) |
| `BYBY_GEMV_VARIANT` | `auto` | `0` 2 rows, `1` 32-bit loads, `2` 4 rows per threadgroup |
| `BYBY_GPU_AUTOTUNE` | `1` | `0` skips the FP4 kernel tuner (faster cold start) |
| `BYBY_NORM_PRESET` | from checkpoint | gated-RMSNorm preset (default `rms-fin-gn`) |
| `BYBY_NORM_AUTO` | `1` | `0` ignores the preset suggested by the norm sweep |
| `BYBY_MAMBA_NORM_GROUPS` | model `n_groups` | gated-RMSNorm group count (`1` = legacy full width) |

Diagnostics (slower, not fast paths):

| Flag | Default | Meaning |
| --- | --- | --- |
| `BYBY_PREFILL_PROFILE` | `1` | per-stage prefill timings |
| `BYBY_PREFILL_DIAG` | `0` | `1` commits after every prefill stage (real GPU time per stage), `2`/`3` finer |
| `BYBY_DECODE_PROFILE` | `0` | per-token decode timings |
| `BYBY_DECODE_SPLIT` | `0` | splits each token into several command buffers to time each part |
| `BYBY_DEBUG_SAMPLING` | `0` | logs the sampler state; `BYBY_DEBUG_SAMPLING_ALL=1` every step, `BYBY_DEBUG_SAMPLING_STEPS` (5) the first N |
| `BYBY_DEBUG_TOP5` | `0` | logs the top-5 logits |
| `BYBY_TRACE_TOKENS` | `0` | traces the first N generated tokens |
| `BYBY_TRACE_BLOCKS` | `0` | traces per-block activations |
| `BYBY_DUMP_LOGITS` | off | file path to dump the logits |
| `BYBY_G_DIAGNOSTICS` | `0` | statistics of the G (gated) blocks |
| `BYBY_OD_CHECK`, `BYBY_OD_VERBOSE` | `0` | one-dispatch error check and logs |
| `BYBY_LPCHECK` | `0` | checks the Linear++ step kernel against its closed form |
| `BYBY_NORM_SWEEP` | `0` | bisects norm presets (`BYBY_NORM_SWEEP_LIST`, `BYBY_NORM_SWEEP_TOKENS` = 12) |
| `BYBY_INFER_VERBOSE` | off | extra inference logs |
| `BYBY_LORA_VERBOSE` | `0` | logs adapters missing on the GPU |
| `BYBY_JIT_DUMP_SOURCE` | `0` | prints the generated kernel source |
| `BYBY_VULKAN_JIT_VERBOSE` | `0` | logs each Vulkan shader compile |
| `BYBY_VULKAN_SPLIT_K_VERBOSE` | `0` | logs the split-K choice |


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
| Mamba-3 SISO 187M or 1.5B, base or LoRA / gated Linear++ | `test/mamba3` shared inference runtime | `test/mamba3/models` with explicit model/config/tokenizer flags |

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
| LoRA (Mamba-2/3) | the only `.bin` with a `NIMBIN1`/`NIMBIN4` header; several is an error | `--lora FILE`, `--no-lora` |

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

A Llama/DeepSeek prompt prefills every layer on the GPU in one chain of dispatches
(RMSNorm, QKV, RoPE, KV-cache append, **flash attention**, oProj, FFN), reading back only
the last row. Flash attention streams the cached keys through on-chip memory with an
online softmax and reads the KV cache, so later chat turns prefill too instead of
decoding token by token. It runs on the Vulkan tensor cores when the GPU has them
(KHR_cooperative_matrix, with FlashAttention-4's lazy rescale of the output), otherwise
on a scalar FP32 kernel on Vulkan and Metal; `--verbose` shows
`[prefill] flash attention (coop|scalar)`. Each kernel is checked against a reference at
load time and skipped if it fails. `BYBY_FLASH_ATTENTION=scalar` skips the tensor-core
kernel, `BYBY_FLASH_PREFILL=0` goes back to the old prefill, and `make flash-attention-test`
checks the kernels on Metal and Vulkan.

On Metal, prefill GEMMs (M >= 8) run a tiled Q4 kernel (64 x 128 output tile per
threadgroup, weights decoded once per 64 rows, 8 x 8 FP32 register tile), about 5x the
previous batched kernel on Intel Iris; `BYBY_METAL_Q4_GEMM=0` restores the old one and
`make q4-gemm-test` checks and times it. On Intel/AMD Macs and Vulkan, prefill and decode
work is split into command buffers of about `BYBY_PREFILL_CB_MS` (default 300) from the
measured speed, so the macOS GPU watchdog never kills a long one; Apple Silicon keeps its
command buffers whole.

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

To check, compile, and run `test_nimformer` with Vulkan explicitly:

```sh
make run-vulkan
```

This uses `-d:backend=vulkan` and starts the Spectron trainer with its configured
datasets and training settings. It requires a Vulkan loader/driver (MoltenVK on
macOS). To preview the commands without starting training, use `make -n run-vulkan`.


### Run the trained Spectron checkpoint

After training, `make run-spectron` builds `test/spectron/test_inferencespectron.nim`, the separate interactive inference program. It loads the checkpoint and tokenizer from `test/spectron/models/`; it does not train the model:

```sh
make run-spectron
```

Type a prompt at `>>>`; enter `exit` or `quit` to stop. The three Hugging Face model repositories above are for DeepSeek/Codestral inference, not the local Spectron training demo.
