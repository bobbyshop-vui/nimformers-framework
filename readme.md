# Nimformer Framework

Nimformer is a Nim framework for transformer inference and training with Metal and Vulkan GPU backends. The API references, usage guides, file formats, and BybyLang GPU programming docs are in [`docs/`](docs/README.md).

Start with the [documentation index](docs/README.md), then see [Getting Started](docs/getting-started.md) for a minimal program.

Model-specific programs and data live under `test/spectron/`, `test/codestral/`, and `test/deepseek-inference/`. Model assets live in each project's `models/` folder. The FP4 GEMV micro-kernel modules live under `kernelfortest/`; shared runtime modules remain at the project root.

## Python and C ABI

The core framework is available as a shared library with a stable C ABI
(`lib_c_api.nim`). It exposes backend selection, owned tensors, every math /
neural-network kernel, int8/int4/fp8/fp4/custom-float quantization, NIMQ
state dicts, the micro-VM bytecode builder and dispatch, resident weights,
tokenizers, and Llama inference.

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
| [DeepSeek Coder 6.7B NimQ](https://huggingface.co/bobbydeveloper2014/deepseek-coder-6.7b-nimq) | `model.nimq`, `config.json`, and tokenizer files for `test/deepseek-inference/nim_inference.nim` | `infer-deepseek` |
| [Codestral v0.1 7B NimQ](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-7b-nimq) | Codestral base weights, model config, and Tekken/HF tokenizer files | `infer-lora` base |
| [Codestral v0.1 QLoRA LinearAttention](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-nimq-qlora-linearattention) | `jamba_parallel_lora.bin` adapter checkpoint; it uses the Codestral NimQ base above | `infer-lora` adapter |

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

### Run Codestral with the QLoRA adapter

The adapter file does not contain the base Mamba weights. `infer_lora` reads the base `.nimq` from `test/codestral/models/` and uses the tokenizer directory recorded in the checkpoint sidecar. The command below writes that sidecar and runs from the project root:

```sh
make infer-lora \
  ARGS='test/codestral/models/jamba_parallel_lora.bin "[INST] Write a Python function for Fibonacci.[/INST]" 0.7 test/codestral/models/hf test/codestral/models/codestral_mamba_only.nimq'
```

`infer-lora` accepts `<checkpoint.bin> <prompt> <temperature>` for this QLoRA checkpoint. The binary metadata selects the base `.nimq`; the sidecar points the tokenizer loader at the current base-model directory. `infer-mamba` is an alias for `infer-lora`. `make infer-lora` defaults to `auto`, preserving automatic backend selection on macOS and other platforms. It checks the selected device type: dGPU weights go to device-local VRAM; iGPU weights use host-visible/shared memory. To override the backend explicitly, use `make infer-lora BYBY_INFER_BACKEND=metal` or `make infer-lora BYBY_INFER_BACKEND=vulkan`.

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
