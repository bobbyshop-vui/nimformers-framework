
# Nimformer Framework Documentation

[English](README.md) | [Tiếng Việt](README.vi.md)

This directory is the user documentation for the public Nimformer APIs. Use the page for the module you import; `nimformers.nim` is the combined facade for the core model, tensor, quantization, backend, session, and micro-VM APIs.


## Getting started

- [Make command guide](make-commands.md): all targets, build/run, binary outputs, backends, arguments, and assets.
- [Spectron walkthrough](spectron-guide.md): run `test_nimformer`, configure data/tokenizers/checkpoints, and read loss.
- [Python walkthrough](python-guide.md): source installation and complete API examples.
- [BybyLang walkthrough](bybylang-guide.md): compiler, DSL/AOT, sessions, micro-kernels, and the C ABI.
- [Getting Started](getting-started.md): install/build prerequisites, imports, backend selection, and a small example.
- [Python and C ABI](python-api.md): `ctypes`/NumPy bindings, the stable C header, quantization, NIMQ state dicts, micro-VM dispatch, tokenizers, and Llama generation.
- [Backend API](backend-api.md): backend creation and `be*` GPU operations.
- [Tensor Operations API](tensor-ops-api.md): tensor views, shape operations, reductions, comparisons, scans, and dropout.
- [Base Model API](base-model-api.md): tensors, layers, transformer blocks, model forward/backward, and APF Adam.
- [Hybrid Jamba Model API](jamba-model-api.md): mixed Mamba/attention model construction, batch forward, training, and checkpoints.
- [LLaMA Inference API](llama-inference-api.md): RoPE, RMSNorm, GQA, KV cache, weight loading, generation, and HF tokenization.
- [LoRA API](lora-api.md): adapters, attaching/merging, GPU preparation, and generation helpers.
- [LoRA Fine-Tuning and Resume APIs](finetune-lora-api.md): streaming batches, disk-backed optimizer state, and checkpoint recovery.
- [Training API](training-api.md): training components, datasets, optimizer/checkpoint operations, and CLI entry points.


## Data formats and precision

- [CustomFloat and APF](customfloat-apf.md): custom float encoding and adaptive precision selection.
- [Quantization and `.nimq`](quantization-nimq.md): quantization kinds, encode/decode, state dictionaries, and `.nimq` export.
- [Checkpoint and Binary Formats](checkpoint-formats.md): `.bin`, `.qlora.bin`, `.nimq`, and resumable layer progress.
- [Tokenizer APIs](tokenizer-api.md): Hugging Face tokenizer loading, encode/decode, and chat templates.


## GPU programming

- [GPU Session API](gpu-session-api.md): encode multiple operations per GPU session, resident buffers, and lifecycle rules.
- [Micro-VM / JIT API](microkernel-api.md): bytecode instructions, lowering, JIT cache, and persistent-kernel constraints.
- [BybyLang integration tests](../test/bybylang_test/README.md): BybyLang scripts that import `lib_c_api` and exercise the C ABI plus the native `gpu ...` DSL.
- [BybyLang GPU API and DSL](bybylang-api.md): GPU backend functions, DSL syntax, session blocks, imports, and native micro-kernel blocks.
- [BybyLang language reference](bybylang-language.md): the project-local bilingual edition of the vendored README, including syntax and hardware simulation.
- [Codestral Low Memory](codestral-low-memory.md): mmap model/adapter loading, bounded merge, device workgroups, and Vulkan Tensor Cores.
- [Vulkan Q4 Autotuner](vulkan-q4-autotune.md): tuning interface and environment controls for Q4 matmul.


## Reference by module

| Module | Documentation |
| --- | --- |
| `nimformers.nim` | [Getting Started](getting-started.md), [Backend](backend-api.md), [Tensor Ops](tensor-ops-api.md), [Base Model](base-model-api.md), [Quantization](quantization-nimq.md), [Micro-VM](microkernel-api.md) |
| `backend.nim` | [Backend API](backend-api.md), [GPU Session API](gpu-session-api.md) |
| `tensor_ops.nim` | [Tensor Operations API](tensor-ops-api.md) |
| `nimformer.nim` | [Base Model API](base-model-api.md) |
| `test/deepseek-inference/nim_inference.nim` | [LLaMA Inference API](llama-inference-api.md), [Tokenizer APIs](tokenizer-api.md) |
| `test/codestral/jamba_hybrid.nim` | [Hybrid Jamba Model API](jamba-model-api.md) |
| `quant.nim` / `customfloat.nim` | [Quantization](quantization-nimq.md), [CustomFloat/APF](customfloat-apf.md) |
| `test/codestral/lora.nim` / `test/codestral/infer_lora.nim` | [LoRA API](lora-api.md) |
| `test/codestral/finetune_lora.nim` | [LoRA Fine-Tuning and Resume APIs](finetune-lora-api.md) |
| `test/spectron/test_nimformer.nim` / `test/spectron/test_inferencespectron.nim` | [Training API](training-api.md) |
| `test/codestral/rawbin.nim` / `test/codestral/qlorabin.nim` / `test/codestral/checkpoint.nim` | [Checkpoint and Binary Formats](checkpoint-formats.md) |
| `vendor/bybylang/` | [BybyLang DSL](bybylang-api.md), [Language reference](bybylang-language.md), [Sessions](gpu-session-api.md), [Micro-VM/JIT](microkernel-api.md) |
| `lib_c_api.nim` / `nimformers/` / `include/nimformers.h` | [Python and C ABI](python-api.md) |

The signatures shown here are for user-facing modules and facade exports. Internal kernels and test/benchmark-only helpers are not stable application APIs. Source files remain the final authority when an API is marked experimental or backend-specific.


## Maintaining both languages

`name.md` is the English edition; `name.vi.md` is the full Vietnamese edition.
Existing English paths are retained for incoming links. Each page has a language
switch; Vietnamese pages link to Vietnamese counterparts where available.

Update both editions together when changing an API, command, or example. Keep
signatures and code blocks identical; translate explanations rather than command
or identifier names. English heading anchors are also retained in Vietnamese
pages so deep links continue to work. READMEs and API references follow the same
convention.

`vendor/bybylang` is a Git submodule. Its bilingual language manual lives in
`docs/bybylang-language*.md` so the framework repository tracks both editions;
the upstream submodule README remains the source document.
