# DeepSeek Coder 6.7B inference benchmark

Model: Q4 `model.nimq` (4488 MB, NIMQ2, mmap), greedy sampling. Prefill uses
GPU flash attention and micro-batched prompt processing.

## Tesla T4 (Vulkan, Colab)

Latest supplied run: 309 prompt tokens, 254 generated tokens.

| Prefill | Decode |
| ---: | ---: |
| **≈210 tok/s** | **≈30 tok/s** |

Prefill throughput depends on prompt length and can rise with sufficiently
large prompts as GPU batches fill, until throughput saturates. The runtime
autotunes fused versus split QKV on first decode and selects the lower median
of three samples, each containing six repeated operations. Set
`BYBY_VULKAN_Q4_SPLIT_QKV=0` to force fused QKV or `=1` to force split.

The T4 Vulkan path uses cooperative-matrix tensor-core kernels for prefill.
`BYBY_UBATCH` caps micro-batch rows (default 512); `BYBY_PREFILL_CB_MS` controls
the command-buffer time target.

## MacBook Pro (Intel Iris Plus 640)

A current Vulkan throughput result has not been recorded after the runtime Q4
QKV autotuner was added. No earlier Iris throughput or profile results are
listed here.

## Test machines

Tesla T4 (Google Colab): Vulkan discrete GPU; cooperative matrix available.

MacBook Pro 13-inch (2017): Intel Iris Plus Graphics 640 (1536 MB), macOS
13.7.8, Vulkan through MoltenVK 1.4.2.

## Commands

Tesla T4 Colab inference:

```sh
BYBY_MODEL_MMAP=1 BYBY_VULKAN_DEVICE_LOCAL=auto BYBY_VULKAN_TENSOR_CORES=1 \
  BYBY_UBATCH=512 BYBY_PREFILL_CB_MS=300 \
  ./nim_inference --no-progress "$MODELS" "$MODELS/model.nimq" vulkan \
  "Write a Python function to return the n-th Fibonacci number." 256 0.0 0.9
```

MacBook Vulkan, 106-token prompt (`$PROMPT`):

```sh
BYBY_UBATCH=512 BYBY_PREFILL_CB_MS=300 \
  other/formers.n/formers-cli --models test/deepseek-inference/models \
  --raw --add-special-tokens -p "$PROMPT" -n 12 --temp 0 \
  --backend vulkan --no-progress -c 512 --verbose
```
