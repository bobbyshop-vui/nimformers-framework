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

Latest paired native benchmark: 106 prompt tokens, 11 generated tokens,
greedy sampling, warm GPU; `CPU_Speed_Limit = 100` was reported before each run.

| Pair | Backend | Prefill | Decode |
| --- | --- | ---: | ---: |
| 1 | Metal | 9.74 tok/s | 3.16 tok/s · 316.3 ms/token |
| 1 | Vulkan (MoltenVK) | **8.14 tok/s** | **2.19 tok/s · 457.7 ms/token** |
| 2 | Metal | 10.94 tok/s | 2.93 tok/s · 341.4 ms/token |
| 2 | Vulkan (MoltenVK) | **8.08 tok/s** | **2.95 tok/s · 339.0 ms/token** |

The greedy continuation matched between the two backends.

## Lifted FIR graph (Tesla T4, Vulkan)

Lifted `model.fir` with FIR micro-kernels (`BYBY_FIR_Q4_BACKEND=micro`): 241
prompt tokens, 150 generated tokens, greedy sampling, context 512. Speeds exclude
kernel setup/JIT (the warm-up runs the prompt and two decode steps first).

| Runtime | Prefill | Decode |
| --- | ---: | ---: |
| FIR graph | **235.65 tok/s** | **34.51 tok/s · 29.0 ms/token** |
| Native | ≈210 tok/s | ≈30 tok/s |

Each decode step replays one command buffer of 419 dispatches, recorded at the
first step: host encode 0.00 ms, GPU wait 28.96 ms, logits read 0.02 ms per
step. `BYBY_FIR_DECODE_REPLAY=0` encodes every step instead. Q4 GEMV rows per
subgroup and workgroup width are autotuned per shape at load
(`BYBY_FIR_Q4_GEMV_TUNE=0` disables it).

## Lifted FIR graph (Metal)

Same 106-token prompt and 11-token decode as the native paired benchmark:

| Runtime | Prefill | Decode |
| --- | ---: | ---: |
| FIR graph | 20.369 s · 9.20 tok/s | 6.449 s · 3.71 tok/s |
| Native | 20.369 s · 9.20 tok/s | 7.678 s · 2.43 tok/s |

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

Tesla T4 Colab lifted FIR graph, 241-token prompt (`$PROMPT`):

```sh
BYBY_UBATCH=512 BYBY_PREFILL_CB_MS=450 BYBY_VULKAN_DEVICE_LOCAL=auto \
  BYBY_FIR_Q4_BACKEND=micro \
  other/formers.n/formers-cli --models "$MODELS" -m "$MODELS/model.nimq" \
  --fir /content/deepseek-t4-lifted/model.fir \
  --raw --add-special-tokens -p "$PROMPT" -n 150 --temp 0 \
  --backend vulkan --no-progress -c 512 --verbose
```

MacBook Vulkan, 106-token prompt (`$PROMPT`):

```sh
BYBY_UBATCH=512 BYBY_PREFILL_CB_MS=300 \
  other/formers.n/formers-cli --models test/deepseek-inference/models \
  --raw --add-special-tokens -p "$PROMPT" -n 12 --temp 0 \
  --backend vulkan --no-progress -c 512 --verbose
```
