# DeepSeek Coder 6.7B inference benchmark

`nim_inference` with the Q4 checkpoint `model.nimq` (4488 MB, NIMQ2, mmap),
prompt `Write a Python function to return the n-th Fibonacci number.` rendered
by the checkpoint's chat template, greedy sampling (`T=0.0`). Prefill is the
flash prefill: every layer of the prompt on the GPU, attention included.

## Tesla T4 (Vulkan, Colab)

| Prefill, 86 tokens | Decode, 255 tokens |
| ---: | ---: |
| 1.011 s · **85.03 tok/s** | 10.278 s · **24.81 tok/s** · 40.3 ms/token |

Prefill attention runs the tensor-core flash kernel (`KHR_cooperative_matrix`
FP16/FP32, lazy rescale), checked against a reference when it loads.

Decode GEMV kernels chosen by the autotuner (M=1, one dispatch):

| Shape (K x N) | Kernel | Time | Weight read rate* |
| --- | --- | ---: | ---: |
| 4096x4096 | `q4_sg_r4_v1_tg64_s32_shm` | 0.078 ms | ~108 GB/s |
| 4096x11008 | `q4_sg_r4_v4_tg64_s32_shm` | 0.154 ms | ~146 GB/s |
| 11008x4096 | `q4_sg_r4_v4_tg64_s32_shm` | 0.162 ms | ~139 GB/s |

\* Packed nibbles only (K·N/2 bytes) over the kernel time; scales and zeros
add a little more. The T4's peak is 320 GB/s.

Prefill GEMMs at M=64 run on the tensor cores (tiled, split-K over the 40 SMs):

| Shape (K x N) | Workgroups | Time |
| --- | --- | ---: |
| 4096x4096 | 64 x split-K 4 | 0.756 ms |
| 4096x11008 | 172 x split-K 1 | 2.154 ms |
| 11008x4096 | 64 x split-K 4 | 2.077 ms |

## MacBook Pro (Intel Iris Plus 640)

| Backend | Prefill | Decode, 127 tokens |
| --- | ---: | ---: |
| Metal (86 tokens) | 10.157 s · **8.47 tok/s** | 42.273 s · **3.00 tok/s** · 332.9 ms/token |
| Vulkan (MoltenVK, 80 tokens) | 21.873 s · 3.66 tok/s | 92.290 s · 1.38 tok/s · 726.7 ms/token |

Metal prefill uses the tiled Q4 GEMM (64x128 tiles) and the flash attention
kernel. Long GPU work is split into command buffers of about 300 ms so the
macOS GPU watchdog does not kill it.

### Warm, same-prompt native re-test (2026-11-14)

The sandbox did not expose the native GPU/thermal state reliably, so these
measurements were repeated with native GPU access. `pmset -g therm` reported
`CPU_Speed_Limit = 100` immediately before each measured leg. One Vulkan run
with the exact prompt was used to warm JIT before measuring; each pair was run
Metal then Vulkan. Prompt: eight copies of `The quick brown fox jumps over the
lazy dog.` followed by `The quick brown fox jumps over the`, raw input with
BOS, greedy, `-c 512`, `-n 12` (106 prompt tokens, 11 decode tokens measured).

| Pair | Backend | Prefill, 106 tokens | Decode, 11 tokens |
| --- | --- | ---: | ---: |
| 1 | Metal | 9.74 tok/s | 3.16 tok/s · 316.3 ms/token |
| 1 | Vulkan | 8.14 tok/s | 2.19 tok/s · 457.7 ms/token |
| 2 | Metal | 10.94 tok/s | 2.93 tok/s · 341.4 ms/token |
| 2 | Vulkan | 8.08 tok/s | 1.95 tok/s · 514.0 ms/token |

Across these two paired runs Vulkan reached 74–84% of Metal prefill throughput
and 67–69% of Metal decode throughput. The visible greedy continuation matched
between backends. The earlier 0.58 tok/s Vulkan run ended with
`CPU_Speed_Limit = 30`; the older 1.38 tok/s entry above used an 80-token
prompt. Neither is a controlled comparison with these 106-token, warm-cache
runs.

### Vulkan decode stage profile

One decode step was profiled on all 32 layers with
`BYBY_DECODE_PROFILE=-2`, `BYBY_VULKAN_PROFILE=1`, and
`BYBY_VULKAN_TIMESTAMPS=1`. The operation profiler intentionally split the
step into 321 command-buffer sessions; use these device timings for stage
attribution, not as the normal decode throughput. The measured GPU timestamp
was 365.30 ms/token, and the labeled stages summed to 361.83 ms/token.

| Stage | GPU time per decode step |
| --- | ---: |
| RMSNorm + fused Q/K/V | 88.66 ms |
| RoPE + Q scale + KV append | 0.61 ms |
| Attention | 7.84 ms |
| oProj | 27.11 ms |
| FFN RMSNorm | 1.40 ms |
| Gate + up / SwiGLU | 139.44 ms |
| Down projection + residual | 96.78 ms |
| **Labeled stages** | **361.83 ms** |

The unlabelled 3.47 ms to the full GPU timestamp is the final-step tail outside
those layer marks (including logits/head work); it was not separately timed in
this profile. Vulkan recorded 418 dispatches, 418 barriers, and zero decode
copies. The all-layer trace therefore puts most device time in the Q4
projection/FFN stages, not RoPE, attention, or KV copies. Barriers were counted
but not timed separately. A separate one-session driver profile measured
417.30 ms GPU time, 3.74 ms queue-submit, and 427.15 ms fence-wait per token;
these overlap and are from a different run. The final unprofiled Vulkan pair
measured 514.0 ms/token end-to-end, with host encode 27.79 ms, GPU wait
473.80 ms, and other host work 12.37 ms.

### Vulkan Q4 startup cache

The Vulkan tuner now persists its selected Q4 plans by device, driver, shader
source, and tuning settings, and the Vulkan JIT/pipeline cache is persistent.
Warm runs reported cache hits for the Q4 shapes and the M=1 `lm_head` plan.
Only M=1 is tuned for Vulkan `lm_head`; this inference route uses one row for
logits, so its M=64 plan was unused. On the re-tests, converting
`lm_head` fp32 to grouped Q4 took 0.99–1.21 s, compared with 25.14 s in the
earlier cold run. The old ten-minute startup was not captured with a complete
stage-by-stage startup trace, so its full time cannot be assigned to one cause.

### Vulkan `BYBY_UBATCH` and Tesla T4

The Vulkan path is paced for every Vulkan device, including discrete GPUs.
Prefill first selects rows from the `BYBY_PREFILL_CB_MS` time target, then caps
that chunk with `BYBY_UBATCH`; the default cap is 512 rows. Thus the T4 uses
the same ubatch scheduler. The T4 also selects cooperative-matrix/tensor-core
prefill kernels when the device exposes `KHR_cooperative_matrix` and a
candidate passes parity. The Iris Plus re-test cannot validate T4 hardware;
the existing Colab T4 measurements above are the device evidence.

### Lifted FIR graph (Metal)

The graph lifted from the native run
(`models/deepseek-coder-6.7b/lifted/model_intel.fir`, 512-token context, the
224 Q4 matrices resident outside the arena) against the native runtime, same
106-token prompt, raw prompt with BOS, greedy, context 512. Both print the same
text.

| Runtime | Prefill, 106 tokens | Decode, 11 tokens |
| --- | ---: | ---: |
| FIR graph | 15.956 s · **6.64 tok/s** | 6.449 s · **1.71 tok/s** · 586.3 ms/token |
| Native | 20.369 s · 5.20 tok/s | 7.678 s · 1.43 tok/s · 698.0 ms/token |

## Test machines

Tesla T4 (Google Colab):

- `[vulkan] device=Tesla T4 type=dGPU residentWeights=device-local/VRAM
  scratch=device-local/VRAM subgroup=32 matrix=KHR_cooperative_matrix FP16/FP32
  prefill=tensor-cores-only units=40`
- 225/225 Q4 weights (including `lm_head`) copied from the mmap into VRAM once

MacBook Pro:

- MacBook Pro 13-inch (2017), 2.3 GHz dual-core Intel Core i5
- GPU: Intel Iris Plus Graphics 640, 1536 MB
- Memory: 16 GB 2133 MHz LPDDR3, macOS Ventura 13.7.8
- Vulkan through MoltenVK 1.4.2: `type=iGPU residentWeights=host-visible/shared
  matrix=scalar`; 225/225 weights read in place from the mmap (zero-copy)

## Commands

Tesla T4, cell 7 of `notebooks/deepseek_colab_vulkan.ipynb` (256 new tokens):

```sh
BYBY_MODEL_MMAP=1 BYBY_VULKAN_DEVICE_LOCAL=auto BYBY_VULKAN_TENSOR_CORES=1 \
  BYBY_UBATCH=512 BYBY_PREFILL_CB_MS=300 \
  ./nim_inference --no-progress "$MODELS" "$MODELS/model.nimq" vulkan \
  "Write a Python function to return the n-th Fibonacci number." 256 0.0 0.9
```

The explicit `BYBY_UBATCH=512` keeps the prefill micro-batch cap enabled in
this T4 command; `BYBY_PREFILL_CB_MS=300` also keeps the time-based command
buffer target enabled.

MacBook, Metal through formers.n (127 new tokens):

```sh
other/formers.n/formers-cli --models test/deepseek-inference/models -cnv --temp 0 --backend metal -n 127 \
  <<< $'Write a Python function to return the n-th Fibonacci number.\n/exit'
```

MacBook, lifted FIR graph and native on the same prompt (`$PROMPT` is "The
quick brown fox jumps over the lazy dog." eight times, then "The quick brown
fox jumps over the"); drop `--fir …` for the native run:

```sh
other/formers.n/formers-cli --models test/deepseek-inference/models \
  --fir models/deepseek-coder-6.7b/lifted/model_intel.fir \
  --raw --add-special-tokens -p "$PROMPT" -n 12 --temp 0 --backend metal --no-progress -c 512
```

MacBook, Vulkan (128 new tokens):

```sh
./nim_inference test/deepseek-inference/models \
  test/deepseek-inference/models/model.nimq vulkan \
  "Write a Python function to return the n-th Fibonacci number." 128 0 0.9
```

MacBook, exact 106-token native re-test prompt (warm Vulkan once with `-n 2`,
then measure with `-n 12`; check `pmset -g therm` before every measured run):

```sh
PROMPT='The quick brown fox jumps over the lazy dog.'
for i in 1 2 3 4 5 6 7; do PROMPT="$PROMPT The quick brown fox jumps over the lazy dog."; done
PROMPT="$PROMPT The quick brown fox jumps over the"
BYBY_UBATCH=512 BYBY_PREFILL_CB_MS=300 \
  other/formers.n/formers-cli --models test/deepseek-inference/models \
  --raw --add-special-tokens -p "$PROMPT" -n 12 --temp 0 \
  --backend vulkan --no-progress -c 512 --verbose
```
