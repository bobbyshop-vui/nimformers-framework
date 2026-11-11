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
  ./nim_inference --no-progress "$MODELS" "$MODELS/model.nimq" vulkan \
  "Write a Python function to return the n-th Fibonacci number." 256 0.0 0.9
```

MacBook, Metal through formers.n (127 new tokens):

```sh
other/formers.n/formers-cli --models test/deepseek-inference/models -cnv --temp 0 --backend metal -n 127 \
  <<< $'Write a Python function to return the n-th Fibonacci number.\n/exit'
```

MacBook, Vulkan (128 new tokens):

```sh
./nim_inference test/deepseek-inference/models \
  test/deepseek-inference/models/model.nimq vulkan \
  "Write a Python function to return the n-th Fibonacci number." 128 0 0.9
```
