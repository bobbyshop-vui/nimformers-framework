# DeepSeek Coder 6.7B inference benchmark

`nim_inference` with the Q4 checkpoint `model.nimq` (4488 MB, NIMQ2, mmap),
prompt `Write a Python function to return the n-th Fibonacci number.` rendered
by the checkpoint's chat template (85 prompt tokens), greedy sampling
(`T=0.0`).

## Tesla T4 (Vulkan, Colab)

| Prefill, 85 tokens | Decode, 255 tokens |
| ---: | ---: |
| 3.062 s · **27.76 tok/s** | 10.080 s · **25.30 tok/s** · 39.5 ms/token |

Decode GEMV kernels chosen by the autotuner (M=1, one dispatch):

| Shape (K x N) | Kernel | Time | Weight read rate* |
| --- | --- | ---: | ---: |
| 4096x4096 | `q4_sg_r4_v4_tg64` | 0.075 ms | ~112 GB/s |
| 4096x11008 | `q4_sg_r4_v4_tg64` | 0.153 ms | ~147 GB/s |
| 11008x4096 | `q4_sg_r4_v4_tg64` | 0.160 ms | ~141 GB/s |

\* Packed nibbles only (K·N/2 bytes) over the kernel time; scales and zeros
add a little more. The T4's peak is 320 GB/s.

Prefill GEMMs at M=64 run on the tensor cores (`KHR_cooperative_matrix`, FP16
tiles with FP32 accumulation): 2.83 ms at 4096x4096 and 7.7–7.8 ms for the FFN
shapes.

## MacBook Pro (Intel Iris Plus 640)

| Backend | Prefill, 85 tokens | Decode, 39 tokens |
| --- | ---: | ---: |
| Metal | 39.021 s · **2.18 tok/s** | 21.764 s · **1.79 tok/s** · 558.1 ms/token |
| Vulkan (MoltenVK) | 56.939 s · 1.49 tok/s | 22.950 s · 1.70 tok/s · 588.5 ms/token |

Peak memory footprint about 2.0 GB on both backends (`/usr/bin/time -l`).

## Test machines

Tesla T4 (Google Colab):

- `[vulkan] device=Tesla T4 type=dGPU residentWeights=device-local/VRAM
  scratch=device-local/VRAM subgroup=32 matrix=KHR_cooperative_matrix FP16/FP32`
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

MacBook (40 new tokens):

```sh
/usr/bin/time -l ./nim_inference test/deepseek-inference/models \
  test/deepseek-inference/models/model.nimq vulkan \
  "Write a Python function to return the n-th Fibonacci number." 40 0 0.9
/usr/bin/time -l ./nim_inference test/deepseek-inference/models \
  test/deepseek-inference/models/model.nimq metal \
  "Write a Python function to return the n-th Fibonacci number." 40 0 0.9
```
