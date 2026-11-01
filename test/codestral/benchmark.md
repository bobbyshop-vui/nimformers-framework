# Codestral inference benchmark

Codestral Mamba 7B with the QLoRA/Linear++ checkpoint
`test/codestral/models/jamba_parallel_lora.bin` on the FP4-KM base
`test/codestral/models/codestral_mamba_only.nimq` (64 blocks: 60 R, 4 G, 128
LoRA adapters), weights mmap'ed, resident GPU prefill and native decode.

## Tesla T4 (Vulkan, Colab)

| Runner | Prefill | Decode |
| --- | ---: | ---: |
| formers.n, 22 tokens | 0.533 s · **41.24 tok/s** | 129 tokens, 7.525 s · **17.14 tok/s** · 58.3 ms/token |
| `infer_lora`, 17 tokens | 3.041 s · 5.59 tok/s | 14.44 tok/s · 69.3 ms/token |

formers.n compiles every pipeline in its warmup (2.2 s, not part of the prefill
time above), so its prefill is GPU work only. The `infer_lora` prefill includes
the one-time setup of a fresh process (shader prewarm, scan autotune, GEMM
parity checks: about 2.1 s).

formers.n prefill runs the Mamba2 scan as the chunked SSD/FLA pass on the
tensor cores (`prefill scan=ssd_q64`, parity max error 1.1e-2 against FP64) and
all 128 prefill GEMMs as tiled `KHR_cooperative_matrix` FP16/FP32 kernels.
Decode FP4 GEMV kernels chosen by the autotuner:

| Shape (K x N) | Kernel | Time |
| --- | --- | ---: |
| 4096x18560 | `wide4-tg64` | 1.630 ms |
| 8192x4096 | `wide4-tg128` | 0.840 ms |
| 4096x32768 (`lm_head`) | `wide4-tg128` | 8.688 ms |

## MacBook Pro (Intel Iris Plus 640)

formers.n:

| Backend | Prefill | Decode |
| --- | ---: | ---: |
| Metal | 11 tokens, 5.893 s · **1.87 tok/s** | 51 tokens, 32.227 s · **1.58 tok/s** · 631.9 ms/token |
| Vulkan (MoltenVK) | 10 tokens, 6.970 s · 1.43 tok/s | 33 tokens, 22.123 s · 1.49 tok/s · 670.4 ms/token |

`infer_lora` (11-token prompt, 60 tokens):

| Backend | Prefill, 11 tokens | Decode, 60 tokens | Total |
| --- | ---: | ---: | ---: |
| Vulkan (MoltenVK), warm caches | 6.821 s · **1.61 tok/s** | 34.471 s · **1.74 tok/s** · 574.5 ms/token | 41.292 s |
| Vulkan (MoltenVK), cold caches | 15.870 s · 0.69 tok/s | 42.008 s · 1.43 tok/s · 700.1 ms/token | 57.878 s |
| Metal | 7.910 s · 1.39 tok/s | 38.174 s · 1.57 tok/s · 636.2 ms/token | 46.083 s |

Vulkan reports `residentWeights=host-visible/shared matrix=scalar`: there is no
cooperative-matrix path on this GPU, so prefill uses the packed FP4 VM kernels
and the scan picks the register lane scan. A cold Vulkan run compiles every
shader three times (glslc to SPIR-V, the lane-scan parity check, MoltenVK
SPIR-V to MSL at the first dispatch); Metal keeps its pipelines in
`metal-jit.binarchive`.

Prefill profile of the `infer_lora` runs:

| Stage | Vulkan, cold | Vulkan, warm | Metal |
| --- | ---: | ---: | ---: |
| `resident.group` | 11242 ms (8 groups) | 6604 ms (8 groups) | 6380 ms (16 groups) |
| `vulkan.prewarm` (glslc) | 2256 ms | 22 ms | — |
| `scan.autotune` (lane parity check) | 2152 ms | 62 ms | 1202 ms |
| `enc.inproj` + `enc.outproj` + `enc.attn` (host) | 1126 ms | 331 ms | 10 ms |

Kernel times on the same GPU with synthetic data (T=11, one dispatch each):

| Kernel | Metal | Vulkan |
| --- | ---: | ---: |
| FP4 2-column m8, 4096x18560 | 22.1 ms | 19.6 ms |
| FP4 2-column m4 tail, 4096x18560 | 13.5 ms | 12.0 ms |
| FP4 2-column m8, 8192x4096 | 13.0 ms | 7.3 ms |
| FP32 batch, 4096x12288 | 27.1 ms | 17.0 ms |
| Lane scan, 128 heads x 64 x 128 | 2.9 ms | 1.4 ms |

## Test machines

Tesla T4 (Google Colab):

- `[vulkan] device=Tesla T4 type=dGPU residentWeights=device-local/VRAM
  scratch=device-local/VRAM subgroup=32 matrix=KHR_cooperative_matrix FP16/FP32
  prefill=tensor-cores-only units=40`

MacBook Pro:

- MacBook Pro 13-inch (2017), 2.3 GHz dual-core Intel Core i5
- GPU: Intel Iris Plus Graphics 640, 1536 MB
- Memory: 16 GB 2133 MHz LPDDR3, macOS Ventura 13.7.8
- MPS disabled on this Intel Mac (`Apple Silicon=false`)

## Commands

Tesla T4, formers.n cell of `notebooks/codestral_colab_vulkan.ipynb`
(prompt `Write a Python function to check if a string is a palindrome`, greedy):

```sh
BYBY_MODEL_MMAP=1 BYBY_VULKAN_DEVICE_LOCAL=auto BYBY_VULKAN_TENSOR_CORES=1 \
  other/formers.n/formers-cli --models test/codestral/models --backend vulkan \
  -p "Write a Python function to check if a string is a palindrome" -n 256 \
  --temp 0 --no-merge-lora --no-progress --verbose
```

Tesla T4, `infer_lora` cell of the same notebook (prompt
`[INST] Write a Python function for Fibonacci. [/INST]`, greedy).

MacBook, formers.n:

```sh
other/formers.n/formers-cli --models test/codestral/models --backend metal
other/formers.n/formers-cli --models test/codestral/models --backend vulkan
```

MacBook, `infer_lora` (prompt `[INST] Xin chao[/INST]`, temperature 0.7):

```sh
make infer-lora BYBY_INFER_BACKEND=vulkan
make infer-lora BYBY_INFER_BACKEND=metal
```
