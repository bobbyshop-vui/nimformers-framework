# GPU inference defaults

`make infer-lora` builds and runs the current inference code. Resident R/P/G
prefill, independent decode kernels in one session, FP4 autotuning and macOS
USER_INITIATED thread QoS are enabled by default. One-dispatch remains opt-in.

`nim_inference` defaults to automatic backend selection, resident weights/KV,
merged QKV/attention and one session across decode layers. Startup prints those
settings; the final MODE3 counters show whether the optimized path succeeded.

## FP4 decode, prefill and workgroups

The packed layout is unchanged. The new bytecode kernels load four packed bytes
at once, decode eight values per lane, and reuse activations across four output
rows (decode) or decoded weights across eight input rows (prefill). Reductions
share barriers across rows and work with Intel SIMD widths as well as wider GPUs.

At preparation time, the first weight of each backend/K/N geometry compares
32/64/128-thread candidates against the previous GPU kernel. Candidates with
nonfinite values or excessive numerical error are rejected. Three interleaved
timing rounds select a kernel; every layer with that geometry reuses the plan.
The `[gpu-tune]` messages report the actual selected group width. Activation and
state arenas remain resident between layers and reuse allocations across prompts
of the same geometry.

Useful overrides:

- `BYBY_GPU_AUTOTUNE=0`: use the tested wide kernel with a 32-thread group.
- `BYBY_GEMV_VARIANT=0/1/2`: retain the previous decode implementations.
- `BYBY_PREFILL_FP4_KERNEL=m8` or `m4`: retain the previous prefix implementation.
- `BYBY_GEMV_TG=32/64/128`: restrict new tuning candidates to that group width;
  the old kernel remains a correctness/performance fallback.
- `BYBY_PREFILL_RESIDENT=0`: retain the legacy sequence prefill.
- `BYBY_MACOS_HIGH_PRIORITY=0`: retain inherited CPU thread QoS.

## NVIDIA Tensor Cores through Vulkan

The backend queries physical-device limits, subgroup size, extensions, features
and cooperative-matrix tile types. On Vulkan 1.3 hardware exposing the KHR
16x16x16 FP16/FP32 tile, it enables cooperative matrix, float16 and Vulkan memory
model features before device creation. GEMM uses FP16 shared-memory input tiles,
FP32 accumulation/output and one subgroup per output tile. FP32 row-major
weights, FP4-KM group32 and asymmetric Q4 are supported. Packed weights are
dequantized into temporary GPU tiles rather than a full FP32 weight copy.

This path is used for batches of at least eight input rows. Single-token decode
uses the bandwidth-oriented FP4/GEMV kernels. Devices without the required
features keep the ordinary GPU path. A compiler rejecting cooperative-matrix
GLSL disables the optional path and reports the reason.

Discrete Vulkan GPUs upload resident weights once into device-local VRAM via a
staging transfer; UMA retains coherent shared allocations. Transfer source and
destination usage flags are declared on buffers used in GPU copies. The backend
prefers a discrete GPU automatically; `BYBY_VULKAN_DEVICE=<index>` selects one.

`BYBY_VULKAN_TENSOR_CORES=0` disables MMA. `BYBY_VULKAN_DEVICE_LOCAL=0/1`
overrides automatic resident-memory selection. Vulkan workgroup and shared-memory
specialization constants are included in pipeline cache keys and bounded by the
queried device limits. `BYBY_VULKAN_MAX_TG` can lower the hardware limit.

cuBLAS is a CUDA-runtime library. This implementation accesses matrix hardware
through Vulkan cooperative matrices, without a cuBLAS/cuDART interop layer.

## Validation and measurement

- `test_resident_prefill.nim`: FP4 plus nonzero LoRA, R/G/P mixing, prefix lengths
  2/3/7/15/33/64, recurrent states, next-token logits and scratch-cache reuse.
  `BYBY_TEST_BACKEND=vulkan` selects Vulkan.
- `bench_fp4_dynamic.nim`: actual Codestral projection dimensions and comparison
  with the old GPU output.
- `bench_resident_prefill.nim`: `BYBY_BENCH_AB=1` loads once and runs warmup,
  old kernels, then tuned kernels. `BYBY_BENCH_TOKENS` controls generation length.
- `test_vulkan_tensor_core.nim`: Vulkan ABI layout, resident staging/copy, and
  FP32/FP4/Q4 arithmetic against an FP64 reference, including partial tiles.
  `BYBY_REQUIRE_TENSOR_CORES=1` requires a real enabled cooperative-matrix GPU.
  Without compatible hardware, arithmetic is explicitly reported as skipped.

On the Iris Plus 640, the same-process warm A/B with the supplied 15-token
prefix and eight generated tokens measured prefill 18.980s -> 15.857s
(0.79 -> 0.95 tok/s), decode 10.217s -> 6.535s (0.78 -> 1.22 tok/s).
The initial cold tuned pass measured 0.36 tok/s prefill and 0.81 tok/s decode.
These short measurements do not establish sustained thermal performance.
No NVIDIA hardware was available for MMA timing or arithmetic execution.

## References

- [llama.cpp Vulkan GEMM](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp):
  tiled shared loads, quantized GEMM and cooperative matrix instructions.
- [KHR cooperative matrix](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html):
  feature and supported tile enumeration.
- [cuBLAS introduction](https://docs.nvidia.com/cuda/cublas/index.html): CUDA-runtime
  library and its memory/API contract.

Vendor changes are included in `patches/bybylang-inference-gpu.patch` and applied
by Make before compilation, after the existing resident-MPS patch.
