# Codestral inference benchmark

## Result

Latest MacBook Pro runs (Intel Iris Plus 640, revision `226e08f` with the lane
scan, LoRA-down and Linear++ lanes). With warm shader caches, Vulkan prefilled
the 11-token prompt in **6.821 s** and decoded **1.75 tok/s** after the first
token; Metal took 7.910 s and decoded 1.57 tok/s.

| Backend | Prefill, 11 tokens | Decode, 60 tokens | Steady decode after first token | Total, 71 tokens |
| --- | ---: | ---: | ---: | ---: |
| Vulkan, warm caches | **6.821 s · 1.61 tok/s** | 34.471 s · **1.74 tok/s** · 574.5 ms/token | **1.75 tok/s** · 572.4 ms/token | **41.292 s** |
| Vulkan, cold caches (first run after new kernels) | 15.870 s · 0.69 tok/s | 42.008 s · 1.43 tok/s · 700.1 ms/token | 1.43 tok/s · 697.9 ms/token | 57.878 s |
| Metal | 7.910 s · 1.39 tok/s | 38.174 s · 1.57 tok/s · 636.2 ms/token | 1.57 tok/s · 637.6 ms/token | 46.083 s |

With warm caches, Vulkan prefill was about **14% faster** than Metal and steady
decode about **11% faster**. The cold Vulkan run was the first one after the
kernels changed, so glslc and MoltenVK had to compile every shader; see
[Vulkan device path](#vulkan-device-path). All three runs produced the same
text. They are separate user runs, not a controlled A/B benchmark.

Earlier runs (before the lane scan) for reference:

| Backend | Prefill, 11 tokens | Decode, 60 tokens | Steady decode | Total |
| --- | ---: | ---: | ---: | ---: |
| Vulkan | 8.629 s · 1.27 tok/s | 30.313 s · 1.98 tok/s · 505.2 ms/token | 2.00 tok/s · 499.5 ms/token | 38.943 s |
| Metal | 10.991 s · 1.00 tok/s | 33.809 s · 1.77 tok/s · 563.5 ms/token | 1.77 tok/s · 564.0 ms/token | 44.800 s |

Prefill improved on both backends (Vulkan 8.629 → 6.821 s, Metal 10.991 →
7.910 s). The prefill work did not touch the decode path, so the lower decode
rate in the latest runs (1.75 vs 2.00 tok/s on Vulkan) is not explained by
these changes; system load and thermal state differ between sessions.

## Tesla T4 prefill (Vulkan, Colab)

Three user runs on a Colab Tesla T4 (Vulkan, `KHR_cooperative_matrix`
FP16/FP32, subgroup 32), one per prefill revision. All three used the same
prompt and greedy sampling and produced identical output text.

- Prompt: `[INST] Write a Python function for Fibonacci. [/INST]` (17 tokens prefilled)
- Generation: greedy (`T=0.0`), stopped at EOS after 233 tokens
- Model: same checkpoint, base model and 64-block topology (60 R, 4 G) as below

| Revision | Prefill, 17 tokens | Scan | `resident.group` | One-time setup* | Decode after first token |
| --- | ---: | --- | ---: | ---: | ---: |
| `6f09d78` (before) | 10.214 s · 1.66 tok/s | parallel_n4 | 5285 ms | 4843 ms | 13.65 tok/s · 73.3 ms/token |
| `6e0a5d5` lane scan, tiled FP4 GEMM, leaner prewarm | **2.772 s · 6.13 tok/s** | lanes | 1124 ms | 1595 ms | 14.96 tok/s · 66.8 ms/token |
| `226e08f` LoRA-down, Linear++ lanes, tiled FP32 GEMM | 3.041 s · 5.59 tok/s | lanes | **865 ms** | 2123 ms | 14.44 tok/s · 69.3 ms/token |

\* `vulkan.prewarm` + `scan.autotune` + `gemm.verify`: glslc shader compiles
and one-time GPU parity checks. On a fresh container they are paid once per
process. Prewarm shrinks on later runs if `BYBY_VULKAN_JIT_DIR` points at
persistent storage.

What changed between runs, from the `[prefill-profile]` lines:

- `6f09d78`: the parallel scan took 55.4 ms per layer (autotune probe), about
  3.5 s of `resident.group` over 64 layers. Autotune timed serial, parallel n1
  and n4 (2654 ms), and prewarm (2189 ms) compiled packed FP4 shaders that the
  T4 never used because the cooperative-matrix route won.
- `6e0a5d5`: the lane scan keeps states in registers and passed its parity
  check (max error 2.42e-8). The tiled FP4 GEMM passed parity for 8192x4096 and
  4096x18560. `resident.group` fell to 1124 ms and prefill to 2.772 s.
- `226e08f`: LoRA A, the Linear++ sidecar and its FP32 GEMMs (4096x12288,
  8192x4096, 4096x4096, all parity-checked) now fill the GPU. `resident.group`
  fell another 23% to 865 ms and `enc.attn` from 340 ms to 188 ms. Total
  prefill rose to 3.041 s because `gemm.verify` checks five shapes instead of
  two (622 ms vs 229 ms) and prewarm took 1335 ms vs 1190 ms.

Without the one-time setup, prefill work was about 5.35 s, then 1.16 s, then
0.90 s. With only 17 prompt tokens the one-time setup dominates the wall
time; a longer prompt, or a second prompt in the same process, would show more
of the `resident.group` gain. Decode code did not change between these
revisions, so the 13.65–14.96 tok/s spread is run-to-run variation.

## Test machine (MacBook runs)

- MacBook Pro 13-inch (2017), two Thunderbolt 3 ports
- CPU: 2.3 GHz dual-core Intel Core i5
- GPU: Intel Iris Plus Graphics 640, 1536 MB
- Memory: 16 GB 2133 MHz LPDDR3
- macOS Ventura 13.7.8

## Inference configuration

- Checkpoint: `test/codestral/models/jamba_parallel_lora.bin`
- Base model: `test/codestral/models/codestral_mamba_only.nimq`
- Prompt: `[INST] Xin chao[/INST]` (11 prompt tokens after trailing whitespace is removed)
- Generation: 60 tokens, temperature 0.7
- Model topology reported by inference: 64 blocks (60 R, 4 G), 128 LoRA adapters
- All runs used resident prefill and native decode; MPS was disabled on this
  Intel Mac (`Apple Silicon=false`).

Commands used:

```sh
make infer-lora BYBY_INFER_BACKEND=vulkan
make infer-lora BYBY_INFER_BACKEND=metal
```

## Vulkan device path

The Vulkan log identifies MoltenVK running on the Intel Iris Plus 640 and
reports `residentWeights=host-visible/shared` and `matrix=scalar`. There is no
cooperative-matrix path on this GPU, so prefill uses the packed FP4 VM kernels
(`packed tiles=256`) and the batched FP32 sidecar kernel (`batched=12`). The
scan auto mode picked the register lane scan on both backends.

Prefill profile of the latest runs:

| Stage | Vulkan, cold | Vulkan, warm | Metal |
| --- | ---: | ---: | ---: |
| `resident.group` | 11242 ms (8 groups) | 6604 ms (8 groups) | 6380 ms (16 groups) |
| `vulkan.prewarm` (glslc) | 2256 ms | 22 ms | — |
| `scan.autotune` (lane parity check) | 2152 ms | 62 ms | 1202 ms |
| `enc.inproj` + `enc.outproj` + `enc.attn` (host) | 1126 ms | 331 ms | 10 ms |

The cold Vulkan run paid for shader compilation three times: glslc for
GLSL → SPIR-V (`vulkan.prewarm`), the lane-scan kernel compiled during its
parity check, and MoltenVK converting SPIR-V → MSL and compiling a Metal
pipeline at the first dispatch of each kernel inside `resident.group`. Metal
compiles its kernels before the prefill loop and keeps them in
`metal-jit.binarchive`. Once the SPIR-V files and the macOS shader cache are
warm, Vulkan's `resident.group` is within 4% of Metal's.

Kernel-level measurements on the same GPU with synthetic data (T=11, one
dispatch each) show the Vulkan kernels are not slower:

| Kernel | Metal | Vulkan |
| --- | ---: | ---: |
| FP4 2-column m8, 4096x18560 | 22.1 ms | 19.6 ms |
| FP4 2-column m4 tail, 4096x18560 | 13.5 ms | 12.0 ms |
| FP4 2-column m8, 8192x4096 | 13.0 ms | 7.3 ms |
| FP32 batch, 4096x12288 | 27.1 ms | 17.0 ms |
| Lane scan, 128 heads x 64 x 128 | 2.9 ms | 1.4 ms |

Replaying the command sequence of 64 R blocks (distinct weights, about 3.4 GB,
copies included) took about 2.7 s on both backends once warm. With freshly
named kernels the first Vulkan group took 2546 ms against 413 ms on Metal, and
424 ms on a second Vulkan run. Buffer copies (blit encoders on MoltenVK,
compute copies on Metal) cost about 10% on both.

## Source logs

- MacBook runs: user-supplied `make infer-lora` logs at revision `226e08f`
  (Vulkan run twice: cold 15.870 s and warm 6.821 s prefill; Metal 7.910 s).
  The warm Vulkan run also included the fix that skips the tiled GEMM check on
  GPUs without cooperative matrices.
- Earlier MacBook runs: Vulkan 1.98 tok/s and Metal 1.77 tok/s decode, before
  the lane scan.
- Tesla T4 runs: three user-supplied Colab Vulkan logs (revisions `6f09d78`,
  `6e0a5d5`, `226e08f`), prefill 10.214 s, 2.772 s and 3.041 s.
- Kernel and replay measurements: synthetic benchmarks on the MacBook GPU,
  not checked into the repository.
