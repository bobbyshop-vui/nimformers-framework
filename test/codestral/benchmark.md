# Codestral inference benchmark

## Result

On the MacBook Pro below, the latest supplied Vulkan run decoded **60 tokens
at 1.98 tok/s overall**. Excluding the first token, the log reports **2.00
tok/s**. The latest supplied Metal run reports **1.77 tok/s** for 60 tokens.

| Backend | Decode, 60 tokens | Steady decode after first token | Prompt prefill, 11 tokens | Total, 71 tokens |
| --- | ---: | ---: | ---: | ---: |
| Vulkan | 30.313 s · **1.98 tok/s** · 505.2 ms/token | **2.00 tok/s** · 499.5 ms/token | 8.629 s · 1.27 tok/s | 38.943 s |
| Metal | 33.809 s · **1.77 tok/s** · 563.5 ms/token | 1.77 tok/s · 564.0 ms/token | 10.991 s · 1.00 tok/s | 44.800 s |

The supplied logs show Vulkan decoding about **12% more tokens per second**
than Metal in these runs. Vulkan prefill was about **21% faster**, and total
wall time was about **13% lower**. These are results from separate user runs,
not a controlled same-binary A/B benchmark; treat the comparison as indicative.

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
- Both runs used resident prefill and native decode; MPS was disabled on this
  Intel Mac (`Apple Silicon=false`).

Commands used:

```sh
make infer-lora BYBY_INFER_BACKEND=vulkan
make infer-lora BYBY_INFER_BACKEND=metal
```

## Vulkan device path

The Vulkan log identifies MoltenVK running on the Intel Iris Plus 640 and
reports `residentWeights=host-visible/shared` and `matrix=scalar`. No Vulkan
cooperative-matrix/tensor-core path was active in this run. The model still
used the packed FP4 VM kernels (`packed tiles=256`); tensor-core GEMMs were 0.
The automatic scan selected serial (`12.327 ms`) over parallel (`40.326 ms`).

The measured Vulkan prefill profile was 8.586 s total: resident groups took
7.920 s across 8 groups, scan autotuning 425 ms, and Vulkan shader prewarm 124
ms. Decode was 30.313 s for 60 tokens; the steady-state profile was about 496
ms/token.

Metal's supplied run used the same Intel GPU. It also reported no tensor-core
GEMMs and no MPS tiles, with 256 packed tiles; its prefill took 10.991 s and
decode took 33.809 s. Since the runs were captured separately, differences in
binary revision, warmup, and system load may affect the comparison.

## Source logs

- Latest Vulkan run: user-supplied `make infer-lora BYBY_INFER_BACKEND=vulkan`
  log, reporting 1.98 tok/s overall and 2.00 tok/s after the first token.
- Metal comparison: user-supplied Metal run in the preceding combined log,
  reporting 1.77 tok/s overall.
- Tesla T4 runs: three user-supplied Colab Vulkan logs (revisions `6f09d78`,
  `6e0a5d5`, `226e08f`), prefill 10.214 s, 2.772 s and 3.041 s.
