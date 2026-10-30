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

## Test machine

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
