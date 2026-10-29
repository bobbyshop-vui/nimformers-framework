# CPU backend

`vendor/bybylang/backends/cpu/` implements a no-GPU backend for BybyLang and
Nimformers. It provides the same immediate operations, session API, resident
weights, and micro-VM bytecode execution as the Metal/Vulkan backends.

## Files

| File | Purpose |
| --- | --- |
| `cpu_kernels.h` | C ABI used by the Nim wrapper |
| `cpu_kernels.c` | feature detection, ISA dispatch tables, cache-aware blocking |
| `cpu_kernels_impl.inc` | kernel bodies included once per ISA |
| `cpu_vm.c` | lockstep micro-VM interpreter (R/U registers, threadgroup/grid barriers) |
| `cpu_backend.nim` | Nim wrapper, sequence marshalling, `cpuInfo()` / `cpuFeatures()` |

## Runtime ISA dispatch

The C constructor probes the CPU once and builds per-ISA function tables:

| ISA | How it is built |
| --- | --- |
| scalar | baseline C loops |
| SSE4.1 | `__attribute__((target("sse4.1")))` |
| AVX | `target("avx")` |
| AVX2 + FMA | `target("avx2,fma")` |
| AVX-512F | `target("avx512f")` |
| NEON | AArch64/ARM NEON (default target) |

The source uses portable loops; Clang/GCC auto-vectorize each target-marked
function. Disassembly of the generated object contains `vaddps`/`vfmadd` with
`ymm`/`zmm` registers, so the ISA selection is real, not nominal.

### Small-operation policy (AVX frequency offset)

On Intel, entering wide AVX/AVX-512 code can reduce the core frequency for a
short period ("AVX offset"). Small elementwise jobs therefore use SSE4/NEON:

```
n <= NIMFORMERS_CPU_SMALL (default 4096)       -> SSE4 / NEON
larger elementwise                             -> AVX2+FMA (or AVX)
work >= NIMFORMERS_CPU_AVX512_MIN (default 131072) -> AVX-512F if present
```

Matmul uses a separate work estimate (`M*K*N`): small products stay on SSE,
large products use AVX2/AVX-512. The threshold is chosen so that vector setup
and frequency transitions do not dominate tiny kernels.

### L3-aware matmul blocking

The dispatcher reads L1/L2/L3 sizes and selects `NC/KC/MC` tiles:

- the packed B panel (`KC x NC`) is sized to be reused from L3 across row
  blocks,
- the C tile (`MC x NC`) is kept inside L2,
- the inner loop is a contiguous `C[i,:] += A[i,k] * B[k,:]` axpy that
  auto-vectorizes with FMA.

Override with `NIMFORMERS_CPU_BLOCK_NC`, `NIMFORMERS_CPU_BLOCK_KC`, and
`NIMFORMERS_CPU_BLOCK_MC`.

## Environment controls

| Variable | Meaning |
| --- | --- |
| `NIMFORMERS_CPU_ISA` | `scalar`, `sse`, `avx`, `avx2`, `avx512`, or unset/auto |
| `NIMFORMERS_CPU_SMALL` | element threshold for preferring SSE |
| `NIMFORMERS_CPU_AVX512_MIN` | work threshold before AVX-512 is considered |
| `NIMFORMERS_CPU_BLOCK_NC` / `_KC` / `_MC` | matmul cache tiles |

Inspect the detected values from Nim or Python:

```nim
import vendor/bybylang/backends/cpu/cpu_backend as cpu
echo cpu.cpuInfo()
# (isa: 3, isaName: "avx2", hasSse4: true, ..., l3Bytes: ...)
```

```python
from nimformers import cpu_info
print(cpu_info())
# {'isa': 3, 'isa_name': 'avx2', ..., 'block_mc': 32, 'block_nc': 256, 'block_kc': 128}
```

## Supported surface

- elementwise add/sub/mul/div, scalar ops, relu/sqrt/exp/log/sigmoid/tanh,
  pow, APFLU + backward, fused add+activation
- cache-blocked matmul, matmul-NT, int4 (Q4) matmul
- softmax, layernorm, RMSNorm, layernorm backward
- causal fused attention (forward), attention backward
- embedding, gather, compare, dropout mask, argmax, permute, split, concat,
  RoPE
- session upload/read/free, session ops, resident FP32/Q4 weights
- micro-VM: core arithmetic, loads/stores, LUT, loops/ifs, U-register address
  ops, threadgroup memory, threadgroup and grid barriers

## Limitations

- The CPU kernels are single-threaded. Matmul is cache-blocked and SIMD, but
  does not yet use OpenMP or multiple cores.
- AVX-512 is used only when the CPU exposes `avx512f`; macOS Intel CPUs often
  do not.
- The micro-VM advances threads in lockstep. This preserves barrier semantics
  but is much slower than GPU dispatch; it is intended for correctness on
  machines without a GPU, not for throughput.
- Virtual-threadgroup opcodes (`VTG_*`) execute as a normal threadgroup;
  `LOAD_VACT` reports active.

## Tests

```bash
make bybylang-test        # includes test/bybylang_test/cpu_test.bybylang
pytest -q test/python_test/test_cpu_backend.py
```

The Python CPU suite covers SIMD math, softmax/layernorm/attention, Q4,
resident weights, the CPU micro-VM, and a tiny Llama generation.

## Benchmark

Measure FP32 matmul GFLOPS across the CPU, Metal, and Vulkan backends:

```bash
python3 benchmarks/bench_gflops.py
# or: make bench-gflops
python3 benchmarks/bench_gflops.py --backends cpu,metal,vulkan \
    --sizes 256,512,1024 --iters 5

# A single 8096^3 dispatch can exceed the Metal/MoltenVK command-buffer
# watchdog on integrated GPUs (the iGPU here times out above 6144^3).
# The tiled mode computes the same 2*N^3 FLOPs in short tile dispatches:
python3 benchmarks/bench_gflops.py --backends metal,vulkan \
    --sizes 8096 --iters 1 --warmup 0 --tile 1024

# same preset, shortest command:
python3 benchmarks/bench_gflops.py --gpu-8096
```

Unavailable backends are reported as `SKIP`; results can be exported with
`--csv results.csv`.  The script reports both best and median samples and the
CPU ISA/cache tiles in use.  GPU CPU-fallback is disabled by default so a
failed kernel is not silently measured as CPU; use `--allow-cpu-fallback`
only if you explicitly want that.
