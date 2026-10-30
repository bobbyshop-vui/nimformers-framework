# Codestral inference

`make infer-lora` defaults to `BYBY_GEMV_TG=auto`, `BYBY_GPU_AUTOTUNE=1`
and `BYBY_DEBUG_SAMPLING=0`. Explicit environment settings override defaults.
The executable applies the same defaults when invoked directly.

Metal prefill uses the packed FP4 kernels by default. The old MPS path expanded
one weight region at a time to FP32 before every matmul; on an integrated GPU
that rewrite costs more bandwidth than the GEMM it feeds, so it is now opt-in
with `BYBY_PREFILL_GEMM=mps`. Decode uses native compute kernels; both native
MPS entry points reject M=1.

MPS is considered only on Apple Silicon. For FP32 batched prefill GEMMs and
LoRA matrix products, inference warms and times the matching MSL and MPS paths
on the current input/weight shape; MPS is selected only when it wins by at
least 3%. Other Macs and Vulkan use MSL/packed kernels. Logs report the device
gate and per-shape choice as `[prefill-matrix] ... MSL=... MPS=... -> ...`.

Vulkan inference checks device features and the exact cooperative-matrix tile
before selecting a low-precision path. For FP4-KM prefill it prefers E4M3 FP8
when supported, decoding packed weights into FP8 only as each tile is loaded;
otherwise it uses the FP16-input/FP32-accumulate matrix kernel. FP32 sidecars
also try the FP16 matrix path. Unsupported shapes, hardware, or shader compiler
fall back to the existing packed VM/scalar kernels. No full dequantized weight
copy is created. FP8 converts activations and decoded weights to E4M3, so it
uses less precision than the FP16 route; startup logs identify the route chosen.
The prefill log counts tensor-core and packed dispatches. Scan
defaults to `auto`, which benchmarks the serial and parallel scan on the
selected GPU and caches the faster route for each shape; `coalesced` remains a
manual option.

Packed FP4 prefill tiles eight prompt rows per dispatch and finishes the last
1..4 rows with a four-row tile, so an 11-token prompt costs 12 rows of work
instead of 16 (`BYBY_PREFILL_TAIL4=0` restores 8-row padding only).
`BYBY_PREFILL_DIAG=1` commits after every prefill stage, so
`BYBY_PREFILL_PROFILE=1` reports GPU time per stage instead of only host
encode time; both are diagnostics, not fast paths.

These settings remain available for comparisons:

```sh
make infer-lora BYBY_INFER_BACKEND=metal
make infer-lora BYBY_INFER_BACKEND=vulkan
make infer-lora BYBY_PREFILL_GEMM=mps BYBY_PREFILL_SCAN=serial
```

Prefill batches default to 128 tokens (`BYBY_PREFILL_BATCH_SIZE=2..128`).
State carries across chunks. `BYBY_PREFILL_LAYERS_PER_COMMIT` defaults to four.

`BYBY_PREFILL_SCAN=auto` benchmarks compact serial and timestep-parallel scans
on the selected GPU; set `serial`, `parallel`, or `coalesced` to force a route.
Packed FP4 prefill uses the 8-row by 2-column tile when the shape is supported
(`BYBY_PREFILL_2COL=0` selects the older one-column tile). The shared activation
loads reduce repeated input traffic across neighboring output columns. A 4-row
2-column tile handles short tails.

Metal JIT pipelines are added to an `MTLBinaryArchive` and saved at process
exit. The default file is under `~/Library/Caches/nimformers/metal-jit.binarchive`;
`BYBY_METAL_JIT_ARCHIVE` overrides the path. The archive is device/driver
specific and Metal may rebuild invalidated binaries. On systems where Metal
does not support archive serialization, pipelines still use the in-process
cache and startup reports that disk caching is unavailable.

## Vulkan first-run shader compilation

Vulkan runs the micro-VM kernels and the FP4 tiles as GLSL that `glslc`
compiles at runtime. That compilation is the only prefill cost that does not
shrink on faster hardware: ~30 shaders at ~0.5 s each dominate the first
prefill on a T4/M-series GPU, where the whole GEMM is a few hundred ms.

- Every shader a prefix needs is now collected and compiled **concurrently**
  (bounded by CPU cores) before the layer loop, instead of one at a time on
  first dispatch. The run reports it in the profile as `vulkan.prewarm`.
- `BYBY_VULKAN_JIT_DIR` moves the SPIR-V cache out of the temp dir. Point it at
  persistent storage (Colab: a mounted Drive, CI: a workspace cache) and only
  the first run ever compiles:

  ```sh
  BYBY_VULKAN_JIT_DIR=/content/drive/MyDrive/nimformers-jit \
    make infer-lora BYBY_INFER_BACKEND=vulkan
  ```

- `BYBY_GPU_AUTOTUNE=0` skips the FP4 tuner, which otherwise compiles ~26
  candidate shaders on a cold cache (it still runs the baseline kernel). Use it
  for a one-shot measurement run or when the first-start latency matters.
- `BYBY_VULKAN_DISPATCH_CACHE=1` (default) keeps the cache enabled; `0` forces
  a recompile on every dispatch and is only for diagnostics.


## Bounded in-place merge

```sh
make infer-lora BYBY_MERGE_LORA=inplace
```

`1` is an alias for `inplace`; `0` keeps separate LoRA (the default).
`1`/`inplace` reproduce the original merge from commit 24111a9: dequantize,
add each LoRA rank in order, then run the same FP4-KM scale search and packing.
GPU kernels compute both passes over 32-weight groups, including LoRA, scale
reduction, scale search and FP4 packing. CPU only uploads small adapter/format
data, schedules dispatches and reads scalar status. No CPU weight merge or
silent CPU fallback runs. Tests compare every packed byte
and global scale with the original loop and `quantizeTensor` at ranks 1/4/8.
Merge requires mmap and independent decode kernels (`BYBY_ONE_DISPATCH=0`).
Each tensor uses two GPU submissions (max/scale, then every tile plus the LUT)
on one reused command queue and one reused scratch buffer, and `loadGroup` is
always inlined so the 32-value group stays in registers. The kernel is
compute-bound on the eight-candidate scale search: about 0.24 s for a
4096x18560 in-projection and 0.11 s for an 8192x4096 out-projection.
`bench_merge_gpu.nim` times one tensor without loading the model.
It packs tiles directly on GPU and overwrites the existing weight buffer
and decode LUT. It never builds a dense FP32 model or changes
checkpoint files. GPU buffer identities are preserved.
GPU tile scratch is at most 8 MiB, within the 64 MiB total merge budget.
The process memory guard targets 7,000 MB and reserves 256 MiB for overhead.
Linux KiB and macOS byte RSS measurements are normalized before checking.
This is a process RAM guard; discrete GPU VRAM has separate accounting.

The existing FP4 requantization behavior is retained. Use `BYBY_MERGE_LORA=0`
to keep separate LoRA. The former CPU `safe` mode is not accepted by the GPU
inference path. The log reports each tensor's GPU merge time, the number
merged and peak RSS. Non-FP4 tensors keep their adapters.

On Iris Plus 640 with actual Codestral tensors (`block0.mambaReal.inProj.weight` and `outProj.weight`):
- `outProj.weight` (4096x8192, 17 MiB packed FP4): Metal 5.013s / Vulkan 13.888s (including initial pipeline compilation)
- `inProj.weight` (18560x4096, 76M parameters, 38 MiB packed FP4): Metal 0.213s / Vulkan 0.290s
- Per-tensor merge throughput reaches ~350M parameters/s directly in GPU memory with zero CPU loops and < 8 MiB scratch budget.

## Validation

The scan benchmark uses actual Codestral dimensions and compares numerical
outputs against the original scan. On Iris Plus 640, median times for 64
tokens were Metal 61.44 -> 25.98 ms, Vulkan 96.65 -> 28.61 ms. These measure
the scan only, including state transpose overhead, not whole-model tokens/s.

`test_resident_prefill_integration.nim` covers R/P/G with nonzero LoRA, bias,
serial/coalesced/parallel scans, MPS/packed GEMMs, state carry and 257 tokens.
`test_inplace_lora_merge.nim` covers error and memory guards, packed-bit
preservation, buffer identity and decode logits after an accepted merge.
