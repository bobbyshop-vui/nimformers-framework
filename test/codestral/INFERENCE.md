# Codestral inference

`make infer-lora` defaults to `BYBY_GEMV_TG=auto`, `BYBY_GPU_AUTOTUNE=1`
and `BYBY_DEBUG_SAMPLING=0`. Explicit environment settings override defaults.
The executable applies the same defaults when invoked directly.

The decode FP4 GEMV autotuner also tries `simd4-tg<N>` candidates on Metal: the same loads as the
`wide4` kernels, reduced with `simd_sum` and one partial per simdgroup instead of the barrier tree
(`buildFp4KmGemvSimdProgram`). A candidate is used only when it matches the baseline kernel.
Mamba-3 shares this tuner through `gpu_layers`.

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

Metal prefill also has a native packed-FP4 GEMM (`kernels/gemm_fp4km.metal`): a 256-thread
register-blocked kernel decodes each 32-value group of the packed weights once into threadgroup
memory and reuses it for every row of the tile (BM = 16/32/64 chosen from the row count). On an
Apple GPU `kernels/gemm_fp4km_sg.metal` runs the same math with `simdgroup_matrix` (8x8 float
accumulators, 4 simdgroups per 64x64 tile); it lives in its own library that is compiled only on an
Apple GPU, and it is admitted only after it reproduces the register-blocked kernel on 16 rows of a
real projection (`[fp4-gemm] simdgroup_matrix parity ...`). `BYBY_PREFILL_FP4_GEMM=auto|rb|sg|off`
selects the kernel; `BYBY_PREFILL_FP4_GEMM_MIN_ROWS` is the smallest prefill row count that uses it
(default 8 on Apple GPUs, 32 elsewhere, where the 8-row micro-VM tiles are as fast below that).
`BYBY_FP4_SG_COMPILE_CHECK=1` compiles the simdgroup source on any GPU without building a pipeline.

The FP32 matmul kernels (`kernels/gemm_blocked.metal`) are register-blocked as well: 4x4, 2x4 or 1x4
accumulators per thread over 64/32/16-row tiles. `BYBY_METAL_GEMM_RB=0` selects the older 8x8 kernels.

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

## Whole-prompt prefill (micro-batch)

A prompt of up to `BYBY_UBATCH` tokens (default 512, range 2..4096, `0` = 4096;
`BYBY_PREFILL_BATCH_SIZE` is the older name) runs as **one** resident prefill pass: every
projection is one GEMM over all of its rows, and the sequence scan covers the whole prompt. On
Vulkan with tensor cores that scan is the chunked SSD pass (its matmuls on the tensor cores, so the
time axis is parallel too); elsewhere the register lane scan runs one 32-lane group per channel over
the whole prompt (the timestep-parallel scan is limited to a threadgroup of rows, so `auto` falls
back to it only for short prompts). A longer prompt runs in passes of this size with the recurrent
state carried between them. One pass also means one set of T-specialized shaders: a 246-token
prompt used to compile and autotune twice (128 + 118 rows).

When the SSD pass runs, the scan autotune (lane parity, serial/parallel timing) is skipped, since no
scan kernel is encoded. `BYBY_PREFILL_LAYERS_PER_COMMIT` defaults to four on Metal, eight on Vulkan.

`formers-cli` keeps 128-token chunks for Codestral unless `BYBY_UBATCH` is set: a lifted FIR graph
holds one phase per chunk and tail length, and the lifted models were recorded with 128.

The prefill copies the Mamba parameters (`convW|convB`, `aLog|dSkip|dtBias|normW`) out of the
decode work buffer. That buffer aligns every region to 64 floats, so the regions are adjacent only
when H and cDim*K are multiples of 64 (Codestral); other shapes are copied region by region. (Before
this, small test models read a shifted `dSkip`/`dtBias` and the integration test failed.)

## Decode GEMV: subgroup rows and the fused residual

The FP4 decode GEMV autotune times on the GPU (Vulkan timestamps are turned on for its runs), so
its choice is not dominated by submit latency. With fixed 32-lane subgroups (NVIDIA on Vulkan,
Apple GPUs on Metal; `BYBY_GEMV_SUBGROUP=auto|1|0`) it also tries subgroup-row micro-kernels
(`buildFp4KmGemvSubgroupProgram`): one subgroup per 2-4 output rows, two weight words per lane in
flight, a `subgroupAdd`/`simd_sum` reduction instead of the shared-memory tree, the LUT in
threadgroup or constant memory. Every candidate must reproduce the baseline kernel first, so a
pipeline the driver runs narrower than 32 lanes (Intel picks SIMD16 under register pressure) is
dropped. When the chosen out_proj kernel is a subgroup-row or decode kernel, the R block's out_proj
adds into the residual itself (`BYBY_DECODE_FUSE_RESIDUAL=1`, default): one dispatch and one barrier
less per layer. `test_fp4_gemv_subgroup.nim` checks every geometry (plain and fused) against the
baseline kernel, and `test_decode_replay.nim` (`REPLAY_EMBED=256`) compares fused and unfused logits.

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

## Prefill tok/s without compile time

The `Prefill:` line of the speed stats is compute only. Two one-time costs are subtracted from
the prefill wall time and printed next to it:

- shader compiles: `glslc` runs and Vulkan pipeline creation, Metal library and pipeline-state
  builds, cache misses only. The drivers keep a running total (`sessionCompileSeconds`).
- prefill autotune and parity checks: the scan autotune, the SSD parity check, the MPS-vs-MSL
  timing, the simdgroup GEMM parity and the tiled GEMM check (`gPrefillTuneSeconds`), without
  the compiles they trigger, so nothing is subtracted twice.

```
Prefill: <N> tokens in <compute>s -> <rate> tok/s (excluding <c>s shader compile + <t>s autotune; wall <w>s -> <rate> tok/s)
```

A warm run, where every kernel is cached and every tune decision is reused, prints the plain
`Prefill: <N> tokens in <w>s -> <rate> tok/s` line. formers.n prints its `prompt eval` line the
same way.


## Decode replay

A decode step encodes the same dispatches on the same buffers for every token; only the
embedding row changes. On Vulkan and Metal the step is therefore encoded once more after the
first token into a kept command buffer (`sessionBeginRecording`), and every later token writes
its embedding row into a host-visible buffer, resubmits the recording and reads the logits from
another host-visible buffer. The host encode of 64 layers disappears from the per-token time,
which matters most on a slow host CPU such as a Colab VM. Prompt tokens fed through decode use a
second recording without the head. The run prints `[infer] decode replay on: ...` once.

The recordings are dropped on GPU model preparation, every state reset and the LoRA merge, and
re-recorded after one encoded step. `BYBY_DECODE_REPLAY=0` encodes every step.
`test_decode_replay.nim` checks on a small model that replayed logits equal encoded logits, with
and without the head, across two resets.

On GPUs with tensor cores the prompt tail now uses the same chunk lengths as elsewhere
(64, 32, 16, 8 and `BYBY_PREFILL_MIN_ROWS`), not repeated 16-row chunks. The formers.n warmup
runs one prompt that covers every chunk length, so none compiles during a timed prompt.

## Bounded in-place merge

```sh
make infer-lora
```

`inplace` is the default, `1` is an alias for it, and `0` keeps separate LoRA.
On by default the merge steps aside when it cannot run (`BYBY_ONE_DISPATCH=1`,
`BYBY_MODEL_MMAP=0`, no memory headroom) and inference continues with the
separate adapters; requested by hand, those cases are errors instead.
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

The existing FP4 requantization behavior is retained. The merge is mandatory: there is
no separate-adapter mode, the memory guard only warns, and an adapter that cannot be
merged (for example on a non-FP4 tensor) stops the run. The log reports each tensor's
GPU merge time, the number merged and peak RSS.

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
serial/coalesced/parallel scans, MPS/packed GEMMs, state carry, 257 tokens in 128-token chunks
and 300 tokens as one micro-batch. It also checks that the first prefill of the process counts
its shader compiles and that a repeated prefill of the same shape excludes almost nothing (only
cache lookups). `test_prefill_lane_scan_gpu.nim` checks the lane scan up to 512 timesteps.
`test_inplace_lora_merge.nim` covers error and memory guards, packed-bit
preservation, buffer identity and decode logits after an accepted merge.
