# Runtime flags (Vulkan, native and FIR graph)

[English](runtime-flags.md) | [Tiếng Việt](runtime-flags.vi.md)

Environment variables read by `nim_inference`, `formers-cli` (native route and
`--fir` graph route), the Vulkan driver in `vendor/bybylang`, and `firlift`.
Every optimization the Colab T4 notebook (`notebooks/deepseek_colab_vulkan.ipynb`)
sets is the default, so a plain run needs none of them. Set a flag only to turn
an optimization off, force one choice, or get diagnostics.

## Notebook settings and their defaults

| Flag | Notebook value | Default | Where |
| --- | --- | --- | --- |
| `BYBY_MODEL_MMAP` | `1` | `1` | weights mmapped from `model.nimq` |
| `BYBY_VULKAN_DEVICE_LOCAL` | `auto` | `auto` | weights in VRAM on a dGPU |
| `BYBY_VULKAN_TENSOR_CORES` | `1` | `1` | cooperative-matrix prefill GEMMs |
| `BYBY_UBATCH` | `512` | `512` | prefill micro-batch rows (native and FIR) |
| `BYBY_PREFILL_CB_MS` | `300` (native), `450` (lift) | `300` | native prefill command-buffer target; the FIR graph does not read it |
| `BYBY_VULKAN_Q4_SPLIT_QKV` | `0` / `1` (A/B) | `auto` | autotunes fused vs split QKV |
| `BYBY_FIR_Q4_BACKEND` | `micro` | `micro` | FIR Q4 nodes run their lifted micro-kernels |
| `FIRLIFT_Q4_GEMM` | `tensor_core` | `auto` | `auto` lifts Tensor Core GEMMs when the trace records Vulkan cooperative matrix or Metal simdgroup matrix support |
| `FIRLIFT_Q4_GEMV` | `word_parallel` / `auto` | `auto` | `auto` lifts the subgroup GEMV micro-kernel on Vulkan (faster than `word_parallel`) |
| `BYBY_FIR_PROBE_STEPS` | `prefill:2,decode:1` | every step | steps the recorder keeps (lift only) |
| `BYBY_FIR_PROBE_DATA` | `0` | `1` under most `firlift --run` routes | upload payload pipe (lift only). Codestral uses metadata only. |
| `BYBY_FIR_PROFILE` | `1` | `0` | diagnostics only |

Decode overlap needs no flag: the FIR decode step is recorded once and replayed
(`BYBY_FIR_DECODE_REPLAY`), and steps that are encoded are submitted in chunks
without waiting (`BYBY_FIR_DECODE_CB_NODES`, `BYBY_VULKAN_COMMIT_WAIT`).

## FIR graph runtime (`formers-cli --fir`)

| Flag | Default | Effect |
| --- | --- | --- |
| `BYBY_FIR_DECODE_REPLAY` | `1` | Vulkan and Metal: record the decode step once at the first step (Vulkan: one reusable command buffer; Metal: the step's dispatches, see `BYBY_METAL_REPLAY_ICB`), then each step only writes token/position into a host-visible buffer, replays it, and reads logits from a mapped buffer (no host encode). The embedding row is gathered on the GPU. Supported exact session nodes (RMSNorm, vector/activation, FP4 Metal GEMM and NT GEMM) also replay their native session operation. `0` encodes every step. Turns itself off with probe, `BYBY_FIR_PROFILE=2` or `BYBY_FIR_Q4_BACKEND=native`, and when the step uploads, allocates or frees buffers while recording. |
| `BYBY_FIR_FP4_GEMV_TUNE` | `1` | At load, recognise the lifted FP4-KM decode GEMV kernels, build the other threadgroup-width variants (and the simdgroup one on Metal), check them against the lifted kernel and time them on the resident weights; every node of that kernel uses the fastest. `0` keeps the lifted kernels. `BYBY_FIR_Q4_COOP_VERBOSE=1` with `--verbose` prints the timings. |
| `BYBY_METAL_REPLAY_ICB` | `1` | Metal replay of a recorded session (`BYBY_FIR_DECODE_REPLAY`): unset: indirect command buffer on Apple GPUs only (on Intel GPUs a plain encoder replay was ~25% faster); `1` forces an indirect command buffer (built once, run with `executeCommandsInBuffer`) when the GPU supports it; `0` re-issues the recorded dispatches on a plain encoder in one loop (no graph walk either way). |
| `BYBY_METAL_REPLAY_LOG` | unset | Set: log each Metal session recording (commands, pipelines, buffers) and why one was not kept. |
| `BYBY_FIR_DECODE_REPLAY_VERBOSE` | `0` | `1` prints `[fir] decode replay on/off: ...` (also printed with `BYBY_FIR_PROFILE`). |
| `BYBY_FIR_DECODE_CB_NODES` | `64` | Encoded decode step (replay off): submit without waiting every this many nodes (the first after 16) so the GPU starts while the host encodes. `0`: one command buffer. |
| `BYBY_FIR_Q4_BACKEND` | `micro` | `micro`: lifted Q4 micro-kernels. `native`: diagnostic switch through `sessionMatmulQ4` (not allowed with Tensor Core nodes). |
| `BYBY_FIR_SESSION_DIRECT` | `1` | `0` stages exact session replays through scratch buffers; `1` replays supported session operations directly on graph buffers. |
| `BYBY_FIR_WARM_RELOAD` | `0` | `1` reloads model weights for the warm run instead of resetting lifted graph state. |
| `BYBY_METAL_SIMDGROUP_MATRIX` | `1` | `0` hides MSL simdgroup matrix support, so Metal traces record `device="plain"` and no Tensor Core micro-kernel is built. |
| `BYBY_FIR_Q4_GEMV_TUNE` | `1` | Vulkan: at load, time every rows-per-subgroup (1, 2, 4) and workgroup width (64, 128, 256) per GEMV shape on the resident weights and keep the fastest. `0` keeps the lifted choice. |
| `BYBY_FIR_Q4_GEMV_ROWS` | lifted | Force GEMV output rows per subgroup (1, 2 or 4). |
| `BYBY_FIR_Q4_GEMV_THREADS` | lifted | Force GEMV workgroup width (64, 128 or 256). Both forced: no tune. |
| `BYBY_FIR_Q4_GEMV_REDUCE` | `auto` | `subgroup` (needs 32-wide subgroups), `shared`, or `auto` by device. |
| `BYBY_FIR_Q4_COOP_ROWS` | per dispatch | Force the Tensor Core row tile (16, 32 or 64). |
| `BYBY_FIR_Q4_COOP_WG_PER_UNIT` | `2` | Workgroups per SM/CU the Tensor Core tile choice aims for. |
| `BYBY_FIR_Q4_COOP_VERBOSE` | `0` | `1` prints Tensor Core tile and `[fir] Q4 GEMV tune ...` lines (needs `--verbose`). |
| `BYBY_FIR_Q4_MAP_LOG` | `0` | `1` logs resident Q4 mapping (native Q4 backend). |
| `BYBY_FIR_RELOC` | `1` | Metal arena graphs: share one pipeline per kernel. `0`: one per node. |
| `BYBY_FIR_PROFILE` | `0` | `1`: mean host encode / GPU wait / logits read per decode step. `2`: ends the session after every node and reports per-kernel wall time (slow; diagnostics only). |
| `BYBY_UBATCH` | `512` | Caps the rows of a prompt phase run (phases with per-row logits excepted). `0`: no cap. |
| `BYBY_PREFILL_MIN_ROWS` | `5` | Codestral CLI: a prompt prefix or final ubatch tail of at least this many tokens uses batched prefill; fewer use decode. `1`: batch every tail of 2+ tokens. Codestral FIR records the native batch widths, including the tail. |

## Native inference (`nim_inference`, `formers-cli` native route)

| Flag | Default | Effect |
| --- | --- | --- |
| `BYBY_MODEL_MMAP` | `1` | mmap `model.nimq` instead of reading it into RAM. |
| `BYBY_PREFAULT` | `1` | One sequential pass over the mapped file to fill the page cache. |
| `BYBY_EMBED_RAM` | `1` | Keep an untied embedding in RAM. |
| `BYBY_LM_HEAD` | `q4` | Quantize an fp32 `lm_head` to Q4 for the tuned GEMV. `fp32` keeps it. |
| `BYBY_UBATCH` | `512` | Prefill micro-batch rows. `0`: uncapped on paced devices. Codestral native and `firlift --run`: a prefix up to this many tokens runs as one resident prefill pass; longer prefixes carry state between passes. Codestral range 2..4096, `0` = 4096. The CLI reserves the final prompt token for decode/logits. Compile/autotune time is reported separately from throughput. |
| `BYBY_PREFILL_CB_MS` | `300` | Target GPU time of one prefill command buffer. |
| `BYBY_PREFILL_MAX_ROWS` | `512` | Upper bound of rows in one prefill chunk. |
| `BYBY_PREFILL_SESSION_MB` | `1024` | Scratch budget of one prefill session. |
| `BYBY_PREFILL_LAYERS_PER_CB` | `4` | Layers per prefill command buffer on unpaced devices. |
| `BYBY_FLASH_PREFILL` | `1` | Flash-attention prefill. |
| `BYBY_DECODE_CB_MS` | `1500` | Target GPU time of one decode command buffer. |
| `BYBY_DECODE_LAYERS_PER_CB` | auto | Layers per decode command buffer (`0`: one per token). |
| `BYBY_DECODE_FIRST_CB` | `0` | Commit after this many layers first so the GPU starts early. |
| `BYBY_DECODE_REPLAY` | `1` | Codestral (`infer_lora`, `formers-cli`), Vulkan and Metal: after one encoded step the decode step is recorded once (with and without the head) and resubmitted per token; the host only writes the embedding row and reads the logits. Recordings are dropped on GPU model preparation, state reset and LoRA merge. Off under the FIR recorder, `BYBY_DECODE_SPLIT`, `BYBY_G_DIAGNOSTICS` and one-dispatch. `0` encodes every step. `BYBY_DECODE_REPLAY_VERBOSE=1` repeats the on/off line. |
| `BYBY_PREFILL_F16_TILE` | `1` | Codestral resident prefill on Vulkan: the FP16 Linear++ sidecar GEMMs (G blocks: `attn.qkvProj`, `attn.outProj`, `fuseGate`) run the tile GEMM micro-kernel with FP16 weights, on cooperative matrices when the device has them (register tiles otherwise), after its once-per-process parity check. `0`: the FP16 micro-VM GEMM. |
| `BYBY_DECODE_FUSE_RESIDUAL` | `1` | Codestral decode: the out_proj GEMV of an R block adds into the residual itself (one dispatch and one barrier less per layer) when its autotuned FP4-KM decode kernel is a subgroup-row (`sg`) or decode (`dec`) kernel and the projection has no bias. `0`: separate GEMV and add. |
| `BYBY_GEMV_SUBGROUP` | `auto` | FP4-KM decode GEMV autotune also tries the subgroup-row micro-kernels: one 32-lane subgroup per 1-8 output rows, two weight words per lane in flight, `subgroupAdd`/`simd_sum` reduction, LUT in threadgroup or constant memory. `auto`: Vulkan with fixed 32-lane subgroups that is not Intel (NVIDIA T4), and Apple GPUs on Metal. `1`: try them on any GPU (a pipeline the driver runs narrower than 32 lanes fails the parity check and is dropped); `0`: never. |
| `BYBY_MERGE_QKV_ATTN_SESSION` | `1` | Merge QKV and attention into one session. |
| `BYBY_MERGE_ALL_LAYERS_SESSION` | `1` | Keep all layers of a token in one session. |
| `BYBY_HEAD_IN_SESSION` | `1` | Run final norm and `lm_head` in the decode session. |
| `BYBY_VULKAN_Q4_SPLIT_QKV` | `auto` | Time fused vs split QKV once per device/shape; `0` fused, `1` split. |
| `BYBY_PREFILL_HEAD_COPIES` | `0` | Keep per-head KV copies on Metal. |
| `BYBY_SPARSITY_PCT` | `0.0` | FFN sparsity experiment (off). |
| `BYBY_GPU_MAX_YIELD_MS` | `8.0` | Cap of the adaptive yield between tokens. |
| `BYBY_PROGRESS` | `auto` | Live tok/s line; `1` forces it in Colab, `0` disables it. |
| `BYBY_PREFILL_PROFILE` | `-1` | Layer whose prefill ops run one per command buffer (with `BYBY_SESSION_LOG`). |
| `BYBY_SESSION_LOG` | unset | Log prefill session timings. |
| `BYBY_DECODE_PROFILE` | `0` | Profile the first N decode tokens. |
| `BYBY_DEBUG_PATH` | `0` | Log which GPU/CPU path each op takes. |
| `BYBY_NO_VERIFY` | unset | Skip the resident KV verification. |
| `BYBY_DISABLE_KV_RESIDENT` | `0` | `1` turns resident KV off (debugging). |
| `BYBY_DISABLE_SESSION_RMSNORM` | unset | Use the old RMSNorm path. |
| `BYBY_TRY_ATTN_DECODE_GQA` | unset | Retry the fused GQA decode attention after it was disabled. |
| `BYBY_LEGACY_DECODE_GC` | unset | Old periodic GC during decode. |
| `BYBY_GPU_SAMPLE`, `BYBY_GPU_SAMPLE_AHEAD`, `BYBY_GRAPH` | Metal only | GPU sampling pipe and decode graph replay on Metal. |

## Mamba-3 decode (`formers-cli` mamba3 route)

| Flag | Default | Effect |
| --- | --- | --- |
| `BYBY_M3_GPU_SAMPLE` | `1` | Greedy decode (`--temp 0`, no repeat penalty) picks tokens on the GPU: embedding gather, stack, `lm_head`, banned-token mask and argmax per step, with the ids in a GPU ring. `0`: host sampler. |
| `BYBY_DECODE_AHEAD` | `16` | Decode steps per command-buffer batch of the GPU sampler (1..62). Steps past a stop token are computed and dropped. |
| `BYBY_M3_DECODE_ASYNC` | `1` | Metal: two batches in flight, the next one encoded while the GPU runs the previous one. `0`: wait for each batch. |
| `BYBY_M3_FUSE` | `1` | One decode row: the output projection adds into the residual and the down projection applies the SiLU gate and the residual itself. `0`: separate dispatches. |
| `BYBY_GEMV_DECODE` | `1` | Autotune also tries the short-K decode GEMVs (no LUT copy loop, 4-16 rows per threadgroup). `0`: older kernels only. |
| `BYBY_M3_PROFILE` | unset | `1`: per-batch GPU ms/token of the GPU sampler and per-step timings of the host path. |

## Vulkan driver (`vendor/bybylang/backends/vulkan`)

| Flag | Default | Effect |
| --- | --- | --- |
| `BYBY_VULKAN_DEVICE` | best scored | Device index; by default a discrete GPU (NVIDIA first) wins. |
| `BYBY_VULKAN_DEVICE_LOCAL` | `auto` | Resident weights in VRAM on a dGPU (`auto` or `1`); never on an iGPU. Anything else keeps host-visible memory. |
| `BYBY_VULKAN_SCRATCH` | `auto` | Scratch in VRAM: `auto` (dGPU), `device` (always), `host` (never). |
| `BYBY_VULKAN_TENSOR_CORES` | `1` | Cooperative-matrix GEMMs. On a dGPU every prefill GEMM must use them; `force` does that on any device; `tune` times them against scalar kernels; `0` turns them off. |
| `BYBY_VULKAN_SPLIT_K` | `auto` | Split-K slices of tiled Tensor Core FP32/Q4 GEMMs (`1`: off, `<n>`: fixed). The FP4-KM tile GEMM micro-kernel does not split K. |
| `BYBY_VULKAN_SPLIT_K_VERBOSE` | `0` | Log every split-K choice. |
| `BYBY_VULKAN_MMAP_IMPORT` | `1` | Import mmapped weights as host memory on an iGPU. |
| `BYBY_VULKAN_POOL_MB` | `2048` | Cap of the reusable buffer pool. |
| `BYBY_VULKAN_DISPATCH_CACHE` | `1` | Cache JIT pipelines and dummy bindings; prewarm them in parallel. |
| `BYBY_VULKAN_COMMIT_WAIT` | `0` | `1`: mid-session commits wait for the GPU (the old behaviour). Commits never wait during a recording. |
| `BYBY_VULKAN_MAX_TG` | device | Cap workgroup width. |
| `BYBY_VULKAN_JIT_DIR` | temp dir | SPIR-V and pipeline cache directory. |
| `BYBY_VULKAN_JIT_VERBOSE` | `0` | Log JIT compiles and dispatch bindings. |
| `BYBY_JIT_DUMP_SOURCE` | `0` | Print the GLSL of every JIT kernel. |
| `BYBY_VULKAN_TIMESTAMPS` | `0` | Device timestamps per session. The FP4 decode GEMV autotune turns them on for its own timing runs, so its choice compares GPU time instead of submit latency. |
| `BYBY_VULKAN_STRICT` | `0` | Fail on a binding offset past the buffer instead of clamping. |
| `BYBY_Q4_TUNED` | `1` | Q4 GEMV/GEMM autotune (`0`: default kernels). |
| `BYBY_GPU_AUTOTUNE` | `1` | Global switch of GPU autotuning. |
| `BYBY_Q4_SUBGROUP` | `1` | Allow subgroup Q4 kernels. |
| `BYBY_Q4_TUNE_VERBOSE` | `0` | Log autotune parity rejects. |
| `BYBY_FLASH_ATTENTION` | `auto` | Flash attention kind; `scalar` forces FP32. |
| `BYBY_FUSE_ROPE` | `1` | Fuse RoPE into the KV append. |
| `BYBY_SSD` | `auto` | Mamba2 SSD kernel. Vulkan Codestral records and replays the selected native SSD pipeline. `scalar` selects FP32 tiles, and `0` uses the micro-VM scan. When SSD runs, native prefill skips scan autotuning. |

## Lifting (`firlift --run`)

| Flag | Default | Effect |
| --- | --- | --- |
| `FIRLIFT_Q4_GEMV` | `auto` | One-row Q4 matmul kernel. `auto` on Vulkan: subgroup micro-kernel (the native decode algorithm), else word-parallel. Also `scalar`, `scalar_fma`, `parallel`, `parallel_fma`, `simd`, `word_parallel`, `subgroup`. |
| `FIRLIFT_Q4_GEMM` | `auto` | Multi-row Q4 matmul kernel. `auto`: Tensor Core micro-kernel when the trace records Vulkan `device=coopmatrix` or Metal `device=simdmatrix`, unless `FIRLIFT_GENERIC` is set; otherwise follows the generic kernel choice. Also `scalar`, `scalar_fma`, `parallel`, `parallel_fma`, `tiled`, `matrix`, `tensor_core`/`coop` (Vulkan or Metal trace required). |
| `FIRLIFT_FP4_GEMM` | `auto` | FP4-KM prefill GEMM. `auto`: Tensor Core micro-kernel for a supported shape on a recorded Vulkan `coopmatrix` or Metal `simdmatrix` device, unless `FIRLIFT_GENERIC` is set. `coop`/`tensor_core` forces it (Vulkan or Metal trace; requires K divisible by 64 and N >= 16); `generic` selects lane-per-column kernels. |
| `FIRLIFT_SLICE_REPORT` | `0` | Any value except `0` prints why the earliest replayed ops were selected for the lift slice. |
| `FIRLIFT_SLICE_REPORT_N` | `40` | Maximum number of slice-reason lines printed when `FIRLIFT_SLICE_REPORT` is enabled. |
| `FIRLIFT_Q4_WORD_MAX_K` | `4096` | Largest K of the word-parallel GEMV in `auto`. |
| `FIRLIFT_Q4_LANES`, `FIRLIFT_TG` | per kernel | Lanes / threadgroup width of the parallel kernels. |
| `FIRLIFT_GEMM_TILE`, `FIRLIFT_GEMM_REG` | `64x128`, `8x8` | f32 GEMM tile and register block. |
| `FIRLIFT_SIMD` | by backend | `1`/`0`: allow simdgroup reductions (off for Vulkan traces). |
| `FIRLIFT_GENERIC` | unset | Lane-per-output kernels any VM backend runs. |
| `FIRLIFT_TARGET` | unset | `apple`: the graph may use `simdgroup_matrix`. |
| `FIRLIFT_FLASH` | tiled | `row`: lift prefill attention as the row-parallel decode kernel instead of the tiled flash kernel. |
| `FIRLIFT_NOFUSE` | unset | Do not fuse activation / vector / RoPE ops into their neighbours. |
| `FIRLIFT_NOSIDE` | unset | Do not merge independent consecutive decode kernels into one launch. |
| `FIRLIFT_NOREUSE` | unset | Give every op output its own buffer (no temporary reuse). |
| `BYBY_FIR_PROBE_STEPS` | every step | Steps the recorder keeps, for example `prefill:2,decode:1`. |
| `BYBY_FIR_PROBE_DATA` | `1` under `--run` | Send exact upload payloads down the data pipe (needed for unnamed uploads such as LUTs). `0` when every weight is named in `model.nimq`. |

Codestral Vulkan uses `BYBY_MEMORY_LIMIT_MB=10000` and `BYBY_CODESTRAL_TOKEN_IDS=1`
by default, including when recording with `firlift --run`. Set either variable
explicitly to override its default.

Trace headers record `device="coopmatrix"` for Vulkan cooperative matrix, `device="simdmatrix"` for Metal simdgroup matrix, or `device="plain"` otherwise. Tensor Core micro-kernels are selected automatically for Vulkan with cooperative matrix or Metal on Apple GPU family 7+; other devices use generic kernels.

The trace records the backend and device features of the run, so a graph lifted
on a T4 targets its Tensor Cores and a graph lifted on a device without
cooperative matrix does not.

FP4-KM prefill GEMMs (Codestral and Mamba 3) always run the tile GEMM micro-kernel
(`OP_TILE_GEMM`, `vendor/bybylang/micro_kernel/tile_gemm.nim`); there is no hand-written
FP4-KM shader and no flag to pick one. The JIT lowers the same program to Vulkan
cooperative matrix, Metal simdgroup matrix (Apple GPU family 7+), or register tiles
on other GPUs, and the CPU VM runs it directly. A lifted FIR FP4 Tensor Core node falls
back to the register-tiled lowering at load when the device has no matrix units, and
`sessionFp4GemmMetal` / `sessionFp4GemmTensorCore` nodes of existing graphs replay
through the same micro-kernel. The cooperative-matrix lowering loads the next K slice into
registers while the tensor cores work on the current one; the Codestral prefill uses it only after
its parity check passes on the device (`[vulkan] tile GEMM micro-kernel check ...`). The same
micro-kernel also takes FP16 weights (format 1, Vulkan only; see `BYBY_PREFILL_F16_TILE`).
The former FP8 (E4M3) FP4-KM route, the
`BYBY_PREFILL_FP4_TILED` switch and `BYBY_FP4_SG_COMPILE_CHECK` are gone.
