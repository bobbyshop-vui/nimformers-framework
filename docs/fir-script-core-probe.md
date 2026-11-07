# FIR from the core: rebuilding graphs by probing the native dispatch stream

## Summary

The idea is to record what the existing native inference cores actually launch while they run one
decode step (and optionally a prefill chunk). A model graph and a `model.fir` can then be rebuilt
from that record, instead of tracing Python forward code or as a check next to it. The study finds:

- Every GPU launch inside a session passes through two places that are easy to instrument:
  `gSessionEncoder` in `metal_shim.m` and `dispatchGeneric` in `vulkan_driver.nim`. Above them, the
  Nim session API in `gpubackend.nim` carries the op identity and the shape scalars. CPU work has
  no encoder, so its chokepoints are the `gpubackend` session procs plus `cpu_backend.vmRun`.
- At those points we can see the program identity, the micro-VM bytecode (for JIT programs), grid
  and threadgroup sizes, buffer handles with offsets and byte sizes, scalar binds, the order of
  dispatches, and the barriers. Tensor names, logical shapes, dtypes, semantic op names (for
  generated programs) and the layer loop are not visible there.
- Most of the Mamba-2, Mamba-3 and FIR-graph work is micro-VM bytecode (`VmProgram`). A recorded
  program can therefore be emitted directly as a FIR kernel block. Llama (`nim_inference`) is
  built mostly from fixed library kernels (Q4 matmul, RMSNorm, RoPE, GQA attention, KV scatter).
  These lift only to `ext` ops, and most of them have no FIR runtime counterpart yet.
- Dataflow is the hard part. FIR, Mamba-2 and Mamba-3 programs all address one resident arena
  through constant offsets baked into the bytecode, so def-use chains built from buffer handles are
  empty. Exact read and write sets come from replaying each recorded program on an instrumented CPU
  interpreter. That replay is deterministic and never needs weight values.
- Implemented approach: a two-tier recorder (a semantic tier at `gpubackend` and a launch tier at
  the encoders), opt-in with `BYBY_FIR_PROBE`, weight-name registration at upload time and an
  offline Nim lifter that emits `model.fir` directly.

## Implementation status

The recorder is in `vendor/bybylang/probe.nim` and writes `fprobe 2`; the lifter,
kernel library and builder are in `other/formers.n/firlift/`. The format, the
canonical `op` record, the graph/core lift modes and the current limits are in
[`FIR from a core trace`](fir-script-lift.md). The analysis below is the
inventory that the implementation uses; `fgir 1` is no longer the lifter's
output format.

## 1. Where every launch goes

### 1.1 The layers a dispatch crosses

| Layer | Location | What passes through |
|---|---|---|
| L3 model encoders | `gpu_layers.nim:93` `encodeVmKernel`, `kernelfortest/fp4_gemv_gpu.nim:116` `encodeFp4Gemv`, `test/codestral/mamba2_fused_step.nim:136` `encodeVm` and `:321` `encodeLayerV2`, `test/mamba3/m3_gpu_runtime.nim:73-79` `encode`, `other/formers.n/src/formers/ir/fir_graph.bybylang:197` `fgEncodeMicro` and `:210` `fgEncodeNode` | kernel name, role/tag (`m3resident_<dims>_norm`), the `VmProgram` in scope, the arena and offset handles |
| L2 session API | `vendor/bybylang/gpubackend.nim:926` `sessionBegin`, `:2246` `sessionEnd`, `:1191` `sessionMatmul`, `:2623` `sessionMatmulQ4`, `:1383` `sessionRmsnormEnc`, `:1459` `sessionAttentionDecodeGqaEnc`, `:1665` `sessionKvScatterEnc`, `:1763` `sessionRope`, `:1835` `sessionCopyInto`, `:993` `sessionJitMicroVmEnc`, `:949` `sessionMicroVmEnc`, `:2339` `sessionRead`, `:1003` `sessionUpload` | semantic op, M/K/N, rows/cols, eps, head counts, the `SessionHandle`s (`:882`), `ResidentWeightHandle` k/n/groupSize (`:2481`) |
| L1 encode bridge | `vendor/bybylang/micro_kernel/session_vm_dispatch.nim:33-63` (Metal `sessionPipeline`, `sessionBindBuffer`, `sessionBindInt/Float`, `sessionDispatch`) and `:94-107` (Vulkan wrappers); shared `:114` `sessionEncPipeline`, `:121` `sessionEncBindVkOrMetal`, `:133` `sessionEncDispatch` | pipeline name, slot index, handle, offset, nThreads, tgSize, tgBytes |
| L0 Metal | `vendor/bybylang/backends/metal/metal_shim.m:766` `metal_session_begin` creates `gSessionEncoder` (`:773`); `:2574` `metal_enc_pipeline` (looks up `gJitPipelines` at `:2572`, then the library cache `getPipeline` at `:111`/`:38`); `:2595` bind buffer; `:2607` bind int; `:2631` and `:2656` dispatch; `:2681` barrier; `:2819` `metal_jit_compile` | every `setComputePipelineState`/`setBuffer`/`setBytes`/`dispatch*` on the session encoder, including the about 20 fixed `*_enc` kernels that skip L1 (`metal_shim.h:141-288`) |
| L0 Vulkan | `vendor/bybylang/backends/vulkan/vulkan_driver.nim:1722` `dispatchGeneric` (the only `vkCmdDispatch` inside a session, `:1747`); fed by `:3225` `vulkanSessionJitMultiBufEnc`, `:3286` `vulkanKernelDispatch`, `:1760` `vulkanSessionMicroVmEnc`, `:1896` `vulkanSessionJitMicroVmEnc` | shader name, dense buffer array with byte offsets, push constants, group counts, local size, shared floats; `VulkanBufferHandle.sizeBytes` (`:1700`) |
| CPU | no encoder: `gpubackend` `of gbCpu:` branches run at once (for example `:1863-1868` copy); micro programs run through `vendor/bybylang/backends/cpu/cpu_backend.nim:361` `vmRun` → `cpu_vm.c:128` `nimf_cpu_vm_run` | words, mem length, nThreads, tgFloats, tgSize; the address of `cpuData[0]` stands in for a buffer handle |

Dispatches outside a session have their own command buffers and do not touch these encoders. On
Metal these are the standalone `metal_vecop`/`metal_matmul`/... helpers (`metal_shim.m:186-721`,
each with `commit` and `waitUntilCompleted`). On Vulkan they are the one-shot dispatches at
`vulkan_driver.nim:913` and `:1075`. The inference decode paths studied here encode inside
sessions. The non-session helpers matter only for the Llama Mode 1 fallback and for training.

The recommended hook points:

- **Metal: one hook.** When the probe is on, wrap `gSessionEncoder` at `metal_session_begin` in a
  logging `NSProxy` that forwards to the real encoder. All 55 dispatch sites in `metal_shim.m` go
  through that object, so the hook catches the fixed `*_enc` kernels as well as JIT pipelines
  without editing each one.
- **Vulkan: one hook.** Log inside `dispatchGeneric` (`:1722`). It already has the shader name, the
  dense buffer array, the offsets, the push constants and the group counts.
  `BYBY_VULKAN_JIT_VERBOSE` (`:3270`) already prints part of this record.
- **Semantic tier: about 30 procs in `gpubackend.nim`.** Each emits one `op` record with its
  arguments before it delegates to the backend. This tier also covers the CPU, where L0 does not
  exist.

### 1.2 What is available at the chokepoints

| Item | Metal L0 | Vulkan L0 | L2 session API | CPU |
|---|---|---|---|---|
| Program identity | pipeline name: `jit_<key>[_tag]` for JIT programs (`jit_vm.nim:730`), library function name for fixed kernels | shader name `jit_vk_<key>` (`jit_vm.nim:787`); SPIR-V stem includes a source hash (`vulkan_driver.nim:1792-1797`) | op name (`sessionRmsnormEnc`, ...) | none at `vmRun`; words are passed |
| Bytecode | not at L0. Recovered from the key: `gWordsKey` maps words to key (`jit_vm.nim:582`, `:654-673`); a reverse table is a few lines | same | same | `code` argument |
| Generated source | `CompiledEntry.metalSource` (`jit_vm.nim:40-47`), `BYBY_JIT_DUMP_SOURCE` | `CompiledEntry.glslSource` | – | – |
| Grid / threadgroup | yes (after the max-TG clamp at `metal_shim.m:2662-2665`) | group counts and local size | caller values | nThreads, tgSize |
| Buffers | `id<MTLBuffer>` pointer, offset, `.length` | handle, byte offset, `sizeBytes` | `SessionHandle`s and offsets | `cpuData` pointer and len |
| Scalars | `setBytes` payloads | push constants | typed arguments | arguments |
| Threadgroup memory | `setThreadgroupMemoryLength` | `sharedFloats` | tgBytes | tgFloats |
| Order and hazards | encoder order plus `memoryBarrier` (`session_vm_dispatch.nim:45-54`) | recorded barriers (`:1747-1756`) | call order | program order |
| Host transfers | upload (`metal_shim.m:779`), `metal_buffer_read` (`:1877`) | `vulkanSessionRead` (`:1581`), copy (`:1599`) | `sessionUpload`, `sessionRead`, `sessionCopyInto` | `cpuData` copies |

Not available at any chokepoint, and supplied instead by loaders, the runtime or analysis:

- **Tensor names.** Buffers are anonymous. Names exist only at load or upload sites:
  `fgUploadWeight` (`fir_graph.bybylang:67-92`, from the FIR tensor template), `gLinCache` keyed by
  the checkpoint name (`gpu_layers.nim:286`), and `loadLinear(..., "model.layers.<l>.self_attn.q_proj")`
  (`test/deepseek-inference/nim_inference.nim:2798-2804`).
- **Logical shapes and dtypes.** Byte sizes only. Q4/FP4 packing hides the element count;
  `ResidentWeightHandle` keeps `k, n, groupSize` (`gpubackend.nim:2481-2490`) and
  `Fp4LinCacheGpu` keeps `rows, cols` (`fp4_gemv_gpu.nim:34-37`).
- **Semantic identity of generated programs.** `jit_<hash>` says nothing. The readable tag
  (`m3resident_..._scan`, FIR `node.op`) is appended to the key (`jit_vm.nim:660-662`). It is a
  namespace hint only; the code itself says it does not prove two programs are equal
  (`jit_vm.nim:757`).
- **Sub-buffer extents.** Arena programs bind the whole arena (FIR binds it four times:
  `fir_graph.bybylang:208`). What a program reads or writes is encoded in its immediates.
- **Loop structure.** The layer loop is unrolled into the stream (`fir_graph.bybylang:271-283`,
  `m3_gpu_runtime.nim:180-208`, `lora.nim` per-block encoders).

## 2. Probe recorder design

### 2.1 Switches and scope

- `BYBY_FIR_PROBE=<path>` turns the recorder on; unset means off. With the recorder off, every
  hook is one global boolean test, so numerics and timings are unchanged.
- `BYBY_FIR_PROBE_STEPS=decode:2` records two consecutive decode steps (two are needed to detect
  state; see 2.4). `prefill:T,decode:2` also records one prefill chunk.
- Recording never changes the dispatch sequence. The recorder only reads arguments. Bytecode is
  copied from tables that already exist (`gWordsKey`, the `CompiledEntry` fields). Readbacks are
  logged and not added.
- The output is a trace artifact for the lifter, not a run log. It holds structure only: names,
  sizes, programs and scalar arguments. It never holds buffer contents. Uploads are logged with
  their size and the registered name, never their data. The only upload payloads that matter for
  structure are the token and position scalars (`fir_graph.bybylang:262-267`), and they are logged
  by role.
- Fast paths that fold the whole model into one launch must be off for probing: `BYBY_ONE_DISPATCH`
  (Codestral `lora.nim:38`, `:1308`; Mamba-3 `m3_gpu_runtime.nim:284-287`, which already refuses
  the provider path with one-dispatch on). The Llama merged-session Mode 3 is fine: it is still
  one launch per op.

### 2.2 Record format

The implemented format is `fprobe 2`; `docs/fir-script-lift.md` lists the
records. The semantic tier emits one canonical `op` per launch with an ATen tag,
named ports (`buffer@byteOffset:byteCount`), scalar attributes and
`impl=session:…` / `impl=micro:<hash>` / `impl=ext:…`. The launch tier emits
`disp` with the program and geometry. Weight names are registered at upload;
buffer byte extents are recorded for CPU, Metal, Vulkan and q4 buffers. Floats
are IEEE bit patterns. The writer never accepts tensor payloads.

### 2.3 Dataflow by buffer def-use, refined by address sets

1. **Buffer-level chains.** Each dispatch has bound slots, each with a (handle, offset) pair. With
   the kernel signature table (which slot a fixed library kernel writes), this gives the def-use
   chains for Llama-style code, where every activation has its own scratch handle
   (`nim_inference.nim:915-917`).
2. **Address-level chains for arena programs.** FIR graph, Mamba-2 V2 and Mamba-3 resident programs
   share one arena handle, so step 1 sees only "arena → arena". For each recorded `VmProgram`, the
   lifter runs the program once on an instrumented copy of the CPU interpreter
   (`vm_interp.nim:99` `runGrid`, or `cpu_vm.c`) over a zero arena of the logged size. It records
   the exact set of float addresses loaded and stored. Address arithmetic depends only on
   `tid`/`lid`/`tgid` and immediates, so the sets hold for any data. The one exception is indirect
   loads (`U_LOAD_U32`/`ULoadMeta` index tables, `session_vm.nim:253-307`). For those, the replay
   binds the logged scalar inputs (position, token), and the lifter marks the access as
   data-dependent. The sets are compressed into intervals, then joined into chains: an interval
   written by d_i and read by d_j with no write in between.
3. **Interval → value.** Each maximal interval written by one dispatch becomes a value. Intervals
   never written in the step are inputs (weights, constants, state). A FIR arena laid out by the
   compiler has disjoint buffers, so intervals line up with FIR `buf` declarations; the round-trip
   test in 5.2 checks this.

### 2.4 State, KV caches and inputs

With two consecutive decode steps recorded:

| Pattern | Class |
|---|---|
| Written before any device op in the step, by an upload or copy of role scalars | input role (`token`, `position`) |
| Never written by any dispatch, uploaded once at load | weight or constant; named by `probeName` |
| Written in the step and read later in the same step only | activation (FIR `buf` without `per_layer`, or a per-layer scratch) |
| Read in step t before it is written in step t, and written in step t | recurrent state (Mamba `ssm_state`, `conv_state`, Linear++ carry) |
| Written at an address that moves by a fixed stride between step t and t+1, and read over a growing range | KV cache, indexed by `position`; stride and row count give the FIR cache shape |
| Copied to a host-visible buffer and read back | output role `logits` |

The Mamba-3 runtime makes the state range explicit (`stateStart`/`stateEnd`, `m3_gpu_runtime.nim:49`,
reset at `:149-150`). That range is a ground-truth check for the classifier.

### 2.5 Repeated layers

1. **Canonical program class.** Hash the words with the address immediates masked out (operand
   words of `LOAD_F32`/`STORE_F32`/`U_*` with an offset, `ULoadConstF`, and loop and base constants
   that `fkAddress` produced). Two programs are in the same class when they differ only in those
   immediates.
2. **Periodic segment.** Map the stream of one step to a string of `(class, grid, tg, bind
   pattern)` tokens. Find the longest run `prefix · body^L · suffix` with L ≥ 2 (Z-function or
   suffix array, linear in the stream length). Mamba-3 with `dynamicOffsets` produces
   bit-identical words per layer (`m3_gpu_runtime.nim:177-179`); only the bound `layerOffsets`
   handle changes. FIR graph programs differ per layer only in immediates (`fir_graph.bybylang:121-128`
   builds one program per layer).
3. **Per-layer strides.** For each address immediate that varies across the L copies, fit
   `base + layer*stride`. A constant stride equal to a buffer's element count marks a `per_layer`
   buffer or a `{i}` weight. A stride of 0 marks a shared buffer.
4. **Fold.** Emit the body once as a `repeat layers` region with `sym layers = L`. Weights become
   templates `h.{i}.…` when their registered names differ only in the layer number. This is the
   same test `extract_repeats` uses on scopes (`python/formers/fir/repeat.py:18-40`), so the lifter
   can reuse it on synthetic scopes `layer.<i>`.
5. **Mixed stacks.** Jamba/Codestral hybrids interleave R and A blocks (`lora.nim:1552-1556`). The
   periodic search then finds two bodies, or a body with period equal to the pattern length.
   FIR v1 represents only one contiguous stack (`repeat.py:21-22`), so mixed patterns need either
   unrolling or a future multi-stack form.

### 2.6 Naming and shapes

- Weights: `probeName` at the upload sites gives the checkpoint name. The upload size and the
  loader dims give the shape: dims from `FirBuffer.dims`, from `ResidentWeightHandle.k/n`, from
  `Fp4LinCacheGpu.rows/cols`, or from the Llama config.
- Activations: the interval length gives the element count. A rank-2 shape is chosen when a
  consumer is a matmul with known K/N (from the `op` record). Otherwise the value stays rank 1,
  which FIR arena buffers accept.
- Ops: a library launch takes its name from its `op` record. A micro program takes the tag, the
  `M3ProgramRequest.role` (`m3_gpu_runtime.nim:12-19`), or the FIR `node.op`, whichever is present.
  With none of these it falls back to `kernel_<canon>`.

## 3. Mapping the record to FIR

### 3.1 What lifts directly

- **Micro programs are already FIR kernels.** A FIR `kernel` block assembles into a `VmProgram`
  through `fkEmit` (`fir_kernel.bybylang:151`) and `firAssembleKernel` (`:198`). The inverse
  disassembler maps each opcode back to its mnemonic. It rewrites address immediates as
  `buffer + expression` using the interval map from 2.3 and the layer stride from 2.5, and gives the
  dispatch geometry (`threads`, `tg`, `tgbytes`) from the record. Opcodes that `fkSignature`
  (`:136`) does not accept, or whose immediates cannot be explained by known buffers, keep a raw
  `words` escape (needs a small FIR extension) or block emission.
- **Resident externals.** `fir.embedding`, `fir.linear` (f32 or q4) and `fir.linear_t` are the
  graph runner's executable externals (`fir_graph.bybylang:210-242`). Recorded
  `sessionMatmul`/`sessionMatmulQ4`/`sessionMatmulNT` and embedding copies map onto them one to
  one.
- **Copies.** `sessionCopyInto` between arena ranges is a FIR copy micro kernel, or is absorbed when
  it only stages an external's input (the graph runner's own pattern, `fir_graph.bybylang:233-242`).

### 3.2 Lifting to the graph IR for passes

The former native `fgir 1` graph is not the lifter's output. The current lifter
emits `model.fir` directly (see `docs/fir-script-lift.md`); the mapping below is
the historical design for a pass-oriented intermediate.

- Each value from 2.3 becomes an `FsValue`. A weight value carries `weight=(name shape dtype)`, a
  state value carries `state=`, and an input value carries `role=`.
- Each `op` record becomes a node with the matching op (`rmsnorm`, `linear`, `rotary`, `embedding`,
  `softmax`).
- Each opaque micro program becomes a `kernel` node. Its attrs are the canonical hash and the
  disassembled body; scope is `layers.<i>` for repeated segments.
- Optional pattern matching turns common micro programs back into pointwise trees, norms and
  scans, so the passes can fuse across them. The pattern library is `patterns.py`/`refusion.py` in
  `python/formers/fir/`, and the native port planned in the design doc reuses it.

Without pattern matching the lifted graph is correct but coarse: passes treat kernel nodes as
barriers. Emission back to FIR works either way.

### 3.3 What cannot be lifted

| Item | Where | Why | What the lifter does |
|---|---|---|---|
| Fixed Metal library kernels (RMSNorm, RoPE, GQA decode attention, flash prefill, KV scatter, softmax, Q4 GEMV/GEMM) | `metal_shim.h:141-288`, Metal kernels in `backends/metal/kernels`, `q4_dynamic.inc` | MSL source only, no bytecode | `ext` node named after the op, with shapes from the `op` record; executable only after a FIR ext or kernel exists for it |
| Fixed GLSL shaders (`micro_vm` interpreter, Q4, coop GEMM) | `backends/vulkan/kernels` | GLSL only | same |
| Hand-written Nim loops on the host | for example `embed.lookupBatch` before upload (`m3_gpu_runtime.nim:298`, `nim_inference.nim:2592`), CPU sampling | never reaches a chokepoint | appears as an upload of an unnamed buffer with a role hint; the lifter needs the embedding role to be registered |
| MPS matmul | `metal_matmul_mps*` (`metal_shim.m:1039`) | opaque framework call | `ext` linear |
| Data-dependent control | retries and fallbacks (`forwardStepJambaGpuRetry`, `lora.nim:1912`; Llama Mode 3 → 2 → 1, `nim_inference.nim:1910-1928`) | the recorded path is one branch | record with the mode pinned; the lifter notes the branch |

## 4. Per-implementation feasibility

| Implementation | Dispatch stream for one decode step | Expressible in current FIR | Effort to probe and lift |
|---|---|---|---|
| FIR graph runner (GPT-2 125M, TinyLlama in `models/`) | `sessionBegin`; 2 scalar copies; per node: micro launch or copy → linear → copy (`fir_graph.bybylang:251-303`); logits copy; `sessionEnd`; one read | 100%: the graph came from FIR. A round-trip test: recover FIR, compare with `model.fir` up to buffer naming, then compare logits | 3–5 days including the disassembler; the reference for every other row |
| Mamba-3 resident (`m3_gpu_runtime.nim`) | Per layer: `norm, prepare, scan, carry, norm2, mlp, residual` micro programs (`:180-208`), plus FP4 linears and bias/LoRA kernels from `encodeLinear` (`:222`); final norm; head. Metal or Vulkan only (`:94`) | The micro roles are already FIR via `firInstallMambaProvider` (`fir_exec.bybylang:256-269`), which is a ground truth. FP4 GEMV is micro bytecode (`fp4_gemv_gpu.nim:116-125`); it lifts as a kernel but has no FIR q4/fp4 linear form yet. Gated and Linear++ blocks (`:198-204`) lift as kernels. Estimate: 90% of launches lift as kernels, 60–70% map to existing FIR node kinds | 1–1.5 weeks; repeat folding is easy because programs are identical per layer |
| Mamba-2 / Codestral V2 (`test/codestral/lora.nim:1552`, `mamba2_fused_step.nim:321`) | Per layer: LayerNorm or RMSNorm (L2 library op), FP4 in-proj GEMV (micro), conv, scan and gate micro programs through `encodeVm`, FP4 out-proj, `sessionVecOp` residuals; hybrid A layers through `encodeAttnLayer` (`gpu_layers.nim:703`) | R layers: mostly micro (80%+). Norms and vec-ops are library calls with simple FIR equivalents. A layers depend on the attention library kernels | 1.5–2 weeks; needs `BYBY_ONE_DISPATCH=0` and the two-body repeat for hybrid patterns |
| Llama (`nim_inference`, `test/deepseek-inference/nim_inference.nim:2304`, `:2587`) | Mode 3: per layer `sessionRmsnormEnc`, 3× `sessionMatmulQ4`, 2× `sessionRope`, `sessionKvScatterEnc`, `sessionAttentionDecodeGqaEnc`, `sessionMatmulQ4` o-proj, vec-op residual, norm, 3× Q4 FFN, activation, vec-op | Structure: 100% recoverable at the semantic tier (L2 calls carry shapes). Executable as FIR today: only the linears and the embedding. RoPE, GQA attention, KV scatter and SiLU-gate have no direct FIR runtime op; the TinyLlama FIR graph writes them as kernels | 1 week to record and lift to `fgir`; emitting runnable FIR means substituting kernels from a library of FIR snippets (the TinyLlama `model.fir` ones), which works as a pattern table |
| GPT-2 / TinyLlama in FIR (`models/gpt2-125m/model.fir`, TinyLlama) | as in the FIR graph row | already FIR | covered by the round-trip test |
| Jamba / LoRA paths (`m3_formers_infer.nim:6-20`, `mergeInUploadStreamingJamba`, `mergeResidentLoraInplace`) | With LoRA merged (the default policy), the same as the base stream. Unmerged LoRA adds a down/up pair per linear (`m3_gpu_runtime.nim:211-220`) | Merged: as the base model. Unmerged: extra kernel nodes, foldable, but FIR has no adapter concept | record merged only; unmerged is diagnostic |

## 5. Native core, validation, risks

### 5.1 Recorder output as `fgir` input

The removed native core read `fgir 1` through `fsReadFgir` and ran its pass
pipeline plus FIR emission. The current lifter skips that intermediate and writes
FIR directly:

```
fprobe 1 ──lift──> fgir 1 (kernel/ext nodes, weights, states, scopes layers.<i>)
                     └─> fs passes (dead code, refusion on matched trees, repeat analysis)
                           └─> FIR emission ──> model.fir
```

- Scopes `layers.<i>` let `fsAnalyzeRepeats` derive the repeat (§3.1 of the design doc: scope is
  the source of truth), so the lifter does not need its own `repeat` lines.
- Opaque micro programs need an fgir node kind that carries bytecode or disassembled FIR text.
  fgir rejects unknown fields, so this is a decision under the design doc's versioning rules: either
  a registered `kernel` op with string attrs in `fgir 1`, or a new feature flag.
- The recorder feeds the lifter and the GPT-2/TinyLlama round trips; the
  native fgir corpus was removed with the native core.

### 5.2 Validation on synthetic tiny weights

1. **Structural round trip.** Record the FIR graph runner on a tiny synthetic FIR model, lift it,
   emit FIR, and compare with the source FIR after canonical renaming. Kernel bodies must be
   word-identical after reassembly.
2. **Numeric replay.** Load the recovered FIR with the same synthetic weights through `firNewGraph`
   on the CPU backend and compare logits with the original over several sequences. The tolerance is
   zero for the round trip (same programs, same order) and the existing test tolerances otherwise.
3. **Cross-core check.** For Mamba-3, replay the recovered FIR through `firInstallMambaProvider`
   and compare dispatch streams. The recorded and replayed streams must be equal up to pipeline
   names.
4. Synthetic weights are produced in-process, as `test_fir_script.bybylang:150-190` and
   `checkpoint(graph, 23)` in `python/tests/test_gpt2_125m.py:99-115` already do. Real checkpoints
   are not needed for any of these steps.

### 5.3 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Caches change the stream | First-token compile (`maybeJit` hot threshold, `jit_vm.nim:715-757`), the Metal binary archive (`metal_shim.m:2775`) and the Vulkan SPIR-V cache move work out of the step, and kernel names can differ between runs | Record from step 2 on; identify programs by words/canonical hash, not by name |
| Programs that are not decode-shaped | Programs are compiled per chunk length `T` (`m3_gpu_runtime.nim:158`); prefill streams have different geometry | Record decode and prefill as separate graphs; symbolic `T` waits for `fgir 2` symbolic dims |
| Fused and fast paths | One-dispatch megakernels, MPS, flash attention (`BYBY_FLASH_ATTENTION`, `metal_shim.m:1744`) and Q4 tuned GEMM (`:1231`) are selected by device and env | Pin env for probing; record the selection in the `backend` line |
| Device-dependent geometry | `sessionMaxTgThreads` clamps and `threadExecutionWidth` (`session_vm_dispatch.nim:56-71`) change grids | Store geometry per backend; the canonical class ignores grid size |
| Readbacks and host logic | Llama Mode 1 reads back between layers; sampling and retries live on the host | Probe the single-session modes; host steps are recorded as `read`/`upload` and lifted to roles |
| Indirect addressing | KV position and token index make some address sets data-dependent | Replay with the logged scalars at two positions; the stride between them gives the cache layout |
| CPU buffer identity | `SessionHandle.cpuData` is a value `seq`; `sessionCopyInto` on CPU can `setLen` (`gpubackend.nim:1865`) and move it; `sessionMicroVmEnc` on CPU runs on a copy of `mem` (`:961-963`) | On CPU, record at `cpu_backend.vmRun` with arena-relative addresses, and key buffers by their FIR role; the graph runner calls `vmRun` in place (`fir_graph.bybylang:202`) |
| Nondeterminism | Atomics (`emitUAtomicAdd`) and multi-threadgroup reductions reorder float sums | Structure is unaffected; numeric replay uses tolerances where atomics appear |
| Trace content | A trace must never contain weights | The format has no data field for `buf`; uploads log only size and name |

## 6. Staged plan

P0–P2 are implemented: the recorder, CPU/Metal/Vulkan launch records, the
disassembler and the GPT-2 graph round trip. P3–P6 remain for the other
families. The table is the original plan, with statuses updated.

| Stage | Work | Effort | Exit check |
|---|---|---|---|
| P0 | Zero-change experiment on the Python VM (below) | 0.5–1 day | Rebuilt node list and buffer intervals equal those of the exported tiny GPT-2 `model.fir` |
| P1 | `fprobe 2` writer; semantic tier in `gpubackend.nim`; CPU hook at `cpu_backend.vmRun` and in `fgEncodeNode`; `probeName` at `fgUploadWeight` | done | Recorder on/off gives bit-identical logits on the tiny FIR graph on CPU |
| P2 | Lifter: CPU address replay, interval dataflow, state classifier, canonical hashing, periodic fold, disassembler to FIR kernel text | 4–6 days | Structural round trip (5.2-1) for tiny GPT-2 and TinyLlama graphs on CPU |
| P3 | Metal `NSProxy` encoder hook and Vulkan `dispatchGeneric` hook; `words` recovery through a reverse `gWordsKey` table | 2–3 days | The same FIR recovered from Metal and from CPU for the tiny graph |
| P4 | Mamba-3 resident path: role tags, `layerOffsets`, FP4 linear as kernel, state range check | 4–5 days | Recovered FIR replays through `firInstallMambaProvider` with an identical stream (5.2-3) on synthetic tiny weights |
| P5 | `fgir` output and native core integration (kernel node kind decision) | 3–4 days, after N1 of the native plan | `fsReadFgir` accepts recorded graphs; repeat analysis matches the lifter's fold |
| P6 | Codestral V2 and Llama: library-op table, hybrid two-body repeat, FIR snippet substitution for RoPE/attention/KV | 1.5–2 weeks | Llama stream lifted to fgir with 100% named ops; runnable FIR for the Mamba-2 R stack |

### 6.1 Minimal first experiment (no real weights, no file changes)

The Python FIR VM (`python/formers/fir/vm.py`) is the reference executor of the same FIR graphs.
`FirModel.run_node` (`vm.py:612`) calls `run_program(p, mem)` (`vm.py:405`) once per micro
dispatch, and ext nodes read and write `self.mem` at `binding(...)` offsets (`vm.py:583`). A
scratch script under `/tmp` can do the experiment without touching repository files:

1. Build the tiny GPT-2 graph exactly as `test_tiny_config_decodes_like_straight_line_gpt2` does
   (`fir.trace(GPT2LMHead(TINY))`, `checkpoint(graph, 23)`, `fir.export`), then construct
   `fir.FirModel(path, weights=params)`.
2. Monkeypatch `vm.run_program` and `FirModel.run_node` at runtime. Wrap `mem` in an `array`
   subclass, or diff snapshots, to log each dispatch's read and written address sets, its program
   words and geometry, and the ext op with its weight name. Run two decode steps.
3. Apply the lifter algorithm from 2.3–2.5 to the log: intervals, def-use chains, state classes,
   periodic fold with layer strides.
4. Compare with the exported `model.fir`: the buffer list (offsets and sizes, `per_layer` flags,
   `key_cache`/`value_cache` as position-indexed state), node order, and `sym layers`.

This checks the analysis (address-set dataflow, state detection, folding) on an exact,
deterministic executor before any native hook exists. P1 then repeats the same experiment on the
production CPU VM (`cpu_backend.vmRun`) with the native recorder.

## 7. Biggest unknowns

1. **Disassembly coverage.** Whether every opcode and immediate pattern that the Mamba-2/3 builders
   emit (`U_*` ops, `VTG`, grid barriers, `ULoadMeta`) maps back to FIR kernel syntax, or whether
   FIR needs a raw-bytecode escape.
2. **Library kernels.** How much of Llama and the attention layers can be represented in FIR at
   all. This depends on a FIR attention/RoPE/KV op set, or on a stable snippet library that the
   lifter substitutes by pattern.
3. **Address-set replay cost and fidelity** on full-size programs (a 1.5B Mamba-3 step on the
   CPU interpreter, with structure only). It may need sampling of threads, or static affine
   analysis instead of replay.
4. **Mixed-stack repeats.** Jamba/Codestral hybrid patterns do not fit FIR v1's single repeat
   region.
5. **The fgir representation of opaque kernels**, and whether passes can work on them usefully
   without semantic pattern matching.
6. **Dynamic shapes.** Prefill per-T programs need symbolic dims (`fgir 2`) before one recovered
   graph can serve every chunk length.
