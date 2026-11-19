## gpu_layers.nim — the remaining layers of the 100% GPU infer path, 1 command buffer / token.
##
## Every kernel here is micro-VM BYTECODE (fp4_gemv_vm.nim) going through jit_vm
## (lower -> JIT compile once / shape -> cache). NO MSL embedded in Nim.
##
##   * GpuLin         : a block-A Linear (resident fp4 or fp32) + bias + LoRA
##   * GpuAttnLayer   : block A = LN1 -> QKV -> Linear++ step -> out -> +res
##                                -> LN2 -> FC1+APFLU -> FC2 -> +res
##
## Everything is encoded into the open session; no sessionRead in between.

import std/[tables, os, strutils, math, monotimes, times, algorithm, strformat]
import backend
import vendor/bybylang/gpubackend as byby
import session_vm
import session_vm_dispatch
import jit_vm
import fp4_gemv_vm
import fp4_gemv_gpu
import vendor/bybylang/backends/vulkan/vulkan_driver as vkDriver
when defined(macosx):
  import vendor/bybylang/backends/metal/metal_backend as metalDriver
import quant      # QuantTensor, QuantKind
import nimformer   # Linear (weightQ/useQuant/weight/bias/loraA..)

# ── Separate JIT cache for the infer kernels (compiled RIGHT AWAY, hotThreshold=0) ─────
var gVmCache = jit_vm.initJitCache(jit_vm.JitConfig(
  hotThreshold: 0, enableMetalJit: true, enableVulkanJit: true,
  enableCpuJit: false, maxCacheEntries: 1024))

# Alias used by fp4_gemv_gpu / encode path
var gJitV2* = gVmCache

proc vmKernel*(p: VmProgram; tag: string): string =
  ## Lowers bytecode -> Metal (jit_vm) -> compile -> pipeline name. Called OUTSIDE
  ## a session (during prepare) so compilation is not in the hot path.
  ## Programs using U-* ops have NO interpreter path -> a compile failure raises right away
  ## (instead of falling back to micro_vm_kernel and silently giving garbage).
  let kn = jit_vm.dispatchProgram(gVmCache, p, tag)
  if kn.len == 0 or kn == "micro_vm_kernel":
    raise newException(CatchableError,
      "vmKernel[" & tag & "]: Metal JIT FAILED (the program uses U-* ops, there is no " &
      "interpreter path). Set BYBY_JIT_DUMP_SOURCE=1 to see the source + compile error.")
  kn

proc vmKernelForBe*(p: VmProgram; tag: string; be: byby.GpuBackend): string =
  ## Picks Metal MSL or Vulkan GLSL by backend.
  if be == byby.gbVulkan:
    result = jit_vm.kernelNameFor(gJitV2, p, tag)
  else:
    result = jit_vm.dispatchProgramNoSession(gJitV2, p, tag)
  if result.len == 0 or result == "micro_vm_kernel":
    raise newException(CatchableError, "vmKernelForBe[" & tag & "]: JIT failed")

var gVkPrewarmedEntries = 0
proc prewarmVulkanJit*(be: byby.GpuBackend) =
  ## Vulkan: compiles every micro-VM shader built so far (kernelNameFor) that has no SPIR-V on
  ## disk yet, several glslc processes at a time, so a cold cache does not compile them one by one
  ## inside the first dispatches. Call after building the kernels of a phase, inside its setup time.
  if be != byby.gbVulkan or gJitV2.entries.len == gVkPrewarmedEntries: return
  var jobs: seq[tuple[name, source: string]]
  for e in gJitV2.entries.values:
    if e.glslSource.len > 0 and e.kernelName.len > 0: jobs.add (e.kernelName, e.glslSource)
  byby.sessionJitPrewarm(be, jobs)
  gVkPrewarmedEntries = gJitV2.entries.len

# ── dummy buffers: JIT kernels always declare buffer(1..3) -> bind the empty slots ─
# Per backend: a single global gDummy gave nil handles when the process switched Metal↔Vulkan
# (bisect/sweep) because gDummyReady was set only once.
var gDummyMetal: byby.SessionHandle
var gDummyVulkan: byby.SessionHandle
var gDummyMetalReady = false
var gDummyVulkanReady = false
proc ensureDummy*(be: byby.GpuBackend) =
  if be == byby.gbVulkan:
    if not gDummyVulkanReady:
      gDummyVulkan = byby.sessionUploadPrivate(be, newSeq[float32](64))
      gDummyVulkanReady = true
  else:
    if not gDummyMetalReady:
      gDummyMetal = byby.sessionUploadPrivate(be, newSeq[float32](64))
      gDummyMetalReady = true

proc gDummyFor*(be: byby.GpuBackend): byby.SessionHandle =
  ensureDummy(be)
  if be == byby.gbVulkan: gDummyVulkan else: gDummyMetal

# Backward-compat alias: most call sites still say `gDummy` after ensureDummy(be).
template gDummy*: byby.SessionHandle =
  ## Prefer gDummyFor(be). This resolves to Metal dummy if ready, else Vulkan.
  if gDummyMetalReady: gDummyMetal
  elif gDummyVulkanReady: gDummyVulkan
  else: gDummyMetal

proc bindOr(idx: int; h: byby.SessionHandle; offElems: int) =
  ## Shared API: routes by h.kind.
  let isVk = h.kind == byby.gbVulkan
  let dummy = gDummyFor(if isVk: byby.gbVulkan else: byby.gbMetal)
  let mbuf = if h.metalHandle != nil: h.metalHandle
             elif dummy.metalHandle != nil: dummy.metalHandle
             else: nil
  discard sessionEncBindVkOrMetal(isVk, idx, mbuf, h.vulkanHandle, offElems)

proc encodeVmKernel*(kname: string; nThreads, tgSize, tgBytes: int;
                     memOut: byby.SessionHandle; memOff: int;
                     wU8, constF, inX: byby.SessionHandle; inOff: int) =
  ## API chung Metal+Vulkan. Layout:
  ##   Metal slots 1..4 = mem / u8 / const / in
  ##   GLSL binding 0..4 = MemF32 / MemU8 / Lut / Input / MemU32
  let isVk = memOut.kind == byby.gbVulkan
  let glslSrc = if isVk: jit_vm.cachedGlslSource(gJitV2, kname) else: ""
  if isVk and glslSrc.len == 0:
    raise newException(CatchableError,
      "encodeVmKernel[" & kname & "]: no GLSL source (call kernelNameFor first)")
  if not sessionEncPipeline(isVk, kname, glslSrc):
    raise newException(CatchableError, "encodeVmKernel: cannot bind pipeline " & kname)
  if isVk:
    bindOr(0, memOut, memOff)
    bindOr(1, wU8, 0)
    bindOr(2, constF, 0)
    bindOr(3, inX, inOff)
    bindOr(4, memOut, memOff)
  else:
    bindOr(1, memOut, memOff)
    bindOr(2, wU8, 0)
    bindOr(3, constF, 0)
    bindOr(4, inX, inOff)
  discard sessionEncTgBytes(isVk, 0, max(tgBytes, 16))
  let maxTG = sessionEncMaxTg(isVk)
  if maxTG > 0 and tgSize > maxTG:
    raise newException(CatchableError,
      "encodeVmKernel[" & kname & "]: needs " & $tgSize & " tg but max=" & $maxTG)
  if not sessionEncDispatch(isVk, nThreads, tgSize, tgBytes):
    raise newException(CatchableError, "encodeVmKernel[" & kname & "]: dispatch failed")

# ── shared kernels (shape independent) ───────────────────────────
var gKnBias = ""
var gKnBiasApflu = ""
var gEluPlus1Kn = ""
var gEluPlus1BackwardKn = ""
var gGpuKernelsReady = false
var gGpuKernelsBe: byby.GpuBackend
proc ensureGpuKernels*(be: byby.GpuBackend) =
  ## Compiles the bias kernels FOR THE RIGHT backend. Switching Metal<->Vulkan needs a recompile
  ## (different kernel names + GLSL/MSL caches).
  if gGpuKernelsReady and gGpuKernelsBe == be: return
  ensureDummy(be)
  gKnBias = vmKernelForBe(buildBiasActProgram(false), "bias", be)
  gKnBiasApflu = vmKernelForBe(buildBiasActProgram(true), "bias_apflu", be)
  # Elu kernels are prepared lazily; clear so next encode recompiles for this be.
  gEluPlus1Kn = ""
  gEluPlus1BackwardKn = ""
  gGpuKernelsBe = be
  gGpuKernelsReady = true

proc encodeBias*(hMem: byby.SessionHandle; memOff: int; hBias: byby.SessionHandle;
                 n: int; apflu = false) =
  ## mem[memOff ..< memOff+n] = act(mem + bias)
  let tg = min(n, 256)
  encodeVmKernel(if apflu: gKnBiasApflu else: gKnBias, n, tg, 16,
                 hMem, memOff, gDummy, hBias, gDummy, 0)

proc encodeEluPlus1*(ctx: Backend; hInput, hOutput: byby.SessionHandle; n: int) =
  ## GPU JIT feature map for Linear++ training: phi(x)=ELU(x)+1.
  let be = ctx.kind.toByby()
  ensureGpuKernels(be)
  if gEluPlus1Kn.len == 0:
    gEluPlus1Kn = vmKernelForBe(buildEluPlus1Program(max(n, 1)), "lp_elu_plus1", be)
  encodeVmKernel(gEluPlus1Kn, n, min(n, 256), 16,
                 hOutput, 0, gDummy, gDummy, hInput, 0)

proc encodeEluPlus1Backward*(ctx: Backend; hInput, hOutput: byby.SessionHandle; n: int) =
  ## GPU JIT feature-map derivative. hInput layout is [x[n] | dPhi[n]].
  let be = ctx.kind.toByby()
  ensureGpuKernels(be)
  if gEluPlus1BackwardKn.len == 0:
    gEluPlus1BackwardKn = vmKernelForBe(buildEluPlus1BackwardProgram(max(n, 1)), "lp_elu_plus1_bwd", be)
  encodeVmKernel(gEluPlus1BackwardKn, n, min(n, 256), 16,
                 hOutput, 0, gDummy, gDummy, hInput, 0)

# ── Linear GPU (fp4 resident | fp32) + bias (LoRA is merged into the base) ──────────────────────
type
  GpuLin* = object
    name*: string
    isFp4*: bool
    wF32*: byby.SessionHandle
    knF32*: string
    K*, N*: int
    biasH*: byby.SessionHandle
    hasBias*: bool
    hasLora*: bool
    isInt4*: bool
    int4WBuf*: byby.SessionHandle
    isF16*: bool                       ## FP32 master stored as IEEE halves on the GPU (BYBY_LINEARPP_DTYPE=fp16)
    wF16*: byby.SessionHandle          ## [N, K] halves, read through the u8 slot
    knF16Scalar*: string               ## decode kernel for an input offset that is not 16-byte aligned (built lazily)
    ready*: bool

type
  Int4LinCacheGpu = object
    wBuf, scZpBuf: byby.SessionHandle
    knName: string
    N*, groupSize*: int
    ready: bool

var gInt4CacheGpu = initTable[string, Int4LinCacheGpu]()

proc uploadInt4WeightOnce(ctx: Backend; name: string; qt: QuantTensor; K, N: int) =
  if name in gInt4CacheGpu and gInt4CacheGpu[name].ready: return
  let be = ctx.kind.toByby()
  let groupSize = qt.groupSize
  doAssert int4AsymSupported(K, groupSize),
    "uploadInt4WeightOnce " & name & ": K=" & $K & " groupSize=" & $groupSize & " not supported on the GPU"
  var c: Int4LinCacheGpu
  c.wBuf = sessionUploadBytesShared(be, qt.bytes)
  c.knName = vmKernelForBe(buildInt4AsymGemvProgram(K, N, groupSize),
    "int4_gemv_" & $K & "x" & $N & "_g" & $groupSize, be)
  c.N = N; c.groupSize = groupSize
  # scale + zero_point back to back (matching buildInt4AsymGemvProgram: [scale(N*nG) | zp(N*nG)])
  var scZp = newSeq[float32](qt.scale.len + qt.zero_point.len)
  for i in 0 ..< qt.scale.len: scZp[i] = qt.scale[i]
  for i in 0 ..< qt.zero_point.len: scZp[qt.scale.len + i] = qt.zero_point[i]
  c.scZpBuf = byby.sessionUploadPrivate(be, scZp)
  c.ready = true
  gInt4CacheGpu[name] = c

proc markFp4Lin*(ctx: Backend; name: string; bias: seq[float32]; K, N: int): GpuLin
proc uploadBiasOrRaise(name: string; bias: seq[float32]; N: int): seq[float32]

var gLinCache* = initTable[string, GpuLin]()   ## the single source of truth: EVERY Linear (R and A, every kind) goes through here.

type Fp4Plan = object
  kn, label: string
  tg, rows, bytes: int   ## rows: output rows per threadgroup (the dispatch covers N / rows groups)
  simd: bool
  kind: string           ## "" (legacy, wide, simd), "dec" or "sg" (kernels with a fused-residual form)
  kernelRows: int        ## rows per threadgroup ("dec") or per subgroup ("sg")
  unroll: int
  lutConst: bool

proc fp4SubgroupCandidates(be: byby.GpuBackend): bool =
  ## BYBY_GEMV_SUBGROUP = auto | 1 | 0: try the subgroup-row FP4 decode GEMV. auto: on Vulkan
  ## devices whose compute subgroups are 32 lanes (NVIDIA, AMD wave32) and on Apple GPUs. The
  ## tuner still checks every candidate against the baseline kernel before timing it.
  let setting = getEnv("BYBY_GEMV_SUBGROUP", "auto")
  if setting == "0": return false
  if setting == "1": return true
  if be == byby.gbVulkan:
    let caps = vkDriver.vulkanDeviceCaps()
    return caps.subgroupSize == Fp4SubgroupWidth and caps.properties.vendorID != 0x8086'u32
  when defined(macosx):
    if be == byby.gbMetal: return metalDriver.metalIsAppleGpu()
  false

var gFp4Plans = initTable[string, Fp4Plan]()

proc tuneFp4Plan(ctx: Backend; c: Fp4LinCacheGpu; batch: int): Fp4Plan =
  ## Tune once per backend/shape using real packed weights. Candidate outputs
  ## must match the existing kernel before their timings are considered.
  let be = ctx.kind.toByby()
  let key = $be & "_" & $c.cols & "_" & $c.rows & "_" & $batch
  if key in gFp4Plans: return gFp4Plans[key]
  let baseline = if batch == 1:
    Fp4Plan(kn: c.jitName, label: "legacy", tg: GemvTg,
      rows: c.rowsPerTg, bytes: c.tgBytes)
    else: Fp4Plan(kn: c.m8JitName, label: "legacy-M8", tg: GemvTg,
      rows: 8, bytes: (Fp4KmLutLen + 8 * GemvTg) * 4)
  var candidates = @[baseline]
  let setting = getEnv("BYBY_GEMV_TG", "max")
  var maxThreads, maxShared: int
  if be == byby.gbVulkan:
    let caps = vkDriver.vulkanDeviceCaps()
    maxThreads = vkDriver.vulkanMaxTgThreads()
    maxShared = caps.maxSharedBytes
    when defined(macosx):
      # Intel/MoltenVK advertises 1024 but its translated eight-row FP4
      # shader can time out there. 512 is GPU-parity tested on Iris 640.
      # Keep NVIDIA/Linux device limits intact.
      if batch > 1 and caps.properties.vendorID == 0x8086'u32:
        maxThreads = min(maxThreads, 512)
  else:
    when defined(macosx):
      maxThreads = int(metalDriver.metalDeviceMaxTgThreads())
      maxShared = int(metalDriver.metalDeviceMaxTgBytes())
  let rows = if batch == 1: 4 else: 8
  for tg in fp4KmTgCandidates(c.cols, rows, maxThreads, maxShared, setting):
    if batch == 1 and c.rows mod rows != 0: continue
    let prog = if batch == 1: buildFp4KmGemvTunedProgram(c.cols, c.rows, tg, rows)
               else: buildFp4KmGemmTunedProgram(c.cols, c.rows, tg, rows)
    let label = "wide" & $rows & "-tg" & $tg
    try:
      let kn = vmKernelForBe(prog, "fp4_" & label & "_" & key, be)
      candidates.add Fp4Plan(kn: kn, label: label, tg: tg, rows: rows,
        bytes: (Fp4KmLutLen + rows * tg) * 4)
    except CatchableError as e:
      stderr.writeLine "[gpu-tune] skip " & label & ": " & e.msg
  if batch == 1 and be != byby.gbVulkan:
    # Metal simdgroup reduction (simd_sum + per-simdgroup partials), the Llama GEMV structure.
    for tg in fp4KmTgCandidates(c.cols, rows, maxThreads, maxShared, setting):
      if c.rows mod rows != 0: continue
      let label = "simd4-tg" & $tg
      try:
        let kn = vmKernelForBe(buildFp4KmGemvSimdProgram(c.cols, c.rows, tg, rows),
          "fp4_" & label & "_" & key, be)
        candidates.add Fp4Plan(kn: kn, label: label, tg: tg, rows: rows,
          bytes: (Fp4KmLutLen + rows * tg) * 4, simd: true)
      except CatchableError as e:
        stderr.writeLine "[gpu-tune] skip " & label & ": " & e.msg
  if batch == 1 and getEnv("BYBY_GEMV_DECODE", "1") != "0":
    # Decode kernels: no LUT copy loop and up to 16 output rows per threadgroup. Include both
    # narrow and wider workgroups; device timing chooses the best K-loop/occupancy tradeoff.
    let simdModes = if be == byby.gbVulkan: @[false] else: @[true]
    # Let autotune test wider tiles too: fewer serial K-loop iterations can win
    # for the large Mamba projections, while parity/timing still reject shapes
    # that hurt occupancy on a given device.
    for tg in [16, 32, 64, 128, 256]:
      if tg > maxThreads: continue
      for dRows in [4, 8, 16]:
        if not fp4KmGemvDecodeSupported(c.cols, c.rows, tg, dRows) or
           fp4KmGemvDecodeTgBytes(dRows, tg) > maxShared: continue
        for simd in simdModes:
          let label = "dec" & $dRows & (if simd: "s" else: "t") & "-tg" & $tg
          try:
            let kn = vmKernelForBe(buildFp4KmGemvDecodeProgram(c.cols, c.rows, tg, dRows, simd),
              "fp4_" & label & "_" & key, be)
            candidates.add Fp4Plan(kn: kn, label: label, tg: tg, rows: dRows,
              bytes: fp4KmGemvDecodeTgBytes(dRows, tg), simd: simd, kind: "dec", kernelRows: dRows)
          except CatchableError as e:
            stderr.writeLine "[gpu-tune] skip " & label & ": " & e.msg
  if batch == 1 and fp4SubgroupCandidates(be):
    # One 32-lane subgroup per few rows, subgroup sums only (the Llama/DeepSeek decode GEMV).
    for tg in [64, 128, 256]:
      if tg > maxThreads: continue
      for sRows in [2, 4]:
        for lutConst in [false, true]:
          let unroll = 2
          if not fp4KmGemvSubgroupSupported(c.cols, c.rows, tg, sRows, unroll): continue
          let label = "sg" & $sRows & "u" & $unroll & (if lutConst: "c" else: "t") & "-tg" & $tg
          try:
            let kn = vmKernelForBe(buildFp4KmGemvSubgroupProgram(c.cols, c.rows, tg, sRows, unroll,
              lutConst), "fp4_" & label & "_" & key, be)
            candidates.add Fp4Plan(kn: kn, label: label, tg: tg,
              rows: fp4KmGemvSubgroupRowsPerTg(tg, sRows), bytes: fp4KmGemvSubgroupTgBytes(lutConst),
              simd: true, kind: "sg", kernelRows: sRows, unroll: unroll, lutConst: lutConst)
          except CatchableError as e:
            stderr.writeLine "[gpu-tune] skip " & label & ": " & e.msg
  if candidates.len == 1: return baseline
  if getEnv("BYBY_GPU_AUTOTUNE", "1") == "0":
    return baseline
  prewarmVulkanJit(be)
  var data = newSeq[float32](batch * c.cols)
  for i in 0 ..< data.len: data[i] = float32((i * 17 mod 101) - 50) / 64
  var hX = byby.sessionUpload(be, data)
  var hY = byby.sessionAllocScratch(be, batch * c.rows)
  # Device timestamps on Vulkan too: one decode GEMV is a fraction of a millisecond, far below the
  # submit + fence latency a wall clock would add to every sample.
  let gpuTimed = byby.sessionSetGpuTiming(be, true)
  defer:
    discard byby.sessionSetGpuTiming(be, false)
    byby.sessionFree(be, hX); byby.sessionFree(be, hY)
  # Several dispatches per command buffer; without device time, enough of them that the submit
  # latency does not decide the ranking.
  let weightBytes = c.rows * (c.cols div 2 + c.cols div 32)
  let reps =
    if batch != 1: 1
    elif gpuTimed: clamp(48 * 1024 * 1024 div max(weightBytes, 1), 3, 16)
    else: clamp(512 * 1024 * 1024 div max(weightBytes, 1), 8, 64)
  proc dispatch(plan: Fp4Plan; n = 1): float =
    let start = getMonoTime()
    let g0 = byby.sessionGpuSeconds(be)
    if not byby.sessionBegin(be): raise newException(CatchableError, "FP4 tuning sessionBegin failed")
    try:
      let groups = if batch == 1: c.rows div plan.rows else: c.rows
      for _ in 0 ..< n:
        encodeVmKernel(plan.kn, groups * plan.tg, plan.tg, plan.bytes,
          hY, 0, c.wBuf, c.lutBuf, hX, 0)
      if not byby.sessionEnd(be):
        raise newException(CatchableError, "FP4 tuning GPU failed: " & byby.sessionGetLastError(be))
    except CatchableError:
      discard byby.sessionEnd(be)
      discard byby.sessionRecoverQueue(be)
      raise
    let gpu = byby.sessionGpuSeconds(be) - g0
    if gpu > 0: gpu * 1e3 / float(n)
    else: float((getMonoTime() - start).inNanoseconds) / 1e6 / float(n)
  var reference: seq[float32]
  var valid = newSeq[bool](candidates.len)
  var timings = newSeq[seq[float]](candidates.len)
  for i, plan in candidates:
    try:
      discard dispatch(plan)
    except CatchableError as e:
      if i == 0: raise
      stderr.writeLine "[gpu-tune] skip " & plan.label & ": " & e.msg
      continue
    let values = byby.sessionRead(be, hY, batch * c.rows)
    if i == 0:
      reference = values
      valid[i] = true
    else:
      var err, mag: float32
      valid[i] = true
      for j, v in values:
        if classify(v) in {fcNan, fcInf, fcNegInf}: valid[i] = false
        err = max(err, abs(v - reference[j])); mag = max(mag, abs(reference[j]))
      valid[i] = valid[i] and err <= max(1e-5'f32, mag * 1e-4'f32)
  # Interleave candidates to reduce bias from GPU clock/thermal changes.
  for round in 0 ..< 3:
    for j in 0 ..< candidates.len:
      let i = if round mod 2 == 0: j else: candidates.high - j
      if valid[i]: timings[i].add dispatch(candidates[i], reps)
  var best = 0
  var bestMs = high(float)
  for i in 0 ..< candidates.len:
    if not valid[i]: continue
    if setting notin ["auto", "0", ""] and i == 0 and valid.len > 1:
      var hasValidRequested = false
      for j in 1 ..< valid.len: hasValidRequested = hasValidRequested or valid[j]
      if hasValidRequested: continue
    timings[i].sort()
    if timings[i][1] < bestMs:
      best = i; bestMs = timings[i][1]
  result = candidates[best]
  gFp4Plans[key] = result
  let gbps = float(weightBytes) / (bestMs * 1e-3) / 1e9
  echo &"[gpu-tune] FP4 {c.cols}x{c.rows} M={batch}: {result.label}, {bestMs:.3f}ms" &
    (if batch == 1: " (" & formatFloat(gbps, ffDecimal, 1) & " GB/s)" else: "") &
    &" ({(if gpuTimed: \"GPU time\" else: \"wall time\")}, parity checked, {candidates.len} candidates)"

proc linearPpHalf*(): bool =
  ## BYBY_LINEARPP_DTYPE = fp16 (default) | fp32: GPU storage of the FP32 Linear++ sidecar weights
  ## (block G qkvProj/outProj/fuseGate). fp16 halves their memory; every kernel that reads them
  ## widens to FP32 and accumulates in FP32.
  let v = getEnv("BYBY_LINEARPP_DTYPE", "fp16").toLowerAscii()
  case v
  of "fp16", "f16", "half": true
  of "fp32", "f32", "float": false
  else: raise newException(ValueError, "BYBY_LINEARPP_DTYPE must be fp16 or fp32 (got '" & v & "')")

proc prepareGpuLinForLinear*(ctx: Backend; name: string; lin: var Linear): GpuLin =
  ## Prepares a GPU Linear for EVERY weight form (R and A: resident fp4-KM,
  ## resident int4-asymmetric, or the random-init fp32 fallback before loading).
  ## The SAME gemv bytecode (no MSL) for all 3 branches, and the result is ALWAYS written
  ## to gLinCache[name] - the SINGLE SOURCE OF TRUTH that encodeLinByName
  ## (used by R through fp4GemvAt) looks up. Before this version there were 2 separate tables
  ## (gFp4CacheGpu/gInt4CacheGpu) and the fp32 fallback branch was NOT registered
  ## anywhere -> encodeLinByName returned false -> a "prepareGpuLinForLinear not called" crash
  ## during warmup, exactly at the first block that hit the fp32/fallback branch.
  if name in gLinCache and gLinCache[name].ready:
    return gLinCache[name]
  let be = ctx.kind.toByby()
  ensureGpuKernels(be)
  let K = lin.inF; let N = lin.outF
  if lin.useQuant and lin.weightQ.kind == qkFp4KM:
    if name notin gFp4CacheGpu or not gFp4CacheGpu[name].residentReady:
      let lut = fp4_gemv_gpu.buildFp4KmLutQ(lin.weightQ)
      let gsz = if lin.weightQ.groupSize > 0: lin.weightQ.groupSize else: 32
      # The fp4-KM gemv kernel = micro-VM bytecode -> JIT compiled once / shape (OUTSIDE a session).
      # jitName MUST be a real pipeline name; empty => sessionPipeline("") fails at encode time.
      # The paired-row specialization assumes both an even output-row count
      # and the kernel's 32-wide input-group layout. Keep odd/non-multiple
      # shapes on the general implementation instead of failing at JIT time.
      let pairRows = gsz == 32 and K mod 32 == 0 and N mod 2 == 0
      # BYBY_GEMV_VARIANT: 0 = 2 rows, 1 = 32-bit wide loads, 2 = 4 rows/threadgroup (default).
      # A variant not applicable to this shape -> falls back to the 2-row kernel. Measured with bench_gemv_variants.nim.
      let variantSetting = getEnv("BYBY_GEMV_VARIANT", "auto")
      let variant = (try: parseInt(variantSetting) except ValueError: 2)
      let useWide = variant == 1 and pairRows and fp4KmGemvWideSupported(K, N)
      let use4Row = variant == 2 and gsz == 32 and K mod 32 == 0 and N mod 4 == 0
      let gemvRows = if use4Row: 4 elif pairRows: 2 else: 1
      let prog = if use4Row: buildFp4KmGemv4RowProgram(K, N)
                 elif useWide: buildFp4KmGemvWideProgram(K, N)
                 elif pairRows: buildFp4KmGemv2RowProgram(K, N)
                 else: buildFp4GemvProgram(K, N, true, gsz)
      let tag = (if use4Row: "fp4km4_gemv_" elif useWide: "fp4kmw_gemv_"
                 elif pairRows: "fp4km2_gemv_" else: "fp4km_gemv_") & $K & "x" & $N & "_g" & $gsz
      # vmKernelForBe: Vulkan -> kernelNameFor+GLSL cache; Metal -> MSL dispatch
      let kn = vmKernelForBe(prog, tag, be)
      uploadFp4WeightOnce(ctx, name, lin.weightQ.bytes, lut, N, K, kn,
                          rowsPerTg = gemvRows)
      if be == byby.gbVulkan and name in gFp4CacheGpu:
        var c = gFp4CacheGpu[name]
        c.glslSrc = jit_vm.glslSourceFor(gJitV2, prog, tag)
        gFp4CacheGpu[name] = c
      # Prefill MBLOCK=4: weights read once per 4 tokens (like matmul_q4_batched)
      if gsz == 32 and K mod 32 == 0 and name in gFp4CacheGpu:
        let m4prog = buildFp4KmGemmM4Program(K, N)
        let m4tag = "fp4km_m4_" & $K & "x" & $N
        let m4kn = vmKernelForBe(m4prog, m4tag, be)
        var c = gFp4CacheGpu[name]
        c.m4JitName = m4kn
        c.m8JitName = vmKernelForBe(buildFp4KmGemmM8Program(K, N),
          "fp4km_m8_" & $K & "x" & $N, be)
        c.m8TgSize = GemvTg
        if be == byby.gbVulkan:
          c.m4GlslSrc = jit_vm.glslSourceFor(gJitV2, m4prog, m4tag)
        if variantSetting == "auto" and K mod 256 == 0 and N mod 4 == 0:
          let plan = tuneFp4Plan(ctx, c, 1)
          c.jitName = plan.kn; c.rowsPerTg = plan.rows
          c.tgSize = plan.tg; c.tgBytes = plan.bytes; c.decSimd = plan.simd
          c.decKind = plan.kind; c.decRows = plan.kernelRows
          c.decUnroll = plan.unroll; c.decLutConst = plan.lutConst
          if be == byby.gbVulkan: c.glslSrc = jit_vm.cachedGlslSource(gJitV2, plan.kn)
        if getEnv("BYBY_PREFILL_FP4_KERNEL", "auto") == "auto" and K mod 256 == 0:
          let plan = tuneFp4Plan(ctx, c, 8)
          c.m8JitName = plan.kn; c.m8TgSize = plan.tg
        gFp4CacheGpu[name] = c
    lin.weightQ.clearPayload()   # duplicate on the host -> freed, the GPU keeps the Shared copy
    lin.gpuCacheName = name
    result = markFp4Lin(ctx, name, lin.bias.data, K, N)
  elif lin.useQuant and lin.weightQ.kind == qkInt4Asymmetric and int4AsymSupported(K, lin.weightQ.groupSize):
    uploadInt4WeightOnce(ctx, name, lin.weightQ, K, N)
    lin.weightQ.clearPayload()   # duplicate on the host -> freed, the GPU keeps the Shared copy
    result.name = name; result.K = K; result.N = N
    result.knF32 = gInt4CacheGpu[name].knName
    result.wF32 = gInt4CacheGpu[name].scZpBuf
    result.biasH = byby.sessionUploadPrivate(be, uploadBiasOrRaise(name, lin.bias.data, N))
    result.hasBias = true
    result.ready = true
    result.isFp4 = false
    result.int4WBuf = gInt4CacheGpu[name].wBuf
    result.isInt4 = true
  elif lin.useQuant and lin.weightQ.kind == qkInt4Asymmetric:
    raise newException(ValueError, "prepareGpuLinForLinear " & name & ": weightQ is int4-asymmetric " &
      "but K=" & $K & " groupSize=" & $lin.weightQ.groupSize & " does not satisfy int4AsymSupported " &
      "(needs K%64==0, groupSize=0 or a power of 2 >=64). NO fp32 fallback.")
  else:
    # FP32 master: block G attn/fuseGate are trained fully. Only weight.len==K*N is accepted.
    let expected = K * N
    if expected <= 0 or lin.weight.floatValues.len != expected:
      raise newException(ValueError, "prepareGpuLinForLinear " & name &
        ": lin.useQuant=" & $lin.useQuant &
        (if lin.useQuant: " weightQ.kind=" & $lin.weightQ.kind else: "") &
        " weight.len=" & $lin.weight.floatValues.len & " expected=" & $expected &
        " - no valid quantization and no FP32 master.")
    let sidecar = ".attn." in name or ".fuseGate." in name   # block G (Linear++) weights only
    if sidecar and linearPpHalf() and f16GemmSupported(K, N) and be in {byby.gbMetal, byby.gbVulkan}:
      # Halves on the GPU, FP32 accumulation. The decode GEMV is the FP16 GEMM with M = 1.
      let halves = toF16Bits(lin.weight.floatValues)
      result.name = name; result.K = K; result.N = N
      result.knF32 = vmKernelForBe(buildF16GemmProgram(K, N, 1), "f16_gemv_" & $K & "x" & $N, be)
      result.wF16 = byby.sessionUploadPrivatePtr(be, unsafeAddr halves[0], halves.len * sizeof(uint16))
      if (be == byby.gbMetal and result.wF16.metalHandle == nil) or
         (be == byby.gbVulkan and result.wF16.vulkanHandle.buf == nil):
        raise newException(CatchableError, "prepareGpuLinForLinear: FP16 GPU upload failed: " & name)
      result.isF16 = true
      # No FP32 resident copy: the host copy is released and the sequence (non-resident) prefill
      # reports these weights unavailable, so prefill stays on the resident path.
      lin.weight.data = @[]
      lin.weight.mappedData = default(MappedView[float32])
      result.biasH = byby.sessionUploadPrivate(be, uploadBiasOrRaise(name, lin.bias.data, N))
      result.hasBias = true
      result.ready = true
      result.isFp4 = false
      result.isInt4 = false
      gLinCache[name] = result
      return
    let prog = buildF32GemvProgram(K, N, 1.0'f32, false)
    let kn = vmKernelForBe(prog, "f32_gemv_" & $K & "x" & $N, be)
    result.name = name; result.K = K; result.N = N
    result.knF32 = kn
    result.wF32 = if be == byby.gbVulkan: byby.sessionUploadPrivate(be, @(lin.weight.floatValues))
                  else: sessionUploadFloatsShared(be, lin.weight.floatValues)
    if (be == byby.gbMetal and result.wF32.metalHandle == nil) or
       (be == byby.gbVulkan and result.wF32.vulkanHandle.buf == nil):
      raise newException(CatchableError, "prepareGpuLinForLinear: FP32 GPU upload failed: " & name)
    # This [out,in] allocation also serves sequence forward. Release the
    # duplicate host copy (~1.5 GiB for four Codestral sidecars).
    lin.residentFp32 = result.wF32
    lin.residentFp32Ready = true
    lin.residentFp32NT = true
    lin.residentFp32Offset = 0
    lin.weight.data = @[]
    lin.weight.mappedData = default(MappedView[float32])
    result.biasH = byby.sessionUploadPrivate(be, uploadBiasOrRaise(name, lin.bias.data, N))
    result.hasBias = true
    result.ready = true
    result.isFp4 = false
    result.isInt4 = false
  gLinCache[name] = result

proc uploadBiasOrRaise(name: string; bias: seq[float32]; N: int): seq[float32] =
  ## Before, bias.len != N silently became newSeq[float32](N) = 0.
  ## fuseGate.bias inits to 4; if it became 0 -> sigmoid(0)=0.5 instead of ~0.98 -> the sidecar
  ## took 50% of the residual -> garbage tokens. outProj/bias-less Linears may validly have bias.len==0.
  if bias.len == N:
    return bias
  if bias.len == 0:
    # Linear without a bias (or not restored yet). fuseGate REQUIRES bias=4.
    if "fuseGate" in name:
      raise newException(ValueError,
        "uploadBiasOrRaise[" & name & "]: fuseGate.bias is empty (len=0) but needs N=" &
        $N & ". The bias was lost on restore -> gate=sigmoid(0)=0.5 -> garbage tokens. " &
        "Check that the checkpoint load / byName / paramList has fuseGate.bias.")
    return newSeq[float32](N)
  raise newException(ValueError,
    "uploadBiasOrRaise[" & name & "]: bias.len=" & $bias.len & " != N=" & $N &
    " (shape mismatch, NO silent zero padding).")

proc markFp4Lin*(ctx: Backend; name: string; bias: seq[float32]; K, N: int): GpuLin =
  ## The weight is already in gFp4CacheGpu[name] (uploadFp4WeightOnce).
  let be = ctx.kind.toByby()
  ensureGpuKernels(be)
  result.name = name; result.K = K; result.N = N; result.isFp4 = true
  result.biasH = byby.sessionUploadPrivate(be, uploadBiasOrRaise(name, bias, N))
  # A bias-less or all-zero bias costs a dispatch per call for nothing (FP4 base projections
  # of Mamba/Codestral have none), so only a real bias is encoded.
  var nonZero = false
  for v in bias:
    if v != 0'f32:
      nonZero = true
      break
  result.hasBias = nonZero
  result.ready = true

proc encodeLin*(l: GpuLin; hIn: byby.SessionHandle; inOff: int;
                hOut: byby.SessionHandle; outOff: int;
                bias = true; apflu = false)

proc encodeLinByName*(name: string; hIn: byby.SessionHandle; inOff: int;
                      hOut: byby.SessionHandle; outOff: int): bool =
  ## Used by R blocks (mamba2_fused_step.encodeLayerV2 calls it through fp4GemvAt).
  ## Looked up ONLY through gLinCache (see prepareGpuLinForLinear) - no more
  ## separate fp4-vs-int4-vs-fp32 guessing here (that was the earlier bug source: the
  ## fp32-fallback branch was not registered in any table).
  if name notin gLinCache or not gLinCache[name].ready: return false
  encodeLin(gLinCache[name], hIn, inOff, hOut, outOff, bias = false)  # R: no bias added (keeps the old behaviour)
  true

proc encodeLin*(l: GpuLin; hIn: byby.SessionHandle; inOff: int;
                hOut: byby.SessionHandle; outOff: int;
                bias = true; apflu = false) =
  ## y = W x (+ LoRA) (+ bias) (+ APFLU). Same order as Linear.forward():
  ## base -> add the LoRA delta -> add the bias.
  if l.isFp4:
    if not encodeFp4Gemv(gFp4CacheGpu[l.name], hIn, inOff, hOut, outOff):
      raise newException(CatchableError, "encodeLin: fp4 gemv failed " & l.name)
  elif l.isInt4:
    encodeVmKernel(l.knF32, l.N * GemvTg, GemvTg, int4GemvTgBytes(),
                   hOut, outOff, l.int4WBuf, l.wF32, hIn, inOff)
  elif l.isF16:
    var kn = l.knF32
    if inOff mod 4 != 0 and hIn.kind == byby.gbMetal:
      # The float4 input loads need a 16-byte aligned start on Metal.
      if l.knF16Scalar.len == 0:
        gLinCache[l.name].knF16Scalar = vmKernelForBe(buildF16GemmProgram(l.K, l.N, 1, vecInput = false),
          "f16_gemv_s_" & $l.K & "x" & $l.N, hIn.kind)
      kn = gLinCache[l.name].knF16Scalar
    encodeVmKernel(kn, l.N * GemvTg, GemvTg, f16GemmTgBytes(1),
                   hOut, outOff, l.wF16, gDummy, hIn, inOff)
  else:
    encodeVmKernel(l.knF32, l.N * GemvTg, GemvTg, f32GemvTgBytes(),
                   hOut, outOff, gDummy, l.wF32, hIn, inOff)
  if bias and l.hasBias:
    encodeBias(hOut, outOff, l.biasH, l.N, apflu)
  elif apflu:
    raise newException(CatchableError, "encodeLin: apflu needs the bias kernel (" & l.name & " has no bias)")

type Fp4FusedGemv* = object
  ## A decode GEMV with its neighbours fused in (see fp4DecodeFused); kn is empty when the
  ## projection has no decode form.
  kn*: string
  threads*, tg*, bytes*: int

var gFp4Fused = initTable[string, Fp4FusedGemv]()

proc fp4DecodeFused*(name: string; accumulate, gatedInput: bool): Fp4FusedGemv =
  ## The decode GEMV of the FP4-KM projection `name`, built with the geometry the autotune chose
  ## (rows and threadgroup width, simdgroup or tree reduction), plus `accumulate` (out += W x: the
  ## residual add without its own dispatch) and/or `gatedInput` (x[k] = in[k] * silu(in[k + K]):
  ## the SiLU-gated MLP activation without its own dispatch). Built once per projection and flags.
  let key = name & "|" & $accumulate & "|" & $gatedInput
  if key in gFp4Fused: return gFp4Fused[key]
  if name in gFp4CacheGpu:
    let c = gFp4CacheGpu[name]
    let tg = if c.tgSize > 0: c.tgSize else: GemvTg
    let rows = max(1, c.rowsPerTg)
    if c.decKind == "sg":
      # The selected subgroup-row kernel itself, with the epilogue; it has no gated-input form.
      if not gatedInput and fp4KmGemvSubgroupSupported(c.cols, c.rows, tg, c.decRows, c.decUnroll):
        let tag = "fp4_sg" & $c.decRows & "u" & $c.decUnroll & (if c.decLutConst: "c" else: "t") &
          "_tg" & $tg & (if accumulate: "_acc" else: "") & "_" & $c.cols & "x" & $c.rows
        result.kn = vmKernelForBe(buildFp4KmGemvSubgroupProgram(c.cols, c.rows, tg, c.decRows,
          c.decUnroll, c.decLutConst, accumulate), tag, c.wBuf.kind)
        result.threads = (c.rows div rows) * tg
        result.tg = tg
        result.bytes = fp4KmGemvSubgroupTgBytes(c.decLutConst)
    elif fp4KmGemvDecodeSupported(c.cols, c.rows, tg, rows):
      let tag = "fp4_dec" & $rows & (if c.decSimd: "s" else: "t") & "_tg" & $tg &
        (if accumulate: "_acc" else: "") & (if gatedInput: "_gated" else: "") & "_" & $c.cols & "x" & $c.rows
      result.kn = vmKernelForBe(buildFp4KmGemvDecodeProgram(c.cols, c.rows, tg, rows, c.decSimd,
        accumulate, gatedInput), tag, c.wBuf.kind)
      result.threads = (c.rows div rows) * tg
      result.tg = tg
      result.bytes = fp4KmGemvDecodeTgBytes(rows, tg)
  gFp4Fused[key] = result

# ── Scratch shared by every A block ──────────────────────────────────────
type
  AttnScratch* = object
    hAttn*: byby.SessionHandle    ## [hidden]
    hFF*: byby.SessionHandle      ## [ffDim]
    hidden*, ffDim*: int
    ready*: bool

var gAttnScratch* : AttnScratch

proc ensureAttnScratch*(be: byby.GpuBackend; hidden, ffDim: int) =
  if gAttnScratch.ready and gAttnScratch.hidden == hidden and gAttnScratch.ffDim >= ffDim: return
  gAttnScratch.hAttn = byby.sessionAllocScratch(be, hidden)
  if ffDim > 0:
    gAttnScratch.hFF = byby.sessionAllocScratch(be, ffDim)
  gAttnScratch.hidden = hidden
  gAttnScratch.ffDim = ffDim
  gAttnScratch.ready = true

# ── Block A (LinearPlusAttention + FFN) ─────────────────────────────────
type
  GpuAttnLayer* = object
    lay*: LpLayout
    hWork*: byby.SessionHandle          ## [proj | y | z | S] (the state lives on the GPU)
    knStep*: string
    qkv*, outp*, fc1*, fc2*: GpuLin
    gate*: GpuLin
    hGateInput*, hGateLogit*, hGateProb*, hGateComplement*, hGateOnes*: byby.SessionHandle
    ## Snapshot before the in-place mul (allocated only with BYBY_G_DIAGNOSTICS): to print the mamba/attn/mix means
    hDiagMamba*, hDiagAttn*: byby.SessionHandle
    gated*: bool
    ln1G*, ln1B*, ln2G*, ln2B*: byby.SessionHandle
    ln1Ones*: byby.SessionHandle        ## gamma=1 (preset "-unit": reads checkpoints trained with the old trainer)
    hidden*, ffDim*, nHeads*: int
    eps*: float32
    ready*: bool

var gAttnLayers* = initTable[int, GpuAttnLayer]()
var gZeroA: byby.SessionHandle
var gZeroALen = 0

proc initAttnLayer*(ctx: Backend; hidden, nHeads, ffDim: int;
                    qkv, outp, fc1, fc2: GpuLin;
                    ln1G, ln1B, ln2G, ln2B: seq[float32]; eps: float32;
                    gate: GpuLin = GpuLin()): GpuAttnLayer =
  let be = ctx.kind.toByby()
  ensureGpuKernels(be)
  if hidden mod nHeads != 0:
    raise newException(ValueError, "initAttnLayer: hidden is not divisible by nHeads")
  if hidden mod 64 != 0:
    raise newException(ValueError,
      "initAttnLayer: hidden must be divisible by 64 (buffer offsets need 256 bytes on Intel)")
  result.lay = lpLayout(hidden, nHeads)
  result.hWork = byby.sessionUploadPrivate(be, newSeq[float32](result.lay.total))
  result.knStep = vmKernelForBe(buildLinearPlusStepProgram(result.lay, 1e-6'f32),
                           "lp_step_" & $nHeads & "_" & $hidden, be)
  result.qkv = qkv; result.outp = outp; result.fc1 = fc1; result.fc2 = fc2
  result.gate = gate
  result.gated = gate.ready
  if result.gated:
    result.hGateInput = byby.sessionAllocScratch(be, 2 * hidden)
    result.hGateLogit = byby.sessionAllocScratch(be, hidden)
    result.hGateProb = byby.sessionAllocScratch(be, hidden)
    result.hGateComplement = byby.sessionAllocScratch(be, hidden)
    var ones = newSeq[float32](hidden)
    for i in 0 ..< hidden: ones[i] = 1'f32
    result.hGateOnes = byby.sessionUploadPrivate(be, ones)
  result.ln1G = byby.sessionUploadPrivate(be, ln1G)
  block:
    var ones = newSeq[float32](ln1G.len)
    for i in 0 ..< ones.len: ones[i] = 1'f32
    result.ln1Ones = byby.sessionUploadPrivate(be, ones)
  result.ln1B = byby.sessionUploadPrivate(be, ln1B)
  result.ln2G = byby.sessionUploadPrivate(be, ln2G)
  result.ln2B = byby.sessionUploadPrivate(be, ln2B)
  result.hidden = hidden; result.nHeads = nHeads; result.ffDim = ffDim
  result.eps = eps
  result.ready = true
  ensureAttnScratch(be, hidden, ffDim)

proc resetAttnLayerState*(ctx: Backend; blockIdx: int): bool =
  ## Clears z and S (starts a new sequence). MUST be called inside an open session.
  if blockIdx notin gAttnLayers: return false
  let be = ctx.kind.toByby()
  let lp = addr gAttnLayers[blockIdx]
  let n = lp.lay.offS + lp.lay.H * lp.lay.D * lp.lay.D - lp.lay.offZ
  if gZeroALen < n:
    gZeroA = byby.sessionUploadPrivate(be, newSeq[float32](n))
    gZeroALen = n
  var hw = lp.hWork
  sessionCopyInto(be, gZeroA, hw, 0, lp.lay.offZ, n)

proc encodeLinearPlusBranch*(ctx: Backend; blockIdx: int;
                             hInput: byby.SessionHandle;
                             hNorm: var byby.SessionHandle;
                             hOut: byby.SessionHandle;
                             useRms = false; unitGamma = false;
                             inputPreNormalized = false) =
  ## Computes only the Linear++ branch from the input hidden; no residual/FFN added.
  ##
  ## The input norm MUST match the R branch of the same block: in training a P block computes ONE
  ## normalised output x1 = rms1(x) and feeds it to BOTH Mamba and Linear++ (jamba_hybrid.forwardBlock).
  ##   useRms=true  : RMSNorm with gamma only (rms-* presets; what current training uses)
  ##   useRms=false : LayerNorm gamma+beta (ln-* presets; plain 'A' blocks always do this)
  ##   unitGamma    : gamma=1 (only the "-unit" preset, for reading old checkpoints)
  ## This function used to ALWAYS use LayerNorm -> the P sidecar at inference got an input of a different scale
  ## than in training (mean + beta subtracted instead of dividing by rms) while the R branch followed the preset.
  doAssert blockIdx in gAttnLayers, "encodeAttnLayer: initAttnLayer(" & $blockIdx & ") not called"
  let lp = addr gAttnLayers[blockIdx]
  template L: untyped = lp[]
  let be = ctx.kind.toByby()
  let lay = L.lay
  if not inputPreNormalized:
    if useRms:
      let gam = if unitGamma: L.ln1Ones else: L.ln1G
      if not sessionRmsnormEnc(be, hInput, gam, hNorm, 1, L.hidden, L.eps):
        raise newException(CatchableError, "encodeLinearPlusBranch: rmsnorm ln1 failed")
    else:
      if not sessionLayernormEnc(be, hInput, L.ln1G, L.ln1B, hNorm, 1, L.hidden, L.eps):
        raise newException(CatchableError, "encodeLinearPlusBranch: layernorm ln1 failed")
  encodeLin(L.qkv, hNorm, 0, L.hWork, lay.offProj)               # proj = [Q|K|V]
  encodeVmKernel(L.knStep, lay.H * lay.D, lay.D, 16,             # Linear++ step (state in place)
                 L.hWork, 0, gDummy, gDummy, gDummy, 0)
  encodeLin(L.outp, L.hWork, lay.offY, hOut, 0)

proc encodeAttnLayer*(ctx: Backend; blockIdx: int;
                      hRes: byby.SessionHandle;          ## [hidden] in + out (in-place)
                      hNorm: var byby.SessionHandle) =   ## [hidden] scratch
  let lp = addr gAttnLayers[blockIdx]
  template L: untyped = lp[]
  let be = ctx.kind.toByby()
  var hA = gAttnScratch.hAttn
  var hF = gAttnScratch.hFF
  encodeLinearPlusBranch(ctx, blockIdx, hRes, hNorm, hA)
  var hDst = hRes
  if not sessionVecOp(be, "add", hRes, hA, hDst, L.hidden):
    raise newException(CatchableError, "encodeAttnLayer: residual attn failed")

  # ── FFN ──
  if not sessionLayernormEnc(be, hRes, L.ln2G, L.ln2B, hNorm, 1, L.hidden, L.eps):
    raise newException(CatchableError, "encodeAttnLayer: layernorm ln2 failed")
  encodeLin(L.fc1, hNorm, 0, hF, 0, bias = true, apflu = true)   # APFLU(0.1, 1.0) co bias
  encodeLin(L.fc2, hF, 0, hA, 0)
  if not sessionVecOp(be, "add", hRes, hA, hDst, L.hidden):
    raise newException(CatchableError, "encodeAttnLayer: residual ffn failed")

proc lpStepSelfCheck*(ctx: Backend; blockIdx: int; T = 4): string =
  ## 100% GPU CHECK - NO model computation on the CPU. Runs the Linear++
  ## step kernel DIRECTLY (the same kernel/buffer/state as inference) with a made-up [Q|K|V] and compares with
  ## the ANALYTIC identities of linear attention:
  ##   (1) K, V fixed over T tokens       => y_t = V               (normalised weights = 1)
  ##   (2) K fixed, V alternating a,b,a,b    => y_t = running mean of V
  ## A wrong head offset (h*D), S/z accumulation, barrier between lanes or normalisation all break it.
  if blockIdx notin gAttnLayers: return "block A " & $blockIdx & " not initialised"
  let be = ctx.kind.toByby()
  let lp = addr gAttnLayers[blockIdx]
  let lay = lp.lay
  let C = lay.H * lay.D
  var seed = 987654321'u32
  proc rnd(): float32 =
    seed = seed * 1664525'u32 + 1013904223'u32
    (float32(seed shr 8) / 16777216.0'f32 - 0.5'f32) * 3.0'f32
  var k = newSeq[float32](C)
  var va = newSeq[float32](C)
  var vb = newSeq[float32](C)
  for i in 0 ..< C:
    k[i] = rnd()
    va[i] = rnd()
    vb[i] = rnd()
  var qs = newSeq[seq[float32]](T)
  for t in 0 ..< T:
    qs[t] = newSeq[float32](C)
    for i in 0 ..< C: qs[t][i] = rnd()
  var worst = [0.0, 0.0]
  for mode in 1 .. 2:
    if not sessionBegin(be): return "sessionBegin failed (reset)"
    discard resetAttnLayerState(ctx, blockIdx)
    if not sessionEnd(be): return "sessionEnd failed (reset)"
    for t in 0 ..< T:
      var proj = newSeq[float32](3 * C)
      for i in 0 ..< C:
        proj[i] = qs[t][i]
        proj[C + i] = k[i]
        proj[2 * C + i] = (if mode == 1 or t mod 2 == 0: va[i] else: vb[i])
      var hP = sessionUpload(be, proj)
      var hw = lp.hWork
      if not sessionBegin(be):
        sessionFree(be, hP)
        return "sessionBegin failed"
      discard sessionCopyInto(be, hP, hw, 0, lay.offProj, 3 * C)
      var err = ""
      try:
        encodeVmKernel(lp.knStep, lay.H * lay.D, lay.D, 16, hw, 0, gDummy, gDummy, gDummy, 0)
      except CatchableError as e:
        err = e.msg
      let okEnd = sessionEnd(be)
      sessionFree(be, hP)
      if err.len > 0: return "encode error: " & err
      if not okEnd: return "sessionEnd failed: " & sessionGetLastError(be)
      let all = sessionRead(be, hw, lay.offY + C)
      var mx = 0.0
      var md = 0.0
      let na = t div 2 + 1          # number of a tokens in 0..t
      let nb = (t + 1) div 2        # number of b tokens in 0..t
      for i in 0 ..< C:
        let want: float64 =
          if mode == 1: float64(va[i])
          else: (float64(na) * float64(va[i]) + float64(nb) * float64(vb[i])) / float64(t + 1)
        md = max(md, abs(float64(all[lay.offY + i]) - want))
        mx = max(mx, abs(want))
      worst[mode - 1] = max(worst[mode - 1], md / max(mx, 1e-12))
  let tol = 2e-3
  result = "kernel Linear++ step (khoi A #" & $blockIdx & ", " & $T & " token): " &
           "K,V fixed error=" & $worst[0] &
           " | V alternating error=" & $worst[1] &
           (if worst[0] < tol and worst[1] < tol: "  => MATCHES the analytic identity"
            else: "  => DIFFERS: the Linear++ step kernel on the GPU is wrong (head offset/state accumulation/barrier)")
