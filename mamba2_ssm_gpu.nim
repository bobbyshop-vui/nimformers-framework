## mamba2_ssm_gpu.nim — microVM only (NO new Metal kernels).
## 1 session/token: encode only, no sessionRead between layers.

import std/tables
import std/bitops
import std/math
import std/[times, os, strformat, strutils, sequtils]
import backend
import vendor/bybylang/gpubackend as byby
import vendor/bybylang/backends/cpu/cpu_backend as cpu
import session_vm
import jit_vm
import session_vm_dispatch
import mamba2_vm_kernels
import prefill_profile   # BYBY_PREFILL_PROFILE: vm.upload / vm.gpu / vm.read
import mamba2_full_vm
import fp4_gemv_gpu
import vendor/bybylang/backends/vulkan/vulkan_driver as vkDrv
import nimformer     # Tensor, ApfAdamState, apfAdamStepGpu (for apfAdamStepFused)
import customfloat

type
  LayerOff* = object
    P, offZ, offXBC, offDt: int
    offALog, offDSkip, offDtBias, offNormW: int
    offConvW, offConvB, offConvSt: int
    offSsm, offIdxH, offIdxG: int
    offY, offNormed: int
    total: int

  Mamba2GpuLayer* = object
    be*: byby.GpuBackend
    hWork*: byby.SessionHandle
    hScratch*: byby.SessionHandle
    off*: LayerOff
    H*, D*, N*, G*, C*, cDim*, K*, HD*, hidden*: int
    ready*: bool

  Mamba2SsmShapeInfo = object
    program: VmProgram
    idxH, idxG: seq[float32]
    offX, offB, offC, offDt, offALog, offDSkip, offState, offY, offIdxH, offIdxG: int
    totalLen: int

var gShapeCache = initTable[string, Mamba2SsmShapeInfo]()
var gSsmAtCache = initTable[string, VmProgram]()  # the scan runs directly on hWork, not through hScratch
var gGpuLayers* = initTable[int, Mamba2GpuLayer]()
var gProfileLastT: float = 0.0   # NEW: used only by profileLayerStepByIdx (timing between steps)
var gCodeBufCache = initTable[string, byby.SessionHandle]()
var gJitCache = jit_vm.initJitCache(jit_vm.JitConfig(
  hotThreshold: 0, enableMetalJit: true, enableVulkanJit: true,
  enableCpuJit: false, maxCacheEntries: 128))

proc shapeKey(H, D, N, G: int): string = $H & "_" & $D & "_" & $N & "_" & $G

proc getSsmAtProgram(H, D, N, G, C: int; o: LayerOff): VmProgram =
  ## OPTIMISATION (merged dispatches): buildMamba2StepAt takes the offsets as PARAMETERS, so it reads/writes
  ## DIRECTLY on hWork at the real positions (o.offXBC/offB/offC/offSsm/offY...), no
  ## 9 sessionCopyInto calls to/from hScratch like the original getShapeInfo(...).program
  ## - of which the 2 state copies (HD*N elements, the largest part of the whole
  ## step) were the main overhead (matmul-adjacent memcpy). Cached by shapeKey
  ## because layers with the same H,D,N,G in one model share C/cDim/K/G, so the offsets
  ## from computeOff() are identical.
  let key = shapeKey(H, D, N, G)
  if key in gSsmAtCache: return gSsmAtCache[key]
  result = buildMamba2StepAt(H, D, N, G,
    o.offXBC, o.offXBC + C, o.offXBC + C + G * N,
    o.offDt, o.offALog, o.offDSkip, o.offSsm, o.offY, o.offIdxH, o.offIdxG)
  gSsmAtCache[key] = result

proc blit*(dst: var seq[float32]; dOff: int; src: openArray[float32]; sOff, n: int) =
  ## Copies a float32 block with copyMem (data marshalling, NOT computation).
  if n > 0:
    copyMem(addr dst[dOff], unsafeAddr src[sOff], n * sizeof(float32))

proc computeOff(C, cDim, H, D, N, G, K: int): LayerOff =
  let HD = H * D
  result.P = C + cDim + H
  result.offZ = result.P
  result.offXBC = result.offZ + C
  result.offDt = result.offXBC + cDim
  result.offALog = result.offDt + H
  result.offDSkip = result.offALog + H
  result.offDtBias = result.offDSkip + H
  result.offNormW = result.offDtBias + H
  result.offConvW = result.offNormW + C
  result.offConvB = result.offConvW + cDim * K
  result.offConvSt = result.offConvB + cDim
  result.offSsm = result.offConvSt + cDim * max(K - 1, 0)
  result.offIdxH = result.offSsm + HD * N
  result.offIdxG = result.offIdxH + HD
  result.offY = result.offIdxG + HD
  result.offNormed = result.offY + max(C, HD)
  result.total = result.offNormed + C

proc kernelNameFor*(ctx: Backend; program: VmProgram; tag: string): string =
  ## Computes the kernel name (GPU bytecode hash) OUTSIDE a session. The tag is only a namespace;
  ## Vulkan uses the same static-per-program lowering strategy as Metal, but
  ## emits GLSL/SPIR-V instead of MSL. Never hand a Metal pipeline name to it.
  if ctx.kind == bkVulkan:
    return "vkvm_" & jit_vm.programKeyNoSession(program, tag)
  if tag.len > 0 and tag[0] == '@': jit_vm.dispatchProgramTagged(gJitCache, program, tag)
  else: jit_vm.dispatchProgram(gJitCache, program, tag)

proc encodeMicroVm*(ctx: Backend; program: VmProgram; hMem: byby.SessionHandle;
                    nThreads: int; tgBytes = 4; tag = "mvm"; tgSize = 0;
                    preKname = "") =
  ## tgSize > 0: FORCES the lanes/threadgroup ("1 threadgroup / token" kernels need exactly this value;
  ## if the GPU cannot provide it, RAISE instead of silently lowering it).
  if program.words.len == 0 or nThreads <= 0:
    raise newException(CatchableError, "encodeMicroVm: bad args")
  let be = ctx.kind.toByby()
  let tgFloats = max(1, tgBytes div 4)
  var tg = if tgSize > 0: tgSize
           elif tgBytes > 4: min(nThreads, 1024) else: min(nThreads, 256)
  # Resolve the kernel name before backend-specific dispatch. Vulkan must use
  # its GLSL/SPIR-V program key, never the Metal JIT name. This was previously
  # referenced before declaration and failed compilation (`kname` undeclared).
  let kname = if preKname.len > 0: preKname
              elif be == byby.gbVulkan: kernelNameFor(ctx, program, tag)
              else: jit_vm.dispatchProgramNoSession(gJitCache, program, tag)

  if be == byby.gbVulkan:
    if tg <= 0 or tg > vkDrv.vulkanMaxTgThreads():
      raise newException(CatchableError,
        "encodeMicroVm[" & tag & "]: unsupported Vulkan workgroup size " & $tg)
    let glsl = jit_vm.lowerBytecodeToGlslCached(program, kname)
    if not byby.sessionJitMicroVmEnc(be, kname, glsl, hMem, nThreads, tg, tgFloats):
      raise newException(CatchableError, "encodeMicroVm[" & tag & "]: Vulkan JIT dispatch failed")
    return

  # preKname: the kernel name computed BEFORE opening the session. This function runs INSIDE an open session, so it must
  # NOT open its own session for GPU hashing (metal_session_begin refuses nested sessions) ->
  # dispatchProgramNoSession (uses the remembered key, or a fallback key that needs no GPU).
  let useJit = kname.len > 0 and kname != "micro_vm_kernel"
  let pipeName = if useJit: kname else: "micro_vm_kernel"
  if not sessionPipeline(pipeName):
    if not sessionPipeline("micro_vm_kernel"):
      raise newException(CatchableError, "encodeMicroVm: no pipeline " & pipeName)

  when defined(macosx):
    if hMem.metalHandle == nil:
      raise newException(CatchableError, "encodeMicroVm: nil metal handle")
    if useJit:
      discard sessionBindBuffer(1, hMem.metalHandle)
      discard sessionBindBuffer(2, hMem.metalHandle)
      discard sessionBindBuffer(3, hMem.metalHandle)
    else:
      let codeKey = tag & "_" & $program.words.len
      if codeKey notin gCodeBufCache:
        var codeF32 = newSeq[float32](program.words.len)
        if program.words.len > 0:
          copyMem(addr codeF32[0], unsafeAddr program.words[0], program.words.len * 4)
        gCodeBufCache[codeKey] = byby.sessionUploadPrivate(be, codeF32)
      let hCode = gCodeBufCache[codeKey]
      if hCode.metalHandle == nil:
        raise newException(CatchableError, "encodeMicroVm: nil code handle")
      discard sessionBindBuffer(0, hCode.metalHandle)
      discard sessionBindBuffer(1, hMem.metalHandle)
      discard sessionBindBuffer(2, hMem.metalHandle)
      discard sessionBindBuffer(3, hMem.metalHandle)
      discard sessionBindInt(4, program.words.len)
      discard sessionBindInt(5, tgFloats)
  discard sessionSetThreadgroupBytes(0, max(tgBytes, 4))
  when defined(macosx):
    # FIX for garbage tokens (bug #9): metal_enc_dispatch_threads SILENTLY lowers tgSize
    # to maxTotalThreadsPerThreadgroup if we ask for more. For kernels
    # WITHOUT threadgroup memory (tgBytes<=4, independent threads) that is
    # harmless - the grid is just split into more threadgroups. But for kernels WITH
    # a reduction (tgBytes>4: rms/rmsL) it is very dangerous: the program is
    # GENERATED for the requested tgSize (index step, reduction tree
    # stride, and R0 = thread_position_in_GRID used as the index into
    # threadgroup memory) - if Metal splits it into several threadgroups, each
    # threadgroup has its OWN MEMORY, most slots are never written, barriers
    # only sync within each group -> wrong sums -> GARBAGE results WITHOUT AN ERROR.
    let maxTG = sessionMaxTgThreads()
    if maxTG > 0 and tg > maxTG:
      if tgBytes > 4 or tgSize > 0:
        raise newException(CatchableError,
          "encodeMicroVm[" & tag & "]: reduction kernel needs " & $tg &
          " threads/threadgroup but pipeline '" & pipeName & "' allows at most " &
          $maxTG & " on this GPU. Dispatching anyway, Metal would create " &
          $((nThreads + maxTG - 1) div maxTG) & " separate threadgroups and the result " &
          "WOULD BE GARBAGE. Use rmsTgSizeFor() to build the program with tgSize=" & $maxTG &
          " (or set BYBY_RMS_TG=" & $maxTG & ").")
      else:
        tg = maxTG
  if not sessionDispatch(nThreads, tg):
    raise newException(CatchableError, "encodeMicroVm: dispatch failed")

type MvmStep* = tuple[prog: VmProgram, nThreads, tgBytes, tgSize: int, tag: string]

proc mamba2ScanStep*(L: M2Lay; ctx: Backend; forceParallel = false): MvmStep =
  ## Parallel affine scan is verified/default on Metal. Vulkan uses the serial
  ## recurrence until its shared-memory/barrier path passes differential tests.
  ## The selection follows the active backend (not merely the host OS).
  ## BYBY_MAMBA_SCAN_PARALLEL=0/1 overrides; forceParallel is for parity tests.
  let scanParDefault = if ctx.kind.toByby() == byby.gbMetal: "1" else: "0"
  let parallel = forceParallel or getEnv("BYBY_MAMBA_SCAN_PARALLEL", scanParDefault) == "1"
  let tg = if parallel: m2ScanTgSize(L.T)
           else: 0
  if tg > 0:
    doAssert L.B * L.HD <= high(int) div tg,
      "Mamba2 parallel scan dispatch size overflow"
    return (buildM2ScanParallelProgram(L, tg), L.B * L.HD * tg, tg * 16, tg,
      "@m2_scan_parallel_" & m2Tag(L) & "_tg" & $tg)
  result = (buildM2ScanProgram(L), L.B * L.HD, 4, 0,
    "@m2_scan_" & m2Tag(L))

proc encodeSteps(ctx: Backend; steps: openArray[MvmStep]; names: openArray[string];
                 h: byby.SessionHandle) =
  ## tail recursion instead of `for st in steps` (no host loop)
  if steps.len == 0: return
  encodeMicroVm(ctx, steps[0].prog, h, steps[0].nThreads, steps[0].tgBytes, steps[0].tag,
                steps[0].tgSize, names[0])
  encodeSteps(ctx, steps.toOpenArray(1, steps.high), names.toOpenArray(1, names.high), h)

const VmTimeoutRetryMarker = "[VM_GPU_TIMEOUT_RETRY]"

# Conservative fallback for unknown/constrained GPUs (e.g. Intel Iris 640 with 1.5 GB shared VRAM).
# On Apple Silicon unified-memory machines the auto-detect below raises this significantly.
const DefaultVmScratchLimitMiB* = 768

when defined(macosx):
  proc sysctl(name: ptr cint; namelen: cuint; oldp: pointer; oldlenp: ptr csize_t;
              newp: pointer; newlen: csize_t): cint {.importc, header: "<sys/sysctl.h>".}

proc autoVmScratchLimitMiB(): int =
  ## Returns a safe Metal scratch limit based on physical RAM.
  ## Apple Silicon shares RAM between CPU and GPU; 1/8 of RAM is a conservative
  ## budget that leaves plenty for the model weights and OS.
  ## Falls back to DefaultVmScratchLimitMiB on non-macOS or query failure.
  when defined(macosx):
    var mib: array[2, cint] = [6.cint, 24.cint]  # CTL_HW=6, HW_MEMSIZE=24
    var memBytes: uint64
    var sz = csize_t(sizeof(memBytes))
    if sysctl(addr mib[0], 2, addr memBytes, addr sz, nil, 0) == 0 and memBytes > 0:
      let ramMiB = int(memBytes div (1024 * 1024))
      # Use 1/8 of RAM, clamped to [DefaultVmScratchLimitMiB, 8192] MiB.
      return max(DefaultVmScratchLimitMiB, min(ramMiB div 8, 8192))
  DefaultVmScratchLimitMiB

let gAutoVmScratchLimitMiB* = autoVmScratchLimitMiB()

proc checkVmScratchCapacity*(ctx: Backend; nFloats: int; label: string) =
  ## Metal needs one contiguous MTLBuffer per VM chain. Intel integrated GPUs
  ## can reject a multi-GiB buffer after the host seq was already allocated.
  ## Fail with an actionable diagnostic; block callers invoke this before
  ## allocating their large scratch seq.
  if nFloats < 0:
    raise newException(ValueError, "microVM " & label & ": negative scratch length")
  if ctx.kind.toByby() != byby.gbMetal: return
  var limitMiB = gAutoVmScratchLimitMiB
  let override = getEnv("BYBY_MVM_MAX_SCRATCH_MB")
  if override.len > 0:
    try:
      limitMiB = parseInt(override)
    except ValueError:
      raise newException(ValueError,
        "BYBY_MVM_MAX_SCRATCH_MB must be a positive integer (MiB), got '" & override & "'")
    if limitMiB <= 0 or limitMiB > 1_048_576:
      raise newException(ValueError,
        "BYBY_MVM_MAX_SCRATCH_MB must be in 1..1048576 MiB")
  let requestedBytes = int64(nFloats) * int64(sizeof(float32))
  let limitBytes = int64(limitMiB) * 1024'i64 * 1024'i64
  if requestedBytes > limitBytes:
    raise newException(CatchableError,
      &"microVM {label}: scratch needs {float64(requestedBytes) / (1024.0 * 1024.0):.1f} MiB " &
      &"({nFloats} float32), above the Metal safety limit {limitMiB} MiB. " &
      "Reduce CFG_SEQ_LEN/CFG_BATCH_SIZE or set BYBY_MVM_MAX_SCRATCH_MB to override.")

proc vmUploadSucceeded(be: byby.GpuBackend; h: byby.SessionHandle): bool =
  when defined(macosx):
    if be == byby.gbMetal: return h.metalHandle != nil
  if be == byby.gbVulkan: return h.vulkanHandle.buf != nil
  true

proc isVmGpuTimeout(msg: string): bool =
  let s = msg.toLowerAscii()
  result = "timeout" in s or "timed out" in s

proc beginVmSession(be: byby.GpuBackend; label: string) =
  if sessionBegin(be): return
  # A prior Metal command buffer may have timed out and left its queue unusable.
  # Capture the diagnostic before recovery, as recovery/next session overwrites it.
  let ended = sessionEnd(be)
  if not ended:
    let detail = sessionGetLastError(be)
    discard sessionRecoverQueue(be)
    stderr.writeLine("[microvm] recovered queue before retrying session " & label &
      (if detail.len > 0: ": " & detail else: ""))
  if not sessionBegin(be):
    raise newException(CatchableError,
      "microVM " & label & ": sessionBegin failed after queue recovery")

proc finishVmSession(be: byby.GpuBackend; label: string; ended: var bool) =
  ## False/failed command buffers must never be followed by sessionRead: Metal
  ## can otherwise return the untouched/partially written scratch as valid data.
  let endOk = sessionEnd(be)
  ended = true
  if endOk: return
  let detail = sessionGetLastError(be) # read before recovery replaces the error
  let recovered = sessionRecoverQueue(be)
  let msg = "microVM " & label & ": sessionEnd failed" &
    (if detail.len > 0: ": " & detail else: " (backend supplied no diagnostic)") &
    (if recovered: "; Metal queue recovered" else: "; queue recovery failed")
  if isVmGpuTimeout(detail):
    raise newException(CatchableError, VmTimeoutRetryMarker & " " & msg)
  raise newException(CatchableError, msg)

var gTgCapReady = false
var gTgCapBackend: BackendKind

proc ensureTgCap*(ctx: Backend) =
  ## Sets gTgCap (mamba2_full_vm) = the largest pow2 <= the REAL maxTotalThreadsPerThreadgroup of the GPU.
  ## Call before building any reduction program (rmsTg / rmsRowTg / lpPhiTg).
  ## BYBY_RMS_TG=<n> forces it. Non-Metal backends: keeps 1024.
  if gTgCapReady and gTgCapBackend == ctx.kind: return
  var cap = 1024
  var queried = true
  if ctx.kind == bkVulkan:
    cap = min(1024, vkDrv.vulkanMaxTgThreads())
  else:
    when defined(macosx):
      queried = false
      let be = ctx.kind.toByby()
      if sessionBegin(be):
        if sessionPipeline("micro_vm_kernel"):
          let m = sessionMaxTgThreads()
          if m > 0:
            cap = min(cap, m)
            queried = true
        discard sessionEnd(be)
      let envTg = getEnv("BYBY_RMS_TG")
      if envTg.len > 0:
        try:
          let v = parseInt(envTg)
          if v > 0: cap = min(cap, v); queried = true
        except ValueError:
          discard
  # JIT pipelines use more registers than micro_vm_kernel, so the real limit is LOWER
  # (Iris 640: micro_vm_kernel 512, JIT kernels only 448). Unknown before compiling,
  # so it is capped at 256 (enough for reductions, runs on every Mac GPU). BYBY_RMS_TG forces it.
  if ctx.kind != bkVulkan and getEnv("BYBY_RMS_TG").len == 0: cap = min(cap, 256)
  let requested = getEnv("BYBY_RMS_TG")
  if requested.len > 0:
    try: cap = min(cap, max(1, parseInt(requested)))
    except ValueError: discard
  cap = 1 shl fastLog2(max(cap, 32))
  gTgCap = cap
  if not queried:
    # In another session / not queryable yet: use 256 for now (<= every Mac GPU), retry later.
    gTgCap = min(cap, 256)
    return
  gTgCapReady = true
  gTgCapBackend = ctx.kind
  stderr.writeLine "[tgcap] threadgroup limit for reduction kernels = " & $cap

var gRmsTgSize = initTable[string, int]()

proc rmsTgSizeFor*(ctx: Backend; C: int): int =
  ## FIX for garbage tokens (bug #9). Picks the threads/threadgroup of the rmsGate kernel
  ## (the gated RMSNorm at the END of the mixer - runs in EVERY block, EVERY token) so Metal
  ## runs EXACTLY ONE threadgroup.
  ##
  ## It used to hardcode 1024 (buildRmsGateLargeProgram(C, eps, 1024) +
  ## dispatch 1024 threads). 1024 exceeds the maxTotalThreadsPerThreadgroup of
  ## integrated GPUs (Intel Iris Plus 640 usually only 256) -> the shim lowered it to 256
  ## -> Metal created 4 threadgroups, each with its OWN threadgroup memory, while
  ## the program used R0 = thread_position_in_GRID (0..1023) as the index into
  ## that memory and reduced from stride 512. Result: most slots were never
  ## written, the sum of squares was wrong, invStd was wrong -> GARBAGE mixer output in
  ## every block, every token (consistent, repeated exactly on every run).
  ##
  ## Force it by hand: BYBY_RMS_TG=<n>.
  let cacheKey = $ctx.kind & "_" & $C
  if cacheKey in gRmsTgSize: return gRmsTgSize[cacheKey]
  # Picks tgSize = the largest power of 2 <= min(1024, C, the GPU limit) AND dividing C.
  # Closed form (bit trick) instead of 2 `while t > ...: t = t div 2` loops.
  let t0 = 1 shl fastLog2(min(1024, C))
  var bound = 1024
  if ctx.kind == bkVulkan: bound = vkDrv.vulkanMaxTgThreads()
  when defined(macosx):
    if ctx.kind != bkVulkan and sessionPipeline("micro_vm_kernel"):
      let m = sessionMaxTgThreads()
      if m > 0: bound = m
    let envTg = getEnv("BYBY_RMS_TG")
    if envTg.len > 0:
      try:
        let v = parseInt(envTg)
        if v > 0: bound = min(bound, v)
      except ValueError:
        discard
  let t = min(C and (-C), 1 shl fastLog2(min(t0, bound)))
  when defined(macosx):
    if t < t0 and t < 32:
      raise newException(CatchableError,
        "rmsTgSizeFor: no valid tgSize found for C=" & $C &
        " with a limit of " & $bound & " threads/threadgroup")
    stderr.writeLine "[rmsgate] tgSize=" & $t & " for C=" & $C &
      " (the GPU allows at most " & $bound & " threads/threadgroup) - FIX bug #9"
  gRmsTgSize[cacheKey] = t
  result = t

proc runMicroVm*(ctx: Backend; program: VmProgram; memF32: var seq[float32];
                 nThreads: int; tgBytes = 4; tag = "mvm"; tgSize = 0): bool =
  ## Host-driven microVM: one session per call (begin→encode→end→read).
  ## Timeout retries apply only to the scratch buffer uploaded from the host here;
  ## the encode APIs working directly on resident recurrent state are NOT replayed.
  let be = ctx.kind.toByby()
  if be == byby.gbCpu:
    let ts = if tgSize > 0: tgSize else: max(1, min(nThreads, 256))
    let tf = max(0, tgBytes div 4)
    if not cpu.vmRun(program.words, memF32, @[], nThreads, tf, ts):
      raise newException(CatchableError, "CPU microVM failed: " & cpu.cpuLastErrorMessage())
    return true
  checkVmScratchCapacity(ctx, memF32.len, tag)
  let kn = kernelNameFor(ctx, program, tag)   # hash/JIT before sessionBegin
  for attempt in 0 .. 1:
    beginVmSession(be, "runMicroVm tag=" & tag)
    var h: byby.SessionHandle
    var uploaded = false
    var ended = false
    try:
      h = sessionUploadBorrowed(be, memF32)
      if not vmUploadSucceeded(be, h):
        raise newException(CatchableError,
          &"runMicroVm[{tag}]: Metal upload failed for {memF32.len} float32 " &
          &"({int64(memF32.len) * 4'i64} bytes); lower sequence/batch or memory use.")
      uploaded = true
      encodeMicroVm(ctx, program, h, nThreads, tgBytes, tag, tgSize, kn)
      finishVmSession(be, "runMicroVm tag=" & tag, ended)
      let output = sessionRead(be, h, memF32.len)
      sessionFree(be, h)
      uploaded = false
      memF32 = output
      return true
    except CatchableError as e:
      var timeoutDetail = ""
      if not ended:
        if not sessionEnd(be):
          timeoutDetail = sessionGetLastError(be)
          discard sessionRecoverQueue(be)
        ended = true
      if uploaded: sessionFree(be, h)
      if timeoutDetail.len > 0 and isVmGpuTimeout(timeoutDetail):
        if attempt == 0:
          stderr.writeLine("[microvm] timeout in " & tag &
            "; recovered Metal queue and retrying once from original host buffer")
          continue
        raise newException(CatchableError,
          "microVM " & tag & ": timeout persisted after one safe retry: " & timeoutDetail)
      if e.msg.startsWith(VmTimeoutRetryMarker):
        if attempt == 0:
          stderr.writeLine("[microvm] timeout in " & tag &
            "; retrying once from original host buffer")
          continue
        raise newException(CatchableError,
          "microVM " & tag & ": timeout persisted after one safe retry: " & e.msg)
      raise
  raise newException(CatchableError, "runMicroVm: unreachable retry state")

proc runMicroVmStepsSeparated(ctx: Backend; steps: openArray[MvmStep]; index: int;
                              memF32: var seq[float32]): bool =
  ## Last-resort GPU fallback for a timed-out multi-dispatch command buffer.
  ## Preserve stage order and the same bytecode, but give each stage its own
  ## completed command buffer. This is slower and host-round-trips scratch,
  ## but avoids replaying a whole poisoned/overlong command buffer or using CPU.
  if index >= steps.len: return true
  let step = steps[index]
  discard runMicroVm(ctx, step.prog, memF32, step.nThreads, step.tgBytes,
                     step.tag, step.tgSize)
  result = runMicroVmStepsSeparated(ctx, steps, index + 1, memF32)

proc runMicroVmChain*(ctx: Backend; steps: openArray[MvmStep]; memF32: var seq[float32]): bool =
  ## SEVERAL consecutive kernels on the SAME buffer in one session: upload once, read back once
  ## (no return to the host between stages). Used for the whole-layer Mamba2 forward/backward.
  ## If the command buffer times out, the whole chain is replayed from the original host scratch;
  ## not used for persistent recurrent state.
  let be = ctx.kind.toByby()
  if be == byby.gbCpu:
    for step in steps:
      let ts = if step.tgSize > 0: step.tgSize else: max(1, min(step.nThreads, 256))
      if not cpu.vmRun(step.prog.words, memF32, @[], step.nThreads,
                       max(0, step.tgBytes div 4), ts):
        raise newException(CatchableError, "CPU microVM chain failed: " & cpu.cpuLastErrorMessage())
    return true
  checkVmScratchCapacity(ctx, memF32.len, "stage chain")
  let names = steps.mapIt(kernelNameFor(ctx, it.prog, it.tag))   # before sessionBegin
  for attempt in 0 .. 1:
    beginVmSession(be, "runMicroVmChain")
    var h: byby.SessionHandle
    var uploaded = false
    var ended = false
    try:
      var pfT0 = getMonoTimeIfPf()
      h = sessionUpload(be, memF32)
      if not vmUploadSucceeded(be, h):
        raise newException(CatchableError,
          &"runMicroVmChain: Metal upload failed for {memF32.len} float32 " &
          &"({int64(memF32.len) * 4'i64} bytes); lower sequence/batch or memory use.")
      uploaded = true
      pfAddIf("vm.upload", pfT0)
      pfT0 = getMonoTimeIfPf()
      encodeSteps(ctx, steps, names, h)
      finishVmSession(be, "runMicroVmChain", ended)
      pfAddIf("vm.encode+gpu", pfT0)
      pfT0 = getMonoTimeIfPf()
      let output = sessionRead(be, h, memF32.len)
      pfAddIf("vm.read", pfT0)
      sessionFree(be, h)
      uploaded = false
      memF32 = output
      return true
    except CatchableError as e:
      var timeoutDetail = ""
      if not ended:
        if not sessionEnd(be):
          timeoutDetail = sessionGetLastError(be)
          discard sessionRecoverQueue(be)
        ended = true
      if uploaded: sessionFree(be, h)
      if timeoutDetail.len > 0 and isVmGpuTimeout(timeoutDetail):
        if attempt == 0:
          stderr.writeLine("[microvm] timeout in Mamba/attention stage chain; " &
            "recovered Metal queue and retrying once from original scratch")
          continue
        stderr.writeLine("[microvm] combined chain timed out twice; replaying the " &
          "same GPU stages in separately synchronized command buffers")
        return runMicroVmStepsSeparated(ctx, steps, 0, memF32)
      if e.msg.startsWith(VmTimeoutRetryMarker):
        if attempt == 0:
          stderr.writeLine("[microvm] timeout in Mamba/attention stage chain; " &
            "retrying once from original scratch")
          continue
        stderr.writeLine("[microvm] combined chain timed out twice; replaying the " &
          "same GPU stages in separately synchronized command buffers")
        return runMicroVmStepsSeparated(ctx, steps, 0, memF32)
      raise
  raise newException(CatchableError, "runMicroVmChain: unreachable retry state")

proc runMicroVmChainGroups*(ctx: Backend; groups: openArray[seq[MvmStep]];
                            memF32: var seq[float32]): bool =
  ## Calls runMicroVmChain for each group in turn, but the buffer is uploaded only once
  ## and read back once. Each group is still its own command buffer (the GPU time per
  ## command buffer stays as before -> no higher watchdog-timeout risk).
  ## An error/timeout in any group -> the GPU result is dropped and every group is rerun FROM THE START
  ## with runMicroVmChain from the original host copy (the kernels accumulate atomically, no replay in the middle).
  ## BYBY_VM_GROUPS=0 -> the old path (one upload/read per group).
  if getEnv("BYBY_VM_GROUPS", "1") == "0" or groups.len <= 1:
    for g in groups:
      discard runMicroVmChain(ctx, g, memF32)
    return true
  let be = ctx.kind.toByby()
  if be == byby.gbCpu:
    for g in groups:
      for step in g:
        let ts = if step.tgSize > 0: step.tgSize else: max(1, min(step.nThreads, 256))
        if not cpu.vmRun(step.prog.words, memF32, @[], step.nThreads,
                         max(0, step.tgBytes div 4), ts):
          raise newException(CatchableError, "CPU grouped microVM chain failed: " & cpu.cpuLastErrorMessage())
    return true
  checkVmScratchCapacity(ctx, memF32.len, "stage chain groups")
  var names: seq[seq[string]]
  for g in groups:
    names.add g.mapIt(kernelNameFor(ctx, it.prog, it.tag))   # before sessionBegin
  var h: byby.SessionHandle
  var uploaded = false
  var ended = true
  var ok = false
  var output: seq[float32]
  try:
    var pfT0 = getMonoTimeIfPf()
    h = sessionUpload(be, memF32)
    if not vmUploadSucceeded(be, h):
      raise newException(CatchableError,
        &"runMicroVmChainGroups: Metal upload failed for {memF32.len} float32")
    uploaded = true
    pfAddIf("vm.upload", pfT0)
    pfT0 = getMonoTimeIfPf()
    for gi in 0 ..< groups.len:
      beginVmSession(be, "runMicroVmChainGroups")
      ended = false
      encodeSteps(ctx, groups[gi], names[gi], h)
      finishVmSession(be, "runMicroVmChainGroups", ended)
    pfAddIf("vm.encode+gpu", pfT0)
    pfT0 = getMonoTimeIfPf()
    output = sessionRead(be, h, memF32.len)
    pfAddIf("vm.read", pfT0)
    ok = output.len == memF32.len
  except CatchableError as e:
    var timeoutDetail = ""
    if not ended:
      if not sessionEnd(be):
        timeoutDetail = sessionGetLastError(be)
        discard sessionRecoverQueue(be)
      ended = true
    if uploaded:
      sessionFree(be, h)
      uploaded = false
    # Like runMicroVmChain: ONLY a GPU timeout reruns; other errors raise right away.
    if not (e.msg.startsWith(VmTimeoutRetryMarker) or
            (timeoutDetail.len > 0 and isVmGpuTimeout(timeoutDetail))):
      raise
    stderr.writeLine("[microvm] chain groups timeout (" & e.msg &
      "); rerunning each group from the original host copy")
  if uploaded: sessionFree(be, h)
  if ok:
    memF32 = output
    return true
  for g in groups:
    discard runMicroVmChain(ctx, g, memF32)
  true

proc runMicroVmChainSparse*(ctx: Backend; steps: openArray[MvmStep]; totalLen: int;
                            head: seq[float32];
                            reads: openArray[tuple[off, len: int]]): seq[seq[float32]] =
  ## Like runMicroVmChain but WITHOUT passing the whole buffer through the host: allocates a GPU scratch of
  ## totalLen floats (a newly allocated Metal/Vulkan buffer = 0), uploads only `head` into
  ## [0, head.len), runs the chain, then reads back only the `reads` slices (gathered into one small
  ## buffer with device-to-device copies). Used for inference prefill: the Mamba2 scratch is
  ## ~57MB/block but the host only needs normedOut + the final state (~4MB).
  ## The caller MUST guarantee: every region >= head.len that a kernel READS is WRITTEN by the chain
  ## itself first (no accumulation/atomics). An error/timeout -> raises (no retry); the caller
  ## falls back to runMicroVmChain itself.
  let be = ctx.kind.toByby()
  if be == byby.gbCpu:
    var mem = newSeq[float32](totalLen)
    for i in 0 ..< head.len: mem[i] = head[i]
    for step in steps:
      let ts = if step.tgSize > 0: step.tgSize else: max(1, min(step.nThreads, 256))
      if not cpu.vmRun(step.prog.words, mem, @[], step.nThreads,
                       max(0, step.tgBytes div 4), ts):
        raise newException(CatchableError, "CPU sparse microVM chain failed: " & cpu.cpuLastErrorMessage())
    for r in reads:
      result.add mem[r.off ..< r.off + r.len]
    return
  checkVmScratchCapacity(ctx, totalLen, "sparse stage chain")
  doAssert head.len > 0 and head.len <= totalLen
  var outLen = 0
  for r in reads:
    doAssert r.off >= 0 and r.len > 0 and r.off + r.len <= totalLen
    outLen += r.len
  let names = steps.mapIt(kernelNameFor(ctx, it.prog, it.tag))   # before sessionBegin
  beginVmSession(be, "runMicroVmChainSparse")
  var hHead, hMem, hOut: byby.SessionHandle
  var ended = false
  try:
    var pfT0 = getMonoTimeIfPf()
    hHead = sessionUpload(be, head)
    if not vmUploadSucceeded(be, hHead):
      raise newException(CatchableError, "runMicroVmChainSparse: upload head failed")
    hMem = sessionAllocScratch(be, totalLen)
    hOut = sessionAllocScratch(be, outLen)
    if not vmUploadSucceeded(be, hMem) or not vmUploadSucceeded(be, hOut):
      raise newException(CatchableError, "runMicroVmChainSparse: alloc scratch failed")
    if not sessionCopyInto(be, hHead, hMem, 0, 0, head.len):
      raise newException(CatchableError, "runMicroVmChainSparse: copy head failed")
    pfAddIf("vm.upload", pfT0)
    pfT0 = getMonoTimeIfPf()
    encodeSteps(ctx, steps, names, hMem)
    var o = 0
    for r in reads:
      if not sessionCopyInto(be, hMem, hOut, r.off, o, r.len):
        raise newException(CatchableError, "runMicroVmChainSparse: copy read-back failed")
      o += r.len
    finishVmSession(be, "runMicroVmChainSparse", ended)
    pfAddIf("vm.encode+gpu", pfT0)
    pfT0 = getMonoTimeIfPf()
    let flat = sessionRead(be, hOut, outLen)
    pfAddIf("vm.read", pfT0)
    o = 0
    for r in reads:
      result.add flat[o ..< o + r.len]
      o += r.len
  finally:
    if not ended:
      if not sessionEnd(be): discard sessionRecoverQueue(be)
    sessionFree(be, hOut)
    sessionFree(be, hMem)
    sessionFree(be, hHead)

proc axpyGpu*(ctx: Backend; dst: var seq[float32]; src: openArray[float32]; scale: float32) =
  ## dst[i] += scale * src[i] on the GPU (replaces LoRA's `for i: dst[i] += src[i]*s` loop)
  let n = min(dst.len, src.len)
  if n == 0: return
  var buf = newSeq[float32](2 * n)
  buf.blit(0, dst, 0, n); buf.blit(n, src, 0, n)
  discard runMicroVm(ctx, buildAxpyProgram(0, n, scale), buf, n, 4, "axpy")
  dst.blit(0, buf, 0, n)

proc buildShapeInfo(ctx: Backend; H, D, N, G: int): Mamba2SsmShapeInfo =
  let HD = H * D
  result.offX = 0
  result.offB = HD
  result.offC = result.offB + G * N
  result.offDt = result.offC + G * N
  result.offALog = result.offDt + H
  result.offDSkip = result.offALog + H
  result.offState = result.offDSkip + H
  result.offY = result.offState + HD * N
  result.offIdxH = result.offY + HD
  result.offIdxG = result.offIdxH + HD
  result.totalLen = result.offIdxG + HD
  # idxH/idxG generated on the GPU (buildIdxTableProgram) — no more for h/for d on the CPU
  var tab = newSeq[float32](2 * HD)
  discard runMicroVm(ctx, buildIdxTableProgram(H, D, G), tab, HD, 4, "idxtab")
  result.idxH = tab[0 ..< HD]
  result.idxG = tab[HD ..< 2 * HD]
  result.program = buildMamba2StepProgram(H, D, N, G)

proc getShapeInfo(ctx: Backend; H, D, N, G: int): Mamba2SsmShapeInfo =
  let key = shapeKey(H, D, N, G)
  if key in gShapeCache: return gShapeCache[key]
  result = buildShapeInfo(ctx, H, D, N, G)
  gShapeCache[key] = result

proc initMamba2GpuLayer*(ctx: Backend; H, D, N, G, C, cDim, K, hidden: int;
                         aLog, dSkip, dtBias, normW, convW, convB: seq[float32]): Mamba2GpuLayer =
  let be = ctx.kind.toByby()
  let HD = H * D
  let off = computeOff(C, cDim, H, D, N, G, K)
  let info = getShapeInfo(ctx, H, D, N, G)
  var buf = newSeq[float32](off.total)
  buf.blit(off.offALog, aLog, 0, H)
  buf.blit(off.offDSkip, dSkip, 0, H)
  buf.blit(off.offDtBias, dtBias, 0, H)
  buf.blit(off.offNormW, normW, 0, C)
  buf.blit(off.offConvW, convW, 0, convW.len)
  buf.blit(off.offConvB, convB, 0, convB.len)
  buf.blit(off.offIdxH, info.idxH, 0, HD)
  buf.blit(off.offIdxG, info.idxG, 0, HD)
  result.be = be
  result.off = off
  result.H = H; result.D = D; result.N = N; result.G = G
  result.C = C; result.cDim = cDim; result.K = K; result.HD = HD
  result.hidden = hidden
  result.hWork = byby.sessionUploadPrivate(be, buf)
  let convPack = cDim + cDim * max(K - 1, 0) + cDim * K + cDim + cDim
  let ssmPack = info.totalLen
  result.hScratch = sessionAllocScratch(be, max(convPack + ssmPack + 4 * C, 1))
  result.ready = true

proc ensureGpuLayer*(ctx: Backend; blockIdx, H, D, N, G, C, cDim, K, hidden: int;
                     aLog, dSkip, dtBias, normW, convW, convB: seq[float32]): Mamba2GpuLayer =
  if blockIdx notin gGpuLayers:
    gGpuLayers[blockIdx] = initMamba2GpuLayer(ctx, H, D, N, G, C, cDim, K, hidden,
      aLog, dSkip, dtBias, normW, convW, convB)
  result = gGpuLayers[blockIdx]

proc resetGpuLayerState*(ctx: Backend; blockIdx: int): bool =
  ## FIX for garbage tokens (bug #12): zeroes the RESIDENT STATE of one layer
  ## (convState + ssmState in hWork). Must be called after warmup, because
  ## warmupJambaKernelsOnce runs encodeLayerStepByIdx ON the REAL
  ## gGpuLayers[0] itself (not a copy): with x=0, proj=0 so z=0 and
  ## xBC=0, BUT conv1d still gives silu(convBias) != 0 -> block 0's ssmState
  ## got values from a FAKE token before prefill started. Blocks
  ## 1..63 were clean -> an asymmetric corruption, very hard to spot.
  ## MUST be called INSIDE an open session (uses sessionCopyInto like every other
  ## step in this file).
  if blockIdx notin gGpuLayers: return false
  var layer = gGpuLayers[blockIdx]   # a copy of the struct; hWork is a handle, so it still points to the same MTLBuffer
  if not layer.ready: return false
  let be = layer.be
  let o = layer.off
  let nConv = layer.cDim * max(layer.K - 1, 0)
  let nSsm = layer.HD * layer.N
  let nMax = max(nConv, nSsm)
  if nMax <= 0: return true
  # uploads REAL zeros (does not rely on a new buffer being zeroed)
  let hZero = sessionUpload(be, newSeq[float32](nMax))
  result = true
  if nConv > 0:
    if not sessionCopyInto(be, hZero, layer.hWork, 0, o.offConvSt, nConv): result = false
  if nSsm > 0:
    if not sessionCopyInto(be, hZero, layer.hWork, 0, o.offSsm, nSsm): result = false

proc mamba2StepGpu*(ctx: Backend; H, D, N, G: int;
                    xPart, bMat, cMat, dt, aLogArr, dSkipArr: seq[float32];
                    ssmStateInOut: var seq[float32]): seq[float32] =
  let info = getShapeInfo(ctx, H, D, N, G)
  let HD = H * D
  var buf = newSeq[float32](info.totalLen)
  buf.blit(info.offX, xPart, 0, HD)
  buf.blit(info.offB, bMat, 0, G * N)
  buf.blit(info.offC, cMat, 0, G * N)
  buf.blit(info.offDt, dt, 0, H)
  buf.blit(info.offALog, aLogArr, 0, H)
  buf.blit(info.offDSkip, dSkipArr, 0, H)
  buf.blit(info.offState, ssmStateInOut, 0, HD * N)
  buf.blit(info.offIdxH, info.idxH, 0, HD)
  buf.blit(info.offIdxG, info.idxG, 0, HD)
  discard runMicroVm(ctx, info.program, buf, HD, 4, shapeKey(H, D, N, G))
  ssmStateInOut.blit(0, buf, info.offState, HD * N)
  result = newSeq[float32](HD)
  result.blit(0, buf, info.offY, HD)

proc mamba2StepGpuFromXbc*(ctx: Backend; H, D, N, G, C: int;
                           xBCConv, dt, aLogArr, dSkipArr: seq[float32];
                           ssmStateInOut: var seq[float32]): seq[float32] =
  var bMat = newSeq[float32](G * N)
  var cMat = newSeq[float32](G * N)
  var xP = newSeq[float32](C)
  if C > 0: copyMem(addr xP[0], unsafeAddr xBCConv[0], C * sizeof(float32))
  if G * N > 0:
    copyMem(addr bMat[0], unsafeAddr xBCConv[C], G * N * sizeof(float32))
    copyMem(addr cMat[0], unsafeAddr xBCConv[C + G * N], G * N * sizeof(float32))
  mamba2StepGpu(ctx, H, D, N, G, xP, bMat, cMat, dt, aLogArr, dSkipArr, ssmStateInOut)

proc softplusDtGpu*(ctx: Backend; H: int; dtPre, dtBias: seq[float32]): seq[float32] =
  var buf = newSeq[float32](3 * H)
  buf.blit(0, dtPre, 0, H)
  buf.blit(H, dtBias, 0, H)
  discard runMicroVm(ctx, buildSoftplusDtProgram(H), buf, H, 4, "softplus")
  result = newSeq[float32](H)
  result.blit(0, buf, 2 * H, H)

proc conv1dIncGpu*(ctx: Backend; cDim, K: int; xBCPre: seq[float32];
                   convState: var seq[float32]; convWeight, convBias: seq[float32]): seq[float32] =
  let offState = cDim
  let offW = offState + cDim * (K - 1)
  let offBias = offW + cDim * K
  let offOut = offBias + cDim
  var buf = newSeq[float32](offOut + cDim)
  buf.blit(0, xBCPre, 0, cDim)
  buf.blit(offState, convState, 0, convState.len)
  buf.blit(offW, convWeight, 0, convWeight.len)
  buf.blit(offBias, convBias, 0, convBias.len)
  discard runMicroVm(ctx, buildConv1dIncProgram(cDim, K), buf, cDim, 4, "conv")
  result = newSeq[float32](cDim)
  result.blit(0, buf, offOut, cDim)
  convState.blit(0, buf, offState, convState.len)

proc splitProjGpu*(ctx: Backend; C, cDim, H: int; proj: seq[float32]):
    tuple[z, xBC, dt: seq[float32]] =
  let Ptot = C + cDim + H
  var buf = newSeq[float32](Ptot + C + cDim + H)
  buf.blit(0, proj, 0, Ptot)
  discard runMicroVm(ctx, buildSplitProjProgram(C, cDim, H), buf, max(C, max(cDim, H)), 4, "split")
  result.z = newSeq[float32](C)
  result.xBC = newSeq[float32](cDim)
  result.dt = newSeq[float32](H)
  result.z.blit(0, buf, Ptot, C)
  result.xBC.blit(0, buf, Ptot + C, cDim)
  result.dt.blit(0, buf, Ptot + C + cDim, H)

proc rmsGateGpu*(ctx: Backend; C: int; eps: float32; yPre, z, w: seq[float32];
                 nGroups = 1): seq[float32] =
  if nGroups <= 0 or C mod nGroups != 0:
    raise newException(ValueError, "rmsGateGpu: C must be divisible by nGroups")
  var buf = newSeq[float32](4 * C)
  buf.blit(0, yPre, 0, C); buf.blit(C, z, 0, C); buf.blit(2 * C, w, 0, C)
  let groupLen = C div nGroups
  let tgN = rmsTgSizeFor(ctx, groupLen)
  let prog = buildRmsGateGroupedAt(C, nGroups, tgN, eps, 0, C, 2 * C, 3 * C)
  discard runMicroVm(ctx, prog, buf, nGroups * tgN, tgN * 4,
                     "rmsg_" & $C & "_" & $nGroups, tgN)
  result = newSeq[float32](C)
  result.blit(0, buf, 3 * C, C)

proc rmsGateGpuLarge*(ctx: Backend; C: int; eps: float32; yPre, z, w: seq[float32];
                      nGroups = 1): seq[float32] =
  rmsGateGpu(ctx, C, eps, yPre, z, w, nGroups)

var gPatchMarkerV3Printed = false

proc encodeLayerStep*(ctx: Backend; layer: var Mamba2GpuLayer;
                      hX: byby.SessionHandle;
                      fp4InName, fp4OutName: string;
                      hResidual: byby.SessionHandle;
                      hOut: var byby.SessionHandle) =
  if not gPatchMarkerV3Printed:
    stderr.writeLine "[PATCH_MARKER_V3] encodeLayerStep FIX (convState slide) RUNNING"
    gPatchMarkerV3Printed = true
  doAssert layer.ready
  let be = layer.be
  let o = layer.off
  let H = layer.H; let D = layer.D; let N = layer.N; let G = layer.G
  let C = layer.C; let cDim = layer.cDim; let K = layer.K; let HD = layer.HD

  var hProj: byby.SessionHandle
  if not fp4GemvGpuInto(ctx, fp4InName, hX, hProj, 1):
    raise newException(CatchableError, "fp4 in failed")
  if not sessionCopyInto(be, hProj, layer.hWork, 0, 0, o.P):
    raise newException(CatchableError, "copy proj failed")

  # CHANGED (merged kernels - following fusedNormQKVRoPE/fusedResidualNormFFN of
  # nim_inference.nim, whose REAL 6.7B model does not lag on the same hardware, using the
  # EXISTING micro-VM - NO hand-written .metal): BEFORE, these were 3 separate
  # dispatches (split/conv/softplus) + 4 separate sessionCopyInto moving xBC/dt
  # back and forth between buffers per layer - per layer x 64 layers x every token that is a great
  # many small pipeline binds/dispatches, the real source of the system
  # lag (not commitContinue/QoS/yield - those only patch
  # symptoms). NOW merged into ONE dispatch through
  # buildFusedSplitConvSoftplusProgram - SAFE to merge because the conv1d here is
  # depthwise in time (see the full explanation in that function), no
  # thread needs another thread's data, so no threadgroup_barrier is needed.
  var hScratch = layer.hScratch
  let fP = C + cDim + H
  let fOffOutZ = fP
  let fOffConvSt = fOffOutZ + C
  let fOffConvW = fOffConvSt + cDim * (K - 1)
  let fOffConvB = fOffConvW + cDim * K
  let fOffOutXBC = fOffConvB + cDim
  let fOffDtBias = fOffOutXBC + cDim
  let fOffOutDt = fOffDtBias + H

  discard sessionCopyInto(be, layer.hWork, hScratch, 0, 0, o.P)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offConvSt, fOffConvSt, cDim * (K - 1))
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offConvW, fOffConvW, cDim * K)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offConvB, fOffConvB, cDim)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offDtBias, fOffDtBias, H)
  encodeMicroVm(ctx, buildFusedSplitConvSoftplusProgram(C, cDim, H, K), hScratch,
                max(C, max(cDim, H)), 4, "fusedsplitconv")
  discard sessionCopyInto(be, hScratch, layer.hWork, fOffOutZ, o.offZ, C)
  discard sessionCopyInto(be, hScratch, layer.hWork, fOffOutXBC, o.offXBC, cDim)
  discard sessionCopyInto(be, hScratch, layer.hWork, fOffOutDt, o.offDt, H)
  # CHANGED (fix for the frozen conv1d bug - see the debug report): BEFORE, the comment here
  # said "convState is NOT written back, the original kernel does not update
  # it either" - WRONG: reading buildConv1dIncProgram again (lines 194-206), the ORIGINAL DID
  # have a "slide state" part updating convState, it was only DROPPED when merged into
  # buildFusedSplitConvSoftplusProgram. The offOutConvSt output was added back to
  # that kernel - copied back to o.offConvSt HERE so convState REALLY slides over
  # each token (before it froze after the first step, corrupting the local signal
  # fed to the SSM of every layer, every token).
  let fOffOutConvSt = fOffOutDt + H
  discard sessionCopyInto(be, hScratch, layer.hWork, fOffOutConvSt, o.offConvSt, cDim * (K - 1))

  # OPTIMISATION (merged dispatches, see getSsmAtProgram above): the SSM scan runs DIRECTLY
  # on hWork through buildMamba2StepAt, no more hScratch - drops 9
  # sessionCopyInto per layer per token, of which the 2 HD*N state copies (the largest
  # buffer of the step) were the most expensive. dtEnc: 1 dispatch, 0 extra copies.
  let ssmProg = getSsmAtProgram(H, D, N, G, C, o)
  encodeMicroVm(ctx, ssmProg, layer.hWork, HD, 4, shapeKey(H, D, N, G) & "_at")

  discard sessionCopyInto(be, layer.hWork, hScratch, o.offY, 0, C)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offZ, C, C)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offNormW, 2 * C, C)
  let tgN = rmsTgSizeFor(ctx, C)   # FIX token rac (bug #9) - xem rmsTgSizeFor
  if C <= tgN and (C and (C - 1)) == 0:
    encodeMicroVm(ctx, buildRmsGateProgram(C, 1e-5'f32), hScratch, C, C * 4, "rms")
  elif C mod tgN == 0:
    encodeMicroVm(ctx, buildRmsGateLargeProgram(C, 1e-5'f32, tgN), hScratch, tgN, tgN * 4, "rmsL")
  else:
    raise newException(CatchableError, "bad C for rms")
  discard sessionCopyInto(be, hScratch, layer.hWork, 3 * C, o.offNormed, C)

  var hNorm = sessionAllocScratch(be, C)
  discard sessionCopyInto(be, layer.hWork, hNorm, o.offNormed, 0, C)
  var hMix: byby.SessionHandle
  if fp4OutName.len > 0:
    if not fp4GemvGpuInto(ctx, fp4OutName, hNorm, hMix, 1):
      raise newException(CatchableError, "fp4 out failed")
  else:
    hMix = hNorm
  hOut = sessionAllocScratch(be, layer.hidden)
  if not sessionVecOp(be, "add", hResidual, hMix, hOut, layer.hidden):
    raise newException(CatchableError, "residual add failed")

proc encodeLayerStepByIdx*(ctx: Backend; blockIdx: int;
                           hX: byby.SessionHandle;
                           fp4InName, fp4OutName: string;
                           hResidual: byby.SessionHandle;
                           hOut: var byby.SessionHandle) =
  doAssert blockIdx in gGpuLayers
  encodeLayerStep(ctx, gGpuLayers[blockIdx], hX, fp4InName, fp4OutName, hResidual, hOut)

proc profileLayerStepByIdx*(ctx: Backend; blockIdx: int;
                            hX: byby.SessionHandle;
                            fp4InName, fp4OutName: string;
                            hResidual: byby.SessionHandle): byby.SessionHandle =
  ## NEW - FOR MEASUREMENT ONLY (runs ONCE, NOT for the normal hot path):
  ## repeats EXACTLY the steps of encodeLayerStep, but inserts a
  ## sessionEnd() (a REAL wait, measuring REAL GPU time) + sessionBegin() again between
  ## EVERY step - the ONLY way to measure the REAL GPU time of each
  ## kernel, because CPU timestamps around encode calls reflect nothing
  ## (commit is asynchronous - the GPU may not have run anything when the CPU "finished encoding").
  ## It is accepted to run MUCH SLOWER than normal (each step adds a REAL
  ## waitUntilCompleted, the same reason the earlier flush-every-N-layers was slow)
  ## - a DELIBERATE trade-off, run only ONCE to get real numbers,
  ## not the production path. Enable with BYBY_JAMBA_PROFILE_STEPS=1.
  doAssert blockIdx in gGpuLayers
  var layer = gGpuLayers[blockIdx]
  doAssert layer.ready
  let be = layer.be
  let o = layer.off
  let H = layer.H; let D = layer.D; let N = layer.N; let G = layer.G
  let C = layer.C; let cDim = layer.cDim; let K = layer.K; let HD = layer.HD

  proc lap(label: string) =
    discard sessionEnd(be)   # a REAL wait - the GPU has finished the previous step
    let t = epochTime()
    stderr.writeLine &"[PROFILE block{blockIdx}] {label}: {(t - gProfileLastT)*1000:.1f} ms"
    gProfileLastT = t
    discard sessionBegin(be)  # reopened RIGHT AWAY - the handles (hWork/hScratch...) stay alive, nothing lost

  gProfileLastT = epochTime()
  discard sessionBegin(be)

  var hProj: byby.SessionHandle
  if not fp4GemvGpuInto(ctx, fp4InName, hX, hProj, 1):
    raise newException(CatchableError, "fp4 in failed")
  if not sessionCopyInto(be, hProj, layer.hWork, 0, 0, o.P):
    raise newException(CatchableError, "copy proj failed")
  lap("fp4_in (" & fp4InName & ")")

  encodeMicroVm(ctx, buildSplitProjProgram(C, cDim, H), layer.hWork,
                max(C, max(cDim, H)), 4, "split")
  lap("split")

  var hScratch = layer.hScratch
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offXBC, 0, cDim)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offConvSt, cDim, cDim * (K - 1))
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offConvW, cDim + cDim * (K - 1), cDim * K)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offConvB, cDim + cDim * (K - 1) + cDim * K, cDim)
  encodeMicroVm(ctx, buildConv1dIncProgram(cDim, K), hScratch, cDim, 4, "conv")
  let offConvOut = cDim + cDim * (K - 1) + cDim * K + cDim
  discard sessionCopyInto(be, hScratch, layer.hWork, offConvOut, o.offXBC, cDim)
  discard sessionCopyInto(be, hScratch, layer.hWork, cDim, o.offConvSt, cDim * (K - 1))
  lap("conv")

  discard sessionCopyInto(be, layer.hWork, hScratch, o.offDt, 0, H)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offDtBias, H, H)
  encodeMicroVm(ctx, buildSoftplusDtProgram(H), hScratch, H, 4, "softplus")
  discard sessionCopyInto(be, hScratch, layer.hWork, 2 * H, o.offDt, H)
  lap("softplus")

  let info = getShapeInfo(ctx, H, D, N, G)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offXBC, info.offX, C)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offXBC + C, info.offB, G * N)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offXBC + C + G * N, info.offC, G * N)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offDt, info.offDt, H)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offALog, info.offALog, H)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offDSkip, info.offDSkip, H)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offSsm, info.offState, HD * N)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offIdxH, info.offIdxH, HD)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offIdxG, info.offIdxG, HD)
  encodeMicroVm(ctx, info.program, hScratch, HD, 4, shapeKey(H, D, N, G))
  discard sessionCopyInto(be, hScratch, layer.hWork, info.offState, o.offSsm, HD * N)
  discard sessionCopyInto(be, hScratch, layer.hWork, info.offY, o.offY, min(C, HD))
  lap("ssm_step (HD=" & $HD & " x N=" & $N & ")")

  discard sessionCopyInto(be, layer.hWork, hScratch, o.offY, 0, C)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offZ, C, C)
  discard sessionCopyInto(be, layer.hWork, hScratch, o.offNormW, 2 * C, C)
  let tgN = rmsTgSizeFor(ctx, C)   # FIX token rac (bug #9) - xem rmsTgSizeFor
  if C <= tgN and (C and (C - 1)) == 0:
    encodeMicroVm(ctx, buildRmsGateProgram(C, 1e-5'f32), hScratch, C, C * 4, "rms")
  elif C mod tgN == 0:
    encodeMicroVm(ctx, buildRmsGateLargeProgram(C, 1e-5'f32, tgN), hScratch, tgN, tgN * 4, "rmsL")
  else:
    raise newException(CatchableError, "bad C for rms")
  discard sessionCopyInto(be, hScratch, layer.hWork, 3 * C, o.offNormed, C)
  lap("rmsnorm_gate")

  var hNorm = sessionAllocScratch(be, C)
  discard sessionCopyInto(be, layer.hWork, hNorm, o.offNormed, 0, C)
  var hMix: byby.SessionHandle
  if fp4OutName.len > 0:
    if not fp4GemvGpuInto(ctx, fp4OutName, hNorm, hMix, 1):
      raise newException(CatchableError, "fp4 out failed")
  else:
    hMix = hNorm
  lap("fp4_out (" & fp4OutName & ")")

  var hOut: byby.SessionHandle
  hOut = sessionAllocScratch(be, layer.hidden)
  if not sessionVecOp(be, "add", hResidual, hMix, hOut, layer.hidden):
    raise newException(CatchableError, "residual add failed")
  lap("residual_add")

  discard sessionEnd(be)
  result = hOut

proc fp4GemvGpuHandle*(ctx: Backend; name: string; x: seq[float32]; M: int): byby.SessionHandle =
  let be = ctx.kind.toByby()
  let hX = sessionUpload(be, x)
  doAssert name in gFp4CacheGpu
  result = sessionAllocScratch(be, M * gFp4CacheGpu[name].rows)
  discard fp4GemvGpuInto(ctx, name, hX, result, M)

# ═══════════════════════════════════════════════════════════════════════
# BACKWARD GPU FUNCTIONS FOR MAMBA2REAL
# ═══════════════════════════════════════════════════════════════════════

proc backwardRmsGateGpu*(ctx: Backend; C: int; eps: float32;
                          yPre, z, w, dNormedOut: seq[float32]):
    tuple[dYpre, dZ, dNormWeight: seq[float32]] =
  ## RMSNorm + SiLU gate backward - one micro-kernel, one dispatch. All the math
  ## (sigmoid, gated, 2 sum/dot reductions, dGated, dSilu...) runs on the GPU; the host only
  ## copies data into/out of the buffer (copyMem).
  let tgN = rmsBwdTgSize(C)
  var buf = newSeq[float32](7 * C)
  buf.blit(0, yPre, 0, C); buf.blit(C, z, 0, C)
  buf.blit(2 * C, w, 0, C); buf.blit(3 * C, dNormedOut, 0, C)
  discard runMicroVm(ctx, buildRmsGateBackwardProgram(C, eps, tgN), buf, tgN, tgN * 4, "rms_bwd")
  result.dYpre = buf[4 * C ..< 5 * C]
  result.dZ = buf[5 * C ..< 6 * C]
  result.dNormWeight = buf[6 * C ..< 7 * C]

proc backwardSoftplusDtGpu*(ctx: Backend; H: int; dtPre, dtBias, dDt: seq[float32]):
    tuple[dDtPre, dDtBias: seq[float32]] =
  ## Backward for dt = softplus(dtPre + dtBias)
  ## dDtPre = dDt * sigmoid(dtPre + dtBias)
  ## dDtBias = dDtPre (same gradient)
  var pre = beAdd(ctx, dtPre, dtBias)
  var sigmoidPre = beSigmoid(ctx, pre)
  var dDtPre = beMul(ctx, dDt, sigmoidPre)
  result = (dDtPre, dDtPre)

# ═══════════════════════════════════════════════════════════════════════
# SELECTIVE SCAN BACKWARD - FULL GPU MICROVM KERNEL (ALL TIMESTEPS)
# ═══════════════════════════════════════════════════════════════════════

proc selectiveScanBackwardGpu*(ctx: Backend; B, T, H, D, N, G: int;
                                xPart, bMat, cMat, dt, aLog, dSkip,
                                states, dYpre: seq[float32]):
    tuple[dXPart, dBMat, dCMat, dDt, dALog, dDSkip: seq[float32]] =
  ## GPU-accelerated backward selective scan using microVM kernel.
  ## Runs 100% on GPU - single dispatch for ALL timesteps.
  
  let C = H * D
  let HD = H * D
  
  var dXPart = newSeq[float32](B * T * C)
  var dBMat = newSeq[float32](B * T * G * N)
  var dCMat = newSeq[float32](B * T * G * N)
  var dDt = newSeq[float32](B * T * H)
  var dALog = newSeq[float32](H)
  var dDSkip = newSeq[float32](H)


  # Build buffer with ALL data for single kernel dispatch
  # Layout:
  #   offX      : xPart[B*T*HD]
  #   offB      : bMat[B*T*G*N]
  #   offC      : cMat[B*T*G*N]
  #   offDt     : dt[B*T*H]
  #   offALog   : aLog[H]
  #   offDSkip  : dSkip[H]
  #   offState  : states[B*(T+1)*HD*N]
  #   offDY     : dYpre[B*T*HD]
  #   offIdxH   : idxH[HD]
  #   offIdxG   : idxG[HD]
  #   offDX     : dXPart[B*T*HD] (output)
  #   offDB     : dBMat[B*T*G*N] (output, atomic)
  #   offDC     : dCMat[B*T*G*N] (output, atomic)
  #   offDDt    : dDt[B*T*H] (output, atomic)
  #   offDALog  : dALog[H] (output, atomic)
  #   offDDSkip : dDSkip[H] (output, atomic)
  
  let offX = 0
  let offB = B * T * HD
  let offC = offB + B * T * G * N
  let offDt = offC + B * T * G * N
  let offALog = offDt + B * T * H
  let offDSkip = offALog + H
  let offState = offDSkip + H
  let offDY = offState + B * (T + 1) * HD * N
  let offDX = offDY + B * T * HD
  let offDB = offDX + B * T * HD
  let offDC = offDB + B * T * G * N
  let offDDt = offDC + B * T * G * N
  let offDALog = offDDt + B * T * H
  let offDDSkip = offDALog + H
  let totalSize = offDDSkip + H
  
  var buf = newSeq[float32](totalSize)
  
  # Copy ALL input data (no CPU loops per timestep)
  # xPart
  buf.blit(offX, xPart, 0, B * T * HD)
  # bMat
  buf.blit(offB, bMat, 0, B * T * G * N)
  # cMat
  buf.blit(offC, cMat, 0, B * T * G * N)
  # dt
  buf.blit(offDt, dt, 0, B * T * H)
  # aLog
  buf.blit(offALog, aLog, 0, H)
  # dSkip
  buf.blit(offDSkip, dSkip, 0, H)
  # states
  buf.blit(offState, states, 0, B * (T + 1) * HD * N)
  # dYpre
  buf.blit(offDY, dYpre, 0, B * T * HD)
  # Atomically accumulated gradients need zeros: newSeq zero-initialises, the kernel overwrites dX itself.
  # h/g are derived RIGHT IN THE KERNEL (no more idxH/idxG tables precomputed on the CPU).
  
  # Build and run the FULL backward kernel (single dispatch)
  let prog = buildMamba2BackwardFullAt(B, T, H, D, N, G,
    offX, offB, offC, offDt, offALog, offDSkip,
    offState, offDY,
    offDX, offDB, offDC, offDDt, offDALog, offDDSkip)
  
  discard runMicroVm(ctx, prog, buf, B * HD, 4, "ssm_backward_full")
  
  # Copy results back
  dXPart.blit(0, buf, offDX, B * T * HD)
  dBMat.blit(0, buf, offDB, B * T * G * N)
  dCMat.blit(0, buf, offDC, B * T * G * N)
  dDt.blit(0, buf, offDDt, B * T * H)
  dALog.blit(0, buf, offDALog, H)
  dDSkip.blit(0, buf, offDDSkip, H)

  result = (dXPart, dBMat, dCMat, dDt, dALog, dDSkip)

# ═════════════════════════════════════════════════════════════
# TRAINING KERNEL HOST WRAPPERS (RMSNorm, Linear++, softmax-CE, fused Adam)
# ═════════════════════════════════════════════════════════════
proc isGpuCtx*(ctx: Backend): bool =
  ctx.kind in {bkMetal, bkCuda, bkVulkan}

proc rmsNormForwardGpu*(ctx: Backend; x, gamma: seq[float32]; rows, C: int;
                        eps: float32): seq[float32] =
  ensureTgCap(ctx)
  let L = rmsLayout(rows, C, eps)
  var buf = newSeq[float32](L.total)
  buf.blit(L.x, x, 0, rows * C); buf.blit(L.gamma, gamma, 0, C)
  let tg = rmsRowTg(C)
  discard runMicroVmChain(ctx, [
    (buildRmsFwdProgram(L), rows * tg, tg * 4, tg, "@rms_fwd_" & $rows & "_" & $C & "_" & $eps & "_" & $tg)], buf)
  result = buf[L.y ..< L.y + rows * C]

proc rmsNormBackwardGpu*(ctx: Backend; x, gamma, dOut: seq[float32]; rows, C: int;
                         eps: float32): tuple[dX, dGamma: seq[float32]] =
  ensureTgCap(ctx)
  let L = rmsLayout(rows, C, eps)
  var buf = newSeq[float32](L.total)          # dGamma region = 0 for the atomics
  buf.blit(L.x, x, 0, rows * C); buf.blit(L.gamma, gamma, 0, C)
  buf.blit(L.dy, dOut, 0, rows * C)
  let tg = rmsRowTg(C)
  discard runMicroVmChain(ctx, [
    (buildRmsBwdProgram(L), rows * tg, tg * 4, tg, "@rms_bwd_" & $rows & "_" & $C & "_" & $eps & "_" & $tg)], buf)
  result = (buf[L.dx ..< L.dx + rows * C], buf[L.dGamma ..< L.dGamma + C])

proc lpTag(L: LpLay): string =
  $L.B & "_" & $L.T & "_" & $L.H & "_" & $L.D & "_" & $lpPhiTg(L)

proc linearPlusForwardSteps*(L: LpLay): seq[MvmStep] =
  ## Use bounded timestep-parallel prefix scans when T fits one legal
  ## threadgroup. Longer sequences (or BYBY_LP_SCAN_SERIAL=1) retain the
  ## original recurrence as a correctness/reference fallback.
  let tg = if getEnv("BYBY_LP_SCAN_SERIAL") == "1": 0 else: lpScanTgSize(L.T)
  let tag = lpTag(L)
  if tg <= 0:
    let phiTg = lpPhiTg(L)
    return @[
      (buildLpPhiProgram(L), L.BT * 2 * L.C, phiTg * 4, phiTg,
        "@lp_phi_" & tag),
      (buildLpDenProgram(L), L.B * L.H, 4, 0, "@lp_den_" & tag),
      (buildLpFwdProgram(L), L.B * L.C, 4, 0, "@lp_fwd_" & tag)]

  doAssert L.B > 0 and L.H > 0 and L.D > 0 and L.T > 0
  let nChunks = (L.D + lpScanStateChunkSize(L.D) - 1) div lpScanStateChunkSize(L.D)
  let groups = L.B * L.H * L.D
  doAssert groups <= high(int) div tg,
    "Linear++ denominator scan dispatch size overflow"
  doAssert groups <= high(int) div nChunks,
    "Linear++ state scan group count overflow"
  let stateGroups = groups * nChunks
  doAssert stateGroups <= high(int) div tg,
    "Linear++ state scan dispatch size overflow"
  let chunk = lpScanStateChunkSize(L.D)
  result = @[
    (buildLpDenScanParallelProgram(L, tg), groups * tg, 2 * tg * 4, tg,
      "@lp_den_scan_" & tag & "_tg" & $tg),
    (buildLpStateScanParallelProgram(L, tg, chunk), stateGroups * tg,
      2 * tg * chunk * 4, tg, "@lp_state_scan_" & tag & "_tg" & $tg),
    (buildLpFinalizeProgram(L), L.BT * L.C, 4, 0,
      "@lp_finalize_" & tag)]

proc linearPlusForwardGpu*(ctx: Backend; qkv: seq[float32]; B, T, H, D: int): seq[float32] =
  ensureTgCap(ctx)
  ## qkv: [B*T, 3C] (q|k|v per token). Returns y [B*T, C].
  let L = lpLayout(B, T, H, D)
  var buf = newSeq[float32](L.total)
  buf.blit(L.qkv, qkv, 0, L.BT * 3 * L.C)
  discard runMicroVmChain(ctx, linearPlusForwardSteps(L), buf)
  result = buf[L.y ..< L.y + L.BT * L.C]

proc linearPlusBackwardGpu*(ctx: Backend; qkv, dY: seq[float32]; B, T, H, D: int): seq[float32] =
  ensureTgCap(ctx)
  ## Rebuilds the forward in the same session, then a 2-pass backward. Returns dQKV [B*T, 3C]
  ## (exactly the dProj layout for qkvProj.backward).
  let L = lpLayout(B, T, H, D)
  var buf = newSeq[float32](L.total)          # scan state region = 0; dQKV is overwritten
  buf.blit(L.qkv, qkv, 0, L.BT * 3 * L.C)
  buf.blit(L.dY, dY, 0, L.BT * L.C)
  var steps = linearPlusForwardSteps(L)
  steps.add((buildLpDDenProgram(L), L.BT * L.H, 4, 0, "@lp_dden_" & lpTag(L)))
  steps.add((buildLpBwdQProgram(L), L.B * L.C, 4, 0, "@lp_bwd_q_" & lpTag(L)))
  steps.add((buildLpBwdKProgram(L), L.B * L.C, 4, 0, "@lp_bwd_k_" & lpTag(L)))
  steps.add((buildLpBwdVProgram(L), L.B * L.C, 4, 0, "@lp_bwd_v_" & lpTag(L)))
  discard runMicroVmChain(ctx, steps, buf)
  result = buf[L.dQKV ..< L.dQKV + L.BT * 3 * L.C]

proc softmaxCrossEntropyGpu*(ctx: Backend; logits: seq[float32]; targets: seq[float32];
                             rows, V, nValid: int): tuple[loss: float32, dLogits: seq[float32]] =
  ## targets[r] < 0 => skipped. nValid = number of valid rows (loss/gradient divided by nValid).
  if nValid == 0: return (0'f32, newSeq[float32](rows * V))
  ensureTgCap(ctx)
  let L = ceLayout(rows, V)
  var buf = newSeq[float32](L.total)
  buf.blit(L.logits, logits, 0, rows * V); buf.blit(L.tgt, targets, 0, rows)
  buf[L.scal] = 1'f32 / float32(nValid)
  let tg = rmsRowTg(V)
  discard runMicroVmChain(ctx, [
    (buildCeProgram(L), rows * tg, tg * 4, tg, "@ce_" & $rows & "_" & $V & "_" & $tg),
    (buildCeSumProgram(L), 1, 4, 1, "@ce_sum_" & $rows)], buf)
  result = (buf[L.sum], buf[L.dLogits ..< L.dLogits + rows * V])

const AdamChunk = 8 * 1024 * 1024     ## elements per dispatch (a multiple of AdamTg)

proc adamStepFusedGpu*(ctx: Backend; param: var seq[float32]; grad: seq[float32];
                       m, v: var seq[float32]; step: int; lr: float32;
                       b1: float32 = 0.9'f32; b2: float32 = 0.999'f32;
                       eps: float32 = 1e-8'f32) =
  ## Adam, 1 kernel per chunk (replacing ~14 separate be* ops). `step` is already incremented (>=1) before the call.
  let n = param.len
  let ibc1 = 1'f32 / (1'f32 - pow(b1, float32(step)))
  let ibc2 = 1'f32 / (1'f32 - pow(b2, float32(step)))
  var off = 0
  while off < n:
    let cnt = min(AdamChunk, n - off)
    let pad = ((cnt + AdamTg - 1) div AdamTg) * AdamTg
    let L = adamLayout(pad)
    var buf = newSeq[float32](L.total)
    buf.blit(L.p, param, off, cnt); buf.blit(L.g, grad, off, cnt)
    buf.blit(L.m, m, off, cnt); buf.blit(L.v, v, off, cnt)
    buf[L.scal] = lr; buf[L.scal + 1] = ibc1; buf[L.scal + 2] = ibc2
    discard runMicroVmChain(ctx, [
      (buildAdamProgram(L, b1, b2, eps), pad, 4, AdamTg, "@adam_" & $pad & "_" & $b1 & "_" & $b2 & "_" & $eps)], buf)
    for i in 0 ..< cnt:
      param[off + i] = buf[L.p + i]; m[off + i] = buf[L.m + i]; v[off + i] = buf[L.v + i]
    off += cnt

proc apfAdamStepFused*(ctx: Backend; param: var Tensor; grad: Tensor; state: var ApfAdamState;
                       lr: float32 = 1e-3'f32; b1: float32 = 0.9'f32; b2: float32 = 0.999'f32;
                       eps: float32 = 1e-8'f32; requantizeEvery: int = 1): CustomFloat =
  ## Drop-in for apfAdamStepGpu: Adam as one micro-kernel. With a finite requantizeEvery (APF
  ## CustomFloat requantization) the old path is kept because requantization is outside this kernel.
  if requantizeEvery > 0 and requantizeEvery != int.high:
    return apfAdamStepGpu(ctx, param, grad, state, lr, b1, b2, eps, requantizeEvery)
  inc state.step
  adamStepFusedGpu(ctx, param.data, grad.data, state.m, state.v, state.step, lr, b1, b2, eps)
  result = buildCustomDtypeForTensor(param.data)
