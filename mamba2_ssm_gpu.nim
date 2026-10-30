## mamba2_ssm_gpu.nim — microVM only (KHÔNG kernel Metal mới).
## 1 session/token: encode only, không sessionRead giữa layer.

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
import nimformer     # Tensor, ApfAdamState, apfAdamStepGpu (cho apfAdamStepFused)
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
var gSsmAtCache = initTable[string, VmProgram]()  # scan chay thang tren hWork, khong qua hScratch
var gGpuLayers* = initTable[int, Mamba2GpuLayer]()
var gProfileLastT: float = 0.0   # MOI: dung rieng cho profileLayerStepByIdx (do timing giua cac buoc)
var gCodeBufCache = initTable[string, byby.SessionHandle]()
var gJitCache = jit_vm.initJitCache(jit_vm.JitConfig(
  hotThreshold: 0, enableMetalJit: true, enableVulkanJit: true,
  enableCpuJit: false, maxCacheEntries: 128))

proc shapeKey(H, D, N, G: int): string = $H & "_" & $D & "_" & $N & "_" & $G

proc getSsmAtProgram(H, D, N, G, C: int; o: LayerOff): VmProgram =
  ## TOI UU (gop dispatch): buildMamba2StepAt nhan offset lam THAM SO nen doc/ghi
  ## THANG tren hWork tai vi tri that (o.offXBC/offB/offC/offSsm/offY...), khong
  ## can 9 lan sessionCopyInto qua/ve hScratch nhu getShapeInfo(...).program ban
  ## goc - trong do 2 lan copy state (HD*N phan tu, phan lon nhat trong toan bo
  ## buoc) la nguon overhead chinh (matmul-adjacent memcpy). Cache theo shapeKey
  ## vi cac layer cung H,D,N,G trong 1 model dung chung C/cDim/K/G nen offset
  ## computeOff() giong het nhau.
  let key = shapeKey(H, D, N, G)
  if key in gSsmAtCache: return gSsmAtCache[key]
  result = buildMamba2StepAt(H, D, N, G,
    o.offXBC, o.offXBC + C, o.offXBC + C + G * N,
    o.offDt, o.offALog, o.offDSkip, o.offSsm, o.offY, o.offIdxH, o.offIdxG)
  gSsmAtCache[key] = result

proc blit*(dst: var seq[float32]; dOff: int; src: openArray[float32]; sOff, n: int) =
  ## Chép khối float32 bằng copyMem (marshalling dữ liệu, KHÔNG phải phép tính).
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
  ## Tinh ten kernel (hash bytecode GPU) NGOAI session. Tag chi la namespace;
  ## Vulkan uses the same static-per-program lowering strategy as Metal, but
  ## emits GLSL/SPIR-V instead of MSL. Never hand a Metal pipeline name to it.
  if ctx.kind == bkVulkan:
    return "vkvm_" & jit_vm.programKeyNoSession(program, tag)
  if tag.len > 0 and tag[0] == '@': jit_vm.dispatchProgramTagged(gJitCache, program, tag)
  else: jit_vm.dispatchProgram(gJitCache, program, tag)

proc encodeMicroVm*(ctx: Backend; program: VmProgram; hMem: byby.SessionHandle;
                    nThreads: int; tgBytes = 4; tag = "mvm"; tgSize = 0;
                    preKname = "") =
  ## tgSize > 0: ÉP số lane/threadgroup (kernel kiểu "1 threadgroup / token" cần đúng giá trị này;
  ## nếu GPU không cho đủ thì RAISE thay vì âm thầm hạ xuống).
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

  # preKname: ten kernel da tinh TRUOC khi mo session. Ham nay chay TRONG session dang mo nen KHONG
  # duoc mo session rieng de hash GPU (metal_session_begin tu choi session long nhau) ->
  # dispatchProgramNoSession (dung key da nho, hoac key fallback khong can GPU).
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
    # FIX token rac (bug #9): metal_enc_dispatch_threads AM THAM ha tgSize
    # xuong maxTotalThreadsPerThreadgroup neu ta xin nhieu hon. Voi kernel
    # KHONG dung threadgroup memory (tgBytes<=4, moi thread doc lap) thi vo
    # hai - chi la chia grid ra nhieu threadgroup. Nhung voi kernel CO
    # reduction (tgBytes>4: rms/rmsL) thi cuc ky nguy hiem: program duoc
    # SINH RA voi tgSize nuong theo gia tri ta xin (buoc nhay idx, cay
    # reduction stride, va R0 = thread_position_in_GRID dung lam chi so vao
    # threadgroup memory) - neu Metal chia thanh nhieu threadgroup thi moi
    # threadgroup co VUNG NHO RIENG, phan lon o nho khong ai ghi, barrier
    # chi dong bo trong tung nhom -> tong sai -> ket qua RAC ma KHONG BAO LOI.
    let maxTG = sessionMaxTgThreads()
    if maxTG > 0 and tg > maxTG:
      if tgBytes > 4 or tgSize > 0:
        raise newException(CatchableError,
          "encodeMicroVm[" & tag & "]: kernel reduction can " & $tg &
          " thread/threadgroup nhung pipeline '" & pipeName & "' chi cho toi da " &
          $maxTG & " tren GPU nay. Neu cu dispatch, Metal se tao " &
          $((nThreads + maxTG - 1) div maxTG) & " threadgroup rieng biet va ket qua " &
          "SE LA RAC. Dung rmsTgSizeFor() de build program voi tgSize=" & $maxTG &
          " (hoac dat BYBY_RMS_TG=" & $maxTG & ").")
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
  ## đệ quy đuôi thay cho `for st in steps` (không vòng lặp host)
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
  ## Đặt gTgCap (mamba2_full_vm) = pow2 lớn nhất <= maxTotalThreadsPerThreadgroup THẬT của GPU.
  ## Gọi trước khi build bất kỳ program reduction nào (rmsTg / rmsRowTg / lpPhiTg).
  ## BYBY_RMS_TG=<n> ép tay. Backend không phải Metal: giữ 1024.
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
  # Pipeline JIT dùng nhiều thanh ghi hơn micro_vm_kernel nên giới hạn thật THẤP hơn
  # (Iris 640: micro_vm_kernel 512, kernel JIT chỉ 448). Không biết trước khi biên dịch,
  # nên chặn ở 256 (đủ cho reduction, mọi GPU Mac đều chạy được). BYBY_RMS_TG ép tay.
  if ctx.kind != bkVulkan and getEnv("BYBY_RMS_TG").len == 0: cap = min(cap, 256)
  let requested = getEnv("BYBY_RMS_TG")
  if requested.len > 0:
    try: cap = min(cap, max(1, parseInt(requested)))
    except ValueError: discard
  cap = 1 shl fastLog2(max(cap, 32))
  gTgCap = cap
  if not queried:
    # Dang trong session khac / chua truy van duoc: dung tam 256 (<= moi GPU Mac), thu lai lan sau.
    gTgCap = min(cap, 256)
    return
  gTgCapReady = true
  gTgCapBackend = ctx.kind
  stderr.writeLine "[tgcap] gioi han threadgroup cho kernel reduction = " & $cap

var gRmsTgSize = initTable[string, int]()

proc rmsTgSizeFor*(ctx: Backend; C: int): int =
  ## FIX token rac (bug #9). Chon so thread/threadgroup cho kernel rmsGate
  ## (gated RMSNorm CUOI mixer - chay MOI block, MOI token) sao cho Metal
  ## chay DUNG 1 threadgroup duy nhat.
  ##
  ## Truoc day cho cung 1024 (buildRmsGateLargeProgram(C, eps, 1024) +
  ## dispatch 1024 thread). 1024 vuot qua maxTotalThreadsPerThreadgroup cua
  ## GPU tich hop (Intel Iris Plus 640 thuong chi 256) -> shim ha xuong 256
  ## -> Metal tao 4 threadgroup, moi cai co threadgroup memory RIENG, trong
  ## khi program dung R0 = thread_position_in_GRID (0..1023) lam chi so vao
  ## vung nho do va reduce tu stride 512. Ket qua: phan lon o nho khong bao
  ## gio duoc ghi, tong sum-of-squares sai, invStd sai -> dau ra mixer RAC o
  ## moi block, moi token (nhat quan, lap lai y het moi lan chay).
  ##
  ## Ep buoc tay: BYBY_RMS_TG=<n>.
  let cacheKey = $ctx.kind & "_" & $C
  if cacheKey in gRmsTgSize: return gRmsTgSize[cacheKey]
  # Chọn tgSize = luỹ thừa 2 lớn nhất <= min(1024, C, giới hạn GPU) VÀ chia hết C.
  # Công thức đóng (bit trick) thay cho 2 vòng `while t > ...: t = t div 2`.
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
        "rmsTgSizeFor: khong tim duoc tgSize hop le cho C=" & $C &
        " voi gioi han " & $bound & " thread/threadgroup")
    stderr.writeLine "[rmsgate] tgSize=" & $t & " cho C=" & $C &
      " (GPU cho toi da " & $bound & " thread/threadgroup) - FIX bug #9"
  gRmsTgSize[cacheKey] = t
  result = t

proc runMicroVm*(ctx: Backend; program: VmProgram; memF32: var seq[float32];
                 nThreads: int; tgBytes = 4; tag = "mvm"; tgSize = 0): bool =
  ## Host-driven microVM: moi lan 1 session rieng (begin→encode→end→read).
  ## Retry timeout chỉ áp dụng cho buffer scratch được upload từ host ở đây;
  ## KHÔNG replay các API encode trực tiếp trên state recurrent resident.
  let be = ctx.kind.toByby()
  if be == byby.gbCpu:
    let ts = if tgSize > 0: tgSize else: max(1, min(nThreads, 256))
    let tf = max(0, tgBytes div 4)
    if not cpu.vmRun(program.words, memF32, @[], nThreads, tf, ts):
      raise newException(CatchableError, "CPU microVM failed: " & cpu.cpuLastErrorMessage())
    return true
  checkVmScratchCapacity(ctx, memF32.len, tag)
  let kn = kernelNameFor(ctx, program, tag)   # hash/JIT truoc sessionBegin
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
  ## NHIỀU kernel liên tiếp trên CÙNG 1 buffer trong 1 session: upload 1 lần, đọc lại 1 lần
  ## (không quay về host giữa các stage). Dùng cho forward/backward cả lớp Mamba2.
  ## Nếu command buffer timeout, cả chain được replay từ bản scratch host gốc;
  ## không dùng cơ chế này cho persistent recurrent state.
  let be = ctx.kind.toByby()
  if be == byby.gbCpu:
    for step in steps:
      let ts = if step.tgSize > 0: step.tgSize else: max(1, min(step.nThreads, 256))
      if not cpu.vmRun(step.prog.words, memF32, @[], step.nThreads,
                       max(0, step.tgBytes div 4), ts):
        raise newException(CatchableError, "CPU microVM chain failed: " & cpu.cpuLastErrorMessage())
    return true
  checkVmScratchCapacity(ctx, memF32.len, "stage chain")
  let names = steps.mapIt(kernelNameFor(ctx, it.prog, it.tag))   # truoc sessionBegin
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
  ## Bang goi runMicroVmChain lan luot cho tung nhom, nhung buffer chi upload 1 lan
  ## va doc lai 1 lan. Moi nhom van la 1 command buffer rieng (thoi gian GPU moi
  ## command buffer giu nguyen nhu cu -> khong tang nguy co watchdog timeout).
  ## Loi/timeout o bat ky nhom nao -> bo ket qua GPU, chay lai TU DAU tung nhom bang
  ## runMicroVmChain tu ban host goc (kernel co atomic cong don, khong replay giua chung).
  ## BYBY_VM_GROUPS=0 -> duong cu (moi nhom 1 lan upload/doc).
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
    names.add g.mapIt(kernelNameFor(ctx, it.prog, it.tag))   # truoc sessionBegin
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
    # Giong runMicroVmChain: CHI timeout GPU moi chay lai; loi khac raise ngay.
    if not (e.msg.startsWith(VmTimeoutRetryMarker) or
            (timeoutDetail.len > 0 and isVmGpuTimeout(timeoutDetail))):
      raise
    stderr.writeLine("[microvm] chain groups timeout (" & e.msg &
      "); chay lai tung nhom tu ban host goc")
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
  ## Nhu runMicroVmChain nhung KHONG di qua ca buffer tren host: cap scratch GPU
  ## totalLen float (buffer Metal/Vulkan moi cap phat = 0), chi upload `head` vao
  ## [0, head.len), chay chain, roi chi doc lai cac doan `reads` (gom vao 1 buffer
  ## nho bang copy device-to-device). Dung cho prefill suy luan: scratch Mamba2
  ## ~57MB/block nhung host chi can normedOut + state cuoi (~4MB).
  ## Caller PHAI dam bao: moi vung >= head.len ma kernel DOC deu do chinh chain GHI
  ## truoc (khong co cong don/atomic). Loi/timeout -> raise (khong retry); caller
  ## tu fallback ve runMicroVmChain.
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
  let names = steps.mapIt(kernelNameFor(ctx, it.prog, it.tag))   # truoc sessionBegin
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
  ## dst[i] += scale * src[i] trên GPU (thay vòng `for i: dst[i] += src[i]*s` của LoRA)
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
  # idxH/idxG sinh bằng GPU (buildIdxTableProgram) — không còn for h/for d trên CPU
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
  ## FIX token rac (bug #12): xoa ve 0 TRANG THAI THUONG TRU cua 1 layer
  ## (convState + ssmState nam trong hWork). Bat buoc goi sau warmup, vi
  ## warmupJambaKernelsOnce chay encodeLayerStepByIdx TREN CHINH
  ## gGpuLayers[0] that (khong phai ban sao): voi x=0 thi proj=0 nen z=0 va
  ## xBC=0, NHUNG conv1d van ra silu(convBias) != 0 -> ssmState cua block 0
  ## bi ghi gia tri tu mot token GIA truoc khi prefill bat dau. Cac block
  ## 1..63 thi sach -> hong bat doi xung, rat kho phat hien.
  ## PHAI goi TRONG mot session dang mo (dung sessionCopyInto nhu moi buoc
  ## khac trong file nay).
  if blockIdx notin gGpuLayers: return false
  var layer = gGpuLayers[blockIdx]   # copy cua struct; hWork la handle nen van tro cung 1 MTLBuffer
  if not layer.ready: return false
  let be = layer.be
  let o = layer.off
  let nConv = layer.cDim * max(layer.K - 1, 0)
  let nSsm = layer.HD * layer.N
  let nMax = max(nConv, nSsm)
  if nMax <= 0: return true
  # upload zero THAT SU (khong dua vao gia dinh buffer moi da duoc zero)
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
    raise newException(ValueError, "rmsGateGpu: C phai chia het nGroups")
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
    stderr.writeLine "[PATCH_MARKER_V3] encodeLayerStep FIX (convState slide) DANG CHAY"
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

  # SUA (gop kernel - hoc theo fusedNormQKVRoPE/fusedResidualNormFFN cua
  # nim_inference.nim, model 6.7B THAT khong lag tren cung phan cung, dung
  # micro-VM CO SAN - KHONG viet .metal tay): TRUOC DAY day la 3 dispatch
  # rieng (split/conv/softplus) + 4 sessionCopyInto rieng chuyen xBC/dt
  # qua lai giua buffer moi layer - moi layer x 64 layer x moi token la rat
  # nhieu lan bind-pipeline/dispatch nho le, day chinh la nguon lag he
  # thong (khong phai do commitContinue/QoS/yield - nhung cai do chi la
  # vá trieu chung). GIO gop thanh 1 dispatch DUY NHAT qua
  # buildFusedSplitConvSoftplusProgram - AN TOAN de gop vi conv1d o day la
  # depthwise theo thoi gian (xem giai thich day du trong ham do), khong
  # thread nao can du lieu cua thread khac nen khong can threadgroup_barrier.
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
  # SUA (fix bug conv1d dong bang - xem bao cao debug): TRUOC DAY comment o
  # day ghi "convState KHONG duoc ghi lai, kernel goc cung khong cap nhat
  # no" - SAI: doc lai buildConv1dIncProgram (dong 194-206) thi ban GOC CO
  # phan "slide state" cap nhat convState, chi la bi BO SOT luc gop thanh
  # buildFusedSplitConvSoftplusProgram. Da them lai output offOutConvSt o
  # kernel do - copy ve o.offConvSt O DAY de convState THUC SU truot qua
  # tung token (truoc day dong bang tu sau buoc dau, lam sai tin hieu cuc
  # bo dua vao SSM moi layer, moi token).
  let fOffOutConvSt = fOffOutDt + H
  discard sessionCopyInto(be, hScratch, layer.hWork, fOffOutConvSt, o.offConvSt, cDim * (K - 1))

  # TOI UU (gop dispatch, xem getSsmAtProgram o tren): scan SSM chay THANG
  # tren hWork bang buildMamba2StepAt, khong qua hScratch nua - bo 9 lan
  # sessionCopyInto/lan/layer/token, trong do 2 lan copy state HD*N (buffer
  # lon nhat trong buoc) la phan dat nhat. dtEnc: 1 dispatch, 0 copy phu.
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
  ## MOI - CHI DE DO (chay 1 LAN, KHONG dung cho hot path binh thuong):
  ## lap lai CHINH XAC cac buoc cua encodeLayerStep, nhung chen 1 lan
  ## sessionEnd() (cho THAT, do THAT gio GPU) + sessionBegin() lai giua
  ## MOI buoc - day la CACH DUY NHAT do dung thoi gian GPU THAT cho tung
  ## kernel rieng le, vi CPU timestamp quanh lenh encode KHONG phan anh gi
  ## (commit la bat dong bo - GPU co the chua chay gi luc CPU "encode xong").
  ## Chap nhan CHAY CHAM HON BINH THUONG rat nhieu (moi buoc them 1 lan
  ## waitUntilCompleted THAT, dung nhu ly do lam cham o lan flush-moi-N-layer
  ## truoc day) - DAY LA DANH DOI CO Y, chi chay 1 LAN de lay so lieu that,
  ## khong phai duong chay san xuat. Bat bang BYBY_JAMBA_PROFILE_STEPS=1.
  doAssert blockIdx in gGpuLayers
  var layer = gGpuLayers[blockIdx]
  doAssert layer.ready
  let be = layer.be
  let o = layer.off
  let H = layer.H; let D = layer.D; let N = layer.N; let G = layer.G
  let C = layer.C; let cDim = layer.cDim; let K = layer.K; let HD = layer.HD

  proc lap(label: string) =
    discard sessionEnd(be)   # cho THAT - danh gia GPU da chay xong buoc truoc
    let t = epochTime()
    stderr.writeLine &"[PROFILE block{blockIdx}] {label}: {(t - gProfileLastT)*1000:.1f} ms"
    gProfileLastT = t
    discard sessionBegin(be)  # mo lai NGAY - handle (hWork/hScratch...) van song, khong mat gi

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
  ## Backward RMSNorm + SiLU gate - 1 micro-kernel, 1 dispatch. Mọi phép tính
  ## (sigmoid, gated, 2 reduction sum/dot, dGated, dSilu...) chạy trên GPU; host chỉ
  ## chép dữ liệu vào/ra buffer (copyMem).
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
  # Gradient cộng dồn atomic cần zero: newSeq đã zero-init, kernel tự ghi đè dX.
  # h/g suy ra NGAY TRONG kernel (không còn bảng idxH/idxG precompute trên CPU).
  
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
# HOST WRAPPER KERNEL TRAIN (RMSNorm, Linear++, softmax-CE, Adam gộp)
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
  var buf = newSeq[float32](L.total)          # vùng dGamma = 0 cho atomic
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
  ## qkv: [B*T, 3C] (q|k|v mỗi token). Trả y [B*T, C].
  let L = lpLayout(B, T, H, D)
  var buf = newSeq[float32](L.total)
  buf.blit(L.qkv, qkv, 0, L.BT * 3 * L.C)
  discard runMicroVmChain(ctx, linearPlusForwardSteps(L), buf)
  result = buf[L.y ..< L.y + L.BT * L.C]

proc linearPlusBackwardGpu*(ctx: Backend; qkv, dY: seq[float32]; B, T, H, D: int): seq[float32] =
  ensureTgCap(ctx)
  ## Dựng lại forward trong cùng session rồi backward 2 pass. Trả dQKV [B*T, 3C]
  ## (đúng layout dProj cho qkvProj.backward).
  let L = lpLayout(B, T, H, D)
  var buf = newSeq[float32](L.total)          # vùng trạng thái scan = 0; dQKV được ghi đè
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
  ## targets[r] < 0 => bỏ qua. nValid = số hàng hợp lệ (chia loss/gradient cho nValid).
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

const AdamChunk = 8 * 1024 * 1024     ## phần tử / lần dispatch (chia hết AdamTg)

proc adamStepFusedGpu*(ctx: Backend; param: var seq[float32]; grad: seq[float32];
                       m, v: var seq[float32]; step: int; lr: float32;
                       b1: float32 = 0.9'f32; b2: float32 = 0.999'f32;
                       eps: float32 = 1e-8'f32) =
  ## Adam 1 kernel / chunk (thay ~14 phép be* rời). `step` đã tăng (>=1) trước khi gọi.
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
  ## Drop-in cho apfAdamStepGpu: Adam 1 micro-kernel. Nếu requantizeEvery hữu hạn (APF
  ## requantize CustomFloat) thì giữ đường cũ vì bước requantize nằm ngoài kernel này.
  if requantizeEvery > 0 and requantizeEvery != int.high:
    return apfAdamStepGpu(ctx, param, grad, state, lr, b1, b2, eps, requantizeEvery)
  inc state.step
  adamStepFusedGpu(ctx, param.data, grad.data, state.m, state.v, state.step, lr, b1, b2, eps)
  result = buildCustomDtypeForTensor(param.data)
