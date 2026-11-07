## fp4_gemv_gpu.nim
## Shared between lora.nim and mamba2_real.nim (avoids an import cycle).
##
## `Backend` (object{kind: BackendKind} in backend.nim) is NOT
## `GpuBackend` (an enum in gpubackend.nim) — every bybylang sessionXxx function
## takes a GpuBackend, so convert with ctx.kind.toByby()
## before calling, the same pattern used across nim_inference.nim/
## test_nimformer.nim (grep "toByby(" to compare).
##
## FP4 weights (uint8 packed): MTLResourceStorageModeShared — unified memory,
## one physical copy, pageable (the llama.cpp mmap/shared spirit, avoiding wired
## Private memory that triggers Jetsam even with free swap).
## Small LUTs (256–272 floats): Private is fine — not GBs, GPU-only.
##
## metalUploadBytes = Shared (metal_upload_bytes_c).
## metalUploadBytesPrivate = Private/wired — NOT for large weights.
## sessionUploadPrivate (fp32 LUT) comes from gpubackend.

import std/[tables, os]
import backend                             # Backend, BackendKind, toByby
import quant       # QuantTensor, buildDecodeLUT via customfloat
import customfloat  # buildDecodeLUT, FP8_E4M3

import fp4_gemv_vm                          # GemvTg (kernel = micro-VM bytecode, NO MSL)
import vendor/bybylang/gpubackend as byby   # sessionUploadPrivate
import vendor/bybylang/probe
import session_vm_dispatch
import prefill_profile   # BYBY_PREFILL_PROFILE: fp4.upload / fp4.gpu / fp4.read
import vendor/bybylang/backends/vulkan/vulkan_driver as vkDrv

when defined(macosx):
  import vendor/bybylang/backends/metal/metal_backend as metalBackendMod

type
  Fp4LinCacheGpu* = object
    wBuf*: byby.SessionHandle
    lutBuf*: byby.SessionHandle
    rows*, cols*: int
    rowsPerTg*: int       ## 2 for FP4-KM group32 tiled GEMV, otherwise 1
    tgSize*: int          ## exact group width used by the selected decode kernel
    jitName*: string
    m4JitName*: string    ## FP4-KM MBLOCK=4 prefill GEMM (empty if not built)
    m8JitName*: string    ## FP4-KM MBLOCK=8, shared reduction for short prefixes
    m8TgSize*: int
    glslSrc*: string       ## cached GLSL for Vulkan multi-buf path
    m4GlslSrc*: string
    tgBytes*: int          ## threadgroup memory (LUT + vung reduce) - xem fp4GemvTgBytes
    residentReady*: bool

var gFp4CacheGpu* = initTable[string, Fp4LinCacheGpu]()
const Fp4KmMblock* = 4

proc sessionUploadBytesShared*(backend: byby.GpuBackend, data: openArray[uint8]): byby.SessionHandle =
  ## Packed FP4 weights → a Shared buffer (llama.cpp style: no Private/wired pinning).
  ## Metal: metalUploadBytes → MTLResourceStorageModeShared.
  ## Vulkan: vulkanSessionUploadU8 / vulkanSessionUploadBytes.
  var chosen = backend
  if chosen == byby.gbAuto: chosen = byby.detectBackend()
  if chosen == byby.gbCuda: chosen = byby.gbVulkan
  result.kind = chosen
  case chosen
  of byby.gbMetal:
    when defined(macosx):
      result.metalHandle = metalBackendMod.metalUploadBytes(data)
    else:
      result.metalHandle = nil
  of byby.gbVulkan:
    result.vulkanHandle = vkDrv.vulkanSessionUploadU8(@data)
  else:
    discard
  if unlikely(firProbeOn):
    byby.probePrepare(result, "upload")
    byby.probeBytesResult(result, data.len)
    probeTransfer("upload", " dst=" & byby.probeValue(result))

proc sessionUploadFloatsShared*(backend: byby.GpuBackend;
                                data: openArray[float32]): byby.SessionHandle =
  let p = if data.len > 0: cast[ptr UncheckedArray[uint8]](unsafeAddr data[0]) else: nil
  sessionUploadBytesShared(backend, toOpenArray(p, 0, data.len * 4 - 1))

proc uploadFp4WeightOnce*(ctx: Backend, name: string; wq: openArray[uint8];
                           lut: seq[float32]; rows, cols: int; jitName: string;
                           rowsPerTg = 1) =
  ## Uploads W (u8 Shared) + lut (fp32 Private, small) ONCE.
  ## Idempotent: returns right away if already resident.
  if name in gFp4CacheGpu and gFp4CacheGpu[name].residentReady: return
  let be = ctx.kind.toByby()
  var c: Fp4LinCacheGpu
  c.rows = rows; c.cols = cols; c.jitName = jitName
  if rowsPerTg notin [1, 2, 4] or (rowsPerTg > 1 and (lut.len != Fp4KmLutLen or rows mod rowsPerTg != 0)):
    raise newException(ValueError, "uploadFp4WeightOnce: invalid rowsPerTg for " & name)
  c.rowsPerTg = rowsPerTg
  c.tgSize = GemvTg
  c.tgBytes = (lut.len + rowsPerTg * GemvTg) * 4  # LUT + one 32-lane reduction per output row
  # Large weights: Shared (not wired). Small LUT: Private is fine.
  c.wBuf = sessionUploadBytesShared(be, wq)
  c.lutBuf = byby.sessionUploadPrivate(be, lut)
  if unlikely(firProbeOn):
    probeName(c.wBuf, name, "fp4", [rows, cols])
    probeName(c.lutBuf, name & ".lut", "f32", [lut.len])
  c.residentReady = true
  gFp4CacheGpu[name] = c

proc buildFp4LutQ*(qt: QuantTensor): seq[float32] =
  result = newSeq[float32](256)
  let base = buildDecodeLUT(qt.cf)
  let sc = if qt.scale.len > 0: qt.scale[0] else: 1'f32
  for i in 0 .. 255:
    result[i] = base[i] * sc

proc buildFp4KmLutQ*(qt: QuantTensor): seq[float32] =
  ## Combined LUT for FP4-KM, 272 elements (see the original lora.nim for the
  ## layout details): [0..15]=E2M1, [16..271]=E4M3 premultiplied by globalScale.
  result = newSeq[float32](272)
  let f4 = fp4kmDecodeLut()
  for i in 0 ..< 16: result[i] = f4[i]
  let e8 = buildDecodeLUT(FP8_E4M3)
  let gsc = if qt.scale.len > 0: qt.scale[0] else: 1'f32
  for b in 0 ..< 256:
    var v = e8[b] * gsc
    if v != v or abs(v) > 1e30'f32: v = 0'f32
    result[16 + b] = v

proc encodeFp4Gemv*(c: Fp4LinCacheGpu; hIn: byby.SessionHandle; inOffElems: int;
                    hOut: byby.SessionHandle; outOffElems: int): bool =
  ## Encodes one fp4 gemv (M=1) into the open session. Kernel = JIT-compiled micro-VM
  ## bytecode (fp4_gemv_vm.buildFp4GemvProgram) - buffer layout of the JIT kernel:
  ## Metal: 1=output, 2=W fp4, 3=LUT, 4=x
  ## GLSL:  0=MemF32, 1=MemU8(WeightU8 uint), 2=Lut, 3=Input, 4=MemU32
  ## Dispatch: N/rowsPerTg threadgroups x 32 threads.
  let isVk = hOut.kind == byby.gbVulkan
  let tg = if c.tgSize > 0: c.tgSize else: GemvTg
  let glsl = if isVk: c.glslSrc else: ""
  if isVk and glsl.len == 0:
    return false
  if not sessionEncPipeline(isVk, c.jitName, glsl): return false
  if isVk:
    discard sessionBindBufferVk(0, hOut.vulkanHandle, outOffElems)
    discard sessionBindBufferVk(1, c.wBuf.vulkanHandle, 0)
    discard sessionBindBufferVk(2, c.lutBuf.vulkanHandle, 0)
    discard sessionBindBufferVk(3, hIn.vulkanHandle, inOffElems)
    discard sessionBindBufferVk(4, hOut.vulkanHandle, outOffElems)
  else:
    when defined(macosx):
      discard sessionBindBuffer(1, hOut.metalHandle, outOffElems * 4)
      discard sessionBindBuffer(2, c.wBuf.metalHandle)
      discard sessionBindBuffer(3, c.lutBuf.metalHandle)
      discard sessionBindBuffer(4, hIn.metalHandle, inOffElems * 4)
  discard sessionEncTgBytes(isVk, 0, max(c.tgBytes, 16))
  let maxTG = sessionEncMaxTg(isVk)
  if maxTG > 0 and maxTG < tg:
    raise newException(CatchableError,
      "encodeFp4Gemv: the pipeline allows only " & $maxTG & " threads/threadgroup < " & $tg)
  result = sessionEncDispatch(isVk, (c.rows div c.rowsPerTg) * tg, tg, c.tgBytes)

proc encodeFp4GemmM4*(c: Fp4LinCacheGpu; hIn: byby.SessionHandle; inOffElems: int;
                      hOut: byby.SessionHandle; outOffElems: int): bool =
  ## Prefill tile: 4 input rows, weights read once per column (MBLOCK=4).
  if c.m4JitName.len == 0: return false
  let isVk = hOut.kind == byby.gbVulkan
  let glsl = if isVk: c.m4GlslSrc else: ""
  if isVk and glsl.len == 0: return false
  if not sessionEncPipeline(isVk, c.m4JitName, glsl): return false
  if isVk:
    discard sessionBindBufferVk(0, hOut.vulkanHandle, outOffElems)
    discard sessionBindBufferVk(1, c.wBuf.vulkanHandle, 0)
    discard sessionBindBufferVk(2, c.lutBuf.vulkanHandle, 0)
    discard sessionBindBufferVk(3, hIn.vulkanHandle, inOffElems)
    discard sessionBindBufferVk(4, hOut.vulkanHandle, outOffElems)
  else:
    when defined(macosx):
      discard sessionBindBuffer(1, hOut.metalHandle, outOffElems * 4)
      discard sessionBindBuffer(2, c.wBuf.metalHandle)
      discard sessionBindBuffer(3, c.lutBuf.metalHandle)
      discard sessionBindBuffer(4, hIn.metalHandle, inOffElems * 4)
  discard sessionEncTgBytes(isVk, 0, max(c.tgBytes, 16))
  result = sessionEncDispatch(isVk, c.rows * GemvTg, GemvTg, c.tgBytes)

var gFp4PadM = -1   ## -1 = BYBY_FP4_PAD_M not read yet

proc fp4GemmPaddedRows*(c: Fp4LinCacheGpu; M: int): int =
  ## Number of rows encodeFp4GemmRows actually dispatches (>= M). A remainder M mod 4 >= 2
  ## is padded to one M4 tile: the M4 kernel has no row bound check, and a per-row GEMV
  ## re-reads ALL weights each time (T=11: 2 M4 + 3 GEMV = 5 weight passes;
  ## padded to 12: 3 passes). A remainder of 1 row stays a GEMV: same single weight pass, less compute.
  ## Turn off with BYBY_FP4_PAD_M=0 for A/B tests.
  if gFp4PadM < 0:
    gFp4PadM = (if getEnv("BYBY_FP4_PAD_M", "1") == "0": 0 else: 1)
  result = M
  if gFp4PadM == 0 or c.m4JitName.len == 0: return
  let r = M mod Fp4KmMblock
  if r >= 2: result = M + (Fp4KmMblock - r)

proc encodeFp4GemmRows*(c: Fp4LinCacheGpu; hIn: byby.SessionHandle;
                        hOut: byby.SessionHandle; M: int): bool =
  ## M rows: M4 tiles preferred (weights once per 4 tokens), the remainder as per-row GEMVs.
  ## Same open session — no sessionBegin/End here.
  ## hIn/hOut MUST hold fp4GemmPaddedRows(c, M) rows (the padding rows of hIn = 0;
  ## the padding rows of hOut are garbage, the caller ignores them).
  if M <= 0: return true
  if hIn.kind == byby.gbVulkan and hOut.kind == byby.gbVulkan:
    var outHandle = hOut.vulkanHandle
    if vkDrv.vulkanSessionFp4GemmEnc(hIn.vulkanHandle, c.wBuf.vulkanHandle,
        c.lutBuf.vulkanHandle, outHandle, M, c.cols, c.rows): return true
  let K = c.cols
  let N = c.rows
  let mRun = fp4GemmPaddedRows(c, M)
  var m = 0
  result = true
  while m + Fp4KmMblock <= mRun and c.m4JitName.len > 0:
    if not encodeFp4GemmM4(c, hIn, m * K, hOut, m * N):
      result = false
      return
    m += Fp4KmMblock
  while m < mRun:
    if not encodeFp4Gemv(c, hIn, m * K, hOut, m * N):
      result = false
      return
    inc m

proc fp4GemmGpuRows*(ctx: Backend; name: string; x: seq[float32]; M: int): seq[float32] =
  ## Prefill/batch host entry: 1 session, M rows. Tile MBLOCK=4, phan du pad (fp4GemmPaddedRows).
  doAssert name in gFp4CacheGpu and gFp4CacheGpu[name].residentReady,
    "fp4GemmGpuRows: uploadFp4WeightOnce not called for " & name
  doAssert M > 0 and x.len == M * gFp4CacheGpu[name].cols
  let c = gFp4CacheGpu[name]
  let be = ctx.kind.toByby()
  let N = c.rows
  let mPad = fp4GemmPaddedRows(c, M)
  var pfT0 = getMonoTimeIfPf()
  if not sessionBegin(be):
    raise newException(CatchableError, "fp4GemmGpuRows: sessionBegin failed")
  var hA: byby.SessionHandle
  if mPad > M:
    var xPad = x
    xPad.setLen(mPad * c.cols)   # hang pad = 0
    hA = sessionUpload(be, xPad)
  else:
    hA = sessionUpload(be, x)
  # Row-major output: the first M*N elements = the M real rows, sessionRead reads only those.
  var hC = sessionAllocScratch(be, mPad * N)
  pfAddIf("fp4.upload", pfT0)
  pfT0 = getMonoTimeIfPf()
  if not encodeFp4GemmRows(c, hA, hC, M):
    discard sessionEnd(be)
    raise newException(CatchableError, "fp4GemmGpuRows: encode failed")
  if not sessionEnd(be):
    raise newException(CatchableError, "fp4GemmGpuRows: sessionEnd failed: " &
      sessionGetLastError(be))
  pfAddIf("fp4.gpu", pfT0)
  pfT0 = getMonoTimeIfPf()
  result = sessionRead(be, hC, M * N)
  pfAddIf("fp4.read", pfT0)
  sessionFree(be, hC)
  sessionFree(be, hA)

proc fp4GemvGpu*(ctx: Backend; name: string; x: seq[float32]; M: int): seq[float32] =
  ## Host path (test/debug): its own sessionBegin..sessionEnd + readback. NOT used
  ## in the infer hot path (hot path = encodeFp4Gemv inside one session/token).
  doAssert M == 1, "fp4GemvGpu: supports only M=1 (one token at a time)"
  doAssert name in gFp4CacheGpu and gFp4CacheGpu[name].residentReady,
    "fp4GemvGpu: uploadFp4WeightOnce not called for " & name
  let c = gFp4CacheGpu[name]
  let be = ctx.kind.toByby()
  let N = c.rows
  if not sessionBegin(be):
    raise newException(CatchableError, "fp4GemvGpu: sessionBegin failed")
  var hA = sessionUpload(be, x)
  var hC = sessionAllocScratch(be, N)
  if not encodeFp4Gemv(c, hA, 0, hC, 0):
    discard sessionEnd(be)
    raise newException(CatchableError, "fp4GemvGpu: encode failed")
  if not sessionEnd(be):
    raise newException(CatchableError, "fp4GemvGpu: sessionEnd failed: " & sessionGetLastError(be))
  result = sessionRead(be, hC, N)
  sessionFree(be, hC)
  sessionFree(be, hA)

proc fp4GemvGpuInto*(ctx: Backend; name: string; hX: byby.SessionHandle;
                     hOut: var byby.SessionHandle; M: int): bool =
  ## Encodes an fp4 gemv into the open session - NO sessionRead.
  doAssert M == 1, "fp4GemvGpuInto: supports only M=1"
  doAssert name in gFp4CacheGpu and gFp4CacheGpu[name].residentReady,
    "fp4GemvGpuInto: uploadFp4WeightOnce not called for " & name
  let c = gFp4CacheGpu[name]
  let be = ctx.kind.toByby()
  if hOut.metalHandle == nil and hOut.vulkanHandle.buf == nil:
    hOut = sessionAllocScratch(be, c.rows)
  result = encodeFp4Gemv(c, hX, 0, hOut, 0)

proc fp4GemvGpuFromSeqInto*(ctx: Backend; name: string; x: seq[float32];
                            hOut: var byby.SessionHandle; M: int): byby.SessionHandle =
  ## Uploads x once, gemv into hOut, returns hX (the caller may free it). No readback.
  let be = ctx.kind.toByby()
  result = sessionUpload(be, x)
  discard fp4GemvGpuInto(ctx, name, result, hOut, M)
