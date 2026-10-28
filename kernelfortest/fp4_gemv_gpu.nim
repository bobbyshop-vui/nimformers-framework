## fp4_gemv_gpu.nim
## Dùng chung giữa lora.nim và mamba2_real.nim (tránh import vòng).
##
## `Backend` (object{kind: BackendKind} trong backend.nim) KHÔNG PHẢI
## `GpuBackend` (enum trong gpubackend.nim) — mọi hàm sessionXxx của
## bybylang nhận GpuBackend, nên phải convert bằng ctx.kind.toByby()
## trước khi gọi, đúng pattern dùng khắp nim_inference.nim/
## test_nimformer.nim (grep "toByby(" để đối chiếu).
##
## Weight FP4 (uint8 packed): MTLResourceStorageModeShared — unified memory,
## một bản vật lý, pageable (tinh thần llama.cpp mmap/shared, tránh wired
## Private gây Jetsam dù swap còn trống).
## LUT nhỏ (256–272 float): Private OK — không chiếm GB, GPU-only.
##
## metalUploadBytes = Shared (metal_upload_bytes_c).
## metalUploadBytesPrivate = Private/wired — KHÔNG dùng cho weight lớn.
## sessionUploadPrivate (fp32 LUT) lấy từ gpubackend.

import std/[tables, os]
import backend                             # Backend, BackendKind, toByby
import quant       # QuantTensor, buildDecodeLUT via customfloat
import customfloat  # buildDecodeLUT, FP8_E4M3

import fp4_gemv_vm                          # GemvTg (kernel = bytecode micro-VM, KHONG co MSL)
import vendor/bybylang/gpubackend as byby   # sessionUploadPrivate
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

proc sessionUploadBytesShared*(backend: byby.GpuBackend, data: seq[uint8]): byby.SessionHandle =
  ## Weight FP4 packed → Shared buffer (llama.cpp-style: không ghim Private/wired).
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
    result.vulkanHandle = vkDrv.vulkanSessionUploadU8(data)
  else:
    discard

proc uploadFp4WeightOnce*(ctx: Backend, name: string; wq: seq[uint8];
                           lut: seq[float32]; rows, cols: int; jitName: string;
                           rowsPerTg = 1) =
  ## Upload W (u8 Shared) + lut (fp32 Private, nhỏ) 1 LẦN.
  ## Idempotent: đã resident thì return ngay.
  if name in gFp4CacheGpu and gFp4CacheGpu[name].residentReady: return
  let be = ctx.kind.toByby()
  var c: Fp4LinCacheGpu
  c.rows = rows; c.cols = cols; c.jitName = jitName
  if rowsPerTg notin [1, 2, 4] or (rowsPerTg > 1 and (lut.len != Fp4KmLutLen or rows mod rowsPerTg != 0)):
    raise newException(ValueError, "uploadFp4WeightOnce: rowsPerTg khong hop le cho " & name)
  c.rowsPerTg = rowsPerTg
  c.tgSize = GemvTg
  c.tgBytes = (lut.len + rowsPerTg * GemvTg) * 4  # LUT + one 32-lane reduction per output row
  # Weight lớn: Shared (không wired). LUT nhỏ: Private OK.
  c.wBuf = sessionUploadBytesShared(be, wq)
  c.lutBuf = byby.sessionUploadPrivate(be, lut)
  c.residentReady = true
  gFp4CacheGpu[name] = c

proc buildFp4LutQ*(qt: QuantTensor): seq[float32] =
  result = newSeq[float32](256)
  let base = buildDecodeLUT(qt.cf)
  let sc = if qt.scale.len > 0: qt.scale[0] else: 1'f32
  for i in 0 .. 255:
    result[i] = base[i] * sc

proc buildFp4KmLutQ*(qt: QuantTensor): seq[float32] =
  ## LUT gộp cho FP4-KM, 272 phần tử (xem lora.nim ban đầu để biết chi tiết
  ## layout): [0..15]=E2M1, [16..271]=E4M3 đã nhân sẵn globalScale.
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
  ## Encode 1 gemv fp4 (M=1) vao session dang mo. Kernel = bytecode micro-VM da
  ## JIT (fp4_gemv_vm.buildFp4GemvProgram) - layout buffer cua kernel JIT:
  ## Metal: 1=output, 2=W fp4, 3=LUT, 4=x
  ## GLSL:  0=MemF32, 1=MemU8(WeightU8 uint), 2=Lut, 3=Input, 4=MemU32
  ## Dispatch: N/rowsPerTg threadgroup x 32 thread.
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
      "encodeFp4Gemv: pipeline chi cho " & $maxTG & " thread/threadgroup < " & $tg)
  result = sessionEncDispatch(isVk, (c.rows div c.rowsPerTg) * tg, tg, c.tgBytes)

proc encodeFp4GemmM4*(c: Fp4LinCacheGpu; hIn: byby.SessionHandle; inOffElems: int;
                      hOut: byby.SessionHandle; outOffElems: int): bool =
  ## Prefill tile: 4 hàng input, weight đọc 1 lần / cột (MBLOCK=4).
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

var gFp4PadM = -1   ## -1 = chua doc env BYBY_FP4_PAD_M

proc fp4GemmPaddedRows*(c: Fp4LinCacheGpu; M: int): int =
  ## So hang thuc te encodeFp4GemmRows se dispatch (>= M). Phan du M mod 4 >= 2
  ## duoc pad len 1 tile M4: kernel M4 khong co bien kiem tra hang nen GEMV tung
  ## hang le moi lan DOC LAI TOAN BO weight (T=11: 2 M4 + 3 GEMV = 5 pass weight;
  ## pad len 12: 3 pass). Du 1 hang giu GEMV: cung 1 pass weight, it tinh toan hon.
  ## Tat bang BYBY_FP4_PAD_M=0 de A/B.
  if gFp4PadM < 0:
    gFp4PadM = (if getEnv("BYBY_FP4_PAD_M", "1") == "0": 0 else: 1)
  result = M
  if gFp4PadM == 0 or c.m4JitName.len == 0: return
  let r = M mod Fp4KmMblock
  if r >= 2: result = M + (Fp4KmMblock - r)

proc encodeFp4GemmRows*(c: Fp4LinCacheGpu; hIn: byby.SessionHandle;
                        hOut: byby.SessionHandle; M: int): bool =
  ## M hàng: ưu tiên tile M4 (weight 1 lần / 4 token), phần dư GEMV từng hàng.
  ## Cùng 1 session đang mở — không sessionBegin/End ở đây.
  ## hIn/hOut PHAI chua du fp4GemmPaddedRows(c, M) hang (hang pad cua hIn = 0;
  ## hang pad cua hOut la rac, caller bo qua).
  if M <= 0: return true
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
    "fp4GemmGpuRows: chua uploadFp4WeightOnce cho " & name
  doAssert M > 0 and x.len == M * gFp4CacheGpu[name].cols
  let c = gFp4CacheGpu[name]
  let be = ctx.kind.toByby()
  let N = c.rows
  let mPad = fp4GemmPaddedRows(c, M)
  var pfT0 = getMonoTimeIfPf()
  if not sessionBegin(be):
    raise newException(CatchableError, "fp4GemmGpuRows: sessionBegin that bai")
  var hA: byby.SessionHandle
  if mPad > M:
    var xPad = x
    xPad.setLen(mPad * c.cols)   # hang pad = 0
    hA = sessionUpload(be, xPad)
  else:
    hA = sessionUpload(be, x)
  # Output row-major: M*N phan tu dau = M hang that, sessionRead chi doc phan do.
  var hC = sessionAllocScratch(be, mPad * N)
  pfAddIf("fp4.upload", pfT0)
  pfT0 = getMonoTimeIfPf()
  if not encodeFp4GemmRows(c, hA, hC, M):
    discard sessionEnd(be)
    raise newException(CatchableError, "fp4GemmGpuRows: encode that bai")
  if not sessionEnd(be):
    raise newException(CatchableError, "fp4GemmGpuRows: sessionEnd that bai: " &
      sessionGetLastError(be))
  pfAddIf("fp4.gpu", pfT0)
  pfT0 = getMonoTimeIfPf()
  result = sessionRead(be, hC, M * N)
  pfAddIf("fp4.read", pfT0)
  sessionFree(be, hC)
  sessionFree(be, hA)

proc fp4GemvGpu*(ctx: Backend; name: string; x: seq[float32]; M: int): seq[float32] =
  ## Duong host (test/debug): tu sessionBegin..sessionEnd + readback. KHONG dung
  ## trong hot path infer (hot path = encodeFp4Gemv trong 1 session/token).
  doAssert M == 1, "fp4GemvGpu: chi ho tro M=1 (1 token/lan)"
  doAssert name in gFp4CacheGpu and gFp4CacheGpu[name].residentReady,
    "fp4GemvGpu: chua uploadFp4WeightOnce cho " & name
  let c = gFp4CacheGpu[name]
  let be = ctx.kind.toByby()
  let N = c.rows
  if not sessionBegin(be):
    raise newException(CatchableError, "fp4GemvGpu: sessionBegin that bai")
  var hA = sessionUpload(be, x)
  var hC = sessionAllocScratch(be, N)
  if not encodeFp4Gemv(c, hA, 0, hC, 0):
    discard sessionEnd(be)
    raise newException(CatchableError, "fp4GemvGpu: encode that bai")
  if not sessionEnd(be):
    raise newException(CatchableError, "fp4GemvGpu: sessionEnd that bai: " & sessionGetLastError(be))
  result = sessionRead(be, hC, N)
  sessionFree(be, hC)
  sessionFree(be, hA)

proc fp4GemvGpuInto*(ctx: Backend; name: string; hX: byby.SessionHandle;
                     hOut: var byby.SessionHandle; M: int): bool =
  ## Encode fp4 gemv vao session dang mo - KHONG sessionRead.
  doAssert M == 1, "fp4GemvGpuInto: chi ho tro M=1"
  doAssert name in gFp4CacheGpu and gFp4CacheGpu[name].residentReady,
    "fp4GemvGpuInto: chua uploadFp4WeightOnce cho " & name
  let c = gFp4CacheGpu[name]
  let be = ctx.kind.toByby()
  if hOut.metalHandle == nil and hOut.vulkanHandle.buf == nil:
    hOut = sessionAllocScratch(be, c.rows)
  result = encodeFp4Gemv(c, hX, 0, hOut, 0)

proc fp4GemvGpuFromSeqInto*(ctx: Backend; name: string; x: seq[float32];
                            hOut: var byby.SessionHandle; M: int): byby.SessionHandle =
  ## Upload x 1 lan, gemv vao hOut, tra hX (caller co the free). Khong readback.
  let be = ctx.kind.toByby()
  result = sessionUpload(be, x)
  discard fp4GemvGpuInto(ctx, name, result, hOut, M)
