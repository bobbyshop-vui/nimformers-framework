## session_low.nim
## Session DƯỚI CHÂN — map 1:1 Metal MTLComputeCommandEncoder.
## Dev KHÔNG viết kernel encode C mới: chỉ pipeline + bind + dispatch.
##
## Yêu cầu: đã dán metal_enc_lowlevel.inc.h/.m vào metal_shim.
##
##   import session_low
##   discard sessionBegin(be)
##   discard sessionPipeline(be, "matmul_fp4_lut_kernel_ndst2_paired")
##   discard sessionBind(be, 0, hA)
##   ...
##   discard sessionDispatch(be, gridX = (N+1) div 2, gridY = M)
##   discard sessionEnd(be)

import std/[strformat]
import bybylang / gpubackend
import bybylang / backends / metal / metal_backend

# ── importc encoder thô ──────────────────────────────────────
when defined(macosx):
  proc metal_enc_pipeline(kernelName: cstring): cint
    {.importc: "metal_enc_pipeline", header: "metal_shim.h".}
  proc metal_enc_bind_buffer(index: cint; buf: MetalBufferHandle; offsetBytes: cint): cint
    {.importc: "metal_enc_bind_buffer", header: "metal_shim.h".}
  proc metal_enc_bind_int(index: cint; v: cint): cint
    {.importc: "metal_enc_bind_int", header: "metal_shim.h".}
  proc metal_enc_bind_float(index: cint; v: cfloat): cint
    {.importc: "metal_enc_bind_float", header: "metal_shim.h".}
  proc metal_enc_bind_bytes(index: cint; data: pointer; nBytes: cint): cint
    {.importc: "metal_enc_bind_bytes", header: "metal_shim.h".}
  proc metal_enc_dispatch(gridX, gridY, gridZ, tgX, tgY, tgZ: cint): cint
    {.importc: "metal_enc_dispatch", header: "metal_shim.h".}
  proc metal_enc_dispatch_threads(threadsX, threadsY, threadsZ, tgX, tgY, tgZ: cint): cint
    {.importc: "metal_enc_dispatch_threads", header: "metal_shim.h".}
  proc metal_enc_barrier(): cint
    {.importc: "metal_enc_barrier", header: "metal_shim.h".}
  proc metal_enc_simd_width(): cint
    {.importc: "metal_enc_simd_width", header: "metal_shim.h".}
  proc metal_enc_max_tg_threads(): cint
    {.importc: "metal_enc_max_tg_threads", header: "metal_shim.h".}

# ── Biến = MTLBuffer (reuse upload/alloc đã có) ──────────────

proc sessionVarF32*(be: GpuBackend; data: openArray[float32]): SessionHandle =
  ## newBuffer từ float32 host (Shared) — activation / LUT nhỏ.
  sessionUpload(be, @data)

proc sessionVarF32Private*(be: GpuBackend; data: openArray[float32]): SessionHandle =
  ## Private/upload weight resident — ít traffic CPU.
  result.kind = be
  when defined(macosx):
    result.metalHandle = metalUploadPrivate(@data)

proc sessionVarU8*(be: GpuBackend; data: openArray[uint8]): SessionHandle =
  ## fp4/fp8/int4 pack bytes.
  result.kind = be
  when defined(macosx):
    var d = @data
    result.metalHandle = metalUploadBytes(d)

proc sessionVarU8Private*(be: GpuBackend; data: openArray[uint8]): SessionHandle =
  result.kind = be
  when defined(macosx):
    var d = @data
    result.metalHandle = metalUploadBytesPrivate(d)

proc sessionVarI32*(be: GpuBackend; data: openArray[int32]): SessionHandle =
  sessionUploadIndices(be, @data)

proc sessionVarAlloc*(be: GpuBackend; nFloats: int): SessionHandle =
  ## Scratch output — metal_alloc_scratch.
  sessionAllocScratch(be, nFloats)

# ── Encoder: pipeline / bind / dispatch / barrier ────────────

proc sessionPipeline*(be: GpuBackend; kernelName: string): bool =
  ## setComputePipelineState — kernel PHẢI đã có trong library Metal.
  when defined(macosx):
    if be == gbMetal or be == gbAuto:
      return metal_enc_pipeline(kernelName.cstring) != 0
  return false

proc sessionBind*(be: GpuBackend; index: int; v: SessionHandle; offsetBytes = 0): bool =
  ## setBuffer:atIndex:
  when defined(macosx):
    if v.metalHandle == nil: return false
    return metal_enc_bind_buffer(cint(index), v.metalHandle, cint(offsetBytes)) != 0
  return false

proc sessionBindInt*(be: GpuBackend; index: int; v: int): bool =
  when defined(macosx):
    return metal_enc_bind_int(cint(index), cint(v)) != 0
  return false

proc sessionBindF32*(be: GpuBackend; index: int; v: float32): bool =
  when defined(macosx):
    return metal_enc_bind_float(cint(index), cfloat(v)) != 0
  return false

proc sessionBindBytes*(be: GpuBackend; index: int; data: pointer; nBytes: int): bool =
  when defined(macosx):
    return metal_enc_bind_bytes(cint(index), data, cint(nBytes)) != 0
  return false

proc sessionDispatch*(be: GpuBackend;
                      gridX: int; gridY = 1; gridZ = 1;
                      tgX = 0; tgY = 1; tgZ = 1): bool =
  ## dispatchThreadgroups. tgX=0 → simd width của pipeline hiện tại.
  when defined(macosx):
    return metal_enc_dispatch(cint(gridX), cint(gridY), cint(gridZ),
                              cint(tgX), cint(tgY), cint(tgZ)) != 0
  return false

proc sessionDispatchThreads*(be: GpuBackend;
                             threadsX: int; threadsY = 1; threadsZ = 1;
                             tgX = 0; tgY = 1; tgZ = 1): bool =
  when defined(macosx):
    return metal_enc_dispatch_threads(cint(threadsX), cint(threadsY), cint(threadsZ),
                                      cint(tgX), cint(tgY), cint(tgZ)) != 0
  return false

proc sessionBarrier*(be: GpuBackend): bool =
  when defined(macosx):
    return metal_enc_barrier() != 0
  return false

proc sessionSimdWidth*(be: GpuBackend): int =
  when defined(macosx):
    return int(metal_enc_simd_width())
  return 32

proc sessionMaxTgThreads*(be: GpuBackend): int =
  when defined(macosx):
    return int(metal_enc_max_tg_threads())
  return 256

# ── Helper: gọi 1 kernel quant 1-byte + LUT (ví dụ fp4 Codestral) ─

proc sessionCallFp4LutMatmul*(be: GpuBackend;
                              hA, hWq, hLut, hC: SessionHandle;
                              M, K, N: int): bool =
  ## Không phải API “cao” — chỉ là mẫu ghép session dưới chân.
  ## Kernel: matmul_fp4_lut_kernel_ndst2_paired (đã có trong vecop_matmul.metal)
  ## Layout: a[M,K] f32, wq[N,K] u8, lut[256] f32, c[M,N] f32
  if not sessionPipeline(be, "matmul_fp4_lut_kernel_ndst2_paired"):
    return false
  if not sessionBind(be, 0, hA): return false
  if not sessionBind(be, 1, hWq): return false
  if not sessionBind(be, 2, hLut): return false
  if not sessionBind(be, 3, hC): return false
  if not sessionBindInt(be, 4, M): return false
  if not sessionBindInt(be, 5, K): return false
  if not sessionBindInt(be, 6, N): return false
  let sw = sessionSimdWidth(be)
  if not sessionBindInt(be, 7, sw): return false
  # NDST=2 trong kernel → grid.x = ceil(N/2), grid.y = M
  let gx = (N + 1) div 2
  if not sessionDispatch(be, gridX = gx, gridY = M, tgX = sw):
    return false
  discard sessionBarrier(be)
  return true
