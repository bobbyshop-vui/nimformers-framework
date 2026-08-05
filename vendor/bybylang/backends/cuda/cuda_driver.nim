# cuda_driver.nim - Runtime bindings to the CUDA Driver API (libcuda) loaded via dynlib.
# Không phụ thuộc lúc build: nếu máy không có CUDA/GPU NVIDIA, module này chỉ trả về
# "không khả dụng" thay vì làm sập chương trình -> cho phép fallback CPU an toàn.
#
# SỬA SO VỚI BẢN CŨ:
# - Context/module/stream/function được tạo ĐÚNG 1 LẦN (lazy, tại lần gọi đầu) và
#   giữ sống suốt vòng đời process, thay vì cuCtxCreate + cuModuleLoadData mỗi lần gọi.
# - Dùng 1 CUstream không-NULL, tạo 1 lần, dùng lại cho mọi kernel + copy.
# - Copy H2D/D2H dùng bản Async trên stream đó, chỉ cuStreamSynchronize MỘT LẦN
#   ở cuối mỗi proc (thay vì đồng bộ ngầm ở từng cuMemcpyHtoD/DtoH).
# - Host staging buffer dùng pinned memory (cuMemAllocHost) thay vì heap thường,
#   để tránh CUDA driver phải tự copy sang buffer ẩn trước khi DMA lên GPU.
# - Device buffer + pinned host buffer đều lấy từ pool theo kích thước (byte),
#   KHÔNG cuMemAlloc/cuMemFree hay cuMemAllocHost/cuMemFreeHost mỗi lần gọi nữa.
import std/dynlib
import std/os
import std/tables
type
  CUdeviceptr* = uint64
  CUstream* = pointer
  CUcontext* = pointer
  CUmodule* = pointer
  CUfunction* = pointer
  CUdevice* = int32
  CUresult* = int32
  CUgraph* = pointer
  CUgraphExec* = pointer

  CudaTensor* = object
    dptr*: CUdeviceptr
    bytes*: csize_t
    numel*: int
type
  CUdevice_attribute = int32

  CudaLib = object
    handle: LibHandle
    cuInit: proc(flags: uint32): CUresult {.cdecl.}
    cuDeviceGetCount: proc(count: ptr int32): CUresult {.cdecl.}
    cuDeviceGet: proc(dev: ptr CUdevice, ordinal: int32): CUresult {.cdecl.}
    cuDeviceGetAttribute: proc(pi: ptr int32, attrib: CUdevice_attribute, dev: CUdevice): CUresult {.cdecl.}
    cuCtxCreate: proc(pctx: ptr CUcontext, flags: uint32, dev: CUdevice): CUresult {.cdecl.}
    cuCtxDestroy: proc(ctx: CUcontext): CUresult {.cdecl.}
    cuModuleLoadData: proc(module: ptr CUmodule, image: cstring): CUresult {.cdecl.}
    cuModuleGetFunction: proc(hfunc: ptr CUfunction, hmod: CUmodule, name: cstring): CUresult {.cdecl.}
    cuMemAlloc: proc(dptr: ptr CUdeviceptr, bytesize: csize_t): CUresult {.cdecl.}
    cuMemFree: proc(dptr: CUdeviceptr): CUresult {.cdecl.}
    cuMemAllocHost: proc(pp: ptr pointer, bytesize: csize_t): CUresult {.cdecl.}
    cuMemFreeHost: proc(p: pointer): CUresult {.cdecl.}
    cuMemcpyHtoD: proc(dst: CUdeviceptr, src: pointer, byteCount: csize_t): CUresult {.cdecl.}
    cuMemcpyDtoH: proc(dst: pointer, src: CUdeviceptr, byteCount: csize_t): CUresult {.cdecl.}
    cuMemcpyHtoDAsync: proc(dst: CUdeviceptr, src: pointer, byteCount: csize_t, stream: CUstream): CUresult {.cdecl.}
    cuMemcpyDtoHAsync: proc(dst: pointer, src: CUdeviceptr, byteCount: csize_t, stream: CUstream): CUresult {.cdecl.}
    cuMemcpyDtoDAsync: proc(dst: CUdeviceptr, src: CUdeviceptr, byteCount: csize_t, stream: CUstream): CUresult {.cdecl.}
    cuStreamCreate: proc(phStream: ptr CUstream, flags: uint32): CUresult {.cdecl.}
    cuStreamSynchronize: proc(stream: CUstream): CUresult {.cdecl.}
    cuLaunchKernel: proc(f: CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ,
                          sharedMemBytes: uint32, stream: CUstream, kernelParams: pointer,
                          extra: pointer): CUresult {.cdecl.}
    cuMemsetD8Async: proc(dst: CUdeviceptr, value: uint8, byteCount: csize_t, stream: CUstream): CUresult {.cdecl.}
    cuStreamBeginCapture: proc(stream: CUstream, mode: uint32): CUresult {.cdecl.}
    cuStreamEndCapture: proc(stream: CUstream, phGraph: ptr CUgraph): CUresult {.cdecl.}
    cuGraphInstantiate: proc(phGraphExec: ptr CUgraphExec, hGraph: CUgraph, flags: uint64): CUresult {.cdecl.}
    cuGraphLaunch: proc(hGraphExec: CUgraphExec, stream: CUstream): CUresult {.cdecl.}
    cuGraphDestroy: proc(hGraph: CUgraph): CUresult {.cdecl.}
    cuGraphExecDestroy: proc(hGraphExec: CUgraphExec): CUresult {.cdecl.}

const
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: CUdevice_attribute = 1
  CU_STREAM_CAPTURE_MODE_RELAXED: uint32 = 2

# --------------------------------------------------------------------------
# PTX kernels (tương đương nvcc -ptx cho 4 phép elementwise float32), nằm ở
# file thật backends/cuda/kernels/vecop.ptx (KHÔNG còn escape-string trong
# .nim nữa) -- 4 entry point riêng: vecop_add/vecop_sub/vecop_mul/vecop_div,
# gộp trong cùng 1 module PTX, nạp 1 lần bằng cuModuleLoadData tại runtime,
# không cần nvcc trên máy chạy.
# --------------------------------------------------------------------------
const ptxSource = staticRead(currentSourcePath().parentDir() / "kernels" / "vecop.ptx")

proc fnNameFor(op: string): string =
  case op
  of "add": "vecop_add"
  of "sub": "vecop_sub"
  of "mul": "vecop_mul"
  of "div": "vecop_div"
  else: "vecop_add"

proc gpuCheck(cond: bool, msg: string) =
  ## Giống doAssert nhưng raise CatchableError thay vì Defect, để gpubackend.nim
  ## có thể bắt lỗi và fallback CPU thay vì làm crash chương trình.
  if not cond:
    raise newException(CatchableError, msg)

var lib: CudaLib
var loaded = false

# ---- State toàn cục, khởi tạo 1 lần (ensureInit), sống tới khi process thoát ----
var gInitialized = false
var gCtx: CUcontext
var gModule: CUmodule
var gStream: CUstream
var gFnCache = initTable[string, CUfunction]()

# ---- Pool device memory + pinned host memory, key theo số byte ----
var gDevicePool = initTable[csize_t, seq[CUdeviceptr]]()
var gHostPool = initTable[csize_t, seq[pointer]]()

# ---- Tối ưu block size theo GPU ----
var gOptimalBlockSize: uint32 = 256
var gBlockSizeInitialized = false

proc getOptimalBlockSize(): uint32 =
  if gBlockSizeInitialized: return gOptimalBlockSize
  var dev: CUdevice
  var maxThreads: int32
  if lib.cuDeviceGet(addr dev, 0) == 0:
    if lib.cuDeviceGetAttribute(addr maxThreads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev) == 0:
      if maxThreads >= 1024: gOptimalBlockSize = 512
      elif maxThreads >= 512: gOptimalBlockSize = 256
      else: gOptimalBlockSize = 128
  gBlockSizeInitialized = true
  result = gOptimalBlockSize

proc gridFor(n: int): uint32 =
  let blockSize = getOptimalBlockSize()
  uint32((n + int(blockSize) - 1) div int(blockSize))

proc tryLoad(): bool =
  if loaded: return lib.handle != nil
  loaded = true
  # Tên thư viện phổ biến trên Linux / Windows.
  for name in ["libcuda.so", "libcuda.so.1", "nvcuda.dll"]:
    lib.handle = loadLib(name)
    if lib.handle != nil: break
  if lib.handle == nil:
    return false
  lib.cuInit = cast[typeof(lib.cuInit)](lib.handle.symAddr("cuInit"))
  lib.cuDeviceGetCount = cast[typeof(lib.cuDeviceGetCount)](lib.handle.symAddr("cuDeviceGetCount"))
  lib.cuDeviceGet = cast[typeof(lib.cuDeviceGet)](lib.handle.symAddr("cuDeviceGet"))
  lib.cuDeviceGetAttribute = cast[typeof(lib.cuDeviceGetAttribute)](lib.handle.symAddr("cuDeviceGetAttribute"))
  lib.cuCtxCreate = cast[typeof(lib.cuCtxCreate)](lib.handle.symAddr("cuCtxCreate_v2"))
  lib.cuCtxDestroy = cast[typeof(lib.cuCtxDestroy)](lib.handle.symAddr("cuCtxDestroy_v2"))
  lib.cuModuleLoadData = cast[typeof(lib.cuModuleLoadData)](lib.handle.symAddr("cuModuleLoadData"))
  lib.cuModuleGetFunction = cast[typeof(lib.cuModuleGetFunction)](lib.handle.symAddr("cuModuleGetFunction"))
  lib.cuMemAlloc = cast[typeof(lib.cuMemAlloc)](lib.handle.symAddr("cuMemAlloc_v2"))
  lib.cuMemFree = cast[typeof(lib.cuMemFree)](lib.handle.symAddr("cuMemFree_v2"))
  lib.cuMemAllocHost = cast[typeof(lib.cuMemAllocHost)](lib.handle.symAddr("cuMemAllocHost_v2"))
  lib.cuMemFreeHost = cast[typeof(lib.cuMemFreeHost)](lib.handle.symAddr("cuMemFreeHost"))
  lib.cuMemcpyHtoD = cast[typeof(lib.cuMemcpyHtoD)](lib.handle.symAddr("cuMemcpyHtoD_v2"))
  lib.cuMemcpyDtoH = cast[typeof(lib.cuMemcpyDtoH)](lib.handle.symAddr("cuMemcpyDtoH_v2"))
  lib.cuMemcpyHtoDAsync = cast[typeof(lib.cuMemcpyHtoDAsync)](lib.handle.symAddr("cuMemcpyHtoDAsync_v2"))
  lib.cuMemcpyDtoHAsync = cast[typeof(lib.cuMemcpyDtoHAsync)](lib.handle.symAddr("cuMemcpyDtoHAsync_v2"))
  lib.cuMemcpyDtoDAsync = cast[typeof(lib.cuMemcpyDtoDAsync)](lib.handle.symAddr("cuMemcpyDtoDAsync_v2"))
  lib.cuStreamCreate = cast[typeof(lib.cuStreamCreate)](lib.handle.symAddr("cuStreamCreate"))
  lib.cuStreamSynchronize = cast[typeof(lib.cuStreamSynchronize)](lib.handle.symAddr("cuStreamSynchronize"))
  lib.cuLaunchKernel = cast[typeof(lib.cuLaunchKernel)](lib.handle.symAddr("cuLaunchKernel"))
  lib.cuMemsetD8Async = cast[typeof(lib.cuMemsetD8Async)](lib.handle.symAddr("cuMemsetD8Async"))
  lib.cuStreamBeginCapture = cast[typeof(lib.cuStreamBeginCapture)](lib.handle.symAddr("cuStreamBeginCapture"))
  lib.cuStreamEndCapture = cast[typeof(lib.cuStreamEndCapture)](lib.handle.symAddr("cuStreamEndCapture"))
  lib.cuGraphInstantiate = cast[typeof(lib.cuGraphInstantiate)](lib.handle.symAddr("cuGraphInstantiate"))
  lib.cuGraphLaunch = cast[typeof(lib.cuGraphLaunch)](lib.handle.symAddr("cuGraphLaunch"))
  lib.cuGraphDestroy = cast[typeof(lib.cuGraphDestroy)](lib.handle.symAddr("cuGraphDestroy"))
  lib.cuGraphExecDestroy = cast[typeof(lib.cuGraphExecDestroy)](lib.handle.symAddr("cuGraphExecDestroy"))
  result = lib.cuInit != nil and lib.cuLaunchKernel != nil

proc cudaAvailable*(): bool =
  ## Dò xem máy có driver NVIDIA (libcuda) và ít nhất 1 GPU hay không.
  if not tryLoad(): return false
  if lib.cuInit(0) != 0: return false
  var count: int32 = 0
  if lib.cuDeviceGetCount(addr count) != 0: return false
  result = count > 0

proc ensureInit() =
  ## Tạo context + module PTX + stream ĐÚNG 1 LẦN cho suốt vòng đời process.
  ## Các lần gọi sau chỉ return ngay - không còn cuCtxCreate/cuModuleLoadData
  ## lặp lại mỗi phép toán như bản cũ.
  if gInitialized: return
  gpuCheck(tryLoad(), "libcuda not found")
  var dev: CUdevice
  let initRes = lib.cuInit(0)
  gpuCheck(initRes == 0, "cuInit failed (error code: " & $initRes & ")")
  
  let devRes = lib.cuDeviceGet(addr dev, 0)
  gpuCheck(devRes == 0, "cuDeviceGet failed (error code: " & $devRes & ")")
  
  let ctxRes = lib.cuCtxCreate(addr gCtx, 0, dev)
  gpuCheck(ctxRes == 0, "cuCtxCreate failed (error code: " & $ctxRes & ")")
  
  let modRes = lib.cuModuleLoadData(addr gModule, ptxSource.cstring)
  gpuCheck(modRes == 0, "cuModuleLoadData failed (PTX) (error code: " & $modRes & "). This usually means the PTX version or format is incompatible with the installed CUDA driver.")
  
  let streamRes = lib.cuStreamCreate(addr gStream, 0)
  gpuCheck(streamRes == 0, "cuStreamCreate failed (error code: " & $streamRes & ")")
  gInitialized = true

proc getStream*(): CUstream =
  ensureInit()
  result = gStream

proc cuStreamSynchronize*(stream: CUstream): int32 =
  ## Sync 1 CUDA stream cụ thể (wrapper public cho lib.cuStreamSynchronize private).
  ensureInit()
  result = lib.cuStreamSynchronize(stream)

proc getFn(name: string): CUfunction =
  ## Cache CUfunction theo tên - tránh cuModuleGetFunction lặp lại.
  if gFnCache.hasKey(name):
    return gFnCache[name]
  var fn: CUfunction
  gpuCheck(lib.cuModuleGetFunction(addr fn, gModule, name.cstring) == 0,
           "cuModuleGetFunction failed for " & name)
  gFnCache[name] = fn
  result = fn

proc getDeviceBuf*(bytes: csize_t): CUdeviceptr =
  ## Lấy device buffer từ pool nếu có sẵn cùng kích thước, không thì cuMemAlloc.
  ## Export (*) để cuda_runtime.nim dùng chung 1 pool cho CudaTensor resident
  ## (matmul resident cũng cấp phát qua đây, thay vì mở pool riêng bằng cudaMalloc,
  ## để tránh 2 hệ thống pool tách rời cho cùng 1 loại buffer).
  if gDevicePool.hasKey(bytes) and gDevicePool[bytes].len > 0:
    return gDevicePool[bytes].pop()
  var dptr: CUdeviceptr
  gpuCheck(lib.cuMemAlloc(addr dptr, bytes) == 0, "cuMemAlloc failed")
  result = dptr

proc putDeviceBuf*(bytes: csize_t, dptr: CUdeviceptr) =
  ## Trả buffer về pool để tái sử dụng thay vì cuMemFree ngay.
  gDevicePool.mgetOrPut(bytes, @[]).add(dptr)

proc getHostBuf(bytes: csize_t): pointer =
  ## Lấy pinned host staging buffer từ pool, không thì cuMemAllocHost.
  if gHostPool.hasKey(bytes) and gHostPool[bytes].len > 0:
    return gHostPool[bytes].pop()
  var p: pointer
  gpuCheck(lib.cuMemAllocHost(addr p, bytes) == 0, "cuMemAllocHost failed")
  result = p

proc putHostBuf(bytes: csize_t, p: pointer) =
  gHostPool.mgetOrPut(bytes, @[]).add(p)
proc uploadAsync*(data: seq[float32]): CudaTensor =
  ensureInit()
  let bytes = csize_t(data.len * sizeof(float32))
  let dptr = getDeviceBuf(bytes)
  gpuCheck(lib.cuMemcpyHtoDAsync(dptr, unsafeAddr data[0], bytes, gStream) == 0, "uploadAsync H2D failed")
  return CudaTensor(dptr: dptr, bytes: bytes, numel: data.len)

proc uploadIndicesAsync*(data: seq[int32]): CudaTensor =
  ensureInit()
  let bytes = csize_t(data.len * sizeof(int32))
  let dptr = getDeviceBuf(bytes)
  gpuCheck(lib.cuMemcpyHtoDAsync(dptr, unsafeAddr data[0], bytes, gStream) == 0, "uploadIndicesAsync H2D failed")
  return CudaTensor(dptr: dptr, bytes: bytes, numel: data.len)

proc downloadSync*(t: CudaTensor): seq[float32] =
  ensureInit()
  result = newSeq[float32](t.numel)
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], t.dptr, t.bytes, gStream) == 0, "downloadSync D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "downloadSync sync failed")

proc freeResident*(t: var CudaTensor) =
  if t.dptr != 0:
    putDeviceBuf(t.bytes, t.dptr)
    t.dptr = 0
    t.bytes = 0
    t.numel = 0
# ============================================================
# VECTOR OPERATIONS
# ============================================================
proc cudaVecOp*(op: string, a, b: seq[float32]): seq[float32] =
  ## Chạy phép toán elementwise (add/sub/mul/div) trên GPU NVIDIA qua CUDA Driver API.
  ## Raise nếu có lỗi bất kỳ, để tầng gọi (gpubackend) fallback về CPU.
  ensureInit()
  let n = a.len
  result = newSeq[float32](n)
  let fn = getFn(fnNameFor(op))
  let bytes = csize_t(n * sizeof(float32))

  let dA = getDeviceBuf(bytes)
  let dB = getDeviceBuf(bytes)
  let dC = getDeviceBuf(bytes)
  defer:
    putDeviceBuf(bytes, dA); putDeviceBuf(bytes, dB); putDeviceBuf(bytes, dC)

  gpuCheck(lib.cuMemcpyHtoDAsync(dA, unsafeAddr a[0], bytes, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dB, unsafeAddr b[0], bytes, gStream) == 0, "H2D failed")

  var dAv = dA; var dBv = dB; var dCv = dC
  var nParam = int32(n)
  var params: array[4, pointer] = [cast[pointer](addr dAv), cast[pointer](addr dBv),
                                    cast[pointer](addr dCv), cast[pointer](addr nParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dC, bytes, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

# ============================================================
# ACTIVATIONS
# ============================================================
proc cudaActivation*(op: string, x: seq[float32]): seq[float32] =
  ensureInit()
  let n = x.len
  result = newSeq[float32](n)
  let fn = getFn("vecop_" & op)
  let bytes = csize_t(n * sizeof(float32))

  let dX = getDeviceBuf(bytes)
  let dY = getDeviceBuf(bytes)
  defer:
    putDeviceBuf(bytes, dX); putDeviceBuf(bytes, dY)

  gpuCheck(lib.cuMemcpyHtoDAsync(dX, unsafeAddr x[0], bytes, gStream) == 0, "H2D failed")

  var dXv = dX; var dYv = dY
  var nParam = int32(n)
  var params: array[3, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dYv), cast[pointer](addr nParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dY, bytes, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

# ============================================================
# APFLU FORWARD
# ============================================================
proc cudaApflu*(x: seq[float32], alpha: float32, beta: float32): seq[float32] =
  ensureInit()
  let n = x.len
  result = newSeq[float32](n)
  let fn = getFn("vecop_apflu")
  let bytes = csize_t(n * sizeof(float32))

  let dX = getDeviceBuf(bytes)
  let dY = getDeviceBuf(bytes)
  defer:
    putDeviceBuf(bytes, dX); putDeviceBuf(bytes, dY)

  gpuCheck(lib.cuMemcpyHtoDAsync(dX, unsafeAddr x[0], bytes, gStream) == 0, "H2D failed")

  var dXv = dX; var dYv = dY
  var nParam = int32(n)
  var aParam = alpha
  var bParam = beta
  var params: array[5, pointer] = [
    cast[pointer](addr dXv), cast[pointer](addr dYv),
    cast[pointer](addr nParam), cast[pointer](addr aParam),
    cast[pointer](addr bParam)
  ]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dY, bytes, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

# ============================================================
# APFLU BACKWARD
# ============================================================
proc cudaApfluBackward*(x, dy: seq[float32], alpha: float32, beta: float32): seq[float32] =
  ensureInit()
  let n = x.len
  result = newSeq[float32](n)
  let fn = getFn("vecop_apflu_backward")
  let bytes = csize_t(n * sizeof(float32))

  let dXin = getDeviceBuf(bytes)      # SỬA: dXin
  let dDy = getDeviceBuf(bytes)
  let dDxOut = getDeviceBuf(bytes)    # SỬA: dDxOut
  defer:
    putDeviceBuf(bytes, dXin); putDeviceBuf(bytes, dDy); putDeviceBuf(bytes, dDxOut)

  gpuCheck(lib.cuMemcpyHtoDAsync(dXin, unsafeAddr x[0], bytes, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dDy, unsafeAddr dy[0], bytes, gStream) == 0, "H2D failed")

  var dXinV = dXin; var dDyV = dDy; var dDxOutV = dDxOut
  var nParam = int32(n)
  var aParam = alpha
  var bParam = beta
  var params: array[6, pointer] = [
    cast[pointer](addr dXinV), cast[pointer](addr dDyV), cast[pointer](addr dDxOutV),
    cast[pointer](addr nParam), cast[pointer](addr aParam),
    cast[pointer](addr bParam)
  ]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dDxOut, bytes, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")
# ============================================================
# SOFTMAX
# ============================================================
proc cudaSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
  ensureInit()
  let n = rows * cols
  result = newSeq[float32](n)
  let fn = getFn("softmax_kernel")
  let bytes = csize_t(n * sizeof(float32))

  let dX = getDeviceBuf(bytes)
  let dY = getDeviceBuf(bytes)
  defer:
    putDeviceBuf(bytes, dX); putDeviceBuf(bytes, dY)

  gpuCheck(lib.cuMemcpyHtoDAsync(dX, unsafeAddr x[0], bytes, gStream) == 0, "H2D failed")

  var dXv = dX; var dYv = dY
  var rParam = int32(rows)
  var cParam = int32(cols)
  var params: array[4, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dYv),
                                    cast[pointer](addr rParam), cast[pointer](addr cParam)]
  gpuCheck(lib.cuLaunchKernel(fn, uint32(rows), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dY, bytes, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

# ============================================================
# LAYERNORM FORWARD
# ============================================================
proc cudaLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
  ensureInit()
  let n = rows * cols
  result = newSeq[float32](n)
  let fn = getFn("layernorm_kernel")

  let bytesX = csize_t(n * sizeof(float32))
  let bytesC = csize_t(cols * sizeof(float32))
  let dX = getDeviceBuf(bytesX)
  let dGamma = getDeviceBuf(bytesC)
  let dBeta = getDeviceBuf(bytesC)
  let dY = getDeviceBuf(bytesX)
  defer:
    putDeviceBuf(bytesX, dX); putDeviceBuf(bytesC, dGamma); putDeviceBuf(bytesC, dBeta); putDeviceBuf(bytesX, dY)

  gpuCheck(lib.cuMemcpyHtoDAsync(dX, unsafeAddr x[0], bytesX, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dGamma, unsafeAddr gamma[0], bytesC, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dBeta, unsafeAddr beta[0], bytesC, gStream) == 0, "H2D failed")

  var dXv = dX; var dGv = dGamma; var dBv = dBeta; var dYv = dY
  var rParam = int32(rows)
  var cParam = int32(cols)
  var eParam = float32(eps)
  var params: array[7, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dGv), cast[pointer](addr dBv),
                                    cast[pointer](addr dYv), cast[pointer](addr rParam), cast[pointer](addr cParam),
                                    cast[pointer](addr eParam)]
  gpuCheck(lib.cuLaunchKernel(fn, uint32(rows), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dY, bytesX, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

proc cudaLayernormBackward*(dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  ensureInit()
  let n = rows * cols
  var dx = newSeq[float32](n)
  var dgamma = newSeq[float32](cols)
  var dbeta = newSeq[float32](cols)

  let fn = getFn("layernorm_backward_kernel")

  let bytesX = csize_t(n * sizeof(float32))
  let bytesC = csize_t(cols * sizeof(float32))

  let dDyBuf = getDeviceBuf(bytesX)      # SỬA: dDyBuf
  let dXBuf = getDeviceBuf(bytesX)       # SỬA: dXBuf
  let dGammaBuf = getDeviceBuf(bytesC)   # SỬA: dGammaBuf
  let dBetaBuf = getDeviceBuf(bytesC)    # SỬA: dBetaBuf
  let dDxBuf = getDeviceBuf(bytesX)      # SỬA: dDxBuf
  let dDgammaBuf = getDeviceBuf(bytesC)  # SỬA: dDgammaBuf
  let dDbetaBuf = getDeviceBuf(bytesC)   # SỬA: dDbetaBuf
  defer:
    putDeviceBuf(bytesX, dDyBuf); putDeviceBuf(bytesX, dXBuf)
    putDeviceBuf(bytesC, dGammaBuf); putDeviceBuf(bytesC, dBetaBuf)
    putDeviceBuf(bytesX, dDxBuf); putDeviceBuf(bytesC, dDgammaBuf)
    putDeviceBuf(bytesC, dDbetaBuf)

  gpuCheck(lib.cuMemcpyHtoDAsync(dDyBuf, unsafeAddr dy[0], bytesX, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dXBuf, unsafeAddr x[0], bytesX, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dGammaBuf, unsafeAddr gamma[0], bytesC, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dBetaBuf, unsafeAddr beta[0], bytesC, gStream) == 0, "H2D failed")

  var dDyV = dDyBuf; var dXV = dXBuf; var dGammaV = dGammaBuf; var dBetaV = dBetaBuf
  var dDxV = dDxBuf; var dDgammaV = dDgammaBuf; var dDbetaV = dDbetaBuf
  var rParam = int32(rows)
  var cParam = int32(cols)
  var eParam = float32(eps)
  var params: array[10, pointer] = [
    cast[pointer](addr dDyV), cast[pointer](addr dXV),
    cast[pointer](addr dGammaV), cast[pointer](addr dBetaV),
    cast[pointer](addr dDxV), cast[pointer](addr dDgammaV),
    cast[pointer](addr dDbetaV),
    cast[pointer](addr rParam), cast[pointer](addr cParam),
    cast[pointer](addr eParam)
  ]
  # FIX: layernorm_backward_kernel la kernel kieu "1 thread = 1 id doc lap"
  # (giong OpenCL get_global_id, khong cong tac giua cac thread trong cung block),
  # khac voi layernorm_kernel (forward) la kieu "1 block = 1 row, nhieu thread cong tac".
  # Truoc day dung getOptimalBlockSize() (128/256/512 thread/block) khien moi block
  # chi tinh duoc id trong khoang [0, blockSize) roi GHI DE len nhau -> sai ket qua /
  # race giua cac block -> CUDA context bi loi ngam -> D2H that bai o lan goi ke tiep.
  gpuCheck(lib.cuLaunchKernel(fn, uint32(max(rows, cols)), 1, 1, 1, 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr dx[0], dDxBuf, bytesX, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr dgamma[0], dDgammaBuf, bytesC, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr dbeta[0], dDbetaBuf, bytesC, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")
  result = (dx, dgamma, dbeta)
# ============================================================
# EMBEDDING LOOKUP
# ============================================================
proc cudaEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
  ensureInit()
  let num = indices.len
  result = newSeq[float32](num * dim)
  let fn = getFn("embedding_lookup_kernel")

  let bytesTable = csize_t(vocab * dim * sizeof(float32))
  let bytesIndices = csize_t(num * sizeof(int32))
  let bytesY = csize_t(num * dim * sizeof(float32))
  let dTable = getDeviceBuf(bytesTable)
  let dIndices = getDeviceBuf(bytesIndices)
  let dY = getDeviceBuf(bytesY)
  defer:
    putDeviceBuf(bytesTable, dTable); putDeviceBuf(bytesIndices, dIndices); putDeviceBuf(bytesY, dY)

  gpuCheck(lib.cuMemcpyHtoDAsync(dTable, unsafeAddr table[0], bytesTable, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dIndices, unsafeAddr indices[0], bytesIndices, gStream) == 0, "H2D failed")

  var dTv = dTable; var dIdxv = dIndices; var dYv = dY
  var vParam = int32(vocab)
  var dParam = int32(dim)
  var numParam = int32(num)
  var params: array[6, pointer] = [cast[pointer](addr dTv), cast[pointer](addr dIdxv), cast[pointer](addr dYv),
                                    cast[pointer](addr vParam), cast[pointer](addr dParam), cast[pointer](addr numParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(num), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dY, bytesY, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

# ============================================================
# ATTENTION FUSED FORWARD
# ============================================================
proc cudaAttentionFused*(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  ensureInit()
  let qkvLen = B * H * S * D
  let sLen = B * H * S * S
  
  result.o = newSeq[float32](qkvLen)
  result.s_matrix = newSeq[float32](sLen)
  
  let bytesQKV = csize_t(qkvLen * sizeof(float32))
  let bytesS = csize_t(sLen * sizeof(float32))
  
  let dQ = getDeviceBuf(bytesQKV)
  let dK = getDeviceBuf(bytesQKV)
  let dV = getDeviceBuf(bytesQKV)
  let dOut = getDeviceBuf(bytesQKV)
  let dS = getDeviceBuf(bytesS)
  defer:
    putDeviceBuf(bytesQKV, dQ)
    putDeviceBuf(bytesQKV, dK)
    putDeviceBuf(bytesQKV, dV)
    putDeviceBuf(bytesQKV, dOut)
    putDeviceBuf(bytesS, dS)
  
  gpuCheck(lib.cuMemcpyHtoDAsync(dQ, unsafeAddr q[0], bytesQKV, gStream) == 0, "H2D Q failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dK, unsafeAddr k[0], bytesQKV, gStream) == 0, "H2D K failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dV, unsafeAddr v[0], bytesQKV, gStream) == 0, "H2D V failed")
  
  let fn = getFn("attention_fused_kernel")
  
  var bQ = dQ; var bK = dK; var bV = dV; var bOut = dOut; var bS = dS
  var bArg = int32(B); var hArg = int32(H); var sArg = int32(S); var dArg = int32(D); var scArg = float32(scale)
  
  var params: array[10, pointer] = [
    cast[pointer](addr bQ), cast[pointer](addr bK), cast[pointer](addr bV),
    cast[pointer](addr bOut), cast[pointer](addr bS),
    cast[pointer](addr bArg), cast[pointer](addr hArg),
    cast[pointer](addr sArg), cast[pointer](addr dArg),
    cast[pointer](addr scArg)
  ]
  
  let blockSize = getOptimalBlockSize()
  gpuCheck(lib.cuLaunchKernel(fn, uint32(B*H), uint32(S), 1, blockSize, 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "attention_fused_kernel launch failed")
  
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.o[0], dOut, bytesQKV, gStream) == 0, "D2H O failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.s_matrix[0], dS, bytesS, gStream) == 0, "D2H S failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "stream sync failed")

proc cudaAttentionFusedR*(q, k, v: CudaTensor, B, H, S, D: int, scale: float32): tuple[o, s_matrix: CudaTensor] =
  ## Bản "resident" của cudaAttentionFused ở trên: nhận thẳng CudaTensor đã
  ## resident trên device (không phải seq[float32] trên host), launch kernel
  ## lên CÙNG stream (gStream) và KHÔNG gọi cuStreamSynchronize/D2H - để lời
  ## gọi này encode xong là quay về ngay, y hệt cudaLayernormR/cudaVecOpR.
  ## Dùng cho session API (xem gpubackend.nim: sessionAttentionFusedEnc) -
  ## nhiều lệnh launch nối tiếp lên cùng stream, chỉ sync đúng 1 lần ở
  ## sessionEnd(), thay vì mỗi attention tự đứng chờ GPU xong ngay tại chỗ.
  ensureInit()
  let qkvLen = B * H * S * D
  let sLen = B * H * S * S
  let bytesQKV = csize_t(qkvLen * sizeof(float32))
  let bytesS = csize_t(sLen * sizeof(float32))

  let dOut = getDeviceBuf(bytesQKV)
  let dS = getDeviceBuf(bytesS)

  let fn = getFn("attention_fused_kernel")
  var bQ = q.dptr; var bK = k.dptr; var bV = v.dptr; var bOut = dOut; var bS = dS
  var bArg = int32(B); var hArg = int32(H); var sArg = int32(S); var dArg = int32(D); var scArg = float32(scale)

  var params: array[10, pointer] = [
    cast[pointer](addr bQ), cast[pointer](addr bK), cast[pointer](addr bV),
    cast[pointer](addr bOut), cast[pointer](addr bS),
    cast[pointer](addr bArg), cast[pointer](addr hArg),
    cast[pointer](addr sArg), cast[pointer](addr dArg),
    cast[pointer](addr scArg)
  ]

  let blockSize = getOptimalBlockSize()
  gpuCheck(lib.cuLaunchKernel(fn, uint32(B*H), uint32(S), 1, blockSize, 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "attention_fused_kernel (resident) launch failed")

  result.o = CudaTensor(dptr: dOut, bytes: bytesQKV, numel: qkvLen)
  result.s_matrix = CudaTensor(dptr: dS, bytes: bytesS, numel: sLen)

# ============================================================
# ATTENTION FUSED BACKWARD
# ============================================================
proc cudaAttentionFusedBackward*(q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  ensureInit()
  let qkvLen = B * H * S * D
  let sLen = B * H * S * S
  
  result.dq = newSeq[float32](qkvLen)
  result.dk = newSeq[float32](qkvLen)
  result.dv = newSeq[float32](qkvLen)
  
  let bytesQKV = csize_t(qkvLen * sizeof(float32))
  let bytesS = csize_t(sLen * sizeof(float32))
  
  let dQ = getDeviceBuf(bytesQKV)
  let dK = getDeviceBuf(bytesQKV)
  let dV = getDeviceBuf(bytesQKV)
  let dS = getDeviceBuf(bytesS)
  let dDy = getDeviceBuf(bytesQKV)
  let dDq = getDeviceBuf(bytesQKV)
  let dDk = getDeviceBuf(bytesQKV)
  let dDv = getDeviceBuf(bytesQKV)
  defer:
    putDeviceBuf(bytesQKV, dQ); putDeviceBuf(bytesQKV, dK)
    putDeviceBuf(bytesQKV, dV); putDeviceBuf(bytesS, dS)
    putDeviceBuf(bytesQKV, dDy); putDeviceBuf(bytesQKV, dDq)
    putDeviceBuf(bytesQKV, dDk); putDeviceBuf(bytesQKV, dDv)
  
  gpuCheck(lib.cuMemcpyHtoDAsync(dQ, unsafeAddr q[0], bytesQKV, gStream) == 0, "H2D Q failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dK, unsafeAddr k[0], bytesQKV, gStream) == 0, "H2D K failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dV, unsafeAddr v[0], bytesQKV, gStream) == 0, "H2D V failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dS, unsafeAddr s_matrix[0], bytesS, gStream) == 0, "H2D S failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dDy, unsafeAddr dy[0], bytesQKV, gStream) == 0, "H2D dy failed")
  
  gpuCheck(lib.cuMemsetD8Async(dDq, 0, bytesQKV, gStream) == 0, "zero init dq failed")
  gpuCheck(lib.cuMemsetD8Async(dDk, 0, bytesQKV, gStream) == 0, "zero init dk failed")
  gpuCheck(lib.cuMemsetD8Async(dDv, 0, bytesQKV, gStream) == 0, "zero init dv failed")
  
  let fn = getFn("attention_fused_backward_kernel")
  
  var bQ = dQ; var bK = dK; var bV = dV; var bS = dS; var bDy = dDy
  var bDq = dDq; var bDk = dDk; var bDv = dDv
  var bArg = int32(B); var hArg = int32(H); var sArg = int32(S); var dArg = int32(D); var scArg = float32(scale)
  
  var params: array[13, pointer] = [
    cast[pointer](addr bQ), cast[pointer](addr bK), cast[pointer](addr bV),
    cast[pointer](addr bS), cast[pointer](addr bDy),
    cast[pointer](addr bDq), cast[pointer](addr bDk), cast[pointer](addr bDv),
    cast[pointer](addr bArg), cast[pointer](addr hArg),
    cast[pointer](addr sArg), cast[pointer](addr dArg),
    cast[pointer](addr scArg)
  ]
  
  let blockSize = getOptimalBlockSize()
  gpuCheck(lib.cuLaunchKernel(fn, uint32(B*H), uint32(S), 1, blockSize, 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "attention_fused_backward_kernel launch failed")
  
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.dq[0], dDq, bytesQKV, gStream) == 0, "D2H dq failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.dk[0], dDk, bytesQKV, gStream) == 0, "D2H dk failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.dv[0], dDv, bytesQKV, gStream) == 0, "D2H dv failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "stream sync failed")
# ─────────────────────────────────────────────────────────────
# CUDA Resident Vector Ops
# ─────────────────────────────────────────────────────────────

proc cudaVecOpR*(op: string, a, b: CudaTensor): CudaTensor =
  ensureInit()
  gpuCheck(a.numel == b.numel, "cudaVecOpR: size mismatch")
  let n = a.numel
  let bytes = a.bytes
  let dC = getDeviceBuf(bytes)
  
  let fn = getFn(fnNameFor(op))
  var dAv = a.dptr; var dBv = b.dptr; var dCv = dC
  var nParam = int32(n)
  var params: array[4, pointer] = [cast[pointer](addr dAv), cast[pointer](addr dBv),
                                    cast[pointer](addr dCv), cast[pointer](addr nParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaVecOpR launch failed")
  return CudaTensor(dptr: dC, bytes: bytes, numel: n)

proc cudaActivationR*(op: string, x: CudaTensor): CudaTensor =
  ensureInit()
  let n = x.numel
  let bytes = x.bytes
  let dY = getDeviceBuf(bytes)
  
  let fn = getFn("vecop_" & op)
  var dXv = x.dptr; var dYv = dY
  var nParam = int32(n)
  var params: array[3, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dYv), cast[pointer](addr nParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaActivationR launch failed")
  return CudaTensor(dptr: dY, bytes: bytes, numel: n)

proc cudaSoftmaxR*(x: CudaTensor, rows, cols: int): CudaTensor =
  ensureInit()
  let n = rows * cols
  let bytes = csize_t(n * sizeof(float32))
  let dY = getDeviceBuf(bytes)
  
  let fn = getFn("softmax_kernel")
  var dXv = x.dptr; var dYv = dY
  var rParam = int32(rows); var cParam = int32(cols)
  var params: array[4, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dYv),
                                    cast[pointer](addr rParam), cast[pointer](addr cParam)]
  gpuCheck(lib.cuLaunchKernel(fn, uint32(rows), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaSoftmaxR launch failed")
  return CudaTensor(dptr: dY, bytes: bytes, numel: n)

proc cudaLayernormR*(x, gamma, beta: CudaTensor, rows, cols: int, eps: float32): CudaTensor =
  ensureInit()
  let n = rows * cols
  let bytesX = csize_t(n * sizeof(float32))
  let bytesC = csize_t(cols * sizeof(float32))
  let dY = getDeviceBuf(bytesX)
  
  let fn = getFn("layernorm_kernel")
  var dXv = x.dptr; var dGv = gamma.dptr; var dBv = beta.dptr; var dYv = dY
  var rParam = int32(rows); var cParam = int32(cols); var eParam = float32(eps)
  var params: array[7, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dGv), cast[pointer](addr dBv),
                                    cast[pointer](addr dYv), cast[pointer](addr rParam), cast[pointer](addr cParam),
                                    cast[pointer](addr eParam)]
  gpuCheck(lib.cuLaunchKernel(fn, uint32(rows), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaLayernormR launch failed")
  return CudaTensor(dptr: dY, bytes: bytesX, numel: n)

# ============================================================
# RMSNORM (mirror cudaLayernormR ở trên, bỏ beta + không trừ mean - xem
# rmsnorm_kernel.cu). CẦN: entry point "rmsnorm_kernel" phải có trong
# kernels/vecop.ptx (build bằng `nvcc -ptx rmsnorm_kernel.cu` rồi nối vào
# .ptx hiện có, ĐÚNG quy trình đã dùng cho rope_kernel - xem comment ở
# cudaRopeR phía trên). Nếu module PTX đang nạp CHƯA có entry này, getFn
# raise CatchableError - sessionRmsnormEnc (gpubackend.nim) BẮT lỗi này,
# coi là "chưa hỗ trợ", rơi về đường CPU cũ, KHÔNG để lọt ngoại lệ ra ngoài.
# ============================================================
proc cudaRmsnormR*(x, gamma: CudaTensor, rows, cols: int, eps: float32): CudaTensor =
  ensureInit()
  let n = rows * cols
  let bytesX = csize_t(n * sizeof(float32))
  let dY = getDeviceBuf(bytesX)

  let fn = getFn("rmsnorm_kernel")
  var dXv = x.dptr; var dGv = gamma.dptr; var dYv = dY
  var rParam = int32(rows); var cParam = int32(cols); var eParam = float32(eps)
  var params: array[6, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dGv),
                                    cast[pointer](addr dYv), cast[pointer](addr rParam),
                                    cast[pointer](addr cParam), cast[pointer](addr eParam)]
  # KHÔNG cuStreamSynchronize ở đây - giữ đúng semantics session giống mọi
  # lệnh session khác (matmul/vecop/rope đã sửa/layernorm/attention): chỉ
  # launch async lên gStream, sync đúng 1 lần ở sessionEnd().
  gpuCheck(lib.cuLaunchKernel(fn, uint32(rows), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaRmsnormR launch failed")
  return CudaTensor(dptr: dY, bytes: bytesX, numel: n)

# ============================================================
# ROPE (rotate_half, in-place) - mirror của rope_kernel bên Metal/OpenCL.
# CẦN: entry point "rope_kernel" phải có trong kernels/vecop.ptx (build từ
# rope_kernel.cu bằng `nvcc -ptx` rồi nối vào file .ptx hiện có). Nếu module
# PTX đang nạp CHƯA có entry này, getFn sẽ raise CatchableError như mọi kernel
# khác - proc gọi nó ở gpubackend.nim (sessionRope) BẮT lỗi này và coi là
# "chưa hỗ trợ", rơi về CPU, giống hệt hành vi cũ trước khi kernel tồn tại.
# ============================================================
proc cudaRopeR*(x, cosT, sinT: CudaTensor, T, Hn, D: int) =
  ## In-place: ghi đè thẳng lên x.dptr, không cấp/trả buffer mới.
  ensureInit()
  let fn = getFn("rope_kernel")
  let halfD = D div 2

  var dXv = x.dptr; var dCv = cosT.dptr; var dSv = sinT.dptr
  var hnParam = int32(Hn)
  var dParam = int32(D)
  var params: array[5, pointer] = [
    cast[pointer](addr dXv), cast[pointer](addr dCv), cast[pointer](addr dSv),
    cast[pointer](addr hnParam), cast[pointer](addr dParam)
  ]

  # grid.x phủ D/2 (chia theo block size tối ưu của GPU), grid.y = Hn, grid.z = T
  # - đúng layout đã dùng bên Metal/OpenCL (gid.x=i, gid.y=h, gid.z=t).
  let blockSize = getOptimalBlockSize()
  let gridX = uint32((halfD + int(blockSize) - 1) div int(blockSize))
  gpuCheck(lib.cuLaunchKernel(fn, gridX, uint32(Hn), uint32(T), blockSize, 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "rope_kernel launch failed")
  # ESCAPE-HATCH: nếu sau khi build/test trên máy CUDA thật mà thấy token
  # rác VÀ nghi ngờ đúng do bỏ sync này (vd. driver/CUDA version cụ thể nào
  # đó không đảm bảo thứ tự lệnh trên cùng stream như tài liệu NVIDIA mô tả
  # - rất khó xảy ra nhưng chưa tự tay xác nhận trên phần cứng thật tại đây),
  # set BYBY_LEGACY_ROPE_SYNC=1 để quay lại NGUYÊN VẸN hành vi cũ (có sync
  # sau mỗi lần gọi RoPE) mà không cần revert code - dùng để bisect nhanh
  # xem token rác có do thay đổi này hay không TRƯỚC KHI nghi ngờ chỗ khác.
  if existsEnv("BYBY_LEGACY_ROPE_SYNC"):
    gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cudaRopeR legacy sync failed")
  # SỬA (BUG THẬT, không phải suy đoán): trước đây có cuStreamSynchronize()
  # NGAY TẠI ĐÂY - tức MỖI lần gọi RoPE (mỗi layer, mỗi token) ép CPU đứng
  # chờ GPU xong NGAY GIỮA session đang mở, dù RoPE chỉ là 1 lệnh encode như
  # matmul/vecop/copyInto (các lệnh đó KHÔNG sync ở đây, chỉ nối tiếp trên
  # cùng gStream, chỉ sync đúng 1 lần ở sessionEnd). Đây chính là điểm phá vỡ
  # toàn bộ thiết kế "gộp N lệnh vào 1 session, chỉ sync 1 lần cuối" CHỈ
  # RIÊNG trên CUDA (Metal/OpenCL sessionRope tương ứng không có dòng sync
  # nào) - biến ropeFused=true trên CUDA từ "1 session/layer" ngầm trở lại
  # thành có thêm 1 sync đồng bộ/layer, y hệt kiểu overhead mà phần còn lại
  # của file này đã cố loại bỏ. Không sync ở đây nữa - để sessionEnd() (gọi
  # cuStreamSynchronize 1 lần) chịu trách nhiệm đồng bộ hoá duy nhất.

# ============================================================
# COPY INTO (GPU-side concat) - THÊM MỚI: dst[dstOffsetElems ..< +n] =
# src[0 ..< n], device-to-device, KHÔNG cần kernel riêng (khác Metal/OpenCL,
# CUDA driver có sẵn cuMemcpyDtoDAsync). Dùng để ghép nhiều buffer con (vd
# outHeads từng KV-head trong attention decode) thành 1 buffer liên tục
# NGAY TRÊN GPU trong session đang mở, tránh phải đọc về CPU rồi upload lại
# chỉ để nối mảng - đây là điểm chặn cuối cùng khiến session attention-decode
# và session oProj không gộp được làm 1 (xem giải thích tương tự bên
# metal_copy_into_enc/openclSessionCopyIntoEnc).
# LƯU Ý: dst PHẢI đã được cấp đủ (dstOffsetElems + n) phần tử từ trước
# (cuAllocScratch) - hàm này không tự resize.
# ============================================================
proc cudaCopyIntoR*(src, dst: CudaTensor, dstOffsetElems, n: int) =
  ensureInit()
  let byteOffset = csize_t(dstOffsetElems * sizeof(float32))
  let n64 = csize_t(n * sizeof(float32))
  gpuCheck(byteOffset + n64 <= dst.bytes, "cudaCopyIntoR: dstOffset+n vuot qua kich thuoc buffer dich")
  gpuCheck(lib.cuMemcpyDtoDAsync(dst.dptr + byteOffset, src.dptr, n64, gStream) == 0,
           "cuMemcpyDtoDAsync failed (cudaCopyIntoR)")
  # Không cuStreamSynchronize ở đây - giữ đúng semantics session: các lệnh
  # khác (matmul/vecop/rope...) trên CÙNG gStream vẫn nối tiếp không blocking,
  # chỉ sync 1 lần ở sessionEnd (giống mọi thao tác khác trong session CUDA).

proc cudaEmbeddingLookupR*(table, indices: CudaTensor, numIdx, vocab, dim: int): CudaTensor =
  ensureInit()
  let bytesY = csize_t(numIdx * dim * sizeof(float32))
  let dY = getDeviceBuf(bytesY)
  
  let fn = getFn("embedding_lookup_kernel")
  var dTv = table.dptr; var dIdxv = indices.dptr; var dYv = dY
  var vParam = int32(vocab); var dParam = int32(dim); var numParam = int32(numIdx)
  var params: array[6, pointer] = [cast[pointer](addr dTv), cast[pointer](addr dIdxv), cast[pointer](addr dYv),
                                    cast[pointer](addr vParam), cast[pointer](addr dParam), cast[pointer](addr numParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(numIdx), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaEmbeddingLookupR launch failed")
  return CudaTensor(dptr: dY, bytes: bytesY, numel: numIdx * dim)

type
  CudaGraph* = object
    graph: CUgraph
    exec: CUgraphExec

var gCudaGraph: CudaGraph
var gCudaGraphCapturing: bool = false

proc cuStreamBeginCapture*(stream: CUstream): bool =
  ensureInit()
  let res = lib.cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED)
  if res == 0:
    gCudaGraphCapturing = true
    return true
  return false

proc cuStreamEndCapture*(stream: CUstream): bool =
  if not gCudaGraphCapturing: return false
  let res = lib.cuStreamEndCapture(stream, addr gCudaGraph.graph)
  if res == 0:
    gCudaGraphCapturing = false
    return true
  return false

proc cuGraphInstantiate*(): bool =
  let res = lib.cuGraphInstantiate(addr gCudaGraph.exec, gCudaGraph.graph, 0'u64)
  return res == 0

proc cuGraphLaunch*(stream: CUstream): bool =
  let res = lib.cuGraphLaunch(gCudaGraph.exec, stream)
  return res == 0

proc cuGraphDestroy*() =
  discard lib.cuGraphDestroy(gCudaGraph.graph)
  discard lib.cuGraphExecDestroy(gCudaGraph.exec)

proc isCudaGraphCapturing*(): bool = gCudaGraphCapturing

# ============================================================
# MATMUL INT4 ASYMMETRIC (packed, per-group scale/zero_point) - THÊM MỚI
# ============================================================
# Mirror của matmul_q4_kernel_naive (Metal) / matmul_q4_naive (OpenCL). Cần
# entry "matmul_q4_kernel" có trong module PTX đang nạp (kernels/vecop.ptx) -
# xem hướng dẫn build trong kernels/matmul_q4_kernel.cu. Nếu module hiện tại
# CHƯA có entry này, getFn() sẽ raise CatchableError - nơi gọi (gpubackend.nim)
# BẮT lỗi này và tự rơi về cpuDequantMatmulQ4Fallback, KHÔNG làm sập process.

proc uploadBytesAsync*(data: seq[uint8]): CudaTensor =
  ## Upload byte thô (weight int4 đã pack) - dùng resident 1 lần lúc load
  ## model (giống uploadAsync ở trên nhưng cho uint8 thay vì float32).
  ensureInit()
  let bytes = csize_t(data.len)
  let dptr = getDeviceBuf(bytes)
  gpuCheck(lib.cuMemcpyHtoDAsync(dptr, unsafeAddr data[0], bytes, gStream) == 0, "uploadBytesAsync H2D failed")
  return CudaTensor(dptr: dptr, bytes: bytes, numel: data.len)

proc cudaMatmulQ4R*(a: CudaTensor, wq, scales, zeros: CudaTensor,
                     M, K, N, groupSize, nGroupsPerRow: int): CudaTensor =
  ## Bản resident: a/wq/scales/zeros đã ở trên GPU (a đến từ uploadAsync mỗi
  ## token, wq/scales/zeros đến từ uploadBytesAsync/uploadAsync MỘT LẦN lúc
  ## load model qua residentUploadQ4). Không upload lại weight mỗi lần gọi -
  ## đây chính là điểm CUDA đang THIẾU so với Metal/OpenCL (xem comment ở
  ## gpubackend.nim::ResidentWeightHandle).
  ensureInit()
  let bytesC = csize_t(M * N * sizeof(float32))
  let dC = getDeviceBuf(bytesC)

  # SUA: dùng kernel warp-shuffle "matmul_q4_kernel" (mirror của Metal
  # matmul_q4_kernel dùng simd_sum / OpenCL matmul_q4_kernel_sg dùng
  # sub_group_reduce_add) thay vì bản naive 1-thread-per-output cũ. Đồng
  # thời SỬA BUG: params trước đây thiếu hẳn ngArg (nGroupsPerRow) - mảng
  # chỉ khai báo 9 phần tử trong khi kernel cần 10 tham số, khiến
  # nGroupsPerRow đọc rác từ vùng nhớ kế tiếp trên GPU (silent corruption,
  # không crash rõ ràng vì CUDA không kiểm tra số lượng params khớp chữ ký).
  let fn = getFn("matmul_q4_kernel")
  var dAv = a.dptr; var dWqv = wq.dptr; var dScv = scales.dptr; var dZpv = zeros.dptr; var dCv = dC
  var mArg = int32(M); var kArg = int32(K); var nArg = int32(N)
  var gsArg = int32(groupSize); var ngArg = int32(nGroupsPerRow)
  var params: array[10, pointer] = [
    cast[pointer](addr dAv), cast[pointer](addr dWqv), cast[pointer](addr dScv), cast[pointer](addr dZpv),
    cast[pointer](addr dCv), cast[pointer](addr mArg), cast[pointer](addr kArg), cast[pointer](addr nArg),
    cast[pointer](addr gsArg), cast[pointer](addr ngArg)]
  # block = (32,1,1): đúng 1 warp/block - MỖI warp xử lý MMQ4_NDST=4 hàng
  # output (n) cùng lúc cho 1 hàng activation (m), dùng __shfl_down_sync để
  # reduce thay vì shared memory (xem matmul_q4_kernel.cu). grid.x =
  # ceilDiv(N,4), grid.y = M (thường =1 lúc decode).
  const MMQ4_NDST = 4'u32
  let blockX = 32'u32; let blockY = 1'u32
  let gridX = (uint32(N) + MMQ4_NDST - 1) div MMQ4_NDST
  let gridY = uint32(M)
  gpuCheck(lib.cuLaunchKernel(fn, gridX, gridY, 1, blockX, blockY, 1, 0, gStream,
                                addr params[0], nil) == 0, "matmul_q4_kernel launch failed")
  return CudaTensor(dptr: dC, bytes: bytesC, numel: M * N)

proc cudaMatmulQ4*(a: seq[float32], wq: seq[uint8], scales, zeros: seq[float32],
                    M, K, N, groupSize, nGroupsPerRow: int): seq[float32] =
  ## Bản immediate (không resident) - dùng khi gọi rời rạc ngoài session dài
  ## hơi (tương đương openclMatmulQ4/metalMatmulQ4). Upload+launch+download
  ## trong 1 lượt, giải phóng buffer ngay sau khi đọc kết quả về.
  ensureInit()
  let ta = uploadAsync(a)
  let twq = uploadBytesAsync(wq)
  let tsc = uploadAsync(scales)
  let tzp = uploadAsync(zeros)
  var tc = cudaMatmulQ4R(ta, twq, tsc, tzp, M, K, N, groupSize, nGroupsPerRow)
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cudaMatmulQ4 sync failed")
  result = downloadSync(tc)
  var taV = ta; var twqV = twq; var tscV = tsc; var tzpV = tzp
  freeResident(taV); freeResident(twqV); freeResident(tscV); freeResident(tzpV); freeResident(tc)