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
  CUresult = int32
  CUdevice = int32
  CUcontext = pointer
  CUmodule = pointer
  CUfunction = pointer
  CUstream = pointer
  CUdeviceptr* = uint64
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
    cuStreamCreate: proc(phStream: ptr CUstream, flags: uint32): CUresult {.cdecl.}
    cuStreamSynchronize: proc(stream: CUstream): CUresult {.cdecl.}
    cuLaunchKernel: proc(f: CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ,
                          sharedMemBytes: uint32, stream: CUstream, kernelParams: pointer,
                          extra: pointer): CUresult {.cdecl.}
    cuMemsetD8Async: proc(dst: CUdeviceptr, value: uint8, byteCount: csize_t, stream: CUstream): CUresult {.cdecl.}

const
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: CUdevice_attribute = 1

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
  lib.cuStreamCreate = cast[typeof(lib.cuStreamCreate)](lib.handle.symAddr("cuStreamCreate"))
  lib.cuStreamSynchronize = cast[typeof(lib.cuStreamSynchronize)](lib.handle.symAddr("cuStreamSynchronize"))
  lib.cuLaunchKernel = cast[typeof(lib.cuLaunchKernel)](lib.handle.symAddr("cuLaunchKernel"))
  lib.cuMemsetD8Async = cast[typeof(lib.cuMemsetD8Async)](lib.handle.symAddr("cuMemsetD8Async"))
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

# ═══════════════════════════════════════════════════════════════════════════
# API RESIDENT
# ═══════════════════════════════════════════════════════════════════════════

type
  CudaTensor* = object
    dptr*: CUdeviceptr
    bytes*: csize_t
    numel*: int

proc debugLog(msg: string) =
  when defined(nimformerCudaDebug):
    stderr.writeLine("[cuda_driver][DEBUG] " & msg)

proc uploadAsync*(data: seq[float32]): CudaTensor =
  ensureInit()
  let n = data.len
  gpuCheck(n > 0, "uploadAsync: empty input")
  let bytes = csize_t(n * sizeof(float32))
  let dptr = getDeviceBuf(bytes)
  gpuCheck(lib.cuMemcpyHtoDAsync(dptr, unsafeAddr data[0], bytes, gStream) == 0,
           "uploadAsync: H2D failed (n=" & $n & ")")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "uploadAsync: stream sync failed")
  debugLog("uploadAsync n=" & $n & " bytes=" & $bytes)
  result = CudaTensor(dptr: dptr, bytes: bytes, numel: n)

proc uploadIndicesAsync*(data: seq[int32]): CudaTensor =
  ensureInit()
  let n = data.len
  gpuCheck(n > 0, "uploadIndicesAsync: empty input")
  let bytes = csize_t(n * sizeof(int32))
  let dptr = getDeviceBuf(bytes)
  gpuCheck(lib.cuMemcpyHtoDAsync(dptr, unsafeAddr data[0], bytes, gStream) == 0,
           "uploadIndicesAsync: H2D failed (n=" & $n & ")")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "uploadIndicesAsync: stream sync failed")
  debugLog("uploadIndicesAsync n=" & $n & " bytes=" & $bytes)
  result = CudaTensor(dptr: dptr, bytes: bytes, numel: n)

proc downloadSync*(t: CudaTensor): seq[float32] =
  ensureInit()
  gpuCheck(t.dptr != 0, "downloadSync: tensor rỗng/đã free")
  result = newSeq[float32](t.numel)
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], t.dptr, t.bytes, gStream) == 0,
           "downloadSync: D2H failed (numel=" & $t.numel & ")")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "downloadSync: stream sync failed")
  debugLog("downloadSync numel=" & $t.numel)

proc freeResident*(t: var CudaTensor) =
  if t.dptr != 0:
    putDeviceBuf(t.bytes, t.dptr)
    debugLog("freeResident bytes=" & $t.bytes)
  t.dptr = 0
  t.bytes = 0
  t.numel = 0

proc cudaVecOpR*(op: string, a, b: CudaTensor): CudaTensor =
  ensureInit()
  gpuCheck(a.dptr != 0 and b.dptr != 0, "cudaVecOpR: input tensor rỗng/đã free (op=" & op & ")")
  gpuCheck(a.numel == b.numel, "cudaVecOpR: shape mismatch a.numel=" & $a.numel &
           " b.numel=" & $b.numel & " (op=" & op & ")")
  let n = a.numel
  let fn = getFn(fnNameFor(op))
  let dC = getDeviceBuf(a.bytes)
  var dAv = a.dptr; var dBv = b.dptr; var dCv = dC
  var nParam = int32(n)
  var params: array[4, pointer] = [cast[pointer](addr dAv), cast[pointer](addr dBv),
                                    cast[pointer](addr dCv), cast[pointer](addr nParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaVecOpR: cuLaunchKernel failed (op=" & op & ")")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cudaVecOpR: stream sync failed (op=" & op & ")")
  debugLog("cudaVecOpR op=" & op & " n=" & $n)
  result = CudaTensor(dptr: dC, bytes: a.bytes, numel: n)

proc cudaActivationR*(op: string, x: CudaTensor): CudaTensor =
  ensureInit()
  gpuCheck(x.dptr != 0, "cudaActivationR: input tensor rỗng/đã free (op=" & op & ")")
  let n = x.numel
  let fn = getFn("vecop_" & op)
  let dY = getDeviceBuf(x.bytes)
  var dXv = x.dptr; var dYv = dY
  var nParam = int32(n)
  var params: array[3, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dYv), cast[pointer](addr nParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaActivationR: cuLaunchKernel failed (op=" & op & ")")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cudaActivationR: stream sync failed (op=" & op & ")")
  debugLog("cudaActivationR op=" & op & " n=" & $n)
  result = CudaTensor(dptr: dY, bytes: x.bytes, numel: n)

proc cudaSoftmaxR*(x: CudaTensor, rows, cols: int): CudaTensor =
  ensureInit()
  gpuCheck(x.dptr != 0, "cudaSoftmaxR: input tensor rỗng/đã free")
  gpuCheck(x.numel == rows * cols, "cudaSoftmaxR: shape mismatch numel=" & $x.numel &
           " rows*cols=" & $(rows*cols))
  let fn = getFn("softmax_kernel")
  let dY = getDeviceBuf(x.bytes)
  var dXv = x.dptr; var dYv = dY
  var rParam = int32(rows); var cParam = int32(cols)
  var params: array[4, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dYv),
                                    cast[pointer](addr rParam), cast[pointer](addr cParam)]
  gpuCheck(lib.cuLaunchKernel(fn, uint32(rows), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaSoftmaxR: cuLaunchKernel failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cudaSoftmaxR: stream sync failed")
  debugLog("cudaSoftmaxR rows=" & $rows & " cols=" & $cols)
  result = CudaTensor(dptr: dY, bytes: x.bytes, numel: x.numel)

proc cudaLayernormR*(x, gamma, beta: CudaTensor, rows, cols: int, eps: float32): CudaTensor =
  ensureInit()
  gpuCheck(x.dptr != 0 and gamma.dptr != 0 and beta.dptr != 0,
           "cudaLayernormR: input tensor rỗng/đã free")
  gpuCheck(x.numel == rows * cols, "cudaLayernormR: shape mismatch x.numel=" & $x.numel &
           " rows*cols=" & $(rows*cols))
  gpuCheck(gamma.numel == cols and beta.numel == cols,
           "cudaLayernormR: gamma/beta phải có numel=cols=" & $cols &
           " (gamma=" & $gamma.numel & " beta=" & $beta.numel & ")")
  let fn = getFn("layernorm_kernel")
  let dY = getDeviceBuf(x.bytes)
  var dXv = x.dptr; var dGv = gamma.dptr; var dBv = beta.dptr; var dYv = dY
  var rParam = int32(rows); var cParam = int32(cols); var eParam = float32(eps)
  var params: array[7, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dGv), cast[pointer](addr dBv),
                                    cast[pointer](addr dYv), cast[pointer](addr rParam), cast[pointer](addr cParam),
                                    cast[pointer](addr eParam)]
  gpuCheck(lib.cuLaunchKernel(fn, uint32(rows), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaLayernormR: cuLaunchKernel failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cudaLayernormR: stream sync failed")
  debugLog("cudaLayernormR rows=" & $rows & " cols=" & $cols & " eps=" & $eps)
  result = CudaTensor(dptr: dY, bytes: x.bytes, numel: x.numel)

proc cudaEmbeddingLookupR*(table, indices: CudaTensor, numIdx, vocab, dim: int): CudaTensor =
  ensureInit()
  gpuCheck(table.dptr != 0 and indices.dptr != 0, "cudaEmbeddingLookupR: input tensor rỗng/đã free")
  gpuCheck(table.numel == vocab * dim, "cudaEmbeddingLookupR: table.numel=" & $table.numel &
           " != vocab*dim=" & $(vocab*dim))
  gpuCheck(indices.numel == numIdx, "cudaEmbeddingLookupR: indices.numel=" & $indices.numel &
           " != numIdx=" & $numIdx)
  let fn = getFn("embedding_lookup_kernel")
  let bytesY = csize_t(numIdx * dim * sizeof(float32))
  let dY = getDeviceBuf(bytesY)
  var dTv = table.dptr; var dIdxv = indices.dptr; var dYv = dY
  var vParam = int32(vocab); var dParam = int32(dim); var numParam = int32(numIdx)
  var params: array[6, pointer] = [cast[pointer](addr dTv), cast[pointer](addr dIdxv), cast[pointer](addr dYv),
                                    cast[pointer](addr vParam), cast[pointer](addr dParam), cast[pointer](addr numParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(numIdx), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaEmbeddingLookupR: cuLaunchKernel failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cudaEmbeddingLookupR: stream sync failed")
  debugLog("cudaEmbeddingLookupR numIdx=" & $numIdx & " vocab=" & $vocab & " dim=" & $dim)
  result = CudaTensor(dptr: dY, bytes: bytesY, numel: numIdx * dim)

# ============================================================
# CUDA ATTENTION FUSED FORWARD
# ============================================================
# ============================================================
# CUDA ATTENTION FUSED FORWARD
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
  let dOut = getDeviceBuf(bytesQKV)          # SỬA: dO -> dOut
  let dS = getDeviceBuf(bytesS)
  defer:
    putDeviceBuf(bytesQKV, dQ)
    putDeviceBuf(bytesQKV, dK)
    putDeviceBuf(bytesQKV, dV)
    putDeviceBuf(bytesQKV, dOut)             # SỬA: dO -> dOut
    putDeviceBuf(bytesS, dS)
  
  gpuCheck(lib.cuMemcpyHtoDAsync(dQ, unsafeAddr q[0], bytesQKV, gStream) == 0, "H2D Q failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dK, unsafeAddr k[0], bytesQKV, gStream) == 0, "H2D K failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dV, unsafeAddr v[0], bytesQKV, gStream) == 0, "H2D V failed")
  
  let fn = getFn("attention_fused_kernel")
  
  var bQ = dQ; var bK = dK; var bV = dV; var bOut = dOut; var bS = dS   # SỬA: bO -> bOut
  var bArg = int32(B); var hArg = int32(H); var sArg = int32(S); var dArg = int32(D); var scArg = float32(scale)
  
  var params: array[10, pointer] = [
    cast[pointer](addr bQ), cast[pointer](addr bK), cast[pointer](addr bV),
    cast[pointer](addr bOut), cast[pointer](addr bS),                   # SỬA: bO -> bOut
    cast[pointer](addr bArg), cast[pointer](addr hArg),
    cast[pointer](addr sArg), cast[pointer](addr dArg),
    cast[pointer](addr scArg)
  ]
  
  let blockSize = getOptimalBlockSize()
  gpuCheck(lib.cuLaunchKernel(fn, uint32(B*H), uint32(S), 1, blockSize, 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "attention_fused_kernel launch failed")
  
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.o[0], dOut, bytesQKV, gStream) == 0, "D2H O failed")  # SỬA: dO -> dOut
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.s_matrix[0], dS, bytesS, gStream) == 0, "D2H S failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "stream sync failed")
# ============================================================
# CUDA ATTENTION FUSED BACKWARD
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