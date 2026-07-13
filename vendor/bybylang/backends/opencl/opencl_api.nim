# opencl_api.nim - Runtime bindings to OpenCL
import std/dynlib
import std/os
import std/tables

type
  cl_int = int32
  cl_uint = uint32
  cl_platform_id = pointer
  cl_device_id = pointer
  cl_context = pointer
  cl_command_queue = pointer
  cl_program = pointer
  cl_kernel = pointer
  cl_mem = pointer

  OclLib = object
    handle: LibHandle
    clGetPlatformIDs: proc(num_entries: cl_uint, platforms: ptr cl_platform_id, num_platforms: ptr cl_uint): cl_int {.cdecl.}
    clGetDeviceIDs: proc(platform: cl_platform_id, device_type: uint64, num_entries: cl_uint,
                          devices: ptr cl_device_id, num_devices: ptr cl_uint): cl_int {.cdecl.}
    clCreateContext: proc(properties: pointer, num_devices: cl_uint, devices: ptr cl_device_id,
                           pfn_notify: pointer, user_data: pointer, errcode_ret: ptr cl_int): cl_context {.cdecl.}
    clCreateCommandQueue: proc(context: cl_context, device: cl_device_id, properties: uint64,
                                errcode_ret: ptr cl_int): cl_command_queue {.cdecl.}
    clCreateBuffer: proc(context: cl_context, flags: uint64, size: csize_t, host_ptr: pointer,
                          errcode_ret: ptr cl_int): cl_mem {.cdecl.}
    clCreateProgramWithSource: proc(context: cl_context, count: cl_uint, strings: ptr cstring,
                                     lengths: ptr csize_t, errcode_ret: ptr cl_int): cl_program {.cdecl.}
    clBuildProgram: proc(program: cl_program, num_devices: cl_uint, device_list: ptr cl_device_id,
                          options: cstring, pfn_notify: pointer, user_data: pointer): cl_int {.cdecl.}
    clCreateKernel: proc(program: cl_program, kernel_name: cstring, errcode_ret: ptr cl_int): cl_kernel {.cdecl.}
    clSetKernelArg: proc(kernel: cl_kernel, arg_index: cl_uint, arg_size: csize_t, arg_value: pointer): cl_int {.cdecl.}
    clEnqueueNDRangeKernel: proc(command_queue: cl_command_queue, kernel: cl_kernel, work_dim: cl_uint,
                                  global_work_offset: ptr csize_t, global_work_size: ptr csize_t,
                                  local_work_size: ptr csize_t, num_events_in_wait_list: cl_uint,
                                  event_wait_list: pointer, event: pointer): cl_int {.cdecl.}
    clEnqueueWriteBuffer: proc(command_queue: cl_command_queue, buffer: cl_mem, blocking_write: uint32,
                                offset, size: csize_t, ptrData: pointer, num_events: cl_uint,
                                wait_list: pointer, event: pointer): cl_int {.cdecl.}
    clEnqueueReadBuffer: proc(command_queue: cl_command_queue, buffer: cl_mem, blocking_read: uint32,
                               offset, size: csize_t, ptrData: pointer, num_events: cl_uint,
                               wait_list: pointer, event: pointer): cl_int {.cdecl.}
    clFinish: proc(command_queue: cl_command_queue): cl_int {.cdecl.}
    clReleaseMemObject: proc(memobj: cl_mem): cl_int {.cdecl.}
    clReleaseContext: proc(context: cl_context): cl_int {.cdecl.}
    clReleaseCommandQueue: proc(command_queue: cl_command_queue): cl_int {.cdecl.}
    clReleaseProgram: proc(program: cl_program): cl_int {.cdecl.}
    clReleaseKernel: proc(kernel: cl_kernel): cl_int {.cdecl.}

const kernelSource = staticRead(currentSourcePath().parentDir() / "kernels" / "vecop_matmul.cl")

const CL_DEVICE_TYPE_ALL: uint64 = 0xFFFFFFFF'u64
const CL_MEM_READ_ONLY: uint64 = 4
const CL_MEM_WRITE_ONLY: uint64 = 2
const CL_MEM_READ_WRITE: uint64 = 1
const CL_TRUE: uint32 = 1
const CL_FALSE: uint32 = 0

proc gpuCheck(cond: bool, msg: string) =
  if not cond:
    raise newException(CatchableError, msg)

var lib: OclLib
var loaded = false

var gInitialized = false
var gPlatform: cl_platform_id
var gDevice: cl_device_id
var gCtx: cl_context
var gQueue: cl_command_queue
var gProgram: cl_program
var gKernelCache = initTable[string, cl_kernel]()
var gBufPool = initTable[(csize_t, uint64), seq[cl_mem]]()

proc tryLoad(): bool =
  if loaded: return lib.handle != nil
  loaded = true
  for name in ["libOpenCL.so", "libOpenCL.so.1", "OpenCL.dll", "/System/Library/Frameworks/OpenCL.framework/OpenCL"]:
    lib.handle = loadLib(name)
    if lib.handle != nil: break
  if lib.handle == nil: return false
  lib.clGetPlatformIDs = cast[typeof(lib.clGetPlatformIDs)](lib.handle.symAddr("clGetPlatformIDs"))
  lib.clGetDeviceIDs = cast[typeof(lib.clGetDeviceIDs)](lib.handle.symAddr("clGetDeviceIDs"))
  lib.clCreateContext = cast[typeof(lib.clCreateContext)](lib.handle.symAddr("clCreateContext"))
  lib.clCreateCommandQueue = cast[typeof(lib.clCreateCommandQueue)](lib.handle.symAddr("clCreateCommandQueue"))
  lib.clCreateBuffer = cast[typeof(lib.clCreateBuffer)](lib.handle.symAddr("clCreateBuffer"))
  lib.clCreateProgramWithSource = cast[typeof(lib.clCreateProgramWithSource)](lib.handle.symAddr("clCreateProgramWithSource"))
  lib.clBuildProgram = cast[typeof(lib.clBuildProgram)](lib.handle.symAddr("clBuildProgram"))
  lib.clCreateKernel = cast[typeof(lib.clCreateKernel)](lib.handle.symAddr("clCreateKernel"))
  lib.clSetKernelArg = cast[typeof(lib.clSetKernelArg)](lib.handle.symAddr("clSetKernelArg"))
  lib.clEnqueueNDRangeKernel = cast[typeof(lib.clEnqueueNDRangeKernel)](lib.handle.symAddr("clEnqueueNDRangeKernel"))
  lib.clEnqueueWriteBuffer = cast[typeof(lib.clEnqueueWriteBuffer)](lib.handle.symAddr("clEnqueueWriteBuffer"))
  lib.clEnqueueReadBuffer = cast[typeof(lib.clEnqueueReadBuffer)](lib.handle.symAddr("clEnqueueReadBuffer"))
  lib.clFinish = cast[typeof(lib.clFinish)](lib.handle.symAddr("clFinish"))
  lib.clReleaseMemObject = cast[typeof(lib.clReleaseMemObject)](lib.handle.symAddr("clReleaseMemObject"))
  lib.clReleaseContext = cast[typeof(lib.clReleaseContext)](lib.handle.symAddr("clReleaseContext"))
  lib.clReleaseCommandQueue = cast[typeof(lib.clReleaseCommandQueue)](lib.handle.symAddr("clReleaseCommandQueue"))
  lib.clReleaseProgram = cast[typeof(lib.clReleaseProgram)](lib.handle.symAddr("clReleaseProgram"))
  lib.clReleaseKernel = cast[typeof(lib.clReleaseKernel)](lib.handle.symAddr("clReleaseKernel"))
  result = lib.clGetPlatformIDs != nil and lib.clCreateContext != nil

proc findPlatformAndDevice(platform_ret: ptr cl_platform_id, device_ret: ptr cl_device_id): bool =
  var num_platforms: cl_uint = 0
  if lib.clGetPlatformIDs(0, nil, addr num_platforms) != 0 or num_platforms == 0:
    return false
  var platforms = newSeq[cl_platform_id](num_platforms)
  if lib.clGetPlatformIDs(num_platforms, addr platforms[0], nil) != 0:
    return false

  const CL_DEVICE_TYPE_GPU: uint64 = 4
  for plat in platforms:
    var num_devices: cl_uint = 0
    if lib.clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nil, addr num_devices) == 0 and num_devices > 0:
      var devices = newSeq[cl_device_id](num_devices)
      if lib.clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devices, addr devices[0], nil) == 0:
        platform_ret[] = plat
        device_ret[] = devices[0]
        return true

  for plat in platforms:
    var num_devices: cl_uint = 0
    if lib.clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nil, addr num_devices) == 0 and num_devices > 0:
      var devices = newSeq[cl_device_id](num_devices)
      if lib.clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, num_devices, addr devices[0], nil) == 0:
        platform_ret[] = plat
        device_ret[] = devices[0]
        return true
  return false

proc ensureInit() =
  if gInitialized: return
  gpuCheck(tryLoad(), "libOpenCL not found")
  gpuCheck(findPlatformAndDevice(addr gPlatform, addr gDevice), "Khong tim thay platform/device OpenCL")

  var err: cl_int
  var props: array[3, int]
  props[0] = 0x1084
  props[1] = cast[int](gPlatform)
  props[2] = 0
  gCtx = lib.clCreateContext(cast[pointer](addr props[0]), 1, addr gDevice, nil, nil, addr err)
  gpuCheck(err == 0 and gCtx != nil, "clCreateContext failed")

  gQueue = lib.clCreateCommandQueue(gCtx, gDevice, 0, addr err)
  gpuCheck(err == 0 and gQueue != nil, "clCreateCommandQueue failed")

  var srcPtr = kernelSource.cstring
  gProgram = lib.clCreateProgramWithSource(gCtx, 1, addr srcPtr, nil, addr err)
  gpuCheck(err == 0 and gProgram != nil, "clCreateProgramWithSource failed")
  gpuCheck(lib.clBuildProgram(gProgram, 1, addr gDevice, nil, nil, nil) == 0, "clBuildProgram failed")

  gInitialized = true

proc getKernel(name: string): cl_kernel =
  if gKernelCache.hasKey(name):
    return gKernelCache[name]
  var err: cl_int
  let k = lib.clCreateKernel(gProgram, name.cstring, addr err)
  gpuCheck(err == 0 and k != nil, "clCreateKernel failed for " & name)
  gKernelCache[name] = k
  result = k

proc getBuf(bytes: csize_t, flags: uint64): cl_mem =
  let key = (bytes, flags)
  if gBufPool.hasKey(key) and gBufPool[key].len > 0:
    return gBufPool[key].pop()
  var err: cl_int
  let m = lib.clCreateBuffer(gCtx, flags, bytes, nil, addr err)
  gpuCheck(err == 0, "clCreateBuffer failed")
  result = m

proc putBuf(bytes: csize_t, flags: uint64, m: cl_mem) =
  gBufPool.mgetOrPut((bytes, flags), @[]).add(m)

proc openclAvailable*(): bool =
  if not tryLoad(): return false
  var platform: cl_platform_id
  var device: cl_device_id
  result = findPlatformAndDevice(addr platform, addr device)

proc openclVecOp*(op: string, a, b: seq[float32]): seq[float32] =
  ensureInit()
  let n = a.len
  result = newSeq[float32](n)
  let bytes = csize_t(n * sizeof(float32))
  let bufA = getBuf(bytes, CL_MEM_READ_ONLY)
  let bufB = getBuf(bytes, CL_MEM_READ_ONLY)
  let bufC = getBuf(bytes, CL_MEM_WRITE_ONLY)
  defer:
    putBuf(bytes, CL_MEM_READ_ONLY, bufA); putBuf(bytes, CL_MEM_READ_ONLY, bufB); putBuf(bytes, CL_MEM_WRITE_ONLY, bufC)
  var aVar = a; var bVar = b
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufA, CL_FALSE, 0, bytes, addr aVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufB, CL_FALSE, 0, bytes, addr bVar[0], 0, nil, nil) == 0, "write failed")
  let kernel = getKernel("vecop_" & op)
  var bA = bufA; var bB = bufB; var bC = bufC
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bA) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bB) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bC) == 0, "setArg failed")
  var globalSize = csize_t(n)
  gpuCheck(lib.clEnqueueNDRangeKernel(gQueue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufC, CL_TRUE, 0, bytes, addr result[0], 0, nil, nil) == 0, "read failed")

proc openclMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
  ensureInit()
  result = newSeq[float32](m * n)
  let bytesA = csize_t(m * k * sizeof(float32))
  let bytesB = csize_t(k * n * sizeof(float32))
  let bytesC = csize_t(m * n * sizeof(float32))
  let bufA = getBuf(bytesA, CL_MEM_READ_ONLY)
  let bufB = getBuf(bytesB, CL_MEM_READ_ONLY)
  let bufC = getBuf(bytesC, CL_MEM_WRITE_ONLY)
  defer:
    putBuf(bytesA, CL_MEM_READ_ONLY, bufA); putBuf(bytesB, CL_MEM_READ_ONLY, bufB); putBuf(bytesC, CL_MEM_WRITE_ONLY, bufC)
  var aVar = a; var bVar = b
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufA, CL_FALSE, 0, bytesA, addr aVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufB, CL_FALSE, 0, bytesB, addr bVar[0], 0, nil, nil) == 0, "write failed")
  let kernel = getKernel("matmul_naive")
  var bA = bufA; var bB = bufB; var bC = bufC
  var mArg = int32(m); var kArg = int32(k); var nArg = int32(n)
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bA) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bB) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bC) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(int32)), addr mArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 4, csize_t(sizeof(int32)), addr kArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 5, csize_t(sizeof(int32)), addr nArg) == 0, "setArg failed")
  const TILE = csize_t(16)
  proc roundUp(x, tile: csize_t): csize_t = ((x + tile - 1) div tile) * tile
  var globalSize2D = [roundUp(csize_t(m), TILE), roundUp(csize_t(n), TILE)]
  var localSize2D = [TILE, TILE]
  gpuCheck(lib.clEnqueueNDRangeKernel(gQueue, kernel, 2, nil, addr globalSize2D[0], addr localSize2D[0], 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufC, CL_TRUE, 0, bytesC, addr result[0], 0, nil, nil) == 0, "read failed")

proc openclActivation*(op: string, x: seq[float32]): seq[float32] =
  ensureInit()
  let n = x.len
  result = newSeq[float32](n)
  let bytes = csize_t(n * sizeof(float32))
  let bufX = getBuf(bytes, CL_MEM_READ_ONLY)
  let bufY = getBuf(bytes, CL_MEM_WRITE_ONLY)
  defer:
    putBuf(bytes, CL_MEM_READ_ONLY, bufX); putBuf(bytes, CL_MEM_WRITE_ONLY, bufY)
  var xVar = x
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufX, CL_FALSE, 0, bytes, addr xVar[0], 0, nil, nil) == 0, "write failed")
  let kernel = getKernel("vecop_" & op)
  var bX = bufX; var bY = bufY
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bX) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bY) == 0, "setArg failed")
  var globalSize = csize_t(n)
  gpuCheck(lib.clEnqueueNDRangeKernel(gQueue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufY, CL_TRUE, 0, bytes, addr result[0], 0, nil, nil) == 0, "read failed")

proc openclSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
  ensureInit()
  let n = rows * cols
  result = newSeq[float32](n)
  let bytes = csize_t(n * sizeof(float32))
  let bufX = getBuf(bytes, CL_MEM_READ_ONLY)
  let bufY = getBuf(bytes, CL_MEM_WRITE_ONLY)
  defer:
    putBuf(bytes, CL_MEM_READ_ONLY, bufX); putBuf(bytes, CL_MEM_WRITE_ONLY, bufY)
  var xVar = x
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufX, CL_FALSE, 0, bytes, addr xVar[0], 0, nil, nil) == 0, "write failed")
  let kernel = getKernel("softmax_kernel")
  var bX = bufX; var bY = bufY
  var cArg = int32(cols)
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bX) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bY) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(int32)), addr cArg) == 0, "setArg failed")
  const BB_WG = csize_t(256)
  var globalSize = csize_t(rows) * BB_WG
  var localSize = BB_WG
  gpuCheck(lib.clEnqueueNDRangeKernel(gQueue, kernel, 1, nil, addr globalSize, addr localSize, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufY, CL_TRUE, 0, bytes, addr result[0], 0, nil, nil) == 0, "read failed")

proc openclLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
  ensureInit()
  let n = rows * cols
  result = newSeq[float32](n)
  let bytesX = csize_t(n * sizeof(float32))
  let bytesC = csize_t(cols * sizeof(float32))
  let bufX = getBuf(bytesX, CL_MEM_READ_ONLY)
  let bufGamma = getBuf(bytesC, CL_MEM_READ_ONLY)
  let bufBeta = getBuf(bytesC, CL_MEM_READ_ONLY)
  let bufY = getBuf(bytesX, CL_MEM_WRITE_ONLY)
  defer:
    putBuf(bytesX, CL_MEM_READ_ONLY, bufX); putBuf(bytesC, CL_MEM_READ_ONLY, bufGamma)
    putBuf(bytesC, CL_MEM_READ_ONLY, bufBeta); putBuf(bytesX, CL_MEM_WRITE_ONLY, bufY)
  var xVar = x; var gammaVar = gamma; var betaVar = beta
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufX, CL_FALSE, 0, bytesX, addr xVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufGamma, CL_FALSE, 0, bytesC, addr gammaVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufBeta, CL_FALSE, 0, bytesC, addr betaVar[0], 0, nil, nil) == 0, "write failed")
  let kernel = getKernel("layernorm_kernel")
  var bX = bufX; var bGamma = bufGamma; var bBeta = bufBeta; var bY = bufY
  var cArg = int32(cols); var eArg = eps
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bX) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bGamma) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bBeta) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(cl_mem)), addr bY) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 4, csize_t(sizeof(int32)), addr cArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 5, csize_t(sizeof(float32)), addr eArg) == 0, "setArg failed")
  const BB_WG = csize_t(256)
  var globalSize = csize_t(rows) * BB_WG
  var localSize = BB_WG
  gpuCheck(lib.clEnqueueNDRangeKernel(gQueue, kernel, 1, nil, addr globalSize, addr localSize, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufY, CL_TRUE, 0, bytesX, addr result[0], 0, nil, nil) == 0, "read failed")

proc openclEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
  ensureInit()
  let num = indices.len
  result = newSeq[float32](num * dim)
  let bytesTable = csize_t(vocab * dim * sizeof(float32))
  let bytesIndices = csize_t(num * sizeof(int32))
  let bytesY = csize_t(num * dim * sizeof(float32))
  let bufTable = getBuf(bytesTable, CL_MEM_READ_ONLY)
  let bufIndices = getBuf(bytesIndices, CL_MEM_READ_ONLY)
  let bufY = getBuf(bytesY, CL_MEM_WRITE_ONLY)
  defer:
    putBuf(bytesTable, CL_MEM_READ_ONLY, bufTable); putBuf(bytesIndices, CL_MEM_READ_ONLY, bufIndices)
    putBuf(bytesY, CL_MEM_WRITE_ONLY, bufY)
  var tableVar = table; var indicesVar = indices
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufTable, CL_FALSE, 0, bytesTable, addr tableVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufIndices, CL_FALSE, 0, bytesIndices, addr indicesVar[0], 0, nil, nil) == 0, "write failed")
  let kernel = getKernel("embedding_lookup_kernel")
  var bTable = bufTable; var bIndices = bufIndices; var bY = bufY
  var vArg = int32(vocab); var dArg = int32(dim)
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bTable) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bIndices) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bY) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(int32)), addr vArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 4, csize_t(sizeof(int32)), addr dArg) == 0, "setArg failed")
  var globalSize = csize_t(num)
  gpuCheck(lib.clEnqueueNDRangeKernel(gQueue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufY, CL_TRUE, 0, bytesY, addr result[0], 0, nil, nil) == 0, "read failed")

proc openclApflu*(x: seq[float32], alpha, beta: float32): seq[float32] =
  ensureInit()
  let n = x.len
  result = newSeq[float32](n)
  let bytes = csize_t(n * sizeof(float32))
  let bufX = getBuf(bytes, CL_MEM_READ_ONLY)
  let bufY = getBuf(bytes, CL_MEM_WRITE_ONLY)
  defer:
    putBuf(bytes, CL_MEM_READ_ONLY, bufX); putBuf(bytes, CL_MEM_WRITE_ONLY, bufY)
  var xVar = x
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufX, CL_FALSE, 0, bytes, addr xVar[0], 0, nil, nil) == 0, "write failed")
  let kernel = getKernel("vecop_apflu")
  var bX = bufX; var bY = bufY
  var aArg = alpha; var bArg = beta
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bX) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bY) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(float32)), addr aArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(float32)), addr bArg) == 0, "setArg failed")
  var globalSize = csize_t(n)
  gpuCheck(lib.clEnqueueNDRangeKernel(gQueue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufY, CL_TRUE, 0, bytes, addr result[0], 0, nil, nil) == 0, "read failed")

proc openclApfluBackward*(x, dy: seq[float32], alpha, beta: float32): seq[float32] =
  ensureInit()
  let n = x.len
  result = newSeq[float32](n)
  let bytes = csize_t(n * sizeof(float32))
  let bufX = getBuf(bytes, CL_MEM_READ_ONLY)
  let bufDy = getBuf(bytes, CL_MEM_READ_ONLY)
  let bufDx = getBuf(bytes, CL_MEM_WRITE_ONLY)
  defer:
    putBuf(bytes, CL_MEM_READ_ONLY, bufX); putBuf(bytes, CL_MEM_READ_ONLY, bufDy)
    putBuf(bytes, CL_MEM_WRITE_ONLY, bufDx)
  var xVar = x; var dyVar = dy
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufX, CL_FALSE, 0, bytes, addr xVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufDy, CL_FALSE, 0, bytes, addr dyVar[0], 0, nil, nil) == 0, "write failed")
  let kernel = getKernel("vecop_apflu_backward")
  var bX = bufX; var bDy = bufDy; var bDx = bufDx
  var aArg = alpha; var bArg = beta
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bX) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bDy) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bDx) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(float32)), addr aArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 4, csize_t(sizeof(float32)), addr bArg) == 0, "setArg failed")
  var globalSize = csize_t(n)
  gpuCheck(lib.clEnqueueNDRangeKernel(gQueue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufDx, CL_TRUE, 0, bytes, addr result[0], 0, nil, nil) == 0, "read failed")

proc openclLayernormBackward*(dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  ensureInit()
  let n = rows * cols
  var dx = newSeq[float32](n)
  var dgamma = newSeq[float32](cols)
  var dbeta = newSeq[float32](cols)
  let bytesX = csize_t(n * sizeof(float32))
  let bytesC = csize_t(cols * sizeof(float32))
  let bufDy = getBuf(bytesX, CL_MEM_READ_ONLY)
  let bufX = getBuf(bytesX, CL_MEM_READ_ONLY)
  let bufGamma = getBuf(bytesC, CL_MEM_READ_ONLY)
  let bufBeta = getBuf(bytesC, CL_MEM_READ_ONLY)
  let bufDx = getBuf(bytesX, CL_MEM_WRITE_ONLY)
  let bufDgamma = getBuf(bytesC, CL_MEM_WRITE_ONLY)
  let bufDbeta = getBuf(bytesC, CL_MEM_WRITE_ONLY)
  defer:
    putBuf(bytesX, CL_MEM_READ_ONLY, bufDy); putBuf(bytesX, CL_MEM_READ_ONLY, bufX)
    putBuf(bytesC, CL_MEM_READ_ONLY, bufGamma); putBuf(bytesC, CL_MEM_READ_ONLY, bufBeta)
    putBuf(bytesX, CL_MEM_WRITE_ONLY, bufDx); putBuf(bytesC, CL_MEM_WRITE_ONLY, bufDgamma)
    putBuf(bytesC, CL_MEM_WRITE_ONLY, bufDbeta)
  var dyVar = dy; var xVar = x; var gammaVar = gamma; var betaVar = beta
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufDy, CL_FALSE, 0, bytesX, addr dyVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufX, CL_FALSE, 0, bytesX, addr xVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufGamma, CL_FALSE, 0, bytesC, addr gammaVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufBeta, CL_FALSE, 0, bytesC, addr betaVar[0], 0, nil, nil) == 0, "write failed")
  let kernel = getKernel("layernorm_backward_kernel")
  var bDy = bufDy; var bX = bufX; var bGamma = bufGamma; var bBeta = bufBeta
  var bDx = bufDx; var bDgamma = bufDgamma; var bDbeta = bufDbeta
  var rArg = int32(rows); var cArg = int32(cols); var eArg = eps
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bDy) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bX) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bGamma) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(cl_mem)), addr bBeta) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 4, csize_t(sizeof(cl_mem)), addr bDx) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 5, csize_t(sizeof(cl_mem)), addr bDgamma) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 6, csize_t(sizeof(cl_mem)), addr bDbeta) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 7, csize_t(sizeof(int32)), addr rArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 8, csize_t(sizeof(int32)), addr cArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 9, csize_t(sizeof(float32)), addr eArg) == 0, "setArg failed")
  var globalSize = csize_t(max(rows, cols))
  gpuCheck(lib.clEnqueueNDRangeKernel(gQueue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufDx, CL_TRUE, 0, bytesX, addr dx[0], 0, nil, nil) == 0, "read failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufDgamma, CL_TRUE, 0, bytesC, addr dgamma[0], 0, nil, nil) == 0, "read failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufDbeta, CL_TRUE, 0, bytesC, addr dbeta[0], 0, nil, nil) == 0, "read failed")
  return (dx, dgamma, dbeta)

# === THÊM MỚI: ATTENTION FORWARD ===
proc openclAttentionFused*(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  ensureInit()
  let qkvLen = B * H * S * D
  let sLen = B * H * S * S
  var o = newSeq[float32](qkvLen)
  var sMatrix = newSeq[float32](sLen)

  let bytesQKV = csize_t(qkvLen * sizeof(float32))
  let bytesS = csize_t(sLen * sizeof(float32))
  let bufQ = getBuf(bytesQKV, CL_MEM_READ_ONLY)
  let bufK = getBuf(bytesQKV, CL_MEM_READ_ONLY)
  let bufV = getBuf(bytesQKV, CL_MEM_READ_ONLY)
  let bufO = getBuf(bytesQKV, CL_MEM_WRITE_ONLY)
  let bufS = getBuf(bytesS, CL_MEM_WRITE_ONLY)
  defer:
    putBuf(bytesQKV, CL_MEM_READ_ONLY, bufQ); putBuf(bytesQKV, CL_MEM_READ_ONLY, bufK)
    putBuf(bytesQKV, CL_MEM_READ_ONLY, bufV); putBuf(bytesQKV, CL_MEM_WRITE_ONLY, bufO)
    putBuf(bytesS, CL_MEM_WRITE_ONLY, bufS)

  var qVar = q; var kVar = k; var vVar = v
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufQ, CL_FALSE, 0, bytesQKV, addr qVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufK, CL_FALSE, 0, bytesQKV, addr kVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufV, CL_FALSE, 0, bytesQKV, addr vVar[0], 0, nil, nil) == 0, "write failed")

  let kernel = getKernel("attention_fused_kernel")
  var bQ = bufQ; var bK = bufK; var bV = bufV; var bO = bufO; var bS = bufS
  var bArg = int32(B); var hArg = int32(H); var sArg = int32(S); var dArg = int32(D); var scArg = scale
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bQ) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bK) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bV) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(cl_mem)), addr bO) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 4, csize_t(sizeof(cl_mem)), addr bS) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 5, csize_t(sizeof(int32)), addr bArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 6, csize_t(sizeof(int32)), addr hArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 7, csize_t(sizeof(int32)), addr sArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 8, csize_t(sizeof(int32)), addr dArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 9, csize_t(sizeof(float32)), addr scArg) == 0, "setArg failed")

  gpuCheck(S <= 256, "openclAttentionFused: S=" & $S & " vuot gioi han 256")
  var globalSize2D = [csize_t(B * H), csize_t(S)]
  gpuCheck(lib.clEnqueueNDRangeKernel(gQueue, kernel, 2, nil, addr globalSize2D[0], nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufO, CL_TRUE, 0, bytesQKV, addr o[0], 0, nil, nil) == 0, "read failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufS, CL_TRUE, 0, bytesS, addr sMatrix[0], 0, nil, nil) == 0, "read failed")
  return (o, sMatrix)

# === THÊM MỚI: ATTENTION BACKWARD ===
proc openclAttentionFusedBackward*(q, k, v, sMatrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  ensureInit()
  let qkvLen = B * H * S * D
  let sLen = B * H * S * S
  var dqOut = newSeq[float32](qkvLen)
  var dkOut = newSeq[float32](qkvLen)
  var dvOut = newSeq[float32](qkvLen)

  let bytesQKV = csize_t(qkvLen * sizeof(float32))
  let bytesS = csize_t(sLen * sizeof(float32))
  let bufQ = getBuf(bytesQKV, CL_MEM_READ_ONLY)
  let bufK = getBuf(bytesQKV, CL_MEM_READ_ONLY)
  let bufV = getBuf(bytesQKV, CL_MEM_READ_ONLY)
  let bufS = getBuf(bytesS, CL_MEM_READ_ONLY)
  let bufDy = getBuf(bytesQKV, CL_MEM_READ_ONLY)
  let bufDq = getBuf(bytesQKV, CL_MEM_READ_WRITE)
  let bufDk = getBuf(bytesQKV, CL_MEM_READ_WRITE)
  let bufDv = getBuf(bytesQKV, CL_MEM_READ_WRITE)
  defer:
    putBuf(bytesQKV, CL_MEM_READ_ONLY, bufQ); putBuf(bytesQKV, CL_MEM_READ_ONLY, bufK)
    putBuf(bytesQKV, CL_MEM_READ_ONLY, bufV); putBuf(bytesS, CL_MEM_READ_ONLY, bufS)
    putBuf(bytesQKV, CL_MEM_READ_ONLY, bufDy)
    putBuf(bytesQKV, CL_MEM_READ_WRITE, bufDq); putBuf(bytesQKV, CL_MEM_READ_WRITE, bufDk)
    putBuf(bytesQKV, CL_MEM_READ_WRITE, bufDv)

  var qVar = q; var kVar = k; var vVar = v; var sVar = sMatrix; var dyVar = dy
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufQ, CL_FALSE, 0, bytesQKV, addr qVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufK, CL_FALSE, 0, bytesQKV, addr kVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufV, CL_FALSE, 0, bytesQKV, addr vVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufS, CL_FALSE, 0, bytesS, addr sVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufDy, CL_FALSE, 0, bytesQKV, addr dyVar[0], 0, nil, nil) == 0, "write failed")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufDq, CL_FALSE, 0, bytesQKV, addr dqOut[0], 0, nil, nil) == 0, "zero-init dq")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufDk, CL_FALSE, 0, bytesQKV, addr dkOut[0], 0, nil, nil) == 0, "zero-init dk")
  gpuCheck(lib.clEnqueueWriteBuffer(gQueue, bufDv, CL_FALSE, 0, bytesQKV, addr dvOut[0], 0, nil, nil) == 0, "zero-init dv")

  let kernel = getKernel("attention_fused_backward_kernel")
  var bQ = bufQ; var bK = bufK; var bV = bufV; var bS = bufS; var bDy = bufDy
  var bDq = bufDq; var bDk = bufDk; var bDv = bufDv
  var bArg = int32(B); var hArg = int32(H); var sArg = int32(S); var dArg = int32(D); var scArg = scale
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bQ) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bK) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bV) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(cl_mem)), addr bS) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 4, csize_t(sizeof(cl_mem)), addr bDy) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 5, csize_t(sizeof(cl_mem)), addr bDq) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 6, csize_t(sizeof(cl_mem)), addr bDk) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 7, csize_t(sizeof(cl_mem)), addr bDv) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 8, csize_t(sizeof(int32)), addr bArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 9, csize_t(sizeof(int32)), addr hArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 10, csize_t(sizeof(int32)), addr sArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 11, csize_t(sizeof(int32)), addr dArg) == 0, "setArg failed")
  gpuCheck(lib.clSetKernelArg(kernel, 12, csize_t(sizeof(float32)), addr scArg) == 0, "setArg failed")

  gpuCheck(S <= 256, "openclAttentionFusedBackward: S=" & $S & " vuot gioi han 256")
  var globalSize2D = [csize_t(B * H), csize_t(S)]
  gpuCheck(lib.clEnqueueNDRangeKernel(gQueue, kernel, 2, nil, addr globalSize2D[0], nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufDq, CL_TRUE, 0, bytesQKV, addr dqOut[0], 0, nil, nil) == 0, "read failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufDk, CL_TRUE, 0, bytesQKV, addr dkOut[0], 0, nil, nil) == 0, "read failed")
  gpuCheck(lib.clEnqueueReadBuffer(gQueue, bufDv, CL_TRUE, 0, bytesQKV, addr dvOut[0], 0, nil, nil) == 0, "read failed")
  return (dqOut, dkOut, dvOut)