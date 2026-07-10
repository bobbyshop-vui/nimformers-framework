# opencl_api.nim - Runtime bindings to OpenCL (libOpenCL / OpenCL.framework) qua dynlib.
# Biên dịch kernel OpenCL C từ chuỗi nguồn ngay lúc chạy (clBuildProgram), không cần
# công cụ build ngoài nào cả -> chạy được trên bất kỳ GPU/CPU nào có driver OpenCL.
import std/dynlib
import std/os

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
const CL_TRUE: uint32 = 1

proc gpuCheck(cond: bool, msg: string) =
  ## Giống doAssert nhưng raise CatchableError thay vì Defect, để gpubackend.nim
  ## có thể bắt lỗi và fallback CPU thay vì làm crash chương trình.
  if not cond:
    raise newException(CatchableError, msg)

var lib: OclLib
var loaded = false

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

proc initOpenCL*(platform_ret: ptr cl_platform_id, device_ret: ptr cl_device_id,
                 context_ret: ptr cl_context, queue_ret: ptr cl_command_queue): bool =
  if not tryLoad(): return false

  # 1. Get all platforms
  var num_platforms: cl_uint = 0
  if lib.clGetPlatformIDs(0, nil, addr num_platforms) != 0 or num_platforms == 0:
    return false
    
  var platforms = newSeq[cl_platform_id](num_platforms)
  if lib.clGetPlatformIDs(num_platforms, addr platforms[0], nil) != 0:
    return false
    
  # 2. Try to find a GPU device first across all platforms
  var found = false
  var platform: cl_platform_id
  var device: cl_device_id
  
  const CL_DEVICE_TYPE_GPU: uint64 = 4
  
  for plat in platforms:
    var num_devices: cl_uint = 0
    if lib.clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nil, addr num_devices) == 0 and num_devices > 0:
      var devices = newSeq[cl_device_id](num_devices)
      if lib.clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devices, addr devices[0], nil) == 0:
        platform = plat
        device = devices[0]
        found = true
        break
        
  # 3. If no GPU device found, try to find any device (CPU, etc.) across all platforms
  if not found:
    for plat in platforms:
      var num_devices: cl_uint = 0
      if lib.clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nil, addr num_devices) == 0 and num_devices > 0:
        var devices = newSeq[cl_device_id](num_devices)
        if lib.clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, num_devices, addr devices[0], nil) == 0:
          platform = plat
          device = devices[0]
          found = true
          break

  if not found:
    return false

  # 4. Create context with CL_CONTEXT_PLATFORM property
  var err: cl_int
  var props: array[3, int]
  props[0] = 0x1084  # CL_CONTEXT_PLATFORM
  props[1] = cast[int](platform)
  props[2] = 0
  
  let ctx = lib.clCreateContext(cast[pointer](addr props[0]), 1, addr device, nil, nil, addr err)
  if err != 0 or ctx == nil:
    return false
    
  let queue = lib.clCreateCommandQueue(ctx, device, 0, addr err)
  if err != 0 or queue == nil:
    if ctx != nil: discard lib.clReleaseContext(ctx)
    return false
    
  platform_ret[] = platform
  device_ret[] = device
  context_ret[] = ctx
  queue_ret[] = queue
  return true

proc openclAvailable*(): bool =
  ## Dò xem có driver OpenCL nào (GPU hoặc CPU) trên máy hay không.
  if not tryLoad(): return false
  var platform: cl_platform_id
  var device: cl_device_id
  var ctx: cl_context
  var queue: cl_command_queue
  if initOpenCL(addr platform, addr device, addr ctx, addr queue):
    discard lib.clReleaseCommandQueue(queue)
    discard lib.clReleaseContext(ctx)
    return true
  return false

proc openclVecOp*(op: string, a, b: seq[float32]): seq[float32] =
  ## Chạy phép toán elementwise trên GPU/CPU qua OpenCL. Raise nếu lỗi để fallback CPU thuần.
  if not tryLoad():
    raise newException(CatchableError, "libOpenCL not found")
  let n = a.len
  result = newSeq[float32](n)

  var platform: cl_platform_id
  var device: cl_device_id
  var ctx: cl_context
  var queue: cl_command_queue
  gpuCheck(initOpenCL(addr platform, addr device, addr ctx, addr queue), "Failed to initialize OpenCL")

  var err: cl_int
  let bytes = csize_t(n * sizeof(float32))
  let bufA = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufB = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufC = lib.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")

  var prog: cl_program = nil
  var kernel: cl_kernel = nil

  defer:
    if kernel != nil: discard lib.clReleaseKernel(kernel)
    if prog != nil: discard lib.clReleaseProgram(prog)
    discard lib.clReleaseMemObject(bufA)
    discard lib.clReleaseMemObject(bufB)
    discard lib.clReleaseMemObject(bufC)
    discard lib.clReleaseCommandQueue(queue)
    discard lib.clReleaseContext(ctx)

  var aVar = a
  var bVar = b
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bytes, addr aVar[0], 0, nil, nil) == 0, "GPU operation failed")
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytes, addr bVar[0], 0, nil, nil) == 0, "GPU operation failed")

  var srcPtr = kernelSource.cstring
  prog = lib.clCreateProgramWithSource(ctx, 1, addr srcPtr, nil, addr err)
  gpuCheck(err == 0 and prog != nil, "clCreateProgramWithSource failed")
  discard lib.clBuildProgram(prog, 1, addr device, nil, nil, nil)

  let kname = "vecop_" & op
  kernel = lib.clCreateKernel(prog, kname.cstring, addr err)
  gpuCheck(err == 0 and kernel != nil, "clCreateKernel failed for " & kname)

  var bA = bufA
  var bB = bufB
  var bC = bufC
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bA) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bB) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bC) == 0, "GPU operation failed")

  var globalSize = csize_t(n)
  gpuCheck(lib.clEnqueueNDRangeKernel(queue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  discard lib.clFinish(queue)
  gpuCheck(lib.clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, addr result[0], 0, nil, nil) == 0, "GPU operation failed")

proc openclMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
  ## C(m x n) = A(m x k) * B(k x n), row-major. Kernel naive (không tối ưu
  ## tile/shared-memory) chạy trên NDRange 2 chiều [m, n] -> đúng trên mọi
  ## GPU/CPU có driver OpenCL, không phụ thuộc đời phần cứng cụ thể.
  if not tryLoad():
    raise newException(CatchableError, "libOpenCL not found")
  result = newSeq[float32](m * n)

  var platform: cl_platform_id
  var device: cl_device_id
  var ctx: cl_context
  var queue: cl_command_queue
  gpuCheck(initOpenCL(addr platform, addr device, addr ctx, addr queue), "Failed to initialize OpenCL")

  var err: cl_int
  let bytesA = csize_t(m * k * sizeof(float32))
  let bytesB = csize_t(k * n * sizeof(float32))
  let bytesC = csize_t(m * n * sizeof(float32))
  let bufA = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytesA, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufB = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytesB, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufC = lib.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytesC, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")

  var prog: cl_program = nil
  var kernel: cl_kernel = nil

  defer:
    if kernel != nil: discard lib.clReleaseKernel(kernel)
    if prog != nil: discard lib.clReleaseProgram(prog)
    discard lib.clReleaseMemObject(bufA)
    discard lib.clReleaseMemObject(bufB)
    discard lib.clReleaseMemObject(bufC)
    discard lib.clReleaseCommandQueue(queue)
    discard lib.clReleaseContext(ctx)

  var aVar = a
  var bVar = b
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bytesA, addr aVar[0], 0, nil, nil) == 0, "GPU operation failed")
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytesB, addr bVar[0], 0, nil, nil) == 0, "GPU operation failed")

  var srcPtr = kernelSource.cstring
  prog = lib.clCreateProgramWithSource(ctx, 1, addr srcPtr, nil, addr err)
  gpuCheck(err == 0 and prog != nil, "clCreateProgramWithSource failed")
  discard lib.clBuildProgram(prog, 1, addr device, nil, nil, nil)

  kernel = lib.clCreateKernel(prog, "matmul_naive".cstring, addr err)
  gpuCheck(err == 0 and kernel != nil, "clCreateKernel failed for matmul_naive")

  var bA = bufA
  var bB = bufB
  var bC = bufC
  var mArg = int32(m)
  var kArg = int32(k)
  var nArg = int32(n)
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bA) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bB) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bC) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(int32)), addr mArg) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 4, csize_t(sizeof(int32)), addr kArg) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 5, csize_t(sizeof(int32)), addr nArg) == 0, "GPU operation failed")

  var globalSize2D = [csize_t(m), csize_t(n)]
  gpuCheck(lib.clEnqueueNDRangeKernel(queue, kernel, 2, nil, addr globalSize2D[0], nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  discard lib.clFinish(queue)
  gpuCheck(lib.clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytesC, addr result[0], 0, nil, nil) == 0, "GPU operation failed")

proc openclActivation*(op: string, x: seq[float32]): seq[float32] =
  if not tryLoad():
    raise newException(CatchableError, "libOpenCL not found")
  let n = x.len
  result = newSeq[float32](n)

  var platform: cl_platform_id
  var device: cl_device_id
  var ctx: cl_context
  var queue: cl_command_queue
  gpuCheck(initOpenCL(addr platform, addr device, addr ctx, addr queue), "Failed to initialize OpenCL")

  var err: cl_int
  let bytes = csize_t(n * sizeof(float32))
  let bufX = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufY = lib.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")

  var prog: cl_program = nil
  var kernel: cl_kernel = nil

  defer:
    if kernel != nil: discard lib.clReleaseKernel(kernel)
    if prog != nil: discard lib.clReleaseProgram(prog)
    discard lib.clReleaseMemObject(bufX)
    discard lib.clReleaseMemObject(bufY)
    discard lib.clReleaseCommandQueue(queue)
    discard lib.clReleaseContext(ctx)

  var xVar = x
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, bytes, addr xVar[0], 0, nil, nil) == 0, "GPU operation failed")

  var srcPtr = kernelSource.cstring
  prog = lib.clCreateProgramWithSource(ctx, 1, addr srcPtr, nil, addr err)
  gpuCheck(err == 0 and prog != nil, "clCreateProgramWithSource failed")
  discard lib.clBuildProgram(prog, 1, addr device, nil, nil, nil)

  let kname = "vecop_" & op
  kernel = lib.clCreateKernel(prog, kname.cstring, addr err)
  gpuCheck(err == 0 and kernel != nil, "clCreateKernel failed for " & kname)

  var bX = bufX
  var bY = bufY
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bX) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bY) == 0, "GPU operation failed")

  var globalSize = csize_t(n)
  gpuCheck(lib.clEnqueueNDRangeKernel(queue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  discard lib.clFinish(queue)
  gpuCheck(lib.clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, bytes, addr result[0], 0, nil, nil) == 0, "GPU operation failed")

proc openclSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
  if not tryLoad():
    raise newException(CatchableError, "libOpenCL not found")
  let n = rows * cols
  result = newSeq[float32](n)

  var platform: cl_platform_id
  var device: cl_device_id
  var ctx: cl_context
  var queue: cl_command_queue
  gpuCheck(initOpenCL(addr platform, addr device, addr ctx, addr queue), "Failed to initialize OpenCL")

  var err: cl_int
  let bytes = csize_t(n * sizeof(float32))
  let bufX = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufY = lib.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")

  var prog: cl_program = nil
  var kernel: cl_kernel = nil

  defer:
    if kernel != nil: discard lib.clReleaseKernel(kernel)
    if prog != nil: discard lib.clReleaseProgram(prog)
    discard lib.clReleaseMemObject(bufX)
    discard lib.clReleaseMemObject(bufY)
    discard lib.clReleaseCommandQueue(queue)
    discard lib.clReleaseContext(ctx)

  var xVar = x
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, bytes, addr xVar[0], 0, nil, nil) == 0, "GPU operation failed")

  var srcPtr = kernelSource.cstring
  prog = lib.clCreateProgramWithSource(ctx, 1, addr srcPtr, nil, addr err)
  gpuCheck(err == 0 and prog != nil, "clCreateProgramWithSource failed")
  discard lib.clBuildProgram(prog, 1, addr device, nil, nil, nil)

  kernel = lib.clCreateKernel(prog, "softmax_kernel".cstring, addr err)
  gpuCheck(err == 0 and kernel != nil, "clCreateKernel failed for softmax_kernel")

  var bX = bufX
  var bY = bufY
  var cArg = int32(cols)
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bX) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bY) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(int32)), addr cArg) == 0, "GPU operation failed")

  var globalSize = csize_t(rows)
  gpuCheck(lib.clEnqueueNDRangeKernel(queue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  discard lib.clFinish(queue)
  gpuCheck(lib.clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, bytes, addr result[0], 0, nil, nil) == 0, "GPU operation failed")

proc openclLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
  if not tryLoad():
    raise newException(CatchableError, "libOpenCL not found")
  let n = rows * cols
  result = newSeq[float32](n)

  var platform: cl_platform_id
  var device: cl_device_id
  var ctx: cl_context
  var queue: cl_command_queue
  gpuCheck(initOpenCL(addr platform, addr device, addr ctx, addr queue), "Failed to initialize OpenCL")

  var err: cl_int
  let bytesX = csize_t(n * sizeof(float32))
  let bytesC = csize_t(cols * sizeof(float32))
  let bufX = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytesX, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufGamma = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytesC, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufBeta = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytesC, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufY = lib.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytesX, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")

  var prog: cl_program = nil
  var kernel: cl_kernel = nil

  defer:
    if kernel != nil: discard lib.clReleaseKernel(kernel)
    if prog != nil: discard lib.clReleaseProgram(prog)
    discard lib.clReleaseMemObject(bufX)
    discard lib.clReleaseMemObject(bufGamma)
    discard lib.clReleaseMemObject(bufBeta)
    discard lib.clReleaseMemObject(bufY)
    discard lib.clReleaseCommandQueue(queue)
    discard lib.clReleaseContext(ctx)

  var xVar = x
  var gammaVar = gamma
  var betaVar = beta
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, bytesX, addr xVar[0], 0, nil, nil) == 0, "GPU operation failed")
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufGamma, CL_TRUE, 0, bytesC, addr gammaVar[0], 0, nil, nil) == 0, "GPU operation failed")
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufBeta, CL_TRUE, 0, bytesC, addr betaVar[0], 0, nil, nil) == 0, "GPU operation failed")

  var srcPtr = kernelSource.cstring
  prog = lib.clCreateProgramWithSource(ctx, 1, addr srcPtr, nil, addr err)
  gpuCheck(err == 0 and prog != nil, "clCreateProgramWithSource failed")
  discard lib.clBuildProgram(prog, 1, addr device, nil, nil, nil)

  kernel = lib.clCreateKernel(prog, "layernorm_kernel".cstring, addr err)
  gpuCheck(err == 0 and kernel != nil, "clCreateKernel failed for layernorm_kernel")

  var bX = bufX
  var bGamma = bufGamma
  var bBeta = bufBeta
  var bY = bufY
  var cArg = int32(cols)
  var eArg = eps
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bX) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bGamma) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bBeta) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(cl_mem)), addr bY) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 4, csize_t(sizeof(int32)), addr cArg) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 5, csize_t(sizeof(float32)), addr eArg) == 0, "GPU operation failed")

  var globalSize = csize_t(rows)
  gpuCheck(lib.clEnqueueNDRangeKernel(queue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  discard lib.clFinish(queue)
  gpuCheck(lib.clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, bytesX, addr result[0], 0, nil, nil) == 0, "GPU operation failed")

proc openclEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
  if not tryLoad():
    raise newException(CatchableError, "libOpenCL not found")
  let num = indices.len
  result = newSeq[float32](num * dim)

  var platform: cl_platform_id
  var device: cl_device_id
  var ctx: cl_context
  var queue: cl_command_queue
  gpuCheck(initOpenCL(addr platform, addr device, addr ctx, addr queue), "Failed to initialize OpenCL")

  var err: cl_int
  let bytesTable = csize_t(vocab * dim * sizeof(float32))
  let bytesIndices = csize_t(num * sizeof(int32))
  let bytesY = csize_t(num * dim * sizeof(float32))
  let bufTable = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytesTable, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufIndices = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytesIndices, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufY = lib.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytesY, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")

  var prog: cl_program = nil
  var kernel: cl_kernel = nil

  defer:
    if kernel != nil: discard lib.clReleaseKernel(kernel)
    if prog != nil: discard lib.clReleaseProgram(prog)
    discard lib.clReleaseMemObject(bufTable)
    discard lib.clReleaseMemObject(bufIndices)
    discard lib.clReleaseMemObject(bufY)
    discard lib.clReleaseCommandQueue(queue)
    discard lib.clReleaseContext(ctx)

  var tableVar = table
  var indicesVar = indices
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufTable, CL_TRUE, 0, bytesTable, addr tableVar[0], 0, nil, nil) == 0, "GPU operation failed")
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufIndices, CL_TRUE, 0, bytesIndices, addr indicesVar[0], 0, nil, nil) == 0, "GPU operation failed")

  var srcPtr = kernelSource.cstring
  prog = lib.clCreateProgramWithSource(ctx, 1, addr srcPtr, nil, addr err)
  gpuCheck(err == 0 and prog != nil, "clCreateProgramWithSource failed")
  discard lib.clBuildProgram(prog, 1, addr device, nil, nil, nil)

  kernel = lib.clCreateKernel(prog, "embedding_lookup_kernel".cstring, addr err)
  gpuCheck(err == 0 and kernel != nil, "clCreateKernel failed for embedding_lookup_kernel")

  var bTable = bufTable
  var bIndices = bufIndices
  var bY = bufY
  var vArg = int32(vocab)
  var dArg = int32(dim)
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bTable) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bIndices) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bY) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(int32)), addr vArg) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 4, csize_t(sizeof(int32)), addr dArg) == 0, "GPU operation failed")

  var globalSize = csize_t(num)
  gpuCheck(lib.clEnqueueNDRangeKernel(queue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  discard lib.clFinish(queue)
  gpuCheck(lib.clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, bytesY, addr result[0], 0, nil, nil) == 0, "GPU operation failed")
