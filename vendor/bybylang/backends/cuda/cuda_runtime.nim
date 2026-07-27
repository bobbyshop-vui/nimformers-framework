# cuda_runtime.nim - Runtime bindings tới CUDA Runtime API (libcudart) + cuBLAS
# qua dynlib. cuda_driver.nim (Driver API + PTX tay) dùng cho vecop elementwise;
# module này dùng riêng cho matmul để tận dụng Tensor Core qua cuBLAS thay vì
# viết tay lệnh PTX `wmma.mma.sync` (rất kén kiến trúc GPU và dễ sai).
#
# cuBLAS tự chọn kernel Tensor Core khi bật math mode CUBLAS_TENSOR_OP_MATH:
# trên GPU có Tensor Core (Volta trở lên) cublasSgemm sẽ hạ xuống kernel dùng
# TF32/FP16 accumulate; trên GPU không có Tensor Core, cuBLAS tự fallback về
# kernel FP32 CUDA core bình thường -> không cần code riêng cho từng đời GPU.
#
# Không phụ thuộc lúc build (dynlib): máy không có libcudart.so/libcublas.so
# thì cudaRuntimeAvailable() trả về false, gpubackend.nim tự fallback CPU.
import std/dynlib
import std/tables
import cuda_driver

# Export CudaTensor từ cuda_driver để các module khác dùng được
export cuda_driver.CudaTensor

type
  cudaError_t = int32
  cublasHandle_t = pointer
  cublasStatus_t = int32
  cublasOperation_t = int32
  cublasMath_t = int32

const
  cudaMemcpyHostToDevice: int32 = 1
  cudaMemcpyDeviceToHost: int32 = 2
  CUBLAS_OP_N: cublasOperation_t = 0
  CUBLAS_OP_T: cublasOperation_t = 1
  CUBLAS_DEFAULT_MATH: cublasMath_t = 0
  CUBLAS_TENSOR_OP_MATH: cublasMath_t = 1

type
  CudartLib = object
    handle: LibHandle
    cudaMalloc: proc(devPtr: ptr pointer, size: csize_t): cudaError_t {.cdecl.}
    cudaFree: proc(devPtr: pointer): cudaError_t {.cdecl.}
    cudaMemcpy: proc(dst, src: pointer, count: csize_t, kind: int32): cudaError_t {.cdecl.}
    cudaGetDeviceCount: proc(count: ptr int32): cudaError_t {.cdecl.}
    cudaSetDevice: proc(device: int32): cudaError_t {.cdecl.}
    cudaDeviceSynchronize: proc(): cudaError_t {.cdecl.}

  CublasLib = object
    handle: LibHandle
    cublasCreate_v2: proc(handle: ptr cublasHandle_t): cublasStatus_t {.cdecl.}
    cublasDestroy_v2: proc(handle: cublasHandle_t): cublasStatus_t {.cdecl.}
    cublasSetMathMode: proc(handle: cublasHandle_t, mode: cublasMath_t): cublasStatus_t {.cdecl.}
    cublasSetStream_v2: proc(handle: cublasHandle_t, streamId: pointer): cublasStatus_t {.cdecl.}
    cublasSgemm_v2: proc(handle: cublasHandle_t, transa, transb: cublasOperation_t,
                          m, n, k: int32, alpha: ptr float32,
                          A: pointer, lda: int32, B: pointer, ldb: int32,
                          beta: ptr float32, C: pointer, ldc: int32): cublasStatus_t {.cdecl.}

var cudart: CudartLib
var cublas: CublasLib
var cudartLoaded = false
var cublasLoaded = false

var gHandle: cublasHandle_t
var gHandleInitialized = false
var gDevicePool = initTable[csize_t, seq[pointer]]()

proc gpuCheck(cond: bool, msg: string) =
  if not cond:
    raise newException(CatchableError, msg)

proc ensureHandle(): cublasHandle_t =
  if gHandleInitialized: return gHandle
  gpuCheck(cublas.cublasCreate_v2(addr gHandle) == 0, "cublasCreate failed")
  discard cublas.cublasSetMathMode(gHandle, CUBLAS_TENSOR_OP_MATH)
  if cublas.cublasSetStream_v2 != nil:
    discard cublas.cublasSetStream_v2(gHandle, getStream())
  gHandleInitialized = true
  result = gHandle

proc getDeviceBuf(bytes: csize_t): pointer =
  if gDevicePool.hasKey(bytes) and gDevicePool[bytes].len > 0:
    return gDevicePool[bytes].pop()
  var p: pointer
  gpuCheck(cudart.cudaMalloc(addr p, bytes) == 0, "cudaMalloc failed")
  result = p

proc putDeviceBuf(bytes: csize_t, p: pointer) =
  gDevicePool.mgetOrPut(bytes, @[]).add(p)

proc tryLoadCudart(): bool =
  if cudartLoaded: return cudart.handle != nil
  cudartLoaded = true
  for name in ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0", "cudart64_120.dll", "cudart64_110.dll"]:
    cudart.handle = loadLib(name)
    if cudart.handle != nil: break
  if cudart.handle == nil: return false
  cudart.cudaMalloc = cast[typeof(cudart.cudaMalloc)](cudart.handle.symAddr("cudaMalloc"))
  cudart.cudaFree = cast[typeof(cudart.cudaFree)](cudart.handle.symAddr("cudaFree"))
  cudart.cudaMemcpy = cast[typeof(cudart.cudaMemcpy)](cudart.handle.symAddr("cudaMemcpy"))
  cudart.cudaGetDeviceCount = cast[typeof(cudart.cudaGetDeviceCount)](cudart.handle.symAddr("cudaGetDeviceCount"))
  cudart.cudaSetDevice = cast[typeof(cudart.cudaSetDevice)](cudart.handle.symAddr("cudaSetDevice"))
  cudart.cudaDeviceSynchronize = cast[typeof(cudart.cudaDeviceSynchronize)](cudart.handle.symAddr("cudaDeviceSynchronize"))
  result = cudart.cudaMalloc != nil and cudart.cudaMemcpy != nil

proc tryLoadCublas(): bool =
  if cublasLoaded: return cublas.handle != nil
  cublasLoaded = true
  for name in ["libcublas.so", "libcublas.so.12", "libcublas.so.11", "cublas64_12.dll", "cublas64_11.dll"]:
    cublas.handle = loadLib(name)
    if cublas.handle != nil: break
  if cublas.handle == nil: return false
  cublas.cublasCreate_v2 = cast[typeof(cublas.cublasCreate_v2)](cublas.handle.symAddr("cublasCreate_v2"))
  cublas.cublasDestroy_v2 = cast[typeof(cublas.cublasDestroy_v2)](cublas.handle.symAddr("cublasDestroy_v2"))
  cublas.cublasSetMathMode = cast[typeof(cublas.cublasSetMathMode)](cublas.handle.symAddr("cublasSetMathMode"))
  cublas.cublasSetStream_v2 = cast[typeof(cublas.cublasSetStream_v2)](cublas.handle.symAddr("cublasSetStream_v2"))
  cublas.cublasSgemm_v2 = cast[typeof(cublas.cublasSgemm_v2)](cublas.handle.symAddr("cublasSgemm_v2"))
  result = cublas.cublasCreate_v2 != nil and cublas.cublasSgemm_v2 != nil

proc cudaRuntimeAvailable*(): bool =
  if not tryLoadCudart(): return false
  var count: int32 = 0
  if cudart.cudaGetDeviceCount(addr count) != 0: return false
  result = count > 0

proc cudaMatmulF32*(a, b: seq[float32], m, k, n: int): seq[float32] =
  gpuCheck(tryLoadCudart(), "libcudart not found")
  gpuCheck(tryLoadCublas(), "libcublas not found")
  result = newSeq[float32](m * n)

  let handle = ensureHandle()

  let bytesA = csize_t(m * k * sizeof(float32))
  let bytesB = csize_t(k * n * sizeof(float32))
  let bytesC = csize_t(m * n * sizeof(float32))
  let dA = getDeviceBuf(bytesA)
  let dB = getDeviceBuf(bytesB)
  let dC = getDeviceBuf(bytesC)
  defer:
    putDeviceBuf(bytesA, dA)
    putDeviceBuf(bytesB, dB)
    putDeviceBuf(bytesC, dC)

  var aVar = a
  var bVar = b
  gpuCheck(cudart.cudaMemcpy(dA, addr aVar[0], bytesA, cudaMemcpyHostToDevice) == 0, "H2D A failed")
  gpuCheck(cudart.cudaMemcpy(dB, addr bVar[0], bytesB, cudaMemcpyHostToDevice) == 0, "H2D B failed")

  var alpha: float32 = 1.0'f32
  var beta: float32 = 0.0'f32
  gpuCheck(cublas.cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  int32(n), int32(m), int32(k),
                                  addr alpha, dB, int32(n), dA, int32(k),
                                  addr beta, dC, int32(n)) == 0, "cublasSgemm failed")
  discard cudart.cudaDeviceSynchronize()
  gpuCheck(cudart.cudaMemcpy(addr result[0], dC, bytesC, cudaMemcpyDeviceToHost) == 0, "D2H C failed")

proc debugLog(msg: string) =
  when defined(nimformerCudaDebug):
    stderr.writeLine("[cuda_runtime][DEBUG] " & msg)

proc cudaMatmulF32R*(a, b: CudaTensor, m, k, n: int): CudaTensor =
  gpuCheck(tryLoadCudart(), "cudaMatmulF32R: libcudart not found")
  gpuCheck(tryLoadCublas(), "cudaMatmulF32R: libcublas not found")
  gpuCheck(a.dptr != 0 and b.dptr != 0, "cudaMatmulF32R: input tensor rỗng/đã free")
  gpuCheck(a.numel == m * k, "cudaMatmulF32R: a.numel=" & $a.numel & " != m*k=" & $(m*k))
  gpuCheck(b.numel == k * n, "cudaMatmulF32R: b.numel=" & $b.numel & " != k*n=" & $(k*n))

  let handle = ensureHandle()
  let bytesC = csize_t(m * n * sizeof(float32))
  let dC = cuda_driver.getDeviceBuf(bytesC)

  let pA = cast[pointer](a.dptr)
  let pB = cast[pointer](b.dptr)
  let pC = cast[pointer](dC)

  var alpha: float32 = 1.0'f32
  var beta: float32 = 0.0'f32
  gpuCheck(cublas.cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  int32(n), int32(m), int32(k),
                                  addr alpha, pB, int32(n), pA, int32(k),
                                  addr beta, pC, int32(n)) == 0, "cudaMatmulF32R: cublasSgemm failed")
  debugLog("cudaMatmulF32R m=" & $m & " k=" & $k & " n=" & $n)
  result = CudaTensor(dptr: cast[cuda_driver.CUdeviceptr](dC), bytes: bytesC, numel: m * n)

proc cudaMatmulNTF32R*(a, b: CudaTensor, m, k, n: int): CudaTensor =
  ## C(row-major m x n) = A(row-major m x k) @ B(row-major n x k)^T.
  ## B GIU NGUYEN layout goc [n,k] (vd K-cache/V-cache [S,D] tu nhien) -
  ## KHONG can transpose CPU truoc khi upload, chi can 1 co CUBLAS_OP_T.
  ##
  ## Suy dan (cublasSgemm luon coi ma tran la COLUMN-MAJOR):
  ## - A row-major (m x k) doc lai theo column-major chinh la (k x m) = A^T.
  ## - B row-major (n x k) doc lai theo column-major chinh la (k x n) = B^T.
  ##   Ta muon dung B (khong phai B^T) trong tich A@B^T -> ap them 1 lan
  ##   CUBLAS_OP_T len B^T (da co san tu viec doc lai bit) se cho lai dung B.
  ## - Ket qua: cublasSgemm(opT tren B [ld=k], opN tren A [ld=k]) tinh ra
  ##   C^T (column-major n x m) - chinh la C (row-major m x n) can, khong
  ##   phai xu ly gi them.
  gpuCheck(tryLoadCudart(), "cudaMatmulNTF32R: libcudart not found")
  gpuCheck(tryLoadCublas(), "cudaMatmulNTF32R: libcublas not found")
  gpuCheck(a.dptr != 0 and b.dptr != 0, "cudaMatmulNTF32R: input tensor rỗng/đã free")
  gpuCheck(a.numel == m * k, "cudaMatmulNTF32R: a.numel=" & $a.numel & " != m*k=" & $(m*k))
  gpuCheck(b.numel == n * k, "cudaMatmulNTF32R: b.numel=" & $b.numel & " != n*k=" & $(n*k))

  let handle = ensureHandle()
  let bytesC = csize_t(m * n * sizeof(float32))
  let dC = cuda_driver.getDeviceBuf(bytesC)

  let pA = cast[pointer](a.dptr)
  let pB = cast[pointer](b.dptr)
  let pC = cast[pointer](dC)

  var alpha: float32 = 1.0'f32
  var beta: float32 = 0.0'f32
  gpuCheck(cublas.cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  int32(n), int32(m), int32(k),
                                  addr alpha, pB, int32(k), pA, int32(k),
                                  addr beta, pC, int32(n)) == 0, "cudaMatmulNTF32R: cublasSgemm failed")
  debugLog("cudaMatmulNTF32R m=" & $m & " k=" & $k & " n=" & $n)
  result = CudaTensor(dptr: cast[cuda_driver.CUdeviceptr](dC), bytes: bytesC, numel: m * n)