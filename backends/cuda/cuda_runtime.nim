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
  CUBLAS_DEFAULT_MATH: cublasMath_t = 0
  CUBLAS_TENSOR_OP_MATH: cublasMath_t = 1   # bật Tensor Core cho cuBLAS gemm

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
    cublasSgemm_v2: proc(handle: cublasHandle_t, transa, transb: cublasOperation_t,
                          m, n, k: int32, alpha: ptr float32,
                          A: pointer, lda: int32, B: pointer, ldb: int32,
                          beta: ptr float32, C: pointer, ldc: int32): cublasStatus_t {.cdecl.}

var cudart: CudartLib
var cublas: CublasLib
var cudartLoaded = false
var cublasLoaded = false

proc gpuCheck(cond: bool, msg: string) =
  if not cond:
    raise newException(CatchableError, msg)

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
  cublas.cublasSgemm_v2 = cast[typeof(cublas.cublasSgemm_v2)](cublas.handle.symAddr("cublasSgemm_v2"))
  result = cublas.cublasCreate_v2 != nil and cublas.cublasSgemm_v2 != nil

proc cudaRuntimeAvailable*(): bool =
  ## Dò libcudart + ít nhất 1 GPU. Dùng riêng cho đường matmul (cudaAvailable()
  ## trong cuda_driver.nim vẫn là cổng chính cho vecop qua Driver API).
  if not tryLoadCudart(): return false
  var count: int32 = 0
  if cudart.cudaGetDeviceCount(addr count) != 0: return false
  result = count > 0

proc cudaMatmulF32*(a, b: seq[float32], m, k, n: int): seq[float32] =
  ## C(m x n) = A(m x k) * B(k x n), toàn bộ row-major (giống layout Nim seq bình thường).
  ## Chạy qua cudart (cudaMalloc/cudaMemcpy) + cublasSgemm với Tensor Core math
  ## mode bật sẵn. cuBLAS là column-major nên dùng thủ thuật chuẩn: tính
  ## C^T = B^T * A^T bằng cách hoán đổi thứ tự & kích thước tham số -> kết quả
  ## ra đúng C row-major mà không cần transpose thủ công trên host.
  gpuCheck(tryLoadCudart(), "libcudart not found")
  gpuCheck(tryLoadCublas(), "libcublas not found")
  result = newSeq[float32](m * n)

  var handle: cublasHandle_t
  gpuCheck(cublas.cublasCreate_v2(addr handle) == 0, "cublasCreate failed")
  defer: discard cublas.cublasDestroy_v2(handle)
  discard cublas.cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)

  let bytesA = csize_t(m * k * sizeof(float32))
  let bytesB = csize_t(k * n * sizeof(float32))
  let bytesC = csize_t(m * n * sizeof(float32))
  var dA, dB, dC: pointer
  gpuCheck(cudart.cudaMalloc(addr dA, bytesA) == 0, "cudaMalloc A failed")
  gpuCheck(cudart.cudaMalloc(addr dB, bytesB) == 0, "cudaMalloc B failed")
  gpuCheck(cudart.cudaMalloc(addr dC, bytesC) == 0, "cudaMalloc C failed")
  defer:
    discard cudart.cudaFree(dA)
    discard cudart.cudaFree(dB)
    discard cudart.cudaFree(dC)

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
