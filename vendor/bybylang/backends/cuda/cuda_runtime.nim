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

# CudaTensor SỐNG Ở cuda_driver.nim (không phải ở đây). Trước đây bị định
# nghĩa nhầm ở module này trong khi gpubackend.nim luôn gọi `cudaDrv.CudaTensor`
# (tức mong đợi nó nằm trong cuda_driver.nim) -> lỗi "undeclared identifier".
# Re-export để code cũ import cuda_runtime rồi dùng CudaTensor vẫn chạy được.
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
    cublasSetStream_v2: proc(handle: cublasHandle_t, streamId: pointer): cublasStatus_t {.cdecl.}
    cublasSgemm_v2: proc(handle: cublasHandle_t, transa, transb: cublasOperation_t,
                          m, n, k: int32, alpha: ptr float32,
                          A: pointer, lda: int32, B: pointer, ldb: int32,
                          beta: ptr float32, C: pointer, ldc: int32): cublasStatus_t {.cdecl.}

var cudart: CudartLib
var cublas: CublasLib
var cudartLoaded = false
var cublasLoaded = false

# ─────────────────────────────────────────────────────────────
# ĐÃ SỬA (tối ưu hiệu năng): TRƯỚC ĐÂY cudaMatmulF32 tạo mới cublasHandle_t
# (cublasCreate_v2/cublasDestroy_v2) VÀ cudaMalloc/cudaFree cho A/B/C ở MỖI
# LẦN GỌI. cublasCreate không rẻ (khởi tạo context cuBLAS nội bộ), và
# cudaMalloc/cudaFree đều là thao tác đồng bộ với driver, tốn tương đương một
# lần round-trip lên GPU. Vì matmul là phép toán được gọi nhiều nhất và tốn
# nhất trong 1 bước train transformer (attention + FFN), đây là nguyên nhân
# chính khiến CUDA "chậm chạm".
#
# Bây giờ: 1 cublasHandle_t DUY NHẤT được tạo 1 lần cho toàn bộ process (giữ
# nguyên Tensor Core math mode đã bật), và device buffer được lấy từ pool
# theo kích thước (byte) thay vì cudaMalloc/cudaFree mỗi lần, y hệt cách
# cuda_driver.nim đã làm cho các phép elementwise.
# ─────────────────────────────────────────────────────────────
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
  ## Lấy device buffer từ pool nếu có sẵn cùng kích thước, không thì cudaMalloc.
  if gDevicePool.hasKey(bytes) and gDevicePool[bytes].len > 0:
    return gDevicePool[bytes].pop()
  var p: pointer
  gpuCheck(cudart.cudaMalloc(addr p, bytes) == 0, "cudaMalloc failed")
  result = p

proc putDeviceBuf(bytes: csize_t, p: pointer) =
  ## Trả buffer về pool để tái sử dụng thay vì cudaFree ngay.
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
  ## Bản resident của cudaMatmulF32: A, B đã sống sẵn trên GPU (upload bởi
  ## cuda_driver.uploadAsync), kết quả C cũng ở lại GPU (không D2H) để chuỗi
  ## resident op tiếp theo (add/relu/softmax/...) dùng thẳng.
  ##
  ## LƯU Ý AN TOÀN: a.dptr/b.dptr được cấp phát bởi cuda_driver.nim qua Driver
  ## API (cuMemAlloc, context tạo bằng cuCtxCreate), còn cublasSgemm_v2 ở đây
  ## gọi qua Runtime API (cudart/cublas). Đây là pattern trộn Driver+Runtime
  ## API đã có sẵn trong chính file này (ensureHandle() gọi
  ## cublasSetStream_v2(handle, getStream()) dùng stream tạo bởi cuda_driver)
  ## - cublas dùng device pointer thuần (không phân biệt driver-alloc hay
  ## runtime-alloc) miễn là cùng context hiện hành, nên cast CUdeviceptr sang
  ## `pointer` ở đây an toàn. C cũng cấp phát qua pool của cuda_driver
  ## (getDeviceBuf/putDeviceBuf export sẵn) để mọi CudaTensor resident dùng
  ## chung 1 pool duy nhất, tránh có 2 hệ thống pool tách rời cho cùng 1 loại
  ## buffer (dễ rò rỉ VRAM nếu lẫn lộn).
  ##
  ## CHƯA test trên GPU thật (không có GPU NVIDIA trong môi trường sinh code
  ## này). Trước khi tin dùng cho train: so khớp số học với cudaMatmulF32
  ## không-resident (hoặc cpuMatmul) trên input ngẫu nhiên nhỏ.
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
  # ĐÃ SỬA (tối ưu hiệu năng): trước đây có cudaDeviceSynchronize() ở đây,
  # tức là CHẶN TOÀN BỘ DEVICE (mọi stream, mọi kernel khác đang chạy) sau
  # MỖI lần gọi matmul resident. Với transformer thì matmul là op resident
  # gọi nhiều nhất (mỗi lớp attention + FFN), nên trước đây mỗi matmul đều
  # ép GPU rỗng hàng đợi rồi mới cho làm tiếp -> cùng một lỗi round-trip
  # đồng bộ mà phần buffer pool/async copy ở cuda_driver.nim đã cố tránh,
  # chỉ khác là lỗi này nằm ở module cublas riêng nên trước không thấy.
  # cublasSetStream_v2(handle, getStream()) đã được set 1 lần trong
  # ensureHandle(), nên cublasSgemm ở đây ĐÃ chạy trên đúng gStream và các
  # resident-op tiếp theo (add/relu/softmax/layernorm resident) tự động
  # được stream-order đúng thứ tự sau nó mà không cần đồng bộ gì thêm.
  # Điểm sync DUY NHẤT cho toàn chuỗi resident vẫn là downloadSync() ở
  # cuda_driver.nim (cuStreamSynchronize trên gStream) khi thật sự cần lấy
  # kết quả về host.
  debugLog("cudaMatmulF32R m=" & $m & " k=" & $k & " n=" & $n)
  result = CudaTensor(dptr: cast[cuda_driver.CUdeviceptr](dC), bytes: bytesC, numel: m * n)