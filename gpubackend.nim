# gpubackend.nim - Lớp điều phối GPU cho BybyLang.
# Cú pháp trong .bybylang:
#   gpu backend is "auto"     # hoặc "cuda" / "metal" / "opencl" / "cpu"
#   gpu array A = [1, 2, 3, 4]
#   gpu add A, B -> C size 4
#   gpu sub A, B -> C
#   gpu mul A, B -> C
#   gpu div A, B -> C
#
# "auto" sẽ tự dò: CUDA (NVIDIA) -> Metal (macOS) -> OpenCL (mọi GPU/CPU khác) -> CPU thuần.
# Nếu backend được chỉ định cụ thể mà không khả dụng, hoặc chạy lỗi, sẽ tự rơi (fallback)
# về CPU để chương trình luôn cho ra kết quả đúng thay vì crash.
import std/[strutils, math]
import backends/cuda/cuda_driver
import backends/cuda/cuda_runtime
import backends/opencl/opencl_api
import backends/metal/metal_backend
import tsic_ir

type
  GpuBackend* = enum
    gbAuto, gbCpu, gbCuda, gbMetal, gbOpenCL, gbTsic

proc parseBackend*(s: string): GpuBackend =
  case s.strip().toLowerAscii()
  of "cuda": gbCuda
  of "metal": gbMetal
  of "opencl": gbOpenCL
  of "cpu": gbCpu
  of "tsic": gbTsic
  else: gbAuto

proc `$`*(b: GpuBackend): string =
  case b
  of gbAuto: "auto"
  of gbCpu: "cpu"
  of gbCuda: "cuda"
  of gbMetal: "metal"
  of gbOpenCL: "opencl"
  of gbTsic: "tsic"

proc detectBackend*(): GpuBackend =
  ## Tự động dò backend GPU tốt nhất hiện có trên máy đang chạy.
  ## "tsic" là backend trung gian cho GPU tự chế -> không nằm trong auto-detect,
  ## phải chọn tường minh bằng `gpu backend is "tsic"`.
  if cudaAvailable():
    return gbCuda
  when defined(macosx):
    if metalAvailable():
      return gbMetal
  if openclAvailable():
    return gbOpenCL
  return gbCpu

# --- CPU implementations of activation and layers ---
proc cpuRelu(x: seq[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0..<x.len:
    result[i] = max(x[i], 0'f32)

proc cpuSigmoid(x: seq[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0..<x.len:
    result[i] = 1'f32 / (1'f32 + exp(-x[i]))

proc cpuTanh(x: seq[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0..<x.len:
    result[i] = tanh(x[i])

proc cpuSoftmax(x: seq[float32], rows, cols: int): seq[float32] =
  result = newSeq[float32](rows * cols)
  for r in 0..<rows:
    let off = r * cols
    var maxVal = x[off]
    for c in 1..<cols:
      if x[off + c] > maxVal:
        maxVal = x[off + c]
    var sum: float32 = 0'f32
    for c in 0..<cols:
      let e = exp(x[off + c] - maxVal)
      result[off + c] = e
      sum += e
    for c in 0..<cols:
      result[off + c] /= sum

proc cpuLayernorm(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
  result = newSeq[float32](rows * cols)
  for r in 0..<rows:
    let off = r * cols
    var mean: float32 = 0'f32
    for c in 0..<cols:
      mean += x[off + c]
    mean /= float32(cols)
    var varr: float32 = 0'f32
    for c in 0..<cols:
      let diff = x[off + c] - mean
      varr += diff * diff
    varr /= float32(cols)
    let invStd = 1'f32 / sqrt(varr + eps)
    for c in 0..<cols:
      result[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c]

proc cpuEmbeddingLookup(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
  let num = indices.len
  result = newSeq[float32](num * dim)
  for i in 0..<num:
    let idx = indices[i]
    if idx >= 0 and idx < vocab:
      for j in 0..<dim:
        result[i * dim + j] = table[idx * dim + j]
    else:
      for j in 0..<dim:
        result[i * dim + j] = 0'f32

proc cpuVecOp(op: string, a, b: seq[float32]): seq[float32] =
  let n = min(a.len, b.len)
  result = newSeq[float32](n)
  case op
  of "add":
    for i in 0..<n: result[i] = a[i] + b[i]
  of "sub":
    for i in 0..<n: result[i] = a[i] - b[i]
  of "mul":
    for i in 0..<n: result[i] = a[i] * b[i]
  of "div":
    for i in 0..<n: result[i] = a[i] / b[i]
  else:
    raise newException(ValueError, "Unknown gpu op: " & op)

proc gpuOp*(op: string, backend: GpuBackend, a, b: seq[float32]): seq[float32] =
  ## Điểm vào chính: chạy phép toán elementwise `op` (add/sub/mul/div) trên `backend`.
  ## Nếu backend là auto thì tự detect. Nếu backend chọn thất bại lúc chạy, fallback CPU.
  var chosen = backend
  if chosen == gbAuto:
    chosen = detectBackend()

  try:
    case chosen
    of gbCuda:
      return cudaVecOp(op, a, b)
    of gbMetal:
      return metalVecOp(op, a, b)
    of gbOpenCL:
      return openclVecOp(op, a, b)
    of gbTsic:
      return tsicVecOp(op, a, b)
    of gbCpu, gbAuto:
      return cpuVecOp(op, a, b)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    stderr.writeLine("[GPU] backend '" & $chosen & "' failed (" & e.msg & "), fallback CPU")
    return cpuVecOp(op, a, b)

# --- Unified GPU entrance points for custom layers/activations ---
proc gpuRelu*(backend: GpuBackend, x: seq[float32]): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbCuda: return cudaActivation("relu", x)
    of gbMetal: return metalActivation("relu", x)
    of gbOpenCL: return openclActivation("relu", x)
    of gbTsic: return tsicRelu(x)
    of gbCpu, gbAuto: return cpuRelu(x)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    stderr.writeLine("[GPU] relu failed on " & $chosen & " (" & e.msg & "), fallback CPU")
    return cpuRelu(x)

proc gpuSigmoid*(backend: GpuBackend, x: seq[float32]): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbCuda: return cudaActivation("sigmoid", x)
    of gbMetal: return metalActivation("sigmoid", x)
    of gbOpenCL: return openclActivation("sigmoid", x)
    of gbTsic: return tsicSigmoid(x)
    of gbCpu, gbAuto: return cpuSigmoid(x)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    stderr.writeLine("[GPU] sigmoid failed on " & $chosen & " (" & e.msg & "), fallback CPU")
    return cpuSigmoid(x)

proc gpuTanh*(backend: GpuBackend, x: seq[float32]): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbCuda: return cudaActivation("tanh", x)
    of gbMetal: return metalActivation("tanh", x)
    of gbOpenCL: return openclActivation("tanh", x)
    of gbTsic: return tsicTanh(x)
    of gbCpu, gbAuto: return cpuTanh(x)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    stderr.writeLine("[GPU] tanh failed on " & $chosen & " (" & e.msg & "), fallback CPU")
    return cpuTanh(x)

proc gpuSoftmax*(backend: GpuBackend, x: seq[float32], rows, cols: int): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbCuda: return cudaSoftmax(x, rows, cols)
    of gbMetal: return metalSoftmax(x, rows, cols)
    of gbOpenCL: return openclSoftmax(x, rows, cols)
    of gbTsic: return tsicSoftmax(x, rows, cols)
    of gbCpu, gbAuto: return cpuSoftmax(x, rows, cols)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    stderr.writeLine("[GPU] softmax failed on " & $chosen & " (" & e.msg & "), fallback CPU")
    return cpuSoftmax(x, rows, cols)

proc gpuLayernorm*(backend: GpuBackend, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbCuda: return cudaLayernorm(x, gamma, beta, rows, cols, eps)
    of gbMetal: return metalLayernorm(x, gamma, beta, rows, cols, eps)
    of gbOpenCL: return openclLayernorm(x, gamma, beta, rows, cols, eps)
    of gbTsic: return tsicLayernorm(x, gamma, beta, rows, cols, eps)
    of gbCpu, gbAuto: return cpuLayernorm(x, gamma, beta, rows, cols, eps)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    stderr.writeLine("[GPU] layernorm failed on " & $chosen & " (" & e.msg & "), fallback CPU")
    return cpuLayernorm(x, gamma, beta, rows, cols, eps)

proc gpuEmbeddingLookup*(backend: GpuBackend, table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbCuda: return cudaEmbeddingLookup(table, indices, vocab, dim)
    of gbMetal: return metalEmbeddingLookup(table, indices, vocab, dim)
    of gbOpenCL: return openclEmbeddingLookup(table, indices, vocab, dim)
    of gbTsic: return tsicEmbeddingLookup(table, indices, vocab, dim)
    of gbCpu, gbAuto: return cpuEmbeddingLookup(table, indices, vocab, dim)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    stderr.writeLine("[GPU] embedding lookup failed on " & $chosen & " (" & e.msg & "), fallback CPU")
    return cpuEmbeddingLookup(table, indices, vocab, dim)

proc cpuMatmul(a, b: seq[float32], m, k, n: int): seq[float32] =
  result = newSeq[float32](m * n)
  for i in 0..<m:
    for j in 0..<n:
      var s: float32 = 0
      for p in 0..<k:
        s += a[i*k + p] * b[p*n + j]
      result[i*n + j] = s

proc gpuMatmul*(backend: GpuBackend, a, b: seq[float32], m, k, n: int): seq[float32] =
  ## C(m x n) = A(m x k) * B(k x n), row-major. Trên "cuda" dùng cudart + cuBLAS
  ## (cuda_runtime.nim) với Tensor Core math mode bật sẵn -> tận dụng Tensor Core
  ## khi GPU hỗ trợ, tự fallback FP32 CUDA core bình thường nếu không.
  var chosen = backend
  if chosen == gbAuto:
    chosen = detectBackend()

  try:
    case chosen
    of gbCuda:
      return cudaMatmulF32(a, b, m, k, n)
    of gbMetal:
      return metalMatmul(a, b, m, k, n)
    of gbOpenCL:
      return openclMatmul(a, b, m, k, n)
    of gbTsic:
      return tsicMatmulOp(a, b, m, k, n)
    of gbCpu, gbAuto:
      return cpuMatmul(a, b, m, k, n)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    stderr.writeLine("[GPU] matmul backend '" & $chosen & "' failed (" & e.msg & "), fallback CPU")
    return cpuMatmul(a, b, m, k, n)

proc gpuMatmul2*(backend: GpuBackend,
                  a1, b1: seq[float32], m1, k1, n1: int,
                  a2, b2: seq[float32], m2, k2, n2: int):
                  tuple[c1, c2: seq[float32]] =
  ## Chạy 2 phép matmul ĐỘC LẬP (a1*b1->c1 và a2*b2->c2, không cái nào phụ
  ## thuộc kết quả cái kia) chỉ với MỘT round-trip GPU khi backend hỗ trợ.
  ##   - metal: gộp thật vào 1 command buffer (metal_matmul2), tiết kiệm
  ##     overhead commit+wait so với gọi gpuMatmul() hai lần.
  ##   - cuda/opencl/tsic/cpu: chưa có API "gộp" ở tầng driver -> gọi
  ##     gpuMatmul() hai lần tuần tự (cuBLAS + CUDA stream đã tự pipeline hoá
  ##     nội bộ khá tốt nên chênh lệch không lớn như trên Metal).
  ## Đây là API được thêm vào để phục vụ các framework nhúng BybyLang làm
  ## backend GPU của họ (vd. các framework train/infer transformer cần dW/dX
  ## trong backward) — không cần tự viết shader/kernel matmul riêng nữa.
  var chosen = backend
  if chosen == gbAuto:
    chosen = detectBackend()

  try:
    case chosen
    of gbMetal:
      let r = metalMatmul2(a1, b1, m1, k1, n1, a2, b2, m2, k2, n2)
      return (r.c1, r.c2)
    else:
      let c1 = gpuMatmul(chosen, a1, b1, m1, k1, n1)
      let c2 = gpuMatmul(chosen, a2, b2, m2, k2, n2)
      return (c1, c2)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    stderr.writeLine("[GPU] matmul2 backend '" & $chosen & "' failed (" & e.msg & "), fallback CPU")
    return (cpuMatmul(a1, b1, m1, k1, n1), cpuMatmul(a2, b2, m2, k2, n2))

# Biến toàn cục backend hiện tại, sinh code sẽ set/đọc biến này (mặc định auto).
var gpuBackendSelected*: GpuBackend = gbAuto
