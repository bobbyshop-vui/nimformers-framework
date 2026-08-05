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

when defined(macosx):
  # metal_backend.nim đã có Begin/Upload/AllocScratch/MatmulEnc/SoftmaxEnc/
  # End/Read/Free. VecOp và Activation encode-in-session được implement
  # thẳng trong shim ObjC (metal_shim.m/.h), khai báo ở đây (PHẢI đặt trước
  # nơi dùng - sessionVecOp/sessionActivation bên dưới - vì Nim không cho
  # forward-reference proc top-level).
  proc metalSessionVecOpEnc*(op: cint, a, b, c: MetalBufferHandle, n: int): cint {.
    importc: "metal_vecop_enc", header: "metal_shim.h".}

  proc metalSessionActivationEnc*(op: cint, x, y: MetalBufferHandle, n: int): cint {.
    importc: "metal_activation_enc", header: "metal_shim.h".}

  proc metalSessionRopeEnc*(x, cosT, sinT: MetalBufferHandle, T, Hn, D: cint): cint {.
    importc: "metal_rope_enc", header: "metal_shim.h".}

type
  GpuBackend* = enum
    gbAuto, gbCpu, gbCuda, gbMetal, gbOpenCL, gbTsic

# ─────────────────────────────────────────────────────────────
# NGHIÊM CẤM TRAIN BẰNG CPU: mặc định trước đây, bất kỳ backend GPU nào lỗi
# (thiếu driver, OOM, kernel launch fail...) đều ÂM THẦM fallback về CPU và
# vẫn cho train tiếp -> người dùng tưởng đang train trên GPU nhưng thực ra là
# CPU (chậm hơn hàng chục-hàng trăm lần, và không ai để ý vì log fallback chỉ
# in ra stderr một dòng).
#
# Cờ dưới đây, khi bật (mặc định BẬT), sẽ làm mọi lần "fallback CPU" RAISE lỗi
# thay vì tự động chạy CPU. Chỉ nên tắt (set false) khi CHỦ ĐỘNG muốn chạy demo
# trên máy không có GPU, không phải trong quá trình train thật.
var gForbidCpuFallback* = true

proc setForbidCpuFallback*(forbid: bool) =
  gForbidCpuFallback = forbid

proc handleGpuFailure(backend: GpuBackend, opName: string, e: ref CatchableError) =
  ## Gọi khi 1 backend GPU đã chọn tường minh (không phải "auto" rơi về cpu vì
  ## detectBackend() không thấy GPU nào) bị lỗi lúc chạy. Raise thẳng nếu đang
  ## cấm fallback CPU, thay vì âm thầm chạy CPU.
  if gForbidCpuFallback:
    raise newException(CatchableError,
      "[GPU] backend '" & $backend & "' that bai khi chay '" & opName & "' (" & e.msg & "). " &
      "Fallback CPU dang bi CAM (gForbidCpuFallback=true) vi day la vong lap TRAIN. " &
      "Neu ban thuc su muon chay CPU (vd. demo/test tren may khong GPU), goi " &
      "setForbidCpuFallback(false) tuong minh truoc.")
  stderr.writeLine("[GPU] backend '" & $backend & "' failed (" & e.msg & "), fallback CPU")

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

proc resolveTsicBackend*(): GpuBackend =
  ## "tsic" không phải 1 thiết bị vật lý riêng - nó tự dò xem CUDA/Metal/OpenCL
  ## nào đang khả dụng rồi chuyển tiếp (giống cách tsic_ir.tsicInit() làm cho
  ## các hàm tsicVecOp/tsicRelu/...). Session API dùng lại đúng logic đó để
  ## backend "tsic" cũng chạy được Session API chung.
  case tsicInit()
  of tbCuda: gbCuda
  of tbMetal: gbMetal
  of tbOpenCL: gbOpenCL
  of tbUnknown: gbCpu

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

proc cpuApflu(x: seq[float32], alpha, beta: float32): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0..<x.len:
    let val = x[i]
    result[i] = if val > 0'f32: val * (1'f32 + alpha * val) else: beta * val * exp(val)

proc cpuApfluBackward(x, dy: seq[float32], alpha, beta: float32): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0..<x.len:
    let val = x[i]
    let d = dy[i]
    result[i] = if val > 0'f32: d * (1'f32 + 2'f32 * alpha * val) else: d * beta * exp(val) * (1'f32 + val)

proc cpuLayernormBackward(dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  var dx = newSeq[float32](rows * cols)
  var dgamma = newSeq[float32](cols)
  var dbeta = newSeq[float32](cols)
  for r in 0..<rows:
    let off = r * cols
    var mean: float32 = 0
    for c in 0..<cols: mean += x[off + c]
    mean /= float32(cols)
    var varr: float32 = 0
    for c in 0..<cols:
      let diff = x[off + c] - mean
      varr += diff * diff
    varr /= float32(cols)
    let invStd = 1'f32 / sqrt(varr + eps)

    # Calculate parameter gradients
    for c in 0..<cols:
      let norm = (x[off + c] - mean) * invStd
      dgamma[c] += dy[off + c] * norm
      dbeta[c] += dy[off + c]

    # Calculate input gradients (dx)
    var sum1: float32 = 0
    var sum2: float32 = 0
    for c in 0..<cols:
      let grad = dy[off + c] * gamma[c] * invStd
      sum1 += grad
      sum2 += grad * (x[off + c] - mean)
    sum2 *= (-invStd * invStd / float32(cols))
    for c in 0..<cols:
      let term1 = dy[off + c] * gamma[c] * invStd
      let term2 = sum1 / float32(cols)
      let term3 = (x[off + c] - mean) * sum2 * 2.0'f32 / float32(cols)
      dx[off + c] = term1 - term2 + term3
  return (dx, dgamma, dbeta)

proc cpuAttentionFused(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  var o = newSeq[float32](B * H * S * D)
  var s_matrix = newSeq[float32](B * H * S * S)
  for b in 0..<B:
    for h in 0..<H:
      let base_idx = (b * H + h) * S * D
      let base_s = (b * H + h) * S * S
      for ti in 0..<S:
        var scores = newSeq[float32](S)
        var mx = -1e30'f32
        for tj in 0..ti:
          var dot: float32 = 0
          for d in 0..<D:
            dot += q[base_idx + ti * D + d] * k[base_idx + tj * D + d]
          scores[tj] = dot * scale
          if scores[tj] > mx: mx = scores[tj]
        var sum_exp = 0'f32
        for tj in 0..ti:
          scores[tj] = exp(scores[tj] - mx)
          sum_exp += scores[tj]
        for tj in 0..ti:
          scores[tj] /= sum_exp
          s_matrix[base_s + ti * S + tj] = scores[tj]

        # Output = S @ V
        for d in 0..<D:
          var acc = 0'f32
          for tj in 0..ti:
            acc += scores[tj] * v[base_idx + tj * D + d]
          o[base_idx + ti * D + d] = acc
  return (o, s_matrix)

proc cpuAttentionFusedBackward(q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  var dq = newSeq[float32](B * H * S * D)
  var dk = newSeq[float32](B * H * S * D)
  var dv = newSeq[float32](B * H * S * D)
  for b in 0..<B:
    for h in 0..<H:
      let base_idx = (b * H + h) * S * D
      let base_s = (b * H + h) * S * S
      for ti in 0..<S:
        var softmaxW = newSeq[float32](ti + 1)
        for tj in 0..ti:
          softmaxW[tj] = s_matrix[base_s + ti * S + tj]

        var dSoftmax = newSeq[float32](ti + 1)
        for tj in 0..ti:
          var dotVal = 0'f32
          for d in 0..<D:
            let dyVal = dy[base_idx + ti * D + d]
            dv[base_idx + tj * D + d] += softmaxW[tj] * dyVal
            dotVal += dyVal * v[base_idx + tj * D + d]
          dSoftmax[tj] = dotVal

        var dotSum = 0'f32
        for tj in 0..ti:
          dotSum += softmaxW[tj] * dSoftmax[tj]

        for tj in 0..ti:
          let dScore = softmaxW[tj] * (dSoftmax[tj] - dotSum) * scale
          for d in 0..<D:
            dq[base_idx + ti * D + d] += dScore * k[base_idx + tj * D + d]
            dk[base_idx + tj * D + d] += dScore * q[base_idx + ti * D + d]
  return (dq, dk, dv)

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

proc cpuFusedAddAct*(a, b: seq[float32], act: string): seq[float32] =
  let n = min(a.len, b.len)
  result = newSeq[float32](n)
  case act
  of "relu":
    for i in 0..<n:
      let s = a[i] + b[i]
      result[i] = max(s, 0'f32)
  of "sigmoid":
    for i in 0..<n:
      let s = a[i] + b[i]
      result[i] = 1'f32 / (1'f32 + exp(-s))
  of "tanh":
    for i in 0..<n:
      let s = a[i] + b[i]
      result[i] = tanh(s)
  of "none":
    for i in 0..<n:
      result[i] = a[i] + b[i]
  else:
    raise newException(ValueError, "cpuFusedAddAct: unknown activation: " & act)

proc gpuFusedAddAct*(backend: GpuBackend, a, b: seq[float32], act: string): seq[float32] =
  var chosen = backend
  if chosen == gbAuto:
    chosen = detectBackend()
  case chosen
  of gbCpu, gbAuto:
    return cpuFusedAddAct(a, b, act)
  else:
    raise newException(CatchableError,
      "gpuFusedAddAct: backend " & $chosen & " CHƯA có kernel fused thật " &
      "(chỉ có bản CPU + bản TSIC IR sinh mã nguồn PTX/MSL/OpenCL C chưa test " &
      "trên phần cứng thật). Dùng cpuFusedAddAct trực tiếp, hoặc tự thêm kernel " &
      "GPU thật + test so khớp trước khi bật đường này.")

proc gpuOp*(op: string, backend: GpuBackend, a, b: seq[float32]): seq[float32] =
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
    handleGpuFailure(chosen, "vecop:" & op, e)
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
    handleGpuFailure(chosen, "relu", e)
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
    handleGpuFailure(chosen, "sigmoid", e)
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
    handleGpuFailure(chosen, "tanh", e)
    return cpuTanh(x)

# ============================================================
# SỬA: GPU APFLU - CUDA GỌI KERNEL THẬT
# ============================================================
proc gpuApflu*(backend: GpuBackend, x: seq[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbCuda:
      return cudaApflu(x, alpha, beta)
    of gbMetal:
      return metalApflu(x, alpha, beta)
    of gbOpenCL:
      return openclApflu(x, alpha, beta)
    of gbTsic:
      return tsicApflu(x, alpha, beta)
    of gbCpu, gbAuto:
      return cpuApflu(x, alpha, beta)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    handleGpuFailure(chosen, "apflu", e)
    return cpuApflu(x, alpha, beta)

# ============================================================
# SỬA: GPU APFLU BACKWARD - CUDA GỌI KERNEL THẬT
# ============================================================
proc gpuApfluBackward*(backend: GpuBackend, x, dy: seq[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbCuda:
      return cudaApfluBackward(x, dy, alpha, beta)
    of gbMetal:
      return metalApfluBackward(x, dy, alpha, beta)
    of gbOpenCL:
      return openclApfluBackward(x, dy, alpha, beta)
    of gbTsic:
      return tsicApfluBackward(x, dy, alpha, beta)
    of gbCpu, gbAuto:
      return cpuApfluBackward(x, dy, alpha, beta)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    handleGpuFailure(chosen, "apflu_backward", e)
    return cpuApfluBackward(x, dy, alpha, beta)

# ============================================================
# SỬA: GPU LAYERNORM BACKWARD - CUDA GỌI KERNEL THẬT
# ============================================================
proc gpuLayernormBackward*(backend: GpuBackend, dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbCuda:
      return cudaLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
    of gbMetal:
      return metalLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
    of gbOpenCL:
      return openclLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
    of gbTsic:
      return tsicLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
    of gbCpu, gbAuto:
      return cpuLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    handleGpuFailure(chosen, "layernorm_backward", e)
    return cpuLayernormBackward(dy, x, gamma, beta, rows, cols, eps)

# ============================================================
# SỬA: GPU ATTENTION FUSED - CUDA GỌI KERNEL THẬT
# ============================================================
proc gpuAttentionFused*(backend: GpuBackend, q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbCuda:
      return cudaAttentionFused(q, k, v, mask, B, H, S, D, scale)
    of gbMetal:
      return metalAttentionFused(q, k, v, mask, B, H, S, D, scale)
    of gbOpenCL:
      return openclAttentionFused(q, k, v, mask, B, H, S, D, scale)
    of gbTsic:
      return tsicAttentionFused(q, k, v, mask, B, H, S, D, scale)
    of gbCpu, gbAuto:
      return cpuAttentionFused(q, k, v, mask, B, H, S, D, scale)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    handleGpuFailure(chosen, "attention_fused", e)
    return cpuAttentionFused(q, k, v, mask, B, H, S, D, scale)

# ============================================================
# SỬA: GPU ATTENTION FUSED BACKWARD - CUDA GỌI KERNEL THẬT
# ============================================================
proc gpuAttentionFusedBackward*(backend: GpuBackend, q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbCuda:
      return cudaAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
    of gbMetal:
      return metalAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
    of gbOpenCL:
      return openclAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
    of gbTsic:
      return tsicAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
    of gbCpu, gbAuto:
      return cpuAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    handleGpuFailure(chosen, "attention_fused_backward", e)
    return cpuAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)

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
    handleGpuFailure(chosen, "softmax", e)
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
    handleGpuFailure(chosen, "layernorm", e)
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
    handleGpuFailure(chosen, "embedding_lookup", e)
    return cpuEmbeddingLookup(table, indices, vocab, dim)

proc cpuMatmul(a, b: seq[float32], m, k, n: int): seq[float32] =
  const blockSize = 64
  result = newSeq[float32](m * n)
  var ii = 0
  while ii < m:
    let iMax = min(ii + blockSize, m)
    var kk = 0
    while kk < k:
      let kMax = min(kk + blockSize, k)
      var jj = 0
      while jj < n:
        let jMax = min(jj + blockSize, n)
        for i in ii..<iMax:
          let cRow = i * n
          for p in kk..<kMax:
            let aVal = a[i*k + p]
            if aVal == 0'f32: continue
            let bRow = p * n
            for j in jj..<jMax:
              result[cRow + j] += aVal * b[bRow + j]
        jj += blockSize
      kk += blockSize
    ii += blockSize

proc gpuMatmul*(backend: GpuBackend, a, b: seq[float32], m, k, n: int): seq[float32] =
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
    handleGpuFailure(chosen, "matmul", e)
    return cpuMatmul(a, b, m, k, n)

proc gpuMatmul2*(backend: GpuBackend,
                  a1, b1: seq[float32], m1, k1, n1: int,
                  a2, b2: seq[float32], m2, k2, n2: int):
                  tuple[c1, c2: seq[float32]] =
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
    handleGpuFailure(chosen, "matmul2", e)
    return (cpuMatmul(a1, b1, m1, k1, n1), cpuMatmul(a2, b2, m2, k2, n2))

proc cpuDequantMatmulQ4Fallback(a: seq[float32], wq: seq[uint8], scales, zeros: seq[float32],
                                 m, k, n, groupSize, nGroupsPerRow: int): seq[float32] =
  ## Fallback CPU-thuan (dung cho CUDA/OpenCL/gbCpu - CHUA co kernel int4
  ## truc tiep tren 2 backend nay, chi Metal moi co matmul_q4_kernel).
  ## Dequant dung CHINH XAC cong thuc trong quant.nim (LUT 16 gia tri/group)
  ## roi goi lai gpuMatmul binh thuong - dam bao ket qua giong het duong
  ## matmul_q4_kernel tren Metal, chi khac la khong tiet kiem duoc bang thong
  ## upload (van phai vat chat hoa wT fp32).
  let bytesPerRow = (k + 1) div 2
  var wT = newSeq[float32](k * n)  # layout [K, N] = transposed, khop voi gpuMatmul(a[m,k], wT[k,n])
  var lut: array[16, float32]
  for row in 0 ..< n:
    let rowByteOff = row * bytesPerRow
    let rowGroupOff = row * nGroupsPerRow
    var g = -1
    for c in 0 ..< k:
      let gNew = if groupSize > 0: c div groupSize else: 0
      if gNew != g:
        g = gNew
        let sc = scales[rowGroupOff + g]
        let zp = zeros[rowGroupOff + g]
        for kk in 0 ..< 16: lut[kk] = (float32(kk) - zp) * sc
      let byteIdx = rowByteOff + c div 2
      let nibble = if c mod 2 == 0: wq[byteIdx] and 0x0F'u8 else: (wq[byteIdx] shr 4) and 0x0F'u8
      wT[c * n + row] = lut[int(nibble)]
  return cpuMatmul(a, wT, m, k, n)

proc gpuMatmulQ4*(backend: GpuBackend, a: seq[float32], wq: seq[uint8], scales, zeros: seq[float32],
                   m, k, n, groupSize, nGroupsPerRow: int): seq[float32] =
  ## SUA (item 3 - quantization): tranh dequantizeTensorTransposed() ra fp32
  ## day du tren CPU + upload nguyen mang fp32 do moi lan Linear.forward()
  ## goi ham nay. Metal va OpenCL co kernel truc tiep tren int4; CUDA con
  ## lai fallback ve cpuDequantMatmulQ4Fallback (van dung, chi chua duoc toi
  ## uu bang thong - TODO: viet kernel .cu tuong tu neu can, hien khong sua
  ## duoc vi vecop.ptx la binary da bien dich san, khong co source .cu di kem
  ## trong repo nay de sua/build lai).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbMetal:
      return metalMatmulQ4(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
    of gbOpenCL:
      return openclMatmulQ4(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
    of gbCuda:
      try:
        return cuda_driver.cudaMatmulQ4(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
      except CatchableError as e:
        stderr.write("[cuda] matmul_q4_kernel launch thất bại (" & e.msg & "), fallback CPU dequant\n")
        return cpuDequantMatmulQ4Fallback(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
    else:
      return cpuDequantMatmulQ4Fallback(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
  except CatchableError as e:
    handleGpuFailure(chosen, "matmul_q4", e)
    return cpuDequantMatmulQ4Fallback(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)

# Biến toàn cục backend hiện tại, sinh code sẽ set/đọc biến này (mặc định auto).
var gpuBackendSelected*: GpuBackend = gbAuto

# ═══════════════════════════════════════════════════════════════════════════
# API resident cho CUDA
# ═══════════════════════════════════════════════════════════════════════════
import backends/cuda/cuda_driver as cudaDrv
import backends/cuda/cuda_runtime as cudaRt

# Export CudaTensor từ cuda_driver để module khác dùng được
export cudaDrv.CudaTensor

type CudaResidentTensor* = cudaDrv.CudaTensor

proc cuUpload*(data: seq[float32]): CudaResidentTensor = cudaDrv.uploadAsync(data)
proc cuUploadIndices*(data: seq[int32]): CudaResidentTensor = cudaDrv.uploadIndicesAsync(data)
proc cuDownload*(t: CudaResidentTensor): seq[float32] = cudaDrv.downloadSync(t)
proc cuFree*(t: var CudaResidentTensor) = cudaDrv.freeResident(t)

proc cuMatmulR*(a, b: CudaResidentTensor, m, k, n: int): CudaResidentTensor =
  cudaRt.cudaMatmulF32R(a, b, m, k, n)

proc cuAddR*(a, b: CudaResidentTensor): CudaResidentTensor =
  cudaDrv.cudaVecOpR("add", a, b)

proc cuSubR*(a, b: CudaResidentTensor): CudaResidentTensor =
  cudaDrv.cudaVecOpR("sub", a, b)

proc cuMulR*(a, b: CudaResidentTensor): CudaResidentTensor =
  cudaDrv.cudaVecOpR("mul", a, b)

proc cuDivR*(a, b: CudaResidentTensor): CudaResidentTensor =
  cudaDrv.cudaVecOpR("div", a, b)

proc cuReluR*(x: CudaResidentTensor): CudaResidentTensor =
  cudaDrv.cudaActivationR("relu", x)

proc cuSigmoidR*(x: CudaResidentTensor): CudaResidentTensor =
  cudaDrv.cudaActivationR("sigmoid", x)

proc cuTanhR*(x: CudaResidentTensor): CudaResidentTensor =
  cudaDrv.cudaActivationR("tanh", x)

proc cuSoftmaxR*(x: CudaResidentTensor, rows, cols: int): CudaResidentTensor =
  cudaDrv.cudaSoftmaxR(x, rows, cols)

proc cuLayernormR*(x, gamma, beta: CudaResidentTensor, rows, cols: int, eps: float32): CudaResidentTensor =
  cudaDrv.cudaLayernormR(x, gamma, beta, rows, cols, eps)

proc cuAttentionFusedR*(q, k, v: CudaResidentTensor, B, H, S, D: int, scale: float32): tuple[o, s_matrix: CudaResidentTensor] =
  cudaDrv.cudaAttentionFusedR(q, k, v, B, H, S, D, scale)

proc cuCopyIntoR*(src, dst: CudaResidentTensor, dstOffsetElems, n: int) =
  cudaDrv.cudaCopyIntoR(src, dst, dstOffsetElems, n)

proc cuRopeR*(x, cosT, sinT: CudaResidentTensor, T, Hn, D: int) =
  cudaDrv.cudaRopeR(x, cosT, sinT, T, Hn, D)

proc cuEmbeddingLookupR*(table, indices: CudaResidentTensor, numIdx, vocab, dim: int): CudaResidentTensor =
  cudaDrv.cudaEmbeddingLookupR(table, indices, numIdx, vocab, dim)

proc noop_marker_end_of_cuda_resident_section*() = discard

type
  SessionHandle* = object
    kind*: GpuBackend
    metalHandle*: MetalBufferHandle  # Metal
    cudaHandle*: CudaResidentTensor  # CUDA
    openclHandle*: pointer           # OpenCL

# ─────────────────────────────────────────────────────────────
# SESSION API CHUNG - Giao diện thống nhất cho cả 3 backend
# ─────────────────────────────────────────────────────────────

proc sessionBegin*(backend: GpuBackend): bool =
  ## Bắt đầu session - khởi tạo command buffer cho backend được chọn
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionBegin()
    else:
      return false
  of gbCuda:
    # CUDA session = dùng cùng 1 stream (đã có sẵn từ cuda_driver)
    # Không cần làm gì đặc biệt, chỉ cần đánh dấu
    return true
  of gbOpenCL:
    # OpenCL session = dùng cùng 1 command queue (đã có sẵn)
    return true
  else:
    return false

proc sessionUpload*(backend: GpuBackend, data: seq[float32]): SessionHandle =
  ## Upload dữ liệu lên GPU trong session hiện tại
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  result.kind = chosen
  case chosen
  of gbMetal:
    when defined(macosx):
      result.metalHandle = metalSessionUpload(data)
    else:
      result.metalHandle = nil
  of gbCuda:
    let t = cuUpload(data)
    result.cudaHandle = t
  of gbOpenCL:
    result.openclHandle = openclSessionUpload(data)
  else:
    discard

proc sessionUploadPrivate*(backend: GpuBackend, data: seq[float32]): SessionHandle =
  ## Bản Private-storage của sessionUpload - CHỈ dùng cho weight RESIDENT
  ## (upload 1 LẦN lúc load model, CPU không đọc/ghi lại sau đó) - hiện dùng
  ## cho beResidentUploadFp32 (weight int8/int4-symmetric/tied-embedding qua
  ## uploadResident trong nimformer.nim). Metal: MTLResourceStorageModePrivate.
  ## OpenCL: CL_MEM_READ_ONLY|CL_MEM_HOST_NO_ACCESS (openclSessionUploadPrivate) -
  ## cùng ý nghĩa "host không đụng lại sau upload". CUDA: alias thẳng về
  ## cuUpload thường - VRAM CUDA vốn đã tách biệt CPU hoàn toàn (không có
  ## khái niệm Shared/Private như Metal UMA hay hint như OpenCL), nên không
  ## có gì thêm để tối ưu ở đây.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  result.kind = chosen
  case chosen
  of gbMetal:
    when defined(macosx):
      result.metalHandle = metalUploadPrivate(data)
    else:
      result.metalHandle = nil
  of gbCuda:
    let t = cuUpload(data)
    result.cudaHandle = t
  of gbOpenCL:
    result.openclHandle = openclSessionUploadPrivate(data)
  else:
    discard

proc sessionUploadIndices*(backend: GpuBackend, data: seq[int32]): SessionHandle =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()
  result.kind = chosen
  case chosen
  of gbCuda:
    let t = cuUploadIndices(data)
    result.cudaHandle = t
  else:
    discard

proc cuAllocScratch*(n: int): CudaResidentTensor =
  ## Cấp buffer rỗng trên CUDA (chưa init)
  let bytes = csize_t(n * sizeof(float32))
  let dptr = getDeviceBuf(bytes)
  return CudaResidentTensor(dptr: dptr, bytes: bytes, numel: n)

proc sessionAllocScratch*(backend: GpuBackend, n: int): SessionHandle =
  ## Cấp buffer output trong session
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()
  result.kind = chosen
  case chosen
  of gbMetal:
    when defined(macosx):
      result.metalHandle = metalSessionAllocScratch(n)
    else:
      result.metalHandle = nil
  of gbCuda:
    # Cấp buffer rỗng (chưa init)
    let t = cuAllocScratch(n)
    result.cudaHandle = t
  of gbOpenCL:
    result.openclHandle = openclSessionAllocScratch(n)
  else:
    discard

# ─────────────────────────────────────────────────────────────
# ENCODE OPS VÀO SESSION
# ─────────────────────────────────────────────────────────────

proc sessionMatmul*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int): bool =
  ## Encode matmul vào session: C(M x N) = A(M x K) @ B(K x N)
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG: c có thể chưa được cấp buffer (codegen chỉ khai báo
      # 'var hc: SessionHandle' rồi truyền thẳng vào đây, không gọi
      # sessionAllocScratch trước) - khác CUDA (tự cấp bên dưới), Metal
      # dùng thẳng buffer có sẵn trong c nên nếu c.metalHandle là nil,
      # kernel ghi vào buffer NULL, sessionRead sau đó đọc ra NULL và raise
      # "metal_buffer_read failed". Tự cấp phát nếu chưa có, để khớp hợp
      # đồng ngầm mà CUDA đã có từ trước.
      if c.metalHandle == nil:
        c.metalHandle = metalSessionAllocScratch(M * N)
      return metalSessionMatmulEnc(a.metalHandle, b.metalHandle, c.metalHandle, M, K, N)
    else:
      return false
  of gbCuda:
    # SỬA: trước đây tính ra cT rồi bỏ (# TODO: copy cT -> c) - c không đổi,
    # đọc lại sau đó ra dữ liệu rác/cũ. Giờ ghi thẳng kết quả vào c (đổi
    # signature c sang 'var' để làm được việc này) và giải phóng buffer cũ
    # của c (đã cấp bởi sessionAllocScratch) để không leak VRAM.
    let old = c.cudaHandle
    c.cudaHandle = cudaMatmulF32R(a.cudaHandle, b.cudaHandle, M, K, N)
    if old.dptr != 0: (var oldVar = old; cuFree(oldVar))
    return true
  of gbOpenCL:
    # SỬA BUG: xem giải thích ở nhánh gbMetal ngay trên - OpenCL cũng dùng
    # thẳng buffer có sẵn trong c, không tự cấp phát.
    if c.openclHandle == nil:
      c.openclHandle = openclSessionAllocScratch(M * N)
    return openclSessionMatmulEnc(cl_mem(a.openclHandle), cl_mem(b.openclHandle), cl_mem(c.openclHandle), M, K, N)
  else:
    return false

proc sessionMatmulNT*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int): bool =
  ## Encode matmul-NT vào session: C(M x N) = A(M x K) @ B(N x K)^T.
  ## B GIỮ NGUYÊN layout gốc [N,K] (vd K-cache/V-cache [S,D] tự nhiên) -
  ## KHÔNG cần transpose CPU trước khi upload/gọi hàm này. Dùng cho attention
  ## decode để loại bỏ hoàn toàn chi phí transpose O(S*D)/token trước đây.
  ## Nếu backend không hỗ trợ (trả về false) - caller PHẢI tự fallback về
  ## sessionMatmul với B đã transpose sẵn (đường cũ), KHÔNG coi false là lỗi.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG: xem giải thích chi tiết ở sessionMatmul - c cần được cấp
      # buffer trước khi dùng làm output trên Metal/OpenCL.
      if c.metalHandle == nil:
        c.metalHandle = metalSessionAllocScratch(M * N)
      return metalSessionMatmulNTEnc(a.metalHandle, b.metalHandle, c.metalHandle, M, K, N)
    else:
      return false
  of gbCuda:
    let old = c.cudaHandle
    c.cudaHandle = cudaMatmulNTF32R(a.cudaHandle, b.cudaHandle, M, K, N)
    if old.dptr != 0: (var oldVar = old; cuFree(oldVar))
    return true
  of gbOpenCL:
    if c.openclHandle == nil:
      c.openclHandle = openclSessionAllocScratch(M * N)
    return openclSessionMatmulNTEnc(cl_mem(a.openclHandle), cl_mem(b.openclHandle), cl_mem(c.openclHandle), M, K, N)
  else:
    return false

proc sessionSoftmax*(backend: GpuBackend, x: SessionHandle, y: var SessionHandle, rows, cols: int): bool =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG: xem giải thích chi tiết ở sessionMatmul.
      if y.metalHandle == nil:
        y.metalHandle = metalSessionAllocScratch(rows * cols)
      return metalSessionSoftmaxEnc(x.metalHandle, y.metalHandle, rows, cols)
    else:
      return false
  of gbCuda:
    let old = y.cudaHandle
    y.cudaHandle = cudaSoftmaxR(x.cudaHandle, rows, cols)
    if old.dptr != 0: (var oldVar = old; cuFree(oldVar))
    return true
  of gbOpenCL:
    if y.openclHandle == nil:
      y.openclHandle = openclSessionAllocScratch(rows * cols)
    return openclSessionSoftmaxEnc(cl_mem(x.openclHandle), cl_mem(y.openclHandle), rows, cols)
  else:
    return false

proc sessionLayernormEnc*(backend: GpuBackend, x, gamma, beta: SessionHandle, y: var SessionHandle,
                           rows, cols: int, eps: float32): bool =
  ## Encode layernorm vào session hiện tại.
  ## Metal: metal_layernorm_enc (đã có sẵn trong shim, giờ mới bind ra Nim).
  ## CUDA: cuLayernormR đã tồn tại từ trước và VỐN DĨ đã launch async lên
  ## cùng gStream không sync - dùng thẳng được, chỉ cần giải phóng buffer cũ
  ## của y (như sessionMatmul/sessionVecOp CUDA đã làm) để không leak VRAM.
  ## OpenCL: openclSessionLayernormEnc (mới thêm - kernel layernorm_kernel
  ## đã có sẵn trong vecop_matmul.cl từ trước, chỉ chưa có bản "enqueue vào
  ## queue đang mở, không tự write/read/finish" - giờ đã thêm).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()
  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG: xem giải thích chi tiết ở sessionMatmul.
      if y.metalHandle == nil:
        y.metalHandle = metalSessionAllocScratch(rows * cols)
      return metalSessionLayernormEnc(x.metalHandle, gamma.metalHandle, beta.metalHandle, y.metalHandle, rows, cols, eps)
    else:
      return false
  of gbCuda:
    let old = y.cudaHandle
    y.cudaHandle = cuLayernormR(x.cudaHandle, gamma.cudaHandle, beta.cudaHandle, rows, cols, eps)
    if old.dptr != 0: (var oldVar = old; cuFree(oldVar))
    return true
  of gbOpenCL:
    if y.openclHandle == nil:
      y.openclHandle = openclSessionAllocScratch(rows * cols)
    return openclSessionLayernormEnc(cl_mem(x.openclHandle), cl_mem(gamma.openclHandle),
                                      cl_mem(beta.openclHandle), cl_mem(y.openclHandle), rows, cols, eps)
  else:
    return false

proc sessionRmsnormEnc*(backend: GpuBackend, x, gamma: SessionHandle, y: var SessionHandle,
                         rows, cols: int, eps: float32): bool =
  ## Encode RMSNorm (không trừ mean, không beta - khác layernorm) vào session
  ## hiện tại. Mirror của sessionLayernormEnc ở trên.
  ## Metal: metalSessionRmsnormEnc (metal_backend.nim - trả về false nếu build
  ## không phải macosx).
  ## CUDA: cudaRmsnormR - nếu kernel "rmsnorm_kernel" chưa có trong .ptx đang
  ## nạp, getFn raise CatchableError; BẮT lỗi đó ở đây và coi là "chưa hỗ
  ## trợ" (trả về false) để nim_inference.nim rơi về đường CPU cũ, KHÔNG cần
  ## bật env var nào để dùng được đường nhanh này.
  ## OpenCL: openclSessionRmsnormEnc.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()
  case chosen
  of gbMetal:
    when defined(macosx):
      if y.metalHandle == nil:
        y.metalHandle = metalSessionAllocScratch(rows * cols)
      return metalSessionRmsnormEnc(x.metalHandle, gamma.metalHandle, y.metalHandle, rows, cols, eps)
    else:
      return false
  of gbCuda:
    try:
      let old = y.cudaHandle
      y.cudaHandle = cudaRmsnormR(x.cudaHandle, gamma.cudaHandle, rows, cols, eps)
      if old.dptr != 0: (var oldVar = old; cuFree(oldVar))
      return true
    except CatchableError:
      return false
  of gbOpenCL:
    if y.openclHandle == nil:
      y.openclHandle = openclSessionAllocScratch(rows * cols)
    return openclSessionRmsnormEnc(cl_mem(x.openclHandle), cl_mem(gamma.openclHandle),
                                    cl_mem(y.openclHandle), rows, cols, eps)
  else:
    return false

proc sessionAttentionFusedEnc*(backend: GpuBackend, q, k, v: SessionHandle, o, sMatrix: var SessionHandle,
                                B, H, S, D: int, scale: float32): bool =
  ## Encode attention_fused vào session hiện tại.
  ## Metal: metal_attention_fused_enc (mới viết).
  ## CUDA: cuAttentionFusedR (mới viết, mirror cudaLayernormR - launch async
  ## lên gStream, không tự sync).
  ## OpenCL: openclSessionAttentionFusedEnc (mới thêm). Kernel
  ## attention_fused_kernel VỐN DĨ đã có sẵn trong vecop_matmul.cl (dùng bởi
  ## openclAttentionFused bản immediate từ trước) - chỉ chưa có bản session
  ## không tự write/read/finish, giờ đã thêm. Cùng giới hạn S<=256 như bản
  ## immediate (chung 1 kernel).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()
  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG: xem giải thích chi tiết ở sessionMatmul - 2 output (o,
      # sMatrix) đều cần buffer trước. o có kích thước B*H*S*D (giống
      # q/k/v), sMatrix là ma trận điểm attention B*H*S*S.
      if o.metalHandle == nil:
        o.metalHandle = metalSessionAllocScratch(B * H * S * D)
      if sMatrix.metalHandle == nil:
        sMatrix.metalHandle = metalSessionAllocScratch(B * H * S * S)
      return metalSessionAttentionFusedEnc(q.metalHandle, k.metalHandle, v.metalHandle,
                                            o.metalHandle, sMatrix.metalHandle, B, H, S, D, scale)
    else:
      return false
  of gbCuda:
    let oldO = o.cudaHandle
    let oldS = sMatrix.cudaHandle
    let r = cuAttentionFusedR(q.cudaHandle, k.cudaHandle, v.cudaHandle, B, H, S, D, scale)
    o.cudaHandle = r.o
    sMatrix.cudaHandle = r.s_matrix
    if oldO.dptr != 0: (var oldVar = oldO; cuFree(oldVar))
    if oldS.dptr != 0: (var oldVar = oldS; cuFree(oldVar))
    return true
  of gbOpenCL:
    if o.openclHandle == nil:
      o.openclHandle = openclSessionAllocScratch(B * H * S * D)
    if sMatrix.openclHandle == nil:
      sMatrix.openclHandle = openclSessionAllocScratch(B * H * S * S)
    return openclSessionAttentionFusedEnc(cl_mem(q.openclHandle), cl_mem(k.openclHandle), cl_mem(v.openclHandle),
                                           cl_mem(o.openclHandle), cl_mem(sMatrix.openclHandle), B, H, S, D, scale)
  else:
    return false

proc sessionVecOp*(backend: GpuBackend, op: string, a, b: SessionHandle, c: var SessionHandle, n: int): bool =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      let opCode =
        case op
        of "add": 0
        of "sub": 1
        of "mul": 2
        of "div": 3
        else: 0
      # SỬA BUG: xem giải thích chi tiết ở sessionMatmul.
      if c.metalHandle == nil:
        c.metalHandle = metalSessionAllocScratch(n)
      return metalSessionVecOpEnc(opCode.cint, a.metalHandle, b.metalHandle, c.metalHandle, n.cint) != 0
    else:
      return false
  of gbCuda:
    let old = c.cudaHandle
    c.cudaHandle = cudaVecOpR(op, a.cudaHandle, b.cudaHandle)
    if old.dptr != 0: (var oldVar = old; cuFree(oldVar))
    return true
  of gbOpenCL:
    if c.openclHandle == nil:
      c.openclHandle = openclSessionAllocScratch(n)
    return openclSessionVecOpEnc(op, cl_mem(a.openclHandle), cl_mem(b.openclHandle), cl_mem(c.openclHandle), n)
  else:
    return false

proc sessionRope*(backend: GpuBackend, x: SessionHandle, cosT, sinT: SessionHandle,
                   T, Hn, D: int): bool =
  ## RoPE rotate_half in-place, encode vào session đang mở. CHỈ có cài đặt
  ## thật trên Metal hiện tại (xem metal_rope_enc trong metal_shim.m). Trên
  ## CUDA/OpenCL trả về false có chủ đích - caller (fusedNormQKVRoPE) PHẢI
  ## coi false là "chưa hỗ trợ, rơi về đường CPU cũ", KHÔNG được coi false là
  ## đã chạy xong, để tránh sinh token rác do RoPE bị bỏ qua trên các
  ## backend chưa implement kernel này.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionRopeEnc(x.metalHandle, cosT.metalHandle, sinT.metalHandle,
                                  T.cint, Hn.cint, D.cint) != 0
    else:
      return false
  of gbOpenCL:
    return openclSessionRopeEnc(cl_mem(x.openclHandle), cl_mem(cosT.openclHandle),
                                 cl_mem(sinT.openclHandle), T, Hn, D)
  of gbCuda:
    # Kernel thật: rope_kernel.cu -> nvcc -ptx -> nối vào kernels/vecop.ptx.
    # cudaRopeR (cuda_driver.nim) gọi getFn("rope_kernel"); nếu module PTX
    # đang nạp trong process CHƯA có entry này (chưa build/nối lại .ptx),
    # getFn raise CatchableError - bắt ở đây và trả về false, GIỮ NGUYÊN
    # đúng hợp đồng cũ của hàm này: false = "chưa hỗ trợ, rơi về CPU",
    # không bao giờ để lọt ngoại lệ ra ngoài khiến caller tưởng đã chạy xong.
    try:
      cuRopeR(x.cudaHandle, cosT.cudaHandle, sinT.cudaHandle, T, Hn, D)
      return true
    except CatchableError:
      return false
  else:
    return false

proc sessionCopyInto*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
                       dstOffsetElems, n: int): bool =
  ## dst[dstOffsetElems ..< +n] = src[0 ..< n], device-to-device, encode vào
  ## session đang mở - dùng để ghép GPU-side nhiều handle con thành 1 buffer
  ## liên tục (vd nối outHeads từng KV-head lại thành [H*D] để feed thẳng
  ## vào oProj matmul mà không cần sessionRead()+sessionUpload() ở giữa).
  ## dst PHẢI đã cấp đủ chỗ từ trước (sessionAllocScratch) - hàm này chỉ ghi
  ## vào 1 vùng con của buffer đã có, không tự resize/alloc.
  ## Có cài đặt thật trên cả 3 backend (Metal: copy_into_kernel; CUDA:
  ## cuMemcpyDtoDAsync thẳng, không cần kernel; OpenCL: clEnqueueCopyBuffer
  ## thẳng) - KHÔNG có nhánh "chưa hỗ trợ, rơi về CPU" nào ở đây vì cả 3 đều
  ## có primitive tương ứng sẵn trong driver/runtime, khác sessionRope
  ## (cần 1 kernel elementwise thật mới làm được rotate_half).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionCopyIntoEnc(src.metalHandle, dst.metalHandle, dstOffsetElems, n)
    else:
      return false
  of gbCuda:
    try:
      cuCopyIntoR(src.cudaHandle, dst.cudaHandle, dstOffsetElems, n)
      return true
    except CatchableError:
      return false
  of gbOpenCL:
    return openclSessionCopyIntoEnc(cl_mem(src.openclHandle), cl_mem(dst.openclHandle), dstOffsetElems, n)
  else:
    return false

proc sessionActivation*(backend: GpuBackend, op: string, x: SessionHandle, y: var SessionHandle, n: int): bool =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      let opCode =
        case op
        of "relu": 0
        of "sigmoid": 1
        of "tanh": 2
        of "apflu": 3
        else: 0
      # SỬA BUG: xem giải thích chi tiết ở sessionMatmul.
      if y.metalHandle == nil:
        y.metalHandle = metalSessionAllocScratch(n)
      return metalSessionActivationEnc(opCode.cint, x.metalHandle, y.metalHandle, n.cint) != 0
    else:
      return false
  of gbCuda:
    let old = y.cudaHandle
    y.cudaHandle = cudaActivationR(op, x.cudaHandle)
    if old.dptr != 0: (var oldVar = old; cuFree(oldVar))
    return true
  of gbOpenCL:
    if y.openclHandle == nil:
      y.openclHandle = openclSessionAllocScratch(n)
    return openclSessionActivationEnc(op, cl_mem(x.openclHandle), cl_mem(y.openclHandle), n)
  else:
    return false

# ─────────────────────────────────────────────────────────────
# KẾT THÚC SESSION - COMMIT + WAIT
# ─────────────────────────────────────────────────────────────

proc sessionEnd*(backend: GpuBackend): bool =
  ## Kết thúc session - commit tất cả ops và wait
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionEnd()
    else:
      return false
  of gbCuda:
    # CUDA: sync stream
    return cuStreamSynchronize(getStream()) == 0
  of gbOpenCL:
    return openclSessionEnd()
  else:
    return false

proc sessionRead*(backend: GpuBackend, h: SessionHandle, n: int): seq[float32] =
  ## Đọc dữ liệu từ GPU về CPU sau khi sessionEnd
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionRead(h.metalHandle, n)
    else:
      return @[]
  of gbCuda:
    return cuDownload(h.cudaHandle)
  of gbOpenCL:
    return openclSessionRead(cl_mem(h.openclHandle), n)
  else:
    return @[]

proc sessionFree*(backend: GpuBackend, h: var SessionHandle) =
  ## Giải phóng buffer trong session
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      metalSessionFree(h.metalHandle)
    else:
      discard
  of gbCuda:
    cuFree(h.cudaHandle)
  of gbOpenCL:
    var ob = cl_mem(h.openclHandle)
    openclSessionFree(ob)
    h.openclHandle = ob
  else:
    discard
  h.kind = gbCpu
  h.metalHandle = nil
  h.cudaHandle = CudaResidentTensor(dptr: 0, bytes: 0, numel: 0)

# ─────────────────────────────────────────────────────────────
# CUDA HELPER - cấp scratch buffer
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# SỬ DỤNG: Linear++ Attention với Session API CHUNG
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# RESIDENT WEIGHT (Q4) - upload weight MỘT LẦN lúc load model, giữ nguyên
# trên GPU suốt vòng đời process. Mỗi bước forward chỉ upload activation
# (nhỏ, đổi mỗi token) qua SessionHandle bình thường rồi gọi sessionMatmulQ4
# với handle weight đã resident sẵn - KHÔNG đụng lại vào bytes weight.
# API CHUNG cho cả 3 backend GPU thật (Metal/OpenCL có kernel int4 gốc chạy
# thẳng trên bytes đã pack; CUDA CHƯA có nguồn .cu cho kernel int4 (chỉ có
# vecop.ptx nhị phân sẵn, không có .cu đi kèm để sửa/build lại - xem
# cpuDequantMatmulQ4Fallback ở trên) nên dequant MỘT LẦN duy nhất lúc upload
# rồi giữ resident dạng fp32 qua sessionUpload thường - vẫn tránh được việc
# dequant+upload lại mỗi token, chỉ khác không tiết kiệm được băng thông
# 4-bit như Metal/OpenCL).
# ─────────────────────────────────────────────────────────────
type
  ResidentWeightHandle* = object
    kind*: GpuBackend
    k*, n*, groupSize*, nGroupsPerRow*: int
    metalWq*, metalScales*, metalZeros*: MetalBufferHandle
    openclWq*, openclScales*, openclZeros*: pointer
    cudaWq*, cudaScales*, cudaZeros*: CudaTensor  ## THÊM MỚI: CUDA giờ có
      ## kernel int4 gốc (matmul_q4_kernel.cu) - không cần dequant-giữ-fp32
      ## nữa. Yêu cầu build lại vecop.ptx với entry "matmul_q4_kernel" (xem
      ## hướng dẫn trong file .cu). Nếu module PTX đang chạy CHƯA có entry
      ## này, cudaMatmulQ4R sẽ raise -> catch ở gpuMatmulQ4/sessionMatmulQ4
      ## và tự rơi về fallbackFp32/cpuDequantMatmulQ4Fallback bên dưới.
    fallbackFp32*: SessionHandle    ## CPU-fallback: weight dequant 1 lần,
                                     ## resident dạng fp32 qua session chung.
    cudaQ4Ready*: bool  ## true nếu cudaWq/cudaScales/cudaZeros đã upload
                         ## thành công (tách khỏi fallbackFp32 để biết dùng
                         ## đường nào ở sessionMatmulQ4/gpuMatmulQ4)

proc residentUploadQ4*(backend: GpuBackend, wq: seq[uint8], scales, zeros: seq[float32],
                        k, n, groupSize, nGroupsPerRow: int): ResidentWeightHandle =
  ## Upload weight int4 (đã pack) lên GPU MỘT LẦN. Gọi lúc load model
  ## (loadLlamaWeightsL), giữ result lại, KHÔNG gọi lại mỗi token.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  result.kind = chosen
  result.k = k; result.n = n
  result.groupSize = groupSize; result.nGroupsPerRow = nGroupsPerRow

  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA: Private storage (không phải Shared) - wq/scales/zeros CHỈ được
      # GPU đọc lại mỗi token sau lúc này, CPU không bao giờ đụng lại. Tốn
      # thêm 1 lần blit qua staging buffer CHỈ ở đây (lúc load model), không
      # lặp lại mỗi token.
      result.metalWq = metalUploadBytesPrivate(wq)
      result.metalScales = metalUploadPrivate(scales)
      result.metalZeros = metalUploadPrivate(zeros)
    else:
      discard
  of gbOpenCL:
    result.openclWq = openclSessionUploadBytes(wq)
    result.openclScales = openclSessionUploadPrivate(scales)
    result.openclZeros = openclSessionUploadPrivate(zeros)
  of gbCuda:
    # THÊM MỚI: thử kernel int4 gốc trước (matmul_q4_kernel.cu) - chỉ cần
    # upload bytes thô + scale/zero, KHÔNG dequant CPU, KHÔNG giữ bản fp32
    # nặng gấp 8 lần. Bọc try/except vì getFn("matmul_q4_kernel") sẽ raise
    # nếu vecop.ptx đang chạy CHƯA được build lại với entry này (repo vốn
    # chỉ có vecop.ptx nhị phân sẵn, xem hướng dẫn nvcc trong file .cu).
    try:
      result.cudaWq = cudaDrv.uploadBytesAsync(wq)
      result.cudaScales = cudaDrv.uploadAsync(scales)
      result.cudaZeros = cudaDrv.uploadAsync(zeros)
      discard cudaDrv.cuStreamSynchronize(cudaDrv.getStream())
      result.cudaQ4Ready = true
    except CatchableError:
      result.cudaQ4Ready = false
      stderr.write("[cuda] matmul_q4_kernel chưa có trong vecop.ptx đang chạy - " &
        "fallback CPU-dequant-giữ-fp32 (xem kernels/matmul_q4_kernel.cu để build lại)\n")
    if not result.cudaQ4Ready:
      # ĐƯỜNG CŨ: dequant MỘT LẦN duy nhất rồi upload resident dạng fp32.
      let bytesPerRow = (k + 1) div 2
      var wT = newSeq[float32](k * n)
      var lut: array[16, float32]
      for row in 0 ..< n:
        let rowByteOff = row * bytesPerRow
        let rowGroupOff = row * nGroupsPerRow
        var g = -1
        for c in 0 ..< k:
          let gNew = if groupSize > 0: c div groupSize else: 0
          if gNew != g:
            g = gNew
            let sc = scales[rowGroupOff + g]
            let zp = zeros[rowGroupOff + g]
            for kk in 0 ..< 16: lut[kk] = (float32(kk) - zp) * sc
          let byteIdx = rowByteOff + c div 2
          let nibble = if c mod 2 == 0: wq[byteIdx] and 0x0F'u8 else: (wq[byteIdx] shr 4) and 0x0F'u8
          wT[c * n + row] = lut[int(nibble)]
      result.fallbackFp32 = sessionUpload(chosen, wT)
  else:
    # CPU: dequant MỘT LẦN duy nhất (không phải mỗi token) rồi upload
    # resident dạng fp32 [K, N] transposed, dùng chung sessionUpload/sessionMatmul.
    let bytesPerRow = (k + 1) div 2
    var wT = newSeq[float32](k * n)
    var lut: array[16, float32]
    for row in 0 ..< n:
      let rowByteOff = row * bytesPerRow
      let rowGroupOff = row * nGroupsPerRow
      var g = -1
      for c in 0 ..< k:
        let gNew = if groupSize > 0: c div groupSize else: 0
        if gNew != g:
          g = gNew
          let sc = scales[rowGroupOff + g]
          let zp = zeros[rowGroupOff + g]
          for kk in 0 ..< 16: lut[kk] = (float32(kk) - zp) * sc
        let byteIdx = rowByteOff + c div 2
        let nibble = if c mod 2 == 0: wq[byteIdx] and 0x0F'u8 else: (wq[byteIdx] shr 4) and 0x0F'u8
        wT[c * n + row] = lut[int(nibble)]
    result.fallbackFp32 = sessionUpload(chosen, wT)

proc sessionMatmulQ4*(backend: GpuBackend, a: SessionHandle, w: ResidentWeightHandle,
                       c: var SessionHandle, M: int): bool =
  ## Encode matmul C(M x N) = A(M x K) @ Wq(K x N) vào session đang mở, dùng
  ## weight resident đã upload qua residentUploadQ4. K/N/groupSize lấy từ w.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbTsic: chosen = resolveTsicBackend()

  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG NGHIÊM TRỌNG (nguồn gốc token rác): c CHƯA được cấp buffer
      # trước khi dùng làm output - giống hệt bug đã fix ở sessionMatmul (xem
      # comment ở đó) nhưng hàm NÀY (dùng cho MỌI Q/K/V/O/gate/up/down
      # projection của model Q4-per-group) lại thiếu đúng bước đó. Caller gọi
      # kiểu sessionMatmulQ4(be, x, w, hOut, M) với hOut là SessionHandle mặc
      # định (metalHandle=nil) - kernel ghi vào buffer không hợp lệ -> logits
      # sai hoàn toàn nhưng KHÔNG crash rõ ràng -> đúng khớp triệu chứng token
      # rác. Model int8/fp32 KHÔNG đi qua hàm này (dùng sessionMatmul đã fix
      # từ trước) nên không bị ảnh hưởng bởi bug này.
      if c.metalHandle == nil:
        c.metalHandle = metalSessionAllocScratch(M * w.n)
      return metalSessionMatmulQ4Enc(a.metalHandle, w.metalWq, w.metalScales, w.metalZeros,
                                      c.metalHandle, M, w.k, w.n, w.groupSize, w.nGroupsPerRow)
    else:
      return false
  of gbOpenCL:
    # SỬA BUG - cùng lỗi thiếu cấp phát như nhánh gbMetal ở trên.
    if c.openclHandle == nil:
      c.openclHandle = openclSessionAllocScratch(M * w.n)
    return openclSessionMatmulQ4Enc(cl_mem(a.openclHandle), cl_mem(w.openclWq),
                                     cl_mem(w.openclScales), cl_mem(w.openclZeros),
                                     cl_mem(c.openclHandle), M, w.k, w.n, w.groupSize, w.nGroupsPerRow)
  of gbCuda:
    if w.cudaQ4Ready:
      try:
        let old = c.cudaHandle
        c.cudaHandle = cudaDrv.cudaMatmulQ4R(a.cudaHandle, w.cudaWq, w.cudaScales, w.cudaZeros,
                                              M, w.k, w.n, w.groupSize, w.nGroupsPerRow)
        if old.dptr != 0: (var oldVar = old; cudaDrv.freeResident(oldVar))
        return true
      except CatchableError:
        discard  # rơi xuống fallbackFp32 bên dưới, KHÔNG sập process
    return sessionMatmul(chosen, a, w.fallbackFp32, c, M, w.k, w.n)
  else:
    # CPU: dùng lại sessionMatmul chung trên bản fp32 đã resident.
    return sessionMatmul(chosen, a, w.fallbackFp32, c, M, w.k, w.n)

proc residentFreeQ4*(w: var ResidentWeightHandle) =
  ## Giải phóng weight resident - chỉ gọi lúc unload model/thoát process,
  ## KHÔNG gọi giữa các token.
  case w.kind
  of gbMetal:
    when defined(macosx):
      metalSessionFree(w.metalWq)
      metalSessionFree(w.metalScales)
      metalSessionFree(w.metalZeros)
  of gbOpenCL:
    var wq = cl_mem(w.openclWq); var sc = cl_mem(w.openclScales); var zp = cl_mem(w.openclZeros)
    openclSessionFree(wq); openclSessionFree(sc); openclSessionFree(zp)
  of gbCuda:
    if w.cudaQ4Ready:
      var wq = w.cudaWq; var sc = w.cudaScales; var zp = w.cudaZeros
      cudaDrv.freeResident(wq); cudaDrv.freeResident(sc); cudaDrv.freeResident(zp)
    else:
      var f = w.fallbackFp32
      sessionFree(w.kind, f)
  else:
    var f = w.fallbackFp32
    sessionFree(w.kind, f)
    w.fallbackFp32 = f

proc linearPlusAttentionSession*(backend: GpuBackend, q, k, v: seq[float32],
                                  B, H, T, D: int, scale: float32): seq[float32] =
  ## Linear++ Attention dùng Session API chung - chạy trên Metal/CUDA/OpenCL
  let qkvLen = B * H * T * D
  result = newSeq[float32](qkvLen)

  if not sessionBegin(backend):
    echo "Session begin failed,stop"
    return

  # Upload dữ liệu
  var hQ = sessionUpload(backend, q)
  var hK = sessionUpload(backend, k)
  var hV = sessionUpload(backend, v)
  var hKV = sessionAllocScratch(backend, qkvLen)
  var hO = sessionAllocScratch(backend, qkvLen)

  # Encode các ops vào session
  # Bước 1: K^T @ V (tạm thời dùng matmul)
  # TODO: cần transpose K và V trước
  # Bước 2: Q @ KV
  # Bước 3: scale + norm

  # Kết thúc session
  if not sessionEnd(backend):
    echo "Session failed,stop"
    return

  # Đọc kết quả
  result = sessionRead(backend, hO, qkvLen)

  # Giải phóng buffer
  sessionFree(backend, hQ); sessionFree(backend, hK)
  sessionFree(backend, hV); sessionFree(backend, hKV)
  sessionFree(backend, hO)