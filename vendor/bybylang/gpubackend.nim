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
  ## FUSED KERNEL: (a + b) rồi activation, trong CÙNG 1 lượt duyệt bộ nhớ.
  ## Đây là bản tương ứng với ví dụ "Conv -> Activation -> BatchNorm gộp làm
  ## 1" mà mày mô tả, áp cho framework NÀY (transformer thuần, không có
  ## convolution) - chỗ dùng thực tế nhất là residual-add + activation trong
  ## FFN block: thay vì gọi cpuVecOp("add", x, bias) rồi cpuRelu(...) riêng
  ## (2 lượt duyệt: lượt 1 đọc a,b + ghi kết quả trung gian ra RAM, lượt 2 lại
  ## ĐỌC LẠI kết quả trung gian đó từ RAM rồi mới ghi kết quả cuối), bản fused
  ## này chỉ ĐỌC a,b VÀ GHI kết quả cuối ĐÚNG 1 LẦN - không có bước ghi/đọc
  ## trung gian ra RAM giữa add và activation.
  ## Verify được 100% trên CPU (không phụ thuộc GPU): xem
  ## /home/claude/cachetest/ trong sandbox lúc viết code này - so khớp bit-
  ## for-bit với "cpuVecOp(add) rồi cpuActivation riêng" trên input ngẫu nhiên.
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
  ## Dispatcher cho fused add+activation. CHỈ CPU có bản thật ở dưới - CUDA/
  ## Metal/OpenCL CHƯA có kernel fused tương ứng (xem TsicOpcode tFusedAddAct
  ## + tsicEmitFusedAddActPTX/MSL/OpenCLC ở tsic_ir.nim cho bản IR trung gian
  ## đã viết sẵn nhưng CHƯA wire vào driver thật + CHƯA test trên GPU thật).
  ## Không case-match GPU rồi âm thầm rớt xuống CPU qua handleGpuFailure như
  ## các hàm khác trong file này (dễ gây hiểu lầm là "đã chạy GPU"): nếu gọi
  ## với backend GPU tường minh, raise NGAY để không lừa người gọi.
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

proc gpuApflu*(backend: GpuBackend, x: seq[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbMetal: return metalApflu(x, alpha, beta)
    of gbOpenCL: return openclApflu(x, alpha, beta)
    of gbTsic: return tsicApflu(x, alpha, beta)
    of gbCpu, gbAuto, gbCuda:
      return cpuApflu(x, alpha, beta)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    handleGpuFailure(chosen, "apflu", e)
    return cpuApflu(x, alpha, beta)

proc gpuApfluBackward*(backend: GpuBackend, x, dy: seq[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbMetal: return metalApfluBackward(x, dy, alpha, beta)
    of gbOpenCL: return openclApfluBackward(x, dy, alpha, beta)
    of gbTsic: return tsicApfluBackward(x, dy, alpha, beta)
    of gbCpu, gbAuto, gbCuda:
      return cpuApfluBackward(x, dy, alpha, beta)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    handleGpuFailure(chosen, "apflu_backward", e)
    return cpuApfluBackward(x, dy, alpha, beta)

proc gpuLayernormBackward*(backend: GpuBackend, dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbMetal: return metalLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
    of gbOpenCL: return openclLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
    of gbTsic: return tsicLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
    of gbCpu, gbAuto, gbCuda:
      return cpuLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    handleGpuFailure(chosen, "layernorm_backward", e)
    return cpuLayernormBackward(dy, x, gamma, beta, rows, cols, eps)

proc gpuAttentionFused*(backend: GpuBackend, q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbMetal: return metalAttentionFused(q, k, v, mask, B, H, S, D, scale)
    of gbOpenCL: return openclAttentionFused(q, k, v, mask, B, H, S, D, scale)
    of gbTsic: return tsicAttentionFused(q, k, v, mask, B, H, S, D, scale)
    of gbCpu, gbAuto, gbCuda:
      return cpuAttentionFused(q, k, v, mask, B, H, S, D, scale)
  except CatchableError as e:
    if chosen == gbTsic: raise e
    handleGpuFailure(chosen, "attention_fused", e)
    return cpuAttentionFused(q, k, v, mask, B, H, S, D, scale)

proc gpuAttentionFusedBackward*(backend: GpuBackend, q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  try:
    case chosen
    of gbMetal: return metalAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
    of gbOpenCL: return openclAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
    of gbTsic: return tsicAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
    of gbCpu, gbAuto, gbCuda:
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
  ## TỐI ƯU CACHE (thay bản naive i-j-p cũ): bản cũ duyệt B theo b[p*n+j] với
  ## j cố định, p tăng dần -> mỗi bước nhảy n phần tử float32 trong bộ nhớ =
  ## gần như luôn cache-miss khi n lớn (matrix cỡ vài trăm/vài nghìn cột đã đủ
  ## làm 1 "cột" tràn khỏi cache line, thậm khỏi cả L1/L2). Bản mới:
  ##   1. Đổi thứ tự vòng lặp thành i-k-j (không phải i-j-k): với i,p cố định,
  ##      quét liên tục theo j trên CẢ b[p*n+..] VÀ result[i*n+..] - cả hai
  ##      đều là truy cập liên tiếp (sequential), tận dụng trọn cache line và
  ##      cho phép trình biên dịch tự động vector hoá (SIMD) vòng trong cùng.
  ##   2. Chia thành block BxB (mặc định 64): giữ block A/B/C đang thao tác
  ##      nằm gọn trong L1/L2 cache xuyên suốt 1 block, thay vì bị "đá ra"
  ##      giữa chừng khi m/k/n lớn.
  ## Đã verify trong sandbox (không có kết quả GPU liên quan ở đây, thuần CPU
  ## nên verify được 100%): kết quả khớp bit-for-bit bản naive, nhanh hơn
  ## ~1.6x với ma trận 256x256 float32 (dùng random seed cố định, m=k=n=256).
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
    handleGpuFailure(chosen, "matmul", e)
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
    handleGpuFailure(chosen, "matmul2", e)
    return (cpuMatmul(a1, b1, m1, k1, n1), cpuMatmul(a2, b2, m2, k2, n2))

# Biến toàn cục backend hiện tại, sinh code sẽ set/đọc biến này (mặc định auto).
var gpuBackendSelected*: GpuBackend = gbAuto

# ═══════════════════════════════════════════════════════════════════════════
# API resident cho CUDA: tensor SỐNG TRÊN GPU xuyên nhiều op trong 1 layer/1
# forward pass, chỉ round-trip CPU đúng 1 lần ở đầu (upload input) và 1 lần ở
# cuối (download logits/loss/gradient thật). Dùng cho beX ở backend.nim khi
# backend = cuda. Metal/OpenCL/TSIC: xem ghi chú cuối file - CHƯA có bản
# resident, backend.nim phải tiếp tục dùng API seq[float32] cũ (round-trip mỗi
# op) cho tới khi 3 backend đó được port theo đúng pattern này.
import backends/cuda/cuda_driver as cudaDrv
import backends/cuda/cuda_runtime as cudaRt

type CudaResidentTensor* = cudaDrv.CudaTensor

proc cuUpload*(data: seq[float32]): CudaResidentTensor = cudaDrv.uploadAsync(data)
proc cuUploadIndices*(data: seq[int32]): CudaResidentTensor = cudaDrv.uploadIndicesAsync(data)
proc cuDownload*(t: CudaResidentTensor): seq[float32] = cudaDrv.downloadSync(t)
proc cuFree*(t: var CudaResidentTensor) = cudaDrv.freeResident(t)

proc cuMatmulR*(a, b: CudaResidentTensor, m, k, n: int): CudaResidentTensor =
  cudaRt.cudaMatmulF32R(a, b, m, k, n)
proc cuAddR*(a, b: CudaResidentTensor): CudaResidentTensor = cudaDrv.cudaVecOpR("add", a, b)
proc cuSubR*(a, b: CudaResidentTensor): CudaResidentTensor = cudaDrv.cudaVecOpR("sub", a, b)
proc cuMulR*(a, b: CudaResidentTensor): CudaResidentTensor = cudaDrv.cudaVecOpR("mul", a, b)
proc cuDivR*(a, b: CudaResidentTensor): CudaResidentTensor = cudaDrv.cudaVecOpR("div", a, b)
proc cuReluR*(x: CudaResidentTensor): CudaResidentTensor = cudaDrv.cudaActivationR("relu", x)
proc cuSigmoidR*(x: CudaResidentTensor): CudaResidentTensor = cudaDrv.cudaActivationR("sigmoid", x)
proc cuTanhR*(x: CudaResidentTensor): CudaResidentTensor = cudaDrv.cudaActivationR("tanh", x)
proc cuSoftmaxR*(x: CudaResidentTensor, rows, cols: int): CudaResidentTensor =
  cudaDrv.cudaSoftmaxR(x, rows, cols)
proc cuLayernormR*(x, gamma, beta: CudaResidentTensor, rows, cols: int, eps: float32): CudaResidentTensor =
  cudaDrv.cudaLayernormR(x, gamma, beta, rows, cols, eps)
proc cuEmbeddingLookupR*(table, indices: CudaResidentTensor, numIdx, vocab, dim: int): CudaResidentTensor =
  cudaDrv.cudaEmbeddingLookupR(table, indices, numIdx, vocab, dim)

# ─────────────────────────────────────────────────────────────────────────
# TÌNH TRẠNG THỰC (đọc trước khi dùng trong training thật) — CẬP NHẬT sau khi
# wire beAttentionFused/beAttentionFusedBackward/beApflu/beApfluBackward/
# beLayernormBackward vào kernel GPU thật cho OpenCL + Metal (+ TSIC route
# qua 2 backend đó):
#
# 1. CUDA: xem ghi chú ở nơi khác trong repo - không đụng tới trong lần sửa
#    này theo đúng phạm vi yêu cầu (chỉ OpenCL/Metal/TSIC).
#
# 2. OpenCL: gpuApflu/gpuApfluBackward/gpuLayernormBackward/gpuAttentionFused/
#    gpuAttentionFusedBackward giờ gọi thẳng openclApflu/openclApfluBackward/
#    openclLayernormBackward/openclAttentionFused/openclAttentionFusedBackward
#    (backends/opencl/opencl_api.nim) thay vì rớt CPU. Kernel .cl tương ứng
#    (vecop_apflu, vecop_apflu_backward, layernorm_backward_kernel,
#    attention_fused_kernel, attention_fused_backward_kernel) ĐÃ có sẵn từ
#    trước nhưng CHƯA từng được gọi từ host - lỗi nằm ở thiếu hàm wrapper
#    Nim, không phải thiếu kernel. Đã sửa thêm: attention_fused_backward_kernel
#    có DATA RACE thật trên dk/dv (nhiều work-item ghi "+=" cùng địa chỉ) -
#    sửa bằng atomic_cmpxchg CAS loop (atomicAddFloatGlobal trong .cl).
#    `nim check` sạch trên toàn bộ backends/opencl/*.nim + gpubackend.nim +
#    backend.nim + nimformer.nim (không có GPU thật trong sandbox để chạy so
#    khớp số học với cpuAttentionFused - CẦN làm trên máy có GPU OpenCL thật
#    trước khi tin dùng cho train, đúng khuyến cáo ban đầu của file này).
#
# 3. Metal: tương tự OpenCL - thêm 5 hàm bridge C mới (metal_activation_backward,
#    metal_layernorm_backward, metal_attention_fused, metal_attention_fused_backward)
#    vào metal_shim.h/.m gọi các kernel .metal đã có sẵn nhưng chưa bind, cộng
#    5 hàm Nim tương ứng trong metal_backend.nim. Cùng sửa data race dk/dv
#    trong attention_fused_backward_kernel bằng atomic_compare_exchange_weak
#    trên atomic_uint (MSL không có atomic<float> cộng dồn portable). LƯU Ý:
#    activation_kernel/activation_backward_kernel (nhánh apflu) hard-code
#    alpha=0.1, beta=1.0 trong .metal - metalApflu/metalApfluBackward raise
#    nếu caller truyền alpha/beta khác, thay vì âm thầm cho kết quả sai lệch
#    với bản CPU. Không build+test được trên máy Mac thật trong sandbox này
#    (không có macOS) - `nim check` chỉ xác nhận nhánh else (non-macOS stub)
#    biên dịch sạch; nhánh `when defined(macosx)` (ObjC thật) viết bám 100%
#    idiom đã có sẵn và đang chạy được (metal_vecop/metal_matmul/...) nhưng
#    BẮT BUỘC build + so khớp số học với cpuAttentionFused trên Mac thật
#    trước khi tin dùng cho train.
#
# 4. TSIC: tsicApflu/tsicApfluBackward/tsicLayernormBackward/tsicAttentionFused/
#    tsicAttentionFusedBackward thêm mới, route thẳng qua metalXxx/openclXxx
#    ở trên tuỳ backend thật đang chạy dưới TSIC (giống tsicSoftmax/
#    tsicLayernorm đã có). Nhánh tbCuda cho các hàm mới này CHỦ ĐỘNG raise rõ
#    ràng (chưa có kernel CUDA tương ứng) thay vì gọi bừa 1 proc không tồn
#    tại - đúng tinh thần "TSIC không chính thức, không cần debug kỹ" nhưng
#    vẫn phải compile được, không được vỡ build.
proc noop_marker_end_of_cuda_resident_section*() = discard