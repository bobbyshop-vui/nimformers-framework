## generate.nim
## File DUY NHẤT: nimformer_llama (RoPE + RMSNorm + SwiGLU + GQA + KV-cache paged)
## + tokenizer + vòng lặp sinh văn bản, gộp chung 1 file theo yêu cầu.

import std/[math, random, sequtils, tables, algorithm, os, strformat, times]
import nimformer  # Tensor, Linear, newLinear, flatten2D, transpose, Embedding, addT...
import backend
import customfloat
import quant       # loadQuantStateDict, dequantizeTensor, QuantTensor
import nimpy

# ===================================================================
# Helper GHÉP SESSION cross-submodule: trước đây MỖI submodule (RMSNorm,
# RoPE, FFN, attention-decode) tự mở/đóng session RIÊNG của nó
# (sessionBegin...sessionEnd), nghĩa là 1 layer = 4-6 lần commit+wait GPU
# (Metal: commandBuffer commit+waitUntilCompleted; CUDA: cuStreamSynchronize)
# dù giữa các submodule đó KHÔNG có phụ thuộc CPU thực sự bắt buộc nào
# (residual-add, RMSNorm-scale, các matmul gate/up/down đều thuần GPU nối
# tiếp nhau). Các proc dưới đây cho phép ENCODE nhiều lệnh vào 1 session ĐANG
# MỞ mà không cần đọc kết quả về CPU giữa chừng (kết quả ở lại dạng
# SessionHandle, chỉ sessionRead() đúng 1 lần ở cuối) - đây là cách "1 session
# cross-submodule" được thực hiện thật, không phải chỉ đổi tên hàm.
proc sMatmul(be: GpuBackend, a: SessionHandle, M, K: int, b: SessionHandle, N: int): SessionHandle =
  result = sessionAllocScratch(be, M * N)
  doAssert sessionMatmul(be, a, b, result, M, K, N),
    "sMatmul: sessionMatmul thất bại giữa session gộp (backend hiện tại có thể không hỗ trợ)"

proc sVecOp(be: GpuBackend, op: string, a, b: SessionHandle, n: int): SessionHandle =
  result = sessionAllocScratch(be, n)
  doAssert sessionVecOp(be, op, a, b, result, n),
    "sVecOp(" & op & "): sessionVecOp thất bại giữa session gộp"

proc sActivation(be: GpuBackend, op: string, a: SessionHandle, n: int): SessionHandle =
  result = sessionAllocScratch(be, n)
  doAssert sessionActivation(be, op, a, result, n),
    "sActivation(" & op & "): sessionActivation thất bại giữa session gộp"

proc broadcastBias(bias: Tensor, rows, outF: int): seq[float32] =
  ## Trải bias [outF] ra full [rows*outF] để cộng bằng 1 sessionVecOp("add")
  ## thay vì vòng lặp CPU "i mod outF" sau khi đọc kết quả về (điều đó sẽ ép
  ## phải kết thúc session sớm). Trả về seq rỗng nếu không có bias thật.
  if bias.data.len != outF: return @[]
  result = newSeq[float32](rows * outF)
  for r in 0 ..< rows:
    copyMem(addr result[r * outF], unsafeAddr bias.data[0], outF * sizeof(float32))

proc dequantTransposedW(l: Linear): seq[float32] =
  ## Weight ĐÃ transpose [inF, outF] flat row-major, sẵn sàng nạp thẳng vào
  ## session bằng sessionUpload() + sMatmul(), dùng chung cho QKV-fuse và
  ## FFN-fuse bên dưới (trước đây mỗi nơi tự viết lại 1 bản).
  ## CHỈ còn dùng làm FALLBACK khi weight KHÔNG phải int4-per-group (xem
  ## linearFast bên dưới) - đường chính (model GPTQ int4 thật) không đi qua
  ## hàm này nữa.
  if l.useQuant: dequantizeTensorTransposed(l.weightQ)
  else: transpose(l.weight).data

proc linearFast(l: Linear, xData: seq[float32], rows: int, ctx: Backend): seq[float32] =
  ## SỬA (NGUYÊN NHÂN CHÍNH của "1.5 phút/token"): TRƯỚC ĐÂY mọi nơi trong
  ## file này (fusedNormQKVRoPE, fusedResidualNormFFN, forwardQKVFused) đều
  ## gọi dequantTransposedW() -> giải nén TOÀN BỘ ma trận trọng số int4 ra
  ## fp32 trên CPU (vd 4096x4096 hoặc 4096x11008 mỗi ma trận) RỒI upload
  ## nguyên mảng fp32 đó lên GPU qua sessionUpload - lặp lại việc này cho
  ## CẢ 7 ma trận (Q,K,V,O,gate,up,down) x 32 layer x MỖI TOKEN sinh ra.
  ## Đây chính là "1.5 phút/token": không phải GPU kernel chậm, mà là CPU
  ## dequant + băng thông upload fp32 khổng lồ lặp lại vô ích mỗi bước decode
  ## (weight KHÔNG đổi giữa các token, không có lý do gì phải giải nén lại).
  ##
  ## nimformer.nim's Linear.forward() đã có sẵn đường tắt đúng chuẩn
  ## llama.cpp: beMatmulQ4() chạy matmul TRỰC TIẾP trên dữ liệu int4 còn nén
  ## (Metal/OpenCL có kernel int4 gốc, đọc thẳng 4-bit -> băng thông bộ nhớ
  ## thấp hơn ~8 lần so với đọc fp32 đã giải nén). File này (nim_inference.nim)
  ## trước đây không hề gọi tới nó. Giờ mọi chỗ dùng chung hàm này thay vì
  ## dequantTransposedW + beMatmul/sMatmul.
  if l.useQuant and l.weightQ.kind == qkInt4Asymmetric and l.weightQ.groupSize > 0:
    let nGroupsPerRow = (l.inF + l.weightQ.groupSize - 1) div l.weightQ.groupSize
    result = beMatmulQ4(ctx, xData, l.weightQ.data, l.weightQ.scale, l.weightQ.zero_point,
                         rows, l.inF, l.outF, l.weightQ.groupSize, nGroupsPerRow)
  else:
    # Fallback (weight không phải int4-per-group, vd lm_head tied-embedding
    # useQuant=false): vẫn phải dequant, nhưng đây là trường hợp hiếm
    # (không nằm trong vòng lặp 32 layer x mỗi token).
    let wT = dequantTransposedW(l)
    result = beMatmul(ctx, xData, rows, l.inF, wT, l.inF, l.outF)
  let expectedSize = rows * l.outF
  if result.len != expectedSize:
    var fixed = newSeq[float32](expectedSize)
    let copyLen = min(result.len, expectedSize)
    for i in 0 ..< copyLen: fixed[i] = result[i]
    result = fixed

proc linearFastBias(l: Linear, xData: seq[float32], rows: int, ctx: Backend): seq[float32] =
  result = linearFast(l, xData, rows, ctx)
  if l.bias.data.len == l.outF:
    for r in 0 ..< rows:
      let off = r * l.outF
      for o in 0 ..< l.outF:
        result[off + o] += l.bias.data[o]


# ===================================================================
# RMSNorm
# ===================================================================
type
  RMSNormL* = object
    weight*: Tensor
    eps*: float32
    dim*: int

proc newRMSNormL*(dim: int, eps: float32 = 1e-6'f32): RMSNormL =
  result.dim = dim
  result.eps = eps
  result.weight = newTensor(@[dim], 1'f32)

proc rmsNormPrep*(rn: RMSNormL, x: Tensor, ctx: Backend): tuple[invBroadcast, wBroadcast: seq[float32], n: int] =
  ## Tách riêng phần "reduce" của RMSNorm (bình phương + sum theo hàng qua
  ## matmul-as-reduce, rồi 1/sqrt(mean+eps) trên CPU) khỏi phần "scale" (2
  ## phép mul kích thước đầy đủ). Lý do tách: invStd là số vô hướng/hàng,
  ## PHẢI đọc sumSq về CPU để chạy sqrt (session API hiện tại không có
  ## kernel reduce/sqrt) - đây là 1 điểm CPU-touch không thể tránh, xảy ra
  ## TRƯỚC bất kỳ session lớn nào. Nhưng phần "scale" (2 mul) thì KHÔNG có lý
  ## do gì phải tự đóng trong session riêng của nó - callers (fusedNormQKVRoPE,
  ## fusedResidualNormFFN) gộp thẳng 2 mul này vào session của submodule kế
  ## tiếp (QKV matmul / FFN) thay vì mở-đóng session chỉ vì 2 phép mul.
  let (rows, cols) = flatten2D(x.shape)
  doAssert cols == rn.dim
  let n = rows * cols
  let x2 = beMul(ctx, x.data, x.data)
  var ones = newSeq[float32](cols)
  for i in 0 ..< cols: ones[i] = 1'f32
  let sumSq = beMatmul(ctx, x2, rows, cols, ones, cols, 1)
  var invStd = newSeq[float32](rows)
  for r in 0 ..< rows:
    invStd[r] = 1'f32 / sqrt(sumSq[r] / float32(cols) + rn.eps)
  var invBroadcast = newSeq[float32](n)
  var wBroadcast = newSeq[float32](n)
  for r in 0 ..< rows:
    let off = r * cols
    for c in 0 ..< cols:
      invBroadcast[off + c] = invStd[r]
      wBroadcast[off + c] = rn.weight.data[c]
  result = (invBroadcast, wBroadcast, n)

proc forward*(rn: RMSNormL, x: Tensor, ctx: Backend): Tensor =
  ## SỬA: backend.nim không có primitive RMSNorm riêng, nhưng có ĐỦ add/mul
  ## (beMul/beAdd, hoặc sessionVecOp) và matmul (beMatmul) để dựng lại toàn
  ## bộ công thức trên GPU, không cần viết vòng lặp CPU thuần nữa:
  ##   - Bình phương từng phần tử: x ⊙ x -> 1 beMul (GPU).
  ##   - Tổng bình phương theo hàng (reduction mà backend không có kernel
  ##     riêng): nhân với vector cột toàn số 1 bằng beMatmul(x2, rows, cols,
  ##     ones, cols, 1) -> ra đúng sumSq [rows,1]. Đây là "matmul-as-reduce",
  ##     GPU-native, không phải giả vờ.
  ##   - 1/sqrt(mean+eps): CHỈ scalar/hàng (rows phần tử, không phải
  ##     rows*cols) nên tính trên CPU - không đáng và không thể thành 1
  ##     kernel GPU riêng chỉ vì rows con số.
  ##   - Scale lại theo invStd rồi theo weight: 2 phép mul kích thước đầy đủ
  ##     [rows,cols] -> gộp vào 1 session GPU (sessionVecOp mul, mul) để dữ
  ##     liệu ở lại GPU xuyên suốt, không upload/download giữa 2 bước.
  let (rows, cols) = flatten2D(x.shape)
  doAssert cols == rn.dim
  let n = rows * cols

  let x2 = beMul(ctx, x.data, x.data)                 # GPU: bình phương từng phần tử
  var ones = newSeq[float32](cols)
  for i in 0 ..< cols: ones[i] = 1'f32
  let sumSq = beMatmul(ctx, x2, rows, cols, ones, cols, 1)  # GPU: reduce theo hàng qua matmul

  var invStd = newSeq[float32](rows)                  # O(rows) scalar, CPU là hợp lý
  for r in 0 ..< rows:
    invStd[r] = 1'f32 / sqrt(sumSq[r] / float32(cols) + rn.eps)

  # Broadcast invStd/weight ra full [rows,cols] - đây là COPY dữ liệu (không
  # phải phép toán số học), backend không có kernel broadcast nên làm ở CPU,
  # rồi đẩy phần NHÂN thật sự (mul) lên GPU ngay sau.
  var invBroadcast = newSeq[float32](n)
  var wBroadcast = newSeq[float32](n)
  for r in 0 ..< rows:
    let off = r * cols
    for c in 0 ..< cols:
      invBroadcast[off + c] = invStd[r]
      wBroadcast[off + c] = rn.weight.data[c]

  let backend = ctx.kind.toByby()
  result.shape = x.shape
  # Session (mul, mul) gộp 2 phép nhân vào 1 encoder, dữ liệu ở lại GPU
  # xuyên suốt. Trên Metal, việc nhiều dispatch khác pipeline chia sẻ 1
  # encoder từng cho kết quả sai trên GPU không phải Apple Silicon (thiếu
  # memory barrier giữa các dispatch) - đã fix tận gốc trong metal_shim.m
  # (memoryBarrierWithScope: sau mỗi *_enc), nên giờ session an toàn trên
  # cả 3 backend, không cần né riêng Metal nữa.
  if sessionBegin(backend):
    var hX = sessionUpload(backend, x.data)
    var hInv = sessionUpload(backend, invBroadcast)
    var hW = sessionUpload(backend, wBroadcast)
    var hTmp = sessionAllocScratch(backend, n)
    var hOut = sessionAllocScratch(backend, n)
    let ok = sessionVecOp(backend, "mul", hX, hInv, hTmp, n) and
             sessionVecOp(backend, "mul", hTmp, hW, hOut, n) and
             sessionEnd(backend)
    if ok:
      result.data = sessionRead(backend, hOut, n)
      sessionFree(backend, hX); sessionFree(backend, hInv); sessionFree(backend, hW)
      sessionFree(backend, hTmp); sessionFree(backend, hOut)
      return
    sessionFree(backend, hX); sessionFree(backend, hInv); sessionFree(backend, hW)
    sessionFree(backend, hTmp); sessionFree(backend, hOut)

  # Fallback: vẫn 100% GPU, chỉ là 2 lệnh beMul riêng (không cùng 1 session)
  # nếu session thất bại giữa chừng trên backend hiện tại.
  let tmp = beMul(ctx, x.data, invBroadcast)
  result.data = beMul(ctx, tmp, wBroadcast)

# ===================================================================
# RoPE
# ===================================================================
type
  RoPEL* = object
    cosT*, sinT*: seq[float32]   # [maxSeqLen, headDim/2]
    headDim*, maxSeqLen*: int

proc toHeadMajor(data: seq[float32], T, H, D: int): seq[float32] =
  ## [T,H,D] (vị trí ngoài, head trong) -> [H,T,D] (head ngoài, vị trí trong).
  ## beAttentionFused (Metal/CUDA/OpenCL) giả định layout [B*H, S, D] — mỗi
  ## head là 1 khối liền S*D phần tử (base_idx = bh*S*D) — trong khi
  ## Linear.forward xuất ra layout [T,H,D] (vị trí ngoài). Không transpose
  ## trước khi gọi kernel -> kernel đọc lẫn head/token -> attention sai hoàn
  ## toàn, sinh token rác ngay từ prefill dù RoPE/Linear/dequant đều đúng.
  result = newSeq[float32](T * H * D)
  for t in 0 ..< T:
    for h in 0 ..< H:
      let src = (t * H + h) * D
      let dst = (h * T + t) * D
      for d in 0 ..< D:
        result[dst + d] = data[src + d]

proc toPosMajor(data: seq[float32], T, H, D: int): seq[float32] =
  ## Ngược lại: [H,T,D] (output của beAttentionFused) -> [T,H,D] (layout mà
  ## phần còn lại của code — attnOut.data rồi oProj.forward — đang mong đợi).
  result = newSeq[float32](T * H * D)
  for h in 0 ..< H:
    for t in 0 ..< T:
      let src = (h * T + t) * D
      let dst = (t * H + h) * D
      for d in 0 ..< D:
        result[dst + d] = data[src + d]

proc newRoPEL*(headDim, maxSeqLen: int, theta: float32 = 10000'f32): RoPEL =
  result.headDim = headDim
  result.maxSeqLen = maxSeqLen
  let half = headDim div 2
  var invFreq = newSeq[float32](half)
  for i in 0 ..< half:
    invFreq[i] = 1'f32 / pow(theta, float32(i) / float32(half))
  result.cosT = newSeq[float32](maxSeqLen * half)
  result.sinT = newSeq[float32](maxSeqLen * half)
  for pos in 0 ..< maxSeqLen:
    for i in 0 ..< half:
      let angle = float32(pos) * invFreq[i]
      result.cosT[pos * half + i] = cos(angle)
      result.sinT[pos * half + i] = sin(angle)

proc applyRoPEInplace*(rope: RoPEL, ctx: Backend, data: var seq[float32], B, T, H, D: int, posOffset: int) =
  ## data layout: [B, T, H, D] phẳng. posOffset = vị trí tuyệt đối của token đầu
  ## tiên trong T (dùng khi decode từng token 1, T=1 nhưng vị trí > 0).
  ##
  ## SỬA: công thức RoPE x1*cos-x2*sin / x1*sin+x2*cos chính là dạng chuẩn
  ## "rotate-half": out = x*cos + rotate_half(x)*sin, với
  ## rotate_half(x) = concat(-x2, x1). Viết lại thế này thì TOÀN BỘ phép
  ## nhân/cộng số học là elementwise trên mảng full [B,T,H,D] -> vừa khít với
  ## beMul/beAdd (hoặc sessionVecOp) sẵn có trong backend, không cần kernel
  ## RoPE riêng. Phần build cosFull/sinFull/rotHalf bên dưới CHỈ là sắp xếp
  ## lại vị trí phần tử (gather + đổi dấu), không phải phép toán nặng, nên
  ## vẫn ở CPU (backend không có kernel gather) - đúng đúng phần "GPU đủ
  ## add/mul" mà không đủ gather/broadcast.
  let half = D div 2
  let n = B * T * H * D
  var cosFull = newSeq[float32](n)
  var sinFull = newSeq[float32](n)
  var rotHalf = newSeq[float32](n)
  for b in 0 ..< B:
    for t in 0 ..< T:
      let pos = posOffset + t
      doAssert pos < rope.maxSeqLen, "RoPE: vị trí vượt quá max_seq_len đã precompute"
      for h in 0 ..< H:
        let base = ((b * T + t) * H + h) * D
        for i in 0 ..< half:
          let cosV = rope.cosT[pos * half + i]
          let sinV = rope.sinT[pos * half + i]
          cosFull[base + i]        = cosV
          cosFull[base + i + half] = cosV
          sinFull[base + i]        = sinV
          sinFull[base + i + half] = sinV
          rotHalf[base + i]        = -data[base + i + half]
          rotHalf[base + i + half] =  data[base + i]

  let backend = ctx.kind.toByby()
  if sessionBegin(backend):
    var hX = sessionUpload(backend, data)
    var hCos = sessionUpload(backend, cosFull)
    var hRot = sessionUpload(backend, rotHalf)
    var hSin = sessionUpload(backend, sinFull)
    var hT1 = sessionAllocScratch(backend, n)
    var hT2 = sessionAllocScratch(backend, n)
    var hOut = sessionAllocScratch(backend, n)
    let ok = sessionVecOp(backend, "mul", hX, hCos, hT1, n) and
             sessionVecOp(backend, "mul", hRot, hSin, hT2, n) and
             sessionVecOp(backend, "add", hT1, hT2, hOut, n) and
             sessionEnd(backend)
    if ok:
      data = sessionRead(backend, hOut, n)
      sessionFree(backend, hX); sessionFree(backend, hCos); sessionFree(backend, hRot)
      sessionFree(backend, hSin); sessionFree(backend, hT1); sessionFree(backend, hT2)
      sessionFree(backend, hOut)
      return
    sessionFree(backend, hX); sessionFree(backend, hCos); sessionFree(backend, hRot)
    sessionFree(backend, hSin); sessionFree(backend, hT1); sessionFree(backend, hT2)
    sessionFree(backend, hOut)

  # Fallback: vẫn 100% GPU (không phải vòng lặp CPU), chỉ là 2 mul + 1 add
  # KHÔNG cùng 1 session nếu backend hiện tại không hỗ trợ session.
  let t1 = beMul(ctx, data, cosFull)
  let t2 = beMul(ctx, rotHalf, sinFull)
  data = beAdd(ctx, t1, t2)

# ===================================================================
# SwiGLU FeedForward (gate_proj, up_proj, down_proj)
# ===================================================================
type
  SwiGLUL* = object
    gateProj*, upProj*, downProj*: Linear

proc newSwiGLUL*(hidden, intermediate: int): SwiGLUL =
  result.gateProj = newLinear(hidden, intermediate)
  result.upProj = newLinear(hidden, intermediate)
  result.downProj = newLinear(intermediate, hidden)

proc forward*(ff: SwiGLUL, x: Tensor, ctx: Backend): Tensor =
  let gateRaw = ff.gateProj.forward(x, ctx)
  let upRaw = ff.upProj.forward(x, ctx)
  let n = gateRaw.data.len

  # SỬA: gộp sigmoid+mul+mul vào 1 SESSION - dữ liệu ở lại GPU xuyên suốt,
  # không upload/download qua CPU giữa từng phép toán (trước đây beSigmoid,
  # rồi vòng for CPU, rồi beMul mỗi cái tự upload/download riêng - 3 lượt
  # CPU<->GPU cho có 3 phép toán liên tiếp).
  let backend = ctx.kind.toByby()
  if sessionBegin(backend):
    var hGate = sessionUpload(backend, gateRaw.data)
    var hUp = sessionUpload(backend, upRaw.data)
    var hSig = sessionAllocScratch(backend, n)
    var hSilu = sessionAllocScratch(backend, n)
    var hGated = sessionAllocScratch(backend, n)

    let ok = sessionActivation(backend, "sigmoid", hGate, hSig, n) and
             sessionVecOp(backend, "mul", hGate, hSig, hSilu, n) and
             sessionVecOp(backend, "mul", hSilu, hUp, hGated, n) and
             sessionEnd(backend)

    if ok:
      var gatedT = newTensor(upRaw.shape)
      gatedT.data = sessionRead(backend, hGated, n)
      sessionFree(backend, hGate); sessionFree(backend, hUp)
      sessionFree(backend, hSig); sessionFree(backend, hSilu)
      sessionFree(backend, hGated)
      return ff.downProj.forward(gatedT, ctx)

    sessionFree(backend, hGate); sessionFree(backend, hUp)
    sessionFree(backend, hSig); sessionFree(backend, hSilu)
    sessionFree(backend, hGated)
    # rơi xuống nhánh per-op bên dưới nếu session thất bại giữa chừng

  # Fallback: từng op riêng qua beXxx (vẫn 100% GPU, chỉ là không cùng 1
  # session nên có upload/download giữa mỗi bước).
  let sig = beSigmoid(ctx, gateRaw.data)
  let silu = beMul(ctx, gateRaw.data, sig)
  let gated = beMul(ctx, silu, upRaw.data)
  var gatedT = newTensor(upRaw.shape)
  gatedT.data = gated
  result = ff.downProj.forward(gatedT, ctx)

# ===================================================================
# KV Cache (paged) cho GQA -- lưu theo layout [nKVHeads, maxSeqLen, headDim]
# ===================================================================
type
  LayerKVCache* = object
    k*, v*: seq[float32]   # flat [nKVHeads * maxSeqLen * headDim], B=1
    filled*: int           # số vị trí đã ghi

  KVCacheL* = object
    layers*: seq[LayerKVCache]
    nKVHeads*, headDim*, maxSeqLen*: int
    seqLen*: int            # vị trí hiện tại (chung cho mọi layer)

proc newKVCacheL*(nLayers, nKVHeads, headDim, maxSeqLen: int): KVCacheL =
  result.nKVHeads = nKVHeads
  result.headDim = headDim
  result.maxSeqLen = maxSeqLen
  result.seqLen = 0
  result.layers = newSeq[LayerKVCache](nLayers)
  for l in 0 ..< nLayers:
    result.layers[l].k = newSeq[float32](nKVHeads * maxSeqLen * headDim)
    result.layers[l].v = newSeq[float32](nKVHeads * maxSeqLen * headDim)
    result.layers[l].filled = 0

proc appendKV*(cache: var KVCacheL, layer: int, kNew, vNew: seq[float32], nTok: int) =
  ## kNew/vNew layout: [nTok, nKVHeads, headDim] (B=1). Ghi tuần tự vào cache
  ## tại vị trí [cache.seqLen .. cache.seqLen+nTok).
  let D = cache.headDim
  let H = cache.nKVHeads
  doAssert cache.seqLen + nTok <= cache.maxSeqLen, "KV cache overflow"
  for t in 0 ..< nTok:
    for h in 0 ..< H:
      let srcOff = (t * H + h) * D
      let dstOff = (h * cache.maxSeqLen + (cache.seqLen + t)) * D
      for d in 0 ..< D:
        cache.layers[layer].k[dstOff + d] = kNew[srcOff + d]
        cache.layers[layer].v[dstOff + d] = vNew[srcOff + d]
  cache.layers[layer].filled = cache.seqLen + nTok

# ===================================================================
# GQA CausalSelfAttention (RoPE + KV-cache) — thay cho CausalSelfAttention gốc
# ===================================================================
type
  GQAAttention* = object
    nHeads*, nKVHeads*, headDim*, hiddenSize*: int
    qProj*, kProj*, vProj*, oProj*: Linear
    scale*: float32

proc newGQAAttention*(hiddenSize, nHeads, nKVHeads: int): GQAAttention =
  doAssert hiddenSize mod nHeads == 0
  doAssert nHeads mod nKVHeads == 0
  result.hiddenSize = hiddenSize
  result.nHeads = nHeads
  result.nKVHeads = nKVHeads
  result.headDim = hiddenSize div nHeads
  let kvDim = nKVHeads * result.headDim
  result.qProj = newLinear(hiddenSize, hiddenSize)
  result.kProj = newLinear(hiddenSize, kvDim)
  result.vProj = newLinear(hiddenSize, kvDim)
  result.oProj = newLinear(hiddenSize, hiddenSize)
  result.scale = 1'f32 / sqrt(float32(result.headDim))

## forwardQKVFused (dequant-fp32-then-matmul QKV fusion) đã bị XOÁ - không
## còn nơi nào gọi tới (dead code) và nó vẫn đi qua đường chậm
## dequantizeTensorTransposed() mà linearFast() ở trên đã thay thế. Xem
## fusedNormQKVRoPE bên dưới cho đường dùng thật (beMatmulQ4 trực tiếp).

proc fusedNormQKVRoPE(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL,
                       ctx: Backend, T, posOffset: int): tuple[q, k, v: seq[float32]] =
  ## GỘP CROSS-SUBMODULE: RMSNorm(inputNorm, phần scale) -> QKV fused
  ## projection -> RoPE(Q) -> RoPE(K).
  ##
  ## TRƯỚC ĐÂY 4 bước này là 3 session GPU riêng (RMSNorm tự sessionBegin/End,
  ## RoPE(Q) tự session, RoPE(K) tự session) CỘNG THÊM 1 lần beMatmul KHÔNG
  ## qua session cho QKV (forwardQKVFused cũ) - tổng ~4 lượt commit+wait GPU/
  ## layer chỉ riêng phần đầu attention. Giờ CHỈ CÒN ĐÚNG 2 session:
  ##   Session #1: RMSNorm-scale (2 mul) + QKV matmul, encode nối tiếp trong
  ##               CÙNG 1 sessionBegin/sessionEnd (không round-trip CPU giữa
  ##               2 bước này nữa).
  ##   Session #2: RoPE(Q) VÀ RoPE(K) GỘP CHUNG (trước đây 2 session riêng)
  ##               thành 1 batch elementwise (mul,mul,add) duy nhất.
  ## Ranh giới GIỮA session #1 và #2 là BẮT BUỘC, không phải tuỳ chọn: RoPE
  ## cần "rotate_half" (gather đổi dấu 2 nửa vector) trên Q/K vừa ra khỏi
  ## matmul, mà session API hiện tại (sessionMatmul/sessionVecOp/
  ## sessionActivation/sessionSoftmax) KHÔNG có kernel gather -> phải đọc
  ## Q/K về CPU để build rotHalf/cosFull/sinFull rồi upload lại. Trên Metal,
  ## đọc dữ liệu giữa chừng đòi hỏi session đã commit+wait xong (xem
  ## metal_shim.m/metalSessionEnd) nên đây là ranh giới cứng của phần cứng/API,
  ## không phải do chưa tối ưu.
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  let (rows, inF) = flatten2D(x.shape)
  let n = rows * inF

  # ---- CPU bắt buộc TRƯỚC session: chỉ phần reduce RMSNorm (đọc sumSq nhỏ
  # để chạy sqrt) - KHÔNG còn dequant/ghép weight QKV ở đây nữa (xem
  # linearFast ở trên: matmul chạy thẳng trên int4 nén, không cần dequant
  # trước) ----
  let (invBroadcast, wBroadcast, _) = rmsNormPrep(norm, x, ctx)

  let be = ctx.kind.toByby()
  var normed: seq[float32]
  var usedSession = sessionBegin(be)
  if usedSession:
    var hX = sessionUpload(be, x.data)
    var hInv = sessionUpload(be, invBroadcast)
    var hW = sessionUpload(be, wBroadcast)
    let hT1 = sVecOp(be, "mul", hX, hInv, n)          # RMSNorm scale bước 1
    let hNormed = sVecOp(be, "mul", hT1, hW, n)       # RMSNorm scale bước 2
    usedSession = sessionEnd(be)   # bắt buộc: normed cần về CPU để làm input cho beMatmulQ4
    if usedSession:
      normed = sessionRead(be, hNormed, n)
    var hT1v = hT1; var hNormedV = hNormed
    sessionFree(be, hX); sessionFree(be, hInv); sessionFree(be, hW)
    sessionFree(be, hT1v); sessionFree(be, hNormedV)
  if not usedSession:
    let t1 = beMul(ctx, x.data, invBroadcast)
    normed = beMul(ctx, t1, wBroadcast)

  # SỬA (điểm chính): beMatmulQ4 chạy TRỰC TIẾP trên int4 còn nén (Metal/
  # OpenCL có kernel đọc thẳng 4-bit), thay vì dequant fp32 + ghép 1 ma trận
  # khổng lồ + upload session như trước - đây là lý do "1.5 phút/token".
  # Mất đi việc "gộp Q+K+V thành 1 matmul" (khác weight scale/zero-point nên
  # không nối trực tiếp được như lúc còn fp32), đổi lại mỗi phép matmul giờ
  # rẻ hơn HẲN vì không phải giải nén + upload lại toàn bộ trọng số mỗi token.
  let qFlat0 = linearFastBias(attn.qProj, normed, rows, ctx)
  let kFlat0 = linearFastBias(attn.kProj, normed, rows, ctx)
  let vFlat = linearFastBias(attn.vProj, normed, rows, ctx)
  let qFlat = qFlat0
  let kFlat = kFlat0

  # ---- Session #2: RoPE(Q) + RoPE(K) GỘP CHUNG 1 session (trước đây 2
  # session riêng, mỗi cái tự upload/download). Ghép Q và K thành 1 mảng
  # liên tục [nQ+nK] rồi chạy đúng 3 sessionVecOp (mul,mul,add) MỘT LẦN cho
  # cả 2, thay vì 2 lần x 3 vecop = 6 lệnh trên 2 session khác nhau. ----
  let half = D div 2
  let nQ = T * H * D
  let nK = T * HKV * D
  let nAll = nQ + nK

  proc buildRoPEArrays(data: seq[float32], Th, Hh: int, base: int,
                        cosFull, sinFull, rotHalf: var seq[float32]) =
    for t in 0 ..< Th:
      let pos = posOffset + t
      doAssert pos < rope.maxSeqLen, "RoPE: vị trí vượt quá max_seq_len đã precompute"
      for h in 0 ..< Hh:
        let localBase = (t * Hh + h) * D
        let dstBase = base + localBase
        for i in 0 ..< half:
          let cosV = rope.cosT[pos * half + i]
          let sinV = rope.sinT[pos * half + i]
          cosFull[dstBase + i]        = cosV
          cosFull[dstBase + i + half] = cosV
          sinFull[dstBase + i]        = sinV
          sinFull[dstBase + i + half] = sinV
          rotHalf[dstBase + i]        = -data[localBase + i + half]
          rotHalf[dstBase + i + half] =  data[localBase + i]

  var combined = newSeq[float32](nAll)
  copyMem(addr combined[0], unsafeAddr qFlat[0], nQ * sizeof(float32))
  copyMem(addr combined[nQ], unsafeAddr kFlat[0], nK * sizeof(float32))
  var cosFull = newSeq[float32](nAll)
  var sinFull = newSeq[float32](nAll)
  var rotHalf = newSeq[float32](nAll)
  buildRoPEArrays(qFlat, T, H, 0, cosFull, sinFull, rotHalf)
  buildRoPEArrays(kFlat, T, HKV, nQ, cosFull, sinFull, rotHalf)

  var usedSession2 = sessionBegin(be)
  if usedSession2:
    var hX = sessionUpload(be, combined)
    var hCos = sessionUpload(be, cosFull)
    var hRot = sessionUpload(be, rotHalf)
    var hSin = sessionUpload(be, sinFull)
    let hT1 = sVecOp(be, "mul", hX, hCos, nAll)
    let hT2 = sVecOp(be, "mul", hRot, hSin, nAll)
    let hOut = sVecOp(be, "add", hT1, hT2, nAll)
    usedSession2 = sessionEnd(be)
    if usedSession2:
      combined = sessionRead(be, hOut, nAll)
    var hT1v = hT1; var hT2v = hT2; var hOutV = hOut
    sessionFree(be, hX); sessionFree(be, hCos); sessionFree(be, hRot); sessionFree(be, hSin)
    sessionFree(be, hT1v); sessionFree(be, hT2v); sessionFree(be, hOutV)
  if not usedSession2:
    let t1 = beMul(ctx, combined, cosFull)
    let t2 = beMul(ctx, rotHalf, sinFull)
    combined = beAdd(ctx, t1, t2)

  result.q = combined[0 ..< nQ]
  result.k = combined[nQ ..< nAll]
  result.v = vFlat

## ---- PREFILL: xử lý cả prompt 1 lần bằng beAttentionFused (đã có causal mask
## nội bộ, đúng mục đích thiết kế của kernel này) ----
proc forwardPrefill*(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL, ctx: Backend,
                      cache: var KVCacheL, layerIdx: int): Tensor =
  let B = x.shape[0]
  let T = x.shape[1]
  let C = x.shape[2]
  doAssert B == 1, "Bản decode hiện chỉ hỗ trợ B=1 (single-sequence generation)"
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  let nRep = H div HKV

  # GỘP SESSION cross-submodule: RMSNorm(inputNorm) -> QKV fused -> RoPE(Q)
  # -> RoPE(K) (xem fusedNormQKVRoPE ở trên) - trước đây đây là 3 session +
  # 1 matmul rời, giờ còn 2 session.
  let (qData, kData, vData) = fusedNormQKVRoPE(attn, norm, x, rope, ctx, T, posOffset = 0)

  # Lưu K,V (đã RoPE) vào cache theo layout [T,HKV,D] -> appendKV tự chuyển
  # sang [HKV, maxSeqLen, D] bên trong.
  appendKV(cache, layerIdx, kData, vData, T)

  # Mở rộng K,V từ HKV head -> H head (mỗi kv-head lặp lại nRep lần liên tục,
  # đúng convention repeat_kv của HF) để dùng beAttentionFused (yêu cầu Q,K,V
  # cùng số head).
  var kExp = newSeq[float32](T * H * D)
  var vExp = newSeq[float32](T * H * D)
  for t in 0 ..< T:
    for hkv in 0 ..< HKV:
      let srcOff = (t * HKV + hkv) * D
      for r in 0 ..< nRep:
        let h = hkv * nRep + r
        let dstOff = (t * H + h) * D
        for d in 0 ..< D:
          kExp[dstOff + d] = kData[srcOff + d]
          vExp[dstOff + d] = vData[srcOff + d]

  # SỬA BUG: beAttentionFused cần layout [H,T,D] (head-major) chứ không phải
  # [T,H,D] (position-major) mà qData/kExp/vExp đang có (xem toHeadMajor ở
  # trên). Thiếu bước transpose này khiến kernel đọc lẫn head/token -> attention
  # sai hoàn toàn -> sinh token rác ngay từ prefill.
  let qHM = toHeadMajor(qData, T, H, D)
  let kHM = toHeadMajor(kExp, T, H, D)
  let vHM = toHeadMajor(vExp, T, H, D)
  let (attnOutFlatHM, _) = beAttentionFused(ctx, qHM, kHM, vHM, [], B, H, T, D, attn.scale)
  let attnOutFlat = toPosMajor(attnOutFlatHM, T, H, D)   # về lại [T,H,D]=[T,C] cho oProj
  var attnOut = newTensor(@[B, T, C])
  attnOut.data = attnOutFlat
  result = attn.oProj.forward(attnOut, ctx)

## ---- DECODE: 1 token mới, attention thủ công qua beMatmul+beSoftmax trên
## toàn bộ cache (O(S), tuyến tính -- đây mới là "paged attention" thật) ----
proc forwardDecode*(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL, ctx: Backend,
                     cache: var KVCacheL, layerIdx: int): Tensor =
  let C = x.shape[2]
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  let nRep = H div HKV
  let S = cache.seqLen + 1   # tổng độ dài sau khi thêm token này

  # GỘP SESSION cross-submodule (giống forwardPrefill ở trên).
  let (qData, kNew, vNew) = fusedNormQKVRoPE(attn, norm, x, rope, ctx, T = 1, posOffset = cache.seqLen)

  appendKV(cache, layerIdx, kNew, vNew, 1)

  var outHeads = newSeq[float32](H * D)
  let maxS = cache.maxSeqLen

  # Gom dữ liệu từng group trước (CPU, rẻ) để có thể encode hết vào 1 session
  var kBlockTs = newSeq[seq[float32]](HKV)
  var vBlocks = newSeq[seq[float32]](HKV)
  var qGroups = newSeq[seq[float32]](HKV)
  for hkv in 0 ..< HKV:
    var kBlock = newSeq[float32](S * D)
    var vBlock = newSeq[float32](S * D)
    for s in 0 ..< S:
      let srcOff = (hkv * maxS + s) * D
      let dstOff = s * D
      for d in 0 ..< D:
        kBlock[dstOff + d] = cache.layers[layerIdx].k[srcOff + d]
        vBlock[dstOff + d] = cache.layers[layerIdx].v[srcOff + d]
    var kBlockT = newSeq[float32](D * S)
    for s in 0 ..< S:
      for d in 0 ..< D:
        kBlockT[d * S + s] = kBlock[s * D + d]
    kBlockTs[hkv] = kBlockT
    vBlocks[hkv] = vBlock

    var qGroup = newSeq[float32](nRep * D)
    for r in 0 ..< nRep:
      let h = hkv * nRep + r
      for d in 0 ..< D:
        # gộp luôn scale vào Q ở CPU (rẻ, mảng nhỏ) để khỏi cần thêm 1 GPU op
        # riêng cho phép nhân scale trong session (session chỉ có matmul/softmax)
        qGroup[r * D + d] = qData[h * D + d] * attn.scale
    qGroups[hkv] = qGroup

  let backend = ctx.kind.toByby()
  var usedSession = false
  if sessionBegin(backend):
    usedSession = true
    # SUA (BUG QUAN TRONG - nghi la nguyen nhan chinh cua "token loi" luc
    # decode): TRUOC DAY sessionRead(hOut) duoc goi NGAY TRONG vong lap,
    # TRUOC KHI sessionEnd() duoc goi mot lan duy nhat sau vong lap. Tren
    # Metal, du lieu GPU ghi vao buffer CHI thuc su "xong" sau
    # commit+waitUntilCompleted - dieu nay CHI xay ra ben trong
    # metalSessionEnd() (xem metal_session_end trong metal_shim.m). Doc
    # ket qua (hOut) truoc khi goi sessionEnd() nghia la doc buffer TRUOC KHI
    # GPU thuc su chay xong lenh - tren storage mode shared cua Metal, dieu
    # nay tra ve du lieu cu/rac chu khong phai loi cua ban than phep tinh.
    # Fix: giu song TAT CA cac hOut handle cua moi KV-head group, chi
    # sessionRead() SAU KHI sessionEnd() da chay xong dung 1 lan.
    var hOuts = newSeq[SessionHandle](HKV)
    for hkv in 0 ..< HKV:
      var hQ = sessionUpload(backend, qGroups[hkv])
      var hKT = sessionUpload(backend, kBlockTs[hkv])
      var hV = sessionUpload(backend, vBlocks[hkv])
      var hScores = sessionAllocScratch(backend, nRep * S)
      var hProbs = sessionAllocScratch(backend, nRep * S)
      hOuts[hkv] = sessionAllocScratch(backend, nRep * D)

      let ok = sessionMatmul(backend, hQ, hKT, hScores, nRep, D, S) and
               sessionSoftmax(backend, hScores, hProbs, nRep, S) and
               sessionMatmul(backend, hProbs, hV, hOuts[hkv], nRep, S, D)

      # Chi giai phong buffer TRUNG GIAN (khong con can sau khi encode xong) -
      # hOuts[hkv] PHAI song toi luc doc duoc sau sessionEnd(), khong free o day.
      sessionFree(backend, hQ); sessionFree(backend, hKT); sessionFree(backend, hV)
      sessionFree(backend, hScores); sessionFree(backend, hProbs)

      if not ok:
        usedSession = false
        break

    if usedSession: usedSession = sessionEnd(backend)

    if usedSession:
      # Doc TAT CA ket qua SAU KHI sessionEnd() da commit+wait xong dung 1 lan.
      for hkv in 0 ..< HKV:
        let outGroup = sessionRead(backend, hOuts[hkv], nRep * D)
        for r in 0 ..< nRep:
          let h = hkv * nRep + r
          for d in 0 ..< D:
            outHeads[h * D + d] = outGroup[r * D + d]

    for hkv in 0 ..< HKV:
      sessionFree(backend, hOuts[hkv])

  # SỬA: session ở trên giữ dữ liệu GPU-resident xuyên suốt matmul->softmax
  # ->matmul cho mỗi KV-head group (trước đây 3 lệnh beMatmul/beSoftmax/
  # beMatmul RIÊNG, mỗi lệnh tự upload+download - đúng cái CPU<->GPU đi lại
  # liên tục mà session sinh ra để tránh).
  #
  # CẢNH BÁO QUAN TRỌNG (đọc trước khi tin kết quả trên Metal): nhánh session
  # NÀY GỌI THẲNG VÀO metal_shim.m (metalSessionMatmulEnc/SoftmaxEnc) - đúng
  # đoạn code Objective-C mà 1 lần thử trước đó (xem comment cũ) ĐÃ RA KẾT
  # QUẢ SAI trên driver Intel Iris khi dùng chung 1 encoder cho nhiều
  # pipeline (matmul rồi softmax rồi matmul) trong 1 command buffer. Em
  # KHÔNG sửa được metal_shim.m vì không compile/test được trên Metal thật ở
  # đây - nếu bạn build và thấy output SAI hoặc chậm bất thường trên Mac,
  # đây chính là chỗ nghi ngờ đầu tiên. CUDA/OpenCL không có lịch sử lỗi này
  # (CUDA dùng stream async tự nhiên, OpenCL dùng queue tuần tự đơn giản, cả
  # 2 không có khái niệm "1 encoder chia sẻ nhiều pipeline" như Metal).
  # Nếu nghi ngờ sai, đổi lại `if false and sessionBegin(...)` ở trên để tắt
  # nhánh session, quay về per-op beMatmul/beSoftmax (chắc chắn đúng, chỉ
  # chậm hơn vì CPU<->GPU đi lại).
  if not usedSession:
    for hkv in 0 ..< HKV:
      var scores = beMatmul(ctx, qGroups[hkv], nRep, D, kBlockTs[hkv], D, S)
      let probs = beSoftmax(ctx, scores, nRep, S)
      let outGroup = beMatmul(ctx, probs, nRep, S, vBlocks[hkv], S, D)
      for r in 0 ..< nRep:
        let h = hkv * nRep + r
        for d in 0 ..< D:
          outHeads[h * D + d] = outGroup[r * D + d]

  var attnOut = newTensor(@[1, 1, C])
  attnOut.data = outHeads
  result = attn.oProj.forward(attnOut, ctx)

# ===================================================================
# TransformerBlockLlama (pre-norm: RMSNorm -> Attn -> residual -> RMSNorm -> SwiGLU -> residual)
# ===================================================================
type
  TransformerBlockLlama* = object
    inputNorm*, postAttnNorm*: RMSNormL
    attn*: GQAAttention
    ff*: SwiGLUL

proc newTransformerBlockLlama*(hidden, nHeads, nKVHeads, intermediate: int, eps: float32): TransformerBlockLlama =
  result.inputNorm = newRMSNormL(hidden, eps)
  result.postAttnNorm = newRMSNormL(hidden, eps)
  result.attn = newGQAAttention(hidden, nHeads, nKVHeads)
  result.ff = newSwiGLUL(hidden, intermediate)

proc fusedResidualNormFFN(norm: RMSNormL, ff: SwiGLUL, x, attnOut: Tensor, ctx: Backend): Tensor =
  ## GỘP CROSS-SUBMODULE toàn bộ đuôi của 1 block: residual-add(sau Attn) ->
  ## RMSNorm(postAttnNorm, scale) -> FFN (gate/up matmul + sigmoid+mul+mul +
  ## down matmul) -> residual-add(sau FFN) THÀNH ĐÚNG 1 SESSION GPU DUY NHẤT.
  ##
  ## TRƯỚC ĐÂY: addT (plain, không session) + RMSNorm (session riêng) + FFN
  ## gate/up matmul (plain, không session) + FFN sigmoid/mul/mul (session
  ## riêng) + FFN down matmul (plain) + addT (plain) = 2 session mở/đóng +
  ## 4 lượt GPU rời rạc khác, mỗi lượt tự upload/download.
  ##
  ## GIỜ: residual-add đầu, RMSNorm-scale, gate-matmul, up-matmul, sigmoid,
  ## silu-mul, gated-mul, down-matmul, residual-add cuối - TẤT CẢ encode nối
  ## tiếp vào CÙNG 1 sessionBegin/sessionEnd, không có điểm đọc GPU->CPU nào
  ## ở giữa (khác với fusedNormQKVRoPE, đuôi này KHÔNG cần gather/rotate_half
  ## nên không có ranh giới cứng nào bắt buộc phải ngắt session).
  let n = x.data.len
  let (rows, hidden) = flatten2D(x.shape)
  doAssert hidden == norm.dim

  # ---- CPU bắt buộc TRƯỚC session: chỉ phần reduce RMSNorm (đọc sumSq nhỏ
  # để chạy sqrt). KHÔNG còn dequant weight gate/up/down ở đây nữa - đây
  # chính là phần tốn thời gian nhất trước đây (3 ma trận full-size giải nén
  # + upload lại MỖI TOKEN, MỖI LAYER dù trọng số không hề đổi giữa các
  # token) - xem linearFast() ở đầu file.
  let x2 = addT(ctx, x, attnOut)
  let (invBroadcast, wBroadcast, _) = rmsNormPrep(norm, x2, ctx)
  let interF = ff.gateProj.outF
  let nInter = rows * interF

  let be = ctx.kind.toByby()
  var normed: seq[float32]
  var usedSession = sessionBegin(be)
  if usedSession:
    var hX2 = sessionUpload(be, x2.data)
    var hInv = sessionUpload(be, invBroadcast)
    var hW = sessionUpload(be, wBroadcast)
    let hT1 = sVecOp(be, "mul", hX2, hInv, n)             # RMSNorm scale 1
    let hNormed = sVecOp(be, "mul", hT1, hW, n)           # RMSNorm scale 2
    usedSession = sessionEnd(be)  # bắt buộc: normed cần về CPU làm input cho beMatmulQ4
    if usedSession:
      normed = sessionRead(be, hNormed, n)
    var hT1v = hT1; var hNormedV = hNormed
    sessionFree(be, hX2); sessionFree(be, hInv); sessionFree(be, hW)
    sessionFree(be, hT1v); sessionFree(be, hNormedV)
  if not usedSession:
    let t1 = beMul(ctx, x2.data, invBroadcast)
    normed = beMul(ctx, t1, wBroadcast)

  # SỬA (điểm chính): beMatmulQ4 trực tiếp trên int4 nén, không dequant fp32
  # + upload toàn bộ ma trận gate/up/down mỗi lần forward nữa.
  let gateRaw = linearFastBias(ff.gateProj, normed, rows, ctx)
  let upRaw = linearFastBias(ff.upProj, normed, rows, ctx)

  # Session nhỏ thứ 2: sigmoid+mul+mul (SwiGLU activation) - chỉ elementwise
  # trên kích thước [rows,interF], không đụng tới weight nào -> vẫn rẻ để
  # gộp session dù không còn nối liền với matmul QKV/gate/up như trước.
  var gated: seq[float32]
  var usedSession2 = sessionBegin(be)
  if usedSession2:
    var hGate = sessionUpload(be, gateRaw)
    var hUp = sessionUpload(be, upRaw)
    let hSig = sActivation(be, "sigmoid", hGate, nInter)
    let hSilu = sVecOp(be, "mul", hGate, hSig, nInter)
    let hGated = sVecOp(be, "mul", hSilu, hUp, nInter)
    usedSession2 = sessionEnd(be)
    if usedSession2:
      gated = sessionRead(be, hGated, nInter)
    var hSigV = hSig; var hSiluV = hSilu; var hGatedV = hGated
    sessionFree(be, hGate); sessionFree(be, hUp)
    sessionFree(be, hSigV); sessionFree(be, hSiluV); sessionFree(be, hGatedV)
  if not usedSession2:
    let sig = beSigmoid(ctx, gateRaw)
    let silu = beMul(ctx, gateRaw, sig)
    gated = beMul(ctx, silu, upRaw)

  let ffOut = linearFastBias(ff.downProj, gated, rows, ctx)
  result = newTensor(x.shape)
  result.data = beAdd(ctx, x2.data, ffOut)   # residual-add cuối

proc forward*(blk: TransformerBlockLlama, x: Tensor, rope: RoPEL, ctx: Backend,
              cache: var KVCacheL, layerIdx: int, isPrefill: bool): Tensor =
  ## RMSNorm(input) nằm BÊN TRONG forwardPrefill/forwardDecode (gộp với QKV+
  ## RoPE - xem fusedNormQKVRoPE), và residual+RMSNorm(post-attn)+FFN+residual
  ## nằm trong fusedResidualNormFFN - cả block giờ chỉ còn 2 session GPU thay
  ## vì 4+ session rời rạc theo từng submodule như trước.
  let attnOut =
    if isPrefill: forwardPrefill(blk.attn, blk.inputNorm, x, rope, ctx, cache, layerIdx)
    else: forwardDecode(blk.attn, blk.inputNorm, x, rope, ctx, cache, layerIdx)
  result = fusedResidualNormFFN(blk.postAttnNorm, blk.ff, x, attnOut, ctx)

# ===================================================================
# LlamaModelL — model đầy đủ
# ===================================================================
type
  LlamaLConfig* = object
    vocabSize*, hiddenSize*, intermediateSize*: int
    nLayers*, nHeads*, nKVHeads*: int
    maxPositionEmbeddings*: int
    rmsNormEps*, ropeTheta*: float32

  LlamaModelL* = object
    config*: LlamaLConfig
    embed*: Embedding
    blocks*: seq[TransformerBlockLlama]
    finalNorm*: RMSNormL
    lmHead*: Linear
    rope*: RoPEL

proc newLlamaModelL*(cfg: LlamaLConfig): LlamaModelL =
  result.config = cfg
  result.embed = newEmbedding(cfg.vocabSize, cfg.hiddenSize)
  result.blocks = newSeq[TransformerBlockLlama](cfg.nLayers)
  for i in 0 ..< cfg.nLayers:
    result.blocks[i] = newTransformerBlockLlama(cfg.hiddenSize, cfg.nHeads, cfg.nKVHeads,
                                                  cfg.intermediateSize, cfg.rmsNormEps)
  result.finalNorm = newRMSNormL(cfg.hiddenSize, cfg.rmsNormEps)
  result.lmHead = newLinear(cfg.hiddenSize, cfg.vocabSize)
  result.rope = newRoPEL(cfg.hiddenSize div cfg.nHeads, cfg.maxPositionEmbeddings, cfg.ropeTheta)

proc forwardStep*(m: LlamaModelL, ids: seq[int], ctx: Backend, cache: var KVCacheL,
                   isPrefill: bool): Tensor =
  ## isPrefill=true: ids = toàn bộ prompt (T token). isPrefill=false: ids = đúng 1 token mới.
  var x = m.embed.lookupBatch(@[ids], ctx)   # [1, T, C]
  when defined(debugStates):
    let Tdbg = ids.len
    let Cdbg = m.config.hiddenSize
    stderr.writeLine "[debug] after_embedding last_token[:8] = " & $x.data[(Tdbg-1)*Cdbg ..< (Tdbg-1)*Cdbg+8]
  for idx in 0 ..< m.blocks.len:
    x = m.blocks[idx].forward(x, m.rope, ctx, cache, idx, isPrefill)
    # SỬA (hiệu năng - nguyên nhân chính của "1 token = 1.5 phút"):
    # TRƯỚC ĐÂY GC_fullCollect() được gọi sau MỖI layer, tức 32 lần/token dù
    # đang decode (T=1). GC_fullCollect() là mark-sweep TOÀN BỘ heap, kể cả
    # phần dữ liệu SỐNG (weight ~4GB đã load, KV-cache, v.v...) không chỉ rác
    # - chi phí của nó tỉ lệ với tổng heap còn sống, không phải với lượng rác
    # tạo ra mỗi layer. Với model 6.7B, quét ~4GB "sống" 32 lần cho MỖI token
    # sinh ra chính là phần lớn thời gian "1.5 phút/token" (không phải GPU
    # kernel chậm). Giữ đúng lý do ban đầu (dequant tạm vài trăm MB/layer
    # chồng lên nhau -> OOM) nhưng chỉ ép dọn định kỳ (mỗi 4 layer + layer
    # cuối) thay vì mọi layer - giảm ~8 lần số lần quét full-heap mỗi bước,
    # vẫn chặn được việc tích luỹ quá 4 layer tạm trước khi GC dọn.
    if (idx + 1) mod 4 == 0 or idx == m.blocks.len - 1:
      GC_fullCollect()
    when defined(debugStates):
      if idx == 0:
        stderr.writeLine "[debug] after_block_0   last_token[:8] = " & $x.data[(Tdbg-1)*Cdbg ..< (Tdbg-1)*Cdbg+8]
  when defined(debugStates):
    stderr.writeLine "[debug] after_last_block last_token[:8] = " & $x.data[(Tdbg-1)*Cdbg ..< (Tdbg-1)*Cdbg+8]
  let normed = m.finalNorm.forward(x, ctx)
  result = m.lmHead.forward(normed, ctx)
  when defined(debugStates):
    let Vdbg = m.config.vocabSize
    let lastRow = result.data[(Tdbg-1)*Vdbg ..< (Tdbg-1)*Vdbg+8]
    stderr.writeLine "[debug] logits last_token[:8] = " & $lastRow
  cache.seqLen += ids.len

# ===================================================================
# Load config + weights từ checkpoint HF/GPTQ (qua nimpy + quant.nim)
# ===================================================================
proc newLlamaLConfig*(modelPath: string): LlamaLConfig =
  let transformers = pyImport("transformers")
  let cfg = transformers.AutoConfig.from_pretrained(modelPath, trust_remote_code = true)
  result.vocabSize = cfg.vocab_size.to(int)
  result.hiddenSize = cfg.hidden_size.to(int)
  result.intermediateSize = cfg.intermediate_size.to(int)
  result.nLayers = cfg.num_hidden_layers.to(int)
  result.nHeads = cfg.num_attention_heads.to(int)
  try:
    result.nKVHeads = cfg.num_key_value_heads.to(int)
  except CatchableError:
    result.nKVHeads = result.nHeads   # model không dùng GQA -> MHA thường
  result.maxPositionEmbeddings = cfg.max_position_embeddings.to(int)
  result.rmsNormEps = (try: cfg.rms_norm_eps.to(float32) except CatchableError: 1e-5'f32)
  result.ropeTheta = (try: cfg.rope_theta.to(float32) except CatchableError: 10000'f32)

proc loadLlamaWeightsL*(m: var LlamaModelL, path: string) =
  echo &"  đang đọc file {path} ..."
  let fileSize = getFileSize(path)
  echo &"  kích thước file: {fileSize div (1024*1024)} MB"
  let (_, sd) = loadQuantStateDict(path)
  echo &"  đã đọc xong {sd.len} tensor từ file, đang build model..."
  var byName = initTable[string, QuantTensor]()
  for (name, qt) in sd: byName[name] = qt

  proc get(name: string): seq[float32] =
    if not byName.hasKey(name): return @[]
    dequantizeTensor(byName[name])

  proc loadLinear(lin: var Linear, prefix: string) =
    ## SỬA: giữ weight NÉN (setQuantWeight, xem nimformer.nim) thay vì
    ## dequant hết ra fp32 ngay lúc load (get() cũ) - với model 6.7B, dequant
    ## hết tất cả Linear ra fp32 tốn ~27GB RAM (nguyên nhân bị "killed" OOM
    ## lúc "Building model..."). Giữ nén chỉ tốn ~4GB, dequant tạm mỗi lần
    ## forward() rồi bỏ ngay (xem nimformer.nim Linear.forward).
    let wName = prefix & ".weight"
    if byName.hasKey(wName):
      setQuantWeight(lin, byName[wName])
    let b = get(prefix & ".bias")
    if b.len > 0: lin.bias.data = b

  let embW = get("model.embed_tokens.weight")
  if embW.len > 0: m.embed.weight.data = embW

  let normW = get("model.norm.weight")
  if normW.len > 0: m.finalNorm.weight.data = normW

  for l in 0 ..< m.config.nLayers:
    echo &"  loading layer {l+1}/{m.config.nLayers} ..."
    let p = "model.layers." & $l
    let ln1 = get(p & ".input_layernorm.weight")
    if ln1.len > 0: m.blocks[l].inputNorm.weight.data = ln1
    let ln2 = get(p & ".post_attention_layernorm.weight")
    if ln2.len > 0: m.blocks[l].postAttnNorm.weight.data = ln2

    loadLinear(m.blocks[l].attn.qProj, p & ".self_attn.q_proj")
    loadLinear(m.blocks[l].attn.kProj, p & ".self_attn.k_proj")
    loadLinear(m.blocks[l].attn.vProj, p & ".self_attn.v_proj")
    loadLinear(m.blocks[l].attn.oProj, p & ".self_attn.o_proj")

    loadLinear(m.blocks[l].ff.gateProj, p & ".mlp.gate_proj")
    loadLinear(m.blocks[l].ff.upProj, p & ".mlp.up_proj")
    loadLinear(m.blocks[l].ff.downProj, p & ".mlp.down_proj")

  # lm_head: nếu có trong file thì setQuantWeight (giữ nén); nếu không (tied
  # embeddings) thì dùng thẳng embW đã dequant sẵn (chỉ 1 lần, không phải
  # 224 lần như các Linear khác nên không đáng lo RAM) - Linear.forward tự
  # fallback sang nhánh useQuant=false, transpose bình thường mỗi lần gọi
  # (chấp nhận được vì lm_head chỉ forward 1 LẦN/token, không phải 224 lần).
  if byName.hasKey("lm_head.weight"):
    setQuantWeight(m.lmHead, byName["lm_head.weight"])
  elif embW.len > 0:
    m.lmHead.weight.data = embW
    m.lmHead.useQuant = false


type
  HFTokenizer* = object
    tok: PyObject
    eosId*, vocabSize*: int

proc newHFTokenizer*(modelPath: string): HFTokenizer =
  let transformers = pyImport("transformers")
  result.tok = transformers.AutoTokenizer.from_pretrained(modelPath, trust_remote_code = true)
  result.eosId = (try: result.tok.eos_token_id.to(int) except CatchableError: -1)
  result.vocabSize = result.tok.vocab_size.to(int)

proc encode*(tok: HFTokenizer, text: string): seq[int] =
  tok.tok.encode(text, add_special_tokens = true).to(seq[int])

proc decode*(tok: HFTokenizer, ids: seq[int]): string =
  tok.tok.decode(ids, skip_special_tokens = true).to(string)

proc sampleTopP(logits: seq[float32], temperature: float32 = 0.7'f32, topP: float32 = 0.9'f32,
                 recentIds: openArray[int] = [], repeatPenalty: float32 = 1.3'f32): int =
  # SỬA: thêm repetition penalty (kiểu llama.cpp) - không có cái này model
  # dễ rơi vào vòng lặp token kiểu "),),),),)" vì mỗi bước cứ chọn lại đúng
  # token có logit cao nhất từng chọn trước đó. Hạ logit của token đã xuất
  # hiện gần đây (chia cho repeatPenalty nếu dương, nhân nếu âm) trước khi
  # tính softmax/top-p.
  var adjLogits = logits
  if repeatPenalty != 1.0'f32:
    for id in recentIds:
      if id >= 0 and id < adjLogits.len:
        if adjLogits[id] > 0'f32: adjLogits[id] = adjLogits[id] / repeatPenalty
        else: adjLogits[id] = adjLogits[id] * repeatPenalty

  if temperature <= 0'f32:
    var best = 0
    for i in 1 ..< adjLogits.len:
      if adjLogits[i] > adjLogits[best]: best = i
    return best
  var probs = newSeq[float32](adjLogits.len)
  var maxL = adjLogits[0]
  for v in adjLogits:
    if v > maxL: maxL = v
  var sumExp = 0'f32
  for i in 0 ..< adjLogits.len:
    probs[i] = exp((adjLogits[i] - maxL) / temperature)
    sumExp += probs[i]
  for i in 0 ..< probs.len: probs[i] /= sumExp

  var idx = toSeq(0 ..< probs.len)
  idx.sort(proc(a, b: int): int = cmp(probs[b], probs[a]))
  var cum = 0'f32
  var cutoff = idx.len
  for i, id in idx:
    cum += probs[id]
    if cum >= topP:
      cutoff = i + 1
      break
  var r = rand(1.0'f32) * cum
  var acc = 0'f32
  for i in 0 ..< cutoff:
    acc += probs[idx[i]]
    if acc >= r: return idx[i]
  return idx[cutoff - 1]

proc generate*(m: LlamaModelL, tok: HFTokenizer, ctx: Backend, prompt: string,
               maxNewTokens: int = 128, temperature: float32 = 0.7'f32, topP: float32 = 0.9'f32): string =
  let promptIds = tok.encode(prompt)
  # SỬA: TRƯỚC ĐÂY dùng thẳng m.config.maxPositionEmbeddings (=16384 với
  # model này) làm kích thước KV-cache -> cấp phát ~16GB RAM (32 layer x 32
  # KV-head x 16384 vị trí x 128 dim x 2(k+v) x 4 byte) NGAY LÚC BẮT ĐẦU,
  # bất kể prompt/maxNewTokens thực tế ngắn cỡ nào - nguyên nhân OOM "killed"
  # lúc Generating. Giờ chỉ cấp đúng độ dài thật sự cần dùng (prompt +
  # token sẽ sinh ra), dư thêm chút cho an toàn.
  let neededSeqLen = promptIds.len + maxNewTokens + 8
  var cache = newKVCacheL(m.config.nLayers, m.config.nKVHeads,
                           m.config.hiddenSize div m.config.nHeads,
                           neededSeqLen)

  stdout.write(prompt)

  # ---- ĐO TOK/S: tách riêng prefill (xử lý cả prompt 1 lần, số liệu
  # "prompt tokens/s") và decode (sinh từng token 1, số liệu "tok/s" thật sự
  # người dùng quan tâm khi chat) - giống cách llama.cpp báo cáo
  # "prompt eval time" / "eval time" riêng biệt, vì 2 giai đoạn có chi phí
  # rất khác nhau (prefill tận dụng song song theo chiều T, decode thì không).
  let prefillStart = epochTime()

  # ---- PREFILL: cả prompt trong 1 lần forward (beAttentionFused, causal nội bộ) ----
  var logits = m.forwardStep(promptIds, ctx, cache, isPrefill = true)
  let prefillTime = epochTime() - prefillStart
  let lastRow = logits.data[(logits.shape[1] - 1) * m.config.vocabSize ..< logits.shape[1] * m.config.vocabSize]
  var nextId = sampleTopP(lastRow, temperature, topP, promptIds[max(0, promptIds.len - 64) ..< promptIds.len])

  var allIds = promptIds
  allIds.add(nextId)
  var generated = 1
  var decodeStart = epochTime()   # tính từ SAU prefill, không tính token đầu tiên (đã sinh trong prefill)

  # ---- DECODE: từng token 1, O(S) mỗi bước qua KV-cache paged ----
  # SỬA: KHÔNG decode từng token id riêng lẻ (tok.decode(@[nextId])) — với
  # BPE/byte-level tokenizer, 1 ký tự multi-byte (UTF-8) hoặc 1 từ thường bị
  # tách thành NHIỀU token id; decode rời từng id làm tokenizer không ghép
  # lại được byte-pair/UTF-8 đúng, ra ký tự rác kiểu "模/同/临" và mất
  # khoảng trắng giữa từ. Fix: decode lại TOÀN BỘ allIds mỗi bước, in phần
  # delta (text mới) so với lần in trước - đúng cách streaming chuẩn.
  var printedText = tok.decode(promptIds)   # phần đã in ra (prompt) rồi
  block:
    let fullText0 = tok.decode(allIds)      # allIds = promptIds + nextId đầu tiên
    if fullText0.len > printedText.len:
      stdout.write(fullText0[printedText.len ..< fullText0.len])
      stdout.flushFile()
    printedText = fullText0

  while generated < maxNewTokens:
    if nextId == tok.eosId: break

    let stepStart = epochTime()
    let stepLogits = m.forwardStep(@[nextId], ctx, cache, isPrefill = false)
    nextId = sampleTopP(stepLogits.data, temperature, topP, allIds[max(0, allIds.len - 64) ..< allIds.len])
    let stepTime = epochTime() - stepStart
    allIds.add(nextId)
    inc generated

    if nextId == tok.eosId: break
    let fullText = tok.decode(allIds)
    if fullText.len > printedText.len:
      stdout.write(fullText[printedText.len ..< fullText.len])
      stdout.flushFile()
    printedText = fullText
    stderr.write(&"\n[tok/s tức thời: {(1.0/stepTime):.2f} | {(stepTime*1000):.1f} ms/token]")
    stderr.flushFile()

  let decodeTime = epochTime() - decodeStart
  let decodedCount = generated - 1   # không tính token đầu (sinh trong lúc prefill)
  stdout.write("\n")
  stderr.write("\n== Thống kê tốc độ ==\n")
  stderr.write(&"Prefill: {promptIds.len} token trong {prefillTime:.3f}s -> {(promptIds.len.float/prefillTime):.2f} tok/s\n")
  if decodedCount > 0 and decodeTime > 0:
    stderr.write(&"Decode:  {decodedCount} token trong {decodeTime:.3f}s -> {(decodedCount.float/decodeTime):.2f} tok/s ({(decodeTime*1000/decodedCount.float):.1f} ms/token)\n")
  stderr.write(&"Tổng: {(promptIds.len+decodedCount)} token trong {(prefillTime+decodeTime):.3f}s\n")
  result = tok.decode(allIds)

when isMainModule:
  setForbidCpuFallback(true)
  let modelPath = "TheBloke/deepseek-coder-6.7B-instruct-GPTQ"
  echo "Loading config..."
  let cfg = newLlamaLConfig(modelPath)
  echo "Loading tokenizer..."
  let tok = newHFTokenizer(modelPath)
  echo "Init backend..."
  let ctx = newBackend("metal")
  echo "Building model..."
  var model = newLlamaModelL(cfg)
  echo "Loading weights..."
  loadLlamaWeightsL(model, "model.nimq")
  echo "Generating (prefill + paged KV-cache decode, 100% GPU qua BybyLang)..."
  let prompt = "def fib(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    "
  discard generate(model, tok, ctx, prompt, maxNewTokens = 16)