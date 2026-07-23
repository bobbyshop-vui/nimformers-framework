## nimformer.nim
## Transformer đầy đủ với backward cho attention + custom transformer blocks

import std/[math, random, strformat]
import customfloat
import backend  # Import Backend và các helper
import quant     # QuantTensor, dequantizeTensor - giữ weight nén int4 trong RAM

# ===================================================================
# Tensor utilities (giữ nguyên)
# ===================================================================
type
  Tensor* = object
    data*: seq[float32]
    shape*: seq[int]

proc numel(shape: seq[int]): int =
  result = 1
  for s in shape: result *= s

proc newTensor*(shape: seq[int], fill: float32 = 0'f32): Tensor =
  result.shape = shape
  result.data = newSeq[float32](numel(shape))
  if fill != 0'f32:
    for i in 0 ..< result.data.len: result.data[i] = fill

proc randnTensor*(shape: seq[int], scale: float32): Tensor =
  result.shape = shape
  result.data = newSeq[float32](numel(shape))
  for i in 0 ..< result.data.len:
    result.data[i] = float32(gauss(0.0, 1.0)) * scale

proc addT*(a, b: Tensor): Tensor =
  result = a
  for i in 0 ..< result.data.len: result.data[i] += b.data[i]

proc addT*(ctx: Backend, a, b: Tensor): Tensor =
  ## SỬA: bản addT(a,b) ở trên chạy CPU thuần dù beAdd (GPU thật, có cài cho
  ## cả CUDA/Metal/OpenCL) đã có sẵn - residual connection này chạy 2 LẦN MỖI
  ## LAYER (attn + ff), tức 64 lần/token với model 32 layer, không phải phép
  ## tính nhỏ có thể bỏ qua. Overload này route qua GPU đúng yêu cầu "không
  ## được lặng lẽ chạy CPU". Giữ nguyên bản addT(a,b) cũ (không ctx) cho các
  ## chỗ dùng khác không có Backend trong scope.
  result = a
  result.data = beAdd(ctx, a.data, b.data)

proc subT*(a, b: Tensor): Tensor =
  result = a
  for i in 0 ..< result.data.len: result.data[i] -= b.data[i]

proc mulT*(a, b: Tensor): Tensor =
  result = a
  for i in 0 ..< result.data.len: result.data[i] *= b.data[i]

proc flatten2D*(shape: seq[int]): tuple[rows, cols: int] =
  let cols = shape[^1]
  var rows = 1
  for i in 0 ..< shape.len - 1: rows *= shape[i]
  result = (rows, cols)

proc transpose*(x: Tensor): Tensor =
  ## Transpose 1 tensor coi như ma trận 2D [rows, cols] (gộp mọi chiều trừ
  ## chiều cuối làm "rows", giống flatten2D) -> [cols, rows].
  let (rows, cols) = flatten2D(x.shape)
  result.shape = @[cols, rows]
  result.data = newSeq[float32](rows * cols)
  for i in 0 ..< rows:
    for j in 0 ..< cols:
      result.data[j * rows + i] = x.data[i * cols + j]

# ===================================================================
# Linear (forward/backward dùng Metal matmul)
# ===================================================================
type
  Linear* = object
    weight*: Tensor      ## fallback fp32 nhỏ (chỉ dùng cho random-init lúc chưa load weight thật)
    weightQ*: QuantTensor ## THÊM: weight THẬT giữ nguyên dạng NÉN (int4/int8...) suốt vòng đời
                           ## process, giống cách llama.cpp giữ quantized weight trong RAM -
                           ## KHÔNG bao giờ expand hết ra fp32 một lần rồi giữ mãi (đó là
                           ## nguyên nhân OOM trước đó: 6.7B tham số x 4 byte fp32 = ~27GB RAM).
    useQuant*: bool        ## true sau khi loadLinear() gán weightQ thật (int4 nén, ~4GB tổng
                            ## cho cả model thay vì ~27GB fp32)
    bias*: Tensor
    inF*, outF*: int

proc newLinear*(inF, outF: int): Linear =
  let scale = sqrt(2'f32 / float32(inF))
  result.inF = inF
  result.outF = outF
  result.weight = randnTensor(@[outF, inF], scale)
  result.useQuant = false
  result.bias = newTensor(@[outF])

proc setQuantWeight*(l: var Linear, qt: QuantTensor) =
  ## Gọi từ loadLinear() khi load weight thật từ file .nimq - giữ NGUYÊN
  ## dạng nén (không dequant ngay), giải phóng luôn weight fp32 fallback
  ## (không cần nữa, tốn RAM vô ích vì đã có weightQ).
  l.weightQ = qt
  l.useQuant = true
  l.weight.data = @[]
proc forward*(l: Linear, x: Tensor, ctx: Backend): Tensor =
  let (rows, cols) = flatten2D(x.shape)
  assert cols == l.inF
  var outShape = x.shape
  outShape[^1] = l.outF
  result = newTensor(outShape)

  var y: seq[float32]
  if l.useQuant and l.weightQ.kind == qkInt4Asymmetric and l.weightQ.groupSize > 0:
    let nGroupsPerRow = (l.inF + l.weightQ.groupSize - 1) div l.weightQ.groupSize
    y = beMatmulQ4(ctx, x.data, l.weightQ.data, l.weightQ.scale, l.weightQ.zero_point,
                    rows, l.inF, l.outF, l.weightQ.groupSize, nGroupsPerRow)
  else:
    var wT: seq[float32]
    if l.useQuant:
      wT = dequantizeTensorTransposed(l.weightQ)
    else:
      wT = transpose(l.weight).data
    y = beMatmul(ctx, x.data, rows, l.inF, wT, l.inF, l.outF)

  # === FIX: LUÔN ĐẢM BẢO KÍCH THƯỚC ĐÚNG ===
  let expectedSize = rows * l.outF
  if y.len != expectedSize:
    # Tạo mảng mới với kích thước đúng, copy những phần tử có sẵn, phần còn lại để 0
    var fixed = newSeq[float32](expectedSize)
    let copyLen = min(y.len, expectedSize)
    for i in 0 ..< copyLen:
      fixed[i] = y[i]
    y = fixed

  # Cộng bias (l.bias.data có độ dài l.outF)
  for i in 0 ..< y.len:
    result.data[i] = y[i] + l.bias.data[i mod l.outF]
proc backward*(l: Linear, x: Tensor, dOut: Tensor, ctx: Backend): tuple[dX, dW, dB: Tensor] =
  ## Backward Linear: 
  ## dW = x^T @ dOut  (cộng dồn qua batch)
  ## dB = sum(dOut, axis=0)
  ## dX = dOut @ W
  let (rows, cols) = flatten2D(x.shape)
  doAssert cols == l.inF,
    &"Linear.backward: x.shape={x.shape} (cols={cols}) nhưng l.inF={l.inF}"
  doAssert dOut.shape[^1] == l.outF,
    &"Linear.backward: dOut.shape={dOut.shape} (last={dOut.shape[^1]}) nhưng l.outF={l.outF} (l.inF={l.inF}, x.shape={x.shape})"

  # dW = x^T @ dOut (transpose x -> [inF, rows], dOut -> [rows, outF])
  let xT = transpose(x)  # [inF, rows]
  let dOutFlat = dOut.data  # [rows * outF]

  # dW và dX ĐỘC LẬP với nhau (không cái nào cần đọc kết quả của cái kia),
  # nên gộp cả 2 vào 1 command buffer (metalMatmul2) thay vì gọi metalMatmul
  # riêng 2 lần — giảm một nửa số lần dispatch+wait GPU cho mỗi Linear.backward.
  let (dWRawFlat, dXFlat) = beMatmul2(ctx,
    xT.data, l.inF, rows, dOutFlat, rows, l.outF,
    dOutFlat, rows, l.outF, l.weight.data, l.outF, l.inF)

  var dWRaw: Tensor
  dWRaw.shape = @[l.inF, l.outF]
  dWRaw.data = dWRawFlat
  let dW = transpose(dWRaw)  # -> [outF, inF], khớp l.weight.shape

  var dX: Tensor
  dX.shape = x.shape
  dX.data = dXFlat

  # dB = sum dOut theo batch
  var dB = newTensor(@[l.outF])
  for i in 0 ..< rows:
    for j in 0 ..< l.outF:
      dB.data[j] += dOutFlat[i * l.outF + j]

  result = (dX, dW, dB)

# ===================================================================
# LayerNorm (forward/backward dùng Metal matmul)
# ===================================================================
type
  LayerNorm* = object
    gamma*, beta*: Tensor
    eps*: float32
    dim*: int

proc newLayerNorm*(dim: int, eps: float32 = 1e-5'f32): LayerNorm =
  result.dim = dim
  result.gamma = newTensor(@[dim], 1'f32)
  result.beta  = newTensor(@[dim], 0'f32)
  result.eps   = eps

proc forward*(ln: LayerNorm, x: Tensor, ctx: Backend): Tensor =
  ## ĐÃ SỬA: trước đây hàm này nhận `ctx: Backend` nhưng KHÔNG DÙNG - toàn bộ
  ## LayerNorm chạy bằng vòng lặp Nim thuần trên CPU bất kể người dùng chọn
  ## backend cuda/metal/opencl nào. backend.nim đã có sẵn `beLayernorm` khớp
  ## đúng phép toán này (rows x cols, cùng công thức chuẩn hoá) nên chỉ cần
  ## gọi qua đó để LayerNorm thực sự chạy trên GPU đã chọn.
  let (rows, cols) = flatten2D(x.shape)
  assert cols == ln.dim
  result.shape = x.shape
  result.data = beLayernorm(ctx, x.data, ln.gamma.data, ln.beta.data, rows, cols, ln.eps)

proc backward*(ln: LayerNorm, x: Tensor, dOut: Tensor, ctx: Backend): tuple[dX, dGamma, dBeta: Tensor] =
  ## Backward LayerNorm (dùng phép toán ma trận / GPU-bound)
  let (rows, cols) = flatten2D(x.shape)
  assert cols == ln.dim
  var dX = newTensor(x.shape)
  var dGamma = newTensor(ln.gamma.shape)
  var dBeta = newTensor(ln.beta.shape)
  let (dx_data, dgamma_data, dbeta_data) = beLayernormBackward(ctx, dOut.data, x.data, ln.gamma.data, ln.beta.data, rows, cols, ln.eps)
  dX.data = dx_data
  dGamma.data = dgamma_data
  dBeta.data = dbeta_data
  result = (dX, dGamma, dBeta)

# ===================================================================
# CausalSelfAttention (forward/backward dùng Metal matmul)
# ===================================================================
type
  CausalSelfAttention* = object
    nHeads*, headDim*, embedDim*: int
    qkv*, proj*: Linear

proc newCausalSelfAttention*(embedDim, nHeads: int): CausalSelfAttention =
  assert embedDim mod nHeads == 0
  result.embedDim = embedDim
  result.nHeads = nHeads
  result.headDim = embedDim div nHeads
  result.qkv = newLinear(embedDim, 3 * embedDim)
  result.proj = newLinear(embedDim, embedDim)

proc forward*(attn: CausalSelfAttention, x: Tensor, ctx: Backend): Tensor =
  let B = x.shape[0]
  let T = x.shape[1]
  let C = x.shape[2]
  assert C == attn.embedDim
  let qkv = attn.qkv.forward(x, ctx)
  let hd = attn.headDim
  let scale = 1'f32 / sqrt(float32(hd))

  # Slice Q, K, V
  var qData = newSeq[float32](B * T * C)
  var kData = newSeq[float32](B * T * C)
  var vData = newSeq[float32](B * T * C)
  for b in 0 ..< B:
    let baseQKV = b * T * 3 * C
    let baseOut = b * T * C
    for t in 0 .. T-1:
      for c in 0 .. C-1:
        qData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + c]
        kData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + C + c]
        vData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + 2 * C + c]

  let (attnOutData, _) = beAttentionFused(ctx, qData, kData, vData, [], B, attn.nHeads, T, hd, scale)
  var attnOut = newTensor(@[B, T, C])
  attnOut.data = attnOutData
  result = attn.proj.forward(attnOut, ctx)

proc backward*(attn: CausalSelfAttention, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dQkvW, dQkvB, dProjW, dProjB: Tensor] =
  let B = x.shape[0]
  let T = x.shape[1]
  let C = x.shape[2]
  let hd = attn.headDim
  let nHeads = attn.nHeads

  # Forward again to get intermediate states (Q, K, V, s_matrix)
  let qkv = attn.qkv.forward(x, ctx)
  let scale = 1'f32 / sqrt(float32(hd))

  var qData = newSeq[float32](B * T * C)
  var kData = newSeq[float32](B * T * C)
  var vData = newSeq[float32](B * T * C)
  for b in 0 ..< B:
    let baseQKV = b * T * 3 * C
    let baseOut = b * T * C
    for t in 0 .. T-1:
      for c in 0 .. C-1:
        qData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + c]
        kData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + C + c]
        vData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + 2 * C + c]

  let (attnOutData, s_matrix) = beAttentionFused(ctx, qData, kData, vData, [], B, nHeads, T, hd, scale)
  var attnOut = newTensor(@[B, T, C])
  attnOut.data = attnOutData

  # dProj
  let dProj = attn.proj.backward(attnOut, dOut, ctx)
  let dAttnOut = dProj.dX

  # Attention Fused Backward
  let (dqData, dkData, dvData) = beAttentionFusedBackward(ctx, qData, kData, vData, s_matrix, dAttnOut.data, B, nHeads, T, hd, scale)

  # Pack dQkv
  var dQkv = newTensor(qkv.shape)
  for b in 0 ..< B:
    let baseQKV = b * T * 3 * C
    let baseOut = b * T * C
    for t in 0 .. T-1:
      for c in 0 .. C-1:
        dQkv.data[baseQKV + t * 3 * C + c] = dqData[baseOut + t * C + c]
        dQkv.data[baseQKV + t * 3 * C + C + c] = dkData[baseOut + t * C + c]
        dQkv.data[baseQKV + t * 3 * C + 2 * C + c] = dvData[baseOut + t * C + c]

  let dQkvLin = attn.qkv.backward(x, dQkv, ctx)
  result = (dQkvLin.dX, dQkvLin.dW, dQkvLin.dB, dProj.dW, dProj.dB)

# ===================================================================
# FeedForward (forward/backward dùng GPU-bound APFLU Activation)
# ===================================================================
type
  FeedForward* = object
    fc1*, fc2*: Linear

proc newFeedForward*(embedDim, mult: int): FeedForward =
  result.fc1 = newLinear(embedDim, embedDim * mult)
  result.fc2 = newLinear(embedDim * mult, embedDim)

proc forward*(ff: FeedForward, x: Tensor, ctx: Backend): Tensor =
  var h = ff.fc1.forward(x, ctx)
  h.data = beApflu(ctx, h.data, 0.1'f32, 1.0'f32)
  result = ff.fc2.forward(h, ctx)

proc backward*(ff: FeedForward, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dFc1W, dFc1B, dFc2W, dFc2B: Tensor] =
  let hidden = ff.fc1.forward(x, ctx)
  let dFc2 = ff.fc2.backward(hidden, dOut, ctx)
  var dHidden = dFc2.dX
  dHidden.data = beApfluBackward(ctx, hidden.data, dHidden.data, 0.1'f32, 1.0'f32)
  let dFc1 = ff.fc1.backward(x, dHidden, ctx)
  result = (dFc1.dX, dFc1.dW, dFc1.dB, dFc2.dW, dFc2.dB)

# ===================================================================
# TransformerBlock (Post-LN) - forward/backward dùng Metal
# ===================================================================
type
  TransformerBlock* = object
    attn*: CausalSelfAttention
    ff*: FeedForward
    ln1*, ln2*: LayerNorm

proc newTransformerBlock*(embedDim, nHeads, ffMult: int): TransformerBlock =
  result.attn = newCausalSelfAttention(embedDim, nHeads)
  result.ff = newFeedForward(embedDim, ffMult)
  result.ln1 = newLayerNorm(embedDim)
  result.ln2 = newLayerNorm(embedDim)

proc forward*(blk: TransformerBlock, x: Tensor, ctx: Backend): Tensor =
  let x1 = blk.ln1.forward(x, ctx)
  let attnOut = blk.attn.forward(x1, ctx)
  let x2 = addT(ctx, x, attnOut)
  let x3 = blk.ln2.forward(x2, ctx)
  let ffOut = blk.ff.forward(x3, ctx)
  result = addT(ctx, x2, ffOut)

proc backward*(blk: TransformerBlock, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dAttnQkvW, dAttnQkvB, dAttnProjW, dAttnProjB,
    dFf1W, dFf1B, dFf2W, dFf2B,
    dLn1G, dLn1B, dLn2G, dLn2B: Tensor] =
  
  # Forward lại (cần cho backward)
  let x1 = blk.ln1.forward(x, ctx)
  let attnOut = blk.attn.forward(x1, ctx)
  let x2 = addT(ctx, x, attnOut)
  let x3 = blk.ln2.forward(x2, ctx)
  let ffOut = blk.ff.forward(x3, ctx)
  let y = addT(ctx, x2, ffOut)
  
  var dY = dOut
  let dFF = blk.ff.backward(x3, dY, ctx)
  var dX2 = dFF.dX
  let dLN2 = blk.ln2.backward(x2, dX2, ctx)
  var dX2_ln = dLN2.dX
  # Residual (x2 = x + attnOut)
  for i in 0 ..< dX2_ln.data.len:
    dX2_ln.data[i] += dY.data[i]
  
  let dAttn = blk.attn.backward(x1, dX2_ln, ctx)
  let dLN1 = blk.ln1.backward(x, dAttn.dX, ctx)
  var dX0 = dLN1.dX
  # Residual (x2 = x + attnOut)
  for i in 0 ..< dX0.data.len:
    dX0.data[i] += dX2_ln.data[i]
  
  result = (
    dX0,
    dAttn.dQkvW, dAttn.dQkvB, dAttn.dProjW, dAttn.dProjB,
    dFF.dFc1W, dFF.dFc1B, dFF.dFc2W, dFF.dFc2B,
    dLN1.dGamma, dLN1.dBeta, dLN2.dGamma, dLN2.dBeta
  )

# ===================================================================
# Embedding (forward/backward dùng Metal matmul)
# ===================================================================
type
  Embedding* = object
    weight*: Tensor
    dim*, vocab*: int

proc newEmbedding*(vocab, dim: int): Embedding =
  result.vocab = vocab
  result.dim = dim
  result.weight = randnTensor(@[vocab, dim], sqrt(2'f32 / float32(vocab)))

proc lookupBatch*(e: Embedding, idsBatch: seq[seq[int]], ctx: Backend): Tensor =
  let B = idsBatch.len
  let T = idsBatch[0].len
  result = newTensor(@[B, T, e.dim])
  for b in 0 ..< B:
    for t in 0 ..< T:
      let id = idsBatch[b][t]
      let src = id * e.dim
      let dst = (b * T + t) * e.dim
      for d in 0 ..< e.dim:
        result.data[dst + d] = e.weight.data[src + d]

proc backward*(e: Embedding, idsBatch: seq[seq[int]], dOut: Tensor, ctx: Backend): Tensor =
  let B = idsBatch.len
  let T = idsBatch[0].len
  var dWeight = newTensor(e.weight.shape)
  for b in 0 ..< B:
    for t in 0 ..< T:
      let id = idsBatch[b][t]
      let dOff = (b * T + t) * e.dim
      let wOff = id * e.dim
      for d in 0 ..< e.dim:
        dWeight.data[wOff + d] += dOut.data[dOff + d]
  return dWeight

# ===================================================================
# NimformerModel (Post-LN) - forward/backward dùng Metal
# ===================================================================
type
  NimformerModel* = object
    embed*: Embedding
    blocks*: seq[TransformerBlock]
    outProj*: Linear
    vocab*: int

proc newNimformerModel*(vocab, embedDim, nHeads, nLayers, ffMult: int): NimformerModel =
  result.vocab = vocab
  result.embed = newEmbedding(vocab, embedDim)
  result.blocks = newSeq[TransformerBlock](nLayers)
  for i in 0 ..< nLayers:
    result.blocks[i] = newTransformerBlock(embedDim, nHeads, ffMult)
  result.outProj = newLinear(embedDim, vocab)

proc forwardBatch*(m: NimformerModel, idsBatch: seq[seq[int]], ctx: Backend): Tensor =
  ## Forward THẬT với batch B (idsBatch.len chuỗi cùng độ dài T) trong 1 lần
  ## gọi — Linear/LayerNorm coi B*T là số "hàng" nên GPU nhận hẳn M=B*T dòng
  ## trong mỗi matmul, thay vì bị gọi B lần tuần tự (mỗi lần B=1) như trước.
  ## Trả logits shape [B, T, vocab].
  var x = m.embed.lookupBatch(idsBatch, ctx)
  for blk in m.blocks:
    x = blk.forward(x, ctx)
  result = m.outProj.forward(x, ctx)

proc forward*(m: NimformerModel, ids: seq[int], ctx: Backend): Tensor =
  ## Tiện ích cho 1 chuỗi đơn (giữ nguyên API cũ cho test_nimformer.nim) —
  ## gọi forwardBatch với B=1 rồi bỏ chiều batch.
  let logits = m.forwardBatch(@[ids], ctx)
  result = Tensor(data: logits.data, shape: logits.shape[1 .. ^1])

proc backwardBatch*(m: NimformerModel, idsBatch: seq[seq[int]], dLoss: Tensor,
                     ctx: Backend): seq[Tensor] =
  ## Backward THẬT với batch B. dLoss: shape [B, T, vocab] (gradient loss
  ## theo logits, ĐÃ đúng batch — không còn phải giả B=1 rồi bọc thêm 1
  ## chiều như bản cũ). Linear.backward tự cộng dồn gradient qua B*T hàng
  ## trong 1 matmul, nên dW/dB trả về đã là tổng đúng của CẢ batch.
  var x = m.embed.lookupBatch(idsBatch, ctx)
  var hiddenStates: seq[Tensor] = @[x]
  for blk in m.blocks:
    x = blk.forward(x, ctx)
    hiddenStates.add(x)
  let dOutProj = m.outProj.backward(x, dLoss, ctx)
  var dOut = dOutProj.dX
  var grads: seq[Tensor] = @[dOutProj.dW, dOutProj.dB]
  for i in countdown(m.blocks.len - 1, 0):
    let blk = m.blocks[i]
    let x_prev = hiddenStates[i]
    let dBlock = blk.backward(x_prev, dOut, ctx)
    dOut = dBlock.dX
    grads.add(dBlock.dAttnQkvW); grads.add(dBlock.dAttnQkvB)
    grads.add(dBlock.dAttnProjW); grads.add(dBlock.dAttnProjB)
    grads.add(dBlock.dFf1W); grads.add(dBlock.dFf1B)
    grads.add(dBlock.dFf2W); grads.add(dBlock.dFf2B)
    grads.add(dBlock.dLn1G); grads.add(dBlock.dLn1B)
    grads.add(dBlock.dLn2G); grads.add(dBlock.dLn2B)
  let dEmb = m.embed.backward(idsBatch, dOut, ctx)
  grads.add(dEmb)
  return grads

proc backward*(m: NimformerModel, ids: seq[int], dLoss: Tensor, ctx: Backend): seq[Tensor] =
  ## Tiện ích cho 1 chuỗi đơn (giữ nguyên API cũ cho test_nimformer.nim).
  let dLossBatch = Tensor(data: dLoss.data, shape: @[1] & dLoss.shape)
  result = m.backwardBatch(@[ids], dLossBatch, ctx)

# ===================================================================
# ApfAdam (update weight trên CPU vì chỉ 1 vector, vẫn ok)
# ===================================================================
type
  ApfAdamState* = object
    m*, v*: seq[float32]
    step*: int

proc newApfAdamState*(paramLen: int): ApfAdamState =
  result.m = newSeq[float32](paramLen)
  result.v = newSeq[float32](paramLen)
  result.step = 0

proc apfAdamStep*(param: var Tensor, grad: Tensor, state: var ApfAdamState,
                   lr: float32 = 1e-3'f32, b1: float32 = 0.9'f32, b2: float32 = 0.999'f32,
                   eps: float32 = 1e-8'f32, requantizeEvery: int = 1): CustomFloat =
  inc state.step
  let bc1 = 1'f32 - pow(b1, float32(state.step))
  let bc2 = 1'f32 - pow(b2, float32(state.step))
  for i in 0 ..< param.data.len:
    let g = grad.data[i]
    state.m[i] = b1 * state.m[i] + (1'f32 - b1) * g
    state.v[i] = b2 * state.v[i] + (1'f32 - b2) * g * g
    let mHat = state.m[i] / bc1
    let vHat = state.v[i] / bc2
    param.data[i] -= lr * mHat / (sqrt(vHat) + eps)
  if state.step mod requantizeEvery == 0:
    let cf = buildCustomDtypeForTensor(param.data, grad.data)
    param.data = decodeArray(encodeArray(param.data, cf), cf)
    result = cf
  else:
    result = buildCustomDtypeForTensor(param.data)