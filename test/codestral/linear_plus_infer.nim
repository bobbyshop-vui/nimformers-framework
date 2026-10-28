## linear_plus_infer.nim
## Standalone incremental Linear++ inference path.
## Does NOT use the Linear++ implementation from test_nimformer.nim.
## Keeps causal sufficient statistics instead of a T x T attention matrix.

import std/math
import std/tables
import backend
import nimformer
import fp4_gemv_gpu

const LP_INFER_EPS* = 1e-6'f32

type
  LinearPlusInferState* = object
    nHeads*: int
    headDim*: int
    embedDim*: int
    ## S: [head][keyDim][valueDim] = sum(phi(k) outer v)
    s*: seq[float32]
    ## z: [head][keyDim] = sum(phi(k))
    z*: seq[float32]
    steps*: int

proc newLinearPlusInferState*(embedDim, nHeads: int): LinearPlusInferState =
  if embedDim <= 0 or nHeads <= 0 or embedDim mod nHeads != 0:
    raise newException(ValueError,
      "LinearPlusInferState: embedDim phai chia het cho nHeads")
  result.embedDim = embedDim
  result.nHeads = nHeads
  result.headDim = embedDim div nHeads
  result.s = newSeq[float32](nHeads * result.headDim * result.headDim)
  result.z = newSeq[float32](nHeads * result.headDim)
  result.steps = 0

proc resetLinearPlusInferState*(state: var LinearPlusInferState) =
  for i in 0 ..< state.s.len:
    state.s[i] = 0'f32
  for i in 0 ..< state.z.len:
    state.z[i] = 0'f32
  state.steps = 0

proc lpPhi(x: float32): float32 {.inline.} =
  if x > 0'f32: x + 1'f32 else: exp(x)

proc forwardLinearFp4InferStep*(lin: var Linear, name: string,
                                 x: Tensor, ctx: Backend): Tensor =
  ## Inference-only FP4 GEMV for one token. Weight MUST already be resident in
  ## gFp4CacheGpu; avoids Linear.forward() fallback that dequantizes large FP4
  ## matrices on every token. Adds the Linear bias exactly once.
  if x.data.len != lin.inF:
    raise newException(ValueError,
      "forwardLinearFp4InferStep: input dim khong khop " & name)
  if name notin gFp4CacheGpu or not gFp4CacheGpu[name].residentReady:
    ## Tren Metal KHONG duoc silently roi ve Linear.forward(): nhanh fallback
    ## co the dequant toan bo FP4 weight moi token va lam cham hang chuc giay.
    ## prepareHybridFp4Weights() phai chuan bi tat ca matrix lon truoc decode.
    if ctx.kind == bkMetal:
      raise newException(CatchableError,
        "[infer] FP4 weight chua resident cho " & name &
        "; dung fallback Linear.forward se rat cham")
    return lin.forward(x, ctx)
  let yData = fp4GemvGpu(ctx, name, x.data, 1)
  if yData.len != lin.outF:
    raise newException(ValueError,
      "forwardLinearFp4InferStep: output len khong khop " & name &
      " (got=" & $yData.len & ", expected=" & $lin.outF & ")")
  var outShape = x.shape
  outShape[^1] = lin.outF
  result = newTensor(outShape)
  for i in 0 ..< lin.outF:
    result.data[i] = yData[i] + lin.bias.data[i]

proc forwardLinearPlusInferStep*(qkvProj, outProj: var Linear,
                                 qkvName, outName: string,
                                 nHeads: int, state: var LinearPlusInferState,
                                 x: Tensor, ctx: Backend): Tensor =
  if x.shape.len != 3 or x.shape[0] != 1 or x.shape[1] != 1:
    raise newException(ValueError,
      "forwardLinearPlusInferStep: chi chap nhan input [1,1,C]")
  let C = x.shape[2]
  if C != qkvProj.inF or qkvProj.outF != 3 * C:
    raise newException(ValueError,
      "forwardLinearPlusInferStep: QKV shape khong khop embedDim")
  if state.embedDim != C or state.nHeads != nHeads or state.headDim * nHeads != C:
    raise newException(ValueError,
      "forwardLinearPlusInferStep: state shape khong khop block Attention")

  let proj = forwardLinearFp4InferStep(qkvProj, qkvName, x, ctx)
  let D = state.headDim
  var y = newTensor(@[1, 1, C])

  ## proj = [Q | K | V], moi phan C phan tu.
  for h in 0 ..< nHeads:
    let base = h * D
    let zOff = base
    let sOff = h * D * D

    var qf = newSeq[float32](D)
    var kf = newSeq[float32](D)
    for d in 0 ..< D:
      qf[d] = lpPhi(proj.data[base + d])
      kf[d] = lpPhi(proj.data[C + base + d])
      state.z[zOff + d] += kf[d]

    ## S += phi(k_t) outer v_t
    for kd in 0 ..< D:
      let kval = kf[kd]
      let row = sOff + kd * D
      for vd in 0 ..< D:
        state.s[row + vd] += kval * proj.data[2 * C + base + vd]

    var den = 0'f32
    for kd in 0 ..< D:
      den += qf[kd] * state.z[zOff + kd]
    let invDen = 1'f32 / (den + LP_INFER_EPS)

    for vd in 0 ..< D:
      var num = 0'f32
      for kd in 0 ..< D:
        num += qf[kd] * state.s[sOff + kd * D + vd]
      y.data[base + vd] = num * invDen

  inc state.steps
  result = forwardLinearFp4InferStep(outProj, outName, y, ctx)


proc forwardFeedForwardFp4InferStep*(ff: var FeedForward, fc1Name, fc2Name: string,
                                     x: Tensor, ctx: Backend): Tensor =
  ## FFN inference path for A-blocks. Both big matrices stay FP4-resident;
  ## this avoids per-token full dequantizeTensorTransposed() in Linear.forward.
  var h = forwardLinearFp4InferStep(ff.fc1, fc1Name, x, ctx)
  h.data = beApflu(ctx, h.data, 0.1'f32, 1.0'f32)
  result = forwardLinearFp4InferStep(ff.fc2, fc2Name, h, ctx)
