## Mamba-3 SISO 187M pipeline forked from test/codestral.
## R = Mamba-3 SISO + gated MLP; P/G add the existing Linear++ sidecar.
## Projection, LoRA, optimizer and checkpoint helpers reuse Codestral formats.
## See README.md for supported paths and validation limits.

import std/[math, os, strutils, tables, strformat]
import backend
import nimformer
import mamba2_ssm_gpu
import mamba3_real
import test_nimformer
import prefill_profile   # BYBY_PREFILL_PROFILE

type
  HybridBlockKernelCache* = object
    xNorm*: Tensor
    mlpInput*, mlpXNorm*, mlpProj*, mlpGated*: Tensor
    mamba*: Mamba3Cache
    hasLinearPlus*: bool
    hasFuseGate*: bool
    linearPlusGpu*: bool
    attnProjection*, linearPlusY*, mambaY*, linearPlusOut*, gateLogit*: Tensor
    gate*: seq[float32]
    ## Terminal recurrent state exported by inference prefill. Mamba conv state
    ## is [cDim, K-1], SSM state is [head*dim, dState]. Linear++ state is
    ## retained in scan layout here and transposed when imported to decode.
    terminalConvState*, terminalSsmState*: seq[float32]
    linearPlusZState*, linearPlusScanState*: seq[float32]

proc sliceTensor(buf: seq[float32]; off, n: int; shape: seq[int]): Tensor =
  result = Tensor(shape: shape, data: newSeq[float32](n))
  result.data.blit(0, buf, off, n)

proc forwardMambaMixerKernel(mamba: Mamba3Real; x: Tensor; ctx: Backend):
    tuple[y: Tensor, cache: Mamba3Cache] = forwardMamba3Real(mamba, x, ctx)

proc terminalMambaStates(cache: Mamba3Cache): tuple[conv, ssm: seq[float32]] =
  result.conv = cache.terminal.aux
  result.ssm = cache.terminal.ssmState

proc forwardMambaMixerInfer(mamba: Mamba3Real; x: Tensor; ctx: Backend):
    tuple[y: Tensor, conv, ssm: seq[float32]] =
  let fw = forwardMamba3Real(mamba, x, ctx)
  result = (fw.y, fw.cache.terminal.aux, fw.cache.terminal.ssmState)

type
  LoraGradSpec* = object
    ## LoRA adapter of one Linear: A [r, inF], B [outF, r] (row-major, like LoraAdapter).
    A*, B*: seq[float32]
    r*: int
    scale*: float32

## LoRA-aware backward (LoRA training): trainStep registers adapters by parameter NAME
## (e.g. "block7.mambaReal.inProj.weight") before backwardBlock and removes them afterwards. For
## a registered Linear the backward does NOT compute the full dW [outF, inF] (19.5 GMAC + reading/writing
## ~300MB for inProj) and then reduce it with gradToAB; it computes directly:
##   dB = s * dY^T (X A^T)      [outF, r]
##   dA = s * (dY B)^T X        [r, inF]
## - mathematically the same as gradToAB(dW = dY^T X). Grads go to grads[key & "::loraA"]
## and grads[key & "::loraB"]; dX is still dY @ W_eff (weights with LoRA merged) as before.
## An empty table (the default, every infer/other path) -> keeps the old backward path.
var gLoraGradRequest* = initTable[string, LoraGradSpec]()

proc linearBackwardLoraAware*(l: Linear; x, dOut: Tensor; spec: LoraGradSpec;
                              ctx: Backend): tuple[dX, dA, dB, dBias: Tensor] =
  let (rows, cols) = flatten2D(x.shape)
  let inF = l.inF
  let outF = l.outF
  let r = spec.r
  doAssert cols == inF and dOut.shape[^1] == outF and dOut.data.len == rows * outF,
    &"LoRA-aware backward: x.shape={x.shape} dOut.shape={dOut.shape} do not match Linear {inF}->{outF}"
  doAssert l.weight.data.len == outF * inF,
    "LoRA-aware backward needs fp32 weights with LoRA merged in l.weight.data"
  doAssert r > 0 and spec.A.len == r * inF and spec.B.len == outF * r,
    "LoRA-aware backward: A/B have the wrong size"
  # dX = dOut @ W_eff  ([rows,outF] @ [outF,inF]) - like the dX branch of Linear.backward.
  result.dX = Tensor(shape: x.shape,
                     data: beMatmul(ctx, dOut.data, rows, outF, l.weight.data, outF, inF))
  # XAt[t,k] = sum_i x[t,i] * A[k,i]
  var xat = newSeq[float64](rows * r)
  for t in 0 ..< rows:
    let xr = t * inF
    for k in 0 ..< r:
      let ar = k * inF
      var acc = 0.0
      for i in 0 ..< inF: acc += float64(x.data[xr + i]) * float64(spec.A[ar + i])
      xat[t * r + k] = acc
  # dB[o,k] = s * sum_t dOut[t,o] * XAt[t,k];  dYB[t,k] = sum_o dOut[t,o] * B[o,k]
  var dB64 = newSeq[float64](outF * r)
  var dyb = newSeq[float64](rows * r)
  var dBias64 = newSeq[float64](outF)
  for t in 0 ..< rows:
    let dr = t * outF
    for o in 0 ..< outF:
      let d = float64(dOut.data[dr + o])
      if d == 0.0: continue
      dBias64[o] += d
      for k in 0 ..< r:
        dB64[o * r + k] += d * xat[t * r + k]
        dyb[t * r + k] += d * float64(spec.B[o * r + k])
  # dA[k,i] = s * sum_t dYB[t,k] * x[t,i]
  var dA64 = newSeq[float64](r * inF)
  for t in 0 ..< rows:
    let xr = t * inF
    for k in 0 ..< r:
      let c = dyb[t * r + k]
      if c == 0.0: continue
      let ar = k * inF
      for i in 0 ..< inF: dA64[ar + i] += c * float64(x.data[xr + i])
  let s = float64(spec.scale)
  result.dA = newTensor(@[r, inF])
  for j in 0 ..< dA64.len: result.dA.data[j] = float32(dA64[j] * s)
  result.dB = newTensor(@[outF, r])
  for j in 0 ..< dB64.len: result.dB.data[j] = float32(dB64[j] * s)
  result.dBias = newTensor(@[outF])
  for j in 0 ..< outF: result.dBias.data[j] = float32(dBias64[j])

proc linearBackwardMaybeLora(l: Linear; key: string; x, dOut: Tensor; ctx: Backend;
                             grads: var Table[string, Tensor]; biasKey: string): Tensor =
  ## A Linear with a registered adapter -> LoRA-aware (grads[key::loraA/::loraB]); otherwise
  ## the full Linear.backward as before (grads[key] = dW).
  if key in gLoraGradRequest:
    let g = linearBackwardLoraAware(l, x, dOut, gLoraGradRequest[key], ctx)
    grads[key & "::loraA"] = g.dA
    grads[key & "::loraB"] = g.dB
    grads[biasKey] = g.dBias
    result = g.dX
  else:
    let (dX, dW, dB) = l.backward(x, dOut, ctx)
    grads[key] = dW
    grads[biasKey] = dB
    result = dX

proc backwardMambaMixerKernel(mamba: Mamba3Real; x: Tensor;
    cache: Mamba3Cache; dOut: Tensor; ctx: Backend; prefix: string;
    grads: var Table[string, Tensor]): Tensor =
  ## Explicit transpose of forwardMambaMixerKernel; every trainable Mamba
  ## parameter receives a gradient, including output projection and gated norm.
  let normed = cache.inner
  let dNormed = linearBackwardMaybeLora(mamba.outProj, prefix & ".outProj.weight",
    normed, dOut, ctx, grads, prefix & ".outProj.bias")
  let dProj = backwardMamba3Core(mamba, cache, dNormed, ctx)
  result = linearBackwardMaybeLora(mamba.inProj, prefix & ".inProj.weight",
    x, dProj, ctx, grads, prefix & ".inProj.bias")

proc blockRmsForward(x, gamma: Tensor; eps: float32; ctx: Backend): Tensor =
  doAssert x.shape.len >= 2 and x.shape[^1] > 0,
    "R/P block RMSNorm: input must have a non-empty feature dimension"
  let C = x.shape[^1]
  doAssert gamma.data.len == C, "R/P block RMSNorm: gamma size must match hidden size"
  let rows = x.data.len div C
  result = newTensor(x.shape)
  if isGpuCtx(ctx):
    result.data = rmsNormForwardGpu(ctx, x.data, gamma.data, rows, C, eps)
    return
  for row in 0 ..< rows:
    let base = row * C
    var meanSq = 0'f32
    for c in 0 ..< C:
      let v = x.data[base + c]
      meanSq += v * v
    let invRms = 1'f32 / sqrt(meanSq / float32(C) + eps)
    for c in 0 ..< C:
      result.data[base + c] = x.data[base + c] * invRms * gamma.data[c]

proc blockRmsBackward(x, gamma, dNorm: Tensor; eps: float32; ctx: Backend):
    tuple[dX, dGamma: Tensor] =
  doAssert x.shape == dNorm.shape,
    "R/P block RMSNorm backward: dNorm shape must match input"
  let C = x.shape[^1]
  doAssert gamma.data.len == C, "R/P block RMSNorm backward: gamma size mismatch"
  let rows = x.data.len div C
  result.dX = newTensor(x.shape)
  result.dGamma = newTensor(gamma.shape)
  if isGpuCtx(ctx):
    let grads = rmsNormBackwardGpu(ctx, x.data, gamma.data, dNorm.data, rows, C, eps)
    result.dX.data = grads.dX
    result.dGamma.data = grads.dGamma
    return
  for row in 0 ..< rows:
    let base = row * C
    var meanSq = 0'f32
    for c in 0 ..< C:
      let v = x.data[base + c]
      meanSq += v * v
    let invRms = 1'f32 / sqrt(meanSq / float32(C) + eps)
    var dot = 0'f32
    for c in 0 ..< C:
      let i = base + c
      result.dGamma.data[c] += dNorm.data[i] * x.data[i] * invRms
      dot += dNorm.data[i] * gamma.data[c] * x.data[i]
    let coeff = dot * invRms * invRms * invRms / float32(C)
    for c in 0 ..< C:
      let i = base + c
      result.dX.data[i] = invRms * gamma.data[c] * dNorm.data[i] - x.data[i] * coeff

proc forwardMlp*(m: Mamba3Real; x:Tensor; ctx:Backend):
    tuple[y, normed, proj, gated:Tensor] =
  result.normed=blockRmsForward(x,m.mlpNorm,m.normEps,ctx)
  result.proj=m.mlpFc1.forward(result.normed,ctx)
  let F=m.mlpFc2.inF
  let rows=x.data.len div m.hiddenSize
  result.gated=newTensor(@[x.shape[0],x.shape[1],F])
  for row in 0..<rows:
    for d in 0..<F:
      let v=result.proj.data[row*2*F+d]
      let g=result.proj.data[row*2*F+F+d]
      result.gated.data[row*F+d]=v*g/(1'f32+exp(-g))
  result.y=addT(ctx,x,m.mlpFc2.forward(result.gated,ctx))

proc finishMlp(m: Mamba3Real; y: var Tensor; cache:var HybridBlockKernelCache; ctx:Backend) =
  let fw=forwardMlp(m,y,ctx)
  cache.mlpInput=y;cache.mlpXNorm=fw.normed;cache.mlpProj=fw.proj;cache.mlpGated=fw.gated
  y=fw.y

proc backwardMlp(m:Mamba3Real; cache:HybridBlockKernelCache; dOut:Tensor;
    ctx:Backend; prefix:string; grads:var Table[string,Tensor]):Tensor =
  let root=prefix & ".mambaReal."
  let dg=linearBackwardMaybeLora(m.mlpFc2,root & "mlpFc2.weight",cache.mlpGated,dOut,ctx,grads,root & "mlpFc2.bias")
  let F=m.mlpFc2.inF
  var dp=newTensor(cache.mlpProj.shape)
  for row in 0..<dOut.data.len div m.hiddenSize:
    for d in 0..<F:
      let v=cache.mlpProj.data[row*2*F+d]
      let g=cache.mlpProj.data[row*2*F+F+d]
      let sig=1'f32/(1'f32+exp(-g))
      dp.data[row*2*F+d]=dg.data[row*F+d]*g*sig
      dp.data[row*2*F+F+d]=dg.data[row*F+d]*v*sig*(1'f32+g*(1'f32-sig))
  let dxn=linearBackwardMaybeLora(m.mlpFc1,root & "mlpFc1.weight",cache.mlpXNorm,dp,ctx,grads,root & "mlpFc1.bias")
  result=addT(ctx,dOut,blockRmsBackward(cache.mlpInput,m.mlpNorm,dxn,m.normEps,ctx).dX)

proc forwardRBlockKernel*(mamba: Mamba3Real; x, normGamma: Tensor;
                          normEps: float32; ctx: Backend;
                          saveBackwardCache = true;
                          exportTerminalState = false):
    tuple[y: Tensor, cache: HybridBlockKernelCache] =
  ## Complete R block: norm, Mamba3Real mixer, and residual.
  ## cache.mambaY = mixer output BEFORE the residual (to measure rms(mambaY)/rms(x)).
  let xNorm = pfExpr("blk.rms", blockRmsForward(x, normGamma, normEps, ctx))
  if exportTerminalState and not saveBackwardCache:
    # Inference (prefill): no backward cache needed -> the mixer does not bring the scratch back to the host.
    let mi = forwardMambaMixerInfer(mamba, xNorm, ctx)
    result.y = pfExpr("blk.residual", addT(ctx, x, mi.y))
    result.cache = HybridBlockKernelCache(xNorm: xNorm, hasLinearPlus: false,
      mambaY: mi.y, terminalConvState: mi.conv, terminalSsmState: mi.ssm)
    finishMlp(mamba,result.y,result.cache,ctx)
    return
  let mixer = forwardMambaMixerKernel(mamba, xNorm, ctx)
  result.y = pfExpr("blk.residual", addT(ctx, x, mixer.y))
  result.cache = HybridBlockKernelCache(xNorm: xNorm,
    mamba: (if saveBackwardCache: mixer.cache else: default(Mamba3Cache)),
    hasLinearPlus: false,
    mambaY: mixer.y)
  if exportTerminalState:
    let terminal = pfExpr("blk.terminalState", terminalMambaStates(mixer.cache))
    result.cache.terminalConvState = terminal.conv
    result.cache.terminalSsmState = terminal.ssm

  finishMlp(mamba,result.y,result.cache,ctx)

proc forwardPBlockKernel*(mamba: Mamba3Real; attn: LinearPlusAttention;
                          x, normGamma: Tensor; normEps: float32; ctx: Backend;
                          saveBackwardCache = true;
                          exportTerminalState = false;
                          fuseGate: ptr Linear = nil):
    tuple[y: Tensor, cache: HybridBlockKernelCache] =
  ## Complete P/G block: shared norm, parallel branches, fixed sum or gated mix.
  let xNorm = pfExpr("blk.rms", blockRmsForward(x, normGamma, normEps, ctx))
  var m, attnProjection, lpY, attnOut: Tensor
  var mambaCache: Mamba3Cache
  var terminalConv, terminalSsm, lpZState, lpScanState: seq[float32]
  var lpGpu = false
  let mResult = forwardMambaMixerKernel(mamba, xNorm, ctx)
  m = mResult.y
  mambaCache = mResult.cache
  attnOut = forwardLinearPlus(attn, xNorm, ctx)
  if exportTerminalState:
    terminalConv = mResult.cache.terminal.aux
    terminalSsm = mResult.cache.terminal.ssmState
  var gateLogit: Tensor
  var gate: seq[float32]
  var branchSum: Tensor
  let pfGate0 = getMonoTimeIfPf()
  if fuseGate != nil:
    var fuseIn = newTensor(@[x.shape[0], x.shape[1], x.shape[2] * 2])
    for row in 0 ..< x.shape[0] * x.shape[1]:
      for c in 0 ..< x.shape[2]:
        fuseIn.data[row * x.shape[2] * 2 + c] = m.data[row * x.shape[2] + c]
        fuseIn.data[row * x.shape[2] * 2 + x.shape[2] + c] =
          attnOut.data[row * x.shape[2] + c]
    gateLogit = fuseGate[].forward(fuseIn, ctx)
    gate = beSigmoid(ctx, gateLogit.data)
    branchSum = newTensor(x.shape)
    for i in 0 ..< branchSum.data.len:
      branchSum.data[i] = gate[i] * m.data[i] + (1'f32 - gate[i]) * attnOut.data[i]
  else:
    branchSum = addT(ctx, m, attnOut)
  pfAddIf("P.gateMix", pfGate0)
  result.y = pfExpr("blk.residual", addT(ctx, x, branchSum))
  result.cache = HybridBlockKernelCache(xNorm: xNorm, mamba: mambaCache,
    hasLinearPlus: true, hasFuseGate: fuseGate != nil, linearPlusGpu: lpGpu,
    attnProjection: attnProjection, linearPlusY: lpY,
    mambaY: m, linearPlusOut: attnOut, gateLogit: gateLogit, gate: gate,
    terminalConvState: terminalConv, terminalSsmState: terminalSsm,
    linearPlusZState: lpZState, linearPlusScanState: lpScanState)

  finishMlp(mamba,result.y,result.cache,ctx)

proc backwardRBlockKernel*(mamba: Mamba3Real; x, normGamma, dFinal: Tensor;
                           cache: HybridBlockKernelCache; normEps: float32;
                           ctx: Backend; prefix: string;
                           grads: var Table[string, Tensor]): Tensor =
  let dOut=backwardMlp(mamba,cache,dFinal,ctx,prefix,grads)
  ## dOut fans to the mixer and the skip. Only the mixer derivative crosses RMSNorm.
  doAssert not cache.hasLinearPlus, "R block received a P-block cache"
  doAssert x.shape == dOut.shape and x.shape == cache.xNorm.shape,
    "R block backward: x, dOut, and cached normalized input must match"
  let dMixerIn = backwardMambaMixerKernel(mamba, cache.xNorm, cache.mamba, dOut,
                                          ctx, prefix & ".mambaReal", grads)
  let dNorm = blockRmsBackward(x, normGamma, dMixerIn, normEps, ctx)
  grads[prefix & ".rms1.gamma"] = dNorm.dGamma
  result = addT(ctx, dNorm.dX, dOut)

proc backwardPBlockKernel*(mamba: Mamba3Real; attn: LinearPlusAttention;
                           x, normGamma, dFinal: Tensor;
                           cache: HybridBlockKernelCache; normEps: float32;
                           ctx: Backend; prefix: string;
                           grads: var Table[string, Tensor];
                           fuseGate: ptr Linear = nil): Tensor =
  let dOut=backwardMlp(mamba,cache,dFinal,ctx,prefix,grads)
  ## The two branch input gradients must be summed before the shared RMSNorm backward.
  doAssert cache.hasLinearPlus, "P block requires a cache created by forwardPBlockKernel"
  doAssert x.shape == dOut.shape and x.shape == cache.xNorm.shape,
    "P block backward: x, dOut, and cached normalized input must match"
  doAssert cache.hasFuseGate == (fuseGate != nil),
    "P/G block backward: gate configuration differs from forward cache"
  var dMambaOut = dOut
  var dAttnOut = dOut
  if fuseGate != nil:
    dMambaOut = newTensor(dOut.shape)
    dAttnOut = newTensor(dOut.shape)
    var dGate = newTensor(dOut.shape)
    var dGateLogit = newTensor(cache.gateLogit.shape)
    var fuseIn = newTensor(@[x.shape[0], x.shape[1], x.shape[2] * 2])
    for row in 0 ..< x.shape[0] * x.shape[1]:
      for c in 0 ..< x.shape[2]:
        let i = row * x.shape[2] + c
        let gate = cache.gate[i]
        dMambaOut.data[i] = dOut.data[i] * gate
        dAttnOut.data[i] = dOut.data[i] * (1'f32 - gate)
        dGate.data[i] = dOut.data[i] * (cache.mambaY.data[i] - cache.linearPlusOut.data[i])
        dGateLogit.data[i] = dGate.data[i] * gate * (1'f32 - gate)
        fuseIn.data[row * x.shape[2] * 2 + c] = cache.mambaY.data[i]
        fuseIn.data[row * x.shape[2] * 2 + x.shape[2] + c] = cache.linearPlusOut.data[i]
    let dFuse = fuseGate[].backward(fuseIn, dGateLogit, ctx)
    grads[prefix & ".fuseGate.weight"] = dFuse.dW
    grads[prefix & ".fuseGate.bias"] = dFuse.dB
    for row in 0 ..< x.shape[0] * x.shape[1]:
      for c in 0 ..< x.shape[2]:
        let i = row * x.shape[2] + c
        dMambaOut.data[i] += dFuse.dX.data[row * x.shape[2] * 2 + c]
        dAttnOut.data[i] += dFuse.dX.data[row * x.shape[2] * 2 + x.shape[2] + c]
  let dMambaIn = backwardMambaMixerKernel(mamba, cache.xNorm, cache.mamba, dMambaOut,
                                          ctx, prefix & ".mambaReal", grads)
  var dAttnIn: Tensor
  if cache.linearPlusGpu:
    let B = x.shape[0]
    let T = x.shape[1]
    let C = x.shape[2]
    let (dLinearPlusY, dOutW, dOutB) =
      attn.outProj.backward(cache.linearPlusY, dAttnOut, ctx)
    grads[prefix & ".attn.outProj.weight"] = dOutW
    grads[prefix & ".attn.outProj.bias"] = dOutB
    let dQKV = linearPlusBackwardGpu(ctx, cache.attnProjection.data,
      dLinearPlusY.data, B, T, attn.nHeads, attn.headDim)
    var dProjection = newTensor(cache.attnProjection.shape)
    dProjection.data = dQKV
    let dQkvP = attn.qkvProj.backward(cache.xNorm, dProjection, ctx)
    grads[prefix & ".attn.qkvProj.weight"] = dQkvP.dW
    grads[prefix & ".attn.qkvProj.bias"] = dQkvP.dB
    dAttnIn = dQkvP.dX
  else:
    let dAttn = backwardLinearPlus(attn, cache.xNorm, dAttnOut, ctx)
    grads[prefix & ".attn.qkvProj.weight"] = dAttn.dQkvW
    grads[prefix & ".attn.qkvProj.bias"] = dAttn.dQkvB
    grads[prefix & ".attn.outProj.weight"] = dAttn.dOutW
    grads[prefix & ".attn.outProj.bias"] = dAttn.dOutB
    dAttnIn = dAttn.dX
  let dNormIn = addT(ctx, dMambaIn, dAttnIn)
  let dNorm = blockRmsBackward(x, normGamma, dNormIn, normEps, ctx)
  grads[prefix & ".rms1.gamma"] = dNorm.dGamma
  result = addT(ctx, dNorm.dX, dOut)
