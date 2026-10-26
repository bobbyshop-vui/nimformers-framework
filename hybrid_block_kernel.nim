## End-to-end logical GPU kernel entrypoints for the pretrained R/P blocks.
##
## R = RMSNorm -> Mamba2Real -> residual
## P = RMSNorm -> (Mamba2Real || Linear++) -> sum -> residual
##
## Each entrypoint owns the full block math and its forward/backward contract.
## The backend still uses multiple dependent GPU dispatches internally: Metal
## has no grid-wide barrier inside a dispatch, and projection GEMMs should use
## the tuned backend kernels rather than a scalar bytecode GEMM.

import std/[math, os, strutils, tables, strformat]
import backend
import nimformer
import mamba2_ssm_gpu
import mamba2_full_vm
import mamba2_real
import test_nimformer
import prefill_profile   # BYBY_PREFILL_PROFILE

type
  HybridBlockKernelCache* = object
    xNorm*: Tensor
    mamba*: Mamba2RealCache
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

proc offsetLpLayout(L: LpLay; base: int): LpLay =
  ## Place the complete Linear++ scratch layout after Mamba2's scratch in the
  ## same device buffer. Shape metadata stays unchanged; only data offsets move.
  result = L
  result.qkv += base; result.phi += base; result.y += base
  result.invDen += base; result.dY += base; result.dDen += base
  result.dQKV += base; result.zF += base; result.cF += base
  result.rQ += base; result.zQ += base; result.rK += base
  result.zK += base; result.cV += base; result.total += base

proc forwardMambaMixerKernel(mamba: Mamba2Real; x: Tensor; ctx: Backend):
    tuple[y: Tensor, cache: Mamba2RealCache] =
  ## The block entrypoint expands Mamba2Real into projection + its four VM kernels
  ## here, instead of calling the separate whole-mixer forward wrapper.
  doAssert x.shape.len == 3 and x.shape[2] == mamba.hiddenSize,
    "Mamba2Real block-kernel expects [B,T,hiddenSize]"
  ensureTgCap(ctx)
  let B = x.shape[0]
  let T = x.shape[1]
  let H = mamba.numHeads
  let D = mamba.headDim
  let N = mamba.dState
  let G = mamba.nGroups
  let C = mamba.intermediateSize
  let K = mamba.convKernel
  let cDim = mamba.convDim
  let L = m2Layout(B, T, H, D, N, G, K, C, mamba.normEps, G)
  checkVmScratchCapacity(ctx, L.total,
    &"Mamba2 block (B={B}, T={T}, H={H}, D={D}, N={N})")
  let projection = pfExpr("blk.inProj", mamba.inProj.forward(x, ctx))
  let pfVm0 = getMonoTimeIfPf()
  var buf = newSeq[float32](L.total)
  buf.blit(L.proj, projection.data, 0, L.BT * L.P)
  buf.blit(L.convW, mamba.convWeight.data, 0, cDim * K)
  buf.blit(L.convB, mamba.convBias.data, 0, cDim)
  buf.blit(L.aLog, mamba.aLog.data, 0, H)
  buf.blit(L.dSkip, mamba.dSkip.data, 0, H)
  buf.blit(L.dtBias, mamba.dtBias.data, 0, H)
  buf.blit(L.normW, mamba.normWeight.data, 0, C)
  let tgR = rmsTg(L)
  discard runMicroVmChain(ctx, [
    (buildM2ConvDtProgram(L), L.BT * M2Tg, 4, M2Tg, "@m2_conv_dt_" & m2Tag(L)),
    mamba2ScanStep(L),
    (buildM2RmsFwdProgram(L), L.BT * L.normGroups * tgR, tgR * 4, tgR,
      "@m2_rms_" & m2Tag(L))], buf)
  pfAddIf("blk.mamba2Vm", pfVm0)
  let normed = sliceTensor(buf, L.normedOut, L.BT * C, @[B, T, C])
  result.y = pfExpr("blk.outProj", mamba.outProj.forward(normed, ctx))
  result.cache = Mamba2RealCache(B: B, T: T, buf: buf, lay: L, proj: projection)

proc terminalConvHistory(L: M2Lay; T: int; proj: Tensor): seq[float32] =
  ## Conv history is the last K-1 *raw projected* xBC rows, oldest first;
  ## L.pre contains convolution pre-activations, not this history.
  let history = max(L.K - 1, 0)
  result = newSeq[float32](L.cDim * history)
  for c in 0 ..< L.cDim:
    for k in 0 ..< history:
      let t = T - history + k
      if t >= 0:
        result[c * history + k] = proj.data[t * L.P + L.C + c]

proc terminalMambaStates(cache: Mamba2RealCache):
    tuple[conv, ssm: seq[float32]] =
  ## Convert the full-sequence cache to the two states consumed by recurrent
  ## decode.
  let L = cache.lay
  result.conv = terminalConvHistory(L, cache.T, cache.proj)
  result.ssm = newSeq[float32](L.HD * L.N)
  let finalState = L.states + cache.T * L.HD * L.N
  result.ssm.blit(0, cache.buf, finalState, result.ssm.len)

var gSparseCheckDone = false

proc forwardMambaMixerInfer(mamba: Mamba2Real; x: Tensor; ctx: Backend):
    tuple[y: Tensor, conv, ssm: seq[float32]] =
  ## Inference-only mixer (prefill): same kernels as forwardMambaMixerKernel, but
  ## the ~57MB scratch never round-trips through the host. Only [0, states+HD*N)
  ## (projection, weights, zero initial state) is uploaded and only normedOut +
  ## the terminal SSM state are read back. No backward cache.
  ## BYBY_PREFILL_SPARSE=0 -> full-buffer path; BYBY_PREFILL_SPARSE_CHECK=1 ->
  ## compare against the full path once (first call) and print the max diff.
  doAssert x.shape.len == 3 and x.shape[2] == mamba.hiddenSize,
    "Mamba2Real block-kernel expects [B,T,hiddenSize]"
  ensureTgCap(ctx)
  let B = x.shape[0]
  let T = x.shape[1]
  let C = mamba.intermediateSize
  let K = mamba.convKernel
  let cDim = mamba.convDim
  let L = m2Layout(B, T, mamba.numHeads, mamba.headDim, mamba.dState,
                   mamba.nGroups, K, C, mamba.normEps, mamba.nGroups)
  let useSparse = B == 1 and isGpuCtx(ctx) and getEnv("BYBY_PREFILL_SPARSE", "1") != "0"
  if useSparse:
    let projection = pfExpr("blk.inProj", mamba.inProj.forward(x, ctx))
    let pfVm0 = getMonoTimeIfPf()
    var head = newSeq[float32](L.states + L.HD * L.N)   # duoi = state[0] = 0
    head.blit(L.proj, projection.data, 0, L.BT * L.P)
    head.blit(L.convW, mamba.convWeight.data, 0, cDim * K)
    head.blit(L.convB, mamba.convBias.data, 0, cDim)
    head.blit(L.aLog, mamba.aLog.data, 0, mamba.numHeads)
    head.blit(L.dSkip, mamba.dSkip.data, 0, mamba.numHeads)
    head.blit(L.dtBias, mamba.dtBias.data, 0, mamba.numHeads)
    head.blit(L.normW, mamba.normWeight.data, 0, C)
    let tgR = rmsTg(L)
    var outs: seq[seq[float32]]
    var ok = true
    try:
      outs = runMicroVmChainSparse(ctx, [
        (buildM2ConvDtProgram(L), L.BT * M2Tg, 4, M2Tg, "@m2_conv_dt_" & m2Tag(L)),
        mamba2ScanStep(L),
        (buildM2RmsFwdProgram(L), L.BT * L.normGroups * tgR, tgR * 4, tgR,
          "@m2_rms_" & m2Tag(L))], L.fwdEnd, head,
        [(off: L.normedOut, len: L.BT * C), (off: L.states + T * L.HD * L.N, len: L.HD * L.N)])
    except CatchableError as e:
      ok = false
      stderr.writeLine "[prefill] sparse Mamba2 chain loi -> dung full buffer: " & e.msg
    if ok:
      pfAddIf("blk.mamba2Vm", pfVm0)
      let normed = Tensor(shape: @[B, T, C], data: outs[0])
      result.y = pfExpr("blk.outProj", mamba.outProj.forward(normed, ctx))
      result.conv = terminalConvHistory(L, T, projection)
      result.ssm = outs[1]
      if getEnv("BYBY_PREFILL_SPARSE_CHECK", "0") != "0" and not gSparseCheckDone:
        gSparseCheckDone = true
        let full = forwardMambaMixerKernel(mamba, x, ctx)
        let fullSsm = terminalMambaStates(full.cache).ssm
        var dy, ds = 0'f32
        for i in 0 ..< result.y.data.len: dy = max(dy, abs(result.y.data[i] - full.y.data[i]))
        for i in 0 ..< result.ssm.len: ds = max(ds, abs(result.ssm[i] - fullSsm[i]))
        stderr.writeLine &"[prefill] sparse-vs-full check (block dau): max|dy|={dy:.3e} max|dssm|={ds:.3e}"
      return
  let mixer = forwardMambaMixerKernel(mamba, x, ctx)
  result.y = mixer.y
  let terminal = pfExpr("blk.terminalState", terminalMambaStates(mixer.cache))
  result.conv = terminal.conv
  result.ssm = terminal.ssm

type
  LoraGradSpec* = object
    ## Adapter LoRA cua 1 Linear: A [r, inF], B [outF, r] (row-major, nhu LoraAdapter).
    A*, B*: seq[float32]
    r*: int
    scale*: float32

## Backward LoRA-aware (train LoRA): trainStep dang ky adapter theo TEN tham so
## (vd "block7.mambaReal.inProj.weight") truoc backwardBlock va xoa sau do. Voi
## Linear co dang ky, backward KHONG tinh dW day du [outF, inF] (19.5 GMAC + doc/ghi
## ~300MB cho inProj) roi rut gon qua gradToAB, ma tinh thang:
##   dB = s * dY^T (X A^T)      [outF, r]
##   dA = s * (dY B)^T X        [r, inF]
## - dung bang toan hoc voi gradToAB(dW = dY^T X). Grad ghi vao grads[key & "::loraA"]
## va grads[key & "::loraB"]; dX van = dY @ W_eff (weight da merge LoRA) nhu cu.
## Bang rong (mac dinh, moi duong infer/khac) -> giu nguyen duong backward cu.
var gLoraGradRequest* = initTable[string, LoraGradSpec]()

proc linearBackwardLoraAware*(l: Linear; x, dOut: Tensor; spec: LoraGradSpec;
                              ctx: Backend): tuple[dX, dA, dB, dBias: Tensor] =
  let (rows, cols) = flatten2D(x.shape)
  let inF = l.inF
  let outF = l.outF
  let r = spec.r
  doAssert cols == inF and dOut.shape[^1] == outF and dOut.data.len == rows * outF,
    &"LoRA-aware backward: x.shape={x.shape} dOut.shape={dOut.shape} khong khop Linear {inF}->{outF}"
  doAssert l.weight.data.len == outF * inF,
    "LoRA-aware backward can weight fp32 da merge LoRA trong l.weight.data"
  doAssert r > 0 and spec.A.len == r * inF and spec.B.len == outF * r,
    "LoRA-aware backward: A/B sai kich thuoc"
  # dX = dOut @ W_eff  ([rows,outF] @ [outF,inF]) - giong nhanh dX cua Linear.backward.
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
  ## Linear co adapter dang ky -> LoRA-aware (grads[key::loraA/::loraB]); khong thi
  ## Linear.backward day du nhu cu (grads[key] = dW).
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

proc backwardMambaMixerKernel(mamba: Mamba2Real; x: Tensor;
    cache: Mamba2RealCache; dOut: Tensor; ctx: Backend; prefix: string;
    grads: var Table[string, Tensor]): Tensor =
  ## Explicit transpose of forwardMambaMixerKernel; every trainable Mamba
  ## parameter receives a gradient, including output projection and gated norm.
  ensureTgCap(ctx)
  let L = cache.lay
  let B = cache.B
  let T = cache.T
  let H = mamba.numHeads
  let C = mamba.intermediateSize
  let cDim = mamba.convDim
  let K = mamba.convKernel
  doAssert x.shape == @[B, T, mamba.hiddenSize] and dOut.shape == x.shape,
    "Mamba2Real backward: cached input / dOut shape mismatch"

  let normed = sliceTensor(cache.buf, L.normedOut, L.BT * C, @[B, T, C])
  let dNormed = linearBackwardMaybeLora(mamba.outProj, prefix & ".outProj.weight", normed, dOut,
                                        ctx, grads, prefix & ".outProj.bias")

  var buf = cache.buf
  buf.blit(L.dNormedOut, dNormed.data, 0, L.BT * C)
  let tgR = rmsTg(L)
  # Moi nhom = 1 command buffer nhu truoc, nhung buf chi upload/doc lai 1 lan cho ca
  # rms_bwd + 16 chunk scan_bwd + dt/conv (truoc: 18 lan upload + 18 lan doc ca scratch).
  var groups: seq[seq[MvmStep]]
  groups.add @[(buildM2RmsBwdProgram(L), L.BT * L.normGroups * tgR, tgR * 4, tgR,
      "@m2_rms_bwd_" & m2Tag(L))]

  var nChunk = 16
  let envChunk = getEnv("BYBY_SCAN_NCHUNK")
  if envChunk.len > 0:
    try: nChunk = max(1, parseInt(envChunk))
    except ValueError: discard
  nChunk = min(nChunk, mamba.dState)
  let perChunk = (mamba.dState + nChunk - 1) div nChunk
  var nBeg = 0
  while nBeg < mamba.dState:
    let nCount = min(perChunk, mamba.dState - nBeg)
    groups.add @[(buildM2ScanBwdProgram(L, nBeg, nCount), B * L.HD, 4, 0,
        "@m2_scan_bwd_" & m2Tag(L) & "_n" & $nBeg & "_" & $nCount)]
    nBeg += nCount
  groups.add @[
    (buildM2DtBwdProgram(L), L.BT * M2Tg, 4, M2Tg, "@m2_dt_bwd_" & m2Tag(L)),
    (buildM2ConvBwdProgram(L), L.BT * M2Tg, 4, M2Tg, "@m2_conv_bwd_" & m2Tag(L))]
  discard runMicroVmChainGroups(ctx, groups, buf)

  grads[prefix & ".normWeight"] = sliceTensor(buf, L.dNormW, C, @[C])
  grads[prefix & ".dSkip"] = sliceTensor(buf, L.dDSkip, H, @[H])
  grads[prefix & ".dtBias"] = sliceTensor(buf, L.dDtBias, H, @[H])
  grads[prefix & ".aLog"] = sliceTensor(buf, L.dALog, H, @[H])
  grads[prefix & ".convWeight"] = sliceTensor(buf, L.dConvW, cDim * K, @[cDim, K])
  grads[prefix & ".convBias"] = sliceTensor(buf, L.dConvB, cDim, @[cDim])
  let dProj = sliceTensor(buf, L.dProj, L.BT * L.P, cache.proj.shape)
  result = linearBackwardMaybeLora(mamba.inProj, prefix & ".inProj.weight", x, dProj,
                                   ctx, grads, prefix & ".inProj.bias")

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

proc forwardRBlockKernel*(mamba: Mamba2Real; x, normGamma: Tensor;
                          normEps: float32; ctx: Backend;
                          saveBackwardCache = true;
                          exportTerminalState = false):
    tuple[y: Tensor, cache: HybridBlockKernelCache] =
  ## Complete R block: norm, Mamba2Real mixer, and residual.
  ## cache.mambaY = mixer output TRƯỚC residual (để đo rms(mambaY)/rms(x)).
  let xNorm = pfExpr("blk.rms", blockRmsForward(x, normGamma, normEps, ctx))
  if exportTerminalState and not saveBackwardCache:
    # Suy luan (prefill): khong can backward cache -> mixer khong dua scratch ve host.
    let mi = forwardMambaMixerInfer(mamba, xNorm, ctx)
    result.y = pfExpr("blk.residual", addT(ctx, x, mi.y))
    result.cache = HybridBlockKernelCache(xNorm: xNorm, hasLinearPlus: false,
      mambaY: mi.y, terminalConvState: mi.conv, terminalSsmState: mi.ssm)
    return
  let mixer = forwardMambaMixerKernel(mamba, xNorm, ctx)
  result.y = pfExpr("blk.residual", addT(ctx, x, mixer.y))
  result.cache = HybridBlockKernelCache(xNorm: xNorm,
    mamba: (if saveBackwardCache: mixer.cache else: default(Mamba2RealCache)),
    hasLinearPlus: false,
    mambaY: mixer.y)
  if exportTerminalState:
    let terminal = pfExpr("blk.terminalState", terminalMambaStates(mixer.cache))
    result.cache.terminalConvState = terminal.conv
    result.cache.terminalSsmState = terminal.ssm

proc forwardPBlockKernel*(mamba: Mamba2Real; attn: LinearPlusAttention;
                          x, normGamma: Tensor; normEps: float32; ctx: Backend;
                          saveBackwardCache = true;
                          exportTerminalState = false;
                          fuseGate: ptr Linear = nil):
    tuple[y: Tensor, cache: HybridBlockKernelCache] =
  ## Complete P/G block: shared norm, parallel branches, fixed sum or gated mix.
  let xNorm = pfExpr("blk.rms", blockRmsForward(x, normGamma, normEps, ctx))
  var m, attnProjection, lpY, attnOut: Tensor
  var mambaCache: Mamba2RealCache
  var terminalConv, terminalSsm, lpZState, lpScanState: seq[float32]
  var lpGpu = false
  if isGpuCtx(ctx):
    ## Both branches share the same normalized input and are combined as a
    ## parallel-branch block mathematically. Their dispatches are currently
    ## ordered in one upload/session (not concurrent GPU streams); projections
    ## remain tuned GEMMs.
    ensureTgCap(ctx)
    let B = x.shape[0]
    let T = x.shape[1]
    let C = x.shape[2]
    let Hm = mamba.numHeads
    let Dm = mamba.headDim
    let Nm = mamba.dState
    let Gm = mamba.nGroups
    let Cm = mamba.intermediateSize
    let Km = mamba.convKernel
    let cDim = mamba.convDim
    let mLay = m2Layout(B, T, Hm, Dm, Nm, Gm, Km, Cm, mamba.normEps, Gm)
    # Keep independent scratch arenas aligned and surround each with a canary
    # red-zone so an offset overrun fails loudly instead of corrupting the other branch.
    const guardLen = 256
    const guardValue = 123456.75'f32
    let lpBase = ((mLay.total + guardLen + 255) div 256) * 256
    let lpLay = offsetLpLayout(lpLayout(B, T, attn.nHeads, attn.headDim), lpBase)
    checkVmScratchCapacity(ctx, lpLay.total + guardLen,
      &"P hybrid block (B={B}, T={T}, Mamba2+Linear++)")
    let mProjection = pfExpr("blk.inProj", mamba.inProj.forward(xNorm, ctx))
    attnProjection = pfExpr("P.qkvProj", attn.qkvProj.forward(xNorm, ctx))
    let pfVm0 = getMonoTimeIfPf()
    var buf = newSeq[float32](lpLay.total + guardLen)
    for i in mLay.total ..< lpBase: buf[i] = guardValue
    for i in lpLay.total ..< buf.len: buf[i] = guardValue
    buf.blit(mLay.proj, mProjection.data, 0, mLay.BT * mLay.P)
    buf.blit(mLay.convW, mamba.convWeight.data, 0, cDim * Km)
    buf.blit(mLay.convB, mamba.convBias.data, 0, cDim)
    buf.blit(mLay.aLog, mamba.aLog.data, 0, Hm)
    buf.blit(mLay.dSkip, mamba.dSkip.data, 0, Hm)
    buf.blit(mLay.dtBias, mamba.dtBias.data, 0, Hm)
    buf.blit(mLay.normW, mamba.normWeight.data, 0, Cm)
    buf.blit(lpLay.qkv, attnProjection.data, 0, B * T * 3 * C)
    let tgR = rmsTg(mLay)
    var blockSteps: seq[MvmStep] = @[
      (buildM2ConvDtProgram(mLay), mLay.BT * M2Tg, 4, M2Tg,
        "@m2_conv_dt_" & m2Tag(mLay)),
      mamba2ScanStep(mLay),
      (buildM2RmsFwdProgram(mLay), mLay.BT * mLay.normGroups * tgR,
        tgR * 4, tgR, "@m2_rms_" & m2Tag(mLay))]
    for lpStep in linearPlusForwardSteps(lpLay):
      blockSteps.add(lpStep)
    discard runMicroVmChain(ctx, blockSteps, buf)
    pfAddIf("P.mamba2+lpVm", pfVm0)
    for i in mLay.total ..< lpBase:
      if buf[i] != guardValue:
        raise newException(CatchableError,
          "P block micro-kernel: Mamba stage overwrote shared-buffer red-zone at " & $i)
    for i in lpLay.total ..< buf.len:
      if buf[i] != guardValue:
        raise newException(CatchableError,
          "P block micro-kernel: Linear++ stage overwrote shared-buffer red-zone at " & $i)
    let mNormed = sliceTensor(buf, mLay.normedOut, mLay.BT * Cm, @[B, T, Cm])
    m = pfExpr("blk.outProj", mamba.outProj.forward(mNormed, ctx))
    lpY = sliceTensor(buf, lpLay.y, B * T * lpLay.C, @[B, T, lpLay.C])
    attnOut = pfExpr("P.attnOutProj", attn.outProj.forward(lpY, ctx))
    if saveBackwardCache:
      mambaCache = Mamba2RealCache(B: B, T: T,
        buf: buf[0 ..< mLay.total], lay: mLay, proj: mProjection)
    if exportTerminalState:
      let terminal = terminalMambaStates(Mamba2RealCache(B: B, T: T,
        buf: buf, lay: mLay, proj: mProjection))
      terminalConv = terminal.conv
      terminalSsm = terminal.ssm
      lpZState = buf[lpLay.zF ..< lpLay.zF + B * lpLay.C]
      lpScanState = buf[lpLay.cF ..< lpLay.cF + B * lpLay.C * lpLay.D]
    lpGpu = true
  else:
    let mResult = forwardMambaMixerKernel(mamba, xNorm, ctx)
    m = mResult.y
    mambaCache = mResult.cache
    attnOut = forwardLinearPlus(attn, xNorm, ctx)
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

proc backwardRBlockKernel*(mamba: Mamba2Real; x, normGamma, dOut: Tensor;
                           cache: HybridBlockKernelCache; normEps: float32;
                           ctx: Backend; prefix: string;
                           grads: var Table[string, Tensor]): Tensor =
  ## dOut fans to the mixer and the skip. Only the mixer derivative crosses RMSNorm.
  doAssert not cache.hasLinearPlus, "R block received a P-block cache"
  doAssert x.shape == dOut.shape and x.shape == cache.xNorm.shape,
    "R block backward: x, dOut, and cached normalized input must match"
  let dMixerIn = backwardMambaMixerKernel(mamba, cache.xNorm, cache.mamba, dOut,
                                          ctx, prefix & ".mambaReal", grads)
  let dNorm = blockRmsBackward(x, normGamma, dMixerIn, normEps, ctx)
  grads[prefix & ".rms1.gamma"] = dNorm.dGamma
  result = addT(ctx, dNorm.dX, dOut)

proc backwardPBlockKernel*(mamba: Mamba2Real; attn: LinearPlusAttention;
                           x, normGamma, dOut: Tensor;
                           cache: HybridBlockKernelCache; normEps: float32;
                           ctx: Backend; prefix: string;
                           grads: var Table[string, Tensor];
                           fuseGate: ptr Linear = nil): Tensor =
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
