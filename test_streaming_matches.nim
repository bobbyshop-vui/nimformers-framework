import std/[tables, strformat, math, strutils, os, random]
import backend
import nimformer
import jamba_hybrid
import mamba2_real
import test_nimformer
import hybrid_block_kernel
import mamba2_full_vm
import mamba2_ssm_gpu
import fp4_gemv_vm
import lora
import quant
import finetune_lora   # trainStepGraftStreaming (import an toan - when isMainModule cua no khong chay khi import)

proc loraObjective(a, b, g: seq[float32], scale: float32): float64 =
  ## CPU float64 oracle: <G, scale * B @ A>. Deliberately independent of
  ## lora.mergedWeight/gradToAB so the test can catch transpose/scale mistakes.
  const outF = 2; const rank = 2; const inF = 3
  for o in 0 ..< outF:
    for i in 0 ..< inF:
      var delta = 0.0'f64
      for k in 0 ..< rank:
        delta += float64(b[o * rank + k]) * float64(a[k * inF + i])
      result += float64(g[o * inF + i]) * float64(scale) * delta

proc cloneData(x: seq[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  for i, v in x: result[i] = v

proc maxAbsDiff(a, b: Tensor): float32 =
  doAssert a.shape == b.shape and a.data.len == b.data.len,
    "tensor comparison shape mismatch"
  for i in 0 ..< a.data.len:
    result = max(result, abs(a.data[i] - b.data[i]))

proc checkMambaParallelScan(ctx: Backend; mamba: Mamba2Real; x: Tensor) =
  ## Compare the timestep-parallel affine scan against both the serial
  ## sequence kernel and the actual one-token decode recurrence. The latter is
  ## the critical prefill->decode boundary: matching sequence scan kernels
  ## alone would not catch a state-layout/import mismatch.
  ensureTgCap(ctx)
  let B = x.shape[0]; let T = x.shape[1]
  doAssert B == 1, "decode-recurrence parity test currently expects batch size 1"
  let L = m2Layout(B, T, mamba.numHeads, mamba.headDim, mamba.dState,
    mamba.nGroups, mamba.convKernel, mamba.intermediateSize,
    mamba.normEps, mamba.nGroups)
  let proj = mamba.inProj.forward(x, ctx)
  var initial = newSeq[float32](L.total)
  initial.blit(L.proj, proj.data, 0, L.BT * L.P)
  initial.blit(L.convW, mamba.convWeight.data, 0, mamba.convDim * mamba.convKernel)
  initial.blit(L.convB, mamba.convBias.data, 0, mamba.convDim)
  initial.blit(L.aLog, mamba.aLog.data, 0, mamba.numHeads)
  initial.blit(L.dSkip, mamba.dSkip.data, 0, mamba.numHeads)
  initial.blit(L.dtBias, mamba.dtBias.data, 0, mamba.numHeads)
  initial.blit(L.normW, mamba.normWeight.data, 0, mamba.intermediateSize)
  var sequential = cloneData(initial)
  var parallel = cloneData(initial)
  let tgR = rmsTg(L)
  discard runMicroVmChain(ctx, [
    (buildM2ConvSiluProgram(L), L.BT * M2Tg, 4, M2Tg, "scan_ref_conv"),
    (buildM2DtProgram(L), L.BT * M2Tg, 4, M2Tg, "scan_ref_dt"),
    (buildM2ScanProgram(L), B * L.HD, 4, 0, "scan_ref_serial")], sequential)
  discard runMicroVmChain(ctx, [
    (buildM2ConvSiluProgram(L), L.BT * M2Tg, 4, M2Tg, "scan_new_conv"),
    (buildM2DtProgram(L), L.BT * M2Tg, 4, M2Tg, "scan_new_dt"),
    mamba2ScanStep(L, ctx, forceParallel = true)], parallel)
  var yErr = 0'f32
  for i in 0 ..< L.BT * L.C:
    yErr = max(yErr, abs(sequential[L.yPre + i] - parallel[L.yPre + i]))
  var stateErr = 0'f32
  for i in 0 ..< B * (T + 1) * L.HD * L.N:
    stateErr = max(stateErr, abs(sequential[L.states + i] - parallel[L.states + i]))
  # Replay each projected timestep through the same causal conv + single-step
  # SSM primitives used by decode. This independently verifies the terminal
  # states and every timestep output, including conv history ordering.
  var recurrentConvState = newSeq[float32](L.cDim * (L.K - 1))
  var recurrentSsmState = newSeq[float32](L.HD * L.N)
  var recurrentProjectionErr = 0'f32
  var recurrentConvErr = 0'f32
  var recurrentDtErr = 0'f32
  var recurrentYErr = 0'f32
  var recurrentStateErr = 0'f32
  for t in 0 ..< T:
    var xToken = newTensor(@[1, 1, x.shape[2]])
    xToken.data.blit(0, x.data, t * x.shape[2], x.shape[2])
    let projStep = mamba.inProj.forward(xToken, ctx)
    for j in 0 ..< L.P:
      recurrentProjectionErr = max(recurrentProjectionErr,
        abs(projStep.data[j] - parallel[L.proj + t * L.P + j]))
    var xbcPre = newSeq[float32](L.cDim)
    xbcPre.blit(0, projStep.data, L.C, L.cDim)
    let convStep = conv1dIncGpu(ctx, L.cDim, L.K, xbcPre, recurrentConvState,
      mamba.convWeight.data, mamba.convBias.data)
    for c in 0 ..< L.cDim:
      recurrentConvErr = max(recurrentConvErr,
        abs(convStep[c] - parallel[L.conv + t * L.cDim + c]))
    var dtPre = newSeq[float32](L.H)
    dtPre.blit(0, projStep.data, L.C + L.cDim, L.H)
    let dtStep = softplusDtGpu(ctx, L.H, dtPre, mamba.dtBias.data)
    for h in 0 ..< L.H:
      recurrentDtErr = max(recurrentDtErr,
        abs(dtStep[h] - parallel[L.dt + t * L.H + h]))
    let yStep = mamba2StepGpuFromXbc(ctx, L.H, L.D, L.N, L.G, L.C,
      convStep, dtStep, mamba.aLog.data, mamba.dSkip.data, recurrentSsmState)
    for c in 0 ..< L.C:
      recurrentYErr = max(recurrentYErr,
        abs(yStep[c] - parallel[L.yPre + t * L.C + c]))
    for i in 0 ..< L.HD * L.N:
      recurrentStateErr = max(recurrentStateErr,
        abs(recurrentSsmState[i] - parallel[L.states + (t + 1) * L.HD * L.N + i]))
  var terminalConvErr = 0'f32
  for c in 0 ..< L.cDim:
    for k in 0 ..< (L.K - 1):
      let sourceT = T - (L.K - 1) + k
      let want = if sourceT < 0: 0'f32
                 else: parallel[L.proj + sourceT * L.P + L.C + c]
      terminalConvErr = max(terminalConvErr,
        abs(recurrentConvState[c * (L.K - 1) + k] - want))
  if yErr > 3e-5'f32 or stateErr > 3e-5'f32 or
     recurrentProjectionErr > 3e-5'f32 or recurrentDtErr > 3e-5'f32 or
     recurrentConvErr > 3e-5'f32 or recurrentYErr > 3e-5'f32 or
     recurrentStateErr > 3e-5'f32 or terminalConvErr > 3e-6'f32:
    raise newException(AssertionDefect,
      &"Mamba2 parallel scan/decode recurrence mismatch: serialY={yErr:.7g}, " &
       &"serialState={stateErr:.7g}, projection={recurrentProjectionErr:.7g}, " &
       &"dt={recurrentDtErr:.7g}, conv={recurrentConvErr:.7g}, " &
       &"decodeY={recurrentYErr:.7g}, decodeState={recurrentStateErr:.7g}, " &
       &"terminalConv={terminalConvErr:.7g}, T={T}")
  echo &"PASS - Mamba2 parallel scan vs serial + single-step decode recurrence " &
       &"(T={T}, paddedTG={m2ScanTgSize(T)}, scanY={yErr:.2g}, " &
       &"projection={recurrentProjectionErr:.2g}, dt={recurrentDtErr:.2g}, " &
       &"decodeY={recurrentYErr:.2g}, terminalState={recurrentStateErr:.2g}, " &
       &"convHistory={terminalConvErr:.2g})"

proc shiftedLpLayout(L: LpLay; base: int): LpLay =
  result = L
  result.qkv += base; result.phi += base; result.y += base
  result.invDen += base; result.dY += base; result.dDen += base
  result.dQKV += base; result.zF += base; result.cF += base
  result.rQ += base; result.zQ += base; result.rK += base
  result.zK += base; result.cV += base; result.total += base

proc checkLoraGradientFiniteDifference(ctx: Backend) =
  ## Exercise the actual GPU gradToAB implementation against central
  ## finite-differences, including nonzero A and B and scale != 1.
  const outF = 2; const rank = 2; const inF = 3
  var ad: LoraAdapter
  ad.r = rank; ad.inF = inF; ad.outF = outF
  ad.alpha = 4'f32; ad.scale = ad.alpha / float32(rank) # intentionally 2
  ad.A = Tensor(shape: @[rank, inF], data: @[0.2'f32, -0.4, 0.7, 0.5, 0.3, -0.6])
  ad.B = Tensor(shape: @[outF, rank], data: @[0.8'f32, -0.2, -0.5, 0.9])
  let g = Tensor(shape: @[outF, inF], data: @[0.3'f32, -0.7, 0.2, -0.4, 0.6, 0.9])
  let base = Tensor(shape: @[outF, inF], data: @[0.1'f32, 0.2, -0.3, 0.4, -0.5, 0.6])
  let effective = mergedWeight(base, ad, ctx)
  var maxMergeErr = 0.0'f64
  for o in 0 ..< outF:
    for i in 0 ..< inF:
      var want = float64(base.data[o * inF + i])
      for k in 0 ..< rank:
        want += float64(ad.scale) * float64(ad.B.data[o * rank + k]) *
                float64(ad.A.data[k * inF + i])
      maxMergeErr = max(maxMergeErr, abs(want - float64(effective.data[o * inF + i])))
  if maxMergeErr > 2e-6'f64:
    raise newException(AssertionDefect,
      &"LoRA mergedWeight vs independent CPU oracle mismatch: {maxMergeErr:.8g}")
  let (dA, dB) = gradToAB(g, ad, ctx)
  const h = 1e-3'f32
  var maxErr = 0.0'f64
  for i in 0 ..< ad.A.data.len:
    var ap = cloneData(ad.A.data); var am = cloneData(ad.A.data)
    ap[i] += h; am[i] -= h
    let fd = (loraObjective(ap, ad.B.data, g.data, ad.scale) -
              loraObjective(am, ad.B.data, g.data, ad.scale)) / float64(2*h)
    maxErr = max(maxErr, abs(fd - float64(dA.data[i])))
  for i in 0 ..< ad.B.data.len:
    var bp = cloneData(ad.B.data); var bm = cloneData(ad.B.data)
    bp[i] += h; bm[i] -= h
    let fd = (loraObjective(ad.A.data, bp, g.data, ad.scale) -
              loraObjective(ad.A.data, bm, g.data, ad.scale)) / float64(2*h)
    maxErr = max(maxErr, abs(fd - float64(dB.data[i])))
  if maxErr > 2e-4'f64:
    raise newException(AssertionDefect,
      &"LoRA gradToAB finite-difference mismatch: max abs error={maxErr:.8g}; scale={ad.scale}")
  echo &"PASS - mergedWeight oracle; actual GPU gradToAB vs CPU float64 FD " &
       &"(scale={ad.scale}, merge err={maxMergeErr:.3g}, grad err={maxErr:.3g})"

proc pBlockObjective(m: NimformerModelJamba, x, dOut: Tensor,
                     ctx: Backend): float64 =
  let y = forwardBlock(m.blocks[0], x, ctx)
  for i in 0 ..< y.data.len:
    result += float64(y.data[i]) * float64(dOut.data[i])

proc checkRBlock(ctx: Backend) =
  ## R entrypoint finite-difference probes cover shared norm, recurrent branch,
  ## output projection, and the residual contribution to dX.
  const C = 16
  randomize(55129)
  var m = newNimformerModelJamba(24, C, 2, 2, "R", useSession = false,
    mambaRealDState = 4, mambaRealNGroups = 1, mambaRealConvKernel = 3,
    mambaRealExpand = 2)
  var x = newTensor(@[1, 2, C])
  var dOut = newTensor(@[1, 2, C])
  for i in 0 ..< x.data.len:
    x.data[i] = float32(sin(float64(i + 3) * 0.27))
    dOut.data[i] = float32(cos(float64(i + 5) * 0.19))
  checkMambaParallelScan(ctx, m.blocks[0].mambaReal, x)
  var grads = initTable[string, Tensor]()
  let dX = backwardBlock(m.blocks[0], x, dOut, ctx, "block0", grads)
  proc verify(label: string; analytic: float32; fd: float64) =
    let err = abs(float64(analytic) - fd)
    let tol = 5e-3'f64 + 2e-2'f64 * abs(fd)
    if err > tol:
      raise newException(AssertionDefect,
        &"R-block finite difference failed for {label}: analytic={analytic:.7g}, " &
         &"fd={fd:.7g}, abs err={err:.4g}, tol={tol:.4g}")
  const h = 2e-3'f32
  let xi = 9
  var xp = x; xp.data = cloneData(x.data); xp.data[xi] += h
  var xm = x; xm.data = cloneData(x.data); xm.data[xi] -= h
  verify("RMSNorm + residual input gradient", dX.data[xi],
    (pBlockObjective(m, xp, dOut, ctx) - pBlockObjective(m, xm, dOut, ctx)) / float64(2*h))

  let gName = "block0.rms1.gamma"
  var w = m.blocks[0].rms1.gamma.data[3]
  m.blocks[0].rms1.gamma.data[3] = w + h
  let gp = pBlockObjective(m, x, dOut, ctx)
  m.blocks[0].rms1.gamma.data[3] = w - h
  let gm = pBlockObjective(m, x, dOut, ctx)
  m.blocks[0].rms1.gamma.data[3] = w
  verify("RMSNorm gamma", grads[gName].data[3], (gp - gm) / float64(2*h))

  let inName = "block0.mambaReal.inProj.weight"
  w = m.blocks[0].mambaReal.inProj.weight.data[7]
  m.blocks[0].mambaReal.inProj.weight.data[7] = w + h
  let ip = pBlockObjective(m, x, dOut, ctx)
  m.blocks[0].mambaReal.inProj.weight.data[7] = w - h
  let im = pBlockObjective(m, x, dOut, ctx)
  m.blocks[0].mambaReal.inProj.weight.data[7] = w
  verify("Mamba2Real inProj weight", grads[inName].data[7], (ip - im) / float64(2*h))

  let convName = "block0.mambaReal.convWeight"
  w = m.blocks[0].mambaReal.convWeight.data[5]
  m.blocks[0].mambaReal.convWeight.data[5] = w + h
  let cp = pBlockObjective(m, x, dOut, ctx)
  m.blocks[0].mambaReal.convWeight.data[5] = w - h
  let cm = pBlockObjective(m, x, dOut, ctx)
  m.blocks[0].mambaReal.convWeight.data[5] = w
  verify("Mamba2Real causal-conv weight", grads[convName].data[5], (cp - cm) / float64(2*h))

  let dtName = "block0.mambaReal.dtBias"
  w = m.blocks[0].mambaReal.dtBias.data[0]
  m.blocks[0].mambaReal.dtBias.data[0] = w + h
  let dtp = pBlockObjective(m, x, dOut, ctx)
  m.blocks[0].mambaReal.dtBias.data[0] = w - h
  let dtm = pBlockObjective(m, x, dOut, ctx)
  m.blocks[0].mambaReal.dtBias.data[0] = w
  verify("Mamba2Real dtBias", grads[dtName].data[0], (dtp - dtm) / float64(2*h))

  let outName = "block0.mambaReal.outProj.weight"
  w = m.blocks[0].mambaReal.outProj.weight.data[17]
  m.blocks[0].mambaReal.outProj.weight.data[17] = w + h
  let op = pBlockObjective(m, x, dOut, ctx)
  m.blocks[0].mambaReal.outProj.weight.data[17] = w - h
  let om = pBlockObjective(m, x, dOut, ctx)
  m.blocks[0].mambaReal.outProj.weight.data[17] = w
  verify("Mamba2Real outProj weight", grads[outName].data[17], (op - om) / float64(2*h))
  let trainBatch = forwardBatch(m, @[@[1, 2]], ctx)
  let inferBatch = forwardBatchSession(m, @[@[1, 2]], ctx)
  let inferBatchErr = maxAbsDiff(trainBatch, inferBatch)
  if inferBatchErr > 2e-5'f32:
    raise newException(AssertionDefect,
      &"R block inference batch path mismatch: {inferBatchErr:.7g}")
  echo "PASS - R block RMSNorm, residual, inProj, causal-conv, dtBias, outProj finite-difference probes"

proc checkParallelPBlock(ctx: Backend) =
  ## P = pretrained Mamba2Real + newly grafted Linear++ in parallel.
  ## First assert exact identity at zero sidecar out-projection; then finite-
  ## difference input and both branches' parameters to catch a broken shared
  ## RMSNorm Jacobian or accidental cross-branch weight/state corruption.
  const C = 16
  randomize(77123)
  var mp = newNimformerModelJamba(24, C, 2, 2, "P", useSession = false,
    mambaRealDState = 4, mambaRealNGroups = 1, mambaRealConvKernel = 3,
    mambaRealExpand = 2)
  var mr = newNimformerModelJamba(24, C, 2, 2, "R", useSession = false,
    mambaRealDState = 4, mambaRealNGroups = 1, mambaRealConvKernel = 3,
    mambaRealExpand = 2)
  mp.blocks[0].mambaReal = mr.blocks[0].mambaReal
  mp.blocks[0].rms1 = mr.blocks[0].rms1
  mp.embed = mr.embed
  mp.rmsF = mr.rmsF
  mp.outProj = mr.outProj
  var x = newTensor(@[1, 2, C])
  var dOut = newTensor(@[1, 2, C])
  for i in 0 ..< x.data.len:
    x.data[i] = float32(sin(float64(i + 1) * 0.31))
    dOut.data[i] = float32(cos(float64(i + 2) * 0.23))
  let yP0 = forwardBlock(mp.blocks[0], x, ctx)
  let yR = forwardBlock(mr.blocks[0], x, ctx)
  var maxIdentityErr = 0'f32
  for i in 0 ..< yP0.data.len:
    maxIdentityErr = max(maxIdentityErr, abs(yP0.data[i] - yR.data[i]))
  if maxIdentityErr > 2e-6'f32:
    raise newException(AssertionDefect,
      &"P block zero-sidecar is not Mamba2Real identity: max err={maxIdentityErr}")
  let testIds = @[@[1, 2]]
  let pureLogits = forwardBatch(mr, testIds, ctx)
  let pIdentityLogits = forwardBatch(mp, testIds, ctx)
  let fullIdentityErr = maxAbsDiff(pureLogits, pIdentityLogits)
  if fullIdentityErr > 3e-6'f32:
    raise newException(AssertionDefect,
      &"Full R vs zero-sidecar P logits mismatch: max err={fullIdentityErr:.7g}")

  # Make the sidecar nonzero and deterministic so both parallel branches
  # contribute measurable gradients (the production init zeros this matrix).
  for i in 0 ..< mp.blocks[0].attn.qkvProj.weight.data.len:
    mp.blocks[0].attn.qkvProj.weight.data[i] =
      0.025'f32 * float32((i * 7 mod 19) - 9)
  for i in 0 ..< mp.blocks[0].attn.qkvProj.bias.data.len:
    mp.blocks[0].attn.qkvProj.bias.data[i] = 0.01'f32 * float32((i mod 5) - 2)
  for i in 0 ..< mp.blocks[0].attn.outProj.weight.data.len:
    mp.blocks[0].attn.outProj.weight.data[i] =
      0.04'f32 * float32((i * 5 mod 17) - 8)
  for i in 0 ..< mp.blocks[0].attn.outProj.bias.data.len:
    mp.blocks[0].attn.outProj.bias.data[i] = 0.015'f32 * float32((i mod 7) - 3)
  let pSidecarLogits = forwardBatch(mp, testIds, ctx)
  let sidecarLogitDiff = maxAbsDiff(pureLogits, pSidecarLogits)
  if sidecarLogitDiff < 1e-7'f32:
    raise newException(AssertionDefect,
      "Full R vs trained-sidecar P logits did not change; Linear++ branch may be disconnected")
  for value in pSidecarLogits.data:
    if value.classify in {fcNan, fcInf, fcNegInf}:
      raise newException(AssertionDefect,
        "Full Mamba2Real + Linear++ logits contain NaN/Inf")

  let xNormRef = mp.blocks[0].rms1.forward(x, ctx)
  let qkvRef = mp.blocks[0].attn.qkvProj.forward(xNormRef, ctx)
  let lpCoreRef = linearPlusForwardGpu(ctx, qkvRef.data, 1, 2,
    mp.blocks[0].attn.nHeads, mp.blocks[0].attn.headDim)
  let mm = mp.blocks[0].mambaReal
  let mLayout = m2Layout(1, 2, mm.numHeads, mm.headDim, mm.dState,
    mm.nGroups, mm.convKernel, mm.intermediateSize, mm.normEps, mm.nGroups)
  let lpLocal = lpLayout(1, 2, mp.blocks[0].attn.nHeads, mp.blocks[0].attn.headDim)
  var lpLocalScratch = newSeq[float32](lpLocal.total)
  lpLocalScratch.blit(lpLocal.qkv, qkvRef.data, 0, qkvRef.data.len)
  let lpLocalTg = lpPhiTg(lpLocal)
  discard runMicroVmChain(ctx, [
    (buildLpPhiProgram(lpLocal), lpLocal.BT * 2 * lpLocal.C,
      lpLocalTg * 4, lpLocalTg, "local_probe_lp_phi"),
    (buildLpDenProgram(lpLocal), lpLocal.B * lpLocal.H, 4, 0,
      "local_probe_lp_den"),
    (buildLpFwdProgram(lpLocal), lpLocal.B * lpLocal.C, 4, 0,
      "local_probe_lp_fwd")], lpLocalScratch)
  let lpBase = ((mLayout.total + 256 + 255) div 256) * 256
  let lpOffset = shiftedLpLayout(lpLocal, lpBase)
  var lpScratch = newSeq[float32](lpOffset.total + 256)
  lpScratch.blit(lpOffset.qkv, qkvRef.data, 0, qkvRef.data.len)
  let lpTg = lpPhiTg(lpOffset)
  discard runMicroVmChain(ctx, [
    (buildLpPhiProgram(lpOffset), 1 * 2 * 2 * lpOffset.C, lpTg * 4, lpTg, "offset_lp_phi"),
    (buildLpDenProgram(lpOffset), 1 * lpOffset.H, 4, 0, "offset_lp_den"),
    (buildLpFwdProgram(lpOffset), 1 * lpOffset.C, 4, 0, "offset_lp_fwd")], lpScratch)
  let lpCoreOffset = lpScratch[lpOffset.y ..< lpOffset.y + lpOffset.BT * lpOffset.C]
  var lpOffsetErr = 0'f32
  for i in 0 ..< lpCoreRef.len:
    lpOffsetErr = max(lpOffsetErr, abs(lpCoreRef[i] - lpCoreOffset[i]))
  proc regionErr(a: seq[float32]; aOff: int; b: seq[float32]; bOff, n: int): float32 =
    for i in 0 ..< n:
      result = max(result, abs(a[aOff + i] - b[bOff + i]))
  let phiErr = regionErr(lpLocalScratch, lpLocal.phi, lpScratch, lpOffset.phi,
    lpLocal.BT * 2 * lpLocal.C)
  let zErr = regionErr(lpLocalScratch, lpLocal.zF, lpScratch, lpOffset.zF,
    lpLocal.B * lpLocal.C)
  let cErr = regionErr(lpLocalScratch, lpLocal.cF, lpScratch, lpOffset.cF,
    lpLocal.B * lpLocal.C * lpLocal.D)
  let invErr = regionErr(lpLocalScratch, lpLocal.invDen, lpScratch, lpOffset.invDen,
    lpLocal.BT * lpLocal.H)
  let yLocalErr = regionErr(lpLocalScratch, lpLocal.y, lpScratch, lpOffset.y,
    lpLocal.BT * lpLocal.C)
  let yApiErr = regionErr(lpCoreRef, 0, lpLocalScratch, lpLocal.y,
    lpLocal.BT * lpLocal.C)
  if lpOffsetErr > 2e-5'f32 or phiErr > 2e-5'f32 or zErr > 2e-5'f32 or
      cErr > 2e-5'f32 or invErr > 2e-5'f32 or yLocalErr > 2e-5'f32 or
      yApiErr > 2e-5'f32:
    raise newException(AssertionDefect,
      &"Linear++ standalone vs prefixed layout mismatch: y={lpOffsetErr:.7g}, " &
       &"yLocal={yLocalErr:.7g}, yApi={yApiErr:.7g}, phi={phiErr:.7g}, " &
       &"z={zErr:.7g}, c={cErr:.7g}, invDen={invErr:.7g}")
  let mambaRef = forwardMamba2Real(mp.blocks[0].mambaReal, xNormRef, ctx).y
  let attnRef = forwardLinearPlus(mp.blocks[0].attn, xNormRef, ctx)
  let branchRef = addT(ctx, mambaRef, attnRef)
  let blockRef = addT(ctx, x, branchRef)
  let pKernel = forwardPBlockKernel(mp.blocks[0].mambaReal, mp.blocks[0].attn,
    x, mp.blocks[0].rms1.gamma, mp.blocks[0].rms1.eps, ctx,
    exportTerminalState = true)
  let mLay = pKernel.cache.mamba.lay
  # Inference-only cache export must preserve exactly the last recurrent
  # Mamba2 state and the final parallel-scan Linear++ sufficient statistics.
  let ssmFinalOff = mLay.states + 2 * mLay.HD * mLay.N
  var terminalStateErr = 0'f32
  for j in 0 ..< pKernel.cache.terminalSsmState.len:
    terminalStateErr = max(terminalStateErr, abs(
      pKernel.cache.terminalSsmState[j] - pKernel.cache.mamba.buf[ssmFinalOff + j]))
  let convHistory = mLay.K - 1
  var terminalConvErr = 0'f32
  for c in 0 ..< mLay.cDim:
    for k in 0 ..< convHistory:
      let t = 2 - convHistory + k
      let want = if t < 0: 0'f32
                 else: pKernel.cache.mamba.proj.data[t * mLay.P + mLay.C + c]
      terminalConvErr = max(terminalConvErr, abs(
        pKernel.cache.terminalConvState[c * convHistory + k] - want))
  let terminalZErr = regionErr(pKernel.cache.linearPlusZState, 0,
    lpLocalScratch, lpLocal.zF, lpLocal.B * lpLocal.C)
  let terminalScanErr = regionErr(pKernel.cache.linearPlusScanState, 0,
    lpLocalScratch, lpLocal.cF, lpLocal.B * lpLocal.C * lpLocal.D)

  # Decode consumes S[h,key,value], whereas the sequence scan stores its
  # final matrix as [h,value,key]. Compare the required transpose against an
  # independent token-by-token Linear++ recurrence from the same QKV rows.
  let lpH = mp.blocks[0].attn.nHeads
  let lpD = mp.blocks[0].attn.headDim
  var wantZ = newSeq[float32](lpH * lpD)
  var wantS = newSeq[float32](lpH * lpD * lpD)
  for t in 0 ..< 2:
    for h in 0 ..< lpH:
      for key in 0 ..< lpD:
        let idx = t * 3 * lpLocal.C + lpLocal.C + h * lpD + key
        let rawK = qkvRef.data[idx]
        let phiK = if rawK > 0'f32: rawK + 1'f32 else: float32(exp(float64(rawK)))
        wantZ[h * lpD + key] += phiK
        for value in 0 ..< lpD:
          let v = qkvRef.data[t * 3 * lpLocal.C + 2 * lpLocal.C + h * lpD + value]
          wantS[(h * lpD + key) * lpD + value] += phiK * v
  var importedS = newSeq[float32](wantS.len)
  for h in 0 ..< lpH:
    for value in 0 ..< lpD:
      for key in 0 ..< lpD:
        let scanIdx = (h * lpD + value) * lpD + key
        let decodeIdx = (h * lpD + key) * lpD + value
        importedS[decodeIdx] = pKernel.cache.linearPlusScanState[scanIdx]
  var terminalLpErr = 0'f32
  for j in 0 ..< wantZ.len:
    terminalLpErr = max(terminalLpErr, abs(pKernel.cache.linearPlusZState[j] - wantZ[j]))
  for j in 0 ..< wantS.len:
    terminalLpErr = max(terminalLpErr, abs(importedS[j] - wantS[j]))

  # Feed the exported prefix state into the exact single-step Linear++ VM
  # used by decode. Compare both output and updated state to an independent
  # scalar recurrence, so a scan/decode transpose or denominator mismatch
  # cannot pass merely because the saved arrays have the expected shape.
  let stepLay = fp4_gemv_vm.lpLayout(lpLocal.C, lpLocal.H)
  var stepBuf = newSeq[float32](stepLay.total)
  stepBuf.blit(stepLay.offProj, qkvRef.data,
    (lpLocal.T - 1) * 3 * lpLocal.C, 3 * lpLocal.C)
  stepBuf.blit(stepLay.offZ, pKernel.cache.linearPlusZState, 0,
    pKernel.cache.linearPlusZState.len)
  stepBuf.blit(stepLay.offS, importedS, 0, importedS.len)
  if not runMicroVm(ctx,
      fp4_gemv_vm.buildLinearPlusStepProgram(stepLay, 1e-6'f32), stepBuf,
      lpH * lpD, 16, "test_lp_prefill_decode_recurrence", lpD):
    raise newException(CatchableError,
      "Linear++ single-step decode kernel failed in prefill parity test")
  var lpDecodeYErr = 0'f32
  var lpDecodeStateErr = 0'f32
  for h in 0 ..< lpH:
    for value in 0 ..< lpD:
      var numerator = 0'f32
      var denominator = 0'f32
      for key in 0 ..< lpD:
        let row = (lpLocal.T - 1) * 3 * lpLocal.C
        let rawQ = qkvRef.data[row + h * lpD + key]
        let rawK = qkvRef.data[row + lpLocal.C + h * lpD + key]
        let rawV = qkvRef.data[row + 2 * lpLocal.C + h * lpD + value]
        let phiQ = if rawQ > 0'f32: rawQ + 1'f32 else: float32(exp(float64(rawQ)))
        let phiK = if rawK > 0'f32: rawK + 1'f32 else: float32(exp(float64(rawK)))
        let stateIdx = (h * lpD + key) * lpD + value
        let nextS = importedS[stateIdx] + phiK * rawV
        numerator += phiQ * nextS
        denominator += phiQ * (pKernel.cache.linearPlusZState[h * lpD + key] + phiK)
        lpDecodeStateErr = max(lpDecodeStateErr,
          abs(stepBuf[stepLay.offS + stateIdx] - nextS))
      let expected = numerator / (denominator + 1e-6'f32)
      lpDecodeYErr = max(lpDecodeYErr,
        abs(stepBuf[stepLay.offY + h * lpD + value] - expected))
    for key in 0 ..< lpD:
      let row = (lpLocal.T - 1) * 3 * lpLocal.C
      let rawK = qkvRef.data[row + lpLocal.C + h * lpD + key]
      let phiK = if rawK > 0'f32: rawK + 1'f32 else: float32(exp(float64(rawK)))
      let zIdx = h * lpD + key
      let expectedZ = pKernel.cache.linearPlusZState[zIdx] + phiK
      lpDecodeStateErr = max(lpDecodeStateErr,
        abs(stepBuf[stepLay.offZ + zIdx] - expectedZ))
  if terminalStateErr > 2e-5'f32 or terminalConvErr > 2e-6'f32 or
     terminalZErr > 2e-5'f32 or terminalScanErr > 2e-5'f32 or
     terminalLpErr > 2e-5'f32 or lpDecodeYErr > 2e-5'f32 or
     lpDecodeStateErr > 2e-5'f32:
    raise newException(AssertionDefect,
      &"parallel-prefill terminal-state export mismatch: ssm={terminalStateErr:.7g}, " &
       &"conv={terminalConvErr:.7g}, z={terminalZErr:.7g}, " &
       &"scan={terminalScanErr:.7g}, decode-layout={terminalLpErr:.7g}, " &
       &"decode-y={lpDecodeYErr:.7g}, decode-state={lpDecodeStateErr:.7g}")
  let mC = mp.blocks[0].mambaReal.intermediateSize
  let mNormedKernel = Tensor(shape: @[1, 2, mC],
    data: pKernel.cache.mamba.buf[mLay.normedOut ..< mLay.normedOut + 2*mC])
  let mambaKernel = mp.blocks[0].mambaReal.outProj.forward(mNormedKernel, ctx)
  let attnKernel = mp.blocks[0].attn.outProj.forward(pKernel.cache.linearPlusY, ctx)
  let blockKernel = pKernel.y
  var maxMambaError = 0'f32
  var maxAttnError = 0'f32
  for i in 0 ..< mambaRef.data.len:
    maxMambaError = max(maxMambaError, abs(mambaRef.data[i] - mambaKernel.data[i]))
    maxAttnError = max(maxAttnError, abs(attnRef.data[i] - attnKernel.data[i]))
  var maxBlockError = 0'f32
  for i in 0 ..< blockRef.data.len:
    maxBlockError = max(maxBlockError, abs(blockRef.data[i] - blockKernel.data[i]))
  if maxBlockError > 2e-5'f32:
    raise newException(AssertionDefect,
      &"P block shared-buffer forward differs from independent branches: total={maxBlockError:.7g}, " &
       &"Mamba={maxMambaError:.7g}, LinearPlus={maxAttnError:.7g}")

  var grads = initTable[string, Tensor]()
  let dX = backwardBlock(mp.blocks[0], x, dOut, ctx, "block0", grads)
  proc verify(label: string, analytic: float32, fd: float64) =
    let err = abs(float64(analytic) - fd)
    let tol = 5e-3'f64 + 2e-2'f64 * abs(fd)
    if err > tol:
      raise newException(AssertionDefect,
        &"P-block finite difference failed for {label}: analytic={analytic:.7g}, " &
         &"fd={fd:.7g}, abs err={err:.4g}, tol={tol:.4g}")
  const h = 2e-3'f32
  let xi = 11
  var xp = x; xp.data = cloneData(x.data); xp.data[xi] += h
  var xm = x; xm.data = cloneData(x.data); xm.data[xi] -= h
  verify("shared RMSNorm input gradient", dX.data[xi],
    (pBlockObjective(mp, xp, dOut, ctx) - pBlockObjective(mp, xm, dOut, ctx)) / float64(2*h))

  let qName = "block0.attn.qkvProj.weight"
  let qi = 2 * C * C + 5
  var qw = mp.blocks[0].attn.qkvProj.weight.data[qi]
  mp.blocks[0].attn.qkvProj.weight.data[qi] = qw + h
  let qp = pBlockObjective(mp, x, dOut, ctx)
  mp.blocks[0].attn.qkvProj.weight.data[qi] = qw - h
  let qm = pBlockObjective(mp, x, dOut, ctx)
  mp.blocks[0].attn.qkvProj.weight.data[qi] = qw
  verify("Linear++ QKV weight", grads[qName].data[qi], (qp - qm) / float64(2*h))

  let attnOutName = "block0.attn.outProj.weight"
  qw = mp.blocks[0].attn.outProj.weight.data[17]
  mp.blocks[0].attn.outProj.weight.data[17] = qw + h
  let aop = pBlockObjective(mp, x, dOut, ctx)
  mp.blocks[0].attn.outProj.weight.data[17] = qw - h
  let aom = pBlockObjective(mp, x, dOut, ctx)
  mp.blocks[0].attn.outProj.weight.data[17] = qw
  verify("Linear++ output projection weight", grads[attnOutName].data[17],
    (aop - aom) / float64(2*h))

  let mName = "block0.mambaReal.inProj.weight"
  let mi = 7
  qw = mp.blocks[0].mambaReal.inProj.weight.data[mi]
  mp.blocks[0].mambaReal.inProj.weight.data[mi] = qw + h
  let mpLoss = pBlockObjective(mp, x, dOut, ctx)
  mp.blocks[0].mambaReal.inProj.weight.data[mi] = qw - h
  let mmLoss = pBlockObjective(mp, x, dOut, ctx)
  mp.blocks[0].mambaReal.inProj.weight.data[mi] = qw
  verify("Mamba2Real inProj under parallel sidecar", grads[mName].data[mi],
    (mpLoss - mmLoss) / float64(2*h))
  let trainBatch = forwardBatch(mp, @[@[1, 2]], ctx)
  let inferBatch = forwardBatchSession(mp, @[@[1, 2]], ctx)
  let inferBatchErr = maxAbsDiff(trainBatch, inferBatch)
  if inferBatchErr > 2e-5'f32:
    raise newException(AssertionDefect,
      &"P block inference batch path mismatch: {inferBatchErr:.7g}")
  echo &"PASS - full R vs P logits: zero-sidecar err={fullIdentityErr:.2g}, " &
       &"trained-sidecar delta={sidecarLogitDiff:.2g}; P block identity ({maxIdentityErr:.2g}), branch parity " &
       &"({maxBlockError:.2g}), parallel-prefill terminal-state export " &
       &"(SSM={terminalStateErr:.2g}, conv={terminalConvErr:.2g}, " &
       &"Linear++={terminalLpErr:.2g}), one-step Linear++ decode parity " &
       &"(y={lpDecodeYErr:.2g}, state={lpDecodeStateErr:.2g}), and finite-difference probes"

proc checkGatedGBlock(ctx: Backend) =
  const C = 16
  randomize(77124)
  var mg = newNimformerModelJamba(24, C, 2, 2, "G", useSession = false,
    mambaRealDState = 4, mambaRealNGroups = 1, mambaRealConvKernel = 3,
    mambaRealExpand = 2)
  var mr = newNimformerModelJamba(24, C, 2, 2, "R", useSession = false,
    mambaRealDState = 4, mambaRealNGroups = 1, mambaRealConvKernel = 3,
    mambaRealExpand = 2)
  mg.blocks[0].mambaReal = mr.blocks[0].mambaReal
  mg.blocks[0].rms1 = mr.blocks[0].rms1
  for i, value in mg.blocks[0].fuseGate.weight.data:
    doAssert value == 0'f32, "G gate weight must start at zero"
  for value in mg.blocks[0].fuseGate.bias.data:
    doAssert value == 4'f32, "G gate bias must initially favor Mamba"

  var x = newTensor(@[1, 2, C])
  var dOut = newTensor(@[1, 2, C])
  for i in 0 ..< x.data.len:
    x.data[i] = float32(sin(float64(i + 1) * 0.31))
    dOut.data[i] = float32(cos(float64(i + 2) * 0.23))
  let yG0 = forwardBlock(mg.blocks[0], x, ctx)
  let yR = forwardBlock(mr.blocks[0], x, ctx)
  let initGate = 1'f32 / (1'f32 + exp(-mg.blocks[0].fuseGate.bias.data[0]))
  let initScaleErr = maxAbsDiff(yG0, yR) / max(1e-6'f32, maxAbsDiff(yR, newTensor(yR.shape)))
  if initGate < 0.98'f32 or initScaleErr > 0.025'f32:
    raise newException(AssertionDefect,
      &"G initialization is not near Mamba identity: gate={initGate:.4f}, relative max error={initScaleErr:.4f}")

  for i in 0 ..< mg.blocks[0].attn.qkvProj.weight.data.len:
    mg.blocks[0].attn.qkvProj.weight.data[i] =
      0.02'f32 * float32((i * 7 mod 19) - 9)
  for i in 0 ..< mg.blocks[0].attn.outProj.weight.data.len:
    mg.blocks[0].attn.outProj.weight.data[i] =
      0.002'f32 * float32((i * 5 mod 17) - 8)
  for i in 0 ..< mg.blocks[0].attn.outProj.bias.data.len:
    mg.blocks[0].attn.outProj.bias.data[i] = 0.001'f32 * float32((i mod 7) - 3)
  for i in 0 ..< mg.blocks[0].fuseGate.weight.data.len:
    mg.blocks[0].fuseGate.weight.data[i] = 0.004'f32 * float32((i mod 9) - 4)

  let xNorm = mg.blocks[0].rms1.forward(x, ctx)
  let mambaOut = forwardMamba2Real(mg.blocks[0].mambaReal, xNorm, ctx).y
  let sideOut = forwardLinearPlus(mg.blocks[0].attn, xNorm, ctx)
  var gateInput = newTensor(@[1, 2, 2 * C])
  for row in 0 ..< 2:
    for c in 0 ..< C:
      gateInput.data[row * 2 * C + c] = mambaOut.data[row * C + c]
      gateInput.data[row * 2 * C + C + c] = sideOut.data[row * C + c]
  let gateLogit = mg.blocks[0].fuseGate.forward(gateInput, ctx)
  let gate = beSigmoid(ctx, gateLogit.data)
  var expected = newTensor(x.shape)
  for i in 0 ..< expected.data.len:
    expected.data[i] = x.data[i] + gate[i] * mambaOut.data[i] +
      (1'f32 - gate[i]) * sideOut.data[i]
  let actual = forwardBlock(mg.blocks[0], x, ctx)
  let blendErr = maxAbsDiff(actual, expected)
  if blendErr > 2e-5'f32:
    raise newException(AssertionDefect, &"G explicit convex blend mismatch: {blendErr:.7g}")

  var grads = initTable[string, Tensor]()
  let dX = backwardBlock(mg.blocks[0], x, dOut, ctx, "block0", grads)
  proc verify(label: string; analytic: float32; fd: float64) =
    let err = abs(float64(analytic) - fd)
    let tol = 5e-3'f64 + 2e-2'f64 * abs(fd)
    if err > tol:
      raise newException(AssertionDefect,
        &"G-block finite difference failed for {label}: analytic={analytic:.7g}, " &
         &"fd={fd:.7g}, abs err={err:.4g}, tol={tol:.4g}")
  const h = 2e-3'f32
  var w = mg.blocks[0].fuseGate.bias.data[3]
  mg.blocks[0].fuseGate.bias.data[3] = w + h
  let bp = pBlockObjective(mg, x, dOut, ctx)
  mg.blocks[0].fuseGate.bias.data[3] = w - h
  let bm = pBlockObjective(mg, x, dOut, ctx)
  mg.blocks[0].fuseGate.bias.data[3] = w
  verify("fuseGate bias", grads["block0.fuseGate.bias"].data[3], (bp - bm) / float64(2 * h))

  let wi = C + 1
  w = mg.blocks[0].fuseGate.weight.data[wi]
  mg.blocks[0].fuseGate.weight.data[wi] = w + h
  let wp = pBlockObjective(mg, x, dOut, ctx)
  mg.blocks[0].fuseGate.weight.data[wi] = w - h
  let wm = pBlockObjective(mg, x, dOut, ctx)
  mg.blocks[0].fuseGate.weight.data[wi] = w
  verify("fuseGate weight", grads["block0.fuseGate.weight"].data[wi], (wp - wm) / float64(2 * h))

  let sideName = "block0.attn.outProj.weight"
  w = mg.blocks[0].attn.outProj.weight.data[17]
  mg.blocks[0].attn.outProj.weight.data[17] = w + h
  let sp = pBlockObjective(mg, x, dOut, ctx)
  mg.blocks[0].attn.outProj.weight.data[17] = w - h
  let sm = pBlockObjective(mg, x, dOut, ctx)
  mg.blocks[0].attn.outProj.weight.data[17] = w
  verify("Linear++ output projection", grads[sideName].data[17], (sp - sm) / float64(2 * h))

  var xp = x; xp.data = cloneData(x.data); xp.data[7] += h
  var xm = x; xm.data = cloneData(x.data); xm.data[7] -= h
  verify("input + residual gradient", dX.data[7],
    (pBlockObjective(mg, xp, dOut, ctx) - pBlockObjective(mg, xm, dOut, ctx)) / float64(2 * h))

  var decodeModel = newNimformerModelJamba(24, 64, 2, 2, "G", useSession = false,
    mambaRealDState = 4, mambaRealNGroups = 1, mambaRealConvKernel = 3,
    mambaRealExpand = 2)
  var decodeLora: JambaLora
  decodeLora.adapters = initTable[string, LoraAdapter]()
  decodeLora.backup = initTable[string, seq[float32]]()
  decodeLora.quantBase = initTable[string, QuantTensor]()
  for name, lin in paramLinears(decodeModel):
    decodeLora.quantBase[name] = quantizeTensor(lin[].weight.data,
      lin[].weight.shape, qkInt4Asymmetric)
  bindQuantBaseToLinears(decodeModel, decodeLora)
  prepareGpuModel(decodeModel, decodeLora, ctx)
  for tokenId in [1, 2]:
    let logits = forwardStepJambaGpu(decodeModel, tokenId, ctx)
    for value in logits.data:
      if value.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(AssertionDefect, "G Metal token decoder produced non-finite logits")

  for i in 0 ..< mg.blocks[0].fuseGate.weight.data.len:
    mg.blocks[0].fuseGate.weight.data[i] = 0'f32
  for i in 0 ..< mg.blocks[0].fuseGate.bias.data.len:
    mg.blocks[0].fuseGate.bias.data[i] = 4'f32
  for i in 0 ..< mg.blocks[0].attn.outProj.bias.data.len:
    mg.blocks[0].attn.outProj.bias.data[i] = 100'f32
  var rejectedExplodingSidecar = false
  try:
    discard forwardBlock(mg.blocks[0], x, ctx, blockIndex = 23)
  except CatchableError as e:
    rejectedExplodingSidecar = "G block 23" in e.msg and "weighted Linear++ RMS" in e.msg
  if not rejectedExplodingSidecar:
    raise newException(AssertionDefect,
      "G per-layer safety check did not reject a deliberately exploding Linear++ sidecar")
  echo "PASS - G safety guard rejects an exploding Linear++ contribution with its layer index"
  echo &"PASS - G Mamba-first initialization, convex blend (err={blendErr:.2g}), " &
       "gate/sidecar/input finite differences, and two Metal token-decode steps"

when isMainModule:
  setForbidCpuFallback(true)
  # This framework intentionally has no CPU backend; the test compares the
  # full GPU streaming path against the non-streaming reference.
  let ctx = newBackend("auto")
  checkGatedGBlock(ctx)
  checkLoraGradientFiniteDifference(ctx)
  checkRBlock(ctx)
  checkParallelPBlock(ctx)

  let vocab = 40; let hidden = 16; let nHeads = 2
  let pattern = "RARA"

  # ── model A: fp32-resident binh thuong (duong cu) ──────────────
  var mA = newNimformerModelJamba(vocab, hidden, nHeads, 2, pattern,
                                   useSession = false, mambaRealDState = 4,
                                   mambaRealNGroups = 1, mambaRealConvKernel = 3,
                                   mambaRealExpand = 2)

  # ── model B: BAT DAU TU CUNG WEIGHT HET (copy tu mA), roi "quant-hoa"
  # bang qkFp32Raw (KHONG mat mat gi - dequant = y het gia tri goc) de tach
  # rieng "logic streaming dung khong" khoi "fp4 mat mat bao nhieu" (da test
  # rieng o test_nimq_roundtrip.nim / test_lut.nim roi) ──────────────
  var mB = mA   # Nim object gan = deep-copy (khong phai ref) - 2 model doc lap

  var quantWeights = initTable[string, QuantTensor]()
  var loraTargetsA: seq[string] = @[]
  for i, c in pattern:
    if c == 'R':
      let inName = &"block{i}.mambaReal.inProj.weight"
      let outName = &"block{i}.mambaReal.outProj.weight"
      loraTargetsA.add(inName); loraTargetsA.add(outName)
      for (name, t) in paramList(mB):
        if name == inName or name == outName:
          quantWeights[name] = quantizeTensor(t[].data, t[].shape, qkFp32Raw)

  # Resume simulation: new-block attention weights exist in the QLoRA quant
  # manifest too, while their mutable FP32 masters are stored separately.
  # Exact fp32-raw quantization isolates the stale-useQuant bug from lossy Q4.
  for (name, t) in paramList(mB):
    if name.startsWith("block1.") or name.startsWith("block3."):
      quantWeights[name] = quantizeTensor(t[].data, t[].shape, qkFp32Raw)

  var loA = attachLora(mA, loraTargetsA, r = 4, alpha = 8'f32)
  var loB = attachLoraQuantResident(mB, quantWeights, loraTargetsA, r = 4, alpha = 8'f32)
  bindQuantBaseToLinears(mB, loB)
  # ep A/B cua loB GIONG HET loA (random-init khac nhau moi lan goi newLoraAdapter)
  for name in loA.adapters.keys:
    loB.adapters[name].A.data = loA.adapters[name].A.data
    loB.adapters[name].B.data = loA.adapters[name].B.data

  var newBlockAdamA = initTable[string, ApfAdamState]()
  for (name, t) in paramList(mA):
    if "block1." in name or "block3." in name:   # block 'A' o vi tri 1,3 trong "RARA"
      newBlockAdamA[name] = newApfAdamState(t[].data.len)
  var stateA = newLoraTrainState(loA, lr = 1e-2'f32)
  var stateB = newLoraTrainState(loB, lr = 1e-2'f32)
  let testRoot = getTempDir() / ("nimformers-streaming-matches-" & $getCurrentProcessId())
  createDir(testRoot)
  var adamStoreB = initAdamDiskStore(testRoot / "adam")
  var attnStoreB = initAttnWeightDiskStore(testRoot / "attn")

  let idsBatch = @[@[1, 2, 3, 4, 5]]
  let targetsBatch = @[@[2, 3, 4, 5, 6]]

  proc trainStepGraftRef(m: var NimformerModelJamba, lo: var JambaLora,
                         loState: var LoraTrainState,
                         newBlockAdam: var Table[string, ApfAdamState],
                         newBlockLr: float32,
                         idsBatch, targetsBatch: seq[seq[int]], ctx: Backend): float32 =
    mergeIn(m, lo, ctx)
    let (loss, grads) = computeLossAndGrads(m, idsBatch, targetsBatch, ctx)
    for name, ad in lo.adapters.mpairs:
      if name notin grads: continue
      let (dA, dB) = gradToAB(grads[name], ad, ctx)
      var stA = loState.adamA[name]
      var stB = loState.adamB[name]
      discard apfAdamStepGpu(ctx, ad.A, dA, stA, loState.lr, requantizeEvery = int.high)
      discard apfAdamStepGpu(ctx, ad.B, dB, stB, loState.lr, requantizeEvery = int.high)
      loState.adamA[name] = stA
      loState.adamB[name] = stB
    for (name, t) in paramList(m):
      if name in lo.adapters: continue
      if name notin grads: continue
      if name notin newBlockAdam: continue
      var st = newBlockAdam[name]
      discard apfAdamStepGpu(ctx, t[], grads[name], st, newBlockLr, requantizeEvery = int.high)
      newBlockAdam[name] = st
    restoreBase(m, lo)
    result = loss

  var lossA = trainStepGraftRef(mA, loA, stateA, newBlockAdamA, 3e-2'f32, idsBatch, targetsBatch, ctx)
  var lossB = trainStepGraftStreaming(mB, loB, stateB, adamStoreB, attnStoreB,
                                      3e-2'f32, idsBatch, targetsBatch, ctx)
  # The second step is essential: it verifies that the fp32 master written by
  # Adam in step 1, rather than the stale quant snapshot, drives resumed-style
  # forward and backward computations after spill/reload.
  lossA = trainStepGraftRef(mA, loA, stateA, newBlockAdamA, 3e-2'f32, idsBatch, targetsBatch, ctx)
  lossB = trainStepGraftStreaming(mB, loB, stateB, adamStoreB, attnStoreB,
                                  3e-2'f32, idsBatch, targetsBatch, ctx)

  echo &"loss non-streaming = {lossA:.6f}"
  echo &"loss streaming     = {lossB:.6f}"
  echo &"chenh lech loss    = {abs(lossA-lossB):.6f}"

  var maxAdapterDiff = 0'f32
  for name in loA.adapters.keys:
    for i in 0 ..< loA.adapters[name].A.data.len:
      maxAdapterDiff = max(maxAdapterDiff, abs(loA.adapters[name].A.data[i] - loB.adapters[name].A.data[i]))
    for i in 0 ..< loA.adapters[name].B.data.len:
      maxAdapterDiff = max(maxAdapterDiff, abs(loA.adapters[name].B.data[i] - loB.adapters[name].B.data[i]))
  var maxNewBlockDiff = 0'f32
  for (name, t) in paramList(mA):
    if not (name.startsWith("block1.") or name.startsWith("block3.")): continue
    let streamed = attnStoreB.loadW(name, t[].data.len)
    for i in 0 ..< t[].data.len:
      maxNewBlockDiff = max(maxNewBlockDiff, abs(t[].data[i] - streamed[i]))
  echo &"chenh lech LoRA A/B sau 1 buoc Adam = {maxAdapterDiff:.6f}"
  echo &"chenh lech new-block sau 1 buoc Adam = {maxNewBlockDiff:.6f}"

  var quantFlagMismatch = false
  for name, enabled in paramQuantFlags(mB):
    if name.startsWith("block1.attn.") or name.startsWith("block3.attn."):
      if not enabled[]: quantFlagMismatch = true

  if abs(lossA-lossB) < 1e-3'f32 and maxAdapterDiff < 1e-3'f32 and
     maxNewBlockDiff < 1e-3'f32 and not quantFlagMismatch:
    echo "PASS - streaming cho ket qua GIONG HET non-streaming"
  else:
    echo "FAIL - streaming SAI, can xem lai"
    quit(1)
