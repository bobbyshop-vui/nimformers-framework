## Differential GPU test: resident FP4+LoRA R/P/G prefill must install the
## same recurrent states and next-token logits as tokenwise decoding.
import std/[math, random, tables, strformat, os, strutils, algorithm]
import backend, nimformer, quant, jamba_hybrid, lora, prefill_gpu
import mamba2_fused_step, gpu_layers
import vendor/bybylang/gpubackend as byby

proc adapter(n: int; phase: float): seq[float32] =
  result = newSeq[float32](n)
  for i in 0 ..< n: result[i] = float32(sin(float(i) * 0.17 + phase) * 0.01)

proc installQuant(lin: var Linear; phase: float) =
  lin.setQuantWeight(quantizeFp4KM(lin.weight.data, lin.weight.shape))
  lin.loraR = 4
  lin.loraScale = if phase < 1: 1.7'f32 else: -0.6'f32
  lin.loraA = adapter(4 * lin.inF, phase)
  lin.loraB = adapter(lin.outF * 4, phase + 0.7)

proc state(ctx: Backend): seq[float32] =
  let be = ctx.kind.toByby()
  for i in 0 ..< 4:
    let l = gLayersV2[i]
    let work = byby.sessionRead(be, l.hWork, l.off.total)
    result.add work[l.off.offConvSt ..< l.off.offConvSt + l.cDim * (l.K - 1)]
    result.add work[l.off.offSsm ..< l.off.offSsm + l.HD * l.N]
    if i in gAttnLayers:
      let a = gAttnLayers[i]
      let aw = byby.sessionRead(be, a.hWork, a.lay.total)
      result.add aw[a.lay.offZ ..< a.lay.offZ + a.lay.H * a.lay.D]
      result.add aw[a.lay.offS ..< a.lay.offS + a.lay.H * a.lay.D * a.lay.D]

proc compare(name: string; a, b: seq[float32]; tolerance: float32) =
  doAssert a.len == b.len
  var err, mag: float32
  var worst = 0
  for i in 0 ..< a.len:
    doAssert classify(a[i]) notin {fcNan, fcInf, fcNegInf}
    doAssert classify(b[i]) notin {fcNan, fcInf, fcNegInf}
    if abs(a[i] - b[i]) > err:
      err = abs(a[i] - b[i]); worst = i
    mag = max(mag, max(abs(a[i]), abs(b[i])))
  let relative = err / max(mag, 1e-6'f32)
  echo &"{name}: maxAbs={err:.4g} relative={relative:.4g}"
  if relative >= tolerance: echo &"worst={worst} ref={a[worst]} resident={b[worst]}"
  doAssert relative < tolerance, name & " differs from tokenwise reference"

when isMainModule:
  putEnv("BYBY_ONE_DISPATCH", "0")
  randomize(143)
  let ctx = newBackend(getEnv("BYBY_TEST_BACKEND", "metal"))
  # Nonuniform projection widths exercise packed FP4 and FP32 sidecars.
  var m = newNimformerModelJamba(64, 512, 8, 2, "RGPR",
    mambaRealDState = 4, mambaRealNGroups = 2, mambaRealExpand = 2)
  for i in 0 ..< m.blocks.len:
    installQuant(m.blocks[i].mambaReal.inProj, 0.1)
    installQuant(m.blocks[i].mambaReal.outProj, 1.1)
    for j in 0 ..< m.blocks[i].ln1.gamma.data.len:
      m.blocks[i].ln1.gamma.data[j] = 0.8'f32 + float32(j mod 11) * 0.025'f32
    if m.blocks[i].kind in {bkMambaRealGated, bkMambaRealPlusAttn}:
      # Nonzero sidecar and gate weights/biases exercise the actual mixing,
      # rather than the constructor's zero-output identity sidecar.
      for j in 0 ..< m.blocks[i].attn.outProj.weight.data.len:
        m.blocks[i].attn.outProj.weight.data[j] = float32(sin(float(j) * 0.19) * 0.025)
      for j in 0 ..< m.blocks[i].attn.qkvProj.bias.data.len:
        m.blocks[i].attn.qkvProj.bias.data[j] = float32(j mod 7 - 3) * 0.01'f32
    if m.blocks[i].kind == bkMambaRealGated:
      for j in 0 ..< m.blocks[i].fuseGate.weight.data.len:
        m.blocks[i].fuseGate.weight.data[j] = float32(cos(float(j) * 0.07) * 0.015)
  var lo: JambaLora
  let qkvWeights = m.blocks[1].attn.qkvProj.weight.data
  let qkvBias = m.blocks[1].attn.qkvProj.bias.data
  doAssert setGpuNormPreset("rms-fin-gn")
  prepareGpuModel(m, lo, ctx)
  doAssert m.blocks[1].attn.qkvProj.weight.data.len == 0
  var x = newTensor(@[1, 3, m.embedDim])
  for i in 0 ..< x.data.len: x.data[i] = float32(sin(float(i) * 0.03))
  let actual = m.blocks[1].attn.qkvProj.forward(x, ctx)
  var expected = newSeq[float32](actual.data.len)
  for row in 0 ..< 3:
    for n in 0 ..< 3 * m.embedDim:
      var sum = 0.0
      for k in 0 ..< m.embedDim:
        sum += float64(x.data[row * m.embedDim + k]) * float64(qkvWeights[n * m.embedDim + k])
      expected[row * 3 * m.embedDim + n] = float32(sum) + qkvBias[n]
  compare("resident FP32 sequence forward", expected, actual.data, 5e-4'f32)
  doAssert residentPrefillReason(m, ctx).len == 0
  # Warm recurrent kernels before taking the reference states.
  discard forwardStepJambaGpu(m, 1, ctx)
  for tokens in [2, 3, 7, 15, 33, 64]:
    if getEnv("RESIDENT_TEST_T").len > 0 and tokens != parseInt(getEnv("RESIDENT_TEST_T")): continue
    var ids: seq[int]
    for t in 0 ..< tokens: ids.add((t * 7 + 1) mod 64)
    resetGpuGenState(m, ctx)
    for id in ids: discard forwardStepJambaGpu(m, id, ctx, produceLogits = false)
    let referenceState = state(ctx)
    let referenceLogits = forwardStepJambaGpu(m, 11, ctx).data
    resetGpuGenState(m, ctx)
    forwardResidentPrefill(m, ids, ctx)
    markGpuPrefillReady()
    compare("state T=" & $tokens, referenceState, state(ctx), 5e-4'f32)
    compare("next logits T=" & $tokens, referenceLogits,
      forwardStepJambaGpu(m, 11, ctx).data, 5e-4'f32)
    if tokens == 15:
      # Same arena, different token values: no allocations or stale state.
      let before = prefillBufferStats()
      ids.reverse()
      resetGpuGenState(m, ctx)
      for id in ids: discard forwardStepJambaGpu(m, id, ctx, produceLogits = false)
      let reversedState = state(ctx)
      resetGpuGenState(m, ctx)
      forwardResidentPrefill(m, ids, ctx)
      let after = prefillBufferStats()
      doAssert after.allocations == before.allocations
      doAssert after.hits - before.hits == after.buffers
      compare("cached arena, changed prompt", reversedState, state(ctx), 5e-4'f32)
  echo "PASS resident prefill: R/G/P, FP4 + nonzero LoRA, padding, short conv history, and next-token decode"
