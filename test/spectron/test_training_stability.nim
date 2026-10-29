## Run with the same --path flags as test_nimformer, on a Metal GPU.
## Uses temporary checkpoints only; never modifies the training checkpoint.
import std/[math, os, random, strformat, tempfiles]
import backend, nimformer, test_nimformer

proc referenceLoss(logits: Tensor, targets: seq[seq[int]]): float64 =
  let V = logits.shape[^1]
  for b in 0 ..< targets.len:
    for t in 0 ..< targets[b].len:
      let off = (b * targets[b].len + t) * V
      var mx = -Inf
      for v in 0 ..< V: mx = max(mx, float64(logits.data[off + v]))
      var sumExp = 0.0
      for v in 0 ..< V: sumExp += exp(float64(logits.data[off + v]) - mx)
      result += (mx - float64(logits.data[off + targets[b][t]])) + ln(sumExp)
  result /= float64(targets.len * targets[0].len)

proc maxDiff(a, b: seq[float32]): float32 =
  doAssert a.len == b.len
  for i in 0 ..< a.len: result = max(result, abs(a[i] - b[i]))

randomize(42)
setForbidCpuFallback(true)
let ctx = newBackend("metal")
let ids = @[@[0, 1, 2, 3], @[2, 3, 0, 1]]
let targets = @[@[1, 2, 3, 0], @[3, 0, 1, 2]]
var model = newNimformerModelSpectron(16, 8, 2, 1, 2)

block largeSsmResiduals:
  var large = newNimformerModelSpectron(16, 64, 4, 2, 2)
  for i in 0 ..< large.blocks[0].mamba.outProj.weight.data.len:
    large.blocks[0].mamba.outProj.weight.data[i] *= 100'f32
  large.useFinalNorm = false
  let raw = large.forwardBatch(ids, ctx)
  large.useFinalNorm = true
  let normalized = large.forwardBatch(ids, ctx)
  var rawMax, normalizedMax = 0'f32
  for x in raw.data: rawMax = max(rawMax, abs(x))
  for x in normalized.data: normalizedMax = max(normalizedMax, abs(x))
  doAssert rawMax > 100 * normalizedMax
  doAssert normalizedMax < 20
  echo "PASS: large SSM residuals do not amplify lm-head logits"

block lossAndGradients:
  let logits = model.forwardBatch(ids, ctx)
  let ce = softmaxCrossEntropy(logits, targets, ctx)
  doAssert abs(float64(ce.loss) - referenceLoss(logits, targets)) < 1e-4
  let grads = model.backwardBatch(ids, ce.dLogits, ctx)
  template checkGradient(label: string, value: untyped, analytic: float32) =
    block:
      let old = value
      let eps = 1e-3'f32
      value = old + eps
      let plus = referenceLoss(model.forwardBatch(ids, ctx), targets)
      value = old - eps
      let minus = referenceLoss(model.forwardBatch(ids, ctx), targets)
      value = old
      let numeric = (plus - minus) / float64(2 * eps)
      let error = abs(numeric - float64(analytic))
      doAssert error < 2e-3 + 0.02 * abs(numeric),
        label & ": analytic=" & $analytic & ", numeric=" & $numeric
  checkGradient("finalNorm.gamma", model.finalNorm.gamma.data[0], grads.dFinalNormG.data[0])
  checkGradient("finalNorm.beta", model.finalNorm.beta.data[1], grads.dFinalNormB.data[1])
  checkGradient("outProj.weight", model.outProj.weight.data[4], grads.dOutProjW.data[4])
  checkGradient("embed.weight", model.embed.weight.data[3], grads.dEmbed.data[3])
  checkGradient("mamba.inProj.weight", model.blocks[0].mamba.inProj.weight.data[5],
                grads.dBlocks[0].dMambaInW.data[5])
  echo "PASS: GPU loss and finite-difference gradients through final LayerNorm"

block checkpoints:
  let path = createTempDir("spectron-stability-", "") / "model.nimq"
  defer:
    removeFile(path)
    removeDir(parentDir(path))
  for normalized in [false, true]:
    model.useFinalNorm = normalized
    let before = model.forwardBatch(ids, ctx)
    model.saveCheckpoint(path, 123)
    let loaded = loadCheckpoint(path)
    doAssert loaded.step == 123
    doAssert loaded.model.useFinalNorm == normalized
    doAssert loaded.model.embed.weight.data == model.embed.weight.data
    doAssert loaded.model.outProj.weight.data == model.outProj.weight.data
    doAssert loaded.model.finalNorm.gamma.data == model.finalNorm.gamma.data
    doAssert loaded.model.finalNorm.beta.data == model.finalNorm.beta.data
    let after = loaded.model.forwardBatch(ids, ctx)
    var scale = 1'f32
    for value in before.data: scale = max(scale, abs(value))
    let diff = maxDiff(before.data, after.data)
    doAssert diff < 1e-5'f32 * scale, &"Checkpoint diff={diff}, scale={scale}"
  echo "PASS: legacy and normalized checkpoints preserve inference"

block stableTraining:
  var adam = newModelAdamState(model)
  let initial = softmaxCrossEntropy(model.forwardBatch(ids, ctx), targets, ctx).loss
  for step in 1 .. 110:
    let loss = trainStep(model, adam, ids, targets, ctx, lr = 3e-3'f32,
                         requantizeEvery = 50)
    doAssert loss.classify notin {fcNan, fcInf, fcNegInf}
    if step mod 25 == 0: echo &"step {step} loss={loss:.4f}"
  let final = softmaxCrossEntropy(model.forwardBatch(ids, ctx), targets, ctx).loss
  doAssert final < initial * 0.5'f32, &"Loss did not fall enough: {initial} -> {final}"
  echo &"PASS: training across APF steps 50/100: {initial:.4f} -> {final:.4f}"

closeBackend(ctx)
