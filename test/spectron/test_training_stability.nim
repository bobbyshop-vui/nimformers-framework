## Run with the same --path flags as test_nimformer, on a Metal GPU.
## Uses temporary checkpoints only; never modifies the training checkpoint.
import std/[math, os, random, strformat, tempfiles]
import backend, nimformer, test_nimformer, block_decode

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

# Stand-ins for a causal model in block decoding: a prediction for block
# position j depends only on the j block tokens before it.
type
  TablePredictor = object
    ## A token and a confident flag for every position j and every sequence of
    ## j earlier block tokens; `offsets[j]` is where position j's rows start.
    vocab: int
    offsets: seq[int]
    tokens: seq[int]
    confident: seq[bool]
  HashedPredictor = object
    ## The same kind of table for larger sizes, derived from a hash of the prefix.
    vocab: int
    seed: uint64
  # Each token is the previous token plus 1, modulo 2, always confident.
  ChainPredictor = object

proc prediction(token: int, confident: bool): Prediction =
  (token: token, confidence: (if confident: 1'f32 else: 0'f32), margin: 1'f32)

proc predictAt(t: TablePredictor, j: int, drafts: seq[int]): Prediction =
  var key = 0
  for i in 0 ..< j: key = key * t.vocab + drafts[i]
  key += t.offsets[j]
  prediction(t.tokens[key], t.confident[key])

proc predictAt(h: HashedPredictor, j: int, drafts: seq[int]): Prediction =
  var x = h.seed * 0x9E3779B97F4A7C15'u64 + uint64(j + 1)
  for i in 0 ..< j:
    x = (x xor (uint64(drafts[i]) + 0x632BE5AB'u64)) * 0x100000001B3'u64
    x = x xor (x shr 29)
  prediction(int(x mod uint64(h.vocab)), (x shr 40) mod 10 < 7)

proc predictAt(c: ChainPredictor, j: int, drafts: seq[int]): Prediction =
  prediction(if j == 0: 0 else: (drafts[j - 1] + 1) mod 2, true)

proc greedyOf[P](p: P, size: int): seq[int] =
  for j in 0 ..< size: result.add(p.predictAt(j, result).token)

proc decodeWith[P](p: P, drafts: seq[int], policy: BlockPolicy,
                   stats: var BlockDecodeStats): tuple[output: seq[int], passes: int] =
  ## Runs the block update used by `blockDecode`, with predictions read from `p`.
  var b = initDraftBlock(drafts)
  var predictions = newSeq[Prediction](drafts.len)
  while b.verified < drafts.len:
    inc result.passes
    for j in 0 ..< drafts.len: predictions[j] = p.predictAt(j, b.drafts)
    b.applyPass(predictions, policy, 0.9'f32, stats)
  result.output = b.drafts

block blockDecodeLogic:
  # Every table, with every initial draft, must give the greedy output in at most
  # K passes; a wrong locked draft must be replaced when verification reaches it.
  for (vocab, size, expected) in [(2, 1, 8), (2, 2, 256), (3, 2, 11664), (2, 3, 131072)]:
    var t = TablePredictor(vocab: vocab, offsets: newSeq[int](size + 1))
    for j in 0 ..< size: t.offsets[j + 1] = t.offsets[j] + vocab ^ j
    let keys = t.offsets[size]
    t.tokens = newSeq[int](keys)
    t.confident = newSeq[bool](keys)
    var stats: BlockDecodeStats
    var cases, worst = 0
    for tokenCode in 0 ..< vocab ^ keys:
      var code = tokenCode
      for k in 0 ..< keys:
        t.tokens[k] = code mod vocab
        code = code div vocab
      for flags in 0 ..< 1 shl keys:
        for k in 0 ..< keys: t.confident[k] = ((flags shr k) and 1) == 1
        let greedy = t.greedyOf(size)
        for draftCode in 0 ..< vocab ^ size:
          var drafts = newSeq[int](size)
          var c = draftCode
          for i in 0 ..< size:
            drafts[i] = c mod vocab
            c = c div vocab
          for policy in [bpJacobi, bpConfidence]:
            let (output, passes) = t.decodeWith(drafts, policy, stats)
            doAssert output == greedy and passes in 1 .. size,
              &"vocab={vocab} block={size} drafts={drafts} {policy}: {output} in {passes} passes"
            worst = max(worst, passes)
          inc cases
    # vocab^keys token tables, 2^keys flag tables and vocab^size initial drafts.
    doAssert cases == vocab ^ keys * (1 shl keys) * vocab ^ size and cases == expected
    doAssert stats.rightLocks + stats.wrongLocks == stats.locks
    echo &"PASS: block update on every table, vocab={vocab} block={size}: {cases} cases, " &
         &"worst {worst} passes, {stats.wrongLocks} wrong locks replaced by verification"
  var rng = initRand(2026)
  for (vocab, size) in [(4, 4), (8, 6), (16, 8)]:
    var stats: BlockDecodeStats
    var worst = 0
    for _ in 1 .. 3000:
      let h = HashedPredictor(vocab: vocab, seed: uint64(rng.rand(int.high)))
      let greedy = h.greedyOf(size)
      var drafts = newSeq[int](size)
      for token in drafts.mitems: token = rng.rand(vocab - 1)
      for policy in [bpJacobi, bpConfidence]:
        let (output, passes) = h.decodeWith(drafts, policy, stats)
        doAssert output == greedy and passes in 1 .. size
        worst = max(worst, passes)
    doAssert stats.rightLocks + stats.wrongLocks == stats.locks
    echo &"PASS: block update on 3000 random tables, vocab={vocab} block={size}: " &
         &"worst {worst} passes, {stats.wrongLocks} wrong locks replaced by verification"
  for size in 1 .. 12:
    var stats: BlockDecodeStats
    var worst = 0
    let greedy = ChainPredictor().greedyOf(size)
    for draftCode in 0 ..< 1 shl size:
      var drafts = newSeq[int](size)
      for i in 0 ..< size: drafts[i] = (draftCode shr i) and 1
      for policy in [bpJacobi, bpConfidence]:
        let (output, passes) = ChainPredictor().decodeWith(drafts, policy, stats)
        doAssert output == greedy
        worst = max(worst, passes)
    doAssert worst == size, &"chain block={size}: worst {worst} passes"
  echo "PASS: the worst initial draft of the mod-2 chain takes exactly K passes for K = 1..12"

randomize(42)
setForbidCpuFallback(true)
let ctx = newBackend("metal")
let ids = @[@[0, 1, 2, 3], @[2, 3, 0, 1]]
let targets = @[@[1, 2, 3, 0], @[3, 0, 1, 2]]
var model = newNimformerModelSpectron(16, 8, 2, 1, 2)

block extremeLogits:
  let cpu = newBackend("cpu")
  for device in [cpu, ctx]:
    for offset in [1e8'f32, -1e35'f32]:
      let logits = Tensor(shape: @[1, 2, 4], data: @[
        offset, offset, offset, offset, NaN.float32, Inf.float32, 0'f32, 0'f32])
      let result = softmaxCrossEntropy(logits, @[@[1, -1]], device)
      doAssert abs(result.loss - ln(4'f32)) < 1e-4
      doAssert maxDiff(result.dLogits.data, @[0.25'f32, -0.75'f32, 0.25'f32, 0.25'f32,
                                             0'f32, 0'f32, 0'f32, 0'f32]) < 1e-5
      var rejected = false
      try:
        discard softmaxCrossEntropy(logits, @[@[1, 2]], device)
      except SpectronNumericalError:
        rejected = true
      doAssert rejected
    let logits = Tensor(shape: @[1, 2, 2], data: @[0'f32, -2e38'f32, 0'f32, -2e38'f32])
    let mean = softmaxCrossEntropy(logits, @[@[1, 1]], device).loss
    doAssert abs(mean / 2e38'f32 - 1) < 1e-5
  echo "PASS: stable cross-entropy for extreme, masked and non-finite logits on CPU and Metal"

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

block invalidAdamUpdate:
  var guarded = model
  var adam = newModelAdamState(guarded)
  # The last update fails after earlier parameters have been updated in the candidate.
  adam.blocks[^1].ln2B.v[0] = -1
  var rejected = false
  try:
    discard trainStep(guarded, adam, ids, targets, ctx, maxGradNorm = 1)
  except SpectronNumericalError:
    rejected = true
  doAssert rejected
  doAssert guarded.embed.weight.data == model.embed.weight.data
  doAssert guarded.outProj.weight.data == model.outProj.weight.data
  doAssert guarded.blocks[0].ff.fc1.weight.data == model.blocks[0].ff.fc1.weight.data
  doAssert adam.outProjW.step == 0 and adam.blocks[^1].ln2B.v[0] == -1
  doAssert trainStep(guarded, adam, ids, @[@[-1, -1, -1, -1], @[-1, -1, -1, -1]], ctx) == 0
  doAssert adam.outProjW.step == 0
  echo "PASS: rejected and all-masked batches preserve weights and Adam state"

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

block hrmCore:
  var hrm = newNimformerModelSpectron(16, 8, 2, 2, 2)
  # With one cycle and one low-level step the one-step gradient is exact.
  hrm.hrm = HrmConfig(lLayers: 1, cycles: 1, lSteps: 1, segments: 1)
  let ce = softmaxCrossEntropy(hrm.forwardBatch(ids, ctx), targets, ctx)
  let grads = hrm.backwardBatch(ids, ce.dLogits, ctx)
  template checkHrmGradient(label: string, value: untyped, analytic: float32) =
    block:
      let old = value
      let eps = 1e-3'f32
      value = old + eps
      let plus = referenceLoss(hrm.forwardBatch(ids, ctx), targets)
      value = old - eps
      let minus = referenceLoss(hrm.forwardBatch(ids, ctx), targets)
      value = old
      let numeric = (plus - minus) / float64(2 * eps)
      doAssert abs(numeric - float64(analytic)) < 2e-3 + 0.02 * abs(numeric),
        label & ": analytic=" & $analytic & ", numeric=" & $numeric
  checkHrmGradient("embed.weight", hrm.embed.weight.data[3], grads.dEmbed.data[3])
  checkHrmGradient("L mamba.inProj.weight", hrm.blocks[0].mamba.inProj.weight.data[5],
                   grads.dBlocks[0].dMambaInW.data[5])
  checkHrmGradient("L ff.fc1.weight", hrm.blocks[0].ff.fc1.weight.data[7],
                   grads.dBlocks[0].dFf1W.data[7])
  checkHrmGradient("H attn.qkvProj.weight", hrm.blocks[1].attn.qkvProj.weight.data[2],
                   grads.dBlocks[1].dAttnQkvW.data[2])
  checkHrmGradient("H ln2.gamma", hrm.blocks[1].ln2.gamma.data[1], grads.dBlocks[1].dLn2G.data[1])
  checkHrmGradient("normX.gamma", hrm.hrmNormX.gamma.data[2], grads.dHrmNormXG.data[2])
  checkHrmGradient("normL.beta", hrm.hrmNormL.beta.data[0], grads.dHrmNormLB.data[0])
  checkHrmGradient("normH.gamma", hrm.hrmNormH.gamma.data[3], grads.dHrmNormHG.data[3])
  echo "PASS: HRM one-step gradients match finite differences for one cycle and one L step"

  hrm.hrm = HrmConfig(lLayers: 1, cycles: 2, lSteps: 2, segments: 2)
  hrm.initHrmResiduals()
  # A contracting recurrence must not amplify GPU rounding between identical runs.
  block:
    let wide = block:
      var m = newNimformerModelSpectron(756, 64, 4, 4, 2)
      m.hrm = HrmConfig(lLayers: 2, cycles: 2, lSteps: 2, segments: 2)
      m.initHrmResiduals()
      m
    var wideIds = newSeq[seq[int]](8)
    for row in wideIds.mitems:
      row = newSeq[int](64)
      for t in row.mitems: t = rand(755)
    let first = wide.forwardBatch(wideIds, ctx)
    var worst = 0'f32
    for _ in 1 .. 3: worst = max(worst, maxDiff(first.data, wide.forwardBatch(wideIds, ctx).data))
    doAssert worst < 1e-3'f32, &"HRM forward is not reproducible: max diff {worst}"
    echo &"PASS: HRM 2x2 with 2 segments is reproducible after residual init (max diff {worst:.3g})"
  let path = createTempDir("spectron-hrm-", "") / "model.nimq"
  defer:
    removeFile(path)
    removeDir(parentDir(path))
  let before = hrm.forwardBatch(ids, ctx)
  hrm.saveCheckpoint(path, 7)
  let loaded = loadCheckpoint(path)
  doAssert loaded.model.hrm == hrm.hrm
  doAssert maxDiff(before.data, loaded.model.forwardBatch(ids, ctx).data) < 1e-5
  echo "PASS: HRM layout and outputs survive a checkpoint round trip"

  var adam = newModelAdamState(hrm)
  let initial = softmaxCrossEntropy(hrm.forwardBatch(ids, ctx), targets, ctx).loss
  for step in 1 .. 110:
    let loss = trainStep(hrm, adam, ids, targets, ctx, lr = 3e-3'f32)
    doAssert loss.classify notin {fcNan, fcInf, fcNegInf}
    if step mod 25 == 0: echo &"hrm step {step} loss={loss:.4f}"
  # Segments are averaged into a single update per training step.
  doAssert adam.outProjW.step == 110
  let final = softmaxCrossEntropy(hrm.forwardBatch(ids, ctx), targets, ctx).loss
  doAssert final < initial * 0.1'f32, &"HRM loss did not fall enough: {initial} -> {final}"
  echo &"PASS: HRM training with 2 cycles, 2 L steps and 2 segments: {initial:.4f} -> {final:.4f}"

  # Parallel block decoding must reproduce greedy decoding with at most K passes per block.
  let prompts = @[@[0, 1, 2, 3, 0, 1], @[2, 3, 0, 1, 2, 3]]
  for (label, decoder) in [("dense", model), ("hrm", hrm)]:
    var greedyStats: BlockDecodeStats
    var margins: seq[seq[float32]]
    let greedy = decoder.greedyBlocks(prompts, 8, 12, 4, ctx, greedyStats, margins)
    doAssert greedyStats.forwards == 8
    for policy in [bpJacobi, bpConfidence]:
      var stats: BlockDecodeStats
      let decoded = decoder.blockDecode(prompts, 8, 12, 4, policy, 0.6'f32, 3, ctx, stats)
      for r in 0 ..< prompts.len:
        let at = firstDifference(decoded[r], greedy[r])
        doAssert at < 0 or margins[r][at - prompts[r].len] < 1e-3'f32,
          &"{label} {policy}: row {r} differs from greedy at token {at}"
      doAssert stats.blocks == 2 and stats.rowPasses <= 2 * 4 * prompts.len
      doAssert stats.forwards <= greedyStats.forwards
      doAssert stats.accepted + stats.written == 8 * prompts.len
      echo &"PASS: {label} {policy} block decoding matches greedy with {stats.forwards} " &
           &"forwards instead of {greedyStats.forwards} (locks={stats.locks}, " &
           &"wrong_locks={stats.wrongLocks})"

closeBackend(ctx)
