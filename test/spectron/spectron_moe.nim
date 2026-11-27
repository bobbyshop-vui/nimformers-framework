## Included by test_nimformer. A shared Spectron trunk feeds one selected
## position-wise expert per token. Expert weights are independently owned.

const MoeBalanceWeight = 0.01

type
  SpectronExpertLayer = object
    norm: LayerNorm
    ff: FeedForward

  SpectronExpert = ref object
    layers: seq[SpectronExpertLayer]

  SpectronMixture = object
    norm: LayerNorm
    router: Linear
    experts: seq[SpectronExpert]
    width, depth: int

  ExpertActivation = object
    input, normalized, preActivation, activated: Tensor

  ExpertForward = object
    rows: seq[int]
    layers: seq[ExpertActivation]
    correction: seq[float32]

  MixtureForward = object
    input, normalized, output: Tensor
    probabilities: seq[float32]
    groups: seq[ExpertForward]

  ExpertGrad = tuple[upW, upB, downW, downB, normG, normB: Tensor]

  MixtureGrad = object
    dx, routerW, routerB, normG, normB: Tensor
    expertLayers: seq[seq[ExpertGrad]]
    routes: seq[int]
    balanceLoss: float64

  ExpertAdam = ref object
    layers: seq[tuple[upW, upB, downW, downB, normG, normB: ApfAdamState]]

  MixtureAdam = object
    routerW, routerB, normG, normB: ApfAdamState
    experts: seq[ExpertAdam]

proc newExpert(dim, width, depth: int, initializeWeights = true): SpectronExpert =
  new(result)
  result.layers = newSeq[SpectronExpertLayer](depth)
  for layer in result.layers.mitems:
    layer.norm = newLayerNorm(dim)
    layer.ff = FeedForward(fc1: newLinear(dim, width, initializeWeights),
                           fc2: newLinear(width, dim, initializeWeights))

proc newRouter(dim, count: int, initializeWeights = true): Linear =
  result = newLinear(dim, count, initializeWeights)
  for value in result.weight.data.mitems:
    value *= 0.02'f32

proc newMixture(dim, count, width, depth: int, initializeWeights = true): SpectronMixture =
  result.norm = newLayerNorm(dim)
  result.router = newRouter(dim, count, initializeWeights)
  result.width = width
  result.depth = depth
  for _ in 0 ..< count:
    result.experts.add(newExpert(dim, width, depth, initializeWeights))

proc expertParameterCount(dim, width, depth: int): int64 =
  int64(depth) * (2 * int64(dim) * width + width + 3 * int64(dim))

proc expertLayerKey(expert, layer: int): string =
  &"moe.expert{expert}.layer{layer}."

template visitExpertLayer(layer, action: untyped) =
  action("up.weight", layer.ff.fc1.weight)
  action("up.bias", layer.ff.fc1.bias)
  action("down.weight", layer.ff.fc2.weight)
  action("down.bias", layer.ff.fc2.bias)
  action("norm.gamma", layer.norm.gamma)
  action("norm.beta", layer.norm.beta)

proc newExpertAdam(expert: SpectronExpert): ExpertAdam =
  new(result)
  for layer in expert.layers:
    result.layers.add((
      newApfAdamState(layer.ff.fc1.weight.data.len),
      newApfAdamState(layer.ff.fc1.bias.data.len),
      newApfAdamState(layer.ff.fc2.weight.data.len),
      newApfAdamState(layer.ff.fc2.bias.data.len),
      newApfAdamState(layer.norm.gamma.data.len),
      newApfAdamState(layer.norm.beta.data.len)))

proc newMixtureAdam(mixture: SpectronMixture): MixtureAdam =
  result.routerW = newApfAdamState(mixture.router.weight.data.len)
  result.routerB = newApfAdamState(mixture.router.bias.data.len)
  result.normG = newApfAdamState(mixture.norm.gamma.data.len)
  result.normB = newApfAdamState(mixture.norm.beta.data.len)
  result.experts = newSeq[ExpertAdam](mixture.experts.len)

proc gatherExpertRows(x: Tensor, rows: openArray[int], dim: int, ctx: Backend): Tensor =
  # Padding keeps the expert weight-gradient reduction aligned to the tiled GEMM.
  let count = if ctx.usesTrainingTensorCores: (rows.len + 63) div 64 * 64 else: rows.len
  result = newTensor(@[count, dim])
  for i, row in rows:
    copyMem(addr result.data[i * dim], unsafeAddr x.data[row * dim], dim * sizeof(float32))

proc scaledExpertTensor(value: sink Tensor, scale: float32): Tensor =
  result = value
  for x in result.data.mitems:
    x *= scale

proc forwardMixture(m: SpectronMixture, x: Tensor, ctx: Backend, training: bool): MixtureForward =
  let (rows, dim) = flatten2D(x.shape)
  let nExperts = m.experts.len
  result.normalized = m.norm.forward(x, ctx)
  let scores = m.router.forward(result.normalized, ctx)
  result.output = x
  result.probabilities = newSeq[float32](scores.data.len)
  result.groups = newSeq[ExpertForward](nExperts)
  if training: result.input = x
  for row in 0 ..< rows:
    let offset = row * nExperts
    var winner = 0
    for i in 0 ..< nExperts:
      let score = scores.data[offset + i]
      if not finite32(score):
        raise newException(SpectronNumericalError, "Non-finite Spectron router logit")
      if score > scores.data[offset + winner]: winner = i
    let peak = float64(scores.data[offset + winner])
    var total = 0.0
    for i in 0 ..< nExperts:
      let expScore = exp(float64(scores.data[offset + i]) - peak)
      result.probabilities[offset + i] = float32(expScore)
      total += expScore
    for i in 0 ..< nExperts:
      result.probabilities[offset + i] *= float32(1.0 / total)
    result.groups[winner].rows.add(row)

  let residualScale = 1'f32 / sqrt(float32(m.depth))
  for i, expert in m.experts:
    template group: untyped = result.groups[i]
    if group.rows.len == 0: continue
    let input = gatherExpertRows(result.normalized, group.rows, dim, ctx)
    var hidden = input
    for layer in expert.layers:
      let normalized = layer.norm.forward(hidden, ctx)
      let preActivation = layer.ff.fc1.forward(normalized, ctx)
      let activated = Tensor(shape: preActivation.shape,
        data: beApflu(ctx, preActivation.data, 0.1'f32, 1'f32))
      let update = layer.ff.fc2.forward(activated, ctx)
      if training:
        group.layers.add(ExpertActivation(input: hidden, normalized: normalized,
          preActivation: preActivation, activated: activated))
      hidden = addT(ctx, hidden, scaledExpertTensor(update, residualScale))
    if training: group.correction = newSeq[float32](hidden.data.len)
    for j, row in group.rows:
      let weight = result.probabilities[row * nExperts + i]
      for c in 0 ..< dim:
        let idx = j * dim + c
        let correction = hidden.data[idx] - input.data[idx]
        result.output.data[row * dim + c] += weight * correction
        if training: group.correction[idx] = correction

proc backwardMixture(m: SpectronMixture, cache: MixtureForward, gradient: Tensor,
                     ctx: Backend): MixtureGrad =
  let (rows, dim) = flatten2D(gradient.shape)
  let nExperts = m.experts.len
  let residualScale = 1'f32 / sqrt(float32(m.depth))
  let balanceScale = MoeBalanceWeight * float64(nExperts) / float64(rows)
  var fractions = newSeq[float64](nExperts)
  result.routes = newSeq[int](nExperts)
  for i, group in cache.groups:
    result.routes[i] = group.rows.len
    fractions[i] = float64(group.rows.len) / float64(rows)
  result.expertLayers = newSeq[seq[ExpertGrad]](nExperts)
  var dNormalized = newTensor(cache.normalized.shape)
  var dScores = newTensor(@[rows, nExperts])

  for i, expert in m.experts:
    template group: untyped = cache.groups[i]
    if group.rows.len == 0: continue
    var dy = gatherExpertRows(gradient, group.rows, dim, ctx)
    for j, row in group.rows:
      let offset = row * nExperts
      let weight = cache.probabilities[offset + i]
      var gateDerivative, expectedFraction = 0.0
      for c in 0 ..< dim:
        let idx = j * dim + c
        gateDerivative += float64(dy.data[idx]) * float64(group.correction[idx])
        dy.data[idx] *= weight
        dNormalized.data[row * dim + c] = -dy.data[idx]
      for k in 0 ..< nExperts:
        expectedFraction += fractions[k] * float64(cache.probabilities[offset + k])
      result.balanceLoss += balanceScale * expectedFraction
      for k in 0 ..< nExperts:
        let selected = if k == i: 1.0 else: 0.0
        dScores.data[offset + k] = float32(float64(cache.probabilities[offset + k]) * (
          gateDerivative * (selected - float64(weight)) + balanceScale * (fractions[k] - expectedFraction)))

    result.expertLayers[i] = newSeq[ExpertGrad](expert.layers.len)
    for j in countdown(expert.layers.len - 1, 0):
      template layer: untyped = expert.layers[j]
      template saved: untyped = group.layers[j]
      let down = layer.ff.fc2.backward(saved.activated, scaledExpertTensor(dy, residualScale), ctx)
      let dPre = Tensor(shape: down.dX.shape,
        data: beApfluBackward(ctx, saved.preActivation.data, down.dX.data, 0.1'f32, 1'f32))
      let up = layer.ff.fc1.backward(saved.normalized, dPre, ctx)
      let norm = layer.norm.backward(saved.input, up.dX, ctx)
      result.expertLayers[i][j] = (up.dW, up.dB, down.dW, down.dB, norm.dGamma, norm.dBeta)
      dy = addT(ctx, dy, norm.dX)
    for j, row in group.rows:
      for c in 0 ..< dim:
        dNormalized.data[row * dim + c] += dy.data[j * dim + c]

  let router = m.router.backward(cache.normalized, dScores, ctx)
  let norm = m.norm.backward(cache.input, addT(ctx, dNormalized,
    Tensor(shape: dNormalized.shape, data: router.dX.data)), ctx)
  result.dx = addT(ctx, gradient, norm.dX)
  result.routerW = router.dW
  result.routerB = router.dB
  result.normG = norm.dGamma
  result.normB = norm.dBeta
