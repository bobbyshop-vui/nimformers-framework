## A policy over discrete operations, backed by Nimformers Linear layers.
import std/[json, math, os, random, tables, tempfiles]
import backend, nimformer, quant
import discrete_math

const
  ActionCount = MaxOrder + 1
  CheckpointVersion = 1
  ParameterNames = ["features.weight", "features.bias", "actions.weight", "actions.bias"]
  MaxHidden* = 1024

type
  MathPolicy* = object
    features, actions: Linear
    constrained*, research*: bool
  PolicyAdam* = object
    steps*: int
    first, second: array[ParameterNames.len, seq[float32]]
  Decision* = object
    frame*: MathFrame
    allowed*: ActionMask
    action*: Action
    weight*: float32
  Rollout* = object
    state*: SolverState
    decisions*: seq[Decision]
    verified*: bool

proc newPolicy*(hidden: int, initializeWeights = true): MathPolicy =
  result.features = newLinear(TokensPerState, hidden, initializeWeights)
  result.actions = newLinear(hidden, ActionCount, initializeWeights)

proc framesToTensor(frames: openArray[MathFrame]): Tensor =
  result = newTensor(@[frames.len, TokensPerState])
  for i in 0 ..< frames.len:
    copyMem(addr result.data[i * TokensPerState], unsafeAddr frames[i][0], sizeof(MathFrame))

proc activate(tensor: var Tensor) =
  for value in tensor.data.mitems:
    if value < 0: value *= 0.1'f32

proc scores(policy: MathPolicy, frames: openArray[MathFrame], ctx: Backend): Tensor =
  var features = policy.features.forward(framesToTensor(frames), ctx)
  features.activate()
  policy.actions.forward(features, ctx)

proc distribution(logits: openArray[float32], allowed: ActionMask): array[Action, float64] =
  var maximum = -Inf
  for action in allowed:
    let score = float64(logits[int(action)])
    if classify(score) in {fcNan, fcInf, fcNegInf}:
      raise newException(ValueError, "Non-finite policy logit")
    maximum = max(maximum, score)
  var total = 0.0
  for action in allowed:
    result[action] = exp(float64(logits[int(action)]) - maximum)
    total += result[action]
  for action in allowed: result[action] /= total

proc choose(logits: openArray[float32], allowed: ActionMask, rng: var Rand, greedy: bool): Action =
  if greedy:
    var best = -Inf
    for action in allowed:
      let score = float64(logits[int(action)])
      if classify(score) in {fcNan, fcInf, fcNegInf}:
        raise newException(ValueError, "Non-finite policy logit")
      if score > best:
        best = score
        result = action
    return
  let probs = distribution(logits, allowed)
  var draw = rng.rand(1.0)
  for action in allowed:
    result = action
    draw -= probs[action]
    if draw <= 0: break

proc runPolicies*(policy: MathPolicy, problems: openArray[MathProblem], limit: int,
                  ctx: Backend, rng: var Rand, greedy = false,
                  recordDecisions = true, recordMethod = false, guided = false,
                  rawActions = false): seq[Rollout] =
  ## Advance forced actions before each GPU batch. Each trajectory keeps its own action limit.
  result = newSeq[Rollout](problems.len)
  for i, problem in problems: result[i].state = newSolver(problem, recordMethod)
  for _ in 0 ..< limit:
    var frames: seq[MathFrame]
    var masks: seq[ActionMask]
    var indices: seq[int]
    for i, rollout in result.mpairs:
      while rollout.state.progress == active and rollout.state.steps < limit:
        let baseMask = rollout.state.legalActions()
        let legal =
          if policy.constrained and not rawActions: rollout.state.constrainedActions(baseMask)
          elif guided: rollout.state.guidedActions(baseMask)
          else: baseMask
        if legal == {}:
          rollout.state.progress = failed
          break
        if card(legal) == 1:
          for action in legal:
            if recordDecisions:
              rollout.decisions.add(Decision(allowed: legal, action: action))
            rollout.state.advance(action)
        else:
          frames.add(rollout.state.policyFrame(legal, policy.constrained))
          masks.add(legal)
          indices.add(i)
          break
    if frames.len == 0: break
    let logits = policy.scores(frames, ctx)
    for i, index in indices:
      let base = i * ActionCount
      let action = choose(logits.data.toOpenArray(base, base + ActionCount - 1), masks[i], rng, greedy)
      if recordDecisions:
        result[index].decisions.add(Decision(frame: frames[i], allowed: masks[i], action: action))
      result[index].state.advance(action)
  for rollout in result.mitems:
    rollout.verified = rollout.state.certified()

proc runPolicy*(policy: MathPolicy, problem: MathProblem, count, limit: int,
                ctx: Backend, rng: var Rand, greedy = false,
                recordDecisions = true, recordMethod = false, guided = false): seq[Rollout] =
  var problems = newSeqOfCap[MathProblem](count)
  for _ in 0 ..< count: problems.add(problem)
  policy.runPolicies(problems, limit, ctx, rng, greedy, recordDecisions, recordMethod, guided)

proc hiddenWidth*(policy: MathPolicy): int = policy.features.outF

proc learnableStates*(decisions: openArray[Decision]): int =
  for i in 0 ..< decisions.len:
    if decisions[i].weight != 0 and card(decisions[i].allowed) > 1: inc result

proc adamUpdate(param: var Tensor, gradient: Tensor, opt: var PolicyAdam, slot: int, lr: float32) =
  if opt.first[slot].len == 0:
    opt.first[slot].setLen(param.data.len)
    opt.second[slot].setLen(param.data.len)
  let firstCorrection = 1'f32 - pow(0.9'f32, float32(opt.steps))
  let secondCorrection = 1'f32 - pow(0.999'f32, float32(opt.steps))
  for i in 0 ..< param.data.len:
    let g = if gradient.data.len == 0: 0'f32 else: gradient.data[i]
    opt.first[slot][i] = 0.9'f32 * opt.first[slot][i] + 0.1'f32 * g
    opt.second[slot][i] = 0.999'f32 * opt.second[slot][i] + 0.001'f32 * g * g
    let m = opt.first[slot][i] / firstCorrection
    let v = opt.second[slot][i] / secondCorrection
    param.data[i] -= lr * m / (sqrt(v) + 1e-8'f32)

proc update*(policy: var MathPolicy, optimizer: var PolicyAdam, decisions: openArray[Decision],
             ctx: Backend, lr: float32, normalizer: int): float64 =
  ## Masked cross entropy. A signed response weight gives the exact gradient
  ## of -weight * log pi(action | state) under the sampling distribution.
  var indices = newSeqOfCap[int](decisions.len)
  for i in 0 ..< decisions.len:
    if decisions[i].weight != 0 and card(decisions[i].allowed) > 1: indices.add(i)
  var featureWeight, featureBias, headWeight, headBias: Tensor
  if indices.len > 0:
    var batch = newTensor(@[indices.len, TokensPerState])
    for row, i in indices:
      copyMem(addr batch.data[row * TokensPerState], unsafeAddr decisions[i].frame[0], sizeof(MathFrame))
    var hidden = policy.features.forward(batch, ctx)
    hidden.activate()
    let logits = policy.actions.forward(hidden, ctx)
    var dLogits = newTensor(logits.shape)
    for row, i in indices:
      template decision: untyped = decisions[i]
      let base = row * ActionCount
      let probs = distribution(logits.data.toOpenArray(base, base + ActionCount - 1), decision.allowed)
      let scale = float64(decision.weight) / float64(normalizer)
      result -= scale * ln(max(probs[decision.action], 1e-300))
      for action in decision.allowed:
        let target = float64(ord(action == decision.action))
        dLogits.data[base + int(action)] = float32(scale * (probs[action] - target))
    let headGrad = policy.actions.backward(hidden, dLogits, ctx)
    var dHidden = headGrad.dX
    for i, value in hidden.data:
      if value < 0: dHidden.data[i] *= 0.1'f32
    let featureGrad = policy.features.backward(batch, dHidden, ctx)
    (featureWeight, featureBias) = (featureGrad.dW, featureGrad.dB)
    (headWeight, headBias) = (headGrad.dW, headGrad.dB)
  # An all-forced batch still advances Adam and decays its stored moments.
  inc optimizer.steps
  adamUpdate(policy.features.weight, featureWeight, optimizer, 0, lr)
  adamUpdate(policy.features.bias, featureBias, optimizer, 1, lr)
  adamUpdate(policy.actions.weight, headWeight, optimizer, 2, lr)
  adamUpdate(policy.actions.bias, headBias, optimizer, 3, lr)

proc encodeMetadata(node: JsonNode): QuantTensor =
  let metadata = $node
  var data = newSeq[float32](metadata.len)
  for i, ch in metadata: data[i] = float32(ord(ch))
  quantizeRaw(data, @[data.len])

proc decodeMetadata(tensor: QuantTensor): JsonNode =
  if tensor.kind != qkFp32Raw or tensor.bytes.len > 4 * 1024 * 1024:
    raise newException(ValueError, "Invalid checkpoint metadata")
  let data = dequantizeRaw(tensor)
  var payload = newString(data.len)
  for i, value in data:
    if value notin 0'f32 .. 255'f32 or value != floor(value):
      raise newException(ValueError, "Invalid byte in checkpoint metadata")
    payload[i] = char(int(value))
  parseJson(payload)

proc save*(policy: MathPolicy, optimizer: PolicyAdam, metadata: JsonNode, path: string) =
  let params = [policy.features.weight, policy.features.bias,
                policy.actions.weight, policy.actions.bias]
  var sd: seq[(string, QuantTensor)]
  for i, name in ParameterNames:
    sd.add((name, quantizeRaw(params[i].data, params[i].shape)))
    sd.add((name & ".m", quantizeRaw(optimizer.first[i], params[i].shape)))
    sd.add((name & ".v", quantizeRaw(optimizer.second[i], params[i].shape)))
  metadata["optimizer_steps"] = %optimizer.steps
  sd.add(("__training__", encodeMetadata(metadata)))
  let fullPath = absolutePath(path)
  createDir(fullPath.parentDir)
  let (file, tempPath) = createTempFile(".math-", ".nimq", fullPath.parentDir)
  file.close()
  try:
    saveQuantStateDict(tempPath, [TokensPerState, policy.features.outF, ActionCount, CheckpointVersion, ord(policy.constrained) + 2 * ord(policy.research)], sd)
    moveFile(tempPath, fullPath)
  finally:
    if fileExists(tempPath): removeFile(tempPath)

proc load*(path: string, restoreOptimizer: bool): tuple[policy: MathPolicy,
    optimizer: PolicyAdam, metadata: JsonNode] =
  let (arch, tensors, _) = loadQuantStateDict(path, mapped = true)
  if arch[0] != TokensPerState or arch[1] notin 1 .. MaxHidden or
      arch[2] != ActionCount or arch[3] != CheckpointVersion or arch[4] notin 0 .. 3:
    raise newException(ValueError, "Not a compatible Math SpectronNexus checkpoint")
  var byName: Table[string, QuantTensor]
  for (name, tensor) in tensors:
    byName[name] = tensor
  template require(name: string): QuantTensor =
    if not byName.hasKey(name):
      raise newException(ValueError, "Missing checkpoint tensor " & name)
    byName[name]
  proc readParam(tensor: QuantTensor, shape: seq[int]): seq[float32] =
    var count = 1
    for dim in shape: count *= dim
    if tensor.kind != qkFp32Raw or tensor.shape != shape or tensor.bytes.len != count * 4:
      raise newException(ValueError, "Invalid checkpoint parameter shape or dtype")
    dequantizeRaw(tensor)
  result.policy = newPolicy(arch[1], initializeWeights = false)
  result.policy.constrained = (arch[4] and 1) != 0
  result.policy.research = (arch[4] and 2) != 0
  result.policy.features.weight.data = readParam(require(ParameterNames[0]), @[arch[1], arch[0]])
  result.policy.features.bias.data = readParam(require(ParameterNames[1]), @[arch[1]])
  result.policy.actions.weight.data = readParam(require(ParameterNames[2]), @[arch[2], arch[1]])
  result.policy.actions.bias.data = readParam(require(ParameterNames[3]), @[arch[2]])
  result.metadata = decodeMetadata(require("__training__"))
  if restoreOptimizer:
    result.optimizer.steps = result.metadata["optimizer_steps"].getInt()
    if result.optimizer.steps <= 0: raise newException(ValueError, "Invalid Adam step")
    let shapes = [@[arch[1], arch[0]], @[arch[1]], @[arch[2], arch[1]], @[arch[2]]]
    for i, name in ParameterNames:
      result.optimizer.first[i] = readParam(require(name & ".m"), shapes[i])
      result.optimizer.second[i] = readParam(require(name & ".v"), shapes[i])
