## A light MLP that orders the sequence explorer's tools. The exact engine
## decides what holds. The policy only chooses what to test first, and the
## exhaustive search still runs after its steps.
import std/[math, os, random, sequtils, strformat, tables, tempfiles]
import backend, nimformer, quant
import big_integer, sequence_explorer

const
  PolicyVersion = 1
  ParameterNames = ["features.weight", "features.bias", "actions.weight", "actions.bias"]
  TeacherBudget* = 400
  EvalBudget* = 300   ## The MCP default for max_tests.
  MaxExplorerHidden* = 1024

type
  ExplorerPolicy* = object
    features, actions: Linear
  ExplorerAdam* = object
    steps*: int
    first, second: array[ParameterNames.len, seq[float32]]
  ExplorerSample* = object
    frame: ExplorerFrame
    allowed: set[ExplorerAction]
    action: ExplorerAction
  SequenceExample* = object
    terms*: seq[BigInt]
    offset*: int
    family*: string
    hint*: seq[Transform]   ## Planted transform path. The teacher uses it to reorder, never as a feature.
  EvalCase* = object
    example*: SequenceExample
    defaultFound*: bool
    defaultTests*: int
  ExplorerEvaluation* = object
    examples*, foundDefault*, foundGuided*: int
    testsDefault*, testsGuided*: int
    decisions*, correct*: int
    families*: seq[string]
    perFamily*: Table[string, tuple[total, plain, guided: int]]

proc newExplorerPolicy*(hidden: int, initializeWeights = true): ExplorerPolicy =
  result.features = newLinear(ExplorerFeatures, hidden, initializeWeights)
  result.actions = newLinear(hidden, ExplorerActionCount, initializeWeights)

proc hiddenWidth*(policy: ExplorerPolicy): int = policy.features.outF

proc framesTensor(frames: openArray[ExplorerFrame]): Tensor =
  result = newTensor(@[frames.len, ExplorerFeatures])
  for i in 0 ..< frames.len:
    copyMem(addr result.data[i * ExplorerFeatures], unsafeAddr frames[i][0], sizeof(ExplorerFrame))

proc leaky(tensor: var Tensor) =
  for value in tensor.data.mitems:
    if value < 0: value *= 0.1'f32

proc scores(policy: ExplorerPolicy, frames: openArray[ExplorerFrame], ctx: Backend): Tensor =
  var hidden = policy.features.forward(framesTensor(frames), ctx)
  hidden.leaky()
  policy.actions.forward(hidden, ctx)

proc best(logits: openArray[float32], allowed: set[ExplorerAction]): ExplorerAction =
  var top = -Inf
  for action in allowed:
    if float64(logits[int(action)]) > top:
      top = float64(logits[int(action)])
      result = action

proc guide*(policy: ExplorerPolicy, ctx: Backend): ExplorerGuide =
  result = proc(ex: var Explorer, legal: set[ExplorerAction]): ExplorerAction =
    let logits = policy.scores([ex.explorerFrame(legal)], ctx)
    best(logits.data.toOpenArray(0, ExplorerActionCount - 1), legal)

# Synthetic families with known structure. Terms are exact.

proc binomialBig(n, k: int): BigInt =
  if k < 0 or k > n: return big(0)
  result = big(1)
  for i in 0 ..< min(k, n - k): result = exactDiv(result * big(n - i), i + 1)

proc factorizeSmall(n: int): seq[tuple[prime, exponent: int]] =
  var rest = n
  var d = 2
  while d * d <= rest:
    if rest mod d == 0:
      var e = 0
      while rest mod d == 0:
        rest = rest div d
        inc e
      result.add((d, e))
    inc d
  if rest > 1: result.add((rest, 1))

proc eulerTransformBig(b: seq[BigInt], length: int): seq[BigInt] =
  ## a(0) = 1 and prod (1 - x^k)^(-b(k)) = sum a(n) x^n, with b(k) = b[k] for k >= 1.
  var c = newSeq[BigInt](length)
  for k in 0 ..< length: c[k] = big(0)
  for d in 1 ..< length:
    var k = d
    while k < length:
      c[k] = c[k] + big(d) * b[d]
      k += d
  result = newSeq[BigInt](length)
  result[0] = big(1)
  for n in 1 ..< length:
    var total = big(0)
    for k in 1 .. n: total = total + c[k] * result[n - k]
    result[n] = exactDiv(total, n)

proc smallRecurrence(rng: var Rand, length: int): seq[BigInt] =
  ## Positive terms of a short linear recurrence, indexed from 0.
  result = newSeq[BigInt](length)
  case rng.rand(0 .. 3)
  of 0:
    let base = rng.rand(2 .. 3)
    result[0] = big(1)
    for i in 1 ..< length: result[i] = result[i - 1] * big(base)
  of 1:
    result[0] = big(rng.rand(0 .. 1))
    result[1] = big(1)
    for i in 2 ..< length: result[i] = result[i - 1] + result[i - 2]
  of 2:
    let (u, v) = (rng.rand(1 .. 2), rng.rand(1 .. 2))
    result[0] = big(1)
    result[1] = big(rng.rand(1 .. 3))
    for i in 2 ..< length: result[i] = big(u) * result[i - 1] + big(v) * result[i - 2]
  else:
    let base = rng.rand(2 .. 3)
    var power = big(1)
    for i in 0 ..< length:
      result[i] = big(i + 1) * power
      power = power * big(base)

proc divisorPolynomial(rng: var Rand, length: int): seq[BigInt] =
  ## b(k) = sum over d | k of P(d) for a small polynomial P, so Mobius inversion gives P.
  let (c0, c1, c2) = (rng.rand(0 .. 2), rng.rand(0 .. 2), rng.rand(0 .. 1))
  let constant = if c0 + c1 + c2 == 0: 1 else: c0
  result = newSeq[BigInt](length)
  result[0] = big(0)
  for k in 1 ..< length:
    var total = 0
    for d in 1 .. k:
      if k mod d == 0: total += constant + c1 * d + c2 * d * d
    result[k] = big(total)

proc polynomialWeights(rng: var Rand, length: int): seq[BigInt] =
  let (c0, c1) = (rng.rand(0 .. 2), rng.rand(0 .. 2))
  let constant = if c0 + c1 == 0: 1 else: c0
  result = newSeq[BigInt](length)
  result[0] = big(0)
  for k in 1 ..< length: result[k] = big(constant + c1 * k)

const
  BasicFamilies = 10
  HardFamilies* = ["apery_like", "holonomic_order_three", "euler_of_recurrence", "euler_of_divisor_sum",
                   "binomial_of_euler", "factorial_times_euler", "divisor_sum_of_recurrence",
                   "algebraic_cubic", "factorial_times_holonomic"]

proc generateHard*(rng: var Rand, length: int, family: int): SequenceExample =
  ## Families whose structure needs a high-order fit or two chained transforms.
  result.family = HardFamilies[family]
  var terms = newSeq[BigInt](length)
  case family
  of 0:
    const shapes = [(1, 1), (2, 1), (3, 0), (4, 0), (2, 2), (5, 0), (3, 1)]
    let (p, q) = shapes[rng.rand(shapes.high)]
    for n in 0 ..< length:
      var total = big(0)
      for k in 0 .. n:
        total = total + power(binomialBig(n, k), p) * power(binomialBig(n + k, k), q)
      terms[n] = total
  of 1:
    let (a, b, c) = (rng.rand(1 .. 2), rng.rand(-2 .. 2), rng.rand(-2 .. 2))
    let (d, e, f) = (rng.rand(0 .. 2), rng.rand(1 .. 3), rng.rand(-1 .. 1))
    for i in 0 .. 2: terms[i] = big(rng.rand(1 .. 3))
    for n in 3 ..< length:
      terms[n] = big(a * n + b) * terms[n - 1] + big(c * n * n + d) * terms[n - 2] + big(e * n + f) * terms[n - 3]
  of 2:
    let r = smallRecurrence(rng, length)
    var b = newSeq[BigInt](length)
    b[0] = big(0)
    for k in 1 ..< length: b[k] = r[k - 1]
    terms = eulerTransformBig(b, length)
    result.hint = @[tInverseEuler]
  of 3:
    terms = eulerTransformBig(divisorPolynomial(rng, length), length)
    result.hint = @[tInverseEuler, tMobius]
  of 4:
    let p = eulerTransformBig(polynomialWeights(rng, length), length)
    for n in 0 ..< length:
      var total = big(0)
      for k in 0 .. n: total = total + binomialBig(n, k) * p[k]
      terms[n] = total
    result.hint = @[tInverseBinomial, tInverseEuler]
  of 5:
    let p = eulerTransformBig(polynomialWeights(rng, length), length)
    var factorial = big(1)
    for n in 0 ..< length:
      if n > 0: factorial = factorial * big(n)
      terms[n] = factorial * p[n]
    result.hint = @[tDivideFactorial, tInverseEuler]
  of 6:
    result.offset = 1
    let f = smallRecurrence(rng, length + 1)
    for i in 0 ..< length:
      let n = i + 1
      var total = big(0)
      for d in 1 .. n:
        if n mod d == 0: total = total + f[d]
      terms[i] = total
    result.hint = @[tMobius]
  of 7:
    let (u, v, w) = (rng.rand(0 .. 2), rng.rand(0 .. 2), rng.rand(1 .. 2))
    var squares = newSeq[BigInt](length)
    var cubes = newSeq[BigInt](length)
    terms[0] = big(1)
    for n in 0 ..< length:
      if n > 0:
        terms[n] = big(u) * terms[n - 1] + big(v) * squares[n - 1] + big(w) * cubes[n - 1]
      # squares and cubes are the coefficients of F^2 and F^3 through index n.
      var s = big(0)
      for k in 0 .. n: s = s + terms[k] * terms[n - k]
      squares[n] = s
      var c = big(0)
      for k in 0 .. n: c = c + terms[k] * squares[n - k]
      cubes[n] = c
  else:
    let (a, b, c) = (rng.rand(1 .. 3), rng.rand(-2 .. 2), rng.rand(1 .. 2))
    var inner = @[big(1), big(rng.rand(1 .. 3))]
    for n in 2 ..< length: inner.add(big(a * n + b) * inner[n - 1] + big(c) * inner[n - 2])
    var factorial = big(1)
    for n in 0 ..< length:
      if n > 0: factorial = factorial * big(n)
      terms[n] = factorial * inner[n]
    result.hint = @[tDivideFactorial]
  result.terms = terms

proc generateExample*(rng: var Rand, length: int, hardFraction = 0.0): SequenceExample =
  if hardFraction > 0 and rng.rand(1.0) < hardFraction:
    return generateHard(rng, length, rng.rand(HardFamilies.high))
  var terms = newSeq[BigInt](length)
  var offset = 0
  let family = rng.rand(0 .. BasicFamilies)
  case family
  of 0:
    result.family = "polynomial"
    offset = rng.rand(0 .. 1)
    let degree = rng.rand(1 .. 5)
    var coefficients: seq[int]
    for _ in 0 .. degree: coefficients.add(rng.rand(-5 .. 5))
    if coefficients[^1] == 0: coefficients[^1] = 1
    for i in 0 ..< length:
      var value = big(0)
      for k, c in coefficients: value = value + big(c) * binomialBig(offset + i, k)
      terms[i] = value
  of 1:
    result.family = "linear_recurrence"
    let order = rng.rand(1 .. 4)
    var c: seq[int]
    for _ in 0 ..< order: c.add(rng.rand(-3 .. 3))
    if c[^1] == 0: c[^1] = 1 + rng.rand(1)
    for i in 0 ..< order: terms[i] = big(rng.rand(-4 .. 4))
    if terms[order - 1].isZero: terms[order - 1] = big(1)
    for i in order ..< length:
      var value = big(0)
      for j in 0 ..< order: value = value + big(c[j]) * terms[i - 1 - j]
      terms[i] = value
  of 2:
    result.family = "hypergeometric"
    let (alpha, beta) = (rng.rand(1 .. 3), rng.rand(0 .. 3))
    terms[0] = big(rng.rand(1 .. 3))
    for n in 1 ..< length: terms[n] = terms[n - 1] * big(alpha * n + beta)
  of 3:
    result.family = "holonomic_order_two"
    let (a, b, c, d) = (rng.rand(-3 .. 3), rng.rand(-3 .. 3), rng.rand(1 .. 3), rng.rand(-3 .. 3))
    terms[0] = big(rng.rand(1 .. 3))
    terms[1] = big(rng.rand(1 .. 3))
    for n in 2 ..< length: terms[n] = big(a * n + b) * terms[n - 1] + big(c * n + d) * terms[n - 2]
  of 4:
    result.family = "algebraic"
    let (u, v) = (rng.rand(0 .. 3), rng.rand(1 .. 3))
    terms[0] = big(1)
    for n in 1 ..< length:
      var convolution = big(0)
      for k in 0 ..< n: convolution = convolution + terms[k] * terms[n - 1 - k]
      terms[n] = big(u) * terms[n - 1] + big(v) * convolution
  of 5:
    result.family = "euler_transform"
    var b = newSeq[int](length)
    if rng.rand(1) == 0:
      let (c0, c1) = (rng.rand(0 .. 2), rng.rand(0 .. 1))
      for k in 1 ..< length: b[k] = c0 + c1 * k
    else:
      let period = rng.rand(1 .. 4)
      var pattern: seq[int]
      for _ in 0 ..< period: pattern.add(rng.rand(0 .. 2))
      if pattern.allIt(it == 0): pattern[0] = 1
      for k in 1 ..< length: b[k] = pattern[k mod period]
    var c = newSeq[BigInt](length)
    for k in 1 ..< length: c[k] = big(0)
    for d in 1 ..< length:
      var k = d
      while k < length:
        c[k] = c[k] + big(d * b[d])
        k += d
    terms[0] = big(1)
    for n in 1 ..< length:
      var total = big(0)
      for k in 1 .. n: total = total + c[k] * terms[n - k]
      terms[n] = exactDiv(total, n)
  of 6:
    result.family = "divisor_sum"
    offset = 1
    let (c0, c1, c2) = (rng.rand(-2 .. 3), rng.rand(0 .. 2), rng.rand(0 .. 1))
    for i in 0 ..< length:
      let n = i + 1
      var total = big(0)
      for d in 1 .. n:
        if n mod d == 0: total = total + big(c0 + c1 * d + c2 * d * d)
      terms[i] = total
  of 7:
    result.family = "multiplicative"
    offset = 1
    var values = initTable[(int, int), int]()
    for i in 0 ..< length:
      let n = i + 1
      var value = big(1)
      for (p, e) in factorizeSmall(n):
        if (p, e) notin values: values[(p, e)] = rng.rand(-3 .. 6)
        value = value * big(values[(p, e)])
      terms[i] = value
  of 8:
    result.family = "polynomial_times_exponential"
    let base = rng.rand(2 .. 4) * (if rng.rand(1) == 0: 1 else: -1)
    let (c0, c1, c2) = (rng.rand(-3 .. 3), rng.rand(-3 .. 3), rng.rand(1 .. 2))
    var power = big(1)
    for i in 0 ..< length:
      terms[i] = big(c0 + c1 * i + c2 * i * i) * power
      power = power * big(base)
  of 9:
    result.family = "factorial_times_linear"
    let (p, q) = (rng.rand(1 .. 3), rng.rand(-2 .. 2))
    var linear = @[big(1), big(rng.rand(1 .. 3))]
    for i in 2 ..< length: linear.add(big(p) * linear[i - 1] + big(q) * linear[i - 2])
    var factorial = big(1)
    for i in 0 ..< length:
      if i > 0: factorial = factorial * big(i)
      terms[i] = factorial * linear[i]
  else:
    result.family = "partial_sums"
    let (alpha, beta) = (rng.rand(1 .. 2), rng.rand(0 .. 2))
    var term = big(1)
    var total = big(0)
    for n in 0 ..< length:
      if n > 0: term = term * big(alpha * n + beta)
      total = total + term
      terms[n] = total
  result.terms = terms
  result.offset = offset

proc highOrderRecurrence(rng: var Rand, length: int): seq[BigInt] =
  # Coefficients of 1 / (1 - c[0]*x - ... - c[r-1]*x^r) give order r.
  var coefficients = newSeq[BigInt](rng.rand(6 .. 10))
  for coefficient in coefficients.mitems: coefficient = big(rng.rand(1 .. 5))
  result = newSeq[BigInt](length)
  result[0] = big(1)
  for n in 1 ..< length:
    for lag in 1 .. min(n, coefficients.len):
      result[n] = result[n] + coefficients[lag - 1] * result[n - lag]

proc multiplyByFactorials(terms: var seq[BigInt], offset: int) =
  var factorial = big(1)
  for n in 2 .. offset: factorial = factorial * big(n)
  for i in 0 ..< terms.len:
    if i > 0: factorial = factorial * big(offset + i)
    terms[i] = terms[i] * factorial

const ProgramHardFamilies = ["holonomic_high_order_degree", "binomial_of_factorial_times_recurrence",
                             "factorial_times_divisor_sum_of_recurrence"]

proc generateProgramExample*(rng: var Rand, length: int, hard: bool): SequenceExample =
  ## Additional families for the Spectron program trainer.
  if not hard: return generateExample(rng, length)
  let family = rng.rand(HardFamilies.len + ProgramHardFamilies.high)
  if family < HardFamilies.len: return generateHard(rng, length, family)

  let programFamily = family - HardFamilies.len
  result.family = ProgramHardFamilies[programFamily]
  result.terms = newSeq[BigInt](length)
  case programFamily
  of 0:
    let order = rng.rand(4 .. 5)
    let degree = rng.rand(4 .. 6)
    var polynomials = newSeq[seq[BigInt]](order)
    for polynomial in polynomials.mitems:
      for _ in 0 .. degree: polynomial.add(big(rng.rand(-3 .. 3)))
    if polynomials[^1][^1].isZero: polynomials[^1][^1] = big(1)
    for i in 0 ..< order: result.terms[i] = big(rng.rand(1 .. 9))
    for n in order ..< length:
      let index = big(n)
      for lag, polynomial in polynomials:
        var weight = big(0)
        for j in countdown(degree, 0): weight = weight * index + polynomial[j]
        result.terms[n] = result.terms[n] + weight * result.terms[n - lag - 1]
  of 1:
    var weighted = highOrderRecurrence(rng, length)
    weighted.multiplyByFactorials(0)
    for n in 0 ..< length:
      var coefficient = big(1)
      var total = weighted[0]
      for k in 1 .. n:
        coefficient = exactDiv(coefficient * big(n - k + 1), k)
        total = total + coefficient * weighted[k]
      result.terms[n] = total
    result.hint = @[tInverseBinomial, tDivideFactorial]
  else:
    let inner = highOrderRecurrence(rng, length + 1)
    result.offset = 1
    for divisor in 1 .. length:
      var multiple = divisor
      while multiple <= length:
        result.terms[multiple - 1] = result.terms[multiple - 1] + inner[divisor]
        multiple += divisor
    result.terms.multiplyByFactorials(result.offset)
    result.hint = @[tDivideFactorial, tMobius]

proc teacherSamples*(example: SequenceExample): seq[ExplorerSample] =
  let root = rootTrack(example.terms, example.offset)
  let route = teacherRoute(root, TeacherBudget, example.hint)
  if route.len == 0: return
  var ex = newExplorer(root)
  for action in route:
    let legal = ex.legalActions()
    if action notin legal: return @[]
    if card(legal) > 1: result.add(ExplorerSample(frame: ex.explorerFrame(legal), allowed: legal, action: action))
    ex.act(action)

# Training.

proc adamUpdate(param: var Tensor, gradient: Tensor, opt: var ExplorerAdam, slot: int, lr: float32) =
  if opt.first[slot].len == 0:
    opt.first[slot].setLen(param.data.len)
    opt.second[slot].setLen(param.data.len)
  let firstCorrection = 1'f32 - pow(0.9'f32, float32(opt.steps))
  let secondCorrection = 1'f32 - pow(0.999'f32, float32(opt.steps))
  for i in 0 ..< param.data.len:
    let g = if gradient.data.len == 0: 0'f32 else: gradient.data[i]
    opt.first[slot][i] = 0.9'f32 * opt.first[slot][i] + 0.1'f32 * g
    opt.second[slot][i] = 0.999'f32 * opt.second[slot][i] + 0.001'f32 * g * g
    param.data[i] -= lr * (opt.first[slot][i] / firstCorrection) /
      (sqrt(opt.second[slot][i] / secondCorrection) + 1e-8'f32)

proc update*(policy: var ExplorerPolicy, adam: var ExplorerAdam, batch: openArray[ExplorerSample],
             ctx: Backend, lr: float32): float64 =
  ## Masked cross entropy toward the teacher's next tool.
  var frames = newSeq[ExplorerFrame](batch.len)
  for i, sample in batch: frames[i] = sample.frame
  let input = framesTensor(frames)
  var hidden = policy.features.forward(input, ctx)
  hidden.leaky()
  let logits = policy.actions.forward(hidden, ctx)
  var dLogits = newTensor(logits.shape)
  for row, sample in batch:
    let base = row * ExplorerActionCount
    var top = -Inf
    for action in sample.allowed: top = max(top, float64(logits.data[base + int(action)]))
    var total = 0.0
    for action in sample.allowed: total += exp(float64(logits.data[base + int(action)]) - top)
    for action in sample.allowed:
      let probability = exp(float64(logits.data[base + int(action)]) - top) / total
      let target = float64(ord(action == sample.action))
      dLogits.data[base + int(action)] = float32((probability - target) / float64(batch.len))
      if action == sample.action: result -= ln(max(probability, 1e-300)) / float64(batch.len)
  let headGrad = policy.actions.backward(hidden, dLogits, ctx)
  var dHidden = headGrad.dX
  for i, value in hidden.data:
    if value < 0: dHidden.data[i] *= 0.1'f32
  let featureGrad = policy.features.backward(input, dHidden, ctx)
  inc adam.steps
  adamUpdate(policy.features.weight, featureGrad.dW, adam, 0, lr)
  adamUpdate(policy.features.bias, featureGrad.dB, adam, 1, lr)
  adamUpdate(policy.actions.weight, headGrad.dW, adam, 2, lr)
  adamUpdate(policy.actions.bias, headGrad.dB, adam, 3, lr)

proc evalCases*(examples: seq[SequenceExample]): seq[EvalCase] =
  ## Run the default order once per example. It does not change during training.
  for example in examples:
    var plain = newExplorer(rootTrack(example.terms, example.offset))
    plain.runDefault(1, EvalBudget)
    result.add(EvalCase(example: example, defaultFound: plain.strongCount > 0, defaultTests: plain.tests))

proc evaluate*(policy: ExplorerPolicy, cases: seq[EvalCase], ctx: Backend): ExplorerEvaluation =
  ## Tests until the first strong direction within the MCP budget, default order versus
  ## policy first. A miss counts the whole budget.
  let steer = policy.guide(ctx)
  for item in cases:
    let example = item.example
    let root = rootTrack(example.terms, example.offset)
    inc result.examples
    if example.family notin result.perFamily:
      result.families.add(example.family)
      result.perFamily[example.family] = (0, 0, 0)
    inc result.perFamily[example.family].total
    if item.defaultFound:
      inc result.foundDefault
      inc result.perFamily[example.family].plain
    result.testsDefault += item.defaultTests
    var guided = newExplorer(root)
    guided.explore(1, EvalBudget, steer)
    if guided.strongCount > 0:
      inc result.foundGuided
      inc result.perFamily[example.family].guided
    result.testsGuided += guided.tests
    for sample in teacherSamples(example):
      let logits = policy.scores([sample.frame], ctx)
      inc result.decisions
      if best(logits.data.toOpenArray(0, ExplorerActionCount - 1), sample.allowed) == sample.action:
        inc result.correct

proc save*(policy: ExplorerPolicy, steps: int, path: string) =
  let params = [policy.features.weight, policy.features.bias, policy.actions.weight, policy.actions.bias]
  var sd: seq[(string, QuantTensor)]
  for i, name in ParameterNames: sd.add((name, quantizeRaw(params[i].data, params[i].shape)))
  let fullPath = absolutePath(path)
  createDir(fullPath.parentDir)
  let (file, tempPath) = createTempFile(".explorer-", ".nimq", fullPath.parentDir)
  file.close()
  try:
    saveQuantStateDict(tempPath, [ExplorerFeatures, policy.features.outF, ExplorerActionCount, PolicyVersion, steps], sd)
    moveFile(tempPath, fullPath)
  finally:
    if fileExists(tempPath): removeFile(tempPath)

proc loadExplorerPolicy*(path: string): tuple[policy: ExplorerPolicy, steps: int] =
  let (arch, tensors, _) = loadQuantStateDict(path, mapped = true)
  if arch.len < 5 or arch[0] != ExplorerFeatures or arch[1] notin 1 .. MaxExplorerHidden or
      arch[2] != ExplorerActionCount or arch[3] != PolicyVersion:
    raise newException(ValueError, "Not a compatible sequence explorer checkpoint")
  var byName: Table[string, QuantTensor]
  for (name, tensor) in tensors: byName[name] = tensor
  proc read(name: string, shape: seq[int]): seq[float32] =
    if not byName.hasKey(name): raise newException(ValueError, "Missing checkpoint tensor " & name)
    let tensor = byName[name]
    if tensor.kind != qkFp32Raw or tensor.shape != shape:
      raise newException(ValueError, "Invalid checkpoint tensor " & name)
    dequantizeRaw(tensor)
  result.policy = newExplorerPolicy(arch[1], initializeWeights = false)
  result.policy.features.weight.data = read(ParameterNames[0], @[arch[1], arch[0]])
  result.policy.features.bias.data = read(ParameterNames[1], @[arch[1]])
  result.policy.actions.weight.data = read(ParameterNames[2], @[arch[2], arch[1]])
  result.policy.actions.bias.data = read(ParameterNames[3], @[arch[2]])
  result.steps = arch[4]

proc report*(evaluation: ExplorerEvaluation): string =
  let examples = max(evaluation.examples, 1)
  &"examples={evaluation.examples} found_default={evaluation.foundDefault} found_guided={evaluation.foundGuided} " &
    &"mean_tests_default={float64(evaluation.testsDefault) / float64(examples):.2f} " &
    &"mean_tests_guided={float64(evaluation.testsGuided) / float64(examples):.2f} " &
    &"teacher_agreement={evaluation.correct}/{evaluation.decisions}"

proc familyReport*(evaluation: ExplorerEvaluation): string =
  ## Found within the budget per family, as default/guided/total.
  for family in evaluation.families:
    let counts = evaluation.perFamily[family]
    if result.len > 0: result.add(' ')
    result.add(&"{family}={counts.plain}/{counts.guided}/{counts.total}")
