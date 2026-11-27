## Seeded generators and a bounded curriculum driven by policy failures.
import std/[json, random]
import discrete_math, polynomial_math, prime_field, recurrence_math

type
  Variant* = enum
    sparseRank, cancellationRank, latePivotRank,
    layeredFlow, cancellationFlow, denseFlow, blockedFlow,
    sharedGcdFactor, repeatedGcdFactor, longEuclidChain,
    sparseRecurrence, denseRecurrence, transferRecurrence,
    researchRank, researchFlow, researchPolynomialGcd
  SampleDifficulty* = enum sdIntro, sdNormal, sdHard, sdResearch, sdHoldout
  Example* = object
    variant*: Variant
    problem*: MathProblem
  HardCase = object
    example: Example
    failure: float64
  HardMiner* = object
    observations: array[Variant, int]
    passRate: array[Variant, float64]
    replay: seq[HardCase]
    mutations*: int

const
  ReplayLimit = 48
  BaseVariants* = sparseRank .. longEuclidChain
  ResearchVariants* = [researchRank, researchFlow, researchPolynomialGcd,
                        sparseRecurrence, denseRecurrence, transferRecurrence]

proc variants*(research: bool): seq[Variant] =
  if research: return @ResearchVariants
  result = newSeqOfCap[Variant](ord(BaseVariants.b) - ord(BaseVariants.a) + 1)
  for variant in BaseVariants: result.add(variant)

proc trainingDifficulty*(rng: var Rand, progress: float64): SampleDifficulty =
  let starterShare = 0.05 + 0.45 * (1.0 - min(4.0 * progress, 1.0))
  let advancedShare = 0.15 + 0.55 * min(2.0 * progress, 1.0)
  let draw = rng.rand(1.0)
  if draw < starterShare: sdIntro
  elif draw < starterShare + advancedShare:
    if progress >= 0.25 and rng.rand(1.0) < 0.35: sdResearch else: sdHard
  else: sdNormal

proc problemSize(rng: var Rand, minimum: int, difficulty: SampleDifficulty): int =
  case difficulty
  of sdIntro: rng.rand(minimum .. 8)
  of sdNormal: rng.rand(minimum .. MaxOrder)
  of sdHard, sdResearch, sdHoldout: rng.rand(max(10, minimum) .. MaxOrder)

proc fieldPrime(rng: var Rand, difficulty: SampleDifficulty, regular: openArray[int]): int =
  const
    IntroPrimes = [2, 3, 5, 7]
    HardPrimes = [7, 17, 31, 61, 97, 127, 193, 229]
    HoldoutPrimes = [43, 103, 251]
  case difficulty
  of sdIntro: IntroPrimes[rng.rand(IntroPrimes.high)]
  of sdNormal: regular[rng.rand(regular.high)]
  of sdHard, sdResearch: HardPrimes[rng.rand(HardPrimes.high)]
  of sdHoldout: HoldoutPrimes[rng.rand(HoldoutPrimes.high)]

proc rankExample(rng: var Rand, variant: Variant, difficulty: SampleDifficulty): MathProblem =
  result.kind = rankProblem
  result.rows = rng.problemSize(3, difficulty)
  result.cols = rng.problemSize(3, difficulty)
  let primes = if variant == cancellationRank: [2, 3, 5, 7, 11] else: [3, 5, 7, 17, 31]
  result.prime = rng.fieldPrime(difficulty, primes)
  let offset = if variant == latePivotRank: rng.rand(1 .. min(5, result.cols - 1)) else: 0
  let maxRank = min(result.rows, result.cols - offset)
  let minRank = if difficulty in {sdHard, sdResearch, sdHoldout}: maxRank div 2 else: 0
  let rank = rng.rand(minRank .. maxRank)
  for i in 0 ..< rank:
    result.values[i][i + offset] = 1

  # Elementary row and column additions preserve the rank by construction.
  let rounds = (if variant == sparseRank: 1 else: 4) * max(result.rows, result.cols)
  for _ in 0 ..< rounds:
    let i = rng.rand(result.rows - 1)
    let j = (i + rng.rand(1 .. result.rows - 1)) mod result.rows
    let a = rng.rand(1 .. result.prime - 1)
    for c in offset ..< result.cols:
      result.values[i][c] = (result.values[i][c] + a * result.values[j][c]) mod result.prime
    let width = result.cols - offset
    if width > 1:
      let c = offset + rng.rand(width - 1)
      let k = offset + (c - offset + rng.rand(1 .. width - 1)) mod width
      let b = rng.rand(1 .. result.prime - 1)
      for r in 0 ..< result.rows:
        result.values[r][c] = (result.values[r][c] + b * result.values[r][k]) mod result.prime

proc relabel(problem: MathProblem, rng: var Rand): MathProblem =
  result = problem
  var labels = newSeq[int](problem.rows)
  for i in 0 ..< labels.len: labels[i] = i
  rng.shuffle(labels)
  result.source = labels[problem.source]
  result.sink = labels[problem.sink]
  for u in 0 ..< problem.rows:
    for v in 0 ..< problem.rows:
      result.values[labels[u]][labels[v]] = problem.values[u][v]

proc flowExample(rng: var Rand, variant: Variant, difficulty: SampleDifficulty): MathProblem =
  result.kind = flowProblem
  let minNodes = if variant == cancellationFlow: 6 else: 4
  result.rows = rng.problemSize(minNodes, difficulty)
  result.cols = result.rows
  result.source = 0
  result.sink = result.rows - 1
  let n = result.rows
  let sink = result.sink
  case variant
  of cancellationFlow:
    # The first augmenting path through 1 -> 3 can force a later reverse edge.
    # A second bipartite branch gives the optimum 2 * capacity.
    let capacity = rng.rand(1 .. 9)
    for (u, v) in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (3, sink), (4, sink)]:
      result.values[u][v] = capacity
    for v in 5 ..< sink:
      result.values[rng.rand(0 .. 4)][v] = rng.rand(1 .. 9)
      if v + 1 < sink: result.values[v][v + 1] = rng.rand(1 .. 9)
  of layeredFlow:
    for u in 0 ..< sink:
      result.values[u][u + 1] = rng.rand(1 .. 9)
      for v in u + 2 .. min(sink, u + 4):
        if rng.rand(1.0) < 0.45: result.values[u][v] = rng.rand(1 .. 9)
  of denseFlow, blockedFlow:
    let split = n div 2
    for u in 0 ..< n:
      for v in 0 ..< n:
        if u == v or u == sink or v == 0: continue
        if variant == blockedFlow and u < split and v >= split: continue
        if rng.rand(1.0) < 0.55: result.values[u][v] = rng.rand(1 .. 13)
    if variant == denseFlow:
      result.values[0][sink] = rng.rand(1 .. 9)
  else:
    raise newException(ValueError, "Expected flow variant")
  if difficulty == sdHoldout:
    for u in 0 ..< n:
      for v in 0 ..< n:
        if result.values[u][v] > 0:
          let offset = if variant == cancellationFlow: 0 else: rng.rand(16)
          result.values[u][v] = 17 * result.values[u][v] + offset
  result = relabel(result, rng)

proc randomPolynomial(rng: var Rand, degree, prime: int): Polynomial =
  result = newSeq[int](degree + 1)
  for i in 0 ..< degree: result[i] = rng.rand(prime - 1)
  result[^1] = rng.rand(1 .. prime - 1)

proc polynomialExample(rng: var Rand, variant: Variant, difficulty: SampleDifficulty): MathProblem =
  let p = rng.fieldPrime(difficulty, [2, 3, 5, 7, 17, 31])
  let commonDegree = rng.rand(1 .. (if difficulty == sdIntro: 2 else: 4))
  let degreeLimit = MaxOrder - 1 - commonDegree
  let maxDegree = if difficulty == sdIntro: min(degreeLimit, 5) else: degreeLimit
  let minDegree = if difficulty in {sdHard, sdResearch, sdHoldout}: maxDegree - 3 else: 1
  var common, f, g: Polynomial
  case variant
  of sharedGcdFactor:
    common = rng.randomPolynomial(commonDegree, p)
    f = rng.randomPolynomial(rng.rand(minDegree .. maxDegree), p)
    g = rng.randomPolynomial(rng.rand(minDegree .. maxDegree), p)
  of repeatedGcdFactor:
    let root = residue(-int64(rng.rand(p - 1)), p)
    common = @[1]
    for _ in 0 ..< commonDegree: common = multiply(common, @[root, 1], p)
    f = rng.randomPolynomial(rng.rand(minDegree .. maxDegree), p)
    g = rng.randomPolynomial(rng.rand(minDegree .. maxDegree), p)
    if difficulty notin {sdHard, sdResearch, sdHoldout} and rng.rand(2) == 0: g = @[]
  of longEuclidChain:
    # Each quotient is linear and the previous remainder is the next input.
    # The last member is a unit, so these two factors are coprime.
    common = rng.randomPolynomial(commonDegree, p)
    f = @[rng.rand(p - 1), 1]
    g = @[1]
    for _ in 2 .. rng.rand(max(4, minDegree) .. maxDegree):
      var next = multiply(f, @[rng.rand(p - 1), 1], p)
      for i, coefficient in g: next[i] = (next[i] + coefficient) mod p
      g = f
      f = next
  else:
    raise newException(ValueError, "Expected polynomial gcd variant")
  MathProblem(kind: polyGcdProblem, prime: p, polynomial: PolynomialProblem(prime: p,
    f: multiply(common, f, p), g: multiply(common, g, p)))

proc distantIndex(rng: var Rand, digits: int): string =
  result = newString(digits)
  result[0] = char(ord('0') + rng.rand(1 .. 9))
  for i in 1 ..< digits: result[i] = char(ord('0') + rng.rand(9))

proc recurrenceExample(rng: var Rand, variant: Variant, difficulty: SampleDifficulty): MathProblem =
  let p = rng.fieldPrime(difficulty, [2, 3, 5, 7, 17, 31])
  let bound = case difficulty
    of sdIntro: rng.rand(3 .. 7)
    of sdNormal: rng.rand(6 .. MaxOrder)
    of sdHard, sdResearch, sdHoldout: rng.rand(10 .. MaxOrder)
  let order = case variant
    of sparseRecurrence: rng.rand(1 .. min(6, bound))
    of denseRecurrence, transferRecurrence: rng.rand((bound + 1) div 2 .. bound)
    else: raise newException(ValueError, "Expected recurrence variant")
  let digits = case variant
    of sparseRecurrence: 12
    of denseRecurrence: 36
    else: 80
  let index = rng.distantIndex(digits)
  if variant == transferRecurrence:
    var matrix = newSeq[seq[int]](bound)
    for i in 0 ..< bound:
      matrix[i] = newSeq[int](bound)
    for i in 0 ..< order:
      for j in 0 ..< bound:
        if rng.rand(2) == 0: matrix[i][j] = rng.rand(1 .. p - 1)
      matrix[i][(i + 1) mod order] = rng.rand(1 .. p - 1)
    # Add vertices that cannot reach the sink, so matrix size can exceed the true order.
    for i in order ..< bound:
      for j in order ..< bound: matrix[i][j] = rng.rand(p - 1)
    let walk = WalkProblem(prime: p, matrix: matrix, source: rng.rand(order - 1),
      sink: rng.rand(order - 1), length: index)
    return MathProblem(kind: recurrenceProblem, prime: p, walk: walk, recurrence: walkRecurrence(walk))
  var coefficients = newSeq[int](order)
  coefficients[^1] = rng.rand(1 .. p - 1)
  if variant == sparseRecurrence:
    if order > 1: coefficients[rng.rand(order - 2)] = rng.rand(1 .. p - 1)
  else:
    for i in 0 ..< order - 1: coefficients[i] = rng.rand(p - 1)
  var samples = newSeq[int](2 * bound)
  for i in 0 ..< order: samples[i] = rng.rand(p - 1)
  for i in order ..< samples.len:
    for j, coefficient in coefficients:
      samples[i] = (samples[i] + coefficient * samples[i - j - 1]) mod p
  MathProblem(kind: recurrenceProblem, prime: p, recurrence: RecurrenceProblem(
    prime: p, orderBound: bound, samples: samples, index: index))

proc generate*(rng: var Rand, variant: Variant; difficulty = sdNormal): Example =
  result.variant = variant
  case variant
  of sparseRank, cancellationRank, latePivotRank:
    result.problem = rankExample(rng, variant, difficulty)
  of layeredFlow, cancellationFlow, denseFlow, blockedFlow:
    result.problem = flowExample(rng, variant, difficulty)
  of sharedGcdFactor, repeatedGcdFactor, longEuclidChain:
    result.problem = polynomialExample(rng, variant, difficulty)
  of researchRank:
    result.problem = rankExample(rng, latePivotRank, difficulty)
  of researchFlow:
    result.problem = flowExample(rng, denseFlow, difficulty)
  of researchPolynomialGcd:
    result.problem = polynomialExample(rng, longEuclidChain, difficulty)
  of sparseRecurrence, denseRecurrence, transferRecurrence:
    result.problem = recurrenceExample(rng, variant, difficulty)

proc generate*(rng: var Rand; difficulty = sdNormal; research = false): Example =
  let selected = variants(research)
  generate(rng, selected[rng.rand(selected.high)], difficulty)

proc mutate(example: Example, rng: var Rand): Example =
  ## Perturb a missed case without changing its certified structural target.
  result = example
  case example.problem.kind
  of flowProblem:
    result.problem = relabel(example.problem, rng)
  of rankProblem:
    let p = result.problem.prime
    let rows = result.problem.rows
    let cols = result.problem.cols
    let r = rng.rand(rows - 1)
    let s = (r + rng.rand(1 .. rows - 1)) mod rows
    let value = rng.rand(1 .. p - 1)
    let c = rng.rand(cols - 1)
    let d = (c + rng.rand(1 .. cols - 1)) mod cols
    for j in 0 ..< cols:
      result.problem.values[r][j] =
        (result.problem.values[r][j] + value * result.problem.values[s][j]) mod p
    for i in 0 ..< rows:
      swap(result.problem.values[i][c], result.problem.values[i][d])
  of polyGcdProblem:
    let p = result.problem.polynomial.prime
    let unit = rng.rand(1 .. p - 1)
    if rng.rand(1) == 0:
      swap(result.problem.polynomial.f, result.problem.polynomial.g)
    var f = result.problem.polynomial.f
    let g = result.problem.polynomial.g
    f.setLen(max(f.len, g.len))
    for i, term in g:
      f[i] = residue(int64(f[i]) + int64(unit) * int64(term), p)
    while f.len > 0 and f[^1] == 0: f.setLen(f.len - 1)
    result.problem.polynomial.f = f
  of recurrenceProblem:
    let index = rng.distantIndex(result.problem.recurrence.index.len)
    let p = result.problem.recurrence.prime
    let unit = if p > 2: rng.rand(2 .. p - 1) else: 1
    if result.problem.walk.matrix.len > 0:
      result.problem.walk.length = index
      # Multiplying the transfer matrix by a unit scales term k by unit^k.
      for row in result.problem.walk.matrix.mitems:
        for weight in row.mitems: weight = (weight * unit) mod p
      result.problem.recurrence = walkRecurrence(result.problem.walk)
      return
    var scale = 1
    for sample in result.problem.recurrence.samples.mitems:
      sample = (sample * scale) mod p
      scale = (scale * unit) mod p
    result.problem.recurrence.index = index

proc select*(miner: var HardMiner, rng: var Rand, enabled: bool; difficulty = sdNormal;
             preferMixed = false; research = false): Example =
  if not enabled or rng.rand(1.0) < 0.25: return generate(rng, difficulty, research)
  let selectedVariants = variants(research)
  if miner.replay.len > 0 and rng.rand(1.0) < 0.45:
    var total = 0.0
    for item in miner.replay: total += item.failure
    var draw = rng.rand(total)
    var selected = miner.replay.high
    for i, item in miner.replay:
      draw -= item.failure
      if draw <= 0:
        selected = i
        break
    result = mutate(miner.replay[selected].example, rng)
    miner.replay.del(selected)
    inc miner.mutations
    return

  var weights: array[Variant, float64]
  var total = 0.0
  for variant in selectedVariants:
    let pass = if miner.observations[variant] > 0: miner.passRate[variant] else: 0.5
    weights[variant] = if preferMixed: 0.15 + 0.55 * (1 - pass) + 3.0 * pass * (1 - pass)
                       else: 0.15 + (1 - pass)
    total += weights[variant]
  var draw = rng.rand(total)
  for variant in selectedVariants:
    draw -= weights[variant]
    if draw <= 0: return generate(rng, variant, difficulty)
  generate(rng, selectedVariants[^1], difficulty)

proc observe*(miner: var HardMiner, example: Example, solved, attempts: int; effortGap = 0.0) =
  let variant = example.variant
  let pass = float64(solved) / float64(attempts)
  if miner.observations[variant] == 0:
    miner.passRate[variant] = pass
  else:
    miner.passRate[variant] = 0.9 * miner.passRate[variant] + 0.1 * pass
  inc miner.observations[variant]
  if solved == attempts and effortGap < 0.10: return
  if miner.replay.len == ReplayLimit: miner.replay.delete(0)
  # Keep verified problems when a sampled method would save substantial work.
  let priority =
    if solved == attempts: min(1.2, 3.0 * effortGap)
    elif solved == 0: 0.75
    else: 1.5 - pass
  miner.replay.add(HardCase(example: example, failure: priority))

proc toJson*(miner: HardMiner): JsonNode =
  var rates = newJArray()
  var counts = newJArray()
  for variant in Variant:
    rates.add(%miner.passRate[variant])
    counts.add(%miner.observations[variant])
  var replay = newJArray()
  for item in miner.replay:
    replay.add(%*{"variant": ord(item.example.variant), "failure": item.failure,
                  "problem": problemJson(item.example.problem)})
  %*{"pass_rate": rates, "groups": counts, "replay": replay, "mutations": miner.mutations}

proc loadMiner*(node: JsonNode): HardMiner =
  let rates = node["pass_rate"]
  let counts = node["groups"]
  let replay = node["replay"]
  result.mutations = node{"mutations"}.getInt()
  let originalCount = ord(blockedFlow) + 1
  let currentCount = ord(high(Variant)) + 1
  let baseCount = ord(BaseVariants.b) + 1
  let legacyResearchCount = ord(transferRecurrence) + 1
  if result.mutations < 0 or rates.len notin [originalCount, baseCount, legacyResearchCount, currentCount] or
      rates.len != counts.len or replay.len > ReplayLimit:
    raise newException(ValueError, "Invalid mining state in checkpoint")
  for i in 0 ..< rates.len:
    let variant = Variant(i)
    result.passRate[variant] = rates[ord(variant)].getFloat()
    result.observations[variant] = counts[ord(variant)].getInt()
    if result.passRate[variant] notin 0.0 .. 1.0 or result.observations[variant] < 0:
      raise newException(ValueError, "Invalid mining statistics in checkpoint")
  for row in replay:
    let variant = row["variant"].getInt()
    let failure = row["failure"].getFloat()
    if variant notin 0 ..< rates.len or failure <= 0 or failure > 2:
      raise newException(ValueError, "Invalid replay item in checkpoint")
    let example = Example(variant: Variant(variant), problem: parseProblem(row["problem"]))
    result.replay.add(HardCase(example: example, failure: failure))
