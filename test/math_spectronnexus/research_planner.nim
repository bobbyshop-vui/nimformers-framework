## A light MLP that plans a chain of exact tools for a question about an integer
## sequence: the value at a far index, exactly or modulo M. It never writes text.
## Every step calls an exact tool, and the reward counts only verified answers.
import std/[json, math, os, random, strformat, strutils, tables, tempfiles]
import backend, nimformer, quant
import big_integer, calculator, explorer_policy, oeis_lookup, root_math, sequence_explorer

const
  PlannerFeatures* = 64
  PlannerActionCount* = 9
  MaxPlannerSteps* = 24
  PlannerVersion = 1
  ActionNames* = ["explore", "more_terms", "target_direct", "target_factors", "target_exact",
                  "target_multimodular", "verify_extra_prime", "answer", "give_up"]
  ActionCost* = 0.02
  WrongPenalty* = 1.0
  MaxPlannerTerms = MaxExplorerTerms
  PlannerTermScale = 400.0   ## Feature scale the policy was trained with; keeps old weights valid.
  MaxMultimodularRounds = 3
  MaxVerifications = 3
  MaxStabilisingPrimes = 50_000 ## About 466,000 digits; primes come on demand.
  PlannerParameterNames = ["features.weight", "features.bias", "actions.weight", "actions.bias"]

type
  PlannerAction* = range[0 .. PlannerActionCount - 1]
  PlannerFrame* = array[PlannerFeatures, float32]
  Candidate = object
    present: bool
    value: BigInt
    modulus: BigInt       ## Zero for an exact value.
    source: PlannerAction
  ExploreResult = object
    ex: Explorer
    root: Track
    found: bool
    path: seq[Transform]
    hypothesis: Hypothesis
    rational, modular: bool
  PlannerTask* = object
    family*: string
    pool*: seq[BigInt]     ## Terms that more_terms may reveal, in order.
    visible*: int
    offset*: int
    index*: int64
    modulus*: int64        ## Zero asks for the exact value.
    truth*: BigInt
    hasTruth*: bool
    generator*: JsonNode   ## A calculator expression in n, for MCP requests.
    explored: Table[int, ExploreResult]
    targets: Table[string, tuple[ok: bool, value: BigInt, reason: string, via: string]]
  PlannerState* = object
    terms: int
    explored, stale: bool
    info: ExploreResult
    tried, failed: set[PlannerAction]
    candidate: Candidate
    residues: seq[tuple[prime, residue: int64]]
    rounds, verified, mismatches, nextPrime: int
    estimateDigits: float64
    steps*: int
    done*, gaveUp*: bool
    answer*: Candidate
    trail*: JsonNode
  PlannerPolicy* = object
    features, actions: Linear
  PlannerAdam* = object
    steps*: int
    first, second: array[PlannerParameterNames.len, seq[float32]]
  PlannerSample* = object
    frame*: PlannerFrame
    allowed*: set[PlannerAction]
    action*: PlannerAction
    weight*: float32
  PlannerChooser* = proc(task: var PlannerTask, state: var PlannerState, legal: set[PlannerAction]): PlannerAction {.closure.}

var largePrimes: seq[int64]

proc verificationPrime(i: int): int64 =
  ## The i-th distinct prime below 2^31, found on demand, so modular products stay in
  ## fast 64 bit arithmetic and the supply never runs out (about 10^8 such primes).
  var candidate = if largePrimes.len == 0: 2_147_483_647'i64 else: largePrimes[^1] - 1
  while largePrimes.len <= i:
    if isPrime64(candidate): largePrimes.add(candidate)
    dec candidate
  largePrimes[i]

proc factorsOf(m: int64): seq[tuple[prime: int64, exponent: int]] = factorize(m)

type CrtAccumulator = object
  ## Residues joined one prime at a time (one Garner step each), instead of redoing the
  ## whole reconstruction after every new prime.
  joined, modulus: BigInt

proc initCrt(): CrtAccumulator = CrtAccumulator(joined: big(0), modulus: big(1))

proc add(acc: var CrtAccumulator, p, residue: int64) =
  let previous = acc.modulus.modSmall(p)
  if gcd(previous, p) != 1: return   # A repeated prime adds nothing.
  let step = mulMod(((residue - acc.joined.modSmall(p)) mod p + p) mod p, modularInverse(previous, p), p)
  acc.joined = acc.joined + acc.modulus * big(step)
  acc.modulus = acc.modulus * big(p)

proc symmetric(acc: CrtAccumulator): BigInt =
  if acc.modulus < acc.joined + acc.joined: acc.joined - acc.modulus else: acc.joined

proc squarefree(m: int64): bool =
  for f in factorsOf(m):
    if f.exponent > 1: return false
  true

# Task plumbing with caches, so repeated rollouts on one task stay cheap.

proc termsFor(task: var PlannerTask, count: int): seq[BigInt] =
  if not task.generator.isNil:
    let problem = %*{"generator": task.generator, "start": task.offset, "count": count}
    return readTerms(problem).terms
  task.pool[0 ..< min(count, task.pool.len)]

proc maxTerms(task: PlannerTask): int =
  if not task.generator.isNil: MaxPlannerTerms else: min(task.pool.len, MaxPlannerTerms)

proc exploreAt(task: var PlannerTask, count: int): ExploreResult =
  if count in task.explored: return task.explored[count]
  let terms = task.termsFor(count)
  result.root = rootTrack(terms, task.offset)
  result.ex = newExplorer(result.root)
  result.ex.explorePlain(1, 300, explorerGuide)
  let index = result.ex.firstEvaluable()
  if index >= 0:
    let d = result.ex.directions[index]
    result.found = true
    result.path = d.path
    result.hypothesis = d.hypothesis
    (result.rational, result.modular) = d.formInfo()
  task.explored[count] = result

proc targetValue(task: var PlannerTask, state: PlannerState, modulus: int64):
    tuple[ok: bool, value: BigInt, reason: string, via: string] =
  ## The far term from the current direction, modulo a number or exact (modulus 0).
  let key = $state.terms & ":" & $modulus
  if key in task.targets: return task.targets[key]
  var target = %*{"index": task.index}
  if modulus > 0: target["modulus"] = %modulus else: target["exact"] = %true
  try:
    let reply = evaluateTarget(state.info.ex, state.info.root, target)
    let value = reply{"value"}
    if value.isNil or value.kind notin {JInt, JString}:
      result = (false, big(0), "The exact value is too large to return in full", "")
    else:
      result = (true, toBig(value, "value"), "", reply{"route"}.getStr(""))
  except ValueError as error:
    result = (false, big(0), error.msg, "")
  task.targets[key] = result

proc estimateDigits(task: var PlannerTask, count: int): float64 =
  ## Least squares on log10 |a(n)| against n log n, n, log n and 1, extrapolated to the index.
  let terms = task.termsFor(count)
  var xs, ys: seq[seq[float64]]
  for i in max(0, terms.len - 24) ..< terms.len:
    if terms[i].isZero: continue
    let n = float64(task.offset + i + 2)
    xs.add(@[n * ln(n), n, ln(n), 1.0])
    ys.add(@[terms[i].log10Abs()])
  let far = float64(task.index + 2)
  let fallback = if terms.len > 0: max(1.0, terms[^1].log10Abs() * far / float64(task.offset + terms.len + 1)) else: 1.0
  if xs.len < 6: return fallback
  var a: array[4, array[5, float64]]
  for r in 0 ..< xs.len:
    for i in 0 .. 3:
      for j in 0 .. 3: a[i][j] += xs[r][i] * xs[r][j]
      a[i][4] += xs[r][i] * ys[r][0]
  for i in 0 .. 3: a[i][i] += 1e-9
  for col in 0 .. 3:
    var pivot = col
    for r in col + 1 .. 3:
      if abs(a[r][col]) > abs(a[pivot][col]): pivot = r
    swap(a[col], a[pivot])
    if abs(a[col][col]) < 1e-300: return fallback
    for r in 0 .. 3:
      if r == col: continue
      let factor = a[r][col] / a[col][col]
      for c in col .. 4: a[r][c] -= factor * a[col][c]
  let coefficients = [a[0][4] / a[0][0], a[1][4] / a[1][1], a[2][4] / a[2][2], a[3][4] / a[3][3]]
  let predicted = coefficients[0] * far * ln(far) + coefficients[1] * far + coefficients[2] * ln(far) + coefficients[3]
  if predicted.classify in {fcNan, fcInf, fcNegInf}: return fallback
  max(1.0, predicted)

proc polynomialDigits(task: var PlannerTask, count: int): tuple[polynomial: bool, digits: float64] =
  ## Digits for a sequence of polynomial growth, such as a multiplicative function:
  ## |a(n)| <= n^k with k the largest log|a(n)| / log n seen, so about k log10(index) digits.
  ## The growth counts as polynomial when that ratio is no more than 25% higher on the
  ## later half of the terms than on the earlier half; exponential growth keeps raising it.
  let terms = task.termsFor(count)
  var early, late = 0.0
  var seen = 0
  for i, t in terms:
    let n = float64(task.offset + i)
    if n < 2.0: continue
    inc seen
    let ratio = if t.isZero: 0.0 else: t.log10Abs() / log10(n)
    if i < terms.len div 2: early = max(early, ratio) else: late = max(late, ratio)
  let degree = max(1.0, max(early, late))
  (seen >= 16 and late <= 1.25 * max(early, 1.0), degree * log10(max(2.0, float64(task.index))) + 2.0)

# State, legality and transitions.

proc newPlannerState*(task: PlannerTask): PlannerState =
  result.terms = task.visible
  result.trail = newJArray()

proc legalActions*(task: PlannerTask, state: PlannerState): set[PlannerAction] =
  if state.done: return
  result.incl(PlannerAction(8))
  if not state.explored or state.stale: result.incl(PlannerAction(0))
  if state.terms < task.maxTerms() and PlannerAction(1) notin state.failed: result.incl(PlannerAction(1))
  let ready = state.explored and not state.stale and state.info.found
  if ready:
    if task.modulus > 0 and PlannerAction(2) notin state.tried: result.incl(PlannerAction(2))
    if task.modulus > 0 and PlannerAction(3) notin state.tried and squarefree(task.modulus) and
        factorsOf(task.modulus).len >= 2:
      result.incl(PlannerAction(3))
    if PlannerAction(4) notin state.tried: result.incl(PlannerAction(4))
    if state.rounds < MaxMultimodularRounds and not (state.candidate.present and state.candidate.source == 5):
      result.incl(PlannerAction(5))
  if state.candidate.present and state.candidate.modulus.isZero and state.verified + state.mismatches < MaxVerifications:
    result.incl(PlannerAction(6))
  if state.candidate.present: result.incl(PlannerAction(7))

proc note(state: var PlannerState, action: PlannerAction, outcome: JsonNode) =
  outcome["action"] = %ActionNames[action]
  state.trail.add(outcome)

proc setCandidate(state: var PlannerState, value, modulus: BigInt, source: PlannerAction) =
  state.candidate = Candidate(present: true, value: value, modulus: modulus, source: source)

proc crtOf(residues: seq[tuple[prime, residue: int64]]): CrtAccumulator =
  result = initCrt()
  for item in residues: result.add(item.prime, item.residue)

proc symmetricCrt(residues: seq[tuple[prime, residue: int64]]): BigInt = crtOf(residues).symmetric()

proc nextVerificationPrime(state: var PlannerState): int64 =
  result = verificationPrime(state.nextPrime)
  inc state.nextPrime

proc applyAction*(task: var PlannerTask, state: var PlannerState, action: PlannerAction) =
  inc state.steps
  case action
  of 0:
    state.info = task.exploreAt(state.terms)
    state.explored = true
    state.stale = false
    state.tried = {}
    state.failed = {}
    state.candidate = Candidate()
    state.residues = @[]
    state.rounds = 0
    # The n log n growth fit is meaningless for terms of polynomial growth such as
    # multiplicative functions: extrapolating it to 10^12 asked for ~10^11 primes.
    let growth = task.polynomialDigits(state.terms)
    state.estimateDigits =
      if growth.polynomial or (state.info.found and state.info.hypothesis == hMultiplicative): growth.digits
      else: task.estimateDigits(state.terms)
    var outcome = %*{"terms": state.terms, "found": state.info.found}
    if state.info.found:
      outcome["direction"] = %(pathExpression(state.info.path) & ": " & $state.info.hypothesis)
    state.note(action, outcome)
  of 1:
    let before = state.terms
    let wanted = min(task.maxTerms(), max(state.terms * 2, state.terms + 8))
    try:
      if not task.generator.isNil: discard task.termsFor(wanted)
      state.terms = wanted
      state.stale = true
      state.note(action, %*{"terms_before": before, "terms_after": state.terms})
    except ValueError as error:
      state.failed.incl(action)
      state.tried.incl(action)
      state.note(action, %*{"failed": error.msg})
  of 2:
    state.tried.incl(action)
    let got = task.targetValue(state, task.modulus)
    if got.ok:
      state.setCandidate(got.value, big(task.modulus), action)
      state.note(action, %*{"modulus": task.modulus, "value": got.value.toJson()})
      if got.via.len > 0: state.trail[^1]["via"] = %got.via
    else:
      state.failed.incl(action)
      state.note(action, %*{"modulus": task.modulus, "failed": got.reason})
  of 3:
    state.tried.incl(action)
    var parts: seq[tuple[prime, residue: int64]]
    var reason = ""
    for f in factorsOf(task.modulus):
      let got = task.targetValue(state, f.prime)
      if not got.ok:
        reason = "modulo " & $f.prime & ": " & got.reason
        break
      parts.add((f.prime, got.value.modSmall(f.prime)))
    if reason.len == 0:
      var r: seq[BigInt]
      var m: seq[int64]
      for p in parts:
        r.add(big(p.residue))
        m.add(p.prime)
      let combined = chineseRemainder(r, m)
      state.setCandidate(combined.value, combined.modulus, action)
      state.note(action, %*{"factors": m, "value": combined.value.toJson()})
    else:
      state.failed.incl(action)
      state.note(action, %*{"failed": reason})
  of 4:
    state.tried.incl(action)
    let got = task.targetValue(state, 0)
    if got.ok:
      state.setCandidate(got.value, big(0), action)
      state.note(action, %*{"digits": got.value.digits()})
    else:
      state.failed.incl(action)
      state.note(action, %*{"failed": got.reason})
  of 5:
    state.tried.incl(action)
    inc state.rounds
    let wanted = min(MaxStabilisingPrimes div 2,
      int(ceil((state.estimateDigits * float64(1 shl (state.rounds - 1)) + 2.0) / 9.3)) + 1)
    var attempts = 0
    var failing = 0
    var reason = ""
    while state.residues.len < wanted and attempts < wanted + 8:
      inc attempts
      let p = state.nextVerificationPrime()
      let got = task.targetValue(state, p)
      if got.ok:
        state.residues.add((p, got.value.modSmall(p)))
        failing = 0
      else:
        if reason.len == 0: reason = got.reason
        # A prime may divide a denominator, but a run of failures means the route is closed.
        inc failing
        if failing >= 16: break
    if state.residues.len == 0:
      state.failed.incl(action)
      state.note(action, %*{"failed": reason})
    else:
      # The digit estimate is only a starting point: growth such as exp(c sqrt n)
      # escapes the fitted model. Keep adding primes until one more prime leaves the
      # symmetric reconstruction unchanged, so the value no longer depends on the guess.
      var acc = crtOf(state.residues)
      var value = acc.symmetric()
      var stable = false
      var extra = 0
      var failing = 0
      while not stable and state.nextPrime < MaxStabilisingPrimes and extra < MaxStabilisingPrimes:
        inc extra
        let p = state.nextVerificationPrime()
        let got = task.targetValue(state, p)
        if not got.ok:
          # A prime may divide a denominator, but a run of failures means the route is closed.
          inc failing
          if failing >= 16: break
          continue
        failing = 0
        let residue = got.value.modSmall(p)
        stable = value.modSmall(p) == residue
        state.residues.add((p, residue))
        if not stable:
          acc.add(p, residue)
          value = acc.symmetric()
      if stable:
        state.estimateDigits = max(state.estimateDigits, float64(value.digits()))
      state.setCandidate(value, big(0), action)
      state.verified = 0
      state.note(action, %*{"primes": state.residues.len, "digits": value.digits(), "stable": stable})
  of 6:
    let p = state.nextVerificationPrime()
    let got = task.targetValue(state, p)
    if not got.ok:
      state.note(action, %*{"prime": p, "failed": got.reason})
    elif got.value.modSmall(p) == state.candidate.value.modSmall(p):
      inc state.verified
      if state.candidate.source == 5: state.residues.add((p, got.value.modSmall(p)))
      state.note(action, %*{"prime": p, "agrees": true})
    else:
      inc state.mismatches
      if state.candidate.source == 5:
        state.residues.add((p, got.value.modSmall(p)))
        state.tried.excl(PlannerAction(5))
      state.candidate = Candidate()
      state.note(action, %*{"prime": p, "agrees": false})
  of 7:
    state.done = true
    var answer = state.candidate
    if task.modulus > 0 and answer.modulus.isZero:
      answer.value = big(answer.value.modSmall(task.modulus))
      answer.modulus = big(task.modulus)
    state.answer = answer
    state.note(action, %*{"value": answer.value.toJson()})
  else:
    state.done = true
    state.gaveUp = true
    state.note(action, %*{})

proc reward*(task: PlannerTask, state: PlannerState): float64 =
  result = -ActionCost * float64(state.steps)
  if state.gaveUp or not state.done or not state.answer.present: return
  let expected = if task.modulus > 0: big(task.truth.modSmall(task.modulus)) else: task.truth
  if state.answer.value == expected: result += 1.0 else: result -= WrongPenalty

proc correct*(task: PlannerTask, state: PlannerState): int =
  ## 1 for a right answer, -1 for a wrong one, 0 when the planner gave up.
  if not state.done or state.gaveUp or not state.answer.present: return 0
  let expected = if task.modulus > 0: big(task.truth.modSmall(task.modulus)) else: task.truth
  if state.answer.value == expected: 1 else: -1

# Features.

proc plannerFrame*(task: PlannerTask, state: PlannerState, legal: set[PlannerAction]): PlannerFrame =
  proc unit(x: float64): float32 = float32(max(-1.0, min(1.0, x)))
  let kind = if task.modulus == 0: 2 elif isPrime64(task.modulus): 0 else: 1
  result[kind] = 1
  if task.modulus > 0:
    let factors = factorsOf(task.modulus)
    result[3] = unit(log10(float64(task.modulus)) / 10.0)
    result[4] = float32(ord(squarefree(task.modulus)))
    result[5] = float32(ord(factors[0].prime <= 100))
    result[6] = unit(float64(factors.len) / 4.0)
  result[7] = unit(log10(float64(task.index + 1)) / 6.0)
  result[8] = unit(float64(state.terms) / PlannerTermScale)
  result[9] = float32(ord(state.terms < task.maxTerms()))
  result[10] = float32(ord(state.explored))
  result[11] = float32(ord(state.stale))
  let found = state.explored and state.info.found
  result[12] = float32(ord(found))
  if found:
    result[13 + min(state.info.path.len, 2)] = 1
    # A self-referential relation is iterated term by term like a holonomic one, so it
    # shares that slot and the trained checkpoint keeps its 64 features.
    let slot = if state.info.hypothesis in {hSelfReferential, hBinaryRecursive}: ord(hHolonomic)
               else: ord(state.info.hypothesis)
    result[16 + slot] = 1
    for t in state.info.path: result[22 + ord(t)] = 1
    result[31] = float32(ord(state.info.rational))
    result[32] = float32(ord(state.info.modular))
  for a in state.tried: result[33 + int(a)] = 1
  for a in state.failed: result[42 + int(a)] = 1
  if state.candidate.present:
    result[51] = 1
    result[52] = float32(ord(state.candidate.modulus.isZero))
    result[53] = float32(ord(state.candidate.source == 5))
    result[54] = float32(ord(state.candidate.source in {PlannerAction(2), PlannerAction(3), PlannerAction(4)}))
  result[55] = unit(float64(state.verified) / 3.0)
  result[56] = unit(float64(state.mismatches) / 3.0)
  result[57] = unit(state.estimateDigits / 1000.0)
  result[58] = unit(log10(float64(task.index + 1) / float64(task.offset + state.terms + 1)) / 4.0)
  result[59] = unit(float64(state.steps) / float64(MaxPlannerSteps))
  result[60] = unit(float64(state.residues.len) / 64.0)
  result[61] = float32(ord(task.index > 100_000))
  result[62] = float32(ord(task.index > 20_000))
  result[63] = float32(ord(task.index > 20_000_000))
  discard legal

# Rule planners.

proc rulePlanner*(task: var PlannerTask, state: var PlannerState, legal: set[PlannerAction]): PlannerAction =
  ## Fixed order without lookahead: the baseline and the fallback without a checkpoint.
  for a in [0, 7]:
    if a == 7 and state.candidate.present and state.candidate.source == 5 and state.verified == 0 and
        PlannerAction(6) in legal: return PlannerAction(6)
    if PlannerAction(a) in legal: return PlannerAction(a)
  if not state.info.found:
    return if PlannerAction(1) in legal: PlannerAction(1) else: PlannerAction(8)
  for a in [2, 3, 4, 5]:
    if PlannerAction(a) in legal: return PlannerAction(a)
  if PlannerAction(1) in legal: return PlannerAction(1)
  PlannerAction(8)

proc expertPlanner*(task: var PlannerTask, state: var PlannerState, legal: set[PlannerAction]): PlannerAction =
  ## Teacher with lookahead: it tries each route on a copy and takes the cheapest that works.
  if PlannerAction(0) in legal: return PlannerAction(0)
  if state.candidate.present:
    let expected = if task.modulus > 0: big(task.truth.modSmall(task.modulus)) else: task.truth
    var answer = state.candidate.value
    if task.modulus > 0 and state.candidate.modulus.isZero: answer = big(answer.modSmall(task.modulus))
    if answer == expected:
      if state.candidate.source == 5 and state.verified == 0 and PlannerAction(6) in legal: return PlannerAction(6)
      return PlannerAction(7)
    if PlannerAction(6) in legal: return PlannerAction(6)
    return PlannerAction(8)
  if not state.info.found:
    return if PlannerAction(1) in legal: PlannerAction(1) else: PlannerAction(8)
  for a in [2, 3, 4, 5]:
    let action = PlannerAction(a)
    if action notin legal: continue
    var probe = state
    probe.trail = newJArray()
    task.applyAction(probe, action)
    if probe.candidate.present:
      var value = probe.candidate.value
      if task.modulus > 0 and probe.candidate.modulus.isZero: value = big(value.modSmall(task.modulus))
      let expected = if task.modulus > 0: big(task.truth.modSmall(task.modulus)) else: task.truth
      if value == expected or a == 5: return action
  if PlannerAction(1) in legal: return PlannerAction(1)
  PlannerAction(8)

proc persistent(chooser: PlannerChooser): PlannerChooser =
  ## A learned policy may stop early on a direction kind it was not trained on, such as a
  ## multiplicative function with a prime power model. While a route to the target is
  ## still untried, serving follows the rule order instead of giving up.
  result = proc(task: var PlannerTask, state: var PlannerState, legal: set[PlannerAction]): PlannerAction =
    result = chooser(task, state, legal)
    if result == PlannerAction(8) and legal * {PlannerAction(2) .. PlannerAction(5)} != {}:
      result = rulePlanner(task, state, legal)

proc runPlanner*(task: var PlannerTask, chooser: PlannerChooser, record: ptr seq[PlannerSample] = nil): PlannerState =
  result = newPlannerState(task)
  while not result.done and result.steps < MaxPlannerSteps:
    let legal = task.legalActions(result)
    if legal == {}: break
    let action = chooser(task, result, legal)
    if action notin legal: break
    if not record.isNil and card(legal) > 1:
      record[].add(PlannerSample(frame: task.plannerFrame(result, legal), allowed: legal, action: action, weight: 1))
    try:
      task.applyAction(result, action)
    except ValueError as error:
      result.failed.incl(action)
      result.tried.incl(action)
      result.note(action, %*{"failed": error.msg})
  if not result.done:
    result.done = true
    result.gaveUp = true

# Synthetic tasks with known answers.

const
  QuestionPrimes = [1_000_000_007'i64, 998_244_353, 1_000_000_009, 65_537, 10_007, 7_919]
  CompositeModuli = [210'i64, 30, 6, 1_001, 10_007 * 65_537, 2 * 1_000_000_007, 6 * 998_244_353,
                     65_537 * 131_071, 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23]

const
  WideModuli = [2_305_843_009_213_693_951'i64, 1_000_000_000_000_000_009, 4_611_686_018_427_387_847,
                9_999_999_967, 1_000_000_000_000, 1_099_511_627_776, 847_288_609_443]
    ## 2^61-1, 10^18+9, a prime below 2^62, a prime past 9*10^9, 10^12, 2^40 and 3^25:
    ## moduli past the old 9*10^9 bound, and composites that share factors with the data.
  FarFamilies* = ["far_multiplicative", "far_linear", "far_hypergeometric", "far_holonomic",
                  "wide_modulus", "long_exact"]

proc farModulus(rng: var Rand, exactShare: int): int64 =
  ## 0 asks for the exact value.
  let roll = rng.rand(99)
  if roll < exactShare: 0'i64
  elif roll < exactShare + (100 - exactShare) div 2: QuestionPrimes[rng.rand(QuestionPrimes.high)]
  else: WideModuli[rng.rand(WideModuli.high)]

proc matrixPowerTerm(coefficients, initial: seq[int64], index: int64, m: int64): int64 =
  ## a(index) mod m for a(k) = sum c_j a(k-1-j), by a companion matrix power. It is
  ## independent of the explorer's Kitamasa code, so it can judge it.
  let r = coefficients.len
  if index < r: return floorMod(big(initial[index]), big(m)).toInt64()
  proc lift(x: int64): int64 = floorMod(big(x), big(m)).toInt64()
  proc multiply(a, b: seq[seq[int64]]): seq[seq[int64]] =
    result = newSeq[seq[int64]](r)
    for i in 0 ..< r:
      result[i] = newSeq[int64](r)
      for j in 0 ..< r:
        var total = 0'i64
        for k in 0 ..< r: total = (total + mulMod(a[i][k], b[k][j], m)) mod m
        result[i][j] = total
  var base = newSeq[seq[int64]](r)
  for i in 0 ..< r: base[i] = newSeq[int64](r)
  for j in 0 ..< r: base[0][j] = lift(coefficients[j])
  for i in 1 ..< r: base[i][i - 1] = 1 mod m
  var power = newSeq[seq[int64]](r)
  for i in 0 ..< r:
    power[i] = newSeq[int64](r)
    power[i][i] = 1 mod m
  var e = index - int64(r - 1)
  while e > 0:
    if (e and 1) != 0: power = multiply(power, base)
    base = multiply(base, base)
    e = e shr 1
  var total = 0'i64
  for j in 0 ..< r: total = (total + mulMod(power[0][j], lift(initial[r - 1 - j]), m)) mod m
  total

proc multiplicativeValue(kind: int, n: int64): BigInt =
  ## phi, sigma, tau, mu or Liouville's lambda at n from its factorization.
  result = big(1)
  var parts = 0
  var squarefree = true
  for (p, e) in factorize(n):
    parts += e
    if e > 1: squarefree = false
    case kind
    of 0: result = result * power(big(p), e - 1) * big(p - 1)
    of 1: result = result * divMod(power(big(p), e + 1) - big(1), big(p - 1)).quotient
    of 2: result = result * big(e + 1)
    else: discard
  case kind
  of 3: result = if not squarefree: big(0) elif parts mod 2 == 0: big(1) else: big(-1)
  of 4: result = if parts mod 2 == 0: big(1) else: big(-1)
  else: discard

proc generateFarTask*(rng: var Rand, hardFraction: float64): PlannerTask =
  ## Questions at far indices, past the old modulus bound, or exact at long range. The
  ## truth comes from code independent of the explorer: factorizations, matrix powers,
  ## direct products and iteration of the generating recurrence.
  let family = rng.rand(FarFamilies.high)
  result.family = FarFamilies[family]
  result.hasTruth = true
  result.visible = [24, 32, 48][rng.rand(2)]
  var terms: seq[BigInt]
  proc settle(task: var PlannerTask, exact: BigInt) =
    task.truth = if task.modulus > 0: big(exact.modSmall(task.modulus)) else: exact
  case family
  of 0:
    let kind = rng.rand(4)
    result.offset = 1
    for n in 1'i64 .. 96: terms.add(multiplicativeValue(kind, n))
    result.index = rng.rand(1_000_000'i64 .. 1_000_000_000_000_000'i64)
    result.modulus = farModulus(rng, 40)
    result.settle(multiplicativeValue(kind, result.index))
  of 1:
    let order = rng.rand(2 .. 3)
    var c, initial: seq[int64]
    for _ in 0 ..< order: c.add(int64(rng.rand(-3 .. 3)))
    if c[^1] == 0: c[^1] = 1
    for _ in 0 ..< order: initial.add(int64(rng.rand(-3 .. 3)))
    if initial[^1] == 0: initial[^1] = 1
    for i in 0 ..< 96:
      if i < order: terms.add(big(initial[i]))
      else:
        var value = big(0)
        for j in 0 ..< order: value = value + big(c[j]) * terms[i - 1 - j]
        terms.add(value)
    result.modulus = farModulus(rng, 15)
    if result.modulus == 0:
      result.index = rng.rand(300'i64 .. 3_000'i64)
      var window = terms
      while window.len <= result.index:
        var value = big(0)
        for j in 0 ..< order: value = value + big(c[j]) * window[^(1 + j)]
        window.add(value)
      result.truth = window[result.index]
    else:
      result.index = rng.rand(10_000'i64 .. 1_000_000_000_000_000'i64)
      result.truth = big(matrixPowerTerm(c, initial, result.index, result.modulus))
  of 2:
    # a(n) = a(n-1) * (alpha n + beta), or the Catalan numbers.
    let catalan = rng.rand(1) == 0
    let (alpha, beta) = (int64(rng.rand(1 .. 3)), int64(rng.rand(0 .. 3)))
    let start = if catalan: 1'i64 else: int64(rng.rand(1 .. 3))
    result.modulus = farModulus(rng, 35)
    result.index = if result.modulus == 0: rng.rand(1_000'i64 .. 6_000'i64)
                   elif catalan: rng.rand(5_000'i64 .. 20_000'i64)
                   else: rng.rand(5_000'i64 .. 200_000'i64)
    let length = if result.modulus == 0: result.index + 1 else: 96'i64
    var value = big(start)
    for n in 0'i64 ..< length:
      if n > 0:
        value = if catalan: exactQuotientBig(value * big(2 * (2 * n - 1)), big(n + 1))
                else: value * big(alpha * n + beta)
      if n < 96: terms.add(value)
    if result.modulus == 0:
      result.truth = value
    else:
      # Residues by a direct product. Catalan(n) = C(2n, n) / (n + 1) modulo m needs no
      # inverse when computed from the exact product of the ratios, so use exact values
      # for the Catalan case at this range and residues for the product case.
      let m = result.modulus
      if catalan:
        var exact = big(1)
        for n in 1'i64 .. result.index: exact = exactQuotientBig(exact * big(2 * (2 * n - 1)), big(n + 1))
        result.truth = big(exact.modSmall(m))
      else:
        var residue = floorMod(big(start), big(m)).toInt64()
        for n in 1'i64 .. result.index: residue = mulMod(residue, (alpha * n + beta) mod m, m)
        result.truth = big(residue)
  of 3:
    # a(n) = (a n + b) a(n-1) + (c n + d) a(n-2), iterated modulo m or exactly.
    let (a, b, c, d) = (int64(rng.rand(-3 .. 3)), int64(rng.rand(-3 .. 3)), int64(rng.rand(1 .. 3)), int64(rng.rand(-3 .. 3)))
    let (u0, u1) = (int64(rng.rand(1 .. 3)), int64(rng.rand(1 .. 3)))
    terms = @[big(u0), big(u1)]
    for n in 2'i64 ..< 96: terms.add(big(a * n + b) * terms[^1] + big(c * n + d) * terms[^2])
    result.modulus = farModulus(rng, 30)
    if result.modulus == 0:
      result.index = rng.rand(500'i64 .. 3_000'i64)
      var (x, y) = (terms[^2], terms[^1])
      for n in 96'i64 .. result.index: (x, y) = (y, big(a * n + b) * y + big(c * n + d) * x)
      result.truth = y
    else:
      let m = result.modulus
      result.index = rng.rand(2_000'i64 .. 50_000'i64)
      proc lift(v: int64): int64 = floorMod(big(v), big(m)).toInt64()
      var (x, y) = (lift(u0), lift(u1))
      for n in 2'i64 .. result.index:
        (x, y) = (y, (mulMod(lift(a * n + b), y, m) + mulMod(lift(c * n + d), x, m)) mod m)
      result.truth = big(y)
  of 4:
    # A basic or hard family at the usual range, modulo a wide or non-squarefree modulus.
    let relative = rng.rand(100 .. 220)
    let example = generateExample(rng, relative + 1, hardFraction)
    result.family = "wide_modulus_" & example.family
    result.offset = example.offset
    result.index = int64(example.offset + relative)
    terms = example.terms[0 ..< min(relative - 10, 96)]
    result.modulus = WideModuli[rng.rand(WideModuli.high)]
    result.settle(example.terms[relative])
  else:
    # A basic family asked exactly at a longer range than the original tasks.
    let relative = rng.rand(300 .. 700)
    let example = generateExample(rng, relative + 1, 0.0)
    result.family = "long_exact_" & example.family
    result.offset = example.offset
    result.index = int64(example.offset + relative)
    terms = example.terms[0 ..< 96]
    result.modulus = 0
    result.truth = example.terms[relative]
  result.pool = terms

proc generateTask*(rng: var Rand, hardFraction: float64, farFraction = 0.0): PlannerTask =
  if farFraction > 0 and rng.rand(1.0) < farFraction: return generateFarTask(rng, hardFraction)
  let kind = rng.rand(99)
  let exact = kind >= 65
  let relative = if exact: rng.rand(70 .. 150) else: rng.rand(100 .. 220)
  let example = generateExample(rng, relative + 1, hardFraction)
  result.family = example.family
  result.offset = example.offset
  result.index = int64(example.offset + relative)
  result.truth = example.terms[relative]
  result.hasTruth = true
  result.visible = [24, 32, 48][rng.rand(2)]
  result.pool = example.terms[0 ..< min(relative - 10, 96)]
  result.modulus = if exact: 0'i64
                   elif kind < 35: QuestionPrimes[rng.rand(QuestionPrimes.high)]
                   else: CompositeModuli[rng.rand(CompositeModuli.high)]

# The policy.

proc newPlannerPolicy*(hidden: int, initializeWeights = true): PlannerPolicy =
  result.features = newLinear(PlannerFeatures, hidden, initializeWeights)
  result.actions = newLinear(hidden, PlannerActionCount, initializeWeights)

proc framesTensor(frames: openArray[PlannerFrame]): Tensor =
  result = newTensor(@[frames.len, PlannerFeatures])
  for i in 0 ..< frames.len:
    copyMem(addr result.data[i * PlannerFeatures], unsafeAddr frames[i][0], sizeof(PlannerFrame))

proc scores(policy: PlannerPolicy, frames: openArray[PlannerFrame], ctx: Backend): Tensor =
  var hidden = policy.features.forward(framesTensor(frames), ctx)
  for value in hidden.data.mitems:
    if value < 0: value *= 0.1'f32
  policy.actions.forward(hidden, ctx)

proc masked(logits: openArray[float32], allowed: set[PlannerAction]): seq[(PlannerAction, float64)] =
  var top = -Inf
  for a in allowed: top = max(top, float64(logits[int(a)]))
  var total = 0.0
  for a in allowed: total += exp(float64(logits[int(a)]) - top)
  for a in allowed: result.add((a, exp(float64(logits[int(a)]) - top) / total))

proc greedy*(policy: PlannerPolicy, ctx: Backend): PlannerChooser =
  result = proc(task: var PlannerTask, state: var PlannerState, legal: set[PlannerAction]): PlannerAction =
    let logits = policy.scores([task.plannerFrame(state, legal)], ctx)
    var best = -1.0
    for (a, p) in masked(logits.data.toOpenArray(0, PlannerActionCount - 1), legal):
      if p > best:
        best = p
        result = a

proc sampler*(policy: PlannerPolicy, ctx: Backend, rng: ptr Rand): PlannerChooser =
  result = proc(task: var PlannerTask, state: var PlannerState, legal: set[PlannerAction]): PlannerAction =
    let logits = policy.scores([task.plannerFrame(state, legal)], ctx)
    let probabilities = masked(logits.data.toOpenArray(0, PlannerActionCount - 1), legal)
    var u = rng[].rand(1.0)
    result = probabilities[^1][0]
    for (a, p) in probabilities:
      u -= p
      if u <= 0:
        result = a
        break

proc adamUpdate(param: var Tensor, gradient: Tensor, opt: var PlannerAdam, slot: int, lr: float32) =
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

proc update*(policy: var PlannerPolicy, adam: var PlannerAdam, batch: openArray[PlannerSample],
             ctx: Backend, lr: float32): float64 =
  ## Weighted masked cross entropy. Weight 1 is supervision; an advantage gives a policy gradient.
  var frames = newSeq[PlannerFrame](batch.len)
  for i, sample in batch: frames[i] = sample.frame
  let input = framesTensor(frames)
  var hidden = policy.features.forward(input, ctx)
  for value in hidden.data.mitems:
    if value < 0: value *= 0.1'f32
  let logits = policy.actions.forward(hidden, ctx)
  var dLogits = newTensor(logits.shape)
  for row, sample in batch:
    let base = row * PlannerActionCount
    for (a, p) in masked(logits.data.toOpenArray(base, base + PlannerActionCount - 1), sample.allowed):
      let target = float64(ord(a == sample.action))
      dLogits.data[base + int(a)] = float32(float64(sample.weight) * (p - target) / float64(batch.len))
      if a == sample.action: result -= float64(sample.weight) * ln(max(p, 1e-300)) / float64(batch.len)
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

proc savePlanner*(policy: PlannerPolicy, steps: int, path: string) =
  let params = [policy.features.weight, policy.features.bias, policy.actions.weight, policy.actions.bias]
  var sd: seq[(string, QuantTensor)]
  for i, name in PlannerParameterNames: sd.add((name, quantizeRaw(params[i].data, params[i].shape)))
  let fullPath = absolutePath(path)
  createDir(fullPath.parentDir)
  let (file, tempPath) = createTempFile(".planner-", ".nimq", fullPath.parentDir)
  file.close()
  try:
    saveQuantStateDict(tempPath, [PlannerFeatures, policy.features.outF, PlannerActionCount, PlannerVersion, steps], sd)
    moveFile(tempPath, fullPath)
  finally:
    if fileExists(tempPath): removeFile(tempPath)

proc loadPlannerPolicy*(path: string): PlannerPolicy =
  let (arch, tensors, _) = loadQuantStateDict(path, mapped = true)
  if arch.len < 5 or arch[0] != PlannerFeatures or arch[1] notin 1 .. 1024 or
      arch[2] != PlannerActionCount or arch[3] != PlannerVersion:
    raise newException(ValueError, "Not a compatible research planner checkpoint")
  var byName: Table[string, QuantTensor]
  for (name, tensor) in tensors: byName[name] = tensor
  proc read(name: string, shape: seq[int]): seq[float32] =
    if not byName.hasKey(name): raise newException(ValueError, "Missing checkpoint tensor " & name)
    let tensor = byName[name]
    if tensor.kind != qkFp32Raw or tensor.shape != shape:
      raise newException(ValueError, "Invalid checkpoint tensor " & name)
    dequantizeRaw(tensor)
  result = newPlannerPolicy(arch[1], initializeWeights = false)
  result.features.weight.data = read(PlannerParameterNames[0], @[arch[1], arch[0]])
  result.features.bias.data = read(PlannerParameterNames[1], @[arch[1]])
  result.actions.weight.data = read(PlannerParameterNames[2], @[arch[2], arch[1]])
  result.actions.bias.data = read(PlannerParameterNames[3], @[arch[2]])

type PlannerScore* = object
  tasks*, right*, wrong*, gaveUp*: int
  actions*: int
  rewardTotal*: float64

proc score*(tasks: var seq[PlannerTask], chooser: PlannerChooser): PlannerScore =
  for task in tasks.mitems:
    let state = task.runPlanner(chooser)
    inc result.tasks
    case task.correct(state)
    of 1: inc result.right
    of -1: inc result.wrong
    else: inc result.gaveUp
    result.actions += state.steps
    result.rewardTotal += task.reward(state)

proc `$`*(s: PlannerScore): string =
  let n = max(s.tasks, 1)
  &"tasks={s.tasks} right={s.right} wrong={s.wrong} gave_up={s.gaveUp} " &
    &"mean_actions={float64(s.actions) / float64(n):.2f} mean_reward={s.rewardTotal / float64(n):.3f}"

# Serving.

var plannerChooser*: PlannerChooser
var plannerContext: Backend

proc installPlanner*(path: string, ctx: Backend) =
  ## ctx is a GPU backend chosen by the caller.
  plannerContext = ctx
  plannerChooser = loadPlannerPolicy(path).greedy(plannerContext)

proc solutionOutline(task: PlannerTask, state: PlannerState, borrowed: string): JsonNode =
  ## The method in plain sentences, built from the direction's report and the steps taken.
  result = newJArray()
  let index = state.info.ex.firstEvaluable()
  let report = if index >= 0: state.info.ex.directions[index].report else: newJObject()
  let subject = if state.info.path.len == 0: "a(n)" else: "b = " & pathExpression(state.info.path) & " of a"
  var found = &"Explored {state.terms} terms"
  if borrowed.len > 0: found.add(&" (the supplied ones extended by the stored terms of {borrowed})")
  found.add(&": {subject} fits a {state.info.hypothesis}")
  let formula = report{"formula"}
  if not formula.isNil and formula.kind == JString: found.add(": " & formula.getStr())
  result.add(%(found & "."))
  let indexNote = report{"index_note"}
  if not indexNote.isNil and indexNote.kind == JString: result.add(indexNote)
  let validFrom = report{"valid_from_n"}
  if state.info.hypothesis == hSelfReferential and not validFrom.isNil and validFrom.getInt() > 0:
    result.add(%(&"The relation holds from n = {validFrom.getInt()}; the terms before it are initial values."))
  let evidence = report{"evidence"}
  if not evidence.isNil and evidence.kind == JObject:
    let status = evidence{"status"}.getStr()
    let primes = evidence{"primes_checked"}.getInt()
    if status == "fits_every_term_and_predicts_integers":
      let equations = evidence{"equations"}.getInt()
      let unknowns = evidence{"unknowns"}.getInt()
      let predicted = evidence{"integer_terms_predicted"}.getInt()
      result.add(%(&"It is the only relation of its shape through all {equations} equations ({unknowns} unknowns) " &
        &"modulo {primes} primes, and it predicts integers for the next {predicted} terms."))
    elif status == "holds_on_every_coprime_pair_in_range":
      let pairs = evidence{"pairs_checked"}.getInt()
      result.add(%(&"It held on all {pairs} coprime pairs inside the data."))
    else:
      let fitted = evidence{"fit_equations"}.getInt()
      let unseen = evidence{"unseen_equations_checked"}.getInt()
      result.add(%(&"It was fitted on {fitted} equations and held on {unseen} unseen ones modulo {primes} primes."))
    let caution = evidence{"caution"}
    if not caution.isNil and caution.kind == JString: result.add(caution)
  let farTerms = report{"far_terms"}
  if not farTerms.isNil and farTerms.kind == JString:
    result.add(%("Prime powers past the data: " & farTerms.getStr() & "."))
  let multiplicative = state.info.hypothesis == hMultiplicative
  if state.info.path.len > 0:
    result.add(%("Far terms of b come from the relation, and a is rebuilt by undoing " &
      pathExpression(state.info.path) & " term by term."))
  for step in state.trail:
    case step{"action"}.getStr()
    of "target_direct":
      if step.hasKey("value"):
        if multiplicative:
          result.add(%(&"Factored n = {task.index} and multiplied the values at its prime powers modulo {task.modulus}."))
        elif step.hasKey("via"):
          result.add(%(&"Evaluated n = {task.index} modulo {task.modulus} through " & step["via"].getStr() & "."))
        else:
          result.add(%(&"Iterated the relation to n = {task.index} directly modulo {task.modulus}."))
    of "target_factors":
      if step.hasKey("value"):
        let factors = $step["factors"]
        result.add(%(&"Iterated modulo each prime factor {factors} of {task.modulus} and joined them by CRT."))
    of "target_exact":
      if step.hasKey("digits"):
        let digits = step["digits"].getInt()
        if multiplicative:
          result.add(%(&"Factored n = {task.index} and multiplied the exact values at its prime powers " &
            &"({digits} digits)."))
        elif state.info.hypothesis == hSelfReferential:
          result.add(%(&"Iterated the relation to n = {task.index} modulo primes below 2^31 and joined them by " &
            &"CRT until one more prime agreed ({digits} digits)."))
        else:
          result.add(%(&"Iterated the relation exactly over the integers to n = {task.index} ({digits} digits)."))
    of "target_multimodular":
      if step.hasKey("primes"):
        let primes = step["primes"].getInt()
        let digits = step["digits"].getInt()
        result.add(%(&"Iterated modulo {primes} primes near 2^31 and rebuilt the exact value ({digits} digits) " &
          "by CRT, adding primes until one more left it unchanged."))
    of "verify_extra_prime":
      if step.hasKey("agrees"):
        let prime = step["prime"].getInt()
        result.add(%(&"Checked the candidate against the independent prime {prime}: " &
          (if step["agrees"].getBool(): "agrees." else: "disagrees, so it was discarded.")))
    else: discard
  var final = "Answer: " & $state.answer.value
  if task.modulus > 0: final.add(&" (mod {task.modulus})")
  result.add(%(final & "."))

proc wideQuestion(problem: JsonNode, index, modulus: BigInt): JsonNode =
  ## An index or modulus past 64 bits. The planner's state is int64, so the explorer's
  ## target ladder answers directly: residues, the exact value reduced, Kitamasa on big
  ## integers, or CRT over prime power parts.
  let (terms, offset) = readTerms(problem)
  let root = rootTrack(terms, offset)
  var ex = newExplorer(root)
  ex.explorePlain(1, 300, explorerGuide)
  if ex.firstEvaluable() < 0: ex.runDefault(4, 2_000)
  let target = %*{"index": $index}
  if not modulus.isZero: target["modulus"] = %($modulus)
  let found = ex.evaluateTarget(root, target)
  result = %*{"answer": found["value"], "status": found{"status"}.getStr("extrapolation_from_conjectured_direction"),
              "method": "explore_then_target", "via": found["via"], "steps": [],
              "solution": [&"Explored the {terms.len} supplied terms and took the first direction that evaluates far terms.",
                           "Evaluated it at n = " & $index & (if modulus.isZero: " exactly." else: " modulo " & $modulus & "."),
                           "The index or modulus is past 64 bits, so the explorer's target answered without the planner."],
              "note": "Valid if the direction holds for every n, which still needs a proof."}
  if not modulus.isZero: result["modulus"] = modulus.toJson()
  if found.hasKey("tier"): result["tier"] = found["tier"]
  if ex.directions.len > 0:
    let d = ex.directions[max(0, ex.firstEvaluable())]
    result["direction"] = %(pathExpression(d.path) & ": " & $d.hypothesis)

proc researchSequence*(problem: JsonNode): JsonNode =
  ## Plan and run exact tools until a verified-route answer or a clear stop.
  let question = problem{"question"}
  if question.isNil or question.kind != JObject: raise newException(ValueError, "question must be an object with index")
  let indexNode = question{"index"}
  if indexNode.isNil or indexNode.kind notin {JInt, JString}:
    raise newException(ValueError, "question.index must be an integer or a decimal string")
  let wideIndex = toBig(indexNode, "question.index")
  if wideIndex.isNegative: raise newException(ValueError, "question.index must be at least 0")
  let modulusNode = question{"modulus"}
  let wideModulus = if modulusNode.isNil or modulusNode.kind == JNull: big(0) else: toBig(modulusNode, "question.modulus")
  if wideModulus.isNegative or wideModulus == big(1):
    raise newException(ValueError, "question.modulus must be at least 2")
  if not wideIndex.fitsInt64() or not wideModulus.fitsInt64():
    return wideQuestion(problem, wideIndex, wideModulus)
  var task = PlannerTask(index: wideIndex.toInt64())
  task.modulus = wideModulus.toInt64()
  if problem.hasKey("generator"):
    task.generator = problem["generator"]
    let (terms, offset) = readTerms(problem)
    task.offset = offset
    task.visible = terms.len
  else:
    let (terms, offset) = readTerms(problem)
    task.pool = terms
    task.offset = offset
    task.visible = terms.len
  if task.index < task.offset: raise newException(ValueError, "question.index precedes the first term")
  if task.index < task.offset + task.visible and task.generator.isNil:
    let value = task.pool[task.index - task.offset]
    return %*{"answer": (if task.modulus > 0: %value.modSmall(task.modulus) else: value.toJson()),
              "status": "supplied_term", "steps": []}
  let chooser = if plannerChooser.isNil: PlannerChooser(rulePlanner) else: persistent(plannerChooser)
  var state = task.runPlanner(chooser)
  # Identify the supplied terms in the offline OEIS. When no route works, use only the
  # stored terms that every entry starting with these terms agrees on: answer from them
  # if they reach the index, else plan again with them, since more terms often expose a
  # relation that a short prefix hides. Entries that disagree are listed, not chosen.
  var identified: JsonNode
  var candidates: JsonNode
  var borrowed = ""
  let plannerName = if plannerChooser.isNil: "rules" else: "mlp_policy"
  if task.generator.isNil and oeisAvailable():
    let visibleTerms = task.pool[0 ..< task.visible]
    identified = newJArray()
    for m in lookupTerms(visibleTerms, 3)["matches"]:
      identified.add(%*{"anumber": m["anumber"], "name": m["name"], "url": m["url"],
                        "normalisation": m["match"]["normalisation"],
                        "first_matching_stored_term": m["match"]["first_matching_stored_term"]})
    if state.gaveUp or not state.answer.present:
      let (entries, capped) = alignedEntries(visibleTerms)
      if entries.len > 0 and not capped:
        var common = entries[0].terms
        for e in entries[1 .. ^1]:
          var agree = 0
          while agree < min(common.len, e.terms.len) and common[agree] == e.terms[agree]: inc agree
          common.setLen(agree)
        var tags: seq[string]
        for e in entries: tags.add(formatEntry(e.anumber))
        let relative = task.index - task.offset
        if relative < common.len:
          let value = common[relative]
          result = %*{"steps": state.trail, "planner": plannerName, "actions": state.steps,
                      "method": "oeis_stored_term",
                      "answer": (if task.modulus > 0: %value.modSmall(task.modulus) else: value.toJson()),
                      "status": "oeis_stored_value", "oeis": identified,
                      "solution": [&"No relation was found in the {task.visible} supplied terms.",
                        "Every OEIS entry that starts with these terms (" & tags.join(", ") &
                        &") stores the same term {task.index}, counting from index {task.offset}."],
                      "note": "The value is read from the OEIS, not computed. It answers the question only " &
                        "if your sequence is one of these entries with the same offset."}
          if task.modulus > 0: result["modulus"] = %task.modulus
          return
        if common.len > task.pool.len:
          var retry = task
          retry.pool = common[0 ..< min(common.len, MaxPlannerTerms)]
          retry.explored.clear()
          retry.targets.clear()
          let second = retry.runPlanner(chooser)
          if not second.gaveUp and second.answer.present:
            borrowed = tags.join(", ")
            task = retry
            state = second
        if borrowed.len == 0:
          candidates = newJArray()
          for e in entries:
            if relative < e.terms.len and candidates.len < 10:
              let value = e.terms[relative]
              candidates.add(%*{"anumber": formatEntry(e.anumber), "name": e.name,
                "value": (if task.modulus > 0: %value.modSmall(task.modulus) else: value.toJson())})
  result = %*{"steps": state.trail, "planner": plannerName, "actions": state.steps, "method": "exact_tool_chain"}
  if not identified.isNil: result["oeis"] = identified
  if state.gaveUp or not state.answer.present:
    result["status"] = %"no_answer"
    result["answer"] = newJNull()
    var note = "No route produced a value. See the steps: supply more terms, a generator, or a different modulus."
    if not candidates.isNil and candidates.len > 0:
      result["oeis_candidates"] = candidates
      note.add(" OEIS entries that start with these terms disagree at this index; oeis_candidates lists " &
        "each entry's stored value. Pick one only after checking its definition against the problem.")
    elif identified.len > 0:
      note.add(" The oeis entries' definitions may suggest a generator or a transform.")
    result["note"] = %note
    return
  result["answer"] = state.answer.value.toJson()
  if task.modulus > 0: result["modulus"] = %task.modulus
  result["status"] = %(if borrowed.len > 0: "extrapolation_from_oeis_terms" else: "extrapolation_from_conjectured_direction")
  result["source"] = %ActionNames[state.answer.source]
  result["verifications"] = %*{"agreeing_primes": state.verified, "disagreements": state.mismatches}
  result["direction"] = %(pathExpression(state.info.path) & ": " & $state.info.hypothesis)
  result["solution"] = solutionOutline(task, state, borrowed)
  result["note"] = %("The value is exact for the direction found, which held on unseen terms. " &
    "It is correct for every n only if that direction is proved." &
    (if borrowed.len > 0: " The extra terms came from " & borrowed & ", so it also assumes your sequence is that entry." else: ""))
