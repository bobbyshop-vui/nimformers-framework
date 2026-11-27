## The Spectron tool policy: Linear++ attention plus a Mamba2SSD branch in every layer,
## driving the exact explorer one tool call at a time.
##
## A step reads the engine's own feature vector for the sequence and the node the walker
## stands on, a short fingerprint (the first 16 terms modulo 1009), the tools that are
## legal there, how deep the transform path is, and the last few tool calls together with
## the engine's verdict on each. It writes one tool from a fixed
## alphabet - a hypothesis, a transform, or a pop - and, when the tool is the holonomic fit,
## the order and the degree that fit takes. The exact engine runs and checks that step before
## the next one is asked for. There is no free text and no separate scan: the tool chain is
## the search, and `program` in the reply is the chain that ran.
import std/[hashes, json, random, sequtils, sets, strformat, times]
import backend, nimformer
import test_nimformer as spectron
import big_integer, sequence_explorer, explorer_policy

const
  Slots = ord(hPeriodic) + 1 + ord(high(Transform)) + 1   ## The search hypotheses and the transforms.
  MaxDepth = 3
  HistorySteps = 4
  MaxProgramOrder = 5
  MaxProgramDegree = 6
  MaxSteps = 4096                  ## One tool call per step, up to a full test budget.
  ProgramSequenceLength = 64
  ProgramFutureTerms = 32
  DefaultProgramEvalSeed* = 42 xor 0x5EED
  SeqMod = 1009
  SeqWindow = 16
  ProgPad = 0
  ProgBos = 1
  ProgSep = 2
  ProgAsk = 3                      ## The decision marker: the tool comes right after it.
  ProgStop = 4
  ProgFound = 5                    ## Verdict: the engine added a strong direction.
  ProgMiss = 6                     ## Verdict: nothing strong came out.
  ProgWait = 7                     ## Verdict: the tool is named, its parameters are pending.
  ProgScan = 8                     ## Verdict: the named shape missed and the shape scan found it.
  HypBase = 9                      ## + ord(Hypothesis)
  TransformBase = HypBase + 9      ## + ord(Transform)
  ProgPop = TransformBase + 9
  OrderBase = ProgPop + 1          ## + (r - 1), r = 1 .. MaxProgramOrder
  DegreeBase = OrderBase + MaxProgramOrder   ## + d, d = 0 .. MaxProgramDegree
  LegalBase = DegreeBase + MaxProgramDegree + 1   ## + slot, the tool is still legal
  BlockedBase = LegalBase + Slots                 ## + slot, the tool is spent here
  DepthBase = BlockedBase + Slots
  SeqResidueBase = DepthBase + MaxDepth + 1
  FeatureLevels = 4               ## Each engine feature is quantised to this many buckets.
  FeatureBase = SeqResidueBase + SeqMod
  PendingLevels = 4               ## Buckets for the work left behind each transform.
  PendingBase = FeatureBase + ExplorerFeatures * FeatureLevels
  ProgVocab* = PendingBase + TransformSlots * PendingLevels
  StateTokens = 1 + SeqWindow + 1 + Slots + 1 + HistorySteps * 2 +
    ExplorerFeatures + TransformSlots + 1
  DefaultProgramCheckpoint* = "models/sequence_program.nimq"

type
  ProgramPolicy* = object
    model*: spectron.NimformerModelSpectron
    step*: int
  ProgramSample = object
    ids: seq[int]
    action: int
  ProgramDifficulty = enum
    pdBasic, pdIntermediate, pdAdvanced
  ProgramRoute = object
    samples: seq[ProgramSample]
    difficulty: ProgramDifficulty
  ProgramTask = object
    root: Track
    hint: seq[Transform]
    target: JsonNode
    answer: BigInt
  AnswerVerdict = enum
    avCorrect, avWrong, avNoAnswer

const
  CurriculumStart: array[ProgramDifficulty, float] = [0.50, 0.35, 0.15]
  CurriculumEnd: array[ProgramDifficulty, float] = [0.10, 0.25, 0.65]

proc toolToken(action: ExplorerAction): int {.inline.} =
  let i = int(action)
  if i <= ord(hPeriodic): HypBase + i
  elif i < int(PopAction): TransformBase + (i - ord(hPeriodic) - 1)
  else: ProgPop

proc tokenAction(tok: int): int {.inline.} =
  if tok >= HypBase and tok < HypBase + 9: tok - HypBase
  elif tok >= TransformBase and tok < ProgPop: ord(hPeriodic) + 1 + (tok - TransformBase)
  elif tok == ProgPop: int(PopAction)
  else: -1

proc pushVerdict(history: var seq[(int, int)], tool, verdict: int) =
  history.add((tool, verdict))
  if history.len > HistorySteps: history.delete(0)

proc featureToken(index: int, value: float32): int {.inline.} =
  ## One engine feature as one token: the value is bucketed, so the alphabet stays fixed.
  let scaled = (float64(value) + 1.0) * 0.5
  var bucket = int(scaled * float64(FeatureLevels))
  if bucket < 0: bucket = 0
  elif bucket >= FeatureLevels: bucket = FeatureLevels - 1
  FeatureBase + index * FeatureLevels + bucket

proc workToken(count: int): int {.inline.} =
  ## Work left behind one transform: nothing to step into, nothing left, a little, or a lot.
  let bucket =
    if count < 0: 0
    elif count == 0: 1
    elif count <= 2: 2
    else: 3
  PendingBase + bucket

proc encodeState(ex: var Explorer, legal: set[ExplorerAction],
                 history: openArray[(int, int)]): seq[int] =
  ## The policy's whole input: the engine's features, the fingerprint, the live legal set,
  ## the depth, the recent calls, and the work left behind each transform.
  result = newSeq[int](StateTokens)
  var i = 0
  result[i] = ProgBos
  inc i
  for r in ex.node(@[]).fingerprint(int64(SeqMod), SeqWindow):
    result[i] = SeqResidueBase + int(r)
    inc i
  result[i] = ProgSep
  inc i
  for slot in 0 ..< Slots:
    result[i] = (if ExplorerAction(slot) in legal: LegalBase + slot else: BlockedBase + slot)
    inc i
  result[i] = DepthBase + min(ex.depth, MaxDepth)
  inc i
  for k in 0 ..< HistorySteps:
    if k < history.len:
      result[i] = history[k][0]
      inc i
      result[i] = history[k][1]
      inc i
    else:
      result[i] = ProgPad
      inc i
      result[i] = ProgPad
      inc i
  let frame = ex.explorerFrame(legal)
  for index in 0 ..< ExplorerFeatures:
    result[i] = featureToken(index, frame[index])
    inc i
  for count in ex.pendingWork():
    result[i] = workToken(count)
    inc i
  result[i] = ProgAsk

proc sampleFrom(state: seq[int], target: int): ProgramSample =
  ProgramSample(ids: state, action: target)

proc pickToken(policy: ProgramPolicy, ctx: Backend, state: seq[int], allowed: seq[int]): int =
  let logits = policy.model.forward(state, ctx)
  let base = (logits.shape[0] - 1) * logits.shape[1]
  result = allowed[0]
  var best = logits.data[base + result]
  for tok in allowed:
    if logits.data[base + tok] > best:
      best = logits.data[base + tok]
      result = tok

proc recordTool(ex: var Explorer, history: var seq[(int, int)],
                action: ExplorerAction, samples: var seq[ProgramSample]) =
  ## A tool that takes no parameters: record the state, run it, record the verdict.
  let legal = ex.legalActions()
  if action notin legal: return
  samples.add(sampleFrom(encodeState(ex, legal, history), toolToken(action)))
  let before = ex.strongCount
  ex.act(action)
  pushVerdict(history, toolToken(action),
    (if ex.strongCount > before: ProgFound else: ProgMiss))

proc recordFit(ex: var Explorer, history: var seq[(int, int)],
               r, d: int, samples: var seq[ProgramSample]) =
  ## A holonomic call with its shape: three tokens, then the engine.
  let legal = ex.legalActions()
  if hypothesisAction(hHolonomic) notin legal: return
  samples.add(sampleFrom(encodeState(ex, legal, history), HypBase + ord(hHolonomic)))
  pushVerdict(history, HypBase + ord(hHolonomic), ProgWait)
  samples.add(sampleFrom(encodeState(ex, legal, history), OrderBase + r - 1))
  pushVerdict(history, OrderBase + r - 1, ProgWait)
  samples.add(sampleFrom(encodeState(ex, legal, history), DegreeBase + d))
  let before = ex.strongCount
  let named = ex.testAtFixed(hHolonomic, r, d)
  pushVerdict(history, DegreeBase + d,
    (if named: ProgFound elif ex.strongCount > before: ProgScan else: ProgMiss))

proc answerVerdict(ex: Explorer, task: ProgramTask): AnswerVerdict =
  try:
    let reply = evaluateTarget(ex, task.root, task.target)
    let value = toBig(reply{"value"}, "answer")
    if value == task.answer: avCorrect else: avWrong
  except ValueError:
    avNoAnswer

proc routeDifficulty(direction: Direction): ProgramDifficulty =
  let order = direction.report{"order"}.getInt()
  let degree = direction.report{"polynomial_degree"}.getInt(direction.report{"degree"}.getInt())
  let largeHolonomic = direction.hypothesis == hHolonomic and (order + 1) * (degree + 1) >= 25
  let longLinear = direction.hypothesis == hLinearRecurrence and order >= 6
  if direction.path.len >= 2 or largeHolonomic or longLinear: return pdAdvanced
  if direction.path.len == 1 or order > 2 or degree > 2 or direction.hypothesis == hAlgebraic:
    return pdIntermediate
  pdBasic

proc collectSamples(task: ProgramTask, budget: int): ProgramRoute =
  ## Label a route only when the engine can reproduce a withheld answer along it.
  let accept: DirectionFilter = proc(direction: Direction): bool =
    direction.hypothesis in StrongHypotheses and
      answerVerdict(Explorer(directions: @[direction]), task) == avCorrect
  var probe = newExplorer(task.root)
  probe.runDefault(1, budget, task.hint, accept)
  let index = probe.firstStrong()
  if index < 0: return
  let found = probe.directions[index]
  var ex = newExplorer(task.root)
  var history: seq[(int, int)]
  var samples: seq[ProgramSample]
  for transform in found.path:
    let action = transformAction(transform)
    if action notin ex.legalActions(): return
    recordTool(ex, history, action, samples)
  let action = hypothesisAction(found.hypothesis)
  if action notin ex.legalActions(): return
  let r = found.report{"order"}.getInt(-1)
  let d = found.report{"polynomial_degree"}.getInt(-1)
  if found.hypothesis == hHolonomic and r in 1 .. MaxProgramOrder and d in 0 .. MaxProgramDegree:
    recordFit(ex, history, r, d, samples)
  else:
    recordTool(ex, history, action, samples)
  if ex.directions.len > 0 and answerVerdict(ex, task) == avCorrect:
    result = ProgramRoute(samples: samples, difficulty: routeDifficulty(ex.directions[0]))

proc writeProgram*(policy: ProgramPolicy, ctx: Backend, ex: var Explorer,
                   maxDirections, budget: int): seq[int] =
  ## Drive the engine one tool at a time until a strong direction is found or the test
  ## budget is spent. Every step is checked before the next one is written.
  var history: seq[(int, int)]
  var steps = 0
  while ex.tests < budget and ex.strongCount < maxDirections and steps < MaxSteps:
    inc steps
    let legal = ex.legalActions()
    if legal == {}: break
    var allowed: seq[int]
    for action in legal: allowed.add(toolToken(action))
    let choice = policy.pickToken(ctx, encodeState(ex, legal, history), allowed)
    let action = tokenAction(choice)
    if action < 0: break
    result.add(choice)
    if action <= ord(hPeriodic) and Hypothesis(action) == hHolonomic:
      pushVerdict(history, choice, ProgWait)
      var orders: seq[int]
      for r in 1 .. MaxProgramOrder: orders.add(OrderBase + r - 1)
      let order = policy.pickToken(ctx, encodeState(ex, legal, history), orders)
      result.add(order)
      pushVerdict(history, order, ProgWait)
      var degrees: seq[int]
      for d in 0 .. MaxProgramDegree: degrees.add(DegreeBase + d)
      let degree = policy.pickToken(ctx, encodeState(ex, legal, history), degrees)
      result.add(degree)
      let before = ex.strongCount
      let named = ex.testAtFixed(hHolonomic, order - OrderBase + 1, degree - DegreeBase)
      pushVerdict(history, degree,
        (if named: ProgFound elif ex.strongCount > before: ProgScan else: ProgMiss))
    else:
      let before = ex.strongCount
      ex.act(action)
      pushVerdict(history, choice, (if ex.strongCount > before: ProgFound else: ProgMiss))

proc renderProgram*(tokens: seq[int]): seq[string] =
  ## The tool chain as a reader sees it: tool names and the fit's parameters, no text.
  var i = 0
  while i < tokens.len:
    let tok = tokens[i]
    let action = tokenAction(tok)
    if action < 0:
      inc i
      continue
    if action <= ord(hPeriodic) and Hypothesis(action) == hHolonomic and i + 2 < tokens.len:
      result.add(&"P-recursive(r={tokens[i + 1] - OrderBase + 1},d={tokens[i + 2] - DegreeBase})")
      i += 3
    elif action == int(PopAction):
      result.add("pop")
      inc i
    elif action <= ord(hPeriodic):
      result.add($Hypothesis(action))
      inc i
    else:
      result.add($Transform(action - ord(hPeriodic) - 1))
      inc i

proc nextProgramTask(rng: var Rand, hardFraction: float, seen: var HashSet[Hash]): ProgramTask =
  # Keep the requested mix of basic and hard examples when replacing a duplicate.
  let hard = rng.rand(1.0) < hardFraction
  let modulus = ExplorerPrimes[0]
  while true:
    let example = generateProgramExample(rng, ProgramSequenceLength + ProgramFutureTerms, hard)
    let visible = example.terms[0 ..< ProgramSequenceLength]
    if seen.containsOrIncl(hash((example.offset, visible))): continue
    let index = example.offset + example.terms.high
    return ProgramTask(root: rootTrack(visible, example.offset), hint: example.hint,
      target: %*{"index": index, "modulus": modulus}, answer: big(example.terms[^1].modSmall(modulus)))

proc programHoldout(size, seed: int, seen: var HashSet[Hash]): seq[ProgramTask] =
  var rng = initRand(seed)
  for _ in 0 ..< size:
    result.add(nextProgramTask(rng, 0.8, seen))

proc scoreProgram(policy: ProgramPolicy, ctx: Backend, tasks: openArray[ProgramTask], budget: int) =
  let started = epochTime()
  var guided, plain: array[AnswerVerdict, int]
  var guidedTests, plainTests = 0
  for task in tasks:
    var ex = newExplorer(task.root)
    discard writeProgram(policy, ctx, ex, 1, budget)
    inc guided[answerVerdict(ex, task)]
    guidedTests += ex.tests
    var probe = newExplorer(task.root)
    probe.runDefault(1, budget)
    inc plain[answerVerdict(probe, task)]
    plainTests += probe.tests
  let n = float(max(1, tasks.len))
  echo &"[program] holdout policy step={policy.step} correct={guided[avCorrect]}/{tasks.len} " &
    &"wrong={guided[avWrong]} no_answer={guided[avNoAnswer]} mean_tests={float(guidedTests) / n:.2f}"
  echo &"[program] holdout plain correct={plain[avCorrect]}/{tasks.len} " &
    &"wrong={plain[avWrong]} no_answer={plain[avNoAnswer]} mean_tests={float(plainTests) / n:.2f} " &
    &"in {epochTime() - started:.1f}s for both orders"

proc evaluateProgram*(policy: ProgramPolicy, ctx: Backend, size, seed, budget: int) =
  var seen: HashSet[Hash]
  scoreProgram(policy, ctx, programHoldout(size, seed, seen), budget)

proc curriculumMix(step, steps: int, available: set[ProgramDifficulty]): array[ProgramDifficulty, float] =
  let progress = float(step - 1) / float(max(1, steps - 1))
  var total = 0.0
  for difficulty in available:
    result[difficulty] = (1.0 - progress) * CurriculumStart[difficulty] + progress * CurriculumEnd[difficulty]
    total += result[difficulty]
  for difficulty in available: result[difficulty] /= total

proc sampleDifficulty(rng: var Rand, mix: array[ProgramDifficulty, float],
                      available: set[ProgramDifficulty]): ProgramDifficulty =
  var draw = rng.rand(1.0)
  for difficulty in available:
    result = difficulty
    draw -= mix[difficulty]
    if draw <= 0.0: return

proc trainProgram*(path: string, steps, batch, pool, seed: int, hidden, heads, layers: int,
                   lr: float32, ctx: Backend, hardFraction = 0.8,
                   evalSize = 200, evalSeed = DefaultProgramEvalSeed, initial = ProgramPolicy()): ProgramPolicy =
  ## Train the next tool decision from routes that compute the correct withheld answer.
  var rng = initRand(seed)
  var samples: array[ProgramDifficulty, seq[ProgramSample]]
  var seen: HashSet[Hash]
  let holdout = programHoldout(evalSize, evalSeed, seen)
  echo &"[program] holdout sequences={holdout.len} seed={evalSeed} vocabulary={ProgVocab} state={StateTokens}"
  var verified, sampleCount = 0
  let poolStart = epochTime()
  echo &"[program] generating {pool} distinct sequences with {ProgramSequenceLength} terms " &
    &"and hard fraction {hardFraction:.2f}"
  for i in 1 .. pool:
    let task = nextProgramTask(rng, hardFraction, seen)
    let route = collectSamples(task, TeacherBudget)
    if route.samples.len > 0:
      inc verified
      sampleCount += route.samples.len
      samples[route.difficulty].add(route.samples)
    if i mod 1_000 == 0 and i < pool:
      echo &"[program] teacher sequences={i}/{pool} verified={verified} samples={sampleCount} in {epochTime() - poolStart:.1f}s"
  if sampleCount == 0: raise newException(ValueError, "The teacher produced no verified routes")
  echo &"[program] teacher verified={verified}/{pool} samples={sampleCount} in {epochTime() - poolStart:.1f}s"
  echo &"[program] curriculum samples basic={samples[pdBasic].len} intermediate={samples[pdIntermediate].len} " &
    &"advanced={samples[pdAdvanced].len}"
  let trainStart = epochTime()
  result = initial
  if result.model.vocab == 0:
    result.model = spectron.newNimformerModelSpectron(ProgVocab, hidden, heads, layers, 4, useSession = true)
  else:
    echo &"[program] starting from step {initial.step} with a fresh Adam optimizer"
  var adam = spectron.newModelAdamState(result.model)
  var available: set[ProgramDifficulty]
  var nextSample: array[ProgramDifficulty, int]
  for difficulty in ProgramDifficulty:
    if samples[difficulty].len > 0:
      available.incl(difficulty)
      rng.shuffle(samples[difficulty])
  for step in 1 .. steps:
    var idsBatch, targetsBatch: seq[seq[int]]
    let mix = curriculumMix(step, steps, available)
    for _ in 0 ..< batch:
      let difficulty = sampleDifficulty(rng, mix, available)
      if nextSample[difficulty] == samples[difficulty].len:
        rng.shuffle(samples[difficulty])
        nextSample[difficulty] = 0
      let sample = samples[difficulty][nextSample[difficulty]]
      inc nextSample[difficulty]
      idsBatch.add(sample.ids)
      var targets = newSeqWith(sample.ids.len, -1)
      targets[^1] = sample.action
      targetsBatch.add(targets)
    let loss = result.model.trainStep(adam, idsBatch, targetsBatch, ctx, lr = lr, requantizeEvery = 0)
    if step == 1 or step mod 100 == 0 or step == steps:
      echo &"[program] step={step}/{steps} action_loss={loss:.4f} samples={sampleCount} " &
        &"mix basic={mix[pdBasic] * 100:.1f}% intermediate={mix[pdIntermediate] * 100:.1f}% " &
        &"advanced={mix[pdAdvanced] * 100:.1f}%"
  result.step += steps
  echo &"[program] trained in {epochTime() - trainStart:.1f}s"
  spectron.saveCheckpoint(result.model, path, result.step)
  echo &"[program] saved {path}"
  scoreProgram(result, ctx, holdout, EvalBudget)

proc loadProgram*(path: string): ProgramPolicy =
  let loaded = spectron.loadCheckpoint(path)
  if loaded.model.vocab != ProgVocab:
    raise newException(ValueError, &"Program checkpoint has vocabulary {loaded.model.vocab}, expected {ProgVocab}: " & path)
  ProgramPolicy(model: loaded.model, step: loaded.step)

var activeProgram: ProgramPolicy
var activeContext: Backend

proc installProgram*(path: string, ctx: Backend): ProgramPolicy =
  ## Load a checkpoint and let it drive every explore_sequence search.
  activeProgram = loadProgram(path)
  activeContext = ctx
  programPrologue = proc(ex: var Explorer, maxDirections, budget: int) =
    programLog = renderProgram(writeProgram(activeProgram, activeContext, ex, maxDirections, budget))
  activeProgram
