## Train, inspect the stream, or solve a problem with the discrete policy.
import std/[json, math, os, parseopt, random, strformat, strutils, tables]
import backend
import discrete_math, exact_math, explorer_policy, math_mcp, math_policy, policy_validation, prepare_dataset,
  program_policy, research_planner, sequence_explorer

const
  DefaultTokens = 1_000_000_000'i64
  DefaultTrainBackend = when defined(macosx): "metal" else: "vulkan"
  ModelsDir = currentSourcePath().parentDir / "models"
  DefaultCheckpoint = ModelsDir / "math_policy.nimq"
  QualityCheckpoint = ModelsDir / "math_policy_quality.nimq"
  ResearchCheckpoint = ModelsDir / "math_policy_research.nimq"
  ExplorerCheckpoint = ModelsDir / "sequence_explorer.nimq"
  PlannerCheckpoint = ModelsDir / "research_planner.nimq"
  ProgramCheckpoint = ModelsDir / "sequence_program.nimq"
  DefaultRolloutSteps = 96
  Usage = """
Usage: math_spectronnexus <train|train-explorer|train-planner|train-program|eval-program|eval|data|calc|solve|explore|research|mcp> [options]

train-planner: --sft-steps=1500 --rl-iterations=300 --batch-states=128 --hidden=128 --pool=600
       --eval-tasks=200 --group=4 --rl-tasks=8 --eval-every=50 --lr=0.001 --rl-lr=0.0003
       --hard-fraction=0.5 --far-fraction=0.4 --seed=42 --backend=<name> --checkpoint=<path>
       Trains the tool-chain planner: a lookahead teacher first, then RL on verified answers.
       --far-fraction is the share of far-index, wide-modulus and long exact tasks.
research: read one research_sequence JSON object from stdin
       --planner-checkpoint=<path> uses a trained planner, otherwise a fixed rule order

train-explorer: --steps=6000 --batch-states=256 --hidden=128 --lr=0.001 --seed=42 --pool=6000
       --length=48 --hard-fraction=0.5 --eval-every=500 --log-every=100 --backend=<name> --checkpoint=<path>
       Trains the tool-ordering MLP for explore_sequence on basic and hard synthetic families.
explore: read one explore_sequence JSON object from stdin
       --explorer-checkpoint=<path> uses a trained ordering policy when no program checkpoint exists
       --program-checkpoint=<path> lets a Spectron policy drive the whole search; models/sequence_program.nimq
       is used when it exists. Each step names one tool, plus the order and degree a holonomic
       fit takes, for example [difference, P-recursive(r=2,d=1)]. The exact engine runs and
       checks that step before the next one is written, and the plain scan does not run.

train-program: --steps=2000 --batch-states=32 --pool=12000 --hidden=64 --heads=4 --layers=2
       --lr=0.001 --seed=42 --hard-fraction=0.8 --eval-size=200 --eval-seed=24263
       --backend=<name> --checkpoint=<path> --init-checkpoint=<path>
       Trains one tool decision at a time on routes that compute a withheld answer.
       The sampling mix gradually shifts toward more advanced verified routes.
       --init-checkpoint starts from saved weights with a fresh Adam optimizer.
eval-program: --checkpoint=<path> --eval-size=200 --eval-seed=24263 --max-tests=300 --backend=<name>
       Scores answers to the same holdout as train-program without updating weights.

train: --profile=standard|quality|research --tokens=1000000000 --warmup-tokens=<count> --batch-states=256
       --rollouts=4 --rl-problems=<count> --rollout-steps=96 --rl-every=4 --no-hard-mining
       --backend=<name> --hidden=128 --lr=0.001 --seed=42
       Training and inference use metal on macOS and vulkan elsewhere. The CPU backend is refused.
       --checkpoint=<path> --checkpoint-every=250 --log-every=100
       --eval-every=<steps> --eval-per-variant=<count> --eval-seed=114907
       Quality and research include periodic validation and a separate best checkpoint.
       Research trains a verified order search for finite-field recurrences.
eval:  --checkpoint=<path> --eval-per-variant=<count> --eval-seed=114907 --backend=<name>
       --research selects the default research checkpoint and its recurrence suite.
       --confirmation selects the separate confirmation suite.
       --constrained enables exact move constraints. --raw-policy scores without them.
       Quality defaults to 1,000 problems per variant (10,000 total).
data:  --samples=10 --seed=42 --research
calc:  pass a quoted exact expression, for example "(7+5)/8"
solve: read one JSON problem from stdin
       --checkpoint=<path> --backend=auto --rollout-steps=96 --policy-only | --exact
mcp:   serve exact math tools over stdio
       --checkpoint=<path> optionally uses a loaded policy for supported problems
       --backend=metal|vulkan --rollout-steps=96 --explorer-checkpoint=<path> --planner-checkpoint=<path>
       models/sequence_explorer.nimq orders explore_sequence when it exists.
"""

type
  Options = Table[string, string]
  Counters = object
    supervised, reinforced, sampled, groups, solved, attempts: int64
  PolicyBatch = object
    decisions: seq[Decision]
    sampled, solved: int
    effortGap: float64

proc parseOptions(args, values, flags: seq[string]): Options =
  if args.len == 0: return
  var parser = initOptParser(args, longNoVal = flags)
  while true:
    parser.next()
    case parser.kind
    of cmdEnd: return
    of cmdLongOption:
      if parser.key notin values and parser.key notin flags:
        raise newException(ValueError, "Unknown option --" & parser.key)
      if parser.key in result:
        raise newException(ValueError, "Duplicate option --" & parser.key)
      if parser.key in values and parser.val.len == 0:
        raise newException(ValueError, "Missing value for --" & parser.key)
      if parser.key in flags and parser.val.len > 0:
        raise newException(ValueError, "Flag --" & parser.key & " takes no value")
      result[parser.key] = parser.val
    else:
      raise newException(ValueError, "Unexpected argument " & parser.key)

proc number(opts: Options, name: string, fallback, minimum, maximum: int64): int64 =
  result = parseBiggestInt(opts.getOrDefault(name, $fallback).replace("_", ""))
  if result < minimum or result > maximum:
    raise newException(ValueError, &"--{name} must be between {minimum} and {maximum}")

proc countStates(tokens: int64): int64 =
  if tokens mod TokensPerState != 0:
    raise newException(ValueError, &"Token counts must be multiples of {TokensPerState}")
  tokens div TokensPerState

proc total(counters: Counters): int64 =
  counters.supervised + counters.reinforced

proc readCounters(node: JsonNode): Counters =
  result.supervised = node["sft_states"].getBiggestInt()
  result.reinforced = node["rlvr_states"].getBiggestInt()
  result.sampled = node["rollout_states"].getBiggestInt()
  result.groups = node["rlvr_groups"].getBiggestInt()
  result.solved = node["verified_rollouts"].getBiggestInt()
  result.attempts = node["attempted_rollouts"].getBiggestInt()
  if result.supervised < 0 or result.reinforced < 0 or result.reinforced > result.sampled or
      result.groups < 0 or result.solved < 0 or result.solved > result.attempts:
    raise newException(ValueError, "Invalid training counters in checkpoint")

proc supervisedBatch(count: int, rng: var Rand, repair: seq[MathProblem];
                     quality, research: bool, progress: float64): seq[Decision] =
  result = newSeqOfCap[Decision](count)
  var state: SolverState
  var needProblem = true
  var repairIndex = 0
  var plan: seq[Action]
  var nextAction = 0
  while result.len < count:
    if needProblem:
      if repairIndex < repair.len:
        state = newSolver(repair[repairIndex])
        inc repairIndex
      else:
        let difficulty = if quality: rng.trainingDifficulty(progress) else: sdNormal
        let example =
          if research:
            generate(rng, difficulty, research = true)
          elif quality and rng.rand(1.0) < 1.0 / 3.0:
            generate(rng, Variant(rng.rand(ord(layeredFlow) .. ord(blockedFlow))), difficulty)
          else:
            generate(rng, difficulty)
        state = newSolver(example.problem)
      needProblem = false
      plan.setLen(0)
      nextAction = 0
    if nextAction == plan.len:
      plan = if quality: state.constrainedExpertPlan() else: state.expertPlan()
      nextAction = 0
    let legal = state.legalActions()
    let allowed = if quality: state.constrainedActions(legal) else: legal
    let action = plan[nextAction]
    inc nextAction
    if action notin allowed:
      raise newException(ValueError, "Teacher action violated the training mask")
    let frame = if card(allowed) > 1: state.policyFrame(allowed, quality) else: default(MathFrame)
    result.add(Decision(frame: frame, allowed: allowed, action: action, weight: 1))
    state.advance(action)
    if state.progress != active:
      if not state.certified():
        raise newException(ValueError, "The exact teacher produced an invalid certificate")
      needProblem = true

proc reinforcementBatch(rollouts: openArray[Rollout], normalize: bool): PolicyBatch =
  var rewards = newSeq[float64](rollouts.len)
  var sum = 0.0
  var low = 2.0
  var high = 0.0
  var easiest = high(int)
  var hardest = 0
  for i, rollout in rollouts:
    result.sampled += rollout.decisions.len
    if rollout.verified:
      inc result.solved
      let effort = rollout.state.steps + rollout.state.work
      easiest = min(easiest, effort)
      hardest = max(hardest, effort)
      rewards[i] = 1.0 + 0.2 / (1.0 + float64(effort) / 16.0)
    sum += rewards[i]
    low = min(low, rewards[i])
    high = max(high, rewards[i])
  if result.solved == rollouts.len and hardest > easiest:
    result.effortGap = float64(hardest - easiest) / float64(max(easiest, 1))
  if high - low < 1e-8: return
  var scale = 1.0
  if normalize:
    let mean = sum / float64(rollouts.len)
    var squares = 0.0
    for reward in rewards: squares += (reward - mean) * (reward - mean)
    scale = 1.0 / max(1e-6, sqrt(squares / float64(rollouts.len)))
  result.decisions = newSeqOfCap[Decision](result.sampled)
  for i, rollout in rollouts:
    let baseline = (sum - rewards[i]) / float64(rollouts.len - 1)
    let advantage = float32((rewards[i] - baseline) * scale)
    for decision in rollout.decisions:
      var weighted = decision
      weighted.weight = advantage
      result.decisions.add(weighted)

proc trainingBackend(opts: Options): Backend =
  ## Training and inference always run on a GPU backend.
  let name = opts.getOrDefault("backend", DefaultTrainBackend)
  if name notin ["metal", "vulkan"]:
    raise newException(ValueError, "--backend must be metal or vulkan. Training and inference run on the GPU.")
  result = newBackend(name)
  if result.kind == bkMetal: setForbidCpuFallback(true)

proc validationSpec(opts: Options, saved: JsonNode; quality, research: bool): ValidationSpec =
  let savedSeed = saved{"seed"}.getBiggestInt(DefaultEvalSeed)
  let size =
    if research: saved{"examples_per_variant"}.getInt(128)
    elif quality: QualityEvalPerVariant
    else: saved{"examples_per_variant"}.getInt(DefaultEvalPerVariant)
  result.seed = opts.number("eval-seed", savedSeed, 0, high(int64))
  result.perVariant = int(opts.number("eval-per-variant", size, 1, MaxEvalPerVariant))
  result.research = research

proc scheduledRate(peak: float32, progress: float64): float32 =
  const warmup = 0.02
  if progress < warmup:
    return peak * float32(0.05 + 0.95 * progress / warmup)
  let phase = (progress - warmup) / (1.0 - warmup)
  peak * float32(0.1 + 0.45 * (1.0 + cos(PI * phase)))

proc train(opts: Options) =
  let profile = opts.getOrDefault("profile", "standard")
  if profile notin ["standard", "quality", "research"]:
    raise newException(ValueError, "--profile must be standard, quality, or research")
  let research = profile == "research"
  let quality = profile != "standard"
  let budget = countStates(opts.number("tokens", DefaultTokens, TokensPerState, high(int64)))
  let batch = int(opts.number("batch-states", (if quality: 512 else: 256), 1, 8192))
  let group = int(opts.number("rollouts", (if quality: 8 else: 4), 2, 32))
  let rlProblems = int(opts.number("rl-problems", (if quality: 4 else: 1), 1, 8))
  # Research rollouts need room for one Hankel test per candidate order.
  let limit = int(opts.number("rollout-steps", (if research: 128 elif quality: 128 else: DefaultRolloutSteps), 1, 1024))
  let rlEvery = int(opts.number("rl-every", (if quality: 2 else: 4), 0, high(int32)))
  let evalEvery = int(opts.number("eval-every", (if quality: 500 else: 0), 0, high(int32)))
  let checkpointEvery = int(opts.number("checkpoint-every", 250, 1, high(int32)))
  let logEvery = int(opts.number("log-every", 100, 1, high(int32)))
  let width = int(opts.number("hidden", (if quality: 384 else: 128), 1, MaxHidden))
  let lr = float32(parseFloat(opts.getOrDefault("lr", (if quality: "0.0007" else: "0.001"))))
  if lr notin 1e-7'f32 .. 0.1'f32:
    raise newException(ValueError, "--lr must be a finite number in [0.0000001, 0.1]")
  let path = opts.getOrDefault("checkpoint",
    (if research: ResearchCheckpoint elif quality: QualityCheckpoint else: DefaultCheckpoint))
  let bestPath = path.changeFileExt("best.nimq")
  let mining = not opts.hasKey("no-hard-mining")
  var seed = opts.number("seed", 42, 0, high(int64))
  let defaultWarmup = (budget div (if quality: 5 else: 10)) * TokensPerState
  var warmup = countStates(opts.number("warmup-tokens", defaultWarmup, 0, high(int64)))
  var scheduleEnd = budget
  var savedValidation: JsonNode
  var miner: HardMiner
  var counters: Counters
  var policy: MathPolicy
  var adam: PolicyAdam
  if fileExists(path):
    let loaded = math_policy.load(path, restoreOptimizer = true)
    let savedProfile = loaded.metadata{"profile"}.getStr()
    if loaded.policy.research != research or (savedProfile.len > 0 and savedProfile != profile):
      raise newException(ValueError, "Resume with --profile=" & savedProfile & " for this checkpoint")
    if opts.hasKey("hidden") and width != loaded.policy.hiddenWidth():
      raise newException(ValueError, "The requested width differs from the checkpoint")
    policy = loaded.policy
    adam = loaded.optimizer
    counters = readCounters(loaded.metadata)
    miner = loadMiner(loaded.metadata["miner"])
    let savedSeed = loaded.metadata["seed"].getBiggestInt()
    if opts.hasKey("seed") and seed != savedSeed:
      raise newException(ValueError, "Use the checkpoint seed to continue this stream")
    seed = savedSeed
    if not opts.hasKey("warmup-tokens"): warmup = loaded.metadata["warmup_states"].getBiggestInt()
    let savedScheduleEnd = loaded.metadata{"schedule_states"}.getBiggestInt(budget)
    if savedScheduleEnd <= 0: raise newException(ValueError, "Invalid checkpoint learning rate schedule")
    # A larger requested budget extends the cosine schedule. Keeping its old
    # endpoint would leave every resumed update at the finished learning rate.
    scheduleEnd = max(savedScheduleEnd, budget)
    if scheduleEnd > savedScheduleEnd:
      stderr.writeLine(&"[resume] learning-rate schedule extended to {scheduleEnd * TokensPerState} tokens")
    savedValidation = loaded.metadata{"validation"}
    stderr.writeLine(&"[resume] {counters.total * TokensPerState} tokens and {adam.steps} updates restored")
  else:
    randomize(seed)
    policy = newPolicy(width)
  if budget < counters.total:
    raise newException(ValueError, "Budget is below the existing checkpoint progress")
  if opts.hasKey("warmup-tokens") and warmup > budget:
    raise newException(ValueError, "Warmup cannot exceed the token budget")
  if budget == counters.total:
    echo &"[math] Done: {budget * TokensPerState} structural tokens"
    return

  policy.constrained = quality
  policy.research = research
  let ctx = trainingBackend(opts)
  defer: closeBackend(ctx)
  let spec = validationSpec(opts, savedValidation, quality, research)
  var suite: ValidationSuite
  var best = ValidationScore(solved: -1)
  if evalEvery > 0:
    suite = makeSuite(spec)
    if fileExists(bestPath) and not savedValidation.isNil:
      best = spec.restoreBest(savedValidation, policy.constrained)
      if best.solved < 0:
        var previous = math_policy.load(bestPath, restoreOptimizer = true)
        if previous.policy.research != research:
          raise newException(ValueError, "The best checkpoint belongs to a different profile")
        previous.policy.constrained = policy.constrained
        let measured = previous.policy.evaluate(suite, ctx)
        best = measured.score
        previous.metadata["validation"] = spec.record(best, policy.constrained)
        previous.policy.save(previous.optimizer, previous.metadata, bestPath)
        echo "[validation saved best] " & $measured.toJson(spec)
  echo &"[math] profile={profile} hidden={policy.hiddenWidth()} tokens={budget * TokensPerState} " &
       &"SFT warmup={warmup * TokensPerState} tokens, RL every={rlEvery}, hard mining={mining}, " &
       &"RL problems={rlProblems}, constrained={policy.constrained}"
  if evalEvery > 0 and adam.steps == 0:
    echo "[validation baseline] " & $policy.evaluate(suite, ctx).toJson(spec)

  template checkpoint(destination: string) =
    let metadata = %*{
      "profile": profile,
      "schedule_states": scheduleEnd,
      "seed": seed,
      "warmup_states": warmup,
      "sft_states": counters.supervised,
      "rlvr_states": counters.reinforced,
      "rollout_states": counters.sampled,
      "rlvr_groups": counters.groups,
      "verified_rollouts": counters.solved,
      "attempted_rollouts": counters.attempts,
      "miner": miner.toJson()
    }
    if evalEvery > 0: metadata["validation"] = spec.record(best, policy.constrained)
    policy.save(adam, metadata, destination)

  while counters.total < budget:
    var rng = initRand(seed xor int64(adam.steps + 1))
    let remaining = budget - counters.total
    let progress = min(1.0, float64(counters.total) / float64(scheduleEnd))
    let rate = if quality: scheduledRate(lr, progress) else: lr
    var loss: float64
    var rl = false
    var repair: seq[MathProblem]
    var learningStates = 0
    let groups = int(min(int64(rlProblems), remaining div int64(group * limit)))
    if rlEvery > 0 and counters.total >= warmup and adam.steps mod rlEvery == 0 and groups > 0:
      var examples = newSeqOfCap[Example](groups)
      var problems = newSeqOfCap[MathProblem](groups * group)
      for _ in 0 ..< groups:
        let difficulty = if quality: rng.trainingDifficulty(progress) else: sdNormal
        let example = miner.select(rng, mining, difficulty, preferMixed = quality, research = research)
        examples.add(example)
        for _ in 0 ..< group: problems.add(example.problem)
      let attempts = policy.runPolicies(problems, limit, ctx, rng)
      var decisions: seq[Decision]
      for i, example in examples:
        let first = i * group
        let rollouts = reinforcementBatch(attempts.toOpenArray(first, first + group - 1), normalize = quality)
        counters.sampled += rollouts.sampled
        counters.solved += rollouts.solved
        counters.attempts += group
        inc counters.groups
        if mining:
          miner.observe(example, rollouts.solved, group, (if quality: rollouts.effortGap else: 0.0))
        decisions.add(rollouts.decisions)
        if rollouts.solved == 0: repair.add(example.problem)
      learningStates = learnableStates(decisions)
      if learningStates > 0:
        let normalizer = if quality: learningStates else: groups * group * limit
        loss = policy.update(adam, decisions, ctx, rate * 0.3'f32, normalizer)
        counters.reinforced += decisions.len
        rl = true
    if not rl:
      let beforeWarmup = if counters.total < warmup: warmup - counters.total else: remaining
      let count = int(min(int64(batch), min(remaining, beforeWarmup)))
      let examples = supervisedBatch(count, rng, repair, quality, research, progress)
      learningStates = learnableStates(examples)
      let normalizer = if quality: max(learningStates, 1) else: count
      loss = policy.update(adam, examples, ctx, rate, normalizer)
      counters.supervised += count
    let done = counters.total == budget
    if adam.steps mod logEvery == 0 or done:
      let stage = if rl: "RLVR" else: "SFT"
      echo &"[{stage}] step={adam.steps} tokens={counters.total * TokensPerState}/{budget * TokensPerState} " &
        &"sft_states={counters.supervised} rlvr_states={counters.reinforced} " &
        &"rollout_states={counters.sampled} verified={counters.solved}/{counters.attempts} " &
        &"mined_variants={miner.mutations} learning_states={learningStates} loss={loss:.5f} lr={rate:.7f}"
    let validateNow = evalEvery > 0 and (adam.steps mod evalEvery == 0 or done)
    if validateNow:
      let measured = policy.evaluate(suite, ctx)
      echo &"[validation step={adam.steps}] " & $measured.toJson(spec)
      if measured.score.betterThan(best):
        best = measured.score
        checkpoint(bestPath)
        echo &"[math] Saved best policy at {bestPath}"
    if adam.steps mod checkpointEvery == 0 or done or validateNow:
      checkpoint(path)
  echo &"[math] Saved {path}"
  if quality and evalEvery > 0:
    let confirmation = spec.confirmationSpec()
    let selected = math_policy.load(bestPath, restoreOptimizer = false).policy
    let measured = selected.evaluate(makeSuite(confirmation, excluding = suite), ctx)
    echo "[confirmation] " & $measured.toJson(confirmation)
    echo &"[target] validation={best.solved}/{spec.size()} confirmation={measured.score.solved}/{confirmation.size()} " &
         &"perfect={best.solved == spec.size() and measured.score.solved == confirmation.size()}"

proc trainExplorer(opts: Options) =
  let steps = int(opts.number("steps", 6_000, 1, high(int32)))
  let batch = int(opts.number("batch-states", 256, 1, 8192))
  let hidden = int(opts.number("hidden", 128, 1, MaxExplorerHidden))
  let poolSize = int(opts.number("pool", 6_000, 16, 1_000_000))
  let length = int(opts.number("length", 48, MinExplorerTerms, MaxExplorerTerms))
  let hardFraction = parseFloat(opts.getOrDefault("hard-fraction", "0.5"))
  if hardFraction notin 0.0 .. 1.0: raise newException(ValueError, "--hard-fraction must be in [0, 1]")
  let evalEvery = int(opts.number("eval-every", 500, 0, high(int32)))
  let logEvery = int(opts.number("log-every", 100, 1, high(int32)))
  let seed = opts.number("seed", 42, 0, high(int64))
  let lr = float32(parseFloat(opts.getOrDefault("lr", "0.001")))
  if lr notin 1e-7'f32 .. 0.1'f32: raise newException(ValueError, "--lr must be in [0.0000001, 0.1]")
  let path = opts.getOrDefault("checkpoint", ExplorerCheckpoint)
  var policy: ExplorerPolicy
  var adam: ExplorerAdam
  if fileExists(path):
    let loaded = loadExplorerPolicy(path)
    policy = loaded.policy
    adam.steps = 0
    stderr.writeLine(&"[resume] explorer policy with {loaded.steps} updates, fresh Adam moments")
  else:
    randomize(seed)
    policy = newExplorerPolicy(hidden)
  let ctx = trainingBackend(opts)
  defer: closeBackend(ctx)
  var rng = initRand(seed)
  var pool: seq[ExplorerSample]
  var families = initCountTable[string]()
  var unsolved = initCountTable[string]()
  for _ in 0 ..< poolSize:
    let example = generateExample(rng, length, hardFraction)
    let samples = teacherSamples(example)
    if samples.len > 0: families.inc(example.family) else: unsolved.inc(example.family)
    pool.add(samples)
  if pool.len == 0: raise newException(ValueError, "The teacher produced no training decisions")
  var evalRng = initRand(seed xor 0x5EED)
  var basicSet, hardSet: seq[SequenceExample]
  for _ in 0 ..< 300: basicSet.add(generateExample(evalRng, length))
  for family in 0 .. HardFamilies.high:
    for _ in 0 ..< 40: hardSet.add(generateHard(evalRng, length, family))
  let basicCases = evalCases(basicSet)
  let hardCases = evalCases(hardSet)
  echo &"[explorer] hidden={policy.hiddenWidth()} pool_decisions={pool.len} families={families}"
  echo &"[explorer] teacher found no strong direction for {unsolved}"
  proc evaluateBoth(label: string) =
    echo &"[explorer {label} basic] " & policy.evaluate(basicCases, ctx).report()
    let hard = policy.evaluate(hardCases, ctx)
    echo &"[explorer {label} hard] " & hard.report()
    echo &"[explorer {label} hard by family default/guided/total] " & hard.familyReport()
  evaluateBoth("baseline")
  for step in 1 .. steps:
    var minibatch = newSeq[ExplorerSample](min(batch, pool.len))
    for sample in minibatch.mitems: sample = pool[rng.rand(pool.high)]
    let progress = float64(step - 1) / float64(steps)
    let rate = lr * float32(0.1 + 0.45 * (1.0 + cos(PI * progress)))
    let loss = policy.update(adam, minibatch, ctx, rate)
    if step mod logEvery == 0 or step == steps:
      echo &"[explorer] step={step}/{steps} loss={loss:.5f} lr={rate:.6f}"
    if (evalEvery > 0 and step mod evalEvery == 0) or step == steps:
      evaluateBoth(&"eval step={step}")
      policy.save(step, path)
  echo &"[explorer] Saved {path}"

var explorerContext: Backend

proc installExplorerGuide(opts: Options) =
  let path = opts.getOrDefault("explorer-checkpoint", ExplorerCheckpoint)
  if not fileExists(path):
    if opts.hasKey("explorer-checkpoint"): raise newException(ValueError, "Checkpoint file not found: " & path)
    return
  let loaded = loadExplorerPolicy(path)
  explorerContext = trainingBackend(opts)
  explorerGuide = loaded.policy.guide(explorerContext)

proc installPlannerPolicy(opts: Options) =
  let path = opts.getOrDefault("planner-checkpoint", PlannerCheckpoint)
  if not fileExists(path):
    if opts.hasKey("planner-checkpoint"): raise newException(ValueError, "Checkpoint file not found: " & path)
    return
  installPlanner(path, trainingBackend(opts))

proc installProgramPolicy(opts: Options) =
  let path = opts.getOrDefault("program-checkpoint", ProgramCheckpoint)
  if not fileExists(path):
    if opts.hasKey("program-checkpoint"): raise newException(ValueError, "Checkpoint file not found: " & path)
    return
  discard installProgram(path, trainingBackend(opts))

proc trainProgramCommand(opts: Options) =
  ## Train the Spectron policy that writes tool programs for explore_sequence.
  let steps = int(opts.number("steps", 2_000, 1, high(int32)))
  let batch = int(opts.number("batch-states", 32, 1, 4096))
  let poolSize = int(opts.number("pool", 12_000, 8, 200_000))
  let initPath = opts.getOrDefault("init-checkpoint")
  let initial = if initPath.len > 0: loadProgram(initPath) else: ProgramPolicy()
  let shape = if initPath.len > 0:
    (hidden: initial.model.embed.dim, heads: initial.model.blocks[0].attn.nHeads,
     layers: initial.model.blocks.len)
  else:
    (hidden: 64, heads: 4, layers: 2)
  let hidden = int(opts.number("hidden", shape.hidden, 1, 1024))
  let heads = int(opts.number("heads", shape.heads, 1, 64))
  let layers = int(opts.number("layers", shape.layers, 1, 32))
  let seed = int(opts.number("seed", 42, 0, high(int64)))
  let lr = float32(parseFloat(opts.getOrDefault("lr", "0.001")))
  let hardFraction = parseFloat(opts.getOrDefault("hard-fraction", "0.8"))
  let evalSize = int(opts.number("eval-size", 200, 8, 20_000))
  let evalSeed = int(opts.number("eval-seed", DefaultProgramEvalSeed, 0, high(int64)))
  if not (0.0 <= hardFraction and hardFraction <= 1.0):
    raise newException(ValueError, "--hard-fraction must be between 0 and 1")
  if hidden mod heads != 0: raise newException(ValueError, "--hidden must be divisible by --heads")
  if initPath.len > 0 and (hidden != shape.hidden or heads != shape.heads or layers != shape.layers):
    raise newException(ValueError, "--hidden, --heads and --layers must match --init-checkpoint")
  let outputPath = if initPath.len > 0: changeFileExt(initPath, "trained.nimq") else: ProgramCheckpoint
  let path = opts.getOrDefault("checkpoint", outputPath)
  let ctx = trainingBackend(opts)
  defer: closeBackend(ctx)
  discard trainProgram(path, steps, batch, poolSize, seed, hidden, heads, layers, lr, ctx,
    hardFraction, evalSize, evalSeed, initial)

proc evalProgramCommand(opts: Options) =
  let path = opts.getOrDefault("checkpoint", ProgramCheckpoint)
  let size = int(opts.number("eval-size", 200, 1, 20_000))
  let seed = int(opts.number("eval-seed", DefaultProgramEvalSeed, 0, high(int64)))
  let budget = int(opts.number("max-tests", EvalBudget, 1, 5_000))
  let policy = loadProgram(path)
  let ctx = trainingBackend(opts)
  defer: closeBackend(ctx)
  evaluateProgram(policy, ctx, size, seed, budget)

proc trainPlanner(opts: Options) =
  ## Supervised warm start from a lookahead teacher, then RL on verified answers.
  let sftSteps = int(opts.number("sft-steps", 1_500, 0, high(int32)))
  let rlIterations = int(opts.number("rl-iterations", 300, 0, high(int32)))
  let batch = int(opts.number("batch-states", 128, 1, 8192))
  let hidden = int(opts.number("hidden", 128, 1, 1024))
  let poolSize = int(opts.number("pool", 600, 8, 100_000))
  let evalSize = int(opts.number("eval-tasks", 200, 8, 100_000))
  let group = int(opts.number("group", 4, 2, 64))
  let rlTasks = int(opts.number("rl-tasks", 8, 1, 1024))
  let evalEvery = int(opts.number("eval-every", 50, 1, high(int32)))
  let seed = opts.number("seed", 42, 0, high(int64))
  let lr = float32(parseFloat(opts.getOrDefault("lr", "0.001")))
  let rlLr = float32(parseFloat(opts.getOrDefault("rl-lr", "0.0003")))
  let hardFraction = parseFloat(opts.getOrDefault("hard-fraction", "0.5"))
  let farFraction = parseFloat(opts.getOrDefault("far-fraction", "0.4"))
  let path = opts.getOrDefault("checkpoint", PlannerCheckpoint)
  installExplorerGuide(opts)
  randomize(seed)
  var policy = newPlannerPolicy(hidden)
  var adam: PlannerAdam
  let ctx = trainingBackend(opts)
  defer: closeBackend(ctx)
  var rng = initRand(seed)
  var pool, evalTasks: seq[PlannerTask]
  for _ in 0 ..< poolSize: pool.add(generateTask(rng, hardFraction, farFraction))
  var evalRng = initRand(seed xor 0x7A5C)
  for _ in 0 ..< evalSize: evalTasks.add(generateTask(evalRng, hardFraction, farFraction))
  echo &"[planner] pool={pool.len} eval={evalTasks.len} hidden={hidden} far_fraction={farFraction}"
  echo "[planner eval rules] " & $score(evalTasks, PlannerChooser(rulePlanner))
  echo "[planner eval lookahead teacher] " & $score(evalTasks, PlannerChooser(expertPlanner))
  var supervised: seq[PlannerSample]
  for task in pool.mitems: discard task.runPlanner(PlannerChooser(expertPlanner), addr supervised)
  echo &"[planner] teacher decisions={supervised.len}"
  for step in 1 .. sftSteps:
    var minibatch = newSeq[PlannerSample](min(batch, supervised.len))
    for sample in minibatch.mitems: sample = supervised[rng.rand(supervised.high)]
    let rate = lr * float32(0.1 + 0.45 * (1.0 + cos(PI * float64(step - 1) / float64(sftSteps))))
    let loss = policy.update(adam, minibatch, ctx, rate)
    if step mod 250 == 0 or step == sftSteps: echo &"[planner sft] step={step}/{sftSteps} loss={loss:.5f}"
  var best = score(evalTasks, policy.greedy(ctx))
  echo "[planner eval after sft] " & $best
  policy.savePlanner(sftSteps, path)
  var rolloutRng = initRand(seed xor 0x51)
  for iteration in 1 .. rlIterations:
    var samples: seq[PlannerSample]
    var rewardSum = 0.0
    for _ in 0 ..< rlTasks:
      let taskIndex = rng.rand(pool.high)
      var rewards: seq[float64]
      var records: seq[seq[PlannerSample]]
      for _ in 0 ..< group:
        var record: seq[PlannerSample]
        let state = pool[taskIndex].runPlanner(policy.sampler(ctx, addr rolloutRng), addr record)
        rewards.add(pool[taskIndex].reward(state))
        records.add(record)
      let mean = rewards.sum() / float64(rewards.len)
      rewardSum += mean
      for g in 0 ..< group:
        let advantage = rewards[g] - mean
        if abs(advantage) < 1e-9: continue
        for sample in records[g].mitems:
          sample.weight = float32(advantage)
          samples.add(sample)
    if samples.len > 0: discard policy.update(adam, samples, ctx, rlLr)
    if iteration mod 25 == 0:
      echo &"[planner rl] iteration={iteration}/{rlIterations} mean_rollout_reward={rewardSum / float64(rlTasks):.3f}"
    if iteration mod evalEvery == 0 or iteration == rlIterations:
      let measured = score(evalTasks, policy.greedy(ctx))
      echo &"[planner eval rl iteration={iteration}] " & $measured
      if measured.rewardTotal > best.rewardTotal:
        best = measured
        policy.savePlanner(sftSteps + iteration, path)
        echo &"[planner] saved best {path}"
  echo "[planner best] " & $best

proc evalPolicy(opts: Options) =
  let research = opts.hasKey("research")
  let defaultPath = if research: ResearchCheckpoint else: QualityCheckpoint
  let path = opts.getOrDefault("checkpoint", defaultPath.changeFileExt("best.nimq"))
  if not fileExists(path):
    raise newException(ValueError, "Checkpoint file not found: " & path)
  let loaded = math_policy.load(path, restoreOptimizer = false)
  if research and not loaded.policy.research:
    raise newException(ValueError, "--research requires a research checkpoint")
  let spec = validationSpec(opts, loaded.metadata{"validation"},
    loaded.metadata{"profile"}.getStr() == "quality", loaded.policy.research)
  let ctx = trainingBackend(opts)
  defer: closeBackend(ctx)
  var suite = makeSuite(spec)
  if opts.hasKey("confirmation"):
    suite = makeSuite(spec.confirmationSpec(), excluding = suite)
  var policy = loaded.policy
  if opts.hasKey("constrained") and opts.hasKey("raw-policy"):
    raise newException(ValueError, "Choose either --constrained or --raw-policy")
  if opts.hasKey("constrained"): policy.constrained = true
  let evaluated = policy.evaluate(suite, ctx, rawActions = opts.hasKey("raw-policy"))
  echo pretty(evaluated.toJson(suite.spec))

proc data(opts: Options) =
  let samples = int(opts.number("samples", 10, 1, high(int32)))
  var rng = initRand(opts.number("seed", 42, 0, high(int64)))
  for _ in 0 ..< samples:
    let example = generate(rng, research = opts.hasKey("research"))
    let solved = runExpert(example.problem)
    echo $(%*{"variant": $example.variant, "problem": problemJson(example.problem),
             "result": resultJson(solved, "exact")})

proc policyResult(problem: JsonNode, policy: MathPolicy, ctx: Backend, limit: int;
                  allowGuidance = true): JsonNode =
  if (problem{"kind"}.getStr() in ["recurrence_term", "weighted_walk_count"]) != policy.research:
    return nil
  var rng = initRand(42)
  let input = parseProblem(problem)
  let rollout = policy.runPolicy(input, 1, limit, ctx, rng,
                                 greedy = true, recordDecisions = false, recordMethod = true)[0]
  if rollout.verified:
    let methodName = if policy.research: "policy" elif policy.constrained: "policy_constrained" else: "policy"
    return resultJson(rollout.state, methodName)
  if allowGuidance and not policy.research:
    let guided = policy.runPolicy(input, 1, limit, ctx, rng, greedy = true,
                                  recordDecisions = false, recordMethod = true, guided = true)[0]
    if guided.verified: return resultJson(guided.state, "policy_guided")

proc rolloutLimit(opts: Options): int =
  int(opts.number("rollout-steps", DefaultRolloutSteps, 1, 1024))

proc solve(opts: Options) =
  installExplorerGuide(opts)
  installProgramPolicy(opts)
  installPlannerPolicy(opts)
  let problem = parseJson(stdin.readAll())
  let path = opts.getOrDefault("checkpoint", DefaultCheckpoint)
  let exact = opts.hasKey("exact")
  let policyOnly = opts.hasKey("policy-only")
  if exact and policyOnly:
    raise newException(ValueError, "--exact and --policy-only cannot be combined")
  if policyOnly and not problem.canUsePolicy():
    raise newException(ValueError, "The policy does not cover this problem kind or size")
  if not exact and (policyOnly or opts.hasKey("checkpoint")) and not fileExists(path):
    raise newException(ValueError, "Checkpoint file not found: " & path)
  if not exact and problem.canUsePolicy() and fileExists(path):
    let policy = math_policy.load(path, restoreOptimizer = false).policy
    let ctx = trainingBackend(opts)
    defer: closeBackend(ctx)
    let solved = policyResult(problem, policy, ctx, opts.rolloutLimit(), allowGuidance = not policyOnly)
    if not solved.isNil:
      echo pretty(solved)
      return
    if policyOnly:
      raise newException(ValueError, "The policy did not produce a verified certificate")
    let fallback = solveExact(problem)
    fallback["method"] = %"exact_fallback"
    echo pretty(fallback)
  else:
    echo pretty(solveExact(problem))

proc mcp(opts: Options) =
  installExplorerGuide(opts)
  installProgramPolicy(opts)
  installPlannerPolicy(opts)
  if not opts.hasKey("checkpoint"):
    serveMcp(solveExact)
    return
  let policy = math_policy.load(opts["checkpoint"], restoreOptimizer = false).policy
  let ctx = trainingBackend(opts)
  defer: closeBackend(ctx)
  let limit = opts.rolloutLimit()
  serveMcp(proc(problem: JsonNode): JsonNode =
    if problem.canUsePolicy():
      let solved = policyResult(problem, policy, ctx, limit)
      if not solved.isNil: return solved
      result = solveExact(problem)
      result["method"] = %"exact_fallback"
    else:
      result = solveExact(problem))

proc main() =
  let args = commandLineParams()
  if args.len == 0 or args[0] in ["help", "--help", "-h"]:
    echo Usage
    return
  let options = args[1 ..< args.len]
  case args[0]
  of "train":
    train(parseOptions(options, @["tokens", "warmup-tokens", "batch-states", "rollouts",
      "rollout-steps", "rl-problems", "rl-every", "backend", "hidden", "lr", "seed", "checkpoint",
      "checkpoint-every", "log-every", "profile", "eval-every", "eval-per-variant", "eval-seed"],
      @["no-hard-mining"]))
  of "train-explorer":
    trainExplorer(parseOptions(options, @["steps", "batch-states", "hidden", "lr", "seed", "pool", "length",
      "hard-fraction", "eval-every", "log-every", "backend", "checkpoint"], @[]))
  of "explore":
    let opts = parseOptions(options, @["explorer-checkpoint", "program-checkpoint", "backend"], @[])
    installExplorerGuide(opts)
    installProgramPolicy(opts)
    echo pretty(exploreSequence(parseJson(stdin.readAll())))
  of "train-program":
    trainProgramCommand(parseOptions(options, @["steps", "batch-states", "pool", "hidden", "heads", "layers",
      "lr", "seed", "hard-fraction", "eval-size", "eval-seed", "backend", "checkpoint", "init-checkpoint"], @[]))
  of "eval-program":
    evalProgramCommand(parseOptions(options, @["checkpoint", "eval-size", "eval-seed", "max-tests", "backend"], @[]))
  of "train-planner":
    trainPlanner(parseOptions(options, @["sft-steps", "rl-iterations", "batch-states", "hidden", "pool",
      "eval-tasks", "group", "rl-tasks", "eval-every", "seed", "lr", "rl-lr", "hard-fraction", "far-fraction",
      "backend", "checkpoint", "explorer-checkpoint"], @[]))
  of "research":
    let opts = parseOptions(options, @["explorer-checkpoint", "planner-checkpoint", "program-checkpoint",
      "backend"], @[])
    installExplorerGuide(opts)
    installProgramPolicy(opts)
    installPlannerPolicy(opts)
    echo pretty(researchSequence(parseJson(stdin.readAll())))
  of "eval":
    evalPolicy(parseOptions(options, @["checkpoint", "backend", "eval-per-variant", "eval-seed"],
      @["confirmation", "constrained", "raw-policy", "research"]))
  of "data":
    data(parseOptions(options, @["samples", "seed"], @["research"]))
  of "calc":
    if options.len != 1: raise newException(ValueError, "Pass one quoted expression")
    echo pretty(solveExact(%*{"kind": "calculate", "expression": options[0]}))
  of "solve":
    solve(parseOptions(options, @["checkpoint", "backend", "rollout-steps", "explorer-checkpoint",
      "program-checkpoint", "planner-checkpoint"], @["exact", "policy-only"]))
  of "mcp":
    mcp(parseOptions(options, @["checkpoint", "backend", "rollout-steps", "explorer-checkpoint",
      "program-checkpoint", "planner-checkpoint"], @[]))
  else:
    raise newException(ValueError,
      "Expected train, train-explorer, train-planner, train-program, eval-program, eval, data, calc, solve, explore, research, or mcp")

when isMainModule:
  try:
    main()
  except CatchableError as error:
    stderr.writeLine("[math] " & error.msg)
    quit(1)
