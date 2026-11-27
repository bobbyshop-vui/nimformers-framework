## Fixed, certified validation problems for greedy policy evaluation.
import std/[json, random, sets]
import backend
import discrete_math, math_policy, prepare_dataset

const
  DefaultEvalSeed* = 114_907'i64
  DefaultEvalPerVariant* = 24
  QualityEvalPerVariant* = 1_000
  MaxEvalPerVariant* = QualityEvalPerVariant
  ValidationActions = 128
  ValidationBatchSize = 192
  ValidationVersion = 5
  ConfirmationSeedOffset* = 0x48D1A24B'i64

type
  ValidationSpec* = object
    seed*: int64
    perVariant*: int
    research*: bool
  ValidationSuite* = object
    spec*: ValidationSpec
    problems: seq[MathProblem]
  ValidationScore* = object
    solved*, worst*: int
    actions*: int64
  ValidationResult* = object
    constrained*: bool
    score*: ValidationScore
    solvedByVariant*: array[Variant, int]

proc variants(spec: ValidationSpec): seq[Variant] =
  variants(spec.research)

proc variantCount(spec: ValidationSpec): int =
  spec.variants().len

proc size*(spec: ValidationSpec): int =
  spec.perVariant * spec.variantCount()

proc confirmationSpec*(spec: ValidationSpec): ValidationSpec =
  result = spec
  result.seed = spec.seed xor ConfirmationSeedOffset

proc makeSuite*(spec: ValidationSpec; excluding: ValidationSuite = default(ValidationSuite)): ValidationSuite =
  if spec.perVariant notin 1 .. MaxEvalPerVariant:
    raise newException(ValueError, "Invalid validation size per variant")
  result.spec = spec
  result.problems = newSeqOfCap[MathProblem](spec.size())
  var seen = initHashSet[string](spec.size() + excluding.problems.len)
  for problem in excluding.problems: seen.incl($problemJson(problem))
  var rng = initRand(spec.seed)
  for variant in spec.variants():
    let target = result.problems.len + spec.perVariant
    for _ in 0 ..< 10 * spec.perVariant:
      if result.problems.len == target: break
      let sample = generate(rng, variant, sdHoldout).problem
      if seen.containsOrIncl($problemJson(sample)): continue
      if not runExpert(sample).certified():
        raise newException(ValueError, "Invalid validation problem from " & $variant)
      result.problems.add(sample)
    if result.problems.len != target:
      raise newException(ValueError, "Unable to draw enough distinct validation problems for " & $variant)

proc evaluate*(policy: MathPolicy, suite: ValidationSuite, ctx: Backend; rawActions = false): ValidationResult =
  ## One greedy attempt per problem, with zero credit for an unverified trajectory.
  result.constrained = policy.constrained and not rawActions
  let variants = suite.spec.variants()
  var rng = initRand(0)
  for first in countup(0, suite.problems.high, ValidationBatchSize):
    let last = min(first + ValidationBatchSize - 1, suite.problems.high)
    let outcomes = policy.runPolicies(suite.problems.toOpenArray(first, last), ValidationActions,
                                      ctx, rng, greedy = true, recordDecisions = false, rawActions = rawActions)
    for offset, outcome in outcomes:
      if outcome.verified:
        let variant = variants[(first + offset) div suite.spec.perVariant]
        inc result.solvedByVariant[variant]
        inc result.score.solved
        result.score.actions += outcome.state.steps
  result.score.worst = suite.spec.perVariant
  for variant in variants:
    result.score.worst = min(result.score.worst, result.solvedByVariant[variant])

proc betterThan*(candidate, best: ValidationScore): bool =
  if candidate.solved != best.solved: candidate.solved > best.solved
  elif candidate.worst != best.worst: candidate.worst > best.worst
  else: candidate.actions < best.actions

proc toJson*(evaluation: ValidationResult, spec: ValidationSpec): JsonNode =
  var byVariant = newJObject()
  for variant in spec.variants():
    byVariant[$variant] = %*{
      "solved": evaluation.solvedByVariant[variant],
      "total": spec.perVariant
    }
  %*{
    "seed": spec.seed,
    "suite": (if spec.research: "research_recurrence" else: "base"),
    "policy_mode": (if spec.research: "verified_order_search" elif evaluation.constrained: "constrained" else: "raw"),
    "solved": evaluation.score.solved,
    "total": spec.size(),
    "pass_at_1": float64(evaluation.score.solved) / float64(spec.size()),
    "perfect": evaluation.score.solved == spec.size(),
    "worst_variant_solved": evaluation.score.worst,
    "examples_per_variant": spec.perVariant,
    "verified_actions": evaluation.score.actions,
    "by_variant": byVariant
  }

proc record*(spec: ValidationSpec, best: ValidationScore, constrained: bool): JsonNode =
  %*{"version": ValidationVersion, "seed": spec.seed, "examples_per_variant": spec.perVariant,
     "constrained": constrained, "max_actions": ValidationActions, "research": spec.research,
     "variant_count": spec.variantCount(),
     "best_solved": best.solved, "best_worst": best.worst,
     "best_actions": best.actions}

proc restoreBest*(spec: ValidationSpec, node: JsonNode, constrained: bool): ValidationScore =
  result.solved = -1
  if node.isNil or node.kind != JObject or
      node{"version"}.getInt() != ValidationVersion or
      node{"seed"}.getBiggestInt(-1) != spec.seed or
      node{"examples_per_variant"}.getInt() != spec.perVariant or
      node{"research"}.getBool() != spec.research or
      node{"variant_count"}.getInt() != spec.variantCount() or
      node{"max_actions"}.getInt() != ValidationActions or
      node{"constrained"}.getBool() != constrained:
    return
  let score = ValidationScore(solved: node{"best_solved"}.getInt(-1),
    worst: node{"best_worst"}.getInt(-1), actions: node{"best_actions"}.getBiggestInt(-1))
  if score.solved in 0 .. spec.size() and score.worst in 0 .. spec.perVariant and score.actions >= 0:
    result = score
