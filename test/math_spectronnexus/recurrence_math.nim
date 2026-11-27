## Exact recurrence reconstruction and large-index evaluation over F_p.
import std/json
import big_integer, polynomial_math, prime_field

const
  MaxRecurrenceOrder* = 512
  MaxRecurrenceSamples* = 4_096
  MaxIndexDigits* = 100_000
  MaxWalkOrder* = 128

type
  RecurrenceProblem* = object
    prime*, orderBound*: int
    index*: string
    samples*, coefficients*: seq[int]
  RecurrenceFit* = object
    ## singular: the order x order Hankel system has no unique solution.
    valid*, singular*: bool
    coefficients*: seq[int]
  WalkMatrix* = seq[seq[int]]
  WalkProblem* = object
    matrix*: WalkMatrix
    prime*, source*, sink*: int
    length*: string

proc readIndex(node: JsonNode): string =
  if node.isNil or node.kind notin {JInt, JString}:
    raise newException(ValueError, "index must be a nonnegative integer or decimal string")
  result = if node.kind == JInt: $node.getBiggestInt() else: node.getStr()
  if result.len notin 1 .. MaxIndexDigits:
    raise newException(ValueError, "Invalid decimal index length")
  for digit in result:
    if digit notin {'0'..'9'}: raise newException(ValueError, "index must be nonnegative")
  var first = 0
  while first + 1 < result.len and result[first] == '0': inc first
  result = result[first .. ^1]

proc readResidues(node: JsonNode, p, limit: int, label: string): seq[int] =
  if node.isNil or node.kind != JArray or node.len notin 1 .. limit:
    raise newException(ValueError, "Invalid number of " & label)
  for value in node: result.add(residue(integer(value, label & " entry"), p))

proc parseRecurrence*(node: JsonNode, limit = MaxRecurrenceOrder): RecurrenceProblem =
  result.prime = readPrime(node)
  result.index = readIndex(node{"index"})
  if node.hasKey("coefficients"):
    if node.hasKey("samples") or node.hasKey("order_bound"):
      raise newException(ValueError, "Choose coefficients with initial, or samples with order_bound")
    result.coefficients = readResidues(node["coefficients"], result.prime, limit, "coefficients")
    result.samples = readResidues(node{"initial"}, result.prime, limit, "initial values")
    if result.samples.len != result.coefficients.len:
      raise newException(ValueError, "Supply one initial value per coefficient")
  else:
    if node.hasKey("initial"):
      raise newException(ValueError, "initial requires coefficients")
    let bound = integer(node{"order_bound"}, "order_bound")
    if bound notin 1'i64 .. int64(limit): raise newException(ValueError, "order_bound is out of range")
    result.orderBound = int(bound)
    result.samples = readResidues(node{"samples"}, result.prime, MaxRecurrenceSamples, "samples")
    if result.samples.len < 2 * result.orderBound:
      raise newException(ValueError, "At least twice order_bound samples are required")

proc recurrenceJson*(problem: RecurrenceProblem): JsonNode =
  if problem.coefficients.len > 0:
    %*{"kind": "recurrence_term", "prime": problem.prime, "index": problem.index,
       "coefficients": problem.coefficients, "initial": problem.samples}
  else:
    %*{"kind": "recurrence_term", "prime": problem.prime, "index": problem.index,
       "order_bound": problem.orderBound, "samples": problem.samples}

proc minimalRecurrence*(samples: openArray[int], p: int): seq[int] =
  ## Berlekamp–Massey. Return c with a[n] = sum c[j]*a[n-1-j].
  var current = @[1]
  var previous = @[1]
  var order = 0
  var shift = 1
  var previousDiscrepancy = 1
  for i in 0 ..< samples.len:
    var discrepancy = int64(samples[i])
    for j in 1 .. order: discrepancy += int64(current[j]) * int64(samples[i - j])
    let d = residue(discrepancy, p)
    if d == 0:
      inc shift
      continue
    let old = current
    let multiplier = productMod(d, reciprocal(previousDiscrepancy, p), p)
    if current.len < previous.len + shift: current.setLen(previous.len + shift)
    for j, term in previous:
      current[j + shift] = subtractProduct(current[j + shift], multiplier, term, p)
    if 2 * order <= i:
      order = i + 1 - order
      previous = old
      previousDiscrepancy = d
      shift = 1
    else:
      inc shift
  for j in 1 .. order: result.add(residue(-int64(current[j]), p))

proc matches(samples: openArray[int], coefficients: openArray[int], p: int): bool =
  for i in coefficients.len ..< samples.len:
    var total = 0'i64
    for j, value in coefficients: total += int64(value) * int64(samples[i - 1 - j])
    if residue(total, p) != samples[i]: return false
  true

proc fitOrder*(problem: RecurrenceProblem, order: int): RecurrenceFit =
  ## Fit the policy's proposed order using Hankel equations.
  if order notin 0 .. problem.orderBound: return
  let p = problem.prime
  let samples = problem.samples
  if order == 0:
    result.valid = matches(samples, [], p)
    return
  var rows = newSeq[seq[int]](order)
  for r in 0 ..< order:
    rows[r] = newSeq[int](order + 1)
    for c in 0 ..< order: rows[r][c] = samples[order + r - 1 - c]
    rows[r][order] = samples[order + r]
  for c in 0 ..< order:
    var pivot = c
    while pivot < order and rows[pivot][c] == 0: inc pivot
    if pivot == order:
      result.singular = true
      return
    if pivot != c: swap(rows[pivot], rows[c])
    let inverse = reciprocal(rows[c][c], p)
    for j in c .. order: rows[c][j] = productMod(rows[c][j], inverse, p)
    for r in 0 ..< order:
      if r == c: continue
      let factor = rows[r][c]
      if factor == 0: continue
      for j in c .. order: rows[r][j] = subtractProduct(rows[r][j], factor, rows[c][j], p)
  for r in 0 ..< order: result.coefficients.add(rows[r][order])
  result.valid = matches(samples, result.coefficients, p)

proc verifies*(problem: RecurrenceProblem, fit: RecurrenceFit): bool =
  fit.valid and fit.coefficients.len <= problem.orderBound and
    fit.coefficients == minimalRecurrence(problem.samples, problem.prime)

proc binaryDigits(index: string): seq[int] =
  ## Bits of a decimal index, least significant first. Peeling 30 bits per long division
  ## keeps a 100,000 digit index cheap.
  var rest = parseBig(index)
  while not rest.isZero:
    let (quotient, chunk) = divSmall(rest, 1'i64 shl 30)
    rest = quotient
    for bit in 0 ..< 30:
      if rest.isZero and (chunk shr bit) == 0: break
      result.add(int((chunk shr bit) and 1))

proc characteristic(coefficients: seq[int], p: int): Polynomial =
  result = newSeq[int](coefficients.len + 1)
  result[^1] = 1
  for i, coefficient in coefficients:
    result[coefficients.len - i - 1] = residue(-int64(coefficient), p)

proc termWeights(coefficients: seq[int], index: string, p: int): Polynomial =
  if coefficients.len == 0: return @[]
  let modulus = characteristic(coefficients, p)
  var base = polynomialRemainder(@[0, 1], modulus, p)
  result = @[1]
  for bit in binaryDigits(index):
    if bit == 1: result = polynomialRemainder(multiply(result, base, p), modulus, p)
    base = polynomialRemainder(multiply(base, base, p), modulus, p)

proc recurrenceResult*(problem: RecurrenceProblem, coefficients: seq[int],
                       methodName = "exact", inferredBy = ""): JsonNode =
  let p = problem.prime
  if not matches(problem.samples, coefficients, p):
    raise newException(ValueError, "The recurrence contradicts the supplied samples")
  let weights = termWeights(coefficients, problem.index, p)
  var total = 0'i64
  for j, weight in weights: total += int64(weight) * int64(problem.samples[j])
  let steps = newJArray()
  if inferredBy.len > 0:
    steps.add(%*{"operation": inferredBy, "order": coefficients.len,
                 "samples_checked": problem.samples.len})
  steps.add(%*{"operation": "reduce_x_power_mod_characteristic", "index": problem.index})
  steps.add(%*{"operation": "apply_weights_to_initial_terms", "weights": weights})
  let explanation = if inferredBy.len > 0:
    "The reconstructed recurrence matches every sample. At least twice the promised order bound fixes the extension within that bound."
    else: "Reducing x to the requested power modulo the characteristic polynomial gives weights on the initial terms."
  %*{"answer": residue(total, p), "solution": explanation, "verified": true, "method": methodName,
     "method_steps": steps,
     "certificate": {"coefficients": coefficients, "weights": weights,
                     "characteristic": characteristic(coefficients, p)}}

proc solveRecurrence*(problem: RecurrenceProblem): JsonNode =
  if problem.coefficients.len > 0: return recurrenceResult(problem, problem.coefficients)
  let coefficients = minimalRecurrence(problem.samples, problem.prime)
  if coefficients.len > problem.orderBound:
    raise newException(ValueError, "Samples contradict the promised recurrence order bound")
  recurrenceResult(problem, coefficients, inferredBy = "berlekamp_massey")

proc walkSamples*(matrix: WalkMatrix, source, sink, count, p: int): seq[int] =
  var row = newSeq[int](matrix.len)
  row[source] = 1
  for _ in 0 ..< count:
    result.add(row[sink])
    var next = newSeq[int](matrix.len)
    for j in 0 ..< matrix.len:
      var total = 0'i64
      for k, value in row: total += int64(value) * int64(matrix[k][j])
      next[j] = residue(total, p)
    row = next

proc matrixPowerEntry(matrix: WalkMatrix, source, sink, p: int, index: string): int =
  let n = matrix.len
  var row = newSeq[int](n)
  row[source] = 1
  var base = matrix
  for bit in binaryDigits(index):
    if bit == 1:
      var next = newSeq[int](n)
      for j in 0 ..< n:
        var total = 0'i64
        for k in 0 ..< n: total += int64(row[k]) * int64(base[k][j])
        next[j] = residue(total, p)
      row = next
    var squared = newSeq[seq[int]](n)
    for i in 0 ..< n:
      squared[i] = newSeq[int](n)
      for j in 0 ..< n:
        var total = 0'i64
        for k in 0 ..< n: total += int64(base[i][k]) * int64(base[k][j])
        squared[i][j] = residue(total, p)
    base = squared
  row[sink]

proc parseWalk*(node: JsonNode, maxOrder = MaxWalkOrder): WalkProblem =
  result.prime = readPrime(node)
  result.length = readIndex(node{"length"})
  let values = node{"matrix"}
  if values.isNil or values.kind != JArray or values.len notin 1 .. maxOrder:
    raise newException(ValueError, "Invalid adjacency matrix size")
  let n = values.len
  for row in values:
    if row.kind != JArray or row.len != n: raise newException(ValueError, "Expected a square matrix")
    result.matrix.add(readResidues(row, result.prime, n, "edge weights"))
  let source = integer(node{"source"}, "source")
  let sink = integer(node{"sink"}, "sink")
  if source notin 0'i64 ..< int64(n) or sink notin 0'i64 ..< int64(n):
    raise newException(ValueError, "Vertex is outside the graph")
  result.source = int(source)
  result.sink = int(sink)

proc walkJson*(problem: WalkProblem): JsonNode =
  %*{"kind": "weighted_walk_count", "prime": problem.prime, "length": problem.length,
     "source": problem.source, "sink": problem.sink, "matrix": problem.matrix}

proc walkRecurrence*(problem: WalkProblem): RecurrenceProblem =
  let n = problem.matrix.len
  RecurrenceProblem(prime: problem.prime, orderBound: n, index: problem.length,
    samples: walkSamples(problem.matrix, problem.source, problem.sink, 2 * n, problem.prime))

proc verifyWalk*(problem: WalkProblem, recurrence: RecurrenceProblem, solved: JsonNode): JsonNode =
  result = solved
  let reference = matrixPowerEntry(problem.matrix, problem.source, problem.sink, problem.prime, problem.length)
  if result["answer"].getInt() != reference:
    raise newException(ValueError, "Walk count disagrees with direct matrix exponentiation")
  let steps = newJArray()
  steps.add(%*{"operation": "sample_transfer_matrix", "samples": recurrence.samples,
               "order_bound": recurrence.orderBound})
  for step in result["method_steps"]: steps.add(step)
  steps.add(%*{"operation": "cross_check_matrix_power", "value": reference})
  result["method_steps"] = steps
  result["certificate"]["matrix_power_entry"] = %reference
  result["solution"] = %"Cayley–Hamilton bounds the walk sequence order by the matrix size. The recurrence and an independent matrix power agree on the requested walk count."

proc solveWalk*(node: JsonNode): JsonNode =
  let problem = parseWalk(node)
  let recurrence = walkRecurrence(problem)
  verifyWalk(problem, recurrence, solveRecurrence(recurrence))
