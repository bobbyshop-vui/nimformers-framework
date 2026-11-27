## Euclidean operations and exact Bézout certificates in F_p[x].
import std/[json, strformat]
import prime_field

const MaxPolynomialDegree* = 511

type
  Polynomial* = seq[int]
  PolynomialProblem* = object
    prime*: int
    f*, g*: Polynomial
  EuclidRow = object
    poly, fromF, fromG: Polynomial
  EuclidState* = object
    left, right: EuclidRow
    prime: int
    work*: int
  PolyMove* = enum claimGcd, divideLeft, divideRight, cancelLeft, cancelRight
  PolyCertificate* = object
    gcd*, fromF*, fromG*, quotientF*, quotientG*: Polynomial

proc trim(poly: var Polynomial) =
  var size = poly.len
  while size > 0 and poly[size - 1] == 0: dec size
  poly.setLen(size)

proc readPolynomial(node: JsonNode, prime, maxLength: int, name: string): Polynomial =
  if node.isNil or node.kind != JArray or node.len > maxLength:
    raise newException(ValueError, &"{name} must have at most {maxLength} integer coefficients")
  result = newSeq[int](node.len)
  for i, term in node.elems:
    result[i] = residue(integer(term, name & " coefficient"), prime)
  result.trim()

proc parsePolynomialProblem*(node: JsonNode; policyWidth = 0; primeLimit = ExactMaxPrime): PolynomialProblem =
  result.prime = readPrime(node, primeLimit)
  let maxLength = if policyWidth > 0: policyWidth else: MaxPolynomialDegree + 1
  result.f = readPolynomial(node{"f"}, result.prime, maxLength, "f")
  result.g = readPolynomial(node{"g"}, result.prime, maxLength, "g")

proc problemJson*(p: PolynomialProblem): JsonNode =
  %*{"kind": "finite_field_polynomial_gcd", "prime": p.prime, "f": p.f, "g": p.g}

proc multiply*(a, b: Polynomial, p: int): Polynomial =
  if a.len == 0 or b.len == 0: return
  result = newSeq[int](a.len + b.len - 1)
  for i, u in a:
    if u == 0: continue
    for j, v in b:
      if v != 0: result[i + j] = residue(int64(result[i + j]) + int64(u) * int64(v), p)
  result.trim()

proc sum(a, b: Polynomial, prime: int): Polynomial =
  result = a
  result.setLen(max(a.len, b.len))
  for i, value in b: result[i] = (result[i] + value) mod prime
  result.trim()

proc scaled(poly: Polynomial, unit, prime: int): Polynomial =
  result = newSeq[int](poly.len)
  for i, value in poly: result[i] = productMod(value, unit, prime)
  result.trim()

proc subtractMultiple(a, q, b: Polynomial, p: int): Polynomial =
  if q.len == 0 or b.len == 0: return a
  result = a
  result.setLen(max(a.len, q.len + b.len - 1))
  for i, u in q:
    if u == 0: continue
    for j, v in b:
      if v != 0: result[i + j] = subtractProduct(result[i + j], u, v, p)
  result.trim()

proc divide(a, b: Polynomial, p: int; leadingOnly = false):
    tuple[quotient, remainder: Polynomial, work: int] =
  doAssert b.len > 0
  result.remainder = a
  if a.len < b.len: return
  result.quotient = newSeq[int](a.len - b.len + 1)
  let inverse = reciprocal(b[^1], p)
  while result.remainder.len >= b.len:
    let shift = result.remainder.len - b.len
    let factor = productMod(result.remainder[^1], inverse, p)
    result.quotient[shift] = factor
    for j in 0 ..< b.len - 1:
      if b[j] != 0:
        result.remainder[shift + j] = subtractProduct(result.remainder[shift + j], factor, b[j], p)
        inc result.work
    result.remainder.setLen(result.remainder.len - 1)
    result.remainder.trim()
    inc result.work
    if leadingOnly: break

proc polynomialRemainder*(a, b: Polynomial, p: int): Polynomial =
  divide(a, b, p).remainder

proc polynomialQuotient*(a, b: Polynomial, p: int): Polynomial =
  let divided = divide(a, b, p)
  if divided.remainder.len != 0:
    raise newException(ValueError, "Polynomial does not divide exactly")
  divided.quotient

proc monic*(value: Polynomial, p: int): Polynomial =
  if value.len == 0: return value
  scaled(value, reciprocal(value[^1], p), p)

proc polynomialGcd*(a, b: Polynomial, p: int): Polynomial =
  var x = a
  var y = b
  while y.len > 0:
    let rem = polynomialRemainder(x, y, p)
    x = y
    y = rem
  monic(x, p)

proc newEuclid*(problem: PolynomialProblem): EuclidState =
  EuclidState(prime: problem.prime, left: EuclidRow(poly: problem.f, fromF: @[1]),
    right: EuclidRow(poly: problem.g, fromG: @[1]))

proc livePolynomials*(state: EuclidState): tuple[left, right: Polynomial] =
  (state.left.poly, state.right.poly)

proc allowed*(state: EuclidState): set[PolyMove] =
  result = {claimGcd}
  let leftLen = state.left.poly.len
  let rightLen = state.right.poly.len
  if leftLen == 0 or rightLen == 0: return
  if leftLen >= rightLen:
    result.incl divideLeft
    if leftLen > rightLen: result.incl cancelLeft
  if rightLen >= leftLen:
    result.incl divideRight
    if rightLen > leftLen: result.incl cancelRight

proc bestMove*(state: EuclidState): PolyMove =
  let moves = state.allowed()
  if divideLeft in moves: divideLeft
  elif divideRight in moves: divideRight
  else: claimGcd

proc reduce(target: var EuclidRow, other: EuclidRow, prime: int, leadingOnly: bool): tuple[work: int, quotient: Polynomial] =
  let (q, remainder, work) = divide(target.poly, other.poly, prime, leadingOnly)
  target.poly = remainder
  target.fromF = subtractMultiple(target.fromF, q, other.fromF, prime)
  target.fromG = subtractMultiple(target.fromG, q, other.fromG, prime)
  (work, q)

proc applyMove*(state: var EuclidState, step: PolyMove, methodSteps: JsonNode = nil) =
  if step == claimGcd:
    if not methodSteps.isNil:
      methodSteps.add(%*{"operation": "normalize_and_certify_gcd"})
    return
  let update = case step
    of divideLeft, cancelLeft: reduce(state.left, state.right, state.prime, step == cancelLeft)
    of divideRight, cancelRight: reduce(state.right, state.left, state.prime, step == cancelRight)
    of claimGcd: (0, newSeq[int]())
  state.work += update[0]
  if not methodSteps.isNil:
    methodSteps.add(%*{"operation": $step, "quotient": update[1]})

proc proof*(problem: PolynomialProblem, state: EuclidState): PolyCertificate =
  if state.left.poly.len == 0 and state.right.poly.len == 0: return
  let r = if state.left.poly.len == 0 or
      (state.right.poly.len > 0 and state.right.poly.len <= state.left.poly.len):
    state.right
  else:
    state.left
  let unit = reciprocal(r.poly[^1], problem.prime)
  result.gcd = scaled(r.poly, unit, problem.prime)
  result.fromF = scaled(r.fromF, unit, problem.prime)
  result.fromG = scaled(r.fromG, unit, problem.prime)
  result.quotientF = divide(problem.f, result.gcd, problem.prime).quotient
  result.quotientG = divide(problem.g, result.gcd, problem.prime).quotient

proc verify*(problem: PolynomialProblem, proof: PolyCertificate): bool =
  ## Divide both inputs by the claimed gcd and independently check the
  ## Bézout identity. Together they prove uniqueness of the monic gcd.
  if proof.gcd.len == 0: return problem.f.len == 0 and problem.g.len == 0
  if proof.gcd[^1] != 1: return false
  let p = problem.prime
  if multiply(proof.gcd, proof.quotientF, p) != problem.f or
      multiply(proof.gcd, proof.quotientG, p) != problem.g:
    return false
  sum(multiply(proof.fromF, problem.f, p), multiply(proof.fromG, problem.g, p), p) == proof.gcd

proc resultJson*(problem: PolynomialProblem, witness: PolyCertificate, methodName: string): JsonNode =
  if not verify(problem, witness):
    raise newException(ValueError, "No verified polynomial gcd")
  let explanation =
    if witness.gcd.len == 0: "Both inputs are zero, so the gcd is zero."
    else: &"Over F_{problem.prime}, the monic polynomial d divides f and g. " &
      "The certificate gives s*f + t*g = d, so every common divisor divides d."
  %*{"answer": witness.gcd, "solution": explanation, "verified": true, "method": methodName,
     "certificate": {"bezout_f": witness.fromF, "bezout_g": witness.fromG,
                     "quotient_f": witness.quotientF, "quotient_g": witness.quotientG}}

proc solvePolynomial*(problem: PolynomialProblem): JsonNode =
  var state = newEuclid(problem)
  let methodSteps = newJArray()
  while true:
    let step = state.bestMove()
    state.applyMove(step, methodSteps)
    if step == claimGcd: break
  result = resultJson(problem, proof(problem, state), "exact")
  result["method_steps"] = methodSteps
