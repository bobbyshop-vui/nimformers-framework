## Square-free, distinct-degree, and equal-degree factoring over F_p.
import std/[algorithm, json, random]
import polynomial_math, prime_field

const MaxFactorDegree* = 128

type Factor = object
  polynomial: Polynomial
  multiplicity: int

proc addPoly(a, b: Polynomial, p: int, subtract = false): Polynomial =
  result = a
  result.setLen(max(a.len, b.len))
  for i, term in b:
    result[i] = (if subtract: result[i] + p - term else: result[i] + term) mod p
  while result.len > 0 and result[^1] == 0: result.setLen(result.len - 1)

proc reducedProduct(a, b, modulus: Polynomial, p: int): Polynomial =
  polynomialRemainder(multiply(a, b, p), modulus, p)

proc powerMod(a: Polynomial, exponent: int, modulus: Polynomial, p: int): Polynomial =
  var base = polynomialRemainder(a, modulus, p)
  var n = exponent
  result = @[1]
  while n > 0:
    if (n and 1) != 0: result = reducedProduct(result, base, modulus, p)
    n = n shr 1
    if n > 0: base = reducedProduct(base, base, modulus, p)

proc equalDegree(f: Polynomial, degree, multiplicity, p: int,
                 rng: var Rand, factors: var seq[Factor], steps: JsonNode) =
  if f.len == degree + 1:
    factors.add(Factor(polynomial: f, multiplicity: multiplicity))
    return
  for _ in 0 ..< 128:
    var a = newSeq[int](f.len - 1)
    for value in a.mitems: value = rng.rand(p - 1)
    while a.len > 0 and a[^1] == 0: a.setLen(a.len - 1)
    if a.len == 0: continue
    var part = polynomialGcd(f, a, p)
    if part.len == 1:
      if p == 2:
        var trace: Polynomial
        for i in 0 ..< degree:
          trace = addPoly(trace, a, p)
          if i + 1 < degree: a = reducedProduct(a, a, f, p)
        part = polynomialGcd(f, trace, p)
      else:
        var h = @[1]
        a = powerMod(a, (p - 1) div 2, f, p)
        for i in 0 ..< degree:
          h = reducedProduct(h, a, f, p)
          if i + 1 < degree: a = powerMod(a, p, f, p)
        part = polynomialGcd(f, addPoly(h, @[1], p, subtract = true), p)
    if part.len <= 1 or part.len >= f.len: continue
    let other = polynomialQuotient(f, part, p)
    steps.add(%*{"operation": "split_equal_degree", "degree": degree,
                 "left_degree": part.len - 1, "right_degree": other.len - 1})
    equalDegree(part, degree, multiplicity, p, rng, factors, steps)
    equalDegree(other, degree, multiplicity, p, rng, factors, steps)
    return
  raise newException(ValueError, "Equal-degree factorization exceeded its split budget")

proc distinctDegree(f: Polynomial, multiplicity, p: int, rng: var Rand,
                    factors: var seq[Factor], steps: JsonNode) =
  const x = @[0, 1]
  var rest = f
  var h = x
  var degree = 1
  while 2 * degree <= rest.len - 1:
    h = powerMod(h, p, rest, p)
    let part = polynomialGcd(rest, addPoly(h, x, p, subtract = true), p)
    if part.len > 1:
      steps.add(%*{"operation": "distinct_degree_bucket", "degree": degree,
                   "bucket_degree": part.len - 1, "multiplicity": multiplicity})
      equalDegree(part, degree, multiplicity, p, rng, factors, steps)
      rest = polynomialQuotient(rest, part, p)
      if rest.len == 1: return
      h = polynomialRemainder(h, rest, p)
    inc degree
  if rest.len > 1: factors.add(Factor(polynomial: rest, multiplicity: multiplicity))

proc squareFree(f: Polynomial, multiplier, p: int, rng: var Rand,
                factors: var seq[Factor], steps: JsonNode) =
  var derivative = newSeq[int](f.len - 1)
  for i in 1 ..< f.len: derivative[i - 1] = productMod(i mod p, f[i], p)
  while derivative.len > 0 and derivative[^1] == 0: derivative.setLen(derivative.len - 1)
  var repeated = polynomialGcd(f, derivative, p)
  var pending = polynomialQuotient(f, repeated, p)
  var multiplicity = 1
  while pending.len > 1:
    let overlap = polynomialGcd(pending, repeated, p)
    let part = polynomialQuotient(pending, overlap, p)
    if part.len > 1:
      distinctDegree(part, multiplier * multiplicity, p, rng, factors, steps)
    pending = overlap
    repeated = polynomialQuotient(repeated, overlap, p)
    inc multiplicity
  if repeated.len > 1:
    var root = newSeq[int]((repeated.len - 1) div p + 1)
    for i, coefficient in repeated:
      if i mod p == 0: root[i div p] = coefficient
      elif coefficient != 0: raise newException(ValueError, "Expected a characteristic power")
    steps.add(%*{"operation": "extract_characteristic_root", "prime": p})
    squareFree(root, multiplier * p, p, rng, factors, steps)

proc irreducible(f: Polynomial, p: int): bool =
  if f.len <= 1 or f[^1] != 1: return false
  if f.len == 2: return true
  let degree = f.len - 1
  let x: Polynomial = @[0, 1]
  var h = x
  for i in 1 .. degree:
    h = powerMod(h, p, f, p)
    if degree mod i == 0 and i < degree and
        polynomialGcd(f, addPoly(h, x, p, subtract = true), p).len != 1:
      return false
  h == x

proc solveFactorization*(node: JsonNode): JsonNode =
  let problem = parsePolynomialProblem(%*{"prime": node{"prime"}, "f": node{"polynomial"}, "g": []},
                                        policyWidth = MaxFactorDegree + 1)
  let f = problem.f
  let p = problem.prime
  if f.len == 0: raise newException(ValueError, "Zero has no finite factorization into irreducibles")
  let leading = f[^1]
  var rng = initRand(482_019)
  var factors: seq[Factor]
  let steps = newJArray()
  steps.add(%*{"operation": "normalize_and_separate_square_free_parts", "degree": f.len - 1})
  if f.len > 1: squareFree(monic(f, p), 1, p, rng, factors, steps)
  factors.sort(proc(a, b: Factor): int =
    result = cmp(a.polynomial.len, b.polynomial.len)
    if result == 0:
      for i in 0 ..< a.polynomial.len:
        result = cmp(a.polynomial[i], b.polynomial[i])
        if result != 0: return)
  let answer = newJArray()
  var rebuilt: Polynomial = @[leading]
  for factor in factors:
    if not irreducible(factor.polynomial, p):
      raise newException(ValueError, "An alleged irreducible factor did not verify")
    for _ in 0 ..< factor.multiplicity: rebuilt = multiply(rebuilt, factor.polynomial, p)
    answer.add(%*{"polynomial": factor.polynomial, "multiplicity": factor.multiplicity})
  if rebuilt != f: raise newException(ValueError, "Factor product did not verify")
  steps.add(%*{"operation": "verify_irreducibility_and_reconstruct", "factor_count": factors.len})
  %*{"answer": {"leading": leading, "factors": answer}, "verified": true, "method": "exact",
     "solution": "Separate repeated factors, split the square-free parts by degree, and factor each bucket. Frobenius and GCD checks certify irreducibility, and multiplying all factors reconstructs the input.",
     "method_steps": steps,
     "certificate": {"prime": p, "product": rebuilt, "irreducibility": "frobenius_gcd"}}
