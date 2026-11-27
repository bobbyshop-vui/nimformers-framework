## k-th roots: exact integer roots, truncated decimal expansions of real roots,
## and every k-th root of a residue modulo a prime.
import std/[algorithm, json, math, strutils, tables]
import big_integer

const
  MaxRootDegree* = 100_000
  MaxRootDigits* = 100_000
  MaxScaledDigits = 9_500_000
  MaxModularPrime = 9_000_000_000'i64
  DefaultListedRoots = 64
  MaxListedRoots = 1_000

proc integerRoot*(n: BigInt, k: int): BigInt =
  ## floor(n^(1/k)) for n >= 0 and k >= 1, by Newton iteration from above.
  if n.isNegative: raise newException(ValueError, "integerRoot needs a nonnegative argument")
  if k < 1: raise newException(ValueError, "The root degree must be at least 1")
  if n.isZero or k == 1: return n
  if n < big(1 shl k.clamp(1, 62)): return big(1)
  let logRoot = n.log10Abs() / float64(k)
  # Leading digits from floating point, then a relative margin of 1e-7 keeps the start above the root.
  let shift = max(0, int(floor(logRoot)) - 15)
  var y = big(int64(floor(pow(10.0, logRoot - float64(shift)))) + 1)
  if shift > 0: y = y * power(big(10), shift)
  y = y + divSmall(y, 10_000_000).quotient + big(1)
  while power(y, k) <= n: y = y + y
  let degree = big(int64(k - 1))
  while true:
    let next = divSmall(degree * y + divMod(n, power(y, k - 1)).quotient, int64(k)).quotient
    if not (next < y): break
    y = next
  y

proc perfectRoot*(n: BigInt, k: int): tuple[ok: bool, root: BigInt] =
  ## The exact k-th root of an integer when one exists. Odd degrees accept negatives.
  if n.isNegative:
    if k mod 2 == 0: return
    let (ok, root) = perfectRoot(-n, k)
    return (ok, -root)
  let r = integerRoot(n, k)
  (power(r, k) == n, r)

proc parseRational(node: JsonNode): tuple[num, den: BigInt] =
  if node.isNil: raise newException(ValueError, "Missing value")
  if node.kind == JString and '/' in node.getStr():
    let parts = node.getStr().split('/')
    if parts.len != 2: raise newException(ValueError, "A fraction must look like p/q")
    result = (parseBig(parts[0]), parseBig(parts[1]))
  else:
    result = (toBig(node, "value"), big(1))
  if result.den.isZero: raise newException(ValueError, "Division by zero")
  if result.den.isNegative: result = (-result.num, -result.den)
  let g = gcdBig(result.num, result.den)
  if not (g == big(1)) and not g.isZero:
    result = (divMod(result.num, g).quotient, divMod(result.den, g).quotient)

proc decimalText(scaled: BigInt, digits: int, negative: bool): string =
  var text = $scaled
  if digits > 0:
    if text.len <= digits: text = repeat('0', digits - text.len + 1) & text
    text = text[0 ..< text.len - digits] & "." & text[text.len - digits .. ^1]
  if negative and scaled != big(0): "-" & text else: text

proc realRoot(problem: JsonNode, k: int): JsonNode =
  let (num, den) = parseRational(problem{"value"})
  let digits = int(problem{"digits"}.getBiggestInt(30))
  if digits notin 0 .. MaxRootDigits:
    raise newException(ValueError, "digits must be between 0 and " & $MaxRootDigits)
  let negative = num.isNegative
  if negative and k mod 2 == 0:
    raise newException(ValueError, "An even root of a negative number is not real. Pass a prime for a modular root.")
  let magnitude = if negative: -num else: num
  if magnitude.digits() + den.digits() + k * digits > MaxScaledDigits:
    raise newException(ValueError, "The scaled value would need more than " & $MaxScaledDigits &
      " digits. Lower digits or the root degree.")
  let numeratorRoot = perfectRoot(magnitude, k)
  let denominatorRoot = perfectRoot(den, k)
  let exact = numeratorRoot.ok and denominatorRoot.ok
  # floor(root(floor(X))) equals floor(root(X)) for X = |value| * 10^(k*digits).
  let scaledValue = divMod(magnitude * power(big(10), k * digits), den).quotient
  let lower = integerRoot(scaledValue, k)
  let low = power(lower, k)
  let high = power(lower + big(1), k)
  let bracketed = low <= scaledValue and scaledValue < high
  if not bracketed: raise newException(ValueError, "The root bracket did not verify")
  result = %*{"degree": k, "digits": digits, "exact": exact,
    "answer": decimalText(lower, digits, negative),
    "rounding": "truncated toward zero",
    "verified": true, "method": "exact"}
  if exact:
    let top = if negative: -numeratorRoot.root else: numeratorRoot.root
    result["exact_root"] = if denominatorRoot.root == big(1): top.toJson()
                           else: %($top & "/" & $denominatorRoot.root)
    result["solution"] = %("The value is a perfect " & $k & "-th power, so the root is rational.")
  else:
    result["solution"] = %("The root is irrational. The answer keeps " & $digits &
      " digits after the decimal point, truncated toward zero.")
  if den == big(1):
    result["integer_root"] = (if negative: -numeratorRoot.root else: numeratorRoot.root).toJson()
  result["certificate"] = %*{
    "scaled_value": "floor(|value| * 10^" & $(k * digits) & ")",
    "check": "y^" & $k & " <= scaled_value < (y+1)^" & $k & " with y = |answer| * 10^" & $digits,
    "y_digits": lower.digits(), "holds": bracketed}

# Modular roots.

proc isPrime64*(n: int64): bool =
  ## Deterministic Miller-Rabin for signed 64 bit inputs.
  if n < 2: return false
  for small in [2'i64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
    if n mod small == 0: return n == small
  var d = n - 1
  var shifts = 0
  while d mod 2 == 0:
    d = d div 2
    inc shifts
  for witness in [2'i64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
    var x = powMod(witness, d, n)
    if x == 1 or x == n - 1: continue
    var composite = true
    for _ in 1 ..< shifts:
      x = mulMod(x, x, n)
      if x == n - 1:
        composite = false
        break
    if composite: return false
  true

proc primeFactors(n: int64): seq[int64] =
  var rest = n
  var d = 2'i64
  while d * d <= rest:
    if rest mod d == 0:
      result.add(d)
      while rest mod d == 0: rest = rest div d
    d += (if d == 2: 1 else: 2)
  if rest > 1: result.add(rest)

proc primitiveRoot(p: int64): int64 =
  if p == 2: return 1
  let factors = primeFactors(p - 1)
  for g in 2'i64 ..< p:
    var generator = true
    for q in factors:
      if powMod(g, (p - 1) div q, p) == 1:
        generator = false
        break
    if generator: return g
  raise newException(ValueError, "No primitive root found")

proc discreteLog(base, target, p: int64): int64 =
  ## Baby-step giant-step for base^e = target with base a primitive root.
  let order = p - 1
  let m = int64(ceil(sqrt(float64(order))))
  var baby = initTable[int64, int64](int(m))
  var value = 1'i64
  for j in 0'i64 ..< m:
    if value notin baby: baby[value] = j
    value = mulMod(value, base, p)
  let giant = modularInverse(powMod(base, m, p), p)
  var gamma = target mod p
  for i in 0'i64 .. m:
    if gamma in baby: return (i * m + baby[gamma]) mod order
    gamma = mulMod(gamma, giant, p)
  raise newException(ValueError, "Discrete logarithm not found")

proc modularRoots*(x, k, p: int64, limit: int): tuple[count: int64, roots: seq[int64]] =
  ## Every y in [0, p) with y^k = x mod p. Only the smallest limit roots are listed.
  if not isPrime64(p) or p > MaxModularPrime:
    raise newException(ValueError, "The modulus must be a prime up to " & $MaxModularPrime)
  if k < 1: raise newException(ValueError, "The root degree must be at least 1")
  let target = ((x mod p) + p) mod p
  if target == 0: return (1'i64, @[0'i64])
  if p == 2: return (1'i64, @[1'i64])
  let order = p - 1
  let g = gcd(k, order)
  if powMod(target, order div g, p) != 1: return (0'i64, @[])
  let generator = primitiveRoot(p)
  let e = discreteLog(generator, target, p)
  let reduced = order div g
  let t0 = mulMod((e div g) mod reduced, modularInverse((k div g) mod reduced, reduced), reduced)
  let first = powMod(generator, t0, p)
  let unity = powMod(generator, reduced, p)
  var roots: seq[int64]
  var current = first
  for _ in 0'i64 ..< g:
    roots.add(current)
    current = mulMod(current, unity, p)
  roots.sort()
  for r in roots:
    if powMod(r, k, p) != target: raise newException(ValueError, "A modular root did not verify")
  (g, roots[0 ..< min(roots.len, limit)])

proc modularRoot(problem: JsonNode, k: int64): JsonNode =
  let p = problem["prime"].getBiggestInt()
  let x = toBig(problem{"value"}, "value").modSmall(p)
  let limit = int(problem{"max_roots"}.getBiggestInt(DefaultListedRoots))
  if limit notin 1 .. MaxListedRoots: raise newException(ValueError, "max_roots must be between 1 and " & $MaxListedRoots)
  let (count, roots) = modularRoots(x, k, p, limit)
  result = %*{"degree": k, "prime": p, "residue": x, "root_count": count, "roots": roots,
    "listed": roots.len, "verified": true, "method": "exact"}
  if count == 0:
    let g = gcd(k, p - 1)
    result["solution"] = %("No root exists: x^((p-1)/" & $g & ") is not 1 modulo p.")
    result["certificate"] = %*{"gcd_degree_order": g, "criterion": powMod(x, (p - 1) div g, p)}
  else:
    result["answer"] = %roots[0]
    result["solution"] = %("With g = gcd(k, p-1) = " & $count & ", a primitive root and a discrete logarithm give " &
      "one root. Multiplying by the g-th roots of unity gives all of them.")
    result["certificate"] = %*{"gcd_degree_order": count, "check": "every listed root satisfies y^k = x mod p"}

proc solveNthRoot*(problem: JsonNode): JsonNode =
  let degreeNode = problem{"degree"}
  if degreeNode.isNil or degreeNode.kind != JInt: raise newException(ValueError, "degree must be an integer")
  let k = degreeNode.getBiggestInt()
  if problem.hasKey("prime"):
    if k notin 1'i64 .. 1_000_000_000_000_000'i64: raise newException(ValueError, "degree is out of range")
    return modularRoot(problem, k)
  if k notin 1'i64 .. int64(MaxRootDegree):
    raise newException(ValueError, "degree must be between 1 and " & $MaxRootDegree)
  realRoot(problem, int(k))
