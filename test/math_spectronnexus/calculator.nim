## Exact calculator over rational numbers and prime fields. The grammar has
## arithmetic and mathematical functions only.
import std/[algorithm, json, math, sequtils, strutils]
import big_integer, exact_number, linear_system, prime_field, root_math

const
  MaxDepth = 32
  MaxEvaluations = 50_000_000
  MaxLoopLength = 10_000_000
  MaxMatrixExponent = 4_096
  MaxScalarExponent = 1_000_000
  MaxExactFactorial = 1_000_000
  MaxBinomialSteps = 200_000     ## Step by step when n is past the sieve.
  MaxBinomialSieve = 30_000_000'i64   ## Legendre's formula sieves primes up to n.
  MaxExpressionLength = 16_384

type
  Parser = object
    source: string
    pos: int
  Value = object
    rows, cols: int
    entries: seq[Exact]
  Elimination = object
    reduced, transform: Value
    rank: int
    determinant: Exact
    pivots: JsonNode
  Evaluator = object
    variables, steps: JsonNode
    nodes, quiet: int

proc checkDepth(depth: int) =
  if depth > MaxDepth: raise newException(ValueError, "Expression is nested too deeply")

proc nextChar(parser: var Parser): char =
  while parser.pos < parser.source.len and parser.source[parser.pos] in Whitespace:
    inc parser.pos
  if parser.pos < parser.source.len: parser.source[parser.pos] else: '\0'

proc accept(parser: var Parser, ch: char): bool =
  if parser.nextChar() != ch: return false
  inc parser.pos
  true

proc operation(op: string, args: openArray[JsonNode]): JsonNode =
  %*{"op": op, "args": @args}

proc parseSum(parser: var Parser, depth: int): JsonNode

proc parseAtom(parser: var Parser, depth: int): JsonNode =
  checkDepth(depth)
  let ch = parser.nextChar()
  if parser.accept('('):
    result = parser.parseSum(depth + 1)
    if not parser.accept(')'): raise newException(ValueError, "Expected ')'")
  elif ch in {'0'..'9', '.'}:
    let start = parser.pos
    while parser.pos < parser.source.len and parser.source[parser.pos] in {'0'..'9', '.'}:
      inc parser.pos
    if parser.pos < parser.source.len and parser.source[parser.pos] in {'e', 'E'}:
      inc parser.pos
      if parser.pos < parser.source.len and parser.source[parser.pos] in {'+', '-'}: inc parser.pos
      while parser.pos < parser.source.len and parser.source[parser.pos] in {'0'..'9'}: inc parser.pos
    result = %parser.source[start ..< parser.pos]
  elif ch in {'A'..'Z', 'a'..'z', '_'}:
    let start = parser.pos
    while parser.pos < parser.source.len and parser.source[parser.pos] in
        {'A'..'Z', 'a'..'z', '0'..'9', '_'}:
      inc parser.pos
    let name = parser.source[start ..< parser.pos]
    if parser.accept('('):
      var args: seq[JsonNode]
      if not parser.accept(')'):
        while true:
          args.add(parser.parseSum(depth + 1))
          if parser.accept(')'): break
          if not parser.accept(','): raise newException(ValueError, "Expected ',' or ')'")
      result = operation(name, args)
    else:
      result = %name
  else:
    raise newException(ValueError, "Expected a number, variable, or function at position " & $(parser.pos + 1))

proc parseUnary(parser: var Parser, depth: int): JsonNode =
  checkDepth(depth)
  if parser.accept('-'): return operation("negate", [parser.parseUnary(depth + 1)])
  if parser.accept('+'): return parser.parseUnary(depth + 1)
  result = parser.parseAtom(depth)
  if parser.accept('^'):
    result = operation("power", [result, parser.parseUnary(depth + 1)])

proc parseProduct(parser: var Parser, depth: int): JsonNode =
  result = parser.parseUnary(depth)
  while true:
    let ch = parser.nextChar()
    if ch notin {'*', '/', '%'}: break
    inc parser.pos
    let name = if ch == '*': "multiply" elif ch == '/': "divide" else: "mod"
    result = operation(name, [result, parser.parseUnary(depth)])

proc parseSum(parser: var Parser, depth: int): JsonNode =
  result = parser.parseProduct(depth)
  while true:
    let ch = parser.nextChar()
    if ch notin {'+', '-'}: break
    inc parser.pos
    let name = if ch == '+': "add" else: "subtract"
    result = operation(name, [result, parser.parseProduct(depth)])

proc parse(source: string): JsonNode =
  if source.len > MaxExpressionLength: raise newException(ValueError, "Expression is too long")
  var parser = Parser(source: source)
  result = parser.parseSum(0)
  if parser.pos < source.len:
    discard parser.nextChar()
    if parser.pos < source.len: raise newException(ValueError, "Unexpected input at position " & $(parser.pos + 1))

proc scalar(value: Exact): Value = Value(entries: @[value])

proc matrix(rows, cols: int): Value =
  if rows notin 1 .. SystemMaxOrder or cols notin 1 .. SystemMaxOrder:
    raise newException(ValueError, "Matrix dimensions must be between 1 and " & $SystemMaxOrder)
  result = Value(rows: rows, cols: cols, entries: newSeq[Exact](rows * cols))
  for entry in result.entries.mitems: entry = Zero

proc isScalar(value: Value): bool = value.rows == 0

proc number(value: Value, name: string): Exact =
  if not value.isScalar(): raise newException(ValueError, name & " requires a scalar")
  value.entries[0]

template item(value: Value, row, col: int): untyped = value.entries[row * value.cols + col]

proc literals(node: JsonNode, p: int): Value =
  if not node.isNil and node.kind == JArray:
    if node.len == 0 or node[0].kind != JArray:
      raise newException(ValueError, "Expected matrix rows")
    result = matrix(node.len, node[0].len)
    for r, row in node.elems:
      if row.kind != JArray or row.len != result.cols:
        raise newException(ValueError, "Matrix rows must have equal lengths")
      for c, entry in row.elems: result.item(r, c) = parseExact(entry).asField(p)
  else:
    result = scalar(parseExact(node).asField(p))

proc toJson(value: Value): JsonNode =
  if value.isScalar(): return value.entries[0].toJson()
  result = newJArray()
  for r in 0 ..< value.rows:
    let row = newJArray()
    for c in 0 ..< value.cols: row.add(value.item(r, c).toJson())
    result.add(row)

proc describe(value: Value): JsonNode =
  if value.isScalar(): value.toJson() else: %*{"shape": [value.rows, value.cols]}

proc sameShape(a, b: Value, name: string) =
  if a.rows != b.rows or a.cols != b.cols:
    raise newException(ValueError, name & " requires matching dimensions")

proc identity(n: int): Value =
  result = matrix(n, n)
  for i in 0 ..< n: result.item(i, i) = One

proc scaled(value: Value, factor: Exact, p: int): Value =
  result = value
  for entry in result.entries.mitems: entry = entry.times(factor, p)

proc product(a, b: Value, p: int): Value =
  if a.isScalar(): return b.scaled(a.entries[0], p)
  if b.isScalar(): return a.scaled(b.entries[0], p)
  if a.cols != b.rows:
    raise newException(ValueError, "Left columns must equal right rows")
  result = matrix(a.rows, b.cols)
  for r in 0 ..< a.rows:
    for k in 0 ..< a.cols:
      let coefficient = a.item(r, k)
      if coefficient.isZero: continue
      for c in 0 ..< b.cols:
        result.item(r, c) = result.item(r, c).plus(coefficient.times(b.item(k, c), p), p)

proc eliminate(value: Value, p: int, trackDeterminant: bool): Elimination =
  if value.isScalar(): raise newException(ValueError, "Elimination requires a matrix")
  result.reduced = value
  result.transform = identity(value.rows)
  result.determinant = One
  result.pivots = newJArray()
  for c in 0 ..< value.cols:
    let r = result.rank
    if r == value.rows: break
    var pivot = r
    while pivot < value.rows and result.reduced.item(pivot, c).isZero: inc pivot
    if pivot == value.rows: continue
    if pivot != r:
      for j in 0 ..< value.cols: swap(result.reduced.item(r, j), result.reduced.item(pivot, j))
      for j in 0 ..< value.rows: swap(result.transform.item(r, j), result.transform.item(pivot, j))
      if trackDeterminant: result.determinant = result.determinant.negative(p)
    let diagonal = result.reduced.item(r, c)
    let inv = diagonal.inverse(p)
    if trackDeterminant: result.determinant = result.determinant.times(diagonal, p)
    for j in c ..< value.cols:
      result.reduced.item(r, j) = result.reduced.item(r, j).times(inv, p)
    for j in 0 ..< value.rows:
      result.transform.item(r, j) = result.transform.item(r, j).times(inv, p)
    for i in 0 ..< value.rows:
      if i == r: continue
      let factor = result.reduced.item(i, c).negative(p)
      if factor.isZero: continue
      for j in c ..< value.cols:
        result.reduced.item(i, j) = result.reduced.item(i, j).plus(result.reduced.item(r, j).times(factor, p), p)
      for j in 0 ..< value.rows:
        result.transform.item(i, j) = result.transform.item(i, j).plus(result.transform.item(r, j).times(factor, p), p)
    result.pivots.add(%*{"column": c, "swap_rows": [r, pivot], "pivot": diagonal.toJson()})
    inc result.rank
  if trackDeterminant and result.rank < value.rows: result.determinant = Zero
  if product(result.transform, value, p) != result.reduced:
    raise newException(ValueError, "Row reduction did not verify")

proc square(value: Value, op: string) =
  if value.isScalar() or value.rows != value.cols:
    raise newException(ValueError, op & " requires a square matrix")

proc inverted(value: Value, p: int, step: JsonNode): Value =
  if value.isScalar(): return scalar(value.entries[0].inverse(p))
  value.square("Inverse")
  let eliminated = eliminate(value, p, trackDeterminant = false)
  if eliminated.rank < value.rows: raise newException(ValueError, "Matrix is singular")
  step["pivots"] = eliminated.pivots
  if product(value, eliminated.transform, p) != identity(value.rows):
    raise newException(ValueError, "Matrix inverse did not verify")
  eliminated.transform

proc power(value: Value, exponent: int64, p: int, step: JsonNode): Value =
  let limit = int64(if value.isScalar(): MaxScalarExponent else: MaxMatrixExponent)
  if exponent notin -limit .. limit:
    raise newException(ValueError, "Exponent must be between -" & $limit & " and " & $limit)
  if not value.isScalar(): value.square("Power")
  var base = if exponent < 0: inverted(value, p, step) else: value
  var remaining = abs(exponent)
  result = if value.isScalar(): scalar(One) else: identity(value.rows)
  while remaining > 0:
    if (remaining and 1'i64) != 0: result = product(result, base, p)
    remaining = remaining shr 1
    if remaining > 0: base = product(base, base, p)

proc compareExact(a, b: Exact): int = compare(a, b)

proc productTree(first, last: int64): BigInt =
  ## first * (first+1) * ... * last by balanced halves, so the big products stay balanced.
  if first > last: return big(1)
  if last - first < 16:
    result = big(1)
    for i in first .. last: result = result * big(i)
    return
  let middle = first + (last - first) div 2
  productTree(first, middle) * productTree(middle + 1, last)

proc truth(value: bool, prime: int): Value =
  scalar(whole(int64(ord(value))).asField(prime))

proc isProbablePrime*(n: int64): bool =
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

proc pollardFactor(n: int64): int64 =
  ## A nontrivial factor of an odd composite n by Brent's variant of Pollard rho.
  var seed = 1'i64
  while true:
    var y = 2'i64
    var r = 1
    var g = 1'i64
    var q = 1'i64
    var x, ys: int64
    template step(v: int64): int64 =
      let square = mulMod(v, v, n)
      if square >= n - seed: square - (n - seed) else: square + seed
    while g == 1:
      x = y
      for _ in 1 .. r: y = step(y)
      var k = 0
      while k < r and g == 1:
        ys = y
        for _ in 1 .. min(128, r - k):
          y = step(y)
          q = mulMod(q, abs(x - y), n)
        g = gcd(q, n)
        k += 128
      r *= 2
    if g == n:
      g = 1
      while g == 1:
        ys = step(ys)
        g = gcd(abs(x - ys), n)
    if g != n: return g
    inc seed

proc factorize*(n: int64): seq[tuple[prime: int64, exponent: int]] =
  ## Trial division by small primes, then Miller-Rabin and Pollard rho, so every
  ## positive int64 is factored.
  if n < 1: raise newException(ValueError, "Arithmetic functions need n >= 1")
  var rest = n
  var found: seq[int64]
  var divisor = 2'i64
  while divisor <= 1000 and divisor * divisor <= rest:
    while rest mod divisor == 0:
      found.add(divisor)
      rest = rest div divisor
    divisor += (if divisor == 2: 1 else: 2)
  var pending = if rest > 1: @[rest] else: @[]
  while pending.len > 0:
    let m = pending.pop()
    if m == 1: continue
    if isProbablePrime(m):
      found.add(m)
      continue
    let root = min(int64(sqrt(float64(m)) + 0.5), 3_037_000_499'i64)
    let factor = if root * root == m: root else: pollardFactor(m)
    pending.add(factor)
    pending.add(m div factor)
  found.sort()
  for p in found:
    if result.len > 0 and result[^1].prime == p: inc result[^1].exponent
    else: result.add((p, 1))

proc isProbablePrimeBig(n: BigInt): bool =
  ## Miller-Rabin past 64 bits with the first 20 prime bases. It is deterministic below
  ## 3.3 * 10^24 and a strong probable prime test above.
  if n.fitsInt64(): return isProbablePrime(n.toInt64())
  if n.isNegative: return false
  const Bases = [2'i64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
  for small in Bases:
    if n.modSmall(small) == 0: return false
  let minusOne = n - big(1)
  var d = minusOne
  var shifts = 0
  while d.modSmall(2) == 0:
    d = divSmall(d, 2).quotient
    inc shifts
  for witness in Bases:
    var x = powModBig(big(witness), d, n)
    if x == big(1) or x == minusOne: continue
    var composite = true
    for _ in 1 ..< shifts:
      x = floorMod(x * x, n)
      if x == minusOne:
        composite = false
        break
    if composite: return false
  true

proc productOf(items: seq[BigInt], first, last: int): BigInt =
  ## items[first] * ... * items[last] by balanced halves.
  if first > last: return big(1)
  if first == last: return items[first]
  let middle = (first + last) div 2
  productOf(items, first, middle) * productOf(items, middle + 1, last)

proc binomialByPrimes(n, k: int64): BigInt =
  ## C(n, k) as the product of p^e over primes p <= n, where Legendre's formula gives
  ## e = sum over i of floor(n/p^i) - floor(k/p^i) - floor((n-k)/p^i). The prime powers
  ## are multiplied in a balanced tree, so the big products are Karatsuba sized.
  var composite = newSeq[bool](int(n) + 1)
  var powers: seq[BigInt]
  var small = 1'i64        # Small prime powers are grouped below 10^18 before going big.
  for p in 2'i64 .. n:
    if composite[int(p)]: continue
    if p * p <= n:
      var multiple = p * p
      while multiple <= n:
        composite[int(multiple)] = true
        multiple += p
    var exponent = 0'i64
    var scale = p
    while scale <= n:
      exponent += n div scale - k div scale - (n - k) div scale
      if scale > n div p: break
      scale *= p
    for _ in 1 .. exponent:
      if small > 1_000_000_000_000_000_000'i64 div p:
        powers.add(big(small))
        small = 1
      small *= p
  if small > 1: powers.add(big(small))
  productOf(powers, 0, powers.high)

proc binomialExact(n, k: int64, prime: int): Exact =
  if n < 0: raise newException(ValueError, "binomial requires n >= 0")
  if k < 0 or k > n: return Zero
  if prime == 0:
    let steps = min(k, n - k)
    if steps > 2_000 and n <= MaxBinomialSieve:
      return wholeBig(binomialByPrimes(n, steps))
    if steps > MaxBinomialSteps:
      raise newException(ValueError, "binomial(n,k) is exact for n up to " & $MaxBinomialSieve &
        ", or for min(k, n-k) up to " & $MaxBinomialSteps)
    var value = big(1)
    for i in 0'i64 ..< steps: value = exactDiv(value * big(n - i), i + 1)
    return wholeBig(value)
  # Lucas theorem over the prime field.
  var factorial = newSeq[int](prime)
  factorial[0] = 1
  for i in 1 ..< prime: factorial[i] = productMod(factorial[i - 1], i, prime)
  var value = 1
  var (top, bottom) = (n, k)
  while top > 0 or bottom > 0:
    let a = int(top mod prime)
    let b = int(bottom mod prime)
    if b > a: return Zero
    value = productMod(value, factorial[a], prime)
    value = productMod(value, reciprocal(productMod(factorial[b], factorial[a - b], prime), prime), prime)
    top = top div prime
    bottom = bottom div prime
  whole(int64(value))

proc divisorSigma(power, n: int64): Exact =
  if power notin 0'i64 .. 64'i64: raise newException(ValueError, "sigma power must be between 0 and 64")
  result = One
  for (p, exponent) in factorize(n):
    var term = One
    var sum = One
    let base = whole(p)
    var step = One
    for _ in 1 .. power: step = step.times(base, 0)
    for _ in 1 .. exponent:
      term = term.times(step, 0)
      sum = sum.plus(term, 0)
    result = result.times(sum, 0)

proc eval(context: var Evaluator, node: JsonNode, prime, depth: int): Value

proc loop(context: var Evaluator, op: string, args: JsonNode, prime, depth: int): Value =
  if args.isNil or args.kind != JArray or args.len != 4:
    raise newException(ValueError, op & " requires (variable, first, last, body)")
  let name = args[0]
  if name.kind != JString or name.getStr().len == 0 or name.getStr()[0] notin {'A'..'Z', 'a'..'z', '_'}:
    raise newException(ValueError, op & " needs a variable name as its first argument")
  let variable = name.getStr()
  let first = context.eval(args[1], 0, depth + 1).number(op).integral(op)
  let last = context.eval(args[2], 0, depth + 1).number(op).integral(op)
  if last >= first and last - first >= MaxLoopLength:
    raise newException(ValueError, op & " ranges are limited to " & $MaxLoopLength & " values")
  let saved = context.variables{variable}
  var total = if op == "prod": One.asField(prime) else: Zero
  var hits = 0'i64
  inc context.quiet
  try:
    for k in first .. last:
      context.variables[variable] = %k
      let value = context.eval(args[3], prime, depth + 1).number(op)
      case op
      of "sum": total = total.plus(value, prime)
      of "prod": total = total.times(value, prime)
      else:
        if not value.isZero: inc hits
  finally:
    dec context.quiet
    if not saved.isNil: context.variables[variable] = saved
    elif context.variables.hasKey(variable): context.variables.delete(variable)
  scalar(if op == "count": whole(hits).asField(prime) else: total)

proc eval(context: var Evaluator, node: JsonNode, prime, depth: int): Value =
  checkDepth(depth)
  inc context.nodes
  if context.nodes > MaxEvaluations: raise newException(ValueError, "Expression needs too many evaluations")
  if node.isNil: raise newException(ValueError, "Missing expression")
  if node.kind == JString and context.variables.hasKey(node.getStr()):
    return literals(context.variables[node.getStr()], prime)
  if node.kind != JObject: return literals(node, prime)
  let original = node{"op"}.getStr().toLowerAscii()
  let op = case original
    of "inv": "inverse"
    of "det": "determinant"
    of "eye": "identity"
    of "pow": "power"
    of "choose", "binom": "binomial"
    of "mu": "mobius"
    of "phi": "totient"
    else: original
  let args = node{"args"}
  # Lazy forms evaluate only the arguments they need.
  case op
  of "sum", "prod", "count":
    return context.loop(op, args, prime, depth)
  of "if":
    if args.isNil or args.kind != JArray or args.len != 3:
      raise newException(ValueError, "if requires (condition, then, else)")
    let condition = context.eval(args[0], prime, depth + 1).number(op)
    return context.eval(args[(if not condition.isZero: 1 else: 2)], prime, depth + 1)
  of "and", "or":
    if args.isNil or args.kind != JArray or args.len != 2:
      raise newException(ValueError, op & " requires 2 arguments")
    let first = context.eval(args[0], prime, depth + 1).number(op).isZero.not
    if op == "and" and not first: return truth(false, prime)
    if op == "or" and first: return truth(true, prime)
    return truth(context.eval(args[1], prime, depth + 1).number(op).isZero.not, prime)
  else: discard
  let arity = case op
    of "add", "subtract", "multiply", "divide", "power", "solve", "mod", "gcd", "lcm", "hadamard", "kronecker",
       "eq", "ne", "lt", "le", "gt", "ge", "min", "max", "div", "binomial", "sigma", "root": 2
    of "negate", "transpose", "determinant", "inverse", "rank", "rref", "trace", "identity", "sqrt", "abs",
       "factorial", "not", "isprime", "mobius", "totient": 1
    of "powmod": 3
    else: raise newException(ValueError, "Unknown calculator operation: " & original)
  if args.isNil or args.kind != JArray or args.len != arity:
    raise newException(ValueError, op & " requires " & $arity & " arguments")
  let integerDomain = op in ["mod", "gcd", "lcm", "identity", "factorial", "abs", "lt", "le", "gt", "ge",
    "min", "max", "div", "binomial", "sigma", "isprime", "mobius", "totient", "powmod"]
  let a = context.eval(args[0], (if integerDomain: 0 else: prime), depth + 1)
  let b = if arity >= 2:
    context.eval(args[1], (if integerDomain or op in ["power", "root"]: 0 else: prime), depth + 1)
  else:
    default(Value)
  let c = if arity == 3: context.eval(args[2], 0, depth + 1) else: default(Value)
  let step = %*{"operation": op}
  case op
  of "eq", "ne":
    result = truth((a == b) == (op == "eq"), prime)
  of "lt", "le", "gt", "ge":
    let order = compareExact(a.number(op), b.number(op))
    let holds = case op
      of "lt": order < 0
      of "le": order <= 0
      of "gt": order > 0
      else: order >= 0
    result = truth(holds, prime)
  of "min", "max":
    let x = a.number(op)
    let y = b.number(op)
    result = scalar((if (compareExact(x, y) <= 0) == (op == "min"): x else: y).asField(prime))
  of "not":
    result = truth(a.number(op).isZero, prime)
  of "div":
    let x = a.number(op).integralBig(op)
    let y = b.number(op).integralBig(op)
    if y.isZero: raise newException(ValueError, "Division by zero")
    var (q, r) = divMod(x, y)
    if not r.isZero and (x.isNegative != y.isNegative): q = q - big(1)
    result = scalar(wholeBig(q).asField(prime))
  of "binomial":
    result = scalar(binomialExact(a.number(op).integral(op), b.number(op).integral(op), prime))
  of "powmod":
    let modulus = c.number(op).integralBig(op)
    let exponent = b.number(op).integralBig(op)
    result = scalar(wholeBig(powModBig(a.number(op).integralBig(op), exponent, modulus)).asField(prime))
  of "isprime":
    result = truth(isProbablePrimeBig(a.number(op).integralBig(op)), prime)
  of "mobius":
    let factors = factorize(a.number(op).integral(op))
    var value = if factors.len mod 2 == 0: 1'i64 else: -1'i64
    for factor in factors:
      if factor.exponent > 1: value = 0
    result = scalar(whole(value).asField(prime))
  of "totient":
    var value = One
    for (p, exponent) in factorize(a.number(op).integral(op)):
      value = value.times(whole(p - 1), 0)
      for _ in 2 .. exponent: value = value.times(whole(p), 0)
    result = scalar(value.asField(prime))
  of "sigma":
    result = scalar(divisorSigma(a.number(op).integral(op), b.number(op).integral(op)).asField(prime))
  of "add", "subtract", "hadamard":
    sameShape(a, b, op)
    result = a
    for i, entry in result.entries.mpairs:
      if op == "hadamard": entry = entry.times(b.entries[i], prime)
      else: entry = entry.plus((if op == "add": b.entries[i] else: b.entries[i].negative(prime)), prime)
  of "multiply": result = product(a, b, prime)
  of "divide": result = product(a, inverted(b, prime, step), prime)
  of "solve":
    a.square(op)
    if b.isScalar(): raise newException(ValueError, "Solve requires a matrix right hand side")
    result = product(inverted(a, prime, step), b, prime)
    if product(a, result, prime) != b: raise newException(ValueError, "Solution did not verify")
  of "negate":
    result = a
    for entry in result.entries.mitems: entry = entry.negative(prime)
  of "power": result = power(a, b.number(op).integral(op), prime, step)
  of "inverse": result = inverted(a, prime, step)
  of "transpose":
    if a.isScalar(): raise newException(ValueError, "Transpose requires a matrix")
    result = matrix(a.cols, a.rows)
    for r in 0 ..< a.rows:
      for c in 0 ..< a.cols: result.item(c, r) = a.item(r, c)
  of "trace":
    a.square(op)
    var sum = Zero
    for i in 0 ..< a.rows: sum = sum.plus(a.item(i, i), prime)
    result = scalar(sum)
  of "determinant", "rank", "rref":
    if op == "determinant": a.square(op)
    let reduced = eliminate(a, prime, trackDeterminant = op == "determinant")
    step["pivots"] = reduced.pivots
    if op == "determinant": result = scalar(reduced.determinant)
    elif op == "rank": result = scalar(whole(int64(reduced.rank)))
    else: result = reduced.reduced
  of "identity":
    let n = a.number(op).integral(op)
    if n notin 1'i64 .. int64(SystemMaxOrder): raise newException(ValueError, "Identity size is out of range")
    result = identity(int(n))
  of "sqrt":
    let value = a.number(op)
    if prime == 0:
      result = scalar(exactRoot(value))
    else:
      var root = 0
      let target = value.asField(prime).num
      while root < prime and productMod(root, root, prime) != target: inc root
      if root == prime: raise newException(ValueError, "No square root in the prime field")
      result = scalar(whole(int64(root)))
  of "root":
    let value = a.number(op)
    let k = b.number(op).integral(op)
    if prime == 0:
      if k notin 1'i64 .. int64(MaxRootDegree):
        raise newException(ValueError, "root degree must be between 1 and " & $MaxRootDegree)
      let top = perfectRoot(value.numerator, int(k))
      let bottom = perfectRoot(value.denominator, int(k))
      if not top.ok or not bottom.ok:
        raise newException(ValueError, "root(x," & $k & ") is not rational. Use the nth_root tool for decimal digits.")
      result = scalar(rationalBig(top.root, bottom.root))
    else:
      if k < 1: raise newException(ValueError, "root degree must be at least 1")
      let found = modularRoots(value.asField(prime).num, k, int64(prime), 1)
      if found.count == 0: raise newException(ValueError, "No " & $k & "-th root in the prime field")
      result = scalar(whole(found.roots[0]))
  of "abs":
    let value = a.number(op)
    result = scalar(value.absolute.asField(prime))
  of "mod", "gcd", "lcm":
    let x = a.number(op).integralBig(op)
    let y = b.number(op).integralBig(op)
    let value = case op
      of "gcd": gcdBig(x, y)
      of "lcm":
        if x.isZero or y.isZero: big(0)
        else: absBig(divMod(x, gcdBig(x, y)).quotient * y)
      else:
        if y.isNegative or y.isZero: raise newException(ValueError, "Modulo requires a positive modulus")
        floorMod(x, y)
    result = scalar(wholeBig(value).asField(prime))
  of "factorial":
    let n = a.number(op).integral(op)
    if n < 0 or n > (if prime == 0: int64(MaxExactFactorial) else: int64(ExactMaxPrime)):
      raise newException(ValueError, "Factorial argument must be between 0 and " &
        $(if prime == 0: int64(MaxExactFactorial) else: int64(ExactMaxPrime)))
    if prime == 0:
      result = scalar(wholeBig(productTree(2, n)))
    else:
      var value = One
      for i in 2'i64 .. n: value = value.times(whole(i).asField(prime), prime)
      result = scalar(value)
  of "kronecker":
    if a.isScalar() or b.isScalar(): raise newException(ValueError, "Kronecker product requires two matrices")
    result = matrix(a.rows * b.rows, a.cols * b.cols)
    for i in 0 ..< a.rows:
      for j in 0 ..< a.cols:
        for r in 0 ..< b.rows:
          for c in 0 ..< b.cols: result.item(i * b.rows + r, j * b.cols + c) = a.item(i, j).times(b.item(r, c), prime)
  else: raise newException(ValueError, "Unsupported calculator operation")
  if context.quiet == 0:
    step["inputs"] = case arity
      of 3: %[a.describe(), b.describe(), c.describe()]
      of 2: %[a.describe(), b.describe()]
      else: %[a.describe()]
    step["result"] = result.describe()
    context.steps.add(step)

proc calculateExact*(problem: JsonNode): JsonNode =
  let p = if problem.hasKey("prime"): readPrime(problem) else: 0
  let input = problem{"expression"}
  if input.isNil: raise newException(ValueError, "Expected an expression")
  let tree = if input.kind == JString: parse(input.getStr()) else: input
  let variables = problem{"variables"}
  if not variables.isNil and (variables.kind != JObject or variables.len > 64):
    raise newException(ValueError, "Expected at most 64 named scalar or matrix variables")
  var context = Evaluator(variables: (if variables.isNil: newJObject() else: variables), steps: newJArray())
  let answer = context.eval(tree, p, 0)
  let arithmetic = if p == 0: "exact rational arithmetic" else: "arithmetic modulo " & $p
  %*{"answer": answer.toJson(), "solution": "Evaluate the listed operations in order using " & arithmetic & ".",
     "method_steps": context.steps, "method": "exact", "verified": true,
     "certificate": {"arithmetic": arithmetic, "operations": context.steps.len}}

proc parseExpression*(source: string): JsonNode = parse(source)

proc evaluateScalar*(tree, variables: JsonNode, prime: int): Exact =
  ## Quiet evaluation for search tools. Loop variables are restored afterwards.
  var context = Evaluator(variables: variables, steps: newJArray(), quiet: 1)
  context.eval(tree, prime, 0).number("Expression")

proc expressionTree*(node: JsonNode, label: string): JsonNode =
  if node.isNil: raise newException(ValueError, "Missing " & label)
  if node.kind == JString: parse(node.getStr()) else: node

proc searchCounterexample*(problem: JsonNode): JsonNode =
  ## Check a predicate, or left = right, on every point of a small integer grid.
  const MaxCases = 2_000_000
  let predicate = problem{"predicate"}
  let left = problem{"left"}
  let right = problem{"right"}
  if predicate.isNil == (left.isNil or right.isNil):
    raise newException(ValueError, "Supply either predicate, or both left and right")
  let trees = if predicate.isNil: @[expressionTree(left, "left"), expressionTree(right, "right")]
              else: @[expressionTree(predicate, "predicate")]
  let ranges = problem{"variables"}
  if ranges.isNil or ranges.kind != JObject or ranges.len notin 1 .. 4:
    raise newException(ValueError, "variables maps one to four names to [first, last] ranges")
  var names: seq[string]
  var lows, highs: seq[int64]
  var total = 1'i64
  for name, bounds in ranges:
    if bounds.kind != JArray or bounds.len != 2 or bounds[0].kind != JInt or bounds[1].kind != JInt:
      raise newException(ValueError, "Each variable range must be [first, last] integers")
    let (lo, hi) = (bounds[0].getBiggestInt(), bounds[1].getBiggestInt())
    if hi < lo or hi - lo >= MaxCases: raise newException(ValueError, "Invalid range for " & name)
    names.add(name)
    lows.add(lo)
    highs.add(hi)
    total *= hi - lo + 1
    if total > MaxCases: raise newException(ValueError, "The grid exceeds " & $MaxCases & " cases")
  let limit = int(problem{"max_cases"}.getBiggestInt(MaxCases))
  let prime = if problem.hasKey("prime"): readPrime(problem) else: 0
  var variables = newJObject()
  let constants = problem{"constants"}
  if not constants.isNil:
    if constants.kind != JObject or constants.len > 64:
      raise newException(ValueError, "constants must be an object with at most 64 entries")
    for key, value in constants: variables[key] = value
  var current = lows
  var checked, skipped = 0
  var firstSkip = ""
  var counterexamples = newJArray()
  while checked + skipped < min(int(total), limit):
    for i, name in names: variables[name] = %current[i]
    try:
      let values = trees.mapIt(evaluateScalar(it, variables, prime))
      let holds = if values.len == 1: not values[0].isZero else: values[0] == values[1]
      inc checked
      if not holds and counterexamples.len < 5:
        let assignment = newJObject()
        for i, name in names: assignment[name] = %current[i]
        let example = %*{"assignment": assignment}
        if values.len == 2:
          example["left"] = values[0].toJson()
          example["right"] = values[1].toJson()
        counterexamples.add(example)
    except ValueError as error:
      inc skipped
      if firstSkip.len == 0: firstSkip = error.msg
    var digit = names.high
    while digit >= 0:
      if current[digit] < highs[digit]:
        inc current[digit]
        break
      current[digit] = lows[digit]
      dec digit
    if digit < 0: break
  result = %*{"holds": counterexamples.len == 0 and checked > 0, "cases_checked": checked,
              "counterexamples": counterexamples, "verified": true, "method": "exhaustive_grid",
              "solution": "Every grid point was evaluated exactly. A finite check is evidence, not a proof.",
              "certificate": {"grid_size": total, "arithmetic": (if prime == 0: "exact rational" else: "modulo " & $prime)}}
  if skipped > 0: result["skipped"] = %*{"count": skipped, "first_reason": firstSkip}
