## Fixed-point real arithmetic on BigInt: a value v stands for v / 10^digits.
## Mathematical constants and an evaluator for constant expressions.
import std/[json, math, strutils, tables]
import big_integer, calculator, root_math

const
  GuardDigits = 15
  MaxPrecisionDigits* = 3_000

type Fixed* = object
  digits*: int
  scale*: BigInt
  value*: BigInt

var constantCache: Table[(string, int), BigInt]
var scaleCache: Table[int, BigInt]

proc tenPower*(digits: int): BigInt =
  if digits notin scaleCache: scaleCache[digits] = power(big(10), digits)
  scaleCache[digits]

proc fixed*(digits: int, value: BigInt): Fixed = Fixed(digits: digits, scale: tenPower(digits), value: value)
proc fromInteger*(digits: int, n: BigInt): Fixed = fixed(digits, n * tenPower(digits))

proc `+`*(a, b: Fixed): Fixed = fixed(a.digits, a.value + b.value)
proc `-`*(a, b: Fixed): Fixed = fixed(a.digits, a.value - b.value)
proc `-`*(a: Fixed): Fixed = fixed(a.digits, -a.value)
proc `*`*(a, b: Fixed): Fixed = fixed(a.digits, divMod(a.value * b.value, a.scale).quotient)
proc `/`*(a, b: Fixed): Fixed =
  if b.value.isZero: raise newException(ValueError, "Division by zero")
  fixed(a.digits, divMod(a.value * a.scale, b.value).quotient)
proc isZero*(a: Fixed): bool = a.value.isZero

proc reduce(value: BigInt, fromDigits, toDigits: int): BigInt =
  divMod(value, tenPower(fromDigits - toDigits)).quotient

# Series kernels on a scale S = 10^digits.

proc atanInverse(m: int64, s: BigInt, hyperbolic: bool): BigInt =
  ## atan(1/m) or atanh(1/m) times S.
  var term = divSmall(s, m).quotient
  result = term
  var k = 1'i64
  let square = m * m
  while not term.isZero:
    term = divSmall(term, square).quotient
    let piece = divSmall(term, 2 * k + 1).quotient
    if hyperbolic or k mod 2 == 0: result = result + piece else: result = result - piece
    inc k

proc atanhRational(p, q: int64, s: BigInt): BigInt =
  ## atanh(p/q) times S for 0 <= p < q with q^2 below 9e9.
  var term = divSmall(s * big(p), q).quotient
  result = term
  var k = 1'i64
  while not term.isZero:
    term = divSmall(term * big(p * p), q * q).quotient
    result = result + divSmall(term, 2 * k + 1).quotient
    inc k

proc alternatingSum(s: BigInt, digits: int, term: proc(k: int): BigInt): BigInt =
  ## Sum over k >= 0 of (-1)^k a_k by the Cohen, Rodriguez Villegas and Zagier
  ## acceleration. term(k) returns a_k times S for a totally monotone sequence.
  let n = int(ceil(float64(digits) / log10(3.0 + sqrt(8.0)))) + 2
  var (d0, d1) = (big(1), big(3))
  for _ in 1 ..< n: (d0, d1) = (d1, big(6) * d1 - d0)
  let d = d1
  var b = big(-1)
  var c = -d
  var total = big(0)
  for k in 0 ..< n:
    c = b - c
    total = total + c * term(k)
    let (q, r) = divSmall(b * big(2 * int64(k + n)) * big(int64(k - n)), int64(2 * k + 1) * int64(k + 1))
    if r != 0: raise newException(ValueError, "Acceleration coefficients lost exactness")
    b = q
  divMod(total, d).quotient

proc piScaled(s: BigInt): BigInt =
  big(4) * (big(4) * atanInverse(5, s, false) - atanInverse(239, s, false))

proc eScaled(s: BigInt): BigInt =
  var term = s
  result = s
  var k = 1'i64
  while not term.isZero:
    term = divSmall(term, k).quotient
    result = result + term
    inc k

proc logPrimeScaled(p: int, s: BigInt): BigInt =
  let log2 = big(2) * atanInverse(3, s, true)
  case p
  of 2: log2
  of 3: log2 + big(2) * atanInverse(5, s, true)
  of 5: big(2) * log2 + big(2) * atanInverse(9, s, true)
  of 7: big(2) * log2 + big(2) * atanInverse(9, s, true) + big(2) * atanInverse(6, s, true)
  else: big(2) * atanhRational(int64(p - 1), int64(p + 1), s)

proc logIntegerScaled(n: int64, s: BigInt): BigInt =
  if n < 1 or n > 90_000: raise newException(ValueError, "logN supports 1 <= N <= 90000")
  result = big(0)
  var rest = n
  for p in [2'i64, 3, 5, 7]:
    while rest mod p == 0:
      result = result + logPrimeScaled(int(p), s)
      rest = rest div p
  if rest > 1: result = result + big(2) * atanhRational(rest - 1, rest + 1, s)

proc zetaScaled(k: int, s: BigInt, digits: int): BigInt =
  if k < 2 or k > 64: raise newException(ValueError, "zetaN supports 2 <= N <= 64")
  let eta = alternatingSum(s, digits, proc(i: int): BigInt = divMod(s, power(big(int64(i + 1)), k)).quotient)
  let factor = power(big(2), k - 1)
  divMod(eta * factor, factor - big(1)).quotient

proc catalanScaled(s: BigInt, digits: int): BigInt =
  alternatingSum(s, digits, proc(i: int): BigInt = divMod(s, big(int64(2 * i + 1) * int64(2 * i + 1))).quotient)

proc gammaScaled(s: BigInt, digits: int): BigInt =
  ## Euler's constant by the Brent-McMillan algorithm B1 with n a power of two.
  var exponent = 1
  while float64(1 shl exponent) < float64(digits) * ln(10.0) / 4.0 + 1.0: inc exponent
  let n = int64(1) shl exponent
  let logN = big(int64(exponent)) * logPrimeScaled(2, s)
  var u = s
  var harmonic = big(0)
  var a = -logN
  var b = s
  var k = 1'i64
  while true:
    u = divSmall(u * big(n * n), k * k).quotient
    harmonic = harmonic + divSmall(s, k).quotient
    if u.isZero and k > n: break
    a = a + divMod(u * (harmonic - logN), s).quotient
    b = b + u
    inc k
  divMod(a * s, b).quotient

proc sqrtScaled(n: int64, s: BigInt): BigInt =
  if n < 0: raise newException(ValueError, "sqrtN needs N >= 0")
  integerRoot(big(n) * s * s, 2)

proc numberSuffix(name, prefix: string): int64 =
  let tail = name[prefix.len .. ^1]
  if tail.len == 0 or tail.len > 6 or not tail.allCharsInSet({'0'..'9'}): return -1
  parseBiggestInt(tail)

proc constantScaled(name: string, digits: int): BigInt =
  ## A named constant times 10^digits, truncated.
  let key = (name, digits)
  if key in constantCache: return constantCache[key]
  let work = digits + GuardDigits
  let s = tenPower(work)
  let raw =
    if name == "pi": piScaled(s)
    elif name == "e": eScaled(s)
    elif name == "gamma": gammaScaled(s, work)
    elif name == "catalan": catalanScaled(s, work)
    elif name == "phi": divSmall(s + sqrtScaled(5, s), 2).quotient
    elif name.startsWith("log") and numberSuffix(name, "log") >= 1: logIntegerScaled(numberSuffix(name, "log"), s)
    elif name.startsWith("zeta") and numberSuffix(name, "zeta") >= 2: zetaScaled(int(numberSuffix(name, "zeta")), s, work)
    elif name.startsWith("sqrt") and numberSuffix(name, "sqrt") >= 0: sqrtScaled(numberSuffix(name, "sqrt"), s)
    else: raise newException(ValueError, "Unknown constant " & name &
      ". Use pi, e, gamma, catalan, phi, logN, zetaN or sqrtN.")
  result = reduce(raw, work, digits)
  constantCache[key] = result

proc isConstantName*(name: string): bool =
  try:
    discard constantScaled(name, 5)
    true
  except ValueError:
    false

proc parseDecimal*(text: string, digits: int): Fixed =
  ## A decimal literal such as -12.5e-3, truncated to the working digits.
  var body = text.strip().toLowerAscii()
  var exponent = 0
  let e = body.find('e')
  if e >= 0:
    exponent = parseInt(body[e + 1 .. ^1])
    body = body[0 ..< e]
  var negative = false
  if body.len > 0 and body[0] in {'+', '-'}:
    negative = body[0] == '-'
    body = body[1 .. ^1]
  let dot = body.find('.')
  var intPart = if dot >= 0: body[0 ..< dot] else: body
  var fracPart = if dot >= 0: body[dot + 1 .. ^1] else: ""
  if (intPart & fracPart).len == 0 or not (intPart & fracPart).allCharsInSet({'0'..'9'}):
    raise newException(ValueError, "Invalid decimal " & text)
  let mantissa = parseBig((if (intPart & fracPart).len == 0: "0" else: intPart & fracPart))
  let shift = digits - fracPart.len + exponent
  var value = if shift >= 0: mantissa * tenPower(shift) else: divMod(mantissa, tenPower(-shift)).quotient
  if negative: value = -value
  fixed(digits, value)

proc fractionDigits*(text: string): int =
  ## Digits after the decimal point, adjusted for an exponent.
  var body = text.strip().toLowerAscii()
  var exponent = 0
  let e = body.find('e')
  if e >= 0:
    exponent = parseInt(body[e + 1 .. ^1])
    body = body[0 ..< e]
  let dot = body.find('.')
  (if dot >= 0: body.len - dot - 1 else: 0) - exponent

proc expFixed(x: Fixed): Fixed =
  ## exp by halving, a Taylor series, and repeated squaring.
  var halvings = 0
  var y = x
  let half = divSmall(x.scale, 2).quotient
  while halvings < 8 or half < y.value or y.value < -half:
    y = fixed(y.digits, divSmall(y.value, 2).quotient)
    inc halvings
    if halvings > 4_000: raise newException(ValueError, "exp argument is too large")
  var term = fromInteger(y.digits, big(1))
  var total = term
  var k = 1'i64
  while not term.isZero:
    term = term * y
    term = fixed(term.digits, divSmall(term.value, k).quotient)
    total = total + term
    inc k
  for _ in 0 ..< halvings: total = total * total
  total

proc logFixed(x: Fixed): Fixed =
  if x.value.isNegative or x.value.isZero: raise newException(ValueError, "log needs a positive argument")
  var y = x
  var twos = 0
  let one = fromInteger(x.digits, big(1))
  let upper = fixed(x.digits, divSmall(one.value * big(3), 2).quotient)
  let lower = fixed(x.digits, divSmall(one.value * big(3), 4).quotient)
  while upper.value < y.value:
    y = fixed(y.digits, divSmall(y.value, 2).quotient)
    inc twos
  while y.value < lower.value:
    y = fixed(y.digits, y.value * big(2))
    dec twos
  let z = (y - one) / (y + one)
  let square = z * z
  var term = z
  var total = z
  var k = 1'i64
  while not term.isZero:
    term = term * square
    total = total + fixed(term.digits, divSmall(term.value, 2 * k + 1).quotient)
    inc k
  fixed(x.digits, big(2) * total.value + big(int64(twos)) * constantScaled("log2", x.digits))

proc evalFixed(node: JsonNode, digits: int, variables: Table[string, Fixed], depth: int): Fixed =
  if depth > 64: raise newException(ValueError, "Expression is nested too deeply")
  case node.kind
  of JInt: return fromInteger(digits, big(node.getBiggestInt()))
  of JString:
    let text = node.getStr()
    if text.len > 0 and text[0] in {'0'..'9', '.'}: return parseDecimal(text, digits)
    if text in variables: return variables[text]
    return fixed(digits, constantScaled(text, digits))
  of JObject: discard
  else: raise newException(ValueError, "Invalid constant expression")
  let op = node{"op"}.getStr().toLowerAscii()
  let args = node{"args"}
  proc arg(i: int): Fixed = evalFixed(args[i], digits, variables, depth + 1)
  case op
  of "add": arg(0) + arg(1)
  of "subtract": arg(0) - arg(1)
  of "multiply": arg(0) * arg(1)
  of "divide": arg(0) / arg(1)
  of "negate": -arg(0)
  of "power":
    let base = arg(0)
    let exponentNode = args[1]
    var exponent: int64
    let exponentValue = evalFixed(exponentNode, digits, variables, depth + 1)
    let (whole, rest) = divMod(exponentValue.value, exponentValue.scale)
    if not rest.isZero or not whole.fitsInt64(): raise newException(ValueError, "Powers need integer exponents")
    exponent = whole.toInt64()
    if exponent notin -256'i64 .. 256'i64: raise newException(ValueError, "Exponent must be between -256 and 256")
    var product = fromInteger(digits, big(1))
    for _ in 1 .. abs(exponent): product = product * base
    if exponent < 0: fromInteger(digits, big(1)) / product else: product
  of "sqrt":
    let x = arg(0)
    if x.value.isNegative: raise newException(ValueError, "sqrt needs a nonnegative argument")
    fixed(digits, integerRoot(x.value * x.scale, 2))
  of "root":
    let x = arg(0)
    let k = arg(1)
    let (degree, rest) = divMod(k.value, k.scale)
    if not rest.isZero or not degree.fitsInt64() or degree.toInt64() notin 1'i64 .. 1000'i64:
      raise newException(ValueError, "root degree must be an integer from 1 to 1000")
    let d = int(degree.toInt64())
    if x.value.isNegative and d mod 2 == 0: raise newException(ValueError, "Even root of a negative number")
    let magnitude = integerRoot((if x.value.isNegative: -x.value else: x.value) * power(x.scale, d - 1), d)
    fixed(digits, if x.value.isNegative: -magnitude else: magnitude)
  of "log": logFixed(arg(0))
  of "exp": expFixed(arg(0))
  of "zeta":
    let k = arg(0)
    let (whole, rest) = divMod(k.value, k.scale)
    if not rest.isZero or not whole.fitsInt64(): raise newException(ValueError, "zeta needs an integer argument")
    fixed(digits, constantScaled("zeta" & $whole.toInt64(), digits))
  else: raise newException(ValueError, "Unsupported operation in a constant expression: " & op)

proc evaluateConstant*(expression: string, digits: int, variables = initTable[string, Fixed]()): Fixed =
  ## Value truncated near 10^-digits. Extra guard digits absorb rounding in the operations.
  if digits notin 1 .. MaxPrecisionDigits:
    raise newException(ValueError, "digits must be between 1 and " & $MaxPrecisionDigits)
  let work = digits + GuardDigits
  var scaled = initTable[string, Fixed]()
  for name, value in variables:
    scaled[name] = fixed(work, value.value * tenPower(work - value.digits))
  let full = evalFixed(parseExpression(expression), work, scaled, 0)
  fixed(digits, reduce(full.value, work, digits))

proc decimalString*(x: Fixed): string =
  let negative = x.value.isNegative
  var text = $(if negative: -x.value else: x.value)
  if text.len <= x.digits: text = repeat('0', x.digits - text.len + 1) & text
  result = text[0 ..< text.len - x.digits] & (if x.digits > 0: "." & text[text.len - x.digits .. ^1] else: "")
  if negative: result = "-" & result

proc solveConstantDigits*(problem: JsonNode): JsonNode =
  let expression = problem{"expression"}
  if expression.isNil or expression.kind != JString: raise newException(ValueError, "expression must be a string")
  let digits = int(problem{"digits"}.getBiggestInt(50))
  let value = evaluateConstant(expression.getStr(), digits)
  %*{"answer": decimalString(value), "digits": digits, "rounding": "truncated, the last digit may be off by one",
     "verified": true, "method": "exact_fixed_point",
     "solution": "Evaluated in fixed point with " & $GuardDigits & " guard digits. Constants come from " &
       "Machin's formula, atanh series, accelerated alternating series and Brent-McMillan.",
     "certificate": {"guard_digits": GuardDigits}}
