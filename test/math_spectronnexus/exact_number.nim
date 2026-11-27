## Exact rational arithmetic for the native calculator, in tiers. A value starts as an
## int64 fraction. An operation that would overflow moves to arbitrary precision
## integers (schoolbook, then Karatsuba products), and a result that fits again comes
## back to the int64 tier. Nothing is rounded and nothing stops at 64 bits.
import std/[json, math, strutils]
import big_integer, prime_field, root_math

const
  Limit = high(int64)
  JsonSafe = 9_007_199_254_740_991'i64
  MaxDecimalExponent = 1_000_000

type Exact* = object
  n, d: int64        ## Small tier: the value n/d with d > 0 and gcd(n, d) = 1.
  wide: bool
  bn, bd: BigInt     ## Wide tier, used only when n/d does not fit in int64.

const Zero* = Exact(n: 0, d: 1)
const One* = Exact(n: 1, d: 1)

proc checkedInt*(value: int64): int64 =
  if value == low(int64): raise newException(ValueError, "Integer is outside the signed 64 bit range")
  value

proc addOk(a, b: int64, r: var int64): bool {.inline.} =
  if (b > 0 and a > Limit - b) or (b < 0 and a < -Limit - b): return false
  r = a + b
  true

proc mulOk(a, b: int64, r: var int64): bool {.inline.} =
  if a != 0 and abs(b) > Limit div abs(a): return false
  r = a * b
  true

proc isWide*(x: Exact): bool = x.wide
proc isZero*(x: Exact): bool = not x.wide and x.n == 0
proc isInteger*(x: Exact): bool = (if x.wide: x.bd == big(1) else: x.d == 1)
proc isNegative*(x: Exact): bool = (if x.wide: x.bn.isNegative else: x.n < 0)
proc numerator*(x: Exact): BigInt = (if x.wide: x.bn else: big(x.n))
proc denominator*(x: Exact): BigInt = (if x.wide: x.bd else: big(x.d))

proc num*(x: Exact): int64 =
  ## The numerator when the value is in the int64 tier.
  if x.wide: raise newException(ValueError, "This step needs a numerator within 64 bits")
  x.n

proc den*(x: Exact): int64 =
  if x.wide: raise newException(ValueError, "This step needs a denominator within 64 bits")
  x.d

proc rationalBig*(numerator, denominator: BigInt): Exact =
  ## Reduce and pick the tier.
  if denominator.isZero: raise newException(ValueError, "Division by zero")
  var (top, bottom) = (numerator, denominator)
  if bottom.isNegative: (top, bottom) = (-top, -bottom)
  let g = gcdBig(top, bottom)
  if not (g == big(1)):
    top = divMod(top, g).quotient
    bottom = divMod(bottom, g).quotient
  if top.fitsInt64() and bottom.fitsInt64(): return Exact(n: top.toInt64(), d: bottom.toInt64())
  Exact(wide: true, bn: top, bd: bottom)

proc wholeBig*(value: BigInt): Exact = rationalBig(value, big(1))

proc rational*(num, den: int64): Exact =
  if den == 0: raise newException(ValueError, "Division by zero")
  if num == low(int64) or den == low(int64): return rationalBig(big(num), big(den))
  let factor = gcd(abs(num), abs(den))
  let sign = if den < 0: -1'i64 else: 1'i64
  Exact(n: sign * (num div factor), d: abs(den) div factor)

proc whole*(value: int64): Exact =
  if value == low(int64): wholeBig(big(value)) else: Exact(n: value, d: 1)

proc residueOf(x: BigInt, p: int): int = int(x.modSmall(int64(p)))

proc smallResidue(x: Exact, p: int): int =
  ## The residue of an integer value modulo p, from either tier.
  if x.wide: residueOf(x.bn, p) else: residue(x.n, p)

proc negative*(value: Exact, p: int): Exact =
  if p != 0: return whole(int64(residue(-value.smallResidue(p), p)))
  if value.wide: Exact(wide: true, bn: -value.bn, bd: value.bd) else: Exact(n: -value.n, d: value.d)

proc plus*(a, b: Exact, p: int): Exact =
  if p != 0: return whole(int64(residue(a.smallResidue(p) + b.smallResidue(p), p)))
  if not a.wide and not b.wide:
    let common = gcd(a.d, b.d)
    var x, y, top, bottom: int64
    if mulOk(a.n, b.d div common, x) and mulOk(b.n, a.d div common, y) and addOk(x, y, top):
      let cancel = gcd(abs(top), common)
      if mulOk(a.d div common, b.d div cancel, bottom):
        return Exact(n: top div cancel, d: bottom)
  rationalBig(a.numerator * b.denominator + b.numerator * a.denominator, a.denominator * b.denominator)

proc times*(a, b: Exact, p: int): Exact =
  if p != 0: return whole(int64(productMod(a.smallResidue(p), b.smallResidue(p), p)))
  if not a.wide and not b.wide:
    let g1 = gcd(abs(a.n), b.d)
    let g2 = gcd(abs(b.n), a.d)
    var top, bottom: int64
    if g1 != 0 and g2 != 0 and mulOk(a.n div g1, b.n div g2, top) and mulOk(a.d div g2, b.d div g1, bottom):
      return Exact(n: top, d: bottom)
    if a.n == 0 or b.n == 0: return Zero
  rationalBig(a.numerator * b.numerator, a.denominator * b.denominator)

proc inverse*(value: Exact, p: int): Exact =
  if p != 0:
    let canonical = value.smallResidue(p)
    if canonical == 0: raise newException(ValueError, "Division by zero modulo the prime")
    return whole(int64(reciprocal(canonical, p)))
  if value.isZero: raise newException(ValueError, "Division by zero")
  if value.wide: rationalBig(value.bd, value.bn) else: rational(value.d, value.n)

proc asField*(value: Exact, p: int): Exact =
  if p == 0: return value
  let divisor = if value.wide: residueOf(value.bd, p) else: residue(value.d, p)
  if divisor == 0: raise newException(ValueError, "The denominator is zero modulo the prime")
  let top = if value.wide: residueOf(value.bn, p) else: residue(value.n, p)
  whole(int64(productMod(top, reciprocal(divisor, p), p)))

proc compare*(a, b: Exact): int =
  if not a.wide and not b.wide and a.d == 1 and b.d == 1: return cmp(a.n, b.n)
  cmp(a.numerator * b.denominator, b.numerator * a.denominator)

proc absolute*(value: Exact): Exact =
  if value.wide: Exact(wide: true, bn: absBig(value.bn), bd: value.bd)
  else: Exact(n: abs(value.n), d: value.d)

proc parseDecimal(source: string): Exact =
  let parts = source.toLowerAscii().split('e')
  if parts.len > 2: raise newException(ValueError, "Invalid number " & source)
  let exponent = if parts.len == 2: parseBiggestInt(parts[1]) else: 0'i64
  if exponent notin -MaxDecimalExponent.int64 .. MaxDecimalExponent.int64:
    raise newException(ValueError, "Decimal exponent is out of range")
  var significand = parts[0]
  let negative = significand.startsWith("-")
  if negative or significand.startsWith("+"): significand = significand[1 .. ^1]
  let decimal = significand.find('.')
  var digits = significand.replace(".", "")
  if digits.len == 0 or (decimal >= 0 and significand.count('.') != 1):
    raise newException(ValueError, "Invalid number " & source)
  for digit in digits:
    if digit notin {'0'..'9'}: raise newException(ValueError, "Invalid number " & source)
  var scale = int(exponent)
  if decimal >= 0: scale -= significand.len - decimal - 1
  while digits.len > 0 and digits[^1] == '0':
    digits.setLen(digits.len - 1)
    inc scale
  if digits.len == 0: return Zero
  let num = parseBig((if negative: "-" else: "") & digits)
  let power = power(big(10), abs(scale))
  if scale >= 0: wholeBig(num * power) else: rationalBig(num, power)

proc parseExact*(node: JsonNode): Exact =
  if node.isNil: raise newException(ValueError, "Missing number")
  case node.kind
  of JInt: whole(node.getBiggestInt())
  of JString:
    let literal = node.getStr().strip()
    let fraction = literal.split('/')
    if fraction.len == 2:
      rationalBig(parseBig(fraction[0].strip()), parseBig(fraction[1].strip()))
    elif fraction.len == 1:
      parseDecimal(literal)
    else:
      raise newException(ValueError, "Invalid rational " & literal)
  else:
    raise newException(ValueError, "Use an integer or quote the exact decimal or fraction")

proc integral*(value: Exact, operation: string): int64 =
  ## An integer argument that drives a loop, a size or a count, so it must fit in int64.
  if not value.isInteger: raise newException(ValueError, operation & " requires integers")
  if value.wide: raise newException(ValueError, operation & " needs an integer argument within 64 bits")
  value.n

proc integralBig*(value: Exact, operation: string): BigInt =
  if not value.isInteger: raise newException(ValueError, operation & " requires integers")
  value.numerator

proc lcmExact*(a, b: int64): int64 =
  if a == 0 or b == 0: return 0
  let step = abs(a) div gcd(abs(a), abs(b))
  var product: int64
  if not mulOk(step, abs(b), product): raise newException(ValueError, "Integer is outside the signed 64 bit range")
  product

proc exactRoot*(value: Exact): Exact =
  if value.isNegative: raise newException(ValueError, "Square root is outside the rational domain")
  let top = perfectRoot(value.numerator, 2)
  let bottom = perfectRoot(value.denominator, 2)
  if not top.ok or not bottom.ok: raise newException(ValueError, "Square root is not an exact rational")
  rationalBig(top.root, bottom.root)

proc toJson*(value: Exact): JsonNode =
  if value.wide:
    if value.bd == big(1): return %($value.bn)
    return %($value.bn & "/" & $value.bd)
  if value.d != 1: return %($value.n & "/" & $value.d)
  if abs(value.n) > JsonSafe: %($value.n) else: %value.n
