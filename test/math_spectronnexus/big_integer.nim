## Signed arbitrary precision integers for exact answers beyond 64 bits.
import std/[json, math, strutils]

const
  LimbBase = 1_000_000_000'i64
  LimbDigits = 9
  MaxBigDigits* = 10_000_000          ## Memory bound only: about 9 MB per value.
  KaratsubaLimbs = 48                  ## Products of longer operands switch to Karatsuba.
  MaxSmallDivisor* = 9_000_000_000'i64
  JsonSafeInteger = 9_007_199_254_740_991'i64

type BigInt* = object
  negative: bool
  limbs: seq[int64]   # little endian, base 10^9, no leading zero limbs

proc trim(x: var BigInt) =
  while x.limbs.len > 0 and x.limbs[^1] == 0: x.limbs.setLen(x.limbs.len - 1)
  if x.limbs.len == 0: x.negative = false

proc checkSize(x: BigInt) =
  if x.limbs.len * LimbDigits > MaxBigDigits + LimbDigits:
    raise newException(ValueError, "Integer exceeds " & $MaxBigDigits & " decimal digits")

proc isZero*(x: BigInt): bool = x.limbs.len == 0
proc isNegative*(x: BigInt): bool = x.negative

proc big*(value: int64): BigInt =
  var magnitude = if value < 0: uint64(not value) + 1 else: uint64(value)
  result.negative = value < 0
  while magnitude > 0:
    result.limbs.add(int64(magnitude mod uint64(LimbBase)))
    magnitude = magnitude div uint64(LimbBase)

proc parseBig*(source: string): BigInt =
  var text = source.strip()
  var negative = false
  if text.len > 0 and text[0] in {'+', '-'}:
    negative = text[0] == '-'
    text = text[1 .. ^1]
  if text.len == 0 or text.len > MaxBigDigits:
    raise newException(ValueError, "Invalid integer length")
  for ch in text:
    if ch notin {'0'..'9'}: raise newException(ValueError, "Invalid integer " & source)
  var stop = text.len
  while stop > 0:
    let start = max(0, stop - LimbDigits)
    result.limbs.add(parseBiggestInt(text[start ..< stop]))
    stop = start
  result.negative = negative
  result.trim()

proc toBig*(node: JsonNode, label: string): BigInt =
  if node.isNil: raise newException(ValueError, "Missing integer " & label)
  case node.kind
  of JInt: big(node.getBiggestInt())
  of JString: parseBig(node.getStr())
  else: raise newException(ValueError, label & " must be an integer or a decimal string")

proc `$`*(x: BigInt): string =
  if x.isZero: return "0"
  if x.negative: result.add('-')
  result.add($x.limbs[^1])
  for i in countdown(x.limbs.high - 1, 0):
    result.add(align($x.limbs[i], LimbDigits, '0'))

proc digits*(x: BigInt): int =
  if x.isZero: 1 else: (x.limbs.len - 1) * LimbDigits + len($x.limbs[^1])

proc cmpAbs(a, b: seq[int64]): int =
  if a.len != b.len: return cmp(a.len, b.len)
  for i in countdown(a.high, 0):
    if a[i] != b[i]: return cmp(a[i], b[i])
  0

proc cmp*(a, b: BigInt): int =
  if a.negative != b.negative: return (if a.negative: -1 else: 1)
  result = cmpAbs(a.limbs, b.limbs)
  if a.negative: result = -result

proc `==`*(a, b: BigInt): bool = a.negative == b.negative and a.limbs == b.limbs
proc `<`*(a, b: BigInt): bool = cmp(a, b) < 0
proc `<=`*(a, b: BigInt): bool = cmp(a, b) <= 0

proc addAbs(a, b: seq[int64]): seq[int64] =
  result = newSeq[int64](max(a.len, b.len) + 1)
  var carry = 0'i64
  for i in 0 ..< result.len:
    var total = carry
    if i < a.len: total += a[i]
    if i < b.len: total += b[i]
    result[i] = total mod LimbBase
    carry = total div LimbBase

proc subAbs(a, b: seq[int64]): seq[int64] =
  ## Requires |a| >= |b|.
  result = newSeq[int64](a.len)
  var borrow = 0'i64
  for i in 0 ..< a.len:
    var total = a[i] - borrow
    if i < b.len: total -= b[i]
    borrow = 0
    if total < 0:
      total += LimbBase
      borrow = 1
    result[i] = total

proc `-`*(a: BigInt): BigInt =
  result = a
  if not result.isZero: result.negative = not a.negative

proc `+`*(a, b: BigInt): BigInt =
  if a.negative == b.negative:
    result.limbs = addAbs(a.limbs, b.limbs)
    result.negative = a.negative
  elif cmpAbs(a.limbs, b.limbs) >= 0:
    result.limbs = subAbs(a.limbs, b.limbs)
    result.negative = a.negative
  else:
    result.limbs = subAbs(b.limbs, a.limbs)
    result.negative = b.negative
  result.trim()
  result.checkSize()

proc `-`*(a, b: BigInt): BigInt = a + (-b)

proc schoolbook(a, b: openArray[int64]): seq[int64] =
  ## Magnitude product in O(len(a) * len(b)), normalised to base 10^9 limbs.
  var acc = newSeq[uint64](a.len + b.len + 1)
  let base = uint64(LimbBase)
  for i, x in a:
    if x == 0: continue
    var carry = 0'u64
    for j, y in b:
      let current = acc[i + j] + uint64(x) * uint64(y) + carry
      acc[i + j] = current mod base
      carry = current div base
    var k = i + b.len
    while carry > 0:
      let current = acc[k] + carry
      acc[k] = current mod base
      carry = current div base
      inc k
  result = newSeq[int64](acc.len)
  for i, value in acc: result[i] = int64(value)

proc trimmed(x: seq[int64]): seq[int64] =
  result = x
  while result.len > 0 and result[^1] == 0: result.setLen(result.len - 1)

proc addInto(target: var seq[int64], source: openArray[int64], shift: int) =
  ## target += source * base^shift on normalised magnitudes.
  if target.len < source.len + shift + 1: target.setLen(source.len + shift + 1)
  var carry = 0'i64
  var i = 0
  while i < source.len or carry != 0:
    if shift + i >= target.len: target.add(0)
    var total = target[shift + i] + carry
    if i < source.len: total += source[i]
    target[shift + i] = total mod LimbBase
    carry = total div LimbBase
    inc i

proc karatsuba(a, b: seq[int64]): seq[int64] =
  ## Magnitude product: schoolbook for short operands, otherwise three half-size products.
  if a.len == 0 or b.len == 0: return
  if min(a.len, b.len) < KaratsubaLimbs: return schoolbook(a, b).trimmed()
  let half = max(a.len, b.len) div 2
  let a0 = a[0 ..< min(half, a.len)].trimmed()
  let a1 = (if a.len > half: a[half .. ^1] else: @[]).trimmed()
  let b0 = b[0 ..< min(half, b.len)].trimmed()
  let b1 = (if b.len > half: b[half .. ^1] else: @[]).trimmed()
  let low = karatsuba(a0, b0)
  let high = karatsuba(a1, b1)
  let middle = subAbs(subAbs(karatsuba(addAbs(a0, a1).trimmed(), addAbs(b0, b1).trimmed()).trimmed(),
                             low).trimmed(), high).trimmed()
  result = low
  result.addInto(middle, half)
  result.addInto(high, 2 * half)
  result = result.trimmed()

proc `*`*(a, b: BigInt): BigInt =
  ## Two tiers: schoolbook multiplication, and Karatsuba once both operands pass
  ## KaratsubaLimbs limbs, so products of million-digit values stay practical.
  if a.isZero or b.isZero: return
  if (a.limbs.len + b.limbs.len - 1) * LimbDigits > MaxBigDigits + LimbDigits:
    raise newException(ValueError, "Integer exceeds " & $MaxBigDigits & " decimal digits")
  result.limbs = if min(a.limbs.len, b.limbs.len) < KaratsubaLimbs: schoolbook(a.limbs, b.limbs)
                 else: karatsuba(a.limbs, b.limbs)
  result.negative = a.negative != b.negative
  result.trim()

proc divSmall*(a: BigInt, divisor: int64): tuple[quotient: BigInt, remainder: int64] =
  ## Truncated division. The remainder has the sign of a.
  if divisor == 0: raise newException(ValueError, "Division by zero")
  let d = abs(divisor)
  if d > MaxSmallDivisor: raise newException(ValueError, "Divisor exceeds the supported size")
  result.quotient.limbs = newSeq[int64](a.limbs.len)
  var remainder = 0'i64
  for i in countdown(a.limbs.high, 0):
    let current = remainder * LimbBase + a.limbs[i]
    result.quotient.limbs[i] = current div d
    remainder = current mod d
  result.quotient.negative = a.negative != (divisor < 0)
  result.quotient.trim()
  result.remainder = if a.negative: -remainder else: remainder

proc mulSmallAbs(a: seq[int64], factor: int64): seq[int64] =
  ## |a| * factor for 0 <= factor < LimbBase.
  result = newSeq[int64](a.len + 1)
  var carry = 0'i64
  for i, x in a:
    let current = x * factor + carry
    result[i] = current mod LimbBase
    carry = current div LimbBase
  result[a.len] = carry

proc divModAbs(a, b: seq[int64]): tuple[quotient, remainder: seq[int64]] =
  ## Knuth's algorithm D on base 10^9 magnitudes. b has at least two limbs.
  let n = b.len
  let m = a.len - n
  let scale = LimbBase div (b[^1] + 1)
  var u = mulSmallAbs(a, scale)
  let v = mulSmallAbs(b, scale)[0 ..< n]
  result.quotient = newSeq[int64](m + 1)
  for j in countdown(m, 0):
    let top = u[j + n] * LimbBase + u[j + n - 1]
    var qhat = top div v[n - 1]
    var rhat = top mod v[n - 1]
    while qhat >= LimbBase or qhat * v[n - 2] > rhat * LimbBase + u[j + n - 2]:
      dec qhat
      rhat += v[n - 1]
      if rhat >= LimbBase: break
    var carry, borrow = 0'i64
    for i in 0 ..< n:
      let product = qhat * v[i] + carry
      carry = product div LimbBase
      var t = u[i + j] - product mod LimbBase - borrow
      borrow = 0
      if t < 0:
        t += LimbBase
        borrow = 1
      u[i + j] = t
    let t = u[j + n] - carry - borrow
    if t < 0:
      # The estimate was one too large. Add the divisor back.
      dec qhat
      var c = 0'i64
      for i in 0 ..< n:
        var s = u[i + j] + v[i] + c
        c = 0
        if s >= LimbBase:
          s -= LimbBase
          c = 1
        u[i + j] = s
      u[j + n] = (t + LimbBase + c) mod LimbBase
    else:
      u[j + n] = t
    result.quotient[j] = qhat
  var remainder = 0'i64
  result.remainder = newSeq[int64](n)
  for i in countdown(n - 1, 0):
    let current = remainder * LimbBase + u[i]
    result.remainder[i] = current div scale
    remainder = current mod scale

proc divMod*(a, b: BigInt): tuple[quotient, remainder: BigInt] =
  ## Truncated division of arbitrary integers. The remainder has the sign of a.
  if b.isZero: raise newException(ValueError, "Division by zero")
  if cmpAbs(a.limbs, b.limbs) < 0:
    return (BigInt(), a)
  if b.limbs.len == 1:
    let (q, r) = divSmall(a, b.limbs[0])
    result.quotient = q
    result.quotient.negative = a.negative != b.negative
    result.quotient.trim()
    result.remainder = big(r)
    return
  let (q, r) = divModAbs(a.limbs, b.limbs)
  result.quotient = BigInt(negative: a.negative != b.negative, limbs: q)
  result.quotient.trim()
  result.remainder = BigInt(negative: a.negative, limbs: r)
  result.remainder.trim()

proc dropLimbs(a: BigInt, count: int): BigInt =
  ## |a| div 10^(9*count).
  if count >= a.limbs.len: return
  result.limbs = a.limbs[count .. ^1]
  result.trim()

proc exactQuotientBig*(a, b: BigInt): BigInt =
  ## a / b when b divides a. Long division costs (quotient limbs) * (divisor limbs), so
  ## when the divisor is much longer than the quotient only its top limbs are divided,
  ## and the one exact candidate among three is confirmed by a single multiplication.
  if b.isZero: raise newException(ValueError, "Division by zero")
  let quotientLimbs = a.limbs.len - b.limbs.len + 1
  if quotientLimbs <= 0 or a.isZero:
    if not a.isZero: raise newException(ValueError, "Division is not exact")
    return
  let shift = b.limbs.len - (quotientLimbs + 2)
  if shift <= 8:
    let (q, r) = divMod(a, b)
    if not r.isZero: raise newException(ValueError, "Division is not exact")
    return q
  var top = dropLimbs(a, shift)
  var bottom = dropLimbs(b, shift)
  let estimate = divMod(top, bottom).quotient
  var magnitudeA = a
  magnitudeA.negative = false
  var magnitudeB = b
  magnitudeB.negative = false
  for delta in [0'i64, 1, -1]:
    let candidate = estimate + big(delta)
    if candidate * magnitudeB == magnitudeA:
      result = candidate
      result.negative = (a.negative != b.negative) and not result.isZero
      return
  raise newException(ValueError, "Division is not exact")

proc gcdBig*(a, b: BigInt): BigInt =
  var (x, y) = (a, b)
  x.negative = false
  y.negative = false
  while not y.isZero:
    let r = divMod(x, y).remainder
    x = y
    y = r
  x

proc absBig*(a: BigInt): BigInt =
  result = a
  result.negative = false

proc floorMod*(a, m: BigInt): BigInt =
  ## Residue in [0, |m|).
  result = divMod(a, m).remainder
  if result.negative: result = result + absBig(m)

proc powModBig*(base, exponent, modulus: BigInt): BigInt =
  ## base^exponent mod modulus for exponent >= 0 and modulus >= 1, in [0, modulus).
  if modulus.isNegative or modulus.isZero or exponent.isNegative:
    raise newException(ValueError, "powmod requires exponent >= 0 and modulus >= 1")
  result = floorMod(big(1), modulus)
  var b = floorMod(base, modulus)
  var e = exponent
  while not e.isZero:
    let (half, bit) = divSmall(e, 2)
    if bit != 0: result = floorMod(result * b, modulus)
    e = half
    if not e.isZero: b = floorMod(b * b, modulus)

proc exactDiv*(a: BigInt, divisor: int64): BigInt =
  let (quotient, remainder) = divSmall(a, divisor)
  if remainder != 0: raise newException(ValueError, "Division is not exact")
  quotient

proc mulModU(a, b, m: uint64): uint64 =
  ## (a*b) mod m for a, b < m < 2^63: a 128-bit product with GCC and Clang, doubling with MSVC.
  when defined(vcc):
    var x = a
    var y = b
    while y > 0:
      if (y and 1) != 0: result = (result + x) mod m
      x = (x + x) mod m
      y = y shr 1
  else:
    var r: uint64
    {.emit: [r, " = (unsigned long long)(((unsigned __int128)", a, " * (unsigned __int128)", b, ") % ", m, ");"].}
    result = r

proc modSmall*(a: BigInt, modulus: int64): int64 =
  ## Residue in [0, modulus) for any positive int64 modulus.
  if modulus <= 0: raise newException(ValueError, "Modulus must be positive")
  if modulus <= MaxSmallDivisor:
    for i in countdown(a.limbs.high, 0):
      result = (result * LimbBase + a.limbs[i]) mod modulus
  else:
    # Wide moduli: the running residue times 10^9 would overflow, so multiply by doubling.
    let m = uint64(modulus)
    let base = uint64(LimbBase) mod m
    var r = 0'u64
    for i in countdown(a.limbs.high, 0):
      r = (mulModU(r, base, m) + uint64(a.limbs[i])) mod m
    result = int64(r)
  if a.negative and result != 0: result = modulus - result

proc fitsInt64*(a: BigInt): bool =
  if a.limbs.len > 3: return false
  var magnitude = 0'u64
  for i in countdown(a.limbs.high, 0):
    if magnitude > (high(uint64) - uint64(a.limbs[i])) div uint64(LimbBase): return false
    magnitude = magnitude * uint64(LimbBase) + uint64(a.limbs[i])
  magnitude <= uint64(high(int64))

proc toInt64*(a: BigInt): int64 =
  if not a.fitsInt64(): raise newException(ValueError, "Integer exceeds the signed 64 bit range")
  for i in countdown(a.limbs.high, 0): result = result * LimbBase + a.limbs[i]
  if a.negative: result = -result

proc log10Abs*(a: BigInt): float64 =
  ## Approximate log10 |a|. Zero gives -Inf.
  if a.isZero: return NegInf
  var top = float64(a.limbs[^1])
  if a.limbs.len > 1: top += float64(a.limbs[^2]) / float64(LimbBase)
  log10(top) + float64(LimbDigits * (a.limbs.len - 1))

proc toFloat*(a: BigInt): float64 =
  if a.isZero: return 0
  let magnitude = if a.log10Abs() > 308: Inf else: pow(10.0, a.log10Abs())
  if a.negative: -magnitude else: magnitude

proc power*(base: BigInt, exponent: int): BigInt =
  if exponent < 0: raise newException(ValueError, "Negative exponent")
  result = big(1)
  var b = base
  var e = exponent
  while e > 0:
    if (e and 1) != 0: result = result * b
    e = e shr 1
    if e > 0: b = b * b

proc toJson*(a: BigInt): JsonNode =
  if a.fitsInt64() and abs(a.toInt64()) <= JsonSafeInteger: %a.toInt64() else: %($a)

proc mulMod*(a, b, m: int64): int64 =
  ## (a*b) mod m for 0 <= a, b < m without overflow, for any positive int64 m.
  if m <= 3_037_000_499'i64: return (a * b) mod m
  int64(mulModU(uint64(a mod m), uint64(b mod m), uint64(m)))

proc liftResidue(value, m: int64): int64 {.inline.} =
  result = value mod m
  if result < 0: result += m

proc powMod*(base, exponent, m: int64): int64 =
  if m == 1: return 0
  result = 1
  var b = liftResidue(base, m)
  var e = exponent
  while e > 0:
    if (e and 1) != 0: result = mulMod(result, b, m)
    b = mulMod(b, b, m)
    e = e shr 1

proc modularInverse*(value, m: int64): int64 =
  var (r0, r1) = (m, liftResidue(value, m))
  var (t0, t1) = (0'i64, 1'i64)
  while r1 != 0:
    let q = r0 div r1
    (r0, r1) = (r1, r0 - q * r1)
    (t0, t1) = (t1, t0 - q * t1)
  if r0 != 1: raise newException(ValueError, "Value is not invertible modulo " & $m)
  liftResidue(t0, m)

proc inverseModBig*(value, modulus: BigInt): BigInt =
  ## The inverse of value modulo modulus (>= 2) by the extended Euclidean algorithm.
  var (r0, r1) = (modulus, floorMod(value, modulus))
  var (t0, t1) = (big(0), big(1))
  while not r1.isZero:
    let q = divMod(r0, r1).quotient
    (r0, r1) = (r1, r0 - q * r1)
    (t0, t1) = (t1, t0 - q * t1)
  if not (r0 == big(1)): raise newException(ValueError, "Moduli must be pairwise coprime")
  floorMod(t0, modulus)

proc chineseRemainderBig*(residues, moduli: openArray[BigInt]): tuple[value, modulus: BigInt] =
  ## The reconstruction for moduli of any size, one congruence at a time.
  if residues.len != moduli.len or moduli.len == 0:
    raise newException(ValueError, "Supply one residue per modulus")
  result.value = big(0)
  result.modulus = big(1)
  for i, m in moduli:
    if m <= big(1): raise newException(ValueError, "Each modulus must be at least 2")
    let step = floorMod((residues[i] - result.value) * inverseModBig(result.modulus, m), m)
    result.value = result.value + result.modulus * step
    result.modulus = result.modulus * m

proc chineseRemainder*(residues: openArray[BigInt], moduli: openArray[int64]):
    tuple[value, modulus: BigInt] =
  ## Garner reconstruction for pairwise coprime moduli. value is in [0, modulus).
  ## Moduli past MaxSmallDivisor move to the big integer tier.
  if residues.len != moduli.len or moduli.len == 0:
    raise newException(ValueError, "Supply one residue per modulus")
  for m in moduli:
    if m < 2: raise newException(ValueError, "Each modulus must be at least 2")
    if m > MaxSmallDivisor:
      var wide: seq[BigInt]
      for x in moduli: wide.add(big(x))
      return chineseRemainderBig(residues, wide)
  result.value = big(0)
  result.modulus = big(1)
  for i, m in moduli:
    # Coprime to the running product means coprime to every earlier modulus.
    let previous = result.modulus.modSmall(m)
    if gcd(previous, m) != 1:
      raise newException(ValueError, "Moduli must be pairwise coprime")
    let r = residues[i].modSmall(m)
    let current = result.value.modSmall(m)
    let step = mulMod(((r - current) mod m + m) mod m, modularInverse(previous, m), m)
    result.value = result.value + result.modulus * big(step)
    result.modulus = result.modulus * big(m)

proc solveChineseRemainder*(problem: JsonNode): JsonNode =
  const MaxCongruences = 100_000
  let residueNodes = problem{"residues"}
  let moduliNodes = problem{"moduli"}
  if residueNodes.isNil or residueNodes.kind != JArray or moduliNodes.isNil or
      moduliNodes.kind != JArray or residueNodes.len != moduliNodes.len or residueNodes.len notin 1 .. MaxCongruences:
    raise newException(ValueError, "Supply residues and moduli arrays of equal length, at most " & $MaxCongruences)
  var residues, bigModuli: seq[BigInt]
  var moduli: seq[int64]
  var small = true
  for i in 0 ..< residueNodes.len:
    residues.add(toBig(residueNodes[i], "residue"))
    let m = toBig(moduliNodes[i], "modulus")
    bigModuli.add(m)
    if m.fitsInt64() and m.toInt64() <= MaxSmallDivisor: moduli.add(m.toInt64())
    else: small = false
  var (value, modulus) = if small: chineseRemainder(residues, moduli)
                         else: chineseRemainderBig(residues, bigModuli)
  let symmetric = problem{"symmetric"}.getBool(false)
  if symmetric and modulus < value + value: value = value - modulus
  for i, m in bigModuli:
    if not (floorMod(value, m) == floorMod(residues[i], m)):
      raise newException(ValueError, "Chinese remainder reconstruction did not verify")
  %*{"answer": value.toJson(), "modulus": modulus.toJson(), "verified": true, "method": "exact",
     "solution": "Garner's algorithm combines the congruences. The answer is checked against every residue.",
     "certificate": {"answer_digits": value.digits(), "representation": (if symmetric: "symmetric" else: "least_nonnegative")}}
