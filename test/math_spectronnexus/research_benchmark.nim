## Far-term benchmark for research_sequence.
##
## Every question gives 40 terms and asks for a far term, exactly or modulo M.
## The truth comes from independent code in this file: closed forms, Lucas'
## theorem, Touchard's congruence, recurrences iterated term by term, or direct
## sums. It shares only the BigInt type with the tool, never a solver. Some
## questions sit past what the tool can reach today. They are marked "gap" and
## show where work is still needed; none is marked at the moment.
##
## Build with `make bench-research` (compiled with AVX2 on x86_64), then:
##   nimcache/research_benchmark <folder with nimcache/math_spectronnexus> [--planner-checkpoint=path]

import std/[algorithm, bitops, json, math, monotimes, os, osproc, sequtils, streams, strformat, strutils, tables, times]
import big_integer

const
  P = 1_000_000_007'i64
  W = 1_000_000_000_039'i64   ## The least prime above 10^12.

type
  Poly = proc(k: int64): int64
  Question = object
    name, group: string
    gap: bool
    arguments: JsonNode
    truth: string

# ---------------------------------------------------------------- modular helpers

proc mulm(a, b, m: int64): int64 {.inline.} =
  ## (a*b) mod m for 0 <= a, b < m < 2^63: a 64-bit product below 2^32, else 128-bit.
  if m <= 4_294_967_296'i64: return int64(uint64(a) * uint64(b) mod uint64(m))
  var r: int64
  {.emit: [r, " = (long long)(((unsigned __int128)(unsigned long long)", a,
    " * (unsigned __int128)(unsigned long long)", b, ") % (unsigned long long)", m, ");"].}
  r

proc md(x, m: int64): int64 {.inline.} =
  result = x mod m
  if result < 0: result += m

proc powm(base, exponent, m: int64): int64 =
  var b = md(base, m)
  var e = exponent
  result = 1 mod m
  while e > 0:
    if (e and 1) == 1: result = mulm(result, b, m)
    b = mulm(b, b, m)
    e = e shr 1

proc invm(x, p: int64): int64 = powm(x, p - 2, p)

proc invm2(x, m: int64): int64 =
  ## The inverse of a unit modulo any m, by the extended Euclidean algorithm.
  var (r0, r1) = (m, md(x, m))
  var (t0, t1) = (0'i64, 1'i64)
  while r1 != 0:
    let quotient = r0 div r1
    (r0, r1) = (r1, r0 - quotient * r1)
    (t0, t1) = (t1, t0 - quotient * t1)
  doAssert r0 == 1
  md(t0, m)

# ---------------------------------------------------------------- arithmetic functions

proc factor(n: int64): seq[(int64, int)] =
  var n = n
  var d = 2'i64
  while d * d <= n:
    var e = 0
    while n mod d == 0:
      inc e
      n = n div d
    if e > 0: result.add((d, e))
    d += (if d == 2: 1 else: 2)
  if n > 1: result.add((n, 1))

proc phi(n: int64): int64 =
  result = n
  for (p, _) in factor(n): result -= result div p

proc sigma(n: int64): BigInt =
  result = big(1)
  for (p, e) in factor(n):
    result = result * ((power(big(p), e + 1) - big(1)).exactDiv(p - 1))

proc tau(n: int64): int64 =
  result = 1
  for (_, e) in factor(n): result *= e + 1

proc mobius(n: int64): int64 =
  let f = factor(n)
  for (_, e) in f:
    if e > 1: return 0
  if f.len mod 2 == 0: 1 else: -1

proc liouville(n: int64): int64 =
  var total = 0
  for (_, e) in factor(n): total += e
  if total mod 2 == 0: 1 else: -1

proc stern(n: int64): int64 =
  var (a, b, rest) = (1'i64, 0'i64, n)
  while rest > 0:
    if (rest and 1) == 1: b += a else: a += b
    rest = rest shr 1
  b

proc popcount(n: int64): int64 =
  var n = n
  while n > 0:
    result += n and 1
    n = n shr 1

# ---------------------------------------------------------------- linear recurrences

proc linearTerms(coefficients, initial: seq[int64], count: int): seq[BigInt] =
  for x in initial: result.add(big(x))
  while result.len < count:
    var total = big(0)
    for j, c in coefficients: total = total + big(c) * result[^(1 + j)]
    result.add(total)

proc linearMod(coefficients, initial: seq[int64], n, m: BigInt): BigInt =
  ## Term n of a(k) = sum c_j a(k-1-j), by a companion matrix power modulo m.
  let r = coefficients.len
  if n < big(r): return floorMod(big(initial[n.toInt64]), m)
  proc mul(a, b: seq[seq[BigInt]]): seq[seq[BigInt]] =
    result = newSeq[seq[BigInt]](r)
    for i in 0 ..< r:
      result[i] = newSeq[BigInt](r)
      for j in 0 ..< r:
        var total = big(0)
        for k in 0 ..< r: total = total + a[i][k] * b[k][j]
        result[i][j] = floorMod(total, m)
  var base = newSeq[seq[BigInt]](r)
  var acc = newSeq[seq[BigInt]](r)
  for i in 0 ..< r:
    base[i] = newSeq[BigInt](r)
    acc[i] = newSeq[BigInt](r)
    for j in 0 ..< r:
      base[i][j] = big(0)
      acc[i][j] = big(if i == j: 1 else: 0)
  for j in 0 ..< r: base[0][j] = floorMod(big(coefficients[j]), m)
  for i in 1 ..< r: base[i][i - 1] = big(1)
  var e = n - big(r - 1)
  while not e.isZero:
    let (half, bit) = divSmall(e, 2)
    if bit == 1: acc = mul(acc, base)
    base = mul(base, base)
    e = half
  var total = big(0)
  for j in 0 ..< r: total = total + acc[0][j] * big(initial[r - 1 - j])
  floorMod(total, m)

# ---------------------------------------------------------------- holonomic recurrences

proc holonomicMod(lead: Poly, rest: seq[Poly], initial: seq[int64], n: int, m: int64): int64 =
  ## lead(k) a(k) = sum_j rest[j](k) a(k-1-j), iterated modulo a prime m.
  var a: seq[int64]
  for x in initial: a.add(md(x, m))
  for k in initial.len .. n:
    var total = 0'i64
    for j, f in rest: total = (total + mulm(md(f(k.int64), m), a[^(1 + j)], m)) mod m
    a.add(mulm(total, invm(md(lead(k.int64), m), m), m))
    if a.len > 8: a.delete(0)
  a[^1]

proc hornerMod(coefficients: seq[int64], k, m: int64): int64 =
  for j in countdown(coefficients.high, 0): result = md(mulm(result, k mod m, m) + md(coefficients[j], m), m)

proc holonomicProjective(lead: seq[int64], rest: seq[seq[int64]], initial: seq[int64], n: int64,
                         m: int64): int64 =
  ## lead(k) a(k) = sum_j rest[j](k) a(k-1-j) modulo a prime m, for long ranges: the window
  ## is kept times one common denominator, so only the final step inverts.
  let r = rest.len
  var window = newSeq[int64](r)               # window[j] = a(k-1-j) * den
  for j in 0 ..< r: window[j] = md(initial[initial.len - 1 - j], m)
  var den = 1'i64
  # Polynomial values at consecutive k by forward differences: additions only per step.
  proc differences(coefficients: seq[int64], start: int64): seq[int64] =
    let degree = coefficients.high
    var values = newSeq[int64](degree + 1)
    for i in 0 .. degree: values[i] = hornerMod(coefficients, start + i, m)
    for level in 1 .. degree:
      for i in countdown(degree, level): values[i] = md(values[i] - values[i - 1], m)
    values
  proc advance(state: var seq[int64]) =
    for i in 0 ..< state.high: state[i] = (state[i] + state[i + 1]) mod m
  let first = int64(initial.len)
  var leadState = differences(lead, first)
  var restState = rest.mapIt(differences(it, first))
  for k in first .. n:
    let scale = leadState[0]
    var next = 0'i64
    for j in 0 ..< r: next = (next + mulm(restState[j][0], window[j], m)) mod m
    leadState.advance()
    for state in restState.mitems: state.advance()
    for j in countdown(r - 1, 1): window[j] = mulm(window[j - 1], scale, m)
    window[0] = next
    den = mulm(den, scale, m)
  doAssert den != 0, "a leading coefficient vanished modulo the prime"
  mulm(window[0], invm(den, m), m)

proc binomialOfPartitions(n: int, m: int64): int64 =
  ## sum_k C(n, k) p(k) modulo a prime m > n.
  var partitions = newSeq[int64](n + 1)
  partitions[0] = 1
  for i in 1 .. n:
    var total = 0'i64
    var k = 1
    while true:
      let g1 = k * (3 * k - 1) div 2
      if g1 > i: break
      let g2 = k * (3 * k + 1) div 2
      var part = partitions[i - g1]
      if g2 <= i: part += partitions[i - g2]
      total = (if k mod 2 == 1: total + part else: total - part) mod m
      inc k
    partitions[i] = md(total, m)
  var factorial = newSeq[int64](n + 1)
  factorial[0] = 1
  for i in 1 .. n: factorial[i] = mulm(factorial[i - 1], i, m)
  let top = factorial[n]
  for k in 0 .. n:
    let c = mulm(top, invm(mulm(factorial[k], factorial[n - k], m), m), m)
    result = (result + mulm(c, partitions[k], m)) mod m

proc factorialsAt(points: seq[int64], p: int64): Table[int64, int64] =
  ## x! mod p for each x < p in one pass up to the largest point.
  var sorted = points
  sorted.sort()
  var value = 1'i64
  var x = 0'i64
  for target in sorted:
    while x < target:
      inc x
      value = mulm(value, x, p)
    result[target] = value

proc binomialLarge(pairs: seq[(int64, int64)], p: int64): seq[int64] =
  ## C(n, k) mod a prime p for each pair, by Lucas' theorem with the factorials of all
  ## digits taken from one pass.
  var digitPairs: seq[seq[(int64, int64)]]
  var points: seq[int64]
  for (n, k) in pairs:
    var (a, b) = (n, k)
    var digits: seq[(int64, int64)]
    while a > 0 or b > 0:
      digits.add((a mod p, b mod p))
      a = a div p
      b = b div p
    digitPairs.add(digits)
    for (x, y) in digits:
      if y <= x: points.add(@[x, y, x - y])
  let table = factorialsAt(points, p)
  for digits in digitPairs:
    var value = 1'i64
    for (x, y) in digits:
      if y > x:
        value = 0
        break
      value = mulm(value, mulm(table[x], invm(mulm(table[y], table[x - y], p), p), p), p)
    result.add(value)

proc lucasDigitsProduct(n: int64, p: int64, at: proc(d: int64): int64): int64 =
  ## prod over the base-p digits d of n of at(d), for a sequence with the Lucas property.
  result = 1
  var rest = n
  while rest > 0:
    result = mulm(result, at(rest mod p), p)
    rest = rest div p

proc bellDobinski(n: int, p: int64): int64 =
  ## B(n) = sum_{j=0..n} j^n / j! * D(n-j), D(m) = sum_{i=0..m} (-1)^i / i!: the truncated
  ## Dobinski formula, exact for j <= n. Needs a prime p > n.
  var inverseFactorial = newSeq[int64](n + 1)
  var factorial = 1'i64
  for i in 1 .. n: factorial = mulm(factorial, i, p)
  inverseFactorial[n] = invm(factorial, p)
  for i in countdown(n, 1): inverseFactorial[i - 1] = mulm(inverseFactorial[i], i, p)
  var alternating = newSeq[int64](n + 1)
  var running = 0'i64
  for i in 0 .. n:
    running = md(running + (if i mod 2 == 0: inverseFactorial[i] else: -inverseFactorial[i]), p)
    alternating[i] = running
  for j in 0 .. n:
    result = (result + mulm(mulm(powm(j, n, p), inverseFactorial[j], p), alternating[n - j], p)) mod p

proc zigzagMod(n: int, p: int64): int64 =
  ## Euler zigzag numbers by the Seidel-Entringer triangle: T(m,0) = 0,
  ## T(m,k) = T(m,k-1) + T(m-1,m-k), E(m) = T(m,m). Additions only.
  var row = @[1'i64 mod p]
  for m in 1 .. n:
    var next = newSeq[int64](m + 1)
    for k in 1 .. m: next[k] = (next[k - 1] + row[m - k]) mod p
    row = next
  row[n]

proc rootedTreesMod(n: int, p: int64): int64 =
  ## A000081 from a(1) = 1: a(m+1) = (1/m) sum_{k=1..m} (sum_{d|k} d a(d)) a(m-k+1).
  var a = newSeq[int64](n + 1)
  var c = newSeq[int64](n + 1)
  a[1] = 1
  for m in 1 ..< n:
    let weight = mulm(m mod p, a[m], p)
    var multiple = m
    while multiple <= n:
      c[multiple] = (c[multiple] + weight) mod p
      multiple += m
    var total = 0'i64
    for k in 1 .. m: total = (total + mulm(c[k], a[m - k + 1], p)) mod p
    a[m + 1] = mulm(total, invm(m, p), p)
  a[n]

proc holonomicExact(lead: Poly, rest: seq[Poly], initial: seq[int64], n: int): BigInt =
  var a: seq[BigInt]
  for x in initial: a.add(big(x))
  for k in initial.len .. n:
    var total = big(0)
    for j, f in rest: total = total + big(f(k.int64)) * a[^(1 + j)]
    let (quotient, remainder) = divMod(total, big(lead(k.int64)))
    doAssert remainder.isZero, "holonomic step is not exact"
    a.add(quotient)
    if a.len > 8: a.delete(0)
  a[^1]

# ---------------------------------------------------------------- binomials, Catalan

proc binomialTable(rows: int): seq[seq[BigInt]] =
  result = newSeq[seq[BigInt]](rows)
  for n in 0 ..< rows:
    result[n] = newSeq[BigInt](n + 1)
    result[n][0] = big(1)
    result[n][n] = big(1)
    for k in 1 ..< n: result[n][k] = result[n - 1][k - 1] + result[n - 1][k]

proc centralBinomial(n: int): BigInt =
  ## C(2n, n) as prod (n+i)/i; every partial product C(n+i, i) is an integer.
  result = big(1)
  for i in 1 .. n: result = (result * big(n + i)).exactDiv(i)

proc catalanExact(n: int): BigInt = centralBinomial(n).exactDiv(n + 1)

proc centralBinomialMod(n: int, p: int64): int64 =
  var num = 1'i64
  var den = 1'i64
  for i in 1 .. n:
    num = mulm(num, md(n + i, p), p)
    den = mulm(den, md(i, p), p)
  mulm(num, invm(den, p), p)

proc binomLucas(n, k, p: int64): int64 =
  var fact = newSeq[int64](p)
  fact[0] = 1
  for i in 1 ..< p: fact[i] = mulm(fact[i - 1], i, p)
  var (top, bottom) = (n, k)
  result = 1
  while top > 0 or bottom > 0:
    let (a, b) = (top mod p, bottom mod p)
    if b > a: return 0
    result = mulm(result, mulm(fact[a], invm(mulm(fact[b], fact[a - b], p), p), p), p)
    top = top div p
    bottom = bottom div p

# ---------------------------------------------------------------- partitions, Bell numbers

proc partitionsExact(n: int): seq[BigInt] =
  ## Euler's pentagonal number recurrence.
  result = newSeq[BigInt](n + 1)
  result[0] = big(1)
  for i in 1 .. n:
    var total = big(0)
    var k = 1
    while true:
      let g1 = k * (3 * k - 1) div 2
      if g1 > i: break
      let positive = k mod 2 == 1
      let g2 = k * (3 * k + 1) div 2
      var part = result[i - g1]
      if g2 <= i: part = part + result[i - g2]
      total = (if positive: total + part else: total - part)
      inc k
    result[i] = total

proc partitionsMod(n: int, m: int64): int64 =
  var p = newSeq[int64](n + 1)
  p[0] = 1
  for i in 1 .. n:
    var total = 0'i64
    var k = 1
    while true:
      let g1 = k * (3 * k - 1) div 2
      if g1 > i: break
      let g2 = k * (3 * k + 1) div 2
      var part = p[i - g1]
      if g2 <= i: part += p[i - g2]
      total = (if k mod 2 == 1: total + part else: total - part) mod m
      inc k
    p[i] = md(total, m)
  p[n]

proc bellExact(n: int): seq[BigInt] =
  ## The Bell triangle: each row starts with the last entry of the previous row.
  var row = @[big(1)]
  result = @[big(1)]
  for _ in 1 .. n:
    var next = newSeqOfCap[BigInt](row.len + 1)
    next.add(row[^1])
    for x in row: next.add(next[^1] + x)
    row = next
    result.add(row[0])

proc bellMod(n: int, m: int64): seq[int64] =
  var row = @[1'i64]
  result = @[1'i64]
  for _ in 1 .. n:
    var next = newSeqOfCap[int64](row.len + 1)
    next.add(row[^1])
    for x in row: next.add((next[^1] + x) mod m)
    row = next
    result.add(row[0])

proc bellModSmallPrime(n: int, p: int64): int64 =
  ## Touchard's congruence B(k+p) = B(k) + B(k+1) mod p, from B(0..2p).
  var values = bellMod(2 * p.int, p)
  for k in values.len .. n: values.add((values[k - p] + values[k - p + 1]) mod p)
  values[n]

proc catalanLucas(n, p: int64): int64 =
  ## Catalan(n) = C(2n, n) - C(2n, n+1), both by Lucas' theorem, so n+1 may be divisible by p.
  md(binomLucas(2 * n, n, p) - binomLucas(2 * n, n + 1, p), p)

proc bellModPrimeFar(n: int64, p: int64): int64 =
  ## Touchard: B(k+p) = B(k) + B(k+1) mod p, so B(n) = sum c_i B(i) with
  ## x^n = sum c_i x^i modulo x^p - x - 1 over GF(p).
  let degree = p.int
  proc reduce(poly: var seq[int64]) =
    for k in countdown(poly.high, degree):
      let top = poly[k]
      if top == 0: continue
      poly[k - degree] = (poly[k - degree] + top) mod p       # x^p = x + 1
      poly[k - degree + 1] = (poly[k - degree + 1] + top) mod p
      poly[k] = 0
    poly.setLen(min(poly.len, degree))
  proc multiply(a, b: seq[int64]): seq[int64] =
    result = newSeq[int64](a.len + b.len - 1)
    for i, x in a:
      for j, y in b: result[i + j] = (result[i + j] + x * y) mod p
    reduce(result)
  var acc = @[1'i64]
  var base = @[0'i64, 1]
  reduce(base)
  var e = n
  while e > 0:
    if (e and 1) == 1: acc = multiply(acc, base)
    base = multiply(base, base)
    e = e shr 1
  let small = bellMod(degree, p)
  for i, c in acc: result = (result + c * small[i]) mod p

proc derangementModFar(n: int64, p: int64): int64 =
  ## D(k) = k D(k-1) + (-1)^k. Modulo p the state (D(k), k mod 2p) repeats, so the
  ## sequence is eventually periodic; find the cycle and fold n into it.
  var seen = initTable[(int64, int64), int64]()
  var values = @[1'i64 mod p, 0'i64]       # D(0), D(1)
  var k = 1'i64
  while true:
    let key = (values[k], k mod (2 * p))
    if key in seen:
      let start = seen[key]
      let period = k - start
      let folded = if n <= k: n else: start + (n - start) mod period
      return values[folded]
    seen[key] = k
    inc k
    values.add(md(mulm(k mod p, values[k - 1], p) + (if k mod 2 == 0: 1 else: -1), p))

var ctPrime = 0'i64
var ctFact, ctInvFact: seq[int64]

proc ctTables(p: int64) =
  if ctPrime == p: return
  ctPrime = p
  ctFact = newSeq[int64](p)
  ctFact[0] = 1
  for i in 1 ..< p: ctFact[i] = mulm(ctFact[i - 1], i, p)
  ctInvFact = newSeq[int64](p)
  ctInvFact[p - 1] = invm(ctFact[p - 1], p)
  for i in countdown(p - 1, 1): ctInvFact[i - 1] = mulm(ctInvFact[i], i, p)

proc laurentCoefficient(a, b, c, m, e, p: int64): int64 =
  ## [x^e] (a/x + b + c x)^m modulo a prime p > m, as the multinomial sum over i factors a/x,
  ## k = i + e factors c x and j = m - 2i - e factors b.
  if abs(e) > m: return 0
  let am = md(a, p)
  let bm = md(b, p)
  let cm = md(c, p)
  let first = max(0'i64, -e)
  let last = (m - e) div 2
  if first > last: return 0
  if bm == 0:
    if (m - e) mod 2 != 0: return 0
    let i = last
    let k = i + e
    return mulm(mulm(ctFact[m], mulm(ctInvFact[i], ctInvFact[k], p), p),
                mulm(powm(am, i, p), powm(cm, k, p), p), p)
  let inverseB2 = invm(mulm(bm, bm, p), p)
  var pa = powm(am, first, p)
  var pc = powm(cm, first + e, p)
  var j = m - 2 * first - e
  var pb = powm(bm, j, p)
  for i in first .. last:
    let k = i + e
    let term = mulm(mulm(ctInvFact[i], ctInvFact[j], p), mulm(ctInvFact[k], mulm(mulm(pa, pb, p), pc, p), p), p)
    result = (result + term) mod p
    pa = mulm(pa, am, p)
    pc = mulm(pc, cm, p)
    pb = mulm(pb, inverseB2, p)
    j -= 2
  result = mulm(result, ctFact[m], p)

proc floorDiv(x, p: int64): int64 =
  if x >= 0: x div p else: -((-x + p - 1) div p)

proc constantTermFar(a, b, c: int64, weights: seq[(int64, int64)], n: int64, p: int64): int64 =
  ## CT[(a/x + b + c x)^n Q(x)] modulo a prime p, Q = sum w x^k. Over F_p the power splits as
  ## prod P(x^(p^i))^(n_i) over the base-p digits n_i, so the constant term is a walk over
  ## carries: digit i takes exponent e_i with e_i + s = t p and passes the carry t on.
  ctTables(p)
  var states = initTable[int64, int64]()
  for (k, w) in weights: states[k] = (states.getOrDefault(k) + md(w, p)) mod p
  var rest = n
  while rest > 0:
    let digit = rest mod p
    rest = rest div p
    var next = initTable[int64, int64]()
    for s, v in states:
      if v == 0: continue
      let low = -floorDiv(-(s - digit), p)   # ceil((s - digit) / p)
      for t in low .. floorDiv(s + digit, p):
        let coefficient = laurentCoefficient(a, b, c, digit, t * p - s, p)
        if coefficient != 0: next[t] = (next.getOrDefault(t) + mulm(v, coefficient, p)) mod p
    states = next
  states.getOrDefault(0)

proc fineBrute(n: int64, p: int64): int64 =
  ## Fine's numbers term by term: 2 F(m) + F(m - 1) = Catalan(m), F(0) = 1, with each Catalan
  ## number C(2m, m) - C(2m, m + 1) by Lucas' theorem from one factorial table.
  ctTables(p)
  proc binom(top0, bottom0: int64): int64 =
    var top = top0
    var bottom = bottom0
    result = 1
    while top > 0 or bottom > 0:
      let a = top mod p
      let b = bottom mod p
      if b > a: return 0
      result = mulm(result, mulm(ctFact[a], mulm(ctInvFact[b], ctInvFact[a - b], p), p), p)
      top = top div p
      bottom = bottom div p
  let half = invm(2, p)
  result = 1
  for m in 1'i64 .. n:
    let catalan = md(binom(2 * m, m) - binom(2 * m, m + 1), p)
    result = mulm(md(catalan - result, p), half, p)

type PrimePowerBinomial = object
  p, q: int64
  units: seq[int64]   ## units[x] = product of k <= x with p not dividing k, modulo q

proc initPrimePowerBinomial(p, q: int64): PrimePowerBinomial =
  result = PrimePowerBinomial(p: p, q: q, units: newSeq[int64](q))
  result.units[0] = 1 mod q
  for k in 1'i64 ..< q: result.units[k] = (if k mod p == 0: result.units[k - 1] else: mulm(result.units[k - 1], k, q))

proc unitFactorial(b: PrimePowerBinomial, n: int64): int64 =
  ## n! with every factor p removed, modulo q = p^k: prod_i F(floor(n / p^i)), where F(x) is the
  ## product of the units up to x, and each full block of q units multiplies to -1 (or 1 for 2^k, k >= 3).
  let wilson = if b.p == 2 and b.q >= 8: 1'i64 else: b.q - 1
  result = 1 mod b.q
  var x = n
  while x > 0:
    result = mulm(result, mulm(powm(wilson, x div b.q, b.q), b.units[x mod b.q], b.q), b.q)
    x = x div b.p

proc binomialPrimePower(b: PrimePowerBinomial, n, k: int64): int64 =
  ## C(n, k) modulo q = p^e: Legendre's valuation and the unit factorials.
  if k < 0 or k > n: return 0
  proc valuation(x: int64): int64 =
    var y = x
    while y > 0:
      y = y div b.p
      result += y
  let v = valuation(n) - valuation(k) - valuation(n - k)
  var power = 1'i64 mod b.q
  for _ in 1 .. v:
    power = mulm(power, b.p mod b.q, b.q)
    if power == 0: return 0
  let unit = mulm(b.unitFactorial(n), invm2(mulm(b.unitFactorial(k), b.unitFactorial(n - k), b.q), b.q), b.q)
  mulm(power, unit, b.q)

proc derangementPrefix(count: int, m: int64): seq[int64] =
  result = @[1'i64 mod m, 0'i64]
  for k in 2 ..< count: result.add(mulm(md(k - 1, m), (result[^1] + result[^2]) mod m, m))

proc periodicFar(prefix: seq[int64], n: int64): int64 =
  ## The residue at n of an eventually periodic list: the smallest period that repeats
  ## over the whole second half, folded from its first repeat (checked by brute force
  ## on a list several periods long).
  let size = prefix.len
  for period in 1 .. size div 4:
    var start = -1
    for s in 0 .. size div 2:
      var ok = true
      for i in s ..< size - period:
        if prefix[i] != prefix[i + period]:
          ok = false
          break
      if ok:
        start = s
        break
    if start >= 0:
      if n < size: return prefix[n]
      return prefix[start + int((n - start) mod period)]
  raise newException(ValueError, "no period found")

proc derangementMod(n: int, m: int64): int64 =
  var (a, b) = (1'i64, 0'i64)
  for k in 2 .. n: (a, b) = (b, mulm(md(k - 1, m), (a + b) mod m, m))
  b

# ---------------------------------------------------------------- questions

proc crtBig(r1, m1, r2, m2: BigInt): BigInt =
  ## The value modulo m1 m2 with the given residues (m1, m2 coprime).
  floorMod(r1 + m1 * floorMod((r2 - r1) * inverseModBig(floorMod(m1, m2), m2), m2), m1 * m2)

proc crtPair(r1, m1, r2, m2: int64): BigInt =
  ## The value modulo m1 m2 with the given residues (m1, m2 coprime and small).
  var x = r1
  while x mod m2 != r2: x += m1
  big(x)

proc jsonNumber(x: BigInt): JsonNode =
  if x.fitsInt64: %x.toInt64 else: %($x)

proc q(name, group: string, terms: seq[BigInt], index: BigInt, truth: BigInt,
       modulus = big(0), offset = 0, gap = false): Question =
  var question = %*{"index": jsonNumber(index)}
  var value = truth
  if not modulus.isZero:
    question["modulus"] = jsonNumber(modulus)
    value = floorMod(truth, modulus)
  var list = newJArray()
  for t in terms: list.add(%($t))
  Question(name: name, group: group, gap: gap, truth: $value,
           arguments: %*{"terms": list, "offset": offset, "question": question})

proc ints(values: seq[int64]): seq[BigInt] =
  for v in values: result.add(big(v))

proc firstTerms(f: proc(n: int64): int64, first: int64): seq[BigInt] =
  for n in first ..< first + 40: result.add(big(f(n)))

const
  motzkinQ = @[(0'i64, 1'i64), (-2'i64, -1'i64)]     # M(n) = [x^0] P^n - [x^2] P^n, P = 1/x + 1 + x
  riordanQ = @[(0'i64, 1'i64), (-1'i64, -1'i64)]     # R(n) = [x^0] P^n - [x^1] P^n
  schroederQ = @[(0'i64, 2'i64), (2'i64, -4'i64)]    # r(n + 1) = 2 [x^0] P^n - 4 [x^-2] P^n, P = 1/x + 3 + 2x

proc catalanPrimePower(n: int64, b: PrimePowerBinomial): int64 =
  md(b.binomialPrimePower(2 * n, n) - b.binomialPrimePower(2 * n, n + 1), b.q)

proc motzkinPrimePower(n: int64, b: PrimePowerBinomial): int64 =
  ## M(n) = sum_k C(n, 2k) Catalan(k).
  for k in 0'i64 .. n div 2:
    result = (result + mulm(b.binomialPrimePower(n, 2 * k), catalanPrimePower(k, b), b.q)) mod b.q

proc binomialParts(top, k, p, q: int64): tuple[valuation, numerator, denominator: int64] =
  ## C(top, k) = p^valuation numerator / denominator straight from prod_{i=1}^{k} (top - k + i) / i,
  ## the factors p counted apart and the units modulo q = p^e: O(k), to check the doubling
  ## product on huge moduli.
  result = (0'i64, 1'i64 mod q, 1'i64 mod q)
  for i in 1'i64 .. k:
    var a = top - k + i
    var b = i
    while a mod p == 0:
      a = a div p
      inc result.valuation
    while b mod p == 0:
      b = b div p
      dec result.valuation
    result.numerator = mulm(result.numerator, a mod q, q)
    result.denominator = mulm(result.denominator, b mod q, q)

proc assemble(valuation, numerator, denominator, p, q: int64): int64 =
  doAssert valuation >= 0
  result = mulm(numerator, invm2(denominator, q), q)
  for _ in 1 .. valuation: result = mulm(result, p mod q, q)

proc binomialDirect(top, k, p, q: int64): int64 =
  let parts = binomialParts(top, k, p, q)
  assemble(parts.valuation, parts.numerator, parts.denominator, p, q)

proc catalanDirect(n, p, q: int64): int64 =
  ## Catalan(n) = C(2n, n) / (n + 1) modulo q = p^e.
  var parts = binomialParts(2 * n, n, p, q)
  var factor = n + 1
  while factor mod p == 0:
    factor = factor div p
    dec parts.valuation
  assemble(parts.valuation, parts.numerator, mulm(parts.denominator, factor mod q, q), p, q)

proc catalanWrap(n: int64): BigInt =
  ## Catalan(n) modulo 2^64 by uint64 wraparound: prod (n + i) / i over i <= n, then / (n + 1),
  ## the factors 2 counted apart and the odd denominator inverted by Newton's iteration.
  var valuation = 0
  var numerator, denominator = 1'u64
  for i in 1'i64 .. n:
    let a = n + i
    let b = i
    let za = countTrailingZeroBits(a)
    let zb = countTrailingZeroBits(b)
    valuation += za - zb
    numerator *= uint64(a shr za)
    denominator *= uint64(b shr zb)
  let c = n + 1
  let zc = countTrailingZeroBits(c)
  valuation -= zc
  denominator *= uint64(c shr zc)
  var inverse = denominator            # right to 3 bits for an odd number
  for _ in 1 .. 6: inverse *= 2'u64 - denominator * inverse
  if valuation >= 64: return big(0)
  let value = (numerator * inverse) shl valuation
  big(int64(value shr 1)) * big(2) + big(int64(value and 1))

proc catalanDirectBig(n, p: int64, q: BigInt): BigInt =
  ## Catalan(n) modulo q = p^e of any size, term by term on big integers.
  var valuation = 0'i64
  var numerator, denominator = big(1)
  for i in 1'i64 .. n:
    var a = n + i
    var b = i
    while a mod p == 0:
      a = a div p
      inc valuation
    while b mod p == 0:
      b = b div p
      dec valuation
    numerator = floorMod(numerator * big(a), q)
    denominator = floorMod(denominator * big(b), q)
  var c = n + 1
  while c mod p == 0:
    c = c div p
    dec valuation
  denominator = floorMod(denominator * big(c), q)
  result = floorMod(numerator * inverseModBig(denominator, q), q)
  for _ in 1 .. valuation: result = floorMod(result * big(p), q)

proc derangementPastPrime(n, p: int64): int64 =
  ## D(n) mod p for n >= p: D(m) = m D(m - 1) + (-1)^m gives D(m0) = (-1)^m0 at the last
  ## multiple m0 of p, then the few steps after it.
  let m0 = n - n mod p
  result = if m0 mod 2 == 0: 1'i64 else: p - 1
  for m in m0 + 1 .. n: result = md(mulm(m mod p, result, p) + (if m mod 2 == 0: 1'i64 else: -1'i64), p)

proc derangementTwoPower(n: int64, bits: int): int64 =
  ## D(n) = sum_m (-1)^m n! / m! modulo 2^bits: terms with m < n - 4 bits hold a product of
  ## 4 bits consecutive integers, divisible by 2^bits, so only the last ones count.
  let mask = (1'u64 shl bits) - 1
  var total = 0'u64
  var product = 1'u64           # prod_{j = m + 1}^{n} j, wrapping modulo 2^64
  var m = n
  while m >= 0 and m >= n - 4 * bits:
    if m mod 2 == 0: total += product else: total -= product
    product *= uint64(m)
    dec m
  int64(total and mask)

proc involutionsPastPrime(n, p: int64): int64 =
  ## I(n) mod p: (I(m - 1), I(m)) -> (I(m), I(m) + m I(m - 1)) depends on m mod p, so the
  ## steps m = 1 .. n - 1 are whole periods of p (one 2x2 product, then a power) and a rest.
  type M = array[4, int64]
  proc mul(a, b: M): M =
    [md(mulm(a[0], b[0], p) + mulm(a[1], b[2], p), p), md(mulm(a[0], b[1], p) + mulm(a[1], b[3], p), p),
     md(mulm(a[2], b[0], p) + mulm(a[3], b[2], p), p), md(mulm(a[2], b[1], p) + mulm(a[3], b[3], p), p)]
  proc steps(count: int64): M =
    result = [1'i64, 0, 0, 1]
    for m in 1'i64 .. count:
      let next = mul([0'i64, 1, m mod p, 1], result)
      result = next
  let count = n - 1
  var period = steps(p)
  var power: M = [1'i64, 0, 0, 1]
  var e = count div p
  while e > 0:
    if (e and 1) == 1:
      let next = mul(period, power)
      power = next
    let squared = mul(period, period)
    period = squared
    e = e shr 1
  let total = mul(steps(count mod p), power)
  md(total[2] + total[3], p)    # applied to (I(0), I(1)) = (1, 1)

proc squarePlusOneProduct(n, p: int64): int64 =
  ## prod_{k=1}^{n} (k^2 + 1) mod p for p = 3 mod 4: over a full period of residues the
  ## product is (i^p - i)((-i)^p + i) = 4, and the rest is direct.
  doAssert p mod 4 == 3
  result = powm(4, n div p, p)
  for k in 1'i64 .. n mod p: result = mulm(result, md(mulm(k, k, p) + 1, p), p)

proc binomialSumDirect(n, p, q: int64, apery: bool): int64 =
  ## Apery sum_k C(n,k)^2 C(n+k,k)^2 or Franel sum_k C(n,k)^3 modulo q = p^e, term by term:
  ## C(n, k + 1) = C(n, k) (n - k) / (k + 1) and C(n + k + 1, k + 1) = C(n + k, k) (n + k + 1) / (k + 1),
  ## with the factors p counted apart and the units inverted from a table.
  var e = 0
  var power = 1'i64
  while power < q:
    power *= p
    inc e
  var inverse = newSeq[int64](q)
  for x in 1'i64 ..< q:
    if x mod p != 0: inverse[x] = invm2(x, q)
  proc pPower(v: int64): int64 =
    if v >= e: return 0
    result = 1
    for _ in 1 .. v: result *= p
  proc update(v, u: var int64, numerator, denominator: int64) =
    var a = numerator
    var b = denominator
    while a mod p == 0:
      a = a div p
      inc v
    while b mod p == 0:
      b = b div p
      dec v
    u = mulm(mulm(u, a mod q, q), inverse[b mod q], q)
  var (v1, u1, v2, u2) = (0'i64, 1'i64 mod q, 0'i64, 1'i64 mod q)
  for k in 0'i64 .. n:
    let term = if apery: mulm(pPower(2 * v1 + 2 * v2), mulm(mulm(u1, u1, q), mulm(u2, u2, q), q), q)
               else: mulm(pPower(3 * v1), mulm(mulm(u1, u1, q), u1, q), q)
    result = (result + term) mod q
    if k == n: break
    update(v1, u1, n - k, k + 1)
    update(v2, u2, n + k + 1, k + 1)

proc fineOddPrimePower(n, p, q: int64): int64 =
  ## Fine's numbers modulo q = p^e for odd p: F(m) = (C(m) - F(m - 1)) / 2, with the Catalan
  ## numbers C(m) = C(m - 1) (4m - 2) / (m + 1) kept as p^v times a unit.
  var e = 0
  var power = 1'i64
  while power < q:
    power *= p
    inc e
  let half = invm2(2, q)
  var (v, unit) = (0'i64, 1'i64 mod q)
  result = 1 mod q
  for m in 1'i64 .. n:
    var a = 4 * m - 2
    var b = m + 1
    while a mod p == 0:
      a = a div p
      inc v
    while b mod p == 0:
      b = b div p
      dec v
    unit = mulm(mulm(unit, a mod q, q), invm2(b mod q, q), q)
    var catalan = 0'i64
    if v < e:
      catalan = unit
      for _ in 1 .. v: catalan = mulm(catalan, p, q)
    result = mulm(md(catalan - result, q), half, q)

type
  PadicCounter = object
    p, q: int64
    alpha: int
    inverse: seq[int64]   ## inverse[x] for the units x modulo q
  PadicValue = tuple[v: int, u: int64]

proc initPadicCounter(p, q: int64): PadicCounter =
  result = PadicCounter(p: p, q: q, inverse: newSeq[int64](q))
  var power = 1'i64
  while power < q:
    power *= p
    inc result.alpha
  for x in 1'i64 ..< q:
    if x mod p != 0: result.inverse[x] = invm2(x, q)

proc scaleBy(c: PadicCounter, a: PadicValue, x: int64, divide = false): PadicValue =
  ## a x or a / x for x > 0, the factors p counted apart.
  var y = x
  var v = 0
  while y mod c.p == 0:
    y = y div c.p
    inc v
  let w = y mod c.q
  if divide: (a.v - v, mulm(a.u, c.inverse[w], c.q)) else: (a.v + v, mulm(a.u, w, c.q))

proc valueOf(c: PadicCounter, a: PadicValue): int64 =
  doAssert a.v >= 0
  if a.v >= c.alpha: return 0
  result = a.u
  for _ in 1 .. a.v: result = mulm(result, c.p, c.q)

proc fineBallot(n, p, q: int64): int64 =
  ## Fine's numbers modulo q = p^e from F(x) = C(x) / (1 + x C(x)): with the ballot numbers
  ## [x^m] C(x)^(j+1) = C(2m+j, m) - C(2m+j, m-1), F(n) = sum_j (-1)^j (C(2n-j, n) - C(2n-j, n+1)).
  let c = initPadicCounter(p, q)
  var binom: PadicValue = (0, 1'i64 mod q)    # C(2n - j, n), starting from C(2n, n)
  for i in 1'i64 .. n:
    binom = c.scaleBy(binom, n + i)
    binom = c.scaleBy(binom, i, divide = true)
  for j in 0'i64 .. n:
    let m = 2 * n - j
    var term = c.valueOf(binom)
    if m > n:
      var upper = c.scaleBy(binom, m - n)
      upper = c.scaleBy(upper, n + 1, divide = true)   # C(m, n + 1) = C(m, n) (m - n) / (n + 1)
      term = md(term - c.valueOf(upper), q)
    result = if j mod 2 == 0: (result + term) mod q else: md(result - term, q)
    if m > n:                                        # C(m - 1, n) = C(m, n) (m - n) / m
      binom = c.scaleBy(binom, m - n)
      binom = c.scaleBy(binom, m, divide = true)

proc motzkinPartialSum(n, p, q: int64): int64 =
  ## sum_{m <= n} M(m) = sum_k C(n + 1, 2k + 1) Catalan(k) modulo q = p^e: the hockey stick
  ## identity on M(m) = sum_k C(m, 2k) Catalan(k).
  let c = initPadicCounter(p, q)
  let top = n + 1
  var binom = c.scaleBy((0, 1'i64 mod q), top)      # C(top, 1)
  var catalan: PadicValue = (0, 1'i64 mod q)
  var k = 0'i64
  while 2 * k + 1 <= top:
    result = (result + mulm(c.valueOf(binom), c.valueOf(catalan), q)) mod q
    if 2 * k + 3 > top: break
    binom = c.scaleBy(binom, top - 2 * k - 1)
    binom = c.scaleBy(binom, top - 2 * k - 2)
    binom = c.scaleBy(binom, 2 * k + 2, divide = true)
    binom = c.scaleBy(binom, 2 * k + 3, divide = true)
    catalan = c.scaleBy(catalan, 4 * k + 2)
    catalan = c.scaleBy(catalan, k + 2, divide = true)
    inc k

proc buildQuestions(): seq[Question] =
  let pb = big(P)
  let mersenne = big(2_305_843_009_213_693_951'i64)        # 2^61 - 1
  let wide = power(big(10), 60) + big(7)
  let t12 = big(1_000_000_000_000'i64)
  let binom = binomialTable(80)
  var catalan40, central40, apery40, franel40, motzkin40, derange40, schroeder40, riordan40: seq[BigInt]
  for n in 0 ..< 40:
    catalan40.add(binom[2 * n][n].exactDiv(n + 1))
    central40.add(binom[2 * n][n])
    var a, f = big(0)
    for k in 0 .. n:
      let c = binom[n][k]
      a = a + c * c * binom[n + k][k] * binom[n + k][k]
      f = f + c * c * c
    apery40.add(a)
    franel40.add(f)
  for n in 0 ..< 40:
    var total = big(0)
    for k in 0 .. n div 2: total = total + binom[n][2 * k] * catalan40[k]
    motzkin40.add(total)
  schroeder40 = @[big(1), big(2)]
  for n in 0 ..< 38:   # (n + 3) r(n + 2) = (6 n + 9) r(n + 1) - n r(n)
    schroeder40.add((big(6 * n + 9) * schroeder40[n + 1] - big(n) * schroeder40[n]).exactDiv(n + 3))
  riordan40 = @[big(1)]
  for n in 1 ..< 40: riordan40.add(motzkin40[n - 1] - riordan40[n - 1])   # M(n) = R(n) + R(n + 1)
  var motzkin80 = @[big(1), big(1)]
  for n in 2 ..< 80:   # (n + 2) M(n) = (2n + 1) M(n - 1) + 3 (n - 1) M(n - 2)
    motzkin80.add((big(2 * n + 1) * motzkin80[n - 1] + big(3 * n - 3) * motzkin80[n - 2]).exactDiv(n + 2))
  var riordan80 = @[big(1)]
  for n in 1 ..< 80: riordan80.add(motzkin80[n - 1] - riordan80[n - 1])
  var riordanEven40, fine40: seq[BigInt]
  for n in 0 ..< 40: riordanEven40.add(riordan80[2 * n])
  fine40 = @[big(1)]
  for n in 1 ..< 40:
    let twice = catalan40[n] - fine40[n - 1]
    fine40.add(twice.exactDiv(2))
  var motzkinPartial40 = @[motzkin40[0]]
  for n in 1 ..< 40: motzkinPartial40.add(motzkinPartial40[^1] + motzkin40[n])
  const ecmLow = 10_000_000_000_000_061'i64       # 10^16 + 61 and 10^17 + 3 are prime
  const ecmHigh = 100_000_000_000_000_003'i64
  var involutions40 = @[big(1), big(1)]
  for n in 2 ..< 40: involutions40.add(involutions40[^1] + big(n - 1) * involutions40[^2])
  var squarePlusOne40 = @[big(1)]
  for n in 1 ..< 40: squarePlusOne40.add(squarePlusOne40[^1] * big(n * n + 1))
  derange40 = @[big(1), big(0)]
  for n in 2 ..< 40: derange40.add(big(n - 1) * (derange40[^1] + derange40[^2]))
  let fib40 = linearTerms(@[1'i64, 1], @[0'i64, 1], 40)
  let trib40 = linearTerms(@[1'i64, 1, 1], @[0'i64, 0, 1], 40)
  let pell40 = linearTerms(@[2'i64, 1], @[0'i64, 1], 40)
  let lucas3 = linearTerms(@[3'i64, -1], @[2'i64, 3], 40)
  let bell1000 = bellExact(1000)
  let bell40 = bell1000[0 ..< 40]
  let part1000 = partitionsExact(1000)
  let part40 = part1000[0 ..< 40]
  var binomialPartitions40: seq[BigInt]
  for n in 0 ..< 40:
    var total = big(0)
    for k in 0 .. n: total = total + binom[n][k] * part1000[k]
    binomialPartitions40.add(total)

  let motzLead: Poly = proc(k: int64): int64 = k + 2
  let motzRest: seq[Poly] = @[Poly(proc(k: int64): int64 = 2 * k + 1), Poly(proc(k: int64): int64 = 3 * k - 3)]
  let franLead: Poly = proc(k: int64): int64 = k * k
  let franRest: seq[Poly] = @[Poly(proc(k: int64): int64 = 7 * (k - 1) * (k - 1) + 7 * (k - 1) + 2),
                              Poly(proc(k: int64): int64 = 8 * (k - 1) * (k - 1))]
  let aperyLead: Poly = proc(k: int64): int64 = k * k * k
  let aperyRest: seq[Poly] = @[Poly(proc(k: int64): int64 = 34 * k * k * k - 51 * k * k + 27 * k - 5),
                               Poly(proc(k: int64): int64 = -((k - 1) * (k - 1) * (k - 1)))]
  let one: Poly = proc(k: int64): int64 = 1
  let derangeRest: seq[Poly] = @[Poly(proc(k: int64): int64 = k - 1), Poly(proc(k: int64): int64 = k - 1)]

  let phiTerms = firstTerms(phi, 1)
  var sigmaTerms: seq[BigInt]
  for n in 1'i64 .. 40: sigmaTerms.add(sigma(n))
  let e15 = 1_000_000_000_000_000'i64
  let e18 = big(1_000_000_000_000_000_000'i64)
  let catalanAt10k = catalanExact(10_000)
  let catalanFar = mulm(binomLucas(2_000_000_000_000, 1_000_000_000_000, 10007),
                        invm(md(1_000_000_000_001, 10007), 10007), 10007)
  let e17 = 100_000_000_000_000_000'i64
  let catalanFourPrimes = block:
    var value = big(0)
    var modulus = big(1)
    for prime in [1_000_003'i64, 1_000_033, 1_000_037, 1_000_039]:
      let n = e18.toInt64
      let residue = md(binomLucas(2 * n, n, prime) - binomLucas(2 * n, n + 1, prime), prime)
      value = crtBig(value, modulus, big(residue), big(prime))
      modulus = modulus * big(prime)
    value
  let two24 = initPrimePowerBinomial(2, 16_777_216)
  let two10 = initPrimePowerBinomial(2, 1024)
  let three7 = initPrimePowerBinomial(3, 2187)
  let catalanBillion = block:
    let two9 = initPrimePowerBinomial(2, 512)
    let five9 = initPrimePowerBinomial(5, 1_953_125)
    let r2 = catalanPrimePower(e18.toInt64, two9)
    let r5 = catalanPrimePower(e18.toInt64, five9)
    var x = r5
    while x mod 512 != r2: x += 1_953_125
    big(x)
  let large = binomialLarge(@[(2 * e17, e17), (2 * e17, e17 + 1)], P)
  let aperySmall = proc(d: int64): int64 =
    if d < 2: (if d == 0: 1'i64 else: 5'i64)
    else: holonomicProjective(@[0'i64, 0, 0, 1], @[@[-5'i64, 27, -51, 34], @[1'i64, -3, 3, -1]], @[1'i64, 5], d, 1_000_003)
  let franelSmall = proc(d: int64): int64 =
    if d < 2: (if d == 0: 1'i64 else: 2'i64)
    else: holonomicProjective(@[0'i64, 0, 1], @[@[2'i64, -7, 7], @[8'i64, -16, 8]], @[1'i64, 2], d, 999_983)
  var zigzag40, trees40: seq[BigInt]
  block:
    var row = @[big(1)]
    zigzag40.add(big(1))
    for m in 1 ..< 40:
      var next = newSeq[BigInt](m + 1)
      next[0] = big(0)
      for k in 1 .. m: next[k] = next[k - 1] + row[m - k]
      row = next
      zigzag40.add(row[m])
    var a = newSeq[BigInt](41)
    var c = newSeq[BigInt](41)
    for i in 0 .. 40:
      a[i] = big(0)
      c[i] = big(0)
    a[1] = big(1)
    for m in 1 ..< 40:
      let weight = big(m) * a[m]
      var multiple = m
      while multiple <= 40:
        c[multiple] = c[multiple] + weight
        multiple += m
      var total = big(0)
      for k in 1 .. m: total = total + c[k] * a[m - k + 1]
      a[m + 1] = total.exactDiv(m)
    for i in 1 .. 40: trees40.add(a[i])
  var sternTerms, popTerms: seq[BigInt]
  for n in 0'i64 ..< 40:
    sternTerms.add(big(stern(n)))
    popTerms.add(big(popcount(n)))

  result = @[
    # Multiplicative functions at far indices (offset 1).
    q("phi(10^12)", "multiplicative", phiTerms, t12, big(phi(1_000_000_000_000)), offset = 1),
    q("phi(10^15+37) mod p", "multiplicative", phiTerms, big(e15 + 37), big(phi(e15 + 37)), pb, offset = 1),
    q("sigma(10^12) mod 2^61-1", "multiplicative", sigmaTerms, t12, sigma(1_000_000_000_000), mersenne, offset = 1),
    q("tau(963761198400)", "multiplicative", firstTerms(tau, 1), big(963761198400'i64), big(tau(963761198400'i64)), offset = 1),
    q("mu(10^12-1)", "multiplicative", firstTerms(mobius, 1), big(999_999_999_999'i64), big(mobius(999_999_999_999)), offset = 1),
    q("lambda(10^15)", "multiplicative", firstTerms(liouville, 1), big(e15), big(liouville(e15)), offset = 1),
    # Linear recurrences at huge indices and wide moduli.
    q("Fibonacci(10^18) mod p", "linear", fib40, e18, linearMod(@[1'i64, 1], @[0'i64, 1], e18, pb), pb),
    q("Tribonacci(10^15) mod 2^61-1", "linear", trib40, big(e15),
      linearMod(@[1'i64, 1, 1], @[0'i64, 0, 1], big(e15), mersenne), mersenne),
    q("Pell(10^100) mod 10^60+7", "linear", pell40, power(big(10), 100),
      linearMod(@[2'i64, 1], @[0'i64, 1], power(big(10), 100), wide), wide),
    q("3a-a (10^12) mod 10^12", "linear", lucas3, t12, linearMod(@[3'i64, -1], @[2'i64, 3], t12, t12), t12),
    q("Fibonacci(3000) exact", "linear", fib40, big(3000), linearTerms(@[1'i64, 1], @[0'i64, 1], 3001)[3000]),
    # Holonomic (P-recursive) sequences, exact and modulo.
    q("Catalan(10^5) mod p", "holonomic", catalan40, big(100_000),
      big(mulm(centralBinomialMod(100_000, P), invm(100_001, P), P)), pb),
    q("Catalan(5000) exact", "holonomic", catalan40, big(5000), catalanExact(5000)),
    q("Catalan(20000) exact", "holonomic", catalan40, big(20000), catalanExact(20000)),
    q("Catalan(10^4) mod 2^30", "holonomic", catalan40, big(10_000), catalanAt10k, big(1'i64 shl 30)),
    q("C(2n,n) at 3*10^5 mod 998244353", "holonomic", central40, big(300_000),
      big(centralBinomialMod(300_000, 998244353)), big(998244353)),
    q("Motzkin(2*10^5) mod p", "holonomic", motzkin40, big(200_000),
      big(holonomicMod(motzLead, motzRest, @[1'i64, 1], 200_000, P)), pb),
    q("Franel(10^4) mod p", "holonomic", franel40, big(10_000),
      big(holonomicMod(franLead, franRest, @[1'i64, 2], 10_000, P)), pb),
    q("Apery(3000) exact", "holonomic", apery40, big(3000), holonomicExact(aperyLead, aperyRest, @[1'i64, 5], 3000)),
    q("Apery(5*10^4) mod p", "holonomic", apery40, big(50_000),
      big(holonomicMod(aperyLead, aperyRest, @[1'i64, 5], 50_000, P)), pb),
    q("derangements(2000) exact", "holonomic", derange40, big(2000), holonomicExact(one, derangeRest, @[1'i64, 0], 2000)),
    q("derangements(10^5) mod 10^12", "holonomic", derange40, big(100_000),
      big(derangementMod(100_000, 1_000_000_000_000)), t12),
    # Transformed directions (inverse Euler).
    q("p(1000) exact", "transformed", part40, big(1000), part1000[1000]),
    q("p(10^4) mod p", "transformed", part40, big(10_000), big(partitionsMod(10_000, P)), pb),
    # Self-referential and binary relations.
    q("Bell(1000) exact", "self", bell40, big(1000), bell1000[1000]),
    q("Bell(3000) mod p", "self", bell40, big(3000), big(bellMod(3000, P)[3000]), pb),
    q("Stern(10^15) exact", "binary", sternTerms, big(e15), big(stern(e15))),
    q("popcount(10^18 - 1) exact", "binary", popTerms, big(999_999_999_999_999_999'i64),
      big(popcount(999_999_999_999_999_999'i64))),
    # Hypergeometric terms modulo small primes at huge indices (p-adic Wilson products).
    q("Catalan(10^12) mod 10007", "small prime", catalan40, t12, big(catalanFar), big(10007)),
    q("Catalan(10^18) mod 999983", "small prime", catalan40, e18, big(catalanLucas(e18.toInt64, 999983)), big(999983)),
    q("Catalan(10^12) mod 10007*10009", "small prime", catalan40, t12,
      crtPair(catalanLucas(1_000_000_000_000, 10007), 10007, catalanLucas(1_000_000_000_000, 10009), 10009),
      big(10007 * 10009)),
    q("C(2n,n) at 10^15 mod 10007", "small prime", central40, big(e15),
      big(binomLucas(2 * e15, e15, 10007)), big(10007)),
    # Linear recurrences of the residues modulo small primes (Berlekamp-Massey).
    q("Bell(10^5) mod 101", "small prime", bell40, big(100_000), big(bellModSmallPrime(100_000, 101)), big(101)),
    q("Bell(10^18) mod 13", "small prime", bell40, e18, big(bellModPrimeFar(e18.toInt64, 13)), big(13)),
    q("derangements(10^15) mod 101", "small prime", derange40, big(e15), big(derangementModFar(e15, 101)), big(101)),
    # Euler transform by NTT convolution.
    q("p(10^5) mod p", "transformed", part40, big(100_000), big(partitionsMod(100_000, P)), pb),
    q("p(5*10^5) mod p", "transformed", part40, big(500_000), big(partitionsMod(500_000, P)), pb),
    q("binomial(p)(10^5) mod p", "transformed", binomialPartitions40, big(100_000),
      big(binomialOfPartitions(100_000, P)), pb),
    # Holonomic terms past 10^6 steps modulo a large prime: the square root matrix product.
    q("Motzkin(10^7) mod p", "sqrt product", motzkin40, big(10_000_000),
      big(holonomicProjective(@[2'i64, 1], @[@[1'i64, 2], @[-3'i64, 3]], @[1'i64, 1], 10_000_000, P)), pb),
    q("Motzkin(10^8) mod p", "sqrt product", motzkin40, big(100_000_000),
      big(holonomicProjective(@[2'i64, 1], @[@[1'i64, 2], @[-3'i64, 3]], @[1'i64, 1], 100_000_000, P)), pb),
    q("Apery(5*10^7) mod 998244353", "sqrt product", apery40, big(50_000_000),
      big(holonomicProjective(@[0'i64, 0, 0, 1], @[@[-5'i64, 27, -51, 34], @[1'i64, -3, 3, -1]], @[1'i64, 5],
                              50_000_000, 998244353)), big(998244353)),
    # Index far past a large prime: p-adic products in square root time, the Lucas property.
    q("Catalan(10^17) mod p", "past modulus", catalan40, big(e17),
      big(md(large[0] - large[1], P)), pb),
    q("C(2n,n) at 10^17 mod p", "past modulus", central40, big(e17), big(large[0]), pb),
    q("Apery(10^18) mod 1000003", "past modulus", apery40, e18,
      big(lucasDigitsProduct(e18.toInt64, 1_000_003, aperySmall)), big(1_000_003)),
    q("Franel(10^15) mod 999983", "past modulus", franel40, big(e15),
      big(lucasDigitsProduct(e15, 999_983, franelSmall)), big(999_983)),
    # Self-referential relations past 20,000 modulo a large prime: online convolutions.
    q("Bell(10^6) mod p", "online", bell40, big(1_000_000), big(bellDobinski(1_000_000, P)), pb),
    q("zigzag(3*10^4) mod p", "online", zigzag40, big(30_000), big(zigzagMod(30_000, P)), pb),
    q("rooted trees(3*10^4) mod p", "online", trees40, big(30_000), big(rootedTreesMod(30_000, P)), pb, offset = 1),
    # A prime modulus past 2^31: six NTT primes rebuild the convolutions.
    q("p(10^5) mod 10^12+39", "wide prime", part40, big(100_000), big(partitionsMod(100_000, W)), big(W)),
    q("Motzkin(2*10^7) mod 10^12+39", "wide prime", motzkin40, big(20_000_000),
      big(holonomicProjective(@[2'i64, 1], @[@[1'i64, 2], @[-3'i64, 3]], @[1'i64, 1], 20_000_000, W)), big(W)),
    q("Bell(2*10^5) mod 10^12+39", "wide prime", bell40, big(200_000), big(bellDobinski(200_000, W)), big(W)),
    # Small composite moduli: eventually periodic residues.
    q("Bell(10^18) mod 64", "composite", bell40, e18, big(periodicFar(bellMod(4096, 64), e18.toInt64)), big(64)),
    q("derangements(10^15) mod 1000", "composite", derange40, big(e15),
      big(periodicFar(derangementPrefix(20_000, 1000), e15)), big(1000)),
    # Constant terms CT[P^n Q] past the prime: one Cartier step per base-p digit.
    q("Motzkin(10^18) mod 1000003", "constant term", motzkin40, e18,
      big(constantTermFar(1, 1, 1, motzkinQ, e18.toInt64, 1_000_003)), big(1_000_003)),
    q("Motzkin(10^18) mod 10000019", "constant term", motzkin40, e18,
      big(constantTermFar(1, 1, 1, motzkinQ, e18.toInt64, 10_000_019)), big(10_000_019)),
    q("Schroeder(10^18) mod 1000003", "constant term", schroeder40, e18,
      big(constantTermFar(1, 3, 2, schroederQ, e18.toInt64 - 1, 1_000_003)), big(1_000_003)),
    q("Riordan(10^15) mod 10000019", "constant term", riordan40, big(e15),
      big(constantTermFar(1, 1, 1, riordanQ, e15, 10_000_019)), big(10_000_019)),
    q("Motzkin(5*10^6) mod 1000003", "constant term", motzkin40, big(5_000_000),
      big(constantTermFar(1, 1, 1, motzkinQ, 5_000_000, 1_000_003)), big(1_000_003)),
    q("Riordan(2n) at 10^18 mod 1000003", "constant term", riordanEven40, e18,
      big(constantTermFar(1, 1, 1, riordanQ, 2 * e18.toInt64, 1_000_003)), big(1_000_003)),
    q("Fine(5*10^6) mod 1000003", "constant term", fine40, big(5_000_000), big(fineBrute(5_000_000, 1_000_003)),
      big(1_000_003)),
    q("Fine(10^7) mod 65537", "constant term", fine40, big(10_000_000), big(fineBrute(10_000_000, 65_537)),
      big(65_537)),
    # Prime power moduli: generalised Wilson products and the Rowland-Yassawi Cartier step.
    q("Catalan(10^18) mod 10^9", "prime power", catalan40, e18, catalanBillion, big(1_000_000_000)),
    q("Catalan(10^18) mod 2^24", "prime power", catalan40, e18,
      big(catalanPrimePower(e18.toInt64, two24)), big(16_777_216)),
    q("Motzkin(10^6) mod 1024", "prime power", motzkin40, big(1_000_000),
      big(motzkinPrimePower(1_000_000, two10)), big(1024)),
    q("Motzkin(3*10^5) mod 3^7", "prime power", motzkin40, big(300_000),
      big(motzkinPrimePower(300_000, three7)), big(2187)),
    q("Catalan(10^7) mod 3^39", "prime power", catalan40, big(10_000_000),
      big(catalanDirect(10_000_000, 3, 4_052_555_153_018_976_267'i64)), big(4_052_555_153_018_976_267'i64)),
    q("C(2n,n) at 2*10^7 mod 2^61", "prime power", central40, big(20_000_000),
      big(binomialDirect(40_000_000, 20_000_000, 2, 2_305_843_009_213_693_952'i64)),
      big(2_305_843_009_213_693_952'i64)),
    q("Catalan(1.2*10^7) mod 2^64", "prime power", catalan40, big(12_000_000), catalanWrap(12_000_000),
      power(big(2), 64)),
    q("Catalan(1.05*10^7) mod 10^30", "prime power", catalan40, big(10_500_000),
      crtBig(big(catalanDirect(10_500_000, 2, 1 shl 30)), power(big(2), 30),
             catalanDirectBig(10_500_000, 5, power(big(5), 30)), power(big(5), 30)), power(big(10), 30)),
    q("derangements(10^18) mod p", "period", derange40, e18, big(derangementPastPrime(e18.toInt64, P)), pb),
    q("derangements(10^18) mod 2^40", "period", derange40, e18,
      big(derangementTwoPower(e18.toInt64, 40)), big(1_099_511_627_776'i64)),
    q("involutions(10^18) mod 1000003", "period", involutions40, e18,
      big(involutionsPastPrime(e18.toInt64, 1_000_003)), big(1_000_003)),
    q("prod(k^2+1) at 10^18 mod p", "period", squarePlusOne40, e18, big(squarePlusOneProduct(e18.toInt64, P)), pb),
    q("Apery(1.05*10^7) mod 3^10", "binomial sum", apery40, big(10_500_000),
      big(binomialSumDirect(10_500_000, 3, 59_049, true)), big(59_049)),
    q("Franel(1.05*10^7) mod 2^10", "binomial sum", franel40, big(10_500_000),
      big(binomialSumDirect(10_500_000, 2, 1024, false)), big(1024)),
    q("Fine(1.05*10^7) mod 3^9", "binomial sum", fine40, big(10_500_000),
      big(fineOddPrimePower(10_500_000, 3, 19_683)), big(19_683)),
    q("Catalan(10^18) mod 4 primes ~10^6", "pollard", catalan40, e18, catalanFourPrimes,
      big(1_000_003) * big(1_000_033) * big(1_000_037) * big(1_000_039)),
    q("Catalan(5*10^6) mod 1009^6", "prime power", catalan40, big(5_000_000),
      big(catalanDirect(5_000_000, 1009, 1_055_229_678_769_825_441'i64)), big(1_055_229_678_769_825_441'i64)),
    q("Fine(1.05*10^7) mod 2^10", "prime power", fine40, big(10_500_000), big(fineBallot(10_500_000, 2, 1024)),
      big(1024)),
    q("Motzkin partial sums at 1.05*10^7 mod 3^7", "prime power", motzkinPartial40, big(10_500_000),
      big(motzkinPartialSum(10_500_000, 3, 2187)), big(2187)),
    q("Catalan(2*10^7) mod (10^16+61)(10^17+3)", "ecm", catalan40, big(20_000_000),
      crtBig(big(catalanDirect(20_000_000, ecmLow, ecmLow)), big(ecmLow),
             big(catalanDirect(20_000_000, ecmHigh, ecmHigh)), big(ecmHigh)), big(ecmLow) * big(ecmHigh)),
  ]

# ---------------------------------------------------------------- self checks

proc selfCheck() =
  ## Known values guard the truth code itself.
  doAssert phi(1_000_000_000_000) == 400_000_000_000
  doAssert tau(963761198400) == 6720
  doAssert $catalanExact(10) == "16796"
  doAssert $partitionsExact(100)[100] == "190569292"
  doAssert partitionsMod(100, P) == 190569292
  doAssert $bellExact(15)[15] == "1382958545"
  doAssert bellModSmallPrime(20, 101) == bellMod(20, 101)[20]
  doAssert $linearTerms(@[1'i64, 1], @[0'i64, 1], 101)[100] == "354224848179261915075"
  doAssert linearMod(@[1'i64, 1], @[0'i64, 1], big(100), big(P)) == floorMod(parseBig("354224848179261915075"), big(P))
  doAssert stern(9) == 4 and stern(10) == 3 and stern(11) == 5
  doAssert binomLucas(10, 5, 7) == 252 mod 7
  for n in [0'i64, 1, 5, 10, 25, 39]:
    doAssert catalanLucas(n, 7) == catalanExact(n.int).modSmall(7)
    doAssert bellModPrimeFar(n, 7) == bellExact(40)[n].modSmall(7)
  var derange = @[big(1), big(0)]
  for n in 2 .. 400: derange.add(big(n - 1) * (derange[^1] + derange[^2]))
  for n in [0'i64, 1, 2, 7, 150, 399, 400]: doAssert derangementModFar(n, 11) == derange[n].modSmall(11)
  doAssert holonomicProjective(@[2'i64, 1], @[@[1'i64, 2], @[-3'i64, 3]], @[1'i64, 1], 30, P) ==
    holonomicMod(proc(k: int64): int64 = k + 2, @[Poly(proc(k: int64): int64 = 2 * k + 1),
                 Poly(proc(k: int64): int64 = 3 * k - 3)], @[1'i64, 1], 30, P)
  doAssert binomialOfPartitions(5, P) == 1 + 5 * 1 + 10 * 2 + 10 * 3 + 5 * 5 + 7
  let bells = bellExact(30)
  for n in [0, 1, 5, 17, 30]: doAssert bellDobinski(n, P) == bells[n].modSmall(P)
  doAssert zigzagMod(9, P) == 7936
  doAssert periodicFar(bellMod(4096, 64), 3000) == bellMod(4096, 64)[3000]
  doAssert periodicFar(derangementPrefix(20_000, 1000), 15_000) == derangementPrefix(20_000, 1000)[15_000]
  doAssert factor(1_000_003).len == 1 and factor(10_000_019).len == 1 and factor(65_537).len == 1
  block:
    # Fine's numbers against their exact values.
    let binom = binomialTable(80)
    var fine = @[big(1)]
    for n in 1 ..< 30: fine.add((binom[2 * n][n].exactDiv(n + 1) - fine[n - 1]).exactDiv(2))
    doAssert $fine[0 .. 9] == "@[1, 0, 1, 2, 6, 18, 57, 186, 622, 2120]"
    for n in 0'i64 ..< 30: doAssert fineBrute(n, 101) == fine[n].modSmall(101)
  block:
    # Period truths against plain iteration.
    var d = @[1'i64, 0]
    for m in 2'i64 .. 3000: d.add(md(mulm(m - 1, (d[^1] + d[^2]) mod 1009, 1009), 1009))
    var inv = @[1'i64, 1]
    for m in 2'i64 .. 3000: inv.add(md(inv[^1] + mulm(m - 1, inv[^2], 1009), 1009))
    var sq = @[1'i64]
    for k in 1'i64 .. 3000: sq.add(mulm(sq[^1], md(k * k + 1, 1019), 1019))
    for n in [1009'i64, 1500, 2018, 2999]:
      doAssert derangementPastPrime(n, 1009) == d[n]
      doAssert involutionsPastPrime(n, 1009) == inv[n]
      doAssert squarePlusOneProduct(n, 1019) == sq[n]
    var exact = @[big(1), big(0)]
    for m in 2 .. 300: exact.add(big(m - 1) * (exact[^1] + exact[^2]))
    for n in [100'i64, 200, 300]: doAssert derangementTwoPower(n, 40) == exact[n].modSmall(1_099_511_627_776)
  block:
    # Direct binomial sums against exact Apery and Franel numbers.
    let binom = binomialTable(80)
    for n in 0 ..< 40:
      var a, f = big(0)
      for k in 0 .. n:
        let c = binom[n][k]
        a = a + c * c * binom[n + k][k] * binom[n + k][k]
        f = f + c * c * c
      for (p, q) in [(3'i64, 59_049'i64), (2'i64, 1024'i64), (5'i64, 625'i64)]:
        doAssert binomialSumDirect(int64(n), p, q, true) == a.modSmall(q)
        doAssert binomialSumDirect(int64(n), p, q, false) == f.modSmall(q)
    var fine = @[big(1)]
    for n in 1 ..< 40: fine.add((binom[2 * n][n].exactDiv(n + 1) - fine[n - 1]).exactDiv(2))
    for n in 0 ..< 40:
      for (p, q) in [(3'i64, 19_683'i64), (5'i64, 625'i64)]: doAssert fineOddPrimePower(int64(n), p, q) == fine[n].modSmall(q)
      for (p, q) in [(2'i64, 1024'i64), (3'i64, 2187'i64)]: doAssert fineBallot(int64(n), p, q) == fine[n].modSmall(q)
    var motzkin = @[big(1), big(1)]
    for n in 2 ..< 40: motzkin.add((big(2 * n + 1) * motzkin[n - 1] + big(3 * n - 3) * motzkin[n - 2]).exactDiv(n + 2))
    var partialSum = big(0)
    for n in 0 ..< 40:
      partialSum = partialSum + motzkin[n]
      for (p, q) in [(3'i64, 2187'i64), (2'i64, 1024'i64)]:
        doAssert motzkinPartialSum(int64(n), p, q) == partialSum.modSmall(q)
  block:
    # Prime power binomials, Catalan and Motzkin numbers against exact values.
    let binom = binomialTable(80)
    for (p, q) in [(2'i64, 1024'i64), (3'i64, 2187'i64), (5'i64, 625'i64)]:
      let b = initPrimePowerBinomial(p, q)
      for n in 0'i64 ..< 80:
        for k in 0'i64 .. n: doAssert b.binomialPrimePower(n, k) == binom[n][k].modSmall(q)
      var motzkin = @[big(1), big(1)]
      for n in 2 ..< 40: motzkin.add((big(2 * n + 1) * motzkin[n - 1] + big(3 * n - 3) * motzkin[n - 2]).exactDiv(n + 2))
      for n in 0'i64 ..< 40:
        doAssert catalanPrimePower(n, b) == binom[2 * n][n].exactDiv(int(n) + 1).modSmall(q)
        doAssert motzkinPrimePower(n, b) == motzkin[n].modSmall(q)
    for n in 0'i64 ..< 40:
      doAssert catalanDirect(n, 3, 4_052_555_153_018_976_267'i64) ==
        binom[2 * n][n].exactDiv(int(n) + 1).modSmall(4_052_555_153_018_976_267'i64)
      doAssert binomialDirect(2 * n, n, 2, 1 shl 61) == binom[2 * n][n].modSmall(1 shl 61)
      let catalan = binom[2 * n][n].exactDiv(int(n) + 1)
      doAssert catalanWrap(n) == floorMod(catalan, power(big(2), 64))
      doAssert catalanDirectBig(n, 5, power(big(5), 30)) == floorMod(catalan, power(big(5), 30))
  block:
    # The constant-term identities against exact terms from each sequence's own recurrence.
    let binom = binomialTable(80)
    var catalan, motzkin, riordan, schroeder: seq[BigInt]
    for n in 0 ..< 40: catalan.add(binom[2 * n][n].exactDiv(n + 1))
    for n in 0 ..< 40:
      var total = big(0)
      for k in 0 .. n div 2: total = total + binom[n][2 * k] * catalan[k]
      motzkin.add(total)
    riordan = @[big(1)]
    for n in 1 ..< 40: riordan.add(motzkin[n - 1] - riordan[n - 1])
    schroeder = @[big(1), big(2)]
    for n in 0 ..< 38: schroeder.add((big(6 * n + 9) * schroeder[n + 1] - big(n) * schroeder[n]).exactDiv(n + 3))
    doAssert $riordan[0 .. 7] == "@[1, 0, 1, 1, 3, 6, 15, 36]"
    doAssert $schroeder[0 .. 5] == "@[1, 2, 6, 22, 90, 394]"
    for n in 0'i64 ..< 40:
      doAssert constantTermFar(1, 1, 1, motzkinQ, n, 1_000_003) == motzkin[n].modSmall(1_000_003)
      doAssert constantTermFar(1, 1, 1, riordanQ, n, 1_000_003) == riordan[n].modSmall(1_000_003)
      if n < 39: doAssert constantTermFar(1, 3, 2, schroederQ, n, 1_000_003) == schroeder[n + 1].modSmall(1_000_003)
    # Past the prime: p = 7 against exact values, so the digit walk itself is checked.
    for n in 0'i64 ..< 40:
      doAssert constantTermFar(1, 1, 1, motzkinQ, n, 7) == motzkin[n].modSmall(7)
      doAssert constantTermFar(1, 1, 1, riordanQ, n, 11) == riordan[n].modSmall(11)
      if n < 39: doAssert constantTermFar(1, 3, 2, schroederQ, n, 13) == schroeder[n + 1].modSmall(13)
  doAssert powm(2, W - 1, W) == 1 and powm(3, W - 1, W) == 1 and powm(5, W - 1, W) == 1
  doAssert rootedTreesMod(10, P) == 719
  # 1_000_000_014 has base-p digits (7, 1) and 3 has (3, 0): C(7,3) C(1,0) = 35.
  doAssert binomialLarge(@[(10'i64, 5'i64), (1_000_000_014'i64, 3'i64)], P) == @[252'i64, 35]
  # The Lucas property (Gessel for Apery, Lucas' theorem for Franel) on exact terms mod 7.
  var apery, franel: seq[BigInt]
  let table = binomialTable(120)
  for n in 0 ..< 60:
    var a, f = big(0)
    for k in 0 .. n:
      let c = table[n][k]
      a = a + c * c * table[n + k][k] * table[n + k][k]
      f = f + c * c * c
    apery.add(a)
    franel.add(f)
  for n in 7 ..< 60:
    doAssert apery[n].modSmall(7) == apery[n div 7].modSmall(7) * apery[n mod 7].modSmall(7) mod 7
    doAssert franel[n].modSmall(7) == franel[n div 7].modSmall(7) * franel[n mod 7].modSmall(7) mod 7

# ---------------------------------------------------------------- runner

proc answerText(node: JsonNode): string =
  if node.isNil or node.kind == JNull: return ""
  case node.kind
  of JString: node.getStr
  of JInt: $node.getBiggestInt
  else: $node

type Server = object
  process: Process
  counter: int

proc send(server: var Server, methodName: string, params: JsonNode, notify = false,
          timeoutSeconds = 0.0): JsonNode =
  ## One JSON-RPC exchange. Returns nil for a notification or when the reply misses the timeout.
  var message = %*{"jsonrpc": "2.0", "method": methodName, "params": params}
  if not notify:
    inc server.counter
    message["id"] = %server.counter
  server.process.inputStream.write($message & "\n")
  server.process.inputStream.flush()
  if notify: return nil
  if timeoutSeconds > 0:
    let start = getMonoTime()
    while not server.process.hasData:
      if (getMonoTime() - start).inMilliseconds.float / 1000.0 > timeoutSeconds: return nil
      sleep(5)
  var line: string
  while server.process.outputStream.readLine(line):
    if line.startsWith("{"): return parseJson(line)
    echo line  # backend log lines share the pipe
  raise newException(IOError, "server closed its output")

proc startServer(binary: string, args: seq[string]): Server =
  result.process = startProcess(binary, args = args, options = {poStdErrToStdOut})
  discard result.send("initialize", %*{"protocolVersion": "2025-06-18", "capabilities": {},
                                       "clientInfo": {"name": "bench", "version": "0"}})
  discard result.send("notifications/initialized", newJObject(), notify = true)

proc main() =
  if paramCount() < 1:
    quit("Usage: research_benchmark <folder with nimcache/math_spectronnexus> " &
         "[--timeout=seconds] [--planner-checkpoint=path]")
  selfCheck()
  let truthStart = getMonoTime()
  let questions = buildQuestions()
  echo &"truths: {questions.len} questions in {(getMonoTime() - truthStart).inMilliseconds} ms"
  var args = @["mcp"]
  var timeoutSeconds = 120.0
  for i in 2 .. paramCount():
    let arg = paramStr(i)
    if arg.startsWith("--timeout="): timeoutSeconds = parseFloat(arg["--timeout=".len .. ^1])
    elif arg.startsWith("--"): args.add(arg)
  let binary = paramStr(1) / "nimcache" / "math_spectronnexus"
  var server = startServer(binary, args)
  var right, wrong, noAnswer, gapRight, gapTotal, actions = 0
  var seconds = 0.0
  for question in questions:
    let start = getMonoTime()
    let reply = server.send("tools/call", %*{"name": "research_sequence", "arguments": question.arguments},
                            timeoutSeconds = timeoutSeconds)
    let elapsed = (getMonoTime() - start).inNanoseconds.float / 1e9
    if reply.isNil:
      # A stuck route: count it as no answer and restart the server for the next question.
      server.process.kill()
      discard server.process.waitForExit()
      server.process.close()
      server = startServer(binary, args)
      if question.gap: inc gapTotal
      else:
        inc noAnswer
        seconds += elapsed
      echo &"{question.group:<14} {question.name:<34} {\"timeout\":<9} {elapsed:7.2f}s actions=-" &
        (if question.gap: " gap" else: "")
      continue
    let result = reply{"result"}
    var payload = result{"structuredContent"}
    if payload.isNil:
      let text = if result.isNil: "" else: result{"content"}{0}{"text"}.getStr
      try: payload = parseJson(text)
      except CatchableError: payload = %*{"error": text}
    let answer = answerText(payload{"answer"})
    let isError = not result.isNil and result{"isError"}.getBool
    let verdict = if isError or answer.len == 0: "no_answer"
                  elif answer == question.truth: "right"
                  else: "wrong"
    let used = payload{"actions"}.getInt
    if question.gap:
      inc gapTotal
      if verdict == "right": inc gapRight
    else:
      case verdict
      of "right": inc right
      of "wrong": inc wrong
      else: inc noAnswer
      actions += used
      seconds += elapsed
    let shownActions = if payload{"actions"}.isNil: "-" else: $used
    echo &"{question.group:<14} {question.name:<34} {verdict:<9} {elapsed:7.2f}s actions={shownActions}" &
      (if question.gap: " gap" else: "")
    if verdict == "no_answer":
      let reason = payload{"error"}.getStr(payload{"note"}.getStr)
      if reason.len > 0: echo "    reason: ", reason
  server.process.inputStream.close()
  discard server.process.waitForExit()
  server.process.close()
  var questionCount = 0
  for question in questions:
    if not question.gap: inc questionCount
  echo $(%*{"right": right, "wrong": wrong, "no_answer": noAnswer, "gap_right": gapRight,
            "gap_total": gapTotal, "actions": actions, "seconds": round(seconds * 100) / 100,
            "questions": questionCount})

when isMainModule:
  main()
