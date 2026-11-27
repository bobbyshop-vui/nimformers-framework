## Exact sequence explorer for the Math SpectronNexus helper. It fits
## structural hypotheses on early terms, possibly after transforms, and
## reports only those that also hold on terms left out of the fit.
import std/[algorithm, json, math, sequtils, strutils, tables]
from std/times import epochTime
import big_integer, calculator, exact_number, prime_field

const
  ExplorerPrimes* = [1_000_000_007'i64, 998_244_353'i64, 1_000_000_009'i64]
  MaxExplorerTerms* = 5_000
  MinExplorerTerms* = 8
  MaxTransformDepth* = 2
  ExplorerFeatures* = 128
  ExplorerActionCount* = 16
  PopAction* = ExplorerActionCount - 1
  GuideSteps* = 12
  Holdout = 4
  MaxPolynomialFitDegree = 12
  MaxHolonomicOrder = 5
  MaxHolonomicDegree = 6
  MaxTargetIterations = 20_000_000'i64
  MaxExactIndex = 10_000_000'i64        ## Binary splitting; the value's digit bound also applies.
  SequentialExactSteps = 2_000'i64      ## Fewer exact steps than this iterate one by one.
  MaxExactDigitsShown = 200_000
  DefaultModuli* = [2'i64, 3, 4, 5, 7, 8, 9, 11, 13]

type
  Hypothesis* = enum
    hPolynomial = "polynomial", hLinearRecurrence = "linear_recurrence",
    hHolonomic = "holonomic_recurrence", hAlgebraic = "algebraic_generating_function",
    hMultiplicative = "multiplicative", hPeriodic = "periodic_residues",
    hSelfReferential = "self_referential_relation",  ## Tested after the search, never a policy action.
    hBinaryRecursive = "binary_recursive_relation"   ## a(2n), a(2n+1) from a(n), a(n+1); also after the search.
  Transform* = enum
    tDifference = "difference", tPartialSum = "partial_sum", tRatio = "ratio",
    tDivideFactorial = "divide_by_factorial", tMultiplyFactorial = "multiply_by_factorial",
    tBinomial = "binomial_transform", tInverseBinomial = "inverse_binomial_transform",
    tMobius = "mobius_inversion", tInverseEuler = "inverse_euler_transform"
  ExplorerAction* = range[0 .. ExplorerActionCount - 1]
  ExplorerFrame* = array[ExplorerFeatures, float32]
  Rational = tuple[num, den: int64]
  Track* = object
    valid: bool
    offset: int
    residues: array[3, seq[int64]]
    approx: seq[float64]
    exact: seq[BigInt]
  TargetForm = object
    kind: Hypothesis
    available: bool
    modular: bool                                 ## Only residue coefficients are known.
    modularBlocks: array[3, seq[seq[int64]]]
    polynomial: seq[Rational]
    linear: seq[Rational]
    blocks: seq[seq[Rational]]
    selfFeatures: seq[int]                        ## a(n+1) = sum of selfCoefficients[j] * feature selfFeatures[j].
    selfCoefficients: seq[Rational]
    selfStart: int                                ## First n (from the first term) where the relation holds.
    binaryCoefficients: array[2, seq[Rational]]   ## a(2n+r) over a(n), a(n+1), a(2n+r-1), n, 1.
    binaryStart: array[2, int]                    ## First index m = 2n+r where each relation holds.
    multiplicativeModel: MultiplicativeModel      ## How a multiplicative function reaches prime powers past the data.
    multiplicativePolynomial: seq[Rational]       ## g(m) for m >= 2, or a(p) for primes p.
  MultiplicativeModel = enum
    mmNone, mmDivisorSum, mmCompletePrime
  Direction* = object
    path*: seq[Transform]
    hypothesis*: Hypothesis
    report*: JsonNode
    form: TargetForm
  Explorer* = object
    nodes: Table[string, Track]
    tested: Table[string, set[Hypothesis]]
    path*: seq[Transform]
    tests*, steps*: int
    directions*: seq[Direction]
    moduli: seq[int64]
    formulaDepth: int
    lastDescend: bool                            ## The previous action walked into the current node.
  ExplorerGuide* = proc(ex: var Explorer, legal: set[ExplorerAction]): ExplorerAction {.closure.}
  DirectionFilter* = proc(direction: Direction): bool {.closure.}
  Fit = object
    ok, rational: bool
    values: seq[Rational]
    residues: seq[int64]
    primeVectors: array[3, seq[int64]]   ## The canonical relation modulo each detection prime.
    fitRows, checkedRows: int
  Outcome = object
    ok: bool
    report: JsonNode
    form: TargetForm

const
  FormulaHypotheses* = {hPolynomial, hLinearRecurrence, hHolonomic, hAlgebraic}
  ## The hypotheses the guided and breadth-first search choose between. Their order fixes
  ## the policy's action numbering, so later hypotheses stay out of this set.
  SearchHypotheses* = {hPolynomial .. hPeriodic}
  LastSearchHypothesis = ord(hPeriodic)
  TransformSlots* = ord(high(Transform)) + 1
  ## Periodic residues are reported but do not end the search for structure.
  StrongHypotheses* = FormulaHypotheses + {hMultiplicative, hSelfReferential, hBinaryRecursive}
  MaxBinarySequential = 2_000_000'i64   ## Sequential evaluation when a(2n+r-1) appears.
  MaxSelfIndex = 20_000'i64
  MaxSelfExactIndex = 2_000'i64
  MaxSelfExactPrimes = 50_000     ## About 466,000 digits; memory is the only bound.
  ShortSpare = 1                  ## Short-data fits keep one spare fit equation ...
  ShortHoldout = 2                ## ... and two unseen ones.
  IntegralityTerms = 60           ## Terms a short holonomic fit must predict as integers.
  MaxMultiplicativeIndex = high(int64)   ## Pollard rho factors every int64 index.

var explorerGuide*: ExplorerGuide
var programPrologue*: proc(ex: var Explorer, maxDirections, budget: int) {.closure.}
  ## A learned tool policy drives the whole search: it names one tool at a time and the
  ## engine checks it before the next one is asked for. It writes the tool chain it ran
  ## into programLog, and while it is installed the plain scan does not run.
var programLog*: seq[string]

# Modular helpers for the three detection primes (all below 2^30).

proc addMod(a, b, p: int64): int64 {.inline.} = (a + b) mod p
proc subMod(a, b, p: int64): int64 {.inline.} = (a - b + p) mod p
proc mulP(a, b, p: int64): int64 {.inline.} = (a * b) mod p
proc invP(a, p: int64): int64 = powMod(a, p - 2, p)
proc liftMod(value, p: int64): int64 {.inline.} = ((value mod p) + p) mod p

proc rationalOf(residue, p: int64): tuple[ok: bool, num, den: int64] =
  ## Rational reconstruction with numerator and denominator below sqrt(p/2).
  let bound = int64(sqrt(float64(p) / 2.0))
  var (r0, r1) = (p, liftMod(residue, p))
  var (t0, t1) = (0'i64, 1'i64)
  while r1 > bound:
    let q = r0 div r1
    (r0, r1) = (r1, r0 - q * r1)
    (t0, t1) = (t1, t0 - q * t1)
  if t1 == 0 or abs(t1) > bound: return
  var (num, den) = (r1, t1)
  if den < 0: (num, den) = (-num, -den)
  if gcd(abs(num), den) != 1: return
  (true, num, den)

proc nullspace(rows: seq[seq[int64]], cols: int, p: int64): seq[seq[int64]] =
  var m = rows
  var pivotOf = newSeq[int](cols)
  for entry in pivotOf.mitems: entry = -1
  var rank = 0
  for col in 0 ..< cols:
    if rank == m.len: break
    var pivot = rank
    while pivot < m.len and m[pivot][col] == 0: inc pivot
    if pivot == m.len: continue
    swap(m[pivot], m[rank])
    let scale = invP(m[rank][col], p)
    for j in col ..< cols: m[rank][j] = mulP(m[rank][j], scale, p)
    for r in 0 ..< m.len:
      if r == rank or m[r][col] == 0: continue
      let factor = m[r][col]
      for j in col ..< cols: m[r][j] = subMod(m[r][j], mulP(factor, m[rank][j], p), p)
    pivotOf[col] = rank
    inc rank
  for free in 0 ..< cols:
    if pivotOf[free] >= 0: continue
    var v = newSeq[int64](cols)
    v[free] = 1
    for col in 0 ..< cols:
      if pivotOf[col] >= 0: v[col] = subMod(0, m[pivotOf[col]][free], p)
    result.add(v)

proc dot(row, v: seq[int64], p: int64): int64 =
  for i in 0 ..< row.len:
    if row[i] != 0 and v[i] != 0: result = (result + row[i] * v[i]) mod p

proc normalizeVector(v: var seq[int64], columns: seq[int], p: int64): bool =
  for c in columns:
    if v[c] != 0:
      let scale = invP(v[c], p)
      for x in v.mitems: x = mulP(x, scale, p)
      return true

proc canonicalVector(basis: seq[seq[int64]], normalize: seq[int], unknowns: int, p: int64): seq[int64] =
  ## The unique vector of the span in reduced echelon form whose pivot comes last
  ## when columns are ordered by normalize, then by decreasing index. It does not
  ## depend on the basis, so every prime reconstructs the same relation.
  var order = normalize
  for c in countdown(unknowns - 1, 0):
    if c notin normalize: order.add(c)
  var vectors = basis
  var pivots: seq[int]
  for c in order:
    var found = -1
    for i in pivots.len ..< vectors.len:
      if vectors[i][c] != 0:
        found = i
        break
    if found < 0: continue
    swap(vectors[found], vectors[pivots.len])
    let row = pivots.len
    let scale = invP(vectors[row][c], p)
    for x in vectors[row].mitems: x = mulP(x, scale, p)
    for i in 0 ..< vectors.len:
      if i != row and vectors[i][c] != 0:
        let factor = vectors[i][c]
        for j in 0 ..< unknowns: vectors[i][j] = subMod(vectors[i][j], mulP(factor, vectors[row][j], p), p)
    pivots.add(c)
    if pivots.len == vectors.len: break
  vectors[pivots.len - 1]

proc solvePrime(rows: seq[seq[int64]], unknowns, fitCount: int, p: int64,
                normalize: seq[int], holdout = Holdout, unique = false):
                tuple[ok: bool, vector: seq[int64], fitRows: int] =
  ## A null vector of the early equations. When several remain, the fit extends to
  ## every equation except the holdout and a canonical vector is chosen, unless the
  ## relation must be unique.
  var used = fitCount
  var basis = nullspace(rows[0 ..< used], unknowns, p)
  if basis.len > 1 and rows.len - holdout > used:
    used = rows.len - holdout
    basis = nullspace(rows[0 ..< used], unknowns, p)
  if basis.len == 0 or (unique and basis.len != 1): return
  var v = if basis.len == 1: basis[0] else: canonicalVector(basis, normalize, unknowns, p)
  if not normalizeVector(v, normalize, p): return
  for row in rows:
    if dot(row, v, p) != 0: return
  (true, v, used)

proc fitLinearModel(rows: array[3, seq[seq[int64]]], unknowns: int, normalize: seq[int],
                    spare = 2, holdout = Holdout, unique = false): Fit =
  ## Fit one null vector on early equations, then require it on every equation
  ## modulo all three primes. The defaults keep two spare fit equations and four
  ## unseen ones; short-data fits lower them and add other checks.
  let total = rows[0].len
  let fitCount = unknowns + spare
  if total < fitCount + holdout: return
  let first = solvePrime(rows[0], unknowns, fitCount, ExplorerPrimes[0], normalize, holdout, unique)
  if not first.ok: return
  result.residues = first.vector
  result.primeVectors[0] = first.vector
  result.fitRows = first.fitRows
  result.checkedRows = total - first.fitRows
  var rational = true
  var values: seq[Rational]
  for x in first.vector:
    let r = rationalOf(x, ExplorerPrimes[0])
    if not r.ok:
      rational = false
      break
    values.add((r.num, r.den))
  if rational:
    for k in 1 .. 2:
      let p = ExplorerPrimes[k]
      var v = newSeq[int64](unknowns)
      for i, r in values:
        let den = liftMod(r.den, p)
        if den == 0:
          rational = false
          break
        v[i] = mulP(liftMod(r.num, p), invP(den, p), p)
      if rational:
        for row in rows[k]:
          if dot(row, v, p) != 0:
            rational = false
            break
      if not rational: break
  if not rational:
    for k in 1 .. 2:
      let other = solvePrime(rows[k], unknowns, fitCount, ExplorerPrimes[k], normalize, holdout, unique)
      if not other.ok: return
      result.primeVectors[k] = other.vector
  result.rational = rational
  if rational: result.values = values
  result.ok = true

proc berlekampMassey(s: openArray[int64], p: int64): seq[int64] =
  ## Return c with s[n] = sum c[j] * s[n-1-j].
  var current = @[1'i64]
  var previous = @[1'i64]
  var order = 0
  var shift = 1
  var lastDiscrepancy = 1'i64
  for i in 0 ..< s.len:
    var d = s[i]
    for j in 1 .. order:
      if j < current.len: d = (d + current[j] * s[i - j]) mod p
    if d == 0:
      inc shift
      continue
    let saved = current
    let factor = mulP(d, invP(lastDiscrepancy, p), p)
    if current.len < previous.len + shift: current.setLen(previous.len + shift)
    for j, value in previous:
      current[j + shift] = subMod(current[j + shift], mulP(factor, value, p), p)
    if 2 * order <= i:
      order = i + 1 - order
      previous = saved
      lastDiscrepancy = d
      shift = 1
    else:
      inc shift
  if current.len < order + 1: current.setLen(order + 1)
  for j in 1 .. order: result.add(subMod(0, current[j], p))

proc predicts(s: openArray[int64], c: seq[int64], p: int64): bool =
  for i in c.len ..< s.len:
    var total = 0'i64
    for j, v in c: total = (total + v * s[i - 1 - j]) mod p
    if total != s[i]: return false
  true

# Text helpers. Coefficients are exact integers or reduced fractions.

proc ratText(r: Rational): string =
  if r.den == 1: $r.num else: $r.num & "/" & $r.den

proc scaledForm(values: seq[Rational]): tuple[common: BigInt, ints: seq[BigInt]] =
  ## The coefficients times their common denominator, which may pass 64 bits.
  result.common = big(1)
  for v in values:
    let d = big(v.den)
    result.common = divMod(result.common, gcdBig(result.common, d)).quotient * d
  for v in values: result.ints.add(big(v.num) * divMod(result.common, big(v.den)).quotient)

proc integerForm(values: seq[Rational]): tuple[ok: bool, ints: seq[BigInt]] =
  (true, scaledForm(values).ints)

proc exactQuotient(value, divisor: BigInt): BigInt =
  let (quotient, remainder) = divMod(value, divisor)
  if not remainder.isZero: raise newException(ValueError, "Division is not exact")
  quotient

proc combination(parts: seq[tuple[coefficient, symbol: string]]): string =
  var pieces: seq[string]
  for (coefficient, symbol) in parts:
    if coefficient == "0": continue
    let piece =
      if symbol.len == 0: coefficient
      elif coefficient == "1": symbol
      elif coefficient == "-1": "-" & symbol
      else: coefficient & "*" & symbol
    pieces.add(piece)
  if pieces.len == 0: return "0"
  result = pieces[0]
  for piece in pieces[1 .. ^1]:
    if piece.startsWith("-"): result.add(" - " & piece[1 .. ^1]) else: result.add(" + " & piece)

proc polynomialText(coefficients: seq[string], variable: string): string =
  var parts: seq[tuple[coefficient, symbol: string]]
  for j in countdown(coefficients.high, 0):
    let symbol = if j == 0: "" elif j == 1: variable else: variable & "^" & $j
    parts.add((coefficients[j], symbol))
  combination(parts)

proc negated(text: string): string =
  if text == "0": text elif text.startsWith("-"): text[1 .. ^1] else: "-" & text

proc relationText(blocks: seq[tuple[coefficients: seq[string], variable, factor: string]]): string =
  ## Join polynomial-weighted terms, pulling a leading minus sign out of each block.
  for (coefficients, variable, factor) in blocks:
    var leading = ""
    for j in countdown(coefficients.high, 0):
      if coefficients[j] != "0":
        leading = coefficients[j]
        break
    if leading.len == 0: continue
    let minus = leading.startsWith("-")
    let polynomial = polynomialText((if minus: coefficients.mapIt(negated(it)) else: coefficients), variable)
    let body = if factor.len == 0: polynomial
               elif polynomial == "1": factor
               elif polynomial.contains(' '): "(" & polynomial & ")*" & factor
               else: polynomial & "*" & factor
    if result.len == 0: result = (if minus: "-" else: "") & body
    else: result.add((if minus: " - " else: " + ") & body)
  if result.len == 0: result = "0"

proc termSymbol(label: string, shift: int): string =
  if shift == 0: label & "(n)" elif shift > 0: label & "(n+" & $shift & ")" else: label & "(n" & $shift & ")"

proc evidence(fit: Fit): JsonNode =
  %*{"status": "holds_on_unseen_terms", "fit_equations": fit.fitRows,
     "unseen_equations_checked": fit.checkedRows, "primes_checked": ExplorerPrimes.len,
     "coefficients": (if fit.rational: "rational" else: "residues_mod_" & $ExplorerPrimes[0])}

proc coefficientTexts(fit: Fit, scaleFree: bool): seq[string] =
  ## Integer form when the relation may be rescaled, reduced fractions otherwise.
  if not fit.rational: return fit.residues.mapIt($it)
  if scaleFree:
    let (ok, ints) = integerForm(fit.values)
    if ok: return ints.mapIt($it)
  fit.values.mapIt(ratText(it))

# Tracks hold one sequence along a transform path, modulo each prime.

proc newTrack(offset, length: int): Track =
  result.valid = true
  result.offset = offset
  for k in 0 .. 2: result.residues[k] = newSeq[int64](length)
  result.approx = newSeq[float64](length)

proc size*(t: Track): int = t.approx.len
proc valid*(t: Track): bool = t.valid

proc fingerprint*(t: Track, modulo: int64, window: int): seq[int64] =
  ## The early residues a program policy reads; a missing term shows as zero.
  let values = t.residues[0]
  result = newSeq[int64](window)
  for i in 0 ..< min(window, values.len): result[i] = values[i] mod modulo

proc rootTrack*(terms: seq[BigInt], offset: int): Track =
  result = newTrack(offset, terms.len)
  result.exact = terms
  for i, term in terms:
    for k, p in ExplorerPrimes: result.residues[k][i] = term.modSmall(p)
    result.approx[i] = term.toFloat()

proc mobiusTable(limit: int): seq[int] =
  result = newSeq[int](limit + 1)
  if limit >= 1: result[1] = 1
  for i in 1 .. limit:
    var j = 2 * i
    while j <= limit:
      result[j] -= result[i]
      j += i

type Transformed = tuple[ok: bool, values: seq[int64], offset: int]

proc binomialRows(n: int, q: int64): seq[seq[int64]] =
  result = newSeq[seq[int64]](n)
  for i in 0 ..< n:
    result[i] = newSeq[int64](i + 1)
    result[i][0] = 1
    result[i][i] = 1
    for j in 1 ..< i: result[i][j] = (result[i - 1][j - 1] + result[i - 1][j]) mod q

proc convolveMod(a, b: openArray[int64], q: int64): seq[int64]

const
  FastEulerModulus* = 4_611_686_018_427_387_904'i64   ## Moduli below 2^62 use NTT convolutions.
  MaxFastIndex = 2_000_000'i64             ## Index window of the NTT transforms.

proc fastBinomialModulus(q: int64): bool =
  ## The EGF route divides by k! for every k in the window, so q must be a prime past it.
  q > MaxFastIndex and q < FastEulerModulus and isProbablePrime(q)

proc applyBinomial(s: seq[int64], q: int64, alternate: bool): seq[int64] =
  ## b(i) = sum C(i,j) (+-1)^(i-j) s(j) over indices counted from the first term.
  let n = s.len
  if n > 2_000 and fastBinomialModulus(q):
    # As exponential generating functions B(x) = e^(+-x) S(x): one convolution.
    var factorial = newSeq[int64](n)
    factorial[0] = 1
    for i in 1 ..< n: factorial[i] = mulMod(factorial[i - 1], int64(i), q)
    var inverse = newSeq[int64](n)
    inverse[n - 1] = modularInverse(factorial[n - 1], q)
    for i in countdown(n - 1, 1): inverse[i - 1] = mulMod(inverse[i], int64(i), q)
    var scaled = newSeq[int64](n)
    var exponential = newSeq[int64](n)
    for i in 0 ..< n:
      scaled[i] = mulMod(s[i], inverse[i], q)
      exponential[i] = if alternate and i mod 2 == 1: (q - inverse[i]) mod q else: inverse[i]
    let product = convolveMod(scaled, exponential, q)
    result = newSeq[int64](n)
    for i in 0 ..< n: result[i] = mulMod(factorial[i], product[i], q)
    return
  result = newSeq[int64](n)
  if n <= 2_000:
    let rows = binomialRows(n, q)
    for i in 0 ..< n:
      for j in 0 .. i:
        var term = mulMod(rows[i][j], s[j], q)
        if alternate and (i - j) mod 2 == 1: term = (q - term) mod q
        result[i] = (result[i] + term) mod q
    return
  var factorial = newSeq[int64](n)
  factorial[0] = 1
  for i in 1 ..< n: factorial[i] = mulMod(factorial[i - 1], int64(i), q)
  var inverse = newSeq[int64](n)
  inverse[n - 1] = modularInverse(factorial[n - 1], q)
  for i in countdown(n - 1, 1): inverse[i - 1] = mulMod(inverse[i], int64(i), q)
  for i in 0 ..< n:
    for j in 0 .. i:
      var term = mulMod(mulMod(factorial[i], mulMod(inverse[j], inverse[i - j], q), q), s[j], q)
      if alternate and (i - j) mod 2 == 1: term = (q - term) mod q
      result[i] = (result[i] + term) mod q

# Convolution modulo any q below 2^62. The exact integer product of two residue lists has
# coefficients below len * q^2; it is computed modulo several NTT primes and rebuilt by
# Garner's mixed radix CRT, reduced modulo q on the way. Three primes (2^86) cover q below
# 2^31, six (2^177) every q below 2^62.

const
  NttPrimes = [(998_244_353'u64, 3'u64), (167_772_161'u64, 3'u64), (469_762_049'u64, 3'u64),
               (754_974_721'u64, 11'u64), (2_013_265_921'u64, 31'u64), (1_811_939_329'u64, 13'u64)]
    ## Each prime with a primitive root; all allow transforms of length 2^23.
  FastEulerFrom = 2_048                    ## Shorter transforms stay quadratic.

proc nttPrimeCount(q: int64): int = (if q < 2_147_483_648'i64: 3 else: 6)

proc powU(base, exponent, m: uint64): uint64 =
  var b = base mod m
  var e = exponent
  result = 1
  while e > 0:
    if (e and 1) == 1: result = result * b mod m
    b = b * b mod m
    e = e shr 1

{.push boundChecks: off, overflowChecks: off, rangeChecks: off.}   # the NTT hot loops

type Montgomery = object
  ## Arithmetic modulo an NTT prime below 2^31 in Montgomery form (R = 2^32): a product
  ## costs two multiplications and a shift instead of a 64-bit division.
  m, root, negInverse, r2: uint64
  roots: seq[seq[uint64]]          ## roots[level][k], forward, in Montgomery form
  inverseRoots: seq[seq[uint64]]

proc reduce(z: Montgomery, t: uint64): uint64 {.inline.} =
  let u = (t * z.negInverse) and 0xFFFF_FFFF'u64
  result = (t + u * z.m) shr 32
  if result >= z.m: result -= z.m

proc mul(z: Montgomery, a, b: uint64): uint64 {.inline.} = z.reduce(a * b)
proc toMont(z: Montgomery, a: uint64): uint64 {.inline.} = z.reduce((a mod z.m) * z.r2)

proc initMontgomery(m, root: uint64): Montgomery =
  result.m = m
  result.root = root
  var inverse = 1'u64                       # m^-1 mod 2^32 by Newton iteration
  for _ in 0 ..< 5: inverse = (inverse * (2'u64 - m * inverse)) and 0xFFFF_FFFF'u64
  result.negInverse = (0x1_0000_0000'u64 - inverse) and 0xFFFF_FFFF'u64
  result.r2 = (1'u64 shl 32) mod m
  result.r2 = result.r2 * result.r2 mod m

proc ensureRoots(z: var Montgomery, levels: int) =
  while z.roots.len < levels:
    let level = z.roots.len                  # transforms of length 2^(level+1)
    let length = 1'u64 shl (level + 1)
    let half = int(length div 2)
    let w = powU(z.root, (z.m - 1) div length, z.m)
    let wInverse = powU(w, z.m - 2, z.m)
    var forward = newSeq[uint64](half)
    var backward = newSeq[uint64](half)
    var a = 1'u64
    var b = 1'u64
    for k in 0 ..< half:
      forward[k] = z.toMont(a)
      backward[k] = z.toMont(b)
      a = a * w mod z.m
      b = b * wInverse mod z.m
    z.roots.add(forward)
    z.inverseRoots.add(backward)

var nttFields: array[NttPrimes.len, Montgomery]
var nttReady = false

proc prepareNtt(levels: int) =
  if not nttReady:
    for t, (m, root) in NttPrimes: nttFields[t] = initMontgomery(m, root)
    nttReady = true
  for t in 0 ..< NttPrimes.len: nttFields[t].ensureRoots(levels)

proc ntt(z: Montgomery, a: var seq[uint64], invert: bool) =
  let n = a.len
  var j = 0
  for i in 1 ..< n:
    var bit = n shr 1
    while (j and bit) != 0:
      j = j xor bit
      bit = bit shr 1
    j = j xor bit
    if i < j: swap(a[i], a[j])
  let m = z.m
  var length = 2
  var level = 0
  while length <= n:
    let half = length div 2
    let roots = if invert: unsafeAddr z.inverseRoots[level] else: unsafeAddr z.roots[level]
    var i = 0
    while i < n:
      for k in 0 ..< half:
        let x = a[i + k]
        let y = z.mul(a[i + k + half], roots[][k])
        a[i + k] = (if x + y >= m: x + y - m else: x + y)
        a[i + k + half] = (if x >= y: x - y else: x + m - y)
      i += length
    length = length shl 1
    inc level

type Spectrum = seq[seq[uint64]]   ## A residue list transformed modulo each NTT prime in use.

proc spectrumOf(values: openArray[int64], length: int, q: int64): Spectrum =
  ## Forward NTTs of values (zero padded to length, a power of two) in Montgomery form.
  var levels = 0
  while (1 shl levels) < length: inc levels
  prepareNtt(levels)
  result = newSeq[seq[uint64]](nttPrimeCount(q))
  for t in 0 ..< result.len:
    let z = addr nttFields[t]
    var data = newSeq[uint64](length)
    for i in 0 ..< min(values.len, length): data[i] = z[].toMont(uint64(values[i]))
    z[].ntt(data, false)
    result[t] = data

proc accumulate(target: var Spectrum, a, b: Spectrum, scale: uint64 = 1) =
  ## target += scale * a * b pointwise (scale a small plain integer).
  if target.len == 0: target = newSeq[seq[uint64]](a.len)
  for t in 0 ..< a.len:
    let z = addr nttFields[t]
    if target[t].len == 0: target[t] = newSeq[uint64](a[t].len)
    for i in 0 ..< a[t].len:
      var product = z[].mul(a[t][i], b[t][i])
      for _ in 2'u64 .. scale:
        product = (if product * 2 >= z[].m: product * 2 - z[].m else: product * 2)
      let sum = target[t][i] + product
      target[t][i] = (if sum >= z[].m: sum - z[].m else: sum)

proc residuesOf(spectrum: var Spectrum, first, last: int, q: int64): seq[int64] =
  ## Inverse NTTs; the values modulo q at indices first ..< last, by Garner's CRT.
  let length = spectrum[0].len
  let k = spectrum.len
  for t in 0 ..< k:
    let z = addr nttFields[t]
    z[].ntt(spectrum[t], true)
    let scale = powU(uint64(length), z[].m - 2, z[].m)
    for i in first ..< last: spectrum[t][i] = z[].mul(spectrum[t][i], scale)
  # Mixed radix x = d0 + p0 (d1 + p1 (d2 + ...)): d_i follows from the residue modulo p_i
  # minus the earlier digits' part, times the inverse of p0 ... p_{i-1}.
  var inverses = newSeq[uint64](k)              # (p0 ... p_{i-1})^-1 mod p_i
  var radixModQ = newSeq[int64](k)              # p0 ... p_{i-1} mod q
  var running = 1'i64 mod q
  for i in 0 ..< k:
    let p = NttPrimes[i][0]
    var product = 1'u64
    for j in 0 ..< i: product = product * (NttPrimes[j][0] mod p) mod p
    inverses[i] = powU(product, p - 2, p)
    radixModQ[i] = running
    running = mulMod(running, int64(NttPrimes[i][0] mod uint64(q)), q)
  result = newSeq[int64](last - first)
  var digits = newSeq[uint64](k)
  for index in first ..< last:
    var value = 0'i64
    for i in 0 ..< k:
      let p = NttPrimes[i][0]
      # The earlier digits' contribution modulo p, by Horner from the top digit down.
      var earlier = 0'u64
      for j in countdown(i - 1, 0): earlier = (earlier * (NttPrimes[j][0] mod p) + digits[j]) mod p
      digits[i] = (spectrum[i][index] + p - earlier) mod p * inverses[i] mod p
      value = (value + mulMod(int64(digits[i]), radixModQ[i], q)) mod q
    result[index - first] = value

{.pop.}

proc convolveMod(a, b: openArray[int64], q: int64): seq[int64] =
  ## The product of two residue lists modulo q < 2^62.
  if a.len == 0 or b.len == 0: return @[]
  if min(a.len, b.len) <= 32:
    result = newSeq[int64](a.len + b.len - 1)
    for i, x in a:
      if x == 0: continue
      for j, y in b: result[i + j] = (result[i + j] + mulMod(x, y, q)) mod q
    return
  var size = 1
  while size < a.len + b.len - 1: size = size shl 1
  var product: Spectrum
  product.accumulate(spectrumOf(a, size, q), spectrumOf(b, size, q))
  product.residuesOf(0, a.len + b.len - 1, q)

proc eulerFromWeights(c: seq[int64], count: int, q: int64): seq[int64] =
  ## n a(n) = sum_{k=1..n} c(k) a(n-k), solved online. A CDQ split over power-of-two
  ## blocks hands each finished half [low, mid) to one cyclic convolution of length
  ## L = high - low with c[0, L): wrapped terms land below L/2, so the outputs for
  ## [mid, high) are exact. The transforms of c[0, L) are shared by every block of that
  ## length. Cost O(n log^2 n) with two NTTs of length L per prime and block.
  var total = 1
  while total < count: total = total shl 1
  var inverses = newSeq[int64](total)
  for n in 1 ..< total: inverses[n] = if n < count: modularInverse(int64(n) mod q, q) else: 0
  var f = newSeq[int64](total)
  var acc = newSeq[int64](total)
  var cached = initTable[int, Spectrum]()
  f[0] = 1 mod q
  proc solve(low, high: int) =
    if high - low <= 64:
      for n in low ..< high:
        if n > 0: f[n] = mulMod(acc[n], inverses[n], q)
        for m in n + 1 ..< high:
          if m - n < c.len: acc[m] = (acc[m] + mulMod(f[n], c[m - n], q)) mod q
      return
    let middle = (low + high) div 2
    solve(low, middle)
    let length = high - low
    if length notin cached: cached[length] = spectrumOf(c.toOpenArray(0, min(length, c.len) - 1), length, q)
    var product: Spectrum
    product.accumulate(spectrumOf(f.toOpenArray(low, middle - 1), length, q), cached[length])
    let values = product.residuesOf(length div 2, length, q)
    for i, value in values:
      let n = low + length div 2 + i
      acc[n] = (acc[n] + value) mod q
    solve(middle, high)
  solve(0, total)
  f.setLen(count)
  f

proc eulerTransform(b: seq[int64], count: int, q: int64): seq[int64] =
  ## a(0) = 1 and prod (1 - x^k)^(-b(k)) = sum a(n) x^n, with b(k) = b[k - 1].
  var c = newSeq[int64](count)
  for d in 1 ..< count:
    let weight = mulMod(int64(d) mod q, b[d - 1], q)
    var k = d
    while k < count:
      c[k] = (c[k] + weight) mod q
      k += d
  if count > FastEulerFrom and q < FastEulerModulus: return eulerFromWeights(c, count, q)
  result = newSeq[int64](count)
  result[0] = 1 mod q
  for n in 1 ..< count:
    var total = 0'i64
    for k in 1 .. n: total = (total + mulMod(c[k], result[n - k], q)) mod q
    result[n] = mulMod(total, modularInverse(int64(n) mod q, q), q)

proc transformResidues(s: seq[int64], offset: int, t: Transform, q: int64): Transformed =
  ## One transform modulo q. Indices keep their meaning: offset is the index of values[0].
  let n = s.len
  if n < 2: return
  case t
  of tDifference:
    result.values = newSeq[int64](n - 1)
    for i in 0 ..< n - 1: result.values[i] = ((s[i + 1] - s[i]) mod q + q) mod q
    result.offset = offset
  of tPartialSum:
    result.values = newSeq[int64](n)
    var total = 0'i64
    for i in 0 ..< n:
      total = (total + s[i]) mod q
      result.values[i] = total
    result.offset = offset
  of tRatio:
    result.values = newSeq[int64](n - 1)
    for i in 0 ..< n - 1:
      if s[i] == 0 or gcd(s[i], q) != 1: return
      result.values[i] = mulMod(s[i + 1], modularInverse(s[i], q), q)
    result.offset = offset
  of tDivideFactorial, tMultiplyFactorial:
    if offset < 0: return
    result.values = newSeq[int64](n)
    var factorial = 1'i64 mod q
    for m in 2 .. offset: factorial = mulMod(factorial, int64(m) mod q, q)
    for i in 0 ..< n:
      if i > 0: factorial = mulMod(factorial, int64(offset + i) mod q, q)
      if factorial == 0 or gcd(factorial, q) != 1: return
      let scale = if t == tDivideFactorial: modularInverse(factorial, q) else: factorial
      result.values[i] = mulMod(s[i], scale, q)
    result.offset = offset
  of tBinomial, tInverseBinomial:
    result.values = applyBinomial(s, q, t == tInverseBinomial)
    result.offset = offset
  of tMobius:
    if offset > 1: return
    let start = 1 - offset
    let count = n - start
    if count < 2: return
    let mu = mobiusTable(count)
    result.values = newSeq[int64](count)
    for d in 1 .. count:
      var m = d
      while m <= count:
        let sign = mu[m div d]
        if sign > 0: result.values[m - 1] = (result.values[m - 1] + s[start + d - 1]) mod q
        elif sign < 0: result.values[m - 1] = (result.values[m - 1] - s[start + d - 1] + q) mod q
        m += d
    result.offset = 1
  of tInverseEuler:
    if offset != 0 or s[0] != 1 mod q: return
    let count = n - 1
    let mu = mobiusTable(count)
    var c = newSeq[int64](count + 1)
    for m in 1 .. count:
      var value = mulMod(int64(m) mod q, s[m], q)
      for j in 1 ..< m: value = (value - mulMod(c[j], s[m - j], q) + q) mod q
      c[m] = value
    result.values = newSeq[int64](count)
    for m in 1 .. count:
      var total = 0'i64
      for d in 1 .. m:
        if m mod d != 0 or mu[m div d] == 0: continue
        total = if mu[m div d] > 0: (total + c[d]) mod q else: (total - c[d] + q) mod q
      if gcd(int64(m), q) != 1: return
      result.values[m - 1] = mulMod(total, modularInverse(int64(m) mod q, q), q)
    result.offset = 1
  result.ok = true

proc transformApprox(a: seq[float64], offset: int, t: Transform): seq[float64] =
  let n = a.len
  case t
  of tDifference:
    for i in 0 ..< n - 1: result.add(a[i + 1] - a[i])
  of tPartialSum:
    var total = 0.0
    for value in a:
      total += value
      result.add(total)
  of tRatio:
    for i in 0 ..< n - 1: result.add(if a[i] != 0: a[i + 1] / a[i] else: 0)
  of tDivideFactorial, tMultiplyFactorial:
    for i in 0 ..< n:
      let logFactorial = lgamma(float64(offset + i + 1))
      result.add(a[i] * exp(if t == tDivideFactorial: -logFactorial else: logFactorial))
  of tBinomial, tInverseBinomial:
    var row = @[1.0]
    for i in 0 ..< n:
      var total = 0.0
      for j in 0 .. i:
        let sign = if t == tInverseBinomial and (i - j) mod 2 == 1: -1.0 else: 1.0
        total += sign * row[j] * a[j]
      result.add(total)
      var next = newSeq[float64](i + 2)
      next[0] = 1
      next[i + 1] = 1
      for j in 1 .. i: next[j] = row[j - 1] + row[j]
      row = next
  of tMobius:
    let start = 1 - offset
    let count = n - start
    let mu = mobiusTable(count)
    for m in 1 .. count:
      var total = 0.0
      for d in 1 .. m:
        if m mod d == 0: total += float64(mu[m div d]) * a[start + d - 1]
      result.add(total)
  of tInverseEuler:
    let count = n - 1
    let mu = mobiusTable(count)
    var c = newSeq[float64](count + 1)
    for m in 1 .. count:
      var value = float64(m) * a[m]
      for j in 1 ..< m: value -= c[j] * a[m - j]
      c[m] = value
    for m in 1 .. count:
      var total = 0.0
      for d in 1 .. m:
        if m mod d == 0: total += float64(mu[m div d]) * c[d]
      result.add(total / float64(m))

proc transformExact(a: Track, t: Transform): seq[BigInt] =
  ## Exact integer values of the simple transforms, empty when they are not integers.
  ## Short-data fits on a transformed track need them for their integer test.
  if a.exact.len != a.size: return
  let n = a.size
  var values: seq[BigInt]
  case t
  of tDifference:
    for i in 0 ..< n - 1: values.add(a.exact[i + 1] - a.exact[i])
  of tPartialSum:
    var total = big(0)
    for x in a.exact:
      total = total + x
      values.add(total)
  of tRatio:
    for i in 0 ..< n - 1:
      if a.exact[i].isZero: return
      let (quotient, remainder) = divMod(a.exact[i + 1], a.exact[i])
      if not remainder.isZero: return
      values.add(quotient)
  of tDivideFactorial, tMultiplyFactorial:
    if a.offset < 0: return
    var factorial = big(1)
    for m in 2 .. a.offset: factorial = factorial * big(m)
    for i in 0 ..< n:
      if i > 0: factorial = factorial * big(a.offset + i)
      if t == tMultiplyFactorial:
        values.add(a.exact[i] * factorial)
      else:
        let (quotient, remainder) = divMod(a.exact[i], factorial)
        if not remainder.isZero: return
        values.add(quotient)
  else: return
  values

proc transformed(a: Track, t: Transform): Track =
  if not a.valid or a.size < 2: return
  var parts: array[3, Transformed]
  for k, p in ExplorerPrimes:
    parts[k] = transformResidues(a.residues[k], a.offset, t, p)
    if not parts[k].ok: return
  result = newTrack(parts[0].offset, parts[0].values.len)
  for k in 0 .. 2: result.residues[k] = parts[k].values
  result.approx = transformApprox(a.approx, a.offset, t)
  let exact = transformExact(a, t)
  if exact.len == result.size: result.exact = exact

# Hypothesis tests.

proc testPolynomial(t: Track, label: string): Outcome =
  let n = t.size
  var current = t.residues[0]
  var degree = -1
  for d in 0 .. MaxPolynomialFitDegree:
    if d + 4 + Holdout > n: break
    var next = newSeq[int64](current.len - 1)
    for i in 0 ..< next.len: next[i] = subMod(current[i + 1], current[i], ExplorerPrimes[0])
    current = next
    if current.allIt(it == 0):
      degree = d
      break
  if degree < 0: return
  let unknowns = degree + 2
  var rows: array[3, seq[seq[int64]]]
  for k, p in ExplorerPrimes:
    for i in 0 ..< n:
      var row = newSeq[int64](unknowns)
      let x = liftMod(int64(t.offset + i), p)
      var power = 1'i64
      for j in 0 .. degree:
        row[j] = power
        power = mulP(power, x, p)
      row[degree + 1] = t.residues[k][i]
      rows[k].add(row)
  let fit = fitLinearModel(rows, unknowns, @[degree + 1])
  if not fit.ok: return
  result.ok = true
  var texts: seq[string]
  if fit.rational:
    result.form = TargetForm(kind: hPolynomial, available: true)
    for j in 0 .. degree:
      let value = (-fit.values[j].num, fit.values[j].den)
      result.form.polynomial.add(value)
      texts.add(ratText(value))
  else:
    for j in 0 .. degree: texts.add($subMod(0, fit.residues[j], ExplorerPrimes[0]))
  result.report = %*{"formula": label & "(n) = " & polynomialText(texts, "n"), "degree": degree,
                     "evidence": evidence(fit)}

proc testLinear(t: Track, label: string): Outcome =
  let n = t.size
  let fitLength = n - Holdout
  let p0 = ExplorerPrimes[0]
  let coefficients = berlekampMassey(t.residues[0][0 ..< fitLength], p0)
  let order = coefficients.len
  if order == 0 or 2 * order + 2 > fitLength: return
  if not predicts(t.residues[0], coefficients, p0): return
  var rational = true
  var values: seq[Rational]
  for c in coefficients:
    let r = rationalOf(c, p0)
    if not r.ok:
      rational = false
      break
    values.add((r.num, r.den))
  if rational:
    for k in 1 .. 2:
      let p = ExplorerPrimes[k]
      var mapped: seq[int64]
      for r in values:
        let den = liftMod(r.den, p)
        if den == 0:
          rational = false
          break
        mapped.add(mulP(liftMod(r.num, p), invP(den, p), p))
      if not rational or not predicts(t.residues[k], mapped, p):
        rational = false
        break
  if not rational:
    for k in 1 .. 2:
      let other = berlekampMassey(t.residues[k][0 ..< fitLength], ExplorerPrimes[k])
      if other.len != order or not predicts(t.residues[k], other, ExplorerPrimes[k]): return
  result.ok = true
  var parts: seq[tuple[coefficient, symbol: string]]
  for j in 0 ..< order:
    let text = if rational: ratText(values[j]) else: $coefficients[j]
    parts.add((text, termSymbol(label, -(j + 1))))
  if rational: result.form = TargetForm(kind: hLinearRecurrence, available: true, linear: values)
  result.report = %*{"formula": termSymbol(label, 0) & " = " & combination(parts), "order": order,
    "valid_from_n": t.offset + order,
    "evidence": {"status": "holds_on_unseen_terms", "fit_terms": fitLength, "unseen_terms_checked": Holdout,
                 "primes_checked": ExplorerPrimes.len,
                 "coefficients": (if rational: "rational" else: "residues_mod_" & $p0)}}

proc blockZero(v: seq[int64], index, width: int): bool =
  for j in 0 ..< width:
    if v[index * width + j] != 0: return false
  true

var holonomicShapes, algebraicShapes: seq[tuple[a, b: int]]

proc shapes(low, highA, highB, lowB: int): seq[tuple[a, b: int]] =
  for a in low .. highA:
    for b in lowB .. highB: result.add((a, b))
  result.sort(proc(x, y: tuple[a, b: int]): int =
    cmp((x.a + 1) * (x.b + 1), (y.a + 1) * (y.b + 1)) * 16 + cmp(x.a, y.a))

proc holonomicRows(t: Track, r, d: int): array[3, seq[seq[int64]]] =
  ## One equation per n: sum over shifts s of P_s(n) * t(n+s) = 0, with P_s of degree d.
  let width = d + 1
  let unknowns = (r + 1) * width
  for k, p in ExplorerPrimes:
    for i in 0 ..< t.size - r:
      var row = newSeq[int64](unknowns)
      let x = liftMod(int64(t.offset + i), p)
      for shift in 0 .. r:
        var power = 1'i64
        for j in 0 .. d:
          row[shift * width + j] = mulP(power, t.residues[k][i + shift], p)
          power = mulP(power, x, p)
      result[k].add(row)

proc holonomicNormalize(r, d: int): seq[int] =
  for j in countdown(d, 0): result.add(r * (d + 1) + j)

proc degenerateLead(t: Track, residues: seq[int64], r, d: int): bool =
  ## A leading coefficient that vanishes where the new term is nonzero leaves that term
  ## unexplained. Sparse sequences admit such degenerate fits that hold on every given
  ## term and fail later, so allow it only in the first two equations (initial values).
  let p0 = ExplorerPrimes[0]
  for i in 2 ..< t.size - r:
    let x = liftMod(int64(t.offset + i), p0)
    var lead = 0'i64
    for j in countdown(d, 0): lead = (mulP(lead, x, p0) + residues[r * (d + 1) + j]) mod p0
    if lead == 0 and t.residues[0][i + r] != 0: return true

proc holonomicOutcome(fit: Fit, r, d: int, label: string): Outcome =
  let width = d + 1
  result.ok = true
  let texts = coefficientTexts(fit, scaleFree = true)
  var blocks: seq[tuple[coefficients: seq[string], variable, factor: string]]
  for shift in countdown(r, 0):
    blocks.add((texts[shift * width ..< (shift + 1) * width], "n", termSymbol(label, shift)))
  if fit.rational:
    result.form = TargetForm(kind: hHolonomic, available: true)
    for shift in 0 .. r: result.form.blocks.add(fit.values[shift * width ..< (shift + 1) * width])
  else:
    result.form = TargetForm(kind: hHolonomic, modular: true)
    for k in 0 .. 2:
      for shift in 0 .. r:
        result.form.modularBlocks[k].add(fit.primeVectors[k][shift * width ..< (shift + 1) * width])
  result.report = %*{"formula": relationText(blocks) & " = 0",
    "order": r, "polynomial_degree": d, "evidence": evidence(fit)}

proc testHolonomic(t: Track, label: string): Outcome =
  if holonomicShapes.len == 0: holonomicShapes = shapes(1, MaxHolonomicOrder, MaxHolonomicDegree, 1)
  let n = t.size
  for (r, d) in holonomicShapes:
    let width = d + 1
    let unknowns = (r + 1) * width
    if n - r < unknowns + 2 + Holdout: continue
    let fit = fitLinearModel(holonomicRows(t, r, d), unknowns, holonomicNormalize(r, d))
    if not fit.ok or blockZero(fit.residues, 0, width) or blockZero(fit.residues, r, width): continue
    if degenerateLead(t, fit.residues, r, d): continue
    return holonomicOutcome(fit, r, d, label)

proc predictsIntegers(blocks: seq[seq[Rational]], t: Track, count: int): bool =
  ## Iterate the relation exactly past the data and require every new term to be an
  ## integer. A relation forced through too few equations almost never does this.
  let r = blocks.high
  let n = t.size
  var flat: seq[Rational]
  for b in blocks: flat.add(b)
  let (ok, ints) = integerForm(flat)
  if not ok: return false
  var polys: seq[seq[BigInt]]
  var position = 0
  for b in blocks:
    polys.add(ints[position ..< position + b.len])
    position += b.len
  proc evalAt(coefficients: seq[BigInt], x: int64): BigInt =
    result = big(0)
    for j in countdown(coefficients.high, 0): result = result * big(x) + coefficients[j]
  var window = t.exact[n - r ..< n]
  for i in n ..< n + count:
    let nAbs = int64(t.offset + i - r)
    var total = big(0)
    for k in 0 ..< r: total = total + evalAt(polys[k], nAbs) * window[k]
    let lead = evalAt(polys[r], nAbs)
    if lead.isZero: return false
    let (quotient, remainder) = divMod(-total, lead)
    if not remainder.isZero: return false
    for k in 0 ..< r - 1: window[k] = window[k + 1]
    window[r - 1] = quotient
  true

proc testHolonomicShort(t: Track, label: string): Outcome =
  ## Shapes the regular test skips for lack of terms. The relation must be the only one
  ## of its shape, hold on every given term modulo three primes, and keep producing
  ## integers for IntegralityTerms more terms. A constant leading coefficient makes the
  ## integer test empty, so then two spare equations are still required.
  if t.exact.len == 0: return
  if holonomicShapes.len == 0: holonomicShapes = shapes(1, MaxHolonomicOrder, MaxHolonomicDegree, 1)
  let n = t.size
  for (r, d) in holonomicShapes:
    let width = d + 1
    let unknowns = (r + 1) * width
    let rowCount = n - r
    if rowCount >= unknowns + 2 + Holdout or rowCount < unknowns: continue
    let fit = fitLinearModel(holonomicRows(t, r, d), unknowns, holonomicNormalize(r, d),
                             spare = 0, holdout = 0, unique = true)
    if not fit.ok or not fit.rational: continue
    if blockZero(fit.residues, 0, width) or blockZero(fit.residues, r, width): continue
    if degenerateLead(t, fit.residues, r, d): continue
    var blocks: seq[seq[Rational]]
    for shift in 0 .. r: blocks.add(fit.values[shift * width ..< (shift + 1) * width])
    let constantLead = blocks[r][1 .. ^1].allIt(it.num == 0)
    if constantLead and rowCount < unknowns + 2: continue
    if not predictsIntegers(blocks, t, IntegralityTerms): continue
    result = holonomicOutcome(fit, r, d, label)
    result.report["evidence"] = %*{"status": "fits_every_term_and_predicts_integers",
      "equations": rowCount, "unknowns": unknowns, "spare_equations": rowCount - unknowns,
      "unique_relation_of_this_shape": true, "integer_terms_predicted": IntegralityTerms,
      "primes_checked": ExplorerPrimes.len, "coefficients": "rational",
      "caution": "Fitted with few spare terms; more terms would make it stronger."}
    return

proc testAlgebraic(t: Track, label: string): Outcome =
  if algebraicShapes.len == 0: algebraicShapes = shapes(2, 3, 4, 0)
  let n = t.size
  var powers: array[3, seq[seq[int64]]]
  for k, p in ExplorerPrimes:
    powers[k] = newSeq[seq[int64]](4)
    powers[k][0] = newSeq[int64](n)
    powers[k][0][0] = 1
    for i in 1 .. 3:
      powers[k][i] = newSeq[int64](n)
      for a in 0 ..< n:
        if powers[k][i - 1][a] == 0: continue
        for b in 0 ..< n - a:
          powers[k][i][a + b] = addMod(powers[k][i][a + b], mulP(powers[k][i - 1][a], t.residues[k][b], p), p)
  for (dy, dx) in algebraicShapes:
    let width = dx + 1
    let unknowns = (dy + 1) * width
    if n < unknowns + 2 + Holdout: continue
    var rows: array[3, seq[seq[int64]]]
    for k in 0 .. 2:
      for c in 0 ..< n:
        var row = newSeq[int64](unknowns)
        for i in 0 .. dy:
          for j in 0 .. dx:
            if c >= j: row[i * width + j] = powers[k][i][c - j]
        rows[k].add(row)
    var normalize: seq[int]
    for j in countdown(dx, 0): normalize.add(dy * width + j)
    let fit = fitLinearModel(rows, unknowns, normalize)
    if not fit.ok: continue
    result.ok = true
    let texts = coefficientTexts(fit, scaleFree = true)
    var blocks: seq[tuple[coefficients: seq[string], variable, factor: string]]
    for i in countdown(dy, 0):
      blocks.add((texts[i * width ..< (i + 1) * width], "x", (if i == 0: "" elif i == 1: "F" else: "F^" & $i)))
    result.report = %*{"formula": relationText(blocks) & " = 0",
      "generating_function": "F = sum of " & label & "(n)*x^(n-" & $t.offset & ") for n >= " & $t.offset,
      "degree_in_F": dy, "degree_in_x": dx, "evidence": evidence(fit)}
    return

proc testMultiplicative(t: Track, label: string): Outcome =
  if t.exact.len == 0 or t.offset > 1 or t.offset + t.size - 1 < 12: return
  let maxIndex = t.offset + t.size - 1
  template at(m: int): BigInt = t.exact[m - t.offset]
  if at(1) != big(1): return
  var checks = 0
  var complete = true
  for m in 2 .. maxIndex:
    for k in m .. maxIndex div m:
      let product = at(m) * at(k)
      if gcd(m, k) == 1:
        if at(m * k) != product: return
        inc checks
      elif complete and at(m * k) != product:
        complete = false
  if checks < 6: return
  result.ok = true
  var values = newJObject()
  for p in 2 .. maxIndex:
    if not isPrime(p): continue
    var powerValues = newJArray()
    var q = p
    while q <= maxIndex:
      powerValues.add(at(q).toJson())
      if q > maxIndex div p: break
      q *= p
    values[$p] = powerValues
    if values.len == 12: break
  result.report = %*{"formula": label & "(m*n) = " & label & "(m)*" & label & "(n) for coprime m, n",
    "completely_multiplicative": complete, "values_at_prime_powers": values,
    "evidence": {"status": "holds_on_every_coprime_pair_in_range", "pairs_checked": checks}}
  result.form = TargetForm(kind: hMultiplicative)
  # Far terms need the values at prime powers past the data. A completely multiplicative
  # function whose value at primes is a polynomial in p is fixed by that polynomial.
  if complete:
    var primes: seq[int]
    for p in 2 .. maxIndex:
      if isPrime(p): primes.add(p)
    for degree in 0 .. 3:
      let unknowns = degree + 2
      if primes.len < unknowns + 2 + Holdout: break
      var rows: array[3, seq[seq[int64]]]
      for k, q in ExplorerPrimes:
        for p in primes:
          var row = newSeq[int64](unknowns)
          var power = 1'i64
          for j in 0 .. degree:
            row[j] = power
            power = mulP(power, int64(p) mod q, q)
          row[degree + 1] = at(p).modSmall(q)
          rows[k].add(row)
      let fit = fitLinearModel(rows, unknowns, @[degree + 1])
      if not fit.ok or not fit.rational: continue
      var polynomial: seq[Rational]
      for j in 0 .. degree: polynomial.add((-fit.values[j].num, fit.values[j].den))
      result.form.multiplicativeModel = mmCompletePrime
      result.form.multiplicativePolynomial = polynomial
      result.report["far_terms"] = %(label & "(p) = " & polynomialText(polynomial.mapIt(ratText(it)), "p") &
        " for primes p (checked on " & $primes.len & " primes), and " & label & "(p^e) = " & label & "(p)^e")
      return
  # Otherwise the divisor sum g = 1 * a may be a polynomial for m >= 2, as for phi (g(m) = m)
  # and mu (g(m) = 0). Then a(p^e) = g(p^e) - g(p^(e-1)) with g(1) = a(1) = 1.
  var sums = newSeq[BigInt](maxIndex + 1)
  for m in 0 .. maxIndex: sums[m] = big(0)
  for d in 1 .. maxIndex:
    var m = d
    while m <= maxIndex:
      sums[m] = sums[m] + at(d)
      m += d
  let divisorSums = rootTrack(sums[2 .. maxIndex], 2)
  let fitted = testPolynomial(divisorSums, "g")
  if fitted.ok and fitted.form.available:
    result.form.multiplicativeModel = mmDivisorSum
    result.form.multiplicativePolynomial = fitted.form.polynomial
    result.report["far_terms"] = %("g(m) = sum_{d|m} " & label & "(d) = " &
      polynomialText(fitted.form.polynomial.mapIt(ratText(it)), "m") & " for m >= 2 (holds on unseen terms), so " &
      label & "(p^e) = g(p^e) - g(p^(e-1)) with g(1) = 1")

proc testPeriodic(t: Track, moduli: seq[int64], label: string): Outcome =
  if t.exact.len == 0: return
  let n = t.size
  var found = newJArray()
  for m in moduli:
    let r = t.exact.mapIt(it.modSmall(m))
    for period in 1 .. n div 3:
      var lastMismatch = -1
      for i in countdown(n - 1 - period, 0):
        if r[i + period] != r[i]:
          lastMismatch = i
          break
      let start = lastMismatch + 1
      if start <= n div 4 and start + 2 * period + Holdout <= n:
        found.add(%*{"modulus": m, "period": period, "from_n": t.offset + start,
                     "pattern": r[start ..< start + period]})
        break
  if found.len == 0: return
  result.ok = true
  result.report = %*{"formula": label & "(n) mod m is eventually periodic", "residue_patterns": found,
    "evidence": {"status": "at_least_two_full_periods_plus_unseen_terms", "terms": n}}

# Self-referential relations: a(n+1) is a fixed combination of transforms of the terms
# so far, such as the Bell recurrence a(n+1) = sum C(n,k) a(k) or the rooted-tree
# Euler product. Such sequences need not satisfy any polynomial-coefficient recurrence.
# Indices n count from the first supplied term.

const SelfFeatureCount = 12

proc selfFeatureText(f: int, label: string): string =
  ## Features 9 to 11 are the quadratic terms of a polynomial map such as a(n+1) = a(n)^2 - a(n) + 1.
  case f
  of 0: label & "(n)"
  of 1: "n*" & label & "(n)"
  of 2: label & "(n-1)"
  of 3: "sum_{k=0..n} C(n,k)*" & label & "(k)"
  of 4: "sum_{k=0..n} " & label & "(k)*" & label & "(n-k)"
  of 5: "sum_{k=0..n} C(n,k)*" & label & "(k)*" & label & "(n-k)"
  of 6: "[x^n] prod_{k>=1} (1-x^k)^(-" & label & "(k))"
  of 7: "[x^(n+1)] prod_{k>=1} (1-x^k)^(-" & label & "(k-1))"
  of 9: label & "(n)^2"
  of 10: label & "(n)*" & label & "(n-1)"
  of 11: label & "(n-1)^2"
  else: ""

proc selfInverse(n: int, q: int64): int64 =
  let value = int64(n) mod q
  if value == 0 or gcd(value, q) != 1:
    raise newException(ValueError, "The Euler product divides by n = " & $n & ", which is not invertible modulo " & $q)
  modularInverse(value, q)

proc selfRun(u: var seq[int64], count: int, features: seq[int], coefficients: seq[int64], q: int64,
             collect: bool): seq[seq[int64]] =
  ## Walk n = 0 .. count-2 and compute the features of u(0..n), each in O(n). With
  ## coefficients, u(n+1) = sum c_j * feature_j(n) fills every term not known yet.
  ## With collect, return each step's feature values.
  var need: set[0 .. SelfFeatureCount - 1]
  for f in features: need.incl(f)
  let pascal = 3 in need or 5 in need
  var row = @[1'i64 mod q]
  var c6, e6, c7, e7: seq[int64]
  if 6 in need:
    c6 = newSeq[int64](count + 1)
    e6 = newSeq[int64](count + 1)
    e6[0] = 1 mod q
  if 7 in need:
    c7 = newSeq[int64](count + 2)
    e7 = newSeq[int64](count + 2)
    e7[0] = 1 mod q
  for i in 0 .. count - 2:
    if pascal and i > 0:
      var next = newSeq[int64](i + 1)
      next[0] = 1 mod q
      next[i] = 1 mod q
      for k in 1 ..< i: next[k] = (row[k - 1] + row[k]) mod q
      row = next
    var values = newSeq[int64](SelfFeatureCount)
    for f in features:
      case f
      of 0: values[0] = u[i]
      of 1: values[1] = mulMod(int64(i) mod q, u[i], q)
      of 2: values[2] = (if i > 0: u[i - 1] else: 0'i64)
      of 3:
        var total = 0'i64
        for k in 0 .. i: total = (total + mulMod(row[k], u[k], q)) mod q
        values[3] = total
      of 4:
        var total = 0'i64
        for k in 0 .. i: total = (total + mulMod(u[k], u[i - k], q)) mod q
        values[4] = total
      of 5:
        var total = 0'i64
        for k in 0 .. i: total = (total + mulMod(row[k], mulMod(u[k], u[i - k], q), q)) mod q
        values[5] = total
      of 6:
        # Product over k >= 1 of (1 - x^k)^(-u(k)): c(m) = sum over d | m of d*u(d).
        if i >= 1:
          let weight = mulMod(int64(i) mod q, u[i], q)
          var m = i
          while m <= count:
            c6[m] = (c6[m] + weight) mod q
            m += i
          var total = 0'i64
          for k in 1 .. i: total = (total + mulMod(c6[k], e6[i - k], q)) mod q
          e6[i] = mulMod(total, selfInverse(i, q), q)
        values[6] = e6[i]
      of 7:
        # Product over k >= 1 of (1 - x^k)^(-u(k-1)), read at x^(n+1).
        let k1 = i + 1
        let weight = mulMod(int64(k1) mod q, u[i], q)
        var m = k1
        while m <= count + 1:
          c7[m] = (c7[m] + weight) mod q
          m += k1
        var total = 0'i64
        for k in 1 .. k1: total = (total + mulMod(c7[k], e7[k1 - k], q)) mod q
        e7[k1] = mulMod(total, selfInverse(k1, q), q)
        values[7] = e7[k1]
      of 8: values[8] = 1 mod q
      of 9: values[9] = mulMod(u[i], u[i], q)
      of 10: values[10] = (if i > 0: mulMod(u[i], u[i - 1], q) else: 0'i64)
      of 11: values[11] = (if i > 0: mulMod(u[i - 1], u[i - 1], q) else: 0'i64)
      else: discard
    if collect: result.add(values)
    if coefficients.len > 0 and i + 1 >= u.len:
      var next = 0'i64
      for j, f in features: next = (next + mulMod(coefficients[j], values[f], q)) mod q
      u.add(next)

proc testSelfReferential(t: Track, label: string, short = false): Outcome =
  ## Try a single transform feature, then one with a(n), n*a(n), a(n-1) and 1, then all
  ## features, each fitted from n = 0, 1 or 2. The first relation that holds wins.
  ## The short pass covers what the regular one skips for lack of terms: only small
  ## feature sets, one spare and two unseen equations, and the relation must be unique.
  let n = t.size
  if n < (if short: MinExplorerTerms else: 10): return
  var all: array[3, seq[seq[int64]]]
  try:
    for k, p in ExplorerPrimes:
      var u = t.residues[k]
      all[k] = selfRun(u, n, @[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], @[], p, collect = true)
  except ValueError:
    return
  var featureSets: seq[seq[int]]
  for f in 3 .. 7: featureSets.add(@[f])
  featureSets.add(@[9, 0, 8])
  if short:
    for f in 3 .. 7: featureSets.add(@[f, 0, 8])
  else:
    featureSets.add(@[9, 10, 11, 0, 2, 8])
    for f in 3 .. 7: featureSets.add(@[f, 0, 1, 2, 8])
    featureSets.add(@[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
  for features in featureSets:
    let unknowns = features.len + 1
    for start in 0 .. 2:
      let available = n - 1 - start
      var (spare, holdout) = (2, Holdout)
      if short:
        if available >= unknowns + 2 + Holdout: continue   # The regular pass covered it.
        (spare, holdout) = (ShortSpare, ShortHoldout)
      if available < unknowns + spare + holdout: continue
      var rows: array[3, seq[seq[int64]]]
      for k in 0 .. 2:
        for i in start .. n - 2:
          var row = newSeq[int64](unknowns)
          for j, f in features: row[j] = all[k][i][f]
          row[features.len] = t.residues[k][i + 1]
          rows[k].add(row)
      let fit = fitLinearModel(rows, unknowns, @[features.len], spare, holdout, unique = short)
      if not fit.ok or not fit.rational: continue
      var used: seq[int]
      var coefficients: seq[Rational]
      var parts: seq[tuple[coefficient, symbol: string]]
      for j, f in features:
        let value = fit.values[j]
        if value.num == 0: continue
        let c = (-value.num, value.den)
        used.add(f)
        coefficients.add(c)
        parts.add((ratText(c), selfFeatureText(f, label)))
      if used.len == 0 or used.allIt(it in {0, 1, 2, 8}): continue
      result.ok = true
      result.form = TargetForm(kind: hSelfReferential, available: true, selfFeatures: used,
                               selfCoefficients: coefficients, selfStart: start)
      result.report = %*{"formula": label & "(n+1) = " & combination(parts),
        "valid_from_n": start, "evidence": evidence(fit)}
      if short:
        result.report["evidence"]["status"] = %"holds_with_few_spare_terms"
        result.report["evidence"]["unique_relation_of_this_shape"] = %true
        result.report["evidence"]["caution"] = %"Fitted with few spare terms; more terms would make it stronger."
      if t.offset != 0:
        result.report["index_note"] = %("n counts from the first supplied term: " & label & "(n) is the term at index n + " &
          $t.offset & ".")
      return

proc selfCoefficientsMod(form: TargetForm, q: int64): seq[int64] =
  for c in form.selfCoefficients:
    let den = liftMod(c.den, q)
    if den == 0 or gcd(den, q) != 1:
      raise newException(ValueError, "A coefficient denominator is not invertible modulo " & $q)
    result.add(mulMod(liftMod(c.num, q), modularInverse(den, q), q))

proc fastSelfModulus(q, relative: int64): bool =
  ## The online route divides by k! and by n, so q must be a prime past the index.
  relative <= MaxFastIndex and q > relative + 2 and q < FastEulerModulus and isProbablePrime(q)

proc selfValuesFast(form: TargetForm, root: Track, relative: int64, q: int64): seq[int64] =
  ## Terms 0 .. relative of a self-referential relation modulo a prime q > relative + 2 in
  ## O(n log^2 n). Each convolution feature is an online product h(t) = sum x(i) y(t-i) of
  ## sequences that grow with the terms:
  ##   sum C(n,k) a(k)        = n! (s * e)(n),   s(k) = a(k)/k!, e(k) = 1/k!
  ##   sum a(k) a(n-k)        = (a * a)(n)
  ##   sum C(n,k) a(k) a(n-k) = n! (s * s)(n)
  ##   Euler products         n E(n) = (c * E)(n), c(m) = sum over d | m of d a(d)
  ## One CDQ recursion over power-of-two blocks fills them all: a finished half [l, mid)
  ## meets x and y on [0, r - l) (or [0, mid) itself when l = 0), and the pairs that
  ## involve index 0 are added at the leaf, where the new term is computed.
  let count = int(relative) + 1
  let coefficients = selfCoefficientsMod(form, q)
  var need: set[0 .. SelfFeatureCount - 1]
  for f in form.selfFeatures: need.incl(f)
  var total = 1
  while total < count + 1: total = total shl 1
  var factorial = newSeq[int64](total + 2)
  factorial[0] = 1
  for i in 1 .. total + 1: factorial[i] = mulMod(factorial[i - 1], int64(i), q)
  var inverseFactorial = newSeq[int64](total + 2)
  inverseFactorial[total + 1] = modularInverse(factorial[total + 1], q)
  for i in countdown(total + 1, 1): inverseFactorial[i - 1] = mulMod(inverseFactorial[i], int64(i), q)
  proc inverseOf(n: int): int64 = mulMod(inverseFactorial[n], factorial[n - 1], q)
  # Series: 0 = a, 1 = s, 2 = 1/k!, 3 = c6, 4 = E6, 5 = c7 shifted (c7(k+1)), 6 = E7.
  var series: array[7, seq[int64]]
  for k in 0 .. 6: series[k] = newSeq[int64](total + 2)
  for i, value in root.exact:
    if i < total: series[0][i] = value.modSmall(q)
  for i in 0 .. total + 1: series[2][i] = inverseFactorial[i]
  series[4][0] = 1 mod q
  series[6][0] = 1 mod q
  type Engine = tuple[feature, x, y: int]
  var engines: seq[Engine]
  if 3 in need: engines.add((3, 1, 2))
  if 4 in need: engines.add((4, 0, 0))
  if 5 in need: engines.add((5, 1, 1))
  if 6 in need: engines.add((6, 3, 4))
  if 7 in need: engines.add((7, 5, 6))
  var partial = newSeq[seq[int64]](engines.len)
  for e in 0 ..< engines.len: partial[e] = newSeq[int64](total)
  let known = root.size
  proc complete(e: int, t: int): int64 =
    ## The pairs with index 0 on either side; the semi-online engine has only (t, 0) left,
    ## since its CDQ products already pair x[0] with every known y.
    let (_, x, y) = engines[e]
    result = (partial[e][t] + mulMod(series[x][t], series[y][0], q)) mod q
    if t > 0 and y != 2: result = (result + mulMod(series[x][0], series[y][t], q)) mod q
  proc leaf(t: int) =
    let a = series[0][t]
    series[1][t] = mulMod(a, inverseFactorial[t], q)
    if 6 in need and t >= 1:
      let weight = mulMod(int64(t) mod q, a, q)
      var m = t
      while m <= total + 1:
        series[3][m] = (series[3][m] + weight) mod q
        m += t
    if 7 in need:
      let k1 = t + 1
      let weight = mulMod(int64(k1) mod q, a, q)
      var m = k1
      while m - 1 <= total + 1:
        series[5][m - 1] = (series[5][m - 1] + weight) mod q
        m += k1
    var values = newSeq[int64](SelfFeatureCount)
    for e, engine in engines:
      let value = complete(e, t)
      case engine.feature
      of 3, 5: values[engine.feature] = mulMod(factorial[t], value, q)
      of 4: values[4] = value
      of 6:
        if t >= 1: series[4][t] = mulMod(value, inverseOf(t), q)
        values[6] = series[4][t]
      of 7:
        series[6][t + 1] = mulMod(value, inverseOf(t + 1), q)
        values[7] = series[6][t + 1]
      else: discard
    let previous = if t > 0: series[0][t - 1] else: 0'i64
    values[0] = a
    values[1] = mulMod(int64(t) mod q, a, q)
    values[2] = previous
    values[8] = 1 mod q
    values[9] = mulMod(a, a, q)
    values[10] = mulMod(a, previous, q)
    values[11] = mulMod(previous, previous, q)
    if t + 1 >= known and t + 1 < count:
      var next = 0'i64
      for j, f in form.selfFeatures: next = (next + mulMod(coefficients[j], values[f], q)) mod q
      series[0][t + 1] = next
  # Every block is one cyclic convolution of its length L: the products land below L
  # except the wrapped terms, which fall under L/2, so the outputs for [mid, high) are
  # exact. Prefixes series[k][0, L) are final once a block of length L starts at l >= L,
  # so their transforms are cached. The series 1/k! is known in full, so its engine is
  # semi-online: one product x[l, mid) * y[0, L) per block, also when l = 0.
  var prefixCache = initTable[(int, int), Spectrum]()
  proc prefix(k, length: int): Spectrum =
    if (k, length) notin prefixCache:
      prefixCache[(k, length)] = spectrumOf(series[k].toOpenArray(0, length - 1), length, q)
    prefixCache[(k, length)]
  proc contribute(e: int, low, middle, high: int) =
    let (_, x, y) = engines[e]
    let length = high - low
    if length <= 64:
      # Small blocks: the same pairs, summed directly.
      for t in middle ..< high:
        var sum = 0'i64
        if low == 0 and y != 2:
          for i in 1 ..< middle:
            if t - i >= 1 and t - i < middle: sum = (sum + mulMod(series[x][i], series[y][t - i], q)) mod q
        else:
          for i in low ..< middle:
            if t - i >= 1: sum = (sum + mulMod(series[x][i], series[y][t - i], q)) mod q
          if y != 2:
            for i in low ..< middle:
              if t - i >= 1: sum = (sum + mulMod(series[y][i], series[x][t - i], q)) mod q
        partial[e][t] = (partial[e][t] + sum) mod q
      return
    var total: Spectrum
    if y == 2:
      total.accumulate(spectrumOf(series[x].toOpenArray(low, middle - 1), length, q), prefix(y, length))
    elif low == 0:
      let half = spectrumOf(series[x].toOpenArray(0, middle - 1), length, q)
      if x == y: total.accumulate(half, half)
      else: total.accumulate(half, spectrumOf(series[y].toOpenArray(0, middle - 1), length, q))
    else:
      let fromX = spectrumOf(series[x].toOpenArray(low, middle - 1), length, q)
      if x == y:
        total.accumulate(fromX, prefix(y, length), 2)
      else:
        total.accumulate(fromX, prefix(y, length))
        total.accumulate(spectrumOf(series[y].toOpenArray(low, middle - 1), length, q), prefix(x, length))
    let values = residuesOf(total, length div 2, length, q)
    for i, value in values:
      let t = low + length div 2 + i
      partial[e][t] = (partial[e][t] + value) mod q
  proc solve(low, high: int) =
    if low >= count: return
    if high - low == 1:
      leaf(low)
      return
    let middle = (low + high) div 2
    solve(low, middle)
    if middle < count:
      for e in 0 ..< engines.len: contribute(e, low, middle, high)
    solve(middle, high)
  solve(0, total)
  result = series[0]
  result.setLen(count)

proc selfValuesMod(form: TargetForm, root: Track, relative: int64, q: int64): seq[int64] =
  ## Terms 0 .. relative of a self-referential relation modulo q.
  if relative > 4_096 and fastSelfModulus(q, relative): return selfValuesFast(form, root, relative, q)
  if relative > MaxSelfIndex:
    raise newException(ValueError, "A self-referential relation is iterated up to index " & $MaxSelfIndex &
      " (to " & $MaxFastIndex & " modulo a prime past the index and below 2^31)")
  let coefficients = selfCoefficientsMod(form, q)
  result = root.exact.mapIt(it.modSmall(q))
  discard selfRun(result, int(relative) + 1, form.selfFeatures, coefficients, q, collect = false)
  result.setLen(int(relative) + 1)

proc selfTermMod(form: TargetForm, root: Track, relative: int64, q: int64): int64 =
  if relative > 4_096 and fastSelfModulus(q, relative): return selfValuesFast(form, root, relative, q)[relative]
  if relative > MaxSelfIndex:
    raise newException(ValueError, "A self-referential relation is iterated up to index " & $MaxSelfIndex &
      " (to " & $MaxFastIndex & " modulo a prime past the index and below 2^31)")
  var coefficients: seq[int64]
  for c in form.selfCoefficients:
    let den = liftMod(c.den, q)
    if den == 0 or gcd(den, q) != 1:
      raise newException(ValueError, "A coefficient denominator is not invertible modulo " & $q)
    coefficients.add(mulMod(liftMod(c.num, q), modularInverse(den, q), q))
  var u = root.exact.mapIt(it.modSmall(q))
  discard selfRun(u, int(relative) + 1, form.selfFeatures, coefficients, q, collect = false)
  u[int(relative)]

proc multimodularExact(residueAt: proc(p: int64): int64, what: string): BigInt =
  ## An exact integer from its residues modulo primes below 2^31, joined one Garner step
  ## at a time. Primes are added until one more prime agrees with the value already
  ## rebuilt (the symmetric representative), so no size estimate is needed. A prime
  ## where the residue is undefined, such as a vanishing denominator, is skipped.
  var used = 0
  var joined = big(0)       # in [0, modulus)
  var modulus = big(1)
  var value = big(0)        # the symmetric representative
  var candidate = 2_147_483_647
  var skipped = 0
  var reason = ""
  while used < MaxSelfExactPrimes:
    while not isPrime(candidate): dec candidate
    let p = int64(candidate)
    dec candidate
    var r: int64
    try:
      r = residueAt(p)
    except ValueError as error:
      inc skipped
      if reason.len == 0: reason = error.msg
      # A few primes may divide a denominator; failing on most of them means no prime works.
      if skipped > 64 and skipped > 2 * used: raise newException(ValueError, what & ": " & reason)
      continue
    if used > 0 and value.modSmall(p) == r: return value
    # The running modulus is a product of earlier primes, so it is coprime to p.
    let step = mulMod(((r - joined.modSmall(p)) mod p + p) mod p, modularInverse(modulus.modSmall(p), p), p)
    joined = joined + modulus * big(step)
    modulus = modulus * big(p)
    inc used
    value = if modulus < joined + joined: joined - modulus else: joined
  raise newException(ValueError, "The exact value needs more than " & $MaxSelfExactPrimes & " primes. Pass a modulus.")

proc selfTermExact(form: TargetForm, root: Track, relative: int64): BigInt =
  ## The exact term from residues modulo primes below 2^31, joined by CRT.
  if relative > MaxSelfExactIndex:
    raise newException(ValueError, "Exact iteration of a self-referential relation is limited to index " &
      $MaxSelfExactIndex & ". Pass a modulus.")
  multimodularExact(proc(p: int64): int64 = selfTermMod(form, root, relative, p),
                    "the self-referential relation has no residue modulo the primes tried")

# Binary recursive relations: a(2n) and a(2n+1) are fixed combinations of a(n), a(n+1),
# the previous term, n and 1. Digit sums, Stern's sequence, the ruler function and the
# Josephus problem are of this kind. Indices here are the true indices m = offset + i.

const BinaryFeatures = 5   ## a(n), a(n+1), a(m-1), n, 1 for m = 2n + r.

proc binarySymbol(f, r: int, label: string): string =
  case f
  of 0: label & "(n)"
  of 1: label & "(n+1)"
  of 2: (if r == 0: label & "(2n-1)" else: label & "(2n)")
  of 3: "n"
  else: ""

proc binaryIndices(m, f: int): int =
  ## The index feature f reads for the term at m, or -1 for n and 1.
  case f
  of 0: m div 2
  of 1: m div 2 + 1
  of 2: m - 1
  else: -1

proc testBinaryRecursive(t: Track, label: string): Outcome =
  let n = t.size
  if n < 16 or t.offset < 0: return
  let subsets = @[@[0], @[0, 4], @[0, 3], @[0, 3, 4], @[4], @[3, 4], @[0, 1], @[0, 1, 4], @[0, 1, 3, 4],
                  @[2, 0], @[2, 0, 4], @[0, 1, 2, 3, 4]]
  var found: array[2, tuple[ok: bool, coefficients: seq[Rational], start: int, fit: Fit]]
  for r in 0 .. 1:
    block search:
      for subset in subsets:
        var valid: seq[int]
        for i in 0 ..< n:
          let m = t.offset + i
          if m mod 2 != r: continue
          var usable = true
          for f in subset:
            let j = binaryIndices(m, f)
            if f <= 2 and (j < t.offset or j >= m): usable = false
          if usable: valid.add(m)
        let unknowns = subset.len + 1
        for skip in 0 .. 2:
          if valid.len - skip < unknowns + 2 + Holdout: break
          var rows: array[3, seq[seq[int64]]]
          for k, p in ExplorerPrimes:
            for m in valid[skip .. ^1]:
              var row = newSeq[int64](unknowns)
              for j, f in subset:
                row[j] = case f
                  of 0, 1, 2: t.residues[k][binaryIndices(m, f) - t.offset]
                  of 3: liftMod(int64(m div 2), p)
                  else: 1'i64
              row[subset.len] = t.residues[k][m - t.offset]
              rows[k].add(row)
          let fit = fitLinearModel(rows, unknowns, @[subset.len])
          if not fit.ok or not fit.rational: continue
          var coefficients = newSeq[Rational](BinaryFeatures)
          for f in 0 ..< BinaryFeatures: coefficients[f] = (0'i64, 1'i64)
          for j, f in subset: coefficients[f] = (-fit.values[j].num, fit.values[j].den)
          found[r] = (true, coefficients, valid[skip], fit)
          break search
    if not found[r].ok: return
  var recursive = false
  for r in 0 .. 1:
    if found[r].coefficients[0].num != 0 or found[r].coefficients[1].num != 0: recursive = true
  if not recursive: return
  var texts: seq[string]
  for r in 0 .. 1:
    var parts: seq[tuple[coefficient, symbol: string]]
    for f in 0 ..< BinaryFeatures:
      let c = found[r].coefficients[f]
      if c.num != 0: parts.add((ratText(c), binarySymbol(f, r, label)))
    texts.add(label & (if r == 0: "(2n)" else: "(2n+1)") & " = " & combination(parts))
  result.ok = true
  result.form = TargetForm(kind: hBinaryRecursive, available: true,
    binaryCoefficients: [found[0].coefficients, found[1].coefficients],
    binaryStart: [found[0].start, found[1].start])
  result.report = %*{"formula": texts.join("; "),
    "valid_from_index": {"even": found[0].start, "odd": found[1].start},
    "evidence": {"status": "holds_on_unseen_terms",
                 "fit_equations": found[0].fit.fitRows + found[1].fit.fitRows,
                 "unseen_equations_checked": found[0].fit.checkedRows + found[1].fit.checkedRows,
                 "primes_checked": ExplorerPrimes.len, "coefficients": "rational"}}

proc binaryUsesPrevious(form: TargetForm): bool =
  form.binaryCoefficients[0][2].num != 0 or form.binaryCoefficients[1][2].num != 0

proc binaryTermMod(form: TargetForm, root: Track, index: int64, q: int64): int64 =
  var c: array[2, seq[int64]]
  for r in 0 .. 1:
    for value in form.binaryCoefficients[r]:
      let den = liftMod(value.den, q)
      if den == 0 or gcd(den, q) != 1:
        raise newException(ValueError, "A coefficient denominator is not invertible modulo " & $q)
      c[r].add(mulMod(liftMod(value.num, q), modularInverse(den, q), q))
  let offset = int64(root.offset)
  let known = int64(root.size)
  proc combine(m: int64, get: proc(j: int64): int64): int64 =
    let r = int(m mod 2)
    let half = m div 2
    let features = [get(half), (if c[r][1] != 0: get(half + 1) else: 0'i64),
                    (if c[r][2] != 0: get(m - 1) else: 0'i64), liftMod(half, q), 1'i64 mod q]
    for f in 0 ..< BinaryFeatures: result = (result + mulMod(c[r][f], features[f], q)) mod q
  if form.binaryUsesPrevious():
    if index > MaxBinarySequential:
      raise newException(ValueError, "This relation reads the previous term, so it runs up to index " & $MaxBinarySequential)
    var values = newSeq[int64](int(index - offset + 1))
    for i in 0 ..< int(min(known, index - offset + 1)): values[i] = root.exact[i].modSmall(q)
    for m in offset + known .. index:
      values[int(m - offset)] = combine(m, proc(j: int64): int64 = values[int(j - offset)])
    return values[int(index - offset)]
  var memo = initTable[int64, int64]()
  proc get(m: int64): int64 =
    if m - offset < known: return root.exact[int(m - offset)].modSmall(q)
    if m in memo: return memo[m]
    result = combine(m, get)
    memo[m] = result
  get(index)

proc binaryTermExact(form: TargetForm, root: Track, index: int64): BigInt =
  var ints: array[2, seq[BigInt]]
  var common: array[2, BigInt]
  for r in 0 .. 1:
    (common[r], ints[r]) = scaledForm(form.binaryCoefficients[r])
  let offset = int64(root.offset)
  let known = int64(root.size)
  proc combine(m: int64, get: proc(j: int64): BigInt): BigInt =
    let r = int(m mod 2)
    let half = m div 2
    var total = big(0)
    if not ints[r][0].isZero: total = total + ints[r][0] * get(half)
    if not ints[r][1].isZero: total = total + ints[r][1] * get(half + 1)
    if not ints[r][2].isZero: total = total + ints[r][2] * get(m - 1)
    total = total + ints[r][3] * big(half) + ints[r][4]
    exactQuotient(total, common[r])
  if form.binaryUsesPrevious():
    if index > MaxExactIndex:
      raise newException(ValueError, "Exact iteration is limited to " & $MaxExactIndex & " steps. Pass a modulus.")
    var values = newSeq[BigInt](int(index - offset + 1))
    for i in 0 ..< int(min(known, index - offset + 1)): values[i] = root.exact[i]
    for m in offset + known .. index:
      values[int(m - offset)] = combine(m, proc(j: int64): BigInt = values[int(j - offset)])
    return values[int(index - offset)]
  var memo = initTable[int64, BigInt]()
  proc get(m: int64): BigInt =
    if m - offset < known: return root.exact[int(m - offset)]
    if m in memo: return memo[m]
    result = combine(m, get)
    memo[m] = result
  get(index)

proc testHypothesis(t: Track, h: Hypothesis, moduli: seq[int64], label: string): Outcome =
  case h
  of hPolynomial: testPolynomial(t, label)
  of hLinearRecurrence: testLinear(t, label)
  of hHolonomic: testHolonomic(t, label)
  of hAlgebraic: testAlgebraic(t, label)
  of hMultiplicative: testMultiplicative(t, label)
  of hPeriodic: testPeriodic(t, moduli, label)
  of hSelfReferential: testSelfReferential(t, label)
  of hBinaryRecursive: testBinaryRecursive(t, label)

# Search state shared by the default order and the learned policy.

proc pathKey(path: openArray[Transform]): string =
  for i, t in path:
    if i > 0: result.add(',')
    result.add($ord(t))

proc pathExpression*(path: seq[Transform]): string =
  result = "a"
  for t in path: result = $t & "(" & result & ")"

proc redundant(path: seq[Transform]): bool =
  const inverse = [(tDifference, tPartialSum), (tPartialSum, tDifference), (tBinomial, tInverseBinomial),
                   (tInverseBinomial, tBinomial), (tDivideFactorial, tMultiplyFactorial),
                   (tMultiplyFactorial, tDivideFactorial)]
  for i in 1 ..< path.len:
    if (path[i - 1], path[i]) in inverse: return true

proc newExplorer*(root: Track, moduli: seq[int64] = @DefaultModuli): Explorer =
  result.nodes[""] = root
  result.moduli = moduli
  result.formulaDepth = high(int)

proc node*(ex: var Explorer, path: seq[Transform]): Track =
  let key = pathKey(path)
  if key in ex.nodes: return ex.nodes[key]
  let parent = ex.node(path[0 ..< path.high])
  var child = if parent.valid: transformed(parent, path[^1]) else: Track()
  if child.valid and child.size < MinExplorerTerms: child.valid = false
  ex.nodes[key] = child
  child

proc testAt(ex: var Explorer, path: seq[Transform], h: Hypothesis, accept: DirectionFilter = nil) =
  let key = pathKey(path)
  ex.lastDescend = false
  inc ex.tests
  ex.tested.mgetOrPut(key, default(set[Hypothesis])).incl(h)
  let track = ex.node(path)
  if not track.valid: return
  let outcome = testHypothesis(track, h, ex.moduli, (if path.len == 0: "a" else: "b"))
  if not outcome.ok: return
  let direction = Direction(path: path, hypothesis: h, report: outcome.report, form: outcome.form)
  if not accept.isNil and not accept(direction): return
  ex.directions.add(direction)
  if h in FormulaHypotheses:
    ex.tested[key] = ex.tested[key] + FormulaHypotheses
    ex.formulaDepth = min(ex.formulaDepth, path.len)

proc testHolonomicFixed(t: Track, r, d: int, label: string): Outcome =
  ## The holonomic relation of exactly order r and degree d, fitted and verified like the
  ## scanned one. A program that names the shape gets this instead of the scan.
  let width = d + 1
  let unknowns = (r + 1) * width
  if t.size - r < unknowns + 2 + Holdout: return
  let fit = fitLinearModel(holonomicRows(t, r, d), unknowns, holonomicNormalize(r, d))
  if not fit.ok or blockZero(fit.residues, 0, width) or blockZero(fit.residues, r, width): return
  if degenerateLead(t, fit.residues, r, d): return
  return holonomicOutcome(fit, r, d, label)

proc testAtFixed*(ex: var Explorer, h: Hypothesis, r, d: int): bool =
  ## testAt on the explorer's current path, with the holonomic shape named by the caller.
  ## The name is a hint and not a filter: when it does not fit, the same shape scan the
  ## plain order runs covers the rest, so a wrong name never costs a direction. Every
  ## other hypothesis ignores the shape. True when the named shape itself produced it.
  let path = ex.path
  let key = pathKey(path)
  ex.lastDescend = false
  inc ex.tests
  ex.tested.mgetOrPut(key, default(set[Hypothesis])).incl(h)
  let track = ex.node(path)
  if not track.valid: return false
  let label = if path.len == 0: "a" else: "b"
  var outcome =
    if h == hHolonomic: testHolonomicFixed(track, r, d, label)
    else: testHypothesis(track, h, ex.moduli, label)
  if outcome.ok:
    result = h == hHolonomic
  elif h == hHolonomic:
    outcome = testHolonomic(track, label)
  if not outcome.ok: return
  ex.directions.add(Direction(path: path, hypothesis: h, report: outcome.report, form: outcome.form))
  if h in FormulaHypotheses:
    ex.tested[key] = ex.tested[key] + FormulaHypotheses
    ex.formulaDepth = min(ex.formulaDepth, path.len)

proc pendingWork*(ex: var Explorer): array[TransformSlots, int] =
  ## What one step in each transform still reaches: the number of untested hypotheses at
  ## that child, or -1 when the step cannot be made. It lets a policy tell a spent branch
  ## from one with work left, without probing the branch itself.
  let depth = ex.path.len
  let hypotheses = if depth == 0: SearchHypotheses else: FormulaHypotheses
  let deep = depth < MaxTransformDepth and ex.formulaDepth > depth
  for t in Transform:
    let child = ex.path & t
    if not deep or redundant(child) or not ex.node(child).valid:
      result[ord(t)] = -1
      continue
    var count = 0
    let tested = ex.tested.getOrDefault(pathKey(child))
    for h in hypotheses:
      if h notin tested: inc count
    result[ord(t)] = count

proc depth*(ex: Explorer): int {.inline.} = ex.path.len

proc legalActions*(ex: var Explorer): set[ExplorerAction] =
  let depth = ex.path.len
  let tested = ex.tested.getOrDefault(pathKey(ex.path))
  for h in SearchHypotheses:
    if h notin tested and (depth == 0 or h in FormulaHypotheses): result.incl(ExplorerAction(ord(h)))
  if depth < MaxTransformDepth and ex.formulaDepth > depth:
    for t in Transform:
      let child = ex.path & t
      if redundant(child) or not ex.node(child).valid: continue
      if FormulaHypotheses <= ex.tested.getOrDefault(pathKey(child)):
        # A spent child can still hold an unspent grandchild, so descend through it.
        if depth + 1 >= MaxTransformDepth: continue
        var live = false
        for next in Transform:
          let grand = child & next
          if redundant(grand) or not ex.node(grand).valid: continue
          if FormulaHypotheses <= ex.tested.getOrDefault(pathKey(grand)): continue
          live = true
          break
        if not live: continue
      result.incl(ExplorerAction(LastSearchHypothesis + 1 + ord(t)))
  # A step back right after a step in walks the same pair of nodes forever, so the
  # engine refuses it: the walker must test where it stands, or go on.
  if depth > 0 and not ex.lastDescend: result.incl(ExplorerAction(PopAction))

proc hypothesisAction*(h: Hypothesis): ExplorerAction = ExplorerAction(ord(h))
proc transformAction*(t: Transform): ExplorerAction = ExplorerAction(LastSearchHypothesis + 1 + ord(t))

proc act*(ex: var Explorer, action: ExplorerAction) =
  inc ex.steps
  let index = int(action)
  if index <= LastSearchHypothesis:
    ex.testAt(ex.path, Hypothesis(index))
  elif index < PopAction:
    ex.path.add(Transform(index - LastSearchHypothesis - 1))
    ex.lastDescend = true
  elif ex.path.len > 0:
    ex.path.setLen(ex.path.len - 1)
    ex.lastDescend = false

proc pathsAtDepth(depth: int): seq[seq[Transform]] =
  result = @[newSeq[Transform]()]
  for _ in 0 ..< depth:
    var next: seq[seq[Transform]]
    for path in result:
      for t in Transform:
        let child = path & t
        if not redundant(child): next.add(child)
    result = next

proc strongCount*(ex: Explorer): int =
  ex.directions.countIt(it.hypothesis in StrongHypotheses)

proc firstStrong*(ex: Explorer): int =
  for i, d in ex.directions:
    if d.hypothesis in StrongHypotheses: return i
  -1

proc evaluable(form: TargetForm): bool =
  form.available or form.modular or form.multiplicativeModel != mmNone

proc firstEvaluable*(ex: Explorer): int =
  ## The first strong direction that can evaluate a far term, else the first strong one.
  for i, d in ex.directions:
    if d.hypothesis in StrongHypotheses and d.form.evaluable: return i
  ex.firstStrong()

proc runDefault*(ex: var Explorer, maxDirections, budget: int, hint: seq[Transform] = @[],
                 accept: DirectionFilter = nil) =
  ## Breadth first over transform depth. A formula at depth d stops deeper search.
  ## Paths that extend a prefix of hint go first within their depth.
  for depth in 0 .. MaxTransformDepth:
    if ex.formulaDepth < depth: return
    var paths = pathsAtDepth(depth)
    if hint.len > 0:
      let wanted = hint[0 ..< min(depth, hint.len)]
      var first, rest: seq[seq[Transform]]
      for path in paths:
        if path.len == wanted.len and path == wanted: first.add(path) else: rest.add(path)
      paths = first & rest
    for path in paths:
      if not ex.node(path).valid: continue
      for h in SearchHypotheses:
        if depth > 0 and h notin FormulaHypotheses: continue
        if ex.strongCount >= maxDirections or ex.tests >= budget: return
        if h in ex.tested.getOrDefault(pathKey(path)): continue
        ex.testAt(path, h, accept)

proc lastResort(ex: var Explorer) =
  ## The engine's own nets, for when nothing strong and evaluable was found. This is not
  ## an ordering scan: it checks four specific forms that the transform search cannot reach.
  if ex.directions.anyIt(it.hypothesis in StrongHypotheses and it.form.evaluable): return
  # A multiplicative direction that found no way to its far terms counts here too. It stays
  # out of the tested sets, which feed the policy. The short-data fits come last, since they
  # rest on fewer spare equations.
  let root = ex.node(@[])
  if root.valid:
    for stage in 0 .. 3:
      inc ex.tests
      let outcome = case stage
        of 0: testBinaryRecursive(root, "a")
        of 1: testSelfReferential(root, "a")
        of 2: testSelfReferential(root, "a", short = true)
        else: testHolonomicShort(root, "a")
      if outcome.ok:
        ex.directions.add(Direction(path: @[], hypothesis: outcome.form.kind, report: outcome.report,
                                    form: outcome.form))
        return
    # A transform often lowers the shape of the relation, as the ratio of a hypergeometric
    # product does, but leaves one term fewer. The short-data holonomic fit also runs on
    # one-step transforms whose terms stay exact integers.
    for t in [tRatio, tDifference, tPartialSum, tDivideFactorial, tMultiplyFactorial]:
      let track = ex.node(@[t])
      if not track.valid or track.exact.len != track.size: continue
      inc ex.tests
      let outcome = testHolonomicShort(track, "b")
      if outcome.ok:
        ex.directions.add(Direction(path: @[t], hypothesis: hHolonomic, report: outcome.report,
                                    form: outcome.form))
        return

proc explorePlain*(ex: var Explorer, maxDirections, budget: int, guide: ExplorerGuide) =
  ## The ordering policy and the exhaustive scan. The planner probes its own routes with
  ## this; a learned program policy does not use it.
  if not guide.isNil:
    while ex.steps < GuideSteps and ex.strongCount < maxDirections and ex.tests < budget:
      let legal = ex.legalActions()
      if legal == {}: break
      let action = guide(ex, legal)
      if action notin legal: break
      ex.act(action)
  ex.runDefault(maxDirections, budget)
  lastResort(ex)

proc explore*(ex: var Explorer, maxDirections, budget: int, guide: ExplorerGuide) =
  ## With a program policy installed the policy drives every step and the plain scan is
  ## not run at all; the engine still checks each tool call as it comes.
  if not programPrologue.isNil:
    programPrologue(ex, maxDirections, budget)
    lastResort(ex)
  else:
    explorePlain(ex, maxDirections, budget, guide)

proc teacherRoute*(root: Track, budget: int, hint: seq[Transform] = @[]): seq[ExplorerAction] =
  ## Shortest action sequence to a strong direction at the lowest transform depth.
  ## A planted hint only reorders paths within a depth, so the depth stays minimal.
  var probe = newExplorer(root)
  probe.runDefault(1, budget, hint)
  let index = probe.firstStrong()
  if index < 0: return
  let found = probe.directions[index]
  for t in found.path: result.add(transformAction(t))
  result.add(hypothesisAction(found.hypothesis))

proc polynomialDegreeHint(s: seq[int64], p: int64): int =
  var current = s
  for d in 0 .. MaxPolynomialFitDegree:
    if current.len < 3: break
    var next = newSeq[int64](current.len - 1)
    for i in 0 ..< next.len: next[i] = subMod(current[i + 1], current[i], p)
    current = next
    if current.allIt(it == 0): return d
  -1

proc explorerFrame*(ex: var Explorer, legal: set[ExplorerAction]): ExplorerFrame =
  ## Features derived from exact residues and approximate magnitudes, never raw residues.
  proc unit(x: float64): float32 =
    if x.classify in {fcNan}: 0'f32 else: float32(max(-1.0, min(1.0, x)))
  let track = ex.node(ex.path)
  let n = track.size
  var logs = newSeq[float64](n)
  var signChanges = 0
  for i, value in track.approx:
    logs[i] = case value.classify
      of fcNan: 0.0
      of fcInf, fcNegInf: 308.0
      else: log10(1.0 + abs(value))
    if i > 0 and value * track.approx[i - 1] < 0: inc signChanges
  for i in 0 ..< min(n, 24):
    let value = track.approx[i]
    result[i] = if value.classify == fcNan or value == 0: 0'f32 elif value > 0: 1'f32 else: -1'f32
    result[24 + i] = unit(logs[i] / 40.0)
  for i in 0 ..< min(n - 2, 22): result[48 + i] = unit(logs[i + 2] - 2 * logs[i + 1] + logs[i])
  result[70] = unit(float64(n) / 64.0)
  result[71] = unit(float64(berlekampMassey(track.residues[0], ExplorerPrimes[0]).len) / float64(n))
  let degree = polynomialDegreeHint(track.residues[0], ExplorerPrimes[0])
  result[72] = if degree < 0: 1'f32 else: unit(float64(degree + 1) / float64(n))
  result[73] = unit(float64(track.residues[0].countIt(it == 0)) / float64(n))
  result[74] = unit(float64(track.offset) / 4.0)
  result[75 + ex.path.len] = 1
  if ex.path.len > 0:
    result[78 + ord(ex.path[^1])] = 1
    result[87 + ord(ex.path[0])] = 1
  for h in ex.tested.getOrDefault(pathKey(ex.path)): result[96 + ord(h)] = 1
  for action in legal: result[102 + int(action)] = 1
  result[118] = unit(float64(ex.steps) / 64.0)
  result[119] = unit(float64(ex.directions.len) / 4.0)
  result[120] = float32(ord(ex.formulaDepth < high(int)))
  if n > 1: result[121] = unit((logs[^1] - logs[0]) / float64(n - 1) / 10.0)
  result[122] = float32(ord(track.residues[0].allIt(it == track.residues[0][0])))
  result[123] = unit(float64(signChanges) / float64(max(n - 1, 1)))

# Far terms from a root direction with rational coefficients.

proc ratMod(r: Rational, q: int64): int64 =
  mulMod(liftMod(r.num, q), modularInverse(liftMod(r.den, q), q), q)

proc linearTermMod(coefficients, initial: seq[int64], steps: BigInt, q: int64): int64 =
  let order = coefficients.len
  proc reduce(poly: var seq[int64]) =
    for k in countdown(poly.high, order):
      let top = poly[k]
      if top == 0: continue
      for j in 1 .. order: poly[k - j] = (poly[k - j] + mulMod(top, coefficients[j - 1], q)) mod q
      poly[k] = 0
    if poly.len > order: poly.setLen(order)
  proc multiply(a, b: seq[int64]): seq[int64] =
    result = newSeq[int64](max(0, a.len + b.len - 1))
    for i, x in a:
      if x == 0: continue
      for j, y in b: result[i + j] = (result[i + j] + mulMod(x, y, q)) mod q
    reduce(result)
  var bits: seq[int64]
  var rest = steps
  while not rest.isZero:
    let (quotient, remainder) = divSmall(rest, 2)
    bits.add(remainder)
    rest = quotient
  var weights = @[1'i64 mod q]
  reduce(weights)
  var base = @[0'i64, 1'i64 mod q]
  reduce(base)
  for bit in bits:
    if bit == 1: weights = multiply(weights, base)
    base = multiply(base, base)
  for j, w in weights: result = (result + mulMod(w, initial[j], q)) mod q

proc linearBlocks(form: TargetForm): seq[seq[Rational]] =
  let order = form.linear.len
  result = newSeq[seq[Rational]](order + 1)
  result[order] = @[(1'i64, 1'i64)]
  for j, c in form.linear: result[order - 1 - j] = @[(-c.num, c.den)]

proc holonomicResiduesMod(residues: seq[seq[int64]], root: Track, steps: int64, q: int64): int64 =
  let r = residues.high
  let n = root.size
  var window = newSeq[int64](r)
  for k in 0 ..< r: window[k] = root.exact[n - r + k].modSmall(q)
  proc evalAt(coefficients: seq[int64], x: int64): int64 =
    let point = liftMod(x, q)
    for j in countdown(coefficients.high, 0): result = (mulMod(result, point, q) + coefficients[j]) mod q
  for i in int64(n) .. steps:
    let nAbs = int64(root.offset) + i - r
    var total = 0'i64
    for k in 0 ..< r: total = (total + mulMod(evalAt(residues[k], nAbs), window[k], q)) mod q
    let lead = evalAt(residues[r], nAbs)
    if lead == 0: raise newException(ValueError, "The leading coefficient vanishes modulo " & $q & " at n = " & $nAbs)
    let next = mulMod((q - total) mod q, modularInverse(lead, q), q)
    for k in 0 ..< r - 1: window[k] = window[k + 1]
    window[r - 1] = next
  window[r - 1]

proc holonomicValuesMod(blocks: seq[seq[Rational]], root: Track, steps: int64, q: int64): seq[int64] =
  ## Terms 0 .. steps (relative) of a holonomic relation modulo q, iterated.
  let residues = blocks.mapIt(it.mapIt(ratMod(it, q)))
  let r = residues.high
  result = root.exact.mapIt(it.modSmall(q))
  proc evalAt(coefficients: seq[int64], x: int64): int64 =
    let point = liftMod(x, q)
    for j in countdown(coefficients.high, 0): result = (mulMod(result, point, q) + coefficients[j]) mod q
  for i in int64(result.len) .. steps:
    let nAbs = int64(root.offset) + i - r
    var total = 0'i64
    for k in 0 ..< r: total = (total + mulMod(evalAt(residues[k], nAbs), result[int(i) - r + k], q)) mod q
    let lead = evalAt(residues[r], nAbs)
    if lead == 0: raise newException(ValueError, "The leading coefficient vanishes modulo " & $q & " at n = " & $nAbs)
    result.add(mulMod((q - total) mod q, modularInverse(lead, q), q))
  result.setLen(int(steps) + 1)

proc holonomicTermMod(blocks: seq[seq[Rational]], root: Track, steps: int64, q: int64): int64 =
  holonomicResiduesMod(blocks.mapIt(it.mapIt(ratMod(it, q))), root, steps, q)

# Holonomic terms modulo a prime in O(sqrt(N) log N): the baby-step giant-step product of
# polynomial matrices (Bostan, Gaudry and Schost). With P_d(x) = M(x+d-1)...M(x), the values
# P_d(x0 + i v) for i = 0 .. D d determine a polynomial of degree D d in i. Doubling d takes
# two Lagrange shifts of those samples (each one NTT convolution) and a pointwise product,
# P_2d(x) = P_d(x + d) P_d(x). At d = v the giant steps P_v(x0 + i v) cover the range.

const
  SqrtProductFrom = 1_000_000'i64        ## Shorter ranges iterate term by term.
  SqrtProductMinModulus = 1_048_576'i64  ## The shifts need q well above D v.
  MaxSqrtProductSamples = 2_000_000      ## D v, bounded by the NTT length (2^23).

proc batchInverse(values: seq[int64], q: int64): seq[int64] =
  ## Inverses of nonzero residues with one modular inversion.
  result = newSeq[int64](values.len)
  var running = 1'i64
  for i, x in values:
    if x == 0: raise newException(ValueError, "A sample shift meets a zero modulo " & $q)
    result[i] = running
    running = mulMod(running, x, q)
  var inverse = modularInverse(running, q)
  for i in countdown(values.high, 0):
    let x = values[i]
    result[i] = mulMod(result[i], inverse, q)
    inverse = mulMod(inverse, x, q)

proc lagrangeWeightsMod(k: int, q: int64): seq[int64] =
  ## (-1)^(k - j) / (j! (k - j)!) modulo a prime q > k, for j = 0 .. k.
  var factorial = newSeq[int64](k + 1)
  factorial[0] = 1
  for i in 1 .. k: factorial[i] = mulMod(factorial[i - 1], int64(i), q)
  var inverseFactorial = newSeq[int64](k + 1)
  inverseFactorial[k] = modularInverse(factorial[k], q)
  for i in countdown(k, 1): inverseFactorial[i - 1] = mulMod(inverseFactorial[i], int64(i), q)
  result = newSeq[int64](k + 1)
  for j in 0 .. k:
    result[j] = mulMod(inverseFactorial[j], inverseFactorial[k - j], q)
    if (k - j) mod 2 == 1: result[j] = (q - result[j]) mod q

type ShiftPlan = object
  ## From f(0 .. k) of a polynomial of degree <= k, the values f(a), ..., f(a + count - 1)
  ## modulo a prime q > k. Lagrange interpolation turns into one convolution:
  ## f(a + t) = prod_{i = 0 .. k} (a + t - i) * sum_j g(j) / (a + t - j), g(j) = f(j) w(j).
  ## The transformed 1 / (a - k + m) and the factors are shared by every column.
  k, count: int
  spectrum: Spectrum
  factors: seq[int64]

proc shiftPlan(a: int64, k, count, size: int, q: int64): ShiftPlan =
  ## size >= k + count keeps the wrapped part of the cyclic product below index k.
  var points = newSeq[int64](k + count)
  for m in 0 ..< k + count: points[m] = liftMod(a - k + m, q)
  let h = batchInverse(points, q)
  result.k = k
  result.count = count
  result.spectrum = spectrumOf(h, size, q)
  result.factors = newSeq[int64](count)
  var product = 1'i64
  for i in 0 .. k: product = mulMod(product, liftMod(a - i, q), q)
  for t in 0 ..< count:
    result.factors[t] = product
    product = mulMod(mulMod(product, liftMod(a + t + 1, q), q), h[t], q)

proc applyShift(plan: ShiftPlan, weighted: Spectrum, q: int64): seq[int64] =
  var product: Spectrum
  product.accumulate(weighted, plan.spectrum)
  result = product.residuesOf(plan.k, plan.k + plan.count, q)
  for t in 0 ..< plan.count: result[t] = mulMod(plan.factors[t], result[t], q)

proc sqrtMatrixProduct(entries: seq[seq[seq[int64]]], x0, total: int64, q: int64): seq[int64] =
  ## prod_{x = x0 .. x0+total-1} M(x), later factors on the left, modulo a prime q, for a
  ## square matrix of polynomials (residue coefficients, ascending). Row-major result.
  let r = entries.len
  let width = r * r
  var degree = 1
  for row in entries:
    for poly in row: degree = max(degree, poly.high)
  proc evalAt(coefficients: seq[int64], x: int64): int64 =
    let point = liftMod(x, q)
    for j in countdown(coefficients.high, 0): result = (mulMod(result, point, q) + coefficients[j]) mod q
  proc stepAt(x: int64): seq[int64] =
    result = newSeq[int64](width)
    for i in 0 ..< r:
      for j in 0 ..< r:
        if entries[i][j].len > 0: result[i * r + j] = evalAt(entries[i][j], x)
  proc multiply(later, earlier: seq[int64]): seq[int64] =
    result = newSeq[int64](width)
    for i in 0 ..< r:
      for j in 0 ..< r:
        var sum = 0'i64
        for k in 0 ..< r: sum = (sum + mulMod(later[i * r + k], earlier[k * r + j], q)) mod q
        result[i * r + j] = sum
  var product = newSeq[int64](width)
  for i in 0 ..< r: product[i * r + i] = 1 mod q
  if total <= 0: return product
  if total < 4096:
    for x in x0 ..< x0 + total: product = multiply(stepAt(x), product)
    return product
  var v = 1'i64
  while int64(degree) * v * v < total: v *= 2
  if int64(degree) * v > MaxSqrtProductSamples:
    raise newException(ValueError, "The range is too long for the square root product")
  if q < SqrtProductMinModulus or q >= FastEulerModulus or q <= 4 * int64(degree) * v:
    raise newException(ValueError, "The square root product needs a prime modulus from 2^20 to 2^31")
  let inverseV = modularInverse(v mod q, q)
  var samples: seq[seq[int64]]                # samples[i] = P_d(x0 + i v)
  for i in 0 .. degree: samples.add(stepAt(x0 + int64(i) * v))
  var d = 1'i64
  while d < v:
    let k = samples.len - 1                   # = degree * d
    let shift = mulMod(d mod q, inverseV, q)  # x + d is i + d / v in sample units
    var extended = newSeq[seq[int64]](2 * k + 1)
    var moved = newSeq[seq[int64]](2 * k + 1)
    for i in 0 .. 2 * k:
      extended[i] = newSeq[int64](width)
      moved[i] = newSeq[int64](width)
    # One weight list and two shift plans serve every column of this level; a column costs
    # one forward transform of its weighted samples and two products.
    var size = 1
    while size < 3 * k + 1: size = size shl 1
    let weights = lagrangeWeightsMod(k, q)
    let upperPlan = shiftPlan(int64(k + 1), k, k, size, q)
    let movePlan = shiftPlan(shift, k, 2 * k + 1, size, q)
    for c in 0 ..< width:
      var column = newSeq[int64](k + 1)
      for i in 0 .. k: column[i] = samples[i][c]
      if column.allIt(it == 0): continue
      var weighted = newSeq[int64](k + 1)
      for j in 0 .. k: weighted[j] = mulMod(column[j], weights[j], q)
      let spectrum = spectrumOf(weighted, size, q)
      let upper = upperPlan.applyShift(spectrum, q)
      let shifted = movePlan.applyShift(spectrum, q)
      for i in 0 .. k: extended[i][c] = column[i]
      for i in 0 ..< k: extended[k + 1 + i][c] = upper[i]
      for i in 0 .. 2 * k: moved[i][c] = shifted[i]
    samples = newSeq[seq[int64]](2 * k + 1)
    for i in 0 .. 2 * k: samples[i] = multiply(moved[i], extended[i])
    d *= 2
  let giant = total div v
  for i in 0 ..< int(giant): product = multiply(samples[i], product)
  for x in x0 + giant * v ..< x0 + total: product = multiply(stepAt(x), product)
  product

proc holonomicSqrtMod(blocks: seq[seq[Rational]], root: Track, steps: int64, q: int64): int64 =
  ## Term `steps` (relative) of the holonomic relation modulo a prime q < 2^31. The step
  ## matrix M(x) (scaled by the leading coefficient L(x)) and L(x) itself form one block
  ## diagonal matrix, so a single product carries both.
  let residues = blocks.mapIt(it.mapIt(ratMod(it, q)))
  let r = residues.high
  let n = root.size
  let total = steps - int64(n) + 1
  if total <= 0: return root.exact[int(steps)].modSmall(q)
  let x0 = int64(root.offset + n - r)
  var entries = newSeq[seq[seq[int64]]](r + 1)
  for i in 0 .. r: entries[i] = newSeq[seq[int64]](r + 1)
  for k in 0 ..< r - 1: entries[k][k + 1] = residues[r]
  for k in 0 ..< r: entries[r - 1][k] = residues[k].mapIt((q - it) mod q)
  entries[r][r] = residues[r]
  let product = sqrtMatrixProduct(entries, x0, total, q)
  let width = r + 1
  let lead = product[r * width + r]
  if lead == 0:
    raise newException(ValueError, "The leading coefficient vanishes modulo " & $q & " inside the range")
  var value = 0'i64
  for k in 0 ..< r:
    value = (value + mulMod(product[(r - 1) * width + k], root.exact[n - r + k].modSmall(q), q)) mod q
  mulMod(value, modularInverse(lead, q), q)

proc holonomicTermExact(blocks: seq[seq[Rational]], root: Track, steps: int64): BigInt =
  let r = blocks.high
  let n = root.size
  var flat: seq[Rational]
  for b in blocks: flat.add(b)
  let (ok, ints) = integerForm(flat)
  if not ok: raise newException(ValueError, "Coefficient denominators are too large for exact iteration")
  var polys: seq[seq[BigInt]]
  var position = 0
  for b in blocks:
    polys.add(ints[position ..< position + b.len])
    position += b.len
  proc evalAt(coefficients: seq[BigInt], x: int64): BigInt =
    result = big(0)
    for j in countdown(coefficients.high, 0): result = result * big(x) + coefficients[j]
  var window = root.exact[n - r ..< n]
  if steps - int64(n) < SequentialExactSteps:
    for i in int64(n) .. steps:
      let nAbs = int64(root.offset) + i - r
      var total = big(0)
      for k in 0 ..< r: total = total + evalAt(polys[k], nAbs) * window[k]
      let lead = evalAt(polys[r], nAbs)
      if lead.isZero: raise newException(ValueError, "The leading coefficient vanishes at n = " & $nAbs)
      let next = exactQuotient(-total, lead)
      for k in 0 ..< r - 1: window[k] = window[k + 1]
      window[r - 1] = next
    return window[r - 1]
  # Binary splitting. One step maps the window w = (a(i-r), ..., a(i-1)) to M(i) w / L(i),
  # where L is the leading polynomial and M shifts the window (scaled by L) and puts
  # -sum P_k(i) w_k last. The product of the M(i) and of the L(i) over all steps is built
  # as a balanced tree, so the big products are Karatsuba sized, and one exact division
  # at the end gives the term.
  type Span = tuple[m: seq[seq[BigInt]], d: BigInt]
  proc leaf(i: int64): Span =
    let nAbs = int64(root.offset) + i - r
    let lead = evalAt(polys[r], nAbs)
    if lead.isZero: raise newException(ValueError, "The leading coefficient vanishes at n = " & $nAbs)
    result.m = newSeq[seq[BigInt]](r)
    for k in 0 ..< r: result.m[k] = newSeq[BigInt](r)
    for k in 0 ..< r - 1: result.m[k][k + 1] = lead
    for k in 0 ..< r: result.m[r - 1][k] = -evalAt(polys[k], nAbs)
    result.d = lead
  proc combine(later, earlier: Span): Span =
    result.m = newSeq[seq[BigInt]](r)
    for i in 0 ..< r:
      result.m[i] = newSeq[BigInt](r)
      for j in 0 ..< r:
        var total = big(0)
        for k in 0 ..< r:
          if not later.m[i][k].isZero and not earlier.m[k][j].isZero:
            total = total + later.m[i][k] * earlier.m[k][j]
        result.m[i][j] = total
    result.d = later.d * earlier.d
  proc span(first, last: int64): Span =
    if first == last: return leaf(first)
    let middle = first + (last - first) div 2
    combine(span(middle + 1, last), span(first, middle))
  let total = span(int64(n), steps)
  var value = big(0)
  for k in 0 ..< r: value = value + total.m[r - 1][k] * window[k]
  exactQuotientBig(value, total.d)

proc modularFitBlocks(form: TargetForm, q: int64): seq[seq[int64]] =
  let k = ExplorerPrimes.find(q)
  if k < 0:
    raise newException(ValueError, "The relation is known modulo " & $ExplorerPrimes[0] & ", " &
      $ExplorerPrimes[1] & " and " & $ExplorerPrimes[2] & " only. Pass one of them as modulus.")
  form.modularBlocks[k]

proc extendForm(form: TargetForm, known: seq[int64], offset, count: int, q: int64): seq[int64] =
  ## Values for indices offset ..< offset + count, keeping the known prefix.
  result = known[0 ..< min(known.len, count)]
  case form.kind
  of hPolynomial:
    let coefficients = form.polynomial.mapIt(ratMod(it, q))
    for i in result.len ..< count:
      let x = liftMod(int64(offset + i), q)
      var value = 0'i64
      for j in countdown(coefficients.high, 0): value = (mulMod(value, x, q) + coefficients[j]) mod q
      result.add(value)
  of hLinearRecurrence:
    let coefficients = form.linear.mapIt(ratMod(it, q))
    for i in result.len ..< count:
      var value = 0'i64
      for j, c in coefficients: value = (value + mulMod(c, result[i - 1 - j], q)) mod q
      result.add(value)
  of hSelfReferential, hBinaryRecursive:
    raise newException(ValueError, "Self-referential and binary relations are found on the original terms only")
  else:
    let residues = if form.modular: modularFitBlocks(form, q) else: form.blocks.mapIt(it.mapIt(ratMod(it, q)))
    let r = residues.high
    proc evalAt(coefficients: seq[int64], x: int64): int64 =
      let point = liftMod(x, q)
      for j in countdown(coefficients.high, 0): result = (mulMod(result, point, q) + coefficients[j]) mod q
    for i in result.len ..< count:
      let nAbs = int64(offset + i - r)
      var total = 0'i64
      for k in 0 ..< r: total = (total + mulMod(evalAt(residues[k], nAbs), result[i - r + k], q)) mod q
      let lead = evalAt(residues[r], nAbs)
      if lead == 0: raise newException(ValueError, "The leading coefficient vanishes modulo " & $q & " at n = " & $nAbs)
      result.add(mulMod((q - total) mod q, modularInverse(lead, q), q))

proc childCount(t: Transform, parentOffset, count: int): int =
  case t
  of tDifference, tRatio, tInverseEuler: count - 1
  of tMobius: parentOffset + count - 1
  else: count

proc untransform(child: seq[int64], parent: Transformed, t: Transform, count: int, q: int64): seq[int64] =
  ## Rebuild count parent values from the child and the parent's first term.
  let known = parent.values
  case t
  of tDifference:
    result = @[known[0]]
    for i in 0 ..< count - 1: result.add((result[i] + child[i]) mod q)
  of tPartialSum:
    for i in 0 ..< count: result.add(if i == 0: child[0] else: (child[i] - child[i - 1] + q) mod q)
  of tRatio:
    result = @[known[0]]
    for i in 0 ..< count - 1: result.add(mulMod(result[i], child[i], q))
  of tDivideFactorial, tMultiplyFactorial:
    var factorial = 1'i64 mod q
    for m in 2 .. parent.offset: factorial = mulMod(factorial, int64(m) mod q, q)
    for i in 0 ..< count:
      if i > 0: factorial = mulMod(factorial, int64(parent.offset + i) mod q, q)
      let scale = if t == tDivideFactorial: factorial else: modularInverse(factorial, q)
      result.add(mulMod(child[i], scale, q))
  of tBinomial: result = applyBinomial(child[0 ..< count], q, alternate = true)
  of tInverseBinomial: result = applyBinomial(child[0 ..< count], q, alternate = false)
  of tMobius:
    let last = parent.offset + count - 1
    var sums = newSeq[int64](last + 1)
    for d in 1 .. last:
      var m = d
      while m <= last:
        sums[m] = (sums[m] + child[d - 1]) mod q
        m += d
    for i in 0 ..< count:
      let n = parent.offset + i
      result.add(if n <= 0: known[i] else: sums[n])
  of tInverseEuler: result = eulerTransform(child, count, q)

proc nodeResidues(root: Track, path: seq[Transform], q: int64): Transformed =
  result = (true, root.exact.mapIt(it.modSmall(q)), root.offset)
  for t in path:
    result = transformResidues(result.values, result.offset, t, q)
    if not result.ok: raise newException(ValueError, "The transform " & $t & " is not defined modulo " & $q)

proc pathLimit(d: Direction, q: int64): int64 =
  ## The index window of a transform path. Binomial transforms are one NTT convolution
  ## modulo a prime between 2*10^6 and 2^31, and the Euler transform O(n log^2 n) modulo
  ## any q < 2^31; otherwise both are O(n^2) and stop at 20,000.
  const BinomialTransforms = {tBinomial, tInverseBinomial}
  if d.path.anyIt(it in BinomialTransforms) and not fastBinomialModulus(q): 20_000'i64
  elif tInverseEuler in d.path and not (q > 0 and q < FastEulerModulus): 20_000'i64
  else: MaxFastIndex

proc pathValuesMod(d: Direction, root: Track, relative: int64, q: int64): seq[int64] =
  ## Terms 0 .. relative (relative to the first term) through the transform path, modulo q.
  let limit = pathLimit(d, q)
  if relative > limit:
    raise newException(ValueError, "Indices through this transform path are limited to " & $limit)
  var nodes = @[nodeResidues(root, @[], q)]
  for i in 0 ..< d.path.len: nodes.add(nodeResidues(root, d.path[0 .. i], q))
  var counts = @[int(relative) + 1]
  for i, t in d.path: counts.add(childCount(t, nodes[i].offset, counts[i]))
  result = extendForm(d.form, nodes[^1].values, nodes[^1].offset, counts[^1], q)
  for i in countdown(d.path.high, 0):
    result = untransform(result, nodes[i], d.path[i], counts[i], q)
    if d.form.modular and d.path[i] == tRatio:
      for j in 0 ..< result.high:
        if result[j] == 0:
          raise newException(ValueError, "The ratio is undefined modulo " & $q & " at n = " & $(nodes[i].offset + j))
  result.setLen(int(relative) + 1)

proc pathTermMod(d: Direction, root: Track, relative: int64, q: int64): int64 =
  pathValuesMod(d, root, relative, q)[relative]

proc exactValueJson(value: BigInt): JsonNode =
  let text = $value
  if text.len <= MaxExactDigitsShown: return value.toJson()
  %*{"digits": value.digits(), "leading": text[0 ..< 40], "trailing": text[^40 .. ^1],
     "mod_1000000007": value.modSmall(1_000_000_007)}

proc formInfo*(d: Direction): tuple[rational, modular: bool] =
  (d.form.available or d.form.multiplicativeModel != mmNone, d.form.modular)

proc factorSmall(m: int64): seq[tuple[prime: int64, exponent: int]] =
  ## Every positive int64 (Pollard rho past trial division).
  factorize(m)

proc polynomialAt(coefficients: seq[Rational], x: BigInt): BigInt =
  let (common, ints) = scaledForm(coefficients)
  var value = big(0)
  for j in countdown(ints.high, 0): value = value * x + ints[j]
  exactQuotient(value, common)

proc polynomialAt(coefficients: seq[Rational], x: int64): BigInt = polynomialAt(coefficients, big(x))

proc multiplicativeTerm(form: TargetForm, root: Track, index: int64): tuple[value: BigInt, factors: string] =
  ## a(N) as the product over prime powers p^e || N. Prime powers inside the data are read
  ## from it; the rest come from the fitted model.
  if index < 1 or index > MaxMultiplicativeIndex:
    raise newException(ValueError, "A multiplicative far term needs an index from 1 to " & $MaxMultiplicativeIndex)
  let maxIndex = int64(root.offset + root.size - 1)
  proc at(m: int64): BigInt = root.exact[int(m - root.offset)]
  proc g(m: int64): BigInt =
    if m == 1: big(1) else: polynomialAt(form.multiplicativePolynomial, m)
  result.value = big(1)
  var pieces: seq[string]
  for (p, e) in factorSmall(index):
    var q = 1'i64
    for _ in 1 .. e: q *= p
    let local =
      if q <= maxIndex: at(q)
      else:
        case form.multiplicativeModel
        of mmCompletePrime:
          power((if p <= maxIndex: at(p) else: polynomialAt(form.multiplicativePolynomial, p)), e)
        of mmDivisorSum: g(q) - g(q div p)
        of mmNone:
          raise newException(ValueError, "The multiplicative direction needs the missing term a(" & $q & ")")
    result.value = result.value * local
    pieces.add((if e == 1: $p else: $p & "^" & $e) & " -> " & $local)
  result.factors = pieces.join(", ")

proc linearTermModBig(form: TargetForm, root: Track, steps: BigInt, m: BigInt): BigInt =
  ## Kitamasa's method with every residue a big integer, for a modulus of any size.
  let order = form.linear.len
  var coefficients: seq[BigInt]
  for c in form.linear: coefficients.add(floorMod(big(c.num) * inverseModBig(big(c.den), m), m))
  proc reduce(poly: var seq[BigInt]) =
    for k in countdown(poly.high, order):
      let top = poly[k]
      if top.isZero: continue
      for j in 1 .. order: poly[k - j] = floorMod(poly[k - j] + top * coefficients[j - 1], m)
      poly[k] = big(0)
    if poly.len > order: poly.setLen(order)
  proc multiply(a, b: seq[BigInt]): seq[BigInt] =
    result = newSeq[BigInt](max(0, a.len + b.len - 1))
    for i, x in a:
      if x.isZero: continue
      for j, y in b: result[i + j] = floorMod(result[i + j] + x * y, m)
    reduce(result)
  var bits: seq[int64]
  var rest = steps
  while not rest.isZero:
    let (quotient, remainder) = divSmall(rest, 2)
    bits.add(remainder)
    rest = quotient
  var weights = @[floorMod(big(1), m)]
  reduce(weights)
  var base = @[big(0), floorMod(big(1), m)]
  reduce(base)
  for bit in bits:
    if bit == 1: weights = multiply(weights, base)
    base = multiply(base, base)
  result = big(0)
  for j, w in weights: result = floorMod(result + w * root.exact[j], m)

proc isDivisorSum(d: Direction): bool =
  ## A Möbius-inverted polynomial: a(N) = sum over d | N of b(d), with b a polynomial.
  ## Factoring N gives the divisors, so any int64 index is in reach.
  d.path == @[tMobius] and d.form.kind == hPolynomial and d.form.available

proc divisorSumTerm(d: Direction, index: BigInt): BigInt =
  if not index.fitsInt64() or index < big(1):
    raise newException(ValueError, "A divisor sum needs an index from 1 to 2^63-1")
  var divisors = @[1'i64]
  for (p, e) in factorSmall(index.toInt64()):
    let current = divisors.len
    var power = 1'i64
    for _ in 1 .. e:
      power *= p
      for i in 0 ..< current: divisors.add(divisors[i] * power)
  result = big(0)
  for divisor in divisors: result = result + polynomialAt(d.form.polynomial, divisor)

# Hypergeometric terms modulo a small prime at any index. An order-one relation
# a(m+1) = R(m) a(m) whose R splits into linear factors over Q gives a(N) as a(m0) times
# products of arithmetic progressions prod (u k + v). Each product modulo p follows from
# Wilson's theorem (every full block of p terms contributes -1 to the unit part) and
# recursion on the terms divisible by p, which form another progression with the same
# step. The p-adic valuation is tracked, so zero divisors mod p cause no trouble: this
# is Lucas' theorem for binomials, generalised.

const
  MaxHypergeometricPrime = 4_000_000_000_000'i64  ## Past 10^7 the unit parts use the square root product.
  DirectProgressionPrime = 10_000_000'i64     ## Up to here each level is a plain O(p) loop.
  MaxRecurrenceModulus = 65_536'i64         ## Berlekamp-Massey tier: small primes only.
  MaxFoundRecurrence = 2_000                ## Kitamasa costs O(order^2 log n).

type
  LinearFactor = tuple[u, v: int64]          ## u x + v with u > 0 and gcd(u, v) = 1.
  PAdic = tuple[zero: bool, valuation: int64, unit: int64]

proc isPrimeModulus(q: int64): bool =
  if q < 2: return false
  let parts = factorSmall(q)
  parts.len == 1 and parts[0].exponent == 1

proc divisorsOf(n: int64): seq[int64] =
  result = @[1'i64]
  for (p, e) in factorSmall(n):
    let current = result.len
    var power = 1'i64
    for _ in 1 .. e:
      power *= p
      for i in 0 ..< current: result.add(result[i] * power)

proc linearFactors(coefficients: seq[Rational]): tuple[ok: bool, constant: tuple[num, den: BigInt],
                                                         factors: seq[LinearFactor]] =
  ## P(x) = constant * prod (u x + v), when P splits into linear factors over Q.
  let (common, ints) = scaledForm(coefficients)
  var poly = ints
  while poly.len > 0 and poly[^1].isZero: poly.setLen(poly.len - 1)
  if poly.len == 0: return
  proc divide(poly: var seq[BigInt], u, v: int64): bool =
    ## Exact division by u x + v in Z[x] (Gauss's lemma), or false.
    let d = poly.high
    var quotient = newSeq[BigInt](d)
    var carry = poly[d]
    for i in countdown(d - 1, 0):
      let (q, r) = divMod(carry, big(u))
      if not r.isZero: return false
      quotient[i] = q
      carry = poly[i] - big(v) * q
    if not carry.isZero: return false
    poly = quotient
    true
  while poly.len > 1:
    if poly[0].isZero:
      discard divide(poly, 1, 0)
      result.factors.add((1'i64, 0'i64))
      continue
    if not poly[0].fitsInt64() or not poly[^1].fitsInt64(): return
    let lowValue = abs(poly[0].toInt64())
    let highValue = abs(poly[^1].toInt64())
    if lowValue > 1_000_000_000_000'i64 or highValue > 1_000_000_000_000'i64: return
    var found = false
    block search:
      for u in divisorsOf(highValue):
        for a in divisorsOf(lowValue):
          for v in [a, -a]:
            if gcd(u, abs(v)) != 1: continue
            var trial = poly
            if divide(trial, u, v):
              poly = trial
              result.factors.add((u, v))
              found = true
              break search
    if not found: return
  result.ok = true
  result.constant = (poly[0], common)

const MaxPrimePowerBase = 50_000_000'i64
  ## A prime power modulus p^k (k >= 2) multiplies out one polynomial of p - 1 linear factors.

proc primePowerOf(q: int64): tuple[ok: bool, p: int64, alpha: int] =
  if q < 2: return
  let parts = factorSmall(q)
  if parts.len != 1: return
  (true, parts[0].prime, parts[0].exponent)

proc exponentOf(p, q: int64): int =
  var rest = q
  while rest > 1:
    rest = rest div p
    inc result

proc rangeProduct(g: seq[int64], count: int64, alpha: int, q: int64): int64 =
  ## prod_{j=0}^{count-1} g(j) modulo q = p^alpha, for a polynomial g whose coefficient of y^i
  ## is divisible by p^i. Such polynomials are closed under products and integer shifts
  ## y -> y + m, and their terms of degree alpha and up vanish modulo q, so
  ## P_m(y) = prod_{j<m} g(y + j) is kept to degree < alpha and doubles as
  ## P_2m(y) = P_m(y) P_m(y + m): O(alpha^2) per bit of count, whatever the size of q.
  let d = alpha
  var binomial = newSeq[seq[int64]](d)
  for k in 0 ..< d:
    binomial[k] = newSeq[int64](k + 1)
    binomial[k][0] = 1 mod q
    for i in 1 .. k: binomial[k][i] = (if i == k: 1 mod q else: (binomial[k - 1][i - 1] + binomial[k - 1][i]) mod q)
  proc shift(a: seq[int64], m: int64): seq[int64] =
    result = newSeq[int64](d)
    var powers = newSeq[int64](d)
    powers[0] = 1 mod q
    for k in 1 ..< d: powers[k] = mulMod(powers[k - 1], m mod q, q)
    for k in 0 ..< d:
      if a[k] == 0: continue
      for i in 0 .. k: result[i] = (result[i] + mulMod(a[k], mulMod(binomial[k][i], powers[k - i], q), q)) mod q
  proc multiply(a, b: seq[int64]): seq[int64] =
    result = newSeq[int64](d)
    for i in 0 ..< d:
      if a[i] == 0: continue
      for j in 0 ..< d - i: result[i + j] = (result[i + j] + mulMod(a[i], b[j], q)) mod q
  var base = newSeq[int64](d)
  for i in 0 ..< min(d, g.len): base[i] = g[i] mod q
  var acc = newSeq[int64](d)
  acc[0] = 1 mod q
  var m = 0'i64
  for bit in countdown(62, 0):
    if m > 0:
      let doubled = multiply(acc, shift(acc, m))
      acc = doubled
      m *= 2
    if ((count shr bit) and 1) == 1:
      let extended = multiply(acc, shift(base, m))
      acc = extended
      inc m
  acc[0]

proc padicOf(value: BigInt, p, q: int64): PAdic =
  ## value = p^valuation * unit, the unit modulo q (a power of p).
  if value.isZero: return (true, 0'i64, 0'i64)
  var rest = value
  while rest.modSmall(p) == 0:
    rest = rest.divSmall(p).quotient
    inc result.valuation
  result.unit = rest.modSmall(q)

proc combine(a, b: PAdic, q: int64): PAdic =
  if a.zero or b.zero: return (true, 0'i64, 0'i64)
  (false, a.valuation + b.valuation, mulMod(a.unit, b.unit, q))

proc unitGroupProduct(p, q: int64): int64 =
  ## The product of all units modulo q = p^k (Gauss's generalisation of Wilson's theorem).
  if p == 2 and q >= 8: 1'i64 else: (q - 1) mod q

proc progressionProduct(u, v, n, p, q: int64): PAdic =
  ## prod_{k=0}^{n-1} (u k + v) for u > 0, v > 0, as p^valuation * unit, the unit modulo q = p^k.
  if n <= 0: return (false, 0'i64, 1'i64 mod q)
  if u mod p == 0:
    # Every term is a unit (gcd(u, v) = 1), and g(y) = v + u y has its y coefficient
    # divisible by p, so the doubling product takes any n.
    if q == p: return (false, 0'i64, powMod(v mod p, n, p))
    return (false, 0'i64, rangeProduct(@[v mod q, u mod q], n, exponentOf(p, q), q))
  let um = u mod p
  let hit = mulMod(liftMod(-v, p), modularInverse(um, p), p)   # u*hit + v = 0 mod p
  var unit: int64
  if q == p:
    unit = if (n div p) mod 2 == 1: (p - 1) mod p else: 1'i64 mod p
    let vm = v mod p
    let rest = n mod p
    if p <= DirectProgressionPrime or rest < 4096:
      for t in 0'i64 ..< rest:
        if t != hit: unit = mulMod(unit, (mulMod(um, t, p) + vm) mod p, p)
    else:
      # prod (u t + v) over t in [0, rest) without t = hit, in O(sqrt(p) log p).
      let linear = @[@[@[vm, um]]]
      unit = mulMod(unit, sqrtMatrixProduct(linear, 0, min(hit, rest), p)[0], p)
      if hit + 1 < rest: unit = mulMod(unit, sqrtMatrixProduct(linear, hit + 1, rest - hit - 1, p)[0], p)
  else:
    # u is a unit mod q, so q consecutive terms run over every residue once: their units
    # multiply to the product of the unit group. The rest, t < n mod q, splits as
    # t = s + p j; the classes s != hit hold units c_s + u p j, and
    # prod_{s != hit} (c_s + u p j) = F(u p j) with F(X) = prod_{s != hit} (c_s + X) kept to
    # degree < alpha. The full rows j < J are one doubling product, the last row is direct.
    if p > MaxPrimePowerBase:
      raise newException(ValueError, "A prime power modulus needs its prime at most " & $MaxPrimePowerBase)
    let alpha = exponentOf(p, q)
    unit = powMod(unitGroupProduct(p, q), n div q, q)
    let rest = n mod q
    let uq = u mod q
    let vq = v mod q
    let step = mulMod(uq, p, q)      # u p
    let rows = rest div p
    if rows > 0:
      var f = newSeq[int64](alpha)
      f[0] = 1 mod q
      for s in 0'i64 ..< p:
        if s == hit: continue
        let c = (mulMod(uq, s, q) + vq) mod q
        for i in countdown(alpha - 1, 1): f[i] = (mulMod(f[i], c, q) + f[i - 1]) mod q
        f[0] = mulMod(f[0], c, q)
      var g = newSeq[int64](alpha)   # g(y) = F(u p y)
      var scale = 1'i64 mod q
      for i in 0 ..< alpha:
        g[i] = mulMod(f[i], scale, q)
        scale = mulMod(scale, step, q)
      unit = mulMod(unit, rangeProduct(g, rows, alpha, q), q)
    let last = mulMod(step, rows mod q, q)
    for s in 0'i64 ..< rest mod p:
      if s != hit: unit = mulMod(unit, ((mulMod(uq, s, q) + vq) mod q + last) mod q, q)
  let count = if hit < n: (n - hit - 1) div p + 1 else: 0'i64
  if count == 0: return (false, 0'i64, unit)
  let inner = progressionProduct(u, (u * hit + v) div p, count, p, q)
  (false, inner.valuation + count, mulMod(unit, inner.unit, q))

proc factorProduct(f: LinearFactor, first, n, p, q: int64): PAdic =
  ## prod_{m=first}^{first+n-1} (u m + v), any signs.
  result = (false, 0'i64, 1'i64 mod q)
  var v = f.u * first + f.v
  var left = n
  while left > 0 and v <= 0:   # the few non-positive terms, one at a time
    if v == 0: return (true, 0'i64, 0'i64)
    result = combine(result, padicOf(big(v), p, q), q)
    v += f.u
    dec left
  result = combine(result, progressionProduct(f.u, v, left, p, q), q)

proc hypergeometricTermMod(blocks: seq[seq[Rational]], root: Track, index: BigInt, q: int64): int64 =
  ## lead(m) a(m+1) = -tail(m) a(m), evaluated at any index modulo a prime or a prime power q.
  if blocks.len != 2: raise newException(ValueError, "not an order-one relation")
  if not index.fitsInt64(): raise newException(ValueError, "index is too large for a hypergeometric term")
  let power = primePowerOf(q)
  if not power.ok: raise newException(ValueError, "The p-adic route needs a prime power modulus")
  let p = power.p
  let tail = linearFactors(blocks[0])
  let lead = linearFactors(blocks[1])
  if not tail.ok or not lead.ok:
    raise newException(ValueError, "The step ratio does not split into linear factors over Q")
  let first = int64(root.offset + root.size - 1)
  let n = index.toInt64() - first
  var total = padicOf(root.exact[^1], p, q)
  if n <= 0: return root.exact[int(index.toInt64() - root.offset)].modSmall(q)
  # The constant -c_tail / c_lead, to the power n.
  let constantNum = padicOf(-(tail.constant.num * lead.constant.den), p, q)
  let constantDen = padicOf(tail.constant.den * lead.constant.num, p, q)
  if constantDen.zero: raise newException(ValueError, "The leading polynomial is zero")
  if constantNum.zero: return 0
  total = combine(total, (false, constantNum.valuation * n, powMod(constantNum.unit, n, q)), q)
  var denominator: PAdic = (false, constantDen.valuation * n, powMod(constantDen.unit, n, q))
  for f in tail.factors: total = combine(total, factorProduct(f, first, n, p, q), q)
  for f in lead.factors:
    let part = factorProduct(f, first, n, p, q)
    if part.zero: raise newException(ValueError, "The leading polynomial vanishes inside the range")
    denominator = combine(denominator, part, q)
  if total.zero: return 0
  let valuation = total.valuation - denominator.valuation
  if valuation < 0: raise newException(ValueError, "The term is not integral at " & $p)
  if valuation >= int64(power.alpha): return 0
  let unit = mulMod(total.unit, modularInverse(denominator.unit, q), q)
  mulMod(powMod(p, valuation, q), unit, q)

# Prime powers above 2^62 (2^64, the 5^30 inside 10^30): the same p-adic route with big
# integer residues. Sums are reduced once per coefficient, not once per product.

const MaxBigPrimePowerBase = 1_000_000'i64

type PAdicBig = tuple[zero: bool, valuation: int64, unit: BigInt]

proc powBig(base: BigInt, exponent: int64, q: BigInt): BigInt =
  result = floorMod(big(1), q)
  var b = floorMod(base, q)
  var e = exponent
  while e > 0:
    if (e and 1) == 1: result = floorMod(result * b, q)
    e = e shr 1
    if e > 0: b = floorMod(b * b, q)

proc zeros(count: int): seq[BigInt] =
  result = newSeq[BigInt](count)
  for i in 0 ..< count: result[i] = big(0)

proc rangeProductBig(g: seq[BigInt], count: int64, alpha: int, q: BigInt): BigInt =
  ## rangeProduct with big integer residues.
  let d = alpha
  var binomial = newSeq[seq[BigInt]](d)
  for k in 0 ..< d:
    binomial[k] = zeros(k + 1)
    binomial[k][0] = big(1)
    for i in 1 .. k: binomial[k][i] = (if i == k: big(1) else: binomial[k - 1][i - 1] + binomial[k - 1][i])
  proc shift(a: seq[BigInt], m: int64): seq[BigInt] =
    result = zeros(d)
    var powers = zeros(d)
    powers[0] = big(1)
    let step = floorMod(big(m), q)
    for k in 1 ..< d: powers[k] = floorMod(powers[k - 1] * step, q)
    for k in 0 ..< d:
      if a[k].isZero: continue
      for i in 0 .. k: result[i] = result[i] + a[k] * floorMod(binomial[k][i] * powers[k - i], q)
    for i in 0 ..< d: result[i] = floorMod(result[i], q)
  proc multiply(a, b: seq[BigInt]): seq[BigInt] =
    result = zeros(d)
    for i in 0 ..< d:
      if a[i].isZero: continue
      for j in 0 ..< d - i: result[i + j] = result[i + j] + a[i] * b[j]
    for i in 0 ..< d: result[i] = floorMod(result[i], q)
  var base = zeros(d)
  for i in 0 ..< min(d, g.len): base[i] = floorMod(g[i], q)
  var acc = zeros(d)
  acc[0] = floorMod(big(1), q)
  var m = 0'i64
  for bit in countdown(62, 0):
    if m > 0:
      let doubled = multiply(acc, shift(acc, m))
      acc = doubled
      m *= 2
    if ((count shr bit) and 1) == 1:
      let extended = multiply(acc, shift(base, m))
      acc = extended
      inc m
  acc[0]

proc progressionProductBig(u, v, n, p: int64, alpha: int, q: BigInt): PAdicBig =
  ## progressionProduct modulo a prime power q = p^alpha of any size.
  if n <= 0: return (false, 0'i64, floorMod(big(1), q))
  if u mod p == 0: return (false, 0'i64, rangeProductBig(@[big(v), big(u)], n, alpha, q))
  let hit = mulMod(liftMod(-v, p), modularInverse(u mod p, p), p)
  var unit = floorMod(big(1), q)
  var rest = n
  if q.fitsInt64():
    let whole = q.toInt64()
    unit = powBig(big(if p == 2 and whole >= 8: 1'i64 else: whole - 1), n div whole, q)
    rest = n mod whole
  let step = big(u) * big(p)
  let rows = rest div p
  if rows > 0:
    if p > MaxBigPrimePowerBase:
      raise newException(ValueError, "A prime power modulus above 2^62 needs its prime at most " & $MaxBigPrimePowerBase)
    var f = zeros(alpha)
    f[0] = big(1)
    for s in 0'i64 ..< p:
      if s == hit: continue
      let c = floorMod(big(u) * big(s) + big(v), q)
      for i in countdown(alpha - 1, 1): f[i] = floorMod(f[i] * c + f[i - 1], q)
      f[0] = floorMod(f[0] * c, q)
    var g = zeros(alpha)
    var scale = big(1)
    for i in 0 ..< alpha:
      g[i] = floorMod(f[i] * scale, q)
      scale = floorMod(scale * step, q)
    unit = floorMod(unit * rangeProductBig(g, rows, alpha, q), q)
  let last = step * big(rows)
  for s in 0'i64 ..< rest mod p:
    if s != hit: unit = floorMod(unit * (big(u) * big(s) + big(v) + last), q)
  let count = if hit < n: (n - hit - 1) div p + 1 else: 0'i64
  if count == 0: return (false, 0'i64, unit)
  let inner = progressionProductBig(u, (u * hit + v) div p, count, p, alpha, q)
  (false, inner.valuation + count, floorMod(unit * inner.unit, q))

proc padicOfBig(value: BigInt, p: int64, q: BigInt): PAdicBig =
  if value.isZero: return (true, 0'i64, big(0))
  var rest = value
  while rest.modSmall(p) == 0:
    rest = rest.divSmall(p).quotient
    inc result.valuation
  result.unit = floorMod(rest, q)

proc combineBig(a, b: PAdicBig, q: BigInt): PAdicBig =
  if a.zero or b.zero: return (true, 0'i64, big(0))
  (false, a.valuation + b.valuation, floorMod(a.unit * b.unit, q))

proc factorProductBig(f: LinearFactor, first, n, p: int64, alpha: int, q: BigInt): PAdicBig =
  result = (false, 0'i64, floorMod(big(1), q))
  var v = f.u * first + f.v
  var left = n
  while left > 0 and v <= 0:
    if v == 0: return (true, 0'i64, big(0))
    result = combineBig(result, padicOfBig(big(v), p, q), q)
    v += f.u
    dec left
  result = combineBig(result, progressionProductBig(f.u, v, left, p, alpha, q), q)

proc hypergeometricTermModBig(blocks: seq[seq[Rational]], root: Track, index: BigInt, p: int64, alpha: int,
                              q: BigInt): BigInt =
  ## hypergeometricTermMod modulo a prime power q = p^alpha of any size.
  if blocks.len != 2: raise newException(ValueError, "not an order-one relation")
  if not index.fitsInt64(): raise newException(ValueError, "index is too large for a hypergeometric term")
  let tail = linearFactors(blocks[0])
  let lead = linearFactors(blocks[1])
  if not tail.ok or not lead.ok:
    raise newException(ValueError, "The step ratio does not split into linear factors over Q")
  let first = int64(root.offset + root.size - 1)
  let n = index.toInt64() - first
  if n <= 0: return floorMod(root.exact[int(index.toInt64() - root.offset)], q)
  var total = padicOfBig(root.exact[^1], p, q)
  let constantNum = padicOfBig(-(tail.constant.num * lead.constant.den), p, q)
  let constantDen = padicOfBig(tail.constant.den * lead.constant.num, p, q)
  if constantDen.zero: raise newException(ValueError, "The leading polynomial is zero")
  if constantNum.zero: return big(0)
  total = combineBig(total, (false, constantNum.valuation * n, powBig(constantNum.unit, n, q)), q)
  var denominator: PAdicBig = (false, constantDen.valuation * n, powBig(constantDen.unit, n, q))
  for f in tail.factors: total = combineBig(total, factorProductBig(f, first, n, p, alpha, q), q)
  for f in lead.factors:
    let part = factorProductBig(f, first, n, p, alpha, q)
    if part.zero: raise newException(ValueError, "The leading polynomial vanishes inside the range")
    denominator = combineBig(denominator, part, q)
  if total.zero: return big(0)
  let valuation = total.valuation - denominator.valuation
  if valuation < 0: raise newException(ValueError, "The term is not integral at " & $p)
  if valuation >= int64(alpha): return big(0)
  floorMod(powBig(big(p), valuation, q) * total.unit * inverseModBig(denominator.unit, q), q)

proc powModBig(base, exponent, m: BigInt): BigInt =
  result = floorMod(big(1), m)
  var b = floorMod(base, m)
  var e = exponent
  while not e.isZero:
    let (quotient, bit) = divSmall(e, 2)
    if bit == 1: result = floorMod(result * b, m)
    e = quotient
    if not e.isZero: b = floorMod(b * b, m)

proc isProbablePrimeBig(n: BigInt): bool =
  ## Miller-Rabin with the first twelve prime bases.
  const bases = [2'i64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
  if n < big(2): return false
  for p in bases:
    if n.modSmall(p) == 0: return n == big(p)
  let minusOne = n - big(1)
  var d = minusOne
  var s = 0
  while d.modSmall(2) == 0:
    d = d.divSmall(2).quotient
    inc s
  for a in bases:
    var x = powModBig(big(a), d, n)
    if x == big(1) or x == minusOne: continue
    var composite = true
    for _ in 1 ..< s:
      x = floorMod(x * x, n)
      if x == minusOne:
        composite = false
        break
    if composite: return false
  true

proc gcdBig(a, b: BigInt): BigInt =
  var x = (if a.isNegative: -a else: a)
  var y = (if b.isNegative: -b else: b)
  while not y.isZero:
    let r = floorMod(x, y)
    x = y
    y = r
  x

proc pollardBig(n: BigInt, budget: int): BigInt =
  ## A nontrivial factor of a composite n with no prime factor below 10^5 (Brent's cycle
  ## search with batched gcds), or 0 when the budget of steps runs out.
  let one = big(1)
  for c in 1'i64 .. 3:
    var y = big(2)
    var x = y
    var ys = y
    var g = one
    var product = one
    var r = 1
    var steps = 0
    while g == one and steps < budget:
      x = y
      for _ in 1 .. r: y = floorMod(y * y + big(c), n)
      var k = 0
      while k < r and g == one:
        ys = y
        for _ in 1 .. min(128, r - k):
          y = floorMod(y * y + big(c), n)
          product = floorMod(product * (if x < y: y - x else: x - y), n)
        g = gcdBig(product, n)
        k += 128
      steps += r
      r *= 2
    if g == n:
      g = one
      while g == one:
        ys = floorMod(ys * ys + big(c), n)
        g = gcdBig(if x < ys: ys - x else: x - ys, n)
    if not (g == n) and not (g == one): return g
  big(0)

const EcmSeconds = 40.0   ## Time budget of the elliptic curve method on one cofactor.

proc primesUpTo(limit: int): seq[int64] =
  var composite = newSeq[bool](limit + 1)
  for i in 2 .. limit:
    if composite[i]: continue
    result.add(int64(i))
    var j = i * i
    while j <= limit:
      composite[j] = true
      j += i

proc ecmBig(n: BigInt, budget: float): BigInt =
  ## A nontrivial factor of a composite n by Lenstra's elliptic curve method: Montgomery curves
  ## with Suyama's parametrisation and x-only arithmetic, stage 1 to B1, then a baby-step
  ## giant-step stage 2 to 50 B1 with width 210. B1 rises from 2,000 to 250,000; 0 when the
  ## time budget runs out. Factors of about 20 digits take seconds, 25 digits about a minute.
  type Point = tuple[x, z: BigInt]
  let started = epochTime()
  let one = big(1)
  var a24 = big(0)
  proc md(x: BigInt): BigInt = floorMod(x, n)
  proc twice(p: Point): Point =
    let s = md(p.x + p.z)
    let d = md(p.x - p.z)
    let ss = md(s * s)
    let dd = md(d * d)
    let t = md(ss - dd)                   # 4 x z
    (md(ss * dd), md(t * md(dd + md(a24 * t))))
  proc addPoints(p, q, difference: Point): Point =
    let u = md(md(p.x - p.z) * md(q.x + q.z))
    let v = md(md(p.x + p.z) * md(q.x - q.z))
    let plus = md(u + v)
    let minus = md(u - v)
    (md(difference.z * md(plus * plus)), md(difference.x * md(minus * minus)))
  proc multiply(p: Point, k: int64): Point =
    ## Montgomery's ladder.
    if k == 1: return p
    var r0 = p
    var r1 = twice(p)
    var bit = 62
    while bit >= 0 and ((k shr bit) and 1) == 0: dec bit
    dec bit
    while bit >= 0:
      if ((k shr bit) and 1) == 1:
        r0 = addPoints(r1, r0, p)
        r1 = twice(r1)
      else:
        r1 = addPoints(r0, r1, p)
        r0 = twice(r0)
      dec bit
    r0
  const width = 210'i64
  var sigma = 6'i64
  for (b1, curves) in [(2_000'i64, 25), (11_000'i64, 90), (50_000'i64, 300), (250_000'i64, 1_000)]:
    let b2 = 50 * b1
    let primes = primesUpTo(int(b1))
    for _ in 1 .. curves:
      if epochTime() - started > budget: return big(0)
      inc sigma
      let s = big(sigma)
      let u = md(s * s - big(5))
      let v = md(big(4) * s)
      let denominator = md(md(big(16) * md(u * u)) * md(u * v))
      let g0 = gcdBig(denominator, n)
      if g0 == n: continue
      if not (g0 == one): return g0
      let gap = md(v - u)
      a24 = md(md(md(gap * gap) * gap) * md(md(big(3) * u + v) * inverseModBig(denominator, n)))
      var point: Point = (md(md(u * u) * u), md(md(v * v) * v))
      for p in primes:
        var power = p
        while power <= b1 div p: power *= p
        point = multiply(point, power)
      var g = gcdBig(point.z, n)
      if g == n: continue
      if not (g == one): return g
      # Stage 2: a prime l in (B1, B2] is 210 k ± j with j < 105 coprime to 210, and [l]Q = O
      # modulo a factor makes x([210 k]Q) = x([j]Q) there.
      var babies: seq[Point]
      let second = twice(point)
      var previous = point
      var current = addPoints(second, point, point)   # [3]Q
      babies.add(point)
      var j = 3'i64
      while j < width div 2:
        if gcd(j, width) == 1: babies.add(current)
        let next = addPoints(current, second, previous)
        previous = current
        current = next
        j += 2
      let step = multiply(point, width)
      let k0 = max(1'i64, b1 div width)
      var gA = multiply(point, width * k0)
      var gB = multiply(point, width * (k0 + 1))
      var acc = one
      for k in k0 .. b2 div width + 1:
        for b in babies: acc = md(acc * md(gA.x * b.z - b.x * gA.z))
        let next = addPoints(gB, step, gA)
        gA = gB
        gB = next
      g = gcdBig(acc, n)
      if not (g == n) and not (g == one): return g
  big(0)

proc primePowerParts(m: BigInt): seq[tuple[p: int64, e: int, part: BigInt]] =
  ## The prime power parts of a modulus of any size: trial division below 10^5, then
  ## Pollard rho and the elliptic curve method on big cofactors, Pollard rho on int64 ones.
  var rest = m
  var d = 2'i64
  while d <= 100_000 and big(1) < rest:
    if rest.modSmall(d) == 0:
      var e = 0
      var part = big(1)
      while rest.modSmall(d) == 0:
        rest = rest.divSmall(d).quotient
        part = part * big(d)
        inc e
      result.add((d, e, part))
    d += (if d == 2: 1 else: 2)
  if big(1) < rest:
    # Pollard rho (Brent) splits a big cofactor until every piece fits int64.
    var primes = initCountTable[int64]()
    var pending = @[rest]
    while pending.len > 0:
      let m = pending.pop()
      if m.fitsInt64():
        for (p, e) in factorSmall(m.toInt64()): primes.inc(p, e)
      elif isProbablePrimeBig(m):
        raise newException(ValueError, "The modulus has the prime factor " & $m & " above 2^63, where only " &
          "linear recurrences (Kitamasa on big integers) and exact values reach")
      else:
        var factor = pollardBig(m, 200_000)
        if factor.isZero: factor = ecmBig(m, EcmSeconds)
        if factor.isZero:
          raise newException(ValueError, "Pollard rho and the elliptic curve method (B1 up to 250,000, " &
            $int(EcmSeconds) & " s) found no factor of the cofactor " & $m &
            ": its prime factors are too large to split (integer factorisation)")
        pending.add(factor)
        pending.add(exactQuotientBig(m, factor))
    var keys: seq[int64]
    for p in primes.keys: keys.add(p)
    keys.sort()
    for p in keys:
      let e = primes[p]
      var part = big(1)
      for _ in 1 .. e: part = part * big(p)
      result.add((p, e, part))

var residueRoute* = ""   ## Names a residue route past the plain iteration, for `via` and `solution`.

# Holonomic terms past the modulus when the leading coefficient L(n) is a unit modulo p for
# every n: derangements, involutions, Bessel-type numbers. Modulo q = p^k the block step
# matrix B(x) = diag(M(x), L(x)) at x = x0 + s + p y gives one row of p steps,
# G(y) = B(x0 + p - 1 + p y) ... B(x0 + p y), a matrix of polynomials in y whose y^i
# coefficients are divisible by p^i, kept to degree < k. J rows multiply as
# P_2m(y) = P_m(y + m) P_m(y); for k = 1 that is G(0)^J, the product over one period.
# The last partial row is a direct (or square root) product, and one division by the
# product of the L(x), a unit, ends it.

const
  MaxPeriodPrimePowerBase = 2_000_000'i64   ## k >= 2: G(y) takes p products of matrix polynomials.
  MaxDirectPeriod = 20_000_000'i64          ## Longer rows modulo a prime use the square root product.

proc hasRootModPrime(poly: seq[int64], p: int64): bool =
  ## Whether the polynomial (ascending coefficients) has a root modulo the prime p.
  var f = poly.mapIt(liftMod(it, p))
  while f.len > 0 and f[^1] == 0: f.setLen(f.len - 1)
  if f.len == 0: return true
  if f.len == 1: return false
  if p <= 1_000_000:
    for x in 0'i64 ..< p:
      var value = 0'i64
      for j in countdown(f.high, 0): value = (mulMod(value, x, p) + f[j]) mod p
      if value == 0: return true
    return false
  # gcd(f, x^p - x) has positive degree exactly when f has a root.
  let inverseLead = modularInverse(f[^1], p)
  f = f.mapIt(mulMod(it, inverseLead, p))
  let d = f.high
  proc reduce(a: seq[int64]): seq[int64] =
    result = a
    for k in countdown(result.high, d):
      let top = result[k]
      if top == 0: continue
      for j in 0 .. d: result[k - d + j] = (result[k - d + j] + p - mulMod(top, f[j], p)) mod p
    if result.len > d: result.setLen(d)
  proc multiply(a, b: seq[int64]): seq[int64] =
    var product = newSeq[int64](a.len + b.len - 1)
    for i, x in a:
      for j, y in b: product[i + j] = (product[i + j] + mulMod(x, y, p)) mod p
    reduce(product)
  var power = @[1'i64]
  var base = reduce(@[0'i64, 1'i64])
  var e = p
  while e > 0:
    if (e and 1) == 1: power = multiply(power, base)
    base = multiply(base, base)
    e = e shr 1
  while power.len < 2: power.add(0)
  power[1] = (power[1] + p - 1) mod p   # x^p - x modulo f
  var a = f
  var b = power
  while b.len > 0 and b[^1] == 0: b.setLen(b.len - 1)
  while b.len > 0:
    # a mod b
    let inverse = modularInverse(b[^1], p)
    var rest = a
    for k in countdown(rest.high, b.high):
      let factor = mulMod(rest[k], inverse, p)
      if factor == 0: continue
      for j in 0 .. b.high: rest[k - b.high + j] = (rest[k - b.high + j] + p - mulMod(factor, b[j], p)) mod p
    rest.setLen(b.high)
    while rest.len > 0 and rest[^1] == 0: rest.setLen(rest.len - 1)
    a = b
    b = rest
  a.len >= 2

proc periodTermMod(blocks: seq[seq[Rational]], root: Track, steps: int64, p: int64, alpha: int,
                   q: int64): int64 =
  ## Term `steps` (relative) modulo q = p^alpha of a holonomic relation whose leading
  ## coefficient has no root modulo p.
  let residues = blocks.mapIt(it.mapIt(ratMod(it, q)))
  let r = residues.high
  let n = root.size
  let total = steps - int64(n) + 1
  if total <= 0: return root.exact[int(steps)].modSmall(q)
  if hasRootModPrime(residues[r], p):
    raise newException(ValueError, "The leading coefficient vanishes modulo " & $p & " at some n")
  if alpha >= 2 and p > MaxPeriodPrimePowerBase:
    raise newException(ValueError, "The period route modulo a prime power needs its prime at most " &
      $MaxPeriodPrimePowerBase)
  let x0 = int64(root.offset + n - r)
  let width = r + 1
  var entries = newSeq[seq[seq[int64]]](width)
  for i in 0 .. r: entries[i] = newSeq[seq[int64]](width)
  for k in 0 ..< r - 1: entries[k][k + 1] = residues[r]
  for k in 0 ..< r: entries[r - 1][k] = residues[k].mapIt((q - it) mod q)
  entries[r][r] = residues[r]
  proc identity(): seq[int64] =
    result = newSeq[int64](width * width)
    for i in 0 ..< width: result[i * width + i] = 1 mod q
  proc multiply(later, earlier: seq[int64]): seq[int64] =
    result = newSeq[int64](width * width)
    for i in 0 ..< width:
      for j in 0 ..< width:
        var sum = 0'i64
        for k in 0 ..< width: sum = (sum + mulMod(later[i * width + k], earlier[k * width + j], q)) mod q
        result[i * width + j] = sum
  proc stepAt(x: int64): seq[int64] =
    result = newSeq[int64](width * width)
    let point = liftMod(x, q)
    for i in 0 ..< width:
      for j in 0 ..< width:
        let poly = entries[i][j]
        var value = 0'i64
        for c in countdown(poly.high, 0): value = (mulMod(value, point, q) + poly[c]) mod q
        result[i * width + j] = value
  proc rowProduct(start, count: int64): seq[int64] =
    ## prod_{x = start .. start + count - 1} B(x), later factors on the left.
    if count >= 4096 and alpha == 1 and q >= SqrtProductMinModulus and q < FastEulerModulus:
      return sqrtMatrixProduct(entries, liftMod(start, q), count, q)
    if count > MaxDirectPeriod:
      raise newException(ValueError, "A row of " & $count & " steps is too long without the square root product")
    result = identity()
    for x in start ..< start + count:
      let next = multiply(stepAt(x), result)
      result = next
  let rows = total div p
  let rest = total mod p
  var product = identity()
  if rows > 0:
    if alpha == 1:
      var base = rowProduct(x0, p)
      var e = rows
      while e > 0:
        if (e and 1) == 1:
          let next = multiply(base, product)
          product = next
        e = e shr 1
        if e > 0:
          let squared = multiply(base, base)
          base = squared
    else:
      # Matrix polynomials: entry (i, j) holds coefficients 0 ..< alpha at [(i * width + j) * alpha + c].
      let size = width * width * alpha
      proc polyIdentity(): seq[int64] =
        result = newSeq[int64](size)
        for i in 0 ..< width: result[(i * width + i) * alpha] = 1 mod q
      proc polyMultiply(later, earlier: seq[int64]): seq[int64] =
        result = newSeq[int64](size)
        for i in 0 ..< width:
          for j in 0 ..< width:
            let target = (i * width + j) * alpha
            for k in 0 ..< width:
              let a = (i * width + k) * alpha
              let b = (k * width + j) * alpha
              for u in 0 ..< alpha:
                let x = later[a + u]
                if x == 0: continue
                for v in 0 ..< alpha - u:
                  result[target + u + v] = (result[target + u + v] + mulMod(x, earlier[b + v], q)) mod q
      var binomial = newSeq[seq[int64]](alpha)
      for k in 0 ..< alpha:
        binomial[k] = newSeq[int64](k + 1)
        binomial[k][0] = 1 mod q
        for i in 1 .. k: binomial[k][i] = (if i == k: 1 mod q else: (binomial[k - 1][i - 1] + binomial[k - 1][i]) mod q)
      proc polyShift(a: seq[int64], m: int64): seq[int64] =
        result = newSeq[int64](size)
        var powers = newSeq[int64](alpha)
        powers[0] = 1 mod q
        for k in 1 ..< alpha: powers[k] = mulMod(powers[k - 1], m mod q, q)
        for e in 0 ..< width * width:
          let o = e * alpha
          for k in 0 ..< alpha:
            if a[o + k] == 0: continue
            for i in 0 .. k:
              result[o + i] = (result[o + i] + mulMod(a[o + k], mulMod(binomial[k][i], powers[k - i], q), q)) mod q
      proc stepPoly(a: int64): seq[int64] =
        ## B(a + p y) as a matrix of polynomials in y, kept to degree < alpha.
        result = newSeq[int64](size)
        let am = liftMod(a, q)
        let pm = p mod q
        for i in 0 ..< width:
          for j in 0 ..< width:
            let poly = entries[i][j]
            let o = (i * width + j) * alpha
            var value = newSeq[int64](alpha)
            for c in countdown(poly.high, 0):
              # value = value * (a + p y) + poly[c]
              var next = newSeq[int64](alpha)
              for u in 0 ..< alpha:
                if value[u] == 0: continue
                next[u] = (next[u] + mulMod(value[u], am, q)) mod q
                if u + 1 < alpha: next[u + 1] = (next[u + 1] + mulMod(value[u], pm, q)) mod q
              next[0] = (next[0] + poly[c]) mod q
              value = next
            for u in 0 ..< alpha: result[o + u] = value[u]
      var g = polyIdentity()
      for s in 0'i64 ..< p:
        let next = polyMultiply(stepPoly(x0 + s), g)
        g = next
      var acc = polyIdentity()
      var m = 0'i64
      for bit in countdown(62, 0):
        if m > 0:
          let doubled = polyMultiply(polyShift(acc, m), acc)
          acc = doubled
          m *= 2
        if ((rows shr bit) and 1) == 1:
          let extended = polyMultiply(polyShift(g, m), acc)
          acc = extended
          inc m
      for e in 0 ..< width * width: product[e] = acc[e * alpha]
  if rest > 0:
    let partial = rowProduct(x0 + rows * p, rest)
    let next = multiply(partial, product)
    product = next
  let lead = product[r * width + r]
  var value = 0'i64
  for k in 0 ..< r:
    value = (value + mulMod(product[(r - 1) * width + k], root.exact[n - r + k].modSmall(q), q)) mod q
  residueRoute = "the product of the step matrices over rows of p = " & $p & " steps (the leading " &
    "coefficient is a unit modulo " & $p & ")" & (if alpha == 1: ", raised to the number of rows"
    else: ", multiplied as matrix polynomials modulo " & $q & " by doubling")
  mulMod(value, modularInverse(lead, q), q)

# Binomial sums S(n) = sum_k w^k prod_i C(T_i, B_i)^e_i, with T and B linear in n and k, modulo
# q = p^a <= 2^22: Apery, Franel, sum C(n,k)^4, central Delannoy and trinomial numbers, Motzkin
# as sum C(n,2k) Cat(k). Granville's theorem gives a binomial modulo p^a from the windows
# X_j = floor(X / p^j) mod p^a of its top, bottom and rest R = T - B:
#   C(T, B) = (+-1)^(e_{a-1}) p^(e_0) prod_j (T_j!)_p / ((B_j!)_p (R_j!)_p)  mod p^a,
# with (x!)_p the product of the units up to x, e_0 the carries of B + R and e_{a-1} those from
# digit a - 1 on (the sign is +1 for 2^a with a >= 3). Each factor depends on one window, and
# floor(F / p^j) = a floor(n / p^j) + b floor(k / p^j) + carry_j for F = a n + b k + c, so a digit
# DP over k keeps the last a - 1 digits of k and one carry per form. A form that ends negative
# leaves a nonzero carry, which removes the out-of-range k at the end.

const
  MaxBinomialSumModulus = 4_194_304'i64      ## 2^22: the unit factorial table and the digit states.
  MaxBinomialSumStates = 16_777_216'i64
  MaxBinomialSumWork = 1_500_000_000'i64     ## states x digits x positions

type
  LinearForm = tuple[a, b, c: int64]          ## a n + b k + c
  BinomialFactor = tuple[top, bottom: LinearForm, power: int]
  BinomialTerm = object                       ## sign * sum_k weight^k prod C(top, bottom)^power
    factors: seq[BinomialFactor]
    weight: Rational
    sign: int64

proc floorDiv64(a, b: int64): int64 =
  ## floor(a / b) for b > 0.
  result = a div b
  if a mod b < 0: dec result

proc minus(x, y: LinearForm): LinearForm = (x.a - y.a, x.b - y.b, x.c - y.c)

proc kRange(t: BinomialTerm, n: int64): tuple[bounded: bool, low, high: int64] =
  ## The k >= 0 for which every top, bottom and rest is nonnegative.
  result = (false, 0'i64, high(int64))
  for f in t.factors:
    for form in [f.top, f.bottom, minus(f.top, f.bottom)]:
      let base = form.a * n + form.c          # form = base + b k
      if form.b == 0:
        if base < 0: return (true, 1'i64, 0'i64)
      elif form.b > 0: result.low = max(result.low, floorDiv64(-base + form.b - 1, form.b))
      else:
        result.high = min(result.high, floorDiv64(base, -form.b))
        result.bounded = true

var unitFactorialCache = initTable[int64, tuple[units, inverses: seq[int32]]]()

proc unitFactorials(p, q: int64): tuple[units, inverses: seq[int32]] =
  ## (x!)_p and its inverse modulo q for every x < q (int32 keeps the tables in cache longer).
  if q in unitFactorialCache: return unitFactorialCache[q]
  var units = newSeq[int32](q)
  units[0] = int32(1 mod q)
  for x in 1'i64 ..< q: units[x] = (if x mod p == 0: units[x - 1] else: int32(int64(units[x - 1]) * x mod q))
  var inverses = newSeq[int32](q)
  inverses[q - 1] = int32(modularInverse(int64(units[q - 1]), q))
  for x in countdown(q - 1, 1):
    inverses[x - 1] = (if x mod p == 0: inverses[x] else: int32(int64(inverses[x]) * x mod q))
  result = (units, inverses)
  if unitFactorialCache.len > 4: unitFactorialCache.clear()
  unitFactorialCache[q] = result

proc binomialTermMod(t: BinomialTerm, n: int64, p: int64, alpha: int, q: int64): int64 =
  ## sum_k weight^k prod C(top, bottom)^power modulo q = p^alpha <= 2^22, any n >= 0.
  let range = kRange(t, n)
  if not range.bounded: raise newException(ValueError, "The binomial sum has no upper bound on k")
  if range.high < range.low: return 0
  let tables = unitFactorials(p, q)
  var forms: seq[LinearForm]
  proc indexOf(f: LinearForm): int =
    result = forms.find(f)
    if result < 0:
      forms.add(f)
      result = forms.high
  var slots: seq[tuple[top, bottom, rest, power: int]]
  for f in t.factors: slots.add((indexOf(f.top), indexOf(f.bottom), indexOf(minus(f.top, f.bottom)), f.power))
  let count = forms.len
  var h = 1'i64
  for _ in 1 ..< alpha: h *= p
  var kDigits = 0
  var x = range.high
  while x > 0:
    x = x div p
    inc kDigits
  var largest = 0'i64
  for f in forms: largest = max(largest, abs(f.a) * n + abs(f.b) * range.high + abs(f.c))
  var topDigits = 0
  x = largest
  while x > 0:
    x = x div p
    inc topDigits
  let positions = max(topDigits, kDigits + 3) + 2
  var nDigit = newSeq[int64](positions)
  var nWindow = newSeq[int64](positions)
  block:
    var rest = n
    for j in 0 ..< positions:
      nDigit[j] = rest mod p
      nWindow[j] = rest mod q
      rest = rest div p
  # Carry ranges, widened until the step floor((a n_j + b k_j + carry) / p) stays inside.
  var low, stride: seq[int64]
  var combos = 1'i64
  for f in forms:
    var lo = f.c
    var hi = f.c
    let minAdd = min(f.a, 0) * (p - 1) + min(f.b, 0) * (p - 1)
    let maxAdd = max(f.a, 0) * (p - 1) + max(f.b, 0) * (p - 1)
    while true:
      let nlo = min(lo, floorDiv64(minAdd + lo, p))
      let nhi = max(hi, floorDiv64(maxAdd + hi, p))
      if nlo == lo and nhi == hi: break
      lo = nlo
      hi = nhi
    low.add(lo)
    stride.add(combos)
    combos *= hi - lo + 1
  var sizes = newSeq[int64](count)
  for i in 0 ..< count: sizes[i] = (if i + 1 < count: stride[i + 1] else: combos) div stride[i]
  let states = h * combos
  if states > MaxBinomialSumStates or states * p * int64(positions) > MaxBinomialSumWork:
    raise newException(ValueError, "The binomial sum modulo " & $q & " needs too many digit states")
  let sign = if p == 2 and alpha >= 3: 1'i64 else: q - 1
  var current = newSeq[int64](states)
  var next = newSeq[int64](states)
  block:
    var startCombo = 0'i64
    for i, f in forms: startCombo += (f.c - low[i]) * stride[i]
    var firstDigits = 1'i64
    for _ in 1 .. min(alpha - 1, kDigits): firstDigits *= p
    for k0 in 0'i64 ..< firstDigits: current[k0 + h * startCombo] = 1 mod q
  let weight = ratMod(t.weight, q)
  var weightStep = weight                      # weight^(p^j)
  var carries = newSeq[int64](count)
  var windows = newSeq[int64](count)
  var spread = newSeq[int64](count)            # a * n_window mod q per form
  var weightPowers = newSeq[int64](p)
  # Per step: the next carry of each form for every (digit of k, carry), as an offset into the
  # combo index, so the inner loop does no division.
  var carryStep = newSeq[seq[int64]](count)
  for j in 0 ..< positions:
    for i, f in forms:
      carryStep[i] = newSeq[int64](p * sizes[i])
      for d in 0'i64 ..< p:
        for c in 0'i64 ..< sizes[i]:
          let carry = floorDiv64(f.a * nDigit[j] + f.b * d + low[i] + c, p)
          carryStep[i][d * sizes[i] + c] = (carry - low[i]) * stride[i]
    for i in 0 ..< states: next[i] = 0
    for i, f in forms: spread[i] = liftMod(f.a mod q * nWindow[j] mod q, q)
    weightPowers[0] = 1 mod q
    for d in 1 ..< p: weightPowers[d] = weightPowers[d - 1] * weightStep mod q
    let digitLimit = if j + alpha - 1 < kDigits: p else: 1'i64
    for combo in 0'i64 ..< combos:
      var anyValue = false
      for k in 0'i64 ..< h:
        if current[k + h * combo] != 0:
          anyValue = true
          break
      if not anyValue: continue
      var carryIndex = newSeq[int64](count)
      for i in 0 ..< count:
        carryIndex[i] = (combo div stride[i]) mod sizes[i]
        carries[i] = low[i] + carryIndex[i]
      for k in 0'i64 ..< h:
        let value = current[k + h * combo]
        if value == 0: continue
        for d in 0'i64 ..< digitLimit:
          let window = k + d * h                 # floor(k / p^j) mod p^alpha
          let kDigit = window mod p
          var newCombo = 0'i64
          for i in 0 ..< count:
            var w = spread[i] + forms[i].b * window + carries[i]
            while w < 0: w += q
            while w >= q: w -= q
            windows[i] = w
            newCombo += carryStep[i][kDigit * sizes[i] + carryIndex[i]]
          var factor = value * weightPowers[kDigit] mod q
          for s in slots:
            let top = windows[s.top]
            let bottom = windows[s.bottom]
            let rest = windows[s.rest]
            var part = int64(tables.units[top]) * int64(tables.inverses[bottom]) mod q *
              int64(tables.inverses[rest]) mod q
            if liftMod(top - bottom - rest, q) == 1:   # a carry into digit j
              part = part * p mod q
              if j >= alpha: part = part * sign mod q
            for _ in 1 .. s.power: factor = factor * part mod q
          if factor != 0:
            let target = window div p + h * newCombo
            next[target] = (next[target] + factor) mod q
    swap(current, next)
    var stepped = 1 mod q
    for _ in 0 ..< p: stepped = stepped * weightStep mod q
    weightStep = stepped
  var zeroCombo = 0'i64
  for i in 0 ..< count: zeroCombo += (0 - low[i]) * stride[i]
  result = current[h * zeroCombo]
  if t.sign < 0: result = (q - result) mod q

type BinomialSumForm = object
  ok: bool
  shift: int64                ## a(N) = scale * S(N + shift)
  scale: Rational
  terms: seq[BinomialTerm]
  name: string

const BinomialMenu = [
  (name: "C(n,k)", top: (1'i64, 0'i64, 0'i64), bottom: (0'i64, 1'i64, 0'i64), bounding: true),
  (name: "C(n+k,k)", top: (1'i64, 1'i64, 0'i64), bottom: (0'i64, 1'i64, 0'i64), bounding: false),
  (name: "C(2k,k)", top: (0'i64, 2'i64, 0'i64), bottom: (0'i64, 1'i64, 0'i64), bounding: false),
  (name: "C(n,2k)", top: (1'i64, 0'i64, 0'i64), bottom: (0'i64, 2'i64, 0'i64), bounding: true),
  (name: "C(n-k,k)", top: (1'i64, -1'i64, 0'i64), bottom: (0'i64, 1'i64, 0'i64), bounding: true),
  (name: "C(2n-2k,n-k)", top: (2'i64, -2'i64, 0'i64), bottom: (1'i64, -1'i64, 0'i64), bounding: true),
  (name: "C(n+k,2k)", top: (1'i64, 1'i64, 0'i64), bottom: (0'i64, 2'i64, 0'i64), bounding: true),
  (name: "Cat(k)", top: (0'i64, 2'i64, 0'i64), bottom: (0'i64, 1'i64, 0'i64), bounding: false)]
  ## Cat(k) = C(2k, k) - C(2k, k + 1) enters as two terms.

const BinomialWeights = [1'i64, -1, 2, -2, 3, -3, 4, -4, 8, -8, 9, 16, 27]

const
  BinomialFitTerms = 40
  BinomialFitPrimes = [2_305_843_009_213_693_951'i64, 1_000_000_007'i64]

var binomialSumCache = initTable[string, BinomialSumForm]()

proc binomialCandidates(): seq[tuple[items: seq[int], weight: int64]] =
  ## Products of up to four menu binomials (at most three distinct, Cat(k) at most once) with at
  ## least one that bounds k, by degree and then by weight size.
  var multisets: seq[seq[int]]
  proc grow(current: seq[int], start: int) =
    if current.len > 0: multisets.add(current)
    if current.len == 4: return
    for i in start ..< BinomialMenu.len: grow(current & @[i], i)
  grow(@[], 0)
  for degree in 1 .. 4:
    for weight in BinomialWeights:
      for items in multisets:
        if items.len != degree: continue
        if items.deduplicate.len > 3 or items.count(7) > 1: continue
        if not items.anyIt(BinomialMenu[it].bounding): continue
        result.add((items, weight))

proc expandCandidate(items: seq[int], weight: int64): seq[BinomialTerm] =
  ## The menu product as signed binomial terms (Cat(k) splits in two).
  var base: seq[BinomialFactor]
  var catalan = false
  for i in items.deduplicate:
    if i == 7:
      catalan = true
      continue
    base.add((BinomialMenu[i].top, BinomialMenu[i].bottom, items.count(i)))
  let w: Rational = (weight, 1'i64)
  if not catalan: return @[BinomialTerm(factors: base, weight: w, sign: 1)]
  @[BinomialTerm(factors: base & @[((0'i64, 2'i64, 0'i64), (0'i64, 1'i64, 0'i64), 1)], weight: w, sign: 1),
    BinomialTerm(factors: base & @[((0'i64, 2'i64, 0'i64), (0'i64, 1'i64, 1'i64), 1)], weight: w, sign: -1)]

proc binomialTermsModPrime(terms: seq[BinomialTerm], n: int64, factorial, inverseFactorial: seq[int64],
                           p: int64): int64 =
  ## The sum directly, modulo a prime p above every top.
  for t in terms:
    let range = kRange(t, n)
    if not range.bounded or range.high < range.low: continue
    var total = 0'i64
    var power = powMod(ratMod(t.weight, p), range.low, p)
    let w = ratMod(t.weight, p)
    for k in range.low .. range.high:
      var value = power
      for f in t.factors:
        let top = f.top.a * n + f.top.b * k + f.top.c
        let bottom = f.bottom.a * n + f.bottom.b * k + f.bottom.c
        let c = mulMod(factorial[top], mulMod(inverseFactorial[bottom], inverseFactorial[top - bottom], p), p)
        for _ in 1 .. f.power: value = mulMod(value, c, p)
      total = (total + value) mod p
      power = mulMod(power, w, p)
    result = (result + (if t.sign < 0: (p - total) mod p else: total)) mod p

proc binomialTermsExact(terms: seq[BinomialTerm], n: int64): BigInt =
  proc binomial(top, bottom: int64): BigInt =
    result = big(1)
    for i in 1'i64 .. bottom: result = (result * big(top - bottom + i)).divSmall(i).quotient
  result = big(0)
  for t in terms:
    let range = kRange(t, n)
    if not range.bounded or range.high < range.low: continue
    for k in range.low .. range.high:
      var value = big(1)
      for _ in 1'i64 .. k: value = value * big(t.weight.num)   # integer weights in the fit
      for f in t.factors:
        let c = binomial(f.top.a * n + f.top.b * k + f.top.c, f.bottom.a * n + f.bottom.b * k + f.bottom.c)
        for _ in 1 .. f.power: value = value * c
      result = (if t.sign < 0: result - value else: result + value)

proc binomialSumFormUncached(blocks: seq[seq[Rational]], root: Track): BinomialSumForm =
  var terms = root.exact
  try:
    for steps in terms.len ..< BinomialFitTerms: terms.add(holonomicTermExact(blocks, root, int64(steps)))
  except ValueError:
    return
  let m = min(terms.len, 64)
  if m < 24: return
  let first = int64(root.offset)
  let last = first + int64(m) + 1
  var tables: seq[tuple[factorial, inverse: seq[int64]]]
  for p in BinomialFitPrimes:
    let size = int(4 * last + 8)
    var factorial = newSeq[int64](size)
    factorial[0] = 1
    for i in 1 ..< size: factorial[i] = mulMod(factorial[i - 1], int64(i), p)
    var inverse = newSeq[int64](size)
    inverse[size - 1] = modularInverse(factorial[size - 1], p)
    for i in countdown(size - 1, 1): inverse[i - 1] = mulMod(inverse[i], int64(i), p)
    tables.add((factorial, inverse))
  var targets: seq[seq[int64]]
  for p in BinomialFitPrimes:
    var residues: seq[int64]
    for i in 0 ..< m: residues.add(terms[i].modSmall(p))
    targets.add(residues)
  for (items, weight) in binomialCandidates():
    let expanded = expandCandidate(items, weight)
    for shift in [0'i64, 1, -1]:
      if first + shift < 0: continue
      var fits = true
      var scales: seq[int64]
      for pi, p in BinomialFitPrimes:
        var scale = -1'i64
        for i in 0 ..< m:
          let value = binomialTermsModPrime(expanded, first + int64(i) + shift, tables[pi].factorial,
                                            tables[pi].inverse, p)
          let target = targets[pi][i]
          if scale < 0:
            if value == 0:
              if target != 0:
                fits = false
                break
              continue
            scale = mulMod(target, modularInverse(value, p), p)
          elif mulMod(scale, value, p) != target:
            fits = false
            break
        if not fits or scale < 0:
          fits = false
          break
        scales.add(scale)
      if not fits: continue
      # The scale as an exact fraction from the first nonzero term.
      var position = -1
      var sumValue = big(0)
      for i in 0 ..< m:
        sumValue = binomialTermsExact(expanded, first + int64(i) + shift)
        if not sumValue.isZero:
          position = i
          break
      if position < 0: continue
      var numerator = terms[position]
      var denominator = sumValue
      if denominator.isNegative:
        numerator = -numerator
        denominator = -denominator
      var common = (if numerator.isNegative: -numerator else: numerator)
      var other = denominator
      while not other.isZero:
        let remainder = floorMod(common, other)
        common = other
        other = remainder
      numerator = exactQuotientBig(numerator, common)
      denominator = exactQuotientBig(denominator, common)
      if not numerator.fitsInt64() or not denominator.fitsInt64(): continue
      let scale: Rational = (numerator.toInt64(), denominator.toInt64())
      var agrees = true
      for pi, p in BinomialFitPrimes:
        if ratMod(scale, p) != scales[pi]: agrees = false
      if not agrees: continue
      var name = (if scale == (1'i64, 1'i64): "" else: $scale.num & (if scale.den != 1: "/" & $scale.den else: "") & " * ") &
        "sum_k " & (if weight == 1: "" else: "(" & $weight & ")^k ")
      var pieces: seq[string]
      for i in items.deduplicate:
        pieces.add(BinomialMenu[i].name & (if items.count(i) > 1: "^" & $items.count(i) else: ""))
      name &= pieces.join(" ")
      if shift != 0: name &= " at n " & (if shift > 0: "+ " & $shift else: "- " & $(-shift))
      return BinomialSumForm(ok: true, shift: shift, scale: scale, terms: expanded, name: name)

proc binomialSumForm(blocks: seq[seq[Rational]], root: Track): BinomialSumForm =
  let key = $root.offset & ":" & $root.exact.len & ":" & $root.exact[^1] & ":" & $blocks
  if key in binomialSumCache: return binomialSumCache[key]
  result = binomialSumFormUncached(blocks, root)
  binomialSumCache[key] = result

proc binomialSumMod(form: BinomialSumForm, index: BigInt, p: int64, alpha: int, q: int64): int64 =
  if q > MaxBinomialSumModulus:
    raise newException(ValueError, "The binomial sum route takes a prime power up to " & $MaxBinomialSumModulus)
  let shifted = index + big(form.shift)
  if not shifted.fitsInt64() or shifted.isNegative or shifted > big(2_000_000_000_000_000_000'i64):
    raise newException(ValueError, "index is out of range for the binomial sum")
  let n = shifted.toInt64()
  var total = 0'i64
  for t in form.terms: total = (total + binomialTermMod(t, n, p, alpha, q)) mod q
  result = mulMod(ratMod(form.scale, q), total, q)
  residueRoute = "the binomial sum a(n) = " & form.name & " (fitted on " & $BinomialFitTerms &
    " terms of the relation and checked modulo two primes, not proved), each binomial modulo " & $q &
    " by Granville's theorem in a digit DP over k"

proc chineseRemainderBig(residues, moduli: seq[BigInt]): BigInt =
  var x = big(0)
  var m = big(1)
  for i in 0 ..< residues.len:
    let t = floorMod((residues[i] - x) * inverseModBig(floorMod(m, moduli[i]), moduli[i]), moduli[i])
    x = x + m * t
    m = m * moduli[i]
  floorMod(x, m)

# The Lucas property a(n p + j) = a(n) a(j) mod p holds for many constant-term sequences:
# C(2n, n), central trinomial and Delannoy numbers, sums of powers of binomials such as the
# Franel numbers, and the Apery numbers (Gessel). When exact terms satisfy it modulo every
# small prime tested, a far term modulo a prime q is the product of the terms at the base-q
# digits of the index, each below q, so the index may far exceed the modulus. The property
# is checked, not proved; the route says so.

const
  LucasCheckTerms = 360
  LucasPrimes = [5'i64, 7, 11, 13, 17, 19, 23, 29, 31]

var lucasCache = initTable[string, bool]()

proc lucasPropertyUncached(blocks: seq[seq[Rational]], root: Track): bool

proc lucasProperty(blocks: seq[seq[Rational]], root: Track): bool =
  if root.offset != 0: return false
  let key = $root.exact.len & ":" & $root.exact[^1] & ":" & $blocks
  if key in lucasCache: return lucasCache[key]
  result = lucasPropertyUncached(blocks, root)
  lucasCache[key] = result

proc lucasPropertyUncached(blocks: seq[seq[Rational]], root: Track): bool =
  var terms = root.exact
  try:
    for steps in terms.len .. LucasCheckTerms: terms.add(holonomicTermExact(blocks, root, int64(steps)))
  except ValueError:
    return false
  var checks = 0
  for p in LucasPrimes:
    let residues = terms.mapIt(it.modSmall(p))
    if residues[0] != 1: return false
    for index in int(p) ..< residues.len:
      let n = index div int(p)
      let j = index mod int(p)
      if residues[index] != residues[n] * residues[j] mod p: return false
      inc checks
  checks >= 1_000

proc holonomicDigitMod(blocks: seq[seq[Rational]], root: Track, digit: int64, q: int64): int64 =
  ## a(digit) modulo q for an index below q: supplied, iterated, or the square root product.
  if digit < int64(root.size): return root.exact[int(digit)].modSmall(q)
  if digit > SqrtProductFrom and q >= SqrtProductMinModulus and q < FastEulerModulus:
    return holonomicSqrtMod(blocks, root, digit, q)
  if digit > MaxTargetIterations:
    raise newException(ValueError, "A base-" & $q & " digit is past the iteration window")
  holonomicTermMod(blocks, root, digit, q)

proc lucasTermMod(blocks: seq[seq[Rational]], root: Track, index: BigInt, q: int64): int64 =
  result = 1 mod q
  var rest = index
  while not rest.isZero:
    let (quotient, digit) = divSmall(rest, q)
    result = mulMod(result, holonomicDigitMod(blocks, root, digit, q), q)
    if result == 0: return
    rest = quotient
  residueRoute = "the Lucas property a(n p + j) = a(n) a(j) mod p (checked on " & $LucasCheckTerms &
    " exact terms modulo primes 5 to 31, not proved): the product of the terms at the base-" & $q &
    " digits of the index"

# Constant-term sequences a(n + s) = CT[P(x)^n Q(x)] with a Laurent polynomial P of three to
# five terms and a Laurent polynomial Q: Motzkin, Riordan and Schroeder numbers, central
# trinomial and Delannoy numbers, sections such as Riordan(2n), and their shifts. Modulo a
# prime p, P^p = P(x^p), so a_R(n p + j) = a_S(n) with S = Lambda_p(P^j R), where Lambda_p
# keeps the coefficients at multiples of p and divides their exponents by p (the Cartier
# operator). S keeps a bounded support, so each base-p digit of the index costs a few
# coefficients of a polynomial power. Those follow a recurrence whose leading factor
# C0 (t + 1) vanishes first at t + 1 = p; a walk that must pass it runs modulo p^2, where the
# one factor p of the running denominator divides out exactly. Long stretches use the square
# root product. A first-order inhomogeneous form b(n) - mu b(n - 1) = CT[P^n Q] (Fine's
# numbers) sums the powers of y = P / mu: the digit sums of y^m are T_d(y) = (y^(d+1) - 1) /
# (y - 1), and (y - 1)^p = y^p - 1 = V(x^p) mod p turns the division into an exact division
# by V(x) = y - 1 after the Cartier step. Every form is fitted on terms of the relation and
# checked modulo three primes, not proved; the route says so.

const
  ConstantTermTerms = 64
  ConstantTermSupport = 4   ## Q has support [-4, 4].
  ConstantTermSpare = 24    ## Equations beyond the rank of the fit.

type
  Laurent = tuple[low: int64, c: seq[int64]]   ## sum c[i] x^(low + i), residues
  ConstantTermForm = object
    ok: bool
    shift: int                ## b(n) = a(n + shift).
    left: int                 ## P = sum laurent[i] x^(i - left).
    laurent: seq[int64]
    weights: seq[Rational]    ## Q_k for k = -support .. support.
    inhomogeneous: bool       ## b(n) - mu b(n - 1) = CT[P^n Q] for n >= 1.
    mu: Rational
    first: BigInt             ## b(0) of an inhomogeneous form.

var constantTermCache = initTable[string, ConstantTermForm]()

proc floorDivision(x, y: int64): int64 =
  ## floor(x / y) for y > 0.
  if x >= 0: x div y else: -((-x + y - 1) div y)

proc solveConsistent(rows: seq[seq[int64]], rhs: seq[int64], p: int64):
    tuple[ok: bool, rank: int, x: seq[int64]] =
  ## A solution of rows * x = rhs modulo a prime p, free unknowns zero, if the system is consistent.
  let cols = rows[0].len
  var m = newSeq[seq[int64]](rows.len)
  for i in 0 ..< rows.len: m[i] = rows[i] & @[rhs[i] mod p]
  var pivots: seq[int]
  var rank = 0
  for col in 0 ..< cols:
    var pivot = rank
    while pivot < m.len and m[pivot][col] == 0: inc pivot
    if pivot == m.len: continue
    swap(m[rank], m[pivot])
    let inverse = modularInverse(m[rank][col], p)
    for k in col .. cols: m[rank][k] = mulMod(m[rank][k], inverse, p)
    for i in 0 ..< m.len:
      if i != rank and m[i][col] != 0:
        let f = m[i][col]
        for k in col .. cols: m[i][k] = (m[i][k] - mulMod(f, m[rank][k], p) + p) mod p
    pivots.add(col)
    inc rank
  for i in rank ..< m.len:
    if m[i][cols] != 0: return
  result.ok = true
  result.rank = rank
  result.x = newSeq[int64](cols)
  for i, col in pivots: result.x[col] = m[i][cols]

proc constantTermBasis(laurent: seq[int64], left, count, support: int, p: int64): seq[seq[int64]] =
  ## Row n holds CT[P^n x^k] = [x^-k] P^n for k = -support .. support, modulo p.
  let coefficients = laurent.mapIt(liftMod(it, p))
  var power = @[1'i64 mod p]   # P^n; index e + n * left holds exponent e
  for n in 0 ..< count:
    var row = newSeq[int64](2 * support + 1)
    for k in -support .. support:
      let index = -k + n * left
      if index >= 0 and index < power.len: row[k + support] = power[index]
    result.add(row)
    var next = newSeq[int64](power.len + laurent.len - 1)
    for i, v in power:
      if v == 0: continue
      for s, c in coefficients:
        if c != 0: next[i + s] = (next[i + s] + mulMod(v, c, p)) mod p
    power = next

proc constantTermShapes(): seq[tuple[left: int, laurent: seq[int64]]] =
  ## Three-term P = a/x + b + c x with |a|, c <= 4 and |b| <= 6, then four- and five-term P
  ## with end coefficients in {±1, ±2} (the top one positive: x -> -x flips odd powers),
  ## inner ones in [-2, 2] and the constant in [-4, 4].
  for c in 1'i64 .. 4:
    for a in [1'i64, -1, 2, -2, 3, -3, 4, -4]:
      for b in -6'i64 .. 6: result.add((1, @[a, b, c]))
  for (left, right) in [(1, 2), (2, 1), (2, 2)]:
    let width = left + right + 1
    var ranges: seq[seq[int64]]
    for i in 0 ..< width:
      if i == width - 1: ranges.add(@[1'i64, 2])
      elif i == 0: ranges.add(@[1'i64, -1, 2, -2])
      elif i == left: ranges.add(@[0'i64, 1, -1, 2, -2, 3, -3, 4, -4])
      else: ranges.add(@[0'i64, 1, -1, 2, -2])
    var choice = newSeq[int](width)
    while true:
      var laurent = newSeq[int64](width)
      for i in 0 ..< width: laurent[i] = ranges[i][choice[i]]
      result.add((left, laurent))
      var i = 0
      while i < width:
        inc choice[i]
        if choice[i] < ranges[i].len: break
        choice[i] = 0
        inc i
      if i == width: break

proc fitConstantTerm(basis: seq[seq[int64]], terms: seq[int64], shift: int, inhomogeneous: bool,
                     p: int64): tuple[ok: bool, rank, rows: int, x: seq[int64]] =
  ## b(n) = sum Q_k CT[P^n x^k] (+ mu b(n - 1)) on every row, b(n) = terms[n + shift].
  var rows: seq[seq[int64]]
  var rhs: seq[int64]
  for n in (if inhomogeneous: 1 else: 0) ..< terms.len - shift:
    var row = basis[n]
    if inhomogeneous: row.add(terms[n + shift - 1])
    rows.add(row)
    rhs.add(terms[n + shift])
  let solved = solveConsistent(rows, rhs, p)
  (solved.ok, solved.rank, rows.len, solved.x)

proc constantTermFormUncached(blocks: seq[seq[Rational]], root: Track): ConstantTermForm =
  const fitPrime = 2_305_843_009_213_693_951'i64   # 2^61 - 1
  const checkPrimes = [1_000_000_007'i64, 998_244_353]
  let count = ConstantTermTerms
  let support = ConstantTermSupport
  var terms: seq[seq[int64]]
  try:
    terms.add(holonomicValuesMod(blocks, root, int64(count - 1), fitPrime))
    for p in checkPrimes: terms.add(holonomicValuesMod(blocks, root, int64(count - 1), p))
  except ValueError:
    return
  let shapes = constantTermShapes()
  for inhomogeneous in [false, true]:
    for shape in shapes:
      if inhomogeneous and shape.laurent.len != 3: continue
      let screenBasis = constantTermBasis(shape.laurent, shape.left, count, support, checkPrimes[0])
      var fitBasis: seq[seq[int64]]
      for shift in 0 .. 2:
        let screen = fitConstantTerm(screenBasis, terms[1], shift, inhomogeneous, checkPrimes[0])
        if not screen.ok or screen.rows - screen.rank < ConstantTermSpare: continue
        if inhomogeneous and screen.x[^1] == 0: continue   # mu = 0 is a plain form
        if fitBasis.len == 0: fitBasis = constantTermBasis(shape.laurent, shape.left, count, support, fitPrime)
        let fit = fitConstantTerm(fitBasis, terms[0], shift, inhomogeneous, fitPrime)
        if not fit.ok: continue
        var unknowns: seq[Rational]
        var good = true
        for x in fit.x:
          let r = rationalOf(x, fitPrime)
          if not r.ok:
            good = false
            break
          unknowns.add((r.num, r.den))
        if not good: continue
        try:
          for t, p in checkPrimes:
            let basis = if t == 0: screenBasis else: constantTermBasis(shape.laurent, shape.left, count, support, p)
            let values = unknowns.mapIt(ratMod(it, p))
            for n in (if inhomogeneous: 1 else: 0) ..< count - shift:
              var total = 0'i64
              for k in 0 ..< basis[n].len: total = (total + mulMod(basis[n][k], values[k], p)) mod p
              if inhomogeneous: total = (total + mulMod(values[^1], terms[t + 1][n + shift - 1], p)) mod p
              if total != terms[t + 1][n + shift]:
                good = false
                break
            if not good: break
        except ValueError:
          good = false
        if not good: continue
        result = ConstantTermForm(ok: true, shift: shift, left: shape.left, laurent: shape.laurent,
                                  weights: unknowns[0 ..< 2 * support + 1], inhomogeneous: inhomogeneous)
        if inhomogeneous:
          result.mu = unknowns[^1]
          result.first = if shift < root.size: root.exact[shift] else: holonomicTermExact(blocks, root, int64(shift))
        return

proc constantTermForm(blocks: seq[seq[Rational]], root: Track): ConstantTermForm =
  let key = $root.offset & ":" & $root.exact.len & ":" & $root.exact[^1] & ":" & $blocks
  if key in constantTermCache: return constantTermCache[key]
  result = constantTermFormUncached(blocks, root)
  constantTermCache[key] = result

# Coefficient walks that pass t = p for primes p from 2^31 to 2^43, where p^2 is above 2^62. A
# residue modulo p^2 is a pair of base-p digits lo + hi p. A product of two low digits splits as
# q p + r with a floating-point quotient corrected in int64. A convolution needs the exact
# product of the low digits and the cross terms, both below 2^110: four NTT primes (2^115) hold
# them, and Garner's mixed radix digits fold in modulo p^2 and modulo p. The square root
# product then runs as for a prime, with these residues.

type Duo = tuple[lo, hi: int64]   ## lo + hi p modulo p^2, both digits in [0, p)

const
  duoZero: Duo = (0'i64, 0'i64)
  duoOne: Duo = (1'i64, 0'i64)
  MaxWidePrime = 8_796_093_022_208'i64       ## 2^43
  WidePrimes = 4                             ## NTT primes for the exact digit products

proc mulSplit(a, b, p: int64): tuple[rem, quot: int64] {.inline.} =
  ## a b = quot p + rem for 0 <= a < 2^31 or a < p, 0 <= b < p < 2^43.
  var q = int64(float64(a) * float64(b) / float64(p))
  var r = cast[int64](cast[uint64](a) * cast[uint64](b) - cast[uint64](q) * cast[uint64](p))
  while r < 0:
    r += p
    dec q
  while r >= p:
    r -= p
    inc q
  (r, q)

proc duo(x, p: int64): Duo =
  let r = liftMod(x, p)
  (r, liftMod((x - r) div p, p))

proc duoAdd(a, b: Duo, p: int64): Duo {.inline.} =
  var lo = a.lo + b.lo
  var hi = a.hi + b.hi
  if lo >= p:
    lo -= p
    inc hi
  if hi >= p: hi -= p
  (lo, hi)

proc duoSub(a, b: Duo, p: int64): Duo {.inline.} =
  var lo = a.lo - b.lo
  var hi = a.hi - b.hi
  if lo < 0:
    lo += p
    dec hi
  if hi < 0: hi += p
  (lo, hi)

proc mulRem(a, b, p: int64): int64 {.inline.} = mulSplit(a, b, p).rem

proc duoMul(a, b: Duo, p: int64): Duo {.inline.} =
  let (r, q) = mulSplit(a.lo, b.lo, p)
  var hi = q + mulRem(a.lo, b.hi, p) + mulRem(a.hi, b.lo, p)   # below 3 p
  if hi >= p: hi -= p
  if hi >= p: hi -= p
  (r, hi)

proc duoScale(d: int64, a: Duo, p: int64): Duo {.inline.} =
  ## d a modulo p^2 for 0 <= d < 2^31.
  let (r, q) = mulSplit(d, a.lo, p)
  var hi = q + mulRem(d, a.hi, p)
  if hi >= p: hi -= p
  if hi >= p: hi = hi mod p
  (r, hi)

proc duoInverse(a: Duo, p: int64): Duo =
  ## (a0 + a1 p)(i0 + x1 p) = 1 with a0 i0 = 1 + c p gives x1 = -(c + a1 i0) i0.
  if a.lo == 0: raise newException(ValueError, "A residue divisible by " & $p & " has no inverse modulo p^2")
  let i0 = modularInverse(a.lo, p)
  let c = mulSplit(a.lo, i0, p).quot
  (i0, mulMod(liftMod(-((c + mulMod(a.hi, i0, p)) mod p), p), i0, p))

proc duoPower(base: Duo, e: int64, p: int64): Duo =
  result = duoOne
  var b = base
  var k = e
  while k > 0:
    if (k and 1) == 1: result = duoMul(result, b, p)
    k = k shr 1
    if k > 0: b = duoMul(b, b, p)

{.push boundChecks: off, overflowChecks: off, rangeChecks: off.}   # the digit convolution loops

type
  DuoSpectrum = array[WidePrimes, array[2, seq[uint64]]]
    ## Forward NTTs of the low and the high digits modulo each of the first NTT primes.
  DuoRadix = object
    ## Garner's mixed radix: value = sum_j d_j (m_0 ... m_{j-1}). The residues modulo m_t
    ## are kept in Montgomery form, so a digit costs products and no division.
    radix: array[WidePrimes, array[WidePrimes, uint64]]   ## m_0 ... m_{j-1} mod m_t
    inverse: array[WidePrimes, uint64]                    ## (m_0 ... m_{t-1})^-1 mod m_t
    weight: array[WidePrimes, Duo]                        ## m_0 ... m_{j-1} mod p^2

proc duoRadix(p: int64): DuoRadix =
  prepareNtt(1)
  var running = duoOne
  for j in 0 ..< WidePrimes:
    result.weight[j] = running
    running = duoMul(running, duo(int64(NttPrimes[j][0]), p), p)
    for t in 0 ..< WidePrimes:
      let m = NttPrimes[t][0]
      var product = 1'u64
      for l in 0 ..< j: product = product * (NttPrimes[l][0] mod m) mod m
      result.radix[j][t] = nttFields[t].toMont(product)
  for t in 0 ..< WidePrimes:
    let m = NttPrimes[t][0]
    var product = 1'u64
    for l in 0 ..< t: product = product * (NttPrimes[l][0] mod m) mod m
    result.inverse[t] = nttFields[t].toMont(powU(product, m - 2, m))

proc garnerDigits(r: DuoRadix, residues: array[WidePrimes, uint64]): array[WidePrimes, uint64] {.inline.} =
  ## Plain digits from plain residues; a plain value times a Montgomery one is plain.
  for t in 0 ..< WidePrimes:
    let m = NttPrimes[t][0]
    var earlier = 0'u64
    for j in 0 ..< t:
      earlier += nttFields[t].mul(result[j], r.radix[j][t])
      if earlier >= m: earlier -= m
    let x = (if residues[t] >= earlier: residues[t] - earlier else: residues[t] + m - earlier)
    result[t] = nttFields[t].mul(x, r.inverse[t])

proc duoSpectrumOf(values: seq[Duo], size: int): DuoSpectrum =
  ## The digit spectra of values zero padded to size, a power of two.
  var levels = 0
  while (1 shl levels) < size: inc levels
  prepareNtt(levels)
  for t in 0 ..< WidePrimes:
    let z = addr nttFields[t]
    for part in 0 .. 1:
      var data = newSeq[uint64](size)
      for i in 0 ..< min(values.len, size):
        data[i] = z[].toMont(uint64(if part == 0: values[i].lo else: values[i].hi))
      z[].ntt(data, false)
      result[t][part] = data

proc duoProducts(left, right: DuoSpectrum, first, last: int, r: DuoRadix, p: int64): seq[Duo] =
  ## Indices first ..< last of the cyclic product of two digit spectra, modulo p^2. The low
  ## digit product and the cross terms are exact below 2^110 and come back by Garner's CRT.
  let size = left[0][0].len
  var low, cross: array[WidePrimes, seq[uint64]]
  for t in 0 ..< WidePrimes:
    let z = addr nttFields[t]
    let m = z[].m
    low[t] = newSeq[uint64](size)
    cross[t] = newSeq[uint64](size)
    for i in 0 ..< size:
      low[t][i] = z[].mul(left[t][0][i], right[t][0][i])
      let s = z[].mul(left[t][0][i], right[t][1][i]) + z[].mul(left[t][1][i], right[t][0][i])
      cross[t][i] = (if s >= m: s - m else: s)
    z[].ntt(low[t], true)
    z[].ntt(cross[t], true)
    let scale = powU(uint64(size), m - 2, m)     # a plain factor, so the products come back plain
    for i in first ..< last:
      low[t][i] = z[].mul(low[t][i], scale)
      cross[t][i] = z[].mul(cross[t][i], scale)
  result = newSeq[Duo](last - first)
  var residues: array[WidePrimes, uint64]
  for i in first ..< last:
    for t in 0 ..< WidePrimes: residues[t] = low[t][i]
    let d = r.garnerDigits(residues)
    var value = duoZero
    for j in 0 ..< WidePrimes: value = duoAdd(value, duoScale(int64(d[j]), r.weight[j], p), p)
    for t in 0 ..< WidePrimes: residues[t] = cross[t][i]
    let e = r.garnerDigits(residues)
    var high = value.hi
    for j in 0 ..< WidePrimes:
      high += mulRem(int64(e[j]), r.weight[j].lo, p)
      if high >= p: high -= p
    result[i - first] = (value.lo, high)

{.pop.}

proc batchInverseDuo(values: seq[Duo], p: int64): seq[Duo] =
  result = newSeq[Duo](values.len)
  var running = duoOne
  for i, x in values:
    if x.lo == 0: raise newException(ValueError, "A sample shift meets a multiple of " & $p)
    result[i] = running
    running = duoMul(running, x, p)
  var inverse = duoInverse(running, p)
  for i in countdown(values.high, 0):
    result[i] = duoMul(result[i], inverse, p)
    inverse = duoMul(inverse, values[i], p)

proc lagrangeWeights(k: int, p: int64): seq[Duo] =
  ## (-1)^(k - j) / (j! (k - j)!) modulo p^2 for j = 0 .. k < p.
  var factorial = newSeq[Duo](k + 1)
  factorial[0] = duoOne
  for i in 1 .. k: factorial[i] = duoMul(factorial[i - 1], duo(int64(i), p), p)
  var inverseFactorial = newSeq[Duo](k + 1)
  inverseFactorial[k] = duoInverse(factorial[k], p)
  for i in countdown(k, 1): inverseFactorial[i - 1] = duoMul(inverseFactorial[i], duo(int64(i), p), p)
  result = newSeq[Duo](k + 1)
  for j in 0 .. k:
    result[j] = duoMul(inverseFactorial[j], inverseFactorial[k - j], p)
    if (k - j) mod 2 == 1: result[j] = duoSub(duoZero, result[j], p)

type DuoShift = object
  ## One Lagrange shift f(0 .. k) -> f(a), ..., f(a + count - 1), shared by every column:
  ## f(a + t) = prod_{i = 0 .. k} (a + t - i) * sum_j g(j) / (a + t - j), g(j) = f(j) w(j).
  k, count: int
  spectrum: DuoSpectrum     ## h(m) = 1 / (a - k + m) for m < k + count
  factors: seq[Duo]         ## prod_{i = 0 .. k} (a + t - i)

proc duoShiftPlan(a: Duo, k, count, size: int, p: int64): DuoShift =
  ## size >= k + count keeps the wrapped part of the cyclic product below index k.
  var points = newSeq[Duo](k + count)
  for m in 0 ..< k + count: points[m] = duoAdd(a, duo(int64(m - k), p), p)
  let h = batchInverseDuo(points, p)
  result.k = k
  result.count = count
  result.spectrum = duoSpectrumOf(h, size)
  result.factors = newSeq[Duo](count)
  var product = duoOne
  for i in 0 .. k: product = duoMul(product, duoSub(a, duo(int64(i), p), p), p)
  for t in 0 ..< count:
    result.factors[t] = product
    product = duoMul(duoMul(product, duoAdd(a, duo(int64(t + 1), p), p), p), h[t], p)

proc applyShift(plan: DuoShift, weighted: DuoSpectrum, r: DuoRadix, p: int64): seq[Duo] =
  result = duoProducts(weighted, plan.spectrum, plan.k, plan.k + plan.count, r, p)
  for t in 0 ..< plan.count: result[t] = duoMul(plan.factors[t], result[t], p)

proc sqrtMatrixProductDuo(entries: seq[seq[seq[Duo]]], x0, total: int64, p: int64): seq[Duo] =
  ## sqrtMatrixProduct modulo p^2 for a prime p from 2^31 to 2^43.
  let r = entries.len
  let width = r * r
  var degree = 1
  for row in entries:
    for poly in row: degree = max(degree, poly.high)
  proc evalAt(coefficients: seq[Duo], x: int64): Duo =
    let point = duo(x, p)
    result = duoZero
    for j in countdown(coefficients.high, 0): result = duoAdd(duoMul(result, point, p), coefficients[j], p)
  proc stepAt(x: int64): seq[Duo] =
    result = newSeq[Duo](width)
    for i in 0 ..< r:
      for j in 0 ..< r:
        if entries[i][j].len > 0: result[i * r + j] = evalAt(entries[i][j], x)
  proc multiply(later, earlier: seq[Duo]): seq[Duo] =
    result = newSeq[Duo](width)
    for i in 0 ..< r:
      for k in 0 ..< r:
        let u = later[i * r + k]
        if u.lo == 0 and u.hi == 0: continue
        for j in 0 ..< r:
          let w = earlier[k * r + j]
          if w.lo != 0 or w.hi != 0: result[i * r + j] = duoAdd(result[i * r + j], duoMul(u, w, p), p)
  var product = newSeq[Duo](width)
  for i in 0 ..< r: product[i * r + i] = duoOne
  if total <= 0: return product
  if total < 4096:
    for x in x0 ..< x0 + total: product = multiply(stepAt(x), product)
    return product
  var v = 1'i64
  while int64(degree) * v * v < total: v *= 2
  if int64(degree) * v > MaxSqrtProductSamples:
    raise newException(ValueError, "The range is too long for the square root product")
  let inverseV = duoInverse(duo(v, p), p)
  let radix = duoRadix(p)
  var samples: seq[seq[Duo]]
  for i in 0 .. degree: samples.add(stepAt(x0 + int64(i) * v))
  var d = 1'i64
  while d < v:
    let k = samples.len - 1
    let shift = duoMul(duo(d, p), inverseV, p)
    var size = 1
    while size < 3 * k + 1: size = size shl 1
    # One weight list and two shift plans serve every column of this level; each column
    # costs one forward transform of its weighted samples and two products.
    let weights = lagrangeWeights(k, p)
    let upperPlan = duoShiftPlan(duo(int64(k + 1), p), k, k, size, p)
    let shiftPlan = duoShiftPlan(shift, k, 2 * k + 1, size, p)
    var extended = newSeq[seq[Duo]](2 * k + 1)
    var moved = newSeq[seq[Duo]](2 * k + 1)
    for i in 0 .. 2 * k:
      extended[i] = newSeq[Duo](width)
      moved[i] = newSeq[Duo](width)
    for c in 0 ..< width:
      var column = newSeq[Duo](k + 1)
      for i in 0 .. k: column[i] = samples[i][c]
      if column.allIt(it.lo == 0 and it.hi == 0): continue
      var weighted = newSeq[Duo](k + 1)
      for j in 0 .. k: weighted[j] = duoMul(column[j], weights[j], p)
      let spectrum = duoSpectrumOf(weighted, size)
      let upper = upperPlan.applyShift(spectrum, radix, p)
      let shifted = shiftPlan.applyShift(spectrum, radix, p)
      for i in 0 .. k: extended[i][c] = column[i]
      for i in 0 ..< k: extended[k + 1 + i][c] = upper[i]
      for i in 0 .. 2 * k: moved[i][c] = shifted[i]
    samples = newSeq[seq[Duo]](2 * k + 1)
    for i in 0 .. 2 * k: samples[i] = multiply(moved[i], extended[i])
    d *= 2
  let giant = total div v
  for i in 0 ..< int(giant): product = multiply(samples[i], product)
  for x in x0 + giant * v ..< x0 + total: product = multiply(stepAt(x), product)
  product

proc powerProductCoefficientsWide(a: seq[int64], alpha: int64, b: seq[int64], beta: int64,
                                  wanted: seq[int64], p: int64): Table[int64, int64] =
  ## powerProductCoefficients past t = p for a prime p from 2^31 to 2^43: the same recurrence
  ## C0 (t + 1) F(t + 1) = sum_{i >= 1} (E(i - 1) - C(i) (t + 1 - i)) F(t + 1 - i) modulo p^2.
  proc reduced(poly: seq[int64]): seq[Duo] = poly.mapIt(duo(it, p))
  proc product(x, y: seq[Duo]): seq[Duo] =
    result = newSeq[Duo](x.len + y.len - 1)
    for i, u in x:
      for j, w in y: result[i + j] = duoAdd(result[i + j], duoMul(u, w, p), p)
  proc derivative(x: seq[Duo]): seq[Duo] =
    result = newSeq[Duo](max(1, x.len - 1))
    for i in 1 ..< x.len: result[i - 1] = duoMul(x[i], duo(int64(i), p), p)
  let ar = reduced(a)
  let br = reduced(b)
  let c = product(ar, br)
  var e = newSeq[Duo](c.len)
  let left = product(derivative(ar), br)
  let right = product(ar, derivative(br))
  for i in 0 ..< left.len: e[i] = duoAdd(e[i], duoMul(duo(alpha, p), left[i], p), p)
  for i in 0 ..< right.len: e[i] = duoAdd(e[i], duoMul(duo(beta, p), right[i], p), p)
  let order = c.len - 1
  if c[0].lo == 0: raise newException(ValueError, "The polynomial power has a zero constant term modulo " & $p)
  var entries = newSeq[seq[seq[Duo]]](order + 1)
  for i in 0 .. order: entries[i] = newSeq[seq[Duo]](order + 1)
  for i in 1 .. order:
    entries[0][i - 1] = @[duoAdd(e[i - 1], duoMul(c[i], duo(int64(i - 1), p), p), p), duoSub(duoZero, c[i], p)]
  for r in 1 ..< order: entries[r][r - 1] = @[c[0], c[0]]
  entries[order][order] = @[c[0], c[0]]
  var state = newSeq[Duo](order + 1)
  state[0] = duoMul(duoPower(ar[0], alpha, p), duoPower(br[0], beta, p), p)
  state[order] = duoOne
  var position = 0'i64
  var targets = wanted
  targets.sort()
  for t in targets:
    if t in result: continue
    let m = sqrtMatrixProductDuo(entries, position, t - position, p)   # direct below 4096 steps
    var next = newSeq[Duo](order + 1)
    for i in 0 .. order:
      for k in 0 .. order: next[i] = duoAdd(next[i], duoMul(m[i * (order + 1) + k], state[k], p), p)
    state = next
    position = t
    let numerator = state[0]
    let denominator = state[order]
    if denominator.lo != 0:
      result[t] = mulMod(numerator.lo, modularInverse(denominator.lo, p), p)
    else:
      if numerator.lo != 0 or denominator.hi == 0:
        raise newException(ValueError, "The coefficient walk lost precision modulo " & $p)
      result[t] = mulMod(numerator.hi, modularInverse(denominator.hi, p), p)

proc powerProductCoefficients(a: seq[int64], alpha: int64, b: seq[int64], beta: int64,
                              wanted: seq[int64], p: int64): Table[int64, int64] =
  ## [x^t] A(x)^alpha B(x)^beta modulo a prime p for each t in `wanted` (all below 2p), for
  ## polynomials A, B (ascending integer coefficients, A(0) B(0) a unit mod p). F = A^alpha B^beta
  ## satisfies C F' = E F with C = A B and E = alpha A' B + beta A B', so
  ## C0 (t + 1) F(t + 1) = sum_{i >= 1} (E(i - 1) - C(i) (t + 1 - i)) F(t + 1 - i).
  ## Past t + 1 = p the walk runs modulo p^2 and divides the factor p out at the end.
  if wanted.len == 0: return
  let reach = max(wanted)
  if reach >= 2 * p: raise newException(ValueError, "A coefficient lies two singular steps away modulo " & $p)
  let crossing = reach >= p
  if crossing and p >= 2_147_483_648'i64:
    if p >= MaxWidePrime: raise newException(ValueError, "Passing t = p takes a prime below 2^43")
    return powerProductCoefficientsWide(a, alpha, b, beta, wanted, p)
  let q = if crossing: p * p else: p
  proc reduced(poly: seq[int64]): seq[int64] = poly.mapIt(liftMod(it, q))
  proc product(x, y: seq[int64]): seq[int64] =
    result = newSeq[int64](x.len + y.len - 1)
    for i, u in x:
      for j, v in y: result[i + j] = (result[i + j] + mulMod(u, v, q)) mod q
  proc derivative(x: seq[int64]): seq[int64] =
    result = newSeq[int64](max(1, x.len - 1))
    for i in 1 ..< x.len: result[i - 1] = mulMod(x[i], int64(i) mod q, q)
  let ar = reduced(a)
  let br = reduced(b)
  let c = product(ar, br)
  var e = newSeq[int64](c.len)
  let left = product(derivative(ar), br)
  let right = product(ar, derivative(br))
  for i in 0 ..< left.len: e[i] = (e[i] + mulMod(alpha mod q, left[i], q)) mod q
  for i in 0 ..< right.len: e[i] = (e[i] + mulMod(beta mod q, right[i], q)) mod q
  let order = c.len - 1
  if c[0] mod p == 0: raise newException(ValueError, "The polynomial power has a zero constant term modulo " & $p)
  var entries = newSeq[seq[seq[int64]]](order + 1)
  for i in 0 .. order: entries[i] = newSeq[seq[int64]](order + 1)
  for i in 1 .. order:
    let constant = (e[i - 1] + mulMod(c[i], liftMod(int64(i - 1), q), q)) mod q   # E - C (1 - i)
    entries[0][i - 1] = @[constant, (q - c[i]) mod q]
  for r in 1 ..< order: entries[r][r - 1] = @[c[0], c[0]]
  entries[order][order] = @[c[0], c[0]]
  var state = newSeq[int64](order + 1)
  state[0] = mulMod(powMod(ar[0], alpha, q), powMod(br[0], beta, q), q)
  state[order] = 1 mod q
  var position = 0'i64
  var targets = wanted
  targets.sort()
  for t in targets:
    if t in result: continue
    let gap = t - position
    if gap >= 4096 and q >= SqrtProductMinModulus:
      let m = sqrtMatrixProduct(entries, position, gap, q)
      var next = newSeq[int64](order + 1)
      for i in 0 .. order:
        for k in 0 .. order: next[i] = (next[i] + mulMod(m[i * (order + 1) + k], state[k], q)) mod q
      state = next
    else:
      for x in position ..< t:
        let xq = liftMod(x, q)
        var next = newSeq[int64](order + 1)
        for k in 0 ..< order:
          let coefficient = (entries[0][k][0] + mulMod(entries[0][k][1], xq, q)) mod q
          next[0] = (next[0] + mulMod(coefficient, state[k], q)) mod q
        let lead = mulMod(c[0], (xq + 1) mod q, q)
        for r in 1 ..< order: next[r] = mulMod(lead, state[r - 1], q)
        next[order] = mulMod(lead, state[order], q)
        state = next
    position = t
    let numerator = state[0]
    let denominator = state[order]
    if denominator mod p != 0:
      result[t] = mulMod(numerator mod p, modularInverse(denominator mod p, p), p)
    else:
      if not crossing or numerator mod p != 0 or (denominator div p) mod p == 0:
        raise newException(ValueError, "The coefficient walk lost precision modulo " & $p)
      result[t] = mulMod(numerator div p, modularInverse((denominator div p) mod p, p), p)

type CoefficientSource = proc(wanted: seq[int64], fromTop: bool): Table[int64, int64]

proc cartierApply(r: Laurent, total, offset: int64, source: CoefficientSource, q: int64): Laurent =
  ## Lambda_q(R(x) x^offset F(x)) for a polynomial F of degree `total`: the coefficients at
  ## multiples of q, with the exponents divided by q. F is read from whichever end is nearer:
  ## source(t, false) gives [x^t] F, source(t, true) gives [x^(total - t)] F.
  if r.c.len == 0: return r
  let lowExponent = r.low + offset
  let highExponent = r.low + int64(r.c.high) + offset + total
  let kLow = -floorDivision(-lowExponent, q)
  let kHigh = floorDivision(highExponent, q)
  var bottom, top: seq[int64]
  for k in kLow .. kHigh:
    for i, w in r.c:
      if w == 0: continue
      let t = k * q - (r.low + int64(i)) - offset
      if t < 0 or t > total: continue
      if 2 * t <= total: bottom.add(t) else: top.add(total - t)
  let low = source(bottom, false)
  let high = source(top, true)
  result.low = kLow
  result.c = newSeq[int64](int(max(0'i64, kHigh - kLow + 1)))
  for k in kLow .. kHigh:
    var sum = 0'i64
    for i, w in r.c:
      if w == 0: continue
      let t = k * q - (r.low + int64(i)) - offset
      if t < 0 or t > total: continue
      let coefficient = if 2 * t <= total: low[t] else: high[total - t]
      sum = (sum + mulMod(w, coefficient, q)) mod q
    result.c[int(k - kLow)] = sum

proc cartierProduct(r: Laurent, a: seq[int64], alpha: int64, b: seq[int64], beta: int64,
                    offset: int64, q: int64): Laurent =
  ## Lambda_q(R(x) x^offset A(x)^alpha B(x)^beta).
  let total = int64(a.len - 1) * alpha + int64(b.len - 1) * beta
  cartierApply(r, total, offset, proc(wanted: seq[int64], fromTop: bool): Table[int64, int64] =
    if fromTop: powerProductCoefficients(a.reversed(), alpha, b.reversed(), beta, wanted, q)
    else: powerProductCoefficients(a, alpha, b, beta, wanted, q), q)

proc geometricSumCoefficients(p3: seq[int64], lambda, m: int64, wanted: seq[int64], q: int64): Table[int64, int64] =
  ## [x^t] U(x) for t in `wanted` (all <= m < q), where U = x^m T_m(y) = (lambda^(m+1) P^(m+1) - x^(m+1)) / V
  ## with P = p3[0] + p3[1] x + p3[2] x^2 and V = lambda P - x. The coefficients c(t) of P^(m+1)
  ## and U(t) walk together: a (t+1) c(t+1) = b (J - t) c(t) + c (2J + 1 - t) c(t-1) with J = m + 1,
  ## and V0 U(t+1) = lambda^J c(t+1) - V1 U(t) - V2 U(t-1); the spike of x^J lies past t = m.
  if wanted.len == 0: return
  let a = liftMod(p3[0], q)
  let b = liftMod(p3[1], q)
  let c = liftMod(p3[2], q)
  let j = (m + 1) mod q
  let v0 = mulMod(lambda, a, q)
  let v1 = (mulMod(lambda, b, q) - 1 + q) mod q
  let v2 = mulMod(lambda, c, q)
  let lj = powMod(lambda, m + 1, q)
  let twoJ = (2 * j + 1) mod q
  proc neg(x: int64): int64 = (q - x mod q) mod q
  var entries = newSeq[seq[seq[int64]]](5)
  for i in 0 .. 4: entries[i] = newSeq[seq[int64]](5)
  let av1 = mulMod(a, v1, q)
  let av2 = mulMod(a, v2, q)
  let lead = mulMod(v0, a, q)
  entries[0][0] = @[neg(av1), neg(av1)]
  entries[0][1] = @[neg(av2), neg(av2)]
  entries[0][2] = @[mulMod(lj, mulMod(b, j, q), q), neg(mulMod(lj, b, q))]
  entries[0][3] = @[mulMod(lj, mulMod(c, twoJ, q), q), neg(mulMod(lj, c, q))]
  entries[1][0] = @[lead, lead]
  entries[2][2] = @[mulMod(v0, mulMod(b, j, q), q), neg(mulMod(v0, b, q))]
  entries[2][3] = @[mulMod(v0, mulMod(c, twoJ, q), q), neg(mulMod(v0, c, q))]
  entries[3][2] = @[lead, lead]
  entries[4][4] = @[lead, lead]
  let am = powMod(a, m, q)
  var state = @[mulMod(powMod(lambda, m, q), am, q), 0'i64, mulMod(am, a, q), 0'i64, 1'i64 mod q]
  var position = 0'i64
  var targets = wanted
  targets.sort()
  for t in targets:
    if t in result: continue
    let gap = t - position
    if gap >= 4096 and q >= SqrtProductMinModulus:
      let product = sqrtMatrixProduct(entries, position, gap, q)
      var next = newSeq[int64](5)
      for i in 0 .. 4:
        for k in 0 .. 4: next[i] = (next[i] + mulMod(product[i * 5 + k], state[k], q)) mod q
      state = next
    else:
      for x in position ..< t:
        var next = newSeq[int64](5)
        for i in 0 .. 4:
          for k in 0 .. 4:
            let poly = entries[i][k]
            if poly.len == 0: continue
            let value = (poly[0] + mulMod(poly[1], x mod q, q)) mod q
            next[i] = (next[i] + mulMod(value, state[k], q)) mod q
        state = next
    position = t
    result[t] = mulMod(state[0], modularInverse(state[4], q), q)

proc reciprocalCoefficient(v: seq[int64], index, q: int64): int64 =
  ## [x^index] 1 / (v0 + v1 x + v2 x^2) as a power series modulo q, by a 2x2 matrix power.
  if index < 0: return 0
  let inverse = modularInverse(v[0], q)
  var m = [(q - mulMod(v[1], inverse, q)) mod q, (q - mulMod(v[2], inverse, q)) mod q, 1'i64 mod q, 0'i64]
  var power = [1'i64 mod q, 0'i64, 0'i64, 1'i64 mod q]
  proc multiply(x, y: array[4, int64]): array[4, int64] =
    [(mulMod(x[0], y[0], q) + mulMod(x[1], y[2], q)) mod q, (mulMod(x[0], y[1], q) + mulMod(x[1], y[3], q)) mod q,
     (mulMod(x[2], y[0], q) + mulMod(x[3], y[2], q)) mod q, (mulMod(x[2], y[1], q) + mulMod(x[3], y[3], q)) mod q]
  var e = index
  while e > 0:
    if (e and 1) == 1:
      let next = multiply(power, m)
      power = next
    let squared = multiply(m, m)
    m = squared
    e = e shr 1
  mulMod(power[0], inverse, q)   # (sigma(index), sigma(index - 1)) = M^index (sigma(0), 0)

proc constantOf(r: Laurent): int64 =
  if r.low <= 0 and -r.low <= int64(r.c.high): r.c[int(-r.low)] else: 0

proc subtract(x, y: Laurent, q: int64): Laurent =
  if x.c.len == 0: return (y.low, y.c.mapIt((q - it) mod q))
  if y.c.len == 0: return x
  result.low = min(x.low, y.low)
  let high = max(x.low + int64(x.c.high), y.low + int64(y.c.high))
  result.c = newSeq[int64](int(high - result.low + 1))
  for i, v in x.c: result.c[int(x.low - result.low) + i] = v
  for i, v in y.c:
    let at = int(y.low - result.low) + i
    result.c[at] = (result.c[at] - v + q) mod q

proc add(x, y: Laurent, q: int64): Laurent =
  subtract(x, (y.low, y.c.mapIt((q - it) mod q)), q)

proc squareRootShape(laurent: seq[int64]): tuple[ok: bool, c: seq[int64]] =
  ## (a, b, c) with (a/x + b + c x)^2 equal to a centred five-term P, if there is one.
  if laurent.len != 5: return
  proc exactRoot(v: int64): int64 =
    if v <= 0: return -1
    var r = int64(sqrt(float64(v)))
    while r * r > v: dec r
    while (r + 1) * (r + 1) <= v: inc r
    if r * r == v: r else: -1
  let c = exactRoot(laurent[4])
  let a0 = exactRoot(laurent[0])
  if c <= 0 or a0 <= 0 or laurent[3] mod (2 * c) != 0: return
  let b = laurent[3] div (2 * c)
  for a in [a0, -a0]:
    if 2 * a * b == laurent[1] and b * b + 2 * a * c == laurent[2]: return (true, @[a, b, c])

proc constantTermMod(form: ConstantTermForm, relative: BigInt, q: int64): int64 =
  ## a(relative) modulo a prime q, one Cartier step per base-q digit.
  if q > MaxHypergeometricPrime:
    raise newException(ValueError, "The constant-term route takes a prime up to " & $MaxHypergeometricPrime)
  if relative < big(form.shift): raise newException(ValueError, "The index is an initial value")
  if not form.inhomogeneous and form.left == 2:
    # P = P3^2 for a three-term P3: CT[P^n Q] = CT[P3^(2n) Q], whose coefficient walks never
    # pass t = p, so no arithmetic modulo p^2 is needed.
    let half = squareRootShape(form.laurent)
    if half.ok:
      var reduced = form
      reduced.laurent = half.c
      reduced.left = 1
      reduced.shift = 0
      result = constantTermMod(reduced, (relative - big(form.shift)) * big(2), q)
      residueRoute = "the constant-term form " & (if form.shift == 0: "a(n)" else: "a(n + " & $form.shift & ")") &
        " = CT[P^n Q] with P = (" & $half.c[0] & "/x + " & $half.c[1] & " + " & $half.c[2] & " x)^2 (fitted on " &
        $ConstantTermTerms & " terms of the relation and checked modulo three primes, not proved), " &
        "so it is CT[P3^(2n) Q], reduced one base-" & $q & " digit at a time by the Cartier operator"
      return
  let laurent = form.laurent
  let left = int64(form.left)
  if liftMod(laurent[0], q) == 0 or liftMod(laurent[^1], q) == 0:
    raise newException(ValueError, "P loses an end coefficient modulo " & $q)
  var start: Laurent = (-int64(ConstantTermSupport), form.weights.mapIt(ratMod(it, q)))
  let n = relative - big(form.shift)
  let mu = if form.inhomogeneous: ratMod(form.mu, q) else: 0'i64
  if mu == 0:
    var r = start
    var rest = n
    while not rest.isZero:
      let (quotient, j) = divSmall(rest, q)
      r = cartierProduct(r, laurent, j, @[1'i64], 0, -left * j, q)
      rest = quotient
    result = constantOf(r)
  else:
    # b(n) = mu^n (b(0) - g(0) + S(n)), S(n) = CT[Q sum_{m <= n} y^m] with y = P / mu.
    # Digit sums: T_d(y) = sum_{m <= d} y^m = 1 + y T_(d-1)(y), and T_(q-1)(y) = (y^q - 1) / V
    # = (lambda P(x^q) - 1) / V mod q, a polynomial read from the power series 1 / V.
    if n.isZero: return form.first.modSmall(q)
    if laurent.len != 3: raise newException(ValueError, "The inhomogeneous constant-term form takes a three-term P")
    let lambda = modularInverse(mu, q)
    let v = @[mulMod(liftMod(laurent[0], q), lambda, q), (mulMod(liftMod(laurent[1], q), lambda, q) - 1 + q) mod q,
              mulMod(liftMod(laurent[2], q), lambda, q)]           # x V(x) = lambda P~ - x
    if v[0] == 0 or v[2] == 0: raise newException(ValueError, "y - 1 loses an end coefficient modulo " & $q)
    let h = v   # y^q - 1 = lambda P(x^q) - 1 = x^-q (v0 + v1 x^q + v2 x^2q) mod q
    proc full(r: Laurent): Laurent =
      ## Lambda(R T_(q-1)(y)), T_(q-1)(y) = x^-(q-1) H(x) / V~(x) with H = sum h_i x^(i q).
      cartierApply(r, 2 * q - 2, -(q - 1), proc(wanted: seq[int64], fromTop: bool): Table[int64, int64] =
        for w in wanted:
          let t = if fromTop: 2 * q - 2 - w else: w
          var total = 0'i64
          for i in 0 .. 2: total = (total + mulMod(h[i], reciprocalCoefficient(v, t - int64(i) * q, q), q)) mod q
          result[w] = total, q)
    proc partial(r: Laurent, m: int64): Laurent =
      ## Lambda(R T_m(y)) with T_m(y) = x^-m U(x), U from geometricSumCoefficients.
      let reversedP = @[laurent[2], laurent[1], laurent[0]]
      cartierApply(r, 2 * m, -m, proc(wanted: seq[int64], fromTop: bool): Table[int64, int64] =
        if fromTop: geometricSumCoefficients(reversedP, lambda, m, wanted, q)
        else: geometricSumCoefficients(laurent, lambda, m, wanted, q), q)
    proc times(r: Laurent): Laurent =
      ## R y = R lambda P.
      if r.c.len == 0: return r
      result.low = r.low - left
      result.c = newSeq[int64](r.c.len + 2)
      for i, w in r.c:
        for k in 0 .. 2:
          result.c[i + k] = (result.c[i + k] + mulMod(w, mulMod(liftMod(laurent[k], q), lambda, q), q)) mod q
    proc picked(r: Laurent): Laurent = cartierApply(r, 0, 0, proc(wanted: seq[int64], fromTop: bool): Table[int64, int64] =
      for w in wanted: result[w] = 1'i64 mod q, q)
    var high = start                                   # the sum up to the current prefix
    var low: Laurent = (0'i64, newSeq[int64]())        # the sum up to the prefix minus one
    var rest = n
    while not rest.isZero:
      let (quotient, digit) = divSmall(rest, q)
      let whole = full(add(high, low, q))
      var next = picked(high)
      if digit >= 1: next = add(next, partial(add(times(high), low, q), digit - 1), q)
      low = subtract(whole, next, q)
      high = next
      rest = quotient
    let sum = constantOf(high)
    let initial = (form.first.modSmall(q) - constantOf(start) + q) mod q
    let power = powMod(mu, n.modSmall(q - 1), q)
    result = mulMod(power, (initial + sum) mod q, q)
  var terms: seq[string]
  for i, c in laurent:
    if c == 0: continue
    let exponent = i - form.left
    terms.add($c & (if exponent == 0: "" elif exponent == 1: " x" else: " x^" & $exponent))
  let shape = "CT[(" & terms.join(" + ") & ")^n Q(x)]"
  let b = if form.shift == 0: "a(n)" else: "a(n + " & $form.shift & ")"
  let lhs = if form.inhomogeneous: b & " - (" & $form.mu.num & "/" & $form.mu.den & ") " &
              (if form.shift == 0: "a(n - 1)" elif form.shift == 1: "a(n)" else: "a(n + " & $(form.shift - 1) & ")")
            else: b
  residueRoute = "the constant-term form " & lhs & " = " & shape & " (fitted on " & $ConstantTermTerms &
    " terms of the relation and checked modulo three primes, not proved), reduced one base-" & $q &
    " digit at a time by the Cartier operator, since P(x)^p = P(x^p) mod p"

# Modulo a prime power q = p^k the Cartier step needs P^(p^k) = P(x^p)^(p^(k-1)) mod p^k
# (Rowland and Yassawi). With h = p^(k-1) and n = m + q t (m < q), a(n) = CT[S P^(h t)] for
# S = Lambda_p(Q P^m), and each base-p digit j of t maps S to Lambda_p(S P^(h j)). The states
# have span O(h), so the products are NTT convolutions modulo q.

const MaxConstantTermPrimePower = 2_097_152'i64

proc laurentMultiply(x, y: Laurent, q: int64): Laurent =
  if x.c.len == 0 or y.c.len == 0: return (0'i64, newSeq[int64]())
  (x.low + y.low, convolveMod(x.c, y.c, q))

proc laurentPower(base: Laurent, e: int64, q: int64): Laurent =
  result = (0'i64, @[1'i64 mod q])
  var b = base
  var k = e
  while k > 0:
    if (k and 1) == 1: result = laurentMultiply(result, b, q)
    k = k shr 1
    if k > 0: b = laurentMultiply(b, b, q)

proc decimate(r: Laurent, p: int64): Laurent =
  ## The coefficients at exponents divisible by p, with the exponents divided by p.
  if r.c.len == 0: return r
  let kLow = -floorDivision(-r.low, p)
  let kHigh = floorDivision(r.low + int64(r.c.high), p)
  result.low = kLow
  result.c = newSeq[int64](int(max(0'i64, kHigh - kLow + 1)))
  for k in kLow .. kHigh: result.c[int(k - kLow)] = r.c[int(k * p - r.low)]

proc constantTermModPrimePower(form: ConstantTermForm, relative: BigInt, p, q: int64): int64 =
  if form.inhomogeneous:
    raise newException(ValueError, "The inhomogeneous constant-term form is evaluated modulo primes only")
  if q > MaxConstantTermPrimePower:
    raise newException(ValueError, "The constant-term route takes a prime power up to " & $MaxConstantTermPrimePower)
  if relative < big(form.shift): raise newException(ValueError, "The index is an initial value")
  let n = relative - big(form.shift)
  let base: Laurent = (-int64(form.left), form.laurent.mapIt(liftMod(it, q)))
  let start: Laurent = (-int64(ConstantTermSupport), form.weights.mapIt(ratMod(it, q)))
  let (outer, m) = divSmall(n, q)
  var state = decimate(laurentMultiply(start, laurentPower(base, m, q), q), p)
  var t = outer
  if not t.isZero:
    let stepPower = laurentPower(base, q div p, q)
    var powers = initTable[int64, Laurent]()
    while not t.isZero:
      let (quotient, digit) = divSmall(t, p)
      if digit notin powers: powers[digit] = laurentPower(stepPower, digit, q)
      state = decimate(laurentMultiply(state, powers[digit], q), p)
      t = quotient
  result = constantOf(state)
  residueRoute = "the constant-term form " & (if form.shift == 0: "a(n)" else: "a(n + " & $form.shift & ")") &
    " = CT[P^n Q] (fitted on " & $ConstantTermTerms &
    " terms of the relation and checked modulo three primes, not proved), reduced modulo " & $q &
    " by P(x)^(p^k) = P(x^p)^(p^(k-1)) mod p^k, one base-" & $p & " digit at a time"

proc constantTermSumPrimePower(form: ConstantTermForm, relative: BigInt, p: int64, alpha: int, q: int64): int64 =
  ## b(n) - mu b(n - 1) = CT[P^n Q] modulo q = p^alpha when P = a/x + b + c x has b^2 = 4ac (Fine's
  ## numbers): then P = (a/x)(1 + t x)^2 with t = b / 2a, [x^-j] P^m = a^m t^(m - j) C(2m, m - j), and
  ## b(n) = mu^n (b(0) - Q_0 + S(n)) with S(n) = sum_j Q_j t^-j sum_{m <= n} (b / 2 mu)^m C(2m, m - j).
  ## Each inner sum is a binomial sum for the digit DP; m <= n enters as the factor C(n - m, 0).
  if not form.inhomogeneous or form.laurent.len != 3 or form.left != 1:
    raise newException(ValueError, "not an inhomogeneous constant-term form with a three-term P")
  let (a, b, c) = (form.laurent[0], form.laurent[1], form.laurent[2])
  if b == 0 or b * b != 4 * a * c:
    raise newException(ValueError, "Modulo a prime power the inhomogeneous form needs P = (a/x)(1 + t x)^2")
  if q > MaxBinomialSumModulus:
    raise newException(ValueError, "The prefix binomial sums take a prime power up to " & $MaxBinomialSumModulus)
  if b mod p == 0 or (2 * a) mod p == 0 or form.mu.num mod p == 0 or form.mu.den mod p == 0:
    raise newException(ValueError, "t = b / 2a or mu is not a unit modulo " & $p)
  if relative < big(form.shift): raise newException(ValueError, "The index is an initial value")
  let shifted = relative - big(form.shift)
  if not shifted.fitsInt64() or shifted > big(2_000_000_000_000_000_000'i64):
    raise newException(ValueError, "index is out of range for the prefix binomial sums")
  let n = shifted.toInt64()
  if n == 0: return form.first.modSmall(q)
  let mu = ratMod(form.mu, q)
  let t = ratMod((b, 2 * a), q)
  let tInverse = modularInverse(t, q)
  var weight: Rational = (b * form.mu.den, 2 * form.mu.num)    # b / (2 mu)
  if weight.den < 0: weight = (-weight.num, -weight.den)
  var total = 0'i64
  for j in -int64(ConstantTermSupport) .. int64(ConstantTermSupport):
    let qj = ratMod(form.weights[int(j + ConstantTermSupport)], q)
    if qj == 0: continue
    let term = BinomialTerm(factors: @[((0'i64, 2'i64, 0'i64), (0'i64, 1'i64, -j), 1),
                                       ((1'i64, -1'i64, 0'i64), (0'i64, 0'i64, 0'i64), 1)],
                            weight: weight, sign: 1)
    let scale = mulMod(qj, powMod(if j >= 0: tInverse else: t, abs(j), q), q)
    total = (total + mulMod(scale, binomialTermMod(term, n, p, alpha, q), q)) mod q
  let initial = liftMod(form.first.modSmall(q) - ratMod(form.weights[ConstantTermSupport], q), q)
  result = mulMod(powMod(mu, n, q), (initial + total) mod q, q)
  residueRoute = "the inhomogeneous constant-term form b(n) - mu b(n - 1) = CT[P^n Q] (fitted on " &
    $ConstantTermTerms & " terms of the relation and checked modulo three primes, not proved), with " &
    "P = (a/x)(1 + t x)^2 so that each CT[P^m x^j] is one binomial, summed over m <= n modulo " & $q &
    " by Granville's theorem in a digit DP"

# Any constant-term form modulo q = p^k, for every P of the fit and every mu. Denominators of Q
# divisible by p are cleared by D = p^d: the work runs modulo p^(k + d) and the result is divided
# by D. Write c(n) = CT[P^n Q]; Y^(p^k) = Y(x^p)^(p^(k-1)) mod p^k for integer coefficients.
# - mu a p-adic unit: b(n) = mu^n (b(0) - Q_0 + S(n)), S(n) = sum_{m <= n} CT[Q Y^m] with Y = P / mu.
#   With h = q / p, Z = Y^h and m = r + q t, CT[Q Y^m] = CT[Lambda_p(Q Y^r) Z^t]. Summing over
#   m <= N = R + q T keeps two states: A for the full blocks t < T and B for t = T. Since
#   Z^p = Z(x^p) mod q, a base-p digit j of T maps them to A' = Lambda_p(A H_p) and
#   B' = Lambda_p(A H_j + B Z^j), with H_j = sum_{i < j} Z^i. The sum is CT[B] at the end.
# - p divides the denominator of mu (Fine's numbers modulo 2^k): nu = 1 / mu is divisible by p, and
#   b(n) = -sum_{j=1}^{L} nu^j c(n + j) + nu^L b(n + L). Once v_p(nu^L) >= k this is the single
#   homogeneous constant term -CT[P^(n+1) Q sum_{j=1}^{L} nu^j P^(j-1)] modulo p^k.
# - p divides the numerator of mu: b(n) = CT[P^(n-L+1) Q sum_{j<L} mu^j P^(L-1-j)] the same way.

proc valuationAt(x, p: int64): int =
  var y = abs(x)
  while y != 0 and y mod p == 0:
    y = y div p
    inc result

proc scaledResidue(r: Rational, d: int, p, q: int64): int64 =
  ## p^d r modulo q = p^k, for v_p(r) >= -d.
  if r.num == 0: return 0
  var den = r.den
  var e = d
  while den mod p == 0:
    if e == 0: raise newException(ValueError, "The coefficient " & $r.num & "/" & $r.den & " is not " & $p & "-integral")
    den = den div p
    dec e
  result = mulMod(liftMod(r.num, q), modularInverse(liftMod(den, q), q), q)
  if e > 0: result = mulMod(result, powMod(p mod q, int64(e), q), q)

proc scaleLaurent(r: Laurent, c: int64, q: int64): Laurent =
  (r.low, r.c.mapIt(mulMod(it, c, q)))

proc ctPowerPrimePower(start, base: Laurent, n: BigInt, p, q: int64): int64 =
  ## CT[start base^n] modulo q = p^k: Lambda_p(start base^m) for m = n mod q, then one product
  ## with base^(h j) and one Cartier step per base-p digit j of n div q.
  let (outer, m) = divSmall(n, q)
  var state = decimate(laurentMultiply(start, laurentPower(base, m, q), q), p)
  var t = outer
  if not t.isZero:
    let stepPower = laurentPower(base, q div p, q)
    var powers = initTable[int64, Laurent]()
    while not t.isZero:
      let (quotient, digit) = divSmall(t, p)
      if digit notin powers: powers[digit] = laurentPower(stepPower, digit, q)
      state = decimate(laurentMultiply(state, powers[digit], q), p)
      t = quotient
  constantOf(state)

proc geometricLaurent(z: Laurent, count: int64, q: int64): Laurent =
  ## sum_{i < count} z^i by doubling: G(2k) = G(k) + z^k G(k), G(k + 1) = 1 + z G(k).
  let one: Laurent = (0'i64, @[1'i64 mod q])
  result = (0'i64, newSeq[int64]())
  if count <= 0: return
  var zk = one
  var k = 0'i64
  for bit in countdown(62, 0):
    if k > 0:
      result = add(result, laurentMultiply(zk, result, q), q)
      if bit > 0: zk = laurentMultiply(zk, zk, q)
      k *= 2
    if ((count shr bit) and 1) == 1:
      result = add(one, laurentMultiply(z, result, q), q)
      if bit > 0: zk = laurentMultiply(zk, z, q)
      inc k

proc ctPowerSumPrimePower(start, base: Laurent, n: BigInt, p, q: int64): int64 =
  ## sum_{m <= n} CT[start base^m] modulo q = p^k, with the two states A and B above.
  let (outer, r) = divSmall(n, q)
  var b = decimate(laurentMultiply(start, geometricLaurent(base, r + 1, q), q), p)
  if outer.isZero: return constantOf(b)
  var a = decimate(laurentMultiply(start, geometricLaurent(base, q, q), q), p)
  let z = laurentPower(base, q div p, q)
  let full = geometricLaurent(z, p, q)
  var powers = initTable[int64, Laurent]()
  var partials = initTable[int64, Laurent]()
  var t = outer
  while not t.isZero:
    let (quotient, digit) = divSmall(t, p)
    if digit notin powers: powers[digit] = laurentPower(z, digit, q)
    var next = laurentMultiply(b, powers[digit], q)
    if digit > 0:
      if digit notin partials: partials[digit] = geometricLaurent(z, digit, q)
      next = add(next, laurentMultiply(a, partials[digit], q), q)
    b = decimate(next, p)
    if not quotient.isZero: a = decimate(laurentMultiply(a, full, q), p)
    t = quotient
  constantOf(b)

proc constantTermGeneralPrimePower(form: ConstantTermForm, relative: BigInt, p: int64, alpha: int,
                                   q: int64): int64 =
  ## Any fitted constant-term form modulo q = p^alpha (also alpha = 1), by the three cases above.
  if relative < big(form.shift): raise newException(ValueError, "The index is an initial value")
  let n = relative - big(form.shift)
  var d = 0
  for w in form.weights:
    if w.num != 0: d = max(d, valuationAt(w.den, p) - valuationAt(w.num, p))
  let e = if form.inhomogeneous: valuationAt(form.mu.num, p) - valuationAt(form.mu.den, p) else: 0
  var scale = 1'i64
  for _ in 1 .. d:
    if scale > MaxConstantTermPrimePower: break
    scale *= p
  if scale > MaxConstantTermPrimePower or q > MaxConstantTermPrimePower div scale:
    raise newException(ValueError, "The constant-term route works modulo a prime power up to " &
      $MaxConstantTermPrimePower & (if d > 0: " (here p^(k + " & $d & ") for the denominators of Q)" else: ""))
  let work = q * scale
  let base: Laurent = (-int64(form.left), form.laurent.mapIt(liftMod(it, work)))
  let start: Laurent = (-int64(ConstantTermSupport), form.weights.mapIt(scaledResidue(it, d, p, work)))
  var x: int64
  var how: string
  if not form.inhomogeneous:
    x = ctPowerPrimePower(start, base, n, p, work)
    how = "reduced by P(x)^(p^k) = P(x^p)^(p^(k-1)) mod p^k, one base-" & $p & " digit at a time"
  elif n.isZero:
    return floorMod(form.first, big(q)).toInt64()
  elif e == 0:
    let muResidue = scaledResidue(form.mu, 0, p, work)
    let lambda = modularInverse(muResidue, work)
    let y: Laurent = scaleLaurent(base, lambda, work)
    let s = ctPowerSumPrimePower(start, y, n, p, work)
    let first = mulMod(floorMod(form.first, big(work)).toInt64(), scale mod work, work)
    let initial = liftMod(first - constantOf(start), work)
    let phi = (work div p) * (p - 1)
    x = mulMod(powMod(muResidue, n.modSmall(phi), work), (initial + s) mod work, work)
    how = "b(n) = mu^n (b(0) - Q_0 + sum_{m <= n} CT[Q (P / mu)^m]), the sum kept as two Cartier states " &
      "(full blocks and the last one) per base-" & $p & " digit, since P(x)^(p^k) = P(x^p)^(p^(k-1)) mod p^k"
  else:
    let length = (alpha + abs(e) - 1) div abs(e)   # v_p of the dropped factor reaches alpha
    let ratio: Rational = if e < 0: (form.mu.den, form.mu.num) else: form.mu
    let r = scaledResidue(ratio, 0, p, work)
    var w: Laurent = (0'i64, newSeq[int64]())
    var power: Laurent = (0'i64, @[1'i64 mod work])
    if e < 0:
      var coefficient = r
      for _ in 1 .. length:
        w = add(w, scaleLaurent(power, coefficient, work), work)
        power = laurentMultiply(power, base, work)
        coefficient = mulMod(coefficient, r, work)
      x = (work - ctPowerPrimePower(laurentMultiply(start, w, work), base, n + big(1), p, work)) mod work
      how = "b(n) = -sum_{j=1}^{" & $length & "} (1/mu)^j CT[P^(n+j) Q] mod p^k, since " & $p &
        " divides 1/mu, as one constant term reduced one base-" & $p & " digit at a time"
    else:
      if n < big(length): raise newException(ValueError, "The index is among the first terms of the tail")
      for i in 0 ..< length:
        w = add(w, scaleLaurent(power, powMod(r, int64(length - 1 - i), work), work), work)
        power = laurentMultiply(power, base, work)
      x = ctPowerPrimePower(laurentMultiply(start, w, work), base, n - big(length - 1), p, work)
      how = "b(n) = sum_{j<" & $length & "} mu^j CT[P^(n-j) Q] mod p^k, since " & $p &
        " divides mu, as one constant term reduced one base-" & $p & " digit at a time"
  if x mod scale != 0:
    raise newException(ValueError, "The scaled constant term is not divisible by " & $scale &
      ", so the fitted form fails modulo " & $q)
  result = (x div scale) mod q
  let b = if form.shift == 0: "a(n)" else: "a(n + " & $form.shift & ")"
  residueRoute = "the constant-term form " & b &
    (if form.inhomogeneous: " - (" & $form.mu.num & "/" & $form.mu.den & ") a(n - 1 + " & $form.shift & ")" else: "") &
    " = CT[P^n Q] (fitted on " & $ConstantTermTerms & " terms of the relation and checked modulo three primes, " &
    "not proved) modulo " & $q & (if d > 0: " (worked modulo " & $work & " for the denominators of Q)" else: "") &
    ": " & how

proc constantTermPrimePowerRoute(form: ConstantTermForm, relative: BigInt, p: int64, alpha: int, q: int64): int64 =
  ## The constant-term routes modulo q = p^alpha: the prefix binomial sums when P = (a/x)(1 + t x)^2,
  ## the plain Cartier states for a homogeneous form, then the general route.
  var failure = ""
  try:
    if form.inhomogeneous: return constantTermSumPrimePower(form, relative, p, alpha, q)
    if alpha >= 2: return constantTermModPrimePower(form, relative, p, q)
  except ValueError as error:
    failure = error.msg & ". "
  try:
    return constantTermGeneralPrimePower(form, relative, p, alpha, q)
  except ValueError as error:
    raise newException(ValueError, failure & error.msg)

proc recurrenceTermMod(prefix: proc(count: int): seq[int64], window: int64, relative: BigInt,
                       q: int64): int64 =
  ## A sequence modulo a small prime is often linear recurrent (Touchard's congruence for
  ## Bell numbers is one). Berlekamp-Massey finds the shortest recurrence on computed
  ## terms; it is used only when it holds on at least a quarter of them beyond the 2L
  ## that determine it. Kitamasa then jumps to the index.
  ## A composite modulus has no field for Berlekamp-Massey; there the residues are tested
  ## for eventual periodicity instead (pre-period plus two full periods and a margin).
  let prime = isPrimeModulus(q)
  var count = 1024
  while true:
    let size = int(min(int64(count), window + 1))
    let terms = prefix(size)
    if prime:
      let recurrence = berlekampMassey(terms, q)
      if recurrence.len <= MaxFoundRecurrence and size - 2 * recurrence.len >= max(64, size div 4):
        return linearTermMod(recurrence, terms[0 ..< recurrence.len], relative, q)
    else:
      for period in 1 .. size div 3:
        var lastMismatch = -1
        for i in countdown(size - 1 - period, 0):
          if terms[i + period] != terms[i]:
            lastMismatch = i
            break
        let start = lastMismatch + 1
        if start + 2 * period + max(64, size div 8) <= size:
          if relative < big(size): return terms[relative.toInt64()]
          let folded = (relative - big(start)).divSmall(int64(period)).remainder
          return terms[start + int(folded)]
    if int64(size) > window or count >= 16_384: break
    count *= 4
  raise newException(ValueError, (if prime: "No linear recurrence" else: "No eventual period") & " modulo " &
    $q & " fits the computed terms")

const ResidueTierLimit = 4_611_686_018_427_387_903'i64
  ## 2^62 - 1. Residue code adds two residues and multiplies with mulMod, so any modulus
  ## up to here is exact in int64. Larger moduli move to the exact or big integer tiers.

proc residueTerm(d: Direction, root: Track, index, relative: BigInt, q: int64): int64 =
  ## The far term modulo q <= ResidueTierLimit.
  let form = d.form
  if d.isDivisorSum and root.offset <= 1: return divisorSumTerm(d, index).modSmall(q)
  let recurrenceTier = q <= MaxRecurrenceModulus
  if d.path.len > 0:
    let limit = pathLimit(d, q)
    if relative <= big(limit): return pathTermMod(d, root, relative.toInt64(), q)
    if recurrenceTier:
      return recurrenceTermMod(proc(count: int): seq[int64] = pathValuesMod(d, root, int64(count - 1), q),
                               limit, relative, q)
    raise newException(ValueError, "Indices through this transform path are limited to " & $limit &
      " (a prime modulus up to " & $MaxRecurrenceModulus & " also tries a linear recurrence of the residues)")
  if form.multiplicativeModel != mmNone and not form.available:
    if not index.fitsInt64(): raise newException(ValueError, "index is too large for a multiplicative far term")
    return multiplicativeTerm(form, root, index.toInt64()).value.modSmall(q)
  case form.kind
  of hPolynomial:
    let x = index.modSmall(q)
    var value = 0'i64
    for j in countdown(form.polynomial.high, 0): value = (mulMod(value, x, q) + ratMod(form.polynomial[j], q)) mod q
    value
  of hLinearRecurrence:
    let coefficients = form.linear.mapIt(ratMod(it, q))
    let initial = root.exact[0 ..< form.linear.len].mapIt(it.modSmall(q))
    linearTermMod(coefficients, initial, relative, q)
  of hSelfReferential:
    if relative <= big(MaxSelfIndex) or (relative.fitsInt64() and fastSelfModulus(q, relative.toInt64())):
      return selfTermMod(form, root, relative.toInt64(), q)
    if recurrenceTier:
      return recurrenceTermMod(proc(count: int): seq[int64] = selfValuesMod(form, root, int64(count - 1), q),
                               MaxSelfIndex, relative, q)
    raise newException(ValueError, "A self-referential relation is iterated up to index " & $MaxSelfIndex &
      " (a prime modulus up to " & $MaxRecurrenceModulus & " also tries a linear recurrence of the residues)")
  of hBinaryRecursive:
    if not index.fitsInt64(): raise newException(ValueError, "index is too large for a binary relation")
    binaryTermMod(form, root, index.toInt64(), q)
  else:
    let power = primePowerOf(q)
    let hypergeometric = form.blocks.len == 2 and power.ok and
      (if power.alpha == 1: q <= MaxHypergeometricPrime else: power.p <= MaxPrimePowerBase)
    if relative.fitsInt64() and relative.toInt64() > SqrtProductFrom and q >= SqrtProductMinModulus and
        q < FastEulerModulus and isProbablePrime(q):
      try:
        return holonomicSqrtMod(form.blocks, root, relative.toInt64(), q)
      except ValueError:
        discard   # iteration, the p-adic route or the residue recurrence may still work
    var iterationFailure = ""
    if relative.fitsInt64() and relative.toInt64() <= MaxTargetIterations:
      try:
        return holonomicTermMod(form.blocks, root, relative.toInt64(), q)
      except ValueError as error:
        # The leading coefficient can vanish modulo a prime below the index; the p-adic,
        # recurrence, Lucas and constant-term routes below cope with that.
        iterationFailure = error.msg
    var hypergeometricFailure = ""
    if hypergeometric:
      try:
        return hypergeometricTermMod(form.blocks, root, index, q)
      except ValueError as error:
        hypergeometricFailure = error.msg
    if power.ok and relative.fitsInt64():
      # A leading coefficient with no root modulo p: rows of p steps, any index.
      try:
        return periodTermMod(form.blocks, root, relative.toInt64(), power.p, power.alpha, q)
      except ValueError as error:
        if iterationFailure.len == 0: iterationFailure = error.msg
    if hypergeometricFailure.len > 0 and not recurrenceTier:
      raise newException(ValueError, hypergeometricFailure)
    if recurrenceTier:
      try:
        return recurrenceTermMod(proc(count: int): seq[int64] =
                                   holonomicValuesMod(form.blocks, root, int64(count - 1), q),
                                 MaxTargetIterations, relative, q)
      except ValueError:
        discard
    if isPrimeModulus(q):
      if lucasProperty(form.blocks, root): return lucasTermMod(form.blocks, root, index, q)
      let constantTerm = constantTermForm(form.blocks, root)
      var sumFailure = ""
      if constantTerm.ok:
        try:
          return constantTermMod(constantTerm, relative, q)
        except ValueError as error:
          sumFailure = error.msg & ". "
        if q <= MaxConstantTermPrimePower:
          # mu with q in its denominator, or Q with q in its denominators: the tails above.
          try:
            return constantTermGeneralPrimePower(constantTerm, relative, q, 1, q)
          except ValueError as error:
            sumFailure &= error.msg & ". "
      if q <= MaxBinomialSumModulus:
        let sum = binomialSumForm(form.blocks, root)
        if sum.ok:
          try:
            return binomialSumMod(sum, index, q, 1, q)
          except ValueError as error:
            sumFailure &= error.msg & ". "
      raise newException(ValueError, (if iterationFailure.len > 0: iterationFailure & ". " else: "") & sumFailure &
        "Past the iteration window and the square root product, this relation has no route modulo " & $q &
        ": it is not of order one, its leading coefficient has a root modulo " & $q & ", its exact terms " &
        "fail the Lucas property a(n p + j) = a(n) a(j) mod small primes (or the offset is not 0), no " &
        "constant-term form a(n + s) = CT[P^n Q] or a(n + s) - mu a(n + s - 1) = CT[P^n Q] fits (P with " &
        "three to five terms, Q on [x^-4, x^4], s <= 2), and no binomial sum of the menu fits" &
        (if q > MaxBinomialSumModulus: " (tried modulo primes up to 2^22 only)" else: ""))
    if power.ok and power.alpha >= 2 and form.blocks.len > 2:
      var failure = ""
      let sum = if q <= MaxBinomialSumModulus: binomialSumForm(form.blocks, root) else: BinomialSumForm()
      let constantTerm = constantTermForm(form.blocks, root)
      # Past 2^16 the digit DP is usually the faster one; either falls back to the other.
      for route in (if q > 65_536: [1, 0] else: [0, 1]):
        if route == 0 and constantTerm.ok:
          try:
            return constantTermPrimePowerRoute(constantTerm, relative, power.p, power.alpha, q)
          except ValueError as error:
            failure &= error.msg & ". "
        if route == 1 and sum.ok:
          try:
            return binomialSumMod(sum, index, power.p, power.alpha, q)
          except ValueError as error:
            failure &= error.msg & ". "
      raise newException(ValueError, (if iterationFailure.len > 0: iterationFailure & ". " else: "") & failure &
        "Modulo the prime power " & $q & " this relation of order two or more needs a constant-term form " &
        "a(n + s) = CT[P^n Q] or a(n + s) - mu a(n + s - 1) = CT[P^n Q] (up to 2^21) or a binomial sum of " &
        "the menu (up to 2^22), and none fits")
    if iterationFailure.len > 0: raise newException(ValueError, iterationFailure)
    raise newException(ValueError, "Holonomic iteration modulo a number is limited to " & $MaxTargetIterations &
      " steps; past that a prime modulus from 2^20 to 2^31 uses the square root product (index below about" &
      " the modulus), and an order-one relation modulo a prime up to " & $MaxHypergeometricPrime & " reaches any index")

proc exactTerm(d: Direction, root: Track, index, relative: BigInt): tuple[value: BigInt, via: string] =
  ## The far term as an exact integer.
  let form = d.form
  if d.isDivisorSum and root.offset <= 1:
    return (divisorSumTerm(d, index), "the divisor sum of the Möbius-inverted polynomial over the divisors of the index")
  if d.path.len > 0:
    if not relative.fitsInt64(): raise newException(ValueError, "index is too large for a transformed direction")
    let steps = relative.toInt64()
    let limit = pathLimit(d, 2_147_483_647)   # the primes used below are under 2^31
    if steps > limit:
      raise newException(ValueError, "Indices through this transform path are limited to " & $limit)
    result.value = multimodularExact(proc(p: int64): int64 = pathTermMod(d, root, steps, p),
                                     "the transformed direction has no residue modulo the primes tried")
    result.via = "the transformed direction modulo primes below 2^31, joined by CRT until one more prime agrees"
    return
  if form.kind == hMultiplicative:
    if not index.fitsInt64(): raise newException(ValueError, "index is too large for a multiplicative far term")
    let (value, factors) = multiplicativeTerm(form, root, index.toInt64())
    return (value, "multiplicative over prime powers: " & factors)
  case form.kind
  of hPolynomial:
    result.value = polynomialAt(form.polynomial, index)
  of hSelfReferential:
    if not relative.fitsInt64(): raise newException(ValueError, "index is too large for a self-referential relation")
    result.value = selfTermExact(form, root, relative.toInt64())
    result.via = "self_referential_relation modulo primes below 2^31, joined by CRT until one more prime agrees"
  of hBinaryRecursive:
    if not index.fitsInt64(): raise newException(ValueError, "index is too large for a binary relation")
    result.value = binaryTermExact(form, root, index.toInt64())
  else:
    if not relative.fitsInt64() or relative.toInt64() > MaxExactIndex:
      raise newException(ValueError, "Exact evaluation is limited to " & $MaxExactIndex & " steps. Pass a modulus.")
    let blocks = if form.kind == hLinearRecurrence: linearBlocks(form) else: form.blocks
    result.value = holonomicTermExact(blocks, root, relative.toInt64())
    if relative.toInt64() - root.size >= SequentialExactSteps:
      result.via = $form.kind & " by binary splitting of the step matrices"

proc evaluateTarget*(ex: Explorer, root: Track, target: JsonNode): JsonNode =
  ## A far term exactly, or modulo any modulus. Tiers: residues for a modulus up to 2^62;
  ## above that the exact value reduced, Kitamasa on big integers for a linear recurrence,
  ## or CRT over the prime power parts of the modulus.
  if target.kind != JObject: raise newException(ValueError, "target must be an object")
  let index = toBig(target{"index"}, "target index")
  let relative = index - big(root.offset)
  if relative.isNegative: raise newException(ValueError, "target index precedes the first term")
  let modulusNode = target{"modulus"}
  let hasModulus = not modulusNode.isNil and modulusNode.kind != JNull
  let modulus = if hasModulus: toBig(modulusNode, "modulus") else: big(0)
  if hasModulus and modulus <= big(1): raise newException(ValueError, "modulus must be at least 2")
  if relative < big(root.size):
    let value = root.exact[relative.toInt64()]
    result = %*{"index": index.toJson(), "via": "supplied_term"}
    if hasModulus:
      result["modulus"] = modulus.toJson()
      result["value"] = floorMod(value, modulus).toJson()
    else:
      result["value"] = value.toJson()
    return
  var chosen = -1
  for i, d in ex.directions:
    if d.form.available and (chosen < 0 or d.path.len < ex.directions[chosen].path.len): chosen = i
  if chosen < 0:
    for i, d in ex.directions:
      if d.form.multiplicativeModel != mmNone and d.path.len == 0:
        chosen = i
        break
  if chosen < 0:
    let q = if hasModulus and modulus.fitsInt64(): modulus.toInt64() else: 0'i64
    var modularFailure = ""
    # Prefer the original sequence, then try the shorter transform paths first.
    for depth in 0 .. MaxTransformDepth:
      for d in ex.directions:
        if not d.form.modular or d.path.len != depth: continue
        try:
          let blocks = modularFitBlocks(d.form, q)
          if not relative.fitsInt64() or relative.toInt64() > MaxTargetIterations:
            raise newException(ValueError, "Holonomic iteration is limited to " & $MaxTargetIterations & " steps")
          let step = relative.toInt64()
          let value = if depth == 0: holonomicResiduesMod(blocks, root, step, q)
                      else: pathTermMod(d, root, step, q)
          let via = $d.hypothesis & " with residue coefficients" &
            (if depth == 0: "" else: " of " & pathExpression(d.path))
          return %*{"index": index.toJson(), "via": via,
            "status": "extrapolation_from_conjectured_direction",
            "note": "Valid if the direction holds for every n, which still needs a proof.",
            "modulus": q, "value": value}
        except ValueError as error:
          if modularFailure.len == 0: modularFailure = error.msg
    for i, d in ex.directions:
      if d.hypothesis == hMultiplicative and d.path.len == 0:
        chosen = i
        break
    if chosen < 0:
      if modularFailure.len > 0: raise newException(ValueError, modularFailure)
      raise newException(ValueError, "No formula direction with rational coefficients was found")
  let direction = ex.directions[chosen]
  let form = direction.form
  let kindText = if form.available: $form.kind else: "multiplicative"
  result = %*{"index": index.toJson(), "via": (if direction.path.len == 0: kindText
      else: kindText & " of " & pathExpression(direction.path)),
    "status": "extrapolation_from_conjectured_direction",
    "note": "Valid if the direction holds for every n, which still needs a proof."}
  if not hasModulus or (form.kind == hMultiplicative and form.multiplicativeModel == mmNone):
    let (value, via) = exactTerm(direction, root, index, relative)
    if hasModulus:
      result["modulus"] = modulus.toJson()
      result["value"] = floorMod(value, modulus).toJson()
      result["tier"] = %"exact value reduced modulo the modulus"
    else:
      result["value"] = exactValueJson(value)
    if via.len > 0: result["via"] = %via
    return
  result["modulus"] = modulus.toJson()
  var failures: seq[string]
  residueRoute = ""
  # Tier 1: residues in int64.
  if modulus.fitsInt64() and modulus.toInt64() <= ResidueTierLimit:
    try:
      result["value"] = %residueTerm(direction, root, index, relative, modulus.toInt64())
      result["tier"] = %"residues modulo the modulus"
      if residueRoute.len > 0:
        result["via"] = %residueRoute
        result["route"] = %residueRoute
      return
    except ValueError as error:
      failures.add("residues: " & error.msg)
  # Tier 2: the exact value, then reduce.
  try:
    let (value, via) = exactTerm(direction, root, index, relative)
    result["value"] = floorMod(value, modulus).toJson()
    result["tier"] = %"exact value reduced modulo the modulus"
    if via.len > 0: result["via"] = %via
    return
  except ValueError as error:
    failures.add("exact: " & error.msg)
  # Tier 3: Kitamasa on big integers, for a linear recurrence at a huge index.
  if direction.path.len == 0 and form.available and form.kind == hLinearRecurrence:
    try:
      result["value"] = linearTermModBig(form, root, relative, modulus).toJson()
      result["tier"] = %"Kitamasa with big integer residues"
      return
    except ValueError as error:
      failures.add("big residues: " & error.msg)
  # Tier 4: CRT over the prime power parts of the modulus. A part up to 2^62 takes the
  # residue routes; a larger one takes the p-adic route of an order-one relation on big integers.
  try:
    var residues, moduli: seq[BigInt]
    var names: seq[string]
    for (p, e, part) in primePowerParts(modulus):
      if part <= big(ResidueTierLimit):
        residues.add(big(residueTerm(direction, root, index, relative, part.toInt64())))
      elif direction.path.len == 0 and form.available and form.kind notin
          {hPolynomial, hLinearRecurrence, hSelfReferential, hBinaryRecursive} and form.blocks.len == 2:
        residues.add(hypergeometricTermModBig(form.blocks, root, index, p, e, part))
      else:
        raise newException(ValueError, "the part " & $part & " exceeds 2^62, where only an order-one " &
          "relation has a residue route")
      moduli.add(part)
      names.add($part)
    result["value"] = chineseRemainderBig(residues, moduli).toJson()
    result["tier"] = %("residues modulo the prime power parts @[" & names.join(", ") & "], joined by CRT")
    if residueRoute.len > 0:
      result["via"] = %residueRoute
      result["route"] = %residueRoute
    return
  except ValueError as error:
    failures.add("prime power parts: " & error.msg)
  raise newException(ValueError, "No tier could evaluate the term modulo " & $modulus & ". " & failures.join("; "))

# Claims, generators and the tool entry point.

proc checkPrimes(): seq[int] =
  var candidate = ExactMaxPrime
  while result.len < 3:
    if isPrime(candidate): result.add(candidate)
    dec candidate

proc checkClaims(claims: JsonNode, root: Track): JsonNode =
  if claims.kind != JArray or claims.len > 16: raise newException(ValueError, "claims must be at most 16 expressions")
  result = newJArray()
  let primes = checkPrimes()
  for claim in claims:
    if claim.kind != JString: raise newException(ValueError, "Each claim is an expression string in n")
    let tree = parseExpression(claim.getStr())
    var variables = newJObject()
    var report = %*{"claim": claim.getStr()}
    var modular = 0
    var failed = false
    for i in 0 ..< root.size:
      let n = root.offset + i
      variables["n"] = %n
      var exactValue: Exact
      var exactOk = true
      try:
        exactValue = evaluateScalar(tree, variables, 0)
      except ValueError:
        exactOk = false
      if exactOk:
        if not exactValue.isInteger or exactValue.numerator != root.exact[i]:
          report["status"] = %"refuted"
          report["first_failure"] = %*{"n": n, "actual": root.exact[i].toJson(), "claimed": exactValue.toJson()}
          failed = true
          break
        continue
      inc modular
      for q in primes:
        var claimed: Exact
        try:
          claimed = evaluateScalar(tree, variables, q)
        except ValueError as error:
          report["status"] = %"undefined"
          report["first_failure"] = %*{"n": n, "reason": error.msg}
          failed = true
          break
        let actual = root.exact[i].modSmall(int64(q))
        if claimed.num != actual:
          report["status"] = %"refuted"
          report["first_failure"] = %*{"n": n, "modulus": q, "actual_residue": actual, "claimed_residue": claimed.num}
          failed = true
          break
      if failed: break
    if not failed:
      report["status"] = %"matches_all_terms"
      report["terms_checked"] = %root.size
      if modular > 0: report["terms_checked_modulo_three_primes"] = %modular
    result.add(report)

proc readTerms*(problem: JsonNode): tuple[terms: seq[BigInt], offset: int] =
  if problem.hasKey("terms"):
    let node = problem["terms"]
    if node.kind != JArray: raise newException(ValueError, "terms must be an array")
    for term in node: result.terms.add(toBig(term, "term"))
    result.offset = int(problem{"offset"}.getBiggestInt(0))
  elif problem.hasKey("generator"):
    let tree = expressionTree(problem["generator"], "generator")
    result.offset = int(problem{"start"}.getBiggestInt(0))
    let count = int(problem{"count"}.getBiggestInt(40))
    if count notin MinExplorerTerms .. MaxExplorerTerms:
      raise newException(ValueError, "count must be between " & $MinExplorerTerms & " and " & $MaxExplorerTerms)
    var variables = newJObject()
    for i in 0 ..< count:
      let n = result.offset + i
      variables["n"] = %n
      var value: Exact
      try:
        value = evaluateScalar(tree, variables, 0)
      except ValueError as error:
        raise newException(ValueError, "generator failed at n = " & $n & ": " & error.msg &
          ". Compute large terms elsewhere and pass them as decimal strings in terms.")
      if not value.isInteger: raise newException(ValueError, "generator gave a non-integer at n = " & $n)
      result.terms.add(value.numerator)
  else:
    raise newException(ValueError, "Supply terms, or a generator expression in n")
  if abs(result.offset) > 1_000_000: raise newException(ValueError, "offset is out of range")
  if result.terms.len notin MinExplorerTerms .. MaxExplorerTerms:
    raise newException(ValueError, "Supply between " & $MinExplorerTerms & " and " & $MaxExplorerTerms & " terms")

proc exploreSequence*(problem: JsonNode): JsonNode =
  let (terms, offset) = readTerms(problem)
  let root = rootTrack(terms, offset)
  var moduli = @DefaultModuli
  if problem.hasKey("moduli"):
    let node = problem["moduli"]
    if node.kind != JArray or node.len notin 1 .. 16: raise newException(ValueError, "moduli must list 1 to 16 integers")
    moduli = @[]
    for m in node:
      if m.kind != JInt or m.getBiggestInt() notin 2'i64 .. 1_000_000_000'i64:
        raise newException(ValueError, "Each modulus must be between 2 and 1000000000")
      moduli.add(m.getBiggestInt())
  let budget = int(problem{"max_tests"}.getBiggestInt(300))
  let maxDirections = int(problem{"max_directions"}.getBiggestInt(3))
  if budget notin 1 .. 5_000 or maxDirections notin 1 .. 8:
    raise newException(ValueError, "max_tests must be 1..5000 and max_directions 1..8")
  result = %*{"terms_used": terms.len, "offset": offset}
  if problem.hasKey("claims"): result["claims"] = checkClaims(problem["claims"], root)
  var ex = newExplorer(root, moduli)
  programLog.setLen(0)
  ex.explore(maxDirections, budget, explorerGuide)
  var directions = newJArray()
  for d in ex.directions:
    var entry = %*{"sequence": (if d.path.len == 0: "a" else: "b = " & pathExpression(d.path)),
                   "hypothesis": $d.hypothesis}
    for key, value in d.report: entry[key] = value
    directions.add(entry)
  result["directions"] = directions
  result["search"] = %*{"tests": ex.tests, "max_tests": budget,
    "order": (if not programPrologue.isNil: "spectron_tool_chain"
              elif explorerGuide.isNil: "exhaustive" else: "mlp_policy_then_exhaustive")}
  if programLog.len > 0: result["program"] = %programLog
  if problem.hasKey("target"):
    try:
      result["target"] = evaluateTarget(ex, root, problem["target"])
    except ValueError as error:
      result["target"] = %*{"status": "not_evaluated", "reason": error.msg}
  result["method"] = %"exact_search"
  result["note"] = %("Directions are fitted on early terms and checked on later ones modulo three primes. " &
    "Treat them as conjectures until proved. Refuted claims include an exact counterexample.")
  if directions.len == 0:
    result["suggestion"] = %("No direction found. Supply more terms, check the index offset, " &
      "or explore a related quantity such as a refined count or a generating function coefficient.")
