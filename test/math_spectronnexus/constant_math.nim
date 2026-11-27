## Integer relation search: identify a real number from its digits as a rational
## combination of known constants, or as a root of an integer polynomial.
import std/[algorithm, json, math, strutils, tables]
import big_integer, high_precision

const
  MaxLatticeDigits = 400
  MaxBasis = 14
  DefaultMinExcess = 10

proc floorDiv(a, b: BigInt): BigInt =
  let (q, r) = divMod(a, b)
  if not r.isZero and (r.isNegative != b.isNegative): q - big(1) else: q

proc exactQuotient(a, b: BigInt): BigInt =
  let (q, r) = divMod(a, b)
  if not r.isZero: raise newException(ValueError, "Integral LLL lost exactness")
  q

proc dotBig(a, b: seq[BigInt]): BigInt =
  result = big(0)
  for i in 0 ..< a.len: result = result + a[i] * b[i]

proc lllReduce*(rows: var seq[seq[BigInt]]) =
  ## Integral LLL with delta 3/4 (Cohen, Algorithm 2.6.7). Rows must be independent.
  let n = rows.len
  if n < 2: return
  var d = newSeq[BigInt](n + 1)
  var lam = newSeq[seq[BigInt]](n + 1)
  for i in 0 .. n:
    lam[i] = newSeq[BigInt](n + 1)
    for j in 0 .. n: lam[i][j] = big(0)
  d[0] = big(1)
  d[1] = dotBig(rows[0], rows[0])
  var k = 2
  var kmax = 1
  template reduceRow(k, l: int) =
    block:
      let large = big(2) * (if lam[k][l].isNegative: -lam[k][l] else: lam[k][l])
      if d[l] < large:
        let q = floorDiv(big(2) * lam[k][l] + d[l], big(2) * d[l])
        for c in 0 ..< rows[k - 1].len: rows[k - 1][c] = rows[k - 1][c] - q * rows[l - 1][c]
        lam[k][l] = lam[k][l] - q * d[l]
        for i in 1 ..< l: lam[k][i] = lam[k][i] - q * lam[l][i]
  template swapRows(k: int) =
    block:
      swap(rows[k - 1], rows[k - 2])
      for j in 1 .. k - 2: swap(lam[k][j], lam[k - 1][j])
      let mu = lam[k][k - 1]
      let b = exactQuotient(d[k - 2] * d[k] + mu * mu, d[k - 1])
      for i in k + 1 .. kmax:
        let t = lam[i][k]
        lam[i][k] = exactQuotient(d[k] * lam[i][k - 1] - mu * t, d[k - 1])
        lam[i][k - 1] = exactQuotient(b * t + mu * lam[i][k], d[k])
      d[k - 1] = b
  var iterations = 0
  while k <= n:
    inc iterations
    if iterations > 200_000: raise newException(ValueError, "LLL did not finish")
    if k > kmax:
      kmax = k
      for j in 1 .. k:
        var u = dotBig(rows[k - 1], rows[j - 1])
        for i in 1 ..< j: u = exactQuotient(d[i] * u - lam[k][i] * lam[j][i], d[i - 1])
        if j < k: lam[k][j] = u
        else:
          if u.isZero: raise newException(ValueError, "The lattice rows are dependent")
          d[k] = u
    reduceRow(k, k - 1)
    if big(4) * d[k] * d[k - 2] < big(3) * d[k - 1] * d[k - 1] - big(4) * lam[k][k - 1] * lam[k][k - 1]:
      swapRows(k)
      k = max(2, k - 1)
    else:
      for l in countdown(k - 2, 1): reduceRow(k, l)
      inc k

type Relation = object
  found: bool
  coefficients: seq[BigInt]
  residualDigits, coefficientDigits, excess: float64

proc absBig(x: BigInt): BigInt = (if x.isNegative: -x else: x)

proc findRelation(values: seq[Fixed], latticeDigits, fullDigits: int, needFirst: bool,
                  minExcess: float64): Relation =
  ## LLL on rows [e_i | round(10^latticeDigits * v_i)]. The relation must hold to
  ## fullDigits and beat the coefficient size by minExcess digits.
  let m = values.len
  var rows = newSeq[seq[BigInt]](m)
  for i in 0 ..< m:
    rows[i] = newSeq[BigInt](m + 1)
    for j in 0 ..< m: rows[i][j] = big(int64(ord(i == j)))
    rows[i][m] = divMod(values[i].value, tenPower(values[i].digits - latticeDigits)).quotient
  lllReduce(rows)
  var candidates: seq[(BigInt, int)]
  for i, row in rows:
    var norm = big(0)
    for j in 0 ..< m: norm = norm + row[j] * row[j]
    candidates.add((norm, i))
  candidates.sort(proc(a, b: (BigInt, int)): int = cmp(a[0], b[0]))
  for (_, index) in candidates:
    let row = rows[index]
    let coefficients = row[0 ..< m]
    if needFirst and coefficients[0].isZero: continue
    var nonzero = 0
    for c in coefficients:
      if not c.isZero: inc nonzero
    if nonzero < 2: continue
    var residual = big(0)
    var sizeDigits = 0.0
    var largest = 0.0
    for i, c in coefficients:
      residual = residual + c * values[i].value
      if not c.isZero:
        sizeDigits += max(0.0, c.log10Abs())
        largest = max(largest, c.log10Abs())
    let residualLog = if residual.isZero: -float64(fullDigits) else: residual.log10Abs() - float64(fullDigits)
    let matched = -residualLog
    if matched < float64(fullDigits) - 3.0 - largest - log10(float64(m)): continue
    let excess = float64(latticeDigits) - sizeDigits
    if excess < minExcess: continue
    return Relation(found: true, coefficients: coefficients, residualDigits: matched,
                    coefficientDigits: sizeDigits, excess: excess)

proc termText(coefficient: BigInt, symbol: string): string =
  let one = big(1)
  if symbol == "1": return $coefficient
  if coefficient == one: symbol
  elif coefficient == -one: "-" & symbol
  else: $coefficient & "*" & symbol

proc joinTerms(terms: seq[string]): string =
  for t in terms:
    if result.len == 0: result = t
    elif t.startsWith("-"): result.add(" - " & t[1 .. ^1])
    else: result.add(" + " & t)
  if result.len == 0: result = "0"

proc wrapped(symbol: string): string =
  if symbol.contains('+') or symbol.contains('-') or symbol.contains('/'): "(" & symbol & ")" else: symbol

proc linearReport(rel: Relation, names: seq[string]): JsonNode =
  ## x = -(sum a_i c_i) / a_0 with a_0 > 0 after normalizing.
  var a = rel.coefficients
  if a[0].isNegative:
    for c in a.mitems: c = -c
  var g = big(0)
  for c in a: g = gcdBig(g, c)
  if not g.isZero and not (g == big(1)):
    for c in a.mitems: c = divMod(c, g).quotient
  var numerator: seq[string]
  var relationTerms = @[termText(a[0], "x")]
  for i in 1 ..< a.len:
    if a[i].isZero: continue
    numerator.add(termText(-a[i], wrapped(names[i - 1])))
    relationTerms.add(termText(a[i], wrapped(names[i - 1])))
  let top = joinTerms(numerator)
  let closed = if a[0] == big(1): top
               elif numerator.len == 1 and not top.contains(' '): top & "/" & $a[0]
               else: "(" & top & ")/" & $a[0]
  %*{"closed_form": closed, "relation": joinTerms(relationTerms) & " = 0"}

proc algebraicReport(rel: Relation): JsonNode =
  var a = rel.coefficients
  var top = a.high
  while top > 0 and a[top].isZero: dec top
  if a[top].isNegative:
    for c in a.mitems: c = -c
  var terms: seq[string]
  for i in countdown(top, 0):
    if a[i].isZero: continue
    let symbol = if i == 0: "1" elif i == 1: "x" else: "x^" & $i
    terms.add(termText(a[i], symbol))
  %*{"minimal_polynomial_candidate": joinTerms(terms) & " = 0", "degree": top}

const DefaultGroups = [
  @["1"],
  @["1", "pi"], @["1", "pi^2"], @["1", "pi", "pi^2", "pi^3"],
  @["1", "log2"], @["1", "log2", "pi", "pi^2", "log2^2", "pi*log2"],
  @["1", "zeta3"], @["1", "zeta3", "pi^2", "pi^2*log2", "log2^3", "log2"],
  @["1", "catalan", "pi", "log2", "pi*log2"],
  @["1", "zeta3", "zeta5", "pi^2", "pi^4"],
  @["1", "sqrt2", "sqrt3", "sqrt5", "sqrt6"],
  @["1", "e", "e^2", "1/e"],
  @["1", "gamma", "log2", "pi"],
  @["1", "log3", "pi*sqrt3", "pi^2", "log2"]]

proc solveIdentifyConstant*(problem: JsonNode): JsonNode =
  let valueNode = problem{"value"}
  if valueNode.isNil or valueNode.kind != JString:
    raise newException(ValueError, "value must be a decimal string with many digits, for example \"1.2020569031595942853997\"")
  let text = valueNode.getStr()
  var available = fractionDigits(text)
  if problem.hasKey("digits"): available = min(available, int(problem["digits"].getBiggestInt()))
  if available < 12: raise newException(ValueError, "Supply at least 12 digits after the decimal point")
  let fullDigits = min(available, MaxLatticeDigits + 20)
  let latticeDigits = min(fullDigits - 2, MaxLatticeDigits)
  let minExcess = problem{"min_excess"}.getFloat(float64(DefaultMinExcess))
  let x = parseDecimal(text, fullDigits)
  var attempts = newJArray()
  proc report(rel: Relation, details: JsonNode, kind: string): JsonNode =
    result = details
    result["kind"] = %kind
    result["status"] = %"integer_relation_found"
    result["evidence"] = %*{"digits_used": fullDigits, "relation_holds_to_digits": round(rel.residualDigits, 1),
      "coefficient_digits": round(rel.coefficientDigits, 1), "excess_digits": round(rel.excess, 1)}
    result["attempts"] = attempts
    result["verified"] = %false
    result["method"] = %"integral_lll"
    result["note"] = %("The relation holds to the supplied precision and its coefficients are much smaller than " &
      "chance would give. It is strong evidence, not a proof.")
  let degreeNode = problem{"algebraic_degree"}
  if not degreeNode.isNil:
    let degree = int(degreeNode.getBiggestInt())
    if degree notin 1 .. MaxBasis - 1: raise newException(ValueError, "algebraic_degree must be between 1 and " & $(MaxBasis - 1))
    var powers = @[fromInteger(fullDigits, big(1))]
    for i in 1 .. degree: powers.add(powers[^1] * x)
    let rel = findRelation(powers, latticeDigits, fullDigits, false, minExcess)
    attempts.add(%*{"basis": "powers of x up to " & $degree, "found": rel.found})
    if rel.found: return report(rel, algebraicReport(rel), "algebraic")
    return %*{"status": "no_relation", "attempts": attempts, "method": "integral_lll", "verified": false,
      "note": "No integer polynomial of degree at most " & $degree & " with small coefficients fits these digits."}
  var groups: seq[seq[string]]
  let basisNode = problem{"basis"}
  if basisNode.isNil:
    for g in DefaultGroups: groups.add(g)
  else:
    if basisNode.kind != JArray or basisNode.len notin 1 .. MaxBasis:
      raise newException(ValueError, "basis must list 1 to " & $MaxBasis & " constant expressions")
    var names: seq[string]
    for item in basisNode:
      if item.kind != JString: raise newException(ValueError, "Each basis entry is a constant expression string")
      names.add(item.getStr())
    groups.add(names)
  var values = initTable[string, Fixed]()
  for names in groups:
    var vector = @[x]
    for name in names:
      if name notin values: values[name] = evaluateConstant(name, fullDigits)
      vector.add(values[name])
    var rel: Relation
    try:
      rel = findRelation(vector, latticeDigits, fullDigits, true, minExcess)
    except ValueError as error:
      attempts.add(%*{"basis": names, "found": false, "reason": error.msg})
      continue
    attempts.add(%*{"basis": names, "found": rel.found})
    if rel.found:
      let details = linearReport(rel, names)
      let check = evaluateConstant(details["closed_form"].getStr(), fullDigits)
      let difference = check - x
      let agree = if difference.value.isZero: fullDigits
                  else: fullDigits - (difference.value.digits())
      details["closed_form_matches_digits"] = %agree
      return report(rel, details, "linear")
  %*{"status": "no_relation", "attempts": attempts, "method": "integral_lll", "verified": false,
     "note": "No small relation in the tried bases. Supply more digits, a custom basis, or algebraic_degree."}
