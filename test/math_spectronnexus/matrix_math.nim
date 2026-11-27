## Exact integer and prime field matrix multiplication.
import std/[json, strformat]
import big_integer, linear_system, prime_field

const MaxIntegerEntry* = 1_000_000

proc readMatrix(node: JsonNode, p: int, label: string): seq[seq[int64]] =
  if node.isNil or node.kind != JArray or node.len notin 1 .. SystemMaxOrder or
      node[0].kind != JArray or node[0].len notin 1 .. SystemMaxOrder:
    raise newException(ValueError, &"{label} must be a nonempty matrix up to {SystemMaxOrder} by {SystemMaxOrder}")
  let cols = node[0].len
  for row in node:
    if row.kind != JArray or row.len != cols:
      raise newException(ValueError, &"{label} must have equal row lengths")
    var values = newSeqOfCap[int64](cols)
    for entry in row:
      if p != 0:
        values.add(toBig(entry, label & " entry").modSmall(int64(p)))
        continue
      let value = integer(entry, label & " entry")
      if value notin -int64(MaxIntegerEntry) .. int64(MaxIntegerEntry):
        raise newException(ValueError, &"{label} entries must be within ±{MaxIntegerEntry}")
      values.add(value)
    result.add(values)

proc readBigMatrix(node: JsonNode, label: string): seq[seq[BigInt]] =
  if node.isNil or node.kind != JArray or node.len notin 1 .. SystemMaxOrder or
      node[0].kind != JArray or node[0].len notin 1 .. SystemMaxOrder:
    raise newException(ValueError, &"{label} must be a nonempty matrix up to {SystemMaxOrder} by {SystemMaxOrder}")
  let cols = node[0].len
  for row in node:
    if row.kind != JArray or row.len != cols:
      raise newException(ValueError, &"{label} must have equal row lengths")
    var values: seq[BigInt]
    for entry in row: values.add(toBig(entry, label & " entry"))
    result.add(values)

proc fitsSmall(node: JsonNode): bool =
  ## Every entry is a JSON integer within ±MaxIntegerEntry, so int64 dot products cannot overflow.
  if node.isNil or node.kind != JArray: return true
  for row in node:
    if row.kind != JArray: return true
    for entry in row:
      if entry.kind != JInt or entry.getBiggestInt() notin -int64(MaxIntegerEntry) .. int64(MaxIntegerEntry):
        return false
  true

proc bigMatrixProduct(problem: JsonNode): JsonNode =
  ## Entries of any size: the same dot products on big integers.
  let a = readBigMatrix(problem{"left"}, "left")
  let b = readBigMatrix(problem{"right"}, "right")
  if a[0].len != b.len:
    raise newException(ValueError, "Left columns must equal right rows")
  let cols = b[0].len
  var answer = newJArray()
  for row in a:
    var line = newJArray()
    for j in 0 ..< cols:
      var dot = big(0)
      for k, coefficient in row:
        if not coefficient.isZero and not b[k][j].isZero: dot = dot + coefficient * b[k][j]
      line.add(dot.toJson())
    answer.add(line)
  %*{"answer": answer, "verified": true, "method": "exact",
     "solution": "Take the dot product of each left row with each right column, with big integers since " &
       "some entry is past " & $MaxIntegerEntry & ".",
     "certificate": {"rows": a.len, "columns": cols, "inner_dimension": b.len, "prime": newJNull(),
                     "arithmetic": "big integers"}}

proc solveMatrixProduct*(problem: JsonNode): JsonNode =
  let p = if problem.hasKey("prime"): readPrime(problem) else: 0
  if p == 0 and not (fitsSmall(problem{"left"}) and fitsSmall(problem{"right"})):
    return bigMatrixProduct(problem)
  let a = readMatrix(problem{"left"}, p, "left")
  let b = readMatrix(problem{"right"}, p, "right")
  if a[0].len != b.len:
    raise newException(ValueError, "Left columns must equal right rows")
  let cols = b[0].len
  var answer = newSeq[seq[int64]](a.len)
  for i, row in a:
    answer[i] = newSeq[int64](cols)
    for k, coefficient in row:
      for j in 0 ..< cols:
        answer[i][j] += coefficient * b[k][j]
    if p != 0:
      for value in answer[i].mitems: value = int64(residue(value, p))
  var exampleTerms = newSeqOfCap[int64](b.len)
  for k in 0 ..< b.len: exampleTerms.add(a[0][k] * b[k][0])
  for i in 0 ..< a.len:
    for j in 0 ..< cols:
      var dot = 0'i64
      for k in 0 ..< b.len: dot += a[i][k] * b[k][j]
      let expected = if p == 0: dot else: int64(residue(dot, p))
      if answer[i][j] != expected:
        raise newException(ValueError, "Matrix product did not verify")
  let solution = "Take the dot product of each left row with each right column." &
    (if p != 0: &" Reduce each entry modulo {p}." else: "")
  let prime = if p == 0: newJNull() else: %p
  %*{"answer": answer, "solution": solution, "verified": true, "method": "exact",
     "method_steps": [{"operation": "row_by_column_dot_products",
       "row": 0, "column": 0, "products": exampleTerms, "entry": answer[0][0]}],
     "certificate": {"rows": a.len, "columns": cols, "inner_dimension": b.len,
                     "prime": prime}}
