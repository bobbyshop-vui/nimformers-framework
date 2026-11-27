## Certified affine solution spaces and inconsistency witnesses over F_p.
import std/[json, strformat]
import prime_field

const SystemMaxOrder* = 64

type
  FieldSystem = object
    prime, rows, cols: int
    matrix: seq[seq[int]]
    rhs: seq[int]
  SystemProof = object
    pivots: seq[int]
    reduced, rowCombination, kernel: seq[seq[int]]
    particular, contradiction: seq[int]
    determinant: int
    methodSteps: JsonNode

proc readVector(node: JsonNode, size, prime: int, label: string): seq[int] =
  if node.isNil or node.kind != JArray or node.len != size:
    raise newException(ValueError, &"{label} must have {size} integer entries")
  result = newSeq[int](size)
  for i, entry in node.elems:
    result[i] = residue(integer(entry, label & " entry"), prime)

proc parseSystem(node: JsonNode, homogeneous = false): FieldSystem =
  result.prime = readPrime(node)
  let matrix = node{"matrix"}
  if matrix.isNil or matrix.kind != JArray or matrix.len notin 1 .. SystemMaxOrder:
    raise newException(ValueError, &"matrix must have 1 to {SystemMaxOrder} rows")
  if matrix[0].kind != JArray or matrix[0].len notin 1 .. SystemMaxOrder:
    raise newException(ValueError, &"matrix must have 1 to {SystemMaxOrder} columns")
  result.rows = matrix.len
  result.cols = matrix[0].len
  for row in matrix:
    result.matrix.add(readVector(row, result.cols, result.prime, "matrix row"))
  result.rhs = if homogeneous: newSeq[int](result.rows)
               else: readVector(node{"rhs"}, result.rows, result.prime, "rhs")

proc reduce(system: FieldSystem, recordMethod = false): SystemProof =
  let rows = system.rows
  let cols = system.cols
  let p = system.prime
  let witnessStart = cols + 1
  let width = witnessStart + rows
  var tableau = newSeq[seq[int]](rows)
  for r in 0 ..< rows:
    tableau[r] = system.matrix[r]
    tableau[r].setLen(width)
    tableau[r][cols] = system.rhs[r]
    tableau[r][witnessStart + r] = 1
  var rank = 0
  result.determinant = 1
  if recordMethod: result.methodSteps = newJArray()
  for c in 0 ..< cols:
    if rank == rows: break
    var pivot = rank
    while pivot < rows and tableau[pivot][c] == 0: inc pivot
    if pivot == rows: continue
    if pivot != rank:
      swap(tableau[pivot], tableau[rank])
      result.determinant = residue(-int64(result.determinant), p)
    let pivotValue = tableau[rank][c]
    result.determinant = productMod(result.determinant, pivotValue, p)
    let inv = reciprocal(pivotValue, p)
    var eliminated: JsonNode
    if recordMethod: eliminated = newJArray()
    tableau[rank][c] = 1
    if inv != 1:
      for j in c + 1 ..< width: tableau[rank][j] = productMod(tableau[rank][j], inv, p)
    for r in 0 ..< rows:
      if r == rank: continue
      let factor = tableau[r][c]
      if factor == 0: continue
      if recordMethod: eliminated.add(%*{"row": r, "subtract_multiple": factor})
      tableau[r][c] = 0
      for j in c + 1 ..< width:
        if tableau[rank][j] != 0:
          tableau[r][j] = subtractProduct(tableau[r][j], factor, tableau[rank][j], p)
    if recordMethod:
      result.methodSteps.add(%*{"operation": "pivot_and_eliminate", "column": c,
        "swap_rows": [rank, pivot], "pivot": pivotValue, "scale_by": inv, "elimination": eliminated})
    result.pivots.add(c)
    inc rank
  if rank < rows: result.determinant = 0

  var pivotMask: array[SystemMaxOrder, bool]
  for r, c in result.pivots:
    pivotMask[c] = true
    result.reduced.add(tableau[r][0 ..< cols])
    result.rowCombination.add(tableau[r][witnessStart ..< width])
  for free in 0 ..< cols:
    if pivotMask[free]: continue
    var basisVector = newSeq[int](cols)
    basisVector[free] = 1
    for r, c in result.pivots:
      basisVector[c] = residue(-int64(tableau[r][free]), p)
    result.kernel.add(basisVector)
  for r in rank ..< rows:
    if tableau[r][cols] != 0:
      result.contradiction = tableau[r][witnessStart ..< width]
      return
  result.particular.setLen(cols)
  for r, c in result.pivots: result.particular[c] = tableau[r][cols]

proc dot(a, b: openArray[int], p: int): int =
  var acc = 0'i64
  for i, value in a: acc += int64(value) * int64(b[i])
  residue(acc, p)

proc verify(system: FieldSystem, proof: SystemProof): bool =
  # The pivot submatrix of P*A is an identity, certifying rank >= r.
  # The kernel vectors have an identity on their free coordinates, proving
  # nullity >= n-r. A particular solution or a left-kernel witness settles b.
  let p = system.prime
  let cols = system.cols
  let rows = system.rows
  let rank = proof.pivots.len
  if rank > min(cols, rows) or proof.kernel.len != cols - rank or
      proof.reduced.len != rank or proof.rowCombination.len != rank: return false
  var pivotMask: array[SystemMaxOrder, bool]
  var lastPivot = -1
  for c in proof.pivots:
    if c <= lastPivot or c >= cols: return false
    pivotMask[c] = true
    lastPivot = c
  var transpose = newSeq[seq[int]](cols)
  for c in 0 ..< cols:
    transpose[c].setLen(rows)
    for r in 0 ..< rows: transpose[c][r] = system.matrix[r][c]
  for i in 0 ..< rank:
    if proof.reduced[i].len != cols or proof.rowCombination[i].len != rows: return false
    for j, c in proof.pivots:
      if proof.reduced[i][c] != ord(i == j): return false
    for c in 0 ..< cols:
      if (c < proof.pivots[i] and proof.reduced[i][c] != 0) or
          dot(proof.rowCombination[i], transpose[c], p) != proof.reduced[i][c]:
        return false
  for i, v in proof.kernel:
    if v.len != cols: return false
    var freeIndex = 0
    for c in 0 ..< cols:
      if not pivotMask[c]:
        if v[c] != ord(i == freeIndex): return false
        inc freeIndex
    for row in system.matrix:
      if dot(row, v, p) != 0: return false
  if proof.contradiction.len > 0:
    if proof.contradiction.len != rows or proof.particular.len != 0: return false
    for column in transpose:
      if dot(proof.contradiction, column, p) != 0: return false
    return dot(proof.contradiction, system.rhs, p) != 0
  if proof.particular.len != cols: return false
  for r, row in system.matrix:
    if dot(row, proof.particular, p) != system.rhs[r]: return false
  true

proc solveSystem*(node: JsonNode): JsonNode =
  let system = parseSystem(node)
  let proof = reduce(system)
  if not verify(system, proof):
    raise newException(ValueError, "Linear system certificate did not verify")
  let rank = proof.pivots.len
  let consistent = proof.contradiction.len == 0
  let particular = if consistent: %proof.particular else: newJNull()
  let solution = if consistent:
    &"Over F_{system.prime}, x0 solves A*x = b. Every solution is x0 plus " &
      &"a linear combination of the {proof.kernel.len} independent kernel vectors."
  else:
    "The left-kernel witness y satisfies y^T*A = 0 and y^T*b is nonzero. The system is inconsistent."
  let contradiction = if consistent: newJNull() else: %*{
    "left_kernel_vector": proof.contradiction, "nonzero_rhs": dot(proof.contradiction, system.rhs, system.prime)}
  %*{"answer": {"consistent": consistent, "rank": rank, "particular": particular,
                "nullspace_basis": proof.kernel, "pivot_columns": proof.pivots},
     "solution": solution, "verified": true, "method": "exact",
     "certificate": {"echelon": proof.reduced, "row_combination": proof.rowCombination,
                     "contradiction": contradiction}}

proc determinantCheck(system: FieldSystem): int =
  # Forward elimination leaves pivots unscaled and only modifies lower rows.
  let p = system.prime
  var upper = system.matrix
  result = 1
  for c in 0 ..< system.cols:
    var pivot = c
    while pivot < system.rows and upper[pivot][c] == 0: inc pivot
    if pivot == system.rows: return 0
    if pivot != c:
      swap(upper[pivot], upper[c])
      result = residue(-int64(result), p)
    let diagonal = upper[c][c]
    result = productMod(result, diagonal, p)
    let inv = reciprocal(diagonal, p)
    for r in c + 1 ..< system.rows:
      let factor = productMod(upper[r][c], inv, p)
      if factor == 0: continue
      for j in c + 1 ..< system.cols:
        upper[r][j] = subtractProduct(upper[r][j], factor, upper[c][j], p)

proc solveSquareMatrix*(node: JsonNode; inverse: bool): JsonNode =
  let system = parseSystem(node, homogeneous = true)
  if system.rows != system.cols:
    raise newException(ValueError, "Expected a square matrix")
  let proof = reduce(system, recordMethod = true)
  if not verify(system, proof):
    raise newException(ValueError, "Matrix elimination certificate did not verify")
  let singular = proof.pivots.len < system.cols
  var answer: JsonNode
  var solution: string
  let witness = if singular: %proof.kernel[0] else: newJNull()
  if inverse:
    answer = if singular: newJNull() else: %proof.rowCombination
    solution = if singular:
      "There is a nonzero vector in the kernel, so the matrix is singular."
    else:
      "Apply the same row operations to A and the identity. The row combination P satisfies P*A = I, so P is the inverse."
  else:
    if determinantCheck(system) != proof.determinant:
      raise newException(ValueError, "Determinant did not verify")
    answer = %proof.determinant
    solution = if singular:
      "The matrix has a nonzero kernel vector, so its determinant is zero."
    else:
      "Multiply the pivot values, changing the sign for each row swap, and reduce modulo the prime."
  %*{"answer": answer, "solution": solution, "verified": true, "method": "exact",
     "method_steps": proof.methodSteps,
     "certificate": {"prime": system.prime, "echelon": proof.reduced,
                     "row_combination": proof.rowCombination, "kernel_vector": witness}}
