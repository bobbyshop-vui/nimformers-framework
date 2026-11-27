## Mathematical state machines and certificate checkers. All transitions use integers.
import std/[algorithm, json, strformat]
import prime_field, polynomial_math, recurrence_math

const
  MaxOrder* = 14
  MaxPrime* = 257
  MaxCapacity* = 4096
  TokensPerState* = 256
  ExpertStepLimit* = MaxOrder * MaxOrder * MaxOrder * MaxOrder + 1

type
  ProblemKind* = enum rankProblem, flowProblem, polyGcdProblem, recurrenceProblem
  Grid* = array[MaxOrder, array[MaxOrder, int]]
  Vertex* = range[0 .. MaxOrder - 1]
  Action* = range[0 .. MaxOrder]
  ActionMask* = set[Action]
  MathFrame* = array[TokensPerState, float32]

  MathProblem* = object
    kind*: ProblemKind
    rows*, cols*: int
    prime*: int
    source*, sink*: int
    values*: Grid
    polynomial*: PolynomialProblem
    recurrence*: RecurrenceProblem
    walk*: WalkProblem

  Progress* = enum active, finished, failed
  Augmentation* = object
    vertices*: seq[int]
    amount*: int

  SolverState* = object
    problem*: MathProblem
    progress*: Progress
    steps*, work*: int
    methodSteps: JsonNode
    reduced, transform, inverse: Grid
    column, rank: int
    pivots: seq[int]
    net: Grid
    path: seq[int]
    visited: set[Vertex]
    flowValue, capacityScale: int
    levelsValid: bool
    residualDistance, residualWidth: array[MaxOrder, int]
    augmentations: seq[Augmentation]
    euclid: EuclidState
    proposedOrder: int
    orderTooLow, orderSingular: set[Action]

  Certificate* = object
    answer*: int
    echelon*, transform*, inverse*: Grid
    pivots*: seq[int]
    flow*: Grid
    sourceSide*: set[Vertex]
    polynomial*: PolyCertificate
    recurrence*: RecurrenceFit

proc fieldMod*(value: int, prime: int): int =
  let residue = value mod prime
  if residue < 0: residue + prime else: residue

proc intField(node: JsonNode, name: string): int =
  if not node.hasKey(name) or node[name].kind != JInt:
    raise newException(ValueError, "Expected integer field " & name)
  node[name].getInt()

proc parseProblem*(node: JsonNode): MathProblem =
  if node.kind != JObject or not node.hasKey("kind"):
    raise newException(ValueError, "Expected a math problem object with a kind")
  case node["kind"].getStr()
  of "finite_field_rank":
    result.kind = rankProblem
    result.prime = readPrime(node, MaxPrime)
    let matrix = node{"matrix"}
    if matrix.isNil or matrix.kind != JArray or matrix.len notin 1 .. MaxOrder:
      raise newException(ValueError, "Invalid matrix row count")
    result.rows = matrix.len
    if matrix[0].kind != JArray:
      raise newException(ValueError, "Expected an array of matrix rows")
    result.cols = matrix[0].len
    if result.cols notin 1 .. MaxOrder:
      raise newException(ValueError, "Invalid matrix column count")
    for r, row in matrix.elems:
      if row.kind != JArray or row.len != result.cols:
        raise newException(ValueError, "Matrix rows must have equal length")
      for c, entry in row.elems:
        if entry.kind != JInt:
          raise newException(ValueError, "Matrix entries must be integers")
        result.values[r][c] = fieldMod(entry.getInt(), result.prime)
  of "max_flow":
    result.kind = flowProblem
    result.rows = intField(node, "nodes")
    result.cols = result.rows
    result.source = intField(node, "source")
    result.sink = intField(node, "sink")
    if result.rows notin 2 .. MaxOrder or result.source notin 0 ..< result.rows or
        result.sink notin 0 ..< result.rows or result.source == result.sink:
      raise newException(ValueError, "Invalid graph size or terminals")
    let edges = node{"edges"}
    if edges.isNil or edges.kind != JArray:
      raise newException(ValueError, "Expected edges as [from, to, capacity] arrays")
    for edge in edges:
      if edge.kind != JArray or edge.len != 3 or
          edge[0].kind != JInt or edge[1].kind != JInt or edge[2].kind != JInt:
        raise newException(ValueError, "Expected an integer edge [from, to, capacity]")
      let u = edge[0].getInt()
      let v = edge[1].getInt()
      let capacity = edge[2].getInt()
      if u notin 0 ..< result.rows or v notin 0 ..< result.rows or u == v or
          capacity notin 1 .. MaxCapacity or result.values[u][v] > MaxCapacity - capacity:
        raise newException(ValueError, "Invalid edge or combined capacity")
      result.values[u][v] += capacity
  of "finite_field_polynomial_gcd":
    result.kind = polyGcdProblem
    result.polynomial = parsePolynomialProblem(node, policyWidth = MaxOrder, primeLimit = MaxPrime)
    result.prime = result.polynomial.prime
  of "recurrence_term":
    result.kind = recurrenceProblem
    result.recurrence = parseRecurrence(node, MaxOrder)
    result.prime = result.recurrence.prime
    if result.recurrence.orderBound == 0 or result.prime > MaxPrime:
      raise newException(ValueError, "The recurrence policy requires a bound and prime at most 257")
  of "weighted_walk_count":
    result.kind = recurrenceProblem
    result.walk = parseWalk(node, MaxOrder)
    if result.walk.prime > MaxPrime:
      raise newException(ValueError, "The walk policy requires a prime at most 257")
    result.prime = result.walk.prime
    result.recurrence = walkRecurrence(result.walk)
  else:
    raise newException(ValueError, "This kind has no policy state representation")

proc gridJson(grid: Grid, rows, cols: int): JsonNode =
  result = newJArray()
  for r in 0 ..< rows:
    result.add(%grid[r][0 ..< cols])

proc edgeList(grid: Grid, n: int): JsonNode =
  result = newJArray()
  for u in 0 ..< n:
    for v in 0 ..< n:
      if grid[u][v] != 0:
        result.add(%[u, v, grid[u][v]])

proc problemJson*(problem: MathProblem): JsonNode =
  case problem.kind
  of rankProblem:
    %*{"kind": "finite_field_rank", "prime": problem.prime,
       "matrix": gridJson(problem.values, problem.rows, problem.cols)}
  of flowProblem:
    %*{"kind": "max_flow", "nodes": problem.rows, "source": problem.source,
       "sink": problem.sink, "edges": edgeList(problem.values, problem.rows)}
  of polyGcdProblem:
    problemJson(problem.polynomial)
  of recurrenceProblem:
    if problem.walk.matrix.len > 0: walkJson(problem.walk) else: recurrenceJson(problem.recurrence)

proc newSolver*(problem: MathProblem, recordMethod = false): SolverState =
  result.problem = problem
  if recordMethod: result.methodSteps = newJArray()
  case problem.kind
  of rankProblem:
    result.reduced = problem.values
    for i in 0 ..< problem.rows:
      result.transform[i][i] = 1
      result.inverse[i][i] = 1
  of flowProblem:
    result.path = @[problem.source]
    result.visited.incl(Vertex(problem.source))
    result.capacityScale = 1
    for u in 0 ..< problem.rows:
      for v in 0 ..< problem.rows:
        result.capacityScale = max(result.capacityScale, problem.values[u][v])
  of polyGcdProblem:
    result.euclid = newEuclid(problem.polynomial)
  of recurrenceProblem:
    result.proposedOrder = -1

proc residual(state: SolverState, u, v: int): int =
  state.problem.values[u][v] - state.net[u][v]

proc orderFloor(state: SolverState): int =
  ## A nonsingular Hankel system of order k that misses a sample proves the
  ## minimal order exceeds k. Below the minimal order L that is the only
  ## nonsingular outcome, and above L every Hankel system is singular.
  for order in state.orderTooLow: result = max(result, int(order) + 1)

proc orderCeiling(state: SolverState): int =
  ## A singular system is evidence, not proof, that the minimal order is lower.
  result = state.problem.recurrence.orderBound
  let floor = state.orderFloor()
  for order in state.orderSingular:
    if int(order) > floor: result = min(result, int(order) - 1)

proc legalActions*(state: SolverState): ActionMask =
  if state.progress != active: return
  let problem = state.problem
  case problem.kind
  of rankProblem:
    # Action 0 is a claim that this column has no pivot in the remaining rows.
    result.incl(0)
    for row in state.rank ..< problem.rows:
      if state.reduced[row][state.column] != 0:
        result.incl(Action(row + 1))
  of flowProblem:
    # A cut can be claimed only between augmentations.
    if state.path.len == 1: result.incl(0)
    let u = state.path[^1]
    for v in 0 ..< problem.rows:
      if Vertex(v) notin state.visited and state.residual(u, v) > 0:
        result.incl(Action(v + 1))

  of polyGcdProblem:
    for step in state.euclid.allowed(): result.incl(Action(ord(step)))
  of recurrenceProblem:
    # Each action tests one order. Orders already tested stay excluded.
    let tested = state.orderTooLow + state.orderSingular
    for order in 0 .. problem.recurrence.orderBound:
      if Action(order) notin tested: result.incl(Action(order))

proc advanceRecurrence(state: var SolverState, action: Action) =
  let fit = fitOrder(state.problem.recurrence, int(action))
  if not state.methodSteps.isNil:
    let outcome = if fit.valid: "consistent" elif fit.singular: "singular" else: "contradicts_samples"
    state.methodSteps.add(%*{"operation": "test_order_hankel_fit", "order": int(action), "outcome": outcome})
  if fit.valid:
    state.proposedOrder = int(action)
    state.progress = finished
    return
  if fit.singular: state.orderSingular.incl(action) else: state.orderTooLow.incl(action)
  if state.legalActions() == {}: state.progress = failed

proc viableOrders(state: SolverState, legal: ActionMask): ActionMask =
  result = legal
  let floor = state.orderFloor()
  for action in legal:
    if int(action) < floor: result.excl(action)

proc advanceRank(state: var SolverState, action: Action) =
  let p = state.problem
  let r = state.rank
  let c = state.column
  if action == 0:
    for i in r ..< p.rows:
      if state.reduced[i][c] != 0:
        state.progress = failed
        return
    if not state.methodSteps.isNil:
      state.methodSteps.add(%*{"operation": "skip_column", "column": c})
  else:
    let pivot = int(action) - 1
    if pivot != r:
      swap(state.reduced[pivot], state.reduced[r])
      swap(state.transform[pivot], state.transform[r])
      for i in 0 ..< p.rows: swap(state.inverse[i][pivot], state.inverse[i][r])
    let value = state.reduced[r][c]
    let inverse = reciprocal(value, p.prime)
    var elimination: JsonNode
    if not state.methodSteps.isNil: elimination = newJArray()
    for j in c ..< p.cols:
      state.reduced[r][j] = state.reduced[r][j] * inverse mod p.prime
    for j in 0 ..< p.rows:
      state.transform[r][j] = state.transform[r][j] * inverse mod p.prime
      state.inverse[j][r] = state.inverse[j][r] * value mod p.prime
    for i in r + 1 ..< p.rows:
      let factor = state.reduced[i][c]
      if factor == 0: continue
      if not elimination.isNil: elimination.add(%*{"row": i, "subtract_multiple": factor})
      for j in c ..< p.cols:
        if state.reduced[r][j] == 0: continue
        state.reduced[i][j] = fieldMod(state.reduced[i][j] - factor * state.reduced[r][j], p.prime)
        inc state.work
      for j in 0 ..< p.rows:
        state.transform[i][j] = fieldMod(state.transform[i][j] - factor * state.transform[r][j], p.prime)
        state.inverse[j][r] = (state.inverse[j][r] + factor * state.inverse[j][i]) mod p.prime
    if not state.methodSteps.isNil:
      state.methodSteps.add(%*{"operation": "pivot_and_eliminate", "column": c, "swap_rows": [r, pivot],
        "pivot_row": r, "scale_by": inverse, "elimination": elimination})
    state.pivots.add(c)
    inc state.rank
  inc state.column
  if state.column == p.cols or state.rank == p.rows:
    state.progress = finished

proc advanceFlow(state: var SolverState, action: Action) =
  let p = state.problem
  if action == 0:
    state.progress = finished
    if not state.methodSteps.isNil: state.methodSteps.add(%*{"operation": "certify_min_cut"})
    return
  let vertex = int(action) - 1
  state.path.add(vertex)
  state.visited.incl(Vertex(vertex))
  inc state.work
  if vertex != p.sink: return
  var bottleneck = high(int)
  for i in 1 ..< state.path.len:
    bottleneck = min(bottleneck, state.residual(state.path[i - 1], state.path[i]))
  for i in 1 ..< state.path.len:
    let u = state.path[i - 1]
    let v = state.path[i]
    state.net[u][v] += bottleneck
    state.net[v][u] -= bottleneck
  state.flowValue += bottleneck
  state.augmentations.add(Augmentation(vertices: state.path, amount: bottleneck))
  if not state.methodSteps.isNil:
    state.methodSteps.add(%*{"operation": "augment_residual_path", "path": state.path, "amount": bottleneck})
  state.path = @[p.source]
  state.visited = {Vertex(p.source)}
  state.levelsValid = false

proc advance*(state: var SolverState, action: Action) =
  if action notin state.legalActions():
    state.progress = failed
    return
  inc state.steps
  case state.problem.kind
  of rankProblem: state.advanceRank(action)
  of flowProblem: state.advanceFlow(action)
  of polyGcdProblem:
    state.euclid.applyMove(PolyMove(action), state.methodSteps)
    state.work = state.euclid.work
    if action == ord(claimGcd): state.progress = finished
  of recurrenceProblem: state.advanceRecurrence(action)

proc bfs(state: SolverState, start: int, blocked: set[Vertex]): array[MaxOrder, int] =
  for entry in result.mitems: entry = -1
  var queue: array[MaxOrder, int]
  queue[0] = start
  result[start] = start
  var head = 0
  var tail = 1
  while head < tail:
    let u = queue[head]
    inc head
    for v in 0 ..< state.problem.rows:
      if result[v] < 0 and Vertex(v) notin blocked and state.residual(u, v) > 0:
        result[v] = u
        queue[tail] = v
        inc tail

proc guidedActions*(state: SolverState, legal: ActionMask): ActionMask =
  ## Keep mathematically viable next moves. The policy still scores the alternatives.
  result = legal
  case state.problem.kind
  of rankProblem, polyGcdProblem:
    if card(legal) > 1: result.excl(0)
  of recurrenceProblem: result = state.viableOrders(legal)
  of flowProblem:
    let sink = state.problem.sink
    var reached = {Vertex(sink)}
    var queue: array[MaxOrder, int]
    queue[0] = sink
    var head = 0
    var tail = 1
    while head < tail:
      let v = queue[head]
      inc head
      for u in 0 ..< state.problem.rows:
        if Vertex(u) notin (reached + state.visited) and state.residual(u, v) > 0:
          reached.incl(Vertex(u))
          queue[tail] = u
          inc tail
    for action in legal:
      if action != 0 and Vertex(int(action) - 1) notin reached: result.excl(action)
    if card(result) > 1: result.excl(0)

proc makeResidualLevels(state: var SolverState) =
  if state.levelsValid: return
  state.levelsValid = true
  state.residualWidth = default(typeof(state.residualWidth))
  for distance in state.residualDistance.mitems: distance = -1
  let sink = state.problem.sink
  state.residualDistance[sink] = 0
  state.residualWidth[sink] = high(int)
  var queue: array[MaxOrder, int]
  queue[0] = sink
  var head = 0
  var tail = 1
  while head < tail:
    let v = queue[head]
    inc head
    for u in 0 ..< state.problem.rows:
      let capacity = state.residual(u, v)
      if capacity <= 0: continue
      let nextDistance = state.residualDistance[v] + 1
      if state.residualDistance[u] < 0:
        state.residualDistance[u] = nextDistance
        queue[tail] = u
        inc tail
      if state.residualDistance[u] == nextDistance:
        state.residualWidth[u] = max(state.residualWidth[u], min(capacity, state.residualWidth[v]))

proc constrainedActions*(state: var SolverState, legal: ActionMask): ActionMask =
  if legal == {}: return
  result = legal
  case state.problem.kind
  of rankProblem, polyGcdProblem:
    if card(legal) > 1: result.excl(0)
  of recurrenceProblem: result = state.viableOrders(legal)
  of flowProblem:
    state.makeResidualLevels()
    let u = state.path[^1]
    let distance = state.residualDistance[u]
    if distance < 0:
      return legal * {Action(0)}
    for action in legal:
      if action == 0 or state.residualDistance[int(action) - 1] != distance - 1:
        result.excl(action)

proc expertPlan*(state: SolverState): seq[Action] =
  case state.problem.kind
  of rankProblem:
    var best = Action(0)
    var bestFill = high(int)
    for action in state.legalActions():
      if action == 0: continue
      var fill = 0
      let row = int(action) - 1
      for c in state.column ..< state.problem.cols:
        if state.reduced[row][c] != 0: inc fill
      if fill < bestFill:
        best = action
        bestFill = fill
    result = @[best]
  of flowProblem:
    let start = state.path[^1]
    let parents = state.bfs(start, state.visited)
    var v = state.problem.sink
    if parents[v] == -1: return @[Action(0)]
    # Plan the whole shortest augmenting path with one BFS.
    while v != start:
      result.add(Action(v + 1))
      v = parents[v]
    result.reverse()
  of polyGcdProblem:
    result = @[Action(ord(state.euclid.bestMove()))]
  of recurrenceProblem:
    # Bisect the untested orders between the proven floor and the singular
    # ceiling. Widen to every order above the floor when that range is empty.
    let legal = state.legalActions()
    let floor = state.orderFloor()
    let ceiling = state.orderCeiling()
    var candidates: seq[int]
    for action in legal:
      if int(action) in floor .. ceiling: candidates.add(int(action))
    if candidates.len == 0:
      for action in legal:
        if int(action) >= floor: candidates.add(int(action))
    result = @[Action(candidates[(candidates.len - 1) div 2])]

proc constrainedExpertPlan*(state: var SolverState): seq[Action] =
  if state.problem.kind != flowProblem: return state.expertPlan()
  state.makeResidualLevels()
  var u = state.path[^1]
  if state.residualDistance[u] < 0: return @[Action(0)]
  while u != state.problem.sink:
    var best = -1
    var width = 0
    for v in 0 ..< state.problem.rows:
      if state.residualDistance[v] != state.residualDistance[u] - 1: continue
      let candidate = min(state.residual(u, v), state.residualWidth[v])
      if candidate > width:
        width = candidate
        best = v
    if best < 0: raise newException(ValueError, "Invalid residual level plan")
    result.add(Action(best + 1))
    u = best

proc runExpert*(problem: MathProblem, recordMethod = false): SolverState =
  result = newSolver(problem, recordMethod)
  while result.progress == active and result.steps < ExpertStepLimit:
    for action in result.expertPlan():
      result.advance(action)

proc makeCertificate*(state: SolverState): Certificate =
  let p = state.problem
  case p.kind
  of rankProblem:
    result.answer = state.rank
    result.echelon = state.reduced
    result.transform = state.transform
    result.inverse = state.inverse
    result.pivots = state.pivots
  of flowProblem:
    result.answer = state.flowValue
    let reachable = state.bfs(p.source, {})
    for u in 0 ..< p.rows:
      if reachable[u] >= 0: result.sourceSide.incl(Vertex(u))
      for v in 0 ..< p.rows:
        result.flow[u][v] = max(0, state.net[u][v])
  of polyGcdProblem:
    result.polynomial = proof(p.polynomial, state.euclid)
  of recurrenceProblem:
    result.recurrence = fitOrder(p.recurrence, state.proposedOrder)

proc verify*(p: MathProblem, cert: Certificate): bool =
  ## Verify the witness against the original problem. The policy's status and
  ## its choice of elementary operations are not inputs to this check.
  case p.kind
  of rankProblem:
    if cert.answer notin 0 .. min(p.rows, p.cols) or cert.pivots.len != cert.answer:
      return false
    var lastPivot = -1
    for r in 0 ..< p.rows:
      for c in 0 ..< p.rows:
        if cert.transform[r][c] notin 0 ..< p.prime or cert.inverse[r][c] notin 0 ..< p.prime:
          return false
      for c in 0 ..< p.cols:
        if cert.echelon[r][c] notin 0 ..< p.prime: return false
    for r in 0 ..< p.rows:
      for c in 0 ..< p.rows:
        var product = 0
        for k in 0 ..< p.rows: product += cert.transform[r][k] * cert.inverse[k][c]
        if product mod p.prime != ord(r == c): return false
      var first = -1
      for c in 0 ..< p.cols:
        var product = 0
        for k in 0 ..< p.rows: product += cert.transform[r][k] * p.values[k][c]
        if product mod p.prime != cert.echelon[r][c]: return false
        if first < 0 and cert.echelon[r][c] != 0: first = c
      if r < cert.answer:
        if first <= lastPivot or cert.pivots[r] != first or cert.echelon[r][first] != 1:
          return false
        lastPivot = first
      elif first >= 0:
        return false
    true
  of flowProblem:
    if cert.answer < 0 or Vertex(p.source) notin cert.sourceSide or Vertex(p.sink) in cert.sourceSide:
      return false
    var cutCapacity = 0
    for u in cert.sourceSide:
      if int(u) >= p.rows: return false
    for u in 0 ..< p.rows:
      for v in 0 ..< p.rows:
        if cert.flow[u][v] notin 0 .. p.values[u][v]: return false
    for u in 0 ..< p.rows:
      var balance = 0
      for v in 0 ..< p.rows:
        balance += cert.flow[u][v] - cert.flow[v][u]
        if Vertex(u) in cert.sourceSide and Vertex(v) notin cert.sourceSide:
          cutCapacity += p.values[u][v]
      let expected = if u == p.source: cert.answer elif u == p.sink: -cert.answer else: 0
      if balance != expected: return false
    cutCapacity == cert.answer
  of polyGcdProblem:
    verify(p.polynomial, cert.polynomial)
  of recurrenceProblem:
    verifies(p.recurrence, cert.recurrence)

proc certified*(state: SolverState): bool =
  state.progress == finished and verify(state.problem, state.makeCertificate())

proc resultJson*(state: SolverState, methodName: string): JsonNode =
  let p = state.problem
  let cert = state.makeCertificate()
  if state.progress != finished or not verify(p, cert):
    raise newException(ValueError, "No verified solution")
  var proof: JsonNode
  var explanation: string
  case p.kind
  of rankProblem:
    proof = %*{"echelon": gridJson(cert.echelon, p.rows, p.cols),
               "row_transform": gridJson(cert.transform, p.rows, p.rows),
               "inverse_transform": gridJson(cert.inverse, p.rows, p.rows),
               "pivot_columns": cert.pivots}
    explanation = &"Over F_{p.prime}, the certificate has P*A = E and P*Q = I. " &
      &"Thus P is invertible. The echelon matrix E has {cert.answer} nonzero rows " &
      &"with pivot columns {$(%cert.pivots)} (zero based), so rank(A) = {cert.answer}."
  of flowProblem:
    var cut: seq[int]
    var paths = newJArray()
    for vertex in cert.sourceSide: cut.add(int(vertex))
    for aug in state.augmentations:
      paths.add(%*{"path": aug.vertices, "amount": aug.amount})
    proof = %*{"flow_edges": edgeList(cert.flow, p.rows),
               "source_side": cut, "augmenting_paths": paths}
    explanation = &"The listed augmenting paths yield a feasible flow of value {cert.answer}. " &
      &"The cut with source side {$(%cut)} has capacity {cert.answer}. " &
      "The equality proves optimality by max flow/min cut."
  of polyGcdProblem:
    result = polynomial_math.resultJson(p.polynomial, cert.polynomial, methodName)
  of recurrenceProblem:
    let solved = recurrenceResult(p.recurrence, cert.recurrence.coefficients, methodName, "policy_order_hankel_fit")
    if not state.methodSteps.isNil:
      let steps = newJArray()
      for step in state.methodSteps: steps.add(step)
      for step in solved["method_steps"]: steps.add(step)
      solved["method_steps"] = steps
    if p.walk.matrix.len > 0: return verifyWalk(p.walk, p.recurrence, solved)
    return solved
  if p.kind != polyGcdProblem:
    result = %*{"answer": cert.answer, "solution": explanation, "certificate": proof,
               "verified": true, "method": methodName}
  if not state.methodSteps.isNil: result["method_steps"] = state.methodSteps

proc encodeState*(state: SolverState, legal: ActionMask): MathFrame =
  ## Input layout is 196 grid entries, three vectors of 14, and 18 controls.
  ## Normalize integer fields for the policy. There is no text tokenizer.
  const
    GridWidth = MaxOrder * MaxOrder
    Position = GridWidth
    RowDegree = Position + MaxOrder
    ColumnDegree = RowDegree + MaxOrder
    Meta = ColumnDegree + MaxOrder
  static: doAssert Meta + 18 == TokensPerState
  let p = state.problem
  let order = float32(MaxOrder)
  let scale = if p.kind != flowProblem: float32(p.prime - 1) else: float32(state.capacityScale * 2)
  for r in 0 ..< p.rows:
    for c in 0 ..< p.cols:
      let value = if p.kind == rankProblem: state.reduced[r][c] else: state.residual(r, c)
      result[r * MaxOrder + c] = float32(value) / scale
      if value != 0:
        result[RowDegree + r] += 1'f32 / order
        result[ColumnDegree + c] += 1'f32 / order
  result[Meta + 2] = float32(p.rows) / order
  result[Meta + 3] = float32(p.cols) / order
  result[Meta + 12] = min(1'f32, float32(state.steps) / (order * order))
  result[Meta + 14] = float32(ord(0 in legal))
  result[Meta + 15] = float32(card(legal)) / (order + 1'f32)
  case p.kind
  of rankProblem:
    result[Meta] = 1
    result[Meta + 4] = float32(p.prime) / float32(MaxPrime)
    result[Meta + 5] = float32(state.rank) / order
    result[Meta + 6] = float32(state.column + 1) / order
    result[Meta + 16] = float32(p.cols - state.column) / order
    result[Meta + 17] = float32(p.rows - state.rank) / order
    for i, pivot in state.pivots:
      result[Position + i] = float32(pivot + 1) / order
  of flowProblem:
    result[Meta + 1] = 1
    result[Meta + 7] = float32(p.source + 1) / order
    result[Meta + 8] = float32(p.sink + 1) / order
    result[Meta + 9] = float32(state.path[^1] + 1) / order
    result[Meta + 10] = float32(state.path.len) / order
    result[Meta + 11] = float32(state.flowValue) / (float32(state.capacityScale) * order)
    result[Meta + 13] = min(1'f32, float32(state.augmentations.len) / order)
    result[Meta + 16] = result[ColumnDegree + p.sink]
    result[Meta + 17] = result[RowDegree + state.path[^1]]
    for i, vertex in state.path:
      result[Position + vertex] = float32(i + 1) / order
  of polyGcdProblem:
    let (left, right) = state.euclid.livePolynomials()
    for i, term in left: result[i] = float32(term) / scale
    for i, term in right: result[MaxOrder + i] = float32(term) / scale
    result[Meta + 2] = float32(left.len) / order
    result[Meta + 3] = float32(right.len) / order
    result[Meta + 4] = float32(p.prime) / float32(MaxPrime)
    for action in legal: result[Position + int(action)] = 1
  of recurrenceProblem:
    const
      OrderStatus = 2 * MaxOrder
      OrderOpen = OrderStatus + MaxOrder + 1
    static: doAssert OrderOpen + MaxOrder < GridWidth
    result[Meta + 8] = 1
    result[Meta + 2] = float32(p.recurrence.orderBound) / order
    result[Meta + 4] = float32(p.prime) / float32(MaxPrime)
    for i in 0 ..< min(p.recurrence.samples.len, 2 * MaxOrder):
      result[i] = float32(p.recurrence.samples[i]) / scale
    # Outcomes of earlier Hankel tests: +1 proves a higher order, -1 is singular.
    for tested in state.orderTooLow: result[OrderStatus + int(tested)] = 1
    for tested in state.orderSingular: result[OrderStatus + int(tested)] = -1
    let floor = state.orderFloor()
    let ceiling = state.orderCeiling()
    var bracketed = 0
    for action in legal:
      result[OrderOpen + int(action)] = 1
      if int(action) in floor .. ceiling: inc bracketed
    result[Meta + 5] = float32(floor) / order
    result[Meta + 6] = float32(ceiling) / order
    result[Meta + 16] = float32(bracketed) / (order + 1'f32)

proc policyFrame*(state: var SolverState, legal: ActionMask, flowFeatures: bool): MathFrame =
  result = state.encodeState(legal)
  if not flowFeatures or state.problem.kind != flowProblem: return
  state.makeResidualLevels()
  const RowFeatures = MaxOrder * MaxOrder + MaxOrder
  const ColumnFeatures = RowFeatures + MaxOrder
  let u = state.path[^1]
  let capacityScale = float32(2 * state.capacityScale)
  for v in 0 ..< state.problem.rows:
    let reach = state.residualDistance[v]
    result[ColumnFeatures + v] = if reach >= 0: 1'f32 - float32(reach) / float32(MaxOrder) else: 0'f32
    let width = min(state.residual(u, v), state.residualWidth[v])
    result[RowFeatures + v] = float32(max(0, width)) / capacityScale
