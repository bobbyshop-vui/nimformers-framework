## Exact linear extension counts via decomposition and subset recurrences.
import std/[algorithm, bitops, json, strformat, strutils]
import prime_field

const MaxPosetOrder* = 20  # 20! fits in int64.

type
  Vertices = uint32
  Relations = array[MaxPosetOrder, Vertices]
  Count = object
    value: int64
    states: int
    proof: JsonNode

func bit(i: int): Vertices = 1'u32 shl i

iterator vertices(mask: Vertices): int =
  var rest = mask
  while rest != 0:
    yield countTrailingZeroBits(rest)
    rest = rest and (rest - 1)

proc vertexList(mask: Vertices): seq[int] =
  for v in vertices(mask): result.add(v)

proc components(nodes: Vertices, adjacent: Relations): seq[Vertices] =
  var remaining = nodes
  while remaining != 0:
    var frontier = bit(countTrailingZeroBits(remaining))
    var part = frontier
    remaining = remaining xor frontier
    while frontier != 0:
      let v = countTrailingZeroBits(frontier)
      let next = adjacent[v] and remaining
      frontier = (frontier xor bit(v)) or next
      remaining = remaining xor next
      part = part or next
    result.add(part)

proc binomial(n, k: int): int64 =
  result = 1
  for i in 1 .. k:
    result = (result * int64(n - k + i)) div int64(i)

proc countByFrontier(nodes: Vertices, prerequisites: Relations): tuple[value: int64, states: int] =
  let ids = vertexList(nodes)
  let full = (1 shl ids.len) - 1
  var local: array[MaxPosetOrder, int]
  for i, v in ids:
    for j, other in ids:
      if (prerequisites[v] and bit(other)) != 0:
        local[i] = local[i] or (1 shl j)
  var memo = newSeq[int64](full + 1)
  memo[full] = 1
  var visited = 1
  proc visit(used: int): int64 =
    if memo[used] != 0: return memo[used]
    var available = full xor used
    while available != 0:
      let next = countTrailingZeroBits(available)
      let choice = 1 shl next
      available = available xor choice
      if (local[next] and used) == local[next]:
        result += visit(used or choice)
    memo[used] = result
    inc visited
  result.value = visit(0)
  result.states = visited

proc countBlock(nodes: Vertices, before, after: Relations): Count =
  let size = countSetBits(nodes)
  if size <= 1:
    result.value = 1
    result.proof = %*{"rule": "trivial", "vertices": vertexList(nodes), "count": "1"}
    return
  var comparable, incomparable: Relations
  for v in vertices(nodes):
    comparable[v] = (before[v] or after[v]) and nodes
    incomparable[v] = nodes and not (comparable[v] or bit(v))
  var blocks = components(nodes, comparable)
  let shuffled = blocks.len > 1
  if not shuffled:
    blocks = components(nodes, incomparable)
    if blocks.len > 1:
      blocks.sort(proc(a, b: Vertices): int =
        # Distinct incomparability components are totally ordered.
        cmp(countSetBits(before[countTrailingZeroBits(a)] and nodes),
            countSetBits(before[countTrailingZeroBits(b)] and nodes)))
  if blocks.len > 1:
    result.value = 1
    var placed = 0
    var children = newJArray()
    for blockNodes in blocks:
      let child = countBlock(blockNodes, before, after)
      let width = countSetBits(blockNodes)
      if shuffled: result.value *= binomial(placed + width, width)
      result.value *= child.value
      result.states += child.states
      placed += width
      children.add(child.proof)
    let rule = if shuffled: "interleave_incomparable_blocks" else: "concatenate_ordered_blocks"
    result.proof = %*{"rule": rule, "vertices": vertexList(nodes), "count": $result.value,
                     "blocks": children}
    return
  let counted = countByFrontier(nodes, before)
  result.value = counted.value
  result.states = counted.states
  result.proof = %*{"rule": "subset_dp", "vertices": vertexList(nodes),
                   "count": $result.value, "states": result.states}

proc verifyBlock(proof: JsonNode, nodes: Vertices, before, after: Relations): tuple[value: int64, states: int] =
  # Check the partition constraints directly. At DP leaves, recompute by
  # choosing maximal elements, the dual of the solver's minimal-element rule.
  case proof["rule"].getStr()
  of "trivial":
    if countSetBits(nodes) > 1:
      raise newException(ValueError, "Invalid trivial block")
    result.value = 1
  of "subset_dp":
    result = countByFrontier(nodes, after)
  of "interleave_incomparable_blocks", "concatenate_ordered_blocks":
    let interleave = proof["rule"].getStr() == "interleave_incomparable_blocks"
    var seen: Vertices
    result.value = 1
    for item in proof["blocks"]:
      var part: Vertices
      for entry in item["vertices"]:
        let v = entry.getInt()
        if v notin 0 ..< MaxPosetOrder or (part and bit(v)) != 0:
          raise newException(ValueError, "Invalid partition vertex")
        part = part or bit(v)
      if part == 0 or (part and nodes) != part or (seen and part) != 0:
        raise newException(ValueError, "Overlapping or invalid partition blocks")
      for v in vertices(part):
        if interleave:
          if ((before[v] or after[v]) and seen) != 0:
            raise newException(ValueError, "Comparable vertices in interleaved blocks")
        elif (before[v] and seen) != seen:
          raise newException(ValueError, "Incorrect ordered block partition")
      let child = verifyBlock(item, part, before, after)
      if interleave:
        result.value *= binomial(countSetBits(seen or part), countSetBits(part))
      result.value *= child.value
      result.states += child.states
      seen = seen or part
    if seen != nodes: raise newException(ValueError, "Incomplete partition")
  else:
    raise newException(ValueError, "Unknown poset proof rule")
  if result.value != parseBiggestInt(proof["count"].getStr()):
    raise newException(ValueError, "Poset proof count does not verify")

proc solvePoset*(node: JsonNode): JsonNode =
  let sizeValue = integer(node{"nodes"}, "nodes")
  if sizeValue notin 0'i64 .. int64(MaxPosetOrder):
    raise newException(ValueError, &"nodes must be between 0 and {MaxPosetOrder}")
  let size = int(sizeValue)
  let pairs = node{"relations"}
  if pairs.isNil or pairs.kind != JArray or pairs.len > size * size:
    raise newException(ValueError, "relations must be an array of [before, after] pairs")
  var before, after: Relations
  for relation in pairs:
    if relation.kind != JArray or relation.len != 2:
      raise newException(ValueError, "Each relation must have two vertices")
    let u = integer(relation[0], "earlier vertex")
    let v = integer(relation[1], "later vertex")
    if u < 0 or u >= size or v < 0 or v >= size:
      raise newException(ValueError, "Relation vertex outside the poset")
    before[int(v)] = before[int(v)] or bit(int(u))
  for k in 0 ..< size:
    for v in 0 ..< size:
      if (before[v] and bit(k)) != 0: before[v] = before[v] or before[k]
  for v in 0 ..< size:
    if (before[v] and bit(v)) != 0:
      raise newException(ValueError, "Relations are cyclic")
    for u in vertices(before[v]): after[u] = after[u] or bit(v)
  let nodes = bit(size) - 1
  let count = countBlock(nodes, before, after)
  let dual = verifyBlock(count.proof, nodes, before, after)
  %*{"answer": $count.value,
     "solution": "We count permissible orderings by choosing a minimal remaining vertex. " &
                 "Independent blocks interleave and ordered blocks concatenate. " &
                 "The recurrence choosing maximal vertices gives the same count.",
     "verified": true, "method": "exact",
     "certificate": {"decomposition": count.proof, "forward_dp_states": count.states,
                     "dual_dp_states": dual.states}}
