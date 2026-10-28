## SentencePiece ModelProto inference in pure Nim, using only the standard library.
## Format references: google/sentencepiece sentencepiece_model.proto, normalizer.cc,
## bpe_model.cc and unigram_model.cc; darts-clone's documented uint32 unit layout.
## No Python, FFI, generated protobuf code, downloaded data, or external libraries.
import std/[algorithm, heapqueue, math, strutils, tables]

const
  SpaceMarker = "\xE2\x96\x81"
  Replacement = "\xEF\xBF\xBD"

type
  SpPieceKind* = enum
    spNormal = 1, spUnknown, spControl, spUserDefined, spUnused, spByte
  PieceTrieNode = object
    firstEdge, id: int
  PieceTrieEdge = object
    ch: char
    child, next: int
  PieceTrie = object
    nodes: seq[PieceTrieNode]
    edges: seq[PieceTrieEdge]
  Normalizer = object
    name: string
    addDummy, removeExtra, escapeSpaces: bool
    units: seq[uint32]
    replacements: string
  SentencePieceModel* = ref object
    pieces*: seq[string]
    kinds*: seq[SpPieceKind]
    bosId*, eosId*, unkId*, padId*: int
    modelType*: int
    scores: seq[float32]
    ids: Table[string, int]
    trie, userTrie: PieceTrie
    normalizer, denormalizer: Normalizer
    hasDenormalizer, suffix, byteFallback: bool
    byteIds: array[256, int]
    byteValues: seq[int]
    minScore, maxScore: float32
    unkSurface: string
  ProtoReader = object
    data: string
    pos: int
  TokenSpan = object
    start, size, id: int
  BpeSymbol = object
    start, size, prev, next: int
    frozen: bool
  Merge = object
    left, right, size: int
    score: float32

proc fail(message: string) {.noreturn.} =
  raise newException(ValueError, "[sentencepiece_native] " & message)

proc readVarint(r: var ProtoReader): uint64 =
  for i in 0 ..< 10:
    if r.pos >= r.data.len: fail("truncated protobuf varint")
    let b = uint64(ord(r.data[r.pos]))
    inc r.pos
    if i == 9 and b > 1: fail("protobuf varint overflow")
    result = result or ((b and 127) shl (7*i))
    if b < 128: return
  fail("invalid protobuf varint")

proc requireBytes(r: ProtoReader, size: int) =
  if size < 0 or size > r.data.len - r.pos: fail("truncated protobuf field")

proc fieldKey(r: var ProtoReader): tuple[number, wire: int] =
  let key = r.readVarint()
  if key shr 3 == 0 or key shr 3 > 536870911: fail("invalid protobuf field number")
  (int(key shr 3), int(key and 7))

proc fieldSize(r: var ProtoReader): int =
  let n = r.readVarint()
  if n > uint64(r.data.len - r.pos): fail("protobuf field length exceeds model")
  int(n)

proc readBytes(r: var ProtoReader): string =
  let n = r.fieldSize()
  result = r.data[r.pos ..< r.pos+n]
  r.pos += n

proc skip(r: var ProtoReader, wire: int) =
  var n = 0
  case wire
  of 0: discard r.readVarint(); return
  of 1: n = 8
  of 2: n = r.fieldSize()
  of 5: n = 4
  else: fail("unsupported protobuf wire type " & $wire)
  r.requireBytes(n)
  r.pos += n

proc little32(data: string, pos: int): uint32 =
  if pos < 0 or pos > data.len - 4: fail("truncated fixed32 value")
  for i in 0 ..< 4: result = result or (uint32(ord(data[pos+i])) shl (8*i))

proc readInt32(r: var ProtoReader): int =
  int(cast[int32](uint32(r.readVarint() and 0xffffffff'u64)))

proc defaultNormalizer(): Normalizer =
  Normalizer(addDummy: true, removeExtra: true, escapeSpaces: true)

proc parseNormalizer(data: string): Normalizer =
  result = defaultNormalizer()
  var r = ProtoReader(data: data)
  while r.pos < data.len:
    let (field, wire) = r.fieldKey()
    if wire == 2 and field == 1:
      result.name = r.readBytes()
    elif wire == 2 and field == 2:
      let blob = r.readBytes()
      if blob.len == 0: continue
      if blob.len < 4: fail("normalizer charsmap header is truncated")
      let n = int(little32(blob, 0))
      if n < 1024 or n mod 1024 != 0 or n > blob.len - 5:
        fail("invalid normalizer double-array size")
      result.units = newSeq[uint32](n div 4)
      for i in 0 ..< result.units.len: result.units[i] = little32(blob, 4 + 4*i)
      result.replacements = blob[4+n .. ^1]
      if result.replacements[^1] != '\0': fail("normalizer strings must end in NUL")
    elif wire == 0 and field in [3, 4, 5]:
      let enabled = r.readVarint() != 0
      case field
      of 3: result.addDummy = enabled
      of 4: result.removeExtra = enabled
      of 5: result.escapeSpaces = enabled
      else: discard
    else: r.skip(wire)

proc initTrie(): PieceTrie =
  result.nodes.add(PieceTrieNode(firstEdge: -1, id: -1))

proc child(t: PieceTrie, node: int, ch: char): int =
  var edge = t.nodes[node].firstEdge
  while edge >= 0:
    if t.edges[edge].ch == ch: return t.edges[edge].child
    edge = t.edges[edge].next
  -1

proc insert(t: var PieceTrie, piece: string, id: int) =
  var node = 0
  for ch in piece:
    var next = t.child(node, ch)
    if next < 0:
      next = t.nodes.len
      t.nodes.add(PieceTrieNode(firstEdge: -1, id: -1))
      let edge = t.edges.len
      t.edges.add(PieceTrieEdge(ch: ch, child: next, next: t.nodes[node].firstEdge))
      t.nodes[node].firstEdge = edge
    node = next
  t.nodes[node].id = id

proc prefix(t: PieceTrie, text: string, start: int): tuple[size, id: int] =
  result.id = -1
  if t.nodes.len == 0: return
  var node = 0
  for at in start ..< text.len:
    node = t.child(node, text[at])
    if node < 0: break
    if t.nodes[node].id >= 0: result = (at-start+1, t.nodes[node].id)

proc utf8Size(text: string, at: int): int =
  ## Zero means malformed UTF-8. Reject overlong encodings, surrogates, and >U+10FFFF.
  let a = ord(text[at])
  if a < 0x80: return 1
  var n: int
  if a in 0xc2 .. 0xdf: n = 2
  elif a in 0xe0 .. 0xef: n = 3
  elif a in 0xf0 .. 0xf4: n = 4
  else: return 0
  if n > text.len - at: return 0
  for i in 1 ..< n:
    if (ord(text[at+i]) and 0xc0) != 0x80: return 0
  let b = ord(text[at+1])
  if (a == 0xe0 and b < 0xa0) or (a == 0xed and b >= 0xa0) or
     (a == 0xf0 and b < 0x90) or (a == 0xf4 and b >= 0x90): return 0
  n

proc dartsOffset(unit: uint32): uint32 {.inline.} =
  (unit shr 10) shl int((unit and 512'u32) shr 6)

proc normalizePrefix(spec: Normalizer, users: PieceTrie, text: string,
                     start: int): tuple[text: string, consumed: int] =
  let user = users.prefix(text, start)
  if user.id >= 0: return (text[start ..< start+user.size], user.size)
  var longest = 0
  var value = -1
  if spec.units.len > 0:
    var node = dartsOffset(spec.units[0])
    for at in start ..< text.len:
      let label = uint32(ord(text[at]))
      let index = node xor label
      if index >= uint32(spec.units.len): break
      let unit = spec.units[int(index)]
      if (unit and 0x800000ff'u32) != label: break
      node = index xor dartsOffset(unit)
      if (unit and 256'u32) != 0:
        if node >= uint32(spec.units.len): fail("normalizer leaf outside double array")
        value = int(spec.units[int(node)] and 0x7fffffff'u32)
        longest = at-start+1
  if longest > 0:
    if value < 0 or value >= spec.replacements.len: fail("invalid normalizer replacement offset")
    var finish = value
    while finish < spec.replacements.len and spec.replacements[finish] != '\0': inc finish
    if finish == spec.replacements.len: fail("unterminated normalizer replacement")
    return (spec.replacements[value ..< finish], longest)
  let n = utf8Size(text, start)
  if n == 0: (Replacement, 1)
  else: (text[start ..< start+n], n)

proc normalize(spec: Normalizer, users: PieceTrie, text: string, suffix: bool): string =
  if text.len == 0: return
  var at = 0
  if spec.removeExtra:
    while at < text.len:
      let p = normalizePrefix(spec, users, text, at)
      if p.text != " ": break
      at += p.consumed
  if at == text.len: return
  let space = if spec.escapeSpaces: SpaceMarker else: " "
  if spec.addDummy and not suffix: result.add(space)
  var previousSpace = spec.removeExtra
  while at < text.len:
    let p = normalizePrefix(spec, users, text, at)
    var first = 0
    if previousSpace:
      while first < p.text.len and p.text[first] == ' ': inc first
    if first < p.text.len:
      for i in first ..< p.text.len:
        if spec.escapeSpaces and p.text[i] == ' ': result.add(SpaceMarker)
        else: result.add(p.text[i])
      previousSpace = spec.removeExtra and p.text[^1] == ' '
    at += p.consumed
  if spec.removeExtra:
    while result.endsWith(space): result.setLen(result.len-space.len)
  if spec.addDummy and suffix: result.add(space)

proc parsePiece(m: SentencePieceModel, data: string) =
  var r = ProtoReader(data: data)
  var piece = ""
  var score = 0'f32
  var kind = spNormal # proto2 default; type is commonly omitted for normal pieces.
  while r.pos < data.len:
    let (field, wire) = r.fieldKey()
    if field == 1 and wire == 2: piece = r.readBytes()
    elif field == 2 and wire == 5:
      r.requireBytes(4)
      score = cast[float32](little32(data, r.pos))
      r.pos += 4
      if classify(score) in {fcNan, fcInf, fcNegInf}: fail("non-finite piece score")
    elif field == 3 and wire == 0:
      let k = r.readVarint()
      if k < 1 or k > 6: fail("invalid vocabulary piece type")
      kind = SpPieceKind(k)
    else: r.skip(wire)
  if piece.len == 0: fail("empty vocabulary piece")
  if m.ids.hasKey(piece): fail("duplicate vocabulary piece: " & piece)
  m.ids[piece] = m.pieces.len
  m.pieces.add(piece)
  m.scores.add(score)
  m.kinds.add(kind)

proc byteValue(piece: string): int =
  if piece.len != 6 or not piece.startsWith("<0x") or piece[^1] != '>': return -1
  var value = 0
  for i in 3 .. 4:
    let ch = piece[i]
    if ch in {'0'..'9'}: value = value * 16 + ord(ch)-ord('0')
    elif ch in {'A'..'F'}: value = value * 16 + ord(ch)-ord('A')+10
    else: return -1
  value

proc parseSentencePieceModel*(data: string): SentencePieceModel =
  result = SentencePieceModel(ids: initTable[string, int](), modelType: 1,
    normalizer: defaultNormalizer(), unkSurface: " \xE2\x81\x87 ",
    bosId: -1, eosId: -1, unkId: -1, padId: -1,
    trie: initTrie(), userTrie: initTrie(), minScore: Inf, maxScore: NegInf)
  var bosPiece = "<s>"
  var eosPiece = "</s>"
  var padPiece = "<pad>"
  var r = ProtoReader(data: data)
  while r.pos < data.len:
    let (field, wire) = r.fieldKey()
    if field == 1 and wire == 2:
      result.parsePiece(r.readBytes())
    elif field == 2 and wire == 2:
      var trainer = ProtoReader(data: r.readBytes())
      while trainer.pos < trainer.data.len:
        let (f, w) = trainer.fieldKey()
        if w == 0 and f in [3, 24, 35]:
          let value = trainer.readVarint()
          case f
          of 3:
            if value notin [1'u64, 2'u64]: fail("expected a Unigram or BPE tokenizer model")
            result.modelType = int(value)
          of 24: result.suffix = value != 0
          of 35: result.byteFallback = value != 0
          else: discard
        elif w == 0 and f in [40, 41, 42, 43]:
          # Actual IDs are resolved from vocabulary kinds/names, as in the processor.
          discard trainer.readInt32()
        elif w == 2 and f in [44, 46, 47, 48]:
          let value = trainer.readBytes()
          case f
          of 44: result.unkSurface = value
          of 46: bosPiece = value
          of 47: eosPiece = value
          of 48: padPiece = value
          else: discard
        else: trainer.skip(w)
    elif field == 3 and wire == 2: result.normalizer = parseNormalizer(r.readBytes())
    elif field == 5 and wire == 2:
      result.denormalizer = parseNormalizer(r.readBytes())
      result.hasDenormalizer = result.denormalizer.units.len > 0
    else: r.skip(wire)
  if result.pieces.len == 0: fail("empty tokenizer model")
  result.byteValues = newSeq[int](result.pieces.len)
  for i in 0 ..< 256: result.byteIds[i] = -1
  for id, piece in result.pieces:
    result.byteValues[id] = -1
    case result.kinds[id]
    of spNormal, spUserDefined, spUnused:
      result.trie.insert(piece, id)
      if result.kinds[id] == spUserDefined: result.userTrie.insert(piece, id)
      if result.kinds[id] == spNormal:
        result.minScore = min(result.minScore, result.scores[id])
        result.maxScore = max(result.maxScore, result.scores[id])
    of spUnknown:
      if result.unkId >= 0: fail("multiple UNKNOWN vocabulary entries")
      result.unkId = id
    of spByte:
      let b = byteValue(piece)
      if not result.byteFallback or b < 0: fail("invalid BYTE vocabulary entry")
      result.byteIds[b] = id
      result.byteValues[id] = b
    of spControl: discard
  if result.unkId < 0: fail("model has no UNKNOWN entry")
  for pair in [(bosPiece, 0), (eosPiece, 1), (padPiece, 2)]:
    let id = result.ids.getOrDefault(pair[0], -1)
    if id >= 0 and result.kinds[id] == spControl:
      case pair[1]
      of 0: result.bosId = id
      of 1: result.eosId = id
      of 2: result.padId = id
      else: discard
  if result.byteFallback:
    for id in result.byteIds:
      if id < 0: fail("byte_fallback requires all 256 byte pieces")
  if result.minScore == Inf: result.minScore = 0
  if result.maxScore == NegInf: result.maxScore = 0

proc loadSentencePieceModel*(path: string): SentencePieceModel =
  parseSentencePieceModel(readFile(path))

proc normalizeSentencePiece*(m: SentencePieceModel, text: string): string =
  normalize(m.normalizer, m.userTrie, text, m.suffix)

proc pieceId*(m: SentencePieceModel, piece: string): int =
  m.ids.getOrDefault(piece, -1)

proc normalizerName*(m: SentencePieceModel): string = m.normalizer.name

proc `<`(a, b: Merge): bool =
  if a.score == b.score: a.left < b.left
  else: a.score > b.score # heapqueue is a min heap; highest merge score wins.

proc bpeSpans(m: SentencePieceModel, input: string): seq[TokenSpan] =
  var symbols: seq[BpeSymbol]
  var at = 0
  while at < input.len:
    let user = m.userTrie.prefix(input, at)
    let n = if user.id >= 0: user.size else: utf8Size(input, at)
    if n <= 0: fail("normalizer emitted malformed UTF-8")
    let index = symbols.len
    symbols.add(BpeSymbol(start: at, size: n, prev: index-1, next: index+1,
                          frozen: user.id >= 0))
    at += n
  if symbols.len == 0: return
  symbols[^1].next = -1
  var agenda = initHeapQueue[Merge]()
  var reverseMerges = initTable[int, tuple[left, right: string]]()
  proc offer(left, right: int) =
    if left < 0 or right < 0 or symbols[left].frozen or symbols[right].frozen: return
    let start = symbols[left].start
    let size = symbols[left].size + symbols[right].size
    let id = m.pieceId(input[start ..< start+size])
    if id < 0 or m.kinds[id] notin {spNormal, spUserDefined, spUnused}: return
    agenda.push(Merge(left: left, right: right, size: size, score: m.scores[id]))
    if m.kinds[id] == spUnused:
      reverseMerges[id] = (input[start ..< start+symbols[left].size],
                          input[symbols[right].start ..< start+size])
  for i in 1 ..< symbols.len: offer(i-1, i)
  while agenda.len > 0:
    let merge = agenda.pop()
    let left = merge.left
    let right = merge.right
    if symbols[left].size == 0 or symbols[right].size == 0 or
       symbols[left].next != right or symbols[left].size + symbols[right].size != merge.size:
      continue
    symbols[left].size = merge.size
    symbols[left].next = symbols[right].next
    if symbols[right].next >= 0: symbols[symbols[right].next].prev = left
    symbols[right].size = 0
    offer(symbols[left].prev, left)
    offer(left, symbols[left].next)
  var index = 0
  while index >= 0:
    var pending = @[TokenSpan(start: symbols[index].start, size: symbols[index].size)]
    while pending.len > 0:
      var span = pending.pop()
      let piece = input[span.start ..< span.start+span.size]
      span.id = m.ids.getOrDefault(piece, m.unkId)
      if m.kinds[span.id] == spUnused and reverseMerges.hasKey(span.id):
        let halves = reverseMerges[span.id]
        pending.add(TokenSpan(start: span.start+halves.left.len, size: halves.right.len))
        pending.add(TokenSpan(start: span.start, size: halves.left.len))
      else: result.add(span)
    index = symbols[index].next

proc unigramSpans(m: SentencePieceModel, input: string): seq[TokenSpan] =
  let n = input.len
  var best = newSeq[float32](n+1)
  var previous = newSeq[int](n+1)
  var ids = newSeq[int](n+1)
  for i in 0 .. n: previous[i] = -1
  previous[0] = 0
  var start = 0
  while start < n:
    let width = utf8Size(input, start)
    if width <= 0: fail("normalizer emitted malformed UTF-8")
    if previous[start] >= 0:
      var node = 0
      var at = start
      var hasSingle = false
      while at < n:
        node = m.trie.child(node, input[at])
        if node < 0: break
        inc at
        let id = m.trie.nodes[node].id
        if id < 0 or m.kinds[id] == spUnused: continue
        let size = at-start
        let score = if m.kinds[id] == spUserDefined:
                      float32(float64(size) * float64(m.maxScore) - 0.1)
                    else: m.scores[id]
        let candidate = best[start] + score
        if previous[at] < 0 or candidate > best[at]:
          best[at] = candidate
          previous[at] = start
          ids[at] = id
        if size == width: hasSingle = true
      if not hasSingle:
        let finish = start+width
        let candidate = best[start] + (m.minScore - 10'f32)
        if previous[finish] < 0 or candidate > best[finish]:
          best[finish] = candidate
          previous[finish] = start
          ids[finish] = m.unkId
    start += width
  var finish = n
  while finish > 0:
    let first = previous[finish]
    if first < 0 or first >= finish: fail("no unigram segmentation path")
    result.add(TokenSpan(start: first, size: finish-first, id: ids[finish]))
    finish = first
  result.reverse()

proc encodeSentencePiece*(m: SentencePieceModel, text: string): seq[int] =
  let input = m.normalizeSentencePiece(text)
  if input.len == 0: return
  let spans = if m.modelType == 2: m.bpeSpans(input) else: m.unigramSpans(input)
  var previousUnknown = false
  for span in spans:
    let unknown = span.id == m.unkId
    if unknown and m.byteFallback:
      for i in span.start ..< span.start+span.size: result.add(m.byteIds[ord(input[i])])
    elif not unknown or not previousUnknown: result.add(span.id)
    previousUnknown = unknown

proc appendValidUtf8(target: var string, bytes: string) =
  var at = 0
  while at < bytes.len:
    let n = utf8Size(bytes, at)
    if n == 0:
      target.add(Replacement)
      inc at
    else:
      target.add(bytes[at ..< at+n])
      at += n

proc decodeSentencePiece*(m: SentencePieceModel, ids: openArray[int]): string =
  var bytes = ""
  var bosWhitespaceConsumed = false
  for id in ids:
    if id < 0 or id >= m.pieces.len: fail("token ID outside vocabulary: " & $id)
    if m.kinds[id] == spByte:
      bytes.add(char(m.byteValues[id]))
      continue
    result.appendValidUtf8(bytes)
    bytes.setLen(0)
    if m.kinds[id] == spControl: continue
    if m.kinds[id] == spUnknown:
      result.add(m.unkSurface)
      continue
    var piece = m.pieces[id]
    if result.len == 0 and not bosWhitespaceConsumed and
       (m.normalizer.addDummy or m.normalizer.removeExtra) and piece.startsWith(SpaceMarker):
      piece = piece[SpaceMarker.len .. ^1]
      bosWhitespaceConsumed = not m.normalizer.removeExtra
    result.add(piece.replace(SpaceMarker, " "))
  result.appendValidUtf8(bytes)
  if m.hasDenormalizer: result = normalize(m.denormalizer, PieceTrie(), result, false)
