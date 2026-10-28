## Local Hugging Face BPE tokenizer inference using Nim's standard library only.
## Loads vocabulary, ranked merges, added tokens, pre-tokenizers, decoders and
## the model's chat template. No embedded Python, FFI or tokenizer library.
import std/[heapqueue, json, os, strutils, tables, unicode]
import tokenizer_unicode, tokenizer_regex, tokenizer_chat_template
import ../codestral/sentencepiece_native

type
  Stage = ref object
    kind, behavior, replacement, scheme: string
    children: seq[Stage]
    regex: TokenizerRegex
    literal: string
    invert, individual, prefixSpace, useRegex: bool
    startCount, stopCount: int
  AddedToken = object
    id: int
    content: string
    singleWord, lstrip, rstrip, normalized, special: bool
  AddedNode = object
    edges: Table[char, int]
    token: int
  TextPart = object
    text: string
    id: int
  MergeRule = tuple[rank, id: int]
  Symbol = object
    id, previous, next, size: int
  AgendaItem = object
    rank, left, right, id, size: int
  NativeBpe = ref object
    pieces: seq[string]
    modelIds, ids: Table[string, int]
    rules: Table[(int, int), MergeRule]
    added: seq[AddedToken]
    rawTrie, normalizedTrie: seq[AddedNode]
    special: seq[bool]
    byteSymbols: array[256, string]
    byteValues: Table[int, int]
    preTokenizer, normalizer, decoder: Stage
    postProcessor: JsonNode
    unkId: int
    prefix, suffix: string
    byteFallback, fuseUnknown, ignoreMerges: bool
  HFTokenizer* = object
    eosId*, bosId*, padId*, vocabSize*, totalVocabSize*: int
    bpe: NativeBpe
    sentencePiece: SentencePieceModel
    config: JsonNode
    templateProgram: ChatTemplate
    bosToken, eosToken, padToken: string
    addBos, addEos, overrideProcessor, cleanup: bool

proc bad(message: string) {.noreturn.} =
  raise newException(ValueError, "[native HF tokenizer] " & message)
proc field(node: JsonNode, name: string): JsonNode =
  if not node.isNil and node.kind == JObject and node.hasKey(name): node[name] else: newJNull()
proc text(node: JsonNode, name: string, fallback = ""): string =
  let v = field(node, name)
  if v.kind == JString: v.getStr() else: fallback
proc flag(node: JsonNode, name: string, fallback = false): bool =
  let v = field(node, name)
  if v.kind == JBool: v.getBool() else: fallback
proc tokenContent(node: JsonNode): string =
  if node.kind == JString: node.getStr() else: text(node, "content")

const Gpt2Pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

proc parseStage(node: JsonNode, purpose: string): Stage =
  if node.kind == JNull: return nil
  result = Stage(kind: text(node, "type"), behavior: text(node, "behavior", "Isolated"),
                 invert: flag(node, "invert"), individual: flag(node, "individual_digits"),
                 prefixSpace: flag(node, "add_prefix_space", true), useRegex: flag(node, "use_regex", true),
                 replacement: text(node, "replacement", "▁"), scheme: text(node, "prepend_scheme", "always"))
  case result.kind
  of "Sequence":
    let name = if purpose == "normalizer": "normalizers" elif purpose == "decoder": "decoders" else: "pretokenizers"
    let parts = field(node, name)
    if parts.kind != JArray: bad("Sequence requires " & name)
    for item in parts: result.children.add(parseStage(item, purpose))
  of "Split", "Replace":
    let pattern = field(node, "pattern")
    if field(pattern, "Regex").kind == JString:
      result.regex = compileTokenizerRegex(pattern["Regex"].getStr())
    elif field(pattern, "String").kind == JString: result.literal = pattern["String"].getStr()
    else: bad("Split/Replace requires a String or Regex pattern")
    if result.kind == "Replace": result.replacement = text(node, "content")
  of "ByteLevel":
    if purpose == "pretokenizer" and result.useRegex: result.regex = compileTokenizerRegex(Gpt2Pattern)
  of "Digits", "Punctuation", "Whitespace", "WhitespaceSplit", "Metaspace":
    if result.kind == "Whitespace": result.regex = compileTokenizerRegex("\\w+|[^\\w\\s]+")
    if result.kind == "Punctuation": result.regex = compileTokenizerRegex("\\p{P}")
    if result.kind == "Metaspace" and node.hasKey("add_prefix_space"):
      result.scheme = if flag(node, "add_prefix_space"): "always" else: "never"
  of "Lowercase", "Strip", "Prepend":
    if result.kind == "Strip":
      if purpose == "decoder":
        result.replacement = text(node, "content", " ")
        result.startCount = field(node, "start").getInt()
        result.stopCount = field(node, "stop").getInt()
        if result.replacement.len == 0 or result.startCount < 0 or result.stopCount < 0:
          bad("invalid Strip decoder")
      else:
        result.prefixSpace = flag(node, "strip_left", true)
        result.useRegex = flag(node, "strip_right", true)
    if result.kind == "Prepend": result.replacement = text(node, "prepend")
  of "Fuse", "ByteFallback", "BPEDecoder":
    if result.kind == "BPEDecoder": result.replacement = text(node, "suffix", "</w>")
  else: bad("unsupported " & purpose & " stage: " & result.kind)
  if purpose == "normalizer" and result.kind notin ["Sequence", "Lowercase", "Strip", "Prepend", "Replace"]:
    bad("unsupported normalizer stage: " & result.kind)
  if purpose == "pretokenizer" and result.kind notin ["Sequence", "Split", "ByteLevel", "Digits", "Punctuation", "Whitespace", "WhitespaceSplit", "Metaspace"]:
    bad("unsupported pre-tokenizer stage: " & result.kind)
  if purpose == "decoder" and result.kind notin ["Sequence", "ByteLevel", "Metaspace", "Replace", "Fuse", "ByteFallback", "Strip", "BPEDecoder"]:
    bad("unsupported decoder stage: " & result.kind)

proc matches(stage: Stage, value: string): seq[RegexMatch] =
  if not stage.regex.isNil: return findTokenizerMatches(stage.regex, value)
  if stage.literal.len == 0: bad("empty literal split pattern")
  var at = 0
  while at < value.len:
    let found = value.find(stage.literal, at)
    if found < 0: break
    result.add((found, found+stage.literal.len))
    at = found+stage.literal.len

proc normalize(stage: Stage, value: string): string =
  if stage.isNil: return value
  case stage.kind
  of "Sequence":
    result = value
    for child in stage.children: result = normalize(child, result)
  of "Lowercase": result = unicodeLower(value)
  of "Strip": result = unicodeStrip(value, leading = stage.prefixSpace, trailing = stage.useRegex)
  of "Prepend": result = stage.replacement & value
  of "Replace":
    var at = 0
    for match in stage.matches(value):
      result.add(value[at..<match.first]); result.add(stage.replacement); at = match.last
    result.add(value[at..^1])
  else: bad("invalid normalizer operation")

proc splitAt(stage: Stage, value: string, delimiters: seq[RegexMatch]): seq[string] =
  var spans: seq[tuple[first, last: int, delimiter: bool]]
  var at = 0
  for match in delimiters:
    if match.first > at: spans.add((at, match.first, stage.invert))
    spans.add((match.first, match.last, not stage.invert))
    at = match.last
  if at < value.len: spans.add((at, value.len, stage.invert))
  var nextPrefix = ""
  var previousDelimiter = false
  for span in spans:
    let part = value[span.first..<span.last]
    if not span.delimiter:
      result.add(nextPrefix & part)
      nextPrefix = ""
    else:
      case stage.behavior
      of "Isolated": result.add(part)
      of "Removed": discard
      of "MergedWithPrevious":
        if result.len > 0: result[^1].add(part)
        else: result.add(part)
      of "MergedWithNext": nextPrefix.add(part)
      of "Contiguous":
        if previousDelimiter and result.len > 0: result[^1].add(part)
        else: result.add(part)
      else: bad("unsupported split behavior: " & stage.behavior)
    previousDelimiter = span.delimiter
  if nextPrefix.len > 0: result.add(nextPrefix)

proc preTokenize(stage: Stage, pieces: seq[string], model: NativeBpe): seq[string] =
  if stage.isNil: return pieces
  if stage.kind == "Sequence":
    result = pieces
    for child in stage.children: result = preTokenize(child, result, model)
    return
  for index, original in pieces:
    if original.len == 0: continue
    case stage.kind
    of "Split", "Punctuation": result.add(splitAt(stage, original, stage.matches(original)))
    of "Whitespace":
      for match in stage.matches(original): result.add(original[match.first..<match.last])
    of "Digits", "WhitespaceSplit":
      let chars = runeSpans(original)
      var marks: seq[RegexMatch]
      var at = 0
      while at < chars.len:
        let yes = if stage.kind == "Digits": unicodeCategory(chars[at].cp, "N") else: unicodeSpace(chars[at].cp)
        if not yes: inc at; continue
        let first = at
        inc at
        if not stage.individual or stage.kind == "WhitespaceSplit":
          while at < chars.len:
            let same = if stage.kind == "Digits": unicodeCategory(chars[at].cp, "N") else: unicodeSpace(chars[at].cp)
            if not same: break
            inc at
        marks.add((chars[first].start, if at == chars.len: original.len else: chars[at].start))
      if stage.kind == "WhitespaceSplit":
        let remove = Stage(behavior: "Removed")
        result.add(splitAt(remove, original, marks))
      else: result.add(splitAt(stage, original, marks))
    of "ByteLevel":
      var value = original
      if stage.prefixSpace and not value.startsWith(" "): value = " " & value
      let chunks = if stage.useRegex: stage.matches(value) else: @[(first: 0, last: value.len)]
      for chunk in chunks:
        var encoded = newStringOfCap((chunk.last-chunk.first)*2)
        for at in chunk.first..<chunk.last: encoded.add(model.byteSymbols[ord(value[at])])
        result.add(encoded)
    of "Metaspace":
      var value = original.replace(" ", stage.replacement)
      if stage.scheme != "never" and (stage.scheme == "always" or index == 0) and not value.startsWith(stage.replacement):
        value = stage.replacement & value
      var first = 0
      var at = value.find(stage.replacement, stage.replacement.len)
      while at >= 0:
        result.add(value[first..<at]); first = at
        at = value.find(stage.replacement, at+stage.replacement.len)
      if first < value.len: result.add(value[first..^1])
    else: bad("invalid pre-tokenizer operation")

proc buildByteAlphabet(model: NativeBpe) =
  model.byteValues = initTable[int, int]()
  var next = 256
  for b in 0..<256:
    var cp = b
    if b notin 33..126 and b notin 161..172 and b notin 174..255:
      cp = next; inc next
    model.byteSymbols[b] = $Rune(cp)
    model.byteValues[cp] = b

proc `<`(a, b: AgendaItem): bool =
  if a.rank == b.rank: a.left < b.left else: a.rank < b.rank

proc encodeWord(model: NativeBpe, word: string): seq[int] =
  if model.ignoreMerges and model.modelIds.hasKey(word): return @[model.modelIds[word]]
  var symbols: seq[Symbol]
  let chars = runeSpans(word)
  proc appendId(id: int) =
    if id < 0: return # BPE without unk_token drops characters outside its alphabet.
    if model.fuseUnknown and id == model.unkId and symbols.len > 0 and symbols[^1].id == id: return
    let i = symbols.len
    symbols.add(Symbol(id: id, previous: i-1, next: i+1, size: 1))
  for i, ch in chars:
    var piece = word[ch.start..<ch.finish]
    if i > 0: piece = model.prefix & piece
    if i == chars.len-1: piece.add(model.suffix)
    let id = model.modelIds.getOrDefault(piece, -1)
    if id >= 0: appendId(id)
    elif model.byteFallback:
      for b in piece:
        let name = "<0x" & toHex(ord(b), 2) & ">"
        let byteId = model.modelIds.getOrDefault(name, -1)
        if byteId < 0: bad("byte fallback entry missing: " & name)
        appendId(byteId)
    else: appendId(model.unkId)
  if symbols.len == 0: return
  symbols[^1].next = -1
  var agenda = initHeapQueue[AgendaItem]()
  proc offer(left, right: int) =
    if left < 0 or right < 0: return
    let key = (symbols[left].id, symbols[right].id)
    if not model.rules.hasKey(key): return
    let rule = model.rules[key]
    agenda.push(AgendaItem(rank: rule.rank, id: rule.id, left: left, right: right,
                          size: symbols[left].size+symbols[right].size))
  for i in 1..<symbols.len: offer(i-1, i)
  while agenda.len > 0:
    let item = agenda.pop()
    let left = item.left
    let right = item.right
    if symbols[left].size == 0 or symbols[right].size == 0 or symbols[left].next != right or
       symbols[left].size+symbols[right].size != item.size: continue
    # A stale pair may retain its length after another merge; recheck its rank/ID.
    let rule = model.rules.getOrDefault((symbols[left].id, symbols[right].id), (rank: -1, id: -1))
    if rule.rank != item.rank or rule.id != item.id: continue
    symbols[left].id = item.id
    symbols[left].size = item.size
    symbols[left].next = symbols[right].next
    if symbols[right].next >= 0: symbols[symbols[right].next].previous = left
    symbols[right].size = 0
    offer(symbols[left].previous, left)
    offer(left, symbols[left].next)
  var index = 0
  while index >= 0:
    result.add(symbols[index].id)
    index = symbols[index].next

proc insert(trie: var seq[AddedNode], content: string, token: int) =
  if trie.len == 0: trie.add(AddedNode(token: -1))
  var node = 0
  for ch in content:
    if not trie[node].edges.hasKey(ch):
      let next = trie.len
      trie[node].edges[ch] = next
      trie.add(AddedNode(token: -1))
    node = trie[node].edges[ch]
  trie[node].token = token

proc before(text: string, at: int): int =
  if at <= 0: return -1
  var first = at-1
  while first > 0 and (ord(text[first]) and 0xc0) == 0x80: dec first
  int(runeAt(text, first))
proc after(text: string, at: int): int =
  if at >= text.len: -1 else: int(runeAt(text, at))

proc extract(model: NativeBpe, trie: seq[AddedNode], value: string): seq[TextPart] =
  if trie.len == 0: return @[TextPart(text: value, id: -1)]
  var at = 0
  var plain = 0
  while at < value.len:
    var node = 0
    var scan = at
    var token = -1
    var finish = at
    while scan < value.len and trie[node].edges.hasKey(value[scan]):
      node = trie[node].edges[value[scan]]; inc scan
      let index = trie[node].token
      if index >= 0:
        let t = model.added[index]
        if not t.singleWord or
           ((at == 0 or not unicodeCategory(before(value, at), "word")) and
            (scan == value.len or not unicodeCategory(after(value, scan), "word"))):
          token = index; finish = scan
    if token < 0:
      at += max(1, utf8Width(value, at))
      continue
    let added = model.added[token]
    var first = at
    if added.lstrip:
      while first > plain and unicodeSpace(before(value, first)):
        dec first
        while first > plain and (ord(value[first]) and 0xc0) == 0x80: dec first
    if added.rstrip:
      while finish < value.len and unicodeSpace(after(value, finish)): finish += utf8Width(value, finish)
    if first > plain: result.add(TextPart(text: value[plain..<first], id: -1))
    result.add(TextPart(id: added.id))
    at = finish; plain = finish
  if plain < value.len: result.add(TextPart(text: value[plain..^1], id: -1))

proc ensurePiece(model: NativeBpe, id: int, content: string) =
  if id < 0 or id > 10000000: bad("invalid vocabulary ID")
  if model.pieces.len <= id: model.pieces.setLen(id+1)
  if model.pieces[id].len > 0 and model.pieces[id] != content: bad("conflicting vocabulary ID " & $id)
  model.pieces[id] = content
  model.ids[content] = id
proc registerAdded(model: NativeBpe, spec: JsonNode, special = false) =
  let content = tokenContent(spec)
  if content.len == 0: return
  var id = model.ids.getOrDefault(content, -1)
  if field(spec, "id").kind == JInt: id = spec["id"].getInt()
  if id < 0: id = model.pieces.len
  model.ensurePiece(id, content)
  let token = AddedToken(id: id, content: content, special: special or flag(spec, "special"),
      normalized: flag(spec, "normalized", not special), singleWord: flag(spec, "single_word"),
      lstrip: flag(spec, "lstrip"), rstrip: flag(spec, "rstrip"))
  for existing in model.added.mitems:
    if existing.content == content:
      existing.special = existing.special or token.special
      if special: existing.normalized = token.normalized
      existing.singleWord = token.singleWord
      existing.lstrip = token.lstrip; existing.rstrip = token.rstrip
      return
  model.added.add(token)

proc buildAddedTries(model: NativeBpe) =
  model.special = newSeq[bool](model.pieces.len)
  for i, token in model.added:
    if token.special: model.special[token.id] = true
    if token.normalized:
      let normalized = normalize(model.normalizer, token.content)
      model.normalizedTrie.insert(normalized, i)
    else: model.rawTrie.insert(token.content, i)

proc tokenId(tok: HFTokenizer, content: string): int =
  if content.len == 0: return -1
  if not tok.bpe.isNil: return tok.bpe.ids.getOrDefault(content, -1)
  tok.sentencePiece.pieceId(content)

proc newHFTokenizer*(modelPath: string): HFTokenizer =
  let configPath = modelPath / "tokenizer_config.json"
  result.config = if fileExists(configPath): parseFile(configPath) else: newJObject()
  let path = modelPath / "tokenizer.json"
  if fileExists(path):
    let root = parseFile(path)
    let spec = field(root, "model")
    if text(spec, "type") != "BPE": bad("tokenizer.json model type must be BPE; use tokenizer.model for SentencePiece")
    if field(spec, "dropout").kind notin {JNull} and field(spec, "dropout").getFloat() != 0:
      bad("inference tokenizer must have BPE dropout disabled")
    let m = NativeBpe(modelIds: initTable[string, int](), ids: initTable[string, int](),
      rules: initTable[(int, int), MergeRule](), prefix: text(spec, "continuing_subword_prefix"),
      suffix: text(spec, "end_of_word_suffix"), byteFallback: flag(spec, "byte_fallback"),
      fuseUnknown: flag(spec, "fuse_unk"), ignoreMerges: flag(spec, "ignore_merges"))
    let vocabulary = field(spec, "vocab")
    if vocabulary.kind != JObject or vocabulary.len == 0: bad("missing BPE vocabulary")
    for content, node in vocabulary:
      if node.kind != JInt: bad("invalid BPE vocabulary entry")
      let id = node.getInt()
      m.ensurePiece(id, content); m.modelIds[content] = id
    result.vocabSize = vocabulary.len
    m.unkId = m.modelIds.getOrDefault(text(spec, "unk_token"), -1)
    let merges = field(spec, "merges")
    if merges.kind != JArray: bad("missing BPE merges")
    for rank in 0..<merges.len:
      let merge = merges[rank]
      var left, right: string
      if merge.kind == JString:
        let value = merge.getStr()
        let gap = value.find(' ')
        if gap < 0: bad("invalid BPE merge")
        left = value[0..<gap]; right = value[gap+1..^1]
      elif merge.kind == JArray and merge.len == 2:
        left = merge[0].getStr(); right = merge[1].getStr()
      else: bad("invalid BPE merge")
      let joined = left & (if m.prefix.len > 0 and right.startsWith(m.prefix): right[m.prefix.len..^1] else: right)
      if not m.modelIds.hasKey(left) or not m.modelIds.hasKey(right) or not m.modelIds.hasKey(joined):
        bad("merge references a missing vocabulary entry")
      m.rules[(m.modelIds[left], m.modelIds[right])] = (rank, m.modelIds[joined])
    m.normalizer = parseStage(field(root, "normalizer"), "normalizer")
    m.preTokenizer = parseStage(field(root, "pre_tokenizer"), "pretokenizer")
    m.decoder = parseStage(field(root, "decoder"), "decoder")
    m.postProcessor = field(root, "post_processor")
    let added = field(root, "added_tokens")
    if added.kind == JArray:
      for entry in added: m.registerAdded(entry)
    let configured = field(result.config, "added_tokens_decoder")
    if configured.kind == JObject:
      for key, value in configured:
        var entry = value.copy()
        entry["id"] = %parseInt(key)
        m.registerAdded(entry)
    for name in ["bos_token", "eos_token", "pad_token", "unk_token", "sep_token", "cls_token", "mask_token"]:
      let value = field(result.config, name)
      if value.kind != JNull: m.registerAdded(value, true)
    let extra = field(result.config, "additional_special_tokens")
    if extra.kind == JArray:
      for value in extra: m.registerAdded(value, true)
    m.buildAddedTries()
    m.buildByteAlphabet()
    result.bpe = m
    result.totalVocabSize = m.pieces.len
  else:
    var spPath = ""
    for name in ["tokenizer.model", "tokenizer.model.v3"]:
      if fileExists(modelPath/name): spPath = modelPath/name; break
    if spPath.len == 0: bad("tokenizer.json/tokenizer.model not found in " & modelPath)
    result.sentencePiece = loadSentencePieceModel(spPath)
    result.vocabSize = result.sentencePiece.pieces.len
    result.totalVocabSize = result.vocabSize
  result.bosToken = tokenContent(field(result.config, "bos_token"))
  result.eosToken = tokenContent(field(result.config, "eos_token"))
  result.padToken = tokenContent(field(result.config, "pad_token"))
  result.bosId = result.tokenId(result.bosToken)
  result.eosId = result.tokenId(result.eosToken)
  result.padId = result.tokenId(result.padToken)
  if not result.sentencePiece.isNil:
    if result.bosId < 0: result.bosId = result.sentencePiece.bosId
    if result.eosId < 0: result.eosId = result.sentencePiece.eosId
    if result.padId < 0: result.padId = result.sentencePiece.padId
  let llama = "LlamaTokenizer" in text(result.config, "tokenizer_class")
  result.overrideProcessor = llama
  result.addBos = flag(result.config, "add_bos_token", llama)
  result.addEos = flag(result.config, "add_eos_token")
  result.cleanup = flag(result.config, "clean_up_tokenization_spaces")
  var templateText = text(result.config, "chat_template")
  let templates = field(result.config, "chat_template")
  if templates.kind == JArray:
    for entry in templates:
      if text(entry, "name") == "default": templateText = text(entry, "template")
  if fileExists(modelPath/"chat_template.jinja"): templateText = readFile(modelPath/"chat_template.jinja")
  if templateText.len > 0: result.templateProgram = compileChatTemplate(templateText)

proc postProcess(tok: HFTokenizer, ids: seq[int], processor: JsonNode): seq[int] =
  let kind = text(processor, "type")
  case kind
  of "", "ByteLevel": return ids
  of "Sequence":
    result = ids
    let list = field(processor, "processors")
    if list.kind != JArray: bad("invalid post-processor sequence")
    for child in list: result = postProcess(tok, result, child)
  of "TemplateProcessing":
    let parts = field(processor, "single")
    if parts.kind != JArray: bad("post-processor single template must be an array")
    for part in parts:
      if field(part, "Sequence").kind != JNull: result.add(ids)
      elif field(part, "SpecialToken").kind != JNull:
        let name = text(part["SpecialToken"], "id")
        let special = field(field(processor, "special_tokens"), name)
        let values = field(special, "ids")
        if values.kind != JArray: bad("missing post-processor token: " & name)
        for value in values: result.add(value.getInt())
      else: bad("unsupported post-processor template item")
  of "BertProcessing", "RobertaProcessing":
    let cls = field(processor, "cls")
    let sep = field(processor, "sep")
    if cls.kind != JArray or sep.kind != JArray: bad("invalid CLS/SEP post-processor")
    result = @[cls[1].getInt()] & ids & @[sep[1].getInt()]
  else: bad("unsupported post-processor: " & kind)

proc encode*(tok: HFTokenizer, text: string, addSpecialTokens = true): seq[int] =
  if not tok.sentencePiece.isNil: result = tok.sentencePiece.encodeSentencePiece(text)
  else:
    let m = tok.bpe
    for raw in m.extract(m.rawTrie, validUtf8(text)):
      if raw.id >= 0: result.add(raw.id); continue
      let normalized = normalize(m.normalizer, raw.text)
      for part in m.extract(m.normalizedTrie, normalized):
        if part.id >= 0: result.add(part.id); continue
        for word in preTokenize(m.preTokenizer, @[part.text], m): result.add(m.encodeWord(word))
  if addSpecialTokens:
    if tok.overrideProcessor or not tok.sentencePiece.isNil:
      if tok.addBos:
        if tok.bosId < 0: bad("add_bos_token=true but BOS token is missing")
        result.insert(tok.bosId, 0)
      if tok.addEos:
        if tok.eosId < 0: bad("add_eos_token=true but EOS token is missing")
        result.add(tok.eosId)
    else: result = postProcess(tok, result, tok.bpe.postProcessor)

proc decodeStage(stage: Stage, parts: seq[string], m: NativeBpe): seq[string] =
  if stage.isNil: return parts
  case stage.kind
  of "Sequence":
    result = parts
    for child in stage.children: result = decodeStage(child, result, m)
  of "ByteLevel":
    var bytes = ""
    for part in parts:
      for r in part.runes:
        if m.byteValues.hasKey(int(r)): bytes.add(char(m.byteValues[int(r)]))
        else: bytes.add($r)
    result = @[validUtf8(bytes)]
  of "Fuse": result = @[parts.join("")]
  of "Metaspace":
    for i, part in parts:
      var value = part.replace(stage.replacement, " ")
      if i == 0 and stage.scheme != "never" and value.startsWith(" "): value = value[1..^1]
      result.add(value)
  of "Replace":
    for part in parts: result.add(normalize(stage, part))
  of "ByteFallback":
    var bytes = ""
    for part in parts:
      if part.len == 6 and part.startsWith("<0x") and part.endsWith(">"):
        try: bytes.add(char(parseHexInt(part[3..4])))
        except ValueError: bad("invalid byte fallback token")
      else:
        if bytes.len > 0: result.add(validUtf8(bytes)); bytes = ""
        result.add(part)
    if bytes.len > 0: result.add(validUtf8(bytes))
  of "Strip":
    for part in parts:
      var value = part
      for i in 0..<stage.startCount:
        if not value.startsWith(stage.replacement): break
        value = value[stage.replacement.len..^1]
      for i in 0..<stage.stopCount:
        if not value.endsWith(stage.replacement): break
        value.setLen(value.len-stage.replacement.len)
      result.add(value)
  of "BPEDecoder":
    var value = parts.join("").replace(stage.replacement, " ")
    if value.endsWith(" "): value.setLen(value.len-1)
    result = @[value]
  else: bad("unsupported decoder")

proc decode*(tok: HFTokenizer, ids: seq[int], skipSpecialTokens = true): string =
  if not tok.sentencePiece.isNil: return tok.sentencePiece.decodeSentencePiece(ids)
  var parts: seq[string]
  for id in ids:
    if id < 0 or id >= tok.bpe.pieces.len: continue
    if skipSpecialTokens and tok.bpe.special[id]: continue
    parts.add(tok.bpe.pieces[id])
  result = decodeStage(tok.bpe.decoder, parts, tok.bpe).join("")
  if tok.cleanup:
    for pair in [(" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"),
                 (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")]:
      result = result.replace(pair[0], pair[1])

type HFTokenStream* = object
  tokenizer: HFTokenizer
  bytes: string
  ids: seq[int]
  emitted: string
  byteStream: bool

proc initTokenStream*(tok: HFTokenizer): HFTokenStream =
  result.tokenizer = tok
  result.byteStream = not tok.bpe.isNil and not tok.bpe.decoder.isNil and
                      tok.bpe.decoder.kind == "ByteLevel" and not tok.cleanup

proc pushTokens*(stream: var HFTokenStream, ids: openArray[int], final = false): string =
  let tok = stream.tokenizer
  if stream.byteStream:
    for id in ids:
      if id < 0 or id >= tok.bpe.pieces.len or tok.bpe.special[id]: continue
      for rune in tok.bpe.pieces[id].runes:
        if tok.bpe.byteValues.hasKey(int(rune)):
          stream.bytes.add(char(tok.bpe.byteValues[int(rune)]))
        else: stream.bytes.add($rune)
    return drainUtf8(stream.bytes, final)
  stream.ids.add(ids)
  let full = tok.decode(stream.ids)
  if not final and full.endsWith("\xef\xbf\xbd"): return ""
  if not full.startsWith(stream.emitted):
    bad("decoder rewrote text already emitted by the stream")
  result = full[stream.emitted.len..^1]
  stream.emitted = full

proc formatChatTemplate*(tok: HFTokenizer, messages: JsonNode, addGenerationPrompt = true): string =
  if tok.templateProgram.isNil: bad("model has no supported chat_template")
  var env = initTable[string, JsonNode]()
  env["messages"] = messages
  env["add_generation_prompt"] = %addGenerationPrompt
  env["bos_token"] = %tok.bosToken
  env["eos_token"] = %tok.eosToken
  env["pad_token"] = %tok.padToken
  renderChatTemplate(tok.templateProgram, env)

proc applyChatTemplate*(tok: HFTokenizer, userText: string, systemText = ""): seq[int] =
  var messages = newJArray()
  if systemText.len > 0: messages.add(%*{"role": "system", "content": systemText})
  messages.add(%*{"role": "user", "content": userText})
  # The template supplies role markers/BOS itself; HF also encodes with
  # add_special_tokens=false here to avoid duplicating the template's tokens.
  tok.encode(tok.formatChatTemplate(messages), addSpecialTokens = false)
