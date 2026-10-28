## Codestral tokenizer adapter. ModelProto loading, normalization and tokenization
## are implemented in sentencepiece_native.nim with Nim's standard library only.
import std/[os, strformat]
import sentencepiece_native

type HfTokenizer* = object
  backend*: string
  model: SentencePieceModel
  vocab*, bosId*, eosId*, unkId*: int
  special*, banned*, garbageSpace*: seq[bool]
  stopIds*: seq[int]

proc sentencePieceFile(hfDir: string): string =
  for name in ["tokenizer.model.v3", "tokenizer.model"]:
    let path = hfDir / name
    if fileExists(path): return path

proc buildSpecialTables(tok: var HfTokenizer) =
  tok.special = newSeq[bool](tok.vocab)
  tok.banned = newSeq[bool](tok.vocab)
  tok.garbageSpace = newSeq[bool](tok.vocab)
  for piece in ["▁", "Ġ", " "]:
    let id = tok.model.pieceId(piece)
    if id >= 0: tok.garbageSpace[id] = true
  for id, piece in tok.model.pieces:
    tok.special[id] = tok.model.kinds[id] in {spUnknown, spControl, spUnused}
    if piece in ["</s>", "[INST]", "[/INST]", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
      tok.special[id] = true
      tok.stopIds.add(id)
  if tok.eosId >= 0:
    tok.special[tok.eosId] = true
    if tok.eosId notin tok.stopIds: tok.stopIds.add(tok.eosId)
  if tok.bosId >= 0: tok.special[tok.bosId] = true
  for id in 0 ..< tok.vocab: tok.banned[id] = tok.special[id] or tok.garbageSpace[id]
  for id in tok.stopIds: tok.banned[id] = false
  if getEnv("BYBY_ALLOW_SPECIAL", "0") == "1":
    for id in 0 ..< tok.vocab: tok.banned[id] = false

proc newHfTokenizer*(hfDir: string): HfTokenizer =
  let path = sentencePieceFile(hfDir)
  if path.len == 0:
    raise newException(IOError, "[hf_tokenizer] tokenizer.model.v3/tokenizer.model not found in " & hfDir)
  result.model = loadSentencePieceModel(path)
  result.backend = "sentencepiece-nim"
  result.vocab = result.model.pieces.len
  result.bosId = result.model.bosId
  result.eosId = result.model.eosId
  result.unkId = result.model.unkId
  result.buildSpecialTables()
  let algorithm = if result.model.modelType == 2: "BPE" else: "Unigram"
  echo &"[hf_tokenizer] pure Nim {algorithm}, vocab={result.vocab}, " &
       &"normalizer={result.model.normalizerName()}, bos={result.bosId} " &
       &"eos={result.eosId} unk={result.unkId} stopIds={result.stopIds}"

proc encodeHf*(tok: HfTokenizer, text: string,
               addEos: bool = false, addBos: bool = true): seq[int] =
  result = encodeSentencePiece(tok.model, text)
  if addBos:
    if tok.bosId < 0: raise newException(ValueError, "[hf_tokenizer] tokenizer has no BOS id")
    result.insert(tok.bosId, 0)
  if addEos:
    if tok.eosId < 0: raise newException(ValueError, "[hf_tokenizer] tokenizer has no EOS id")
    result.add(tok.eosId)

proc decodeHf*(tok: HfTokenizer, ids: seq[int], skipSpecial: bool = true): string =
  var clean = newSeqOfCap[int](ids.len)
  for id in ids:
    if id < 0 or id >= tok.vocab: continue
    if skipSpecial and tok.special[id]: continue
    clean.add(id)
  decodeSentencePiece(tok.model, clean)

proc vocabSizeHf*(tok: HfTokenizer): int = tok.vocab
proc isStopIdHf*(tok: HfTokenizer, id: int): bool = id in tok.stopIds
proc pieceOfHf*(tok: HfTokenizer, id: int): string =
  if id >= 0 and id < tok.vocab: result = tok.model.pieces[id]
