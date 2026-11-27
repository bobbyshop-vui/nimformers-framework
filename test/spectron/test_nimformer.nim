## test_nimformer.nim
## Full version: LinearPlusAttention (Linear++) + Mamba-2 SSD + the Spectron block
## (2-branch gated fusion) + NimformerModelSpectron (embed -> N blocks -> lm
## head) with FULL forward/backward for every component, plus a
## train() loop (Adam) and generate() (greedy/temperature) to run real training and
## inference, nothing left "for later".
##
## Shared conventions (like nimformer.nim): activations are always [B,T,C] or
## treated as [B*T, C] through Linear/LayerNorm. Every heavy op (matmul, add,
## sigmoid...) goes through backend.nim (beMatmul/beAdd/beSigmoid...) so it runs
## on the chosen backend (cpu/metal/cuda/opencl/vulkan), NEVER silently falling back
## to plain Nim loops for large ops.
##
## GPU-ONLY: when ctx is a GPU (isGpuCtx: Metal/Vulkan) EVERY computation of
## Linear++, Mamba2SSD, the Spectron block, loss and Adam runs on the GPU; a GPU error
## RAISES (gpuFail), it is NEVER silently recomputed on the CPU. The cpu* functions
## below are only reference versions for the CPU backend and
## for comparison tests. What is left on the host is data arrangement (splitting/joining
## per-head slices, since Tensor.data lives on the host), numerical checks,
## gradient clipping and APF quantization (customfloat has no GPU kernel yet).

import std/[math, random, strformat, os, strutils, json, tables]
import customfloat
import backend
import quant
import nimformer
import mamba2_ssm_gpu   # kernel GPU Linear++ train (forward+backward), runMicroVm, CE/Adam GPU
from fp4_gemv_vm import buildLinearPlusStepProgram   # Linear++ 1-token GPU kernel (same kernel as infer)
import fp4_gemv_vm as lpvm                            # lpvm.lpLayout(C, H): layout [proj|y|z|S]
import gpu_layers  # encodeEluPlus1 / encodeEluPlus1Backward for GPU-only Linear++  # Tensor, Linear, LayerNorm, FeedForward, Embedding, addT,
                   # transpose, flatten2D, newTensor, randnTensor,
                   # ApfAdamState, apfAdamStep, newLinear, newLayerNorm,
                   # newFeedForward, newEmbedding

# ═══════════════════════════════════════════════════════════════
# CONFIG - EDIT HERE, NO NEED TO DIG INTO main BELOW
# ═══════════════════════════════════════════════════════════════
const
  CFG_EMBED_DIM* = 64
  CFG_N_HEADS* = 4
  CFG_N_LAYERS* = 2
  CFG_FF_MULT* = 4
  CFG_SEQ_LEN* = 128
  CFG_BATCH_SIZE* = 32
  CFG_NUM_MERGES* = 500        # final vocab = 256 + merges (stops early when no pairs are left)
  CFG_N_STEPS* = 6_000        # demo training steps
  CFG_LR* = 3e-4'f32           # peak learning rate (reached at the end of warmup)
  CFG_MIN_LR* = 3e-5'f32       # floor of the cosine decay
  CFG_WARMUP_STEPS* = 300      # linear warmup; also replayed after every resume (Adam restarts)
  CFG_LR_DECAY_STEPS* = 30_000 # global step where the cosine decay reaches CFG_MIN_LR
  CFG_RESUME_LR* = 1e-4'f32    # peak of the cosine restarted for a run resumed past CFG_LR_DECAY_STEPS
  CFG_RESUME_MIN_LR* = 1e-5'f32 # floor of that restarted cosine, reached at the end of the run
  CFG_REQUANTIZE_EVERY* = 50   # see the explanation in trainStep below
  CFG_LOG_EVERY* = 100         # log the loss every N steps
  CFG_BPE_TRAIN_CHARS* = 300_000  # number of leading corpus chars used to train BPE
  CFG_CORPUS_PATH* = "test/spectron/databricks-dolly-15k.jsonl"
  CFG_CODE_CORPUS_PATHS* = [
    "test/spectron/nim_python_code_dataset_mega.jsonl",
    "test/spectron/python_code_dataset_1.jsonl",
    "test/spectron/python_code_dataset_2.jsonl",
    "test/spectron/python_code_dataset_3.jsonl",
    "test/spectron/python_code_dataset_4.jsonl",
    "test/spectron/python_code_dataset_5.jsonl",
  ]
  CFG_CODE_EXAMPLES_PER_FILE* = 4_000 # reservoir sample across each full JSONL
  CFG_GEN_MAX_NEW_TOKENS* = 40
  CFG_GEN_TEMPERATURE* = 0'f32
  CFG_GEN_PROMPT* = "When did Virgin Australia start operating?"
  CFG_CHECKPOINT_PATH* = "test/spectron/models/finetune.nimq"
  CFG_CHECKPOINT_EVERY* = 5    # save a checkpoint every N training steps

# ═══════════════════════════════════════════════════════════════
# GPU-ONLY: a GPU error -> raise, NOT recomputed on the CPU (this used to be
# warnFallbackOnce then cpuMamba2Forward/cpuLinearPlusForward).
# ═══════════════════════════════════════════════════════════════
proc gpuFail(what: string; backend: GpuBackend) {.noreturn.} =
  let detail = sessionGetLastError(backend)
  raise newException(CatchableError,
    "[test_nimformer] GPU " & what & " failed on " & $backend &
    (if detail.len > 0: ": " & detail else: "") & " (GPU-only: no CPU fallback)")

# ═══════════════════════════════════════════════════════════════
# SPECTRON'S OWN BPE TOKENIZER (byte-level BPE, trained on this
# project's own corpus - does NOT reuse the 32256-vocab model's tokenizer.json
# in nim_inference.nim, those are 2 entirely different models).
#
# Standard GPT-2 style byte-level BPE algorithm:
#   - 256 base tokens = the 256 byte values (0..255), which can always represent ANY
#     byte string (there is never an "unknown token").
#   - Repeat numMerges times: count the frequency of every adjacent token pair in the whole
#     corpus, pick the most frequent pair, merge it into 1 new token
#     id = 256 + step, record the rule (a,b)->newId in EXACT creation order.
#   - Encoding new text: start from raw bytes and apply the merge rules
#     in the trained order (this is the key to BPE decoding being
#     deterministic).
#   - Decode: unfold newId -> (a,b) -> ... -> raw bytes.
# ═══════════════════════════════════════════════════════════════
type
  BPETokenizer* = object
    ## A REAL GPT-2/Llama-style subword tokenizer (byte-level BPE + pre-tokenizer
    ## by words/numbers/punctuation), no longer "1 byte = 1 token":
    ##   1) byte-to-unicode: maps the 256 byte values to 256 ALWAYS printable
    ##      Unicode characters (the original GPT-2 bytes_to_unicode() algorithm - see
    ##      newGpt2ByteEncoder below) -> vocab/merges are readable STRINGS,
    ##      exported to tokenizer.json in the same format as real HF/tokenizers.
    ##   2) pre-tokenize: split text into pieces (words with a leading space,
    ##      numbers, punctuation, whitespace) BEFORE BPE - like the GPT-2 regex
    ##      (\p{L}+, \p{N}+, punctuation, ...) - so tokens never "jump"
    ##      across word/number/punctuation boundaries like raw whole-file BPE.
    ##   3) Real BPE merges only INSIDE each pre-token piece.
    vocab*: Table[string, int]        # subword string -> id
    id2tok*: seq[string]              # id -> subword string
    merges*: seq[(string, string)]    # in exact training order (rank = index)
    mergeRank*: Table[(string, string), int]
    byteEncoder*: array[256, string]  # raw byte -> unicode char (bytes_to_unicode)
    byteDecoder*: Table[string, int]  # unicode char -> raw byte (reverse direction)

  RawTokenRecord* = object
    ## One recoverable byte-level BPE token from the original corpus stream.
    id*: int
    corpusIndex*: int
    text*: string

  RawTokenWindow* = object
    ## A sampled contiguous input window. Its tokens are the exact inputs sent
    ## through both Linear++ attention and the Mamba2 parallel scan.
    step*: int
    batchRow*: int
    corpusStart*: int
    tokenCount*: int

  RawTokenBatch* = object
    ## Training tensors plus the raw-corpus spans that produced each row.
    ids*: seq[seq[int]]
    targets*: seq[seq[int]]
    windows*: seq[RawTokenWindow]

  RawTokenMemory* = object
    ## The complete encoded corpus and a per-run ledger of windows sampled from
    ## it. Raw token text is reconstructed losslessly from each BPE id on demand.
    tokenIds: seq[int]
    tokenizer: BPETokenizer
    windows*: seq[RawTokenWindow]

proc newGpt2ByteEncoder(): array[256, string] =
  ## Exactly bytes_to_unicode() in OpenAI GPT-2's original encoder.py: every
  ## "printable" byte (33..126, 161..172, 174..255) maps to itself, the
  ## remaining bytes (control chars, whitespace, ...) are moved to the unicode range
  ## 256.. so the vocab NEVER contains invisible/ambiguous characters.
  var bs: seq[int] = @[]
  for b in ord('!') .. ord('~'): bs.add(b)
  for b in 0xA1 .. 0xAC: bs.add(b)
  for b in 0xAE .. 0xFF: bs.add(b)
  var cs = bs
  var n = 0
  for b in 0 .. 255:
    if b notin bs:
      bs.add(b)
      cs.add(256 + n)
      inc n
  proc encodeCodepoint(cp: int): string =
    if cp < 0x80:
      result = $char(cp)
    else:
      # cp is always < 0x800 (n is at most ~68) so 2-byte UTF-8 is enough.
      result = newString(2)
      result[0] = char(0xC0 or (cp shr 6))
      result[1] = char(0x80 or (cp and 0x3F))
  for i in 0 ..< bs.len:
    result[bs[i]] = encodeCodepoint(cs[i])

proc isAsciiLetter(c: char): bool = c in {'a'..'z', 'A'..'Z'}
proc isAsciiDigit(c: char): bool = c in {'0'..'9'}
proc isAsciiSpace(c: char): bool = c in {' ', '\t', '\n', '\r'}

iterator bpePieces*(s: string): tuple[start, stop: int] =
  ## Rough split in the spirit of the GPT-2 regex
  ## ( 's| 't| 're| 've| 'm| 'll| 'd| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+ )
  ## hand-written as an ASCII scanner (enough for the demo, no Unicode regex
  ## library needed - bytes >=128 are treated as a single "letter"/punctuation depending
  ## on context, still encoded/decoded CORRECTLY thanks to the byte-level fallback later).
  var i = 0
  let n = s.len
  const contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
  while i < n:
    var matched = false
    if s[i] == '\'':
      for c in contractions:
        if s.continuesWith(c, i):
          yield (i, i + c.len)
          i += c.len
          matched = true
          break
    if matched: continue

    var start = i
    var hasLeadingSpace = false
    if s[i] == ' ' and i + 1 < n and not isAsciiSpace(s[i + 1]):
      hasLeadingSpace = true
      inc i

    if i < n and isAsciiLetter(s[i]):
      if not hasLeadingSpace: start = i
      while i < n and isAsciiLetter(s[i]): inc i
      yield (start, i)
    elif i < n and isAsciiDigit(s[i]):
      if not hasLeadingSpace: start = i
      while i < n and isAsciiDigit(s[i]): inc i
      yield (start, i)
    elif i < n and isAsciiSpace(s[i]):
      start = i
      while i < n and isAsciiSpace(s[i]):
        if i + 1 < n and s[i] == ' ' and not isAsciiSpace(s[i + 1]) and i > start:
          break   # leave 1 space as the "leading space" of the next piece
        inc i
      yield (start, i)
    elif i < n:
      if not hasLeadingSpace: start = i
      # 1 non-ascii/punctuation byte = 1 piece (safe for multi-byte UTF-8 and punctuation)
      inc i
      yield (start, i)

proc gpt2PreTokenize(s: string): seq[string] =
  for (start, stop) in bpePieces(s):
    result.add(s[start ..< stop])

proc toByteEncodedChars(enc: array[256, string], s: string): seq[string] =
  ## Each raw BYTE of the string (not each Unicode character) -> 1 "base
  ## character" in the byte-encoder table - this is the "byte-level" of GPT-2/Llama:
  ## it can always represent ANY byte string, no unknown token.
  result = newSeq[string](s.len)
  for i in 0 ..< s.len:
    result[i] = enc[uint8(s[i]).int]

proc bpeMergeWord(tokens: seq[string], mergeRank: Table[(string, string), int]): seq[string] =
  ## Standard HF-style BPE: repeatedly find the adjacent pair with the LOWEST RANK (created
  ## earliest in training) inside 1 pre-token and merge it, until no pair
  ## is in the merge table.
  var word = tokens
  while word.len > 1:
    var bestRank = int.high
    var bestIdx = -1
    for i in 0 ..< word.len - 1:
      let pair = (word[i], word[i + 1])
      if pair in mergeRank:
        let r = mergeRank[pair]
        if r < bestRank:
          bestRank = r
          bestIdx = i
    if bestIdx < 0: break
    var newWord = newSeqOfCap[string](word.len - 1)
    var i = 0
    while i < word.len:
      if i == bestIdx:
        newWord.add(word[i] & word[i + 1])
        i += 2
      else:
        newWord.add(word[i])
        inc i
    word = newWord
  result = word

proc trainBPE*(corpusText: string, numMerges: int): BPETokenizer =
  result.byteEncoder = newGpt2ByteEncoder()
  result.byteDecoder = initTable[string, int]()
  for b in 0 .. 255: result.byteDecoder[result.byteEncoder[b]] = b

  # pre-tokenize once, keeping the seq[seq[string]] form (each pre-token is
  # a seq of byte-encoded chars) so adjacent pairs are counted ONLY within each
  # pre-token (never merging across word/number/punctuation boundaries).
  let pretoks = gpt2PreTokenize(corpusText)
  var words = newSeq[seq[string]](pretoks.len)
  for i, pt in pretoks:
    words[i] = toByteEncodedChars(result.byteEncoder, pt)

  var merges: seq[(string, string)] = @[]
  var mergeRank = initTable[(string, string), int]()

  for step in 0 ..< numMerges:
    var counts = initTable[(string, string), int]()
    for w in words:
      for i in 0 ..< w.len - 1:
        let p = (w[i], w[i + 1])
        counts[p] = counts.getOrDefault(p, 0) + 1
    if counts.len == 0: break
    var bestPair: (string, string) = ("", "")
    var bestCount = 0
    for p, c in counts:
      if c > bestCount:
        bestCount = c
        bestPair = p
    if bestCount < 2: break

    merges.add(bestPair)
    mergeRank[bestPair] = merges.len - 1
    let merged = bestPair[0] & bestPair[1]
    for wi in 0 ..< words.len:
      var w = words[wi]
      if w.len < 2: continue
      var newW = newSeqOfCap[string](w.len)
      var i = 0
      while i < w.len:
        if i < w.len - 1 and w[i] == bestPair[0] and w[i + 1] == bestPair[1]:
          newW.add(merged)
          i += 2
        else:
          newW.add(w[i])
          inc i
      words[wi] = newW

  result.merges = merges
  result.mergeRank = mergeRank

  # Build the vocab: the 256 base bytes first (in byte order 0..255), then
  # each merge (256+step) - exactly the GPT-2/Llama convention.
  result.id2tok = newSeq[string](256 + merges.len)
  result.vocab = initTable[string, int]()
  for b in 0 .. 255:
    result.id2tok[b] = result.byteEncoder[b]
    result.vocab[result.byteEncoder[b]] = b
  for step, (a, b) in merges:
    let tokStr = a & b
    let id = 256 + step
    result.id2tok[id] = tokStr
    result.vocab[tokStr] = id

proc vocabSize*(tok: BPETokenizer): int = tok.id2tok.len

proc encodeBpePiece*(tok: BPETokenizer, text: string): seq[int] =
  let baseChars = toByteEncodedChars(tok.byteEncoder, text)
  for piece in bpeMergeWord(baseChars, tok.mergeRank):
    if piece in tok.vocab:
      result.add(tok.vocab[piece])
    else:
      for base in baseChars:
        result.add(tok.vocab.getOrDefault(base, 0))

proc encodeBPE*(tok: BPETokenizer, s: string): seq[int] =
  for (start, stop) in bpePieces(s):
    result.add(tok.encodeBpePiece(s[start ..< stop]))

proc decodeBPEStr*(tok: BPETokenizer, ids: seq[int]): string =
  var unicodeStr = ""
  for id in ids:
    if id >= 0 and id < tok.id2tok.len:
      unicodeStr.add(tok.id2tok[id])
  # unicodeStr is now a string of byte-encoded chars (each char = 1-2 UTF-8 bytes
  # representing 1 original byte) - decode back to the original bytes and join into a string.
  var outBytes: seq[uint8] = @[]
  var i = 0
  while i < unicodeStr.len:
    let c0 = uint8(unicodeStr[i])
    var glyph: string
    if c0 < 0x80:
      glyph = $unicodeStr[i]
      inc i
    else:
      glyph = unicodeStr[i .. i + 1]
      i += 2
    if glyph in tok.byteDecoder:
      outBytes.add(uint8(tok.byteDecoder[glyph]))
  result = newString(outBytes.len)
  for k in 0 ..< outBytes.len: result[k] = char(outBytes[k])

proc decodeBPE*(tok: BPETokenizer, ids: seq[int]): seq[uint8] =
  let s = decodeBPEStr(tok, ids)
  result = newSeq[uint8](s.len)
  for i in 0 ..< s.len: result[i] = uint8(s[i])

proc newRawTokenMemory*(tok: BPETokenizer, tokenIds: sink seq[int]): RawTokenMemory =
  ## Keep the full tokenized corpus as the source for training windows. The
  ## tokenizer is retained so a stored id can be inspected as its raw byte
  ## piece without duplicating decoded strings for every corpus occurrence.
  result.tokenIds = tokenIds
  result.tokenizer = tok
  result.windows = @[]

proc tokenCount*(memory: RawTokenMemory): int = memory.tokenIds.len

proc rawTokenAt*(memory: RawTokenMemory, corpusIndex: int): RawTokenRecord =
  ## Recover one original BPE token and its corpus position. `text` contains
  ## the token's exact decoded bytes; an individual BPE piece can end midway
  ## through a multi-byte UTF-8 character, just as in the source stream.
  doAssert corpusIndex >= 0 and corpusIndex < memory.tokenIds.len,
    "raw token memory corpus index is out of range"
  let id = memory.tokenIds[corpusIndex]
  result = RawTokenRecord(id: id, corpusIndex: corpusIndex,
    text: decodeBPEStr(memory.tokenizer, @[id]))

proc rawTokensForWindow*(memory: RawTokenMemory,
                         window: RawTokenWindow): seq[RawTokenRecord] =
  ## Materialize the raw token records for a sampled attention/scan window.
  doAssert window.corpusStart >= 0 and window.tokenCount >= 0 and
    window.corpusStart + window.tokenCount <= memory.tokenIds.len,
    "raw token window is outside the corpus memory"
  result = newSeqOfCap[RawTokenRecord](window.tokenCount)
  for i in 0 ..< window.tokenCount:
    result.add(memory.rawTokenAt(window.corpusStart + i))

proc rawTextForWindow*(memory: RawTokenMemory,
                       window: RawTokenWindow): string =
  ## Reconstruct the complete UTF-8 byte sequence for one sampled window.
  doAssert window.corpusStart >= 0 and window.tokenCount >= 0 and
    window.corpusStart + window.tokenCount <= memory.tokenIds.len,
    "raw token window is outside the corpus memory"
  var ids = newSeq[int](window.tokenCount)
  for i in 0 ..< window.tokenCount:
    ids[i] = memory.tokenIds[window.corpusStart + i]
  result = decodeBPEStr(memory.tokenizer, ids)

proc sampleRawTokenBatch*(memory: var RawTokenMemory, B, T, step: int,
                          rng: var Rand): RawTokenBatch =
  ## Sample next-token windows through RawTokenMemory and record every input
  ## span. The resulting `ids` are consumed unchanged by Spectron's Linear++
  ## attention and Mamba2 parallel-scan branches.
  doAssert B > 0 and T > 0, "raw token batch dimensions must be positive"
  doAssert memory.tokenIds.len > T + 1,
    "raw token corpus is too short for the requested sequence length"
  result.ids = newSeq[seq[int]](B)
  result.targets = newSeq[seq[int]](B)
  result.windows = newSeqOfCap[RawTokenWindow](B)
  for b in 0 ..< B:
    let start = rng.rand(memory.tokenIds.len - T - 1)
    let window = RawTokenWindow(step: step, batchRow: b,
      corpusStart: start, tokenCount: T)
    memory.windows.add(window)
    result.windows.add(window)
    result.ids[b] = newSeq[int](T)
    result.targets[b] = newSeq[int](T)
    for t in 0 ..< T:
      result.ids[b][t] = memory.tokenIds[start + t]
      result.targets[b][t] = memory.tokenIds[start + t + 1]

proc saveBPE*(tok: BPETokenizer, path: string) =
  ## Saves EXACTLY the real HF/tokenizers tokenizer.json schema (BPE model):
  ## {"vocab_size", "vocab": {token_string: id}, "merges": ["a b", ...]}.
  ## export_to_nimq.py also exports this exact schema (see the change there) so
  ## the 2 sides are compatible and can load each other's files.
  var vocabJson = newJObject()
  for k, v in tok.vocab: vocabJson[k] = %v
  var mergesJson = newJArray()
  for (a, b) in tok.merges: mergesJson.add(%(a & " " & b))
  let j = %*{"vocab_size": tok.vocabSize, "vocab": vocabJson, "merges": mergesJson}
  writeFile(path, pretty(j))

proc loadBPE*(path: string): BPETokenizer =
  result.byteEncoder = newGpt2ByteEncoder()
  result.byteDecoder = initTable[string, int]()
  for b in 0 .. 255: result.byteDecoder[result.byteEncoder[b]] = b

  let j = parseJson(readFile(path))
  result.vocab = initTable[string, int]()
  var maxId = -1
  for k, v in j["vocab"].pairs:
    let id = v.getInt()
    result.vocab[k] = id
    if id > maxId: maxId = id
  result.id2tok = newSeq[string](maxId + 1)
  for k, id in result.vocab: result.id2tok[id] = k

  result.merges = @[]
  result.mergeRank = initTable[(string, string), int]()
  for m in j["merges"]:
    let parts = m.getStr().split(' ', 1)
    let pair = (parts[0], parts[1])
    result.merges.add(pair)
    result.mergeRank[pair] = result.merges.len - 1

const BPE_TOKENIZER_PATH = "test/spectron/models/spectron_bpe_tokenizer.json"

proc getOrTrainBPE*(corpusText: string, numMerges: int): BPETokenizer =
  ## If already trained (the file exists), reload it for speed/consistency
  ## across runs; otherwise train a new one on `corpusText` and save it to
  ## `spectron_bpe_tokenizer.json` (Spectron's own file, a real
  ## GPT-2/Llama-style BPE schema, NOT the old fake byte-identity table,
  ## and it does NOT touch the tokenizer.json of other models in the repo).
  if fileExists(BPE_TOKENIZER_PATH):
    result = loadBPE(BPE_TOKENIZER_PATH)
  else:
    result = trainBPE(corpusText, numMerges)
    saveBPE(result, BPE_TOKENIZER_PATH)


# ═══════════════════════════════════════════════════════════════
# LINEAR++ ATTENTION (causal linear attention, feature map elu(x)+1)
# ═══════════════════════════════════════════════════════════════
## Idea (Katharopoulos et al., "Transformers are RNNs"): instead of
## softmax(QK^T)V, use the feature map phi(x) = elu(x)+1 (always >0) and then:
##   S[i,j]   = phi(Q)_i . phi(K)_j          (j<=i, causal)
##   Num[i]   = sum_{j<=i} S[i,j] * V[j]
##   Den[i]   = sum_{j<=i} S[i,j]
##   Out[i]   = Num[i] / (Den[i] + eps)
## Written as a [T,T] "quadratic form" exactly like Mamba2SSD below (causal
## mask a=1, NO decay) - shares a code path and is verifiable,
## no separate kernel/scan needed. Forward+backward are both closed-form, no
## hidden parameters (fixed feature map, nothing learned besides the Q/K/V/out projections).

const LP_EPS = 1e-6'f32

proc eluPlus1(x: float32): float32 =
  if x > 0'f32: x + 1'f32
  else: exp(x)

proc eluPlus1Grad(x: float32): float32 =
  ## d/dx [elu(x)+1]: 1 if x>0, exp(x) if x<=0 (matches the forward above).
  if x > 0'f32: 1'f32
  else: exp(x)

proc applyFeatureMap(x: seq[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len: result[i] = eluPlus1(x[i])

proc applyFeatureMapGpu(ctx: Backend; x: seq[float32]): seq[float32] =
  ## GPU JIT feature map for Linear++ training (no CPU loop).
  let n = x.len
  var hIn = sessionUpload(ctx.kind.toByby(), x)
  var hOut = sessionAllocScratch(ctx.kind.toByby(), n)
  encodeEluPlus1(ctx, hIn, hOut, n)
  result = sessionRead(ctx.kind.toByby(), hOut, n)
  sessionFree(ctx.kind.toByby(), hIn)
  sessionFree(ctx.kind.toByby(), hOut)

proc cpuLinearPlusForward*(qData, kData, vData: seq[float32], B, H, T, D: int):
    tuple[outData, qf, kf, den: seq[float32]] =
  ## CPU reference (used by existing callers without ctx).
  result.qf = applyFeatureMap(qData)
  result.kf = applyFeatureMap(kData)
  result.den = newSeq[float32](B * T * H)
  let C = H * D
  result.outData = newSeq[float32](B * T * C)
  template idx(b, t, h, d: int): int = b * T * C + t * C + h * D + d
  template dIdx(b, t, h: int): int = b * T * H + t * H + h
  for h in 0 ..< H:
    for b in 0 ..< B:
      for i in 0 ..< T:
        var den = 0'f32
        var num = newSeq[float32](D)
        for j in 0 .. i:
          var score = 0'f32
          for d in 0 ..< D:
            score += result.qf[idx(b, i, h, d)] * result.kf[idx(b, j, h, d)]
          den += score
          for d in 0 ..< D:
            num[d] += score * vData[idx(b, j, h, d)]
        result.den[dIdx(b, i, h)] = den
        let invDen = 1'f32 / (den + LP_EPS)
        for d in 0 ..< D:
          result.outData[idx(b, i, h, d)] = num[d] * invDen

proc cpuLinearPlusBackward*(qData, kData, vData, qf, kf, den, dOutData: seq[float32],
                            B, H, T, D: int): tuple[dQ, dK, dV: seq[float32]] =
  let C = H * D
  result.dQ = newSeq[float32](B * T * C)
  result.dK = newSeq[float32](B * T * C)
  result.dV = newSeq[float32](B * T * C)
  var dQf = newSeq[float32](B * T * C)
  var dKf = newSeq[float32](B * T * C)
  template idx(b, t, h, d: int): int = b * T * C + t * C + h * D + d
  template dIdx(b, t, h: int): int = b * T * H + t * H + h
  for h in 0 ..< H:
    for b in 0 ..< B:
      # Step 1: rebuild Num[i] (needed for dDen), then propagate the gradient over i,j.
      for i in 0 ..< T:
        let denI = den[dIdx(b, i, h)]
        let invDen = 1'f32 / (denI + LP_EPS)
        var numI = newSeq[float32](D)
        for j in 0 .. i:
          var score = 0'f32
          for d in 0 ..< D:
            score += qf[idx(b, i, h, d)] * kf[idx(b, j, h, d)]
          for d in 0 ..< D:
            numI[d] += score * vData[idx(b, j, h, d)]

        # dOut_i -> dNum_i, dDen_i
        var dDotNumDOut = 0'f32   # dOut_i . Num_i (used for dDen)
        for d in 0 ..< D:
          dDotNumDOut += dOutData[idx(b, i, h, d)] * numI[d]
        let dDen = -dDotNumDOut * invDen * invDen

        for j in 0 .. i:
          var score = 0'f32
          for d in 0 ..< D:
            score += qf[idx(b, i, h, d)] * kf[idx(b, j, h, d)]
          # dScore from the Num branch: dNum_i . V_j ; from the Den branch: dDen (same for every j<=i)
          var dScoreNum = 0'f32
          for d in 0 ..< D:
            dScoreNum += (dOutData[idx(b, i, h, d)] * invDen) * vData[idx(b, j, h, d)]
          let dScore = dScoreNum + dDen
          # dV_j += score * dNum_i
          for d in 0 ..< D:
            result.dV[idx(b, j, h, d)] += score * (dOutData[idx(b, i, h, d)] * invDen)
          # S[i,j] = qf_i . kf_j
          for d in 0 ..< D:
            dQf[idx(b, i, h, d)] += dScore * kf[idx(b, j, h, d)]
            dKf[idx(b, j, h, d)] += dScore * qf[idx(b, i, h, d)]

  # dQf/dKf -> dQ/dK through the derivative of the elu(x)+1 feature map
  for i in 0 ..< dQf.len:
    result.dQ[i] = dQf[i] * eluPlus1Grad(qData[i])
    result.dK[i] = dKf[i] * eluPlus1Grad(kData[i])

type
  LinearPlusAttention* = object
    nHeads*, headDim*, embedDim*: int
    qkvProj*: Linear   # x -> [Q|K|V], each part embedDim
    outProj*: Linear
    useSession*: bool  # kept for API compatibility; a GPU ctx ALWAYS runs on the GPU
                        # (no CPU path any more when useSession=false).

proc newLinearPlusAttention*(embedDim, nHeads: int, useSession: bool = true;
                            initializeWeights = true): LinearPlusAttention =
  assert embedDim mod nHeads == 0
  result.embedDim = embedDim
  result.nHeads = nHeads
  result.headDim = embedDim div nHeads
  result.qkvProj = newLinear(embedDim, 3 * embedDim, initializeWeights)
  result.outProj = newLinear(embedDim, embedDim, initializeWeights)
  result.useSession = useSession

# Incremental state for causal Linear++ attention.
# This is the exact associative form of:
#   S_t = sum_{j<=t} phi(K_j) outer V_j
#   z_t = sum_{j<=t} phi(K_j)
#   y_t = phi(Q_t) S_t / (phi(Q_t) dot z_t + eps)
# It is mathematically equivalent to the causal quadratic reference above,
# but O(H*D*D) work and O(H*D*D + H*D) state per token instead of O(T^2).
type
  LinearPlusAttentionState* = object
    ## S[h,dq,dv], row-major, then z[h,dq].
    kv*: seq[float32]
    z*: seq[float32]
    nHeads*: int
    headDim*: int

proc newLinearPlusAttentionState*(m: LinearPlusAttention): LinearPlusAttentionState =
  result.nHeads = m.nHeads
  result.headDim = m.headDim
  result.kv = newSeq[float32](m.nHeads * m.headDim * m.headDim)
  result.z = newSeq[float32](m.nHeads * m.headDim)

proc reset*(s: var LinearPlusAttentionState) =
  for i in 0 ..< s.kv.len: s.kv[i] = 0'f32
  for i in 0 ..< s.z.len: s.z[i] = 0'f32

proc forwardLinearPlusStep*(m: LinearPlusAttention,
                            x: Tensor,
                            state: var LinearPlusAttentionState,
                            ctx: Backend): Tensor =
  ## One-token causal Linear++ step. No T*T score/mask is constructed.
  ## x must be [1,1,C]. Q/K/V projection still uses the normal GPU-resident
  ## Linear path; only the small recurrent attention state is updated here.
  doAssert x.data.len == m.embedDim, "Linear++ step requires exactly one token"
  doAssert state.nHeads == m.nHeads and state.headDim == m.headDim,
    "Linear++ attention state does not match model"

  let proj = m.qkvProj.forward(x, ctx)
  let C = m.embedDim
  let D = m.headDim
  let H = m.nHeads

  var y = newTensor(@[1, 1, C])
  if isGpuCtx(ctx):
    # GPU: uses the infer Linear++ step kernel itself (buildLinearPlusStepProgram, 1 TG/head,
    # D lanes) on the buffer [proj | y | z | S]; state host <-> GPU each step.
    let lay = lpvm.lpLayout(C, H)
    var buf = newSeq[float32](lay.total)
    buf.blit(lay.offProj, proj.data, 0, 3 * C)
    buf.blit(lay.offZ, state.z, 0, H * D)
    buf.blit(lay.offS, state.kv, 0, H * D * D)
    if not runMicroVm(ctx, buildLinearPlusStepProgram(lay, LP_EPS), buf, H * D, 16,
                      "lp_step_host_" & $H & "_" & $C, tgSize = D):
      gpuFail("Linear++ step", ctx.kind.toByby())
    y.data = buf[lay.offY ..< lay.offY + C]
    state.z = buf[lay.offZ ..< lay.offZ + H * D]
    state.kv = buf[lay.offS ..< lay.offS + H * D * D]
    return m.outProj.forward(y, ctx)
  let qBase = 0
  let kBase = C
  let vBase = 2 * C
  # Reuse these buffers across heads: no per-head heap allocation in decode.
  var qf = newSeq[float32](C)
  var kf = newSeq[float32](C)

  for h in 0 ..< H:
    let qOff = h * D
    let kvBase = h * D * D
    let zBase = h * D

    # phi(Q), phi(K) = ELU(x)+1. Update the causal sufficient statistics
    # before reading them, so the current token can attend to itself.
    for d in 0 ..< D:
      qf[qOff + d] = eluPlus1(proj.data[qBase + qOff + d])
      kf[qOff + d] = eluPlus1(proj.data[kBase + qOff + d])
      state.z[zBase + d] += kf[qOff + d]

    # S += phi(K)^T outer V.
    for dq in 0 ..< D:
      let kval = kf[qOff + dq]
      let row = kvBase + dq * D
      for dv in 0 ..< D:
        state.kv[row + dv] += kval * proj.data[vBase + qOff + dv]

    # O = Q S / (Q z + eps).
    var den = 0'f32
    for dq in 0 ..< D:
      den += qf[qOff + dq] * state.z[zBase + dq]
    let invDen = 1'f32 / (den + LP_EPS)

    for dv in 0 ..< D:
      var num = 0'f32
      for dq in 0 ..< D:
        num += qf[qOff + dq] * state.kv[kvBase + dq * D + dv]
      y.data[qOff + dv] = num * invDen

  result = m.outProj.forward(y, ctx)

proc splitQKV*(proj: Tensor, B, T, C: int): tuple[q, k, v: seq[float32]] =
  result.q = newSeq[float32](B * T * C)
  result.k = newSeq[float32](B * T * C)
  result.v = newSeq[float32](B * T * C)
  for b in 0 ..< B:
    let baseIn = b * T * 3 * C
    let baseOut = b * T * C
    for t in 0 ..< T:
      for c in 0 ..< C:
        result.q[baseOut + t * C + c] = proj.data[baseIn + t * 3 * C + c]
        result.k[baseOut + t * C + c] = proj.data[baseIn + t * 3 * C + C + c]
        result.v[baseOut + t * C + c] = proj.data[baseIn + t * 3 * C + 2 * C + c]

proc forwardLinearPlus*(m: LinearPlusAttention, x: Tensor, ctx: Backend): Tensor =
  let B = x.shape[0]; let T = x.shape[1]; let C = x.shape[2]
  assert C == m.embedDim
  let proj = m.qkvProj.forward(x, ctx)
  var y = newTensor(@[B, T, C])
  if isGpuCtx(ctx):
    # GPU micro-kernel (train_gpu_vm): phi + causal Linear++ in 1 session.
    y.data = linearPlusForwardGpu(ctx, proj.data, B, T, m.nHeads, m.headDim)
  else:
    let (qData, kData, vData) = splitQKV(proj, B, T, C)
    let (outData, _, _, _) = cpuLinearPlusForward(qData, kData, vData, B, m.nHeads, T, m.headDim)
    y.data = outData
  result = m.outProj.forward(y, ctx)

proc forwardLinearPlusSession*(m: LinearPlusAttention, x: Tensor, ctx: Backend): Tensor =
  ## Linear++ forward for INFERENCE (jamba_hybrid/lora). GPU: the SAME micro-kernel
  ## as forwardLinearPlus (linearPlusForwardGpu: phi(x)=elu+1, causal scan,
  ## normalising division - all in 1 GPU session). This used to compute phi and
  ## the division on the CPU, split per-head slices, and fall back to cpuLinearPlusForward
  ## when the session failed; now a GPU error -> raise. Non-GPU backends: the reference version.
  let B = x.shape[0]; let T = x.shape[1]; let C = x.shape[2]
  assert C == m.embedDim
  let proj = m.qkvProj.forward(x, ctx)
  var y = newTensor(@[B, T, C])
  if isGpuCtx(ctx):
    y.data = linearPlusForwardGpu(ctx, proj.data, B, T, m.nHeads, m.headDim)
  else:
    let (qData, kData, vData) = splitQKV(proj, B, T, C)
    y.data = cpuLinearPlusForward(qData, kData, vData, B, m.nHeads, T, m.headDim).outData
  result = m.outProj.forward(y, ctx)

proc backwardLinearPlus*(m: LinearPlusAttention, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dQkvW, dQkvB, dOutW, dOutB: Tensor] =
  let B = x.shape[0]; let T = x.shape[1]; let C = x.shape[2]
  let proj = m.qkvProj.forward(x, ctx)
  var y = newTensor(@[B, T, C])
  var dProj = newTensor(proj.shape)
  var dOutP: tuple[dX, dW, dB: Tensor]
  if isGpuCtx(ctx):
    # GPU: rebuilt forward + 2-pass backward in 1 session. outProj.backward needs y
    # so the forward runs first (1 session), then the backward (1 session, rebuilding phi/y itself).
    y.data = linearPlusForwardGpu(ctx, proj.data, B, T, m.nHeads, m.headDim)
    dOutP = m.outProj.backward(y, dOut, ctx)
    dProj.data = linearPlusBackwardGpu(ctx, proj.data, dOutP.dX.data, B, T, m.nHeads, m.headDim)
  else:
    let (qData, kData, vData) = splitQKV(proj, B, T, C)
    let (outData, qf, kf, den) = cpuLinearPlusForward(qData, kData, vData, B, m.nHeads, T, m.headDim)
    y.data = outData
    dOutP = m.outProj.backward(y, dOut, ctx)
    let dY = dOutP.dX.data
    let (dQ, dK, dV) = cpuLinearPlusBackward(qData, kData, vData, qf, kf, den, dY,
                                              B, m.nHeads, T, m.headDim)
    for b in 0 ..< B:
      let baseIn = b * T * 3 * C
      let baseOut = b * T * C
      for t in 0 ..< T:
        for c in 0 ..< C:
          dProj.data[baseIn + t * 3 * C + c]         = dQ[baseOut + t * C + c]
          dProj.data[baseIn + t * 3 * C + C + c]     = dK[baseOut + t * C + c]
          dProj.data[baseIn + t * 3 * C + 2 * C + c] = dV[baseOut + t * C + c]

  let dQkvP = m.qkvProj.backward(x, dProj, ctx)
  result = (dQkvP.dX, dQkvP.dW, dQkvP.dB, dOutP.dW, dOutP.dB)

# ═══════════════════════════════════════════════════════════════
# LINEAR++ TRANSFORMER BLOCK (Pre-LN, same shape as TransformerBlock)
# ═══════════════════════════════════════════════════════════════
type
  TransformerBlockLinearPP* = object
    attn*: LinearPlusAttention
    ff*: FeedForward
    ln1*, ln2*: LayerNorm

proc newTransformerBlockLinearPP*(embedDim, nHeads, ffMult: int, useSession: bool = true): TransformerBlockLinearPP =
  result.attn = newLinearPlusAttention(embedDim, nHeads, useSession)
  result.ff = newFeedForward(embedDim, ffMult)
  result.ln1 = newLayerNorm(embedDim)
  result.ln2 = newLayerNorm(embedDim)

proc forward*(blk: TransformerBlockLinearPP, x: Tensor, ctx: Backend): Tensor =
  let x1 = blk.ln1.forward(x, ctx)
  let attnOut = forwardLinearPlus(blk.attn, x1, ctx)
  let x2 = addT(ctx, x, attnOut)
  let x3 = blk.ln2.forward(x2, ctx)
  let ffOut = blk.ff.forward(x3, ctx)
  result = addT(ctx, x2, ffOut)

proc backward*(blk: TransformerBlockLinearPP, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dAttnQkvW, dAttnQkvB, dAttnOutW, dAttnOutB,
    dFf1W, dFf1B, dFf2W, dFf2B,
    dLn1G, dLn1B, dLn2G, dLn2B: Tensor] =
  let x1 = blk.ln1.forward(x, ctx)
  let attnOut = forwardLinearPlus(blk.attn, x1, ctx)
  let x2 = addT(ctx, x, attnOut)
  let x3 = blk.ln2.forward(x2, ctx)

  let dFF = blk.ff.backward(x3, dOut, ctx)
  let dLN2 = blk.ln2.backward(x2, dFF.dX, ctx)
  let dX2Total = addT(ctx, dLN2.dX, dOut)     # residual x2 = x + attnOut

  let dAttn = backwardLinearPlus(blk.attn, x1, dX2Total, ctx)
  let dLN1 = blk.ln1.backward(x, dAttn.dX, ctx)
  let dX0 = addT(ctx, dLN1.dX, dX2Total)      # original residual

  result = (
    dX0,
    dAttn.dQkvW, dAttn.dQkvB, dAttn.dOutW, dAttn.dOutB,
    dFF.dFc1W, dFF.dFc1B, dFF.dFc2W, dFF.dFc2B,
    dLN1.dGamma, dLN1.dBeta, dLN2.dGamma, dLN2.dBeta
  )

# ═══════════════════════════════════════════════════════════════
# MAMBA-2 SSD - DÙNG SESSION MATMUL (KHÔNG SCAN, KHÔNG KERNEL MỚI)
# ═══════════════════════════════════════════════════════════════

proc buildDecayMask(T: int, a: float32): seq[float32] =
  ## L[i,j] = a^(i-j) if i>=j, else 0.
  result = newSeq[float32](T * T)
  for i in 0 ..< T:
    var v = 1'f32
    for j in countdown(i, 0):
      result[i * T + j] = v
      v *= a

proc cpuMamba2Forward*(xgData, bData, cData: seq[float32], B, H, T, D: int,
                       aVals: seq[float32]): seq[float32] =
  let C = H * D
  result = newSeq[float32](B * T * C)
  template idx(b, t, h, d: int): int = b * T * C + t * C + h * D + d
  for h in 0 ..< H:
    let mask = buildDecayMask(T, aVals[h])
    for b in 0 ..< B:
      for i in 0 ..< T:
        for j in 0 .. i:
          let w = mask[i * T + j]
          if w == 0'f32: continue
          var score = 0'f32
          for d in 0 ..< D:
            score += cData[idx(b, i, h, d)] * bData[idx(b, j, h, d)]
          score *= w
          for d in 0 ..< D:
            result[idx(b, i, h, d)] += score * xgData[idx(b, j, h, d)]

proc cpuMamba2Backward*(xgData, bData, cData, dOutData: seq[float32], B, H, T, D: int,
                        aVals: seq[float32]): tuple[dXg, dB, dC: seq[float32]] =
  let C = H * D
  result.dXg = newSeq[float32](B * T * C)
  result.dB = newSeq[float32](B * T * C)
  result.dC = newSeq[float32](B * T * C)
  template idx(b, t, h, d: int): int = b * T * C + t * C + h * D + d
  for h in 0 ..< H:
    let mask = buildDecayMask(T, aVals[h])
    for b in 0 ..< B:
      for i in 0 ..< T:
        for j in 0 .. i:
          let w = mask[i * T + j]
          if w == 0'f32: continue
          var cb = 0'f32
          for d in 0 ..< D:
            cb += cData[idx(b, i, h, d)] * bData[idx(b, j, h, d)]
          var dScore = 0'f32
          for d in 0 ..< D:
            dScore += dOutData[idx(b, i, h, d)] * xgData[idx(b, j, h, d)]
          dScore *= w
          for d in 0 ..< D:
            result.dXg[idx(b, j, h, d)] += (cb * w) * dOutData[idx(b, i, h, d)]
            result.dC[idx(b, i, h, d)] += dScore * bData[idx(b, j, h, d)]
            result.dB[idx(b, j, h, d)] += dScore * cData[idx(b, i, h, d)]

type
  Mamba2SSD* = object
    nHeads*, headDim*, embedDim*: int
    inProj*: Linear
    outProj*: Linear
    aVals*: seq[float32]
    useSession*: bool

proc newMamba2SSD*(embedDim, nHeads: int, useSession: bool = true): Mamba2SSD =
  assert embedDim mod nHeads == 0
  result.embedDim = embedDim
  result.nHeads = nHeads
  result.headDim = embedDim div nHeads
  result.inProj = newLinear(embedDim, 4 * embedDim)
  result.outProj = newLinear(embedDim, embedDim)
  result.aVals = newSeq[float32](nHeads)
  for h in 0 ..< nHeads:
    result.aVals[h] = 0.7'f32 + 0.28'f32 * (float32(h) / float32(max(nHeads - 1, 1)))
  result.useSession = useSession

proc splitXgBC(proj: Tensor, B, T, C: int): tuple[xg, b, c, gate: seq[float32]] =
  result.xg = newSeq[float32](B * T * C)
  result.b = newSeq[float32](B * T * C)
  result.c = newSeq[float32](B * T * C)
  result.gate = newSeq[float32](B * T * C)
  for bb in 0 ..< B:
    let baseIn = bb * T * 4 * C
    let baseOut = bb * T * C
    for t in 0 ..< T:
      for cc in 0 ..< C:
        result.xg[baseOut + t * C + cc]   = proj.data[baseIn + t * 4 * C + cc]
        result.b[baseOut + t * C + cc]    = proj.data[baseIn + t * 4 * C + C + cc]
        result.c[baseOut + t * C + cc]    = proj.data[baseIn + t * 4 * C + 2 * C + cc]
        result.gate[baseOut + t * C + cc] = proj.data[baseIn + t * 4 * C + 3 * C + cc]

proc mamba2SsdGpu*(ctx: Backend; xgData, bData, cData, dYData: seq[float32];
                  B, H, T, D: int; aVals: seq[float32]):
    tuple[y, dXg, dB, dC: seq[float32]] =
  ## "Quadratic" SSD on the GPU, 1 session per (b, h):
  ##   S  = M o (C B^T)          (M = decay mask a^(i-j), j <= i)
  ##   Y  = S Xg
  ## If dYData is not empty (backward), also:
  ##   dXg = S^T dY,  S^T = M^T o (B C^T)
  ##   dS  = M o (dY Xg^T)  -> dC = dS B
  ##   dS^T = M^T o (Xg dY^T) -> dB = dS^T C
  ## Every A B^T product uses sessionMatmulNT (no transpose on the host). Splitting [T,D]
  ## slices per head is only data arrangement. GPU error -> raise.
  let backend = ctx.kind.toByby()
  let C = H * D
  let bwd = dYData.len > 0
  result.y = newSeq[float32](B * T * C)
  if bwd:
    result.dXg = newSeq[float32](B * T * C)
    result.dB = newSeq[float32](B * T * C)
    result.dC = newSeq[float32](B * T * C)
  proc slice(src: seq[float32]; b, h: int): seq[float32] =
    result = newSeq[float32](T * D)
    for t in 0 ..< T:
      let off = b * T * C + t * C + h * D
      for d in 0 ..< D: result[t * D + d] = src[off + d]
  proc scatter(dst: var seq[float32]; raw: seq[float32]; b, h: int) =
    for t in 0 ..< T:
      let off = b * T * C + t * C + h * D
      for d in 0 ..< D: dst[off + d] = raw[t * D + d]
  for h in 0 ..< H:
    let mask = buildDecayMask(T, aVals[h])
    var maskT = newSeq[float32](T * T)
    if bwd:
      for i in 0 ..< T:
        for j in 0 ..< T: maskT[j * T + i] = mask[i * T + j]
    for b in 0 ..< B:
      if not sessionBegin(backend): gpuFail("sessionBegin (Mamba2SSD)", backend)
      var owned: seq[SessionHandle]
      proc up(v: seq[float32]): SessionHandle =
        result = sessionUpload(backend, v); owned.add result
      proc scratch(n: int): SessionHandle =
        result = sessionAllocScratch(backend, n); owned.add result
      var ok = true
      let hM = up(mask)
      let hC = up(slice(cData, b, h))
      let hB = up(slice(bData, b, h))
      let hXg = up(slice(xgData, b, h))
      var hRaw = scratch(T * T)
      var hS = scratch(T * T)
      var hY = scratch(T * D)
      ok = ok and sessionMatmulNT(backend, hC, hB, hRaw, T, D, T)        # C B^T
      ok = ok and sessionVecOp(backend, "mul", hRaw, hM, hS, T * T)       # S
      ok = ok and sessionMatmul(backend, hS, hXg, hY, T, T, D)            # Y
      var hDXg, hDC, hDB: SessionHandle
      if bwd:
        let hMT = up(maskT)
        let hDY = up(slice(dYData, b, h))
        var hRawT = scratch(T * T)
        var hST = scratch(T * T)
        var hDSraw = scratch(T * T)
        var hDS = scratch(T * T)
        var hDSTraw = scratch(T * T)
        var hDST = scratch(T * T)
        hDXg = scratch(T * D); hDC = scratch(T * D); hDB = scratch(T * D)
        ok = ok and sessionMatmulNT(backend, hB, hC, hRawT, T, D, T)       # B C^T
        ok = ok and sessionVecOp(backend, "mul", hRawT, hMT, hST, T * T)   # S^T
        ok = ok and sessionMatmul(backend, hST, hDY, hDXg, T, T, D)        # dXg
        ok = ok and sessionMatmulNT(backend, hDY, hXg, hDSraw, T, D, T)    # dY Xg^T
        ok = ok and sessionVecOp(backend, "mul", hDSraw, hM, hDS, T * T)   # dS
        ok = ok and sessionMatmul(backend, hDS, hB, hDC, T, T, D)          # dC
        ok = ok and sessionMatmulNT(backend, hXg, hDY, hDSTraw, T, D, T)   # Xg dY^T
        ok = ok and sessionVecOp(backend, "mul", hDSTraw, hMT, hDST, T * T) # dS^T
        ok = ok and sessionMatmul(backend, hDST, hC, hDB, T, T, D)         # dB
      let ended = sessionEnd(backend)
      if not ok or not ended:
        for hd in owned.mitems: sessionFree(backend, hd)
        gpuFail("encode/sessionEnd (Mamba2SSD)", backend)
      scatter(result.y, sessionRead(backend, hY, T * D), b, h)
      if bwd:
        scatter(result.dXg, sessionRead(backend, hDXg, T * D), b, h)
        scatter(result.dC, sessionRead(backend, hDC, T * D), b, h)
        scatter(result.dB, sessionRead(backend, hDB, T * D), b, h)
      for hd in owned.mitems: sessionFree(backend, hd)

proc forwardMamba2*(m: Mamba2SSD, x: Tensor, ctx: Backend): Tensor =
  let B = x.shape[0]; let T = x.shape[1]; let C = x.shape[2]
  assert C == m.embedDim
  let H = m.nHeads; let D = m.headDim

  let proj = m.inProj.forward(x, ctx)
  let (xgData, bData, cData, gateData) = splitXgBC(proj, B, T, C)

  let yData = if isGpuCtx(ctx): mamba2SsdGpu(ctx, xgData, bData, cData, @[], B, H, T, D, m.aVals).y
              else: cpuMamba2Forward(xgData, bData, cData, B, H, T, D, m.aVals)

  let gateAct = beSigmoid(ctx, gateData)
  let gated = beMul(ctx, yData, gateAct)

  var y = newTensor(@[B, T, C])
  y.data = gated
  result = m.outProj.forward(y, ctx)

proc backwardMamba2*(m: Mamba2SSD, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dInProjW, dInProjB, dOutProjW, dOutProjB: Tensor] =
  let B = x.shape[0]; let T = x.shape[1]; let C = x.shape[2]
  let H = m.nHeads; let D = m.headDim

  let proj = m.inProj.forward(x, ctx)
  let (xgData, bData, cData, gateData) = splitXgBC(proj, B, T, C)

  let gpu = isGpuCtx(ctx)
  let yData = if gpu: mamba2SsdGpu(ctx, xgData, bData, cData, @[], B, H, T, D, m.aVals).y
              else: cpuMamba2Forward(xgData, bData, cData, B, H, T, D, m.aVals)
  let gateAct = beSigmoid(ctx, gateData)
  let gated = beMul(ctx, yData, gateAct)
  var y = newTensor(@[B, T, C]); y.data = gated

  let dOutP = m.outProj.backward(y, dOut, ctx)
  let dGated = dOutP.dX.data

  # gated = y * sigmoid(g):  dY = dGated*s,  dg = dGated*y*s*(1-s)
  let dY = beMul(ctx, dGated, gateAct)
  let sPrime = beSub(ctx, gateAct, beMul(ctx, gateAct, gateAct))
  let dGateRaw = beMul(ctx, beMul(ctx, dGated, yData), sPrime)

  let (dXg, dB, dC) =
    if gpu:
      let r = mamba2SsdGpu(ctx, xgData, bData, cData, dY, B, H, T, D, m.aVals)
      (r.dXg, r.dB, r.dC)
    else: cpuMamba2Backward(xgData, bData, cData, dY, B, H, T, D, m.aVals)

  var dProj = newTensor(proj.shape)
  for b in 0 ..< B:
    let baseIn = b * T * 4 * C
    let baseOut = b * T * C
    for t in 0 ..< T:
      for c in 0 ..< C:
        dProj.data[baseIn + t * 4 * C + c]        = dXg[baseOut + t * C + c]
        dProj.data[baseIn + t * 4 * C + C + c]    = dB[baseOut + t * C + c]
        dProj.data[baseIn + t * 4 * C + 2 * C + c] = dC[baseOut + t * C + c]
        dProj.data[baseIn + t * 4 * C + 3 * C + c] = dGateRaw[baseOut + t * C + c]

  let dInP = m.inProj.backward(x, dProj, ctx)
  result = (dInP.dX, dInP.dW, dInP.dB, dOutP.dW, dOutP.dB)

# ═══════════════════════════════════════════════════════════════
# SPECTRON BLOCK: LinearPlusAttention + Mamba2SSD, GATED FUSION
# ═══════════════════════════════════════════════════════════════

type
  TransformerBlockSpectron* = object
    attn*: LinearPlusAttention
    mamba*: Mamba2SSD
    fuseGate*: Linear
    ff*: FeedForward
    ln1*, ln2*: LayerNorm

proc newTransformerBlockSpectron*(embedDim, nHeads, ffMult: int, useSession: bool): TransformerBlockSpectron =
  result.attn = newLinearPlusAttention(embedDim, nHeads, useSession)
  result.mamba = newMamba2SSD(embedDim, nHeads, useSession)
  result.fuseGate = newLinear(embedDim * 2, embedDim)
  result.ff = newFeedForward(embedDim, ffMult)
  result.ln1 = newLayerNorm(embedDim)
  result.ln2 = newLayerNorm(embedDim)

proc concatLastDim*(a, b: Tensor; ctx: Backend): Tensor =
  let B = a.shape[0]; let T = a.shape[1]; let C = a.shape[2]
  result = newTensor(@[B, T, 2 * C])
  if isGpuCtx(ctx):
    result.data = beConcatGpu(ctx, a.data & b.data, @[0'i32, int32(B * T * C)],
                              @[int32(C), int32(C)], @[0'i32, int32(C)], 2 * C, 1, B * T * 2 * C)
    return
  for i in 0 ..< B * T:
    for c in 0 ..< C:
      result.data[i * 2 * C + c] = a.data[i * C + c]
      result.data[i * 2 * C + C + c] = b.data[i * C + c]

proc splitLastDim*(x: Tensor, C: int; ctx: Backend): tuple[a, b: Tensor] =
  ## Inverse of concatLastDim: [B,T,2C] -> 2 x [B,T,C].
  let B = x.shape[0]; let T = x.shape[1]
  result.a = newTensor(@[B, T, C])
  result.b = newTensor(@[B, T, C])
  if isGpuCtx(ctx):
    result.a.data = beSplitSliceGpu(ctx, x.data, 2 * C, C, 0, 1, B * T * C)
    result.b.data = beSplitSliceGpu(ctx, x.data, 2 * C, C, C, 1, B * T * C)
    return
  for i in 0 ..< B * T:
    for c in 0 ..< C:
      result.a.data[i * C + c] = x.data[i * 2 * C + c]
      result.b.data[i * C + c] = x.data[i * 2 * C + C + c]

proc gateMix(ctx: Backend; gate: seq[float32]; a, b: Tensor): Tensor =
  ## gate*a + (1-gate)*b = b + gate*(a-b), GPU elementwise.
  result = newTensor(a.shape)
  result.data = beAdd(ctx, b.data, beMul(ctx, gate, beSub(ctx, a.data, b.data)))

type
  SpectronBlockForward = object
    x, x1, attnOut, mambaOut, fuseIn, gate, x2, x3: Tensor
  SpectronBlockGrads = tuple[
    dX: Tensor,
    dAttnQkvW, dAttnQkvB, dAttnOutW, dAttnOutB: Tensor,
    dMambaInW, dMambaInB, dMambaOutW, dMambaOutB: Tensor,
    dFuseW, dFuseB: Tensor,
    dFf1W, dFf1B, dFf2W, dFf2B: Tensor,
    dLn1G, dLn1B, dLn2G, dLn2B: Tensor]

proc blockForward(blk: TransformerBlockSpectron, x: Tensor, ctx: Backend): SpectronBlockForward =
  result.x = x
  result.x1 = blk.ln1.forward(x, ctx)
  result.attnOut = forwardLinearPlus(blk.attn, result.x1, ctx)
  result.mambaOut = forwardMamba2(blk.mamba, result.x1, ctx)
  result.fuseIn = concatLastDim(result.attnOut, result.mambaOut, ctx)
  result.gate = blk.fuseGate.forward(result.fuseIn, ctx)
  result.gate.data = beSigmoid(ctx, result.gate.data)
  let combined = gateMix(ctx, result.gate.data, result.attnOut, result.mambaOut)
  result.x2 = addT(ctx, x, combined)
  result.x3 = blk.ln2.forward(result.x2, ctx)

proc blockOutput(blk: TransformerBlockSpectron, activations: SpectronBlockForward,
                 ctx: Backend): Tensor =
  addT(ctx, activations.x2, blk.ff.forward(activations.x3, ctx))

proc forward*(blk: TransformerBlockSpectron, x: Tensor, ctx: Backend): Tensor =
  blk.blockOutput(blk.blockForward(x, ctx), ctx)

proc blockBackward(blk: TransformerBlockSpectron, cache: SpectronBlockForward,
                   dOut: Tensor, ctx: Backend): SpectronBlockGrads =
  let C = cache.x.shape[2]
  let gate = cache.gate.data

  # ---- backward FF + LN2 ----
  let dFF = blk.ff.backward(cache.x3, dOut, ctx)
  let dLN2 = blk.ln2.backward(cache.x2, dFF.dX, ctx)
  let dX2 = addT(ctx, dLN2.dX, dOut)   # residual x2 = x + combined

  # ---- backward gated fusion (GPU elementwise) ----
  let dGate = beMul(ctx, dX2.data, beSub(ctx, cache.attnOut.data, cache.mambaOut.data))
  let dAttnOutPartial = beMul(ctx, dX2.data, gate)
  let dMambaOutPartial = beSub(ctx, dX2.data, dAttnOutPartial)          # dX2*(1-gate)

  var dGateLogit = newTensor(cache.gate.shape)
  dGateLogit.data = beMul(ctx, dGate, beSub(ctx, gate, beMul(ctx, gate, gate)))   # sigmoid'

  let dFuse = blk.fuseGate.backward(cache.fuseIn, dGateLogit, ctx)
  let (dAttnOutFuse, dMambaOutFuse) = splitLastDim(dFuse.dX, C, ctx)

  var dAttnOutTotal = newTensor(cache.attnOut.shape)
  var dMambaOutTotal = newTensor(cache.mambaOut.shape)
  dAttnOutTotal.data = beAdd(ctx, dAttnOutPartial, dAttnOutFuse.data)
  dMambaOutTotal.data = beAdd(ctx, dMambaOutPartial, dMambaOutFuse.data)

  # ---- backward of the 2 branches (Linear++ attn and Mamba2) ----
  let dAttn = backwardLinearPlus(blk.attn, cache.x1, dAttnOutTotal, ctx)
  let dMamba = backwardMamba2(blk.mamba, cache.x1, dMambaOutTotal, ctx)

  let dX1Total = addT(ctx, dAttn.dX, dMamba.dX)

  let dLN1 = blk.ln1.backward(cache.x, dX1Total, ctx)
  let dX0 = addT(ctx, dLN1.dX, dX2)    # original residual x2 = x + combined

  result = (
    dX0,
    dAttn.dQkvW, dAttn.dQkvB, dAttn.dOutW, dAttn.dOutB,
    dMamba.dInProjW, dMamba.dInProjB, dMamba.dOutProjW, dMamba.dOutProjB,
    dFuse.dW, dFuse.dB,
    dFF.dFc1W, dFF.dFc1B, dFF.dFc2W, dFF.dFc2B,
    dLN1.dGamma, dLN1.dBeta, dLN2.dGamma, dLN2.dBeta
  )

proc backward*(blk: TransformerBlockSpectron, x: Tensor, dOut: Tensor,
               ctx: Backend): SpectronBlockGrads =
  blk.blockBackward(blk.blockForward(x, ctx), dOut, ctx)

# ═══════════════════════════════════════════════════════════════
# NIMFORMER MODEL VỚI SPECTRON BLOCKS (embed -> N blocks -> lm head)
# ═══════════════════════════════════════════════════════════════

type SpectronNumericalError* = object of CatchableError

func finite32(value: float32): bool {.inline.} =
  (cast[uint32](value) and 0x7f800000'u32) != 0x7f800000'u32

include spectron_moe

type
  HrmConfig* = object
    ## Hierarchical Reasoning Model core (Wang et al., 2025, arXiv:2506.21734).
    ## blocks[0 ..< lLayers] form the fast low-level module L and the remaining
    ## blocks form the slow high-level module H. Each segment runs `cycles`
    ## high-level updates; each one follows `lSteps` low-level updates:
    ##   zL = L(zL + zH + embed(x))      (lSteps times)
    ##   zH = H(zH + zL)                 (once per cycle)
    ## Each module ends with its own LayerNorm, so zL and zH stay normalized
    ## across iterations like the post-norm states in the paper. The injected
    ## embeddings pass through a third LayerNorm so the token signal has the
    ## same scale as the states (the paper scales embeddings by sqrt(d)).
    ## Training uses the one-step gradient: only the last L and the last H
    ## update keep activations, and earlier states are treated as constants.
    ## `segments` repeats the whole computation from the previous segment's
    ## detached state (deep supervision). cycles == 0 disables HRM.
    lLayers*, cycles*, lSteps*, segments*: int

  NimformerModelSpectron* = object
    embed*: Embedding
    blocks*: seq[TransformerBlockSpectron]
    finalNorm*: LayerNorm
    useFinalNorm*: bool
    outProj*: Linear
    vocab*: int
    mixture: SpectronMixture
    hrm*: HrmConfig
    hrmNormL*, hrmNormH*, hrmNormX*: LayerNorm

  HrmCarry = object
    ## Recurrent HRM state between segments; empty tensors mean zero state.
    zH, zL: Tensor

proc hrmEnabled*(m: NimformerModelSpectron): bool {.inline.} = m.hrm.cycles > 0

proc expertCount*(m: NimformerModelSpectron): int {.inline.} = m.mixture.experts.len

proc validateHrm*(m: NimformerModelSpectron) =
  if not m.hrmEnabled: return
  let h = m.hrm
  if h.lLayers < 1 or h.lLayers >= m.blocks.len or h.lSteps < 1 or h.segments < 1:
    raise newException(ValueError,
      "HRM needs 1 <= l-layers < total layers, and at least one L step and one segment")

const HrmResidualInitScale* = 0.1'f32

proc initHrmResiduals*(m: var NimformerModelSpectron, scale = HrmResidualInitScale) =
  ## Shrinks every block's residual branch outputs for a freshly initialized HRM
  ## core. Unscaled random blocks make the recurrent L/H map chaotic: a 2-cycle,
  ## 2-step, 2-segment core amplified a 1e-4 weight change into logit changes
  ## larger than the logits themselves, so GPU rounding alone changed results.
  ## Near-identity blocks start the recurrence contracting; training grows them.
  for blk in m.blocks.mitems:
    for t in [addr blk.attn.outProj.weight, addr blk.attn.outProj.bias,
              addr blk.mamba.outProj.weight, addr blk.mamba.outProj.bias,
              addr blk.ff.fc2.weight, addr blk.ff.fc2.bias]:
      for w in t[].data.mitems: w *= scale

proc runHrmLevel(m: NimformerModelSpectron, first, last: int, input: Tensor, ctx: Backend,
                 cache: ptr seq[SpectronBlockForward]): Tensor =
  result = input
  for i in first ..< last:
    if cache == nil:
      result = m.blocks[i].forward(result, ctx)
    else:
      cache[][i] = m.blocks[i].blockForward(result, ctx)
      result = m.blocks[i].blockOutput(cache[][i], ctx)

proc hrmSegment(m: NimformerModelSpectron, embedded: Tensor, carry: var HrmCarry,
                ctx: Backend, cache: ptr seq[SpectronBlockForward] = nil,
                rawL, rawH: ptr Tensor = nil) =
  ## One HRM segment. With `cache`, the final L and H updates record their
  ## block activations and their outputs before the module norm (`rawL`,
  ## `rawH`) for the one-step gradient; all earlier updates run forward only.
  if carry.zH.data.len == 0:
    carry.zH = newTensor(embedded.shape)
    carry.zL = newTensor(embedded.shape)
  let nL = m.hrm.lLayers
  for cycle in 0 ..< m.hrm.cycles:
    let lastCycle = cycle == m.hrm.cycles - 1
    for step in 0 ..< m.hrm.lSteps:
      let tracked = if lastCycle and step == m.hrm.lSteps - 1: cache else: nil
      let injected = addT(ctx, addT(ctx, carry.zL, carry.zH), embedded)
      let lOut = m.runHrmLevel(0, nL, injected, ctx, tracked)
      if tracked != nil: rawL[] = lOut
      carry.zL = m.hrmNormL.forward(lOut, ctx)
    let tracked = if lastCycle: cache else: nil
    let hOut = m.runHrmLevel(nL, m.blocks.len, addT(ctx, carry.zH, carry.zL), ctx, tracked)
    if tracked != nil: rawH[] = hOut
    carry.zH = m.hrmNormH.forward(hOut, ctx)

proc newNimformerModelSpectron*(vocab, embedDim, nHeads, nLayers, ffMult: int,
                                 useSession: bool = true): NimformerModelSpectron =
  result.vocab = vocab
  result.embed = newEmbedding(vocab, embedDim)
  result.blocks = newSeq[TransformerBlockSpectron](nLayers)
  for i in 0 ..< nLayers:
    result.blocks[i] = newTransformerBlockSpectron(embedDim, nHeads, ffMult, useSession)
  result.finalNorm = newLayerNorm(embedDim)
  result.useFinalNorm = true
  result.outProj = newLinear(embedDim, vocab)
  result.hrmNormL = newLayerNorm(embedDim)
  result.hrmNormH = newLayerNorm(embedDim)
  result.hrmNormX = newLayerNorm(embedDim)

proc forwardBatch*(m: NimformerModelSpectron, idsBatch: seq[seq[int]], ctx: Backend): Tensor =
  var x = m.embed.lookupBatch(idsBatch, ctx)
  if m.hrmEnabled:
    var carry: HrmCarry
    let injected = m.hrmNormX.forward(x, ctx)
    for _ in 0 ..< m.hrm.segments:
      m.hrmSegment(injected, carry, ctx)
    x = carry.zH
  else:
    for blk in m.blocks:
      x = blk.forward(x, ctx)
  if m.mixture.experts.len > 0:
    x = m.mixture.forwardMixture(x, ctx, false).output
  if m.useFinalNorm:
    x = m.finalNorm.forward(x, ctx)
  result = m.outProj.forward(x, ctx)

proc forward*(m: NimformerModelSpectron, ids: seq[int], ctx: Backend): Tensor =
  let logits = m.forwardBatch(@[ids], ctx)
  result = Tensor(data: logits.data, shape: logits.shape[1 .. ^1])

type
  SpectronGrads* = object
    ## Collects all model gradients in the exact order they will be updated -
    ## a "flat" array of (param Tensor, gradient Tensor) pairs so train()
    ## can loop over them and call apfAdamStep on each without knowing
    ## the inner structure of each block.
    dOutProjW*, dOutProjB*: Tensor
    dFinalNormG*, dFinalNormB*: Tensor
    dBlocks*: seq[tuple[
      dAttnQkvW, dAttnQkvB, dAttnOutW, dAttnOutB,
      dMambaInW, dMambaInB, dMambaOutW, dMambaOutB,
      dFuseW, dFuseB,
      dFf1W, dFf1B, dFf2W, dFf2B,
      dLn1G, dLn1B, dLn2G, dLn2B: Tensor]]
    dEmbed*: Tensor
    dHrmNormLG*, dHrmNormLB*, dHrmNormHG*, dHrmNormHB*, dHrmNormXG*, dHrmNormXB*: Tensor
    mixture: MixtureGrad

type SpectronForward = object
  blocks: seq[SpectronBlockForward]
  hidden, headInput: Tensor
  hrmRawL, hrmRawH: Tensor   # final L/H outputs before their module norms
  hrmEmbed: Tensor           # embeddings before the injection norm
  mixture: MixtureForward

proc forwardForBackward(m: NimformerModelSpectron, idsBatch: seq[seq[int]],
                        ctx: Backend, carry: var HrmCarry): SpectronForward =
  ## With HRM, `carry` enters as the previous segment's state and leaves as
  ## this segment's state. Input injection is an addition, so in backward the
  ## gradient at the H input is the gradient of the final zL, and the gradient
  ## at the L input is the gradient of the embeddings (carried states are constants).
  var x = m.embed.lookupBatch(idsBatch, ctx)
  result.blocks = newSeq[SpectronBlockForward](m.blocks.len)
  if m.hrmEnabled:
    result.hrmEmbed = x
    m.hrmSegment(m.hrmNormX.forward(x, ctx), carry, ctx, addr result.blocks,
                 addr result.hrmRawL, addr result.hrmRawH)
    x = carry.zH
  else:
    for i, blk in m.blocks:
      result.blocks[i] = blk.blockForward(x, ctx)
      x = blk.blockOutput(result.blocks[i], ctx)
  if m.mixture.experts.len > 0:
    result.mixture = m.mixture.forwardMixture(x, ctx, true)
    x = result.mixture.output
  result.hidden = x
  result.headInput = if m.useFinalNorm: m.finalNorm.forward(x, ctx) else: x

proc forwardForBackward(m: NimformerModelSpectron, idsBatch: seq[seq[int]],
                        ctx: Backend): SpectronForward =
  var carry: HrmCarry
  m.forwardForBackward(idsBatch, ctx, carry)

proc backwardCached(m: NimformerModelSpectron, idsBatch: seq[seq[int]], dLoss: Tensor,
                    cache: SpectronForward, ctx: Backend): SpectronGrads =
  # The SSM branch is unnormalized; residual magnitudes must not set the
  # softmax temperature. Backpropagate through the same final norm as forward.
  let dOutProj = m.outProj.backward(cache.headInput, dLoss, ctx)
  result.dOutProjW = dOutProj.dW
  result.dOutProjB = dOutProj.dB
  var dOut = dOutProj.dX
  if m.useFinalNorm:
    let dNorm = m.finalNorm.backward(cache.hidden, dOut, ctx)
    dOut = dNorm.dX
    result.dFinalNormG = dNorm.dGamma
    result.dFinalNormB = dNorm.dBeta
  if m.mixture.experts.len > 0:
    result.mixture = m.mixture.backwardMixture(cache.mixture, dOut, ctx)
    dOut = result.mixture.dx

  result.dBlocks = newSeq[typeof(result.dBlocks[0])](m.blocks.len)
  proc blocksBackward(m: NimformerModelSpectron, first, last: int, dOut: var Tensor,
                      grads: var SpectronGrads) =
    for i in countdown(last - 1, first):
      let dBlk = m.blocks[i].blockBackward(cache.blocks[i], dOut, ctx)
      dOut = dBlk.dX
      grads.dBlocks[i] = (
        dBlk.dAttnQkvW, dBlk.dAttnQkvB, dBlk.dAttnOutW, dBlk.dAttnOutB,
        dBlk.dMambaInW, dBlk.dMambaInB, dBlk.dMambaOutW, dBlk.dMambaOutB,
        dBlk.dFuseW, dBlk.dFuseB,
        dBlk.dFf1W, dBlk.dFf1B, dBlk.dFf2W, dBlk.dFf2B,
        dBlk.dLn1G, dBlk.dLn1B, dBlk.dLn2G, dBlk.dLn2B
      )
  if m.hrmEnabled:
    let nL = m.hrm.lLayers
    let dNormH = m.hrmNormH.backward(cache.hrmRawH, dOut, ctx)
    result.dHrmNormHG = dNormH.dGamma
    result.dHrmNormHB = dNormH.dBeta
    dOut = dNormH.dX
    m.blocksBackward(nL, m.blocks.len, dOut, result)
    let dNormL = m.hrmNormL.backward(cache.hrmRawL, dOut, ctx)
    result.dHrmNormLG = dNormL.dGamma
    result.dHrmNormLB = dNormL.dBeta
    dOut = dNormL.dX
    m.blocksBackward(0, nL, dOut, result)
    let dNormX = m.hrmNormX.backward(cache.hrmEmbed, dOut, ctx)
    result.dHrmNormXG = dNormX.dGamma
    result.dHrmNormXB = dNormX.dBeta
    dOut = dNormX.dX
  else:
    m.blocksBackward(0, m.blocks.len, dOut, result)
  result.dEmbed = m.embed.backward(idsBatch, dOut, ctx)

proc backwardBatch*(m: NimformerModelSpectron, idsBatch: seq[seq[int]], dLoss: Tensor,
                    ctx: Backend): SpectronGrads =
  m.backwardCached(idsBatch, dLoss, m.forwardForBackward(idsBatch, ctx), ctx)

proc backward*(m: NimformerModelSpectron, ids: seq[int], dLoss: Tensor, ctx: Backend): SpectronGrads =
  let dLossBatch = Tensor(data: dLoss.data, shape: @[1] & dLoss.shape)
  result = m.backwardBatch(@[ids], dLossBatch, ctx)

# ═══════════════════════════════════════════════════════════════
# LOSS: softmax cross-entropy (batch), returns the mean loss + dLogits
# ═══════════════════════════════════════════════════════════════
proc checkedCrossEntropy(logits: Tensor, targets: seq[seq[int]], ctx: Backend):
    tuple[loss: float32, dLogits: Tensor, nValid: int] =
  ## logits: [B,T,V]. A negative target skips that row on either backend.
  let B = logits.shape[0]; let T = logits.shape[1]; let V = logits.shape[2]
  if targets.len != B:
    raise newException(ValueError, "Spectron targets do not match the batch size")
  let gpu = isGpuCtx(ctx)
  var tgt: seq[float32]
  if gpu: tgt = newSeq[float32](B * T)
  for b in 0 ..< B:
    if targets[b].len != T:
      raise newException(ValueError, "Spectron targets do not match the context length")
    for t in 0 ..< T:
      let target = targets[b][t]
      if target >= V:
        raise newException(ValueError, "Spectron target is outside the vocabulary")
      let row = b * T + t
      if gpu: tgt[row] = float32(max(target, -1))
      if target < 0: continue
      inc result.nValid
      # Validate on the host before calling the GPU. A fused GPU math operation
      # can otherwise mask an invalid input under fast-math.
      if gpu:
        for i in row * V ..< (row + 1) * V:
          if not finite32(logits.data[i]):
            raise newException(SpectronNumericalError, "Non-finite Spectron logits at row " & $row)
  if gpu and result.nValid > 0:
    let ce = softmaxCrossEntropyGpu(ctx, logits.data, tgt, B * T, V, result.nValid)
    result.loss = ce.loss
    result.dLogits = Tensor(data: ce.dLogits, shape: logits.shape)
  else:
    result.dLogits = newTensor(logits.shape)
    if result.nValid == 0: return
    var totalLoss = 0.0
    let invCount = 1.0 / float64(result.nValid)
    for b in 0 ..< B:
      for t in 0 ..< T:
        let target = targets[b][t]
        if target < 0: continue
        let row = b * T + t
        let base = row * V
        var peak = float64(logits.data[base])
        for i in base ..< base + V:
          let value = logits.data[i]
          if not finite32(value):
            raise newException(SpectronNumericalError, "Non-finite Spectron logits at row " & $row)
          peak = max(peak, float64(value))
        var expSum = 0.0
        for i in base ..< base + V:
          let e = exp(float64(logits.data[i]) - peak)
          expSum += e
          result.dLogits.data[i] = float32(e)
        totalLoss += ((peak - float64(logits.data[base + target])) + ln(expSum)) * invCount
        let weight = float32(invCount / expSum)
        for i in base ..< base + V:
          result.dLogits.data[i] *= weight
        result.dLogits.data[base + target] -= float32(invCount)
    result.loss = float32(totalLoss)
  if not finite32(result.loss):
    raise newException(SpectronNumericalError, "Non-finite Spectron cross-entropy loss")

proc softmaxCrossEntropy*(logits: Tensor, targets: seq[seq[int]], ctx: Backend):
    tuple[loss: float32, dLogits: Tensor] =
  let ce = checkedCrossEntropy(logits, targets, ctx)
  (ce.loss, ce.dLogits)

# ═══════════════════════════════════════════════════════════════
# ADAM STATE FOR THE WHOLE MODEL + 1 train() STEP
# ═══════════════════════════════════════════════════════════════
type
  ModelAdamState* = object
    outProjW*, outProjB*: ApfAdamState
    finalNormG*, finalNormB*: ApfAdamState
    blocks*: seq[tuple[
      attnQkvW, attnQkvB, attnOutW, attnOutB,
      mambaInW, mambaInB, mambaOutW, mambaOutB,
      fuseW, fuseB,
      ff1W, ff1B, ff2W, ff2B,
      ln1G, ln1B, ln2G, ln2B: ApfAdamState]]
    embed*: ApfAdamState
    hrmNormLG*, hrmNormLB*, hrmNormHG*, hrmNormHB*, hrmNormXG*, hrmNormXB*: ApfAdamState
    mixture: MixtureAdam

proc newModelAdamState*(m: NimformerModelSpectron): ModelAdamState =
  result.outProjW = newApfAdamState(m.outProj.weight.data.len)
  result.outProjB = newApfAdamState(m.outProj.bias.data.len)
  result.finalNormG = newApfAdamState(m.finalNorm.gamma.data.len)
  result.finalNormB = newApfAdamState(m.finalNorm.beta.data.len)
  result.embed = newApfAdamState(m.embed.weight.data.len)
  result.hrmNormLG = newApfAdamState(m.hrmNormL.gamma.data.len)
  result.hrmNormLB = newApfAdamState(m.hrmNormL.beta.data.len)
  result.hrmNormHG = newApfAdamState(m.hrmNormH.gamma.data.len)
  result.hrmNormHB = newApfAdamState(m.hrmNormH.beta.data.len)
  result.hrmNormXG = newApfAdamState(m.hrmNormX.gamma.data.len)
  result.hrmNormXB = newApfAdamState(m.hrmNormX.beta.data.len)
  if m.mixture.experts.len > 0:
    result.mixture = newMixtureAdam(m.mixture)
  result.blocks = newSeq[typeof(result.blocks[0])](m.blocks.len)
  for i, blk in m.blocks:
    result.blocks[i] = (
      newApfAdamState(blk.attn.qkvProj.weight.data.len),
      newApfAdamState(blk.attn.qkvProj.bias.data.len),
      newApfAdamState(blk.attn.outProj.weight.data.len),
      newApfAdamState(blk.attn.outProj.bias.data.len),
      newApfAdamState(blk.mamba.inProj.weight.data.len),
      newApfAdamState(blk.mamba.inProj.bias.data.len),
      newApfAdamState(blk.mamba.outProj.weight.data.len),
      newApfAdamState(blk.mamba.outProj.bias.data.len),
      newApfAdamState(blk.fuseGate.weight.data.len),
      newApfAdamState(blk.fuseGate.bias.data.len),
      newApfAdamState(blk.ff.fc1.weight.data.len),
      newApfAdamState(blk.ff.fc1.bias.data.len),
      newApfAdamState(blk.ff.fc2.weight.data.len),
      newApfAdamState(blk.ff.fc2.bias.data.len),
      newApfAdamState(blk.ln1.gamma.data.len),
      newApfAdamState(blk.ln1.beta.data.len),
      newApfAdamState(blk.ln2.gamma.data.len),
      newApfAdamState(blk.ln2.beta.data.len)
    )

# ═══════════════════════════════════════════════════════════════
# CHECKPOINT (.nimq) - REUSES the existing NIMQ2 binary format in
# quant.nim (saveQuantStateDict/loadQuantStateDict - the same functions export_to_nimq.py
# uses to export model.nimq for large models). Each tensor is saved here as
# qkFp32Raw (NOT compressed) because the demo model is small, favouring exact numbers over
# disk space. The mmap-backed trainer also saves its Adam state.
# ═══════════════════════════════════════════════════════════════

const
  MoeLayoutKey = "__moe_layout"
  HrmLayoutKey = "__hrm_layout"   # [lLayers, cycles, lSteps, segments]

proc modelToStateDict(m: NimformerModelSpectron): seq[(string, QuantTensor)] =
  result = @[]
  template add(name: string, t: Tensor) =
    result.add((name, quantizeTensor(t.data, t.shape, qkFp32Raw)))
  add("embed.weight", m.embed.weight)
  add("outProj.weight", m.outProj.weight)
  add("outProj.bias", m.outProj.bias)
  if m.useFinalNorm:
    add("finalNorm.gamma", m.finalNorm.gamma)
    add("finalNorm.beta", m.finalNorm.beta)
  for i, blk in m.blocks:
    let p = &"block{i}."
    add(p & "attn.qkvProj.weight", blk.attn.qkvProj.weight)
    add(p & "attn.qkvProj.bias", blk.attn.qkvProj.bias)
    add(p & "attn.outProj.weight", blk.attn.outProj.weight)
    add(p & "attn.outProj.bias", blk.attn.outProj.bias)
    add(p & "mamba.inProj.weight", blk.mamba.inProj.weight)
    add(p & "mamba.inProj.bias", blk.mamba.inProj.bias)
    add(p & "mamba.outProj.weight", blk.mamba.outProj.weight)
    add(p & "mamba.outProj.bias", blk.mamba.outProj.bias)
    add(p & "fuseGate.weight", blk.fuseGate.weight)
    add(p & "fuseGate.bias", blk.fuseGate.bias)
    add(p & "ff.fc1.weight", blk.ff.fc1.weight)
    add(p & "ff.fc1.bias", blk.ff.fc1.bias)
    add(p & "ff.fc2.weight", blk.ff.fc2.weight)
    add(p & "ff.fc2.bias", blk.ff.fc2.bias)
    add(p & "ln1.gamma", blk.ln1.gamma)
    add(p & "ln1.beta", blk.ln1.beta)
    add(p & "ln2.gamma", blk.ln2.gamma)
    add(p & "ln2.beta", blk.ln2.beta)
  if m.hrmEnabled:
    let h = m.hrm
    let layout = [float32(h.lLayers), float32(h.cycles), float32(h.lSteps), float32(h.segments)]
    result.add((HrmLayoutKey, quantizeTensor(layout, @[layout.len], qkFp32Raw)))
    add("hrm.normL.gamma", m.hrmNormL.gamma)
    add("hrm.normL.beta", m.hrmNormL.beta)
    add("hrm.normH.gamma", m.hrmNormH.gamma)
    add("hrm.normH.beta", m.hrmNormH.beta)
    add("hrm.normX.gamma", m.hrmNormX.gamma)
    add("hrm.normX.beta", m.hrmNormX.beta)
  if m.mixture.experts.len > 0:
    let layout = [float32(m.mixture.experts.len), float32(m.mixture.width), float32(m.mixture.depth)]
    result.add((MoeLayoutKey, quantizeTensor(layout, @[layout.len], qkFp32Raw)))
    add("moe.norm.gamma", m.mixture.norm.gamma)
    add("moe.norm.beta", m.mixture.norm.beta)
    add("moe.router.weight", m.mixture.router.weight)
    add("moe.router.bias", m.mixture.router.bias)
    for i, expert in m.mixture.experts:
      for j in 0 ..< expert.layers.len:
        let key = expertLayerKey(i, j)
        template addLayer(name: string, value: Tensor) = add(key & name, value)
        visitExpertLayer(expert.layers[j], addLayer)

proc stateDictToModel(sd: seq[(string, QuantTensor)],
                       vocab, embedDim, nHeads, nLayers, ffMult: int): NimformerModelSpectron =
  result = newNimformerModelSpectron(vocab, embedDim, nHeads, nLayers, ffMult, useSession = true)
  var lookup = initTable[string, QuantTensor]()
  for (name, qt) in sd: lookup[name] = qt
  template restore(name: string, t: var Tensor) =
    if name in lookup:
      t.data = dequantizeTensor(lookup[name])
    else:
      stderr.writeLine("[WARNING] Checkpoint is missing tensor '" & name & "' - keeping its random initial value")
  restore("embed.weight", result.embed.weight)
  restore("outProj.weight", result.outProj.weight)
  restore("outProj.bias", result.outProj.bias)
  # Old checkpoints have no final norm. Preserve their inference exactly;
  # the training entry point explicitly enables it when resuming those files.
  let hasNormG = "finalNorm.gamma" in lookup
  let hasNormB = "finalNorm.beta" in lookup
  if hasNormG != hasNormB:
    raise newException(ValueError, "Checkpoint is missing a finalNorm tensor")
  result.useFinalNorm = hasNormG
  if result.useFinalNorm:
    restore("finalNorm.gamma", result.finalNorm.gamma)
    restore("finalNorm.beta", result.finalNorm.beta)
  for i in 0 ..< result.blocks.len:
    let p = &"block{i}."
    restore(p & "attn.qkvProj.weight", result.blocks[i].attn.qkvProj.weight)
    restore(p & "attn.qkvProj.bias", result.blocks[i].attn.qkvProj.bias)
    restore(p & "attn.outProj.weight", result.blocks[i].attn.outProj.weight)
    restore(p & "attn.outProj.bias", result.blocks[i].attn.outProj.bias)
    restore(p & "mamba.inProj.weight", result.blocks[i].mamba.inProj.weight)
    restore(p & "mamba.inProj.bias", result.blocks[i].mamba.inProj.bias)
    restore(p & "mamba.outProj.weight", result.blocks[i].mamba.outProj.weight)
    restore(p & "mamba.outProj.bias", result.blocks[i].mamba.outProj.bias)
    restore(p & "fuseGate.weight", result.blocks[i].fuseGate.weight)
    restore(p & "fuseGate.bias", result.blocks[i].fuseGate.bias)
    restore(p & "ff.fc1.weight", result.blocks[i].ff.fc1.weight)
    restore(p & "ff.fc1.bias", result.blocks[i].ff.fc1.bias)
    restore(p & "ff.fc2.weight", result.blocks[i].ff.fc2.weight)
    restore(p & "ff.fc2.bias", result.blocks[i].ff.fc2.bias)
    restore(p & "ln1.gamma", result.blocks[i].ln1.gamma)
    restore(p & "ln1.beta", result.blocks[i].ln1.beta)
    restore(p & "ln2.gamma", result.blocks[i].ln2.gamma)
    restore(p & "ln2.beta", result.blocks[i].ln2.beta)
  if HrmLayoutKey in lookup:
    let layout = dequantizeTensor(lookup[HrmLayoutKey])
    if layout.len != 4:
      raise newException(ValueError, "Invalid Spectron HRM layout")
    for value in layout:
      if not finite32(value) or value < 1 or value > 65_536'f32 or floor(value) != value:
        raise newException(ValueError, "Invalid Spectron HRM setting")
    result.hrm = HrmConfig(lLayers: int(layout[0]), cycles: int(layout[1]),
                           lSteps: int(layout[2]), segments: int(layout[3]))
    result.validateHrm()
    restore("hrm.normL.gamma", result.hrmNormL.gamma)
    restore("hrm.normL.beta", result.hrmNormL.beta)
    restore("hrm.normH.gamma", result.hrmNormH.gamma)
    restore("hrm.normH.beta", result.hrmNormH.beta)
    restore("hrm.normX.gamma", result.hrmNormX.gamma)
    restore("hrm.normX.beta", result.hrmNormX.beta)
  if MoeLayoutKey in lookup:
    let layout = dequantizeTensor(lookup[MoeLayoutKey])
    if layout.len != 3:
      raise newException(ValueError, "Invalid Spectron MoE layout")
    for size in layout:
      if not finite32(size) or size < 1 or size > 16_777_216'f32 or floor(size) != size:
        raise newException(ValueError, "Invalid Spectron MoE dimension")
    let (count, width, depth) = (int(layout[0]), int(layout[1]), int(layout[2]))
    if int64(count) * depth > sd.len div 6:
      raise newException(ValueError, "Missing Spectron expert tensors")
    result.mixture = newMixture(embedDim, count, width, depth, initializeWeights = false)
    proc restoreMoe(name: string, value: var Tensor) =
      if name notin lookup or lookup[name].shape != value.shape:
        raise newException(ValueError, "Missing or incorrectly shaped Spectron tensor " & name)
      let data = dequantizeTensor(lookup[name])
      let size = if value.shape.len == 1: value.shape[0] else: value.shape[0] * value.shape[1]
      if data.len != size:
        raise newException(ValueError, "Invalid Spectron tensor length for " & name)
      value.data = data
    restoreMoe("moe.norm.gamma", result.mixture.norm.gamma)
    restoreMoe("moe.norm.beta", result.mixture.norm.beta)
    restoreMoe("moe.router.weight", result.mixture.router.weight)
    restoreMoe("moe.router.bias", result.mixture.router.bias)
    for i, expert in result.mixture.experts:
      for j in 0 ..< expert.layers.len:
        let key = expertLayerKey(i, j)
        template restoreLayer(name: string, value: var Tensor) = restoreMoe(key & name, value)
        visitExpertLayer(expert.layers[j], restoreLayer)

const MetaStepKey = "__meta_step__"
  ## A "fake" tensor name (shape [1], kind qkFp32Raw) used to hide the trained step
  ## count in the state dict - NO need to change the NIMQ2 binary format, old
  ## files (without this key) still load normally (step defaults to 0).

proc checkpointArch(m: NimformerModelSpectron): array[5, int] =
  let ffMult = m.blocks[0].ff.fc1.outF div m.embed.dim
  [m.vocab, m.embed.dim, m.blocks[0].attn.nHeads, m.blocks.len, ffMult]

proc saveCheckpoint*(m: NimformerModelSpectron, path: string, step: int = 0) =
  ## arch = [vocab, embedDim, nHeads, nLayers, ffMult] - exactly the array[5,int]
  ## saveQuantStateDict expects (no separate format needed).
  ## `step` = the number of training steps completed UP TO this checkpoint, saved
  ## alongside so resume uses the right step instead of always going back to step 1.
  var sd = modelToStateDict(m)
  sd.add((MetaStepKey, quantizeTensor(@[float32(step)], @[1], qkFp32Raw)))
  saveQuantStateDict(path, m.checkpointArch(), sd)

proc loadCheckpoint*(path: string): tuple[model: NimformerModelSpectron, step: int] =
  let (arch, sd, _) = loadQuantStateDict(path, mapped = true)
  var step = 0
  var sdModel: seq[(string, QuantTensor)] = @[]
  for (name, qt) in sd:
    if name == MetaStepKey:
      let v = dequantizeTensor(qt)
      if v.len > 0: step = int(v[0])
    elif not name.startsWith("__packed."):
      sdModel.add((name, qt))
  result.model = stateDictToModel(sdModel, arch[0], arch[1], arch[2], arch[3], arch[4])
  result.step = step

proc checkAndClipGradients(grads: var SpectronGrads, limit: float32): float64 =
  template eachGradient(body: untyped) =
    for gradient {.inject.} in [addr grads.dEmbed, addr grads.dOutProjW, addr grads.dOutProjB,
                               addr grads.dFinalNormG, addr grads.dFinalNormB,
                               addr grads.dHrmNormLG, addr grads.dHrmNormLB,
                               addr grads.dHrmNormHG, addr grads.dHrmNormHB,
                               addr grads.dHrmNormXG, addr grads.dHrmNormXB,
                               addr grads.mixture.routerW, addr grads.mixture.routerB,
                               addr grads.mixture.normG, addr grads.mixture.normB]:
      body
    for blockGrad in grads.dBlocks.mitems:
      for value in fields(blockGrad):
        let gradient {.inject.} = addr value
        body
    for layers in grads.mixture.expertLayers.mitems:
      for layer in layers.mitems:
        for value in fields(layer):
          let gradient {.inject.} = addr value
          body

  var squaredNorm = 0.0
  eachGradient:
    for value in gradient[].data:
      squaredNorm += float64(value) * float64(value)
  if squaredNorm.classify in {fcNan, fcInf, fcNegInf}:
    raise newException(SpectronNumericalError, "Non-finite Spectron gradient norm")
  result = sqrt(squaredNorm)
  if limit > 0 and result > float64(limit):
    let scale = float64(limit) / result
    eachGradient:
      for value in gradient[].data.mitems:
        value = float32(float64(value) * scale)

proc applyGradients(m: var NimformerModelSpectron, adam: var ModelAdamState,
                    grads: SpectronGrads, ctx: Backend, lr, beta2: float32, requantizeEvery: int) =
  # GPU: Adam is 1 micro-kernel / parameter (adamStepFusedGpu). Only the periodic APF
  # requantize step (every `requantizeEvery` steps) still runs on the host - customfloat has no
  # GPU kernel yet. Non-GPU backends: the original apfAdamStep.
  let gpu = isGpuCtx(ctx)
  proc apfAdamStep(param: var Tensor, grad: Tensor, state: var ApfAdamState) =
    if gpu:
      inc state.step
      adamStepFusedGpu(ctx, param.data, grad.data, state.m, state.v, state.step, lr, b2 = beta2)
    else:
      discard nimformer.apfAdamStep(param, grad, state, lr, b2 = beta2, requantizeEvery = int.high)
    for i in 0 ..< param.data.len:
      if not finite32(param.data[i]) or not finite32(state.m[i]) or
          not finite32(state.v[i]) or state.v[i] < 0:
        raise newException(SpectronNumericalError, "Non-finite weight or invalid Adam moment")
    if requantizeEvery > 0 and state.step mod requantizeEvery == 0:
      let cf = buildCustomDtypeForTensor(param.data, grad.data)
      let quantized = decodeArray(encodeArray(param.data, cf), cf)
      for value in quantized:
        if not finite32(value):
          raise newException(SpectronNumericalError, "Non-finite APF weight")
      param.data = quantized

  apfAdamStep(m.outProj.weight, grads.dOutProjW, adam.outProjW)
  apfAdamStep(m.outProj.bias, grads.dOutProjB, adam.outProjB)
  apfAdamStep(m.embed.weight, grads.dEmbed, adam.embed)
  if m.useFinalNorm:
    apfAdamStep(m.finalNorm.gamma, grads.dFinalNormG, adam.finalNormG)
    apfAdamStep(m.finalNorm.beta, grads.dFinalNormB, adam.finalNormB)
  if m.hrmEnabled:
    apfAdamStep(m.hrmNormL.gamma, grads.dHrmNormLG, adam.hrmNormLG)
    apfAdamStep(m.hrmNormL.beta, grads.dHrmNormLB, adam.hrmNormLB)
    apfAdamStep(m.hrmNormH.gamma, grads.dHrmNormHG, adam.hrmNormHG)
    apfAdamStep(m.hrmNormH.beta, grads.dHrmNormHB, adam.hrmNormHB)
    apfAdamStep(m.hrmNormX.gamma, grads.dHrmNormXG, adam.hrmNormXG)
    apfAdamStep(m.hrmNormX.beta, grads.dHrmNormXB, adam.hrmNormXB)
  if m.mixture.experts.len > 0:
    apfAdamStep(m.mixture.norm.gamma, grads.mixture.normG, adam.mixture.normG)
    apfAdamStep(m.mixture.norm.beta, grads.mixture.normB, adam.mixture.normB)
    apfAdamStep(m.mixture.router.weight, grads.mixture.routerW, adam.mixture.routerW)
    apfAdamStep(m.mixture.router.bias, grads.mixture.routerB, adam.mixture.routerB)
  for i in 0 ..< m.mixture.experts.len:
    if grads.mixture.expertLayers[i].len == 0: continue
    # The candidate shares inactive experts with the previous training state.
    # Stage only the experts participating in this step.
    let current = m.mixture.experts[i]
    var next: SpectronExpert
    new(next)
    next[] = current[]
    var moments: ExpertAdam
    if adam.mixture.experts[i] == nil:
      moments = newExpertAdam(current)
    else:
      new(moments)
      moments[] = adam.mixture.experts[i][]
    for j in 0 ..< next.layers.len:
      template layer: untyped = next.layers[j]
      template g: untyped = grads.mixture.expertLayers[i][j]
      template a: untyped = moments.layers[j]
      apfAdamStep(layer.ff.fc1.weight, g.upW, a.upW)
      apfAdamStep(layer.ff.fc1.bias, g.upB, a.upB)
      apfAdamStep(layer.ff.fc2.weight, g.downW, a.downW)
      apfAdamStep(layer.ff.fc2.bias, g.downB, a.downB)
      apfAdamStep(layer.norm.gamma, g.normG, a.normG)
      apfAdamStep(layer.norm.beta, g.normB, a.normB)
    m.mixture.experts[i] = next
    adam.mixture.experts[i] = moments

  for i in 0 ..< m.blocks.len:
    template g: untyped = grads.dBlocks[i]
    template a: untyped = adam.blocks[i]
    apfAdamStep(m.blocks[i].attn.qkvProj.weight, g.dAttnQkvW, a.attnQkvW)
    apfAdamStep(m.blocks[i].attn.qkvProj.bias, g.dAttnQkvB, a.attnQkvB)
    apfAdamStep(m.blocks[i].attn.outProj.weight, g.dAttnOutW, a.attnOutW)
    apfAdamStep(m.blocks[i].attn.outProj.bias, g.dAttnOutB, a.attnOutB)
    apfAdamStep(m.blocks[i].mamba.inProj.weight, g.dMambaInW, a.mambaInW)
    apfAdamStep(m.blocks[i].mamba.inProj.bias, g.dMambaInB, a.mambaInB)
    apfAdamStep(m.blocks[i].mamba.outProj.weight, g.dMambaOutW, a.mambaOutW)
    apfAdamStep(m.blocks[i].mamba.outProj.bias, g.dMambaOutB, a.mambaOutB)
    apfAdamStep(m.blocks[i].fuseGate.weight, g.dFuseW, a.fuseW)
    apfAdamStep(m.blocks[i].fuseGate.bias, g.dFuseB, a.fuseB)
    apfAdamStep(m.blocks[i].ff.fc1.weight, g.dFf1W, a.ff1W)
    apfAdamStep(m.blocks[i].ff.fc1.bias, g.dFf1B, a.ff1B)
    apfAdamStep(m.blocks[i].ff.fc2.weight, g.dFf2W, a.ff2W)
    apfAdamStep(m.blocks[i].ff.fc2.bias, g.dFf2B, a.ff2B)
    apfAdamStep(m.blocks[i].ln1.gamma, g.dLn1G, a.ln1G)
    apfAdamStep(m.blocks[i].ln1.beta, g.dLn1B, a.ln1B)
    apfAdamStep(m.blocks[i].ln2.gamma, g.dLn2G, a.ln2G)
    apfAdamStep(m.blocks[i].ln2.beta, g.dLn2B, a.ln2B)

type SpectronTrainingStep = object
  loss: float32
  gradNorm, balanceLoss: float64
  routes: seq[int]

proc addInto(total: var Tensor, value: Tensor) =
  if total.data.len == 0:
    total = value
  elif value.data.len > 0:
    if total.data.len != value.data.len:
      raise newException(ValueError, "Mismatched gradient shapes across HRM segments")
    for i in 0 ..< total.data.len: total.data[i] += value.data[i]

proc addTupleInto[T: tuple](total: var T, value: T) =
  for totalName, totalField in fieldPairs(total):
    for valueName, valueField in fieldPairs(value):
      when totalName == valueName and totalField is Tensor:
        addInto(totalField, valueField)

proc accumulate(total: var SpectronGrads, grads: SpectronGrads) =
  ## Sum the gradients of one HRM segment into the step total.
  addInto(total.dOutProjW, grads.dOutProjW)
  addInto(total.dOutProjB, grads.dOutProjB)
  addInto(total.dFinalNormG, grads.dFinalNormG)
  addInto(total.dFinalNormB, grads.dFinalNormB)
  addInto(total.dEmbed, grads.dEmbed)
  addInto(total.dHrmNormLG, grads.dHrmNormLG)
  addInto(total.dHrmNormLB, grads.dHrmNormLB)
  addInto(total.dHrmNormHG, grads.dHrmNormHG)
  addInto(total.dHrmNormHB, grads.dHrmNormHB)
  addInto(total.dHrmNormXG, grads.dHrmNormXG)
  addInto(total.dHrmNormXB, grads.dHrmNormXB)
  if total.dBlocks.len == 0:
    total.dBlocks = grads.dBlocks
  else:
    for i in 0 ..< total.dBlocks.len: addTupleInto(total.dBlocks[i], grads.dBlocks[i])
  addInto(total.mixture.routerW, grads.mixture.routerW)
  addInto(total.mixture.routerB, grads.mixture.routerB)
  addInto(total.mixture.normG, grads.mixture.normG)
  addInto(total.mixture.normB, grads.mixture.normB)
  total.mixture.balanceLoss += grads.mixture.balanceLoss
  if total.mixture.routes.len == 0:
    total.mixture.routes = grads.mixture.routes
  else:
    for i, count in grads.mixture.routes: total.mixture.routes[i] += count
  if total.mixture.expertLayers.len == 0:
    total.mixture.expertLayers = grads.mixture.expertLayers
  else:
    for i, layers in grads.mixture.expertLayers:
      if layers.len == 0: continue
      if total.mixture.expertLayers[i].len == 0:
        total.mixture.expertLayers[i] = layers
      else:
        for j in 0 ..< layers.len: addTupleInto(total.mixture.expertLayers[i][j], layers[j])

proc checkedTrainStep(m: var NimformerModelSpectron, adam: var ModelAdamState,
                      idsBatch, targetsBatch: seq[seq[int]], ctx: Backend,
                      lr, beta2, maxGradNorm: float32, requantizeEvery: int): SpectronTrainingStep =
  # HRM deep supervision: every segment is supervised from the previous
  # segment's detached state. Segment gradients are averaged into one update
  # so the optimizer step count stays equal to the training step.
  let segments = if m.hrmEnabled: m.hrm.segments else: 1
  var carry: HrmCarry
  var grads: SpectronGrads
  var lossSum = 0.0
  for _ in 0 ..< segments:
    let cache = m.forwardForBackward(idsBatch, ctx, carry)
    var ce = checkedCrossEntropy(m.outProj.forward(cache.headInput, ctx), targetsBatch, ctx)
    if ce.nValid == 0: return
    if segments > 1:
      let scale = 1'f32 / float32(segments)
      for value in ce.dLogits.data.mitems: value *= scale
    grads.accumulate(m.backwardCached(idsBatch, ce.dLogits, cache, ctx))
    lossSum += float64(ce.loss)
  grads.mixture.balanceLoss /= float64(segments)
  result.loss = float32(lossSum / float64(segments))
  result.gradNorm = grads.checkAndClipGradients(maxGradNorm)
  result.balanceLoss = grads.mixture.balanceLoss
  result.routes = move(grads.mixture.routes)

  # Keep the current state live until the proposed update passes the numeric checks.
  var nextModel = m
  var nextAdam = adam
  nextModel.applyGradients(nextAdam, grads, ctx, lr, beta2, requantizeEvery)
  m = move(nextModel)
  adam = move(nextAdam)

proc trainStep*(m: var NimformerModelSpectron, adam: var ModelAdamState,
                idsBatch, targetsBatch: seq[seq[int]], ctx: Backend,
                lr: float32 = 3e-4'f32, requantizeEvery: int = 50,
                maxGradNorm: float32 = 0): float32 =
  ## Update only from finite loss and gradients. A failed Adam or APF update
  ## leaves the model and optimizer at the previous step.
  checkedTrainStep(m, adam, idsBatch, targetsBatch, ctx, lr, 0.999'f32,
                   maxGradNorm, requantizeEvery).loss

# ═══════════════════════════════════════════════════════════════
# INFERENCE: greedy / temperature sampling, autoregressive
# ═══════════════════════════════════════════════════════════════
proc sampleFromLogits(logits: seq[float32], temperature: float32, rng: var Rand): int =
  if temperature <= 0'f32:
    # greedy
    var best = 0
    var bestV = logits[0]
    for i in 1 ..< logits.len:
      if logits[i] > bestV:
        bestV = logits[i]; best = i
    return best
  var probs = newSeq[float32](logits.len)
  var maxL = logits[0]
  for v in logits:
    if v > maxL: maxL = v
  var sum = 0'f32
  for i in 0 ..< logits.len:
    probs[i] = exp((logits[i] - maxL) / temperature)
    sum += probs[i]
  let r = float32(rng.rand(1.0)) * sum
  var acc = 0'f32
  for i in 0 ..< probs.len:
    acc += probs[i]
    if acc >= r: return i
  return probs.len - 1

proc generate*(m: NimformerModelSpectron, promptIds: seq[int], maxNewTokens: int,
               ctx: Backend, temperature: float32 = 0'f32, seed: int64 = 1234): seq[int] =
  ## Simple autoregressive text generation (no KV cache: the whole sequence is
  ## re-run each step - enough for the demo/test; for long sequences a KV cache
  ## can be added later without changing the result).
  var rng = initRand(seed)
  result = promptIds
  for _ in 0 ..< maxNewTokens:
    let logits = m.forward(result, ctx)   # [T, vocab]
    let T = logits.shape[0]
    let V = logits.shape[1]
    let lastLogits = logits.data[(T - 1) * V ..< T * V]
    let nextId =
      if isGpuCtx(ctx) and temperature <= 0'f32:
        int(beArgmaxIndexGpu(ctx, lastLogits, 1, V, 1).idx[0])   # greedy on the GPU
      else: sampleFromLogits(lastLogits, temperature, rng)
    result.add(nextId)

# ═══════════════════════════════════════════════════════════════
# DEMO: train on Dolly + Python code JSONL (byte-level tokenizer)
# then generate a few tokens - runs with `make run` from the repo root.
# ═══════════════════════════════════════════════════════════════
when isMainModule:
  include packed_training
  if paramCount() > 0:
    try:
      trainPacked()
    except CatchableError as error:
      quit("[packed] " & error.msg, 1)
    quit(0)

  randomize(1234)
  createDir("test/spectron/models")
  let ctx = newBackend()

  # ---- Load the raw text corpus from files in the repo, fallback when missing ----
  proc loadCorpusText(path: string): string =
    if fileExists(path):
      result = readFile(path)
    else:
      stderr.writeLine("[corpus] Not found: " & path)

  proc loadCodeCorpusText(path: string, maxExamples: int): string =
    if not fileExists(path):
      stderr.writeLine("[corpus] Not found: " & path)
      return
    var examples = newSeqOfCap[string](maxExamples)
    var seen = 0
    for line in lines(path):
      try:
        let row = parseJson(line)
        if row.kind != JObject: continue
        let topic =
          if row.hasKey("topic") and row["topic"].kind == JString: row["topic"].getStr()
          elif row.hasKey("context") and row["context"].kind == JString: row["context"].getStr()
          else: ""
        let language = if row.hasKey("language") and row["language"].kind == JString: row["language"].getStr() else: ""
        let code = if row.hasKey("code") and row["code"].kind == JString: row["code"].getStr() else: ""
        let note =
          if row.hasKey("note") and row["note"].kind == JString: row["note"].getStr()
          elif row.hasKey("explanation") and row["explanation"].kind == JString: row["explanation"].getStr()
          else: ""
        if topic.len == 0 or language.len == 0 or code.len == 0: continue
        let prompt = "Write " & language & " code for: " & topic & "\n" &
          (if note.len > 0: note & "\n" else: "") & code
        inc seen
        if examples.len < maxExamples:
          examples.add(prompt)
        else:
          let slot = rand(seen - 1)
          if slot < maxExamples: examples[slot] = prompt
      except JsonParsingError:
        continue
    for example in examples:
      if result.len > 0: result.add("\n\n")
      result.add(example)
    echo &"[corpus] Reservoir {examples.len} examples from {seen} valid rows: {path}"

  var corpusParts = @[loadCorpusText(CFG_CORPUS_PATH)]
  for path in CFG_CODE_CORPUS_PATHS:
    corpusParts.add(loadCodeCorpusText(path, CFG_CODE_EXAMPLES_PER_FILE))
  var rawCorpus = ""
  var bpeTrainSlice = ""
  let perSourceBpeChars = max(1, CFG_BPE_TRAIN_CHARS div corpusParts.len)
  for part in corpusParts:
    if part.len == 0: continue
    if rawCorpus.len > 0: rawCorpus.add('\n')
    rawCorpus.add(part)
    # Take a slice from each source so the tokenizer learns both dialogue and Python,
    # without the first Dolly file eating the whole BPE training budget.
    bpeTrainSlice.add(part[0 ..< min(part.len, perSourceBpeChars)])
    bpeTrainSlice.add('\n')
  if rawCorpus.len == 0:
    rawCorpus = "Nimformer Spectron demo corpus. ".repeat(200)
    bpeTrainSlice = rawCorpus[0 ..< min(rawCorpus.len, CFG_BPE_TRAIN_CHARS)]
  echo "== Training the BPE tokenizer (or reloading it if present) =="
  let tok = getOrTrainBPE(bpeTrainSlice, CFG_NUM_MERGES)
  let vocab = tok.vocabSize
  echo &"Tokenizer: vocab_size = {vocab} (256 base bytes + {tok.merges.len} merges)"

  var model: NimformerModelSpectron
  var startStep = 0   # steps trained up to the loaded checkpoint (0 when training from scratch)
  if fileExists(CFG_CHECKPOINT_PATH):
    echo &"== Found checkpoint {CFG_CHECKPOINT_PATH} -> reloading to resume training =="
    (model, startStep) = loadCheckpoint(CFG_CHECKPOINT_PATH)
    echo &"  -> resuming from step {startStep + 1} (checkpoint trained {startStep} steps)"
    if not model.useFinalNorm:
      model.useFinalNorm = true
      echo "  -> enabling the final LayerNorm before the lm head to stabilise the old checkpoint's logits"
  else:
    model = newNimformerModelSpectron(vocab, CFG_EMBED_DIM, CFG_N_HEADS, CFG_N_LAYERS,
                                        CFG_FF_MULT, useSession = true)
  var adam = newModelAdamState(model)
  if startStep > 0:
    echo "  [resume] Adam moments were not saved; the optimizer restarts from step 0"

  # Encode the whole corpus once with the trained BPE, shared by sampling.
  var corpusIds = encodeBPE(tok, rawCorpus)
  var rawTokenMemory = newRawTokenMemory(tok, move(corpusIds))
  doAssert rawTokenMemory.tokenCount > CFG_SEQ_LEN + 1,
    "The corpus is too short for the training demo"
  echo &"[raw-token-memory] indexed {rawTokenMemory.tokenCount} raw BPE tokens; " &
       "Linear++ and Mamba2 scan batches will retain their corpus windows"

  proc lrAt(step, startStep: int): float32 =
    ## Warmup + cosine schedule. The decay follows the global step, so a resumed run continues
    ## where the checkpoint stopped instead of jumping back to the peak. Adam moments are not
    ## saved, so every run (fresh or resumed) warms up again from its own first step - a full
    ## LR step on zeroed moments is what makes the loss spike right after a resume.
    ## A run resumed past CFG_LR_DECAY_STEPS would otherwise sit at CFG_MIN_LR forever, so it
    ## restarts a cosine over its own CFG_N_STEPS, from CFG_RESUME_LR down to CFG_RESUME_MIN_LR.
    var peak = float64(CFG_LR)
    var lrFloor = float64(CFG_MIN_LR)
    var progress = clamp(float64(step) / float64(max(1, CFG_LR_DECAY_STEPS)), 0.0, 1.0)
    if startStep >= CFG_LR_DECAY_STEPS:
      peak = float64(CFG_RESUME_LR)
      lrFloor = float64(CFG_RESUME_MIN_LR)
      progress = clamp(float64(step - startStep) / float64(max(1, CFG_N_STEPS)), 0.0, 1.0)
    let cosine = 0.5 * (1.0 + cos(PI * progress))
    let decayed = lrFloor + (peak - lrFloor) * cosine
    let local = step - startStep
    let warm = if CFG_WARMUP_STEPS > 0: min(1.0, float64(local) / float64(CFG_WARMUP_STEPS)) else: 1.0
    float32(decayed * warm)

  # Seeded by the resume point: a fixed seed made every resumed run train on the same batches.
  var rng = initRand(42 + startStep)
  echo &"== Starting demo training: {CFG_N_STEPS} steps (from step {startStep + 1}), batch={CFG_BATCH_SIZE}, seq={CFG_SEQ_LEN} =="
  # The loop runs the NEXT CFG_N_STEPS steps after startStep (not
  # always back to step 1) - resume uses the step count saved in the checkpoint.
  var lossSum = 0.0
  var lossCount = 0
  for step in (startStep + 1) .. (startStep + CFG_N_STEPS):
    let tokenBatch = rawTokenMemory.sampleRawTokenBatch(
      CFG_BATCH_SIZE, CFG_SEQ_LEN, step, rng)
    let lr = lrAt(step, startStep)
    let loss = trainStep(model, adam, tokenBatch.ids, tokenBatch.targets, ctx, lr = lr,
                          requantizeEvery = CFG_REQUANTIZE_EVERY)
    lossSum += float64(loss)
    inc lossCount
    if step mod CFG_LOG_EVERY == 0 or step == startStep + 1 or step == startStep + CFG_N_STEPS:
      let meanLoss = lossSum / float64(lossCount)
      echo &"step {step:>5} | lr = {lr:.2e} | loss = {loss:.4f} | mean_loss({lossCount}) = {meanLoss:.4f}"
      lossSum = 0.0
      lossCount = 0

    if step mod CFG_CHECKPOINT_EVERY == 0:
      saveCheckpoint(model, CFG_CHECKPOINT_PATH, step)
      if step mod CFG_LOG_EVERY == 0 or step == startStep + CFG_CHECKPOINT_EVERY:
        echo &"  [checkpoint] saved {CFG_CHECKPOINT_PATH} (step {step})"

  saveCheckpoint(model, CFG_CHECKPOINT_PATH, startStep + CFG_N_STEPS)
  echo &"== Saved the final checkpoint: {CFG_CHECKPOINT_PATH} =="
  echo "[raw-token-memory] captured ", rawTokenMemory.windows.len,
       " training windows (", rawTokenMemory.windows.len * CFG_SEQ_LEN,
       " Linear++/scan input-token uses)"

  echo "== Test generation (greedy) from a new sentence =="
  let promptIds = encodeBPE(tok, CFG_GEN_PROMPT)
  let outIds = model.generate(promptIds, maxNewTokens = CFG_GEN_MAX_NEW_TOKENS, ctx = ctx,
                               temperature = CFG_GEN_TEMPERATURE)
  echo "Output: ", decodeBPEStr(tok, outIds)

  closeBackend(ctx)
