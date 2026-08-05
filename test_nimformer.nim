## test_nimformer.nim
## Bản đầy đủ: LinearPlusAttention (Linear++) + Mamba-2 SSD + khối Spectron
## (gated fusion 2 nhánh) + NimformerModelSpectron (embed -> N blocks -> lm
## head) với forward/backward ĐẦY ĐỦ cho mọi thành phần, cộng vòng lặp
## train() (Adam) và generate() (greedy/temperature) để chạy training và
## inference thật, không còn phần "để sau".
##
## Quy ước chung (giống nimformer.nim): layout activation luôn [B,T,C] hoặc
## coi [B*T, C] khi qua Linear/LayerNorm. Mọi phép toán nặng (matmul, add,
## sigmoid...) đi qua backend.nim (beMatmul/beAdd/beSigmoid...) để chạy đúng
## trên backend đã chọn (cpu/metal/cuda/opencl/tsic), KHÔNG lặng lẽ rơi về
## vòng lặp Nim thuần cho những phép toán lớn.

import std/[math, random, strformat, os, strutils, json, tables]
import customfloat
import backend
import quant
import nimformer  # Tensor, Linear, LayerNorm, FeedForward, Embedding, addT,
                   # transpose, flatten2D, newTensor, randnTensor,
                   # ApfAdamState, apfAdamStep, newLinear, newLayerNorm,
                   # newFeedForward, newEmbedding

# ═══════════════════════════════════════════════════════════════
# CONFIG - SỬA Ở ĐÂY CHO DỄ, KHÔNG PHẢI LỤC XUỐNG main BÊN DƯỚI
# ═══════════════════════════════════════════════════════════════
const
  CFG_EMBED_DIM* = 64
  CFG_N_HEADS* = 4
  CFG_N_LAYERS* = 2
  CFG_FF_MULT* = 4
  CFG_SEQ_LEN* = 128
  CFG_BATCH_SIZE* = 32
  CFG_NUM_MERGES* = 500        # vocab cuoi = 256 + merges (dung som neu het cap lap)
  CFG_N_STEPS* = 6_000        # so buoc train demo
  CFG_LR* = 3e-4'f32
  CFG_REQUANTIZE_EVERY* = 50   # xem giai thich trong trainStep ben duoi
  CFG_LOG_EVERY* = 100         # log loss moi bao nhieu buoc
  CFG_BPE_TRAIN_CHARS* = 300_000  # so ky tu dau corpus dung de train BPE
  CFG_CORPUS_PATH* = "databricks-dolly-15k.jsonl"
  CFG_GEN_MAX_NEW_TOKENS* = 40
  CFG_GEN_TEMPERATURE* = 0'f32
  CFG_GEN_PROMPT* = "When did Virgin Australia start operating?"
  CFG_CHECKPOINT_PATH* = "finetune.nimq"
  CFG_CHECKPOINT_EVERY* = 5    # luu checkpoint moi bao nhieu buoc train

# ═══════════════════════════════════════════════════════════════
# CANH BAO FALLBACK CPU - in ra 1 LAN DUY NHAT khi Mamba2SSD phai roi ve
# CPU thay vi chay session GPU (tranh spam log qua 10.000 buoc train).
# ═══════════════════════════════════════════════════════════════
var
  warnedNoSession = false
  warnedSessionBeginFail = false
  warnedSessionMidFail = false

proc warnFallbackOnce(flag: var bool, reason: string) =
  if not flag:
    stderr.writeLine(&"[CANH BAO] Mamba2SSD roi ve CPU fallback: {reason} (chi in 1 lan, cac buoc sau van fallback tuong tu neu cung ly do)")
    flag = true

# ═══════════════════════════════════════════════════════════════
# BPE TOKENIZER RIÊNG CHO SPECTRON (byte-level BPE, tự train trên corpus
# của chính project này - KHÔNG dùng lại tokenizer.json của model 32256-vocab
# trong nim_inference.nim, đó là 2 model khác nhau hoàn toàn).
#
# Thuật toán chuẩn GPT-2 style byte-level BPE:
#   - 256 token cơ sở = 256 giá trị byte (0..255), luôn biểu diễn được MỌI
#     chuỗi byte bất kỳ (không bao giờ có "unknown token").
#   - Lặp lại numMerges lần: đếm tần suất mọi cặp token liền kề trong toàn
#     corpus, chọn cặp xuất hiện nhiều nhất, gộp thành 1 token mới
#     id = 256 + bước, ghi lại rule (a,b)->newId theo ĐÚNG thứ tự tạo ra.
#   - Encode văn bản mới: bắt đầu từ byte thô, áp lần lượt các merge rule
#     theo đúng thứ tự đã train (đây là điểm mấu chốt để BPE decode được
#     xác định/deterministic).
#   - Decode: mở ngược newId -> (a,b) -> ... -> byte thô.
# ═══════════════════════════════════════════════════════════════
type
  BPETokenizer* = object
    ## Tokenizer subword kiểu GPT-2/Llama THẬT (byte-level BPE + pre-tokenizer
    ## theo từ/số/dấu câu), KHÔNG phải "1 byte = 1 token" nữa:
    ##   1) byte-to-unicode: map 256 giá trị byte sang 256 ký tự Unicode LUÔN
    ##      in được (đúng thuật toán bytes_to_unicode() của GPT-2 gốc - xem
    ##      newGpt2ByteEncoder bên dưới) -> vocab/merges là CHUỖI, đọc được,
    ##      xuất ra tokenizer.json giống hệt format thật của HF/tokenizers.
    ##   2) pre-tokenize: cắt câu thành các mẩu (chữ dính khoảng trắng đầu,
    ##      số, dấu câu, khoảng trắng) TRƯỚC khi BPE - giống regex GPT-2
    ##      (\p{L}+, \p{N}+, dấu câu, ...) - để token không bao giờ "nhảy"
    ##      qua ranh giới từ/số/dấu câu như BPE-thô-trên-toàn-file.
    ##   3) BPE thật sự chỉ merge bên TRONG từng mẩu pre-token.
    vocab*: Table[string, int]        # subword string -> id
    id2tok*: seq[string]              # id -> subword string
    merges*: seq[(string, string)]    # theo đúng thứ tự train (rank = index)
    mergeRank*: Table[(string, string), int]
    byteEncoder*: array[256, string]  # byte thô -> ký tự unicode (bytes_to_unicode)
    byteDecoder*: Table[string, int]  # ký tự unicode -> byte thô (chiều ngược)

proc newGpt2ByteEncoder(): array[256, string] =
  ## Y hệt bytes_to_unicode() trong encoder.py gốc của OpenAI GPT-2: mọi
  ## byte "in được" (33..126, 161..172, 174..255) giữ nguyên chính nó, các
  ## byte còn lại (control char, whitespace, ...) được dồn ra dải unicode
  ## 256.. để KHÔNG BAO GIỜ có ký tự vô hình/mơ hồ trong vocab.
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
      # cp luôn < 0x800 (n tối đa ~68) nên chỉ cần UTF-8 2 byte.
      result = newString(2)
      result[0] = char(0xC0 or (cp shr 6))
      result[1] = char(0x80 or (cp and 0x3F))
  for i in 0 ..< bs.len:
    result[bs[i]] = encodeCodepoint(cs[i])

proc isAsciiLetter(c: char): bool = c in {'a'..'z', 'A'..'Z'}
proc isAsciiDigit(c: char): bool = c in {'0'..'9'}
proc isAsciiSpace(c: char): bool = c in {' ', '\t', '\n', '\r'}

proc gpt2PreTokenize(s: string): seq[string] =
  ## Cắt thô theo tinh thần regex GPT-2
  ## ( 's| 't| 're| 've| 'm| 'll| 'd| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+ )
  ## viết tay bằng scanner ASCII (đủ dùng cho demo, không cần thư viện regex
  ## Unicode - byte>=128 được coi như 1 "chữ cái" đơn lẻ/dấu câu tùy ngữ
  ## cảnh, vẫn encode/decode ĐÚNG nhờ byte-level fallback ở bước sau).
  result = @[]
  var i = 0
  let n = s.len
  const contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
  while i < n:
    var matched = false
    for c in contractions:
      if i + c.len <= n and s[i ..< i + c.len] == c:
        result.add(c)
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
      result.add(s[start ..< i])
    elif i < n and isAsciiDigit(s[i]):
      if not hasLeadingSpace: start = i
      while i < n and isAsciiDigit(s[i]): inc i
      result.add(s[start ..< i])
    elif i < n and isAsciiSpace(s[i]):
      start = i
      while i < n and isAsciiSpace(s[i]):
        if i + 1 < n and s[i] == ' ' and not isAsciiSpace(s[i + 1]) and i > start:
          break   # để lại 1 space làm "leading space" cho mẩu tiếp theo
        inc i
      result.add(s[start ..< i])
    elif i < n:
      if not hasLeadingSpace: start = i
      # 1 byte non-ascii/dấu câu = 1 mẩu (an toàn cho UTF-8 đa byte lẫn dấu câu)
      inc i
      result.add(s[start ..< i])
    else:
      if hasLeadingSpace: result.add(" ")

proc toByteEncodedChars(enc: array[256, string], s: string): seq[string] =
  ## Mỗi BYTE thô của chuỗi (không phải mỗi ký tự Unicode) -> 1 "ký tự cơ
  ## sở" trong bảng byte-encoder - đây chính là "byte-level" của GPT-2/Llama:
  ## luôn biểu diễn được MỌI chuỗi byte bất kỳ, không có unknown-token.
  result = newSeq[string](s.len)
  for i in 0 ..< s.len:
    result[i] = enc[uint8(s[i]).int]

proc bpeMergeWord(tokens: seq[string], mergeRank: Table[(string, string), int]): seq[string] =
  ## BPE chuẩn kiểu HF: lặp lại tìm cặp liền kề có RANK NHỎ NHẤT (tạo sớm
  ## nhất khi train) trong 1 pre-token rồi gộp, tới khi không còn cặp nào
  ## nằm trong bảng merge.
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

  # pre-tokenize 1 lần, giữ nguyên dạng seq[seq[string]] (mỗi pre-token là
  # 1 seq ký tự byte-encoded) để đếm tần suất cặp liền kề CHỈ trong từng
  # pre-token (không bao giờ merge xuyên ranh giới từ/số/dấu câu).
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

  # Dựng vocab: 256 byte cơ sở trước (đúng thứ tự byte 0..255), rồi tới
  # từng merge (256+step) - giống hệt quy ước GPT-2/Llama.
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

proc encodeBPE*(tok: BPETokenizer, s: string): seq[int] =
  result = @[]
  for pt in gpt2PreTokenize(s):
    let baseChars = toByteEncodedChars(tok.byteEncoder, pt)
    let merged = bpeMergeWord(baseChars, tok.mergeRank)
    for piece in merged:
      # piece luôn có trong vocab vì mọi merge đều được đăng ký lúc train;
      # phòng hờ piece lạ (model chưa từng thấy tổ hợp này khi train) thì
      # rơi về từng byte-encoded char gốc (vẫn luôn có trong vocab).
      if piece in tok.vocab:
        result.add(tok.vocab[piece])
      else:
        for ch in toByteEncodedChars(tok.byteEncoder, ""):
          discard ch
        for b in 0 ..< pt.len:
          let bc = tok.byteEncoder[uint8(pt[b]).int]
          result.add(tok.vocab.getOrDefault(bc, 0))

proc decodeBPEStr*(tok: BPETokenizer, ids: seq[int]): string =
  var unicodeStr = ""
  for id in ids:
    if id >= 0 and id < tok.id2tok.len:
      unicodeStr.add(tok.id2tok[id])
  # unicodeStr hiện là chuỗi ký tự byte-encoded (mỗi ký tự = 1-2 byte UTF-8
  # đại diện 1 byte gốc) - giải mã ngược lại byte gốc rồi ghép thành string.
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

proc saveBPE*(tok: BPETokenizer, path: string) =
  ## Lưu ĐÚNG schema kiểu tokenizer.json thật của HF/tokenizers (model BPE):
  ## {"vocab_size", "vocab": {token_string: id}, "merges": ["a b", ...]}.
  ## export_to_nimq.py cũng xuất theo đúng schema này (xem sửa đổi ở đó) để
  ## 2 bên tương thích, load qua lại được.
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

const BPE_TOKENIZER_PATH = "spectron_bpe_tokenizer.json"

proc getOrTrainBPE*(corpusText: string, numMerges: int): BPETokenizer =
  ## Nếu đã train trước đó (file tồn tại) thì nạp lại cho nhanh/nhất quán
  ## giữa các lần chạy; nếu chưa thì train mới trên `corpusText` rồi lưu ra
  ## `spectron_bpe_tokenizer.json` (file riêng của Spectron, đúng schema
  ## GPT-2/Llama-style BPE thật, KHÔNG phải bảng byte-identity giả trước
  ## đây và KHÔNG đụng tới tokenizer.json của model khác trong repo).
  if fileExists(BPE_TOKENIZER_PATH):
    result = loadBPE(BPE_TOKENIZER_PATH)
  else:
    result = trainBPE(corpusText, numMerges)
    saveBPE(result, BPE_TOKENIZER_PATH)


# ═══════════════════════════════════════════════════════════════
# LINEAR++ ATTENTION (causal linear attention, feature map elu(x)+1)
# ═══════════════════════════════════════════════════════════════
## Ý tưởng (Katharopoulos et al., "Transformers are RNNs"): thay vì
## softmax(QK^T)V, dùng feature map phi(x) = elu(x)+1 (>0 luôn) rồi:
##   S[i,j]   = phi(Q)_i . phi(K)_j          (j<=i, nhân quả)
##   Num[i]   = sum_{j<=i} S[i,j] * V[j]
##   Den[i]   = sum_{j<=i} S[i,j]
##   Out[i]   = Num[i] / (Den[i] + eps)
## Viết dưới dạng "quadratic form" [T,T] giống hệt Mamba2SSD bên dưới (mask
## nhân quả a=1, KHÔNG suy giảm) - dùng chung code path/kiểm chứng được,
## không cần kernel/scan riêng. Forward+backward đều closed-form, không
## tham số ẩn (feature map cố định, không học thêm gì ngoài Q/K/V/out proj).

const LP_EPS = 1e-6'f32

proc eluPlus1(x: float32): float32 =
  if x > 0'f32: x + 1'f32
  else: exp(x)

proc eluPlus1Grad(x: float32): float32 =
  ## d/dx [elu(x)+1]: 1 nếu x>0, exp(x) nếu x<=0 (khớp forward ở trên).
  if x > 0'f32: 1'f32
  else: exp(x)

proc applyFeatureMap(x: seq[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len: result[i] = eluPlus1(x[i])

proc cpuLinearPlusForward(qData, kData, vData: seq[float32], B, H, T, D: int):
    tuple[outData, qf, kf, den: seq[float32]] =
  ## qf/kf: feature-mapped Q/K (cùng layout [B,T,H*D]) - trả về để backward
  ## khỏi phải tính lại. den: mẫu số [B,T,H] (1 số/head/token).
  let C = H * D
  result.qf = applyFeatureMap(qData)
  result.kf = applyFeatureMap(kData)
  result.den = newSeq[float32](B * T * H)
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

proc cpuLinearPlusBackward(qData, kData, vData, qf, kf, den, dOutData: seq[float32],
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
      # Bước 1: dựng lại Num[i] (cần cho dDen), rồi lan gradient theo i,j.
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
        var dDotNumDOut = 0'f32   # dOut_i . Num_i (dùng cho dDen)
        for d in 0 ..< D:
          dDotNumDOut += dOutData[idx(b, i, h, d)] * numI[d]
        let dDen = -dDotNumDOut * invDen * invDen

        for j in 0 .. i:
          var score = 0'f32
          for d in 0 ..< D:
            score += qf[idx(b, i, h, d)] * kf[idx(b, j, h, d)]
          # dScore từ nhánh Num: dNum_i . V_j ; từ nhánh Den: dDen (như nhau mọi j<=i)
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

  # dQf/dKf -> dQ/dK qua đạo hàm feature map elu(x)+1
  for i in 0 ..< dQf.len:
    result.dQ[i] = dQf[i] * eluPlus1Grad(qData[i])
    result.dK[i] = dKf[i] * eluPlus1Grad(kData[i])

type
  LinearPlusAttention* = object
    nHeads*, headDim*, embedDim*: int
    qkvProj*: Linear   # x -> [Q|K|V], mỗi phần embedDim
    outProj*: Linear
    useSession*: bool  # giữ cờ cho tương thích API (bản CPU-tham chiếu ở
                        # trên đã correctness-first; đường GPU-session có
                        # thể thêm sau theo đúng khuôn Mamba2SSD nếu cần
                        # tăng tốc, không đổi kết quả).

proc newLinearPlusAttention*(embedDim, nHeads: int, useSession: bool = true): LinearPlusAttention =
  assert embedDim mod nHeads == 0
  result.embedDim = embedDim
  result.nHeads = nHeads
  result.headDim = embedDim div nHeads
  result.qkvProj = newLinear(embedDim, 3 * embedDim)
  result.outProj = newLinear(embedDim, embedDim)
  result.useSession = useSession

proc splitQKV(proj: Tensor, B, T, C: int): tuple[q, k, v: seq[float32]] =
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
  let (qData, kData, vData) = splitQKV(proj, B, T, C)
  let (outData, _, _, _) = cpuLinearPlusForward(qData, kData, vData, B, m.nHeads, T, m.headDim)
  var y = newTensor(@[B, T, C])
  y.data = outData
  result = m.outProj.forward(y, ctx)

proc backwardLinearPlus*(m: LinearPlusAttention, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dQkvW, dQkvB, dOutW, dOutB: Tensor] =
  let B = x.shape[0]; let T = x.shape[1]; let C = x.shape[2]
  let proj = m.qkvProj.forward(x, ctx)
  let (qData, kData, vData) = splitQKV(proj, B, T, C)
  let (outData, qf, kf, den) = cpuLinearPlusForward(qData, kData, vData, B, m.nHeads, T, m.headDim)

  var y = newTensor(@[B, T, C]); y.data = outData
  let dOutP = m.outProj.backward(y, dOut, ctx)
  let dY = dOutP.dX.data

  let (dQ, dK, dV) = cpuLinearPlusBackward(qData, kData, vData, qf, kf, den, dY,
                                            B, m.nHeads, T, m.headDim)

  var dProj = newTensor(proj.shape)
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
# TRANSFORMER BLOCK LINEAR++ (Pre-LN, giống khuôn TransformerBlock)
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
  var dX2 = dFF.dX
  let dLN2 = blk.ln2.backward(x2, dX2, ctx)
  var dX2Total = dLN2.dX
  for i in 0 ..< dX2Total.data.len:
    dX2Total.data[i] += dOut.data[i]   # residual x2 = x + attnOut

  let dAttn = backwardLinearPlus(blk.attn, x1, dX2Total, ctx)
  let dLN1 = blk.ln1.backward(x, dAttn.dX, ctx)
  var dX0 = dLN1.dX
  for i in 0 ..< dX0.data.len:
    dX0.data[i] += dX2Total.data[i]    # residual gốc

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
  ## L[i,j] = a^(i-j) nếu i>=j, else 0.
  result = newSeq[float32](T * T)
  for i in 0 ..< T:
    var v = 1'f32
    for j in countdown(i, 0):
      result[i * T + j] = v
      v *= a

proc cpuMamba2Forward(xgData, bData, cData: seq[float32], B, H, T, D: int,
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

proc cpuMamba2Backward(xgData, bData, cData, dOutData: seq[float32], B, H, T, D: int,
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

proc forwardMamba2*(m: Mamba2SSD, x: Tensor, ctx: Backend): Tensor =
  let B = x.shape[0]; let T = x.shape[1]; let C = x.shape[2]
  assert C == m.embedDim
  let H = m.nHeads; let D = m.headDim

  let proj = m.inProj.forward(x, ctx)
  let (xgData, bData, cData, gateData) = splitXgBC(proj, B, T, C)

  var yData: seq[float32]

  if m.useSession and ctx.kind in {bkMetal, bkCuda, bkOpenCL}:
    let backend = ctx.kind.toByby()
    yData = newSeq[float32](B * T * C)
    var sessionOk = sessionBegin(backend)

    if sessionOk:
      var scoreHandles: seq[SessionHandle] = @[]
      var maskHandles: seq[SessionHandle] = @[]
      var oHandles: seq[SessionHandle] = @[]

      block encodeLoop:
        for h in 0 ..< H:
          var hMask = sessionUpload(backend, buildDecayMask(T, m.aVals[h]))
          for b in 0 ..< B:
            var cSlice = newSeq[float32](T * D)
            var bSliceT = newSeq[float32](D * T)
            var xgSlice = newSeq[float32](T * D)
            for t in 0 ..< T:
              let off = b * T * C + t * C + h * D
              for d in 0 ..< D:
                cSlice[t * D + d] = cData[off + d]
                bSliceT[d * T + t] = bData[off + d]
                xgSlice[t * D + d] = xgData[off + d]

            var hC = sessionUpload(backend, cSlice)
            var hBT = sessionUpload(backend, bSliceT)
            var hXg = sessionUpload(backend, xgSlice)
            var hScore = sessionAllocScratch(backend, T * T)
            var hScoreMasked = sessionAllocScratch(backend, T * T)
            var hO = sessionAllocScratch(backend, T * D)

            if not sessionMatmul(backend, hC, hBT, hScore, T, D, T):
              sessionOk = false
            elif not sessionVecOp(backend, "mul", hScore, hMask, hScoreMasked, T * T):
              sessionOk = false
            elif not sessionMatmul(backend, hScoreMasked, hXg, hO, T, T, D):
              sessionOk = false

            sessionFree(backend, hC); sessionFree(backend, hBT); sessionFree(backend, hXg)
            sessionFree(backend, hScore)
            scoreHandles.add(hScoreMasked)
            oHandles.add(hO)
            if not sessionOk: break encodeLoop
          maskHandles.add(hMask)

      if sessionOk and sessionEnd(backend):
        var hi = 0
        for h in 0 ..< H:
          for b in 0 ..< B:
            let raw = sessionRead(backend, oHandles[hi], T * D)
            for t in 0 ..< T:
              let dstOff = b * T * C + t * C + h * D
              for d in 0 ..< D:
                yData[dstOff + d] = raw[t * D + d]
            inc hi
      else:
        warnFallbackOnce(warnedSessionMidFail, "sessionMatmul/sessionVecOp/sessionEnd that bai giua chung tren backend " & $ctx.kind)
        yData = cpuMamba2Forward(xgData, bData, cData, B, H, T, D, m.aVals)

      for hd in scoreHandles.mitems: sessionFree(backend, hd)
      for hd in oHandles.mitems: sessionFree(backend, hd)
      for hd in maskHandles.mitems: sessionFree(backend, hd)
    else:
      warnFallbackOnce(warnedSessionBeginFail, "sessionBegin() that bai tren backend " & $ctx.kind)
      yData = cpuMamba2Forward(xgData, bData, cData, B, H, T, D, m.aVals)
  else:
    if m.useSession and ctx.kind in {bkMetal, bkCuda, bkOpenCL}:
      discard
    else:
      warnFallbackOnce(warnedNoSession, "khong dung session (useSession=false hoac backend " & $ctx.kind & " khong ho tro session)")
    yData = cpuMamba2Forward(xgData, bData, cData, B, H, T, D, m.aVals)

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

  let yData = cpuMamba2Forward(xgData, bData, cData, B, H, T, D, m.aVals)
  let gateAct = beSigmoid(ctx, gateData)
  let gated = beMul(ctx, yData, gateAct)
  var y = newTensor(@[B, T, C]); y.data = gated

  let dOutP = m.outProj.backward(y, dOut, ctx)
  let dGated = dOutP.dX.data

  var dGateAct = newSeq[float32](dGated.len)
  var dY = newSeq[float32](dGated.len)
  for i in 0 ..< dGated.len:
    dGateAct[i] = dGated[i] * yData[i]
    dY[i] = dGated[i] * gateAct[i]
  var dGateRaw = newSeq[float32](dGateAct.len)
  for i in 0 ..< dGateAct.len:
    dGateRaw[i] = dGateAct[i] * gateAct[i] * (1'f32 - gateAct[i])

  let (dXg, dB, dC) = cpuMamba2Backward(xgData, bData, cData, dY, B, H, T, D, m.aVals)

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

proc concatLastDim(a, b: Tensor): Tensor =
  let B = a.shape[0]; let T = a.shape[1]; let C = a.shape[2]
  result = newTensor(@[B, T, 2 * C])
  for i in 0 ..< B * T:
    for c in 0 ..< C:
      result.data[i * 2 * C + c] = a.data[i * C + c]
      result.data[i * 2 * C + C + c] = b.data[i * C + c]

proc splitLastDim(x: Tensor, C: int): tuple[a, b: Tensor] =
  ## Ngược lại concatLastDim: [B,T,2C] -> 2 x [B,T,C].
  let B = x.shape[0]; let T = x.shape[1]
  result.a = newTensor(@[B, T, C])
  result.b = newTensor(@[B, T, C])
  for i in 0 ..< B * T:
    for c in 0 ..< C:
      result.a.data[i * C + c] = x.data[i * 2 * C + c]
      result.b.data[i * C + c] = x.data[i * 2 * C + C + c]

proc forward*(blk: TransformerBlockSpectron, x: Tensor, ctx: Backend): Tensor =
  let x1 = blk.ln1.forward(x, ctx)
  let attnOut = forwardLinearPlus(blk.attn, x1, ctx)
  let mambaOut = forwardMamba2(blk.mamba, x1, ctx)

  let fuseIn = concatLastDim(attnOut, mambaOut)
  let gateLogit = blk.fuseGate.forward(fuseIn, ctx)
  let gate = beSigmoid(ctx, gateLogit.data)

  var combined = newTensor(attnOut.shape)
  for i in 0 ..< combined.data.len:
    combined.data[i] = gate[i] * attnOut.data[i] + (1'f32 - gate[i]) * mambaOut.data[i]

  let x2 = addT(ctx, x, combined)
  let x3 = blk.ln2.forward(x2, ctx)
  let ffOut = blk.ff.forward(x3, ctx)
  result = addT(ctx, x2, ffOut)

proc backward*(blk: TransformerBlockSpectron, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX: Tensor,
    dAttnQkvW, dAttnQkvB, dAttnOutW, dAttnOutB: Tensor,
    dMambaInW, dMambaInB, dMambaOutW, dMambaOutB: Tensor,
    dFuseW, dFuseB: Tensor,
    dFf1W, dFf1B, dFf2W, dFf2B: Tensor,
    dLn1G, dLn1B, dLn2G, dLn2B: Tensor] =
  let C = x.shape[2]

  # ---- forward lại để có mọi activation trung gian ----
  let x1 = blk.ln1.forward(x, ctx)
  let attnOut = forwardLinearPlus(blk.attn, x1, ctx)
  let mambaOut = forwardMamba2(blk.mamba, x1, ctx)
  let fuseIn = concatLastDim(attnOut, mambaOut)
  let gateLogit = blk.fuseGate.forward(fuseIn, ctx)
  let gate = beSigmoid(ctx, gateLogit.data)
  var combined = newTensor(attnOut.shape)
  for i in 0 ..< combined.data.len:
    combined.data[i] = gate[i] * attnOut.data[i] + (1'f32 - gate[i]) * mambaOut.data[i]
  let x2 = addT(ctx, x, combined)
  let x3 = blk.ln2.forward(x2, ctx)

  # ---- backward FF + LN2 ----
  let dFF = blk.ff.backward(x3, dOut, ctx)
  let dLN2 = blk.ln2.backward(x2, dFF.dX, ctx)
  var dX2 = dLN2.dX
  for i in 0 ..< dX2.data.len:
    dX2.data[i] += dOut.data[i]   # residual x2 = x + combined

  # ---- backward gated fusion ----
  var dGate = newSeq[float32](dX2.data.len)
  var dAttnOutPartial = newSeq[float32](dX2.data.len)
  var dMambaOutPartial = newSeq[float32](dX2.data.len)
  for i in 0 ..< dX2.data.len:
    dGate[i] = dX2.data[i] * (attnOut.data[i] - mambaOut.data[i])
    dAttnOutPartial[i] = dX2.data[i] * gate[i]
    dMambaOutPartial[i] = dX2.data[i] * (1'f32 - gate[i])

  var dGateLogit = newTensor(gateLogit.shape)
  for i in 0 ..< dGateLogit.data.len:
    dGateLogit.data[i] = dGate[i] * gate[i] * (1'f32 - gate[i])   # sigmoid'

  let dFuse = blk.fuseGate.backward(fuseIn, dGateLogit, ctx)
  let (dAttnOutFuse, dMambaOutFuse) = splitLastDim(dFuse.dX, C)

  var dAttnOutTotal = newTensor(attnOut.shape)
  var dMambaOutTotal = newTensor(mambaOut.shape)
  for i in 0 ..< dAttnOutTotal.data.len:
    dAttnOutTotal.data[i] = dAttnOutPartial[i] + dAttnOutFuse.data[i]
    dMambaOutTotal.data[i] = dMambaOutPartial[i] + dMambaOutFuse.data[i]

  # ---- backward 2 nhánh (attn Linear++ và Mamba2) ----
  let dAttn = backwardLinearPlus(blk.attn, x1, dAttnOutTotal, ctx)
  let dMamba = backwardMamba2(blk.mamba, x1, dMambaOutTotal, ctx)

  var dX1Total = newTensor(x1.shape)
  for i in 0 ..< dX1Total.data.len:
    dX1Total.data[i] = dAttn.dX.data[i] + dMamba.dX.data[i]

  let dLN1 = blk.ln1.backward(x, dX1Total, ctx)
  var dX0 = dLN1.dX
  for i in 0 ..< dX0.data.len:
    dX0.data[i] += dX2.data[i]   # residual gốc x2 = x + combined

  result = (
    dX0,
    dAttn.dQkvW, dAttn.dQkvB, dAttn.dOutW, dAttn.dOutB,
    dMamba.dInProjW, dMamba.dInProjB, dMamba.dOutProjW, dMamba.dOutProjB,
    dFuse.dW, dFuse.dB,
    dFF.dFc1W, dFF.dFc1B, dFF.dFc2W, dFF.dFc2B,
    dLN1.dGamma, dLN1.dBeta, dLN2.dGamma, dLN2.dBeta
  )

# ═══════════════════════════════════════════════════════════════
# NIMFORMER MODEL VỚI SPECTRON BLOCKS (embed -> N blocks -> lm head)
# ═══════════════════════════════════════════════════════════════

type
  NimformerModelSpectron* = object
    embed*: Embedding
    blocks*: seq[TransformerBlockSpectron]
    outProj*: Linear
    vocab*: int

proc newNimformerModelSpectron*(vocab, embedDim, nHeads, nLayers, ffMult: int,
                                 useSession: bool = true): NimformerModelSpectron =
  result.vocab = vocab
  result.embed = newEmbedding(vocab, embedDim)
  result.blocks = newSeq[TransformerBlockSpectron](nLayers)
  for i in 0 ..< nLayers:
    result.blocks[i] = newTransformerBlockSpectron(embedDim, nHeads, ffMult, useSession)
  result.outProj = newLinear(embedDim, vocab)

proc forwardBatch*(m: NimformerModelSpectron, idsBatch: seq[seq[int]], ctx: Backend): Tensor =
  var x = m.embed.lookupBatch(idsBatch, ctx)
  for blk in m.blocks:
    x = blk.forward(x, ctx)
  result = m.outProj.forward(x, ctx)

proc forward*(m: NimformerModelSpectron, ids: seq[int], ctx: Backend): Tensor =
  let logits = m.forwardBatch(@[ids], ctx)
  result = Tensor(data: logits.data, shape: logits.shape[1 .. ^1])

type
  SpectronGrads* = object
    ## Gom toàn bộ gradient của model theo đúng thứ tự ta sẽ cập nhật -
    ## mảng "phẳng" các cặp (Tensor tham số, Tensor gradient) để train()
    ## có thể lặp qua và gọi apfAdamStep cho từng cái mà không cần biết
    ## cấu trúc chi tiết bên trong từng block.
    dOutProjW*, dOutProjB*: Tensor
    dBlocks*: seq[tuple[
      dAttnQkvW, dAttnQkvB, dAttnOutW, dAttnOutB,
      dMambaInW, dMambaInB, dMambaOutW, dMambaOutB,
      dFuseW, dFuseB,
      dFf1W, dFf1B, dFf2W, dFf2B,
      dLn1G, dLn1B, dLn2G, dLn2B: Tensor]]
    dEmbed*: Tensor

proc backwardBatch*(m: NimformerModelSpectron, idsBatch: seq[seq[int]], dLoss: Tensor,
                     ctx: Backend): SpectronGrads =
  var x = m.embed.lookupBatch(idsBatch, ctx)
  var hiddenStates: seq[Tensor] = @[x]
  for blk in m.blocks:
    x = blk.forward(x, ctx)
    hiddenStates.add(x)

  let dOutProj = m.outProj.backward(x, dLoss, ctx)
  result.dOutProjW = dOutProj.dW
  result.dOutProjB = dOutProj.dB
  var dOut = dOutProj.dX

  result.dBlocks = newSeq[typeof(result.dBlocks[0])](m.blocks.len)
  for i in countdown(m.blocks.len - 1, 0):
    let blk = m.blocks[i]
    let xPrev = hiddenStates[i]
    let dBlk = blk.backward(xPrev, dOut, ctx)
    dOut = dBlk.dX
    result.dBlocks[i] = (
      dBlk.dAttnQkvW, dBlk.dAttnQkvB, dBlk.dAttnOutW, dBlk.dAttnOutB,
      dBlk.dMambaInW, dBlk.dMambaInB, dBlk.dMambaOutW, dBlk.dMambaOutB,
      dBlk.dFuseW, dBlk.dFuseB,
      dBlk.dFf1W, dBlk.dFf1B, dBlk.dFf2W, dBlk.dFf2B,
      dBlk.dLn1G, dBlk.dLn1B, dBlk.dLn2G, dBlk.dLn2B
    )
  result.dEmbed = m.embed.backward(idsBatch, dOut, ctx)

proc backward*(m: NimformerModelSpectron, ids: seq[int], dLoss: Tensor, ctx: Backend): SpectronGrads =
  let dLossBatch = Tensor(data: dLoss.data, shape: @[1] & dLoss.shape)
  result = m.backwardBatch(@[ids], dLossBatch, ctx)

# ═══════════════════════════════════════════════════════════════
# LOSS: softmax cross-entropy (batch), trả về loss trung bình + dLogits
# ═══════════════════════════════════════════════════════════════
proc softmaxCrossEntropy*(logits: Tensor, targets: seq[seq[int]], ctx: Backend):
    tuple[loss: float32, dLogits: Tensor] =
  ## logits: [B,T,V]. targets: [B][T] token id kế tiếp (next-token).
  let B = logits.shape[0]; let T = logits.shape[1]; let V = logits.shape[2]
  let probs = beSoftmax(ctx, logits.data, B * T, V)
  var dLogits = newTensor(logits.shape)
  var totalLoss = 0'f32
  for b in 0 ..< B:
    for t in 0 ..< T:
      let base = (b * T + t) * V
      let tgt = targets[b][t]
      totalLoss += -ln(max(probs[base + tgt], 1e-9'f32))
      for v in 0 ..< V:
        dLogits.data[base + v] = probs[base + v]
      dLogits.data[base + tgt] -= 1'f32
  let n = float32(B * T)
  totalLoss /= n
  for i in 0 ..< dLogits.data.len:
    dLogits.data[i] /= n
  result = (totalLoss, dLogits)

# ═══════════════════════════════════════════════════════════════
# ADAM STATE CHO TOÀN BỘ MODEL + 1 BƯỚC train()
# ═══════════════════════════════════════════════════════════════
type
  ModelAdamState* = object
    outProjW*, outProjB*: ApfAdamState
    blocks*: seq[tuple[
      attnQkvW, attnQkvB, attnOutW, attnOutB,
      mambaInW, mambaInB, mambaOutW, mambaOutB,
      fuseW, fuseB,
      ff1W, ff1B, ff2W, ff2B,
      ln1G, ln1B, ln2G, ln2B: ApfAdamState]]
    embed*: ApfAdamState

proc newModelAdamState*(m: NimformerModelSpectron): ModelAdamState =
  result.outProjW = newApfAdamState(m.outProj.weight.data.len)
  result.outProjB = newApfAdamState(m.outProj.bias.data.len)
  result.embed = newApfAdamState(m.embed.weight.data.len)
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
# CHECKPOINT (.nimq) - DÙNG LẠI ĐÚNG format nhị phân NIMQ2 đã có sẵn trong
# quant.nim (saveQuantStateDict/loadQuantStateDict - cùng hàm export_to_nimq.py
# dùng để xuất model.nimq cho model lớn). Ở đây lưu từng tensor dạng
# qkFp32Raw (KHÔNG nén) vì model demo nhỏ, ưu tiên đúng số hơn tiết kiệm
# dung lượng. Adam state (m/v/step) CHƯA được lưu - resume sẽ reset lại
# momentum Adam, nhưng trọng số (phần quan trọng nhất để không mất công
# train) được giữ nguyên 100%.
# ═══════════════════════════════════════════════════════════════

proc modelToStateDict(m: NimformerModelSpectron): seq[(string, QuantTensor)] =
  result = @[]
  template add(name: string, t: Tensor) =
    result.add((name, quantizeTensor(t.data, t.shape, qkFp32Raw)))
  add("embed.weight", m.embed.weight)
  add("outProj.weight", m.outProj.weight)
  add("outProj.bias", m.outProj.bias)
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

proc stateDictToModel(sd: seq[(string, QuantTensor)],
                       vocab, embedDim, nHeads, nLayers, ffMult: int): NimformerModelSpectron =
  result = newNimformerModelSpectron(vocab, embedDim, nHeads, nLayers, ffMult, useSession = true)
  var lookup = initTable[string, QuantTensor]()
  for (name, qt) in sd: lookup[name] = qt
  template restore(name: string, t: var Tensor) =
    if name in lookup:
      t.data = dequantizeTensor(lookup[name])
    else:
      stderr.writeLine("[CANH BAO] Checkpoint thieu tensor '" & name & "' - giu nguyen gia tri khoi tao ngau nhien")
  restore("embed.weight", result.embed.weight)
  restore("outProj.weight", result.outProj.weight)
  restore("outProj.bias", result.outProj.bias)
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

const MetaStepKey = "__meta_step__"
  ## Tên tensor "giả" (shape [1], kind qkFp32Raw) dùng để giấu số step đã
  ## train vào trong state dict - KHÔNG cần đổi format nhị phân NIMQ2, file
  ## cũ (chưa có key này) vẫn load được bình thường (step mặc định = 0).

proc saveCheckpoint*(m: NimformerModelSpectron, path: string, step: int = 0) =
  ## arch = [vocab, embedDim, nHeads, nLayers, ffMult] - đúng array[5,int]
  ## mà saveQuantStateDict yêu cầu (không cần định dạng riêng).
  ## `step` = số bước train đã hoàn thành TÍNH ĐẾN checkpoint này, được lưu
  ## kèm theo để resume đúng số step thay vì luôn quay lại step 1.
  let ffMult = m.blocks[0].ff.fc1.outF div m.embed.dim
  let arch = [m.vocab, m.embed.dim, m.blocks[0].attn.nHeads, m.blocks.len, ffMult]
  var sd = modelToStateDict(m)
  sd.add((MetaStepKey, quantizeTensor(@[float32(step)], @[1], qkFp32Raw)))
  saveQuantStateDict(path, arch, sd)

proc loadCheckpoint*(path: string): tuple[model: NimformerModelSpectron, step: int] =
  let (arch, sd) = loadQuantStateDict(path)
  var step = 0
  var sdModel: seq[(string, QuantTensor)] = @[]
  for (name, qt) in sd:
    if name == MetaStepKey:
      let v = dequantizeTensor(qt)
      if v.len > 0: step = int(v[0])
    else:
      sdModel.add((name, qt))
  result.model = stateDictToModel(sdModel, arch[0], arch[1], arch[2], arch[3], arch[4])
  result.step = step

proc trainStep*(m: var NimformerModelSpectron, adam: var ModelAdamState,
                 idsBatch: seq[seq[int]], targetsBatch: seq[seq[int]],
                 ctx: Backend, lr: float32 = 3e-4'f32,
                 requantizeEvery: int = 50): float32 =
  ## Một bước train đầy đủ: forward -> loss -> backward -> Adam update.
  ## Trả về loss (đã average theo batch*seq) để log.
  ##
  ## LƯU Ý QUAN TRỌNG về `requantizeEvery`: apfAdamStep (trong nimformer.nim)
  ## mặc định requantizeEvery=1, tức là RE-QUANTIZE trọng số về CustomFloat
  ## (APF) NGAY SAU MỖI bước Adam. Ở vài chục bước đầu, độ lớn update còn
  ## rất nhỏ so với độ phân giải mantissa mà buildCustomDtypeForTensor chọn
  ## -> update vừa cộng vào bị làm tròn mất NGAY trong cùng bước, trọng số
  ## coi như bị đóng băng ở gần giá trị khởi tạo. Hậu quả: logits ~ 0 suốt
  ## quá trình train -> softmax ~ đều -> loss ~ ln(vocab_size) và đứng yên y
  ## hệt (ví dụ vocab=756 thì ln(756)=6.628 - đúng triệu chứng "loss đứng ở
  ## 6.62-6.63 mãi" quan sát được). Để vẫn DÙNG APF (theo yêu cầu) nhưng
  ## không giết tín hiệu gradient, ta chỉ requantize định kỳ mỗi
  ## `requantizeEvery` bước (kiểu quantization-aware training thật: train ở
  ## fp32 đầy đủ giữa các lần, rồi mới ép về APF để giữ weight trong dải
  ## biểu diễn được khi export) thay vì mỗi bước 1 lần.
  let logits = m.forwardBatch(idsBatch, ctx)
  let (loss, dLogits) = softmaxCrossEntropy(logits, targetsBatch, ctx)
  let grads = m.backwardBatch(idsBatch, dLogits, ctx)

  discard apfAdamStep(m.outProj.weight, grads.dOutProjW, adam.outProjW, lr, requantizeEvery = requantizeEvery)
  discard apfAdamStep(m.outProj.bias, grads.dOutProjB, adam.outProjB, lr, requantizeEvery = requantizeEvery)
  discard apfAdamStep(m.embed.weight, grads.dEmbed, adam.embed, lr, requantizeEvery = requantizeEvery)

  for i in 0 ..< m.blocks.len:
    let g = grads.dBlocks[i]
    var a = adam.blocks[i]
    discard apfAdamStep(m.blocks[i].attn.qkvProj.weight, g.dAttnQkvW, a.attnQkvW, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].attn.qkvProj.bias, g.dAttnQkvB, a.attnQkvB, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].attn.outProj.weight, g.dAttnOutW, a.attnOutW, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].attn.outProj.bias, g.dAttnOutB, a.attnOutB, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].mamba.inProj.weight, g.dMambaInW, a.mambaInW, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].mamba.inProj.bias, g.dMambaInB, a.mambaInB, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].mamba.outProj.weight, g.dMambaOutW, a.mambaOutW, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].mamba.outProj.bias, g.dMambaOutB, a.mambaOutB, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].fuseGate.weight, g.dFuseW, a.fuseW, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].fuseGate.bias, g.dFuseB, a.fuseB, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].ff.fc1.weight, g.dFf1W, a.ff1W, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].ff.fc1.bias, g.dFf1B, a.ff1B, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].ff.fc2.weight, g.dFf2W, a.ff2W, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].ff.fc2.bias, g.dFf2B, a.ff2B, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].ln1.gamma, g.dLn1G, a.ln1G, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].ln1.beta, g.dLn1B, a.ln1B, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].ln2.gamma, g.dLn2G, a.ln2G, lr, requantizeEvery = requantizeEvery)
    discard apfAdamStep(m.blocks[i].ln2.beta, g.dLn2B, a.ln2B, lr, requantizeEvery = requantizeEvery)
    adam.blocks[i] = a

  result = loss

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
  ## Sinh văn bản autoregressive đơn giản (không KV-cache: forward lại toàn
  ## bộ chuỗi mỗi bước - đủ dùng cho demo/test; muốn nhanh hơn cho chuỗi dài
  ## thì thêm KV-cache sau, không đổi kết quả).
  var rng = initRand(seed)
  result = promptIds
  for _ in 0 ..< maxNewTokens:
    let logits = m.forward(result, ctx)   # [T, vocab]
    let T = logits.shape[0]
    let V = logits.shape[1]
    let lastLogits = logits.data[(T - 1) * V ..< T * V]
    let nextId = sampleFromLogits(lastLogits, temperature, rng)
    result.add(nextId)

# ═══════════════════════════════════════════════════════════════
# DEMO: train nhỏ trên databricks-dolly-15k.jsonl (byte-level tokenizer)
# rồi sinh thử vài token - chạy được ngay bằng `nim c -r test_nimformer.nim`
# ═══════════════════════════════════════════════════════════════
when isMainModule:
  randomize(1234)
  let ctx = newBackend()

  # ---- Nap corpus tho (text) tu file co san trong repo, fallback neu thieu ----
  proc loadCorpusText(path: string): string =
    if fileExists(path):
      result = readFile(path)
    else:
      result = "Nimformer Spectron demo corpus. ".repeat(200)

  let rawCorpus = loadCorpusText(CFG_CORPUS_PATH)
  # Train BPE tren 1 phan dau corpus cho nhanh (numMerges lan dem cap tren
  # toan bo file lon se rat cham voi vong lap Nim thuan); phan con lai van
  # dung de lay mau train model ben duoi.
  let bpeTrainSlice = rawCorpus[0 ..< min(rawCorpus.len, CFG_BPE_TRAIN_CHARS)]
  echo "== Dang train BPE tokenizer (hoac nap lai neu da co) =="
  let tok = getOrTrainBPE(bpeTrainSlice, CFG_NUM_MERGES)
  let vocab = tok.vocabSize
  echo &"Tokenizer: vocab_size = {vocab} (256 byte co so + {tok.merges.len} merge)"

  var model: NimformerModelSpectron
  var startStep = 0   # step da train xong tinh den checkpoint duoc nap (0 neu train tu dau)
  if fileExists(CFG_CHECKPOINT_PATH):
    echo &"== Tim thay checkpoint {CFG_CHECKPOINT_PATH} -> nap lai de resume train =="
    (model, startStep) = loadCheckpoint(CFG_CHECKPOINT_PATH)
    echo &"  -> resume tu step {startStep + 1} (checkpoint da train {startStep} step)"
  else:
    model = newNimformerModelSpectron(vocab, CFG_EMBED_DIM, CFG_N_HEADS, CFG_N_LAYERS,
                                        CFG_FF_MULT, useSession = true)
  var adam = newModelAdamState(model)   # Adam moment chua duoc luu trong checkpoint - luon bat dau lai tu 0

  # Encode toan bo corpus 1 lan bang BPE da train, dung chung cho sampling.
  let corpusIds = encodeBPE(tok, rawCorpus)
  doAssert corpusIds.len > CFG_SEQ_LEN + 1, "Corpus qua ngan de train demo"

  proc sampleBatch(ids: seq[int], B, T: int, rng: var Rand):
      tuple[ids, targets: seq[seq[int]]] =
    result.ids = newSeq[seq[int]](B)
    result.targets = newSeq[seq[int]](B)
    for b in 0 ..< B:
      let start = rng.rand(ids.len - T - 2)
      var xIds = newSeq[int](T)
      var tgt = newSeq[int](T)
      for t in 0 ..< T:
        xIds[t] = ids[start + t]
        tgt[t] = ids[start + t + 1]
      result.ids[b] = xIds
      result.targets[b] = tgt

  var rng = initRand(42)
  echo &"== Bat dau train demo: {CFG_N_STEPS} buoc (tu step {startStep + 1}), batch={CFG_BATCH_SIZE}, seq={CFG_SEQ_LEN} =="
  # SUA: vong lap chay CFG_N_STEPS buoc TIEP THEO ke tu startStep (khong phai
  # luon quay lai step 1) - resume dung so step da luu trong checkpoint.
  for step in (startStep + 1) .. (startStep + CFG_N_STEPS):
    let (ids, targets) = sampleBatch(corpusIds, CFG_BATCH_SIZE, CFG_SEQ_LEN, rng)
    let loss = trainStep(model, adam, ids, targets, ctx, lr = CFG_LR,
                          requantizeEvery = CFG_REQUANTIZE_EVERY)
    if step mod CFG_LOG_EVERY == 0 or step == startStep + 1:
      echo &"step {step:>5} | loss = {loss:.4f}"

    if step mod CFG_CHECKPOINT_EVERY == 0:
      saveCheckpoint(model, CFG_CHECKPOINT_PATH, step)
      if step mod CFG_LOG_EVERY == 0 or step == startStep + CFG_CHECKPOINT_EVERY:
        echo &"  [checkpoint] da luu {CFG_CHECKPOINT_PATH} (step {step})"

  saveCheckpoint(model, CFG_CHECKPOINT_PATH, startStep + CFG_N_STEPS)
  echo &"== Da luu checkpoint cuoi cung: {CFG_CHECKPOINT_PATH} =="

  echo "== Sinh thu (greedy) tu 1 cau moi =="
  let promptIds = encodeBPE(tok, CFG_GEN_PROMPT)
  let outIds = model.generate(promptIds, maxNewTokens = CFG_GEN_MAX_NEW_TOKENS, ctx = ctx,
                               temperature = CFG_GEN_TEMPERATURE)
  echo "Output: ", decodeBPEStr(tok, outIds)

  closeBackend(ctx)