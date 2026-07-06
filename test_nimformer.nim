## main.nim
## Bản port của main.py sang framework Nim thuần (nimformer.nim + quant.nim +
## customfloat.nim + metal_ai.nim) — KHÔNG còn phụ thuộc Python/tinygrad/
## MetalCharLM nữa. Toàn bộ training loop, tokenizer, ghép batch, tối ưu Adam
## (ApfAdam) và lượng tử hoá checkpoint (int8/int4/fp8/APF) đều tự viết ở đây,
## dùng đúng API thật đã có trong 4 file thư viện (không sửa các file đó).
##
## Vì các file thư viện (nimformer/quant/customfloat/metal_ai) không có
## `when isMainModule`, main.nim chính là "code chạy" — giống vai trò của
## main.py trong bản Python gốc, nhưng linh hoạt hơn: mọi hằng số (SEQ,
## BATCH_SIZE, STEPS, kiểu lượng tử hoá khi lưu...) đều chỉnh được qua CLI
## thay vì hard-code.
##
## Build & chạy (macOS + Metal, xem thêm README_BUILD.md):
##   nim c -d:release -o:main main.nim
##   ./main --steps=2000 --seq=128 --batch=32 --lr=3e-3 \
##          --embed-dim=128 --heads=4 --layers=4 --ff-mult=4 \
##          --quant=auto --save=finetune.nimq
##
## Các nguồn dữ liệu (giống main.py, đọc file cục bộ nếu có / gọi mạng nếu
## không, và BỎ QUA êm nếu lỗi, giống hệt tinh thần try/except của bản gốc):
##   grammar.txt, million_games.pgn, english_words.txt,
##   databricks-dolly-15k.jsonl, Wikipedia (vài trang tech/history),
##   StackOverflow (StackExchange API), tự-đấu Stockfish (nếu có binary).

import std/[os, math, random, strformat, strutils, sequtils, tables, json,
            times, parseopt, osproc, streams, re, httpclient]
import quant, nimformer, metal_ai, customfloat

# ═══════════════════════════════════════════════════════════════
# Config — thay cho các hằng số cứng SEQ/BATCH_SIZE/STEPS ở đầu main.py
# ═══════════════════════════════════════════════════════════════

type
  QuantChoice = enum
    qcInt8, qcInt4, qcFp8E4M3, qcFp8E5M2, qcAuto, qcNone

  Config = object
    seqLen: int
    batchSize: int
    steps: int
    lr: float32
    embedDim, nHeads, nLayers, ffMult: int
    requantizeEvery: int      ## mỗi bao nhiêu bước Adam thì APF requantize lại param (trong lúc train)
    savePath: string
    tokenizerPath: string
    dataDir: string
    quant: QuantChoice        ## kiểu nén dùng khi LƯU checkpoint cuối (trọng số; bias/LN luôn fp32)
    ckptEvery: int
    logEvery: int
    stockfishPath: string
    stockfishGames: int
    stockfishPlies: int
    soTags: seq[string]
    soMaxPages: int
    wikiMaxPages: int
    seed: int

proc parseQuant(s: string): QuantChoice =
  case s.toLowerAscii
  of "int8": qcInt8
  of "int4": qcInt4
  of "fp8_e4m3", "fp8e4m3", "fp8": qcFp8E4M3
  of "fp8_e5m2", "fp8e5m2": qcFp8E5M2
  of "auto", "apf": qcAuto
  of "none", "fp32", "raw": qcNone
  else:
    stderr.writeLine &"[cảnh báo] --quant='{s}' không nhận diện được, dùng mặc định 'auto' (APF)"
    qcAuto

proc quantKindOf(choice: QuantChoice): QuantKind =
  case choice
  of qcInt8:    qkInt8
  of qcInt4:    qkInt4
  of qcFp8E4M3: qkFp8E4M3
  of qcFp8E5M2: qkFp8E5M2
  of qcAuto:    qkAuto
  of qcNone:    qkFp32Raw

proc defaultConfig(): Config =
  Config(
    seqLen: 128,
    batchSize: 32,
    steps: 10000,
    lr: 3e-3'f32,
    embedDim: 128, nHeads: 4, nLayers: 4, ffMult: 4,
    requantizeEvery: 50,
    savePath: "finetune.nimq",
    tokenizerPath: "tokenizer.json",
    dataDir: ".",
    quant: qcAuto,
    ckptEvery: 5,
    logEvery: 10,
    stockfishPath: "/usr/local/bin/stockfish",
    stockfishGames: 40,
    stockfishPlies: 10,
    soTags: @["python", "c", "swift", "objective-c", "nim"],
    soMaxPages: 100,
    wikiMaxPages: 10,
    seed: 1337
  )

proc parseArgs(): Config =
  result = defaultConfig()
  for kind, key, val in getopt():
    if kind != cmdLongOption: continue
    try:
      case key
      of "seq":              result.seqLen = parseInt(val)
      of "batch":             result.batchSize = parseInt(val)
      of "steps":             result.steps = parseInt(val)
      of "lr":                result.lr = parseFloat(val).float32
      of "embed-dim":         result.embedDim = parseInt(val)
      of "heads":             result.nHeads = parseInt(val)
      of "layers":            result.nLayers = parseInt(val)
      of "ff-mult":           result.ffMult = parseInt(val)
      of "requantize-every":  result.requantizeEvery = parseInt(val)
      of "save":              result.savePath = val
      of "tokenizer":         result.tokenizerPath = val
      of "data-dir":          result.dataDir = val
      of "quant":             result.quant = parseQuant(val)
      of "ckpt-every":        result.ckptEvery = parseInt(val)
      of "log-every":         result.logEvery = parseInt(val)
      of "stockfish-path":    result.stockfishPath = val
      of "stockfish-games":   result.stockfishGames = parseInt(val)
      of "stockfish-plies":   result.stockfishPlies = parseInt(val)
      of "so-tags":           result.soTags = val.split(",")
      of "so-max-pages":      result.soMaxPages = parseInt(val)
      of "wiki-max-pages":    result.wikiMaxPages = parseInt(val)
      of "seed":              result.seed = parseInt(val)
      of "help", "h":
        echo "Xem phần comment đầu main.nim để biết danh sách cờ (--steps, --seq, --batch, --lr, --quant, --save, ...)"
        quit(0)
      else:
        stderr.writeLine &"[cảnh báo] cờ không rõ: --{key}"
    except ValueError:
      stderr.writeLine &"[cảnh báo] giá trị không hợp lệ cho --{key}='{val}', bỏ qua"

# ═══════════════════════════════════════════════════════════════
# CharTokenizer — tokenizer byte-level thuần Nim (không cần thư viện ngoài,
# xử lý tốt cả UTF-8 vì mỗi byte là 1 token; vocab tối đa 256).
# ═══════════════════════════════════════════════════════════════

type
  CharTokenizer* = object
    vocabSize*: int
    itos: array[256, char]
    stoi: array[256, int]

proc newCharTokenizer*(texts: seq[string]): CharTokenizer =
  var present: array[256, bool]
  for t in texts:
    for ch in t:
      present[ord(ch)] = true
  for b in 0 .. 255: result.stoi[b] = -1
  var idx = 0
  for b in 0 .. 255:
    if present[b]:
      result.itos[idx] = char(b)
      result.stoi[b] = idx
      inc idx
  result.vocabSize = max(idx, 1)  # tối thiểu 1 để tránh model vocab=0

proc encode*(tok: CharTokenizer, s: string): seq[int] =
  result = newSeq[int](s.len)
  for i in 0 ..< s.len:
    let id = tok.stoi[ord(s[i])]
    result[i] = if id >= 0: id else: 0

proc decode*(tok: CharTokenizer, ids: seq[int]): string =
  result = newString(ids.len)
  for i, id in ids:
    result[i] = if id >= 0 and id < tok.vocabSize: tok.itos[id] else: '?'

proc saveTokenizer*(tok: CharTokenizer, path: string) =
  var arr = newJArray()
  for b in 0 .. 255:
    if tok.stoi[b] >= 0: arr.add(%b)
  writeFile(path, $(%*{"vocab_size": tok.vocabSize, "bytes": arr}))

proc loadTokenizer*(path: string): CharTokenizer =
  let j = parseJson(readFile(path))
  for b in 0 .. 255: result.stoi[b] = -1
  var idx = 0
  for v in j["bytes"]:
    let b = v.getInt
    result.itos[idx] = char(b)
    result.stoi[b] = idx
    inc idx
  result.vocabSize = max(idx, 1)

# ═══════════════════════════════════════════════════════════════
# Data loaders — port của các hàm load_* trong main.py.
# Mỗi hàm tự "nuốt" lỗi (file không tồn tại / mạng lỗi) và trả về @[],
# giống hệt tinh thần try/except im lặng của bản Python gốc.
# ═══════════════════════════════════════════════════════════════

proc loadGrammarTexts*(path: string): seq[string] =
  result = @[]
  if not fileExists(path): return
  for line in lines(path):
    let l = line.strip()
    if l.len > 0: result.add(l)

proc loadEnglishDict*(path: string): seq[string] =
  result = @[]
  if fileExists(path):
    for line in lines(path):
      let l = line.strip()
      if l.len > 0: result.add(l)
    return
  # fallback: tải danh sách từ tiếng Anh phổ biến từ GitHub (giống bản Python)
  try:
    var client = newHttpClient(timeout = 10000)
    defer: client.close()
    let body = client.getContent(
      "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt")
    for w in body.splitLines():
      if w.len > 0: result.add(w)
  except CatchableError:
    discard

proc loadDolly*(path: string): seq[string] =
  result = @[]
  if not fileExists(path): return
  for line in lines(path):
    if line.strip().len == 0: continue
    try:
      let obj = parseJson(line)
      let ins = obj{"instruction"}.getStr("").strip()
      let ctx = obj{"context"}.getStr("").strip()
      var resp = obj{"response"}.getStr("").strip()
      if resp.len == 0: resp = obj{"output"}.getStr("").strip()
      if ins.len == 0 and resp.len == 0: continue
      if ctx.len > 0: result.add(ins & "\n" & ctx & "\n" & resp)
      else: result.add(ins & "\n" & resp)
    except CatchableError:
      continue

proc loadPgnTxt*(path: string): seq[string] =
  ## Rút movetext (SAN) trực tiếp từ file PGN bằng regex thay vì replay từng
  ## nước đi qua 1 chess engine luật đầy đủ (Nim không có sẵn thư viện chess
  ## tương đương python-chess) — vẫn cho ra đúng chuỗi SAN thật trong file,
  ## chỉ là không kiểm tra tính hợp lệ, điều mà 1 char-LM không cần tới.
  result = @[]
  if not fileExists(path): return
  let content = readFile(path)
  for blk in content.split("\n\n"):
    let b = blk.strip()
    if b.len == 0 or b.startsWith("["): continue  # bỏ qua block tag-header
    var moves = b.replace(re"\{[^}]*\}", " ")            # bỏ comment {...}
    moves = moves.replace(re"\d+\.(\.\.)?", " ")           # bỏ số nước "12." / "12..."
    moves = moves.replace(re"(1-0|0-1|1/2-1/2|\*)\s*$", "") # bỏ ký hiệu kết quả
    moves = moves.replace(re"\s+", " ").strip()
    if moves.len > 0: result.add(moves)

proc loadStockfishSelfPlay*(nGames, plies: int, enginePath: string): seq[string] =
  ## Tự-đấu bằng chính Stockfish (nó tự chọn nước qua "bestmove"), nên
  ## KHÔNG cần cài luật cờ vua trong Nim. Ghi lại chuỗi nước UCI + eval cuối,
  ## thay vì FEN + "Final evaluation" như bản Python (không cần build board).
  result = @[]
  if enginePath.len == 0 or not fileExists(enginePath): return
  var p: Process
  try:
    p = startProcess(enginePath, options = {poUsePath})
  except CatchableError:
    return
  defer:
    try: p.close() except CatchableError: discard
  let pin = p.inputStream
  let pout = p.outputStream

  proc send(cmd: string) =
    try:
      pin.writeLine(cmd)
      pin.flush()
    except CatchableError: discard

  proc waitFor(token: string, maxLines = 2000) =
    var line: string
    var n = 0
    while n < maxLines and pout.readLine(line):
      inc n
      if token in line: return

  send("uci"); waitFor("uciok")
  send("isready"); waitFor("readyok")

  for g in 0 ..< nGames:
    var moveList: seq[string] = @[]
    for ply in 0 ..< plies:
      let posCmd = if moveList.len == 0: "position startpos"
                   else: "position startpos moves " & moveList.join(" ")
      send(posCmd)
      send("go depth 8")
      var bestMove = ""
      var line: string
      var n = 0
      while n < 2000 and pout.readLine(line):
        inc n
        if line.startsWith("bestmove"):
          let parts = line.splitWhitespace()
          if parts.len >= 2: bestMove = parts[1]
          break
      if bestMove.len == 0 or bestMove == "(none)": break
      moveList.add(bestMove)
    if moveList.len == 0: continue
    send("position startpos moves " & moveList.join(" "))
    send("eval")
    var evalLine = ""
    var line: string
    var n = 0
    while n < 500 and pout.readLine(line):
      inc n
      if "Final evaluation" in line:
        evalLine = line.strip()
        break
    result.add("MOVES: " & moveList.join(" ") & "\nEVAL: " & evalLine)

  send("quit")

proc stripHtmlTags(s: string): string =
  result = s.replace(re"<[^>]+>", " ")
  result = result.replace(re"\s+", " ").strip()

proc loadStackoverflowFirstPage*(tags: seq[string], pagesize, maxPages: int): seq[string] =
  result = @[]
  var client: HttpClient
  try:
    client = newHttpClient(timeout = 10000)
  except CatchableError:
    return
  defer: client.close()

  var firstPage = true
  for tag in tags:
    let page = if firstPage: 1 else: rand(1 .. max(maxPages, 1))
    firstPage = false
    let url = "https://api.stackexchange.com/2.3/questions" &
      "?pagesize=" & $pagesize & "&page=" & $page &
      "&order=desc&sort=activity&site=stackoverflow&filter=withbody&tagged=" & tag
    try:
      let body = client.getContent(url)
      let js = parseJson(body)
      if js.hasKey("items"):
        for item in js["items"]:
          let raw = item{"body"}.getStr("")
          let txt = stripHtmlTags(raw)
          if txt.len > 50: result.add(txt)
    except CatchableError:
      discard
    sleep(1000)

proc loadWikipediaTechHistory*(maxPages: int): seq[string] =
  result = @[]
  var topics = @["History_of_computing_hardware", "Operating_system", "Unix",
                 "Linux", "Artificial_intelligence"]
  shuffle(topics)
  let n = rand(1 .. min(max(maxPages, 1), topics.len))
  var client: HttpClient
  try:
    client = newHttpClient(timeout = 15000)
  except CatchableError:
    return
  defer: client.close()

  for i in 0 ..< n:
    try:
      let html = client.getContent("https://en.wikipedia.org/wiki/" & topics[i])
      var paragraphs: seq[string] = @[]
      for blk in html.findAll(re"(?s)<p[^>]*>.*?</p>"):
        var t = stripHtmlTags(blk)
        t = t.replace(re"\[\d+\]", "")
        if t.len > 60: paragraphs.add(t)
      let joined = paragraphs.join(" ")
      if joined.len > 200: result.add(joined)
    except CatchableError:
      discard
    sleep(1000)

proc loadAllTexts*(cfg: Config): seq[string] =
  ## Chỉ load Dolly — các nguồn khác (grammar/pgn/stockfish/english_words/
  ## wikipedia/stackoverflow) vẫn còn định nghĩa ở trên, chỉ tạm không gọi.
  ## Muốn bật lại nguồn nào, thêm dòng result.add load...(...) tương ứng.
  result = @[]
  echo "  -> databricks-dolly-15k.jsonl ..."
  result.add loadDolly(cfg.dataDir / "databricks-dolly-15k.jsonl")

# ═══════════════════════════════════════════════════════════════
# Xây sample huấn luyện (thay cho build_dataset/batch_generator bản Python)
# ═══════════════════════════════════════════════════════════════

type Sample = tuple[x, y: seq[int]]

proc buildSamples*(texts: seq[string], tok: CharTokenizer, seqLen: int): seq[Sample] =
  result = @[]
  for t in texts:
    if t.len < seqLen + 1: continue
    let ids = tok.encode(t)
    var i = 0
    while i + seqLen + 1 <= ids.len:
      result.add((ids[i ..< i + seqLen], ids[i + 1 ..< i + seqLen + 1]))
      i += seqLen

# ═══════════════════════════════════════════════════════════════
# Loss — batched thật: logits shape [B, T, vocab], targetsBatch: B chuỗi.
# Trung bình theo B*T (khớp cách chia trung bình cũ khi B=1: chia theo T).
# ═══════════════════════════════════════════════════════════════

proc crossEntropyLossBatch*(logits: Tensor, targetsBatch: seq[seq[int]]): tuple[loss: float32, dLogits: Tensor] =
  let B = logits.shape[0]
  let T = logits.shape[1]
  let vocab = logits.shape[2]
  var loss = 0'f32
  var dLogits = newTensor(logits.shape)
  let denom = float32(B * T)
  for b in 0 ..< B:
    let baseB = b * T * vocab
    for t in 0 ..< T:
      let off = baseB + t * vocab
      let target = targetsBatch[b][t]
      let maxVal = logits.data[off ..< off + vocab].max()
      var sumExp = 0'f32
      for i in 0 ..< vocab:
        sumExp += exp(logits.data[off + i] - maxVal)
      let prob = exp(logits.data[off + target] - maxVal) / sumExp
      loss += -ln(max(prob, 1e-12'f32))
      for i in 0 ..< vocab:
        dLogits.data[off + i] = exp(logits.data[off + i] - maxVal) / sumExp
      dLogits.data[off + target] -= 1.0
  loss /= denom
  for i in 0 ..< dLogits.data.len:
    dLogits.data[i] /= denom
  result = (loss, dLogits)


# ═══════════════════════════════════════════════════════════════
# Quản lý tham số: gom thành 1 danh sách CÙNG THỨ TỰ mà
# NimformerModel.backward() trả gradient về (outProj trước, rồi từng
# block THEO CHIỀU NGƯỢC, cuối cùng là embedding) — xem nimformer.nim.
# ═══════════════════════════════════════════════════════════════

template forEachParam(model: NimformerModel, op: untyped) =
  ## Duyệt (tên, Tensor) theo đúng thứ tự grads trả về từ model.backward().
  ## `op` là 1 template/proc nhận (name: string, t: Tensor).
  op("outProj.weight", model.outProj.weight)
  op("outProj.bias", model.outProj.bias)
  for bi {.inject.} in countdown(model.blocks.len - 1, 0):
    let blk {.inject.} = model.blocks[bi]
    op(&"blocks.{bi}.attn.qkv.weight", blk.attn.qkv.weight)
    op(&"blocks.{bi}.attn.qkv.bias", blk.attn.qkv.bias)
    op(&"blocks.{bi}.attn.proj.weight", blk.attn.proj.weight)
    op(&"blocks.{bi}.attn.proj.bias", blk.attn.proj.bias)
    op(&"blocks.{bi}.ff.fc1.weight", blk.ff.fc1.weight)
    op(&"blocks.{bi}.ff.fc1.bias", blk.ff.fc1.bias)
    op(&"blocks.{bi}.ff.fc2.weight", blk.ff.fc2.weight)
    op(&"blocks.{bi}.ff.fc2.bias", blk.ff.fc2.bias)
    op(&"blocks.{bi}.ln1.gamma", blk.ln1.gamma)
    op(&"blocks.{bi}.ln1.beta", blk.ln1.beta)
    op(&"blocks.{bi}.ln2.gamma", blk.ln2.gamma)
    op(&"blocks.{bi}.ln2.beta", blk.ln2.beta)
  op("embed.weight", model.embed.weight)

proc paramLens(model: NimformerModel): seq[int] =
  result = @[]
  template rec(name: string, t: Tensor) = result.add(t.data.len)
  forEachParam(model, rec)

proc initOptStates(model: NimformerModel): seq[ApfAdamState] =
  result = @[]
  for l in paramLens(model): result.add newApfAdamState(l)

proc applyGrads(model: var NimformerModel, grads: seq[Tensor],
                 states: var seq[ApfAdamState], lr: float32, requantizeEvery: int) =
  var idx = 0
  template step(param: untyped) =
    discard apfAdamStep(param, grads[idx], states[idx], lr,
                         requantizeEvery = requantizeEvery)
    inc idx
  step(model.outProj.weight)
  step(model.outProj.bias)
  for bi in countdown(model.blocks.len - 1, 0):
    step(model.blocks[bi].attn.qkv.weight)
    step(model.blocks[bi].attn.qkv.bias)
    step(model.blocks[bi].attn.proj.weight)
    step(model.blocks[bi].attn.proj.bias)
    step(model.blocks[bi].ff.fc1.weight)
    step(model.blocks[bi].ff.fc1.bias)
    step(model.blocks[bi].ff.fc2.weight)
    step(model.blocks[bi].ff.fc2.bias)
    step(model.blocks[bi].ln1.gamma)
    step(model.blocks[bi].ln1.beta)
    step(model.blocks[bi].ln2.gamma)
    step(model.blocks[bi].ln2.beta)
  step(model.embed.weight)

# ═══════════════════════════════════════════════════════════════
# Lưu / tải checkpoint .nimq — chọn kiểu lượng tử hoá cho TRỌNG SỐ (weight
# ma trận Linear/Embedding); bias và LayerNorm (gamma/beta) LUÔN giữ fp32
# (đúng quy ước ghi trong README_BUILD.md).
# ═══════════════════════════════════════════════════════════════

proc isBiasOrNorm(name: string): bool =
  name.endsWith(".bias") or name.endsWith(".gamma") or name.endsWith(".beta")

proc saveNimformerModel*(model: NimformerModel, path: string, weightKind: QuantKind,
                          embedDim, nHeads, nLayers, ffMult: int) =
  var sd: seq[(string, QuantTensor)] = @[]
  template rec(name: string, t: Tensor) =
    let kind = if isBiasOrNorm(name): qkFp32Raw else: weightKind
    sd.add (name, quantizeTensor(t.data, t.shape, kind))
  forEachParam(model, rec)
  saveQuantStateDict(path, [model.vocab, embedDim, nHeads, nLayers, ffMult], sd)
  echo &"  đã lưu {sd.len} tensor -> {path} (weight={weightKind}, bias/LN=qkFp32Raw)"

proc loadNimformerModel*(path: string): NimformerModel =
  let (arch, sd) = loadQuantStateDict(path)
  let vocab = arch[0]; let embedDim = arch[1]; let nHeads = arch[2]
  let nLayers = arch[3]; let ffMult = arch[4]
  result = newNimformerModel(vocab, embedDim, nHeads, nLayers, ffMult)
  var byName = initTable[string, QuantTensor]()
  for (name, qt) in sd: byName[name] = qt
  template load(name: string, t: var Tensor) =
    if byName.hasKey(name):
      let qt = byName[name]
      t.data = dequantizeTensor(qt)
      t.shape = qt.shape
  load("outProj.weight", result.outProj.weight)
  load("outProj.bias", result.outProj.bias)
  for bi in 0 ..< result.blocks.len:
    load(&"blocks.{bi}.attn.qkv.weight", result.blocks[bi].attn.qkv.weight)
    load(&"blocks.{bi}.attn.qkv.bias", result.blocks[bi].attn.qkv.bias)
    load(&"blocks.{bi}.attn.proj.weight", result.blocks[bi].attn.proj.weight)
    load(&"blocks.{bi}.attn.proj.bias", result.blocks[bi].attn.proj.bias)
    load(&"blocks.{bi}.ff.fc1.weight", result.blocks[bi].ff.fc1.weight)
    load(&"blocks.{bi}.ff.fc1.bias", result.blocks[bi].ff.fc1.bias)
    load(&"blocks.{bi}.ff.fc2.weight", result.blocks[bi].ff.fc2.weight)
    load(&"blocks.{bi}.ff.fc2.bias", result.blocks[bi].ff.fc2.bias)
    load(&"blocks.{bi}.ln1.gamma", result.blocks[bi].ln1.gamma)
    load(&"blocks.{bi}.ln1.beta", result.blocks[bi].ln1.beta)
    load(&"blocks.{bi}.ln2.gamma", result.blocks[bi].ln2.gamma)
    load(&"blocks.{bi}.ln2.beta", result.blocks[bi].ln2.beta)
  load("embed.weight", result.embed.weight)

# ═══════════════════════════════════════════════════════════════
# Checkpoint CÓ THỂ RESUME — khác với saveNimformerModel/loadNimformerModel
# ở trên (chỉ lưu weight, dùng cho bản "export" cuối cùng để suy luận),
# 2 hàm dưới đây lưu THÊM: trạng thái optimizer (m/v của ApfAdam cho từng
# tham số) và số step đã train, để lần chạy sau load lên là train tiếp
# đúng chỗ (không mất đà "momentum", không bị reset bias-correction về
# step=1 làm loss nhảy vọt lúc mới resume).
#
# Đúng tinh thần "cả thư viện dùng APF custom": m/v của optimizer cũng
# được nén bằng qkAuto (APF — buildCustomDtypeForTensor trong
# customfloat.nim, quant.nim gọi lại) thay vì giữ float32 thô — checkpoint
# tự chọn số bit exponent/mantissa phù hợp với chính phân bố m/v tại thời
# điểm lưu, y hệt cách weight đang được nén. Khác biệt duy nhất: weight có
# thể chọn int8/int4/fp8/... theo --quant, còn m/v LUÔN dùng qkAuto vì cần
# bám sát phân bố thực mỗi lần lưu hơn là ép theo 1 dtype cố định.
# ═══════════════════════════════════════════════════════════════

proc saveCheckpoint*(model: NimformerModel, states: seq[ApfAdamState], stepNo: int,
                      path: string, weightKind: QuantKind,
                      embedDim, nHeads, nLayers, ffMult: int) =
  var sd: seq[(string, QuantTensor)] = @[]
  var idx = 0
  template rec(name: string, t: Tensor) =
    let kind = if isBiasOrNorm(name): qkFp32Raw else: weightKind
    sd.add (name, quantizeTensor(t.data, t.shape, kind))
    # optimizer state (m/v) — luôn APF custom (qkAuto), không phụ thuộc --quant
    sd.add (name & ".opt_m", quantizeTensor(states[idx].m, @[states[idx].m.len], qkAuto))
    sd.add (name & ".opt_v", quantizeTensor(states[idx].v, @[states[idx].v.len], qkAuto))
    inc idx
  forEachParam(model, rec)
  # applyGrads gọi apfAdamStep đúng 1 lần/tham số/bước train nên state.step
  # của MỌI tham số đều bằng nhau và bằng stepNo -> chỉ cần lưu 1 số dùng chung.
  sd.add ("__step__", quantizeTensor(@[float32(stepNo)], @[1], qkFp32Raw))
  saveQuantStateDict(path, [model.vocab, embedDim, nHeads, nLayers, ffMult], sd)
  echo &"  đã lưu checkpoint (weight+optimizer, step={stepNo}) -> {path} " &
      &"(weight={weightKind}, optimizer m/v=qkAuto/APF, bias/LN=qkFp32Raw)"
  if weightKind == qkAuto:
    # In vài ví dụ dtype THẬT mà APF chọn (đọc lại từ chính sd vừa build) để
    # thấy rõ nó không phải fp32 "trá hình" — mỗi tensor có thể ra 1 dtype
    # khác nhau tuỳ range dữ liệu (exponent bit adaptive theo range, mantissa
    # bit ~cố định theo APF_DEFAULT_REL_ERROR_TOL trừ khi có gradient).
    var shown = 0
    for (name, qt) in sd:
      if qt.kind == qkAuto and shown < 3:
        echo &"    [APF] {name}: {qt.cf.name} (e{qt.cf.exponentBits}m{qt.cf.mantissaBits}, " &
            &"{qt.cf.totalBits} bit/phần tử, so với fp32=32 bit)"
        inc shown

proc loadCheckpointFull*(path: string): tuple[model: NimformerModel, states: seq[ApfAdamState], stepNo: int] =
  let (arch, sd) = loadQuantStateDict(path)
  let vocab = arch[0]; let embedDim = arch[1]; let nHeads = arch[2]
  let nLayers = arch[3]; let ffMult = arch[4]
  var model = newNimformerModel(vocab, embedDim, nHeads, nLayers, ffMult)
  var byName = initTable[string, QuantTensor]()
  for (name, qt) in sd: byName[name] = qt

  var stepNo = 0
  if byName.hasKey("__step__"):
    let arr = dequantizeTensor(byName["__step__"])
    if arr.len > 0: stepNo = int(round(arr[0]))

  template load(name: string, t: var Tensor) =
    if byName.hasKey(name):
      let qt = byName[name]
      t.data = dequantizeTensor(qt)
      t.shape = qt.shape

  var states: seq[ApfAdamState] = @[]
  template loadWithOpt(name: string, t: var Tensor) =
    load(name, t)
    var st = newApfAdamState(t.data.len)
    st.step = stepNo   # đồng bộ lại bias-correction Adam đúng chỗ đã dừng
    if byName.hasKey(name & ".opt_m"): st.m = dequantizeTensor(byName[name & ".opt_m"])
    if byName.hasKey(name & ".opt_v"): st.v = dequantizeTensor(byName[name & ".opt_v"])
    states.add st

  # Thứ tự PHẢI khớp forEachParam/applyGrads (outProj -> block ngược -> embed)
  loadWithOpt("outProj.weight", model.outProj.weight)
  loadWithOpt("outProj.bias", model.outProj.bias)
  for bi in countdown(model.blocks.len - 1, 0):
    loadWithOpt(&"blocks.{bi}.attn.qkv.weight", model.blocks[bi].attn.qkv.weight)
    loadWithOpt(&"blocks.{bi}.attn.qkv.bias", model.blocks[bi].attn.qkv.bias)
    loadWithOpt(&"blocks.{bi}.attn.proj.weight", model.blocks[bi].attn.proj.weight)
    loadWithOpt(&"blocks.{bi}.attn.proj.bias", model.blocks[bi].attn.proj.bias)
    loadWithOpt(&"blocks.{bi}.ff.fc1.weight", model.blocks[bi].ff.fc1.weight)
    loadWithOpt(&"blocks.{bi}.ff.fc1.bias", model.blocks[bi].ff.fc1.bias)
    loadWithOpt(&"blocks.{bi}.ff.fc2.weight", model.blocks[bi].ff.fc2.weight)
    loadWithOpt(&"blocks.{bi}.ff.fc2.bias", model.blocks[bi].ff.fc2.bias)
    loadWithOpt(&"blocks.{bi}.ln1.gamma", model.blocks[bi].ln1.gamma)
    loadWithOpt(&"blocks.{bi}.ln1.beta", model.blocks[bi].ln1.beta)
    loadWithOpt(&"blocks.{bi}.ln2.gamma", model.blocks[bi].ln2.gamma)
    loadWithOpt(&"blocks.{bi}.ln2.beta", model.blocks[bi].ln2.beta)
  loadWithOpt("embed.weight", model.embed.weight)

  result = (model, states, stepNo)

# ═══════════════════════════════════════════════════════════════
# Training loop chính — thay cho model.train_streaming(...) bên Python.
#
# TRƯỚC: --batch=256 chỉ là 1 vòng for Nim gọi forward/backward B=1 TUẦN TỰ
# 256 lần rồi cộng dồn gradient trên CPU — mỗi lần lại tự upload/dispatch/
# wait/download GPU, nên phần lớn thời gian là round-trip CPU<->GPU chờ
# nhau, không phải tính toán thật -> CPU lẫn GPU đều "rảnh" theo Activity
# Monitor dù batch để rất lớn.
#
# GIỜ: gom cả --batch chuỗi thành 1 idsBatch thật, gọi forwardBatch/
# backwardBatch ĐÚNG MỘT LẦN cho cả batch — Linear/LayerNorm coi B*T là số
# hàng nên mỗi matmul GPU nhận hẳn M=B*T dòng (to hơn hẳn, ít round-trip hẳn
# --batch lần), và Linear.backward tự cộng dồn gradient qua cả batch trong
# chính phép matmul đó (không cần cộng dồn tay trên CPU nữa).
# ═══════════════════════════════════════════════════════════════

proc train(model: var NimformerModel, samples: seq[Sample], ctx: MetalContext,
           cfg: Config, states: var seq[ApfAdamState], startStep: int = 0) =
  ## startStep > 0 khi resume từ checkpoint (xem loadCheckpointFull trong main) —
  ## vòng lặp tiếp tục đúng từ đó tới cfg.steps thay vì train lại từ 0.
  if startStep >= cfg.steps:
    echo &"  checkpoint đã ở step {startStep} >= --steps={cfg.steps}, không train thêm."
    return
  var order = toSeq(0 ..< samples.len)
  var stepNo = startStep
  while stepNo < cfg.steps:
    shuffle(order)
    var pos = 0
    while pos < order.len and stepNo < cfg.steps:
      let chunkEnd = min(pos + cfg.batchSize, order.len)
      var idsBatch: seq[seq[int]] = @[]
      var targetsBatch: seq[seq[int]] = @[]
      for k in pos ..< chunkEnd:
        let (x, y) = samples[order[k]]
        idsBatch.add(x)
        targetsBatch.add(y)
      pos = chunkEnd
      if idsBatch.len == 0: continue

      let logits = model.forwardBatch(idsBatch, ctx)                    # [B,T,vocab] — 1 lần forward cho CẢ batch
      let (loss, dLogits) = crossEntropyLossBatch(logits, targetsBatch)
      let grads = model.backwardBatch(idsBatch, dLogits, ctx)           # đã tự cộng dồn gradient qua cả batch
      applyGrads(model, grads, states, cfg.lr, cfg.requantizeEvery)

      inc stepNo
      if stepNo mod cfg.logEvery == 0 or stepNo == 1:
        echo &"[step {stepNo}/{cfg.steps}] loss={loss:.6f} (batch={idsBatch.len})"
      if cfg.requantizeEvery > 0 and stepNo mod cfg.requantizeEvery == 0:
        # applyGrads() vừa gọi apfAdamStep() cho MỌI tham số ở bước này, và vì
        # requantizeEvery giống nhau cho tất cả nên outProj.weight cũng vừa
        # bị APF requantize xong — build lại (cùng data, cùng công thức) chỉ
        # để IN RA cho thấy dtype thật đang dùng, không phải để tính lại gì.
        let exCf = buildCustomDtypeForTensor(model.outProj.weight.data)
        echo &"  [APF] step {stepNo}: vừa requantize toàn bộ weight " &
            &"(ví dụ outProj.weight -> {exCf.name}, e{exCf.exponentBits}m{exCf.mantissaBits}, " &
            &"{exCf.totalBits} bit/phần tử — so với fp32=32 bit)"
      if cfg.ckptEvery > 0 and stepNo mod cfg.ckptEvery == 0:
        saveCheckpoint(model, states, stepNo, cfg.savePath & ".ckpt", quantKindOf(cfg.quant),
                        cfg.embedDim, cfg.nHeads, cfg.nLayers, cfg.ffMult)

# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

proc main() =
  let cfg = parseArgs()
  randomize(cfg.seed)

  echo "== Load training texts =="
  let texts = loadAllTexts(cfg)
  echo &"TOTAL TEXTS: {texts.len}"
  if texts.len == 0:
    stderr.writeLine "Không load được dữ liệu nào — kiểm tra lại --data-dir / mạng / --stockfish-path."
    quit(1)

  # ─────────────────────────────────────────────────────────────
  # Resume: nếu đã có checkpoint (--save & ".ckpt") từ lần chạy trước, load
  # lại model + optimizer (m/v) + step thay vì train lại từ đầu. Tokenizer
  # cũng load lại từ --tokenizer (nếu có) để giữ đúng mapping id<->byte cũ,
  # thay vì build lại (có thể lệch nếu texts nạp vào khác thứ tự/tập hợp).
  # ─────────────────────────────────────────────────────────────
  let ckptPath = cfg.savePath & ".ckpt"
  var tok: CharTokenizer
  var model: NimformerModel
  var states: seq[ApfAdamState]
  var startStep = 0
  let resuming = fileExists(ckptPath)

  if resuming:
    echo &"== Tìm thấy checkpoint {ckptPath} -> resume thay vì train lại từ đầu =="
    if fileExists(cfg.tokenizerPath):
      tok = loadTokenizer(cfg.tokenizerPath)
      echo &"  đã tải lại tokenizer cũ -> {cfg.tokenizerPath} (vocab={tok.vocabSize})"
    else:
      stderr.writeLine &"  [cảnh báo] không thấy {cfg.tokenizerPath}, build tokenizer mới từ texts " &
          "(có thể lệch id với checkpoint nếu tập ký tự khác lần trước)"
      tok = newCharTokenizer(texts)
      tok.saveTokenizer(cfg.tokenizerPath)
    let loaded = loadCheckpointFull(ckptPath)
    model = loaded.model
    states = loaded.states
    startStep = loaded.stepNo
    echo &"  resume từ step {startStep} " &
        "(đã nạp lại weight + optimizer m/v nén APF/qkAuto + step)"
  else:
    echo "== Build tokenizer =="
    tok = newCharTokenizer(texts)
    tok.saveTokenizer(cfg.tokenizerPath)
    echo &"Vocab size: {tok.vocabSize}  (đã lưu -> {cfg.tokenizerPath})"

  echo &"== Build training samples (seq_len={cfg.seqLen}) =="
  let samples = buildSamples(texts, tok, cfg.seqLen)
  echo &"Total samples: {samples.len}"
  if samples.len == 0:
    stderr.writeLine "Không đủ dữ liệu để tạo sample (texts ngắn hơn seq_len). Giảm --seq hoặc thêm dữ liệu."
    quit(1)

  echo "== Init Metal GPU context =="
  let ctx = newMetalContext()

  if not resuming:
    echo "== Build model =="
    model = newNimformerModel(vocab = tok.vocabSize, embedDim = cfg.embedDim,
                               nHeads = cfg.nHeads, nLayers = cfg.nLayers,
                               ffMult = cfg.ffMult)
    states = initOptStates(model)
  echo &"  vocab={model.vocab} embedDim={cfg.embedDim} nHeads={cfg.nHeads} " &
      &"nLayers={cfg.nLayers} ffMult={cfg.ffMult}  ({states.len} tensor tham số)"

  echo &"== Training (bắt đầu từ step {startStep}/{cfg.steps}) =="
  let t0 = epochTime()
  train(model, samples, ctx, cfg, states, startStep)
  echo &"== Done in {epochTime() - t0:.1f}s =="

  echo &"== Save checkpoint cuối ({cfg.savePath}, quant={cfg.quant}) =="
  saveNimformerModel(model, cfg.savePath, quantKindOf(cfg.quant),
                      cfg.embedDim, cfg.nHeads, cfg.nLayers, cfg.ffMult)

when isMainModule:
  main()