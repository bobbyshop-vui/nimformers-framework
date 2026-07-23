## test_nimformer.nim - Training với Linear++ Attention + Session API
## Build: nim c -d:release -o:train_linearpp test_nimformer.nim
## Chạy: ./train_linearpp --backend=metal --steps=1000 --seq=128 --batch=32

import std/[os, math, random, strformat, strutils, sequtils, tables, json,
            times, parseopt, osproc, streams, re, httpclient]
import quant, nimformer, backend, customfloat

# ═══════════════════════════════════════════════════════════════
# IMPORT CÁC BACKEND CÓ SẴN TRONG vendor/bybylang/backends/
# ═══════════════════════════════════════════════════════════════

when defined(macosx):
  import vendor/bybylang/backends/metal/metal_backend as metal
import vendor/bybylang/backends/cuda/cuda_driver
import vendor/bybylang/backends/opencl/opencl_api
import vendor/bybylang/gpubackend

# ═══════════════════════════════════════════════════════════════
# SESSION API CHUNG - dùng thẳng SessionHandle + session* từ gpubackend.nim
# (không định nghĩa lại ở đây để tránh trùng/lệch với bản gốc, vốn đã hỗ
#  trợ đủ Metal/CUDA/OpenCL/TSIC).
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# CPU IMPLEMENTATION CHO LINEAR++ ATTENTION
# ═══════════════════════════════════════════════════════════════

proc cpuLinearPlusForward(qData, kData, vData: seq[float32], B, H, T, D: int, scale: float32): seq[float32] =
  ## qData/kData/vData layout is [B, T, H*D] (heads interleaved per timestep,
  ## matching how forwardLinearPlus slices them out of the fused QKV
  ## projection). Output is written back in that SAME [B, T, H*D] layout so
  ## it can be dropped straight into a Tensor of shape [B,T,C] afterwards.
  ##
  ## KV = K^T @ V is a proper [D, D] matrix (contracted over T), and
  ## O = Q @ KV is [T, D]. This is O(T*D^2), linear in sequence length T --
  ## unlike a naive T x T attention matrix. The previous version of this
  ## function built a "kv" array sized [T, D] and indexed it with a T-ranged
  ## loop variable used AS a D-index, which only produced correct results by
  ## coincidence when T == D and silently read wrong memory otherwise.
  let C = H * D
  result = newSeq[float32](B * T * C)
  let norm = 1'f32 / sqrt(float32(T))
  template idx(b, t, h, d: int): int = b * T * C + t * C + h * D + d
  for b in 0 ..< B:
    for h in 0 ..< H:
      # KV[i,j] = sum_t K[t,i] * V[t,j]  -> [D, D]
      var kv = newSeq[float32](D * D)
      for t in 0 ..< T:
        for i in 0 ..< D:
          let kt = kData[idx(b, t, h, i)]
          if kt == 0'f32: continue
          for j in 0 ..< D:
            kv[i * D + j] += kt * vData[idx(b, t, h, j)]
      # O[t,j] = sum_i Q[t,i] * KV[i,j]
      for t in 0 ..< T:
        for j in 0 ..< D:
          var acc = 0'f32
          for i in 0 ..< D:
            acc += qData[idx(b, t, h, i)] * kv[i * D + j]
          result[idx(b, t, h, j)] = acc * scale * norm

proc cpuLinearPlusBackward(qData, kData, vData, dOutData: seq[float32], B, H, T, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  ## Same [B, T, H*D] layout and same D×D KV convention as cpuLinearPlusForward.
  let C = H * D
  result.dq = newSeq[float32](B * T * C)
  result.dk = newSeq[float32](B * T * C)
  result.dv = newSeq[float32](B * T * C)
  let norm = 1'f32 / sqrt(float32(T))
  template idx(b, t, h, d: int): int = b * T * C + t * C + h * D + d

  for b in 0 ..< B:
    for h in 0 ..< H:
      # Recompute KV[i,j] = sum_t K[t,i] * V[t,j]
      var kv = newSeq[float32](D * D)
      for t in 0 ..< T:
        for i in 0 ..< D:
          let kt = kData[idx(b, t, h, i)]
          if kt == 0'f32: continue
          for j in 0 ..< D:
            kv[i * D + j] += kt * vData[idx(b, t, h, j)]

      # dKV[i,j] = sum_t Q[t,i] * (dOut[t,j] * scale * norm)   (since O = Q @ KV)
      var dkv = newSeq[float32](D * D)
      for t in 0 ..< T:
        for j in 0 ..< D:
          let g = dOutData[idx(b, t, h, j)] * scale * norm
          if g == 0'f32: continue
          for i in 0 ..< D:
            dkv[i * D + j] += qData[idx(b, t, h, i)] * g

      # dQ[t,i] = sum_j (dOut[t,j] * scale * norm) * KV[i,j]
      for t in 0 ..< T:
        for i in 0 ..< D:
          var acc = 0'f32
          for j in 0 ..< D:
            acc += dOutData[idx(b, t, h, j)] * scale * norm * kv[i * D + j]
          result.dq[idx(b, t, h, i)] += acc

      # KV = K^T @ V  ->  dK = V @ dKV^T ,  dV = K @ dKV
      for t in 0 ..< T:
        for i in 0 ..< D:
          var accK = 0'f32
          for j in 0 ..< D:
            accK += dkv[i * D + j] * vData[idx(b, t, h, j)]
          result.dk[idx(b, t, h, i)] += accK
        for j in 0 ..< D:
          var accV = 0'f32
          for i in 0 ..< D:
            accV += kData[idx(b, t, h, i)] * dkv[i * D + j]
          result.dv[idx(b, t, h, j)] += accV

# ═══════════════════════════════════════════════════════════════
# LINEAR++ ATTENTION - DÙNG SESSION API
# ═══════════════════════════════════════════════════════════════

type
  LinearPlusAttention* = object
    nHeads*, headDim*, embedDim*: int
    qkv*, proj*: Linear
    useSession*: bool

proc newLinearPlusAttention*(embedDim, nHeads: int, useSession: bool = true): LinearPlusAttention =
  assert embedDim mod nHeads == 0
  result.embedDim = embedDim
  result.nHeads = nHeads
  result.headDim = embedDim div nHeads
  result.qkv = newLinear(embedDim, 3 * embedDim)
  result.proj = newLinear(embedDim, embedDim)
  result.useSession = useSession
proc forwardLinearPlus*(attn: LinearPlusAttention, x: Tensor, ctx: Backend): Tensor =
  let B = x.shape[0]
  let T = x.shape[1]
  let C = x.shape[2]
  assert C == attn.embedDim

  let qkv = attn.qkv.forward(x, ctx)
  let hd = attn.headDim
  let H = attn.nHeads
  let scale = 1'f32 / sqrt(float32(hd))

  # === FIX: KHÔNG GÁN qkv.data, DÙNG BIẾN TẠM ===
  var qkvData = qkv.data
  let expectedQkvSize = B * T * 3 * C
  if qkvData.len != expectedQkvSize:
    stderr.writeLine "[WARN] forwardLinearPlus: qkv.data.len=", qkvData.len, " expected=", expectedQkvSize
    var fixedQkv = newSeq[float32](expectedQkvSize)
    let copyLen = min(qkvData.len, expectedQkvSize)
    for i in 0 ..< copyLen:
      fixedQkv[i] = qkvData[i]
    qkvData = fixedQkv

  # Slice Q, K, V
  var qData = newSeq[float32](B * T * C)
  var kData = newSeq[float32](B * T * C)
  var vData = newSeq[float32](B * T * C)

  for b in 0 ..< B:
    let baseQKV = b * T * 3 * C
    let baseOut = b * T * C
    for t in 0 ..< T:
      for c in 0 ..< C:
        qData[baseOut + t * C + c] = qkvData[baseQKV + t * 3 * C + c]
        kData[baseOut + t * C + c] = qkvData[baseQKV + t * 3 * C + C + c]
        vData[baseOut + t * C + c] = qkvData[baseQKV + t * 3 * C + 2 * C + c]

  var oData: seq[float32]
  let norm = 1'f32 / sqrt(float32(T))

  if attn.useSession and ctx.kind in {bkMetal, bkCuda, bkOpenCL}:
    let backend = ctx.kind.toByby()
    oData = newSeq[float32](B * T * C)
    var sessionOk = sessionBegin(backend)

    if sessionOk:
      # Một cặp buffer (KV, O) cho MỖI (batch, head), nhưng TẤT CẢ các lệnh
      # matmul được encode vào CÙNG 1 session -> chỉ commit+waitUntilCompleted
      # MỘT LẦN ở sessionEnd() bên dưới, thay vì B*H*2 lần dispatch rời rạc.
      # Đây chính là điểm của session API (giảm overhead command-buffer trên
      # iGPU yếu), phần trước đây bị bỏ trống bằng 3 dòng TODO.
      var kvHandles: seq[SessionHandle] = @[]
      var oHandles: seq[SessionHandle] = @[]

      block encodeLoop:
        for b in 0 ..< B:
          for h in 0 ..< H:
            # Gom Q, V liên tục [T,D] và K CHUYỂN VỊ [D,T] (matmul không có
            # tham số transpose, nên phải tự chuyển vị trước khi upload).
            var qSlice = newSeq[float32](T * hd)
            var kSliceT = newSeq[float32](hd * T)
            var vSlice = newSeq[float32](T * hd)
            for t in 0 ..< T:
              let srcOff = b * T * C + t * C + h * hd
              for d in 0 ..< hd:
                qSlice[t * hd + d] = qData[srcOff + d]
                kSliceT[d * T + t] = kData[srcOff + d]
                vSlice[t * hd + d] = vData[srcOff + d]

            var hQ = sessionUpload(backend, qSlice)
            var hKT = sessionUpload(backend, kSliceT)
            var hV = sessionUpload(backend, vSlice)
            var hKV = sessionAllocScratch(backend, hd * hd)
            var hO = sessionAllocScratch(backend, T * hd)

            # KV[D,D] = K^T[D,T] @ V[T,D]
            if not sessionMatmul(backend, hKT, hV, hKV, hd, T, hd):
              sessionOk = false
            # O[T,D] = Q[T,D] @ KV[D,D]
            elif not sessionMatmul(backend, hQ, hKV, hO, T, hd, hd):
              sessionOk = false

            sessionFree(backend, hQ)
            sessionFree(backend, hKT)
            sessionFree(backend, hV)
            kvHandles.add(hKV)
            oHandles.add(hO)
            if not sessionOk: break encodeLoop

      if sessionOk and sessionEnd(backend):
        var hi = 0
        for b in 0 ..< B:
          for h in 0 ..< H:
            let raw = sessionRead(backend, oHandles[hi], T * hd)
            for t in 0 ..< T:
              let dstOff = b * T * C + t * C + h * hd
              for d in 0 ..< hd:
                oData[dstOff + d] = raw[t * hd + d] * scale * norm
            inc hi
      else:
        oData = cpuLinearPlusForward(qData, kData, vData, B, H, T, hd, scale)

      for h in kvHandles.mitems: sessionFree(backend, h)
      for h in oHandles.mitems: sessionFree(backend, h)
    else:
      oData = cpuLinearPlusForward(qData, kData, vData, B, H, T, hd, scale)
  else:
    oData = cpuLinearPlusForward(qData, kData, vData, B, H, T, hd, scale)

  var o = newTensor(@[B, T, C])
  o.data = oData
  result = attn.proj.forward(o, ctx)
proc backwardLinearPlus*(attn: LinearPlusAttention, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dQkvW, dQkvB, dProjW, dProjB: Tensor] =

  let B = x.shape[0]
  let T = x.shape[1]
  let C = x.shape[2]
  let hd = attn.headDim
  let H = attn.nHeads
  let scale = 1'f32 / sqrt(float32(hd))

  let qkv = attn.qkv.forward(x, ctx)
  
  # === FIX: KHÔNG GÁN qkv.data, DÙNG BIẾN TẠM ===
  var qkvData = qkv.data
  let expectedQkvSize = B * T * 3 * C
  if qkvData.len != expectedQkvSize:
    stderr.writeLine "[WARN] backwardLinearPlus: qkv.data.len=", qkvData.len, " expected=", expectedQkvSize
    var fixedQkv = newSeq[float32](expectedQkvSize)
    let copyLen = min(qkvData.len, expectedQkvSize)
    for i in 0 ..< copyLen:
      fixedQkv[i] = qkvData[i]
    qkvData = fixedQkv

  var qData = newSeq[float32](B * T * C)
  var kData = newSeq[float32](B * T * C)
  var vData = newSeq[float32](B * T * C)

  for b in 0 ..< B:
    let baseQKV = b * T * 3 * C
    let baseOut = b * T * C
    for t in 0 ..< T:
      for c in 0 ..< C:
        qData[baseOut + t * C + c] = qkvData[baseQKV + t * 3 * C + c]
        kData[baseOut + t * C + c] = qkvData[baseQKV + t * 3 * C + C + c]
        vData[baseOut + t * C + c] = qkvData[baseQKV + t * 3 * C + 2 * C + c]

  # Forward để lấy output
  let oData = cpuLinearPlusForward(qData, kData, vData, B, H, T, hd, scale)
  var attnOut = newTensor(@[B, T, C])
  attnOut.data = oData

  # dProj
  let dProj = attn.proj.backward(attnOut, dOut, ctx)
  let dAttnOut = dProj.dX

  # Backward Linear++
  let (dq, dk, dv) = cpuLinearPlusBackward(qData, kData, vData, dAttnOut.data, B, H, T, hd, scale)

  var dQkv = newTensor(qkv.shape)
  for b in 0 ..< B:
    let baseQKV = b * T * 3 * C
    let baseOut = b * T * C
    for t in 0 ..< T:
      for c in 0 ..< C:
        dQkv.data[baseQKV + t * 3 * C + c] = dq[baseOut + t * C + c]
        dQkv.data[baseQKV + t * 3 * C + C + c] = dk[baseOut + t * C + c]
        dQkv.data[baseQKV + t * 3 * C + 2 * C + c] = dv[baseOut + t * C + c]

  let dQkvLin = attn.qkv.backward(x, dQkv, ctx)
  result = (dQkvLin.dX, dQkvLin.dW, dQkvLin.dB, dProj.dW, dProj.dB)
# ═══════════════════════════════════════════════════════════════
# CONFIG - THAY CHO HẰNG SỐ CỨNG
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
    requantizeEvery: int
    savePath: string
    tokenizerPath: string
    dataDir: string
    quant: QuantChoice
    ckptEvery: int
    logEvery: int
    stockfishPath: string
    stockfishGames: int
    stockfishPlies: int
    soTags: seq[string]
    soMaxPages: int
    wikiMaxPages: int
    seed: int
    backend: string
    useSession: bool

proc parseQuant(s: string): QuantChoice =
  case s.toLowerAscii
  of "int8": qcInt8
  of "int4": qcInt4
  of "fp8_e4m3", "fp8e4m3", "fp8": qcFp8E4M3
  of "fp8_e5m2", "fp8e5m2": qcFp8E5M2
  of "auto", "apf": qcAuto
  of "none", "fp32", "raw": qcNone
  else:
    stderr.writeLine &"[cảnh báo] --quant='{s}' không nhận diện được, dùng mặc định 'auto'"
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
    steps: 1581,
    lr: 3e-3'f32,
    embedDim: 128,
    nHeads: 4,
    nLayers: 4,
    ffMult: 4,
    requantizeEvery: 50,
    savePath: "finetune.nimq",
    tokenizerPath: "tokenizer-testmodel.json",
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
    seed: 1337,
    backend: "metal",
    useSession: true
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
      of "seed":              result.seed = parseInt(val)
      of "backend":           result.backend = val
      of "no-session":        result.useSession = false
      of "help", "h":
        echo "Usage: ./train_linearpp [options]"
        echo "Options:"
        echo "  --seq=128            Sequence length"
        echo "  --batch=32           Batch size"
        echo "  --steps=10000        Training steps"
        echo "  --lr=3e-3            Learning rate"
        echo "  --embed-dim=128      Embedding dimension"
        echo "  --heads=4            Number of attention heads"
        echo "  --layers=4           Number of transformer layers"
        echo "  --ff-mult=4          FFN multiplier"
        echo "  --quant=auto         Quantization type: int8|int4|fp8|auto|none"
        echo "  --backend=metal      Backend: cpu|metal|cuda|opencl|auto"
        echo "  --no-session         Disable session API"
        echo "  --save=finetune.nimq Output checkpoint path"
        echo "  --help               Show this help"
        quit(0)
      else:
        stderr.writeLine &"[cảnh báo] cờ không rõ: --{key}"
    except ValueError:
      stderr.writeLine &"[cảnh báo] giá trị không hợp lệ cho --{key}='{val}', bỏ qua"

# ═══════════════════════════════════════════════════════════════
# CHARTOKENIZER - BYTE-LEVEL
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
  result.vocabSize = max(idx, 1)

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
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════

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

proc loadAllTexts*(cfg: Config): seq[string] =
  result = @[]
  echo "  -> databricks-dolly-15k.jsonl ..."
  result.add loadDolly(cfg.dataDir / "databricks-dolly-15k.jsonl")
  if result.len == 0:
    echo "  -> No data loaded, using dummy data for testing"
    for i in 0 ..< 100:
      result.add("This is dummy training text number " & $i & ". ")

# ═══════════════════════════════════════════════════════════════
# BUILD SAMPLES
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
# CROSS ENTROPY LOSS - BATCHED
# ═══════════════════════════════════════════════════════════════
proc crossEntropyLossBatch*(logits: Tensor, targetsBatch: seq[seq[int]]): tuple[loss: float32, dLogits: Tensor] =
  let B = logits.shape[0]
  let T = logits.shape[1]
  let vocab = logits.shape[2]
  
  # === FIX: KHÔNG GÁN logits.data, DÙNG BIẾN TẠM ===
  var logitsData = logits.data
  let expectedSize = B * T * vocab
  if logitsData.len != expectedSize:
    stderr.writeLine "[WARN] crossEntropyLossBatch: logits.data.len=", logitsData.len, " expected=", expectedSize
    var fixedLogits = newSeq[float32](expectedSize)
    let copyLen = min(logitsData.len, expectedSize)
    for i in 0 ..< copyLen:
      fixedLogits[i] = logitsData[i]
    logitsData = fixedLogits
  
  # Kiểm tra targetsBatch có đúng kích thước không
  if targetsBatch.len != B:
    stderr.writeLine "[WARN] targetsBatch.len=", targetsBatch.len, " != B=", B
    return (0.0'f32, newTensor(logits.shape))
  for b in 0 ..< B:
    if targetsBatch[b].len != T:
      stderr.writeLine "[WARN] targetsBatch[", b, "].len=", targetsBatch[b].len, " != T=", T
      return (0.0'f32, newTensor(logits.shape))
  
  var loss = 0'f32
  var dLogits = newTensor(logits.shape)
  let denom = float32(B * T)
  for b in 0 ..< B:
    let baseB = b * T * vocab
    for t in 0 ..< T:
      let off = baseB + t * vocab
      let target = targetsBatch[b][t]
      let maxVal = logitsData[off ..< off + vocab].max()
      var sumExp = 0'f32
      for i in 0 ..< vocab:
        sumExp += exp(logitsData[off + i] - maxVal)
      let prob = exp(logitsData[off + target] - maxVal) / sumExp
      loss += -ln(max(prob, 1e-12'f32))
      for i in 0 ..< vocab:
        dLogits.data[off + i] = exp(logitsData[off + i] - maxVal) / sumExp
      dLogits.data[off + target] -= 1.0
  loss /= denom
  for i in 0 ..< dLogits.data.len:
    dLogits.data[i] /= denom
  result = (loss, dLogits)
# ═══════════════════════════════════════════════════════════════
# NIMFORMER MODEL VỚI LINEAR++ ATTENTION
# ═══════════════════════════════════════════════════════════════

type
  TransformerBlockLinearPP* = object
    attn*: LinearPlusAttention
    ff*: FeedForward
    ln1*, ln2*: LayerNorm

proc newTransformerBlockLinearPP*(embedDim, nHeads, ffMult: int, useSession: bool): TransformerBlockLinearPP =
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
    dX, dAttnQkvW, dAttnQkvB, dAttnProjW, dAttnProjB,
    dFf1W, dFf1B, dFf2W, dFf2B,
    dLn1G, dLn1B, dLn2G, dLn2B: Tensor] =

  let x1 = blk.ln1.forward(x, ctx)
  let attnOut = forwardLinearPlus(blk.attn, x1, ctx)
  let x2 = addT(ctx, x, attnOut)
  let x3 = blk.ln2.forward(x2, ctx)
  let ffOut = blk.ff.forward(x3, ctx)
  let y = addT(ctx, x2, ffOut)

  var dY = dOut
  let dFF = blk.ff.backward(x3, dY, ctx)
  var dX2 = dFF.dX
  let dLN2 = blk.ln2.backward(x2, dX2, ctx)
  var dX2_ln = dLN2.dX
  for i in 0 ..< dX2_ln.data.len:
    dX2_ln.data[i] += dY.data[i]

  let dAttn = backwardLinearPlus(blk.attn, x1, dX2_ln, ctx)
  let dLN1 = blk.ln1.backward(x, dAttn.dX, ctx)
  var dX0 = dLN1.dX
  for i in 0 ..< dX0.data.len:
    dX0.data[i] += dX2_ln.data[i]

  result = (
    dX0,
    dAttn.dQkvW, dAttn.dQkvB, dAttn.dProjW, dAttn.dProjB,
    dFF.dFc1W, dFF.dFc1B, dFF.dFc2W, dFF.dFc2B,
    dLN1.dGamma, dLN1.dBeta, dLN2.dGamma, dLN2.dBeta
  )

type
  NimformerModelLinearPP* = object
    embed*: Embedding
    blocks*: seq[TransformerBlockLinearPP]
    outProj*: Linear
    vocab*: int

proc newNimformerModelLinearPP*(vocab, embedDim, nHeads, nLayers, ffMult: int, useSession: bool): NimformerModelLinearPP =
  result.vocab = vocab
  result.embed = newEmbedding(vocab, embedDim)
  result.blocks = newSeq[TransformerBlockLinearPP](nLayers)
  for i in 0 ..< nLayers:
    result.blocks[i] = newTransformerBlockLinearPP(embedDim, nHeads, ffMult, useSession)
  result.outProj = newLinear(embedDim, vocab)

proc forwardBatch*(m: NimformerModelLinearPP, idsBatch: seq[seq[int]], ctx: Backend): Tensor =
  var x = m.embed.lookupBatch(idsBatch, ctx)
  for blk in m.blocks:
    x = blk.forward(x, ctx)
  result = m.outProj.forward(x, ctx)

proc backwardBatch*(m: NimformerModelLinearPP, idsBatch: seq[seq[int]], dLoss: Tensor, ctx: Backend): seq[Tensor] =
  var x = m.embed.lookupBatch(idsBatch, ctx)
  var hiddenStates: seq[Tensor] = @[x]
  for blk in m.blocks:
    x = blk.forward(x, ctx)
    hiddenStates.add(x)
  let dOutProj = m.outProj.backward(x, dLoss, ctx)
  var dOut = dOutProj.dX
  var grads: seq[Tensor] = @[dOutProj.dW, dOutProj.dB]
  for i in countdown(m.blocks.len - 1, 0):
    let blk = m.blocks[i]
    let x_prev = hiddenStates[i]
    let dBlock = blk.backward(x_prev, dOut, ctx)
    dOut = dBlock.dX
    grads.add(dBlock.dAttnQkvW); grads.add(dBlock.dAttnQkvB)
    grads.add(dBlock.dAttnProjW); grads.add(dBlock.dAttnProjB)
    grads.add(dBlock.dFf1W); grads.add(dBlock.dFf1B)
    grads.add(dBlock.dFf2W); grads.add(dBlock.dFf2B)
    grads.add(dBlock.dLn1G); grads.add(dBlock.dLn1B)
    grads.add(dBlock.dLn2G); grads.add(dBlock.dLn2B)
  let dEmb = m.embed.backward(idsBatch, dOut, ctx)
  grads.add(dEmb)
  return grads

# ═══════════════════════════════════════════════════════════════
# APF ADAM OPTIMIZER
# ═══════════════════════════════════════════════════════════════

type
  ApfAdamState* = object
    m*, v*: seq[float32]
    step*: int

proc newApfAdamState*(paramLen: int): ApfAdamState =
  result.m = newSeq[float32](paramLen)
  result.v = newSeq[float32](paramLen)
  result.step = 0

proc apfAdamStep*(param: var Tensor, grad: Tensor, state: var ApfAdamState,
                   lr: float32 = 1e-3'f32, b1: float32 = 0.9'f32, b2: float32 = 0.999'f32,
                   eps: float32 = 1e-8'f32, requantizeEvery: int = 50): CustomFloat =
  inc state.step
  let bc1 = 1'f32 - pow(b1, float32(state.step))
  let bc2 = 1'f32 - pow(b2, float32(state.step))
  for i in 0 ..< param.data.len:
    let g = grad.data[i]
    state.m[i] = b1 * state.m[i] + (1'f32 - b1) * g
    state.v[i] = b2 * state.v[i] + (1'f32 - b2) * g * g
    let mHat = state.m[i] / bc1
    let vHat = state.v[i] / bc2
    param.data[i] -= lr * mHat / (sqrt(vHat) + eps)
  if state.step mod requantizeEvery == 0:
    let cf = buildCustomDtypeForTensor(param.data, grad.data)
    param.data = decodeArray(encodeArray(param.data, cf), cf)
    result = cf
  else:
    result = buildCustomDtypeForTensor(param.data)

# ═══════════════════════════════════════════════════════════════
# PARAMETER MANAGEMENT
# ═══════════════════════════════════════════════════════════════

template forEachParam(model: NimformerModelLinearPP, op: untyped) =
  op("outProj.weight", model.outProj.weight)
  op("outProj.bias", model.outProj.bias)
  for bi {.inject.} in countdown(model.blocks.len - 1, 0):
    let blk = model.blocks[bi]
    let nQkvW = &"blocks.{bi}.attn.qkv.weight"
    let nQkvB = &"blocks.{bi}.attn.qkv.bias"
    let nProjW = &"blocks.{bi}.attn.proj.weight"
    let nProjB = &"blocks.{bi}.attn.proj.bias"
    let nFc1W = &"blocks.{bi}.ff.fc1.weight"
    let nFc1B = &"blocks.{bi}.ff.fc1.bias"
    let nFc2W = &"blocks.{bi}.ff.fc2.weight"
    let nFc2B = &"blocks.{bi}.ff.fc2.bias"
    let nLn1G = &"blocks.{bi}.ln1.gamma"
    let nLn1B = &"blocks.{bi}.ln1.beta"
    let nLn2G = &"blocks.{bi}.ln2.gamma"
    let nLn2B = &"blocks.{bi}.ln2.beta"
    op(nQkvW, blk.attn.qkv.weight)
    op(nQkvB, blk.attn.qkv.bias)
    op(nProjW, blk.attn.proj.weight)
    op(nProjB, blk.attn.proj.bias)
    op(nFc1W, blk.ff.fc1.weight)
    op(nFc1B, blk.ff.fc1.bias)
    op(nFc2W, blk.ff.fc2.weight)
    op(nFc2B, blk.ff.fc2.bias)
    op(nLn1G, blk.ln1.gamma)
    op(nLn1B, blk.ln1.beta)
    op(nLn2G, blk.ln2.gamma)
    op(nLn2B, blk.ln2.beta)
  op("embed.weight", model.embed.weight)

proc paramLens(model: NimformerModelLinearPP): seq[int] =
  result = @[]
  template rec(name: string, t: Tensor) = result.add(t.data.len)
  forEachParam(model, rec)

proc initOptStates(model: NimformerModelLinearPP): seq[ApfAdamState] =
  result = @[]
  for l in paramLens(model): result.add newApfAdamState(l)
proc applyGrads(model: var NimformerModelLinearPP, grads: seq[Tensor],
                 states: var seq[ApfAdamState], lr: float32, requantizeEvery: int) =
  var idx = 0
  template step(param: untyped) =
    if idx < grads.len:
      discard apfAdamStep(param, grads[idx], states[idx], lr, requantizeEvery = requantizeEvery)
    else:
      stderr.writeLine "[WARN] applyGrads: idx=", idx, " >= grads.len=", grads.len
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
# SAVE/LOAD CHECKPOINT
# ═══════════════════════════════════════════════════════════════

proc isBiasOrNorm(name: string): bool =
  name.endsWith(".bias") or name.endsWith(".gamma") or name.endsWith(".beta")

proc saveCheckpoint*(model: NimformerModelLinearPP, states: seq[ApfAdamState], stepNo: int,
                      path: string, weightKind: QuantKind,
                      embedDim, nHeads, nLayers, ffMult: int) =
  var sd: seq[(string, QuantTensor)] = @[]
  var idx = 0
  template rec(name: string, t: Tensor) =
    let kind = if isBiasOrNorm(name): qkFp32Raw else: weightKind
    sd.add (name, quantizeTensor(t.data, t.shape, kind))
    sd.add (name & ".opt_m", quantizeTensor(states[idx].m, @[states[idx].m.len], qkAuto))
    sd.add (name & ".opt_v", quantizeTensor(states[idx].v, @[states[idx].v.len], qkAuto))
    inc idx
  forEachParam(model, rec)
  sd.add ("__step__", quantizeTensor(@[float32(stepNo)], @[1], qkFp32Raw))
  saveQuantStateDict(path, [model.vocab, embedDim, nHeads, nLayers, ffMult], sd)
  echo "  đã lưu checkpoint (weight+optimizer, step=" & $stepNo & ") -> " & path

proc loadCheckpointFull*(path: string): tuple[model: NimformerModelLinearPP, states: seq[ApfAdamState], stepNo: int] =
  let (arch, sd) = loadQuantStateDict(path)
  let vocab = arch[0]; let embedDim = arch[1]; let nHeads = arch[2]
  let nLayers = arch[3]; let ffMult = arch[4]
  var model = newNimformerModelLinearPP(vocab, embedDim, nHeads, nLayers, ffMult, true)
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
    st.step = stepNo
    if byName.hasKey(name & ".opt_m"): st.m = dequantizeTensor(byName[name & ".opt_m"])
    if byName.hasKey(name & ".opt_v"): st.v = dequantizeTensor(byName[name & ".opt_v"])
    states.add st

  loadWithOpt("outProj.weight", model.outProj.weight)
  loadWithOpt("outProj.bias", model.outProj.bias)
  for bi in countdown(model.blocks.len - 1, 0):
    let kQkvW = &"blocks.{bi}.attn.qkv.weight"
    let kQkvB = &"blocks.{bi}.attn.qkv.bias"
    let kProjW = &"blocks.{bi}.attn.proj.weight"
    let kProjB = &"blocks.{bi}.attn.proj.bias"
    let kFc1W = &"blocks.{bi}.ff.fc1.weight"
    let kFc1B = &"blocks.{bi}.ff.fc1.bias"
    let kFc2W = &"blocks.{bi}.ff.fc2.weight"
    let kFc2B = &"blocks.{bi}.ff.fc2.bias"
    let kLn1G = &"blocks.{bi}.ln1.gamma"
    let kLn1B = &"blocks.{bi}.ln1.beta"
    let kLn2G = &"blocks.{bi}.ln2.gamma"
    let kLn2B = &"blocks.{bi}.ln2.beta"
    loadWithOpt(kQkvW, model.blocks[bi].attn.qkv.weight)
    loadWithOpt(kQkvB, model.blocks[bi].attn.qkv.bias)
    loadWithOpt(kProjW, model.blocks[bi].attn.proj.weight)
    loadWithOpt(kProjB, model.blocks[bi].attn.proj.bias)
    loadWithOpt(kFc1W, model.blocks[bi].ff.fc1.weight)
    loadWithOpt(kFc1B, model.blocks[bi].ff.fc1.bias)
    loadWithOpt(kFc2W, model.blocks[bi].ff.fc2.weight)
    loadWithOpt(kFc2B, model.blocks[bi].ff.fc2.bias)
    loadWithOpt(kLn1G, model.blocks[bi].ln1.gamma)
    loadWithOpt(kLn1B, model.blocks[bi].ln1.beta)
    loadWithOpt(kLn2G, model.blocks[bi].ln2.gamma)
    loadWithOpt(kLn2B, model.blocks[bi].ln2.beta)
  loadWithOpt("embed.weight", model.embed.weight)

  result = (model, states, stepNo)

# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

proc train(model: var NimformerModelLinearPP, samples: seq[Sample], ctx: Backend,
           cfg: Config, states: var seq[ApfAdamState], startStep: int = 0) =
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

      let logits = model.forwardBatch(idsBatch, ctx)
      let (loss, dLogits) = crossEntropyLossBatch(logits, targetsBatch)
      let grads = model.backwardBatch(idsBatch, dLogits, ctx)
      applyGrads(model, grads, states, cfg.lr, cfg.requantizeEvery)

      inc stepNo
      if stepNo mod cfg.logEvery == 0 or stepNo == 1:
        echo &"[step {stepNo}/{cfg.steps}] loss={loss:.6f} (batch={idsBatch.len})"
      if cfg.ckptEvery > 0 and stepNo mod cfg.ckptEvery == 0:
        saveCheckpoint(model, states, stepNo, cfg.savePath & ".ckpt", quantKindOf(cfg.quant),
                        cfg.embedDim, cfg.nHeads, cfg.nLayers, cfg.ffMult)

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

proc main() =
  let cfg = parseArgs()
  randomize(cfg.seed)

  echo "== Linear++ Attention Training =="
  echo &"  Backend: {cfg.backend}, Session: {cfg.useSession}"
  echo &"  Steps: {cfg.steps}, SeqLen: {cfg.seqLen}, Batch: {cfg.batchSize}"
  echo &"  EmbedDim: {cfg.embedDim}, Heads: {cfg.nHeads}, Layers: {cfg.nLayers}"

  echo "== Load training texts =="
  let texts = loadAllTexts(cfg)
  echo &"TOTAL TEXTS: {texts.len}"
  if texts.len == 0:
    stderr.writeLine "Không load được dữ liệu nào"
    quit(1)

  let ckptPath = cfg.savePath & ".ckpt"
  var tok: CharTokenizer
  var model: NimformerModelLinearPP
  var states: seq[ApfAdamState]
  var startStep = 0
  let resuming = fileExists(ckptPath)

  if resuming:
    echo &"== Tìm thấy checkpoint {ckptPath} -> resume =="
    if fileExists(cfg.tokenizerPath):
      tok = loadTokenizer(cfg.tokenizerPath)
      echo &"  đã tải lại tokenizer cũ -> {cfg.tokenizerPath}"
    else:
      tok = newCharTokenizer(texts)
      tok.saveTokenizer(cfg.tokenizerPath)
    let loaded = loadCheckpointFull(ckptPath)
    model = loaded.model
    states = loaded.states
    startStep = loaded.stepNo
    echo &"  resume từ step {startStep}"
  else:
    echo "== Build tokenizer =="
    tok = newCharTokenizer(texts)
    tok.saveTokenizer(cfg.tokenizerPath)
    echo &"Vocab size: {tok.vocabSize}"

  echo &"== Build training samples (seq_len={cfg.seqLen}) =="
  let samples = buildSamples(texts, tok, cfg.seqLen)
  echo &"Total samples: {samples.len}"
  if samples.len == 0:
    stderr.writeLine "Không đủ dữ liệu để tạo sample"
    quit(1)

  echo "== Init backend =="
  let ctx = newBackend(cfg.backend)

  if not resuming:
    echo "== Build model =="
    model = newNimformerModelLinearPP(vocab = tok.vocabSize, embedDim = cfg.embedDim,
                                       nHeads = cfg.nHeads, nLayers = cfg.nLayers,
                                       ffMult = cfg.ffMult, useSession = cfg.useSession)
    states = initOptStates(model)

  echo &"  vocab={model.vocab}, {states.len} tensor tham số"

  echo &"== Training (bắt đầu từ step {startStep}/{cfg.steps}) =="
  let t0 = epochTime()
  train(model, samples, ctx, cfg, states, startStep)
  echo &"== Done in {epochTime() - t0:.1f}s =="

  echo &"== Save checkpoint cuối ({cfg.savePath}) =="
  var sd: seq[(string, QuantTensor)] = @[]
  template rec(name: string, t: Tensor) =
    let kind = if isBiasOrNorm(name): qkFp32Raw else: quantKindOf(cfg.quant)
    sd.add (name, quantizeTensor(t.data, t.shape, kind))
  forEachParam(model, rec)
  saveQuantStateDict(cfg.savePath, [model.vocab, cfg.embedDim, cfg.nHeads, cfg.nLayers, cfg.ffMult], sd)
  echo &"  đã lưu {sd.len} tensor -> {cfg.savePath}"

when isMainModule:
  main()
