## One-time migration without loading the base model or the full JSON document.

import std/[os, strutils, parsejson, streams, tables, json, strformat]
import rawbin

type Header = object
  epoch: int
  nimqPath, pattern: string
  adapterNames: seq[string]
  adapterMeta: Table[string, array[5, int]]
  tensorCount: int

proc readArray(p: var JsonParser): seq[float32] =
  if p.kind != jsonArrayStart: raiseParseErr(p, "expected array")
  p.next()
  while p.kind != jsonArrayEnd:
    result.add(float32(if p.kind == jsonInt: p.getInt().float else: p.getFloat()))
    p.next()
  p.next()

proc skipArray(p: var JsonParser) =
  if p.kind != jsonArrayStart: raiseParseErr(p, "expected array")
  p.next()
  while p.kind != jsonArrayEnd: p.next()
  p.next()

proc nextKey(p: var JsonParser): string =
  if p.kind == jsonObjectEnd: return ""
  if p.kind != jsonString: raiseParseErr(p, "expected object key")
  result = p.str
  p.next()

proc openRoot(path: string): JsonParser =
  result.open(newFileStream(path, fmRead), path)
  result.next()
  if result.kind != jsonObjectStart: raiseParseErr(result, "checkpoint root must be object")
  result.next()

proc scanHeader(path: string): Header =
  var p = openRoot(path)
  defer: p.close()
  result.adapterMeta = initTable[string, array[5, int]]()
  while true:
    let key = nextKey(p)
    if key.len == 0: break
    case key
    of "epoch": result.epoch = int(p.getInt()); p.next()
    of "nimqPath": result.nimqPath = p.str; p.next()
    of "pattern": result.pattern = p.str; p.next()
    of "adapters":
      p.next()
      while true:
        let name = nextKey(p)
        if name.len == 0: break
        result.adapterNames.add(name)
        var sizes: array[5, int]
        p.next()
        while true:
          let field = nextKey(p)
          if field.len == 0: break
          case field
          of "A": discard readArray(p)
          of "B": discard readArray(p)
          of "inF": sizes[2] = int(p.getInt()); p.next()
          of "outF": sizes[3] = int(p.getInt()); p.next()
          of "r": sizes[4] = int(p.getInt()); p.next()
          of "alpha": p.next()
          else: raiseParseErr(p, "unknown adapter field")
        result.adapterMeta[name] = sizes
      p.next()
    of "newBlockWeights":
      p.next()
      while true:
        let name = nextKey(p)
        if name.len == 0: break
        discard name
        skipArray(p)
        inc result.tensorCount
      p.next()
    else: p.next()

proc writeMeta(h: Header): Table[string, string] =
  result = initTable[string, string]()
  result["format"] = "nimformers-lora-v1"
  result["epoch"] = $h.epoch
  result["nimqPath"] = h.nimqPath
  result["pattern"] = h.pattern
  result["vocab"] = "32768"
  result["hidden"] = "4096"
  result["nHeads"] = "128"
  result["ffMult"] = "4"
  for name in h.adapterNames:
    let s = h.adapterMeta[name]
    result["adapter." & name & ".inF"] = $s[2]
    result["adapter." & name & ".outF"] = $s[3]
    result["adapter." & name & ".r"] = $s[4]
    result["adapter." & name & ".alpha"] = "8"

type AdapterData = object
  name: string
  a, b: seq[float32]
  inF, outF, rank: int
  alpha: float32

proc migrate(source, target: string) =
  let jsonPath = if source.endsWith(".json"): source else: source & ".lora.json"
  if not fileExists(jsonPath): raise newException(IOError, "Khong tim thay: " & jsonPath)
  echo "[migrate] Streaming mot pass, khong load base model."
  var p = openRoot(jsonPath)
  defer: p.close()
  var epoch = 0
  var nimqPath = ""
  var pattern = ""
  var adapters: seq[AdapterData] = @[]
  var writer: RawBinWriter
  var writerStarted = false

  while true:
    let key = nextKey(p)
    if key.len == 0: break
    case key
    of "epoch": epoch = int(p.getInt()); p.next()
    of "rank", "alpha": p.next()
    of "nimqPath": nimqPath = p.str; p.next()
    of "pattern": pattern = p.str; p.next()
    of "adapters":
      p.next()
      while true:
        let name = nextKey(p)
        if name.len == 0: break
        var ad = AdapterData(name: name, alpha: 8'f32)
        p.next()
        while true:
          let field = nextKey(p)
          if field.len == 0: break
          case field
          of "A": ad.a = readArray(p)
          of "B": ad.b = readArray(p)
          of "inF": ad.inF = int(p.getInt()); p.next()
          of "outF": ad.outF = int(p.getInt()); p.next()
          of "r": ad.rank = int(p.getInt()); p.next()
          of "alpha": ad.alpha = p.getFloat().float32; p.next()
          else: raiseParseErr(p, "unknown adapter field: " & field)
        adapters.add(ad)
      p.next()
      var meta = initTable[string, string]()
      meta["format"] = "nimformers-lora-v1"
      meta["epoch"] = $epoch
      meta["nimqPath"] = nimqPath
      meta["pattern"] = pattern
      meta["vocab"] = "32768"
      meta["hidden"] = "4096"
      meta["nHeads"] = "128"
      meta["ffMult"] = "4"
      for ad in adapters:
        meta["adapter." & ad.name & ".inF"] = $ad.inF
        meta["adapter." & ad.name & ".outF"] = $ad.outF
        meta["adapter." & ad.name & ".r"] = $ad.rank
        meta["adapter." & ad.name & ".alpha"] = $ad.alpha
      writer = beginRawBin(target, meta)
      writerStarted = true
      for ad in adapters:
        writer.writeRawTensor(RawBinTensor(name: "adapter::" & ad.name & "::A",
                                            shape: @[ad.rank, ad.inF], data: ad.a))
        writer.writeRawTensor(RawBinTensor(name: "adapter::" & ad.name & "::B",
                                            shape: @[ad.outF, ad.rank], data: ad.b))
      adapters = @[]
    of "newBlockWeights":
      if not writerStarted: raiseParseErr(p, "adapters must precede newBlockWeights")
      p.next()
      while true:
        let name = nextKey(p)
        if name.len == 0: break
        let values = readArray(p)
        # shape suy ra tu ten tensor (hidden=4096); fallback 1D neu khong khop
        let nbShape: seq[int] =
          if name.endsWith(".attn.qkvProj.weight"): @[12288, 4096]
          elif name.endsWith(".attn.outProj.weight"): @[4096, 4096]
          elif name.endsWith(".attn.qkvProj.bias"): @[12288]
          elif name.endsWith(".attn.outProj.bias"): @[4096]
          else: @[values.len]
        writer.writeRawTensor(RawBinTensor(name: "newBlock::" & name, shape: nbShape, data: values))
      p.next()
    else: p.next()
  if not writerStarted: raise newException(ValueError, "checkpoint thieu adapters")
  writer.finishRawBin()
  if fileExists(source & ".hf_tokenizer_dir"): copyFile(source & ".hf_tokenizer_dir", target & ".hf_tokenizer_dir")
  elif fileExists(source & ".bpe.json"): copyFile(source & ".bpe.json", target & ".bpe.json")
  echo &"[migrate] Hoan tat: {target}; base khong duoc ghi. JSON goc giu nguyen."

when isMainModule:
  let source = if paramCount() >= 1: paramStr(1) else: "jamba_lora.bin"
  let target = if paramCount() >= 2: paramStr(2) else: source
  migrate(source, target)
