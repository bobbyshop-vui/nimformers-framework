## Lift an `fprobe 2` dispatch trace into a runnable FIR graph.
##
## The lifter reads structure only: buffer intervals, per-launch ports, op tags,
## session_vm bytecode and registered weight names. It rebuilds the dataflow by
## port identity, folds one homogeneous layer stack into `repeat layers`, names
## weights from their registered checkpoint tensors, replaces concrete counts
## with config symbols and disassembles each micro program back into kernel text.
##
## Usage:
##   firlift --trace t.fprobe --config config.json --out DIR [--name NAME] [--report]
import std/[os, strutils, tables, json, algorithm, math]
import ./firvm

type
  PortRef* = object
    name*: string
    buffer*: int
    offset*: int
    bytes*: int

  Op* = ref object
    id*: int
    step*: int
    tag*, impl*, prog*, rawAttrs*: string
    parent*: int
    ins*, outs*: seq[PortRef]
    attrs*: Table[string, string]
    words*: seq[uint32]

  BufferRec* = object
    bytes*: int
    origin*, name*, dtype*, shape*: string

  RoleRec* = object
    offset*, bytes*: int
    role*: string

  NameRec* = object
    buffer*: int
    offset*, bytes*: int
    name*, dtype*, shape*: string

  TransferRec* = object
    kind*: string
    op*: int
    dst*, src*: PortRef
    srcOffsetByte*, dstOffsetByte*, n*: int
    data*: seq[uint32]   ## a short host table (permute strides), when the recorder kept it

  Trace* = object
    backend*: string
    bufs*: seq[BufferRec]
    programs*: Table[string, seq[uint32]]
    ops*: seq[Op]
    roles*: seq[RoleRec]
    names*: seq[NameRec]
    transfers*: seq[TransferRec]
    hasStep*: bool
    steps*: int
    stepPosition*, stepToken*: int   ## of the last recorded step (-1 when absent)

  BufferDef* = object
    name*: string
    elements*: int
    perLayer*: bool
    role*: string
    kind*: string
    dtype*: string
    tensor*: string
    dims*: seq[int]
    packedBytes*: int

  KernelDef* = object
    name*, tag*, text*, threadsExpr*: string
    tg*, tgBytes*, ins*, outs*: int
    portSizes*: seq[int]
    instructions*: int

  Lifter* = ref object
    trace*: Trace
    config*: JsonNode
    modelName*: string
    symbols*: Table[string, string]   # alias -> definition expression
    exprByValue*: Table[int, string]
    buffers*: seq[BufferDef]
    bufferByName*: Table[string, int]
    kernels*: seq[KernelDef]
    kernelKey*: Table[string, int]
    kernelRename*: Table[string, string]
    layers*: int
    nodesMicro*, nodesExt*, instructions*: int

# ── tokenizer helpers ────────────────────────────────────────────────

proc splitTop(s: string; sep: char): seq[string] =
  var depth = 0
  var quote = false
  var start = 0
  var i = 0
  while i < s.len:
    let c = s[i]
    if quote:
      if c == '\\' and i + 1 < s.len: inc i
      elif c == '"': quote = false
    else:
      case c
      of '"': quote = true
      of '(': inc depth
      of ')': dec depth
      else:
        if c == sep and depth == 0:
          result.add s[start ..< i]
          start = i + 1
    inc i
  result.add s[start .. ^1]

proc matchingParen(s: string; openIdx: int): int =
  ## Index of the ')' that closes s[openIdx] == '('.
  var depth = 0
  var quote = false
  var i = openIdx
  while i < s.len:
    let c = s[i]
    if quote:
      if c == '\\' and i + 1 < s.len: inc i
      elif c == '"': quote = false
    else:
      case c
      of '"': quote = true
      of '(': inc depth
      of ')':
        dec depth
        if depth == 0: return i
      else: discard
    inc i
  raise newException(ValueError, "unbalanced parentheses: " & s)

proc unquote(s: string): string =
  if s.len >= 2 and s[0] == '"' and s[^1] == '"':
    result = s[1 .. ^2]
    result = result.replace("\\\"", "\"").replace("\\\\", "\\").replace("\\n", "\n")
  else:
    result = s

proc parseKV(s: string): Table[string, string] =
  result = initTable[string, string]()
  for item in splitTop(s, ','):
    if item.len == 0: continue
    let eq = item.find('=')
    if eq <= 0: continue
    result[item[0 ..< eq]] = unquote(item[eq+1 .. ^1])

proc parseRef(s: string): PortRef =
  ## b12@345:678 (byte offset, byte count)
  if s == "none": return PortRef(buffer: -1)
  let at = s.find('@')
  let colon = s.rfind(':')
  if at < 0 or colon < 0:
    raise newException(ValueError, "malformed port reference: " & s)
  result.buffer = parseInt(s[1 ..< at])
  result.offset = parseInt(s[at+1 ..< colon])
  result.bytes = parseInt(s[colon+1 .. ^1])

proc parsePorts(s: string): seq[PortRef] =
  for item in splitTop(s, ','):
    if item.len == 0: continue
    let eq = item.find('=')
    if eq <= 0: raise newException(ValueError, "malformed port list: " & s)
    var p = parseRef(item[eq+1 .. ^1])
    p.name = item[0 ..< eq]
    result.add p

proc parseOp(line: string): Op =
  result = Op(parent: -1, attrs: initTable[string, string]())
  let body = line[3 .. ^1]
  let inPos = body.find("in=(")
  let outPos = body.find("out=(", if inPos >= 0: inPos else: 0)
  let attrsPos = body.find("attrs(", if outPos >= 0: outPos else: 0)
  let implPos = body.find("impl=")
  let parentPos = body.find("parent=")
  let progPos = body.find("prog=")
  let tagPos = body.find("tag=")
  if min(min(inPos, outPos), min(attrsPos, implPos)) < 0 or tagPos < 0 or
     parentPos < 0 or progPos < 0:
    raise newException(ValueError, "malformed op line: " & line)
  let spacePos = body.find(' ')
  result.id = parseInt(body[1 ..< spacePos])
  result.tag = body[tagPos+4 ..< inPos].strip()
  let inEnd = matchingParen(body, inPos+3)
  let outEnd = matchingParen(body, outPos+4)
  let attrsEnd = matchingParen(body, attrsPos+5)
  result.ins = parsePorts(body[inPos+4 ..< inEnd])
  result.outs = parsePorts(body[outPos+5 ..< outEnd])
  result.rawAttrs = body[attrsPos+6 ..< attrsEnd]
  result.attrs = parseKV(result.rawAttrs)
  result.impl = body[implPos+5 ..< parentPos].strip()
  let parentText = body[parentPos+7 ..< progPos].strip()
  result.parent = if parentText == "none": -1 else: parseInt(parentText[1 .. ^1])
  result.prog = body[progPos+5 .. ^1].strip()
  if result.prog.len == 0: result.prog = "none"

proc parseTrace*(text: string): Trace =
  result.programs = initTable[string, seq[uint32]]()
  result.bufs = @[]
  result.roles = @[]
  var inWords = ""
  var words: seq[uint32]
  var seenStep = false
  var inside = false
  for raw in text.splitLines():
    if inWords.len > 0:
      let t = raw.strip()
      var hexLine = t.len > 0
      if hexLine:
        for tok in t.splitWhitespace():
          if not tok.allCharsInSet(HexDigits): hexLine = false
      if hexLine:
        for w in t.splitWhitespace(): words.add parseHexInt(w).uint32
        continue
      result.programs[inWords] = words
      inWords = ""
    if raw.startsWith("backend "):
      let w = raw.splitWhitespace()
      if w.len >= 2: result.backend = w[1]
    elif raw.startsWith("buf "):
      var b = BufferRec()
      for item in raw[4 .. ^1].splitWhitespace():
        let eq = item.find('=')
        if eq < 0: continue
        let k = item[0 ..< eq]
        let v = unquote(item[eq+1 .. ^1])
        case k
        of "bytes": b.bytes = parseInt(v)
        of "origin": b.origin = v
        of "name": b.name = v
        of "dtype": b.dtype = v
        of "shape": b.shape = v
        else: discard
      result.bufs.add b
    elif raw.startsWith("role "):
      var r = RoleRec()
      let at = raw.find('@')
      let colon = raw.find(':', at)
      let rolePos = raw.find("role=")
      r.offset = parseInt(raw[at+1 ..< colon])
      r.bytes = parseInt(raw[colon+1 ..< rolePos-1])
      r.role = unquote(raw[rolePos+5 .. ^1])
      result.roles.add r
    elif raw.startsWith("name "):
      var n = NameRec()
      let at = raw.find('@')
      let colon = raw.find(':', at)
      n.buffer = parseInt(raw[5 ..< at][1 .. ^1])
      n.offset = parseInt(raw[at+1 ..< colon])
      let namePos = raw.find("name=")
      let dtypePos = raw.find("dtype=")
      let shapePos = raw.find("shape=")
      n.bytes = parseInt(raw[colon+1 ..< namePos-1])
      n.name = unquote(raw[namePos+5 ..< dtypePos-1])
      n.dtype = raw[dtypePos+6 ..< shapePos-1]
      n.shape = raw[shapePos+6 .. ^1]
      result.names.add n
    elif raw.startsWith("upload ") or raw.startsWith("copy ") or raw.startsWith("read "):
      var t = TransferRec()
      let toks = raw.splitWhitespace()
      t.kind = toks[0]
      for tok in toks[1 .. ^1]:
        let eq = tok.find('=')
        if eq < 0: continue
        let k = tok[0 ..< eq]
        let v = tok[eq+1 .. ^1]
        case k
        of "op": t.op = (if v == "none": -1 else: parseInt(v[1 .. ^1]))
        of "src": t.src = parseRef(v)
        of "dst": t.dst = parseRef(v)
        of "srcOffsetElems": t.srcOffsetByte = parseInt(v)*4
        of "dstOffsetElems": t.dstOffsetByte = parseInt(v)*4
        of "n": t.n = parseInt(v)
        of "bytes": t.n = parseInt(v) div 4
        of "data":
          for w in v.split(','): t.data.add uint32(parseHexInt(w))
        else: discard
      if t.kind == "upload" and t.n == 0: t.n = t.dst.bytes div 4
      result.transfers.add t
    elif raw.startsWith("step "):
      seenStep = true
      inside = true
      inc result.steps
      result.stepPosition = -1
      result.stepToken = -1
      for tok in raw.splitWhitespace():
        if tok.startsWith("position="): result.stepPosition = parseInt(tok[9 .. ^1])
        elif tok.startsWith("token="): result.stepToken = parseInt(tok[6 .. ^1])
    elif raw == "end step":
      inside = false
    elif raw.startsWith("op "):
      if seenStep and not inside: continue
      var op = parseOp(raw)
      op.step = result.steps
      result.ops.add op
    elif raw.startsWith("words "):
      inWords = raw.splitWhitespace()[1]
      words = @[]
  if inWords.len > 0:
    result.programs[inWords] = words
  result.hasStep = seenStep
  # Attach the recorded bytecode to each op.
  for op in result.ops:
    if op.prog.len > 1 and result.programs.hasKey(op.prog):
      op.words = result.programs[op.prog]

# ── config symbols ──────────────────────────────────────────────────

proc jsonInt(c: JsonNode; keys: openArray[string]): int =
  for k in keys:
    if c.hasKey(k) and c[k].kind == JInt: return c[k].getInt()
  -1

proc jsonFloat(c: JsonNode; keys: openArray[string]): float =
  for k in keys:
    if c.hasKey(k) and c[k].kind in {JFloat, JInt}: return c[k].getFloat()
  -1.0

proc addAlias(l: var Lifter; name: string; value: int; source: string) =
  if value <= 0 or source.len == 0: return
  l.symbols[name] = source
  if not l.exprByValue.hasKey(value):
    l.exprByValue[value] = name

proc firstPresent(c: JsonNode; keys: openArray[string]): string =
  for k in keys:
    if c.hasKey(k): return k
  ""

proc setupSymbols(l: var Lifter; repeatLayers: int) =
  let c = l.config
  let hidden = jsonInt(c, ["hidden_size", "n_embd", "d_model", "dim"])
  let nheads = jsonInt(c, ["num_attention_heads", "n_head", "n_heads"])
  let headdim = if hidden > 0 and nheads > 0 and hidden mod nheads == 0: hidden div nheads else: -1
  let mlp = jsonInt(c, ["intermediate_size", "n_inner", "ffn_dim"])
  let layers = jsonInt(c, ["num_hidden_layers", "n_layer", "num_layers", "n_layers"])
  let positions = jsonInt(c, ["max_position_embeddings", "n_positions", "n_ctx", "seq_length"])
  let vocab = jsonInt(c, ["vocab_size"])
  l.addAlias("hidden", hidden, firstPresent(c, ["hidden_size", "n_embd", "d_model", "dim"]))
  l.addAlias("nheads", nheads, firstPresent(c, ["num_attention_heads", "n_head", "n_heads"]))
  l.addAlias("headdim", headdim, "hidden/nheads")
  l.addAlias("mlp", mlp, firstPresent(c, ["intermediate_size", "n_inner", "ffn_dim"]))
  if layers > 0:
    l.addAlias("layers", layers, firstPresent(c, ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]))
  if positions > 0:
    l.addAlias("positions", positions, firstPresent(c, ["max_position_embeddings", "n_positions", "n_ctx", "seq_length"]))
  if vocab > 0:
    l.addAlias("vocab", vocab, firstPresent(c, ["vocab_size"]))
  # The trace decides the repeat count; keep the symbol consistent with it.
  if repeatLayers > 0:
    if l.symbols.hasKey("layers") and layers == repeatLayers: discard
    else: l.symbols["layers"] = $repeatLayers
    l.exprByValue[repeatLayers] = "layers"
  # Common derived sizes.
  if hidden > 0:
    for (mult, expr) in [(2, "2*hidden"), (3, "3*hidden"), (4, "4*hidden")]:
      if not l.exprByValue.hasKey(hidden*mult): l.exprByValue[hidden*mult] = expr
  if nheads > 0 and positions > 0:
    if not l.exprByValue.hasKey(nheads*positions): l.exprByValue[nheads*positions] = "nheads*positions"
  if hidden > 0 and positions > 0:
    if not l.exprByValue.hasKey(hidden*positions): l.exprByValue[hidden*positions] = "positions*hidden"
  if hidden > 0 and headdim > 0:
    if not l.exprByValue.hasKey(hidden-headdim): l.exprByValue[hidden-headdim] = "hidden-headdim"
  if hidden > 0 and nheads > 0:
    if not l.exprByValue.hasKey(hidden*nheads): l.exprByValue[hidden*nheads] = "hidden*nheads"

proc exprOf(l: Lifter; value: int): string =
  if l.exprByValue.hasKey(value): return l.exprByValue[value]
  $value

# ── analysis ────────────────────────────────────────────────────────

proc kindOf(op: Op): string = op.attrs.getOrDefault("kind", "")
proc layerOf(op: Op): int =
  if op.attrs.hasKey("layer"): parseInt(op.attrs["layer"]) else: 0
proc isRepeat(op: Op): bool = op.attrs.getOrDefault("repeat", "0") == "1"

proc graphOps*(t: Trace): seq[Op] =
  # One recorded step describes the graph; the second step is used to classify
  # persistent state by this lifter when present.
  let step = if t.steps > 0: t.steps else: 0
  for op in t.ops:
    if op.step == step and op.parent < 0 and op.attrs.hasKey("kind"): result.add op

proc weightTensor(op: Op; port: string): string = op.attrs.getOrDefault(port & ".tensor", "")

proc portElements(op: Op; p: PortRef; isWeight: bool): int =
  if isWeight:
    let shape = op.attrs.getOrDefault(p.name & ".shape", "")
    if shape.len >= 2:
      var n = 1
      for part in shape[1 .. ^2].splitWhitespace():
        if part.len > 0: n *= parseInt(part)
      return n
  p.bytes div 4

proc buildBuffers(l: var Lifter; ops: seq[Op]; repeatLayers: int) =
  var offsets = initTable[string, seq[int]]()
  var seen = initTable[string, bool]()
  for op in ops:
    for p in op.ins & op.outs:
      let isWeight = op.attrs.hasKey(p.name & ".tensor")
      if isWeight:
        if l.bufferByName.hasKey(p.name): continue
        var b = BufferDef(name: p.name, kind: "weight",
          dtype: op.attrs.getOrDefault(p.name & ".dtype", "f32"),
          tensor: op.attrs.getOrDefault(p.name & ".tensor", ""),
          elements: portElements(op, p, true), packedBytes: p.bytes)
        let shape = op.attrs.getOrDefault(p.name & ".shape", "")
        if shape.len >= 2:
          for part in shape[1 .. ^2].splitWhitespace():
            if part.len > 0: b.dims.add parseInt(part)
        l.bufferByName[b.name] = l.buffers.len
        l.buffers.add b
      else:
        if not l.bufferByName.hasKey(p.name):
          l.bufferByName[p.name] = l.buffers.len
          l.buffers.add BufferDef(name: p.name, kind: "activation",
            elements: p.bytes div 4)
        else:
          let idx = l.bufferByName[p.name]
          if p.bytes div 4 > l.buffers[idx].elements:
            l.buffers[idx].elements = p.bytes div 4
        if isRepeat(op):
          if not offsets.hasKey(p.name): offsets[p.name] = @[]
          if p.offset notin offsets[p.name]: offsets[p.name].add p.offset
  if repeatLayers > 0:
    for name, offs in offsets:
      let idx = l.bufferByName[name]
      let elems = l.buffers[idx].elements
      if offs.len <= 1: continue
      var sorted = offs
      sorted.sort()
      var uniform = true
      for i in 1 ..< sorted.len:
        if sorted[i] - sorted[i-1] != elems*4: uniform = false
      if not uniform:
        raise newException(ValueError, "buffer " & name & " has a non-uniform per-layer layout")
      if sorted.len == repeatLayers:
        l.buffers[idx].perLayer = true
      elif sorted.len > repeatLayers:
        raise newException(ValueError, "buffer " & name & " repeats more often than the layer count")
  # Roles come from the trace's registered arena intervals, matched to the
  # port that covers the same interval.
  for r in l.trace.roles:
    for op in ops:
      for p in op.ins & op.outs:
        if p.offset == r.offset and p.bytes == r.bytes:
          let idx = l.bufferByName.getOrDefault(p.name, -1)
          if idx >= 0: l.buffers[idx].role = r.role

proc weightTemplates(l: var Lifter; ops: seq[Op]; repeatLayers: int) =
  var perLayer = initTable[string, Table[int, string]]()
  for op in ops:
    if not isRepeat(op): continue
    for p in op.ins & op.outs:
      if op.attrs.hasKey(p.name & ".tensor"):
        if not perLayer.hasKey(p.name): perLayer[p.name] = initTable[int, string]()
        perLayer[p.name][layerOf(op)] = op.attrs[p.name & ".tensor"]
  for name, tensors in perLayer:
    let idx = l.bufferByName[name]
    if tensors.len != repeatLayers: continue
    var tokens: seq[seq[string]]
    for layer in 0 ..< repeatLayers:
      if not tensors.hasKey(layer): continue
      tokens.add tensors[layer].split('.')
    if tokens.len == 0: continue
    var varying = -1
    var ok = true
    for i in 0 ..< tokens[0].len:
      var differs = false
      for j in 1 ..< tokens.len:
        if tokens[j].len != tokens[0].len or tokens[j][i] != tokens[0][i]: differs = true
      if differs:
        if varying >= 0: ok = false
        varying = i
    if ok and varying >= 0:
      var parts = tokens[0]
      var numeric = true
      for layer in 0 ..< tokens.len:
        if tokens[layer][varying] != $layer: numeric = false
      if numeric:
        parts[varying] = "{i}"
        l.buffers[idx].tensor = parts.join(".")
        continue
    l.buffers[idx].tensor = tensors[0]

# ── kernel lifting ──────────────────────────────────────────────────

proc liftPorts(op: Op; refs: seq[PortRef]; prefix: string): seq[LiftPort] =
  for i, p in refs:
    let isWeight = op.attrs.hasKey(p.name & ".tensor")
    # f32 weights live in the arena at their recorded offset; q4 weights are
    # resident in their own buffers and are never touched by a micro kernel.
    let resident = isWeight and op.attrs.getOrDefault(p.name & ".dtype", "") == "q4"
    result.add LiftPort(name: prefix & $i,
      base: (if resident: 0 else: p.offset div 4),
      elements: portElements(op, p, isWeight))

proc threadsExpr(l: Lifter; op: Op; threads: int): string =
  if op.outs.len > 0:
    let out0 = portElements(op, op.outs[0], op.attrs.hasKey(op.outs[0].name & ".tensor"))
    if threads == out0: return "out0.size"
  if op.ins.len > 0:
    let in0 = portElements(op, op.ins[0], op.attrs.hasKey(op.ins[0].name & ".tensor"))
    if threads == in0: return "in0.size"
  if l.exprByValue.getOrDefault(threads, "") == "nheads": return "nheads"
  if threads == 1: return "1"
  $threads

proc kernelSignature(text, threadsExpr: string; tg, tgBytes: int): string =
  text & "\n@threads=" & threadsExpr & " tg=" & $tg & " bytes=" & $tgBytes

proc addKernel(l: var Lifter; op: Op) =
  let threads = if op.attrs.hasKey("threads"): parseInt(op.attrs["threads"]) else: 0
  let tg = if op.attrs.hasKey("tg"): parseInt(op.attrs["tg"]) else: 1
  let tgBytes = if op.attrs.hasKey("tgbytes"): parseInt(op.attrs["tgbytes"]) else: 16
  let ports = liftPorts(op, op.ins, "in") & liftPorts(op, op.outs, "out")
  let text = disassemble(op.words, ports)
  let baseName = op.attrs.getOrDefault("op", "kernel")
  let texpr = l.threadsExpr(op, threads)
  let sig = kernelSignature(text, texpr, tg, tgBytes)
  let key = baseName & "\n" & sig
  if l.kernelKey.hasKey(key):
    let idx = l.kernelKey[key]
    if l.kernels[idx].tag != op.tag and l.kernels[idx].tag.len > 0:
      l.kernels[idx].tag = l.kernels[idx].tag & "+" & op.tag
    return
  var name = baseName
  var suffix = 2
  while true:
    var taken = false
    for k in l.kernels:
      if k.name == name: taken = true
    if not taken: break
    name = baseName & "_" & $suffix
    inc suffix
  var k = KernelDef(name: name, tag: op.tag, text: text, threadsExpr: texpr,
    tg: tg, tgBytes: tgBytes, ins: op.ins.len, outs: op.outs.len)
  for p in op.ins & op.outs:
    k.portSizes.add portElements(op, p, op.attrs.hasKey(p.name & ".tensor"))
  k.instructions = text.splitLines().len
  l.kernelKey[key] = l.kernels.len
  l.kernels.add k

proc verifyRepeatKernels(l: Lifter; ops: seq[Op]; repeatLayers: int) =
  ## Every layer must lift to the same kernel text after addresses become
  ## port-relative. A difference means the recorded path is not homogeneous.
  var first = initTable[string, string]()
  for op in ops:
    if not isRepeat(op) or kindOf(op) != "micro": continue
    if op.words.len == 0:
      var keys: seq[string]
      for k in l.trace.programs.keys: keys.add k
      raise newException(ValueError, "micro op " & op.attrs.getOrDefault("name", "?") &
        " has no recorded program (prog=" & op.prog & ", words=" & $(if l.trace.programs.hasKey(op.prog): l.trace.programs[op.prog].len else: -1) & ", programs=" & $l.trace.programs.len &
        " keys=" & keys.join(",") & ")")
    let ports = liftPorts(op, op.ins, "in") & liftPorts(op, op.outs, "out")
    let text = disassemble(op.words, ports)
    let name = op.attrs.getOrDefault("op", "kernel")
    if not first.hasKey(name): first[name] = text
    elif first[name] != text:
      raise newException(ValueError, "kernel " & name & " differs between layers; " &
        "the trace is not one homogeneous repeat\nfirst:\n" & first[name] &
        "\nother:\n" & text)

proc buildKernels(l: var Lifter; ops: seq[Op]) =
  for op in ops:
    if kindOf(op) == "micro":
      addKernel(l, op)

# ── emission ────────────────────────────────────────────────────────

proc emitSymbols(l: Lifter): string =
  var order = ["hidden", "nheads", "headdim", "mlp", "layers", "positions", "vocab"]
  for k in order:
    if l.symbols.hasKey(k):
      result.add "sym " & k & " = " & l.symbols[k] & "\n"

proc epsilonKey(l: Lifter): string =
  for k in ["layer_norm_epsilon", "rms_norm_eps", "layer_norm_eps", "norm_eps"]:
    if l.config.hasKey(k) and l.config[k].kind in {JFloat, JInt}: return k
  ""

proc emitHeader(l: Lifter): string =
  result.add "fir 1\n"
  result.add "model " & l.modelName & "\n"
  result.add "config config.json\n"
  result.add "runtime graph\n"
  result.add "layout script\n"
  result.add "script_abi 1\n"
  let eps = l.epsilonKey()
  if eps.len > 0: result.add "epsilon " & eps & "\n"
  result.add l.emitSymbols()

proc emitBuffers(l: Lifter): string =
  for b in l.buffers:
    if b.kind == "weight": continue
    result.add "buf " & b.name & " f32 " & l.exprOf(b.elements)
    if b.perLayer: result.add " per_layer"
    if b.role.len > 0: result.add " " & (if b.role == "logits": "output:" else: "input:") & b.role
    result.add "\n"

proc emitWeights(l: Lifter): string =
  for b in l.buffers:
    if b.kind != "weight": continue
    var dims = ""
    for i, d in b.dims:
      if i > 0: dims.add ","
      dims.add l.exprOf(d)
    result.add "weight " & b.name & " " & b.dtype & " " & b.tensor & " tensor [" & dims & "]\n"

proc emitKernels(l: Lifter): string =
  for k in l.kernels:
    result.add "kernel " & k.name & " in=" & $k.ins & " out=" & $k.outs
    result.add " threads=" & k.threadsExpr & " tg=" & $k.tg & " tgbytes=" & $k.tgBytes
    var allEqual = k.portSizes.len > 0
    if allEqual:
      for s in k.portSizes:
        if s != k.portSizes[0]: allEqual = false
    if allEqual:
      var ports: seq[string]
      for i in 0 ..< k.ins: ports.add "in" & $i
      for i in 0 ..< k.outs: ports.add "out" & $i
      result.add " size=" & ports.join(",")
    else:
      var expect: seq[string]
      for i in 0 ..< k.ins: expect.add "in" & $i & ":" & l.exprOf(k.portSizes[i])
      for i in 0 ..< k.outs: expect.add "out" & $i & ":" & l.exprOf(k.portSizes[k.ins+i])
      if expect.len > 0: result.add " expect=" & expect.join(",")
    if k.tag.len > 0: result.add " tag=" & k.tag
    result.add "\n"
    result.add k.text & "\n"
    result.add "endkernel\n"

proc kernelNameFor(l: Lifter; op: Op): string =
  let base = op.attrs.getOrDefault("op", "kernel")
  # Find the kernel with this base name whose text matches this op.
  let ports = liftPorts(op, op.ins, "in") & liftPorts(op, op.outs, "out")
  let text = disassemble(op.words, ports)
  let threads = if op.attrs.hasKey("threads"): parseInt(op.attrs["threads"]) else: 0
  let tg = if op.attrs.hasKey("tg"): parseInt(op.attrs["tg"]) else: 1
  let tgBytes = if op.attrs.hasKey("tgbytes"): parseInt(op.attrs["tgbytes"]) else: 16
  let sig = kernelSignature(text, l.threadsExpr(op, threads), tg, tgBytes)
  if l.kernelKey.hasKey(base & "\n" & sig):
    return l.kernels[l.kernelKey[base & "\n" & sig]].name
  raise newException(ValueError, "missing kernel for op " & op.attrs.getOrDefault("name", "?"))

proc emitNodes(l: Lifter; ops: seq[Op]; bodyStart, bodyLen, repeatLayers: int): string =
  var seen = initTable[string, int]()
  proc nodeLine(op: Op): string =
    let name0 = op.attrs.getOrDefault("name", "node")
    var name = name0
    if seen.hasKey(name):
      seen[name] = seen[name] + 1
      name = name0 & "_" & $seen[name0]
    else:
      seen[name0] = 1
    let kind = kindOf(op)
    let opName = if kind == "micro": l.kernelNameFor(op) else: op.attrs.getOrDefault("op", "ext")
    var ins: seq[string]
    for p in op.ins: ins.add p.name
    var outs: seq[string]
    for p in op.outs: outs.add p.name
    "node " & name & " " & kind & " " & opName & " " & ins.join(" ") & " -> " & outs.join(" ") & "\n"
  var i = 0
  while i < ops.len:
    if i == bodyStart and repeatLayers > 0:
      result.add "repeat layers\n"
      for j in i ..< i + bodyLen:
        result.add nodeLine(ops[j])
        if ops[j].attrs.getOrDefault("kind", "") == "micro": inc l.nodesMicro else: inc l.nodesExt
      result.add "endrepeat\n"
      i += bodyLen*repeatLayers
      # The body is recorded once per layer; skip the remaining blocks.
      continue
    result.add nodeLine(ops[i])
    if kindOf(ops[i]) == "micro": inc l.nodesMicro else: inc l.nodesExt
    inc i

proc buildModel*(trace: Trace; configText, modelName: string): string =
  var l = Lifter(trace: trace, config: parseJson(configText), modelName: modelName,
    symbols: initTable[string, string](), exprByValue: initTable[int, string](),
    bufferByName: initTable[string, int](), kernelKey: initTable[string, int](),
    kernelRename: initTable[string, string]())
  let ops = trace.graphOps()
  if ops.len == 0:
    raise newException(ValueError, "the trace has no graph nodes; " &
      "record a FIR graph run or use the session-op lifter")
  var repeatLayers = 0
  var bodyStart = -1
  var bodyLen = 0
  for i, op in ops:
    if isRepeat(op):
      if bodyStart < 0:
        bodyStart = i
        bodyLen = 1
        while bodyStart + bodyLen < ops.len and isRepeat(ops[bodyStart+bodyLen]) and layerOf(ops[bodyStart+bodyLen]) == 0:
          inc bodyLen
      repeatLayers = max(repeatLayers, layerOf(op) + 1)
  if bodyStart >= 0:
    for i in 0 ..< bodyStart:
      if isRepeat(ops[i]): raise newException(ValueError, "non-contiguous layer repeat")
    for i in bodyStart + bodyLen*repeatLayers ..< ops.len:
      if isRepeat(ops[i]): raise newException(ValueError, "non-contiguous layer repeat")
    verifyRepeatKernels(l, ops, repeatLayers)
  # The repeat body is the layer-0 block; all other blocks must match it.
  if repeatLayers > 0:
    for layer in 1 ..< repeatLayers:
      for j in 0 ..< bodyLen:
        let a = ops[bodyStart + j]
        let b = ops[bodyStart + bodyLen*layer + j]
        if a.attrs.getOrDefault("name", "") != b.attrs.getOrDefault("name", "") or
           kindOf(a) != kindOf(b) or
           a.attrs.getOrDefault("op", "") != b.attrs.getOrDefault("op", ""):
          raise newException(ValueError, "layer " & $layer & " does not match the first layer at " &
            a.attrs.getOrDefault("name", "?"))
  l.layers = repeatLayers
  l.setupSymbols(repeatLayers)
  l.buildBuffers(ops, repeatLayers)
  l.weightTemplates(ops, repeatLayers)
  l.buildKernels(ops)
  var text = l.emitHeader()
  text.add l.emitBuffers()
  text.add l.emitWeights()
  text.add l.emitKernels()
  text.add l.emitNodes(ops, bodyStart, bodyLen, repeatLayers)
  return text

proc report*(text: string) =
  var kernels = 0
  var nodes = 0
  var extNodes = 0
  var microNodes = 0
  var instructions = 0
  var inKernel = false
  for line in text.splitLines():
    if line.startsWith("kernel "):
      inc kernels
      inKernel = true
    elif line == "endkernel":
      inKernel = false
    elif inKernel:
      if line.len > 0: inc instructions
    elif line.startsWith("node "):
      inc nodes
      if line.splitWhitespace()[2] == "ext": inc extNodes else: inc microNodes
  echo "report: nodes=", nodes, " micro=", microNodes, " ext=", extNodes,
    " kernels=", kernels, " instructions=", instructions
