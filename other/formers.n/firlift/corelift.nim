## Lift a session-level decode trace (the native Llama runtime) into FIR.
##
## The semantic tier records one `op` per session call with its handles and
## scalar attributes. This module recognises the decoder idiom, fuses
## RoPE + KV-cache write + GQA decode attention into the library attention
## kernel, coalesces the per-head KV handles into per-layer cache buffers and
## folds the layer stack.
import std/[strutils, tables, json]
import ./firlift
import ./kernel_library

type
  CoreNode* = object
    kind*, op*, name*, tag*: string
    ins*, outs*: seq[string]

  CoreModel* = object
    buffers*: seq[(string, int, bool, string)] # name, elements, perLayer, role
    weights*: seq[(string, string, seq[int])]  # fir name, tensor, dims
    q4t*: Table[string, bool]
    nodes*: seq[CoreNode]
    layers*: int

proc cfgInt(c: JsonNode; keys: openArray[string]): int =
  for k in keys:
    if c.hasKey(k) and c[k].kind == JInt: return c[k].getInt()
  -1

proc cfgFloat(c: JsonNode; keys: openArray[string]): float =
  for k in keys:
    if c.hasKey(k) and c[k].kind in {JFloat, JInt}: return c[k].getFloat()
  -1.0

proc firstKey(c: JsonNode; keys: openArray[string]): string =
  for k in keys:
    if c.hasKey(k): return k
  ""

proc findName(t: Trace; p: PortRef): NameRec =
  for n in t.names:
    if n.buffer == p.buffer and n.offset == p.offset and n.bytes == p.bytes:
      return n
  for n in t.names:
    if n.buffer == p.buffer and n.offset == p.offset:
      return n
  raise newException(ValueError, "no registered weight name for buffer b" & $p.buffer &
    "@" & $p.offset)

proc layerOfTensor(t: string): int =
  let parts = t.split('.')
  for i, p in parts:
    if (p == "layers" or p == "h") and i + 1 < parts.len:
      try: return parseInt(parts[i+1])
      except ValueError: discard
  -1

proc shortOf(t: string): string =
  let parts = t.split('.')
  if parts.len >= 2: return parts[^2]
  t

proc templateOf(t: string): string =
  var parts = t.split('.')
  for i, p in parts:
    if (p == "layers" or p == "h") and i + 1 < parts.len:
      try:
        discard parseInt(parts[i+1])
        parts[i+1] = "{i}"
      except ValueError: discard
  parts.join(".")

proc addBuf(m: var CoreModel; name: string; elements: int; perLayer = false; role = "") =
  for b in m.buffers:
    if b[0] == name: return
  m.buffers.add (name, elements, perLayer, role)

proc addWeight(m: var CoreModel; firName, tensor: string; dims: seq[int]) =
  for w in m.weights:
    if w[0] == firName: return
  m.weights.add (firName, tensor, dims)

proc shapeOf(n: NameRec): seq[int] =
  if n.shape.len >= 2 and n.shape[0] == '(':
    for part in n.shape[1 .. ^2].splitWhitespace():
      if part.len > 0: result.add parseInt(part)

proc refMapName(names: Table[int, string]; p: PortRef; fallback: string): string =
  if names.hasKey(p.buffer): return names[p.buffer]
  fallback

proc nodeSig(n: CoreNode): string =
  n.kind & "|" & n.op & "|" & n.ins.join(",") & "|" & n.outs.join(",")

proc fuseLayer(nodes: seq[CoreNode]): seq[CoreNode] =
  ## Fold the residual add into the following RMSNorm and the SiLU gate into
  ## one kernel, as the inference path does: 15 launches per layer become 12.
  var i = 0
  while i < nodes.len:
    if i+2 < nodes.len and nodes[i].op == "script.sigmoid" and
         nodes[i+1].op == "script.mul" and nodes[i+2].op == "script.mul" and
         nodes[i].outs[0] == nodes[i+1].ins[1] and nodes[i+1].outs[0] == nodes[i+2].ins[0]:
      result.add CoreNode(kind: "micro", op: "script.silu_mul", name: "silu_mul",
        ins: @[nodes[i+2].ins[1], nodes[i].ins[0]], outs: @[nodes[i+2].outs[0]], tag: nodes[i].tag)
      i += 3
    else:
      result.add nodes[i]
      inc i

proc emitCore*(m: CoreModel; config: JsonNode; modelName: string): string =
  let hidden = cfgInt(config, ["hidden_size", "n_embd", "d_model"])
  let nheads = cfgInt(config, ["num_attention_heads", "n_head"])
  let nkv = cfgInt(config, ["num_key_value_heads", "n_kv_heads"])
  let mlp = cfgInt(config, ["intermediate_size", "n_inner"])
  let positions = cfgInt(config, ["max_position_embeddings", "n_positions", "n_ctx"])
  let vocab = cfgInt(config, ["vocab_size"])
  if min(min(hidden, nheads), min(nkv, min(mlp, min(positions, vocab)))) <= 0:
    raise newException(ValueError, "core lift needs hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, max_position_embeddings and vocab_size")
  let headdim = hidden div nheads
  var outText = ""
  outText.add "fir 1\n"
  outText.add "model " & modelName & "\n"
  outText.add "config config.json\n"
  outText.add "runtime graph\n"
  outText.add "layout script\n"
  outText.add "script_abi 1\n"
  let eps = firstKey(config, ["rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps"])
  if eps.len > 0: outText.add "epsilon " & eps & "\n"
  if config.hasKey("torch_dtype") and config["torch_dtype"].getStr("") in ["bfloat16", "float16"]:
    outText.add "require feature weights_bf16\n"
  outText.add "sym hidden = " & firstKey(config, ["hidden_size", "n_embd", "d_model"]) & "\n"
  outText.add "sym nheads = " & firstKey(config, ["num_attention_heads", "n_head"]) & "\n"
  outText.add "sym nkvheads = " & firstKey(config, ["num_key_value_heads", "n_kv_heads"]) & "\n"
  outText.add "sym headdim = hidden/nheads\n"
  outText.add "sym mlp = " & firstKey(config, ["intermediate_size", "n_inner"]) & "\n"
  outText.add "sym layers = " & firstKey(config, ["num_hidden_layers", "n_layer", "num_layers"]) & "\n"
  outText.add "sym positions = " & firstKey(config, ["max_position_embeddings", "n_positions", "n_ctx"]) & "\n"
  outText.add "sym vocab = " & firstKey(config, ["vocab_size"]) & "\n"
  outText.add "require sym layers = " & $m.layers & "\n"
  outText.add "buf token f32 1 input:token\n"
  outText.add "buf position f32 1 input:position\n"
  proc bufExpr(n: int): string =
    if n == hidden: "hidden"
    elif n == mlp: "mlp"
    elif n == vocab: "vocab"
    elif n == nkv*headdim: "nkvheads*headdim"
    elif n == nkv*positions*headdim: "nkvheads*positions*headdim"
    elif n == nheads*positions + hidden: "nheads*positions+hidden"
    else: $n
  for b in m.buffers:
    if b[3] == "token" or b[3] == "position": continue
    outText.add "buf " & b[0] & " f32 " & bufExpr(b[1])
    if b[2]: outText.add " per_layer"
    if b[3] == "logits": outText.add " output:logits"
    outText.add "\n"
  for w in m.weights:
    var dims: seq[string]
    for d in w[2]:
      dims.add (if d == hidden: "hidden" elif d == mlp: "mlp" elif d == vocab: "vocab" elif d == nkv*headdim: "nkvheads*headdim" elif d == positions*headdim: "positions*headdim" else: $d)
    let dtype = if m.q4t.getOrDefault(w[0], false): "q4t" else: "f32"
    outText.add "weight " & w[0] & " " & dtype & " " & w[1] & " tensor [" & dims.join(",") & "]\n"
  # Library kernels used by this graph.
  var used = initTable[string, bool]()
  for n in m.nodes:
    if n.kind == "micro": used[n.op] = true
  if used.hasKey("script.rmsnorm"):
    outText.add "kernel script.rmsnorm in=2 out=1 threads=256 tg=256 tgbytes=1024 size=in0,in1,out0 tag=aten::rms_norm\n"
    outText.add RmsnormParBody & "\nendkernel\n"
  if used.hasKey("script.add"):
    outText.add "kernel script.add in=2 out=1 threads=out0.size tg=1 tgbytes=16 size=in0,in1,out0 tag=aten::add\n"
    outText.add AddBody & "\nendkernel\n"
  if used.hasKey("script.add_rmsnorm"):
    outText.add "kernel script.add_rmsnorm in=3 out=2 threads=1 tg=1 tgbytes=16 size=in0,in1,in2,out0,out1 tag=aten::add+aten::rms_norm\n"
    outText.add AddRmsnormBody & "\nendkernel\n"
  if used.hasKey("script.silu_mul"):
    outText.add "kernel script.silu_mul in=2 out=1 threads=out0.size tg=1 tgbytes=16 size=in0,in1,out0 tag=aten::silu+aten::mul\n"
    outText.add SiluMulBody & "\nendkernel\n"
  if used.hasKey("script.sigmoid"):
    outText.add "kernel script.sigmoid in=1 out=1 threads=out0.size tg=1 tgbytes=16 size=in0,out0 tag=aten::sigmoid\n"
    outText.add SigmoidBody & "\nendkernel\n"
  if used.hasKey("script.mul"):
    outText.add "kernel script.mul in=2 out=1 threads=out0.size tg=1 tgbytes=16 size=in0,in1,out0 tag=aten::mul\n"
    outText.add MulBody & "\nendkernel\n"
  if hidden != 2048 or nheads != 32 or nkv != 4 or positions != 2048 or headdim != 64:
    raise newException(ValueError, "no library attention kernel for hidden=" & $hidden &
      " heads=" & $nheads & " kv=" & $nkv & " positions=" & $positions & " head_dim=" & $headdim &
      " (the shipped kernel covers hidden=2048 heads=32 kv=4 positions=2048 head_dim=64)")
  outText.add "kernel script.attention in=7 out=4 threads=nkvheads tg=1 tgbytes=16 tag=aten::scaled_dot_product_attention"
  outText.add " expect=in0:hidden,in1:nkvheads*headdim,in2:nkvheads*headdim,in3:1,in4:nkvheads*positions*headdim,in5:nkvheads*positions*headdim,in6:nheads*positions+hidden,out0:hidden,out1:nkvheads*positions*headdim,out2:nkvheads*positions*headdim,out3:nheads*positions+hidden\n"
  outText.add AttentionBody & "\nendkernel\n"
  # Nodes: prefix, repeat body, suffix.
  var prefix, body, suffix: seq[CoreNode]
  var inBody, afterBody = false
  for n in m.nodes:
    if n.tag == "layer":
      body.add n
      inBody = true
    elif not inBody:
      prefix.add n
    else:
      suffix.add n
  proc emitNode(n: CoreNode) =
    outText.add "node " & n.name & " " & n.kind & " " & n.op & " " & n.ins.join(" ") & " -> " & n.outs.join(" ") & "\n"
  for n in prefix: emitNode(n)
  outText.add "repeat layers\n"
  for n in body: emitNode(n)
  outText.add "endrepeat\n"
  for n in suffix: emitNode(n)
  result = outText

proc liftCore*(trace: Trace; configText, modelName, embeddingName: string; useQ4 = false): string =
  let config = parseJson(configText)
  let hidden = cfgInt(config, ["hidden_size", "n_embd", "d_model"])
  let nheads = cfgInt(config, ["num_attention_heads", "n_head"])
  let nkv = cfgInt(config, ["num_key_value_heads", "n_kv_heads"])
  let headdim = hidden div nheads
  let mlp = cfgInt(config, ["intermediate_size", "n_inner"])
  let positions = cfgInt(config, ["max_position_embeddings", "n_positions", "n_ctx"])
  let vocab = cfgInt(config, ["vocab_size"])
  let cfgLayers = cfgInt(config, ["num_hidden_layers", "n_layer", "num_layers"])
  if min(min(hidden, nheads), min(nkv, min(mlp, min(positions, vocab)))) <= 0:
    raise newException(ValueError, "core lift needs a decoder config with hidden_size, heads, kv heads, intermediate_size, positions and vocab_size")
  let step = trace.steps
  var ops: seq[Op]
  for op in trace.ops:
    if op.step == step and op.parent < 0: ops.add op
  var m = CoreModel(q4t: initTable[string, bool]())
  # The first upload of one hidden row is the host embedding.
  var embedBuffer = -1
  for t in trace.transfers:
    if t.kind == "upload" and t.op >= 0 and t.n == hidden:
      for op in ops:
        if op.id == t.op and op.impl == "session:sessionUpload":
          embedBuffer = t.dst.buffer
          break
      if embedBuffer >= 0: break
  if embedBuffer < 0:
    raise newException(ValueError, "core lift found no host upload of one hidden row (embedding output)")
  var embTensor = embeddingName
  if embTensor.len == 0:
    embTensor = if config.hasKey("tie_word_embeddings") and config["tie_word_embeddings"].getBool(false):
      "lm_head.weight" else: "model.embed_tokens.weight"
  m.addBuf("x", hidden)
  m.addWeight("embed_tokens_weight", embTensor, @[vocab, hidden])
  let embeddingNode = CoreNode(kind: "ext", op: "fir.embedding", name: "embedding",
    ins: @["token", "embed_tokens_weight"], outs: @["x"])
  m.addBuf("norm", hidden)
  m.addBuf("q_proj_linear", hidden)
  m.addBuf("k_proj_linear", nkv*headdim)
  m.addBuf("v_proj_linear", nkv*headdim)
  m.addBuf("attn", hidden)
  m.addBuf("o_proj_linear", hidden)
  m.addBuf("residual", hidden)
  m.addBuf("norm2", hidden)
  m.addBuf("gate_proj_linear", mlp)
  m.addBuf("up_proj_linear", mlp)
  m.addBuf("sig", mlp)
  m.addBuf("silu", mlp)
  m.addBuf("act", mlp)
  m.addBuf("down_proj_linear", hidden)
  m.addBuf("key_cache", nkv*positions*headdim, true)
  m.addBuf("value_cache", nkv*positions*headdim, true)
  m.addBuf("scores", nheads*positions + hidden)
  m.addBuf("normed", hidden)
  m.addBuf("logits", vocab, false, "logits")
  var names = initTable[int, string]()
  names[embedBuffer] = "x"
  var bodyNodes: seq[CoreNode]
  var prefixNodes = @[embeddingNode]
  var suffixNodes: seq[CoreNode]
  var layerNodes: seq[seq[CoreNode]]
  var currentLayer = -1
  var seenLayer = false
  var addCount = 0
  var normCount = 0
  var layerComplete = false
  var finalNormEmitted = false
  var finalNormTensor = ""
  var finalNormDims: seq[int]
  for n in trace.names:
    if n.name == "model.norm.weight" or n.name.endsWith(".norm.weight"):
      finalNormTensor = n.name
      finalNormDims = shapeOf(n)
      break
  if finalNormTensor.len == 0:
    finalNormTensor = "model.norm.weight"
    finalNormDims = @[hidden]
  m.addWeight("norm_weight", finalNormTensor, finalNormDims)
  proc emit(n0: CoreNode; layer: int) =
    var n = n0
    n.tag = if layer >= 0: "layer" else: "other"
    if layer >= 0:
      seenLayer = true
      if layerNodes.len <= layer: layerNodes.setLen(layer+1)
      layerNodes[layer].add n
    elif seenLayer:
      suffixNodes.add n
    else:
      prefixNodes.add n
  for op in ops:
    if op.impl in ["session:sessionBegin", "session:sessionEnd", "session:sessionFree",
        "session:sessionAllocScratch", "session:sessionRead", "session:sessionUpload",
        "session:sessionUploadPrivate", "session:sessionUploadBorrowed",
        "session:sessionJitPrewarm", "session:sessionPacesCommandBuffers",
        "session:sessionRope", "session:sessionScale", "session:sessionCopyInto",
        "session:sessionKvScatterEnc", "session:residentPrepareQ4",
        "session:sessionFlashAttentionPrepare", "session:sessionMaxTgThreads"]:
      continue
    var weightTensor = ""
    var weightName = ""
    var weightDims: seq[int]
    if op.impl in ["session:sessionMatmulQ4", "session:sessionMatmulNT",
        "session:sessionMatmul", "session:sessionRmsnormEnc"]:
      let w = findName(trace, op.ins[1])
      weightTensor = w.name
      weightName = shortOf(w.name) & "_weight"
      weightDims = shapeOf(w)
      if weightDims.len == 0 and op.attrs.hasKey("K") and op.attrs.hasKey("N"):
        weightDims = @[parseInt(op.attrs["N"]), parseInt(op.attrs["K"])]
    var layer = -1
    if weightTensor.len > 0: layer = layerOfTensor(weightTensor)
    if layer >= 0:
      if layer != currentLayer:
        currentLayer = layer
        addCount = 0
        normCount = 0
    elif op.impl in ["session:sessionVecOp", "session:sessionActivation",
        "session:sessionAttentionDecodeAllHeadsEnc"]:
      layer = currentLayer
    else:
      layer = -1
    case op.impl
    of "session:sessionRmsnormEnc":
      let x = refMapName(names, op.ins[0], "x")
      var outName = if normCount == 0: "norm" else: "norm2"
      inc normCount
      if weightTensor == "model.norm.weight":
        outName = "normed"
        finalNormEmitted = true
      names[op.outs[0].buffer] = outName
      emit(CoreNode(kind: "micro", op: "script.rmsnorm", name: "rmsnorm_" & $normCount,
        ins: @[x, weightName], outs: @[outName]), layer)
      m.addWeight(weightName, templateOf(weightTensor), weightDims)
      if useQ4 and weightTensor.startsWith("model.layers.") and weightDims.len == 2:
        m.q4t[weightName] = true
    of "session:sessionMatmulQ4", "session:sessionMatmulNT":
      let isHead = weightTensor == "lm_head.weight" or weightTensor.endsWith(".lm_head.weight")
      let a = if isHead and layerComplete: "normed" else: refMapName(names, op.ins[0], "x")
      if isHead and layerComplete and not finalNormEmitted:
        emit(CoreNode(kind: "micro", op: "script.rmsnorm", name: "final_norm",
          ins: @["x", "norm_weight"], outs: @["normed"]), -1)
        finalNormEmitted = true
      let outName = if isHead: "logits" else: shortOf(weightTensor) & "_linear"
      names[op.outs[0].buffer] = outName
      emit(CoreNode(kind: "ext", op: "fir.linear_t",
        name: (if isHead: "head" else: shortOf(weightTensor)),
        ins: @[a, weightName], outs: @[outName]), (if isHead and layerComplete: -1 else: layer))
      m.addWeight(weightName, templateOf(weightTensor), weightDims)
      if useQ4 and weightTensor.startsWith("model.layers.") and weightDims.len == 2:
        m.q4t[weightName] = true
    of "session:sessionVecOp":
      let kind = op.attrs.getOrDefault("op", "add")
      let a = refMapName(names, op.ins[0], "x")
      let b = refMapName(names, op.ins[1], "x")
      var outName = ""
      if kind == "add":
        outName = if addCount == 0: "residual" else: "x"
        inc addCount
      elif kind == "mul":
        let na = names.getOrDefault(op.ins[0].buffer, "")
        let nb = names.getOrDefault(op.ins[1].buffer, "")
        outName = if (na == "sig" and nb == "gate_proj_linear") or (nb == "sig" and na == "gate_proj_linear"): "silu" else: "act"
      else:
        raise newException(ValueError, "core lift does not cover sessionVecOp op=" & kind)
      names[op.outs[0].buffer] = outName
      if not layerComplete or kind == "add":
        emit(CoreNode(kind: "micro", op: "script." & (if kind == "add": "add" else: "mul"),
          name: outName, ins: @[a, b], outs: @[outName]), layer)
      if kind == "add" and layer == cfgLayers-1 and addCount == 2:
        layerComplete = true
    of "session:sessionActivation":
      let kind = op.attrs.getOrDefault("op", "")
      if kind != "sigmoid":
        raise newException(ValueError, "core lift does not cover activation " & kind)
      let a = refMapName(names, op.ins[0], "x")
      names[op.outs[0].buffer] = "sig"
      emit(CoreNode(kind: "micro", op: "script.sigmoid", name: "sigmoid", ins: @[a], outs: @["sig"]), layer)
    of "session:sessionAttentionDecodeAllHeadsEnc":
      let q = names.getOrDefault(op.ins[0].buffer, "q_proj_linear")
      names[op.outs[0].buffer] = "attn"
      emit(CoreNode(kind: "micro", op: "script.attention", name: "attention",
        ins: @[q, "k_proj_linear", "v_proj_linear", "position", "key_cache", "value_cache", "scores"],
        outs: @["attn", "key_cache", "value_cache", "scores"]), layer)
    else:
      raise newException(ValueError, "core lift has no rule for op " & op.impl &
        " (tag " & op.tag & "); add a library kernel instead of dropping it")
  # Validate the layer blocks.
  var layers = 0
  for i, blk in layerNodes:
    if blk.len > 0: layers = i + 1
  if cfgLayers > 0 and layers != cfgLayers:
    raise newException(ValueError, "trace repeats " & $layers & " layers but the config declares " & $cfgLayers)
  if layers < 1: raise newException(ValueError, "core lift found no repeated layer block")
  for i in 1 ..< layers:
    if layerNodes[i].len != layerNodes[0].len:
      var names0, namesI: seq[string]
      for n in layerNodes[0]: names0.add n.op & ":" & n.ins.join("+") & "->" & n.outs.join("+")
      for n in layerNodes[i]: namesI.add n.op & ":" & n.ins.join("+") & "->" & n.outs.join("+")
      raise newException(ValueError, "layer " & $i & " has " & $layerNodes[i].len &
        " nodes, layer 0 has " & $layerNodes[0].len & "\n0: " & names0.join(" | ") &
        "\n" & $i & ": " & namesI.join(" | "))
    for j in 0 ..< layerNodes[0].len:
      if nodeSig(layerNodes[i][j]) != nodeSig(layerNodes[0][j]):
        raise newException(ValueError, "layer " & $i & " differs at node " & layerNodes[0][j].name)
  m.layers = layers
  m.nodes = prefixNodes
  m.nodes.add fuseLayer(layerNodes[0])
  m.nodes.add suffixNodes
  var usedBuffers = initTable[string, bool]()
  for n in m.nodes:
    for b in n.ins & n.outs: usedBuffers[b] = true
  var kept: seq[(string, int, bool, string)]
  for b in m.buffers:
    if usedBuffers.hasKey(b[0]) or b[3] == "logits": kept.add b
  m.buffers = kept
  result = emitCore(m, config, modelName)
