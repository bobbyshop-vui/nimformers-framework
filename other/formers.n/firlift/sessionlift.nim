## Lift a raw core trace (session calls) into a FIR graph made of generated micro-kernels.
##
## Every `session*` call the native runtime records for one step is replaced by the
## micro-kernel `kernel_gen` generates for it (same arguments, same arithmetic as the
## Metal/Vulkan shaders). Buffers are named by the trace buffer they stand for; a write
## creates a fresh FIR buffer except for state that is updated in place (the KV cache).
## The result is the unrolled step: weights come from the registered tensor names,
## host-side per-step data becomes graph inputs (`token`, `position`), nodes whose
## results are never read are dropped.
##
## Scope notes (see docs/fir-script-lift.md):
##  - host uploads are classified by their consumer: the embedding row (the first upload of
##    the step) becomes `fir.embedding`, RoPE cos/sin tables become a position-driven RoPE
##    kernel (frequencies from the config), everything else is host-computed data;
##  - a host round trip (GPU read -> host math -> upload) that follows the layer stack and
##    ends in a head matmul is the final RMSNorm of the native runtime and is lifted as one.
import std/[strutils, tables, json, sets, math, os]
import ./firlift
import ./firvm
import ./kernel_gen

type
  SBuf = object
    name: string
    elements: int
    role: string

  SWeight = object
    name, dtype, tensor: string
    dims: seq[int]

  SKern = object
    name, body, tag: string
    threads, tg, tgBytes, ins, outs: int
    sizes: seq[int]

  SNode = object
    kind, op, name: string
    ins, outs: seq[string]
    rmw: bool            ## the outputs are updated in place: their old contents are inputs too

  SessionLift = ref object
    trace: Trace
    config: JsonNode
    modelName: string
    embeddingName: string
    bufs: seq[SBuf]
    bufIdx: Table[string, int]
    weights: seq[SWeight]
    weightIdx: Table[string, int]
    kernels: seq[SKern]
    kernelIdx: Table[string, int]
    kernelHints: Table[string, int]
    nodes: seq[SNode]
    cur: Table[(int, int), string]       # (trace buffer id, byte offset) -> FIR buffer
    tainted: HashSet[string]
    hidden, positions, vocab: int
    eps: float
    usedPosition: bool
    coverage: bool                       ## lower every op, tolerate unnamed weights and unread inputs
    okCount: CountTable[string]
    failures: seq[string]

const Bookkeeping = [
  "sessionBegin", "sessionEnd", "sessionFree", "sessionAllocScratch", "sessionAllocHeap",
  "sessionAllocFromHeap", "sessionFreeHeap", "sessionJitPrewarm", "sessionPacesCommandBuffers",
  "sessionFlashAttentionPrepare", "sessionMamba2SsdPrepare", "sessionMaxTgThreads",
  "residentPrepareQ4", "residentUploadQ4", "residentUploadQ4Mapped", "residentFreeQ4",
  "sessionWrapHostRange", "sessionUploadPrivatePtr", "sessionCommitContinue",
  "sessionRecoverQueue", "sessionGetLastError", "sessionSetF32GemmTiled",
  "sessionSetFp4GemmTiled", "sessionCooperativeGemmEnabled", "sessionCooperativeGemmJobs",
  "sessionTensorCoreRequired"]

const UploadOps = ["sessionUpload", "sessionUploadPrivate", "sessionUploadBorrowed",
                   "sessionUploadIndices"]

proc fail(msg: string) {.noreturn.} =
  raise newException(ValueError, "session lift: " & msg)

# ── trace helpers ───────────────────────────────────────────────────

proc implName(op: Op): string =
  if op.impl.startsWith("session:"): op.impl[8 .. ^1]
  elif op.impl.startsWith("resident:"): op.impl[9 .. ^1]
  else: op.impl

proc attrInt(op: Op; key: string; default = low(int)): int =
  if op.attrs.hasKey(key):
    try: return parseInt(op.attrs[key])
    except ValueError: discard
  if default == low(int):
    fail("op " & op.implName & " (o" & $op.id & ") has no integer attribute " & key)
  default

proc attrFloat(op: Op; key: string): float =
  if not op.attrs.hasKey(key): fail("op " & op.implName & " has no attribute " & key)
  let v = op.attrs[key]
  if v.len == 9 and v[0] == 'h':
    return float(cast[float32](uint32(parseHexInt(v[1 .. ^1]))))
  parseFloat(v)

proc attrStr(op: Op; key: string): string =
  if not op.attrs.hasKey(key): fail("op " & op.implName & " has no attribute " & key)
  op.attrs[key]

proc port(ports: seq[PortRef]; name: string): PortRef =
  for p in ports:
    if p.name == name: return p
  fail("missing port " & name)

proc key(p: PortRef): (int, int) = (p.buffer, p.offset)

proc cfgInt(c: JsonNode; keys: openArray[string]): int =
  for k in keys:
    if c.hasKey(k) and c[k].kind == JInt: return c[k].getInt()
  -1

proc cfgKey(c: JsonNode; keys: openArray[string]): string =
  for k in keys:
    if c.hasKey(k): return k
  ""

proc shapeOf(n: NameRec): seq[int] =
  if n.shape.len >= 2 and n.shape[0] == '(':
    for part in n.shape[1 .. ^2].splitWhitespace():
      if part.len > 0: result.add parseInt(part)

proc sanitize(s: string): string =
  for c in s:
    result.add (if c in {'a'..'z', 'A'..'Z', '0'..'9', '_'}: c else: '_')

proc findName(t: Trace; p: PortRef): NameRec =
  for n in t.names:
    if n.buffer == p.buffer and n.offset == p.offset and n.bytes == p.bytes: return n
  for n in t.names:
    if n.buffer == p.buffer and n.offset == p.offset: return n
  fail("no registered weight name for buffer b" & $p.buffer & "@" & $p.offset)

proc hasName(t: Trace; p: PortRef): bool =
  for n in t.names:
    if n.buffer == p.buffer and n.offset == p.offset: return true
  false

# ── buffers, weights, kernels ───────────────────────────────────────

proc addBuf(l: SessionLift; name: string; elements: int; role = "") =
  if l.bufIdx.hasKey(name):
    let i = l.bufIdx[name]
    if l.bufs[i].elements != elements:
      fail("buffer " & name & " used with " & $l.bufs[i].elements & " and " & $elements & " elements")
    return
  l.bufIdx[name] = l.bufs.len
  l.bufs.add SBuf(name: name, elements: elements, role: role)

proc rd(l: SessionLift; p: PortRef): string =
  if not l.cur.hasKey(p.key) and l.coverage:
    let name = "s" & $p.buffer
    l.addBuf(name, p.bytes div 4)
    l.cur[p.key] = name
  if not l.cur.hasKey(p.key):
    fail("port " & p.name & " reads buffer b" & $p.buffer & "@" & $p.offset &
      " that no recorded op wrote and that is not a registered weight")
  l.cur[p.key]

proc wr(l: SessionLift; op: Op; p: PortRef; elements = -1): string =
  result = "t" & $op.id & "_" & p.name
  l.addBuf(result, if elements >= 0: elements else: p.bytes div 4)
  l.cur[p.key] = result

proc stateBuf(l: SessionLift; p: PortRef; elements: int): string =
  ## A buffer updated in place that keeps its contents between steps.
  if l.cur.hasKey(p.key): return l.cur[p.key]
  result = "s" & $p.buffer
  l.addBuf(result, elements)
  l.cur[p.key] = result

proc q4Elements(n, k, group: int): int =
  (n*((k+1) div 2) + 3) div 4 + 2*n*((k + group - 1) div group)

proc weightName(l: SessionLift; n: NameRec; dtype: string; dims: seq[int]): string =
  let key = n.name & "|" & dtype
  if l.weightIdx.hasKey(key): return l.weights[l.weightIdx[key]].name
  var name = sanitize(n.name)
  if dtype.startsWith("q4b"): name.add "_q"
  elif dtype == "f32" and dims.len == 2: name.add "_emb"
  var unique = name
  var i = 2
  for w in l.weights:
    if w.name == unique:
      unique = name & "_" & $i
      inc i
  l.weightIdx[key] = l.weights.len
  l.weights.add SWeight(name: unique, dtype: dtype, tensor: n.name, dims: dims)
  unique

proc q4Group(dtype: string): int =
  ## q4b is group 32; q4b<N> is group N.
  if dtype == "q4b": 32 else: parseInt(dtype[3 .. ^1])

proc elementsOf(w: SWeight): int =
  if w.dtype.startsWith("q4b"):
    return q4Elements(w.dims[0], w.dims[1], q4Group(w.dtype))
  result = 1
  for d in w.dims: result *= d

proc rdWeight(l: SessionLift; p: PortRef; dtype: string): string =
  ## A registered weight read by a kernel. f32 vectors and `f32a`/`q4b` matrices live in the arena.
  if l.coverage and not hasName(l.trace, p): return l.rd(p)
  let n = findName(l.trace, p)
  var dims = shapeOf(n)
  if dims.len == 0: fail("weight " & n.name & " has no recorded shape")
  var dt = dtype
  if dt == "f32" and dims.len == 2: dt = "f32a"
  l.weightName(n, dt, dims)

proc sizeOfBuf(l: SessionLift; name: string): int =
  if l.bufIdx.hasKey(name): return l.bufs[l.bufIdx[name]].elements
  for w in l.weights:
    if w.name == name: return w.elementsOf
  fail("unknown buffer " & name)

proc nodeName(l: SessionLift; hint: string): string =
  var n = 0
  for x in l.nodes:
    if x.name.startsWith(hint): inc n
  hint & "_" & $n

proc groupSize(threads: int): int =
  ## Lanes are independent, so any divisor works; wider groups keep the GPU's SIMD lanes busy.
  for c in [64, 32, 16, 8, 4, 2]:
    if threads mod c == 0: return c
  1

proc addKernelNode(l: SessionLift; op: Op; hint: string; g: Gen; ins, outs: seq[string];
                   writtenIns: seq[int] = @[]; tgBytes = 16) =
  ## Register the generated kernel (deduplicated by text and port sizes) and append its node.
  var outAll = outs
  for i in writtenIns: outAll.add ins[i]
  var sizes: seq[int]
  for n in ins & outAll: sizes.add l.sizeOfBuf(n)
  let tag = if op.tag.len > 0 and not op.tag.startsWith("fir::"): op.tag else: ""
  let sig = g.body & "\n" & $g.threads & "|" & $ins.len & "|" & $outAll.len & "|" & sizes.join(",")
  var idx: int
  if l.kernelIdx.hasKey(sig):
    idx = l.kernelIdx[sig]
  else:
    let count = l.kernelHints.getOrDefault(hint, 0) + 1
    l.kernelHints[hint] = count
    idx = l.kernels.len
    l.kernelIdx[sig] = idx
    l.kernels.add SKern(name: "s." & hint & (if count > 1: "_" & $count else: ""),
      body: g.body, tag: tag, threads: g.threads,
      tg: (if g.tg > 0: g.tg else: groupSize(g.threads)),
      tgBytes: (if g.tgBytes > 0: g.tgBytes else: tgBytes),
      ins: ins.len, outs: outAll.len, sizes: sizes)
  l.nodes.add SNode(kind: "micro", op: l.kernels[idx].name, name: l.nodeName(hint),
    ins: ins, outs: outAll)

# ── op lowering ─────────────────────────────────────────────────────

proc positionBuf(l: SessionLift): string =
  l.usedPosition = true
  l.addBuf("position", 1, "position")
  "position"

proc lowerRope(l: SessionLift; op: Op; outScale = 1.0) =
  let x = op.ins.port("x")
  let hn = op.attrInt("Hn")
  let d = op.attrInt("D")
  let tabled = l.cur.getOrDefault(op.ins.port("cosT").key, "") != "ropetable"
  if op.attrInt("T") != 1 or tabled:
    # Tables that the graph itself computed (or several tokens): the table-driven kernel.
    let xn = l.rd(x)
    for nm in ["cosT", "sinT"]:
      let p = op.ins.port(nm)
      if l.cur.getOrDefault(p.key, "") == "ropetable":
        if not l.coverage:
          fail("RoPE with T=" & $op.attrInt("T") & " reads host tables; only a one-token step is liftable")
        l.cur.del p.key
    l.addKernelNode(op, "rope_table", genRope(op.attrInt("T"), hn, d),
      @[xn, l.rd(op.ins.port("cosT")), l.rd(op.ins.port("sinT"))], @[xn])
    return
  if x.bytes div 4 != hn*d: fail("RoPE port size does not match Hn*D")
  var theta = 10000.0
  if l.config.hasKey("rope_theta") and l.config["rope_theta"].kind in {JFloat, JInt}:
    theta = l.config["rope_theta"].getFloat()
  var posScale = 1.0
  if l.config.hasKey("rope_scaling") and l.config["rope_scaling"].kind == JObject:
    let rs = l.config["rope_scaling"]
    let kind = rs.getOrDefault("type").getStr(rs.getOrDefault("rope_type").getStr(""))
    if kind == "linear":
      posScale = 1.0 / rs["factor"].getFloat()
    elif kind notin ["", "default"]:
      fail("rope_scaling type " & kind & " has no generated kernel")
  let g = genRopePos(hn, d, pow(theta, -1.0 / float(d div 2)), posScale, outScale)
  let xn = l.rd(x)
  l.addKernelNode(op, "rope", g, @[xn, l.positionBuf()], @[xn])

proc lowerAttentionDecode(l: SessionLift; op: Op) =
  let h = op.attrInt("H")
  let hkv = op.attrInt("HKV")
  let d = op.attrInt("D")
  let s = op.attrInt("S")
  let pos = l.trace.stepPosition
  if not l.coverage and pos >= 0 and s != pos + 1:
    fail("decode attention length " & $s & " is not position+1 (" & $(pos+1) &
      "); a windowed or truncated cache has no generated kernel")
  if l.positions <= 0: fail("the config has no context length (max_position_embeddings)")
  let cache = hkv*l.positions*d
  let q = l.rd(op.ins.port("q"))
  let kAll = l.stateBuf(op.ins.port("kAll"), cache)
  let vAll = l.stateBuf(op.ins.port("vAll"), cache)
  let o = l.wr(op, op.outs.port("o"))
  l.addKernelNode(op, "attention", genAttentionDecode(h, hkv, d, l.positions, true),
    @[q, kAll, vAll, l.positionBuf()], @[o])

proc lowerCopyGroup(l: SessionLift; ops: seq[Op]; first, last: int) =
  ## A run of sessionCopyInto calls. Copies from one source into one destination that advance by
  ## a fixed stride are the per-head KV-cache write of the decode step and become one scatter at
  ## the traced position; anything else stays a plain offset copy.
  type Copy = object
    op: Op
    srcOff, dstOff, n: int
  var order: seq[(int, int)]
  var groups = initTable[(int, int), seq[Copy]]()
  for i in first .. last:
    let op = ops[i]
    if op.implName != "sessionCopyInto": continue
    let gk = (op.ins.port("src").buffer, op.outs.port("dst").buffer)
    if gk notin groups:
      groups[gk] = @[]
      order.add gk
    groups[gk].add Copy(op: op, srcOff: op.attrInt("srcOffsetElems", 0),
      dstOff: op.attrInt("dstOffsetElems", 0), n: op.attrInt("n"))
  for gk in order:
    let cs = groups[gk]
    var fused = cs.len >= 2
    if fused:
      let n = cs[0].n
      let stride = cs[1].dstOff - cs[0].dstOff
      fused = stride > 0 and stride mod n == 0
      for i, c in cs:
        if c.n != n or c.srcOff != cs[0].srcOff + i*n or c.dstOff != cs[0].dstOff + i*stride:
          fused = false
    if fused:
      let n = cs[0].n
      let hkv = cs.len
      let maxSeq = (cs[1].dstOff - cs[0].dstOff) div n
      let pos = cs[0].dstOff div n
      if cs[0].srcOff != 0: fail("KV scatter source does not start at offset 0")
      if not l.coverage and l.trace.stepPosition >= 0 and pos != l.trace.stepPosition:
        fail("KV write at row " & $pos & " but the step position is " & $l.trace.stepPosition)
      if maxSeq < pos + 1: fail("KV cache stride is smaller than the position")
      if l.positions <= 0: fail("the config has no context length (max_position_embeddings)")
      let op = cs[0].op
      let src = l.rd(op.ins.port("src"))
      let dst = l.stateBuf(op.outs.port("dst"), hkv*l.positions*n)
      l.addKernelNode(op, "kv_scatter", genKvScatter(1, hkv, n, l.positions),
        @[src, l.positionBuf(), dst], @[], @[2])
    else:
      for c in cs:
        let src = l.rd(c.op.ins.port("src"))
        let dstPort = c.op.outs.port("dst")
        let dst = l.stateBuf(dstPort, dstPort.bytes div 4)
        l.addKernelNode(c.op, "copy", genCopyOffset(c.n, c.srcOff, c.dstOff), @[src], @[dst])
        l.nodes[^1].rmw = true

proc parLanes(): int =
  ## Lanes per output column of the threadgroup-parallel matmuls. The default 1 keeps the
  ## lane-per-column kernels: on Intel Iris Plus 640 (Metal) the tree reduction was 25% slower
  ## per token than serial lanes. FIRLIFT_TG=16|32|64 selects the parallel kernels for GPUs
  ## where they win.
  let v = getEnv("FIRLIFT_TG")
  if v.len > 0: parseInt(v) else: 1

proc lowerMatmulQ4(l: SessionLift; op: Op) =
  let m = op.attrInt("M")
  let k = op.attrInt("k")
  let n = op.attrInt("n")
  let group = op.attrInt("group")
  let w = op.ins.port("w")
  let a = l.rd(op.ins.port("a"))
  let rec = (if l.coverage and not hasName(l.trace, w):
               NameRec(name: "coverage.w" & $w.buffer, shape: "(" & $n & " " & $k & ")")
             else: findName(l.trace, w))
  let dims = shapeOf(rec)
  if dims != @[n, k]: fail("Q4 weight " & rec.name & " has shape " & $dims & ", the op is [" & $n & "," & $k & "]")
  let wn = l.weightName(rec, (if group == 32: "q4b" else: "q4b" & $group), dims)
  let c = l.wr(op, op.outs.port("c"))
  l.addKernelNode(op, "matmul_q4", genMatmulQ4Par(m, k, n, group, op.attrInt("groupsPerRow"), parLanes()),
    @[a, wn], @[c])

proc lowerMatmulNT(l: SessionLift; op: Op; bPort = "b") =
  let m = op.attrInt("M")
  let k = op.attrInt("K")
  let n = op.attrInt("N")
  let b = op.ins.port(bPort)
  let off = op.attrInt("bOffsetElems", 0)
  let a = l.rd(op.ins.port("a"))
  var bn: string
  if l.cur.hasKey(b.key): bn = l.cur[b.key]
  else: bn = l.rdWeight(b, "f32a")
  let c = l.wr(op, op.outs.port("c"))
  l.addKernelNode(op, "matmul_nt", (if off == 0: genMatmulNTPar(m, k, n, parLanes()) else: genMatmulNT(m, k, n, off)),
    @[a, bn], @[c])

proc table(l: SessionLift; p: PortRef): seq[int] =
  ## A short host table (permute strides) the recorder kept with its upload.
  for i in countdown(l.trace.transfers.high, 0):
    let t = l.trace.transfers[i]
    if t.kind == "upload" and t.dst.buffer == p.buffer and t.data.len > 0:
      for w in t.data: result.add int(cast[int32](w))
      return
  fail("table " & p.name & " (b" & $p.buffer & ") was not recorded; the table upload is outside the recorded step")

proc lowerOp(l: SessionLift; op: Op) =
  case op.implName
  of "sessionRmsnormEnc":
    let x = l.rd(op.ins.port("x"))
    let gamma = l.rdWeight(op.ins.port("gamma"), "f32")
    let y = l.wr(op, op.outs.port("y"))
    l.addKernelNode(op, "rmsnorm", genRmsnorm(op.attrInt("rows"), op.attrInt("cols"), op.attrFloat("eps")),
      @[x, gamma], @[y])
  of "sessionLayernormEnc":
    let hasBeta = op.ins.len >= 3
    let x = l.rd(op.ins.port("x"))
    var ins = @[x, l.rdWeight(op.ins.port("gamma"), "f32")]
    if hasBeta: ins.add l.rdWeight(op.ins.port("beta"), "f32")
    let y = l.wr(op, op.outs.port("y"))
    l.addKernelNode(op, "layernorm", genLayernorm(op.attrInt("rows"), op.attrInt("cols"),
      op.attrFloat("eps"), hasBeta), ins, @[y])
  of "sessionMatmulQ4":
    l.lowerMatmulQ4(op)
  of "sessionMatmulNT", "sessionMatmulNTAccelerated", "sessionMatmulNTFp16Coop",
     "sessionMatmulNTTiledEnc":
    l.lowerMatmulNT(op, (if op.ins.len >= 2 and op.ins[1].name == "w": "w" else: "b"))
  of "sessionMatmul":
    let a = l.rd(op.ins.port("a"))
    let b = (if l.cur.hasKey(op.ins.port("b").key): l.cur[op.ins.port("b").key]
             else: l.rdWeight(op.ins.port("b"), "f32a"))
    let c = l.wr(op, op.outs.port("c"))
    l.addKernelNode(op, "matmul", genMatmul(op.attrInt("M"), op.attrInt("K"), op.attrInt("N")),
      @[a, b], @[c])
  of "sessionVecOp":
    let a = l.rd(op.ins.port("a"))
    let b = l.rd(op.ins.port("b"))
    let c = l.wr(op, op.outs.port("c"))
    let kind = op.attrStr("op")
    l.addKernelNode(op, "vec_" & kind, genVecOp(kind, op.attrInt("n")), @[a, b], @[c])
  of "sessionActivation":
    let x = l.rd(op.ins.port("x"))
    let y = l.wr(op, op.outs.port("y"))
    let kind = op.attrStr("op")
    let p = (if op.attrs.hasKey("p"): op.attrFloat("p") else: 0.0)
    l.addKernelNode(op, "act_" & kind, genActivation(kind, op.attrInt("n"), p), @[x], @[y])
  of "sessionScale":
    let x = l.rd(op.ins.port("x"))
    l.addKernelNode(op, "scale", genScale(op.attrInt("n"), op.attrFloat("scalar")), @[x], @[x])
  of "sessionRope":
    l.lowerRope(op)
  of "sessionAttentionDecodeAllHeadsEnc":
    l.lowerAttentionDecode(op)
  of "sessionKvScatterEnc":
    let pos = op.attrInt("pos")
    if not l.coverage and l.trace.stepPosition >= 0 and pos != l.trace.stepPosition:
      fail("KV scatter at pos " & $pos & " but the step position is " & $l.trace.stepPosition)
    let hkv = op.attrInt("HKV")
    let d = op.attrInt("D")
    let src = l.rd(op.ins.port("src"))
    let dst = l.stateBuf(op.outs.port("dst"), hkv*l.positions*d)
    l.addKernelNode(op, "kv_scatter", genKvScatter(op.attrInt("T"), hkv, d, l.positions),
      @[src, l.positionBuf(), dst], @[], @[2])
  of "sessionFlashAttentionPrefillEnc":
    let hkv = op.attrInt("HKV")
    let d = op.attrInt("D")
    let q = l.rd(op.ins.port("q"))
    let kAll = l.stateBuf(op.ins.port("kAll"), hkv*l.positions*d)
    let vAll = l.stateBuf(op.ins.port("vAll"), hkv*l.positions*d)
    let o = l.wr(op, op.outs.port("o"))
    l.addKernelNode(op, "flash_prefill", genFlashPrefill(op.attrInt("T"), op.attrInt("H"), hkv, d,
      l.positions, op.attrFloat("scale")), @[q, kAll, vAll, l.positionBuf()], @[o])
  of "sessionSoftmax":
    let x = l.rd(op.ins.port("x"))
    let y = l.wr(op, op.outs.port("y"))
    l.addKernelNode(op, "softmax", genSoftmax(op.attrInt("rows"), op.attrInt("cols")), @[x], @[y])
  of "sessionAttentionFusedEnc":
    let q = l.rd(op.ins.port("q"))
    let k = l.rd(op.ins.port("k"))
    let v = l.rd(op.ins.port("v"))
    let o = l.wr(op, op.outs.port("o"))
    let s = l.wr(op, op.outs.port("sMatrix"))
    l.addKernelNode(op, "attention_fused", genAttentionFused(op.attrInt("B"), op.attrInt("H"),
      op.attrInt("S"), op.attrInt("D"), op.attrFloat("scale")), @[q, k, v], @[o, s])
  of "sessionAttentionDecodeGqaEnc":
    let q = l.rd(op.ins.port("q"))
    let k = l.rd(op.ins.port("k"))
    let v = l.rd(op.ins.port("v"))
    let o = l.wr(op, op.outs.port("o"))
    let s = l.wr(op, op.outs.port("scores"))
    l.addKernelNode(op, "attention_gqa", genAttentionDecodeGqa(op.attrInt("H"), op.attrInt("S"),
      op.attrInt("D"), op.attrFloat("scale")), @[q, k, v], @[o, s])
  of "sessionAttentionFusedBackwardEnc":
    var ins: seq[string]
    for nm in ["q", "k", "v", "sMatrix", "dy"]: ins.add l.rd(op.ins.port(nm))
    var outs: seq[string]
    for nm in ["dq", "dk", "dv"]:
      let p = op.outs.port(nm)
      outs.add l.stateBuf(p, p.bytes div 4)
    # dq/dk/dv accumulate (+=) into their buffers, which are the kernel's outputs.
    l.addKernelNode(op, "attention_fused_backward", genAttentionFusedBackward(op.attrInt("B"),
      op.attrInt("H"), op.attrInt("S"), op.attrInt("D"), op.attrFloat("scale")), ins, outs)
  of "sessionApfluEnc":
    let x = l.rd(op.ins.port("x"))
    let y = l.wr(op, op.outs.port("y"))
    l.addKernelNode(op, "apflu", genApflu(op.attrInt("n"), op.attrFloat("alpha"), op.attrFloat("beta")),
      @[x], @[y])
  of "sessionApfluBackwardEnc":
    let x = l.rd(op.ins.port("x"))
    let dy = l.rd(op.ins.port("dy"))
    let dx = l.wr(op, op.outs.port("dx"))
    l.addKernelNode(op, "apflu_backward", genApfluBackward(op.attrInt("n"), op.attrFloat("alpha"),
      op.attrFloat("beta")), @[x, dy], @[dx])
  of "sessionGatherEnc":
    let src = l.rd(op.ins.port("src"))
    let idx = l.rd(op.ins.port("idx"))
    let dst = l.wr(op, op.outs.port("dst"))
    l.addKernelNode(op, "gather", genGather(op.attrInt("n")), @[src, idx], @[dst])
  of "sessionCompareEnc":
    let a = l.rd(op.ins.port("a"))
    let b = l.rd(op.ins.port("b"))
    let m = l.wr(op, op.outs.port("mask"))
    l.addKernelNode(op, "compare", genCompare(op.attrInt("op"), op.attrInt("n")), @[a, b], @[m])
  of "sessionDropoutMaskEnc":
    let m = l.wr(op, op.outs.port("mask"))
    l.addKernelNode(op, "dropout_mask", genDropoutMask(op.attrInt("n"), op.attrFloat("p"),
      uint32(op.attrInt("seed"))), @[], @[m])
  of "sessionArgmaxIndexEnc":
    let x = l.rd(op.ins.port("x"))
    let v = l.wr(op, op.outs.port("val"))
    let i = l.wr(op, op.outs.port("idx"))
    l.addKernelNode(op, "argmax", genArgmax(op.attrInt("outer"), op.attrInt("dimSize"),
      op.attrInt("inner")), @[x], @[v, i])
  of "sessionSplitSliceEnc":
    let src = l.rd(op.ins.port("src"))
    let dst = l.wr(op, op.outs.port("dst"))
    l.addKernelNode(op, "split_slice", genSplitSlice(op.attrInt("dimTotal"), op.attrInt("thisSize"),
      op.attrInt("offset"), op.attrInt("inner"), op.attrInt("n")), @[src], @[dst])
  of "sessionEmbeddingLookupEnc":
    let table = l.rd(op.ins.port("table"))
    let idx = l.rd(op.ins.port("indices"))
    let y = l.wr(op, op.outs.port("y"))
    l.addKernelNode(op, "embedding", genEmbedding(op.attrInt("dim"), op.attrInt("numIndices")),
      @[table, idx], @[y])
  of "sessionLayernormBackwardEnc":
    let dy = l.rd(op.ins.port("dy"))
    let x = l.rd(op.ins.port("x"))
    let gamma = l.rdWeight(op.ins.port("gamma"), "f32")
    let rows = op.attrInt("rows")
    let cols = op.attrInt("cols")
    let eps = op.attrFloat("eps")
    let dx = l.wr(op, op.outs.port("dx"))
    l.addKernelNode(op, "layernorm_backward_dx", genLayernormBackwardDx(rows, cols, eps),
      @[dy, x, gamma], @[dx])
    let dg = l.wr(op, op.outs.port("dgamma"))
    let db = l.wr(op, op.outs.port("dbeta"))
    l.addKernelNode(op, "layernorm_backward_params", genLayernormBackwardParams(rows, cols, eps),
      @[dy, x], @[dg, db])
  of "sessionMamba2SsdPrefillEnc":
    # The whole SSD pass over one arena: conv rows, dt, A, D, the state slots and the output
    # are regions of that buffer, so the kernel addresses them through offsets of one port.
    let arenaPort = op.ins.port("arena")
    let a = l.rd(arenaPort)
    let g = genMamba2Ssd(op.attrInt("T"), op.attrInt("H"), op.attrInt("G"), op.attrInt("N"),
      op.attrInt("D"), (if op.attrStr("cont") == "true": 1 else: 0))
    let rebased = g.rebasePorts(@[op.attrInt("offConv"), op.attrInt("offDt"), op.attrInt("offALog"),
      op.attrInt("offDSkip"), op.attrInt("offStates")], @[op.attrInt("offYPre")])
    l.addKernelNode(op, "mamba2_ssd", rebased, @[a, a, a, a, a], @[a])
    l.cur[arenaPort.key] = a
  of "sessionPermuteEnc":
    let src = l.rd(op.ins.port("src"))
    let dst = l.wr(op, op.outs.port("dst"))
    let ns = l.table(op.ins.port("newStrides"))
    let os = l.table(op.ins.port("oldStrides"))
    let dm = l.table(op.ins.port("dimsMap"))
    l.addKernelNode(op, "permute", genPermute(ns, os, dm, op.attrInt("n")), @[src], @[dst])
  of "sessionFp4GemmTiledEnc", "sessionFp4GemmTensorCore":
    let m = op.attrInt("M")
    let k = op.attrInt("K")
    let n = op.attrInt("N")
    let a = l.rd(op.ins.port("a"))
    let w = op.ins.port("w")
    let rec = (if l.coverage and not hasName(l.trace, w):
                 NameRec(name: "coverage.fp4" & $w.buffer, shape: "(" & $n & " " & $k & ")")
               else: findName(l.trace, w))
    if shapeOf(rec) != @[n, k]: fail("FP4 weight " & rec.name & " is not [" & $n & "," & $k & "]")
    let wn = l.weightName(rec, "fp4km", @[n, k])
    let c = l.wr(op, op.outs.port("c"))
    l.addKernelNode(op, "fp4_gemm", genFp4Gemm(m, k, n), @[a, wn], @[c])
  else:
    fail("no micro-kernel mapping for " & op.impl & " (tag " & op.tag & "); add a generator")

# ── whole step ──────────────────────────────────────────────────────

proc isBookkeeping(op: Op): bool = op.implName in Bookkeeping

proc transferOf(t: Trace; op: Op): int =
  for i, tr in t.transfers:
    if tr.kind == "upload" and tr.op == op.id: return i
  -1

proc consumerOf(ops: seq[Op]; after: int; buffer: int): (Op, string) =
  ## The first later op that reads `buffer`, and the port it reads it through.
  for j in after+1 ..< ops.len:
    for p in ops[j].ins:
      if p.buffer == buffer: return (ops[j], p.name)
  (nil, "")

proc finalNormWeight(l: SessionLift; ops: seq[Op]): NameRec =
  ## The one registered [hidden] vector called *norm* that no recorded op reads: the host-side final
  ## RMSNorm weight.
  var used = initHashSet[int]()
  for op in ops:
    for p in op.ins: used.incl p.buffer
  var found = 0
  for n in l.trace.names:
    let lower = n.name.toLowerAscii()
    if "layers." in lower or ".h." in lower: continue
    if "norm" notin lower and "ln_f" notin lower: continue
    if shapeOf(n) != @[l.hidden] or n.buffer in used: continue
    result = n
    inc found
  if found != 1:
    fail("the host-side final norm weight is ambiguous or missing (" & $found & " candidates)")

proc nextReal(ops: seq[Op]; i: int): int =
  ## Index of the next op after `i` that is not allocation/bookkeeping, or -1.
  var j = i + 1
  while j < ops.len:
    if not ops[j].isBookkeeping: return j
    inc j
  -1

proc readsLater(ops: seq[Op]; after: int; buffer: int; skip: seq[int] = @[]): bool =
  for j in after+1 ..< ops.len:
    if j in skip: continue
    for p in ops[j].ins:
      if p.buffer == buffer: return true
  false

proc touches(op: Op; buffer: int): bool =
  for p in op.ins & op.outs:
    if p.buffer == buffer: return true
  false

proc tryFuse(l: SessionLift; ops: seq[Op]; i: int; consumed: var HashSet[int]): int =
  ## Peephole fusion of launches the native runtime also keeps separate. Returns the number of
  ## extra ops consumed (0 = no fusion) after lowering the fused kernel.
  let op = ops[i]
  case op.implName
  of "sessionActivation":
    # sigmoid(x) -> y; x * y -> c1; c1 * u -> c2  (SwiGLU gate)
    if op.attrStr("op") != "sigmoid": return 0
    let j1 = nextReal(ops, i)
    if j1 < 0 or ops[j1].implName != "sessionVecOp" or ops[j1].attrStr("op") != "mul": return 0
    let j2 = nextReal(ops, j1)
    if j2 < 0 or ops[j2].implName != "sessionVecOp" or ops[j2].attrStr("op") != "mul": return 0
    let x = op.ins.port("x")
    let y = op.outs.port("y")
    let m1 = ops[j1]
    if not (m1.ins.port("a").buffer == x.buffer and m1.ins.port("b").buffer == y.buffer): return 0
    let c1 = m1.outs.port("c")
    let m2 = ops[j2]
    if m2.ins.port("a").buffer != c1.buffer: return 0
    if readsLater(ops, j2, y.buffer) or readsLater(ops, j2, c1.buffer): return 0
    let xn = l.rd(x)
    let un = l.rd(m2.ins.port("b"))
    let c = l.wr(m2, m2.outs.port("c"))
    l.addKernelNode(op, "silu_mul", genSiluMul(op.attrInt("n")), @[xn, un], @[c])
    consumed.incl j1
    consumed.incl j2
    return 2
  of "sessionVecOp":
    # a + b -> c; rmsnorm(c) -> y
    if op.attrStr("op") != "add": return 0
    let j1 = nextReal(ops, i)
    if j1 < 0 or ops[j1].implName != "sessionRmsnormEnc": return 0
    let nm = ops[j1]
    let c = op.outs.port("c")
    if nm.ins.port("x").buffer != c.buffer or nm.attrInt("rows") != 1: return 0
    if op.attrInt("n") != nm.attrInt("cols"): return 0
    let an = l.rd(op.ins.port("a"))
    let bn = l.rd(op.ins.port("b"))
    let gamma = l.rdWeight(nm.ins.port("gamma"), "f32")
    let sum = l.wr(op, c)
    let y = l.wr(nm, nm.outs.port("y"))
    l.addKernelNode(op, "add_rmsnorm", genAddRmsnorm(nm.attrInt("cols"), nm.attrFloat("eps")),
      @[an, bn, gamma], @[sum, y])
    consumed.incl j1
    return 1
  of "sessionRope":
    # rope(q) ... scale(q): fold the scale into the RoPE store when nothing else touches q between
    if op.attrInt("T") != 1: return 0
    let x = op.ins.port("x")
    var j = nextReal(ops, i)
    var tries = 0
    while j >= 0 and tries < 3:
      if ops[j].implName == "sessionScale" and ops[j].ins.port("x").buffer == x.buffer:
        if l.cur.getOrDefault(op.ins.port("cosT").key, "") != "ropetable": return 0
        l.lowerRope(op, ops[j].attrFloat("scalar"))
        consumed.incl j
        return 1
      if ops[j].touches(x.buffer): return 0
      j = nextReal(ops, j)
      inc tries
    return 0
  else:
    return 0

proc liftSession*(trace: Trace; configText, modelName, embeddingName: string;
                  coverage = false): string =
  ## With `coverage`, every recorded op is lowered on its own (unnamed weights and host inputs
  ## become plain buffers, failures are collected) and the result is a coverage report.
  let l = SessionLift(trace: trace, config: parseJson(configText), modelName: modelName,
    embeddingName: embeddingName, coverage: coverage, okCount: initCountTable[string]())
  l.hidden = cfgInt(l.config, ["hidden_size", "n_embd", "d_model", "dim"])
  l.positions = cfgInt(l.config, ["max_position_embeddings", "n_positions", "n_ctx", "seq_length"])
  l.vocab = cfgInt(l.config, ["vocab_size"])
  var ops: seq[Op]
  for op in trace.ops:
    if op.step == trace.steps and op.parent < 0: ops.add op
  if ops.len == 0: fail("the trace has no recorded step")
  l.addBuf("token", 1, "token")
  for k in ["rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps", "norm_eps"]:
    if l.eps == 0.0 and l.config.hasKey(k) and l.config[k].kind in {JFloat, JInt}:
      l.eps = l.config[k].getFloat()
  var embedded = false
  var afterTaintedRead = false
  var lastCleanRead = ""
  var logits = ""
  var consumed = initHashSet[int]()
  var i = 0
  while i < ops.len:
    let op = ops[i]
    let name = op.implName
    if i in consumed:
      inc i
      continue
    if name in UploadOps:
      let ti = trace.transferOf(op)
      if ti < 0:
        inc i
        continue
      let tr = trace.transfers[ti]
      let (consumer, portName) = ops.consumerOf(i, tr.dst.buffer)
      if consumer != nil and consumer.implName == "sessionRope" and portName in ["cosT", "sinT"]:
        # RoPE tables: the lifted RoPE derives them from `position`.
        l.cur[tr.dst.key] = "ropetable"
      elif consumer != nil and consumer.implName == "sessionPermuteEnc" and
           portName in ["newStrides", "oldStrides", "dimsMap"] and tr.data.len > 0:
        l.cur[tr.dst.key] = "table"
      elif l.coverage:
        let t = "u" & $op.id
        l.addBuf(t, tr.n)
        l.cur[tr.dst.key] = t
      elif afterTaintedRead and lastCleanRead.len > 0:
        # Host result of the final RMSNorm round trip.
        let nw = l.finalNormWeight(ops)
        let gamma = l.weightName(nw, "f32", shapeOf(nw))
        let outName = "t" & $op.id & "_host_norm"
        l.addBuf(outName, tr.n)
        if tr.n != l.hidden: fail("host round trip result has " & $tr.n & " elements, hidden is " & $l.hidden)
        l.cur[tr.dst.key] = outName
        l.addKernelNode(op, "rmsnorm", genRmsnorm(1, l.hidden, l.eps), @[lastCleanRead, gamma], @[outName])
        afterTaintedRead = false
      elif not embedded and trace.stepToken >= 0:
        embedded = true
        if l.hidden > 0 and tr.n != l.hidden: fail("the first upload has " & $tr.n & " elements, hidden is " & $l.hidden)
        var emb = embeddingName
        if emb.len == 0:
          emb = (if l.config.hasKey("tie_word_embeddings") and l.config["tie_word_embeddings"].getBool(false):
                   "lm_head.weight" else: "model.embed_tokens.weight")
        if l.vocab <= 0: fail("the config has no vocab_size")
        let w = l.weightName(NameRec(name: emb), "f32", @[l.vocab, tr.n])
        let outName = "t" & $op.id & "_embedding"
        l.addBuf(outName, tr.n)
        l.cur[tr.dst.key] = outName
        l.nodes.add SNode(kind: "ext", op: "fir.embedding", name: "embedding", ins: @["token", w], outs: @[outName])
      else:
        let t = "host" & $op.id
        l.cur[tr.dst.key] = t
        l.tainted.incl t
      inc i
      continue
    if op.isBookkeeping:
      if name == "sessionBegin": afterTaintedRead = false
      inc i
      continue
    if name == "sessionRead":
      let h = op.ins.port("h")
      let n = op.attrInt("n")
      if not l.cur.hasKey(h.key):
        if coverage:
          inc i
          continue
        fail("sessionRead of a buffer no op wrote")
      let src = l.cur[h.key]
      if src in l.tainted:
        afterTaintedRead = true
      elif l.vocab > 0 and n == l.vocab:
        logits = src
      else:
        lastCleanRead = src
      inc i
      continue
    # Taint: any op reading host-computed data we cannot lift is dropped with its results.
    var taintedIn = false
    for p in op.ins:
      if l.cur.hasKey(p.key) and l.cur[p.key] in l.tainted: taintedIn = true
    if taintedIn:
      for p in op.outs:
        let t = "host" & $op.id & "_" & p.name
        l.cur[p.key] = t
        l.tainted.incl t
      inc i
      continue
    if name == "sessionCopyInto":
      var last = i
      while last + 1 < ops.len and (ops[last+1].implName == "sessionCopyInto" or
          (ops[last+1].isBookkeeping and ops[last+1].implName != "sessionBegin" and
           ops[last+1].implName != "sessionEnd")):
        inc last
      l.lowerCopyGroup(ops, i, last)
      i = last + 1
      continue
    if getEnv("FIRLIFT_NOFUSE").len == 0 and not coverage and name in ["sessionActivation", "sessionVecOp", "sessionRope"]:
      if l.tryFuse(ops, i, consumed) > 0:
        inc i
        continue
    if coverage:
      try:
        l.lowerOp(op)
        l.okCount.inc name
      except ValueError as e:
        l.failures.add name & " (o" & $op.id & "): " & e.msg
    else:
      l.lowerOp(op)
    inc i
  if coverage:
    result = "lowered ops:\n"
    for k, v in l.okCount: result.add "  " & k & " x" & $v & "\n"
    result.add "kernels: " & $l.kernels.len & "\n"
    if l.failures.len > 0:
      result.add "FAILED:\n"
      for f in l.failures: result.add "  " & f & "\n"
    return
  if logits.len == 0:
    fail("no read of a vocab-sized buffer: the step has no logits")
  l.bufs[l.bufIdx[logits]].role = "logits"
  l.addBuf("position", 1, "position")
  # Dead-node elimination from the logits backwards.
  var live = initHashSet[string]()
  live.incl logits
  var keep = newSeq[bool](l.nodes.len)
  for idx in countdown(l.nodes.high, 0):
    let n = l.nodes[idx]
    var needed = false
    for o in n.outs:
      if o in live: needed = true
    if needed:
      keep[idx] = true
      for x in n.ins: live.incl x
      if n.rmw:
        for o in n.outs: live.incl o
  var nodes: seq[SNode]
  for idx, n in l.nodes:
    if keep[idx]: nodes.add n
  # Emit.
  var usedBuf = initHashSet[string]()
  var usedWeight = initHashSet[string]()
  var usedKernel = initHashSet[string]()
  for n in nodes:
    for b in n.ins & n.outs:
      usedBuf.incl b
      usedWeight.incl b
    if n.kind == "micro": usedKernel.incl n.op
  var outText = "fir 1\nmodel " & modelName & "\nconfig config.json\nruntime graph\nlayout script\nscript_abi 1\n"
  let epsKey = l.config.cfgKey(["rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps", "norm_eps"])
  if epsKey.len > 0: outText.add "epsilon " & epsKey & "\n"
  if l.config.hasKey("torch_dtype") and l.config["torch_dtype"].getStr("") in ["bfloat16", "float16"]:
    outText.add "require feature weights_" & (if l.config["torch_dtype"].getStr("") == "bfloat16": "bf16" else: "f16") & "\n"
  proc symLine(alias: string; keys: openArray[string]; fallback: int) =
    let k = l.config.cfgKey(keys)
    if k.len > 0 and l.config[k].kind == JInt: outText.add "sym " & alias & " = " & k & "\n"
    elif fallback > 0: outText.add "sym " & alias & " = " & $fallback & "\n"
  symLine("hidden", ["hidden_size", "n_embd", "d_model", "dim"], l.hidden)
  symLine("layers", ["num_hidden_layers", "n_layer", "num_layers", "n_layers"], 1)
  symLine("positions", ["max_position_embeddings", "n_positions", "n_ctx", "seq_length"], max(l.positions, 1))
  symLine("vocab", ["vocab_size"], l.vocab)
  for b in l.bufs:
    if b.name notin usedBuf and b.role notin ["token", "position", "logits"]: continue
    outText.add "buf " & b.name & " f32 " & $b.elements
    if b.role == "logits": outText.add " output:logits"
    elif b.role.len > 0: outText.add " input:" & b.role
    outText.add "\n"
  for w in l.weights:
    if w.name notin usedWeight: continue
    var dims: seq[string]
    for d in w.dims: dims.add $d
    outText.add "weight " & w.name & " " & w.dtype & " " & w.tensor & " tensor [" & dims.join(",") & "]\n"
  for k in l.kernels:
    if k.name notin usedKernel: continue
    var expect: seq[string]
    for idx in 0 ..< k.ins: expect.add "in" & $idx & ":" & $k.sizes[idx]
    for idx in 0 ..< k.outs: expect.add "out" & $idx & ":" & $k.sizes[k.ins + idx]
    outText.add "kernel " & k.name & " in=" & $k.ins & " out=" & $k.outs & " threads=" & $k.threads &
      " tg=" & $k.tg & " tgbytes=" & $k.tgBytes
    if expect.len > 0: outText.add " expect=" & expect.join(",")
    if k.tag.len > 0: outText.add " tag=" & k.tag
    outText.add "\n" & k.body & "\nendkernel\n"
  var seen = initCountTable[string]()
  for n in nodes:
    seen.inc n.name
    let nm = (if seen[n.name] > 1: n.name & "_" & $seen[n.name] else: n.name)
    outText.add "node " & nm & " " & n.kind & " " & n.op & " " & n.ins.join(" ") & " -> " & n.outs.join(" ") & "\n"
  result = outText
