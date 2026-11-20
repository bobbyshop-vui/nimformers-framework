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
import std/[strutils, tables, json, sets, math, os, algorithm]
import ./firlift
import ./firvm
import ./kernel_gen
import ./kernel_asm
import session_vm
import ./fuse
import ./sidecar
from ./steps import withoutSingleRowUnits, isRopeTableUpload, isUploadedRopeTable, dispatchAccess
import micro_kernel/slot_layout
import probe_weights

type
  SBuf = object
    name: string
    elements: int
    role: string
    base, initFile, tensor, fingerprint: string
    offset, initOffset, initBytes: int
    data: seq[uint32]
    rawWeight: bool

  SWeight = object
    name, dtype, tensor: string
    dims: seq[int]

  SKern = object
    name, body, tag: string
    threads, tg, tgBytes, ins, outs: int
    sizes: seq[int]
    slots: string
    raw: bool
    rowBlock, rowThreads, rows: int   ## row-major launch: every `rowBlock` rows take `rowThreads` threads (0: unknown)
    hiR, hiU: int                     ## highest float / integer register the body uses

  SNode = object
    kind, op, name: string
    ins, outs: seq[string]
    rmw: bool            ## the outputs are updated in place: their old contents are inputs too
    phase: int           ## 0: the selected unit; k >= 1: k-th token-row unit replayed before it

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
    roleBufs: HashSet[int]               ## host scalar-input buffers (token/position roles) of a raw trace
    tainted: HashSet[string]
    hidden, positions, vocab: int
    eps: float
    usedPosition: bool
    coverage: bool                       ## lower every op, tolerate unnamed weights and unread inputs
    okCount: CountTable[string]
    failures: seq[string]
    raw: bool
    phaseRows: Table[int, int]           ## phase number -> token rows that phase consumes per run
    phaseLogits: Table[int, (string, int)] ## phase number -> (buffer, rows) of vocabulary-sized output rows
    phaseLast: Table[int, string]          ## phase number -> buffer with the logits after the phase's last row
    fp4Blobs: HashSet[(int, int)]          ## (phase, trace buffer) of raw FP4-KM weights whose decode table that phase already appended

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
  if l.raw:
    if p.buffer in l.roleBufs:
      if not l.cur.hasKey(p.key):
        fail("host input buffer b" & $p.buffer & " is read at @" & $p.offset & ", which has no token/position role")
      return l.cur[p.key]
    let base = "b" & $p.buffer
    if not l.bufIdx.hasKey(base): fail("missing raw buffer " & base)
    let size = if p.bytes > 0: p.bytes else: l.bufs[l.bufIdx[base]].elements*4-p.offset
    if p.offset == 0 and size == l.bufs[l.bufIdx[base]].elements*4: return base
    if p.offset mod 4 != 0 or size mod 4 != 0 or size <= 0:
      fail("unaligned or empty slot range " & base & "@" & $p.offset & ":" & $size)
    let view = base & "_" & $p.offset & "_" & $size
    if not l.bufIdx.hasKey(view):
      l.addBuf(view, size div 4)
      l.bufs[l.bufIdx[view]].base = base
      l.bufs[l.bufIdx[view]].offset = p.offset div 4
    return view
  if not l.cur.hasKey(p.key) and l.coverage:
    let name = "s" & $p.buffer
    l.addBuf(name, p.bytes div 4)
    l.cur[p.key] = name
  if not l.cur.hasKey(p.key):
    fail("port " & p.name & " reads buffer b" & $p.buffer & "@" & $p.offset &
      " that no recorded op wrote and that is not a registered weight")
  l.cur[p.key]

proc wr(l: SessionLift; op: Op; p: PortRef; elements = -1): string =
  if l.raw: return l.rd(p)
  result = "t" & $op.id & "_" & p.name
  l.addBuf(result, if elements >= 0: elements else: p.bytes div 4)
  l.cur[p.key] = result

proc stateBuf(l: SessionLift; p: PortRef; elements: int): string =
  ## A buffer updated in place that keeps its contents between steps.
  if l.raw: return l.rd(p)
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
  elif dtype.startsWith("q4r"): name.add "_qr"
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
  ## q4b/q4r default to group 32; the suffix selects the group size.
  let prefix = if dtype.startsWith("q4r"): "q4r" else: "q4b"
  if dtype == prefix: 32 else: parseInt(dtype[prefix.len .. ^1])

proc elementsOf(w: SWeight): int =
  if w.dtype.startsWith("q4b") or w.dtype.startsWith("q4r"):
    return q4Elements(w.dims[0], w.dims[1], q4Group(w.dtype))
  result = 1
  for d in w.dims: result *= d

proc rdWeight(l: SessionLift; p: PortRef; dtype: string): string =
  ## A registered weight read by a kernel. f32 vectors and `f32a`/`q4b` matrices live in the arena.
  if l.raw: return l.rd(p)
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
                   writtenIns: seq[int] = @[]; tgBytes = 16; rowBlock = 0; rows = 0; residentQ4 = false) =
  ## Register the generated kernel (deduplicated by text and port sizes) and append its node.
  ## rowBlock/rows: the kernel works on `rows` rows and its threads are ordered by row, `rowBlock`
  ## rows at a time, each thread touching only its own rows (and state of earlier rows). The runner
  ## may then launch only the threads of the first rows of a phase (`rows=B:T` on the kernel line).
  var rb = 0
  var rt = 0
  if rowBlock > 0 and rows > 0:
    let blocks = (rows + rowBlock - 1) div rowBlock
    let tgSize = (if g.tg > 0: g.tg else: groupSize(g.threads))
    if g.threads mod blocks == 0 and (g.threads div blocks) mod tgSize == 0:
      rb = rowBlock
      rt = g.threads div blocks
  var outAll = outs
  for i in writtenIns: outAll.add ins[i]
  var sizes: seq[int]
  for n in ins & outAll: sizes.add l.sizeOfBuf(n)
  var body = g.body
  var slots = ""
  if residentQ4:
    if ins.len != 2 or outAll.len != 1:
      fail("resident Q4 kernel needs activation, weight and one output")
    var inSizes, outSizes: seq[int]
    for name in ins: inSizes.add l.sizeOfBuf(name)
    for name in outAll: outSizes.add l.sizeOfBuf(name)
    let program = assemble(g.body, ins.len, outAll.len,
      inSizes, outSizes, l.eps.float32, slotLayout = true)
    body = "words"
    for word in program.words: body.add " " & toHex(word, 8).toLowerAscii
    slots = "mem:out0,u8:in1,lut:in1,input:in0"
  let tag = if op.tag.len > 0 and not op.tag.startsWith("fir::"): op.tag else: ""
  let sig = body & "\n" & slots & "|" & $g.threads & "|" & $ins.len & "|" & $outAll.len & "|" & sizes.join(",") &
    "|" & $rb & ":" & $rt & ":" & $rows
  var idx: int
  if l.kernelIdx.hasKey(sig):
    idx = l.kernelIdx[sig]
  else:
    let count = l.kernelHints.getOrDefault(hint, 0) + 1
    l.kernelHints[hint] = count
    idx = l.kernels.len
    l.kernelIdx[sig] = idx
    l.kernels.add SKern(name: "s." & hint & (if count > 1: "_" & $count else: ""),
      body: body, tag: tag, threads: g.threads,
      tg: (if g.tg > 0: g.tg else: groupSize(g.threads)),
      tgBytes: (if g.tgBytes > 0: g.tgBytes else: tgBytes),
      ins: ins.len, outs: outAll.len, sizes: sizes, rowBlock: rb, rowThreads: rt,
      slots: slots, raw: residentQ4, rows: (if rb > 0: rows else: 0), hiR: kernel_gen.genHiR, hiU: kernel_gen.genHiU)
  l.nodes.add SNode(kind: "micro", op: l.kernels[idx].name, name: l.nodeName(hint),
    ins: ins, outs: outAll)

proc q4TensorCoreKernel(k: SKern): bool =
  ## A shape-op Q4 micro-kernel (Tensor Core GEMM or subgroup GEMV): one instruction the
  ## Vulkan JIT lowers whole, so it is never merged into a chain of raw kernels.
  let words = k.body.splitWhitespace()
  if words.len < 2 or words[0] != "words": return false
  try:
    result = (uint32(parseHexInt(words[1])) and 0xFF'u32) in [OP_Q4_COOP_GEMM, OP_Q4_GEMV, OP_FP4_COOP_GEMM]
  except ValueError:
    result = false

# ── op lowering ─────────────────────────────────────────────────────

proc unitRows(l: SessionLift; op: Op; n: int): int =
  ## Token rows of the unit `op` belongs to when its `n` elements split evenly over them (else 0):
  ## an elementwise kernel over n elements is then row-major with n / rows threads per row.
  let rows = (if op.phase > 0: l.phaseRows.getOrDefault(op.phase, 0) else: 1)
  if rows > 0 and n mod rows == 0: rows else: 0

proc positionBuf(l: SessionLift): string =
  l.usedPosition = true
  l.addBuf("position", 1, "position")
  "position"

proc genericKernels(): bool

proc ropeKernel(l: SessionLift; rows, hn, d: int; outScale: float): Gen =
  ## RoPE derived from the position (theta and scaling from the config), optionally scaling the output.
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
  let ratio = pow(theta, -1.0 / float(d div 2))
  let half = d div 2
  result = if not genericKernels() and half >= 2 and (half and (half - 1)) == 0 and rows * hn * half <= (1 shl 20):   # thread ids stay exact in float
             genRopePosPar(rows, hn, d, ratio, posScale, outScale)
           elif rows == 1: genRopePos(hn, d, ratio, posScale, outScale)
           else: genRopePosRows(rows, hn, d, ratio, posScale, outScale)

proc appleTarget(): bool =
  ## FIRLIFT_TARGET=apple: the graph is for an Apple GPU, so GEMMs may use simdgroup_matrix
  ## (Intel and AMD Metal devices cannot create those pipelines).
  getEnv("FIRLIFT_TARGET") == "apple" and getEnv("FIRLIFT_GENERIC").len == 0

proc genericKernels(): bool =
  ## FIRLIFT_GENERIC=1: lane-per-output kernels that every VM backend can run. The default kernels
  ## follow the Metal ones (threadgroups, simdgroups) and need simd ops.
  getEnv("FIRLIFT_GENERIC").len > 0

proc traceHasTensorCore(l: SessionLift): bool =
  ## The recorded device runs the Tensor Core GEMM micro-kernels: a Vulkan device with
  ## KHR_cooperative_matrix ("coopmatrix") or an Apple GPU with MSL simdgroup_matrix ("simdmatrix").
  (l.trace.backend == "vulkan" and l.trace.device == "coopmatrix") or
    (l.trace.backend == "metal" and l.trace.device == "simdmatrix")

proc simdKernels(l: SessionLift): bool =
  ## Whether the decode matmuls may reduce with simdgroup ops. Those kernels follow Metal: a
  ## simdgroup is a contiguous run of at least 16 threads of the grid. Vulkan promises neither for
  ## its subgroups (the width is the driver's choice, down to 1), and on MoltenVK the simdgroup Q4
  ## GEMV returned wrong sums, so a graph lifted from a Vulkan run reduces through threadgroup
  ## memory instead. FIRLIFT_SIMD=0|1 overrides.
  let v = getEnv("FIRLIFT_SIMD")
  if v.len > 0: return v != "0"
  not genericKernels() and l.trace.backend != "vulkan"

proc lowerRope(l: SessionLift; op: Op; outScale = 1.0) =
  let x = op.ins.port("x")
  let hn = op.attrInt("Hn")
  let d = op.attrInt("D")
  let uploadedCos = l.raw and l.trace.isUploadedRopeTable(op, "cosT")
  let uploadedSin = l.raw and l.trace.isUploadedRopeTable(op, "sinT")
  if uploadedCos != uploadedSin:
    fail("RoPE cos/sin inputs mix uploaded and graph-computed tables")
  let tabled = not uploadedCos and l.cur.getOrDefault(op.ins.port("cosT").key, "") != "ropetable"
  let rows = op.attrInt("T")
  if tabled:
    # Tables that the graph itself computed: the table-driven kernel.
    let xn = l.rd(x)
    for nm in ["cosT", "sinT"]:
      let p = op.ins.port(nm)
      if l.cur.getOrDefault(p.key, "") == "ropetable":
        if not l.coverage:
          fail("RoPE tables are neither graph-computed nor position-derived")
        l.cur.del p.key
    l.addKernelNode(op, "rope_table", genRope(op.attrInt("T"), hn, d),
      @[xn, l.rd(op.ins.port("cosT")), l.rd(op.ins.port("sinT"))], @[xn])
    return
  if x.bytes div 4 != rows*hn*d: fail("RoPE port size does not match T*Hn*D")
  let g = l.ropeKernel(rows, hn, d, outScale)
  let xn = l.rd(x)
  # one thread per (row, head, frequency) is row-major; the lane-per-head kernels are not
  l.addKernelNode(op, "rope", g, @[xn, l.positionBuf()], @[xn],
    rowBlock = (if g.threads == rows * hn * (d div 2): 1 else: 0), rows = rows)

proc kvScatterKernel(t, hkv, d, positions: int): Gen =
  if genericKernels(): genKvScatter(t, hkv, d, positions)
  else: genKvScatterPar(t, hkv, d, positions)

proc lowerKvScatterOne(l: SessionLift; op: Op; src: string; dstPort: PortRef; hkv, d: int) =
  let dst = l.stateBuf(dstPort, hkv*l.positions*d)
  l.addKernelNode(op, "kv_scatter", kvScatterKernel(1, hkv, d, l.positions),
    @[src, l.positionBuf(), dst], @[], @[2])

proc checkKvPosition(l: SessionLift; op: Op) =
  let pos = op.attrInt("pos")
  if not l.coverage and op.phase == 0 and l.trace.stepPosition >= 0 and pos != l.trace.stepPosition:
    fail("KV append at pos " & $pos & " but the step position is " & $l.trace.stepPosition)
  if l.positions <= 0: fail("the config has no context length (max_position_embeddings)")

proc lowerKvAppend(l: SessionLift; op: Op) =
  ## sessionKvAppendEnc: K and V of one token into the combined caches.
  l.checkKvPosition(op)
  let hkv = op.attrInt("hkv")
  let d = op.attrInt("d")
  l.lowerKvScatterOne(op, l.rd(op.ins.port("k")), op.outs.port("kAll"), hkv, d)
  l.lowerKvScatterOne(op, l.rd(op.ins.port("v")), op.outs.port("vAll"), hkv, d)

proc lowerRopeKv(l: SessionLift; op: Op) =
  ## sessionRopeKvEnc: RoPE and scale on Q, RoPE on K, then K and V into the caches.
  l.checkKvPosition(op)
  let h = op.attrInt("h")
  let hkv = op.attrInt("hkv")
  let d = op.attrInt("d")
  let q = l.rd(op.ins.port("q"))
  l.addKernelNode(op, "rope", l.ropeKernel(1, h, d, op.attrFloat("scale")), @[q, l.positionBuf()], @[q])
  let k = l.rd(op.ins.port("k"))
  l.addKernelNode(op, "rope", l.ropeKernel(1, hkv, d, 1.0), @[k, l.positionBuf()], @[k])
  l.lowerKvScatterOne(op, k, op.outs.port("kAll"), hkv, d)
  l.lowerKvScatterOne(op, l.rd(op.ins.port("v")), op.outs.port("vAll"), hkv, d)

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
  l.addKernelNode(op, "attention", (if not genericKernels(): genAttentionDecodePar(h, hkv, d, l.positions)
                                    else: genAttentionDecode(h, hkv, d, l.positions, true)),
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
      if not l.coverage and cs[0].op.phase == 0 and l.trace.stepPosition >= 0 and pos != l.trace.stepPosition:
        fail("KV write at row " & $pos & " but the step position is " & $l.trace.stepPosition)
      if maxSeq < pos + 1: fail("KV cache stride is smaller than the position")
      if l.positions <= 0: fail("the config has no context length (max_position_embeddings)")
      let op = cs[0].op
      let src = l.rd(op.ins.port("src"))
      let dst = l.stateBuf(op.outs.port("dst"), hkv*l.positions*n)
      l.addKernelNode(op, "kv_scatter", kvScatterKernel(1, hkv, n, l.positions),
        @[src, l.positionBuf(), dst], @[], @[2])
    else:
      for c in cs:
        let src = l.rd(c.op.ins.port("src"))
        let dstPort = c.op.outs.port("dst")
        let dst = l.stateBuf(dstPort, dstPort.bytes div 4)
        let rows = (if c.op.phase > 0: l.phaseRows.getOrDefault(c.op.phase, 0) else: 0)
        if rows > 1 and c.dstOff == 0 and c.srcOff == (rows - 1) * c.n and l.sizeOfBuf(src) == rows * c.n and
            l.sizeOfBuf(dst) == c.n:
          # The last row of a multi-row unit (what the head reads): the runner takes the last row it
          # really ran, so the unit also works over fewer rows.
          l.nodes.add SNode(kind: "session", name: l.nodeName("last_row"), ins: @[src], outs: @[dst],
            op: "sessionCopyInto:srcOff=0,dstOff=0,n=" & $c.n & ",lastRowOf=" & $rows)
          continue
        l.addKernelNode(c.op, "copy", genCopyOffset(c.n, c.srcOff, c.dstOff), @[src], @[dst])
        l.nodes[^1].rmw = true

proc parLanes(): int =
  ## Lanes per output column of the threadgroup-parallel matmuls. The default 1 keeps the
  ## lane-per-column kernels: on Intel Iris Plus 640 (Metal) the tree reduction was 25% slower
  ## per token than serial lanes. FIRLIFT_TG=16|32|64 selects the parallel kernels for GPUs
  ## where they win.
  let v = getEnv("FIRLIFT_TG")
  if v.len > 0: parseInt(v) else: 1

proc tunerKnob(name: string; default: int; allowed: openArray[int]): int =
  ## Tuner variant knobs of the simdgroup Q4 GEMV (rows per simdgroup, simdgroups per threadgroup,
  ## 8-weight packs per lane step), set per device through the environment when lifting.
  let v = getEnv(name)
  if v.len == 0: return default
  result = parseInt(v)
  if result notin allowed: fail(name & "=" & v & " is not one of " & $allowed)

proc gemmTile(): (int, int, int, int) =
  ## Tile of the tiled Q4 prefill GEMM, set per device when lifting: the output tile of a
  ## threadgroup (FIRLIFT_GEMM_TILE=32x64|64x64|32x128|64x128, default 64x128) and the register tile
  ## of a thread (FIRLIFT_GEMM_REG=4x4|8x4|4x8|8x8, default 8x8). A threadgroup has
  ## rows/regRows * columns/regColumns threads. The Metal variants are 32x64 with 4x4, 64x64 with
  ## 8x4, 32x128 with 4x8 and 64x128 with 8x8 (128 threads each). The defaults match native Metal's
  ## variant 3; the environment variables remain available for device-specific tuning.
  let tile = getEnv("FIRLIFT_GEMM_TILE", "64x128").split('x')
  let reg = getEnv("FIRLIFT_GEMM_REG", "8x8").split('x')
  if tile.len != 2 or tile[0] notin ["32", "64"] or tile[1] notin ["64", "128"]:
    fail("FIRLIFT_GEMM_TILE is not one of 32x64, 64x64, 32x128, 64x128")
  if reg.len != 2 or reg[0] notin ["4", "8"] or reg[1] notin ["4", "8"]:
    fail("FIRLIFT_GEMM_REG is not one of 4x4, 8x4, 4x8, 8x8")
  (parseInt(tile[0]), parseInt(tile[1]), parseInt(reg[0]), parseInt(reg[1]))

proc q4Choice(name, fallback: string; allowed: openArray[string]): string =
  result = getEnv(name, fallback).toLowerAscii()
  if result notin allowed:
    fail(name & "=" & result & " is not one of " & allowed.join(", "))

proc q4ParallelLanes(): int =
  let value = getEnv("FIRLIFT_Q4_LANES", getEnv("FIRLIFT_TG", "32"))
  result = parseInt(value)
  if result notin [16, 32, 64]:
    fail("FIRLIFT_Q4_LANES must be 16, 32, or 64")

proc q4GemmFma(m, k, n, group, nGroups: int): tuple[gen: Gen, rowBlock, rowTg: int] =
  let tile = gemmTile()
  result.rowBlock = tile[0]
  result.rowTg = (tile[0] div tile[2]) * (tile[1] div tile[3])
  result.gen = genMatmulQ4Gemm(m, k, n, group, nGroups, tile[0], tile[1], tile[2], tile[3], true)
  if result.gen.tg != result.rowTg: result.rowBlock = 0

proc q4MatrixGemm(m, k, n, group, nGroups: int): tuple[gen: Gen, rowBlock, rowTg: int] =
  result.rowBlock = 32
  result.rowTg = 128
  result.gen = genMatmulQ4Mm(m, k, n, group, nGroups, true)
  if result.gen.tg != result.rowTg: result.rowBlock = 0

proc q4TensorCoreShape(m, k, n, group, nGroups: int): bool =
  ## The shapes genMatmulQ4TensorCore accepts; auto selection falls back to the other kernels
  ## for the rest instead of failing the lift.
  m > 0 and n >= 16 and k > 0 and k mod 64 == 0 and group >= 2 and group mod 2 == 0 and
    k mod group == 0 and nGroups == k div group

proc q4TensorCoreGemm(m, k, n, group, nGroups: int): tuple[gen: Gen, rowBlock, rowTg: int] =
  result.rowBlock = q4CoopRowTile(m, n)
  result.rowTg = ((n + 63) div 64) * 128
  result.gen = genMatmulQ4TensorCore(m, k, n, group, nGroups)
  if result.gen.tg != 128: result.rowBlock = 0

proc q4SimdGemv(k, n, group, nGroups: int): tuple[gen: Gen, variant: string] =
  let rows = tunerKnob("FIRLIFT_Q4_ROWS", 4, [1, 2, 4, 8])
  let sg = tunerKnob("FIRLIFT_Q4_SG", 2, [1, 2, 4, 8])
  let packs = tunerKnob("FIRLIFT_Q4_PACKS", 1, [1, 2])
  result.gen = genMatmulQ4Simd(k, n, group, nGroups, rows, sg, packs, true)
  let gs = if group > 0: group else: k
  let supported = k mod (8 * packs) == 0 and k mod gs == 0 and (gs and (gs - 1)) == 0 and
    n mod rows == 0 and gs >= 8 * packs
  result.variant = if supported: "simd"
    elif result.gen.tg == 0: "scalar"
    else: "parallel"

proc q4ParallelVariant(gen: Gen; useFma: bool): string =
  if gen.tg == 0:
    if useFma: "scalar_fma" else: "scalar"
  elif useFma: "parallel_fma"
  else: "parallel"

proc q4WordParallel(k, n, group, nGroups: int; maxK = high(int)): tuple[gen: Gen, variant: string] =
  ## The word-load kernel needs whole 32-bit Q4 words within each quantization group.
  ## Keep the established byte-load reduction for shapes it cannot represent.
  let lanes = q4ParallelLanes()
  if n > 0 and k <= maxK and supportsQ4WordPar(k, group, nGroups, lanes):
    result.gen = genMatmulQ4WordPar(k, n, group, nGroups, lanes, true)
    result.variant = "word_parallel"
  else:
    result.gen = genMatmulQ4Par(1, k, n, group, nGroups, q4ParallelLanes(), false, true)
    result.variant = q4ParallelVariant(result.gen, false)

proc q4SubgroupGemv(l: SessionLift; m, k, n, group, nGroups: int): tuple[gen: Gen, variant: string] =
  ## Vulkan decode GEMV with the native decode kernel's algorithm, as a FIR shape-op micro-kernel.
  if l.trace.backend != "vulkan":
    fail("FIRLIFT_Q4_GEMV=subgroup requires a Vulkan trace; this kernel is lowered by the Vulkan JIT")
  if not q4GemvSupported(k, group, nGroups):
    fail("FIRLIFT_Q4_GEMV=subgroup needs K and the group size divisible by 8 (K=" & $k & ", group=" & $group & ")")
  result.gen = genMatmulQ4SubgroupMicro(m, k, n, group, nGroups)
  result.variant = "subgroup_micro"

proc q4WordAutoMaxK(): int =
  ## The word kernel wins on K=4096 on the tested Iris Vulkan device, but loses on K=11008.
  ## Keep auto conservative; a device can tune this threshold, or set it to 0 to disable it.
  result = parseInt(getEnv("FIRLIFT_Q4_WORD_MAX_K", "4096"))
  if result < 0: fail("FIRLIFT_Q4_WORD_MAX_K must be non-negative")

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
  let wn = l.weightName(rec, (if group == 32: "q4r" else: "q4r" & $group), dims)
  let c = l.wr(op, op.outs.port("c"))
  var rowBlock = 0
  var rowTg = 0
  let nGroups = op.attrInt("groupsPerRow")
  var gen: Gen
  var variant = ""
  if m == 1:
    let choice = q4Choice("FIRLIFT_Q4_GEMV", "auto", ["auto", "scalar", "scalar_fma", "fma", "parallel", "parallel_fma", "simd", "word_parallel", "subgroup", "subgroup_micro"])
    case choice
    of "scalar":
      gen = genMatmulQ4(m, k, n, group, nGroups, 0, true, false, true)
      variant = "scalar"
    of "scalar_fma", "fma":
      gen = genMatmulQ4(m, k, n, group, nGroups, 0, true, true, true)
      variant = "scalar_fma"
    of "parallel":
      gen = genMatmulQ4Par(m, k, n, group, nGroups, q4ParallelLanes(), false, true)
      variant = q4ParallelVariant(gen, false)
    of "parallel_fma":
      gen = genMatmulQ4Par(m, k, n, group, nGroups, q4ParallelLanes(), true, true)
      variant = q4ParallelVariant(gen, true)
    of "simd":
      let simd = q4SimdGemv(k, n, group, nGroups)
      gen = simd.gen; variant = simd.variant
    of "word_parallel":
      let word = q4WordParallel(k, n, group, nGroups)
      gen = word.gen; variant = word.variant
    of "subgroup", "subgroup_micro":
      let sg = l.q4SubgroupGemv(m, k, n, group, nGroups)
      gen = sg.gen; variant = sg.variant
    else:
      if genericKernels():
        gen = genMatmulQ4Par(m, k, n, group, nGroups, parLanes(), false, true)
        variant = q4ParallelVariant(gen, false)
      elif l.simdKernels():
        let simd = q4SimdGemv(k, n, group, nGroups)
        gen = simd.gen; variant = simd.variant
      else:
        if l.trace.backend == "vulkan" and q4GemvSupported(k, group, nGroups):
          # The native decode algorithm as a FIR micro-kernel: the fastest Vulkan GEMV.
          let sg = l.q4SubgroupGemv(m, k, n, group, nGroups)
          gen = sg.gen; variant = sg.variant
        elif l.trace.backend == "vulkan":
          let word = q4WordParallel(k, n, group, nGroups, q4WordAutoMaxK())
          gen = word.gen; variant = word.variant
        else:
          gen = genMatmulQ4Par(m, k, n, group, nGroups, q4ParallelLanes(), false, true)
          variant = q4ParallelVariant(gen, false)
  else:
    let choice = q4Choice("FIRLIFT_Q4_GEMM", "auto", ["auto", "scalar", "scalar_fma", "parallel", "parallel_fma", "fma", "tiled", "tiled_fma", "matrix", "mm", "tensor_core", "coop"])
    case choice
    of "scalar":
      gen = genMatmulQ4(m, k, n, group, nGroups, 0, true, false, true)
      variant = "scalar"
    of "scalar_fma":
      gen = genMatmulQ4(m, k, n, group, nGroups, 0, true, true, true)
      variant = "scalar_fma"
    of "parallel":
      gen = genMatmulQ4Par(m, k, n, group, nGroups, q4ParallelLanes(), false, true)
      variant = q4ParallelVariant(gen, false)
    of "parallel_fma":
      gen = genMatmulQ4Par(m, k, n, group, nGroups, q4ParallelLanes(), true, true)
      variant = q4ParallelVariant(gen, true)
    of "fma", "tiled", "tiled_fma":
      let tiled = q4GemmFma(m, k, n, group, nGroups)
      gen = tiled.gen; rowBlock = tiled.rowBlock; rowTg = tiled.rowTg
      variant = (if gen.tg == 128: "tiled_fma" else: q4ParallelVariant(gen, true))
    of "matrix", "mm":
      if not appleTarget():
        fail("FIRLIFT_Q4_GEMM=matrix requires FIRLIFT_TARGET=apple with FIRLIFT_GENERIC unset (simdgroup_matrix is Metal-only)")
      let matrix = q4MatrixGemm(m, k, n, group, nGroups)
      gen = matrix.gen; rowBlock = matrix.rowBlock; rowTg = matrix.rowTg
      variant = (if gen.tg == 128: "matrix" else: q4ParallelVariant(gen, false))
    of "tensor_core", "coop":
      if l.trace.backend notin ["vulkan", "metal"]:
        fail("FIRLIFT_Q4_GEMM=tensor_core requires a Vulkan or Metal trace; the FIR runtime lowers this kernel to KHR_cooperative_matrix (Vulkan) or simdgroup_matrix (Apple GPUs)")
      let tensorCore = q4TensorCoreGemm(m, k, n, group, nGroups)
      gen = tensorCore.gen; rowBlock = tensorCore.rowBlock; rowTg = tensorCore.rowTg
      variant = "tensor_core_micro"
    else:
      if appleTarget():
        let matrix = q4MatrixGemm(m, k, n, group, nGroups)
        gen = matrix.gen; rowBlock = matrix.rowBlock; rowTg = matrix.rowTg
        variant = (if gen.tg == 128: "matrix" else: q4ParallelVariant(gen, false))
      elif l.traceHasTensorCore() and not genericKernels() and q4TensorCoreShape(m, k, n, group, nGroups):
        # The recorded device has KHR_cooperative_matrix (Vulkan) or simdgroup_matrix (Apple GPU)
        # and the shape fits the kernel: the prefill GEMMs run on the matrix units.
        let tensorCore = q4TensorCoreGemm(m, k, n, group, nGroups)
        gen = tensorCore.gen; rowBlock = tensorCore.rowBlock; rowTg = tensorCore.rowTg
        variant = "tensor_core_micro"
      elif m >= 16 and not genericKernels():
        let tiled = q4GemmFma(m, k, n, group, nGroups)
        gen = tiled.gen; rowBlock = tiled.rowBlock; rowTg = tiled.rowTg
        variant = (if gen.tg == 128: "tiled_fma" else: q4ParallelVariant(gen, true))
      else:
        gen = genMatmulQ4Par(m, k, n, group, nGroups, parLanes(), false, true)
        variant = q4ParallelVariant(gen, false)
  if gen.tg != rowTg: rowBlock = 0   # a fallback uses no row-tile launch metadata
  l.addKernelNode(op, "matmul_q4_" & variant, gen, @[a, wn], @[c], rowBlock = rowBlock, rows = m, residentQ4 = true)

const Fp4GemmOps = ["sessionFp4GemmTiledEnc", "sessionFp4GemmTensorCore"]

proc fp4TensorCoreGemm(m, k, n: int): tuple[gen: Gen, rowBlock, rowTg: int] =
  ## Same launch shape as the Q4 Tensor Core kernel: 128-thread workgroups ordered by row tile,
  ## one per 64 output columns.
  result.rowBlock = q4CoopRowTile(m, n)
  result.rowTg = ((n + 63) div 64) * 128
  result.gen = genFp4GemmTensorCore(m, k, n)
  if result.gen.tg != 128: result.rowBlock = 0

proc fp4UsesTensorCore(l: SessionLift; k, n: int): bool =
  ## FIRLIFT_FP4_GEMM=auto|coop|generic. auto picks the Tensor Core micro-kernel when the trace
  ## comes from a Vulkan device with KHR_cooperative_matrix or an Apple GPU with simdgroup_matrix
  ## and the shape is supported; coop forces it (the graph then needs such a device); generic keeps
  ## the lane-per-column kernel.
  let choice = q4Choice("FIRLIFT_FP4_GEMM", "auto", ["auto", "coop", "tensor_core", "generic"])
  let supported = fp4CoopSupported(k, n, 32, k div 32)
  case choice
  of "coop", "tensor_core":
    if l.trace.backend notin ["vulkan", "metal"]:
      fail("FIRLIFT_FP4_GEMM=coop requires a Vulkan or Metal trace; the FIR runtime lowers this kernel to KHR_cooperative_matrix (Vulkan) or simdgroup_matrix (Apple GPUs)")
    if not supported:
      fail("FIRLIFT_FP4_GEMM=coop needs K divisible by 64 and N >= 16 (K=" & $k & ", N=" & $n & ")")
    true
  of "generic": false
  else:
    l.traceHasTensorCore() and not genericKernels() and supported

proc fp4RawBlob(l: SessionLift; op: Op; w, lut: PortRef; k, n: int): string =
  ## A raw trace keeps an FP4-KM weight as two recorded buffers: the packed bytes and the 272-float
  ## decode table. The FIR kernels read one blob (fp4kmLayout: the table after the packed bytes),
  ## so the weight buffer is extended by the table and the phase copies the table into that tail
  ## before its first GEMM on that weight (272 floats per weight and phase run).
  if w.offset != 0: fail("FP4 weight b" & $w.buffer & " starts at byte " & $w.offset & "; the blob needs its own buffer")
  let base = "b" & $w.buffer
  if not l.bufIdx.hasKey(base): fail("missing raw FP4 weight buffer " & base)
  let idx = l.bufIdx[base]
  let lutWord = fp4CoopLutWord(k, n)
  if w.bytes > 0 and w.bytes < lutWord * 4:
    fail("FP4 weight " & base & " has " & $w.bytes & " bytes, the packed [" & $n & "," & $k & "] weight needs " & $(lutWord * 4))
  if l.bufs[idx].elements < lutWord + 272:
    l.bufs[idx].elements = lutWord + 272
  l.bufs[idx].rawWeight = false   # the graph writes the table tail
  let key = (op.phase, w.buffer)
  if key notin l.fp4Blobs:
    l.fp4Blobs.incl key
    l.nodes.add SNode(kind: "session", name: l.nodeName("fp4_table"), ins: @[l.rd(lut)], outs: @[base],
      op: "sessionCopyInto:srcOff=0,dstOff=" & $lutWord & ",n=272", rmw: true)
  base

proc lowerFp4Gemm(l: SessionLift; op: Op) =
  let m = op.attrInt("M")
  let k = op.attrInt("K")
  let n = op.attrInt("N")
  let a = l.rd(op.ins.port("a"))
  let w = op.ins.port("w")
  var wn: string
  if l.raw:
    wn = l.fp4RawBlob(op, w, op.ins.port("lut"), k, n)
  else:
    let rec = (if l.coverage and not hasName(l.trace, w):
                 NameRec(name: "coverage.fp4" & $w.buffer, shape: "(" & $n & " " & $k & ")")
               else: findName(l.trace, w))
    if shapeOf(rec) != @[n, k]: fail("FP4 weight " & rec.name & " is not [" & $n & "," & $k & "]")
    wn = l.weightName(rec, "fp4km", @[n, k])
  let c = l.wr(op, op.outs.port("c"))
  if l.fp4UsesTensorCore(k, n):
    let tensorCore = fp4TensorCoreGemm(m, k, n)
    l.addKernelNode(op, "fp4_gemm_tensor_core_micro", tensorCore.gen, @[a, wn], @[c],
      rowBlock = tensorCore.rowBlock, rows = m, residentQ4 = true)
  else:
    l.addKernelNode(op, "fp4_gemm_generic", genFp4Gemm(m, k, n), @[a, wn], @[c])

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
  let g =
    if off != 0: genMatmulNT(m, k, n, off)
    elif genericKernels(): genMatmulNTPar(m, k, n, parLanes())
    elif m == 1 and l.simdKernels(): genMatmulNTSimd(k, n)
    elif appleTarget(): genMatmulNTMm(m, k, n)
    else: genMatmulNTPar(m, k, n, parLanes())
  l.addKernelNode(op, "matmul_nt", g, @[a, bn], @[c])

proc table(l: SessionLift; p: PortRef): seq[int] =
  ## A short host table (permute strides) the recorder kept with its upload.
  for i in countdown(l.trace.transfers.high, 0):
    let t = l.trace.transfers[i]
    if t.kind == "upload" and t.dst.buffer == p.buffer and t.data.len > 0:
      for w in t.data: result.add int(cast[int32](w))
      return
  fail("table " & p.name & " (b" & $p.buffer & ") was not recorded; the table upload is outside the recorded step")

const ExactSessionOps = ["sessionRmsnormEnc", "sessionVecOp", "sessionActivation", "sessionCopyInto",
  "sessionFp4GemmMetal", "sessionF16GemmTile", "sessionMatmulNT", "sessionMatmulNTAccelerated"]

proc lowerSessionExact(l: SessionLift; op: Op) =
  ## Raw replay of a session API call. The runner stages the operands in dedicated
  ## buffers and calls the same backend op, so the result is bit-identical to the
  ## recorded runtime (generated VM kernels only approximate these shaders).
  var ins, outs: seq[string]
  var attrs = ""
  case op.implName
  of "sessionRmsnormEnc":
    ins = @[l.rd(op.ins.port("x")), l.rdWeight(op.ins.port("gamma"), "f32")]
    outs = @[l.rd(op.outs.port("y"))]
    attrs = "rows=" & $op.attrInt("rows") & ",cols=" & $op.attrInt("cols") &
      ",eps=" & op.attrs.getOrDefault("eps", "h0")
  of "sessionVecOp":
    ins = @[l.rd(op.ins.port("a")), l.rd(op.ins.port("b"))]
    outs = @[l.rd(op.outs.port("c"))]
    attrs = "op=" & op.attrStr("op") & ",n=" & $op.attrInt("n")
  of "sessionActivation":
    ins = @[l.rd(op.ins.port("x"))]
    outs = @[l.rd(op.outs.port("y"))]
    attrs = "op=" & op.attrStr("op") & ",n=" & $op.attrInt("n") &
      ",p=" & op.attrs.getOrDefault("p", "h0")
  of "sessionCopyInto":
    let src = op.ins.port("src")
    var srcOff = op.attrInt("srcOffsetElems", 0)
    if src.buffer in l.roleBufs:
      # A host scalar buffer: each role (token / position) is its own one-element graph input.
      var role = src
      role.offset = src.offset + srcOff*4
      ins = @[l.rd(role)]
      srcOff = 0
    else:
      ins = @[l.rd(src)]
    outs = @[l.rd(op.outs.port("dst"))]
    attrs = "srcOff=" & $srcOff &
      ",dstOff=" & $op.attrInt("dstOffsetElems", 0) & ",n=" & $op.attrInt("n")
  of "sessionFp4GemmMetal":
    # The packed FP4-KM prefill GEMM of the Metal backend (a native shader, no micro-VM program):
    # the runner calls the same op on the graph's buffers.
    ins = @[l.rd(op.ins.port("a")), l.rd(op.ins.port("w")), l.rd(op.ins.port("lut"))]
    outs = @[l.rd(op.outs.port("c"))]
    attrs = "M=" & $op.attrInt("M") & ",K=" & $op.attrInt("K") & ",N=" & $op.attrInt("N") &
      ",simdgroup=" & (if op.attrStr("simdgroup") == "true": "1" else: "0")
  of "sessionF16GemmTile":
    # Replay the same FP16 sidecar GEMM, including its cooperative-matrix choice.
    ins = @[l.rd(op.ins.port("a")), l.rd(op.ins.port("w"))]
    outs = @[l.rd(op.outs.port("c"))]
    attrs = "M=" & $op.attrInt("M") & ",K=" & $op.attrInt("K") & ",N=" & $op.attrInt("N") &
      ",tensorCore=" & (if op.attrStr("tensorCore") == "true": "1" else: "0")
  of "sessionMatmulNT", "sessionMatmulNTAccelerated":
    # The FP32 GEMM C = A @ B^T of the backend (a native shader; on an FP16 cooperative-matrix
    # device the accelerated call rounds its inputs to FP16): the runner calls the same op.
    let bPort = (if op.ins.len >= 2 and op.ins[1].name == "w": "w" else: "b")
    ins = @[l.rd(op.ins.port("a")), l.rd(op.ins.port(bPort))]
    outs = @[l.rd(op.outs.port("c"))]
    attrs = "M=" & $op.attrInt("M") & ",K=" & $op.attrInt("K") & ",N=" & $op.attrInt("N") &
      ",bOff=" & $op.attrInt("bOffsetElems", 0)
  else:
    fail("no exact session replay for " & op.implName)
  let label = op.implName[7 .. ^1]
  l.nodes.add SNode(kind: "session", op: op.implName & ":" & attrs,
    name: "session_" & label & "_" & $(-1-op.id), ins: ins, outs: outs, rmw: true)

proc lowerOp(l: SessionLift; op: Op) =
  if l.raw and not l.coverage and op.implName in ExactSessionOps:
    l.lowerSessionExact(op)
    return
  case op.implName
  of "raw:dispatch":
    var ins, outs, slots: seq[string]
    for p in op.ins:
      slots.add p.name & ":in" & $ins.len
      ins.add l.rd(p)
    for p in op.outs: outs.add l.rd(p)
    var body = "words"
    for w in op.words: body.add " " & toHex(w, 8).toLowerAscii
    let sig = body & slots.join(",") & "|" & $op.attrInt("threads") & ":" & $op.attrInt("tg") & ":" & $op.attrInt("tgbytes", 0)
    var idx = l.kernelIdx.getOrDefault(sig, -1)
    if idx < 0:
      idx = l.kernels.len
      l.kernelIdx[sig] = idx
      l.kernels.add SKern(name: "raw" & $idx, body: body, threads: op.attrInt("threads"),
        tg: op.attrInt("tg"), tgBytes: op.attrInt("tgbytes", 0), ins: ins.len,
        outs: outs.len, slots: slots.join(","), raw: true)
    l.nodes.add SNode(kind: "micro", op: l.kernels[idx].name, name: "dispatch" & $(-1-op.id),
      ins: ins, outs: outs, rmw: true)
  of "sessionRmsnormEnc":
    let x = l.rd(op.ins.port("x"))
    let gamma = l.rdWeight(op.ins.port("gamma"), "f32")
    let y = l.wr(op, op.outs.port("y"))
    l.addKernelNode(op, "rmsnorm", (if genericKernels(): genRmsnorm(op.attrInt("rows"), op.attrInt("cols"), op.attrFloat("eps"))
                       else: genRmsnormPar(op.attrInt("rows"), op.attrInt("cols"), op.attrFloat("eps"))),
      @[x, gamma], @[y], rowBlock = 1, rows = op.attrInt("rows"))
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
    l.addKernelNode(op, "vec_" & kind, genVecOp(kind, op.attrInt("n")), @[a, b], @[c],
      rowBlock = 1, rows = l.unitRows(op, op.attrInt("n")))
  of "sessionActivation":
    let x = l.rd(op.ins.port("x"))
    let y = l.wr(op, op.outs.port("y"))
    let kind = op.attrStr("op")
    let p = (if op.attrs.hasKey("p"): op.attrFloat("p") else: 0.0)
    l.addKernelNode(op, "act_" & kind, genActivation(kind, op.attrInt("n"), p), @[x], @[y],
      rowBlock = 1, rows = l.unitRows(op, op.attrInt("n")))
  of "sessionScale":
    let x = l.rd(op.ins.port("x"))
    l.addKernelNode(op, "scale", genScale(op.attrInt("n"), op.attrFloat("scalar")), @[x], @[x],
      rowBlock = 1, rows = l.unitRows(op, op.attrInt("n")))
  of "sessionRope":
    l.lowerRope(op)
  of "sessionKvAppendEnc":
    l.lowerKvAppend(op)
  of "sessionRopeKvEnc":
    l.lowerRopeKv(op)
  of "sessionAttentionDecodeAllHeadsEnc":
    l.lowerAttentionDecode(op)
  of "sessionKvScatterEnc":
    let pos = op.attrInt("pos")
    if not l.coverage and op.phase == 0 and l.trace.stepPosition >= 0 and pos != l.trace.stepPosition:
      fail("KV scatter at pos " & $pos & " but the step position is " & $l.trace.stepPosition)
    let hkv = op.attrInt("HKV")
    let d = op.attrInt("D")
    let src = l.rd(op.ins.port("src"))
    let dst = l.stateBuf(op.outs.port("dst"), hkv*l.positions*d)
    let scatter = kvScatterKernel(op.attrInt("T"), hkv, d, l.positions)
    # one thread per element is row-major; the lane-per-head kernel is not
    l.addKernelNode(op, "kv_scatter", scatter, @[src, l.positionBuf(), dst], @[], @[2],
      rowBlock = (if scatter.threads == op.attrInt("T") * hkv * d: 1 else: 0), rows = op.attrInt("T"))
  of "sessionFlashAttentionPrefillEnc":
    let hkv = op.attrInt("HKV")
    let d = op.attrInt("D")
    let q = l.rd(op.ins.port("q"))
    let kAll = l.stateBuf(op.ins.port("kAll"), hkv*l.positions*d)
    let vAll = l.stateBuf(op.ins.port("vAll"), hkv*l.positions*d)
    let o = l.wr(op, op.outs.port("o"))
    let flash = (if genericKernels(): genFlashPrefill(op.attrInt("T"), op.attrInt("H"), hkv, d, l.positions, op.attrFloat("scale"))
                 elif getEnv("FIRLIFT_FLASH") == "row":
                   genAttentionDecodePar(op.attrInt("H"), hkv, d, l.positions, 128, op.attrInt("T"), op.attrFloat("scale"))
                 else: genFlashPrefillTiled(op.attrInt("T"), op.attrInt("H"), hkv, d, l.positions, op.attrFloat("scale")))
    # threadgroups ordered by row (one per (row, head)) or by 16-row tile (one per (tile, head)) are
    # row-major and causal: row t reads keys 0 .. position + t
    let flashRows = op.attrInt("T")
    let flashBlock = (if flash.tg <= 0: 0
                      elif flash.threads == flashRows * op.attrInt("H") * flash.tg: 1
                      elif flash.threads == ((flashRows + 15) div 16) * op.attrInt("H") * flash.tg: 16
                      else: 0)
    l.addKernelNode(op, "flash_prefill", flash, @[q, kAll, vAll, l.positionBuf()], @[o],
      rowBlock = flashBlock, rows = flashRows)
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
    l.lowerFp4Gemm(op)
  else:
    fail("no micro-kernel mapping for " & op.impl & " (tag " & op.tag & "); add a generator")

# ── whole step ──────────────────────────────────────────────────────

proc isBookkeeping(op: Op): bool = op.implName in Bookkeeping

proc returnedFalse(op: Op): bool =
  ## A session call that reported it did not run (an unsupported route): it changed nothing.
  op.impl.startsWith("session:") and op.attrs.getOrDefault("result") == "false"

const OptionalGemmOps = ["sessionFp4GemmTiledEnc", "sessionFp4GemmTensorCore", "sessionF16GemmTile", "sessionMatmulNTFp16Coop",
  "sessionMatmulNTAccelerated", "sessionMatmulNTTiledEnc"]
  ## Session GEMMs that report false on a device without their route; the caller then runs a fallback.

proc dispatchWrites(op: Op; buffer: int): bool =
  let access = dispatchAccess(op)
  for p in op.bindings:
    if p.buffer == buffer and p.name in access.writes: return true
  false

proc dispatchReads(op: Op; buffer: int): bool =
  let access = dispatchAccess(op)
  for p in op.bindings:
    if p.buffer == buffer and p.name in access.reads: return true
  false

proc gemmPorts(op: Op): (int, int) =
  ## (weight buffer, output buffer) of a session GEMM; -1 when absent.
  result = (-1, -1)
  for p in op.ins:
    if p.name in ["w", "b"]: result[0] = p.buffer
  for p in op.outs:
    if p.name == "c": result[1] = p.buffer

proc gemmFallback(ops: seq[Op]; i: int): seq[int] =
  ## The raw dispatches right after the optional GEMM ops[i] (bookkeeping aside) that compute its
  ## result again: they read its weight and write its output buffer. The native caller only does
  ## that when the session call did not run (the recorder does not keep a call's return value), so
  ## such a call is a no-op and these dispatches are what really computed the product.
  let (weight, output) = ops[i].gemmPorts
  if weight < 0 or output < 0: return
  var j = i + 1
  while j < ops.len:
    let o = ops[j]
    if o.isBookkeeping:
      inc j
      continue
    if o.implName in OptionalGemmOps and o.gemmPorts == (weight, output):
      # A second optional route tried for the same product (a caller may try one route per
      # branch before its fallback): no-op too when the fallback follows.
      inc j
      continue
    if o.impl != "raw:dispatch" or o.words.len == 0 or not (o.dispatchReads(weight) and o.dispatchWrites(output)):
      break
    result.add j
    inc j

var replacedFp4Fallbacks* = 0   ## failed FP4 GEMMs of the last lift lowered in place of their native fallback
var droppedFailedCalls* = 0     ## session calls of the last lift that did not run (returned false)

proc withoutFailedCalls(l: SessionLift; ops: seq[Op]): seq[Op] =
  ## Drops session calls that did not run: a recorded false result, or an optional GEMM whose
  ## result the native fallback (packed micro-VM tiles) computes right after it. When the lifted
  ## graph should run such an FP4 GEMM on the Tensor Cores (FIRLIFT_FP4_GEMM), the GEMM is
  ## lowered instead and its fallback dispatches are dropped.
  var dropped = initHashSet[int]()
  for i, op in ops:
    if op.isBookkeeping or not op.impl.startsWith("session:"): continue
    var fallback: seq[int]
    if op.implName in OptionalGemmOps: fallback = gemmFallback(ops, i)
    if fallback.len == 0 and not op.returnedFalse: continue
    if op.implName in Fp4GemmOps and fallback.len > 0 and l.fp4UsesTensorCore(op.attrInt("K"), op.attrInt("N")):
      inc replacedFp4Fallbacks
      for j in fallback: dropped.incl j
      continue
    inc droppedFailedCalls
    dropped.incl i
  for i, op in ops:
    if i notin dropped: result.add op

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
    l.addKernelNode(op, "silu_mul", genSiluMul(op.attrInt("n")), @[xn, un], @[c],
      rowBlock = 1, rows = l.unitRows(op, op.attrInt("n")))
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
    l.addKernelNode(op, "add_rmsnorm", (if genericKernels(): genAddRmsnorm(nm.attrInt("cols"), nm.attrFloat("eps"))
                          else: genAddRmsnormPar(nm.attrInt("cols"), nm.attrFloat("eps"))),
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

proc captureAllZero(l: SessionLift; tr: TransferRec): bool =
  ## True when the recorded upload bytes are present and all zero (a host-cleared scratch).
  if tr.captureOffset < 0 or tr.captureBytes <= 0 or tr.captureBytes != tr.dst.bytes: return false
  sidecarAllZero(l.trace.sourcePath, tr.captureOffset, tr.captureBytes)

var droppedSingleRowUnits* = 0   ## single-token units left out of the last lift
var reusedBuffers* = 0           ## temporaries of the last lift that took over an earlier buffer

proc prepareRaw(l: SessionLift; ops: seq[Op]; weightsPaths: seq[string]) =
  var sizes = initTable[int, int]()
  var firstUse = initTable[int, int]()
  var firstPhase = initTable[int, int]()
  var written = initHashSet[int]()
  var seenWrites = initHashSet[int]()
  var needsInitial = initHashSet[int]()
  var unknown: seq[string]
  for op in ops:
    if op.impl == "raw:dispatch":
      let access = dispatchAccess(op)
      op.ins = @[]
      op.outs = @[]
      for domain in ["mem", "u8", "lut", "input", "u32"]:
        var synchronizationSlot = false
        if domain == "u32":
          for p in op.bindings:
            if p.name == domain: synchronizationSlot = true
        if domain notin access.reads and domain notin access.writes and not synchronizationSlot: continue
        var found = false
        var binding: PortRef
        for p in op.bindings:
          if p.name == domain: binding = p; found = true
        # The original CPU/single-arena API aliases its other domains to mem.
        if not found and l.trace.backend == "cpu" and
            (domain != "input" or op.attrInt("input", 0) == 0):
          for p in op.bindings:
            if p.name == "mem": binding = p; binding.name = domain; found = true
        if not found: fail("dispatch d" & $(-1-op.id) & " missing binding " & domain)
        if binding.bytes <= 0 and binding.buffer >= 0 and binding.buffer < l.trace.bufs.len:
          binding.bytes = max(0, l.trace.bufs[binding.buffer].bytes-binding.offset)
        if binding.bytes <= 0:
          if l.coverage:
            binding.bytes = 4
          else:
            unknown.add "d" & $(-1-op.id) & " " & domain & "=b" & $binding.buffer & " has zero recorded bytes"
        op.ins.add binding
        if domain in access.writes or synchronizationSlot: op.outs.add binding
    if op.isBookkeeping or op.implName in UploadOps or op.implName == "sessionRead": continue
    for p in op.ins & op.outs:
      let extent = if p.bytes > 0: p.offset+p.bytes
                   elif p.buffer >= 0 and p.buffer < l.trace.bufs.len: l.trace.bufs[p.buffer].bytes
                   else: 0
      sizes[p.buffer] = max(sizes.getOrDefault(p.buffer), extent)
      if not firstUse.hasKey(p.buffer):
        firstUse[p.buffer] = op.order
        firstPhase[p.buffer] = op.phase
    let access = if op.impl == "raw:dispatch": dispatchAccess(op) else: VmAccess()
    for p in op.ins:
      if op.impl == "raw:dispatch" and p.name notin access.reads: continue
      if p.buffer notin seenWrites: needsInitial.incl p.buffer
    for p in op.outs: written.incl p.buffer
    for p in op.outs: seenWrites.incl p.buffer
  if unknown.len > 0:
    fail("raw capture is incomplete: " & $unknown.len & " required bindings have no byte extent; first: " &
      unknown[0] & ". Re-record with the core upload recorder and BYBY_FIR_PROBE_DATA=1; checkpoint size/order cannot recover unnamed zero-sized buffers")
  let index = block:
    var combined: seq[ProbeTensor]
    for path in weightsPaths:
      if path.len > 0: combined.add probeTensorIndex(path)
    combined
  let primary = if weightsPaths.len > 0: weightsPaths[0] else: ""
  let first = ops[0].order
  var perStep = initTable[int, TransferRec]()
  for tr in l.trace.transfers:
    if tr.kind == "upload":
      if tr.order >= first and tr.order <= ops[^1].order and firstUse.hasKey(tr.dst.buffer) and tr.order < firstUse[tr.dst.buffer]:
        perStep[tr.dst.buffer] = tr
      for op in ops:
        if op.id == tr.op and op.implName in UploadOps: perStep[tr.dst.buffer] = tr
  # A RoPE table can contain exactly hidden_size floats. Its semantic input
  # port, not its size, distinguishes it from token rows. Key this by upload
  # operation so allocator reuse does not give a buffer a permanent role.
  var ropeTableUploads = initHashSet[int]()
  for op in l.trace.ops:
    if op.implName notin UploadOps: continue
    let ti = l.trace.transferOf(op)
    if ti >= 0 and l.trace.isRopeTableUpload(op, l.trace.transfers[ti].dst.buffer):
      ropeTableUploads.incl op.id
  var embeddings: seq[int]
  var zeroUploads = initHashSet[int]()
  var embeddingPhase = initTable[int, int]()
  var phaseCount = initCountTable[int]()
  var embeddingCandidates: seq[string]
  for id, tr in perStep:
    if not l.coverage and l.captureAllZero(tr): zeroUploads.incl id
  for id, tr in perStep:
    if not l.coverage and sizes.hasKey(id) and id notin zeroUploads and l.hidden > 0 and
        tr.n >= l.hidden and tr.n mod l.hidden == 0 and
        tr.op notin ropeTableUploads:
      embeddings.add id
      let phase = min(1, firstPhase.getOrDefault(id))
      phaseCount.inc phase
      embeddingCandidates.add "b" & $id & " n=" & $tr.n & " phase=" & $phase
  if phaseCount.getOrDefault(0) > 1:
    fail("several per-step token-row uploads in the selected unit (" & embeddingCandidates.join("; ") &
      "); choose a slice with one embedding input")
  # Every token-row upload of the replayed ops starts a unit of its own (any row count, any
  # number of units). Ops before the first upload belong to the first unit.
  var starts: seq[(int, int)]
  for id in embeddings:
    if firstPhase.getOrDefault(id) > 0: starts.add (firstUse[id], id)
  starts.sort()
  for k, start in starts: embeddingPhase[start[1]] = k+1
  for op in ops:
    if op.phase == 0: continue
    var k = 0
    for j, start in starts:
      if start[0] <= op.order: k = j
    op.phase = k+1
  # Host scalar inputs of a driver (token / position): the recorder tagged their offsets with
  # roles, so they become the graph's role inputs instead of frozen uploads.
  for r in l.trace.roles:
    if r.buffer < 0 or r.role notin ["token", "position"] or not sizes.hasKey(r.buffer): continue
    if r.bytes != 4: fail("role " & r.role & " on b" & $r.buffer & " covers " & $r.bytes & " bytes, expected one f32")
    l.roleBufs.incl r.buffer
    l.cur[(r.buffer, r.offset)] = (if r.role == "position": l.positionBuf() else: "token")
  var ids: seq[int]
  for id in sizes.keys: ids.add id
  ids.sort()
  for id in ids:
    if id in l.roleBufs: continue
    var bytes = sizes[id]
    if id >= 0 and id < l.trace.bufs.len and l.trace.bufs[id].bytes > bytes:
      bytes = l.trace.bufs[id].bytes
    if bytes <= 0 and l.coverage: bytes = 4
    # A mapped checkpoint view can start at any byte (NIMQ2): the arena is rounded up to whole words.
    if bytes > 0 and bytes mod 4 != 0: bytes = (bytes + 3) div 4 * 4
    if bytes <= 0: fail("raw buffer b" & $id & " has invalid byte size " & $bytes)
    let name = "b" & $id
    l.addBuf(name, bytes div 4)
    l.cur[(id, 0)] = name
    if l.coverage: continue # Structural coverage needs ports/programs, not checkpoint payloads.
    if id in embeddings: continue
    var upload = TransferRec(captureOffset: -1)
    var found = false
    for tr in l.trace.transfers:
      if tr.kind == "upload" and tr.dst.buffer == id and tr.dst.offset == 0 and
          tr.order <= firstUse[id] and tr.dst.bytes > 0 and tr.dst.bytes <= bytes:
        upload = tr; found = true
    if perStep.hasKey(id): upload = perStep[id]; found = true
    var b = l.bufs[l.bufIdx[name]]
    b.rawWeight = id notin written
    if found:
      let match = if index.len > 0 and b.rawWeight:
          matchProbeTensor(index, primary, upload.dst.bytes, upload.fingerprint) else: -1
      if perStep.hasKey(id) and l.trace.selectionSession > 0 and id notin zeroUploads and
          perStep[id].op notin ropeTableUploads:
        var constantWeight = match >= 0 and upload.fingerprint.len > 0
        for transfer in l.trace.transfers:
          if transfer.kind == "upload" and transfer.dst.buffer == id and transfer.fingerprint.len > 0 and
              transfer.fingerprint != upload.fingerprint:
            constantWeight = false
        if not constantWeight:
          var consumer = "none"
          for op in ops:
            if op.order < upload.order: continue
            var reads = false
            for p in op.ins:
              if p.buffer == id: reads = true
            if reads:
              consumer = op.implName & (if op.impl == "raw:dispatch": " pipe=" & op.attrs.getOrDefault("pipe") else: "") &
                " (o" & $op.id & ")"
              break
          fail("per-step host upload b" & $id & " @" & $upload.order & " (replayed ops start @" & $first & "; " & $upload.n & " floats, " & $upload.dst.bytes & " bytes, data " &
            (if upload.data.len > 0: $upload.data[0 ..< min(8, upload.data.len)] elif upload.captureOffset >= 0: "captured" else: "not recorded") &
            ", first reader " & consumer & ") has no token/position/embedding rule; refusing to freeze host-computed data")
      if match >= 0:
        b.tensor = index[match].name
        b.initFile = if index[match].source == primary: "checkpoint"
                     else: extractFilename(index[match].source) & ".raw"
        b.initOffset = index[match].offset; b.initBytes = upload.dst.bytes
        b.fingerprint = upload.fingerprint
        var dims: seq[string]
        for dim in index[match].shape: dims.add $dim
        l.trace.names.add NameRec(buffer: id, bytes: bytes, name: index[match].name,
          shape: "(" & dims.join(" ") & ")", dtype: "raw")
      elif upload.captureOffset >= 0 and upload.captureBytes == upload.dst.bytes:
        if not sidecarAvailable(l.trace.sourcePath):
          fail("missing upload sidecar " & sidecarPath(l.trace.sourcePath))
        b.tensor = "capture." & upload.fingerprint
        b.initFile = "trace.data"; b.initOffset = upload.captureOffset; b.initBytes = upload.captureBytes
        b.fingerprint = upload.fingerprint
      elif upload.data.len*4 == upload.dst.bytes:
        b.data = upload.data
        while b.data.len < b.elements: b.data.add 0'u32
        b.rawWeight = false
      else:
        unknown.add name & " (" & $bytes & " bytes): " &
          (if match == -2: "ambiguous checkpoint match" else: "no checkpoint identity or captured upload bytes")
    elif b.rawWeight or id in needsInitial:
      var reader = "none"
      for op in ops:
        var reads = false
        for p in op.ins:
          if p.buffer == id: reads = true
        if reads:
          reader = op.implName & (if op.impl == "raw:dispatch": " pipe=" & op.attrs.getOrDefault("pipe") else: "") &
            " (o" & $op.id & ", phase " & $op.phase & ")"
          break
      var history: seq[string]
      for op in l.trace.ops:
        if history.len >= 6: break
        var touches = false
        for p in op.ins & op.outs & op.bindings:
          if p.buffer == id: touches = true
        if touches: history.add op.implName & " o" & $op.id & "@" & $op.order
      # The ops recorded just before the first reader, with their ports (who should have written it).
      var firstOrder = high(int)
      for op in l.trace.ops:
        var touches = false
        for p in op.ins & op.outs & op.bindings:
          if p.buffer == id: touches = true
        if touches: firstOrder = min(firstOrder, op.order)
      var before: seq[string]
      for op in l.trace.ops:
        if op.order >= firstOrder or op.order < firstOrder-12 or op.parent >= 0: continue
        var ports: seq[string]
        for p in op.ins: ports.add "in " & p.name & "=b" & $p.buffer
        for p in op.outs: ports.add "out " & p.name & "=b" & $p.buffer
        before.add op.implName & " o" & $op.id & "@" & $op.order & "[" & ports.join(" ") & "]"
      history.add "preceded by: " & before.join("; ")
      let origin = if id < l.trace.bufs.len: l.trace.bufs[id].origin else: "?"
      unknown.add name & " (" & $bytes & " bytes, origin " & origin & "): no recorded upload or checkpoint identity; first reader " &
        reader & "; recorded uses: " & history.join(", ")
    # A never-uploaded scratch buffer starts undefined, and must be written before reads.
    # Do not silently invent contents for externally initialized state.
    l.bufs[l.bufIdx[name]] = b
  if unknown.len > 0: fail("unresolved raw inputs:\n  " & unknown.join("\n  ") & "\nRe-record with BYBY_FIR_PROBE_DATA=1 (core recorder only)")
  for id in embeddings:
    let tr = perStep[id]
    var emb = l.embeddingName
    if emb.len == 0:
      # The per-token upload is host f32 rows, so prefer a native f32 table in
      # the primary checkpoint whose name identifies an embedding table.
      var candidates: seq[string]
      for tensor in index:
        if tensor.shape == @[l.vocab, l.hidden] and (primary.len == 0 or tensor.source == primary):
          candidates.add tensor.name
      var f32Candidates: seq[string]
      for name in candidates:
        for tensor in index:
          if tensor.name == name and tensor.kind == 0: f32Candidates.add name
      var named: seq[string]
      for name in (if f32Candidates.len > 0: f32Candidates else: candidates):
        let lower = name.toLowerAscii()
        if "embed" in lower or "token_embd" in lower or "wte" in lower: named.add name
      let pool = if named.len > 0: named
                 elif f32Candidates.len > 0: f32Candidates
                 else: candidates
      if pool.len == 1: emb = pool[0]
      elif pool.len > 1:
        # Fall back to a second checkpoint only when the primary cannot decide.
        fail("ambiguous embedding tensor; use --embedding NAME (" & pool[0] & ", " & pool[1] & ")")
      if emb.len == 0:
        for tensor in index:
          if tensor.shape == @[l.vocab, l.hidden]:
            if emb.len > 0: fail("ambiguous embedding tensor; use --embedding NAME")
            emb = tensor.name
    if emb.len == 0: fail("per-step embedding upload needs --weights or --embedding NAME")
    let w = l.weightName(NameRec(name: emb), "f32", @[l.vocab, l.hidden])
    let phase = embeddingPhase.getOrDefault(id)
    if phase != 0: l.phaseRows[phase] = tr.n div l.hidden
    l.nodes.add SNode(kind: "ext", op: "fir.embedding", name: "embedding", ins: @["token", w], outs: @["b" & $id], phase: phase)

proc liftSessionFiles*(trace: Trace; configText, modelName, embeddingName: string;
                       coverage: bool; weightsPaths: seq[string]; fuseDispatches = false;
                       gridTg = 0; dynamicGrid = false): string =
  ## With `coverage`, every recorded op is lowered on its own (unnamed weights and host inputs
  ## become plain buffers, failures are collected) and the result is a coverage report.
  let l = SessionLift(trace: trace, config: parseJson(configText), modelName: modelName,
    embeddingName: embeddingName, coverage: coverage, okCount: initCountTable[string]())
  l.raw = weightsPaths.len > 0
  l.hidden = cfgInt(l.config, ["hidden_size", "n_embd", "d_model", "dim"])
  l.positions = cfgInt(l.config, ["max_position_embeddings", "n_positions", "n_ctx", "seq_length"])
  l.vocab = cfgInt(l.config, ["vocab_size"])
  var ops: seq[Op]
  var rawParents = initHashSet[int]()
  for op in trace.ops:
    if op.impl == "raw:dispatch" and op.parent >= 0: rawParents.incl op.parent
  # A recorded multi-row prefill step before the decode step becomes the multi-row phase of the graph.
  var prefillStep = 0
  if not coverage:
    for i in countdown(trace.steps - 1, 1):
      if i <= trace.stepKinds.len and trace.stepKinds[i-1] == "prefill" and trace.stepRows[i-1] > 1:
        prefillStep = i
        break
  for op in trace.ops:
    if prefillStep > 0 and op.step == prefillStep:
      if op.impl == "raw:dispatch": continue
      if op.parent < 0 and not (op.id in rawParents and op.impl.startsWith("micro:")):
        op.phase = 1
        ops.add op
    elif op.step == trace.steps:
      if op.impl == "raw:dispatch":
        # Semantic session ops already describe their child dispatches. Micro wrappers do not.
        var keepDispatch = op.parent < 0
        if op.parent >= 0:
          for parent in trace.ops:
            if parent.id == op.parent and parent.impl.startsWith("micro:"): keepDispatch = true
        if keepDispatch: ops.add op; l.raw = true
      elif op.parent < 0 and not (op.id in rawParents and op.impl.startsWith("micro:")): ops.add op
  if ops.len == 0: fail("the trace has no recorded step")
  l.addBuf("token", 1, "token")
  if l.raw and l.positions <= 0:
    # Recurrent raw graphs need no context-sized allocation. This is an I/O
    # position counter limit; recorded buffer extents still constrain kernels.
    l.positions = int(high(int32))
  if not coverage: ops = l.withoutFailedCalls(ops)
  if l.raw and not coverage: ops = withoutSingleRowUnits(trace, ops, l.hidden, droppedSingleRowUnits)
  if l.raw: l.prepareRaw(ops, weightsPaths)
  for k in ["rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps", "norm_eps"]:
    if l.eps == 0.0 and l.config.hasKey(k) and l.config[k].kind in {JFloat, JInt}:
      l.eps = l.config[k].getFloat()
  var embedded = false
  var afterTaintedRead = false
  var lastCleanRead = ""
  var logits = ""
  var consumed = initHashSet[int]()
  var i = 0
  var stamped = l.nodes.len
  var stampPhase = 0
  while i < ops.len:
    while stamped < l.nodes.len:
      l.nodes[stamped].phase = stampPhase
      inc stamped
    stampPhase = ops[i].phase
    let op = ops[i]
    let name = op.implName
    if i in consumed:
      inc i
      continue
    if name in UploadOps:
      if l.raw:
        inc i
        continue
      let ti = trace.transferOf(op)
      if ti < 0:
        inc i
        continue
      let tr = trace.transfers[ti]
      let (consumer, portName) = ops.consumerOf(i, tr.dst.buffer)
      if consumer != nil and consumer.implName in ["sessionRope", "sessionRopeKvEnc"] and portName in ["cosT", "sinT"]:
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
        l.addKernelNode(op, "rmsnorm", (if genericKernels(): genRmsnorm(1, l.hidden, l.eps) else: genRmsnormPar(1, l.hidden, l.eps)), @[lastCleanRead, gamma], @[outName])
        afterTaintedRead = false
      elif op.phase > 0 and trace.stepToken >= 0:
        # Token rows of the multi-row prefill phase: the embedding of `rows` tokens.
        if l.phaseRows.hasKey(op.phase):
          # A further token-row upload in the same recorded step (the runtime split the prompt into
          # chunks): it starts a unit of its own, so every chunk size becomes its own phase.
          for j in i ..< ops.len:
            if ops[j].phase > 0: inc ops[j].phase
          stampPhase = op.phase
        if l.hidden <= 0 or tr.n mod l.hidden != 0: fail("the prefill upload has " & $tr.n & " elements, hidden is " & $l.hidden)
        var emb = embeddingName
        if emb.len == 0:
          emb = (if l.config.hasKey("tie_word_embeddings") and l.config["tie_word_embeddings"].getBool(false):
                   "lm_head.weight" else: "model.embed_tokens.weight")
        if l.vocab <= 0: fail("the config has no vocab_size")
        let w = l.weightName(NameRec(name: emb), "f32", @[l.vocab, l.hidden])
        let outName = "t" & $op.id & "_embedding"
        l.addBuf(outName, tr.n)
        l.cur[tr.dst.key] = outName
        l.phaseRows[op.phase] = tr.n div l.hidden
        l.nodes.add SNode(kind: "ext", op: "fir.embedding", name: "embedding_p" & $op.phase, ins: @["token", w], outs: @[outName], phase: op.phase)
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
    if name == "sessionRead" and op.phase > 0 and not l.raw:
      # A vocabulary-sized read ends the unit: the logits after its last row. Other reads are dropped.
      let h = op.ins.port("h")
      if l.vocab > 0 and op.attrInt("n") == l.vocab and l.cur.hasKey(h.key) and l.cur[h.key] notin l.tainted:
        l.phaseLast[op.phase] = l.cur[h.key]
      inc i
      continue
    if name == "sessionRead":
      let h = op.ins.port("h")
      let n = op.attrInt("n")
      if not l.raw and not l.cur.hasKey(h.key):
        if coverage:
          inc i
          continue
        fail("sessionRead of a buffer no op wrote")
      let src = if l.raw: l.rd(PortRef(buffer: h.buffer, offset: h.offset, bytes: n*4)) else: l.cur[h.key]
      if src in l.tainted:
        afterTaintedRead = true
      elif l.raw and op.phase > 0 and l.vocab > 0 and n mod l.vocab == 0:
        let rows = n div l.vocab
        if rows == 1:
          # Native prefill usually reads back only the final row's logits, even when the
          # recorded embedding/GEMM phase consumed a full micro-batch.
          l.phaseLast[op.phase] = src
        elif rows == l.phaseRows.getOrDefault(op.phase):
          l.phaseLogits[op.phase] = (src, rows)
        else:
          fail("prefill read back " & $rows & " logits rows for phase " & $op.phase &
            ", expected one final row or the full " & $l.phaseRows.getOrDefault(op.phase) & " rows")
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
    if name == "sessionCopyInto" and not l.raw:
      var last = i
      while last + 1 < ops.len and (ops[last+1].implName == "sessionCopyInto" or
          (ops[last+1].isBookkeeping and ops[last+1].implName != "sessionBegin" and
           ops[last+1].implName != "sessionEnd")):
        inc last
      l.lowerCopyGroup(ops, i, last)
      i = last + 1
      continue
    if not l.raw and getEnv("FIRLIFT_NOFUSE").len == 0 and not coverage and name in ["sessionActivation", "sessionVecOp", "sessionRope"]:
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
  while stamped < l.nodes.len:
    l.nodes[stamped].phase = stampPhase
    inc stamped
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
  for phase, name in l.phaseLast: live.incl name
  var keep = newSeq[bool](l.nodes.len)
  for idx in countdown(l.nodes.high, 0):
    let n = l.nodes[idx]
    var needed = l.raw  # Preserve all writes to aliased persistent state, including non-logits outputs.
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
  if not l.raw and not coverage and getEnv("FIRLIFT_NOSIDE").len == 0:
    # Side-by-side fusion: consecutive decode kernels that do not depend on each other (no one reads
    # or writes a buffer another one writes) and share a threadgroup size become one launch, each
    # kernel on its own thread range. Every dispatch has a fixed cost, so fewer launches per token
    # is what the fused native kernels buy.
    var kernelOf = initTable[string, int]()
    for idx, k in l.kernels: kernelOf[k.name] = idx
    var sideIdx = initTable[string, int]()
    proc fusable(n: SNode): bool =
      n.kind == "micro" and n.phase == 0 and kernelOf.hasKey(n.op) and not l.kernels[kernelOf[n.op]].raw and
        not l.kernels[kernelOf[n.op]].name.startsWith("s.side")
    proc independent(group: seq[SNode]; n: SNode): bool =
      for g in group:
        for o in g.outs:
          if o in n.ins or o in n.outs: return false
        for o in n.outs:
          if o in g.ins or o in g.outs: return false
      true
    var merged: seq[SNode]
    var group: seq[SNode]
    proc settle() =
      if group.len >= 2:
        var bodies: seq[string]
        var threads, ins, outs, hiR, hiU, inSizes, outSizes: seq[int]
        var allIns, allOuts: seq[string]
        var tgBytes = 16
        let tg = l.kernels[kernelOf[group[0].op]].tg
        for n in group:
          let k = l.kernels[kernelOf[n.op]]
          bodies.add k.body
          threads.add k.threads
          ins.add k.ins
          outs.add k.outs
          hiR.add k.hiR
          hiU.add k.hiU
          tgBytes = max(tgBytes, k.tgBytes)
          inSizes.add k.sizes[0 ..< k.ins]
          outSizes.add k.sizes[k.ins ..< k.ins + k.outs]
          allIns.add n.ins
          allOuts.add n.outs
        let body = fuseSideBySide(bodies, threads, ins, outs, hiR, hiU, tg)
        if body.len > 0:
          let sizes = inSizes & outSizes
          let sig = body & "\n" & sizes.join(",")
          if not sideIdx.hasKey(sig):
            sideIdx[sig] = l.kernels.len
            l.kernels.add SKern(name: "s.side" & (if sideIdx.len > 1: "_" & $sideIdx.len else: ""), body: body,
              threads: sum(threads), tg: tg, tgBytes: tgBytes, ins: allIns.len, outs: allOuts.len, sizes: sizes)
            kernelOf[l.kernels[^1].name] = l.kernels.high
          merged.add SNode(kind: "micro", op: l.kernels[sideIdx[sig]].name, name: group[0].name & "_side",
            ins: allIns, outs: allOuts)
          group = @[]
          return
      for n in group: merged.add n
      group = @[]
    for n in nodes:
      if not fusable(n):
        settle()
        merged.add n
        continue
      if group.len > 0 and (l.kernels[kernelOf[group[0].op]].tg != l.kernels[kernelOf[n.op]].tg or
          not independent(group, n)):
        settle()
      group.add n
    settle()
    nodes = merged
  if fuseDispatches and l.raw:
    # Merge consecutive raw dispatches whose threadgroups are provably independent.
    var roots = initTable[string, string]()
    for b in l.bufs:
      if b.base.len > 0: roots[b.name] = b.base
    var kernelOf = initTable[string, int]()
    for idx, k in l.kernels: kernelOf[k.name] = idx
    proc asFuse(n: SNode): FuseKernel =
      let k = l.kernels[kernelOf[n.op]]
      let parts = k.body.splitWhitespace()
      for w in parts[1 .. ^1]: result.words.add uint32(parseHexInt(w))
      result.threads = k.threads; result.tg = k.tg; result.tgBytes = k.tgBytes
      let slots = k.slots.split(',')
      for i, sl in slots:
        result.bindings.add (sl.split(':')[0], n.ins[i])
    var merged: seq[SNode]
    var cur: SNode
    var curK: FuseKernel
    var chain = false
    proc flush() =
      if chain: merged.add cur
      chain = false
    for n in nodes:
      let isRaw = n.kind == "micro" and kernelOf.hasKey(n.op) and l.kernels[kernelOf[n.op]].raw and
        not q4TensorCoreKernel(l.kernels[kernelOf[n.op]])
      if not isRaw:
        flush()
        merged.add n
        continue
      let kn = asFuse(n)
      if chain and cur.phase == n.phase:
        var plan: FusePlan
        let why = fusable(curK, kn, roots, plan)
        record(why)
        if bindingReason(curK, kn, roots).len == 0: inc fuseStats.bindable
        if why.len == 0:
          var ins = cur.ins
          var slotText: seq[string]
          var bindings: seq[(string, string)]
          for (d, b) in curK.bindings & kn.bindings:
            var at = ins.find(b)
            if at < 0:
              ins.add b
              at = ins.high
            var dup = false
            for x in bindings:
              if x[0] == d: dup = true
            if not dup:
              bindings.add (d, b)
              slotText.add d & ":in" & $at
          var outs = cur.outs
          for o in n.outs:
            if o notin outs: outs.add o
          let words = concatWords(curK, kn, plan)
          let threads = fusedThreads(curK, kn)
          var body = "words"
          for w in words: body.add " " & toHex(w, 8).toLowerAscii
          let name = "fused" & $l.kernels.len
          l.kernels.add SKern(name: name, body: body, threads: threads, tg: curK.tg,
            tgBytes: max(curK.tgBytes, kn.tgBytes), ins: ins.len, outs: outs.len,
            slots: slotText.join(","), raw: true)
          kernelOf[name] = l.kernels.high
          cur = SNode(kind: "micro", op: name, name: cur.name & "_" & n.name, ins: ins,
            outs: outs, rmw: true, phase: cur.phase)
          curK = FuseKernel(words: words, threads: threads, tg: curK.tg,
            tgBytes: max(curK.tgBytes, kn.tgBytes), bindings: bindings)
          continue
      flush()
      cur = n; curK = kn; chain = true
    flush()
    nodes = merged
  if gridTg > 0 and l.raw:
    # Persistent chains: consecutive raw dispatches become one launch of `gridTg` threadgroups.
    const ChainWeightFloats = 48 * 1024 * 1024
    var kernelOf = initTable[string, int]()
    for idx, k in l.kernels: kernelOf[k.name] = idx
    proc rootName(name: string): string =
      let b = l.bufs[l.bufIdx[name]]
      if b.base.len > 0: b.base else: name
    proc viewOffset(name: string): int =
      let b = l.bufs[l.bufIdx[name]]
      if b.base.len > 0: b.offset else: 0
    # The runner packs a raw weight into a pool only when every consumer is a slot kernel.
    var offPool = initHashSet[string]()
    for n in nodes:
      let viaSlots = n.kind == "micro" and kernelOf.hasKey(n.op) and l.kernels[kernelOf[n.op]].raw
      if not viaSlots:
        for name in n.ins & n.outs:
          offPool.incl (if l.bufIdx.hasKey(name) and l.bufs[l.bufIdx[name]].base.len > 0: l.bufs[l.bufIdx[name]].base else: name)
    proc isRawWeight(name: string): bool =
      let b = l.bufs[l.bufIdx[name]]
      b.rawWeight and (if b.base.len > 0: b.base else: name) notin offPool
    proc stageOf(n: SNode): PersistentStage =
      let k = l.kernels[kernelOf[n.op]]
      let parts = k.body.splitWhitespace()
      for w in parts[1 .. ^1]: result.kernel.words.add uint32(parseHexInt(w))
      result.kernel.threads = k.threads; result.kernel.tg = k.tg; result.kernel.tgBytes = k.tgBytes
      let slotList = k.slots.split(',')
      for i in 0 ..< slotList.len:
        result.kernel.bindings.add (slotList[i].split(':')[0], n.ins[i])
    let domains = ["mem", "u8", "lut", "input"]
    proc compatible(chain: seq[SNode]; n: SNode): bool =
      let b = stageOf(n)
      var weights = 0
      for c in chain & @[n]:
        for name in c.ins:
          if isRawWeight(name): weights += l.bufs[l.bufIdx[name]].elements
      if weights > ChainWeightFloats: return false
      for d in domains:
        var other = ""
        for (dd, nm) in b.kernel.bindings:
          if dd == d: other = nm
        if other.len == 0: continue
        for c in chain:
          var first = ""
          for (dd, nm) in stageOf(c).kernel.bindings:
            if dd == d: first = nm
          if first.len == 0: continue
          # Every buffer of a domain must live in one allocation: the same weight pool, or the same root.
          if isRawWeight(first) != isRawWeight(other): return false
          if not isRawWeight(first) and rootName(first) != rootName(other): return false
      return true
    proc build(chain: seq[SNode]): (SNode, string) =
      var ins: seq[string]
      var stages: seq[PersistentStage]
      var outs: seq[string]
      var memAnchor = high(int)
      var memRoot = ""
      var bound = initTable[string, int]()
      for n in chain:
        var st = stageOf(n)
        for (d, nm) in st.kernel.bindings:
          var at = ins.find(nm)
          if at < 0:
            ins.add nm
            at = ins.high
          if d == "mem":
            memAnchor = min(memAnchor, viewOffset(nm))
            memRoot = rootName(nm)
          if d notin bound: bound[d] = at
          st.rebase.add (d, at)
        stages.add st
        for o in n.outs:
          if o notin outs: outs.add o
      let root = l.bufs[l.bufIdx[memRoot]]
      if root.data.len > 0:
        return (SNode(), "mem buffer " & memRoot & " has initwords")
      # --dynamic-grid-tg: prefer gridTg threadgroups, but a chain whose barrier stages cannot be
      # split that way runs with the largest smaller count that works.
      var chainTg = gridTg
      var program: Persistent
      var why = ""
      while true:
        (program, why) = persistentChain(stages, chainTg, 0xFFFFFFFF)
        if why.len == 0 or not dynamicGrid or chainTg <= 1 or "divisible" notin why: break
        dec chainTg
      if why.len > 0: return (SNode(), why)
      l.addBuf("gridctrl", 64)
      if "gridctrl" notin ins: ins.add "gridctrl"
      var slotText: seq[string]
      for d in domains:
        slotText.add d & ":in" & $bound.getOrDefault(d, bound["mem"])
      slotText.add "u32:in" & $ins.find("gridctrl")
      let name = "grid" & $l.kernels.len
      l.kernels.add SKern(name: name, body: program.lines.join("\n"), threads: chainTg * PhysicalTg,
        tg: PhysicalTg, tgBytes: program.tgBytes, ins: ins.len, outs: outs.len,
        slots: slotText.join(","), raw: true)
      kernelOf[name] = l.kernels.high
      return (SNode(kind: "micro", op: name, name: chain[0].name & "_grid", ins: ins, outs: outs,
        rmw: true, phase: chain[0].phase), "")
    var merged: seq[SNode]
    var run: seq[SNode]
    proc isRaw(n: SNode): bool = n.kind == "micro" and kernelOf.hasKey(n.op) and
      l.kernels[kernelOf[n.op]].raw and not q4TensorCoreKernel(l.kernels[kernelOf[n.op]])
    proc settle() =
      var pending = run
      run = @[]
      while pending.len > 0:
        var take = pending.len
        var built: SNode
        while take >= 2:
          let (node, why) = build(pending[0 ..< take])
          if why.len == 0:
            built = node
            break
          fuseStats.reasons.inc "persistent: " & why
          dec take
        if take >= 2:
          merged.add built
          inc fuseStats.chains
          fuseStats.chained += take
          pending = pending[take .. ^1]
        else:
          merged.add pending[0]
          pending = pending[1 .. ^1]
    for n in nodes:
      if not isRaw(n):
        settle()
        merged.add n
        continue
      if run.len > 0 and (run[0].phase != n.phase or not compatible(run, n)):
        settle()
      run.add n
    settle()
    nodes = merged
  if not l.raw and not coverage and getEnv("FIRLIFT_NOREUSE").len == 0:
    # Buffer reuse. Every op output got a buffer of its own, so a graph holds one activation per
    # op (GiBs for a large multi-row phase). A temporary only has to live from the node that
    # writes it to the last node that reads it: later temporaries of the same size take over its
    # buffer. Only buffers that one phase writes before it reads them qualify; state, inputs,
    # outputs and anything two phases share keep their own buffer.
    var persistent = initHashSet[string]()
    persistent.incl logits
    for phase, name in l.phaseLast: persistent.incl name
    for phase, output in l.phaseLogits: persistent.incl output[0]
    var phaseOf = initTable[string, int]()
    var phaseList: seq[int]
    for n in nodes:
      if n.phase notin phaseList: phaseList.add n.phase
      for b in n.ins & n.outs:
        if not l.bufIdx.hasKey(b): continue          # a weight
        let sb = l.bufs[l.bufIdx[b]]
        if sb.role.len > 0 or sb.base.len > 0 or sb.initFile.len > 0 or sb.data.len > 0 or sb.rawWeight or
            not b.startsWith("t"):
          persistent.incl b
        if phaseOf.hasKey(b) and phaseOf[b] != n.phase: persistent.incl b
        phaseOf[b] = n.phase
    proc temporary(b: string): bool = l.bufIdx.hasKey(b) and b notin persistent
    # first writer and last user of every temporary, per phase
    var firstAt = initTable[string, int]()
    var lastAt = initTable[string, int]()
    for idx, n in nodes:
      for b in n.ins:
        if not l.bufIdx.hasKey(b) or b in persistent: continue
        if not firstAt.hasKey(b): persistent.incl b    # read before this phase wrote it
        lastAt[b] = idx
      for b in n.outs:
        if not l.bufIdx.hasKey(b) or b in persistent: continue
        if not firstAt.hasKey(b):
          if n.rmw: persistent.incl b                  # updated in place: its old contents are read
          else: firstAt[b] = idx
        lastAt[b] = idx
    var rename = initTable[string, string]()
    var slots = 0
    for phase in phaseList:
      var free = initTable[int, seq[string]]()         # elements -> buffers free to take over
      var live: seq[(int, string, int)]                # (last user, buffer, elements)
      for idx, n in nodes:
        if n.phase != phase: continue
        var still: seq[(int, string, int)]
        for item in live:
          if item[0] < idx: free.mgetOrPut(item[2], @[]).add item[1]
          else: still.add item
        live = still
        for b in n.outs:
          if not temporary(b) or firstAt.getOrDefault(b, -1) != idx or rename.hasKey(b): continue
          let size = l.bufs[l.bufIdx[b]].elements
          var slot = b
          if free.getOrDefault(size).len > 0: slot = free[size].pop()
          else: inc slots
          rename[b] = slot
          live.add (lastAt[b], slot, size)
    for n in nodes.mitems:
      for b in n.ins.mitems:
        if rename.hasKey(b): b = rename[b]
      for b in n.outs.mitems:
        if rename.hasKey(b): b = rename[b]
    reusedBuffers = rename.len - slots
  # Emit.
  var usedBuf = initHashSet[string]()
  var usedWeight = initHashSet[string]()
  var usedKernel = initHashSet[string]()
  for n in nodes:
    for b in n.ins & n.outs:
      usedBuf.incl b
      usedWeight.incl b
    if n.kind == "micro": usedKernel.incl n.op
  for phase, output in l.phaseLogits:
    usedBuf.incl output[0]
  for phase, name in l.phaseLast: usedBuf.incl name
  # The logits read may cover only part of its buffer (a scratch longer than the vocabulary): the
  # role then sits on a view that no node names, which must still be emitted.
  usedBuf.incl logits
  for b in l.bufs:
    if b.name in usedBuf and b.base.len > 0: usedBuf.incl b.base
  var hasResidentQ4 = false
  for weight in l.weights:
    if weight.dtype.startsWith("q4r"):
      hasResidentQ4 = true
  for k in l.kernels:
    # Shape-op micro-kernels (FP4 Tensor Core GEMM) write through an output slot, which the
    # runner accepts from script_abi 2 with the resident_q4 feature.
    if k.name in usedKernel and ":out" in k.slots:
      hasResidentQ4 = true
  var outText = "fir 1\nmodel " & modelName & "\nconfig config.json\nruntime graph\nlayout script\nscript_abi " &
    (if hasResidentQ4: "2" else: "1") & "\n"
  if hasResidentQ4: outText.add "require feature resident_q4\n"
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
    if b.base.len > 0: continue
    if b.rawWeight:
      outText.add "weight " & b.name & " raw " & b.tensor & " tensor [" & $b.elements & "]\n"
    else:
      outText.add "buf " & b.name & " f32 " & $b.elements
      if b.role == "logits": outText.add " output:logits"
      elif b.role.len > 0: outText.add " input:" & b.role
      outText.add "\n"
    if b.initFile.len > 0:
      outText.add "init " & b.name & " " & b.initFile & " " & $b.initOffset & " " & $b.initBytes
      if b.fingerprint.len > 0: outText.add " " & b.fingerprint
      outText.add "\n"
    if b.data.len > 0:
      var data: seq[string]
      for word in b.data: data.add toHex(word, 8).toLowerAscii
      outText.add "initwords " & b.name & " " & data.join(" ") & "\n"
  for b in l.bufs:
    if b.name notin usedBuf or b.base.len == 0: continue
    outText.add "view " & b.name & " " & b.base & " " & $b.offset & " " & $b.elements
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
    if not k.raw:
      for idx in 0 ..< k.ins: expect.add "in" & $idx & ":" & $k.sizes[idx]
      for idx in 0 ..< k.outs: expect.add "out" & $idx & ":" & $k.sizes[k.ins + idx]
    outText.add "kernel " & k.name & " in=" & $k.ins & " out=" & $k.outs & " threads=" & $k.threads &
      " tg=" & $k.tg & " tgbytes=" & $k.tgBytes
    if expect.len > 0: outText.add " expect=" & expect.join(",")
    if k.rowBlock > 0: outText.add " rows=" & $k.rowBlock & ":" & $k.rowThreads
    if k.raw:
      outText.add " layout=slots"
      if k.slots.len > 0: outText.add " slots=" & k.slots
    if k.tag.len > 0: outText.add " tag=" & k.tag
    outText.add "\n" & k.body & "\nendkernel\n"
  var seen = initCountTable[string]()
  var ordered: seq[SNode]
  var maxPhase = 0
  for n in nodes: maxPhase = max(maxPhase, n.phase)
  for phase in 1 .. maxPhase:
    for n in nodes:
      if n.phase == phase: ordered.add n
  let hasPrefill = ordered.len > 0
  for n in nodes:
    if n.phase == 0: ordered.add n
  proc partialPhase(phase: int): bool =
    ## Every node of the phase is an embedding or a row-major kernel over the phase rows: the
    ## runner may then run the phase over fewer rows (the tail of a prompt).
    var kernelOf = initTable[string, int]()
    for idx, k in l.kernels: kernelOf[k.name] = idx
    for n in nodes:
      if n.phase != phase: continue
      if n.kind == "ext" and n.op == "fir.embedding": continue
      if n.kind == "session" and "lastRowOf=" in n.op: return true   # what follows works on that one row
      if n.kind != "micro" or not kernelOf.hasKey(n.op): return false
      let k = l.kernels[kernelOf[n.op]]
      if k.rowBlock <= 0 or k.rows != l.phaseRows.getOrDefault(phase): return false
    true
  var emittedPhase = -1
  for n in ordered:
    seen.inc n.name
    let nm = (if seen[n.name] > 1: n.name & "_" & $seen[n.name] else: n.name)
    if hasPrefill and n.phase != emittedPhase:
      emittedPhase = n.phase
      if n.phase != 0:
        if l.phaseRows.getOrDefault(n.phase) <= 0: fail("replayed unit " & $n.phase & " has no token-row embedding input")
        outText.add "phase chunk" & $n.phase & " rows=" & $l.phaseRows[n.phase]
        if l.phaseLogits.hasKey(n.phase): outText.add " logits=" & l.phaseLogits[n.phase][0]
        else:
          if l.phaseLast.hasKey(n.phase): outText.add " last=" & l.phaseLast[n.phase]
          if partialPhase(n.phase): outText.add " partial"
        outText.add "\n"
      else:
        outText.add "phase decode\n"
    outText.add "node " & nm & " " & n.kind & " " & n.op & " " & n.ins.join(" ") & " -> " & n.outs.join(" ") & "\n"
  result = outText

proc liftSession*(trace: Trace; configText, modelName, embeddingName: string;
                  coverage = false; weightsPath = ""): string =
  ## Single-checkpoint convenience wrapper; the CLI passes every --weights file.
  if weightsPath.len > 0: liftSessionFiles(trace, configText, modelName, embeddingName, coverage, @[weightsPath])
  else: liftSessionFiles(trace, configText, modelName, embeddingName, coverage, newSeq[string]())
