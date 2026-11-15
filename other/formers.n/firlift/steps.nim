## Marker-free selection uses repeated session signatures, never model/kernel names.
import std/[strutils, tables, sets, algorithm, os]
import micro_kernel/slot_layout
import ./firlift
import ./firvm
import ./sidecar

type
  SessionUnit* = object
    first*, last*, dispatches*: int
    signature*: string
    launches*: seq[Op]
  StepSelection* = object
    trace*: Trace
    firstSession*, sessionCount*, occurrence*, dispatches*: int
    prefillFirst*, decodeFirst*: int
    resetCopies*: int                ## state-clearing copies left out of the replayed units

proc sessions*(t: Trace): seq[SessionUnit] =
  var current = SessionUnit(first: -1)
  for op in t.ops:
    if op.impl == "session:sessionBegin":
      current = SessionUnit(first: op.order)
    elif current.first >= 0:
      if op.impl == "session:sessionEnd":
        current.last = op.order
        result.add current
        current.first = -1
      elif op.impl == "raw:dispatch":
        inc current.dispatches
        current.launches.add op
        var pipe = op.attrs.getOrDefault("pipe")
        # JIT prefixes encode a changing program hash, followed by the caller's kind.
        if pipe.startsWith("jit_"):
          let sep = pipe.find('_', 4)
          if sep >= 0: pipe = pipe[sep+1 .. ^1]
        if pipe.len == 0:
          var pc = 0
          while pc < op.words.len:
            let word = op.words[pc]
            pipe.add toHex(word, 8)
            inc pc
            if needsWord(word and 255): pc += arityWord(word and 255)
        current.signature.add "D:" & pipe & ";"
      elif op.parent < 0 and op.impl notin ["session:sessionFree", "session:sessionAllocScratch",
          "session:sessionAllocFromHeap", "session:sessionCommitContinue"]:
        let kind = if op.impl.startsWith("micro:"): "micro" else: op.impl
        current.signature.add kind & ":" & op.attrs.getOrDefault("op") & ";"

proc isWarmupSession(t: Trace; s: SessionUnit): bool =
  if s.dispatches > 1: return false
  for op in t.ops:
    if op.order < s.first: continue
    if op.order > s.last: break
    if op.impl in ["session:sessionBegin", "session:sessionEnd", "session:sessionFree",
        "session:sessionAllocScratch", "session:sessionAllocFromHeap",
        "session:sessionGetLastError", "session:sessionTensorCoreRequired",
        "session:sessionJitPrewarm"]: continue
    if op.impl.startsWith("raw:"): continue
    if op.impl.startsWith("session:sessionUpload") or op.impl == "session:sessionCopyInto": return false
    if op.parent < 0: return false
  true


const NonDataOps = ["sessionBegin", "sessionEnd", "sessionFree", "sessionAllocScratch", "sessionAllocHeap",
  "sessionAllocFromHeap", "sessionFreeHeap", "sessionJitPrewarm", "sessionGetLastError",
  "sessionCommitContinue", "sessionTensorCoreRequired", "sessionRead"]

proc opName(op: Op): string =
  let at = op.impl.find(':')
  if at >= 0: op.impl[at+1 .. ^1] else: op.impl

proc isUpload(op: Op): bool = op.opName.startsWith("sessionUpload")

proc isRopeTableUpload*(t: Trace; upload: Op; buffer: int): bool =
  ## RoPE tables can be exactly one hidden vector in size. They are persistent inputs,
  ## not token-row embeddings, even when the same buffer ID is reused as decode scratch later.
  var afterUpload = false
  for candidate in t.ops:
    if not afterUpload:
      if candidate.id == upload.id: afterUpload = true
      continue
    for p in candidate.ins:
      if p.buffer == buffer:
        return candidate.opName in ["sessionRope", "sessionRopeKvEnc"] and
          p.name in ["cosT", "sinT"]
    # A later upload/free/write starts another lifetime of this buffer. Do not infer
    # the old upload's role from a consumer of the replacement contents.
    for p in candidate.outs:
      if p.buffer == buffer: return false
  false

proc isUploadedRopeTable*(t: Trace; consumer: Op; portName: string): bool =
  ## Resolve the producer at this use, rather than assigning a permanent role to a
  ## buffer ID that the allocator may reuse for computed tables or token rows.
  var buffer = -1
  for p in consumer.ins:
    if p.name == portName: buffer = p.buffer
  if buffer < 0: return false
  for i in countdown(t.ops.high, 0):
    let producer = t.ops[i]
    if producer.order >= consumer.order: continue
    # A recorded upload names its destination in its transfer; its op has no output ports.
    var writes = false
    for p in producer.outs:
      if p.buffer == buffer: writes = true
    if producer.isUpload:
      for tr in t.transfers:
        if tr.kind == "upload" and tr.op == producer.id and tr.dst.buffer == buffer: writes = true
    if writes:
      return producer.isUpload and t.isRopeTableUpload(producer, buffer)
  false

proc accessOf(op: Op; t: Trace; reads, writes: var seq[int]) =
  ## Buffers an op reads and writes, from named ports or from the bytecode's domain access.
  if op.impl == "raw:dispatch":
    let access = vmAccess(op.words)
    for p in op.bindings:
      if p.name in access.reads: reads.add p.buffer
      if p.name in access.writes: writes.add p.buffer
  else:
    for p in op.ins: reads.add p.buffer
    for p in op.outs: writes.add p.buffer
    if op.isUpload:
      for tr in t.transfers:
        if tr.kind == "upload" and tr.op == op.id: writes.add tr.dst.buffer

type Dims* = object
  hidden*, vocab*: int

proc captureZeroRange(t: Trace; tr: TransferRec; byteOffset, bytes: int): bool =
  ## True when bytes [byteOffset, byteOffset+bytes) of an upload's recorded payload are present and all zero.
  if bytes <= 0 or byteOffset < 0 or tr.captureOffset < 0 or tr.captureBytes <= 0 or
      tr.captureBytes != tr.dst.bytes or byteOffset+bytes > tr.captureBytes: return false
  sidecarAllZero(t.sourcePath, tr.captureOffset+byteOffset, bytes)

proc captureZero(t: Trace; tr: TransferRec): bool =
  t.captureZeroRange(tr, 0, tr.captureBytes)

proc attrNum(op: Op; key: string): int =
  if op.attrs.hasKey(key):
    try: return parseInt(op.attrs[key])
    except ValueError: discard
  0

proc uploadedZero(t: Trace; buffer, byteOffset, bytes, before: int): int =
  ## Op order of the last upload into `buffer` before op order `before` that touches the byte range,
  ## when that upload covers the whole range and its recorded payload is all zero there; -1 otherwise.
  var best = -1
  var found = -1
  for i, tr in t.transfers:
    if tr.kind != "upload" or tr.dst.buffer != buffer or tr.order >= before: continue
    if tr.dst.offset < byteOffset+bytes and byteOffset < tr.dst.offset+tr.dst.bytes and tr.order >= best:
      best = tr.order
      found = i
  if found < 0: return -1
  let tr = t.transfers[found]
  if byteOffset >= tr.dst.offset and byteOffset+bytes <= tr.dst.offset+tr.dst.bytes and
      t.captureZeroRange(tr, byteOffset-tr.dst.offset, bytes): best else: -1

proc writesBuffer(op: Op; t: Trace; buffer: int): bool =
  if op.impl == "raw:dispatch":
    var bound = false
    for p in op.bindings:
      if p.buffer == buffer: bound = true
    if not bound: return false
  var reads, writes: seq[int]
  op.accessOf(t, reads, writes)
  buffer in writes

proc holdsZeros(t: Trace; buffer, byteOffset, bytes, before: int): bool =
  ## `buffer` still holds the zeros it was uploaded with in the byte range at op order `before`:
  ## no other op wrote it since that upload.
  let uploaded = t.uploadedZero(buffer, byteOffset, bytes, before)
  if uploaded < 0: return false
  for op in t.ops:
    if op.order >= before: break
    if op.order <= uploaded or op.isUpload or op.opName in NonDataOps: continue
    if op.writesBuffer(t, buffer): return false
  true

type ResetCopy = object
  op: Op
  dstBuffer, dstByte, bytes: int   ## the cleared byte range

proc zeroFill(t: Trace; op: Op; reset: var ResetCopy): bool =
  ## A sessionCopyInto from a range of a still-zero uploaded buffer.
  var src, dst: PortRef
  var haveSrc = false
  var haveDst = false
  for p in op.ins:
    if p.name == "src":
      src = p
      haveSrc = true
  for p in op.outs:
    if p.name == "dst":
      dst = p
      haveDst = true
  let n = op.attrNum("n")
  if not haveSrc or not haveDst or n <= 0: return false
  if not t.holdsZeros(src.buffer, src.offset+op.attrNum("srcOffsetElems")*4, n*4, op.order): return false
  reset = ResetCopy(op: op, dstBuffer: dst.buffer, dstByte: dst.offset+op.attrNum("dstOffsetElems")*4, bytes: n*4)
  true

const ResetNeutralOps = ["sessionFree", "sessionAllocScratch", "sessionAllocHeap", "sessionAllocFromHeap",
  "sessionFreeHeap", "sessionJitPrewarm", "sessionGetLastError", "sessionCommitContinue",
  "sessionTensorCoreRequired", "sessionPacesCommandBuffers", "sessionFlashAttentionPrepare",
  "sessionMamba2SsdPrepare", "sessionMaxTgThreads", "sessionRecoverQueue", "sessionSetF32GemmTiled",
  "sessionSetFp4GemmTiled", "sessionCooperativeGemmEnabled", "sessionCooperativeGemmJobs"]

proc stateResets(t: Trace): seq[ResetCopy] =
  ## The copies of every "reset session": a session that launches no kernel and only copies ranges
  ## of a zero-filled uploaded buffer over other buffers (the native runtime clears its recurrent
  ## state between prompts that way).
  var open = false
  var clean = false
  var found: seq[ResetCopy]
  for op in t.ops:
    if op.impl == "session:sessionBegin":
      open = true
      clean = true
      found.setLen(0)
    elif not open:
      continue
    elif op.impl == "session:sessionEnd":
      open = false
      if clean:
        for c in found: result.add c
    elif op.impl == "session:sessionCopyInto" and op.parent < 0:
      var reset: ResetCopy
      if t.zeroFill(op, reset): found.add reset
      else: clean = false
    elif op.impl.startsWith("session:") and op.opName in ResetNeutralOps:
      discard
    else:
      clean = false

proc droppableResets(t: Trace; carried: HashSet[int]; first, last: int): HashSet[int] =
  ## The reset copies among the carried ops that the lifted graph does not need. Its buffers start
  ## from their uploaded contents, so a clear is redundant when the cleared range was uploaded as
  ## zeros and no replayed op wrote the buffer before the copy. Kept inside a replayed phase the
  ## clear would wipe the recurrent state again at every use of that phase.
  let resets = t.stateResets()
  if resets.len == 0: return
  var isReset = initHashSet[int]()
  for r in resets: isReset.incl r.op.order
  var firstTouch = initTable[int, int]()
  var firstWrite = initTable[int, int]()
  for op in t.ops:
    if op.opName in NonDataOps or op.isUpload: continue
    if not ((op.order in carried and op.order notin isReset) or (op.order >= first and op.order <= last)):
      continue
    var reads, writes: seq[int]
    op.accessOf(t, reads, writes)
    for b in reads & writes:
      if b notin firstTouch: firstTouch[b] = op.order
    for b in writes:
      if b notin firstWrite: firstWrite[b] = op.order
  for r in resets:
    if r.op.order notin carried: continue
    if firstWrite.getOrDefault(r.dstBuffer, high(int)) < r.op.order: continue
    if firstTouch.hasKey(r.dstBuffer) and
        t.uploadedZero(r.dstBuffer, r.dstByte, r.bytes, firstTouch[r.dstBuffer]) < 0: continue
    result.incl r.op.order

proc sliceBefore(t: Trace; seeds: HashSet[int]; before: int; dims: Dims;
                 tokenRows: var bool): HashSet[int] =
  ## Backward data-flow slice: orders of the ops before `before` that wrote buffers in `seeds`
  ## (growing with what those ops read), plus host uploads of per-run buffers (buffers released
  ## later) that feed them. Buffer-level and conservative: a dispatch may use only part of a
  ## binding, so a write never proves that a later read was produced inside the slice (one arena
  ## can hold both scratch and persistent state). `tokenRows` is set when a non-zero upload
  ## of whole hidden-size rows (token embeddings) feeds the slice.
  var rawParents = initHashSet[int]()
  for op in t.ops:
    if op.impl == "raw:dispatch" and op.parent >= 0: rawParents.incl op.parent
  var micro = initHashSet[int]()
  for op in t.ops:
    if op.impl.startsWith("micro:"): micro.incl op.id
  var needed = seeds
  var released = initHashSet[int]()
  for op in t.ops:
    if op.order < before and op.opName == "sessionFree":
      for p in op.outs: released.incl p.buffer
  for i in countdown(t.ops.high, 0):
    let op = t.ops[i]
    if op.order >= before or op.opName in NonDataOps: continue
    let isTop = if op.impl == "raw:dispatch": op.parent < 0 or op.parent in micro
                else: op.parent < 0 and not (op.id in rawParents and op.impl.startsWith("micro:"))
    if not isTop: continue
    var reads, writes: seq[int]
    op.accessOf(t, reads, writes)
    var hit = false
    for b in writes:
      if b in needed and (not op.isUpload or b in released): hit = true
    if not hit: continue
    result.incl op.order
    for b in reads: needed.incl b
    if op.isUpload and dims.hidden > 0:
      for tr in t.transfers:
        if tr.kind == "upload" and tr.op == op.id and tr.n >= dims.hidden and
            tr.n mod dims.hidden == 0 and not t.captureZero(tr): tokenRows = true

proc stateProducers(t: Trace; first, last: int; dims: Dims): HashSet[int] =
  ## Orders of the earlier ops the unit [first, last] depends on, plus the earlier passes that
  ## return vocabulary-sized rows to the host and are fed by token-row uploads (their reads
  ## are outputs of replayed phases). Warm-up, self-test and unrelated read-back sessions of
  ## the original run are not part of it.
  var dummy = false
  var seeds = initHashSet[int]()
  for op in t.ops:
    if op.order < first or op.order > last: continue
    var reads, writes: seq[int]
    op.accessOf(t, reads, writes)
    for b in reads: seeds.incl b
  result = sliceBefore(t, seeds, first, dims, dummy)
  if dims.vocab <= 0: return
  for op in t.ops:
    if op.order >= first or op.opName != "sessionRead" or op.parent >= 0: continue
    let n = (if op.attrs.hasKey("n"): parseInt(op.attrs["n"]) else: 0)
    if n < dims.vocab or n mod dims.vocab != 0: continue
    var readSeeds = initHashSet[int]()
    for p in op.ins: readSeeds.incl p.buffer
    var rows = false
    let produced = sliceBefore(t, readSeeds, op.order, dims, rows)
    if not rows: continue
    for order in produced: result.incl order
    result.incl op.order

proc selectStep*(t: Trace; step = -1; slice = ""; dims = Dims()): StepSelection =
  result.trace = t
  if t.hasStep and slice.len == 0 and step < 0: return
  let ss = sessions(t)
  if ss.len == 0:
    if slice.len > 0 or step >= 0: raise newException(ValueError, "step selection needs complete sessions")
    return
  var start = 0
  var width = ss.len
  if slice.len > 0:
    let parts = slice.split(':')
    if parts.len != 2: raise newException(ValueError, "--slice expects session indices A:B (zero-based, B exclusive)")
    start = parseInt(parts[0])
    width = parseInt(parts[1])-start
  else:
    var bestScore = 0
    var occurrences = 0
    for a in 0 ..< ss.len:
      # No fixed maximum unit width: a decode unit may span many sessions.
      for period in 1 .. (ss.len-a) div 2:
        var count = 1
        while a+(count+1)*period <= ss.len:
          var equal = true
          for j in 0 ..< period:
            if ss[a+j].signature != ss[a+count*period+j].signature: equal = false; break
          if not equal: break
          inc count
        if count < 2: continue
        var cost = 0
        for j in 0 ..< period:
          cost += max(1, ss[a+j].signature.len)
        let score = count*cost
        if score > bestScore or (score == bestScore and period < width):
          bestScore = score
          start = a
          width = period
          occurrences = count
    if bestScore == 0:
      if ss.len != 1:
        raise newException(ValueError, "no repeating complete session unit; choose --slice A:B")
      width = 1
      occurrences = 1
    let chosen = if step < 0: 0 else: step
    if chosen >= occurrences:
      raise newException(ValueError, "--step exceeds repeating unit count " & $occurrences)
    for occurrence in 1 ..< occurrences:
      for j in 0 ..< width:
        let reference = ss[start+j].launches
        let observed = ss[start+occurrence*width+j].launches
        for k in 0 ..< reference.len:
          if k >= observed.len or reference[k].words != observed[k].words or
              reference[k].attrs.getOrDefault("threads") != observed[k].attrs.getOrDefault("threads") or
              reference[k].attrs.getOrDefault("tg") != observed[k].attrs.getOrDefault("tg") or
              reference[k].attrs.getOrDefault("tgbytes") != observed[k].attrs.getOrDefault("tgbytes"):
            raise newException(ValueError, "repeating unit changes raw program/geometry; dynamic launch lowering is required (use --slice A:B only for fixed-launch replay)")
    result.occurrence = chosen
    start += chosen*width
  if start < 0 or width <= 0 or start+width > ss.len:
    raise newException(ValueError, "--slice is outside complete session range 0:" & $ss.len)
  var first = ss[start].first
  let lastSession = ss[start+width-1].last
  let stop = if start+width < ss.len: ss[start+width].first else: high(int)
  # Include the immediately preceding host uploads, and trailing logits read.
  for i in countdown(t.ops.high, 0):
    let op = t.ops[i]
    if op.order >= first: continue
    if op.impl.startsWith("session:sessionUpload"): first = op.order
    else: break
  result.decodeFirst = ss[start].first
  result.prefillFirst = -1
  var carried = initHashSet[int]()   # order of earlier ops that produced state the unit reads
  if slice.len == 0 and result.occurrence == 0:
    carried = stateProducers(t, ss[start].first, ss[start+width-1].last, dims)
  var last = lastSession
  for op in t.ops:
    if op.order <= lastSession or op.order >= stop: continue
    if op.impl.startsWith("session:sessionUpload"): break
    last = op.order
  if carried.len > 0:
    # State clears between prompts are the initial contents of the lifted graph, not replayed work.
    for order in t.droppableResets(carried, first, last):
      carried.excl order
      inc result.resetCopies
    var earliest = first
    for order in carried:
      if order < earliest: earliest = order
    if earliest < first: result.prefillFirst = earliest
  result.trace.ops = @[]
  for op in t.ops:
    if op.order in carried or (op.order >= first and op.order <= last):
      if result.trace.ops.len == 0 and op.step > 0 and op.step <= t.stepPositions.len:
        result.trace.stepPosition = t.stepPositions[op.step-1]
        result.trace.stepToken = t.stepTokens[op.step-1]
      let copied = Op()
      copied[] = op[]
      copied.step = 0
      copied.phase = (if op.order in carried: 1 else: 0)
      result.trace.ops.add copied
  result.trace.steps = 0
  result.trace.hasStep = false
  result.firstSession = start
  result.trace.selectionSession = start
  result.trace.decodeFirst = result.decodeFirst
  result.sessionCount = width
  for j in 0 ..< width: result.dispatches += ss[start+j].dispatches

proc withoutSingleRowUnits*(t: Trace; ops: seq[Op]; hidden: int; dropped: var int): seq[Op] =
  ## The replayed ops without the units that feed a single token (`dropped` counts them). The native
  ## chat path runs the few tokens left over after its bucketed prefill as plain decode steps, so such
  ## a unit is the decode unit without its vocabulary head. A phase has two or more rows; the runner
  ## feeds the remaining tokens through the decode step, which advances the same state.
  if hidden <= 0: return ops
  var starts: seq[(int, int)]   # (op order of the token-row upload, rows)
  for op in ops:
    if op.phase == 0 or not op.isUpload: continue
    for tr in t.transfers:
      if tr.kind == "upload" and tr.op == op.id:
        if tr.n >= hidden and tr.n mod hidden == 0 and not t.captureZero(tr) and
            not t.isRopeTableUpload(op, tr.dst.buffer):
          starts.add (op.order, tr.n div hidden)
        break
  starts.sort()
  var singles = 0
  for s in starts:
    if s[1] == 1: inc singles
  if singles == 0: return ops
  for op in ops:
    if op.phase > 0:
      # Ops before the first token-row upload belong to the first unit.
      var unit = 0
      for k, s in starts:
        if s[0] <= op.order: unit = k
      if starts[unit][1] == 1: continue
    result.add op
  dropped += singles
