## Marker-free selection uses repeated session signatures, never model/kernel names.
import std/[strutils, tables, sets, algorithm, os]
import micro_kernel/slot_layout
import ./firlift
import ./firvm

type
  SessionUnit* = object
    first*, last*, dispatches*: int
    signature*: string
    launches*: seq[Op]
  StepSelection* = object
    trace*: Trace
    firstSession*, sessionCount*, occurrence*, dispatches*: int
    prefillFirst*, decodeFirst*: int

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
    for p in producer.outs:
      if p.buffer == buffer:
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

proc captureZero(t: Trace; tr: TransferRec): bool =
  if tr.captureOffset < 0 or tr.captureBytes <= 0 or tr.captureBytes != tr.dst.bytes: return false
  let path = t.sourcePath & ".data"
  if t.sourcePath.len == 0 or not fileExists(path): return false
  let f = open(path, fmRead)
  defer: f.close()
  f.setFilePos(tr.captureOffset)
  var buffer = newString(min(tr.captureBytes, 1 shl 20))
  var left = tr.captureBytes
  while left > 0:
    let n = min(buffer.len, left)
    if f.readBuffer(addr buffer[0], n) != n: return false
    for c in buffer.toOpenArray(0, n-1):
      if c != '\0': return false
    left -= n
  true

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
    var earliest = first
    for order in carried:
      if order < earliest: earliest = order
    if earliest < first: result.prefillFirst = earliest
  var last = lastSession
  for op in t.ops:
    if op.order <= lastSession or op.order >= stop: continue
    if op.impl.startsWith("session:sessionUpload"): break
    last = op.order
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
