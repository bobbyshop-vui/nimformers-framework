## Selection of the replayed units on synthetic traces: the state clear that the lifted graph leaves
## to its initial contents, and the single-token units that the decode step covers.
##
##   nim c --path:vendor/bybylang -o:other/formers.n/build/test_steps other/formers.n/firlift/test_steps.nim
import std/[os, strutils, tables]
import ./firlift
import ./steps
import ./sidecar

const
  Hidden = 8        ## elements of one token row
  Arena = 1024      ## buffer b0: the arena that holds the recurrent state, elements
  Zeros = 512       ## buffer b1: an uploaded zero array, elements

var failures = 0

proc check(ok: bool; what: string) =
  if not ok:
    inc failures
    echo "FAIL: ", what

type Builder = object
  t: Trace
  data: string      ## the upload sidecar (`<trace>.data`)

proc port(name: string; buffer, bytes: int): PortRef =
  PortRef(name: name, buffer: buffer, offset: 0, bytes: bytes)

proc add(b: var Builder; impl: string; ins: seq[PortRef] = @[]; outs: seq[PortRef] = @[];
         attrs: openArray[(string, string)] = []): Op {.discardable.} =
  result = Op(id: b.t.ops.len, order: b.t.ops.len, impl: impl, parent: -1, ins: ins, outs: outs)
  for (key, value) in attrs: result.attrs[key] = value
  b.t.ops.add result

proc upload(b: var Builder; buffer, elements: int; fill: char) =
  ## A host upload of `elements` floats whose recorded bytes are all `fill`.
  let op = b.add("session:sessionUpload", outs = @[port("dst", buffer, elements*4)])
  b.t.transfers.add TransferRec(kind: "upload", op: op.id, order: op.order,
    dst: port("dst", buffer, elements*4), n: elements,
    captureOffset: b.data.len, captureBytes: elements*4)
  b.data.add repeat(fill, elements*4)

proc begin(b: var Builder) = b.add("session:sessionBegin")
proc finish(b: var Builder) = b.add("session:sessionEnd")
proc free(b: var Builder; buffer: int) = b.add("session:sessionFree", outs = @[port("buffer", buffer, 4)])

proc copy(b: var Builder; src, srcBytes, dstOff, n: int) =
  ## sessionCopyInto from the start of `src` into the arena.
  b.add("session:sessionCopyInto", ins = @[port("src", src, srcBytes)], outs = @[port("dst", 0, Arena*4)],
    attrs = {"srcOffsetElems": "0", "dstOffsetElems": $dstOff, "n": $n})

proc vec(b: var Builder; kind: string) =
  b.add("session:sessionVecOp", ins = @[port("a", 0, Arena*4), port("b", 0, Arena*4)],
    outs = @[port("c", 0, Arena*4)], attrs = {"op": kind, "n": "8"})

proc tokens(b: var Builder; buffer, rows: int; kind: string; extra = 0) =
  ## One prompt or decode session: a token-row upload, its copy into the arena, then arithmetic on
  ## the arena. `kind` and `extra` give each kind of session its own signature.
  b.begin()
  b.upload(buffer, rows*Hidden, '\x01')
  b.copy(buffer, rows*Hidden*4, 16, rows*Hidden)
  b.vec(kind)
  for i in 0 ..< extra: b.vec("sub")
  b.finish()
  b.free(buffer)

type Scenario = enum
  plain          ## clear, three prompt units (4, 1 and 4 rows), two decode steps
  writtenBefore  ## the arena is written before the clear
  dirtySource    ## the "zero" buffer was uploaded with non-zero bytes
  computeInClear ## the clearing session also computes

proc build(name: string; kind: Scenario; inMemory = false): Builder =
  result.t.sourcePath = if inMemory: MemorySource
                        else: getTempDir() / ("fir-steps-test-" & name & ".fprobe")
  result.upload(0, Arena, '\0')
  result.upload(1, Zeros, (if kind == dirtySource: '\x07' else: '\0'))
  if kind == writtenBefore:
    result.begin()
    result.vec("mul")
    result.finish()
  result.begin()
  result.copy(1, Zeros*4, 0, 64)
  result.copy(1, Zeros*4, 64, 256)
  if kind == computeInClear: result.vec("add")
  result.finish()
  result.tokens(2, 4, "relu")
  result.tokens(3, 1, "gelu")
  result.tokens(4, 4, "tanh")
  result.tokens(5, 1, "add", extra = 5)
  result.tokens(6, 1, "add", extra = 5)
  if inMemory: useMemory(result.data)       # a trace recorded through pipes has no file
  else: writeFile(result.t.sourcePath & ".data", result.data)

proc selectOf(b: Builder): StepSelection =
  selectStep(b.t, -1, "", Dims(hidden: Hidden))

proc copiesFrom(ops: seq[Op]; buffer: int): int =
  for op in ops:
    if op.impl == "session:sessionCopyInto" and op.ins.len > 0 and op.ins[0].buffer == buffer: inc result

proc uploadOrder(ops: seq[Op]; buffer: int): int =
  ## Op order of the upload into `buffer`, -1 when there is none.
  result = -1
  for op in ops:
    if op.impl == "session:sessionUpload" and op.outs.len > 0 and op.outs[0].buffer == buffer: return op.order

proc remove(b: Builder) =
  if b.t.sourcePath != MemorySource: removeFile(b.t.sourcePath & ".data")

proc main() =
  block plainScenario:
    let b = build("plain", plain)
    let sel = b.selectOf()
    check sel.sessionCount == 1 and sel.occurrence == 0, "the decode step is the selected unit"
    check sel.resetCopies == 2, "both clearing copies are left to the initial contents (got " & $sel.resetCopies & ")"
    check copiesFrom(sel.trace.ops, 1) == 0, "no copy from the zero buffer is replayed"
    check copiesFrom(sel.trace.ops, 2) == 1 and copiesFrom(sel.trace.ops, 3) == 1 and
      copiesFrom(sel.trace.ops, 4) == 1 and copiesFrom(sel.trace.ops, 5) == 1,
      "the prompt units and the decode step are replayed"
    for op in sel.trace.ops:
      check (op.phase == 0) == (op.order >= sel.decodeFirst), "only the decode step is phase 0 (op " & $op.order & ")"
    check sel.prefillFirst == uploadOrder(b.t.ops, 2), "the replay starts at the first prompt unit (got " &
      $sel.prefillFirst & ")"
    var dropped = 0
    let kept = withoutSingleRowUnits(sel.trace, sel.trace.ops, Hidden, dropped)
    check dropped == 1, "one single-token unit is dropped (got " & $dropped & ")"
    check copiesFrom(kept, 3) == 0, "the single-token unit is gone"
    check copiesFrom(kept, 2) == 1 and copiesFrom(kept, 4) == 1 and copiesFrom(kept, 5) == 1,
      "the multi-row units and the decode step stay"
    check kept.len < sel.trace.ops.len, "ops of the dropped unit are removed"
    b.remove()

  block writtenBeforeScenario:
    let b = build("written", writtenBefore)
    let sel = b.selectOf()
    check sel.resetCopies == 0, "a clear after a replayed write stays (got " & $sel.resetCopies & ")"
    check copiesFrom(sel.trace.ops, 1) == 2, "both clearing copies are replayed"
    b.remove()

  block dirtySourceScenario:
    let b = build("dirty", dirtySource)
    let sel = b.selectOf()
    check sel.resetCopies == 0, "copies from a non-zero buffer are not a clear (got " & $sel.resetCopies & ")"
    check copiesFrom(sel.trace.ops, 1) == 2, "they are replayed"
    b.remove()

  block computeScenario:
    let b = build("compute", computeInClear)
    let sel = b.selectOf()
    check sel.resetCopies == 0, "a session that also computes is not a clear (got " & $sel.resetCopies & ")"
    b.remove()

  block memoryScenario:
    # Recorded straight into the lifter: the sidecar is a string, and no file exists for it.
    let b = build("memory", plain, inMemory = true)
    check not fileExists(getTempDir() / "fir-steps-test-memory.fprobe.data"), "no sidecar file is written"
    let sel = b.selectOf()
    check sel.resetCopies == 2, "the clears are found from the in-memory sidecar (got " & $sel.resetCopies & ")"
    check copiesFrom(sel.trace.ops, 1) == 0, "no copy from the zero buffer is replayed"
    check sidecarFnv(MemorySource, 0, 16) == sidecarFnv(MemorySource, Arena*4, 16),
      "equal zero payloads hash alike"
    check sidecarAllZero(MemorySource, 0, Arena*4) and not sidecarAllZero(MemorySource, Arena*4 + Zeros*4, 16),
      "zero and non-zero ranges are told apart"
    check not sidecarAllZero(MemorySource, b.data.len - 4, 16), "a range past the end is not zero"
    memoryActive = false

  block ropeUploadsScenario:
    var b: Builder
    b.t.sourcePath = MemorySource
    b.begin()
    b.upload(2, 4*Hidden, '\x01')
    b.upload(3, Hidden, '\x02')
    let cosUpload = b.t.ops[^1]
    b.upload(4, Hidden, '\x03')
    let sinUpload = b.t.ops[^1]
    let rope = b.add("session:sessionRope",
      ins = @[port("x", 2, 4*Hidden*4), port("cosT", 3, Hidden*4), port("sinT", 4, Hidden*4)],
      outs = @[port("x", 2, 4*Hidden*4)])
    b.finish()
    for op in b.t.ops: op.phase = 1
    useMemory(b.data)
    check b.t.isRopeTableUpload(cosUpload, 3), "cos upload is a RoPE table, not a token"
    check b.t.isRopeTableUpload(sinUpload, 4), "sin upload is a RoPE table, not a token"
    check b.t.isUploadedRopeTable(rope, "cosT") and b.t.isUploadedRopeTable(rope, "sinT"),
      "RoPE resolves its host table producers"
    var dropped = 0
    let kept = withoutSingleRowUnits(b.t, b.t.ops, Hidden, dropped)
    check dropped == 0 and kept.len == b.t.ops.len, "hidden-sized RoPE tables do not drop a prompt unit"
    b.free(3)
    b.upload(3, Hidden, '\x04')
    let tokenUpload = b.t.ops[^1]
    b.add("session:sessionRmsnormEnc", ins = @[port("x", 3, Hidden*4)], outs = @[port("y", 5, Hidden*4)])
    check not b.t.isRopeTableUpload(tokenUpload, 3), "a reused RoPE buffer may contain a token embedding"
    b.upload(6, Hidden, '\x05')
    let overwrittenUpload = b.t.ops[^1]
    b.add("session:sessionVecOp", outs = @[port("c", 6, Hidden*4)])
    let computed = b.add("session:sessionRope", ins = @[port("cosT", 6, Hidden*4)])
    check not b.t.isRopeTableUpload(overwrittenUpload, 6), "overwritten upload is not a host RoPE table"
    check not b.t.isUploadedRopeTable(computed, "cosT"), "graph-computed RoPE tables keep their dataflow"
    memoryActive = false

  if failures > 0:
    echo failures, " check(s) failed"
    quit(1)
  echo "test_steps: all checks passed"

main()
