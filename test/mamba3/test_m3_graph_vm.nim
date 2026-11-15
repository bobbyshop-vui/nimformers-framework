## Decode-graph (indirect command buffer) check for micro-VM kernels, no model needed.
##
## Records two JIT micro-VM kernels once - the token-embedding gather of the Mamba-3 graph and a
## dependent scale kernel - and replays them for several tokens. A command recorded into the
## graph needs a pipeline created for indirect command buffers. With BYBY_JIT_ICB=0 the ordinary
## JIT pipeline is recorded instead, which is expected to FAIL check 2 on Intel: that was the
## "garbage token" bug of BYBY_GRAPH=1.
##
## Build and run from test/mamba3:
##   nim c -d:release --hints:off --warnings:off -o:/tmp/test_m3_graph_vm test_m3_graph_vm.nim
##   /tmp/test_m3_graph_vm
##   BYBY_JIT_ICB=0 /tmp/test_m3_graph_vm
import gpu_layers, session_vm
import vendor/bybylang/gpubackend as byby

const
  Vocab = 8
  Hidden = 512
  Tg = 256
  X0 = 64
  Y0 = X0 + Hidden
  ArenaFloats = Y0 + Hidden

proc gatherProgram(): VmProgram =
  ## mem[X0 + g] = table[ring[0] * Hidden + g]; the ring is bound as the byte buffer.
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitUMulK(2, 0, uint32(Tg)); p.emitUAdd(2, 2, 1)
  p.emitUConst(3, 0); p.emitULoadU32(4, 3, 0)
  p.emitUMulK(4, 4, uint32(Hidden)); p.emitUAdd(4, 4, 2)
  p.emitULoadIn(1, 4, 0)
  p.emitUStoreF32(1, 2, uint32(X0))
  p.emitHalt(); p

proc scaleProgram(): VmProgram =
  ## mem[Y0 + g] = mem[X0 + g] * 2 + 1
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitUMulK(2, 0, uint32(Tg)); p.emitUAdd(2, 2, 1)
  p.emitULoadF32(1, 2, uint32(X0))
  p.emitMulConst(1, 1, 2.0'f32)
  p.emitAddConst(1, 1, 1.0'f32)
  p.emitUStoreF32(1, 2, uint32(Y0))
  p.emitHalt(); p

proc main() =
  let be = byby.gbMetal
  ensureDummy(be)
  var tableData = newSeq[float32](Vocab * Hidden)
  for v in 0 ..< Vocab:
    for j in 0 ..< Hidden:
      tableData[v * Hidden + j] = float32(v * 1000 + j)
  var table = sessionUploadPrivate(be, tableData)
  var ring = sessionAllocScratch(be, 8)
  var arena = sessionUpload(be, newSeq[float32](ArenaFloats))
  let gatherKn = vmKernelForBe(gatherProgram(), "tgv_gather_" & $Hidden, be)
  let scaleKn = vmKernelForBe(scaleProgram(), "tgv_scale_" & $Hidden, be)

  proc encodeBoth() =
    encodeVmKernel(gatherKn, Hidden, Tg, 16, arena, 0, ring, gDummyFor(be), table, 0)
    encodeVmKernel(scaleKn, Hidden, Tg, 16, arena, 0, gDummyFor(be), gDummyFor(be), gDummyFor(be), 0)

  proc check(token: int; label: string): bool =
    let mem = sessionRead(be, arena, ArenaFloats)
    var worst = 0'f32
    for j in 0 ..< Hidden:
      let row = tableData[token * Hidden + j]
      worst = max(worst, abs(mem[X0 + j] - row))
      worst = max(worst, abs(mem[Y0 + j] - (2'f32 * row + 1'f32)))
    result = worst == 0'f32
    echo label, " token ", token, ": max abs error ", worst, (if result: "  PASS" else: "  FAIL")

  var ok = true
  # Check 1: the ordinary session (no graph).
  sessionTokenRingSet(be, ring, 0, 5)
  doAssert sessionBegin(be), "sessionBegin"
  encodeBoth()
  doAssert sessionEnd(be), "sessionEnd: " & sessionGetLastError(be)
  ok = check(5, "eager") and ok

  # Check 2: recorded once, replayed for several tokens.
  var recorded = false
  for token in [3, 7, 1, 5]:
    sessionTokenRingSet(be, ring, 0, token)
    doAssert sessionBegin(be), "sessionBegin"
    if not recorded:
      if not sessionGraphBegin(be, 1):
        discard sessionEnd(be)
        echo "graph: sessionGraphBegin failed"
        quit(1)
      try:
        encodeBoth()
      except CatchableError as e:
        discard sessionGraphEnd(be)
        discard sessionEnd(be)
        echo "graph: recording raised: ", e.msg
        quit(1)
      if not sessionGraphEnd(be):
        discard sessionEnd(be)
        echo "graph: recording failed (something could not be recorded)"
        quit(1)
      recorded = true
    doAssert sessionGraphRun(be, 0, 1, 0, 0), "sessionGraphRun"
    doAssert sessionEnd(be), "sessionEnd: " & sessionGetLastError(be)
    ok = check(token, "graph") and ok

  sessionGraphReset(be)
  sessionFree(be, table)
  sessionFree(be, ring)
  sessionFree(be, arena)
  if not ok:
    echo "FAIL"
    quit(1)
  echo "PASS"

main()
