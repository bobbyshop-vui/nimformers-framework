## Hardware check for the software grid barrier (OP_GRID_BARRIER). Launches N threadgroups of
## 256 threads, each writes a marker, waits at the barrier, then reads the marker of the mirrored
## threadgroup. It passes only if every threadgroup was resident at the same time. The barrier has a
## built-in spin limit, so a non-resident configuration reports an error instead of hanging forever.
## usage: grid_barrier_probe [ntg ...]   (default 2 4 8 16 32)
import std/[os, strutils]
import vendor/bybylang/gpubackend as byby
import session_vm
import session_vm_dispatch
import ../../../test/codestral/od_core

proc probe(be: byby.GpuBackend; ntg: int): bool =
  var p: VmProgram
  p.emitUTgid(0)
  p.emitUToR(1, 0); p.emitAddConst(1, 1, 1.0'f32)
  p.emitUStoreF32(1, 0, 64)
  p.emitGridBarrier(0)
  p.emitUMulK(3, 0, 0xFFFFFFFF'u32); p.emitUAddK(3, 3, uint32(ntg - 1))
  p.emitULoadF32(2, 3, 64)
  p.emitUStoreF32(2, 0, 128)
  p.emitGridEpilogue(0)
  p.emitHalt()
  let kname = odCompile(p, "gridprobe" & $ntg)
  var data = newSeq[float32](192)
  var hMem = byby.sessionUploadPrivate(be, data)
  var hW = byby.sessionUpload(be, newSeq[float32](4))
  var hLut = byby.sessionUpload(be, newSeq[float32](4))
  if not byby.sessionBegin(be): raise newException(CatchableError, "sessionBegin failed")
  odDispatch(kname, hMem, hW, hLut, ntg, 16)
  discard byby.sessionEnd(be)
  let got = byby.sessionRead(be, hMem, 192)
  var errBits: uint32
  copyMem(addr errBits, unsafeAddr got[3], 4)
  result = errBits == 0
  for t in 0 ..< ntg:
    if got[128 + t] != float32(ntg - t): result = false
  echo "ntg=", ntg, if result: "  OK (all threadgroups resident)" else: "  FAIL (barrier spun out or wrong data), err=" & $errBits

proc main() =
  let be = byby.gbMetal
  var counts: seq[int]
  for i in 1 .. paramCount(): counts.add parseInt(paramStr(i))
  if counts.len == 0: counts = @[2, 4, 8, 16, 32]
  for n in counts:
    if not probe(be, n): quit(1)
main()
