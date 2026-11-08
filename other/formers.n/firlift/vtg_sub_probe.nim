## Checks the sub-threadgroup virtual-TG mode for every tg size that divides 256: each virtual
## threadgroup v and lane l must write v*1000+l exactly once, whatever the physical grid is.
import std/[os, strutils]
import vendor/bybylang/gpubackend as byby
import session_vm
import session_vm_dispatch
import ../../../test/codestral/od_core

proc probe(be: byby.GpuBackend; sub, nVirt, ntg: int): bool =
  var body: VmProgram
  body.emitUTgid(0); body.emitULid(1)
  body.emitUMulK(2, 0, uint32(sub)); body.emitUAdd(2, 2, 1)          # slot = v*sub + l
  body.emitUMulK(3, 0, 1000); body.emitUAdd(3, 3, 1); body.emitUToR(1, 3)  # value = v*1000+l
  body.emitUStoreF32(1, 2, 64)
  body.emitHalt()
  var p: VmProgram
  odStage(p, body, 1, uint32(nVirt), uint32(sub shl 20) or 16'u32)
  p.emitHalt()
  let kname = odCompile(p, "vtgsub" & $sub & "_" & $nVirt & "_" & $ntg)
  let n = nVirt * sub
  var hMem = byby.sessionUploadPrivate(be, newSeq[float32](64 + n))
  var hW = byby.sessionUpload(be, newSeq[float32](4))
  var hLut = byby.sessionUpload(be, newSeq[float32](4))
  if not byby.sessionBegin(be): raise newException(CatchableError, "sessionBegin failed")
  odDispatch(kname, hMem, hW, hLut, ntg, 16 * 4 * (256 div sub))
  discard byby.sessionEnd(be)
  let got = byby.sessionRead(be, hMem, 64 + n)
  result = true
  for v in 0 ..< nVirt:
    for l in 0 ..< sub:
      if got[64 + v * sub + l] != float32(v * 1000 + l): result = false
  echo "sub=", sub, " nVirt=", nVirt, " ntg=", ntg, if result: "  OK" else: "  FAIL"

for (sub, nv, ntg) in [(32, 37, 3), (64, 21, 2), (128, 9, 4), (256, 11, 3), (16, 70, 2)]:
  if not probe(byby.gbMetal, sub, nv, ntg): quit(1)
