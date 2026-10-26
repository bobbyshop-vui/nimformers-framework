## t_grid_barrier.nim — KIEM TRA PHAN CUNG truoc khi tin vao kien truc 1-dispatch.
##
## Hoi 3 cau (chay tren chinh may Iris cua ban):
##   1. OP_GRID_BARRIER (atomic + spin) co chay xong khong hay deadlock? (so TG toi da resident?)
##   2. Du lieu ghi boi TG khac TRUOC barrier co THAY DUOC sau barrier khong (coherence)?
##   3. Barrier ton bao nhieu us?  (so voi ~3.6 ms / dispatch hien tai)
##
## Chay:
##   nim c -d:release --opt:speed -r t_grid_barrier.nim
##   BYBY_OD_TG_LIST=1,2,4,8,16,24,32 BYBY_OD_REPS=5 nim c -d:release -r t_grid_barrier.nim
##
## PASS = moi pha doc dung gia tri pha truoc cua TG DOI DIEN (mirror) va lastErr == 0.
## FAIL (sai gia tri) = khong coherent -> KHONG dung grid barrier tren may nay.
## ERR  (lastErr=1)   = spin het gioi han -> so TG khong dong thoi resident -> giam ntg.

import std/[os, strutils, strformat, times]
import backend
import vendor/bybylang/gpubackend as byby
import session_vm
import od_core

const
  Phases = 64
  SaltIdx = 8                    ## float index (trong khoi ctrl) giu "muoi" cua lan chay
  DataBase = OdCtrlFloats        ## A[k][t] nam tu day

proc buildProg(ntg: int): VmProgram =
  let total = ntg * OdTg
  var p: VmProgram
  for k in 0 ..< Phases:
    var b: VmProgram
    let base = uint32(DataBase + k * total)
    if k == 0:
      b.emitLoadConstF32(4, 0.0'f32)
      b.emitLoadF32(3, 4, uint32(SaltIdx))      # R3 = salt
      b.emitAddConst(1, 0, 1.0'f32)             # R1 = tid + 1
      b.emitAdd(1, 1, 3)
      b.emitStoreF32(1, 0, base)                # A[0][tid] = tid + 1 + salt
    else:
      let prev = uint32(DataBase + (k - 1) * total)
      b.emitMulConst(1, 0, -1.0'f32)
      b.emitAddConst(1, 1, float32(total - 1))  # R1 = mirror(tid)
      b.emitLoadF32(2, 1, prev)                 # R2 = A[k-1][mirror]  (do TG KHAC ghi)
      b.emitAddConst(2, 2, 1.0'f32)
      b.emitStoreF32(2, 0, base)
    b.emitHalt()
    odStage(p, b, 0'u32, uint32(ntg), 0'u32)
    p.emitGridBarrier(0)
  p.emitGridEpilogue(0)
  p.emitHalt()
  result = p

proc runOne(be: byby.GpuBackend; ntg, reps: int): bool =
  let total = ntg * OdTg
  let nFloats = DataBase + Phases * total
  var hMem = byby.sessionUploadPrivate(be, newSeq[float32](nFloats))
  var hW = byby.sessionUploadPrivate(be, newSeq[float32](64))
  var hLut = byby.sessionUploadPrivate(be, newSeq[float32](64))
  let kname = odCompile(buildProg(ntg), "t_grid_" & $ntg)
  var okAll = true
  var tTot = 0.0
  for r in 0 ..< reps:
    let salt = float32(r * 7 + 3)
    var hSalt = byby.sessionUpload(be, @[salt])
    if not byby.sessionBegin(be):
      echo "  sessionBegin that bai"; return false
    if not byby.sessionCopyInto(be, hSalt, hMem, 0, SaltIdx, 1):
      discard byby.sessionEnd(be); echo "  copyInto salt that bai"; return false
    odDispatch(kname, hMem, hW, hLut, ntg, 16)
    let t0 = epochTime()
    let ended = byby.sessionEnd(be)
    let dt = epochTime() - t0
    tTot += dt
    if not ended:
      echo "  sessionEnd that bai: ", byby.sessionGetLastError(be)
      discard byby.sessionRecoverQueue(be)
      return false
    let v = byby.sessionRead(be, hMem, nFloats)
    var bad = 0
    var firstBad = -1
    for k in 0 ..< Phases:
      for t in 0 ..< total:
        let src = if k mod 2 == 0: t else: total - 1 - t
        let expect = float32(src) + 1.0'f32 + salt + float32(k)
        if v[DataBase + k * total + t] != expect:
          inc bad
          if firstBad < 0: firstBad = k * total + t
    var cnt, done, err, lastErr: uint32
    copyMem(addr cnt, unsafeAddr v[0], 4); copyMem(addr done, unsafeAddr v[1], 4)
    copyMem(addr err, unsafeAddr v[2], 4); copyMem(addr lastErr, unsafeAddr v[3], 4)
    let pass = bad == 0 and lastErr == 0 and cnt == 0 and done == 0 and err == 0
    echo &"  rep={r} {(if pass: \"PASS\" else: \"FAIL\")} bad={bad} firstBad={firstBad} cnt={cnt} done={done} err={err} lastErr={lastErr} sessionEnd={dt*1000.0:.2f}ms"
    if not pass: okAll = false
  let perBarrier = tTot / float(reps) / float(Phases) * 1e6
  echo &"  ntg={ntg}: ~{perBarrier:.1f} us / (stage+barrier)   (gom ca thoi gian cho GPU, 1 dispatch = {Phases} barrier)"
  okAll

proc main() =
  let ctx = newBackend("auto")
  let be = ctx.kind.toByby()
  echo "backend=", be
  let reps = parseInt(getEnv("BYBY_OD_REPS", "3"))
  var good: seq[int]
  for s in getEnv("BYBY_OD_TG_LIST", "1,2,4,8,16,32").split(','):
    let ntg = parseInt(s.strip())
    echo "── ntg=", ntg, " (", ntg * OdTg, " thread) ──"
    var ok = false
    try: ok = runOne(be, ntg, reps)
    except CatchableError as e: echo "  EXCEPTION: ", e.msg
    if ok: good.add ntg
  echo "TG hop le (PASS): ", good
  echo "=> dat BYBY_OD_TG = so lon nhat trong danh sach tren (hoac nho hon mot nac de du phong)."

main()
