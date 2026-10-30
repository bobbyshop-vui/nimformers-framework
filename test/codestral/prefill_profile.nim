## prefill_profile.nim — BYBY_PREFILL_PROFILE=1: cong don thoi gian wall-clock theo
## stage trong parallel prefill (infer_lora.forwardParallelPrefill), in bang tong ket
## 1 lan sau prefill. Chi do khi gPfActive = true (bat/tat quanh prefill) de cac
## lan goi Linear.forward / runMicroVmChain ngoai prefill khong bi tinh vao.

import std/[os, monotimes, times, tables, algorithm, strformat]
export monotimes

var gPfEnabled* = getEnv("BYBY_PREFILL_PROFILE", "1") != "0"
var gPfActive* = false
var gPfTotal = initOrderedTable[string, float]()   ## ms
var gPfCount = initOrderedTable[string, int]()

proc pfOn*(): bool {.inline.} = gPfEnabled and gPfActive

proc pfAdd*(name: string; t0: MonoTime) =
  let ms = float((getMonoTime() - t0).inNanoseconds) / 1e6
  gPfTotal.mgetOrPut(name, 0.0) += ms
  gPfCount.mgetOrPut(name, 0) += 1

template pfTime*(name: string; body: untyped) =
  ## Statement form. Bien khai bao trong body KHONG thoat ra ngoai.
  if pfOn():
    let pfT0 = getMonoTime()
    body
    pfAdd(name, pfT0)
  else:
    body

template pfExpr*(name: string; e: untyped): untyped =
  ## Expression form: `let y = pfExpr("stage", f(x))`.
  block:
    let pfOnNow = pfOn()
    let pfT0 = if pfOnNow: getMonoTime() else: default(MonoTime)
    let pfR = e
    if pfOnNow: pfAdd(name, pfT0)
    pfR

proc getMonoTimeIfPf*(): MonoTime {.inline.} =
  ## Moc thoi gian cho doan code dai (co khai bao bien dung tiep phia sau).
  if pfOn(): getMonoTime() else: default(MonoTime)

proc pfAddIf*(name: string; t0: MonoTime) {.inline.} =
  if pfOn() and t0 != default(MonoTime): pfAdd(name, t0)

proc pfReset*() =
  gPfTotal.clear()
  gPfCount.clear()

proc pfReport*(wallMs: float) =
  ## Cac stage long nhau (vd "blk.inProj" chua "lin.gemm" + "lin.lora", "lin.gemm" chua "fp4.*"), nen % la
  ## so voi tong wall-clock prefill, KHONG cong lai thanh 100%.
  if gPfTotal.len == 0: return
  var rows: seq[(string, float, int)]
  for k, v in gPfTotal: rows.add((k, v, gPfCount[k]))
  rows.sort(proc(a, b: (string, float, int)): int = cmp(b[1], a[1]))
  stderr.writeLine &"[prefill-profile] wall={wallMs:.0f}ms (stage long nhau, % theo wall)"
  for (k, v, n) in rows:
    stderr.writeLine &"[prefill-profile]   {k:<22} {v:>10.1f}ms {100.0 * v / max(wallMs, 1e-9):>5.1f}%  n={n:<5} avg={v / float(max(n, 1)):.2f}ms"
