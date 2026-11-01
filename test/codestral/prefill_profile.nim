## prefill_profile.nim — BYBY_PREFILL_PROFILE=1: accumulates wall-clock time per
## stage of the parallel prefill (infer_lora.forwardParallelPrefill) and prints a summary table
## once after prefill. Only measures while gPfActive = true (toggled around prefill) so
## Linear.forward / runMicroVmChain calls outside prefill are not counted.

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
  ## Statement form. Variables declared in the body do NOT escape it.
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
  ## Timing marker for a long code section (declaring variables used afterwards).
  if pfOn(): getMonoTime() else: default(MonoTime)

proc pfAddIf*(name: string; t0: MonoTime) {.inline.} =
  if pfOn() and t0 != default(MonoTime): pfAdd(name, t0)

proc pfReset*() =
  gPfTotal.clear()
  gPfCount.clear()

proc pfReport*(wallMs: float) =
  ## Stages are nested (e.g. "blk.inProj" contains "lin.gemm" + "lin.lora", "lin.gemm" contains "fp4.*"), so % is
  ## relative to the total prefill wall clock and does NOT add up to 100%.
  if gPfTotal.len == 0: return
  var rows: seq[(string, float, int)]
  for k, v in gPfTotal: rows.add((k, v, gPfCount[k]))
  rows.sort(proc(a, b: (string, float, int)): int = cmp(b[1], a[1]))
  stderr.writeLine &"[prefill-profile] wall={wallMs:.0f}ms (nested stages, % of wall)"
  for (k, v, n) in rows:
    stderr.writeLine &"[prefill-profile]   {k:<22} {v:>10.1f}ms {100.0 * v / max(wallMs, 1e-9):>5.1f}%  n={n:<5} avg={v / float(max(n, 1)):.2f}ms"
