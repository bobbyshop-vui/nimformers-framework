## bench_dispatch_overhead.nim
## Micro-benchmark for the bybylang session dispatch path.
##
## Purpose: test the hypothesis that Codestral decode on the target GPU is
## dominated by dispatch-encoding overhead rather than GPU compute.
## encodeOneToken issues ~448 dispatches per token (64 blocks x 7 ops, plus
## extra for G blocks); at ~1.6 s/token that is ~3.6 ms per dispatch, which is
## far above any plausible kernel execution time for these sizes.
##
## This benchmark allocates two scratch buffers and encodes N vecOp dispatches
## inside ONE session, splitting the measurement into sessionBegin / encode
## loop / sessionEnd. No model is loaded, so it runs in seconds.
##
## Usage:
##   nim c -d:release --opt:speed -r bench_dispatch_overhead.nim
##   BYBY_BENCH_N=100 nim c -d:release --opt:speed -r bench_dispatch_overhead.nim

import std/[os, strutils, strformat, times]
import backend
import vendor/bybylang/gpubackend as byby

proc main() =
  let ctx = newBackend("auto")
  let be = ctx.kind.toByby()
  let n = parseInt(getEnv("BYBY_BENCH_N", "2000"))
  let sz = parseInt(getEnv("BYBY_BENCH_SZ", "4096"))
  var hA = byby.sessionAllocScratch(be, sz)
  var hB = byby.sessionAllocScratch(be, sz)
  var hC = byby.sessionAllocScratch(be, sz)

  # Warm the pipeline cache so first-encode JIT compile does not skew timing.
  if byby.sessionBegin(be):
    for _ in 0 ..< 10: discard byby.sessionVecOp(be, "add", hA, hB, hC, sz)
    discard byby.sessionEnd(be)

  let t0 = epochTime()
  if not byby.sessionBegin(be):
    echo "sessionBegin failed"
    quit(1)
  let t1 = epochTime()
  for _ in 0 ..< n:
    discard byby.sessionVecOp(be, "add", hA, hB, hC, sz)
  let t2 = epochTime()
  discard byby.sessionEnd(be)
  let t3 = epochTime()

  let encMs = (t2 - t1) * 1000.0
  let endMs = (t3 - t2) * 1000.0
  echo &"backend={be} N={n} size={sz}"
  echo &"begin={(t1 - t0) * 1000.0:.2f}ms encode={encMs:.2f}ms end={endMs:.2f}ms"
  echo &"per-encode={encMs * 1000.0 / float(n):.1f}us  encode/end-ratio={(t2 - t1) / max(t3 - t2, 1e-9):.1f}"

main()
