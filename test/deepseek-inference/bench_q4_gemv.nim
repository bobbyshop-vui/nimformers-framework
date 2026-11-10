## Decode-shape Q4 GEMV (M=1) throughput: many matmuls in one session, reported
## as effective weight bandwidth. Usage: bench_q4_gemv [metal|vulkan]
import std/[os, times, random, strutils, algorithm]
import backend

proc bench(be: GpuBackend; K, N: int) =
  const group = 128
  const reps = 64
  let groups = K div group
  var rng = initRand(K * 31 + N)
  var wq = newSeq[uint8](N * K div 2)
  for i in 0 ..< wq.len: wq[i] = uint8(rng.rand(255))
  var scales = newSeq[float32](N * groups)
  var zeros = newSeq[float32](N * groups)
  for i in 0 ..< scales.len:
    scales[i] = float32(rng.rand(0.002 .. 0.02))
    zeros[i] = float32(rng.rand(6.0 .. 9.0))
  var a = newSeq[float32](K)
  for i in 0 ..< a.len: a[i] = float32(rng.rand(-1.0 .. 1.0))
  let w = residentUploadQ4(be, wq, scales, zeros, K, N, group, groups)
  discard residentPrepareQ4(w, 1)
  var times: seq[float]
  var first: seq[float32]
  for rep in 0 ..< 5:
    doAssert sessionBegin(be)
    var ha = sessionUpload(be, a)
    var hc = sessionAllocScratch(be, N)
    let t0 = epochTime()
    for i in 0 ..< reps:
      doAssert sessionMatmulQ4(be, ha, w, hc, 1)
    doAssert sessionEnd(be)
    times.add (epochTime() - t0) / reps.float
    if rep == 0: first = sessionRead(be, hc, N)
    sessionFree(be, ha)
    sessionFree(be, hc)
  times.sort()
  let ms = times[times.len div 2] * 1000
  let bytes = float(N * K div 2) + float(N * groups * 8)
  var sum = 0.0
  for v in first: sum += float(v)
  echo "  ", K, "x", N, ": ", formatFloat(ms, ffDecimal, 3), " ms, ",
    formatFloat(bytes / (ms / 1000) / 1e9, ffDecimal, 1), " GB/s (checksum ", formatFloat(sum, ffDecimal, 3), ")"

when isMainModule:
  let be = if paramCount() >= 1 and paramStr(1) == "vulkan": gbVulkan else: gbMetal
  echo "[q4-gemv] ", be
  for c in [(K: 2048, N: 2048), (K: 2048, N: 256), (K: 2048, N: 5632), (K: 5632, N: 2048), (K: 2048, N: 32000)]:
    bench(be, c.K, c.N)
