## Prefill Q4 GEMM (sessionMatmulQ4 with M > 1) against a CPU reference, plus
## its speed at the DeepSeek 6.7B projection shapes. Random asymmetric Q4
## weights (group 128), so it runs without the model.
## Usage: test_q4_gemm [metal|vulkan] [M]
## BYBY_METAL_Q4_GEMM=0 times the previous Metal kernels for comparison.
import std/[os, times, random, strutils, algorithm]
import backend

proc reference(a: seq[float32]; wq: seq[uint8]; scales, zeros: seq[float32];
               M, K, N, group: int): seq[float32] =
  let groups = K div group
  result = newSeq[float32](M * N)
  var w = newSeq[float32](K)
  for n in 0 ..< N:
    for k in 0 ..< K:
      let b = wq[n * (K div 2) + k div 2]
      let q = if k mod 2 == 0: b and 15 else: b shr 4
      let g = n * groups + k div group
      w[k] = (float32(q) - zeros[g]) * scales[g]
    for m in 0 ..< M:
      var s = 0.0
      for k in 0 ..< K: s += float(a[m * K + k]) * float(w[k])
      result[m * N + n] = float32(s)

proc runShape(be: GpuBackend; M, K, N: int; check: bool): bool =
  const group = 128
  let groups = K div group
  var rng = initRand(K * 31 + N)
  var wq = newSeq[uint8](N * K div 2)
  for i in 0 ..< wq.len: wq[i] = uint8(rng.rand(255))
  var scales = newSeq[float32](N * groups)
  var zeros = newSeq[float32](N * groups)
  for i in 0 ..< scales.len:
    scales[i] = float32(rng.rand(0.002 .. 0.02))
    zeros[i] = float32(rng.rand(6.0 .. 9.0))
  var a = newSeq[float32](M * K)
  for i in 0 ..< a.len: a[i] = float32(rng.rand(-1.0 .. 1.0))
  let w = residentUploadQ4(be, wq, scales, zeros, K, N, group, groups)
  var times: seq[float]
  var got: seq[float32]
  for rep in 0 ..< 6:
    doAssert sessionBegin(be)
    var ha = sessionUpload(be, a)
    var hc = sessionAllocScratch(be, M * N)
    let t0 = epochTime()
    let ok = sessionMatmulQ4(be, ha, w, hc, M)
    doAssert sessionEnd(be) and ok, "sessionMatmulQ4 failed"
    times.add epochTime() - t0
    if rep == 0: got = sessionRead(be, hc, M * N)
    sessionFree(be, ha)
    sessionFree(be, hc)
  times.sort()
  let ms = times[times.len div 2] * 1000
  let gflops = 2.0 * float(M) * float(K) * float(N) / (ms / 1000) / 1e9
  result = true
  var info = ""
  if check:
    let expected = reference(a, wq, scales, zeros, M, K, N, group)
    var err = 0.0
    var scaleRef = 0.0
    for i in 0 ..< got.len:
      err = max(err, abs(float(got[i]) - float(expected[i])))
      scaleRef = max(scaleRef, abs(float(expected[i])))
    let rel = err / max(scaleRef, 1e-9)
    result = rel < 1e-4
    info = "  max rel err " & formatFloat(rel, ffScientific, 2) & (if result: " PASS" else: " FAIL")
  echo "  ", K, "x", N, " M=", M, ": ", formatFloat(ms, ffDecimal, 2), " ms, ",
    formatFloat(gflops, ffDecimal, 1), " GFLOPS", info

when isMainModule:
  let be = if paramCount() >= 1 and paramStr(1) == "vulkan": gbVulkan else: gbMetal
  let M = if paramCount() >= 2: parseInt(paramStr(2)) else: 64
  echo "[q4-gemm] ", be, " M=", M
  var ok = true
  # Small shapes (also odd M) checked against the CPU, then the model shapes timed.
  for c in [(M: 9, K: 256, N: 192), (M: 37, K: 512, N: 130), (M: 64, K: 1024, N: 256)]:
    ok = runShape(be, c.M, c.K, c.N, true) and ok
  ok = runShape(be, M, 4096, 512, true) and ok
  for c in [(K: 4096, N: 4096), (K: 4096, N: 11008), (K: 11008, N: 4096)]:
    discard runShape(be, M, c.K, c.N, false)
  if not ok: quit "q4 gemm test FAILED", 1
  echo "q4 gemm test PASSED"
