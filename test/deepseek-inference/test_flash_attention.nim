## Prefill flash attention + KV scatter on Metal / Vulkan against an FP64 reference.
## GQA, a continued prompt (pos > 0) and NaN past the filled cache, so a wrong
## mask or an out-of-range read shows up as an error.
## Usage: test_flash_attention [metal|vulkan|all]
import std/[os, math, strutils]
import gpubackend

proc reference(q, k, v: seq[float32]; T, H, HKV, D, pos, maxSeqLen: int; scale: float32): seq[float32] =
  result = newSeq[float32](T * H * D)
  let rep = H div HKV
  var sc = newSeq[float64](pos + T)
  for t in 0 ..< T:
    for h in 0 ..< H:
      let base = (h div rep) * maxSeqLen * D
      let n = pos + t + 1
      var mx = -1e300
      for key in 0 ..< n:
        var s = 0.0
        for d in 0 ..< D: s += float64(q[(t * H + h) * D + d]) * float64(k[base + key * D + d])
        sc[key] = s * float64(scale)
        mx = max(mx, sc[key])
      var total = 0.0
      for key in 0 ..< n:
        sc[key] = exp(sc[key] - mx)
        total += sc[key]
      for d in 0 ..< D:
        var acc = 0.0
        for key in 0 ..< n: acc += sc[key] * float64(v[base + key * D + d])
        result[(t * H + h) * D + d] = float32(acc / total)

var seed = 12345'u32
proc rnd(): float32 =
  seed = seed * 1664525'u32 + 1013904223'u32
  float32(seed shr 8) / float32(1 shl 24) * 2.0'f32 - 1.0'f32

proc runCase(be: GpuBackend; T, H, HKV, D, pos, maxSeqLen: int): bool =
  let kind = sessionFlashAttentionPrepare(be, D)
  if kind.len == 0:
    echo "  SKIP D=", D, ": no flash-attention kernel on this backend"
    return true
  # Cache: positions < pos come from earlier turns, the T new ones go through
  # the scatter kernel, the rest stays NaN.
  var k = newSeq[float32](HKV * maxSeqLen * D)
  var v = newSeq[float32](HKV * maxSeqLen * D)
  for i in 0 ..< k.len:
    k[i] = NaN
    v[i] = NaN
  for h in 0 ..< HKV:
    for key in 0 ..< pos:
      for d in 0 ..< D:
        k[(h * maxSeqLen + key) * D + d] = rnd()
        v[(h * maxSeqLen + key) * D + d] = rnd()
  var kNew = newSeq[float32](T * HKV * D)
  var vNew = newSeq[float32](T * HKV * D)
  for i in 0 ..< kNew.len:
    kNew[i] = rnd()
    vNew[i] = rnd()
  var q = newSeq[float32](T * H * D)
  for i in 0 ..< q.len: q[i] = rnd() * 4.0'f32
  var kFull = k
  var vFull = v
  for t in 0 ..< T:
    for h in 0 ..< HKV:
      for d in 0 ..< D:
        kFull[(h * maxSeqLen + pos + t) * D + d] = kNew[(t * HKV + h) * D + d]
        vFull[(h * maxSeqLen + pos + t) * D + d] = vNew[(t * HKV + h) * D + d]
  let scale = 1.0'f32 / sqrt(float32(D))
  let expected = reference(q, kFull, vFull, T, H, HKV, D, pos, maxSeqLen, scale)

  doAssert sessionBegin(be), "sessionBegin failed"
  var hq = sessionUpload(be, q)
  var hk = sessionUpload(be, k)
  var hv = sessionUpload(be, v)
  var hkn = sessionUpload(be, kNew)
  var hvn = sessionUpload(be, vNew)
  var ho = sessionAllocScratch(be, T * H * D)
  let okK = sessionKvScatterEnc(be, hkn, hk, T, HKV, D, pos, maxSeqLen)
  let okV = sessionKvScatterEnc(be, hvn, hv, T, HKV, D, pos, maxSeqLen)
  let okA = sessionFlashAttentionPrefillEnc(be, hq, hk, hv, ho, T, H, HKV, D, pos, maxSeqLen, scale)
  doAssert sessionEnd(be), "sessionEnd failed: " & sessionGetLastError(be)
  doAssert okK and okV and okA, "encode failed (scatter K/V, attention): " & $okK & " " & $okV & " " & $okA
  let got = sessionRead(be, ho, T * H * D)
  let kGot = sessionRead(be, hk, HKV * maxSeqLen * D)
  for h in [hq, hk, hv, hkn, hvn, ho]:
    var hh = h
    sessionFree(be, hh)
  var scatterOk = true
  for h in 0 ..< HKV:
    for key in 0 ..< pos + T:
      for d in 0 ..< D:
        let i = (h * maxSeqLen + key) * D + d
        if kGot[i] != kFull[i]: scatterOk = false
  var err = 0.0
  for i in 0 ..< got.len:
    let e = abs(float(got[i]) - float(expected[i]))
    err = if e != e: Inf else: max(err, e)
  let tol = if kind == "coop": 2e-2 else: 1e-4
  result = scatterOk and err <= tol
  echo "  ", (if result: "PASS" else: "FAIL"), " ", kind, " T=", T, " H=", H, " HKV=", HKV,
    " D=", D, " pos=", pos, ": max err ", formatFloat(err, ffScientific, 2),
    (if scatterOk: "" else: " (KV scatter wrong)")

proc runBackend(be: GpuBackend): bool =
  echo "[flash-attention] ", be
  result = true
  for c in [(T: 1, H: 2, HKV: 1, D: 64, pos: 0), (T: 37, H: 4, HKV: 2, D: 128, pos: 11),
            (T: 86, H: 8, HKV: 8, D: 128, pos: 0), (T: 20, H: 4, HKV: 1, D: 96, pos: 50)]:
    if not runCase(be, c.T, c.H, c.HKV, c.D, c.pos, 128): result = false

when isMainModule:
  let which = if paramCount() >= 1: paramStr(1) else: "all"
  var ok = true
  if which in ["metal", "all"]:
    when defined(macosx):
      ok = runBackend(gbMetal) and ok
  if which in ["vulkan", "all"]:
    ok = runBackend(gbVulkan) and ok
  if not ok: quit "flash attention test FAILED", 1
  echo "flash attention test PASSED"
