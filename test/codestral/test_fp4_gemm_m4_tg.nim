## The prefill M4 tile (4 input rows per dispatch) must match four single-row GEMVs whatever GEMV
## plan the autotuner picked. The tile used to take its threadgroup memory size from the tuned GEMV
## plan; the short-K decode GEMVs need far less, so the tile ran with too little threadgroup memory
## and corrupted every prompt with a 4-row remainder.
##
##   nim c -r -d:release test/codestral/test_fp4_gemm_m4_tg.nim
##   BYBY_BENCH_BACKEND=vulkan test/codestral/test_fp4_gemm_m4_tg
import std/[math, strformat, os, tables]
import backend, nimformer, quant, gpu_layers, fp4_gemv_gpu, fp4_gemv_vm
import vendor/bybylang/gpubackend as byby

proc run(ctx: Backend; k, n: int) =
  let be = ctx.kind.toByby()
  let name = "m4_tg_test_" & $k & "_" & $n
  var qt = QuantTensor(kind: qkFp4KM, shape: @[n, k], groupSize: 32, scale: @[0.01'f32])
  qt.data = newSeq[uint8](n * (k div 2) + n * (k div 32))
  var rng = 7'u32
  for i in 0 ..< qt.data.len:
    rng = rng * 1664525'u32 + 1013904223'u32
    qt.data[i] = if i < n * (k div 2): uint8(rng shr 16) else: uint8(32 + (rng mod 48))
  var lin = Linear(inF: k, outF: n, useQuant: true, weightQ: qt)
  discard prepareGpuLinForLinear(ctx, name, lin)
  var c = gFp4CacheGpu[name]
  doAssert c.m4JitName.len > 0, "no M4 tile for this shape"
  var x = newSeq[float32](4 * k)
  for i in 0 ..< x.len: x[i] = float32(sin(float(i) * 0.37))
  var hx = byby.sessionUpload(be, x)
  var hy = byby.sessionAllocScratch(be, 4 * n)
  defer:
    byby.sessionFree(be, hx); byby.sessionFree(be, hy)
  doAssert byby.sessionBegin(be)
  for row in 0 ..< 4:
    doAssert encodeFp4Gemv(c, hx, row * k, hy, row * n)
  doAssert byby.sessionEnd(be), byby.sessionGetLastError(be)
  let want = byby.sessionRead(be, hy, 4 * n)
  # The smallest plan the autotuner can pick: a 4-row, 16-lane decode GEMV.
  var small = c
  small.tgBytes = fp4KmGemvDecodeTgBytes(4, 16)
  for (label, plan) in [("tuned", c), ("smallest", small)]:
    doAssert byby.sessionBegin(be)
    doAssert encodeFp4GemmM4(plan, hx, 0, hy, 0)
    doAssert byby.sessionEnd(be), byby.sessionGetLastError(be)
    let got = byby.sessionRead(be, hy, 4 * n)
    var err, mag: float32
    for i, v in got:
      doAssert classify(v) notin {fcNan, fcInf, fcNegInf}
      err = max(err, abs(v - want[i])); mag = max(mag, abs(want[i]))
    let rel = err / max(mag, 1e-6'f32)
    doAssert rel < 1e-4'f32, &"K={k} N={n} plan={label} rel={rel}"
    echo &"PASS fp4 M4 tile K={k} N={n} plan={label} (GEMV tgBytes={plan.tgBytes}) rel={rel:.3g}"

when isMainModule:
  let ctx = newBackend(getEnv("BYBY_BENCH_BACKEND", "metal"))
  run(ctx, 768, 3072)
  run(ctx, 1536, 768)
