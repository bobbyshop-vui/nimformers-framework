## Exact-size, model-free FP4 prefill benchmark and differential check.
import std/[math, times, monotimes, strformat, tables, algorithm]
import backend, nimformer, quant, gpu_layers, fp4_gemv_gpu, fp4_gemv_vm
import vendor/bybylang/gpubackend as byby

proc run(ctx: Backend; k, n: int) =
  const rows = 15
  const padded = 16
  let be = ctx.kind.toByby()
  let name = "prefix_bench_" & $k & "_" & $n
  var qt = QuantTensor(kind: qkFp4KM, shape: @[n, k], groupSize: 32, scale: @[0.01'f32])
  qt.data = newSeq[uint8](n * (k div 2) + n * (k div 32))
  var rng = 123'u32
  for i in 0 ..< n * (k div 2):
    rng = rng * 1664525'u32 + 1013904223'u32
    qt.data[i] = uint8(rng shr 16)
  for i in n * (k div 2) ..< qt.data.len: qt.data[i] = 56'u8
  var lin = Linear(inF: k, outF: n, useQuant: true, weightQ: qt)
  discard prepareGpuLinForLinear(ctx, name, lin)
  var x = newSeq[float32](padded * k)
  for i in 0 ..< rows * k: x[i] = float32(sin(float(i) * 0.13))
  var hx = byby.sessionUpload(be, x)
  var hy = byby.sessionAllocScratch(be, padded * n)
  let kn = vmKernelForBe(buildFp4KmGemmM8Program(k, n), "bench_prefix_m8_" & $k & "_" & $n, be)
  var reference: seq[float32]
  for tiled in [false, true]:
    var samples: seq[float]
    for rep in 0 ..< 6:
      let t0 = getMonoTime()
      doAssert byby.sessionBegin(be)
      let c = gFp4CacheGpu[name]
      if tiled:
        for row in [0, 8]:
          encodeVmKernel(kn, n * GemvTg, GemvTg, (Fp4KmLutLen + 8 * GemvTg) * 4,
            hy, row * n, c.wBuf, c.lutBuf, hx, row * k)
      else: doAssert encodeFp4GemmRows(c, hx, hy, rows)
      doAssert byby.sessionEnd(be)
      if rep > 0: samples.add float((getMonoTime() - t0).inNanoseconds) / 1e6
    let output = byby.sessionRead(be, hy, rows * n)
    var err, mag: float32
    if not tiled: reference = output
    else:
      for i in 0 ..< output.len:
        doAssert classify(output[i]) notin {fcNan, fcInf, fcNegInf}
        err = max(err, abs(output[i] - reference[i])); mag = max(mag, abs(reference[i]))
      doAssert err / max(mag, 1e-6'f32) < 5e-4'f32
    samples.sort()
    let relative = err / max(mag, 1e-6'f32)
    let kernel = if tiled: "M8" else: "M4"
    echo &"K={k} N={n} M={rows} kernel={kernel} median={samples[2]:.3f}ms relativeErr={relative:.3g}"
  byby.sessionFree(be, hx); byby.sessionFree(be, hy)

when isMainModule:
  let ctx = newBackend("metal")
  doAssert byby.sessionBegin(ctx.kind.toByby())
  doAssert byby.sessionEnd(ctx.kind.toByby())
  run(ctx, 4096, 18560)
  run(ctx, 8192, 4096)
