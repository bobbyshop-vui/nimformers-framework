## Exact Codestral shapes; compare dynamic groups to the existing GPU kernels.
import std/[math, monotimes, times, strformat, tables, algorithm, os]
import backend, nimformer, quant, gpu_layers, fp4_gemv_gpu, fp4_gemv_vm
import vendor/bybylang/gpubackend as byby

proc run(ctx: Backend; k, n: int) =
  let be = ctx.kind.toByby()
  let name = "dynamic_bench_" & $k & "_" & $n
  var qt = QuantTensor(kind: qkFp4KM, shape: @[n, k], groupSize: 32, scale: @[0.01'f32])
  qt.data = newSeq[uint8](n * (k div 2) + n * (k div 32))
  var rng = 123'u32
  for i in 0 ..< qt.data.len:
    rng = rng * 1664525'u32 + 1013904223'u32
    qt.data[i] = if i < n * (k div 2): uint8(rng shr 16) else: uint8(32 + (rng mod 48))
  var lin = Linear(inF: k, outF: n, useQuant: true, weightQ: qt)
  discard prepareGpuLinForLinear(ctx, name, lin)
  var x = newSeq[float32](16 * k)
  for i in 0 ..< x.len: x[i] = float32(sin(float(i) * 0.13))
  var hx = byby.sessionUpload(be, x)
  var hy = byby.sessionAllocScratch(be, 16 * n)
  defer:
    byby.sessionFree(be, hx); byby.sessionFree(be, hy)
  let c = gFp4CacheGpu[name]
  for batch in [1, 8]:
    var reference: seq[float32]
    for tg in [0, 32, 64, 128]:
      for rows in (if batch == 1: @[2, 4] else: @[8]):
        if tg == 0 and rows == 4: continue
        var kn: string
        if tg > 0:
          let p = if batch == 1: buildFp4KmGemvTunedProgram(k, n, tg, rows)
                  else: buildFp4KmGemmTunedProgram(k, n, tg, batch)
          kn = vmKernelForBe(p, &"dynamic_{k}_{n}_{tg}_{rows}_{batch}", be)
        var samples: seq[float]
        for rep in 0 ..< 4:
          let t0 = getMonoTime()
          doAssert byby.sessionBegin(be)
          if tg == 0:
            if batch == 1: doAssert encodeFp4Gemv(c, hx, 0, hy, 0)
            else: doAssert encodeFp4GemmRows(c, hx, hy, batch)
          else:
            let groups = if batch == 1: n div rows else: n
            encodeVmKernel(kn, groups * tg, tg, (Fp4KmLutLen + rows * tg) * 4,
              hy, 0, c.wBuf, c.lutBuf, hx, 0)
          doAssert byby.sessionEnd(be)
          if rep > 0: samples.add float((getMonoTime() - t0).inNanoseconds) / 1e6
        let output = byby.sessionRead(be, hy, batch * n)
        var err, mag: float32
        if tg == 0: reference = output
        else:
          for i, v in output:
            doAssert classify(v) notin {fcNan, fcInf, fcNegInf}
            err = max(err, abs(v - reference[i])); mag = max(mag, abs(reference[i]))
          doAssert err / max(mag, 1e-6'f32) < 1e-4'f32
        samples.sort()
        let relative = err / max(mag, 1e-6'f32)
        echo &"K={k} N={n} M={batch} TG={tg} rows={rows} median={samples[1]:.3f}ms rel={relative:.3g}"

when isMainModule:
  let ctx = newBackend(getEnv("BYBY_BENCH_BACKEND", "metal"))
  run(ctx, 4096, 18560)
  run(ctx, 8192, 4096)
