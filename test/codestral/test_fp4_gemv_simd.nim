## The simdgroup-reduction FP4-KM decode GEMV must reproduce the tuned barrier-tree kernel.
import std/[math, strformat, os, tables]
import backend, nimformer, quant, gpu_layers, fp4_gemv_gpu, fp4_gemv_vm
import vendor/bybylang/gpubackend as byby
import vendor/bybylang/micro_kernel/session_vm

proc run(ctx: Backend; k, n: int) =
  let be = ctx.kind.toByby()
  let name = "simd_gemv_test_" & $k & "_" & $n
  var qt = QuantTensor(kind: qkFp4KM, shape: @[n, k], groupSize: 32, scale: @[0.01'f32])
  qt.data = newSeq[uint8](n * (k div 2) + n * (k div 32))
  var rng = 99'u32
  for i in 0 ..< qt.data.len:
    rng = rng * 1664525'u32 + 1013904223'u32
    qt.data[i] = if i < n * (k div 2): uint8(rng shr 16) else: uint8(32 + (rng mod 48))
  var lin = Linear(inF: k, outF: n, useQuant: true, weightQ: qt)
  discard prepareGpuLinForLinear(ctx, name, lin)
  var x = newSeq[float32](k)
  for i in 0 ..< x.len: x[i] = float32(sin(float(i) * 0.13))
  var hx = byby.sessionUpload(be, x)
  var hy = byby.sessionAllocScratch(be, n)
  defer:
    byby.sessionFree(be, hx); byby.sessionFree(be, hy)
  let c = gFp4CacheGpu[name]
  proc runKernel(p: VmProgram; tag: string; tg, rows: int): seq[float32] =
    let kn = vmKernelForBe(p, tag, be)
    doAssert byby.sessionBegin(be)
    encodeVmKernel(kn, (n div rows) * tg, tg, (Fp4KmLutLen + rows * tg) * 4, hy, 0, c.wBuf, c.lutBuf, hx, 0)
    doAssert byby.sessionEnd(be), byby.sessionGetLastError(be)
    byby.sessionRead(be, hy, n)
  for tg in [32, 64, 128]:
    if k mod (tg * 8) != 0: continue
    let want = runKernel(buildFp4KmGemvTunedProgram(k, n, tg, 4), &"ref_{k}_{n}_{tg}", tg, 4)
    let got = runKernel(buildFp4KmGemvSimdProgram(k, n, tg, 4), &"simd_{k}_{n}_{tg}", tg, 4)
    var err, mag: float32
    for i, v in got:
      doAssert classify(v) notin {fcNan, fcInf, fcNegInf}
      err = max(err, abs(v - want[i])); mag = max(mag, abs(want[i]))
    doAssert err / max(mag, 1e-6'f32) < 1e-4'f32, &"K={k} N={n} tg={tg} rel={err / mag}"
    let rel = err / max(mag, 1e-6'f32)
    echo &"PASS fp4 simd gemv K={k} N={n} tg={tg} rel={rel:.3g}"

when isMainModule:
  let ctx = newBackend(getEnv("BYBY_BENCH_BACKEND", "metal"))
  run(ctx, 4096, 256)
  run(ctx, 2048, 512)
  run(ctx, 4096, 18560)
