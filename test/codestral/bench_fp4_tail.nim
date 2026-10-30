## Model-free timing of the prefill FP4 tiling options on real packed weights:
## the tuned 8-row tile (wide8) and the tuned 4-row tail tile that the resident
## prefill uses for the last 1..4 rows of a prompt.
##   nim c -d:release --mm:arc --opt:speed -o:bench_fp4_tail bench_fp4_tail.nim
##   BYBY_BENCH_BACKEND=vulkan ./bench_fp4_tail
import std/[os, strutils, math, times, monotimes, strformat, algorithm, tables]
import backend, nimformer, quant, gpu_layers, fp4_gemv_gpu, fp4_gemv_vm
import vendor/bybylang/gpubackend as byby

proc median(xs: var seq[float]): float =
  xs.sort()
  xs[xs.len div 2]

proc run(ctx: Backend; k, n: int) =
  let be = ctx.kind.toByby()
  let name = "tail_bench_" & $k & "_" & $n
  var qt = QuantTensor(kind: qkFp4KM, shape: @[n, k], groupSize: 32, scale: @[0.01'f32])
  qt.data = newSeq[uint8](n * (k div 2) + n * (k div 32))
  var rng = 123'u32
  for i in 0 ..< n * (k div 2):
    rng = rng * 1664525'u32 + 1013904223'u32
    qt.data[i] = uint8(rng shr 16)
  for i in n * (k div 2) ..< qt.data.len: qt.data[i] = 56'u8
  var lin = Linear(inF: k, outF: n, useQuant: true, weightQ: qt)
  discard prepareGpuLinForLinear(ctx, name, lin)
  const rows = 16
  var x = newSeq[float32](rows * k)
  for i in 0 ..< rows * k: x[i] = float32(sin(float(i) * 0.13))
  var hx = byby.sessionUpload(be, x)
  var hy = byby.sessionAllocScratch(be, rows * n)
  let c = gFp4CacheGpu[name]
  let kn8 = if c.m8JitName.len > 0: c.m8JitName
            else: vmKernelForBe(buildFp4KmGemmM8Program(k, n), "tail_m8_" & $k & "_" & $n, be)
  let tg8 = if c.m8TgSize > 0: c.m8TgSize else: GemvTg
  let kn4 = vmKernelForBe(buildFp4KmGemmTunedProgram(k, n, GemvTg, 4),
    "tail_m4_" & $k & "_" & $n, be)
  var t8: seq[float]
  var t4: seq[float]
  for rep in 0 ..< 8:
    let t0 = getMonoTime()
    doAssert byby.sessionBegin(be)
    encodeVmKernel(kn8, n * tg8, tg8, (Fp4KmLutLen + 8 * tg8) * 4,
      hy, 0, c.wBuf, c.lutBuf, hx, 0)
    doAssert byby.sessionEnd(be)
    if rep > 0: t8.add float((getMonoTime() - t0).inNanoseconds) / 1e6
    let t1 = getMonoTime()
    doAssert byby.sessionBegin(be)
    encodeVmKernel(kn4, n * GemvTg, GemvTg, (Fp4KmLutLen + 4 * GemvTg) * 4,
      hy, 0, c.wBuf, c.lutBuf, hx, 0)
    doAssert byby.sessionEnd(be)
    if rep > 0: t4.add float((getMonoTime() - t1).inNanoseconds) / 1e6
  let m8 = median(t8)
  let m4 = median(t4)
  echo &"K={k} N={n}: 8-row={m8:.3f}ms, 4-row={m4:.3f}ms, " &
    &"per-row {m8 / 8:.3f} vs {m4 / 4:.3f} ms -> 12 rows = {m8 + m4:.2f}ms vs 16 rows = {2 * m8:.2f}ms"
  # Correctness: the 4-row tile must reproduce the first 4 rows of the 8-row
  # tile exactly (same weight decode + same per-row FMA order).
  doAssert byby.sessionBegin(be)
  encodeVmKernel(kn8, n * tg8, tg8, (Fp4KmLutLen + 8 * tg8) * 4,
    hy, 0, c.wBuf, c.lutBuf, hx, 0)
  doAssert byby.sessionEnd(be)
  let ref8 = byby.sessionRead(be, hy, 8 * n)
  doAssert byby.sessionBegin(be)
  encodeVmKernel(kn4, n * GemvTg, GemvTg, (Fp4KmLutLen + 4 * GemvTg) * 4,
    hy, 0, c.wBuf, c.lutBuf, hx, 0)
  doAssert byby.sessionEnd(be)
  let out4 = byby.sessionRead(be, hy, 4 * n)
  var err, mag: float32
  for i in 0 ..< out4.len:
    doAssert classify(out4[i]) notin {fcNan, fcInf, fcNegInf}
    err = max(err, abs(out4[i] - ref8[i]))
    mag = max(mag, abs(ref8[i]))
  let relErr = err / max(mag, 1e-6'f32)
  doAssert relErr < 1e-5'f32, "4-row tail mismatch: " & $relErr
  echo &"  4-row tail matches 8-row tile rows 0..3: maxAbsErr={err:.3g} (relative {relErr:.3g})"
  byby.sessionFree(be, hx)
  byby.sessionFree(be, hy)

when isMainModule:
  let ctx = newBackend(getEnv("BYBY_BENCH_BACKEND", "metal"))
  doAssert byby.sessionBegin(ctx.kind.toByby())
  doAssert byby.sessionEnd(ctx.kind.toByby())
  let kv = getEnv("BYBY_BENCH_K", "4096,8192")
  for part in kv.split(','):
    let k = parseInt(part)
    run(ctx, k, if k == 4096: 18560 else: 4096)
