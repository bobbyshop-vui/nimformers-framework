## Model-free sweep of the FP4-KM 8-row GEMM over threadgroup widths, to see
## whether the autotuned pick is really the best and how far the kernel is from
## the machine's FMA peak. Kernel = shared micro-VM bytecode, so the winner
## applies to Metal and Vulkan alike.
##   nim c -d:release --mm:arc --opt:speed -o:bench_fp4_gemm_tg bench_fp4_gemm_tg.nim
##   BYBY_BENCH_BACKEND=metal ./bench_fp4_gemm_tg 4096 18560
import std/[os, strutils, math, times, monotimes, strformat, algorithm, tables]
import backend, quant, gpu_layers, fp4_gemv_gpu, fp4_gemv_vm
import vendor/bybylang/gpubackend as byby

proc median(xs: var seq[float]): float =
  xs.sort()
  xs[xs.len div 2]

proc main() =
  let k = if paramCount() >= 1: parseInt(paramStr(1)) else: 4096
  let n = if paramCount() >= 2: parseInt(paramStr(2)) else: 18560
  let ctx = newBackend(getEnv("BYBY_BENCH_BACKEND", "metal"))
  let be = ctx.kind.toByby()
  var qt = QuantTensor(kind: qkFp4KM, shape: @[n, k], groupSize: 32, scale: @[0.01'f32])
  qt.data = newSeq[uint8](n * (k div 2) + n * (k div 32))
  var rng = 123'u32
  for i in 0 ..< n * (k div 2):
    rng = rng * 1664525'u32 + 1013904223'u32
    qt.data[i] = uint8(rng shr 16)
  for i in n * (k div 2) ..< qt.data.len: qt.data[i] = 56'u8
  let lut = buildFp4KmLutQ(qt)
  let name = "tgsweep_" & $k & "_" & $n
  uploadFp4WeightOnce(ctx, name, qt.data, lut, n, k, "")
  let c = gFp4CacheGpu[name]
  var x = newSeq[float32](16 * k)
  for i in 0 ..< x.len: x[i] = float32(sin(float(i) * 0.13))
  let hx = byby.sessionUpload(be, x)
  let hy = byby.sessionAllocScratch(be, 16 * n)
  let macs = float(k) * float(n) * 8.0
  echo &"backend={be} K={k} N={n} weights={k * n div 8 div (1024 * 1024)}MiB rows=8"
  for tg in [32, 64, 128, 256]:
    let kn = vmKernelForBe(buildFp4KmGemmTunedProgram(k, n, tg, 8),
      "tgsweep_m8_" & $k & "x" & $n & "_tg" & $tg, be)
    var ts: seq[float]
    for rep in 0 ..< 5:
      let t0 = getMonoTime()
      doAssert byby.sessionBegin(be)
      encodeVmKernel(kn, n * tg, tg, (Fp4KmLutLen + 8 * tg) * 4, hy, 0,
        c.wBuf, c.lutBuf, hx, 0)
      doAssert byby.sessionEnd(be)
      if rep > 0: ts.add float((getMonoTime() - t0).inNanoseconds) / 1e6
    var t = ts
    let m = median(t)
    echo &"  tg={tg:<4} {m:7.2f} ms  {macs / (m * 1e6):6.1f} GMAC/s"
  for unroll in [2, 4]:
    let kn = vmKernelForBe(buildFp4KmGemmUnrollProgram(k, n, 32, unroll, 8),
      "tgsweep_m8u" & $unroll & "_" & $k & "x" & $n, be)
    var ts: seq[float]
    for rep in 0 ..< 5:
      let t0 = getMonoTime()
      doAssert byby.sessionBegin(be)
      encodeVmKernel(kn, n * 32, 32, (Fp4KmLutLen + 8 * 32) * 4, hy, 0,
        c.wBuf, c.lutBuf, hx, 0)
      doAssert byby.sessionEnd(be)
      if rep > 0: ts.add float((getMonoTime() - t0).inNanoseconds) / 1e6
    var t = ts
    let m = median(t)
    echo &"  u{unroll}-tg=32  {m:7.2f} ms  {macs / (m * 1e6):6.1f} GMAC/s"
    let kn0 = vmKernelForBe(buildFp4KmGemmTunedProgram(k, n, 32, 8),
      "tgsweep_ref8_" & $k & "x" & $n, be)
    doAssert byby.sessionBegin(be)
    encodeVmKernel(kn0, n * 32, 32, (Fp4KmLutLen + 8 * 32) * 4, hy, 0,
      c.wBuf, c.lutBuf, hx, 0)
    doAssert byby.sessionEnd(be)
    let refOut = byby.sessionRead(be, hy, 8 * n)
    doAssert byby.sessionBegin(be)
    encodeVmKernel(kn, n * 32, 32, (Fp4KmLutLen + 8 * 32) * 4, hy, 0,
      c.wBuf, c.lutBuf, hx, 0)
    doAssert byby.sessionEnd(be)
    let got = byby.sessionRead(be, hy, 8 * n)
    var diff = 0
    for i in 0 ..< refOut.len:
      if refOut[i] != got[i]: inc diff
    echo &"  u{unroll} vs tuned: {diff}/{refOut.len} values differ"
    doAssert diff * 20 <= refOut.len, "unroll khong khop (FMA order sai)"

  for rows in [4, 6]:
    let kn = vmKernelForBe(buildFp4KmGemmVecProgram(k, n, 32, rows),
      "tgsweep_vec_r" & $rows & "_" & $k & "x" & $n, be)
    var ts: seq[float]
    let mmul = macs * float(rows) / 8.0
    for rep in 0 ..< 5:
      let t0 = getMonoTime()
      doAssert byby.sessionBegin(be)
      encodeVmKernel(kn, n * 32, 32, f32GemmV4TgBytes(rows, 32), hy, 0,
        c.wBuf, c.lutBuf, hx, 0)
      doAssert byby.sessionEnd(be)
      if rep > 0: ts.add float((getMonoTime() - t0).inNanoseconds) / 1e6
    var t = ts
    let m = median(t)
    echo &"  vec rows={rows}  {m:7.2f} ms  {mmul / (m * 1e6):6.1f} GMAC/s"
    let kn0 = vmKernelForBe(buildFp4KmGemmTunedProgram(k, n, 32, 8),
      "tgsweep_ref8c_" & $k & "x" & $n, be)
    doAssert byby.sessionBegin(be)
    encodeVmKernel(kn0, n * 32, 32, (Fp4KmLutLen + 8 * 32) * 4, hy, 0,
      c.wBuf, c.lutBuf, hx, 0)
    doAssert byby.sessionEnd(be)
    let refOut = byby.sessionRead(be, hy, rows * n)
    doAssert byby.sessionBegin(be)
    encodeVmKernel(kn, n * 32, 32, f32GemmV4TgBytes(rows, 32), hy, 0,
      c.wBuf, c.lutBuf, hx, 0)
    doAssert byby.sessionEnd(be)
    let got = byby.sessionRead(be, hy, rows * n)
    var err, mag = 0.0
    for i in 0 ..< refOut.len:
      err = max(err, abs(float(refOut[i] - got[i])))
      mag = max(mag, abs(float(refOut[i])))
    let rel = err / max(mag, 1e-9)
    echo &"  vec rows={rows} vs tuned: maxAbsErr={err:.3g} rel={rel:.3g}"
    doAssert rel < 1e-5, "vec khong khop"

  block:
    let kn = vmKernelForBe(buildFp4KmGemm2ColProgram(k, n, 32, 8),
      "tgsweep_2col_" & $k & "x" & $n, be)
    var ts: seq[float]
    for rep in 0 ..< 5:
      let t0 = getMonoTime()
      doAssert byby.sessionBegin(be)
      encodeVmKernel(kn, (n div 2) * 32, 32, (Fp4KmLutLen + 16 * 32) * 4, hy, 0,
        c.wBuf, c.lutBuf, hx, 0)
      doAssert byby.sessionEnd(be)
      if rep > 0: ts.add float((getMonoTime() - t0).inNanoseconds) / 1e6
    var t = ts
    let m = median(t)
    echo &"  2col       {m:7.2f} ms  {macs / (m * 1e6):6.1f} GMAC/s"
    let kn0 = vmKernelForBe(buildFp4KmGemmTunedProgram(k, n, 32, 8),
      "tgsweep_ref8d_" & $k & "x" & $n, be)
    doAssert byby.sessionBegin(be)
    encodeVmKernel(kn0, n * 32, 32, (Fp4KmLutLen + 8 * 32) * 4, hy, 0,
      c.wBuf, c.lutBuf, hx, 0)
    doAssert byby.sessionEnd(be)
    let refOut = byby.sessionRead(be, hy, 8 * n)
    doAssert byby.sessionBegin(be)
    encodeVmKernel(kn, (n div 2) * 32, 32, (Fp4KmLutLen + 16 * 32) * 4, hy, 0,
      c.wBuf, c.lutBuf, hx, 0)
    doAssert byby.sessionEnd(be)
    let got = byby.sessionRead(be, hy, 8 * n)
    var err, mag = 0.0
    for i in 0 ..< refOut.len:
      err = max(err, abs(float(refOut[i] - got[i])))
      mag = max(mag, abs(float(refOut[i])))
    let rel = err / max(mag, 1e-9)
    echo &"  2col vs tuned: maxAbsErr={err:.3g} rel={rel:.3g}"
    doAssert rel < 1e-5, "2col khong khop"

  for rows in [4]:
    let kn = vmKernelForBe(buildFp4KmGemmTunedProgram(k, n, 32, rows),
      "tgsweep_m" & $rows & "_" & $k & "x" & $n, be)
    var ts: seq[float]
    for rep in 0 ..< 5:
      let t0 = getMonoTime()
      doAssert byby.sessionBegin(be)
      encodeVmKernel(kn, n * 32, 32, (Fp4KmLutLen + rows * 32) * 4, hy, 0,
        c.wBuf, c.lutBuf, hx, 0)
      doAssert byby.sessionEnd(be)
      if rep > 0: ts.add float((getMonoTime() - t0).inNanoseconds) / 1e6
    var t = ts
    let m = median(t)
    let mc = macs * float(rows) / 8.0
    echo &"  rows={rows:<3} {m:7.2f} ms  {mc / (m * 1e6):6.1f} GMAC/s"

main()
