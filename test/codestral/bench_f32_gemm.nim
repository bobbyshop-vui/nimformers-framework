## Model-free bandwidth probe for the FP32 sidecar GEMM that block G uses in
## prefill. Compares the batched kernel (T rows per weight read) against the
## decode GEMV on the *same* buffer, so a slow kernel can be told apart from a
## slow buffer.
##   nim c -d:release --mm:arc --opt:speed -o:bench_f32_gemm bench_f32_gemm.nim
##   BYBY_BENCH_BACKEND=metal ./bench_f32_gemm [K N]
import std/[os, strutils, times, monotimes, strformat, algorithm, math]
import backend, fp4_gemv_vm, fp4_gemv_gpu, gpu_layers
import vendor/bybylang/gpubackend as byby

proc median(xs: var seq[float]): float =
  xs.sort()
  xs[xs.len div 2]

proc main() =
  let k = if paramCount() >= 1: parseInt(paramStr(1)) else: 4096
  let n = if paramCount() >= 2: parseInt(paramStr(2)) else: 12288
  let rows = if paramCount() >= 3: parseInt(paramStr(3)) else: 11
  let be = detectBackend()
  echo &"backend={be} K={k} N={n} rows={rows} weightMiB={k * n * 4 div (1024 * 1024)}"
  var w = newSeq[float32](k * n)
  for i in 0 ..< w.len: w[i] = float32(sin(float(i) * 0.017))
  let hW = byby.sessionUpload(be, w)
  var x = newSeq[float32](rows * k)
  for i in 0 ..< x.len: x[i] = float32(cos(float(i) * 0.011))
  let hX = byby.sessionUpload(be, x)
  var hOut = byby.sessionAllocScratch(be, rows * n)
  let tg = GemvTg
  let knB = vmKernelForBe(buildF32GemmBatchProgram(k, n, tg, rows),
    "bench_f32gemm_" & $k & "x" & $n & "_r" & $rows, be)
  let kn1 = vmKernelForBe(buildF32GemvProgram(k, n, 1.0'f32, false),
    "bench_f32gemv_" & $k & "x" & $n, be)
  proc timeKernel(kn: string; grid, tgSize, tgBytes, outOff: int): float =
    var ts: seq[float]
    for rep in 0 ..< 5:
      let t0 = getMonoTime()
      doAssert byby.sessionBegin(be)
      encodeVmKernel(kn, grid, tgSize, tgBytes, hOut, outOff,
        gDummyFor(be), hW, hX, 0)
      doAssert byby.sessionEnd(be)
      if rep > 1: ts.add float((getMonoTime() - t0).inNanoseconds) / 1e6
    var t = ts
    median(t)
  let mB = timeKernel(knB, n * tg, tg, f32GemmBatchTgBytes(rows, tg), 0)
  let m1 = timeKernel(kn1, n * tg, tg, f32GemvTgBytes(), 0)
  let mib = float(k * n * 4) / (1024.0 * 1024.0)
  echo &"batched r={rows}: {mB:8.2f} ms  -> {mib / (mB / 1000.0) / 1024.0:6.2f} GiB/s"
  echo &"gemv    r=1   : {m1:8.2f} ms  -> {mib / (m1 / 1000.0) / 1024.0:6.2f} GiB/s"
  # Correctness: batched row 0 must equal the single-row GEMV result.
  doAssert byby.sessionBegin(be)
  encodeVmKernel(knB, n * tg, tg, f32GemmBatchTgBytes(rows, tg), hOut, 0,
    gDummyFor(be), hW, hX, 0)
  doAssert byby.sessionEnd(be)
  let yB = byby.sessionRead(be, hOut, rows * n)
  doAssert byby.sessionBegin(be)
  encodeVmKernel(kn1, n * tg, tg, f32GemvTgBytes(), hOut, n,  # write row 1 slot
    gDummyFor(be), hW, hX, 0)
  doAssert byby.sessionEnd(be)
  let y1 = byby.sessionRead(be, hOut, n * 2)
  var worst = 0.0
  var scale = 0.0
  for i in 0 ..< n:
    worst = max(worst, abs(float(yB[i] - y1[n + i])))
    scale = max(scale, abs(float(yB[i])))
  echo &"row0 vs gemv: max abs err {worst:.6g} (scale {scale:.4g})"
  doAssert worst < 1e-3 * max(scale, 1.0), "batched fp32 GEMM khong khop GEMV"
  # 4 block G sidecars each own a fresh 192 MiB buffer that the pipeline reads
  # exactly once per prefill. Time the first dispatch on a brand-new shared
  # buffer to see whether that first touch (page-in / wiring) is the real cost.
  for blk in 0 ..< 3:
    let hFresh = sessionUploadFloatsShared(be, w)
    let t0 = getMonoTime()
    doAssert byby.sessionBegin(be)
    encodeVmKernel(knB, n * tg, tg, f32GemmBatchTgBytes(rows, tg), hOut, 0,
      gDummyFor(be), hFresh, hX, 0)
    doAssert byby.sessionEnd(be)
    let first = float((getMonoTime() - t0).inNanoseconds) / 1e6
    let t1 = getMonoTime()
    doAssert byby.sessionBegin(be)
    encodeVmKernel(knB, n * tg, tg, f32GemmBatchTgBytes(rows, tg), hOut, 0,
      gDummyFor(be), hFresh, hX, 0)
    doAssert byby.sessionEnd(be)
    let again = float((getMonoTime() - t1).inNanoseconds) / 1e6
    echo &"fresh buffer #{blk}: first={first:.2f} ms  second={again:.2f} ms"
  echo "OK"

main()
