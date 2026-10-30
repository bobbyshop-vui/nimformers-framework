## Model-free timing of the FP4 LoRA merge kernel on one resident tensor.
##   nim c -d:release --mm:arc --opt:speed -o:bench_merge_gpu bench_merge_gpu.nim
##   BYBY_BENCH_BACKEND=metal BYBY_MERGE_FAST=1 ./bench_merge_gpu
## Repeats the same deterministic merge on a packed buffer so the median is
## comparable between the fast table magnitude and BYBY_MERGE_FAST=0.
import std/[os, strutils, math, times, monotimes, strformat, algorithm, tables]
import backend, quant, lora, fp4_gemv_gpu, merge_gpu
import vendor/bybylang/gpubackend as byby

proc median(xs: var seq[float]): float =
  xs.sort()
  xs[xs.len div 2]

when isMainModule:
  let ctx = newBackend(getEnv("BYBY_BENCH_BACKEND", "metal"))
  let be = ctx.kind.toByby()
  let rows = parseInt(getEnv("BYBY_BENCH_ROWS", "4096"))
  let cols = parseInt(getEnv("BYBY_BENCH_COLS", "18560"))
  let reps = parseInt(getEnv("BYBY_BENCH_REPS", "5"))
  var qt = QuantTensor(kind: qkFp4KM, shape: @[rows, cols], groupSize: 32, scale: @[0.01'f32])
  let elements = rows * cols
  qt.data = newSeq[uint8](elements div 2 + elements div 32)
  var rng = 1234'u32
  for i in 0 ..< elements div 2:
    rng = rng * 1664525'u32 + 1013904223'u32
    qt.data[i] = uint8(rng shr 16)
  for i in elements div 2 ..< qt.data.len: qt.data[i] = 56'u8
  var ad = newLoraAdapter(cols, rows, 4, 2.0'f32)
  for i in 0 ..< ad.A.data.len: ad.A.data[i] = cos(float32(i) * 0.3'f32) * 0.03'f32
  for i in 0 ..< ad.B.data.len: ad.B.data[i] = sin(float32(i) * 0.2'f32) * 0.02'f32
  var cache = Fp4LinCacheGpu(rows: rows, cols: cols,
    wBuf: sessionUploadBytesShared(be, qt.bytes),
    lutBuf: byby.sessionUploadPrivate(be, buildFp4KmLutQ(qt)))
  var samples: seq[float]
  for rep in 0 ..< reps:
    let t0 = getMonoTime()
    discard mergeFp4ResidentGpu(ctx, cache, ad, 64 * 1024 * 1024)
    let ms = float((getMonoTime() - t0).inNanoseconds) / 1e6
    if rep > 0: samples.add ms
    echo &"  rep {rep}: {ms:.1f}ms"
  echo &"{rows}x{cols} merge ({elements} weights): median={median(samples):.1f}ms"
  sessionFree(be, cache.wBuf)
  sessionFree(be, cache.lutBuf)
