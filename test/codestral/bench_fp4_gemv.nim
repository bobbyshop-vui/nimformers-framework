## bench_fp4_gemv.nim
## Measure the fp4-KM GEMV kernel that dominates Codestral decode.
##
## Why this benchmark: decode issues ~7 dispatches per block x 64 blocks per
## token. The previous `bench_dispatch_overhead.nim` measured a trivial vecOp
## and mis-allocated its scratch buffers, so its numbers were unreliable.
## This benchmark uploads a SYNTHETIC fp4-KM weight with the exact shapes of
## one Codestral block (inProj 4096->8192, outProj 8192->4096) and times the
## real JIT kernel inside one session, splitting host-encode from GPU time.
##
## Usage:
##   nim c -d:release --opt:speed -r bench_fp4_gemv.nim
##   BYBY_BENCH_ITERS=30 nim c -d:release --opt:speed -r bench_fp4_gemv.nim

import std/[os, strutils, strformat, times, tables]
import backend
import vendor/bybylang/gpubackend as byby
import fp4_gemv_gpu
import fp4_gemv_vm
import gpu_layers   # vmKernel, encodeFp4Gemv via fp4_gemv_gpu

proc benchOne(ctx: Backend; name: string; K, N, iters: int) =
  let be = ctx.kind.toByby()
  # Synthetic fp4-KM weight: N rows, K cols, group 32 (matches Codestral).
  var wq = newSeq[uint8]((N * K) div 2)
  for i in 0 ..< wq.len: wq[i] = uint8((i * 37) and 0xFF)
  var lut = newSeq[float32](Fp4KmLutLen)
  for i in 0 ..< Fp4KmLutLen: lut[i] = float32(i and 15) * 0.01'f32

  let pairRows = (K mod 32 == 0) and (N mod 2 == 0)
  let prog = if pairRows: buildFp4KmGemv2RowProgram(K, N)
             else: buildFp4GemvProgram(K, N, true, 32)
  let tag = (if pairRows: "bench_fp4km2_" else: "bench_fp4km_") & $K & "x" & $N & "_g32"
  let kn = gpu_layers.vmKernel(prog, tag)
  uploadFp4WeightOnce(ctx, name, wq, lut, N, K, kn,
                      rowsPerTg = (if pairRows: 2 else: 1))

  var hX = byby.sessionUploadPrivate(be, newSeq[float32](K))
  var hOut = byby.sessionUploadPrivate(be, newSeq[float32](N))

  # Warm the pipeline so the first (JIT-binding) dispatch does not skew timing.
  if byby.sessionBegin(be):
    for _ in 0 ..< 2:
      discard encodeFp4Gemv(gFp4CacheGpu[name], hX, 0, hOut, 0)
    discard byby.sessionEnd(be)

  let t0 = epochTime()
  if not byby.sessionBegin(be):
    echo "sessionBegin failed for " & name
    return
  let t1 = epochTime()
  for _ in 0 ..< iters:
    discard encodeFp4Gemv(gFp4CacheGpu[name], hX, 0, hOut, 0)
  let t2 = epochTime()
  if not byby.sessionEnd(be):
    echo "sessionEnd failed for " & name
    return
  let t3 = epochTime()

  let encUs = (t2 - t1) * 1.0e6 / float(iters)
  let gpuUs = (t3 - t2) * 1.0e6 / float(iters)
  let weightMB = float((N * K) div 2 + N * (K div 32)) / (1024.0 * 1024.0)
  let bwGBs = weightMB / max(gpuUs / 1.0e6, 1.0e-12)
  let rpt = if pairRows: 2 else: 1
  echo "fp4-KM  K=" & $K & " N=" & $N & " rowsPerTg=" & $rpt &
       " (weight+scales=" & formatFloat(weightMB, ffDecimal, 1) & "MB)"
  echo "   per-dispatch: encode=" & formatFloat(encUs, ffDecimal, 1) &
       "us  gpu(dispatch+wait)=" & formatFloat(gpuUs, ffDecimal, 1) & "us"
  echo "   effective bandwidth during gpu phase = " &
       formatFloat(bwGBs, ffDecimal, 2) & " GB/s"

proc main() =
  let ctx = newBackend("auto")
  # Force lazy device init before any upload/alloc: Metal allocates its queue
  # on the first session, and a scratch alloc before that fails silently.
  if byby.sessionBegin(ctx.kind.toByby()):
    discard byby.sessionEnd(ctx.kind.toByby())
  let iters = parseInt(getEnv("BYBY_BENCH_ITERS", "10"))
  # inProj of one Codestral block: in=4096 -> out=8192
  benchOne(ctx, "bench_inProj", 4096, 8192, iters)
  # outProj of one Codestral block: in=8192 -> out=4096
  benchOne(ctx, "bench_outProj", 8192, 4096, iters)

main()
