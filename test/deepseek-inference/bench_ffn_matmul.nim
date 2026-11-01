## bench_ffn_matmul.nim
## Measures the REAL throughput of sessionMatmulQ4 (the Q4 GEMV kernel, M=1 - the shape
## used in decode), completely isolated from: loading the 4.4GB model, the tokenizer,
## Python/torch warmup and the "first matmul call" (Metal pipeline compile
## + pickMatmulPipeline - see the comment in nim_inference.nim around line ~2349).
##
## Purpose: separate two hypotheses -
##   (a) 40ms/call is the REAL compute cost of the Q4 GEMV kernel on Iris 640
##   (b) 40ms/call is noise (thermal throttling, background load, Nim GC,...)
##      mixed into the DECODE_PROFILE measurement (only layer 0, first 2 tokens - too
##      small a sample, easily hit by spikes)
##
## Build:  nim c -d:release -o:bench_ffn_matmul bench_ffn_matmul.nim
## Run:     ./bench_ffn_matmul
##
## Reading the result: compare the median ms/call HERE with the 40ms DECODE_PROFILE measured.
## If they match (~same order) -> (a) holds, the Q4 GEMV kernel is the bottleneck,
## optimise it in bybylang (tiling/threadgroup/vectorised int4 dequant).
## If it is MUCH lower here (e.g. <10ms) -> (b) holds, the resident path/kernel
## is not the problem, look for system noise (Activity Monitor GPU History,
## `pmset -g therm`, close background apps, plug in instead of running on battery).

import std/[times, random, algorithm, strformat, os]
import backend

randomize(1234)

proc packedInt4Row(inF, groupSize: int): seq[uint8] =
  ## nGroupsPerRow*ceil(groupSize/2) bytes/row - the real layout is
  ## ceil(inF/2) bytes/row (2 int4 values packed per byte), as
  ## quantize_int4_asymmetric_per_group in export_to_nimq.py writes it.
  result = newSeq[uint8]((inF + 1) div 2)
  for i in 0 ..< result.len:
    result[i] = uint8(rand(255))

proc buildFakeQ4Weight(inF, outF, groupSize: int):
    tuple[wq: seq[uint8], scales, zeros: seq[float32], nGroupsPerRow: int] =
  let nGroupsPerRow = (inF + groupSize - 1) div groupSize
  var wq = newSeq[uint8](outF * ((inF + 1) div 2))
  for i in 0 ..< wq.len: wq[i] = uint8(rand(255))
  var scales = newSeq[float32](outF * nGroupsPerRow)
  var zeros = newSeq[float32](outF * nGroupsPerRow)
  for i in 0 ..< scales.len:
    scales[i] = 0.01'f32 + rand(0.02'f32)
    zeros[i] = rand(15.0'f32)
  result = (wq, scales, zeros, nGroupsPerRow)

proc benchOne(ctx: Backend, label: string, inF, outF, groupSize, nIter: int) =
  echo &"--- {label}: inF={inF} outF={outF} groupSize={groupSize} (M=1, exactly the decode shape) ---"
  let (wq, scales, zeros, nGroupsPerRow) = buildFakeQ4Weight(inF, outF, groupSize)
  let wHandle = beResidentUploadQ4(ctx, wq, scales, zeros, inF, outF, groupSize, nGroupsPerRow)
  var w: ResidentWeightHandle = wHandle

  var x = newSeq[float32](inF)
  for i in 0 ..< inF: x[i] = rand(1.0'f32) - 0.5'f32

  # Warmup: lets Metal compile the pipeline + pickMatmulPipeline finish BEFORE
  # timing - the same reason generate() runs one "priming" matmul before measuring
  # (see khoi_dong_gpu_pipeline in nim_inference.nim).
  for i in 0 ..< 5:
    discard beResidentLinearQ4(ctx, x, w, 1)

  var times: seq[float] = @[]
  for i in 0 ..< nIter:
    let t0 = epochTime()
    let y = beResidentLinearQ4(ctx, x, w, 1)
    let t1 = epochTime()
    doAssert y.len == outF, "matmul returned the wrong size - kernel error, the benchmark result is not trustworthy"
    times.add((t1 - t0) * 1000.0)

  times.sort()
  let median = times[times.len div 2]
  let p10 = times[times.len div 10]
  let p90 = times[(times.len * 9) div 10]
  var total = 0.0
  for t in times: total += t
  let avg = total / times.len.float
  echo &"  median={median:.2f}ms  avg={avg:.2f}ms  p10={p10:.2f}ms  p90={p90:.2f}ms  " &
       &"(min={times[0]:.2f}ms max={times[^1]:.2f}ms, n={nIter})"
  beResidentFreeQ4(w)

when isMainModule:
  let backendName = if paramCount() > 0: paramStr(1) else: "metal"
  setForbidCpuFallback(true)
  echo &"Backend: {backendName}"
  let ctx = newBackend(backendName)

  # Prime the shared pipeline once before everything (exactly like the real generate()) so
  # the "first compile" cost is not counted in any bench.
  discard beMatmul(ctx, @[0'f32, 0, 0, 0], 2, 2, @[0'f32, 0, 0, 0], 2, 2)

  const groupSize = 128
  const nIter = 200

  # the 3 real FFN shapes of deepseek-coder-6.7B (hidden=4096, intermediate=11008)
  benchOne(ctx, "FFN gate/up proj (4096 -> 11008)", 4096, 11008, groupSize, nIter)
  benchOne(ctx, "FFN down proj (11008 -> 4096)", 11008, 4096, groupSize, nIter)
  # attention oProj/qProj shape (hidden -> hidden) to compare with the
  # "Attention-loop + oProj" figures of DECODE_PROFILE
  benchOne(ctx, "Attn q/k/v/o proj (4096 -> 4096)", 4096, 4096, groupSize, nIter)

  echo ""
  echo "Compare: one real FFN layer = 2x(gate/up, same shape) + 1x(down)."
  echo "Tong uoc tinh 1 layer FFN = 2*median(gate/up) + median(down)."
  echo "Times 32 layers gives the estimated FFN cost of one token - compare with"
  echo "129.694s / 99 tokens = ~1310ms/token shown by the real log."