## Standalone correctness/performance comparison:
##   reference = RMSNorm -> forwardMamba2Real -> residual
##   block-kernel = forwardRBlockKernel (same weights and input)
##
## Run with:
##   nim c -d:release --hints:off -o:/tmp/bench_mamba2real_block_kernel bench_mamba2real_block_kernel.nim
##   /tmp/bench_mamba2real_block_kernel
## Optional environment overrides: BYBY_BENCH_T, BYBY_BENCH_ITERS,
## BYBY_BENCH_WARMUP, BYBY_BENCH_HIDDEN, BYBY_BENCH_HEADS,
## BYBY_BENCH_DSTATE, BYBY_BENCH_GROUPS, BYBY_BENCH_EXPAND.

import std/[algorithm, math, os, random, strformat, strutils, times]
import backend
import nimformer
import mamba2_real
import hybrid_block_kernel
import mamba2_ssm_gpu

proc envInt(name: string; defaultValue: int): int =
  let raw = getEnv(name, "")
  if raw.len == 0: return defaultValue
  try:
    result = parseInt(raw)
  except ValueError:
    stderr.writeLine "[bench] Invalid " & name & "='" & raw & "'; using " & $defaultValue
    result = defaultValue

proc referenceRBlock(mamba: Mamba2Real; x, gamma: Tensor; eps: float32;
                     ctx: Backend): Tensor =
  ## The pre-block-kernel reference composition, preserving the trained R math.
  let normedData = rmsNormForwardGpu(ctx, x.data, gamma.data,
    x.data.len div x.shape[^1], x.shape[^1], eps)
  let xNorm = Tensor(shape: x.shape, data: normedData)
  let mixer = forwardMamba2Real(mamba, xNorm, ctx).y
  result = addT(ctx, x, mixer)

proc maxAbsDiff(a, b: Tensor): float32 =
  doAssert a.shape == b.shape and a.data.len == b.data.len,
    "benchmark comparison shape mismatch"
  for i in 0 ..< a.data.len:
    result = max(result, abs(a.data[i] - b.data[i]))

proc maxRelativeDiff(a, b: Tensor): float32 =
  doAssert a.shape == b.shape and a.data.len == b.data.len,
    "benchmark comparison shape mismatch"
  for i in 0 ..< a.data.len:
    let denom = max(1e-6'f32, max(abs(a.data[i]), abs(b.data[i])))
    result = max(result, abs(a.data[i] - b.data[i]) / denom)

proc assertFinite(label: string; x: Tensor) =
  for i, v in x.data:
    if classify(v) in {fcNan, fcInf, fcNegInf}:
      raise newException(ValueError, label & " contains non-finite value at " & $i)

proc meanSeconds(samples: seq[float]): float =
  for s in samples: result += s
  result /= float(max(1, samples.len))

proc medianSeconds(samples: seq[float]): float =
  var sorted = samples
  sorted.sort()
  if sorted.len == 0: return 0
  if sorted.len mod 2 == 1:
    result = sorted[sorted.len div 2]
  else:
    result = (sorted[sorted.len div 2 - 1] + sorted[sorted.len div 2]) / 2

when isMainModule:
  let hidden = max(16, envInt("BYBY_BENCH_HIDDEN", 128))
  let heads = max(1, envInt("BYBY_BENCH_HEADS", 8))
  let dState = max(1, envInt("BYBY_BENCH_DSTATE", 16))
  let groups = max(1, envInt("BYBY_BENCH_GROUPS", 2))
  let expand = max(1, envInt("BYBY_BENCH_EXPAND", 2))
  let steps = max(1, envInt("BYBY_BENCH_T", 8))
  let iterations = max(1, envInt("BYBY_BENCH_ITERS", 30))
  let warmups = max(1, envInt("BYBY_BENCH_WARMUP", 5))
  let convKernel = 4
  let intermediate = hidden * expand
  if intermediate mod heads != 0 or heads mod groups != 0:
    raise newException(ValueError,
      "Require (hidden*expand) divisible by heads and heads divisible by groups")

  randomize(20261009)
  let ctx = newBackend("auto")
  let mamba = newMamba2Real(hidden, intermediate, heads, dState, groups, convKernel)
  var gamma = newTensor(@[hidden], 1'f32)
  let eps = 1e-5'f32
  var x = newTensor(@[1, steps, hidden])
  for i in 0 ..< x.data.len:
    x.data[i] = float32(sin(float64(i + 1) * 0.017) + 0.25 * cos(float64(i + 3) * 0.031))

  let xNorm = Tensor(shape: x.shape, data: rmsNormForwardGpu(ctx, x.data,
    gamma.data, steps, hidden, eps))
  let referenceMixer = forwardMamba2Real(mamba, xNorm, ctx).y
  let reference = addT(ctx, x, referenceMixer)
  let blockRun = forwardRBlockKernel(mamba, x, gamma, eps, ctx)
  let blockOutput = blockRun.y
  let lay = blockRun.cache.mamba.lay
  let normedOut = Tensor(shape: @[1, steps, intermediate],
    data: blockRun.cache.mamba.buf[lay.normedOut ..< lay.normedOut + steps * intermediate])
  let blockMixer = mamba.outProj.forward(normedOut, ctx)
  assertFinite("reference Mamba2Real", referenceMixer)
  assertFinite("block-kernel Mamba2Real", blockMixer)
  assertFinite("reference R block", reference)
  assertFinite("block-kernel R block", blockOutput)

  let mixerAbsErr = maxAbsDiff(referenceMixer, blockMixer)
  let mixerRelErr = maxRelativeDiff(referenceMixer, blockMixer)
  let blockAbsErr = maxAbsDiff(reference, blockOutput)
  let blockRelErr = maxRelativeDiff(reference, blockOutput)
  echo &"[correctness] backend={ctx.kind} B=1 T={steps} hidden={hidden} " &
       &"intermediate={intermediate} heads={heads} dState={dState} groups={groups}"
  echo &"[correctness] Mamba2Real mixer: maxAbs={mixerAbsErr:.8g} maxRel={mixerRelErr:.8g}"
  echo &"[correctness] full R block:    maxAbs={blockAbsErr:.8g} maxRel={blockRelErr:.8g}"
  if mixerAbsErr > 2e-4'f32 or blockAbsErr > 2e-4'f32:
    raise newException(AssertionDefect,
      "Mega block-kernel differs from the reference beyond 2e-4 absolute error")

  # Warm both routes so first-use JIT/library compilation is not timed.
  for _ in 0 ..< warmups:
    discard referenceRBlock(mamba, x, gamma, eps, ctx)
    discard forwardRBlockKernel(mamba, x, gamma, eps, ctx)

  var referenceTimes = newSeqOfCap[float](iterations)
  var blockTimes = newSeqOfCap[float](iterations)
  var checksum = 0'f32
  for i in 0 ..< iterations:
    if i mod 2 == 0:
      let t0 = epochTime()
      let yRef = referenceRBlock(mamba, x, gamma, eps, ctx)
      referenceTimes.add(epochTime() - t0)
      checksum += yRef.data[i mod yRef.data.len]
      let t1 = epochTime()
      let yBlock = forwardRBlockKernel(mamba, x, gamma, eps, ctx).y
      blockTimes.add(epochTime() - t1)
      checksum += yBlock.data[i mod yBlock.data.len]
    else:
      let t0 = epochTime()
      let yBlock = forwardRBlockKernel(mamba, x, gamma, eps, ctx).y
      blockTimes.add(epochTime() - t0)
      checksum += yBlock.data[i mod yBlock.data.len]
      let t1 = epochTime()
      let yRef = referenceRBlock(mamba, x, gamma, eps, ctx)
      referenceTimes.add(epochTime() - t1)
      checksum += yRef.data[i mod yRef.data.len]

  let refMean = meanSeconds(referenceTimes)
  let blockMean = meanSeconds(blockTimes)
  let refMedian = medianSeconds(referenceTimes)
  let blockMedian = medianSeconds(blockTimes)
  let tokenCount = float(steps)
  echo &"[speed] iterations={iterations} warmups={warmups} (first-use JIT excluded)"
  echo &"[speed] reference Mamba2Real composition: mean={refMean * 1000:.3f} ms " &
       &"median={refMedian * 1000:.3f} ms, median throughput={tokenCount / refMedian:.2f} token/s"
  echo &"[speed] R mega block-kernel:             mean={blockMean * 1000:.3f} ms " &
       &"median={blockMedian * 1000:.3f} ms, median throughput={tokenCount / blockMedian:.2f} token/s"
  echo &"[speed] block/reference latency ratio: mean={blockMean / refMean:.3f}x " &
       &"median={blockMedian / refMedian:.3f}x; checksum={checksum:.6g}"
