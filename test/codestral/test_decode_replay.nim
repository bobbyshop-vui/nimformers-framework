## Decode replay (BYBY_DECODE_REPLAY): a recorded decode step resubmitted per token must give the
## same logits as encoding every step, with and without the head, across a state reset.
##
##   nim c -r -d:release --mm:arc --path:. --path:test/codestral --path:test/deepseek-inference \
##     --path:test/spectron --path:kernelfortest -d:codestralInference test/codestral/test_decode_replay.nim
##   BYBY_INFER_BACKEND=vulkan test/codestral/test_decode_replay
##   REPLAY_EMBED=256 test/codestral/test_decode_replay   # also the fused out_proj + residual
import std/[math, os, strutils, strformat, tables]
import backend, quant, lora, nimformer, jamba_hybrid

let ctx = newBackend(getEnv("BYBY_INFER_BACKEND", "metal"))
let pattern = getEnv("REPLAY_PATTERN", "RGR")
# REPLAY_EMBED=256 makes K a multiple of 256, so the FP4 decode GEMVs are autotuned and the fused
# out_proj + residual GEMV runs (the tuner skips smaller shapes).
let embed = parseInt(getEnv("REPLAY_EMBED", "64"))
var m = newNimformerModelJamba(128, embed, 8, 2, pattern, mambaRealDState = 8,
  mambaRealNGroups = 2, mambaRealExpand = 2, initializeWeights = false)
for (name, t) in paramList(m):
  var count = 1
  for d in t[].shape: count *= d
  t[].data = newSeq[float32](count)
  for i in 0 ..< count:
    t[].data[i] =
      if name.endsWith("gamma") or name.endsWith("normWeight") or name.endsWith("dSkip"): 1'f32
      elif name.endsWith("aLog"): -1'f32
      elif name.endsWith("dtBias"): -2'f32
      # Codestral's Mamba projections have no bias (and only bias-free ones take the fused form).
      elif "mambaReal." in name and name.endsWith("Proj.bias"): 0'f32
      else: sin(float32(i) * 0.1'f32 + float32(name.len)) * 0.05'f32
var lo: JambaLora
for name, lin in paramLinears(m):
  if lin[].weight.shape.len != 2: continue
  setQuantWeight(lin[], quantizeFp4KM(lin[].weight.data, lin[].weight.shape))
discard setGpuNormPreset("mamba-ssm")
prepareGpuModel(m, lo, ctx)

const tokens = [3, 10, 17, 5, 99, 42, 7, 64, 1, 120, 33, 8]

proc run(replay: bool): seq[seq[float32]] =
  putEnv("BYBY_DECODE_REPLAY", if replay: "1" else: "0")
  resetGpuGenState(m, ctx)
  for i, t in tokens:
    # Every third step skips the head, like prompt tokens fed through decode.
    let head = i mod 3 != 1
    let r = forwardStepJambaGpu(m, t, ctx, produceLogits = head).data
    if head: result.add r

proc compare(want, got: seq[seq[float32]]; label: string) =
  doAssert want.len == got.len
  var worst = 0.0
  for s in 0 ..< want.len:
    doAssert want[s].len == got[s].len and want[s].len == m.vocab
    var mag = 1e-9
    for v in want[s]: mag = max(mag, abs(v.float))
    for i in 0 ..< want[s].len:
      doAssert classify(got[s][i]) notin {fcNan, fcInf, fcNegInf}
      worst = max(worst, abs(float(got[s][i] - want[s][i])) / mag)
  doAssert worst < 1e-5, &"{label}: replay logits differ, rel={worst}"
  echo &"PASS {label}: {want.len} logit steps match, rel={worst:.3g}"

putEnv("BYBY_DECODE_FUSE_RESIDUAL", "0")
let unfused = run(false)
putEnv("BYBY_DECODE_FUSE_RESIDUAL", "1")
let want = run(false)
doAssert gReplaySteps == 0
# out_proj + residual as one GEMV (when the tuned kernel has that form) must not change the logits.
compare(unfused, want, "fused out_proj + residual")
let first = run(true)
let replayed = gReplaySteps
# One encoded step, then one recording per slot (with and without the head); the rest replay.
doAssert replayed >= tokens.len - 3, &"replay engaged on only {replayed} of {tokens.len} steps"
compare(want, first, "replay after reset")
let second = run(true)
doAssert gReplaySteps - replayed >= tokens.len - 3
compare(want, second, "replay after a second reset")
echo &"PASS decode replay on {ctx.kind}, pattern {pattern}: {gReplaySteps} replayed steps"
