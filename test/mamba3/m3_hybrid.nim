## Mamba-3 SISO 187M pipeline forked from test/codestral.
## R = Mamba-3 SISO + gated MLP; P/G add the existing Linear++ sidecar.
## Projection, LoRA, optimizer and checkpoint helpers reuse Codestral formats.
## See README.md for supported paths and validation limits.

import std/[math, random, strformat, tables, json, os, strutils, sets]
import customfloat
import backend
import rawbin            # raw .bin checkpoints - used ONLY by the fine-tune pipeline,
                          # .nimq (quant.saveQuantStateDict/loadQuantStateDict) is NOT used
                          # here - .nimq is only for nim_inference.nim/test_nimformer.nim
                            # NO intermediate file is written to disk
import nimformer
import mamba2_ssm_gpu       # RMSNorm fwd/bwd with GPU micro-kernels
import mamba3_real         # the REAL Mamba3Real (unlike Mamba2SSD) - conv1d/A_log/dt/D/RMSNormGated
import test_nimformer   # Mamba2SSD, LinearPlusAttention, TransformerBlockSpectron,
                         # Model/training dependency only; hybrid inference uses the separate module below.
import m3_block_kernel # end-to-end logical GPU kernel entrypoints for R/P blocks
import linear_plus_infer
                         # FeedForward, LayerNorm, Embedding, Linear, forwardMamba2,
                         # forwardLinearPlus, backwardMamba2, backwardLinearPlus,
                         # newLinear/newLayerNorm/newFeedForward/newEmbedding,
                         # ApfAdamState, apfAdamStep, softmaxCrossEntropy

type
  BlockKind* = enum
    bkMamba = "M"
    bkAttn = "A"
    bkSpectron = "S"
    bkMambaReal = "R"    # REAL Mamba2 (mamba3_real.nim) - used when loading the Codestral-7B pretrain
    bkMambaRealPlusAttn = "P" # Mamba3Real + a Linear++ sidecar running in parallel
    bkMambaRealGated = "G" # Mamba3Real + Linear++ sidecar, g=gate Mamba

  RmsNorm* = object
    ## RMSNorm exactly as in the Mamba2/Codestral pretrain (HF Mamba2RMSNorm):
    ##     y = x / sqrt(mean(x^2) + eps) * gamma     (NO mean subtraction, NO beta)
    ## This is the base norm: backbone.layers[i].norm AND backbone.norm_f in
    ## HF Mamba2Model are both RMSNorm. The old training path used LayerNorm (with mean
    ## subtraction + beta) for 'R'/'P' blocks and did NOT apply norm_f before lm_head
    ## (preset "ln-nof-g1") - far off the pretrain distribution, LoRA r=4 had to absorb
    ## that whole shift. New preset ("rms-fin"): norms IDENTICAL to the base, the only
    ## difference from the base is the attention sidecar in 'P' blocks.
    gamma*: Tensor      # [dim], initialised = 1 (identity), pretrain weights loaded later
    eps*: float32

  JambaBlock* = object
    kind*: BlockKind
    ln1*, ln2*: LayerNorm
    rms1*: RmsNorm     # NEW (preset rms-fin): block input norm for 'R'/'P', REPLACES
                       # ln1 on the forward/backward path of the 2 pretrain block kinds.
                       # ln1 still exists for the legacy 'M'/'A'/'S' blocks (not
                       # pretrained, trained from scratch, so LayerNorm is fine).
    ff*: FeedForward
    # only the fields for the kind are actually used, the others keep their defaults
    mamba*: Mamba2SSD
    mambaReal*: Mamba3Real
    attn*: LinearPlusAttention
    # attnLn1 REMOVED - block P shares ln1 with Mamba (like the original Jamba)
    fuseGate*: Linear        # only used when kind == bkSpectron

  NimformerModelJamba* = object
    vocab*, embedDim*, nHeads*, ffMult*: int
    pattern*: string
    embed*: Embedding
    blocks*: seq[JambaBlock]
    outProj*: Linear
    normF*: LayerNorm    # LEGACY - NO LONGER USED on the forward/backward path.
                         # Field kept so external modules that may reference it
                         # (old loaders) do not break. The real final norm is rmsF below.
    rmsF*: RmsNorm       # NEW (preset rms-fin): equivalent to backbone.norm_f in
                         # HF Mamba2Model - RMSNorm applied ONCE after all
                         # blocks, BEFORE lm_head. normF used to exist but
                         # NO forward called it -> the hidden state entered lm_head at the wrong scale.

const
  G_MAX_SIDE_RMS_RATIO* = 0.5'f32
  G_MAX_OUTPUT_RMS_RATIO* = 8'f32
  G_MAX_MAMBA_RMS_RATIO* = 1.5'f32
    ## rms(mambaY)/rms(x) — the Mamba branch BEFORE gate/residual. Linear++ is already
    ## capped at ≤0.5×; same scale as the R mixer 1.5×; 2.0 is too wide.
  G_MIN_GATE_MEAN* = 0.75'f32
    ## The mean gate (sigmoid) over all elements of a G block must be
    ## >= this threshold. The gate starts at ~0.982 (bias=4); if the mean drops
    ## below 0.75 the Linear++ branch is dominating the
    ## residual - the guard raises before checkpointing.

proc newRmsNorm*(dim: int, eps: float32 = 1e-5'f32): RmsNorm =

  ## eps = 1e-5: the HF Mamba2Config layer_norm_eps default (Codestral).
  result.gamma = newTensor(@[dim])
  for i in 0 ..< dim: result.gamma.data[i] = 1'f32   # identity until weights are loaded
  result.eps = eps

proc forward*(n: RmsNorm, x: Tensor, ctx: Backend): Tensor =
  ## Norm over the last dim. GPU (micro-kernel) when the backend is Metal/CUDA/Vulkan;
  ## the fp32 CPU loop is only the fallback for the cpu backend.
  let C = x.shape[^1]
  let rows = x.data.len div C
  result = newTensor(x.shape)
  if isGpuCtx(ctx):
    # GPU micro-kernel (train_gpu_vm.buildRmsFwdProgram), CPU is only the cpu-backend fallback.
    result.data = rmsNormForwardGpu(ctx, x.data, n.gamma.data, rows, C, n.eps)
    return
  for r in 0 ..< rows:
    let base = r * C
    var ms = 0'f32
    for c in 0 ..< C:
      let v = x.data[base + c]
      ms += v * v
    ms /= float32(C)
    let invRms = 1'f32 / sqrt(ms + n.eps)
    for c in 0 ..< C:
      result.data[base + c] = x.data[base + c] * invRms * n.gamma.data[c]

proc backward*(n: RmsNorm, x, dOut: Tensor, ctx: Backend): tuple[dX, dGamma: Tensor] =
  ## Correct RMSNorm backward (linear in dOut - important for 'P' blocks
  ## whose 2 parallel branches share 1 norm, see backwardBlock):
  ##   y = x * r * g,  r = (mean(x^2)+eps)^(-1/2)
  ##   dX_j      = r*g_j*dy_j - (r^3/C) * x_j * sum_c(dy_c*g_c*x_c)
  ##   dGamma_c += sum_rows dy_c * x_c * r
  let C = x.shape[^1]
  let rows = x.data.len div C
  result.dX = newTensor(x.shape)
  result.dGamma = newTensor(n.gamma.shape)   # newTensor is zero-initialised
  if isGpuCtx(ctx):
    let g = rmsNormBackwardGpu(ctx, x.data, n.gamma.data, dOut.data, rows, C, n.eps)
    result.dX.data = g.dX
    result.dGamma.data = g.dGamma
    return
  for r in 0 ..< rows:
    let base = r * C
    var ms = 0'f32
    for c in 0 ..< C:
      let v = x.data[base + c]
      ms += v * v
    ms /= float32(C)
    let invRms = 1'f32 / sqrt(ms + n.eps)
    var dot = 0'f32
    for c in 0 ..< C:
      let xc = x.data[base + c]
      let dyc = dOut.data[base + c]
      result.dGamma.data[c] += dyc * xc * invRms
      dot += dyc * n.gamma.data[c] * xc
    let coeff = dot * invRms * invRms * invRms / float32(C)
    for c in 0 ..< C:
      result.dX.data[base + c] =
        invRms * n.gamma.data[c] * dOut.data[base + c] - x.data[base + c] * coeff

proc concatLastDim(a, b: Tensor): Tensor =
  ## Local version (test_nimformer.nim does not export the original concatLastDim/splitLastDim
  ## - the logic is copied exactly to avoid editing the original file).
  let B = a.shape[0]; let T = a.shape[1]; let C = a.shape[2]
  result = newTensor(@[B, T, 2 * C])
  for i in 0 ..< B * T:
    for c in 0 ..< C:
      result.data[i * 2 * C + c] = a.data[i * C + c]
      result.data[i * 2 * C + C + c] = b.data[i * C + c]

proc splitLastDim(x: Tensor, C: int): tuple[a, b: Tensor] =
  let B = x.shape[0]; let T = x.shape[1]
  result.a = newTensor(@[B, T, C])
  result.b = newTensor(@[B, T, C])
  for i in 0 ..< B * T:
    for c in 0 ..< C:
      result.a.data[i * C + c] = x.data[i * 2 * C + c]
      result.b.data[i * C + c] = x.data[i * 2 * C + C + c]

proc charToKind(c: char): BlockKind =
  case c
  of 'M', 'm': bkMamba
  of 'A', 'a': bkAttn
  of 'S', 's': bkSpectron
  of 'R', 'r': bkMambaReal
  of 'P', 'p': bkMambaRealPlusAttn
  of 'G', 'g': bkMambaRealGated
  else: raise newException(ValueError, &"Invalid pattern character: '{c}' (only M/A/S/R/P/G accepted)")

proc newJambaBlock(kind: BlockKind, embedDim, nHeads, ffMult: int, useSession: bool,
                    mambaRealDState: int = 128, mambaRealNGroups: int = 1,
                    mambaRealConvKernel: int = 0, mambaRealExpand: int = 2,
                    quantResident: bool = false; initializeWeights = true): JambaBlock =
  result.kind = kind
  result.ln1 = newLayerNorm(embedDim)
  result.ln2 = newLayerNorm(embedDim)
  result.rms1 = newRmsNorm(embedDim)   # used for 'R'/'P' (see forwardBlock)
  case kind
  of bkMamba:
    result.ff = newFeedForward(embedDim, ffMult)
    result.mamba = newMamba2SSD(embedDim, nHeads, useSession)
  of bkAttn:
    result.ff = newFeedForward(embedDim, ffMult)
    result.attn = newLinearPlusAttention(embedDim, nHeads, useSession)
  of bkSpectron:
    result.ff = newFeedForward(embedDim, ffMult)
    result.attn = newLinearPlusAttention(embedDim, nHeads, useSession)
    result.mamba = newMamba2SSD(embedDim, nHeads, useSession)
    result.fuseGate = newLinear(embedDim * 2, embedDim)
  of bkMambaReal:
    # intermediateSize = embedDim * mambaRealExpand (the real Codestral-7B
    # uses expand=2 -> intermediateSize=8192 when hiddenSize=4096, confirmed
    # from the REAL shapes: mixer.norm.weight=[8192], mixer.out_proj.weight=[4096,8192]
    # - the old hard-coded expand=1 was WRONG and would mismatch every shape when loading
    # real weights (m3_loader would report a shape mismatch on every
    # mixer tensor). result.ff/ln2 are STILL created above (fields not removed,
    # so other code assuming the fields always exist does not break) but are NOT
    # USED in the bkMambaReal forward/backward (see forwardBlock/
    # backwardBlock below) - the real HF Mamba2 has NO FFN between
    # layers (only norm->mixer->residual), unlike a regular Transformer block.
    result.mambaReal = newMamba3Real(embedDim, embedDim * mambaRealExpand, nHeads,
                                      mambaRealDState, mambaRealNGroups, mambaRealConvKernel,
                                      initializeWeights = initializeWeights)
    if quantResident:
      # Fix for the ~29GB OOM while BUILDING THE MODEL, before the .nimq is even loaded: inProj/
      # outProj are the 2 largest matrices of each block (~90% of a Mamba3Real
      # block's parameters). Quant-resident mode (see m3_loader.nim
      # + lora.nim mergeInOne/restoreBaseOne) is designed for these 2 tensors to be
      # EMPTY (.data = @[]) at rest, dequantized only temporarily through the
      # LUT for 1 block right before forward/backward and freed right after (see
      # trainStepGraftStreaming) - but newMamba3Real() STILL allocated FULL fp32
      # for both tensors at block creation (randnTensor/newTensor),
      # wasted immediately (quant-resident never reads these values).
      # With 52 'R' blocks (embedDim=4096, intermediateSize=8192) ->
      # ~29GB of USELESS allocation, even if only transient - enough for an OOM kill
      # on RAM-limited machines (e.g. integrated GPUs sharing system
      # RAM). Freed RIGHT HERE, IN THE LOOP (not after newNimformerModelJamba
      # returns) - the GC reclaims each block, so peak RAM
      # while building the model is ~1 block, not 52 blocks at once.
      result.mambaReal.inProj.weight.data = @[]
      result.mambaReal.outProj.weight.data = @[]
      result.mambaReal.mlpFc1.weight.data = @[]
      result.mambaReal.mlpFc2.weight.data = @[]
  of bkMambaRealPlusAttn, bkMambaRealGated:
    result.mambaReal = newMamba3Real(embedDim, embedDim * mambaRealExpand, nHeads,
                                      mambaRealDState, mambaRealNGroups, mambaRealConvKernel,
                                      initializeWeights = initializeWeights)
    let initializeSidecar = initializeWeights or kind == bkMambaRealGated
    result.attn = newLinearPlusAttention(embedDim, nHeads, useSession, initializeSidecar)
    # Fix: ln1 is SHARED BY MAMBA AND ATTENTION (like the original Jamba).
    # attnLn1 REMOVED - both branches see the same normalised input → no scale mismatch when adding the residual.
    # P remains exact Mamba identity at initialization. G adds a near-identity
    # Mamba-first gate below, and must keep the sidecar output projection at zero.
    for i in 0 ..< result.attn.outProj.weight.data.len: result.attn.outProj.weight.data[i] = 0'f32
    for i in 0 ..< result.attn.outProj.bias.data.len: result.attn.outProj.bias.data[i] = 0'f32
    if kind == bkMambaRealGated:
      result.fuseGate = newLinear(embedDim * 2, embedDim, initializeSidecar)
      for i in 0 ..< result.fuseGate.weight.data.len: result.fuseGate.weight.data[i] = 0'f32
      for i in 0 ..< result.fuseGate.bias.data.len: result.fuseGate.bias.data[i] = 4'f32
    if quantResident:
      result.mambaReal.inProj.weight.data = @[]
      result.mambaReal.outProj.weight.data = @[]
      result.mambaReal.mlpFc1.weight.data = @[]
      result.mambaReal.mlpFc2.weight.data = @[]

proc newNimformerModelJamba*(vocab, embedDim, nHeads, ffMult: int, pattern: string,
                              useSession: bool = true, mambaRealDState: int = 128,
                              mambaRealNGroups: int = 1, mambaRealConvKernel: int = 0,
                              mambaRealExpand: int = 2, quantResident: bool = false;
                              initializeWeights = true): NimformerModelJamba =
  assert pattern.len > 0, "empty pattern - need at least 1 layer"
  result.vocab = vocab
  result.embedDim = embedDim
  result.nHeads = nHeads
  result.ffMult = ffMult
  result.pattern = pattern
  result.embed = newEmbedding(vocab, embedDim, initializeWeights)
  result.outProj = newLinear(embedDim, vocab, initializeWeights)
  result.normF = newLayerNorm(embedDim)   # LEGACY - no longer used (see struct)
  result.rmsF = newRmsNorm(embedDim)      # NEW (rms-fin): the REAL norm_f before lm_head
  result.blocks = newSeq[JambaBlock](pattern.len)
  for i, c in pattern:
    result.blocks[i] = newJambaBlock(charToKind(c), embedDim, nHeads, ffMult, useSession,
                                      mambaRealDState, mambaRealNGroups, mambaRealConvKernel,
                                      mambaRealExpand, quantResident, initializeWeights)

# ═══════════════════════════════════════════════════════════════
# Per-block FORWARD / BACKWARD. R/P are handled by the block-kernel entrypoints
# in m3_block_kernel.nim; legacy M/A/S keep the existing path below.
# ═══════════════════════════════════════════════════════════════

proc checkGBlockScale*(blockIndex: int; x, output: Tensor;
                       cache: HybridBlockKernelCache) =
  ## Guard G: Linear++ weighted ≤0.5×, gate mean ≥0.75, AND the Mamba branch
  ## rms(mambaY)/rms(x) ≤ G_MAX_MAMBA_RMS_RATIO. cache.mambaY must be
  ## filled by forwardPBlockKernel (before the residual).
  doAssert x.shape == output.shape and x.data.len == cache.gate.len
  doAssert cache.mambaY.data.len == x.data.len,
    "checkGBlockScale: cache.mambaY missing/wrong size (block kernel must expose mambaY)"
  doAssert cache.linearPlusOut.data.len == x.data.len,
    "checkGBlockScale: cache.linearPlusOut missing/wrong size"
  var inputSq, sideSq, mambaSq, outputSq, gateSum = 0.0
  var inputMax, sideMax, mambaMax, outputMax = 0.0
  for i in 0 ..< x.data.len:
    let gate = cache.gate[i]
    let input = x.data[i]
    let mamba = cache.mambaY.data[i]
    let side = cache.linearPlusOut.data[i]
    let outputValue = output.data[i]
    if gate.classify in {fcNan, fcInf, fcNegInf} or
       input.classify in {fcNan, fcInf, fcNegInf} or
       mamba.classify in {fcNan, fcInf, fcNegInf} or
       side.classify in {fcNan, fcInf, fcNegInf} or
       outputValue.classify in {fcNan, fcInf, fcNegInf}:
      raise newException(CatchableError, "G block produced non-finite branch/gate values")
    if gate < 0'f32 or gate > 1'f32:
      raise newException(CatchableError, "G block sigmoid gate escaped [0,1]")
    inputSq += float64(input) * float64(input)
    mambaSq += float64(mamba) * float64(mamba)
    let weightedSide = (1'f32 - gate) * side
    sideSq += float64(weightedSide) * float64(weightedSide)
    outputSq += float64(outputValue) * float64(outputValue)
    inputMax = max(inputMax, abs(float64(input)))
    mambaMax = max(mambaMax, abs(float64(mamba)))
    sideMax = max(sideMax, abs(float64(weightedSide)))
    outputMax = max(outputMax, abs(float64(outputValue)))
    gateSum += float64(gate)
  let inputRms = sqrt(inputSq / float64(max(1, x.data.len)))
  let mambaRms = sqrt(mambaSq / float64(max(1, x.data.len)))
  let sideRms = sqrt(sideSq / float64(max(1, x.data.len)))
  let outputRms = sqrt(outputSq / float64(max(1, x.data.len)))
  let mambaRmsRatio = mambaRms / max(inputRms, 1e-6)
  let mambaMaxRatio = mambaMax / max(inputMax, 1e-6)
  let sideRmsRatio = sideRms / max(inputRms, 1e-6)
  let outputRmsRatio = outputRms / max(inputRms, 1e-6)
  let sideMaxRatio = sideMax / max(inputMax, 1e-6)
  let outputMaxRatio = outputMax / max(inputMax, 1e-6)
  if mambaRmsRatio > float64(G_MAX_MAMBA_RMS_RATIO):
    raise newException(CatchableError,
      &"G block {blockIndex}: mambaY RMS/input RMS={mambaRmsRatio:.3f} " &
       &"exceeds limit {G_MAX_MAMBA_RMS_RATIO:.2f} (SSM branch poisoning residual)")
  if sideRmsRatio > float64(G_MAX_SIDE_RMS_RATIO):
    raise newException(CatchableError,
      &"G block {blockIndex}: weighted Linear++ RMS/input RMS={sideRmsRatio:.3f} " &
       &"exceeds limit {G_MAX_SIDE_RMS_RATIO:.2f}")
  if sideMaxRatio > 8.0:
    raise newException(CatchableError,
      &"G block {blockIndex}: weighted Linear++ max/input max={sideMaxRatio:.3f} exceeds limit 8.0")
  if outputRmsRatio > float64(G_MAX_OUTPUT_RMS_RATIO) or outputMaxRatio > 16.0:
    raise newException(CatchableError,
      &"G block {blockIndex}: residual output/input RMS={outputRmsRatio:.3f}, " &
       &"max={outputMaxRatio:.3f}; limit is RMS 8.0x / max 16.0x")
  let gateMean = gateSum / float64(max(1, cache.gate.len))
  if gateMean < float64(G_MIN_GATE_MEAN):
    raise newException(CatchableError,
      &"G block {blockIndex}: mean gate {gateMean:.4f} below floor {G_MIN_GATE_MEAN:.2f}; " &
       "Linear++ branch is dominating the residual")
  if getEnv("BYBY_G_DIAGNOSTICS") == "1":
    stderr.writeLine &"[G-check] block={blockIndex} gateMean={gateMean:.4f} " &
      &"mambaRms/input={mambaRmsRatio:.3f} mambaMax/input={mambaMaxRatio:.3f} " &
      &"sideRms/input={sideRmsRatio:.3f} sideMax/input={sideMaxRatio:.3f} " &
      &"outputRms/input={outputRmsRatio:.3f} outputMax/input={outputMaxRatio:.3f}"

proc forwardBlockEx*(blk: JambaBlock, x: Tensor, ctx: Backend;
                     blockIndex = -1):
    tuple[y: Tensor, cache: HybridBlockKernelCache] =
  ## Like forwardBlock but returns the cache (with mambaY before the residual for R/P/G).
  if blk.kind == bkMambaReal:
    return forwardRBlockKernel(blk.mambaReal, x, blk.rms1.gamma, blk.rms1.eps, ctx)
  if blk.kind == bkMambaRealPlusAttn:
    return forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                               blk.rms1.eps, ctx)
  if blk.kind == bkMambaRealGated:
    let fw = forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                 blk.rms1.eps, ctx, fuseGate = unsafeAddr blk.fuseGate)
    checkGBlockScale(blockIndex, x, fw.y, fw.cache)
    return fw
  raise newException(CatchableError,
    "forwardBlockEx: only R/P/G are supported (needs HybridBlockKernelCache.mambaY)")

proc forwardBlock*(blk: JambaBlock, x: Tensor, ctx: Backend;
                   blockIndex = -1): Tensor =

  if blk.kind == bkMambaReal:
    return forwardRBlockKernel(blk.mambaReal, x, blk.rms1.gamma, blk.rms1.eps, ctx).y
  if blk.kind == bkMambaRealPlusAttn:
    return forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                               blk.rms1.eps, ctx).y
  if blk.kind == bkMambaRealGated:
    let fw = forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                 blk.rms1.eps, ctx, fuseGate = unsafeAddr blk.fuseGate)
    checkGBlockScale(blockIndex, x, fw.y, fw.cache)
    return fw.y
  let x1 = blk.ln1.forward(x, ctx)
  var combined: Tensor
  case blk.kind
  of bkMamba:
    combined = forwardMamba2(blk.mamba, x1, ctx)
  of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
    raise newException(CatchableError, "R/P block missed its block-kernel entrypoint")
  of bkAttn:
    combined = forwardLinearPlus(blk.attn, x1, ctx)
  of bkSpectron:
    let attnOut = forwardLinearPlus(blk.attn, x1, ctx)
    let mambaOut = forwardMamba2(blk.mamba, x1, ctx)
    let fuseIn = concatLastDim(attnOut, mambaOut)   # exported (*) from test_nimformer
    let gateLogit = blk.fuseGate.forward(fuseIn, ctx)
    let gate = beSigmoid(ctx, gateLogit.data)
    combined = newTensor(attnOut.shape)
    for i in 0 ..< combined.data.len:
      combined.data[i] = gate[i] * attnOut.data[i] + (1'f32 - gate[i]) * mambaOut.data[i]
  let x2 = addT(ctx, x, combined)
  let x3 = blk.ln2.forward(x2, ctx)
  let ffOut = blk.ff.forward(x3, ctx)
  result = addT(ctx, x2, ffOut)

proc backwardBlock*(blk: JambaBlock, x: Tensor, dOut: Tensor, ctx: Backend,
                    prefix: string, grads: var Table[string, Tensor];
                    blockIndex = -1): Tensor =
  if blk.kind == bkMambaReal:
    let fw = forwardRBlockKernel(blk.mambaReal, x, blk.rms1.gamma, blk.rms1.eps, ctx)
    return backwardRBlockKernel(blk.mambaReal, x, blk.rms1.gamma, dOut, fw.cache,
                                blk.rms1.eps, ctx, prefix, grads)
  if blk.kind == bkMambaRealGated:
    let fw = forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                 blk.rms1.eps, ctx, fuseGate = unsafeAddr blk.fuseGate)
    checkGBlockScale(blockIndex, x, fw.y, fw.cache)
    return backwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                dOut, fw.cache, blk.rms1.eps, ctx, prefix, grads,
                                fuseGate = unsafeAddr blk.fuseGate)
  if blk.kind == bkMambaRealPlusAttn:
    let fw = forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                 blk.rms1.eps, ctx)
    return backwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                dOut, fw.cache, blk.rms1.eps, ctx, prefix, grads)
  ## Returns dX (to propagate to the previous block) and WRITES the gradients
  ## of every parameter of this block into `grads` with key = prefix & ".<name>".
  let x1 = blk.ln1.forward(x, ctx)
  var combined: Tensor
  var attnOut, mambaOut, fuseIn, gateLogit: Tensor
  var gate: seq[float32]
  case blk.kind
  of bkMamba:
    combined = forwardMamba2(blk.mamba, x1, ctx)
  of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
    raise newException(CatchableError, "R/P block missed its block-kernel entrypoint")
  of bkAttn:
    combined = forwardLinearPlus(blk.attn, x1, ctx)
  of bkSpectron:
    attnOut = forwardLinearPlus(blk.attn, x1, ctx)
    mambaOut = forwardMamba2(blk.mamba, x1, ctx)
    fuseIn = concatLastDim(attnOut, mambaOut)
    gateLogit = blk.fuseGate.forward(fuseIn, ctx)
    gate = beSigmoid(ctx, gateLogit.data)
    combined = newTensor(attnOut.shape)
    for i in 0 ..< combined.data.len:
      combined.data[i] = gate[i] * attnOut.data[i] + (1'f32 - gate[i]) * mambaOut.data[i]
  let x2 = addT(ctx, x, combined)

  let x3 = blk.ln2.forward(x2, ctx)
  let dFF = blk.ff.backward(x3, dOut, ctx)
  let dLN2 = blk.ln2.backward(x2, dFF.dX, ctx)
  grads[prefix & ".ln2.gamma"] = dLN2.dGamma
  grads[prefix & ".ln2.beta"] = dLN2.dBeta
  grads[prefix & ".ff.fc1.weight"] = dFF.dFc1W
  grads[prefix & ".ff.fc1.bias"] = dFF.dFc1B
  grads[prefix & ".ff.fc2.weight"] = dFF.dFc2W
  grads[prefix & ".ff.fc2.bias"] = dFF.dFc2B
  var dX2 = dLN2.dX
  for i in 0 ..< dX2.data.len: dX2.data[i] += dOut.data[i]   # residual x2 = x + combined

  var dX1Total: Tensor
  case blk.kind
  of bkMamba:
    let dM = backwardMamba2(blk.mamba, x1, dX2, ctx)
    grads[prefix & ".mamba.inProj.weight"] = dM.dInProjW
    grads[prefix & ".mamba.inProj.bias"] = dM.dInProjB
    grads[prefix & ".mamba.outProj.weight"] = dM.dOutProjW
    grads[prefix & ".mamba.outProj.bias"] = dM.dOutProjB
    dX1Total = dM.dX
  of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
    raise newException(CatchableError, "R/P block missed its block-kernel entrypoint")
  of bkAttn:
    let dA = backwardLinearPlus(blk.attn, x1, dX2, ctx)
    grads[prefix & ".attn.qkvProj.weight"] = dA.dQkvW
    grads[prefix & ".attn.qkvProj.bias"] = dA.dQkvB
    grads[prefix & ".attn.outProj.weight"] = dA.dOutW
    grads[prefix & ".attn.outProj.bias"] = dA.dOutB
    dX1Total = dA.dX
  of bkSpectron:
    var dGate = newSeq[float32](dX2.data.len)
    var dAttnOutPartial = newSeq[float32](dX2.data.len)
    var dMambaOutPartial = newSeq[float32](dX2.data.len)
    for i in 0 ..< dX2.data.len:
      dGate[i] = dX2.data[i] * (attnOut.data[i] - mambaOut.data[i])
      dAttnOutPartial[i] = dX2.data[i] * gate[i]
      dMambaOutPartial[i] = dX2.data[i] * (1'f32 - gate[i])
    var dGateLogit = newTensor(gateLogit.shape)
    for i in 0 ..< dGateLogit.data.len:
      dGateLogit.data[i] = dGate[i] * gate[i] * (1'f32 - gate[i])
    let dFuse = blk.fuseGate.backward(fuseIn, dGateLogit, ctx)
    grads[prefix & ".fuseGate.weight"] = dFuse.dW
    grads[prefix & ".fuseGate.bias"] = dFuse.dB
    let (dAttnOutFuse, dMambaOutFuse) = splitLastDim(dFuse.dX, x.shape[2])
    var dAttnOutTotal = newTensor(attnOut.shape)
    var dMambaOutTotal = newTensor(mambaOut.shape)
    for i in 0 ..< dAttnOutTotal.data.len:
      dAttnOutTotal.data[i] = dAttnOutPartial[i] + dAttnOutFuse.data[i]
      dMambaOutTotal.data[i] = dMambaOutPartial[i] + dMambaOutFuse.data[i]
    let dAttn = backwardLinearPlus(blk.attn, x1, dAttnOutTotal, ctx)
    let dMamba = backwardMamba2(blk.mamba, x1, dMambaOutTotal, ctx)
    grads[prefix & ".attn.qkvProj.weight"] = dAttn.dQkvW
    grads[prefix & ".attn.qkvProj.bias"] = dAttn.dQkvB
    grads[prefix & ".attn.outProj.weight"] = dAttn.dOutW
    grads[prefix & ".attn.outProj.bias"] = dAttn.dOutB
    grads[prefix & ".mamba.inProj.weight"] = dMamba.dInProjW
    grads[prefix & ".mamba.inProj.bias"] = dMamba.dInProjB
    grads[prefix & ".mamba.outProj.weight"] = dMamba.dOutProjW
    grads[prefix & ".mamba.outProj.bias"] = dMamba.dOutProjB
    dX1Total = newTensor(x1.shape)
    for i in 0 ..< dX1Total.data.len:
      dX1Total.data[i] = dAttn.dX.data[i] + dMamba.dX.data[i]

  let dLN1 = blk.ln1.backward(x, dX1Total, ctx)
  grads[prefix & ".ln1.gamma"] = dLN1.dGamma
  grads[prefix & ".ln1.beta"] = dLN1.dBeta
  result = dLN1.dX
  for i in 0 ..< result.data.len: result.data[i] += dX2.data[i]   # residual goc

proc forwardBatch*(m: NimformerModelJamba, idsBatch: seq[seq[int]], ctx: Backend): Tensor =
  var x = m.embed.lookupBatch(idsBatch, ctx)
  for i, blk in m.blocks:
    x = forwardBlock(blk, x, ctx, blockIndex = i)
  # (rms-fin): norm_f (RMSNorm) BEFORE lm_head, as in the HF base. This used to
  # call outProj on the un-normalised hidden -> wrong scale accumulated over 64 blocks.
  let xNorm = m.rmsF.forward(x, ctx)
  result = m.outProj.forward(xNorm, ctx)

# ═══════════════════════════════════════════════════════════════
# forwardBatchSession: INFERENCE-ONLY version, using the Session API THROUGHOUT when
# the backend supports it (Metal/CUDA/OpenCL), no backward (infer_lora.nim
# does not train through this path). NOT used for fine-tuning (finetune_lora.nim
# still uses forwardBlock/forwardBatch + backwardBlock above, unchanged).
#   - Block 'M' (Mamba2SSD): forwardMamba2 (test_nimformer.nim) ALREADY uses
#     the Session API itself when model.useSession=true - nothing to rewrite,
#     called exactly like forwardBlock above.
#   - Block 'A'/'S' (LinearPlusAttention): the original forwardLinearPlus does NOT use
#     a session (only its 2 Linears go through ctx, the core attention
#     always ran on the CPU - see the comment "a GPU-session path can be added later following
#     the Mamba2SSD template" where LinearPlusAttention is declared in
#     test_nimformer.nim). forwardBatchSession uses forwardLinearPlusSession
#     (NEW, also in test_nimformer.nim) instead - same math,
#     the same CPU fallback if the session fails, only the speed differs.
# ═══════════════════════════════════════════════════════════════
proc forwardBlockSession(blk: JambaBlock, x: Tensor, ctx: Backend;
                         blockIndex = -1): Tensor =
  # R/P inference batch path shares the same end-to-end block entrypoint as
  # training. Keep the stateful one-token decode path separate: this sequence
  # kernel starts from zero state and does not export persistent decode state.
  if blk.kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}:
    return forwardBlock(blk, x, ctx, blockIndex = blockIndex)

  # Legacy session path for M/A/S blocks.
  let x1 = blk.ln1.forward(x, ctx)
  var combined: Tensor
  case blk.kind
  of bkMamba:
    combined = forwardMamba2(blk.mamba, x1, ctx)                 # already session-capable
  of bkMambaReal:
    raise newException(CatchableError, "R block missed the inference block-kernel entrypoint")
  of bkMambaRealPlusAttn, bkMambaRealGated:
    raise newException(CatchableError, "P block missed the inference block-kernel entrypoint")
  of bkAttn:
    combined = forwardLinearPlusSession(blk.attn, x1, ctx)        # NEW - session-based
  of bkSpectron:
    let attnOut = forwardLinearPlusSession(blk.attn, x1, ctx)     # NEW - session-based
    let mambaOut = forwardMamba2(blk.mamba, x1, ctx)              # already session-capable
    let fuseIn = concatLastDim(attnOut, mambaOut)
    let gateLogit = blk.fuseGate.forward(fuseIn, ctx)
    let gate = beSigmoid(ctx, gateLogit.data)
    combined = newTensor(attnOut.shape)
    for i in 0 ..< combined.data.len:
      combined.data[i] = gate[i] * attnOut.data[i] + (1'f32 - gate[i]) * mambaOut.data[i]
  let x2 = addT(ctx, x, combined)
  let x3 = blk.ln2.forward(x2, ctx)
  let ffOut = blk.ff.forward(x3, ctx)
  result = addT(ctx, x2, ffOut)

proc forwardBatchSession*(m: NimformerModelJamba, idsBatch: seq[seq[int]], ctx: Backend): Tensor =
  ## Used by infer_lora.nim (generation loop) instead of forwardBatch - SAME
  ## math/result, different execution (GPU-resident Session API
  ## instead of each op uploading/downloading on its own) when the backend supports it. If
  ## the backend has no session support or the session fails midway, it falls back to
  ## the CPU as before (see forwardMamba2/forwardLinearPlusSession) - it NEVER
  ## produces tokens different from forwardBatch just because of a session error.
  var x = m.embed.lookupBatch(idsBatch, ctx)
  for i, blk in m.blocks:
    x = forwardBlockSession(blk, x, ctx, blockIndex = i)
  # (rms-fin): norm_f before lm_head to match the base + the training path.
  let xNorm = m.rmsF.forward(x, ctx)
  result = m.outProj.forward(xNorm, ctx)

# ═══════════════════════════════════════════════════════════════
# INCREMENTAL STATE (NEW) for the WHOLE MODEL - O(1)/token instead of forwardBatchSession
# above re-running the WHOLE sequence from the start each token (O(T), slower the longer it
# generates - confirmed by a real log at ~136s/token). Uses the incremental state
# of each Mamba3Real block (see Mamba3State/forwardStepMamba3Real in
# mamba3_real.nim, checked against the REAL HF formula
# transformers/models/mamba2/modeling_mamba2.py).
#
# EXPLICIT LIMITATION (not ignored): ONLY usable when the model pattern is ENTIRELY
# 'R' (Mamba3Real) - no 'A'/'M'/'S' blocks at all. Reason: an
# Attention block ('A') needs its own KV cache (keeping K/V of every past token) to attend
# to the past correctly - forwarding a single token WITHOUT history gives a WRONG
# result (not just slow), unlike Mamba3Real (whose state is enough to reproduce
# the right result). No KV cache is written for 'A' blocks here - if the pattern
# has non-'R' blocks, an EXPLICIT error is raised instead of silently running wrong.
# ═══════════════════════════════════════════════════════════════

type
  JambaGenState* = object
    ## 1 element/block - only holds a real value (Mamba3State) for 'R'
    ## blocks; other blocks (if any) are not used through this path (see the raise in
    ## forwardStepJamba).
    mambaStates*: seq[Mamba3State]

proc isPureMambaRealPattern*(m: NimformerModelJamba): bool =
  ## Checks UP FRONT (no GPU) whether the model is PURE 'R' -
  ## a REQUIRED condition for using forwardStepJamba safely. Call it before
  ## choosing between the incremental path and the old forwardBatchSession path.
  for blk in m.blocks:
    if blk.kind != bkMambaReal: return false
  return true

type
  HybridGenState* = object
    ## Incremental hybrid: R uses Mamba3State; A uses its own Linear++ state.
    mambaStates*: seq[Mamba3State]
    attnStates*: seq[LinearPlusInferState]
    hidden*: int

proc newHybridGenState*(m: NimformerModelJamba): HybridGenState =
  result.hidden = m.embedDim
  result.mambaStates = newSeq[Mamba3State](m.blocks.len)
  result.attnStates = newSeq[LinearPlusInferState](m.blocks.len)
  for i, blk in m.blocks:
    if blk.kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}:
      result.mambaStates[i] = newMamba3State(blk.mambaReal)
    if blk.kind in {bkAttn, bkMambaRealPlusAttn, bkMambaRealGated}:
      result.attnStates[i] = newLinearPlusInferState(m.embedDim, blk.attn.nHeads)

proc newJambaGenState*(m: NimformerModelJamba): JambaGenState =
  ## RAISES explicitly if the pattern is NOT pure 'R' - no silently creating a state
  ## and letting forwardStepJamba detect it later (fail early, clear message).
  if not isPureMambaRealPattern(m):
    raise newException(ValueError,
      "[JambaGenState] Model co block KHAC 'R' (Attention/Mamba2SSD/Spectron) - " &
      "incremental state is NOT supported for those blocks yet (needs a separate KV cache, " &
      "not written). Use forwardBatchSession (slower but CORRECT) for this pattern.")
  result.mambaStates = newSeq[Mamba3State](m.blocks.len)
  for i, blk in m.blocks:
    result.mambaStates[i] = newMamba3State(blk.mambaReal)

proc forwardStepJamba*(m: NimformerModelJamba, tokenId: int, state: var JambaGenState,
                        ctx: Backend): Tensor =
  ## EXACTLY 1 new token, O(1) (independent of the generated length) - REPLACES
  ## forwardBatchSession(m, @[ids], ctx) in the generate() loop when the
  ## pattern is pure 'R' (see isPureMambaRealPattern). Returns logits [1,1,vocab].
  assert m.blocks.len == state.mambaStates.len,
    "[forwardStepJamba] state does not match the model's block count (old JambaGenState?)"
  var x = m.embed.lookupBatch(@[@[tokenId]], ctx)   # [1,1,hiddenSize]
  for i, blk in m.blocks:
    doAssert blk.kind == bkMambaReal,
      "[forwardStepJamba] found a non-'R' block at position " & $i &
      " - this function must not be called for this pattern (see isPureMambaRealPattern)."
    let x1 = blk.rms1.forward(x, ctx)   # (rms-fin): RMSNorm instead of ln1
    let combined = forwardStepMamba3Real(blk.mambaReal, x1, state.mambaStates[i], ctx)
    x = forwardMlp(blk.mambaReal, addT(ctx, x, combined), ctx).y   # bkMambaReal: NO ln2/ff, exactly like forwardBlock
  # (rms-fin): norm_f before lm_head.
  let xNorm = m.rmsF.forward(x, ctx)
  result = m.outProj.forward(xNorm, ctx)

proc uploadResidentJamba*(m: var NimformerModelJamba, ctx: Backend,
                           mergedQuantNames: HashSet[string] = initHashSet[string]()) =
  ## (following uploadAllResidentWeights in generate.nim/
  ## nim_inference): uploads ALL weights to the GPU ONCE before
  ## entering the generate loop. mamba3_real.nim (inProj/outProj) and
  ## test_nimformer.nim (LinearPlusAttention qkvProj/outProj) ALREADY CALL
  ## `.forward()` of the same Linear type used by generate.nim (see
  ## nimformer.nim) - Linear.forward() ALREADY CHECKS residentReady/
  ## residentFp32Ready and takes the fast path (uploads only the small activation each
  ## step, no re-dequant/transpose of the weight) - so NOTHING needs changing in
  ## mamba3_real.nim/test_nimformer.nim, just calling uploadResident() once
  ## here for each Linear makes every later forward() faster AUTOMATICALLY.
  ##
  ## This MUST be called AFTER the weights are in FULL fp32 form in
  ## `.weight.data` (e.g. after mergeIn() in infer_lora.nim - once the dequant-
  ## resident inProj/outProj from the .nimq are filled into t[].data and LoRA
  ## is permanently added for the whole generate session) - calling it earlier (while
  ## `.weight.data` is still empty due to quant-resident) gives NOTHING to upload
  ## (uploadResident skips when `l.weight.data.len == 0`, see nimformer.nim).
  ##
  ## RAM saving (follow-up to the useQuant bug in lora.nim):
  ## mergedQuantNames = the set of parameter names the CALLER (infer_lora.generate) knows
  ## for sure were just dequantized + LoRA-added by mergeIn() for this session only
  ## (in practice lora.quantBase.keys) - marking "this fp32 copy is OUR
  ## OWN, not aliased/shared with any other Tensor". For EXACTLY
  ## those names, `.weight.data` is freed RIGHT after uploadResident()
  ## - otherwise the merged parameter would exist as 2 fp32 copies AT ONCE (CPU
  ## + GPU-resident); on integrated GPUs/unified memory (Intel Iris) both copies
  ## share system RAM = twice the RAM really needed, which caused the
  ## OOM kill right after the "matmul_q4 active" log line. The default is
  ## empty (empty set = free nothing, exactly the OLD behaviour) so other
  ## callers (if any, e.g. tests) are unaffected unless they pass it explicitly.
  for i in 0 ..< m.blocks.len:
    let p = &"block{i}"
    case m.blocks[i].kind
    of bkMambaReal:
      uploadResident(m.blocks[i].mambaReal.inProj, ctx)
      if (p & ".mambaReal.inProj.weight") in mergedQuantNames:
        m.blocks[i].mambaReal.inProj.weight.data = @[]
      uploadResident(m.blocks[i].mambaReal.outProj, ctx)
      if (p & ".mambaReal.outProj.weight") in mergedQuantNames:
        m.blocks[i].mambaReal.outProj.weight.data = @[]
    of bkMambaRealPlusAttn, bkMambaRealGated:
      uploadResident(m.blocks[i].mambaReal.inProj, ctx)
      if (p & ".mambaReal.inProj.weight") in mergedQuantNames:
        m.blocks[i].mambaReal.inProj.weight.data = @[]
      uploadResident(m.blocks[i].mambaReal.outProj, ctx)
      if (p & ".mambaReal.outProj.weight") in mergedQuantNames:
        m.blocks[i].mambaReal.outProj.weight.data = @[]
      uploadResident(m.blocks[i].attn.qkvProj, ctx)
      if (p & ".attn.qkvProj.weight") in mergedQuantNames:
        m.blocks[i].attn.qkvProj.weight.data = @[]
      uploadResident(m.blocks[i].attn.outProj, ctx)
      if (p & ".attn.outProj.weight") in mergedQuantNames:
        m.blocks[i].attn.outProj.weight.data = @[]
    of bkMamba:
      uploadResident(m.blocks[i].mamba.inProj, ctx)
      if (p & ".mamba.inProj.weight") in mergedQuantNames:
        m.blocks[i].mamba.inProj.weight.data = @[]
      uploadResident(m.blocks[i].mamba.outProj, ctx)
      if (p & ".mamba.outProj.weight") in mergedQuantNames:
        m.blocks[i].mamba.outProj.weight.data = @[]
    of bkAttn:
      uploadResident(m.blocks[i].attn.qkvProj, ctx)
      if (p & ".attn.qkvProj.weight") in mergedQuantNames:
        m.blocks[i].attn.qkvProj.weight.data = @[]
      uploadResident(m.blocks[i].attn.outProj, ctx)
      if (p & ".attn.outProj.weight") in mergedQuantNames:
        m.blocks[i].attn.outProj.weight.data = @[]
    of bkSpectron:
      uploadResident(m.blocks[i].mamba.inProj, ctx)
      if (p & ".mamba.inProj.weight") in mergedQuantNames:
        m.blocks[i].mamba.inProj.weight.data = @[]
      uploadResident(m.blocks[i].mamba.outProj, ctx)
      if (p & ".mamba.outProj.weight") in mergedQuantNames:
        m.blocks[i].mamba.outProj.weight.data = @[]
      uploadResident(m.blocks[i].attn.qkvProj, ctx)
      if (p & ".attn.qkvProj.weight") in mergedQuantNames:
        m.blocks[i].attn.qkvProj.weight.data = @[]
      uploadResident(m.blocks[i].attn.outProj, ctx)
      if (p & ".attn.outProj.weight") in mergedQuantNames:
        m.blocks[i].attn.outProj.weight.data = @[]
      uploadResident(m.blocks[i].fuseGate, ctx)
      if (p & ".fuseGate.weight") in mergedQuantNames:
        m.blocks[i].fuseGate.weight.data = @[]
  uploadResident(m.outProj, ctx)
  if "outProj.weight" in mergedQuantNames:
    m.outProj.weight.data = @[]

# ═══════════════════════════════════════════════════════════════
# PARAM REGISTRY - generic, no hand edits needed when a block kind is added.
# ═══════════════════════════════════════════════════════════════

proc paramList*(m: var NimformerModelJamba): seq[(string, ptr Tensor)] =
  result = @[("embed.weight", addr m.embed.weight), ("outProj.weight", addr m.outProj.weight),
             ("outProj.bias", addr m.outProj.bias),
             ("rmsF.gamma", addr m.rmsF.gamma)]   # norm_f - saved/loaded by the .bin checkpoint
  for i in 0 ..< m.blocks.len:
    let p = &"block{i}"
    result.add((p & ".ln1.gamma", addr m.blocks[i].ln1.gamma))
    result.add((p & ".ln1.beta", addr m.blocks[i].ln1.beta))
    result.add((p & ".ln2.gamma", addr m.blocks[i].ln2.gamma))
    result.add((p & ".ln2.beta", addr m.blocks[i].ln2.beta))
    if m.blocks[i].kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}:
      # the REAL norm used on the 'R'/'P' forward/backward path (rms-fin)
      result.add((p & ".rms1.gamma", addr m.blocks[i].rms1.gamma))
    result.add((p & ".ff.fc1.weight", addr m.blocks[i].ff.fc1.weight))
    result.add((p & ".ff.fc1.bias", addr m.blocks[i].ff.fc1.bias))
    result.add((p & ".ff.fc2.weight", addr m.blocks[i].ff.fc2.weight))
    result.add((p & ".ff.fc2.bias", addr m.blocks[i].ff.fc2.bias))
    case m.blocks[i].kind
    of bkMamba:
      result.add((p & ".mamba.inProj.weight", addr m.blocks[i].mamba.inProj.weight))
      result.add((p & ".mamba.inProj.bias", addr m.blocks[i].mamba.inProj.bias))
      result.add((p & ".mamba.outProj.weight", addr m.blocks[i].mamba.outProj.weight))
      result.add((p & ".mamba.outProj.bias", addr m.blocks[i].mamba.outProj.bias))
    of bkMambaReal:
      for (nm, pt) in paramListMamba3Real(m.blocks[i].mambaReal, p & ".mambaReal"):
        result.add((nm, pt))
    of bkMambaRealPlusAttn, bkMambaRealGated:
      for (nm, pt) in paramListMamba3Real(m.blocks[i].mambaReal, p & ".mambaReal"):
        result.add((nm, pt))
      # Fix: attnLn1 REMOVED - shares ln1 with Mamba
      result.add((p & ".attn.qkvProj.weight", addr m.blocks[i].attn.qkvProj.weight))
      result.add((p & ".attn.qkvProj.bias", addr m.blocks[i].attn.qkvProj.bias))
      result.add((p & ".attn.outProj.weight", addr m.blocks[i].attn.outProj.weight))
      result.add((p & ".attn.outProj.bias", addr m.blocks[i].attn.outProj.bias))
      if m.blocks[i].kind == bkMambaRealGated:
        result.add((p & ".fuseGate.weight", addr m.blocks[i].fuseGate.weight))
        result.add((p & ".fuseGate.bias", addr m.blocks[i].fuseGate.bias))
    of bkAttn:
      result.add((p & ".attn.qkvProj.weight", addr m.blocks[i].attn.qkvProj.weight))
      result.add((p & ".attn.qkvProj.bias", addr m.blocks[i].attn.qkvProj.bias))
      result.add((p & ".attn.outProj.weight", addr m.blocks[i].attn.outProj.weight))
      result.add((p & ".attn.outProj.bias", addr m.blocks[i].attn.outProj.bias))
    of bkSpectron:
      result.add((p & ".attn.qkvProj.weight", addr m.blocks[i].attn.qkvProj.weight))
      result.add((p & ".attn.qkvProj.bias", addr m.blocks[i].attn.qkvProj.bias))
      result.add((p & ".attn.outProj.weight", addr m.blocks[i].attn.outProj.weight))
      result.add((p & ".attn.outProj.bias", addr m.blocks[i].attn.outProj.bias))
      result.add((p & ".mamba.inProj.weight", addr m.blocks[i].mamba.inProj.weight))
      result.add((p & ".mamba.inProj.bias", addr m.blocks[i].mamba.inProj.bias))
      result.add((p & ".mamba.outProj.weight", addr m.blocks[i].mamba.outProj.weight))
      result.add((p & ".mamba.outProj.bias", addr m.blocks[i].mamba.outProj.bias))
      result.add((p & ".fuseGate.weight", addr m.blocks[i].fuseGate.weight))
      result.add((p & ".fuseGate.bias", addr m.blocks[i].fuseGate.bias))

type
  JambaAdamState* = object
    states*: Table[string, ApfAdamState]

proc paramQuantFlags*(m: var NimformerModelJamba): Table[string, ptr bool] =
  ## BUG FIX: in sync with paramList above, but returns pointers STRAIGHT to
  ## each Linear's `useQuant` field (only for ".weight" names actually
  ## backed by a Linear - embed.weight is skipped since Embedding has no
  ## useQuant, .bias is skipped since LoRA does not apply to biases).
  ##
  ## WHY: mergeIn/mergeInOne (lora.nim) only hold a `ptr Tensor` (pointing
  ## straight at the `.weight` field INSIDE the Linear) from paramList, NOT
  ## the Linear object containing it - so before this patch, after
  ## mergeIn wrote `t[].data = merged.data` (base + LoRA delta, REAL fp32),
  ## Linear.useQuant was STILL true. Then uploadResident()/Linear.forward()
  ## (nimformer.nim) checks `l.useQuant` FIRST, sees true and DEQUANTIZES AGAIN
  ## from `l.weightQ` (the ORIGINAL compressed copy, WITHOUT LoRA) - ignoring the freshly merged
  ## t[].data entirely. No crash, no warning, LoRA just silently "vanished" on
  ## every quant-resident parameter (exactly the checkpoint kind being hit: non-'R'
  ## blocks, log showing "mergeIn + uploadResident + forwardBatchSession").
  ##
  ## mergeIn/mergeInOne use this table to set useQuant=false RIGHT when writing
  ## merged.data (telling Linear.forward()/uploadResident() "this is the REAL fp32
  ## weight, use it directly, do not dequantize again"); restoreBase/restoreBaseOne set
  ## it back to true when clearing t[].data (so the next mergeIn - or any
  ## other uploadResident running in between - dequantizes correctly again from
  ## weightQ as originally designed).
  result = initTable[string, ptr bool]()
  result["outProj.weight"] = addr m.outProj.useQuant
  for i in 0 ..< m.blocks.len:
    let p = &"block{i}"
    if m.blocks[i].kind in {bkMambaReal,bkMambaRealPlusAttn,bkMambaRealGated}:
      result[p & ".mambaReal.mlpFc1.weight"] = addr m.blocks[i].mambaReal.mlpFc1.useQuant
    if m.blocks[i].kind in {bkMambaReal,bkMambaRealPlusAttn,bkMambaRealGated}:
      result[p & ".mambaReal.mlpFc2.weight"] = addr m.blocks[i].mambaReal.mlpFc2.useQuant
    result[p & ".ff.fc1.weight"] = addr m.blocks[i].ff.fc1.useQuant
    result[p & ".ff.fc2.weight"] = addr m.blocks[i].ff.fc2.useQuant
    case m.blocks[i].kind
    of bkMamba:
      result[p & ".mamba.inProj.weight"] = addr m.blocks[i].mamba.inProj.useQuant
      result[p & ".mamba.outProj.weight"] = addr m.blocks[i].mamba.outProj.useQuant
    of bkMambaReal:
      result[p & ".mambaReal.inProj.weight"] = addr m.blocks[i].mambaReal.inProj.useQuant
      result[p & ".mambaReal.outProj.weight"] = addr m.blocks[i].mambaReal.outProj.useQuant
    of bkMambaRealPlusAttn, bkMambaRealGated:
      result[p & ".mambaReal.inProj.weight"] = addr m.blocks[i].mambaReal.inProj.useQuant
      result[p & ".mambaReal.outProj.weight"] = addr m.blocks[i].mambaReal.outProj.useQuant
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj.useQuant
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj.useQuant
      if m.blocks[i].kind == bkMambaRealGated:
        result[p & ".fuseGate.weight"] = addr m.blocks[i].fuseGate.useQuant
    of bkAttn:
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj.useQuant
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj.useQuant
    of bkSpectron:
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj.useQuant
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj.useQuant
      result[p & ".mamba.inProj.weight"] = addr m.blocks[i].mamba.inProj.useQuant
      result[p & ".mamba.outProj.weight"] = addr m.blocks[i].mamba.outProj.useQuant
      result[p & ".fuseGate.weight"] = addr m.blocks[i].fuseGate.useQuant

proc paramLinears*(m: var NimformerModelJamba): Table[string, ptr Linear] =
  ## Returns a name → ptr Linear table (only Linears with a .weight, in sync with
  ## paramQuantFlags / paramList). Used to bind quantBase → weightQ/useQuant
  ## (setQuantWeight) before uploadResident / mergeInUploadStreamingJamba.
  ## Missing this step caused the IndexDefect "container is empty" on the
  ## quantResident + NIMBIN4 path (weight.data was @[] at newJambaBlock,
  ## but weightQ/useQuant had not been wired from lora.quantBase).
  result = initTable[string, ptr Linear]()
  result["outProj.weight"] = addr m.outProj
  for i in 0 ..< m.blocks.len:
    let p = &"block{i}"
    if m.blocks[i].kind in {bkMambaReal,bkMambaRealPlusAttn,bkMambaRealGated}:
      result[p & ".mambaReal.mlpFc1.weight"] = addr m.blocks[i].mambaReal.mlpFc1
    if m.blocks[i].kind in {bkMambaReal,bkMambaRealPlusAttn,bkMambaRealGated}:
      result[p & ".mambaReal.mlpFc2.weight"] = addr m.blocks[i].mambaReal.mlpFc2
    result[p & ".ff.fc1.weight"] = addr m.blocks[i].ff.fc1
    result[p & ".ff.fc2.weight"] = addr m.blocks[i].ff.fc2
    case m.blocks[i].kind
    of bkMamba:
      result[p & ".mamba.inProj.weight"] = addr m.blocks[i].mamba.inProj
      result[p & ".mamba.outProj.weight"] = addr m.blocks[i].mamba.outProj
    of bkMambaReal:
      result[p & ".mambaReal.inProj.weight"] = addr m.blocks[i].mambaReal.inProj
      result[p & ".mambaReal.outProj.weight"] = addr m.blocks[i].mambaReal.outProj
    of bkMambaRealPlusAttn, bkMambaRealGated:
      result[p & ".mambaReal.inProj.weight"] = addr m.blocks[i].mambaReal.inProj
      result[p & ".mambaReal.outProj.weight"] = addr m.blocks[i].mambaReal.outProj
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj
      if m.blocks[i].kind == bkMambaRealGated:
        result[p & ".fuseGate.weight"] = addr m.blocks[i].fuseGate
    of bkAttn:
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj
    of bkSpectron:
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj
      result[p & ".mamba.inProj.weight"] = addr m.blocks[i].mamba.inProj
      result[p & ".mamba.outProj.weight"] = addr m.blocks[i].mamba.outProj
      result[p & ".fuseGate.weight"] = addr m.blocks[i].fuseGate

proc newJambaAdamState*(m: var NimformerModelJamba): JambaAdamState =
  result.states = initTable[string, ApfAdamState]()
  for (name, t) in paramList(m):
    result.states[name] = newApfAdamState(t[].data.len)

proc computeLossAndGrads*(m: var NimformerModelJamba, idsBatch: seq[seq[int]],
                           targetsBatch: seq[seq[int]], ctx: Backend,
                           trainLmHead: bool = true, trainEmbed: bool = true):
                          tuple[loss: float32, grads: Table[string, Tensor]] =
  ## Forward+backward only (does NOT update Adam itself) so lora.nim can
  ## decide which parameters to update (LoRA A/B) instead of updating the base weights
  ## directly like trainStep below does for the whole model.
  var x = m.embed.lookupBatch(idsBatch, ctx)
  var xs: seq[Tensor] = @[x]
  for i, blk in m.blocks:
    x = forwardBlock(blk, x, ctx, blockIndex = i)
    xs.add(x)
  # (rms-fin): norm_f BEFORE lm_head in both forward and backward.
  let xNorm = m.rmsF.forward(x, ctx)
  let logits = m.outProj.forward(xNorm, ctx)
  let (loss, dLogits) = softmaxCrossEntropy(logits, targetsBatch, ctx)

  var grads = initTable[string, Tensor]()
  var dHeadInput: Tensor
  if trainLmHead:
    let dOutProj = m.outProj.backward(xNorm, dLogits, ctx)
    grads["outProj.weight"] = dOutProj.dW
    grads["outProj.bias"] = dOutProj.dB
    dHeadInput = dOutProj.dX
  else:
    dHeadInput = m.outProj.backwardInputOnly(xNorm, dLogits, ctx)
  let dNormF = m.rmsF.backward(x, dHeadInput, ctx)
  if trainLmHead: grads["rmsF.gamma"] = dNormF.dGamma
  var dX = dNormF.dX
  for i in countdown(m.blocks.len - 1, 0):
    dX = backwardBlock(m.blocks[i], xs[i], dX, ctx, &"block{i}", grads)
  if trainEmbed:
    grads["embed.weight"] = m.embed.backward(idsBatch, dX, ctx)
  result = (loss, grads)

proc trainStep*(m: var NimformerModelJamba, adam: var JambaAdamState,
                idsBatch: seq[seq[int]], targetsBatch: seq[seq[int]],
                ctx: Backend, lr: float32 = 3e-4'f32,
                requantizeEvery: int = 50): float32 =
  let (loss, grads) = computeLossAndGrads(m, idsBatch, targetsBatch, ctx)
  for (name, t) in paramList(m):
    if name in grads:
      var st = adam.states[name]
      discard apfAdamStepFused(ctx, t[], grads[name], st, lr, requantizeEvery = requantizeEvery)
      adam.states[name] = st
    else:
      stderr.writeLine("[WARNING] Missing gradient for parameter: " & name)
  result = loss

# ═══════════════════════════════════════════════════════════════
# CHECKPOINT: uses raw .bin directly (rawbin.nim), NOT .nimq here.
# .nimq (quant.nim) is reserved for nim_inference.nim/test_nimformer.nim -
# the whole fine-tune pipeline (graft_attention/finetune_lora/infer_lora) reads
# and writes checkpoints directly as .bin. arch + pattern + step are packed
# into the "meta" section of the .bin, no .pattern.json sidecar needed any more.
# ═══════════════════════════════════════════════════════════════

proc saveCheckpoint*(m: var NimformerModelJamba, path: string, step: int = 0) =
  var tensors: seq[RawBinTensor] = @[]
  for (name, t) in paramList(m):
    tensors.add(RawBinTensor(name: name, shape: t[].shape, data: t[].data))
  var meta = initTable[string, string]()
  meta["vocab"] = $m.vocab
  meta["embedDim"] = $m.embedDim
  meta["nHeads"] = $m.nHeads
  meta["ffMult"] = $m.ffMult
  meta["pattern"] = m.pattern
  meta["step"] = $step
  saveRawBin(path, meta, tensors)
  echo &"[jamba] Saved .bin checkpoint -> {path} ({tensors.len} tensors, step={step})"

proc loadCheckpoint*(path: string, useSession: bool = true): tuple[model: NimformerModelJamba, step: int] =
  let (meta, tensors) = loadRawBin(path)
  template reqMeta(key: string): string =
    if key notin meta:
      raise newException(IOError, ".bin checkpoint is missing required meta '" & key & "': " & path)
    meta[key]
  let vocab = parseInt(reqMeta("vocab"))
  let embedDim = parseInt(reqMeta("embedDim"))
  let nHeads = parseInt(reqMeta("nHeads"))
  let ffMult = parseInt(reqMeta("ffMult"))
  let pattern = reqMeta("pattern")
  let step = (if "step" in meta: parseInt(meta["step"]) else: 0)

  var m = newNimformerModelJamba(vocab, embedDim, nHeads, ffMult, pattern, useSession)
  var lookup = initTable[string, RawBinTensor]()
  for t in tensors:
    lookup[t.name] = t
  for (name, t) in paramList(m):
    if name in lookup:
      t[].data = lookup[name].data
    else:
      stderr.writeLine("[WARNING] .bin checkpoint is missing parameter '" & name & "' - keeping its random init")
  result = (m, step)

proc loadNormsFromHfTensors*(m: var NimformerModelJamba, tensors: seq[RawBinTensor]): int =
  ## `tensors` is the list of HF tensors already read (loadAllHFTensors from safetensors,
  ## or the dequantized list from a .nimq - any list with the HF name schema). Returns the number
  ## of norms loaded. Expected = number of 'R'/'P' blocks + 1 (norm_f). Warns when
  ## some are missing - does NOT raise (the caller decides), but a missing norm_f is a serious error.
  var lookup = initTable[string, RawBinTensor]()
  for t in tensors: lookup[t.name] = t
  result = 0
  for i in 0 ..< m.blocks.len:
    if m.blocks[i].kind notin {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}: continue
    let key = "backbone.layers." & $i & ".norm.weight"
    if key in lookup:
      if lookup[key].data.len == m.blocks[i].rms1.gamma.data.len:
        m.blocks[i].rms1.gamma.data = lookup[key].data
        inc result
      else:
        stderr.writeLine(&"[WARNING] {key}: shape mismatch ({lookup[key].data.len} != " &
                         $m.blocks[i].rms1.gamma.data.len & ") - keeping gamma=1")
    else:
      stderr.writeLine("[WARNING] Missing " & key & " in the weight source - keeping gamma=1")
  var foundF = false
  for key in ["backbone.norm_f.weight", "norm_f.weight"]:
    if key in lookup and lookup[key].data.len == m.rmsF.gamma.data.len:
      m.rmsF.gamma.data = lookup[key].data
      inc result
      foundF = true
      break
  if not foundF:
    stderr.writeLine("[WARNING] backbone.norm_f.weight NOT found - rmsF keeps gamma=1, " &
                     "the hidden state entering lm_head will be off the base. Check the weight source.")
  echo &"[jamba] rms-fin: loaded {result} pretrain norm weights (rms1 + norm_f)"

const TRAIN_NORM_PRESET* = "rms-fin-gn"
const TRAIN_MAMBA_NORM_IMPL* = "mamba3-siso-mlp-rope-trapezoidal-v1"
  ## The infer preset name (lora.setGpuNormPreset) matching EXACTLY this trainer's forward:
  ## base RMSNorm gamma (syncRmsFromBaseNorms) for BOTH the R branch and the P sidecar of each block
  ## (forwardBlock: x1 = rms1(x) shared), norm_f (rmsF) before lm_head, gated RMSNorm
  ## over 8 groups (Mamba-SSM reference, group_size=8192/8). finetune_lora writes it into the checkpoint's meta["normPreset"],
  ## infer_lora reads it back to use EXACTLY that preset - the two sides cannot diverge.
  ## CHANGING the forward normalisation => change this constant (and setGpuNormPreset in lora.nim).

proc syncRmsFromBaseNorms*(m: var NimformerModelJamba) =
  ## (train/infer preset match): loadNormsFromHfTensors was never called,
  ## so rms1/rmsF kept gamma=1 during training while infer (lora.nim) used
  ## ln1.gamma / normF.gamma = the REAL pretrain weights (loaded by load_codestral_*).
  ## Call this AFTER every base load: copies ln1.gamma -> rms1.gamma (R/P) and
  ## normF.gamma -> rmsF.gamma so training and inference use the SAME gamma.
  var n = 0
  for i in 0 ..< m.blocks.len:
    if m.blocks[i].kind notin {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}: continue
    if m.blocks[i].ln1.gamma.data.len == m.blocks[i].rms1.gamma.data.len:
      m.blocks[i].rms1.gamma.data = m.blocks[i].ln1.gamma.data
      inc n
  if m.normF.gamma.data.len == m.rmsF.gamma.data.len:
    m.rmsF.gamma.data = m.normF.gamma.data
    inc n
  echo "[jamba] syncRmsFromBaseNorms: synced " & $n & " gammas (rms1 + rmsF) from the base"
