## lora.nim
## LoRA (low-rank adapter) for NimformerModelJamba, + a function that "grafts"
## NEW LinearPlusAttention blocks into an existing checkpoint that is
## only Mamba2SSD (a pattern like "MMMMMM...").
##
## The LoRA idea is the same as the Python script provided (LoRALinear: frozen base,
## only the low-rank A,B are learned): instead of writing a separate "LoRALinear" detached from
## Mamba2SSD/LinearPlusAttention (which would require reworking forwardMamba2/
## forwardLinearPlus to accept a replaceable Linear - too invasive),
## we use one mathematical trick:
##   W_eff = W_base + (alpha/r) * B @ A       (B:[outF,r], A:[r,inF])
## then TEMPORARILY OVERWRITE w_base.data = W_eff, run the existing forwardBlock/
## backwardBlock/computeLossAndGrads unchanged (not a single line changed in
## jamba_hybrid.nim), the backward returns dW_eff for that parameter name - the
## chain rule then gives dA, dB (the base has NO gradient and is not updated),
## and w_base.data is RESTORED to the original before the next step.

import std/[math, random, strformat, tables, strutils, times, os, monotimes]

proc monoSecs*(): float =
  ## Monotonic clock, does NOT advance while the machine sleeps (epochTime does).
  getMonoTime().ticks.float / 1e9

import nimformer
import backend
import jamba_hybrid   # HybridGenState, JambaGenState, isPureMambaRealPattern, addT, ...
import linear_plus_infer
import mamba2_real     # forwardStepMamba2Real
import mamba2_ssm_gpu
import mamba2_fused_step
import fp4_gemv_gpu    # prepare path: gFp4CacheGpu, fp4GemvGpu, Shared weight
import test_nimformer  # forwardLinearPlusSession (Attention GPU)
import quant           # QuantTensor, dequantizeTensor
import customfloat     # buildDecodeLUT, decodeArrayLUT
import session_vm      # bytecode emit / .bvm / sessionVmFor*
import jit_vm          # HotSpot-style: interpret -> profile -> Metal JIT cache
when not defined(noOneDispatch):
  import onedispatch   # ON by default (disable at build: -d:noOneDispatch) : 1 compute dispatch / token for the whole R stack (see ONE_DISPATCH_PLAN.md)

# Global JIT cache (shared by infer + finetune)
var gJitCache* = initJitCache(defaultJitConfig())
var gMicroVmBackend* = gbAuto

# Decode profiling (opt-in via BYBY_DECODE_PROFILE=1). Accumulates wall-clock
# time per phase across all tokens so the caller can print a breakdown at the
# end of generation. Zero overhead when disabled (timers are skipped).
var gProfBegin*, gProfEncode*, gProfEnd*, gProfRead*: float64
var gProfTokens*: int
var gProfFirst*: string   ## the first token is reported separately (usually much slower: first pipeline/driver use)
var gProfFirstMs*: float64
# BYBY_DECODE_SPLIT=1 (diagnostics, SLOWER): sessionEnd+sessionBegin between the parts
# of 1 token (one-dispatch section / G blocks / lm_head) to time each part on the GPU.
var gDecSplitOn = getEnv("BYBY_DECODE_SPLIT", "0") != "0"
var gDecSplitActive = false
var gDecSplitMs = initOrderedTable[string, float64]()
var gDecSplitTokens = 0

proc decSplitMark(be: GpuBackend; label: string; t0: var float64) =
  ## Closes the current command buffer (waits for the GPU), adds the time to label, opens a new buffer.
  if not (gDecSplitOn and gDecSplitActive): return
  if not sessionEnd(be):
    raise newException(CatchableError, "BYBY_DECODE_SPLIT: sessionEnd failed after " & label &
      ": " & sessionGetLastError(be))
  let t1 = monoSecs()
  gDecSplitMs.mgetOrPut(label, 0.0) += (t1 - t0) * 1000.0
  if not sessionBegin(be):
    raise newException(CatchableError, "BYBY_DECODE_SPLIT: sessionBegin failed after " & label)
  t0 = monoSecs()

proc decodeSplitSummary*(): string =
  if gDecSplitTokens == 0 or gDecSplitMs.len == 0: return ""
  var tot = 0.0
  for _, v in gDecSplitMs: tot += v
  result = &"[infer] BYBY_DECODE_SPLIT over {gDecSplitTokens} tokens (first token excluded), ms/tok:"
  for k, v in gDecSplitMs:
    let per = v / float64(gDecSplitTokens)
    result.add &"\n[infer]   {k:<14} {per:>8.1f}ms {100.0 * v / max(tot, 1e-9):>5.1f}%"

proc decodeProfileSummary*(): string =
  ## Human-readable per-token breakdown of the decode hot path. Empty string
  ## when no tokens were profiled. Called by infer_lora.nim after generation.
  ## The averages EXCLUDE the first token (printed separately on the 2nd line).
  if gProfTokens <= 1: return gProfFirst
  let n = float64(gProfTokens - 1)
  let perBegin = gProfBegin * 1000.0 / n
  let perEncode = gProfEncode * 1000.0 / n
  let perEnd = gProfEnd * 1000.0 / n
  let perRead = gProfRead * 1000.0 / n
  let total = perBegin + perEncode + perEnd + perRead
  result = &"[infer] decode profile over {gProfTokens - 1} tokens (first token excluded): " &
    &"sessionBegin={perBegin:.1f}ms encodeOneToken={perEncode:.1f}ms " &
    &"sessionEnd={perEnd:.1f}ms sessionRead={perRead:.1f}ms TOTAL={total:.1f}ms/tok\n" & gProfFirst

proc initMicroVmJit*(tag = "default"; selectedBackend = gbAuto) =
  ## Metal compiles per-program MSL; Vulkan uses the shared SPIR-V VM and must
  ## not initialize Metal as a side effect of JIT warmup.
  gMicroVmBackend = selectedBackend
  var cfg = defaultJitConfig()
  if selectedBackend == gbVulkan:
    cfg.enableMetalJit = false
  else:
    if not ensureJitDeviceReady():
      raise newException(CatchableError, "initMicroVmJit: GPU initialisation failed")
  gJitCache = initJitCache(cfg)

proc warmupMicroVmJit*(rounds = 12) =
  if gMicroVmBackend == gbVulkan: return
  var prog = buildExampleLutFma()
  for i in 1 .. rounds:
    discard gJitCache.dispatchProgram(prog, "warmup_lut_fma")

type
  LoraAdapter* = object
    r*, inF*, outF*: int
    alpha*: float32
    scale*: float32       ## = alpha / r
    A*: Tensor            ## shape [r, inF]
    B*: Tensor            ## shape [outF, r]
    mappedA*, mappedB*: MappedView[float32]

  LoraTrainState* = object
    lr*, wd*: float32
    adamA*, adamB*: Table[string, ApfAdamState]

  JambaLora* = object
    adapters*: Table[string, LoraAdapter]   ## key = parameter name in paramList (e.g. "block2.mamba.inProj.weight")
    backup*: Table[string, seq[float32]]    ## original (frozen) base values - restored after each step
    quantBase*: Table[string, QuantTensor]  ## NEW: only has entries for "quant-resident" parameters (no
                                             ## resident fp32 copy kept in RAM - dequantized through the LUT
                                             ## ON EVERY mergeIn, freed right after restoreBase). Empty = old
                                             ## mode, behaviour UNCHANGED (100% backward-compatible).

template aValues*(ad: LoraAdapter): untyped = weightValues(ad.A.data, ad.mappedA)
template bValues*(ad: LoraAdapter): untyped = weightValues(ad.B.data, ad.mappedB)

proc attachAdapter(lin: var Linear; ad: LoraAdapter) =
  lin.loraA = ad.A.data
  lin.loraB = ad.B.data
  lin.mappedLoraA = ad.mappedA
  lin.mappedLoraB = ad.mappedB

proc newLoraAdapter*(inF, outF, r: int, alpha: float32 = -1'f32): LoraAdapter =
  ## alpha defaults to 2*r (the common rule of thumb, also alpha=8,r=4
  ## in the Python script provided -> alpha=2r).
  ##
  ## FP16 WARNING: the base weights of Mamba-Chat-2.8B were originally stored/trained in
  ## fp16 (see pytorch_bin_loader.nim). Even though everything is fp32 once read into Nim
  ## (this framework's Tensor is ALWAYS fp32, see nimformer.nim), the values
  ## INSIDE still carry fp16 "traces" - many weights are very close to 0
  ## (below ~6e-5, the fp16 min-normal) or rounded to exactly 1
  ## of the ~2048 possible values in their magnitude range. Practical
  ## consequences for LoRA:
  ##   - A/B must NOT be requantized through low-bit CustomFloat/APF (requantizeEvery
  ##     is forced to int.high in trainStepLora/trainStepGraft below
  ##     - KEEP IT, do not disable it, or LoRA gradients
  ##     vanish by being rounded to 0).
  ##   - A small A init (sqrt(1/inF), as below) is SAFE since B=0 so the initial delta
  ##     is 0 even with a small A, not accumulating with the "near 0" part of the base.
  ##   - the LoRA learning rate should be 1e-4 or lower (CFG_LORA_LR in
  ##     finetune_lora.nim) - a higher lr lets gradients from near-0 base weights make A/B
  ##     jump too fast and destabilise (similar to the "loss spike" often seen when
  ##     fine-tuning an fp16 model with an fp32 optimizer without lowering lr).
  result.r = r
  result.inF = inF
  result.outF = outF
  result.alpha = if alpha < 0'f32: float32(2 * r) else: alpha
  result.scale = result.alpha / float32(r)
  # A: small random init (kaiming-ish); B: init 0 -> the initial delta = 0,
  # the model right after attaching LoRA is identical to the original (standard per the LoRA paper).
  result.A = randnTensor(@[r, inF], sqrt(1'f32 / float32(inF)))
  result.B = newTensor(@[outF, r], 0'f32)

proc matmul2D(ctx: Backend, a: Tensor, b: Tensor): Tensor =
  ## a:[M,K] @ b:[K,N] -> [M,N], using the existing beMatmul in backend.nim.
  let M = a.shape[0]; let K = a.shape[1]; let K2 = b.shape[0]; let N = b.shape[1]
  assert K == K2
  result = newTensor(@[M, N])
  result.data = beMatmul(ctx, a.data, M, K, b.data, K2, N)

proc constArr(scale: float32, n: int): seq[float32] =
  ## A constant array to broadcast 1 scalar into beMul/beAdd (backend.nim has no
  ## separate "scalar multiply" kernel, only elementwise ops on 2 same-sized arrays - this
  ## reuses the existing GPU API instead of a hand-written multiply loop).
  ## Filling this array is O(n) CPU but it is just a memset, NOT arithmetic
  ## (no multiply/add) - the real computation (multiply, add) is all in
  ## beMul/beAdd below, running on ctx (GPU).
  result = newSeq[float32](n)
  for i in 0 ..< n: result[i] = scale

proc mergedWeight*(base: Tensor, ad: LoraAdapter, ctx: Backend): Tensor =
  ## W_eff = W_base + scale * B @ A, shape unchanged [outF, inF].
  ## All multiplies/adds here go through beMul/beAdd (GPU), NO more CPU
  ## for loop accumulating element by element as before.
  let delta = matmul2D(ctx, Tensor(shape: ad.B.shape, data: @(ad.bValues)),
    Tensor(shape: ad.A.shape, data: @(ad.aValues)))   # [outF,r]@[r,inF] -> [outF,inF], on the GPU (beMatmul)
  let scaled = beMul(ctx, delta.data, constArr(ad.scale, delta.data.len))
  result = newTensor(base.shape)
  result.data = beAdd(ctx, base.data, scaled)

proc zeroAdapterOutRows*(ad: var LoraAdapter; row0, row1: int) =
  ## Zeroes rows B[row0 ..< row1] (LoRA delta = 0 on those output rows).
  if ad.r <= 0 or ad.B.data.len != ad.outF * ad.r: return
  let a = max(0, row0)
  let b = min(ad.outF, row1)
  for row in a ..< b:
    let off = row * ad.r
    for k in 0 ..< ad.r:
      ad.B.data[off + k] = 0'f32

proc zeroGradOutRows*(dB: var Tensor; row0, row1: int) =
  if dB.shape.len != 2: return
  let outF = dB.shape[0]
  let r = dB.shape[1]
  if dB.data.len != outF * r: return
  let a = max(0, row0)
  let b = min(outF, row1)
  for row in a ..< b:
    let off = row * r
    for k in 0 ..< r:
      dB.data[off + k] = 0'f32

proc zeroAdamOutRows*(st: var ApfAdamState; outF, r, row0, row1: int) =
  let n = outF * r
  if st.m.len != n or st.v.len != n or r <= 0: return
  let a = max(0, row0)
  let b = min(outF, row1)
  for row in a ..< b:
    let off = row * r
    for k in 0 ..< r:
      st.m[off + k] = 0'f32
      st.v[off + k] = 0'f32

proc adapterOutRowsAbsMax*(ad: LoraAdapter; row0, row1: int): float32 =
  result = 0'f32
  if ad.r <= 0 or ad.B.data.len != ad.outF * ad.r: return
  let a = max(0, row0)
  let b = min(ad.outF, row1)
  for row in a ..< b:
    let off = row * ad.r
    for k in 0 ..< ad.r:
      result = max(result, abs(ad.B.data[off + k]))

proc freezeInProjSsmLora*(m: NimformerModelJamba; lo: var JambaLora;
                          loState: var LoraTrainState) =
  ## G/R: LoRA inProj must not write B, C, dt. Old resume checkpoints may have
  ## drifted on these rows — zero B and the matching Adam m/v again.
  for i, blk in m.blocks:
    if blk.kind notin {bkMambaReal, bkMambaRealGated, bkMambaRealPlusAttn}:
      continue
    let name = "block" & $i & ".mambaReal.inProj.weight"
    if name notin lo.adapters: continue
    let (row0, row1) = inProjSsmRowRange(blk.mambaReal)
    var ad = lo.adapters[name]
    zeroAdapterOutRows(ad, row0, row1)
    lo.adapters[name] = ad
    if name in loState.adamB:
      var stB = loState.adamB[name]
      zeroAdamOutRows(stB, ad.outF, ad.r, row0, row1)
      loState.adamB[name] = stB

proc gradToAB*(dWeff: Tensor, ad: LoraAdapter, ctx: Backend): tuple[dA, dB: Tensor] =
  ## dWeff: [outF,inF] (gradient w.r.t. W_eff, computed by the backward as
  ## usual through Linear.backward). Chain rule:
  ##   dB = scale * dWeff @ A^T     : [outF,inF]@[inF,r] -> [outF,r]
  ##   dA = scale * B^T @ dWeff     : [r,outF]@[outF,inF] -> [r,inF]
  ## The multiplication by `scale` uses beMul (GPU) instead of a CPU for loop.
  let At = transpose(ad.A)   # [inF, r]
  let Bt = transpose(ad.B)   # [r, outF]
  var dB = matmul2D(ctx, dWeff, At)
  var dA = matmul2D(ctx, Bt, dWeff)
  dB.data = beMul(ctx, dB.data, constArr(ad.scale, dB.data.len))
  dA.data = beMul(ctx, dA.data, constArr(ad.scale, dA.data.len))
  result = (dA, dB)

# ═══════════════════════════════════════════════════════════════
# Attach LoRA to the given weight tensors (by name in jamba_hybrid's paramList,
# e.g. "block0.mamba.inProj.weight",
# "block3.attn.qkvProj.weight" ...). Only 2D tensors (Linear
# weights [outF,inF]) can take it - NOT biases/LayerNorm/embeddings.
# ═══════════════════════════════════════════════════════════════
proc attachLora*(m: var NimformerModelJamba, targetNames: seq[string],
                  r: int = 4, alpha: float32 = -1'f32): JambaLora =
  result.adapters = initTable[string, LoraAdapter]()
  result.backup = initTable[string, seq[float32]]()
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  for name in targetNames:
    if name notin byName:
      stderr.writeLine("[lora] WARNING: parameter '" & name & "' not found, skipped")
      continue
    let t = byName[name]
    if t[].shape.len != 2:
      stderr.writeLine("[lora] WARNING: '" & name & "' is not a 2D matrix, skipped")
      continue
    let outF = t[].shape[0]; let inF = t[].shape[1]
    result.adapters[name] = newLoraAdapter(inF, outF, r, alpha)
  echo &"[lora] Attached LoRA (r={r}) to {result.adapters.len}/{targetNames.len} parameters"

proc attachLoraQuantResident*(m: var NimformerModelJamba, quantWeights: Table[string, QuantTensor],
                               targetNames: seq[string], r: int = 4,
                               alpha: float32 = -1'f32): JambaLora =
  ## Like attachLora, but for parameters that are NO longer resident fp32
  ## in t[].data (too heavy on RAM for a 7B model - see the streaming explanation
  ## in finetune_lora.nim) but live in `quantWeights` (from
  ## load_codestral_into_jamba.loadCodestralQuantResident). The shape comes FROM
  ## QuantTensor.shape (t[].data is not read, since at this point t[].data is still garbage/
  ## random init from newMamba2Real, NOT yet filled - it is filled ON EVERY
  ## mergeIn through the LUT dequant, not at attach time).
  result.adapters = initTable[string, LoraAdapter]()
  result.backup = initTable[string, seq[float32]]()
  result.quantBase = quantWeights
  if targetNames.len == 0:
    raise newException(ValueError,
      "attachLoraQuantResident: targetNames is required; quant weights are not LoRA-ified automatically")
  var seen = initTable[string, bool]()
  for name in targetNames:
    if name in seen:
      raise newException(ValueError, "attachLoraQuantResident: LoRA target bi lap: " & name)
    seen[name] = true
    if name notin quantWeights:
      raise newException(ValueError,
        "attachLoraQuantResident: LoRA target not found in the quant weights: " & name)
    let qt = quantWeights[name]
    if qt.shape.len != 2:
      raise newException(ValueError,
        "attachLoraQuantResident: quant-resident LoRA target is not a 2D matrix: " & name)
    let outF = qt.shape[0]; let inF = qt.shape[1]
    result.adapters[name] = newLoraAdapter(inF, outF, r, alpha)
  echo &"[lora] Attached quant-resident LoRA (r={r}) to {result.adapters.len} targets"

proc mergeQuantAdapters*(other: JambaLora, into: var JambaLora) =
  ## Merges adapters from a regular attachLora (e.g. for newly grafted 'A' blocks) into
  ## the SAME JambaLora that already has quant-resident adapters - so 1 training loop
  ## covers both kinds. Do NOT call attachLora twice and OVERWRITE the result
  ## (that loses the previous quantBase) - use this function to MERGE.
  for name, ad in other.adapters:
    into.adapters[name] = ad
  for name, b in other.backup:
    into.backup[name] = b

proc allProjectionWeightNames*(m: var NimformerModelJamba): seq[string] =
  ## Helper: lists every 2D weight name "LoRA-able" (qkvProj/
  ## outProj of attn, inProj/outProj of mamba, fuseGate) - used to
  ## LoRA-ify the WHOLE model instead of picking blocks by hand.
  result = @[]
  for (name, t) in paramList(m):
    if t[].shape.len == 2 and (".weight" in name) and
       ("ln1" notin name) and ("ln2" notin name) and (name != "embed.weight"):
      result.add(name)

proc newLoraTrainState*(lora: JambaLora, lr: float32 = 1e-4'f32, wd: float32 = 0.01'f32): LoraTrainState =
  result.lr = lr
  result.wd = wd
  result.adamA = initTable[string, ApfAdamState]()
  result.adamB = initTable[string, ApfAdamState]()
  for name, ad in lora.adapters:
    result.adamA[name] = newApfAdamState(ad.A.data.len)
    result.adamB[name] = newApfAdamState(ad.B.data.len)

var gLutNameCache = initTable[string, array[256, float32]]()  # LUT cache by CustomFloat name - built once per spec (fp4 e2m1 shared by all 385 tensors, not rebuilt each time)

proc dequantViaLUT(qt: QuantTensor): seq[float32] =
  ## Like quant.dequantizeTensor but uses a LUT (faster, see customfloat.
  ## buildDecodeLUT) when qt.cf.itemSize==1 (fp4/fp8) - falls back to the
  ## regular dequantizeTensor (int8/int4/fp32raw or itemSize!=1) when it
  ## does not apply. The RESULT IS IDENTICAL to dequantizeTensor (verified in test_lut.nim).
  case qt.kind
  of qkFp8E4M3, qkFp8E5M2, qkCustom, qkAuto:
    if qt.cf.itemSize == 1:
      var lut: array[256, float32]
      if qt.cf.name in gLutNameCache:
        lut = gLutNameCache[qt.cf.name]
      else:
        lut = buildDecodeLUT(qt.cf)
        gLutNameCache[qt.cf.name] = lut
      let raw = decodeArrayLUT(qt.bytes, lut)
      let s = if qt.scale.len > 0: qt.scale[0] else: 1'f32
      if s == 1'f32: return raw
      result = newSeq[float32](raw.len)
      for i in 0 ..< raw.len: result[i] = raw[i] * s
      return result
  else: discard
  result = dequantizeTensor(qt)   # fallback for every other case

proc mergeInOnePlain*(m: var NimformerModelJamba, lora: JambaLora, name: string) =
  ## The ADAPTER-FREE version of mergeInOne - for quant-resident parameters
  ## that are NOT LoRA-ified (e.g. lm_head/outProj.weight: fully frozen,
  ## not trained, only dequantized TEMPORARILY while used and then freed - EXACTLY
  ## the spirit of mergeInOne for 'R' blocks, but the original mergeInOne reads
  ## `lora.adapters[name]` UNCONDITIONALLY (a KeyError if this parameter has no
  ## adapter, as with lm_head). This function ONLY dequantizes through the LUT,
  ## it adds NO LoRA delta (there is no adapter to add).
  var byName = initTable[string, ptr Tensor]()
  for (nm, t) in paramList(m):
    if nm == name: byName[nm] = t
  if name notin byName: return
  if name notin lora.quantBase: return
  byName[name][].data = dequantViaLUT(lora.quantBase[name])

proc restoreBaseOnePlain*(m: var NimformerModelJamba, lora: JambaLora, name: string) =
  ## Mirror of mergeInOnePlain - frees right after use (like
  ## restoreBaseOne, but does not touch lora.backup since there is nothing to
  ## back up - this parameter was never really "LoRA-merged").
  var byName = initTable[string, ptr Tensor]()
  for (nm, t) in paramList(m):
    if nm == name: byName[nm] = t
  if name notin byName: return
  if name notin lora.quantBase: return
  byName[name][].data = @[]

proc dropZeroLoraAdapters*(lo: var JambaLora): seq[string] =
  ## Drops adapters whose A or B is ALL ZERO: delta = scale * B@A = 0 EXACTLY, so dropping them changes
  ## the result by not a single bit, it only saves work (prefill: 2 LoRA GEMMs/Linear; one-dispatch decode:
  ## the kernel skips that layer's LoRA stage via a flag in the meta table). Returns the dropped names.
  ## (Adapters initialised with B=0 that get no gradient during training end up here.)
  for name, ad in lo.adapters:
    var zeroA = true
    for v in ad.aValues:
      if v != 0'f32:
        zeroA = false
        break
    var zeroB = true
    for v in ad.bValues:
      if v != 0'f32:
        zeroB = false
        break
    if zeroA or zeroB: result.add name
  for name in result: lo.adapters.del(name)

proc mergeLoraIntoQuantBase*(lo: var JambaLora, verbose = true): int =
  ## SAFE MERGE: never drops an adapter before the merged tensor is verified.
  ## W_ref = dequant(base) + scale*B@A, W_test = dequant(requant(W_ref)).
  ## The relative error is checked against the LoRA delta ITSELF (not against |W_base|:
  ## LoRA is ~0.1% of |W|, comparing to |W| would hide losing the whole delta).
  ##
  ## By default FP4 / INT4 requant is NOT attempted (the quantization step is larger than LoRA,
  ## and trying costs minutes + RAM). Set BYBY_MERGE_LORA_TRY_FP4=1 to
  ## TEST IT WITH THE GUARD. If it fails, base + adapter are KEPT intact,
  ## the forward still fuses LoRA in one-dispatch; the .bin checkpoint is not damaged.
  ##
  ## NOTE: there is no general "lossless merge into 4-bit" for an arbitrary
  ## fp32 LoRA. Dense fp16/fp32 needs another backend and more RAM.
  const maxRelativeDeltaError = 0.001'f64  # 0.1% RMS error of the delta
  const maxRelativePeakError = 0.01'f64    # 1% of the peak delta
  let tryLowBit = getEnv("BYBY_MERGE_LORA_TRY_FP4", "0") == "1"
  var names: seq[string] = @[]
  for name in lo.adapters.keys: names.add(name)
  var merged = 0
  var skippedLowBit = 0
  var rejected = 0
  var withoutBase = 0
  for name in names:
    if name notin lo.quantBase:
      inc withoutBase
      continue
    let base = lo.quantBase[name]
    if base.shape.len != 2: continue
    let ad = lo.adapters[name]
    let outF = base.shape[0]
    let inF = base.shape[1]
    if ad.inF != inF or ad.outF != outF or ad.r <= 0 or
       ad.aValues.len != ad.r * inF or ad.bValues.len != outF * ad.r:
      raise newException(ValueError, "mergeLoraIntoQuantBase: adapter shape does not match the base: " & name)
    # Preflight only on low-precision tensors, do NOT allocate a full-size
    # ~128MB dequant or requantize hundreds of millions of elements for nothing.
    if base.kind in {qkFp4KM, qkInt4, qkInt4Asymmetric, qkCustom} and not tryLowBit:
      inc skippedLowBit
      continue
    let budgetMiB = (try: parseInt(getEnv("BYBY_MERGE_SCRATCH_MIB", "256")) except ValueError: 256)
    # Two dense FP32 matrices plus quant candidate and row scratch. Division
    # avoids overflow; skip preserves the exact fused adapter on GPU.
    let elements = int64(outF) * int64(inF)
    if budgetMiB <= 0 or elements > int64(budgetMiB) * 1024 * 1024 div 32:
      inc rejected
      if verbose: echo "[lora-merge] scratch budget exceeded; keep fused adapter: " & name
      continue
    # INT8/FP8 can also lose the delta; they must pass the guard like FP4.
    var w = dequantViaLUT(base)
    if w.len != outF * inF:
      raise newException(ValueError, "mergeLoraIntoQuantBase: dequant has the wrong size: " & name)
    var deltaRow = newSeq[float32](inF)
    var deltaNormSq = 0.0
    var maxDelta = 0.0
    # Track FP32 weight-add rounding too: a tiny LoRA delta can disappear
    # when added to W even BEFORE requantizing to FP4.
    var addRoundSq = 0.0
    var maxAddRound = 0.0
    for o in 0 ..< outF:
      for i in 0 ..< inF: deltaRow[i] = 0'f32
      for r in 0 ..< ad.r:
        let b = ad.bValues[o * ad.r + r] * ad.scale
        if b == 0'f32: continue
        let ar = r * inF
        for i in 0 ..< inF:
          deltaRow[i] += b * ad.aValues[ar + i]
      let row = o * inF
      for i in 0 ..< inF:
        let d = deltaRow[i]
        deltaNormSq += float64(d) * float64(d)
        if abs(float64(d)) > maxDelta: maxDelta = abs(float64(d))
        let oldW = w[row + i]
        let afterAdd = oldW + d
        let addErr = abs(float64(afterAdd) - (float64(oldW) + float64(d)))
        addRoundSq += addErr * addErr
        if addErr > maxAddRound: maxAddRound = addErr
        w[row + i] = afterAdd
    # If the total delta = 0, KEEP the adapter: B@A may cancel out,
    # only provably safe zero A/B are dropped, in a separate proc.
    if deltaNormSq == 0.0:
      inc rejected
      continue
    # Preserve groupSize; quantizeTensor() with qkFp4KM defaults to 32,
    # and must not silently change the layout when the base has another groupSize.
    let candidate =
      if base.kind == qkFp4KM:
        quantizeFp4KM(w, base.shape, groupSize =
          (if base.groupSize > 0: base.groupSize else: 32))
      else:
        quantizeTensor(w, base.shape, base.kind, customCf = base.cf)
    # Check the REAL result after quantization, not before the requant.
    let roundtrip = dequantViaLUT(candidate)
    if roundtrip.len != w.len:
      raise newException(ValueError, "mergeLoraIntoQuantBase: roundtrip shape mismatch: " & name)
    var errSq = 0.0
    var maxErr = 0.0
    var valid = true
    for i in 0 ..< w.len:
      let got = roundtrip[i]
      let expected = w[i]
      if got != got or expected != expected or
         abs(got) > float32.high or abs(expected) > float32.high:
        valid = false
        break
      let e = abs(float64(got) - float64(expected))
      errSq += e * e
      if e > maxErr: maxErr = e
    # Triangle-inequality upper bound vs EXACT real-valued W_base + delta.
    # Must not compare only against float32 w: a small d can be lost in the FP32 add.
    let rel = if valid: (sqrt(errSq) + sqrt(addRoundSq)) / sqrt(deltaNormSq) else: Inf
    if valid and rel <= maxRelativeDeltaError and
       maxErr + maxAddRound <= maxRelativePeakError * maxDelta:
      # COMMIT ONLY AFTER THE VALIDATION; on failure both base and adapter
      # are bitwise unchanged, so lossless fused LoRA remains available.
      lo.quantBase[name] = candidate
      lo.adapters.del(name)
      inc merged
      if verbose: echo &"[lora-merge] ACCEPT {name}: relErr/delta={rel:.6f}"
    else:
      inc rejected
      if verbose: echo &"[lora-merge] REJECT {name}: relErr/delta={rel:.6f} " &
        "(base + LoRA KEPT; adapter not dropped)"
  if verbose:
    echo &"[lora-merge] SAFE: {merged} adapter merged, {skippedLowBit} low-bit skip, " &
      &"{rejected} rejected, {withoutBase} without base; {lo.adapters.len} adapters left."
    if skippedLowBit + rejected > 0:
      echo "[lora-merge] FP4-km cannot merge an fp32 LoRA exactly at 4-bit: " &
           "keeping the packed base + LoRA fused on the GPU to avoid wrong tokens."
  result = merged

proc mergeInOne*(m: var NimformerModelJamba, lora: var JambaLora, ctx: Backend, name: string) =
  ## SCOPED version of mergeIn - handles a SINGLE parameter (name). Used for
  ## streaming training (see trainStepGraftStreaming in finetune_lora.nim):
  ## call mergeInOne right BEFORE the weight is needed (for the current block),
  ## restoreBaseOne RIGHT AFTER it is used - never more than 1-2 quant-resident
  ## blocks in fp32 in RAM at once (unlike the original mergeIn/restoreBase,
  ## which handle ALL adapters at once - dequantizing all 52 blocks together, defeating
  ## streaming if used here by mistake).
  var byName = initTable[string, ptr Tensor]()
  for (nm, t) in paramList(m):
    if nm == name: byName[nm] = t
  if name notin byName: return
  let t = byName[name]
  let ad = lora.adapters[name]
  if name in lora.quantBase:
    t[].data = dequantViaLUT(lora.quantBase[name])
  else:
    lora.backup[name] = t[].data
  let merged = mergedWeight(t[], ad, ctx)
  t[].data = merged.data
  if name in lora.quantBase:
    # BUG FIX (see paramQuantFlags in jamba_hybrid.nim): t[].data is now
    # REAL fp32 (base+LoRA), no longer compressed - useQuant is turned off so
    # Linear.forward()/uploadResident() use t[].data DIRECTLY instead of wrongly
    # dequantizing from the original weightQ (which would silently lose the LoRA delta).
    let qFlags = paramQuantFlags(m)
    if name in qFlags: qFlags[name][] = false

proc restoreBaseOne*(m: var NimformerModelJamba, lora: JambaLora, name: string) =
  var byName = initTable[string, ptr Tensor]()
  for (nm, t) in paramList(m):
    if nm == name: byName[nm] = t
  if name notin byName: return
  if name in lora.quantBase:
    byName[name][].data = @[]   # giai phong ngay
    # BUG FIX: restore useQuant=true - the next mergeInOne/uploadResident
    # dequantizes CORRECTLY from weightQ again (t[].data is empty).
    let qFlags = paramQuantFlags(m)
    if name in qFlags: qFlags[name][] = true
  else:
    byName[name][].data = lora.backup[name]

proc bindQuantBaseToLinears*(m: var NimformerModelJamba, lora: var JambaLora) =
  ## Fix for the IndexDefect "container is empty" on the quantResident + NIMBIN4 path:
  ##
  ## newJambaBlock(quantResident=true) deliberately clears weight.data = @[] for
  ## inProj/outProj (avoiding a ~29GB fp32 allocation). The real QuantTensors live in
  ## lora.quantBase (from loadCodestralQuantResidentFromNimq + the Q4 newBlock).
  ## But NOBODY called setQuantWeight → Linear.useQuant stayed false,
  ## weightQ empty → uploadResident uploaded nothing → forward() fell
  ## through to transpose(l.weight) with empty data → crash.
  ##
  ## Call this proc RIGHT AFTER assigning lo.quantBase, BEFORE mergeInUploadStreamingJamba
  ## / generate. Binds every quantBase entry to the matching Linear (R/A/M/S + outProj).
  ## useQuant=true, weightQ stays compressed; LoRA A/B are attached separately afterwards in
  ## mergeInUploadStreamingJamba as in the QLoRA design (no physical merge).
  let linears = paramLinears(m)
  # (REAL BUG): quantize_lora_bin_4bit.convert() compresses EVERY "newBlock::*" tensor,
  # including small 1-D ones (LN gamma/beta, A block biases). These 96 tensors
  # (12 A blocks x 8) used to be SKIPPED here ("96 names not mapped to a Linear") => A blocks
  # ran with LN gamma=1/beta=0/bias=0 instead of the TRAINED values. They are now decompressed
  # into the model's own Tensors.
  var byName = initTable[string, ptr Tensor]()
  for (pname, pt) in paramList(m): byName[pname] = pt
  var bound = 0
  var restoredSmall = 0
  var missing = 0
  var missingNames: seq[string] = @[]
  for name, qt in lora.quantBase.pairs:
    # Guard: skip empty QuantTensors (shape.len == 0) - possibly a checkpoint
    # that is not fully written (e.g. fuseGate/attn of a freshly initialised, untrained G block).
    if qt.shape.len == 0:
      stderr.writeLine "[lora] bindQuantBaseToLinears: SKIPPING '" & name & "' (empty QuantTensor, shape.len=0)"
      continue
    if name in linears:
      setQuantWeight(linears[name][], qt)
      inc bound
    elif name in byName:
      let dq = dequantizeTensor(qt)
      let t = byName[name]
      var want = 1
      for d in t[].shape: want *= d
      if dq.len == want:
        t[].data = dq
        inc restoredSmall
      else:
        inc missing
        missingNames.add(name & " (dequant=" & $dq.len & " != model=" & $want & ")")
    else:
      # The name maps to no model parameter — skipped, no crash.
      inc missing
      missingNames.add(name)
  echo &"[lora] bindQuantBaseToLinears: {bound} Linears useQuant=true (weightQ from quantBase), " &
       &"{restoredSmall} small parameters (LN gamma/beta, bias) decompressed back into the model, " &
       &"{missing} names not mapped (skipped)."
  for nm in missingNames:
    stderr.writeLine "[lora]   Could NOT restore: " & nm

proc mergeInUploadStreamingJamba*(m: var NimformerModelJamba, lora: var JambaLora, ctx: Backend) =
  ## Attaches LoRA SEPARATELY (NO physical merge) + uploads to the GPU for all 105
  ## quant-resident parameters of the model, used by the inference path (generate()) when
  ## the pattern has non-'R' blocks.
  ##
  ## HISTORY (why mergeIn()/mergeInOne() are no longer called here): the
  ## previous version called mergeInOne (dequant weightQ through the LUT + mergedWeight
  ## = W_base + scale*B@A into a FULL fp32 matrix) then uploaded+freed each
  ## parameter in turn. That reduced the peak RAM of the merge step ALONE (from O(105) parameters
  ## at once down to O(1)), but it was STILL Killed - because the physical merge itself
  ## FORCES the weight out of the compressed Q4 form (~0.5 byte/element) into
  ## fp32 (~4 bytes/element) FOR UPLOAD - for the 105 large parameters of the 7B model
  ## that is ~3.5GB (compressed) -> ~28GB (fp32), REGARDLESS of freeing right after,
  ## because when the GPU/Metal allocates a resident fp32 buffer of that size
  ## RAM is already over the limit. The original comment on the `weightQ` field
  ## of Linear (nimformer.nim) already warned about exactly this: "NEVER
  ## expanded to fp32 all at once and kept (that was the earlier
  ## OOM: 6.7B params x 4 bytes fp32 = ~27GB RAM)" - a physical merge goes
  ## against the original design of this very type.
  ##
  ## NEW APPROACH (standard QLoRA inference, NO merge): the base
  ## weight (`weightQ`/`useQuant`) stays INTACT and compressed forever - NO
  ## dequant, NO merge, NO useQuant change. Only `loraA`/`loraB`/
  ## `loraR`/`loraScale` (very small: [r,inF]+[outF,r] with r~4-64, instead of
  ## the full [outF,inF]) are attached directly to the Linear. The GPU inference path must call
  ## prepareGpuLinForLinear(): the FP4-KM base is uploaded packed once into the
  ## Shared GPU cache, and the small fp32 A/B are uploaded privately once and fused
  ## in the same session. Linear.forward() must never fall back automatically to
  ## a CPU dequant for FP4-KM.
  # (debugging aid: print which blocks get LoRA - 'r' or 'a'):
  var attachedCount = 0
  var fullQuantCount = 0   # full quant weights (useQuant) - NOT LoRA
  var fullFp32Count = 0
  var missingCount = 0
  # No nested proc capturing `lora` (ARC: cannot capture JambaLora). Only reads the Linear.
  template logWeightSource(name: string, lin: Linear) =
    if lin.useQuant:
      inc fullQuantCount
      echo "[lora]     " & name & ": NO LoRA adapter — FULL quant weight (newBlock/base), useQuant=true"
    elif lin.weight.data.len > 0:
      inc fullFp32Count
      echo "[lora]     " & name & ": NO LoRA adapter — FULL fp32 weight on the Linear"
    else:
      inc missingCount
      echo "[lora]     " & name & ": MISSING weight (no LoRA, no useQuant, no fp32)"

  # Fix for Killed:9 / OOM on integrated GPUs (Iris unified memory):
  # a Q4 uploadResident for ~100+ LARGE Linears = holding the CPU weightQ copy +
  # the GPU resident buffer at the same time (both in system RAM) -> the peak exceeds the limit, the kernel kills it.
  # No more resident upload: the resident path creates a separate fp32/Q4 copy per
  # Linear and is not the fused FP4-KM kernel. Packed weights are only cached
  # on the GPU in prepareGpuLinForLinear(), then encodeLinByName() uses them
  # directly in the inference session.
  let doUploadResident = false
  echo &"[lora] Starting LoRA attach, no resident upload; GPU packed cache for {m.blocks.len} blocks:"
  echo "[lora]   (Note: A blocks are trained in FULL — their weights are in newBlock::* in jamba_lora.bin, not in adapters.)"
  for i in 0 ..< m.blocks.len:
    let p = &"block{i}"
    let kindLetter = case m.blocks[i].kind
      of bkMambaReal: "R"
      of bkMambaRealPlusAttn: "P"
      of bkMambaRealGated: "G"
      of bkMamba: "M"
      of bkAttn: "A"
      of bkSpectron: "S"
    echo &"[lora]   block{i} = '{kindLetter}'" &
         (if m.blocks[i].kind == bkMambaRealPlusAttn: " (R + Linear++ sidecar)" else: "")
    case m.blocks[i].kind
    of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
      block:
        let name = p & ".mambaReal.inProj.weight"
        if name in lora.adapters:
          # NO physical merge (mergeInOne) any more - see Linear.loraA/loraB and
          # forward() in nimformer.nim: the adapter is attached SEPARATELY, the base (weightQ/
          # useQuant) stays INTACT and compressed, never dequantized/merged
          # to fp32 - avoiding the ~27GB bloat that caused OOM.
          let ad = lora.adapters[name]
          attachAdapter(m.blocks[i].mambaReal.inProj, ad)
          m.blocks[i].mambaReal.inProj.loraR = ad.r
          m.blocks[i].mambaReal.inProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mambaReal.inProj)
        if doUploadResident:

          uploadResident(m.blocks[i].mambaReal.inProj, ctx)
      block:
        let name = p & ".mambaReal.outProj.weight"
        if name in lora.adapters:
          # NO physical merge (mergeInOne) any more - see Linear.loraA/loraB and
          # forward() in nimformer.nim: the adapter is attached SEPARATELY, the base (weightQ/
          # useQuant) stays INTACT and compressed, never dequantized/merged
          # to fp32 - avoiding the ~27GB bloat that caused OOM.
          let ad = lora.adapters[name]
          attachAdapter(m.blocks[i].mambaReal.outProj, ad)
          m.blocks[i].mambaReal.outProj.loraR = ad.r
          m.blocks[i].mambaReal.outProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mambaReal.outProj)
        if doUploadResident:
          uploadResident(m.blocks[i].mambaReal.outProj, ctx)
      if m.blocks[i].kind == bkMambaRealPlusAttn:
        block:
          let name = p & ".attn.qkvProj.weight"
          if name in lora.adapters:
            let ad = lora.adapters[name]
            attachAdapter(m.blocks[i].attn.qkvProj, ad)
            m.blocks[i].attn.qkvProj.loraR = ad.r
            m.blocks[i].attn.qkvProj.loraScale = ad.scale
            echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
            inc attachedCount
          else:
            logWeightSource(name, m.blocks[i].attn.qkvProj)
          if doUploadResident:
            uploadResident(m.blocks[i].attn.qkvProj, ctx)
        block:
          let name = p & ".attn.outProj.weight"
          if name in lora.adapters:
            let ad = lora.adapters[name]
            attachAdapter(m.blocks[i].attn.outProj, ad)
            m.blocks[i].attn.outProj.loraR = ad.r
            m.blocks[i].attn.outProj.loraScale = ad.scale
            echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
            inc attachedCount
          else:
            logWeightSource(name, m.blocks[i].attn.outProj)
          if doUploadResident:
            uploadResident(m.blocks[i].attn.outProj, ctx)
    of bkMamba:
      block:
        let name = p & ".mamba.inProj.weight"
        if name in lora.adapters:
          # NO physical merge (mergeInOne) any more - see Linear.loraA/loraB and
          # forward() in nimformer.nim: the adapter is attached SEPARATELY, the base (weightQ/
          # useQuant) stays INTACT and compressed, never dequantized/merged
          # to fp32 - avoiding the ~27GB bloat that caused OOM.
          let ad = lora.adapters[name]
          attachAdapter(m.blocks[i].mamba.inProj, ad)
          m.blocks[i].mamba.inProj.loraR = ad.r
          m.blocks[i].mamba.inProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mamba.inProj)
        if doUploadResident:

          uploadResident(m.blocks[i].mamba.inProj, ctx)
      block:
        let name = p & ".mamba.outProj.weight"
        if name in lora.adapters:
          # NO physical merge (mergeInOne) any more - see Linear.loraA/loraB and
          # forward() in nimformer.nim: the adapter is attached SEPARATELY, the base (weightQ/
          # useQuant) stays INTACT and compressed, never dequantized/merged
          # to fp32 - avoiding the ~27GB bloat that caused OOM.
          let ad = lora.adapters[name]
          attachAdapter(m.blocks[i].mamba.outProj, ad)
          m.blocks[i].mamba.outProj.loraR = ad.r
          m.blocks[i].mamba.outProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mamba.outProj)
        if doUploadResident:

          uploadResident(m.blocks[i].mamba.outProj, ctx)
    of bkAttn:
      block:
        let name = p & ".attn.qkvProj.weight"
        if name in lora.adapters:
          # NO physical merge (mergeInOne) any more - see Linear.loraA/loraB and
          # forward() in nimformer.nim: the adapter is attached SEPARATELY, the base (weightQ/
          # useQuant) stays INTACT and compressed, never dequantized/merged
          # to fp32 - avoiding the ~27GB bloat that caused OOM.
          let ad = lora.adapters[name]
          attachAdapter(m.blocks[i].attn.qkvProj, ad)
          m.blocks[i].attn.qkvProj.loraR = ad.r
          m.blocks[i].attn.qkvProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].attn.qkvProj)
        if doUploadResident:

          uploadResident(m.blocks[i].attn.qkvProj, ctx)
      block:
        let name = p & ".attn.outProj.weight"
        if name in lora.adapters:
          # NO physical merge (mergeInOne) any more - see Linear.loraA/loraB and
          # forward() in nimformer.nim: the adapter is attached SEPARATELY, the base (weightQ/
          # useQuant) stays INTACT and compressed, never dequantized/merged
          # to fp32 - avoiding the ~27GB bloat that caused OOM.
          let ad = lora.adapters[name]
          attachAdapter(m.blocks[i].attn.outProj, ad)
          m.blocks[i].attn.outProj.loraR = ad.r
          m.blocks[i].attn.outProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].attn.outProj)
        if doUploadResident:

          uploadResident(m.blocks[i].attn.outProj, ctx)
    of bkSpectron:
      block:
        let name = p & ".mamba.inProj.weight"
        if name in lora.adapters:
          # NO physical merge (mergeInOne) any more - see Linear.loraA/loraB and
          # forward() in nimformer.nim: the adapter is attached SEPARATELY, the base (weightQ/
          # useQuant) stays INTACT and compressed, never dequantized/merged
          # to fp32 - avoiding the ~27GB bloat that caused OOM.
          let ad = lora.adapters[name]
          attachAdapter(m.blocks[i].mamba.inProj, ad)
          m.blocks[i].mamba.inProj.loraR = ad.r
          m.blocks[i].mamba.inProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mamba.inProj)
        if doUploadResident:

          uploadResident(m.blocks[i].mamba.inProj, ctx)
      block:
        let name = p & ".mamba.outProj.weight"
        if name in lora.adapters:
          # NO physical merge (mergeInOne) any more - see Linear.loraA/loraB and
          # forward() in nimformer.nim: the adapter is attached SEPARATELY, the base (weightQ/
          # useQuant) stays INTACT and compressed, never dequantized/merged
          # to fp32 - avoiding the ~27GB bloat that caused OOM.
          let ad = lora.adapters[name]
          attachAdapter(m.blocks[i].mamba.outProj, ad)
          m.blocks[i].mamba.outProj.loraR = ad.r
          m.blocks[i].mamba.outProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mamba.outProj)
        if doUploadResident:

          uploadResident(m.blocks[i].mamba.outProj, ctx)
      block:
        let name = p & ".attn.qkvProj.weight"
        if name in lora.adapters:
          # NO physical merge (mergeInOne) any more - see Linear.loraA/loraB and
          # forward() in nimformer.nim: the adapter is attached SEPARATELY, the base (weightQ/
          # useQuant) stays INTACT and compressed, never dequantized/merged
          # to fp32 - avoiding the ~27GB bloat that caused OOM.
          let ad = lora.adapters[name]
          attachAdapter(m.blocks[i].attn.qkvProj, ad)
          m.blocks[i].attn.qkvProj.loraR = ad.r
          m.blocks[i].attn.qkvProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].attn.qkvProj)
        if doUploadResident:

          uploadResident(m.blocks[i].attn.qkvProj, ctx)
      block:
        let name = p & ".attn.outProj.weight"
        if name in lora.adapters:
          # NO physical merge (mergeInOne) any more - see Linear.loraA/loraB and
          # forward() in nimformer.nim: the adapter is attached SEPARATELY, the base (weightQ/
          # useQuant) stays INTACT and compressed, never dequantized/merged
          # to fp32 - avoiding the ~27GB bloat that caused OOM.
          let ad = lora.adapters[name]
          attachAdapter(m.blocks[i].attn.outProj, ad)
          m.blocks[i].attn.outProj.loraR = ad.r
          m.blocks[i].attn.outProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].attn.outProj)
        if doUploadResident:

          uploadResident(m.blocks[i].attn.outProj, ctx)
      block:
        let name = p & ".fuseGate.weight"
        if name in lora.adapters:
          # NO physical merge (mergeInOne) any more - see Linear.loraA/loraB and
          # forward() in nimformer.nim: the adapter is attached SEPARATELY, the base (weightQ/
          # useQuant) stays INTACT and compressed, never dequantized/merged
          # to fp32 - avoiding the ~27GB bloat that caused OOM.
          let ad = lora.adapters[name]
          attachAdapter(m.blocks[i].fuseGate, ad)
          m.blocks[i].fuseGate.loraR = ad.r
          m.blocks[i].fuseGate.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].fuseGate)
        if doUploadResident:

          uploadResident(m.blocks[i].fuseGate, ctx)
  block:
    let name = "outProj.weight"
    if name in lora.adapters:
      # By default NO LoRA on lm_head: checkpoints with 105 adapters (outProj.weight)
      # do not match the current trainer (loraTargets is R only). Enable: BYBY_APPLY_LMHEAD_LORA=1.
      if getEnv("BYBY_APPLY_LMHEAD_LORA") == "1":
        let ad = lora.adapters[name]
        attachAdapter(m.outProj, ad)
        m.outProj.loraR = ad.r
        m.outProj.loraScale = ad.scale
        echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale & " (BYBY_APPLY_LMHEAD_LORA=1)"
        inc attachedCount
      else:
        echo "[lora]     " & name & ": SKIPPING lm_head LoRA (default). Enable: BYBY_APPLY_LMHEAD_LORA=1"
        inc missingCount
    else:
      logWeightSource(name, m.outProj)
    if doUploadResident:

      uploadResident(m.outProj, ctx)
  echo &"[lora] Tong ket: {attachedCount} LoRA adapter | {fullQuantCount} FULL quant (newBlock/base, vd Attention) | {fullFp32Count} FULL fp32 | {missingCount} THIEU weight."
  echo "[lora]   → Attention (A blocks) is in the FULL quant/fp32 column, NOT 'missing adapters'."
  echo "[lora]   → uploadResident SKIPPED. The FP4-KM base + LoRA are encoded on the GPU in the session."

proc mergeIn*(m: var NimformerModelJamba, lora: var JambaLora, ctx: Backend) =
  ## Before forward: overwrites the base weight with W_eff = base + LoRA delta.
  ## If the parameter is in lora.quantBase (quant-resident, no resident
  ## fp32 copy) -> dequantize FRESH through the LUT right here (instead of backing up
  ## the old t[].data, which is garbage/unusable) then add the LoRA
  ## delta - NO backup is kept (restoreBase FREES instead of restoring, see below).
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  let qFlags = paramQuantFlags(m)
  for name, ad in lora.adapters:
    let t = byName[name]
    if name in lora.quantBase:
      t[].data = dequantViaLUT(lora.quantBase[name])   # a "fresh" base every time, no backup
    else:
      lora.backup[name] = t[].data
    let merged = mergedWeight(t[], ad, ctx)
    t[].data = merged.data
    if name in lora.quantBase and name in qFlags:
      # BUG FIX: see the full explanation in mergeInOne/paramQuantFlags
      # above - without turning this off, uploadResidentJamba() called right after
      # mergeIn() would dequantize again from the ORIGINAL weightQ, wiping the LoRA just merged.
      qFlags[name][] = false

proc restoreBase*(m: var NimformerModelJamba, lora: JambaLora) =
  ## After backward: restores the base to the original.
  ## For quant-resident parameters there is NO backup to restore from -
  ## t[].data is FREED directly (releasing RAM right away - this is the KEY point of
  ## the RAM fix: never more than 1-2 fp32 blocks resident at once).
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  let qFlags = paramQuantFlags(m)
  for name in lora.adapters.keys:
    if name in lora.quantBase:
      byName[name][].data = @[]   # freed - the next mergeIn dequantizes again from scratch
      if name in qFlags: qFlags[name][] = true   # BUG FIX: see mergeIn above
    else:
      byName[name][].data = lora.backup[name]

proc trainStepLora*(m: var NimformerModelJamba, lora: var JambaLora,
                     state: var LoraTrainState,
                     idsBatch, targetsBatch: seq[seq[int]], ctx: Backend,
                     requantizeEvery: int = 0): float32 =
  ## 1 training step that ONLY updates the LoRA A,B - every other parameter (the previously
  ## fine-tuned Mamba base, embed, outProj, LayerNorm...) stays UNCHANGED. This is
  ## what lets weak machines run it: the number of parameters actually updated = a few
  ## tens of thousands (r * (inF+outF) per adapter) instead of the whole model.
  mergeIn(m, lora, ctx)
  let (loss, grads) = computeLossAndGrads(m, idsBatch, targetsBatch, ctx,
                                           trainLmHead = false, trainEmbed = false)
  for name, ad in lora.adapters.mpairs:
    if name notin grads:
      stderr.writeLine("[lora] WARNING: missing grad for " & name)
      continue
    let (dA, dB) = gradToAB(grads[name], ad, ctx)
    var stA = state.adamA[name]
    var stB = state.adamB[name]
    # requantizeEvery=0 means NO APF/CustomFloat snapping for A/B (low-rank LoRA,
    # small values, snapping to APF early kills the gradient, exactly as warned
    # in the original trainStep of test_nimformer.nim) - apfAdamStep treats
    # requantizeEvery<=0 as "never requantize", plain fp32 Adam only.
    discard apfAdamStepFused(ctx, ad.A, dA, stA, state.lr,
                            requantizeEvery = (if requantizeEvery <= 0: int.high else: requantizeEvery))
    discard apfAdamStepFused(ctx, ad.B, dB, stB, state.lr,
                            requantizeEvery = (if requantizeEvery <= 0: int.high else: requantizeEvery))
    state.adamA[name] = stA
    state.adamB[name] = stB
  restoreBase(m, lora)
  result = loss

# MMMM...") -> a new model with LinearPlusAttention/Spectron blocks
# inserted, keeping the trained Mamba weights.
# ═══════════════════════════════════════════════════════════════
proc graftBlocks*(old: var NimformerModelJamba, newPattern: string, ctx: Backend): NimformerModelJamba =
  ## `newPattern` must contain the 'M' characters in the same COUNT and
  ## relative ORDER as `old.pattern` (the 'M's are matched in sequence to copy
  ## weights), plus new 'A'/'S' characters inserted anywhere. Example:
  ##   old.pattern  = "MMMM"
  ##   newPattern   = "MMAMMSM"   # 4 'M's in the same order, plus 1 'A' + 1 'S'
  ## The 'M' blocks in newPattern get their weights copied as is (ln1, ln2,
  ## ff, mamba.*) from old in order. The new 'A'/'S' blocks are fully
  ## random-init - they MUST be trained (no LoRA, since there is nothing to "low-rank
  ## approximate" - these blocks are trained in full, AND LoRA can be attached to the
  ## copied 'M' blocks to lightly fine-tune the Mamba part to fit the new
  ## Attention, see attachLora above).
  var oldMIdx = 0
  var mPositionsInNew: seq[int] = @[]
  for i, c in newPattern:
    if c in {'M', 'm'}: mPositionsInNew.add(i)
  let oldMCount = old.pattern.count('M') + old.pattern.count('m')
  assert mPositionsInNew.len == oldMCount,
    &"The number of 'M' blocks in newPattern ({mPositionsInNew.len}) must match old.pattern ({oldMCount})"

  result = newNimformerModelJamba(old.vocab, old.embedDim, old.nHeads, old.ffMult,
                                   newPattern, useSession = true)
  result.embed.weight = old.embed.weight
  result.outProj.weight = old.outProj.weight
  result.outProj.bias = old.outProj.bias

  for newIdx in mPositionsInNew:
    let oi = oldMIdx
    inc oldMIdx
    result.blocks[newIdx].ln1 = old.blocks[oi].ln1
    result.blocks[newIdx].ln2 = old.blocks[oi].ln2
    result.blocks[newIdx].ff = old.blocks[oi].ff
    result.blocks[newIdx].mamba = old.blocks[oi].mamba
  echo &"[graft] Copied {oldMIdx} Mamba blocks from the old model -> new model with pattern '{newPattern}' " &
       &"({newPattern.len - oldMIdx} new (A/S) blocks random-init, need training)"
# ═══════════════════════════════════════════════════════════════
# INFER — 100% GPU, 1 command buffer / token, NO MSL embedded in Nim.
#
# Every kernel (fp4-KM / int4-asymmetric / fp32 gemv, LoRA, bias+APFLU,
# Linear++ step, split/conv/softplus, SSM step, grouped RMSNorm) is
# micro-VM BYTECODE (session_vm.nim + fp4_gemv_vm.nim), lowered + JIT-compiled
# by jit_vm.nim (in the HotSpot/tinygrad MetalCompiler spirit noted at the top of
# jit_vm.nim: bytecode -> Metal source GENERATED at runtime -> newLibraryWithSource
# -> pipeline cache). NO .metal kernel is hand-written in Nim.
#
# R (Mamba2Real): mamba2_fused_step.nim (encodeLayerV2, 7 commands/layer — kept
#   from the previous version, measured ~30 tok/s before being slowed down by
#   commitContinue/yield; fp4GemvAt now also covers int4-asymmetric).
# A (LinearPlusAttention + FFN): gpu_layers.nim (encodeAttnLayer) — NEW,
#   fully replaces the old forwardLinearPlusInferStep branch (which called a function with the wrong signature — see
#   the nim check error before the fix: nHeads passed in qkvName's place) with
#   a pure GPU path, in the same session as R.
#
# Every Linear (R in/out_proj, A qkv/out/fc1/fc2, lm_head) goes through EXACTLY 1
# preparation function: gpu_layers.prepareGpuLinForLinear — it picks the kernel from
# lin.weightQ.kind (fp4-KM or int4-asymmetric) or an fp32 fallback, and frees
# the duplicate host copy (the Shared/Private GPU buffer holds 1 copy) after upload.
# ═══════════════════════════════════════════════════════════════

import session_vm_dispatch
import gpu_layers
import vendor/bybylang/gpubackend as byby

var gInferStepCount = 0
var gInferVerbose = existsEnv("BYBY_INFER_VERBOSE")
var gGpuGenStatePoisoned = false

proc ilog(msg: string) =
  if not gInferVerbose: return
  stderr.writeLine(msg)
  stderr.flushFile()

proc gNormGroups*(nGroups: int): int =
  ## Group count for the gated RMSNorm in the R mixer (BAO_CAO.md section 3.2):
  ##   default = the model's n_groups  -> uses mamba_ssm / vLLM
  ##   BYBY_MAMBA_NORM_GROUPS=1       -> diagnostic legacy/full-width variant
  let raw = getEnv("BYBY_MAMBA_NORM_GROUPS", "")
  if raw.len > 0:
    try: return max(1, parseInt(raw))
    except ValueError: discard
  nGroups

var gHXPersist, gHNormPersist, gHFinalPersist, gHLogitsPersist: SessionHandle
var gNormFGamma: SessionHandle      ## normF gamma (== HF backbone.norm_f, RMSNorm, NO beta)
var gNormFOnes: SessionHandle       ## gamma=1 (only used with the "-unit" preset: see setGpuNormPreset)
var gLmHead: GpuLin
var gGpuModelReady = false
var gGpuWarmedUp = false

# ── Norm configuration of the GPU infer path ─────────────────────────────────
# A REAL mismatch between train and infer (from reading the code):
#   TRAIN (jamba_hybrid.forwardBlock/forwardBatch + finetune_lora): the R block's ln1
#          = LayerNorm (mean subtraction + beta), NO normF before lm_head.
#   Original HF Mamba2: RMSNorm for ln1 and normF = RMSNorm(norm_f) before lm_head.
# LoRA checkpoints + A blocks only match the training configuration => preset "train".
var gUseNormF* = true
proc setGpuDisableLora*(v: bool) = gDisableLora = v
var gSkipAttn* = false
  ## diagnostics (bisect): true = skip EVERY A block (the residual passes straight through) -> only Mamba R remains.
var gNormPresetName* = "hf"

proc setGpuNormPreset*(name: string): bool =
  ## Preset = "<norm>-<final>-<groups>" (case-insensitive):
  ##   norm   : ln  = LayerNorm (mean subtraction + beta)   | rms = RMSNorm
  ##   final  : nof = NO normF before lm_head     | fin = with normF (RMSNorm)
  ##   groups : g1  = full-width gated-RMSNorm | gn = prepared model groups
  ## Aliases:
  ##   "train" = rms-fin-gn  (current trainer: Mamba-SSM gated-RMSNorm groups)
  ##   "train-legacy" = ln-nof-g1 (the old LayerNorm-ln1 era trainer, no normF)
  ##   "hf"    = rms-fin-g1   (legacy Transformers implementation: full-width reduction)
  ##   "mamba-ssm" = rms-fin-gn (grouped RMSNorm used by mamba_ssm implementations)
  ## Optional "-unit" suffix (e.g. "rms-fin-g1-unit", "hf-unit"): gamma of the pre-block
  ## RMSNorm (R + P sidecar) and normF = 1. ONLY for reading checkpoints trained with the OLD
  ## trainer - that trainer never loaded the pretrain gamma into rms1/rmsF (see
  ## jamba_hybrid.syncRmsFromBaseNorms), so LoRA + sidecar learned on gamma=1 inputs.
  var spec = name.strip().toLowerAscii()
  var unit = false
  if spec.endsWith("-unit"):
    unit = true
    spec = spec[0 ..< spec.len - 5]
  if spec == "train": spec = "rms-fin-gn"
  elif spec == "train-legacy": spec = "ln-nof-g1"
  elif spec == "hf": spec = "rms-fin-g1"
  elif spec == "mamba-ssm": spec = "rms-fin-gn"
  let parts = spec.split('-')
  if parts.len != 3: return false
  if parts[0] notin ["ln", "rms"]: return false
  if parts[1] notin ["nof", "fin"]: return false
  if parts[2] notin ["g1", "gn"]: return false
  if unit and parts[0] != "rms": return false   # "-unit" only makes sense for RMSNorm
  gRNormOverride = (if parts[0] == "ln": 1 else: 0)
  gUseNormF = (parts[1] == "fin")
  gRNormGroupsOverride = (if parts[2] == "g1": 1 else: -1)
  gRmsUnitGamma = unit
  gNormPresetName = spec & (if unit: "-unit" else: "")
  result = true

proc normPresetDescription*(m: NimformerModelJamba): string =
  var groups: seq[int] = @[]
  if gRNormGroupsOverride == 1:
    groups = @[1]
  else:
    for blk in m.blocks:
      if blk.kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}:
        let n = gNormGroups(blk.mambaReal.nGroups)
        if n notin groups: groups.add(n)
  result = "R+P-sidecar ln1=" & (if gRNormOverride == 1: "LayerNorm" else: "RMSNorm") &
    (if gRmsUnitGamma: " gamma=1(unit)" else: "") &
    ", normF=" & (if gUseNormF: "on" else: "off") &
    ", gated-norm-groups=" & (if groups.len == 0: "n/a" else: $groups)

when not defined(noOneDispatch):
  var gOdActive* = false

  proc odLinHasLora(lin: Linear; K, N: int): bool =
    lin.loraR > 0 and lin.loraAValues.len == lin.loraR * K and lin.loraBValues.len == N * lin.loraR

  proc odLinOk(l: Linear): string =
    if not (l.useQuant and l.weightQ.kind == qkFp4KM): return "not fp4-KM"
    if l.weightQ.groupSize != 0 and l.weightQ.groupSize != 32: return "groupSize != 32"
    ""

  proc odTryPrepare(m: var NimformerModelJamba; ctx: Backend): string =
    ## Returns "" when packed + installed (gOdState ready); non-empty = the reason it is NOT eligible.
    ## Every eligibility check happens BEFORE touching the weights -> the old path still works.
    ## Uses pointers (addr) to NOT deep-copy Linear/Mamba2Real (holding tens of MB of weights).
    if not isGpuCtx(ctx): return "can GPU Metal"
    if gRNormOverride != -1 or gRNormGroupsOverride != -1 or gRmsUnitGamma:
      return "norm preset override is on (LayerNorm / 1 group / unit gamma)"
    if m.blocks.len == 0: return "model has no blocks"
    # HYBRID: R blocks (bkMambaReal) -> OD, per run of consecutive blocks (1 dispatch/run).
    # G blocks (bkMambaRealGated) run the old path between runs. Other kinds are not supported yet.
    var odBlocks: seq[int]
    for i in 0 ..< m.blocks.len:
      case m.blocks[i].kind
      of bkMambaReal:
        let w1 = odLinOk(m.blocks[i].mambaReal.inProj)
        if w1.len > 0: return "block " & $i & " inProj " & w1
        let w2 = odLinOk(m.blocks[i].mambaReal.outProj)
        if w2.len > 0: return "block " & $i & " outProj " & w2
        odBlocks.add i
      of bkMambaRealGated: discard
      else:
        return "block " & $i & " of kind " & $m.blocks[i].kind & " is not supported (only R + G)"
    if odBlocks.len == 0: return "no R blocks"
    let b0 = odBlocks[0]
    let hidden = m.embedDim
    let C = m.blocks[b0].mambaReal.intermediateSize
    let cDim = m.blocks[b0].mambaReal.convDim
    let H = m.blocks[b0].mambaReal.numHeads
    let P = C + cDim + H
    # A layer may have NO adapter (e.g. a B=0 adapter dropped at load): the kernel skips that layer's
    # LoRA stage (per-layer flag). Rank/scale must be uniform across the layers WITH adapters.
    var rIn, rOut = 0
    var sIn, sOut = 1'f32
    for i in odBlocks:
      if rIn == 0 and odLinHasLora(m.blocks[i].mambaReal.inProj, hidden, P):
        rIn = m.blocks[i].mambaReal.inProj.loraR; sIn = m.blocks[i].mambaReal.inProj.loraScale
      if rOut == 0 and odLinHasLora(m.blocks[i].mambaReal.outProj, C, hidden):
        rOut = m.blocks[i].mambaReal.outProj.loraR; sOut = m.blocks[i].mambaReal.outProj.loraScale
    for i in odBlocks:
      let hi = odLinHasLora(m.blocks[i].mambaReal.inProj, hidden, P)
      let ho = odLinHasLora(m.blocks[i].mambaReal.outProj, C, hidden)
      if (hi and (m.blocks[i].mambaReal.inProj.loraR != rIn or m.blocks[i].mambaReal.inProj.loraScale != sIn)) or
         (ho and (m.blocks[i].mambaReal.outProj.loraR != rOut or m.blocks[i].mambaReal.outProj.loraScale != sOut)):
        return "LoRA rank/scale is not uniform across layers (block " & $i & ")"
    try:
      var st = odBegin(ctx, hidden, C, cDim, H, m.blocks[b0].mambaReal.headDim,
        m.blocks[b0].mambaReal.dState, m.blocks[b0].mambaReal.nGroups,
        m.blocks[b0].mambaReal.convKernel, odBlocks.len,
        gNormGroups(m.blocks[b0].mambaReal.nGroups), m.blocks[b0].rms1.eps, rIn, rOut, sIn, sOut)
      st.slotOfBlock = newSeq[int](m.blocks.len)
      for i in 0 ..< m.blocks.len: st.slotOfBlock[i] = -1
      var slot = 0
      for i in odBlocks:
        st.slotOfBlock[i] = slot
        inc slot
      # Runs of consecutive R blocks (split by G blocks), each run = 1 dispatch + 1 separate weight buffer.
      # Each Metal buffer is at most ~2GB -> split further past ~1.5GB.
      let perLayerBytes = odFp4Bytes(hidden, P) + odFp4Bytes(C, hidden) + 512
      let maxLayersPerSeg = max(1, 1_500_000_000 div perLayerBytes)
      var bi = 0
      while bi < m.blocks.len:
        if st.slotOfBlock[bi] < 0:
          inc bi
          continue
        let startSlot = st.slotOfBlock[bi]
        var cnt = 0
        while bi < m.blocks.len and st.slotOfBlock[bi] >= 0 and cnt < maxLayersPerSeg:
          inc cnt
          inc bi
        st.segs.add (start: startSlot, count: cnt)
      for i in odBlocks:
        let blk = addr m.blocks[i]
        let mr = addr blk[].mambaReal
        let lutIn = fp4_gemv_gpu.buildFp4KmLutQ(mr[].inProj.weightQ)
        let lutOut = fp4_gemv_gpu.buildFp4KmLutQ(mr[].outProj.weightQ)
        odAddLayer(st, st.slotOfBlock[i], mr[].aLog.data, mr[].dSkip.data, mr[].dtBias.data, mr[].normWeight.data,
          mr[].convWeight.data, mr[].convBias.data, blk[].ln1.gamma.data,
          mr[].inProj.weightQ.bytes, lutIn, mr[].outProj.weightQ.bytes, lutOut,
          (if odLinHasLora(mr[].inProj, hidden, P): @(mr[].inProj.loraAValues) else: @[]),
          (if odLinHasLora(mr[].inProj, hidden, P): @(mr[].inProj.loraBValues) else: @[]),
          (if odLinHasLora(mr[].outProj, C, hidden): @(mr[].outProj.loraAValues) else: @[]),
          (if odLinHasLora(mr[].outProj, C, hidden): @(mr[].outProj.loraBValues) else: @[]))
      odFinish(ctx, st)
      odInstall(st)
      # Host weights are NOT freed here: the prepareGpuModel loop below still loads a separate fp4 cache
      # for R blocks (so the PARALLEL PREFILL can run the M-block GEMM), then drops the host copy after upload.
      return ""
    except CatchableError as e:
      return e.msg

proc prepareGpuModel*(m: var NimformerModelJamba, lora: var JambaLora, ctx: Backend) =
  let be = ctx.kind.toByby()
  let hidden = m.embedDim
  when not defined(noOneDispatch):
    # Iris 640 end-to-end A/B: independent kernels in one session are faster.
    # Keep the persistent mega-kernel available as an explicit experiment.
    if getEnv("BYBY_ONE_DISPATCH", "0") != "0" and not gOdActive:
      let odWhy = odTryPrepare(m, ctx)
      if odWhy.len == 0:
        gOdActive = true
        stderr.writeLine "[one-dispatch] ON: " & $gOdState.sh.nLayers & " R layers in " & $gOdState.segs.len &
          " dispatches/token (G blocks run the old path in between, ntg=" & $gOdState.ntg & ")"
      else:
        stderr.writeLine "[one-dispatch] OFF -> using the old path: " & odWhy
  for i in 0 ..< m.blocks.len:
    when not defined(noOneDispatch):
      if gOdActive and i < gOdState.slotOfBlock.len and gOdState.slotOfBlock[i] >= 0:
        # R blocks are owned by OD (state + decode). A separate fp4 cache is still loaded for the parallel prefill.
        discard prepareGpuLinForLinear(ctx, "block" & $i & ".mambaReal.inProj.weight", m.blocks[i].mambaReal.inProj)
        discard prepareGpuLinForLinear(ctx, "block" & $i & ".mambaReal.outProj.weight", m.blocks[i].mambaReal.outProj)
        continue
    case m.blocks[i].kind
    of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
      let pin = "block" & $i & ".mambaReal.inProj.weight"
      let pout = "block" & $i & ".mambaReal.outProj.weight"
      discard prepareGpuLinForLinear(ctx, pin, m.blocks[i].mambaReal.inProj)
      discard prepareGpuLinForLinear(ctx, pout, m.blocks[i].mambaReal.outProj)
      if i notin gLayersV2:
        let mr = m.blocks[i].mambaReal
        ensureLayerV2(ctx, i,
          mr.numHeads, mr.headDim, mr.dState, mr.nGroups,
          mr.intermediateSize, mr.convDim, mr.convKernel, mr.hiddenSize,
          mr.aLog.data, mr.dSkip.data, mr.dtBias.data, mr.normWeight.data,
          mr.convWeight.data, mr.convBias.data, m.blocks[i].ln1.gamma.data,
          gNormGroups(mr.nGroups), m.blocks[i].rms1.eps,
          ln1Beta = m.blocks[i].ln1.beta.data)
      if m.blocks[i].kind in {bkMambaRealPlusAttn, bkMambaRealGated} and i notin gAttnLayers:
        let pq = "block" & $i & ".attn.qkvProj.weight"
        let po = "block" & $i & ".attn.outProj.weight"
        let qkv = prepareGpuLinForLinear(ctx, pq, m.blocks[i].attn.qkvProj)
        let outp = prepareGpuLinForLinear(ctx, po, m.blocks[i].attn.outProj)
        var fc1, fc2: GpuLin
        # Fix: the Attention sidecar SHARES ln1 (like Mamba) - no separate attnLn1
        gAttnLayers[i] = initAttnLayer(ctx, hidden, m.blocks[i].attn.nHeads, 0,
          qkv, outp, fc1, fc2,
          m.blocks[i].ln1.gamma.data, m.blocks[i].ln1.beta.data,
          m.blocks[i].ln2.gamma.data, m.blocks[i].ln2.beta.data, m.blocks[i].rms1.eps)
        if m.blocks[i].kind == bkMambaRealGated:
          let gateName = "block" & $i & ".fuseGate.weight"
          gAttnLayers[i].gate = prepareGpuLinForLinear(ctx, gateName, m.blocks[i].fuseGate)
          gAttnLayers[i].gated = true
          gAttnLayers[i].hGateInput = sessionAllocScratch(be, 2 * hidden)
          gAttnLayers[i].hGateLogit = sessionAllocScratch(be, hidden)
          gAttnLayers[i].hGateProb = sessionAllocScratch(be, hidden)
          gAttnLayers[i].hGateComplement = sessionAllocScratch(be, hidden)
          var gateOnes = newSeq[float32](hidden)
          for j in 0 ..< hidden: gateOnes[j] = 1'f32
          gAttnLayers[i].hGateOnes = sessionUploadPrivate(be, gateOnes)
    of bkAttn:
      if i notin gAttnLayers:
        let pq = "block" & $i & ".attn.qkvProj.weight"
        let po = "block" & $i & ".attn.outProj.weight"
        let pf1 = "block" & $i & ".ff.fc1.weight"
        let pf2 = "block" & $i & ".ff.fc2.weight"
        var qkv, outp, fc1, fc2: GpuLin
        qkv = prepareGpuLinForLinear(ctx, pq, m.blocks[i].attn.qkvProj)
        outp = prepareGpuLinForLinear(ctx, po, m.blocks[i].attn.outProj)
        fc1 = prepareGpuLinForLinear(ctx, pf1, m.blocks[i].ff.fc1)
        fc2 = prepareGpuLinForLinear(ctx, pf2, m.blocks[i].ff.fc2)
        gAttnLayers[i] = initAttnLayer(ctx, hidden, m.blocks[i].attn.nHeads, hidden * m.ffMult,
          qkv, outp, fc1, fc2,
          m.blocks[i].ln1.gamma.data, m.blocks[i].ln1.beta.data,
          m.blocks[i].ln2.gamma.data, m.blocks[i].ln2.beta.data, 1e-5'f32)
    of bkMamba, bkSpectron:
      raise newException(ValueError,
        "prepareGpuModel: M/S blocks have no 100% GPU path yet")

  # ── Persistent buffers + normF + lm_head ─────────────────────────────────
  # Nobody used to initialise these: gHXPersist/gHNormPersist/gHFinalPersist/
  # gHLogitsPersist were NIL handles, gLmHead was an empty GpuLin (knF32 = "") ->
  #   - sessionCopyInto(hEmb -> gHXPersist) reported "dst offset exceeds the size"
  #   - encodeLin(gLmHead) -> "encodeVmKernel: could not bind pipeline ''"
  # Scratch = Shared, lives for the whole process (like the old infer version that worked).
  if not gGpuModelReady:
    gHXPersist      = sessionAllocScratch(be, hidden)          # residual, overwritten in place by every layer
    gHNormPersist   = sessionAllocScratch(be, hidden)          # shared pre-norm scratch
    gHFinalPersist  = sessionAllocScratch(be, hidden)          # output normF
    gHLogitsPersist = sessionAllocScratch(be, m.outProj.outF)  # logits [vocab]
    if m.normF.gamma.data.len != hidden:
      raise newException(ValueError,
        "prepareGpuModel: normF.gamma co " & $m.normF.gamma.data.len &
        " elements, need " & $hidden)
    gNormFGamma = sessionUploadPrivate(be, m.normF.gamma.data)
    block:
      var ones = newSeq[float32](hidden)
      for i in 0 ..< ones.len: ones[i] = 1'f32
      gNormFOnes = sessionUploadPrivate(be, ones)
    # lm_head goes through the SAME preparation path as every other Linear (fp4-KM / int4-asym,
    # no fp32 fallback). The name "outProj.weight" matches bindQuantBaseToLinears.
    gLmHead = prepareGpuLinForLinear(ctx, "outProj.weight", m.outProj)
    # Checks the handle for the real backend (Vulkan -> vulkanHandle, Metal -> metalHandle).
    # This used to check only metalHandle on macosx -> backend=vulkan always failed
    # (metalHandle nil) even when the Vulkan buffer was allocated successfully.
    proc persistNil(h: SessionHandle): bool =
      # VulkanBufferHandle is an object {buf, mem, sizeBytes} - check .buf
      if be == gbVulkan: return h.vulkanHandle.buf == nil
      if be == gbMetal: return h.metalHandle == nil
      false
    if persistNil(gHXPersist) or persistNil(gHNormPersist) or
       persistNil(gHFinalPersist) or persistNil(gHLogitsPersist) or
       persistNil(gNormFGamma):
      let which =
        if be == gbVulkan: "vulkanHandle nil"
        elif be == gbMetal: "metalHandle nil"
        else: "handle nil"
      raise newException(CatchableError,
        "prepareGpuModel: allocating persistent buffers failed (" & which &
        ", backend=" & $be & ") - out of GPU/unified memory?")
    gGpuModelReady = true

proc encodeHeadGpu(m: var NimformerModelJamba, ctx: Backend) =
  ## normF (depending on the preset) + lm_head. Shared by encodeOneToken and the trace version.
  let be = ctx.kind.toByby()
  let hidden = m.embedDim
  if gUseNormF:
    let gamF = if gRmsUnitGamma: gNormFOnes else: gNormFGamma
    if not sessionRmsnormEnc(be, gHXPersist, gamF, gHFinalPersist, 1, hidden, 1e-5'f32):
      raise newException(CatchableError, "encodeHeadGpu: normF rmsnorm failed")
    encodeLin(gLmHead, gHFinalPersist, 0, gHLogitsPersist, 0, bias = false)
  else:
    # Matches training: lm_head reads the residual DIRECTLY, without normF.
    encodeLin(gLmHead, gHXPersist, 0, gHLogitsPersist, 0, bias = false)

proc encodeSidecarGpu(ctx: Backend, blockIdx: int) =
  ## The Linear++ branch of P blocks (R + a parallel sidecar). Its input norm is IDENTICAL to the R branch
  ## of the same block: in training both branches read the shared x1 = rms1(x) (jamba_hybrid.forwardBlock),
  ## so here too 1 LN/RMS decision (layerUsesLayerNorm) + 1 gamma flag are used.
  ## Must be called BEFORE that block's encodeLayerV2 (encodeLayerV2 overwrites the residual in place).
  encodeLinearPlusBranch(ctx, blockIdx, gHXPersist, gHNormPersist, gAttnScratch.hAttn,
                         useRms = not layerUsesLayerNorm(blockIdx),
                         unitGamma = gRmsUnitGamma)

proc encodeBlockOld(m: var NimformerModelJamba, ctx: Backend, i: int) =
  ## Encodes 1 block through the old path (used for every block when OD is off, and for G blocks when OD is on).
  case m.blocks[i].kind
  of bkMambaReal:
    encodeLayerV2(ctx, i, gHXPersist, gHNormPersist,
                  "block" & $i & ".mambaReal.inProj.weight",
                  "block" & $i & ".mambaReal.outProj.weight")
  of bkMambaRealPlusAttn, bkMambaRealGated:
    encodeMamba2RealLinearPlusBlock(ctx, i, gHXPersist, gHNormPersist,
      skipLinearPlus = gSkipAttn, unitGamma = gRmsUnitGamma,
      gated = m.blocks[i].kind == bkMambaRealGated)
  of bkAttn:
    if not gSkipAttn:
      encodeAttnLayer(ctx, i, gHXPersist, gHNormPersist)
  of bkMamba, bkSpectron:
    raise newException(ValueError, "encodeOneToken: M/S blocks are not supported yet")

proc encodeOneToken(m: var NimformerModelJamba, ctx: Backend;
                    produceLogits: bool = true) =
  ## Encodes a whole forward step (R+A following the pattern -> normF -> lm_head)
  ## into the open session. No sessionRead in between.
  let be = ctx.kind.toByby()
  let hidden = m.embedDim
  when not defined(noOneDispatch):
    if gOdActive:
      # Each run of consecutive R blocks = 1 compute dispatch; G blocks run the old path in between.
      var i = 0
      var segK = 0
      var tSplit = monoSecs()
      while i < m.blocks.len:
        if gOdState.slotOfBlock[i] >= 0:
          odEncodeSegment(gOdState, segK, gHXPersist)
          decSplitMark(be, "od.seg" & $segK & "(" & $gOdState.segs[segK].count & "L)", tSplit)
          i += gOdState.segs[segK].count
          inc segK
        else:
          encodeBlockOld(m, ctx, i)
          decSplitMark(be, "G.block" & $i, tSplit)
          inc i
      if produceLogits:
        encodeHeadGpu(m, ctx)
      if produceLogits and gDecSplitOn and gDecSplitActive:
        # lm_head: ends at the caller's sessionEnd; measured with a separate buffer for consistency.
        decSplitMark(be, "normF+lm_head", tSplit)
        inc gDecSplitTokens
      return
  var tSplit = monoSecs()
  for i in 0 ..< m.blocks.len:
    encodeBlockOld(m, ctx, i)
    if produceLogits and gDecSplitOn and gDecSplitActive:
      decSplitMark(be, "block" & $i & "(" & $m.blocks[i].kind & ")", tSplit)
  # normF == HF Mamba2 backbone.norm_f = a gamma-only RMSNorm (matches the old infer version
  # that worked, PATCH_MARKER_V4). NO more "copy straight through if the norm fails": silently skipping
  # normF would give wrong logits without anyone noticing.
  if produceLogits:
    encodeHeadGpu(m, ctx)
    if gDecSplitOn and gDecSplitActive:
      decSplitMark(be, "normF+lm_head", tSplit)
      inc gDecSplitTokens

proc warmupGpuKernelsOnce(m: var NimformerModelJamba, ctx: Backend) =
  ## Forces the JIT-compile + first PSO bind for EVERY kernel kind, in 1 ISOLATED
  ## session (input = the embedding of token 0, x = 0 since gHXPersist has not received
  ## the real embedding yet) then cleanly RESETS the R/A state before the real prefill
  ## starts (so the warmup does not leak into the pre-prefill state - in the spirit of
  ## the previous version: "this warmup forces EVERY kernel kind to be JIT-compiled AND dispatched/
  ## PSO-bound for the first time HERE, completely safe from the watchdog").
  let be = ctx.kind.toByby()
  let svSkip = gSkipAttn
  let svLora = gDisableLora
  gSkipAttn = false
  gDisableLora = false
  defer:
    gSkipAttn = svSkip
    gDisableLora = svLora
  if sessionBegin(be):
    var warmErr = ""
    try:
      encodeOneToken(m, ctx)
    except CatchableError as e:
      warmErr = e.msg
    if not sessionEnd(be):
      stderr.writeLine "[warmup] session flush error: " & sessionGetLastError(be) &
        " -> sessionRecoverQueue() before the next step"
      if not sessionRecoverQueue(be):
        stderr.writeLine "[warmup] sessionRecoverQueue failed"
    if warmErr.len > 0:
      # Do NOT swallow the error: if the encoder is broken the later reset sessions break too
      # (a cascade of "copy_into dst offset" errors), and the real forward would repeat the error.
      raise newException(CatchableError, "warmup encode error: " & warmErr)

  if sessionBegin(be):
    for i in 0 ..< m.blocks.len:
      case m.blocks[i].kind
      of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
        if not resetLayerStateV2(ctx, i):
          stderr.writeLine "[warmup] WARNING: could not reset the state of R block " & $i
        if m.blocks[i].kind in {bkMambaRealPlusAttn, bkMambaRealGated} and
           not resetAttnLayerState(ctx, i):
          stderr.writeLine "[warmup] WARNING: could not reset the sidecar A state of block " & $i
      of bkAttn:
        if not resetAttnLayerState(ctx, i):
          stderr.writeLine "[warmup] WARNING: could not reset the state of A block " & $i
      else: discard
    if not sessionEnd(be):
      stderr.writeLine "[warmup] session state reset error: " & sessionGetLastError(be)
      if not sessionRecoverQueue(be): discard

proc forwardStepJambaGpu*(m: var NimformerModelJamba, tokenId: int, ctx: Backend;
                          produceLogits: bool = true): Tensor =
  ## ONE session / token for the WHOLE model (R + A interleaved following the pattern),
  ## using a single command buffer - as required: "fuse into 1 kernel, i.e. 1
  ## session per token". No sessionRead between layers; logits are read only once.
  inc gInferStepCount
  let be = ctx.kind.toByby()
  let hidden = m.embedDim

  if gGpuGenStatePoisoned:
    raise newException(CatchableError,
      "forwardStepJambaGpu: the recurrent GPU state is no longer trustworthy (the previous step hit an error/timeout). " &
      "Call resetGpuGenState() successfully before continuing; do not replay a token on a state that may be partially updated.")

  let profDecode = produceLogits and getEnv("BYBY_DECODE_PROFILE", "0") != "0"
  let tProfEntry = (if profDecode: monoSecs() else: 0.0)
  if not gGpuWarmedUp:
    warmupGpuKernelsOnce(m, ctx)
    gGpuWarmedUp = true

  var xTensor = m.embed.lookupBatch(@[@[tokenId]], ctx)
  var hEmb = sessionUpload(be, xTensor.data)
  var statefulDispatchStarted = false
  var stepComplete = false
  var tProf0, tProf1, tProf2, tProf3: float64
  try:
    if profDecode: tProf0 = monoSecs()
    if not sessionBegin(be):
      gGpuGenStatePoisoned = true
      discard sessionRecoverQueue(be)
      raise newException(CatchableError,
        "forwardStepJambaGpu: sessionBegin failed; no logits, reset the recurrent state before retrying.")
    if profDecode: tProf1 = monoSecs()

    if not sessionCopyInto(be, hEmb, gHXPersist, 0, 0, hidden):
      let copyErr = "copy embedding -> gHXPersist failed (has prepareGpuModel run?)"
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
      gGpuGenStatePoisoned = true
      raise newException(CatchableError, "forwardStepJambaGpu: " & copyErr)

    try:
      statefulDispatchStarted = true
      gDecSplitActive = gInferStepCount > 0   # excludes the first token + warmup
      try:
        encodeOneToken(m, ctx, produceLogits = produceLogits)
      finally:
        gDecSplitActive = false
    except CatchableError as e:
      # encodeOneToken has already queued writes to persistent SSM/Linear++
      # state. Even if sessionEnd later reports success, this token cannot be
      # replayed safely; reset all recurrent state before another attempt.
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
      gGpuGenStatePoisoned = true
      raise newException(CatchableError,
        "forwardStepJambaGpu: encode failed after stateful dispatch; state invalidated, reset required: " & e.msg)
    if profDecode: tProf2 = monoSecs()

    if not sessionEnd(be):
      let err = sessionGetLastError(be)
      discard sessionRecoverQueue(be)
      gGpuGenStatePoisoned = true
      raise newException(CatchableError,
        "forwardStepJambaGpu: GPU command buffer failed (" & err & "). " &
        "Stateful token was NOT replayed because Mamba2/Linear++ state may be partially advanced; " &
        "resetGpuGenState() and restart the sequence. Stateless matmul2 kernels retry internally.")
    if profDecode: tProf3 = monoSecs()
    when not defined(noOneDispatch):
      if gOdActive and getEnv("BYBY_OD_CHECK", "0") != "0":
        let odErr = odCheckErr(gOdState)
        if odErr != 0'u32:
          gGpuGenStatePoisoned = true
          raise newException(CatchableError,
            "one-dispatch: grid barrier het gioi han spin (lastErr=" & $odErr &
            ") - the threadgroups are not co-resident. Lower BYBY_OD_TG and run t_grid_barrier.nim.")

    # BYBY_G_DIAGNOSTICS: read the numbers AFTER sessionEnd (reading mid-session = garbage)
    if getEnv("BYBY_G_DIAGNOSTICS", "0") != "0":
      proc vecStats(name: string; bi: int; v: seq[float32]) =
        if v.len == 0:
          stderr.writeLine "[G_diag] block=" & $bi & " " & name & " EMPTY"
          return
        var sm = 0.0
        var s2 = 0.0
        var mn = 1e30
        var mx = -1e30
        var nanN = 0
        var n0 = 0
        var n1 = 0
        for x0 in v:
          let x = float(x0)
          if x != x or abs(x) > 1e30:
            inc nanN
            continue
          sm += x
          s2 += x * x
          if x < mn: mn = x
          if x > mx: mx = x
          if name == "gate":
            if x < 0.05: inc n0
            if x > 0.95: inc n1
        let n = float(v.len - nanN)
        if n <= 0:
          stderr.writeLine "[G_diag] block=" & $bi & " " & name & " ALL NaN/Inf"
          return
        let mean = sm / n
        let rms = sqrt(s2 / n)
        var line = "[G_diag] block=" & $bi & " " & name &
          " mean=" & $mean & " rms=" & $rms & " min=" & $mn & " max=" & $mx &
          " nan/inf=" & $nanN
        if name == "gate":
          line &= " frac~0(attn)=" & $(float(n0)/n) & " frac~1(mamba)=" & $(float(n1)/n)
        stderr.writeLine line
      for bi in 0 ..< m.blocks.len:
        if m.blocks[bi].kind != bkMambaRealGated: continue
        if bi notin gAttnLayers: continue
        let al = gAttnLayers[bi]
        if not al.gated: continue
        let n = m.embedDim
        stderr.writeLine "[G_diag] ---- block " & $bi & "  mix = g*mamba + (1-g)*attn ; x += mix ----"
        vecStats("gate", bi, sessionRead(be, al.hGateProb, n))
        let hasSnap =
          (be == gbVulkan and al.hDiagMamba.vulkanHandle.buf != nil) or
          (be == gbMetal and al.hDiagMamba.metalHandle != nil)
        if hasSnap:
          vecStats("mamba(before)", bi, sessionRead(be, al.hDiagMamba, n))
          vecStats("attn(before)", bi, sessionRead(be, al.hDiagAttn, n))
        if bi in gLayersV2:
          vecStats("mix(after)", bi, sessionRead(be, gLayersV2[bi].hMix, n))
        vecStats("residual x", bi, sessionRead(be, gHXPersist, n))
    # Prefill tokens before the last prompt token only need to update the recurrent state.
    # normF/lm_head are not run and logits are not read (the head has no dependency
    # on the state, so the result after prefill is exactly the same as the old path).
    let logits: seq[float32] =
      if produceLogits: sessionRead(be, gHLogitsPersist, m.outProj.outF)
      else: @[]
    block:
      let dumpPath = if produceLogits: getEnv("BYBY_DUMP_LOGITS") else: ""   # compare the old path vs 1-dispatch: od_compare_logits.py
      if dumpPath.len > 0:
        var df: File
        if open(df, dumpPath, fmAppend):
          if logits.len > 0: discard df.writeBuffer(unsafeAddr logits[0], logits.len * 4)
          df.close()
    if profDecode:
      let tProf4 = monoSecs()
      if gProfTokens == 0:
        let ms = proc(a, b: float64): float64 = (b - a) * 1000.0
        gProfFirstMs = ms(tProfEntry, tProf4)
        gProfFirst = &"[infer] decode token dau: warmup+embed={ms(tProfEntry, tProf0):.1f}ms " &
          &"sessionBegin={ms(tProf0, tProf1):.1f}ms encodeOneToken={ms(tProf1, tProf2):.1f}ms " &
          &"sessionEnd={ms(tProf2, tProf3):.1f}ms sessionRead={ms(tProf3, tProf4):.1f}ms " &
          &"TOTAL={gProfFirstMs:.1f}ms"
      else:
        gProfBegin += tProf1 - tProf0
        gProfEncode += tProf2 - tProf1
        gProfEnd += tProf3 - tProf2
        gProfRead += tProf4 - tProf3
      inc gProfTokens
    if produceLogits:
      result = newTensor(@[1, 1, m.outProj.outF])
      result.data = logits
    else:
      result = Tensor(shape: @[1, 1, 0], data: @[])
    inc gInferStepCount
    stepComplete = true
  except CatchableError:
    # Includes failures during the final logits read. Once any stateful
    # kernels were queued, only an explicit recurrent-state reset can make
    # subsequent tokens safe again.
    if statefulDispatchStarted and not stepComplete:
      gGpuGenStatePoisoned = true
    raise
  finally:
    sessionFree(be, hEmb)

type
  BlockStat* = object
    idx*: int
    kind*: char          ## 'R' / 'A' / 'P'
    rms*, maxAbs*, mean*: float32
    nonFinite*: int
    inRms*: float32      ## residual rms BEFORE the block (to compare with the sidecar contribution)
    sideRms*, sideMax*: float32   ## output of the P block's Linear++ branch (before being added to the residual); -1 = none

proc vecStats(v: seq[float32]): tuple[rms, maxAbs, mean: float32, bad: int] =
  var ss = 0.0
  var sm = 0.0
  var mx = 0.0
  var bad = 0
  for x in v:
    if x.classify in {fcNan, fcInf, fcNegInf}:
      inc bad
      continue
    let d = float(x)
    ss += d * d
    sm += d
    if abs(d) > mx: mx = abs(d)
  let n = float(max(1, v.len - bad))
  result = (float32(sqrt(ss / n)), float32(mx), float32(sm / n), bad)

proc forwardStepJambaGpuTrace*(m: var NimformerModelJamba, tokenId: int, ctx: Backend,
                               stats: var seq[BlockStat]): Tensor =
  ## Like forwardStepJambaGpu (SAME math, SAME state) but split into
  ## 1 session / block, reading the residual after each block -> measures rms/max/mean/NaN.
  ## Somewhat SLOWER (64 commits) - ONLY for diagnostics (BYBY_TRACE_BLOCKS=1
  ## or in BYBY_NORM_SWEEP). Shows which block makes the residual blow up/NaN.
  let be = ctx.kind.toByby()
  let hidden = m.embedDim
  if gGpuGenStatePoisoned:
    raise newException(CatchableError,
      "forwardStepJambaGpuTrace: the recurrent GPU state is no longer trustworthy; call resetGpuGenState() first.")
  if not gGpuWarmedUp:
    warmupGpuKernelsOnce(m, ctx)
    gGpuWarmedUp = true
  stats.setLen(0)
  var xTensor = m.embed.lookupBatch(@[@[tokenId]], ctx)
  var hEmb = sessionUpload(be, xTensor.data)
  try:
    if not sessionBegin(be):
      raise newException(CatchableError, "trace: sessionBegin (embedding) failed")
    if not sessionCopyInto(be, hEmb, gHXPersist, 0, 0, hidden):
      discard sessionEnd(be)
      raise newException(CatchableError, "trace: copy embedding failed")
    if not sessionEnd(be):
      raise newException(CatchableError, "trace: sessionEnd (embedding) failed")
    var prevRms = vecStats(sessionRead(be, gHXPersist, hidden)).rms
    for i in 0 ..< m.blocks.len:
      if gSkipAttn and m.blocks[i].kind == bkAttn: continue
      if not sessionBegin(be):
        raise newException(CatchableError, "trace: sessionBegin failed at block " & $i)
      var kindCh = '?'
      var sideRms = -1'f32
      var sideMax = -1'f32
      try:
        case m.blocks[i].kind
        of bkMambaReal:
          kindCh = 'R'
          encodeLayerV2(ctx, i, gHXPersist, gHNormPersist,
                        "block" & $i & ".mambaReal.inProj.weight",
                        "block" & $i & ".mambaReal.outProj.weight")
        of bkMambaRealPlusAttn:
          kindCh = 'P'
          if not gSkipAttn:
            encodeSidecarGpu(ctx, i)
            # Split the session to READ the sidecar output (rms/max) before it is added to the residual:
            # sidecar >> residual is a sign that this branch's norm/weights drifted during training.
            if not sessionEnd(be):
              raise newException(CatchableError, "trace: sessionEnd (sidecar) failed")
            let sst = vecStats(sessionRead(be, gAttnScratch.hAttn, hidden))
            sideRms = sst.rms
            sideMax = sst.maxAbs
            if not sessionBegin(be):
              raise newException(CatchableError, "trace: sessionBegin (after sidecar) failed at block " & $i)
          encodeLayerV2(ctx, i, gHXPersist, gHNormPersist,
                        "block" & $i & ".mambaReal.inProj.weight",
                        "block" & $i & ".mambaReal.outProj.weight",
                        inputPreNormalized = not gSkipAttn,
                        addResidual = gSkipAttn)
          if not gSkipAttn:
            encodeMamba2RealLinearPlusResidual(ctx, i, gHXPersist,
              gHNormPersist, gAttnScratch.hAttn)
        of bkMambaRealGated:
          kindCh = 'G'
          encodeMamba2RealLinearPlusBlock(ctx, i, gHXPersist, gHNormPersist,
            skipLinearPlus = gSkipAttn, unitGamma = gRmsUnitGamma, gated = true)
        of bkAttn:
          kindCh = 'A'
          encodeAttnLayer(ctx, i, gHXPersist, gHNormPersist)
        of bkMamba, bkSpectron:
          raise newException(ValueError, "trace: M/S blocks are not supported yet")
      except CatchableError as e:
        discard sessionEnd(be)
        raise newException(CatchableError, "trace: encode error at block " & $i & ": " & e.msg)
      if not sessionEnd(be):
        raise newException(CatchableError,
          "trace: sessionEnd failed at block " & $i & ": " & sessionGetLastError(be))
      let xv = sessionRead(be, gHXPersist, hidden)
      let st = vecStats(xv)
      stats.add BlockStat(idx: i, kind: kindCh, rms: st.rms, maxAbs: st.maxAbs,
                          mean: st.mean, nonFinite: st.bad, inRms: prevRms,
                          sideRms: sideRms, sideMax: sideMax)
      prevRms = st.rms
    # head: normF (tuy preset) + lm_head
    if not sessionBegin(be):
      raise newException(CatchableError, "trace: sessionBegin (head) failed")
    try:
      encodeHeadGpu(m, ctx)
    except CatchableError as e:
      discard sessionEnd(be)
      raise newException(CatchableError, "trace: head encode error: " & e.msg)
    if not sessionEnd(be):
      raise newException(CatchableError,
        "trace: sessionEnd (head) failed: " & sessionGetLastError(be))
  except CatchableError:
    gGpuGenStatePoisoned = true
    raise
  finally:
    sessionFree(be, hEmb)
  let logits = sessionRead(be, gHLogitsPersist, m.outProj.outF)
  result = newTensor(@[1, 1, m.outProj.outF])
  result.data = logits

proc resetGpuGenState*(m: var NimformerModelJamba, ctx: Backend) =
  ## Clears the R (conv/ssm) + A (z/S) state - called once before starting a
  ## NEW sequence (unlike the warmup: this is for when generate() really starts,
  ## after the warmup has run at least once).
  let be = ctx.kind.toByby()
  # Reset is idempotent (writes zeros), so retrying the entire reset after a
  # failed command buffer cannot double-advance recurrent state.
  var lastErr = "sessionBegin failed"
  for attempt in 0 ..< 2:
    if not sessionBegin(be):
      discard sessionRecoverQueue(be)
      lastErr = "sessionBegin failed"
      continue
    var resetOk = true
    for i in 0 ..< m.blocks.len:
      case m.blocks[i].kind
      of bkMambaReal:
        resetOk = resetLayerStateV2(ctx, i) and resetOk
      of bkMambaRealPlusAttn, bkMambaRealGated:
        resetOk = resetLayerStateV2(ctx, i) and resetOk
        resetOk = resetAttnLayerState(ctx, i) and resetOk
      of bkAttn:
        resetOk = resetAttnLayerState(ctx, i) and resetOk
      else: discard
    if not resetOk:
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
      lastErr = "one or more state reset encodes failed"
      continue
    if sessionEnd(be):
      gGpuGenStatePoisoned = false
      return
    lastErr = sessionGetLastError(be)
    discard sessionRecoverQueue(be)
  gGpuGenStatePoisoned = true
  raise newException(CatchableError,
    "resetGpuGenState: could not reset recurrent GPU state after 2 idempotent attempts: " & lastErr)

proc warmupGpuForParallelPrefill*(m: var NimformerModelJamba; ctx: Backend) =
  ## Compile/bind recurrent decode kernels before parallel prefill imports its
  ## terminal states. The normal first-token path warms up and then resets all
  ## states, which would otherwise erase states imported by prefill.
  if not gGpuWarmedUp:
    warmupGpuKernelsOnce(m, ctx)
    gGpuWarmedUp = true
  resetGpuGenState(m, ctx)

proc markGpuPrefillReady*() =
  ## A successful resident prefill has already installed every recurrent
  ## state. A first-token warmup would erase those states; compile any cold
  ## decode kernels normally when encoding that token instead.
  gGpuWarmedUp = true

proc isGpuTimeoutFailure(msg: string): bool =
  let lower = msg.toLowerAscii()
  result = "timeout" in lower or "timed out" in lower

proc forwardStepJambaGpuRetry*(m: var NimformerModelJamba; tokenId: int;
                               replayTokens: openArray[int]; replayCount: int;
                               ctx: Backend; produceLogits: bool = true): Tensor =
  ## A recurrent token cannot be blindly replayed after a failed command buffer:
  ## Mamba2/Linear++ may already have advanced some persistent states. On timeout,
  ## reset all states, rebuild the known-good prefix, and only then retry tokenId.
  doAssert replayCount >= 0 and replayCount <= replayTokens.len
  try:
    return forwardStepJambaGpu(m, tokenId, ctx, produceLogits = produceLogits)
  except CatchableError as firstErr:
    if not isGpuTimeoutFailure(firstErr.msg): raise
    stderr.writeLine("[infer] recurrent GPU timeout; reset state, replaying " &
      $replayCount & " committed prefix tokens, then retrying current token once")
    try:
      resetGpuGenState(m, ctx)
      for i in 0 ..< replayCount:
        discard forwardStepJambaGpu(m, replayTokens[i], ctx, produceLogits = false)
      return forwardStepJambaGpu(m, tokenId, ctx, produceLogits = produceLogits)
    except CatchableError as retryErr:
      raise newException(CatchableError,
        "forwardStepJambaGpuRetry: timeout recovery/replay failed; original=" &
        firstErr.msg & "; recovery=" & retryErr.msg)

proc forwardStepJambaGpuTraceRetry*(m: var NimformerModelJamba; tokenId: int;
                                    replayTokens: openArray[int]; replayCount: int;
                                    ctx: Backend; stats: var seq[BlockStat]): Tensor =
  ## Trace variant with the same sequence-safe retry policy as normal inference.
  doAssert replayCount >= 0 and replayCount <= replayTokens.len
  try:
    return forwardStepJambaGpuTrace(m, tokenId, ctx, stats)
  except CatchableError as firstErr:
    if not isGpuTimeoutFailure(firstErr.msg): raise
    stderr.writeLine("[infer] traced recurrent GPU timeout; reset state, replaying " &
      $replayCount & " committed prefix tokens, then retrying traced token once")
    try:
      resetGpuGenState(m, ctx)
      for i in 0 ..< replayCount:
        discard forwardStepJambaGpu(m, replayTokens[i], ctx)
      stats.setLen(0)
      return forwardStepJambaGpuTrace(m, tokenId, ctx, stats)
    except CatchableError as retryErr:
      raise newException(CatchableError,
        "forwardStepJambaGpuTraceRetry: timeout recovery/replay failed; original=" &
        firstErr.msg & "; recovery=" & retryErr.msg)

proc lpSelfCheck*(ctx: Backend, blockIdx: int): string =
  ## Checks that the Linear++ step kernel runs 100% on the GPU (see gpu_layers.lpStepSelfCheck).
  lpStepSelfCheck(ctx, blockIdx)
