## Mamba-3 SISO 187M pipeline forked from test/codestral.
## R = Mamba-3 SISO + gated MLP; P/G add the existing Linear++ sidecar.
## Projection, LoRA, optimizer and checkpoint helpers reuse Codestral formats.
## See README.md for supported paths and validation limits.

import std/[math, random, strformat, tables, strutils, times, os, monotimes]

proc monoSecs*(): float =
  ## Monotonic clock, does NOT advance while the machine sleeps (epochTime does).
  getMonoTime().ticks.float / 1e9

import nimformer
import backend
import m3_gpu_runtime
import m3_block_kernel
import m3_hybrid   # HybridGenState, JambaGenState, isPureMambaRealPattern, addT, ...
import linear_plus_infer
import mamba3_real     # forwardStepMamba3Real
import mamba2_ssm_gpu
import m3_diagnostics

import gpu_layers
import fp4_gemv_gpu    # prepare path: gFp4CacheGpu, fp4GemvGpu, Shared weight
import test_nimformer  # forwardLinearPlusSession (Attention GPU)
import quant           # QuantTensor, dequantizeTensor
import customfloat     # buildDecodeLUT, decodeArrayLUT
import session_vm      # bytecode emit / .bvm / sessionVmFor*
import jit_vm          # HotSpot-style: interpret -> profile -> Metal JIT cache

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
  let perResident = gProfEncode * 1000.0 / n
  result = &"[infer] decode profile over {gProfTokens - 1} tokens (first token excluded): " &
    &"runM3Resident={perResident:.1f}ms/tok\n" & gProfFirst

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
  ## Kept for the copied trainer API. All Mamba-3 projection rows are trainable.
  discard # Mamba-3 LoRA learns x/B/C/dt/A/trap/angles as well as the gate.

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
# Attach LoRA to the given weight tensors (by name in m3_hybrid's paramList,
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
  ## m3_loader.loadMamba3QuantResident). The shape comes FROM
  ## QuantTensor.shape (t[].data is not read, since at this point t[].data is still garbage/
  ## random init from newMamba3Real, NOT yet filled - it is filled ON EVERY
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
    # BUG FIX (see paramQuantFlags in m3_hybrid.nim): t[].data is now
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
  ## lora.quantBase (from loadMamba3QuantResidentFromNimq + the Q4 newBlock).
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
  for i in 0..<m.blocks.len:
    for (suffix,lin) in [("mlpFc1.weight",addr m.blocks[i].mambaReal.mlpFc1),("mlpFc2.weight",addr m.blocks[i].mambaReal.mlpFc2)]:
      let name="block" & $i & ".mambaReal." & suffix
      if name in lora.adapters: attachAdapter(lin[],lora.adapters[name])
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

# Mamba-3 decode owns angle/K/V/SSM state; never invokes the Mamba-2 fused kernels.
var gState: HybridGenState
var gPreparedLinears: seq[GpuLin]
var gUseNormF* = true
var gSkipAttn* = false
var gNormPresetName* = "rms-fin-gn"
proc setGpuDisableLora*(v: bool) =
  gDisableLora = v
proc setGpuNormPreset*(name: string): bool =
  result=name in ["hf","train","mamba-ssm","rms-fin-g1","rms-fin-gn"]
  if result: gNormPresetName="rms-fin-gn"
proc normPresetDescription*(m: NimformerModelJamba): string = "Mamba-3: RMS pre-norm + final RMS; B/C RMS; SiLU(z), no mixer output norm"
proc gNormGroups*(nGroups:int): int = nGroups
proc prepareGpuModel*(m: var NimformerModelJamba; lora: var JambaLora; ctx: Backend) =
  syncRmsFromBaseNorms(m)
  gState=newHybridGenState(m)
  if isGpuCtx(ctx):
    for name,lin in paramLinears(m):
      if lin[].inF<=0 or lin[].outF<=0: continue
      gPreparedLinears.add(prepareGpuLinForLinear(ctx,name,lin[]))
    prepareM3Runtime(m,ctx)
proc resetGpuGenState*(m: var NimformerModelJamba; ctx: Backend) =
  gState=newHybridGenState(m)
  if isGpuCtx(ctx):resetM3Runtime(ctx)
proc prefillResident*(m:var NimformerModelJamba;ids:seq[int];ctx:Backend)=
  if isGpuCtx(ctx):prefillM3Resident(m,ids,ctx)
  else:
    raise newException(ValueError,"resident prefill requires GPU")
proc forwardStepJambaGpuRetry*(m: var NimformerModelJamba; tokenId: int;
    tokenIds: seq[int]; tokenIndex: int; ctx: Backend; produceLogits=true): Tensor =
  if isGpuCtx(ctx):
    gM3Runtime.skipAttn=gSkipAttn
    let profile = getEnv("BYBY_DECODE_PROFILE", "0") != "0"
    let t0 = if profile: monoSecs() else: 0.0
    result = runM3Resident(m,@[tokenId],ctx,produceLogits)
    if profile:
      let elapsed = monoSecs() - t0
      inc gProfTokens
      gProfEncode += elapsed
      if gProfTokens == 1:
        gProfFirstMs = elapsed * 1000.0
        gProfFirst = &"[infer] first token runM3Resident={gProfFirstMs:.1f}ms"
    return
  if tokenId<0 or tokenId>=m.vocab: raise newException(ValueError,"token outside vocabulary")
  if gState.mambaStates.len!=m.blocks.len: gState=newHybridGenState(m)
  var x=m.embed.lookupBatch(@[@[tokenId]],ctx)
  for i in 0..<m.blocks.len:
    let xn=m.blocks[i].rms1.forward(x,ctx)
    let y=forwardStepMamba3Real(m.blocks[i].mambaReal,xn,gState.mambaStates[i],ctx)
    var mixed=y
    if m.blocks[i].kind in {bkMambaRealPlusAttn,bkMambaRealGated} and not gSkipAttn:
      let root="block" & $i & ".attn."
      let a=forwardLinearPlusInferStep(m.blocks[i].attn.qkvProj,m.blocks[i].attn.outProj,
        root & "qkvProj.weight",root & "outProj.weight",m.blocks[i].attn.nHeads,gState.attnStates[i],xn,ctx)
      if m.blocks[i].kind==bkMambaRealPlusAttn: mixed=addT(ctx,y,a)
      else:
        let gate=beSigmoid(ctx,m.blocks[i].fuseGate.forward(concatLastDim(y,a,ctx),ctx).data)
        mixed=newTensor(y.shape)
        for j in 0..<mixed.data.len: mixed.data[j]=gate[j]*y.data[j]+(1'f32-gate[j])*a.data[j]
    x=forwardMlp(m.blocks[i].mambaReal,addT(ctx,x,mixed),ctx).y
  if produceLogits: result=m.outProj.forward(m.rmsF.forward(x,ctx),ctx)
  else: result=newTensor(@[1,1,0])

type BlockStat* = M3BlockStat
proc forwardStepJambaGpuTraceRetry*(m: var NimformerModelJamba; tokenId:int;
    tokenIds:seq[int]; tokenIndex:int; ctx:Backend; stats:var seq[BlockStat]): Tensor =
  if not isGpuCtx(ctx):raise newException(ValueError,"Block GPU trace requires Metal/Vulkan")
  gM3Runtime.skipAttn=gSkipAttn
  result=runM3Resident(m,@[tokenId],ctx,trace=true)
  stats=gM3Runtime.lastTrace
proc lpSelfCheck*(ctx:Backend; blockIdx:int): string =
  discard blockIdx
  linearPlusSelfCheck(ctx)
