## mamba2_fused_step.nim
## ─────────────────────────────────────────────────────────────────────────
## "1 SESSION / TOKEN, 0 COPIES" — uses the NEW FUNCTIONS of mamba2_vm_kernels.nim
## (buildSplitConvSoftplusAt / buildMamba2StepAt / buildRmsGateGroupedAt).
##
## ONE LAYER = 7 GPU commands (the old encodeLayerStep in mamba2_ssm_gpu.nim had
## ~34 commands, ~27 of them sessionCopyInto - see BAO_CAO.md section 2):
##   1. rmsnorm_kernel   : pre-norm ln1        (hRes   -> hNorm)
##   2. jit_fp4km_gemv   : in_proj             (hNorm  -> hWork[0 .. P))
##   3. micro-VM fsc     : split+conv1d+softplus IN PLACE on hWork
##   4. micro-VM ssm     : SSM step IN PLACE on hWork (the state never moves)
##   5. micro-VM rmsg    : GROUPED gated RMSNorm in place on hWork
##   6. jit_fp4km_gemv   : out_proj            (hWork[offNormed] -> hMix)
##   7. vecop add        : hRes = hRes + hMix  (residual IN PLACE)
##
## Requirements (in this order, see PATCH.md):
##   1) session_vm.nim + jit_vm.nim + micro_vm.metal patched (OP_LOAD_LID/TGID)
##   2) the 3 *_At/*_Grouped functions added at the end of mamba2_vm_kernels.nim
##   3) lora.nim changed to call the functions here instead of the old encodeLayerStepByIdx

import std/[tables, os, strutils, bitops]
import backend
import vendor/bybylang/gpubackend as byby
import session_vm
import session_vm_dispatch
import jit_vm
import mamba2_vm_kernels
import fp4_gemv_gpu
import gpu_layers
import mamba2_ssm_gpu

const OffAlign = 64   ## 64 float = 256 byte. Vulkan minStorageBufferOffsetAlignment
                       ## is guaranteed ≤ 256 by the spec, so 256-byte alignment is
                       ## always safe without querying the device. Do NOT lower to
                       ## 32 (128 byte) — that can fail validation or read garbage.

proc alignUp(x, a: int): int {.inline.} = ((x + a - 1) div a) * a

type
  LayerOffV2* = object
    P*: int
    offZ*, offXBC*, offDt*: int
    offALog*, offDSkip*, offDtBias*, offNormW*: int
    offConvW*, offConvB*, offConvSt*: int
    offSsm*, offIdxH*, offIdxG*: int
    offY*, offNormed*: int
    total*: int

  Mamba2LayerV2* = object
    be*: byby.GpuBackend
    hWork*: byby.SessionHandle     ## the whole state + small weights of the layer
    hGamma*: byby.SessionHandle    ## ln1.gamma, upload 1 LAN doi process
    hGammaOnes*: byby.SessionHandle ## gamma = 1 (only with gRmsUnitGamma: reading checkpoints
                                    ## trained by the old trainer, rms1.gamma never loaded from the base)
    hBeta*: byby.SessionHandle     ## ln1.beta (only when useLn)
    useLn*: bool                   ## true: LayerNorm (matches training), false: RMSNorm (original Mamba2)
    hMix*: byby.SessionHandle      ## out_proj output [hidden]
    off*: LayerOffV2
    progFsc*, progSsm*, progRms*, progRms1*: VmProgram
    fscThreads*, ssmThreads*, rmsThreads*, rmsTg*, rmsThreads1*, rmsTg1*: int
    ssmTag*: string
    H*, D*, N*, G*, C*, cDim*, K*, HD*, hidden*, normGroups*: int
    eps*: float32
    ready*: bool

var gLayersV2* = initTable[int, Mamba2LayerV2]()

## RUNTIME override of the ln1 norm kind of R blocks (no re-prepare needed):
##   -1 = follows each layer's `useLn` flag (old behaviour)
##    0 = RMSNorm (Mamba2 goc / HF)
##    1 = LayerNorm with mean subtraction + beta (MATCHES finetune_lora: jamba_hybrid.forwardBlock
##        uses `blk.ln1.forward` = LayerNorm for R blocks too)
var gRNormOverride* = -1

## RUNTIME: the gamma of the pre-block RMSNorm (R and the P sidecar) and normF = 1 instead of the base gamma.
## ONLY for reading checkpoints trained by the OLD trainer (before the syncRmsFromBaseNorms fix): that trainer
## never loaded the pretrained gamma into rms1/rmsF, so the LoRA + sidecar learned on gamma=1
## normalised inputs. The "rms-fin-g1-unit" preset turns this flag on. Checkpoints trained by the
## new trainer (gamma synced) NEVER need it.
var gRmsUnitGamma* = false

## RUNTIME override of the group count of the gated RMSNorm in the R mixer:
##   -1 = follows the prepared group count (the model's n_groups = mamba_ssm/vLLM)
##    1 = 1 group over all of C (MATCHES mamba2_real.forwardMamba2Real in TRAINING:
##        varSum over all of C, the old HF transformers way)
var gRNormGroupsOverride* = -1
var gJitV2 = jit_vm.initJitCache(jit_vm.JitConfig(
  hotThreshold: 0, enableMetalJit: true, enableVulkanJit: true,
  enableCpuJit: false, maxCacheEntries: 128))

proc computeOffV2*(C, cDim, H, D, N, G, K: int): LayerOffV2 =
  let HD = H * D
  result.P         = C + cDim + H
  result.offZ      = alignUp(result.P, OffAlign)
  result.offXBC    = alignUp(result.offZ + C, OffAlign)
  result.offDt     = alignUp(result.offXBC + cDim, OffAlign)
  result.offALog   = alignUp(result.offDt + H, OffAlign)
  result.offDSkip  = alignUp(result.offALog + H, OffAlign)
  result.offDtBias = alignUp(result.offDSkip + H, OffAlign)
  result.offNormW  = alignUp(result.offDtBias + H, OffAlign)
  result.offConvW  = alignUp(result.offNormW + C, OffAlign)
  result.offConvB  = alignUp(result.offConvW + cDim * K, OffAlign)
  result.offConvSt = alignUp(result.offConvB + cDim, OffAlign)
  result.offSsm    = alignUp(result.offConvSt + cDim * max(K - 1, 0), OffAlign)
  result.offIdxH   = alignUp(result.offSsm + HD * N, OffAlign)
  result.offIdxG   = alignUp(result.offIdxH + HD, OffAlign)
  result.offY      = alignUp(result.offIdxG + HD, OffAlign)
  result.offNormed = alignUp(result.offY + max(C, HD), OffAlign)
  result.total     = alignUp(result.offNormed + C, OffAlign)

proc pickRmsTg(groupLen: int): int =
  ## Threads per threadgroup of the gated-RMSNorm kernel. Default 256: a
  ## maxTotalThreadsPerThreadgroup value safe on EVERY GPU (including Intel iGPUs).
  ## Raise it through BYBY_RMS_TG if desired.
  var t = 256
  let envTg = getEnv("BYBY_RMS_TG")
  if envTg.len > 0:
    try:
      let v = parseInt(envTg)
      if v > 0: t = v
    except ValueError: discard
  ## Closed form replacing `while t > ... : t = t div 2`: the largest power of 2
  ## <= min(t, groupLen) that divides groupLen (bit trick, no loop).
  let start = t
  t = min(groupLen and (-groupLen), 1 shl fastLog2(max(1, min(start, groupLen))))
  if t != start and t < 32:
    raise newException(CatchableError,
      "pickRmsTg: no valid tgSize found for groupLen=" & $groupLen)
  t

# ════════════════════════════════════════════════════════════════════
# micro-VM encode (EXPLICIT tgSize — unlike the old encodeMicroVm in
# mamba2_ssm_gpu.nim, which picked tg = min(nThreads,1024) itself and so could not
# be used for multi-threadgroup kernels + threadgroup memory).
# ════════════════════════════════════════════════════════════════════
proc encodeVm*(program: VmProgram; hMem: byby.SessionHandle;
               nThreads, tgSize, tgBytes: int; tag: string) =
  ## Micro-VM 1 buffer (fsc/ssm/rmsg). API chung: sessionEnc* route Metal/Vulkan.
  let isVk = hMem.kind == byby.gbVulkan
  let kname =
    if isVk: jit_vm.kernelNameFor(gJitV2, program, tag)
    else: jit_vm.dispatchProgramNoSession(gJitV2, program, tag)
  if kname.len == 0 or kname == "micro_vm_kernel":
    raise newException(CatchableError,
      "encodeVm[" & tag & "]: " & (if isVk: "Vulkan" else: "Metal") & " JIT FAILED.")
  let glslSrc = if isVk: jit_vm.cachedGlslSource(gJitV2, kname) else: ""
  if isVk and glslSrc.len == 0:
    raise newException(CatchableError,
      "encodeVm[" & tag & "]: no GLSL source (kernel=" & kname & ")")
  if isVk:
    if hMem.vulkanHandle.buf == nil:
      raise newException(CatchableError, "encodeVm[" & tag & "]: vulkanHandle.buf nil")
  elif hMem.metalHandle == nil:
    raise newException(CatchableError, "encodeVm[" & tag & "]: metalHandle nil")

  if not sessionEncPipeline(isVk, kname, glslSrc):
    raise newException(CatchableError, "encodeVm: cannot bind pipeline " & kname)

  # Metal slots 1..3; Vulkan GLSL bindings 0..3 — the same hWork
  let base = if isVk: 0 else: 1
  for s in 0 .. 2:
    discard sessionEncBindVkOrMetal(isVk, base + s, hMem.metalHandle, hMem.vulkanHandle, 0)
  if isVk:
    discard sessionEncBindVkOrMetal(true, 3, hMem.metalHandle, hMem.vulkanHandle, 0)

  discard sessionEncTgBytes(isVk, 0, max(tgBytes, 4))
  var tg = tgSize
  if tgBytes > 4:
    let maxTG = sessionEncMaxTg(isVk)
    if maxTG > 0 and tg > maxTG:
      raise newException(CatchableError,
        "encodeVm[" & tag & "]: needs " & $tg & " tg but max=" & $maxTG)
    if nThreads mod tg != 0:
      raise newException(CatchableError,
        "encodeVm[" & tag & "]: nThreads is not divisible by tgSize")
  else:
    tg = min(tg, max(nThreads, 1))
    let maxTG = sessionEncMaxTg(isVk)
    if maxTG > 0 and tg > maxTG: tg = maxTG
  if not sessionEncDispatch(isVk, nThreads, tg, tgBytes):
    raise newException(CatchableError, "encodeVm[" & tag & "]: dispatch failed")

# ════════════════════════════════════════════════════════════════════
# fp4 GEMV with an OFFSET on both input and output — reads/writes DIRECTLY
# into the middle of the hWork buffer, no intermediate buffer + copy.
# ════════════════════════════════════════════════════════════════════
proc fp4GemvAt*(ctx: Backend; name: string;
                hIn: byby.SessionHandle; inOffElems: int;
                hOut: byby.SessionHandle; outOffElems: int; M = 1): bool =
  ## A legacy name (backward compatible): the weight may be fp4-KM OR int4-asymmetric
  ## (see gpu_layers.prepareGpuLinForLinear) - encodeLinByName picks the right one.
  doAssert M == 1, "fp4GemvAt: supports only M=1"
  result = gpu_layers.encodeLinByName(name, hIn, inOffElems, hOut, outOffElems)
  if not result:
    raise newException(CatchableError, "fp4GemvAt: prepareGpuLinForLinear not called for " & name)

# ════════════════════════════════════════════════════════════════════
proc initLayerV2*(ctx: Backend; H, D, N, G, C, cDim, K, hidden: int;
                  aLog, dSkip, dtBias, normW, convW, convB, ln1Gamma: seq[float32];
                  normGroups: int; eps: float32;
                  ln1Beta: seq[float32] = @[]; useLayerNorm = false): Mamba2LayerV2 =
  let be = ctx.kind.toByby()
  let HD = H * D
  let o = computeOffV2(C, cDim, H, D, N, G, K)

  var buf = newSeq[float32](o.total)
  buf.blit(o.offALog, aLog, 0, H)
  buf.blit(o.offDSkip, dSkip, 0, H)
  buf.blit(o.offDtBias, dtBias, 0, H)
  buf.blit(o.offNormW, normW, 0, C)
  buf.blit(o.offConvW, convW, 0, convW.len)
  buf.blit(o.offConvB, convB, 0, convB.len)
  # idxH/idxG generated on the GPU (buildIdxTableProgram), then copied into the layout
  var tab = newSeq[float32](2 * HD)
  discard runMicroVm(ctx, buildIdxTableProgram(H, D, G), tab, HD, 4, "idxtab")
  buf.blit(o.offIdxH, tab, 0, HD)
  buf.blit(o.offIdxG, tab, HD, HD)
  # offConvSt / offSsm stay 0 = the start-of-sequence state (s_{-1}=0, conv=0)

  result.be = be
  result.off = o
  result.H = H; result.D = D; result.N = N; result.G = G
  result.C = C; result.cDim = cDim; result.K = K; result.HD = HD
  result.hidden = hidden
  result.normGroups = normGroups
  result.eps = eps
  result.hWork  = byby.sessionUploadPrivate(be, buf)
  result.hGamma = byby.sessionUploadPrivate(be, ln1Gamma)
  block:
    var ones = newSeq[float32](ln1Gamma.len)
    for i in 0 ..< ones.len: ones[i] = 1'f32
    result.hGammaOnes = byby.sessionUploadPrivate(be, ones)
  result.useLn = useLayerNorm
  # ALWAYS upload beta (default 0): allows switching RMSNorm <-> LayerNorm at run time
  # through gRNormOverride without re-preparing the layer.
  result.hBeta = byby.sessionUploadPrivate(be,
    (if ln1Beta.len == ln1Gamma.len: ln1Beta else: newSeq[float32](ln1Gamma.len)))
  result.hMix   = sessionAllocScratch(be, hidden)

  result.progFsc = buildSplitConvSoftplusAt(C, cDim, H, K,
    o.offZ, o.offXBC, o.offDt, o.offConvW, o.offConvB, o.offConvSt, o.offDtBias)
  result.progSsm = buildMamba2StepAt(H, D, N, G,
    o.offXBC, o.offXBC + C, o.offXBC + C + G * N, o.offDt,
    o.offALog, o.offDSkip, o.offSsm, o.offY, o.offIdxH, o.offIdxG)
  result.rmsTg = pickRmsTg(C div normGroups)
  result.progRms = buildRmsGateGroupedAt(C, normGroups, result.rmsTg, eps,
    o.offY, o.offZ, o.offNormW, o.offNormed)

  # The 1-GROUP gated-RMSNorm program (matching training) - prebuilt for switching at run time.
  if normGroups == 1:
    result.progRms1 = result.progRms
    result.rmsTg1 = result.rmsTg
  else:
    result.rmsTg1 = pickRmsTg(C)
    result.progRms1 = buildRmsGateGroupedAt(C, 1, result.rmsTg1, eps,
      o.offY, o.offZ, o.offNormW, o.offNormed)
  result.rmsThreads1 = 1 * result.rmsTg1

  result.fscThreads = max(C, max(cDim, H))
  result.ssmThreads = HD
  result.rmsThreads = normGroups * result.rmsTg
  result.ssmTag = "ssm_" & $H & "_" & $D & "_" & $N & "_" & $G
  result.ready = true

proc ensureLayerV2*(ctx: Backend; blockIdx, H, D, N, G, C, cDim, K, hidden: int;
                    aLog, dSkip, dtBias, normW, convW, convB, ln1Gamma: seq[float32];
                    normGroups: int; eps: float32;
                    ln1Beta: seq[float32] = @[]; useLayerNorm = false) =
  if blockIdx notin gLayersV2:
    gLayersV2[blockIdx] = initLayerV2(ctx, H, D, N, G, C, cDim, K, hidden,
      aLog, dSkip, dtBias, normW, convW, convB, ln1Gamma, normGroups, eps,
      ln1Beta, useLayerNorm)

var gZeroBuf: byby.SessionHandle
var gZeroLen = 0

## Hook for onedispatch.nim (no reverse import possible: onedispatch imports this module).
## With 1-dispatch mode on, the conv/ssm state lives in the pack buffer -> the reset goes through the hook.
var gOdResetHook*: proc (blockIdx: int): bool {.nimcall.} = nil
## Blocks in OD (R); G/other blocks still use the old state path.
var gOdOwnsHook*: proc (blockIdx: int): bool {.nimcall.} = nil

proc resetLayerStateV2*(ctx: Backend; blockIdx: int): bool =
  ## Zeroes convState + ssmState (start of a new sequence / after warmup) - like
  ## the old resetGpuLayerState (fix for bug #12: the warmup ran on the REAL layer itself,
  ## so the state was dirtied before prefill started). MUST be called inside an
  ## open session. The zero buffer is CACHED (no sessionFree): the open command
  ## buffer still references it, and the next reset can reuse it.
  if not gOdResetHook.isNil and (gOdOwnsHook.isNil or gOdOwnsHook(blockIdx)):
    return gOdResetHook(blockIdx)
  if blockIdx notin gLayersV2: return false
  let lp = addr gLayersV2[blockIdx]
  template l: untyped = lp[]
  if not l.ready: return false
  let n = max(l.cDim * max(l.K - 1, 0), l.HD * l.N)
  if n <= 0: return true
  if gZeroLen < n:
    gZeroBuf = byby.sessionUploadPrivate(l.be, newSeq[float32](n))
    gZeroLen = n
  result = true
  if not sessionCopyInto(l.be, gZeroBuf, l.hWork, 0, l.off.offConvSt, l.cDim * (l.K - 1)):
    result = false
  if not sessionCopyInto(l.be, gZeroBuf, l.hWork, 0, l.off.offSsm, l.HD * l.N):
    result = false

# ════════════════════════════════════════════════════════════════════
# ONE LAYER = 7 commands, 0 COPIES. hRes is overwritten IN PLACE (in-place residual).
# ════════════════════════════════════════════════════════════════════
proc layerUsesLayerNorm*(blockIdx: int): bool =
  ## Whether the pre-block norm of R/P block `blockIdx` IS LayerNorm (true) or RMSNorm (false).
  ## The single source of truth for encodeLayerV2 (the R branch) AND encodeLinearPlusBranch
  ## (the P sidecar): the sidecar used to always use LayerNorm whatever the preset while the R branch followed the
  ## preset -> at inference the sidecar got an input of a different scale than in training.
  if gRNormOverride == 1: return true
  if gRNormOverride == 0: return false
  if blockIdx in gLayersV2: return gLayersV2[blockIdx].useLn
  false

## `inputPreNormalized=true` is reserved for a parallel P block whose sidecar
## has just written the exact same pre-norm result into `hNorm`.
proc encodeLayerV2*(ctx: Backend; blockIdx: int;
                    hRes: byby.SessionHandle;        # [hidden] in + out (in-place)
                    hNorm: var byby.SessionHandle;   # [hidden] scratch shared by every layer
                    fp4InName, fp4OutName: string;
                    inputPreNormalized = false;
                    addResidual = true) =
  doAssert blockIdx in gLayersV2, "encodeLayerV2: ensureLayerV2(" & $blockIdx & ") not called"
  # `addr`, not `var l = gLayersV2[i]`: Mamba2LayerV2 holds 3
  # VmPrograms (seq[uint32]) - a value assignment would DEEP COPY all 3 seqs on every call
  # (64 layers x every token) with `var`.
  let lp = addr gLayersV2[blockIdx]
  template l: untyped = lp[]
  doAssert l.ready
  let be = l.be
  let o = l.off

  # 1) pre-norm (ln1): RMSNorm (original Mamba2) or LayerNorm (matches finetune_lora)
  #    The LN/RMS decision lives in layerUsesLayerNorm - the P sidecar branch calls the SAME function,
  #    because in training both branches read the SAME normalised output (x1 = rms1(x)).
  if not inputPreNormalized:
    if layerUsesLayerNorm(blockIdx):
      if not sessionLayernormEnc(be, hRes, l.hGamma, l.hBeta, hNorm, 1, l.hidden, l.eps):
        raise newException(CatchableError, "encodeLayerV2: layernorm ln1 failed")
    else:
      let gam = if gRmsUnitGamma: l.hGammaOnes else: l.hGamma
      if not sessionRmsnormEnc(be, hRes, gam, hNorm, 1, l.hidden, l.eps):
        raise newException(CatchableError, "encodeLayerV2: rmsnorm ln1 failed")

  # 2) in_proj -> hWork[0 .. P)   (+ the in_proj LoRA delta BEFORE conv/SSM: W' = W + BA)
  # NOTE: fp4GemvAt -> encodeLinByName -> encodeLin ALREADY adds the LoRA delta inside
  # (gpu_layers.encodeLin). Do NOT call encodeLoraDelta again here: lora_b accumulates,
  # so calling it twice = the LoRA added TWICE (scale x2) - a silent error.
  if not fp4GemvAt(ctx, fp4InName, hNorm, 0, l.hWork, 0, 1):
    raise newException(CatchableError, "encodeLayerV2: fp4 in_proj failed")

  # 3) split z/xBC/dt + conv1d (slides convState in place) + softplus(dt)
  encodeVm(l.progFsc, l.hWork, l.fscThreads, 256, 4, "fsc2")

  # 4) SSM step — reads/writes ssmState IN PLACE, no copies
  encodeVm(l.progSsm, l.hWork, l.ssmThreads, 256, 4, l.ssmTag)

  # 5) grouped gated RMSNorm
  if gRNormGroupsOverride == 1 and l.normGroups != 1:
    encodeVm(l.progRms1, l.hWork, l.rmsThreads1, l.rmsTg1, l.rmsTg1 * 4, "rmsg1")
  else:
    encodeVm(l.progRms, l.hWork, l.rmsThreads, l.rmsTg, l.rmsTg * 4, "rmsg2")

  # 6) out_proj: reads DIRECTLY from hWork[offNormed]
  # (the out_proj LoRA delta is already in encodeLin, activation = hWork[offNormed .. +C))
  if not fp4GemvAt(ctx, fp4OutName, l.hWork, o.offNormed, l.hMix, 0, 1):
    raise newException(CatchableError, "encodeLayerV2: fp4 out_proj failed")

  # 7) Residual can be deferred by the composite P-block encoder. That lets it
  # preserve the trainer's exact order: combined = mambaOut + attnOut; x += combined.
  if addResidual:
    var hDst = hRes
    if not sessionVecOp(be, "add", hRes, l.hMix, hDst, l.hidden):
      raise newException(CatchableError, "encodeLayerV2: residual add failed")

## Merge helper shared by normal decode and the diagnostic trace path.
proc encodeMamba2RealLinearPlusResidual*(ctx: Backend; blockIdx: int;
                                          hRes: byby.SessionHandle;
                                          hCombined: var byby.SessionHandle;
                                          hAttnOut: byby.SessionHandle) =
  doAssert blockIdx in gLayersV2,
    "encodeMamba2RealLinearPlusResidual: Mamba2 layer not prepared"
  let ml = addr gLayersV2[blockIdx]
  # hMix holds mambaOut. hCombined is the now-dead pre-norm scratch, so write
  # the sum there without aliasing either branch output.
  if not sessionVecOp(ctx.kind.toByby(), "add", ml[].hMix, hAttnOut,
      hCombined, ml[].hidden):
    raise newException(CatchableError,
      "encodeMamba2RealLinearPlusResidual: branch combine failed")
  var hDst = hRes
  if not sessionVecOp(ctx.kind.toByby(), "add", hRes, hCombined, hDst, ml[].hidden):
    raise newException(CatchableError,
      "encodeMamba2RealLinearPlusResidual: residual add failed")

## Composite micro-kernel program for one P block (Mamba2Real + Linear++).
## This is a block-level schedule of specialized GPU kernels in one session,
## not one Metal dispatch: Mamba and Linear++ have dependent recurrent stages
## that need dispatch boundaries on the supported GPUs.
## The two branch outputs are mathematically parallel, but this encoder queues
## their stateful dispatches in order; it does not launch concurrent streams.
proc encodeMamba2RealLinearPlusBlock*(ctx: Backend; blockIdx: int;
                                      hRes: byby.SessionHandle;
                                      hNorm: var byby.SessionHandle;
                                      skipLinearPlus = false;
                                      unitGamma = false;
                                      gated = false) =
  doAssert blockIdx in gLayersV2,
    "encodeMamba2RealLinearPlusBlock: Mamba2 layer not prepared"
  let inName = "block" & $blockIdx & ".mambaReal.inProj.weight"
  let outName = "block" & $blockIdx & ".mambaReal.outProj.weight"

  if skipLinearPlus:
    encodeLayerV2(ctx, blockIdx, hRes, hNorm, inName, outName)
    return

  doAssert blockIdx in gpu_layers.gAttnLayers,
    "encodeMamba2RealLinearPlusBlock: Linear++ layer not prepared"
  let attnLayer = addr gpu_layers.gAttnLayers[blockIdx]
  doAssert attnLayer[].gated == gated,
    "encodeMamba2RealLinearPlusBlock: gate configuration differs from prepared layer"
  let useRms = not layerUsesLayerNorm(blockIdx)
  gpu_layers.encodeLinearPlusBranch(ctx, blockIdx, hRes, hNorm,
    gpu_layers.gAttnScratch.hAttn, useRms = useRms, unitGamma = unitGamma)

  # Both branches read exactly the same normalized input. The sidecar encoder
  # above produced it in hNorm; Mamba must not normalize again.
  encodeLayerV2(ctx, blockIdx, hRes, hNorm, inName, outName,
    inputPreNormalized = true, addResidual = false)

  if gated:
    let be = ctx.kind.toByby()
    let ml = addr gLayersV2[blockIdx]
    var hGateInput = attnLayer[].hGateInput
    var hGateLogit = attnLayer[].hGateLogit
    var hGateProb = attnLayer[].hGateProb
    var hGateComplement = attnLayer[].hGateComplement
    var hMamba = ml[].hMix
    var hAttn = gpu_layers.gAttnScratch.hAttn
    var hResidual = hRes
    let gDiag = getEnv("BYBY_G_DIAGNOSTICS", "0") != "0"
    if gDiag:
      stderr.writeLine "[G] block=" & $blockIdx & " ====== fuseGate DETAILS ======"
      stderr.writeLine "[G] block=" & $blockIdx & " (1) mambaOut = hMix  |  attnOut = hAttn  (same RMS(x))"
      stderr.writeLine "[G] block=" & $blockIdx & " (2) gate_in = concat(mambaOut, attnOut) size 2*H"
    if not sessionCopyInto(be, ml[].hMix, hGateInput, 0, 0, ml[].hidden) or
       not sessionCopyInto(be, hAttn, hGateInput, 0, ml[].hidden, ml[].hidden):
      raise newException(CatchableError, "G block: cannot assemble the fuseGate input")
    if gDiag:
      stderr.writeLine "[G] block=" & $blockIdx & " (3) logit = fuseGate(gate_in)  [Linear H*2 -> H]"
    gpu_layers.encodeLin(attnLayer[].gate, hGateInput, 0, hGateLogit, 0)
    if gDiag:
      stderr.writeLine "[G] block=" & $blockIdx & " (4) g = sigmoid(logit)  |  (1-g) = 1 - g"
    if not sessionActivation(be, "sigmoid", hGateLogit, hGateProb, ml[].hidden) or
       not sessionVecOp(be, "sub", attnLayer[].hGateOnes, hGateProb,
                        hGateComplement, ml[].hidden):
      raise newException(CatchableError, "G block: fuseGate sigmoid/sub failed")
    # Snapshot BEFORE the in-place mul (numbers read after sessionEnd)
    if gDiag:
      let needSnap =
        (be == byby.gbVulkan and attnLayer[].hDiagMamba.vulkanHandle.buf == nil) or
        (be == byby.gbMetal and attnLayer[].hDiagMamba.metalHandle == nil)
      if needSnap:
        attnLayer[].hDiagMamba = byby.sessionAllocScratch(be, ml[].hidden)
        attnLayer[].hDiagAttn = byby.sessionAllocScratch(be, ml[].hidden)
      if not sessionCopyInto(be, hMamba, attnLayer[].hDiagMamba, 0, 0, ml[].hidden) or
         not sessionCopyInto(be, hAttn, attnLayer[].hDiagAttn, 0, 0, ml[].hidden):
        raise newException(CatchableError, "G block: mamba/attn snapshot failed")
      stderr.writeLine "[G] block=" & $blockIdx & " (5) mix = g*mamba + (1-g)*attn  (in place on hMix)"
      stderr.writeLine "[G] block=" & $blockIdx & " (6) x = x + mix  (residual)"
    if not sessionVecOp(be, "mul", hGateProb, hMamba, hMamba, ml[].hidden) or
       not sessionVecOp(be, "mul", hGateComplement, hAttn, hAttn, ml[].hidden) or
       not sessionVecOp(be, "add", hMamba, hAttn, hMamba, ml[].hidden) or
       not sessionVecOp(be, "add", hRes, hMamba, hResidual, ml[].hidden):
      raise newException(CatchableError, "G block: fuseGate convex mix failed")
    if gDiag:
      stderr.writeLine "[G] block=" & $blockIdx & " fuse queued OK (numbers printed after sessionEnd)"
  else:
    encodeMamba2RealLinearPlusResidual(ctx, blockIdx, hRes, hNorm,
      gpu_layers.gAttnScratch.hAttn)
