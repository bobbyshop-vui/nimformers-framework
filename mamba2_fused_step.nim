## mamba2_fused_step.nim
## ─────────────────────────────────────────────────────────────────────────
## "1 SESSION / TOKEN, 0 COPY" — dung CAC HAM MOI trong mamba2_vm_kernels.nim
## (buildSplitConvSoftplusAt / buildMamba2StepAt / buildRmsGateGroupedAt).
##
## MOT LAYER = 7 lenh GPU (encodeLayerStep cu trong mamba2_ssm_gpu.nim co
## ~34 lenh, trong do ~27 la sessionCopyInto - xem BAO_CAO.md muc 2):
##   1. rmsnorm_kernel   : pre-norm ln1        (hRes   -> hNorm)
##   2. jit_fp4km_gemv   : in_proj             (hNorm  -> hWork[0 .. P))
##   3. micro-VM fsc     : split+conv1d+softplus TAI CHO tren hWork
##   4. micro-VM ssm     : SSM step TAI CHO tren hWork (state khong di dau)
##   5. micro-VM rmsg    : gated RMSNorm THEO NHOM tai cho tren hWork
##   6. jit_fp4km_gemv   : out_proj            (hWork[offNormed] -> hMix)
##   7. vecop add        : hRes = hRes + hMix  (residual TAI CHO)
##
## Yeu cau (theo dung thu tu, xem PATCH.md):
##   1) da va session_vm.nim + jit_vm.nim + micro_vm.metal (OP_LOAD_LID/TGID)
##   2) da them 3 ham *_At/*_Grouped vao cuoi mamba2_vm_kernels.nim
##   3) sua lora.nim de goi cac ham o day thay vi encodeLayerStepByIdx cu

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
    hWork*: byby.SessionHandle     ## toan bo trang thai + weight nho cua layer
    hGamma*: byby.SessionHandle    ## ln1.gamma, upload 1 LAN doi process
    hGammaOnes*: byby.SessionHandle ## gamma = 1 (chi dung khi gRmsUnitGamma: doc checkpoint
                                    ## train bang trainer cu, rms1.gamma chua bao gio nap tu base)
    hBeta*: byby.SessionHandle     ## ln1.beta (chi dung khi useLn)
    useLn*: bool                   ## true: LayerNorm (khop luc train), false: RMSNorm (Mamba2 goc)
    hMix*: byby.SessionHandle      ## out_proj output [hidden]
    off*: LayerOffV2
    progFsc*, progSsm*, progRms*, progRms1*: VmProgram
    fscThreads*, ssmThreads*, rmsThreads*, rmsTg*, rmsThreads1*, rmsTg1*: int
    ssmTag*: string
    H*, D*, N*, G*, C*, cDim*, K*, HD*, hidden*, normGroups*: int
    eps*: float32
    ready*: bool

var gLayersV2* = initTable[int, Mamba2LayerV2]()

## Ghi de RUNTIME kieu norm ln1 cua khoi R (khong can prepare lai):
##   -1 = theo co `useLn` cua tung layer (hanh vi cu)
##    0 = RMSNorm (Mamba2 goc / HF)
##    1 = LayerNorm co tru mean + beta (KHOP finetune_lora: jamba_hybrid.forwardBlock
##        dung `blk.ln1.forward` = LayerNorm cho ca khoi R)
var gRNormOverride* = -1

## RUNTIME: gamma cua RMSNorm tien-block (R lan sidecar P) va normF = 1 thay vi gamma base.
## CHI de doc checkpoint train bang trainer CU (truoc ban sua syncRmsFromBaseNorms): trainer
## do khong bao gio nap gamma pretrain vao rms1/rmsF nen LoRA + sidecar hoc tren dau vao
## chuan hoa gamma=1. Preset "rms-fin-g1-unit" bat co nay. Checkpoint train bang trainer
## moi (da dong bo gamma) KHONG bao gio can no.
var gRmsUnitGamma* = false

## Ghi de RUNTIME so nhom cua gated-RMSNorm trong mixer R:
##   -1 = theo so nhom da prepare (n_groups cua model = mamba_ssm/vLLM)
##    1 = 1 nhom tren toan bo C (KHOP mamba2_real.forwardMamba2Real luc TRAIN:
##        varSum tinh tren ca C, dung kieu HF transformers cu)
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
  ## So thread/threadgroup cho kernel gated-RMSNorm. Mac dinh 256: gia tri
  ## maxTotalThreadsPerThreadgroup an toan tren MOI GPU (ke ca iGPU Intel).
  ## Tang qua BYBY_RMS_TG neu muon.
  var t = 256
  let envTg = getEnv("BYBY_RMS_TG")
  if envTg.len > 0:
    try:
      let v = parseInt(envTg)
      if v > 0: t = v
    except ValueError: discard
  ## Công thức đóng thay cho `while t > ... : t = t div 2`: luỹ thừa 2 lớn nhất
  ## <= min(t, groupLen) và chia hết groupLen (bit trick, không vòng lặp).
  let start = t
  t = min(groupLen and (-groupLen), 1 shl fastLog2(max(1, min(start, groupLen))))
  if t != start and t < 32:
    raise newException(CatchableError,
      "pickRmsTg: khong tim duoc tgSize hop le cho groupLen=" & $groupLen)
  t

# ════════════════════════════════════════════════════════════════════
# encode micro-VM (tgSize TUONG MINH — khac encodeMicroVm cu trong
# mamba2_ssm_gpu.nim vi ban cu tu chon tg = min(nThreads,1024) nen khong
# dung duoc cho kernel nhieu threadgroup + threadgroup memory).
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
      "encodeVm[" & tag & "]: JIT " & (if isVk: "Vulkan" else: "Metal") & " THAT BAI.")
  let glslSrc = if isVk: jit_vm.cachedGlslSource(gJitV2, kname) else: ""
  if isVk and glslSrc.len == 0:
    raise newException(CatchableError,
      "encodeVm[" & tag & "]: khong co GLSL source (kernel=" & kname & ")")
  if isVk:
    if hMem.vulkanHandle.buf == nil:
      raise newException(CatchableError, "encodeVm[" & tag & "]: vulkanHandle.buf nil")
  elif hMem.metalHandle == nil:
    raise newException(CatchableError, "encodeVm[" & tag & "]: metalHandle nil")

  if not sessionEncPipeline(isVk, kname, glslSrc):
    raise newException(CatchableError, "encodeVm: khong bind duoc pipeline " & kname)

  # Metal slots 1..3; Vulkan GLSL binding 0..3 — cung hWork
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
        "encodeVm[" & tag & "]: can " & $tg & " tg nhung max=" & $maxTG)
    if nThreads mod tg != 0:
      raise newException(CatchableError,
        "encodeVm[" & tag & "]: nThreads khong chia het tgSize")
  else:
    tg = min(tg, max(nThreads, 1))
    let maxTG = sessionEncMaxTg(isVk)
    if maxTG > 0 and tg > maxTG: tg = maxTG
  if not sessionEncDispatch(isVk, nThreads, tg, tgBytes):
    raise newException(CatchableError, "encodeVm[" & tag & "]: dispatch that bai")

# ════════════════════════════════════════════════════════════════════
# fp4 GEMV voi OFFSET tren ca input lan output — cho phep doc/ghi THANG
# vao giua buffer hWork, khong can buffer trung gian + copy.
# ════════════════════════════════════════════════════════════════════
proc fp4GemvAt*(ctx: Backend; name: string;
                hIn: byby.SessionHandle; inOffElems: int;
                hOut: byby.SessionHandle; outOffElems: int; M = 1): bool =
  ## Ten gia (tuong thich nguoc): weight co the la fp4-KM HOAC int4-asymmetric
  ## (xem gpu_layers.prepareGpuLinForLinear) - encodeLinByName tu chon dung.
  doAssert M == 1, "fp4GemvAt: chi ho tro M=1"
  result = gpu_layers.encodeLinByName(name, hIn, inOffElems, hOut, outOffElems)
  if not result:
    raise newException(CatchableError, "fp4GemvAt: chua prepareGpuLinForLinear cho " & name)

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
  # idxH/idxG sinh bằng GPU (buildIdxTableProgram) rồi chép vào layout
  var tab = newSeq[float32](2 * HD)
  discard runMicroVm(ctx, buildIdxTableProgram(H, D, G), tab, HD, 4, "idxtab")
  buf.blit(o.offIdxH, tab, 0, HD)
  buf.blit(o.offIdxG, tab, HD, HD)
  # offConvSt / offSsm giu nguyen 0 = trang thai dau chuoi (s_{-1}=0, conv=0)

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
  # LUON upload beta (mac dinh 0): cho phep doi RMSNorm <-> LayerNorm luc chay
  # qua gRNormOverride ma khong phai prepare lai layer.
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

  # Chuong trinh gated-RMSNorm 1 NHOM (khop luc train) - build san de doi luc chay.
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

## Hook cho onedispatch.nim (khong import nguoc duoc: onedispatch import module nay).
## Khi che do 1-dispatch bat, state conv/ssm nam trong buffer pack -> reset di qua hook.
var gOdResetHook*: proc (blockIdx: int): bool {.nimcall.} = nil
## Block nao thuoc OD (R); block G/khac van dung state path cu.
var gOdOwnsHook*: proc (blockIdx: int): bool {.nimcall.} = nil

proc resetLayerStateV2*(ctx: Backend; blockIdx: int): bool =
  ## Xoa convState + ssmState ve 0 (bat dau chuoi moi / sau warmup) - giong
  ## y resetGpuLayerState cu (fix bug #12: warmup chay tren CHINH layer that
  ## nen state bi ghi ban truoc khi prefill bat dau). PHAI goi trong mot
  ## session dang mo. Buffer zero duoc CACHE (khong sessionFree): command
  ## buffer dang mo con tham chieu toi no, va lan reset sau dung lai duoc.
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
# MOT LAYER = 7 lenh, 0 COPY. hRes bi ghi de TAI CHO (residual in-place).
# ════════════════════════════════════════════════════════════════════
proc layerUsesLayerNorm*(blockIdx: int): bool =
  ## Norm tien-block cua khoi R/P `blockIdx` LA LayerNorm (true) hay RMSNorm (false).
  ## MOT nguon su that duy nhat cho encodeLayerV2 (nhanh R) VA encodeLinearPlusBranch
  ## (sidecar P): truoc day sidecar luon LayerNorm bat ke preset trong khi nhanh R theo
  ## preset -> infer cap cho sidecar dau vao khac hang so voi luc train.
  if gRNormOverride == 1: return true
  if gRNormOverride == 0: return false
  if blockIdx in gLayersV2: return gLayersV2[blockIdx].useLn
  false

## `inputPreNormalized=true` is reserved for a parallel P block whose sidecar
## has just written the exact same pre-norm result into `hNorm`.
proc encodeLayerV2*(ctx: Backend; blockIdx: int;
                    hRes: byby.SessionHandle;        # [hidden] vao + ra (in-place)
                    hNorm: var byby.SessionHandle;   # [hidden] scratch dung chung moi layer
                    fp4InName, fp4OutName: string;
                    inputPreNormalized = false;
                    addResidual = true) =
  doAssert blockIdx in gLayersV2, "encodeLayerV2: chua ensureLayerV2(" & $blockIdx & ")"
  # `addr` chu khong phai `var l = gLayersV2[i]`: Mamba2LayerV2 chua 3
  # VmProgram (seq[uint32]) - gan-gia-tri se DEEP COPY ca 3 seq moi lan goi
  # (64 layer x moi token) neu dung `var`.
  let lp = addr gLayersV2[blockIdx]
  template l: untyped = lp[]
  doAssert l.ready
  let be = l.be
  let o = l.off

  # 1) pre-norm (ln1): RMSNorm (Mamba2 goc) hoac LayerNorm (khop finetune_lora)
  #    Quyet dinh LN/RMS nam o layerUsesLayerNorm - nhanh sidecar P goi CUNG ham do,
  #    vi luc train ca 2 nhanh doc CHUNG 1 dau ra chuan hoa (x1 = rms1(x)).
  if not inputPreNormalized:
    if layerUsesLayerNorm(blockIdx):
      if not sessionLayernormEnc(be, hRes, l.hGamma, l.hBeta, hNorm, 1, l.hidden, l.eps):
        raise newException(CatchableError, "encodeLayerV2: layernorm ln1 that bai")
    else:
      let gam = if gRmsUnitGamma: l.hGammaOnes else: l.hGamma
      if not sessionRmsnormEnc(be, hRes, gam, hNorm, 1, l.hidden, l.eps):
        raise newException(CatchableError, "encodeLayerV2: rmsnorm ln1 that bai")

  # 2) in_proj -> hWork[0 .. P)   (+ delta LoRA in_proj TRUOC conv/SSM: W' = W + BA)
  # LUU Y: fp4GemvAt -> encodeLinByName -> encodeLin DA cong delta LoRA ben trong
  # (gpu_layers.encodeLin). KHONG goi encodeLoraDelta them o day: lora_b accumulate
  # nen goi 2 lan = LoRA bi cong GAP DOI (scale x2) - sai so im lang.
  if not fp4GemvAt(ctx, fp4InName, hNorm, 0, l.hWork, 0, 1):
    raise newException(CatchableError, "encodeLayerV2: fp4 in_proj that bai")

  # 3) split z/xBC/dt + conv1d (truot convState tai cho) + softplus(dt)
  encodeVm(l.progFsc, l.hWork, l.fscThreads, 256, 4, "fsc2")

  # 4) SSM step — doc/ghi ssmState TAI CHO, khong copy di dau
  encodeVm(l.progSsm, l.hWork, l.ssmThreads, 256, 4, l.ssmTag)

  # 5) gated RMSNorm theo nhom
  if gRNormGroupsOverride == 1 and l.normGroups != 1:
    encodeVm(l.progRms1, l.hWork, l.rmsThreads1, l.rmsTg1, l.rmsTg1 * 4, "rmsg1")
  else:
    encodeVm(l.progRms, l.hWork, l.rmsThreads, l.rmsTg, l.rmsTg * 4, "rmsg2")

  # 6) out_proj: doc THANG tu hWork[offNormed]
  # (delta LoRA out_proj da nam trong encodeLin, activation = hWork[offNormed .. +C))
  if not fp4GemvAt(ctx, fp4OutName, l.hWork, o.offNormed, l.hMix, 0, 1):
    raise newException(CatchableError, "encodeLayerV2: fp4 out_proj that bai")

  # 7) Residual can be deferred by the composite P-block encoder. That lets it
  # preserve the trainer's exact order: combined = mambaOut + attnOut; x += combined.
  if addResidual:
    var hDst = hRes
    if not sessionVecOp(be, "add", hRes, l.hMix, hDst, l.hidden):
      raise newException(CatchableError, "encodeLayerV2: residual add that bai")

## Merge helper shared by normal decode and the diagnostic trace path.
proc encodeMamba2RealLinearPlusResidual*(ctx: Backend; blockIdx: int;
                                          hRes: byby.SessionHandle;
                                          hCombined: var byby.SessionHandle;
                                          hAttnOut: byby.SessionHandle) =
  doAssert blockIdx in gLayersV2,
    "encodeMamba2RealLinearPlusResidual: Mamba2 layer chua duoc prepare"
  let ml = addr gLayersV2[blockIdx]
  # hMix holds mambaOut. hCombined is the now-dead pre-norm scratch, so write
  # the sum there without aliasing either branch output.
  if not sessionVecOp(ctx.kind.toByby(), "add", ml[].hMix, hAttnOut,
      hCombined, ml[].hidden):
    raise newException(CatchableError,
      "encodeMamba2RealLinearPlusResidual: branch combine that bai")
  var hDst = hRes
  if not sessionVecOp(ctx.kind.toByby(), "add", hRes, hCombined, hDst, ml[].hidden):
    raise newException(CatchableError,
      "encodeMamba2RealLinearPlusResidual: residual add that bai")

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
    "encodeMamba2RealLinearPlusBlock: Mamba2 layer chua duoc prepare"
  let inName = "block" & $blockIdx & ".mambaReal.inProj.weight"
  let outName = "block" & $blockIdx & ".mambaReal.outProj.weight"

  if skipLinearPlus:
    encodeLayerV2(ctx, blockIdx, hRes, hNorm, inName, outName)
    return

  doAssert blockIdx in gpu_layers.gAttnLayers,
    "encodeMamba2RealLinearPlusBlock: Linear++ layer chua duoc prepare"
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
      stderr.writeLine "[G] block=" & $blockIdx & " ====== fuseGate CHI TIET ======"
      stderr.writeLine "[G] block=" & $blockIdx & " (1) mambaOut = hMix  |  attnOut = hAttn  (cung RMS(x))"
      stderr.writeLine "[G] block=" & $blockIdx & " (2) gate_in = concat(mambaOut, attnOut) size 2*H"
    if not sessionCopyInto(be, ml[].hMix, hGateInput, 0, 0, ml[].hidden) or
       not sessionCopyInto(be, hAttn, hGateInput, 0, ml[].hidden, ml[].hidden):
      raise newException(CatchableError, "G block: khong ghep duoc dau vao fuseGate")
    if gDiag:
      stderr.writeLine "[G] block=" & $blockIdx & " (3) logit = fuseGate(gate_in)  [Linear H*2 -> H]"
    gpu_layers.encodeLin(attnLayer[].gate, hGateInput, 0, hGateLogit, 0)
    if gDiag:
      stderr.writeLine "[G] block=" & $blockIdx & " (4) g = sigmoid(logit)  |  (1-g) = 1 - g"
    if not sessionActivation(be, "sigmoid", hGateLogit, hGateProb, ml[].hidden) or
       not sessionVecOp(be, "sub", attnLayer[].hGateOnes, hGateProb,
                        hGateComplement, ml[].hidden):
      raise newException(CatchableError, "G block: fuseGate sigmoid/sub that bai")
    # Snapshot TRUOC mul in-place (doc so lieu sau sessionEnd)
    if gDiag:
      let needSnap =
        (be == byby.gbVulkan and attnLayer[].hDiagMamba.vulkanHandle.buf == nil) or
        (be == byby.gbMetal and attnLayer[].hDiagMamba.metalHandle == nil)
      if needSnap:
        attnLayer[].hDiagMamba = byby.sessionAllocScratch(be, ml[].hidden)
        attnLayer[].hDiagAttn = byby.sessionAllocScratch(be, ml[].hidden)
      if not sessionCopyInto(be, hMamba, attnLayer[].hDiagMamba, 0, 0, ml[].hidden) or
         not sessionCopyInto(be, hAttn, attnLayer[].hDiagAttn, 0, 0, ml[].hidden):
        raise newException(CatchableError, "G block: snapshot mamba/attn that bai")
      stderr.writeLine "[G] block=" & $blockIdx & " (5) mix = g*mamba + (1-g)*attn  (in-place tren hMix)"
      stderr.writeLine "[G] block=" & $blockIdx & " (6) x = x + mix  (residual)"
    if not sessionVecOp(be, "mul", hGateProb, hMamba, hMamba, ml[].hidden) or
       not sessionVecOp(be, "mul", hGateComplement, hAttn, hAttn, ml[].hidden) or
       not sessionVecOp(be, "add", hMamba, hAttn, hMamba, ml[].hidden) or
       not sessionVecOp(be, "add", hRes, hMamba, hResidual, ml[].hidden):
      raise newException(CatchableError, "G block: fuseGate convex mix failed")
    if gDiag:
      stderr.writeLine "[G] block=" & $blockIdx & " fuse queued OK (so lieu in sau sessionEnd)"
  else:
    encodeMamba2RealLinearPlusResidual(ctx, blockIdx, hRes, hNorm,
      gpu_layers.gAttnScratch.hAttn)
