## onedispatch.nim — "1 DISPATCH / TOKEN for all N Mamba2 layers (R blocks)" — micro-kernels only, NO hand-written MSL.
## ─────────────────────────────────────────────────────────────────────────────────────────
## STATUS: written WITHOUT a compiler/GPU at hand -> NOT compiled, NOT run. Read ONE_DISPATCH_PLAN.md.
##
## IDEA
##   * 1 VmProgram = FOR layer (GPU) { rms | in_proj(+LoRA) | fsc | ssm | rmsg | out_proj(+LoRA) | residual }
##   * Each stage sits in an OP_VTG_BEGIN/END block (VIRTUAL threadgroups, persistent loop) and the stages are separated by
##     OP_GRID_BARRIER (a grid-wide SOFTWARE barrier: Metal has none). One TG does not have enough bandwidth for a GEMV,
##     so ntg TGs run (default 16, env BYBY_OD_TG) with the software barrier.
##   * Each layer has its own "region" in one float buffer (memF32) + one meta slot (offset). OP_BASE_F32/U8/LUT/IN
##     move the base pointers per layer -> the old builders are REUSED AS-IS (progFsc/progSsm/progRms/gemv2row),
##     only composed, not a single line of their bytecode changed.
##
## 4 BUFFER / DISPATCH (Metal slot): 1 memF32 | 2 weight fp4 packed | 3 meta+LUT+LoRA | 4 = memF32 (alias, input)
##
## CONSTRAINTS (eligibility): every block is bkMambaReal, fp4-KM groupSize 32, RMSNorm (no LayerNorm), no bias,
##   hidden % 256 == 0, (C / nGroups) % 256 == 0, P even, LoRA rank <= 32 and uniform across layers, total weights < 4 GiB.
##   Otherwise -> odBegin/odAddLayer raise ValueError(reason) -> the caller falls back to the old path.

import std/[math, os, strutils]
import backend
import vendor/bybylang/gpubackend as byby
import session_vm
import session_vm_dispatch
import fp4_gemv_vm
import fp4_gemv_gpu
import mamba2_vm_kernels
import mamba2_fused_step      # LayerOffV2, computeOffV2
import od_core
export od_core

const
  OdSlotWords* = 16            ## uint32 per layer in the meta table
  OdSubStrideReal* = 336       ## float tgF32 / sub-TG 32-lane: LUT 272 + 2 vung reduce 32
  OdTgBytes* = 8 * OdSubStrideReal * 4   ## 8 sub-TG / TG vat ly 256

static: doAssert OdSubStrideReal >= Fp4KmLutLen + 2 * GemvTg

type
  OdLayerSlot* = object
    fBase*: int                 ## float offset of the layer region in memF32
    wIn*, wOut*: int            ## byte offset in memU8
    lutIn*, lutOut*: int        ## float offset in buffer 3
    loraAIn*, loraBIn*, loraAOut*, loraBOut*: int
    hasLoraIn*, hasLoraOut*: bool  ## the layer has NO adapter (e.g. B=0 dropped at load) -> the kernel skips its LoRA stage

  OdShape* = object
    hidden*, P*, C*, cDim*, H*, D*, N*, G*, K*, HD*: int
    nLayers*, normGroups*: int
    eps*: float32
    rIn*, rOut*: int
    scaleIn*, scaleOut*: float32
    off*: LayerOffV2
    offGamma*, regionStride*: int
    gX*, gNorm*, gMix*, gScr*, regionStart*, totalF32*: int

  OdState* = object
    be*: byby.GpuBackend
    sh*: OdShape
    ntg*: int
    kname*: string
    prog*: VmProgram
    hMem*, hZero*: byby.SessionHandle
    hLutSeg*: seq[byby.SessionHandle]          ## meta table: ONE copy PER SEGMENT (the ctrl row [start,count] set in advance) -> no copy at dispatch
    hWSeg*: seq[byby.SessionHandle]            ## fp4 weights: ONE buffer PER SEGMENT (Metal allows ~2GB/buffer at most)
    ready*: bool
    # staging (freed after odFinish)
    memInit: seq[float32]
    lutStage: seq[float32]
    wBytes: seq[seq[uint8]]                    ## weight staging per segment
    slots*: seq[OdLayerSlot]
    idxH, idxG: seq[float32]
    # --- hybrid R/G: only the R blocks are in OD; G/other blocks run the old path in between ---
    slotOfBlock*: seq[int]                     ## model block index -> OD slot (-1 = not in OD)
    segs*: seq[tuple[start, count: int]]       ## runs of consecutive R slots (one dispatch per run)

proc odAlign(x, a: int): int {.inline.} = ((x + a - 1) div a) * a

proc odFail(msg: string) {.noreturn.} =
  raise newException(ValueError, "onedispatch: " & msg)

# ════════════════════════════════════════════════════════════════════════
# A1. Layout
# ════════════════════════════════════════════════════════════════════════
proc odLayout*(hidden, C, cDim, H, D, N, G, K, nLayers, normGroups: int;
               eps: float32; rIn, rOut: int; scaleIn, scaleOut: float32): OdShape =
  result.hidden = hidden; result.C = C; result.cDim = cDim
  result.H = H; result.D = D; result.N = N; result.G = G; result.K = K
  result.HD = H * D
  result.P = C + cDim + H
  result.nLayers = nLayers; result.normGroups = normGroups; result.eps = eps
  result.rIn = rIn; result.rOut = rOut
  result.scaleIn = scaleIn; result.scaleOut = scaleOut
  result.off = computeOffV2(C, cDim, H, D, N, G, K)
  result.offGamma = result.off.total
  result.regionStride = odAlign(result.offGamma + hidden, 64)
  result.gX = OdCtrlFloats
  result.gNorm = result.gX + odAlign(hidden, 64)
  result.gMix = result.gNorm + odAlign(hidden, 64)
  result.gScr = result.gMix + odAlign(hidden, 64)
  result.regionStart = result.gScr + odAlign(max(max(rIn, rOut), 1), 64)
  result.totalF32 = result.regionStart + nLayers * result.regionStride

proc odLayerBase*(sh: OdShape; blockIdx: int): int = sh.regionStart + blockIdx * sh.regionStride

proc odConvStateOff*(sh: OdShape; blockIdx: int): int = sh.odLayerBase(blockIdx) + sh.off.offConvSt
proc odSsmStateOff*(sh: OdShape; blockIdx: int): int = sh.odLayerBase(blockIdx) + sh.off.offSsm

# ════════════════════════════════════════════════════════════════════════
# A2/A3. Pack FP4 weights + LUT + LoRA + state (staging on the host)
# ════════════════════════════════════════════════════════════════════════
proc odBegin*(ctx: Backend; hidden, C, cDim, H, D, N, G, K, nLayers, normGroups: int;
              eps: float32; rIn, rOut: int; scaleIn, scaleOut: float32): OdState =
  ## Checks the conditions + allocates staging. Raises ValueError if not eligible.
  if hidden mod OdTg != 0: odFail("hidden=" & $hidden & " must be divisible by " & $OdTg)
  if normGroups < 1 or C mod normGroups != 0: odFail("C must be divisible by normGroups")
  if (C div normGroups) mod OdTg != 0:
    odFail("groupLen=" & $(C div normGroups) & " must be divisible by " & $OdTg & " (rmsg runs exactly 1 physical TG per group)")
  if (C + cDim + H) mod 2 != 0 or hidden mod 2 != 0: odFail("P and hidden must be even (gemv 2 rows/TG)")
  if C mod 32 != 0 or hidden mod 32 != 0: odFail("K (hidden, C) must be divisible by 32 (fp4-KM group32)")
  if K < 2 or K > 8: odFail("convKernel ngoai [2,8]")
  if rIn < 0 or rOut < 0 or rIn > 32 or rOut > 32: odFail("LoRA rank must be 0..32")
  result.be = ctx.kind.toByby()
  result.sh = odLayout(hidden, C, cDim, H, D, N, G, K, nLayers, normGroups, eps, rIn, rOut, scaleIn, scaleOut)
  if result.sh.regionStride >= 16_000_000: odFail("layer region too large for float R[] addresses (>2^24)")
  if result.sh.totalF32 >= 0x7FFF_FFFF: odFail("total memF32 floats exceed 2^31")
  result.memInit = newSeq[float32](result.sh.totalF32)
  # meta table at the START of buffer 3
  # +1 control row (ctrl row) at the end of the meta table: [start, count] of the running segment
  result.lutStage = newSeq[float32](odAlign(nLayers * OdSlotWords + OdSlotWords, 64))
  # h/g index table (one table for every layer) — CPU, matching buildIdxTableProgram
  let HD = H * D
  let hpg = max(1, H div G)
  result.idxH = newSeq[float32](HD)
  result.idxG = newSeq[float32](HD)
  for t in 0 ..< HD:
    let h = t div D
    result.idxH[t] = float32(h)
    result.idxG[t] = float32(min(h div hpg, G - 1))

proc putAt(dst: var seq[float32]; off: int; src: openArray[float32]) =
  for i in 0 ..< src.len: dst[off + i] = src[i]

proc appendLut(st: var OdState; v: openArray[float32]): int =
  result = st.lutStage.len
  for x in v: st.lutStage.add x
  while st.lutStage.len mod 64 != 0: st.lutStage.add 0'f32

proc odSegOfSlot*(st: OdState; slot: int): int =
  for k, sg in st.segs:
    if slot >= sg.start and slot < sg.start + sg.count: return k
  0

proc appendW(st: var OdState; seg: int; w: openArray[uint8]): int =
  ## RELATIVE offset within the buffer of segment `seg` (each segment dispatch binds its own buffer).
  while st.wBytes.len <= seg: st.wBytes.add @[]
  result = st.wBytes[seg].len
  for b in w: st.wBytes[seg].add b
  while st.wBytes[seg].len mod 256 != 0: st.wBytes[seg].add 0'u8

proc odFp4Bytes*(K, N: int): int = N * (K div 2) + N * (K div 32)

proc odAddLayer*(st: var OdState; blockIdx: int;
                 aLog, dSkip, dtBias, normW, convW, convB, ln1Gamma: seq[float32];
                 inW: openArray[uint8]; inLut: seq[float32];
                 outW: openArray[uint8]; outLut: seq[float32];
                 loraAIn, loraBIn, loraAOut, loraBOut: seq[float32] = @[]) =
  ## Layers MUST be added in order 0,1,2,... (slot = block index).
  let sh = st.sh
  if blockIdx != st.slots.len: odFail("odAddLayer: layer " & $blockIdx & " out of order (expected " & $st.slots.len & ")")
  if blockIdx >= sh.nLayers: odFail("odAddLayer: exceeds nLayers")
  if inW.len != odFp4Bytes(sh.hidden, sh.P): odFail("in_proj bytes=" & $inW.len & " != " & $odFp4Bytes(sh.hidden, sh.P) & " (not fp4-KM group32?)")
  if outW.len != odFp4Bytes(sh.C, sh.hidden): odFail("out_proj bytes=" & $outW.len & " != " & $odFp4Bytes(sh.C, sh.hidden))
  if inLut.len != Fp4KmLutLen or outLut.len != Fp4KmLutLen: odFail("LUT must have length " & $Fp4KmLutLen)
  if ln1Gamma.len != sh.hidden: odFail("ln1Gamma.len != hidden")
  # LoRA PER layer: rank/scale (sh.rIn/rOut) are uniform, but a layer may have NO
  # adapter (empty arrays) -> flag = 0 in the meta, the kernel skips that layer's LoRA stage.
  if sh.rIn == 0 and loraAIn.len > 0 or sh.rOut == 0 and loraAOut.len > 0:
    odFail("layer " & $blockIdx & " has LoRA but OdShape rank = 0")
  if loraAIn.len > 0 and (loraAIn.len != sh.rIn * sh.hidden or loraBIn.len != sh.P * sh.rIn): odFail("LoRA in has the wrong size")
  if loraAOut.len > 0 and (loraAOut.len != sh.rOut * sh.C or loraBOut.len != sh.hidden * sh.rOut): odFail("LoRA out has the wrong size")

  let o = sh.off
  let base = sh.odLayerBase(blockIdx)
  # layer region: like initLayerV2 (+ gamma). convState / ssmState = 0 (start of the sequence).
  st.memInit.putAt(base + o.offALog, aLog)
  st.memInit.putAt(base + o.offDSkip, dSkip)
  st.memInit.putAt(base + o.offDtBias, dtBias)
  st.memInit.putAt(base + o.offNormW, normW)
  st.memInit.putAt(base + o.offConvW, convW)
  st.memInit.putAt(base + o.offConvB, convB)
  st.memInit.putAt(base + o.offIdxH, st.idxH)
  st.memInit.putAt(base + o.offIdxG, st.idxG)
  st.memInit.putAt(base + sh.offGamma, ln1Gamma)

  var s: OdLayerSlot
  s.fBase = base
  if st.segs.len == 0: st.segs.add (start: 0, count: sh.nLayers)   # the caller should set segs BEFORE odAddLayer
  let segK = st.odSegOfSlot(blockIdx)
  s.wIn = st.appendW(segK, inW)
  s.wOut = st.appendW(segK, outW)
  s.lutIn = st.appendLut(inLut)
  s.lutOut = st.appendLut(outLut)
  if loraAIn.len > 0:
    s.loraAIn = st.appendLut(loraAIn); s.loraBIn = st.appendLut(loraBIn); s.hasLoraIn = true
  if loraAOut.len > 0:
    s.loraAOut = st.appendLut(loraAOut); s.loraBOut = st.appendLut(loraBOut); s.hasLoraOut = true
  st.slots.add s

# ════════════════════════════════════════════════════════════════════════
# B. Bytecode: pre-norm RMS, residual, and one program combining every layer
# ════════════════════════════════════════════════════════════════════════
proc buildOdRmsNorm*(hidden: int; eps: float32; gX, gNorm: int): VmProgram =
  ## hNorm = hRes * rsqrt(mean(hRes^2) + eps) * gamma.  One physical TG (256 lanes) does it all.
  ## memF32 base = 0 (original); gamma read through inF32 (BASE_IN = the layer's gamma).
  doAssert hidden mod OdTg == 0
  let ept = hidden div OdTg
  var p: VmProgram
  p.emitLoadLid(1)                                  # R1 = lid
  p.emitULid(1)                                     # U1 = lid
  p.emitUAddK(3, 1, uint32(gX))                     # U3 = &hRes[lid]
  p.emitLoadConstF32(3, 0.0'f32)                    # R3 = sum
  p.emitLoadConstF32(8, float32(ept))
  sessionVmForGpu(p, 8):
    p.emitULoadF32(2, 3, 0)
    p.emitFma(3, 2, 2, 3)
    p.emitUAddK(3, 3, uint32(OdTg))
  p.emitTgStoreF32(3, 1, 0)
  p.emitBarrierTg()
  p.emitTgTreeReduce(1, OdTg, 4, 5, 6, 10, 11)      # R4 stride, R5 cnt, R6 tmp, R10/R11 a/b
  p.emitLoadConstF32(4, 0.0'f32)
  p.emitTgLoadF32(15, 4, 0)
  p.emitMulConst(15, 15, 1.0'f32 / float32(hidden))
  p.emitAddConst(15, 15, eps)
  p.emitRsqrt(15, 15)                               # R15 = r
  p.emitUAddK(3, 1, uint32(gX))
  p.emitUAddK(4, 1, 0)                              # U4 = &gamma[lid] (BASE_IN)
  p.emitUAddK(5, 1, uint32(gNorm))
  p.emitLoadConstF32(8, float32(ept))
  sessionVmForGpu(p, 8):
    p.emitULoadF32(2, 3, 0)
    p.emitULoadIn(7, 4, 0)
    p.emitMul(2, 2, 15)
    p.emitMul(2, 2, 7)
    p.emitUStoreF32(2, 5, 0)
    p.emitUAddK(3, 3, uint32(OdTg))
    p.emitUAddK(4, 4, uint32(OdTg))
    p.emitUAddK(5, 5, uint32(OdTg))
  p.emitHalt()
  result = p

proc buildOdResidual*(gX, gMix: int): VmProgram =
  ## hRes[tid] += hMix[tid]  (mode 2: guard tid < hidden o ngoai)
  var p: VmProgram
  p.emitLoadF32(1, 0, uint32(gX))
  p.emitLoadF32(2, 0, uint32(gMix))
  p.emitAdd(1, 1, 2)
  p.emitStoreF32(1, 0, uint32(gX))
  p.emitHalt()
  result = p

proc cdiv(a, b: int): uint32 = uint32((a + b - 1) div b)

proc buildOdProgram*(sh: OdShape): VmProgram =
  ## One program for ALL nLayers. Outer U-regs: U0 = layer, U1 tmp, U2 fBase, U3 wIn, U4 wOut,
  ## U5 lutIn, U6 lutOut, U7 laAIn, U8 lbAIn(B in), U9 laOut, U14 lbOut, U10 = 0 (original), U11 gNorm,
  ## U12 gMix, U13 gScr, U15 tmp address.
  let o = sh.off
  var p: VmProgram
  p.emitUConst(10, 0'u32)
  p.emitUConst(11, uint32(sh.gNorm))
  p.emitUConst(12, uint32(sh.gMix))
  p.emitUConst(13, uint32(sh.gScr))
  # The layer range [start, start+count) is read from the ctrl row of the meta table (each segment only changes 2 numbers,
  # NO kernel recompile): U0 = start, R5 = count.
  p.emitUConst(15, uint32(sh.nLayers * OdSlotWords))
  p.emitULoadMeta(0, 15, 0)
  p.emitULoadMeta(1, 15, 1)
  p.emitUToR(5, 1)
  p.emitForBegin(5)
  p.emitUMulK(1, 0, uint32(OdSlotWords))
  p.emitULoadMeta(2, 1, 0)       # fBase
  p.emitULoadMeta(3, 1, 1)       # wIn
  p.emitULoadMeta(4, 1, 2)       # wOut
  p.emitULoadMeta(5, 1, 3)       # lutIn
  p.emitULoadMeta(6, 1, 4)       # lutOut
  p.emitULoadMeta(7, 1, 5)       # loraAIn
  p.emitULoadMeta(8, 1, 6)       # loraBIn
  p.emitULoadMeta(9, 1, 7)       # loraAOut
  p.emitULoadMeta(14, 1, 8)      # loraBOut
  if sh.rIn > 0 or sh.rOut > 0:
    # Per-layer LoRA flags (outer R: R5 = layer loop count, R6/R7 = in/out flags). The condition is uniform
    # across EVERY thread of the layer -> the grid barriers inside the if are skipped uniformly
    # (the gbEpoch of every TG still agrees).
    p.emitULoadMeta(15, 1, 9);  p.emitUToR(6, 15)
    p.emitULoadMeta(15, 1, 10); p.emitUToR(7, 15)

  # 1) pre-norm
  p.emitUAddK(15, 2, uint32(sh.offGamma))
  p.emitBaseF32(10); p.emitBaseIn(15)
  odStage(p, buildOdRmsNorm(sh.hidden, sh.eps, sh.gX, sh.gNorm), 0, 1, 0)
  p.emitGridBarrier(0)

  # 2) in_proj  (hNorm -> hWork[0..P))
  p.emitBaseF32(2); p.emitBaseU8(3); p.emitBaseLut(5); p.emitBaseIn(11)
  odStage(p, buildFp4KmGemv2RowProgram(sh.hidden, sh.P), 1, uint32(sh.P div 2), uint32(OdSubStrideReal))
  p.emitGridBarrier(0)
  if sh.rIn > 0:
    p.emitIfBegin(6)
    p.emitBaseF32(13); p.emitBaseLut(7); p.emitBaseIn(11)
    odStage(p, buildF32GemvProgram(sh.hidden, sh.rIn, 1.0'f32, false), 1, uint32(sh.rIn), uint32(OdSubStrideReal))
    p.emitGridBarrier(0)
    p.emitBaseF32(2); p.emitBaseLut(8); p.emitBaseIn(13)
    odStage(p, buildF32GemvSmallKProgram(sh.rIn, sh.P, sh.scaleIn, true), 2, cdiv(sh.P, OdTg), uint32(sh.P))
    p.emitGridBarrier(0)
    p.emitIfEnd()

  # 3) split + conv1d + softplus
  p.emitBaseF32(2)
  let fscThreads = max(sh.C, max(sh.cDim, sh.H))
  let progFsc = buildSplitConvSoftplusAt(sh.C, sh.cDim, sh.H, sh.K,
    o.offZ, o.offXBC, o.offDt, o.offConvW, o.offConvB, o.offConvSt, o.offDtBias)
  odStage(p, progFsc, 2, cdiv(fscThreads, OdTg), uint32(fscThreads))
  p.emitGridBarrier(0)

  # 4) SSM step (state in place)
  let progSsm = buildMamba2StepAt(sh.H, sh.D, sh.N, sh.G,
    o.offXBC, o.offXBC + sh.C, o.offXBC + sh.C + sh.G * sh.N, o.offDt,
    o.offALog, o.offDSkip, o.offSsm, o.offY, o.offIdxH, o.offIdxG)
  odStage(p, progSsm, 2, cdiv(sh.HD, OdTg), uint32(sh.HD))
  p.emitGridBarrier(0)

  # 5) grouped gated RMSNorm (each group = 1 virtual TG = 1 physical TG of 256)
  let progRms = buildRmsGateGroupedAt(sh.C, sh.normGroups, OdTg, sh.eps,
    o.offY, o.offZ, o.offNormW, o.offNormed)
  odStage(p, progRms, 0, uint32(sh.normGroups), 0)
  p.emitGridBarrier(0)

  # 6) out_proj (hWork[offNormed] -> hMix)
  p.emitUAddK(15, 2, uint32(o.offNormed))
  p.emitBaseF32(12); p.emitBaseU8(4); p.emitBaseLut(6); p.emitBaseIn(15)
  odStage(p, buildFp4KmGemv2RowProgram(sh.C, sh.hidden), 1, uint32(sh.hidden div 2), uint32(OdSubStrideReal))
  p.emitGridBarrier(0)
  if sh.rOut > 0:
    p.emitIfBegin(7)
    p.emitBaseF32(13); p.emitBaseLut(9); p.emitBaseIn(15)
    odStage(p, buildF32GemvProgram(sh.C, sh.rOut, 1.0'f32, false), 1, uint32(sh.rOut), uint32(OdSubStrideReal))
    p.emitGridBarrier(0)
    p.emitBaseF32(12); p.emitBaseLut(14); p.emitBaseIn(13)
    odStage(p, buildF32GemvSmallKProgram(sh.rOut, sh.hidden, sh.scaleOut, true), 2, cdiv(sh.hidden, OdTg), uint32(sh.hidden))
    p.emitGridBarrier(0)
    p.emitIfEnd()

  # 7) residual
  p.emitBaseF32(10)
  odStage(p, buildOdResidual(sh.gX, sh.gMix), 2, cdiv(sh.hidden, OdTg), uint32(sh.hidden))
  p.emitGridBarrier(0)

  p.emitUAddK(0, 0, 1)
  p.emitForEnd()
  p.emitGridEpilogue(0)
  p.emitHalt()
  result = p

# ════════════════════════════════════════════════════════════════════════
# C. Finish (upload 1 lan) + encode
# ════════════════════════════════════════════════════════════════════════
proc setU32(v: var seq[float32]; i: int; x: int) =
  var u = uint32(x)
  copyMem(addr v[i], addr u, 4)

proc odFinish*(ctx: Backend; st: var OdState) =
  ## Requires: odAddLayer called for all nLayers. Uploads 3 buffers + zeroes buf, JIT-compiles the program.
  let sh = st.sh
  if st.slots.len != sh.nLayers: odFail("odFinish: only " & $st.slots.len & "/" & $sh.nLayers & " layers")
  for k in 0 ..< st.wBytes.len:
    if st.wBytes[k].len >= 2_000_000_000:
      odFail("segment " & $k & " has " & $st.wBytes[k].len & " bytes >= 2GB (Metal buffer limit): use smaller segments")
  for i, s in st.slots:
    let b = i * OdSlotWords
    st.lutStage.setU32(b + 0, s.fBase)
    st.lutStage.setU32(b + 1, s.wIn)
    st.lutStage.setU32(b + 2, s.wOut)
    st.lutStage.setU32(b + 3, s.lutIn)
    st.lutStage.setU32(b + 4, s.lutOut)
    st.lutStage.setU32(b + 5, s.loraAIn)
    st.lutStage.setU32(b + 6, s.loraBIn)
    st.lutStage.setU32(b + 7, s.loraAOut)
    st.lutStage.setU32(b + 8, s.loraBOut)
    st.lutStage.setU32(b + 9, (if s.hasLoraIn: 1 else: 0))
    st.lutStage.setU32(b + 10, (if s.hasLoraOut: 1 else: 0))
  let be = st.be
  st.hMem = byby.sessionUploadPrivate(be, st.memInit)
  for k in 0 ..< st.wBytes.len:
    st.hWSeg.add sessionUploadBytesShared(be, st.wBytes[k])
  st.hZero = byby.sessionUploadPrivate(be, newSeq[float32](max(sh.cDim * max(sh.K - 1, 1), sh.HD * sh.N)))
  if st.segs.len == 0: st.segs.add (start: 0, count: sh.nLayers)
  # Each segment has its OWN meta table copy with the ctrl row [start,count] written at upload:
  # NO copy command right before dispatch (avoids a blit->compute ordering dependency).
  let ctrl = sh.nLayers * OdSlotWords
  for k, sg in st.segs:
    st.lutStage.setU32(ctrl + 0, sg.start)
    st.lutStage.setU32(ctrl + 1, sg.count)
    st.hLutSeg.add byby.sessionUploadPrivate(be, st.lutStage)
  st.memInit = @[]; st.lutStage = @[]; st.wBytes = @[]
  var ntg = 16
  let e = getEnv("BYBY_OD_TG")
  if e.len > 0:
    try: ntg = parseInt(e)
    except ValueError: discard
  if ntg < 1: ntg = 1
  st.ntg = ntg
  st.prog = buildOdProgram(sh)
  st.kname = odCompile(st.prog, "od_all_" & $sh.nLayers)
  st.ready = true
  if getEnv("BYBY_OD_VERBOSE", "0") != "0":
    stderr.writeLine "[onedispatch] layers=" & $sh.nLayers & " ntg=" & $ntg & " words=" & $st.prog.words.len &
      " memFloats=" & $sh.totalF32 & " kernel=" & st.kname

var gOdState*: OdState   ## 1 model / process

proc odResetLayerState*(st: OdState; blockIdx: int): bool =
  ## Zeroes the convState + ssmState of a layer (MUST be inside an open session). Equivalent to resetLayerStateV2.
  if not st.ready: return false
  let sh = st.sh
  var hm = st.hMem
  result = true
  if not sessionCopyInto(st.be, st.hZero, hm, 0, sh.odConvStateOff(blockIdx), sh.cDim * (sh.K - 1)): result = false
  if not sessionCopyInto(st.be, st.hZero, hm, 0, sh.odSsmStateOff(blockIdx), sh.HD * sh.N): result = false

proc odImportState*(st: OdState; blockIdx: int; hConv, hSsm: byby.SessionHandle): bool =
  ## For parallel prefill: copies the computed state into the layer region (instead of writing to the old l.hWork).
  if not st.ready: return false
  let sh = st.sh
  var hm = st.hMem
  result = true
  if not sessionCopyInto(st.be, hConv, hm, 0, sh.odConvStateOff(blockIdx), sh.cDim * (sh.K - 1)): result = false
  if not sessionCopyInto(st.be, hSsm, hm, 0, sh.odSsmStateOff(blockIdx), sh.HD * sh.N): result = false

proc odEncodeSegment*(st: OdState; segIdx: int; hRes: byby.SessionHandle) =
  ## Runs R segment segIdx (consecutive R layers) in EXACTLY one compute dispatch.
  ## hRes = gHXPersist. G blocks between the segments are run by the caller on the old path.
  if not st.ready: raise newException(CatchableError, "odEncodeSegment: odFinish not called")
  if segIdx < 0 or segIdx >= st.segs.len: raise newException(CatchableError, "odEncodeSegment: invalid segIdx")
  let sh = st.sh
  var hm = st.hMem
  var hr = hRes
  if not sessionCopyInto(st.be, hr, hm, 0, sh.gX, sh.hidden):
    raise newException(CatchableError, "odEncodeSegment: copy hRes -> pack failed")
  odDispatch(st.kname, st.hMem, st.hWSeg[segIdx], st.hLutSeg[segIdx], st.ntg, OdTgBytes)
  if not sessionCopyInto(st.be, hm, hr, sh.gX, 0, sh.hidden):
    raise newException(CatchableError, "odEncodeSegment: copy pack -> hRes failed")

proc odEncodeToken*(st: OdState; hRes: byby.SessionHandle) =
  ## hRes = gHXPersist (already holds the embedding). Writes the result back into hRes. EXACTLY one compute dispatch.
  if not st.ready: raise newException(CatchableError, "odEncodeToken: odFinish not called")
  let sh = st.sh
  var hm = st.hMem
  var hr = hRes
  if not sessionCopyInto(st.be, hr, hm, 0, sh.gX, sh.hidden):
    raise newException(CatchableError, "odEncodeToken: copy hRes -> pack failed")
  for k in 0 ..< st.segs.len:
    odDispatch(st.kname, st.hMem, st.hWSeg[k], st.hLutSeg[k], st.ntg, OdTgBytes)
  if not sessionCopyInto(st.be, hm, hr, sh.gX, 0, sh.hidden):
    raise newException(CatchableError, "odEncodeToken: copy pack -> hRes failed")

proc odCheckErr*(st: OdState): uint32 =
  ## AFTER sessionEnd: reads lastErr (float idx 3). != 0 => spun past the limit (TGs not resident at the same time).
  let v = sessionRead(st.be, st.hMem, 4)
  copyMem(addr result, unsafeAddr v[3], 4)

proc odResetHookImpl(blockIdx: int): bool {.nimcall.} =
  # blockIdx = MODEL block index -> converted to the OD slot
  if blockIdx < 0 or blockIdx >= gOdState.slotOfBlock.len or gOdState.slotOfBlock[blockIdx] < 0: return false
  odResetLayerState(gOdState, gOdState.slotOfBlock[blockIdx])

proc odOwnsHookImpl(blockIdx: int): bool {.nimcall.} =
  blockIdx >= 0 and blockIdx < gOdState.slotOfBlock.len and gOdState.slotOfBlock[blockIdx] >= 0

proc odInstall*(st: OdState) =
  ## Call AFTER odFinish: sets the global state + reset hook (resetLayerStateV2 -> pack).
  gOdState = st
  gOdResetHook = odResetHookImpl
  gOdOwnsHook = odOwnsHookImpl
