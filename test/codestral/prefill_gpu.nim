## Resident R/P/G prefix: reuse device scratch and import terminal states by
## device-to-device copies. No host activations, transposed FP32 weights, or
## backward caches between layers. The caller owns state reset/recovery.
import std/[tables, strformat, os, strutils]
import backend, nimformer, jamba_hybrid, mamba2_full_vm, mamba2_ssm_gpu
import mamba2_fused_step, gpu_layers, fp4_gemv_gpu, session_vm
import fp4_gemv_vm
import prefill_profile
import vendor/bybylang/gpubackend as byby
when defined(macosx):
  import vendor/bybylang/backends/metal/metal_backend as metalBackend

proc require(ok: bool; what: string) =
  if not ok: raise newException(CatchableError, "resident prefill: " & what)

type PrefixBuffer = object
  handle: SessionHandle
  elems: int

# One bounded arena: reuse across prompts with the same shape, evict on shape
# changes. The decoder already has one global model/state, so concurrent prefix
# use is rejected rather than allowing two callers to overwrite this arena.
var gPrefixBuffers: seq[PrefixBuffer]
var gPrefixShape: string
var gPrefixBusy = false
var gPrefixAllocs, gPrefixHits: int

proc prefillBufferStats*(): tuple[allocations, hits, buffers: int] =
  (gPrefixAllocs, gPrefixHits, gPrefixBuffers.len)

proc clearPrefixBuffers() =
  for buf in gPrefixBuffers.mitems: sessionFree(buf.handle.kind, buf.handle)
  gPrefixBuffers.setLen(0)
  gPrefixShape = ""

proc rowIndex(p: var VmProgram; cols: int) =
  # U0=flat index, U1=row, U2=column. All indices here are below 2^24.
  p.emitRToU(0, 0)
  p.emitLoadConstF32(1, float32(cols)); p.emitDiv(2, 0, 1)
  p.emitRToU(1, 2); p.emitUMulK(2, 1, uint32(cols))
  p.emitUToR(3, 2); p.emitSub(3, 0, 3); p.emitRToU(2, 3)

proc biasRows(cols: int): VmProgram =
  var p: VmProgram
  p.rowIndex(cols)
  p.emitULoadF32(4, 0); p.emitULoadConstF(5, 2)
  p.emitAdd(4, 4, 5); p.emitUStoreF32(4, 0); p.emitHalt()
  p

proc loraRows(cols, rank: int; scale: float32): VmProgram =
  # mem += scale * (X A^T) B^T, one thread per (token, output).
  var p: VmProgram
  p.rowIndex(cols)
  p.emitUMulK(3, 1, uint32(rank)); p.emitUMulK(4, 2, uint32(rank))
  p.emitLoadConstF32(4, 0); p.emitLoadConstF32(8, float32(rank))
  sessionVmForGpu(p, 8):
    p.emitULoadIn(5, 3); p.emitULoadConstF(6, 4)
    p.emitFma(4, 5, 6, 4)
    p.emitUAddK(3, 3, 1); p.emitUAddK(4, 4, 1)
  p.emitMulConst(4, 4, scale); p.emitULoadF32(5, 0)
  p.emitAdd(4, 4, 5); p.emitUStoreF32(4, 0); p.emitHalt()
  p

proc convHistory(L: M2Lay): VmProgram =
  # Projection is input; output is [channel, oldest..newest raw xBC].
  var p: VmProgram
  p.emitRToU(0, 0); p.emitUMulK(1, 0, uint32(L.K - 1))
  for k in 0 ..< L.K - 1:
    let t = L.T - (L.K - 1) + k
    if t >= 0:
      p.emitULoadIn(1, 0, uint32(t * L.P + L.C))
    else: p.emitLoadConstF32(1, 0)
    p.emitUStoreF32(1, 1, uint32(k))
  p.emitHalt()
  p

proc transposeState(heads, dim: int): VmProgram =
  var p: VmProgram
  p.rowIndex(dim * dim) # U1=head; U2=key*D+value
  p.emitUToR(4, 2); p.emitLoadConstF32(5, float32(dim)); p.emitDiv(4, 4, 5)
  p.emitRToU(3, 4); p.emitUMulK(4, 3, uint32(dim))
  p.emitUToR(5, 4); p.emitUToR(6, 2); p.emitSub(6, 6, 5); p.emitRToU(4, 6)
  p.emitUMulK(5, 1, uint32(dim * dim)); p.emitUMulK(4, 4, uint32(dim))
  p.emitUAdd(5, 5, 4); p.emitUAdd(5, 5, 3)
  p.emitULoadIn(7, 5); p.emitUStoreF32(7, 0); p.emitHalt()
  p

proc gateInput(hidden: int): VmProgram =
  var p: VmProgram
  p.rowIndex(hidden)
  p.emitUMulK(3, 1, uint32(2 * hidden)); p.emitUAdd(3, 3, 2)
  p.emitULoadConstF(4, 0); p.emitULoadIn(5, 0)
  p.emitUStoreF32(4, 3); p.emitUStoreF32(5, 3, uint32(hidden)); p.emitHalt()
  p

proc gateMix(): VmProgram =
  var p: VmProgram
  p.emitRToU(0, 0)
  p.emitULoadF32(1, 0); p.emitULoadIn(2, 0); p.emitULoadConstF(3, 0)
  p.emitSigmoid(3, 3); p.emitLoadConstF32(4, 1); p.emitSub(4, 4, 3)
  p.emitMul(1, 1, 3); p.emitMul(2, 2, 4); p.emitAdd(1, 1, 2)
  p.emitUStoreF32(1, 0); p.emitHalt()
  p

proc residentPrefillReason*(m: NimformerModelJamba; ctx: Backend): string =
  let be = ctx.kind.toByby()
  if be notin {gbMetal, gbVulkan}: return "resident prefix requires Metal or Vulkan"
  if m.blocks.len == 0: return "empty model"
  let first = m.blocks[0].mambaReal
  for i, blk in m.blocks:
    if blk.kind notin {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}:
      return "unsupported block kind"
    let mr = blk.mambaReal
    if mr.hiddenSize != first.hiddenSize or mr.intermediateSize != first.intermediateSize or
       mr.numHeads != first.numHeads or mr.headDim != first.headDim or
       mr.dState != first.dState or mr.nGroups != first.nGroups or
       mr.convKernel != first.convKernel or mr.normEps != first.normEps:
      return "nonuniform Mamba shapes"
    if i notin gLayersV2: return "recurrent state is owned by one-dispatch"
    var names = @["block" & $i & ".mambaReal.inProj.weight",
                  "block" & $i & ".mambaReal.outProj.weight"]
    if blk.kind in {bkMambaRealPlusAttn, bkMambaRealGated}:
      names.add "block" & $i & ".attn.qkvProj.weight"
      names.add "block" & $i & ".attn.outProj.weight"
      if i notin gAttnLayers: return "missing Linear++ state"
    if blk.kind == bkMambaRealGated: names.add "block" & $i & ".fuseGate.weight"
    for name in names:
      if name notin gLinCache or not gLinCache[name].ready: return "missing linear " & name
      let lin = gLinCache[name]
      if lin.isInt4: return "int4 batched cache is not supported"
      if lin.isFp4 and gFp4CacheGpu[name].m4JitName.len == 0:
        return "missing FP4 batch kernel"

proc forwardResidentPrefill*(m: var NimformerModelJamba; ids: seq[int]; ctx: Backend) =
  ensureTgCap(ctx)
  require(ids.len >= 2 and ids.len <= 64, "prefix length must be 2..64")
  let be = ctx.kind.toByby()
  let T = ids.len
  let tileM = if T < 8: 4 else: 8
  let padded = ((T + tileM - 1) div tileM) * tileM
  let hidden = m.embedDim
  let mr = m.blocks[0].mambaReal
  var useMps = false
  when defined(macosx):
    useMps = getEnv("BYBY_PREFILL_MPS", "1") != "0" and metalBackend.metalMpsSupported()
  let L = m2Layout(1, T, mr.numHeads, mr.headDim, mr.dState,
    mr.nGroups, mr.convKernel, mr.intermediateSize, mr.normEps, mr.nGroups)
  checkVmScratchCapacity(ctx, L.fwdEnd, "resident prefill")
  var maxRank = 1
  for _, adapter in gLoraGpu: maxRank = max(maxRank, adapter.r)
  var maxState = 1
  var shape = $be & "_" & m2Tag(L) & "_" & $hidden & "_" & m.pattern
  for i, blk in m.blocks:
    if blk.kind in {bkMambaRealPlusAttn, bkMambaRealGated}:
      let al = gAttnLayers[i]
      maxState = max(maxState, al.lay.H * al.lay.D * al.lay.D)
      shape.add "_lp" & $al.lay.H & "x" & $al.lay.D
  shape.add "_r" & $maxRank
  require(not gPrefixBusy, "scratch arena is already in use")
  if gPrefixShape != shape:
    clearPrefixBuffers()
    gPrefixShape = shape
  gPrefixBusy = true
  let hitsBefore = gPrefixHits
  var nextBuffer = 0
  var owned: seq[SessionHandle]
  var sessionOpen = false
  var complete = false
  defer:
    if sessionOpen:
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
    for h in owned.mitems: sessionFree(be, h)
    if not complete: clearPrefixBuffers()
    gPrefixBusy = false
  proc alloc(n: int; zero = false): SessionHandle =
    let slot = nextBuffer
    inc nextBuffer
    if slot < gPrefixBuffers.len and gPrefixBuffers[slot].elems >= n:
      inc gPrefixHits
      return gPrefixBuffers[slot].handle
    if slot < gPrefixBuffers.len:
      sessionFree(be, gPrefixBuffers[slot].handle)
    else: gPrefixBuffers.add PrefixBuffer()
    result = if zero: sessionUpload(be, newSeq[float32](n))
             else: sessionAllocScratch(be, n)
    let valid = if be == gbVulkan: result.vulkanHandle.buf != nil
                else: result.metalHandle != nil
    require(valid, "buffer allocation failed")
    gPrefixBuffers[slot] = PrefixBuffer(handle: result, elems: n)
    inc gPrefixAllocs
  proc copy(src: SessionHandle; dst: var SessionHandle; srcOff, dstOff, n: int) =
    require(sessionCopyInto(be, src, dst, srcOff, dstOff, n), "device copy failed")
  # Allocate once for the entire stack. Padding is zero only in input buffers;
  # M4 output rows never overlap the packed scan's small parameters.
  var hX = alloc(padded * hidden, true)
  var hNorm = alloc(padded * hidden, true)
  var hProjection = alloc(padded * L.P)
  var hInner = alloc(padded * L.C, true)
  var hMix = alloc(padded * hidden)
  var hMamba = alloc(L.fwdEnd)
  var hConv = alloc(L.cDim * (L.K - 1))
  var hLora = alloc(T * maxRank)
  let zeroLen = max(L.HD * L.N, max(T * hidden, maxState))
  let hZero = alloc(zeroLen, true)
  let embed = m.embed.lookupBatch(@[ids], ctx)
  let hEmbed = sessionUpload(be, embed.data)
  owned.add hEmbed
  require(sessionBegin(be), "embedding sessionBegin failed")
  sessionOpen = true
  copy(hEmbed, hX, 0, 0, T * hidden)
  require(sessionEnd(be), "embedding sessionEnd failed")
  sessionOpen = false

  # Programs are compiled before opening a command buffer (no nested sessions).
  let historyKn = vmKernelForBe(convHistory(L), "prefix_conv_" & m2Tag(L), be)
  let scanStep: MvmStep =
    if getEnv("BYBY_PREFILL_SCAN", "serial") == "parallel": mamba2ScanStep(L, ctx, forceParallel = true)
    else: (buildM2ScanProgram(L), L.HD, 4, 0, "@prefix_m2_serial_" & m2Tag(L))
  var steps: seq[MvmStep] = @[
    (buildM2ConvDtProgram(L), T * M2Tg, 4, M2Tg, "@m2_conv_dt_" & m2Tag(L)),
    scanStep,
    (buildM2RmsFwdProgram(L), T * L.normGroups * rmsTg(L), rmsTg(L) * 4,
      rmsTg(L), "@m2_rms_" & m2Tag(L))]
  var stepNames: seq[string]
  for st in steps: stepNames.add kernelNameFor(ctx, st.prog, st.tag)
  var biasKns = initTable[int, string]()
  var loraKns = initTable[string, string]()
  var m8Kns = initTable[string, string]()
  let useM8 = T >= 8 and getEnv("BYBY_PREFILL_FP4_KERNEL", "auto") != "m4"
  for name, lin in gLinCache:
    if not name.startsWith("block"): continue
    if lin.N notin biasKns:
      biasKns[lin.N] = vmKernelForBe(biasRows(lin.N), "prefix_bias_" & $lin.N, be)
    if name in gLoraGpu:
      let ad = gLoraGpu[name]
      loraKns[name] = vmKernelForBe(loraRows(lin.N, ad.r, ad.scale),
        "prefix_lora_" & $lin.N & "_" & $ad.r & "_" & $ad.scale, be)
    if lin.isFp4 and useM8:
      let prepared = gFp4CacheGpu[name].m8JitName
      m8Kns[name] = if prepared.len > 0: prepared
                   else: vmKernelForBe(buildFp4KmGemmM8Program(lin.K, lin.N),
                     "fp4km_m8_" & $lin.K & "x" & $lin.N, be)
  proc matmul(input, weight: SessionHandle; output: var SessionHandle;
              rows, k, cols: int): bool =
    when defined(macosx):
      if useMps and metalBackend.metalSessionMatmulMpsNTEnc(input.metalHandle,
          weight.metalHandle, output.metalHandle, rows, k, cols, cols, 0): return true
    byby.sessionMatmulNTAccelerated(be, input, weight, output, rows, k, cols)

  proc linear(name: string; input: SessionHandle; output: var SessionHandle;
              bias: bool) =
    let lin = gLinCache[name]
    if lin.isFp4 and byby.sessionFp4GemmTensorCore(be, input,
        gFp4CacheGpu[name].wBuf, gFp4CacheGpu[name].lutBuf, output, T, lin.K, lin.N):
      discard
    elif name in m8Kns:
      let c = gFp4CacheGpu[name]
      let tg = if c.m8TgSize > 0: c.m8TgSize else: GemvTg
      for row in countup(0, padded - 1, 8):
        encodeVmKernel(m8Kns[name], lin.N * tg, tg,
          (Fp4KmLutLen + 8 * tg) * 4, output, row * lin.N,
          c.wBuf, c.lutBuf, input, row * lin.K)
    elif lin.isFp4:
      require(encodeFp4GemmRows(gFp4CacheGpu[name], input, output, T), "FP4 GEMM " & name)
    else:
      require(matmul(input, lin.wF32, output, T, lin.K, lin.N), "FP32 GEMM " & name)
    if name in gLoraGpu and not gDisableLora:
      let ad = gLoraGpu[name]
      require(matmul(input, ad.aBuf, hLora, T, lin.K, ad.r), "LoRA A " & name)
      encodeVmKernel(loraKns[name], T * lin.N, 64, 4,
        output, 0, gDummyFor(be), ad.bBuf, hLora, 0)
    if bias and lin.hasBias:
      encodeVmKernel(biasKns[lin.N], T * lin.N, 64, 4,
        output, 0, gDummyFor(be), lin.biasH, gDummyFor(be), 0)

  # Linear++ arenas are shared by all sidecars with the same H/D shape.
  var lpArenas = initTable[string, SessionHandle]()
  var lpPrograms = initTable[string, seq[MvmStep]]()
  var lpNames = initTable[string, seq[string]]()
  var transposeKns = initTable[string, string]()
  for i, blk in m.blocks:
    if blk.kind notin {bkMambaRealPlusAttn, bkMambaRealGated}: continue
    let al = gAttnLayers[i]
    let lay = lpLayout(1, T, al.lay.H, al.lay.D)
    let key = $lay.H & "_" & $lay.D
    if key notin lpArenas:
      checkVmScratchCapacity(ctx, lay.total, "resident Linear++ prefill")
      lpArenas[key] = alloc(lay.total)
      # Short prefixes use the bounded serial recurrence. No float atomics or
      # large per-timestep state scan are needed to compute 15 prefix tokens.
      let phiTg = lpPhiTg(lay)
      let tag = $T & "_" & key & "_" & $phiTg
      lpPrograms[key] = @[
        (buildLpPhiProgram(lay), T * 2 * hidden, phiTg * 4, phiTg, "@prefix_lp_phi_" & tag),
        (buildLpDenProgram(lay), lay.H, 4, 0, "@prefix_lp_den_" & tag),
        (buildLpFwdProgram(lay), hidden, 4, 0, "@prefix_lp_fwd_" & tag)]
      for st in lpPrograms[key]: lpNames.mgetOrPut(key, @[]).add kernelNameFor(ctx, st.prog, st.tag)
      transposeKns[key] = vmKernelForBe(transposeState(lay.H, lay.D), "prefix_lp_state_" & key, be)
  var hQkv = alloc(padded * 3 * hidden)
  var hAttnIn = alloc(padded * hidden, true)
  var hAttnOut = alloc(padded * hidden)
  var hGateIn = alloc(padded * 2 * hidden, true)
  var hGate = alloc(padded * hidden)
  var hLpState = alloc(maxState)
  let gateInKn = vmKernelForBe(gateInput(hidden), "prefix_gate_input_" & $hidden, be)
  let gateMixKn = vmKernelForBe(gateMix(), "prefix_gate_mix", be)

  for i, blk in m.blocks:
    let t0 = getMonoTimeIfPf()
    let root = "block" & $i
    let layer = gLayersV2[i]
    require(sessionBegin(be), "block sessionBegin failed")
    sessionOpen = true
    require(sessionRmsnormEnc(be, hX, layer.hGamma, hNorm, T, hidden, layer.eps), "RMSNorm")
    linear(root & ".mambaReal.inProj.weight", hNorm, hProjection, false)
    copy(hProjection, hMamba, 0, L.proj, T * L.P)
    copy(layer.hWork, hMamba, layer.off.offConvW, L.convW, L.cDim * L.K)
    copy(layer.hWork, hMamba, layer.off.offConvB, L.convB, L.cDim)
    copy(layer.hWork, hMamba, layer.off.offALog, L.aLog, L.H)
    copy(layer.hWork, hMamba, layer.off.offDSkip, L.dSkip, L.H)
    copy(layer.hWork, hMamba, layer.off.offDtBias, L.dtBias, L.H)
    copy(layer.hWork, hMamba, layer.off.offNormW, L.normW, L.C)
    copy(hZero, hMamba, 0, L.states, L.HD * L.N)
    for s, st in steps:
      encodeMicroVm(ctx, st.prog, hMamba, st.nThreads, st.tgBytes, st.tag, st.tgSize, stepNames[s])
    copy(hMamba, hInner, L.normedOut, 0, T * L.C)
    linear(root & ".mambaReal.outProj.weight", hInner, hMix, false)
    encodeVmKernel(historyKn, L.cDim, 64, 4, hConv, 0, gDummyFor(be), gDummyFor(be), hProjection, 0)
    var hState = layer.hWork
    copy(hConv, hState, 0, layer.off.offConvSt, L.cDim * (L.K - 1))
    copy(hMamba, hState, L.states + T * L.HD * L.N, layer.off.offSsm, L.HD * L.N)
    if blk.kind in {bkMambaRealPlusAttn, bkMambaRealGated}:
      let al = gAttnLayers[i]
      let lay = lpLayout(1, T, al.lay.H, al.lay.D)
      let key = $lay.H & "_" & $lay.D
      var hLp = lpArenas[key]
      linear(root & ".attn.qkvProj.weight", hNorm, hQkv, true)
      copy(hQkv, hLp, 0, lay.qkv, T * 3 * hidden)
      copy(hZero, hLp, 0, lay.zF, hidden)
      copy(hZero, hLp, 0, lay.cF, hidden * lay.D)
      for s, st in lpPrograms[key]:
        encodeMicroVm(ctx, st.prog, hLp, st.nThreads, st.tgBytes, st.tag, st.tgSize, lpNames[key][s])
      copy(hLp, hAttnIn, lay.y, 0, T * hidden)
      linear(root & ".attn.outProj.weight", hAttnIn, hAttnOut, true)
      copy(hLp, hLpState, lay.cF, 0, hidden * lay.D)
      var hAttnState = al.hWork
      encodeVmKernel(transposeKns[key], hidden * lay.D, 64, 4,
        hAttnState, al.lay.offS, gDummyFor(be), gDummyFor(be), hLpState, 0)
      copy(hLp, hAttnState, lay.zF, al.lay.offZ, hidden)
      if blk.kind == bkMambaRealGated:
        encodeVmKernel(gateInKn, T * hidden, 64, 4, hGateIn, 0, gDummyFor(be), hMix, hAttnOut, 0)
        linear(root & ".fuseGate.weight", hGateIn, hGate, true)
        encodeVmKernel(gateMixKn, T * hidden, 64, 4, hMix, 0, gDummyFor(be), hGate, hAttnOut, 0)
      else: require(sessionVecOp(be, "add", hMix, hAttnOut, hMix, T * hidden), "branch sum")
    require(sessionVecOp(be, "add", hX, hMix, hX, T * hidden), "residual")
    require(sessionEnd(be), "block sessionEnd failed: " & sessionGetLastError(be))
    sessionOpen = false
    pfAddIf(if blk.kind == bkMambaReal: "resident.R" else: "resident.P/G", t0)
  complete = true
  echo &"[infer] Resident parallel prefill OK: T={T}, blocks={m.blocks.len}; " &
    &"GPU scratch cache reused {gPrefixHits - hitsBefore}/{nextBuffer} buffers; terminal states copied GPU-to-GPU."
