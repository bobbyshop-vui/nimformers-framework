## Resident R/P/G prefix: reuse device scratch and import terminal states by
## device-to-device copies. No host activations, transposed FP32 weights, or
## backward caches between layers. The caller owns state reset/recovery.
import std/[tables, strformat, os, strutils, algorithm, times]
import backend, nimformer, jamba_hybrid, mamba2_full_vm, mamba2_ssm_gpu
import mamba2_fused_step, gpu_layers, fp4_gemv_gpu, session_vm, jit_vm
import fp4_gemv_vm
import prefill_inference_kernels
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
var gPrefixMps, gPrefixTensorCore, gPrefixPackedTiles, gPrefixFusedEpilogues: int
var gScanTuneCache = initTable[string, string]()
var gF32MpsTuneCache = initTable[string, bool]()

proc prefillExecutionStats*(): tuple[mps,tensorCore,packedTiles,fusedEpilogues:int] =
  (gPrefixMps,gPrefixTensorCore,gPrefixPackedTiles,gPrefixFusedEpilogues)

proc prefillBufferStats*(): tuple[allocations, hits, buffers: int] =
  (gPrefixAllocs, gPrefixHits, gPrefixBuffers.len)

proc clearPrefixBuffers() =
  for buf in gPrefixBuffers.mitems: sessionFree(buf.handle.kind, buf.handle)
  gPrefixBuffers.setLen(0)
  gPrefixShape = ""

proc releaseResidentPrefill*(): int =
  ## Persistent recurrent states are separate, populated by device copies.
  ## Drop prefix-only buffers before decode so its working set stays small.
  require(not gPrefixBusy, "cannot release an active prefix arena")
  for buffer in gPrefixBuffers: result += buffer.elems * sizeof(float32)
  clearPrefixBuffers()

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

proc convHistory(L: M2Lay; previousOffset = -1): VmProgram =
  # Projection is input; output is [channel, oldest..newest raw xBC].
  var p: VmProgram
  p.emitRToU(0, 0); p.emitUMulK(1, 0, uint32(L.K - 1))
  for k in 0 ..< L.K - 1:
    let t = L.T - (L.K - 1) + k
    if t >= 0:
      p.emitULoadIn(1, 0, uint32(t * L.P + L.C))
    elif previousOffset >= 0:
      p.emitUMulK(2,0,uint32(L.K-1))
      p.emitULoadConstF(1,2,uint32(previousOffset+L.K-1+t))
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

proc medianMs(samples: var seq[float64]): float64 =
  samples.sort()
  if samples.len mod 2 == 1: samples[samples.len div 2]
  else: (samples[samples.len div 2 - 1] + samples[samples.len div 2]) / 2

proc scanStateLaneCandidate(N, tg: int): int =
  ## Increase N parallelism only when the dispatch and VM shared-memory limits
  ## permit it. The runtime parity probe below can still reject this candidate.
  for lanes in [4, 2]:
    if N mod lanes == 0 and tg * lanes <= gTgCap and tg * 4 * lanes <= 1024:
      return lanes
  1

proc scanProbeSeed(L: M2Lay; arenaElems: int): seq[float32] =
  ## Small finite, nonzero inputs make the scan autotuner's parity check catch
  ## bad state indexing and reduction races instead of comparing zero outputs.
  result = newSeq[float32](arenaElems)
  for h in 0 ..< L.H:
    result[L.aLog + h] = -0.35'f32 - float32(h mod 5) * 0.025'f32
    result[L.dSkip + h] = 0.2'f32 + float32(h mod 7) * 0.03'f32
  for t in 0 ..< L.T:
    for h in 0 ..< L.H:
      result[L.dt + t * L.H + h] = 0.05'f32 +
        float32((t * 3 + h * 5) mod 13) * 0.002'f32
      let hdBase = t * L.cDim + h * L.D
      for d in 0 ..< L.D:
        result[L.conv + hdBase + d] =
          float32(((t * 11 + h * 3 + d * 7) mod 19) - 9) * 0.004'f32
      let group = h div (L.H div L.G)
      for n in 0 ..< L.N:
        let bOff = L.conv + t * L.cDim + L.C + group * L.N + n
        let cOff = L.conv + t * L.cDim + L.C + L.GN + group * L.N + n
        result[bOff] = 0.3'f32 + float32((t + h + n) mod 11) * 0.01'f32
        result[cOff] = float32(((t * 5 + h + n * 3) mod 17) - 8) * 0.015'f32
  for hd in 0 ..< L.HD:
    for n in 0 ..< L.N:
      result[L.states + hd * L.N + n] =
        float32(((hd * 3 + n * 7) mod 13) - 6) * 0.01'f32

proc verifyScanStateLanes(ctx: Backend; L: M2Lay; tg, candidateLanes: int;
                          probe: SessionHandle;
                          baselineProg, candidateProg: VmProgram;
                          baselineKn, candidateKn: string): tuple[ok: bool, maxError, tolerance: float32] =
  let be = ctx.kind.toByby()
  let yCount = L.T * L.C
  let stateCount = L.HD * L.N
  let span = yCount + stateCount
  var readback = sessionAllocScratch(be, 2 * span)
  let readbackReady = if be == byby.gbVulkan: readback.vulkanHandle.buf != nil
                      else: readback.metalHandle != nil
  require(readbackReady, "scan parity readback allocation failed")
  var sessionOpen = false
  try:
    require(sessionBegin(be), "scan parity sessionBegin failed")
    sessionOpen = true
    encodeMicroVm(ctx, baselineProg, probe, L.HD * tg, tg * 16,
      "scan_parity_baseline", tg, baselineKn)
    require(sessionCopyInto(be, probe, readback, L.yPre, 0, yCount),
      "scan parity baseline y copy failed")
    require(sessionCopyInto(be, probe, readback,
      L.states + L.T * stateCount, yCount, stateCount),
      "scan parity baseline terminal-state copy failed")
    encodeMicroVm(ctx, candidateProg, probe, L.HD * tg * candidateLanes,
      tg * 16 * candidateLanes, "scan_parity_candidate", tg * candidateLanes,
      candidateKn)
    require(sessionCopyInto(be, probe, readback, L.yPre, span, yCount),
      "scan parity candidate y copy failed")
    require(sessionCopyInto(be, probe, readback,
      L.states + L.T * stateCount, span + yCount, stateCount),
      "scan parity candidate terminal-state copy failed")
    let ended = sessionEnd(be)
    sessionOpen = false
    require(ended, "scan parity sessionEnd failed: " & sessionGetLastError(be))
    let got = sessionRead(be, readback, 2 * span)
    if got.len < 2 * span: return (false, 1.0e30'f32, 0'f32)
    var maxAbs, maxValue = 0'f32
    for i in 0 ..< span:
      let a = got[i]
      let b = got[span + i]
      if a != a or b != b or abs(a) > 1e20'f32 or abs(b) > 1e20'f32:
        return (false, 1.0e30'f32, 0'f32)
      maxAbs = max(maxAbs, abs(a - b))
      maxValue = max(maxValue, max(abs(a), abs(b)))
    result.maxError = maxAbs
    result.tolerance = max(2e-5'f32, maxValue * 1e-4'f32)
    result.ok = maxAbs <= result.tolerance
  finally:
    if sessionOpen: discard sessionEnd(be)
    sessionFree(be, readback)

proc chooseScanMode*(ctx: Backend; T, H, D, N, G, K, C: int; eps: float32;
                     groups: int; continuation: bool): string =
  ## Compare the real GPU kernels once per backend/shape. The parallel probe
  ## has a full state per timestep, so its temporary allocation is bounded.
  let requested = getEnv("BYBY_PREFILL_SCAN", "auto")
  if requested != "auto": return requested
  let be = ctx.kind.toByby()
  let cacheKey = $be & "_" & $T & "_" & $H & "_" & $D & "_" & $N & "_" &
    $G & "_" & $K & "_" & $C & "_" & $groups & "_" & $continuation
  if cacheKey in gScanTuneCache: return gScanTuneCache[cacheKey]

  result = "serial"
  let tg = m2ScanTgSize(T)
  if tg <= 0:
    gScanTuneCache[cacheKey] = result
    echo &"[prefill-scan] auto T={T}: parallel scan unavailable; serial"
    return
  let serialL = prefixLayout(1, T, H, D, N, G, K, C, eps, groups, compact = true)
  let parallelL = prefixLayout(1, T, H, D, N, G, K, C, eps, groups, compact = false)
  let serialArena = serialL.fwdEnd + serialL.cDim * (serialL.K - 1)
  let parallelArena = parallelL.fwdEnd + parallelL.cDim * (parallelL.K - 1)
  let maxProbeMiB = try: parseInt(getEnv("BYBY_PREFILL_SCAN_AUTOTUNE_MAX_MIB", "512"))
                    except ValueError: 512
  if maxProbeMiB <= 0 or parallelArena > maxProbeMiB * 1024 * 1024 div sizeof(float32):
    gScanTuneCache[cacheKey] = result
    echo &"[prefill-scan] auto T={T}: parallel probe needs " &
      &"{parallelArena * sizeof(float32) div (1024*1024)} MiB (limit {maxProbeMiB}); serial"
    return

  var serialSamples, baselineSamples, candidateSamples: seq[float64]
  let tuneStart = getMonoTimeIfPf()
  var probe: SessionHandle
  var probeReady = false
  try:
    let serialProg = buildCompactPrefixScanProgram(serialL)
    let baselineProg = buildPrefixParallelScanProgram(parallelL, tg, continuation, 1)
    let candidateLanes = scanStateLaneCandidate(N, tg)
    let candidateProg = if candidateLanes > 1:
      buildPrefixParallelScanProgram(parallelL, tg, continuation, candidateLanes)
      else: baselineProg
    let serialTag = "@prefix_compact_" & m2Tag(serialL) & "_coalescedfalse"
    let baselineTag = "@prefix_parallel_carry_" & m2Tag(parallelL) & "_" & $continuation & "_n1"
    let candidateTag = "@prefix_parallel_carry_" & m2Tag(parallelL) & "_" & $continuation & "_n" & $candidateLanes
    checkVmScratchCapacity(ctx, tg * 4 * candidateLanes, "parallel scan autotune")
    let serialKn = vmKernelForBe(serialProg, serialTag, be)
    let baselineKn = vmKernelForBe(baselineProg, baselineTag, be)
    let candidateKn = if candidateLanes > 1:
      vmKernelForBe(candidateProg, candidateTag, be)
      else: baselineKn
    if be == byby.gbVulkan:
      var jobs = @[
        (serialKn, jit_vm.lowerBytecodeToGlslCached(serialProg, serialKn)),
        (baselineKn, jit_vm.lowerBytecodeToGlslCached(baselineProg, baselineKn))]
      if candidateLanes > 1:
        jobs.add((candidateKn, jit_vm.lowerBytecodeToGlslCached(candidateProg, candidateKn)))
      byby.sessionJitPrewarm(be, jobs)
    var candidateAccepted = false
    var seed = scanProbeSeed(parallelL, max(serialArena, parallelArena))
    probe = sessionUpload(be, seed)
    seed = @[]
    probeReady = if be == byby.gbVulkan: probe.vulkanHandle.buf != nil
                 else: probe.metalHandle != nil
    require(probeReady, "scan autotune scratch allocation failed")
    if candidateLanes > 1:
      let parity = verifyScanStateLanes(ctx, parallelL, tg, candidateLanes,
        probe, baselineProg, candidateProg, baselineKn, candidateKn)
      if parity.ok:
        candidateAccepted = true
        echo &"[prefill-scan] auto T={T}: state-lanes={candidateLanes} parity " &
          &"passed (max-error={parity.maxError:.3g} tolerance={parity.tolerance:.3g})"
      else:
        echo &"[prefill-scan] auto T={T}: state-lanes={candidateLanes} rejected " &
          &"(max-error={parity.maxError:.3g} tolerance={parity.tolerance:.3g}); using 1"
    proc sampleBatch(prog: VmProgram; kn: string; nThreads, tgBytes, tgSize,
                     repeats: int): float64 =
      require(sessionBegin(be), "scan autotune sessionBegin failed")
      let t0 = getMonoTime()
      for _ in 0 ..< repeats:
        encodeMicroVm(ctx, prog, probe, nThreads, tgBytes, "scan_autotune", tgSize, kn)
      require(sessionEnd(be), "scan autotune sessionEnd failed: " & sessionGetLastError(be))
      float64((getMonoTime() - t0).inNanoseconds) / 1e6 / float64(repeats)
    # Warm all candidates in one command buffer. Vulkan fence waits are costly
    # on MoltenVK, so measure batches of dispatches instead of ending a session
    # after every single sample.
    require(sessionBegin(be), "scan autotune warmup sessionBegin failed")
    encodeMicroVm(ctx, serialProg, probe, serialL.HD, 4, "scan_autotune_warm", 0, serialKn)
    encodeMicroVm(ctx, baselineProg, probe, parallelL.HD * tg, tg * 16,
      "scan_autotune_warm", tg, baselineKn)
    if candidateAccepted:
      encodeMicroVm(ctx, candidateProg, probe, parallelL.HD * tg * candidateLanes,
        tg * 16 * candidateLanes, "scan_autotune_warm", tg * candidateLanes, candidateKn)
    require(sessionEnd(be), "scan autotune warmup sessionEnd failed: " & sessionGetLastError(be))
    # Alternate candidate order; each command buffer has two repeated dispatches.
    if candidateAccepted:
      for i in 0 ..< 2:
        if i == 0:
          serialSamples.add sampleBatch(serialProg, serialKn, serialL.HD, 4, 0, 2)
          baselineSamples.add sampleBatch(baselineProg, baselineKn,
            parallelL.HD * tg, tg * 16, tg, 2)
          candidateSamples.add sampleBatch(candidateProg, candidateKn,
            parallelL.HD * tg * candidateLanes, tg * 16 * candidateLanes,
            tg * candidateLanes, 2)
        else:
          candidateSamples.add sampleBatch(candidateProg, candidateKn,
            parallelL.HD * tg * candidateLanes, tg * 16 * candidateLanes,
            tg * candidateLanes, 2)
          baselineSamples.add sampleBatch(baselineProg, baselineKn,
            parallelL.HD * tg, tg * 16, tg, 2)
          serialSamples.add sampleBatch(serialProg, serialKn, serialL.HD, 4, 0, 2)
    else:
      for i in 0 ..< 2:
        if i mod 2 == 0:
          serialSamples.add sampleBatch(serialProg, serialKn, serialL.HD, 4, 0, 2)
          baselineSamples.add sampleBatch(baselineProg, baselineKn,
            parallelL.HD * tg, tg * 16, tg, 2)
        else:
          baselineSamples.add sampleBatch(baselineProg, baselineKn,
            parallelL.HD * tg, tg * 16, tg, 2)
          serialSamples.add sampleBatch(serialProg, serialKn, serialL.HD, 4, 0, 2)
    let serialMs = medianMs(serialSamples)
    let baselineMs = medianMs(baselineSamples)
    var bestMs = serialMs
    result = "serial"
    if baselineMs < bestMs * 0.97:
      bestMs = baselineMs
      result = "parallel"
    if candidateAccepted:
      let candidateMs = medianMs(candidateSamples)
      if candidateMs < bestMs * 0.97:
        bestMs = candidateMs
        result = "parallel_n" & $candidateLanes
      echo &"[prefill-scan] auto T={T}: serial={serialMs:.3f}ms " &
        &"parallel-n1={baselineMs:.3f}ms parallel-n{candidateLanes}={candidateMs:.3f}ms " &
        &"-> {result}"
    else:
      echo &"[prefill-scan] auto T={T}: serial={serialMs:.3f}ms " &
        &"parallel-n1={baselineMs:.3f}ms -> {result}"
  except CatchableError as e:
    discard sessionRecoverQueue(be)
    result = "serial"
    echo "[prefill-scan] auto benchmark failed; fallback serial: " & e.msg
  finally:
    if probeReady: sessionFree(be, probe)
    pfAddIf("scan.autotune", tuneStart)
  gScanTuneCache[cacheKey] = result

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

proc forwardResidentPrefill*(m: var NimformerModelJamba; ids: seq[int]; ctx: Backend; continuation = false) =
  ensureTgCap(ctx)
  require(ids.len >= 2 and ids.len <= 128, "prefix length must be 2..128")
  require(residentPrefillReason(m,ctx).len == 0, residentPrefillReason(m,ctx))
  let be = ctx.kind.toByby()
  gPrefixMps=0;gPrefixTensorCore=0;gPrefixPackedTiles=0;gPrefixFusedEpilogues=0
  for id in ids: require(id >= 0 and id < m.vocab, "invalid prefix token ID")
  let defaultCommitLayers = if be == gbVulkan: "8" else: "4"
  let commitLayers = parseInt(getEnv("BYBY_PREFILL_LAYERS_PER_COMMIT", defaultCommitLayers))
  require(commitLayers > 0, "BYBY_PREFILL_LAYERS_PER_COMMIT must be positive")
  let T = ids.len
  let tileM = if T < 8: 4 else: 8
  let padded = ((T + tileM - 1) div tileM) * tileM
  let hidden = m.embedDim
  let mr = m.blocks[0].mambaReal
  var useMps = false
  when defined(macosx):
    useMps = be == gbMetal and getEnv("BYBY_PREFILL_MPS", "1") != "0" and
      metalBackend.metalIsAppleSilicon() and metalBackend.metalMpsSupported()
  if be == gbMetal:
    when defined(macosx):
      echo &"[prefill-matrix] Apple Silicon={metalBackend.metalIsAppleSilicon()} " &
        &"MPS eligible={useMps}"
    else:
      discard
  let requestedScan = getEnv("BYBY_PREFILL_SCAN","auto")
  require(requestedScan in ["auto","serial","coalesced","parallel"], "invalid BYBY_PREFILL_SCAN")
  let scanMode=if requestedScan=="auto":
      chooseScanMode(ctx,T,mr.numHeads,mr.headDim,mr.dState,mr.nGroups,
        mr.convKernel,mr.intermediateSize,mr.normEps,mr.nGroups,continuation)
    else:requestedScan
  let parallel = scanMode.startsWith("parallel")
  let parallelStateLanes =
    if scanMode.startsWith("parallel_n"): parseInt(scanMode["parallel_n".len .. ^1])
    else: 1
  let coalesced = scanMode == "coalesced"
  let L = prefixLayout(1, T, mr.numHeads, mr.headDim, mr.dState,
    mr.nGroups, mr.convKernel, mr.intermediateSize, mr.normEps, mr.nGroups, compact = not parallel)
  let historyOff = L.fwdEnd
  let arenaSize = historyOff + L.cDim*(L.K-1)
  let terminalOff = L.states + (if coalesced: 0 elif parallel: T else: 1)*L.HD*L.N
  checkVmScratchCapacity(ctx, arenaSize, "resident prefill")
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
  # BYBY_PREFILL_DIAG=1: commit + wait after every stage so prefill-profile
  # attributes real GPU time instead of only host encode time. This adds one
  # sync per stage, so it is a diagnostic mode, not a fast path.
  # BYBY_PREFILL_DIAG=2 additionally syncs between the 8-row tile and the 4-row
  # tail of every packed GEMM and inside the Linear++ sidecar, so a stage that
  # looks expensive can be split without reading the shader.
  let diagLevel =
    block:
      let v = getEnv("BYBY_PREFILL_DIAG", "0")
      if v == "2": 2
      elif v == "3": 3
      elif v.len > 0 and v != "0": 1
      else: 0
  let diagStages = diagLevel >= 1
  proc flushStage(stage: string) =
    if not diagStages: return
    let t0 = getMonoTimeIfPf()
    if not sessionEnd(be):
      raise newException(CatchableError, "diag flush failed: " & sessionGetLastError(be))
    sessionOpen = false
    pfAddIf("gpu." & stage, t0)
    require(sessionBegin(be), "diag reopen failed")
    sessionOpen = true
  proc flushFine(stage: string) =
    if diagLevel >= 2: flushStage(stage)
  # Allocate once for the entire stack. Padding is zero only in input buffers;
  # M4 output rows never overlap the packed scan's small parameters.
  var hX = alloc(padded * hidden, true)
  var hNorm = alloc(padded * hidden, true)
  var hProjection = alloc(padded * L.P)
  var hInner = alloc(padded * L.C, true)
  var hMix = alloc(padded * hidden)
  var hMamba = alloc(arenaSize)
  var hConv = alloc(L.cDim * (L.K - 1))
  var hLora = alloc(T * maxRank)
  let zeroLen = max(L.HD * L.N, max(T * hidden, maxState))
  let hZero = alloc(zeroLen, true)
  let gemmMode = getEnv("BYBY_PREFILL_GEMM","auto")
  require(gemmMode in ["auto","mps","m8"], "invalid BYBY_PREFILL_GEMM")
  # MPS needs an expanded FP32 copy of every FP4 weight, which costs far more
  # bandwidth than the packed kernels themselves on integrated GPUs. Keep it
  # opt-in (BYBY_PREFILL_GEMM=mps) instead of the default.
  let expandFp4 = useMps and T >= 8 and gemmMode == "mps"
  if gemmMode == "mps": require(useMps, "MPS is unavailable")
  var expandedSize = 0
  if expandFp4:
    for name,lin in gLinCache:
      if name.startsWith("block") and lin.isFp4:
        expandedSize = max(expandedSize,lin.K*min(lin.N,4096))
  var hExpanded,hTile: SessionHandle
  let tileLoraOff=T*4096
  let tileBiasOff=tileLoraOff+T*maxRank
  if expandedSize > 0:
    checkVmScratchCapacity(ctx,expandedSize,"prefill GPU weight tile")
    hExpanded=alloc(expandedSize)
    hTile=alloc(tileBiasOff+4096)
  let embed = m.embed.lookupBatch(@[ids], ctx)
  let hEmbed = sessionUpload(be, embed.data)
  owned.add hEmbed
  require(sessionBegin(be), "embedding sessionBegin failed")
  sessionOpen = true
  copy(hEmbed, hX, 0, 0, T * hidden)
  require(sessionEnd(be), "embedding sessionEnd failed")
  sessionOpen = false

  # Programs are compiled before opening a command buffer (no nested sessions).
  let historyProg = convHistory(L, if continuation: gLayersV2[0].off.offConvSt else: -1)
  let historyKn = vmKernelForBe(historyProg,
    "prefix_conv_" & m2Tag(L) & "_carry" & $continuation, be)
  let scanStep: MvmStep =
    if parallel:
      let tg=m2ScanTgSize(T)
      require(tg>0,"parallel scan unavailable")
      (buildPrefixParallelScanProgram(L,tg,continuation,parallelStateLanes),
        L.HD*tg*parallelStateLanes,tg*16*parallelStateLanes,tg*parallelStateLanes,
        "@prefix_parallel_carry_" & m2Tag(L) & "_" & $continuation & "_n" & $parallelStateLanes)
    else:
      (buildCompactPrefixScanProgram(L,coalesced),L.HD,4,0,
        "@prefix_compact_" & m2Tag(L) & "_coalesced" & $coalesced)
  var steps: seq[MvmStep] = @[
    (buildPrefixConvDtProgram(L,if continuation: historyOff else: -1),T*M2Tg,4,M2Tg,
      "@prefix_conv_dt_carry_" & m2Tag(L) & "_" & $continuation)]
  if coalesced:
    steps.add((buildPrefixStateTransposeProgram(L,true),L.HD*L.N,4,256,
      "@prefix_state_in_" & m2Tag(L)))
  steps.add(scanStep)
  if coalesced:
    steps.add((buildPrefixStateTransposeProgram(L,false),L.HD*L.N,4,256,
      "@prefix_state_out_" & m2Tag(L)))
  steps.add((buildM2RmsFwdProgram(L),T*L.normGroups*rmsTg(L),rmsTg(L)*4,rmsTg(L),
    "@prefix_rms_" & m2Tag(L) & "_compact" & $(not parallel)))
  var stepNames: seq[string]
  for st in steps: stepNames.add kernelNameFor(ctx, st.prog, st.tag)
  var biasKns = initTable[int, string]()
  var loraKns = initTable[string, string]()
  var m8Kns = initTable[string, string]()
  var m4Kns = initTable[string, string]()
  var m8x2Kns = initTable[string, string]()
  var m4x2Kns = initTable[string, string]()
  var f32Kns = initTable[string, string]()
  var expandKns=initTable[string,seq[tuple[kn,scatter: string,row,count,tg: int]]]()
  let useM8 = T >= 8 and getEnv("BYBY_PREFILL_FP4_KERNEL", "auto") != "m4"
  # Only the Linear++ sidecar linears pass bias=true; the Mamba in/out
  # projections never do, so their bias kernels are not worth compiling.
  var biasCols: seq[int]
  for i, blk in m.blocks:
    if blk.kind in {bkMambaRealPlusAttn, bkMambaRealGated}:
      for suffix in ["block" & $i & ".attn.qkvProj.weight",
                     "block" & $i & ".attn.outProj.weight",
                     "block" & $i & ".fuseGate.weight"]:
        if suffix in gLinCache and gLinCache[suffix].N notin biasCols:
          biasCols.add gLinCache[suffix].N
  for name, lin in gLinCache:
    if not name.startsWith("block"): continue
    if lin.N in biasCols and lin.N notin biasKns:
      biasKns[lin.N] = vmKernelForBe(biasRows(lin.N), "prefix_bias_" & $lin.N, be)
    if name in gLoraGpu:
      let ad = gLoraGpu[name]
      loraKns[name] = vmKernelForBe(loraRows(lin.N, ad.r, ad.scale),
        "prefix_lora_" & $lin.N & "_" & $ad.r & "_" & $ad.scale, be)
    if lin.isFp4 and expandFp4:
      let rank=if name in gLoraGpu and not gDisableLora:gLoraGpu[name].r else:0
      let scale=if rank>0:gLoraGpu[name].scale else:1'f32
      require(lin.K mod 32 == 0 and (lin.K*lin.N div 2) mod 256 == 0,"unsupported FP4 tile shape")
      for row in countup(0,lin.N-1,4096):
        let count=min(4096,lin.N-row)
        var tg=64
        while count mod tg != 0: tg=tg div 2
        let kn=vmKernelForBe(buildPrefixExpandFp4(lin.K,lin.N,row),
          "prefix_expand_" & $lin.K & "_" & $lin.N & "_" & $row,be)
        let scatter=vmKernelForBe(buildPrefixScatter(lin.N,count,row,tg,rank,scale,
          tileLoraOff,if lin.hasBias:tileBiasOff else: -1),
          "prefix_scatter_fused_" & $T & "_" & $lin.N & "_" & $count & "_" & $row &
          "_" & $rank & "_" & $scale & "_" & $lin.hasBias,be)
        expandKns.mgetOrPut(name,@[]).add((kn,scatter,row,count,tg))
    if lin.isFp4 and useM8 and not expandFp4:
      # Keep the autotuned threadgroup width but use the two-words-in-flight
      # variant: the tuned tile serialises one weight word -> rows*8 FMAs per
      # iteration and measured 17-19 GMAC/s (~1.5x slower than this one).
      let tg8 = if gFp4CacheGpu[name].m8TgSize > 0: gFp4CacheGpu[name].m8TgSize
                else: GemvTg
      let unrollW = (if getEnv("BYBY_PREFILL_UNROLL", "2") == "1": 1 else: 2)
      let kUnroll = tg8 * 8 * unrollW
      if unrollW == 2 and lin.K mod kUnroll == 0:
        m8Kns[name] = vmKernelForBe(
          buildFp4KmGemmUnrollProgram(lin.K, lin.N, tg8, 2, 8),
          "fp4kmu2_m8_" & $lin.K & "x" & $lin.N & "_tg" & $tg8, be)
      else:
        let prepared = gFp4CacheGpu[name].m8JitName
        m8Kns[name] = if prepared.len > 0: prepared
                     else: vmKernelForBe(buildFp4KmGemmM8Program(lin.K, lin.N),
                       "fp4km_m8_" & $lin.K & "x" & $lin.N, be)
      # Tail tile for the last 1..4 rows of the prefix. The 8-row tile rounds
      # T up to a multiple of 8 (an 11-token prompt becomes 16 rows of work);
      # a 4-row tile caps that at 12 rows for the same weight traffic.
      if unrollW == 2 and lin.K mod (GemvTg * 16) == 0 and getEnv("BYBY_PREFILL_TAIL4", "1") != "0":
        m4Kns[name] = vmKernelForBe(
          buildFp4KmGemmUnrollProgram(lin.K, lin.N, GemvTg, 2, 4),
          "fp4kmu2_m4_" & $lin.K & "x" & $lin.N, be)
      elif lin.K mod (GemvTg * 8) == 0 and getEnv("BYBY_PREFILL_TAIL4", "1") != "0":
        m4Kns[name] = vmKernelForBe(
          buildFp4KmGemmTunedProgram(lin.K, lin.N, GemvTg, 4),
          "fp4km4g_" & $lin.K & "x" & $lin.N, be)
      # The 2-column tile reuses each float4 activation load across adjacent
      # output columns. It has 32 live registers (16 accumulators for 8x2),
      # enabled by the expanded micro-VM register encoding.
      if getEnv("BYBY_PREFILL_2COL", "1") != "0" and lin.N mod 2 == 0 and
          lin.K mod (GemvTg * 8) == 0:
        m8x2Kns[name] = vmKernelForBe(
          buildFp4KmGemm2ColProgram(lin.K, lin.N, GemvTg, 8),
          "fp4km2c_m8_" & $lin.K & "x" & $lin.N, be)
        if getEnv("BYBY_PREFILL_TAIL4", "1") != "0":
          m4x2Kns[name] = vmKernelForBe(
            buildFp4KmGemm2ColProgram(lin.K, lin.N, GemvTg, 4),
            "fp4km2c_m4_" & $lin.K & "x" & $lin.N, be)
    elif not lin.isFp4 and T <= 12 and lin.K mod GemvTg == 0:
      # FP32 sidecar (block G: qkvProj/outProj/fuseGate). MPS runs these at
      # ~1.7 GFLOP/s for an 11-row prompt (654 ms for one qkvProj), so batch
      # the rows in our own kernel instead: one weight read for all T rows.
      f32Kns[name] = vmKernelForBe(
        buildF32GemmBatchProgram(lin.K, lin.N, GemvTg, T),
        "prefix_f32gemm_" & $lin.K & "x" & $lin.N & "_r" & $T, be)
  var f32Mps, f32Fallback, f32Batch: int
  proc mpsMslWins(name: string; input, weight: SessionHandle;
                  output: var SessionHandle; rows, k, cols: int;
                  batchKernel: bool): bool =
    when defined(macosx):
      if not useMps or rows <= 1 or k > 16384 or cols > 16384: return false
      if batchKernel and name notin f32Kns: return false
      let key = $rows & "_" & $k & "_" & $cols & "_" & $batchKernel
      if key in gF32MpsTuneCache: return gF32MpsTuneCache[key]
      if sessionOpen:
        require(sessionEnd(be), "MPS autotune preflush failed")
        sessionOpen = false
      if output.metalHandle == nil:
        output.metalHandle = metalBackend.metalSessionAllocScratch(rows * cols)
      let outputHandle = output
      let tuneStart = getMonoTimeIfPf()
      proc run(mps: bool): float64 =
        require(sessionBegin(be), "MPS autotune sessionBegin failed")
        sessionOpen = true
        let t0 = getMonoTime()
        if mps:
          require(metalBackend.metalSessionMatmulMpsNTEnc(input.metalHandle,
            weight.metalHandle, outputHandle.metalHandle, rows, k, cols, cols, 0),
            "MPS candidate failed")
        elif batchKernel:
          encodeVmKernel(f32Kns[name], cols * GemvTg, GemvTg,
            f32GemmBatchTgBytes(rows, GemvTg), outputHandle, 0,
            gDummyFor(be), weight, input, 0)
        else:
          var candidateOutput = outputHandle
          require(byby.sessionMatmulNTAccelerated(be, input, weight, candidateOutput,
            rows, k, cols), "MSL NT candidate failed")
        let ok = sessionEnd(be)
        sessionOpen = false
        require(ok, "MPS autotune sessionEnd failed: " & sessionGetLastError(be))
        float64((getMonoTime() - t0).inNanoseconds) / 1e6
      try:
        # Warm each backend first, then alternate synchronized samples. Both
        # candidates use the real activation and weight for this shape.
        discard run(false)
        discard run(true)
        var mslSamples, mpsSamples: seq[float64]
        for i in 0 ..< 4:
          if i mod 2 == 0:
            mslSamples.add run(false)
            mpsSamples.add run(true)
          else:
            mpsSamples.add run(true)
            mslSamples.add run(false)
        let mslMs = medianMs(mslSamples)
        let mpsMs = medianMs(mpsSamples)
        result = mpsMs < mslMs * 0.97
        gF32MpsTuneCache[key] = result
        echo &"[prefill-matrix] {k}x{cols} M={rows}: " &
          (if batchKernel: "MSL-batch" else: "MSL-NT") & &"={mslMs:.3f}ms " &
          &"MPS={mpsMs:.3f}ms -> " & (if result: "MPS" else: "MSL")
      except CatchableError as e:
        if sessionOpen:
          discard sessionEnd(be)
          sessionOpen = false
        discard sessionRecoverQueue(be)
        gF32MpsTuneCache[key] = false
        echo "[prefill-matrix] MPS autotune failed; use MSL: " & e.msg
        result = false
      finally:
        if not sessionOpen:
          require(sessionBegin(be), "could not reopen prefill session after MPS autotune")
          sessionOpen = true
        pfAddIf("matrix.autotune", tuneStart)
    else:
      discard name; discard input; discard weight; discard output
      discard rows; discard k; discard cols; discard batchKernel
      result = false

  proc matmul(input, weight: SessionHandle; output: var SessionHandle;
              rows, k, cols: int): bool =
    if mpsMslWins("nt_" & $k & "x" & $cols, input, weight, output,
        rows, k, cols, false):
      when defined(macosx):
        require(metalBackend.metalSessionMatmulMpsNTEnc(input.metalHandle,
          weight.metalHandle, output.metalHandle, rows, k, cols, cols, 0),
          "selected MPS NT matmul failed")
        inc f32Mps
        result = true
      else:
        discard
    else:
      inc f32Fallback
      result = byby.sessionMatmulNTAccelerated(be, input, weight, output, rows, k, cols)

  var pfSeen = initTable[string, bool]()
  proc linear(name: string; input: SessionHandle; output: var SessionHandle;
              bias: bool) =
    let lin = gLinCache[name]
    if diagLevel >= 2 and name notin pfSeen:
      pfSeen[name] = true
      echo &"[pf] {name} K={lin.K} N={lin.N} fp4={lin.isFp4} m8={name in m8Kns} " &
        &"m4={name in m4Kns} bias={lin.hasBias}"
    let fusedEpilogue=name in expandKns
    let hasLora=name in gLoraGpu and not gDisableLora
    if hasLora:
      let ad=gLoraGpu[name]
      require(matmul(input,ad.aBuf,hLora,T,lin.K,ad.r),"LoRA A " & name)
    if name in expandKns:
      let c=gFp4CacheGpu[name]
      for tile in expandKns[name]:
        encodeVmKernel(tile.kn,lin.K*tile.count div 2,256,4,hExpanded,0,c.wBuf,c.lutBuf,input,0)
        when defined(macosx):
          require(metalBackend.metalSessionMatmulMpsNTEnc(input.metalHandle,hExpanded.metalHandle,
            hTile.metalHandle,T,lin.K,tile.count,tile.count,0),"prefill MPS " & name)
          inc gPrefixMps
        if hasLora:copy(hLora,hTile,0,tileLoraOff,T*gLoraGpu[name].r)
        if lin.hasBias and bias:copy(lin.biasH,hTile,tile.row,tileBiasOff,tile.count)
        # mamba in/out projections have no bias; only sidecar calls enable it.
        elif lin.hasBias:copy(hZero,hTile,0,tileBiasOff,tile.count)
        encodeVmKernel(tile.scatter,T*tile.count,tile.tg,4,output,0,c.wBuf,
          if hasLora:gLoraGpu[name].bBuf else:gDummyFor(be),hTile,0)
        inc gPrefixFusedEpilogues
    # Vulkan's 8x2 packed microkernel is a useful fallback, but it currently
    # gets selected before the FP4 cooperative-matrix GEMM below. On GPUs such
    # as the Colab T4, that meant three packed dispatches per projection and
    # zero tensor-core GEMMs even though KHR_cooperative_matrix was available.
    # Try the single-dispatch FP4 cooperative path first on Vulkan; if the
    # shape, shader compiler, or device does not support it, fall through to
    # the existing 8x2 kernel. Keep Metal's tuned 8x2 route unchanged.
    elif lin.isFp4 and be == gbVulkan and byby.sessionFp4GemmTensorCore(be, input,
        gFp4CacheGpu[name].wBuf, gFp4CacheGpu[name].lutBuf, output, T, lin.K, lin.N):
      inc gPrefixTensorCore
    elif name in m8x2Kns:
      let c = gFp4CacheGpu[name]
      var row = 0
      while row + 8 <= T:
        inc gPrefixPackedTiles
        encodeVmKernel(m8x2Kns[name], (lin.N div 2) * GemvTg, GemvTg,
          (Fp4KmLutLen + 16 * GemvTg) * 4, output, row * lin.N,
          c.wBuf, c.lutBuf, input, row * lin.K)
        row += 8
        flushFine("tile8x2_" & $lin.N & "x" & $lin.K)
      if row < T:
        inc gPrefixPackedTiles
        if row + 4 <= T and T - row <= 4 and name in m4x2Kns:
          encodeVmKernel(m4x2Kns[name], (lin.N div 2) * GemvTg, GemvTg,
            (Fp4KmLutLen + 8 * GemvTg) * 4, output, row * lin.N,
            c.wBuf, c.lutBuf, input, row * lin.K)
          flushFine("tile4x2_" & $lin.N & "x" & $lin.K)
        else:
          # For a 1..3 row tail, the 4-row program reads zero-padded input
          # rows from hX and writes only into the already padded output arena.
          let tailKernel = if name in m4x2Kns: m4x2Kns[name] else: m8x2Kns[name]
          let tailRows = if name in m4x2Kns: 4 else: 8
          encodeVmKernel(tailKernel, (lin.N div 2) * GemvTg, GemvTg,
            (Fp4KmLutLen + tailRows * 2 * GemvTg) * 4, output, row * lin.N,
            c.wBuf, c.lutBuf, input, row * lin.K)
          flushFine("tile" & $tailRows & "x2tail_" & $lin.N & "x" & $lin.K)
    elif lin.isFp4 and byby.sessionFp4GemmTensorCore(be, input,
        gFp4CacheGpu[name].wBuf, gFp4CacheGpu[name].lutBuf, output, T, lin.K, lin.N):
      inc gPrefixTensorCore
    elif name in m8Kns:
      let c = gFp4CacheGpu[name]
      let tg = if c.m8TgSize > 0: c.m8TgSize else: GemvTg
      var row = 0
      while row + 8 <= T:
        inc gPrefixPackedTiles
        encodeVmKernel(m8Kns[name], lin.N * tg, tg,
          (Fp4KmLutLen + 8 * tg) * 4, output, row * lin.N,
          c.wBuf, c.lutBuf, input, row * lin.K)
        row += 8
        flushFine("tile8_" & $lin.N & "x" & $lin.K)
      if row < T:
        inc gPrefixPackedTiles
        # 4-row tail: reads at most 3 zero-padded rows, writes rows that the
        # scan never consumes, so the extra rows stay inside the padded arena.
        # A tail wider than 4 rows needs the 8-row tile anyway.
        if name in m4Kns and T - row <= 4:
          let tg4 = GemvTg
          encodeVmKernel(m4Kns[name], lin.N * tg4, tg4,
            (Fp4KmLutLen + 4 * tg4) * 4, output, row * lin.N,
            c.wBuf, c.lutBuf, input, row * lin.K)
          flushFine("tile4_" & $lin.N & "x" & $lin.K)
        else:
          encodeVmKernel(m8Kns[name], lin.N * tg, tg,
            (Fp4KmLutLen + 8 * tg) * 4, output, row * lin.N,
            c.wBuf, c.lutBuf, input, row * lin.K)
          flushFine("tile8tail_" & $lin.N & "x" & $lin.K)
    elif lin.isFp4:
      inc gPrefixPackedTiles
      require(encodeFp4GemmRows(gFp4CacheGpu[name], input, output, T), "FP4 GEMM " & name)
    elif name in f32Kns:
      if mpsMslWins(name, input, lin.wF32, output, T, lin.K, lin.N, true):
        when defined(macosx):
          require(metalBackend.metalSessionMatmulMpsNTEnc(input.metalHandle,
            lin.wF32.metalHandle, output.metalHandle, T, lin.K, lin.N, lin.N, 0),
            "selected MPS GEMM failed for " & name)
          inc f32Mps
        else:
          discard
      elif T >= 8 and byby.sessionMatmulNTFp16Coop(be, input, lin.wF32,
          output, T, lin.K, lin.N):
        # Vulkan FP16 cooperative matrix first, with FP32 accumulation/output.
        # Unsupported hardware or shapes return false and keep the VM fallback.
        inc gPrefixTensorCore
      else:
        inc f32Batch
        encodeVmKernel(f32Kns[name], lin.N * GemvTg, GemvTg,
          f32GemmBatchTgBytes(T, GemvTg), output, 0,
          gDummyFor(be), lin.wF32, input, 0)
        if diagLevel >= 3:
          # Same dispatch twice: tells a cold-buffer cost apart from a slow kernel.
          flushStage("f32.first_" & $lin.N & "x" & $lin.K)
          encodeVmKernel(f32Kns[name], lin.N * GemvTg, GemvTg,
            f32GemmBatchTgBytes(T, GemvTg), output, 0,
            gDummyFor(be), lin.wF32, input, 0)
          flushStage("f32.second_" & $lin.N & "x" & $lin.K)
    else:
      require(matmul(input, lin.wF32, output, T, lin.K, lin.N), "FP32 GEMM " & name)
    if hasLora and not fusedEpilogue:
      let ad = gLoraGpu[name]
      encodeVmKernel(loraKns[name], T * lin.N, 64, 4,
        output, 0, gDummyFor(be), ad.bBuf, hLora, 0)
    if bias and lin.hasBias and not fusedEpilogue:
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

  # Vulkan shells out to glslc once per shader; on a fresh container that
  # serial compile is the only prefill cost that does not shrink on a faster
  # GPU (measured: ~15 s of a 32 s first prefill here). Compile the whole set
  # concurrently before the hot loop instead of on each first dispatch.
  if be == gbVulkan:
    var jobs: seq[tuple[name, source: string]]
    proc addProg(name: string; prog: VmProgram) =
      if name.len > 0:
        jobs.add((name, jit_vm.lowerBytecodeToGlslCached(prog, name)))
    proc addNamed(name: string) =
      if name.len == 0: return
      let src = jit_vm.cachedGlslSource(gJitV2, name)
      if src.len > 0: jobs.add((name, src))
    addProg(historyKn, historyProg)
    for s, st in steps: addProg(stepNames[s], st.prog)
    for key, progs in lpPrograms:
      for s, st in progs: addProg(lpNames[key][s], st.prog)
    addProg(gateInKn, gateInput(hidden))
    addProg(gateMixKn, gateMix())
    for _, kn in m8Kns: addNamed(kn)
    for _, kn in m4Kns: addNamed(kn)
    for _, kn in m8x2Kns: addNamed(kn)
    for _, kn in m4x2Kns: addNamed(kn)
    for _, kn in biasKns: addNamed(kn)
    for _, kn in loraKns: addNamed(kn)
    let prewarmStart = getMonoTimeIfPf()
    byby.sessionJitPrewarm(be, jobs)
    pfAddIf("vulkan.prewarm", prewarmStart)

  var groupStart = getMonoTimeIfPf()
  for i, blk in m.blocks:
    let root = "block" & $i
    let layer = gLayersV2[i]
    if not sessionOpen:
      groupStart = getMonoTimeIfPf()
      require(sessionBegin(be), "block sessionBegin failed")
      sessionOpen = true
    pfTime("enc.rmsnorm"):
      require(sessionRmsnormEnc(be, hX, layer.hGamma, hNorm, T, hidden, layer.eps), "RMSNorm")
    flushStage("rmsnorm")
    pfTime("enc.inproj"): linear(root & ".mambaReal.inProj.weight", hNorm, hProjection, false)
    flushStage("inproj")
    pfTime("enc.copy"):
      copy(hProjection, hMamba, 0, L.proj, T * L.P)
      # convW|convB are adjacent in both layouts; aLog|dSkip|dtBias|normW are
      # adjacent in both too (the two blocks are ordered differently, so this
      # stays two copies instead of six, halving the Vulkan global barriers).
      copy(layer.hWork, hMamba, layer.off.offConvW, L.convW, L.cDim * L.K + L.cDim)
      copy(layer.hWork, hMamba, layer.off.offALog, L.aLog, 3 * L.H + L.C)
      if continuation:
        copy(layer.hWork,hMamba,layer.off.offSsm,L.states,L.HD*L.N)
        copy(layer.hWork,hMamba,layer.off.offConvSt,historyOff,L.cDim*(L.K-1))
      else: copy(hZero, hMamba, 0, L.states, L.HD * L.N)
    flushStage("copy")
    pfTime("enc.scan"):
      for s, st in steps:
        encodeMicroVm(ctx, st.prog, hMamba, st.nThreads, st.tgBytes, st.tag, st.tgSize, stepNames[s])
    flushStage("scan")
    copy(hMamba, hInner, L.normedOut, 0, T * L.C)
    pfTime("enc.outproj"): linear(root & ".mambaReal.outProj.weight", hInner, hMix, false)
    flushStage("outproj")
    encodeVmKernel(historyKn, L.cDim, 64, 4, hConv, 0, gDummyFor(be), layer.hWork, hProjection, 0)
    var hState = layer.hWork
    copy(hConv, hState, 0, layer.off.offConvSt, L.cDim * (L.K - 1))
    copy(hMamba, hState, terminalOff, layer.off.offSsm, L.HD * L.N)
    flushStage("outro")
    pfTime("enc.attn"):
      if blk.kind in {bkMambaRealPlusAttn, bkMambaRealGated}:
        let al = gAttnLayers[i]
        let lay = lpLayout(1, T, al.lay.H, al.lay.D)
        let key = $lay.H & "_" & $lay.D
        var hLp = lpArenas[key]
        linear(root & ".attn.qkvProj.weight", hNorm, hQkv, true)
        flushFine("lp.qkv")
        copy(hQkv, hLp, 0, lay.qkv, T * 3 * hidden)
        if continuation:
          copy(al.hWork,hLp,al.lay.offZ,lay.zF,hidden)
          copy(al.hWork,hLpState,al.lay.offS,0,hidden*lay.D)
          encodeVmKernel(transposeKns[key],hidden*lay.D,64,4,hLp,lay.cF,
            gDummyFor(be),gDummyFor(be),hLpState,0)
        else:
          copy(hZero, hLp, 0, lay.zF, hidden)
          copy(hZero, hLp, 0, lay.cF, hidden * lay.D)
        for s, st in lpPrograms[key]:
          encodeMicroVm(ctx, st.prog, hLp, st.nThreads, st.tgBytes, st.tag, st.tgSize, lpNames[key][s])
        flushFine("lp.core")
        copy(hLp, hAttnIn, lay.y, 0, T * hidden)
        linear(root & ".attn.outProj.weight", hAttnIn, hAttnOut, true)
        flushFine("lp.outproj")
        copy(hLp, hLpState, lay.cF, 0, hidden * lay.D)
        var hAttnState = al.hWork
        encodeVmKernel(transposeKns[key], hidden * lay.D, 64, 4,
          hAttnState, al.lay.offS, gDummyFor(be), gDummyFor(be), hLpState, 0)
        copy(hLp, hAttnState, lay.zF, al.lay.offZ, hidden)
        flushFine("lp.state")
        if blk.kind == bkMambaRealGated:
          encodeVmKernel(gateInKn, T * hidden, 64, 4, hGateIn, 0, gDummyFor(be), hMix, hAttnOut, 0)
          linear(root & ".fuseGate.weight", hGateIn, hGate, true)
          encodeVmKernel(gateMixKn, T * hidden, 64, 4, hMix, 0, gDummyFor(be), hGate, hAttnOut, 0)
        else: require(sessionVecOp(be, "add", hMix, hAttnOut, hMix, T * hidden), "branch sum")
        flushFine("lp.gate")
    flushStage("attn")
    require(sessionVecOp(be, "add", hX, hMix, hX, T * hidden), "residual")
    if (i+1) mod commitLayers == 0 or i == m.blocks.high:
      require(sessionEnd(be), "block sessionEnd failed: " & sessionGetLastError(be))
      sessionOpen = false
      pfAddIf("resident.group", groupStart)
  complete = true
  echo &"[infer] Optimized resident prefill OK: T={T}, blocks={m.blocks.len}; " &
    &"GPU scratch cache reused {gPrefixHits - hitsBefore}/{nextBuffer} buffers; terminal states copied GPU-to-GPU."
  echo &"[infer] prefill scan={scanMode}; MPS tiles={gPrefixMps}, tensor-core GEMMs={gPrefixTensorCore}, " &
    &"packed tiles={gPrefixPackedTiles}, fused epilogues={gPrefixFusedEpilogues}; " &
    &"f32 matmul MPS={f32Mps} fallback={f32Fallback} batched={f32Batch}."
