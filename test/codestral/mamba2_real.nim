## mamba2_real.nim
## A Mamba2 with the CORRECT original formula (unlike the "simplified linear attention
## with decay" Mamba2SSD in test_nimformer.nim). Matches the HF `Mamba2Mixer` architecture
## used by mistralai/Mamba-Codestral-7B-v0.1, so the pretrained weights map 1-1.
##
## Math (per timestep t, head h, channel d within the head, state n):
##   proj            = inProj(x)
##   z, xBC, dt_in   = split(proj)
##   xBC_conv        = SiLU(CausalConv1dDepthwise(xBC))
##   x, B, C         = split(xBC_conv)          # B, C shared per GROUP
##   A_h             = -exp(A_log[h])
##   dt_t,h          = softplus(dt_in_t,h + dt_bias[h])
##   dA_t,h          = exp(dt_t,h * A_h)
##   s_t             = s_{t-1} * dA_t,h + dt_t,h * B_t,g(h) ⊗ x_t,h,d
##   y_t,h,d         = Σ_n C_t,g(h),n * s_t,h,d,n + D[h] * x_t,h,d
##   gated           = y * SiLU(z)
##   normed          = RMSNorm(gated) * normWeight
##   out             = outProj(normed)
## g(h) = h div (numHeads div nGroups)  (B/C shared per group, GQA style).
##
## PERFORMANCE: the backward is a SEQUENTIAL BPTT over every timestep (correct math, not
## the parallel chunked-SSD of the original paper) - the full state is cached, so
## it takes O(T*H*headDim*dState) RAM. Fine for verification + finetuning short sequences/
## small batches; long sequences on a real 7B need a chunked version later.

import std/[math, tables, os, strutils, strformat]
import backend
import nimformer   # Tensor, Linear, newLinear, randnTensor, newTensor
import fp4_gemv_gpu  # fp4GemvGpu - used when fp4InName/fp4OutName != "" (called from lora.nim)
import mamba2_ssm_gpu  # mamba2StepGpu - used when useGpuSsm=true
import mamba2_full_vm  # whole-layer forward/backward with micro-kernels

type
  Mamba2Real* = object
    hiddenSize*, intermediateSize*, numHeads*, headDim*, nGroups*, dState*, convKernel*: int
    inProj*: Linear
    convWeight*: Tensor    # [convDim, convKernel]
    convBias*: Tensor      # [convDim]
    aLog*: Tensor          # [numHeads]
    dSkip*: Tensor         # [numHeads]  (= "D" in HF)
    dtBias*: Tensor        # [numHeads]
    normWeight*: Tensor    # [intermediateSize]
    outProj*: Linear
    normEps*: float32

proc convDim*(m: Mamba2Real): int = m.intermediateSize + 2 * m.nGroups * m.dState

proc newMamba2Real*(hiddenSize, intermediateSize, numHeads, dState, nGroups: int,
                     convKernel: int = 4, normEps: float32 = 1e-5'f32;
                     initializeWeights = true): Mamba2Real =
  assert intermediateSize mod numHeads == 0, "intermediateSize must be divisible by numHeads"
  assert numHeads mod nGroups == 0, "numHeads must be divisible by nGroups"
  result.hiddenSize = hiddenSize
  result.intermediateSize = intermediateSize
  result.numHeads = numHeads
  result.headDim = intermediateSize div numHeads
  result.nGroups = nGroups
  result.dState = dState
  result.convKernel = convKernel
  result.normEps = normEps
  let cDim = intermediateSize + 2 * nGroups * dState
  result.inProj = newLinear(hiddenSize, 2 * intermediateSize + 2 * nGroups * dState + numHeads,
    initializeWeights)
  result.convWeight = randnTensor(@[cDim, convKernel], 1'f32 / sqrt(float32(convKernel)))
  result.convBias = newTensor(@[cDim])
  result.aLog = newTensor(@[numHeads])
  for h in 0 ..< numHeads:
    result.aLog.data[h] = ln(float32(h mod 16 + 1))
  result.dSkip = newTensor(@[numHeads], 1'f32)
  result.dtBias = newTensor(@[numHeads])
  result.normWeight = newTensor(@[intermediateSize], 1'f32)
  result.outProj = newLinear(intermediateSize, hiddenSize, initializeWeights)

proc inProjSsmRowRange*(intermediateSize, nGroups, dState, numHeads: int): (int, int) =
  ## Rows [lo, hi) of inProj.weight belonging to the SSM branch (xBC + dt), without the z gate.
  ## Output layout of inProj: [z(C) | xBC(C + 2*G*N) | dt(H)].
  ## Used when LoRA should only touch what feeds the SSM (avoids poisoning dt/B/C through z).
  let C = intermediateSize
  let cDim = C + 2 * nGroups * dState
  (C, C + cDim + numHeads)

proc inProjSsmRowRange*(m: Mamba2Real): (int, int) =
  inProjSsmRowRange(m.intermediateSize, m.nGroups, m.dState, m.numHeads)

type
  Mamba2RealCache* = object
    ## Every forward intermediate lives in ONE GPU buffer (layout M2Lay) — the backward reuses
    ## exactly that buffer, nothing is rebuilt on the CPU.
    B*, T*: int
    buf*: seq[float32]
    lay*: M2Lay
    proj*: Tensor

proc sliceTensor(buf: seq[float32]; off, n: int; shape: seq[int]): Tensor =
  ## Copies a region of the GPU buffer into a Tensor (copyMem, no computation)
  result = Tensor(shape: shape, data: newSeq[float32](n))
  result.data.blit(0, buf, off, n)

proc forwardMamba2Real*(m: Mamba2Real, x: Tensor, ctx: Backend):
                        tuple[y: Tensor, cache: Mamba2RealCache] =
  ## inProj / outProj: GEMMs on the GPU (Linear.forward). The rest (conv1d+SiLU + softplus dt,
  ## selective scan over every timestep, gated RMSNorm) = 3 micro-kernels run back to back in one session
  ## on the same buffer; the scan uses a parallel affine prefix over T if T<=64 and the
  ## threadgroup is valid, otherwise a sequential recurrence. The backward still uses a sequential reverse scan.
  ensureTgCap(ctx)   # the reduction tgSize must be <= the GPU limit (Iris 640: 256)
  let B = x.shape[0]; let T = x.shape[1]
  assert x.shape[2] == m.hiddenSize
  let H = m.numHeads; let D = m.headDim; let N = m.dState; let G = m.nGroups
  let cDim = m.convDim
  let K = m.convKernel
  let C = m.intermediateSize

  let L = m2Layout(B, T, H, D, N, G, K, C, m.normEps, G)
  checkVmScratchCapacity(ctx, L.total,
    &"Mamba2Real (B={B}, T={T}, H={H}, D={D}, N={N})")
  let proj = m.inProj.forward(x, ctx)
  var buf = newSeq[float32](L.total)
  buf.blit(L.proj, proj.data, 0, L.BT * L.P)
  buf.blit(L.convW, m.convWeight.data, 0, cDim * K)
  buf.blit(L.convB, m.convBias.data, 0, cDim)
  buf.blit(L.aLog, m.aLog.data, 0, H)
  buf.blit(L.dSkip, m.dSkip.data, 0, H)
  buf.blit(L.dtBias, m.dtBias.data, 0, H)
  buf.blit(L.normW, m.normWeight.data, 0, C)
  let tgR = rmsTg(L)
  discard runMicroVmChain(ctx, [
    (buildM2ConvDtProgram(L), L.BT * M2Tg, 4, M2Tg, "@m2_conv_dt_" & m2Tag(L)),
    mamba2ScanStep(L, ctx),
    (buildM2RmsFwdProgram(L), L.BT * L.normGroups * tgR, tgR * 4, tgR, "@m2_rms_" & m2Tag(L))], buf)
  let normedOut = sliceTensor(buf, L.normedOut, L.BT * C, @[B, T, C])
  let y = m.outProj.forward(normedOut, ctx)
  result = (y, Mamba2RealCache(B: B, T: T, buf: buf, lay: L, proj: proj))

# ─────────────────────────────────────────────────────────────
# Backward - 4 micro-kernel (rms -> scan -> dt -> conv), 1 session.
# ─────────────────────────────────────────────────────────────

proc backwardMamba2Real*(m: Mamba2Real, x: Tensor, cache: Mamba2RealCache,
                          dOut: Tensor, ctx: Backend, prefix: string,
                          grads: var Table[string, Tensor]): Tensor =
  ensureTgCap(ctx)
  let L = cache.lay
  let B = cache.B; let T = cache.T
  let H = m.numHeads; let C = m.intermediateSize
  let cDim = m.convDim
  let K = m.convKernel

  # ---- outProj backward (GEMM GPU) ----
  let normedOut = sliceTensor(cache.buf, L.normedOut, L.BT * C, @[B, T, C])
  let (dNormedOutFull, dOutProjW, dOutProjB) = m.outProj.backward(normedOut, dOut, ctx)
  grads[prefix & ".outProj.weight"] = dOutProjW
  grads[prefix & ".outProj.bias"] = dOutProjB

  # ---- RMSNormGated + selective scan + softplus-dt + conv1d/SiLU backward: 4 kernel GPU ----
  var buf = cache.buf                       # the d* regions of cache.buf are still 0 -> safe for the atomics
  buf.blit(L.dNormedOut, dNormedOutFull.data, 0, L.BT * C)
  let tgR = rmsTg(L)
  discard runMicroVmChain(ctx, [
    (buildM2RmsBwdProgram(L), L.BT * L.normGroups * tgR, tgR * 4, tgR, "@m2_rms_bwd_" & m2Tag(L))], buf)
  # scan backward: n is split into chunks, EACH chunk its own command buffer (the Intel GPU watchdog
  # kills command buffers > a few seconds). The atomics accumulate, so the sum of the chunks == one run.
  var nChunk = 16
  let envC = getEnv("BYBY_SCAN_NCHUNK")
  if envC.len > 0:
    try: nChunk = max(1, parseInt(envC))
    except ValueError: discard
  nChunk = min(nChunk, m.dState)
  let per = (m.dState + nChunk - 1) div nChunk
  var nb = 0
  while nb < m.dState:
    let cnt = min(per, m.dState - nb)
    discard runMicroVmChain(ctx, [
      (buildM2ScanBwdProgram(L, nb, cnt), B * L.HD, 4, 0,
       "@m2_scan_bwd_" & m2Tag(L) & "_n" & $nb & "_" & $cnt)], buf)
    nb += cnt
  discard runMicroVmChain(ctx, [
    (buildM2DtBwdProgram(L), L.BT * M2Tg, 4, M2Tg, "@m2_dt_bwd_" & m2Tag(L)),
    (buildM2ConvBwdProgram(L), L.BT * M2Tg, 4, M2Tg, "@m2_conv_bwd_" & m2Tag(L))], buf)

  grads[prefix & ".normWeight"] = sliceTensor(buf, L.dNormW, C, @[C])
  grads[prefix & ".dSkip"] = sliceTensor(buf, L.dDSkip, H, @[H])
  grads[prefix & ".dtBias"] = sliceTensor(buf, L.dDtBias, H, @[H])
  grads[prefix & ".aLog"] = sliceTensor(buf, L.dALog, H, @[H])
  grads[prefix & ".convWeight"] = sliceTensor(buf, L.dConvW, cDim * K, @[cDim, K])
  grads[prefix & ".convBias"] = sliceTensor(buf, L.dConvB, cDim, @[cDim])

  # ---- inProj backward (GPU GEMM); dProj is already assembled by the kernel [dZ | dXBC | dDtPre] ----
  let dProj = sliceTensor(buf, L.dProj, L.BT * L.P, cache.proj.shape)
  let (dX, dInProjW, dInProjB) = m.inProj.backward(x, dProj, ctx)
  grads[prefix & ".inProj.weight"] = dInProjW
  grads[prefix & ".inProj.bias"] = dInProjB
  result = dX

# ─────────────────────────────────────────────────────────────
# Param registry (same convention as paramList in jamba_hybrid.nim)
# ─────────────────────────────────────────────────────────────

proc paramListMamba2Real*(m: var Mamba2Real, prefix: string): seq[(string, ptr Tensor)] =
  result = @[]
  result.add((prefix & ".inProj.weight", addr m.inProj.weight))
  result.add((prefix & ".inProj.bias", addr m.inProj.bias))
  result.add((prefix & ".convWeight", addr m.convWeight))
  result.add((prefix & ".convBias", addr m.convBias))
  result.add((prefix & ".aLog", addr m.aLog))
  result.add((prefix & ".dSkip", addr m.dSkip))
  result.add((prefix & ".dtBias", addr m.dtBias))
  result.add((prefix & ".normWeight", addr m.normWeight))
  result.add((prefix & ".outProj.weight", addr m.outProj.weight))
  result.add((prefix & ".outProj.bias", addr m.outProj.bias))

# ─────────────────────────────────────────────────────────────
# Maps the HF tensor names (mistralai/Mamba-Codestral-7B-v0.1, the mixer of one layer)
# -> the Mamba2Real parameter names. Used with loadAllHFTensors (pytorch_bin_loader)
# or loadAllHFTensorsSharded (sharded_hf_loader) to load the REAL pretrained weights
# DIRECTLY into this struct - UNLIKE the old Mamba2SSD, this maps 1-1
# MEANINGFULLY because it is the same math.
# NOTE: check the exact names in the manifest (pytorch_bin_loader --export
# or listStateDictInfo) because HF may rename them between transformers versions
# (e.g. "mixer.dt_bias" vs "mixer.dt_proj.bias", "mixer.D" vs "mixer.dSkip").
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# INCREMENTAL STATE (NEW) - for 1-new-token-per-step INFERENCE, O(1)/token instead of
# O(T) rerun from scratch (forwardMamba2Real above). Formulas checked EXACTLY
# against the real HF source (transformers/models/mamba2/modeling_mamba2.py, the
# `mamba2_selective_state_update` function + the conv_state update in
# `cuda_kernels_forward`/`torch_forward`, the "cached_start" branch), NOT invented:
#   dt[h]      = softplus(dtPre[h] + dtBias[h])
#   dA[h]      = exp(dt[h] * A_h),  A_h = -exp(aLog[h])
#   state[h,d,n] = state[h,d,n]*dA[h] + dt[h]*B[g,n]*x[h,d]
#   y[h,d]     = sum_n( C[g,n] * state[h,d,n] ) + D[h]*x[h,d]
# conv1d: convState keeps the K-1 MOST RECENT xBCPre steps (rolling window), each
# conv step = sum_k convWeight[c,k]*window[k] (window = convState + the new xBCPre,
# K elements), then convState slides (drops the oldest element, adds the new xBCPre).
# ONLY for B=1, T=1 (one sequence, token by token) - exactly what generate() needs.
# ─────────────────────────────────────────────────────────────

type
  Mamba2RealState* = object
    ## convState: [convDim, convKernel-1] - the K-1 MOST RECENT xBCPre steps (before
    ## SiLU, before splitting x/B/C) - EXACTLY what HF stores in
    ## `cache_params.conv_states[layer_idx]`.
    ## ssmState: [numHeads, headDim, dState] - EXACTLY what HF stores in
    ## `cache_params.ssm_states[layer_idx]` (called recurrent_state).
    convState*: seq[float32]
    ssmState*: seq[float32]

proc newMamba2RealState*(m: Mamba2Real): Mamba2RealState =
  ## Zero-init - exactly "s_{-1}=0" / "conv_state=0" at the start of a new
  ## sequence (no token before it).
  result.convState = newSeq[float32](m.convDim * (m.convKernel - 1))
  result.ssmState = newSeq[float32](m.numHeads * m.headDim * m.dState)

proc forwardStepMamba2Real*(m: Mamba2Real, x: Tensor, state: var Mamba2RealState,
                             ctx: Backend, fp4InName: string = "", fp4OutName: string = "",
                             useGpuSsm: bool = true): Tensor =
  ## x: [1,1,hiddenSize] - EXACTLY one new token. Updates the state IN PLACE (var),
  ## returns y: [1,1,hiddenSize]. NO backward cache (inference only).
  ## fp4InName/fp4OutName: see the old note.
  ## useGpuSsm: true (default) -> SSM + softplus + (RMS when C<=1024) through the
  ## microVM (mamba2StepGpu / buildRmsGateProgram / buildConv1dIncProgram).
  ## The interpreter has every opcode on Metal/CUDA/Vulkan. Conv1d + large RMS
  ## may still fall back to the host if a kernel is not fully wired; the call site must
  ## sessionBegin first to merge the session.
  assert x.shape == @[1, 1, m.hiddenSize], "forwardStepMamba2Real: accepts only B=1,T=1"
  let H = m.numHeads; let D = m.headDim; let N = m.dState; let G = m.nGroups
  let hpg = H div G
  let cDim = m.convDim
  let K = m.convKernel
  let C = m.intermediateSize

  var proj: Tensor
  if fp4InName.len > 0:
    let projData = fp4GemvGpu(ctx, fp4InName, x.data, 1)   # 100% GPU, nothing materialised
    proj = newTensor(@[1, 1, 2*C + 2*G*N + H])
    proj.data = projData
    # LoRA inProj: the delta MUST be added INTO proj BEFORE conv/SSM (W'=W+BA), it cannot
    # be added after the SSM. Linear.forward() does this when FP4 is not used.
    if m.inProj.loraR > 0 and m.inProj.loraA.len == m.inProj.loraR * m.inProj.inF and
       m.inProj.loraB.len == m.inProj.outF * m.inProj.loraR:
      let r = m.inProj.loraR
      let At = transpose(Tensor(shape: @[r, m.inProj.inF], data: m.inProj.loraA))
      let Bt = transpose(Tensor(shape: @[m.inProj.outF, r], data: m.inProj.loraB))
      let xA = beMatmul(ctx, x.data, 1, m.inProj.inF, At.data, m.inProj.inF, r)
      let delta = beMatmul(ctx, xA, 1, r, Bt.data, r, m.inProj.outF)
      axpyGpu(ctx, proj.data, delta, m.inProj.loraScale)   # GPU: proj += scale*delta
  else:
    proj = m.inProj.forward(x, ctx)   # [1,1, 2C+2GN+H] - CPU/resident path (with LoRA if attached)
  # ---- 100% GPU path (microVM) — useGpuSsm defaults to true, no CPU ----
  var z: seq[float32]
  var xBCPre: seq[float32]
  var dtPre: seq[float32]
  var xBCConv: seq[float32]
  var xPart: seq[float32]
  var bMat: seq[float32]
  var cMat: seq[float32]
  var dt: seq[float32]
  var yPre: seq[float32]
  var normedOut: Tensor

  if not useGpuSsm:
    raise newException(CatchableError,
      "forwardStepMamba2Real: useGpuSsm=false is FORBIDDEN — the framework requires 100% GPU microVM")
  if true:
    # 0) split proj on the GPU
    let spl = splitProjGpu(ctx, C, cDim, H, proj.data)
    z = spl.z
    xBCPre = spl.xBC
    dtPre = spl.dt

    # 1) conv1d
    xBCConv = conv1dIncGpu(ctx, cDim, K, xBCPre, state.convState,
                           m.convWeight.data, m.convBias.data)
    # the xBCConv layout is already [xPart | B | C] — a zero-copy slice view through openArray
    # mamba2StepGpuFromXbc packs on the GPU buffer (no host for loop)
    dt = softplusDtGpu(ctx, H, dtPre, m.dtBias.data)
    yPre = mamba2StepGpuFromXbc(ctx, H, D, N, G, C, xBCConv, dt,
                                m.aLog.data, m.dSkip.data, state.ssmState)

    # 4) RMS+gate 100% on the GPU (including C=8192)
    let outData = rmsGateGpu(ctx, C, m.normEps, yPre, z, m.normWeight.data, G)
    normedOut = newTensor(@[1, 1, C])
    normedOut.data = outData
  if fp4OutName.len > 0:
    let outData = fp4GemvGpu(ctx, fp4OutName, normedOut.data, 1)
    result = newTensor(@[1, 1, m.hiddenSize])
    result.data = outData
    # LoRA outProj: activation = normedOut [1,1,C] (C=intermediate, e.g. 8192),
    # NOT x/ln1 (hidden 4096).
    if m.outProj.loraR > 0 and m.outProj.loraA.len == m.outProj.loraR * m.outProj.inF and
       m.outProj.loraB.len == m.outProj.outF * m.outProj.loraR:
      let r = m.outProj.loraR
      let At = transpose(Tensor(shape: @[r, m.outProj.inF], data: m.outProj.loraA))
      let Bt = transpose(Tensor(shape: @[m.outProj.outF, r], data: m.outProj.loraB))
      let xA = beMatmul(ctx, normedOut.data, 1, m.outProj.inF, At.data, m.outProj.inF, r)
      let delta = beMatmul(ctx, xA, 1, r, Bt.data, r, m.outProj.outF)
      axpyGpu(ctx, result.data, delta, m.outProj.loraScale)   # GPU: out += scale*delta
  else:
    result = m.outProj.forward(normedOut, ctx)

proc hfNameMapMamba2*(layerIdx: int): Table[string, string] =
  result = initTable[string, string]()
  let p = "backbone.layers." & $layerIdx & ".mixer."
  result["inProj.weight"] = p & "in_proj.weight"
  result["inProj.bias"]   = p & "in_proj.bias"      # may NOT exist (the original Mamba2 usually has bias=False)
  result["convWeight"]    = p & "conv1d.weight"
  result["convBias"]      = p & "conv1d.bias"
  result["aLog"]          = p & "A_log"
  result["dSkip"]         = p & "D"
  result["dtBias"]        = p & "dt_bias"
  result["normWeight"]    = p & "norm.weight"
  result["outProj.weight"] = p & "out_proj.weight"
  result["outProj.bias"]   = p & "out_proj.bias"    # usually does NOT exist
  # FIX for garbage tokens (bug #7): the pre-norm OF THE WHOLE BLOCK (ln1 in Nim) corresponds to
  # Mamba2Block.norm in HF, OUTSIDE the .mixer submodule, so `p` is NOT used
  # (`p` already contains ".mixer."). This key used to be MISSING entirely -> ln1.gamma
  # was never loaded from the checkpoint and stayed gamma=1.0 (the default identity
  # of newLayerNorm) for EVERY block on EVERY run -> wrong scaling before
  # the mixer in all 64 blocks -> garbage tokens.
  result["ln1Weight"] = "backbone.layers." & $layerIdx & ".norm.weight"
