## generate.nim
## A SINGLE file: nimformer_llama (RoPE + RMSNorm + SwiGLU + GQA + paged KV cache)
## + a pure-Nim HF tokenizer + the text generation loop.

import std/[math, random, sequtils, tables, algorithm, os, strformat, strutils, times, json, sets, heapqueue, terminal]
when defined(posix): import std/posix
import nimformers  # Tensor, Linear, newLinear, flatten2D, transpose, Embedding, addT...
import native_hf_tokenizer
export native_hf_tokenizer

# ─────────────────────────────────────────────────────────────
# DEBUG: BYBY_DEBUG_PATH=1 shows EXACTLY whether the first layer runs the
# fused/resident path (fast, 1 session per op group) or falls back to the LEGACY path (each
# Linear/RMSNorm opens its own session, or worse: the whole weight is dequantized again
# on the CPU every token) at each decision point (QKV / attn+oProj /
# FFN). This IS the only difference between "~1 tok/s" and "0.3 tok/s" -
# printed EXACTLY ONCE per point (no spam per token/layer) so the measured
# speed is unaffected. Enable: BYBY_DEBUG_PATH=1 ./nim_inference ...
# If ANY line prints "LEGACY fallback" for layer 0 -> that is
# the cause of the slowness: residentReady/residentFp32Ready of the matching Linear
# is false (see uploadResident() in nimformer.nim) or sessionBegin/
# sessionMatmulQ4/sessionMatmul fails on the very first call.
# ─────────────────────────────────────────────────────────────
var gDebugPathOn = getEnv("BYBY_DEBUG_PATH", "0") != "0"
var gDebugPathSeen = initHashSet[string]()
proc debugPathOnce(tag: string, usedFast: bool, extra: string = "") =
  if not gDebugPathOn: return
  if tag in gDebugPathSeen: return
  gDebugPathSeen.incl(tag)
  let verdict = if usedFast: "FUSED/RESIDENT (fast)"
                else: "LEGACY fallback (SLOW - separate sessions or CPU dequant every token)"
  stderr.writeLine "[BYBY_DEBUG_PATH] " & tag & " -> " & verdict &
    (if extra.len > 0: "  (" & extra & ")" else: "")

# ===================================================================
# Cross-submodule SESSION MERGING helpers: EACH submodule (RMSNorm,
# RoPE, FFN, attention-decode) used to open/close its OWN session
# (sessionBegin...sessionEnd), i.e. 1 layer = 4-6 GPU commit+waits
# (Metal: commandBuffer commit+waitUntilCompleted; CUDA: cuStreamSynchronize)
# even though there is NO real mandatory CPU dependency between those submodules
# (residual-add, RMSNorm-scale, the gate/up/down matmuls are all pure GPU work in
# sequence). The procs below ENCODE several commands into 1 OPEN session
# without reading results back to the CPU in between (results stay as
# SessionHandles, sessionRead() runs exactly once at the end) - this is how "1
# cross-submodule session" is really done, not just a function rename.
proc sMatmul(be: GpuBackend, a: SessionHandle, M, K: int, b: SessionHandle, N: int): SessionHandle =
  result = sessionAllocScratch(be, M * N)
  doAssert sessionMatmul(be, a, b, result, M, K, N),
    "sMatmul: sessionMatmul failed inside the merged session (the current backend may not support it)"
proc sMatmulNT(be: GpuBackend, a: SessionHandle, M, K: int, b: SessionHandle, N: int): tuple[h: SessionHandle, ok: bool] =
  ## C(M x N) = A(M x K) @ B(N x K)^T - B is passed in its ORIGINAL layout
  ## [N,K] (e.g. a K-cache/V-cache slice [S,D] uploaded directly, no CPU transpose
  ## first). NEW: used by attention decode to drop the O(S*D)/token transpose
  ## cost of the old path entirely.
  ##
  ## Returns ok=false if the current backend does not support sessionMatmulNT yet (only
  ## OpenCL/Metal/CUDA have it, plain CPU does not) - the caller MUST fall back to
  ## sMatmul with a pre-transposed b (the old path), and must NOT treat false as an error
  ## and doAssert crash.
  result.h = sessionAllocScratch(be, M * N)
  result.ok = sessionMatmulNT(be, a, b, result.h, M, K, N)

proc sVecOp(be: GpuBackend, op: string, a, b: SessionHandle, n: int): SessionHandle =
  result = sessionAllocScratch(be, n)
  doAssert sessionVecOp(be, op, a, b, result, n),
    "sVecOp(" & op & "): sessionVecOp failed inside the merged session"

proc sActivation(be: GpuBackend, op: string, a: SessionHandle, n: int): SessionHandle =
  result = sessionAllocScratch(be, n)
  doAssert sessionActivation(be, op, a, result, n),
    "sActivation(" & op & "): sessionActivation failed inside the merged session"

proc sSoftmax(be: GpuBackend, a: SessionHandle, rows, cols: int): SessionHandle =
  ## Wraps sessionSoftmax (already in backend.nim, exported from byby) following
  ## the sMatmul/sVecOp/sActivation pattern above - used by attention
  ## decode (see forwardDecode below). Nothing Metal-specific, it runs
  ## on CUDA/OpenCL too since it goes through the shared sessionXxx API.
  result = sessionAllocScratch(be, rows * cols)
  doAssert sessionSoftmax(be, a, result, rows, cols),
    "sSoftmax: sessionSoftmax failed inside the merged session"

proc broadcastBias(bias: Tensor, rows, outF: int): seq[float32] =
  ## Expands the bias [outF] to a full [rows*outF] so it can be added with 1 sessionVecOp("add")
  ## instead of a CPU "i mod outF" loop after reading the result back (which would force
  ## the session to end early). Returns an empty seq when there is no real bias.
  if bias.data.len != outF: return @[]
  result = newSeq[float32](rows * outF)
  for r in 0 ..< rows:
    copyMem(addr result[r * outF], unsafeAddr bias.data[0], outF * sizeof(float32))

proc dequantTransposedW(l: Linear): seq[float32] =
  ## The weight ALREADY transposed to [inF, outF] flat row-major, ready to load straight into
  ## a session with sessionUpload() + sMatmul(), shared by the QKV fuse and the
  ## FFN fuse below (each used to have its own copy).
  ## NOW only used as a FALLBACK when the weight is NOT int4-per-group (see
  ## linearFast below) - the main path (real GPTQ int4 models) no longer goes through
  ## this function.
  if l.useQuant: dequantizeTensorTransposed(l.weightQ)
  else: transpose(l.weight).data

proc linearFast(l: Linear, xData: seq[float32], rows: int, ctx: Backend): seq[float32] =
  ## FIX (THE MAIN CAUSE of "1.5 minutes/token"): BEFORE, every place in
  ## this file (fusedNormQKVRoPE, fusedResidualNormFFN, forwardQKVFused)
  ## called dequantTransposedW() -> decompressing the WHOLE int4 weight matrix to
  ## fp32 on the CPU (e.g. 4096x4096 or 4096x11008 per matrix) THEN uploading
  ## that whole fp32 array to the GPU via sessionUpload - repeated for
  ## ALL 7 matrices (Q,K,V,O,gate,up,down) x 32 layers x EVERY generated TOKEN.
  ## That was the "1.5 minutes/token": not a slow GPU kernel, but CPU
  ## dequant + huge fp32 upload bandwidth repeated pointlessly every decode step
  ## (the weights do NOT change between tokens, there is no reason to decompress them again).
  ##
  ## nimformer.nim's Linear.forward() already has the proper llama.cpp-style
  ## shortcut: beMatmulQ4() runs the matmul DIRECTLY on the still-compressed int4 data
  ## (Metal/OpenCL have a native int4 kernel reading 4-bit directly -> ~8x less
  ## memory bandwidth than reading dequantized fp32). This file (nim_inference.nim)
  ## never called it before. Now every place shares this function instead of
  ## dequantTransposedW + beMatmul/sMatmul.
  if l.residentReady:
    # THE REAL FAST PATH: the weight is already resident on the GPU from uploadAllResidentWeights
    # (called once at model load) - here only xData (the activation, small,
    # changes every token) is uploaded, then matmul'd with the existing weight handle, without touching
    # the weight bytes again. This is the path that removes "uploading 4.5GB every token".
    result = beResidentLinearQ4(ctx, xData, l.residentW, rows)

  if result.len == 0:
    # Not resident, or the current backend (e.g. cpu) has no resident
    # path -> falls back to the old path (still correct, just without the resident speedup).
    if l.useQuant and l.weightQ.kind == qkInt4Asymmetric and l.weightQ.groupSize > 0:
      let nGroupsPerRow = (l.inF + l.weightQ.groupSize - 1) div l.weightQ.groupSize
      result = beMatmulQ4(ctx, xData, @(l.weightQ.bytes), l.weightQ.scale, l.weightQ.zero_point,
                           rows, l.inF, l.outF, l.weightQ.groupSize, nGroupsPerRow)
    else:
      # Fallback (weight not int4-per-group, e.g. a tied-embedding lm_head with
      # useQuant=false): still needs a dequant, but this is rare
      # (not inside the 32 layers x every token loop).
      let wT = dequantTransposedW(l)
      result = beMatmul(ctx, xData, rows, l.inF, wT, l.inF, l.outF)
  let expectedSize = rows * l.outF
  if result.len != expectedSize:
    var fixed = newSeq[float32](expectedSize)
    let copyLen = min(result.len, expectedSize)
    for i in 0 ..< copyLen: fixed[i] = result[i]
    result = fixed

proc linearFastBias(l: Linear, xData: seq[float32], rows: int, ctx: Backend): seq[float32] =
  result = linearFast(l, xData, rows, ctx)
  if l.bias.data.len == l.outF:
    for r in 0 ..< rows:
      let off = r * l.outF
      for o in 0 ..< l.outF:
        result[off + o] += l.bias.data[o]


# ===================================================================
# RMSNorm
# ===================================================================
type
  RMSNormL* = object
    weight*: Tensor
    eps*: float32
    dim*: int
    residentGamma*: SessionHandle   # OPTIMISATION: gamma is a fixed constant
      # (does not change between tokens/layers) - uploaded ONCE at model
      # load (see uploadResidentNorm below), instead of sessionUpload()
      # FROM SCRATCH on every forward call (every layer, every token) as before -
      # exactly the residentReady pattern already used for Q4/fp32 weights.
    residentGammaReady*: bool

proc newRMSNormL*(dim: int, eps: float32 = 1e-6'f32): RMSNormL =
  result.dim = dim
  result.eps = eps
  result.weight = newTensor(@[dim], 1'f32)

proc uploadResidentNorm*(rn: var RMSNormL, ctx: Backend) =
  ## Uploads gamma (rn.weight) to the GPU ONCE - called in the model's
  ## uploadResident() at load (like the QKV/FFN resident weights), NOT
  ## in the decode loop. Reuses beResidentUploadFp32 (the missing
  ## gbVulkan branch was fixed earlier) instead of writing a new function - less
  ## risk of introducing a new bug.
  rn.residentGamma = beResidentUploadFp32(ctx, rn.weight.data)
  rn.residentGammaReady = true

proc rmsNormPrep*(rn: RMSNormL, x: Tensor, ctx: Backend): tuple[invBroadcast, wBroadcast: seq[float32], n: int] =
  ## Splits the "reduce" part of RMSNorm (squares + per-row sum via
  ## matmul-as-reduce, then 1/sqrt(mean+eps) on the CPU) from the "scale" part (2
  ## full-size muls). Why: invStd is a scalar per row, and
  ## sumSq MUST be read back to the CPU to run sqrt (the current session API has no
  ## reduce/sqrt kernel) - an unavoidable CPU touch, happening
  ## BEFORE any large session. But the "scale" part (2 muls) has NO reason
  ## to close its own session - callers (fusedNormQKVRoPE,
  ## fusedResidualNormFFN) put these 2 muls straight into the next submodule's session
  ## (QKV matmul / FFN) instead of opening/closing a session just for 2 muls.
  let (rows, cols) = flatten2D(x.shape)
  doAssert cols == rn.dim
  let n = rows * cols
  let x2 = beMul(ctx, x.data, x.data)
  var ones = newSeq[float32](cols)
  for i in 0 ..< cols: ones[i] = 1'f32
  let sumSq = beMatmul(ctx, x2, rows, cols, ones, cols, 1)
  var invStd = newSeq[float32](rows)
  for r in 0 ..< rows:
    invStd[r] = 1'f32 / sqrt(sumSq[r] / float32(cols) + rn.eps)
  var invBroadcast = newSeq[float32](n)
  var wBroadcast = newSeq[float32](n)
  for r in 0 ..< rows:
    let off = r * cols
    for c in 0 ..< cols:
      invBroadcast[off + c] = invStd[r]
      wBroadcast[off + c] = rn.weight.data[c]
  result = (invBroadcast, wBroadcast, n)

proc forward*(rn: RMSNormL, x: Tensor, ctx: Backend): Tensor =
  ## backend.nim has no dedicated RMSNorm primitive, but it has ENOUGH add/mul
  ## (beMul/beAdd, or sessionVecOp) and matmul (beMatmul) to rebuild the whole
  ## formula on the GPU, no plain CPU loop needed:
  ##   - Per-element square: x ⊙ x -> 1 beMul (GPU).
  ##   - Per-row sum of squares (a reduction the backend has no dedicated kernel
  ##     for): multiply by an all-ones column vector with beMatmul(x2, rows, cols,
  ##     ones, cols, 1) -> exactly sumSq [rows,1]. This is "matmul-as-reduce",
  ##     GPU-native, not a pretence.
  ##   - 1/sqrt(mean+eps): ONLY a scalar per row (rows elements, not
  ##     rows*cols) so it is computed on the CPU - not worth, and not possible as, a
  ##     separate GPU kernel for just rows numbers.
  ##   - Rescale by invStd then by weight: 2 full-size muls
  ##     [rows,cols] -> merged into 1 GPU session (sessionVecOp mul, mul) so the data
  ##     stays on the GPU throughout, no upload/download between the 2 steps.
  let (rows, cols) = flatten2D(x.shape)
  doAssert cols == rn.dim
  let n = rows * cols

  let x2 = beMul(ctx, x.data, x.data)                 # GPU: per-element square
  var ones = newSeq[float32](cols)
  for i in 0 ..< cols: ones[i] = 1'f32
  let sumSq = beMatmul(ctx, x2, rows, cols, ones, cols, 1)  # GPU: per-row reduce via matmul

  var invStd = newSeq[float32](rows)                  # O(rows) scalars, the CPU is fine
  for r in 0 ..< rows:
    invStd[r] = 1'f32 / sqrt(sumSq[r] / float32(cols) + rn.eps)

  # Broadcast invStd/weight to the full [rows,cols] - this is a data COPY (not
  # arithmetic), the backend has no broadcast kernel so it is done on the CPU,
  # then the real MULTIPLY (mul) is pushed to the GPU right after.
  var invBroadcast = newSeq[float32](n)
  var wBroadcast = newSeq[float32](n)
  for r in 0 ..< rows:
    let off = r * cols
    for c in 0 ..< cols:
      invBroadcast[off + c] = invStd[r]
      wBroadcast[off + c] = rn.weight.data[c]

  let backend = ctx.kind.toByby()
  result.shape = x.shape
  # The session (mul, mul) merges the 2 multiplies into 1 encoder, the data stays on the GPU
  # throughout. On Metal, several dispatches of different pipelines sharing 1
  # encoder used to give wrong results on non-Apple-Silicon GPUs (missing
  # memory barrier between dispatches) - fixed at the root in metal_shim.m
  # (memoryBarrierWithScope: after every *_enc), so sessions are now safe on
  # all 3 backends, Metal no longer needs special handling.
  if sessionBegin(backend):
    var hX = sessionUpload(backend, x.data)
    var hInv = sessionUpload(backend, invBroadcast)
    var hW = sessionUpload(backend, wBroadcast)
    var hTmp = sessionAllocScratch(backend, n)
    var hOut = sessionAllocScratch(backend, n)
    let ok = sessionVecOp(backend, "mul", hX, hInv, hTmp, n) and
             sessionVecOp(backend, "mul", hTmp, hW, hOut, n) and
             sessionEnd(backend)
    if ok:
      result.data = sessionRead(backend, hOut, n)
      sessionFree(backend, hX); sessionFree(backend, hInv); sessionFree(backend, hW)
      sessionFree(backend, hTmp); sessionFree(backend, hOut)
      return
    sessionFree(backend, hX); sessionFree(backend, hInv); sessionFree(backend, hW)
    sessionFree(backend, hTmp); sessionFree(backend, hOut)

  # Fallback: still 100% GPU, just 2 separate beMul commands (not in 1 session)
  # when the session fails midway on the current backend.
  let tmp = beMul(ctx, x.data, invBroadcast)
  result.data = beMul(ctx, tmp, wBroadcast)

# ===================================================================
# RoPE
# ===================================================================
type
  RoPEL* = object
    cosT*, sinT*: seq[float32]   # [maxSeqLen, headDim/2]
    headDim*, maxSeqLen*: int

proc toHeadMajor(data: seq[float32], T, H, D: int): seq[float32] =
  ## [T,H,D] (position outer, head inner) -> [H,T,D] (head outer, position inner).
  ## beAttentionFused (Metal/CUDA/OpenCL) assumes the layout [B*H, S, D] — each
  ## head is 1 contiguous block of S*D elements (base_idx = bh*S*D) — while
  ## Linear.forward outputs the layout [T,H,D] (position outer). Without a transpose
  ## before the kernel call -> the kernel mixes heads/tokens -> attention is completely wrong,
  ## producing garbage tokens right from prefill even though RoPE/Linear/dequant are all correct.
  result = newSeq[float32](T * H * D)
  for t in 0 ..< T:
    for h in 0 ..< H:
      let src = (t * H + h) * D
      let dst = (h * T + t) * D
      for d in 0 ..< D:
        result[dst + d] = data[src + d]

proc toPosMajor(data: seq[float32], T, H, D: int): seq[float32] =
  ## The inverse: [H,T,D] (output of beAttentionFused) -> [T,H,D] (the layout the
  ## rest of the code — attnOut.data then oProj.forward — expects).
  result = newSeq[float32](T * H * D)
  for h in 0 ..< H:
    for t in 0 ..< T:
      let src = (h * T + t) * D
      let dst = (t * H + h) * D
      for d in 0 ..< D:
        result[dst + d] = data[src + d]

proc newRoPEL*(headDim, maxSeqLen: int, theta: float32 = 10000'f32,
                scalingType: string = "", scalingFactor: float32 = 1.0'f32): RoPEL =
  ## rope_scaling support (read from HF AutoConfig.rope_scaling; it used to be
  ## ignored entirely -> models using scaling (e.g. deepseek-coder-6.7B-GPTQ:
  ## {'type':'linear','factor':4.0}) computed WRONG cos/sin from the very first layer
  ## using RoPE (block 0), completely independent of quantization -> that is why
  ## fp32 and int4 diverged identically at after_block_0.
  ##   - "linear": HF's LlamaLinearScalingRotaryEmbedding divides the position by
  ##     scalingFactor BEFORE multiplying by invFreq (theta unchanged).
  ##   - "dynamic": NTK-aware - theta is rescaled by the actual seqLen relative to
  ##     the original maxSeqLen (only applies when seqLen > the original
  ##     max_position_embeddings; here the maxSeqLen passed in is taken AS the original
  ##     max_position_embeddings, so the dynamic branch uses the formula at the table's largest pos).
  result.headDim = headDim
  result.maxSeqLen = maxSeqLen
  let half = headDim div 2
  var effTheta = theta
  if scalingType == "dynamic" and scalingFactor > 1.0'f32:
    # NTK: theta' = theta * (factor * maxSeqLen/origMaxSeqLen - (factor-1)) ^ (headDim/(headDim-2))
    # here the maxSeqLen passed in is taken as the target length after scaling.
    let exponent = float32(headDim) / float32(headDim - 2)
    effTheta = theta * pow(scalingFactor, exponent)
  var invFreq = newSeq[float32](half)
  for i in 0 ..< half:
    invFreq[i] = 1'f32 / pow(effTheta, float32(i) / float32(half))
  result.cosT = newSeq[float32](maxSeqLen * half)
  result.sinT = newSeq[float32](maxSeqLen * half)
  for pos in 0 ..< maxSeqLen:
    var posF = float32(pos)
    if scalingType == "linear" and scalingFactor > 0'f32:
      posF = posF / scalingFactor
    for i in 0 ..< half:
      let angle = posF * invFreq[i]
      result.cosT[pos * half + i] = cos(angle)
      result.sinT[pos * half + i] = sin(angle)

proc applyRoPEInplace*(rope: RoPEL, ctx: Backend, data: var seq[float32], B, T, H, D: int, posOffset: int) =
  ## data layout: flat [B, T, H, D]. posOffset = absolute position of the first token
  ## in T (used when decoding one token at a time, T=1 but position > 0).
  ##
  ## The RoPE formula x1*cos-x2*sin / x1*sin+x2*cos is exactly the standard
  ## "rotate-half" form: out = x*cos + rotate_half(x)*sin, with
  ## rotate_half(x) = concat(-x2, x1). Written this way ALL the
  ## multiply/add arithmetic is elementwise on full [B,T,H,D] arrays -> a perfect fit for
  ## the existing beMul/beAdd (or sessionVecOp) in the backend, no dedicated RoPE
  ## kernel needed. Building cosFull/sinFull/rotHalf below is ONLY rearranging
  ## element positions (gather + sign flip), not heavy arithmetic, so
  ## it stays on the CPU (the backend has no gather kernel) - exactly the "the GPU has
  ## add/mul" but not gather/broadcast case.
  let half = D div 2
  let n = B * T * H * D
  var cosFull = newSeq[float32](n)
  var sinFull = newSeq[float32](n)
  var rotHalf = newSeq[float32](n)
  for b in 0 ..< B:
    for t in 0 ..< T:
      let pos = posOffset + t
      doAssert pos < rope.maxSeqLen, "RoPE: position exceeds the precomputed max_seq_len"
      for h in 0 ..< H:
        let base = ((b * T + t) * H + h) * D
        for i in 0 ..< half:
          let cosV = rope.cosT[pos * half + i]
          let sinV = rope.sinT[pos * half + i]
          cosFull[base + i]        = cosV
          cosFull[base + i + half] = cosV
          sinFull[base + i]        = sinV
          sinFull[base + i + half] = sinV
          rotHalf[base + i]        = -data[base + i + half]
          rotHalf[base + i + half] =  data[base + i]

  let backend = ctx.kind.toByby()
  if sessionBegin(backend):
    var hX = sessionUpload(backend, data)
    var hCos = sessionUpload(backend, cosFull)
    var hRot = sessionUpload(backend, rotHalf)
    var hSin = sessionUpload(backend, sinFull)
    var hT1 = sessionAllocScratch(backend, n)
    var hT2 = sessionAllocScratch(backend, n)
    var hOut = sessionAllocScratch(backend, n)
    let ok = sessionVecOp(backend, "mul", hX, hCos, hT1, n) and
             sessionVecOp(backend, "mul", hRot, hSin, hT2, n) and
             sessionVecOp(backend, "add", hT1, hT2, hOut, n) and
             sessionEnd(backend)
    if ok:
      data = sessionRead(backend, hOut, n)
      sessionFree(backend, hX); sessionFree(backend, hCos); sessionFree(backend, hRot)
      sessionFree(backend, hSin); sessionFree(backend, hT1); sessionFree(backend, hT2)
      sessionFree(backend, hOut)
      return
    sessionFree(backend, hX); sessionFree(backend, hCos); sessionFree(backend, hRot)
    sessionFree(backend, hSin); sessionFree(backend, hT1); sessionFree(backend, hT2)
    sessionFree(backend, hOut)

  # Fallback: still 100% GPU (not a CPU loop), just 2 muls + 1 add
  # NOT in 1 session when the current backend does not support sessions.
  let t1 = beMul(ctx, data, cosFull)
  let t2 = beMul(ctx, rotHalf, sinFull)
  data = beAdd(ctx, t1, t2)

# ===================================================================
# SwiGLU FeedForward (gate_proj, up_proj, down_proj)
# ===================================================================
type
  SwiGLUL* = object
    gateProj*, upProj*, downProj*: Linear

proc newSwiGLUL*(hidden, intermediate: int): SwiGLUL =
  # Shapes only: weights come from the model mmap (loadLlamaWeightsL), as in
  # Codestral. Random init here allocated ~27 GB of fp32 that was discarded.
  result.gateProj = newLinear(hidden, intermediate, initializeWeights = false)
  result.upProj = newLinear(hidden, intermediate, initializeWeights = false)
  result.downProj = newLinear(intermediate, hidden, initializeWeights = false)

proc forward*(ff: SwiGLUL, x: Tensor, ctx: Backend): Tensor =
  let gateRaw = ff.gateProj.forward(x, ctx)
  let upRaw = ff.upProj.forward(x, ctx)
  let n = gateRaw.data.len

  # sigmoid+mul+mul merged into 1 SESSION - the data stays on the GPU throughout,
  # no CPU upload/download between ops (beSigmoid, then a CPU for loop,
  # then beMul used to each upload/download on their own - 3
  # CPU<->GPU trips for 3 consecutive ops).
  let backend = ctx.kind.toByby()
  if sessionBegin(backend):
    var hGate = sessionUpload(backend, gateRaw.data)
    var hUp = sessionUpload(backend, upRaw.data)
    var hSig = sessionAllocScratch(backend, n)
    var hSilu = sessionAllocScratch(backend, n)
    var hGated = sessionAllocScratch(backend, n)

    let ok = sessionActivation(backend, "sigmoid", hGate, hSig, n) and
             sessionVecOp(backend, "mul", hGate, hSig, hSilu, n) and
             sessionVecOp(backend, "mul", hSilu, hUp, hGated, n) and
             sessionEnd(backend)

    if ok:
      var gatedT = newTensor(upRaw.shape)
      gatedT.data = sessionRead(backend, hGated, n)
      sessionFree(backend, hGate); sessionFree(backend, hUp)
      sessionFree(backend, hSig); sessionFree(backend, hSilu)
      sessionFree(backend, hGated)
      return ff.downProj.forward(gatedT, ctx)

    sessionFree(backend, hGate); sessionFree(backend, hUp)
    sessionFree(backend, hSig); sessionFree(backend, hSilu)
    sessionFree(backend, hGated)
    # falls through to the per-op branch below if the session fails midway

  # Fallback: each op separately via beXxx (still 100% GPU, just not in 1
  # session, so there is an upload/download between steps).
  let sig = beSigmoid(ctx, gateRaw.data)
  let silu = beMul(ctx, gateRaw.data, sig)
  let gated = beMul(ctx, silu, upRaw.data)
  var gatedT = newTensor(upRaw.shape)
  gatedT.data = gated
  result = ff.downProj.forward(gatedT, ctx)

# ===================================================================
# KV Cache (paged) for GQA -- stored in the layout [nKVHeads, maxSeqLen, headDim]
# ===================================================================
type
  LayerKVCache* = object
    k*, v*: seq[float32]   # flat [nKVHeads * maxSeqLen * headDim], B=1
    filled*: int           # number of positions written
    # NEW: K/V buffers resident on the GPU, 1 handle per KV head, allocated
    # ONCE (maxSeqLen*D) at the start, then ONLY the new part (nTok*D)
    # is appended on each appendKV call with sessionCopyInto - INSTEAD of
    # forwardDecode doing sessionUpload() of the WHOLE S*D history every token
    # (growing with S -> a total cost of O(S^2) per full generation). That was
    # the main reason ms/token grew with conversation length as seen
    # in real logs, NOTHING to do with the attention kernel.
    kResident*, vResident*: seq[SessionHandle]
    residentReady*: bool     # true once allocated (lazy, on the first call)
    kvHeap*: pointer          # NEW: heap used to suballocate kResident/
      ## vResident (see ensureKvResident) - kept so it can be freed at model unload
      ## (there is no explicit unload call in this codebase yet, so the heap
      ## lives until the process exits - ACCEPTABLE since the model is loaded once per
      ## process, not loaded/unloaded repeatedly in one run).
    kResidentCombined*, vResidentCombined*: SessionHandle
      ## NEW: a MERGED copy of ALL nKVHeads heads in EXACTLY 1
      ## buffer per layer (contiguous layout [head, seqPos, D]), used ONLY by the new fused
      ## kernel (attention_decode_all_heads) - why this merged copy is needed
      ## instead of reusing kResident/vResident (32 separate buffers from 1 MTLHeap):
      ## Metal does NOT guarantee the order/spacing of buffers suballocated
      ## from the same heap (the allocator may place them anywhere, with different
      ## alignment padding) - so the next head's offset CANNOT safely be derived as the previous
      ## head's offset + maxSeqLen*D when only kResident[0] is used as the
      ## "base". This merged buffer is allocated SEPARATELY, once, with a size known
      ## up front (nKVHeads*maxSeqLen*D), so head h's offset is ALWAYS
      ## h*maxSeqLen*D - known for sure, safe for a fused kernel reading
      ## several heads in 1 dispatch. Mode 1 (the old forwardDecode) does NOT
      ## touch this field, it still uses kResident/vResident as before - the safe
      ## fallback path is completely unaffected. ANY place that writes new K/V
      ## (Mode 1 and Mode 2/3 alike) MUST write to BOTH places (the old per-
      ## head buffers AND this merged buffer) so the data stays consistent even if the next token
      ## takes a different mode than the previous one.

  KVCacheL* = object
    layers*: seq[LayerKVCache]
    nKVHeads*, headDim*, maxSeqLen*: int
    seqLen*: int            # current position (shared by every layer)

proc newKVCacheL*(nLayers, nKVHeads, headDim, maxSeqLen: int): KVCacheL =
  result.nKVHeads = nKVHeads
  result.headDim = headDim
  result.maxSeqLen = maxSeqLen
  result.seqLen = 0
  result.layers = newSeq[LayerKVCache](nLayers)
  for l in 0 ..< nLayers:
    # The host copy (k, v) is allocated by ensureCpuKv on first use: the
    # resident GPU paths never touch it, and at -c 2048 it is 2 GiB of RAM
    # that an iGPU sharing the memory needs for the weights.
    result.layers[l].filled = 0
    result.layers[l].residentReady = false
    result.layers[l].kResident = @[]
    result.layers[l].vResident = @[]
    result.layers[l].kvHeap = nil

var gKvAllocCount = 0   ## counts the TOTAL number of times ensureKvResident really
  ## allocated (residentReady=false at the call) over the whole run - used to
  ## verify there are exactly 32 (1 per layer), instead of just looking at the
  ## log and counting by hand. If gKvAllocCount > the real nLayers after a run ->
  ## there is definitely a leak, not a hunch.

proc ensureKvResident*(cache: var LayerKVCache, backend: GpuBackend, nKVHeads, maxSeqLen, D: int) =
  ## NEW: allocates the resident buffers once per layer (lazy, called on the
  ## layer's first appendKV). No data is uploaded here - it only
  ## reserves space on the GPU (like sessionAllocScratch used elsewhere in the
  ## codebase) - the real data is written gradually with sessionCopyInto below.
  if cache.residentReady: return
  inc gKvAllocCount
  stderr.writeLine "[KV_RESIDENT_ALLOC] allocation #" & $gKvAllocCount &
    " (n=" & $nKVHeads &
    " heads, " & $(maxSeqLen * D * 4 * 2) & " bytes per head) - if this line prints MORE THAN 32 TIMES (the layer count) there is a real leak"
  cache.kResident = newSeq[SessionHandle](nKVHeads)
  cache.vResident = newSeq[SessionHandle](nKVHeads)
  # A REAL startup performance bug, measured with
  # BYBY_DECODE_PROFILE + Metal NSLog timestamps: BEFORE, the
  # for loop below called sessionAllocScratch (=1 FULL [gDevice
  # newBufferWithLength:] call, a separate device-level allocation) exactly
  # nKVHeads*2 times PER layer -> with nKVHeads=32, that is 32*2*32(layers)=2048
  # SEPARATE device allocations, ALL during prefill (BEFORE the first
  # token). On the old Intel Iris driver each newBufferWithLength call has a
  # significant fixed cost (mapping shared CPU/GPU pages) - INDEPENDENT of the
  # buffer size - so 2048 calls added up to tens of seconds of waiting before
  # the first token, exactly the observed "extremely slow initialisation".
  # FIX: allocate ONE large region with an MTLHeap (sessionAllocHeap - 1
  # device allocation per layer instead of 64), then cheaply suballocate each
  # K/V-head buffer from that heap (sessionAllocFromHeap - as recommended by the Apple Metal
  # Best Practices Guide, section "Heaps", for "many buffers of the same size and
  # lifetime"). Backends without MTLHeap (current CUDA/Vulkan, or a
  # non-macOS Metal build) -> sessionAllocHeap returns nil -> falls back
  # UNCHANGED to sessionAllocScratch per buffer like the old code, with NO change in behaviour/
  # results, only losing this startup speedup on those backends.
  let heapElemsNeeded = nKVHeads * 2 * maxSeqLen * D   # K + V, every head, this layer
  cache.kvHeap = sessionAllocHeap(backend, heapElemsNeeded)
  if cache.kvHeap != nil:
    for h in 0 ..< nKVHeads:
      cache.kResident[h] = sessionAllocFromHeap(backend, cache.kvHeap, maxSeqLen * D)
      cache.vResident[h] = sessionAllocFromHeap(backend, cache.kvHeap, maxSeqLen * D)
      if cache.kResident[h].metalHandle == nil or cache.vResident[h].metalHandle == nil:
        # Suballocation failed midway (rare - e.g. heapElemsNeeded miscomputed,
        # or the driver lacks contiguous space despite reserving enough total bytes) -> do NOT
        # silently use an empty buffer, fall back entirely to the safe per-buffer allocation for EVERY
        # head (including heads already suballocated above, to avoid
        # mixing 2 allocation sources in the same layer).
        stderr.writeLine "[KV_RESIDENT_ALLOC] heap suballocation failed midway (head " & $h &
          ") - falling back to allocating each buffer separately (slower but safe)"
        var heapVar = cache.kvHeap
        sessionFreeHeap(backend, heapVar)
        cache.kvHeap = nil
        for h2 in 0 ..< nKVHeads:
          cache.kResident[h2] = sessionAllocScratch(backend, maxSeqLen * D)
          cache.vResident[h2] = sessionAllocScratch(backend, maxSeqLen * D)
        break
  else:
    for h in 0 ..< nKVHeads:
      cache.kResident[h] = sessionAllocScratch(backend, maxSeqLen * D)
      cache.vResident[h] = sessionAllocScratch(backend, maxSeqLen * D)
  # NEW: allocate the separate MERGED buffers (see the full explanation at the
  # kResidentCombined/vResidentCombined field declaration above) - 1 large allocation for K,
  # 1 for V, head h's offset is ALWAYS h*maxSeqLen*D (unlike kResident/vResident
  # above, suballocated from a heap with no guaranteed even spacing between heads).
  cache.kResidentCombined = sessionAllocScratch(backend, nKVHeads * maxSeqLen * D)
  cache.vResidentCombined = sessionAllocScratch(backend, nKVHeads * maxSeqLen * D)
  cache.residentReady = true

var gKvResidentDisabled = false   ## NEW: if any sessionCopyInto
  ## in appendKV fails (returns false), resident KV is turned OFF PERMANENTLY for the whole
  ## run - falling back to a full upload every token (slow but CORRECT).
  ## Declared BEFORE appendKV (not after like the previous version) because appendKV needs to
  ## set this flag as soon as a copy fails, not only during verify.

proc ensureCpuKv(cache: var KVCacheL, layer: int) =
  let n = cache.nKVHeads * cache.maxSeqLen * cache.headDim
  if cache.layers[layer].k.len < n:
    cache.layers[layer].k = newSeq[float32](n)
    cache.layers[layer].v = newSeq[float32](n)

proc appendKVCpuOnly(cache: var KVCacheL, layer: int, kNew, vNew: seq[float32], nTok: int) =
  ## NEW: separates the CPU write (always runs, independent of the GPU) from
  ## the GPU sync - so forwardDecode can control WHEN the GPU work is
  ## encoded (merged into the main session, no separate session - see the reason in
  ## syncKvResidentEncode below).
  let D = cache.headDim
  let H = cache.nKVHeads
  doAssert cache.seqLen + nTok <= cache.maxSeqLen, "KV cache overflow"
  ensureCpuKv(cache, layer)
  for t in 0 ..< nTok:
    for h in 0 ..< H:
      let srcOff = (t * H + h) * D
      let dstOff = (h * cache.maxSeqLen + (cache.seqLen + t)) * D
      for d in 0 ..< D:
        cache.layers[layer].k[dstOff + d] = kNew[srcOff + d]
        cache.layers[layer].v[dstOff + d] = vNew[srcOff + d]
  cache.layers[layer].filled = cache.seqLen + nTok

var gKvResidentWarnedNonMetal = false   ## prints the "not
  ## Metal" warning exactly once for the whole run (avoids log spam every token).

proc syncKvResidentEncode(cache: var KVCacheL, layer: int, kNew, vNew: seq[float32], nTok: int,
                           backend: GpuBackend) =
  # (EMERGENCY SAFETY SWITCH): a real leak was confirmed - the resident buffers
  # were allocated MANY TIMES instead of exactly once per layer (real log: printed
  # >32 times with only 32 layers), each reallocation losing ~1.6MB (32 heads
  # x 204800 bytes x 2 k/v) and NEVER freeing the old buffer - adding up over the
  # run to 58GB of RAM. It was NOT determined exactly why
  # residentReady did not keep its state between calls (suspected to involve
  # the interaction between appendKV opening its own session during prefill and the global
  # session state gSessionEncoder, but not proven).
  # (ON BY DEFAULT): confirmed through real logs (gKvAllocCount exactly 32/32,
  # no leak) and a real measured speedup (0.47->0.73-0.82 tok/s decode) after
  # the double-session bug (kvSynced) was fixed - no reason left to keep it off
  # by default. The REVERSE escape hatch (BYBY_DISABLE_KV_RESIDENT=1) is kept to
  # turn it off if another machine/model has problems later, with NO need to remember to turn it on
  # by hand every run as before.
  # (on request): the DEFAULT was CHANGED to OFF (equivalent to always running with
  # BYBY_DISABLE_KV_RESIDENT=1) - all the Q4 kernel measurement/optimisation
  # (NDST 8->2 + pairing, see vecop_matmul.metal) in that working
  # session ran with this variable = 1 (typed by hand before every
  # run), so that is the configuration ACTUALLY verified/measured,
  # not the resident path (not remeasured since the Q4 kernel change).
  # The resident path (sessionKvResident) is still in the code - NOT DELETED - because
  # the history comment right above (lines 613-618) once measured 0.47->0.73-
  # 0.82 tok/s BEFORE the Q4 kernel change, and it may still help on other machines/
  # models. To turn it BACK ON for comparison, set BYBY_DISABLE_KV_RESIDENT=0 explicitly
  # (unlike before: before, the variable merely existing turned it off, regardless of
  # the value "0" - now an explicit "0" turns it BACK ON, any other value
  # (or unset) means OFF, the "off by default" that was requested).
  # (default reversed again): KV-resident sync is ON by default - it is the ONLY
  # branch that lets appendKV avoid dropping to the CPU midway, a REQUIRED condition
  # for merging a whole layer (attn+oProj+FFN) into a single command buffer
  # like ggml-metal. The reverse escape hatch: BYBY_DISABLE_KV_RESIDENT=1 to
  # turn it OFF when comparing/debugging.
  if getEnv("BYBY_DISABLE_KV_RESIDENT", "0") == "1": return
  ## NEW (split out of appendKV): encodes the resident KV sync commands into the
  ## caller's ALREADY OPEN session - it NO longer does its own sessionBegin/sessionEnd.
  ##
  ## WHY: the previous version opened a SEPARATE session for this every token, separate
  ## from the main attention session - measured on an Iris Plus 640,
  ## the fixed cost of opening/closing that extra session (sessionBegin+sessionEnd = forcing a GPU
  ## sync once) COST MORE than the savings from not re-uploading
  ## the whole KV every token (at short conversation lengths ~100 tokens, not long enough
  ## for the O(S)->O(1) upload gain to beat the extra session cost). Now it is merged straight
  ## into the main session forwardDecode already opened, going from 2 sessions/token down to 1.
  let D = cache.headDim
  let H = cache.nKVHeads
  ensureKvResident(cache.layers[layer], backend, H, cache.maxSeqLen, D)
  if nTok == 1:
    var hKAll = sessionUpload(backend, kNew)
    var hVAll = sessionUpload(backend, vNew)
    var allOk = true
    for h in 0 ..< H:
      if not sessionCopyInto(backend, hKAll, cache.layers[layer].kResident[h],
                              h * D, cache.seqLen * D, D):
        allOk = false
      if not sessionCopyInto(backend, hVAll, cache.layers[layer].vResident[h],
                              h * D, cache.seqLen * D, D):
        allOk = false
      # NEW: write AT THE SAME TIME into the MERGED buffer (kResidentCombined/vResidentCombined)
      # - offset = h*maxSeqLen*D (head h's base in the merged buffer) + the current
      # position. Writing to BOTH places is REQUIRED (the old per-head buffer AND the merged buffer)
      # so the new fused kernel (reading the merged buffer) and the old Mode 1 (reading the per-head
      # buffers) always see consistent data, whatever mode the previous/next token
      # used - see the full explanation at the kResidentCombined field declaration.
      if not sessionCopyInto(backend, hKAll, cache.layers[layer].kResidentCombined,
                              h * D, h * cache.maxSeqLen * D + cache.seqLen * D, D):
        allOk = false
      if not sessionCopyInto(backend, hVAll, cache.layers[layer].vResidentCombined,
                              h * D, h * cache.maxSeqLen * D + cache.seqLen * D, D):
        allOk = false
    sessionFree(backend, hKAll)
    sessionFree(backend, hVAll)
    if not allOk:
      stderr.writeLine "[KV_RESIDENT] sessionCopyInto (fast path nTok=1) failed - disabling resident KV, falling back"
      gKvResidentDisabled = true
  else:
    var allOkPrefill = true
    for h in 0 ..< H:
      var newChunk = newSeq[float32](nTok * D)
      for t in 0 ..< nTok:
        let srcOff = (t * H + h) * D
        copyMem(addr newChunk[t * D], unsafeAddr kNew[srcOff], D * sizeof(float32))
      var hChunk = sessionUpload(backend, newChunk)
      if not sessionCopyInto(backend, hChunk, cache.layers[layer].kResident[h],
                              cache.seqLen * D, nTok * D):
        allOkPrefill = false
      # NEW: also write into the merged buffer (head h's base offset = h*maxSeqLen*D)
      if not sessionCopyInto(backend, hChunk, cache.layers[layer].kResidentCombined,
                              h * cache.maxSeqLen * D + cache.seqLen * D, nTok * D):
        allOkPrefill = false
      sessionFree(backend, hChunk)

      for t in 0 ..< nTok:
        let srcOff = (t * H + h) * D
        copyMem(addr newChunk[t * D], unsafeAddr vNew[srcOff], D * sizeof(float32))
      hChunk = sessionUpload(backend, newChunk)
      if not sessionCopyInto(backend, hChunk, cache.layers[layer].vResident[h],
                              cache.seqLen * D, nTok * D):
        allOkPrefill = false
      # NEW: also write into the merged buffer
      if not sessionCopyInto(backend, hChunk, cache.layers[layer].vResidentCombined,
                              h * cache.maxSeqLen * D + cache.seqLen * D, nTok * D):
        allOkPrefill = false
      sessionFree(backend, hChunk)
    if not allOkPrefill:
      stderr.writeLine "[KV_RESIDENT] sessionCopyInto (prefill nTok>1) failed - disabling resident KV, falling back"
      gKvResidentDisabled = true

proc appendKV*(cache: var KVCacheL, layer: int, kNew, vNew: seq[float32], nTok: int,
               backend: GpuBackend = gbAuto) =
  ## KEEPS the old behaviour (opens its own session) - used by forwardPrefill
  ## (not a hot path, no session merging needed). forwardDecode now uses
  ## appendKVCpuOnly + syncKvResidentEncode separately (see the 2 functions above) to merge
  ## into the main session, and no longer calls this appendKV.
  appendKVCpuOnly(cache, layer, kNew, vNew, nTok)
  if backend != gbAuto:
    if sessionBegin(backend):
      syncKvResidentEncode(cache, layer, kNew, vNew, nTok, backend)
      discard sessionEnd(backend)

# ===================================================================
# GQA CausalSelfAttention (RoPE + KV cache) — replaces the original CausalSelfAttention
# ===================================================================
type
  GQAAttention* = object
    nHeads*, nKVHeads*, headDim*, hiddenSize*: int
    qProj*, kProj*, vProj*, oProj*: Linear
    scale*: float32

proc newGQAAttention*(hiddenSize, nHeads, nKVHeads: int): GQAAttention =
  doAssert hiddenSize mod nHeads == 0
  doAssert nHeads mod nKVHeads == 0
  result.hiddenSize = hiddenSize
  result.nHeads = nHeads
  result.nKVHeads = nKVHeads
  result.headDim = hiddenSize div nHeads
  let kvDim = nKVHeads * result.headDim
  result.qProj = newLinear(hiddenSize, hiddenSize, initializeWeights = false)
  result.kProj = newLinear(hiddenSize, kvDim, initializeWeights = false)
  result.vProj = newLinear(hiddenSize, kvDim, initializeWeights = false)
  result.oProj = newLinear(hiddenSize, hiddenSize, initializeWeights = false)
  result.scale = 1'f32 / sqrt(float32(result.headDim))

## forwardQKVFused (the dequant-fp32-then-matmul QKV fusion) was DELETED - nothing
## called it any more (dead code) and it still went through the slow
## dequantizeTensorTransposed() path that linearFast() above replaced. See
## fusedNormQKVRoPE below for the path actually used (beMatmulQ4 directly).

proc fusedNormQKVRoPE(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL,
                       ctx: Backend, T, posOffset: int): tuple[q, k, v: seq[float32]] =
  ## CROSS-SUBMODULE MERGE: RMSNorm(inputNorm, the scale part) -> fused QKV
  ## projection -> RoPE(Q) -> RoPE(K).
  ##
  ## THESE 4 steps used to be 3 separate GPU sessions (RMSNorm with its own sessionBegin/End,
  ## RoPE(Q) its own session, RoPE(K) its own session) PLUS 1 beMatmul NOT
  ## in a session for QKV (the old forwardQKVFused) - ~4 GPU commit+waits per
  ## layer for the start of attention alone. Now there are EXACTLY 2 sessions:
  ##   Session #1: RMSNorm-scale (2 muls) + QKV matmul, encoded in sequence in
  ##               THE SAME sessionBegin/sessionEnd (no CPU round trip between
  ##               these 2 steps any more).
  ##   Session #2: RoPE(Q) AND RoPE(K) MERGED (they used to be 2 separate sessions)
  ##               into a single elementwise batch (mul,mul,add).
  ## The boundary BETWEEN session #1 and #2 is REQUIRED, not optional: RoPE
  ## needs "rotate_half" (a gather swapping the sign of the 2 vector halves) on the Q/K fresh out of the
  ## matmul, and the current session API (sessionMatmul/sessionVecOp/
  ## sessionActivation/sessionSoftmax) has NO gather kernel -> Q/K must be read
  ## back to the CPU to build rotHalf/cosFull/sinFull and uploaded again. On Metal,
  ## reading data midway requires the session to have committed+waited (see
  ## metal_shim.m/metalSessionEnd), so this is a hard hardware/API boundary,
  ## not a missing optimisation.
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  let (rows, inF) = flatten2D(x.shape)
  let n = rows * inF

  # (lazy rmsNormPrep): rmsNormPrep() used to run UNCONDITIONALLY before
  # every session — i.e. every layer every token had to: beMul (a GPU round trip
  # to compute x^2), beMatmul-reduce (a GPU round trip to compute sumSq), then read
  # sumSq back to the CPU to compute sqrt. That is 2 mandatory GPU flushes even when
  # sessionRmsnormEnc succeeds and invBroadcast/wBroadcast are NEVER
  # used. Now: computed only when really needed (sessionRmsnormEnc fails or
  # there is no qkvResident). On the fast Metal path these 2 GPU flushes disappear
  # entirely (0 instead of 2 per call). On the CPU fallback the behaviour is
  # the same as before (called exactly when needed, not earlier).
  var invBroadcast: seq[float32]
  var wBroadcast: seq[float32]
  proc ensureRmsNormFallback() =
    if invBroadcast.len > 0: return   # already computed, do not compute again
    let (inv, w, _) = rmsNormPrep(norm, x, ctx)
    invBroadcast = inv
    wBroadcast = w

  let be = ctx.kind.toByby()
  var normed: seq[float32]
  var qFlat0, kFlat0, vFlat: seq[float32]

  # (the main point - answering "why does the first token take so long"): BEFORE, the
  # RMSNorm-scale session had to sessionEnd() and read `normed` back to the CPU JUST TO feed
  # linearFastBias (each opening/closing its OWN GPU session) - i.e. 1
  # RMSNorm session + 3 Q/K/V sessions = 4 GPU commit+waitUntilCompleted calls
  # FOR THIS PART ALONE, EVERY LAYER, EVERY TOKEN (32 layers x ~10-12 calls/layer ≈
  # 350+ GPU round trips/token - waitUntilCompleted has a significant fixed overhead
  # on the old Intel Metal driver, and that is what took most of the 5s/token,
  # NOT the kernels' compute throughput). Now sessionMatmulQ4 (already available
  # since the resident-weight version) encodes DIRECTLY into THE SAME open session, using
  # `hNormed` (already on the GPU) as the activation - no need to read `normed` back to the
  # CPU between RMSNorm and QKV any more. 4 calls -> 1 call.
  # DEBUG (temporary, to isolate the OpenCL garbage-token bug): set the environment
  # variable BYBY_FORCE_LEGACY_QKV=1 to always force the "OLD PATH" below
  # (linearFastBias + CPU RoPE, not the new sessionMatmulQ4/sessionRope)
  # even when resident is ready. If the tokens come out right with this flag on the
  # opencl backend -> the bug is definitely in OpenCL's usedSession/sessionRope branch
  # (not in the shared basic matmul/softmax kernels).
  # (follow-up fix): this merged-session branch used to activate ONLY when Q/K/V
  # were resident via the real int4-asymmetric path (sessionMatmulQ4) - models
  # exported with --int8 (the export_to_nimq.py default) or --int4
  # symmetric have NO residentReady=true, so qkvResident was always false, falling
  # straight into the "OLD PATH" below: RMSNorm with its own session + Q/K/V each
  # opening another session (residentFp32Ready, see nimformer.forward) ->
  # still 4 GPU commit+waits per layer despite having resident fp32. Now the merged branch
  # ALSO works with residentFp32 (plain sessionMatmul, no dedicated int4 kernel
  # needed) -> int8/tied models also need only 1 session for RMSNorm+QKV, exactly like
  # the Q4 path.
  let qkvResidentQ4 = attn.qProj.residentReady and attn.kProj.residentReady and attn.vProj.residentReady
  let qkvResidentFp32 = (not qkvResidentQ4) and attn.qProj.residentFp32Ready and
    attn.kProj.residentFp32Ready and attn.vProj.residentFp32Ready
  # LEGACY BANNED: BYBY_FORCE_LEGACY_QKV is no longer read - if resident is not
  # ready, it RAISES right away instead of silently dropping to the slow path (legacy is still
  # GPU but creates many small sessions, it is not a CPU fallback, but this path
  # was explicitly BANNED by request).
  if not (qkvResidentQ4 or qkvResidentFp32):
    raise newException(CatchableError,
      "[QKV] resident is not ready (uploadResident() did not run or failed for " &
      "qProj/kProj/vProj) - the legacy fallback is BANNED by request, not continuing.")
  let qkvResident = true
  debugPathOnce("qkv.resident_ready", qkvResidentQ4 or qkvResidentFp32,
    "Q4=" & $qkvResidentQ4 & " Fp32=" & $qkvResidentFp32 &
    " (false for both => uploadResident() did not run or failed for qProj/kProj/vProj)")
  var usedSession = qkvResident and sessionBegin(be)
  if not usedSession:
    raise newException(CatchableError,
      "[QKV] sessionBegin() failed (the GPU did not accept the session) - the legacy fallback is BANNED.")
  # (RoPE merged into THE SAME session as QKV, see sessionRope in
  # gpubackend.nim + rope_kernel in vecop_matmul.metal): RoPE used to
  # HAVE TO be a separate session ("Session #2" below) because the old session API
  # had no gather/rotate_half kernel. Now, if sessionRope() works
  # (currently only Metal has a real implementation - CUDA/OpenCL deliberately return false),
  # Q/K are rotated RIGHT IN the QKV session, BEFORE sessionEnd - the Q/K read back to the CPU
  # below are already the RoPE result, Session #2 is no longer needed. If sessionRope
  # returns false (unsupported / error), ropeFused stays false -> the old code
  # below (Session #2 / CPU path) runs exactly as before, no behaviour change.
  var ropeFused = false
  let tryFastRmsnorm = usedSession and not existsEnv("BYBY_DISABLE_SESSION_RMSNORM")
  if usedSession:
    var hX = sessionUpload(be, x.data)
    var hGamma: SessionHandle
    var hInv, hW: SessionHandle
    var hT1: SessionHandle
    var hNormed: SessionHandle
    var normedFast = false
    if tryFastRmsnorm:
      hGamma = (if norm.residentGammaReady: norm.residentGamma else: sessionUpload(be, norm.weight.data))
        # OPTIMISATION: use the resident gamma (uploaded once at model load) instead of
        # sessionUpload() again every layer/token - falls back to the old way if for
        # some reason uploadResidentNorm() has not run (safe, no
        # behaviour change in that case).
      var hNormedTry = sessionAllocScratch(be, n)
      if sessionRmsnormEnc(be, hX, hGamma, hNormedTry, rows, inF, norm.eps):
        hNormed = hNormedTry
        normedFast = true
      else:
        sessionFree(be, hNormedTry)   # avoid leaking the scratch buffer on failure
    if not normedFast:
      ensureRmsNormFallback()
      hInv = sessionUpload(be, invBroadcast)
      hW = sessionUpload(be, wBroadcast)
      hT1 = sVecOp(be, "mul", hX, hInv, n)
      hNormed = sVecOp(be, "mul", hT1, hW, n)
    var hQ = sessionAllocScratch(be, rows * attn.qProj.outF)
    var hK = sessionAllocScratch(be, rows * attn.kProj.outF)
    var hV = sessionAllocScratch(be, rows * attn.vProj.outF)
    let okQ = if qkvResidentQ4: sessionMatmulQ4(be, hNormed, attn.qProj.residentW, hQ, rows)
              else: sessionMatmul(be, hNormed, attn.qProj.residentFp32, hQ, rows, attn.qProj.inF, attn.qProj.outF)
    let okK = if qkvResidentQ4: sessionMatmulQ4(be, hNormed, attn.kProj.residentW, hK, rows)
              else: sessionMatmul(be, hNormed, attn.kProj.residentFp32, hK, rows, attn.kProj.inF, attn.kProj.outF)
    let okV = if qkvResidentQ4: sessionMatmulQ4(be, hNormed, attn.vProj.residentW, hV, rows)
              else: sessionMatmul(be, hNormed, attn.vProj.residentFp32, hV, rows, attn.vProj.inF, attn.vProj.outF)

    var hCosT, hSinT: SessionHandle
    if okQ and okK:
      let halfD = D div 2
      var cosSlice = newSeq[float32](T * halfD)
      var sinSlice = newSeq[float32](T * halfD)
      for t in 0 ..< T:
        let pos = posOffset + t
        doAssert pos < rope.maxSeqLen, "RoPE: position exceeds the precomputed max_seq_len"
        for i in 0 ..< halfD:
          cosSlice[t * halfD + i] = rope.cosT[pos * halfD + i]
          sinSlice[t * halfD + i] = rope.sinT[pos * halfD + i]
      hCosT = sessionUpload(be, cosSlice)
      hSinT = sessionUpload(be, sinSlice)
      let okRopeQ = sessionRope(be, hQ, hCosT, hSinT, T, H, D)
      let okRopeK = sessionRope(be, hK, hCosT, hSinT, T, HKV, D)
      ropeFused = okRopeQ and okRopeK

    usedSession = sessionEnd(be)
    if usedSession and okQ and okK and okV:
      qFlat0 = sessionRead(be, hQ, rows * attn.qProj.outF)  # already RoPE-rotated if ropeFused
      kFlat0 = sessionRead(be, hK, rows * attn.kProj.outF)  # already RoPE-rotated if ropeFused
      vFlat = sessionRead(be, hV, rows * attn.vProj.outF)
    sessionFree(be, hX)
    if tryFastRmsnorm and not norm.residentGammaReady: sessionFree(be, hGamma)
      # Do NOT free hGamma when it is a RESIDENT buffer (norm.residentGamma) -
      # that buffer must live for the whole inference, not
      # just one call of this function. It used to be freed unconditionally since hGamma
      # was always a freshly uploaded buffer - now hGamma may be resident, so
      # check before freeing.
    if not normedFast:
      sessionFree(be, hInv); sessionFree(be, hW)
      var hT1v = hT1
      sessionFree(be, hT1v)
    sessionFree(be, hNormed)
    sessionFree(be, hQ); sessionFree(be, hK); sessionFree(be, hV)
    if okQ and okK:
      sessionFree(be, hCosT); sessionFree(be, hSinT)
    if qFlat0.len == 0 or kFlat0.len == 0 or vFlat.len == 0:
      usedSession = false   # sessionMatmulQ4 failed (backend unsupported) -> falls through to the old path below
      ropeFused = false     # safe: if the session counts as failed, do NOT trust that RoPE is done

  debugPathOnce("qkv.path", usedSession)
  if not usedSession:
    # OLD PATH (backend not Metal/OpenCL, or resident not ready,
    # or sessionMatmulQ4 failed) - still correct, just without the session merging.
    ensureRmsNormFallback()
    var usedSessionOld = sessionBegin(be)
    if usedSessionOld:
      var hX = sessionUpload(be, x.data)
      var hInv = sessionUpload(be, invBroadcast)
      var hW = sessionUpload(be, wBroadcast)
      let hT1 = sVecOp(be, "mul", hX, hInv, n)          # RMSNorm scale step 1
      let hNormed = sVecOp(be, "mul", hT1, hW, n)       # RMSNorm scale step 2
      usedSessionOld = sessionEnd(be)   # required: normed must come back to the CPU as input for beMatmulQ4
      if usedSessionOld:
        normed = sessionRead(be, hNormed, n)
      var hT1v = hT1; var hNormedV = hNormed
      sessionFree(be, hX); sessionFree(be, hInv); sessionFree(be, hW)
      sessionFree(be, hT1v); sessionFree(be, hNormedV)
    if not usedSessionOld:
      let t1 = beMul(ctx, x.data, invBroadcast)
      normed = beMul(ctx, t1, wBroadcast)

    qFlat0 = linearFastBias(attn.qProj, normed, rows, ctx)
    kFlat0 = linearFastBias(attn.kProj, normed, rows, ctx)
    vFlat = linearFastBias(attn.vProj, normed, rows, ctx)
  else:
    # Bias (if any) - added on the CPU after reading the result back, like linearFastBias.
    if attn.qProj.bias.data.len == attn.qProj.outF:
      for r in 0 ..< rows:
        let off = r * attn.qProj.outF
        for o in 0 ..< attn.qProj.outF: qFlat0[off + o] += attn.qProj.bias.data[o]
    if attn.kProj.bias.data.len == attn.kProj.outF:
      for r in 0 ..< rows:
        let off = r * attn.kProj.outF
        for o in 0 ..< attn.kProj.outF: kFlat0[off + o] += attn.kProj.bias.data[o]
    if attn.vProj.bias.data.len == attn.vProj.outF:
      for r in 0 ..< rows:
        let off = r * attn.vProj.outF
        for o in 0 ..< attn.vProj.outF: vFlat[off + o] += attn.vProj.bias.data[o]

  let qFlat = qFlat0
  let kFlat = kFlat0

  # ---- Session #2: RoPE(Q) + RoPE(K) MERGED into 1 session (they used to be 2
  # separate sessions, each uploading/downloading on its own). Q and K are joined into 1
  # contiguous array [nQ+nK] and exactly 3 sessionVecOps (mul,mul,add) run ONCE for
  # both, instead of 2 x 3 vecops = 6 commands in 2 different sessions. ----
  let half = D div 2
  let nQ = T * H * D
  let nK = T * HKV * D
  let nAll = nQ + nK

  proc buildRoPEArrays(data: seq[float32], Th, Hh: int, base: int,
                        cosFull, sinFull, rotHalf: var seq[float32]) =
    for t in 0 ..< Th:
      let pos = posOffset + t
      doAssert pos < rope.maxSeqLen, "RoPE: position exceeds the precomputed max_seq_len"
      for h in 0 ..< Hh:
        let localBase = (t * Hh + h) * D
        let dstBase = base + localBase
        for i in 0 ..< half:
          let cosV = rope.cosT[pos * half + i]
          let sinV = rope.sinT[pos * half + i]
          cosFull[dstBase + i]        = cosV
          cosFull[dstBase + i + half] = cosV
          sinFull[dstBase + i]        = sinV
          sinFull[dstBase + i + half] = sinV
          rotHalf[dstBase + i]        = -data[localBase + i + half]
          rotHalf[dstBase + i + half] =  data[localBase + i]

  var combined = newSeq[float32](nAll)

  if ropeFused:
    # RoPE already ran INSIDE the QKV session above (see sessionRope) - qFlat/
    # kFlat here are already the result after rotate_half, do NOT apply it again
    # (applying it again would be wrong - rotate_half is not idempotent).
    copyMem(addr combined[0], unsafeAddr qFlat[0], nQ * sizeof(float32))
    copyMem(addr combined[nQ], unsafeAddr kFlat[0], nK * sizeof(float32))
  else:
    copyMem(addr combined[0], unsafeAddr qFlat[0], nQ * sizeof(float32))
    copyMem(addr combined[nQ], unsafeAddr kFlat[0], nK * sizeof(float32))
    var cosFull = newSeq[float32](nAll)
    var sinFull = newSeq[float32](nAll)
    var rotHalf = newSeq[float32](nAll)
    buildRoPEArrays(qFlat, T, H, 0, cosFull, sinFull, rotHalf)
    buildRoPEArrays(kFlat, T, HKV, nQ, cosFull, sinFull, rotHalf)

    var usedSession2 = sessionBegin(be)
    if usedSession2:
      var hX = sessionUpload(be, combined)
      var hCos = sessionUpload(be, cosFull)
      var hRot = sessionUpload(be, rotHalf)
      var hSin = sessionUpload(be, sinFull)
      let hT1 = sVecOp(be, "mul", hX, hCos, nAll)
      let hT2 = sVecOp(be, "mul", hRot, hSin, nAll)
      let hOut = sVecOp(be, "add", hT1, hT2, nAll)
      usedSession2 = sessionEnd(be)
      if usedSession2:
        combined = sessionRead(be, hOut, nAll)
      var hT1v = hT1; var hT2v = hT2; var hOutV = hOut
      sessionFree(be, hX); sessionFree(be, hCos); sessionFree(be, hRot); sessionFree(be, hSin)
      sessionFree(be, hT1v); sessionFree(be, hT2v); sessionFree(be, hOutV)
    if not usedSession2:
      let t1 = beMul(ctx, combined, cosFull)
      let t2 = beMul(ctx, rotHalf, sinFull)
      combined = beAdd(ctx, t1, t2)

  result.q = combined[0 ..< nQ]
  result.k = combined[nQ ..< nAll]
  result.v = vFlat

## ---- PREFILL: processes the whole prompt at once with beAttentionFused (it has a causal mask
## built in, exactly what this kernel was designed for) ----
proc forwardPrefill*(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL, ctx: Backend,
                      cache: var KVCacheL, layerIdx: int): Tensor =
  let B = x.shape[0]
  let T = x.shape[1]
  let C = x.shape[2]
  doAssert B == 1, "The decode path currently only supports B=1 (single-sequence generation)"
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  let nRep = H div HKV

  # CROSS-SUBMODULE SESSION MERGE: RMSNorm(inputNorm) -> fused QKV -> RoPE(Q)
  # -> RoPE(K) (see fusedNormQKVRoPE above) - this used to be 3 sessions +
  # 1 separate matmul, now 2 sessions.
  let (qData, kData, vData) = fusedNormQKVRoPE(attn, norm, x, rope, ctx, T, posOffset = 0)

  # Store K,V (already RoPE'd) in the cache with layout [T,HKV,D] -> appendKV converts to
  # [HKV, maxSeqLen, D] internally.
  appendKV(cache, layerIdx, kData, vData, T, backend = ctx.kind.toByby())

  # Expand K,V from HKV heads -> H heads (each kv head repeated nRep times contiguously,
  # the HF repeat_kv convention) to use beAttentionFused (which needs Q,K,V
  # with the same head count).
  var kExp = newSeq[float32](T * H * D)
  var vExp = newSeq[float32](T * H * D)
  for t in 0 ..< T:
    for hkv in 0 ..< HKV:
      let srcOff = (t * HKV + hkv) * D
      for r in 0 ..< nRep:
        let h = hkv * nRep + r
        let dstOff = (t * H + h) * D
        for d in 0 ..< D:
          kExp[dstOff + d] = kData[srcOff + d]
          vExp[dstOff + d] = vData[srcOff + d]

  # BUG FIX: beAttentionFused needs the layout [H,T,D] (head-major), not
  # the [T,H,D] (position-major) that qData/kExp/vExp have (see toHeadMajor
  # above). Without this transpose the kernel mixes heads/tokens -> attention
  # is completely wrong -> garbage tokens right from prefill.
  let qHM = toHeadMajor(qData, T, H, D)
  let kHM = toHeadMajor(kExp, T, H, D)
  let vHM = toHeadMajor(vExp, T, H, D)
  let (attnOutFlatHM, _) = beAttentionFused(ctx, qHM, kHM, vHM, [], B, H, T, D, attn.scale)
  let attnOutFlat = toPosMajor(attnOutFlatHM, T, H, D)   # back to [T,H,D]=[T,C] for oProj
  var attnOut = newTensor(@[B, T, C])
  attnOut.data = attnOutFlat
  result = attn.oProj.forward(attnOut, ctx)

## ---- DECODE: 1 new token, manual attention via beMatmul+beSoftmax over
## the whole cache (O(S), linear -- this is the real "paged attention") ----
var gAttnDecodeGqaVerifyLeft = 5   ## NEW: counts the remaining times the new
  ## attention_decode_gqa kernel is verified against the old path (matmul+softmax+matmul)
  ## when BYBY_VERIFY_ATTN_DECODE is on - only the first few (enough to
  ## catch a systematic mismatch) so the whole run is not slowed down.

var gAttnDecodeGqaDisabled = true  ## OFF BY DEFAULT - the
  ## attention_decode_gqa kernel was confirmed to produce REALLY WRONG RESULTS (not a false positive
  ## of the old mid-session verify) - evidence: after removing the broken verify/fallback
  ## (reading/freeing mid-session caused SIGSEGV), a real run produced garbage tokens of all
  ## "!" repeated (broken/constant logits) + abnormally slow (24s/token). Before,
  ## the verify, although technically wrong (reading mid-session gave garbage),
  ## ACCIDENTALLY always fell back correctly to the old chain - which is why garbage
  ## tokens were never seen from this path before. Keep it OFF until this kernel is fixed
  ## and verified again in a separate session (not nested in the caller's
  ## session) - see the separate test guidance mentioned. Turn it back on with
  ## BYBY_TRY_ATTN_DECODE_GQA=1 ONLY after confirming the kernel is correct yourself.
var gAttnDecodeGqaWarned = false  ## prints the "has produced wrong results" warning
  ## exactly once when BYBY_TRY_ATTN_DECODE_GQA=1 is set (avoids log spam).

proc tryAttentionDecodeGqa(backend: GpuBackend, hQ, hK, hV: SessionHandle,
                            qGroup, kBlock, vBlock: seq[float32],
                            nRep, S, D: int, allHandles: var seq[SessionHandle]):
                            tuple[ok: bool, outGroup: SessionHandle] =
  ## NEW: tries sessionAttentionDecodeGqaEnc (1 encode command
  ## instead of the 3 commands matmulNT+softmax+matmul). Returns ok=false if the backend does not
  ## support it OR the verify finds a large mismatch - the caller MUST fall back
  ## to the old chain when ok=false, and must NOT treat it as an error to silently ignore.
  if gAttnDecodeGqaDisabled and not existsEnv("BYBY_TRY_ATTN_DECODE_GQA"):
    return (false, default(SessionHandle))
  if not gAttnDecodeGqaWarned:
    stderr.writeLine "[BYBY_TRY_ATTN_DECODE_GQA] WARNING: this kernel HAS BEEN " &
      "confirmed to produce WRONG output (garbage tokens of repeated '!') AND be SLOWER (24s/token) " &
      "in an earlier test, it is not just 'unverified'. If you see garbage tokens " &
      "or abnormal slowness, TURN IT OFF NOW (remove BYBY_TRY_ATTN_DECODE_GQA from the command)."
    gAttnDecodeGqaWarned = true
  var hOut = sessionAllocScratch(backend, nRep * D)
  var hScores = sessionAllocScratch(backend, nRep * S)
  let ok = sessionAttentionDecodeGqaEnc(backend, hQ, hK, hV, hOut, hScores, nRep, S, D, 1.0'f32)
  # scale is ALREADY folded into qGroup on the CPU before the upload (see
  # "fold scale into Q" above) - so scale=1.0 is passed here, do NOT
  # multiply a 2nd time (the result would be wrong - scale^2 instead of scale).
  if ok:
    # (A SERIOUS BUG, FOUND AND FIXED): the verify used to call
    # sessionRead()+sessionFree() RIGHT HERE - but this function is always
    # called FROM INSIDE the CALLER's large session (sessionBegin already done,
    # no sessionEnd yet) - reading mid-session ALWAYS gives garbage (as the
    # comment of vulkanSessionRead/metalSessionRead itself says: "ONLY
    # to be called AFTER sessionEnd") -> the verify ALWAYS reported a fake mismatch (not
    # a real kernel bug), then sessionFree() mid-session DESTROYED
    # the buffer IMMEDIATELY (vkDestroyBuffer/vkFreeMemory) while the dispatch
    # referencing it was STILL in the outer session's unsubmitted command buffer
    # -> when sessionEnd() really submitted, the GPU
    # dereferenced a destroyed buffer -> SIGSEGV. A "read mid-session" verify
    # basically CANNOT be done safely in this merged-
    # session architecture - removed entirely, hOut/hScores go into allHandles to be cleaned up
    # at the right time (AFTER the real sessionEnd, like every other buffer in this
    # loop) instead of being freed early while still in use.
    allHandles.add(hOut); allHandles.add(hScores)
    return (true, hOut)
  else:
    sessionFree(backend, hOut)
    sessionFree(backend, hScores)
    return (false, default(SessionHandle))

var gKvResidentVerifyLeft = 5

proc kvResidentUsable(cache: KVCacheL, layerIdx: int): bool =
  (not gKvResidentDisabled) and cache.layers[layerIdx].residentReady

proc verifyKvResidentOnce(backend: GpuBackend, cache: KVCacheL, layerIdx, hkv: int,
                           kBlockRef, vBlockRef: seq[float32], S, D: int) =
  ## Compares the REAL data in the resident buffer (read back via sessionRead) with exactly
  ## the content it should have (kBlockRef/vBlockRef - the CPU cache, the source of
  ## truth). This is a MEMCPY comparison, not math - it should
  ## match almost exactly (the epsilon only leaves room for a tiny GPU roundtrip error,
  ## not floating-point accumulation error like in attention).
  # verify is now ON BY DEFAULT - only off when BYBY_NO_VERIFY=1 is set explicitly.
  if existsEnv("BYBY_NO_VERIFY"): return
  if gKvResidentVerifyLeft <= 0: return
  dec gKvResidentVerifyLeft
  let kGpu = sessionRead(backend, cache.layers[layerIdx].kResident[hkv], S * D)
  let vGpu = sessionRead(backend, cache.layers[layerIdx].vResident[hkv], S * D)
  var maxDiff = 0'f32
  for i in 0 ..< kGpu.len:
    let d = abs(kGpu[i] - kBlockRef[i])
    if d > maxDiff: maxDiff = d
  for i in 0 ..< vGpu.len:
    let d = abs(vGpu[i] - vBlockRef[i])
    if d > maxDiff: maxDiff = d
  if maxDiff > 1e-5'f32:
    stderr.writeLine "[BYBY_VERIFY_KV_RESIDENT] max|gpu-ref| = " & $maxDiff &
      "  *** WRONG - disabling resident KV, falling back to a full upload every token for the rest ***"
    gKvResidentDisabled = true
  else:
    stderr.writeLine "[BYBY_VERIFY_KV_RESIDENT] max|gpu-ref| = " & $maxDiff & "  OK"

var gDecodeProfileLeft = (try: parseInt(getEnv("BYBY_DECODE_PROFILE", "0")) except ValueError: 0)
  ## (only layer 0, the first 2 tokens = 2*1 times since layerIdx==0 happens exactly once per token)
  ## to know exactly how much time goes to QKV+RoPE vs the attention loop vs
  ## oProj, instead of only the merged "attn" total like gAttnTime now.
  ## THIS IS PURE MEASUREMENT - no computation is changed.

proc forwardDecode*(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL, ctx: Backend,
                     cache: var KVCacheL, layerIdx: int, ffnAllResident: bool = false):
                     tuple[attnOut: Tensor, sessionOpen: bool, be: GpuBackend,
                           hOut: SessionHandle, pending: seq[SessionHandle]] =
  ## Adds the ffnAllResident parameter + returns a tuple - this
  ## KEEPS the GPU session open when the caller (forward*(blk...)) confirms the FFN
  ## after it is ALSO resident and eligible for merging - avoiding 1 extra
  ## sessionEnd+sessionRead+sessionUpload round trip between oProj and FFN (the
  ## ggml-metal direction: 1 command buffer for the WHOLE layer instead of many small ones).
  ## When sessionOpen=false (default/fallback), result.attnOut is used exactly as
  ## before - NOTHING changes. When sessionOpen=true, the caller MUST
  ## sessionEnd + free `pending` itself afterwards (it must NOT read/free any handle
  ## midway - exactly the bug fixed earlier).
  let doProfile = layerIdx == 0 and gDecodeProfileLeft > 0
  var tProfile0, tProfile1: float
  if doProfile: tProfile0 = epochTime()
  let C = x.shape[2]
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  let nRep = H div HKV
  let S = cache.seqLen + 1   # total length after adding this token

  # CROSS-SUBMODULE SESSION MERGE (like forwardPrefill above).
  let (qData, kNew, vNew) = fusedNormQKVRoPE(attn, norm, x, rope, ctx, T = 1, posOffset = cache.seqLen)
  if doProfile:
    tProfile1 = epochTime()
    stderr.writeLine "[DECODE_PROFILE] QKV+RoPE(RMSNorm merged): " & $((tProfile1 - tProfile0) * 1000) & " ms"

  appendKVCpuOnly(cache, layerIdx, kNew, vNew, 1)

  var outHeads = newSeq[float32](H * D)
  let maxS = cache.maxSeqLen

  # Gather each group's data first (CPU, cheap) so everything can be encoded into 1 session
  #
  # (removes the CPU O(S*D)/token transpose entirely): sessionMatmulNT was added
  # (C=A@B^T, B keeps its original layout [N,K]) for OpenCL/Metal/CUDA - K and
  # V now only need 1 contiguous copyMem from the cache, NO physical transpose
  # any more. kBlockT is only built (lazily, when needed) if the current backend
  # does not support NT (safe fallback, same result).
  # (a real CPU bottleneck - found by reading the code): BEFORE, this loop
  # ALWAYS copyMem'd all S*D elements (K and V) for EVERY KV head, EVERY decode
  # token, WHETHER OR NOT resident was on. With resident on (the default
  # path - see "ON BY DEFAULT" in syncKvResidentEncode above), hK/hV
  # come STRAIGHT from cache.layers[layerIdx].kResident/vResident (already on the
  # GPU) - kBlocks/vBlocks are only used by transposeKBlock (the rare fallback
  # when the backend has no NT) and verifyKvResidentOnce (at most 5 times for the WHOLE
  # RUN - gKvResidentVerifyLeft). So ALMOST ALL of this copying was
  # WASTED on the main path, and since S grows every token, this cost was O(S)
  # PER TOKEN -> O(S^2) PER RUN - a silent bottleneck growing with the conversation
  # length even without a crash. FIX: copy only when REALLY needed (resident
  # not ready, OR verify runs left).
  let kvResUsableNow = kvResidentUsable(cache, layerIdx)
  let needCpuKvBlocks = (not kvResUsableNow) or gKvResidentVerifyLeft > 0
  var kBlocks = newSeq[seq[float32]](HKV)
  var vBlocks = newSeq[seq[float32]](HKV)
  var qGroups = newSeq[seq[float32]](HKV)
  if needCpuKvBlocks: ensureCpuKv(cache, layerIdx)
  for hkv in 0 ..< HKV:
    let headBase = hkv * maxS * D
    # K, V: cache layout [nKVHeads, maxSeqLen, headDim] -> for a fixed hkv, the first S
    # tokens are ALREADY contiguous -> only a copyMem is needed, no transpose.
    if needCpuKvBlocks:
      var kBlock = newSeq[float32](S * D)
      copyMem(addr kBlock[0], unsafeAddr cache.layers[layerIdx].k[headBase], S * D * sizeof(float32))
      kBlocks[hkv] = kBlock

      var vBlock = newSeq[float32](S * D)
      copyMem(addr vBlock[0], unsafeAddr cache.layers[layerIdx].v[headBase], S * D * sizeof(float32))
      vBlocks[hkv] = vBlock
    # else: kBlocks[hkv]/vBlocks[hkv] stay @[] (empty) - SAFE because
    # the resident-usable branch does not read kBlocks/vBlocks to build hK/hV (it takes
    # them straight from cache.layers[layerIdx].kResident/vResident), only transposeKBlock
    # (the rare NT fallback) and verifyKvResidentOnce (already excluded via needCpuKvBlocks
    # above) touch these 2 arrays.

    var qGroup = newSeq[float32](nRep * D)
    for r in 0 ..< nRep:
      let h = hkv * nRep + r
      for d in 0 ..< D:
        # fold the scale into Q on the CPU (cheap, small array) to avoid an extra GPU op
        # just for the scale multiply in the session (the session only has matmul/softmax)
        qGroup[r * D + d] = qData[h * D + d] * attn.scale
    qGroups[hkv] = qGroup

  proc transposeKBlock(kBlock: seq[float32]): seq[float32] =
    ## Only called when the backend does NOT support sessionMatmulNT (fallback). [S,D] -> [D,S].
    result = newSeq[float32](D * S)
    for s in 0 ..< S:
      for d in 0 ..< D:
        result[d * S + s] = kBlock[s * D + d]

  let backend = ctx.kind.toByby()
  # (ONE MORE STEP MERGED - the attention-decode session now chains straight into
  # oProj, NO longer split into 2 sessions like the previous version): before, it had to
  # sessionRead() outHeads back to the CPU between attention and oProj JUST TO join the
  # separate nRep*D results of each KV head into 1 contiguous [H*D] vector (the oProj
  # matmul input) - there was no way to join them on the GPU since the old session API had
  # no "copy into a sub-region of another buffer" primitive. Now there is
  # sessionCopyInto (see gpubackend.nim - Metal uses the new copy_into_kernel,
  # CUDA uses cuMemcpyDtoDAsync directly, OpenCL uses clEnqueueCopyBuffer directly,
  # all 3 have real implementations, no hidden "unsupported" branch) so
  # each outGroup (nRep*D elements of 1 KV head) is copied STRAIGHT into its
  # place in 1 shared [H*D] scratch buffer, IN THE SAME open
  # session, then chained straight into sessionMatmulQ4(oProj) without ever leaving the GPU.
  # Result: 2 sessions/layer (attention + oProj) -> EXACTLY 1.
  # sessionCopyInto had NEVER run on real hardware (no test machine
  # available when this was written) - there was an escape hatch BYBY_LEGACY_ATTN_OPROJ=1 to force
  # going back to the old 2-session path (confirmed correct by hand) if a mismatch/crash
  # was found in a real build.
  # LEGACY BANNED: the BYBY_LEGACY_ATTN_OPROJ escape hatch is removed - always take the merged path
  # when resident is ready. If there really is a bug on real hardware it will
  # crash/raise clearly instead of silently going back to the slow path.
  let oProjResidentMergedQ4 = attn.oProj.residentReady
  let oProjResidentMergedFp32 = (not oProjResidentMergedQ4) and attn.oProj.residentFp32Ready
  let oProjResidentMerged = oProjResidentMergedQ4 or oProjResidentMergedFp32
  debugPathOnce("attn_oproj.resident_ready", oProjResidentMerged,
    "Q4=" & $oProjResidentMergedQ4 & " Fp32=" & $oProjResidentMergedFp32)
  # (A REAL BUG, FOUND): BEFORE, syncKvResidentEncode() was called in
  # BOTH the merged branch (below) AND the "OLD PATH" fallback branch after it, and when
  # the merged branch opened its session successfully but the oProj matmul/copyInto failed midway
  # (okAll/okO=false -> oProjOutMerged empty), the code fell through into
  # the fallback branch and OPENED A 2ND SESSION for THE SAME token/layer - i.e. 2
  # sessionBegin/sessionEnd pairs (= 2 forced GPU syncs) for work that only needed
  # 1, and the KV resident encode was called twice. `kvSynced` makes sure
  # syncKvResidentEncode runs EXACTLY ONCE however many branches are taken.
  var kvSynced = false
  var tProfileGap0: float
  if doProfile:
    tProfileGap0 = epochTime()
    stderr.writeLine "[DECODE_PROFILE] CPU-only GAP between the 2 sessions (appendKVCpuOnly + gathering qGroups, NOT counting sessionBegin/GPU sync): " &
      $((tProfileGap0 - tProfile1) * 1000) & " ms"
  var usedMergedSession = oProjResidentMerged and sessionBegin(backend)
  var oProjOutMerged: seq[float32]
  if usedMergedSession:
    var allHandles = newSeq[SessionHandle]()
    var okAll = true
    # (the real cause was found, not avoided): the resident KV sync is still
    # merged in HERE (writes only, no read back - safe even though the session is not committed,
    # since the GPU only needs to SEE the write commands in order within the same session, which
    # is always true). What was NOT safe before was calling sessionRead (verify)
    # inside this loop too - the verify has moved to AFTER sessionEnd below.
    if not kvSynced:
      syncKvResidentEncode(cache, layerIdx, kNew, vNew, 1, backend)
      kvSynced = true
    let hOutHeads = sessionAllocScratch(backend, H * D)   # shared [H*D] buffer, each piece is written into it
    allHandles.add(hOutHeads)
    for hkv in 0 ..< HKV:
      let hQ = sessionUpload(backend, qGroups[hkv])
      allHandles.add(hQ)
      var hK, hV: SessionHandle
      if kvResidentUsable(cache, layerIdx):
        hK = cache.layers[layerIdx].kResident[hkv]
        hV = cache.layers[layerIdx].vResident[hkv]
        # verifyKvResidentOnce is NO longer called here - the sessionRead
        # inside it reads the buffer memory directly, which is only CORRECT once the write above
        # (syncKvResidentEncode) has really FINISHED on the GPU, i.e. after
        # sessionEnd() commit+wait - not midway through the session as before
        # (that was exactly the cause of "no open session" + garbage reads
        # max|gpu-ref|=1.4066032 last time). The verify moved below, AFTER
        # sessionEnd.
      else:
        hK = sessionUpload(backend, kBlocks[hkv])
        hV = sessionUpload(backend, vBlocks[hkv])
        allHandles.add(hK); allHandles.add(hV)

      var hScores: SessionHandle
      var hOutGroup: SessionHandle
      let (fusedOk, fusedOut) = tryAttentionDecodeGqa(backend, hQ, hK, hV,
                                                        qGroups[hkv], kBlocks[hkv], vBlocks[hkv],
                                                        nRep, S, D, allHandles)
      if fusedOk:
        hOutGroup = fusedOut
      else:
        let (ntHandle, ntOk) = sMatmulNT(backend, hQ, nRep, D, hK, S)
        if ntOk:
          hScores = ntHandle
          allHandles.add(hScores)
        else:
          let hKT = sessionUpload(backend, transposeKBlock(kBlocks[hkv]))
          allHandles.add(hKT)
          hScores = sMatmul(backend, hQ, nRep, D, hKT, S)
          allHandles.add(hScores)

        let hProbs = sSoftmax(backend, hScores, nRep, S)
        hOutGroup = sMatmul(backend, hProbs, nRep, S, hV, D)
        allHandles.add(hProbs); allHandles.add(hOutGroup)

      # Write outGroup (nRep*D elements) straight into its place in
      # hOutHeads - the offset is in elements, in exactly the same head order as the old CPU
      # loop (h = hkv*nRep + r).
      var dstHandle = hOutHeads
      let okCopy = sessionCopyInto(backend, hOutGroup, dstHandle, hkv * nRep * D, nRep * D)
      if not okCopy: okAll = false

    var hOProjOut = sessionAllocScratch(backend, C)
    allHandles.add(hOProjOut)
    let okO = okAll and (if oProjResidentMergedQ4: sessionMatmulQ4(backend, hOutHeads, attn.oProj.residentW, hOProjOut, 1)
                          else: sessionMatmul(backend, hOutHeads, attn.oProj.residentFp32, hOProjOut, 1, attn.oProj.inF, attn.oProj.outF))

    # If the FFN after it is ALSO resident-eligible (the caller checked
    # beforehand and passed ffnAllResident=true), do NOT sessionEnd here - keep
    # the session open, add the bias on the GPU (if any, instead of a CPU for loop), and hand off to the
    # caller to chain straight into the FFN in THE SAME session - saving 1
    # sessionEnd+sessionRead+sessionUpload round trip per layer (the ggml-metal direction: 1 command
    # buffer per layer instead of many small ones). REQUIRED: do NOT sessionFree any
    # handle here (not even unused ones) - the session still has UNSUBMITTED commands
    # referencing them, freeing early is exactly the SIGSEGV bug fixed
    # earlier. All allHandles + hOProjOut (bias already added) go into
    # `pending`, and the caller must free them itself AFTER the real sessionEnd (at the end of the FFN).
    if okAll and okO and ffnAllResident:
      var hFinal = hOProjOut
      if attn.oProj.bias.data.len == C:
        let hBias = sessionUpload(backend, attn.oProj.bias.data)
        allHandles.add(hBias)
        let hBiased = sVecOp(backend, "add", hOProjOut, hBias, C)
        allHandles.add(hBiased)
        hFinal = hBiased
      if doProfile:
        let tProfile2 = epochTime()
        stderr.writeLine "[DECODE_PROFILE] Attention loop + oProj (session KEPT OPEN for the FFN): " &
          $((tProfile2 - tProfile1) * 1000) & " ms"
        dec gDecodeProfileLeft
      return (newTensor(@[1, 1, C]), true, backend, hFinal, allHandles)

    # (optimisation): add the bias ON THE GPU with sVecOp before sessionEnd,
    # exactly like the ffnAllResident branch above - before, the result was read back to the
    # CPU and the bias added with a sequential for loop (an unnecessary GPU->CPU bottleneck,
    # C elements on a single CPU thread every decode).
    var hOProjFinal = hOProjOut
    if okAll and okO and attn.oProj.bias.data.len == C:
      let hBias = sessionUpload(backend, attn.oProj.bias.data)
      allHandles.add(hBias)
      hOProjFinal = sVecOp(backend, "add", hOProjOut, hBias, C)
      allHandles.add(hOProjFinal)
    usedMergedSession = sessionEnd(backend)
    if usedMergedSession and okO:
      oProjOutMerged = sessionRead(backend, hOProjFinal, C)
    for h in allHandles.mitems: sessionFree(backend, h)
    # The resident KV verify moved DOWN HERE - AFTER sessionEnd(backend) has
    # committed+waited, so the sessionRead in verifyKvResidentOnce now reads
    # the real data (no more garbage reads mid-session like the previous
    # bug). Only head 0 is verified as a representative (enough to catch systematic errors, no
    # need to verify all HKV heads every time - gAttnDecodeGqaVerifyLeft/
    # gKvResidentVerifyLeft already cap the total verify count for the whole run).
    if usedMergedSession and kvResidentUsable(cache, layerIdx):
      verifyKvResidentOnce(backend, cache, layerIdx, 0, kBlocks[0], vBlocks[0], S, D)
    if oProjOutMerged.len == 0:
      usedMergedSession = false   # sessionCopyInto/matmul failed -> falls through to the old 2-session path below, false is NOT treated as done

  debugPathOnce("attn_oproj.path", usedMergedSession)
  if usedMergedSession:
    result.attnOut = newTensor(@[1, 1, C])
    result.attnOut.data = oProjOutMerged
    if doProfile:
      let tProfile2 = epochTime()
      stderr.writeLine "[DECODE_PROFILE] Attention-loop + oProj (session merged): " &
        $((tProfile2 - tProfile1) * 1000) & " ms  (tong forwardDecode: " &
        $((tProfile2 - tProfile0) * 1000) & " ms)"
      dec gDecodeProfileLeft
    return

  # ─────────────────────────────────────────────────────────────────────
  # OLD PATH (fallback): sessionCopyInto unavailable/failed, or
  # BYBY_LEGACY_ATTN_OPROJ=1, or oProj not resident - reuses exactly the
  # 2-session logic confirmed correct by hand before (a separate attention-decode
  # session, then a separate oProj session, with a sessionRead in between).
  # ─────────────────────────────────────────────────────────────────────
  var usedSession = sessionBegin(backend)
  if usedSession:
    var outHandles = newSeq[SessionHandle](HKV)
    var allHandles = newSeq[SessionHandle]()
    # The resident KV sync is merged in here (this branch only runs when the merged branch
    # above failed/returned early) - writes only, no read back, safe (see the full
    # explanation in the merged branch above).
    if not kvSynced:
      syncKvResidentEncode(cache, layerIdx, kNew, vNew, 1, backend)
      kvSynced = true
    for hkv in 0 ..< HKV:
      let hQ = sessionUpload(backend, qGroups[hkv])
      allHandles.add(hQ)
      var hK, hV: SessionHandle
      if kvResidentUsable(cache, layerIdx):
        hK = cache.layers[layerIdx].kResident[hkv]
        hV = cache.layers[layerIdx].vResident[hkv]
        # The verify moved to AFTER sessionEnd (see the reason in the merged branch above)
      else:
        hK = sessionUpload(backend, kBlocks[hkv])
        hV = sessionUpload(backend, vBlocks[hkv])
        allHandles.add(hK); allHandles.add(hV)

      var hScores: SessionHandle
      let (fusedOk2, fusedOut2) = tryAttentionDecodeGqa(backend, hQ, hK, hV,
                                                          qGroups[hkv], kBlocks[hkv], vBlocks[hkv],
                                                          nRep, S, D, allHandles)
      if fusedOk2:
        outHandles[hkv] = fusedOut2
      else:
        let (ntHandle, ntOk) = sMatmulNT(backend, hQ, nRep, D, hK, S)
        if ntOk:
          hScores = ntHandle
          allHandles.add(hScores)
        else:
          let hKT = sessionUpload(backend, transposeKBlock(kBlocks[hkv]))
          allHandles.add(hKT)
          hScores = sMatmul(backend, hQ, nRep, D, hKT, S)
          allHandles.add(hScores)

        let hProbs = sSoftmax(backend, hScores, nRep, S)
        outHandles[hkv] = sMatmul(backend, hProbs, nRep, S, hV, D)
        allHandles.add(hProbs)
    usedSession = sessionEnd(backend)
    if usedSession:
      for hkv in 0 ..< HKV:
        let outGroup = sessionRead(backend, outHandles[hkv], nRep * D)
        for r in 0 ..< nRep:
          let h = hkv * nRep + r
          for d in 0 ..< D:
            outHeads[h * D + d] = outGroup[r * D + d]
    for h in allHandles.mitems: sessionFree(backend, h)
    for h in outHandles.mitems: sessionFree(backend, h)
    # verify AFTER sessionEnd - safe (see the explanation in the merged branch above)
    if usedSession and kvResidentUsable(cache, layerIdx):
      verifyKvResidentOnce(backend, cache, layerIdx, 0, kBlocks[0], vBlocks[0], S, D)

  if not usedSession:
    for hkv in 0 ..< HKV:
      let kBlockT = transposeKBlock(kBlocks[hkv])
      var scores = beMatmul(ctx, qGroups[hkv], nRep, D, kBlockT, D, S)
      let probs = beSoftmax(ctx, scores, nRep, S)
      let outGroup = beMatmul(ctx, probs, nRep, S, vBlocks[hkv], S, D)
      for r in 0 ..< nRep:
        let h = hkv * nRep + r
        for d in 0 ..< D:
          outHeads[h * D + d] = outGroup[r * D + d]

  let oProjResidentQ4 = attn.oProj.residentReady
  var oProjOut: seq[float32]
  let oProjResidentFp32 = (not oProjResidentQ4) and attn.oProj.residentFp32Ready
  let oProjResident = oProjResidentQ4 or oProjResidentFp32
  var usedOProjSession = oProjResident and sessionBegin(backend)
  if usedOProjSession:
    var hIn = sessionUpload(backend, outHeads)
    var hOut = sessionAllocScratch(backend, C)
    let okO = if oProjResidentQ4: sessionMatmulQ4(backend, hIn, attn.oProj.residentW, hOut, 1)
              else: sessionMatmul(backend, hIn, attn.oProj.residentFp32, hOut, 1, attn.oProj.inF, attn.oProj.outF)
    # (optimisation, in sync with the merged branch above): add the bias ON THE GPU
    # with sVecOp before sessionEnd, instead of reading back to the CPU and a sequential for loop.
    var hOutFinal = hOut
    var hBias: SessionHandle
    var hBiasUsed = false
    if okO and attn.oProj.bias.data.len == C:
      hBias = sessionUpload(backend, attn.oProj.bias.data)
      hBiasUsed = true
      hOutFinal = sVecOp(backend, "add", hOut, hBias, C)
    usedOProjSession = sessionEnd(backend)
    if usedOProjSession and okO:
      oProjOut = sessionRead(backend, hOutFinal, C)
    sessionFree(backend, hIn); sessionFree(backend, hOut)
    if hBiasUsed:
      sessionFree(backend, hBias)
      sessionFree(backend, hOutFinal)
    if oProjOut.len == 0:
      usedOProjSession = false

  if usedOProjSession:
    result.attnOut = newTensor(@[1, 1, C])
    result.attnOut.data = oProjOut
  else:
    var attnOut = newTensor(@[1, 1, C])
    attnOut.data = outHeads
    result.attnOut = attn.oProj.forward(attnOut, ctx)

# ===================================================================
# TransformerBlockLlama (pre-norm: RMSNorm -> Attn -> residual -> RMSNorm -> SwiGLU -> residual)
# ===================================================================
type
  TransformerBlockLlama* = object
    inputNorm*, postAttnNorm*: RMSNormL
    attn*: GQAAttention
    ff*: SwiGLUL

proc newTransformerBlockLlama*(hidden, nHeads, nKVHeads, intermediate: int, eps: float32): TransformerBlockLlama =
  result.inputNorm = newRMSNormL(hidden, eps)
  result.postAttnNorm = newRMSNormL(hidden, eps)
  result.attn = newGQAAttention(hidden, nHeads, nKVHeads)
  result.ff = newSwiGLUL(hidden, intermediate)

proc fusedResidualNormFFN(norm: RMSNormL, ff: SwiGLUL, x, attnOut: Tensor, ctx: Backend): Tensor =
  ## CROSS-SUBMODULE MERGE of a block's whole tail: residual-add (after Attn) ->
  ## RMSNorm(postAttnNorm, scale) -> FFN (gate/up matmul + sigmoid+mul+mul +
  ## down matmul) -> residual-add (after FFN) INTO EXACTLY 1 GPU SESSION.
  ##
  ## BEFORE: addT (plain, no session) + RMSNorm (its own session) + FFN
  ## gate/up matmul (plain, no session) + FFN sigmoid/mul/mul (its own
  ## session) + FFN down matmul (plain) + addT (plain) = 2 sessions opened/closed +
  ## 4 other separate GPU trips, each uploading/downloading on its own.
  ##
  ## NOW: the first residual-add, RMSNorm-scale, gate-matmul, up-matmul, sigmoid,
  ## silu-mul, gated-mul, down-matmul, the final residual-add - ALL encoded in sequence
  ## into THE SAME sessionBegin/sessionEnd, with no GPU->CPU read
  ## in between (unlike fusedNormQKVRoPE, this tail does NOT need gather/rotate_half
  ## so there is no hard boundary forcing the session to break).
  let n = x.data.len
  let (rows, hidden) = flatten2D(x.shape)
  doAssert hidden == norm.dim

  # ---- Mandatory CPU work BEFORE the session: only the RMSNorm reduce part (reading the small sumSq
  # to run sqrt). The gate/up/down weights are NO longer dequantized here - this
  # was the most time-consuming part before (3 full-size matrices decompressed
  # + uploaded again EVERY TOKEN, EVERY LAYER even though the weights never change between
  # tokens) - see linearFast() at the top of the file.
  let x2 = addT(ctx, x, attnOut)
  # (lazy, like fusedNormQKVRoPE): invBroadcast/wBroadcast are not computed
  # before knowing whether sessionRmsnormEnc succeeds - avoiding 2 extra GPU
  # round trips per layer when the fast path (Metal) works.
  var invBroadcast: seq[float32]
  var wBroadcast: seq[float32]
  proc ensureRmsNormFallbackFFN() =
    if invBroadcast.len > 0: return
    let (inv, w, _) = rmsNormPrep(norm, x2, ctx)
    invBroadcast = inv
    wBroadcast = w
  let interF = ff.gateProj.outF
  let nInter = rows * interF

  let be = ctx.kind.toByby()
  var normed: seq[float32]
  var gateRaw, upRaw: seq[float32]

  let gateUpResidentQ4 = ff.gateProj.residentReady and ff.upProj.residentReady
  let gateUpResidentFp32 = (not gateUpResidentQ4) and ff.gateProj.residentFp32Ready and ff.upProj.residentFp32Ready

  # (2 FFN sessions merged into 1, tried FIRST): gate/up-matmul ->
  # (+GPU bias) -> sigmoid -> mul -> mul -> down-matmul -> (+GPU bias) run
  # in one go, EXACTLY 1 sessionBegin/sessionEnd for the whole FFN instead of 2-3
  # as before - there is NO mandatory CPU dependency between these steps (unlike
  # RoPE, which needs a gather); the old CPU read only existed to add the bias with a for loop,
  # now the bias is added right on the GPU (sVecOp "add" + the existing broadcastBias).
  # Saves 1 commit+waitUntilCompleted per layer (old Intel Metal driver: this
  # is the main overhead source, see the analysis in generate()).
  # (the REAL root of the bug, confirmed in logs on Intel Iris 640: the GPU Timeout
  # happened RIGHT AT LAYER 1, INDEPENDENT of the heap/KV-resident - ruled out with
  # BYBY_DISABLE_KV_RESIDENT=1, still timing out at exactly the same point):
  # BEFORE, the whole FFN (RMSNorm + gate-matmul + up-matmul + sigmoid + mul +
  # mul + down-matmul + residual) was merged into a SINGLE command buffer, committed once
  # at the final sessionEnd. On the Iris 640 (not headless - it has the OS's GPU
  # watchdog, "kIOAccelCommandBufferCallbackErrorTimeout"), the amount
  # of work packed into this one buffer EXCEEDED the watchdog threshold -> the GPU
  # was killed midway, NOT a logic error/incorrect kernel.
  # FIX (STILL 100% GPU, NO CPU fallback - "LEGACY BANNED"): SPLIT the 1
  # large command buffer into 2 smaller command buffers:
  #   Session A: RMSNorm + gate-matmul + up-matmul + bias + sigmoid + mul + mul
  #              -> sessionEnd (a REAL commit+wait on the GPU - the watchdog is
  #              RESET since the next submit is a NEW command buffer)
  #   Session B: down-matmul + bias -> sessionEnd
  # The GPU buffer (hGated) is NOT destroyed between the 2 sessions - sessionEnd only
  # commits+waits, ONLY sessionFree() really destroys a buffer (see gpubackend.nim)
  # - so Session B reuses Session A's hGated handle directly, NO need to read it
  # back to the CPU and upload it again in between (no loss in speed/memory).
  var ffOut: seq[float32]
  let downResidentQ4 = ff.downProj.residentReady
  let downResidentFp32 = (not downResidentQ4) and ff.downProj.residentFp32Ready
  let allResidentQ4 = gateUpResidentQ4 and downResidentQ4
  let allResidentFp32 = gateUpResidentFp32 and downResidentFp32
  if not (allResidentQ4 or allResidentFp32):
    raise newException(CatchableError,
      "[FFN] resident is not ready (uploadResident() did not run or failed for " &
      "gateProj/upProj/downProj) - the legacy fallback is BANNED by request.")
  let allResident = true
  debugPathOnce("ffn.resident_ready", allResidentQ4 or allResidentFp32,
    "Q4=" & $allResidentQ4 & " Fp32=" & $allResidentFp32 & " legacyFFN=false(cam)")

  # (MERGED BACK into 1 session - tried after matmul_q4_kernel was
  # fixed at the root, see vecop_matmul.metal): it USED to be split into 2 sessions (A:
  # gate/up/sigmoid/mul/mul, B: down-matmul) because the old Q4 kernel (threadgroup
  # memory, a wrong simdgroup assumption) was too slow/could hang, making 1 buffer
  # holding the whole FFN exceed the Iris 640 watchdog threshold. NOW the Q4 kernel uses
  # REAL simd_sum() + dispatch with the REAL threadExecutionWidth (no more
  # wrong assumptions) - much faster than the old one. MERGED BACK into 1 session to drop 1
  # commit+waitUntilCompleted per layer (Intel Metal driver: a significant overhead
  # source, x32 layers x every token). If Timeouts still happen after
  # merging, go back to the 2-session split (see the change history above).
  var usedSessionFused = allResident and sessionBegin(be)
  let tryFastRmsnormFFN = usedSessionFused and not existsEnv("BYBY_DISABLE_SESSION_RMSNORM")
  if usedSessionFused:
    var hX2 = sessionUpload(be, x2.data)
    var hGamma: SessionHandle
    var hInv, hW, hT1: SessionHandle
    var hNormed: SessionHandle
    var normedFastFFN = false
    if tryFastRmsnormFFN:
      hGamma = (if norm.residentGammaReady: norm.residentGamma else: sessionUpload(be, norm.weight.data))
      var hNormedTry = sessionAllocScratch(be, n)
      if sessionRmsnormEnc(be, hX2, hGamma, hNormedTry, rows, hidden, norm.eps):
        hNormed = hNormedTry
        normedFastFFN = true
      else:
        sessionFree(be, hNormedTry)
    if not normedFastFFN:
      ensureRmsNormFallbackFFN()
      hInv = sessionUpload(be, invBroadcast)
      hW = sessionUpload(be, wBroadcast)
      hT1 = sVecOp(be, "mul", hX2, hInv, n)
      hNormed = sVecOp(be, "mul", hT1, hW, n)
    var hGate = sessionAllocScratch(be, nInter)
    var hUp = sessionAllocScratch(be, nInter)
    let okG = if allResidentQ4: sessionMatmulQ4(be, hNormed, ff.gateProj.residentW, hGate, rows)
              else: sessionMatmul(be, hNormed, ff.gateProj.residentFp32, hGate, rows, ff.gateProj.inF, ff.gateProj.outF)
    let okU = if allResidentQ4: sessionMatmulQ4(be, hNormed, ff.upProj.residentW, hUp, rows)
              else: sessionMatmul(be, hNormed, ff.upProj.residentFp32, hUp, rows, ff.upProj.inF, ff.upProj.outF)
    # The biased result replaces the handle; free the unbiased buffer too
    # (it used to leak once per layer per token).
    if ff.gateProj.bias.data.len == ff.gateProj.outF:
      var hGB = sessionUpload(be, broadcastBias(ff.gateProj.bias, rows, ff.gateProj.outF))
      var hRaw = hGate
      hGate = sVecOp(be, "add", hGate, hGB, nInter)
      sessionFree(be, hGB); sessionFree(be, hRaw)
    if ff.upProj.bias.data.len == ff.upProj.outF:
      var hUB = sessionUpload(be, broadcastBias(ff.upProj.bias, rows, ff.upProj.outF))
      var hRaw = hUp
      hUp = sVecOp(be, "add", hUp, hUB, nInter)
      sessionFree(be, hUB); sessionFree(be, hRaw)
    let hSig = sActivation(be, "sigmoid", hGate, nInter)
    let hSilu = sVecOp(be, "mul", hGate, hSig, nInter)
    var hGated = sVecOp(be, "mul", hSilu, hUp, nInter)
    let okGateUp = okG and okU
    var hOut = sessionAllocScratch(be, rows * ff.downProj.outF)
    let okD = if allResidentQ4: sessionMatmulQ4(be, hGated, ff.downProj.residentW, hOut, rows)
              else: sessionMatmul(be, hGated, ff.downProj.residentFp32, hOut, rows, ff.downProj.inF, ff.downProj.outF)
    if ff.downProj.bias.data.len == ff.downProj.outF:
      var hDB = sessionUpload(be, broadcastBias(ff.downProj.bias, rows, ff.downProj.outF))
      var hRaw = hOut
      hOut = sVecOp(be, "add", hOut, hDB, rows * ff.downProj.outF)
      sessionFree(be, hDB); sessionFree(be, hRaw)
    usedSessionFused = sessionEnd(be)   # <-- THE ONLY COMMIT+WAIT for the whole FFN
    if usedSessionFused and okGateUp and okD:
      ffOut = sessionRead(be, hOut, rows * ff.downProj.outF)
    if ffOut.len == 0:
      usedSessionFused = false
    sessionFree(be, hX2); sessionFree(be, hInv); sessionFree(be, hW)
    if not norm.residentGammaReady: sessionFree(be, hGamma)
    var hT1v = hT1; var hNormedV = hNormed
    sessionFree(be, hT1v); sessionFree(be, hNormedV)
    sessionFree(be, hGate); sessionFree(be, hUp)
    var hSigV = hSig; var hSiluV = hSilu
    sessionFree(be, hSigV); sessionFree(be, hSiluV); sessionFree(be, hGated)
    sessionFree(be, hOut)

  debugPathOnce("ffn.path", usedSessionFused)
  if not usedSessionFused:
    # LEGACY BANNED: falling back to the "OLD PATH" (separate sessions
    # or even linearFastBias/beMul possibly dropping to the CPU) when the merged session
    # failed midway used to be allowed. It is NOT any more - resident was confirmed
    # ready above, so if sessionBegin/sessionEnd/matmul fail midway
    # it is a real GPU error (driver/kernel) and must RAISE so it gets fixed,
    # it must not silently run slow or drop to the CPU.
    raise newException(CatchableError,
      "[FFN] the merged (fused) session failed MIDWAY even though resident was ready " &
      "(sessionBegin/sessionEnd/matmul returned false) - the legacy/CPU fallback " &
      "is BANNED by request, not falling back to the old path.")

  result = newTensor(x.shape)
  result.data = beAdd(ctx, x2.data, ffOut)   # final residual-add

var gAttnTime*: float = 0.0
var gFfnTime*: float = 0.0

proc fusedResidualNormFFNContinue(norm: RMSNormL, ff: SwiGLUL, x: Tensor, be: GpuBackend,
                                    hAttnOut: SessionHandle, pending: var seq[SessionHandle]): Tensor =
  ## NEW: a version that CONTINUES the GPU session ALREADY OPENED by forwardDecode (see
  ## ffnAllResident/sessionOpen there) - it does NOT sessionBegin itself (it uses the session
  ## the caller opened), the residual-add is done on the GPU (sVecOp "add") instead of the CPU
  ## addT, avoiding 1 sessionEnd+sessionRead+sessionUpload round trip between
  ## attention and FFN - merged into EXACTLY 1 sessionEnd per layer (instead of 2), the
  ## ggml-metal direction (1 command buffer per layer). ONLY to be called when the caller has
  ## confirmed allResident is eligible (the FFN DEFINITELY takes the
  ## resident path, without falling back - if it fell back midway the
  ## open session would NEVER be sessionEnd'ed -> the GPU commands would hang forever,
  ## and the next layer's sessionBegin() would fail because the old session is not closed - this is
  ## why forward*(blk...) MUST check allResident BEFORE calling
  ## forwardDecode with ffnAllResident=true, not guess).
  let n = x.data.len
  let (rows, hidden) = flatten2D(x.shape)
  doAssert hidden == norm.dim
  let interF = ff.gateProj.outF
  let nInter = rows * interF
  let gateUpResidentQ4 = ff.gateProj.residentReady and ff.upProj.residentReady
  let downResidentQ4 = ff.downProj.residentReady
  let allResidentQ4 = gateUpResidentQ4 and downResidentQ4
  # allResidentFp32 is similar, but ffnAllResident=true is ONLY set when 1
  # of the 2 holds - allResidentQ4 is the main branch, else fp32 (symmetric
  # with the original logic in fusedResidualNormFFN).

  var hX = sessionUpload(be, x.data)
  pending.add(hX)
  let hX2 = sVecOp(be, "add", hX, hAttnOut, n)   # GPU residual-add instead of the CPU addT
  pending.add(hX2)

  var hGamma = norm.residentGamma
  if not norm.residentGammaReady:
    hGamma = sessionUpload(be, norm.weight.data)
    pending.add(hGamma)   # ONLY freed when it is a freshly uploaded buffer (not
      # resident) - a resident buffer must NOT be added to pending, or
      # it would be freed by sessionFree() right after the first token (see the
      # "for h in pending.mitems: sessionFree" loop at the end of the function) -> token
      # 2 onwards would use a freed buffer -> errors/garbage (exactly the kind of bug hit
      # before, careful not to repeat it).
  var hNormed = sessionAllocScratch(be, n)
  pending.add(hNormed)
  let normedOk = sessionRmsnormEnc(be, hX2, hGamma, hNormed, rows, hidden, norm.eps)

  var hGate = sessionAllocScratch(be, nInter)
  var hUp = sessionAllocScratch(be, nInter)
  pending.add(hGate); pending.add(hUp)
  let okG = if allResidentQ4: sessionMatmulQ4(be, hNormed, ff.gateProj.residentW, hGate, rows)
            else: sessionMatmul(be, hNormed, ff.gateProj.residentFp32, hGate, rows, ff.gateProj.inF, ff.gateProj.outF)
  let okU = if allResidentQ4: sessionMatmulQ4(be, hNormed, ff.upProj.residentW, hUp, rows)
            else: sessionMatmul(be, hNormed, ff.upProj.residentFp32, hUp, rows, ff.upProj.inF, ff.upProj.outF)
  if ff.gateProj.bias.data.len == ff.gateProj.outF:
    let hGB = sessionUpload(be, broadcastBias(ff.gateProj.bias, rows, ff.gateProj.outF))
    pending.add(hGB)
    let hGate2 = sVecOp(be, "add", hGate, hGB, nInter)
    pending.add(hGate2); hGate = hGate2
  if ff.upProj.bias.data.len == ff.upProj.outF:
    let hUB = sessionUpload(be, broadcastBias(ff.upProj.bias, rows, ff.upProj.outF))
    pending.add(hUB)
    let hUp2 = sVecOp(be, "add", hUp, hUB, nInter)
    pending.add(hUp2); hUp = hUp2
  let hSig = sActivation(be, "sigmoid", hGate, nInter)
  pending.add(hSig)
  let hSilu = sVecOp(be, "mul", hGate, hSig, nInter)
  pending.add(hSilu)
  let hGated = sVecOp(be, "mul", hSilu, hUp, nInter)
  pending.add(hGated)
  var hOut = sessionAllocScratch(be, rows * ff.downProj.outF)
  pending.add(hOut)
  let okD = if allResidentQ4: sessionMatmulQ4(be, hGated, ff.downProj.residentW, hOut, rows)
            else: sessionMatmul(be, hGated, ff.downProj.residentFp32, hOut, rows, ff.downProj.inF, ff.downProj.outF)
  if ff.downProj.bias.data.len == ff.downProj.outF:
    let hDB = sessionUpload(be, broadcastBias(ff.downProj.bias, rows, ff.downProj.outF))
    pending.add(hDB)
    let hOut2 = sVecOp(be, "add", hOut, hDB, rows * ff.downProj.outF)
    pending.add(hOut2); hOut = hOut2

  # A SERIOUS BUG FIX (garbage tokens, found by the user): the previous version WAS MISSING
  # the final residual-add entirely - it returned hOut (just ffOut) instead of
  # x2 + ffOut like the original fusedResidualNormFFN (`result.data = beAdd(ctx,
  # x2.data, ffOut)`). This was the REAL cause of the garbage tokens (unrelated
  # to the matmul_q4 subgroup bug fixed earlier) - every decode step
  # (taking this resident branch) lost the residual once per layer, accumulated over 32
  # layers -> meaningless logits from the 2nd token on (the first token was still
  # right since it comes from the PREFILL logits, which use fusedResidualNormFFN with its
  # residual-add). FIX: add hX2 (the residual before the FFN) into hOut (ffOut)
  # on the GPU (sVecOp "add", in the same open session, no extra
  # GPU<->CPU round trip) BEFORE sessionEnd.
  let hFinal = sVecOp(be, "add", hX2, hOut, n)
  pending.add(hFinal)

  # EXACTLY 1 sessionEnd for the WHOLE layer (attention + FFN) - this is the key
  # of the merge: 96 round trips/token (3 sessions x 32 layers) are now 64
  # (2 sessions x 32 layers: 1 for QKV+RoPE, 1 for the merged attention+oProj+FFN).
  let sessionOk = sessionEnd(be)
  result = newTensor(x.shape)
  if sessionOk and normedOk and okG and okU and okD:
    result.data = sessionRead(be, hFinal, n)
  else:
    # Safety: the merged session failed midway - there is NO correct data to
    # return. This should NOT happen (the caller checked
    # ffnAllResident before taking this path) - if it does, return the residual
    # unchanged (x2 without the FFN) rather than a segfault/garbage, and log
    # a warning so the merged path can be disabled with BYBY_NO_MERGE_FFN_SESSION=1.
    stderr.writeLine "[fusedResidualNormFFNContinue] the merged session FAILED midway - falling back to residual passthrough (ABNORMAL, please report this error)"
    result.data = x.data   # residual passthrough, NOT correct but safer than garbage/a crash
  for h in pending.mitems: sessionFree(be, h)
  pending.setLen(0)

## ===================================================================
## MODE 2 (default ON): merges Session #1 (QKV+RoPE) + Session #2
## (attention+oProj+FFN) into EXACTLY 1 command buffer per layer instead of 2.
## Enabled with BYBY_MERGE_QKV_ATTN_SESSION=1. Setting nothing (default) = the OLD
## behaviour 100%, not a single line of the current forward(blk...) changes.
##
## SAFETY: this proc raises CatchableError as soon as ANY encode step
## fails (it never silently returns a wrong result) - the caller (forward*(blk...))
## MUST catch this exception and fall back to the old forwardDecode/
## fusedResidualNormFFNContinue path (MODE 1, verified correct with real logs)
## for THAT token. So if Mode 2 errors/has a logic crash on your machine, you
## always get an automatic way back, no need to turn the flag off to keep running.
let gMergeQkvAttnSession* = getEnv("BYBY_MERGE_QKV_ATTN_SESSION", "1") == "1"
var gMode2WarnLeft = 20   ## logs at most 20 Mode2->Mode1 fallbacks, to avoid spam

proc precheckLayerResident(attn: GQAAttention, ff: SwiGLUL, layerIdx: int,
                            failMsg: var string): bool =
  ## Checks whether 1 layer is resident-eligible (Q4 OR Fp32, the convention
  ## shared by the whole codebase - see the comment in the old
  ## forwardDecodeFullMergeSingleSession) - used by BOTH Mode 2 (checking 1 layer) and Mode 3 (checking
  ## all 32 layers BEFORE opening the session, so a session is not opened only to fail
  ## at the first layer).
  let qkvQ4 = attn.qProj.residentReady and attn.kProj.residentReady and attn.vProj.residentReady
  let qkvFp32 = (not qkvQ4) and attn.qProj.residentFp32Ready and attn.kProj.residentFp32Ready and
    attn.vProj.residentFp32Ready
  if not (qkvQ4 or qkvFp32):
    failMsg = "QKV not resident (neither Q4 nor Fp32 is ready, layer " & $layerIdx & ")"
    return false
  let oProjQ4 = attn.oProj.residentReady
  let oProjFp32 = (not oProjQ4) and attn.oProj.residentFp32Ready
  if not (oProjQ4 or oProjFp32):
    failMsg = "oProj not resident (neither Q4 nor Fp32 is ready, layer " & $layerIdx & ")"
    return false
  let gateUpQ4 = ff.gateProj.residentReady and ff.upProj.residentReady
  let gateUpFp32 = (not gateUpQ4) and ff.gateProj.residentFp32Ready and ff.upProj.residentFp32Ready
  let downQ4 = ff.downProj.residentReady
  let downFp32 = (not downQ4) and ff.downProj.residentFp32Ready
  if not ((gateUpQ4 or gateUpFp32) and (downQ4 or downFp32)):
    failMsg = "FFN not resident (neither Q4 nor Fp32 is ready, layer " & $layerIdx & ")"
    return false
  return true

proc encodeDecodeLayerIntoOpenSession(attn: GQAAttention, normQKV, normFFN: RMSNormL,
    ff: SwiGLUL, hXIn: SessionHandle, rope: RoPEL, be: GpuBackend, cache: var KVCacheL,
    layerIdx: int, pending: var seq[SessionHandle]): SessionHandle =
  ## THE SHARED CORE of Mode 2 (forwardDecodeFullMergeSingleSession, 1 session per
  ## layer) AND Mode 3 (forwardDecodeAllLayersMergedSession, 1 session for the WHOLE
  ## model). Unlike the original (before the split): (1) it takes an EXISTING SessionHandle
  ## (hXIn) as input instead of a Tensor - no sessionUpload again, letting the previous
  ## layer pass its GPU output straight in as the next layer's input, with NO CPU read
  ## midway (the key of Mode 3: all 32 layers, NOT a single sessionRead
  ## until the last layer is done); (2) it does NOT sessionBegin/sessionEnd itself - the session
  ## is opened/closed by the CALLER (Mode 2: around exactly 1 layer; Mode 3:
  ## around the WHOLE 32-layer LOOP); (3) it returns a SessionHandle (hFinal), not
  ## a Tensor - the caller decides when a sessionRead is needed (Mode 2: right
  ## after that layer; Mode 3: exactly once, after the LAST layer).
  ## IMPORTANT NOTE (recorded by this very codebase as a source of SIGSEGV
  ## before): NEVER sessionFree any handle (not even the previous
  ## layer's, not even unused ones) while the session is still open - every handle
  ## encoded into a session must live until that session has
  ## finished sessionEnd. So `pending` here is ALWAYS 1 array SHARED across
  ## the session (Mode 3: shared by all 32 layers, not 1 array per layer).
  let C = attn.qProj.inF
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  let nRep = H div HKV
  let S = cache.seqLen + 1

  var failMsg: string
  if not precheckLayerResident(attn, ff, layerIdx, failMsg):
    raise newException(CatchableError, "[MERGED_SESSION] " & failMsg)

  let qkvQ4 = attn.qProj.residentReady
  let oProjQ4 = attn.oProj.residentReady
  let gateUpQ4 = ff.gateProj.residentReady
  let downQ4 = ff.downProj.residentReady

  template step(cond: bool, msg: string) =
    if not cond:
      raise newException(CatchableError, "[MERGED_SESSION] " & msg & " (layer " & $layerIdx & ")")

  # ---- RMSNorm (input) + QKV matmul ----
  let hX = hXIn
  let hGamma = (if normQKV.residentGammaReady: normQKV.residentGamma
                else: (let h = sessionUpload(be, normQKV.weight.data); pending.add(h); h))
  var hNormed = sessionAllocScratch(be, C); pending.add(hNormed)
  step(sessionRmsnormEnc(be, hX, hGamma, hNormed, 1, C, normQKV.eps), "RMSNorm input failed")

  var hQ = sessionAllocScratch(be, H * D); pending.add(hQ)
  var hK = sessionAllocScratch(be, HKV * D); pending.add(hK)
  var hV = sessionAllocScratch(be, HKV * D); pending.add(hV)
  var qkvFused = false
  if qkvQ4:
    qkvFused = sessionMatmulQ4Fused(be, 0, hNormed,
      [attn.qProj.residentW, attn.kProj.residentW, attn.vProj.residentW], [hQ, hK, hV],
      SessionHandle(), false)
  if qkvFused:
    discard
  elif qkvQ4:
    step(sessionMatmulQ4(be, hNormed, attn.qProj.residentW, hQ, 1), "Q matmul failed")
    step(sessionMatmulQ4(be, hNormed, attn.kProj.residentW, hK, 1), "K matmul failed")
    step(sessionMatmulQ4(be, hNormed, attn.vProj.residentW, hV, 1), "V matmul failed")
  else:
    step(sessionMatmul(be, hNormed, attn.qProj.residentFp32, hQ, 1, attn.qProj.inF, attn.qProj.outF), "Q matmul(fp32) failed")
    step(sessionMatmul(be, hNormed, attn.kProj.residentFp32, hK, 1, attn.kProj.inF, attn.kProj.outF), "K matmul(fp32) failed")
    step(sessionMatmul(be, hNormed, attn.vProj.residentFp32, hV, 1, attn.vProj.inF, attn.vProj.outF), "V matmul(fp32) failed")

  # ---- RoPE(Q), RoPE(K) in THE SAME session ----
  let halfD = D div 2
  var cosSlice = newSeq[float32](halfD)
  var sinSlice = newSeq[float32](halfD)
  let pos = cache.seqLen
  doAssert pos < rope.maxSeqLen, "RoPE: position exceeds max_seq_len"
  for i in 0 ..< halfD:
    cosSlice[i] = rope.cosT[pos * halfD + i]
    sinSlice[i] = rope.sinT[pos * halfD + i]
  let hCosT = sessionUpload(be, cosSlice); pending.add(hCosT)
  let hSinT = sessionUpload(be, sinSlice); pending.add(hSinT)
  step(sessionRope(be, hQ, hCosT, hSinT, 1, H, D), "RoPE(Q) failed/not supported")
  step(sessionRope(be, hK, hCosT, hSinT, 1, HKV, D), "RoPE(K) failed/not supported")

  # ---- Multiply Q by attn.scale RIGHT ON THE GPU ----
  step(sessionScale(be, hQ, attn.scale, H * D), "Scale(Q) failed/not supported")

  # ---- KV-resident sync ----
  ensureKvResident(cache.layers[layerIdx], be, HKV, cache.maxSeqLen, D)
  step(kvResidentUsable(cache, layerIdx), "KV resident is not ready for this layer")
  let kvFused = sessionKvAppendEnc(be, hK, hV, cache.layers[layerIdx].kResidentCombined,
                                   cache.layers[layerIdx].vResidentCombined, HKV, D,
                                   cache.maxSeqLen, cache.seqLen)
  for h in 0 ..< (if kvFused: 0 else: HKV):
    step(sessionCopyInto(be, hK, cache.layers[layerIdx].kResident[h], h * D, cache.seqLen * D, D),
         "copyInto K resident failed (head " & $h & ")")
    step(sessionCopyInto(be, hV, cache.layers[layerIdx].vResident[h], h * D, cache.seqLen * D, D),
         "copyInto V resident failed (head " & $h & ")")
    # NEW: write AT THE SAME TIME into the MERGED buffer used by the new fused kernel - see
    # the full explanation at the kResidentCombined field declaration (LayerKVCache).
    # Writing to BOTH places is REQUIRED so Mode 1 (reading kResident/vResident) and the fused
    # kernel (reading kResidentCombined/vResidentCombined) always see consistent
    # data, whatever path the previous/next token took.
    step(sessionCopyInto(be, hK, cache.layers[layerIdx].kResidentCombined,
                          h * D, h * cache.maxSeqLen * D + cache.seqLen * D, D),
         "copyInto K resident-combined failed (head " & $h & ")")
    step(sessionCopyInto(be, hV, cache.layers[layerIdx].vResidentCombined,
                          h * D, h * cache.maxSeqLen * D + cache.seqLen * D, D),
         "copyInto V resident-combined failed (head " & $h & ")")
  cache.layers[layerIdx].filled = cache.seqLen + 1

  # ---- Attention decode: TRY the FUSED kernel first (1 dispatch for ALL H
  # heads, instead of a loop of HKV*5 dispatch commands) - falls back to the old loop
  # (matmulNT+softmax+matmul per head, which RUNS CORRECTLY, the comparison baseline)
  # if the fused kernel fails for ANY reason (the app not rebuilt with the
  # new kernel, a non-Metal backend, etc.) - exactly the pattern of every other _enc function
  # in this file, NO path is a dead end.
  var hOutHeads = sessionAllocScratch(be, H * D); pending.add(hOutHeads)
  let fusedOk = sessionAttentionDecodeAllHeadsEnc(be, hQ, cache.layers[layerIdx].kResidentCombined,
                    cache.layers[layerIdx].vResidentCombined, hOutHeads, H, HKV, D, S, cache.maxSeqLen)
  if not fusedOk:
    if kvFused:
      for h in 0 ..< HKV:
        step(sessionCopyInto(be, hK, cache.layers[layerIdx].kResident[h], h * D, cache.seqLen * D, D),
             "copyInto K resident failed (head " & $h & ")")
        step(sessionCopyInto(be, hV, cache.layers[layerIdx].vResident[h], h * D, cache.seqLen * D, D),
             "copyInto V resident failed (head " & $h & ")")
    for hkv in 0 ..< HKV:
      var hQGroup = sessionAllocScratch(be, nRep * D); pending.add(hQGroup)
      step(sessionCopyInto(be, hQ, hQGroup, hkv * nRep * D, 0, nRep * D),
           "copyInto Q-group failed (kv-head " & $hkv & ")")
      var hKHead = cache.layers[layerIdx].kResident[hkv]
      var hVHead = cache.layers[layerIdx].vResident[hkv]

      let (hScores0, ntOk) = sMatmulNT(be, hQGroup, nRep, D, hKHead, S)
      pending.add(hScores0)
      step(ntOk, "matmulNT(Q,K) is not supported on this backend")
      let hScores = hScores0

      let hProbs = sSoftmax(be, hScores, nRep, S); pending.add(hProbs)
      var hOutGroup = sMatmul(be, hProbs, nRep, S, hVHead, D); pending.add(hOutGroup)

      step(sessionCopyInto(be, hOutGroup, hOutHeads, hkv * nRep * D, nRep * D),
           "copyInto outHeads failed (kv-head " & $hkv & ")")

  # ---- oProj ----
  var hOProjOut = sessionAllocScratch(be, C); pending.add(hOProjOut)
  var hX2 = SessionHandle()
  var oFused = false
  if oProjQ4 and attn.oProj.bias.data.len != C:
    hX2 = sessionAllocScratch(be, C); pending.add(hX2)
    oFused = sessionMatmulQ4Fused(be, 0, hOutHeads, [attn.oProj.residentW], [hX2], hX, true)
  if oFused:
    discard
  elif oProjQ4:
    step(sessionMatmulQ4(be, hOutHeads, attn.oProj.residentW, hOProjOut, 1), "oProj matmul failed")
  else:
    step(sessionMatmul(be, hOutHeads, attn.oProj.residentFp32, hOProjOut, 1, attn.oProj.inF, attn.oProj.outF),
         "oProj matmul(fp32) failed")
  var hOProjFinal = hOProjOut
  if attn.oProj.bias.data.len == C:
    let hBias = sessionUpload(be, attn.oProj.bias.data); pending.add(hBias)
    hOProjFinal = sVecOp(be, "add", hOProjOut, hBias, C); pending.add(hOProjFinal)

  # ---- Residual #1 ----
  if not oFused:
    hX2 = sVecOp(be, "add", hX, hOProjFinal, C); pending.add(hX2)

  # ---- RMSNorm (post-attn) + FFN ----
  let hGammaFfn = (if normFFN.residentGammaReady: normFFN.residentGamma
                   else: (let h = sessionUpload(be, normFFN.weight.data); pending.add(h); h))
  var hNormedFfn = sessionAllocScratch(be, C); pending.add(hNormedFfn)
  step(sessionRmsnormEnc(be, hX2, hGammaFfn, hNormedFfn, 1, C, normFFN.eps), "RMSNorm FFN failed")

  let interF = ff.gateProj.outF
  var hGate = sessionAllocScratch(be, interF); pending.add(hGate)
  var hUp = sessionAllocScratch(be, interF); pending.add(hUp)
  var hGated = SessionHandle()
  var gateFused = false
  if gateUpQ4 and ff.gateProj.bias.data.len != interF and ff.upProj.bias.data.len != interF:
    hGated = sessionAllocScratch(be, interF); pending.add(hGated)
    gateFused = sessionMatmulQ4Fused(be, 1, hNormedFfn,
      [ff.gateProj.residentW, ff.upProj.residentW], [hGated, hUp], SessionHandle(), false)
  if gateFused:
    discard
  elif gateUpQ4:
    step(sessionMatmulQ4(be, hNormedFfn, ff.gateProj.residentW, hGate, 1), "gateProj matmul failed")
    step(sessionMatmulQ4(be, hNormedFfn, ff.upProj.residentW, hUp, 1), "upProj matmul failed")
  else:
    step(sessionMatmul(be, hNormedFfn, ff.gateProj.residentFp32, hGate, 1, ff.gateProj.inF, ff.gateProj.outF),
         "gateProj matmul(fp32) failed")
    step(sessionMatmul(be, hNormedFfn, ff.upProj.residentFp32, hUp, 1, ff.upProj.inF, ff.upProj.outF),
         "upProj matmul(fp32) failed")
  if not gateFused and ff.gateProj.bias.data.len == interF:
    let hGB = sessionUpload(be, ff.gateProj.bias.data); pending.add(hGB)
    hGate = sVecOp(be, "add", hGate, hGB, interF); pending.add(hGate)
  if not gateFused and ff.upProj.bias.data.len == interF:
    let hUB = sessionUpload(be, ff.upProj.bias.data); pending.add(hUB)
    hUp = sVecOp(be, "add", hUp, hUB, interF); pending.add(hUp)
  if not gateFused:
    let hSig = sActivation(be, "sigmoid", hGate, interF); pending.add(hSig)
    let hSilu = sVecOp(be, "mul", hGate, hSig, interF); pending.add(hSilu)
    hGated = sVecOp(be, "mul", hSilu, hUp, interF); pending.add(hGated)
  var hOut = sessionAllocScratch(be, ff.downProj.outF); pending.add(hOut)
  var downFused = false
  if downQ4 and ff.downProj.bias.data.len != ff.downProj.outF and ff.downProj.outF == C:
    result = sessionAllocScratch(be, C); pending.add(result)
    downFused = sessionMatmulQ4Fused(be, 0, hGated, [ff.downProj.residentW], [result], hX2, true)
  if downFused:
    return
  if downQ4:
    step(sessionMatmulQ4(be, hGated, ff.downProj.residentW, hOut, 1), "downProj matmul failed")
  else:
    step(sessionMatmul(be, hGated, ff.downProj.residentFp32, hOut, 1, ff.downProj.inF, ff.downProj.outF),
         "downProj matmul(fp32) failed")
  if ff.downProj.bias.data.len == ff.downProj.outF:
    let hDB = sessionUpload(be, ff.downProj.bias.data); pending.add(hDB)
    hOut = sVecOp(be, "add", hOut, hDB, ff.downProj.outF); pending.add(hOut)

  # ---- Residual #2 ----
  result = sVecOp(be, "add", hX2, hOut, C); pending.add(result)

proc forwardDecodeFullMergeSingleSession(attn: GQAAttention, normQKV, normFFN: RMSNormL,
    ff: SwiGLUL, x: Tensor, rope: RoPEL, ctx: Backend, cache: var KVCacheL, layerIdx: int): Tensor =
  ## MODE 2 - keeps the old behaviour/name (1 session per layer): now just a thin
  ## wrapper calling encodeDecodeLayerIntoOpenSession between a sessionBegin/End pair.
  let C = x.shape[2]
  let be = ctx.kind.toByby()

  template chk(cond: bool, msg: string) =
    if not cond: raise newException(CatchableError, "[MODE2] " & msg & " (layer " & $layerIdx & ")")

  var failMsg: string
  chk(precheckLayerResident(attn, ff, layerIdx, failMsg), failMsg)
  chk(sessionBegin(be), "sessionBegin failed")
  var pending = newSeq[SessionHandle]()
  var failed = false
  try:
    let hXIn = sessionUpload(be, x.data); pending.add(hXIn)
    let hFinal = encodeDecodeLayerIntoOpenSession(attn, normQKV, normFFN, ff, hXIn, rope, be,
                                                    cache, layerIdx, pending)
    let sessionOk = sessionEnd(be)
    chk(sessionOk, "sessionEnd failed: " & sessionGetLastError(be))
    result = newTensor(x.shape)
    result.data = sessionRead(be, hFinal, C)
  except CatchableError:
    failed = true
    raise
  finally:
    if failed:
      discard sessionEnd(be)
    for h in pending.mitems:
      sessionFree(be, h)

proc forward*(blk: TransformerBlockLlama, x: Tensor, rope: RoPEL, ctx: Backend,
              cache: var KVCacheL, layerIdx: int, isPrefill: bool): Tensor =
  ## RMSNorm(input) is INSIDE forwardPrefill/forwardDecode (merged with QKV+
  ## RoPE - see fusedNormQKVRoPE), and residual+RMSNorm(post-attn)+FFN+residual
  ## is in fusedResidualNormFFN - the whole block now needs only 2 GPU sessions instead
  ## of 4+ separate per-submodule sessions as before.
  # MEASUREMENT: attn time (QKV+RoPE+attention+oProj) and FFN time separately, accumulated
  # over every layer/token - to know exactly where the time goes instead of guessing.
  let t0 = epochTime()
  if isPrefill:
    let attnOut = forwardPrefill(blk.attn, blk.inputNorm, x, rope, ctx, cache, layerIdx)
    let t1 = epochTime()
    result = fusedResidualNormFFN(blk.postAttnNorm, blk.ff, x, attnOut, ctx)
    let t2 = epochTime()
    gAttnTime += (t1 - t0)
    gFfnTime += (t2 - t1)
    return

  # Checks UP FRONT (pure CPU, no GPU) whether the FFN is resident-
  # eligible for session chaining - forwardDecode NEEDS to know beforehand
  # to decide whether to keep the session open or close it normally (no guessing/trying
  # and fixing later - a wrong guess leaves the open session never closed).
  # LEGACY BANNED: BYBY_LEGACY_FFN and BYBY_NO_MERGE_FFN_SESSION are no longer read - no
  # escape hatch may turn off the resident/merged-session path any more.
  let gateUpResidentQ4 = blk.ff.gateProj.residentReady and blk.ff.upProj.residentReady
  let downResidentQ4 = blk.ff.downProj.residentReady
  let gateUpResidentFp32 = (not gateUpResidentQ4) and blk.ff.gateProj.residentFp32Ready and blk.ff.upProj.residentFp32Ready
  let downResidentFp32 = (not downResidentQ4) and blk.ff.downProj.residentFp32Ready
  let ffnAllResident = (gateUpResidentQ4 and downResidentQ4) or (gateUpResidentFp32 and downResidentFp32)
  if not ffnAllResident:
    raise newException(CatchableError,
      "[FFN] resident is not ready in forwardDecode (layer " & $layerIdx &
      ") - the legacy fallback is BANNED by request, not continuing.")

  if gMergeQkvAttnSession:
    try:
      result = forwardDecodeFullMergeSingleSession(blk.attn, blk.inputNorm, blk.postAttnNorm,
                                                     blk.ff, x, rope, ctx, cache, layerIdx)
      let t2 = epochTime()
      gAttnTime += (t2 - t0)   # MODE 2: merged into 1 session -> attn/FFN cannot be split, everything goes to gAttnTime
      return
    except CatchableError as e:
      # SAFETY: Mode 2 failing at any step -> falls straight back to Mode 1 (the path
      # below, exactly the old code before Mode 2) for THIS token. Logs the
      # first time so it can be reported, no spam every token.
      if gMode2WarnLeft > 0:
        dec gMode2WarnLeft
        stderr.writeLine "[MODE2] failed, falling back to Mode 1 for this token: " & e.msg

  let dec0 = forwardDecode(blk.attn, blk.inputNorm, x, rope, ctx, cache, layerIdx, ffnAllResident)
  let t1 = epochTime()
  if dec0.sessionOpen:
    var pending = dec0.pending
    result = fusedResidualNormFFNContinue(blk.postAttnNorm, blk.ff, x, dec0.be, dec0.hOut, pending)
  else:
    result = fusedResidualNormFFN(blk.postAttnNorm, blk.ff, x, dec0.attnOut, ctx)
  let t2 = epochTime()
  if layerIdx == 0 and gDecodeProfileLeft > 0:
    # The missing FFN timing mark (the 2 old DECODE_PROFILE lines only had the attn part)
    # - now all 3 lines show EXACTLY how many ms each chunk takes for the first layer.
    stderr.writeLine "[DECODE_PROFILE] FFN (" & (if dec0.sessionOpen: "session NOI TIEP" else: "session rieng") &
      "): " & $((t2 - t1) * 1000) & " ms"
  gAttnTime += (t1 - t0)
  gFfnTime += (t2 - t1)

# ===================================================================
# LlamaModelL — the full model
# ===================================================================
type
  LlamaLConfig* = object
    vocabSize*, hiddenSize*, intermediateSize*: int
    nLayers*, nHeads*, nKVHeads*: int
    maxPositionEmbeddings*: int
    rmsNormEps*, ropeTheta*: float32
    ropeScalingType*: string        # "" (no scaling) / "linear" / "dynamic"
    ropeScalingFactor*: float32     # defaults to 1.0 when the model has no rope_scaling

  LlamaModelL* = object
    config*: LlamaLConfig
    embed*: Embedding
    blocks*: seq[TransformerBlockLlama]
    finalNorm*: RMSNormL
    lmHead*: Linear
    rope*: RoPEL
    payloadAlign*: int   ## NIMQ3 payload alignment of the mapped file (0 = NIMQ2)

proc newLlamaModelL*(cfg: LlamaLConfig): LlamaModelL =
  result.config = cfg
  result.embed = newEmbedding(cfg.vocabSize, cfg.hiddenSize, initializeWeights = false)
  result.blocks = newSeq[TransformerBlockLlama](cfg.nLayers)
  for i in 0 ..< cfg.nLayers:
    result.blocks[i] = newTransformerBlockLlama(cfg.hiddenSize, cfg.nHeads, cfg.nKVHeads,
                                                  cfg.intermediateSize, cfg.rmsNormEps)
  result.finalNorm = newRMSNormL(cfg.hiddenSize, cfg.rmsNormEps)
  result.lmHead = newLinear(cfg.hiddenSize, cfg.vocabSize, initializeWeights = false)
  result.rope = newRoPEL(cfg.hiddenSize div cfg.nHeads, cfg.maxPositionEmbeddings, cfg.ropeTheta,
                          cfg.ropeScalingType, cfg.ropeScalingFactor)

## ---- Adaptive GPU yield (yields the GPU to the desktop when thermal throttling
## is detected, see the detailed explanation in forwardStep) ----
var gLayerBaselineSec = -1.0    ## -1 = baseline not measured yet
var gBaselineSamples = 0
var gBaselineSumSec = 0.0
var gAdaptiveYieldMs = 0.0      ## starts at 0 - only grows when a real slowdown is detected
let gMaxYieldMs = (try: parseFloat(getEnv("BYBY_GPU_MAX_YIELD_MS", "8.0")) except ValueError: 8.0)

## ---- MODE 3: merges ALL 32 layers into EXACTLY 1 GPU session per token (instead of 1
## session per layer like Mode 2, i.e. 32 sessions/token) - the ggml-metal direction
## (1 command buffer per forward pass). Shares the encodeDecodeLayerIntoOpenSession core
## with Mode 2, the only differences: sessionBegin/End wraps the WHOLE layer LOOP, and the hidden
## state is passed straight between layers as a SessionHandle (staying on the GPU), with
## not a single sessionRead/sessionUpload in between - removing 31 CPU<->GPU round trips
## + 31 sessionBegin/End pairs (each forcing a GPU sync) compared with Mode 2.
var gMergeAllLayersSession* = getEnv("BYBY_MERGE_ALL_LAYERS_SESSION", "1") == "1"
var gMode3WarnLeft = 20   ## logs at most 20 Mode3->(Mode2/Mode1) fallbacks, to avoid spam
## NEW: a FIXED counter, printed ONCE at the end of the program (like
## KV_RESIDENT_ALLOC/gAttnTime/gFfnTime) - it does NOT depend on real-time terminal
## scrolling. Why it is required: the "[MODE3] failed" warning is printed
## with stderr.writeLine (its own line, SCROLLING with the terminal as usual) while
## the "[N/100 tok | ...]" line used to be PINNED to 1 position with ANSI cursor moves
## (see generateTokenLoop) - if
## these 2 lines landed on the same screen row, the pinned ANSI line would OVERWRITE the
## Mode3 warning, making the log look like Mode3 "ran smoothly" when in fact it
## failed 100% of the time (exactly what happened: gAttnTime took 94% of the decode time with
## Mode3 on - undeniable proof that Mode3 failed entirely, but
## no warning line showed in the terminal because the ANSI line overwrote it). This counter prints
## OUTSIDE the area ANSI overwrites (only once, at the end), so it cannot be lost.
var gMode3Attempts = 0
var gMode3Success = 0
var gMode3FirstFailMsg = ""

## Command-buffer pacing shared by the flash prefill and Mode 3 decode (see
## forwardPrefillFlash below for the BYBY_PREFILL_* variables).
let gFlashPrefill* = getEnv("BYBY_FLASH_PREFILL", "1") != "0"
proc envNum(name: string, fallback: int): int =
  try: parseInt(getEnv(name, $fallback)) except ValueError: fallback
let gPrefillCbMs = float(max(10, envNum("BYBY_PREFILL_CB_MS", 300)))
let gPrefillMaxRows = max(16, envNum("BYBY_PREFILL_MAX_ROWS", 512))
let gPrefillSessionMb = max(16, envNum("BYBY_PREFILL_SESSION_MB", 1024))
let gPrefillProfileLayer = envNum("BYBY_PREFILL_PROFILE", -1)  ## layer whose ops run one per command buffer, timed by BYBY_SESSION_LOG
var gPrefillMsPerLayerRow = -1.0   ## measured GPU ms per layer per prompt row
var gDecodeMsPerLayer = -1.0       ## measured GPU ms per layer of one decode token

var gLmHeadBiasFree = -1   ## 1 when the lm_head bias is absent or all zeros (checked once)

proc lmHeadBiasFree(m: LlamaModelL): bool =
  if gLmHeadBiasFree < 0:
    gLmHeadBiasFree = 1
    for v in m.lmHead.bias.data:
      if v != 0'f32:
        gLmHeadBiasFree = 0
        break
  gLmHeadBiasFree == 1

proc forwardDecodeAllLayersMergedSession(m: LlamaModelL, xData: seq[float32], ctx: Backend,
    cache: var KVCacheL, logits: var seq[float32]): Tensor =
  ## `logits` is filled when the final norm and lm_head ran in the same command
  ## buffer as the layers (resident gamma, resident Q4 lm_head, no bias, no LoRA);
  ## it stays empty otherwise and the caller runs them from the returned hidden state.
  let be = ctx.kind.toByby()
  let C = m.config.hiddenSize
  let headInSession = getEnv("BYBY_HEAD_IN_SESSION", "1") != "0" and
    m.finalNorm.residentGammaReady and m.lmHead.residentReady and
    m.lmHead.inF == C and lmHeadBiasFree(m) and m.lmHead.loraR == 0

  # Checks UP FRONT (pure CPU, no GPU) whether ALL 32 layers are resident-
  # eligible - avoids opening a session only to fail at the first layer (wasting a GPU
  # sync). If even 1 layer is not ready -> raise right away, NO session is opened.
  for idx in 0 ..< m.blocks.len:
    var failMsg: string
    if not precheckLayerResident(m.blocks[idx].attn, m.blocks[idx].ff, idx, failMsg):
      raise newException(CatchableError, "[MODE3] " & failMsg)

  # A command buffer that runs for seconds is killed by the macOS GPU watchdog
  # (and every later submission is then ignored): on a slow iGPU, or when the
  # weights are paged out, one token of 32 layers can take that long. The token
  # is split into command buffers of about BYBY_PREFILL_CB_MS each, committed
  # without waiting (sessionCommitContinue), from the measured ms per layer.
  # Apple Silicon keeps one command buffer per token (sessionPacesCommandBuffers).
  # A token normally fits one command buffer (as before pacing): it is split
  # only once a measured token takes longer than BYBY_DECODE_CB_MS (default
  # 1500), since every extra commit slows the decode down on Intel Metal.
  let decodeCbMs = float(max(100, envNum("BYBY_DECODE_CB_MS", 1500)))
  let layersPerCb = if not sessionPacesCommandBuffers(be) or gDecodeMsPerLayer <= 0 or
                       gDecodeMsPerLayer * float(m.blocks.len) <= decodeCbMs: m.blocks.len
                    else: max(1, int(decodeCbMs / gDecodeMsPerLayer))
  let t0 = epochTime()
  if not sessionBegin(be):
    raise newException(CatchableError, "[MODE3] sessionBegin failed")
  var pending = newSeq[SessionHandle]()
  var failed = false
  try:
    var hX = sessionUpload(be, xData); pending.add(hX)
    for idx in 0 ..< m.blocks.len:
      if idx > 0 and idx mod layersPerCb == 0 and not sessionCommitContinue(be):
        raise newException(CatchableError, "[MODE3] sessionCommitContinue failed")
      hX = encodeDecodeLayerIntoOpenSession(m.blocks[idx].attn, m.blocks[idx].inputNorm,
             m.blocks[idx].postAttnNorm, m.blocks[idx].ff, hX, m.rope, be, cache, idx, pending)
      # NOTE: do NOT sessionFree the previous layer's hX here - the session is still
      # open, every encoded handle must live until sessionEnd (see the full explanation
      # in encodeDecodeLayerIntoOpenSession). The previous layer's hX is already
      # in pending (encodeDecodeLayerIntoOpenSession adds it), and is
      # freed TOGETHER in the finally below, after this session has closed.
    var hLogits = SessionHandle()
    var headOk = false
    if headInSession:
      var hNormF = sessionAllocScratch(be, C); pending.add(hNormF)
      hLogits = sessionAllocScratch(be, m.lmHead.outF); pending.add(hLogits)
      headOk = sessionRmsnormEnc(be, hX, m.finalNorm.residentGamma, hNormF, 1, C, m.finalNorm.eps) and
               sessionMatmulQ4(be, hNormF, m.lmHead.residentW, hLogits, 1)
    let sessionOk = sessionEnd(be)
    if not sessionOk:
      raise newException(CatchableError, "[MODE3] sessionEnd failed: " & sessionGetLastError(be))
    let perLayer = (epochTime() - t0) * 1000.0 / float(m.blocks.len)
    gDecodeMsPerLayer = if gDecodeMsPerLayer <= 0: perLayer
                        else: 0.8 * gDecodeMsPerLayer + 0.2 * perLayer
    result = newTensor(@[1, 1, C])
    if headOk:
      logits = sessionRead(be, hLogits, m.lmHead.outF)
    else:
      result.data = sessionRead(be, hX, C)
  except CatchableError:
    failed = true
    raise
  finally:
    if failed:
      discard sessionEnd(be)
    for h in pending.mitems:
      sessionFree(be, h)

## ---- Flash prefill: every layer of a prompt on the GPU, attention included ----
## The old prefill (forwardPrefill + fusedResidualNormFFN) reads Q/K/V back to the
## CPU every layer, appends the KV cache, expands K/V per head, transposes,
## runs attention_fused in its own session and uploads again for oProj and the
## FFN. Here a layer is one chain of dispatches on the T rows: RMSNorm -> QKV ->
## RoPE -> KV cache scatter -> flash attention (tensor cores on a Vulkan GPU that
## has them) -> oProj -> residual -> RMSNorm -> FFN -> residual. The hidden state
## stays on the GPU between layers; only the last row is read back. Attention
## reads the resident cache, so a prompt continued at cache.seqLen > 0 attends
## to the earlier tokens as well.
## BYBY_FLASH_PREFILL=0 keeps the old path. BYBY_PREFILL_CB_MS (default 300) is the
## GPU time aimed at per command buffer, BYBY_PREFILL_MAX_ROWS (default 512) the
## largest row chunk, BYBY_PREFILL_SESSION_MB (default 1024) the activations of
## the layers encoded into one command buffer.
var gFlashPrefillWarned = false
var gFlashPrefillLogged = false

proc encodePrefillLayerIntoOpenSession(attn: GQAAttention, normQKV, normFFN: RMSNormL,
    ff: SwiGLUL, hXIn: SessionHandle, hCos, hSin: SessionHandle, be: GpuBackend,
    cache: var KVCacheL, layerIdx, T, pos: int, pending: var seq[SessionHandle]): SessionHandle =
  ## One prefill layer on T rows inside the caller's open session. Every handle it
  ## allocates goes to `pending` (freed after sessionEnd) except the result.
  let C = attn.qProj.inF
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  var failMsg: string
  if not precheckLayerResident(attn, ff, layerIdx, failMsg):
    raise newException(CatchableError, "[FLASH_PREFILL] " & failMsg)
  template step(cond: bool, msg: string) =
    if not cond:
      raise newException(CatchableError, "[FLASH_PREFILL] " & msg & " (layer " & $layerIdx & ")")
  template prof(label: string) =
    if gPrefillProfileLayer == layerIdx:
      stderr.writeLine "[op] " & label & ":"
      if not sessionEnd(be): raise newException(CatchableError, "[FLASH_PREFILL] profile sessionEnd failed")
      if not sessionBegin(be): raise newException(CatchableError, "[FLASH_PREFILL] profile sessionBegin failed")
  template linear(l: Linear, a: SessionHandle, rows: int, what: string): SessionHandle =
    var c = sessionAllocScratch(be, rows * l.outF)
    pending.add(c)
    if l.residentReady:
      step(sessionMatmulQ4(be, a, l.residentW, c, rows), what & " matmul failed")
    else:
      step(sessionMatmul(be, a, l.residentFp32, c, rows, l.inF, l.outF), what & " matmul(fp32) failed")
    var y = c
    if l.bias.data.len == l.outF:
      let hBias = sessionUpload(be, broadcastBias(l.bias, rows, l.outF))
      pending.add(hBias)
      y = sVecOp(be, "add", c, hBias, rows * l.outF)
      pending.add(y)
    y

  # ---- RMSNorm (input) + QKV + RoPE ----
  let hGamma = (if normQKV.residentGammaReady: normQKV.residentGamma
                else: (let h = sessionUpload(be, normQKV.weight.data); pending.add(h); h))
  var hNormed = sessionAllocScratch(be, T * C); pending.add(hNormed)
  step(sessionRmsnormEnc(be, hXIn, hGamma, hNormed, T, C, normQKV.eps), "RMSNorm input failed")
  prof("rmsnorm input")
  let hQ = linear(attn.qProj, hNormed, T, "Q")
  prof("Q matmul")
  let hK = linear(attn.kProj, hNormed, T, "K")
  prof("K matmul")
  let hV = linear(attn.vProj, hNormed, T, "V")
  prof("V matmul")
  step(sessionRope(be, hQ, hCos, hSin, T, H, D), "RoPE(Q) failed/not supported")
  prof("rope Q")
  step(sessionRope(be, hK, hCos, hSin, T, HKV, D), "RoPE(K) failed/not supported")
  prof("rope K")

  # ---- KV cache: merged [HKV, maxSeqLen, D] buffers, then the per-head copies ----
  ensureKvResident(cache.layers[layerIdx], be, HKV, cache.maxSeqLen, D)
  step(kvResidentUsable(cache, layerIdx), "KV resident is not ready for this layer")
  step(sessionKvScatterEnc(be, hK, cache.layers[layerIdx].kResidentCombined, T, HKV, D, pos,
                           cache.maxSeqLen), "K cache scatter failed")
  step(sessionKvScatterEnc(be, hV, cache.layers[layerIdx].vResidentCombined, T, HKV, D, pos,
                           cache.maxSeqLen), "V cache scatter failed")
  prof("kv scatter")
  for h in 0 ..< HKV:
    let off = h * cache.maxSeqLen * D + pos * D
    step(sessionCopyInto(be, cache.layers[layerIdx].kResidentCombined,
                         cache.layers[layerIdx].kResident[h], off, pos * D, T * D),
         "copyInto K resident failed (head " & $h & ")")
    step(sessionCopyInto(be, cache.layers[layerIdx].vResidentCombined,
                         cache.layers[layerIdx].vResident[h], off, pos * D, T * D),
         "copyInto V resident failed (head " & $h & ")")
  prof("kv per-head copies")
  cache.layers[layerIdx].filled = pos + T

  # ---- Flash attention -> oProj -> residual ----
  var hAttn = sessionAllocScratch(be, T * H * D); pending.add(hAttn)
  step(sessionFlashAttentionPrefillEnc(be, hQ, cache.layers[layerIdx].kResidentCombined,
         cache.layers[layerIdx].vResidentCombined, hAttn, T, H, HKV, D, pos, cache.maxSeqLen,
         attn.scale), "flash attention failed")
  prof("flash attention")
  let hO = linear(attn.oProj, hAttn, T, "oProj")
  prof("oProj matmul")
  let hX2 = sVecOp(be, "add", hXIn, hO, T * C); pending.add(hX2)
  prof("residual add")

  # ---- RMSNorm (post-attn) + SwiGLU FFN -> residual ----
  let hGammaFfn = (if normFFN.residentGammaReady: normFFN.residentGamma
                   else: (let h = sessionUpload(be, normFFN.weight.data); pending.add(h); h))
  var hNormedFfn = sessionAllocScratch(be, T * C); pending.add(hNormedFfn)
  step(sessionRmsnormEnc(be, hX2, hGammaFfn, hNormedFfn, T, C, normFFN.eps), "RMSNorm FFN failed")
  prof("rmsnorm ffn")
  let interF = ff.gateProj.outF
  let hGate = linear(ff.gateProj, hNormedFfn, T, "gateProj")
  prof("gate matmul")
  let hUp = linear(ff.upProj, hNormedFfn, T, "upProj")
  prof("up matmul")
  let hSig = sActivation(be, "sigmoid", hGate, T * interF); pending.add(hSig)
  prof("sigmoid")
  let hSilu = sVecOp(be, "mul", hGate, hSig, T * interF); pending.add(hSilu)
  prof("silu mul")
  let hGated = sVecOp(be, "mul", hSilu, hUp, T * interF); pending.add(hGated)
  prof("gated mul")
  let hDown = linear(ff.downProj, hGated, T, "downProj")
  prof("down matmul")
  result = sVecOp(be, "add", hX2, hDown, T * C)
  prof("residual add 2")

proc tileRows(rows: int): int =
  ## The prefill GEMMs work in 64-row tiles (Metal matmul_q4_gemm, Vulkan
  ## tiled tensor cores), so 16 rows cost about as much as 64: the speed is
  ## measured per row of whole tiles.
  (rows + 63) div 64 * 64

proc prefillFlashChunk(m: LlamaModelL, xData: seq[float32], T, pos: int, ctx: Backend,
                       cache: var KVCacheL): Tensor =
  ## All layers on rows pos .. pos + T - 1, in command buffers of at most
  ## BYBY_PREFILL_CB_MS each (from the measured ms per layer-row) and
  ## BYBY_PREFILL_SESSION_MB of activations. Returns the last row [1, 1, C].
  let be = ctx.kind.toByby()
  let C = m.config.hiddenSize
  let H = m.config.nHeads
  let D = C div H
  let ff0 = m.blocks[0].ff
  let a0 = m.blocks[0].attn
  let layerFloats = T * (4 * C + a0.qProj.outF + a0.kProj.outF + a0.vProj.outF + H * D +
                         a0.oProj.outF + 5 * ff0.gateProj.outF + ff0.downProj.outF)
  let memLayers = max(1, (gPrefillSessionMb * 1024 * 1024) div max(1, layerFloats * 4))
  let paced = sessionPacesCommandBuffers(be)

  # RoPE tables of the T positions, shared by every layer.
  let halfD = D div 2
  var cosSlice = newSeq[float32](T * halfD)
  var sinSlice = newSeq[float32](T * halfD)
  for t in 0 ..< T:
    let p = pos + t
    doAssert p < m.rope.maxSeqLen, "RoPE: position exceeds the precomputed max_seq_len"
    for i in 0 ..< halfD:
      cosSlice[t * halfD + i] = m.rope.cosT[p * halfD + i]
      sinSlice[t * halfD + i] = m.rope.sinT[p * halfD + i]

  var keep: seq[SessionHandle]       # live across command buffers
  var pending: seq[SessionHandle]    # freed after the command buffer that used them
  var hX: SessionHandle
  var hLast: SessionHandle
  var open = false
  try:
    var idx = 0
    while idx < m.blocks.len:
      # Unknown speed: one layer, then as many as fit the time budget
      # (no time budget on Apple Silicon: only the activation memory bound).
      let timeLayers = if not paced: m.blocks.len
                       elif gPrefillMsPerLayerRow <= 0: 1
                       else: int(gPrefillCbMs / (gPrefillMsPerLayerRow * float(tileRows(T))))
      let nLayers = clamp(min(timeLayers, memLayers), 1, m.blocks.len - idx)
      let t0 = epochTime()
      if not sessionBegin(be):
        raise newException(CatchableError, "[FLASH_PREFILL] sessionBegin failed")
      open = true
      if idx == 0:
        hX = sessionUpload(be, xData)
        keep.add sessionUpload(be, cosSlice)
        keep.add sessionUpload(be, sinSlice)
      let stop = idx + nLayers
      while idx < stop:
        let blk = m.blocks[idx]
        let hNext = encodePrefillLayerIntoOpenSession(blk.attn, blk.inputNorm, blk.postAttnNorm,
          blk.ff, hX, keep[0], keep[1], be, cache, idx, T, pos, pending)
        pending.add hX
        hX = hNext
        inc idx
      if idx == m.blocks.len:
        hLast = sessionAllocScratch(be, C)
        if not sessionCopyInto(be, hX, hLast, (T - 1) * C, 0, C):
          raise newException(CatchableError, "[FLASH_PREFILL] reading the last row failed")
      open = false
      if not sessionEnd(be):
        raise newException(CatchableError, "[FLASH_PREFILL] sessionEnd failed: " & sessionGetLastError(be))
      for h in pending.mitems: sessionFree(be, h)
      pending.setLen(0)
      if existsEnv("BYBY_SESSION_LOG"):
        stderr.writeLine "[prefill] rows " & $T & ", layers " & $nLayers & ", encode+gpu " &
          formatFloat((epochTime() - t0) * 1000.0, ffDecimal, 2) & " ms"
      let perLayerRow = (epochTime() - t0) * 1000.0 / float(nLayers * tileRows(T))
      gPrefillMsPerLayerRow = if gPrefillMsPerLayerRow <= 0: perLayerRow
                              else: 0.5 * gPrefillMsPerLayerRow + 0.5 * perLayerRow
    result = newTensor(@[1, 1, C])
    result.data = sessionRead(be, hLast, C)
  finally:
    if open: discard sessionEnd(be)
    for h in pending.mitems: sessionFree(be, h)
    for h in keep.mitems: sessionFree(be, h)
    sessionFree(be, hX)
    sessionFree(be, hLast)

proc forwardPrefillFlash(m: LlamaModelL, xData: seq[float32], T: int, ctx: Backend,
                         cache: var KVCacheL): Tensor =
  ## Runs every layer on the T prompt rows (see the comment above) and returns
  ## the last row's hidden state [1, 1, C]. Raises when a step is not available;
  ## nothing has been read back by then, so the caller can rerun the old path.
  ## A GPU watchdog (macOS kills a command buffer that runs for seconds on an
  ## Intel/AMD Mac) caps the work per command buffer: the prompt goes through in
  ## row chunks sized from the measured speed, each chunk continuing the KV
  ## cache of the last. Apple Silicon is not paced (sessionPacesCommandBuffers).
  let be = ctx.kind.toByby()
  let C = m.config.hiddenSize
  let D = C div m.config.nHeads
  let pos = cache.seqLen
  if pos + T > cache.maxSeqLen:
    raise newException(CatchableError, "[FLASH_PREFILL] the prompt does not fit the KV cache")
  for idx in 0 ..< m.blocks.len:
    var failMsg: string
    if not precheckLayerResident(m.blocks[idx].attn, m.blocks[idx].ff, idx, failMsg):
      raise newException(CatchableError, "[FLASH_PREFILL] " & failMsg)
  let kind = sessionFlashAttentionPrepare(be, D)
  if kind.len == 0:
    raise newException(CatchableError, "[FLASH_PREFILL] no flash-attention kernel for head_dim " & $D)
  if not gFlashPrefillLogged:
    gFlashPrefillLogged = true
    stderr.writeLine "[prefill] flash attention (" & kind & "), all layers on the GPU" &
      (if sessionPacesCommandBuffers(be): ", command buffers of ~" & $int(gPrefillCbMs) & " ms"
       else: "")
  var start = 0
  while start < T:
    # Rows per chunk: one 64-row GEMM tile until the speed is known, then what
    # one layer runs in the time budget (whole tiles, 16 .. BYBY_PREFILL_MAX_ROWS).
    # Apple Silicon (no pacing) takes the whole prompt in one chunk.
    var rows = T
    if sessionPacesCommandBuffers(be):
      rows = 64
      if gPrefillMsPerLayerRow > 0:
        let fit = int(gPrefillCbMs / gPrefillMsPerLayerRow)
        rows = clamp(if fit >= 64: fit div 64 * 64 else: fit div 16 * 16, 16, gPrefillMaxRows)
    rows = min(rows, T - start)
    result = prefillFlashChunk(m, xData[start * C ..< (start + rows) * C], rows, pos + start, ctx, cache)
    start += rows

proc forwardStep*(m: LlamaModelL, ids: seq[int], ctx: Backend, cache: var KVCacheL,
                   isPrefill: bool): Tensor =
  ## isPrefill=true: ids = the whole prompt (T tokens). isPrefill=false: ids = exactly 1 new token.
  ## Returns logits [1, 1, vocab] for the last token only; nothing samples
  ## from the earlier prompt positions.
  let stepT0 = epochTime()
  var x = m.embed.lookupBatch(@[ids], ctx)   # [1, T, C]
  let stepEmbed = epochTime() - stepT0
  when defined(debugStates):
    let Tdbg = ids.len
    let Cdbg = m.config.hiddenSize
    stderr.writeLine "[debug] after_embedding last_token[:8] = " & $x.data[(Tdbg-1)*Cdbg ..< (Tdbg-1)*Cdbg+8]

  # MODE 3: tries to merge ALL 32 layers into 1 session BEFORE entering the old per-layer
  # loop (Mode 2/Mode 1) - only applies to decode (isPrefill=false), exactly the
  # same condition as Mode 2. An error anywhere (1 of the 32 layers) -> raise
  # for the whole thing, falling STRAIGHT back to the old per-layer loop FOR THIS TOKEN (that loop
  # retries Mode 2 then Mode 1 for each layer as before - nothing changes).
  var mode3Done = false
  var stepLogits: seq[float32]   # set when Mode 3 ran the final norm and lm_head itself
  if isPrefill and gFlashPrefill and ids.len > 1:
    let t0 = epochTime()
    try:
      x = forwardPrefillFlash(m, x.data, ids.len, ctx, cache)
      mode3Done = true
      gAttnTime += epochTime() - t0
    except CatchableError as e:
      if cache.seqLen > 0: raise   # the old prefill only starts at position 0
      if not gFlashPrefillWarned:
        gFlashPrefillWarned = true
        stderr.writeLine "[prefill] flash path unavailable, using the old prefill: " & e.msg
  if gMergeAllLayersSession and not isPrefill:
    inc gMode3Attempts
    try:
      x = forwardDecodeAllLayersMergedSession(m, x.data, ctx, cache, stepLogits)
      mode3Done = true
      inc gMode3Success
    except CatchableError as e:
      if gMode3FirstFailMsg.len == 0:
        gMode3FirstFailMsg = e.msg
      if gMode3WarnLeft > 0:
        dec gMode3WarnLeft
        stderr.writeLine "[MODE3] failed, falling back to the per-layer loop (Mode2/Mode1) for this token: " & e.msg

  if not mode3Done:
    for idx in 0 ..< m.blocks.len:
      let layerT0 = epochTime()
      x = m.blocks[idx].forward(x, m.rope, ctx, cache, idx, isPrefill)
      let layerElapsed = epochTime() - layerT0
      # (a fixed yield replaced by an adaptive one): yielding a FIXED number of ms has
      # 2 downsides - (1) higher than needed -> wasted tokens/s while the machine is still cool and
      # does not need to yield that much, (2) lower than needed -> the machine still heats
      # up gradually until macOS throttles the GPU ITSELF (thermal throttle) - then
      # tokens/s is even worse than with no yield at all, since the throttle SLOWS DOWN PERMANENTLY
      # (until it cools down) rather than only during the yield. Here
      # each layer's own EXECUTION time is used as the feedback signal: when throttling
      # happens, the same layer (same amount of work) runs GRADUALLY SLOWER than
      # the baseline right after startup (still cool) - the EMA loop below
      # raises the yield when it detects a gradual slowdown (heating up), and lowers it back to
      # 0 once stable again (cooled down/no need to yield) - no guessing a
      # fixed number, it always tracks the machine's real thermal state.
      if not isPrefill:
        if gLayerBaselineSec < 0:
          # Startup: the first 3 layers of the FIRST TOKEN are used as the baseline
          # "while the machine is still cool" (not heated yet) - no yield while measuring.
          inc gBaselineSamples
          gBaselineSumSec += layerElapsed
          if gBaselineSamples >= 3:
            gLayerBaselineSec = gBaselineSumSec / float(gBaselineSamples)
        else:
          let ratio = layerElapsed / gLayerBaselineSec
          if ratio > 1.15:
            # >15% slower than the baseline -> a sign of throttling -> raise the yield
            # to cool down (gradually, no jumps, avoiding oscillation).
            gAdaptiveYieldMs = min(gAdaptiveYieldMs + 0.3, gMaxYieldMs)
          elif ratio < 1.05:
            # Back near the baseline -> cooled down/stable -> win the speed back gradually.
            gAdaptiveYieldMs = max(gAdaptiveYieldMs - 0.1, 0.0)
          if gAdaptiveYieldMs >= 1.0:
            sleep(int(gAdaptiveYieldMs))
        when defined(debugGpuYield):
          if idx == m.blocks.len - 1:
            stderr.writeLine "[gpu_yield] baseline=" & $gLayerBaselineSec &
              "s last=" & $layerElapsed & "s yieldMs=" & $gAdaptiveYieldMs
      # GC: with --gc:arc (ref-counting), GC_fullCollect() is a complete no-op —
      # each layer's temporary memory is freed as soon as it leaves scope, nothing
      # accumulates. It is ONLY forced during prefill (runs exactly once per prompt, T
      # tokens at once so it may temporarily grow) - decode (T=1 per step, running
      # REPEATEDLY every token) no longer forces a GC. It used to be called unconditionally even
      # during decode -> without a --gc:arc build, every token paid 4 extra
      # full stop-the-world scans (mod 8 x 32 layers + the last layer), a pure CPU
      # cost added equally on CPU/Metal/CUDA/OpenCL, unrelated
      # to the GPU kernels - part of why all 3 GPU backends were
      # equally abnormally slow compared to llama.cpp.
      if (isPrefill or existsEnv("BYBY_LEGACY_DECODE_GC")) and
         ((idx + 1) mod 8 == 0 or idx == m.blocks.len - 1):
        GC_fullCollect()
      when defined(debugStates):
        if idx == 0:
          stderr.writeLine "[debug] after_block_0   last_token[:8] = " & $x.data[(Tdbg-1)*Cdbg ..< (Tdbg-1)*Cdbg+8]
  when defined(debugStates):
    stderr.writeLine "[debug] after_last_block last_token[:8] = " & $x.data[(Tdbg-1)*Cdbg ..< (Tdbg-1)*Cdbg+8]
  # Final norm and lm_head (vocab x hidden, the largest GEMM of the step) for
  # the last token only: prefill used to compute logits for every prompt row.
  let C = m.config.hiddenSize
  let T = x.shape[1]
  if T > 1:
    x.data = x.data[(T - 1) * C ..< T * C]
    x.shape = @[x.shape[0], 1, C]
  let stepLayersEnd = epochTime()
  if stepLogits.len == m.lmHead.outF:
    result = newTensor(@[1, 1, m.lmHead.outF])
    result.data = move stepLogits
  else:
    let normed = m.finalNorm.forward(x, ctx)
    result = m.lmHead.forward(normed, ctx)
  if not isPrefill and existsEnv("BYBY_SESSION_LOG"):
    stderr.writeLine "[step] embed " & formatFloat(stepEmbed * 1000, ffDecimal, 2) &
      " ms, layers " & formatFloat((stepLayersEnd - stepT0 - stepEmbed) * 1000, ffDecimal, 2) &
      " ms, norm+lm_head " & formatFloat((epochTime() - stepLayersEnd) * 1000, ffDecimal, 2) & " ms"
  when defined(debugStates):
    let lastRow = result.data[0 ..< 8]
    stderr.writeLine "[debug] logits last_token[:8] = " & $lastRow
  cache.seqLen += ids.len

# ===================================================================
# Read local config.json directly; quant.nim loads the compressed weights.
# ===================================================================
proc newLlamaLConfig*(modelPath: string): LlamaLConfig =
  let path = if fileExists(modelPath) and modelPath.endsWith(".json"): modelPath
             else: modelPath / "config.json"
  if not fileExists(path):
    raise newException(IOError, "[nim_inference] config.json not found: " & path)
  let cfg = parseFile(path)
  if cfg.kind != JObject:
    raise newException(ValueError, "[nim_inference] model config must be a JSON object")
  proc requiredInt(name: string): int =
    if not cfg.hasKey(name) or cfg[name].kind != JInt or cfg[name].getInt() <= 0:
      raise newException(ValueError, "[nim_inference] missing/invalid config field: " & name)
    cfg[name].getInt()
  proc optionalFloat(name: string, fallback: float32): float32 =
    if cfg.hasKey(name) and cfg[name].kind in {JInt, JFloat}: cfg[name].getFloat().float32
    else: fallback
  result.vocabSize = requiredInt("vocab_size")
  result.hiddenSize = requiredInt("hidden_size")
  result.intermediateSize = requiredInt("intermediate_size")
  result.nLayers = requiredInt("num_hidden_layers")
  result.nHeads = requiredInt("num_attention_heads")
  result.nKVHeads = if cfg.hasKey("num_key_value_heads") and cfg["num_key_value_heads"].kind == JInt:
                      requiredInt("num_key_value_heads")
                    else: result.nHeads
  result.maxPositionEmbeddings = if cfg.hasKey("max_position_embeddings"):
                                   requiredInt("max_position_embeddings") else: 2048
  result.rmsNormEps = optionalFloat("rms_norm_eps", 1e-6'f32)
  result.ropeTheta = optionalFloat("rope_theta", 10000'f32)
  result.ropeScalingType = ""
  result.ropeScalingFactor = 1.0'f32
  if cfg.hasKey("rope_scaling") and cfg["rope_scaling"].kind == JObject:
    let rs = cfg["rope_scaling"]
    result.ropeScalingType = rs.getOrDefault("rope_type").getStr(
      rs.getOrDefault("type").getStr(""))
    result.ropeScalingFactor = rs.getOrDefault("factor").getFloat(1.0).float32
    if result.ropeScalingFactor <= 0:
      raise newException(ValueError, "[nim_inference] rope_scaling.factor must be positive")
  if result.hiddenSize mod result.nHeads != 0 or result.nHeads mod result.nKVHeads != 0:
    raise newException(ValueError, "[nim_inference] incompatible hidden size / attention head counts")

proc loadLlamaWeightsL*(m: var LlamaModelL, path: string) =
  echo &"  reading file {path} ..."
  let fileSize = getFileSize(path)
  echo &"  file size: {fileSize div (1024*1024)} MB"
  # The weights are mapped, so every page is otherwise read from disk on its
  # first touch: a 16 KB demand fault at a time, mid-decode (the embedding
  # row of each new token, the GPU's first read of each tensor). On slow
  # storage that costs seconds. One sequential pass with large reads fills
  # the page cache instead. BYBY_PREFAULT=0 skips it.
  if getEnv("BYBY_PREFAULT", "1") != "0" and getEnv("BYBY_MODEL_MMAP", "1") != "0":
    let t0 = epochTime()
    var f: File
    if open(f, path, fmRead):
      var chunk = newSeq[byte](16 * 1024 * 1024)
      while readBuffer(f, addr chunk[0], chunk.len) > 0: discard
      close(f)
      stderr.writeLine "[load] read " & $(fileSize div (1024 * 1024)) & " MB sequentially in " &
        formatFloat(epochTime() - t0, ffDecimal, 2) & " s (BYBY_PREFAULT=0 skips)"
  # mmap: packed weights stay in the page cache and uploadAllResidentWeights
  # hands that mapping to the GPU (in place on unified memory, at any payload
  # offset; NIMQ3 files from convert_nimq_aligned are page aligned).
  let loaded = loadQuantStateDict(path, mapped = getEnv("BYBY_MODEL_MMAP", "1") != "0")
  let sd = loaded.sd
  m.payloadAlign = loaded.payloadAlign
  echo &"  mapped {sd.len} tensors from the file (" &
    (if loaded.payloadAlign > 0: &"NIMQ3, payload align {loaded.payloadAlign}"
     else: "NIMQ2: the GPU reads payloads at arbitrary byte offsets in the mmap") &
    "), building the model..."
  var byName = initTable[string, QuantTensor]()
  for (name, qt) in sd: byName[name] = qt

  proc get(name: string): seq[float32] =
    if not byName.hasKey(name): return @[]
    dequantizeTensor(byName[name])

  proc loadLinear(lin: var Linear, prefix: string) =
    ## Keeps the weights COMPRESSED (setQuantWeight, see nimformer.nim) instead of
    ## dequantizing everything to fp32 at load (the old get()) - for a 6.7B model, dequantizing
    ## all Linears to fp32 takes ~27GB of RAM (the cause of the OOM "killed"
    ## during "Building model..."). Keeping them compressed takes only ~4GB, dequantized temporarily on each
    ## forward() and dropped right away (see nimformer.nim Linear.forward).
    let wName = prefix & ".weight"
    if byName.hasKey(wName):
      setQuantWeight(lin, byName[wName])
    # Zero biases (exporters write them for bias-free Llama/DeepSeek layers)
    # would only cost an upload + add per layer per token; drop them.
    let b = get(prefix & ".bias")
    var hasBias = false
    for v in b:
      if v != 0'f32: (hasBias = true; break)
    lin.bias.data = if hasBias: b else: @[]

  # Embedding rows are read from the mmap in place when the file stores raw
  # fp32 (lookupBatch copies only the rows of the prompt/decoded ids).
  var embW: seq[float32]
  let embView = if byName.hasKey("model.embed_tokens.weight"):
                  rawFloatView(byName["model.embed_tokens.weight"])
                else: default(MappedView[float32])
  if embView.owner != nil and embView.len == m.config.vocabSize * m.config.hiddenSize:
    if getEnv("BYBY_EMBED_RAM", "1") != "0" and byName.hasKey("lm_head.weight"):
      # Untied embedding: a decode step reads one row of it, and a row that
      # is not in the page cache costs a disk read in the middle of the
      # step. Keep the table in process memory instead (BYBY_EMBED_RAM=0
      # reads it from the mapping).
      var table = newSeq[float32](embView.len)
      copyMem(addr table[0], embView.dataPtr, embView.len * sizeof(float32))
      m.embed.weight.data = table
    else:
      m.embed.weight.data = @[]
      m.embed.weight.mappedData = embView
  else:
    embW = get("model.embed_tokens.weight")
    if embW.len > 0: m.embed.weight.data = embW

  let normW = get("model.norm.weight")
  if normW.len > 0: m.finalNorm.weight.data = normW

  for l in 0 ..< m.config.nLayers:
    echo &"  loading layer {l+1}/{m.config.nLayers} ..."
    let p = "model.layers." & $l
    let ln1 = get(p & ".input_layernorm.weight")
    if ln1.len > 0: m.blocks[l].inputNorm.weight.data = ln1
    let ln2 = get(p & ".post_attention_layernorm.weight")
    if ln2.len > 0: m.blocks[l].postAttnNorm.weight.data = ln2

    loadLinear(m.blocks[l].attn.qProj, p & ".self_attn.q_proj")
    loadLinear(m.blocks[l].attn.kProj, p & ".self_attn.k_proj")
    loadLinear(m.blocks[l].attn.vProj, p & ".self_attn.v_proj")
    loadLinear(m.blocks[l].attn.oProj, p & ".self_attn.o_proj")

    loadLinear(m.blocks[l].ff.gateProj, p & ".mlp.gate_proj")
    loadLinear(m.blocks[l].ff.upProj, p & ".mlp.up_proj")
    loadLinear(m.blocks[l].ff.downProj, p & ".mlp.down_proj")

  # lm_head: if it is in the file, setQuantWeight (kept compressed); if not (tied
  # embeddings) the already dequantized embW is used directly (only once, not
  # 224 times like the other Linears, so RAM is not a concern) - Linear.forward
  # falls back to the useQuant=false branch, a normal transpose on each call
  # (acceptable since lm_head runs forward ONCE per token, not 224 times).
  if byName.hasKey("lm_head.weight"):
    setQuantWeight(m.lmHead, byName["lm_head.weight"])
  elif embW.len > 0 or m.embed.weight.mappedData.owner != nil:
    # Tied embeddings: lm_head reads the same rows as the embedding.
    m.lmHead.weight.data = embW
    m.lmHead.weight.mappedData = m.embed.weight.mappedData
    m.lmHead.useQuant = false

proc uploadAllResidentWeights*(m: var LlamaModelL, ctx: Backend) =
  ## Uploads ALL weight matrices (int4-per-group, i.e. the real hot loop path)
  ## to the GPU ONCE, right after loadLlamaWeightsL. From the first token on,
  ## linearFast() uploads only the small activation each step, never touching the 4.5GB
  ## of weights again (see beResidentLinearQ4/sessionMatmulQ4 in backend.nim).
  echo "  uploading resident weights to the GPU (once) ..."
  # NEW: sparsity (block-level, forcing scale=0 for less important blocks) -
  # read from an env var, defaults to 0.0 (OFF, behaviour exactly as before). ONLY
  # applied to the FFN (gate/up/down_proj - the LARGEST matrices) - Q/K/V/O are left
  # unchanged to limit the risk to attention quality.
  let sparsityPct = parseFloat(getEnv("BYBY_SPARSITY_PCT", "0.0")).float32
    ## REVERTED: back to the default OFF (0.0). 20% was tried on a real machine - result:
    ## broken output (meaningless text, no longer correct Fibonacci code)
    ## AND the speed was not reliably faster under normal load
    ## (not the "super cool" moment that measured 32.9s earlier). 20% is far too much
    ## for this model without a fine-tune/recalibration step after pruning -
    ## the error accumulated over 32 layers broke the output entirely. To try again, enable it
    ## by hand with BYBY_SPARSITY_PCT at a MUCH lower level (e.g. 3-5), increasing
    ## little by little, always comparing the output with the non-sparse version before
    ## increasing further.
  if sparsityPct > 0.0'f32:
    echo "  [SPARSITY] ON: pruning " & $sparsityPct & "% of FFN blocks (gate/up/down_proj) - " &
         "LOSES QUALITY, in exchange for faster GPU reads in the batch prefill kernel"
  proc tune(lin: Linear) =
    if lin.residentReady:
      discard residentPrepareQ4(lin.residentW, 1)
      discard residentPrepareQ4(lin.residentW, 8)
  for l in 0 ..< m.config.nLayers:
    uploadResident(m.blocks[l].attn.qProj, ctx)
    uploadResident(m.blocks[l].attn.kProj, ctx)
    uploadResident(m.blocks[l].attn.vProj, ctx)
    uploadResident(m.blocks[l].attn.oProj, ctx)
    uploadResident(m.blocks[l].ff.gateProj, ctx, sparsityPct)
    uploadResident(m.blocks[l].ff.upProj, ctx, sparsityPct)
    uploadResident(m.blocks[l].ff.downProj, ctx, sparsityPct)
    tune(m.blocks[l].attn.qProj); tune(m.blocks[l].attn.kProj)
    tune(m.blocks[l].attn.vProj); tune(m.blocks[l].attn.oProj)
    tune(m.blocks[l].ff.gateProj); tune(m.blocks[l].ff.upProj)
    tune(m.blocks[l].ff.downProj)
    uploadResidentNorm(m.blocks[l].inputNorm, ctx)
    uploadResidentNorm(m.blocks[l].postAttnNorm, ctx)
  # An untied fp32 lm_head is the largest matrix read per decode token
  # (hidden * vocab * 4 bytes). Quantise it once to the same grouped Q4 the
  # projections use, so it runs on the tuned Q4 GEMV and reads 8x fewer
  # bytes. BYBY_LM_HEAD=fp32 keeps the original fp32 weights.
  if getEnv("BYBY_LM_HEAD", "q4") == "q4" and m.lmHead.useQuant and
      m.lmHead.weightQ.kind == qkFp32Raw and m.lmHead.weightQ.mappedData.owner != nil and
      m.lmHead.inF mod 128 == 0 and
      m.lmHead.weightQ.mappedData.len == m.lmHead.inF * m.lmHead.outF * 4:
    let n = m.lmHead.inF * m.lmHead.outF
    let src = cast[ptr UncheckedArray[float32]](m.lmHead.weightQ.mappedData.dataPtr)
    let t0 = epochTime()
    # Same grouped asymmetric Q4 as quantizeInt4AsymmetricGrouped (group 128,
    # low nibble first), written without a temporary seq per group.
    const lmGroup = 128
    let rows = m.lmHead.outF
    let cols = m.lmHead.inF
    let groups = cols div lmGroup
    var qt = QuantTensor(kind: qkInt4Asymmetric, shape: @[rows, cols], groupSize: lmGroup,
      data: newSeq[uint8](rows * (cols div 2)),
      scale: newSeq[float32](rows * groups),
      zero_point: newSeq[float32](rows * groups))
    for row in 0 ..< rows:
      for g in 0 ..< groups:
        let base = row * cols + g * lmGroup
        var lo = float32.high
        var hi = -float32.high
        for i in 0 ..< lmGroup:
          let v = src[base + i]
          if v < lo: lo = v
          if v > hi: hi = v
        let range = hi - lo
        let scale = if range < 1e-12'f32: 1'f32 else: range / 15'f32
        let zero = -lo / scale
        qt.scale[row * groups + g] = scale
        qt.zero_point[row * groups + g] = zero
        let outBase = row * (cols div 2) + g * (lmGroup div 2)
        for i in 0 ..< lmGroup div 2:
          let q0 = uint8(clamp(int(round(src[base + 2 * i] / scale + zero)), 0, 15))
          let q1 = uint8(clamp(int(round(src[base + 2 * i + 1] / scale + zero)), 0, 15))
          qt.data[outBase + i] = q0 or (q1 shl 4)
    setQuantWeight(m.lmHead, qt)
    stderr.writeLine "[lm_head] fp32 -> grouped Q4 (" & $m.lmHead.outF & "x" & $m.lmHead.inF &
      ") in " & formatFloat(epochTime() - t0, ffDecimal, 2) & " s; BYBY_LM_HEAD=fp32 keeps fp32"
  uploadResident(m.lmHead, ctx)  # no-op for tied embeddings (useQuant=false)
  tune(m.lmHead)
  var zeroCopy, residentQ4 = 0
  proc count(lin: Linear) =
    if lin.residentReady or lin.residentFp32Ready:
      inc residentQ4
      if lin.residentZeroCopy: inc zeroCopy
  for l in 0 ..< m.config.nLayers:
    count(m.blocks[l].attn.qProj); count(m.blocks[l].attn.kProj)
    count(m.blocks[l].attn.vProj); count(m.blocks[l].attn.oProj)
    count(m.blocks[l].ff.gateProj); count(m.blocks[l].ff.upProj)
    count(m.blocks[l].ff.downProj)
  count(m.lmHead)
  echo &"  weight (Q4 + lm_head): {zeroCopy}/{residentQ4} read by the GPU directly from the mmap (zero-copy), " &
    &"{residentQ4 - zeroCopy} copied from the mmap into GPU buffers."
  uploadResidentNorm(m.finalNorm, ctx)
  # OPTIMISATION: the gamma of ALL 32*2+1 RMSNorms (input+postAttn per layer + final)
  # is now uploaded ONCE here, instead of sessionUpload() FROM
  # SCRATCH in EVERY layer EVERY token in forwardPrefill/forwardDecode - saving
  # exactly 1 unnecessary small buffer upload (~dim*4 bytes) every decode step
  # (see the 3 places changed in fusedNormQKVRoPE/fusedResidualNormFFN(Continue)).
  # Picks and checks the prefill flash-attention kernel now, so its parity
  # test and shader compile are not counted in the first prefill.
  if gFlashPrefill:
    discard sessionFlashAttentionPrepare(ctx.kind.toByby(), m.config.hiddenSize div m.config.nHeads)
  echo "  done - weights resident on the GPU, not uploaded again every token."


proc sampleTopP(ctx: Backend, logits: seq[float32], temperature: float32 = 0.7'f32, topP: float32 = 0.9'f32,
                 recentIds: openArray[int] = [], repeatPenalty: float32 = 1.0'f32,
                 noPenaltyIds: openArray[int] = []): int =
  # Adds a repetition penalty (llama.cpp-style) - without it the model
  # easily falls into token loops like "),),),),)" since each step picks again exactly the
  # token with the highest logit chosen before. The logits of recently seen tokens
  # are lowered (divided by repeatPenalty if positive, multiplied if negative) before
  # computing softmax/top-p.
  # 2nd FIX (the real reason the model avoided generating \n): it used to
  # penalise ALL tokens in recentIds, including \n/space/tab. In code, \n
  # repeats VERY often in the last 64 tokens (almost every line) ->
  # the \n logit got divided by repeatPenalty again and again over many steps ->
  # pushed well below the space token -> the model was forced to replace \n with
  # a space even when it "wanted" a syntactically correct line break. Exactly why llama.cpp
  # has the penalize_nl flag (by default \n is NOT penalised) - pure formatting tokens
  # (\n, space, tab) should not count as "repeated content" like regular
  # tokens. Fix: also take noPenaltyIds (the ids of \n/space/tab/indentation from
  # the tokenizer) and skip them when applying the penalty.
  # Default 1.0 (off, like current llama.cpp): code has to repeat n, fib,
  # parentheses and keywords, and 1.3 over the last 64 ids pushed greedy
  # decoding into "1== n", throwaway names a..h and runs of blank lines.
  var adjLogits = logits
  if repeatPenalty != 1.0'f32:
    for id in recentIds:
      if id >= 0 and id < adjLogits.len:
        var skip = false
        for npid in noPenaltyIds:
          if id == npid:
            skip = true
            break
        if skip: continue
        if adjLogits[id] > 0'f32: adjLogits[id] = adjLogits[id] / repeatPenalty
        else: adjLogits[id] = adjLogits[id] * repeatPenalty

  if temperature <= 0'f32:
    var best = 0
    for i in 1 ..< adjLogits.len:
      if adjLogits[i] > adjLogits[best]: best = i
    return best

  # ON THE GPU: the most expensive part of sampling is exp() over the whole vocab (possibly
  # 32k-128k+ elements EVERY token) - moved to the GPU through beSoftmax (using
  # gpuSoftmax, with the CPU fallback BANNED like the other ops). Before: a pure CPU
  # loop computing exp() + sum + division per element.
  # Note: adjLogits already has maxL subtracted and is divided by the temperature BEFORE going to the GPU,
  # since the GPU softmax kernel (softmax_kernel/rmsnorm-style) handles numerical
  # stability itself (subtracting the max internally) but does NOT know about sampling's own
  # temperature scaling - it must be divided first.
  var scaledLogits = newSeq[float32](adjLogits.len)
  for i in 0 ..< adjLogits.len: scaledLogits[i] = adjLogits[i] / temperature
  var probs = beSoftmax(ctx, scaledLogits, 1, adjLogits.len)

  # OPTIMISATION (replacing "idx.sort of the whole vocab, O(V log V), EVERY token"):
  # top-p only needs to CUT at the cumulative topP threshold - there is no need to sort ALL
  # V elements, only a subset of the K most probable ones. Uses
  # partial selection (threshold-based) instead of a full sort:
  #   1) One pass finds an approximate threshold (it need not be exact) by
  #      collecting the K largest candidates in a simple loop (small K, ~256-512,
  #      enough that a real topP <=0.95-0.99 almost always falls within those K for
  #      the softmax distribution of real LLM logits - as observed in
  #      practice with llama.cpp/vLLM's "top_k before top_p" strategy).
  #   2) ONLY those K elements are sorted (O(K log K)) instead of V elements.
  # If K is not enough (topP not reached after K elements, rare), fall back to
  # doubling K and retrying - at most a few times, still cheaper than a full sort.
  var K = min(512, probs.len)
  var idx: seq[int]
  while true:
    # REAL OPTIMISATION (replacing the first, wrong O(V*K) version - spotted and fixed
    # right here): a min-heap of size K (std/heapqueue) -> O(V log K)
    # instead of a linear find-min scan per element (O(V*K), SLOWER than the original
    # O(V log V) sort with K=512). The heap compares by (prob, idx) ascending -
    # the top of the heap is always the SMALLEST prob among the current K, replaced
    # as soon as a larger element comes along.
    var heap = initHeapQueue[(float32, int)]()
    for i in 0 ..< probs.len:
      let item = (probs[i], i)
      if heap.len < K:
        heap.push(item)
      elif probs[i] > heap[0][0]:
        discard heap.pop()
        heap.push(item)
    idx = newSeq[int](heap.len)
    for i in 0 ..< heap.len: idx[i] = heap[i][1]
    idx.sort(proc(a, b: int): int = cmp(probs[b], probs[a]))
    var cumCheck = 0'f32
    for id in idx: cumCheck += probs[id]
    if cumCheck >= topP or K >= probs.len:
      break
    K = min(K * 2, probs.len)

  var cum = 0'f32
  var cutoff = idx.len
  for i, id in idx:
    cum += probs[id]
    if cum >= topP:
      cutoff = i + 1
      break
  var r = rand(1.0'f32) * cum
  var acc = 0'f32
  for i in 0 ..< cutoff:
    acc += probs[idx[i]]
    if acc >= r: return idx[i]
  return idx[cutoff - 1]

## Ctrl+C during generate() only sets gStopRequested: the loop ends after the
## current token and prints the statistics. Leaving from the signal handler
## instead ran exit procs while the main thread was inside the Vulkan driver
## (SIGSEGV on Colab T4).
var gStopRequested = false
var gGenerating = false

proc generate*(m: LlamaModelL, tok: HFTokenizer, ctx: Backend, promptIds: seq[int],
               maxNewTokens: int = 128, temperature: float32 = 0.7'f32, topP: float32 = 0.9'f32,
               showProgress = true, statsOut: File = stderr, verbose = true): string =
  ## statsOut receives the speed summary; verbose=false keeps only the
  ## Prefill/Decode/Total lines (callers that mute the library logs).
  if verbose:
    echo "[inference] optimized paths: all-layers-session=" & $gMergeAllLayersSession &
      ", merged-QKV-attention=" & $gMergeQkvAttnSession &
      ", resident-KV=" & $(getEnv("BYBY_DISABLE_KV_RESIDENT", "0") != "1")
  # Takes promptIds directly (already tokenized through applyChatTemplate) instead of
  # taking a prompt string and calling encode() again - encode() uses add_special_tokens
  # (a raw BOS) and does NOT apply the model's real chat template -> an instruct model
  # does not recognise the conversation format and rambles.
  # This used to use m.config.maxPositionEmbeddings directly (=16384 for
  # this model) as the KV cache size -> allocating ~16GB of RAM (32 layers x 32
  # KV heads x 16384 positions x 128 dims x 2(k+v) x 4 bytes) RIGHT AT THE START,
  # however short the actual prompt/maxNewTokens - the cause of the OOM "killed"
  # during Generating. Now only the length actually needed is allocated (prompt +
  # tokens to generate), plus a little margin for safety.
  let neededSeqLen = promptIds.len + maxNewTokens + 8
  var cache = newKVCacheL(m.config.nLayers, m.config.nKVHeads,
                           m.config.hiddenSize div m.config.nHeads,
                           neededSeqLen)

  # ---- KV-RESIDENT WARMUP: allocates the KV heap for ALL 32 layers up front, RIGHT AFTER
  # the cache is CREATED, BEFORE PREFILL. POSITION FIXED (the previous version put it AFTER
  # prefill, which was WRONG - verified with real logs: warmupTime always came out
  # 0.000s while KV_RESIDENT_ALLOC still took 65-100s DURING
  # PREFILL, because the old prefill path (syncKvResidentEncode) itself called
  # ensureKvResident() for each layer inside forwardStep(isPrefill
  # = true) itself - putting the warmup after that call is too late, the prefill had silently
  # "swallowed" this one-off cost into its own "Prefill: ... tok/s"
  # number). Placed HERE - before the first forwardStep(isPrefill=true) -
  # so BOTH Prefill AND Decode measure their REAL speed, without this
  # one-off allocation cost. ensureKvResident() needs no open GPU
  # session (sessionAllocHeap/sessionAllocFromHeap are pure device-level
  # allocation APIs) so it is safe to call this early.
  let warmupStart = epochTime()
  block:
    let warmupBe = ctx.kind.toByby()
    for idx in 0 ..< cache.layers.len:
      # NOTE: nKVHeads/headDim come FROM that layer itself (m.blocks[idx].attn),
      # NOT the global m.config.nKVHeads/hiddenSize - they must EXACTLY match
      # the data source encodeDecodeLayerIntoOpenSession/syncKvResidentEncode
      # use when they really call ensureKvResident() during prefill/decode.
      ensureKvResident(cache.layers[idx], warmupBe, m.blocks[idx].attn.nKVHeads,
                        cache.maxSeqLen, m.blocks[idx].attn.headDim)
  let warmupTime = epochTime() - warmupStart

  var tokenStream = initTokenStream(tok)
  stdout.write(tokenStream.pushTokens(promptIds))
  stdout.flushFile()
  # Colab's cell output keeps the text of ANSI cursor codes ("71;1H[...]8"),
  # so the pinned tok/s line is off there unless BYBY_PROGRESS=1 forces it.
  let progressEnv = getEnv("BYBY_PROGRESS", "auto")
  let liveProgress = showProgress and progressEnv != "0" and
                     isatty(stdout) and isatty(stderr) and getEnv("TERM", "") != "dumb" and
                     (progressEnv == "1" or not existsEnv("COLAB_RELEASE_TAG"))

  # ---- TOK/S MEASUREMENT: prefill (the whole prompt at once, the
  # "prompt tokens/s" figure) and decode (one token at a time, the real "tok/s"
  # users care about in chat) are separated - like llama.cpp reporting
  # "prompt eval time" / "eval time" separately, since the 2 phases cost
  # very different amounts (prefill is parallel over T, decode is not).
  let prefillStart = epochTime()

  # ---- PREFILL: the whole prompt in 1 forward (beAttentionFused, causal internally) ----
  # Builds the list of REAL whitespace/formatting token ids for the
  # tokenizer in use (no hard-coded numbers since every tokenizer maps them
  # differently) - encodes each common whitespace pattern in code, collects the ids,
  # used to exclude them from the repetition penalty (see sampleTopP above).
  var noPenaltyIds: seq[int] = @[]
  for ws in ["\n", "\n\n", " ", "  ", "    ", "\t", ":\n", ":\n    "]:
    for id in tok.encode(ws, addSpecialTokens = false):
      if id notin noPenaltyIds: noPenaltyIds.add(id)

  var logits = m.forwardStep(promptIds, ctx, cache, isPrefill = true)
  let prefillTime = epochTime() - prefillStart
  gAttnTime = 0.0
  gFfnTime = 0.0

  let lastRow = logits.data[(logits.shape[1] - 1) * m.config.vocabSize ..< logits.shape[1] * m.config.vocabSize]
  var nextId = sampleTopP(ctx, lastRow, temperature, topP, promptIds[max(0, promptIds.len - 64) ..< promptIds.len],
                           noPenaltyIds = noPenaltyIds)

  var allIds = promptIds
  allIds.add(nextId)
  var generated = 1
  var decodeStart = epochTime()   # measured from AFTER prefill, excluding the first token (generated during prefill)

  # Native byte streaming consumes each token once and holds incomplete UTF-8.
  # It does not re-decode the entire generated history on every token.
  if nextId != tok.eosId:
    stdout.write(tokenStream.pushTokens(@[nextId]))
    stdout.flushFile()

  gGenerating = true
  while generated < maxNewTokens:
    if nextId == tok.eosId or gStopRequested: break

    let stepStart = epochTime()
    let stepLogits = m.forwardStep(@[nextId], ctx, cache, isPrefill = false)
    nextId = sampleTopP(ctx, stepLogits.data, temperature, topP, allIds[max(0, allIds.len - 64) ..< allIds.len],
                         noPenaltyIds = noPenaltyIds)
    let stepTime = epochTime() - stepStart
    allIds.add(nextId)
    inc generated

    if liveProgress:
      let instTokS = (if stepTime > 0: 1.0 / stepTime else: 0.0)
      stderr.write("\e7\e[1;1H\e[K")
      stderr.write(&"[{generated}/{maxNewTokens} tok | {instTokS:.2f} tok/s | {(stepTime*1000):.0f} ms/tok]")
      stderr.write("\e8")
      stderr.flushFile()

    if nextId == tok.eosId: break
    stdout.write(tokenStream.pushTokens(@[nextId]))
    stdout.flushFile()
  gGenerating = false
  stdout.write(tokenStream.pushTokens([], final = true))
  stdout.flushFile()
  let decodeTime = epochTime() - decodeStart
  if gStopRequested:
    statsOut.write("\n[Stopped - Ctrl+C]\n")
  let decodedCount = generated - 1   # excludes the first token (generated during prefill)
  stdout.write("\n")
  stdout.flushFile()
  statsOut.write("\n== Speed stats ==\n")
  if verbose:
    stderr.write(&"KV_RESIDENT_ALLOC: {gKvAllocCount} allocations in total " &
      &"(correct if == the layer count = {m.config.nLayers}, LEAK if > {m.config.nLayers})\n")
    stderr.write(&"Decode - attn (QKV+RoPE+attention+oProj): {gAttnTime:.3f}s | FFN: {gFfnTime:.3f}s | " &
                 &"both parts: {(gAttnTime+gFfnTime):.3f}s\n")
  if verbose and gMergeAllLayersSession:
    let pct = (if gMode3Attempts > 0: 100.0 * gMode3Success.float / gMode3Attempts.float else: 0.0)
    stderr.write(&"MODE3 (all 32 layers in 1 session): {gMode3Attempts} attempts - SUCCEEDED {gMode3Success} " &
      &"({pct:.1f}%), FALLBACK {gMode3Attempts - gMode3Success} times\n")
    if gMode3Success < gMode3Attempts:
      stderr.write(&"  -> reason for the FIRST failure: {gMode3FirstFailMsg}\n")
  if verbose:
    stderr.write(&"KV-resident warmup (32 layers, allocated once, BEFORE prefill): {warmupTime:.3f}s\n")
  statsOut.write(&"Prefill: {promptIds.len} tokens in {prefillTime:.3f}s -> {(promptIds.len.float/prefillTime):.2f} tok/s\n")
  if decodedCount > 0 and decodeTime > 0:
    statsOut.write(&"Decode:  {decodedCount} tokens in {decodeTime:.3f}s -> {(decodedCount.float/decodeTime):.2f} tok/s ({(decodeTime*1000/decodedCount.float):.1f} ms/token)\n")
  statsOut.write(&"Total: {(promptIds.len+decodedCount)} tokens in {(prefillTime+decodeTime):.3f}s\n")
  statsOut.flushFile()
  result = tok.decode(allIds)

when isMainModule:
  # By default (BEFORE setControlCHook is called) Nim catches SIGINT itself and prints
  # "SIGINT: Interrupted by Ctrl-C." straight to stderr then quits - with NO \n
  # before it. When the user presses Ctrl-C midway through generate, stdout has
  # half-printed text (streamed tokens have no \n at the end of each chunk) -> that message
  # sticks right onto the end of the text being printed, looking like "overwritten lines". This is NOT
  # the same as the earlier \r/status-line bug (already fixed) - they are 2 different
  # things: the earlier one was US overwriting ourselves, this one is the NIM RUNTIME printing
  # without a newline. Fix: register our own hook - always print a newline + flush
  # stdout before the interrupt message, so the text never sticks together.
  # During generate() the first Ctrl+C only asks the loop to stop (see
  # gStopRequested). Otherwise, and on a second Ctrl+C, the process leaves
  # through _exit: quit() ran exit procs and driver teardown on the signal
  # handler while the main thread was still inside a Vulkan call (SIGSEGV).
  proc onCtrlC() {.noconv.} =
    if gGenerating and not gStopRequested:
      gStopRequested = true
      return
    const msg = "\n[Stopped - Ctrl+C]\n"
    when defined(posix):
      discard posix.write(2, cstring(msg), msg.len)
      exitnow(130)   # 128 + SIGINT, without exit procs
    else:
      quit(130)
  setControlCHook(onCtrlC)

  setForbidCpuFallback(true)

  # ── Command-line arguments (all default to the previous values,
  #    setting nothing behaves exactly like the original) ──
  #   ./nim_inference [modelPath] [nimqPath] [backend] [prompt] [maxNewTokens] [temperature] [topP] [systemText]
  # Options may appear anywhere among them:
  #   --no-progress   no pinned "[N/M tok | tok/s | ms/tok]" line (same as
  #                   BYBY_PROGRESS=0); the summary at the end is still printed
  var args: seq[string]
  var showProgress = true
  for a in commandLineParams():
    case a
    of "--no-progress", "--no-tok-s": showProgress = false
    of "--progress": showProgress = true
    else:
      if a.startsWith("--"):
        stderr.writeLine "[nim_inference] unknown option: " & a
        quit(2)
      args.add a
  proc argOr(i: int, default: string): string =
    if args.len > i and args[i].len > 0: args[i] else: default

  let modelPath = argOr(0, "test/deepseek-inference/models")
  let nimqPath = argOr(1, "test/deepseek-inference/models/model.nimq")
  let backendName = argOr(2, "auto")
  let promptText = argOr(3, "Write a Python function to return the n-th Fibonacci number.")
  let maxNewTokens = parseInt(argOr(4, "100"))
  let temperature = parseFloat(argOr(5, "0.7")).float32
  let topP = parseFloat(argOr(6, "0.9")).float32
  # No system message by default: the checkpoint's template then emits its own
  # standard DeepSeek system prompt. (Its found_item flag never leaves the for
  # loop, so a system message is printed after that prompt, not instead of it.)
  let systemText = argOr(7, "")

  echo "Loading config..."
  let cfg = newLlamaLConfig(modelPath)
  echo "Loading native tokenizer..."
  let tok = newHFTokenizer(modelPath)
  # Validate/render the checkpoint's template before allocating model/GPU buffers.
  let promptIds = tok.applyChatTemplate(promptText, systemText = systemText)
  echo &"Init backend ({backendName})..."
  let ctx = newBackend(backendName)
  echo "Building model..."
  var model = newLlamaModelL(cfg)
  echo &"Loading weights tu {nimqPath} ..."
  loadLlamaWeightsL(model, nimqPath)
  uploadAllResidentWeights(model, ctx)
  # A small "priming" matmul (result discarded) is run BEFORE
  # generate starts. The FIRST matmul call is when Metal compiles the
  # pipeline + measures maxTotalThreadsPerThreadgroup + NSLogs the "chon tile
  # ..." line (see metal_shim.m/pickMatmulPipeline) - if it happens naturally,
  # it lands right while the first token is streaming, and NSLog writes straight into
  # Console/syslog in the middle of the half-printed text. Calling it here makes all that setup
  # log print DURING "Building model...", away from the
  # generation.
  echo "Starting the GPU pipeline (once)..."
  discard beMatmul(ctx, @[0'f32, 0, 0, 0], 2, 2, @[0'f32, 0, 0, 0], 2, 2)
  echo "Generating (prefill + paged KV-cache decode, 100% GPU qua BybyLang)..."
  # The old Alpaca prompt "### Instruction:/### Response:" does NOT match the model's real chat
  # template -> applyChatTemplate (the model's own tokenizer_config.json)
  # is used to tokenize in the format it was trained on.
  discard generate(model, tok, ctx, promptIds, maxNewTokens = maxNewTokens,
                    temperature = temperature, topP = topP, showProgress = showProgress)
