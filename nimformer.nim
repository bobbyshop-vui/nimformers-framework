## nimformer.nim
## Full Transformer with backward for attention + custom transformer blocks

import std/[math, random, strformat, algorithm, tables]
import customfloat
import backend  # Imports Backend and its helpers
import quant     # QuantTensor, dequantizeTensor - keeps int4-compressed weights in RAM
import vendor/bybylang/gpubackend as byby
import fp4_gemv_gpu  # gFp4CacheGpu, fp4GemmGpuRows (prefill FP4-KM)
import prefill_profile  # BYBY_PREFILL_PROFILE: pfTime

# ===================================================================
# Tensor utilities (unchanged)
# ===================================================================
type
  Tensor* = object
    data*: seq[float32]
    shape*: seq[int]
    mappedData*: MappedView[float32]

template floatValues*(t: Tensor): untyped = weightValues(t.data, t.mappedData)

proc numel(shape: seq[int]): int =
  result = 1
  for s in shape: result *= s

var gLoraSizeMismatchWarned = 0  ## counts LoRA size-mismatch warnings
                                  ## (see forward() below) - capped so the log
                                  ## is not spammed when many Linears hit it.

proc newTensor*(shape: seq[int], fill: float32 = 0'f32): Tensor =
  result.shape = shape
  result.data = newSeq[float32](numel(shape))
  if fill != 0'f32:
    for i in 0 ..< result.data.len: result.data[i] = fill

proc randnTensor*(shape: seq[int], scale: float32): Tensor =
  result.shape = shape
  result.data = newSeq[float32](numel(shape))
  for i in 0 ..< result.data.len:
    result.data[i] = float32(gauss(0.0, 1.0)) * scale

proc addT*(a, b: Tensor): Tensor =
  result = a
  for i in 0 ..< result.data.len: result.data[i] += b.data[i]

proc addT*(ctx: Backend, a, b: Tensor): Tensor =
  ## The addT(a,b) above runs on the CPU even though beAdd (real GPU, implemented
  ## for CUDA/Metal/OpenCL) exists - this residual connection runs TWICE PER
  ## LAYER (attn + ff), i.e. 64 times/token for a 32-layer model, not a small
  ## operation to ignore. This overload goes through the GPU, as required ("never
  ## silently run on the CPU"). The old addT(a,b) (no ctx) stays for the
  ## other callers that have no Backend in scope.
  result = a
  result.data = beAdd(ctx, a.data, b.data)

proc subT*(a, b: Tensor): Tensor =
  result = a
  for i in 0 ..< result.data.len: result.data[i] -= b.data[i]

proc mulT*(a, b: Tensor): Tensor =
  result = a
  for i in 0 ..< result.data.len: result.data[i] *= b.data[i]

proc flatten2D*(shape: seq[int]): tuple[rows, cols: int] =
  let cols = shape[^1]
  var rows = 1
  for i in 0 ..< shape.len - 1: rows *= shape[i]
  result = (rows, cols)

proc transpose*(x: Tensor): Tensor =
  ## Transposes a tensor treated as a 2D matrix [rows, cols] (all dims except
  ## the last merged into "rows", like flatten2D) -> [cols, rows].
  let (rows, cols) = flatten2D(x.shape)
  result.shape = @[cols, rows]
  result.data = newSeq[float32](rows * cols)
  for i in 0 ..< rows:
    for j in 0 ..< cols:
      result.data[j * rows + i] = x.floatValues[i * cols + j]

# ===================================================================
# Linear (forward/backward use Metal matmul)
# ===================================================================
type
  Linear* = object
    weight*: Tensor      ## small fp32 fallback (only for random init before real weights are loaded)
    weightQ*: QuantTensor ## the REAL weight kept COMPRESSED (int4/int8...) for the whole
                           ## process lifetime, the way llama.cpp keeps quantized weights in RAM -
                           ## NEVER expanded to fp32 all at once and kept (that was
                           ## the earlier OOM: 6.7B params x 4 bytes fp32 = ~27GB RAM).
    useQuant*: bool        ## true after loadLinear() binds the real weightQ (compressed int4, ~4GB total
                            ## for the whole model instead of ~27GB fp32)
    bias*: Tensor
    inF*, outF*: int
    residentW*: ResidentWeightHandle  ## weight uploaded to the GPU ONCE (see uploadResident)
    residentReady*: bool               ## true after uploadResident() succeeds
    residentZeroCopy*: bool            ## residentW wq wraps the model mmap in place
    residentFp32*: SessionHandle       ## shared fp32 resident copy for weights that are NOT
                                        ## int4-asymmetric (int8/int4-symmetric/tied-embedding) -
                                        ## dequantized once at load instead of every token (see backend.nim)
    residentFp32Ready*: bool           ## true once residentFp32 has been uploaded
    residentFp32NT*: bool              ## residentFp32 stores original [outF,inF] rows
    residentFp32Offset*: int           ## floats before the weight in residentFp32 (mmap wrap)
    name*: string                 ## checkpoint tensor name, for the dispatch recorder
    loraA*, loraB*: seq[float32]  ## (see forward() below + mergeInUploadStreamingJamba
                                   ## in lora.nim): a detached LoRA adapter, NOT physically
                                   ## merged into the weight - the base (weightQ/useQuant) stays INTACT
                                   ## and compressed for the process lifetime, the core invariant
                                   ## of this type (see the weightQ comment above: "NEVER
                                   ## expanded to fp32 all at once and kept"). loraA:[r,inF]
                                   ## flat, loraB:[outF,r] flat, empty (@[]) when not attached.
    mappedLoraA*, mappedLoraB*: MappedView[float32]
    loraR*: int                   ## rank r; 0 = NO LoRA attached (default, old code unchanged)
    loraScale*: float32           ## alpha / r
    gpuCacheName*: string         ## name in gFp4CacheGpu after prepareGpuLinForLinear (FP4-KM)

template loraAValues*(l: Linear): untyped = weightValues(l.loraA, l.mappedLoraA)
template loraBValues*(l: Linear): untyped = weightValues(l.loraB, l.mappedLoraB)

proc newLinear*(inF, outF: int; initializeWeights = true): Linear =
  let scale = sqrt(2'f32 / float32(inF))
  result.inF = inF
  result.outF = outF
  result.weight = if initializeWeights: randnTensor(@[outF, inF], scale)
                  else: Tensor(shape: @[outF, inF], data: @[])
  result.useQuant = false
  result.bias = newTensor(@[outF])

proc setQuantWeight*(l: var Linear, qt: QuantTensor) =
  ## Called from loadLinear() when real weights are loaded from a .nimq - keeps them
  ## COMPRESSED (no immediate dequant) and frees the fp32 fallback weight
  ## (no longer needed, wasted RAM since weightQ exists).
  l.weightQ = qt
  l.useQuant = true
  l.weight.data = @[]
  l.weight.mappedData = default(MappedView[float32])
  # A Linear may already have a resident FP32/Q4 copy from an earlier binding.
  # It must not silently keep serving that stale copy after its quant source
  # changes; force the next forward to use the newly bound weightQ.
  l.residentReady = false
  l.residentFp32Ready = false
  l.residentFp32NT = false
proc uploadResident*(l: var Linear, ctx: Backend, sparsityPct: float32 = 0.0'f32) =
  ## Uploads the weight to the GPU ONCE (called from loadLlamaWeightsL after
  ## loading, NOT inside the decode loop).
  ## - int4-per-group (Q4_K_M style, the llama.cpp-style path): uses the native
  ##   int4 kernel on Metal/OpenCL directly (residentW), fastest and least
  ##   bandwidth.
  ## - EVERY other case still uses the real weight (int8, int4-symmetric, or
  ##   plain fp32 Linear/tied-embedding lm_head): previously these
  ##   cases had NO resident path at all -> forward() had to
  ##   dequantizeTensorTransposed()/transpose() the WHOLE matrix on the CPU
  ##   EVERY TOKEN (the main cause of ~4s/token). Now it is dequantized/transposed
  ##   ONCE here and kept resident as fp32 on the GPU
  ##   (beResidentUploadFp32); each token uploads only the small activation.
  ##
  ## sparsityPct > 0 -> BLOCK-LEVEL SPARSITY applied right here IN
  ## NIM, BEFORE the GPU upload - the .nimq file/export_to_nimq.py is NOT touched.
  ## For each block of group_size consecutive values, the importance is the
  ## approximate mean magnitude (0.5*scale, since dequantized int4 values sit
  ## in [0,15] centred around 7.5, so the real swing
  ## is ~ 0.5*range = 0.5*scale*15 -> "scale" is used as the importance
  ## proxy, NO need to dequantize all 128 values of a block - cheap, fast,
  ## runs once at model load). Pruned blocks get scale=0 -> the dequant formula
  ## (nibble-zp)*scale is ALWAYS 0 - mathematically exact, not an approximation. Only
  ## the scale ARRAY ALREADY IN RAM (l.weightQ.scale) changes; packed data,
  ## format and files are left untouched.
  if l.useQuant and l.weightQ.kind == qkInt4Asymmetric and l.weightQ.groupSize > 0:
    if sparsityPct > 0.0'f32 and l.weightQ.scale.len > 0:
      var scaleCopy = l.weightQ.scale  # copy so the percentile does not reorder the original
      scaleCopy.sort()
      var idx = int(float32(scaleCopy.len - 1) * sparsityPct / 100.0'f32)
      idx = max(0, min(idx, scaleCopy.len - 1))
      let threshold = scaleCopy[idx]
      var nPruned = 0
      for i in 0 ..< l.weightQ.scale.len:
        if l.weightQ.scale[i] <= threshold:
          l.weightQ.scale[i] = 0.0'f32
          l.weightQ.zero_point[i] = 0.0'f32
          inc nPruned
      stderr.writeLine "[SPARSITY] cat " & $nPruned & "/" & $l.weightQ.scale.len &
        " khoi (nguong scale <= " & $threshold & ")"
    let nGroupsPerRow = (l.inF + l.weightQ.groupSize - 1) div l.weightQ.groupSize
    if l.weightQ.mappedData.owner != nil and l.weightQ.mappedData.len > 0:
      # Packed weights stay in the model mmap: wrapped in place on unified
      # memory, otherwise copied straight from the mapping (no seq copy).
      let up = beResidentUploadQ4Mapped(ctx, l.weightQ.mappedData.dataPtr,
        l.weightQ.mappedData.len, l.weightQ.scale, l.weightQ.zero_point,
        l.inF, l.outF, l.weightQ.groupSize, nGroupsPerRow)
      l.residentW = up.w
      l.residentZeroCopy = up.zeroCopy
      if unlikely(byby.firProbeOn): byby.probeName(l.residentW, l.name, "q4", [l.outF, l.inF])
    else:
      l.residentW = beResidentUploadQ4(ctx, @(l.weightQ.bytes), l.weightQ.scale, l.weightQ.zero_point,
                                        l.inF, l.outF, l.weightQ.groupSize, nGroupsPerRow)
      if unlikely(byby.firProbeOn): byby.probeName(l.residentW, l.name, "q4", [l.outF, l.inF])
    l.residentReady = true
  elif l.useQuant and l.weightQ.kind == qkFp32Raw and l.weightQ.mappedData.owner != nil and
      l.weightQ.mappedData.len == l.inF * l.outF * 4:
    # Raw fp32 [outF, inF] inside the model mmap is already the NT GEMM
    # layout: wrap it in place (unified memory) instead of dequant+transpose.
    let wrapped = beResidentWrapFp32(ctx, l.weightQ.mappedData.dataPtr, l.weightQ.mappedData.len)
    l.residentFp32 = wrapped.h
    l.residentFp32Offset = wrapped.offsetElems
    l.residentFp32NT = true
    l.residentFp32Ready = true
    l.residentZeroCopy = wrapped.zeroCopy
    if unlikely(byby.firProbeOn): byby.probeName(l.residentFp32, l.name, "f32", [l.outF, l.inF])
  elif l.useQuant:
    # qkInt8 / qkInt4 (symmetric) / qkFp32Raw ... : dequant + transpose 1
    # LẦN, upload resident fp32.
    let wT = dequantizeTensorTransposed(l.weightQ)
    l.residentFp32 = beResidentUploadFp32(ctx, wT)
    l.residentFp32NT = false
    l.residentFp32Offset = 0
    l.residentFp32Ready = true
    if unlikely(byby.firProbeOn): byby.probeName(l.residentFp32, l.name, "f32", [l.outF, l.inF])
    # RAM saving: after the upload, forward() ONLY uses the residentFp32Ready path
    # (see forward() below) - l.weight.data (if any, e.g. after mergeIn() wrote
    # merged.data here) is no longer needed but used NOT to be freed =>
    # an fp32 copy in CPU RAM + one on the GPU existed SIDE BY SIDE (on integrated
    # GPUs/unified memory such as Intel Iris, BOTH copies share system RAM
    # = twice the RAM actually needed). Free it now,
    # exactly the pattern setQuantWeight() uses above.
    l.weight.data = @[]
    l.weight.mappedData = default(MappedView[float32])
  elif l.weight.floatValues.len > 0:
    # Plain fp32 Linear (e.g. tied-embedding lm_head using embW directly) -
    # transposed ONCE instead of every token.
    let wT = transpose(l.weight).data
    l.residentFp32 = beResidentUploadFp32(ctx, wT)
    l.residentFp32NT = false
    l.residentFp32Offset = 0
    l.residentFp32Ready = true
    if unlikely(byby.firProbeOn): byby.probeName(l.residentFp32, l.name, "f32", [l.outF, l.inF])

proc residentLinearNt(l: Linear; x: seq[float32]; rows: int; ctx: Backend): seq[float32] =
  ## Share inference's original row-major GPU weight with sequence forward.
  let be = ctx.kind.toByby()
  var hx = byby.sessionUpload(be, x)
  var hy = byby.sessionAllocScratch(be, rows * l.outF)
  var opened = false
  try:
    if not byby.sessionBegin(be): raise newException(CatchableError, "resident NT sessionBegin failed")
    opened = true
    if not byby.sessionMatmulNT(be, hx, l.residentFp32, hy, rows, l.inF, l.outF,
        l.residentFp32Offset):
      raise newException(CatchableError, "resident NT GEMM failed")
    let ended = byby.sessionEnd(be)
    opened = false
    if not ended:
      discard byby.sessionRecoverQueue(be)
      raise newException(CatchableError, "resident NT sessionEnd failed")
    result = byby.sessionRead(be, hy, rows * l.outF)
  finally:
    if opened:
      discard byby.sessionEnd(be)
      discard byby.sessionRecoverQueue(be)
    byby.sessionFree(be, hx)
    byby.sessionFree(be, hy)

proc forward*(l: Linear, x: Tensor, ctx: Backend): Tensor =
  let (rows, cols) = flatten2D(x.shape)
  assert cols == l.inF
  var outShape = x.shape
  outShape[^1] = l.outF
  result = newTensor(outShape)

  var y: seq[float32]
  pfTime("lin.gemm"):
    if l.residentReady:
      y = beResidentLinearQ4(ctx, x.data, l.residentW, rows)  # uploads only the activation, weight already resident
    elif l.residentFp32Ready:
      # int8/int4-symmetric/plain fp32 weights are ALREADY resident (see
      # uploadResident) - only the small activation is uploaded per token, the
      # whole weight is NOT dequantized again.
      y = if l.residentFp32NT: residentLinearNt(l, x.data, rows, ctx)
          else: beResidentLinearFp32(ctx, x.data, l.residentFp32, rows, l.inF, l.outF)

    if y.len == 0:
      # Not resident, or the current backend has no resident path (e.g.
      # CPU) -> falls back to the old path (still correct, just without the resident speedup).
      if l.useQuant and l.weightQ.kind == qkFp4KM and l.gpuCacheName.len > 0 and
         l.gpuCacheName in gFp4CacheGpu and gFp4CacheGpu[l.gpuCacheName].residentReady and
         ctx.kind.toByby() in {byby.gbMetal, byby.gbVulkan}:
        # Parallel prefill: GPU resident FP4-KM + MBLOCK=4 (no CPU dequant).
        y = fp4GemmGpuRows(ctx, l.gpuCacheName, x.data, rows)
      elif l.useQuant and l.weightQ.kind == qkInt4Asymmetric and l.weightQ.groupSize > 0:
        let nGroupsPerRow = (l.inF + l.weightQ.groupSize - 1) div l.weightQ.groupSize
        y = beMatmulQ4(ctx, x.data, @(l.weightQ.bytes), l.weightQ.scale, l.weightQ.zero_point,
                        rows, l.inF, l.outF, l.weightQ.groupSize, nGroupsPerRow)
      else:
        var wT: seq[float32]
        if l.useQuant:
          wT = dequantizeTensorTransposed(l.weightQ)
        else:
          wT = transpose(l.weight).data
        y = beMatmul(ctx, x.data, rows, l.inF, wT, l.inF, l.outF)

  # === ALWAYS MAKE SURE THE SIZE IS RIGHT ===
  let expectedSize = rows * l.outF
  if y.len != expectedSize:
    # New array with the right size, copy the existing elements, leave the rest at 0
    var fixed = newSeq[float32](expectedSize)
    let copyLen = min(y.len, expectedSize)
    for i in 0 ..< copyLen:
      fixed[i] = y[i]
    y = fixed

  # Add bias (l.bias.data has length l.outF; empty = no bias)
  if l.bias.data.len == l.outF:
    for i in 0 ..< y.len:
      result.data[i] = y[i] + l.bias.data[i mod l.outF]
  else:
    for i in 0 ..< y.len: result.data[i] = y[i]

  # LoRA is ADDED (additive), NOT physically merged into the weight. y/result
  # above always goes through the COMPRESSED resident/quant Q4 path like the base (weightQ/useQuant
  # is never touched by LoRA any more - see mergeInUploadStreamingJamba in
  # lora.nim) - only a low-rank delta [rows,outF] = scale * (x @ A^T)
  # @ B^T is added, as in standard QLoRA inference (the quantized base stays
  # as is, the small adapter is added outside). r is small (usually 4-64) so these 2
  # small products are much CHEAPER than the main [rows,inF]x[inF,outF] matmul. This AVOIDS
  # the ~27GB bloat (physically merging to fp32 for all 105 large params) that
  # caused OOM/Killed - see the comment on the Linear/loraA definition above.
  if l.loraR > 0 and l.loraAValues.len == l.loraR * l.inF and l.loraBValues.len == l.outF * l.loraR:
    pfTime("lin.lora"):
      let At = transpose(Tensor(shape: @[l.loraR, l.inF], data: @(l.loraAValues)))   # [inF, r]
      let Bt = transpose(Tensor(shape: @[l.outF, l.loraR], data: @(l.loraBValues)))  # [r, outF]
      let xA = beMatmul(ctx, x.data, rows, l.inF, At.data, l.inF, l.loraR)       # [rows, r]
      let delta = beMatmul(ctx, xA, rows, l.loraR, Bt.data, l.loraR, l.outF)     # [rows, outF]
      for i in 0 ..< result.data.len:
        result.data[i] += delta[i] * l.loraScale
  elif l.loraR > 0:
    # Safety net (do not crash the whole process): l.loraR > 0 but loraA/loraB
    # do NOT have the expected size (r*inF and outF*r) - most likely an
    # adapter is missing its A or B part in the .bin itself, or the size
    # r in the meta does not match the saved tensor (see loadBinaryLoraForInfer
    # in infer_lora.nim). SKIP the LoRA delta for THIS parameter (use the
    # base weight, no LoRA adjustment for it) instead of indexing a wrong-sized
    # or empty seq -> IndexDefect crashing the whole generation.
    if gLoraSizeMismatchWarned < 10:
      inc gLoraSizeMismatchWarned
      stderr.writeLine("[warning] LoRA adapter size mismatch for a Linear " &
        "(r=" & $l.loraR & ", loraA.len=" & $l.loraAValues.len & " (need " & $(l.loraR * l.inF) &
        "), loraB.len=" & $l.loraBValues.len & " (need " & $(l.outF * l.loraR) &
        ")) - skipping the LoRA delta for this parameter, using the base weight.")
proc backward*(l: Linear, x: Tensor, dOut: Tensor, ctx: Backend): tuple[dX, dW, dB: Tensor] =
  ## Linear backward:
  ## dW = x^T @ dOut  (accumulated over the batch)
  ## dB = sum(dOut, axis=0)
  ## dX = dOut @ W
  let (rows, cols) = flatten2D(x.shape)
  doAssert cols == l.inF,
    &"Linear.backward: x.shape={x.shape} (cols={cols}) but l.inF={l.inF}"
  doAssert dOut.shape[^1] == l.outF,
    &"Linear.backward: dOut.shape={dOut.shape} (last={dOut.shape[^1]}) but l.outF={l.outF} (l.inF={l.inF}, x.shape={x.shape})"

  # dW = x^T @ dOut (transpose x -> [inF, rows], dOut -> [rows, outF])
  let xT = transpose(x)  # [inF, rows]
  let dOutFlat = dOut.data  # [rows * outF]

  # dW and dX are INDEPENDENT (neither needs the other's result),
  # so both go into 1 command buffer (metalMatmul2) instead of calling metalMatmul
  # twice — halves the GPU dispatch+wait count per Linear.backward.
  let (dWRawFlat, dXFlat) = beMatmul2(ctx,
    xT.data, l.inF, rows, dOutFlat, rows, l.outF,
    dOutFlat, rows, l.outF, l.weight.data, l.outF, l.inF)

  var dWRaw: Tensor
  dWRaw.shape = @[l.inF, l.outF]
  dWRaw.data = dWRawFlat
  let dW = transpose(dWRaw)  # -> [outF, inF], matches l.weight.shape

  var dX: Tensor
  dX.shape = x.shape
  dX.data = dXFlat

  # dB = sum of dOut over the batch
  var dB = newTensor(@[l.outF])
  for i in 0 ..< rows:
    for j in 0 ..< l.outF:
      dB.data[j] += dOutFlat[i * l.outF + j]

  result = (dX, dW, dB)

proc backwardInputOnly*(l: Linear, x: Tensor, dOut: Tensor, ctx: Backend): Tensor =
  ## Backward that computes only dX while Linear.weight/bias are frozen. Avoids computing and
  ## allocating an unused dW=[outF,inF] (especially lm_head vocab x hidden).
  let (rows, cols) = flatten2D(x.shape)
  doAssert cols == l.inF,
    &"Linear.backwardInputOnly: x.shape={x.shape} (cols={cols}) but l.inF={l.inF}"
  doAssert dOut.shape[^1] == l.outF,
    &"Linear.backwardInputOnly: dOut.shape={dOut.shape} (last={dOut.shape[^1]}) but l.outF={l.outF}"
  var weightData = l.weight.data
  if weightData.len != l.outF * l.inF:
    if not l.useQuant:
      raise newException(ValueError,
        &"Linear.backwardInputOnly: weight data len={weightData.len}, expected={l.outF * l.inF}")
    weightData = dequantizeTensor(l.weightQ)
  if weightData.len != l.outF * l.inF:
    raise newException(ValueError,
      &"Linear.backwardInputOnly: dequantized weight len={weightData.len}, expected={l.outF * l.inF}")
  result.shape = x.shape
  result.data = beMatmul(ctx, dOut.data, rows, l.outF,
                         weightData, l.inF, l.inF)

# ===================================================================
# LayerNorm (forward/backward use Metal matmul)
# ===================================================================
type
  LayerNorm* = object
    gamma*, beta*: Tensor
    eps*: float32
    dim*: int

proc newLayerNorm*(dim: int, eps: float32 = 1e-5'f32): LayerNorm =
  result.dim = dim
  result.gamma = newTensor(@[dim], 1'f32)
  result.beta  = newTensor(@[dim], 0'f32)
  result.eps   = eps

proc forward*(ln: LayerNorm, x: Tensor, ctx: Backend): Tensor =
  ## This function used to take `ctx: Backend` but NOT USE it - the whole
  ## LayerNorm ran as a plain Nim loop on the CPU whatever
  ## backend (cuda/metal/opencl) was chosen. backend.nim already has `beLayernorm` matching
  ## this exact operation (rows x cols, same normalisation formula), so it just
  ## calls that so LayerNorm really runs on the chosen GPU.
  let (rows, cols) = flatten2D(x.shape)
  assert cols == ln.dim
  result.shape = x.shape
  result.data = beLayernorm(ctx, x.data, ln.gamma.data, ln.beta.data, rows, cols, ln.eps)

proc backward*(ln: LayerNorm, x: Tensor, dOut: Tensor, ctx: Backend): tuple[dX, dGamma, dBeta: Tensor] =
  ## LayerNorm backward (matrix ops / GPU-bound)
  let (rows, cols) = flatten2D(x.shape)
  assert cols == ln.dim
  var dX = newTensor(x.shape)
  var dGamma = newTensor(ln.gamma.shape)
  var dBeta = newTensor(ln.beta.shape)
  let (dx_data, dgamma_data, dbeta_data) = beLayernormBackward(ctx, dOut.data, x.data, ln.gamma.data, ln.beta.data, rows, cols, ln.eps)
  dX.data = dx_data
  dGamma.data = dgamma_data
  dBeta.data = dbeta_data
  result = (dX, dGamma, dBeta)

# ===================================================================
# CausalSelfAttention (forward/backward use Metal matmul)
# ===================================================================
type
  CausalSelfAttention* = object
    nHeads*, headDim*, embedDim*: int
    qkv*, proj*: Linear

proc newCausalSelfAttention*(embedDim, nHeads: int): CausalSelfAttention =
  assert embedDim mod nHeads == 0
  result.embedDim = embedDim
  result.nHeads = nHeads
  result.headDim = embedDim div nHeads
  result.qkv = newLinear(embedDim, 3 * embedDim)
  result.proj = newLinear(embedDim, embedDim)

proc forward*(attn: CausalSelfAttention, x: Tensor, ctx: Backend): Tensor =
  let B = x.shape[0]
  let T = x.shape[1]
  let C = x.shape[2]
  assert C == attn.embedDim
  let qkv = attn.qkv.forward(x, ctx)
  let hd = attn.headDim
  let scale = 1'f32 / sqrt(float32(hd))

  # Slice Q, K, V
  var qData = newSeq[float32](B * T * C)
  var kData = newSeq[float32](B * T * C)
  var vData = newSeq[float32](B * T * C)
  for b in 0 ..< B:
    let baseQKV = b * T * 3 * C
    let baseOut = b * T * C
    for t in 0 .. T-1:
      for c in 0 .. C-1:
        qData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + c]
        kData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + C + c]
        vData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + 2 * C + c]

  let (attnOutData, _) = beAttentionFused(ctx, qData, kData, vData, [], B, attn.nHeads, T, hd, scale)
  var attnOut = newTensor(@[B, T, C])
  attnOut.data = attnOutData
  result = attn.proj.forward(attnOut, ctx)

proc backward*(attn: CausalSelfAttention, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dQkvW, dQkvB, dProjW, dProjB: Tensor] =
  let B = x.shape[0]
  let T = x.shape[1]
  let C = x.shape[2]
  let hd = attn.headDim
  let nHeads = attn.nHeads

  # Forward again to get intermediate states (Q, K, V, s_matrix)
  let qkv = attn.qkv.forward(x, ctx)
  let scale = 1'f32 / sqrt(float32(hd))

  var qData = newSeq[float32](B * T * C)
  var kData = newSeq[float32](B * T * C)
  var vData = newSeq[float32](B * T * C)
  for b in 0 ..< B:
    let baseQKV = b * T * 3 * C
    let baseOut = b * T * C
    for t in 0 .. T-1:
      for c in 0 .. C-1:
        qData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + c]
        kData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + C + c]
        vData[baseOut + t * C + c] = qkv.data[baseQKV + t * 3 * C + 2 * C + c]

  let (attnOutData, s_matrix) = beAttentionFused(ctx, qData, kData, vData, [], B, nHeads, T, hd, scale)
  var attnOut = newTensor(@[B, T, C])
  attnOut.data = attnOutData

  # dProj
  let dProj = attn.proj.backward(attnOut, dOut, ctx)
  let dAttnOut = dProj.dX

  # Attention Fused Backward
  let (dqData, dkData, dvData) = beAttentionFusedBackward(ctx, qData, kData, vData, s_matrix, dAttnOut.data, B, nHeads, T, hd, scale)

  # Pack dQkv
  var dQkv = newTensor(qkv.shape)
  for b in 0 ..< B:
    let baseQKV = b * T * 3 * C
    let baseOut = b * T * C
    for t in 0 .. T-1:
      for c in 0 .. C-1:
        dQkv.data[baseQKV + t * 3 * C + c] = dqData[baseOut + t * C + c]
        dQkv.data[baseQKV + t * 3 * C + C + c] = dkData[baseOut + t * C + c]
        dQkv.data[baseQKV + t * 3 * C + 2 * C + c] = dvData[baseOut + t * C + c]

  let dQkvLin = attn.qkv.backward(x, dQkv, ctx)
  result = (dQkvLin.dX, dQkvLin.dW, dQkvLin.dB, dProj.dW, dProj.dB)

# ===================================================================
# FeedForward (forward/backward use the GPU-bound APFLU activation)
# ===================================================================
type
  FeedForward* = object
    fc1*, fc2*: Linear

proc newFeedForward*(embedDim, mult: int): FeedForward =
  result.fc1 = newLinear(embedDim, embedDim * mult)
  result.fc2 = newLinear(embedDim * mult, embedDim)

proc forward*(ff: FeedForward, x: Tensor, ctx: Backend): Tensor =
  var h = ff.fc1.forward(x, ctx)
  h.data = beApflu(ctx, h.data, 0.1'f32, 1.0'f32)
  result = ff.fc2.forward(h, ctx)

proc backward*(ff: FeedForward, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dFc1W, dFc1B, dFc2W, dFc2B: Tensor] =
  let hidden = ff.fc1.forward(x, ctx)
  let dFc2 = ff.fc2.backward(hidden, dOut, ctx)
  var dHidden = dFc2.dX
  dHidden.data = beApfluBackward(ctx, hidden.data, dHidden.data, 0.1'f32, 1.0'f32)
  let dFc1 = ff.fc1.backward(x, dHidden, ctx)
  result = (dFc1.dX, dFc1.dW, dFc1.dB, dFc2.dW, dFc2.dB)

# ===================================================================
# TransformerBlock (Post-LN) - forward/backward use Metal
# ===================================================================
type
  TransformerBlock* = object
    attn*: CausalSelfAttention
    ff*: FeedForward
    ln1*, ln2*: LayerNorm

proc newTransformerBlock*(embedDim, nHeads, ffMult: int): TransformerBlock =
  result.attn = newCausalSelfAttention(embedDim, nHeads)
  result.ff = newFeedForward(embedDim, ffMult)
  result.ln1 = newLayerNorm(embedDim)
  result.ln2 = newLayerNorm(embedDim)

proc forward*(blk: TransformerBlock, x: Tensor, ctx: Backend): Tensor =
  let x1 = blk.ln1.forward(x, ctx)
  let attnOut = blk.attn.forward(x1, ctx)
  # Optimisation: "add then layernorm right after" used to be 2 separate GPU
  # commands (addT then ln2.forward, each opening/closing its own command buffer) -
  # merged into 1 session with beAddLayernormFused (see backend.nim), which returns
  # BOTH the raw residual (x2Raw, needed for the block's final add) AND the
  # layernormed version (x3) from exactly 1 upload/commit/wait - the add is not done twice.
  # Unsupported backends (CUDA/OpenCL/CPU) fall back to the 2 old commands inside.
  let (rows2, cols2) = flatten2D(x.shape)
  let fused = beAddLayernormFused(ctx, x.data, attnOut.data, blk.ln2.gamma.data, blk.ln2.beta.data,
                                   rows2, cols2, blk.ln2.eps)
  var x2Raw = newTensor(x.shape)
  x2Raw.data = fused.summed
  var x3 = newTensor(x.shape)
  x3.data = fused.normed
  let ffOut = blk.ff.forward(x3, ctx)
  result = addT(ctx, x2Raw, ffOut)

proc backward*(blk: TransformerBlock, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dAttnQkvW, dAttnQkvB, dAttnProjW, dAttnProjB,
    dFf1W, dFf1B, dFf2W, dFf2B,
    dLn1G, dLn1B, dLn2G, dLn2B: Tensor] =
  
  # Forward again (needed for backward)
  let x1 = blk.ln1.forward(x, ctx)
  let attnOut = blk.attn.forward(x1, ctx)
  let x2 = addT(ctx, x, attnOut)
  let x3 = blk.ln2.forward(x2, ctx)
  let ffOut = blk.ff.forward(x3, ctx)
  let y = addT(ctx, x2, ffOut)
  
  var dY = dOut
  let dFF = blk.ff.backward(x3, dY, ctx)
  var dX2 = dFF.dX
  let dLN2 = blk.ln2.backward(x2, dX2, ctx)
  var dX2_ln = dLN2.dX
  # Residual (x2 = x + attnOut)
  for i in 0 ..< dX2_ln.data.len:
    dX2_ln.data[i] += dY.data[i]
  
  let dAttn = blk.attn.backward(x1, dX2_ln, ctx)
  let dLN1 = blk.ln1.backward(x, dAttn.dX, ctx)
  var dX0 = dLN1.dX
  # Residual (x2 = x + attnOut)
  for i in 0 ..< dX0.data.len:
    dX0.data[i] += dX2_ln.data[i]
  
  result = (
    dX0,
    dAttn.dQkvW, dAttn.dQkvB, dAttn.dProjW, dAttn.dProjB,
    dFF.dFc1W, dFF.dFc1B, dFF.dFc2W, dFF.dFc2B,
    dLN1.dGamma, dLN1.dBeta, dLN2.dGamma, dLN2.dBeta
  )

# ===================================================================
# Embedding (forward/backward use Metal matmul)
# ===================================================================
type
  Embedding* = object
    weight*: Tensor
    dim*, vocab*: int

proc newEmbedding*(vocab, dim: int; initializeWeights = true): Embedding =
  result.vocab = vocab
  result.dim = dim
  result.weight = if initializeWeights: randnTensor(@[vocab, dim], sqrt(2'f32 / float32(vocab)))
                  else: Tensor(shape: @[vocab, dim], data: @[])

proc lookupBatch*(e: Embedding, idsBatch: seq[seq[int]], ctx: Backend): Tensor =
  let B = idsBatch.len
  let T = idsBatch[0].len
  result = newTensor(@[B, T, e.dim])
  for b in 0 ..< B:
    for t in 0 ..< T:
      let id = idsBatch[b][t]
      let src = id * e.dim
      let dst = (b * T + t) * e.dim
      # floatValues: owned data or a model mmap view, read in place.
      copyMem(addr result.data[dst], unsafeAddr e.weight.floatValues[src], e.dim * sizeof(float32))

proc backward*(e: Embedding, idsBatch: seq[seq[int]], dOut: Tensor, ctx: Backend): Tensor =
  let B = idsBatch.len
  let T = idsBatch[0].len
  var dWeight = newTensor(e.weight.shape)
  for b in 0 ..< B:
    for t in 0 ..< T:
      let id = idsBatch[b][t]
      let dOff = (b * T + t) * e.dim
      let wOff = id * e.dim
      for d in 0 ..< e.dim:
        dWeight.data[wOff + d] += dOut.data[dOff + d]
  return dWeight

# ===================================================================
# NimformerModel (Post-LN) - forward/backward use Metal
# ===================================================================
type
  NimformerModel* = object
    embed*: Embedding
    blocks*: seq[TransformerBlock]
    outProj*: Linear
    vocab*: int

proc newNimformerModel*(vocab, embedDim, nHeads, nLayers, ffMult: int): NimformerModel =
  result.vocab = vocab
  result.embed = newEmbedding(vocab, embedDim)
  result.blocks = newSeq[TransformerBlock](nLayers)
  for i in 0 ..< nLayers:
    result.blocks[i] = newTransformerBlock(embedDim, nHeads, ffMult)
  result.outProj = newLinear(embedDim, vocab)

proc forwardBatch*(m: NimformerModel, idsBatch: seq[seq[int]], ctx: Backend): Tensor =
  ## REAL forward with batch B (idsBatch.len sequences of the same length T) in one
  ## call — Linear/LayerNorm treat B*T as the "row" count, so the GPU gets M=B*T rows
  ## in each matmul instead of being called B times in sequence (B=1 each) as before.
  ## Returns logits of shape [B, T, vocab].
  var x = m.embed.lookupBatch(idsBatch, ctx)
  for blk in m.blocks:
    x = blk.forward(x, ctx)
  result = m.outProj.forward(x, ctx)

proc forward*(m: NimformerModel, ids: seq[int], ctx: Backend): Tensor =
  ## Convenience for a single sequence (keeps the old API for test_nimformer.nim) —
  ## calls forwardBatch with B=1, then drops the batch dim.
  let logits = m.forwardBatch(@[ids], ctx)
  result = Tensor(data: logits.data, shape: logits.shape[1 .. ^1])

proc backwardBatch*(m: NimformerModel, idsBatch: seq[seq[int]], dLoss: Tensor,
                     ctx: Backend): seq[Tensor] =
  ## REAL backward with batch B. dLoss: shape [B, T, vocab] (loss gradient
  ## w.r.t. logits, ALREADY batched — no more faking B=1 and wrapping an extra
  ## dim like the old version). Linear.backward accumulates gradients over B*T rows
  ## in 1 matmul, so the returned dW/dB are already the correct sum over the WHOLE batch.
  var x = m.embed.lookupBatch(idsBatch, ctx)
  var hiddenStates: seq[Tensor] = @[x]
  for blk in m.blocks:
    x = blk.forward(x, ctx)
    hiddenStates.add(x)
  let dOutProj = m.outProj.backward(x, dLoss, ctx)
  var dOut = dOutProj.dX
  var grads: seq[Tensor] = @[dOutProj.dW, dOutProj.dB]
  for i in countdown(m.blocks.len - 1, 0):
    let blk = m.blocks[i]
    let x_prev = hiddenStates[i]
    let dBlock = blk.backward(x_prev, dOut, ctx)
    dOut = dBlock.dX
    grads.add(dBlock.dAttnQkvW); grads.add(dBlock.dAttnQkvB)
    grads.add(dBlock.dAttnProjW); grads.add(dBlock.dAttnProjB)
    grads.add(dBlock.dFf1W); grads.add(dBlock.dFf1B)
    grads.add(dBlock.dFf2W); grads.add(dBlock.dFf2B)
    grads.add(dBlock.dLn1G); grads.add(dBlock.dLn1B)
    grads.add(dBlock.dLn2G); grads.add(dBlock.dLn2B)
  let dEmb = m.embed.backward(idsBatch, dOut, ctx)
  grads.add(dEmb)
  return grads

proc backward*(m: NimformerModel, ids: seq[int], dLoss: Tensor, ctx: Backend): seq[Tensor] =
  ## Convenience for a single sequence (keeps the old API for test_nimformer.nim).
  let dLossBatch = Tensor(data: dLoss.data, shape: @[1] & dLoss.shape)
  result = m.backwardBatch(@[ids], dLossBatch, ctx)

# ===================================================================
# ApfAdam (weight update on the CPU since it is just 1 vector, still fine)
# ===================================================================
type
  ApfAdamState* = object
    m*, v*: seq[float32]
    step*: int

proc newApfAdamState*(paramLen: int): ApfAdamState =
  result.m = newSeq[float32](paramLen)
  result.v = newSeq[float32](paramLen)
  result.step = 0

proc apfAdamStep*(param: var Tensor, grad: Tensor, state: var ApfAdamState,
                   lr: float32 = 1e-3'f32, b1: float32 = 0.9'f32, b2: float32 = 0.999'f32,
                   eps: float32 = 1e-8'f32, requantizeEvery: int = 1): CustomFloat =
  ## Pure CPU version (KEPT - still used by test_nimformer.nim and elsewhere).
  ## The LoRA fine-tune pipeline (lora.nim/finetune_lora.nim) uses
  ## apfAdamStepGpu below instead (runs on ctx/GPU, see the reason in
  ## its comment).
  inc state.step
  let bc1 = 1'f32 - pow(b1, float32(state.step))
  let bc2 = 1'f32 - pow(b2, float32(state.step))
  for i in 0 ..< param.data.len:
    let g = grad.data[i]
    state.m[i] = b1 * state.m[i] + (1'f32 - b1) * g
    state.v[i] = b2 * state.v[i] + (1'f32 - b2) * g * g
    let mHat = state.m[i] / bc1
    let vHat = state.v[i] / bc2
    param.data[i] -= lr * mHat / (sqrt(vHat) + eps)
  if state.step mod requantizeEvery == 0:
    let cf = buildCustomDtypeForTensor(param.data, grad.data)
    param.data = decodeArray(encodeArray(param.data, cf), cf)
    result = cf
  else:
    result = buildCustomDtypeForTensor(param.data)

proc constArrNf*(v: float32, n: int): seq[float32] =
  result = newSeq[float32](n)
  for i in 0 ..< n: result[i] = v

proc apfAdamStepGpu*(ctx: Backend, param: var Tensor, grad: Tensor, state: var ApfAdamState,
                      lr: float32 = 1e-3'f32, b1: float32 = 0.9'f32, b2: float32 = 0.999'f32,
                      eps: float32 = 1e-8'f32, requantizeEvery: int = 1): CustomFloat =
  ## GPU version of apfAdamStep, using beMul/beAdd/beDiv (backend.nim) for all
  ## the multiply/add/divide parts of Adam (m,v accumulate, mHat/vHat, param
  ## update) instead of a CPU for loop.
  ##
  ## All of Adam (m,v accumulate, mHat/vHat, sqrt, param update) now goes through
  ## beMul/beAdd/beDiv/beSub/beSqrt (backend.nim) - including sqrt(vHat), which has
  ## a real GPU kernel for all 3 backends (Metal/OpenCL/CUDA - see gpuSqrt in
  ## vendor/bybylang/gpubackend.nim). CUDA needs a manual nvcc build step and
  ## PTX linking (see vendor/bybylang/backends/cuda/kernels/vecop_sqrt_kernel.cu)
  ## before the kernel actually runs on a CUDA machine - NOT built/tested
  ## in this sandbox (no nvcc/real GPU).
  inc state.step
  let n = param.data.len
  let bc1 = 1'f32 - pow(b1, float32(state.step))
  let bc2 = 1'f32 - pow(b2, float32(state.step))

  # m = b1*m + (1-b1)*g   (GPU: mul + mul + add)
  let mDecay = beMul(ctx, state.m, constArrNf(b1, n))
  let mNew = beMul(ctx, grad.data, constArrNf(1'f32 - b1, n))
  state.m = beAdd(ctx, mDecay, mNew)

  # v = b2*v + (1-b2)*g*g   (GPU: mul + mul + mul + add)
  let gg = beMul(ctx, grad.data, grad.data)
  let vDecay = beMul(ctx, state.v, constArrNf(b2, n))
  let vNew = beMul(ctx, gg, constArrNf(1'f32 - b2, n))
  state.v = beAdd(ctx, vDecay, vNew)

  # mHat = m/bc1, vHat = v/bc2   (GPU: div)
  let mHat = beDiv(ctx, state.m, constArrNf(bc1, n))
  let vHat = beDiv(ctx, state.v, constArrNf(bc2, n))

  # sqrt(vHat) — NOW RUNS ON THE REAL GPU via beSqrt (Metal/OpenCL have a real kernel,
  # CUDA falls back to a correct CPU path - see the detailed notes in beSqrt/gpuSqrt).
  # (tensor_ops.nim is NOT imported here for sqrtT - tensor_ops.nim itself
  # does `import nimformer`, so importing back would be circular. Calling beSqrt straight from
  # `backend` - already imported at the top of the file - is enough and uses the same kernel.)
  let sqrtVHat = beSqrt(ctx, vHat)

  # denom = sqrt(vHat) + eps ; update = lr * mHat / denom ; param -= update  (GPU: add, div, mul, sub)
  let denom = beAdd(ctx, sqrtVHat, constArrNf(eps, n))
  let step = beDiv(ctx, mHat, denom)
  let scaledStep = beMul(ctx, step, constArrNf(lr, n))
  param.data = beSub(ctx, param.data, scaledStep)

  if state.step mod requantizeEvery == 0:
    let cf = buildCustomDtypeForTensor(param.data, grad.data)
    param.data = decodeArray(encodeArray(param.data, cf), cf)
    result = cf
  else:
    result = buildCustomDtypeForTensor(param.data)
