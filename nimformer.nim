## nimformer.nim
## Full Transformer with backward for attention + custom transformer blocks

import std/[math, random, strformat, algorithm, tables, strutils, os]
import customfloat
import backend  # Imports Backend and its helpers
import quant     # QuantTensor, dequantizeTensor - keeps int4-compressed weights in RAM
import vendor/bybylang/gpubackend as byby
from vendor/bybylang/probe import apiHooked, bybyApiHooksActive,
    runBybyApiHooks, bybyApiBefore, bybyApiAfter, bybyHookValue,
    bybyApiValueHooksActive, runBybyApiValueHooks, BybyApiCall, BybyApiArg,
    bybyHookTypeName
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

proc bybyHookValue*(value: Tensor): string =
  "Tensor(shape=" & $value.shape & ", elements=" & $value.data.len & ")"

type
  ForwardHook* = proc(ctx: Backend, input: Tensor): Tensor {.nimcall.}
  ForwardOutputHook* = proc(ctx: Backend, input, output: Tensor): Tensor {.nimcall.}
  ApiInputHook* = proc(api: string; ctx: Backend; inputs: var seq[Tensor]) {.closure.}
  ApiOutputHook* = proc(api: string; ctx: Backend; inputs: openArray[Tensor];
      output: Tensor): Tensor {.closure.}
  ForwardHookId* = int
  ApiHookId* = int
  ForwardHookRegistration = object
    id: ForwardHookId
    api: string
    forwardOnly: bool
    before: ForwardHook
    after: ForwardOutputHook
    apiBefore: ApiInputHook
    apiAfter: ApiOutputHook

const ForwardApiAll* = "*"
const ForwardApiNames* = [
  "linear", "layer_norm", "causal_self_attention", "feed_forward",
  "transformer_block", "embedding", "nimformer_model_batch", "nimformer_model",
]
const HookApiNames* = [
  "linear", "layer_norm", "causal_self_attention", "feed_forward",
  "transformer_block", "embedding", "nimformer_model_batch", "nimformer_model",
  "newtensor", "randntensor", "addt", "subt", "mult", "transpose",
  "expt", "logt", "sqrtt", "powt", "view", "reshape", "permute",
  "concat", "stack", "split", "cumsumt", "cumprodt", "maskedfill", "wheret",
]

var gForwardHooks: seq[ForwardHookRegistration]
var gNextForwardHookId = 1

proc normalizeForwardApi(api: string): string {.inline.} =
  api.strip().toLowerAscii()

proc registerForwardHook*(api: string; before: ForwardHook = nil;
    after: ForwardOutputHook = nil): ForwardHookId =
  ## Register the legacy single-input callbacks for a module forward API.
  ## Use `registerApiHook` to also hook the low-level Tensor operations.
  let normalized = normalizeForwardApi(api)
  if normalized.len == 0:
    raise newException(ValueError, "forward hook API name must not be empty")
  if normalized != ForwardApiAll and normalized notin ForwardApiNames:
    raise newException(ValueError, "unknown forward hook API: " & api)
  if before == nil and after == nil:
    raise newException(ValueError, "a forward hook needs a before or after callback")
  result = gNextForwardHookId
  inc gNextForwardHookId
  gForwardHooks.add(ForwardHookRegistration(id: result, api: normalized,
    forwardOnly: true, before: before, after: after))

proc registerApiHook*(api: string; before: ApiInputHook = nil;
    after: ApiOutputHook = nil): ApiHookId =
  ## Register a hook for a module forward or a public Tensor operation. Before
  ## callbacks can replace every Tensor operand; after callbacks can replace
  ## the output. Use `"*"` to apply the hook throughout nested operations.
  let normalized = normalizeForwardApi(api)
  if normalized.len == 0:
    raise newException(ValueError, "API hook name must not be empty")
  if normalized != ForwardApiAll and normalized notin HookApiNames:
    raise newException(ValueError, "unknown hook API: " & api)
  if before == nil and after == nil:
    raise newException(ValueError, "an API hook needs a before or after callback")
  result = gNextForwardHookId
  inc gNextForwardHookId
  gForwardHooks.add(ForwardHookRegistration(id: result, api: normalized,
    forwardOnly: false, apiBefore: before, apiAfter: after))

proc removeForwardHook*(id: ForwardHookId): bool =
  ## Remove one registration. Returns false when the ID is not registered.
  for i in 0 ..< gForwardHooks.len:
    if gForwardHooks[i].id == id:
      gForwardHooks.delete(i)
      return true

proc removeApiHook*(id: ApiHookId): bool =
  ## Remove one hook registered through either hook API.
  removeForwardHook(id)

proc clearForwardHooks*(api = "") =
  ## Remove legacy forward hooks, or only those registered for one API name.
  if api.len == 0:
    var retained: seq[ForwardHookRegistration]
    for registration in gForwardHooks:
      if not registration.forwardOnly:
        retained.add(registration)
    gForwardHooks = retained
    return
  let normalized = normalizeForwardApi(api)
  var retained: seq[ForwardHookRegistration]
  for registration in gForwardHooks:
    if not registration.forwardOnly or registration.api != normalized:
      retained.add(registration)
  gForwardHooks = retained

proc clearApiHooks*(api = "") =
  ## Remove API hooks, or only those registered for one API name.
  if api.len == 0:
    var retained: seq[ForwardHookRegistration]
    for registration in gForwardHooks:
      if registration.forwardOnly:
        retained.add(registration)
    gForwardHooks = retained
    return
  let normalized = normalizeForwardApi(api)
  var retained: seq[ForwardHookRegistration]
  for registration in gForwardHooks:
    if registration.forwardOnly or registration.api != normalized:
      retained.add(registration)
  gForwardHooks = retained

proc forwardApiNames*(): seq[string] =
  ## Return the high-level forward APIs that accept custom hooks.
  @ForwardApiNames

proc hookApiNames*(): seq[string] =
  ## Return every module-forward and Tensor operation API that accepts hooks.
  @HookApiNames

proc registrationMatches(registration: ForwardHookRegistration;
    api: string): bool {.inline.} =
  result = registration.api == api or
    (registration.api == ForwardApiAll and
      (not registration.forwardOnly or api in ForwardApiNames))

proc applyApiBefore*(api: string; ctx: Backend; inputs: var seq[Tensor]) =
  let normalized = normalizeForwardApi(api)
  let expectedInputs = inputs.len
  for registration in gForwardHooks:
    if registrationMatches(registration, normalized):
      if registration.apiBefore != nil:
        registration.apiBefore(normalized, ctx, inputs)
      if registration.forwardOnly and registration.before != nil and
          normalized in ForwardApiNames and inputs.len > 0:
        inputs[0] = registration.before(ctx, inputs[0])
      if inputs.len != expectedInputs:
        raise newException(ValueError,
          "API hooks must preserve the number of Tensor operands for " & normalized)

proc applyApiAfter*(api: string; ctx: Backend; inputs: openArray[Tensor];
    output: Tensor): Tensor =
  result = output
  let normalized = normalizeForwardApi(api)
  for registration in gForwardHooks:
    if registrationMatches(registration, normalized):
      if registration.apiAfter != nil:
        result = registration.apiAfter(normalized, ctx, inputs, result)
      if registration.forwardOnly and registration.after != nil and
          normalized in ForwardApiNames and inputs.len > 0:
        result = registration.after(ctx, inputs[0], result)

template noTensorHooks(): bool =
  ## True when no forward/API hook is registered: hot paths then skip the
  ## operand seq copies that the hook plumbing needs.
  gForwardHooks.len == 0

proc applyForwardBefore(api: string; ctx: Backend; input: Tensor): Tensor =
  var inputs = @[input]
  applyApiBefore(api, ctx, inputs)
  result = if inputs.len > 0: inputs[0] else: input

proc applyForwardAfter(api: string; ctx: Backend; input, output: Tensor): Tensor =
  applyApiAfter(api, ctx, [input], output)

proc numel(shape: seq[int]): int =
  result = 1
  for s in shape: result *= s

var gLoraSizeMismatchWarned = 0  ## counts LoRA size-mismatch warnings
                                  ## (see forward() below) - capped so the log
                                  ## is not spammed when many Linears hit it.

proc newTensor*(shape: seq[int], fill: float32 = 0'f32): Tensor {.apiHooked.} =
  if noTensorHooks():
    result.shape = shape
    result.data = newSeq[float32](numel(shape))
    if fill != 0'f32:
      for i in 0 ..< result.data.len: result.data[i] = fill
    return
  let ctx = Backend(kind: bkCpu)
  var inputs: seq[Tensor]
  applyApiBefore("newtensor", ctx, inputs)
  result.shape = shape
  result.data = newSeq[float32](numel(shape))
  if fill != 0'f32:
    for i in 0 ..< result.data.len: result.data[i] = fill
  result = applyApiAfter("newtensor", ctx, inputs, result)

proc randnTensor*(shape: seq[int], scale: float32): Tensor {.apiHooked.} =
  let ctx = Backend(kind: bkCpu)
  var inputs: seq[Tensor]
  applyApiBefore("randntensor", ctx, inputs)
  result.shape = shape
  result.data = newSeq[float32](numel(shape))
  for i in 0 ..< result.data.len:
    result.data[i] = float32(gauss(0.0, 1.0)) * scale
  result = applyApiAfter("randntensor", ctx, inputs, result)

proc addT*(a, b: Tensor): Tensor {.apiHooked.} =
  let ctx = Backend(kind: bkCpu)
  var inputs = @[a, b]
  applyApiBefore("addt", ctx, inputs)
  if inputs.len != 2:
    raise newException(ValueError, "addT hooks must preserve the two Tensor operands")
  result = inputs[0]
  for i in 0 ..< result.data.len: result.data[i] += inputs[1].data[i]
  result = applyApiAfter("addt", ctx, inputs, result)

proc addT*(ctx: Backend, a, b: Tensor): Tensor {.apiHooked.} =
  ## The addT(a,b) above runs on the CPU even though beAdd (real GPU, implemented
  ## for CUDA/Metal/OpenCL) exists - this residual connection runs TWICE PER
  ## LAYER (attn + ff), i.e. 64 times/token for a 32-layer model, not a small
  ## operation to ignore. This overload goes through the GPU, as required ("never
  ## silently run on the CPU"). The old addT(a,b) (no ctx) stays for the
  ## other callers that have no Backend in scope.
  if noTensorHooks():
    result.shape = a.shape
    result.data = beAdd(ctx, a.data, b.data)
    return
  var inputs = @[a, b]
  applyApiBefore("addt", ctx, inputs)
  if inputs.len != 2:
    raise newException(ValueError, "addT hooks must preserve the two Tensor operands")
  result = inputs[0]
  result.data = beAdd(ctx, inputs[0].data, inputs[1].data)
  result = applyApiAfter("addt", ctx, inputs, result)

proc subT*(a, b: Tensor): Tensor {.apiHooked.} =
  let ctx = Backend(kind: bkCpu)
  var inputs = @[a, b]
  applyApiBefore("subt", ctx, inputs)
  if inputs.len != 2:
    raise newException(ValueError, "subT hooks must preserve the two Tensor operands")
  result = inputs[0]
  for i in 0 ..< result.data.len: result.data[i] -= inputs[1].data[i]
  result = applyApiAfter("subt", ctx, inputs, result)

proc mulT*(a, b: Tensor): Tensor {.apiHooked.} =
  let ctx = Backend(kind: bkCpu)
  var inputs = @[a, b]
  applyApiBefore("mult", ctx, inputs)
  if inputs.len != 2:
    raise newException(ValueError, "mulT hooks must preserve the two Tensor operands")
  result = inputs[0]
  for i in 0 ..< result.data.len: result.data[i] *= inputs[1].data[i]
  result = applyApiAfter("mult", ctx, inputs, result)

proc flatten2D*(shape: seq[int]): tuple[rows, cols: int] {.apiHooked.} =
  let cols = shape[^1]
  var rows = 1
  for i in 0 ..< shape.len - 1: rows *= shape[i]
  result = (rows, cols)

proc transposeCore(input: Tensor): Tensor =
  let (rows, cols) = flatten2D(input.shape)
  result.shape = @[cols, rows]
  result.data = newSeqUninit[float32](rows * cols)
  if input.mappedData.owner == nil and input.data.len == rows * cols:
    for i in 0 ..< rows:
      let src = i * cols
      for j in 0 ..< cols:
        result.data[j * rows + i] = input.data[src + j]
  else:
    for i in 0 ..< rows:
      for j in 0 ..< cols:
        result.data[j * rows + i] = input.floatValues[i * cols + j]

proc transpose*(x: Tensor): Tensor {.apiHooked.} =
  ## Transposes a tensor treated as a 2D matrix [rows, cols] (all dims except
  ## the last merged into "rows", like flatten2D) -> [cols, rows].
  if noTensorHooks():
    return transposeCore(x)
  let ctx = Backend(kind: bkCpu)
  var inputs = @[x]
  applyApiBefore("transpose", ctx, inputs)
  result = transposeCore(inputs[0])
  result = applyApiAfter("transpose", ctx, inputs, result)

# ===================================================================
# Linear (forward/backward use Metal matmul)
# ===================================================================
type
  Linear* = object
    weight*: Tensor      ## small fp32 fallback (only for random init before real weights are loaded)
    # ...

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
    preForwardHook*: ForwardHook  ## Hook for custom layer logic
    tensorName*: string           ## checkpoint tensor name, given to the FIR recorder at upload
    gpuCacheName*: string         ## name in gFp4CacheGpu after prepareGpuLinForLinear (FP4-KM)

template loraAValues*(l: Linear): untyped = weightValues(l.loraA, l.mappedLoraA)
template loraBValues*(l: Linear): untyped = weightValues(l.loraB, l.mappedLoraB)

proc newLinear*(inF, outF: int; initializeWeights = true): Linear {.apiHooked.} =
  let scale = sqrt(2'f32 / float32(inF))
  result.inF = inF
  result.outF = outF
  result.weight = if initializeWeights: randnTensor(@[outF, inF], scale)
                  else: Tensor(shape: @[outF, inF], data: @[])
  result.useQuant = false
  result.bias = newTensor(@[outF])

proc setQuantWeight*(l: var Linear, qt: QuantTensor) {.apiHooked.} =
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
proc uploadResident*(l: var Linear, ctx: Backend, sparsityPct: float32 = 0.0'f32) {.apiHooked.} =
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
    else:
      l.residentW = beResidentUploadQ4(ctx, @(l.weightQ.bytes), l.weightQ.scale, l.weightQ.zero_point,
                                        l.inF, l.outF, l.weightQ.groupSize, nGroupsPerRow)
    if l.tensorName.len > 0: probeName(l.residentW, l.tensorName, "q4", [l.outF, l.inF])
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
  elif l.useQuant:
    # qkInt8 / qkInt4 (symmetric) / qkFp32Raw ... : dequant + transpose 1
    # LẦN, upload resident fp32.
    let wT = dequantizeTensorTransposed(l.weightQ)
    l.residentFp32 = beResidentUploadFp32(ctx, wT)
    l.residentFp32NT = false
    l.residentFp32Offset = 0
    l.residentFp32Ready = true
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

proc linearForwardCore(l: Linear, input: Tensor, ctx: Backend): Tensor =
  let (rows, cols) = flatten2D(input.shape)
  assert cols == l.inF
  result.shape = input.shape
  result.shape[^1] = l.outF

  var y: seq[float32]
  pfTime("lin.gemm"):
    if l.residentReady:
      y = beResidentLinearQ4(ctx, input.data, l.residentW, rows)  # uploads only the activation, weight already resident
    elif l.residentFp32Ready:
      # int8/int4-symmetric/plain fp32 weights are ALREADY resident (see
      # uploadResident) - only the small activation is uploaded per token, the
      # whole weight is NOT dequantized again.
      y = if l.residentFp32NT: residentLinearNt(l, input.data, rows, ctx)
          else: beResidentLinearFp32(ctx, input.data, l.residentFp32, rows, l.inF, l.outF)

    if y.len == 0:
      # Not resident, or the current backend has no resident path (e.g.
      # CPU) -> falls back to the old path (still correct, just without the resident speedup).
      if l.useQuant and l.weightQ.kind == qkFp4KM and l.gpuCacheName.len > 0 and
         l.gpuCacheName in gFp4CacheGpu and gFp4CacheGpu[l.gpuCacheName].residentReady and
         ctx.kind.toByby() in {byby.gbMetal, byby.gbVulkan}:
        # Parallel prefill: GPU resident FP4-KM + MBLOCK=4 (no CPU dequant).
        y = fp4GemmGpuRows(ctx, l.gpuCacheName, input.data, rows)
      elif l.useQuant and l.weightQ.kind == qkInt4Asymmetric and l.weightQ.groupSize > 0:
        let nGroupsPerRow = (l.inF + l.weightQ.groupSize - 1) div l.weightQ.groupSize
        y = beMatmulQ4(ctx, input.data, @(l.weightQ.bytes), l.weightQ.scale, l.weightQ.zero_point,
                        rows, l.inF, l.outF, l.weightQ.groupSize, nGroupsPerRow)
      elif (ctx.usesTrainingTensorCores or (getEnv("BYBY_FINETUNE_ACCELERATED_GEMM", "0") != "0" and
           ctx.kind.toByby() in {byby.gbMetal, byby.gbVulkan})) and
           l.weight.data.len == l.outF * l.inF:
        # Keep the trainable weight in its native [outF,inF] layout. Vulkan
        # can lower this NT GEMM to cooperative FP16 matrix instructions with
        # FP32 accumulation; Metal uses its register-blocked SIMD NT kernel.
        y = beMatmulNT(ctx, input.data, rows, l.inF, l.weight.data, l.outF)
      else:
        var wT: seq[float32]
        if l.useQuant:
          wT = dequantizeTensorTransposed(l.weightQ)
        else:
          wT = transpose(l.weight).data
        y = beMatmul(ctx, input.data, rows, l.inF, wT, l.inF, l.outF)

  # === ALWAYS MAKE SURE THE SIZE IS RIGHT ===
  let expectedSize = rows * l.outF
  if y.len != expectedSize:
    # New array with the right size, copy the existing elements, leave the rest at 0
    var fixed = newSeq[float32](expectedSize)
    let copyLen = min(y.len, expectedSize)
    for i in 0 ..< copyLen:
      fixed[i] = y[i]
    y = fixed

  # Add bias in place (l.bias.data has length l.outF; empty = no bias), then
  # hand the GEMM buffer over to the result without another copy.
  if l.bias.data.len == l.outF:
    for r in 0 ..< rows:
      let base = r * l.outF
      for j in 0 ..< l.outF:
        y[base + j] = y[base + j] + l.bias.data[j]
  result.data = move(y)

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
      var xA, delta: seq[float32]
      if getEnv("BYBY_FINETUNE_ACCELERATED_GEMM", "0") != "0" and
         ctx.kind.toByby() in {byby.gbMetal, byby.gbVulkan}:
        xA = beMatmulNT(ctx, input.data, rows, l.inF, l.loraAValues, l.loraR)
        delta = beMatmulNT(ctx, xA, rows, l.loraR, l.loraBValues, l.outF)
      else:
        let At = transpose(Tensor(shape: @[l.loraR, l.inF], data: @(l.loraAValues)))   # [inF, r]
        let Bt = transpose(Tensor(shape: @[l.outF, l.loraR], data: @(l.loraBValues)))  # [r, outF]
        xA = beMatmul(ctx, input.data, rows, l.inF, At.data, l.inF, l.loraR)   # [rows, r]
        delta = beMatmul(ctx, xA, rows, l.loraR, Bt.data, l.loraR, l.outF)     # [rows, outF]
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

proc forward*(l: Linear, x: Tensor, ctx: Backend): Tensor {.apiHooked.} =
  if noTensorHooks() and l.preForwardHook == nil:
    return linearForwardCore(l, x, ctx)
  var input = applyForwardBefore("linear", ctx, x)
  if l.preForwardHook != nil:
    input = l.preForwardHook(ctx, input)
  result = linearForwardCore(l, input, ctx)
  result = applyForwardAfter("linear", ctx, input, result)
proc backward*(l: Linear, x: Tensor, dOut: Tensor, ctx: Backend): tuple[dX, dW, dB: Tensor] {.apiHooked.} =
  ## Linear backward:
  ## dW = x^T @ dOut  (accumulated over the batch)
  ## dB = sum(dOut, axis=0)
  ## dX = dOut @ W
  let (rows, cols) = flatten2D(x.shape)
  doAssert cols == l.inF,
    &"Linear.backward: x.shape={x.shape} (cols={cols}) but l.inF={l.inF}"
  doAssert dOut.shape[^1] == l.outF,
    &"Linear.backward: dOut.shape={dOut.shape} (last={dOut.shape[^1]}) but l.outF={l.outF} (l.inF={l.inF}, x.shape={x.shape})"

  # dW = dOut^T @ x is produced directly in l.weight's [outF, inF] layout, so
  # no transposed copy of x and no transpose of the [inF, outF] product.
  let dOutT = transpose(dOut)  # [outF, rows]
  let dOutFlat = dOut.data  # [rows * outF]

  # dW and dX are independent. The optional fine-tune route sends dW through
  # the backend's NT GEMM (Vulkan cooperative matrices / Metal SIMD). Tensor Core
  # training uses the same mixed-precision path for the input gradient.
  var dWFlat, dXFlat: seq[float32]
  if ctx.usesTrainingTensorCores or (getEnv("BYBY_FINETUNE_ACCELERATED_GEMM", "0") != "0" and
     ctx.kind.toByby() in {byby.gbMetal, byby.gbVulkan}):
    let xT = transpose(x)  # [inF, rows]: NT GEMM wants B as [n, k]
    dWFlat = beMatmulNT(ctx, dOutT.data, l.outF, rows, xT.data, l.inF)
    if ctx.usesTrainingTensorCores:
      let weightT = transpose(l.weight)
      dXFlat = beMatmulNT(ctx, dOutFlat, rows, l.outF, weightT.data, l.inF)
    else:
      dXFlat = beMatmul(ctx, dOutFlat, rows, l.outF, l.weight.data, l.outF, l.inF)
  else:
    (dWFlat, dXFlat) = beMatmul2(ctx,
      dOutT.data, l.outF, rows, x.floatValues, rows, l.inF,
      dOutFlat, rows, l.outF, l.weight.data, l.outF, l.inF)

  var dW: Tensor
  dW.shape = @[l.outF, l.inF]  # matches l.weight.shape
  dW.data = move(dWFlat)

  var dX: Tensor
  dX.shape = x.shape
  dX.data = move(dXFlat)

  # dB = sum of dOut over the batch
  var dB = newTensor(@[l.outF])
  for i in 0 ..< rows:
    for j in 0 ..< l.outF:
      dB.data[j] += dOutFlat[i * l.outF + j]

  result = (dX, dW, dB)

proc backwardInputOnly*(l: Linear, x: Tensor, dOut: Tensor, ctx: Backend): Tensor {.apiHooked.} =
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

proc newLayerNorm*(dim: int, eps: float32 = 1e-5'f32): LayerNorm {.apiHooked.} =
  result.dim = dim
  result.gamma = newTensor(@[dim], 1'f32)
  result.beta  = newTensor(@[dim], 0'f32)
  result.eps   = eps

proc forward*(ln: LayerNorm, x: Tensor, ctx: Backend): Tensor {.apiHooked.} =
  ## This function used to take `ctx: Backend` but NOT USE it - the whole
  ## LayerNorm ran as a plain Nim loop on the CPU whatever
  ## backend (cuda/metal/opencl) was chosen. backend.nim already has `beLayernorm` matching
  ## this exact operation (rows x cols, same normalisation formula), so it just
  ## calls that so LayerNorm really runs on the chosen GPU.
  if noTensorHooks():
    let (rows, cols) = flatten2D(x.shape)
    assert cols == ln.dim
    result.shape = x.shape
    result.data = beLayernorm(ctx, x.data, ln.gamma.data, ln.beta.data, rows, cols, ln.eps)
    return
  let input = applyForwardBefore("layer_norm", ctx, x)
  let (rows, cols) = flatten2D(input.shape)
  assert cols == ln.dim
  result.shape = input.shape
  result.data = beLayernorm(ctx, input.data, ln.gamma.data, ln.beta.data, rows, cols, ln.eps)
  result = applyForwardAfter("layer_norm", ctx, input, result)

proc backward*(ln: LayerNorm, x: Tensor, dOut: Tensor, ctx: Backend): tuple[dX, dGamma, dBeta: Tensor] {.apiHooked.} =
  ## LayerNorm backward (matrix ops / GPU-bound)
  let (rows, cols) = flatten2D(x.shape)
  assert cols == ln.dim
  var (dx_data, dgamma_data, dbeta_data) = beLayernormBackward(ctx, dOut.data, x.data, ln.gamma.data, ln.beta.data, rows, cols, ln.eps)
  result.dX.shape = x.shape
  result.dX.data = move(dx_data)
  result.dGamma.shape = ln.gamma.shape
  result.dGamma.data = move(dgamma_data)
  result.dBeta.shape = ln.beta.shape
  result.dBeta.data = move(dbeta_data)

# ===================================================================
# CausalSelfAttention (forward/backward use Metal matmul)
# ===================================================================
type
  CausalSelfAttention* = object
    nHeads*, headDim*, embedDim*: int
    qkv*, proj*: Linear

proc newCausalSelfAttention*(embedDim, nHeads: int): CausalSelfAttention {.apiHooked.} =
  assert embedDim mod nHeads == 0
  result.embedDim = embedDim
  result.nHeads = nHeads
  result.headDim = embedDim div nHeads
  result.qkv = newLinear(embedDim, 3 * embedDim)
  result.proj = newLinear(embedDim, embedDim)

proc forward*(attn: CausalSelfAttention, x: Tensor, ctx: Backend): Tensor {.apiHooked.} =
  let input = applyForwardBefore("causal_self_attention", ctx, x)
  let B = input.shape[0]
  let T = input.shape[1]
  let C = input.shape[2]
  assert C == attn.embedDim
  let qkv = attn.qkv.forward(input, ctx)
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
  result = applyForwardAfter("causal_self_attention", ctx, input, result)

proc backward*(attn: CausalSelfAttention, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dQkvW, dQkvB, dProjW, dProjB: Tensor] {.apiHooked.} =
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

proc newFeedForward*(embedDim, mult: int): FeedForward {.apiHooked.} =
  result.fc1 = newLinear(embedDim, embedDim * mult)
  result.fc2 = newLinear(embedDim * mult, embedDim)

proc forward*(ff: FeedForward, x: Tensor, ctx: Backend): Tensor {.apiHooked.} =
  if noTensorHooks():
    var h = ff.fc1.forward(x, ctx)
    h.data = beApflu(ctx, h.data, 0.1'f32, 1.0'f32)
    return ff.fc2.forward(h, ctx)
  let input = applyForwardBefore("feed_forward", ctx, x)
  var h = ff.fc1.forward(input, ctx)
  h.data = beApflu(ctx, h.data, 0.1'f32, 1.0'f32)
  result = ff.fc2.forward(h, ctx)
  result = applyForwardAfter("feed_forward", ctx, input, result)

proc backward*(ff: FeedForward, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dFc1W, dFc1B, dFc2W, dFc2B: Tensor] {.apiHooked.} =
  let hidden = ff.fc1.forward(x, ctx)
  let activated = Tensor(shape: hidden.shape, data: beApflu(ctx, hidden.data, 0.1'f32, 1.0'f32))
  let dFc2 = ff.fc2.backward(activated, dOut, ctx)
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

proc newTransformerBlock*(embedDim, nHeads, ffMult: int): TransformerBlock {.apiHooked.} =
  result.attn = newCausalSelfAttention(embedDim, nHeads)
  result.ff = newFeedForward(embedDim, ffMult)
  result.ln1 = newLayerNorm(embedDim)
  result.ln2 = newLayerNorm(embedDim)

proc forward*(blk: TransformerBlock, x: Tensor, ctx: Backend): Tensor {.apiHooked.} =
  let input = applyForwardBefore("transformer_block", ctx, x)
  let x1 = blk.ln1.forward(input, ctx)
  let attnOut = blk.attn.forward(x1, ctx)
  # Optimisation: "add then layernorm right after" used to be 2 separate GPU
  # commands (addT then ln2.forward, each opening/closing its own command buffer) -
  # merged into 1 session with beAddLayernormFused (see backend.nim), which returns
  # BOTH the raw residual (x2Raw, needed for the block's final add) AND the
  # layernormed version (x3) from exactly 1 upload/commit/wait - the add is not done twice.
  # Unsupported backends (CUDA/OpenCL/CPU) fall back to the 2 old commands inside.
  let (rows2, cols2) = flatten2D(input.shape)
  let fused = beAddLayernormFused(ctx, input.data, attnOut.data, blk.ln2.gamma.data, blk.ln2.beta.data,
                                   rows2, cols2, blk.ln2.eps)
  var x2Raw = newTensor(input.shape)
  x2Raw.data = fused.summed
  var x3 = newTensor(input.shape)
  x3.data = fused.normed
  let ffOut = blk.ff.forward(x3, ctx)
  result = addT(ctx, x2Raw, ffOut)
  result = applyForwardAfter("transformer_block", ctx, input, result)

proc backward*(blk: TransformerBlock, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
    dX, dAttnQkvW, dAttnQkvB, dAttnProjW, dAttnProjB,
    dFf1W, dFf1B, dFf2W, dFf2B,
    dLn1G, dLn1B, dLn2G, dLn2B: Tensor] {.apiHooked.} =
  
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

proc newEmbedding*(vocab, dim: int; initializeWeights = true): Embedding {.apiHooked.} =
  result.vocab = vocab
  result.dim = dim
  result.weight = if initializeWeights: randnTensor(@[vocab, dim], sqrt(2'f32 / float32(vocab)))
                  else: Tensor(shape: @[vocab, dim], data: @[])

proc lookupBatch*(e: Embedding, idsBatch: seq[seq[int]], ctx: Backend): Tensor {.apiHooked.} =
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
  result = applyForwardBefore("embedding", ctx, result)
  result = applyForwardAfter("embedding", ctx, result, result)

proc backward*(e: Embedding, idsBatch: seq[seq[int]], dOut: Tensor, ctx: Backend): Tensor {.apiHooked.} =
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

proc newNimformerModel*(vocab, embedDim, nHeads, nLayers, ffMult: int): NimformerModel {.apiHooked.} =
  result.vocab = vocab
  result.embed = newEmbedding(vocab, embedDim)
  result.blocks = newSeq[TransformerBlock](nLayers)
  for i in 0 ..< nLayers:
    result.blocks[i] = newTransformerBlock(embedDim, nHeads, ffMult)
  result.outProj = newLinear(embedDim, vocab)

proc forwardModelIds(m: NimformerModel; idsBatch: seq[seq[int]]; ctx: Backend;
    api: string): Tensor =
  var x = m.embed.lookupBatch(idsBatch, ctx)
  x = applyForwardBefore(api, ctx, x)
  for blk in m.blocks:
    x = blk.forward(x, ctx)
  result = m.outProj.forward(x, ctx)
  result = applyForwardAfter(api, ctx, x, result)

proc forwardBatch*(m: NimformerModel, idsBatch: seq[seq[int]], ctx: Backend): Tensor {.apiHooked.} =
  ## REAL forward with batch B (idsBatch.len sequences of the same length T) in one
  ## call — Linear/LayerNorm treat B*T as the "row" count, so the GPU gets M=B*T rows
  ## in each matmul instead of being called B times in sequence (B=1 each) as before.
  ## Returns logits of shape [B, T, vocab].
  result = forwardModelIds(m, idsBatch, ctx, "nimformer_model_batch")

proc forward*(m: NimformerModel, ids: seq[int], ctx: Backend): Tensor {.apiHooked.} =
  ## Convenience for a single sequence (keeps the old API for test_nimformer.nim) —
  ## calls forwardBatch with B=1, then drops the batch dim.
  let logits = forwardModelIds(m, @[ids], ctx, "nimformer_model")
  result = Tensor(data: logits.data, shape: logits.shape[1 .. ^1])

proc backwardBatch*(m: NimformerModel, idsBatch: seq[seq[int]], dLoss: Tensor,
                     ctx: Backend): seq[Tensor] {.apiHooked.} =
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

proc backward*(m: NimformerModel, ids: seq[int], dLoss: Tensor, ctx: Backend): seq[Tensor] {.apiHooked.} =
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

proc newApfAdamState*(paramLen: int): ApfAdamState {.apiHooked.} =
  result.m = newSeq[float32](paramLen)
  result.v = newSeq[float32](paramLen)
  result.step = 0

proc apfAdamStep*(param: var Tensor, grad: Tensor, state: var ApfAdamState,
                   lr: float32 = 1e-3'f32, b1: float32 = 0.9'f32, b2: float32 = 0.999'f32,
                   eps: float32 = 1e-8'f32, requantizeEvery: int = 1): CustomFloat {.apiHooked.} =
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

proc constArrNf*(v: float32, n: int): seq[float32] {.apiHooked.} =
  result = newSeq[float32](n)
  for i in 0 ..< n: result[i] = v

proc apfAdamStepGpu*(ctx: Backend, param: var Tensor, grad: Tensor, state: var ApfAdamState,
                      lr: float32 = 1e-3'f32, b1: float32 = 0.9'f32, b2: float32 = 0.999'f32,
                      eps: float32 = 1e-8'f32, requantizeEvery: int = 1): CustomFloat {.apiHooked.} =
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
