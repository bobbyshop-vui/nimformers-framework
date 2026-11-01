## tensor_ops.nim
## Adds the low-level operations still missing compared with torch (see the requested
## list): exp/log/sqrt/pow, view/permute, reductions (sum/mean/max/argmax),
## concat/split/stack, dropout, comparisons (gt/lt/eq).
##
## GPU UPDATE: exp/log/sqrt/pow (section 1) and cumsum/cumprod/affine-scan
## (section 8, used by Mamba/SSM) now take `ctx: Backend` to run on a REAL GPU -
## exp/log/sqrt through byby.gpuExp/gpuLog/gpuSqrt (Metal + OpenCL kernels compiled
## from source at run time, so they run right away; CUDA needs one nvcc rebuild
## of backends/cuda/kernels/vecop_ext.cu, see the comment in that file - until
## then it falls back to the CPU through the codebase-wide handleGpuFailure
## mechanism, no silently wrong results). The scans (cumsum/cumprod/affine) run
## Hillis-Steele step by step through real beAdd/beMul on the GPU (see section 8).
##
## view/permute/reduction(sum/mean/max/argmax)/concat/split/stack/dropout/
## comparison (sections 2-7) are still on the CPU: these are small indexing/reduction
## ops (not FLOP bottlenecks like the scans/activations), kept on the CPU for simplicity
## and immediate correctness; to speed up a specific op of this group later, add
## a dedicated kernel following the sessionRmsnormEnc/sessionRope pattern already in
## gpubackend.nim.

import std/[math, sequtils]
import backend
import nimformer  # Tensor, numel-style helpers (flatten2D, newTensor, ...)

## GPU PORT ROUND 2 NOTE (permute/concat/split/stack/argmax-index/
## comparison/dropout): backend.nim now also exports beGatherGpu/
## beArgmaxIndexGpu/beCompareGpuOp/beDropoutMaskGpu (see gpubackend_additions.nim
## + the accompanying cuda/vulkan/metal driver additions) - GATHER ops (reading
## through an index array), ARGMAX-INDEX, comparisons producing bitmasks, and dropout mask generation
## all run real GPU kernels (gather_kernel.cu/.comp/.metal, argmax_kernel...,
## compare_kernel..., dropout_rng_kernel...) instead of CPU loops/RNG as
## before. `ctx: Backend` is now a REQUIRED parameter for this whole group of functions.

# ─────────────────────────────────────────────────────────────
# 1. EXP / LOG / SQRT / POW (element-wise, required for Mamba/RMSNorm/Loss)
# ─────────────────────────────────────────────────────────────

## beExp/beLog/beSqrt/bePow: the REAL definitions live in backend.nim (dispatched through
## byby.gpuExp/gpuLog/gpuSqrt - real GPU kernels on Metal/OpenCL right away,
## CUDA needs one nvcc rebuild of vecop_ext.cu before the kernel exists, until then
## it falls back to the CPU through the codebase-wide handleGpuFailure mechanism).
## NOT redefined here - the `import backend` above brings these procs
## into scope.

proc expT*(ctx: Backend, x: Tensor): Tensor =
  result.shape = x.shape
  result.data = beExp(ctx, x.data)

proc logT*(ctx: Backend, x: Tensor): Tensor =
  result.shape = x.shape
  result.data = beLog(ctx, x.data)

proc sqrtT*(ctx: Backend, x: Tensor): Tensor =
  result.shape = x.shape
  result.data = beSqrt(ctx, x.data)

proc powT*(ctx: Backend, x: Tensor, p: float32): Tensor =
  result.shape = x.shape
  result.data = bePow(ctx, x.data, p)

# ─────────────────────────────────────────────────────────────
# 2. VIEW / RESHAPE - no data copy (shares the seq through Nim's ref semantics
#    when possible; a Nim seq[float32] is copy-on-write, so a "view" really
#    does not copy until the receiver WRITES to it - enough to avoid read copies).
# ─────────────────────────────────────────────────────────────

proc numelShape(shape: seq[int]): int =
  result = 1
  for s in shape: result *= s

proc view*(x: Tensor, newShape: seq[int]): Tensor =
  ## A plain reshape (no copy) - supports one "-1" dimension (inferred) like torch.
  var shape = newShape
  var inferIdx = -1
  var known = 1
  for i, s in shape:
    if s == -1:
      doAssert inferIdx == -1, "view: -1 may be used in only ONE dimension"
      inferIdx = i
    else:
      known *= s
  if inferIdx >= 0:
    doAssert known != 0 and x.data.len mod known == 0,
      "view: cannot infer the -1 dimension from shape " & $x.shape & " -> " & $newShape
    shape[inferIdx] = x.data.len div known
  doAssert numelShape(shape) == x.data.len,
    "view: element count mismatch (" & $x.data.len & " vs " & $numelShape(shape) & ")"
  result.shape = shape
  result.data = x.data   # Nim seq = copy-on-write: reading costs no copy.

proc reshape*(x: Tensor, newShape: seq[int]): Tensor = view(x, newShape)

# ─────────────────────────────────────────────────────────────
# 3. PERMUTE - reorders dimensions (e.g. attention [B,T,H,D] -> [B,H,T,D]).
#    This IS a real copy (unlike view) because the memory layout changes - like torch,
#    .permute() returns a "logical" view that is not contiguous; here we always
#    return a contiguous tensor so every later op (matmul, ...) can use it right away.
# ─────────────────────────────────────────────────────────────

proc strides(shape: seq[int]): seq[int] =
  result = newSeq[int](shape.len)
  var acc = 1
  for i in countdown(shape.len - 1, 0):
    result[i] = acc
    acc *= shape[i]

proc permute*(ctx: Backend, x: Tensor, dims: seq[int]): Tensor =
  ## LIKE TORCH: only the strides are computed (O(rank), rank<=8, independent of n) on
  ## the CPU and sent to the permute_strided kernel (CUDA/Vulkan/Metal) - each GPU
  ## thread decodes its own source index right on the device. No more
  ## O(n) CPU loop building an index array like the previous version.
  doAssert dims.len == x.shape.len, "permute: the number of dims must match the shape"
  doAssert dims.len <= 8, "permute: rank > 8 not supported (fixed 8-entry push-constant table/kernel)"
  var newShape = newSeq[int](dims.len)
  for i, d in dims: newShape[i] = x.shape[d]
  let oldStrides = strides(x.shape)
  let newStrides = strides(newShape)
  var newStridesI32 = newSeq[int32](dims.len)
  var oldStridesI32 = newSeq[int32](dims.len)
  var dimsMapI32 = newSeq[int32](dims.len)
  for i in 0 ..< dims.len:
    newStridesI32[i] = int32(newStrides[i])
    oldStridesI32[i] = int32(oldStrides[i])
    dimsMapI32[i] = int32(dims[i])
  result.shape = newShape
  result.data = bePermuteGpu(ctx, x.data, newStridesI32, oldStridesI32, dimsMapI32, x.data.len)

# ─────────────────────────────────────────────────────────────
# 4. REDUCTION: sum / mean / max / argmax along one dimension (the last dim is the most
#    common for LayerNorm/Loss/Sampling, so every dim is supported generally through
#    the flatten [outer, dimSize, inner]).
# ─────────────────────────────────────────────────────────────

proc reduceShapes(shape: seq[int], dim: int): tuple[outer, dimSize, inner: int] =
  var outer = 1
  for i in 0 ..< dim: outer *= shape[i]
  let dimSize = shape[dim]
  var inner = 1
  for i in dim + 1 ..< shape.len: inner *= shape[i]
  (outer, dimSize, inner)

## WHOLE reduction on the GPU (sum/mean/max/argmax): every per-element ADD/MUL/MAX
## now goes through beAdd/beSub/beRelu (real GPU kernels) instead of a CPU
## loop. Technique: a log-depth tree reduction (like Hillis-Steele in section 8)
## - each step combines adjacent pairs with one GPU command over the WHOLE BLOCK at once.
## max(a,b) = b + relu(a-b) -> expressed purely with beSub/beRelu/beAdd, no
## separate max kernel needed. The index gather/scatter (NOT arithmetic)
## still goes through the CPU because backend.nim has no gather/scatter kernel yet -
## a limit of the available kernels, not a choice of this file.

proc gpuTreeReduce(ctx: Backend, data: openArray[float32], outer, dimSize, inner: int,
                    combine: proc(ctx: Backend, a, b: openArray[float32]): seq[float32] {.nimcall.}): seq[float32] =
  ## Combines all `dimSize` elements in adjacent pairs, log2(dimSize) steps, each
  ## step one GPU command over the whole block (no CPU loop adding/maxing per element).
  var cur = newSeq[float32](outer * dimSize * inner)
  for i in 0 ..< cur.len: cur[i] = data[i]
  var curDim = dimSize
  while curDim > 1:
    let half = curDim div 2
    let rem = curDim mod 2
    var a = newSeq[float32](outer * half * inner)
    var b = newSeq[float32](outer * half * inner)
    for o in 0 ..< outer:
      for d in 0 ..< half:
        for i in 0 ..< inner:
          let outIdx = (o * half + d) * inner + i
          a[outIdx] = cur[(o * curDim + 2 * d) * inner + i]
          b[outIdx] = cur[(o * curDim + 2 * d + 1) * inner + i]
    let combined = combine(ctx, a, b)
    let newDim = half + rem
    var next = newSeq[float32](outer * newDim * inner)
    for o in 0 ..< outer:
      for d in 0 ..< half:
        for i in 0 ..< inner:
          next[(o * newDim + d) * inner + i] = combined[(o * half + d) * inner + i]
      if rem == 1:
        for i in 0 ..< inner:
          next[(o * newDim + half) * inner + i] = cur[(o * curDim + curDim - 1) * inner + i]
    cur = next
    curDim = newDim
  result = cur  # curDim == 1: outer*inner elements

proc gpuMaxCombine(ctx: Backend, a, b: openArray[float32]): seq[float32] =
  ## max(a,b) = b + relu(a-b): purely beSub + beRelu + beAdd, on a real GPU.
  let diff = beSub(ctx, a, b)
  let reluDiff = beRelu(ctx, diff)
  result = beAdd(ctx, reluDiff, b)

proc beSum*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  result = gpuTreeReduce(ctx, x, outer, dimSize, inner, beAdd)

proc beMean*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  let summed = gpuTreeReduce(ctx, x, outer, dimSize, inner, beAdd)
  # Divides with one GPU command (beMul by 1/d) instead of a CPU division loop.
  let invD = newSeq[float32](summed.len).mapIt(1'f32 / float32(dimSize))
  result = beMul(ctx, summed, invD)

proc beMax*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  result = gpuTreeReduce(ctx, x, outer, dimSize, inner, gpuMaxCombine)

proc beArgMax*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[int] =
  ## REAL GPU PORT: both the value AND the index computed on the GPU in one command through
  ## beArgmaxIndexGpu (argmax_kernel.cu/.comp/.metal - a tree reduce keeping the
  ## (value, index) pair in shared/threadgroup memory right on the device),
  ## no more "GPU max then read back to the CPU to find the index" step as before.
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  let (_, idx) = beArgmaxIndexGpu(ctx, @x, outer, dimSize, inner)
  result = newSeq[int](idx.len)
  for i in 0 ..< idx.len: result[i] = int(idx[i])

proc beArgMax*(ctx: Backend, x: openArray[float32]): int =
  ## Utility: flat argmax over a whole 1D array (e.g. the logits of one token, greedy
  ## sampling) - value + index computed together on the GPU (see the note above).
  doAssert x.len > 0
  let (_, idx) = beArgmaxIndexGpu(ctx, @x, 1, x.len, 1)
  result = int(idx[0])

# ─────────────────────────────────────────────────────────────
# 5. CONCAT / STACK / SPLIT
# ─────────────────────────────────────────────────────────────

proc concat*(ctx: Backend, tensors: openArray[Tensor], dim: int): Tensor =
  ## LIKE TORCH: the host only concatenates the data flat (sequential memcpy - NO
  ## per-element index computation) + prepares a small O(numTensors) table (base offset
  ## in the flat buffer, size along dim, prefix offset along dim) and
  ## lets the concat_gather kernel (CUDA/Vulkan/Metal) pick the right source for
  ## each element right on the GPU.
  doAssert tensors.len > 0, "concat: needs at least 1 tensor"
  let baseShape = tensors[0].shape
  for t in tensors:
    for i, s in t.shape:
      if i != dim: doAssert s == baseShape[i], "concat: shapes differ outside dimension dim"
  var newShape = baseShape
  var totalDim = 0
  for t in tensors: totalDim += t.shape[dim]
  newShape[dim] = totalDim
  var inner = 1
  for i in dim + 1 ..< baseShape.len: inner *= baseShape[i]

  var flatSrc = newSeq[float32]()
  var srcBase = newSeq[int32](tensors.len)
  var srcDimSize = newSeq[int32](tensors.len)
  var srcDimOffset = newSeq[int32](tensors.len)
  var dOff = 0
  for i, t in tensors:
    srcBase[i] = int32(flatSrc.len)
    srcDimSize[i] = int32(t.shape[dim])
    srcDimOffset[i] = int32(dOff)
    flatSrc.add(t.data)
    dOff += t.shape[dim]

  result.shape = newShape
  result.data = beConcatGpu(ctx, flatSrc, srcBase, srcDimSize, srcDimOffset, totalDim, inner, numelShape(newShape))

proc stack*(ctx: Backend, tensors: openArray[Tensor], dim: int = 0): Tensor =
  ## stack adds a NEW dimension at position `dim` (unlike concat - it does not join along an
  ## existing dimension). Implemented by reshaping each tensor with an extra dimension of 1, then concat
  ## (the concat path is really on the GPU, see above).
  doAssert tensors.len > 0, "stack: needs at least 1 tensor"
  var reshaped = newSeq[Tensor](tensors.len)
  for i, t in tensors:
    var s = t.shape
    s.insert(1, dim)
    reshaped[i] = view(t, s)
  result = concat(ctx, reshaped, dim)

proc split*(ctx: Backend, x: Tensor, splitSize: int, dim: int): seq[Tensor] =
  ## LIKE TORCH: each piece is a contiguous slice - the host passes only 5
  ## integers (dimTotal/thisSize/offset/inner/n), the split_slice kernel computes
  ## each element's source index right on the GPU, no CPU loop.
  let dimTotal = x.shape[dim]
  var outer = 1
  for i in 0 ..< dim: outer *= x.shape[i]
  var inner = 1
  for i in dim + 1 ..< x.shape.len: inner *= x.shape[i]
  var offset = 0
  while offset < dimTotal:
    let thisSize = min(splitSize, dimTotal - offset)
    var newShape = x.shape
    newShape[dim] = thisSize
    var piece: Tensor
    piece.shape = newShape
    piece.data = beSplitSliceGpu(ctx, x.data, dimTotal, thisSize, offset, inner, numelShape(newShape))
    result.add(piece)
    offset += thisSize

# ─────────────────────────────────────────────────────────────
# 6. DROPOUT - REAL GPU PORT (round 2): the RNG now runs a counter-based xorshift32
#    RIGHT ON THE GPU (dropout_rng_kernel.cu/.comp/.metal through beDropoutMaskGpu)
#    instead of std/random on the host - the seed (uint32, incremented by the caller every
#    training step so each dropout differs) is the only input, no
#    `var Rand` needed any more.
# ─────────────────────────────────────────────────────────────

proc beDropout*(ctx: Backend, x: openArray[float32], p: float32, training: bool,
                 seed: uint32): seq[float32] =
  ## Inverted dropout (like torch.nn.Dropout): returns x unchanged in eval, in
  ## training multiplies by a mask generated on the GPU (0 or 1/(1-p)) through beMul (GPU).
  if not training or p <= 0'f32:
    result = newSeq[float32](x.len)
    for i in 0 ..< x.len: result[i] = x[i]
    return
  doAssert p < 1'f32, "dropout: p must be < 1"
  let mask = beDropoutMaskGpu(ctx, x.len, p, seed)
  result = beMul(ctx, @x, mask)

proc dropoutMask*(ctx: Backend, n: int, p: float32, seed: uint32): seq[float32] =
  ## Returns the mask (0 or 1/(1-p)), generated directly on the GPU (beDropoutMaskGpu), so
  ## it is reused for BOTH forward and backward (multiplying by the same mask during
  ## backprop instead of drawing again - required for correct training: same seed
  ## -> same mask, because the kernel is counter-based and deterministic in (seed, i)).
  if p <= 0'f32:
    result = newSeq[float32](n)
    for i in 0 ..< n: result[i] = 1'f32
    return
  result = beDropoutMaskGpu(ctx, n, p, seed)

# ─────────────────────────────────────────────────────────────
# 7. COMPARISON (gt/lt/eq/ge/le) - used for top-p sampling, masking
# ─────────────────────────────────────────────────────────────

## COMPARISONS ON THE GPU: the COMPUTATION (the difference a-b, and relu to separate the sign) runs
## through beSub/beRelu (real GPU). Materialising the result as a seq[bool] is only
## reading back an already computed float (compared with 0) - not compute;
## backend.nim has no kernel producing a bitmask directly, so this final read
## is required whatever the implementation.

## REAL GPU PORT (round 2): comparisons now run one compare_kernel command on the GPU
## (beCompareGpuOp - CUDA/Vulkan/Metal, opcode 0=gt 1=lt 2=eq 3=ge 4=le,
## see compare_kernel.cu/.comp/.metal) producing the bitmask directly, no more
## intermediate sub/relu + host readback as in the previous version.

proc beGt*(ctx: Backend, a, b: openArray[float32]): seq[bool] =
  doAssert a.len == b.len
  result = beCompareGpuOp(ctx, @a, @b, 0)

proc beLt*(ctx: Backend, a, b: openArray[float32]): seq[bool] =
  doAssert a.len == b.len
  result = beCompareGpuOp(ctx, @a, @b, 1)

proc beEq*(ctx: Backend, a, b: openArray[float32]): seq[bool] =
  doAssert a.len == b.len
  result = beCompareGpuOp(ctx, @a, @b, 2)

proc beGe*(ctx: Backend, a, b: openArray[float32]): seq[bool] =
  doAssert a.len == b.len
  result = beCompareGpuOp(ctx, @a, @b, 3)

proc beLe*(ctx: Backend, a, b: openArray[float32]): seq[bool] =
  doAssert a.len == b.len
  result = beCompareGpuOp(ctx, @a, @b, 4)

proc maskedFill*(x: Tensor, mask: openArray[bool], value: float32): Tensor =
  doAssert mask.len == x.data.len
  result.shape = x.shape
  result.data = newSeq[float32](x.data.len)
  for i in 0 ..< x.data.len:
    result.data[i] = if mask[i]: value else: x.data[i]

proc whereT*(cond: openArray[bool], a, b: Tensor): Tensor =
  doAssert cond.len == a.data.len and a.data.len == b.data.len
  result.shape = a.shape
  result.data = newSeq[float32](a.data.len)
  for i in 0 ..< a.data.len:
    result.data[i] = if cond[i]: a.data[i] else: b.data[i]

# ─────────────────────────────────────────────────────────────
# 8. PARALLEL SCAN (Hillis-Steele, log-depth) - torch.cumsum/cumprod along any
#    dimension, and a general scan with any associative operator
#    - needed for Mamba/SSM (the selective-scan recurrence h_t = a_t*h_{t-1}+b_t
#    is an "affine scan": combining (a1,b1) and (a2,b2) -> (a1*a2,
#    a2*b1+b2), then a parallel prefix over it).
#
#    REAL GPU: each Hillis-Steele step (offset = 1,2,4,8,...) puts ALL the
#    elements of that step into one real GPU command (beAdd/beMul - kernels running for
#    real on Metal/CUDA/OpenCL/Vulkan, not emulated) instead of a per-element
#    CPU loop. The cur/prev index "gather" is still done on the CPU (an
#    indexing/reshape step, not arithmetic), but the actual
#    ADDS/MULS - most of the FLOPs when dimSize is large - run on the
#    GPU exactly like the beAdd/beMul used everywhere else in this codebase.
# ─────────────────────────────────────────────────────────────

proc scanStepIndices(outer, dimSize, inner, offset: int): tuple[cur, prev: seq[int]] =
  ## Flat indices (cur, prev) of ALL positions taking part in one Hillis-Steele step
  ## with the given `offset` - exactly the set of MUTUALLY INDEPENDENT positions of this step
  ## (read from a snapshot, not overwriting each other), so gathering them into one pair of
  ## flat arrays and feeding the whole block to one GPU command (beAdd/beMul) is valid.
  var cur = newSeq[int]()
  var prev = newSeq[int]()
  for o in 0 ..< outer:
    for d in offset ..< dimSize:
      for i in 0 ..< inner:
        cur.add((o * dimSize + d) * inner + i)
        prev.add((o * dimSize + (d - offset)) * inner + i)
  (cur, prev)

proc applyHillisSteeleGpu(ctx: Backend, data: var seq[float32], outer, dimSize, inner: int, useMul: bool) =
  ## In-place, inclusive prefix sum (useMul=false) or prefix product
  ## (useMul=true) along `dimSize`. Each step offset=1,2,4,... gathers
  ## a snapshot of the participating positions into 2 flat arrays (cur, prev) and calls
  ## beAdd/beMul DIRECTLY (real GPU through ctx) on the WHOLE BLOCK at once, then
  ## scatters the result back into `data` - no more CPU loop adding/multiplying
  ## element by element.
  var offset = 1
  while offset < dimSize:
    let snapshot = data
    let (curIdx, prevIdx) = scanStepIndices(outer, dimSize, inner, offset)
    if curIdx.len > 0:
      var curVals = newSeq[float32](curIdx.len)
      var prevVals = newSeq[float32](curIdx.len)
      for j in 0 ..< curIdx.len:
        curVals[j] = snapshot[curIdx[j]]
        prevVals[j] = snapshot[prevIdx[j]]
      let combined = if useMul: beMul(ctx, curVals, prevVals) else: beAdd(ctx, curVals, prevVals)
      for j in 0 ..< curIdx.len:
        data[curIdx[j]] = combined[j]
    offset = offset shl 1

proc beCumsum*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  ## Equivalent to torch.cumsum(x, dim): an INCLUSIVE prefix sum,
  ## keeping the shape. Each Hillis-Steele step runs through beAdd (real GPU).
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len: result[i] = x[i]
  applyHillisSteeleGpu(ctx, result, outer, dimSize, inner, useMul = false)

proc cumsumT*(ctx: Backend, x: Tensor, dim: int): Tensor =
  result.shape = x.shape
  result.data = beCumsum(ctx, x.data, x.shape, dim)

proc beCumprod*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  ## torch.cumprod(x, dim). The same Hillis-Steele structure as cumsum but
  ## combining with multiplication (still associative -> a valid parallel
  ## scan), through beMul (real GPU). Kept separate instead of "exp(cumsum(log(x)))"
  ## to be CORRECT for negative/zero values like torch (log(negative) = NaN, torch.cumprod
  ## is not).
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len: result[i] = x[i]
  applyHillisSteeleGpu(ctx, result, outer, dimSize, inner, useMul = true)

proc cumprodT*(ctx: Backend, x: Tensor, dim: int): Tensor =
  result.shape = x.shape
  result.data = beCumprod(ctx, x.data, x.shape, dim)

type
  ScanPair* = tuple[a, b: float32]
    ## An (a,b) pair for the affine scan h_t = a_t * h_{t-1} + b_t (exactly the
    ## Mamba/SSM selective-scan recurrence after discretisation).

proc combineAffine(l, r: ScanPair): ScanPair =
  ## Combines 2 consecutive affine steps: l applied FIRST, r AFTER (h -> l(h) -> r(l(h))).
  ## r(l(h)) = r.a*(l.a*h + l.b) + r.b = (r.a*l.a)*h + (r.a*l.b + r.b)
  ## (this CPU-scalar version is kept so testops.nim can compare sequentially - see
  ## sequentialAffine in testops.nim; the REAL GPU path is beAffineScan
  ## below, using exactly the same combination formula but with beMul/beAdd over the
  ## WHOLE BLOCK instead of one scalar float pair.)
  (a: r.a * l.a, b: r.a * l.b + r.b)

proc beAffineScan*(ctx: Backend, aCoef, bCoef: openArray[float32],
                    shape: seq[int], dim: int): seq[float32] =
  ## GENERAL parallel scan for the affine recurrence h_t = a_t*h_{t-1} + b_t
  ## (state space/SSM/Mamba selective scan, linear GRU, ...) along
  ## `dim`. Returns h_t (INCLUSIVE, h_{-1} = 0) for EVERY t at once. Each
  ## Hillis-Steele step applies the combineAffine FORMULA (r.a*l.a, r.a*l.b + r.b) to
  ## ALL positions of that step AT ONCE with 3 real GPU commands
  ## (beMul, beMul, beAdd) - no more CPU loop computing each (a,b) pair.
  doAssert aCoef.len == bCoef.len
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  doAssert inner == 1, "beAffineScan: currently supports only the last dimension (inner=1); " &
    "call permute() first to scan along another dimension"
  var a = newSeq[float32](aCoef.len)
  var b = newSeq[float32](bCoef.len)
  for i in 0 ..< aCoef.len: (a[i] = aCoef[i]; b[i] = bCoef[i])
  var offset = 1
  while offset < dimSize:
    let aSnap = a
    let bSnap = b
    let (curIdx, prevIdx) = scanStepIndices(outer, dimSize, inner, offset)
    if curIdx.len > 0:
      var aCur = newSeq[float32](curIdx.len)
      var bCur = newSeq[float32](curIdx.len)
      var aPrev = newSeq[float32](curIdx.len)
      var bPrev = newSeq[float32](curIdx.len)
      for j in 0 ..< curIdx.len:
        aCur[j] = aSnap[curIdx[j]]; bCur[j] = bSnap[curIdx[j]]
        aPrev[j] = aSnap[prevIdx[j]]; bPrev[j] = bSnap[prevIdx[j]]
      # combined.a = r.a * l.a  (r = cur, l = prev)
      let combinedA = beMul(ctx, aCur, aPrev)
      # combined.b = r.a * l.b + r.b
      let rablTerm = beMul(ctx, aCur, bPrev)
      let combinedB = beAdd(ctx, rablTerm, bCur)
      for j in 0 ..< curIdx.len:
        a[curIdx[j]] = combinedA[j]
        b[curIdx[j]] = combinedB[j]
    offset = offset shl 1
  result = b  # h_t with h_{-1}=0 is exactly the b component after the scan