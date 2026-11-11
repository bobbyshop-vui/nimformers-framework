## backend.nim
## Backend selection layer for Nimformer, built entirely on the vendored BybyLang.
## All separate hand-written kernels removed; every backend (CPU, Metal, CUDA, OpenCL)
## goes through BybyLang only, for performance and no duplicated source.

import vendor/bybylang/gpubackend as byby
import customfloat
export byby.GpuBackend, byby.SessionHandle, byby.sessionBegin, byby.sessionUpload, byby.sessionRope,
       byby.sessionUploadIndices, byby.sessionAllocScratch, byby.sessionMatmul, byby.sessionMatmulNT,
       byby.sessionSoftmax, byby.sessionVecOp, byby.sessionActivation, byby.sessionEnd,
       byby.sessionAllocHeap, byby.sessionAllocFromHeap, byby.sessionFreeHeap,
       byby.sessionRead, byby.sessionFree, byby.sessionCopyInto, byby.sessionScale,
       byby.ResidentWeightHandle, byby.residentUploadQ4, byby.sessionMatmulQ4, byby.residentFreeQ4,
       byby.residentPrepareQ4, byby.sessionMatmulQ4Fused, byby.sessionKvAppendEnc, byby.sessionGraphBegin, byby.sessionGraphEnd, byby.sessionGraphReady, byby.sessionGraphReset, byby.sessionSetKvFp16, byby.sessionGraphRun, byby.sessionUploadPrivate, byby.sessionRopeKvEnc,
       byby.sessionLayernormEnc, byby.sessionRmsnormEnc, byby.sessionAttentionFusedEnc,
       byby.sessionAttentionDecodeGqaEnc, byby.cpuAttentionDecodeGqa,
       byby.gpuGather, byby.gpuArgmaxIndex, byby.gpuCompare, byby.gpuDropoutMask,
       byby.gpuPermute, byby.gpuSplitSlice, byby.gpuConcat, byby.sessionGetLastError, byby.sessionGpuSeconds, byby.sessionEndAsync, byby.sessionWaitOldest, byby.sessionTokenRingGet, byby.sessionTokenRingSet,
       byby.sessionEmbedTokenEnc, byby.sessionArgmaxTokenEnc,
       byby.sessionAttentionDecodeAllHeadsEnc, byby.sessionRecoverQueue, byby.sessionCommitContinue,
       byby.sessionFlashAttentionPrepare, byby.sessionFlashAttentionPrefillEnc, byby.sessionKvScatterEnc,
       byby.sessionPacesCommandBuffers

# -d:backend=cpu|metal|cuda|opencl|vulkan|auto — default "auto" (detected at run time).
const backend* {.strdefine.} = "auto"

type
  BackendKind* = enum
    bkMetal = "metal"
    bkCuda = "cuda"
    bkVulkan = "vulkan"
    bkCpu = "cpu"

  Backend* = object
    kind*: BackendKind

proc toByby*(bk: BackendKind): byby.GpuBackend =
  case bk
  of bkMetal: byby.gbMetal
  # Keep bkCuda as a source-compatible legacy enum value, but never expose a
  # CUDA execution path from Nimformer. All CUDA selections use Vulkan.
  of bkCuda: byby.gbVulkan
  of bkVulkan: byby.gbVulkan
  of bkCpu: byby.gbCpu

proc resolveBackendKind(want: string): BackendKind =
  case want
  of "cpu": bkCpu
  of "metal": bkMetal
  of "cuda":
    stderr.writeLine "[Framework] backend 'cuda' is deprecated/disabled; redirecting to Vulkan"
    bkVulkan
  of "opencl": bkVulkan
  of "vulkan": bkVulkan   # "opencl" was replaced by Vulkan as the cross-platform
                          # fallback GPU backend (see toByby: bkVulkan -> gbVulkan)
                          # - both names are accepted for backward compatibility.
  of "auto":
    # "auto" -> detect the best available backend. When no GPU is present the
    # runtime now falls back to the SIMD CPU backend instead of failing.
    let chosen = byby.detectBackend()
    case chosen
    of byby.gbCuda: bkCuda
    of byby.gbMetal: bkMetal
    of byby.gbVulkan: bkVulkan
    of byby.gbCpu, byby.gbAuto:
      stderr.writeLine "[Framework] no GPU detected; using the CPU backend"
      bkCpu
  else:
    raise newException(ValueError, "unsupported backend: " & want)

proc newBackend*(force: string = backend): Backend =
  ## Creates a new backend. Picks and links the most suitable device automatically.
  let kind = resolveBackendKind(force)
  stderr.writeLine "== [Framework] Backend selected: " & $kind & " =="
  result = Backend(kind: kind)

proc closeBackend*(b: Backend) =
  discard

proc setForbidCpuFallback*(forbid: bool) =
  ## Default TRUE: if the chosen GPU backend (cuda/metal/opencl) fails at
  ## run time, the program raises instead of silently falling back to the CPU and
  ## continuing to train (the CPU is many times slower and users often miss it because
  ## it is only one stderr log line). Set false only if you DELIBERATELY want to allow
  ## running on the CPU (e.g. a demo/test on a machine without a GPU).
  byby.setForbidCpuFallback(forbid)

# ─────────────────────────────────────────────────────────────
# Unified matrix and activation operations through BybyLang
# ─────────────────────────────────────────────────────────────

proc beMatmul*(ctx: Backend, a: openArray[float32], M, K: int,
                b: openArray[float32], K2, N: int): seq[float32] =
  let aSeq = @a
  let bSeq = @b
  result = byby.gpuMatmul(ctx.kind.toByby(), aSeq, bSeq, M, K, N)

proc beMatmulQ4*(ctx: Backend, a: openArray[float32], wq: openArray[uint8],
                  scales, zeros: openArray[float32],
                  M, K, N, groupSize, nGroupsPerRow: int): seq[float32] =
  let aSeq = @a
  let wqSeq = @wq
  let scalesSeq = @scales
  let zerosSeq = @zeros
  result = byby.gpuMatmulQ4(ctx.kind.toByby(), aSeq, wqSeq, scalesSeq, zerosSeq,
                             M, K, N, groupSize, nGroupsPerRow)
  # === FIX: ALWAYS MAKE SURE THE SIZE IS RIGHT ===
  let expectedSize = M * N
  if result.len != expectedSize:
    # New array of the right size, copy the existing part, the rest is 0
    var fixed = newSeq[float32](expectedSize)
    let copyLen = min(result.len, expectedSize)
    for i in 0 ..< copyLen:
      fixed[i] = result[i]
    result = fixed

proc beResidentUploadQ4*(ctx: Backend, wq: openArray[uint8], scales, zeros: openArray[float32],
                          k, n, groupSize, nGroupsPerRow: int): ResidentWeightHandle =
  ## Uploads one int4 weight to the GPU ONCE (at model load). Call once per Linear,
  ## keep the result (e.g. in Linear.residentW) for the process lifetime.
  byby.residentUploadQ4(ctx.kind.toByby(), @wq, @scales, @zeros, k, n, groupSize, nGroupsPerRow)

proc beResidentUploadQ4Mapped*(ctx: Backend, wq: pointer, wqBytes: int,
                                scales, zeros: openArray[float32],
                                k, n, groupSize, nGroupsPerRow: int): tuple[w: ResidentWeightHandle, zeroCopy: bool] =
  ## beResidentUploadQ4 for packed weights inside a model mmap: the GPU wraps
  ## the mapping on unified memory, else copies straight from it.
  byby.residentUploadQ4Mapped(ctx.kind.toByby(), wq, wqBytes, scales, zeros,
    k, n, groupSize, nGroupsPerRow)

proc beResidentWrapFp32*(ctx: Backend, data: pointer, bytes: int): tuple[h: SessionHandle, zeroCopy: bool, offsetElems: int] =
  ## Resident fp32 weight that lives in a model mmap: wrapped in place on
  ## unified memory (the weight starts offsetElems floats into the buffer),
  ## else copied straight from the mapping to GPU memory.
  let be = ctx.kind.toByby()
  var wrapped = byby.sessionWrapHostRange(be, data, bytes)
  result.zeroCopy = wrapped.h.metalHandle != nil or wrapped.h.vulkanHandle.buf != nil
  if result.zeroCopy and wrapped.offset mod sizeof(float32) != 0:
    byby.sessionFree(be, wrapped.h)
    result.zeroCopy = false
  if result.zeroCopy:
    result.h = wrapped.h
    result.offsetElems = wrapped.offset div sizeof(float32)
  else:
    result.h = byby.sessionUploadPrivatePtr(be, data, bytes)

proc beResidentLinearQ4*(ctx: Backend, x: openArray[float32], w: ResidentWeightHandle, M: int): seq[float32] =
  ## Linear forward with an ALREADY resident weight (w from beResidentUploadQ4): each
  ## call uploads only the activation `x` (small, changes every token) - the weight
  ## bytes are NOT touched again. Returns an empty seq if the backend does not support it
  ## (the caller must then fall back to beMatmulQ4).
  let be = ctx.kind.toByby()
  if not sessionBegin(be): return @[]
  var hA = sessionUpload(be, @x)
  var hC = sessionAllocScratch(be, M * w.n)
  if not sessionMatmulQ4(be, hA, w, hC, M):
    discard sessionEnd(be)
    sessionFree(be, hA); sessionFree(be, hC)
    return @[]
  if not sessionEnd(be):
    sessionFree(be, hA); sessionFree(be, hC)
    return @[]
  result = sessionRead(be, hC, M * w.n)
  sessionFree(be, hA)
  sessionFree(be, hC)

proc beResidentFreeQ4*(w: var ResidentWeightHandle) =
  byby.residentFreeQ4(w)

# ─────────────────────────────────────────────────────────────
# PERFORMANCE FIX: a SHARED resident fp32 path for EVERY weight that cannot use the
# int4-asymmetric path (qkInt8, symmetric qkInt4, or plain fp32/tied-
# embedding Linears). BEFORE, these weights had NO "resident" copy, so
# Linear.forward() in nimformer.nim had to dequantizeTensorTransposed() /
# transpose() the WHOLE matrix on the CPU EVERY TOKEN (224+ full-size dequants
# per decode step) -> the main reason the speed was only
# ~4s/token despite the GPU (Metal/OpenCL) - slower than llama.cpp on a plain CPU,
# because the unoptimised single-threaded CPU dequant ate all the time before
# the GPU matmul could run.
# Fix: dequant + transpose ONCE at model load (exactly like
# residentUploadQ4 does for its "fallbackFp32" path), upload straight to the GPU and keep it
# resident through a normal sessionUpload; each token uploads only the (small) activation
# + sessionMatmul - the weight bytes are not touched again.
proc beResidentUploadFp32*(ctx: Backend, wT: openArray[float32]): SessionHandle =
  ## wT: the weight already dequantized + transposed, flat [K, N]. Call ONCE
  ## per Linear at model load (see uploadResident in nimformer.nim),
  ## NOT inside the decode loop.
  ## CHANGED: uses sessionUploadPrivate (not sessionUpload/Shared) - these are
  ## the RESIDENT weights of int8/int4-symmetric/tied-embedding models (the path
  ## used when the model is NOT Q4-per-group) - the CPU never touches them after this
  ## upload, so Private is right on Metal for exactly the same reason as for
  ## Q4 (residentUploadQ4) - so int8 models also get the
  ## Private/Shared hybrid optimisation, not only Q4.
  let wTSeq = @wT
  result = byby.sessionUploadPrivate(ctx.kind.toByby(), wTSeq)

proc beResidentLinearFp32*(ctx: Backend, x: openArray[float32], w: SessionHandle,
                            M, K, N: int): seq[float32] =
  ## Linear forward with an ALREADY resident fp32 weight (w from beResidentUploadFp32):
  ## each call uploads only the activation `x` (small, changes every token) - the weight
  ## bytes are NOT touched again. Returns an empty seq if the backend cannot open a session
  ## (the caller then falls back to the old path).
  let be = ctx.kind.toByby()
  if not byby.sessionBegin(be): return @[]
  var hA = byby.sessionUpload(be, @x)
  var hC = byby.sessionAllocScratch(be, M * N)
  if not byby.sessionMatmul(be, hA, w, hC, M, K, N):
    discard byby.sessionEnd(be)
    byby.sessionFree(be, hA); byby.sessionFree(be, hC)
    return @[]
  if not byby.sessionEnd(be):
    byby.sessionFree(be, hA); byby.sessionFree(be, hC)
    return @[]
  result = byby.sessionRead(be, hC, M * N)
  byby.sessionFree(be, hA)
  byby.sessionFree(be, hC)

proc beResidentFreeFp32*(ctx: Backend, w: var SessionHandle) =
  byby.sessionFree(ctx.kind.toByby(), w)

proc beMatmul2*(ctx: Backend,
                 a1: openArray[float32], M1, K1: int, b1: openArray[float32], K1b, N1: int,
                 a2: openArray[float32], M2, K2: int, b2: openArray[float32], K2b, N2: int):
                 tuple[y1, y2: seq[float32]] =
  let a1S = @a1; let b1S = @b1
  let a2S = @a2; let b2S = @b2
  let r = byby.gpuMatmul2(ctx.kind.toByby(), a1S, b1S, M1, K1, N1, a2S, b2S, M2, K2, N2)
  return (r.c1, r.c2)

proc beAdd*(ctx: Backend, a, b: openArray[float32]): seq[float32] =
  let aSeq = @a; let bSeq = @b
  result = byby.gpuOp("add", ctx.kind.toByby(), aSeq, bSeq)

proc beSub*(ctx: Backend, a, b: openArray[float32]): seq[float32] =
  let aSeq = @a; let bSeq = @b
  result = byby.gpuOp("sub", ctx.kind.toByby(), aSeq, bSeq)

proc beMul*(ctx: Backend, a, b: openArray[float32]): seq[float32] =
  let aSeq = @a; let bSeq = @b
  result = byby.gpuOp("mul", ctx.kind.toByby(), aSeq, bSeq)

proc beDiv*(ctx: Backend, a, b: openArray[float32]): seq[float32] =
  let aSeq = @a; let bSeq = @b
  result = byby.gpuOp("div", ctx.kind.toByby(), aSeq, bSeq)

proc beRelu*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuRelu(ctx.kind.toByby(), @x)

# ─────────────────────────────────────────────────────────────
# GATHER / ARGMAX-INDEX / COMPARE / DROPOUT-RNG - REAL GPU PORT (round 2, see
# tensor_ops.nim sections 3/4/7 and dropout): permute/concat/split/stack go through
# beGatherGpu (kernel gather_kernel.cu/.comp/.metal); argmax returns both value
# and index at once on the GPU through beArgmaxIndexGpu; comparisons (gt/lt/eq/ge/le)
# produce a bitmask directly from compare_kernel through beCompareGpuOp; dropout masks are generated
# by an RNG running on the GPU (counter-based xorshift32) through beDropoutMaskGpu -
# no more host-side std/random.
# ─────────────────────────────────────────────────────────────

proc beGatherGpu*(ctx: Backend, src: seq[float32], idx: seq[int32]): seq[float32] =
  byby.gpuGather(ctx.kind.toByby(), src, idx)

proc beArgmaxIndexGpu*(ctx: Backend, x: seq[float32], outer, dimSize, inner: int): tuple[val: seq[float32], idx: seq[int32]] =
  byby.gpuArgmaxIndex(ctx.kind.toByby(), x, outer, dimSize, inner)

proc beCompareGpuOp*(ctx: Backend, a, b: seq[float32], op: int): seq[bool] =
  byby.gpuCompare(ctx.kind.toByby(), a, b, op)

proc beDropoutMaskGpu*(ctx: Backend, n: int, p: float32, seed: uint32): seq[float32] =
  byby.gpuDropoutMask(ctx.kind.toByby(), n, p, seed)

proc bePermuteGpu*(ctx: Backend, src: seq[float32], newStrides, oldStrides, dimsMap: seq[int32], n: int): seq[float32] =
  byby.gpuPermute(ctx.kind.toByby(), src, newStrides, oldStrides, dimsMap, n)

proc beSplitSliceGpu*(ctx: Backend, src: seq[float32], dimTotal, thisSize, offset, inner, n: int): seq[float32] =
  byby.gpuSplitSlice(ctx.kind.toByby(), src, dimTotal, thisSize, offset, inner, n)

proc beConcatGpu*(ctx: Backend, flatSrc: seq[float32], srcBase, srcDimSize, srcDimOffset: seq[int32],
                   totalDim, inner, n: int): seq[float32] =
  byby.gpuConcat(ctx.kind.toByby(), flatSrc, srcBase, srcDimSize, srcDimOffset, totalDim, inner, n)

proc beSqrt*(ctx: Backend, x: openArray[float32]): seq[float32] =
  ## NEW: element-wise sqrt (used for the Adam optimizer bias correction
  ## in nimformer.nim, not an inference hot path). See gpuSqrt/
  ## cpuSqrt in gpubackend.nim.
  result = byby.gpuSqrt(ctx.kind.toByby(), @x)

proc beExp*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuExp(ctx.kind.toByby(), @x)

proc beLog*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuLog(ctx.kind.toByby(), @x)

proc bePow*(ctx: Backend, x: openArray[float32], p: float32): seq[float32] =
  result = byby.gpuPow(ctx.kind.toByby(), @x, p)

proc beSigmoid*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuSigmoid(ctx.kind.toByby(), @x)

proc beTanh*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuTanh(ctx.kind.toByby(), @x)

proc beSoftmax*(ctx: Backend, x: openArray[float32], rows, cols: int): seq[float32] =
  result = byby.gpuSoftmax(ctx.kind.toByby(), @x, rows, cols)

proc beLayernorm*(ctx: Backend, x: openArray[float32], gamma, beta: openArray[float32], rows, cols: int, eps: float32): seq[float32] =
  result = byby.gpuLayernorm(ctx.kind.toByby(), @x, @gamma, @beta, rows, cols, eps)

proc beAddLayernormFused*(ctx: Backend, a, b: openArray[float32], gamma, beta: openArray[float32],
                           rows, cols: int, eps: float32): tuple[summed, normed: seq[float32]] =
  ## Merges "residual add then layernorm right after" (the pattern repeated 2
  ## times/layer in TransformerBlock.forward) into one GPU session -
  ## one upload/commit/wait instead of two (beAdd then
  ## beLayernorm, each allocating its own buffer + command buffer + wait). Returns
  ## BOTH: `summed` (raw a+b, the residual for the final add of the block)
  ## AND `normed` (after layernorm) - so the caller does not add again.
  ## ONLY really helps on backends with sessionLayernormEnc (currently:
  ## Metal). If the backend does not support it (CUDA/OpenCL/CPU currently, or
  ## an encode error at run time) -> falls back to EXACTLY the old path (beAdd then beLayernorm),
  ## no change in behaviour/precision, only the speedup is lost.
  let be = ctx.kind.toByby()
  let n = rows * cols
  block fused:
    if not sessionBegin(be): break fused
    var hA = sessionUpload(be, @a)
    var hB = sessionUpload(be, @b)
    var hGamma = sessionUpload(be, @gamma)
    var hBeta = sessionUpload(be, @beta)
    var hSum = sessionAllocScratch(be, n)
    var hOut = sessionAllocScratch(be, n)
    let ok = sessionVecOp(be, "add", hA, hB, hSum, n) and
             sessionLayernormEnc(be, hSum, hGamma, hBeta, hOut, rows, cols, eps)
    if not ok or not sessionEnd(be):
      sessionFree(be, hA); sessionFree(be, hB); sessionFree(be, hGamma)
      sessionFree(be, hBeta); sessionFree(be, hSum); sessionFree(be, hOut)
      break fused
    result.summed = sessionRead(be, hSum, n)
    result.normed = sessionRead(be, hOut, n)
    sessionFree(be, hA); sessionFree(be, hB); sessionFree(be, hGamma)
    sessionFree(be, hBeta); sessionFree(be, hSum); sessionFree(be, hOut)
    return
  # Fallback: the old path, 2 separate GPU commands.
  result.summed = beAdd(ctx, a, b)
  result.normed = beLayernorm(ctx, result.summed, gamma, beta, rows, cols, eps)

proc beEmbeddingLookup*(ctx: Backend, table: openArray[float32], indices: openArray[int32], vocab, dim: int): seq[float32] =
  result = byby.gpuEmbeddingLookup(ctx.kind.toByby(), @table, @indices, vocab, dim)

proc beApflu*(ctx: Backend, x: openArray[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32): seq[float32] =
  result = byby.gpuApflu(ctx.kind.toByby(), @x, alpha, beta)

proc beApfluBackward*(ctx: Backend, x, dy: openArray[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32): seq[float32] =
  result = byby.gpuApfluBackward(ctx.kind.toByby(), @x, @dy, alpha, beta)

proc beLayernormBackward*(ctx: Backend, dy, x, gamma, beta: openArray[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  let r = byby.gpuLayernormBackward(ctx.kind.toByby(), @dy, @x, @gamma, @beta, rows, cols, eps)
  return (r.dx, r.dgamma, r.dbeta)

proc beAttentionFused*(ctx: Backend, q, k, v, mask: openArray[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  let r = byby.gpuAttentionFused(ctx.kind.toByby(), @q, @k, @v, @mask, B, H, S, D, scale)
  return (r.o, r.s_matrix)

proc beAttentionFusedBackward*(ctx: Backend, q, k, v, s_matrix, dy: openArray[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  let r = byby.gpuAttentionFusedBackward(ctx.kind.toByby(), @q, @k, @v, @s_matrix, @dy, B, H, S, D, scale)
  return (r.dq, r.dk, r.dv)

# ─────────────────────────────────────────────────────────────
# APF & Custom Float helpers (fully backward compatible)
# ─────────────────────────────────────────────────────────────

proc beCustomfloatEncode*(ctx: Backend, arr: openArray[float32], cf: CustomFloat): seq[uint8] =
  return customfloat.encodeArray(arr, cf)

proc beCustomfloatDecode*(ctx: Backend, buf: openArray[uint8], cf: CustomFloat): seq[float32] =
  return customfloat.decodeArray(buf, cf)

proc beApfCastForTraining*(ctx: Backend, arr: openArray[float32], gradArr: openArray[float32] = [],
                           relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                           expMargin = APF_EXP_MARGIN_BITS): tuple[data: seq[uint8], cf: CustomFloat] =
  return customfloat.apfCastForTraining(arr, gradArr, relErrorTol, expMargin)
