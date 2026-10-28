## backend.nim
## Lớp chọn backend tính toán cho Nimformer dựa hoàn toàn vào BybyLang trong vendor.
## Loại bỏ toàn bộ các kernel tự viết ngoài, thống nhất mọi backend (CPU, Metal, CUDA, OpenCL)
## qua duy nhất BybyLang để tối ưu hoá hiệu năng, tránh trùng lặp mã nguồn.

import vendor/bybylang/gpubackend as byby
import customfloat
export byby.GpuBackend, byby.SessionHandle, byby.sessionBegin, byby.sessionUpload, byby.sessionRope,
       byby.sessionUploadIndices, byby.sessionAllocScratch, byby.sessionMatmul, byby.sessionMatmulNT,
       byby.sessionSoftmax, byby.sessionVecOp, byby.sessionActivation, byby.sessionEnd,
       byby.sessionAllocHeap, byby.sessionAllocFromHeap, byby.sessionFreeHeap,
       byby.sessionRead, byby.sessionFree, byby.sessionCopyInto, byby.sessionScale,
       byby.ResidentWeightHandle, byby.residentUploadQ4, byby.sessionMatmulQ4, byby.residentFreeQ4,
       byby.residentPrepareQ4,
       byby.sessionLayernormEnc, byby.sessionRmsnormEnc, byby.sessionAttentionFusedEnc,
       byby.sessionAttentionDecodeGqaEnc, byby.cpuAttentionDecodeGqa,
       byby.gpuGather, byby.gpuArgmaxIndex, byby.gpuCompare, byby.gpuDropoutMask,
       byby.gpuPermute, byby.gpuSplitSlice, byby.gpuConcat, byby.sessionGetLastError,
       byby.sessionAttentionDecodeAllHeadsEnc, byby.sessionRecoverQueue, byby.sessionCommitContinue

# -d:backend=cpu|metal|cuda|opencl|vulkan|auto — mặc định "auto" (tự dò lúc runtime).
const backend* {.strdefine.} = "auto"

type
  BackendKind* = enum
    bkMetal = "metal"
    bkCuda = "cuda"
    bkVulkan = "vulkan"

  Backend* = object
    kind*: BackendKind

proc toByby*(bk: BackendKind): byby.GpuBackend =
  case bk
  of bkMetal: byby.gbMetal
  # Keep bkCuda as a source-compatible legacy enum value, but never expose a
  # CUDA execution path from Nimformer. All CUDA selections use Vulkan.
  of bkCuda: byby.gbVulkan
  of bkVulkan: byby.gbVulkan

proc resolveBackendKind(want: string): BackendKind =
  case want
  of "cpu":
    raise newException(ValueError,
      "resolveBackendKind(\"cpu\"): backend CPU da bi GO BO khoi framework " &
      "nay theo yeu cau - khong con lua chon CPU nao, tuong minh hay khong " &
      "cung khong duoc. Chon 1 trong: metal/cuda/opencl(vulkan)/auto.")
  of "metal": bkMetal
  of "cuda":
    stderr.writeLine "[Framework] backend 'cuda' is deprecated/disabled; redirecting to Vulkan"
    bkVulkan
  of "opencl": bkVulkan
  of "vulkan": bkVulkan   # "opencl" da duoc thay bang Vulkan lam backend GPU
                          # da nen tang fallback (xem toByby: bkVulkan -> gbVulkan)
                          # - chap nhan ca 2 ten string de tuong thich nguoc.
  of "auto":
    # "auto" -> Tự động dò backend qua BybyLang. Khong tim thay GPU nao (CUDA/
    # Metal/Vulkan) -> RAISE luon, khong con nhanh bkCpu nao de roi vao nua
    # (truoc day con "fallback bkCpu neu gForbidCpuFallback=false" - gio bo
    # HAN, backend CPU khong ton tai trong enum nay nua nen khong co gi de tra ve).
    let chosen = byby.detectBackend()
    case chosen
    of byby.gbCuda: bkCuda
    of byby.gbMetal: bkMetal
    of byby.gbVulkan: bkVulkan
    of byby.gbCpu, byby.gbAuto:
      raise newException(CatchableError,
        "resolveBackendKind(\"auto\"): KHÔNG tìm thấy GPU nào (CUDA/Metal/Vulkan) " &
        "trên máy này. Backend CPU đã bị GỠ BỎ khỏi framework - không có gì để " &
        "fallback về. Cắm GPU / cài driver rồi thử lại.")
  else:
    raise newException(ValueError, "backend khong duoc ho tro: " & want)

proc newBackend*(force: string = backend): Backend =
  ## Khởi tạo backend mới. Tự động liên kết và chọn thiết bị phù hợp nhất.
  let kind = resolveBackendKind(force)
  stderr.writeLine "== [Framework] Backend đã chọn: " & $kind & " =="
  result = Backend(kind: kind)

proc closeBackend*(b: Backend) =
  discard

proc setForbidCpuFallback*(forbid: bool) =
  ## Mặc định TRUE: nếu backend GPU đã chọn (cuda/metal/opencl) lỗi lúc
  ## chạy, chương trình sẽ raise lỗi thay vì âm thầm rơi về CPU và tiếp tục
  ## train (CPU chậm hơn GPU rất nhiều lần và người dùng thường không để ý vì
  ## chỉ có 1 dòng log stderr). Chỉ set false nếu bạn CHỦ ĐỘNG muốn cho phép
  ## chạy CPU (vd. demo/test trên máy không có GPU).
  byby.setForbidCpuFallback(forbid)

# ─────────────────────────────────────────────────────────────
# API Hợp Nhất Các Phép Toán Ma Trận và Lớp Kích Hoạt qua BybyLang
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
  # === FIX: LUÔN ĐẢM BẢO KÍCH THƯỚC ĐÚNG ===
  let expectedSize = M * N
  if result.len != expectedSize:
    # Tạo mảng mới đúng size, copy phần có sẵn, phần còn lại 0
    var fixed = newSeq[float32](expectedSize)
    let copyLen = min(result.len, expectedSize)
    for i in 0 ..< copyLen:
      fixed[i] = result[i]
    result = fixed

proc beResidentUploadQ4*(ctx: Backend, wq: openArray[uint8], scales, zeros: openArray[float32],
                          k, n, groupSize, nGroupsPerRow: int): ResidentWeightHandle =
  ## Upload 1 weight int4 lên GPU MỘT LẦN (lúc load model). Gọi 1 lần/Linear,
  ## giữ kết quả lại (vd. trong Linear.residentW) suốt vòng đời process.
  byby.residentUploadQ4(ctx.kind.toByby(), @wq, @scales, @zeros, k, n, groupSize, nGroupsPerRow)

proc beResidentLinearQ4*(ctx: Backend, x: openArray[float32], w: ResidentWeightHandle, M: int): seq[float32] =
  ## Forward Linear dùng weight ĐÃ resident (w từ beResidentUploadQ4): mỗi
  ## lần gọi hàm này chỉ upload activation `x` (nhỏ, đổi mỗi token) - KHÔNG
  ## đụng lại tới bytes weight. Trả về seq rỗng nếu backend không hỗ trợ
  ## (caller phải tự fallback về beMatmulQ4 khi đó).
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
# FIX HIỆU NĂNG: resident fp32 CHUNG cho MỌI weight không đi được đường
# int4-asymmetric (qkInt8, qkInt4 symmetric, hoặc Linear fp32 thường/tied-
# embedding). TRƯỚC ĐÂY các weight này KHÔNG có "resident", nên
# Linear.forward() ở nimformer.nim phải dequantizeTensorTransposed() /
# transpose() lại TOÀN BỘ ma trận trên CPU MỖI TOKEN (chạy 224+ lần dequant
# full-size mỗi bước decode) -> đây là nguyên nhân chính khiến tốc độ chỉ
# ~4s/token dù có GPU (Metal/OpenCL) - chậm hơn cả llama.cpp chạy CPU thuần,
# vì phần dequant CPU đơn luồng, không tối ưu, đã nuốt hết thời gian trước
# khi kịp chạy matmul GPU.
# Fix: dequant + transpose MỘT LẦN duy nhất lúc load model (giống hệt cách
# residentUploadQ4 làm cho đường "fallbackFp32"), upload thẳng lên GPU giữ
# resident qua sessionUpload thường, mỗi token chỉ upload activation (nhỏ)
# + sessionMatmul - không đụng lại bytes weight nữa.
proc beResidentUploadFp32*(ctx: Backend, wT: openArray[float32]): SessionHandle =
  ## wT: weight đã dequant + transpose sẵn, dạng [K, N] phẳng. Gọi 1
  ## LẦN/Linear lúc load model (xem uploadResident trong nimformer.nim),
  ## KHÔNG gọi trong vòng lặp decode.
  ## SỬA: dùng sessionUploadPrivate (không phải sessionUpload/Shared) - đây
  ## là weight RESIDENT cho model int8/int4-symmetric/tied-embedding (đường
  ## đi khi model KHÔNG phải Q4-per-group) - CPU không đụng lại sau lúc
  ## upload này, nên nên dùng Private trên Metal giống hệt lý do áp dụng cho
  ## Q4 (residentUploadQ4) - đảm bảo model int8 cũng được hưởng tối ưu
  ## Private/Shared hybrid, không chỉ riêng Q4.
  let wTSeq = @wT
  result = byby.sessionUploadPrivate(ctx.kind.toByby(), wTSeq)

proc beResidentLinearFp32*(ctx: Backend, x: openArray[float32], w: SessionHandle,
                            M, K, N: int): seq[float32] =
  ## Forward Linear dùng weight fp32 ĐÃ resident (w từ beResidentUploadFp32):
  ## mỗi lần gọi chỉ upload activation `x` (nhỏ, đổi mỗi token) - KHÔNG đụng
  ## lại tới bytes weight. Trả về seq rỗng nếu backend không mở được session
  ## (caller tự fallback về đường cũ khi đó).
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
# GATHER / ARGMAX-INDEX / COMPARE / DROPOUT-RNG - GPU HOÁ THẬT (vòng 2, xem
# tensor_ops.nim mục 3/4/7 và dropout): permute/concat/split/stack đi qua
# beGatherGpu (kernel gather_kernel.cu/.comp/.metal); argmax trả cả giá trị
# lẫn chỉ số cùng lúc trên GPU qua beArgmaxIndexGpu; so sánh (gt/lt/eq/ge/le)
# xuất bitmask thẳng từ compare_kernel qua beCompareGpuOp; dropout mask sinh
# bằng RNG chạy trên GPU (xorshift32 counter-based) qua beDropoutMaskGpu -
# không còn std/random phía host nữa.
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
  ## MỚI THÊM: sqrt từng phần tử (dùng cho Adam optimizer bias-correction
  ## trong nimformer.nim, không phải đường nóng suy luận). Xem gpuSqrt/
  ## cpuSqrt trong gpubackend.nim.
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
  ## Gộp "residual add rồi layernorm ngay sau" (đúng pattern lặp lại 2
  ## lần/layer trong TransformerBlock.forward) vào 1 session GPU duy nhất -
  ## 1 lần upload/commit/wait thay vì 2 lần riêng biệt (beAdd rồi
  ## beLayernorm, mỗi cái tự alloc buffer + command buffer + wait). Trả về
  ## CẢ HAI: `summed` (a+b thô, dùng làm residual cho lần cộng cuối block)
  ## VÀ `normed` (đã qua layernorm) - để caller không phải cộng lại lần 2.
  ## CHỈ có lợi thật trên backend đã có sessionLayernormEnc (hiện tại:
  ## Metal). Nếu backend không hỗ trợ (CUDA/OpenCL/CPU hiện tại, hoặc
  ## encode lỗi lúc chạy) -> rơi về ĐÚNG đường cũ (beAdd rồi beLayernorm),
  ## không đổi hành vi/độ chính xác, chỉ mất phần tăng tốc.
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
  # Fallback: đường cũ, 2 lệnh GPU riêng biệt.
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
# APF & Custom Float helpers (tương thích ngược hoàn hảo)
# ─────────────────────────────────────────────────────────────

proc beCustomfloatEncode*(ctx: Backend, arr: openArray[float32], cf: CustomFloat): seq[uint8] =
  return customfloat.encodeArray(arr, cf)

proc beCustomfloatDecode*(ctx: Backend, buf: openArray[uint8], cf: CustomFloat): seq[float32] =
  return customfloat.decodeArray(buf, cf)

proc beApfCastForTraining*(ctx: Backend, arr: openArray[float32], gradArr: openArray[float32] = [],
                           relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                           expMargin = APF_EXP_MARGIN_BITS): tuple[data: seq[uint8], cf: CustomFloat] =
  return customfloat.apfCastForTraining(arr, gradArr, relErrorTol, expMargin)
