## backend.nim
## Lớp chọn backend tính toán cho Nimformer dựa hoàn toàn vào BybyLang trong vendor.
## Loại bỏ toàn bộ các kernel tự viết ngoài, thống nhất mọi backend (CPU, Metal, CUDA, OpenCL, TSIC)
## qua duy nhất BybyLang để tối ưu hoá hiệu năng, tránh trùng lặp mã nguồn và hỗ trợ hoàn hảo TSIC.

import vendor/bybylang/gpubackend as byby
import customfloat

# -d:backend=cpu|metal|cuda|opencl|tsic|auto — mặc định "auto" (tự dò lúc runtime).
const backend* {.strdefine.} = "auto"

type
  BackendKind* = enum
    bkCpu = "cpu"
    bkMetal = "metal"
    bkCuda = "cuda"
    bkOpenCL = "opencl"
    bkTsic = "tsic"

  Backend* = object
    kind*: BackendKind

proc toByby(bk: BackendKind): byby.GpuBackend =
  case bk
  of bkCpu: byby.gbCpu
  of bkMetal: byby.gbMetal
  of bkCuda: byby.gbCuda
  of bkOpenCL: byby.gbOpenCL
  of bkTsic: byby.gbTsic

proc resolveBackendKind(want: string): BackendKind =
  case want
  of "cpu": bkCpu   # yêu cầu CPU TƯỜNG MINH -> vẫn cho phép (demo/debug chủ động)
  of "metal": bkMetal
  of "cuda": bkCuda
  of "opencl": bkOpenCL
  of "tsic": bkTsic
  else:
    # "auto" -> Tự động dò backend qua BybyLang.
    # SỬA LỖI: trước đây nếu không dò thấy GPU nào (detectBackend trả về
    # gbCpu/gbAuto), hàm này ÂM THẦM trả về bkCpu và newBackend() cứ thế train
    # trên CPU - không hề đụng tới gForbidCpuFallback (cờ đó chỉ canh runtime-
    # failure của 1 backend GPU đã CHỌN TƯỜNG MINH, không canh bước "auto tìm
    # không ra GPU" này). Đây là lỗ hổng khiến "train luôn chạy CPU mà không
    # ai biết" dù comment đầu gpubackend.nim khẳng định đã ngăn chuyện đó.
    # Giờ: "auto" không tìm thấy GPU nào -> raise (trừ khi CHỦ ĐỘNG tắt forbid).
    let chosen = byby.detectBackend()
    case chosen
    of byby.gbCuda: bkCuda
    of byby.gbMetal: bkMetal
    of byby.gbOpenCL: bkOpenCL
    of byby.gbTsic: bkTsic
    of byby.gbCpu, byby.gbAuto:
      if byby.gForbidCpuFallback:
        raise newException(CatchableError,
          "resolveBackendKind(\"auto\"): KHÔNG tìm thấy GPU nào (CUDA/Metal/OpenCL/TSIC) " &
          "trên máy này -> từ chối tự rơi về CPU (gForbidCpuFallback=true). " &
          "Nếu CHỦ ĐỘNG muốn chạy demo/debug trên CPU: gọi setForbidCpuFallback(false) " &
          "trước newBackend(), hoặc dùng newBackend(\"cpu\") tường minh.")
      bkCpu

proc newBackend*(force: string = backend): Backend =
  ## Khởi tạo backend mới. Tự động liên kết và chọn thiết bị phù hợp nhất.
  let kind = resolveBackendKind(force)
  stderr.writeLine "== [Framework] Backend đã chọn: " & $kind & " =="
  result = Backend(kind: kind)

proc closeBackend*(b: Backend) =
  discard

proc setForbidCpuFallback*(forbid: bool) =
  ## Mặc định TRUE: nếu backend GPU đã chọn (cuda/metal/opencl/tsic) lỗi lúc
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

proc beSigmoid*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuSigmoid(ctx.kind.toByby(), @x)

proc beTanh*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuTanh(ctx.kind.toByby(), @x)

proc beSoftmax*(ctx: Backend, x: openArray[float32], rows, cols: int): seq[float32] =
  result = byby.gpuSoftmax(ctx.kind.toByby(), @x, rows, cols)

proc beLayernorm*(ctx: Backend, x: openArray[float32], gamma, beta: openArray[float32], rows, cols: int, eps: float32): seq[float32] =
  result = byby.gpuLayernorm(ctx.kind.toByby(), @x, @gamma, @beta, rows, cols, eps)

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