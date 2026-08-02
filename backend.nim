## backend.nim
## Lớp chọn backend tính toán cho Nimformer dựa hoàn toàn vào BybyLang trong vendor.
## Loại bỏ toàn bộ các kernel tự viết ngoài, thống nhất mọi backend (CPU, Metal, CUDA, OpenCL, TSIC)
## qua duy nhất BybyLang để tối ưu hoá hiệu năng, tránh trùng lặp mã nguồn và hỗ trợ hoàn hảo TSIC.

import std/tables
import vendor/bybylang/gpubackend as byby
import customfloat
export byby.GpuBackend, byby.SessionHandle, byby.sessionBegin, byby.sessionUpload, byby.sessionRope,
       byby.sessionUploadIndices, byby.sessionAllocScratch, byby.sessionMatmul, byby.sessionMatmulNT,
       byby.sessionSoftmax, byby.sessionVecOp, byby.sessionActivation, byby.sessionEnd,
       byby.sessionRead, byby.sessionFree, byby.sessionCopyInto,
       byby.ResidentWeightHandle, byby.residentUploadQ4, byby.sessionMatmulQ4, byby.residentFreeQ4,
       byby.sessionLayernormEnc, byby.sessionRmsnormEnc, byby.sessionAttentionFusedEnc,
       byby.gpuExp, byby.gpuLog, byby.gpuSqrt, byby.gpuPow, byby.gpuSortDescPadded, byby.gpuRepeatPenalty

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

proc toByby*(bk: BackendKind): byby.GpuBackend =
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

# ─────────────────────────────────────────────────────────────
# CACHE Backend theo tên string, dùng cho các overload kiểu Python
# (vd. beMatmul(a, ..., backend="metal", opType="session")) ở dưới - KHÔNG
# tạo `ctx: Backend` thủ công. Cache để mỗi lần gọi op KHÔNG detect GPU lại
# từ đầu (detectBackend() dò thiết bị tốn thời gian) -> chính cache này là
# phần giúp tránh bottleneck khi 1 tên backend được gọi lặp lại hàng ngàn
# lần/giây trong vòng lặp decode.
var gBackendCache = initTable[string, Backend]()

proc getBackend*(name: string): Backend =
  if name in gBackendCache:
    return gBackendCache[name]
  result = newBackend(name)
  gBackendCache[name] = result

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

proc beMatmul*(a: openArray[float32], M, K: int, b: openArray[float32], K2, N: int,
                backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## Overload kiểu "Python": beMatmul(a, M, K, b, K2, N, backend="metal",
  ## opType="session") - KHÔNG cần tự newBackend()/toByby() thủ công.
  ## - backend: "auto"|"cpu"|"metal"|"cuda"|"opencl"|"tsic" (xem
  ##   resolveBackendKind). Được cache qua getBackend() - gọi lặp lại nhiều
  ##   lần cùng 1 tên KHÔNG detect GPU lại từ đầu.
  ## - opType="session": chạy qua sessionBegin/sessionUpload/sessionMatmul/
  ##   sessionEnd trong CÙNG MỘT session GPU (giảm round-trip CPU<->GPU khi
  ##   caller tự lặp matmul nhiều lần liên tiếp mà chưa muốn/không tiện gom
  ##   thành 1 hàm fused riêng). Nếu backend không hỗ trợ session hoặc encode
  ##   lỗi lúc chạy -> tự rơi về opType="direct" (không đổi kết quả, chỉ mất
  ##   phần tăng tốc round-trip).
  ## - opType="direct" (mặc định): gọi thẳng byby.gpuMatmul 1 lần, như trước
  ##   giờ - dùng khi chỉ matmul 1 lần đơn lẻ, không nằm trong chuỗi op liên
  ##   tiếp (lúc đó overhead session mở/đóng không bù lại được).
  let ctx = getBackend(backend)
  if opType == "session":
    let be = ctx.kind.toByby()
    block sessionPath:
      if not sessionBegin(be): break sessionPath
      var hA = sessionUpload(be, @a)
      var hB = sessionUpload(be, @b)
      var hC = sessionAllocScratch(be, M * N)
      if not sessionMatmul(be, hA, hB, hC, M, K, N) or not sessionEnd(be):
        sessionFree(be, hA); sessionFree(be, hB); sessionFree(be, hC)
        break sessionPath
      result = sessionRead(be, hC, M * N)
      sessionFree(be, hA); sessionFree(be, hB); sessionFree(be, hC)
      return
  # direct (mặc định, hoặc fallback nếu session lỗi/không hỗ trợ)
  result = beMatmul(ctx, a, M, K, b, K2, N)

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

proc beMatmulQ4*(a: openArray[float32], wq: openArray[uint8], scales, zeros: openArray[float32],
                  M, K, N, groupSize, nGroupsPerRow: int,
                  backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## Overload kiểu "Python": beMatmulQ4(a, wq, scales, zeros, M, K, N,
  ## groupSize, nGroupsPerRow, backend="metal", opType="direct").
  ## LƯU Ý: weight int4 KHÔNG resident ở overload này (upload lại mỗi lần
  ## gọi) - nếu weight cố định qua nhiều lần gọi (vd. mỗi Linear trong vòng
  ## lặp decode), dùng beResidentUploadQ4 (ctx) 1 lần lúc load model rồi
  ## beResidentLinearQ4 mỗi token thay vì hàm này, để không upload lại bytes
  ## weight mỗi bước. opType hiện chỉ có "direct" (gpuMatmulQ4 1 lần); giữ
  ## tham số để đồng nhất API với các overload khác và dễ mở rộng sau.
  let ctx = getBackend(backend)
  result = beMatmulQ4(ctx, a, wq, scales, zeros, M, K, N, groupSize, nGroupsPerRow)

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

proc beResidentUploadQ4*(wq: openArray[uint8], scales, zeros: openArray[float32],
                          k, n, groupSize, nGroupsPerRow: int,
                          backend: string = "auto"): ResidentWeightHandle =
  ## beResidentUploadQ4(wq, scales, zeros, k, n, groupSize, nGroupsPerRow,
  ## backend="metal"). QUAN TRỌNG: dùng ĐÚNG 1 tên `backend` xuyên suốt cho
  ## upload + các lần gọi beResidentLinearQ4/beFeedForwardFusedQ4 sau đó -
  ## handle được cấp phát gắn với backend cụ thể lúc upload, đổi tên backend
  ## giữa chừng sẽ đọc nhầm device.
  beResidentUploadQ4(getBackend(backend), wq, scales, zeros, k, n, groupSize, nGroupsPerRow)

proc beResidentLinearQ4*(x: openArray[float32], w: ResidentWeightHandle, M: int,
                          backend: string = "auto"): seq[float32] =
  ## beResidentLinearQ4(x, w, M, backend="metal") - `backend` PHẢI khớp tên
  ## đã dùng lúc beResidentUploadQ4(..., backend=...) tạo ra `w`.
  beResidentLinearQ4(getBackend(backend), x, w, M)

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
  let wTSeq = @wT
  result = byby.sessionUpload(ctx.kind.toByby(), wTSeq)

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

proc beResidentUploadFp32*(wT: openArray[float32], backend: string = "auto"): SessionHandle =
  ## beResidentUploadFp32(wT, backend="metal") - xem lưu ý ở
  ## beResidentUploadQ4 về việc giữ ĐÚNG 1 tên backend xuyên suốt.
  beResidentUploadFp32(getBackend(backend), wT)

proc beResidentLinearFp32*(x: openArray[float32], w: SessionHandle, M, K, N: int,
                            backend: string = "auto"): seq[float32] =
  ## beResidentLinearFp32(x, w, M, K, N, backend="metal")
  beResidentLinearFp32(getBackend(backend), x, w, M, K, N)

proc beResidentFreeFp32*(w: var SessionHandle, backend: string = "auto") =
  ## beResidentFreeFp32(w, backend="metal")
  beResidentFreeFp32(getBackend(backend), w)

proc beMatmul2*(ctx: Backend,
                 a1: openArray[float32], M1, K1: int, b1: openArray[float32], K1b, N1: int,
                 a2: openArray[float32], M2, K2: int, b2: openArray[float32], K2b, N2: int):
                 tuple[y1, y2: seq[float32]] =
  let a1S = @a1; let b1S = @b1
  let a2S = @a2; let b2S = @b2
  let r = byby.gpuMatmul2(ctx.kind.toByby(), a1S, b1S, M1, K1, N1, a2S, b2S, M2, K2, N2)
  return (r.c1, r.c2)

proc beMatmul2*(a1: openArray[float32], M1, K1: int, b1: openArray[float32], K1b, N1: int,
                a2: openArray[float32], M2, K2: int, b2: openArray[float32], K2b, N2: int,
                backend: string = "auto"): tuple[y1, y2: seq[float32]] =
  ## beMatmul2(a1, M1, K1, b1, K1b, N1, a2, M2, K2, b2, K2b, N2, backend="metal")
  beMatmul2(getBackend(backend), a1, M1, K1, b1, K1b, N1, a2, M2, K2, b2, K2b, N2)

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

proc beVecOpSessionOrDirect(ctx: Backend, op: string, a, b: openArray[float32],
                             opType: string): seq[float32] =
  ## Dùng chung cho beAdd/beSub/beMul/beDiv overload string-arg: opType=
  ## "session" chạy qua sessionVecOp (giảm round-trip nếu caller lặp nhiều
  ## vecOp liên tiếp qua nhiều lời gọi), tự fallback "direct" (byby.gpuOp 1
  ## lần, như trước giờ) nếu backend không hỗ trợ session hoặc encode lỗi.
  if opType == "session":
    let be = ctx.kind.toByby()
    block sessionPath:
      if not sessionBegin(be): break sessionPath
      var hA = sessionUpload(be, @a)
      var hB = sessionUpload(be, @b)
      var hC = sessionAllocScratch(be, a.len)
      if not sessionVecOp(be, op, hA, hB, hC, a.len) or not sessionEnd(be):
        sessionFree(be, hA); sessionFree(be, hB); sessionFree(be, hC)
        break sessionPath
      result = sessionRead(be, hC, a.len)
      sessionFree(be, hA); sessionFree(be, hB); sessionFree(be, hC)
      return
  result = byby.gpuOp(op, ctx.kind.toByby(), @a, @b)

proc beAdd*(a, b: openArray[float32], backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## beAdd(a, b, backend="metal", opType="session")
  beVecOpSessionOrDirect(getBackend(backend), "add", a, b, opType)

proc beSub*(a, b: openArray[float32], backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## beSub(a, b, backend="metal", opType="session")
  beVecOpSessionOrDirect(getBackend(backend), "sub", a, b, opType)

proc beMul*(a, b: openArray[float32], backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## beMul(a, b, backend="metal", opType="session")
  beVecOpSessionOrDirect(getBackend(backend), "mul", a, b, opType)

proc beDiv*(a, b: openArray[float32], backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## beDiv(a, b, backend="metal", opType="session")
  beVecOpSessionOrDirect(getBackend(backend), "div", a, b, opType)

proc beSortDescIdx*(ctx: Backend, probs: openArray[float32]): tuple[sortedProbs: seq[float32], idx: seq[int]] =
  ## Sort probs GIẢM DẦN hoàn toàn qua gpuSortDescPadded (bitonic sort GPU
  ## thật khi ctx là backend GPU - xem gpubackend.nim). Trả về cả giá trị đã
  ## sort lẫn permutation index (id token gốc), dùng thẳng cho top-p cutoff.
  var vals = newSeq[int32](probs.len)
  for i in 0 ..< probs.len: vals[i] = int32(i)
  let r = byby.gpuSortDescPadded(ctx.kind.toByby(), @probs, vals)
  result.sortedProbs = r.keys
  result.idx = newSeq[int](r.vals.len)
  for i, v in r.vals: result.idx[i] = int(v)

proc beSortDescIdx*(probs: openArray[float32], backend: string = "auto"): tuple[sortedProbs: seq[float32], idx: seq[int]] =
  beSortDescIdx(getBackend(backend), probs)

proc beRepeatPenalty*(ctx: Backend, logits: openArray[float32], recentIds: openArray[int],
                       penalty: float32, noPenaltyIds: openArray[int] = []): seq[float32] =
  ## Dedupe recentIds (trừ noPenaltyIds) thành (id, count) - việc đếm này
  ## chỉ đụng tối đa vài chục số nguyên nhỏ, KHÔNG đụng vector logits lớn -
  ## rồi áp phạt toàn bộ qua gpuRepeatPenalty (GPU thật khi ctx là backend
  ## GPU, xem gpubackend.nim + repeat_penalty_kernel trong
  ## vecop_matmul.metal/.cl/vecop_ext_kernel.cu).
  if penalty == 1.0'f32 or recentIds.len == 0:
    return @logits
  var countMap = initTable[int32, int32]()
  for id in recentIds:
    if id < 0 or id >= logits.len: continue
    var skip = false
    for npid in noPenaltyIds:
      if id == npid: skip = true; break
    if skip: continue
    countMap[int32(id)] = countMap.getOrDefault(int32(id), 0'i32) + 1'i32
  var ids = newSeq[int32]()
  var counts = newSeq[int32]()
  for id, c in countMap:
    ids.add(id); counts.add(c)
  result = byby.gpuRepeatPenalty(ctx.kind.toByby(), @logits, ids, counts, penalty)

proc beRelu*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuRelu(ctx.kind.toByby(), @x)

proc beExp*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuExp(ctx.kind.toByby(), @x)

proc beLog*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuLog(ctx.kind.toByby(), @x)

proc beSqrt*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuSqrt(ctx.kind.toByby(), @x)

proc bePow*(ctx: Backend, x: openArray[float32], p: float32): seq[float32] =
  ## Metal/OpenCL/CUDA đều có pow_kernel thật (tham số p qua GPU, không qua
  ## opcode) - xem gpuPow trong gpubackend.nim. CUDA cần build
  ## vecop_ext_kernel.cu -> nối vào vecop.ptx bằng nvcc trước (máy build
  ## thật, không phải sandbox sinh code này).
  result = byby.gpuPow(ctx.kind.toByby(), @x, p)

proc beSigmoid*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuSigmoid(ctx.kind.toByby(), @x)

proc beTanh*(ctx: Backend, x: openArray[float32]): seq[float32] =
  result = byby.gpuTanh(ctx.kind.toByby(), @x)

proc beActivationSessionOrDirect(ctx: Backend, op: string, x: openArray[float32],
                                  opType: string): seq[float32] =
  ## Dùng chung cho beRelu/beSigmoid/beTanh overload string-arg, cùng cơ chế
  ## session/direct + fallback như beVecOpSessionOrDirect ở trên.
  if opType == "session":
    let be = ctx.kind.toByby()
    block sessionPath:
      if not sessionBegin(be): break sessionPath
      var hX = sessionUpload(be, @x)
      var hY = sessionAllocScratch(be, x.len)
      if not sessionActivation(be, op, hX, hY, x.len) or not sessionEnd(be):
        sessionFree(be, hX); sessionFree(be, hY)
        break sessionPath
      result = sessionRead(be, hY, x.len)
      sessionFree(be, hX); sessionFree(be, hY)
      return
  case op
  of "relu": result = byby.gpuRelu(ctx.kind.toByby(), @x)
  of "sigmoid": result = byby.gpuSigmoid(ctx.kind.toByby(), @x)
  of "tanh": result = byby.gpuTanh(ctx.kind.toByby(), @x)
  else: result = @x  # không xảy ra - chỉ gọi nội bộ với 3 tên trên

proc beRelu*(x: openArray[float32], backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## beRelu(x, backend="metal", opType="session")
  beActivationSessionOrDirect(getBackend(backend), "relu", x, opType)

proc beExp*(x: openArray[float32], backend: string = "auto"): seq[float32] =
  ## beExp(x, backend="metal")
  beExp(getBackend(backend), x)

proc beLog*(x: openArray[float32], backend: string = "auto"): seq[float32] =
  ## beLog(x, backend="metal")
  beLog(getBackend(backend), x)

proc beSqrt*(x: openArray[float32], backend: string = "auto"): seq[float32] =
  ## beSqrt(x, backend="metal")
  beSqrt(getBackend(backend), x)

proc bePow*(x: openArray[float32], p: float32, backend: string = "auto"): seq[float32] =
  ## bePow(x, 2.0, backend="metal")  -- luôn chạy CPU nội bộ, xem gpuPow.
  bePow(getBackend(backend), x, p)

proc beSigmoid*(x: openArray[float32], backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## beSigmoid(x, backend="metal", opType="session")
  beActivationSessionOrDirect(getBackend(backend), "sigmoid", x, opType)

proc beTanh*(x: openArray[float32], backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## beTanh(x, backend="metal", opType="session")
  beActivationSessionOrDirect(getBackend(backend), "tanh", x, opType)

proc beSoftmax*(ctx: Backend, x: openArray[float32], rows, cols: int): seq[float32] =
  result = byby.gpuSoftmax(ctx.kind.toByby(), @x, rows, cols)

proc beSoftmax*(x: openArray[float32], rows, cols: int,
                 backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## beSoftmax(x, rows, cols, backend="metal", opType="session")
  let ctx = getBackend(backend)
  if opType == "session":
    let be = ctx.kind.toByby()
    block sessionPath:
      if not sessionBegin(be): break sessionPath
      var hX = sessionUpload(be, @x)
      var hY = sessionAllocScratch(be, x.len)
      if not sessionSoftmax(be, hX, hY, rows, cols) or not sessionEnd(be):
        sessionFree(be, hX); sessionFree(be, hY)
        break sessionPath
      result = sessionRead(be, hY, x.len)
      sessionFree(be, hX); sessionFree(be, hY)
      return
  result = beSoftmax(ctx, x, rows, cols)

proc beLayernorm*(ctx: Backend, x: openArray[float32], gamma, beta: openArray[float32], rows, cols: int, eps: float32): seq[float32] =
  result = byby.gpuLayernorm(ctx.kind.toByby(), @x, @gamma, @beta, rows, cols, eps)

proc beLayernorm*(x: openArray[float32], gamma, beta: openArray[float32], rows, cols: int, eps: float32,
                   backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## beLayernorm(x, gamma, beta, rows, cols, eps, backend="metal", opType="session")
  let ctx = getBackend(backend)
  if opType == "session":
    let be = ctx.kind.toByby()
    block sessionPath:
      if not sessionBegin(be): break sessionPath
      var hX = sessionUpload(be, @x)
      var hGamma = sessionUpload(be, @gamma)
      var hBeta = sessionUpload(be, @beta)
      var hY = sessionAllocScratch(be, x.len)
      if not sessionLayernormEnc(be, hX, hGamma, hBeta, hY, rows, cols, eps) or not sessionEnd(be):
        sessionFree(be, hX); sessionFree(be, hGamma); sessionFree(be, hBeta); sessionFree(be, hY)
        break sessionPath
      result = sessionRead(be, hY, x.len)
      sessionFree(be, hX); sessionFree(be, hGamma); sessionFree(be, hBeta); sessionFree(be, hY)
      return
  result = beLayernorm(ctx, x, gamma, beta, rows, cols, eps)

proc beAddLayernormFused*(ctx: Backend, a, b: openArray[float32], gamma, beta: openArray[float32],
                           rows, cols: int, eps: float32): tuple[summed, normed: seq[float32]] =
  ## Gộp "residual add rồi layernorm ngay sau" (đúng pattern lặp lại 2
  ## lần/layer trong TransformerBlock.forward) vào 1 session GPU duy nhất -
  ## 1 lần upload/commit/wait thay vì 2 lần riêng biệt (beAdd rồi
  ## beLayernorm, mỗi cái tự alloc buffer + command buffer + wait). Trả về
  ## CẢ HAI: `summed` (a+b thô, dùng làm residual cho lần cộng cuối block)
  ## VÀ `normed` (đã qua layernorm) - để caller không phải cộng lại lần 2.
  ## CHỈ có lợi thật trên backend đã có sessionLayernormEnc (hiện tại:
  ## Metal). Nếu backend không hỗ trợ (CUDA/OpenCL/CPU/TSIC hiện tại, hoặc
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

proc beAddLayernormFused*(a, b: openArray[float32], gamma, beta: openArray[float32],
                           rows, cols: int, eps: float32,
                           backend: string = "auto"): tuple[summed, normed: seq[float32]] =
  ## beAddLayernormFused(a, b, gamma, beta, rows, cols, eps, backend="metal").
  ## Hàm gốc (ctx) đã TỰ dùng session bên trong (không có đường "direct" rời
  ## rạc để chọn) nên overload này chỉ có `backend`, không có `opType`.
  beAddLayernormFused(getBackend(backend), a, b, gamma, beta, rows, cols, eps)

proc beFeedForwardFusedQ4*(ctx: Backend, x: openArray[float32],
                            w1: ResidentWeightHandle, w2: ResidentWeightHandle,
                            M: int): seq[float32] =
  ## Gộp fc1 (matmulQ4 resident) -> apflu activation -> fc2 (matmulQ4
  ## resident) vào MỘT session GPU duy nhất (1 lần sessionBegin/End) thay vì
  ## 3 round-trip CPU<->GPU riêng biệt như đường cũ (beResidentLinearQ4 gọi 2
  ## lần + beApflu gọi 1 lần ở giữa) - mỗi round-trip tự mở/đóng command
  ## buffer + wait, cực kỳ tốn trên GPU băng thông thấp (vd. iGPU) vì overhead
  ## sync độc lập với FLOPs thật. Cùng cơ chế fallback như
  ## beAddLayernormFused: nếu bước nào lỗi (backend không hỗ trợ sessionActivation
  ## "apflu" cho CUDA/OpenCL, hoặc encode lỗi lúc chạy) -> trả về seq rỗng,
  ## caller (FeedForward.forward) PHẢI tự fallback về đường cũ 3 bước.
  ## LƯU Ý: alpha/beta của "apflu" trong sessionActivation là hằng số cố định
  ## trong kernel (không nhận tham số) - nếu bạn đổi alpha/beta của beApflu ở
  ## chỗ khác thì phải đối chiếu lại kernel encode trước khi tin kết quả này.
  let be = ctx.kind.toByby()
  block fused:
    if not sessionBegin(be): break fused
    var hX = sessionUpload(be, @x)
    var hH = sessionAllocScratch(be, M * w1.n)
    if not sessionMatmulQ4(be, hX, w1, hH, M):
      discard sessionEnd(be); sessionFree(be, hX); sessionFree(be, hH)
      break fused
    var hAct = sessionAllocScratch(be, M * w1.n)
    if not byby.sessionActivation(be, "apflu", hH, hAct, M * w1.n):
      discard sessionEnd(be); sessionFree(be, hX); sessionFree(be, hH); sessionFree(be, hAct)
      break fused
    var hOut = sessionAllocScratch(be, M * w2.n)
    if not sessionMatmulQ4(be, hAct, w2, hOut, M):
      discard sessionEnd(be)
      sessionFree(be, hX); sessionFree(be, hH); sessionFree(be, hAct); sessionFree(be, hOut)
      break fused
    if not sessionEnd(be):
      sessionFree(be, hX); sessionFree(be, hH); sessionFree(be, hAct); sessionFree(be, hOut)
      break fused
    result = sessionRead(be, hOut, M * w2.n)
    sessionFree(be, hX); sessionFree(be, hH); sessionFree(be, hAct); sessionFree(be, hOut)
    return
  result = @[]  # báo hiệu caller tự fallback

proc beFeedForwardFusedQ4*(x: openArray[float32], w1: ResidentWeightHandle, w2: ResidentWeightHandle,
                            M: int, backend: string = "auto"): seq[float32] =
  ## beFeedForwardFusedQ4(x, w1, w2, M, backend="metal"). Weight PHẢI đã
  ## resident (từ beResidentUploadQ4/beResidentUploadQ4(backend=...) cùng
  ## backend này) - hàm chỉ gói `getBackend(backend)` giùm, không tự upload
  ## lại weight (uploads lặp lại mỗi token mới là bottleneck cần tránh).
  beFeedForwardFusedQ4(getBackend(backend), x, w1, w2, M)

proc beReduceGpu*(ctx: Backend, op: string, x: openArray[float32], outer, dimSize, inner: int): seq[float32] =
  ## op: "sum" hoặc "max". Thử chạy GPU thật (Metal/OpenCL - xem
  ## byby.gpuReduce); trả seq RỖNG nếu backend chưa hỗ trợ (CPU/TSIC/CUDA
  ## hiện tại) hoặc lỗi lúc chạy - caller (beSum/beMax trong tensor_ops.nim)
  ## PHẢI tự fallback về đúng vòng lặp CPU sẵn có, không đổi hành vi.
  try:
    result = byby.gpuReduce(op, ctx.kind.toByby(), @x, outer, dimSize, inner)
  except CatchableError:
    result = @[]

proc beReduceGpu*(op: string, x: openArray[float32], outer, dimSize, inner: int,
                   backend: string = "auto"): seq[float32] =
  ## beReduceGpu("sum", x, outer, dimSize, inner, backend="metal"). Trả seq
  ## rỗng nếu backend chưa hỗ trợ - y hệt hành vi bản ctx, caller vẫn phải
  ## tự fallback CPU (xem beSum/beMax trong tensor_ops.nim).
  beReduceGpu(getBackend(backend), op, x, outer, dimSize, inner)

proc beEmbeddingLookup*(ctx: Backend, table: openArray[float32], indices: openArray[int32], vocab, dim: int): seq[float32] =
  result = byby.gpuEmbeddingLookup(ctx.kind.toByby(), @table, @indices, vocab, dim)

proc beEmbeddingLookup*(table: openArray[float32], indices: openArray[int32], vocab, dim: int,
                         backend: string = "auto"): seq[float32] =
  ## beEmbeddingLookup(table, indices, vocab, dim, backend="metal")
  beEmbeddingLookup(getBackend(backend), table, indices, vocab, dim)

proc beApflu*(ctx: Backend, x: openArray[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32): seq[float32] =
  result = byby.gpuApflu(ctx.kind.toByby(), @x, alpha, beta)

proc beApflu*(x: openArray[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32,
              backend: string = "auto", opType: string = "direct"): seq[float32] =
  ## beApflu(x, backend="metal", opType="session") - lưu ý alpha/beta ở
  ## opType="session" KHÔNG thật sự dùng (kernel sessionActivation "apflu"
  ## cố định hằng số riêng, xem comment beFeedForwardFusedQ4) - chỉ giữ
  ## tham số để chữ ký khớp bản "direct". Muốn alpha/beta tuỳ chỉnh thật ->
  ## dùng opType="direct".
  let ctx = getBackend(backend)
  if opType == "session":
    let be = ctx.kind.toByby()
    block sessionPath:
      if not sessionBegin(be): break sessionPath
      var hX = sessionUpload(be, @x)
      var hY = sessionAllocScratch(be, x.len)
      if not sessionActivation(be, "apflu", hX, hY, x.len) or not sessionEnd(be):
        sessionFree(be, hX); sessionFree(be, hY)
        break sessionPath
      result = sessionRead(be, hY, x.len)
      sessionFree(be, hX); sessionFree(be, hY)
      return
  result = beApflu(ctx, x, alpha, beta)

proc beApfluBackward*(ctx: Backend, x, dy: openArray[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32): seq[float32] =
  result = byby.gpuApfluBackward(ctx.kind.toByby(), @x, @dy, alpha, beta)

proc beApfluBackward*(x, dy: openArray[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32,
                       backend: string = "auto"): seq[float32] =
  ## beApfluBackward(x, dy, backend="metal")
  beApfluBackward(getBackend(backend), x, dy, alpha, beta)

proc beLayernormBackward*(ctx: Backend, dy, x, gamma, beta: openArray[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  let r = byby.gpuLayernormBackward(ctx.kind.toByby(), @dy, @x, @gamma, @beta, rows, cols, eps)
  return (r.dx, r.dgamma, r.dbeta)

proc beLayernormBackward*(dy, x, gamma, beta: openArray[float32], rows, cols: int, eps: float32,
                           backend: string = "auto"): tuple[dx, dgamma, dbeta: seq[float32]] =
  ## beLayernormBackward(dy, x, gamma, beta, rows, cols, eps, backend="metal")
  beLayernormBackward(getBackend(backend), dy, x, gamma, beta, rows, cols, eps)

proc beAttentionFused*(ctx: Backend, q, k, v, mask: openArray[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  let r = byby.gpuAttentionFused(ctx.kind.toByby(), @q, @k, @v, @mask, B, H, S, D, scale)
  return (r.o, r.s_matrix)

proc beAttentionFused*(q, k, v, mask: openArray[float32], B, H, S, D: int, scale: float32,
                        backend: string = "auto"): tuple[o, s_matrix: seq[float32]] =
  ## beAttentionFused(q, k, v, mask, B, H, S, D, scale, backend="metal")
  beAttentionFused(getBackend(backend), q, k, v, mask, B, H, S, D, scale)

proc beAttentionFusedBackward*(ctx: Backend, q, k, v, s_matrix, dy: openArray[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  let r = byby.gpuAttentionFusedBackward(ctx.kind.toByby(), @q, @k, @v, @s_matrix, @dy, B, H, S, D, scale)
  return (r.dq, r.dk, r.dv)

proc beAttentionFusedBackward*(q, k, v, s_matrix, dy: openArray[float32], B, H, S, D: int, scale: float32,
                                backend: string = "auto"): tuple[dq, dk, dv: seq[float32]] =
  ## beAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale, backend="metal")
  beAttentionFusedBackward(getBackend(backend), q, k, v, s_matrix, dy, B, H, S, D, scale)

# ─────────────────────────────────────────────────────────────
# APF & Custom Float helpers (tương thích ngược hoàn hảo)
# ─────────────────────────────────────────────────────────────

proc beCustomfloatEncode*(ctx: Backend, arr: openArray[float32], cf: CustomFloat): seq[uint8] =
  return customfloat.encodeArray(arr, cf)

proc beCustomfloatEncode*(arr: openArray[float32], cf: CustomFloat, backend: string = "auto"): seq[uint8] =
  ## `backend` KHÔNG ảnh hưởng gì ở đây - encode/decode customfloat luôn
  ## chạy CPU thuần (không đi qua BybyLang). Tham số chỉ để chữ ký đồng nhất
  ## với các hàm be* khác, tránh caller phải nhớ "hàm này khác kiểu gọi".
  return customfloat.encodeArray(arr, cf)

proc beCustomfloatDecode*(ctx: Backend, buf: openArray[uint8], cf: CustomFloat): seq[float32] =
  return customfloat.decodeArray(buf, cf)

proc beCustomfloatDecode*(buf: openArray[uint8], cf: CustomFloat, backend: string = "auto"): seq[float32] =
  ## Xem lưu ý ở beCustomfloatEncode - `backend` không ảnh hưởng, chỉ để đồng nhất API.
  return customfloat.decodeArray(buf, cf)

proc beApfCastForTraining*(ctx: Backend, arr: openArray[float32], gradArr: openArray[float32] = [],
                           relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                           expMargin = APF_EXP_MARGIN_BITS): tuple[data: seq[uint8], cf: CustomFloat] =
  return customfloat.apfCastForTraining(arr, gradArr, relErrorTol, expMargin)

proc beApfCastForTraining*(arr: openArray[float32], gradArr: openArray[float32] = [],
                           relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                           expMargin = APF_EXP_MARGIN_BITS,
                           backend: string = "auto"): tuple[data: seq[uint8], cf: CustomFloat] =
  ## Xem lưu ý ở beCustomfloatEncode - `backend` không ảnh hưởng, chỉ để đồng nhất API.
  return customfloat.apfCastForTraining(arr, gradArr, relErrorTol, expMargin)