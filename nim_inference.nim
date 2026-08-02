## generate.nim
## File DUY NHẤT: nimformer_llama (RoPE + RMSNorm + SwiGLU + GQA + KV-cache paged)
## + tokenizer + vòng lặp sinh văn bản, gộp chung 1 file theo yêu cầu.

import std/[math, random, sequtils, tables, algorithm, os, strformat, times, json, sets]
import nimformer  # Tensor, Linear, newLinear, flatten2D, transpose, Embedding, addT...
import backend
import tensor_ops  # beMax, beArgMax, beExp (GPU thật cho sampleTopP)
import customfloat
import quant       # loadQuantStateDict, dequantizeTensor, QuantTensor
import nimpy

# ─────────────────────────────────────────────────────────────
# DEBUG: BYBY_DEBUG_PATH=1 để biết CHÍNH XÁC layer đầu tiên đang chạy đường
# fused/resident (nhanh, 1 session/nhóm-op) hay rơi về đường LEGACY (mỗi
# Linear/RMSNorm tự mở session riêng, hoặc tệ hơn: dequant lại toàn bộ
# weight trên CPU mỗi token) ở từng điểm quyết định (QKV / attn+oProj /
# FFN). Đây LÀ điểm khác biệt duy nhất giữa "~1 tok/s" và "0.3 tok/s" bạn
# đang thấy - in ĐÚNG 1 LẦN/điểm (không spam mỗi token/layer) để không ảnh
# hưởng tốc độ đo được. Bật: BYBY_DEBUG_PATH=1 ./nim_inference ...
# Nếu bạn thấy BẤT KỲ dòng nào in "LEGACY fallback" cho layer 0 -> đó chính
# là nguyên nhân chậm: residentReady/residentFp32Ready của Linear tương ứng
# đang false (xem uploadResident() trong nimformer.nim) hoặc sessionBegin/
# sessionMatmulQ4/sessionMatmul đang fail ngay từ lệnh gọi đầu tiên.
# ─────────────────────────────────────────────────────────────
var gDebugPathOn = existsEnv("BYBY_DEBUG_PATH")
# ─────────────────────────────────────────────────────────────
# KV-CACHE GPU-RESIDENT: TRƯỚC ĐÂY forwardDecode copy TOÀN BỘ lịch sử K/V
# (S token đã sinh) từ CPU cache RỒI upload lại lên GPU MỖI TOKEN, MỖI
# LAYER, MỖI KV-HEAD - chi phí O(S) mỗi bước, O(S^2) toàn phiên sinh, PHÌNH
# DẦN theo độ dài đã sinh (độc lập hoàn toàn với buffer-pool hay
# whole-session - đây là lượng DỮ LIỆU thật phải di chuyển CPU->GPU, không
# phải overhead tạo object mà buffer-pool giải quyết). llama.cpp không bao
# giờ làm vậy - KV cache của nó Ở LẠI GPU vĩnh viễn suốt phiên sinh, mỗi
# token chỉ ghi thêm ĐÚNG 1 dòng mới.
# FIX: mỗi (layer, kv-head) có 1 buffer GPU cố định [maxSeqLen*headDim],
# cấp phát 1 LẦN (ensurePersistentKV, gọi NGOÀI mọi session - giống hệt
# cách uploadResident() cấp resident weight 1 lần trong nimformer.nim).
# Mỗi token chỉ upload ĐÚNG 1 dòng mới (headDim phần tử, KHÔNG phải
# S*headDim) rồi sessionCopyInto() ghi thẳng vào đúng vị trí
# (offset = cache.seqLen*headDim) trong buffer GPU-resident đó - dữ liệu cũ
# (S-1 token trước) không bao giờ bị đụng lại. Attention matmul dùng THẲNG
# buffer này (offset 0, đọc S*D phần tử đầu, phần còn lại trong buffer chưa
# dùng tới không ảnh hưởng gì vì matmul chỉ nhận đúng N=S) làm hK/hV.
var gPersistK = initTable[int, seq[SessionHandle]]()
var gPersistV = initTable[int, seq[SessionHandle]]()

proc ensurePersistentKV(be: GpuBackend, layerIdx, HKV, maxSeqLen, D: int) =
  if layerIdx in gPersistK: return
  var ks = newSeq[SessionHandle](HKV)
  var vs = newSeq[SessionHandle](HKV)
  for h in 0 ..< HKV:
    ks[h] = sessionAllocScratch(be, maxSeqLen * D)
    vs[h] = sessionAllocScratch(be, maxSeqLen * D)
  gPersistK[layerIdx] = ks
  gPersistV[layerIdx] = vs
var gDebugPathSeen = initHashSet[string]()
proc debugPathOnce(tag: string, usedFast: bool, extra: string = "") =
  if not gDebugPathOn: return
  if tag in gDebugPathSeen: return
  gDebugPathSeen.incl(tag)
  let verdict = if usedFast: "FUSED/RESIDENT (nhanh)"
                else: "LEGACY fallback (CHAM - session rieng le hoac dequant CPU moi token)"
  stderr.writeLine "[BYBY_DEBUG_PATH] " & tag & " -> " & verdict &
    (if extra.len > 0: "  (" & extra & ")" else: "")

# ===================================================================
# Helper GHÉP SESSION cross-submodule: trước đây MỖI submodule (RMSNorm,
# RoPE, FFN, attention-decode) tự mở/đóng session RIÊNG của nó
# (sessionBegin...sessionEnd), nghĩa là 1 layer = 4-6 lần commit+wait GPU
# (Metal: commandBuffer commit+waitUntilCompleted; CUDA: cuStreamSynchronize)
# dù giữa các submodule đó KHÔNG có phụ thuộc CPU thực sự bắt buộc nào
# (residual-add, RMSNorm-scale, các matmul gate/up/down đều thuần GPU nối
# tiếp nhau). Các proc dưới đây cho phép ENCODE nhiều lệnh vào 1 session ĐANG
# MỞ mà không cần đọc kết quả về CPU giữa chừng (kết quả ở lại dạng
# SessionHandle, chỉ sessionRead() đúng 1 lần ở cuối) - đây là cách "1 session
# cross-submodule" được thực hiện thật, không phải chỉ đổi tên hàm.
proc sMatmul(be: GpuBackend, a: SessionHandle, M, K: int, b: SessionHandle, N: int): SessionHandle =
  result = sessionAllocScratch(be, M * N)
  doAssert sessionMatmul(be, a, b, result, M, K, N),
    "sMatmul: sessionMatmul thất bại giữa session gộp (backend hiện tại có thể không hỗ trợ)"

proc sMatmulNT(be: GpuBackend, a: SessionHandle, M, K: int, b: SessionHandle, N: int): tuple[h: SessionHandle, ok: bool] =
  ## C(M x N) = A(M x K) @ B(N x K)^T - B truyền vào GIỮ NGUYÊN layout gốc
  ## [N,K] (vd upload thẳng slice K-cache/V-cache [S,D], không transpose CPU
  ## trước). THÊM MỚI: dùng cho attention decode để bỏ hẳn chi phí transpose
  ## O(S*D)/token của đường cũ.
  ##
  ## Trả về ok=false nếu backend hiện tại chưa hỗ trợ sessionMatmulNT (chỉ
  ## OpenCL/Metal/CUDA có, CPU thuần/Tsic chưa) - caller PHẢI tự fallback về
  ## sMatmul với b đã transpose sẵn (đường cũ), KHÔNG được coi false là lỗi
  ## rồi doAssert crash.
  result.h = sessionAllocScratch(be, M * N)
  result.ok = sessionMatmulNT(be, a, b, result.h, M, K, N)

proc sVecOp(be: GpuBackend, op: string, a, b: SessionHandle, n: int): SessionHandle =
  result = sessionAllocScratch(be, n)
  doAssert sessionVecOp(be, op, a, b, result, n),
    "sVecOp(" & op & "): sessionVecOp thất bại giữa session gộp"

proc sActivation(be: GpuBackend, op: string, a: SessionHandle, n: int): SessionHandle =
  result = sessionAllocScratch(be, n)
  doAssert sessionActivation(be, op, a, result, n),
    "sActivation(" & op & "): sessionActivation thất bại giữa session gộp"

proc sSoftmax(be: GpuBackend, a: SessionHandle, rows, cols: int): SessionHandle =
  ## Bọc sessionSoftmax (đã có sẵn trong backend.nim, export từ byby) theo
  ## đúng pattern sMatmul/sVecOp/sActivation ở trên - dùng cho attention
  ## decode (xem forwardDecode bên dưới). Không có gì Metal-riêng, chạy
  ## được trên cả CUDA/OpenCL vì đi qua đúng API sessionXxx chung.
  result = sessionAllocScratch(be, rows * cols)
  doAssert sessionSoftmax(be, a, result, rows, cols),
    "sSoftmax: sessionSoftmax thất bại giữa session gộp"

proc broadcastBias(bias: Tensor, rows, outF: int): seq[float32] =
  ## Trải bias [outF] ra full [rows*outF] để cộng bằng 1 sessionVecOp("add")
  ## thay vì vòng lặp CPU "i mod outF" sau khi đọc kết quả về (điều đó sẽ ép
  ## phải kết thúc session sớm). Trả về seq rỗng nếu không có bias thật.
  if bias.data.len != outF: return @[]
  result = newSeq[float32](rows * outF)
  for r in 0 ..< rows:
    copyMem(addr result[r * outF], unsafeAddr bias.data[0], outF * sizeof(float32))

proc dequantTransposedW(l: Linear): seq[float32] =
  ## Weight ĐÃ transpose [inF, outF] flat row-major, sẵn sàng nạp thẳng vào
  ## session bằng sessionUpload() + sMatmul(), dùng chung cho QKV-fuse và
  ## FFN-fuse bên dưới (trước đây mỗi nơi tự viết lại 1 bản).
  ## CHỈ còn dùng làm FALLBACK khi weight KHÔNG phải int4-per-group (xem
  ## linearFast bên dưới) - đường chính (model GPTQ int4 thật) không đi qua
  ## hàm này nữa.
  if l.useQuant: dequantizeTensorTransposed(l.weightQ)
  else: transpose(l.weight).data

proc linearFast(l: Linear, xData: seq[float32], rows: int, ctx: Backend): seq[float32] =
  ## SỬA (NGUYÊN NHÂN CHÍNH của "1.5 phút/token"): TRƯỚC ĐÂY mọi nơi trong
  ## file này (fusedNormQKVRoPE, fusedResidualNormFFN, forwardQKVFused) đều
  ## gọi dequantTransposedW() -> giải nén TOÀN BỘ ma trận trọng số int4 ra
  ## fp32 trên CPU (vd 4096x4096 hoặc 4096x11008 mỗi ma trận) RỒI upload
  ## nguyên mảng fp32 đó lên GPU qua sessionUpload - lặp lại việc này cho
  ## CẢ 7 ma trận (Q,K,V,O,gate,up,down) x 32 layer x MỖI TOKEN sinh ra.
  ## Đây chính là "1.5 phút/token": không phải GPU kernel chậm, mà là CPU
  ## dequant + băng thông upload fp32 khổng lồ lặp lại vô ích mỗi bước decode
  ## (weight KHÔNG đổi giữa các token, không có lý do gì phải giải nén lại).
  ##
  ## nimformer.nim's Linear.forward() đã có sẵn đường tắt đúng chuẩn
  ## llama.cpp: beMatmulQ4() chạy matmul TRỰC TIẾP trên dữ liệu int4 còn nén
  ## (Metal/OpenCL có kernel int4 gốc, đọc thẳng 4-bit -> băng thông bộ nhớ
  ## thấp hơn ~8 lần so với đọc fp32 đã giải nén). File này (nim_inference.nim)
  ## trước đây không hề gọi tới nó. Giờ mọi chỗ dùng chung hàm này thay vì
  ## dequantTransposedW + beMatmul/sMatmul.
  if l.residentReady:
    # ĐƯỜNG NHANH THẬT: weight đã resident trên GPU từ uploadAllResidentWeights
    # (gọi 1 lần lúc load model) - ở đây chỉ upload xData (activation, nhỏ,
    # đổi mỗi token) rồi matmul thẳng với handle weight có sẵn, không đụng
    # lại bytes weight. Đây chính là đường loại bỏ "upload 4.5GB mỗi token".
    result = beResidentLinearQ4(ctx, xData, l.residentW, rows)
  elif l.residentFp32Ready:
    # SỬA (BUG THẬT, NGUYÊN NHÂN "0.44 token/s"): trước đây hàm này chỉ kiểm
    # tra residentReady (nhánh int4-asymmetric-per-group) rồi rơi THẲNG xuống
    # dequant CPU toàn bộ ma trận + upload lại fp32 MỖI TOKEN cho MỌI weight
    # khác (int8, int4-symmetric, fp32/tied-embedding lm_head) - dù
    # nimformer.nim's Linear.forward() đã có sẵn tầng resident fp32 này từ
    # trước (uploadResident() upload 1 lần lúc load model). File này
    # (nim_inference.nim, dùng cho đường fused decode thật) đơn giản là quên
    # gọi tầng đó. Giờ dùng chung residentFp32 + beResidentLinearFp32 giống
    # hệt nimformer.nim, chỉ upload activation nhỏ mỗi token.
    result = beResidentLinearFp32(ctx, xData, l.residentFp32, rows, l.inF, l.outF)

  if result.len == 0:
    # Chưa resident, hoặc backend hiện tại (vd cpu/tsic) không hỗ trợ đường
    # resident -> rơi về đường cũ (vẫn đúng, chỉ mất phần tăng tốc resident).
    if l.useQuant and l.weightQ.kind == qkInt4Asymmetric and l.weightQ.groupSize > 0:
      let nGroupsPerRow = (l.inF + l.weightQ.groupSize - 1) div l.weightQ.groupSize
      result = beMatmulQ4(ctx, xData, l.weightQ.data, l.weightQ.scale, l.weightQ.zero_point,
                           rows, l.inF, l.outF, l.weightQ.groupSize, nGroupsPerRow)
    else:
      # Fallback (weight không phải int4-per-group, vd lm_head tied-embedding
      # useQuant=false): vẫn phải dequant, nhưng đây là trường hợp hiếm
      # (không nằm trong vòng lặp 32 layer x mỗi token).
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

proc newRMSNormL*(dim: int, eps: float32 = 1e-6'f32): RMSNormL =
  result.dim = dim
  result.eps = eps
  result.weight = newTensor(@[dim], 1'f32)

proc rmsNormPrep*(rn: RMSNormL, x: Tensor, ctx: Backend): tuple[invBroadcast, wBroadcast: seq[float32], n: int] =
  ## Tách riêng phần "reduce" của RMSNorm (bình phương + sum theo hàng qua
  ## matmul-as-reduce, rồi 1/sqrt(mean+eps) trên CPU) khỏi phần "scale" (2
  ## phép mul kích thước đầy đủ). Lý do tách: invStd là số vô hướng/hàng,
  ## PHẢI đọc sumSq về CPU để chạy sqrt (session API hiện tại không có
  ## kernel reduce/sqrt) - đây là 1 điểm CPU-touch không thể tránh, xảy ra
  ## TRƯỚC bất kỳ session lớn nào. Nhưng phần "scale" (2 mul) thì KHÔNG có lý
  ## do gì phải tự đóng trong session riêng của nó - callers (fusedNormQKVRoPE,
  ## fusedResidualNormFFN) gộp thẳng 2 mul này vào session của submodule kế
  ## tiếp (QKV matmul / FFN) thay vì mở-đóng session chỉ vì 2 phép mul.
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
  ## SỬA: backend.nim không có primitive RMSNorm riêng, nhưng có ĐỦ add/mul
  ## (beMul/beAdd, hoặc sessionVecOp) và matmul (beMatmul) để dựng lại toàn
  ## bộ công thức trên GPU, không cần viết vòng lặp CPU thuần nữa:
  ##   - Bình phương từng phần tử: x ⊙ x -> 1 beMul (GPU).
  ##   - Tổng bình phương theo hàng (reduction mà backend không có kernel
  ##     riêng): nhân với vector cột toàn số 1 bằng beMatmul(x2, rows, cols,
  ##     ones, cols, 1) -> ra đúng sumSq [rows,1]. Đây là "matmul-as-reduce",
  ##     GPU-native, không phải giả vờ.
  ##   - 1/sqrt(mean+eps): CHỈ scalar/hàng (rows phần tử, không phải
  ##     rows*cols) nên tính trên CPU - không đáng và không thể thành 1
  ##     kernel GPU riêng chỉ vì rows con số.
  ##   - Scale lại theo invStd rồi theo weight: 2 phép mul kích thước đầy đủ
  ##     [rows,cols] -> gộp vào 1 session GPU (sessionVecOp mul, mul) để dữ
  ##     liệu ở lại GPU xuyên suốt, không upload/download giữa 2 bước.
  let (rows, cols) = flatten2D(x.shape)
  doAssert cols == rn.dim
  let n = rows * cols

  let x2 = beMul(ctx, x.data, x.data)                 # GPU: bình phương từng phần tử
  var ones = newSeq[float32](cols)
  for i in 0 ..< cols: ones[i] = 1'f32
  let sumSq = beMatmul(ctx, x2, rows, cols, ones, cols, 1)  # GPU: reduce theo hàng qua matmul

  var invStd = newSeq[float32](rows)                  # O(rows) scalar, CPU là hợp lý
  for r in 0 ..< rows:
    invStd[r] = 1'f32 / sqrt(sumSq[r] / float32(cols) + rn.eps)

  # Broadcast invStd/weight ra full [rows,cols] - đây là COPY dữ liệu (không
  # phải phép toán số học), backend không có kernel broadcast nên làm ở CPU,
  # rồi đẩy phần NHÂN thật sự (mul) lên GPU ngay sau.
  var invBroadcast = newSeq[float32](n)
  var wBroadcast = newSeq[float32](n)
  for r in 0 ..< rows:
    let off = r * cols
    for c in 0 ..< cols:
      invBroadcast[off + c] = invStd[r]
      wBroadcast[off + c] = rn.weight.data[c]

  let backend = ctx.kind.toByby()
  result.shape = x.shape
  # Session (mul, mul) gộp 2 phép nhân vào 1 encoder, dữ liệu ở lại GPU
  # xuyên suốt. Trên Metal, việc nhiều dispatch khác pipeline chia sẻ 1
  # encoder từng cho kết quả sai trên GPU không phải Apple Silicon (thiếu
  # memory barrier giữa các dispatch) - đã fix tận gốc trong metal_shim.m
  # (memoryBarrierWithScope: sau mỗi *_enc), nên giờ session an toàn trên
  # cả 3 backend, không cần né riêng Metal nữa.
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

  # Fallback: vẫn 100% GPU, chỉ là 2 lệnh beMul riêng (không cùng 1 session)
  # nếu session thất bại giữa chừng trên backend hiện tại.
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
  ## [T,H,D] (vị trí ngoài, head trong) -> [H,T,D] (head ngoài, vị trí trong).
  ## beAttentionFused (Metal/CUDA/OpenCL) giả định layout [B*H, S, D] — mỗi
  ## head là 1 khối liền S*D phần tử (base_idx = bh*S*D) — trong khi
  ## Linear.forward xuất ra layout [T,H,D] (vị trí ngoài). Không transpose
  ## trước khi gọi kernel -> kernel đọc lẫn head/token -> attention sai hoàn
  ## toàn, sinh token rác ngay từ prefill dù RoPE/Linear/dequant đều đúng.
  result = newSeq[float32](T * H * D)
  for t in 0 ..< T:
    for h in 0 ..< H:
      let src = (t * H + h) * D
      let dst = (h * T + t) * D
      for d in 0 ..< D:
        result[dst + d] = data[src + d]

proc toPosMajor(data: seq[float32], T, H, D: int): seq[float32] =
  ## Ngược lại: [H,T,D] (output của beAttentionFused) -> [T,H,D] (layout mà
  ## phần còn lại của code — attnOut.data rồi oProj.forward — đang mong đợi).
  result = newSeq[float32](T * H * D)
  for h in 0 ..< H:
    for t in 0 ..< T:
      let src = (h * T + t) * D
      let dst = (t * H + h) * D
      for d in 0 ..< D:
        result[dst + d] = data[src + d]

proc newRoPEL*(headDim, maxSeqLen: int, theta: float32 = 10000'f32,
                scalingType: string = "", scalingFactor: float32 = 1.0'f32): RoPEL =
  ## THÊM: hỗ trợ rope_scaling (đọc từ HF AutoConfig.rope_scaling, trước đây
  ## bị bỏ hoàn toàn -> model dùng scaling (vd deepseek-coder-6.7B-GPTQ:
  ## {'type':'linear','factor':4.0}) bị tính cos/sin SAI ngay từ layer đầu
  ## tiên dùng RoPE (block 0), độc lập hoàn toàn với quantize -> đây là lý
  ## do fp32 và int4 đều lệch giống hệt nhau ở after_block_0.
  ##   - "linear": HF's LlamaLinearScalingRotaryEmbedding chia position cho
  ##     scalingFactor TRƯỚC khi nhân invFreq (theta không đổi).
  ##   - "dynamic": NTK-aware - theta được scale lại theo seqLen thực tế so
  ##     với maxSeqLen gốc (chỉ áp dụng khi seqLen > max_position_embeddings
  ##     gốc; ở đây coi maxSeqLen truyền vào ĐÃ LÀ max_position_embeddings
  ##     gốc nên nhánh dynamic dùng chung công thức tại pos lớn nhất bảng).
  result.headDim = headDim
  result.maxSeqLen = maxSeqLen
  let half = headDim div 2
  var effTheta = theta
  if scalingType == "dynamic" and scalingFactor > 1.0'f32:
    # NTK: theta' = theta * (factor * maxSeqLen/origMaxSeqLen - (factor-1)) ^ (headDim/(headDim-2))
    # ở đây maxSeqLen truyền vào coi như đã là độ dài mục tiêu sau scale.
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
  ## data layout: [B, T, H, D] phẳng. posOffset = vị trí tuyệt đối của token đầu
  ## tiên trong T (dùng khi decode từng token 1, T=1 nhưng vị trí > 0).
  ##
  ## SỬA: công thức RoPE x1*cos-x2*sin / x1*sin+x2*cos chính là dạng chuẩn
  ## "rotate-half": out = x*cos + rotate_half(x)*sin, với
  ## rotate_half(x) = concat(-x2, x1). Viết lại thế này thì TOÀN BỘ phép
  ## nhân/cộng số học là elementwise trên mảng full [B,T,H,D] -> vừa khít với
  ## beMul/beAdd (hoặc sessionVecOp) sẵn có trong backend, không cần kernel
  ## RoPE riêng. Phần build cosFull/sinFull/rotHalf bên dưới CHỈ là sắp xếp
  ## lại vị trí phần tử (gather + đổi dấu), không phải phép toán nặng, nên
  ## vẫn ở CPU (backend không có kernel gather) - đúng đúng phần "GPU đủ
  ## add/mul" mà không đủ gather/broadcast.
  let half = D div 2
  let n = B * T * H * D
  var cosFull = newSeq[float32](n)
  var sinFull = newSeq[float32](n)
  var rotHalf = newSeq[float32](n)
  for b in 0 ..< B:
    for t in 0 ..< T:
      let pos = posOffset + t
      doAssert pos < rope.maxSeqLen, "RoPE: vị trí vượt quá max_seq_len đã precompute"
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

  # Fallback: vẫn 100% GPU (không phải vòng lặp CPU), chỉ là 2 mul + 1 add
  # KHÔNG cùng 1 session nếu backend hiện tại không hỗ trợ session.
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
  result.gateProj = newLinear(hidden, intermediate)
  result.upProj = newLinear(hidden, intermediate)
  result.downProj = newLinear(intermediate, hidden)

proc forward*(ff: SwiGLUL, x: Tensor, ctx: Backend): Tensor =
  let gateRaw = ff.gateProj.forward(x, ctx)
  let upRaw = ff.upProj.forward(x, ctx)
  let n = gateRaw.data.len

  # SỬA: gộp sigmoid+mul+mul vào 1 SESSION - dữ liệu ở lại GPU xuyên suốt,
  # không upload/download qua CPU giữa từng phép toán (trước đây beSigmoid,
  # rồi vòng for CPU, rồi beMul mỗi cái tự upload/download riêng - 3 lượt
  # CPU<->GPU cho có 3 phép toán liên tiếp).
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
    # rơi xuống nhánh per-op bên dưới nếu session thất bại giữa chừng

  # Fallback: từng op riêng qua beXxx (vẫn 100% GPU, chỉ là không cùng 1
  # session nên có upload/download giữa mỗi bước).
  let sig = beSigmoid(ctx, gateRaw.data)
  let silu = beMul(ctx, gateRaw.data, sig)
  let gated = beMul(ctx, silu, upRaw.data)
  var gatedT = newTensor(upRaw.shape)
  gatedT.data = gated
  result = ff.downProj.forward(gatedT, ctx)

# ===================================================================
# KV Cache (paged) cho GQA -- lưu theo layout [nKVHeads, maxSeqLen, headDim]
# ===================================================================
type
  LayerKVCache* = object
    k*, v*: seq[float32]   # flat [nKVHeads * maxSeqLen * headDim], B=1
    filled*: int           # số vị trí đã ghi

  KVCacheL* = object
    layers*: seq[LayerKVCache]
    nKVHeads*, headDim*, maxSeqLen*: int
    seqLen*: int            # vị trí hiện tại (chung cho mọi layer)

proc newKVCacheL*(nLayers, nKVHeads, headDim, maxSeqLen: int): KVCacheL =
  result.nKVHeads = nKVHeads
  result.headDim = headDim
  result.maxSeqLen = maxSeqLen
  result.seqLen = 0
  result.layers = newSeq[LayerKVCache](nLayers)
  for l in 0 ..< nLayers:
    result.layers[l].k = newSeq[float32](nKVHeads * maxSeqLen * headDim)
    result.layers[l].v = newSeq[float32](nKVHeads * maxSeqLen * headDim)
    result.layers[l].filled = 0

proc appendKV*(cache: var KVCacheL, layer: int, kNew, vNew: seq[float32], nTok: int) =
  ## kNew/vNew layout: [nTok, nKVHeads, headDim] (B=1). Ghi tuần tự vào cache
  ## tại vị trí [cache.seqLen .. cache.seqLen+nTok).
  let D = cache.headDim
  let H = cache.nKVHeads
  doAssert cache.seqLen + nTok <= cache.maxSeqLen, "KV cache overflow"
  for t in 0 ..< nTok:
    for h in 0 ..< H:
      let srcOff = (t * H + h) * D
      let dstOff = (h * cache.maxSeqLen + (cache.seqLen + t)) * D
      for d in 0 ..< D:
        cache.layers[layer].k[dstOff + d] = kNew[srcOff + d]
        cache.layers[layer].v[dstOff + d] = vNew[srcOff + d]
  cache.layers[layer].filled = cache.seqLen + nTok

# ===================================================================
# GQA CausalSelfAttention (RoPE + KV-cache) — thay cho CausalSelfAttention gốc
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
  result.qProj = newLinear(hiddenSize, hiddenSize)
  result.kProj = newLinear(hiddenSize, kvDim)
  result.vProj = newLinear(hiddenSize, kvDim)
  result.oProj = newLinear(hiddenSize, hiddenSize)
  result.scale = 1'f32 / sqrt(float32(result.headDim))

## forwardQKVFused (dequant-fp32-then-matmul QKV fusion) đã bị XOÁ - không
## còn nơi nào gọi tới (dead code) và nó vẫn đi qua đường chậm
## dequantizeTensorTransposed() mà linearFast() ở trên đã thay thế. Xem
## fusedNormQKVRoPE bên dưới cho đường dùng thật (beMatmulQ4 trực tiếp).

proc fusedNormQKVRoPE(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL,
                       ctx: Backend, T, posOffset: int): tuple[q, k, v: seq[float32]] =
  ## GỘP CROSS-SUBMODULE: RMSNorm(inputNorm, phần scale) -> QKV fused
  ## projection -> RoPE(Q) -> RoPE(K).
  ##
  ## TRƯỚC ĐÂY 4 bước này là 3 session GPU riêng (RMSNorm tự sessionBegin/End,
  ## RoPE(Q) tự session, RoPE(K) tự session) CỘNG THÊM 1 lần beMatmul KHÔNG
  ## qua session cho QKV (forwardQKVFused cũ) - tổng ~4 lượt commit+wait GPU/
  ## layer chỉ riêng phần đầu attention. Giờ CHỈ CÒN ĐÚNG 2 session:
  ##   Session #1: RMSNorm-scale (2 mul) + QKV matmul, encode nối tiếp trong
  ##               CÙNG 1 sessionBegin/sessionEnd (không round-trip CPU giữa
  ##               2 bước này nữa).
  ##   Session #2: RoPE(Q) VÀ RoPE(K) GỘP CHUNG (trước đây 2 session riêng)
  ##               thành 1 batch elementwise (mul,mul,add) duy nhất.
  ## Ranh giới GIỮA session #1 và #2 là BẮT BUỘC, không phải tuỳ chọn: RoPE
  ## cần "rotate_half" (gather đổi dấu 2 nửa vector) trên Q/K vừa ra khỏi
  ## matmul, mà session API hiện tại (sessionMatmul/sessionVecOp/
  ## sessionActivation/sessionSoftmax) KHÔNG có kernel gather -> phải đọc
  ## Q/K về CPU để build rotHalf/cosFull/sinFull rồi upload lại. Trên Metal,
  ## đọc dữ liệu giữa chừng đòi hỏi session đã commit+wait xong (xem
  ## metal_shim.m/metalSessionEnd) nên đây là ranh giới cứng của phần cứng/API,
  ## không phải do chưa tối ưu.
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  let (rows, inF) = flatten2D(x.shape)
  let n = rows * inF

  # SỬA (bug thật): TRƯỚC ĐÂY rmsNormPrep() (chạm CPU để chạy sqrt + tốn 1
  # lượt GPU round-trip beMul/beMatmul riêng) bị gọi VÔ ĐIỀU KIỆN ở đây, TRƯỚC
  # CẢ khi biết đường fused GPU thật (sessionRmsnormEnc bên dưới, mean+rsqrt+
  # scale gộp hết trong 1 kernel GPU) có chạy được hay không. Nghĩa là ngay
  # cả khi sessionRmsnormEnc THÀNH CÔNG (đường nhanh nhất, không đụng CPU),
  # code vẫn tốn thêm 1 lượt GPU thừa + 1 lần chạm CPU vô ích MỖI TOKEN MỖI
  # LAYER cho 1 giá trị (invBroadcast/wBroadcast) không hề dùng tới. Giờ chỉ
  # gọi rmsNormPrep() ở ĐÚNG những nhánh fallback thật sự cần nó (xem các chỗ
  # gọi `rmsNormPrep(norm, x, ctx)` bên dưới).

  let be = ctx.kind.toByby()
  var normed: seq[float32]
  var qFlat0, kFlat0, vFlat: seq[float32]

  # SỬA (điểm chính - trả lời "sao lâu mới ra token"): TRƯỚC ĐÂY session
  # RMSNorm-scale phải sessionEnd() rồi đọc `normed` về CPU CHỈ ĐỂ làm input
  # cho linearFastBias (mỗi cái TỰ mở/đóng session GPU RIÊNG của nó) - tức 1
  # session RMSNorm + 3 session Q/K/V = 4 lượt commit+waitUntilCompleted GPU
  # CHỈ RIÊNG phần này, MỖI LAYER, MỖI TOKEN (32 layer x ~10-12 lượt/layer ≈
  # 350+ round-trip GPU/token - waitUntilCompleted có overhead cố định đáng
  # kể trên driver Intel Metal cũ, đây mới là thứ chiếm phần lớn 5s/token,
  # KHÔNG phải throughput compute của kernel). Giờ sessionMatmulQ4 (đã có sẵn
  # từ bản resident-weight) encode THẲNG vào CÙNG session đang mở, dùng
  # `hNormed` (đang ở trên GPU) làm activation - không cần đọc `normed` về
  # CPU giữa RMSNorm và QKV nữa. 4 lượt -> còn 1 lượt.
  # DEBUG (tạm, để khoanh vùng bug OpenCL sinh token rác): set biến môi
  # trường BYBY_FORCE_LEGACY_QKV=1 để ép luôn đi "ĐƯỜNG CŨ" bên dưới
  # (linearFastBias + RoPE CPU, không qua sessionMatmulQ4/sessionRope mới)
  # dù resident đã sẵn sàng. Nếu token ra lại đúng khi bật cờ này trên
  # backend opencl -> bug chắc chắn nằm trong nhánh usedSession/sessionRope
  # của OpenCL (không phải ở kernel matmul/softmax cơ bản dùng chung).
  # FIX (tiếp): TRƯỚC ĐÂY nhánh gộp-session này CHỈ kích hoạt khi Q/K/V
  # resident theo đường int4-asymmetric thật (sessionMatmulQ4) - model
  # export bằng --int8 (mặc định của export_to_nimq.py) hoặc --int4
  # symmetric KHÔNG có residentReady=true nên qkvResident luôn false, rơi
  # thẳng xuống "ĐƯỜNG CŨ" bên dưới: RMSNorm tự 1 session + Q/K/V mỗi cái
  # lại tự mở 1 session riêng (residentFp32Ready, xem nimformer.forward) ->
  # vẫn 4 lượt commit+wait GPU/layer dù đã có resident fp32. Giờ nhánh gộp
  # dùng ĐƯỢC CẢ residentFp32 (sessionMatmul thường, không cần kernel int4
  # riêng) -> model int8/tied cũng chỉ còn 1 session cho RMSNorm+QKV, y hệt
  # đường Q4.
  let qkvResidentQ4 = attn.qProj.residentReady and attn.kProj.residentReady and attn.vProj.residentReady
  let qkvResidentFp32 = (not qkvResidentQ4) and attn.qProj.residentFp32Ready and
    attn.kProj.residentFp32Ready and attn.vProj.residentFp32Ready
  let qkvResident = (not existsEnv("BYBY_FORCE_LEGACY_QKV")) and (qkvResidentQ4 or qkvResidentFp32)
  debugPathOnce("qkv.resident_ready", qkvResidentQ4 or qkvResidentFp32,
    "Q4=" & $qkvResidentQ4 & " Fp32=" & $qkvResidentFp32 &
    " (false ca 2 => uploadResident() chua chay hoac chay that bai cho qProj/kProj/vProj)")
  var usedSession = qkvResident and sessionBegin(be)
  # SỬA (gộp RoPE vào CÙNG session với QKV, xem sessionRope trong
  # gpubackend.nim + rope_kernel trong vecop_matmul.metal/.cl): trước đây
  # RoPE BẮT BUỘC phải là 1 session riêng ("Session #2" bên dưới) vì session
  # API cũ không có kernel gather/rotate_half. Giờ Metal VÀ OpenCL đều có
  # kernel rope_kernel thật (openclSessionRopeEnc/metalSessionRopeEnc), chỉ
  # CUDA còn thiếu (cần build rope_kernel.cu bằng nvcc thành .ptx trước, xem
  # cudaRopeR trong cuda_driver.nim) - sessionRope() tự trả về false cho CUDA
  # nếu getFn("rope_kernel") không tìm thấy trong PTX đang nạp. Nếu
  # sessionRope() chạy được, ta xoay Q/K NGAY TRONG session QKV, TRƯỚC
  # sessionEnd - Q/K đọc về CPU ở dưới đã là kết quả RoPE xong, không cần
  # Session #2 nữa. Nếu sessionRope trả về false (chưa hỗ trợ / lỗi),
  # ropeFused vẫn là false -> code cũ bên dưới (Session #2 / đường CPU) chạy
  # y hệt như trước, không đổi hành vi.
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
      hGamma = sessionUpload(be, norm.weight.data)
      var hNormedTry = sessionAllocScratch(be, n)
      if sessionRmsnormEnc(be, hX, hGamma, hNormedTry, rows, inF, norm.eps):
        hNormed = hNormedTry
        normedFast = true
      else:
        sessionFree(be, hNormedTry)   # tránh leak scratch buffer khi thất bại
    if not normedFast:
      let (invBroadcast, wBroadcast, _) = rmsNormPrep(norm, x, ctx)
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
        doAssert pos < rope.maxSeqLen, "RoPE: vị trí vượt quá max_seq_len đã precompute"
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
      qFlat0 = sessionRead(be, hQ, rows * attn.qProj.outF)  # đã xoay RoPE sẵn nếu ropeFused
      kFlat0 = sessionRead(be, hK, rows * attn.kProj.outF)  # đã xoay RoPE sẵn nếu ropeFused
      vFlat = sessionRead(be, hV, rows * attn.vProj.outF)
    sessionFree(be, hX)
    if tryFastRmsnorm: sessionFree(be, hGamma)
    if not normedFast:
      sessionFree(be, hInv); sessionFree(be, hW)
      var hT1v = hT1
      sessionFree(be, hT1v)
    sessionFree(be, hNormed)
    sessionFree(be, hQ); sessionFree(be, hK); sessionFree(be, hV)
    if okQ and okK:
      sessionFree(be, hCosT); sessionFree(be, hSinT)
    if qFlat0.len == 0 or kFlat0.len == 0 or vFlat.len == 0:
      usedSession = false   # sessionMatmulQ4 fail (backend không hỗ trợ) -> rơi về đường cũ bên dưới
      ropeFused = false     # an toàn: nếu session coi như thất bại, KHÔNG được tin đã RoPE xong

  debugPathOnce("qkv.path", usedSession)
  if not usedSession:
    # ĐƯỜNG CŨ (backend không phải Metal/OpenCL, hoặc resident chưa sẵn sàng,
    # hoặc sessionMatmulQ4 thất bại) - vẫn đúng, chỉ mất phần gộp session.
    let (invBroadcast, wBroadcast, _) = rmsNormPrep(norm, x, ctx)
    var usedSessionOld = sessionBegin(be)
    if usedSessionOld:
      var hX = sessionUpload(be, x.data)
      var hInv = sessionUpload(be, invBroadcast)
      var hW = sessionUpload(be, wBroadcast)
      let hT1 = sVecOp(be, "mul", hX, hInv, n)          # RMSNorm scale bước 1
      let hNormed = sVecOp(be, "mul", hT1, hW, n)       # RMSNorm scale bước 2
      usedSessionOld = sessionEnd(be)   # bắt buộc: normed cần về CPU để làm input cho beMatmulQ4
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
    # Bias (nếu có) - cộng trên CPU sau khi đọc kết quả về, giống linearFastBias.
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

  # ---- Session #2: RoPE(Q) + RoPE(K) GỘP CHUNG 1 session (trước đây 2
  # session riêng, mỗi cái tự upload/download). Ghép Q và K thành 1 mảng
  # liên tục [nQ+nK] rồi chạy đúng 3 sessionVecOp (mul,mul,add) MỘT LẦN cho
  # cả 2, thay vì 2 lần x 3 vecop = 6 lệnh trên 2 session khác nhau. ----
  let half = D div 2
  let nQ = T * H * D
  let nK = T * HKV * D
  let nAll = nQ + nK

  proc buildRoPEArrays(data: seq[float32], Th, Hh: int, base: int,
                        cosFull, sinFull, rotHalf: var seq[float32]) =
    for t in 0 ..< Th:
      let pos = posOffset + t
      doAssert pos < rope.maxSeqLen, "RoPE: vị trí vượt quá max_seq_len đã precompute"
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
    # RoPE đã chạy xong TRONG session QKV ở trên (xem sessionRope) - qFlat/
    # kFlat ở đây đã là kết quả sau rotate_half rồi, KHÔNG áp dụng lại lần
    # nữa (áp lại sẽ ra sai - rotate_half không phải phép idempotent).
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

## ---- PREFILL: xử lý cả prompt 1 lần bằng beAttentionFused (đã có causal mask
## nội bộ, đúng mục đích thiết kế của kernel này) ----
proc forwardPrefill*(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL, ctx: Backend,
                      cache: var KVCacheL, layerIdx: int): Tensor =
  let B = x.shape[0]
  let T = x.shape[1]
  let C = x.shape[2]
  doAssert B == 1, "Bản decode hiện chỉ hỗ trợ B=1 (single-sequence generation)"
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  let nRep = H div HKV

  # GỘP SESSION cross-submodule: RMSNorm(inputNorm) -> QKV fused -> RoPE(Q)
  # -> RoPE(K) (xem fusedNormQKVRoPE ở trên) - trước đây đây là 3 session +
  # 1 matmul rời, giờ còn 2 session.
  let (qData, kData, vData) = fusedNormQKVRoPE(attn, norm, x, rope, ctx, T, posOffset = 0)

  # Lưu K,V (đã RoPE) vào cache theo layout [T,HKV,D] -> appendKV tự chuyển
  # sang [HKV, maxSeqLen, D] bên trong.
  appendKV(cache, layerIdx, kData, vData, T)

  # Mở rộng K,V từ HKV head -> H head (mỗi kv-head lặp lại nRep lần liên tục,
  # đúng convention repeat_kv của HF) để dùng beAttentionFused (yêu cầu Q,K,V
  # cùng số head).
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

  # SỬA BUG: beAttentionFused cần layout [H,T,D] (head-major) chứ không phải
  # [T,H,D] (position-major) mà qData/kExp/vExp đang có (xem toHeadMajor ở
  # trên). Thiếu bước transpose này khiến kernel đọc lẫn head/token -> attention
  # sai hoàn toàn -> sinh token rác ngay từ prefill.
  let qHM = toHeadMajor(qData, T, H, D)
  let kHM = toHeadMajor(kExp, T, H, D)
  let vHM = toHeadMajor(vExp, T, H, D)
  let (attnOutFlatHM, _) = beAttentionFused(ctx, qHM, kHM, vHM, [], B, H, T, D, attn.scale)
  let attnOutFlat = toPosMajor(attnOutFlatHM, T, H, D)   # về lại [T,H,D]=[T,C] cho oProj
  var attnOut = newTensor(@[B, T, C])
  attnOut.data = attnOutFlat
  result = attn.oProj.forward(attnOut, ctx)

## ---- DECODE: 1 token mới, attention thủ công qua beMatmul+beSoftmax trên
## toàn bộ cache (O(S), tuyến tính -- đây mới là "paged attention" thật) ----
proc forwardDecode*(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL, ctx: Backend,
                     cache: var KVCacheL, layerIdx: int): Tensor =
  let C = x.shape[2]
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  let nRep = H div HKV
  let S = cache.seqLen + 1   # tổng độ dài sau khi thêm token này

  # GỘP SESSION cross-submodule (giống forwardPrefill ở trên).
  let (qData, kNew, vNew) = fusedNormQKVRoPE(attn, norm, x, rope, ctx, T = 1, posOffset = cache.seqLen)

  appendKV(cache, layerIdx, kNew, vNew, 1)

  var outHeads = newSeq[float32](H * D)
  let maxS = cache.maxSeqLen
  let backend = ctx.kind.toByby()

  ensurePersistentKV(backend, layerIdx, HKV, maxS, D)
  var persistK = gPersistK[layerIdx]
  var persistV = gPersistV[layerIdx]

  # Ghi ĐÚNG 1 dòng mới (kNew/vNew, T=1, layout [HKV,D]) vào buffer
  # GPU-resident của layer này - KHÔNG đụng lại S-1 dòng cũ đã có sẵn trên
  # GPU từ các bước trước (xem giải thích ở ensurePersistentKV phía trên).
  var kvWriteOk = sessionBegin(backend)
  if kvWriteOk:
    for hkv in 0 ..< HKV:
      var kSlice = newSeq[float32](D)
      var vSlice = newSeq[float32](D)
      for d in 0 ..< D:
        kSlice[d] = kNew[hkv * D + d]
        vSlice[d] = vNew[hkv * D + d]
      let hKNew = sessionUpload(backend, kSlice)
      let hVNew = sessionUpload(backend, vSlice)
      let okK = sessionCopyInto(backend, hKNew, persistK[hkv], cache.seqLen * D, D)
      let okV = sessionCopyInto(backend, hVNew, persistV[hkv], cache.seqLen * D, D)
      sessionFree(backend, hKNew); sessionFree(backend, hVNew)
      if not (okK and okV): kvWriteOk = false
    kvWriteOk = sessionEnd(backend) and kvWriteOk

  # Fallback AN TOÀN (hiếm - chỉ khi sessionCopyInto không khả dụng trên
  # backend/máy hiện tại): quay lại đúng đường CŨ (copy CPU cache + upload
  # full-S mỗi head, chậm nhưng ĐÚNG) - dùng kBlocks/vBlocks CPU y hệt bản
  # trước, KHÔNG đổi kết quả, chỉ mất phần tăng tốc GPU-resident.
  var kBlocksFallback = newSeq[seq[float32]](HKV)
  var vBlocksFallback = newSeq[seq[float32]](HKV)
  if not kvWriteOk:
    for hkv in 0 ..< HKV:
      let headBase = hkv * maxS * D
      var kBlock = newSeq[float32](S * D)
      copyMem(addr kBlock[0], unsafeAddr cache.layers[layerIdx].k[headBase], S * D * sizeof(float32))
      kBlocksFallback[hkv] = kBlock
      var vBlock = newSeq[float32](S * D)
      copyMem(addr vBlock[0], unsafeAddr cache.layers[layerIdx].v[headBase], S * D * sizeof(float32))
      vBlocksFallback[hkv] = vBlock

  proc getHK(hkv: int): SessionHandle =
    if kvWriteOk: persistK[hkv] else: sessionUpload(backend, kBlocksFallback[hkv])
  proc getHV(hkv: int): SessionHandle =
    if kvWriteOk: persistV[hkv] else: sessionUpload(backend, vBlocksFallback[hkv])

  var qGroups = newSeq[seq[float32]](HKV)
  for hkv in 0 ..< HKV:
    var qGroup = newSeq[float32](nRep * D)
    for r in 0 ..< nRep:
      let h = hkv * nRep + r
      for d in 0 ..< D:
        # gộp luôn scale vào Q ở CPU (rẻ, mảng nhỏ) để khỏi cần thêm 1 GPU op
        # riêng cho phép nhân scale trong session (session chỉ có matmul/softmax)
        qGroup[r * D + d] = qData[h * D + d] * attn.scale
    qGroups[hkv] = qGroup

  proc transposeKBlock(kBlock: seq[float32]): seq[float32] =
    ## Chỉ gọi khi backend KHÔNG hỗ trợ sessionMatmulNT (fallback). [S,D] -> [D,S].
    result = newSeq[float32](D * S)
    for s in 0 ..< S:
      for d in 0 ..< D:
        result[d * S + s] = kBlock[s * D + d]

  proc buildKBlockFromCache(hkv: int): seq[float32] =
    ## Dùng CHO transposeKBlock fallback (rất hiếm - chỉ khi sessionMatmulNT
    ## không hỗ trợ) - luôn đúng bất kể kvWriteOk vì appendKV() ở trên đã ghi
    ## vào cache.layers CPU rồi, không phụ thuộc gì vào GPU-resident buffer.
    let headBase = hkv * maxS * D
    result = newSeq[float32](S * D)
    copyMem(addr result[0], unsafeAddr cache.layers[layerIdx].k[headBase], S * D * sizeof(float32))

  proc buildVBlockFromCache(hkv: int): seq[float32] =
    ## Dùng CHO nhánh KHÔNG session nào chạy được (fallback sâu nhất, CPU
    ## thuần) - cùng lý do như buildKBlockFromCache ở trên.
    let headBase = hkv * maxS * D
    result = newSeq[float32](S * D)
    copyMem(addr result[0], unsafeAddr cache.layers[layerIdx].v[headBase], S * D * sizeof(float32))
  # SỬA (GỘP THÊM 1 BƯỚC NỮA - session attention-decode giờ nối thẳng vào
  # oProj, KHÔNG còn tách 2 session như bản trước): trước đây phải
  # sessionRead() outHeads về CPU giữa attention và oProj CHỈ ĐỂ ghép nRep*D
  # kết quả rời rạc của từng KV-head thành 1 vector [H*D] liên tục (input
  # matmul oProj) - không có cách nào ghép trên GPU vì session API cũ không
  # có primitive "copy vào 1 vùng con của buffer khác". Giờ có
  # sessionCopyInto (xem gpubackend.nim - Metal dùng copy_into_kernel mới,
  # CUDA dùng cuMemcpyDtoDAsync thẳng, OpenCL dùng clEnqueueCopyBuffer thẳng,
  # cả 3 đều có cài đặt thật, không có nhánh "chưa hỗ trợ" ngầm nào) nên
  # từng outGroup (nRep*D phần tử của 1 KV-head) được copy THẲNG vào đúng vị
  # trí của nó trong 1 buffer scratch [H*D] chung, TRONG CÙNG session đang
  # mở, rồi nối thẳng sang sessionMatmulQ4(oProj) mà không rời GPU lần nào.
  # Kết quả: 2 session/layer (attention + oProj) -> còn ĐÚNG 1.
  # sessionCopyInto CHƯA từng chạy trên phần cứng thật (không có máy test ở
  # đây khi viết bản này) - có escape-hatch BYBY_LEGACY_ATTN_OPROJ=1 để ép
  # quay lại đường 2-session cũ (đã tự tay xác nhận đúng) nếu phát hiện sai
  # lệch/crash khi build thật.
  let legacyAttnOProj = existsEnv("BYBY_LEGACY_ATTN_OPROJ")
  let oProjResidentMergedQ4 = (not legacyAttnOProj) and attn.oProj.residentReady
  let oProjResidentMergedFp32 = (not legacyAttnOProj) and (not oProjResidentMergedQ4) and attn.oProj.residentFp32Ready
  let oProjResidentMerged = oProjResidentMergedQ4 or oProjResidentMergedFp32
  debugPathOnce("attn_oproj.resident_ready", oProjResidentMerged,
    "Q4=" & $oProjResidentMergedQ4 & " Fp32=" & $oProjResidentMergedFp32)
  var usedMergedSession = oProjResidentMerged and sessionBegin(backend)
  var oProjOutMerged: seq[float32]
  if usedMergedSession:
    var allHandles = newSeq[SessionHandle]()
    var okAll = true
    let hOutHeads = sessionAllocScratch(backend, H * D)   # buffer chung [H*D], ghi từng mảnh vào đây
    allHandles.add(hOutHeads)
    for hkv in 0 ..< HKV:
      let hQ = sessionUpload(backend, qGroups[hkv])
      let hK = getHK(hkv)
      let hV = getHV(hkv)
      allHandles.add(hQ)
      if not kvWriteOk: allHandles.add(hK); allHandles.add(hV)  # buffer GPU-resident KHONG duoc free

      var hScores: SessionHandle
      let (ntHandle, ntOk) = sMatmulNT(backend, hQ, nRep, D, hK, S)
      if ntOk:
        hScores = ntHandle
        allHandles.add(hScores)
      else:
        let hKT = sessionUpload(backend, transposeKBlock(buildKBlockFromCache(hkv)))
        allHandles.add(hKT)
        hScores = sMatmul(backend, hQ, nRep, D, hKT, S)
        allHandles.add(hScores)

      let hProbs = sSoftmax(backend, hScores, nRep, S)
      let hOutGroup = sMatmul(backend, hProbs, nRep, S, hV, D)
      allHandles.add(hProbs); allHandles.add(hOutGroup)

      # Ghi thẳng outGroup (nRep*D phần tử) vào đúng vị trí của nó trong
      # hOutHeads - offset tính bằng phần tử, đúng thứ tự head y hệt vòng
      # lặp CPU cũ (h = hkv*nRep + r).
      var dstHandle = hOutHeads
      let okCopy = sessionCopyInto(backend, hOutGroup, dstHandle, hkv * nRep * D, nRep * D)
      if not okCopy: okAll = false

    var hOProjOut = sessionAllocScratch(backend, C)
    allHandles.add(hOProjOut)
    let okO = okAll and (if oProjResidentMergedQ4: sessionMatmulQ4(backend, hOutHeads, attn.oProj.residentW, hOProjOut, 1)
                          else: sessionMatmul(backend, hOutHeads, attn.oProj.residentFp32, hOProjOut, 1, attn.oProj.inF, attn.oProj.outF))
    usedMergedSession = sessionEnd(backend)
    if usedMergedSession and okO:
      oProjOutMerged = sessionRead(backend, hOProjOut, C)
    for h in allHandles.mitems: sessionFree(backend, h)
    if oProjOutMerged.len == 0:
      usedMergedSession = false   # sessionCopyInto/matmul thất bại -> rơi về đường cũ 2-session bên dưới, KHÔNG coi false là đã chạy xong
    elif attn.oProj.bias.data.len == C:
      for o in 0 ..< C: oProjOutMerged[o] += attn.oProj.bias.data[o]

  debugPathOnce("attn_oproj.path", usedMergedSession)
  if usedMergedSession:
    result = newTensor(@[1, 1, C])
    result.data = oProjOutMerged
    return

  # ─────────────────────────────────────────────────────────────────────
  # ĐƯỜNG CŨ (fallback): sessionCopyInto không khả dụng/thất bại, hoặc
  # BYBY_LEGACY_ATTN_OPROJ=1, hoặc oProj chưa resident - dùng lại đúng logic
  # 2-session đã tự tay xác nhận đúng trước đây (attention-decode session
  # riêng, rồi oProj session riêng, có sessionRead ở giữa).
  # ─────────────────────────────────────────────────────────────────────
  var usedSession = sessionBegin(backend)
  if usedSession:
    var outHandles = newSeq[SessionHandle](HKV)
    var allHandles = newSeq[SessionHandle]()
    for hkv in 0 ..< HKV:
      let hQ = sessionUpload(backend, qGroups[hkv])
      let hK = getHK(hkv)
      let hV = getHV(hkv)
      allHandles.add(hQ)
      if not kvWriteOk: allHandles.add(hK); allHandles.add(hV)  # buffer GPU-resident KHONG duoc free

      var hScores: SessionHandle
      let (ntHandle, ntOk) = sMatmulNT(backend, hQ, nRep, D, hK, S)
      if ntOk:
        hScores = ntHandle
        allHandles.add(hScores)
      else:
        let hKT = sessionUpload(backend, transposeKBlock(buildKBlockFromCache(hkv)))
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

  if not usedSession:
    for hkv in 0 ..< HKV:
      let kBlockT = transposeKBlock(buildKBlockFromCache(hkv))
      var scores = beMatmul(ctx, qGroups[hkv], nRep, D, kBlockT, D, S)
      let probs = beSoftmax(ctx, scores, nRep, S)
      let outGroup = beMatmul(ctx, probs, nRep, S, buildVBlockFromCache(hkv), S, D)
      for r in 0 ..< nRep:
        let h = hkv * nRep + r
        for d in 0 ..< D:
          outHeads[h * D + d] = outGroup[r * D + d]

  let legacyOProj = existsEnv("BYBY_LEGACY_OPROJ")
  var oProjOut: seq[float32]
  let oProjResidentQ4 = (not legacyOProj) and attn.oProj.residentReady
  let oProjResidentFp32 = (not legacyOProj) and (not oProjResidentQ4) and attn.oProj.residentFp32Ready
  let oProjResident = oProjResidentQ4 or oProjResidentFp32
  var usedOProjSession = oProjResident and sessionBegin(backend)
  if usedOProjSession:
    var hIn = sessionUpload(backend, outHeads)
    var hOut = sessionAllocScratch(backend, C)
    let okO = if oProjResidentQ4: sessionMatmulQ4(backend, hIn, attn.oProj.residentW, hOut, 1)
              else: sessionMatmul(backend, hIn, attn.oProj.residentFp32, hOut, 1, attn.oProj.inF, attn.oProj.outF)
    usedOProjSession = sessionEnd(backend)
    if usedOProjSession and okO:
      oProjOut = sessionRead(backend, hOut, C)
    sessionFree(backend, hIn); sessionFree(backend, hOut)
    if oProjOut.len == 0:
      usedOProjSession = false
    elif attn.oProj.bias.data.len == C:
      for o in 0 ..< C: oProjOut[o] += attn.oProj.bias.data[o]

  if usedOProjSession:
    result = newTensor(@[1, 1, C])
    result.data = oProjOut
  else:
    var attnOut = newTensor(@[1, 1, C])
    attnOut.data = outHeads
    result = attn.oProj.forward(attnOut, ctx)

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
  ## GỘP CROSS-SUBMODULE toàn bộ đuôi của 1 block: residual-add(sau Attn) ->
  ## RMSNorm(postAttnNorm, scale) -> FFN (gate/up matmul + sigmoid+mul+mul +
  ## down matmul) -> residual-add(sau FFN) THÀNH ĐÚNG 1 SESSION GPU DUY NHẤT.
  ##
  ## TRƯỚC ĐÂY: addT (plain, không session) + RMSNorm (session riêng) + FFN
  ## gate/up matmul (plain, không session) + FFN sigmoid/mul/mul (session
  ## riêng) + FFN down matmul (plain) + addT (plain) = 2 session mở/đóng +
  ## 4 lượt GPU rời rạc khác, mỗi lượt tự upload/download.
  ##
  ## GIỜ: residual-add đầu, RMSNorm-scale, gate-matmul, up-matmul, sigmoid,
  ## silu-mul, gated-mul, down-matmul, residual-add cuối - TẤT CẢ encode nối
  ## tiếp vào CÙNG 1 sessionBegin/sessionEnd, không có điểm đọc GPU->CPU nào
  ## ở giữa (khác với fusedNormQKVRoPE, đuôi này KHÔNG cần gather/rotate_half
  ## nên không có ranh giới cứng nào bắt buộc phải ngắt session).
  let n = x.data.len
  let (rows, hidden) = flatten2D(x.shape)
  doAssert hidden == norm.dim

  # ---- CPU bắt buộc TRƯỚC session: chỉ phần reduce RMSNorm (đọc sumSq nhỏ
  # để chạy sqrt). KHÔNG còn dequant weight gate/up/down ở đây nữa - đây
  # chính là phần tốn thời gian nhất trước đây (3 ma trận full-size giải nén
  # + upload lại MỖI TOKEN, MỖI LAYER dù trọng số không hề đổi giữa các
  # token) - xem linearFast() ở đầu file.
  # SỬA (bug thật, cùng loại với fusedNormQKVRoPE ở trên): rmsNormPrep() chỉ
  # gọi LAZY ở đúng nhánh fallback cần nó bên dưới, không gọi vô điều kiện ở
  # đây nữa (trước đây tốn 1 lượt GPU + 1 lần chạm CPU thừa mỗi token/layer
  # kể cả khi sessionRmsnormEnc chạy được, không cần giá trị này).
  let x2 = addT(ctx, x, attnOut)
  let interF = ff.gateProj.outF
  let nInter = rows * interF

  let be = ctx.kind.toByby()
  var normed: seq[float32]
  var gateRaw, upRaw: seq[float32]

  let gateUpResidentQ4 = ff.gateProj.residentReady and ff.upProj.residentReady
  let gateUpResidentFp32 = (not gateUpResidentQ4) and ff.gateProj.residentFp32Ready and ff.upProj.residentFp32Ready

  # SỬA (gộp 2 session FFN thành 1, thử TRƯỚC TIÊN): gate/up-matmul ->
  # (+bias GPU) -> sigmoid -> mul -> mul -> down-matmul -> (+bias GPU) chạy
  # liền 1 mạch, ĐÚNG 1 sessionBegin/sessionEnd cho toàn bộ FFN thay vì 2-3
  # như trước - giữa các bước này KHÔNG có phụ thuộc CPU bắt buộc nào (khác
  # RoPE cần gather), lần đọc CPU cũ chỉ tồn tại để cộng bias bằng vòng for,
  # giờ cộng bias ngay trên GPU (sVecOp "add" + broadcastBias có sẵn).
  # Giảm 1 lượt commit+waitUntilCompleted/layer (driver Metal Intel cũ: đây
  # là nguồn overhead chính, xem phân tích ở generate()).
  var ffOut: seq[float32]
  let legacyFFN = existsEnv("BYBY_LEGACY_FFN")  # cửa thoát hiểm: set BYBY_LEGACY_FFN=1 để tắt đường gộp mới
  let downResidentQ4 = ff.downProj.residentReady
  let downResidentFp32 = (not downResidentQ4) and ff.downProj.residentFp32Ready
  # FIX: TRƯỚC ĐÂY chỉ nhận resident Q4 (model --q4km) - model int8/int4-
  # symmetric (mặc định export_to_nimq.py) luôn rơi về "ĐƯỜNG CŨ" bên dưới dù
  # đã có resident fp32, mất hết phần gộp session. Giờ allResident đúng khi
  # CẢ 3 Linear đi CÙNG 1 đường (toàn Q4 hoặc toàn fp32) - model dùng đồng
  # nhất 1 kind cho mọi Linear nên trong thực tế luôn rơi vào 1 trong 2 nhánh.
  let allResidentQ4 = gateUpResidentQ4 and downResidentQ4
  let allResidentFp32 = gateUpResidentFp32 and downResidentFp32
  let allResident = (not legacyFFN) and (allResidentQ4 or allResidentFp32)
  debugPathOnce("ffn.resident_ready", allResidentQ4 or allResidentFp32,
    "Q4=" & $allResidentQ4 & " Fp32=" & $allResidentFp32 & " legacyFFN=" & $legacyFFN)
  var usedSessionFused = allResident and sessionBegin(be)
  let tryFastRmsnormFFN = usedSessionFused and not existsEnv("BYBY_DISABLE_SESSION_RMSNORM")
  if usedSessionFused:
    var hX2 = sessionUpload(be, x2.data)
    var hGamma: SessionHandle
    var hInv, hW, hT1: SessionHandle
    var hNormed: SessionHandle
    var normedFastFFN = false
    if tryFastRmsnormFFN:
      hGamma = sessionUpload(be, norm.weight.data)
      var hNormedTry = sessionAllocScratch(be, n)
      if sessionRmsnormEnc(be, hX2, hGamma, hNormedTry, rows, hidden, norm.eps):
        hNormed = hNormedTry
        normedFastFFN = true
      else:
        sessionFree(be, hNormedTry)
    if not normedFastFFN:
      let (invBroadcast, wBroadcast, _) = rmsNormPrep(norm, x2, ctx)
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
    # Cộng bias GPU (nếu có) - thay cho vòng for CPU cũ, giữ session mở liền mạch.
    if ff.gateProj.bias.data.len == ff.gateProj.outF:
      let hGB = sessionUpload(be, broadcastBias(ff.gateProj.bias, rows, ff.gateProj.outF))
      let hGate2 = sVecOp(be, "add", hGate, hGB, nInter)
      hGate = hGate2
    if ff.upProj.bias.data.len == ff.upProj.outF:
      let hUB = sessionUpload(be, broadcastBias(ff.upProj.bias, rows, ff.upProj.outF))
      let hUp2 = sVecOp(be, "add", hUp, hUB, nInter)
      hUp = hUp2
    let hSig = sActivation(be, "sigmoid", hGate, nInter)
    let hSilu = sVecOp(be, "mul", hGate, hSig, nInter)
    var hGated = sVecOp(be, "mul", hSilu, hUp, nInter)
    var hOut = sessionAllocScratch(be, rows * ff.downProj.outF)
    let okD = if allResidentQ4: sessionMatmulQ4(be, hGated, ff.downProj.residentW, hOut, rows)
              else: sessionMatmul(be, hGated, ff.downProj.residentFp32, hOut, rows, ff.downProj.inF, ff.downProj.outF)
    if ff.downProj.bias.data.len == ff.downProj.outF:
      let hDB = sessionUpload(be, broadcastBias(ff.downProj.bias, rows, ff.downProj.outF))
      let hOut2 = sVecOp(be, "add", hOut, hDB, rows * ff.downProj.outF)
      hOut = hOut2
    usedSessionFused = sessionEnd(be)
    if usedSessionFused and okG and okU and okD:
      ffOut = sessionRead(be, hOut, rows * ff.downProj.outF)
    if ffOut.len == 0:
      usedSessionFused = false
    # free hết handle trung gian - không quan trọng free đúng cái nào 2 lần
    # vì sessionFree trên handle đã free là no-op an toàn trong gpubackend.nim
    sessionFree(be, hX2); sessionFree(be, hInv); sessionFree(be, hW); sessionFree(be, hGamma)
    var hT1v = hT1; var hNormedV = hNormed
    sessionFree(be, hT1v); sessionFree(be, hNormedV)
    sessionFree(be, hGate); sessionFree(be, hUp)
    var hSigV = hSig; var hSiluV = hSilu
    sessionFree(be, hSigV); sessionFree(be, hSiluV); sessionFree(be, hGated)
    sessionFree(be, hOut)

  debugPathOnce("ffn.path", usedSessionFused)
  if not usedSessionFused:
    # ĐƯỜNG CŨ (fallback nếu chưa resident hết hoặc session thất bại giữa
    # chừng) - giữ nguyên y hệt logic 2-session trước đây, KHÔNG đổi hành vi.
    let (invBroadcast, wBroadcast, _) = rmsNormPrep(norm, x2, ctx)
    let gateUpResidentOld = ff.gateProj.residentReady and ff.upProj.residentReady
    var usedSession = gateUpResidentOld and sessionBegin(be)
    if usedSession:
      var hX2 = sessionUpload(be, x2.data)
      var hInv = sessionUpload(be, invBroadcast)
      var hW = sessionUpload(be, wBroadcast)
      let hT1 = sVecOp(be, "mul", hX2, hInv, n)
      var hNormed = sVecOp(be, "mul", hT1, hW, n)
      var hGate = sessionAllocScratch(be, nInter)
      var hUp = sessionAllocScratch(be, nInter)
      let okG = sessionMatmulQ4(be, hNormed, ff.gateProj.residentW, hGate, rows)
      let okU = sessionMatmulQ4(be, hNormed, ff.upProj.residentW, hUp, rows)
      usedSession = sessionEnd(be)
      if usedSession and okG and okU:
        gateRaw = sessionRead(be, hGate, nInter)
        upRaw = sessionRead(be, hUp, nInter)
      sessionFree(be, hX2); sessionFree(be, hInv); sessionFree(be, hW)
      var hT1v = hT1
      sessionFree(be, hT1v); sessionFree(be, hNormed)
      sessionFree(be, hGate); sessionFree(be, hUp)
      if gateRaw.len == 0 or upRaw.len == 0:
        usedSession = false

    if not usedSession:
      var usedSessionOld = sessionBegin(be)
      if usedSessionOld:
        var hX2 = sessionUpload(be, x2.data)
        var hInv = sessionUpload(be, invBroadcast)
        var hW = sessionUpload(be, wBroadcast)
        let hT1 = sVecOp(be, "mul", hX2, hInv, n)             # RMSNorm scale 1
        let hNormed = sVecOp(be, "mul", hT1, hW, n)           # RMSNorm scale 2
        usedSessionOld = sessionEnd(be)  # bắt buộc: normed cần về CPU làm input cho beMatmulQ4
        if usedSessionOld:
          normed = sessionRead(be, hNormed, n)
        var hT1v = hT1; var hNormedV = hNormed
        sessionFree(be, hX2); sessionFree(be, hInv); sessionFree(be, hW)
        sessionFree(be, hT1v); sessionFree(be, hNormedV)
      if not usedSessionOld:
        let t1 = beMul(ctx, x2.data, invBroadcast)
        normed = beMul(ctx, t1, wBroadcast)

      gateRaw = linearFastBias(ff.gateProj, normed, rows, ctx)
      upRaw = linearFastBias(ff.upProj, normed, rows, ctx)
    else:
      if ff.gateProj.bias.data.len == ff.gateProj.outF:
        for r in 0 ..< rows:
          let off = r * ff.gateProj.outF
          for o in 0 ..< ff.gateProj.outF: gateRaw[off + o] += ff.gateProj.bias.data[o]
      if ff.upProj.bias.data.len == ff.upProj.outF:
        for r in 0 ..< rows:
          let off = r * ff.upProj.outF
          for o in 0 ..< ff.upProj.outF: upRaw[off + o] += ff.upProj.bias.data[o]

    let downResident = ff.downProj.residentReady
    var usedSession2 = downResident and sessionBegin(be)
    if usedSession2:
      var hGate = sessionUpload(be, gateRaw)
      var hUp = sessionUpload(be, upRaw)
      let hSig = sActivation(be, "sigmoid", hGate, nInter)
      let hSilu = sVecOp(be, "mul", hGate, hSig, nInter)
      var hGated = sVecOp(be, "mul", hSilu, hUp, nInter)
      var hOut = sessionAllocScratch(be, rows * ff.downProj.outF)
      let okD = sessionMatmulQ4(be, hGated, ff.downProj.residentW, hOut, rows)
      usedSession2 = sessionEnd(be)
      if usedSession2 and okD:
        ffOut = sessionRead(be, hOut, rows * ff.downProj.outF)
      sessionFree(be, hGate); sessionFree(be, hUp)
      var hSigV = hSig; var hSiluV = hSilu
      sessionFree(be, hSigV); sessionFree(be, hSiluV); sessionFree(be, hGated)
      sessionFree(be, hOut)
      if ffOut.len == 0:
        usedSession2 = false

    if not usedSession2:
      var gated: seq[float32]
      var usedSessionOld2 = sessionBegin(be)
      if usedSessionOld2:
        var hGate = sessionUpload(be, gateRaw)
        var hUp = sessionUpload(be, upRaw)
        let hSig = sActivation(be, "sigmoid", hGate, nInter)
        let hSilu = sVecOp(be, "mul", hGate, hSig, nInter)
        let hGated = sVecOp(be, "mul", hSilu, hUp, nInter)
        usedSessionOld2 = sessionEnd(be)
        if usedSessionOld2:
          gated = sessionRead(be, hGated, nInter)
        var hSigV = hSig; var hSiluV = hSilu; var hGatedV = hGated
        sessionFree(be, hGate); sessionFree(be, hUp)
        sessionFree(be, hSigV); sessionFree(be, hSiluV); sessionFree(be, hGatedV)
      if not usedSessionOld2:
        let sig = beSigmoid(ctx, gateRaw)
        let silu = beMul(ctx, gateRaw, sig)
        gated = beMul(ctx, silu, upRaw)
      ffOut = linearFastBias(ff.downProj, gated, rows, ctx)
    elif ff.downProj.bias.data.len == ff.downProj.outF:
      for r in 0 ..< rows:
        let off = r * ff.downProj.outF
        for o in 0 ..< ff.downProj.outF: ffOut[off + o] += ff.downProj.bias.data[o]

  result = newTensor(x.shape)
  result.data = beAdd(ctx, x2.data, ffOut)   # residual-add cuối

var gAttnTime*: float = 0.0
var gFfnTime*: float = 0.0

proc forward*(blk: TransformerBlockLlama, x: Tensor, rope: RoPEL, ctx: Backend,
              cache: var KVCacheL, layerIdx: int, isPrefill: bool): Tensor =
  ## RMSNorm(input) nằm BÊN TRONG forwardPrefill/forwardDecode (gộp với QKV+
  ## RoPE - xem fusedNormQKVRoPE), và residual+RMSNorm(post-attn)+FFN+residual
  ## nằm trong fusedResidualNormFFN - cả block giờ chỉ còn 2 session GPU thay
  ## vì 4+ session rời rạc theo từng submodule như trước.
  # ĐO: tách thời gian attn (QKV+RoPE+attention+oProj) và FFN riêng, cộng dồn
  # qua mọi layer/token - để biết chính xác thời gian rơi vào đâu thay vì đoán.
  let t0 = epochTime()
  let attnOut =
    if isPrefill: forwardPrefill(blk.attn, blk.inputNorm, x, rope, ctx, cache, layerIdx)
    else: forwardDecode(blk.attn, blk.inputNorm, x, rope, ctx, cache, layerIdx)
  let t1 = epochTime()
  result = fusedResidualNormFFN(blk.postAttnNorm, blk.ff, x, attnOut, ctx)
  let t2 = epochTime()
  gAttnTime += (t1 - t0)
  gFfnTime += (t2 - t1)

# ===================================================================
# LlamaModelL — model đầy đủ
# ===================================================================
type
  LlamaLConfig* = object
    vocabSize*, hiddenSize*, intermediateSize*: int
    nLayers*, nHeads*, nKVHeads*: int
    maxPositionEmbeddings*: int
    rmsNormEps*, ropeTheta*: float32
    ropeScalingType*: string        # THÊM: "" (không scale) / "linear" / "dynamic"
    ropeScalingFactor*: float32     # THÊM: mặc định 1.0 nếu model không có rope_scaling

  LlamaModelL* = object
    config*: LlamaLConfig
    embed*: Embedding
    blocks*: seq[TransformerBlockLlama]
    finalNorm*: RMSNormL
    lmHead*: Linear
    rope*: RoPEL

proc newLlamaModelL*(cfg: LlamaLConfig): LlamaModelL =
  result.config = cfg
  result.embed = newEmbedding(cfg.vocabSize, cfg.hiddenSize)
  result.blocks = newSeq[TransformerBlockLlama](cfg.nLayers)
  for i in 0 ..< cfg.nLayers:
    result.blocks[i] = newTransformerBlockLlama(cfg.hiddenSize, cfg.nHeads, cfg.nKVHeads,
                                                  cfg.intermediateSize, cfg.rmsNormEps)
  result.finalNorm = newRMSNormL(cfg.hiddenSize, cfg.rmsNormEps)
  result.lmHead = newLinear(cfg.hiddenSize, cfg.vocabSize)
  result.rope = newRoPEL(cfg.hiddenSize div cfg.nHeads, cfg.maxPositionEmbeddings, cfg.ropeTheta,
                          cfg.ropeScalingType, cfg.ropeScalingFactor)

proc forwardStep*(m: LlamaModelL, ids: seq[int], ctx: Backend, cache: var KVCacheL,
                   isPrefill: bool): Tensor =
  ## isPrefill=true: ids = toàn bộ prompt (T token). isPrefill=false: ids = đúng 1 token mới.
  var x = m.embed.lookupBatch(@[ids], ctx)   # [1, T, C]
  when defined(debugStates):
    let Tdbg = ids.len
    let Cdbg = m.config.hiddenSize
    stderr.writeLine "[debug] after_embedding last_token[:8] = " & $x.data[(Tdbg-1)*Cdbg ..< (Tdbg-1)*Cdbg+8]
  for idx in 0 ..< m.blocks.len:
    x = m.blocks[idx].forward(x, m.rope, ctx, cache, idx, isPrefill)
    # GC: với --gc:arc (ref-counting), GC_fullCollect() là no-op hoàn toàn —
    # bộ nhớ tạm của mỗi layer được giải phóng ngay khi ra khỏi scope, không
    # tích luỹ. SỬA: CHỈ ép dọn khi đang prefill (chạy đúng 1 lần/prompt, T
    # token cùng lúc nên tạm thời có thể phình to) - decode (T=1/bước, chạy
    # LẶP LẠI mỗi token) không còn ép GC nữa. Trước đây gọi vô điều kiện cả
    # lúc decode -> nếu build không dùng --gc:arc, mỗi token phải trả thêm 4
    # lần full stop-the-world scan (mod 8 x 32 layer + layer cuối), một chi
    # phí CPU thuần cộng dồn như nhau trên CPU/Metal/CUDA/OpenCL, không liên
    # quan gì kernel GPU - góp phần giải thích vì sao cả 3 backend GPU cùng
    # chậm bất thường như nhau so với llama.cpp.
    if (isPrefill or existsEnv("BYBY_LEGACY_DECODE_GC")) and
       ((idx + 1) mod 8 == 0 or idx == m.blocks.len - 1):
      GC_fullCollect()
    when defined(debugStates):
      if idx == 0:
        stderr.writeLine "[debug] after_block_0   last_token[:8] = " & $x.data[(Tdbg-1)*Cdbg ..< (Tdbg-1)*Cdbg+8]
  when defined(debugStates):
    stderr.writeLine "[debug] after_last_block last_token[:8] = " & $x.data[(Tdbg-1)*Cdbg ..< (Tdbg-1)*Cdbg+8]
  let normed = m.finalNorm.forward(x, ctx)
  result = m.lmHead.forward(normed, ctx)
  when defined(debugStates):
    let Vdbg = m.config.vocabSize
    let lastRow = result.data[(Tdbg-1)*Vdbg ..< (Tdbg-1)*Vdbg+8]
    stderr.writeLine "[debug] logits last_token[:8] = " & $lastRow
  cache.seqLen += ids.len

# ===================================================================
# Load config + weights từ checkpoint HF/GPTQ (qua nimpy + quant.nim)
# ===================================================================
proc newLlamaLConfig*(modelPath: string): LlamaLConfig =
  let transformers = pyImport("transformers")
  let cfg = transformers.AutoConfig.from_pretrained(modelPath, trust_remote_code = true)
  result.vocabSize = cfg.vocab_size.to(int)
  result.hiddenSize = cfg.hidden_size.to(int)
  result.intermediateSize = cfg.intermediate_size.to(int)
  result.nLayers = cfg.num_hidden_layers.to(int)
  result.nHeads = cfg.num_attention_heads.to(int)
  try:
    result.nKVHeads = cfg.num_key_value_heads.to(int)
  except CatchableError:
    result.nKVHeads = result.nHeads   # model không dùng GQA -> MHA thường
  result.maxPositionEmbeddings = cfg.max_position_embeddings.to(int)
  result.rmsNormEps = (try: cfg.rms_norm_eps.to(float32) except CatchableError: 1e-5'f32)
  result.ropeTheta = (try: cfg.rope_theta.to(float32) except CatchableError: 10000'f32)
  # THÊM: đọc rope_scaling - trước đây bị bỏ hoàn toàn -> model dùng
  # rope_scaling (vd deepseek-coder-6.7B-GPTQ: {'type':'linear','factor':4.0})
  # bị tính RoPE sai ngay từ block 0, không liên quan gì tới quantize.
  result.ropeScalingType = ""
  result.ropeScalingFactor = 1.0'f32
  try:
    let rs = cfg.rope_scaling
    result.ropeScalingType = rs["type"].to(string)
    result.ropeScalingFactor = rs["factor"].to(float32)
  except CatchableError:
    discard

proc loadLlamaWeightsL*(m: var LlamaModelL, path: string) =
  echo &"  đang đọc file {path} ..."
  let fileSize = getFileSize(path)
  echo &"  kích thước file: {fileSize div (1024*1024)} MB"
  let (_, sd) = loadQuantStateDict(path)
  echo &"  đã đọc xong {sd.len} tensor từ file, đang build model..."
  var byName = initTable[string, QuantTensor]()
  for (name, qt) in sd: byName[name] = qt

  proc get(name: string): seq[float32] =
    if not byName.hasKey(name): return @[]
    dequantizeTensor(byName[name])

  proc loadLinear(lin: var Linear, prefix: string) =
    ## SỬA: giữ weight NÉN (setQuantWeight, xem nimformer.nim) thay vì
    ## dequant hết ra fp32 ngay lúc load (get() cũ) - với model 6.7B, dequant
    ## hết tất cả Linear ra fp32 tốn ~27GB RAM (nguyên nhân bị "killed" OOM
    ## lúc "Building model..."). Giữ nén chỉ tốn ~4GB, dequant tạm mỗi lần
    ## forward() rồi bỏ ngay (xem nimformer.nim Linear.forward).
    let wName = prefix & ".weight"
    if byName.hasKey(wName):
      setQuantWeight(lin, byName[wName])
    let b = get(prefix & ".bias")
    if b.len > 0: lin.bias.data = b

  let embW = get("model.embed_tokens.weight")
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

  # lm_head: nếu có trong file thì setQuantWeight (giữ nén); nếu không (tied
  # embeddings) thì dùng thẳng embW đã dequant sẵn (chỉ 1 lần, không phải
  # 224 lần như các Linear khác nên không đáng lo RAM) - Linear.forward tự
  # fallback sang nhánh useQuant=false, transpose bình thường mỗi lần gọi
  # (chấp nhận được vì lm_head chỉ forward 1 LẦN/token, không phải 224 lần).
  if byName.hasKey("lm_head.weight"):
    setQuantWeight(m.lmHead, byName["lm_head.weight"])
  elif embW.len > 0:
    m.lmHead.weight.data = embW
    m.lmHead.useQuant = false

proc uploadAllResidentWeights*(m: var LlamaModelL, ctx: Backend) =
  ## Upload TẤT CẢ ma trận trọng số (int4-per-group, tức đường hot loop thật)
  ## lên GPU MỘT LẦN, ngay sau loadLlamaWeightsL. Từ token đầu tiên trở đi,
  ## linearFast() chỉ upload activation nhỏ mỗi bước, không đụng lại 4.5GB
  ## weight nữa (xem beResidentLinearQ4/sessionMatmulQ4 trong backend.nim).
  echo "  đang upload weight resident lên GPU (1 lần duy nhất) ..."
  for l in 0 ..< m.config.nLayers:
    uploadResident(m.blocks[l].attn.qProj, ctx)
    uploadResident(m.blocks[l].attn.kProj, ctx)
    uploadResident(m.blocks[l].attn.vProj, ctx)
    uploadResident(m.blocks[l].attn.oProj, ctx)
    uploadResident(m.blocks[l].ff.gateProj, ctx)
    uploadResident(m.blocks[l].ff.upProj, ctx)
    uploadResident(m.blocks[l].ff.downProj, ctx)
  uploadResident(m.lmHead, ctx)  # no-op nếu tied-embedding (useQuant=false)
  echo "  xong - weight resident trên GPU, không upload lại mỗi token."


type
  HFTokenizer* = object
    tok: PyObject
    eosId*, vocabSize*: int

proc newHFTokenizer*(modelPath: string): HFTokenizer =
  let transformers = pyImport("transformers")
  result.tok = transformers.AutoTokenizer.from_pretrained(modelPath, trust_remote_code = true)
  result.eosId = (try: result.tok.eos_token_id.to(int) except CatchableError: -1)
  result.vocabSize = result.tok.vocab_size.to(int)

proc encode*(tok: HFTokenizer, text: string): seq[int] =
  tok.tok.encode(text, add_special_tokens = true).to(seq[int])

proc decode*(tok: HFTokenizer, ids: seq[int]): string =
  tok.tok.decode(ids, skip_special_tokens = true).to(string)

proc applyChatTemplate*(tok: HFTokenizer, userText: string, systemText: string = ""): seq[int] =
  ## Dùng ĐÚNG chat template khai báo trong tokenizer_config.json của model
  ## (tokenizer.apply_chat_template) thay vì tự bịa prompt dạng Alpaca
  ## "### Instruction:/### Response:" - model instruct-tuned (Llama/Qwen/...)
  ## được train với special token/định dạng RIÊNG (vd <|start_header_id|>...
  ## hay <|im_start|>...) nên prompt sai định dạng -> model không nhận ra
  ## ranh giới lượt hỏi/đáp, sinh lan man/không bám câu hỏi (đúng triệu chứng
  ## gặp phải: sinh 99 token không ra code).
  ## Trả thẳng token ids (tokenize=true) - KHÔNG qua encode() lần nữa, vì
  ## chat template đã tự chèn sẵn special token (BOS/header...), encode lại
  ## với add_special_tokens=true sẽ chèn BOS 2 lần.
  ## SUA: `%*` là macro của std/json (dựng JsonNode Nim thuần), KHÔNG phải
  ## API của nimpy - nimpy không có `%*`. Build messages bằng std/json (chắc
  ## chắn có sẵn) rồi đẩy qua Python bằng json.loads(), tránh phụ thuộc API
  ## dict/list không rõ ràng của nimpy.
  var msgs = newJArray()
  if systemText.len > 0:
    msgs.add(%*{"role": "system", "content": systemText})
  msgs.add(%*{"role": "user", "content": userText})
  let jsonMod = pyImport("json")
  let messages = jsonMod.loads($msgs)
  result = tok.tok.apply_chat_template(messages, tokenize = true, add_generation_prompt = true).to(seq[int])

proc sampleTopP(ctx: Backend, logits: seq[float32], temperature: float32 = 0.7'f32, topP: float32 = 0.9'f32,
                 recentIds: openArray[int] = [], repeatPenalty: float32 = 1.3'f32,
                 noPenaltyIds: openArray[int] = []): int =
  # SỬA: thêm repetition penalty (kiểu llama.cpp) - không có cái này model
  # dễ rơi vào vòng lặp token kiểu "),),),),)" vì mỗi bước cứ chọn lại đúng
  # token có logit cao nhất từng chọn trước đó. Hạ logit của token đã xuất
  # hiện gần đây (chia cho repeatPenalty nếu dương, nhân nếu âm) trước khi
  # tính softmax/top-p.
  # SỬA LẦN 2 (nguyên nhân thật của việc model tránh sinh \n): trước đây
  # phạt TẤT CẢ token trong recentIds, kể cả \n/space/tab. Với code, \n
  # lặp lại RẤT nhiều lần trong 64 token gần nhất (gần như mỗi dòng) ->
  # logit của \n bị chia dồn cho repeatPenalty liên tục qua nhiều bước ->
  # bị đè xuống thấp hơn hẳn token space -> model buộc phải thay \n bằng
  # space dù vẫn "muốn" xuống dòng đúng cú pháp. Giống hệt lý do llama.cpp
  # có cờ penalize_nl (mặc định KHÔNG phạt \n) - token định dạng thuần
  # (\n, space, tab) không nên bị coi là "lặp lại nội dung" như token
  # thường. Fix: nhận thêm noPenaltyIds (id của \n/space/tab/thụt lề lấy
  # từ tokenizer), bỏ qua các id này khi áp penalty.
  # SỬA: phạt lặp từ giờ chạy qua beRepeatPenalty (GPU thật khi ctx là
  # backend GPU - xem gpuRepeatPenalty trong gpubackend.nim, kernel
  # repeat_penalty_kernel trong vecop_matmul.metal/.cl/vecop_ext_kernel.cu).
  # Đây là bước CUỐI CÙNG còn chạm CPU trong sampleTopP (scatter-write rời
  # rạc theo index, không phải elementwise/reduce/sort đơn giản) - giờ đã có
  # kernel riêng, dedupe id chỉ đụng vài chục số nguyên nhỏ (không phải
  # vector logits/vocab lớn).
  var adjLogits = beRepeatPenalty(ctx, logits, recentIds, repeatPenalty, noPenaltyIds)

  if temperature <= 0'f32:
    # SỬA: dùng beArgMax (GPU thật khi ctx là backend GPU - xem gpuReduce/
    # metalReduce op="argmax" trong gpubackend.nim) thay vì vòng for CPU thuần
    # trước đây. Trên backend cpu, beArgMax tự rơi về đúng cùng thuật toán
    # linear-scan này (không đổi kết quả, chỉ đổi nơi chạy).
    return beArgMax(adjLogits, backend = $ctx.kind)
  # SỬA: max-find và exp giờ chạy qua beMax/beExp (GPU thật khi ctx là
  # backend GPU) thay vì 2 vòng for CPU thuần trước đây - đây là 2 lượt quét
  # toàn bộ vector logits (kích thước = vocab, có thể tới 150k phần tử).
  var probs: seq[float32]
  block:
    let maxArr = beMax(adjLogits, @[adjLogits.len], 0, backend = $ctx.kind)
    let maxL = maxArr[0]
    let shiftedRaw = beSub(adjLogits, newSeqWith(adjLogits.len, maxL), backend = $ctx.kind)
    let shifted = beDiv(shiftedRaw, newSeqWith(adjLogits.len, temperature), backend = $ctx.kind)
    probs = beExp(shifted, backend = $ctx.kind)
  let sumExp = beSum(probs, @[probs.len], 0, backend = $ctx.kind)[0]
  probs = beDiv(probs, newSeqWith(probs.len, sumExp), backend = $ctx.kind)

  # SỬA: sort giảm dần giờ chạy qua beSortDescIdx (bitonic sort GPU thật khi
  # ctx là backend GPU - xem gpuSortDescPadded trong gpubackend.nim, kernel
  # bitonic_sort_kernel trong vecop_matmul.metal/.cl/vecop_ext_kernel.cu).
  # Đây là bước CUỐI CÙNG của sampleTopP từng buộc phải chạm CPU (sort/so
  # sánh cặp không phải elementwise/reduce đơn giản) - giờ đã có kernel riêng.
  let (sortedProbs, idx) = beSortDescIdx(probs, backend = $ctx.kind)
  let cumArr = beCumsum(sortedProbs, @[sortedProbs.len], 0, backend = $ctx.kind)
  var cutoff = idx.len
  for i, c in cumArr:
    if c >= topP:
      cutoff = i + 1
      break
  let cum = cumArr[cutoff - 1]
  var r = rand(1.0'f32) * cum
  var acc = 0'f32
  for i in 0 ..< cutoff:
    acc += probs[idx[i]]
    if acc >= r: return idx[i]
  return idx[cutoff - 1]


## whole_model_session_decode.nim
## ============================================================================
## SỬA (bản 3 - sau khi bản layer-session/1-session-mỗi-layer VẪN lag và
## chậm hơn cả đường forwardDecode cũ ở trên): quay lại đúng yêu cầu ban đầu
## - "session toàn model" (1 sessionBegin/sessionEnd DUY NHẤT cho CẢ 32 layer
## + finalNorm + lm_head, MỖI TOKEN) - nhưng sửa 2 lỗi THẬT đã gây lag/chậm ở
## các bản trước, không phải do bản thân ý tưởng "1 session/token" sai:
##
## 1) LEAK BUFFER (nguyên nhân lag máy tích luỹ qua nhiều token): bản 1-
##    session/token ĐẦU TIÊN gọi sessionUpload/sessionAllocScratch cho MỌI
##    buffer tạm (hGamma, hNormed, hQh/hKh/hVh, hScores, hProbs...) nhưng
##    KHÔNG BAO GIỜ gọi sessionFree - mỗi MTLBuffer là 1 object Objective-C
##    retain bằng __bridge_retained (xem metal_shim.m), sống mãi tới khi
##    process thoát nếu không free tay. Với ~15-20 buffer tạm/layer x 32
##    layer x hàng chục/hàng trăm token sinh ra = hàng chục nghìn buffer
##    RAM/VRAM không giải phóng -> máy lag dần, đúng triệu chứng quan sát
##    được. Bản này free NGAY sau sessionEnd() (khi buffer đã an toàn để
##    release - GPU đã waitUntilCompleted xong).
##
## 2) RE-UPLOAD TOÀN BỘ KV-CACHE MỖI TOKEN (nguyên nhân "chậm hơn cả bản
##    cũ", CÀNG SINH DÀI CÀNG CHẬM): forwardDecode() ở TRÊN (đường "cũ")
##    lưu KV-cache dạng mảng CPU (KVCacheL.layers[].k/v) rồi COPY + UPLOAD
##    LẠI TOÀN BỘ S*D phần tử (S = độ dài đã sinh tới hiện tại) lên GPU MỖI
##    TOKEN MỖI LAYER qua sessionUpload(kBlock)/sessionUpload(vBlock) - đây
##    là O(S) băng thông upload/token, tăng dần tuyến tính theo độ dài đã
##    sinh (token 200 upload nhiều gấp 10 token 20) - đây MỚI là lý do chính
##    khiến sinh càng lâu càng chậm, ĐỘC LẬP với số lượng session/token.
##    Nếu giữ nguyên cách lưu KV-cache này mà chỉ gộp session lại (bản
##    layer-session ở trên), phần O(S) upload KHÔNG hề giảm -> vẫn chậm dần,
##    đúng như quan sát "vẫn chậm hơn cả bản cũ".
##    Fix: KV-cache chuyển hẳn sang buffer GPU THƯỜNG TRÚ (SessionHandle,
##    cấp phát 1 LẦN DUY NHẤT lúc chuẩn bị, kích thước maxSeqLen*headDim),
##    APPEND đúng 1 token/lần bằng sessionCopyInto (ghi D phần tử vào đúng
##    offset pos*D, KHÔNG đụng lại phần đã ghi trước đó) - upload O(1)/token
##    thay vì O(S)/token, và KHÔNG rời GPU nên gộp thẳng vào session toàn
##    model được.
##
## Q/K/V projection của mỗi layer được TÁCH thành N_KV_HEADS resident-weight
## Q4 riêng (thay vì 1 resident lớn rồi cắt) vì sessionCopyInto hiện tại CHỈ
## hỗ trợ offset ĐÍCH, không hỗ trợ offset NGUỒN (xem gpubackend.nim) - tách
## theo dải hàng là cách duy nhất để mỗi KV-head có 1 buffer kết quả matmul
## riêng biệt, không cần cắt buffer chung. KHÔNG đổi giá trị toán học.
##
## !! CHƯA BUILD-TEST TRÊN PHẦN CỨNG THẬT (không có compiler/GPU ở môi
## trường viết code này) !! Trước khi tin kết quả sinh ra:
##   1. Build thử, sinh vài chục token, so token id với đường forwardDecode
##      cũ (BYBY_WHOLE_SESSION=0) trên CÙNG 1 prompt - lệch token là có bug.
##   2. Theo dõi RAM/VRAM qua Activity Monitor trong lúc sinh dài (200+
##      token) - nếu vẫn tăng dần đều, còn leak buffer ở đâu đó chưa free.
## ============================================================================

proc sliceResidentQ4ByRowsW(ctx: Backend, qt: QuantTensor, rowsPerChunk, nChunks: int): seq[ResidentWeightHandle] =
  ## Giống hệt logic tách GPTQ theo dải hàng đã dùng ở uploadResident()/
  ## linearFast() phía trên (xem đầu file) - chỉ khác chỗ TRẢ VỀ N phần
  ## riêng thay vì 1 khối, để mỗi KV-head có ResidentWeightHandle độc lập.
  ## Nếu weight không phải int4-per-group -> trả @[] rỗng, caller PHẢI coi
  ## là "không đi được đường whole-session", fallback forwardStep/forwardDecode cũ.
  result = @[]
  if qt.kind != qkInt4Asymmetric or qt.groupSize <= 0:
    return
  let k = qt.shape[1]
  let bytesPerRow = (k + 1) div 2
  let nGroupsPerRow = (k + qt.groupSize - 1) div qt.groupSize
  for c in 0 ..< nChunks:
    let rowStart = c * rowsPerChunk
    let rowEnd = rowStart + rowsPerChunk
    let byteStart = rowStart * bytesPerRow
    let byteEnd = rowEnd * bytesPerRow
    let sgStart = rowStart * nGroupsPerRow
    let sgEnd = rowEnd * nGroupsPerRow
    let dataSlice = qt.data[byteStart ..< byteEnd]
    let scaleSlice = qt.scale[sgStart ..< sgEnd]
    let zeroSlice = qt.zero_point[sgStart ..< sgEnd]
    let h = beResidentUploadQ4(ctx, dataSlice, scaleSlice, zeroSlice,
                                k, rowsPerChunk, qt.groupSize, nGroupsPerRow)
    result.add(h)

type
  LayerWholeState = object
    ready*: bool
    qHeadW*, kHeadW*, vHeadW*: seq[ResidentWeightHandle]   # len = nKVHeads
    kCacheH*, vCacheH*: seq[SessionHandle]                  # len = nKVHeads, mỗi cái [maxSeqLen*headDim] - THƯỜNG TRÚ

  WholeModelSessionState* = object
    layers*: seq[LayerWholeState]
    maxSeqLen*: int
    ready*: bool

proc prepareWholeModelSession*(m: LlamaModelL, ctx: Backend, maxSeqLen: int): WholeModelSessionState =
  ## Gọi 1 LẦN duy nhất, NGAY SAU uploadAllResidentWeights(). Cấp KV-cache
  ## GPU thường trú (kCacheH/vCacheH) - SỐNG SUỐT vòng đời generate(), khác
  ## hẳn KVCacheL (CPU-array) mà forwardDecode() cũ dùng - 2 đường KHÔNG
  ## dùng chung bộ nhớ, chỉ dùng chung `cache.seqLen` để biết vị trí hiện tại.
  result.maxSeqLen = maxSeqLen
  result.ready = true
  result.layers = newSeq[LayerWholeState](m.blocks.len)
  for li in 0 ..< m.blocks.len:
    let attn = m.blocks[li].attn
    let H = attn.nHeads
    let HKV = attn.nKVHeads
    let D = attn.headDim
    let nRep = H div HKV
    var st: LayerWholeState
    st.qHeadW = sliceResidentQ4ByRowsW(ctx, attn.qProj.weightQ, nRep * D, HKV)
    st.kHeadW = sliceResidentQ4ByRowsW(ctx, attn.kProj.weightQ, D, HKV)
    st.vHeadW = sliceResidentQ4ByRowsW(ctx, attn.vProj.weightQ, D, HKV)
    let ffReadyQ4 = m.blocks[li].ff.gateProj.residentReady and m.blocks[li].ff.upProj.residentReady and
                    m.blocks[li].ff.downProj.residentReady
    let oProjReadyQ4 = attn.oProj.residentReady
    st.ready = st.qHeadW.len == HKV and st.kHeadW.len == HKV and st.vHeadW.len == HKV and
               ffReadyQ4 and oProjReadyQ4
    if not st.ready:
      result.ready = false
      result.layers[li] = st
      continue
    let be = ctx.kind.toByby()
    discard sessionBegin(be)
    st.kCacheH = newSeq[SessionHandle](HKV)
    st.vCacheH = newSeq[SessionHandle](HKV)
    let zeros = newSeq[float32](maxSeqLen * D)
    for hkv in 0 ..< HKV:
      st.kCacheH[hkv] = sessionUpload(be, zeros)
      st.vCacheH[hkv] = sessionUpload(be, zeros)
    discard sessionEnd(be)
    result.layers[li] = st

proc forwardStepDecodeWholeSession*(m: LlamaModelL, tokenId: int, ctx: Backend,
                                     cache: var KVCacheL, whole: WholeModelSessionState): Tensor =
  ## Forward 1 token qua TOÀN BỘ model trong ĐÚNG 1 sessionBegin/sessionEnd.
  ## Trả Tensor(shape=@[]) nếu bất kỳ bước nào fail -> caller PHẢI fallback
  ## `m.forwardStep(@[tokenId], ctx, cache, isPrefill=false)`.
  if not whole.ready:
    return Tensor(shape: @[])
  let C = m.config.hiddenSize
  let be = ctx.kind.toByby()
  let pos = cache.seqLen

  if not sessionBegin(be):
    return Tensor(shape: @[])

  # TẤT CẢ buffer tạm của CẢ 32 layer (không gồm resident weight/KV-cache
  # thường trú) được gom vào đây, free ĐÚNG 1 LẦN ngay sau sessionEnd() -
  # đây là điểm sửa lỗi leak chính so với bản 1-session/token đầu tiên.
  var freeList: seq[SessionHandle] = @[]
  template failWhole(): untyped =
    discard sessionEnd(be)
    for fh in freeList.mitems: sessionFree(be, fh)
    return Tensor(shape: @[])

  var xData = newSeq[float32](C)
  block:
    let embOff = tokenId * C
    for c in 0 ..< C: xData[c] = m.embed.weight.data[embOff + c]
  var hX = sessionUpload(be, xData)
  freeList.add(hX)

  for li in 0 ..< m.blocks.len:
    let blk = m.blocks[li]
    let attn = blk.attn
    let lst = whole.layers[li]
    let H = attn.nHeads
    let HKV = attn.nKVHeads
    let D = attn.headDim
    let nRep = H div HKV
    let scale = attn.scale

    # ---- RMSNorm(inputNorm) ----
    var hGamma1 = sessionUpload(be, blk.inputNorm.weight.data)
    freeList.add(hGamma1)
    var hNormed: SessionHandle
    if not sessionRmsnormEnc(be, hX, hGamma1, hNormed, 1, C, blk.inputNorm.eps): failWhole()
    freeList.add(hNormed)

    # ---- Q/K/V projection, per-KV-head, resident Q4 riêng từng phần ----
    var hQh = newSeq[SessionHandle](HKV)
    var hKh = newSeq[SessionHandle](HKV)
    var hVh = newSeq[SessionHandle](HKV)
    for hkv in 0 ..< HKV:
      if not sessionMatmulQ4(be, hNormed, lst.qHeadW[hkv], hQh[hkv], 1): failWhole()
      if not sessionMatmulQ4(be, hNormed, lst.kHeadW[hkv], hKh[hkv], 1): failWhole()
      if not sessionMatmulQ4(be, hNormed, lst.vHeadW[hkv], hVh[hkv], 1): failWhole()
      freeList.add(hQh[hkv]); freeList.add(hKh[hkv]); freeList.add(hVh[hkv])

    # ---- RoPE(Q head-group), RoPE(K head) ----
    let half = D div 2
    var cosSlice = newSeq[float32](half)
    var sinSlice = newSeq[float32](half)
    for i in 0 ..< half:
      cosSlice[i] = m.rope.cosT[pos * half + i]
      sinSlice[i] = m.rope.sinT[pos * half + i]
    # Q phải nhân attn.scale TRƯỚC Q@K^T (giống forwardDecode cũ). RoPE
    # tuyến tính: rotate(Q)*s == rotate_với(cos*s, sin*s)(Q) - gộp scale
    # thẳng vào cos/sin CHỈ dùng cho Q (KHÔNG áp cho K).
    var cosSliceQ = newSeq[float32](half)
    var sinSliceQ = newSeq[float32](half)
    for i in 0 ..< half:
      cosSliceQ[i] = cosSlice[i] * scale
      sinSliceQ[i] = sinSlice[i] * scale
    var hCos = sessionUpload(be, cosSlice)
    var hSin = sessionUpload(be, sinSlice)
    var hCosQ = sessionUpload(be, cosSliceQ)
    var hSinQ = sessionUpload(be, sinSliceQ)
    freeList.add(hCos); freeList.add(hSin); freeList.add(hCosQ); freeList.add(hSinQ)
    for hkv in 0 ..< HKV:
      # hQh[hkv] chứa nRep head liên tiếp (mỗi head D phần tử) -> coi mỗi
      # "sub-head" như 1 token độc lập cho sessionRope: T=nRep, Hn=1, D=D.
      if not sessionRope(be, hQh[hkv], hCosQ, hSinQ, nRep, 1, D): failWhole()
      if not sessionRope(be, hKh[hkv], hCos, hSin, 1, 1, D): failWhole()

    # ---- Append K/V mới vào GPU-resident cache (dst-offset only, O(1)/token
    #      - KHÔNG upload lại lịch sử, khác hẳn kBlocks/vBlocks re-upload
    #      O(S) của forwardDecode cũ ở trên). KV-cache là buffer THƯỜNG TRÚ
    #      (không nằm trong freeList) - sống sang token SAU. ----
    for hkv in 0 ..< HKV:
      var dstK = lst.kCacheH[hkv]
      var dstV = lst.vCacheH[hkv]
      if not sessionCopyInto(be, hKh[hkv], dstK, pos * D, D): failWhole()
      if not sessionCopyInto(be, hVh[hkv], dstV, pos * D, D): failWhole()

    # ---- Attention: mỗi KV-head, Q gồm nRep head con, K/V = toàn bộ cache
    #      đến vị trí pos (đã gồm token mới) ----
    let S = pos + 1
    var hOutHeads = sessionAllocScratch(be, H * D)
    freeList.add(hOutHeads)
    for hkv in 0 ..< HKV:
      var hScores: SessionHandle
      # sessionMatmulNT: C(MxN)=A(MxK)@B(NxK)^T. a=hQh[hkv] (nRep x D),
      # b=kCacheH[hkv] (đọc N=S hàng ĐẦU của buffer maxSeqLen*D) -> C=nRep x S.
      # GIẢ ĐỊNH kernel NT đọc đúng S*D phần tử ĐẦU của buffer lớn hơn (token
      # mới nhất luôn ở cuối vùng đã dùng, layout liên tục theo token) -
      # **CẦN KIỂM CHỨNG TRÊN PHẦN CỨNG THẬT**, giống hệt rủi ro đã ghi ở
      # bản layer-session trước đó.
      if not sessionMatmulNT(be, hQh[hkv], lst.kCacheH[hkv], hScores, nRep, D, S): failWhole()
      freeList.add(hScores)
      var hProbs: SessionHandle
      if not sessionSoftmax(be, hScores, hProbs, nRep, S): failWhole()
      freeList.add(hProbs)
      var hOutGroup: SessionHandle
      if not sessionMatmul(be, hProbs, lst.vCacheH[hkv], hOutGroup, nRep, S, D): failWhole()
      freeList.add(hOutGroup)
      var dstOut = hOutHeads
      if not sessionCopyInto(be, hOutGroup, dstOut, hkv * nRep * D, nRep * D): failWhole()

    # ---- oProj ----
    var hAttnProj: SessionHandle
    if not sessionMatmulQ4(be, hOutHeads, attn.oProj.residentW, hAttnProj, 1): failWhole()
    freeList.add(hAttnProj)

    # ---- residual #1 ----
    var hX2: SessionHandle
    if not sessionVecOp(be, "add", hX, hAttnProj, hX2, C): failWhole()
    freeList.add(hX2)

    # ---- RMSNorm(postAttnNorm) ----
    var hGamma2 = sessionUpload(be, blk.postAttnNorm.weight.data)
    freeList.add(hGamma2)
    var hNormed2: SessionHandle
    if not sessionRmsnormEnc(be, hX2, hGamma2, hNormed2, 1, C, blk.postAttnNorm.eps): failWhole()
    freeList.add(hNormed2)

    # ---- FFN (SwiGLU) ----
    var hGate, hUp: SessionHandle
    if not sessionMatmulQ4(be, hNormed2, blk.ff.gateProj.residentW, hGate, 1): failWhole()
    if not sessionMatmulQ4(be, hNormed2, blk.ff.upProj.residentW, hUp, 1): failWhole()
    freeList.add(hGate); freeList.add(hUp)
    let interF = blk.ff.gateProj.outF
    var hSig: SessionHandle
    if not sessionActivation(be, "sigmoid", hGate, hSig, interF): failWhole()
    freeList.add(hSig)
    var hSilu: SessionHandle
    if not sessionVecOp(be, "mul", hGate, hSig, hSilu, interF): failWhole()
    freeList.add(hSilu)
    var hGated: SessionHandle
    if not sessionVecOp(be, "mul", hSilu, hUp, hGated, interF): failWhole()
    freeList.add(hGated)
    var hFfOut: SessionHandle
    if not sessionMatmulQ4(be, hGated, blk.ff.downProj.residentW, hFfOut, 1): failWhole()
    freeList.add(hFfOut)

    # ---- residual #2 -> hX cho layer kế tiếp (VẪN Ở TRÊN GPU, không đọc
    #      về CPU - đây là điểm khác biệt chính so với bản layer-session:
    #      KHÔNG có round-trip CPU giữa các layer nữa). ----
    var hXNext: SessionHandle
    if not sessionVecOp(be, "add", hX2, hFfOut, hXNext, C): failWhole()
    freeList.add(hXNext)
    hX = hXNext

  # ---- RMSNorm cuối + lm_head (VẪN TRONG CÙNG session đang mở) ----
  var hGammaF = sessionUpload(be, m.finalNorm.weight.data)
  freeList.add(hGammaF)
  var hFinal: SessionHandle
  if not sessionRmsnormEnc(be, hX, hGammaF, hFinal, 1, C, m.finalNorm.eps): failWhole()
  freeList.add(hFinal)

  var hLogits: SessionHandle
  let vocab = m.config.vocabSize
  if m.lmHead.residentReady:
    if not sessionMatmulQ4(be, hFinal, m.lmHead.residentW, hLogits, 1): failWhole()
  elif m.lmHead.residentFp32Ready:
    if not sessionMatmul(be, hFinal, m.lmHead.residentFp32, hLogits, 1, C, vocab): failWhole()
  else:
    failWhole()   # lm_head chưa resident dạng nào -> không đi được đường whole-session
  freeList.add(hLogits)

  if not sessionEnd(be):
    for fh in freeList.mitems: sessionFree(be, fh)
    return Tensor(shape: @[])

  let logitsData = sessionRead(be, hLogits, vocab)
  for fh in freeList.mitems: sessionFree(be, fh)
  if logitsData.len != vocab:
    return Tensor(shape: @[])
  cache.seqLen += 1
  result = Tensor(data: logitsData, shape: @[1, 1, vocab])

proc forwardStepDecodeSafeWhole*(m: LlamaModelL, tokenId: int, ctx: Backend,
                                  cache: var KVCacheL, whole: WholeModelSessionState): Tensor =
  ## Gọi hàm này thay cho forwardStep() trong vòng lặp decode. Tự fallback
  ## đường cũ (forwardStep/forwardDecode, 100% đã xác nhận đúng trước đây)
  ## nếu whole-session fail hoặc bị tắt qua BYBY_WHOLE_SESSION=0.
  ## CẢNH BÁO: nếu whole-session fail GIỮA CHỪNG 1 lần sinh (không phải fail
  ## ngay từ đầu do resident chưa sẵn sàng), fallback về forwardStep() cho
  ## bước đó sẽ ĐỌC SAI KV-cache (forwardStep dùng KVCacheL.layers CPU, chưa
  ## được ghi bởi các bước đã chạy qua đường whole-session GPU-resident) -
  ## đây là rủi ro CỐ HỮU của thiết kế 2 đường KV-cache song song, giống hệt
  ## rủi ro đã có ở bản layer-session trước đó. Trong thực tế `whole.ready`
  ## được quyết định 1 LẦN lúc chuẩn bị (residency tĩnh, không đổi theo
  ## token) nên nhánh fallback giữa chừng gần như không xảy ra, nhưng vẫn
  ## PHẢI biết rủi ro này nếu thấy log "whole_session fail" xuất hiện.
  let forceOff = existsEnv("BYBY_WHOLE_SESSION") and getEnv("BYBY_WHOLE_SESSION") == "0"
  if whole.ready and not forceOff:
    let r = forwardStepDecodeWholeSession(m, tokenId, ctx, cache, whole)
    if r.shape.len > 0:
      return r
    stderr.writeLine "[whole_session] session toan model fail/khong ho tro -> fallback forwardStep cu cho buoc nay"
  result = m.forwardStep(@[tokenId], ctx, cache, isPrefill = false)


proc generate*(m: LlamaModelL, tok: HFTokenizer, ctx: Backend, promptIds: seq[int],
               maxNewTokens: int = 128, temperature: float32 = 0.7'f32, topP: float32 = 0.9'f32): string =
  # SỬA: nhận thẳng promptIds (đã tokenize qua applyChatTemplate) thay vì
  # nhận chuỗi prompt rồi tự encode() lại - encode() dùng add_special_tokens
  # (BOS thô), KHÔNG áp dụng chat template thật của model -> model instruct
  # không nhận ra định dạng hội thoại, sinh lan man.
  # SỬA: TRƯỚC ĐÂY dùng thẳng m.config.maxPositionEmbeddings (=16384 với
  # model này) làm kích thước KV-cache -> cấp phát ~16GB RAM (32 layer x 32
  # KV-head x 16384 vị trí x 128 dim x 2(k+v) x 4 byte) NGAY LÚC BẮT ĐẦU,
  # bất kể prompt/maxNewTokens thực tế ngắn cỡ nào - nguyên nhân OOM "killed"
  # lúc Generating. Giờ chỉ cấp đúng độ dài thật sự cần dùng (prompt +
  # token sẽ sinh ra), dư thêm chút cho an toàn.
  let neededSeqLen = promptIds.len + maxNewTokens + 8
  var cache = newKVCacheL(m.config.nLayers, m.config.nKVHeads,
                           m.config.hiddenSize div m.config.nHeads,
                           neededSeqLen)
  # THÊM: chuẩn bị 1 LẦN đường session-toàn-model (1 sessionBegin/End cho cả
  # 32 layer + finalNorm + lm_head MỖI TOKEN, KV-cache GPU-resident append
  # O(1)/token - xem comment ở forwardStepDecodeWholeSession phía trên). Nếu
  # model không đi được đường này (chưa đủ resident Q4 ở đâu đó) ->
  # whole.ready=false, decode loop bên dưới tự fallback 100% về
  # m.forwardStep() cũ, KHÔNG đổi hành vi/kết quả.
  let whole = prepareWholeModelSession(m, ctx, neededSeqLen)
  if whole.ready:
    stderr.writeLine "[whole_session] session toan model SAN SANG (thi nghiem, chua kiem chung phan cung - xem canh bao dau file)"
  else:
    stderr.writeLine "[whole_session] session toan model KHONG du dieu kien (thieu resident Q4 o dau do) -> dung duong cu"

  stdout.write(tok.decode(promptIds))

  # ---- ĐO TOK/S: tách riêng prefill (xử lý cả prompt 1 lần, số liệu
  # "prompt tokens/s") và decode (sinh từng token 1, số liệu "tok/s" thật sự
  # người dùng quan tâm khi chat) - giống cách llama.cpp báo cáo
  # "prompt eval time" / "eval time" riêng biệt, vì 2 giai đoạn có chi phí
  # rất khác nhau (prefill tận dụng song song theo chiều T, decode thì không).
  let prefillStart = epochTime()

  # ---- PREFILL: cả prompt trong 1 lần forward (beAttentionFused, causal nội bộ) ----
  # SỬA: xây danh sách id token whitespace/định dạng THẬT theo đúng
  # tokenizer đang dùng (không đoán số cứng vì mỗi tokenizer map khác
  # nhau) - encode từng mẫu whitespace hay gặp trong code, gom id lại,
  # dùng để loại trừ khỏi repetition penalty (xem sampleTopP ở trên).
  var noPenaltyIds: seq[int] = @[]
  for ws in ["\n", "\n\n", " ", "  ", "    ", "\t", ":\n", ":\n    "]:
    for id in tok.tok.encode(ws, add_special_tokens = false).to(seq[int]):
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
  var decodeStart = epochTime()   # tính từ SAU prefill, không tính token đầu tiên (đã sinh trong prefill)

  # ---- DECODE: từng token 1, O(S) mỗi bước qua KV-cache paged ----
  # SỬA: KHÔNG decode từng token id riêng lẻ (tok.decode(@[nextId])) — với
  # BPE/byte-level tokenizer, 1 ký tự multi-byte (UTF-8) hoặc 1 từ thường bị
  # tách thành NHIỀU token id; decode rời từng id làm tokenizer không ghép
  # lại được byte-pair/UTF-8 đúng, ra ký tự rác kiểu "模/同/临" và mất
  # khoảng trắng giữa từ. Fix: decode lại TOÀN BỘ allIds mỗi bước, in phần
  # delta (text mới) so với lần in trước - đúng cách streaming chuẩn.
  var printedText = tok.decode(promptIds)   # phần đã in ra (prompt) rồi
  block:
    let fullText0 = tok.decode(allIds)      # allIds = promptIds + nextId đầu tiên
    if fullText0.len > printedText.len:
      stdout.write(fullText0[printedText.len ..< fullText0.len])
      stdout.flushFile()
    printedText = fullText0

  while generated < maxNewTokens:
    if nextId == tok.eosId: break

    let stepStart = epochTime()
    let stepLogits = forwardStepDecodeSafeWhole(m, nextId, ctx, cache, whole)
    nextId = sampleTopP(ctx, stepLogits.data, temperature, topP, allIds[max(0, allIds.len - 64) ..< allIds.len],
                         noPenaltyIds = noPenaltyIds)
    let stepTime = epochTime() - stepStart
    allIds.add(nextId)
    inc generated

    # Ghim tok/s cố định 1 chỗ (dòng 1, góc trên terminal) bằng ANSI cursor
    # save/restore - không đụng vị trí đang in chữ generate ở dưới.
    let instTokS = (if stepTime > 0: 1.0 / stepTime else: 0.0)
    stdout.write("\e7\e[1;1H\e[K")
    stdout.write(&"[{generated}/{maxNewTokens} tok | {instTokS:.2f} tok/s | {(stepTime*1000):.0f} ms/tok]")
    stdout.write("\e8")
    stdout.flushFile()

    if nextId == tok.eosId: break
    # SỬA (loại bỏ O(T^2)): KHÔNG decode lại allIds (toàn bộ lịch sử) mỗi
    # bước - trước đây mỗi token mới lại tokenizer.decode() TOÀN BỘ chuỗi
    # đã sinh, chi phí O(T) mỗi bước x T bước = O(T^2) tổng, độc lập với
    # GPU/backend (đây là CPU/Python HF work). Với generate dài, cái này tự
    # nó có thể vượt xa chi phí forward pass GPU.
    # Fix: chỉ decode 1 cửa sổ đuôi nhỏ (đủ ghép lại byte-pair/UTF-8 đa byte
    # bị cắt ở token cuối) thay vì toàn bộ allIds - giống streaming
    # incremental của llama.cpp (chỉ detokenize token mới + vài token ngữ
    # cảnh liền trước để xử lý biên UTF-8, không phải cả chuỗi).
    const tailWindow = 8
    let winStart = max(0, allIds.len - tailWindow - 1)
    let prevWinText = tok.decode(allIds[winStart ..< allIds.len - 1])
    let fullWinText = tok.decode(allIds[winStart ..< allIds.len])
    if fullWinText.len > prevWinText.len and fullWinText[0 ..< prevWinText.len] == prevWinText:
      printedText.add(fullWinText[prevWinText.len ..< fullWinText.len])
      stdout.write(fullWinText[prevWinText.len ..< fullWinText.len])
      stdout.flushFile()
    else:
      # Hiếm: biên token làm đổi cách decode phần trong cửa sổ -> rơi về
      # decode toàn bộ CHỈ lần này để đảm bảo đúng, không đánh đổi đúng đắn
      # lấy tốc độ.
      let fullText = tok.decode(allIds)
      if fullText.len > printedText.len:
        stdout.write(fullText[printedText.len ..< fullText.len])
        stdout.flushFile()
      printedText = fullText







  let decodeTime = epochTime() - decodeStart
  let decodedCount = generated - 1   # không tính token đầu (sinh trong lúc prefill)
  stdout.write("\n")
  stderr.write("\n== Thống kê tốc độ ==\n")
  stderr.write(&"Decode - attn (QKV+RoPE+attention+oProj): {gAttnTime:.3f}s | FFN: {gFfnTime:.3f}s | " &
               &"tổng 2 phần: {(gAttnTime+gFfnTime):.3f}s\n")
  stderr.write(&"Prefill: {promptIds.len} token trong {prefillTime:.3f}s -> {(promptIds.len.float/prefillTime):.2f} tok/s\n")
  if decodedCount > 0 and decodeTime > 0:
    stderr.write(&"Decode:  {decodedCount} token trong {decodeTime:.3f}s -> {(decodedCount.float/decodeTime):.2f} tok/s ({(decodeTime*1000/decodedCount.float):.1f} ms/token)\n")
  stderr.write(&"Tổng: {(promptIds.len+decodedCount)} token trong {(prefillTime+decodeTime):.3f}s\n")
  result = tok.decode(allIds)

when isMainModule:
  # SỬA: Nim mặc định (khi CHƯA gọi setControlCHook) tự bắt SIGINT và in
  # thẳng "SIGINT: Interrupted by Ctrl-C." ra stderr rồi quit - KHÔNG có \n
  # trước nó. Lúc người dùng nhấn Ctrl-C giữa chừng generate, stdout đang
  # có chữ in dở (stream token không có \n ở cuối mỗi chunk) -> message đó
  # bị dính thẳng vào cuối chữ đang in, nhìn như bị "đè dòng". Đây KHÔNG
  # phải lỗi giống bug \r/status-line trước đó (đã sửa) - là 2 chuyện khác
  # nhau: cái trước là TA tự đè lên nhau, cái này là RUNTIME NIM tự in
  # không xuống dòng. Fix: tự đăng ký hook riêng - luôn xuống dòng + flush
  # stdout xong mới in thông báo ngắt, để không bao giờ dính chữ.
  proc onCtrlC() {.noconv.} =
    stdout.write("\n")
    stdout.flushFile()
    stderr.write("\n[Đã dừng - Ctrl+C]\n")
    stderr.flushFile()
    quit(130)  # quy ước exit code chuẩn shell cho lệnh bị SIGINT (128+2)
  setControlCHook(onCtrlC)

  setForbidCpuFallback(true)
  let modelPath = "TheBloke/deepseek-coder-6.7B-instruct-GPTQ"
  echo "Loading config..."
  let cfg = newLlamaLConfig(modelPath)
  echo "Loading tokenizer..."
  let tok = newHFTokenizer(modelPath)
  echo "Init backend..."
  let ctx = newBackend("metal")
  echo "Building model..."
  var model = newLlamaModelL(cfg)
  echo "Loading weights..."
  loadLlamaWeightsL(model, "model.nimq")
  uploadAllResidentWeights(model, ctx)
  # SỬA: gọi 1 lần matmul "mồi" (kích thước nhỏ, kết quả bỏ đi) TRƯỚC khi
  # bắt đầu generate. Lần gọi matmul ĐẦU TIÊN mới là lúc Metal compile
  # pipeline + tự đo maxTotalThreadsPerThreadgroup + NSLog dòng "chon tile
  # ..." (xem metal_shim.m/pickMatmulPipeline) - nếu để nó xảy ra tự nhiên
  # thì đúng lúc token đầu tiên đang stream ra, NSLog ghi thẳng vào
  # Console/syslog xen giữa chữ đang in dở. Gọi trước ở đây để toàn bộ log
  # setup này in ra TRONG lúc "Building model...", không đụng vào phần
  # generate nữa.
  echo "Khoi dong GPU pipeline (mot lan)..."
  discard beMatmul(ctx, @[0'f32, 0, 0, 0], 2, 2, @[0'f32, 0, 0, 0], 2, 2)
  echo "Generating (prefill + paged KV-cache decode, 100% GPU qua BybyLang)..."
  # SỬA: prompt Alpaca "### Instruction:/### Response:" cũ KHÔNG khớp chat
  # template thật của model -> dùng applyChatTemplate (tokenizer_config.json
  # của chính model) để token hoá đúng định dạng nó được train.
  let promptIds = tok.applyChatTemplate(
    "Write a Python function to return the n-th Fibonacci number.",
    systemText = "You are an AI programming assistant.")
  discard generate(model, tok, ctx, promptIds, maxNewTokens = 100, temperature = 0.7'f32, topP = 0.9'f32)