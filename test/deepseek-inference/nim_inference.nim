## generate.nim
## File DUY NHẤT: nimformer_llama (RoPE + RMSNorm + SwiGLU + GQA + KV-cache paged)
## + tokenizer + vòng lặp sinh văn bản, gộp chung 1 file theo yêu cầu.

import std/[math, random, sequtils, tables, algorithm, os, strformat, strutils, times, json, sets, heapqueue]
import nimformers  # Tensor, Linear, newLinear, flatten2D, transpose, Embedding, addT...
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
var gDebugPathOn = getEnv("BYBY_DEBUG_PATH", "0") != "0"
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
  ## OpenCL/Metal/CUDA có, CPU thuần chưa) - caller PHẢI tự fallback về
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

  if result.len == 0:
    # Chưa resident, hoặc backend hiện tại (vd cpu) không hỗ trợ đường
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
    residentGamma*: SessionHandle   # TOI UU: gamma la hang so co dinh
      # (khong doi giua cac token/layer) - upload 1 LAN duy nhat luc load
      # model (xem uploadResidentNorm ben duoi), thay vi sessionUpload()
      # lai TU DAU moi lan goi forward (moi layer, moi token) nhu truoc -
      # dung y het pattern residentReady da ap dung cho weight Q4/fp32.
    residentGammaReady*: bool

proc newRMSNormL*(dim: int, eps: float32 = 1e-6'f32): RMSNormL =
  result.dim = dim
  result.eps = eps
  result.weight = newTensor(@[dim], 1'f32)

proc uploadResidentNorm*(rn: var RMSNormL, ctx: Backend) =
  ## Upload gamma (rn.weight) len GPU 1 LAN DUY NHAT - goi trong
  ## uploadResident() cua model luc load (giong QKV/FFN resident), KHONG
  ## goi trong vong lap decode. Dung chung beResidentUploadFp32 (da fix
  ## bug thieu nhanh gbVulkan o luot truoc) thay vi viet ham moi - giam
  ## rui ro dua bug moi vao.
  rn.residentGamma = beResidentUploadFp32(ctx, rn.weight.data)
  rn.residentGammaReady = true

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
    # MỚI THÊM: buffer K/V resident trên GPU, 1 handle/KV-head, cấp phát 1
    # LẦN DUY NHẤT (maxSeqLen*D) lúc bắt đầu, sau đó CHỈ append phần mới
    # (nTok*D) mỗi lần gọi appendKV bằng sessionCopyInto - THAY vì trước đây
    # forwardDecode phải sessionUpload() TOÀN BỘ lịch sử S*D mỗi token
    # (tăng dần theo S -> tổng chi phí O(S^2)/lần sinh hết chuỗi). Đây là
    # nguyên nhân chính khiến ms/token tăng dần theo độ dài hội thoại quan
    # sát được trong log thực tế, KHÔNG liên quan gì kernel attention.
    kResident*, vResident*: seq[SessionHandle]
    residentReady*: bool     # true sau khi cấp phát xong (lazy, lần gọi đầu)
    kvHeap*: pointer          # THÊM MỚI: heap dùng để suballocate kResident/
      ## vResident (xem ensureKvResident) - giữ lại để free lúc model unload
      ## (hiện chưa có chỗ gọi unload rõ ràng trong codebase này, heap sẽ
      ## sống tới khi process thoát - CHẤP NHẬN ĐƯỢC vì model chỉ load 1 lần/
      ## process, không phải load/unload lặp lại nhiều lần trong 1 phiên chạy).
    kResidentCombined*, vResidentCombined*: SessionHandle
      ## THÊM MỚI: bản GỘP của TOÀN BỘ nKVHeads head vào ĐÚNG 1 buffer duy
      ## nhất/layer (layout [head, seqPos, D] liên tục), dùng RIÊNG cho kernel
      ## fused mới (attention_decode_all_heads) - lý do cần thêm bản gộp này
      ## thay vì tái dùng kResident/vResident (32 buffer rời từ 1 MTLHeap):
      ## Metal KHÔNG đảm bảo thứ tự/khoảng cách giữa các buffer suballocate
      ## từ cùng 1 heap (allocator có thể đặt bất kỳ đâu, có padding align
      ## khác nhau) - nên KHÔNG thể suy ra offset head kế tiếp = offset head
      ## trước + maxSeqLen*D một cách an toàn nếu chỉ dùng kResident[0] làm
      ## "gốc". Buffer gộp này được cấp phát RIÊNG, 1 lần, kích thước biết
      ## trước (nKVHeads*maxSeqLen*D), nên offset head h LUÔN LÀ
      ## h*maxSeqLen*D - tính được chắc chắn, an toàn cho kernel fused đọc
      ## nhiều head trong 1 lần dispatch. Mode 1 (forwardDecode cũ) KHÔNG
      ## đụng tới field này, vẫn dùng kResident/vResident như cũ - hoàn toàn
      ## không ảnh hưởng đường an toàn dự phòng. BẤT KỲ nơi nào ghi K/V mới
      ## (cả Mode 1 lẫn Mode 2/3) đều PHẢI ghi vào CẢ 2 nơi (buffer cũ per-
      ## head VÀ buffer gộp này) để đảm bảo dữ liệu nhất quán dù token sau
      ## rơi vào mode khác token trước.

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
    result.layers[l].residentReady = false
    result.layers[l].kResident = @[]
    result.layers[l].vResident = @[]
    result.layers[l].kvHeap = nil

var gKvAllocCount = 0   ## THÊM: đếm TỔNG số lần ensureKvResident thực sự cấp
  ## phát (residentReady=false lúc gọi) xuyên suốt phiên chạy - dùng để xác
  ## minh chính xác có đúng 32 lần (1/layer) hay không, thay vì chỉ nhìn số
  ## dòng log rồi đếm tay. Nếu chạy xong mà gKvAllocCount > nLayers thật ->
  ## chắc chắn còn leak, không phải cảm giác.

proc ensureKvResident(cache: var LayerKVCache, backend: GpuBackend, nKVHeads, maxSeqLen, D: int) =
  ## MỚI THÊM: cấp phát buffer resident 1 lần duy nhất/layer (lazy, gọi ở lần
  ## appendKV đầu tiên của layer đó). Không upload dữ liệu gì ở đây - chỉ
  ## cấp chỗ trống trên GPU (giống sessionAllocScratch dùng ở chỗ khác trong
  ## codebase) - dữ liệu thật được ghi dần bằng sessionCopyInto bên dưới.
  if cache.residentReady: return
  inc gKvAllocCount
  stderr.writeLine "[KV_RESIDENT_ALLOC] lan thu " & $gKvAllocCount &
    " (n=" & $nKVHeads &
    " head, moi head " & $(maxSeqLen * D * 4 * 2) & " byte) - neu dong nay in ra QUA 32 LAN (so layer) thi co ro ri that"
  cache.kResident = newSeq[SessionHandle](nKVHeads)
  cache.vResident = newSeq[SessionHandle](nKVHeads)
  # SỬA (bug hiệu năng THẬT tại khởi động, đo thực tế bằng
  # BYBY_DECODE_PROFILE + đối chiếu timestamp NSLog Metal): TRƯỚC ĐÂY vòng
  # for dưới đây gọi sessionAllocScratch (=1 lượt [gDevice
  # newBufferWithLength:] ĐẦY ĐỦ, cấp phát device-level riêng) đúng
  # nKVHeads*2 lần MỖI layer -> với nKVHeads=32, đó là 32*2*32(layer)=2048
  # lượt device-alloc RIÊNG LẺ, TẤT CẢ xảy ra trong lúc prefill (TRƯỚC token
  # đầu tiên). Trên driver Intel Iris cũ, mỗi lượt newBufferWithLength có
  # phí cố định đáng kể (map trang nhớ shared CPU/GPU) - ĐỘC LẬP với kích
  # thước buffer - nên 2048 lượt cộng dồn thành hàng chục giây chờ trước khi
  # sinh token đầu tiên, đúng như quan sát thực tế "khởi tạo cực kỳ chậm".
  # FIX: cấp 1 vùng lớn DUY NHẤT bằng MTLHeap (sessionAllocHeap - 1 lượt
  # device-alloc/layer thay vì 64 lượt), rồi suballocate rẻ từng buffer
  # K/V-head từ heap đó (sessionAllocFromHeap - đúng khuyến nghị Apple Metal
  # Best Practices Guide mục "Heaps" cho case "nhiều buffer cùng cỡ, cùng
  # vòng đời"). Backend không hỗ trợ MTLHeap (CUDA/Vulkan hiện tại, hoặc
  # Metal build không phải macOS) -> sessionAllocHeap trả nil -> fallback
  # NGUYÊN VẸN về sessionAllocScratch từng cái như code cũ, KHÔNG đổi hành
  # vi/kết quả, chỉ mất phần tăng tốc khởi động này trên các backend đó.
  let heapElemsNeeded = nKVHeads * 2 * maxSeqLen * D   # K + V, mọi head, layer này
  cache.kvHeap = sessionAllocHeap(backend, heapElemsNeeded)
  if cache.kvHeap != nil:
    for h in 0 ..< nKVHeads:
      cache.kResident[h] = sessionAllocFromHeap(backend, cache.kvHeap, maxSeqLen * D)
      cache.vResident[h] = sessionAllocFromHeap(backend, cache.kvHeap, maxSeqLen * D)
      if cache.kResident[h].metalHandle == nil or cache.vResident[h].metalHandle == nil:
        # Suballocate thất bại giữa chừng (hiếm - vd tính sai heapElemsNeeded,
        # hoặc driver không đủ contiguous dù đã reserve đủ tổng byte) -> KHÔNG
        # âm thầm dùng buffer rỗng, rơi hẳn về đường cấp lẻ an toàn cho MỌI
        # head (kể cả head đã suballocate thành công phía trên, để tránh
        # trộn lẫn 2 nguồn cấp phát khác nhau trong cùng 1 layer).
        stderr.writeLine "[KV_RESIDENT_ALLOC] suballocate tu heap that bai giua chung (head " & $h &
          ") - roi ve cap le tung buffer (cham hon nhung an toan)"
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
  # THÊM MỚI: cấp buffer GỘP riêng (xem giải thích đầy đủ ở khai báo field
  # kResidentCombined/vResidentCombined phía trên) - 1 lần cấp phát lớn/K,
  # 1 lần/V, offset head h LUÔN LÀ h*maxSeqLen*D (khác hẳn kResident/vResident
  # ở trên vốn suballocate từ heap, không đảm bảo khoảng cách đều giữa các head).
  cache.kResidentCombined = sessionAllocScratch(backend, nKVHeads * maxSeqLen * D)
  cache.vResidentCombined = sessionAllocScratch(backend, nKVHeads * maxSeqLen * D)
  cache.residentReady = true

var gKvResidentDisabled = false   ## MỚI THÊM: nếu bất kỳ sessionCopyInto nào
  ## trong appendKV thất bại (trả false), TẮT VĨNH VIỄN resident KV cho cả
  ## phiên chạy - fallback về upload đầy đủ mỗi token (chậm nhưng ĐÚNG).
  ## Khai báo TRƯỚC appendKV (không phải sau như bản trước) vì appendKV cần
  ## set cờ này ngay khi phát hiện copy thất bại, không chỉ verify mới set.

proc appendKVCpuOnly(cache: var KVCacheL, layer: int, kNew, vNew: seq[float32], nTok: int) =
  ## MỚI THÊM: tách riêng phần ghi CPU (luôn chạy, không phụ thuộc GPU) khỏi
  ## phần đồng bộ GPU - để forwardDecode có thể tự kiểm soát LÚC NÀO encode
  ## GPU (gộp vào session chính, không mở session riêng - xem lý do ở
  ## syncKvResidentEncode bên dưới).
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

var gKvResidentWarnedNonMetal = false   ## THÊM: chỉ in cảnh báo "khong phai
  ## Metal" đúng 1 lần cho cả phiên chạy (tránh spam log mỗi token).

proc syncKvResidentEncode(cache: var KVCacheL, layer: int, kNew, vNew: seq[float32], nTok: int,
                           backend: GpuBackend) =
  # SỬA (CÔNG TẮC AN TOÀN KHẨN CẤP): đã xác nhận rò rỉ thật - buffer resident
  # bị cấp phát lại NHIỀU LẦN thay vì đúng 1 lần/layer (log thực tế: in ra
  # >32 lần dù chỉ có 32 layer), mỗi lần cấp lại làm mất luôn ~1.6MB (32 head
  # x 204800 byte x 2 k/v) KHÔNG BAO GIỜ giải phóng buffer cũ - cộng dồn suốt
  # phiên chạy ra tới 58GB RAM. CHƯA xác định được chính xác tại sao
  # residentReady không giữ được trạng thái giữa các lần gọi (nghi ngờ liên
  # quan tương tác giữa appendKV tự mở session riêng lúc prefill và trạng
  # thái session toàn cục gSessionEncoder, nhưng chưa chứng minh được).
  # SỬA (BẬT MẶC ĐỊNH): đã xác nhận qua log thật (gKvAllocCount đúng 32/32,
  # không leak) và đo tốc độ tăng thật (0.47->0.73-0.82 tok/s decode) sau
  # khi vá xong bug double-session (kvSynced) - không còn lý do gì để tắt
  # mặc định nữa. Giữ escape-hatch NGƯỢC LẠI (BYBY_DISABLE_KV_RESIDENT=1) để
  # tắt nếu sau này gặp máy/model khác bị vấn đề, KHÔNG cần phải nhớ bật
  # bằng tay mỗi lần chạy như trước.
  # SUA (theo yeu cau): DOI MAC DINH thanh TAT (tuong duong luon chay nhu
  # dang set BYBY_DISABLE_KV_RESIDENT=1) - toan bo qua trinh do dac/toi uu
  # kernel Q4 (NDST 8->2 + pairing, xem vecop_matmul.metal) trong phien lam
  # viec nay DEU chay voi bien nay = 1 (nguoi dung luon go tay truoc moi
  # lan chay), nen day moi la cau hinh THAT SU da duoc kiem chung/do dac,
  # khong phai duong resident (chua do lai tu sau khi doi kernel Q4).
  # Van giu duong resident (sessionKvResident) trong code - CHUA XOA - vi
  # comment lich su ngay ben tren (dong 613-618) tung do duoc 0.47->0.73-
  # 0.82 tok/s luc CHUA doi kernel Q4, co the van co ich tren may/model
  # khac. Muon BAT LAI de so sanh, set tuong minh BYBY_DISABLE_KV_RESIDENT=0
  # (khac voi truoc day: truoc day CHI can bien ton tai la tat, khong phan
  # biet gia tri "0" - gio "0" tuong minh se BAT LAI, moi gia tri khac
  # (hoac khong set) deu la TAT, dung y "mac dinh tat" nguoi dung yeu cau).
  # SUA (dao nguoc mac dinh): BAT KV-resident sync mac dinh - day la nhanh
  # DUY NHAT cho phep appendKV khong roi CPU giua chung, dieu kien BAT BUOC
  # de gop toan bo layer (attn+oProj+FFN) vao 1 command buffer duy nhat
  # giong ggml-metal. Escape-hatch nguoc lai: BYBY_DISABLE_KV_RESIDENT=1 de
  # TAT neu can so sanh/debug.
  if getEnv("BYBY_DISABLE_KV_RESIDENT", "0") == "1": return
  ## MỚI THÊM (tách khỏi appendKV): encode các lệnh đồng bộ resident KV vào
  ## session ĐANG MỞ SẴN của caller - KHÔNG tự sessionBegin/sessionEnd nữa.
  ##
  ## LÝ DO: bản trước tự mở 1 session RIÊNG cho việc này mỗi token, tách
  ## biệt với session chính của attention - đo thực tế trên Iris Plus 640,
  ## phí cố định mở/đóng session riêng này (sessionBegin+sessionEnd = ép GPU
  ## sync 1 lần) TỐN HƠN cả lợi ích tiết kiệm được từ việc không re-upload
  ## toàn bộ KV mỗi token (ở độ dài hội thoại ngắn ~100 token, chưa đủ dài
  ## để lợi ích O(S)->O(1) upload thắng phí session thêm). Giờ: gộp thẳng
  ## vào session chính forwardDecode đã mở, giảm từ 2 session/token xuống 1.
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
      # THÊM MỚI: ghi ĐỒNG THỜI vào buffer GỘP (kResidentCombined/vResidentCombined)
      # - offset = h*maxSeqLen*D (gốc của head h trong buffer gộp) + vị trí
      # hiện tại. BẮT BUỘC ghi cả 2 nơi (buffer cũ per-head VÀ buffer gộp)
      # để kernel fused mới (đọc từ buffer gộp) và Mode 1 cũ (đọc từ buffer
      # per-head) luôn thấy dữ liệu nhất quán, bất kể token trước/sau chạy
      # mode nào - xem giải thích đầy đủ ở khai báo field kResidentCombined.
      if not sessionCopyInto(backend, hKAll, cache.layers[layer].kResidentCombined,
                              h * D, h * cache.maxSeqLen * D + cache.seqLen * D, D):
        allOk = false
      if not sessionCopyInto(backend, hVAll, cache.layers[layer].vResidentCombined,
                              h * D, h * cache.maxSeqLen * D + cache.seqLen * D, D):
        allOk = false
    sessionFree(backend, hKAll)
    sessionFree(backend, hVAll)
    if not allOk:
      stderr.writeLine "[KV_RESIDENT] sessionCopyInto (fast-path nTok=1) that bai - TAT resident KV, fallback"
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
      # THÊM MỚI: ghi thêm vào buffer gộp (offset gốc head h = h*maxSeqLen*D)
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
      # THÊM MỚI: ghi thêm vào buffer gộp
      if not sessionCopyInto(backend, hChunk, cache.layers[layer].vResidentCombined,
                              h * cache.maxSeqLen * D + cache.seqLen * D, nTok * D):
        allOkPrefill = false
      sessionFree(backend, hChunk)
    if not allOkPrefill:
      stderr.writeLine "[KV_RESIDENT] sessionCopyInto (prefill nTok>1) that bai - TAT resident KV, fallback"
      gKvResidentDisabled = true

proc appendKV*(cache: var KVCacheL, layer: int, kNew, vNew: seq[float32], nTok: int,
               backend: GpuBackend = gbAuto) =
  ## GIỮ NGUYÊN hành vi cũ (tự mở session riêng) - dùng cho forwardPrefill
  ## (không phải hot path, không cần gộp session). forwardDecode giờ dùng
  ## appendKVCpuOnly + syncKvResidentEncode riêng (xem 2 hàm trên) để gộp
  ## vào session chính, không gọi appendKV này nữa.
  appendKVCpuOnly(cache, layer, kNew, vNew, nTok)
  if backend != gbAuto:
    if sessionBegin(backend):
      syncKvResidentEncode(cache, layer, kNew, vNew, nTok, backend)
      discard sessionEnd(backend)

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

  # SỬA (lazy rmsNormPrep): TRƯỚC ĐÂY rmsNormPrep() chạy VÔ ĐIỀU KIỆN trước
  # mọi session — tức là mỗi layer mỗi token đều phải: beMul (GPU round-trip
  # để tính x^2), beMatmul-reduce (GPU round-trip để tính sumSq), rồi đọc
  # sumSq về CPU để tính sqrt. Đây là 2 lần GPU flush bắt buộc ngay cả khi
  # sessionRmsnormEnc thành công và invBroadcast/wBroadcast KHÔNG BAO GIỜ
  # được dùng. Giờ: chỉ tính khi thực sự cần (sessionRmsnormEnc fail hoặc
  # không có qkvResident). Với Metal đường nhanh, 2 GPU flush này biến mất
  # hoàn toàn (0 thay vì 2 mỗi lần gọi). Với CPU fallback, hành vi
  # giống hệt trước (gọi đúng lúc cần, không sớm hơn).
  var invBroadcast: seq[float32]
  var wBroadcast: seq[float32]
  proc ensureRmsNormFallback() =
    if invBroadcast.len > 0: return   # đã tính rồi, không tính lại
    let (inv, w, _) = rmsNormPrep(norm, x, ctx)
    invBroadcast = inv
    wBroadcast = w

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
  # CAM LEGACY: khong con doc BYBY_FORCE_LEGACY_QKV nua - neu resident chua
  # san sang thi RAISE thang, khong am tham roi ve duong cham (legacy van la
  # GPU nhung tao nhieu session nho, khong phai la fallback CPU, nhung nguoi
  # dung yeu cau CAM luon duong nay).
  if not (qkvResidentQ4 or qkvResidentFp32):
    raise newException(CatchableError,
      "[QKV] resident chua san sang (uploadResident() chua chay hoac fail cho " &
      "qProj/kProj/vProj) - legacy fallback dang bi CAM theo yeu cau, khong chay tiep.")
  let qkvResident = true
  debugPathOnce("qkv.resident_ready", qkvResidentQ4 or qkvResidentFp32,
    "Q4=" & $qkvResidentQ4 & " Fp32=" & $qkvResidentFp32 &
    " (false ca 2 => uploadResident() chua chay hoac chay that bai cho qProj/kProj/vProj)")
  var usedSession = qkvResident and sessionBegin(be)
  if not usedSession:
    raise newException(CatchableError,
      "[QKV] sessionBegin() that bai (GPU khong nhan session) - legacy fallback dang bi CAM.")
  # SỬA (gộp RoPE vào CÙNG session với QKV, xem sessionRope trong
  # gpubackend.nim + rope_kernel trong vecop_matmul.metal): trước đây RoPE
  # BẮT BUỘC phải là 1 session riêng ("Session #2" bên dưới) vì session API
  # cũ không có kernel gather/rotate_half. Giờ nếu sessionRope() chạy được
  # (hiện chỉ Metal có cài đặt thật - CUDA/OpenCL trả về false có chủ đích),
  # ta xoay Q/K NGAY TRONG session QKV, TRƯỚC sessionEnd - Q/K đọc về CPU ở
  # dưới đã là kết quả RoPE xong, không cần Session #2 nữa. Nếu sessionRope
  # trả về false (chưa hỗ trợ / lỗi), ropeFused vẫn là false -> code cũ bên
  # dưới (Session #2 / đường CPU) chạy y hệt như trước, không đổi hành vi.
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
        # TOI UU: dung gamma resident (upload 1 lan luc load model) thay vi
        # sessionUpload() lai moi layer/token - fallback ve cach cu neu vi
        # ly do nao do uploadResidentNorm() chua chay (an toan, khong doi
        # hanh vi trong truong hop do).
      var hNormedTry = sessionAllocScratch(be, n)
      if sessionRmsnormEnc(be, hX, hGamma, hNormedTry, rows, inF, norm.eps):
        hNormed = hNormedTry
        normedFast = true
      else:
        sessionFree(be, hNormedTry)   # tránh leak scratch buffer khi thất bại
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
    if tryFastRmsnorm and not norm.residentGammaReady: sessionFree(be, hGamma)
      # SUA: KHONG free hGamma neu no la buffer RESIDENT (norm.residentGamma) -
      # buffer do phai song xuyen suot ca qua trinh inference, khong phai
      # chi 1 lan goi ham nay. Truoc day luon free vo dieu kien vi hGamma
      # luon la buffer MOI upload - gio hGamma co the la resident nen phai
      # kiem tra truoc khi free.
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
    ensureRmsNormFallback()
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
  appendKV(cache, layerIdx, kData, vData, T, backend = ctx.kind.toByby())

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
var gAttnDecodeGqaVerifyLeft = 5   ## MỚI THÊM: đếm số lần còn lại sẽ verify
  ## kernel attention_decode_gqa mới so với đường cũ (matmul+softmax+matmul)
  ## khi BYBY_VERIFY_ATTN_DECODE được bật - chỉ verify vài lần đầu (đủ để
  ## phát hiện sai lệch hệ thống) tránh làm chậm suốt cả phiên chạy.

var gAttnDecodeGqaDisabled = true  ## SỬA: TẮT MẶC ĐỊNH - đã xác nhận kernel
  ## attention_decode_gqa sinh KẾT QUẢ SAI THẬT (không phải false-positive
  ## của verify mid-session cũ) - bằng chứng: sau khi bỏ verify/fallback bị
  ## lỗi (đọc/free giữa session gây SIGSEGV), chạy thật ra token rác toàn
  ## "!" lặp lại (logits hỏng/hằng số) + chậm bất thường (24s/token). Trước
  ## đây verify tuy đo sai kỹ thuật (đọc giữa session ra rác) nhưng lại
  ## VÔ TÌNH luôn fallback đúng về chain cũ - đó là lý do never thấy garbage
  ## token từ path này trước giờ. Giữ TẮT cho tới khi kernel này được sửa
  ## và verify lại bằng 1 session riêng biệt (không lồng trong session
  ## caller) - xem hướng dẫn viết test riêng đã đề cập. Bật lại bằng
  ## BYBY_TRY_ATTN_DECODE_GQA=1 CHỈ khi đã tự xác nhận kernel đúng.
var gAttnDecodeGqaWarned = false  ## THÊM: chỉ in cảnh báo "da tung sinh sai"
  ## đúng 1 lần khi BYBY_TRY_ATTN_DECODE_GQA=1 được set (tránh spam log).

proc tryAttentionDecodeGqa(backend: GpuBackend, hQ, hK, hV: SessionHandle,
                            qGroup, kBlock, vBlock: seq[float32],
                            nRep, S, D: int, allHandles: var seq[SessionHandle]):
                            tuple[ok: bool, outGroup: SessionHandle] =
  ## MỚI THÊM: thử dùng sessionAttentionDecodeGqaEnc (1 lệnh encode duy nhất
  ## thay 3 lệnh matmulNT+softmax+matmul). Trả về ok=false nếu backend chưa
  ## hỗ trợ HOẶC verify phát hiện sai lệch lớn - người gọi PHẢI tự fallback
  ## về chain cũ khi ok=false, KHÔNG được coi đây là lỗi bỏ qua im lặng.
  if gAttnDecodeGqaDisabled and not existsEnv("BYBY_TRY_ATTN_DECODE_GQA"):
    return (false, default(SessionHandle))
  if not gAttnDecodeGqaWarned:
    stderr.writeLine "[BYBY_TRY_ATTN_DECODE_GQA] CANH BAO: kernel nay DA TUNG " &
      "xac nhan sinh SAI (token rac toan '!' lap lai) VA CHAM HON (24s/token) " &
      "trong lan test truoc, khong phai chi 'chua verify'. Neu thay token rac " &
      "hoac cham bat thuong, TAT NGAY (bo BYBY_TRY_ATTN_DECODE_GQA khoi lenh chay)."
    gAttnDecodeGqaWarned = true
  var hOut = sessionAllocScratch(backend, nRep * D)
  var hScores = sessionAllocScratch(backend, nRep * S)
  let ok = sessionAttentionDecodeGqaEnc(backend, hQ, hK, hV, hOut, hScores, nRep, S, D, 1.0'f32)
  # SỬA: scale ĐÃ được gộp sẵn vào qGroup ở CPU trước khi upload (xem
  # "gộp luôn scale vào Q" phía trên) - nên truyền scale=1.0 ở đây, KHÔNG
  # nhân lại lần 2 (nếu không sẽ sai kết quả - scale^2 thay vì scale).
  if ok:
    # SỬA (BUG NGHIÊM TRỌNG ĐÃ TÌM RA VÀ SỬA): trước đây verify gọi
    # sessionRead()+sessionFree() NGAY TẠI ĐÂY - nhưng hàm này luôn được
    # gọi TỪ BÊN TRONG session lớn CỦA CALLER (đã sessionBegin từ trước,
    # chưa sessionEnd) - đọc giữa chừng session LUÔN ra rác (đúng như
    # comment của chính vulkanSessionRead/metalSessionRead đã ghi: "CHỈ
    # được gọi SAU sessionEnd") -> verify LUÔN báo sai lệch giả (không
    # phải lỗi kernel thật), rồi sessionFree() ngay giữa session lại HỦY
    # buffer NGAY LẬP TỨC (vkDestroyBuffer/vkFreeMemory) trong khi lệnh
    # dispatch tham chiếu buffer đó VẪN CÒN NẰM TRONG command buffer chưa
    # submit của session ngoài -> khi sessionEnd() thật sự submit, GPU
    # dereference buffer đã bị hủy -> SIGSEGV. Verify kiểu "đọc ngay giữa
    # session" về cơ bản KHÔNG THỂ làm an toàn trong kiến trúc merged-
    # session này - đã bỏ hẳn, dồn hOut/hScores vào allHandles để cleanup
    # đúng lúc (SAU sessionEnd thật, giống mọi buffer khác trong vòng lặp
    # này) thay vì free non lúc còn đang dùng.
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
  ## So dữ liệu THẬT trong buffer resident (đọc về qua sessionRead) với đúng
  ## nội dung đáng lẽ phải có (kBlockRef/vBlockRef - CPU cache, nguồn sự
  ## thật). Đây là phép so sánh MEMCPY, không phải toán học - lẽ ra phải
  ## khớp gần như tuyệt đối (epsilon chỉ để chừa sai số roundtrip GPU rất
  ## nhỏ, không phải sai số cộng dồn floating-point như bên attention).
  # SỬA: verify giờ BẬT MẶC ĐỊNH - chỉ tắt khi CHỦ ĐỘNG đặt BYBY_NO_VERIFY=1.
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
      "  *** SAI - TẮT resident KV, fallback upload đầy đủ mỗi token cho phần còn lại ***"
    gKvResidentDisabled = true
  else:
    stderr.writeLine "[BYBY_VERIFY_KV_RESIDENT] max|gpu-ref| = " & $maxDiff & "  OK"

var gDecodeProfileLeft = (try: parseInt(getEnv("BYBY_DECODE_PROFILE", "0")) except ValueError: 0)
  ## (chỉ layer 0, 2 token đầu = 2*1 lần vì layerIdx==0 chỉ đúng 1 lần/token)
  ## để biết chính xác thời gian rơi vào QKV+RoPE hay attention-loop hay
  ## oProj, thay vì chỉ có tổng "attn" gộp chung như gAttnTime hiện tại.
  ## ĐÂY LÀ ĐO THUẦN TÚY - không đổi bất kỳ phép tính nào.

proc forwardDecode*(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL, ctx: Backend,
                     cache: var KVCacheL, layerIdx: int, ffnAllResident: bool = false):
                     tuple[attnOut: Tensor, sessionOpen: bool, be: GpuBackend,
                           hOut: SessionHandle, pending: seq[SessionHandle]] =
  ## SỬA: THÊM tham số ffnAllResident + đổi kiểu trả về thành tuple - cho
  ## phép GIỮ session GPU mở khi caller (forward*(blk...)) xác nhận FFN
  ## phía sau CŨNG resident đủ điều kiện gộp - tránh 1 round-trip
  ## sessionEnd+sessionRead+sessionUpload thừa giữa oProj và FFN (đúng
  ## hướng ggml-metal: 1 command buffer cho CẢ layer thay vì nhiều cái nhỏ).
  ## Khi sessionOpen=false (mặc định/fallback), dùng result.attnOut y hệt
  ## hành vi cũ - KHÔNG đổi gì. Khi sessionOpen=true, caller BẮT BUỘC phải
  ## tự sessionEnd + free `pending` sau đó (KHÔNG được đọc/free handle nào
  ## giữa chừng - đúng bug đã sửa trước đó).
  let doProfile = layerIdx == 0 and gDecodeProfileLeft > 0
  var tProfile0, tProfile1: float
  if doProfile: tProfile0 = epochTime()
  let C = x.shape[2]
  let H = attn.nHeads
  let HKV = attn.nKVHeads
  let D = attn.headDim
  let nRep = H div HKV
  let S = cache.seqLen + 1   # tổng độ dài sau khi thêm token này

  # GỘP SESSION cross-submodule (giống forwardPrefill ở trên).
  let (qData, kNew, vNew) = fusedNormQKVRoPE(attn, norm, x, rope, ctx, T = 1, posOffset = cache.seqLen)
  if doProfile:
    tProfile1 = epochTime()
    stderr.writeLine "[DECODE_PROFILE] QKV+RoPE(RMSNorm gộp): " & $((tProfile1 - tProfile0) * 1000) & " ms"

  appendKVCpuOnly(cache, layerIdx, kNew, vNew, 1)

  var outHeads = newSeq[float32](H * D)
  let maxS = cache.maxSeqLen

  # Gom dữ liệu từng group trước (CPU, rẻ) để có thể encode hết vào 1 session
  #
  # SỬA (loại bỏ hẳn transpose CPU O(S*D)/token): thêm sessionMatmulNT
  # (C=A@B^T, B giữ nguyên layout gốc [N,K]) cho cả OpenCL/Metal/CUDA - K và
  # V giờ đều chỉ cần 1 copyMem liền mạch từ cache, KHÔNG transpose vật lý
  # nữa. kBlockT chỉ được build (lazy, đúng lúc cần) nếu backend hiện tại
  # không hỗ trợ NT (fallback an toàn, không đổi kết quả).
  # SUA (bottleneck CPU thuc su - tim thay qua doc code): TRUOC DAY vong lap
  # nay LUON copyMem toan bo S*D phan tu (K va V) cho MOI KV-head, MOI token
  # decode, BAT KE resident co dang bat hay khong. Khi resident bat (duong
  # mac dinh - xem "BAT MAC DINH" o syncKvResidentEncode phia tren), hK/hV
  # lay THANG tu cache.layers[layerIdx].kResident/vResident (da san tren
  # GPU) - kBlocks/vBlocks chi con dung cho transposeKBlock (fallback hiem
  # khi backend khong ho tro NT) va verifyKvResidentOnce (toi da 5 lan CA
  # PHIEN - gKvResidentVerifyLeft). Tuc la GAN NHU TOAN BO cong copy nay bi
  # LANG PHI o duong chinh, va vi S tang dan moi token, chi phi nay la O(S)
  # MOI TOKEN -> O(S^2) CA PHIEN - bottleneck am tham phinh to theo do dai
  # hoi thoai du khong gay crash. FIX: chi copy khi THAT SU can (resident
  # chua san sang, HOAC con luot verify).
  let kvResUsableNow = kvResidentUsable(cache, layerIdx)
  let needCpuKvBlocks = (not kvResUsableNow) or gKvResidentVerifyLeft > 0
  var kBlocks = newSeq[seq[float32]](HKV)
  var vBlocks = newSeq[seq[float32]](HKV)
  var qGroups = newSeq[seq[float32]](HKV)
  for hkv in 0 ..< HKV:
    let headBase = hkv * maxS * D
    # K, V: cache layout [nKVHeads, maxSeqLen, headDim] -> với hkv cố định, S
    # token đầu ĐÃ contiguous sẵn -> chỉ cần copyMem, không transpose.
    if needCpuKvBlocks:
      var kBlock = newSeq[float32](S * D)
      copyMem(addr kBlock[0], unsafeAddr cache.layers[layerIdx].k[headBase], S * D * sizeof(float32))
      kBlocks[hkv] = kBlock

      var vBlock = newSeq[float32](S * D)
      copyMem(addr vBlock[0], unsafeAddr cache.layers[layerIdx].v[headBase], S * D * sizeof(float32))
      vBlocks[hkv] = vBlock
    # else: kBlocks[hkv]/vBlocks[hkv] giữ nguyên @[] (rỗng) - AN TOÀN vì
    # nhánh resident-usable không đọc kBlocks/vBlocks để build hK/hV (lấy
    # thẳng từ cache.layers[layerIdx].kResident/vResident), chỉ transposeKBlock
    # (fallback NT hiếm) và verifyKvResidentOnce (đã loại qua needCpuKvBlocks
    # ở trên) mới đụng tới 2 mảng này.

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

  let backend = ctx.kind.toByby()
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
  # CAM LEGACY: bo escape-hatch BYBY_LEGACY_ATTN_OPROJ - luon di duong merged
  # neu resident san sang. Neu day thuc su co bug tren phan cung that thi se
  # crash/raise ro rang thay vi am tham lui ve duong cham.
  let oProjResidentMergedQ4 = attn.oProj.residentReady
  let oProjResidentMergedFp32 = (not oProjResidentMergedQ4) and attn.oProj.residentFp32Ready
  let oProjResidentMerged = oProjResidentMergedQ4 or oProjResidentMergedFp32
  debugPathOnce("attn_oproj.resident_ready", oProjResidentMerged,
    "Q4=" & $oProjResidentMergedQ4 & " Fp32=" & $oProjResidentMergedFp32)
  # SỬA (BUG THẬT ĐÃ TÌM RA): TRƯỚC ĐÂY syncKvResidentEncode() được gọi ở
  # CẢ nhánh merged (dưới) LẪN nhánh fallback "ĐƯỜNG CŨ" phía sau, và khi
  # nhánh merged mở session thành công nhưng oProj matmul/copyInto fail giữa
  # chừng (okAll/okO=false -> oProjOutMerged rỗng), code rơi tiếp xuống
  # nhánh fallback và MỞ SESSION THỨ 2 cho CÙNG 1 token/layer - tức là 2 lần
  # sessionBegin/sessionEnd (= 2 lần ép GPU sync) cho công việc lẽ ra chỉ
  # cần 1, và encode KV resident bị gọi lặp. `kvSynced` đảm bảo
  # syncKvResidentEncode chỉ chạy ĐÚNG 1 LẦN bất kể rơi qua bao nhiêu nhánh.
  var kvSynced = false
  var tProfileGap0: float
  if doProfile:
    tProfileGap0 = epochTime()
    stderr.writeLine "[DECODE_PROFILE] GAP CPU-only giua 2 session (appendKVCpuOnly + gom qGroups, KHONG tinh phi sessionBegin/GPU sync): " &
      $((tProfileGap0 - tProfile1) * 1000) & " ms"
  var usedMergedSession = oProjResidentMerged and sessionBegin(backend)
  var oProjOutMerged: seq[float32]
  if usedMergedSession:
    var allHandles = newSeq[SessionHandle]()
    var okAll = true
    # SỬA (đã tìm đúng nguyên nhân, không né tránh): sync resident KV vẫn
    # gộp vào ĐÂY (chỉ GHI, không đọc lại - an toàn dù session chưa commit,
    # vì GPU chỉ cần THẤY lệnh ghi theo đúng thứ tự trong cùng session, việc
    # đó luôn đúng). Điều KHÔNG an toàn trước đây là gọi sessionRead (verify)
    # cũng ở trong vòng lặp này - đã chuyển verify ra SAU sessionEnd bên dưới.
    if not kvSynced:
      syncKvResidentEncode(cache, layerIdx, kNew, vNew, 1, backend)
      kvSynced = true
    let hOutHeads = sessionAllocScratch(backend, H * D)   # buffer chung [H*D], ghi từng mảnh vào đây
    allHandles.add(hOutHeads)
    for hkv in 0 ..< HKV:
      let hQ = sessionUpload(backend, qGroups[hkv])
      allHandles.add(hQ)
      var hK, hV: SessionHandle
      if kvResidentUsable(cache, layerIdx):
        hK = cache.layers[layerIdx].kResident[hkv]
        hV = cache.layers[layerIdx].vResident[hkv]
        # SỬA: KHÔNG gọi verifyKvResidentOnce ở đây nữa - sessionRead bên
        # trong nó đọc thẳng bộ nhớ buffer, chỉ ĐÚNG khi lệnh ghi phía trên
        # (syncKvResidentEncode) đã thực sự CHẠY XONG trên GPU, tức là sau
        # sessionEnd() commit+wait - không phải giữa chừng session như trước
        # (đó chính là nguyên nhân gây "chua co session dang mo" + đọc rác
        # max|gpu-ref|=1.4066032 lần trước). Verify dời xuống dưới, SAU
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

    # THÊM: nếu FFN phía sau CŨNG đủ điều kiện resident (caller đã kiểm tra
    # trước và truyền ffnAllResident=true), KHÔNG sessionEnd ở đây - giữ
    # session mở, cộng bias GPU (nếu có, thay vòng for CPU), trả handoff cho
    # caller nối trực tiếp vào FFN CÙNG 1 session - giảm 1 round-trip
    # sessionEnd+sessionRead+sessionUpload/layer (hướng ggml-metal: 1 command
    # buffer/layer thay vì nhiều cái nhỏ). BẮT BUỘC: KHÔNG sessionFree bất kỳ
    # handle nào ở đây (kể cả handle không dùng nữa) - session còn command
    # CHƯA submit tham chiếu chúng, free sớm là đúng bug SIGSEGV đã sửa
    # trước đó. Toàn bộ allHandles + hOProjOut(đã cộng bias) dồn vào
    # `pending`, caller phải tự free SAU sessionEnd thật (ở cuối FFN).
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
        stderr.writeLine "[DECODE_PROFILE] Attention-loop + oProj (session GIU MO cho FFN): " &
          $((tProfile2 - tProfile1) * 1000) & " ms"
        dec gDecodeProfileLeft
      return (newTensor(@[1, 1, C]), true, backend, hFinal, allHandles)

    # SỬA (tối ưu): cộng bias TRÊN GPU bằng sVecOp trước khi sessionEnd,
    # giống hệt nhánh ffnAllResident phía trên - trước đây đọc kết quả về
    # CPU rồi mới cộng bias bằng vòng for tuần tự (bottleneck GPU->CPU
    # không cần thiết, C phần tử chạy CPU đơn luồng mỗi lần decode).
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
    # SỬA: verify resident KV chuyển XUỐNG ĐÂY - SAU sessionEnd(backend) đã
    # commit+wait xong, nên sessionRead trong verifyKvResidentOnce giờ đọc
    # đúng dữ liệu thật (không còn đọc rác giữa chừng session như bug lần
    # trước). Chỉ verify head 0 làm đại diện (đủ để bắt lỗi hệ thống, không
    # cần verify hết HKV head mỗi lần - gAttnDecodeGqaVerifyLeft/
    # gKvResidentVerifyLeft đã giới hạn tổng số lần verify toàn phiên chạy).
    if usedMergedSession and kvResidentUsable(cache, layerIdx):
      verifyKvResidentOnce(backend, cache, layerIdx, 0, kBlocks[0], vBlocks[0], S, D)
    if oProjOutMerged.len == 0:
      usedMergedSession = false   # sessionCopyInto/matmul thất bại -> rơi về đường cũ 2-session bên dưới, KHÔNG coi false là đã chạy xong

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
  # ĐƯỜNG CŨ (fallback): sessionCopyInto không khả dụng/thất bại, hoặc
  # BYBY_LEGACY_ATTN_OPROJ=1, hoặc oProj chưa resident - dùng lại đúng logic
  # 2-session đã tự tay xác nhận đúng trước đây (attention-decode session
  # riêng, rồi oProj session riêng, có sessionRead ở giữa).
  # ─────────────────────────────────────────────────────────────────────
  var usedSession = sessionBegin(backend)
  if usedSession:
    var outHandles = newSeq[SessionHandle](HKV)
    var allHandles = newSeq[SessionHandle]()
    # SỬA: gộp sync resident KV vào đây (nhánh này chỉ chạy khi nhánh merged
    # ở trên fail/return sớm) - chỉ GHI, không đọc lại, an toàn (xem giải
    # thích đầy đủ ở nhánh merged phía trên).
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
        # SỬA: verify dời xuống SAU sessionEnd (xem lý do ở nhánh merged trên)
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
    # SỬA: verify SAU sessionEnd - an toàn (xem giải thích ở nhánh merged trên)
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
    # SỬA (tối ưu, đồng bộ với nhánh merged phía trên): cộng bias TRÊN GPU
    # bằng sVecOp trước sessionEnd, thay vì đọc về CPU rồi vòng for tuần tự.
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
  let x2 = addT(ctx, x, attnOut)
  # SỬA (lazy, giống fusedNormQKVRoPE): không tính invBroadcast/wBroadcast
  # trước khi biết sessionRmsnormEnc có thành công hay không - tránh 2 GPU
  # round-trip thừa/layer khi đường nhanh (Metal) chạy tốt.
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

  # SỬA (gộp 2 session FFN thành 1, thử TRƯỚC TIÊN): gate/up-matmul ->
  # (+bias GPU) -> sigmoid -> mul -> mul -> down-matmul -> (+bias GPU) chạy
  # liền 1 mạch, ĐÚNG 1 sessionBegin/sessionEnd cho toàn bộ FFN thay vì 2-3
  # như trước - giữa các bước này KHÔNG có phụ thuộc CPU bắt buộc nào (khác
  # RoPE cần gather), lần đọc CPU cũ chỉ tồn tại để cộng bias bằng vòng for,
  # giờ cộng bias ngay trên GPU (sVecOp "add" + broadcastBias có sẵn).
  # Giảm 1 lượt commit+waitUntilCompleted/layer (driver Metal Intel cũ: đây
  # là nguồn overhead chính, xem phân tích ở generate()).
  # SUA (goc bug THAT SU, xac nhan qua log tren Intel Iris 640: GPU Timeout
  # xay ra NGAY LAYER 1, KHONG phu thuoc heap/KV-resident - da loai tru bang
  # BYBY_DISABLE_KV_RESIDENT=1 va van timeout y het, cung 1 diem):
  # TRUOC DAY toan bo FFN (RMSNorm + gate-matmul + up-matmul + sigmoid + mul +
  # mul + down-matmul + residual) gop vao DUY NHAT 1 command buffer, commit 1
  # lan o sessionEnd cuoi cung. Tren Iris 640 (khong headless - co watchdog
  # GPU cua he dieu hanh, "kIOAccelCommandBufferCallbackErrorTimeout"), khoi
  # luong cong viec don vao 1 buffer nay VUOT NGUONG watchdog cho phep -> GPU
  # bi kill giua chung, KHONG phai loi logic/dung sai kernel.
  # FIX (VAN 100% GPU, KHONG dung CPU fallback - dung "CAM LEGACY"): TACH 1
  # command buffer lon thanh 2 command buffer nho hon:
  #   Session A: RMSNorm + gate-matmul + up-matmul + bias + sigmoid + mul + mul
  #              -> sessionEnd (commit+wait THAT SU tren GPU - watchdog duoc
  #              RESET vi day la command buffer MOI cho lan submit tiep theo)
  #   Session B: down-matmul + bias -> sessionEnd
  # Buffer GPU (hGated) KHONG bi huy giua 2 session - sessionEnd chi
  # commit+wait, CHI sessionFree() moi huy buffer that su (xem gpubackend.nim)
  # - nen Session B dung lai DUNG handle hGated cua Session A, KHONG can doc
  # ve CPU roi upload lai giua chung (khong mat gi ve toc do/dung luong).
  var ffOut: seq[float32]
  let downResidentQ4 = ff.downProj.residentReady
  let downResidentFp32 = (not downResidentQ4) and ff.downProj.residentFp32Ready
  let allResidentQ4 = gateUpResidentQ4 and downResidentQ4
  let allResidentFp32 = gateUpResidentFp32 and downResidentFp32
  if not (allResidentQ4 or allResidentFp32):
    raise newException(CatchableError,
      "[FFN] resident chua san sang (uploadResident() chua chay hoac fail cho " &
      "gateProj/upProj/downProj) - legacy fallback dang bi CAM theo yeu cau.")
  let allResident = true
  debugPathOnce("ffn.resident_ready", allResidentQ4 or allResidentFp32,
    "Q4=" & $allResidentQ4 & " Fp32=" & $allResidentFp32 & " legacyFFN=false(cam)")

  # SUA (GOP LAI thanh 1 session - thu nghiem sau khi matmul_q4_kernel da
  # sua dung goc, xem vecop_matmul.metal): TRUOC DAY tach lam 2 session (A:
  # gate/up/sigmoid/mul/mul, B: down-matmul) vi kernel Q4 cu (threadgroup-
  # memory, sai gia dinh simdgroup) qua cham/co the treo, khien 1 buffer
  # gop het FFN vuot nguong watchdog Iris 640. GIO kernel Q4 da doi sang
  # simd_sum() THAT + dispatch dung threadExecutionWidth THAT (khong con
  # gia dinh sai) - nhanh hon han ban cu. GOP LAI 1 session de bo 1 lan
  # commit+waitUntilCompleted/layer (driver Metal Intel: day la nguon
  # overhead dang ke, x32 layer x moi token). Neu van con Timeout sau khi
  # gop, quay lai ban tach 2 session (xem lich su sua doi ben tren).
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
    if ff.gateProj.bias.data.len == ff.gateProj.outF:
      var hGB = sessionUpload(be, broadcastBias(ff.gateProj.bias, rows, ff.gateProj.outF))
      hGate = sVecOp(be, "add", hGate, hGB, nInter)
      sessionFree(be, hGB)
    if ff.upProj.bias.data.len == ff.upProj.outF:
      var hUB = sessionUpload(be, broadcastBias(ff.upProj.bias, rows, ff.upProj.outF))
      hUp = sVecOp(be, "add", hUp, hUB, nInter)
      sessionFree(be, hUB)
    let hSig = sActivation(be, "sigmoid", hGate, nInter)
    let hSilu = sVecOp(be, "mul", hGate, hSig, nInter)
    var hGated = sVecOp(be, "mul", hSilu, hUp, nInter)
    let okGateUp = okG and okU
    var hOut = sessionAllocScratch(be, rows * ff.downProj.outF)
    let okD = if allResidentQ4: sessionMatmulQ4(be, hGated, ff.downProj.residentW, hOut, rows)
              else: sessionMatmul(be, hGated, ff.downProj.residentFp32, hOut, rows, ff.downProj.inF, ff.downProj.outF)
    if ff.downProj.bias.data.len == ff.downProj.outF:
      var hDB = sessionUpload(be, broadcastBias(ff.downProj.bias, rows, ff.downProj.outF))
      hOut = sVecOp(be, "add", hOut, hDB, rows * ff.downProj.outF)
      sessionFree(be, hDB)
    usedSessionFused = sessionEnd(be)   # <-- COMMIT+WAIT DUY NHAT cho ca FFN
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
    # CAM LEGACY: truoc day cho phep lui ve "DUONG CU" (session rieng le
    # hoac tham chi linearFastBias/beMul co the roi CPU) khi session gop
    # that bai giua chung. Gio KHONG con duoc phep - resident da xac nhan
    # san sang o tren, neu sessionBegin/sessionEnd/matmul that bai giua
    # chung thi la loi GPU that (driver/kernel), phai RAISE de biet ma sua,
    # khong duoc am tham chay cham hoac roi CPU.
    raise newException(CatchableError,
      "[FFN] session gop (fused) that bai GIUA CHUNG dù resident da san sang " &
      "(sessionBegin/sessionEnd/matmul tra ve false) - legacy/CPU fallback " &
      "dang bi CAM theo yeu cau, khong lui ve duong cu.")

  result = newTensor(x.shape)
  result.data = beAdd(ctx, x2.data, ffOut)   # residual-add cuối

var gAttnTime*: float = 0.0
var gFfnTime*: float = 0.0

proc fusedResidualNormFFNContinue(norm: RMSNormL, ff: SwiGLUL, x: Tensor, be: GpuBackend,
                                    hAttnOut: SessionHandle, pending: var seq[SessionHandle]): Tensor =
  ## MỚI THÊM: bản NỐI TIẾP session GPU đang MỞ SẴN từ forwardDecode (xem
  ## ffnAllResident/sessionOpen ở đó) - KHÔNG tự sessionBegin (dùng session
  ## caller đã mở), residual-add làm bằng GPU (sVecOp "add") thay vì CPU
  ## addT, tránh 1 round-trip sessionEnd+sessionRead+sessionUpload giữa
  ## attention và FFN - gộp thành ĐÚNG 1 sessionEnd/layer (thay vì 2), đúng
  ## hướng ggml-metal (1 command buffer/layer). CHỈ được gọi khi caller đã
  ## tự xác nhận allResident đủ điều kiện (FFN CHẮC CHẮN đi được đường
  ## resident, không rơi về fallback - nếu rơi fallback giữa chừng thì
  ## session đang mở sẽ KHÔNG bao giờ được sessionEnd -> lệnh GPU treo vĩnh
  ## viễn, layer sau sessionBegin() sẽ fail vì session cũ chưa đóng - đây là
  ## lý do forward*(blk...) PHẢI kiểm tra allResident TRƯỚC khi gọi
  ## forwardDecode với ffnAllResident=true, không phải đoán).
  let n = x.data.len
  let (rows, hidden) = flatten2D(x.shape)
  doAssert hidden == norm.dim
  let interF = ff.gateProj.outF
  let nInter = rows * interF
  let gateUpResidentQ4 = ff.gateProj.residentReady and ff.upProj.residentReady
  let downResidentQ4 = ff.downProj.residentReady
  let allResidentQ4 = gateUpResidentQ4 and downResidentQ4
  # allResidentFp32 tương tự nhưng ffnAllResident=true CHỈ được set khi 1
  # trong 2 đúng - dùng allResidentQ4 làm nhánh chính, else fp32 (đối xứng
  # với logic gốc trong fusedResidualNormFFN).

  var hX = sessionUpload(be, x.data)
  pending.add(hX)
  let hX2 = sVecOp(be, "add", hX, hAttnOut, n)   # residual-add GPU thay CPU addT
  pending.add(hX2)

  var hGamma = norm.residentGamma
  if not norm.residentGammaReady:
    hGamma = sessionUpload(be, norm.weight.data)
    pending.add(hGamma)   # CHỈ free nếu là buffer MỚI upload (không phải
      # resident) - buffer resident KHÔNG được add vào pending, nếu không
      # sẽ bị sessionFree() giải phóng ngay sau token đầu tiên (xem vòng
      # lặp "for h in pending.mitems: sessionFree" cuối hàm) -> token thứ
      # 2 trở đi dùng buffer đã free -> lỗi/rác (đúng kiểu bug đã gặp
      # trước đây, cẩn thận không lặp lại).
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

  # SUA BUG NGHIEM TRONG (token rac, tim ra boi nguoi dung): ban truoc THIEU
  # HAN phep residual-add cuoi cung - tra thang hOut (chi la ffOut) thay vi
  # x2 + ffOut nhu ham goc fusedResidualNormFFN (`result.data = beAdd(ctx,
  # x2.data, ffOut)`). Day la nguyen nhan token rac THAT SU (khong lien
  # quan gi den bug subgroup matmul_q4 da sua truoc do) - moi buoc decode
  # (dung dung nhanh resident nay) mat residual 1 lan/layer, don qua 32
  # layer -> logits vo nghia tu token thu 2 tro di (token dau tien van
  # dung vi lay tu logits cua PREFILL, dung ham fusedResidualNormFFN gom co
  # residual-add). FIX: cong hX2 (residual truoc FFN) vao hOut (ffOut)
  # bang GPU (sVecOp "add", cung 1 session dang mo, khong them round-trip
  # GPU<->CPU nao) TRUOC KHI sessionEnd.
  let hFinal = sVecOp(be, "add", hX2, hOut, n)
  pending.add(hFinal)

  # ĐÚNG 1 sessionEnd cho CẢ layer (attention + FFN) - đây là điểm mấu chốt
  # của việc gộp: 96 round-trip/token (3 session x 32 layer) giờ còn 64
  # (2 session x 32 layer: 1 cho QKV+RoPE, 1 cho attention+oProj+FFN gộp).
  let sessionOk = sessionEnd(be)
  result = newTensor(x.shape)
  if sessionOk and normedOk and okG and okU and okD:
    result.data = sessionRead(be, hFinal, n)
  else:
    # An toàn: session gộp thất bại giữa chừng - KHÔNG có dữ liệu đúng để
    # trả về. Đây là tình huống KHÔNG nên xảy ra (caller đã check
    # ffnAllResident trước khi vào đường này) - nếu xảy ra, trả residual
    # nguyên trạng (x2 chưa qua FFN) còn hơn segfault/dữ liệu rác, và log
    # cảnh báo để biết mà tắt đường gộp qua BYBY_NO_MERGE_FFN_SESSION=1.
    stderr.writeLine "[fusedResidualNormFFNContinue] session gộp THẤT BẠI giữa chừng - fallback residual passthrough (BẤT THƯỜNG, báo lỗi này lại)"
    result.data = x.data   # residual passthrough, KHÔNG đúng nhưng an toàn hơn rác/crash
  for h in pending.mitems: sessionFree(be, h)
  pending.setLen(0)

## ===================================================================
## MODE 2 (default ON): gộp Session #1 (QKV+RoPE) + Session #2
## (attention+oProj+FFN) thành ĐÚNG 1 command buffer/layer thay vì 2.
## Bật bằng BYBY_MERGE_QKV_ATTN_SESSION=1. KHÔNG set gì (mặc định) = hành
## vi CŨ 100%, không đổi 1 dòng nào trong forward(blk...) hiện tại.
##
## AN TOÀN: proc này raise CatchableError ngay khi BẤT KỲ bước encode nào
## thất bại (không âm thầm trả kết quả sai) - caller (forward*(blk...))
## BẮT BUỘC bắt exception này và rơi về đường forwardDecode/
## fusedResidualNormFFNContinue cũ (MODE 1, đã verify đúng bằng log thật)
## cho ĐÚNG token đó. Vì vậy nếu Mode 2 lỗi/crash-logic ở máy bạn, bạn
## luôn có đường quay lại tự động, không cần tắt cờ mới chạy tiếp được.
let gMergeQkvAttnSession* = getEnv("BYBY_MERGE_QKV_ATTN_SESSION", "1") == "1"
var gMode2WarnLeft = 20   ## chỉ log tối đa 20 lần fallback Mode2->Mode1, tránh spam

proc precheckLayerResident(attn: GQAAttention, ff: SwiGLUL, layerIdx: int,
                            failMsg: var string): bool =
  ## Kiểm tra 1 layer đã resident đủ điều kiện (Q4 HOẶC Fp32, đúng convention
  ## chung toàn bộ codebase - xem comment ở forwardDecodeFullMergeSingleSession
  ## cũ) hay chưa - dùng CHUNG cho cả Mode 2 (check 1 layer) và Mode 3 (check
  ## cả 32 layer TRƯỚC KHI mở session, để không phí công mở session rồi fail
  ## ngay layer đầu).
  let qkvQ4 = attn.qProj.residentReady and attn.kProj.residentReady and attn.vProj.residentReady
  let qkvFp32 = (not qkvQ4) and attn.qProj.residentFp32Ready and attn.kProj.residentFp32Ready and
    attn.vProj.residentFp32Ready
  if not (qkvQ4 or qkvFp32):
    failMsg = "QKV chua resident (Q4 lan Fp32 deu chua san sang, layer " & $layerIdx & ")"
    return false
  let oProjQ4 = attn.oProj.residentReady
  let oProjFp32 = (not oProjQ4) and attn.oProj.residentFp32Ready
  if not (oProjQ4 or oProjFp32):
    failMsg = "oProj chua resident (Q4 lan Fp32 deu chua san sang, layer " & $layerIdx & ")"
    return false
  let gateUpQ4 = ff.gateProj.residentReady and ff.upProj.residentReady
  let gateUpFp32 = (not gateUpQ4) and ff.gateProj.residentFp32Ready and ff.upProj.residentFp32Ready
  let downQ4 = ff.downProj.residentReady
  let downFp32 = (not downQ4) and ff.downProj.residentFp32Ready
  if not ((gateUpQ4 or gateUpFp32) and (downQ4 or downFp32)):
    failMsg = "FFN chua resident (Q4 lan Fp32 deu chua san sang, layer " & $layerIdx & ")"
    return false
  return true

proc encodeDecodeLayerIntoOpenSession(attn: GQAAttention, normQKV, normFFN: RMSNormL,
    ff: SwiGLUL, hXIn: SessionHandle, rope: RoPEL, be: GpuBackend, cache: var KVCacheL,
    layerIdx: int, pending: var seq[SessionHandle]): SessionHandle =
  ## LÕI DÙNG CHUNG cho Mode 2 (forwardDecodeFullMergeSingleSession, 1 session/
  ## layer) VÀ Mode 3 (forwardDecodeAllLayersMergedSession, 1 session/TOÀN BỘ
  ## model). Khác bản gốc (trước khi tách): (1) nhận input là SessionHandle
  ## SẴN CÓ (hXIn) thay vì Tensor - không sessionUpload lại, cho phép layer
  ## trước truyền thẳng output GPU làm input layer sau, KHÔNG đọc CPU giữa
  ## chừng (đây là mấu chốt của Mode 3: cả 32 layer, KHÔNG một lần sessionRead
  ## nào cho tới hết layer cuối); (2) KHÔNG tự sessionBegin/sessionEnd - session
  ## do CALLER mở/đóng (Mode 2: mở/đóng quanh đúng 1 layer; Mode 3: mở/đóng
  ## quanh CẢ VÒNG LẶP 32 layer); (3) trả về SessionHandle (hFinal), không
  ## phải Tensor - caller tự quyết định lúc nào cần sessionRead (Mode 2: ngay
  ## sau layer đó; Mode 3: chỉ đúng 1 lần, sau layer CUỐI CÙNG).
  ## LƯU Ý QUAN TRỌNG (đã bị chính codebase này ghi nhận là nguồn SIGSEGV
  ## trước đây): TUYỆT ĐỐI không sessionFree bất kỳ handle nào (kể cả của
  ## layer trước, kể cả không dùng nữa) khi session còn đang mở - mọi handle
  ## được encode vào 1 session đều phải sống tới khi chính session đó
  ## sessionEnd xong. Vì vậy `pending` ở đây LUÔN LÀ 1 mảng DÙNG CHUNG xuyên
  ## suốt session (Mode 3: dùng chung cho cả 32 layer, không phải 1 mảng/layer).
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
  step(sessionRmsnormEnc(be, hX, hGamma, hNormed, 1, C, normQKV.eps), "RMSNorm input that bai")

  var hQ = sessionAllocScratch(be, H * D); pending.add(hQ)
  var hK = sessionAllocScratch(be, HKV * D); pending.add(hK)
  var hV = sessionAllocScratch(be, HKV * D); pending.add(hV)
  if qkvQ4:
    step(sessionMatmulQ4(be, hNormed, attn.qProj.residentW, hQ, 1), "Q matmul that bai")
    step(sessionMatmulQ4(be, hNormed, attn.kProj.residentW, hK, 1), "K matmul that bai")
    step(sessionMatmulQ4(be, hNormed, attn.vProj.residentW, hV, 1), "V matmul that bai")
  else:
    step(sessionMatmul(be, hNormed, attn.qProj.residentFp32, hQ, 1, attn.qProj.inF, attn.qProj.outF), "Q matmul(fp32) that bai")
    step(sessionMatmul(be, hNormed, attn.kProj.residentFp32, hK, 1, attn.kProj.inF, attn.kProj.outF), "K matmul(fp32) that bai")
    step(sessionMatmul(be, hNormed, attn.vProj.residentFp32, hV, 1, attn.vProj.inF, attn.vProj.outF), "V matmul(fp32) that bai")

  # ---- RoPE(Q), RoPE(K) trong CÙNG session ----
  let halfD = D div 2
  var cosSlice = newSeq[float32](halfD)
  var sinSlice = newSeq[float32](halfD)
  let pos = cache.seqLen
  doAssert pos < rope.maxSeqLen, "RoPE: vi tri vuot qua max_seq_len"
  for i in 0 ..< halfD:
    cosSlice[i] = rope.cosT[pos * halfD + i]
    sinSlice[i] = rope.sinT[pos * halfD + i]
  let hCosT = sessionUpload(be, cosSlice); pending.add(hCosT)
  let hSinT = sessionUpload(be, sinSlice); pending.add(hSinT)
  step(sessionRope(be, hQ, hCosT, hSinT, 1, H, D), "RoPE(Q) that bai/chua ho tro")
  step(sessionRope(be, hK, hCosT, hSinT, 1, HKV, D), "RoPE(K) that bai/chua ho tro")

  # ---- Nhân attn.scale vào Q NGAY TRÊN GPU ----
  step(sessionScale(be, hQ, attn.scale, H * D), "Scale(Q) that bai/chua ho tro")

  # ---- Đồng bộ KV-resident ----
  ensureKvResident(cache.layers[layerIdx], be, HKV, cache.maxSeqLen, D)
  step(kvResidentUsable(cache, layerIdx), "KV resident chua san sang cho layer nay")
  for h in 0 ..< HKV:
    step(sessionCopyInto(be, hK, cache.layers[layerIdx].kResident[h], h * D, cache.seqLen * D, D),
         "copyInto K resident that bai (head " & $h & ")")
    step(sessionCopyInto(be, hV, cache.layers[layerIdx].vResident[h], h * D, cache.seqLen * D, D),
         "copyInto V resident that bai (head " & $h & ")")
    # THÊM MỚI: ghi ĐỒNG THỜI vào buffer GỘP dùng cho kernel fused mới - xem
    # giải thích đầy đủ ở khai báo field kResidentCombined (LayerKVCache).
    # BẮT BUỘC ghi cả 2 nơi để Mode 1 (đọc kResident/vResident) và kernel
    # fused (đọc kResidentCombined/vResidentCombined) luôn thấy dữ liệu nhất
    # quán, bất kể token trước/sau chạy đường nào.
    step(sessionCopyInto(be, hK, cache.layers[layerIdx].kResidentCombined,
                          h * D, h * cache.maxSeqLen * D + cache.seqLen * D, D),
         "copyInto K resident-combined that bai (head " & $h & ")")
    step(sessionCopyInto(be, hV, cache.layers[layerIdx].vResidentCombined,
                          h * D, h * cache.maxSeqLen * D + cache.seqLen * D, D),
         "copyInto V resident-combined that bai (head " & $h & ")")
  cache.layers[layerIdx].filled = cache.seqLen + 1

  # ---- Attention decode: THỬ kernel FUSED trước (1 dispatch cho TOÀN BỘ H
  # head, thay vì vòng lặp HKV*5 lệnh dispatch) - fallback về vòng lặp cũ
  # (matmulNT+softmax+matmul từng head, đã CHẠY ĐÚNG, làm baseline đối chiếu)
  # nếu kernel fused fail vì BẤT KỲ lý do gì (chưa build lại app với kernel
  # mới, backend không phải Metal, v.v.) - giống hệt pattern mọi hàm _enc
  # khác trong file này, KHÔNG có đường nào bị cụt.
  var hOutHeads = sessionAllocScratch(be, H * D); pending.add(hOutHeads)
  let fusedOk = sessionAttentionDecodeAllHeadsEnc(be, hQ, cache.layers[layerIdx].kResidentCombined,
                    cache.layers[layerIdx].vResidentCombined, hOutHeads, H, HKV, D, S, cache.maxSeqLen)
  if not fusedOk:
    for hkv in 0 ..< HKV:
      var hQGroup = sessionAllocScratch(be, nRep * D); pending.add(hQGroup)
      step(sessionCopyInto(be, hQ, hQGroup, hkv * nRep * D, 0, nRep * D),
           "copyInto Q-group that bai (kv-head " & $hkv & ")")
      var hKHead = cache.layers[layerIdx].kResident[hkv]
      var hVHead = cache.layers[layerIdx].vResident[hkv]

      let (hScores0, ntOk) = sMatmulNT(be, hQGroup, nRep, D, hKHead, S)
      pending.add(hScores0)
      step(ntOk, "matmulNT(Q,K) khong ho tro tren backend nay")
      let hScores = hScores0

      let hProbs = sSoftmax(be, hScores, nRep, S); pending.add(hProbs)
      var hOutGroup = sMatmul(be, hProbs, nRep, S, hVHead, D); pending.add(hOutGroup)

      step(sessionCopyInto(be, hOutGroup, hOutHeads, hkv * nRep * D, nRep * D),
           "copyInto outHeads that bai (kv-head " & $hkv & ")")

  # ---- oProj ----
  var hOProjOut = sessionAllocScratch(be, C); pending.add(hOProjOut)
  if oProjQ4:
    step(sessionMatmulQ4(be, hOutHeads, attn.oProj.residentW, hOProjOut, 1), "oProj matmul that bai")
  else:
    step(sessionMatmul(be, hOutHeads, attn.oProj.residentFp32, hOProjOut, 1, attn.oProj.inF, attn.oProj.outF),
         "oProj matmul(fp32) that bai")
  var hOProjFinal = hOProjOut
  if attn.oProj.bias.data.len == C:
    let hBias = sessionUpload(be, attn.oProj.bias.data); pending.add(hBias)
    hOProjFinal = sVecOp(be, "add", hOProjOut, hBias, C); pending.add(hOProjFinal)

  # ---- Residual #1 ----
  let hX2 = sVecOp(be, "add", hX, hOProjFinal, C); pending.add(hX2)

  # ---- RMSNorm (post-attn) + FFN ----
  let hGammaFfn = (if normFFN.residentGammaReady: normFFN.residentGamma
                   else: (let h = sessionUpload(be, normFFN.weight.data); pending.add(h); h))
  var hNormedFfn = sessionAllocScratch(be, C); pending.add(hNormedFfn)
  step(sessionRmsnormEnc(be, hX2, hGammaFfn, hNormedFfn, 1, C, normFFN.eps), "RMSNorm FFN that bai")

  let interF = ff.gateProj.outF
  var hGate = sessionAllocScratch(be, interF); pending.add(hGate)
  var hUp = sessionAllocScratch(be, interF); pending.add(hUp)
  if gateUpQ4:
    step(sessionMatmulQ4(be, hNormedFfn, ff.gateProj.residentW, hGate, 1), "gateProj matmul that bai")
    step(sessionMatmulQ4(be, hNormedFfn, ff.upProj.residentW, hUp, 1), "upProj matmul that bai")
  else:
    step(sessionMatmul(be, hNormedFfn, ff.gateProj.residentFp32, hGate, 1, ff.gateProj.inF, ff.gateProj.outF),
         "gateProj matmul(fp32) that bai")
    step(sessionMatmul(be, hNormedFfn, ff.upProj.residentFp32, hUp, 1, ff.upProj.inF, ff.upProj.outF),
         "upProj matmul(fp32) that bai")
  if ff.gateProj.bias.data.len == interF:
    let hGB = sessionUpload(be, ff.gateProj.bias.data); pending.add(hGB)
    hGate = sVecOp(be, "add", hGate, hGB, interF); pending.add(hGate)
  if ff.upProj.bias.data.len == interF:
    let hUB = sessionUpload(be, ff.upProj.bias.data); pending.add(hUB)
    hUp = sVecOp(be, "add", hUp, hUB, interF); pending.add(hUp)
  let hSig = sActivation(be, "sigmoid", hGate, interF); pending.add(hSig)
  let hSilu = sVecOp(be, "mul", hGate, hSig, interF); pending.add(hSilu)
  let hGated = sVecOp(be, "mul", hSilu, hUp, interF); pending.add(hGated)
  var hOut = sessionAllocScratch(be, ff.downProj.outF); pending.add(hOut)
  if downQ4:
    step(sessionMatmulQ4(be, hGated, ff.downProj.residentW, hOut, 1), "downProj matmul that bai")
  else:
    step(sessionMatmul(be, hGated, ff.downProj.residentFp32, hOut, 1, ff.downProj.inF, ff.downProj.outF),
         "downProj matmul(fp32) that bai")
  if ff.downProj.bias.data.len == ff.downProj.outF:
    let hDB = sessionUpload(be, ff.downProj.bias.data); pending.add(hDB)
    hOut = sVecOp(be, "add", hOut, hDB, ff.downProj.outF); pending.add(hOut)

  # ---- Residual #2 ----
  result = sVecOp(be, "add", hX2, hOut, C); pending.add(result)

proc forwardDecodeFullMergeSingleSession(attn: GQAAttention, normQKV, normFFN: RMSNormL,
    ff: SwiGLUL, x: Tensor, rope: RoPEL, ctx: Backend, cache: var KVCacheL, layerIdx: int): Tensor =
  ## MODE 2 - giữ nguyên hành vi/tên cũ (1 session/layer): giờ chỉ là 1 lớp
  ## vỏ mỏng gọi encodeDecodeLayerIntoOpenSession giữa 1 cặp sessionBegin/End.
  let C = x.shape[2]
  let be = ctx.kind.toByby()

  template chk(cond: bool, msg: string) =
    if not cond: raise newException(CatchableError, "[MODE2] " & msg & " (layer " & $layerIdx & ")")

  var failMsg: string
  chk(precheckLayerResident(attn, ff, layerIdx, failMsg), failMsg)
  chk(sessionBegin(be), "sessionBegin that bai")
  var pending = newSeq[SessionHandle]()
  var failed = false
  try:
    let hXIn = sessionUpload(be, x.data); pending.add(hXIn)
    let hFinal = encodeDecodeLayerIntoOpenSession(attn, normQKV, normFFN, ff, hXIn, rope, be,
                                                    cache, layerIdx, pending)
    let sessionOk = sessionEnd(be)
    chk(sessionOk, "sessionEnd that bai: " & sessionGetLastError(be))
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
  ## RMSNorm(input) nằm BÊN TRONG forwardPrefill/forwardDecode (gộp với QKV+
  ## RoPE - xem fusedNormQKVRoPE), và residual+RMSNorm(post-attn)+FFN+residual
  ## nằm trong fusedResidualNormFFN - cả block giờ chỉ còn 2 session GPU thay
  ## vì 4+ session rời rạc theo từng submodule như trước.
  # ĐO: tách thời gian attn (QKV+RoPE+attention+oProj) và FFN riêng, cộng dồn
  # qua mọi layer/token - để biết chính xác thời gian rơi vào đâu thay vì đoán.
  let t0 = epochTime()
  if isPrefill:
    let attnOut = forwardPrefill(blk.attn, blk.inputNorm, x, rope, ctx, cache, layerIdx)
    let t1 = epochTime()
    result = fusedResidualNormFFN(blk.postAttnNorm, blk.ff, x, attnOut, ctx)
    let t2 = epochTime()
    gAttnTime += (t1 - t0)
    gFfnTime += (t2 - t1)
    return

  # THÊM: kiểm tra TRƯỚC (thuần CPU, không đụng GPU) xem FFN có đủ điều
  # kiện resident để nối session hay không - forwardDecode CẦN biết trước
  # để quyết định giữ session mở hay đóng bình thường (không được đoán/thử
  # rồi sửa sau - nếu đoán sai, session mở sẽ không bao giờ đóng).
  # CAM LEGACY: bo doc BYBY_LEGACY_FFN va BYBY_NO_MERGE_FFN_SESSION - khong
  # con cua thoat hiem nao duoc phep tat duong resident/merged-session nua.
  let gateUpResidentQ4 = blk.ff.gateProj.residentReady and blk.ff.upProj.residentReady
  let downResidentQ4 = blk.ff.downProj.residentReady
  let gateUpResidentFp32 = (not gateUpResidentQ4) and blk.ff.gateProj.residentFp32Ready and blk.ff.upProj.residentFp32Ready
  let downResidentFp32 = (not downResidentQ4) and blk.ff.downProj.residentFp32Ready
  let ffnAllResident = (gateUpResidentQ4 and downResidentQ4) or (gateUpResidentFp32 and downResidentFp32)
  if not ffnAllResident:
    raise newException(CatchableError,
      "[FFN] resident chua san sang o forwardDecode (layer " & $layerIdx &
      ") - legacy fallback dang bi CAM theo yeu cau, khong chay tiep.")

  if gMergeQkvAttnSession:
    try:
      result = forwardDecodeFullMergeSingleSession(blk.attn, blk.inputNorm, blk.postAttnNorm,
                                                     blk.ff, x, rope, ctx, cache, layerIdx)
      let t2 = epochTime()
      gAttnTime += (t2 - t0)   # MODE 2: gộp 1 session -> không tách được attn/FFN riêng, cộng hết vào gAttnTime
      return
    except CatchableError as e:
      # AN TOÀN: Mode 2 lỗi bất kỳ bước nào -> rơi thẳng về Mode 1 (đường
      # dưới, y hệt code cũ trước khi có Mode 2) cho ĐÚNG token này. Log 1
      # lần đầu tiên để biết mà báo lại, không spam mỗi token.
      if gMode2WarnLeft > 0:
        dec gMode2WarnLeft
        stderr.writeLine "[MODE2] that bai, fallback ve Mode 1 cho token nay: " & e.msg

  let dec0 = forwardDecode(blk.attn, blk.inputNorm, x, rope, ctx, cache, layerIdx, ffnAllResident)
  let t1 = epochTime()
  if dec0.sessionOpen:
    var pending = dec0.pending
    result = fusedResidualNormFFNContinue(blk.postAttnNorm, blk.ff, x, dec0.be, dec0.hOut, pending)
  else:
    result = fusedResidualNormFFN(blk.postAttnNorm, blk.ff, x, dec0.attnOut, ctx)
  let t2 = epochTime()
  if layerIdx == 0 and gDecodeProfileLeft > 0:
    # THÊM: mốc đo FFN còn thiếu (2 dòng DECODE_PROFILE cũ chỉ có phần attn)
    # - giờ đủ 3 dòng để biết CHÍNH XÁC 3 khúc ăn bao nhiêu ms/layer đầu.
    stderr.writeLine "[DECODE_PROFILE] FFN (" & (if dec0.sessionOpen: "session NOI TIEP" else: "session rieng") &
      "): " & $((t2 - t1) * 1000) & " ms"
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

## ---- Adaptive GPU yield (nhường GPU cho desktop khi phát hiện hạ xung
## nhiệt, xem giải thích chi tiết trong forwardStep) ----
var gLayerBaselineSec = -1.0    ## -1 = chưa đo xong baseline
var gBaselineSamples = 0
var gBaselineSumSec = 0.0
var gAdaptiveYieldMs = 0.0      ## bắt đầu ở 0 - chỉ tăng khi thật sự phát hiện chậm dần
let gMaxYieldMs = (try: parseFloat(getEnv("BYBY_GPU_MAX_YIELD_MS", "8.0")) except ValueError: 8.0)

## ---- MODE 3: gộp TOÀN BỘ 32 layer vào ĐÚNG 1 session GPU/token (thay vì 1
## session/layer như Mode 2, tức 32 session/token) - đúng hướng ggml-metal
## (1 command buffer/forward pass). Dùng chung lõi encodeDecodeLayerIntoOpenSession
## với Mode 2, chỉ khác: sessionBegin/End bọc quanh CẢ VÒNG LẶP layer, và hidden
## state được truyền thẳng giữa các layer bằng SessionHandle (ở lại GPU), không
## một lần sessionRead/sessionUpload nào ở giữa - loại bỏ hẳn 31 lượt round-trip
## CPU<->GPU + 31 lượt sessionBegin/End (mỗi lượt ép GPU sync 1 lần) so với Mode 2.
var gMergeAllLayersSession* = getEnv("BYBY_MERGE_ALL_LAYERS_SESSION", "1") == "1"
var gMode3WarnLeft = 20   ## chỉ log tối đa 20 lần fallback Mode3->(Mode2/Mode1), tránh spam
## THÊM MỚI: bộ đếm CỐ ĐỊNH, in 1 LẦN DUY NHẤT ở cuối chương trình (giống
## KV_RESIDENT_ALLOC/gAttnTime/gFfnTime) - KHÔNG phụ thuộc terminal scroll
## real-time. Lý do bắt buộc phải có: dòng cảnh báo "[MODE3] that bai" in
## bằng stderr.writeLine (dòng riêng, TRÔI theo terminal bình thường) trong
## khi dòng "[N/100 tok | ...]" lại dùng ANSI cursor GHIM CỐ ĐỊNH 1 vị trí
## (xem generateTokenLoop, "Ghim tok/s cố định 1 chỗ bằng ANSI cursor") - nếu
## 2 dòng này rơi trúng cùng 1 hàng màn hình, dòng ANSI ghim sẽ ĐÈ MẤT dòng
## cảnh báo Mode3, khiến nhìn log tưởng như Mode3 "chạy êm" trong khi thực ra
## đã fail 100% (đúng ca đã xảy ra: gAttnTime chiếm 94% decode time dù bật
## Mode3 - bằng chứng không thể chối được rằng Mode3 fail toàn bộ, nhưng
## không dòng cảnh báo nào hiện trên terminal vì bị ANSI đè). Bộ đếm này in
## RA NGOÀI vùng bị ANSI ghi đè (chỉ 1 lần, cuối cùng), không thể mất được.
var gMode3Attempts = 0
var gMode3Success = 0
var gMode3FirstFailMsg = ""

proc forwardDecodeAllLayersMergedSession(m: LlamaModelL, xData: seq[float32], ctx: Backend,
    cache: var KVCacheL): Tensor =
  let be = ctx.kind.toByby()
  let C = m.config.hiddenSize

  # Kiểm tra TRƯỚC (thuần CPU, không đụng GPU) TOÀN BỘ 32 layer đã resident đủ
  # điều kiện chưa - tránh mở session rồi fail ngay layer đầu (phí 1 lượt GPU
  # sync vô ích). Chỉ cần 1 layer chưa sẵn sàng -> raise ngay, KHÔNG mở session.
  for idx in 0 ..< m.blocks.len:
    var failMsg: string
    if not precheckLayerResident(m.blocks[idx].attn, m.blocks[idx].ff, idx, failMsg):
      raise newException(CatchableError, "[MODE3] " & failMsg)

  if not sessionBegin(be):
    raise newException(CatchableError, "[MODE3] sessionBegin that bai")
  var pending = newSeq[SessionHandle]()
  var failed = false
  try:
    var hX = sessionUpload(be, xData); pending.add(hX)
    for idx in 0 ..< m.blocks.len:
      hX = encodeDecodeLayerIntoOpenSession(m.blocks[idx].attn, m.blocks[idx].inputNorm,
             m.blocks[idx].postAttnNorm, m.blocks[idx].ff, hX, m.rope, be, cache, idx, pending)
      # LƯU Ý: KHÔNG sessionFree hX của layer trước ở đây - session còn đang
      # mở, mọi handle đã encode đều phải sống tới sessionEnd (xem giải thích
      # đầy đủ ở encodeDecodeLayerIntoOpenSession). hX layer trước đã nằm
      # trong pending (do encodeDecodeLayerIntoOpenSession tự add), sẽ được
      # free CHUNG 1 lượt ở finally bên dưới, sau khi session này đã đóng.
    let sessionOk = sessionEnd(be)
    if not sessionOk:
      raise newException(CatchableError, "[MODE3] sessionEnd that bai: " & sessionGetLastError(be))
    result = newTensor(@[1, 1, C])
    result.data = sessionRead(be, hX, C)
  except CatchableError:
    failed = true
    raise
  finally:
    if failed:
      discard sessionEnd(be)
    for h in pending.mitems:
      sessionFree(be, h)

proc forwardStep*(m: LlamaModelL, ids: seq[int], ctx: Backend, cache: var KVCacheL,
                   isPrefill: bool): Tensor =
  ## isPrefill=true: ids = toàn bộ prompt (T token). isPrefill=false: ids = đúng 1 token mới.
  var x = m.embed.lookupBatch(@[ids], ctx)   # [1, T, C]
  when defined(debugStates):
    let Tdbg = ids.len
    let Cdbg = m.config.hiddenSize
    stderr.writeLine "[debug] after_embedding last_token[:8] = " & $x.data[(Tdbg-1)*Cdbg ..< (Tdbg-1)*Cdbg+8]

  # MODE 3: thử gộp CẢ 32 layer vào 1 session TRƯỚC KHI vào vòng lặp per-layer
  # cũ (Mode 2/Mode 1) - chỉ áp dụng cho decode (isPrefill=false), giống hệt
  # điều kiện áp dụng của Mode 2. Lỗi bất kỳ đâu (1 trong 32 layer) -> raise
  # toàn bộ, rơi THẲNG về vòng lặp per-layer cũ CHO ĐÚNG TOKEN NÀY (vòng lặp
  # đó tự thử lại Mode 2 rồi Mode 1 cho từng layer như trước - không đổi gì).
  var mode3Done = false
  if gMergeAllLayersSession and not isPrefill:
    inc gMode3Attempts
    try:
      x = forwardDecodeAllLayersMergedSession(m, x.data, ctx, cache)
      mode3Done = true
      inc gMode3Success
    except CatchableError as e:
      if gMode3FirstFailMsg.len == 0:
        gMode3FirstFailMsg = e.msg
      if gMode3WarnLeft > 0:
        dec gMode3WarnLeft
        stderr.writeLine "[MODE3] that bai, fallback ve vong lap per-layer (Mode2/Mode1) cho token nay: " & e.msg

  if not mode3Done:
    for idx in 0 ..< m.blocks.len:
      let layerT0 = epochTime()
      x = m.blocks[idx].forward(x, m.rope, ctx, cache, idx, isPrefill)
      let layerElapsed = epochTime() - layerT0
      # SỬA (thay hẳn yield cố định bằng adaptive): yield 1 số ms CỐ ĐỊNH có
      # 2 mặt xấu - (1) cao hơn cần thiết -> phí token/s lúc máy còn mát,
      # không cần nhường nhiều vậy, (2) thấp hơn cần thiết -> máy vẫn tích
      # nhiệt dần rồi macOS TỰ hạ xung GPU (thermal throttle) - lúc đó
      # token/s còn tệ hơn cả không yield gì, vì throttle làm CHẬM VĨNH VIỄN
      # (tới khi nguội) chứ không phải chỉ chậm đúng lúc yield. Ở đây dùng
      # chính thời gian THỰC THI mỗi layer làm tín hiệu phản hồi: throttle
      # xảy ra thì cùng 1 layer (cùng lượng việc) sẽ chạy CHẬM DẦN so với
      # baseline lúc mới khởi động (còn mát) - vòng lặp EMA dưới đây tự
      # tăng yield khi phát hiện chậm dần (đang nóng lên), tự giảm yield về
      # 0 khi ổn định trở lại (đã mát/không cần nhường) - không đoán mò 1
      # con số cố định, luôn bám sát trạng thái nhiệt thực tế của máy.
      if not isPrefill:
        if gLayerBaselineSec < 0:
          # Khởi động: 3 layer đầu tiên của TOKEN ĐẦU TIÊN dùng làm baseline
          # "lúc máy còn mát" (chưa kịp nóng lên) - không yield trong lúc đo.
          inc gBaselineSamples
          gBaselineSumSec += layerElapsed
          if gBaselineSamples >= 3:
            gLayerBaselineSec = gBaselineSumSec / float(gBaselineSamples)
        else:
          let ratio = layerElapsed / gLayerBaselineSec
          if ratio > 1.15:
            # Chậm hơn baseline >15% -> dấu hiệu đang hạ xung -> tăng yield
            # để hạ nhiệt (tăng dần, không nhảy vọt, tránh dao động qua lại).
            gAdaptiveYieldMs = min(gAdaptiveYieldMs + 0.3, gMaxYieldMs)
          elif ratio < 1.05:
            # Về gần baseline -> đã mát/ổn định -> đòi lại tốc độ dần dần.
            gAdaptiveYieldMs = max(gAdaptiveYieldMs - 0.1, 0.0)
          if gAdaptiveYieldMs >= 1.0:
            sleep(int(gAdaptiveYieldMs))
        when defined(debugGpuYield):
          if idx == m.blocks.len - 1:
            stderr.writeLine "[gpu_yield] baseline=" & $gLayerBaselineSec &
              "s last=" & $layerElapsed & "s yieldMs=" & $gAdaptiveYieldMs
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
  # THÊM MỚI: sparsity (block-level, ép scale=0 cho khối kém quan trọng) -
  # đọc từ env var, mặc định 0.0 (TẮT, hành vi giống hệt trước đây). CHỈ áp
  # dụng cho FFN (gate/up/down_proj - ma trận LỚN NHẤT) - Q/K/V/O giữ nguyên
  # không đổi gì để hạn chế rủi ro ảnh hưởng chất lượng phần attention.
  let sparsityPct = parseFloat(getEnv("BYBY_SPARSITY_PCT", "0.0")).float32
    ## SỬA LẠI: quay về mặc định TẮT (0.0). Đã thử 20% trên máy thật - kết
    ## quả: đầu ra hỏng (sinh chữ vô nghĩa, không còn là code Fibonacci đúng
    ## nữa) VÀ tốc độ cũng không chắc nhanh hơn khi máy chạy tải bình thường
    ## (không phải lúc "siêu mát" đo được 32.9s trước đó). 20% là quá tay
    ## cho model này khi không có bước fine-tune/calibrate lại sau khi cắt -
    ## sai số cộng dồn qua 32 layer làm hỏng hẳn đầu ra. Muốn thử lại, bật
    ## thủ công bằng BYBY_SPARSITY_PCT với mức THẤP hơn nhiều (vd 3-5), tăng
    ## dần từng chút, luôn so sánh đầu ra với bản không sparsity trước khi
    ## tăng thêm.
  if sparsityPct > 0.0'f32:
    echo "  [SPARSITY] BAT: cat " & $sparsityPct & "% khoi FFN (gate/up/down_proj) - " &
         "LAM MAT CHAT LUONG, doi lay toc do doc GPU nhanh hon o kernel batch prefill"
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
  uploadResident(m.lmHead, ctx)  # no-op nếu tied-embedding (useQuant=false)
  tune(m.lmHead)
  uploadResidentNorm(m.finalNorm, ctx)
  # TOI UU: gamma cua CA 32*2+1 RMSNorm (input+postAttn moi layer + final)
  # gio duoc upload 1 LAN DUY NHAT o day, thay vi sessionUpload() lai TU
  # DAU o MOI layer MOI token trong forwardPrefill/forwardDecode - giam
  # dung 1 upload buffer nho (~dim*4 byte) khong can thiet moi buoc decode
  # (xem 3 vi tri da sua trong fusedNormQKVRoPE/fusedResidualNormFFN(Continue)).
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

  # GPU HOA: phan dat nhat cua sampling la exp() tren toan bo vocab (co the
  # 32k-128k+ phan tu MOI token) - chuyen sang GPU qua beSoftmax (dung
  # gpuSoftmax, da CAM CPU fallback y het cac op khac). Truoc day: vong lap
  # CPU thuan tinh exp() + tong + chia cho tung phan tu.
  # Luu y: adjLogits da tru maxL va chia temperature TRUOC khi dua vao GPU,
  # vi kernel softmax GPU (softmax_kernel/rmsnorm-style) tu lo phan on dinh
  # so hoc (tru max noi bo) nhung KHONG biet ve temperature scaling cua rieng
  # sampling - phai chia truoc.
  var scaledLogits = newSeq[float32](adjLogits.len)
  for i in 0 ..< adjLogits.len: scaledLogits[i] = adjLogits[i] / temperature
  var probs = beSoftmax(ctx, scaledLogits, 1, adjLogits.len)

  # TOI UU (thay the cho "idx.sort toan bo vocab O(V log V) MOI token"):
  # top-p chi can CAT tai nguong tich luy topP - khong can sap xep TOAN BO
  # V phan tu, chi can 1 tap con K phan tu co xac suat lon nhat. Dung
  # partial selection (selection theo nguong) thay vi full sort:
  #   1) Quet 1 lan tim nguong xap xi (khong can chinh xac tuyet doi) bang
  #      cach lay mau K ung vien lon nhat qua vong lap don (K nho, ~256-512,
  #      du de topP thuc te <=0.95-0.99 hau nhu luon nam trong K nay voi
  #      phan phoi softmax tu logits LLM that - da duoc quan sat trong
  #      thuc te llama.cpp/vLLM dung chien luoc "top_k lam truoc top_p").
  #   2) CHI sort K phan tu do (O(K log K)) thay vi V phan tu.
  # Neu K khong du (topP chua dat sau K phan tu, hiem khi xay ra), fallback
  # mo rong K gap doi va thu lai - toi da vai lan, van re hon full sort.
  var K = min(512, probs.len)
  var idx: seq[int]
  while true:
    # TOI UU THAT (thay the ban dau O(V*K) sai lam - da phat hien va sua
    # ngay tai day): dung min-heap kich thuoc K (std/heapqueue) -> O(V log K)
    # thay vi quet tuyen tinh tim-min moi phan tu (O(V*K), CHAM HON ca sort
    # goc O(V log V) voi K=512). Heap so sanh theo (prob, idx) tang dan -
    # phan tu dinh heap luon la prob NHO NHAT trong K hien tai, bi thay the
    # ngay khi gap phan tu moi lon hon.
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

proc generate*(m: LlamaModelL, tok: HFTokenizer, ctx: Backend, promptIds: seq[int],
               maxNewTokens: int = 128, temperature: float32 = 0.7'f32, topP: float32 = 0.9'f32): string =
  echo "[inference] optimized paths: all-layers-session=" & $gMergeAllLayersSession &
    ", merged-QKV-attention=" & $gMergeQkvAttnSession &
    ", resident-KV=" & $(getEnv("BYBY_DISABLE_KV_RESIDENT", "0") != "1")
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

  # ---- WARMUP KV-RESIDENT: cấp phát TRƯỚC heap KV cho CẢ 32 layer, NGAY SAU
  # KHI TẠO cache, TRƯỚC CẢ PREFILL. SỬA LẠI VỊ TRÍ (bản trước đặt SAU
  # prefill là SAI - đã tự kiểm chứng bằng log thật: warmupTime luôn ra
  # 0.000s trong khi KV_RESIDENT_ALLOC vẫn tốn 65-100s NGAY TRONG LÚC
  # PREFILL, vì đường prefill cũ (syncKvResidentEncode, dòng ~646) tự gọi
  # ensureKvResident() cho từng layer bên trong CHÍNH forwardStep(isPrefill
  # = true) - đặt warmup sau lệnh gọi đó là quá trễ, prefill đã âm thầm
  # "nuốt" hết chi phí 1 lần này vào số đo "Prefill: ... tok/s" của chính
  # nó rồi). Đặt Ở ĐÂY - trước dòng forwardStep(isPrefill=true) đầu tiên -
  # đảm bảo CẢ Prefill LẪN Decode đều đo được tốc độ THẬT, không dính chi
  # phí cấp phát 1-lần-duy-nhất này. ensureKvResident() không cần session
  # GPU nào đang mở (sessionAllocHeap/sessionAllocFromHeap là API cấp phát
  # device-level thuần) nên an toàn gọi sớm thế này.
  let warmupStart = epochTime()
  block:
    let warmupBe = ctx.kind.toByby()
    for idx in 0 ..< cache.layers.len:
      # LƯU Ý: lấy nKVHeads/headDim TỪ CHÍNH layer đó (m.blocks[idx].attn),
      # KHÔNG dùng m.config.nKVHeads/hiddenSize toàn cục - phải khớp CHÍNH
      # XÁC nguồn dữ liệu mà encodeDecodeLayerIntoOpenSession/syncKvResidentEncode
      # dùng khi gọi ensureKvResident() thật sự lúc prefill/decode.
      ensureKvResident(cache.layers[idx], warmupBe, m.blocks[idx].attn.nKVHeads,
                        cache.maxSeqLen, m.blocks[idx].attn.headDim)
  let warmupTime = epochTime() - warmupStart

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
    let stepLogits = m.forwardStep(@[nextId], ctx, cache, isPrefill = false)
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
  stderr.write(&"KV_RESIDENT_ALLOC: cap phat tong cong {gKvAllocCount} lan " &
    &"(dung neu == so layer = {m.config.nLayers}, LEAK neu > {m.config.nLayers})\n")
  stderr.write(&"Decode - attn (QKV+RoPE+attention+oProj): {gAttnTime:.3f}s | FFN: {gFfnTime:.3f}s | " &
               &"tổng 2 phần: {(gAttnTime+gFfnTime):.3f}s\n")
  if gMergeAllLayersSession:
    let pct = (if gMode3Attempts > 0: 100.0 * gMode3Success.float / gMode3Attempts.float else: 0.0)
    stderr.write(&"MODE3 (gop ca 32 layer/session): thu {gMode3Attempts} lan - THANH CONG {gMode3Success} " &
      &"({pct:.1f}%), FALLBACK {gMode3Attempts - gMode3Success} lan\n")
    if gMode3Success < gMode3Attempts:
      stderr.write(&"  -> ly do fail LAN DAU: {gMode3FirstFailMsg}\n")
  stderr.write(&"KV-resident warmup (32 layer, cap phat 1 lan, TRUOC prefill): {warmupTime:.3f}s\n")
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

  # ── Đối số dòng lệnh (đều có mặc định = giá trị cũ trước đây,
  #    không set gì thì hành vi y hệt bản gốc) ──
  #   ./nim_inference [modelPath] [nimqPath] [backend] [prompt] [maxNewTokens] [temperature] [topP] [systemText]
  let args = commandLineParams()
  proc argOr(i: int, default: string): string =
    if args.len > i and args[i].len > 0: args[i] else: default

  let modelPath = argOr(0, "test/deepseek-inference/models")
  let nimqPath = argOr(1, "test/deepseek-inference/models/model.nimq")
  let backendName = argOr(2, "auto")
  let promptText = argOr(3, "Write a Python function to return the n-th Fibonacci number.")
  let maxNewTokens = parseInt(argOr(4, "100"))
  let temperature = parseFloat(argOr(5, "0.7")).float32
  let topP = parseFloat(argOr(6, "0.9")).float32
  let systemText = argOr(7, "You are an AI programming assistant.")

  echo "Loading config..."
  let cfg = newLlamaLConfig(modelPath)
  echo "Loading tokenizer..."
  let tok = newHFTokenizer(modelPath)
  echo &"Init backend ({backendName})..."
  let ctx = newBackend(backendName)
  echo "Building model..."
  var model = newLlamaModelL(cfg)
  echo &"Loading weights tu {nimqPath} ..."
  loadLlamaWeightsL(model, nimqPath)
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
  let promptIds = tok.applyChatTemplate(promptText, systemText = systemText)
  discard generate(model, tok, ctx, promptIds, maxNewTokens = maxNewTokens,
                    temperature = temperature, topP = topP)
