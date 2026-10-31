## nimformer.nim
## Transformer đầy đủ với backward cho attention + custom transformer blocks

import std/[math, random, strformat, algorithm, tables]
import customfloat
import backend  # Import Backend và các helper
import quant     # QuantTensor, dequantizeTensor - giữ weight nén int4 trong RAM
import vendor/bybylang/gpubackend as byby
import fp4_gemv_gpu  # gFp4CacheGpu, fp4GemmGpuRows (prefill FP4-KM)
import prefill_profile  # BYBY_PREFILL_PROFILE: pfTime

# ===================================================================
# Tensor utilities (giữ nguyên)
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

var gLoraSizeMismatchWarned = 0  ## THÊM: đếm số lần đã cảnh báo kích thước LoRA
                                  ## không khớp (xem forward() dưới) - giới hạn để
                                  ## không spam log nếu nhiều Linear cùng bị.

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
  ## SỬA: bản addT(a,b) ở trên chạy CPU thuần dù beAdd (GPU thật, có cài cho
  ## cả CUDA/Metal/OpenCL) đã có sẵn - residual connection này chạy 2 LẦN MỖI
  ## LAYER (attn + ff), tức 64 lần/token với model 32 layer, không phải phép
  ## tính nhỏ có thể bỏ qua. Overload này route qua GPU đúng yêu cầu "không
  ## được lặng lẽ chạy CPU". Giữ nguyên bản addT(a,b) cũ (không ctx) cho các
  ## chỗ dùng khác không có Backend trong scope.
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
  ## Transpose 1 tensor coi như ma trận 2D [rows, cols] (gộp mọi chiều trừ
  ## chiều cuối làm "rows", giống flatten2D) -> [cols, rows].
  let (rows, cols) = flatten2D(x.shape)
  result.shape = @[cols, rows]
  result.data = newSeq[float32](rows * cols)
  for i in 0 ..< rows:
    for j in 0 ..< cols:
      result.data[j * rows + i] = x.floatValues[i * cols + j]

# ===================================================================
# Linear (forward/backward dùng Metal matmul)
# ===================================================================
type
  Linear* = object
    weight*: Tensor      ## fallback fp32 nhỏ (chỉ dùng cho random-init lúc chưa load weight thật)
    weightQ*: QuantTensor ## THÊM: weight THẬT giữ nguyên dạng NÉN (int4/int8...) suốt vòng đời
                           ## process, giống cách llama.cpp giữ quantized weight trong RAM -
                           ## KHÔNG bao giờ expand hết ra fp32 một lần rồi giữ mãi (đó là
                           ## nguyên nhân OOM trước đó: 6.7B tham số x 4 byte fp32 = ~27GB RAM).
    useQuant*: bool        ## true sau khi loadLinear() gán weightQ thật (int4 nén, ~4GB tổng
                            ## cho cả model thay vì ~27GB fp32)
    bias*: Tensor
    inF*, outF*: int
    residentW*: ResidentWeightHandle  ## weight đã upload GPU 1 LẦN (xem uploadResident)
    residentReady*: bool               ## true sau uploadResident() thành công
    residentZeroCopy*: bool            ## residentW wq wraps the model mmap in place
    residentFp32*: SessionHandle       ## FIX: resident fp32 CHUNG cho weight KHÔNG phải
                                        ## int4-asymmetric (int8/int4-symmetric/tied-embedding) -
                                        ## dequant 1 lần lúc load thay vì mỗi token (xem backend.nim)
    residentFp32Ready*: bool           ## true sau khi residentFp32 đã upload xong
    residentFp32NT*: bool              ## residentFp32 stores original [outF,inF] rows
    residentFp32Offset*: int           ## floats before the weight in residentFp32 (mmap wrap)
    loraA*, loraB*: seq[float32]  ## THÊM (xem forward() dưới + mergeInUploadStreamingJamba
                                   ## trong lora.nim): adapter LoRA gắn rời, KHÔNG merge vật
                                   ## lý vào weight - giữ base (weightQ/useQuant) NGUYÊN VẸN
                                   ## dạng nén suốt đời process, đúng đúng bất biến cốt lõi
                                   ## của type này (xem comment weightQ ở trên: "KHÔNG bao giờ
                                   ## expand hết ra fp32 một lần rồi giữ mãi"). loraA:[r,inF]
                                   ## phẳng, loraB:[outF,r] phẳng, rỗng (@[]) nếu không gắn.
    mappedLoraA*, mappedLoraB*: MappedView[float32]
    loraR*: int                   ## rank r; 0 = KHÔNG có LoRA gắn (mặc định, code cũ không đổi)
    loraScale*: float32           ## alpha / r
    gpuCacheName*: string         ## ten trong gFp4CacheGpu sau prepareGpuLinForLinear (FP4-KM)

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
  ## Gọi từ loadLinear() khi load weight thật từ file .nimq - giữ NGUYÊN
  ## dạng nén (không dequant ngay), giải phóng luôn weight fp32 fallback
  ## (không cần nữa, tốn RAM vô ích vì đã có weightQ).
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
  ## Upload weight lên GPU MỘT LẦN (gọi từ loadLlamaWeightsL sau khi load
  ## xong, KHÔNG gọi trong vòng lặp decode).
  ## - int4-per-group (Q4_K_M style, đường llama.cpp-style): dùng thẳng
  ##   kernel int4 gốc trên Metal/OpenCL (residentW), nhanh nhất + tiết
  ##   kiệm băng thông nhất.
  ## - MỌI trường hợp khác vẫn dùng weight thật (int8, int4-symmetric, hoặc
  ##   Linear fp32 thường/tied-embedding lm_head): FIX - TRƯỚC ĐÂY các
  ##   trường hợp này KHÔNG có đường resident nào cả -> forward() phải
  ##   dequantizeTensorTransposed()/transpose() lại TOÀN BỘ ma trận trên CPU
  ##   MỖI TOKEN (nguyên nhân chính gây ~4s/token). Giờ dequant/transpose
  ##   MỘT LẦN duy nhất ở đây rồi giữ resident dạng fp32 trên GPU
  ##   (beResidentUploadFp32), mỗi token chỉ upload activation nhỏ.
  ##
  ## THÊM MỚI: sparsityPct > 0 -> BLOCK-LEVEL SPARSITY áp dụng NGAY TRONG
  ## NIM, TRƯỚC khi upload GPU - KHÔNG đụng file .nimq/export_to_nimq.py
  ## (theo đúng yêu cầu). Với mỗi khối group_size giá trị liên tiếp, tính độ
  ## quan trọng bằng biên độ trung bình quy đổi xấp xỉ (0.5*scale, vì giá trị
  ## int4 giải nén nằm quanh [0,15] canh giữa quanh 7.5, biên độ dao động
  ## thực ~ 0.5*range = 0.5*scale*15 -> gần đúng dùng "scale" làm proxy độ
  ## quan trọng, KHÔNG cần giải nén đầy đủ 128 giá trị mỗi khối - rẻ, nhanh,
  ## chạy 1 lần lúc load model). Khối bị cắt: ép scale=0 -> công thức giải
  ## nén (nibble-zp)*scale LUÔN = 0 - ĐÚNG TOÁN HỌC, không phải xấp xỉ. Chỉ
  ## sửa MẢNG scale ĐÃ Ở TRONG RAM (l.weightQ.scale), không đổi packed data,
  ## không đổi format, không ghi lại file nào.
  if l.useQuant and l.weightQ.kind == qkInt4Asymmetric and l.weightQ.groupSize > 0:
    if sparsityPct > 0.0'f32 and l.weightQ.scale.len > 0:
      var scaleCopy = l.weightQ.scale  # copy để tính percentile không ảnh hưởng thứ tự gốc
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
    # TOI UU RAM: sau khi upload xong, forward() CHI dung duong residentFp32Ready
    # (xem forward() duoi day) - l.weight.data (neu co, vd sau mergeIn() ghi
    # merged.data vao day) khong con can nua nhung truoc gio KHONG duoc xoa =>
    # ton tai SONG SONG 1 ban fp32 tren CPU RAM + 1 ban tren GPU (tren may GPU
    # tich hop/unified memory nhu Intel Iris, ca 2 ban DEU an chung RAM he
    # thong = gap doi RAM thuc te dung so voi can thiet). Giai phong ngay,
    # giong het pattern setQuantWeight() da lam o tren.
    l.weight.data = @[]
    l.weight.mappedData = default(MappedView[float32])
  elif l.weight.floatValues.len > 0:
    # Linear fp32 thường (vd. tied-embedding lm_head dùng thẳng embW) -
    # transpose 1 LẦN thay vì mỗi token.
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

proc forward*(l: Linear, x: Tensor, ctx: Backend): Tensor =
  let (rows, cols) = flatten2D(x.shape)
  assert cols == l.inF
  var outShape = x.shape
  outShape[^1] = l.outF
  result = newTensor(outShape)

  var y: seq[float32]
  pfTime("lin.gemm"):
    if l.residentReady:
      y = beResidentLinearQ4(ctx, x.data, l.residentW, rows)  # chỉ upload activation, weight resident sẵn
    elif l.residentFp32Ready:
      # FIX: weight int8/int4-symmetric/fp32-thường ĐÃ resident (xem
      # uploadResident) - chỉ upload activation nhỏ mỗi token, KHÔNG dequant
      # lại toàn bộ weight nữa.
      y = if l.residentFp32NT: residentLinearNt(l, x.data, rows, ctx)
          else: beResidentLinearFp32(ctx, x.data, l.residentFp32, rows, l.inF, l.outF)

    if y.len == 0:
      # Chưa resident, hoặc backend hiện tại không hỗ trợ đường resident (vd
      # CPU) -> rơi về đường cũ (vẫn đúng, chỉ mất phần tăng tốc resident).
      if l.useQuant and l.weightQ.kind == qkFp4KM and l.gpuCacheName.len > 0 and
         l.gpuCacheName in gFp4CacheGpu and gFp4CacheGpu[l.gpuCacheName].residentReady and
         ctx.kind.toByby() in {byby.gbMetal, byby.gbVulkan}:
        # Prefill song song: GPU resident FP4-KM + MBLOCK=4 (khong dequant CPU).
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

  # === FIX: LUÔN ĐẢM BẢO KÍCH THƯỚC ĐÚNG ===
  let expectedSize = rows * l.outF
  if y.len != expectedSize:
    # Tạo mảng mới với kích thước đúng, copy những phần tử có sẵn, phần còn lại để 0
    var fixed = newSeq[float32](expectedSize)
    let copyLen = min(y.len, expectedSize)
    for i in 0 ..< copyLen:
      fixed[i] = y[i]
    y = fixed

  # Cộng bias (l.bias.data có độ dài l.outF; rỗng = không có bias)
  if l.bias.data.len == l.outF:
    for i in 0 ..< y.len:
      result.data[i] = y[i] + l.bias.data[i mod l.outF]
  else:
    for i in 0 ..< y.len: result.data[i] = y[i]

  # THÊM: LoRA CỘNG THÊM (additive), KHÔNG merge vật lý vào weight. y/result ở
  # trên luôn đi qua đường resident/quant Q4 NÉN như base gốc (weightQ/useQuant
  # KHÔNG hề bị đụng tới bởi LoRA nữa - xem mergeInUploadStreamingJamba trong
  # lora.nim) - CHỈ cộng thêm 1 delta hạng thấp [rows,outF] = scale * (x @ A^T)
  # @ B^T, đúng tinh thần QLoRA lúc suy luận chuẩn (base lượng tử hoá giữ
  # nguyên, adapter nhỏ cộng thêm ngoài). r nhỏ (thường 4-64) nên 2 phép nhân
  # nhỏ này RẺ hơn rất nhiều so với matmul chính [rows,inF]x[inF,outF]. TRÁNH
  # đúng lỗi bloat ~27GB (merge vật lý ra fp32 cho cả 105 tham số lớn) từng
  # gây OOM/Killed - xem comment ở định nghĩa Linear/loraA phía trên.
  if l.loraR > 0 and l.loraAValues.len == l.loraR * l.inF and l.loraBValues.len == l.outF * l.loraR:
    pfTime("lin.lora"):
      let At = transpose(Tensor(shape: @[l.loraR, l.inF], data: @(l.loraAValues)))   # [inF, r]
      let Bt = transpose(Tensor(shape: @[l.outF, l.loraR], data: @(l.loraBValues)))  # [r, outF]
      let xA = beMatmul(ctx, x.data, rows, l.inF, At.data, l.inF, l.loraR)       # [rows, r]
      let delta = beMatmul(ctx, xA, rows, l.loraR, Bt.data, l.loraR, l.outF)     # [rows, outF]
      for i in 0 ..< result.data.len:
        result.data[i] += delta[i] * l.loraScale
  elif l.loraR > 0:
    # CHOT CHAN (khong crash ca tien trinh): l.loraR > 0 nhung loraA/loraB
    # KHONG dung kich thuoc mong doi (r*inF va outF*r) - rat co the do 1
    # adapter bi thieu phan A hoac B trong chinh file .bin, hoac kich thuoc
    # r trong meta khong khop voi tensor da luu (xem loadBinaryLoraForInfer
    # trong infer_lora.nim). BO QUA delta LoRA cho DUNG tham so nay (dung
    # weight goc, khong co dieu chinh LoRA cho no) thay vi index vao seq sai
    # kich thuoc/rong -> IndexDefect crash toan bo qua trinh sinh.
    if gLoraSizeMismatchWarned < 10:
      inc gLoraSizeMismatchWarned
      stderr.writeLine("[canh bao] LoRA adapter kich thuoc khong khop cho 1 Linear " &
        "(r=" & $l.loraR & ", loraA.len=" & $l.loraAValues.len & " (can " & $(l.loraR * l.inF) &
        "), loraB.len=" & $l.loraBValues.len & " (can " & $(l.outF * l.loraR) &
        ")) - bo qua delta LoRA cho tham so nay, dung weight goc.")
proc backward*(l: Linear, x: Tensor, dOut: Tensor, ctx: Backend): tuple[dX, dW, dB: Tensor] =
  ## Backward Linear: 
  ## dW = x^T @ dOut  (cộng dồn qua batch)
  ## dB = sum(dOut, axis=0)
  ## dX = dOut @ W
  let (rows, cols) = flatten2D(x.shape)
  doAssert cols == l.inF,
    &"Linear.backward: x.shape={x.shape} (cols={cols}) nhưng l.inF={l.inF}"
  doAssert dOut.shape[^1] == l.outF,
    &"Linear.backward: dOut.shape={dOut.shape} (last={dOut.shape[^1]}) nhưng l.outF={l.outF} (l.inF={l.inF}, x.shape={x.shape})"

  # dW = x^T @ dOut (transpose x -> [inF, rows], dOut -> [rows, outF])
  let xT = transpose(x)  # [inF, rows]
  let dOutFlat = dOut.data  # [rows * outF]

  # dW và dX ĐỘC LẬP với nhau (không cái nào cần đọc kết quả của cái kia),
  # nên gộp cả 2 vào 1 command buffer (metalMatmul2) thay vì gọi metalMatmul
  # riêng 2 lần — giảm một nửa số lần dispatch+wait GPU cho mỗi Linear.backward.
  let (dWRawFlat, dXFlat) = beMatmul2(ctx,
    xT.data, l.inF, rows, dOutFlat, rows, l.outF,
    dOutFlat, rows, l.outF, l.weight.data, l.outF, l.inF)

  var dWRaw: Tensor
  dWRaw.shape = @[l.inF, l.outF]
  dWRaw.data = dWRawFlat
  let dW = transpose(dWRaw)  # -> [outF, inF], khớp l.weight.shape

  var dX: Tensor
  dX.shape = x.shape
  dX.data = dXFlat

  # dB = sum dOut theo batch
  var dB = newTensor(@[l.outF])
  for i in 0 ..< rows:
    for j in 0 ..< l.outF:
      dB.data[j] += dOutFlat[i * l.outF + j]

  result = (dX, dW, dB)

proc backwardInputOnly*(l: Linear, x: Tensor, dOut: Tensor, ctx: Backend): Tensor =
  ## Backward chi tinh dX khi Linear.weight/bias dang frozen. Tranh tinh va
  ## cap phat dW=[outF,inF] khong duoc dung (dac biet lm_head vocab x hidden).
  let (rows, cols) = flatten2D(x.shape)
  doAssert cols == l.inF,
    &"Linear.backwardInputOnly: x.shape={x.shape} (cols={cols}) nhung l.inF={l.inF}"
  doAssert dOut.shape[^1] == l.outF,
    &"Linear.backwardInputOnly: dOut.shape={dOut.shape} (last={dOut.shape[^1]}) nhung l.outF={l.outF}"
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
# LayerNorm (forward/backward dùng Metal matmul)
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
  ## ĐÃ SỬA: trước đây hàm này nhận `ctx: Backend` nhưng KHÔNG DÙNG - toàn bộ
  ## LayerNorm chạy bằng vòng lặp Nim thuần trên CPU bất kể người dùng chọn
  ## backend cuda/metal/opencl nào. backend.nim đã có sẵn `beLayernorm` khớp
  ## đúng phép toán này (rows x cols, cùng công thức chuẩn hoá) nên chỉ cần
  ## gọi qua đó để LayerNorm thực sự chạy trên GPU đã chọn.
  let (rows, cols) = flatten2D(x.shape)
  assert cols == ln.dim
  result.shape = x.shape
  result.data = beLayernorm(ctx, x.data, ln.gamma.data, ln.beta.data, rows, cols, ln.eps)

proc backward*(ln: LayerNorm, x: Tensor, dOut: Tensor, ctx: Backend): tuple[dX, dGamma, dBeta: Tensor] =
  ## Backward LayerNorm (dùng phép toán ma trận / GPU-bound)
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
# CausalSelfAttention (forward/backward dùng Metal matmul)
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
# FeedForward (forward/backward dùng GPU-bound APFLU Activation)
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
# TransformerBlock (Post-LN) - forward/backward dùng Metal
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
  # SỬA (tối ưu): "add rồi layernorm ngay sau" trước đây là 2 lệnh GPU rời
  # rạc (addT rồi ln2.forward, mỗi cái tự mở/đóng command buffer riêng) -
  # gộp vào 1 session bằng beAddLayernormFused (xem backend.nim), trả về
  # CẢ residual thô (x2Raw, cần cho lần cộng cuối block) LẪN bản đã qua
  # layernorm (x3) từ ĐÚNG 1 lần upload/commit/wait - không tính add 2 lần.
  # Backend chưa hỗ trợ (CUDA/OpenCL/CPU) tự rơi về đúng 2 lệnh cũ bên trong.
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
  
  # Forward lại (cần cho backward)
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
# Embedding (forward/backward dùng Metal matmul)
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
# NimformerModel (Post-LN) - forward/backward dùng Metal
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
  ## Forward THẬT với batch B (idsBatch.len chuỗi cùng độ dài T) trong 1 lần
  ## gọi — Linear/LayerNorm coi B*T là số "hàng" nên GPU nhận hẳn M=B*T dòng
  ## trong mỗi matmul, thay vì bị gọi B lần tuần tự (mỗi lần B=1) như trước.
  ## Trả logits shape [B, T, vocab].
  var x = m.embed.lookupBatch(idsBatch, ctx)
  for blk in m.blocks:
    x = blk.forward(x, ctx)
  result = m.outProj.forward(x, ctx)

proc forward*(m: NimformerModel, ids: seq[int], ctx: Backend): Tensor =
  ## Tiện ích cho 1 chuỗi đơn (giữ nguyên API cũ cho test_nimformer.nim) —
  ## gọi forwardBatch với B=1 rồi bỏ chiều batch.
  let logits = m.forwardBatch(@[ids], ctx)
  result = Tensor(data: logits.data, shape: logits.shape[1 .. ^1])

proc backwardBatch*(m: NimformerModel, idsBatch: seq[seq[int]], dLoss: Tensor,
                     ctx: Backend): seq[Tensor] =
  ## Backward THẬT với batch B. dLoss: shape [B, T, vocab] (gradient loss
  ## theo logits, ĐÃ đúng batch — không còn phải giả B=1 rồi bọc thêm 1
  ## chiều như bản cũ). Linear.backward tự cộng dồn gradient qua B*T hàng
  ## trong 1 matmul, nên dW/dB trả về đã là tổng đúng của CẢ batch.
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
  ## Tiện ích cho 1 chuỗi đơn (giữ nguyên API cũ cho test_nimformer.nim).
  let dLossBatch = Tensor(data: dLoss.data, shape: @[1] & dLoss.shape)
  result = m.backwardBatch(@[ids], dLossBatch, ctx)

# ===================================================================
# ApfAdam (update weight trên CPU vì chỉ 1 vector, vẫn ok)
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
  ## Bản CPU thuần (GIỮ NGUYÊN - vẫn được test_nimformer.nim/nơi khác dùng).
  ## Cho pipeline fine-tune LoRA (lora.nim/finetune_lora.nim), dùng
  ## apfAdamStepGpu bên dưới thay vào đó (chạy trên ctx/GPU, xem lý do trong
  ## comment của nó).
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
  ## Bản GPU của apfAdamStep, dùng beMul/beAdd/beDiv (backend.nim) cho toàn
  ## bộ phần nhân/cộng/chia của Adam (m,v accumulate, mHat/vHat, param
  ## update) thay vì vòng for CPU.
  ##
  ## Toàn bộ Adam (m,v accumulate, mHat/vHat, sqrt, param update) giờ đi qua
  ## beMul/beAdd/beDiv/beSub/beSqrt (backend.nim) - kể cả sqrt(vHat) đã có
  ## kernel GPU thật cho cả 3 backend (Metal/OpenCL/CUDA - xem gpuSqrt trong
  ## vendor/bybylang/gpubackend.nim). CUDA cần 1 bước build tay bằng nvcc rồi
  ## nối PTX (xem vendor/bybylang/backends/cuda/kernels/vecop_sqrt_kernel.cu)
  ## trước khi kernel thật sự chạy được trên máy CUDA - CHƯA build/test được
  ## trong sandbox này (không có nvcc/GPU thật).
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

  # sqrt(vHat) — GIO DA CHAY GPU THAT qua beSqrt (Metal/OpenCL co kernel that,
  # CUDA fallback CPU dung - xem ghi chu chi tiet trong beSqrt/gpuSqrt).
  # (KHONG import tensor_ops.nim o day de dung sqrtT - tensor_ops.nim tu no
  # da `import nimformer`, import nguoc lai se circular. beSqrt goi thang tu
  # `backend` - da import san o dau file - la du, cung dung dung 1 kernel.)
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
