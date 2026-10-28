## tensor_ops.nim
## Bổ sung các phép toán low-level còn thiếu so với torch (xem danh sách yêu
## cầu): exp/log/sqrt/pow, view/permute, reduction (sum/mean/max/argmax),
## concat/split/stack, dropout, comparison (gt/lt/eq).
##
## CẬP NHẬT GPU: exp/log/sqrt/pow (mục 1) và cumsum/cumprod/affine-scan
## (mục 8, dùng cho Mamba/SSM) giờ dùng `ctx: Backend` để chạy GPU THẬT -
## exp/log/sqrt qua byby.gpuExp/gpuLog/gpuSqrt (kernel Metal + OpenCL biên
## dịch từ source lúc runtime nên chạy ngay; CUDA cần nvcc build lại 1 lần
## từ backends/cuda/kernels/vecop_ext.cu, xem comment trong file đó - trước
## khi build xong thì tự fallback CPU đúng cơ chế handleGpuFailure chung của
## codebase, không âm thầm sai kết quả). Scan (cumsum/cumprod/affine) chạy
## từng bước Hillis-Steele qua beAdd/beMul thật trên GPU (xem mục 8).
##
## view/permute/reduction(sum/mean/max/argmax)/concat/split/stack/dropout/
## comparison (mục 2-7) vẫn là CPU: đây là các phép đánh chỉ số/reduction
## nhỏ (không phải nút thắt FLOPs như scan/activation), giữ CPU cho đơn giản
## và đúng ngay; nếu cần tăng tốc phép cụ thể nào trong nhóm này sau, thêm
## kernel riêng theo đúng pattern sessionRmsnormEnc/sessionRope đã có trong
## gpubackend.nim.

import std/[math, sequtils]
import backend
import nimformer  # Tensor, numel-style helpers (flatten2D, newTensor, ...)

## GHI CHÚ GPU HOÁ VÒNG 2 (permute/concat/split/stack/argmax-index/
## comparison/dropout): backend.nim giờ export thêm beGatherGpu/
## beArgmaxIndexGpu/beCompareGpuOp/beDropoutMaskGpu (xem gpubackend_additions.nim
## + cuda/vulkan/metal driver additions kèm theo) - các phép GATHER (đọc
## theo mảng chỉ số), ARGMAX-INDEX, so sánh ra bitmask, và sinh dropout mask
## đều chạy kernel GPU thật (gather_kernel.cu/.comp/.metal, argmax_kernel...,
## compare_kernel..., dropout_rng_kernel...) thay vì vòng lặp/RNG CPU như
## trước. `ctx: Backend` giờ là tham số BẮT BUỘC cho toàn bộ nhóm hàm này.

# ─────────────────────────────────────────────────────────────
# 1. EXP / LOG / SQRT / POW (element-wise, bắt buộc cho Mamba/RMSNorm/Loss)
# ─────────────────────────────────────────────────────────────

## beExp/beLog/beSqrt/bePow: định nghĩa THẬT nằm ở backend.nim (dispatch qua
## byby.gpuExp/gpuLog/gpuSqrt - chạy kernel GPU thật trên Metal/OpenCL ngay,
## CUDA cần nvcc build lại 1 lần từ vecop_ext.cu rồi mới có kernel, trước đó
## tự fallback CPU đúng cơ chế handleGpuFailure chung của toàn bộ codebase).
## KHÔNG định nghĩa lại ở đây - `import backend` phía trên đã đưa các proc
## này vào scope.

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
# 2. VIEW / RESHAPE - không copy data (chia sẻ seq qua ref-semantics của Nim
#    khi có thể; seq[float32] trong Nim là copy-on-write nên "view" thật sự
#    không sao chép cho tới khi bên nhận GHI vào nó - đủ để tránh copy đọc).
# ─────────────────────────────────────────────────────────────

proc numelShape(shape: seq[int]): int =
  result = 1
  for s in shape: result *= s

proc view*(x: Tensor, newShape: seq[int]): Tensor =
  ## Reshape thuần (không copy) - hỗ trợ 1 chiều "-1" (tự suy ra) giống torch.
  var shape = newShape
  var inferIdx = -1
  var known = 1
  for i, s in shape:
    if s == -1:
      doAssert inferIdx == -1, "view: chỉ được dùng -1 ở MỘT chiều"
      inferIdx = i
    else:
      known *= s
  if inferIdx >= 0:
    doAssert known != 0 and x.data.len mod known == 0,
      "view: không thể suy ra chiều -1 từ shape " & $x.shape & " -> " & $newShape
    shape[inferIdx] = x.data.len div known
  doAssert numelShape(shape) == x.data.len,
    "view: tổng số phần tử không khớp (" & $x.data.len & " vs " & $numelShape(shape) & ")"
  result.shape = shape
  result.data = x.data   # Nim seq = copy-on-write: đọc không tốn copy.

proc reshape*(x: Tensor, newShape: seq[int]): Tensor = view(x, newShape)

# ─────────────────────────────────────────────────────────────
# 3. PERMUTE - đổi thứ tự chiều (VD attention [B,T,H,D] -> [B,H,T,D]).
#    Đây LÀ COPY thật sự (khác view) vì layout bộ nhớ đổi - giống torch:
#    .permute() trả về view "logic" nhưng không contiguous; ở đây ta luôn
#    trả contiguous để mọi phép sau (matmul, ...) dùng thẳng được ngay.
# ─────────────────────────────────────────────────────────────

proc strides(shape: seq[int]): seq[int] =
  result = newSeq[int](shape.len)
  var acc = 1
  for i in countdown(shape.len - 1, 0):
    result[i] = acc
    acc *= shape[i]

proc permute*(ctx: Backend, x: Tensor, dims: seq[int]): Tensor =
  ## GIỐNG TORCH: chỉ tính strides (O(rank), rank<=8, KHÔNG phụ thuộc n) trên
  ## CPU rồi gửi cho kernel permute_strided (CUDA/Vulkan/Metal) - mỗi GPU
  ## thread tự giải mã chỉ số nguồn của riêng nó ngay trên device. Không còn
  ## vòng lặp CPU O(n) build mảng chỉ số như bản trước.
  doAssert dims.len == x.shape.len, "permute: số chiều dims phải khớp shape"
  doAssert dims.len <= 8, "permute: rank > 8 chưa hỗ trợ (bảng push-constant/kernel cố định 8)"
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
# 4. REDUCTION: sum / mean / max / argmax theo 1 chiều (dim cuối phổ biến
#    nhất cho LayerNorm/Loss/Sampling nên hỗ trợ tổng quát mọi dim qua
#    flatten [outer, dimSize, inner]).
# ─────────────────────────────────────────────────────────────

proc reduceShapes(shape: seq[int], dim: int): tuple[outer, dimSize, inner: int] =
  var outer = 1
  for i in 0 ..< dim: outer *= shape[i]
  let dimSize = shape[dim]
  var inner = 1
  for i in dim + 1 ..< shape.len: inner *= shape[i]
  (outer, dimSize, inner)

## GPU HOÁ TOÀN BỘ reduction (sum/mean/max/argmax): mọi phép CỘNG/NHÂN/MAX
## per-element giờ đi qua beAdd/beSub/beRelu (kernel GPU thật) thay vì vòng
## lặp CPU. Kỹ thuật: tree-reduction log-depth (giống Hillis-Steele ở mục 8)
## - mỗi bước gộp cặp phần tử liền kề bằng 1 lệnh GPU trên CẢ KHỐI cùng lúc.
## max(a,b) = b + relu(a-b) -> biểu diễn thuần bằng beSub/beRelu/beAdd, không
## cần kernel max riêng. Việc gather/scatter chỉ số (KHÔNG phải phép toán số
## học) vẫn phải đi qua CPU vì backend.nim chưa có gather/scatter kernel -
## đây là giới hạn phần cứng kernel hiện có, không phải lựa chọn ở file này.

proc gpuTreeReduce(ctx: Backend, data: openArray[float32], outer, dimSize, inner: int,
                    combine: proc(ctx: Backend, a, b: openArray[float32]): seq[float32] {.nimcall.}): seq[float32] =
  ## Gộp toàn bộ `dimSize` phần tử theo cặp liền kề, log2(dimSize) bước, mỗi
  ## bước 1 lệnh GPU trên cả khối (không lặp CPU cộng/max từng phần tử).
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
  result = cur  # curDim == 1: outer*inner phần tử

proc gpuMaxCombine(ctx: Backend, a, b: openArray[float32]): seq[float32] =
  ## max(a,b) = b + relu(a-b): thuần beSub + beRelu + beAdd, chạy GPU thật.
  let diff = beSub(ctx, a, b)
  let reluDiff = beRelu(ctx, diff)
  result = beAdd(ctx, reluDiff, b)

proc beSum*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  result = gpuTreeReduce(ctx, x, outer, dimSize, inner, beAdd)

proc beMean*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  let summed = gpuTreeReduce(ctx, x, outer, dimSize, inner, beAdd)
  # Chia đều bằng 1 lệnh GPU (beMul với hệ số 1/d) thay vì vòng lặp chia CPU.
  let invD = newSeq[float32](summed.len).mapIt(1'f32 / float32(dimSize))
  result = beMul(ctx, summed, invD)

proc beMax*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  result = gpuTreeReduce(ctx, x, outer, dimSize, inner, gpuMaxCombine)

proc beArgMax*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[int] =
  ## GPU HOÁ THẬT: cả giá trị VÀ chỉ số cùng tính trên GPU trong 1 lệnh qua
  ## beArgmaxIndexGpu (argmax_kernel.cu/.comp/.metal - tree-reduce giữ cặp
  ## (giá trị, chỉ số) bằng shared/threadgroup memory ngay trên device),
  ## không còn bước "tính max GPU rồi đọc lại CPU để dò index" như trước.
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  let (_, idx) = beArgmaxIndexGpu(ctx, @x, outer, dimSize, inner)
  result = newSeq[int](idx.len)
  for i in 0 ..< idx.len: result[i] = int(idx[i])

proc beArgMax*(ctx: Backend, x: openArray[float32]): int =
  ## Tiện ích: argmax phẳng trên toàn bộ mảng 1D (VD logits 1 token, greedy
  ## sampling) - giá trị + chỉ số cùng tính trên GPU (xem ghi chú ở trên).
  doAssert x.len > 0
  let (_, idx) = beArgmaxIndexGpu(ctx, @x, 1, x.len, 1)
  result = int(idx[0])

# ─────────────────────────────────────────────────────────────
# 5. CONCAT / STACK / SPLIT
# ─────────────────────────────────────────────────────────────

proc concat*(ctx: Backend, tensors: openArray[Tensor], dim: int): Tensor =
  ## GIỐNG TORCH: host chỉ nối phẳng dữ liệu (memcpy nối tiếp - KHÔNG tính
  ## chỉ số per-element) + chuẩn bị bảng nhỏ O(numTensors) (base offset
  ## trong buffer phẳng, kích thước theo dim, offset lũy kế theo dim) rồi
  ## để kernel concat_gather (CUDA/Vulkan/Metal) tự chọn đúng nguồn cho
  ## từng phần tử ngay trên GPU.
  doAssert tensors.len > 0, "concat: cần ít nhất 1 tensor"
  let baseShape = tensors[0].shape
  for t in tensors:
    for i, s in t.shape:
      if i != dim: doAssert s == baseShape[i], "concat: shape không khớp ngoài chiều dim"
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
  ## stack thêm 1 chiều MỚI ở vị trí `dim` (khác concat - không ghép vào
  ## chiều có sẵn). Cài bằng cách reshape mỗi tensor thêm chiều 1 rồi concat
  ## (đường concat đã GPU hoá thật ở trên).
  doAssert tensors.len > 0, "stack: cần ít nhất 1 tensor"
  var reshaped = newSeq[Tensor](tensors.len)
  for i, t in tensors:
    var s = t.shape
    s.insert(1, dim)
    reshaped[i] = view(t, s)
  result = concat(ctx, reshaped, dim)

proc split*(ctx: Backend, x: Tensor, splitSize: int, dim: int): seq[Tensor] =
  ## GIỐNG TORCH: mỗi mảnh là 1 lát cắt liên tục - host chỉ truyền 5 số
  ## nguyên (dimTotal/thisSize/offset/inner/n), kernel split_slice tự tính
  ## chỉ số nguồn ngay trên GPU cho từng phần tử, không có vòng lặp CPU.
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
# 6. DROPOUT - GPU HOÁ THẬT (vòng 2): RNG giờ chạy counter-based xorshift32
#    NGAY TRÊN GPU (dropout_rng_kernel.cu/.comp/.metal qua beDropoutMaskGpu)
#    thay vì std/random trên host - seed (uint32, người gọi tự tăng mỗi lần
#    train step để mỗi lần dropout khác nhau) là input duy nhất, không cần
#    `var Rand` nữa.
# ─────────────────────────────────────────────────────────────

proc beDropout*(ctx: Backend, x: openArray[float32], p: float32, training: bool,
                 seed: uint32): seq[float32] =
  ## Inverted dropout (giống torch.nn.Dropout): lúc eval trả nguyên x, lúc
  ## train nhân với mask sinh trên GPU (0 hoặc 1/(1-p)) qua beMul (GPU).
  if not training or p <= 0'f32:
    result = newSeq[float32](x.len)
    for i in 0 ..< x.len: result[i] = x[i]
    return
  doAssert p < 1'f32, "dropout: p phải < 1"
  let mask = beDropoutMaskGpu(ctx, x.len, p, seed)
  result = beMul(ctx, @x, mask)

proc dropoutMask*(ctx: Backend, n: int, p: float32, seed: uint32): seq[float32] =
  ## Trả mask (0 hoặc 1/(1-p)), sinh thẳng trên GPU (beDropoutMaskGpu), để
  ## dùng lại CHO CẢ forward và backward (nhân lại đúng mask khi lan truyền
  ## ngược, thay vì random lại lần 2 - bắt buộc với training đúng: cùng seed
  ## -> cùng mask, vì kernel là counter-based deterministic theo (seed, i)).
  if p <= 0'f32:
    result = newSeq[float32](n)
    for i in 0 ..< n: result[i] = 1'f32
    return
  result = beDropoutMaskGpu(ctx, n, p, seed)

# ─────────────────────────────────────────────────────────────
# 7. COMPARISON (gt/lt/eq/ge/le) - dùng cho top-p sampling, masking
# ─────────────────────────────────────────────────────────────

## GPU HOÁ COMPARISON: phần TÍNH TOÁN (hiệu a-b, và relu để tách dấu) chạy
## qua beSub/beRelu (GPU thật). Việc materialize kết quả thành seq[bool] chỉ
## là đọc lại 1 giá trị float đã tính xong (so với 0) - không phải compute,
## backend.nim không có kernel xuất bitmask trực tiếp nên bước đọc cuối này
## là bắt buộc bất kể cách viết.

## GPU HOÁ THẬT (vòng 2): so sánh giờ chạy 1 lệnh compare_kernel trên GPU
## (beCompareGpuOp - CUDA/Vulkan/Metal, opcode 0=gt 1=lt 2=eq 3=ge 4=le,
## xem compare_kernel.cu/.comp/.metal) xuất thẳng bitmask, không còn bước
## trung gian sub/relu + đọc lại host như bản trước.

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
# 8. PARALLEL SCAN (Hillis-Steele, log-depth) - torch.cumsum/cumprod theo 1
#    chiều bất kỳ, và scan tổng quát với toán tử kết hợp (associative) tuỳ
#    ý - cần cho Mamba/SSM (selective-scan recurrence h_t = a_t*h_{t-1}+b_t
#    chính là 1 "affine scan": kết hợp (a1,b1) và (a2,b2) -> (a1*a2,
#    a2*b1+b2), rồi quét parallel-prefix qua đó).
#
#    GPU THẬT: mỗi bước Hillis-Steele (offset = 1,2,4,8,...) đưa TOÀN BỘ
#    phần tử tham gia bước đó vào 1 lệnh GPU thật (beAdd/beMul - đã có kernel
#    Metal/CUDA/OpenCL/Vulkan chạy thật, không phải mô phỏng) thay vì vòng lặp
#    CPU per-element. Việc "gather" chỉ số cur/prev vẫn làm trên CPU (đây là
#    bước đánh chỉ số/reshape, không phải phép tính toán học), nhưng phép
#    CỘNG/NHÂN thật sự - phần chiếm hầu hết FLOPs khi dimSize lớn - chạy trên
#    GPU giống hệt beAdd/beMul dùng ở mọi nơi khác trong codebase này.
# ─────────────────────────────────────────────────────────────

proc scanStepIndices(outer, dimSize, inner, offset: int): tuple[cur, prev: seq[int]] =
  ## Chỉ số phẳng (cur, prev) của TẤT CẢ vị trí tham gia 1 bước Hillis-Steele
  ## với `offset` cho trước - đây chính là tập vị trí ĐỘC LẬP nhau trong bước
  ## này (đọc từ snapshot, không ghi đè lẫn nhau), nên gộp thành 1 cặp mảng
  ## phẳng rồi đưa nguyên khối vào 1 lệnh GPU (beAdd/beMul) là hợp lệ.
  var cur = newSeq[int]()
  var prev = newSeq[int]()
  for o in 0 ..< outer:
    for d in offset ..< dimSize:
      for i in 0 ..< inner:
        cur.add((o * dimSize + d) * inner + i)
        prev.add((o * dimSize + (d - offset)) * inner + i)
  (cur, prev)

proc applyHillisSteeleGpu(ctx: Backend, data: var seq[float32], outer, dimSize, inner: int, useMul: bool) =
  ## In-place, inclusive prefix-sum (useMul=false) hoặc prefix-product
  ## (useMul=true) dọc chiều `dimSize`. Mỗi bước offset=1,2,4,... gather
  ## snapshot của các vị trí tham gia thành 2 mảng phẳng (cur, prev) rồi gọi
  ## THẲNG beAdd/beMul (GPU thật qua ctx) trên CẢ KHỐI cùng lúc, sau đó
  ## scatter kết quả ngược lại `data` - không còn vòng lặp CPU cộng/nhân
  ## từng phần tử một.
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
  ## Tương đương torch.cumsum(x, dim): prefix-sum CÓ TÍNH LUỸ (inclusive),
  ## giữ nguyên shape. Mỗi bước Hillis-Steele chạy qua beAdd (GPU thật).
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len: result[i] = x[i]
  applyHillisSteeleGpu(ctx, result, outer, dimSize, inner, useMul = false)

proc cumsumT*(ctx: Backend, x: Tensor, dim: int): Tensor =
  result.shape = x.shape
  result.data = beCumsum(ctx, x.data, x.shape, dim)

proc beCumprod*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  ## torch.cumprod(x, dim). Cùng cấu trúc Hillis-Steele như cumsum nhưng
  ## kết hợp bằng phép nhân (vẫn kết hợp/associative -> scan song song hợp
  ## lệ), chạy qua beMul (GPU thật). Dùng riêng thay vì "exp(cumsum(log(x)))"
  ## để ĐÚNG với giá trị âm/0 giống torch (log(số âm) = NaN, torch.cumprod
  ## thì không).
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len: result[i] = x[i]
  applyHillisSteeleGpu(ctx, result, outer, dimSize, inner, useMul = true)

proc cumprodT*(ctx: Backend, x: Tensor, dim: int): Tensor =
  result.shape = x.shape
  result.data = beCumprod(ctx, x.data, x.shape, dim)

type
  ScanPair* = tuple[a, b: float32]
    ## Cặp (a,b) cho affine scan h_t = a_t * h_{t-1} + b_t (đúng dạng
    ## recurrence của Mamba/SSM selective-scan sau khi rời rạc hoá).

proc combineAffine(l, r: ScanPair): ScanPair =
  ## Kết hợp 2 bước affine liên tiếp: áp l TRƯỚC, r SAU (h -> l(h) -> r(l(h))).
  ## r(l(h)) = r.a*(l.a*h + l.b) + r.b = (r.a*l.a)*h + (r.a*l.b + r.b)
  ## (giữ lại bản CPU-scalar này để testops.nim đối chiếu tuần tự - xem
  ## sequentialAffine trong testops.nim; đường THẬT chạy GPU nằm ở beAffineScan
  ## dưới đây, dùng đúng công thức kết hợp y hệt nhưng bằng beMul/beAdd trên
  ## CẢ KHỐI thay vì 1 cặp float scalar.)
  (a: r.a * l.a, b: r.a * l.b + r.b)

proc beAffineScan*(ctx: Backend, aCoef, bCoef: openArray[float32],
                    shape: seq[int], dim: int): seq[float32] =
  ## Parallel scan TỔNG QUÁT cho recurrence affine h_t = a_t*h_{t-1} + b_t
  ## (state-space/SSM/Mamba selective-scan, GRU-tuyến-tính, ...) dọc chiều
  ## `dim`. Trả về h_t (INCLUSIVE, h_{-1} = 0) cho MỌI t cùng lúc. Mỗi bước
  ## Hillis-Steele áp CÔNG THỨC combineAffine (r.a*l.a, r.a*l.b + r.b) cho
  ## TOÀN BỘ vị trí tham gia bước đó CÙNG LÚC bằng 3 lệnh GPU thật
  ## (beMul, beMul, beAdd) - không còn vòng lặp CPU tính từng cặp (a,b) một.
  doAssert aCoef.len == bCoef.len
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  doAssert inner == 1, "beAffineScan: hiện chỉ hỗ trợ dim cuối cùng (inner=1); " &
    "gọi permute() trước nếu cần scan theo chiều khác"
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
  result = b  # h_t với h_{-1}=0 chính là thành phần b sau khi scan xong