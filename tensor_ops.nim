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

import std/[math, random]
import backend
import nimformer  # Tensor, numel-style helpers (flatten2D, newTensor, ...)

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

proc permute*(x: Tensor, dims: seq[int]): Tensor =
  doAssert dims.len == x.shape.len, "permute: số chiều dims phải khớp shape"
  var newShape = newSeq[int](dims.len)
  for i, d in dims: newShape[i] = x.shape[d]
  let oldStrides = strides(x.shape)
  let newStrides = strides(newShape)
  result.shape = newShape
  result.data = newSeq[float32](x.data.len)
  var idx = newSeq[int](dims.len)
  for linear in 0 ..< x.data.len:
    # Giải mã linear index (theo layout MỚI) -> multi-index MỚI -> multi-index CŨ -> linear index CŨ.
    var rem = linear
    for i in 0 ..< dims.len:
      idx[i] = rem div newStrides[i]
      rem = rem mod newStrides[i]
    var oldLinear = 0
    for i in 0 ..< dims.len:
      oldLinear += idx[i] * oldStrides[dims[i]]
    result.data[linear] = x.data[oldLinear]

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

proc beSum*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  result = newSeq[float32](outer * inner)
  for o in 0 ..< outer:
    for i in 0 ..< inner:
      var acc = 0'f32
      for d in 0 ..< dimSize:
        acc += x[(o * dimSize + d) * inner + i]
      result[o * inner + i] = acc

proc beMean*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  result = beSum(ctx, x, shape, dim)
  let d = float32(shape[dim])
  for i in 0 ..< result.len: result[i] = result[i] / d

proc beMax*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32] =
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  result = newSeq[float32](outer * inner)
  for o in 0 ..< outer:
    for i in 0 ..< inner:
      var best = x[(o * dimSize) * inner + i]
      for d in 1 ..< dimSize:
        let v = x[(o * dimSize + d) * inner + i]
        if v > best: best = v
      result[o * inner + i] = best

proc beArgMax*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[int] =
  let (outer, dimSize, inner) = reduceShapes(shape, dim)
  result = newSeq[int](outer * inner)
  for o in 0 ..< outer:
    for i in 0 ..< inner:
      var best = x[(o * dimSize) * inner + i]
      var bestIdx = 0
      for d in 1 ..< dimSize:
        let v = x[(o * dimSize + d) * inner + i]
        if v > best: (best = v; bestIdx = d)
      result[o * inner + i] = bestIdx

proc beArgMax*(ctx: Backend, x: openArray[float32]): int =
  ## Tiện ích: argmax phẳng trên toàn bộ mảng 1D (VD logits 1 token, greedy
  ## sampling) - tránh phải bọc shape=[n], dim=0 cho trường hợp đơn giản nhất.
  doAssert x.len > 0
  result = 0
  var best = x[0]
  for i in 1 ..< x.len:
    if x[i] > best: (best = x[i]; result = i)

# ─────────────────────────────────────────────────────────────
# 5. CONCAT / STACK / SPLIT
# ─────────────────────────────────────────────────────────────

proc concat*(tensors: openArray[Tensor], dim: int): Tensor =
  doAssert tensors.len > 0, "concat: cần ít nhất 1 tensor"
  let baseShape = tensors[0].shape
  for t in tensors:
    for i, s in t.shape:
      if i != dim: doAssert s == baseShape[i], "concat: shape không khớp ngoài chiều dim"
  var newShape = baseShape
  var totalDim = 0
  for t in tensors: totalDim += t.shape[dim]
  newShape[dim] = totalDim
  var outer = 1
  for i in 0 ..< dim: outer *= baseShape[i]
  var inner = 1
  for i in dim + 1 ..< baseShape.len: inner *= baseShape[i]
  result.shape = newShape
  result.data = newSeq[float32](numelShape(newShape))
  for o in 0 ..< outer:
    var dOff = 0
    for t in tensors:
      let dSize = t.shape[dim]
      for d in 0 ..< dSize:
        for i in 0 ..< inner:
          result.data[(o * totalDim + dOff + d) * inner + i] = t.data[(o * dSize + d) * inner + i]
      dOff += dSize

proc stack*(tensors: openArray[Tensor], dim: int = 0): Tensor =
  ## stack thêm 1 chiều MỚI ở vị trí `dim` (khác concat - không ghép vào
  ## chiều có sẵn). Cài bằng cách reshape mỗi tensor thêm chiều 1 rồi concat.
  doAssert tensors.len > 0, "stack: cần ít nhất 1 tensor"
  var reshaped = newSeq[Tensor](tensors.len)
  for i, t in tensors:
    var s = t.shape
    s.insert(1, dim)
    reshaped[i] = view(t, s)
  result = concat(reshaped, dim)

proc split*(x: Tensor, splitSize: int, dim: int): seq[Tensor] =
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
    piece.data = newSeq[float32](numelShape(newShape))
    for o in 0 ..< outer:
      for d in 0 ..< thisSize:
        for i in 0 ..< inner:
          piece.data[(o * thisSize + d) * inner + i] = x.data[(o * dimTotal + offset + d) * inner + i]
    result.add(piece)
    offset += thisSize

# ─────────────────────────────────────────────────────────────
# 6. DROPOUT (CPU, dùng cho training)
# ─────────────────────────────────────────────────────────────

proc beDropout*(ctx: Backend, x: openArray[float32], p: float32, training: bool,
                 rng: var Rand): seq[float32] =
  ## Inverted dropout (giống torch.nn.Dropout): lúc eval trả nguyên x, lúc
  ## train zero ngẫu nhiên theo xác suất p rồi scale phần còn lại 1/(1-p) để
  ## kỳ vọng không đổi.
  result = newSeq[float32](x.len)
  if not training or p <= 0'f32:
    for i in 0 ..< x.len: result[i] = x[i]
    return
  doAssert p < 1'f32, "dropout: p phải < 1"
  let scale = 1'f32 / (1'f32 - p)
  for i in 0 ..< x.len:
    if rng.rand(1.0) < p.float: result[i] = 0'f32
    else: result[i] = x[i] * scale

proc dropoutMask*(n: int, p: float32, rng: var Rand): seq[float32] =
  ## Trả mask (0 hoặc 1/(1-p)) để dùng lại CHO CẢ forward và backward (nhân
  ## lại đúng mask khi lan truyền ngược, thay vì random lại lần 2 - bắt buộc
  ## với training đúng).
  result = newSeq[float32](n)
  if p <= 0'f32:
    for i in 0 ..< n: result[i] = 1'f32
    return
  let scale = 1'f32 / (1'f32 - p)
  for i in 0 ..< n:
    result[i] = if rng.rand(1.0) < p.float: 0'f32 else: scale

# ─────────────────────────────────────────────────────────────
# 7. COMPARISON (gt/lt/eq/ge/le) - dùng cho top-p sampling, masking
# ─────────────────────────────────────────────────────────────

proc beGt*(ctx: Backend, a, b: openArray[float32]): seq[bool] =
  doAssert a.len == b.len
  result = newSeq[bool](a.len)
  for i in 0 ..< a.len: result[i] = a[i] > b[i]

proc beLt*(ctx: Backend, a, b: openArray[float32]): seq[bool] =
  doAssert a.len == b.len
  result = newSeq[bool](a.len)
  for i in 0 ..< a.len: result[i] = a[i] < b[i]

proc beEq*(ctx: Backend, a, b: openArray[float32]): seq[bool] =
  doAssert a.len == b.len
  result = newSeq[bool](a.len)
  for i in 0 ..< a.len: result[i] = a[i] == b[i]

proc beGe*(ctx: Backend, a, b: openArray[float32]): seq[bool] =
  doAssert a.len == b.len
  result = newSeq[bool](a.len)
  for i in 0 ..< a.len: result[i] = a[i] >= b[i]

proc beLe*(ctx: Backend, a, b: openArray[float32]): seq[bool] =
  doAssert a.len == b.len
  result = newSeq[bool](a.len)
  for i in 0 ..< a.len: result[i] = a[i] <= b[i]

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
#    Metal/CUDA/OpenCL/TSIC chạy thật, không phải mô phỏng) thay vì vòng lặp
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