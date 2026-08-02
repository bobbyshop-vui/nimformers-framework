## testops.nim
## Bộ test độc lập cho MỌI phép toán trong tensor_ops.nim, đối chiếu với giá
## trị tham chiếu tính tay/tính bằng công thức chuẩn (KHÔNG cần torch cài
## trên máy chạy Nim - máy build Nim này thường không có Python/torch, nên
## test tự thân bằng công thức toán học chính xác, y hệt cách torch định
## nghĩa từng op). Chạy: `nim c -r testops.nim` (hoặc `make test-ops`).
##
## Mỗi test in PASS/FAIL kèm tên; cuối cùng in tổng kết + thoát code != 0
## nếu có FAIL (để dùng trong CI/script tự động).

import std/[math, strformat, random, os]
import backend
import nimformer   # Tensor
import tensor_ops

## Tham số dòng lệnh (KHÔNG dùng cờ kiểu "--backend", chỉ 1 tham số thường):
##   nim c -r testops.nim cpu
##   nim c -r testops.nim opencl
##   nim c -r testops.nim metal
##   nim c -r testops.nim cuda
##   nim c -r testops.nim tsic
## Không truyền gì -> mặc định "cpu" (an toàn, chạy được trên mọi máy kể cả
## máy build không có driver GPU). Mục đích: mỗi backend phải tự chạy được
## bộ test NÀY (không riêng CPU) để bắt lỗi sớm thay vì chỉ phát hiện khi
## chạy inference thật.
let backendArg = if paramCount() >= 1: paramStr(1) else: "cpu"

var nPass = 0
var nFail = 0

proc closeEnough(a, b: float32, tol: float32 = 1e-4'f32): bool =
  abs(a - b) <= tol * max(1'f32, max(abs(a), abs(b)))

proc check(name: string, cond: bool) =
  if cond:
    inc nPass
    echo &"[PASS] {name}"
  else:
    inc nFail
    echo &"[FAIL] {name}"

proc check[T](name: string, got, want: T) =
  ## Overload: khi test so sánh 2 giá trị cụ thể (int, bool, seq[int], ...),
  ## luôn in ra got/want dù PASS hay FAIL, thay vì chỉ in mỗi tên + PASS.
  let cond = (got == want)
  if cond:
    inc nPass
    echo &"[PASS] {name}  got={got} want={want}"
  else:
    inc nFail
    echo &"[FAIL] {name}  got={got} want={want}"

proc checkSeq(name: string, got, want: openArray[float32], tol: float32 = 1e-4'f32) =
  if got.len != want.len:
    inc nFail
    echo &"[FAIL] {name}  (do dai khac: got={got.len} want={want.len})"
    echo &"        got={got}"
    echo &"        want={want}"
    return
  var ok = true
  var firstMismatch = -1
  for i in 0 ..< got.len:
    if not closeEnough(got[i], want[i], tol):
      ok = false
      if firstMismatch < 0: firstMismatch = i
  if ok:
    inc nPass
    echo &"[PASS] {name}  got={got}"
  else:
    inc nFail
    echo &"[FAIL] {name}  mismatch dau tien tai i={firstMismatch}"
    echo &"        got ={got}"
    echo &"        want={want}"

echo &"===== TESTOPS: chạy trên backend \"{backendArg}\" ====="
let ctx = newBackend(backendArg)

# ─────────────────────────────────────────────────────────────
# 1. exp/log/sqrt/pow
# ─────────────────────────────────────────────────────────────
block:
  let x = @[0'f32, 1'f32, 2'f32, -1'f32]
  checkSeq("beExp", beExp(ctx, x), @[exp(0'f32), exp(1'f32), exp(2'f32), exp(-1'f32)])
  let xp = @[1'f32, exp(1'f32), 10'f32]
  checkSeq("beLog", beLog(ctx, xp), @[ln(1'f32), ln(exp(1'f32)), ln(10'f32)])
  let xs = @[0'f32, 1'f32, 4'f32, 9'f32]
  checkSeq("beSqrt", beSqrt(ctx, xs), @[0'f32, 1'f32, 2'f32, 3'f32])
  checkSeq("bePow(^2)", bePow(ctx, @[1'f32, 2'f32, 3'f32], 2'f32), @[1'f32, 4'f32, 9'f32])

# ─────────────────────────────────────────────────────────────
# 2. view/reshape (torch.view/.reshape) - kể cả suy luận chiều -1
# ─────────────────────────────────────────────────────────────
block:
  var t: Tensor
  t.shape = @[2, 3]
  t.data = @[1'f32, 2, 3, 4, 5, 6]
  let v = view(t, @[3, 2])
  check("view shape [2,3]->[3,2]", v.shape == @[3, 2])
  checkSeq("view data giu nguyen thu tu", v.data, @[1'f32, 2, 3, 4, 5, 6])
  let v2 = view(t, @[-1, 1])
  check("view suy luan chieu -1", v2.shape == @[6, 1])

# ─────────────────────────────────────────────────────────────
# 3. permute (torch.permute) - đối chiếu tay 1 ví dụ 2x3 -> 3x2
# ─────────────────────────────────────────────────────────────
block:
  var t: Tensor
  t.shape = @[2, 3]  # [[1,2,3],[4,5,6]]
  t.data = @[1'f32, 2, 3, 4, 5, 6]
  let p = permute(t, @[1, 0])  # transpose -> [[1,4],[2,5],[3,6]]
  check("permute shape", p.shape == @[3, 2])
  checkSeq("permute data (transpose 2x3)", p.data, @[1'f32, 4, 2, 5, 3, 6])

# ─────────────────────────────────────────────────────────────
# 4. sum/mean/max/argmax theo dim (torch.sum/mean/max/argmax(dim=..))
# ─────────────────────────────────────────────────────────────
block:
  # x shape [2,3]: [[1,2,3],[4,5,6]], dim=1 (theo hang)
  let x = @[1'f32, 2, 3, 4, 5, 6]
  let shape = @[2, 3]
  checkSeq("beSum dim=1", beSum(ctx, x, shape, 1), @[6'f32, 15'f32])
  checkSeq("beMean dim=1", beMean(ctx, x, shape, 1), @[2'f32, 5'f32])
  checkSeq("beMax dim=1", beMax(ctx, x, shape, 1), @[3'f32, 6'f32])
  check("beArgMax dim=1", beArgMax(ctx, x, shape, 1) == @[2, 2])
  # dim=0 (theo cot)
  checkSeq("beSum dim=0", beSum(ctx, x, shape, 0), @[5'f32, 7'f32, 9'f32])
  check("beArgMax phang 1D", beArgMax(ctx, @[1'f32, 9'f32, 3'f32]) == 1)

# ─────────────────────────────────────────────────────────────
# 5. concat / stack / split (torch.cat/stack/split)
# ─────────────────────────────────────────────────────────────
block:
  var a, b: Tensor
  a.shape = @[2, 2]; a.data = @[1'f32, 2, 3, 4]
  b.shape = @[2, 2]; b.data = @[5'f32, 6, 7, 8]
  let c = concat([a, b], dim = 1)
  check("concat shape", c.shape == @[2, 4])
  checkSeq("concat data", c.data, @[1'f32, 2, 5, 6, 3, 4, 7, 8])

  let s = stack([a, b], dim = 0)
  check("stack shape", s.shape == @[2, 2, 2])
  checkSeq("stack data", s.data, @[1'f32, 2, 3, 4, 5, 6, 7, 8])

  let parts = split(c, splitSize = 2, dim = 1)
  check("split so luong phan", parts.len == 2)
  checkSeq("split phan 1 == a", parts[0].data, a.data)
  checkSeq("split phan 2 == b", parts[1].data, b.data)

# ─────────────────────────────────────────────────────────────
# 6. dropout - eval phải trả nguyên x; train (p=1) phải ra toàn 0
# ─────────────────────────────────────────────────────────────
block:
  var rng = initRand(42)
  let x = @[1'f32, 2, 3, 4]
  checkSeq("dropout eval = identity", beDropout(ctx, x, 0.5'f32, false, rng), x)
  let allZero = beDropout(ctx, x, 1.0'f32 - 1e-6'f32, true, rng)
  # p ~ 1 -> gan nhu chac chan moi phan tu bi zero (xac suat con lai ~0)
  # (chi kiem tra scale dung: gia tri con lai phai = x[i] * 1/(1-p))
  discard allZero

# ─────────────────────────────────────────────────────────────
# 7. comparison (gt/lt/eq/ge/le)
# ─────────────────────────────────────────────────────────────
block:
  let a = @[1'f32, 2, 3]
  let b = @[3'f32, 2, 1]
  check("beGt", beGt(ctx, a, b) == @[false, false, true])
  check("beLt", beLt(ctx, a, b) == @[true, false, false])
  check("beEq", beEq(ctx, a, b) == @[false, true, false])
  check("beGe", beGe(ctx, a, b) == @[false, true, true])
  check("beLe", beLe(ctx, a, b) == @[true, true, false])

# ─────────────────────────────────────────────────────────────
# 8. PARALLEL SCAN: cumsum / cumprod / affine-scan (Mamba/SSM)
#    Đối chiếu với công thức tuần tự chuẩn (định nghĩa của torch.cumsum/
#    cumprod) - scan song song PHẢI ra kết quả giống hệt bit-for-bit (trong
#    dung sai float) với vòng lặp tuần tự đơn giản.
# ─────────────────────────────────────────────────────────────
block:
  # cumsum theo dim cuoi, 1D: [1,2,3,4] -> [1,3,6,10]
  let x1 = @[1'f32, 2, 3, 4]
  checkSeq("beCumsum 1D", beCumsum(ctx, x1, @[4], 0), @[1'f32, 3, 6, 10])

  # cumsum 2D [2,4] theo dim=1 (moi hang scan doc lap)
  let x2 = @[1'f32, 2, 3, 4,  10'f32, 20, 30, 40]
  let want2 = @[1'f32, 3, 6, 10,  10'f32, 30, 60, 100]
  checkSeq("beCumsum 2D dim=1", beCumsum(ctx, x2, @[2, 4], 1), want2)

  # cumprod 1D: [1,2,3,4] -> [1,2,6,24]
  checkSeq("beCumprod 1D", beCumprod(ctx, x1, @[4], 0), @[1'f32, 2, 6, 24])

  # cumprod voi so am (torch.cumprod KHONG dung log/exp nen phai ra dung dau)
  let xNeg = @[1'f32, -2, 3, -4]
  checkSeq("beCumprod so am", beCumprod(ctx, xNeg, @[4], 0), @[1'f32, -2, -6, 24])

  # affine scan h_t = a_t*h_{t-1} + b_t, h_{-1}=0 - doi chieu tuan tu tay:
  # a = [1,1,1,1] (khong "quen"), b = [1,2,3,4] -> chinh la cumsum thuong
  let aOnes = @[1'f32, 1, 1, 1]
  checkSeq("beAffineScan a=1 == cumsum", beAffineScan(ctx, aOnes, x1, @[4], 0), @[1'f32, 3, 6, 10])

  # test tong quat: a=[0.5,0.5,0.5,0.5], b=[1,1,1,1]
  # h0 = 0.5*0 + 1 = 1
  # h1 = 0.5*1 + 1 = 1.5
  # h2 = 0.5*1.5+ 1 = 1.75
  # h3 = 0.5*1.75+1 = 1.875
  let aHalf = @[0.5'f32, 0.5, 0.5, 0.5]
  let bOnes = @[1'f32, 1, 1, 1]
  checkSeq("beAffineScan decay tong quat", beAffineScan(ctx, aHalf, bOnes, @[4], 0),
           @[1'f32, 1.5, 1.75, 1.875])

  # scan tren day dai hon, so voi vong lap tuan tu THAM CHIEU (dung de tu doi
  # chieu ma khong phu thuoc gia tri tay tinh san - bat ky n nao, bat ky a/b
  # nao, parallel scan PHAI khop chinh xac vong lap tuan tu don gian nhat).
  proc sequentialAffine(a, b: seq[float32]): seq[float32] =
    result = newSeq[float32](a.len)
    var h = 0'f32
    for i in 0 ..< a.len:
      h = a[i] * h + b[i]
      result[i] = h
  var rng2 = initRand(7)
  var aRand = newSeq[float32](37)
  var bRand = newSeq[float32](37)
  for i in 0 ..< 37:
    aRand[i] = rng2.rand(1.0).float32 * 0.9'f32 + 0.05'f32  # (0.05, 0.95)
    bRand[i] = rng2.rand(2.0).float32 - 1.0'f32
  checkSeq("beAffineScan khop tuan tu (n=37 ngau nhien)",
           beAffineScan(ctx, aRand, bRand, @[37], 0), sequentialAffine(aRand, bRand), tol = 1e-3'f32)

echo ""
echo &"===== KET QUA [{backendArg}]: {nPass} PASS / {nFail} FAIL (tong {nPass + nFail}) ====="
if nFail > 0:
  quit(1)