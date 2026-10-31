## quant.nim
## Lượng tử hoá tensor cho Nimformer: int8, int4, fp8 (e4m3/e5m2), và "auto"
## (APF tự build CustomFloat theo tensor — dùng lại customfloat.nim).
## Kèm định dạng file nhị phân đơn giản để lưu/tải state dict đã lượng tử hoá.
##
## Không có main/test ở đây — đây là thư viện thuần, import vào chỗ khác dùng.

import std/math
import mapped_weights
export mapped_weights
import customfloat

type
  QuantKind* = enum
    qkFp32Raw   ## không nén, giữ nguyên float32 (dùng cho bias/layernorm — nhạy sai số)
    qkInt8      ## symmetric int8, scale = max(|x|)/127
    qkInt4      ## symmetric int4 (pack 2 giá trị/byte), scale = max(|x|)/7
    qkInt4Asymmetric  ## THÊM: asymmetric int4 (Q4_K_M style), scale=(max-min)/15, zero_point
    qkFp8E4M3   ## CustomFloat(4,3) — dùng customfloat.nim
    qkFp8E5M2   ## CustomFloat(5,2)
    qkCustom    ## CustomFloat tuỳ ý do người dùng khai (bao nhiêu bit cũng được)
    qkAuto      ## APF: tự build CustomFloat theo chính tensor (buildCustomDtypeForTensor)
    qkFp4KM     ## MỚI — "Q4_K_M nhưng FP4": E2M1 CHUẨN OCP (không phí mã cho
                ## inf/nan, round-to-nearest) + 1 scale FP8-E4M3 cho MỖI NHÓM 32
                ## phần tử + 1 scale FP32 per-tensor. KHÔNG có zero_point vì FP4
                ## đã có bit dấu -> rẻ hơn Q4_K_M đúng 0.25 bit/weight.
                ## 4 + 8/32 = 4.25 bit/weight. Xem quantizeFp4KM.

  QuantTensor* = object
    kind*: QuantKind
    shape*: seq[int]
    scale*: seq[float32]      ## THÊM: mảng - len=1 (per-tensor), len=shape[0] (per-row),
                              ## hoặc len=shape[0]*ceil(shape[1]/groupSize) (per-group, xem groupSize)
    zero_point*: seq[float32] ## THÊM: mảng - song song với scale
    groupSize*: int      ## THÊM: 0 = per-tensor/per-row (hành vi cũ, giữ nguyên tương thích).
                         ## >0 = per-group: mỗi hàng chia thành ceil(nCols/groupSize) group,
                         ## mỗi group 1 scale/zero_point riêng — mịn gần bằng GPTQ gốc (group=128)
                         ## thay vì 1 scale cho NGUYÊN 1 hàng 4096 cột (--q4km cũ, sai số quá lớn
                         ## do double-quantization thô — xem quantize_int4_asymmetric_per_group
                         ## bên export_to_nimq.py).
    cf*: CustomFloat     ## dùng cho qkFp8E4M3/E5M2/qkCustom/qkAuto
    mappedData*: MappedView[uint8]
    data*: seq[uint8]

template bytes*(qt: QuantTensor): untyped =
  weightValues(qt.data, qt.mappedData)

proc clearPayload*(qt: var QuantTensor) =
  qt.data = @[]
  qt.mappedData = default(MappedView[uint8])

proc numelQ(shape: seq[int]): int =
  result = 1
  for s in shape: result *= s

# ─────────────────────────────────────────────────────────────
# int8 / int4 — symmetric quantization thuần tay, không cần CustomFloat
# ─────────────────────────────────────────────────────────────

proc maxAbs(arr: openArray[float32]): float32 =
  result = 0'f32
  for v in arr:
    let a = abs(v)
    if a > result: result = a

proc quantizeInt8*(arr: openArray[float32], shape: seq[int]): QuantTensor =
  let m = maxAbs(arr)
  let scale = if m == 0'f32: 1'f32 else: m / 127'f32
  result.kind = qkInt8
  result.shape = shape
  result.scale = @[scale]
  result.zero_point = @[0'f32]
  result.data = newSeq[uint8](arr.len)
  for i, v in arr:
    let q = clamp(int(round(v / scale)), -127, 127)
    result.data[i] = cast[uint8](int8(q))

proc dequantizeInt8*(qt: QuantTensor): seq[float32] =
  let sc = qt.scale[0]
  result = newSeq[float32](qt.bytes.len)
  for i, b in qt.bytes:
    result[i] = float32(cast[int8](b)) * sc

proc quantizeInt4*(arr: openArray[float32], shape: seq[int]): QuantTensor =
  let m = maxAbs(arr)
  let scale = if m == 0'f32: 1'f32 else: m / 7'f32
  result.kind = qkInt4
  result.shape = shape
  result.scale = @[scale]
  result.zero_point = @[0'f32]
  let n = arr.len
  result.data = newSeq[uint8]((n + 1) div 2)
  for i in 0 ..< n:
    let q = clamp(int(round(arr[i] / scale)), -7, 7)
    let nibble = uint8(q and 0x0F)
    let byteIdx = i div 2
    if i mod 2 == 0:
      result.data[byteIdx] = (result.data[byteIdx] and 0xF0'u8) or nibble
    else:
      result.data[byteIdx] = (result.data[byteIdx] and 0x0F'u8) or (nibble shl 4)

proc dequantizeInt4*(qt: QuantTensor): seq[float32] =
  ## THÊM LUT: path nay (khong phai Q4_K_M/asymmetric-per-group, xem
  ## dequantizeInt4Asymmetric ben duoi da co LUT roi) dung 1 scale DUY
  ## NHAT cho ca tensor -> LUT chi can xay 1 LAN cho toan bo ham (khong
  ## phai xay lai theo group nhu ban asymmetric), re hon nua. Truoc day
  ## moi phan tu tu lam (signed-8)*sc rieng - gio chi 16 phep tinh 1 lan
  ## roi tra bang cho toan bo n phan tu.
  let sc = qt.scale[0]
  let n = numelQ(qt.shape)
  result = newSeq[float32](n)
  var lut: array[16, float32]
  for k in 0 ..< 16:
    var signed = k
    if signed >= 8: signed -= 16
    lut[k] = float32(signed) * sc
  for i in 0 ..< n:
    let byteIdx = i div 2
    var nibble: uint8
    if i mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
    else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
    result[i] = lut[int(nibble)]

# ─────────────────────────────────────────────────────────────
# THÊM: asymmetric int4 (Q4_K_M style)
# ─────────────────────────────────────────────────────────────

proc quantizeInt4Asymmetric*(arr: openArray[float32], shape: seq[int]): QuantTensor =
  var minVal = float32.high
  var maxVal = -float32.high
  for v in arr:
    if v < minVal: minVal = v
    if v > maxVal: maxVal = v
  
  let range = maxVal - minVal
  let scale = if range < 1e-12'f32: 1'f32 else: range / 15'f32
  let zeroPoint = if range < 1e-12'f32: 0'f32 else: -minVal / scale
  
  result.kind = qkInt4Asymmetric
  result.shape = shape
  result.scale = @[scale]
  result.zero_point = @[zeroPoint]
  
  let n = arr.len
  result.data = newSeq[uint8]((n + 1) div 2)
  for i in 0 ..< n:
    let q = clamp(int(round(arr[i] / scale + zeroPoint)), 0, 15)
    let nibble = uint8(q and 0x0F)
    let byteIdx = i div 2
    if i mod 2 == 0:
      result.data[byteIdx] = (result.data[byteIdx] and 0xF0'u8) or nibble
    else:
      result.data[byteIdx] = (result.data[byteIdx] and 0x0F'u8) or (nibble shl 4)

proc dequantizeInt4Asymmetric*(qt: QuantTensor): seq[float32] =
  ## Hỗ trợ per-row: nếu qt.scale.len > 1 (== shape[0]), mỗi hàng dùng
  ## scale/zero_point riêng (packed data cũng theo hàng: mỗi hàng round lên
  ## byte riêng - xem quantize_int4_asymmetric_per_row bên export_to_nimq.py).
  ## Hỗ trợ per-group (qt.groupSize > 0): mỗi hàng chia nhỏ hơn nữa thành
  ## ceil(nCols/groupSize) group, mỗi group 1 scale/zero_point (xem
  ## quantize_int4_asymmetric_per_group bên export_to_nimq.py) — mịn gần
  ## bằng GPTQ gốc, tránh sai số double-quantization quá thô của per-row.
  let n = numelQ(qt.shape)
  result = newSeq[float32](n)
  var lut: array[16, float32]   ## THÊM: bảng tra 16 giá trị (kiểu llama.cpp) — int4 chỉ có
                                 ## đúng 16 mã khả dĩ (0..15) trong 1 hàng/group (cùng chung
                                 ## 1 scale/zero_point), nên tính SẴN 16 giá trị 1 lần rồi TRA
                                 ## BẢNG cho từng phần tử — nhanh hơn hẳn so với tính lại phép
                                 ## trừ + nhân (nibble-zp)*scale cho MỖI phần tử riêng lẻ.
  if qt.groupSize > 0:
    doAssert qt.shape.len == 2, "per-group int4 chỉ hỗ trợ tensor 2D"
    let nRows = qt.shape[0]
    let nCols = qt.shape[1]
    let bytesPerRow = (nCols + 1) div 2
    let nGroupsPerRow = (nCols + qt.groupSize - 1) div qt.groupSize
    for r in 0 ..< nRows:
      let rowByteOff = r * bytesPerRow
      let rowGroupOff = r * nGroupsPerRow
      var g = -1
      for c in 0 ..< nCols:
        let gNew = c div qt.groupSize
        if gNew != g:
          g = gNew
          let sc = qt.scale[rowGroupOff + g]
          let zp = qt.zero_point[rowGroupOff + g]
          for k in 0 ..< 16: lut[k] = (float32(k) - zp) * sc   # tính lại LUT chỉ khi sang group mới
        let byteIdx = rowByteOff + c div 2
        var nibble: uint8
        if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
        else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
        result[r * nCols + c] = lut[int(nibble)]
  elif qt.scale.len > 1:
    doAssert qt.shape.len == 2, "per-row int4 chỉ hỗ trợ tensor 2D"
    let nRows = qt.shape[0]
    let nCols = qt.shape[1]
    let bytesPerRow = (nCols + 1) div 2
    for r in 0 ..< nRows:
      let sc = qt.scale[r]
      let zp = qt.zero_point[r]
      for k in 0 ..< 16: lut[k] = (float32(k) - zp) * sc   # tính LUT 1 lần/hàng, tra bảng cho cả hàng
      let rowByteOff = r * bytesPerRow
      for c in 0 ..< nCols:
        let byteIdx = rowByteOff + c div 2
        var nibble: uint8
        if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
        else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
        result[r * nCols + c] = lut[int(nibble)]
  else:
    let sc = qt.scale[0]
    let zp = qt.zero_point[0]
    for i in 0 ..< n:
      let byteIdx = i div 2
      var nibble: uint8
      if i mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
      else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
      result[i] = (float32(nibble) - zp) * sc

# ─────────────────────────────────────────────────────────────
# fp8 / custom / auto — dựa hoàn toàn trên CustomFloat (customfloat.nim)
# ─────────────────────────────────────────────────────────────

proc customFloatMaxMagnitude(cf: CustomFloat): float32 =
  ## Do bien so lon nhat CO THE bieu dien duoc boi dinh dang CustomFloat nay
  ## (khong tinh inf/nan - exponent field = maxExp danh rieng cho inf/nan).
  ## Dung de tinh scale truoc khi quantize - KHONG co buoc nay se lam cac
  ## dinh dang mu-hep (vd fp4 e2m1: bias=1, maxExp=3 -> bien so lon nhat
  ## bieu dien duoc chi ~3.0, NHO NHAT (normal) la ~1.0) am tham VE 0 toan
  ## bo weight thuc te (thuong co magnitude << 1.0) khi khong scale truoc.
  if cf.exponentBits == 0:
    return 1'f32   # int8/int4 khong di qua duong nay (co ham quantize rieng)
  let maxRealExp = (cf.maxExp - 1) - cf.bias   # maxExp danh cho inf/nan, exp hop le lon nhat la maxExp-1
  let maxMantissa = 2'f32 - pow(2'f32, -float32(cf.mantissaBits))  # 1.mantissa toi da (ngu y bit dau)
  result = pow(2'f32, float32(maxRealExp)) * maxMantissa

proc quantizeFloatDtype*(arr: openArray[float32], shape: seq[int], cf: CustomFloat, kind: QuantKind): QuantTensor =
  ## SUA: THAT SU tinh scale (truoc day hard-code scale=[1.0], KHONG lam gi
  ## ca - dinh dang mu-hep nhu fp4 se quantize gan het weight thuc ve 0, xem
  ## giai thich trong customFloatMaxMagnitude). Gio: scale = maxAbs(arr) /
  ## maxMagnitude(cf), chia du lieu cho scale TRUOC khi encode (dua ve dung
  ## dai bieu dien cua dinh dang), luu scale de dequantizeTensor nhan lai.
  ## AN TOAN NGUOC: file .nimq CU co scale=[1.0] (chua qua ham nay ban sua)
  ## van dung dung (dequantizeTensor gio nhan qt.scale[0], 1.0 la no-op).
  var buf: seq[float32]
  for v in arr: buf.add(v)
  let maxMag = customFloatMaxMagnitude(cf)
  let dataMax = maxAbs(buf)
  # scale > 1 nghia la "thu nho du lieu lai truoc khi encode" (dataMax lon
  # hon dai bieu dien) - scale < 1 nghia la "phong to len" (dataMax nho hon
  # dai bieu dien, dung cho fp4 voi weight thuc te ~0.01-0.1). Neu du lieu
  # toan 0 (dataMax=0), giu scale=1 tranh chia cho 0.
  let scale = if dataMax > 0'f32: dataMax / maxMag else: 1'f32
  if scale != 0'f32:
    for i in 0 ..< buf.len: buf[i] = buf[i] / scale
  result.kind = kind
  result.shape = shape
  result.scale = @[scale]
  result.zero_point = @[0'f32]
  result.cf = cf
  result.data = encodeArray(buf, cf)

proc quantizeAuto*(arr: openArray[float32], shape: seq[int],
                    gradArr: openArray[float32] = []): QuantTensor =
  ## APF: tự build CustomFloat theo chính tensor (+gradient nếu có)
  var buf: seq[float32]
  for v in arr: buf.add(v)
  let cf = buildCustomDtypeForTensor(buf, gradArr)
  result = quantizeFloatDtype(buf, shape, cf, qkAuto)

# ─────────────────────────────────────────────────────────────
# Dispatcher chung
# ─────────────────────────────────────────────────────────────

proc quantizeRaw*(arr: openArray[float32], shape: seq[int]): QuantTensor =
  result.kind = qkFp32Raw
  result.shape = shape
  result.scale = @[1'f32]
  result.zero_point = @[0'f32]
  result.data = newSeq[uint8](arr.len * 4)
  for i, v in arr:
    var vv = v
    copyMem(addr result.data[i*4], addr vv, 4)

proc dequantizeRaw*(qt: QuantTensor): seq[float32] =
  let n = qt.bytes.len div 4
  result = newSeq[float32](n)
  for i in 0 ..< n:
    copyMem(addr result[i], unsafeAddr qt.bytes[i*4], 4)

# ═════════════════════════════════════════════════════════════
# FP4-KM  —  "Q4_K_M nhưng FP4"
# ═════════════════════════════════════════════════════════════
#
# VÌ SAO CẦN: `quantizeFloatDtype` + FP4_E2M1 (đường cũ) dùng MỘT scale cho
# CẢ tensor, cộng với 2 khiếm khuyết của codec E2M1 trong customfloat.nim:
#   (a) maxExp bị dành cho inf/nan -> mất 4/16 mã, độ lớn lớn nhất chỉ 3.0
#       thay vì 6.0, dải động 4x thay vì 12x;
#   (b) encodeArray TRUNCATE về 0 chứ không làm tròn (0.90 -> 0.00).
# Hệ quả đo được: ~99.9% weight của ma trận 7B bị ép về đúng 0, rel-L2 0.94.
#
# FP4-KM sửa cả ba:
#   1. Dùng tập mức E2M1 CHUẨN OCP: {0, .5, 1, 1.5, 2, 3, 4, 6} x dấu,
#      đủ 16 mã, KHÔNG mã nào phí cho inf/nan.
#   2. Round-to-nearest, cộng dò scale tối thiểu MSE (kiểu make_qx_quants
#      của llama.cpp).
#   3. Scale theo NHÓM 32 phần tử, lưu bằng 1 byte FP8-E4M3, chuẩn hoá bởi
#      1 scale FP32 per-tensor (đúng bố cục NVFP4 của Blackwell).
#
# KHÁC Q4_K_M Ở ĐÂU: Q4_K_M phải lưu CẢ scale VÀ min cho mỗi sub-block vì
# int4 không có dấu (6+6 bit / 32 phần tử = 4.5 bpw). FP4 đã có bit dấu nên
# đối xứng quanh 0, chỉ cần scale -> 4 + 8/32 = 4.25 bpw.
#
# BỐ CỤC `data` (một khối byte phẳng, không đổi format file .nimq):
#     [0                 , R*C/2            )  nibble, 2 phần tử/byte
#     [R*C/2             , R*C/2 + R*nGroups)  1 byte scale E4M3 cho mỗi nhóm
# Trong mỗi nhóm 32 phần tử, byte j (j=0..15) giữ phần tử j ở nibble THẤP và
# phần tử j+16 ở nibble CAO — đúng kiểu llama.cpp, để lane 0..15 của một
# simdgroup đọc đúng 16 byte liên tục (coalesced).
# `qt.scale = @[globalScale]`, `qt.groupSize = 32`.

const Fp4kmMags*: array[8, float32] =
  [0'f32, 0.5'f32, 1.0'f32, 1.5'f32, 2.0'f32, 3.0'f32, 4.0'f32, 6.0'f32]

proc fp4kmDecodeLut*(): array[16, float32] =
  ## nibble = (dấu shl 3) or magIdx.  magIdx = (e shl 1) or m của E2M1 chuẩn.
  for i in 0 ..< 8:
    result[i] = Fp4kmMags[i]
    result[8 + i] = -Fp4kmMags[i]

proc fp4kmMagIndex(a: float32): int {.inline.} =
  ## Chỉ số độ lớn GẦN NHẤT (round-to-nearest). Ngưỡng = trung điểm.
  if a < 0.25'f32: 0
  elif a < 0.75'f32: 1
  elif a < 1.25'f32: 2
  elif a < 1.75'f32: 3
  elif a < 2.5'f32: 4
  elif a < 3.5'f32: 5
  elif a < 5.0'f32: 6
  else: 7

proc fp4kmGroupErr(g: openArray[float32], n: int, invS, s: float32): float32 {.inline.} =
  var acc = 0'f32
  for i in 0 ..< n:
    let t = g[i] * invS
    let a = abs(t)
    let q = Fp4kmMags[fp4kmMagIndex(a)] * s
    let d = (if t < 0'f32: -q else: q) - g[i]
    acc += d * d
  acc

proc fp4kmMaxAbs(g: openArray[float32], n: int): float32 {.inline.} =
  result = 0'f32
  for i in 0 ..< n:
    if abs(g[i]) > result: result = abs(g[i])

proc fp4kmBestScaleByte(g: openArray[float32], n: int, globalScale: float32,
                        e4m3Lut: array[256, float32], searchSteps: int):
                        tuple[b: uint8, s: float32] =
  ## Dò scale tối thiểu MSE, NHƯNG chỉ trên tập scale THỰC SỰ lưu được
  ## (giá trị E4M3 sau khi giải mã). Dò trên fp32 rồi mới ép về E4M3 là
  ## VÔ ÍCH - đo thực tế cho thấy không cải thiện gì, vì bước làm tròn
  ## scale xoá sạch kết quả dò. Đây là điểm khác biệt so với make_qx_quants
  ## của llama.cpp (ở đó scale lưu fp16 nên gần như liên tục).
  let mx = fp4kmMaxAbs(g, n)
  if mx <= 0'f32 or globalScale <= 0'f32: return (0'u8, 0'f32)
  let base = mx / 6.0'f32
  var bestB = 0'u8
  var bestS = 0'f32
  var bestE = Inf.float32
  let steps = max(searchSteps, 1)
  for k in 0 ..< steps:
    let r = if steps == 1: 1.0'f32
            else: 0.62'f32 + 0.62'f32 * float32(k) / float32(steps - 1)
    let cand = base * r
    let enc = encodeArray([cand / globalScale], FP8_E4M3)
    let b = enc[0]
    let dec = e4m3Lut[int(b)]
    if dec <= 0'f32 or dec > 1e30'f32 or dec != dec: continue
    let s = dec * globalScale
    if s <= 0'f32: continue
    let e = fp4kmGroupErr(g, n, 1'f32 / s, s)
    if e < bestE:
      bestE = e; bestB = b; bestS = s
  (bestB, bestS)

proc quantizeFp4KM*(arr: openArray[float32], shape: seq[int],
                    groupSize: int = 32, searchSteps: int = 8): QuantTensor =
  doAssert shape.len == 2, "quantizeFp4KM: chỉ cho tensor 2D [rows, cols]"
  let R = shape[0]
  let C = shape[1]
  doAssert C mod groupSize == 0,
    "quantizeFp4KM: cols (" & $C & ") phải chia hết groupSize (" & $groupSize & ")"
  doAssert groupSize mod 2 == 0, "quantizeFp4KM: groupSize phải chẵn"
  doAssert arr.len == R * C, "quantizeFp4KM: arr.len không khớp shape"
  let nG = C div groupSize
  let half = groupSize div 2

  # ── pass 1: chỉ cần max|w| từng nhóm để chọn scale per-tensor ──
  var buf = newSeq[float32](groupSize)
  var maxScale = 0'f32
  for r in 0 ..< R:
    for j in 0 ..< nG:
      let off = r * C + j * groupSize
      var mx = 0'f32
      for t in 0 ..< groupSize:
        if abs(arr[off + t]) > mx: mx = abs(arr[off + t])
      let s = mx / 6.0'f32
      if s > maxScale: maxScale = s

  # ── scale per-tensor để đưa scale nhóm vào vùng "ngọt" của E4M3 ──
  # (đúng bố cục NVFP4: 1 FP32 per-tensor + 1 FP8 per-block).
  # 224 là giá trị E4M3 an toàn nằm dưới trần 240 của codec repo.
  let globalScale = if maxScale > 0'f32: maxScale / 224.0'f32 else: 1'f32

  # ── pass 2: chọn THẲNG scale-byte E4M3 tối thiểu MSE cho từng nhóm ──
  # Bắt buộc quantize weight theo ĐÚNG scale sẽ được lưu, không phải scale
  # fp32 trung gian - nếu không hai bên lệch nhau.
  var scaleBytes = newSeq[uint8](R * nG)
  let e4m3Lut = buildDecodeLUT(FP8_E4M3)
  var gsReal = newSeq[float32](R * nG)
  for r in 0 ..< R:
    for j in 0 ..< nG:
      let off = r * C + j * groupSize
      for t in 0 ..< groupSize: buf[t] = arr[off + t]
      let (b, s) = fp4kmBestScaleByte(buf, groupSize, globalScale, e4m3Lut, searchSteps)
      scaleBytes[r * nG + j] = b
      gsReal[r * nG + j] = s

  # ── pass 3: đóng gói nibble ──
  var data = newSeq[uint8](R * C div 2 + R * nG)
  for r in 0 ..< R:
    for j in 0 ..< nG:
      let gi = r * nG + j
      let s = gsReal[gi]
      let off = r * C + j * groupSize
      let blk = gi * half
      if s <= 0'f32:
        for t in 0 ..< half: data[blk + t] = 0'u8
        continue
      let invS = 1'f32 / s
      for t in 0 ..< half:
        let vLo = arr[off + t] * invS
        let vHi = arr[off + t + half] * invS
        let cLo = uint8(fp4kmMagIndex(abs(vLo))) or (if vLo < 0'f32: 8'u8 else: 0'u8)
        let cHi = uint8(fp4kmMagIndex(abs(vHi))) or (if vHi < 0'f32: 8'u8 else: 0'u8)
        data[blk + t] = cLo or (cHi shl 4)
  copyMem(addr data[R * C div 2], addr scaleBytes[0], R * nG)

  result.kind = qkFp4KM
  result.shape = shape
  result.scale = @[globalScale]
  result.zero_point = @[0'f32]
  result.groupSize = groupSize
  result.cf = FP8_E4M3            # ghi lại codec dùng cho scale
  result.data = data

proc dequantizeFp4KM*(qt: QuantTensor): seq[float32] =
  doAssert qt.shape.len == 2, "dequantizeFp4KM: cần tensor 2D"
  let R = qt.shape[0]
  let C = qt.shape[1]
  let g = if qt.groupSize > 0: qt.groupSize else: 32
  let nG = C div g
  let half = g div 2
  let nibEnd = R * C div 2
  let globalScale = if qt.scale.len > 0: qt.scale[0] else: 1'f32
  let lut = fp4kmDecodeLut()
  let e4m3Lut = buildDecodeLUT(FP8_E4M3)
  result = newSeq[float32](R * C)
  for r in 0 ..< R:
    for j in 0 ..< nG:
      let gi = r * nG + j
      let sb = qt.bytes[nibEnd + gi]
      var s = e4m3Lut[int(sb)] * globalScale
      if s > 1e30'f32 or s != s: s = 0'f32
      let off = r * C + j * g
      let blk = gi * half
      for t in 0 ..< half:
        let b = qt.bytes[blk + t]
        result[off + t]        = lut[int(b and 0x0F'u8)] * s
        result[off + t + half] = lut[int(b shr 4)] * s

proc quantizeTensor*(arr: openArray[float32], shape: seq[int], kind: QuantKind,
                      customCf: CustomFloat = FP24, gradArr: openArray[float32] = []): QuantTensor =
  case kind
  of qkFp32Raw: quantizeRaw(arr, shape)
  of qkInt8:    quantizeInt8(arr, shape)
  of qkInt4:    quantizeInt4(arr, shape)
  of qkInt4Asymmetric: quantizeInt4Asymmetric(arr, shape)  # THÊM
  of qkFp8E4M3: quantizeFloatDtype(arr, shape, FP8_E4M3, qkFp8E4M3)
  of qkFp8E5M2: quantizeFloatDtype(arr, shape, FP8_E5M2, qkFp8E5M2)
  of qkCustom:  quantizeFloatDtype(arr, shape, customCf, qkCustom)
  of qkAuto:    quantizeAuto(arr, shape, gradArr)
  of qkFp4KM:   quantizeFp4KM(arr, shape)

proc dequantizeTensor*(qt: QuantTensor): seq[float32] =
  case qt.kind
  of qkFp32Raw: dequantizeRaw(qt)
  of qkInt8:    dequantizeInt8(qt)
  of qkInt4:    dequantizeInt4(qt)
  of qkInt4Asymmetric: dequantizeInt4Asymmetric(qt)  # THÊM
  of qkFp4KM:   dequantizeFp4KM(qt)
  of qkFp8E4M3, qkFp8E5M2, qkCustom, qkAuto:
    # SUA: NHAN LAI qt.scale[0] (truoc day bo qua hoan toan, luon coi nhu
    # scale=1 - dung voi file CU (scale that su la 1.0, no-op) nhung SAI voi
    # file MOI qua quantizeFloatDtype ban sua (co scale thuc, xem giai thich
    # o do) - neu khong nhan lai o day, decode ra toan gia tri be xiu/0 sai
    # hoan toan so voi weight goc.
    let raw = decodeArray(qt.bytes, qt.cf)
    let s = if qt.scale.len > 0: qt.scale[0] else: 1'f32
    if s == 1'f32: raw
    else:
      var res = newSeq[float32](raw.len)
      for i in 0 ..< raw.len: res[i] = raw[i] * s
      res

proc dequantizeTensorTransposed*(qt: QuantTensor): seq[float32] =
  ## THÊM: dequant + transpose GỘP LÀM 1 PASS, ghi thẳng vào layout đích
  ## [nCols, nRows] (transposed) thay vì dequantizeTensor() (ra [nRows,nCols])
  ## rồi transpose() riêng — cách cũ cấp phát 2 mảng fp32 full-size cùng lúc
  ## (Linear.forward gọi hàm này mỗi lần forward cho weight lớn, vd lm_head/
  ## các q/k/v/o/gate/up/down proj của model 6.7B), giờ chỉ 1 mảng.
  ## Dùng LUT 16 giá trị (kiểu llama.cpp) cho int4/int4-asymmetric vì mỗi
  ## hàng/group chỉ có 16 mã khả dĩ — tính sẵn 16 giá trị rồi tra bảng cho
  ## từng phần tử, thay vì tính lại phép trừ+nhân mỗi phần tử.
  ## FIX: hỗ trợ tensor 1D (bias, LN gamma/beta) — coi như hàng đơn (1 x n)
  ## để tránh crash khi Linear.useQuant=true nhưng weightQ là 1D (quantBase
  ## chứa cả bias/gamma 1D từ checkpoint NIMBIN4).
  if qt.shape.len <= 1:
    return dequantizeTensor(qt)  # 0D/1D: không cần transpose, trả về nguyên bản
  let nRows = qt.shape[0]
  let nCols = qt.shape[1]
  result = newSeq[float32](nRows * nCols)
  case qt.kind
  of qkInt4Asymmetric:
    var lut: array[16, float32]
    let bytesPerRow = (nCols + 1) div 2
    if qt.groupSize > 0:
      let nGroupsPerRow = (nCols + qt.groupSize - 1) div qt.groupSize
      for r in 0 ..< nRows:
        let rowByteOff = r * bytesPerRow
        let rowGroupOff = r * nGroupsPerRow
        var g = -1
        for c in 0 ..< nCols:
          let gNew = c div qt.groupSize
          if gNew != g:
            g = gNew
            let sc = qt.scale[rowGroupOff + g]
            let zp = qt.zero_point[rowGroupOff + g]
            for k in 0 ..< 16: lut[k] = (float32(k) - zp) * sc
          let byteIdx = rowByteOff + c div 2
          var nibble: uint8
          if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
          else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
          result[c * nRows + r] = lut[int(nibble)]   # ghi thẳng layout transposed
    elif qt.scale.len > 1:
      for r in 0 ..< nRows:
        let sc = qt.scale[r]
        let zp = qt.zero_point[r]
        for k in 0 ..< 16: lut[k] = (float32(k) - zp) * sc
        let rowByteOff = r * bytesPerRow
        for c in 0 ..< nCols:
          let byteIdx = rowByteOff + c div 2
          var nibble: uint8
          if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
          else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
          result[c * nRows + r] = lut[int(nibble)]
    else:
      let sc = qt.scale[0]
      let zp = qt.zero_point[0]
      for k in 0 ..< 16: lut[k] = (float32(k) - zp) * sc
      for r in 0 ..< nRows:
        let rowByteOff = r * bytesPerRow
        for c in 0 ..< nCols:
          let byteIdx = rowByteOff + c div 2
          var nibble: uint8
          if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
          else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
          result[c * nRows + r] = lut[int(nibble)]
  of qkInt4:
    let sc = qt.scale[0]
    let bytesPerRow = (nCols + 1) div 2
    var lut: array[16, float32]
    for k in 0 ..< 16:
      var signed = int(k)
      if signed >= 8: signed -= 16
      lut[k] = float32(signed) * sc
    for r in 0 ..< nRows:
      let rowByteOff = r * bytesPerRow
      for c in 0 ..< nCols:
        let byteIdx = rowByteOff + c div 2
        var nibble: uint8
        if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
        else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
        result[c * nRows + r] = lut[int(nibble)]
  of qkInt8:
    let sc = qt.scale[0]
    for r in 0 ..< nRows:
      let rowOff = r * nCols
      for c in 0 ..< nCols:
        result[c * nRows + r] = float32(cast[int8](qt.bytes[rowOff + c])) * sc
  else:
    # Hiếm gặp (fp32 raw / fp8 / custom / auto) — không phải đường nóng
    # (đường nóng q/k/v/o/gate/up/down/lm_head luôn dùng int4 asymmetric
    # per-group), nên chấp nhận dequant thường rồi transpose (2 mảng tạm).
    let flat = dequantizeTensor(qt)
    for r in 0 ..< nRows:
      let rowOff = r * nCols
      for c in 0 ..< nCols:
        result[c * nRows + r] = flat[rowOff + c]

# ─────────────────────────────────────────────────────────────
# I/O nhị phân — lưu/tải 1 state dict (danh sách (tên, QuantTensor)) + kiến trúc
# ─────────────────────────────────────────────────────────────

const MagicStr = "NIMQ2"   ## THÊM: bump tu NIMQ1 -> NIMQ2 vi them field groupSize (per-group int4).
                            ## File .nimq cu (NIMQ1) phai export lai bang export_to_nimq.py moi.
const MagicAligned* = "NIMQ3"
  ## NIMQ2 plus an int32 payload alignment after the magic, padding up to that
  ## alignment before the arch fields, and each tensor's packed payload
  ## starting at a multiple of it. Lets a GPU on unified memory wrap each
  ## payload of the mmap in place. The header padding also guarantees every
  ## NIMQ2 byte moves forward by >= align - 9 bytes when a file is converted
  ## in place (convert_nimq_aligned).
const NimqPayloadAlign* = 65536
  ## Default for NIMQ3: covers 4 KiB (x86) and 16 KiB (Apple Silicon) pages and
  ## Vulkan host-pointer import alignments.

proc writeString(f: File, s: string) =
  var n = int32(s.len)
  discard f.writeBuffer(addr n, sizeof(n))
  if s.len > 0: discard f.writeBuffer(unsafeAddr s[0], s.len)

proc readString(f: File): string =
  var n: int32
  discard f.readBuffer(addr n, sizeof(n))
  result = newString(n)
  if n > 0: discard f.readBuffer(addr result[0], n)

proc writeQuantTensor(f: File, qt: QuantTensor; payloadAlign = 0) =
  var kindByte = uint8(ord(qt.kind))
  discard f.writeBuffer(addr kindByte, 1)
  var ndims = int32(qt.shape.len)
  discard f.writeBuffer(addr ndims, 4)
  for d in qt.shape:
    var dv = int32(d)
    discard f.writeBuffer(addr dv, 4)
  var nScales = int32(qt.scale.len)
  discard f.writeBuffer(addr nScales, 4)
  if qt.scale.len > 0:
    discard f.writeBuffer(unsafeAddr qt.scale[0], qt.scale.len * 4)
  if qt.zero_point.len > 0:
    discard f.writeBuffer(unsafeAddr qt.zero_point[0], qt.zero_point.len * 4)
  var gs = int32(qt.groupSize)
  discard f.writeBuffer(addr gs, 4)
  var eb = int32(max(qt.cf.exponentBits, 1))
  var mb = int32(qt.cf.mantissaBits)
  discard f.writeBuffer(addr eb, 4)
  discard f.writeBuffer(addr mb, 4)
  var dlen = int32(qt.bytes.len)
  discard f.writeBuffer(addr dlen, 4)
  if payloadAlign > 1:
    let pad = (payloadAlign - int(f.getFilePos() mod payloadAlign)) mod payloadAlign
    if pad > 0:
      var zeros = newSeq[uint8](pad)
      discard f.writeBuffer(addr zeros[0], pad)
  if qt.bytes.len > 0:
    discard f.writeBuffer(unsafeAddr qt.bytes[0], qt.bytes.len)

proc readQuantTensor(f: File): QuantTensor =
  var kindByte: uint8
  discard f.readBuffer(addr kindByte, 1)
  result.kind = QuantKind(kindByte)
  var ndims: int32
  discard f.readBuffer(addr ndims, 4)
  result.shape = newSeq[int](ndims)
  for i in 0 ..< ndims:
    var dv: int32
    discard f.readBuffer(addr dv, 4)
    result.shape[i] = int(dv)
  var nScales: int32
  discard f.readBuffer(addr nScales, 4)
  result.scale = newSeq[float32](nScales)
  if nScales > 0:
    discard f.readBuffer(addr result.scale[0], nScales * 4)
  result.zero_point = newSeq[float32](nScales)
  if nScales > 0:
    discard f.readBuffer(addr result.zero_point[0], nScales * 4)
  var gs: int32
  discard f.readBuffer(addr gs, 4)
  result.groupSize = int(gs)
  var eb, mb: int32
  discard f.readBuffer(addr eb, 4)
  discard f.readBuffer(addr mb, 4)
  result.cf = newCustomFloat(int(eb), int(mb))
  var dlen: int32
  discard f.readBuffer(addr dlen, 4)
  result.data = newSeq[uint8](dlen)
  if dlen > 0:
    discard f.readBuffer(addr result.data[0], dlen)

proc rawFloatView*(qt: QuantTensor): MappedView[float32] =
  ## The float32 payload of a mapped qkFp32Raw tensor, read in place (NIMQ2
  ## payloads may be unaligned; callers copy rows with copyMem).
  ## owner == nil when the tensor is not mapped raw fp32.
  if qt.kind == qkFp32Raw and qt.mappedData.owner != nil and qt.mappedData.len mod 4 == 0:
    result = mappedView[float32](qt.mappedData.owner, qt.mappedData.offset,
      qt.mappedData.len div 4)

proc readMappedQuantTensor*(c: var WeightCursor; mapped = true;
                            payloadAlign = 0): QuantTensor =
  let kind = readValue[uint8](c)
  if kind > uint8(high(QuantKind)): raise newException(IOError, "Invalid quant kind")
  result.kind = QuantKind(kind)
  let ndims = c.readCount(4)
  if ndims > 8: raise newException(IOError, "Invalid tensor rank")
  for i in 0 ..< ndims:
    let d = int(readValue[int32](c))
    if d <= 0: raise newException(IOError, "Invalid tensor dimension")
    result.shape.add d
  let ns = c.readCount(4)
  result.scale = readSeq[float32](c, ns)
  result.zero_point = readSeq[float32](c, ns)
  result.groupSize = int(readValue[int32](c))
  let eb = int(readValue[int32](c))
  let mb = int(readValue[int32](c))
  if eb < 0 or mb < 0 or eb > 8 or mb > 23:
    raise newException(IOError, "Invalid custom float format")
  result.cf = newCustomFloat(eb, mb)
  let n = c.readCount()
  if payloadAlign > 1:
    c.pos += (payloadAlign - c.pos mod payloadAlign) mod payloadAlign
    c.owner.checkRange(c.pos, n)
  if mapped:
    result.mappedData = mappedView[uint8](c.owner, c.pos, n)
    c.pos += n
  else:
    result.data = readSeq[uint8](c, n)

proc saveQuantStateDict*(path: string, arch: array[5, int], sd: seq[(string, QuantTensor)];
                         payloadAlign = 0) =
  ## payloadAlign > 0 writes NIMQ3 (payloads aligned for in-place GPU use);
  ## 0 keeps the NIMQ2 layout that export_to_nimq.py produces.
  if payloadAlign < 0 or payloadAlign > (1 shl 24) or
      (payloadAlign > 0 and (payloadAlign < 64 or (payloadAlign and (payloadAlign - 1)) != 0)):
    raise newException(ValueError, "Invalid NIMQ payload alignment")
  var f = open(path, fmWrite)
  defer: f.close()
  if payloadAlign > 0:
    writeString(f, MagicAligned)
    var a = int32(payloadAlign)
    discard f.writeBuffer(addr a, 4)
    var headerPad = newSeq[uint8](payloadAlign - int(f.getFilePos() mod payloadAlign))
    discard f.writeBuffer(addr headerPad[0], headerPad.len)
  else:
    writeString(f, MagicStr)
  var archI32: array[5, int32]
  for i, v in arch: archI32[i] = int32(v)
  discard f.writeBuffer(addr archI32[0], sizeof(archI32))
  var n = int32(sd.len)
  discard f.writeBuffer(addr n, 4)
  for (name, qt) in sd:
    writeString(f, name)
    writeQuantTensor(f, qt, payloadAlign)
  if payloadAlign > 1:
    # GPUs wrap each payload rounded up to their page/import alignment; the
    # last one must not run past the end of the mapped file.
    let pad = (payloadAlign - int(f.getFilePos() mod payloadAlign)) mod payloadAlign
    if pad > 0:
      var zeros = newSeq[uint8](pad)
      discard f.writeBuffer(addr zeros[0], pad)

proc loadQuantStateDict*(path: string; mapped = false): tuple[arch: array[5, int],
    sd: seq[(string, QuantTensor)], payloadAlign: int] =
  ## Reads NIMQ2 and NIMQ3. payloadAlign is 0 for NIMQ2.
  var c = WeightCursor(owner: mapWeights(path))
  let magic = c.readString()
  if magic == MagicAligned:
    result.payloadAlign = int(readValue[int32](c))
    if result.payloadAlign <= 13 or result.payloadAlign > (1 shl 24):
      raise newException(IOError, "Invalid NIMQ payload alignment: " & path)
    c.pos = result.payloadAlign
  elif magic != MagicStr:
    raise newException(IOError, "Invalid NIMQ checkpoint: " & path)
  for i in 0 ..< 5: result.arch[i] = int(readValue[int32](c))
  let n = c.readCount()
  for i in 0 ..< n:
    let name = c.readString()
    result.sd.add((name, readMappedQuantTensor(c, mapped, result.payloadAlign)))
