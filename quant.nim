## quant.nim
## Lượng tử hoá tensor cho Nimformer: int8, int4, fp8 (e4m3/e5m2), và "auto"
## (APF tự build CustomFloat theo tensor — dùng lại customfloat.nim).
## Kèm định dạng file nhị phân đơn giản để lưu/tải state dict đã lượng tử hoá.
##
## Không có main/test ở đây — đây là thư viện thuần, import vào chỗ khác dùng.

import std/math
import customfloat

type
  QuantKind* = enum
    qkFp32Raw   ## không nén, giữ nguyên float32 (dùng cho bias/layernorm — nhạy sai số)
    qkInt8      ## symmetric int8, scale = max(|x|)/127
    qkInt4      ## symmetric int4 (pack 2 giá trị/byte), scale = max(|x|)/7
    qkFp8E4M3   ## CustomFloat(4,3) — dùng customfloat.nim
    qkFp8E5M2   ## CustomFloat(5,2)
    qkCustom    ## CustomFloat tuỳ ý do người dùng khai (bao nhiêu bit cũng được)
    qkAuto      ## APF: tự build CustomFloat theo chính tensor (buildCustomDtypeForTensor)

  QuantTensor* = object
    kind*: QuantKind
    shape*: seq[int]
    scale*: float32     ## dùng cho qkInt8/qkInt4, bỏ qua ở các kind khác
    cf*: CustomFloat     ## dùng cho qkFp8E4M3/E5M2/qkCustom/qkAuto
    data*: seq[uint8]

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
  result.scale = scale
  result.data = newSeq[uint8](arr.len)
  for i, v in arr:
    let q = clamp(int(round(v / scale)), -127, 127)
    result.data[i] = cast[uint8](int8(q))

proc dequantizeInt8*(qt: QuantTensor): seq[float32] =
  result = newSeq[float32](qt.data.len)
  for i, b in qt.data:
    result[i] = float32(cast[int8](b)) * qt.scale

proc quantizeInt4*(arr: openArray[float32], shape: seq[int]): QuantTensor =
  let m = maxAbs(arr)
  let scale = if m == 0'f32: 1'f32 else: m / 7'f32
  result.kind = qkInt4
  result.shape = shape
  result.scale = scale
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
  let n = numelQ(qt.shape)
  result = newSeq[float32](n)
  for i in 0 ..< n:
    let byteIdx = i div 2
    var nibble: uint8
    if i mod 2 == 0: nibble = qt.data[byteIdx] and 0x0F'u8
    else: nibble = (qt.data[byteIdx] shr 4) and 0x0F'u8
    var signed = int(nibble)
    if signed >= 8: signed -= 16
    result[i] = float32(signed) * qt.scale

# ─────────────────────────────────────────────────────────────
# fp8 / custom / auto — dựa hoàn toàn trên CustomFloat (customfloat.nim)
# ─────────────────────────────────────────────────────────────

proc quantizeFloatDtype*(arr: openArray[float32], shape: seq[int], cf: CustomFloat, kind: QuantKind): QuantTensor =
  var buf: seq[float32]
  for v in arr: buf.add(v)
  result.kind = kind
  result.shape = shape
  result.scale = 1'f32
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
  result.scale = 1'f32
  result.data = newSeq[uint8](arr.len * 4)
  for i, v in arr:
    var vv = v
    copyMem(addr result.data[i*4], addr vv, 4)

proc dequantizeRaw*(qt: QuantTensor): seq[float32] =
  let n = qt.data.len div 4
  result = newSeq[float32](n)
  for i in 0 ..< n:
    copyMem(addr result[i], unsafeAddr qt.data[i*4], 4)

proc quantizeTensor*(arr: openArray[float32], shape: seq[int], kind: QuantKind,
                      customCf: CustomFloat = FP24, gradArr: openArray[float32] = []): QuantTensor =
  case kind
  of qkFp32Raw: quantizeRaw(arr, shape)
  of qkInt8:    quantizeInt8(arr, shape)
  of qkInt4:    quantizeInt4(arr, shape)
  of qkFp8E4M3: quantizeFloatDtype(arr, shape, FP8_E4M3, qkFp8E4M3)
  of qkFp8E5M2: quantizeFloatDtype(arr, shape, FP8_E5M2, qkFp8E5M2)
  of qkCustom:  quantizeFloatDtype(arr, shape, customCf, qkCustom)
  of qkAuto:    quantizeAuto(arr, shape, gradArr)

proc dequantizeTensor*(qt: QuantTensor): seq[float32] =
  case qt.kind
  of qkFp32Raw: dequantizeRaw(qt)
  of qkInt8:    dequantizeInt8(qt)
  of qkInt4:    dequantizeInt4(qt)
  of qkFp8E4M3, qkFp8E5M2, qkCustom, qkAuto: decodeArray(qt.data, qt.cf)

# ─────────────────────────────────────────────────────────────
# I/O nhị phân — lưu/tải 1 state dict (danh sách (tên, QuantTensor)) + kiến trúc
# ─────────────────────────────────────────────────────────────

const MagicStr = "NIMQ1"

proc writeString(f: File, s: string) =
  var n = int32(s.len)
  discard f.writeBuffer(addr n, sizeof(n))
  if s.len > 0: discard f.writeBuffer(unsafeAddr s[0], s.len)

proc readString(f: File): string =
  var n: int32
  discard f.readBuffer(addr n, sizeof(n))
  result = newString(n)
  if n > 0: discard f.readBuffer(addr result[0], n)

proc writeQuantTensor(f: File, qt: QuantTensor) =
  var kindByte = uint8(ord(qt.kind))
  discard f.writeBuffer(addr kindByte, 1)
  var ndims = int32(qt.shape.len)
  discard f.writeBuffer(addr ndims, 4)
  for d in qt.shape:
    var dv = int32(d)
    discard f.writeBuffer(addr dv, 4)
  var scale = qt.scale
  discard f.writeBuffer(addr scale, 4)
  var eb = int32(max(qt.cf.exponentBits, 1))
  var mb = int32(qt.cf.mantissaBits)
  discard f.writeBuffer(addr eb, 4)
  discard f.writeBuffer(addr mb, 4)
  var dlen = int32(qt.data.len)
  discard f.writeBuffer(addr dlen, 4)
  if qt.data.len > 0:
    discard f.writeBuffer(unsafeAddr qt.data[0], qt.data.len)

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
  discard f.readBuffer(addr result.scale, 4)
  var eb, mb: int32
  discard f.readBuffer(addr eb, 4)
  discard f.readBuffer(addr mb, 4)
  result.cf = newCustomFloat(int(eb), int(mb))
  var dlen: int32
  discard f.readBuffer(addr dlen, 4)
  result.data = newSeq[uint8](dlen)
  if dlen > 0:
    discard f.readBuffer(addr result.data[0], dlen)

proc saveQuantStateDict*(path: string, arch: array[5, int], sd: seq[(string, QuantTensor)]) =
  var f = open(path, fmWrite)
  defer: f.close()
  writeString(f, MagicStr)
  var archI32: array[5, int32]
  for i, v in arch: archI32[i] = int32(v)
  discard f.writeBuffer(addr archI32[0], sizeof(archI32))
  var n = int32(sd.len)
  discard f.writeBuffer(addr n, 4)
  for (name, qt) in sd:
    writeString(f, name)
    writeQuantTensor(f, qt)

proc loadQuantStateDict*(path: string): tuple[arch: array[5, int], sd: seq[(string, QuantTensor)]] =
  var f = open(path, fmRead)
  defer: f.close()
  let magic = readString(f)
  doAssert magic == MagicStr, "File không đúng định dạng NIMQ1: " & path
  var archI32: array[5, int32]
  discard f.readBuffer(addr archI32[0], sizeof(archI32))
  for i in 0 ..< 5: result.arch[i] = int(archI32[i])
  var n: int32
  discard f.readBuffer(addr n, 4)
  for i in 0 ..< n:
    let name = readString(f)
    let qt = readQuantTensor(f)
    result.sd.add((name, qt))