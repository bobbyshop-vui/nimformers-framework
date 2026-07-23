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
    qkInt4Asymmetric  ## THÊM: asymmetric int4 (Q4_K_M style), scale=(max-min)/15, zero_point
    qkFp8E4M3   ## CustomFloat(4,3) — dùng customfloat.nim
    qkFp8E5M2   ## CustomFloat(5,2)
    qkCustom    ## CustomFloat tuỳ ý do người dùng khai (bao nhiêu bit cũng được)
    qkAuto      ## APF: tự build CustomFloat theo chính tensor (buildCustomDtypeForTensor)

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
  result.scale = @[scale]
  result.zero_point = @[0'f32]
  result.data = newSeq[uint8](arr.len)
  for i, v in arr:
    let q = clamp(int(round(v / scale)), -127, 127)
    result.data[i] = cast[uint8](int8(q))

proc dequantizeInt8*(qt: QuantTensor): seq[float32] =
  let sc = qt.scale[0]
  result = newSeq[float32](qt.data.len)
  for i, b in qt.data:
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
  let sc = qt.scale[0]
  let n = numelQ(qt.shape)
  result = newSeq[float32](n)
  for i in 0 ..< n:
    let byteIdx = i div 2
    var nibble: uint8
    if i mod 2 == 0: nibble = qt.data[byteIdx] and 0x0F'u8
    else: nibble = (qt.data[byteIdx] shr 4) and 0x0F'u8
    var signed = int(nibble)
    if signed >= 8: signed -= 16
    result[i] = float32(signed) * sc

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
        if c mod 2 == 0: nibble = qt.data[byteIdx] and 0x0F'u8
        else: nibble = (qt.data[byteIdx] shr 4) and 0x0F'u8
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
        if c mod 2 == 0: nibble = qt.data[byteIdx] and 0x0F'u8
        else: nibble = (qt.data[byteIdx] shr 4) and 0x0F'u8
        result[r * nCols + c] = lut[int(nibble)]
  else:
    let sc = qt.scale[0]
    let zp = qt.zero_point[0]
    for i in 0 ..< n:
      let byteIdx = i div 2
      var nibble: uint8
      if i mod 2 == 0: nibble = qt.data[byteIdx] and 0x0F'u8
      else: nibble = (qt.data[byteIdx] shr 4) and 0x0F'u8
      result[i] = (float32(nibble) - zp) * sc

# ─────────────────────────────────────────────────────────────
# fp8 / custom / auto — dựa hoàn toàn trên CustomFloat (customfloat.nim)
# ─────────────────────────────────────────────────────────────

proc quantizeFloatDtype*(arr: openArray[float32], shape: seq[int], cf: CustomFloat, kind: QuantKind): QuantTensor =
  var buf: seq[float32]
  for v in arr: buf.add(v)
  result.kind = kind
  result.shape = shape
  result.scale = @[1'f32]
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
  of qkInt4Asymmetric: quantizeInt4Asymmetric(arr, shape)  # THÊM
  of qkFp8E4M3: quantizeFloatDtype(arr, shape, FP8_E4M3, qkFp8E4M3)
  of qkFp8E5M2: quantizeFloatDtype(arr, shape, FP8_E5M2, qkFp8E5M2)
  of qkCustom:  quantizeFloatDtype(arr, shape, customCf, qkCustom)
  of qkAuto:    quantizeAuto(arr, shape, gradArr)

proc dequantizeTensor*(qt: QuantTensor): seq[float32] =
  case qt.kind
  of qkFp32Raw: dequantizeRaw(qt)
  of qkInt8:    dequantizeInt8(qt)
  of qkInt4:    dequantizeInt4(qt)
  of qkInt4Asymmetric: dequantizeInt4Asymmetric(qt)  # THÊM
  of qkFp8E4M3, qkFp8E5M2, qkCustom, qkAuto: decodeArray(qt.data, qt.cf)

proc dequantizeTensorTransposed*(qt: QuantTensor): seq[float32] =
  ## THÊM: dequant + transpose GỘP LÀM 1 PASS, ghi thẳng vào layout đích
  ## [nCols, nRows] (transposed) thay vì dequantizeTensor() (ra [nRows,nCols])
  ## rồi transpose() riêng — cách cũ cấp phát 2 mảng fp32 full-size cùng lúc
  ## (Linear.forward gọi hàm này mỗi lần forward cho weight lớn, vd lm_head/
  ## các q/k/v/o/gate/up/down proj của model 6.7B), giờ chỉ 1 mảng.
  ## Dùng LUT 16 giá trị (kiểu llama.cpp) cho int4/int4-asymmetric vì mỗi
  ## hàng/group chỉ có 16 mã khả dĩ — tính sẵn 16 giá trị rồi tra bảng cho
  ## từng phần tử, thay vì tính lại phép trừ+nhân mỗi phần tử.
  doAssert qt.shape.len == 2, "dequantizeTensorTransposed chỉ hỗ trợ tensor 2D (weight matrix)"
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
          if c mod 2 == 0: nibble = qt.data[byteIdx] and 0x0F'u8
          else: nibble = (qt.data[byteIdx] shr 4) and 0x0F'u8
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
          if c mod 2 == 0: nibble = qt.data[byteIdx] and 0x0F'u8
          else: nibble = (qt.data[byteIdx] shr 4) and 0x0F'u8
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
          if c mod 2 == 0: nibble = qt.data[byteIdx] and 0x0F'u8
          else: nibble = (qt.data[byteIdx] shr 4) and 0x0F'u8
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
        if c mod 2 == 0: nibble = qt.data[byteIdx] and 0x0F'u8
        else: nibble = (qt.data[byteIdx] shr 4) and 0x0F'u8
        result[c * nRows + r] = lut[int(nibble)]
  of qkInt8:
    let sc = qt.scale[0]
    for r in 0 ..< nRows:
      let rowOff = r * nCols
      for c in 0 ..< nCols:
        result[c * nRows + r] = float32(cast[int8](qt.data[rowOff + c])) * sc
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