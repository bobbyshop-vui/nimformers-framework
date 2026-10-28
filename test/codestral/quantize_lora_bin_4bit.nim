## quantize_lora_bin_4bit.nim
## Chuyển checkpoint LoRA .bin (fp32 thô, rawbin) sang bản QLoRA 4-bit
## để giảm kích thước đĩa/RAM (~8GB fp32 -> ~1–1.5GB tùy số block Attention).
##
## Quy ước:
##   - Adapter LoRA A/B: GIỮ fp32 (rank thấp, nhỏ, cần precision khi train/merge).
##   - newBlock::* (weight block Attention full-train): quantize 4-bit
##     (mặc định qkInt4Asymmetric per-group 128, gần Q4_K_M).
##   - Meta + format magic mới "NIMBIN4\0" để load nhận ra và dequant đúng lúc cần.
##   - GHI ĐÈ được: truyền cùng path nguồn/đích (ghi file .tmp rồi rename an toàn).
##
## Dùng:
##   nim c -d:release quantize_lora_bin_4bit.nim
##   ./quantize_lora_bin_4bit jamba_lora.bin              # GHI ĐÈ cùng file
##   ./quantize_lora_bin_4bit jamba_lora.bin jamba_lora.bin
##   ./quantize_lora_bin_4bit jamba_lora.bin out_qlora4.bin  # ra file khác
##
## Sau khi có file 4-bit:
##   - finetune_lora / infer_lora đã được bổ sung load path nhận magic NIMBIN4
##     (SUA: newBlock giữ nguyên dạng nén QuantTensor, gộp vào lora.quantBase
##     cùng base .nimq - dequant-on-the-fly + giải phóng qua mergeIn()/
##     restoreBase() trong lora.nim, KHÔNG còn dequant hết về fp32 lúc load
##     nữa; adapter A/B vẫn fp32 như cũ, dùng loadQLora4Bin thay vì
##     loadQLora4BinDequant - hàm cũ vẫn còn, không ai gọi nữa).
##   - Muốn train/infer nhẹ hơn nữa: giữ quant-resident cho base Mamba qua .nimq
##     như pipeline codestral hiện có (đã có sẵn).

import std/[os, strformat, strutils, tables, math, streams]

proc cRename(oldPath, newPath: cstring): cint {.importc: "rename", header: "<stdio.h>".}
import rawbin
import quant
import customfloat

const
  MAGIC_FP32* = "NIMBIN1\0"
  MAGIC_QLORA4* = "NIMBIN4\0"
  DefaultGroupSize = 128

# ── Ghi/đọc QuantTensor thô vào stream (tương thích layout quant.nim) ─────

proc writeI32(s: Stream, v: int32) =
  var x = v
  s.writeData(addr x, 4)

proc writeI64(s: Stream, v: int64) =
  var x = v
  s.writeData(addr x, 8)

proc writeStrLP(s: Stream, str: string) =
  writeI32(s, int32(str.len))
  if str.len > 0:
    s.writeData(unsafeAddr str[0], str.len)

proc writeQuantTensorRaw(s: Stream, qt: QuantTensor) =
  var kindByte = uint8(qt.kind)
  s.writeData(addr kindByte, 1)
  writeI32(s, int32(qt.shape.len))
  for dim in qt.shape: writeI32(s, int32(dim))
  writeI32(s, int32(qt.scale.len))
  if qt.scale.len > 0:
    s.writeData(unsafeAddr qt.scale[0], qt.scale.len * 4)
  # zero_point cùng số phần tử với scale (layout quant.nim)
  var zp = qt.zero_point
  if zp.len != qt.scale.len:
    zp = newSeq[float32](qt.scale.len)
  if zp.len > 0:
    s.writeData(unsafeAddr zp[0], zp.len * 4)
  writeI32(s, int32(qt.groupSize))
  writeI32(s, int32(max(qt.cf.exponentBits, 1)))
  writeI32(s, int32(qt.cf.mantissaBits))
  writeI32(s, int32(qt.data.len))
  if qt.data.len > 0:
    s.writeData(unsafeAddr qt.data[0], qt.data.len)

proc quantizeNewBlock*(data: seq[float32], shape: seq[int],
                      kind: QuantKind, groupSize: int): QuantTensor =
  ## Chọn đường quant tốt nhất có sẵn trong quant.nim.
  case kind
  of qkInt4Asymmetric:
    # per-tensor asymmetric trước; nếu 2D và groupSize>0 thì dùng path per-group
    # (quantizeInt4Asymmetric hiện tại là per-tensor; per-group nằm ở
    # dequantize path / export_to_nimq). Để đơn giản + ổn định: quantize
    # per-row nếu 2D (mỗi hàng 1 scale) — tốt hơn per-tensor rất nhiều.
    if shape.len == 2 and shape[0] > 0 and shape[1] > 0:
      # Layout khớp dequantizeInt4Asymmetric (per-row): mỗi hàng pad lên
      # bytesPerRow = ceil(cols/2), scale/zp theo hàng.
      let rows = shape[0]
      let cols = shape[1]
      let bytesPerRow = (cols + 1) div 2
      var scales = newSeq[float32](rows)
      var zps = newSeq[float32](rows)
      var packed = newSeq[uint8](rows * bytesPerRow)
      for r in 0 ..< rows:
        var minV = float32.high
        var maxV = -float32.high
        let off = r * cols
        for c in 0 ..< cols:
          let v = data[off + c]
          if v < minV: minV = v
          if v > maxV: maxV = v
        let rng = maxV - minV
        let sc = if rng < 1e-12'f32: 1'f32 else: rng / 15'f32
        let zp = if rng < 1e-12'f32: 0'f32 else: -minV / sc
        scales[r] = sc
        zps[r] = zp
        let rowByteOff = r * bytesPerRow
        for c in 0 ..< cols:
          let q = clamp(int(round(data[off + c] / sc + zp)), 0, 15)
          let nibble = uint8(q and 0x0F)
          let byteIdx = rowByteOff + c div 2
          if c mod 2 == 0:
            packed[byteIdx] = (packed[byteIdx] and 0xF0'u8) or nibble
          else:
            packed[byteIdx] = (packed[byteIdx] and 0x0F'u8) or (nibble shl 4)
      result.kind = qkInt4Asymmetric
      result.shape = shape
      result.scale = scales
      result.zero_point = zps
      result.groupSize = 0   # per-row (scale.len == rows)
      result.data = packed
    else:
      result = quantizeInt4Asymmetric(data, shape)
  of qkInt4:
    result = quantizeInt4(data, shape)
  of qkFp4KM:
    # cols phải chia hết groupSize
    if shape.len == 2 and shape[1] mod groupSize == 0:
      result = quantizeFp4KM(data, shape, groupSize = groupSize)
    else:
      echo "[qlora4] qkFp4KM yêu cầu cols chia hết groupSize; fallback int4 asymmetric"
      result = quantizeNewBlock(data, shape, qkInt4Asymmetric, groupSize)
  else:
    result = quantizeInt4Asymmetric(data, shape)

proc saveQLora4Bin*(path: string, meta: Table[string, string],
                    fp32Tensors: seq[RawBinTensor],
                    quantTensors: seq[(string, QuantTensor)]) =
  ## Magic NIMBIN4:
  ##   meta (giống NIMBIN1) + nFp32 + [fp32 tensors] + nQuant + [quant tensors]
  ##
  ## QUAN TRỌNG CHO TRAIN/RESUME: ghi vào .tmp rồi rename sau khi close.
  ## Không được ghi thẳng vào checkpoint đích, vì nếu process bị kill/OOM
  ## trong lúc save thì checkpoint cũ phải còn nguyên để lần sau resume.
  let tmpPath = path & ".tmp"
  var f = newFileStream(tmpPath, fmWrite)
  if f == nil:
    raise newException(IOError, "Không thể ghi: " & tmpPath)
  try:
    f.write(MAGIC_QLORA4)
    writeI32(f, int32(meta.len))
    for k, v in meta.pairs:
      writeStrLP(f, k)
      writeStrLP(f, v)
    writeI32(f, int32(fp32Tensors.len))
    for t in fp32Tensors:
      writeStrLP(f, t.name)
      writeI32(f, int32(t.shape.len))
      for s in t.shape: writeI32(f, int32(s))
      writeI64(f, int64(t.data.len))
      if t.data.len > 0:
        f.writeData(unsafeAddr t.data[0], t.data.len * 4)
    writeI32(f, int32(quantTensors.len))
    for (name, qt) in quantTensors:
      writeStrLP(f, name)
      writeQuantTensorRaw(f, qt)
  finally:
    f.close()
  # POSIX/macOS: rename là atomic ở mức pathname -> reader thấy checkpoint cũ
  # hoặc checkpoint mới hoàn chỉnh, không thấy file nửa chừng.
  if cRename(tmpPath.cstring, path.cstring) != 0:
    try: removeFile(tmpPath)
    except OSError: discard
    raise newException(IOError, "Khong the commit checkpoint atomic: " & tmpPath & " -> " & path)

proc isQLora4Bin*(path: string): bool =
  var f = newFileStream(path, fmRead)
  if f == nil: return false
  defer: f.close()
  var magic = newString(8)
  if f.readData(addr magic[0], 8) != 8: return false
  result = magic == MAGIC_QLORA4

proc loadQLora4BinDequant*(path: string): tuple[meta: Table[string, string],
                                                 tensors: seq[RawBinTensor]] =
  ## Load NIMBIN4 và dequant toàn bộ newBlock về fp32 (adapters vốn đã fp32).
  ## Kết quả tương thích loadRawBin / loadCheckpointLoraBin cũ.
  var f = newFileStream(path, fmRead)
  if f == nil:
    raise newException(IOError, "Không đọc được: " & path)
  defer: f.close()
  var magic = newString(8)
  if f.readData(addr magic[0], 8) != 8 or magic != MAGIC_QLORA4:
    raise newException(IOError, "Không phải NIMBIN4: " & path)

  proc readI32Local(): int32 =
    var x: int32 = 0
    if f.readData(addr x, 4) != 4:
      raise newException(IOError, "EOF int32")
    result = x
  proc readI64Local(): int64 =
    var x: int64 = 0
    if f.readData(addr x, 8) != 8:
      raise newException(IOError, "EOF int64")
    result = x
  proc readStrLocal(): string =
    let n = readI32Local().int
    result = newString(n)
    if n > 0 and f.readData(addr result[0], n) != n:
      raise newException(IOError, "EOF string")

  result.meta = initTable[string, string]()
  let nMeta = readI32Local().int
  for i in 0 ..< nMeta:
    let k = readStrLocal()
    let v = readStrLocal()
    result.meta[k] = v

  let nFp32 = readI32Local().int
  for i in 0 ..< nFp32:
    var t: RawBinTensor
    t.name = readStrLocal()
    let nd = readI32Local().int
    t.shape = newSeq[int](nd)
    for d in 0 ..< nd: t.shape[d] = readI32Local().int
    let ne = readI64Local().int
    t.data = newSeq[float32](ne)
    if ne > 0 and f.readData(addr t.data[0], ne * 4) != ne * 4:
      raise newException(IOError, "EOF fp32 data: " & t.name)
    result.tensors.add(t)

  let nQuant = readI32Local().int
  for i in 0 ..< nQuant:
    let name = readStrLocal()
    # Đọc QuantTensor (layout giống quant.nim writeQuantTensor)
    var qt: QuantTensor
    var kindByte: uint8
    if f.readData(addr kindByte, 1) != 1:
      raise newException(IOError, "EOF kind")
    qt.kind = QuantKind(kindByte)
    let ndims = readI32Local().int
    qt.shape = newSeq[int](ndims)
    for d in 0 ..< ndims: qt.shape[d] = readI32Local().int
    let nScales = readI32Local().int
    qt.scale = newSeq[float32](nScales)
    if nScales > 0 and f.readData(addr qt.scale[0], nScales * 4) != nScales * 4:
      raise newException(IOError, "EOF scale")
    qt.zero_point = newSeq[float32](nScales)
    if nScales > 0 and f.readData(addr qt.zero_point[0], nScales * 4) != nScales * 4:
      raise newException(IOError, "EOF zero_point")
    qt.groupSize = readI32Local().int
    let eb = readI32Local().int
    let mb = readI32Local().int
    qt.cf = newCustomFloat(max(eb, 1), mb)
    let dlen = readI32Local().int
    qt.data = newSeq[uint8](dlen)
    if dlen > 0 and f.readData(addr qt.data[0], dlen) != dlen:
      raise newException(IOError, "EOF quant data: " & name)
    let deq = dequantizeTensor(qt)
    result.tensors.add(RawBinTensor(name: name, shape: qt.shape, data: deq))

proc loadQLora4Bin*(path: string): tuple[meta: Table[string, string],
                                          fp32Tensors: seq[RawBinTensor],
                                          quantWeights: Table[string, QuantTensor]] =
  ## SUA (bug RAM): loadQLora4BinDequant (phia tren) dequant TOAN BO newBlock
  ## ve fp32 NGAY LUC LOAD roi giu resident SUOT doi song process - dung y
  ## het van de ma quant-resident (quantBase/mergeIn/restoreBase trong
  ## lora.nim) da giai quyet cho base Mamba, chi khac la ap dung nham cho
  ## newBlock. Ham nay: doc dung 1 lan tu dia (khong doi dinh dang file),
  ## nhung KHONG goi dequantizeTensor - tra newBlock ve dang QuantTensor
  ## (con nen) trong bang rieng `quantWeights`, key la TEN THAT cua tham so
  ## (da bo tien to "newBlock::") de khop truc tiep voi paramList(m)/byName.
  ## Ben goi (loadBinaryLoraForInfer/loadCheckpointLoraBin) chi can gop bang
  ## nay vao lora.quantBase (cung bang voi quantResident cua base .nimq) -
  ## roi mergeIn()/restoreBase() (hoac forwardStepJambaWithLora cho pattern
  ## thuan 'R') se TU DEQUANT-ON-THE-FLY + GIAI PHONG y het nhu base, khong
  ## can code rieng. newBlock chi con nam trong RAM o dang fp32 trong LUC
  ## thuc su forward (mergeIn...restoreBase), khong con "song" tu luc load
  ## cho toi khi thoat chuong trinh nua.
  var f = newFileStream(path, fmRead)
  if f == nil:
    raise newException(IOError, "Không đọc được: " & path)
  defer: f.close()
  var magic = newString(8)
  if f.readData(addr magic[0], 8) != 8 or magic != MAGIC_QLORA4:
    raise newException(IOError, "Không phải NIMBIN4: " & path)

  proc readI32Local(): int32 =
    var x: int32 = 0
    if f.readData(addr x, 4) != 4:
      raise newException(IOError, "EOF int32")
    result = x
  proc readI64Local(): int64 =
    var x: int64 = 0
    if f.readData(addr x, 8) != 8:
      raise newException(IOError, "EOF int64")
    result = x
  proc readStrLocal(): string =
    let n = readI32Local().int
    result = newString(n)
    if n > 0 and f.readData(addr result[0], n) != n:
      raise newException(IOError, "EOF string")

  result.meta = initTable[string, string]()
  let nMeta = readI32Local().int
  for i in 0 ..< nMeta:
    let k = readStrLocal()
    let v = readStrLocal()
    result.meta[k] = v

  let nFp32 = readI32Local().int
  for i in 0 ..< nFp32:
    var t: RawBinTensor
    t.name = readStrLocal()
    let nd = readI32Local().int
    t.shape = newSeq[int](nd)
    for d in 0 ..< nd: t.shape[d] = readI32Local().int
    let ne = readI64Local().int
    t.data = newSeq[float32](ne)
    if ne > 0 and f.readData(addr t.data[0], ne * 4) != ne * 4:
      raise newException(IOError, "EOF fp32 data: " & t.name)
    result.fp32Tensors.add(t)

  result.quantWeights = initTable[string, QuantTensor]()
  let nQuant = readI32Local().int
  for i in 0 ..< nQuant:
    let name = readStrLocal()
    if name in result.quantWeights:
      raise newException(IOError, "NIMBIN4 co quant tensor trung ten: " & name)
    # Doc QuantTensor (layout giong quant.nim writeQuantTensor) - GIU NGUYEN
    # dang nen, KHONG dequantizeTensor o day (khac loadQLora4BinDequant).
    var qt: QuantTensor
    var kindByte: uint8
    if f.readData(addr kindByte, 1) != 1:
      raise newException(IOError, "EOF kind")
    qt.kind = QuantKind(kindByte)
    let ndims = readI32Local().int
    qt.shape = newSeq[int](ndims)
    for d in 0 ..< ndims: qt.shape[d] = readI32Local().int
    let nScales = readI32Local().int
    qt.scale = newSeq[float32](nScales)
    if nScales > 0 and f.readData(addr qt.scale[0], nScales * 4) != nScales * 4:
      raise newException(IOError, "EOF scale")
    qt.zero_point = newSeq[float32](nScales)
    if nScales > 0 and f.readData(addr qt.zero_point[0], nScales * 4) != nScales * 4:
      raise newException(IOError, "EOF zero_point")
    qt.groupSize = readI32Local().int
    let eb = readI32Local().int
    let mb = readI32Local().int
    qt.cf = newCustomFloat(max(eb, 1), mb)
    let dlen = readI32Local().int
    qt.data = newSeq[uint8](dlen)
    if dlen > 0 and f.readData(addr qt.data[0], dlen) != dlen:
      raise newException(IOError, "EOF quant data: " & name)
    # name luu trong file la "newBlock::<ten that>" (xem convert()) - bo tien
    # to de khop paramList(m)/byName nhu quantResident cua base .nimq.
    let realName = if name.startsWith("newBlock::"): name[10 .. ^1] else: name
    result.quantWeights[realName] = qt

proc humanSize(bytes: int64): string =
  if bytes >= 1_073_741_824: return &"{bytes.float / 1_073_741_824.0:.2f} GB"
  if bytes >= 1_048_576: return &"{bytes.float / 1_048_576.0:.2f} MB"
  if bytes >= 1024: return &"{bytes.float / 1024.0:.1f} KB"
  return $bytes & " B"

proc convert*(src, dst: string, kind: QuantKind = qkInt4Asymmetric, groupSize: int = DefaultGroupSize) =
  if not fileExists(src):
    raise newException(IOError, "Không tìm thấy: " & src)
  if isQLora4Bin(src):
    echo "[qlora4] File nguồn đã là NIMBIN4 — bỏ qua (đã quantize rồi)."
    return

  let overwrite = (absolutePath(src) == absolutePath(dst))
  echo &"[qlora4] Đọc checkpoint fp32: {src}" & (if overwrite: " (sẽ GHI ĐÈ cùng file)" else: "")
  let srcSz = getFileSize(src)
  let loaded = loadRawBin(src)
  var meta = loaded.meta
  meta["format"] = "nimformers-lora-v1-qlora4"
  meta["quantKind"] = $kind
  meta["quantGroupSize"] = $groupSize

  var fp32Keep: seq[RawBinTensor] = @[]
  var quantList: seq[(string, QuantTensor)] = @[]

  for t in loaded.tensors:
    if t.name.startsWith("adapter::"):
      # Giữ fp32 cho A/B
      fp32Keep.add(t)
    elif t.name.startsWith("newBlock::"):
      if t.data.len == 0:
        echo &"[qlora4] Bỏ qua tensor rỗng: {t.name}"
        continue
      # FIX: 1-D (bias/gamma/beta) giữ fp32 raw, chỉ 2-D chia hết 32 mới quantize 4-bit
      # tránh sai số lượng tử 13-15% RMS làm hỏng LayerNorm.
      # SUA2: attn.* / fuseGate.* train full từ 0 với lr nhỏ → giá trị gần nhiễu
      # FP4-KM; giữ fp32 raw (format mixed vẫn hỗ trợ FP4-KM cho tensor khác).
      let bare = if t.name.startsWith("newBlock::"): t.name[10 .. ^1] else: t.name
      let isSensitiveFullTrain =
        bare.contains(".attn.") or bare.contains(".fuseGate.")
      if (not isSensitiveFullTrain) and
         t.shape.len == 2 and t.shape[1] mod 32 == 0:
        let qt = quantizeNewBlock(t.data, t.shape, kind, groupSize)
        quantList.add((t.name, qt))
        let ratio = if t.data.len > 0: (t.data.len * 4).float / max(qt.data.len.float, 1.0) else: 0.0
        echo &"[qlora4]   {t.name}: shape={t.shape}  {humanSize(int64(t.data.len)*4)} -> quant {qt.kind} {humanSize(int64(qt.data.len))} (~{ratio:.1f}x)"
      else:
        fp32Keep.add(t)
        let reason = if isSensitiveFullTrain: "attn/fuseGate full-train (fp32)"
                     else: "1D/bias/gamma"
        echo &"[qlora4]   {t.name}: shape={t.shape}  giữ nguyên FP32 raw ({reason})"
    else:
      # Giữ nguyên (an toàn)
      fp32Keep.add(t)

  # Ghi đè an toàn: luôn ghi file tạm rồi rename → tránh hỏng file giữa chừng
  let outPath = if overwrite: dst & ".qlora4.tmp" else: dst
  echo &"[qlora4] Ghi {outPath} (adapters fp32={fp32Keep.len}, newBlock 4bit={quantList.len})"
  saveQLora4Bin(outPath, meta, fp32Keep, quantList)

  if overwrite:
    moveFile(outPath, dst)
    echo &"[qlora4] Đã GHI ĐÈ {dst}"
  let dstSz = getFileSize(dst)
  echo &"[qlora4] Xong. {humanSize(srcSz)} -> {humanSize(dstSz)} (giảm ~{(1.0 - dstSz.float/max(srcSz.float,1.0))*100:.0f}%)"
  echo "[qlora4] Magic NIMBIN4 — finetune/infer load được ngay."

when isMainModule:
  # Mặc định ghi đè cùng file:
  #   ./quantize_lora_bin_4bit jamba_lora.bin
  #   ./quantize_lora_bin_4bit jamba_lora.bin jamba_lora.bin
  let src = if paramCount() >= 1: paramStr(1) else: "jamba_lora.bin"
  let dst = if paramCount() >= 2: paramStr(2) else: src
  var kind = qkInt4Asymmetric
  var gs = DefaultGroupSize
  for i in 3 .. paramCount():
    let a = paramStr(i)
    if a == "--int4": kind = qkInt4
    elif a == "--int4asym": kind = qkInt4Asymmetric
    elif a == "--fp4km": kind = qkFp4KM
    elif a.startsWith("--group="): gs = parseInt(a[8 .. ^1])
  convert(src, dst, kind, gs)
