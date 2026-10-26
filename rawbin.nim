## rawbin.nim
## Định dạng checkpoint ".bin" THÔ (fp32, không nén/quantize) dùng RIÊNG cho
## pipeline fine-tune (graft_attention.nim / finetune_lora.nim / infer_lora.nim
## / pytorch_bin_loader.nim). Đây là module dùng chung để 2 file kia không
## phải viết trùng code đọc/ghi binary.
##
## CẢNH BÁO / QUY ƯỚC BẮT BUỘC CỦA PROJECT:
##   - .nimq (quant.nim, saveQuantStateDict/loadQuantStateDict) CHỈ dùng cho
##     nim_inference.nim và test_nimformer.nim. KHÔNG được dùng .nimq ở bất
##     kỳ đâu trong pipeline fine-tune.
##   - Toàn bộ checkpoint đọc/ghi trong graft_attention.nim, finetune_lora.nim,
##     infer_lora.nim, và bước xuất thô của pytorch_bin_loader.nim PHẢI dùng
##     .bin (định dạng trong file này). Không có QuantTensor/nén ở đây - lưu
##     thẳng float32 thô, đơn giản, dễ debug, đọc lại không mất precision.
##
## Cấu trúc file (little-endian, ghi thẳng bằng writeData/readData - không lệ
## thuộc API cấp cao có thể khác nhau giữa các bản Nim):
##   8 byte   magic "NIMBIN1\0"
##   int32    nMeta (số cặp meta string->string, vd step/pattern/…)
##   [meta]*  int32 keyLen, key bytes, int32 valLen, val bytes
##   int32    nTensors
##   [tensor]*
##       int32 nameLen, name bytes
##       int32 nDims, int32[nDims] shape
##       int64 nElems
##       float32[nElems] data thô (4 byte/phần tử, không nén)

import std/streams
import std/tables

type
  RawBinTensor* = object
    name*: string
    shape*: seq[int]
    data*: seq[float32]

const RAWBIN_MAGIC* = "NIMBIN1\0"  # đúng 8 byte

# ── helper: đọc/ghi nhị phân cấp thấp, chỉ dựa vào writeData/readData ──────
proc writeI32(s: Stream, v: int32) =
  var x = v
  s.writeData(addr x, 4)

proc readI32(s: Stream): int32 =
  var x: int32 = 0
  let got = s.readData(addr x, 4)
  if got != 4:
    raise newException(IOError, "EOF bat ngo khi doc int32 (.bin hong hoac cut)")
  result = x

proc writeI64(s: Stream, v: int64) =
  var x = v
  s.writeData(addr x, 8)

proc readI64(s: Stream): int64 =
  var x: int64 = 0
  let got = s.readData(addr x, 8)
  if got != 8:
    raise newException(IOError, "EOF bat ngo khi doc int64 (.bin hong hoac cut)")
  result = x

proc writeStrLP(s: Stream, str: string) =
  writeI32(s, int32(str.len))
  if str.len > 0:
    s.writeData(unsafeAddr str[0], str.len)

proc readStrLP(s: Stream): string =
  let n = readI32(s).int
  result = newString(n)
  if n > 0:
    let got = s.readData(addr result[0], n)
    if got != n:
      raise newException(IOError, "EOF bat ngo khi doc string do dai " & $n)

# ── API chính ───────────────────────────────────────────────────────────

proc saveRawBin*(path: string, meta: Table[string, string], tensors: seq[RawBinTensor]) =
  ## Ghi state_dict thô (meta + danh sach tensor fp32) ra 1 file .bin.
  var f = newFileStream(path, fmWrite)
  if f == nil:
    raise newException(IOError, "Khong the ghi file .bin: " & path)
  try:
    f.write(RAWBIN_MAGIC)   # string 8 byte co dinh - dung write() thay vi unsafeAddr (const khong co dia chi)
    writeI32(f, int32(meta.len))
    for k, v in meta.pairs:
      writeStrLP(f, k)
      writeStrLP(f, v)
    writeI32(f, int32(tensors.len))
    for t in tensors:
      writeStrLP(f, t.name)
      writeI32(f, int32(t.shape.len))
      for s in t.shape:
        writeI32(f, int32(s))
      writeI64(f, int64(t.data.len))
      if t.data.len > 0:
        f.writeData(unsafeAddr t.data[0], t.data.len * 4)
  finally:
    f.close()

type RawBinWriter* = object
  stream*: Stream
  tensorCountPos*: int64
  tensorCount*: int

type RawBinReader* = object
  stream*: Stream
  meta*: Table[string, string]
  remaining*: int

proc beginRawBin*(path: string, meta: Table[string, string], tensorCount: int = 0): RawBinWriter =
  result.stream = newFileStream(path, fmWrite)
  if result.stream == nil:
    raise newException(IOError, "Khong the ghi file .bin: " & path)
  result.stream.write(RAWBIN_MAGIC)
  writeI32(result.stream, int32(meta.len))
  for k, v in meta.pairs:
    writeStrLP(result.stream, k)
    writeStrLP(result.stream, v)
  result.tensorCountPos = result.stream.getPosition
  writeI32(result.stream, int32(tensorCount))
  result.tensorCount = 0

proc writeRawTensor*(writer: var RawBinWriter, tensor: RawBinTensor) =
  writeStrLP(writer.stream, tensor.name)
  writeI32(writer.stream, int32(tensor.shape.len))
  for dim in tensor.shape: writeI32(writer.stream, int32(dim))
  writeI64(writer.stream, int64(tensor.data.len))
  if tensor.data.len > 0:
    writer.stream.writeData(unsafeAddr tensor.data[0], tensor.data.len * 4)
  inc writer.tensorCount

proc finishRawBin*(writer: var RawBinWriter) =
  if writer.stream != nil:
    let endPos = writer.stream.getPosition
    writer.stream.setPosition(int(writer.tensorCountPos))
    writeI32(writer.stream, int32(writer.tensorCount))
    writer.stream.setPosition(endPos)
    writer.stream.close()
    writer.stream = nil

proc openRawBinReader*(path: string): RawBinReader =
  result.stream = newFileStream(path, fmRead)
  if result.stream == nil:
    raise newException(IOError, "Khong the doc file .bin: " & path)
  var magic = newString(8)
  if result.stream.readData(addr magic[0], 8) != 8 or magic != RAWBIN_MAGIC:
    result.stream.close()
    raise newException(IOError, "File khong dung dinh dang .bin: " & path)
  result.meta = initTable[string, string]()
  let nMeta = readI32(result.stream).int
  for _ in 0 ..< nMeta:
    result.meta[readStrLP(result.stream)] = readStrLP(result.stream)
  result.remaining = readI32(result.stream).int

proc readRawTensor*(reader: var RawBinReader): RawBinTensor =
  if reader.remaining <= 0:
    raise newException(IOError, "Khong con tensor trong .bin")
  result.name = readStrLP(reader.stream)
  let nDims = readI32(reader.stream).int
  result.shape = newSeq[int](nDims)
  for i in 0 ..< nDims: result.shape[i] = readI32(reader.stream).int
  let nElems = readI64(reader.stream).int
  result.data = newSeq[float32](nElems)
  if nElems > 0:
    let got = reader.stream.readData(addr result.data[0], nElems * 4)
    if got != nElems * 4: raise newException(IOError, "EOF khi doc tensor " & result.name)
  dec reader.remaining

proc closeRawBinReader*(reader: var RawBinReader) =
  if reader.stream != nil:
    reader.stream.close()
    reader.stream = nil

proc loadRawBin*(path: string): tuple[meta: Table[string, string], tensors: seq[RawBinTensor]] =
  ## Đọc lại 1 file .bin đã ghi bởi saveRawBin.
  var f = newFileStream(path, fmRead)
  if f == nil:
    raise newException(IOError, "Khong the doc file .bin: " & path)
  try:
    var magic = newString(8)
    let gotMagic = f.readData(addr magic[0], 8)
    if gotMagic != 8 or magic != RAWBIN_MAGIC:
      raise newException(IOError, "File khong dung dinh dang .bin cua nimformers " &
        "(magic sai, co the ban dang tro nham vao 1 file .nimq?): " & path)
    var meta = initTable[string, string]()
    let nMeta = readI32(f).int
    for _ in 0 ..< nMeta:
      let k = readStrLP(f)
      let v = readStrLP(f)
      meta[k] = v
    let nTensors = readI32(f).int
    var tensors = newSeq[RawBinTensor](nTensors)
    for i in 0 ..< nTensors:
      let name = readStrLP(f)
      let nDims = readI32(f).int
      var shape = newSeq[int](nDims)
      for d in 0 ..< nDims:
        shape[d] = readI32(f).int
      let nElems = readI64(f).int
      var data = newSeq[float32](nElems)
      if nElems > 0:
        let got = f.readData(addr data[0], nElems * 4)
        if got != nElems * 4:
          raise newException(IOError, "Doc thieu du lieu tensor '" & name & "' trong " & path)
      tensors[i] = RawBinTensor(name: name, shape: shape, data: data)
    result = (meta, tensors)
  finally:
    f.close()