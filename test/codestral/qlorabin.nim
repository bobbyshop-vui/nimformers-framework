## qlorabin.nim
## Self-contained QLoRA checkpoint in a single .bin file.
##
## Format: NIMQLR1\0 + metadata + tensors.
## Quantized tensors use asymmetric INT4, per-group (default group=32):
##   - packed 2 nibbles / byte
##   - one FP32 scale + zero-point / group
## Raw tensors stay FP32 (embedding, norms, biases, small state, etc.).
## LoRA A/B matrices are raw FP32 and are stored as adapter::name::A/B.
##
## This is intentionally a NEW .bin format; rawbin.nim/NIMBIN1 stays
## backward-compatible with the old FP32 checkpoints.

import std/[streams, tables]
import quant

type
  QLoraTensorKind* = enum
    qlkRaw = 0
    qlkQ4Asym = 1

  QLoraBinTensor* = object
    name*: string
    kind*: QLoraTensorKind
    shape*: seq[int]
    raw*: seq[float32]
    qt*: QuantTensor

const QLORA_BIN_MAGIC* = "NIMQLR1\0"

proc writeI32(s: Stream, v: int32) =
  var x = v
  s.writeData(addr x, 4)

proc readI32(s: Stream): int32 =
  var x: int32
  if s.readData(addr x, 4) != 4:
    raise newException(IOError, "EOF khi doc int32 QLoRA .bin")
  x

proc writeI64(s: Stream, v: int64) =
  var x = v
  s.writeData(addr x, 8)

proc readI64(s: Stream): int64 =
  var x: int64
  if s.readData(addr x, 8) != 8:
    raise newException(IOError, "EOF khi doc int64 QLoRA .bin")
  x

proc writeU8(s: Stream, v: uint8) =
  var x = v
  s.writeData(addr x, 1)

proc readU8(s: Stream): uint8 =
  var x: uint8
  if s.readData(addr x, 1) != 1:
    raise newException(IOError, "EOF khi doc uint8 QLoRA .bin")
  x

proc writeStrLP(s: Stream, str: string) =
  writeI32(s, int32(str.len))
  if str.len > 0:
    s.writeData(unsafeAddr str[0], str.len)

proc readStrLP(s: Stream): string =
  let n = readI32(s).int
  if n < 0 or n > 1024 * 1024:
    raise newException(IOError, "String length khong hop le trong QLoRA .bin")
  result = newString(n)
  if n > 0 and s.readData(addr result[0], n) != n:
    raise newException(IOError, "EOF khi doc string QLoRA .bin")

type
  QLoraBinWriter* = object
    stream*: Stream
    tensorCountPos*: int64
    tensorCount*: int

  QLoraBinReader* = object
    stream*: Stream
    meta*: Table[string, string]
    remaining*: int

proc beginQLoraBin*(path: string, meta: Table[string, string], tensorCount: int = 0): QLoraBinWriter =
  result.stream = newFileStream(path, fmWrite)
  if result.stream == nil:
    raise newException(IOError, "Khong the ghi QLoRA .bin: " & path)
  result.stream.write(QLORA_BIN_MAGIC)
  writeI32(result.stream, int32(meta.len))
  for k, v in meta.pairs:
    writeStrLP(result.stream, k)
    writeStrLP(result.stream, v)
  result.tensorCountPos = result.stream.getPosition
  writeI32(result.stream, int32(tensorCount))
  result.tensorCount = 0

proc writeQLoraRawTensor*(writer: var QLoraBinWriter, name: string,
                          shape: seq[int], data: seq[float32]) =
  writeStrLP(writer.stream, name)
  writeU8(writer.stream, uint8(qlkRaw.ord))
  writeI32(writer.stream, int32(shape.len))
  for d in shape: writeI32(writer.stream, int32(d))
  writeI64(writer.stream, int64(data.len))
  if data.len > 0:
    writer.stream.writeData(unsafeAddr data[0], data.len * 4)
  inc writer.tensorCount

proc writeQLoraQ4Tensor*(writer: var QLoraBinWriter, name: string, qt: QuantTensor) =
  doAssert qt.kind == qkInt4Asymmetric, "QLoRA .bin chi ho tro INT4 asymmetric"
  doAssert qt.shape.len == 2, "QLoRA Q4 tensor phai 2D"
  doAssert qt.groupSize > 0, "QLoRA Q4 tensor phai per-group"
  writeStrLP(writer.stream, name)
  writeU8(writer.stream, uint8(qlkQ4Asym.ord))
  writeI32(writer.stream, int32(qt.shape.len))
  for d in qt.shape: writeI32(writer.stream, int32(d))
  writeI32(writer.stream, int32(qt.groupSize))
  writeI32(writer.stream, int32(qt.scale.len))
  for v in qt.scale:
    writer.stream.writeData(unsafeAddr v, 4)
  writeI32(writer.stream, int32(qt.zero_point.len))
  for v in qt.zero_point:
    writer.stream.writeData(unsafeAddr v, 4)
  writeI64(writer.stream, int64(qt.data.len))
  if qt.data.len > 0:
    writer.stream.writeData(unsafeAddr qt.data[0], qt.data.len)
  inc writer.tensorCount

proc writeQLoraTensor*(writer: var QLoraBinWriter, t: QLoraBinTensor) =
  case t.kind
  of qlkRaw:
    writeQLoraRawTensor(writer, t.name, t.shape, t.raw)
  of qlkQ4Asym:
    writeQLoraQ4Tensor(writer, t.name, t.qt)

proc finishQLoraBin*(writer: var QLoraBinWriter) =
  if writer.stream != nil:
    let endPos = writer.stream.getPosition
    writer.stream.setPosition(writer.tensorCountPos)
    writeI32(writer.stream, int32(writer.tensorCount))
    writer.stream.setPosition(endPos)
    writer.stream.close()
    writer.stream = nil

proc saveQLoraBin*(path: string, meta: Table[string, string], tensors: seq[QLoraBinTensor]) =
  var w = beginQLoraBin(path, meta)
  try:
    for t in tensors:
      writeQLoraTensor(w, t)
    finishQLoraBin(w)
  except:
    if w.stream != nil:
      w.stream.close()
      w.stream = nil
    raise

proc openQLoraBinReader*(path: string): QLoraBinReader =
  result.stream = newFileStream(path, fmRead)
  if result.stream == nil:
    raise newException(IOError, "Khong the doc QLoRA .bin: " & path)
  var magic = newString(8)
  if result.stream.readData(addr magic[0], 8) != 8 or magic != QLORA_BIN_MAGIC:
    result.stream.close()
    raise newException(IOError, "Khong dung format NIMQLR1: " & path)
  result.meta = initTable[string, string]()
  let nMeta = readI32(result.stream).int
  if nMeta < 0 or nMeta > 10000:
    raise newException(IOError, "nMeta khong hop le trong QLoRA .bin")
  for _ in 0 ..< nMeta:
    result.meta[readStrLP(result.stream)] = readStrLP(result.stream)
  result.remaining = readI32(result.stream).int
  if result.remaining < 0:
    raise newException(IOError, "nTensors khong hop le trong QLoRA .bin")

proc readQLoraTensor*(reader: var QLoraBinReader): QLoraBinTensor =
  if reader.remaining <= 0:
    raise newException(IOError, "Khong con tensor trong QLoRA .bin")
  result.name = readStrLP(reader.stream)
  let kind = readU8(reader.stream)
  if kind > uint8(qlkQ4Asym.ord):
    raise newException(IOError, "Tensor kind khong hop le: " & $kind)
  result.kind = QLoraTensorKind(kind)
  let nDims = readI32(reader.stream).int
  if nDims < 0 or nDims > 8:
    raise newException(IOError, "nDims khong hop le: " & $nDims)
  result.shape = newSeq[int](nDims)
  for i in 0 ..< nDims:
    result.shape[i] = readI32(reader.stream).int
  case result.kind
  of qlkRaw:
    let n = readI64(reader.stream).int
    if n < 0:
      raise newException(IOError, "nElem am trong QLoRA raw tensor")
    result.raw = newSeq[float32](n)
    if n > 0 and reader.stream.readData(addr result.raw[0], n * 4) != n * 4:
      raise newException(IOError, "EOF khi doc raw tensor " & result.name)
  of qlkQ4Asym:
    if nDims != 2:
      raise newException(IOError, "Q4 tensor phai 2D: " & result.name)
    let gs = readI32(reader.stream).int
    if gs <= 0 or gs > 4096:
      raise newException(IOError, "groupSize khong hop le: " & $gs)
    let ns = readI32(reader.stream).int
    if ns < 0 or ns > 100000000:
      raise newException(IOError, "scale count khong hop le")
    result.qt.scale = newSeq[float32](ns)
    for i in 0 ..< ns:
      if reader.stream.readData(addr result.qt.scale[i], 4) != 4:
        raise newException(IOError, "EOF khi doc scales " & result.name)
    let nz = readI32(reader.stream).int
    result.qt.zero_point = newSeq[float32](nz)
    for i in 0 ..< nz:
      if reader.stream.readData(addr result.qt.zero_point[i], 4) != 4:
        raise newException(IOError, "EOF khi doc zero-points " & result.name)
    let nd = readI64(reader.stream).int
    if nd < 0:
      raise newException(IOError, "nData am trong Q4 tensor")
    result.qt.kind = qkInt4Asymmetric
    result.qt.shape = result.shape
    result.qt.groupSize = gs
    result.qt.data = newSeq[uint8](nd)
    if nd > 0 and reader.stream.readData(addr result.qt.data[0], nd) != nd:
      raise newException(IOError, "EOF khi doc Q4 data " & result.name)
  dec reader.remaining

proc closeQLoraBinReader*(reader: var QLoraBinReader) =
  if reader.stream != nil:
    reader.stream.close()
    reader.stream = nil

proc isQLoraBin*(path: string): bool =
  var f = newFileStream(path, fmRead)
  if f == nil: return false
  defer: f.close()
  var magic = newString(8)
  f.readData(addr magic[0], 8) == 8 and magic == QLORA_BIN_MAGIC
