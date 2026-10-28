## Read-only, reference-counted model mappings. Views retain the file mapping;
## metadata is copied, tensor payloads stay in the OS page cache.
import std/[memfiles, os]

type
  MappingObj = object
    file: MemFile
  WeightMapping* = ref MappingObj
  MappedView*[T] = object
    owner*: WeightMapping
    offset*, len*: int
  WeightCursor* = object
    owner*: WeightMapping
    pos*: int

proc `=destroy`(m: var MappingObj) =
  if m.file.mem != nil:
    try: m.file.close()
    except OSError: discard

proc mapWeights*(path: string): WeightMapping =
  if getFileSize(path) <= 0: raise newException(IOError, "Empty weight file: " & path)
  new(result)
  result.file = memfiles.open(path, mode = fmRead)

proc checkRange*(m: WeightMapping; offset, bytes: int) =
  if m == nil or offset < 0 or bytes < 0 or offset > m.file.size or
      bytes > m.file.size - offset:
    raise newException(IOError, "Truncated or invalid weight payload")

proc mappedView*[T](m: WeightMapping; offset, count: int): MappedView[T] =
  if count < 0 or count > high(int) div sizeof(T):
    raise newException(IOError, "Invalid tensor length")
  m.checkRange(offset, count * sizeof(T))
  MappedView[T](owner: m, offset: offset, len: count)

proc dataPtr*[T](v: MappedView[T]): ptr UncheckedArray[T] =
  if v.owner != nil:
    cast[ptr UncheckedArray[T]](cast[uint](v.owner.file.mem) + uint(v.offset))
  else: nil

template values*[T](v: MappedView[T]): untyped =
  toOpenArray(v.dataPtr, 0, v.len - 1)

template weightValues*[T](owned: seq[T]; mapped: MappedView[T]): untyped =
  toOpenArray((if mapped.owner != nil: mapped.dataPtr
              elif owned.len > 0: cast[ptr UncheckedArray[T]](unsafeAddr owned[0])
              else: cast[ptr UncheckedArray[T]](nil)),
             0, (if mapped.owner != nil: mapped.len else: owned.len) - 1)

proc readValue*[T](c: var WeightCursor): T =
  c.owner.checkRange(c.pos, sizeof(T))
  let view = mappedView[uint8](c.owner, c.pos, sizeof(T))
  copyMem(addr result, view.dataPtr, sizeof(T))
  c.pos += sizeof(T)

proc readCount*(c: var WeightCursor; width = 1): int =
  result = int(readValue[int32](c))
  if width <= 0 or result < 0 or result > high(int) div width:
    raise newException(IOError, "Invalid weight count")
  c.owner.checkRange(c.pos, result * width)

proc readString*(c: var WeightCursor): string =
  let n = c.readCount()
  result = newString(n)
  if n > 0: copyMem(addr result[0], mappedView[uint8](c.owner, c.pos, n).dataPtr, n)
  c.pos += n

proc readSeq*[T](c: var WeightCursor; n: int): seq[T] =
  let v = mappedView[T](c.owner, c.pos, n)
  result = newSeq[T](n)
  if n > 0: copyMem(addr result[0], v.dataPtr, n * sizeof(T))
  c.pos += n * sizeof(T)
