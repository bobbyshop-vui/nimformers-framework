## Exact upload bytes of a trace (the "sidecar").
##
## A trace recorded to a file keeps them in `<trace>.data`. A trace recorded straight into the
## lifter (`firlift --run`) never touches the disk: the lifter receives them over a pipe and holds
## them here, in memory, under the same offsets the recorder wrote. Every reader goes through this
## module, so the lifter does not care which of the two it has.
import std/os

const MemorySource* = "(recorded in memory)"

var memoryData*: string        ## the sidecar bytes of an in-memory recording
var memoryActive* = false

proc useMemory*(data: sink string) =
  memoryData = data
  memoryActive = true

proc sidecarPath*(sourcePath: string): string = sourcePath & ".data"

proc sidecarAvailable*(sourcePath: string): bool =
  if memoryActive: return true
  sourcePath.len > 0 and fileExists(sidecarPath(sourcePath))

type Visit = proc (chunk: pointer; bytes: int): bool

proc sidecarVisit*(sourcePath: string; offset, bytes: int; visit: Visit): bool =
  ## Hand the range to `visit` in pieces of at most 1 MiB. `visit` returns false to stop; the
  ## result is false when it did, or when the range lies outside the sidecar.
  if bytes <= 0: return true
  if offset < 0: return false
  if memoryActive:
    if offset + bytes > memoryData.len: return false
    var at = offset
    var left = bytes
    while left > 0:
      let n = min(left, 1 shl 20)
      if not visit(addr memoryData[at], n): return false
      at += n
      left -= n
    return true
  if not sidecarAvailable(sourcePath): return false
  let f = open(sidecarPath(sourcePath), fmRead)
  defer: f.close()
  f.setFilePos(offset)
  var buffer = newString(min(bytes, 1 shl 20))
  var left = bytes
  while left > 0:
    let n = min(buffer.len, left)
    if f.readBuffer(addr buffer[0], n) != n: return false
    if not visit(addr buffer[0], n): return false
    left -= n
  true

proc sidecarAllZero*(sourcePath: string; offset, bytes: int): bool =
  ## True when the range is present and every byte of it is zero.
  if bytes <= 0: return false
  sidecarVisit(sourcePath, offset, bytes, proc (chunk: pointer; n: int): bool =
    let p = cast[ptr UncheckedArray[byte]](chunk)
    for i in 0 ..< n:
      if p[i] != 0: return false
    true)

proc sidecarFnv*(sourcePath: string; offset, bytes: int): string =
  ## FNV-1a of the range as 16 hex digits; raises when the range is not in the sidecar.
  var hash = 0xcbf29ce484222325'u64
  let ok = sidecarVisit(sourcePath, offset, bytes, proc (chunk: pointer; n: int): bool =
    let p = cast[ptr UncheckedArray[byte]](chunk)
    for i in 0 ..< n:
      hash = (hash xor uint64(p[i])) * 0x100000001b3'u64
    true)
  if not ok: raise newException(IOError, "truncated probe sidecar")
  const digits = "0123456789ABCDEF"
  result = newString(16)
  for i in 0 ..< 16:
    result[i] = digits[int((hash shr (60 - 4*i)) and 0xf)]

proc sidecarWrite*(sourcePath: string; offset, bytes: int; target: File) =
  ## Copy the range to `target`.
  let ok = sidecarVisit(sourcePath, offset, bytes, proc (chunk: pointer; n: int): bool =
    if target.writeBuffer(chunk, n) != n:
      raise newException(IOError, "cannot write compacted trace.data")
    true)
  if not ok: raise newException(IOError, "truncated probe sidecar")
