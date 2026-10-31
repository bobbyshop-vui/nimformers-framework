## Convert a .nimq (NIMQ2) to NIMQ3 in place, so nim_inference can hand the
## model mmap to a unified-memory GPU. No second copy of the model is written;
## the file only grows by the alignment padding.
##
## NIMQ3 only inserts padding, so every NIMQ2 byte moves forward. Segments are
## moved from the end of the file to the start, each in chunks no larger than
## its shift: a chunk never overwrites bytes that still have to be read, and
## re-copying a chunk after a crash reads the same source bytes. A small
## journal (<file>.nimq3-progress) keeps the plan, a hash of every segment and
## the progress; rerunning the same command resumes. While converting, the
## magic reads "NIMQ!" so no loader takes the half-moved file for a model.
##
##   nim c -d:release --path:. test/deepseek-inference/convert_nimq_aligned.nim
##   ./convert_nimq_aligned test/deepseek-inference/models/model.nimq [align=65536]
import std/[os, strutils, strformat, hashes, posix]
import quant, mapped_weights

type
  Segment = object
    oldOff, newOff, len, digest: int64
  Journal = object
    align, oldSize, newSize: int64
    segs: seq[Segment]
    nextSeg, doneBytes: int64   ## segment being moved (counting down) and bytes moved from its end

const
  JournalMagic = "NQ3J"
  MaxChunk = 64 * 1024 * 1024
  HashChunk = 4 * 1024 * 1024

proc syncFile(f: File) =
  f.flushFile()
  if fsync(getFileHandle(f)) != 0: raiseOSError(osLastError())

proc readAt(f: File; off: int64; buf: var seq[uint8]; n: int) =
  f.setFilePos(off)
  if f.readBuffer(addr buf[0], n) != n: raise newException(IOError, "short read at " & $off)

proc writeAt(f: File; off: int64; data: pointer; n: int) =
  f.setFilePos(off)
  if f.writeBuffer(data, n) != n: raise newException(IOError, "short write at " & $off)

proc hashRange(f: File; off, len: int64; buf: var seq[uint8]): int64 =
  var h: Hash = 0
  var pos = off
  while pos < off + len:
    let n = int(min(int64(HashChunk), off + len - pos))
    f.readAt(pos, buf, n)
    h = h !& hash(buf.toOpenArray(0, n - 1))
    pos += n
  int64(!$h)

proc plan(path: string; align: int): Journal =
  ## Old and new offsets of every NIMQ2 segment: the arch/count header, then
  ## per tensor its name+metadata and its packed payload.
  result.align = align
  result.oldSize = getFileSize(path)
  var c = WeightCursor(owner: mapWeights(path))
  let magic = c.readString()
  if magic == MagicAligned: quit path & " is already NIMQ3", 0
  if magic != "NIMQ2": raise newException(IOError, path & " is not a NIMQ2 checkpoint")
  let headerStart = c.pos
  for i in 0 ..< 5: discard readValue[int32](c)
  let count = c.readCount()
  result.segs.add Segment(oldOff: headerStart, newOff: align, len: c.pos - headerStart)
  var pos = int64(align + c.pos - headerStart)
  for i in 0 ..< count:
    let recordStart = c.pos
    discard c.readString()
    let qt = readMappedQuantTensor(c, mapped = true)
    let payload = qt.mappedData.offset
    result.segs.add Segment(oldOff: recordStart, newOff: pos, len: payload - recordStart)
    pos += payload - recordStart
    pos = (pos + align - 1) div align * align
    result.segs.add Segment(oldOff: payload, newOff: pos, len: qt.mappedData.len)
    pos += qt.mappedData.len
  if c.pos != result.oldSize: raise newException(IOError, "trailing bytes after the last tensor")
  result.newSize = (pos + align - 1) div align * align
  for s in result.segs:
    doAssert s.newOff >= s.oldOff + align - 9, "NIMQ3 must only move data forward"
  result.nextSeg = result.segs.high
  result.doneBytes = 0

proc progressOffset(j: Journal): int64 = int64(JournalMagic.len + 8 * 4 + j.segs.len * 32)

proc saveJournal(path: string; j: Journal) =
  let tmp = path & ".tmp"
  var f = open(tmp, fmWrite)
  f.write(JournalMagic)
  for v in [j.align, j.oldSize, j.newSize, int64(j.segs.len)]:
    var x = v
    discard f.writeBuffer(addr x, 8)
  for s in j.segs:
    var x = s
    discard f.writeBuffer(addr x, sizeof(Segment))
  for v in [j.nextSeg, j.doneBytes]:
    var x = v
    discard f.writeBuffer(addr x, 8)
  f.syncFile()
  f.close()
  moveFile(tmp, path)

proc loadJournal(path: string): Journal =
  var f = open(path, fmRead)
  defer: f.close()
  var magic = newString(JournalMagic.len)
  if f.readChars(toOpenArray(magic, 0, magic.high)) != magic.len or magic != JournalMagic:
    raise newException(IOError, "invalid journal " & path)
  var n: int64
  for p in [addr result.align, addr result.oldSize, addr result.newSize, addr n]:
    if f.readBuffer(p, 8) != 8: raise newException(IOError, "truncated journal " & path)
  result.segs = newSeq[Segment](int(n))
  for s in result.segs.mitems:
    if f.readBuffer(addr s, sizeof(Segment)) != sizeof(Segment):
      raise newException(IOError, "truncated journal " & path)
  for p in [addr result.nextSeg, addr result.doneBytes]:
    if f.readBuffer(p, 8) != 8: raise newException(IOError, "truncated journal " & path)

proc saveProgress(journal: File; j: Journal) =
  var progress = [j.nextSeg, j.doneBytes]
  journal.writeAt(j.progressOffset, addr progress[0], 16)
  journal.syncFile()

proc main() =
  if paramCount() < 1: quit "usage: convert_nimq_aligned <model.nimq> [align=65536]", 1
  let path = paramStr(1)
  let journalPath = path & ".nimq3-progress"
  var buf = newSeq[uint8](MaxChunk)
  var j: Journal
  if fileExists(journalPath):
    j = loadJournal(journalPath)
    echo &"Resuming conversion of {path} (segment {j.nextSeg + 1}/{j.segs.len})"
  else:
    let align = if paramCount() >= 2: parseInt(paramStr(2)) else: NimqPayloadAlign
    if align < 64 or align > (1 shl 24) or (align and (align - 1)) != 0:
      quit "align must be a power of two in 64 .. 16 MiB", 1
    j = plan(path, align)
    var f = open(path, fmRead)
    for s in j.segs.mitems: s.digest = f.hashRange(s.oldOff, s.len, buf)
    f.close()
    echo &"{path}: {j.segs.len div 2} tensors, {j.oldSize div (1 shl 20)} MiB -> " &
      &"{j.newSize div (1 shl 20)} MiB (+{(j.newSize - j.oldSize) div 1024} KiB padding)"
    saveJournal(journalPath, j)
  var f = open(path, fmReadWriteExisting)
  defer: f.close()
  var journal = open(journalPath, fmReadWriteExisting)
  defer: journal.close()
  # Mark the file as unusable until the new header is written last.
  let marker = "NIMQ!"
  f.writeAt(4, unsafeAddr marker[0], marker.len)
  if getFileSize(path) < j.newSize:
    if ftruncate(getFileHandle(f), Off(j.newSize)) != 0: raiseOSError(osLastError())
  f.syncFile()
  # Test hook: stop after N chunk copies to exercise resume.
  var abortAfter = parseInt(getEnv("BYBY_NIMQ3_ABORT_AFTER_CHUNKS", "-1"))
  while j.nextSeg >= 0:
    let s = j.segs[j.nextSeg]
    let shift = s.newOff - s.oldOff
    let chunk = int(min(int64(MaxChunk), shift))
    while j.doneBytes < s.len:
      let n = int(min(int64(chunk), s.len - j.doneBytes))
      let src = s.oldOff + s.len - j.doneBytes - n
      f.readAt(src, buf, n)
      f.writeAt(src + shift, addr buf[0], n)
      f.syncFile()
      j.doneBytes += n
      journal.saveProgress(j)
      if abortAfter == 0: quit "aborted by BYBY_NIMQ3_ABORT_AFTER_CHUNKS", 3
      dec abortAfter
    dec j.nextSeg
    j.doneBytes = 0
    journal.saveProgress(j)
    if j.nextSeg mod 50 == 0: echo &"  {j.segs.len - j.nextSeg - 1}/{j.segs.len} segments moved"
  # New header: magic, alignment, zero padding up to the arch fields.
  var header = newSeq[uint8](int(j.align))
  var magic = MagicAligned
  var magicLen = int32(magic.len)
  var alignValue = int32(j.align)
  copyMem(addr header[0], addr magicLen, 4)
  copyMem(addr header[4], addr magic[0], magic.len)
  copyMem(addr header[4 + magic.len], addr alignValue, 4)
  f.writeAt(0, addr header[0], header.len)
  f.syncFile()
  for s in j.segs:
    if f.hashRange(s.newOff, s.len, buf) != s.digest:
      quit &"Verification FAILED at new offset {s.newOff}; keep {journalPath} and report this", 2
  let check = loadQuantStateDict(path, mapped = true)
  doAssert check.payloadAlign == int(j.align) and check.sd.len * 2 + 1 == j.segs.len
  for (name, qt) in check.sd:
    doAssert qt.mappedData.offset mod int(j.align) == 0, name & " payload not aligned"
  removeFile(journalPath)
  echo &"{path}: NIMQ3 in place, payload align {j.align}, {getFileSize(path) div (1 shl 20)} MiB, verified"

main()
