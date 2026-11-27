## Encode UTF-8 documents arriving on stdin with the Spectron tokenizer.
## Each input record has a little-endian uint32 byte count followed by the text.
## Stdout is a little-endian bit stream. The legacy default is 16 bits per ID.
import std/[os, strutils, tables]
import test_nimformer

const CacheSlots = 1 shl 17

type CachedPiece = object
  text: string
  ids: seq[uint16]

proc encodeInto(tok: BPETokenizer, cache: var seq[CachedPiece],
                text: string, ids: var seq[uint16]) =
  ids.setLen(0)
  for (start, stop) in bpePieces(text):
    var hash = 2166136261'u32
    for i in start ..< stop:
      hash = (hash xor uint32(text[i])) * 16777619'u32
    let slot = int(hash and (CacheSlots - 1).uint32)
    let length = stop - start
    if cache[slot].text.len != length or
        not equalMem(unsafeAddr(text[start]), unsafeAddr(cache[slot].text[0]), length):
      let piece = text[start ..< stop]
      cache[slot].text = piece
      cache[slot].ids.setLen(0)
      for id in tok.encodeBpePiece(piece):
        cache[slot].ids.add(uint16(id))
    ids.add(cache[slot].ids)

proc main() =
  if paramCount() notin 2 .. 3:
    quit("usage: pack_tokens TOKENIZER.json TOKEN_COUNT|all [BITS_PER_TOKEN]", 1)
  let tok = loadBPE(paramStr(1))
  let unlimited = paramStr(2) == "all"
  let quota = if unlimited: int64.high else: parseBiggestInt(paramStr(2))
  let bits = if paramCount() == 3: parseInt(paramStr(3)) else: 16
  if bits notin 8 .. 16 or tok.vocabSize notin 256 .. (1 shl bits) or quota <= 0:
    quit("Invalid token count or bits per token for this vocabulary", 1)
  for byte in 0 .. 255:
    if tok.byteEncoder[byte] notin tok.vocab:
      quit("Tokenizer is missing a base byte", 1)
  var cache = newSeq[CachedPiece](CacheSlots)
  var ids: seq[uint16]
  var bytes: seq[uint8]
  var bitBuffer: uint32
  var heldBits = 0
  var text = ""
  var written = 0'i64
  while written < quota:
    var header: array[4, uint8]
    let size = stdin.readBuffer(addr header[0], header.len)
    if size == 0:
      if unlimited: break
      quit("Input ended after " & $written & " of " & $quota & " tokens", 1)
    if size != header.len:
      quit("Truncated document header", 1)
    let length = int(header[0]) or (int(header[1]) shl 8) or
                 (int(header[2]) shl 16) or (int(header[3]) shl 24)
    if length notin 1 .. (32 * 1024 * 1024):
      quit("Invalid document size", 1)
    text.setLen(length)
    if stdin.readBuffer(addr text[0], length) != length:
      quit("Truncated document", 1)
    tok.encodeInto(cache, text, ids)
    let count = int(min(quota - written, int64(ids.len)))
    if bits == 16:
      when cpuEndian == bigEndian:
        for i in 0 ..< count:
          ids[i] = (ids[i] shl 8) or (ids[i] shr 8)
      if count > 0 and stdout.writeBuffer(addr ids[0], count * 2) != count * 2:
        quit("Failed to write token stream", 1)
    else:
      bytes.setLen((heldBits + count * bits) div 8)
      var position = 0
      for i in 0 ..< count:
        bitBuffer = bitBuffer or (uint32(ids[i]) shl heldBits)
        heldBits += bits
        while heldBits >= 8:
          bytes[position] = uint8(bitBuffer and 255)
          inc position
          heldBits -= 8
          bitBuffer = bitBuffer shr 8
      if position > 0 and stdout.writeBuffer(addr bytes[0], position) != position:
        quit("Failed to write token stream", 1)
    written += count
  if heldBits > 0:
    var lastByte = uint8(bitBuffer)
    if stdout.writeBuffer(addr lastByte, 1) != 1:
      quit("Failed to write the final token byte", 1)
  stdout.flushFile()
  if unlimited or paramCount() == 3:
    stderr.writeLine(written)

main()
