## Independent ModelProto fixtures: no tokenizer assets or external packages needed.
import std/strutils
import sentencepiece_native

proc varint(n: uint64): string =
  var value = n
  while value >= 128:
    result.add(char((value and 127) or 128))
    value = value shr 7
  result.add(char(value))

proc number(field: int, value: uint64): string =
  varint(uint64(field shl 3)) & varint(value)

proc bytes(field: int, value: string): string =
  varint(uint64((field shl 3) or 2)) & varint(uint64(value.len)) & value

proc piece(text: string, score: float32, kind = 1): string =
  var data = bytes(1, text) & varint(21) # field 2, fixed32 score
  let bits = cast[uint32](score)
  for i in 0 ..< 4: data.add(char((bits shr (8*i)) and 255))
  # Omit NORMAL to exercise the proto2 default, as real ModelProto files do.
  if kind != 1: data.add(number(3, uint64(kind)))
  bytes(1, data)

proc fixture(algorithm = 2, abKind = 1, includeAbc = true,
             useBytes = false, tie = false): string =
  result = piece("<unk>", 0, 2) & piece("<s>", 0, 3) & piece("</s>", 0, 3) &
           piece("a", 0) & piece("b", 0) & piece("c", 0) &
           piece("ab", 9, abKind) & piece("bc", if tie: 9'f32 else: 8'f32)
  if includeAbc: result.add(piece("abc", 7))
  var trainer = number(3, uint64(algorithm))
  if useBytes:
    trainer.add(number(35, 1))
    for i in 0 ..< 256: result.add(piece("<0x" & toHex(i, 2) & ">", 0, 6))
  result.add(bytes(2, trainer))
  result.add(bytes(3, bytes(1, "identity") & number(3, 0) & number(4, 0) & number(5, 1)))

let bpe = parseSentencePieceModel(fixture())
doAssert bpe.modelType == 2 and bpe.bosId == 1 and bpe.eosId == 2
doAssert bpe.encodeSentencePiece("abc") == @[8] # greedy BPE merges ab, then abc
let unigram = parseSentencePieceModel(fixture(algorithm = 1))
doAssert unigram.encodeSentencePiece("abc") == @[6, 5] # best sum is ab+c
let tied = parseSentencePieceModel(fixture(includeAbc = false, tie = true))
doAssert tied.encodeSentencePiece("abc") == @[6, 5] # equal BPE rank: leftmost first
let unused = parseSentencePieceModel(fixture(abKind = 5))
doAssert unused.encodeSentencePiece("ab") == @[3, 4]
doAssert unused.encodeSentencePiece("abc") == @[8]
let userDefined = parseSentencePieceModel(fixture(abKind = 4))
doAssert userDefined.encodeSentencePiece("abc") == @[6, 5]
doAssert bpe.encodeSentencePiece("???") == @[0] # adjacent unknowns coalesce
doAssert bpe.decodeSentencePiece(@[0]) == " ⁇ "
doAssert bpe.decodeSentencePiece(@[1, 8, 2]) == "abc"
doAssert bpe.encodeSentencePiece("") == @[]
doAssert bpe.normalizeSentencePiece("  a\t\nＡﬁ") == "▁▁a\t\nＡﬁ"
doAssert bpe.normalizeSentencePiece("\xff") == "\xef\xbf\xbd"
let byteModel = parseSentencePieceModel(fixture(useBytes = true))
doAssert byteModel.encodeSentencePiece("🙂") == @[9+0xf0, 9+0x9f, 9+0x99, 9+0x82]
doAssert byteModel.decodeSentencePiece(@[9+0xf0, 9+0x9f, 9+0x99, 9+0x82]) == "🙂"
doAssert byteModel.decodeSentencePiece(@[9+0xc0, 9+0x80]) == "\xef\xbf\xbd\xef\xbf\xbd"
doAssert byteModel.encodeSentencePiece("\xff") == @[9+0xef, 9+0xbf, 9+0xbd]
for broken in ["", "\x0a\x08x", "\x00", repeat('\xff', 11),
               piece("<unk>", 0, 2) & bytes(3, bytes(2, "\x01\x00\x00\x00"))]:
  var rejected = false
  try: discard parseSentencePieceModel(broken)
  except ValueError: rejected = true
  doAssert rejected, "malformed ModelProto was accepted"
echo "PASS: pure Nim BPE/Unigram, ranks, UNUSED, user symbols, bytes, UTF-8 and malformed models"
