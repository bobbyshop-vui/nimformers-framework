## Exact Codestral V3 regression IDs; loads only the tokenizer, no model/GPU.
import std/os
import hf_tokenizer

let dir = if paramCount() > 0: paramStr(1) else: "test/codestral/models/hf"
let tok = newHfTokenizer(dir)
doAssert vocabSizeHf(tok) == 32768
doAssert tok.unkId == 0 and tok.bosId == 1 and tok.eosId == 2
doAssert isStopIdHf(tok, 2)
let cases: seq[tuple[text: string, ids: seq[int]]] = @[
  ("", @[]),
  (" ", @[1027]),
  ("   ", @[1028]),
  ("\t", @[29473, 780]),
  ("\n", @[29473, 781]),
  ("\r\n", @[2070, 781]),
  ("Hello world!", @[23325, 2294, 29576]),
  ("Xin chào Việt Nam", @[2268, 1030, 1252, 29607, 29477, 11772, 30307, 29475, 17676]),
  ("def add(a, b):\n    return a + b\n", @[1569, 1735, 29500, 29476, 29493, 1055, 2097, 781, 3055, 1372, 1032, 1416, 1055, 781]),
  ("  a\t\tb\n\n", @[1027, 1032, 780, 780, 29494, 781, 781]),
  ("a\x00b", @[1032, 771, 29494]),
  ("[INST] Xin chào[/INST]", @[1501, 17057, 29561, 2268, 1030, 1252, 29607, 29477, 29560, 29516, 17057, 29561]),
  ("ＡＢＣ ① ﬁ ㍑", @[29473, 1010, 959, 932, 1010, 959, 933, 1010, 959, 934, 29473, 997, 916, 931, 29473, 30928, 29473, 998, 912, 916]),
  ("é é Å Å", @[1085, 29717, 2001, 29473, 997, 903, 942, 16512]),
  ("👩🏽‍💻🙂🫠 🧬", @[29473, 1011, 930, 916, 940, 31775, 30042, 1011, 930, 917, 958, 30108, 1011, 930, 942, 931, 29473, 1011, 930, 938, 943]),
  ("漢字 日本語 한국어", @[29473, 1001, 959, 933, 29799, 29473, 29910, 29887, 31089, 29473, 30050, 31547, 30201])
]
for sample in cases:
  doAssert encodeHf(tok, sample.text, addBos = false) == sample.ids, sample.text
  let ids = encodeHf(tok, sample.text)
  doAssert ids == @[1] & sample.ids, sample.text
  doAssert decodeHf(tok, ids) == sample.text, "whitespace/Unicode changed: " & sample.text
  doAssert encodeHf(tok, sample.text, addEos = true) == ids & @[2]
echo "PASS: ", cases.len, " exact Codestral V3 ID cases, byte fallback, whitespace, BOS and EOS"
