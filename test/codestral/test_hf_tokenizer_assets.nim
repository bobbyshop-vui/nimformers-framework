## Smoke check against the actual checkpoint tokenizer; no model/GPU allocation.
import std/[os, strutils]
import hf_tokenizer

let dir = if paramCount() > 0: paramStr(1) else: "test/codestral/models/hf"
let tok = newHfTokenizer(dir)
doAssert vocabSizeHf(tok) == 32768
# IDs must agree with the pretrained embedding and stopping rules.
doAssert tok.unkId == 0 and tok.bosId == 1 and tok.eosId == 2
doAssert isStopIdHf(tok, 2)
for sample in ["Hello world!", "Xin chào Việt Nam", "def add(a, b):\n    return a + b\n"]:
  let ids = encodeHf(tok, sample)
  doAssert ids.len > 1 and ids[0] == 1
  for id in ids: doAssert id >= 0 and id < 32768
  let decoded = decodeHf(tok, ids)
  doAssert decoded.strip == sample.strip
  let withEos = encodeHf(tok, sample, addEos = true)
  doAssert withEos[^1] == 2
  echo "PASS: ", sample.splitLines()[0], " -> ", ids.len, " tokens"
echo "PASS: checkpoint tokenizer vocab=32768 BOS=1 EOS=2 UNK=0; encode/decode and EOS"
