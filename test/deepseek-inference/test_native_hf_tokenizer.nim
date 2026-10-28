## Native tokenizer regressions, independent of Python and checkpoint weights.
## Optional argv[1]: local DeepSeek Coder 6.7B tokenizer directory for exact IDs.
import std/[json, os, strutils, tables, unicode]
import native_hf_tokenizer, tokenizer_chat_template, tokenizer_regex, tokenizer_unicode

proc fixtures() =
  let folder = getTempDir() / ("nim-native-hf-tokenizer-" & $getCurrentProcessId())
  createDir(folder)
  defer: removeDir(folder)
  var vocabulary = newJObject()
  var next = 256
  for b in 0..<256:
    var cp = b
    if b notin 33..126 and b notin 161..172 and b notin 174..255: cp = next; inc next
    vocabulary[$Rune(cp)] = %b
  for i, piece in ["He", "ll", "Hell", "Hello"]: vocabulary[piece] = %(256+i)
  let tokenizer = %*{
    "model": {"type":"BPE", "vocab":vocabulary, "merges":["H e", "l l", "He ll", "Hell o"], "unk_token":nil},
    "normalizer":{"type":"Sequence", "normalizers":[]},
    "pre_tokenizer":{"type":"Sequence", "pretokenizers":[
      {"type":"Split", "pattern":{"Regex":"[\r\n]"}, "behavior":"Isolated", "invert":false},
      {"type":"Split", "pattern":{"Regex":"\\s?\\p{L}+"}, "behavior":"Isolated", "invert":false},
      {"type":"Digits", "individual_digits":true},
      {"type":"ByteLevel", "add_prefix_space":false, "use_regex":false, "trim_offsets":false}]},
    "post_processor":{"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":false},
    "decoder":{"type":"ByteLevel"},
    "added_tokens":[
      {"id":260, "content":"<s>", "special":true, "normalized":false},
      {"id":261, "content":"</s>", "special":true, "normalized":false},
      {"id":262, "content":"[TAG]", "special":false, "normalized":false, "lstrip":true, "rstrip":true}]
  }
  let config = %*{"tokenizer_class":"LlamaTokenizerFast", "bos_token":"<s>", "eos_token":"</s>",
    "add_bos_token":true, "add_eos_token":false, "clean_up_tokenization_spaces":false,
    "chat_template":"{%- for message in messages %}{{ message['role'] + ':' + message['content'] + '\\n' }}{%- endfor %}{{'assistant:\\n'}}\n"}
  writeFile(folder/"tokenizer.json", $tokenizer)
  writeFile(folder/"tokenizer_config.json", $config)
  let tok = newHFTokenizer(folder)
  doAssert tok.vocabSize == 260 and tok.totalVocabSize == 263
  doAssert tok.bosId == 260 and tok.eosId == 261
  doAssert tok.encode("Hello!\n    Hello") == @[260,259,33,10,32,32,32,32,259]
  doAssert tok.encode("Hello", addSpecialTokens=false) == @[259]
  doAssert tok.decode(@[260,259,261]) == "Hello"
  doAssert tok.decode(@[260,259,261], skipSpecialTokens=false) == "<s>Hello</s>"
  doAssert tok.encode("A  [TAG]  B", addSpecialTokens=false) == @[65,262,66]
  doAssert tok.decode(@[65,262,66]) == "A[TAG]B"
  doAssert tok.decode(@[0xe2,0x82]) == "�" # one truncated UTF-8 subsequence
  doAssert tok.decode(@[0xe2,0x82,65]) == "�A"
  var stream = initTokenStream(tok)
  doAssert stream.pushTokens(@[260,0xf0]) == ""
  doAssert stream.pushTokens(@[0x9f]) == ""
  doAssert stream.pushTokens(@[0x99]) == ""
  doAssert stream.pushTokens(@[0x82]) == "🙂"
  doAssert stream.pushTokens(@[0xe2,0x82]) == ""
  doAssert stream.pushTokens([], final=true) == "�"
  for text in ["Hello!", "Xin chào Việt Nam", "  a\t\tb\n\n", "🙂 漢字 é", "a\x00b"]:
    let ids = tok.encode(text)
    doAssert tok.decode(ids) == text
    var incremental = initTokenStream(tok)
    var actual = ""
    for id in ids: actual.add(incremental.pushTokens(@[id]))
    actual.add(incremental.pushTokens([], final=true))
    doAssert actual == text
  let rendered = tok.formatChatTemplate(%*[{"role":"system", "content":"S"}, {"role":"user", "content":"Hi"}])
  doAssert rendered == "system:S\nuser:Hi\nassistant:\n"
  doAssert tok.applyChatTemplate("Hi", "S") == tok.encode(rendered, addSpecialTokens=false)
  var env = initTable[string, JsonNode]()
  env["messages"] = %*[{"role":"user"}]
  let scoped = compileChatTemplate("{%- set name = 'outer' -%}{%- for message in messages -%}{%- set name = 'inner' -%}{%- endfor -%}{{name}}\n")
  doAssert renderChatTemplate(scoped, env) == "outer"
  doAssert unicodeLower("İ Ⅶ Ａ") == "i̇ ⅶ ａ"
  doAssert unicodeStrip("\u00a0\t A\u3000") == "A"
  let regex = compileTokenizerRegex("\\s?\\p{L}+|\\p{N}{1,3}")
  let input = " Xin 12345"
  var parts: seq[string]
  for match in findTokenizerMatches(regex, input): parts.add(input[match.first..<match.last])
  doAssert parts == @[" Xin", "123", "45"]
  echo "PASS: pure Nim BPE, added tokens, BOS/EOS, Unicode, byte streaming, regex and chat templates"

proc deepseekAssets(folder: string) =
  let tok = newHFTokenizer(folder)
  doAssert tok.vocabSize == 32000 and tok.totalVocabSize == 32022
  doAssert tok.bosId == 32013 and tok.eosId == 32021
  let samples = @[
    ("Hello world!", @[32013,17535,1835,0]),
    ("Xin chào Việt Nam", @[32013,55,246,494,672,78,24297,155,119,216,83,461,302]),
    ("def add(a, b):\n    return a + b\n", @[32013,1551,957,7,64,11,270,1772,185,315,967,245,945,270,185]),
    ("  a\t\tb\n\n", @[32013,207,245,184,184,65,185,185])]
  for sample in samples:
    let ids = tok.encode(sample[0])
    doAssert ids == sample[1]
    doAssert tok.decode(ids) == sample[0]
    var stream = initTokenStream(tok)
    var decoded = ""
    for id in ids: decoded.add(stream.pushTokens(@[id]))
    decoded.add(stream.pushTokens([], final=true))
    doAssert decoded == sample[0]
  let prompt = tok.formatChatTemplate(%*[{"role":"user", "content":"Hi"}])
  doAssert prompt.endsWith("### Instruction:\nHi\n### Response:\n")
  doAssert tok.applyChatTemplate("Hi") == tok.encode(prompt, addSpecialTokens=false)
  echo "PASS: exact DeepSeek tokenizer IDs, whitespace, native streaming and chat template"

fixtures()
if paramCount() > 0: deepseekAssets(paramStr(1))
