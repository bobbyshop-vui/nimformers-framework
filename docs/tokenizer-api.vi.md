
# API tokenizer

[English](tokenizer-api.md) | [Tiếng Việt](tokenizer-api.vi.md)


## Tokenizer Hugging Face (`hf_tokenizer.nim`)

```nim
type HfTokenizer* = object
  # loaded vocabulary, special-token IDs, and model-specific decode settings

proc newHfTokenizer*(hfDir: string): HfTokenizer
proc encodeHf*(tok: HfTokenizer, text: string,
               addEos: bool = false, addBos: bool = true): seq[int]
proc decodeHf*(tok: HfTokenizer, ids: seq[int],
               skipSpecial: bool = true): string
proc pieceOfHf*(tok: HfTokenizer, id: int): string
proc vocabSizeHf*(tok: HfTokenizer): int
proc isStopIdHf*(tok: HfTokenizer, id: int): bool
```

`hfDir` trỏ tới thư mục model/tokenizer Codestral cục bộ. Loader ưu tiên
`tokenizer.model.v3`, sau đó `tokenizer.model`. `hf_tokenizer.nim` dùng bản
Nim thuần trong `test/codestral/sentencepiece_native.nim`; build/chạy
Codestral inference không cần Nimpy, gói tokenizer Python, thư viện
SentencePiece, C++ shim hay tải thư viện tokenizer.

Bản này đọc ModelProto trực tiếp, chọn BPE/Unigram từ model, áp dụng
normalizer charsmap và cờ whitespace nhúng, xử lý byte fallback, user
symbol, control token và denormalization tùy chọn. Codestral V3 dùng BPE
với identity normalizer, giữ indentation, tab và newline. Vocabulary trie
được dựng một lần khi load.

Chạy fixture native bằng `nim c -r --path:test/codestral
test/codestral/test_sentencepiece_native.nim`. Đối chiếu ID checkpoint thật
bằng `nim c -r --path:test/codestral
test/codestral/test_hf_tokenizer_assets.nim [hfDir]`. Hai test chỉ dùng thư
viện chuẩn Nim và file cục bộ.


## Wrapper tokenizer cho LLaMA inference

`test/deepseek-inference/nim_inference.nim` có wrapper nhận đường dẫn model
và hỗ trợ chat template:

```nim
proc newHFTokenizer*(modelPath: string): HFTokenizer
proc encode*(tok: HFTokenizer, text: string,
             addSpecialTokens: bool = true): seq[int]
proc decode*(tok: HFTokenizer, ids: seq[int],
             skipSpecialTokens: bool = true): string
proc applyChatTemplate*(tok: HFTokenizer, userText: string,
                        systemText: string = ""): seq[int]
```

`HFTokenizer` được triển khai trong
`test/deepseek-inference/native_hf_tokenizer.nim` và export lại qua
`nim_inference.nim`. Nó đọc `tokenizer.json`/`tokenizer_config.json` cục bộ:
vocabulary và merge BPE byte-level, tách Unicode pre-tokenizer, added
token, BOS/EOS và quy tắc decode. `tokenizer.model` cục bộ có thể dùng
reader SentencePiece native. Regex và render template đều viết bằng Nim;
không cần PCRE, thư viện tokenizer, Nimpy hay Python runtime. Stage hoặc
cú pháp template không hỗ trợ sẽ báo lỗi rõ ràng thay vì chọn tokenizer khác.

`applyChatTemplate` render template của checkpoint theo phạm vi vòng lặp
Jinja và quy tắc whitespace, rồi encode mà không thêm special token lần
hai. Truyền ID này cho `generate` trong [API inference LLaMA](llama-inference-api.vi.md).
Decoder byte-level DeepSeek stream bytes mới một lần, giữ UTF-8 chưa đủ
đến token tiếp theo thay vì decode toàn lịch sử mỗi token.
`newLlamaLConfig` cũng đọc `config.json` native, gồm GQA và RoPE scaling.

Các fixture tokenizer native:

```sh
nim c -r --path:test/deepseek-inference test/deepseek-inference/test_native_hf_tokenizer.nim
```

Thêm đường dẫn tokenizer DeepSeek Coder 6.7B cục bộ để đối chiếu ID thật.
Test config là `test/deepseek-inference/test_native_llama_config.nim`, dùng
các search path giống `make build-nim-inference`.


## BPE tokenizer cho training

Trainer Spectron `test/spectron/test_nimformer.nim` cung cấp các helper BPE byte-level:

```nim
proc trainBPE*(corpusText: string, numMerges: int): BPETokenizer
proc getOrTrainBPE*(corpusText: string, numMerges: int): BPETokenizer
proc loadBPE*(path: string): BPETokenizer
proc saveBPE*(tok: BPETokenizer, path: string)
proc encodeBPE*(tok: BPETokenizer, s: string): seq[int]
proc decodeBPEStr*(tok: BPETokenizer, ids: seq[int]): string
```

`corpusText` là văn bản training. `getOrTrainBPE` load tokenizer Spectron
đã lưu nếu có, nếu chưa thì train rồi lưu. `loadBPE`/`saveBPE` dùng file
JSON tokenizer cục bộ. Tokenizer này khác `HfTokenizer`; dùng đúng type đi
cùng pipeline model.
