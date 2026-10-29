
# Tokenizer APIs

[English](tokenizer-api.md) | [Tiếng Việt](tokenizer-api.vi.md)


## Hugging Face tokenizer (`hf_tokenizer.nim`)

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

`hfDir` points to the local Codestral model/tokenizer directory. The loader selects
`tokenizer.model.v3`, then `tokenizer.model`. `hf_tokenizer.nim` uses the pure Nim
implementation in `test/codestral/sentencepiece_native.nim`; building and running
Codestral inference requires no Nimpy, Python tokenizer package, SentencePiece
library, C++ shim, or tokenizer library download.

The implementation reads ModelProto directly, selects BPE or Unigram from the
model, applies its embedded normalizer charsmap and whitespace flags, and handles
byte fallback, user symbols, control tokens, and optional denormalization. Codestral
V3 uses BPE with the identity normalizer and preserves indentation, tabs, and
newlines. Vocabulary tries are built once when loading the tokenizer.

Run the native fixture tests with `nim c -r --path:test/codestral
test/codestral/test_sentencepiece_native.nim`. Run the actual checkpoint ID
regressions with `nim c -r --path:test/codestral
test/codestral/test_hf_tokenizer_assets.nim [hfDir]`. Both tests use only Nim's
standard library and local files.


## Tokenizer wrapper used by LLaMA inference

`test/deepseek-inference/nim_inference.nim` provides a model-path wrapper with chat-template support:

```nim
proc newHFTokenizer*(modelPath: string): HFTokenizer
proc encode*(tok: HFTokenizer, text: string,
             addSpecialTokens: bool = true): seq[int]
proc decode*(tok: HFTokenizer, ids: seq[int],
             skipSpecialTokens: bool = true): string
proc applyChatTemplate*(tok: HFTokenizer, userText: string,
                        systemText: string = ""): seq[int]
```

`HFTokenizer` is implemented in `test/deepseek-inference/native_hf_tokenizer.nim`
and re-exported by `nim_inference.nim`. It reads local `tokenizer.json` and
`tokenizer_config.json` directly: byte-level BPE vocabulary/merges, Unicode
pre-tokenizer splits, added tokens, BOS/EOS and decode rules. A local
`tokenizer.model` can instead use the native SentencePiece reader. Regex matching
and chat-template rendering are implemented in Nim; no PCRE, tokenizer library,
Nimpy or Python runtime is needed. Unsupported tokenizer stages or template syntax
produce an explicit error rather than silently selecting a different tokenizer.

`applyChatTemplate` renders the checkpoint's template with Jinja loop scoping and
whitespace control, then encodes without adding a second set of special tokens.
Pass these IDs to `generate` from [LLaMA Inference API](llama-inference-api.md).
DeepSeek's byte-level decoder streams new bytes once and retains incomplete UTF-8
until the next token; it does not decode the entire history for every new token.
`newLlamaLConfig` also reads `config.json` natively, including GQA and RoPE scaling.

Native tokenizer fixture tests:

```sh
nim c -r --path:test/deepseek-inference test/deepseek-inference/test_native_hf_tokenizer.nim
```

Append a local DeepSeek Coder 6.7B tokenizer directory to also check exact checkpoint
IDs. The config regression is `test/deepseek-inference/test_native_llama_config.nim`;
it uses the same module search paths as `make build-nim-inference`.


## Training BPE tokenizer

The Spectron training program in `test/spectron/test_nimformer.nim` exposes its byte-level BPE helpers:

```nim
proc trainBPE*(corpusText: string, numMerges: int): BPETokenizer
proc getOrTrainBPE*(corpusText: string, numMerges: int): BPETokenizer
proc loadBPE*(path: string): BPETokenizer
proc saveBPE*(tok: BPETokenizer, path: string)
proc encodeBPE*(tok: BPETokenizer, s: string): seq[int]
proc decodeBPEStr*(tok: BPETokenizer, ids: seq[int]): string
```

`corpusText` is the training text; `getOrTrainBPE` loads the saved Spectron tokenizer if it exists, otherwise trains and saves one. `loadBPE`/`saveBPE` use a local tokenizer JSON file. This training tokenizer is distinct from `HfTokenizer`; use the type paired with the model pipeline.
