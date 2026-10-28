# Tokenizer APIs

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
proc encode*(tok: HFTokenizer, text: string): seq[int]
proc decode*(tok: HFTokenizer, ids: seq[int]): string
proc applyChatTemplate*(tok: HFTokenizer, userText: string,
                        systemText: string = ""): seq[int]
```

`applyChatTemplate` reads the model's tokenizer configuration and returns token IDs, including the model's expected role markers. Pass those IDs to `generate` from [LLaMA Inference API](llama-inference-api.md).

## Training BPE tokenizer

The Spectron training program in `test/spectron/test_nimformer.nim` exposes its byte-level BPE helpers:

```nim
proc newBpeTokenizer*(pathOrName: string): BpeTokenizer
proc encode*(tok: BpeTokenizer, s: string): seq[int]
proc decode*(tok: BpeTokenizer, ids: seq[int]): string
```

`pathOrName` may be a local `tokenizer.json` or a Transformers model path/name. This training tokenizer is distinct from `HfTokenizer`; use the type paired with the model pipeline.
