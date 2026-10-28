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

`hfDir` points to a local Hugging Face model/tokenizer directory. `encodeHf` and `decodeHf` are the direct APIs for `HfTokenizer`.

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
