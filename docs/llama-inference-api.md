
## 6. LLaMA-style Inference (`test/deepseek-inference/nim_inference.nim`)

[English](llama-inference-api.md) | [Tiếng Việt](llama-inference-api.vi.md)

This file bundles the whole inference pipeline for LLaMA-architecture models: RoPE, RMSNorm, SwiGLU FFN, GQA (Grouped-Query Attention) with a "paged" KV-cache, an HF tokenizer, and the text-generation loop (prefill + decode).

### 6.1. RMSNorm

```nim
proc newRMSNormL*(dim: int, eps: float32 = 1e-6'f32): RMSNormL
proc uploadResidentNorm*(rn: var RMSNormL, ctx: Backend)
proc rmsNormPrep*(rn: RMSNormL, x: Tensor, ctx: Backend): tuple[invBroadcast, wBroadcast: seq[float32], n: int]
proc forward*(rn: RMSNormL, x: Tensor, ctx: Backend): Tensor
```
`rmsNormPrep` separates the "reduce" part (squaring + row-sum, then `1/sqrt(mean+eps)` on CPU) from the "scale" part, so callers can fuse the scale step into the next submodule's GPU session instead of opening a separate one.


### 6.2. RoPE (Rotary Position Embedding)

```nim
proc newRoPEL*(headDim, maxSeqLen: int, theta: float32 = 10000'f32,
              scalingType: string = "", scalingFactor: float32 = 1.0'f32): RoPEL
proc applyRoPEInplace*(rope: RoPEL, ctx: Backend, data: var seq[float32],
                       B, T, H, D: int, posOffset: int)
```
Supports `rope_scaling` read from an HF `AutoConfig` (`scalingType = "linear"` or `"dynamic"` NTK-aware), needed for models that use scaling (e.g. `deepseek-coder-6.7B-GPTQ`). `applyRoPEInplace`: `data` layout is flat `[B, T, H, D]`; `posOffset` is the absolute position of the first token in `T` (used when decoding one token at a time, `T=1` but position > 0).


### 6.3. SwiGLU FeedForward

```nim
proc newSwiGLUL*(hidden, intermediate: int): SwiGLUL
proc forward*(ff: SwiGLUL, x: Tensor, ctx: Backend): Tensor
```
Fuses sigmoid + mul + mul into a single GPU session; falls back to individual `beXxx` calls if the backend can't open a session.


### 6.4. KV-Cache (paged, for GQA)

```nim
proc newKVCacheL*(nLayers, nKVHeads, headDim, maxSeqLen: int): KVCacheL
proc appendKV*(cache: var KVCacheL, layer: int, kNew, vNew: seq[float32], nTok: int)
```
Per-layer layout: `[nKVHeads, maxSeqLen, headDim]`, `B=1`. `appendKV` writes `kNew`/`vNew` (layout `[nTok, nKVHeads, headDim]`) sequentially into the cache at `[cache.seqLen .. cache.seqLen+nTok)`.


### 6.5. GQA Attention & TransformerBlockLlama

```nim
proc newGQAAttention*(hiddenSize, nHeads, nKVHeads: int): GQAAttention
proc forwardPrefill*(attn: GQAAttention, norm: RMSNormL, x: Tensor,
  rope: RoPEL, ctx: Backend, cache: var KVCacheL, layerIdx: int): Tensor
proc forwardDecode*(attn: GQAAttention, norm: RMSNormL, x: Tensor,
  rope: RoPEL, ctx: Backend, cache: var KVCacheL, layerIdx: int,
  ffnAllResident: bool = false): tuple[attnOut: Tensor, sessionOpen: bool,
  be: GpuBackend, hOut: SessionHandle, pending: seq[SessionHandle]]

proc newTransformerBlockLlama*(hidden, nHeads, nKVHeads, intermediate: int, eps: float32): TransformerBlockLlama
proc forward*(blk: TransformerBlockLlama, x: Tensor, rope: RoPEL,
  ctx: Backend, cache: var KVCacheL, layerIdx: int,
  isPrefill: bool): Tensor
```
`forwardPrefill` processes the whole prompt in a single forward pass; `forwardDecode` generates one new token at a time, reading/writing the KV-cache.


### 6.6. Full LLaMA model

```nim
proc newLlamaModelL*(cfg: LlamaLConfig): LlamaModelL
proc forwardStep*(m: LlamaModelL, ids: seq[int], ctx: Backend, cache: var KVCacheL, isPrefill: bool): Tensor
```
`isPrefill=true`: `ids` = the full prompt (T tokens). `isPrefill=false`: `ids` = exactly 1 new token (decode).


### 6.7. Loading config & weights from Hugging Face / GPTQ

```nim
proc newLlamaLConfig*(modelPath: string): LlamaLConfig
proc loadLlamaWeightsL*(m: var LlamaModelL, path: string)
proc uploadAllResidentWeights*(m: var LlamaModelL, ctx: Backend)
```
`newLlamaLConfig` reads the local `config.json` directly with Nim, including GQA and RoPE scaling; it does not require `nimpy`/`transformers`. `loadLlamaWeightsL` reads a `.nimq` checkpoint (via `loadQuantStateDict`) and builds the state dict by tensor name. `uploadAllResidentWeights` uploads **all** weights resident to the GPU once after loading — required to get good decode speed.


### 6.8. HF tokenizer & text generation

```nim
proc newHFTokenizer*(modelPath: string): HFTokenizer
proc encode*(tok: HFTokenizer, text: string): seq[int]
proc decode*(tok: HFTokenizer, ids: seq[int]): string
proc applyChatTemplate*(tok: HFTokenizer, userText: string, systemText: string = ""): seq[int]
proc generate*(m: LlamaModelL, tok: HFTokenizer, ctx: Backend, promptIds: seq[int],
              maxNewTokens: int = 128, temperature: float32 = 0.7'f32, topP: float32 = 0.9'f32): string
```
`applyChatTemplate` uses the exact chat template declared in the model's `tokenizer_config.json`, instead of an ad-hoc Alpaca-style prompt — needed for instruct-tuned models with their own special tokens/format. Returns token ids directly, **without** re-encoding. `generate` takes `promptIds` already tokenized via `applyChatTemplate`, and allocates a KV-cache sized to what's actually needed (`promptIds.len + maxNewTokens + 8`).

```nim
import nim_inference, backend

let ctx = newBackend("metal")
let config = newLlamaLConfig("path/to/hf/model")
let tokenizer = newHFTokenizer("path/to/hf/model")
var model = newLlamaModelL(config)
loadLlamaWeightsL(model, "model.nimq")
uploadAllResidentWeights(model, ctx)
let promptIds = tokenizer.applyChatTemplate("Hello")
let output = generate(model, tokenizer, ctx, promptIds, maxNewTokens = 64)
echo output
```

---
