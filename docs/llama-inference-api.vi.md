
## 6. Inference kiểu LLaMA (`test/deepseek-inference/nim_inference.nim`)

[English](llama-inference-api.md) | [Tiếng Việt](llama-inference-api.vi.md)

File này chứa pipeline inference cho kiến trúc LLaMA: RoPE, RMSNorm,
SwiGLU FFN, GQA (Grouped-Query Attention) với KV cache dạng trang,
HF tokenizer và vòng sinh văn bản gồm prefill/decode.

### 6.1. RMSNorm

```nim
proc newRMSNormL*(dim: int, eps: float32 = 1e-6'f32): RMSNormL
proc uploadResidentNorm*(rn: var RMSNormL, ctx: Backend)
proc rmsNormPrep*(rn: RMSNormL, x: Tensor, ctx: Backend): tuple[invBroadcast, wBroadcast: seq[float32], n: int]
proc forward*(rn: RMSNormL, x: Tensor, ctx: Backend): Tensor
```

`rmsNormPrep` tách phần reduce (bình phương, tổng từng hàng rồi
`1/sqrt(mean+eps)` trên CPU) khỏi phần scale. Nhờ đó bên gọi có thể gộp
scale vào session GPU của submodule sau thay vì mở session riêng.


### 6.2. RoPE — Rotary Position Embedding

```nim
proc newRoPEL*(headDim, maxSeqLen: int, theta: float32 = 10000'f32,
              scalingType: string = "", scalingFactor: float32 = 1.0'f32): RoPEL
proc applyRoPEInplace*(rope: RoPEL, ctx: Backend, data: var seq[float32],
                       B, T, H, D: int, posOffset: int)
```

Hỗ trợ `rope_scaling` trong cấu hình HF với `scalingType="linear"` hoặc
`"dynamic"` theo NTK-aware, phục vụ các model dùng scaling như
`deepseek-coder-6.7B-GPTQ`. `applyRoPEInplace` nhận dữ liệu phẳng theo
`[B,T,H,D]`; `posOffset` là vị trí tuyệt đối của token đầu tiên trong T.
Khi decode từng token, `T=1` nhưng vị trí có thể lớn hơn 0.


### 6.3. FeedForward SwiGLU

```nim
proc newSwiGLUL*(hidden, intermediate: int): SwiGLUL
proc forward*(ff: SwiGLUL, x: Tensor, ctx: Backend): Tensor
```

Gộp sigmoid và hai phép nhân vào một session GPU; nếu backend không mở
được session thì dùng các lệnh `beXxx` riêng.


### 6.4. KV cache dạng trang cho GQA

```nim
proc newKVCacheL*(nLayers, nKVHeads, headDim, maxSeqLen: int): KVCacheL
proc appendKV*(cache: var KVCacheL, layer: int, kNew, vNew: seq[float32], nTok: int)
```

Layout mỗi layer là `[nKVHeads,maxSeqLen,headDim]`, `B=1`. `appendKV` ghi
`kNew`/`vNew` theo layout `[nTok,nKVHeads,headDim]` vào dải
`[cache.seqLen .. cache.seqLen+nTok)` của cache.


### 6.5. GQA attention và TransformerBlockLlama

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

`forwardPrefill` xử lý toàn prompt trong một lần forward; `forwardDecode`
sinh mỗi lần một token mới và đọc/ghi KV cache.


### 6.6. Model LLaMA đầy đủ

```nim
proc newLlamaModelL*(cfg: LlamaLConfig): LlamaModelL
proc forwardStep*(m: LlamaModelL, ids: seq[int], ctx: Backend, cache: var KVCacheL, isPrefill: bool): Tensor
```

`isPrefill=true`: `ids` là toàn prompt T token. `isPrefill=false`: `ids`
chứa đúng một token mới để decode.


### 6.7. Load config và trọng số Hugging Face/GPTQ

```nim
proc newLlamaLConfig*(modelPath: string): LlamaLConfig
proc loadLlamaWeightsL*(m: var LlamaModelL, path: string)
proc uploadAllResidentWeights*(m: var LlamaModelL, ctx: Backend)
```

`newLlamaLConfig` đọc trực tiếp `config.json` cục bộ bằng Nim, gồm GQA và
RoPE scaling, không cần `nimpy`/`transformers`. `loadLlamaWeightsL` đọc
checkpoint `.nimq` qua `loadQuantStateDict` và lập state dict theo tên
tensor. `uploadAllResidentWeights` upload **toàn bộ** trọng số resident lên
GPU một lần sau khi load để có tốc độ decode tốt.


### 6.8. HF tokenizer và sinh văn bản

```nim
proc newHFTokenizer*(modelPath: string): HFTokenizer
proc encode*(tok: HFTokenizer, text: string): seq[int]
proc decode*(tok: HFTokenizer, ids: seq[int]): string
proc applyChatTemplate*(tok: HFTokenizer, userText: string, systemText: string = ""): seq[int]
proc generate*(m: LlamaModelL, tok: HFTokenizer, ctx: Backend, promptIds: seq[int],
              maxNewTokens: int = 128, temperature: float32 = 0.7'f32, topP: float32 = 0.9'f32): string
```

`applyChatTemplate` dùng template trong `tokenizer_config.json` của model,
không tự ghép prompt kiểu Alpaca. Điều này cần cho model instruct có định
dạng/special token riêng. Hàm trả thẳng token ID, **không encode lại**.
`generate` nhận `promptIds` đã tokenize qua `applyChatTemplate` và cấp KV
cache theo nhu cầu thực (`promptIds.len + maxNewTokens + 8`).

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
