# Nimformer Framework

Nimformer is a **pure Nim** transformer framework supporting Metal, Vulkan, and TSIC GPU execution through `vendor/bybylang`, adaptive quantization (int8/int4/int4-asymmetric/fp8/APF-auto), compressed `.nimq` checkpoints, and a full LLaMA/GQA inference pipeline (RoPE, RMSNorm, SwiGLU, KV-cache). CUDA is disabled; the legacy `cuda` backend spelling redirects to Vulkan.

## 1. Project Structure

```
nimformers-framework/
├── backend.nim                  # Unified compute API (matmul, activations, layernorm, attention...)
├── customfloat.nim              # CustomFloat (arbitrary bit-width) + APF auto-format
├── quant.nim                    # Tensor quantization (int8/int4/int4-asym/fp8/auto) + .nimq read/write
├── nimformer.nim                 # Base transformer model (Linear, LayerNorm, Attention, FeedForward, Embedding, Adam)
├── nim_inference.nim             # Inference for LLaMA-style models (RoPE, RMSNorm, SwiGLU, GQA, KV-cache, generate)
├── test_nimformer.nim            # Train/finetune script + BPE tokenizer + data loader + checkpoint
├── check_groupsize.nim           # Utility to check group size when exporting per-group int4
├── export_to_nimq.py             # Export a Hugging Face model (standard or GPTQ) -> .nimq checkpoint
├── verify_gptq_dequant.py        # Verify GPTQ dequant math against export_to_nimq.py
├── dump_reference_states.py      # Dump reference activations (PyTorch) to compare against Nim
├── harness_attn.nim / harness_attn2.nim   # Standalone attention test harnesses
├── databricks-dolly-15k.jsonl     # Instruction-tuning dataset (Databricks Dolly 15k)
├── nim_python_code_dataset_mega.jsonl  # Parallel Nim/Python code dataset (topic/language/code/note)
├── tokenizer.json / tokenizer-testmodel.json
├── finetune.nimq.ckpt            # Sample checkpoint (weights + optimizer state + step)
├── config.nim.json
├── LICENSE
├── Makefile
└── vendor/
    └── bybylang/
        ├── LICENSE
        ├── backends/
        │   ├── cuda/
        │   │   ├── cuda_driver.nim
        │   │   ├── cuda_runtime.nim
        │   │   └── kernels/vecop.ptx
        │   ├── metal/
        │   │   ├── kernels/vecop_matmul.metal
        │   │   ├── metal_backend.nim
        │   │   ├── metal_shim.h
        │   │   └── metal_shim.m
        │   └── opencl/
        │       ├── kernels/vecop_matmul.cl
        │       └── opencl_api.nim
        ├── bybylang.nim
        ├── demo/
        │   ├── demo_gpu.bybylang
        │   └── demo_gpu_out.nim
        ├── gpubackend.nim
        ├── makefile
        ├── readme.md
        └── tsic_ir.nim
```

---

## 2. Backend API (`backend.nim`)

`backend.nim` is the single unified layer for every GPU/CPU computation, built entirely on top of `vendor/bybylang`. Every function below takes `ctx: Backend` as its first argument.

### 2.1. Backend init / management

```nim
proc newBackend*(force: string = backend): Backend
```
Creates a new backend. `force` = `"metal"` | `"vulkan"` | `"cuda"` (legacy alias for Vulkan) | `"tsic"` | `"auto"`.
- `"auto"`: auto-detects Metal, then Vulkan. If **no** GPU is found, it **raises** an error by default instead of silently falling back to CPU (see `gForbidCpuFallback` below).
- An explicit choice (`"cpu"`, `"metal"`, ...) is always accepted, including `"cpu"` (for deliberate demo/debug use).

```nim
proc closeBackend*(b: Backend)
```
Closes the backend (currently a no-op).

```nim
proc setForbidCpuFallback*(forbid: bool)
```
Defaults to `true`: if the explicitly chosen GPU backend fails at runtime, the program raises instead of silently continuing on CPU. Call `setForbidCpuFallback(false)` **before** `newBackend()` if you deliberately want to allow falling back to CPU.

```nim
proc toByby*(bk: BackendKind): byby.GpuBackend
```
Converts a `BackendKind` into BybyLang's internal `GpuBackend`. `newBackend("cuda")` is retained as a compatibility alias and selects Vulkan; it does not invoke CUDA.

### 2.2. Matrix operations

```nim
proc beMatmul*(ctx: Backend, a: openArray[float32], M, K: int,
               b: openArray[float32], K2, N: int): seq[float32]
```
Multiplies matrix `A (M x K)` by `B (K x N)`, returning a flattened `C (M x N)`.

```nim
proc beMatmul2*(ctx: Backend,
                a1: openArray[float32], M1, K1: int, b1: openArray[float32], K1b, N1: int,
                a2: openArray[float32], M2, K2: int, b2: openArray[float32], K2b, N2: int):
                tuple[y1, y2: seq[float32]]
```
Two independent matrix multiplications inside a single command buffer where supported, reducing overhead compared to calling `beMatmul` twice.

```nim
proc beMatmulQ4*(ctx: Backend, a: openArray[float32], wq: openArray[uint8],
                 scales, zeros: openArray[float32],
                 M, K, N, groupSize, nGroupsPerRow: int): seq[float32]
```
Matrix multiply against an int4-quantized weight (asymmetric, per-group) without dequantizing the whole thing to fp32 first. Always guarantees a correctly sized `M*N` result (zero-padded if the backend returns fewer elements).

### 2.3. Resident weights on GPU (inference optimization)

```nim
proc beResidentUploadQ4*(ctx: Backend, wq: openArray[uint8], scales, zeros: openArray[float32],
                         k, n, groupSize, nGroupsPerRow: int): ResidentWeightHandle
proc beResidentLinearQ4*(ctx: Backend, x: openArray[float32], w: ResidentWeightHandle, M: int): seq[float32]
proc beResidentFreeQ4*(w: var ResidentWeightHandle)
```
Uploads an int4 weight to the GPU **once** at model-load time (`beResidentUploadQ4`); every decode step then only uploads the small activation (`beResidentLinearQ4`) without touching the weight bytes again. Returns an empty `seq` if the backend doesn't support it (caller must fall back to `beMatmulQ4`). Freed with `beResidentFreeQ4`.

```nim
proc beResidentUploadFp32*(ctx: Backend, wT: openArray[float32]): SessionHandle
proc beResidentLinearFp32*(ctx: Backend, x: openArray[float32], w: SessionHandle, M, K, N: int): seq[float32]
proc beResidentFreeFp32*(ctx: Backend, w: var SessionHandle)
```
Same idea for regular fp32/tied-embedding weights: `wT` must be dequantized + transposed **once** before calling `beResidentUploadFp32`. This is the main mechanism that keeps `Linear.forward` from having to dequant + transpose the whole matrix every token.

### 2.4. Element-wise ops & activations

```nim
proc beAdd*(ctx: Backend, a, b: openArray[float32]): seq[float32]
proc beSub*(ctx: Backend, a, b: openArray[float32]): seq[float32]   # a - b
proc beMul*(ctx: Backend, a, b: openArray[float32]): seq[float32]
proc beDiv*(ctx: Backend, a, b: openArray[float32]): seq[float32]
proc beRelu*(ctx: Backend, x: openArray[float32]): seq[float32]
proc beSigmoid*(ctx: Backend, x: openArray[float32]): seq[float32]
proc beTanh*(ctx: Backend, x: openArray[float32]): seq[float32]
proc beApflu*(ctx: Backend, x: openArray[float32],
              alpha: float32 = 0.1, beta: float32 = 0.1): seq[float32]
proc beApfluBackward*(ctx: Backend, x, dy: openArray[float32],
                       alpha: float32 = 0.1, beta: float32 = 0.1): seq[float32]
```
`beApflu`/`beApfluBackward`: a custom "APF Linear Unit" activation configurable via `alpha`/`beta`, plus its backward gradient.

### 2.5. Softmax / LayerNorm / Embedding / Attention

```nim
proc beSoftmax*(ctx: Backend, x: openArray[float32], rows, cols: int): seq[float32]
```
Row-wise softmax over a `rows x cols` matrix.

```nim
proc beLayernorm*(ctx: Backend, x: openArray[float32], gamma, beta: openArray[float32],
                  rows, cols: int, eps: float32): seq[float32]
proc beLayernormBackward*(ctx: Backend, dy, x, gamma, beta: openArray[float32],
                          rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]]
```

```nim
proc beAddLayernormFused*(ctx: Backend, a, b: openArray[float32], gamma, beta: openArray[float32],
                          rows, cols: int, eps: float32): tuple[summed, normed: seq[float32]]
```
Fuses "residual add followed by layernorm" into a **single GPU session**. Returns both `summed` (raw a+b, used as the residual for the block's final add) and `normed`. Only actually beneficial on backends that support `sessionLayernormEnc` (currently Metal); other backends fall back to `beAdd` + `beLayernorm` with identical behavior/precision.

```nim
proc beEmbeddingLookup*(ctx: Backend, table: openArray[float32], indices: openArray[int32],
                        vocab, dim: int): seq[float32]
```
Looks up rows in embedding `table (vocab x dim)` by `indices`, returning `(len(indices) x dim)`.

```nim
proc beAttentionFused*(ctx: Backend, q, k, v, mask: openArray[float32],
                       B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]]
proc beAttentionFusedBackward*(ctx: Backend, q, k, v, s_matrix, dy: openArray[float32],
                               B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]]
```
Attention (Q, K, V, mask) fused into a single kernel call. `B`=batch size, `H`=number of heads, `S`=sequence length, `D`=head dimension. `beAttentionFusedBackward` reuses the `s_matrix` saved from forward to compute `dq`, `dk`, `dv`.

### 2.6. CustomFloat / APF helpers (via `backend.nim`)

```nim
proc beCustomfloatEncode*(ctx: Backend, arr: openArray[float32], cf: CustomFloat): seq[uint8]
proc beCustomfloatDecode*(ctx: Backend, buf: openArray[uint8], cf: CustomFloat): seq[float32]
proc beApfCastForTraining*(ctx: Backend, arr: openArray[float32], gradArr: openArray[float32] = [],
                           relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                           expMargin = APF_EXP_MARGIN_BITS): tuple[data: seq[uint8], cf: CustomFloat]
```
Thin wrappers calling `customfloat.encodeArray` / `decodeArray` / `apfCastForTraining` directly (see section 3).

---

## 3. CustomFloat / APF (`customfloat.nim`)

A custom floating-point format with arbitrary exponent/mantissa bit widths, used to compress data below standard float32 precision, plus an "APF" mechanism that automatically picks the right format.

```nim
proc newCustomFloat*(exponentBits, mantissaBits: int, name = ""): CustomFloat
```
Creates a new CustomFloat format with arbitrary exponent/mantissa bit counts.

```nim
proc cpuEncodeArray*(arr: openArray[float32], cf: CustomFloat): seq[uint8]
proc cpuDecodeArray*(buf: openArray[uint8], cf: CustomFloat): seq[float32]
proc encodeArray*(arr: openArray[float32], cf: CustomFloat): seq[uint8]
proc decodeArray*(buf: openArray[uint8], cf: CustomFloat): seq[float32]
```
Encode/decode a `float32` array to/from bytes in format `cf`. `encodeArray`/`decodeArray` are the main public API.

```nim
proc computeRequiredExponentBits*(arr: openArray[float32], expMargin = APF_EXP_MARGIN_BITS): int
proc computeRequiredMantissaBits*(arr: openArray[float32], relErrorTol = APF_DEFAULT_REL_ERROR_TOL): int
proc buildCustomDtypeForTensor*(arr: openArray[float32], gradArr: openArray[float32] = [],
                               relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                               expMargin = APF_EXP_MARGIN_BITS): CustomFloat
```
Computes the minimum exponent/mantissa bits needed to represent `arr` (and `gradArr`, if given) within the allowed error, then builds the best-fitting `CustomFloat` — this is the core "APF" (Adaptive Precision Format) algorithm.

```nim
proc apfCastForTraining*(arr: openArray[float32], gradArr: openArray[float32] = [],
                         relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                         expMargin = APF_EXP_MARGIN_BITS): tuple[data: seq[uint8], cf: CustomFloat]
proc apfDecodeForTraining*(encoded: openArray[uint8], cf: CustomFloat): seq[float32]
```
`apfCastForTraining`: picks a suitable `CustomFloat` for `arr` and encodes it in one step — used inside `apfAdamStep` (section 5.8) to compress tensors during training. `apfDecodeForTraining` is the reverse.

---

## 4. Quantization (`quant.nim`)

### 4.1. Quantization kinds

```nim
type QuantKind* = enum
  qkFp32Raw          # no compression, kept as plain float32 (used for bias/layernorm — precision-sensitive)
  qkInt8             # symmetric int8, scale = max(|x|)/127
  qkInt4             # symmetric int4 (2 values per byte), scale = max(|x|)/7
  qkInt4Asymmetric   # asymmetric int4, Q4_K_M style, scale=(max-min)/15, with zero_point
  qkFp8E4M3          # CustomFloat(4 exponent, 3 mantissa)
  qkFp8E5M2          # CustomFloat(5 exponent, 2 mantissa)
  qkCustom           # arbitrary user-declared CustomFloat
  qkAuto             # APF — auto-builds a CustomFloat from the tensor itself (buildCustomDtypeForTensor)

type QuantTensor* = object
  kind*: QuantKind
  shape*: seq[int]
  scale*: seq[float32]       # len=1 (per-tensor), len=shape[0] (per-row),
                             # or len=shape[0]*ceil(shape[1]/groupSize) (per-group)
  zero_point*: seq[float32] # parallel to scale
  groupSize*: int           # 0 = per-tensor/per-row; >0 = per-group (GPTQ-style, group=128 is common)
  cf*: CustomFloat          # used for qkFp8E4M3/E5M2/qkCustom/qkAuto
  data*: seq[uint8]
```

### 4.2. Quantize / dequantize

```nim
proc quantizeTensor*(arr: openArray[float32], shape: seq[int], kind: QuantKind,
                     customCf: CustomFloat = FP24, gradArr: openArray[float32] = []): QuantTensor
proc dequantizeTensor*(qt: QuantTensor): seq[float32]
proc dequantizeTensorTransposed*(qt: QuantTensor): seq[float32]
```
`quantizeTensor` is the general dispatcher by `kind`. `dequantizeTensorTransposed` dequantizes + transposes in a **single pass**, writing directly to the `[nCols, nRows]` layout — avoiding two full-size fp32 allocations every time `Linear.forward` is called for a large weight (e.g. `lm_head`).

```nim
proc quantizeRaw*(arr: openArray[float32], shape: seq[int]): QuantTensor
proc dequantizeRaw*(qt: QuantTensor): seq[float32]
proc quantizeFloatDtype*(arr: openArray[float32], shape: seq[int], cf: CustomFloat, kind: QuantKind): QuantTensor
proc quantizeAuto*(arr: openArray[float32], shape: seq[int], gradArr: openArray[float32] = []): QuantTensor
```

```nim
import quant

let qt = quantizeTensor(data, shape, qkInt8)
let raw = dequantizeTensor(qt)

# Using APF (automatic)
let qtAuto = quantizeTensor(data, shape, qkAuto, gradArr = grad)
```

### 4.3. Reading/writing `.nimq` checkpoints

```nim
proc saveQuantStateDict*(path: string, arch: array[5, int], sd: seq[(string, QuantTensor)])
proc loadQuantStateDict*(path: string): tuple[arch: array[5, int], sd: seq[(string, QuantTensor)]]
```

```nim
saveQuantStateDict("model.nimq", [vocab, embedDim, heads, layers, ffMult], sd)
let (arch, sd) = loadQuantStateDict("model.nimq")
```

---

## 5. Base Model (`nimformer.nim`)

### 5.1. Tensor & basic ops

```nim
proc newTensor*(shape: seq[int], fill: float32 = 0'f32): Tensor
proc randnTensor*(shape: seq[int], scale: float32): Tensor
proc addT*(a, b: Tensor): Tensor
proc addT*(ctx: Backend, a, b: Tensor): Tensor
proc subT*(a, b: Tensor): Tensor
proc mulT*(a, b: Tensor): Tensor
proc transpose*(x: Tensor): Tensor
```

### 5.2. Linear

```nim
proc newLinear*(inF, outF: int): Linear
proc setQuantWeight*(l: var Linear, qt: QuantTensor)
proc uploadResident*(l: var Linear, ctx: Backend)
proc forward*(l: Linear, x: Tensor, ctx: Backend): Tensor
proc backward*(l: Linear, x: Tensor, dOut: Tensor, ctx: Backend): tuple[dX, dW, dB: Tensor]
```
`setQuantWeight` assigns a quantized weight (`QuantTensor`) to the layer. `uploadResident` uploads the weight to the GPU once so `forward` doesn't need to re-upload it every call.

### 5.3. LayerNorm

```nim
proc newLayerNorm*(dim: int, eps: float32 = 1e-5'f32): LayerNorm
proc forward*(ln: LayerNorm, x: Tensor, ctx: Backend): Tensor
proc backward*(ln: LayerNorm, x: Tensor, dOut: Tensor, ctx: Backend): tuple[dX, dGamma, dBeta: Tensor]
```

### 5.4. Causal Self-Attention

```nim
proc newCausalSelfAttention*(embedDim, nHeads: int): CausalSelfAttention
proc forward*(attn: CausalSelfAttention, x: Tensor, ctx: Backend): Tensor
proc backward*(attn: CausalSelfAttention, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
  dX, dQkvW, dQkvB, dProjW, dProjB: Tensor]
```

### 5.5. FeedForward & TransformerBlock

```nim
proc newFeedForward*(embedDim, mult: int): FeedForward
proc forward*(ff: FeedForward, x: Tensor, ctx: Backend): Tensor
proc backward*(ff: FeedForward, x: Tensor, dOut: Tensor, ctx: Backend): tuple[dX, dW1, dB1, dW2, dB2: Tensor]

proc newTransformerBlock*(embedDim, nHeads, ffMult: int): TransformerBlock
proc forward*(blk: TransformerBlock, x: Tensor, ctx: Backend): Tensor
proc backward*(blk: TransformerBlock, x: Tensor, dOut: Tensor, ctx: Backend): tuple[...]
```

### 5.6. Embedding

```nim
proc newEmbedding*(vocab, dim: int): Embedding
proc lookupBatch*(e: Embedding, idsBatch: seq[seq[int]], ctx: Backend): Tensor
proc backward*(e: Embedding, idsBatch: seq[seq[int]], dOut: Tensor, ctx: Backend): Tensor
```

### 5.7. Full model

```nim
proc newNimformerModel*(vocab, embedDim, nHeads, nLayers, ffMult: int): NimformerModel
proc forwardBatch*(m: NimformerModel, idsBatch: seq[seq[int]], ctx: Backend): Tensor
proc forward*(m: NimformerModel, ids: seq[int], ctx: Backend): Tensor
proc backwardBatch*(m: NimformerModel, idsBatch: seq[seq[int]], dLoss: Tensor, ctx: Backend): seq[Tensor]
proc backward*(m: NimformerModel, ids: seq[int], dLoss: Tensor, ctx: Backend): seq[Tensor]
```

```nim
import nimformer, backend

var model = newNimformerModel(vocab = 64, embedDim = 32, nHeads = 4, nLayers = 2, ffMult = 4)

let idsBatch = @[@[1, 2, 3], @[4, 5, 6]]        # [B=2, T=3]
let logits = model.forwardBatch(idsBatch, ctx)    # shape [2, 3, vocab]

let dLoss = randnTensor(@[2, 3, vocab], 0.01)
let grads = model.backwardBatch(idsBatch, dLoss, ctx)
# grads: outProj.W, outProj.B, blocks from last to first (12 gradients per block), then embed.weight
```

### 5.8. Optimizer (ApfAdam)

```nim
proc newApfAdamState*(paramLen: int): ApfAdamState
proc apfAdamStep*(param: var Tensor, grad: Tensor, state: var ApfAdamState,
                  lr: float32, requantizeEvery: int): CustomFloat
```
Adam combined with APF: every `requantizeEvery` steps, the parameter tensor is re-compressed into the optimal CustomFloat format (via `apfCastForTraining`, section 3).

```nim
var state = newApfAdamState(param.data.len)
let cf = apfAdamStep(param, grad, state, lr = 3e-3, requantizeEvery = 50)
```

---

## 6. LLaMA-style Inference (`nim_inference.nim`)

This file bundles the whole inference pipeline for LLaMA-architecture models: RoPE, RMSNorm, SwiGLU FFN, GQA (Grouped-Query Attention) with a "paged" KV-cache, an HF tokenizer, and the text-generation loop (prefill + decode).

### 6.1. RMSNorm

```nim
proc newRMSNormL*(dim: int, eps: float32 = 1e-6'f32): RMSNormL
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
proc forwardPrefill*(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL, ctx: Backend, ...): Tensor
proc forwardDecode*(attn: GQAAttention, norm: RMSNormL, x: Tensor, rope: RoPEL, ctx: Backend, ...): Tensor

proc newTransformerBlockLlama*(hidden, nHeads, nKVHeads, intermediate: int, eps: float32): TransformerBlockLlama
proc forward*(blk: TransformerBlockLlama, x: Tensor, rope: RoPEL, ctx: Backend, ...): Tensor
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
`newLlamaLConfig` reads `AutoConfig` via `nimpy`/`transformers`. `loadLlamaWeightsL` reads a `.nimq` checkpoint (via `loadQuantStateDict`) and builds the state dict by tensor name. `uploadAllResidentWeights` uploads **all** weights resident to the GPU once after loading — required to get good decode speed.

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
import nim_inference, metal_ai

let ctx = newMetalContext()
let config = newLlamaConfig("path/to/hf/model")
let tokenizer = newHFTokenizer("path/to/hf/model")
var model = loadLlamaModel("model.nimq", config)

let output = generate(model, tokenizer, ctx, "def fib(n):", max_new_tokens = 64)
echo output
```

---

## 7. Train/Finetune script (`test_nimformer.nim`)

This is the main CLI script for training/finetuning the `NimformerModelLinearPP` model (a "Linear Plus Attention" variant of the base model), including a BPE tokenizer (via `transformers`), a data loader that combines multiple `.jsonl` sources, and full checkpointing.

### 7.1. Command-line flags (`--help`)

```
--seq=128              Sequence length
--batch=32             Batch size
--steps=10000          Training steps
--lr=3e-3              Learning rate
--embed-dim=128        Embedding dimension
--heads=4              Number of attention heads
--layers=4             Number of transformer layers
--ff-mult=4            FFN multiplier
--quant=auto           Quantization type: int8|int4|fp8|auto|none
--backend=metal        Backend: metal|vulkan|cuda(alias)|auto
--no-session           Disable session API
--save=finetune.nimq   Output checkpoint path
--tokenizer=tokenizer.json  Path to a tokenizer.json (BPE) file, OR a HF model name/path
--data-dir=.           Directory containing the .jsonl dataset files (defaults to current directory)
--code-dataset=path.jsonl   Code dataset (topic/language/code/note); empty = auto-scan every *.jsonl in --data-dir
--chat="..."           After training, generate text continuing this prompt then exit
--chat-tokens=200      Number of new tokens to generate with --chat
--chat-temp=0.8        Sampling temperature (0 = greedy/argmax)
--requantize-every=50  Steps between each requantize (Adam APF)
--ckpt-every=5         Steps between each checkpoint save
--log-every=10         Steps between each log line
--seed=1337            Random seed
--help / -h            Show this help
```

### 7.2. BPE tokenizer

```nim
proc newBpeTokenizer*(pathOrName: string): BpeTokenizer
proc encode*(tok: BpeTokenizer, s: string): seq[int]
proc decode*(tok: BpeTokenizer, ids: seq[int]): string
```
Uses `nimpy` to call the Python `transformers` library directly instead of a hand-rolled char-level tokenizer. `pathOrName` can be a path to a `tokenizer.json` file (pure BPE, loaded via `PreTrainedTokenizerFast`) or a full model name/path on the Hub/local disk (loaded via `AutoTokenizer.from_pretrained`).

### 7.3. Data loaders — combining both `.jsonl` files

Both `databricks-dolly-15k.jsonl` **and** `nim_python_code_dataset_mega.jsonl` are loaded and **merged** into the same training set through `loadAllTexts`:

```nim
proc loadDolly*(path: string): seq[string]
```
Parses each JSON line of `databricks-dolly-15k.jsonl` (the `instruction`/`context`/`response` or `output` fields) into a training text of the form `instruction\n[context\n]response`.

```nim
proc loadCodeDataset*(path: string): seq[string]
```
Parses each JSON line of the form `{"topic", "language", "code", "note"}` (matching `nim_python_code_dataset_mega.jsonl`'s format) into a training text formatted as a markdown code block with topic + language as context, so the model learns to associate (topic, language) → the corresponding code.

```nim
proc loadAllTexts*(cfg: Config): seq[string]
```
1. Always loads `databricks-dolly-15k.jsonl` from `cfg.dataDir` via `loadDolly`.
2. Then loads the code dataset:
   - If `cfg.codeDatasetPath` is non-empty (hardcoded by default to `"nim_python_code_dataset_mega.jsonl"`): loads that file with `loadCodeDataset`. The path is now **joined with `cfg.dataDir`** unless it's an absolute path or the file already exists exactly as given — this fixes a previous bug where `--code-dataset`'s default value wasn't found when running with a `--data-dir` other than `"."`.
   - If `--code-dataset=""` (empty): auto-scans **every** other `*.jsonl` file in `cfg.dataDir` (excluding `databricks-dolly-15k.jsonl`) and loads each one with `loadCodeDataset` — letting you add as many code-dataset files as you like without touching the code.
3. If no data was loaded at all, falls back to dummy data so the script still runs for testing purposes.

The final result of `loadAllTexts` is **a single `seq[string]` containing both sources (Dolly + code dataset) mixed together**, fed directly into `buildSamples` for tokenization and chunking by `seqLen`.

```nim
proc buildSamples*(texts: seq[string], tok: BpeTokenizer, seqLen: int): seq[Sample]
```
Tokenizes each text and chunks/pads it into `(x, y)` samples of length `seqLen` for next-token-prediction training.

### 7.4. "Linear Plus Attention" model

```nim
proc newLinearPlusAttention*(embedDim, nHeads: int, useSession: bool = true): LinearPlusAttention
proc forwardLinearPlus*(attn: LinearPlusAttention, x: Tensor, ctx: Backend): Tensor
proc backwardLinearPlus*(attn: LinearPlusAttention, x: Tensor, dOut: Tensor, ctx: Backend): tuple[...]

proc newTransformerBlockLinearPP*(embedDim, nHeads, ffMult: int, useSession: bool): TransformerBlockLinearPP
proc forward*(blk: TransformerBlockLinearPP, x: Tensor, ctx: Backend): Tensor
proc backward*(blk: TransformerBlockLinearPP, x: Tensor, dOut: Tensor, ctx: Backend): tuple[...]

proc newNimformerModelLinearPP*(vocab, embedDim, nHeads, nLayers, ffMult: int, useSession: bool): NimformerModelLinearPP
proc forwardBatch*(m: NimformerModelLinearPP, idsBatch: seq[seq[int]], ctx: Backend): Tensor
proc backwardBatch*(m: NimformerModelLinearPP, idsBatch: seq[seq[int]], dLoss: Tensor, ctx: Backend): seq[Tensor]
```

```nim
proc crossEntropyLossBatch*(logits: Tensor, targetsBatch: seq[seq[int]]): tuple[loss: float32, dLogits: Tensor]
```
Computes cross-entropy loss + gradient over a batch, used in the main training loop.

```nim
proc generateText*(model: NimformerModelLinearPP, tok: BpeTokenizer, ctx: Backend, ...): string
```
Generates text after training finishes (triggered by `--chat="..."`), using `--chat-tokens`/`--chat-temp` to control length/temperature.

### 7.5. Optimizer & Checkpoint

```nim
proc newApfAdamState*(paramLen: int): ApfAdamState
proc apfAdamStep*(param: var Tensor, grad: Tensor, state: var ApfAdamState, lr: float32, requantizeEvery: int): CustomFloat

proc saveCheckpoint*(model: NimformerModelLinearPP, states: seq[ApfAdamState], stepNo: int, path: string,
                     weightKind: QuantKind, embedDim, nHeads, nLayers, ffMult: int)
proc loadCheckpointFull*(path: string, forceBackend: string = ""): tuple[
  model: NimformerModelLinearPP, states: seq[ApfAdamState], stepNo: int]
```

```nim
saveCheckpoint(model, states, stepNo, "ckpt.nimq.ckpt",
              weightKind = qkAuto,
              embedDim, nHeads, nLayers, ffMult)

let (model, states, stepNo) = loadCheckpointFull("ckpt.nimq.ckpt")
# Resume training from stepNo
```

---

## 8. Selecting a Backend

```nim
import backend

# Automatically select: Metal, then Vulkan
newBackend("auto")

# Explicit selection
newBackend("cpu")
newBackend("metal")
newBackend("vulkan")
newBackend("cuda") # legacy alias; redirects to Vulkan
newBackend("tsic")
```

Or via build flags (Makefile):

```
make run-metal   # use Metal
make run-cpu     # use CPU
make run-cuda    # compatibility target; backend selection redirects to Vulkan
```

---

## 9. Exporting from Hugging Face → `.nimq`

```
python export_to_nimq.py --model TheBloke/deepseek-coder-6.7B-instruct-GPTQ --output model.nimq --int8
```

Supports both standard and GPTQ models. Use `verify_gptq_dequant.py` to double-check GPTQ dequant math, and `dump_reference_states.py` to dump reference activations from PyTorch to compare against `nim_inference.nim`.

---

## 10. `.nimq` file format

- Header: `NIMQ1`
- Architecture: 5 integers (`vocab`, `embedDim`, `nHeads`, `nLayers`, `ffMult`)
- A list of `(name, QuantTensor)` entries
- Convention: weights use the chosen quantization kind; biases and LayerNorm (`gamma`/`beta`) always use `qkFp32Raw`

---

## 11. Session API (`sessionXxx`, low-level multi-op GPU sessions)

Every `beXxx` function in section 2 opens and closes its own GPU session internally (one upload/commit/wait per call). The **Session API**, re-exported by `backend.nim` from `vendor/bybylang/gpubackend.nim`, lets you open a **single session** and encode several operations into it back-to-back, reading results back to the CPU only once at the end — this is how `beAddLayernormFused`, `SwiGLUL.forward`, and the fused RoPE/attention/decode paths in `nim_inference.nim` avoid multiple GPU round-trips per layer.

Available symbols (imported by using `backend`, no separate import needed):

```nim
proc sessionBegin*(backend: GpuBackend): bool
```
Opens a new session on `backend`. Returns `false` if the backend doesn't support sessions (caller must fall back to individual `beXxx` calls).

```nim
proc sessionUpload*(backend: GpuBackend, data: seq[float32]): SessionHandle
proc sessionUploadIndices*(backend: GpuBackend, data: seq[int32]): SessionHandle
proc sessionAllocScratch*(backend: GpuBackend, n: int): SessionHandle
```
`sessionUpload`/`sessionUploadIndices` upload a float32/int32 array to the GPU and return a handle to it. `sessionAllocScratch` allocates an uninitialized `n`-element GPU buffer for a result you haven't computed yet.

```nim
proc sessionMatmul*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int): bool
proc sessionMatmulNT*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int): bool
proc sessionMatmulQ4*(backend: GpuBackend, a: SessionHandle, w: ResidentWeightHandle, c: var SessionHandle, M: int): bool
```
Encode a matmul into the open session, writing the result into `c` (usually a handle from `sessionAllocScratch`). `sessionMatmulNT` computes `C(M,N) = A(M,K) @ B(N,K)^T` with `B` kept in its original `[N,K]` layout (no CPU-side transpose needed) — used for attention decode against K/V-cache slices. `sessionMatmulQ4` multiplies against an int4-resident weight handle (see `beResidentUploadQ4`). All return `false` if the backend doesn't support that op inside a session — the caller must not treat `false` as a hard error, just fall back to the non-session path.

```nim
proc sessionVecOp*(backend: GpuBackend, op: string, a, b: SessionHandle, c: var SessionHandle, n: int): bool
proc sessionActivation*(backend: GpuBackend, op: string, x: SessionHandle, y: var SessionHandle, n: int): bool
```
`sessionVecOp`: element-wise op (`"add"`, `"sub"`, `"mul"`, `"div"`) between two uploaded/scratch buffers. `sessionActivation`: activation op (`"relu"`, `"sigmoid"`, `"tanh"`, ...) on a single buffer.

```nim
proc sessionSoftmax*(backend: GpuBackend, x: SessionHandle, y: var SessionHandle, rows, cols: int): bool
proc sessionLayernormEnc*(backend: GpuBackend, x, gamma, beta: SessionHandle, y: var SessionHandle,
                          rows, cols: int, eps: float32): bool
proc sessionAttentionFusedEnc*(backend: GpuBackend, q, k, v: SessionHandle, o, sMatrix: var SessionHandle, ...): bool
proc sessionRope*(backend: GpuBackend, x: SessionHandle, cosT, sinT: SessionHandle, ...): bool
```
Higher-level ops encodable inside a session: row-wise softmax, layernorm, fused attention, and RoPE rotation. Support varies by backend (currently richest on Metal) — always check the returned `bool` and fall back if `false`.

```nim
proc sessionCopyInto*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle, ...): bool
```
Copies data between two GPU-resident buffers without a CPU round-trip (e.g. writing a K/V slice into a KV-cache buffer).

```nim
proc sessionEnd*(backend: GpuBackend): bool
```
Commits and waits for every op encoded since `sessionBegin` to finish. Must be called before reading any result out.

```nim
proc sessionRead*(backend: GpuBackend, h: SessionHandle, n: int): seq[float32]
proc sessionFree*(backend: GpuBackend, h: var SessionHandle)
```
`sessionRead` reads `n` float32 values back from a GPU handle to the CPU — call this **once**, after `sessionEnd`, not between ops. `sessionFree` releases a handle's GPU-side memory; always free every handle you allocated (uploads and scratch buffers alike), even on the failure path.

### 11.1. Typical usage pattern

```nim
let be = ctx.kind.toByby()
if sessionBegin(be):
  var hA = sessionUpload(be, aData)
  var hB = sessionUpload(be, bData)
  var hC = sessionAllocScratch(be, M * N)

  let ok = sessionMatmul(be, hA, hB, hC, M, K, N) and
           sessionEnd(be)

  if ok:
    result = sessionRead(be, hC, M * N)
  sessionFree(be, hA); sessionFree(be, hB); sessionFree(be, hC)
else:
  # Backend doesn't support sessions — fall back to the plain beMatmul path
  result = beMatmul(ctx, aData, M, K, bData, K, N)
```

Rules of thumb:
- Always check `sessionBegin`'s return value; not every backend supports sessions.
- Chain every op's `bool` result with `and` and check it once, right before `sessionEnd` — don't call `sessionRead` until after a successful `sessionEnd`.
- Free every handle (`sessionFree`) on both the success and the fallback path, to avoid leaking GPU memory.
- Use `--no-session` on `test_nimformer.nim` to force the non-session (per-op `beXxx`) path everywhere, useful for debugging or on backends with flaky session support.

### 11.2. Where the Session API is already used in this codebase

- `beResidentLinearQ4` / `beResidentLinearFp32` / `beAddLayernormFused` in `backend.nim`
- `SwiGLUL.forward` in `nim_inference.nim` (sigmoid + mul + mul fused)
- `forwardPrefill` / `forwardDecode` (GQA attention) and the cross-submodule "1 session per layer" helpers (`sMatmul`, `sMatmulNT`) in `nim_inference.nim`
- `LinearPlusAttention` in `test_nimformer.nim` (controlled by the `useSession` flag / `--no-session`)

---

## 12. Important Notes

- **Biases and LayerNorm** are always kept in fp32 to avoid numerical errors.
- **Buffer pool** in Metal: use `poolGet`/`poolPut` to avoid RAM leaks.
- **Pipeline cache**: compile kernels only once.
- **`@autoreleasepool`** in `metal_bridge.m`: required to clean up autorelease objects.
- **Backend `"auto"`**: if no GPU backend (CUDA/Metal/OpenCL/TSIC) is detected, it raises an error instead of silently falling back to CPU (`gForbidCpuFallback = true`). To allow CPU execution when no GPU is available, call `setForbidCpuFallback(false)` before `newBackend()`, or explicitly use `newBackend("cpu")`.
- **Training dataset**: `test_nimformer.nim` always combines `databricks-dolly-15k.jsonl` (instruction-tuning) with a code dataset (`nim_python_code_dataset_mega.jsonl` by default, or any other `.jsonl` file in `--data-dir` if `--code-dataset` is left empty) into a single training set — see section 7.3.
