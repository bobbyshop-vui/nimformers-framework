
## 5. Base Model (`nimformer.nim`)

[English](base-model-api.md) | [Tiếng Việt](base-model-api.vi.md)


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
proc backward*(blk: TransformerBlock, x: Tensor, dOut: Tensor, ctx: Backend): tuple[
  dX, dAttnQkvW, dAttnQkvB, dAttnProjW, dAttnProjB,
  dFf1W, dFf1B, dFf2W, dFf2B,
  dLn1G, dLn1B, dLn2G, dLn2B: Tensor]
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
proc apfAdamStepGpu*(ctx: Backend, param: var Tensor, grad: Tensor,
                     state: var ApfAdamState, lr: float32,
                     requantizeEvery: int): CustomFloat
```
Adam combined with APF: every `requantizeEvery` steps, the parameter tensor is re-compressed into the optimal CustomFloat format (via `apfCastForTraining`, section 3).

```nim
var state = newApfAdamState(param.data.len)
let cf = apfAdamStep(param, grad, state, lr = 3e-3, requantizeEvery = 50)
```

---
