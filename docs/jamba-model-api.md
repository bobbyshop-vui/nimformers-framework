
# Hybrid Jamba Model API (`test/codestral/jamba_hybrid.nim`)

[English](jamba-model-api.md) | [Tiếng Việt](jamba-model-api.vi.md)

`NimformerModelJamba` supports mixed Mamba/attention/Spectron and Mamba2Real block patterns. Import `jamba_hybrid` for model construction/training or `infer_lora`/`lora` for the optimized LoRA inference path.


## Model and block construction

```nim
proc newNimformerModelJamba*(vocab, embedDim, nHeads, ffMult: int,
    pattern: string, useSession: bool = true,
    mambaRealDState: int = 64, mambaRealNGroups: int = 1,
    mambaRealConvKernel: int = 4, mambaRealExpand: int = 2,
    quantResident: bool = false, initializeWeights = true): NimformerModelJamba

proc newRmsNorm*(dim: int, eps: float32 = 1e-5'f32): RmsNorm
proc forward*(n: RmsNorm, x: Tensor, ctx: Backend): Tensor
proc backward*(n: RmsNorm, x, dOut: Tensor,
               ctx: Backend): tuple[dX, dGamma: Tensor]
```

The `pattern` selects a block per character; consult `charToKind` in `test/codestral/jamba_hybrid.nim` for the exact accepted symbols. `quantResident` constructs supported quantized Mamba2Real weights in resident form.


## Forward and generation

```nim
proc forwardBlockEx*(blk: JambaBlock, x: Tensor, ctx: Backend,
  blockIndex = -1): tuple[y: Tensor, cache: HybridBlockKernelCache]
proc forwardBlock*(blk: JambaBlock, x: Tensor, ctx: Backend,
  blockIndex = -1): Tensor
proc backwardBlock*(blk: JambaBlock, x: Tensor, dOut: Tensor, ctx: Backend,
  prefix: string, grads: var Table[string, Tensor],
  blockIndex = -1): Tensor
proc forwardBatch*(m: NimformerModelJamba,
                   idsBatch: seq[seq[int]], ctx: Backend): Tensor
proc forwardBatchSession*(m: NimformerModelJamba,
                          idsBatch: seq[seq[int]], ctx: Backend): Tensor
proc isPureMambaRealPattern*(m: NimformerModelJamba): bool
proc newHybridGenState*(m: NimformerModelJamba): HybridGenState
proc newJambaGenState*(m: NimformerModelJamba): JambaGenState
proc forwardStepJamba*(m: NimformerModelJamba, tokenId: int,
                       state: var JambaGenState, ctx: Backend): Tensor
proc uploadResidentJamba*(m: var NimformerModelJamba, ctx: Backend,
  mergedQuantNames: HashSet[string] = initHashSet[string]())
```

`forwardBatchSession` is the session-oriented inference path. Incremental `newJambaGenState`/`forwardStepJamba` currently require a pure Mamba2Real pattern; mixed patterns need their sequence/batch path and the appropriate cache support.


## Training and checkpoint

```nim
proc paramList*(m: var NimformerModelJamba): seq[(string, ptr Tensor)]
proc paramQuantFlags*(m: var NimformerModelJamba): Table[string, ptr bool]
proc paramLinears*(m: var NimformerModelJamba): Table[string, ptr Linear]
proc newJambaAdamState*(m: var NimformerModelJamba): JambaAdamState
proc computeLossAndGrads*(m: var NimformerModelJamba,
  idsBatch, targetsBatch: seq[seq[int]], ctx: Backend,
  trainLmHead: bool = true, trainEmbed: bool = true):
  tuple[loss: float32, grads: Table[string, Tensor]]
proc trainStep*(m: var NimformerModelJamba, adam: var JambaAdamState,
  idsBatch, targetsBatch: seq[seq[int]], ctx: Backend,
  lr: float32 = 3e-4'f32, requantizeEvery: int = 50): float32
proc saveCheckpoint*(m: var NimformerModelJamba, path: string, step: int = 0)
proc loadCheckpoint*(path: string, useSession: bool = true):
  tuple[model: NimformerModelJamba, step: int]
```

Jamba checkpoints use the raw `.bin` container, not `.nimq`. See [Checkpoint and Binary Formats](checkpoint-formats.md).


## Hugging Face loaders

```nim
proc loadEmbeddingFromHF*(m: var NimformerModelJamba, hfBinPath: string,
                          hfEmbedKey: string = "backbone.embedding.weight")
proc loadMambaOnlyFromHFBin*(hfBinPath: string,
    vocab, embedDim, nHeads, ffMult, nLayers: int,
    nameMap: Table[string, string] = initTable[string, string](),
    useSession: bool = true):
    tuple[model: NimformerModelJamba, matched, total: int]
proc loadNormsFromHfTensors*(m: var NimformerModelJamba,
                             tensors: seq[RawBinTensor]): int
proc syncRmsFromBaseNorms*(m: var NimformerModelJamba)
```

These loaders map tensor names to this model's parameter names. Check their warnings/manifest output; unmapped tensors are not implicitly guaranteed to load.


## LoRA extension

For low-rank adaptation and optimized decode, use [LoRA API](lora-api.md). `test/codestral/lora.nim` builds on `paramList`, adapter merge/restore, and the GPU step functions from this module.
