
# API model Jamba lai (`test/codestral/jamba_hybrid.nim`)

[English](jamba-model-api.md) | [Tiếng Việt](jamba-model-api.vi.md)

`NimformerModelJamba` hỗ trợ pattern trộn Mamba, attention, Spectron và
Mamba2Real. Import `jamba_hybrid` để tạo/train model, hoặc `infer_lora`/`lora`
để dùng nhánh LoRA inference tối ưu.


## Tạo model và block

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

Mỗi ký tự trong `pattern` chọn một block; xem `charToKind` trong
`test/codestral/jamba_hybrid.nim` để biết symbol được chấp nhận.
`quantResident` tạo trọng số Mamba2Real lượng tử hóa ở dạng resident khi hỗ trợ.


## Forward và sinh token

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

`forwardBatchSession` là nhánh inference dùng session. Hiện
`newJambaGenState`/`forwardStepJamba` tăng dần yêu cầu pattern thuần
Mamba2Real; pattern hỗn hợp cần nhánh sequence/batch và cache tương ứng.


## Training và checkpoint

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

Checkpoint Jamba dùng container `.bin` raw, không phải `.nimq`. Xem
[định dạng checkpoint và binary](checkpoint-formats.vi.md).


## Loader Hugging Face

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

Các loader ánh xạ tên tensor sang tên tham số của model này. Kiểm tra cảnh
báo và manifest; tensor không ánh xạ được không mặc nhiên được load thành công.


## Mở rộng LoRA

Dùng [API LoRA](lora-api.vi.md) để thích nghi hạng thấp và decode tối ưu.
`test/codestral/lora.nim` xây trên `paramList`, merge/restore adapter và các
hàm bước GPU của module này.
