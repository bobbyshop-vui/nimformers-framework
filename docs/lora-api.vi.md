
# API LoRA

[English](lora-api.md) | [Tiếng Việt](lora-api.vi.md)

LoRA nằm trong `test/codestral/lora.nim`, dành cho `NimformerModelJamba`.
Training và CLI inference nằm trong `test/codestral/finetune_lora.nim` và
`test/codestral/infer_lora.nim`.


## Type adapter và khởi tạo

```nim
type
  LoraAdapter* = object
    r*, inF*, outF*: int
    alpha*, scale*: float32
    A*, B*: Tensor
  LoraTrainState* = object
    lr*, wd*: float32
    adamA*, adamB*: Table[string, ApfAdamState]
  JambaLora* = object
    adapters*: Table[string, LoraAdapter]
    backup*: Table[string, seq[float32]]
    quantBase*: Table[string, QuantTensor]

proc newLoraAdapter*(inF, outF, r: int, alpha: float32 = -1'f32): LoraAdapter
proc newLoraTrainState*(lora: JambaLora, lr: float32 = 1e-4'f32,
                        wd: float32 = 0.01'f32): LoraTrainState
```

A có shape `[r,inF]`, B có shape `[outF,r]`; cập nhật merge là
`(alpha / r) * B @ A`. `alpha` âm chọn mặc định `2*r`. Sau mỗi bước train
LoRA, tham số base được khôi phục.


## Gắn, train, merge và restore

```nim
proc attachLora*(m: var NimformerModelJamba, targetNames: seq[string],
                 r: int = 4, alpha: float32 = -1'f32): JambaLora
proc attachLoraQuantResident*(m: var NimformerModelJamba,
                              quantWeights: Table[string, QuantTensor],
                              targetNames: seq[string], r: int = 4,
                              alpha: float32 = -1'f32): JambaLora
proc trainStepLora*(m: var NimformerModelJamba, lora: var JambaLora,
                    state: var LoraTrainState,
                    idsBatch, targetsBatch: seq[seq[int]], ctx: Backend,
                    requantizeEvery: int = 0): float32

proc mergeIn*(m: var NimformerModelJamba, lora: var JambaLora,
              ctx: Backend, name: string)
proc restoreBase*(m: var NimformerModelJamba, lora: JambaLora)
proc mergeLoraIntoQuantBase*(lo: var JambaLora, verbose = true): int
proc mergeQuantAdapters*(other: JambaLora, into: var JambaLora)
proc dropZeroLoraAdapters*(lo: var JambaLora): seq[string]
```

`attachLora` chọn trọng số projection theo tên. Gọi `mergeIn` trước
forward/backward của base model và `restoreBase` sau bước đó để khôi phục
base đóng băng. `trainStepLora` thực hiện vòng đời này và chỉ cập nhật A/B
adapter. Adapter quant-resident dùng `attachLoraQuantResident` để tránh
giữ thêm bản đầy đủ độ chính xác của base lượng tử hóa trong RAM host.

Các helper tham số và merge khác:

```nim
proc mergedWeight*(base: Tensor, ad: LoraAdapter, ctx: Backend): Tensor
proc gradToAB*(dWeff: Tensor, ad: LoraAdapter, ctx: Backend): tuple[dA, dB: Tensor]
proc allProjectionWeightNames*(m: var NimformerModelJamba): seq[string]
proc bindQuantBaseToLinears*(m: var NimformerModelJamba, lora: var JambaLora)
proc mergeInUploadStreamingJamba*(m: var NimformerModelJamba,
                                  lora: var JambaLora, ctx: Backend)
proc graftBlocks*(old: var NimformerModelJamba, newPattern: string,
                  ctx: Backend): NimformerModelJamba
```

`graftBlocks` chèn block mới `A` (attention) hoặc `S` (Spectron) vào
pattern, đồng thời sao chép các block `M` hiện có theo thứ tự.


## Helper inference GPU

```nim
proc prepareGpuModel*(m: var NimformerModelJamba,
                      lora: var JambaLora, ctx: Backend)
proc forwardStepJambaGpu*(m: var NimformerModelJamba, tokenId: int,
                          ctx: Backend, produceLogits: bool = true): Tensor
proc forwardStepJambaGpuRetry*(m: var NimformerModelJamba, tokenId: int,
                               replayTokens: openArray[int], replayCount: int,
                               ctx: Backend,
                               produceLogits: bool = true): Tensor
proc resetGpuGenState*(m: var NimformerModelJamba, ctx: Backend)
proc warmupGpuForParallelPrefill*(m: var NimformerModelJamba, ctx: Backend)
proc generate*(m: var NimformerModelJamba, lora: var JambaLora,
               tok: AnyTokenizer, prompt: string, maxNewTokens: int,
               ctx: Backend, temperature: float32 = 0'f32,
               seed: int64 = 1234): string
```

Helper profiling decode gồm `decodeProfileSummary()` và `decodeSplitSummary()`.
`initMicroVmJit` khởi tạo cache bytecode/JIT; `warmupMicroVmJit` làm nóng
cache. `setGpuDisableLora`, `setGpuNormPreset`, `normPresetDescription`,
`lpSelfCheck`, `graftBlocks` phục vụ chuẩn bị/debug model nâng cao; xem
chú thích source để biết giới hạn.


## CLI inference

`test/codestral/infer_lora.nim` là entry point cho checkpoint `.bin`/LoRA.
Các helper công khai gồm `loadBinaryLoraForInfer`, `defaultSamplerCfg`,
`forwardPrefillBatch`, `generate`. CLI nhận
`<checkpoint> <prompt> <temperature> [hfDir] [nimqPath]`, chưa có tùy chọn
`--help`. Xem [hướng dẫn lệnh Make](make-commands.vi.md) cho ví dụ và biến
backend. CLI và biến môi trường có thể thay đổi thường xuyên hơn API thư viện.
