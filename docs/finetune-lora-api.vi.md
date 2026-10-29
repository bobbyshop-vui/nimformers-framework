
# API fine-tune và resume LoRA (`test/codestral/finetune_lora.nim`)

[English](finetune-lora-api.md) | [Tiếng Việt](finetune-lora-api.vi.md)

Module chứa các tiện ích training/resume lưu trạng thái trên đĩa cho
`NimformerModelJamba` và `JambaLora`. Entry point chính được build/chạy
thành chương trình; chỉ import type/hàm export khi viết trainer riêng.
Các phép adapter cơ bản ở [API LoRA](lora-api.vi.md).


## Batch dạng stream

```nim
type
  DatasetSchema* = enum
    dsDolly, dsTopicCode
  DatasetSpec* = object
    path*: string
    schema*: DatasetSchema
  StreamingBatcher* = object

const CFG_DATASETS*: seq[DatasetSpec]
const TargetIgnore* = -1

proc newStreamingBatcher*(specs: seq[DatasetSpec], seqLen, batchSize: int,
                          tok: BPETokenizer, hfTok: HfTokenizer,
                          usingHfTok: bool): StreamingBatcher
proc nextBatch*(b: var StreamingBatcher):
  tuple[ids, targets, targetsMasked: seq[seq[int]]]
proc skipSteps*(b: var StreamingBatcher, steps: int)
```

`DatasetSchema` chọn các trường JSONL cần đọc. `TargetIgnore` đánh dấu
label không tham gia masked loss. `nextBatch` stream/tokenize mẫu vào
buffer token tái sử dụng, thay vì nạp toàn corpus vào RAM.


## Checkpoint và resume

```nim
proc saveCheckpointLoraBin*(m: var NimformerModelJamba, lora: JambaLora,
    path: string, epoch: int, newBlockParamNames: seq[string],
    attnStore: AttnWeightDiskStore)
proc loadCheckpointLoraBin*(path: string):
    tuple[model: NimformerModelJamba, lora: JambaLora, epoch: int]
proc loadCheckpointLora*(path: string):
    tuple[model: NimformerModelJamba, lora: JambaLora, epoch: int]

proc writeResumeCommit*(checkpointPath: string, step, seqLen, batchSize: int,
                        pattern, datasetSig, tokenizerDir: string)
proc readResumeCommit*(checkpointPath: string,
    stepOut, seqLenOut, batchOut: var int,
    patternOut, datasetSigOut, tokenizerDirOut: var string): bool
proc markResumeInProgress*(ckptPath: string, nextStep: int)
proc clearResumeInProgress*(ckptPath: string)
proc restoreLayerProgressForStep*(lo: var JambaLora, ckptPath: string,
                                  step, nBlocks: int,
                                  updateMaxBlock = high(int)): int
```

Checkpoint LoRA dùng `.bin` hoặc biến thể QLoRA 4-bit được hỗ trợ. Metadata
ghi pattern model, đường dẫn lượng tử hóa, step/epoch và thiết lập chuẩn
hóa; metadata chuẩn hóa không tương thích sẽ bị từ chối khi resume. Trạng
thái tạm từng layer được mô tả ở [định dạng checkpoint](checkpoint-formats.vi.md).


## Trạng thái optimizer trên đĩa

```nim
type AdamDiskStore* = object
  root*: string

proc initAdamDiskStore*(root: string): AdamDiskStore
proc has*(store: AdamDiskStore, name: string, paramLen: int): bool
proc init*(store: AdamDiskStore, name: string, paramLen: int)
proc load*(store: AdamDiskStore, name: string, paramLen: int): ApfAdamState
proc save*(store: AdamDiskStore, name: string, st: ApfAdamState)
proc cleanup*(store: var AdamDiskStore)
proc clearAdamState*(st: var ApfAdamState)
```

Moment Adam (`m`, `v`, `step`) nằm trên đĩa để train model lớn không phải
giữ mọi tensor optimizer trong RAM. Store kiểm tra độ dài payload trước
khi chấp nhận snapshot.


## Bước train và helper kiểm tra tính toàn vẹn

`trainStepGraftStreaming` train block mới/graft trong khi stream activation
và Adam state ra đĩa. Các helper resume/toàn vẹn export khác gồm
`saveLoraAdamStates`, `restoreLoraAdamStates`, `resetAdamParameters`,
`adamFileStep`, `adamStoreAllAtStep`, `adamStoreAllAtStepOrNext`,
`syncLoraAdamFromStore`, `forceLoraAdamToStep`, `forceAdamStoreToStep`,
`loraAdamStateAllAtStep`, `datasetSignature`, `attnStoreAllValid`,
`applyLayerProgressToLora`, `layerProgressStepState`,
`fillUntrainedLoraBlocks`, `auditStep0Core`, `auditStep0LoraBlocks`,
`checkLoraCheckpointIntegrity`.

Các helper phục vụ giao thức phục hồi của trainer, ở cấp thấp hơn luồng
`trainStepLora`/checkpoint thông thường. Xem implementation và tham số
của phiên bản hiện tại trong `test/codestral/finetune_lora.nim`.
