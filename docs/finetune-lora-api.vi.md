
# API fine-tune và resume LoRA (`test/codestral/finetune_lora.nim`)

[English](finetune-lora-api.md) | [Tiếng Việt](finetune-lora-api.vi.md)

Module chứa các tiện ích training cho
`NimformerModelJamba` và `JambaLora`. Entry point chính được build/chạy
thành chương trình; chỉ import type/hàm export khi viết trainer riêng.
Các phép adapter cơ bản ở [API LoRA](lora-api.vi.md).

## Fine-tune Codestral chỉ dùng RAM

`make finetune-mamba` mặc định chạy chế độ chỉ dùng RAM. Activation, Adam
moment và trọng số attention trainable dạng full-precision được giữ trong
bộ nhớ. Chế độ này không ghi activation scratch, Adam sidecar hay snapshot
resume từng layer. Checkpoint LoRA `.bin` vẫn được ghi sau mỗi training
step hoàn tất (`CFG_CHECKPOINT_EVERY = 1`). Log gọi mỗi step là epoch; mỗi
step thực tế chỉ xử lý một batch được stream, không phải một lượt đầy đủ qua
corpus.

```sh
make finetune-codestral-vulkan  # chọn Vulkan; dùng cooperative matrix nếu thiết bị hỗ trợ
make finetune-codestral-metal   # chọn Metal register-blocked SIMD GEMM

# Chọn backend trực tiếp:
make finetune-mamba FINETUNE_BACKEND=auto FINETUNE_RAM_ONLY=1

# Dùng lại chế độ cũ lưu activation/Adam/layer-resume trên đĩa:
make finetune-mamba FINETUNE_RAM_ONLY=0
```

Các biến runtime tương ứng: `BYBY_FINETUNE_RAM_ONLY=1|0`,
`BYBY_FINETUNE_BACKEND=auto|vulkan|metal` và
`BYBY_FINETUNE_ACCELERATED_GEMM=1|0`. Make target bật accelerated GEMM mặc
định. Trên Vulkan, NT GEMM dùng matrix input FP16 và tích lũy FP32 qua
cooperative matrix khi thiết bị cùng shape hỗ trợ; nếu không, BybyLang dùng
đường FP32 chính xác. `BYBY_VULKAN_TENSOR_CORES=0` tắt đường cooperative.
Trên Metal, NT GEMM dùng kernel SIMD register-blocked;
`BYBY_METAL_GEMM_RB=0` tắt kernel này. Metal không chạy đường Vulkan Tensor
Core.

Khi khởi động lại, RAM-only nạp trọng số từ checkpoint `.bin` mới nhất nhưng
khởi tạo Adam từ đầu và bỏ qua layer-resume cũ. Nếu bị ngắt giữa step, step
đó bị bỏ; checkpoint hoàn tất gần nhất vẫn dùng được. Chế độ này cần nhiều
RAM hơn đáng kể vì Adam moments và trọng số trainable full-precision luôn ở
trong bộ nhớ. Checkpoint `.bin` và sidecar tham chiếu tokenizer vẫn được ghi
ra đĩa.


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


## Store trạng thái optimizer

```nim
type AdamDiskStore* = object
  root*: string
  ramOnly*: bool

proc initAdamDiskStore*(root: string, ramOnly = false): AdamDiskStore
proc has*(store: AdamDiskStore, name: string, paramLen: int): bool
proc init*(store: AdamDiskStore, name: string, paramLen: int)
proc load*(store: AdamDiskStore, name: string, paramLen: int): ApfAdamState
proc save*(store: AdamDiskStore, name: string, st: ApfAdamState)
proc cleanup*(store: var AdamDiskStore)
proc clearAdamState*(st: var ApfAdamState)
```

`AdamDiskStore` hỗ trợ cả lưu đĩa và lưu RAM. Chế độ đĩa ghi moment Adam
(`m`, `v`, `step`) ra file để không phải giữ mọi tensor optimizer trong
RAM. Chế độ RAM giữ state trong bộ nhớ và không tạo thư mục store. Cả hai
chế độ đều kiểm tra kích thước parameter khi load.


## Bước train và helper kiểm tra tính toàn vẹn

`trainStepGraftStreaming` train block mới/graft, giữ activation và trọng số
full-precision trong RAM hoặc lưu chúng ra đĩa tùy cấu hình. Các helper
resume/toàn vẹn export khác gồm
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
