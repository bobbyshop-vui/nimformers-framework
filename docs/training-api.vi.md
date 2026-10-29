
# API training Spectron

[English](training-api.md) | [Tiếng Việt](training-api.vi.md)

Trainer là [`test/spectron/test_nimformer.nim`](../test/spectron/test_nimformer.nim).
Xem [hướng dẫn Spectron](spectron-guide.vi.md) về build/run, cấu hình, schema
JSONL, checkpoint, loss và ví dụ Nim độc lập.
[Hướng dẫn lệnh Make](make-commands.vi.md) mô tả mọi target. Chạy từ thư mục gốc:

```sh
make run
```

Trainer chưa có tùy chọn CLI. Sửa `CFG_*` đầu source để đổi kích thước
model, sequence length, batch, số step, learning rate và giới hạn mẫu.
`make run-metal` chọn Metal trên macOS. Target cũ `run-cpu` lỗi khi chạy vì
CPU model execution không còn được hỗ trợ. `make build` chỉ build trainer;
`make run-spectron` build/chạy chương trình inference tương tác riêng.


## Dataset và file model

Trainer đọc `test/spectron/databricks-dolly-15k.jsonl` và sáu dataset code:
`nim_python_code_dataset_mega.jsonl` cùng `python_code_dataset_1.jsonl` đến
`python_code_dataset_5.jsonl`. Mỗi file code được quét và giữ tối đa 4.000
row hợp lệ bằng reservoir sampling để giới hạn RAM mà vẫn lấy được row từ
khắp file. Phần train BPE lấy văn bản từ các nguồn. File Dolly được đọc
như text thô, gồm JSONL. `make download-datasets` tải sáu file code, không
tải Dolly. Nguồn thiếu bị bỏ qua; nếu mọi nguồn trống/thiếu, trainer dùng
một chuỗi demo lặp lại.

Tokenizer và checkpoint đã train nằm dưới `test/spectron/models/`:

- `test/spectron/models/spectron_bpe_tokenizer.json`.
- `test/spectron/models/finetune.nimq`.

Nếu checkpoint đã có, trainer resume từ đó. File giữ trọng số và số step
hoàn thành; moment optimizer Adam khởi tạo lại khi resume.

`CFG_N_STEPS` là số bước thêm vào mỗi lần chạy. Model mới có final
LayerNorm trước projection output. Checkpoint cũ thiếu các tensor này vẫn
giữ hành vi inference cũ khi load; main training chủ động bật final
LayerNorm trước khi train tiếp. Lần save sau có tham số mới. Log có `loss`
của batch hiện tại và trung bình từ lần log trước; demo chưa tính validation loss.


## BPE byte-level

```nim
proc getOrTrainBPE*(corpusText: string, numMerges: int): BPETokenizer
proc encodeBPE*(tok: BPETokenizer, s: string): seq[int]
proc decodeBPEStr*(tok: BPETokenizer, ids: seq[int]): string
```

Tokenizer do chương trình Spectron train và lưu tại
`test/spectron/models/spectron_bpe_tokenizer.json`; tách biệt với tokenizer
Hugging Face của DeepSeek/Codestral.


## Model, training và checkpoint

```nim
proc newNimformerModelSpectron*(vocab, embedDim, nHeads, nLayers, ffMult: int,
                                 useSession: bool = true): NimformerModelSpectron
proc forward*(m: NimformerModelSpectron, ids: seq[int], ctx: Backend): Tensor
proc forwardBatch*(m: NimformerModelSpectron, idsBatch: seq[seq[int]],
                   ctx: Backend): Tensor
proc softmaxCrossEntropy*(logits: Tensor, targets: seq[seq[int]], ctx: Backend):
    tuple[loss: float32, dLogits: Tensor]
proc newModelAdamState*(m: NimformerModelSpectron): ModelAdamState
proc trainStep*(m: var NimformerModelSpectron, adam: var ModelAdamState,
                idsBatch, targetsBatch: seq[seq[int]], ctx: Backend,
                lr: float32 = 3e-4'f32,
                requantizeEvery: int = 50): float32
proc saveCheckpoint*(m: NimformerModelSpectron, path: string, step: int = 0)
proc loadCheckpoint*(path: string): tuple[model: NimformerModelSpectron, step: int]
```

Source còn export helper forward/backward Linear++ attention và block
Spectron để tạo/train từng layer.

`forward` trả `[T,V]`; `forwardBatch` trả `[B,T,V]`. Input và target cùng
shape `[B][T]`, target dịch một token về phía trước. `trainStep` trả
cross-entropy trung bình token trước cập nhật trọng số. Giữ cùng
`ModelAdamState` qua các step trong tiến trình.


## Dữ liệu LoRA Codestral

Trainer Codestral là
[`test/codestral/finetune_lora.nim`](../test/codestral/finetune_lora.nim).
Nó đọc Dolly và `test/spectron/nim_python_code_dataset_mega.jsonl`; cấu
hình đường dẫn qua `CFG_DATASETS` trong source đó. Trọng số, checkpoint
adapter và tokenizer nằm dưới `test/codestral/models/`.
