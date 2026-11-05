# Mamba-3 SISO 187M và 1.5B + LoRA + Linear++

Pipeline riêng, sao chép trainer/infer Codestral và thay mixer bằng Mamba-3 thực.
Chạy các lệnh bên dưới từ thư mục gốc repository.

Kết quả benchmark được ghi riêng theo model tại
[`benchmark.md`](benchmark.md): [187M](benchmark.md#mamba-3-siso-187m)
và [1.5B](benchmark.md#mamba-3-siso-15b). Lượt 1.5B mới nhất trên Intel Iris Plus
Graphics 640/Metal đạt prefill 9.88 tok/s và decode 4.89 tok/s; đo một lượt,
không tính load model/merge LoRA và không gộp với kết quả 187M.

## Hai model và assets hiện có

Repository hiện có hai model riêng; cấu hình nhỏ là **187M**, không phải 125M.
Không dùng chung checkpoint LoRA hoặc thư mục scratch giữa hai model.

- **187M:** config/HF assets ở `models/mamba3-siso-187m`, 12 layer, hidden 768,
  intermediate 1536; quantized model `models/mamba3_siso_187m.nimq`;
  LoRA `models/runs/mamba3-siso-187m-gated/lora.bin` và scratch
  `models/runs/mamba3-siso-187m-gated/scratch` (11 R + G).
- **1.5B:** config ở `models/mamba3-siso-1.5b`, 24 layer, hidden 2048,
  intermediate 4096; quantized model `models/mamba3_siso_1.5b.nimq`;
  LoRA `models/runs/mamba3-siso-1.5b-gated/lora-g1.bin` và scratch
  `models/runs/mamba3-siso-1.5b-gated/scratch-g1` (23 R + G).

Cả hai dùng vocab 128256 và state 128. Các binary `infer_lora_bin` và
`finetune_lora_bin` dùng chung, tải model theo đường dẫn và cấu hình truyền vào.
Trong danh sách trên, `models/...` tính từ thư mục `test/mamba3/`. Nếu asset
chưa có trong checkout thì cần chuẩn bị đúng model tương ứng trước khi chạy.

## Kiến trúc và định dạng

- Model 187M: `state-spaces/mamba3-siso-187m`, 12 block, hidden 768,
  intermediate 1536, 24 heads × 64, state 128, ngroups 1.
- Model 1.5B: 24 block, hidden 2048, intermediate 4096, vocab 128256,
  state 128; cấu hình tại `test/mamba3/models/mamba3-siso-1.5b`.
- Mỗi block có RMSNorm → Mamba-3 → residual → RMSNorm → gated MLP → residual.
- Mixer có RMSNorm B/C, bias B/C theo head, cumulative pairwise RoPE 50%,
  heavy-tail A phụ thuộc input, softplus dt, trapezoidal recurrence và SiLU(z).
  Không có conv1d hoặc mixer output RMSNorm của Mamba 2.
- Tokenizer chính thức: `meta-llama/Llama-3.1-8B`, vocab 128256;
  dùng native `tokenizer.json` của framework cho cả train/infer.
- `R`: block pretrained nguyên bản. `P`: cấu hình Linear++ song song kiểu fixed-sum,
  chỉ giữ để so sánh legacy và không dùng cho finetuning mới. `G`: cấu hình
  sidecar có learned gate như Codestral; gate khởi tạo gần nhánh Mamba và là
  gated fusion đúng cho finetuning mới.
- Giữ nguyên toàn bộ số block pretrained: 12 với 187M, 24 với 1.5B; không thay
  thế hoặc bỏ lớp pretrained.

## Chuẩn bị và build

Cần Nim, nimpy và Python có PyTorch cho bước chuyển checkpoint. Inference build
và training không import Python. Tải checkpoint và tokenizer (cần quyền truy cập tokenizer
Llama trên Hugging Face) bằng môi trường Python hiện có:

```python
from huggingface_hub import snapshot_download
snapshot_download("state-spaces/mamba3-siso-187m",
    local_dir="test/mamba3/models/mamba3-siso-187m",
    allow_patterns=["config.json", "pytorch_model.bin"])
snapshot_download("meta-llama/Llama-3.1-8B",
    local_dir="test/mamba3/models/mamba3-siso-187m",
    allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
```

```sh
make -f test/mamba3/Makefile all
test/mamba3/quantize_mamba3_bin \
  test/mamba3/models/mamba3-siso-187m/pytorch_model.bin \
  test/mamba3/models/mamba3_siso_187m.nimq
```

Converter dùng `qkFp4KM` hiện có cho in/out projection, gated MLP và lm_head;
embedding và tensor nhạy cảm nhỏ giữ FP32. lm_head được tạo từ tied embedding
nếu checkpoint không lưu riêng. Không đổi định dạng FP4-KM dùng ở Codestral.
Converter nhận `backbone.embedding.weight` làm tên chuẩn và cũng hỗ trợ alias
`backbone.embeddings.weight`.

Có thể build trainer/infer để đọc trực tiếp HF `.bin` hoặc checkpoint sharded
thay vì `.nimq` bằng cờ `-d:mamba3HfSource` (cần nimpy và Python torch):

```sh
make -f test/mamba3/Makefile all \
  FLAGS='-d:release --hints:off -d:mamba3HfSource'
```

## Finetune và infer 187M

```sh
# Cấu hình train mới: gated fusion ở block cuối, giữ đủ 12 block pretrained.
# Checkpoint và scratch riêng cho cấu hình gated để train/resume độc lập.
mkdir -p test/mamba3/models/runs/mamba3-siso-187m-gated
MAMBA3_PATTERN=RRRRRRRRRRRG MAMBA3_STEPS=100 \
BYBY_CHECKPOINT_PATH=test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
BYBY_SCRATCH_DIR=test/mamba3/models/runs/mamba3-siso-187m-gated/scratch \
  test/mamba3/finetune_lora_bin

# Đọc checkpoint gated; temperature là 0.7, token limit đặt qua biến môi trường.
BYBY_MAX_NEW_TOKENS=64 test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
  'Write a Python function to reverse a list' 0.7
```

Để chạy base thuần, dùng `BYBY_BASE_ONLY=1` và truyền HF config cùng
checkpoint quantized ở đối số thứ tư/năm; ví dụ đầy đủ nằm ở phần so sánh bên dưới.

## Finetune và infer 1.5B

Checkpoint 1.5B dùng model quantized và adapter riêng. Pattern của model này là
23 R + G (`RRRRRRRRRRRRRRRRRRRRRRRG`):

```sh
mkdir -p test/mamba3/models/runs/mamba3-siso-1.5b-gated
MAMBA3_PATTERN=RRRRRRRRRRRRRRRRRRRRRRRG MAMBA3_STEPS=100 \
BYBY_SEQ_LEN=64 \
BYBY_CHECKPOINT_PATH=test/mamba3/models/runs/mamba3-siso-1.5b-gated/lora-g1.bin \
BYBY_SCRATCH_DIR=test/mamba3/models/runs/mamba3-siso-1.5b-gated/scratch-g1 \
  test/mamba3/finetune_lora_bin mamba3 - \
  test/mamba3/models/mamba3-siso-1.5b nimq \
  test/mamba3/models/mamba3_siso_1.5b.nimq

# Temperature là 0.7; giới hạn 64 token qua BYBY_MAX_NEW_TOKENS.
BYBY_MAX_NEW_TOKENS=64 test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-1.5b-gated/lora-g1.bin 'hello' 0.7
```

So sánh base với checkpoint trên cùng prompt (cùng seed và sampling); `BYBY_BASE_ONLY=1`
bỏ qua checkpoint, còn `0` dùng LoRA như bình thường:

```sh
BYBY_BASE_ONLY=1 BYBY_MAX_NEW_TOKENS=64 test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin 'hello' 0.7 \
  test/mamba3/models/mamba3-siso-187m test/mamba3/models/mamba3_siso_187m.nimq
BYBY_MAX_NEW_TOKENS=64 test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin 'hello' 0.7
```

Ví dụ tương ứng cho 1.5B:

```sh
BYBY_BASE_ONLY=1 BYBY_MAX_NEW_TOKENS=64 test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-1.5b-gated/lora-g1.bin 'hello' 0.7 \
  test/mamba3/models/mamba3-siso-1.5b test/mamba3/models/mamba3_siso_1.5b.nimq
BYBY_MAX_NEW_TOKENS=64 test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-1.5b-gated/lora-g1.bin 'hello' 0.7
```

`BYBY_BASE_ONLY=1` dùng thêm đối số thứ tư/năm để chỉ định HF config và
checkpoint quantized khi chạy base thuần; các tham số này không cần cho LoRA
checkpoint dạng nhị phân. Đối số thứ ba của `infer_lora_bin` là temperature,
không phải số token. Số token được đặt bằng `BYBY_MAX_NEW_TOKENS`.

`MAMBA3_STEPS` là số minibatch update. Loss giữa các batch streaming khác nhau
không nhất thiết giảm đơn điệu, nên mặc định trainer không dừng chỉ vì loss của
một batch cao hơn batch trước. Có thể bật kiểm tra dừng cũ bằng
`BYBY_STOP_ON_LOSS_INCREASE=1`.

Trainer giữ streaming dataset, LoRA rank 4/alpha 8, SSD Adam/activations,
checkpoint commit và resume của Codestral. LoRA train cả in/out projection của
mixer và hai projection MLP; toàn bộ hàng projection Mamba-3 được phép cập nhật.
Linear++ train đầy đủ. Hai dataset Dolly và Nim/Python đã được copy vào `data/`.
Positional arguments của trainer được giữ: `initMode oldCheckpoint hfDir hfSource
nimqPath output scratchDir adamDir attnWeightsDir`, dùng `-` để giữ mặc định.
Mode mặc định là `mamba3`, source mặc định `nimq`.
Không đổi pattern, dataset hoặc tokenizer khi resume; dùng output và scratch riêng
khi bắt đầu một cấu hình mới.

## Runtime GPU và kiểm tra

`make -f test/mamba3/Makefile smoke` chạy một mô hình rất nhỏ: sequence/decode,
gradient các split mixer, gradient block có MLP và Linear++, tích hợp LoRA/FP4-KM.
Trên Metal, smoke với `MAMBA3_SMOKE_GPU=1` đã qua: GPU preprocessing forward/reverse,
resident RPG decode với FP4 và LoRA khác zero, batched prefix rồi continued
prefix/decode, cùng one-dispatch ở 1 threadgroup đều khớp đường tham chiếu của
smoke. Chạy bằng `MAMBA3_SMOKE_GPU=1 make -f test/mamba3/Makefile smoke`.
Đây là kiểm tra mô hình nhỏ; trainer, infer và converter cũng đã build thành công.
Smoke mở rộng có bounded GPU merge, trace và kiểm tra one-dispatch sau merge:

```sh
MAMBA3_SMOKE_GPU=1 make -f test/mamba3/Makefile smoke \
  FLAGS='-d:release --hints:off -d:mamba3SmokeMerge'
```

Inference có runtime GPU resident trong `m3_gpu_runtime.nim`, với các kernel
`m3_inference_kernels.nim`: batched prefix, recurrence scan và activation/state
được giữ trên GPU qua các lớp; decode tiếp tục từ state đó. Đây là đường nhiều
dispatch trong một command buffer/token. `m3_onedispatch.nim` cung cấp đường
one-dispatch tùy chọn. Bounded GPU merge đã tích hợp và smoke mở rộng đã merge
17/17 adapter; decode sau merge trên persistent stack khớp resident decode.

Trên Metal, prefill và token đầu tiên bao gồm thời gian setup/JIT; thống kê tốc
độ tách phần setup kernel nhưng vẫn giữ tổng wall time. Runtime truyền các
offset của address và gamma dưới dạng descriptor GPU `uint32` thô theo từng
layer, nhờ đó helper có cùng shape/T có thể dùng lại một kernel qua cả 24 layer.
Đây là cách truyền tham số mới, không đổi phép tính; đường static OD vẫn được
giữ nguyên. MTLBinaryArchive không phải cache đầy đủ cho biên dịch source: vẫn có
setup kernel theo process/shape, nhưng với các helper này setup dự kiến không còn
tăng theo số layer. Đặt
`BYBY_METAL_JIT_LOG=0` để ẩn log thông tin `[metal_jit]` về kernel và archive;
`BYBY_METAL_JIT_LOG=1` hoặc không đặt biến sẽ bật lại. Warning và error vẫn được
in.

Lượt kiểm tra ngày 2026-11-05 trên Intel Iris Plus 640/Metal, checkpoint 1.5B
`runs/mamba3-1.5b/lora-g1.bin`, prompt
`Write a Python function to check if a string is a palindrome`: prefill 18 token
mất 3.038 giây (setup/JIT 1.740 giây, phần ngoài setup 1.298 giây). Log trước
thay đổi trên cùng prompt ghi 49.373 giây (setup/JIT 47.951 giây). Lượt mới chỉ
sinh 8 token để kiểm tra; đây là một lần đo, không lấy trung bình nhiều lượt,
và timer prefill không bao gồm load model hay merge LoRA. Smoke GPU so với CPU
đã qua cho prefix, continued prefix, decode, trace và merge/one-dispatch.

One-dispatch dùng mặc định 1 threadgroup và smoke nhỏ đã qua sau khi sửa race.
`BYBY_OD_TG=8` đã timeout; giữ ở 1 và không tăng số threadgroup. Linear++ dùng
buffer carry `ZNext` riêng để tránh race giữa đọc và ghi state trong carry.
Để bật one-dispatch trong inference:

```sh
BYBY_ONE_DISPATCH=1 BYBY_OD_TG=1 test/mamba3/infer_lora_bin \
  test/mamba3/models/mamba3_lora.bin 'prompt' 64
```

Mặc định `BYBY_ONE_DISPATCH=0` dùng resident multi-dispatch; `BYBY_OD_TG` mặc
định là 1. Merge LoRA có thể đặt `BYBY_MERGE_LORA=0` để giữ A/B tách riêng,
hoặc `BYBY_MERGE_LORA=inplace` (mặc định) / `BYBY_MERGE_LORA=1` để yêu cầu
bounded GPU merge. Smoke mở rộng xác nhận merge 17/17 adapter và kết quả decode
merged với persistent stack.

Smoke mở rộng cũng xác nhận GPU block trace giữ nguyên logits và `LPcheck`
cho output tối đa sai số `2.98e-8`, state tối đa `2.38e-7` trên mô hình nhỏ.

Training đã nối preprocessing GPU trong `mamba3_real`: `prepareGpu` và
`reversePrepareGpu` dùng các kernel ở `m3_training_kernels.nim` cho preprocessing
và backward tương ứng; đường này đã qua đối chiếu forward/reverse trên mô hình
smoke nhỏ. Chưa chạy toàn bộ checkpoint 187M end-to-end và chưa đo hiệu năng để
xác nhận parity với Codestral. Các smoke nhỏ không chứng minh parity về tốc độ
hay kết quả với checkpoint upstream.

R-block safety kiểm tra riêng SSM output trong `cache.mambaY`; phép đo `y-x`
trước đó còn gộp cả MLP pretrained và gây false rescue ở R block 0. Trên
activation đã lưu, block 0 pretrained trước update có RMS SSM/input = 2.061,
trong khi RMS tổng SSM+MLP/input = 2.418. Giới hạn R giữ nguyên floor và cho
phép SSM tăng tối đa 8% so với mức trước update. Residual RMS/max cũng được
calibrate từ baseline trước optimizer với biên 8% và floor 16/32; baseline này
không nâng ngưỡng cố định lên theo residual max khoảng 525 quan sát được ở log
user1.5B. Kiểm tra NaN, residual cap và layer probe vẫn giữ nguyên. Đây không
xác nhận 100 training steps. Pattern mặc định lấy số layer của config trừ một
block R rồi thêm G: 187M dùng 11 R + G (`RRRRRRRRRRRG`), còn 1.5B dùng
23 R + G (`RRRRRRRRRRRRRRRRRRRRRRRG`).

Nguồn: [model card](https://huggingface.co/state-spaces/mamba3-siso-187m),
[config](https://huggingface.co/state-spaces/mamba3-siso-187m/blob/main/config.json),
[official Mamba-3](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py),
[gated MLP](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mlp.py).
Tham chiếu công khai đã dùng được lưu trong `upstream/`.
