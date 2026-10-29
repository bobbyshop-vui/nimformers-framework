
# Dùng `test_nimformer` để train Spectron

[English](spectron-guide.md) | [Tiếng Việt](spectron-guide.vi.md)

`test/spectron/test_nimformer.nim` vừa là module model/training có thể import,
vừa là chương trình train khi được compile làm main module. Nó dùng embedding,
các block Spectron kết hợp Linear++ attention và Mamba2SSD, final LayerNorm,
rồi projection ra vocabulary. Đây là model riêng với tokenizer riêng.


## Build, chạy và chat

Từ thư mục gốc repo:

```sh
make download-datasets
make build
./test/spectron/test_nimformer
```

Dòng đầu tải sáu JSONL code. Nếu đã có dữ liệu thì bắt đầu từ `make build`.
`make build` chỉ check/compile. Binary mới train khi được chạy.
Cách gộp trên macOS:

```sh
make run-metal
```

Hoặc dùng `make run` cho backend tự dò. Mỗi lần chạy trainer sẽ thực hiện
thêm `CFG_N_STEPS` bước, sau đó sinh thử từ `CFG_GEN_PROMPT` và thoát.

Để chat với checkpoint đã train:

```sh
make run-spectron
```

Lệnh này compile `test/spectron/test_inferencespectron.nim` thành
`./test_inferencespectron`, load checkpoint/tokenizer, rồi nhận prompt ở
`>>>`. Gõ `exit` hoặc `quit` để thoát. Chat giữ lịch sử trong phiên, sinh tối
đa 128 token/lượt với temperature 0.7 theo constants của file inference.
Hiện binary chat không đọc `ARGS`; sửa constants rồi build lại để đổi cấu hình.


## Khi trainer khởi động

1. Đọc corpus Dolly nếu file có sẵn.
2. Quét từng JSONL code và giữ tối đa 4.000 mẫu hợp lệ/file bằng reservoir
   sampling; không đọc toàn bộ các shard code vào một chuỗi khổng lồ.
3. Load BPE đã có, hoặc train BPE từ tối đa khoảng 300.000 ký tự lấy từ các
   nguồn rồi lưu tokenizer.
4. Load checkpoint hiện có, hoặc khởi tạo model mới.
5. Encode corpus, lấy các cửa sổ ngẫu nhiên, train next-token với Adam.
6. Lưu checkpoint định kỳ, lưu lần cuối và sinh thử.


## Dữ liệu đầu vào

Các đường dẫn mặc định trong `CFG_CORPUS_PATH`/`CFG_CODE_CORPUS_PATHS`:

- `test/spectron/databricks-dolly-15k.jsonl`.
- `test/spectron/nim_python_code_dataset_mega.jsonl`.
- `test/spectron/python_code_dataset_1.jsonl` đến `python_code_dataset_5.jsonl`.

`make download-datasets` tải sáu file code; Dolly phải có sẵn hoặc được đặt
thêm vào đúng đường dẫn. Nhánh Dolly hiện đọc **nội dung file thô**, gồm cả
cú pháp JSONL, không chuyển từng row thành instruction/response như trainer
Codestral. Nếu muốn một corpus văn bản riêng, có thể trỏ `CFG_CORPUS_PATH`
tới file UTF-8 của mình.

Mỗi row code là một JSON object. Ví dụ hợp lệ:

```json
{"topic":"add two numbers","language":"Python","code":"def add(a, b):\n    return a + b","note":"Return the sum."}
```

`topic` có thể được thay bằng `context`; `note` có thể được thay bằng
`explanation`. Phải có giá trị không rỗng cho topic/context, language và code.
Trainer ghép thành văn bản `Write <language> code for: <topic>`, phần note,
rồi code. Row lỗi JSON hoặc thiếu dữ liệu sẽ bị bỏ qua.

File không tồn tại được báo qua `[corpus] Khong tim thay ...`; trainer tiếp
tục dùng nguồn còn lại. Nếu không nguồn nào có nội dung, nó dùng corpus demo
`Nimformer Spectron demo corpus.` lặp lại. Kiểm tra log corpus trước khi
đánh giá chất lượng model để biết đang train trên dữ liệu nào.


## Đổi cấu hình

Sửa các `CFG_*` ở đầu `test/spectron/test_nimformer.nim`, rồi build lại:

- `CFG_EMBED_DIM=64`, `CFG_N_HEADS=4`, `CFG_N_LAYERS=2`, `CFG_FF_MULT=4`:
  kích thước model mới; embed dimension phải chia hết cho số head.
- `CFG_BATCH_SIZE=32`, `CFG_SEQ_LEN=128`: số cửa sổ/batch và số token/cửa sổ.
- `CFG_N_STEPS=6000`: số bước thêm vào trong **mỗi lần chạy**.
- `CFG_LR=3e-4`: learning rate của Adam.
- `CFG_REQUANTIZE_EVERY=50`: sau mỗi 50 bước optimizer, ép lại tham số sang
  CustomFloat do APF chọn; giữa các lần đó giữ tham số float32.
- `CFG_LOG_EVERY=100`: khoảng log.
- `CFG_CHECKPOINT_EVERY=5`, `CFG_CHECKPOINT_PATH`: tần suất và file lưu.
- `CFG_NUM_MERGES=500`, `CFG_BPE_TRAIN_CHARS=300000`: cấu hình BPE mới.
- `CFG_CODE_EXAMPLES_PER_FILE=4000`: giới hạn mẫu giữ lại cho mỗi file code.
- `CFG_GEN_PROMPT`, `CFG_GEN_MAX_NEW_TOKENS=40`, `CFG_GEN_TEMPERATURE=0`:
  lần sinh thử cuối training; temperature 0 dùng greedy.

Ví dụ thử ít bước hơn: sửa `CFG_N_STEPS` thành `100`, batch thành `4`, seq
thành `64`, rồi chạy `make run-metal`. Trainer chưa có `--steps`, `--batch`,
hay `--lr`; các biến `BYBY_SEQ_LEN` của Codestral không áp dụng ở đây.

Khi load checkpoint, kiến trúc được lấy từ checkpoint; thay constants về
kích thước model không tự đổi một model đã lưu.


## Checkpoint, tokenizer và resume

File mặc định:

```text
test/spectron/models/finetune.nimq
test/spectron/models/spectron_bpe_tokenizer.json
```

Checkpoint chứa tham số fp32 raw, kiến trúc và số step hoàn thành. Nếu file
đang ghi nhận 25.080 step, chạy lại sẽ bắt đầu ở step 25.081 và mặc định kết
thúc ở step 31.080. Adam `m`, `v` và step optimizer chưa được lưu, nên resume
là tiếp tục trọng số với optimizer mới; không tái hiện chính xác một lần
train liên tục. RNG lấy batch cũng được khởi tạo lại.

Checkpoint cũ thiếu final LayerNorm vẫn load được cho inference với hành vi
cũ. Entry point training tự bật LayerNorm khi resume và in thông báo; các
checkpoint lưu sau đó có thêm `finalNorm.gamma` và `finalNorm.beta`.

Để giữ một bản trước khi train tiếp:

```sh
cp test/spectron/models/finetune.nimq test/spectron/models/finetune.backup.nimq
cp test/spectron/models/spectron_bpe_tokenizer.json test/spectron/models/tokenizer.backup.json
```

Muốn bắt đầu model mới, đặt `CFG_CHECKPOINT_PATH` thành một file chưa có.
Muốn tạo cả tokenizer mới, dùng một thư mục thử riêng: cập nhật thêm
`BPE_TOKENIZER_PATH` trong module và `createDir` tương ứng. Sau đó chỉnh
`CHECKPOINT_PATH`/`BPE_TOKENIZER_PATH` ở binary chat nếu muốn chat với bản này.

Giữ checkpoint và tokenizer thành một cặp. Đổi số merge khi tokenizer đã có
không train lại tokenizer; `getOrTrainBPE` sẽ load file hiện tại. Tokenizer
khác có thể dùng cùng vocab size nhưng gán token ID khác, làm checkpoint
không còn khớp dữ liệu.


## Đọc loss

Log có dạng minh họa:

```text
step   100 | loss = 5.1200 | mean_loss(99) = 5.4800
```

`loss` là cross-entropy trung bình trên `batch * seq` token của batch hiện
tại, đo trước Adam update. `mean_loss(n)` là trung bình của `n` batch kể từ
lần log trước; lần log đầu có một batch, nên cửa sổ sau đó có thể chưa đủ
100 batch. Mỗi batch lấy cửa sổ khác nhau, vì vậy loss từng dòng có thể tăng
giảm. Đây là training loss, chưa có validation split/loss trong entry point.

Với logits đồng đều, vocab 756 cho loss khoảng `ln(756) = 6.628`; đây là mốc
tham khảo, không phải giới hạn trên. Logits quá tự tin vào token sai có thể
cho loss lớn hơn nhiều. Final LayerNorm giữ biên độ hidden trước lm head
ổn định, còn APF phải chọn dải exponent chứa được giá trị thực của tensor.


## Dùng từ một chương trình Nim

Lưu đoạn sau thành `spectron_demo.nim` ở thư mục gốc:

```nim
import std/random
import backend, test_nimformer

proc main() =
  randomize(42)
  let ctx = newBackend("auto")
  defer: closeBackend(ctx)
  let tok = trainBPE("hello nim hello world ", 8)
  let tokens = encodeBPE(tok, "hello nim hello world ")
  let ids = @[tokens[0 ..< tokens.len - 1]]
  let targets = @[tokens[1 .. ^1]]
  var model = newNimformerModelSpectron(tok.vocabSize, 16, 2, 1, 2)
  var adam = newModelAdamState(model)
  for step in 1 .. 10:
    echo step, ": ", trainStep(model, adam, ids, targets, ctx)
  let prompt = encodeBPE(tok, "hello ")
  let generated = model.generate(prompt, 8, ctx, temperature = 0'f32)
  echo decodeBPEStr(tok, generated)

main()
```

Compile và chạy:

```sh
nim c -r -d:release --mm:arc --opt:speed \
  --path:. --path:test/codestral --path:test/deepseek-inference \
  --path:test/spectron --path:kernelfortest spectron_demo.nim
```

Import module không chạy vòng lặp train demo vì entry point nằm dưới
`when isMainModule`. Mười bước với corpus nhỏ chỉ minh họa cách gọi API;
văn bản sinh ra chưa phản ánh chất lượng của một model đã train đủ dữ liệu.
API đầy đủ tại [Training API](training-api.vi.md).


## Kiểm tra nhanh bản sửa loss

Test APF không cần GPU:

```sh
nim c -r -d:release --path:. -o:/tmp/test_customfloat_range test/test_customfloat_range.nim
```

Test Spectron trên Metal dùng model nhỏ và checkpoint tạm, kiểm tra loss,
gradient bằng sai phân, checkpoint cũ/mới và 110 bước train qua mốc APF 50/100:

```sh
nim c -r -d:release --mm:arc --opt:speed \
  --path:. --path:test/codestral --path:test/deepseek-inference \
  --path:test/spectron --path:kernelfortest \
  -o:/tmp/test_training_stability test/spectron/test_training_stability.nim
```
