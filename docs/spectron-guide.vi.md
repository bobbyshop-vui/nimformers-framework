
# Dùng `test_nimformer` để train Spectron

[English](spectron-guide.md) | [Tiếng Việt](spectron-guide.vi.md)

`test/spectron/test_nimformer.nim` vừa là module model/training có thể import,
vừa là chương trình train khi được compile làm main module. Nó dùng embedding,
các block Spectron kết hợp Linear++ attention và Mamba2SSD, final LayerNorm,
rồi projection ra vocabulary. Đây là model riêng với tokenizer riêng.

## Train MoE với 5 tỷ token

`test/spectron/data/scale5b/` có 4,5 tỷ token tiếng Anh và 500 triệu token
cho code, distill và bài tập cú pháp. Phần tiếng Anh lấy từ
[Wikipedia tiếng Anh](https://huggingface.co/datasets/wikimedia/wikipedia) và
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Phần code
chủ yếu là mã Python nguyên văn từ các repo có giấy phép permissive trong
[CodeParrot Clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean).
Một triệu token là bài tập điền ký hiệu như `(`, `)`, `]` và `:`.

Builder giữ nguyên từng hội thoại trong `test/spectron/claude_distill_dataset.jsonl`
dưới dạng JSONL thu gọn. Mọi lượt đều đi vào dữ liệu dự đoán token tiếp theo,
gồm `tool_calls` của assistant, tên hàm và chuỗi tham số, ID gọi tool,
kết quả của lượt `tool` và cờ `is_error` nếu có. Manifest ghi số hội thoại,
số lời gọi và số kết quả để kiểm tra phần distill.

BPE giữ nguyên 756 ID. Mỗi ID dùng 10 bit trong luồng byte `uint8` little-endian.
Tập train chiếm 6,25 GB, cộng 1,25 MB cho 500.000 token validation ở mỗi nhóm.
Trainer mmap các shard và lấy mẫu theo tỷ lệ token trong manifest.

Corpus đã đóng gói được công bố tại
[bobbydeveloper2014/bybyai-spectron](https://huggingface.co/bobbydeveloper2014/bybyai-spectron/tree/main/data/scale5b)
trong thư mục `data/scale5b/`. Để cài vào `test/spectron/data/scale5b/`, chạy
từ gốc repo với Python 3.11 trở lên và `curl`:

```sh
make download
```

`make download` chạy `download-spectron-scale-data` cùng các bước tải dữ
liệu khác. Chỉ cần corpus đóng gói thì chạy `make download-spectron-scale-data`.
Trình tải ghim commit trên Hub, tải tiếp file đang dở, và kiểm số byte cùng
mã SHA-256 của từng shard theo manifest. Shard đã có và khớp mã băm sẽ
không bị tải lại.

Trong shard distill công bố, địa chỉ email, tên người dùng và tên máy, định
danh đường dẫn cục bộ và các mẫu khóa bí mật đã được che. Mọi hội thoại,
lời gọi tool và kết quả tool vẫn được giữ nguyên.

Nếu cần tạo lại corpus từ nguồn, cài thêm PyArrow. Tham số `--distill=PATH`
chọn file hội thoại khác, `--sources=DIR` chọn chỗ cache các file nguồn tải
về, còn `--redact-distill` che thông tin giống bản đã công bố.

```sh
python3 -m pip install pyarrow
make prepare-spectron-scale-data
```

Cấu hình MoE mặc định dùng 4 block Spectron chung, hidden size 512 và 8 head.
Router chọn một expert cho mỗi token. Mỗi expert có 4 lớp feed-forward độc
lập với **29.985.344 tham số**, ngoài phần trọng số chung. Loss phụ cân bằng
định tuyến và log `expert_tokens` cho biết số token được gửi đến mỗi expert.

```sh
make run-metal ARGS='--dataset=test/spectron/data/scale5b --moe-experts=2 --batch=16 --seq=128 --steps=6000'
```

Lượt train MoE lưu vào `test/spectron/models/moe/finetune.nimq` nếu không
truyền `--checkpoint=PATH`. Nhờ vậy checkpoint dense dùng để chat,
`test/spectron/models/finetune.nimq`, không bị ghi đè. Hãy chọn batch lớn nhất
mà bộ nhớ chứa được; mỗi bước lấy `batch × seq` vị trí dự đoán.

Chạy lại cùng lệnh để train thêm 6.000 bước từ checkpoint. Log có `step`,
`epoch` và số token đã xử lý. Một epoch tính theo 5 tỷ vị trí dự đoán thành công.
Checkpoint lưu router, từng expert và Adam, gồm số bước cập nhật riêng của
từng expert. Có thể dùng `--steps=0 --moe-experts=3` trên cùng checkpoint để
tăng tổng số expert lên 3 và lưu ngay.

`--steps=0 --merge-experts=PATH` ghép các expert từ một checkpoint packed MoE
khác dùng cùng BPE, hidden size, chiều rộng và số lớp mỗi expert. Trainer giữ
trọng số các expert được nhập và khởi tạo các hàng router tương ứng. Dùng
checkpoint từ cùng backbone rồi train tiếp để router học sử dụng các expert
vừa ghép.

### Tensor Core và upload checkpoint

Trên GPU NVIDIA, build với Vulkan rồi truyền `--tensor-cores=require`. Khi đó
GEMM của các lớp Linear ở cả lượt forward và backward chạy qua
`VK_KHR_cooperative_matrix`, đầu vào FP16, cộng dồn FP32. Trọng số và trạng
thái Adam vẫn giữ FP32. Trước khi train, trainer kiểm một GEMM 64×64 với kết
quả đúng. Nếu driver không có cooperative matrix, trainer dừng và báo lỗi chứ
không quay về FP32. Mỗi dòng log `[tensor-core]` đếm số GEMM đã chạy trên
Tensor Core và số GEMM nhỏ chạy FP32.

```sh
make shaders
make build-spectron-trainer SPECTRON_TRAIN_BACKEND=vulkan
./test/spectron/test_nimformer --dataset=test/spectron/data/scale5b --moe-experts=2 \
  --batch=64 --seq=128 --steps=6000 --tensor-cores=require
```

Chế độ Hugging Face upload mỗi checkpoint đã lưu bằng CLI `hf` trong khi vẫn
train. `--hf-repo=REPO --hf-path=PATH` chọn model repo và đường dẫn trong repo.
Nếu có bản lưu mới trong lúc đang upload, bản đó được upload kế tiếp và các
bản chờ cũ hơn bị bỏ. Trước khi thoát, trainer chờ lượt upload cuối xong.
Token có quyền ghi được đọc từ `--hf-token-file` (mặc định
`~/.env_spectron_token_huggingface`). Nếu chưa có file, trainer hỏi token với
ô nhập bị che rồi lưu vào đó với quyền 600. Token chỉ được truyền cho tiến
trình `hf upload`.

```sh
pip install -U huggingface_hub
hf download bobbydeveloper2014/bybyai-spectron moe/finetune.nimq --local-dir test/spectron/models
./test/spectron/test_nimformer --dataset=test/spectron/data/scale5b --moe-experts=2 \
  --batch=64 --steps=6000 --tensor-cores=require --checkpoint-every=20 \
  --hf-repo=bobbydeveloper2014/bybyai-spectron --hf-path=moe/finetune.nimq
```

`--checkpoint-hook=PATH` vẫn chạy một file thực thi sau mỗi lần lưu, với
đường dẫn checkpoint và số bước.

### Train trên Colab T4

[`notebooks/spectron_moe_colab_t4.ipynb`](../notebooks/spectron_moe_colab_t4.ipynb)
cài Nim và các file driver Vulkan của NVIDIA, tải corpus cùng bản
`moe/finetune.nimq` mới nhất, rồi build trainer Vulkan. Sau đó notebook chạy
ba bước train ở từng kích thước batch, bắt đầu từ 16. Notebook đo tok/s sau
bước đầu tiên và giữ batch nhanh nhất mà T4 và bộ nhớ của Colab chạy được.
Lượt train dùng `--tensor-cores=require` và lõi HRM (`HRM_CYCLES = 2`,
`HRM_L_STEPS = 2`, `HRM_SEGMENTS = 2` trong cell cấu hình), lưu và upload
mỗi 20 bước. Bước đo batch cũng chạy với cùng cấu hình HRM. Khi
Colab ngắt kết nối, chạy lại notebook để train tiếp từ checkpoint và trạng
thái Adam đã upload gần nhất. Cell cấu hình mount Google Drive và giữ token
có quyền ghi trong `MyDrive/.env_spectron_token_huggingface`; nếu chưa có
file, cell hỏi token một lần. Cell train chạy trainer ở chế độ Hugging Face
với file đó làm `--hf-token-file`.

### Lõi HRM

Trainer có thể chạy các block Spectron theo kiểu Hierarchical Reasoning
Model ([HRM, arXiv:2506.21734](https://arxiv.org/abs/2506.21734)).
`--hrm-l-layers` block đầu tạo thành module mức thấp L chạy nhanh, các block
còn lại tạo thành module mức cao H chạy chậm. Trong mỗi chu kỳ mức cao, L cập
nhật trạng thái `--hrm-l-steps` lần từ `zL + zH + x`, với `x` là embedding
token đã chuẩn hoá. Sau đó H cập nhật một lần từ `zH + zL`. Sau mỗi module có
một LayerNorm, và trạng thái H đi tiếp vào MoE và LM head.

HRM bật mặc định cho model mới, kể cả model bắt đầu bằng `--init-checkpoint`.
Các cờ HRM trong lệnh này chỉ nhắc lại giá trị mặc định:

```sh
./test/spectron/test_nimformer --dataset=test/spectron/data/scale5b --moe-experts=2 \
  --batch=64 --seq=128 --steps=6000 --tensor-cores=require \
  --hrm-cycles=2 --hrm-l-steps=2 --hrm-segments=2
```

- `--hrm-cycles` đặt số chu kỳ mức cao trong mỗi segment (mặc định 2).
  `--hrm-cycles=0` tạo model mới không có HRM.
- `--hrm-l-steps` đặt số bước mức thấp trong mỗi chu kỳ (mặc định 2).
- `--hrm-l-layers` đặt số block thuộc L (mặc định một nửa số block). Các
  block còn lại thuộc H.
- `--hrm-segments` đặt số segment deep supervision trong mỗi bước optimizer
  (mặc định 2). Mỗi segment bắt đầu từ trạng thái của segment trước và tính
  loss riêng. Trainer lấy trung bình gradient của các segment rồi cập nhật
  Adam một lần.

Lượt train dùng gradient một bước như trong bài báo: chỉ lần cập nhật L cuối
và lần cập nhật H cuối của mỗi segment lưu activation cho backward. Mỗi
segment chạy `cycles × (l_steps × l_layers + h_layers)` lượt block, số này in
ở dòng `[hrm]` khi khởi động. Với 4 block và cấu hình mặc định, mỗi segment
chạy 12 lượt block thay vì 4.

Lượt train HRM mới nhân 0,1 vào lớp chiếu đầu ra của nhánh attention, Mamba
và feed-forward trong mọi block. Block ngẫu nhiên cỡ đầy đủ làm vòng lặp hỗn
loạn. Khi chưa nhân, lõi 2 chu kỳ, 2 bước, 2 segment khuếch đại sai số làm
tròn của GPU thành chênh lệch logit lớn hơn chính logit, nên cùng một
checkpoint cho loss validation khác nhau ở mỗi lần chạy. Sau khi nhân, các
lượt forward lặp lại khớp nhau tới khoảng 2e-4, và lượt resume tái tạo đúng
loss validation đã lưu. Trọng số từ `--init-checkpoint` không bị nhân.

Checkpoint lưu bố cục HRM (`__hrm_layout`) cùng ba LayerNorm của HRM và trạng
thái Adam của chúng. Khi resume, trainer đọc bố cục từ checkpoint nên có thể
bỏ các cờ HRM. `--hrm-cycles`, `--hrm-l-steps` và `--hrm-segments` được đổi
khi resume vì chúng chỉ đổi lượng tính toán. `--hrm-l-layers` phải khớp với
checkpoint. Checkpoint lưu khi chưa bật HRM không thể thêm HRM lúc resume.
Nó train tiếp không có HRM và log in `[hrm] off`. Muốn thêm HRM, hãy tạo
`--checkpoint` mới kèm `--init-checkpoint`.

### Học token dừng

`make download-spectron-hrm-eos` lấy checkpoint khởi đầu ở bước 1353 vào
`test/spectron/models/hrm-eos/finetune.nimq`. Checkpoint có trọng số, Adam
và ID EOS để train tiếp. [Checkpoint trên Hugging Face](https://huggingface.co/bobbydeveloper2014/bybyai-spectron/blob/main/hrm-eos/finetune.nimq)

`make prepare-spectron-eos-data` đóng gói câu trả lời từ Dolly và file Python
hoàn chỉnh từ CodeParrot, giữ ranh giới của từng mẫu. Với
`--eos-data=test/spectron/data/scale5b/eos`, trainer dành một phần tư batch
cho các cửa sổ sạch đi qua những ranh giới đó. EOS có ID 756, nằm ngoài
756 ID BPE. Trọng số loss mặc định của mỗi nhãn EOS là 8. Điều chỉnh bằng
`--eos-rows` và `--eos-weight`.

`--resume-from=PATH` dùng trọng số, Adam và số bước từ checkpoint packed
khi file `--checkpoint` chưa tồn tại. Các lượt sau resume từ file đích.
Lần đầu bật EOS mở rộng embedding và lớp chiếu đầu ra thêm một ID.
Log `[eos-val]` đo số lần chọn EOS đúng ranh giới, số lần chọn sớm và xác
suất trung bình tại ranh giới. Chat dừng ở EOS. Giải mã theo block dừng
khi EOS đã được xác minh trong từng dòng.

### Giải mã song song theo block

`test/spectron/block_decode.nim` giải mã một lúc một block gồm K vị trí. Mỗi
block bắt đầu từ các token nháp ngẫu nhiên. Mỗi lượt chạy model một lần trên
cửa sổ ngữ cảnh cộng K token nháp, rồi dự đoán song song mọi vị trí trong
block. Sau đó:

1. Kiểm chứng. Gọi e là vị trí đầu tiên chưa được kiểm chứng. Dự đoán ở e chỉ
   đọc các token đã kiểm chứng nên chắc chắn đúng. Nếu token nháp ở e đã trùng
   dự đoán đó, dự đoán kế tiếp cũng chắc chắn đúng và e đi tiếp. Ở chỗ lệch
   đầu tiên, token nháp được thay bằng dự đoán đúng.
2. Các vị trí sau e theo `--policy`. `jacobi` thay mọi token nháp chưa kiểm
   chứng bằng dự đoán mới. `confidence` khoá một token nháp khi dự đoán của nó
   có xác suất softmax từ `--tau` trở lên và giữ nguyên qua hai lượt liên tiếp.
   Token đã khoá chỉ đổi khi model dự đoán một token khác với xác suất từ
   `--tau` trở lên.
3. Block xong khi mọi vị trí đã được kiểm chứng.

`--tau` phải lớn hơn 0,5 và không quá 1, nên nhiều nhất một token đạt được
ngưỡng này. `--policy` mặc định là `confidence`. Truyền `--policy=jacobi`
hoặc `--policy=both` để so sánh các policy.

Mỗi lượt kiểm chứng được ít nhất một vị trí, nên một block cần tối đa K lượt,
bằng số lượt forward của greedy. Đầu ra chính là đầu ra greedy: một token nháp
có confidence cao nhưng sai sẽ bị thay khi bước kiểm chứng đi tới nó, nên
confidence chỉ ảnh hưởng tới tốc độ. Greedy ở đây dùng cùng một cửa sổ cho mọi
vị trí trong block: `context - K` token cuối trước block cộng chính block đó.
Model là causal nên các token phía sau một vị trí không làm đổi logit của nó.

Test Metal ở cuối hướng dẫn này chạy đúng hàm cập nhật block (`applyPass`)
trên các bảng đại diện cho mọi model causal có thể có: một token và một cờ
confident ứng với mỗi vị trí trong block và mỗi dãy token đứng trước nó trong
block. Với n = (V^K − 1)/(V − 1) dãy như vậy, có V^n · 2^n bảng và V^K bản
nháp ban đầu. Với cỡ bộ từ vựng và cỡ block (V, K) = (2, 1), (2, 2), (3, 2)
và (2, 3), test thử mọi bảng với mọi bản nháp ban đầu và cả hai policy: 8,
256, 11.664 và 131.072 trường hợp. Với (4, 4), (8, 6) và (16, 8), test thử
3.000 bảng ngẫu nhiên mỗi cỡ, mỗi bảng một bản nháp ban đầu ngẫu nhiên. Đầu
ra luôn trùng greedy và không bao giờ cần quá K lượt. Bước kiểm chứng đã thay
mọi token nháp khoá sai mà nó gặp: 4.096 token trong 131.072 trường hợp. Khi
mỗi token bằng token trước cộng 1 theo modulo 2, bản nháp ban đầu tệ nhất cần
đúng K lượt với K = 1..12, nên một block có thể cần đủ K lượt.

```sh
nim c -d:release --mm:arc --opt:speed -d:backend=metal \
  --path:. --path:test/codestral --path:test/deepseek-inference \
  --path:test/spectron --path:kernelfortest \
  -o:/tmp/block_decode test/spectron/block_decode.nim
/tmp/block_decode --checkpoint=test/spectron/models/finetune.nimq \
  --block=16 --tokens=64 --context=128 --tau=0.9 --trace-blocks=1
```

Checkpoint dense, MoE và HRM đều nạp được, ví dụ
`test/spectron/models/moe/finetune.nimq`. Khi không có `--prompt`, chương
trình giải mã tám prompt tiếng Anh và code có sẵn trong một batch. Một batch
cần các prompt dài bằng nhau, nên mỗi prompt giữ N token cuối, với N là độ dài
của prompt ngắn nhất và không quá `context - K`.

Các dòng `[trace]` in từng lượt của prompt đầu tiên: token đã kiểm chứng để
nguyên, token nháp đã khoá nằm trong `{ngoặc nhọn}`, các token nháp khác bắt
đầu bằng `?`. Dòng tổng kết đếm số lượt forward, số lượt cho mỗi prompt, số
token nháp được chấp nhận (token nháp đã đúng sẵn khi bước kiểm chứng đi tới),
số lần khoá đúng và khoá sai, và so mọi đầu ra với greedy. Mỗi dòng
`[passes]` liệt kê số lượt mà từng block của một prompt cần. Các dòng
`[prompt]` và `[sample]` in từng prompt, đã cắt như trên, cùng phần greedy
sinh tiếp.

Khi nhiều prompt chung một batch, mỗi lượt phải chờ prompt chậm nhất, nên
`forwards` chỉ giảm dưới K mỗi block khi mọi prompt trong batch đều xong block
sớm. `row_passes` đếm riêng số lượt của từng prompt, còn `tokens_per_row_pass`
cho biết mức lợi khi giải mã riêng từng prompt.

#### Train cho giải mã theo block

Model chỉ train trên văn bản sạch dự đoán rất kém khi các token nháp phía
trước còn sai, nên đa số block cần gần K lượt. `--draft-block=K
--draft-noise=ρ` train model cho đúng tình huống của các lượt giải mã. Chế độ
này bật mặc định với K = 16 và ρ = 0,5 khi `--seq` từ 16 trở lên.
`--draft-block=0` tắt nó. Cửa sổ
được cắt thành các block K input bắt đầu từ một vị trí ngẫu nhiên. Input đầu
mỗi block đóng vai token cuối đã kiểm chứng nên giữ sạch. Mỗi input sau đó
đóng vai token nháp và bị thay bằng một token khác với xác suất r, trong đó
r ~ U(0, ρ) được rút riêng cho từng block. Target vẫn là token đúng kế tiếp.
Trung bình ρ/2 số input nháp bị thay. Với `--seq=64`, K = 16 và ρ = 0,5, mỗi
cửa sổ có đúng 4 điểm bắt đầu block, nên tỉ lệ input bị thay theo kỳ vọng là
(1/4)·(60/64) = 15/64 = 23,44%. Dòng log `[draft]` in tỉ lệ đo được.

`--probe-text=PATH` cắt một file văn bản thành các cửa sổ `--context` token và
coi K token cuối của mỗi cửa sổ là một block. Với mỗi mức nhiễu 0,5 và 1, nó
dự đoán block một lần với token nháp là token đúng và một lần với mỗi token
nháp bị thay bằng token khác theo xác suất đó. Nó in tỉ lệ dự đoán giữ nguyên
và độ chính xác top-1 của cả hai lần. Vị trí 0 của block không đọc token nháp
nào nên không được tính. Vì mọi checkpoint đều chạy trên cùng văn bản, đây là
phép so sánh công bằng giữa các model.

Đo trên Intel Iris Plus 640 (Metal): checkpoint dense
`test/spectron/models/finetune.nimq` ở bước 6030 được fine-tune 200 bước trên
corpus 5 tỷ token (`--steps=200 --batch=16 --seq=64 --lr=1e-3
--warmup=10`), một lần có `--draft-block=16 --draft-noise=0.5` và một lần
với `--draft-block=0`. Validation đọc văn bản sạch. Sau 200 bước, perplexity validation là
68,05 khi có nhiễu nháp và 60,50 khi không. Phép đo dùng 64 cửa sổ mỗi file
với `--context=64 --block=16`, tức 960 vị trí block mỗi file:

| Văn bản | Checkpoint | Giữ nguyên khi 50% nháp sai | Giữ nguyên khi mọi nháp sai | Top-1 khi nháp đúng |
|---|---|---|---|---|
| `readme.md` | bước 6030 | 30,9% | 5,9% | 9,5% |
| `readme.md` | +200 bước, không nhiễu nháp | 35,5% | 4,9% | 9,7% |
| `readme.md` | +200 bước, có nhiễu nháp | 44,3% | 15,3% | 8,9% |
| `prepare_scale_data.py` | bước 6030 | 29,7% | 3,8% | 19,9% |
| `prepare_scale_data.py` | +200 bước, không nhiễu nháp | 38,8% | 5,8% | 20,0% |
| `prepare_scale_data.py` | +200 bước, có nhiễu nháp | 53,4% | 26,6% | 19,1% |

Các dòng của mỗi checkpoint lấy từ lệnh:

```sh
/tmp/block_decode --checkpoint=PATH --context=64 --block=16 \
  --probe-text=readme.md --probe-text=test/spectron/prepare_scale_data.py
```

## Train với bộ dữ liệu đã đóng gói

Bộ dữ liệu có **2 tỷ token huấn luyện**, chia đều giữa code Python tự sinh
và văn bản tiếng Anh. Tập validation có thêm 500.000 token cho mỗi nhóm.
Phần Python gồm 32 họ bài tập, từ vòng lặp và thao tác danh sách đến tìm kiếm,
đồ thị và quy hoạch động. Generator thay đổi đầu vào, sinh lời giải và các
`assert` có thể chạy. Validation dùng ví dụ riêng thuộc cùng các họ bài.
Phần tiếng Anh gồm 750 triệu token từ [Wikipedia tiếng Anh](https://huggingface.co/datasets/wikimedia/wikipedia)
và 250 triệu từ [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
Manifest ghi phiên bản nguồn và số token.

Dùng bộ đã có ở `test/spectron/data/packed/` hoặc tái tạo với Python,
PyArrow và `curl`:

```sh
python3 -m pip install pyarrow
make prepare-spectron-data
```

Builder dùng tokenizer BPE Spectron hiện tại. Các file nhị phân lưu ID token
dạng `uint16` little-endian. Sáu file dữ liệu chiếm 4.002.000.000 byte
(khoảng 3,73 GiB). Trainer dùng mmap chỉ đọc và chép các cửa sổ được lấy mẫu
vào từng batch. Theo đường dẫn build mặc định, các file Parquet nguồn được
cache riêng trong `test/spectron/data/sources/`.

Chạy trên Metal từ trọng số Spectron sẵn có:

```sh
make run-metal ARGS='--dataset=test/spectron/data/packed --init-checkpoint=test/spectron/models/finetune.nimq --steps=6000'
```

Lượt train mới lưu vào `test/spectron/models/finetune_packed.nimq`. Khi chạy
tiếp từ file đó, bỏ `--init-checkpoint`. Checkpoint của chế độ packed lưu
số bước hoàn thành và trạng thái Adam. Số bước quyết định cách lấy batch và
lịch learning rate khi resume. Dùng `--checkpoint=PATH` để chọn file khác.
Tokenizer phải khớp với bộ dữ liệu. Trên hệ thống Vulkan dùng `make run-vulkan`
với cùng tham số.

Model tái dùng activation đã tính ở forward khi chạy backward. Chế độ
packed dùng warmup learning rate, cosine decay và giới hạn chuẩn gradient
toàn cục. Lượt train mới dùng learning rate đỉnh `0.0012`, hệ số moment
bậc hai của Adam `0.98` và giữ trọng số train ở FP32.
Truyền `--requantize-every=N` để chạy APF mỗi N
bước optimizer. Khi resume checkpoint packed cũ, trainer dùng hệ số Adam
đã dùng để train checkpoint đó.

Trainer kiểm tra logits, loss, gradient, trọng số và moment Adam. Nếu một
kiểm tra thất bại, trainer loại bước cập nhật. Khi gặp lỗi số học trong
bước train, trainer thử lại tối đa hai lần bằng batch khác và giảm learning
rate một nửa trước mỗi lần thử lại. Checkpoint lưu cả learning rate đỉnh
và hệ số giảm. Truyền `--lr=VALUE` để đổi learning rate đỉnh khi resume. Log ghi
chuẩn gradient lớn nhất trước clip và tỷ lệ batch bị clip.

Batch chia đều giữa code và tiếng Anh. Trong 4.000 bước đầu, tỷ lệ
code cơ bản, trung cấp và nâng cao chuyển từ 65/25/10 sang 35/35/30. Log ghi
token/s và loss validation riêng cho code và tiếng Anh. Với mặc định
`--batch=16 --seq=128`, 6.000 bước xử lý 12.288.000 vị trí dự đoán.

Log ghi số epoch đã hoàn thành và `epoch_tokens` của epoch kế tiếp. Một
epoch tương ứng với số vị trí dự đoán đã train thành công bằng số token
trong corpus huấn luyện. Vì lấy mẫu ngẫu nhiên, các cửa sổ có thể trùng
nhau trong một epoch. Với 2 tỷ token và batch, độ dài chuỗi mặc định, mốc
đầu tiên là `[epoch] epoch=1 step=976563`. Checkpoint lưu tổng token đã
train để vẫn tính đúng khi đổi batch hoặc độ dài chuỗi lúc resume. Với
checkpoint cũ, trainer ước tính số token trước đó theo batch và độ dài
chuỗi khi resume rồi ghi `epoch_estimated=true`.

Chạy `./test/spectron/test_nimformer --help` để xem các tùy chọn packed.

Để chat với checkpoint packed:

```sh
make run-spectron ARGS='--checkpoint=test/spectron/models/finetune_packed.nimq --tokenizer=test/spectron/data/packed/tokenizer.json'
```


## Build, chạy và chat với corpus demo

Từ thư mục gốc repo:

```sh
make download-datasets
make build
./test/spectron/test_nimformer
```

Dòng đầu tải sáu JSONL code. Nếu đã có dữ liệu thì bắt đầu từ `make build`.
Target này build shared runtime và worker, rồi check/compile trainer Spectron;
binary mới train khi được chạy. Dùng `make build-shared` để chỉ build shared
runtime và worker.
Cách gộp trên macOS:

```sh
make run-metal
```

Trên hệ thống có Vulkan loader và driver, dùng `make run-vulkan` để check,
compile với `-d:backend=vulkan`, rồi chạy cùng trainer Spectron. Target không
giới hạn macOS. Trainer dùng các thiết lập `CFG_*` bên dưới; đây không phải
lệnh inference Mamba3.

Hoặc dùng `make run` cho backend tự dò. Mỗi lần chạy trainer sẽ thực hiện
thêm `CFG_N_STEPS` bước, sau đó sinh thử từ `CFG_GEN_PROMPT` và thoát.

Để chat với checkpoint đã train:

```sh
make run-spectron
```

Lệnh này compile `test/spectron/test_inferencespectron.nim` thành
`./test_inferencespectron`, load checkpoint/tokenizer, rồi nhận prompt ở
`>>>`. Gõ `exit` hoặc `quit` để thoát. Chat giữ lịch sử trong phiên, sinh tối
đa 128 token/lượt. Mặc định nó nạp `test/spectron/models/hrm-eos/finetune.nimq`
với `test/spectron/data/scale5b/tokenizer.json`. Câu trả lời dùng
confidence-guided parallel block generation (`--block=16 --context=128
--tau=0.9`), nên mỗi câu trả lời trùng với giải mã greedy và dừng ở EOS khi
checkpoint có EOS. Sau mỗi câu trả lời, dòng `[block]` in số token và số
lượt. `--temperature=0.7` chuyển sang lấy mẫu mỗi lượt forward một token.
Truyền các tùy chọn này qua `ARGS`. Checkpoint demo cần
`ARGS='--checkpoint=test/spectron/models/finetune.nimq --tokenizer=test/spectron/models/spectron_bpe_tokenizer.json'`.


## Khi trainer khởi động

1. Đọc corpus Dolly nếu file có sẵn.
2. Quét từng JSONL code và giữ tối đa 4.000 mẫu hợp lệ/file bằng reservoir
   sampling; không đọc toàn bộ các shard code vào một chuỗi khổng lồ.
3. Load BPE đã có, hoặc train BPE từ tối đa khoảng 300.000 ký tự lấy từ các
   nguồn rồi lưu tokenizer.
4. Load checkpoint hiện có, hoặc khởi tạo model mới.
5. Encode corpus, lấy các cửa sổ ngẫu nhiên, train next-token với Adam.
6. Lưu checkpoint định kỳ, lưu lần cuối và sinh thử.

Trong lúc train, `RawTokenMemory` giữ token ID BPE của corpus và ghi lại từng
cửa sổ input được lấy mẫu: step, hàng trong batch, vị trí trong corpus và độ
dài. Cùng các input ID đó được đưa vào cả Linear++ attention lẫn Mamba2
parallel scan. `rawTokenAt` và `rawTokensForWindow` khôi phục ID BPE cùng mảnh
byte thô để xem lại hoặc replay; chuỗi token được dựng khi cần để không nhân
bản các chuỗi trong toàn corpus. Bộ nhớ này chỉ tồn tại trong tiến trình train,
không được ghi vào checkpoint model.


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

Để thử nhanh với corpus demo, sửa `CFG_N_STEPS` thành `100`, batch thành
`4`, seq thành `64`, rồi chạy `make run-metal`. Ở chế độ packed, truyền
`--steps`, `--batch` và `--lr` qua `ARGS`.

Khi load checkpoint, kiến trúc được lấy từ checkpoint; thay constants về
kích thước model không tự đổi một model đã lưu.


## Checkpoint, tokenizer và resume

File mặc định:

```text
test/spectron/models/finetune.nimq
test/spectron/models/spectron_bpe_tokenizer.json
```

Checkpoint demo chứa tham số fp32 raw, kiến trúc và số step hoàn thành. Nếu file
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
`BPE_TOKENIZER_PATH` trong module và `createDir` tương ứng. Khi chat với bản
này, dùng `--checkpoint` và `--tokenizer`.

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
giảm. Chế độ packed ghi thêm loss trên các cửa sổ validation cố định ở những
dòng bắt đầu bằng `[val]`.

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
gradient bằng sai phân, checkpoint cũ/mới và 110 bước train qua mốc APF 50/100.
Với HRM, test so gradient một bước với sai phân, chạy lặp forward của lõi 2
chu kỳ, 2 bước, 2 segment, lưu rồi nạp lại checkpoint HRM và train 110 bước.
Test cũng kiểm hàm cập nhật block trên các bảng dự đoán, và kiểm giải mã theo
block khớp với greedy cho model dense và HRM đã train, với cả hai policy:

```sh
nim c -r -d:release --mm:arc --opt:speed \
  --path:. --path:test/codestral --path:test/deepseek-inference \
  --path:test/spectron --path:kernelfortest \
  -o:/tmp/test_training_stability test/spectron/test_training_stability.nim
```
