
# Hướng dẫn các lệnh Make

[English](make-commands.md) | [Tiếng Việt](make-commands.vi.md)

Chạy các lệnh dưới đây từ thư mục gốc `nimformers-framework`, nơi có `Makefile`.
Các đường dẫn dữ liệu và checkpoint mặc định được tính từ thư mục này.


## Chuẩn bị

Cần Nim và C compiler trong `PATH`, cùng GPU Metal trên macOS hoặc Vulkan
loader/driver trên hệ điều hành có hỗ trợ. Kiểm tra công cụ:

```sh
nim --version
make --version
```

Các target tải dữ liệu dùng `curl`; `bybylang-test` và `mamba-pipeline` dùng Bash.
Python API cần Python >= 3.9 và NumPy; bộ test cần thêm pytest. Các nhánh
Codestral training/conversion có import `nimpy`; cài bằng `nimble install nimpy`
nếu compiler báo thiếu module này. Nhánh đọc PyTorch còn cần môi trường Python
và các gói tương ứng, xem [LoRA fine-tuning](finetune-lora-api.vi.md).

Backend framework hỗ trợ `auto`, `metal`, `vulkan`. `cuda` và `opencl` là tên
tương thích được chuyển sang Vulkan; chúng không bật CUDA/OpenCL runtime riêng.
CPU không còn là backend chạy model.


## Bắt đầu với Spectron

```sh
make build
./test/spectron/test_nimformer
```

`make build` chạy `nim check`, rồi `nim c` cho
`test/spectron/test_nimformer.nim`. File thực thi nằm tại
`test/spectron/test_nimformer`. Lệnh build chưa train; dòng chạy binary mới
bắt đầu đọc corpus, load/train tokenizer và train model.

Có thể gộp build và chạy:

```sh
make run
```

Trên macOS, chọn Metal rõ ràng:

```sh
make run-metal
```

Chi tiết cấu hình, dữ liệu, loss và resume nằm trong
[Hướng dẫn Spectron](spectron-guide.vi.md).


## Target Spectron và backend

- `make` hoặc `make all`: chạy `build`, rồi `run`, nên **có khởi động training**.
  `run` tự check/compile lại; dùng `make build` nếu chỉ muốn biên dịch.
- `make build`: check và compile trainer; backend mặc định `auto`.
- `make run`: check, compile và chạy trainer với backend `auto`.
- `make run-metal`: check, compile với `-d:backend=metal`, rồi train. Ngoài
  macOS target chỉ in thông báo Metal không được hỗ trợ.
- `make run-opencl`: check, compile với tên backend `opencl`, rồi chạy bằng
  Vulkan theo mapping hiện tại.
- `make run-cuda`: tương tự với tên `cuda`, được chuyển sang Vulkan.
- `make run-cpu`: target cũ vẫn còn trong Makefile, nhưng binary sẽ báo lỗi
  khi tạo backend CPU. Dùng Metal/Vulkan/auto.
- `make build-spectron`: check và compile **chương trình chat**, tạo
  `./test_inferencespectron` ở thư mục gốc.
- `make run-spectron`: build chương trình chat, rồi chạy checkpoint Spectron
  đã train. Nhập prompt tại `>>>`, dùng `exit` hoặc `quit` để thoát.

`test_nimformer` là trainer dù có chữ `test` trong tên. `test_inferencespectron`
là chương trình inference. Hai binary có vị trí output khác nhau.

Trainer và chat Spectron hiện không đọc tham số CLI về số step, batch, prompt
hay đường dẫn checkpoint. `make run ARGS='--steps 100'` không đổi cấu hình
trainer; `ARGS` ở `run-spectron` được chuyển cho binary nhưng binary chưa đọc nó.


## Tải dữ liệu và model

```sh
make download-datasets
make download-models
make download-codestral-tokenizer
```

- `download-datasets`: tải `python_code_dataset_1.jsonl` đến
  `python_code_dataset_5.jsonl` và `nim_python_code_dataset_mega.jsonl` vào
  `test/spectron/`. Target này **không tải** `databricks-dolly-15k.jsonl`.
- `download-models`: tải DeepSeek `.nimq`, config/tokenizer và Codestral base,
  adapter, config/tokenizer vào các thư mục `test/*/models/`; đồng thời ghi
  sidecar `jamba_parallel_lora.bin.hf_tokenizer_dir`.
- `download-codestral-tokenizer`: chỉ tải config/tokenizer Codestral vào
  `INFER_LORA_DEFAULT_HFDIR`, tiện khi đã có trọng số.

Các target tải dùng retry/resume của `curl`. `download-models` tải các model
lớn; Spectron tự train không cần các model DeepSeek/Codestral này.

## DeepSeek inference

```sh
make build-nim-inference
./nim_inference test/deepseek-inference/models \
  test/deepseek-inference/models/model.nimq auto \
  'Write a Python Fibonacci function.' 128 0.7 0.9 \
  'You are an AI programming assistant.'
```

`build-nim-inference` tạo `./nim_inference`, chưa chạy model. Để build và chạy:

```sh
make infer-deepseek ARGS='test/deepseek-inference/models test/deepseek-inference/models/model.nimq auto "Write a Python Fibonacci function." 128 0.7 0.9 "You are an AI programming assistant."'
```

Thứ tự đối số: `modelDir`, `nimqPath`, `backend`, `prompt`, `maxNewTokens`,
`temperature`, `topP`, `systemText`. Không truyền `ARGS` thì dùng mặc định.
Đặt prompt trong dấu nháy kép bên trong chuỗi `ARGS` để giữ thành một đối số.
Xem [LLaMA inference API](llama-inference-api.vi.md).

## Codestral/Mamba LoRA

```sh
make build-infer-lora
make infer-lora BYBY_INFER_BACKEND=metal
```

`build-infer-lora` tạo `./infer_lora_bin`, dùng define `codestralInference`
để loại các loader PyTorch/Nimpy khỏi nhánh inference. `infer-lora` build rồi
chạy. Có thể dùng `BYBY_INFER_BACKEND=vulkan` hoặc `auto`.

```sh
make infer-lora ARGS='test/codestral/models/jamba_parallel_lora.bin "[INST] Write a Python function to add two numbers.[/INST]" 0.7 test/codestral/models/hf test/codestral/models/codestral_mamba_only.nimq'
```

Thứ tự đối số: `checkpoint`, `prompt`, `temperature`, `hfDir`, `nimqPath`.
Checkpoint adapter còn cần base weights và tokenizer; metadata/sidecar của
checkpoint được loader sử dụng. `hfDir` và `nimqPath` CLI phục vụ nhánh
fallback base Mamba. Khi chỉ định checkpoint tường minh nhưng file không hợp
lệ, CLI báo lỗi; không nên dựa vào fallback để bỏ qua một đường dẫn sai.
Prompt mẫu kết thúc ở `[/INST]`; CLI mặc định cắt khoảng trắng cuối prompt.

`make infer-mamba` là alias của `infer-lora`, không phải chương trình khác.

### Fine-tune

```sh
make build-finetune
make finetune-mamba ARGS='codestral - test/codestral/models/hf nimq test/codestral/models/codestral_mamba_only.nimq test/codestral/models/my_lora.bin'
```

`build-finetune` tạo `./finetune_lora_bin`. `finetune-mamba` build rồi train.
Thứ tự đối số:

1. `initMode`: thường là `codestral`, hoặc `graft` cho pipeline checkpoint cũ.
2. `oldCheckpoint`: dùng khi `initMode=graft`; `-` giữ mặc định.
3. `hfDir`: thư mục config/tokenizer.
4. `hfSource`: `nimq` hoặc `safetensors`.
5. `nimqPath`: base đã quantize nếu `hfSource=nimq`.
6. `outCheckpointPath`: checkpoint LoRA đầu ra.
7. `scratchDir`: tùy chọn, thư mục lưu activation tạm.

Riêng CLI này nhận `-` để giữ mặc định cho một vị trí. Một số biến môi trường
ghi đè CLI, như `BYBY_CHECKPOINT_PATH`, `BYBY_SCRATCH_DIR`; `BYBY_SEQ_LEN`
đổi sequence length của trainer Codestral. Các biến này không đổi `CFG_*`
của trainer Spectron. Xem [fine-tuning và resume](finetune-lora-api.vi.md).


### Pipeline tự động

```sh
SKIP_TO_FINETUNE=1 make mamba-pipeline
```

Target gọi `test/codestral/finetunemamba.sh`: quantize nếu cần, fine-tune,
rồi inference. Script tự bỏ qua quantize khi base `.nimq` đã có, hoặc khi
`SKIP_TO_FINETUNE=1`; trường hợp sau vẫn yêu cầu file base tồn tại. Các biến
chính: `HF_DIR`, `OLD_MAMBA_NIMQ`, `LORA_CHECKPOINT`, `INFER_PROMPT`,
`INFER_TEMPERATURE`. Script tự đặt lệnh compile, không dùng toàn bộ `NIMFLAGS`
của Makefile.


### Graft checkpoint Jamba cũ

```sh
make build-graft
make graft ARGS='path/to/mamba.bin MMAM path/to/grafted.bin'
```

`build-graft` tạo `./graft_attention_bin`. Ví dụ `MMAM` giả định checkpoint
đầu vào có ba block `M`, rồi chèn thêm một block attention `A`. Pattern phải
phù hợp kiến trúc checkpoint; xem [Jamba API](jamba-model-api.vi.md).

Hiện mặc định `GRAFT_DEFAULT_OLD`/`GRAFT_DEFAULT_OUT` trong Makefile là `.nimq`,
nhưng CLI `graft_attention.nim` từ chối `.nimq` ở nhánh này. Vì vậy
`make graft` không tham số chưa phải lệnh hoạt động cho base Codestral mặc
định. Truyền checkpoint Jamba `.bin` hợp lệ qua `ARGS`, hoặc dùng
`make finetune-mamba` để train trực tiếp từ base Codestral `.nimq`.


### Công cụ chuyển checkpoint

```sh
make build-compact-lora
./compact_lora_checkpoint path/to/legacy_checkpoint.bin path/to/compact.nimq

make build-migrate-lora
./migrate_lora_json_to_bin path/to/legacy_checkpoint.bin path/to/migrated.bin
```

Hai target chỉ build các binary ở thư mục gốc. Công cụ compact đọc
`<input>.lora.json`, ghi trọng số compact và metadata; không phải công cụ
nén mọi file `.bin`. Công cụ migrate chuyển checkpoint JSON cũ sang binary;
chọn output riêng như ví dụ để giữ bản gốc. Xem [định dạng checkpoint](checkpoint-formats.vi.md).


## Python, C và BybyLang

```sh
make build-for-python
make python-example
make python-test
make c-example
make bybylang-test
```

- `build-for-python`: check và build `lib_c_api.nim` thành
  `libnimformers.dylib` trên macOS, `libnimformers.so` trên Linux hoặc
  `nimformers.dll` trên Windows. Chưa cài package Python.
- `python-example`: build shared library rồi chạy `examples/python/basic.py`.
- `python-test`: build shared library rồi chạy `test/python_test/` bằng pytest.
- `c-example`: build shared library, compile `examples/c/smoke.c`, rồi chạy
  `examples/c/smoke`. Recipe dùng tùy chọn linker kiểu Unix.
- `bybylang-test`: gọi runner build và chạy hai script native DSL/C ABI.
  Runner dùng compiler vendored và compile `lib_c_api.nim` vào AOT test;
  không cần cài package Python hoặc build shared library trước.

Để dùng đúng Python trong virtualenv:

```sh
make python-example PYTHON=python
make python-test PYTEST='python -m pytest' PYTEST_FLAGS=-q
```

Xem [hướng dẫn Python](python-guide.vi.md), [Python API](python-api.vi.md),
[hướng dẫn BybyLang](bybylang-guide.vi.md) và [BybyLang API](bybylang-api.vi.md).


## Shader Vulkan

```sh
make shaders
```

Cần `glslc` trong `PATH`. Target compile các file `.comp` trong
`vendor/bybylang/backends/vulkan/kernels/` sang `.spv` nếu output chưa có hoặc
cũ hơn source. Recipe vẫn kiểm tra timestamp khi dùng `make -B shaders`;
`-B` không tự bỏ điều kiện timestamp trong vòng lặp shell.


## Xem lệnh, đổi compiler và dọn build

```sh
make -n run-metal
make build NIM=/path/to/nim
```

`make -n` in recipe để kiểm tra mà không chạy training. `NIMFLAGS` có thể ghi
đè qua command line, nhưng sẽ **thay toàn bộ** flags mặc định; cần giữ các
`--path` của project khi tự đặt lại. Các target build hiện là `.PHONY`, nên
gọi lại sẽ chạy compiler dù binary đã tồn tại. Compiler có thể tái sử dụng cache.

```sh
make clean
```

Target xóa các binary được liệt kê trong `CLEAN_BINS`, shared library,
`nimcache/`, và các file `model_*.nimq` ở thư mục gốc. Danh sách hiện không
bao gồm `test/spectron/test_nimformer` do binary trainer nằm trong thư mục
source, và không reset `test/spectron/models/finetune.nimq` hay tokenizer.
Muốn bắt đầu một lần train Spectron mới, xem mục checkpoint trong
[hướng dẫn Spectron](spectron-guide.vi.md).
