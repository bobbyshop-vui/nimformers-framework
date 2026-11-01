# Nimformer Framework

[English](readme.md) | [Tiếng Việt](readme.vi.md)

Nimformer là framework Nim cho inference và training transformer trên GPU
Metal/Vulkan. Tài liệu API, hướng dẫn sử dụng, định dạng file và lập trình
GPU BybyLang nằm trong [`docs/`](docs/README.vi.md).

Bắt đầu ở [mục lục tài liệu](docs/README.vi.md), rồi xem
[Bắt đầu sử dụng](docs/getting-started.vi.md) để có chương trình tối thiểu.

Các hướng dẫn thực hành:

- [Lệnh Make](docs/make-commands.vi.md): target build/run, vị trí binary,
  dữ liệu cần có, `ARGS`, backend và cleanup.
- [Train/chat Spectron với `test_nimformer`](docs/spectron-guide.vi.md): cấu
  hình, JSONL, BPE, checkpoint/resume, loss và ví dụ import Nim.
- [Python từ source đến API](docs/python-guide.vi.md): wrapper/shared library,
  NumPy, lượng tử hóa, NIMQ, micro-VM và xử lý lỗi.
- [BybyLang thực hành](docs/bybylang-guide.vi.md): compiler, DSL/AOT, session,
  micro-kernel và C ABI.
- [formers.n](#chạy-deepseek-hoặc-codestral-bằng-formersn-formers-cli): một
  `formers-cli` kiểu llama.cpp cho DeepSeek/Llama và Codestral (Mamba-2, LoRA),
  một prompt hoặc chat.

Chương trình/dữ liệu theo model nằm trong `test/spectron/`,
`test/codestral/`, `test/deepseek-inference/`; asset nằm trong `models/` của
mỗi phần. Module micro-kernel FP4 GEMV nằm trong `kernelfortest/`;
runtime chung ở thư mục gốc.


## Python và C ABI

Core framework có shared library với C ABI ổn định (`lib_c_api.nim`), cung
cấp chọn backend, tensor sở hữu bộ nhớ, kernel toán/neural network,
lượng tử hóa int8/int4/fp8/fp4/CustomFloat, state dict NIMQ, tạo/dispatch
bytecode micro-VM, resident weights, tokenizer và inference Llama.

Build và chạy test Python:

```sh
make build-for-python
make python-test
```

```python
import numpy as np
from nimformers import Context

with Context() as ctx:                 # auto -> Metal/Vulkan
    a = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(8, 16).astype(np.float32)
    print(ctx.matmul(a, b).shape)
    print(ctx.softmax(a).sum(axis=-1))
```

Package ở `nimformers/`, header C ở `include/nimformers.h`, ví dụ ở
`examples/python/` và `examples/c/`. Tài liệu đầy đủ:
[`docs/python-api.md`](docs/python-api.vi.md).

Compiler BybyLang vendored cũng có thể import cùng ABI:

```sh
make bybylang-test     # test/bybylang_test/*.bybylang
```


## Ghi nhận đồng tác giả model AI và lời cảm ơn

Các nhà phát triển model sau được ghi nhận; liên kết GitHub dùng để tham khảo:

- **Mistral AI** ([@mistralai](https://github.com/mistralai)): Devstral, Codestral, Mistral Large
- **RWKV** ([@RWKV](https://github.com/RWKV)): RWKV-3
- **Anthropic** ([@anthropics](https://github.com/anthropics), [@claude](https://github.com/claude)): Claude Sonnet 4.6, Claude Sonnet 5, Claude Sonnet 5.5, Claude Opus 4.6, Claude Opus 5, Claude Opus 5.5
- **Google DeepMind** ([@google-deepmind](https://github.com/google-deepmind)): Gemini 3.8 Flash, Gemini 3.7 Flash
- **Moonshot AI** ([@MoonshotAI](https://github.com/MoonshotAI)): Kimi K3
- **Allen Institute for AI (Ai2)** ([@allenai](https://github.com/allenai)): SERA-8B
- **NVIDIA** ([@nvidia](https://github.com/nvidia)): Nemotron 3 Ultra
- **DeepSeek** ([@deepseek-ai](https://github.com/deepseek-ai)): DeepSeek V4 Pro, DeepSeek V4.1 Flash
- **Meta** ([@facebookresearch](https://github.com/facebookresearch), [@meta-llama](https://github.com/meta-llama)): Muse Spark, Llama 3.2
- **Alibaba / Qwen team** ([@QwenLM](https://github.com/QwenLM)): Qwen-Coder-Next
- **TinyLlama** ([@TinyLlama](https://github.com/TinyLlama)): TinyLlama 1.1B
- **OpenAI** ([@openai](https://github.com/openai)): GPT-6 Luna, GPT-5.6 Luna, GPT-6.1 Sol
- **xAI** ([@xai-org](https://github.com/xai-org)): Grok 4.6, Grok 4.7

Danh sách ghi nhận model AI và nhà phát triển này không thay thế công lao
cá nhân hoặc hàm ý các nhà phát triển bảo chứng cho project.


## Model Hugging Face và inference

Các repository chứa file model dùng bởi target inference:

| Repository | Nội dung | Target Make |
| --- | --- | --- |
| [DeepSeek Coder 6.7B NimQ](https://huggingface.co/bobbydeveloper2014/deepseek-coder-6.7b-nimq) | `model.nimq`, `config.json` và tokenizer cho `test/deepseek-inference/nim_inference.nim` | `infer-deepseek`, `formers` |
| [Codestral v0.1 7B NimQ](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-7b-nimq) | trọng số base Codestral, config và tokenizer Tekken/HF | base của `infer-lora`, `formers` |
| [Codestral v0.1 QLoRA LinearAttention](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-nimq-qlora-linearattention) | checkpoint adapter `jamba_parallel_lora.bin`, dùng base Codestral NimQ trên | adapter của `infer-lora`, `formers` |


### Tải model

Các repo Hugging Face này công khai. Từ thư mục gốc, tải asset DeepSeek và
Codestral vào đúng đường dẫn target inference sử dụng:

```sh
make download-models
```

Target tạo thư mục model dưới `test/` khi cần, tải trọng số base,
tokenizer/config và adapter, rồi ghi sidecar thư mục tokenizer Codestral
cho `infer-lora`. Lệnh dùng `curl` với retry và resume.

Nếu đã có trọng số Codestral nhưng `newHfTokenizer` báo thiếu file, chỉ tải
config/tokenizer:

```sh
make download-codestral-tokenizer
```

Codestral inference đọc trực tiếp `tokenizer.model.v3` hoặc
`tokenizer.model` qua tokenizer Nim thuần trong
`test/codestral/sentencepiece_native.nim`, dùng quy tắc BPE/Unigram,
normalization và byte fallback của model. Không cần gói tokenizer Python
hay thư viện SentencePiece. `make build-infer-lora` dùng
`-d:codestralInference`, loại loader PyTorch/Nimpy khỏi build inference.


### Chạy DeepSeek Coder bằng `nim_inference`

`infer-deepseek` build `test/deepseek-inference/nim_inference.nim`. Đối số
đầu là thư mục model cục bộ chứa config/tokenizer, đối số hai là trọng số
`.nimq`; tiếp theo là backend, prompt, giới hạn token, temperature, top-p
và system prompt.

```sh
make infer-deepseek ARGS='./test/deepseek-inference/models ./test/deepseek-inference/models/model.nimq auto "Write a Python function for Fibonacci." 128 0.7 0.9 "You are an AI programming assistant."'
```

Chọn `metal`, `vulkan` hoặc `auto` khi máy hỗ trợ.

`nim_inference` load config/tokenizer bằng thư viện chuẩn Nim, đọc trực
tiếp vocabulary BPE byte-level, merge, added token và chat template
DeepSeek cục bộ. Không cần Nimpy, Transformers hay gói tokenizer Python.
Build riêng bằng `make build-nim-inference`.


### Chạy Codestral với adapter QLoRA

File adapter không chứa trọng số base Mamba. `infer_lora` đọc base `.nimq`
và thư mục tokenizer từ sidecar checkpoint. `make download-models` ghi
sidecar tokenizer mặc định. Chạy từ thư mục gốc:

```sh
make infer-lora \
  ARGS='test/codestral/models/jamba_parallel_lora.bin "[INST] Write a Python function for Fibonacci.[/INST]" 0.7 test/codestral/models/hf test/codestral/models/codestral_mamba_only.nimq'
```

`infer-lora` nhận `<checkpoint.bin> <prompt> <temperature> [hfDir]
[nimqPath]`. Metadata binary chọn base `.nimq`; sidecar trỏ thư mục
tokenizer model. Hai đối số cuối cấu hình nhánh base-Mamba fallback.
`infer-mamba` là alias `infer-lora`. Mặc định `auto` tự chọn backend trên
macOS và hệ khác. Theo loại thiết bị, dGPU dùng VRAM device-local, iGPU
dùng host-visible/shared memory. Ghi đè backend bằng
`make infer-lora BYBY_INFER_BACKEND=metal` hoặc `BYBY_INFER_BACKEND=vulkan`.


### Chạy DeepSeek hoặc Codestral bằng formers.n (`formers-cli`)

formers.n (`other/formers.n/`) là CLI kiểu llama.cpp viết bằng BybyLang trên các
block của nimformers. Chỉ cần trỏ vào thư mục model: nó nhận diện kiến trúc từ
`config.json` và header `.nimq`, rồi chạy đúng nhánh với một binary và một bộ flag:

| Nhận diện | Nhánh | Thư mục ví dụ |
| --- | --- | --- |
| Họ Llama (`LlamaForCausalLM`, DeepSeek Coder) | block transformer của `nim_inference` | `test/deepseek-inference/models` |
| Mamba-2 (Codestral Mamba) | nhánh base-Mamba của `infer_lora` | thư mục không có LoRA `.bin` |
| Mamba-2 + LoRA / Linear++ | nhánh checkpoint của `infer_lora` | `test/codestral/models` |

#### Build

```sh
make formers-cli
```

Target này build compiler BybyLang trong `vendor/bybylang` khi cần, biên dịch AOT
`other/formers.n/src/formers_cli.bybylang` sang Nim và ghi ra
`other/formers.n/formers-cli`. `make build-formers` là cùng target. Chỉ cần Nim và
trình biên dịch C; backend GPU giống các target inference khác (Metal/Vulkan).

#### Chạy nhanh

Tải model trước (`make download-models`), rồi từ thư mục gốc:

```sh
# Codestral + QLoRA, chat nhiều lượt
other/formers.n/formers-cli --models test/codestral/models -cnv -n 256

# DeepSeek Coder, một prompt
other/formers.n/formers-cli --models test/deepseek-inference/models \
  -p "Write a Python function for Fibonacci." -n 128

# Chỉ in kết quả nhận diện, không nạp trọng số
other/formers.n/formers-cli --models test/codestral/models --detect-only
```

Hoặc qua make (`make formers` tự build trước):

```sh
make formers                                              # chat với test/codestral/models
make formers FORMERS_MODELS=test/deepseek-inference/models
make formers ARGS='--models test/codestral/models -p "Write quicksort in Python." -n 200'
```

#### `--models` cần có gì

| File | Cách tìm | Ghi đè |
| --- | --- | --- |
| trọng số | `.nimq` duy nhất; nếu nhiều thì lấy file ghi trong metadata `nimqPath` của LoRA, không thì `model.nimq` | `-m FILE` |
| config | `config.json` trong thư mục hoặc thư mục con (Codestral để trong `hf/`) | `--config FILE` |
| tokenizer | thư mục có `tokenizer.json` hoặc `tokenizer.model` | `--tokenizer DIR` |
| LoRA (Mamba-2) | `.bin` duy nhất có header `NIMBIN1`/`NIMBIN4`; nhiều file là lỗi | `--lora FILE`, `--no-lora` |

`--detect-only` in các file, kiến trúc, nhánh chạy, kích thước model, và lý do nếu
model không được hỗ trợ (ví dụ Mamba-1 hoặc `model_type` lạ).

#### Một prompt và chat

- Không có `-cnv`: `-p TEXT` (hoặc `-f FILE`) được bọc trong chat template của model và
  sinh một lần; `--system TEXT` đặt system prompt, `--raw-prompt` gửi nguyên văn.
- `-cnv` (hoặc `-i`) mở chat: ngữ cảnh (KV cache với Llama, recurrent state với Mamba-2)
  được giữ giữa các lượt, `-n` giới hạn mỗi câu trả lời, `-p` thành system prompt. Gõ
  `/exit` (hoặc `/quit`, `/bye`) để thoát, kết thúc dòng bằng `\` để viết tiếp dòng sau,
  Ctrl+C dừng câu trả lời, Ctrl+C ở dấu nhắc thì thoát.
- `-c N` đặt kích thước context cho chat Llama (mặc định 1024 token).
- Trong chat, mỗi câu trả lời kết thúc bằng tok/s của `prompt eval` và `eval`; một prompt
  Llama đơn kết thúc bằng khối `Speed stats` (prefill, decode, total).

Trước prompt đầu tiên, warmup biên dịch pipeline GPU và in `ready: warmup ...`;
`--no-warmup` bỏ qua, khi đó câu trả lời đầu chịu thời gian biên dịch. Với Mamba-2,
prompt ngắn hơn 129 token chạy bằng kernel decode đã warm; prompt dài hơn đưa các chunk
đủ 128 token (`BYBY_PREFILL_BATCH_SIZE`) qua prefill song song. Trên dGPU Vulkan có tensor
core (dòng thiết bị trong `--verbose` có `prefill=tensor-cores-only`), phần còn lại của
prompt cũng prefill theo chunk 16 token, các lượt chat sau cũng prefill (tiếp tục
recurrent state), và warmup biên dịch sẵn kernel 16 token; mọi GEMM prefill khi đó chạy
trên tensor core, không bao giờ trên kernel scalar. `BYBY_VULKAN_TENSOR_CORES=tune` cho
kernel scalar cạnh tranh lại, `0` tắt tensor core.

#### Merge LoRA

LoRA của Mamba-2 được merge vào trọng số FP4-KM nằm sẵn trên GPU lúc khởi động
(`infer_lora.mergeResidentLoraIfEnabled`, giống `make infer-lora`). Kernel decode khi đó
bỏ phần adapter A/B, và output khớp với checkpoint đã merge. Merge thêm một lượt xử lý
lúc khởi động.

- `--no-merge-lora` (hoặc `BYBY_MERGE_LORA=0`) giữ adapter tách rời, fuse trong kernel.
- Khi bật mặc định, merge tự bỏ qua nếu `BYBY_ONE_DISPATCH=1`, `BYBY_MODEL_MMAP=0` hoặc
  tiến trình không đủ RAM (`BYBY_MEMORY_LIMIT_MB`, mặc định 7000), và inference chạy tiếp
  bằng adapter tách rời. Tự đặt `BYBY_MERGE_LORA=inplace` thì các trường hợp đó thành lỗi.

#### Log

formers-cli mặc định im lặng. Log của loader, LoRA, JIT, Metal và tune kernel đi vào
`/dev/null`; chỉ hiện dòng loading, `ready`, text sinh ra, thống kê và lỗi.
`-v`/`--verbose` hiện mọi dòng log. Lỗi ở chế độ im lặng in thông báo và gợi ý chạy lại
với `--verbose`.

#### Flag chính

| Flag | Ý nghĩa | Mặc định |
| --- | --- | --- |
| `-n, --n-predict N` | số token sinh (mỗi câu trả lời khi chat) | 128 |
| `--temp T` | temperature, `0` = greedy | 0.7 |
| `--top-p P` | nucleus sampling, `1` = tắt | 0.9 |
| `--min-p P` | min-p, `0` = tắt | 0.05 |
| `--repeat-penalty R`, `--repeat-last-n N` | phạt lặp và cửa sổ của nó | 1.0 (tắt), 64 |
| `-s, --seed N` | seed sampling | ngẫu nhiên |
| `--backend NAME` | `auto`, `metal`, `vulkan`, `cuda` (= `vulkan`) | `auto` |
| `-ngl N` | số layer trên GPU; mọi layer chạy trên GPU, giá trị nhỏ hơn bị từ chối | tất cả |
| `--no-progress` | không hiện tok/s trên tiêu đề cửa sổ terminal | |

`formers-cli --help` liệt kê mọi flag. Mã thoát: 0 ok, 1 lỗi khi chạy, 2 sai option,
3 model không hỗ trợ.

#### Target make

| Target | Làm gì |
| --- | --- |
| `make formers-cli` / `make build-formers` | build `other/formers.n/formers-cli` |
| `make formers` | build rồi chat với `FORMERS_MODELS` (mặc định `test/codestral/models`); `ARGS='...'` thay toàn bộ đối số |
| `make formers-test` | test nhận diện kiến trúc, không nạp model, không cần GPU |
| `make formers-smoke` | tạo một Llama ngẫu nhiên tí hon và chạy end to end trên GPU |
| `make formers-clean` | xoá build và binary của formers.n (`make clean` cũng xoá) |

Chi tiết hơn (cấu trúc mã nguồn, luật nhận diện, phần chưa làm):
[`other/formers.n/README.md`](other/formers.n/README.md).


### Train Spectron bằng `test/spectron/test_nimformer.nim`

Đây là chương trình training Spectron. `make run` compile/chạy, đọc Dolly
và sáu JSONL Python/Nim dưới `test/spectron/`, train/load BPE byte-level,
train model và lưu checkpoint/tokenizer dưới `test/spectron/models/`.
Mỗi file code được quét, giữ tối đa 4.000 mẫu hợp lệ bằng reservoir
sampling để dùng cả sáu nguồn mà không load toàn bộ khoảng 1,3 GB vào RAM.
Phần train BPE lấy văn bản từ mỗi nguồn. Sửa `CFG_*` đầu source để đổi
đường dẫn, giới hạn mẫu, kích thước model, step, batch, sequence length
hoặc learning rate.


#### Tải dataset code Python và Nim

Năm shard Python nằm ở
[bobbydeveloper2014/python-code](https://huggingface.co/datasets/bobbydeveloper2014/python-code),
dataset Nim/Python ở
[bobbydeveloper2014/mega-nim-python-code](https://huggingface.co/datasets/bobbydeveloper2014/mega-nim-python-code).
Từ thư mục gốc, tải cả sáu file vào `test/spectron/`:

```sh
make download-datasets
```

Target tạo thư mục nếu cần và dùng `curl` retry/resume. Mỗi shard Python
khoảng 250 MB; trainer chỉ giữ mẫu nên không cần load toàn dataset vào RAM.

```sh
make run
```

`make run` dùng backend `auto`, train 6.000 step, batch 32, sequence length
128. Nếu chưa có tokenizer thì ghi
`test/spectron/models/spectron_bpe_tokenizer.json`; lưu
`test/spectron/models/finetune.nimq` mỗi năm step và resume nếu checkpoint
đã có. File lưu trọng số và step, nhưng moment Adam khởi tạo lại khi
resume. Trên macOS, `make run-metal` chọn Metal lúc compile.


### Chạy checkpoint Spectron đã train

Sau training, `make run-spectron` build chương trình tương tác
`test/spectron/test_inferencespectron.nim`, load checkpoint/tokenizer từ
`test/spectron/models/` để inference:

```sh
make run-spectron
```

Nhập prompt ở `>>>`; dùng `exit` hoặc `quit` để thoát. Ba repo Hugging Face
phía trên phục vụ inference DeepSeek/Codestral, khác demo training Spectron cục bộ.
