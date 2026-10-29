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
| [DeepSeek Coder 6.7B NimQ](https://huggingface.co/bobbydeveloper2014/deepseek-coder-6.7b-nimq) | `model.nimq`, `config.json` và tokenizer cho `test/deepseek-inference/nim_inference.nim` | `infer-deepseek` |
| [Codestral v0.1 7B NimQ](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-7b-nimq) | trọng số base Codestral, config và tokenizer Tekken/HF | base của `infer-lora` |
| [Codestral v0.1 QLoRA LinearAttention](https://huggingface.co/bobbydeveloper2014/codestral-v0.1-nimq-qlora-linearattention) | checkpoint adapter `jamba_parallel_lora.bin`, dùng base Codestral NimQ trên | adapter của `infer-lora` |


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
