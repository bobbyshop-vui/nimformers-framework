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

Backend `cpu` chạy cùng API, session, resident weights và micro-kernel mà
không cần GPU. Nó dùng kernel SSE4/AVX2/AVX-512/NEON chọn lúc chạy, chính
sách SSE cho op nhỏ (tránh hạ xung AVX) và chia block matmul theo L3; xem
[`docs/cpu-backend.vi.md`](docs/cpu-backend.vi.md).

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

Đo GFLOPS matmul FP32 trên CPU, Metal và Vulkan:

```sh
python3 benchmarks/bench_gflops.py --backends cpu,metal,vulkan

# chế độ tile 8096^3 (tránh watchdog iGPU với một dispatch quá lớn)
python3 benchmarks/bench_gflops.py --gpu-8096
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


### Benchmark inference

Tok/s prefill và decode của từng model trên Colab Tesla T4 (Vulkan) và
MacBook Pro Intel Iris Plus 640 (Metal và Vulkan/MoltenVK), kèm máy test và
lệnh đã chạy:

| Model | Benchmark |
| --- | --- |
| DeepSeek Coder 6.7B (`nim_inference`) | [`test/deepseek-inference/benchmark.md`](test/deepseek-inference/benchmark.md) |
| Codestral Mamba + QLoRA (`infer_lora`) | [`test/codestral/benchmark.md`](test/codestral/benchmark.md) |


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

Thêm `--no-progress` để ẩn dòng tok/s trực tiếp. Tốc độ:
[`test/deepseek-inference/benchmark.md`](test/deepseek-inference/benchmark.md).


### Chạy Codestral với adapter QLoRA

File adapter không chứa trọng số base Mamba. `infer_lora` đọc base `.nimq`
và thư mục tokenizer từ sidecar checkpoint. `make download-models` ghi
sidecar tokenizer mặc định. Chạy từ thư mục gốc:

```sh
make infer-lora \
  ARGS='test/codestral/models/jamba_parallel_lora.bin "Write a Python function to check if a string is a palindrome" 0.7 test/codestral/models/hf test/codestral/models/codestral_mamba_only.nimq'
```

Prompt mặc định được bọc trong template instruct Mistral v3:
`Write a Python function to check if a string is a palindrome` được gửi thành
`[INST] Write a Python function to check if a string is a palindrome[/INST]`
(đúng định dạng adapter đã train). Prompt đã bắt đầu bằng `[INST]` được gửi
nguyên văn; `BYBY_RAW_PROMPT=1` tắt việc bọc. Không truyền prompt thì dùng
prompt palindrome này.

`infer-lora` nhận `<checkpoint.bin> <prompt> <temperature> [hfDir]
[nimqPath]`. Metadata binary chọn base `.nimq`; sidecar trỏ thư mục
tokenizer model. Hai đối số cuối cấu hình nhánh base-Mamba fallback.
`infer-mamba` là alias `infer-lora`. Mặc định `auto` tự chọn backend trên
macOS và hệ khác. Theo loại thiết bị, dGPU dùng VRAM device-local, iGPU
dùng host-visible/shared memory. Ghi đè backend bằng
`make infer-lora BYBY_INFER_BACKEND=metal` hoặc `BYBY_INFER_BACKEND=vulkan`.

Tốc độ: [`test/codestral/benchmark.md`](test/codestral/benchmark.md).

#### Flag inference (`infer_lora`)

Mọi flag là biến môi trường, đặt trên dòng `make infer-lora` hoặc trước binary
(`BYBY_INFER_BACKEND=vulkan ./test/codestral/infer_lora_bin ...`). Giá trị đặt
tay luôn thắng mặc định. Chi tiết các nhánh prefill:
[`test/codestral/INFERENCE.md`](test/codestral/INFERENCE.md).

Prompt, sampling và output:

| Flag | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `BYBY_RAW_PROMPT` | `0` | `1` gửi prompt nguyên văn, không bọc `[INST] ...[/INST]` |
| `BYBY_KEEP_TRAILING_WS` | `0` | `1` giữ khoảng trắng cuối prompt (mặc định bị cắt: nó tạo token `▁` lẻ) |
| `BYBY_MAX_NEW_TOKENS` | `60` | số token sinh |
| `BYBY_TOP_P` | `0.9` | nucleus sampling |
| `BYBY_MIN_P` | `0.05` | min-p sampling |
| `BYBY_REPEAT_PENALTY` | `1.1` | phạt lặp |
| `BYBY_REPEAT_WINDOW` | `64` | số token phạt lặp nhìn lại |
| `BYBY_ALLOW_SPECIAL` | `0` | `1` cho sampler chọn token điều khiển/đặc biệt |
| `BYBY_PROGRESS` | `auto` | `0` tắt dòng tok/s trực tiếp (chỉ bật khi stdout/stderr là TTY) |
| `BYBY_MACOS_HIGH_PRIORITY` | `1` | macOS: chạy thread inference ở QoS user-interactive; `0` giữ QoS kế thừa |

Backend và bộ nhớ:

| Flag | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `BYBY_INFER_BACKEND` | `auto` | `auto`, `metal`, `vulkan` |
| `BYBY_MODEL_MMAP` | `1` | map `.nimq` thay vì đọc vào RAM (merge in-place cần) |
| `BYBY_MERGE_LORA` | `inplace` | merge LoRA vào trọng số FP4 trên GPU; `1` = `inplace`, `0` giữ adapter tách rời |
| `BYBY_MERGE_LORA_TRY_FP4` | `0` | `1` cho merge có guard thử lượng tử lại về 4-bit (chỉ giữ khi sai số delta RMS ≤ 0,1%) |
| `BYBY_MERGE_SCRATCH_MIB` | `64` | scratch GPU của merge in-place (tối đa 64) |
| `BYBY_MEMORY_LIMIT_MB` | `7000` | ngưỡng RSS tiến trình của merge in-place |
| `BYBY_KEEP_ZERO_LORA` | `0` | `1` giữ adapter toàn số 0 (mặc định bỏ: delta đúng bằng 0) |
| `BYBY_APPLY_LMHEAD_LORA` | tắt | `1` áp adapter LoRA lên `lm_head` |
| `BYBY_ONE_DISPATCH` | `0` | `1` chạy decode bằng một mega-kernel thường trú (thử nghiệm Metal) |
| `BYBY_OD_TG` | `16` | số threadgroup của kernel one-dispatch (phải cùng thường trú) |
| `BYBY_MVM_MAX_SCRATCH_MB` | tự động | Metal: giới hạn một buffer scratch micro-VM |
| `BYBY_VM_GROUPS` | `1` | `0` upload/đọc từng nhóm micro-VM riêng (nhánh cũ) |
| `BYBY_RMS_TG` | `256` | kích thước threadgroup của kernel RMSNorm |
| `BYBY_METAL_JIT_ARCHIVE` | `~/Library/Caches/nimformers/metal-jit.binarchive` | binary archive pipeline Metal |
| `BYBY_VULKAN_DEVICE` | GPU tốt nhất (dGPU, NVIDIA trước) | chỉ số thiết bị Vulkan |
| `BYBY_VULKAN_DEVICE_LOCAL` | `auto` | trọng số trong VRAM trên dGPU (`auto`/`1`), `0` để host-visible; iGPU luôn dùng bộ nhớ chung |
| `BYBY_VULKAN_SCRATCH` | `auto` | buffer scratch: `auto` (VRAM trên dGPU), `device`, `host` |
| `BYBY_VULKAN_MMAP_IMPORT` | `1` | iGPU: import trọng số đã mmap không copy (`VK_EXT_external_memory_host`) |
| `BYBY_VULKAN_POOL_MB` | `2048` | giới hạn buffer Vulkan trong pool |
| `BYBY_VULKAN_TENSOR_CORES` | `1` | `0` tắt `KHR_cooperative_matrix` (tensor core FP16/FP8) |
| `BYBY_VULKAN_SPLIT_K` | `auto` | số lát split-K của GEMM tensor core (`1` = tắt) |
| `BYBY_VULKAN_MAX_TG` | max của thiết bị (≤1024) | giới hạn kích thước workgroup Vulkan |
| `BYBY_VULKAN_JIT_DIR` | thư mục tạm | thư mục cache SPIR-V (trỏ vào nơi lưu lâu dài) |
| `BYBY_VULKAN_DISPATCH_CACHE` | `1` | `0` biên dịch lại shader mỗi dispatch (chẩn đoán) |
| `BYBY_VULKAN_STRICT` | `0` | `1` báo lỗi khi offset buffer vượt phạm vi thay vì kẹp lại |

Prefill:

| Flag | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `BYBY_PREFILL_RESIDENT` | `1` | prefill thường trú trên GPU (mọi layer trên GPU, state chuyển GPU sang GPU) |
| `BYBY_PREFILL_PARALLEL` | `1` | `0` tắt prefill song song theo chuỗi (từng token một) |
| `BYBY_PREFILL_PAR_MIN_T` | `2` (resident), `32` | prompt ngắn nhất đi prefill song song |
| `BYBY_PREFILL_BATCH_SIZE` | `128` | số token prompt mỗi chunk prefill (`2..128`) |
| `BYBY_PREFILL_LAYERS_PER_COMMIT` | `8` Vulkan, `4` Metal | số layer mỗi command buffer |
| `BYBY_SSD` | `auto` | scan Mamba2 bằng pass SSD/FLA theo chunk: tensor core, không có thì tile FP32; `scalar` bỏ tensor core, `0` dùng kernel scan (Vulkan) |
| `BYBY_SSD_CHUNK` | `64` | độ dài chunk SSD (lũy thừa 2, 16..256) |
| `BYBY_SSD_MIN_T` | `0` | prompt ngắn hơn giá trị này dùng scan thay vì SSD |
| `BYBY_SSD_STATE_MB` | `1024` | giới hạn scratch state các chunk SSD; chunk tăng gấp đôi đến khi vừa |
| `BYBY_PREFILL_SCAN` | `auto` | nhánh scan khi tắt SSD: `auto` (đo rồi chọn), `serial`, `parallel`, `lanes`, `coalesced` |
| `BYBY_PREFILL_SCAN_LANES` | `1` | `0` bỏ lane scan dùng thanh ghi khỏi ứng viên của `auto` |
| `BYBY_PREFILL_SCAN_AUTOTUNE_MAX_MIB` | `512` | arena thử lớn nhất mà autotune scan cấp phát |
| `BYBY_MAMBA_SCAN_PARALLEL` | `1` Metal, `0` Vulkan | scan song song ở nhánh không thường trú |
| `BYBY_LP_SCAN_SERIAL` | tắt | `1` ép scan Linear++ tuần tự |
| `BYBY_PREFILL_LP_LANES` | `1` | `0` tắt lane scan Linear++ |
| `BYBY_PREFILL_GEMM` | `auto` | `auto`, `m8` (tile FP4 đóng gói), `mps` (FP32 giải nén + MPS, Apple Silicon) |
| `BYBY_PREFILL_MPS` | `1` | `0` không bao giờ xét MPS |
| `BYBY_PREFILL_FP4_KERNEL` | `auto` | `m4` ép tile FP4 4 hàng |
| `BYBY_PREFILL_2COL` | `1` | `0` chọn tile FP4 một cột |
| `BYBY_PREFILL_TAIL4` | `1` | `0` đệm các hàng cuối lên 8 thay vì tile đuôi 4 hàng |
| `BYBY_PREFILL_UNROLL` | `2` | `1` nạp một word trọng số mỗi vòng trong tile FP4 |
| `BYBY_FP4_PAD_M` | `1` | `0` tắt đệm số hàng lẻ lên tile FP4 |
| `BYBY_PREFILL_FP4_TILED`, `BYBY_PREFILL_FP32_TILED` | `1` | `0` bỏ GEMM tensor core dạng tile (được kiểm parity trước) |
| `BYBY_PREFILL_PREWARM_PACKED` | `0` | `1` biên dịch sẵn cả tile FP4 đóng gói khi dùng tensor core |
| `BYBY_PREFILL_LORA_DOWN` | `1` | `0` tắt kernel fuse phép chiếu xuống LoRA |
| `BYBY_PREFILL_SPARSE` | `1` | `0` tắt prefill sparse trong hybrid block |
| `BYBY_PREFILL_KEEP_SCRATCH` | `0` | `1` giữ arena prefill sau prefill (thử nghiệm throughput) |
| `BYBY_SCAN_NCHUNK` | `16` | số command buffer chia scan backward (tránh watchdog) |

Decode và tuning:

| Flag | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `BYBY_GEMV_TG` | `auto` | độ rộng threadgroup GEMV (`auto` = đã tune, `max` hoặc một số) |
| `BYBY_GEMV_VARIANT` | `auto` | `0` 2 hàng, `1` nạp 32-bit, `2` 4 hàng mỗi threadgroup |
| `BYBY_GPU_AUTOTUNE` | `1` | `0` bỏ tune kernel FP4 (khởi động lạnh nhanh hơn) |
| `BYBY_NORM_PRESET` | theo checkpoint | preset gated-RMSNorm (mặc định `rms-fin-gn`) |
| `BYBY_NORM_AUTO` | `1` | `0` bỏ qua preset do norm sweep gợi ý |
| `BYBY_MAMBA_NORM_GROUPS` | `n_groups` của model | số nhóm gated-RMSNorm (`1` = kiểu cũ full width) |

Chẩn đoán (chậm hơn, không phải nhánh nhanh):

| Flag | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `BYBY_PREFILL_PROFILE` | `1` | thời gian từng giai đoạn prefill |
| `BYBY_PREFILL_DIAG` | `0` | `1` commit sau mỗi giai đoạn prefill (thời gian GPU thật mỗi giai đoạn), `2`/`3` chi tiết hơn |
| `BYBY_DECODE_PROFILE` | `0` | thời gian decode mỗi token |
| `BYBY_DECODE_SPLIT` | `0` | chia mỗi token thành nhiều command buffer để đo từng phần |
| `BYBY_DEBUG_SAMPLING` | `0` | log trạng thái sampler; `BYBY_DEBUG_SAMPLING_ALL=1` mọi bước, `BYBY_DEBUG_SAMPLING_STEPS` (5) N bước đầu |
| `BYBY_DEBUG_TOP5` | `0` | log top-5 logit |
| `BYBY_TRACE_TOKENS` | `0` | trace N token sinh đầu tiên |
| `BYBY_TRACE_BLOCKS` | `0` | trace activation từng block |
| `BYBY_DUMP_LOGITS` | tắt | đường dẫn file để dump logit |
| `BYBY_G_DIAGNOSTICS` | `0` | thống kê các block G (gated) |
| `BYBY_OD_CHECK`, `BYBY_OD_VERBOSE` | `0` | kiểm lỗi và log của one-dispatch |
| `BYBY_LPCHECK` | `0` | so kernel bước Linear++ với công thức đóng |
| `BYBY_NORM_SWEEP` | `0` | chia đôi tìm preset norm (`BYBY_NORM_SWEEP_LIST`, `BYBY_NORM_SWEEP_TOKENS` = 12) |
| `BYBY_INFER_VERBOSE` | tắt | thêm log inference |
| `BYBY_LORA_VERBOSE` | `0` | log adapter thiếu trên GPU |
| `BYBY_JIT_DUMP_SOURCE` | `0` | in mã nguồn kernel sinh ra |
| `BYBY_VULKAN_JIT_VERBOSE` | `0` | log mỗi lần biên dịch shader Vulkan |
| `BYBY_VULKAN_SPLIT_K_VERBOSE` | `0` | log lựa chọn split-K |


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

Prompt Llama/DeepSeek prefill mọi layer trên GPU trong một chuỗi dispatch (RMSNorm, QKV,
RoPE, ghi KV cache, **flash attention**, oProj, FFN) và chỉ đọc về hàng cuối. Flash
attention đưa các key trong cache qua bộ nhớ on-chip với softmax online và đọc thẳng KV
cache, nên các lượt chat sau cũng prefill thay vì decode từng token. Nó chạy trên tensor
core Vulkan khi GPU có (KHR_cooperative_matrix, với lazy rescale output của
FlashAttention-4), nếu không thì chạy kernel scalar FP32 trên Vulkan và Metal; `--verbose`
in `[prefill] flash attention (coop|scalar)`. Mỗi kernel được so với kết quả tham chiếu
lúc nạp model và bị bỏ qua nếu sai. `BYBY_FLASH_ATTENTION=scalar` bỏ kernel tensor core,
`BYBY_FLASH_PREFILL=0` quay về prefill cũ, `make flash-attention-test` kiểm tra kernel trên
Metal và Vulkan.

Trên Metal, GEMM prefill (M >= 8) chạy kernel Q4 dạng tile (tile output 64 x 128 mỗi
threadgroup, trọng số giải nén một lần cho 64 hàng, tile thanh ghi FP32 8 x 8), nhanh khoảng
5 lần kernel batched cũ trên Intel Iris; `BYBY_METAL_Q4_GEMM=0` quay về kernel cũ, `make
q4-gemm-test` kiểm tra và đo nó. Trên Mac Intel/AMD và Vulkan, việc prefill và decode được
chia thành command buffer khoảng `BYBY_PREFILL_CB_MS` (mặc định 300) theo tốc độ đo được,
để watchdog GPU của macOS không giết command buffer dài; Apple Silicon giữ nguyên command
buffer.

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
