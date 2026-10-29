
# Tài liệu Nimformer Framework

[English](README.md) | [Tiếng Việt](README.vi.md)

Thư mục chứa tài liệu sử dụng API công khai Nimformer. Chọn trang theo
module import; `nimformers.nim` là facade chung cho model cơ bản, tensor,
lượng tử hóa, backend, session và micro-VM.


## Bắt đầu sử dụng

- [Lệnh Make](make-commands.vi.md): mọi target, build/run, vị trí binary,
  backend, đối số và asset.
- [Hướng dẫn Spectron](spectron-guide.vi.md): `test_nimformer`, cấu hình,
  dữ liệu, tokenizer, checkpoint và loss.
- [Hướng dẫn Python](python-guide.vi.md): cài source và ví dụ API đầy đủ.
- [Hướng dẫn BybyLang](bybylang-guide.vi.md): compiler, DSL/AOT, session,
  micro-kernel và C ABI.
- [Bắt đầu sử dụng](getting-started.vi.md): yêu cầu cài/build, import,
  chọn backend và chương trình nhỏ.
- [Python và C ABI](python-api.vi.md): binding ctypes/NumPy, header C,
  lượng tử hóa, state dict NIMQ, micro-VM, tokenizer và sinh Llama.
- [API backend](backend-api.vi.md): khởi tạo backend và phép GPU `be*`.
- [API tensor](tensor-ops-api.vi.md): view, shape, giảm chiều, so sánh, scan, dropout.
- [API model cơ bản](base-model-api.vi.md): tensor, layer, transformer block,
  forward/backward và APF Adam.
- [API model Jamba lai](jamba-model-api.vi.md): model Mamba/attention hỗn hợp,
  batch forward, train và checkpoint.
- [API inference LLaMA](llama-inference-api.vi.md): RoPE, RMSNorm, GQA,
  KV cache, load trọng số, sinh token và tokenizer HF.
- [API LoRA](lora-api.vi.md): adapter, gắn/merge, chuẩn bị GPU và sinh token.
- [Fine-tune và resume LoRA](finetune-lora-api.vi.md): batch stream,
  optimizer trên đĩa và phục hồi checkpoint.
- [API training](training-api.vi.md): thành phần training, dataset,
  optimizer/checkpoint và entry point CLI.


## Định dạng dữ liệu và độ chính xác

- [CustomFloat và APF](customfloat-apf.vi.md): mã hóa số thực tùy chỉnh,
  chọn độ chính xác thích nghi.
- [Lượng tử hóa và `.nimq`](quantization-nimq.vi.md): các kind, encode/decode,
  state dict và xuất `.nimq`.
- [Checkpoint và binary](checkpoint-formats.vi.md): `.bin`, `.qlora.bin`,
  `.nimq` và tiến độ từng layer có thể resume.
- [API tokenizer](tokenizer-api.vi.md): load tokenizer HF, encode/decode, chat template.


## Lập trình GPU

- [API session GPU](gpu-session-api.vi.md): encode nhiều phép/session,
  buffer resident và vòng đời.
- [API micro-VM/JIT](microkernel-api.vi.md): instruction bytecode, lower,
  cache JIT và giới hạn kernel persistent.
- [Test tích hợp BybyLang](../test/bybylang_test/README.vi.md): import
  `lib_c_api`, kiểm tra C ABI và native DSL `gpu ...`.
- [API BybyLang GPU/DSL](bybylang-api.vi.md): hàm backend, cú pháp DSL,
  session, import và micro-kernel native.
- [Tham khảo ngôn ngữ BybyLang](bybylang-language.vi.md): bản tài liệu
  song ngữ cục bộ của README vendored, gồm cú pháp và mô phỏng phần cứng.
- [Codestral ít RAM](codestral-low-memory.vi.md): mmap model/adapter,
  merge có giới hạn, workgroup và Vulkan Tensor Cores.
- [Autotune Q4 Vulkan](vulkan-q4-autotune.vi.md): API tuning và biến điều khiển Q4 matmul.


## Tra cứu theo module

- `nimformers.nim`: [bắt đầu](getting-started.vi.md), [backend](backend-api.vi.md),
  [tensor](tensor-ops-api.vi.md), [model](base-model-api.vi.md),
  [lượng tử hóa](quantization-nimq.vi.md), [micro-VM](microkernel-api.vi.md).
- `backend.nim`: [backend](backend-api.vi.md), [session GPU](gpu-session-api.vi.md).
- `tensor_ops.nim`: [API tensor](tensor-ops-api.vi.md).
- `nimformer.nim`: [API model cơ bản](base-model-api.vi.md).
- `test/deepseek-inference/nim_inference.nim`:
  [inference LLaMA](llama-inference-api.vi.md), [tokenizer](tokenizer-api.vi.md).
- `test/codestral/jamba_hybrid.nim`: [Jamba lai](jamba-model-api.vi.md).
- `quant.nim` / `customfloat.nim`: [lượng tử hóa](quantization-nimq.vi.md),
  [CustomFloat/APF](customfloat-apf.vi.md).
- `test/codestral/lora.nim` / `test/codestral/infer_lora.nim`: [LoRA](lora-api.vi.md).
- `test/codestral/finetune_lora.nim`: [fine-tune và resume](finetune-lora-api.vi.md).
- `test/spectron/test_nimformer.nim` / `test/spectron/test_inferencespectron.nim`:
  [API training](training-api.vi.md).
- `test/codestral/rawbin.nim` / `test/codestral/qlorabin.nim` /
  `test/codestral/checkpoint.nim`: [định dạng checkpoint](checkpoint-formats.vi.md).
- `vendor/bybylang/`: [DSL](bybylang-api.vi.md), [ngôn ngữ](bybylang-language.vi.md),
  [session](gpu-session-api.vi.md), [micro-VM/JIT](microkernel-api.vi.md).
- `lib_c_api.nim` / `nimformers/` / `include/nimformers.h`: [Python và C ABI](python-api.vi.md).

Chữ ký hàm trong docs thuộc module công khai và facade export. Kernel nội
bộ và helper chỉ dùng test/benchmark không phải API ứng dụng ổn định.
Khi API ghi experimental hoặc phụ thuộc backend, source là căn cứ cuối cùng.


## Duy trì hai ngôn ngữ

File `name.md` là bản tiếng Anh; `name.vi.md` là bản tiếng Việt đầy đủ.
Giữ đường dẫn tiếng Anh hiện tại để bảo toàn liên kết đã có. Mỗi trang có
liên kết chuyển ngôn ngữ; liên kết trong bản Việt dẫn tới bản Việt khi có.

Khi đổi API, lệnh hoặc ví dụ, cập nhật cả hai bản trong cùng thay đổi.
Giữ chữ ký hàm và code block giống nhau; dịch phần giải thích thay vì đổi
identifier/lệnh. Anchor tiếng Anh được giữ thêm ở bản Việt để liên kết sâu
vẫn hoạt động sau dịch tiêu đề. README và API tham chiếu cũng theo quy ước này.

`vendor/bybylang` là Git submodule. Bản song ngữ của tài liệu ngôn ngữ đặt
ở `docs/bybylang-language*.md` để được theo dõi trong repo framework; README
upstream trong submodule vẫn là tài liệu nguồn.
