
# Bắt đầu sử dụng

[English](getting-started.md) | [Tiếng Việt](getting-started.vi.md)


## Yêu cầu

- Nim compiler; project được viết bằng Nim.
- GPU runtime phù hợp: Metal trên macOS hoặc Vulkan với loader/ICD hoạt
  động. Một số phép toán và session encoder phụ thuộc backend.
- Python/Transformers chỉ cần cho các nhánh chuyển config/tokenizer/model
  Hugging Face có gọi hệ sinh thái Python.


## Import facade

`nimformers.nim` export lại API model cơ bản, backend, lượng tử hóa,
CustomFloat, phép toán tensor, session BybyLang và micro-VM.

```nim
import nimformers

proc main() =
  let ctx = newBackend("auto")
  defer: closeBackend(ctx)

  let a = @[1'f32, 2, 3, 4]
  let b = @[10'f32, 20, 30, 40]
  echo beAdd(ctx, a, b)

main()
```

Lưu thành `your_app.nim` ở thư mục gốc rồi build/chạy với các search path
module giống Makefile:

```sh
nim c -r -d:release --mm:arc --opt:speed \
  --path:. --path:test/codestral --path:test/deepseek-inference \
  --path:test/spectron --path:kernelfortest your_app.nim
```

Module backend đã cung cấp chỉ thị linker theo nền tảng. Với chương trình
có sẵn, dùng [hướng dẫn lệnh Make](make-commands.vi.md). `make build` tạo trainer
Spectron tại `test/spectron/test_nimformer`; `make run` còn bắt đầu train.
Chỉ chạy `make` cũng bắt đầu training.


## Tên backend

```nim
let ctx = newBackend("metal")  # macOS Metal
let ctx = newBackend("vulkan") # Vulkan
let ctx = newBackend("auto")   # probe available GPU backends
```

Tên cũ `"cuda"` được chuyển sang Vulkan trong framework, không bật CUDA
execution. `newBackend("cpu")` không được hỗ trợ. Nếu không có GPU,
`"auto"` báo lỗi thay vì âm thầm chạy framework trên CPU.

Một số phép BybyLang cấp thấp còn nhánh CPU tham chiếu/fallback; điều đó
không biến CPU thành lựa chọn `Backend` cho chạy model.


## Đọc tiếp

- Xem [Spectron](spectron-guide.vi.md), [Python](python-guide.vi.md) hoặc
  [BybyLang](bybylang-guide.vi.md) để có ví dụ đầy đủ.
- [API backend](backend-api.vi.md) cho kernel GPU.
- [API model cơ bản](base-model-api.vi.md) cho layer phục vụ training.
- [API inference LLaMA](llama-inference-api.vi.md) cho decoder kiểu Hugging Face.
- [API BybyLang GPU và DSL](bybylang-api.vi.md) cho chương trình GPU/DSL.
