# Cài Nimformers như một thư viện Nim

[English](nimble-package.md) | [Tiếng Việt](nimble-package.vi.md)

Package Nimble cài facade Nim cùng source phụ thuộc, runtime BybyLang và shader
backend, kèm shared library C ABI và worker được build cho máy đích. Package
không chứa trọng số model, checkpoint, dataset hoặc build cache.

## Cài trực tiếp từ GitHub

Có thể cài package trực tiếp từ repository công khai này mà không cần chờ được
thêm vào package index của Nimble:

```sh
nimble install https://github.com/bobbyshop-vui/nimformers-framework
```

Nimble chấp nhận URL Git làm nguồn package. Sau khi package được đăng ký trong
index chính thức, lệnh ngắn `nimble install nimformers` cũng sẽ dùng được.

Trong lúc cài, Nimble build shared library và worker cho máy hiện tại rồi đặt
cả hai trong thư mục `lib/` của package. Luồng cài không cần lệnh build riêng.

## Cài từ checkout

Nếu đã clone repository, cài toàn bộ bằng một lệnh Nimble. Hook cài đặt tự build
shared library và worker cho nền tảng hiện tại trước khi Nimble chép package:

```sh
git clone --recurse-submodules <repository-url> nimformers-framework
cd nimformers-framework
nimble install
```

Lệnh cài trực tiếp từ GitHub ở phần trước cũng cài từ đầu đến cuối như vậy.
Shared library được cài tại `lib/libnimformers.dylib` (macOS),
`lib/libnimformers.so` (Linux) hoặc `lib/nimformers.dll` (Windows), cùng
`lib/nimformers-worker`. File model vẫn là đầu vào runtime riêng. `nimformers.nim`
cung cấp API Nim ở tầng source; dùng `nimformers_shared` để gọi C shared library
đã cài mà không compile phần implementation của framework vào ứng dụng.

## Dùng facade Nim hoặc binding động

API Nim liên kết source:

```nim
import nimformers

let group = initProcessGroup(["cpu"])
echo group.partition(12)
```

API process group nằm trong `nimformers.nim`; xem [tài liệu thực thi phân tán
trên một máy](distributed.vi.md) về thiết bị và worker.
Ví dụ tối thiểu có thể chạy ở
[`examples/nim/native_process_group.nim`](../examples/nim/native_process_group.nim).

`nimformers_shared` dùng cùng thư viện C ABI v2 từ Nim mà không liên kết phần
implementation vào ứng dụng:

```nim
import nimformers_shared

var ctx = newContext("cpu")
defer: close(ctx)
echo "shared library: ", findLibrary()
echo add(ctx, [1'f32, 2], [10'f32, 20])
```

`findLibrary()` trả về đường dẫn đầy đủ trong package đã cài. Đặt
`NIMFORMERS_LIB` để ghi đè việc tự tìm. Wrapper cũng tự tìm worker chạy micro-VM
cạnh shared library. Ví dụ hoàn chỉnh ở
[`examples/nim/shared_library.nim`](../examples/nim/shared_library.nim).
Binding shared cũng cung cấp `newProcessGroup`, `partition`, `vmRun`,
`generateBatch` và `generateIdsBatch` trên native parallel C ABI. Hai hàm batch
khởi chạy executable `formers-cli` đã cấu hình một lần cho mỗi prompt; kết quả
văn bản hoặc chuỗi token ID cách nhau bằng dấu cách được trả theo thứ tự prompt.
Mọi API `forward` model cấp cao cũng hỗ trợ hook tùy chỉnh trước/sau; xem
[tài liệu forward hook](forward-hooks.md).

```nim
var group = newProcessGroup(["cpu", "vulkan:0"])
defer: close(group)
let outputs = generateBatch(group, "/path/to/formers-cli",
  ["--models", "/models"], ["Hello", "Write a short poem"], maxTokens = 16)
```

## Dùng binding từ BybyLang

BybyLang import facade Nim vào chương trình AOT sinh ra:

```bybylang
import nim "nimformers"

let group = initProcessGroup(["cpu"])
echo group.partition(12)
```

BybyLang cũng có thể load binding shared library qua `nimformers_shared`:

```bybylang
import nim "nimformers_shared"

ctx = nimformers_shared.newContext("cpu")
group = nimformers_shared.newProcessGroup(@["cpu"])
print nimformers_shared.backendName(ctx)
print nimformers_shared.add(ctx, @[1'f32, 2, 3], @[10'f32, 20, 30])
print nimformers_shared.partition(group, 12)
nimformers_shared.close(group)
nimformers_shared.close(ctx)
```

Build compiler BybyLang từ submodule đã khởi tạo, rồi compile và chạy script:

```sh
make -C vendor/bybylang build
./vendor/bybylang/bybylang examples/bybylang/shared_library.bybylang \
  --aot=/tmp/nimformers-shared-example
/tmp/nimformers-shared-example
```

Trên Linux dùng `libnimformers.so`; Windows dùng `nimformers.dll`. Khi compile
chương trình Nim sinh ra, cần để Nim thấy package Nimble đã cài, ví dụ đặt
`BYBYLANG_NIM_FLAGS="--path:/path/to/nimble/package"`.
`NIMFORMERS_LIB` ghi đè việc tự tìm thư viện lúc chạy.

## Nội dung package và hỗ trợ nền tảng

`nimble install` cài các module Nim trong allowlist, source/asset runtime
BybyLang, cùng shared library và worker được build cho hệ điều hành đích.
Source GPU kernel và shader đã compile là dependency runtime nên được đóng gói.
Checkpoint, tokenizer, model tải về, build cache và artifact benchmark đều bị
loại trừ.
