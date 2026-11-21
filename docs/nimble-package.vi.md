# Cài Nimformers như một thư viện Nim

[English](nimble-package.md) | [Tiếng Việt](nimble-package.vi.md)

Package Nimble cài facade Nim, các source phụ thuộc, runtime BybyLang cùng file
shader backend, và binding `nimformers_shared` để load C ABI động. Package
không chứa trọng số model, checkpoint, dataset, binary sinh ra hoặc build cache.

## Cài trực tiếp từ GitHub

Có thể cài package trực tiếp từ repository công khai này mà không cần chờ được
thêm vào package index của Nimble:

```sh
nimble install https://github.com/bobbyshop-vui/nimformers-framework
```

Nimble chấp nhận URL Git làm nguồn package. Sau khi package được đăng ký trong
index chính thức, lệnh ngắn `nimble install nimformers` cũng sẽ dùng được.

## Checkout và cài đặt

Runtime BybyLang là Git submodule. Khởi tạo submodule trước khi build hoặc cài
từ source checkout:

```sh
git clone --recurse-submodules <repository-url> nimformers-framework
cd nimformers-framework
nimble develop
nimble buildShared
```

Với checkout hiện tại chưa có nội dung submodule, chạy
`git submodule update --init --recursive` trước. `nimble buildShared` tạo
`libnimformers.dylib` (macOS), `libnimformers.so` (Linux) hoặc
`nimformers.dll` (Windows), cùng executable `nimformers-worker` bên cạnh.
File model vẫn là đầu vào runtime riêng và không thuộc package Nimble.

`nimble develop` thêm checkout hiện tại vào các project Nimble. Sau khi phát
hành và được đưa vào registry Nimble, cài trong project consumer bằng
`nimble install nimformers`. `nimformers.nim` cung cấp API Nim native; ứng dụng
chỉ cần C shared library nếu dùng `nimformers_shared`.

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
import bindings/nimformers_shared as nimformers_shared

var ctx = newContext("cpu")
defer: close(ctx)
echo add(ctx, [1'f32, 2], [10'f32, 20])
```

Đặt `NIMFORMERS_LIB` thành đường dẫn đầy đủ của shared library nếu thư viện
không nằm trong đường dẫn tìm kiếm của dynamic loader. Ví dụ hoàn chỉnh ở
[`examples/nim/shared_library.nim`](../examples/nim/shared_library.nim).
Binding shared cũng cung cấp `newProcessGroup`, `partition`, `vmRun`,
`generateBatch` và `generateIdsBatch` trên native parallel C ABI. Hai hàm batch
khởi chạy executable `formers-cli` đã cấu hình một lần cho mỗi prompt; kết quả
văn bản hoặc chuỗi token ID cách nhau bằng dấu cách được trả theo thứ tự prompt.

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
import nim "bindings/nimformers_shared"

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
export NIMFORMERS_LIB="/absolute/path/to/libnimformers.dylib"
./vendor/bybylang/bybylang examples/bybylang/shared_library.bybylang \
  --aot=/tmp/nimformers-shared-example
/tmp/nimformers-shared-example
```

Trên Linux dùng `libnimformers.so`; Windows dùng `nimformers.dll`. Khi compile
chương trình Nim sinh ra, cần để Nim thấy package Nimble đã cài, ví dụ đặt
`BYBYLANG_NIM_FLAGS="--path:/path/to/nimble/package"`. Biến `NIMFORMERS_LIB`
chọn thư viện khi chạy chương trình.

## Nội dung package và hỗ trợ nền tảng

`nimble install` chỉ cài các module Nim trong allowlist và source/asset runtime
BybyLang mà chúng cần. Source GPU kernel và shader đã compile là dependency
runtime nên được đóng gói. Checkpoint, tokenizer, model tải về, executable sinh
ra, cache và artifact benchmark đều bị loại trừ. Hãy cài và build trên hệ điều
hành đích để tạo shared library và worker executable tương ứng.
