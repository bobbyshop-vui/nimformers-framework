
# Dùng BybyLang với Nimformers

[English](bybylang-guide.md) | [Tiếng Việt](bybylang-guide.vi.md)

Có ba cách dùng runtime trong repo:

- Viết `.bybylang` với native `gpu ...` DSL.
- Viết `.bybylang` import module Nim, rồi gọi API framework hoặc C ABI.
- Từ Nim import `vendor/bybylang/gpubackend` để gọi các hàm `gpu*` trực tiếp.

Các cách này dùng runtime vendored trong `vendor/bybylang/`.
Xem [BybyLang API](bybylang-api.vi.md) cho signatures,
[GPU sessions](gpu-session-api.vi.md) và [micro-VM](microkernel-api.vi.md) cho API sâu hơn.


## Build compiler và chạy ví dụ có sẵn

Từ thư mục gốc repo:

```sh
make -C vendor/bybylang build
mkdir -p nimcache/bybylang-demo
BYBYLANG_NIMCACHE="$PWD/nimcache/bybylang-demo" \
  ./vendor/bybylang/bybylang test/bybylang_test/native_gpu_test.bybylang \
  --aot=test/bybylang_test/native_gpu_test_bin
./test/bybylang_test/native_gpu_test_bin
```

`--aot` tạo executable và file Nim sinh ra cạnh executable. Compile AOT
chưa chạy các phép tính trong binary; dòng cuối mới chạy chúng. Biến
`BYBYLANG_NIMCACHE` chọn thư mục cache compiler, không phải đường dẫn output.
Kiểm tra tùy chọn CLI bằng `./vendor/bybylang/bybylang --help`.

Cách chạy cả hai test native DSL và C ABI:

```sh
make bybylang-test
```

Runner tự build compiler nếu chưa có binary executable. Khi sửa source
compiler, chạy lại `make -C vendor/bybylang build` vì runner không tự kiểm
tra compiler đã cũ hay chưa. Kết quả mong đợi có `PASS native add`,
`PASS native micro kernel`, `PASS matmul`, rồi `[bybylang-test] all tests passed`.


## Viết script GPU đầu tiên

Lưu thành `test/bybylang_test/hello.bybylang`:

```bybylang
gpu backend is "auto"

gpu array A = [1, 2, 3, 4]
gpu array B = [10, 20, 30, 40]
gpu add A, B -> C size 4
print C

gpu array X = [1, 2, 3, 4, 5, 6]
gpu array W = [1, 0, 0, 1, 1, 1]
gpu matmul X, W -> Y m 2 k 3 n 2
print Y
```

Build và chạy từ thư mục gốc:

```sh
./vendor/bybylang/bybylang test/bybylang_test/hello.bybylang \
  --aot=test/bybylang_test/hello_bin
./test/bybylang_test/hello_bin
```

Kết quả cộng là `[11,22,33,44]`; matmul là `[4,5,10,11]`, biểu diễn ma trận
`[[4,5],[10,11]]`. Các array phẳng theo row-major; `m=2,k=3,n=2` mô tả
`[2,3] @ [3,2]`. DSL không tự suy ra shape từ sáu phần tử input.
Đổi dòng backend thành `gpu backend is "metal"` hoặc `"vulkan"` để chọn rõ.


## Session và micro-kernel

Có thể gom các phép toán được backend hỗ trợ vào session:

```bybylang
gpu backend is "auto"
gpu array A = [-1, 2, -3, 4]
gpu array B = [0, 1, 0, 1]
session:
  gpu add A, B -> C size 4
  gpu relu C -> D
print D
```

Kết quả là `[0,3,0,5]`. Các encoder session tùy backend; xem
[GPU Session API](gpu-session-api.vi.md) về các nhánh immediate/fallback.

Ví dụ micro-kernel độc lập nhân đôi bốn phần tử:

```bybylang
gpu backend is "auto"
gpu array X = [1, 2, 3, 4]
micro kernel double X -> OUT threads 4 tg 4 out 4:
  emitULid 0
  emitULoadIn 1 0 0
  emitAdd 1 1 1
  emitUStoreF32 1 0 0
  emitHalt
print OUT
```

`threads 4` và `tg 4` tạo đúng một threadgroup. `emitULid 0` lấy local ID
vào U0, nên ví dụ chỉ đánh địa chỉ đúng cho một group; khi dùng nhiều group
cần tính global index từ group ID và local ID. Hai số 0 cuối các lệnh
load/store ở đây là address register và offset, không được bỏ thiếu.


## Gọi C ABI từ `.bybylang`

Lưu đoạn sau trong `test/bybylang_test/abi_hello.bybylang` để dùng các search
path của `test/bybylang_test/nim.cfg`:

```bybylang
import nim "../../lib_c_api"

ctx = nimf_context_create(0)
if ctx == nil:
  print $nimf_last_error()
  quit(1)

A = @[1'f32, 2, 3, 4]
B = @[10'f32, 20, 30, 40]
C = newSeq[float32](4)
if nimf_add(ctx, addr A[0], addr B[0], addr C[0], 4) != 0:
  print $nimf_last_error()
  discard nimf_context_destroy(ctx)
  quit(1)
print C
discard nimf_context_destroy(ctx)
```

```sh
./vendor/bybylang/bybylang test/bybylang_test/abi_hello.bybylang \
  --aot=test/bybylang_test/abi_hello_bin
./test/bybylang_test/abi_hello_bin
```

Ở đây `import nim` compile module `lib_c_api.nim` vào chương trình AOT; nó
không load `libnimformers.dylib` bằng dynamic FFI như wrapper Python.
Các symbol/signatures dùng chung C ABI, xem [public header](../include/nimformers.h).
`nimf_context_create(0)` chọn auto; status `0` là thành công; các handle cần
hàm destroy tương ứng. Lỗi chi tiết lấy từ `nimf_last_error()`.

Đường dẫn import được giải quyết theo compiler và vị trí file Nim được
sinh ra. Các ví dụ để source và output cạnh nhau trong `test/bybylang_test/`
để giữ đúng relative import cùng `nim.cfg`. Nếu chuyển script sang thư mục
khác, cập nhật import/search paths.


## Gọi GPU API từ Nim

Lưu thành `byby_demo.nim` ở thư mục gốc:

```nim
import vendor/bybylang/gpubackend

setForbidCpuFallback(true)
let be = detectBackend()
let a = @[1'f32, 2, 3, 4, 5, 6]
let b = @[1'f32, 0, 0, 1, 1, 1]
echo gpuMatmul(be, a, b, 2, 3, 2)
```

```sh
nim c -r -d:release --path:. byby_demo.nim
```

Khi viết model bằng framework, dùng `Backend` và các hàm `be*` trong
`backend.nim`. Khi gọi BybyLang trực tiếp, dùng `GpuBackend` và các hàm
`gpu*`; hai type không thay thế trực tiếp cho nhau. API BybyLang cấp thấp
còn một số nhánh CPU tham chiếu/fallback; `setForbidCpuFallback(true)` yêu
cầu báo lỗi khi phép GPU phải fallback.


## Lỗi thường gặp

- Không tìm thấy `lib_c_api`, `fp4_gemv_gpu` hoặc module model khi AOT:
  kiểm tra import tương đối và các `--path` trong `nim.cfg`.
- Sửa compiler mà script vẫn dùng hành vi cũ: build lại compiler vendored.
- Không truy cập Metal/Vulkan: thử binary từ terminal có quyền dùng GPU;
  kiểm tra driver/runtime trước khi đổi code DSL.
- Parser hiểu nhầm dấu `=` trong comment cuối dòng Nim: đặt comment trên
  một dòng riêng. Parser có nhánh nhận assignment dựa trên dấu `=`.

Các script kiểm thử đầy đủ nằm trong [test/bybylang_test](../test/bybylang_test/README.vi.md).
