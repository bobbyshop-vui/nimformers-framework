# Tham khảo test tích hợp BybyLang

[English](bybylang-tests.md) | [Tiếng Việt](bybylang-tests.vi.md)

Trang này giải thích cách chạy, đọc và mở rộng hai chương trình `.bybylang`
trong `test/bybylang_test`. Nội dung chỉ mô tả những API hai test đang dùng;
xem [API BybyLang](bybylang-api.vi.md) và
[tham khảo ngôn ngữ](bybylang-language.vi.md) để biết toàn bộ bề mặt ngôn ngữ.


## Các file test

- [`native_gpu_test.bybylang`](../test/bybylang_test/native_gpu_test.bybylang)
  dùng lệnh native `gpu ...` và block `micro kernel ...:`. File kiểm tra cộng
  vector, nhân ma trận, softmax, ReLU và dispatch micro-VM.
- [`capi_test.bybylang`](../test/bybylang_test/capi_test.bybylang) import
  `../../lib_c_api` như một module Nim. File kiểm tra ABI version, backend
  context, phép cộng, matmul, softmax, LayerNorm, lượng tử INT8, dispatch
  micro-VM và fused attention.
- [`run.sh`](../test/bybylang_test/run.sh) compile AOT rồi chạy cả hai file.
- [`nim.cfg`](../test/bybylang_test/nim.cfg) cung cấp search path trong repo cho
  mã Nim sinh ra khi import `lib_c_api.nim`.

Các file sinh ra `native_gpu_test_bin`, `capi_test_bin` và source `.nim` của
chúng là build artifact. Git bỏ qua các file này và có thể sinh lại bất cứ lúc nào.


## Yêu cầu và cách chạy nhanh

Chạy từ thư mục gốc repo:

```sh
make bybylang-test
```

Test cần Nim, compiler BybyLang vendored và GPU Metal hoặc Vulkan. Backend
`auto` thử Metal trước trên macOS, sau đó thử Vulkan. Tên cũ `cuda` là alias
của Vulkan. Parser có nhận `cpu`, nhưng native micro-kernel bắt buộc Metal hoặc
Vulkan, nên toàn bộ suite không phải là test CPU.

Runner chỉ build `vendor/bybylang/bybylang` khi chưa có compiler executable.
Sau khi sửa source compiler, cần build lại rõ ràng:

```sh
make -C vendor/bybylang build
make bybylang-test
```

Một lần chạy thành công kết thúc bằng:

```text
=== native BybyLang GPU test PASSED ===
=== BybyLang C ABI test PASSED ===
[bybylang-test] all tests passed
```


## Compile và chạy riêng từng file

Với test native, runner tương đương các lệnh sau:

```sh
mkdir -p nimcache/bybylang-test
BYBYLANG_NIMCACHE="$PWD/nimcache/bybylang-test" \
  ./vendor/bybylang/bybylang \
  test/bybylang_test/native_gpu_test.bybylang \
  --aot=test/bybylang_test/native_gpu_test_bin
./test/bybylang_test/native_gpu_test_bin
```

Với test C ABI:

```sh
BYBYLANG_NIMCACHE="$PWD/nimcache/bybylang-test" \
  ./vendor/bybylang/bybylang \
  test/bybylang_test/capi_test.bybylang \
  --aot=test/bybylang_test/capi_test_bin
./test/bybylang_test/capi_test_bin
```

`--aot=PATH` sinh `PATH.nim`, gọi `nim c -d:release` rồi tạo native executable
tại `PATH`. Bước compile không chạy test; lệnh cuối mới thực thi binary.
`BYBYLANG_NIMCACHE` điều khiển thư mục cache của Nim, không đổi vị trí source
sinh ra hay executable.

Đường dẫn output rất quan trọng với `capi_test.bybylang`: giữ output trong
`test/bybylang_test` giúp Nim tìm `nim.cfg` của thư mục đó, còn import
`../../lib_c_api` trỏ ngược về thư mục gốc repo.


## Cách lệnh native GPU DSL được chuyển đổi

Phần chính của `native_gpu_test.bybylang` là:

```bybylang
gpu backend is "auto"
gpu array X = [1, 2, 3, 4]
gpu array Y = [10, 20, 30, 40]
gpu array P = [1, 2, 3, 4, 5, 6]
gpu array Q = [1, 0, 0, 1, 1, 1]

gpu add X, Y -> Z size 4
gpu matmul P, Q -> R m 2 k 3 n 2
gpu softmax X -> SM rows 1 cols 4
gpu relu X -> RELU
```

Compiler sinh các lời gọi trực tiếp vào GPU runtime vendored:

- `gpu backend is "auto"` đặt `gpuBackendSelected = parseBackend("auto")`.
  Các tên hợp lệ là `auto`, `metal`, `vulkan`, `cuda` và `cpu`.
- `gpu array NAME = [...]` tạo một `seq[float32]` phẳng. Array không mang shape.
- `gpu add X, Y -> Z size 4` gọi `gpuOp("add", ...)`. Giá trị `size` hiện chỉ
  mang tính mô tả trong generator; nó không được truyền vào `gpuOp`.
- `gpu matmul P, Q -> R m 2 k 3 n 2` coi `P` là row-major `[2,3]`, `Q` là
  row-major `[3,2]` và trả về kết quả phẳng `[2,2]`.
- `gpu softmax X -> SM rows 1 cols 4` chuẩn hóa từng row được chỉ định. Input
  phải chứa ít nhất `rows * cols` giá trị.
- `gpu relu X -> RELU` gọi `gpuRelu`; do input của test đều dương nên `RELU`
  bằng `X`.

Khi source ghi `auto`, có thể dùng `BYBYLANG_BACKEND` để override trong lúc
compiler đang sinh mã Nim:

```sh
BYBYLANG_BACKEND=vulkan \
  ./vendor/bybylang/bybylang \
  test/bybylang_test/native_gpu_test.bybylang \
  --aot=test/bybylang_test/native_gpu_test_bin
```

Override được ghi vào chương trình đã sinh. Chỉ đặt biến môi trường khi chạy
một binary đã build sẵn sẽ không viết lại chương trình đó.


## API native `micro kernel` mà test sử dụng

Test nhân đôi bốn giá trị input:

```bybylang
micro kernel byby_double X -> DOUBLED threads 4 tg 4 out 4:
  emitULid 0
  emitULoadIn 1 0 0
  emitAdd 1 1 1
  emitUStoreF32 1 0 0
  emitHalt
```

Header có dạng:

`micro kernel NAME INPUT -> OUTPUT threads N tg N out N:`

- `threads` là tổng số thread được dispatch.
- `tg` là số thread trong mỗi threadgroup.
- `out` là số phần tử `float32` đọc ngược về `OUTPUT`.
- Phần body thụt lề nhận các procedure `emit...` được export từ
  `vendor/bybylang/micro_kernel/session_vm.nim`.

Năm instruction được dùng ở đây:

- `emitULid 0`: ghi local thread ID vào unsigned register `U0`.
- `emitULoadIn 1 0 0`: load `input[U0 + 0]` vào float register `R1`.
- `emitAdd 1 1 1`: tính `R1 = R1 + R1`.
- `emitUStoreF32 1 0 0`: store `R1` vào `output[U0 + 0]`.
- `emitHalt`: dừng chương trình của thread hiện tại.

Ví dụ cố ý dispatch đúng một group (`threads == tg == 4`), nên local ID cũng
là global index. Với nhiều threadgroup, cần tính global index từ `emitUTgid`,
kích thước group và `emitULid`; nếu không, mỗi group sẽ ghi vào cùng các vị trí
output. ID register bị giới hạn trong 16 float register `R0..R15` và 16
unsigned register `U0..U15` của micro-VM.


## Vì sao `import nim` gọi được C ABI

Test C ABI bắt đầu bằng:

```bybylang
import nim "../../lib_c_api"
```

BybyLang đặt lệnh này thành một `import` Nim thật ở đầu source sinh ra. Nó
compile `lib_c_api.nim` vào executable AOT; không load động
`libnimformers.dylib`. Sau đó test dùng trực tiếp các biểu thức Nim như `@[]`,
`newSeq`, `addr`, `nil`, `discard` và hậu tố số như `1'f32`. Những dòng không
phải lệnh BybyLang đặc biệt phần lớn được phát ra như mã Nim.


## API runtime và lỗi

Test dùng các lời gọi sau:

- `nimf_abi_version(): uint32` trả về ABI version dạng số.
- `nimf_context_create(backendKind): pointer` tạo context. Các giá trị là
  `0=auto`, `1=Metal`, `2=Vulkan`, `3=CUDA alias Vulkan`. Hàm trả `nil` khi lỗi.
- `nimf_context_backend_name(ctx): cstring` trả về `metal` hoặc `vulkan` cho
  context hợp lệ.
- `nimf_last_error(): cstring` trả lỗi cuối của thread hiện tại.
- `nimf_context_destroy(ctx): cint` giải phóng context; chấp nhận handle `nil`.

Hàm trả `cint` dùng `0` cho thành công và khác `0` cho thất bại. Constructor
trả pointer sẽ trả `nil` khi lỗi. `nimf_quant_dequantize` là ngoại lệ: hàm trả
số giá trị đã ghi hoặc `-1` khi lỗi. Đọc `nimf_last_error()` ngay sau phép gọi lỗi.


## C ABI tensor mà test sử dụng

Mọi buffer tensor dưới đây đều là chuỗi `float32` phẳng, liên tục:

- `nimf_add(ctx, a, b, out, n)` cộng từng phần tử của `n` giá trị. Test truyền
  địa chỉ bằng `addr A[0]` và cấp trước `out` bằng `newSeq`.
- `nimf_matmul(ctx, a, b, out, m, k, n)` tính row-major
  `A[m,k] @ B[k,n] -> out[m,n]`. Test dùng `(m,k,n) = (2,3,2)`.
- `nimf_softmax(ctx, x, out, rows, cols)` coi `x` là `[rows,cols]` và chuẩn hóa
  từng row. Test kiểm tra tổng của một row xấp xỉ một.
- `nimf_layernorm(ctx, x, gamma, beta, out, rows, cols, eps)` yêu cầu độ dài
  `x/out` là `rows*cols`, `gamma/beta` là `cols`. Test dùng gamma bằng một,
  beta bằng không rồi kiểm tra mean output gần bằng không.

Các procedure này copy sequence host lên backend được chọn và copy kết quả về
sequence output mà caller đã cấp phát.


## C ABI lượng tử hóa mà test sử dụng

Lời gọi
`nimf_quantize(x, n, shape, rank, kind, exponentBits, mantissaBits)` trả một
handle tensor lượng tử có ownership. Test truyền shape `[4]`, rank `1` và kind
`1`, tức `QK_INT8`. Tham số exponent và mantissa bằng `0` vì chúng chỉ cấu hình
`QK_CUSTOM`.

`numel(shape)` phải bằng `n`. Test gọi
`nimf_quant_dequantize(handle, out, capacity)` và mong giá trị trả về là `4`.
Cuối cùng, `nimf_quant_destroy(handle)` giải phóng tensor lượng tử. Mọi lần
quantize thành công phải có một lần destroy tương ứng.


## C ABI micro-VM mà test sử dụng

Test C ABI tạo phép nhân đôi tương tự bằng opcode dạng số:

```bybylang
prog = nimf_vm_create()
discard nimf_vm_emit(prog, 0x10'u32, 1, 0, 0, 0)
discard nimf_vm_emit(prog, 0x20'u32, 1, 1, 1, 0)
discard nimf_vm_emit(prog, 0x11'u32, 1, 0, 0, 0)
discard nimf_vm_emit(prog, 0x01'u32, 0, 0, 0, 0)

VMEM = @[1'f32, 2, 3, 4]
discard nimf_vm_run(ctx, prog, addr VMEM[0], 4, 4, 0, 4)
discard nimf_vm_destroy(prog)
```

- `nimf_vm_create()` cấp phát chương trình rỗng.
- `nimf_vm_emit(program, opcode, a, b, c, value)` nối thêm một instruction có
  kiểu. Ở đây `0x10` load `memory[R0]` vào `R1`, `0x20` cộng `R1 + R1`, `0x11`
  store `R1` vào `memory[R0]`, còn `0x01` dừng.
- GPU kernel sinh ra khởi tạo float register `R0` bằng global thread ID, nên
  bốn thread xử lý bốn phần tử.
- `nimf_vm_run(ctx, program, memory, memoryLen, nThreads, tgFloats, tgSize)` sửa
  `memory` tại chỗ. Test dùng bốn phần tử, bốn thread, không yêu cầu vùng nhớ
  scratch threadgroup dạng float và group size bằng bốn.
- `nimf_vm_destroy(program)` giải phóng chương trình.

Trong code ứng dụng nên kiểm tra status của mọi emit/run. Test bỏ qua status
emit vì dùng constant đã biết hợp lệ và có kiểm tra riêng kết quả cuối.


## C ABI fused attention mà test sử dụng

`nimf_attention` dùng layout logic sau:

- `q`, `k`, `v` và `out`: phẳng `[batch, heads, sequenceLen, headDim]`.
- `scores`: phẳng `[batch, heads, sequenceLen, sequenceLen]`.
- `mask` cùng `maskLen`: mask phẳng tùy chọn. `nil, 0` nghĩa là không có mask.
- `scale`: hệ số nhân score, thường là `1 / sqrt(headDim)`.

Test dùng `(batch, heads, sequenceLen, headDim) = (1,1,4,4)`, nên Q/K/V và
output đều có 16 float, còn scores có 16 float. Test truyền scale `0.5`, sau đó
kiểm tra output đầu tiên không phải NaN. Đây là smoke test cho dispatch và shape,
không phải kiểm thử tham chiếu số học cho mọi giá trị.


## Quy tắc ownership

Test tạo ba handle có ownership và giải phóng từng handle bằng đúng API:

- context: `nimf_context_create` -> `nimf_context_destroy`
- tensor lượng tử: `nimf_quantize` -> `nimf_quant_destroy`
- chương trình VM: `nimf_vm_create` -> `nimf_vm_destroy`

Sequence tạo bằng `newSeq` do Nim quản lý; không giải phóng chúng qua C ABI.
Giữ input/output sequence còn sống khi pointer `addr seq[0]` đang được dùng,
và không bao giờ lấy phần tử số không của sequence rỗng.


## Thêm một test `.bybylang`

1. Đặt source trong `test/bybylang_test` để kế thừa `nim.cfg`.
2. Khi lỗi, gọi `quit(1)`; `run.sh` dùng `set -e` và dừng khi exit status khác 0.
3. Trước tiên compile thủ công với tên output `--aot` riêng.
4. Thêm basename vào vòng lặp trong `run.sh` nếu muốn test chạy cùng
   `make bybylang-test`.
5. Tự build lại compiler nếu đã sửa source trong `vendor/bybylang`.

Tránh dấu `=` trong comment cuối dòng Nim thô. Parser hiện tại có thể coi mọi
dòng không phải command nhưng chứa `=` là assignment; hãy đặt comment đó trên
một dòng riêng.


## Tài liệu liên quan

- [Hướng dẫn BybyLang](bybylang-guide.vi.md)
- [API GPU và DSL BybyLang](bybylang-api.vi.md)
- [Tham khảo ngôn ngữ BybyLang](bybylang-language.vi.md)
- [Python và C ABI](python-api.vi.md)
- [API micro-VM/JIT](microkernel-api.vi.md)

