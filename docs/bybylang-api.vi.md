
# API BybyLang GPU và DSL

[English](bybylang-api.md) | [Tiếng Việt](bybylang-api.vi.md)

BybyLang được đặt trong `vendor/bybylang`. Chương trình Nim có thể import
`vendor/bybylang/bybylang` cho compiler DSL hoặc `vendor/bybylang/gpubackend`
cho các phép GPU. Facade framework export lại một số symbol backend/session
qua `backend.nim` và `nimformers.nim`.

Hướng dẫn trình bày cách build compiler, chạy script và gọi C ABI của
framework. Gói Python gọi cùng C ABI qua thư viện dùng chung.
[Hướng dẫn BybyLang](bybylang-guide.vi.md) · [API Python](python-api.vi.md)


## Chọn backend

```nim
type GpuBackend* = enum
  gbAuto, gbCpu, gbCuda, gbMetal, gbVulkan

proc parseBackend*(s: string): GpuBackend
proc normalizeBackend*(b: GpuBackend): GpuBackend
proc detectBackend*(): GpuBackend
proc setForbidCpuFallback*(forbid: bool)
```

`parseBackend("cuda")` ánh xạ sang Vulkan. Cả `parseBackend("cpu")` lẫn
`newBackend("cpu")` của framework đều chọn CPU trực tiếp. Với `auto`, runtime
thử Metal trên macOS, rồi Vulkan, rồi CPU. Nếu phép toán có nhánh CPU dự phòng,
`setForbidCpuFallback(true)` yêu cầu báo lỗi khi chạy GPU thất bại.


## Phép GPU immediate

Các hàm nhận `backend: GpuBackend`; input/output là mảng phẳng row-major.

```nim
gpuMatmul(backend, a, b, m, k, n)
gpuMatmul2(backend, a1, b1, m1, k1, n1, a2, b2, m2, k2, n2)
gpuMatmulQ4(backend, a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
gpuOp(op, backend, a, b) # add, sub, mul, div
gpuRelu(backend, x); gpuSigmoid(backend, x); gpuTanh(backend, x)
gpuSqrt(backend, x); gpuExp(backend, x); gpuLog(backend, x); gpuPow(backend, x, p)
gpuSoftmax(backend, x, rows, cols)
gpuLayernorm(backend, x, gamma, beta, rows, cols, eps)
gpuLayernormBackward(backend, dy, x, gamma, beta, rows, cols, eps)
gpuEmbeddingLookup(backend, table, indices, vocab, dim)
gpuAttentionFused(backend, q, k, v, mask, B, H, S, D, scale)
gpuAttentionFusedBackward(backend, q, k, v, sMatrix, dy, B, H, S, D, scale)
gpuApflu(backend, x, alpha, beta); gpuApfluBackward(backend, x, dy, alpha, beta)
gpuFusedAddAct(backend, a, b, act)
gpuGather(backend, src, indices)
gpuArgmaxIndex(backend, x, outer, dimSize, inner)
gpuCompare(backend, a, b, op) # op: 0 gt, 1 lt, 2 eq, 3 ge, 4 le
gpuDropoutMask(backend, n, p, seed)
gpuPermute(backend, src, newStrides, oldStrides, dimsMap, n)
gpuSplitSlice(backend, src, dimTotal, thisSize, offset, inner, n)
gpuConcat(backend, flatSrc, srcBase, srcDimSize, srcDimOffset, numTensors, totalDim, inner)
```

Buffer Q4 resident và các phép khác ở [API backend](backend-api.vi.md) và
[API session GPU](gpu-session-api.vi.md).


## Cú pháp source BybyLang

Parser đọc `.bybylang` và sinh chương trình Nim AOT. Import được tìm tương
đối với thư mục hiện tại hoặc thư mục file import. `import nim` import
module Nim vào module AOT sinh ra.

```bybylang
gpu backend is "auto"
import nim "math"

gpu array A = [1, 2, 3, 4]
gpu array B = [10, 20, 30, 40]
gpu add A, B -> C size 4
print C
print sqrt(16.0)
```

Các lệnh phổ biến gồm `gpu array`, `gpu add|sub|mul|div`,
`gpu relu|sigmoid|tanh|apflu`, `gpu softmax`, `gpu matmul`, `gpu layernorm`,
`gpu embedding`, `gpu attention`, `gpu gather|compare|fused_add_act` và các
phép một ngôi `gpu sqrt|exp|log|pow`. Kích thước và tham số tùy chọn đặt sau
`->`; dùng `vendor/bybylang/demo/demo_gpu.bybylang` làm mẫu cú pháp chạy được.


## Block session

Gom nhiều phép vào một command buffer:

```bybylang
session:
  gpu add A, B -> C size 4
  gpu relu C -> D
```

Mức hỗ trợ phụ thuộc backend. Compiler dùng lệnh immediate khi thiếu
encoder session và có nhánh thay thế; xem [API session GPU](gpu-session-api.vi.md).


## Chạy lại command buffer

Dùng API session của Nim cho các bước có kích thước cố định và buffer được
giữ qua nhiều lần chạy. Replay gửi lại cùng lệnh với các handle đã ghi, vì
vậy có thể ghi dữ liệu đầu vào mới từ host giữa hai bước.

1. Kiểm tra `sessionCanRecord(backend)`. Cấp phát trước input, output và
   scratch trước khi gọi `sessionBeginRecording(backend)`.
2. Encode bước rồi gọi `sessionEnd(backend)`. Kiểm tra kết quả và ID từ
   `sessionLastRecording(backend)`. ID `-1` sau một `sessionEnd` thành công
   nghĩa là bước đã chạy nhưng runtime không giữ lệnh để replay. Upload qua
   staging, giải phóng trì hoãn và scratch nội bộ có thể dẫn tới trường hợp này.
3. Dùng `sessionHostBuffer` và `sessionHostWrite` cho input thay đổi. Để đọc
   kết quả, encode copy vào buffer output ánh xạ với host rồi gọi
   `sessionHostReadBarrier(backend)` trước khi kết thúc session. Gọi
   `sessionReplay(backend, id)` ngoài session đang mở và kiểm tra kết quả
   trước khi đọc bằng `sessionHostRead`.
4. Gọi `sessionFreeRecording(backend, id)` trước khi giải phóng các buffer
   mà recording tham chiếu.

Vulkan giữ command buffer để gửi lại cùng các descriptor pool liên quan.
Metal dùng indirect command buffer hoặc phát lại các dispatch trên encoder
thường. `sessionRecordingNote(backend, id)` báo nhánh Metal đã chọn hoặc
lý do không giữ recording. `BYBY_METAL_REPLAY_ICB=0` chọn encoder thường.
[Test replay](../vendor/bybylang/tests/t_metal_session_replay.nim)

## Phép prefill trong session

- `sessionFp4GemmTensorCore` dùng cooperative matrix trên Vulkan cho trọng
  số FP4-KM đã pack. `sessionFp4GemmTiledEnc` dùng tile thanh ghi trên Vulkan.
  `sessionFp4GemmMetal` dùng tile thanh ghi hoặc Apple `simdgroup_matrix` khi
  được yêu cầu và phần cứng hỗ trợ. Cấp phát handle output của Metal trước
  khi encode.
- `sessionF16GemmTile(backend, a, w, c, M, K, N, tensorCore)` nhận trọng số
  IEEE FP16 theo thứ tự `[N, K]` và tính `c = a @ w.T` trên Vulkan. Khi
  `tensorCore` là true, kernel dùng cooperative matrix. Nhánh còn lại dùng
  tile thanh ghi. Gọi `sessionTileGemmReady` ngoài session trước lần dùng đầu
  tiên để kiểm tra kết quả của nhánh FP4-KM hoặc FP16. Khi encoder trả về
  `false`, bên gọi chọn phép tính thay thế.
- `sessionMamba2SsdPrepare(backend, N, D, Q)` chuẩn bị SSD prefill theo chunk
  trên Vulkan ngoài session. Giá trị trả về là `"coop"`, `"scalar"` hoặc `""`
  nếu không có kernel phù hợp. `sessionMamba2SsdPrefillEnc` encode năm dispatch
  trên arena `float32` được cấp sẵn, gồm cả việc truyền state giữa các chunk.
- `sessionCompileSeconds()` cộng dồn thời gian biên dịch shader và tạo
  pipeline khi cache miss. Lấy hiệu trước và sau prefill để ghi riêng thời
  gian chuẩn bị kernel.

`buildFp4TileGemmProgram` và `buildF16TileGemmProgram` tạo chương trình VM.
`tile_gemm.nim` lower chương trình đó sang mã cho GPU đã chọn.
[Chữ ký session](../vendor/bybylang/gpubackend.nim) ·
[Lowering tile GEMM](../vendor/bybylang/micro_kernel/tile_gemm.nim)


## Block micro-kernel native

Có thể khai báo bytecode micro-kernel ngay trong `.bybylang`:

```bybylang
micro kernel double A -> OUT threads 4 tg 4 out 4:
  emitULid 0
  emitULoadIn 1 0 0
  emitLoadConstF32 2 2.0'f32
  emitMul 1 1 2
  emitUStoreF32 1 0 0
  emitHalt
```

`A` là mảng float đầu vào. `threads` đặt kích thước dispatch, `tg` đặt kích
thước threadgroup, `out` đặt số phần tử output. `emitULid` lấy chỉ số trong
group duy nhất của ví dụ. Khi dùng nhiều group, tính thêm offset của group.
Compiler tạo bytecode `VmProgram` để lower cho GPU hoặc chạy bằng CPU VM.
[API micro-VM/JIT](microkernel-api.vi.md)


## Build AOT và chạy

```sh
# From the Nimformers repository root:
make -C vendor/bybylang build
./vendor/bybylang/bybylang vendor/bybylang/demo/demo_gpu.bybylang \
  --aot=vendor/bybylang/demo/demo_gpu_out
./vendor/bybylang/demo/demo_gpu_out
```

`--aot` ghi và compile file Nim; cần chạy binary sinh ra riêng.
`BYBYLANG_NIMCACHE` chọn thư mục cache. Ví dụ import `nimformers_lib` cần search
path framework; thư mục test cung cấp chúng qua
`test/bybylang_test/nim.cfg`. `import nim` module `nimformers_lib` compile
implementation ABI vào binary, không load động shared library Python.

Trên macOS, thêm `caffeinate -i` trước lệnh cần giữ máy thức khi build/chạy.
Tiến trình GPU phải truy cập được thiết bị; môi trường sandbox/headless
có thể không thấy Metal.

## Trace

`trace.nim` ghi lại một lần chạy cho bộ lift FIR. Script BybyLang import nó như mọi module Nim và gọi
đúng các hàm như code Nim:

```bybylang
import nim "../../trace"

traceStart("run.fprobe", "decode:2", true, 0)   # path, steps, data, dataMax
# ... dựng và chạy model ...
traceStop()
```

Bắt đầu trace trước các phép cần ghi. Bytecode đã đăng ký được giữ qua các
lần trace, nên có thể compile pipeline từ trước. `traceBegin` và `traceEnd`
đánh dấu bước. Có thể dùng `traceActive`, `traceFile`, `withTrace` và `traceStep`.

Với Codestral Vulkan, dùng `firlift --run` cùng CLI native để ghi qua pipe
trong RAM. Graph nạp checkpoint và LoRA gốc khi chạy. Lift xuất `model.fir`
và `config.json`. Mốc `codestral_prompt` giới hạn các phép có thể đưa vào
replay. [API trace](trace-api.vi.md) · [Lift Codestral FIR](fir-script-lift.md#codestral-on-vulkan)
