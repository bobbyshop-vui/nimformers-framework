
# API BybyLang GPU và DSL

[English](bybylang-api.md) | [Tiếng Việt](bybylang-api.vi.md)

BybyLang được đặt trong `vendor/bybylang`. Chương trình Nim có thể import
`vendor/bybylang/bybylang` cho compiler DSL hoặc `vendor/bybylang/gpubackend`
cho các phép GPU. Facade framework export lại một số symbol backend/session
qua `backend.nim` và `nimformers.nim`.

Xem [hướng dẫn BybyLang](bybylang-guide.vi.md) cho ví dụ đầy đủ: build compiler,
compile/chạy script, gọi C ABI và quản lý đường dẫn source/output.
`make bybylang-test` chạy cả ví dụ native DSL và C ABI import.


## Chọn backend

```nim
type GpuBackend* = enum
  gbAuto, gbCpu, gbCuda, gbMetal, gbVulkan

proc parseBackend*(s: string): GpuBackend
proc normalizeBackend*(b: GpuBackend): GpuBackend
proc detectBackend*(): GpuBackend
proc setForbidCpuFallback*(forbid: bool)
```

`parseBackend("cuda")` ánh xạ sang Vulkan. `newBackend("cpu")` cấp framework
không được hỗ trợ; hàm GPU BybyLang trực tiếp còn nhánh CPU tham chiếu/fallback
ở các phép có implementation. `setForbidCpuFallback(true)` biến fallback
của phép toán thành lỗi.


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


## Block micro-kernel native

Có thể khai báo bytecode micro-kernel ngay trong `.bybylang`:

```bybylang
micro kernel double A -> OUT threads 4 tg 4 out 4:
  emitRToU 0 0
  emitULoadIn 1 0 0
  emitLoadConstF32 2 2.0'f32
  emitMul 1 1 2
  emitUStoreF32 1 0 0
  emitHalt
```

A là mảng float đầu vào. `threads` là kích thước dispatch, `tg` là kích
thước threadgroup, `out` là số phần tử output. Compiler biến các dòng
`emit*` thành bytecode `VmProgram` rồi chạy qua session GPU đã chọn.
Xem [API micro-VM/JIT](microkernel-api.vi.md).


## Build AOT và chạy

```sh
# From the Nimformers repository root:
make -C vendor/bybylang build
./vendor/bybylang/bybylang vendor/bybylang/demo/demo_gpu.bybylang \
  --aot=vendor/bybylang/demo/demo_gpu_out
./vendor/bybylang/demo/demo_gpu_out
```

`--aot` ghi và compile file Nim; cần chạy binary sinh ra riêng.
`BYBYLANG_NIMCACHE` chọn thư mục cache. Ví dụ import `lib_c_api` cần search
path framework; thư mục test cung cấp chúng qua
`test/bybylang_test/nim.cfg`. `import nim` module `lib_c_api` compile
implementation ABI vào binary, không load động shared library Python.

Trên macOS, thêm `caffeinate -i` trước lệnh cần giữ máy thức khi build/chạy.
Tiến trình GPU phải truy cập được thiết bị; môi trường sandbox/headless
có thể không thấy Metal.
