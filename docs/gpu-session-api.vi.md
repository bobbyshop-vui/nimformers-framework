
## 11. API session (`sessionXxx`, session GPU nhiều phép toán cấp thấp)

[English](gpu-session-api.md) | [Tiếng Việt](gpu-session-api.vi.md)

Mỗi lệnh immediate `beXxx` thực hiện một phép và có thể upload/đọc lại dữ
liệu riêng. **Session API**, được `backend.nim` export lại từ
`vendor/bybylang/gpubackend.nim`, cho phép mở **một session**, encode nhiều
phép liên tiếp rồi đọc kết quả về CPU một lần cuối. Nhờ đó
`beAddLayernormFused`, `SwiGLUL.forward` và các nhánh gộp RoPE/attention/decode
tránh nhiều lượt truyền GPU mỗi layer.

Các symbol có sẵn khi import `backend`, không cần import riêng:

```nim
proc sessionBegin*(backend: GpuBackend): bool
```

Mở session mới trên `backend`. Trả `false` nếu backend không hỗ trợ session;
bên gọi cần dùng các lệnh `beXxx` riêng.

```nim
proc sessionUpload*(backend: GpuBackend, data: seq[float32]): SessionHandle
proc sessionUploadIndices*(backend: GpuBackend, data: seq[int32]): SessionHandle
proc sessionAllocScratch*(backend: GpuBackend, n: int): SessionHandle
```

`sessionUpload`/`sessionUploadIndices` upload mảng float32/int32 lên GPU và
trả handle. `sessionAllocScratch` cấp buffer GPU `n` phần tử chưa khởi tạo,
dành cho kết quả sẽ được tính sau.

```nim
proc sessionMatmul*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int): bool
proc sessionMatmulNT*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int): bool
proc sessionMatmulQ4*(backend: GpuBackend, a: SessionHandle, w: ResidentWeightHandle, c: var SessionHandle, M: int): bool
```

Encode phép nhân ma trận vào session đang mở, ghi kết quả vào `c`, thường là
handle từ `sessionAllocScratch`. `sessionMatmulNT` tính
`C(M,N)=A(M,K) @ B(N,K)^T` và giữ B ở layout `[N,K]`, không cần transpose
trên CPU; nhánh attention decode dùng nó cho các lát K/V cache.
`sessionMatmulQ4` nhân với handle trọng số resident int4; xem
`beResidentUploadQ4`. Các hàm trả `false` nếu phép không được hỗ trợ trong
session; bên gọi có thể chọn nhánh không dùng session thay vì coi mọi
`false` là lỗi nghiêm trọng.

```nim
proc sessionVecOp*(backend: GpuBackend, op: string, a, b: SessionHandle, c: var SessionHandle, n: int): bool
proc sessionActivation*(backend: GpuBackend, op: string, x: SessionHandle, y: var SessionHandle, n: int): bool
```

`sessionVecOp` thực hiện phép từng phần tử (`"add"`, `"sub"`, `"mul"`,
`"div"`) giữa hai buffer upload/scratch. `sessionActivation` chạy activation
(`"relu"`, `"sigmoid"`, `"tanh"`, ...) trên một buffer.

```nim
proc sessionSoftmax*(backend: GpuBackend, x: SessionHandle, y: var SessionHandle, rows, cols: int): bool
proc sessionLayernormEnc*(backend: GpuBackend, x, gamma, beta: SessionHandle, y: var SessionHandle,
                          rows, cols: int, eps: float32): bool
proc sessionAttentionFusedEnc*(backend: GpuBackend, q, k, v: SessionHandle,
  o, sMatrix: var SessionHandle, B, H, S, D: int, scale: float32): bool
proc sessionRope*(backend: GpuBackend, x: SessionHandle, cosT, sinT: SessionHandle,
  T, Hn, D: int): bool
```

Các phép cấp cao có thể encode gồm softmax từng hàng, LayerNorm, fused
attention và xoay RoPE. Mức hỗ trợ phụ thuộc backend, hiện đầy đủ nhất ở
Metal; luôn kiểm tra `bool` và chọn nhánh thay thế nếu trả `false`.

```nim
proc sessionCopyInto*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
  srcOffsetElems, dstOffsetElems, n: int): bool
proc sessionCopyInto*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
  dstOffsetElems, n: int): bool
```

Sao chép giữa hai buffer resident GPU mà không qua CPU, chẳng hạn ghi một
lát K/V vào buffer KV cache.

```nim
proc sessionEnd*(backend: GpuBackend): bool
```

Commit và đợi mọi phép đã encode từ `sessionBegin` hoàn thành. Phải gọi
trước khi đọc kết quả.

```nim
proc sessionRead*(backend: GpuBackend, h: SessionHandle, n: int): seq[float32]
proc sessionFree*(backend: GpuBackend, h: var SessionHandle)
```

`sessionRead` đọc `n` giá trị float32 từ handle GPU về CPU: gọi sau
`sessionEnd`, không xen giữa các phép. `sessionFree` giải phóng bộ nhớ GPU
của handle. Giải phóng mọi handle đã cấp, kể cả upload và scratch, trên cả
nhánh thất bại.


### 11.1. Mẫu sử dụng

```nim
let be = ctx.kind.toByby()
if sessionBegin(be):
  var hA = sessionUpload(be, aData)
  var hB = sessionUpload(be, bData)
  var hC = sessionAllocScratch(be, M * N)

  let ok = sessionMatmul(be, hA, hB, hC, M, K, N) and
           sessionEnd(be)

  if ok:
    result = sessionRead(be, hC, M * N)
  sessionFree(be, hA); sessionFree(be, hB); sessionFree(be, hC)
else:
  # Backend doesn't support sessions — fall back to the plain beMatmul path
  result = beMatmul(ctx, aData, M, K, bData, K, N)
```

Các nguyên tắc:

- Kiểm tra giá trị trả về của `sessionBegin`; không phải backend nào cũng hỗ trợ.
- Ghép các `bool` bằng `and` và kiểm tra trước `sessionEnd`; chỉ gọi
  `sessionRead` khi `sessionEnd` thành công.
- Giải phóng mọi handle bằng `sessionFree` trên cả nhánh thành công lẫn
  fallback để tránh rò bộ nhớ GPU.
- Trainer Spectron trong `test/spectron/test_nimformer.nim` tạo model với
  `useSession=true`; sửa đối số trong main để đối chiếu nhánh không session.


### 11.2. Các nơi đã dùng Session API

- `beResidentLinearQ4`, `beResidentLinearFp32`, `beAddLayernormFused` trong `backend.nim`.
- `SwiGLUL.forward` trong `test/deepseek-inference/nim_inference.nim`: gộp sigmoid và hai phép nhân.
- `forwardPrefill`/`forwardDecode` cho GQA và các helper liên module
  "một session mỗi layer" (`sMatmul`, `sMatmulNT`) trong `nim_inference.nim`.
- `LinearPlusAttention` trong `test/spectron/test_nimformer.nim`, điều khiển qua `useSession`.

---
