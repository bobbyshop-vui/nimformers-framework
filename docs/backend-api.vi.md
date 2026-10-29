
## 2. API backend (`backend.nim`)

[English](backend-api.md) | [Tiếng Việt](backend-api.vi.md)

`backend.nim` là lớp ứng dụng thống nhất trên `vendor/bybylang`. Mọi hàm
`be*` bên dưới đều nhận `ctx: Backend` làm đối số đầu tiên.


### 2.1. Khởi tạo và quản lý backend

```nim
proc newBackend*(force: string = backend): Backend
```

Tạo backend. Các lựa chọn là `"metal"`, `"vulkan"`, `"cuda"` (alias cũ chuyển
sang Vulkan), `"opencl"` (tên tương thích chuyển sang Vulkan) và `"auto"`.
Constructor framework từ chối `"cpu"`. `"auto"` dò GPU có sẵn và báo lỗi
nếu không tìm thấy.

```nim
proc closeBackend*(b: Backend)
```

Đóng backend; hiện hàm này không thực hiện thao tác nào.

```nim
proc setForbidCpuFallback*(forbid: bool)
```

Điều khiển việc BybyLang chuyển từng phép toán về CPU khi kernel GPU thất
bại. Module backend khởi tạo thiết lập này thành `false`; gọi
`setForbidCpuFallback(true)` để báo lỗi ngay. Thiết lập này không cho phép
`newBackend("cpu")`.

```nim
proc toByby*(bk: BackendKind): byby.GpuBackend
```

Đổi `BackendKind` sang `GpuBackend` nội bộ của BybyLang. `newBackend("cuda")`
chỉ là alias tương thích chọn Vulkan, không gọi CUDA.

`BackendKind` có `bkMetal`, `bkCuda` (giá trị tương thích chạy qua Vulkan) và
`bkVulkan`. `Backend` lưu loại đã chọn trong trường công khai `kind`.


### 2.2. Phép toán ma trận

```nim
proc beMatmul*(ctx: Backend, a: openArray[float32], M, K: int,
               b: openArray[float32], K2, N: int): seq[float32]
```

Nhân `A (M x K)` với `B (K x N)`, trả về `C (M x N)` dạng mảng phẳng.

```nim
proc beMatmul2*(ctx: Backend,
                a1: openArray[float32], M1, K1: int, b1: openArray[float32], K1b, N1: int,
                a2: openArray[float32], M2, K2: int, b2: openArray[float32], K2b, N2: int):
                tuple[y1, y2: seq[float32]]
```

Hai phép nhân độc lập trong một command buffer nếu backend hỗ trợ, giảm chi
phí so với gọi `beMatmul` hai lần.

```nim
proc beMatmulQ4*(ctx: Backend, a: openArray[float32], wq: openArray[uint8],
                 scales, zeros: openArray[float32],
                 M, K, N, groupSize, nGroupsPerRow: int): seq[float32]
```

Nhân với trọng số int4 bất đối xứng theo group mà không giải lượng tử toàn
bộ sang fp32 trước. Luôn bảo đảm kết quả có `M*N` phần tử; thêm số 0 nếu
backend trả ít phần tử hơn.


### 2.3. Trọng số resident trên GPU để tối ưu inference

```nim
proc beResidentUploadQ4*(ctx: Backend, wq: openArray[uint8], scales, zeros: openArray[float32],
                         k, n, groupSize, nGroupsPerRow: int): ResidentWeightHandle
proc beResidentLinearQ4*(ctx: Backend, x: openArray[float32], w: ResidentWeightHandle, M: int): seq[float32]
proc beResidentFreeQ4*(w: var ResidentWeightHandle)
```

Upload trọng số int4 lên GPU **một lần** khi load model bằng
`beResidentUploadQ4`. Mỗi bước decode chỉ upload activation nhỏ qua
`beResidentLinearQ4`, không upload lại bytes trọng số. Hàm trả `seq` rỗng nếu
backend không hỗ trợ; bên gọi cần dùng `beMatmulQ4`. Giải phóng bằng
`beResidentFreeQ4`.

```nim
proc beResidentUploadFp32*(ctx: Backend, wT: openArray[float32]): SessionHandle
proc beResidentLinearFp32*(ctx: Backend, x: openArray[float32], w: SessionHandle, M, K, N: int): seq[float32]
proc beResidentFreeFp32*(ctx: Backend, w: var SessionHandle)
```

Cơ chế tương tự cho trọng số fp32/tied embedding: giải lượng tử và transpose
`wT` **một lần** trước `beResidentUploadFp32`. Cách này tránh giải lượng tử
và transpose toàn bộ ma trận trong mỗi token của `Linear.forward`.


### 2.4. Phép toán từng phần tử và activation

```nim
proc beAdd*(ctx: Backend, a, b: openArray[float32]): seq[float32]
proc beSub*(ctx: Backend, a, b: openArray[float32]): seq[float32]   # a - b
proc beMul*(ctx: Backend, a, b: openArray[float32]): seq[float32]
proc beDiv*(ctx: Backend, a, b: openArray[float32]): seq[float32]
proc beRelu*(ctx: Backend, x: openArray[float32]): seq[float32]
proc beSigmoid*(ctx: Backend, x: openArray[float32]): seq[float32]
proc beTanh*(ctx: Backend, x: openArray[float32]): seq[float32]
proc beApflu*(ctx: Backend, x: openArray[float32],
              alpha: float32 = 0.1, beta: float32 = 0.1): seq[float32]
proc beApfluBackward*(ctx: Backend, x, dy: openArray[float32],
                       alpha: float32 = 0.1, beta: float32 = 0.1): seq[float32]
```

`beApflu`/`beApfluBackward` cung cấp activation tùy chỉnh "APF Linear Unit"
với `alpha`/`beta` và gradient backward tương ứng.


### 2.5. Softmax, LayerNorm, Embedding và Attention

```nim
proc beSoftmax*(ctx: Backend, x: openArray[float32], rows, cols: int): seq[float32]
```

Softmax theo từng hàng của ma trận `rows x cols`.

```nim
proc beLayernorm*(ctx: Backend, x: openArray[float32], gamma, beta: openArray[float32],
                  rows, cols: int, eps: float32): seq[float32]
proc beLayernormBackward*(ctx: Backend, dy, x, gamma, beta: openArray[float32],
                          rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]]
```

```nim
proc beAddLayernormFused*(ctx: Backend, a, b: openArray[float32], gamma, beta: openArray[float32],
                          rows, cols: int, eps: float32): tuple[summed, normed: seq[float32]]
```

Gộp residual add rồi LayerNorm vào **một session GPU**. Trả cả `summed`
(`a+b` thô, làm residual cho phép cộng cuối block) và `normed`. Chỉ có lợi
trên backend hỗ trợ `sessionLayernormEnc` (hiện là Metal); backend khác
chạy `beAdd` + `beLayernorm` với hành vi và độ chính xác tương đương.

```nim
proc beEmbeddingLookup*(ctx: Backend, table: openArray[float32], indices: openArray[int32],
                        vocab, dim: int): seq[float32]
```

Tra các hàng của `table (vocab x dim)` bằng `indices`, trả về
`len(indices) x dim`.

```nim
proc beAttentionFused*(ctx: Backend, q, k, v, mask: openArray[float32],
                       B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]]
proc beAttentionFusedBackward*(ctx: Backend, q, k, v, s_matrix, dy: openArray[float32],
                               B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]]
```

Gộp attention (Q, K, V, mask) vào một lần gọi kernel. `B` là batch, `H` là
số head, `S` là sequence length, `D` là head dimension.
`beAttentionFusedBackward` tái sử dụng `s_matrix` từ forward để tính `dq`,
`dk`, `dv`.


### 2.6. Chỉ số và bố trí tensor

```nim
proc beGatherGpu*(ctx: Backend, src: seq[float32], idx: seq[int32]): seq[float32]
proc beArgmaxIndexGpu*(ctx: Backend, x: seq[float32], outer, dimSize, inner: int):
  tuple[val: seq[float32], idx: seq[int32]]
proc beCompareGpuOp*(ctx: Backend, a, b: seq[float32], op: int): seq[bool]
proc beDropoutMaskGpu*(ctx: Backend, n: int, p: float32, seed: uint32): seq[float32]
proc bePermuteGpu*(ctx: Backend, src: seq[float32], newStrides, oldStrides,
                   dimsMap: seq[int32], n: int): seq[float32]
proc beSplitSliceGpu*(ctx: Backend, src: seq[float32],
                      dimTotal, thisSize, offset, inner, n: int): seq[float32]
proc beConcatGpu*(ctx: Backend, flatSrc: seq[float32], srcBase, srcDimSize,
                  srcDimOffset: seq[int32], numTensors, totalDim, inner: int): seq[float32]
```

Các wrapper cấp thấp này được `tensor_ops.nim` sử dụng. Phiên bản nhận tensor
được mô tả trong [API phép toán tensor](tensor-ops-api.vi.md).


### 2.7. Hàm CustomFloat/APF qua `backend.nim`

```nim
proc beCustomfloatEncode*(ctx: Backend, arr: openArray[float32], cf: CustomFloat): seq[uint8]
proc beCustomfloatDecode*(ctx: Backend, buf: openArray[uint8], cf: CustomFloat): seq[float32]
proc beApfCastForTraining*(ctx: Backend, arr: openArray[float32], gradArr: openArray[float32] = [],
                           relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                           expMargin = APF_EXP_MARGIN_BITS): tuple[data: seq[uint8], cf: CustomFloat]
```

Các wrapper mỏng gọi trực tiếp `customfloat.encodeArray`, `decodeArray` và
`apfCastForTraining`; xem phần 3.

---


## 8. Chọn backend

```nim
import backend

# Automatically select: Metal, then Vulkan
newBackend("auto")

# Explicit GPU selections
newBackend("metal")
newBackend("vulkan")
newBackend("cuda") # legacy alias; redirects to Vulkan
```

`newBackend("cpu")` bị từ chối. `"opencl"` là tên tương thích chuyển sang Vulkan.

Các target backend trong Makefile là tiện ích build/run, không phải tên API.
Xem [hướng dẫn lệnh Make](make-commands.vi.md) hoặc Makefile gốc để biết các
target hiện có.
