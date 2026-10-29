
# API phép toán tensor (`tensor_ops.nim`)

[English](tensor-ops-api.md) | [Tiếng Việt](tensor-ops-api.vi.md)

Import trực tiếp `tensor_ops` hoặc dùng các symbol từ `import nimformers`.
`Tensor` và `Backend` được định nghĩa trong `nimformer.nim` và `backend.nim`.


## Shape và phép toán từng phần tử

```nim
proc expT*(ctx: Backend, x: Tensor): Tensor
proc logT*(ctx: Backend, x: Tensor): Tensor
proc sqrtT*(ctx: Backend, x: Tensor): Tensor
proc powT*(ctx: Backend, x: Tensor, p: float32): Tensor

proc view*(x: Tensor, newShape: seq[int]): Tensor
proc reshape*(x: Tensor, newShape: seq[int]): Tensor
proc permute*(ctx: Backend, x: Tensor, dims: seq[int]): Tensor
```

`view`/`reshape` giữ dữ liệu phẳng và yêu cầu số phần tử tương thích.
`permute` sắp xếp lại các chiều theo thứ tự được yêu cầu.


## Phép giảm chiều

```nim
proc beSum*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32]
proc beMean*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32]
proc beMax*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32]
proc beArgMax*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[int]
proc beArgMax*(ctx: Backend, x: openArray[float32]): int
```

Các phiên bản nhận shape giảm một chiều rồi trả giá trị dạng phẳng.
`beArgMax` trả scalar là chỉ số phần tử lớn nhất trong input phẳng.


## Ghép, tách và dropout

```nim
proc concat*(ctx: Backend, tensors: openArray[Tensor], dim: int): Tensor
proc stack*(ctx: Backend, tensors: openArray[Tensor], dim: int = 0): Tensor
proc split*(ctx: Backend, x: Tensor, splitSize: int, dim: int): seq[Tensor]

proc beDropout*(ctx: Backend, x: openArray[float32], p: float32,
                training: bool, seed: uint32): seq[float32]
proc dropoutMask*(ctx: Backend, n: int, p: float32, seed: uint32): seq[float32]
```

Dropout dùng inverted scaling khi training và trả nguyên input khi
`training=false`. `dropoutMask` cho cùng kết quả với cùng seed và có thể
được dùng chung trong forward/backward.


## So sánh và chọn phần tử

```nim
proc beGt*(ctx: Backend, a, b: openArray[float32]): seq[bool]
proc beLt*(ctx: Backend, a, b: openArray[float32]): seq[bool]
proc beEq*(ctx: Backend, a, b: openArray[float32]): seq[bool]
proc beGe*(ctx: Backend, a, b: openArray[float32]): seq[bool]
proc beLe*(ctx: Backend, a, b: openArray[float32]): seq[bool]
proc maskedFill*(x: Tensor, mask: openArray[bool], value: float32): Tensor
proc whereT*(cond: openArray[bool], a, b: Tensor): Tensor
```

`maskedFill` và `whereT` thao tác trên tensor phía host. Kernel so sánh trả
`seq[bool]` trên host.


## Scan

```nim
proc beCumsum*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32]
proc cumsumT*(ctx: Backend, x: Tensor, dim: int): Tensor
proc beCumprod*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32]
proc cumprodT*(ctx: Backend, x: Tensor, dim: int): Tensor

type ScanPair* = tuple[a, b: float32]
proc beAffineScan*(ctx: Backend, aCoef, bCoef: openArray[float32],
                   shape: seq[int], dim: int): seq[float32]
```

`beAffineScan` tính truy hồi inclusive
`h[t] = a[t] * h[t-1] + b[t]`, với `h[-1]=0`. Bản hiện tại yêu cầu scan
chiều cuối (`inner == 1`). `ScanPair` biểu diễn cặp dùng trong kernel affine scan.


## Ví dụ

```nim
import nimformers

let ctx = newBackend("metal")
let x = newTensor(@[2, 2])
let flat = @[1'f32, 2, 3, 4]
echo beSum(ctx, flat, @[2, 2], 1) # @[3, 7]
```
