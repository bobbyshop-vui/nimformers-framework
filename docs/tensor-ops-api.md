
# Tensor Operations API (`tensor_ops.nim`)

[English](tensor-ops-api.md) | [Tiếng Việt](tensor-ops-api.vi.md)

Import `tensor_ops` directly or get these symbols from `import nimformers`. `Tensor` and `Backend` are defined in `nimformer.nim` and `backend.nim`.


## Shape and elementwise helpers

```nim
proc expT*(ctx: Backend, x: Tensor): Tensor
proc logT*(ctx: Backend, x: Tensor): Tensor
proc sqrtT*(ctx: Backend, x: Tensor): Tensor
proc powT*(ctx: Backend, x: Tensor, p: float32): Tensor

proc view*(x: Tensor, newShape: seq[int]): Tensor
proc reshape*(x: Tensor, newShape: seq[int]): Tensor
proc permute*(ctx: Backend, x: Tensor, dims: seq[int]): Tensor
```

`view`/`reshape` preserve the flattened data and require a compatible element count. `permute` rearranges dimensions using the requested dimension order.


## Reductions

```nim
proc beSum*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32]
proc beMean*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32]
proc beMax*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32]
proc beArgMax*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[int]
proc beArgMax*(ctx: Backend, x: openArray[float32]): int
```

The shape-taking variants reduce one dimension and return flattened values. The scalar `beArgMax` returns the index of the maximum element in a flat input.


## Combine, split, and dropout

```nim
proc concat*(ctx: Backend, tensors: openArray[Tensor], dim: int): Tensor
proc stack*(ctx: Backend, tensors: openArray[Tensor], dim: int = 0): Tensor
proc split*(ctx: Backend, x: Tensor, splitSize: int, dim: int): seq[Tensor]

proc beDropout*(ctx: Backend, x: openArray[float32], p: float32,
                training: bool, seed: uint32): seq[float32]
proc dropoutMask*(ctx: Backend, n: int, p: float32, seed: uint32): seq[float32]
```

Dropout uses inverted scaling during training and returns unchanged input when `training` is false. `dropoutMask` is deterministic for a given seed and can be shared by forward/backward passes.


## Comparisons and selection

```nim
proc beGt*(ctx: Backend, a, b: openArray[float32]): seq[bool]
proc beLt*(ctx: Backend, a, b: openArray[float32]): seq[bool]
proc beEq*(ctx: Backend, a, b: openArray[float32]): seq[bool]
proc beGe*(ctx: Backend, a, b: openArray[float32]): seq[bool]
proc beLe*(ctx: Backend, a, b: openArray[float32]): seq[bool]
proc maskedFill*(x: Tensor, mask: openArray[bool], value: float32): Tensor
proc whereT*(cond: openArray[bool], a, b: Tensor): Tensor
```

`maskedFill` and `whereT` operate on host-side tensors. Comparison kernels return a host `seq[bool]`.


## Scans

```nim
proc beCumsum*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32]
proc cumsumT*(ctx: Backend, x: Tensor, dim: int): Tensor
proc beCumprod*(ctx: Backend, x: openArray[float32], shape: seq[int], dim: int): seq[float32]
proc cumprodT*(ctx: Backend, x: Tensor, dim: int): Tensor

type ScanPair* = tuple[a, b: float32]
proc beAffineScan*(ctx: Backend, aCoef, bCoef: openArray[float32],
                   shape: seq[int], dim: int): seq[float32]
```

`beAffineScan` computes the inclusive recurrence `h[t] = a[t] * h[t-1] + b[t]` with `h[-1] = 0`; the current implementation requires scanning the last dimension (`inner == 1`). `ScanPair` is the pair representation used by affine scan kernels.


## Example

```nim
import nimformers

let ctx = newBackend("metal")
let x = newTensor(@[2, 2])
let flat = @[1'f32, 2, 3, 4]
echo beSum(ctx, flat, @[2, 2], 1) # @[3, 7]
```
