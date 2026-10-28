## 2. Backend API (`backend.nim`)

`backend.nim` is the unified application layer over `vendor/bybylang`. Every `be*` function below takes `ctx: Backend` first.

### 2.1. Backend init / management

```nim
proc newBackend*(force: string = backend): Backend
```
Creates a backend. Supported choices are `"metal"`, `"vulkan"`, `"cuda"` (legacy alias redirected to Vulkan), `"opencl"` (legacy spelling redirected to Vulkan), and `"auto"`. `"cpu"` is rejected by the framework backend constructor. `"auto"` detects an available GPU and raises if none is found.

```nim
proc closeBackend*(b: Backend)
```
Closes the backend (currently a no-op).

```nim
proc setForbidCpuFallback*(forbid: bool)
```
Controls BybyLang's per-operation CPU fallback after a GPU kernel failure. The backend module initializes this setting to `false`; call `setForbidCpuFallback(true)` for fail-fast runs. This does not enable `newBackend("cpu")`.

```nim
proc toByby*(bk: BackendKind): byby.GpuBackend
```
Converts a `BackendKind` into BybyLang's internal `GpuBackend`. `newBackend("cuda")` is retained as a compatibility alias and selects Vulkan; it does not invoke CUDA.

`BackendKind` exposes `bkMetal`, `bkCuda` (compatibility value routed to Vulkan), and `bkVulkan`. `Backend` stores the selected kind in its public `kind` field.

### 2.2. Matrix operations

```nim
proc beMatmul*(ctx: Backend, a: openArray[float32], M, K: int,
               b: openArray[float32], K2, N: int): seq[float32]
```
Multiplies matrix `A (M x K)` by `B (K x N)`, returning a flattened `C (M x N)`.

```nim
proc beMatmul2*(ctx: Backend,
                a1: openArray[float32], M1, K1: int, b1: openArray[float32], K1b, N1: int,
                a2: openArray[float32], M2, K2: int, b2: openArray[float32], K2b, N2: int):
                tuple[y1, y2: seq[float32]]
```
Two independent matrix multiplications inside a single command buffer where supported, reducing overhead compared to calling `beMatmul` twice.

```nim
proc beMatmulQ4*(ctx: Backend, a: openArray[float32], wq: openArray[uint8],
                 scales, zeros: openArray[float32],
                 M, K, N, groupSize, nGroupsPerRow: int): seq[float32]
```
Matrix multiply against an int4-quantized weight (asymmetric, per-group) without dequantizing the whole thing to fp32 first. Always guarantees a correctly sized `M*N` result (zero-padded if the backend returns fewer elements).

### 2.3. Resident weights on GPU (inference optimization)

```nim
proc beResidentUploadQ4*(ctx: Backend, wq: openArray[uint8], scales, zeros: openArray[float32],
                         k, n, groupSize, nGroupsPerRow: int): ResidentWeightHandle
proc beResidentLinearQ4*(ctx: Backend, x: openArray[float32], w: ResidentWeightHandle, M: int): seq[float32]
proc beResidentFreeQ4*(w: var ResidentWeightHandle)
```
Uploads an int4 weight to the GPU **once** at model-load time (`beResidentUploadQ4`); every decode step then only uploads the small activation (`beResidentLinearQ4`) without touching the weight bytes again. Returns an empty `seq` if the backend doesn't support it (caller must fall back to `beMatmulQ4`). Freed with `beResidentFreeQ4`.

```nim
proc beResidentUploadFp32*(ctx: Backend, wT: openArray[float32]): SessionHandle
proc beResidentLinearFp32*(ctx: Backend, x: openArray[float32], w: SessionHandle, M, K, N: int): seq[float32]
proc beResidentFreeFp32*(ctx: Backend, w: var SessionHandle)
```
Same idea for regular fp32/tied-embedding weights: `wT` must be dequantized + transposed **once** before calling `beResidentUploadFp32`. This is the main mechanism that keeps `Linear.forward` from having to dequant + transpose the whole matrix every token.

### 2.4. Element-wise ops & activations

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
`beApflu`/`beApfluBackward`: a custom "APF Linear Unit" activation configurable via `alpha`/`beta`, plus its backward gradient.

### 2.5. Softmax / LayerNorm / Embedding / Attention

```nim
proc beSoftmax*(ctx: Backend, x: openArray[float32], rows, cols: int): seq[float32]
```
Row-wise softmax over a `rows x cols` matrix.

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
Fuses "residual add followed by layernorm" into a **single GPU session**. Returns both `summed` (raw a+b, used as the residual for the block's final add) and `normed`. Only actually beneficial on backends that support `sessionLayernormEnc` (currently Metal); other backends fall back to `beAdd` + `beLayernorm` with identical behavior/precision.

```nim
proc beEmbeddingLookup*(ctx: Backend, table: openArray[float32], indices: openArray[int32],
                        vocab, dim: int): seq[float32]
```
Looks up rows in embedding `table (vocab x dim)` by `indices`, returning `(len(indices) x dim)`.

```nim
proc beAttentionFused*(ctx: Backend, q, k, v, mask: openArray[float32],
                       B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]]
proc beAttentionFusedBackward*(ctx: Backend, q, k, v, s_matrix, dy: openArray[float32],
                               B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]]
```
Attention (Q, K, V, mask) fused into a single kernel call. `B`=batch size, `H`=number of heads, `S`=sequence length, `D`=head dimension. `beAttentionFusedBackward` reuses the `s_matrix` saved from forward to compute `dq`, `dk`, `dv`.

### 2.6. Indexing / tensor-layout helpers

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

These low-level wrappers are used by `tensor_ops.nim`; tensor-oriented forms are documented in [Tensor Operations API](tensor-ops-api.md).

### 2.7. CustomFloat / APF helpers (via `backend.nim`)

```nim
proc beCustomfloatEncode*(ctx: Backend, arr: openArray[float32], cf: CustomFloat): seq[uint8]
proc beCustomfloatDecode*(ctx: Backend, buf: openArray[uint8], cf: CustomFloat): seq[float32]
proc beApfCastForTraining*(ctx: Backend, arr: openArray[float32], gradArr: openArray[float32] = [],
                           relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                           expMargin = APF_EXP_MARGIN_BITS): tuple[data: seq[uint8], cf: CustomFloat]
```
Thin wrappers calling `customfloat.encodeArray` / `decodeArray` / `apfCastForTraining` directly (see section 3).

---


---

## 8. Selecting a Backend

```nim
import backend

# Automatically select: Metal, then Vulkan
newBackend("auto")

# Explicit GPU selections
newBackend("metal")
newBackend("vulkan")
newBackend("cuda") # legacy alias; redirects to Vulkan
```

`newBackend("cpu")` is rejected. `"opencl"` is a compatibility spelling redirected to Vulkan.

The repository Makefile has target names for backend-specific commands; they are build/run conveniences, not backend API names. See `make help` or the root Makefile for targets in the current checkout.

---
