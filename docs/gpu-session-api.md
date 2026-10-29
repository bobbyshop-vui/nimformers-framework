
## 11. Session API (`sessionXxx`, low-level multi-op GPU sessions)

[English](gpu-session-api.md) | [Tiếng Việt](gpu-session-api.vi.md)

Each immediate `beXxx` call performs its own operation and may upload/read data for that call. The **Session API**, re-exported by `backend.nim` from `vendor/bybylang/gpubackend.nim`, lets you open a **single session** and encode several operations into it back-to-back, reading results back to the CPU once at the end — this is how `beAddLayernormFused`, `SwiGLUL.forward`, and fused RoPE/attention/decode paths avoid multiple GPU round-trips per layer.

Available symbols (imported by using `backend`, no separate import needed):

```nim
proc sessionBegin*(backend: GpuBackend): bool
```
Opens a new session on `backend`. Returns `false` if the backend doesn't support sessions (caller must fall back to individual `beXxx` calls).

```nim
proc sessionUpload*(backend: GpuBackend, data: seq[float32]): SessionHandle
proc sessionUploadIndices*(backend: GpuBackend, data: seq[int32]): SessionHandle
proc sessionAllocScratch*(backend: GpuBackend, n: int): SessionHandle
```
`sessionUpload`/`sessionUploadIndices` upload a float32/int32 array to the GPU and return a handle to it. `sessionAllocScratch` allocates an uninitialized `n`-element GPU buffer for a result you haven't computed yet.

```nim
proc sessionMatmul*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int): bool
proc sessionMatmulNT*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int): bool
proc sessionMatmulQ4*(backend: GpuBackend, a: SessionHandle, w: ResidentWeightHandle, c: var SessionHandle, M: int): bool
```
Encode a matmul into the open session, writing the result into `c` (usually a handle from `sessionAllocScratch`). `sessionMatmulNT` computes `C(M,N) = A(M,K) @ B(N,K)^T` with `B` kept in its original `[N,K]` layout (no CPU-side transpose needed) — used for attention decode against K/V-cache slices. `sessionMatmulQ4` multiplies against an int4-resident weight handle (see `beResidentUploadQ4`). All return `false` if the backend doesn't support that op inside a session — the caller must not treat `false` as a hard error, just fall back to the non-session path.

```nim
proc sessionVecOp*(backend: GpuBackend, op: string, a, b: SessionHandle, c: var SessionHandle, n: int): bool
proc sessionActivation*(backend: GpuBackend, op: string, x: SessionHandle, y: var SessionHandle, n: int): bool
```
`sessionVecOp`: element-wise op (`"add"`, `"sub"`, `"mul"`, `"div"`) between two uploaded/scratch buffers. `sessionActivation`: activation op (`"relu"`, `"sigmoid"`, `"tanh"`, ...) on a single buffer.

```nim
proc sessionSoftmax*(backend: GpuBackend, x: SessionHandle, y: var SessionHandle, rows, cols: int): bool
proc sessionLayernormEnc*(backend: GpuBackend, x, gamma, beta: SessionHandle, y: var SessionHandle,
                          rows, cols: int, eps: float32): bool
proc sessionAttentionFusedEnc*(backend: GpuBackend, q, k, v: SessionHandle,
  o, sMatrix: var SessionHandle, B, H, S, D: int, scale: float32): bool
proc sessionRope*(backend: GpuBackend, x: SessionHandle, cosT, sinT: SessionHandle,
  T, Hn, D: int): bool
```
Higher-level ops encodable inside a session: row-wise softmax, layernorm, fused attention, and RoPE rotation. Support varies by backend (currently richest on Metal) — always check the returned `bool` and fall back if `false`.

```nim
proc sessionCopyInto*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
  srcOffsetElems, dstOffsetElems, n: int): bool
proc sessionCopyInto*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
  dstOffsetElems, n: int): bool
```
Copies data between two GPU-resident buffers without a CPU round-trip (e.g. writing a K/V slice into a KV-cache buffer).

```nim
proc sessionEnd*(backend: GpuBackend): bool
```
Commits and waits for every op encoded since `sessionBegin` to finish. Must be called before reading any result out.

```nim
proc sessionRead*(backend: GpuBackend, h: SessionHandle, n: int): seq[float32]
proc sessionFree*(backend: GpuBackend, h: var SessionHandle)
```
`sessionRead` reads `n` float32 values back from a GPU handle to the CPU — call this **once**, after `sessionEnd`, not between ops. `sessionFree` releases a handle's GPU-side memory; always free every handle you allocated (uploads and scratch buffers alike), even on the failure path.


### 11.1. Typical usage pattern

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

Rules of thumb:
- Always check `sessionBegin`'s return value; not every backend supports sessions.
- Chain every op's `bool` result with `and` and check it once, right before `sessionEnd` — don't call `sessionRead` until after a successful `sessionEnd`.
- Free every handle (`sessionFree`) on both the success and the fallback path, to avoid leaking GPU memory.
- The Spectron trainer in `test/spectron/test_nimformer.nim` currently creates the model with `useSession = true`; change that argument in its main block to compare the non-session path.


### 11.2. Where the Session API is already used in this codebase

- `beResidentLinearQ4` / `beResidentLinearFp32` / `beAddLayernormFused` in `backend.nim`
- `SwiGLUL.forward` in `test/deepseek-inference/nim_inference.nim` (sigmoid + mul + mul fused)
- `forwardPrefill` / `forwardDecode` (GQA attention) and the cross-submodule "1 session per layer" helpers (`sMatmul`, `sMatmulNT`) in `test/deepseek-inference/nim_inference.nim`
- `LinearPlusAttention` in `test/spectron/test_nimformer.nim` (controlled by its `useSession` argument)

---
