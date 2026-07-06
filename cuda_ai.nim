## cuda_ai.nim
## Wrapper Nim gọi CUDA qua cuda_bridge.h (cuBLAS cho matmul + kernel .cu tự
## viết cho add/relu/sigmoid/tanh/softmax/layernorm/embedding_lookup) — cùng
## pattern generic-dispatch với metal_ai.nim, chỉ khác tầng bridge.
##
## LƯU Ý BUILD (khác Metal): file .cu KHÔNG dùng {.compile.} thẳng như
## metal_bridge.m, vì trình biên dịch C mặc định Nim gọi không phải nvcc.
## Phải build 2 bước — xem target `cuda`/`run-cuda` trong Makefile:
##   1) nvcc compile cuda_kernels.cu -> libcudakernels.a
##   2) nim c -d:withCuda --passL:"-L. -lcudakernels -lcudart -lcublas" ...
##
## Chỉ import module này khi build với -d:withCuda (xem backend.nim).

{.passL: "-lcudart -lcublas".}

type
  CudaBufRef = pointer
  CudaCtxHandle = pointer

# LƯU Ý: mọi hàm importc gọi vào C đều cần {.cdecl.} tường minh — không có
# nó, Nim suy ra calling convention mặc định {.nimcall.}, khiến các hàm này
# không thể truyền như first-class proc value có kiểu {.cdecl.} (ví dụ
# truyền cudaReluRaw vào cudaElementwise bên dưới) -> lỗi type mismatch
# "Calling convention mismatch: got 'nimcall', but expected 'cdecl'".
proc cudaDeviceCountRaw(): cint {.importc: "cuda_device_count", header: "cuda_bridge.h", cdecl.}
proc cudaCreateCtxRaw(): CudaCtxHandle {.importc: "cuda_create_context", header: "cuda_bridge.h", cdecl.}
proc cudaDestroyCtxRaw(c: CudaCtxHandle) {.importc: "cuda_destroy_context", header: "cuda_bridge.h", cdecl.}
proc cudaAllocRaw(bytes: csize_t): CudaBufRef {.importc: "cuda_alloc", header: "cuda_bridge.h", cdecl.}
proc cudaFreeRaw(b: CudaBufRef) {.importc: "cuda_free", header: "cuda_bridge.h", cdecl.}
proc cudaUploadRaw(dst: CudaBufRef, src: pointer, bytes: csize_t) {.importc: "cuda_upload", header: "cuda_bridge.h", cdecl.}
proc cudaDownloadRaw(src: CudaBufRef, dst: pointer, bytes: csize_t) {.importc: "cuda_download", header: "cuda_bridge.h", cdecl.}
proc cudaMatmulRaw(ctx: CudaCtxHandle, dA: pointer, M, K: cint, dB: pointer, K2, N: cint, dC: pointer)
  {.importc: "cuda_matmul", header: "cuda_bridge.h", cdecl.}
proc cudaAddRaw(ctx: CudaCtxHandle, a, b, c: pointer, n: cint) {.importc: "cuda_add", header: "cuda_bridge.h", cdecl.}
proc cudaReluRaw(ctx: CudaCtxHandle, x, y: pointer, n: cint) {.importc: "cuda_relu", header: "cuda_bridge.h", cdecl.}
proc cudaSigmoidRaw(ctx: CudaCtxHandle, x, y: pointer, n: cint) {.importc: "cuda_sigmoid", header: "cuda_bridge.h", cdecl.}
proc cudaTanhRaw(ctx: CudaCtxHandle, x, y: pointer, n: cint) {.importc: "cuda_tanh_act", header: "cuda_bridge.h", cdecl.}
proc cudaSoftmaxRaw(ctx: CudaCtxHandle, x, y: pointer, rows, cols: cint) {.importc: "cuda_softmax", header: "cuda_bridge.h", cdecl.}
proc cudaLayernormRaw(ctx: CudaCtxHandle, x, y, gamma, beta: pointer, rows, cols: cint, eps: cfloat)
  {.importc: "cuda_layernorm", header: "cuda_bridge.h", cdecl.}
proc cudaEmbeddingRaw(ctx: CudaCtxHandle, table: pointer, idx: pointer, outp: pointer, vocab, dim, num: cint)
  {.importc: "cuda_embedding_lookup", header: "cuda_bridge.h", cdecl.}

proc cudaDeviceAvailable*(): bool =
  ## Probe rẻ cho backend.nim auto-detect — chỉ hỏi driver, không tạo context.
  cudaDeviceCountRaw() > 0

type
  CudaContext* = object
    handle: CudaCtxHandle

proc newCudaContext*(): CudaContext =
  if cudaDeviceCountRaw() <= 0:
    raise newException(IOError, "Không tìm thấy GPU CUDA (cuda_device_count() == 0)")
  result.handle = cudaCreateCtxRaw()
  if result.handle == nil:
    raise newException(IOError, "cublasCreate thất bại (xem stderr log từ cuda_kernels.cu)")

proc closeCudaContext*(ctx: CudaContext) =
  cudaDestroyCtxRaw(ctx.handle)

# ── helper upload/download — CHƯA có buffer pool như metal_ai (TODO nếu
#    cần tối ưu RSS/latency sau này, cùng pattern poolGet/poolPut). ──
proc uploadF32(data: openArray[float32]): CudaBufRef =
  result = cudaAllocRaw(csize_t(data.len * 4))
  if data.len > 0: cudaUploadRaw(result, unsafeAddr data[0], csize_t(data.len * 4))

proc uploadI32(data: openArray[int32]): CudaBufRef =
  result = cudaAllocRaw(csize_t(data.len * 4))
  if data.len > 0: cudaUploadRaw(result, unsafeAddr data[0], csize_t(data.len * 4))

proc downloadF32(buf: CudaBufRef, n: int): seq[float32] =
  result = newSeq[float32](n)
  if n > 0: cudaDownloadRaw(buf, addr result[0], csize_t(n * 4))

# ─────────────────────────────────────────────────────────────
# Wrapper cho từng kernel — cùng chữ ký với bản metalXxx tương ứng
# ─────────────────────────────────────────────────────────────

proc cudaMatmul*(ctx: CudaContext, a: openArray[float32], M, K: int,
                  b: openArray[float32], K2, N: int): seq[float32] =
  assert K == K2
  let dA = uploadF32(a)
  let dB = uploadF32(b)
  let dC = cudaAllocRaw(csize_t(M * N * 4))
  cudaMatmulRaw(ctx.handle, dA, cint(M), cint(K), dB, cint(K2), cint(N), dC)
  result = downloadF32(dC, M * N)
  cudaFreeRaw(dA); cudaFreeRaw(dB); cudaFreeRaw(dC)

proc cudaAdd*(ctx: CudaContext, a, b: openArray[float32]): seq[float32] =
  assert a.len == b.len
  let n = a.len
  let dA = uploadF32(a)
  let dB = uploadF32(b)
  let dC = cudaAllocRaw(csize_t(n * 4))
  cudaAddRaw(ctx.handle, dA, dB, dC, cint(n))
  result = downloadF32(dC, n)
  cudaFreeRaw(dA); cudaFreeRaw(dB); cudaFreeRaw(dC)

proc cudaElementwise(ctx: CudaContext, x: openArray[float32],
                      raw: proc(ctx: CudaCtxHandle, x, y: pointer, n: cint) {.cdecl.}): seq[float32] =
  let n = x.len
  let dX = uploadF32(x)
  let dY = cudaAllocRaw(csize_t(n * 4))
  raw(ctx.handle, dX, dY, cint(n))
  result = downloadF32(dY, n)
  cudaFreeRaw(dX); cudaFreeRaw(dY)

proc cudaRelu*(ctx: CudaContext, x: openArray[float32]): seq[float32] =
  cudaElementwise(ctx, x, cudaReluRaw)

proc cudaSigmoid*(ctx: CudaContext, x: openArray[float32]): seq[float32] =
  cudaElementwise(ctx, x, cudaSigmoidRaw)

proc cudaTanh*(ctx: CudaContext, x: openArray[float32]): seq[float32] =
  cudaElementwise(ctx, x, cudaTanhRaw)

proc cudaSoftmax*(ctx: CudaContext, x: openArray[float32], rows, cols: int): seq[float32] =
  assert x.len == rows * cols
  let dX = uploadF32(x)
  let dY = cudaAllocRaw(csize_t(rows * cols * 4))
  cudaSoftmaxRaw(ctx.handle, dX, dY, cint(rows), cint(cols))
  result = downloadF32(dY, rows * cols)
  cudaFreeRaw(dX); cudaFreeRaw(dY)

proc cudaLayernorm*(ctx: CudaContext, x: openArray[float32], gamma, beta: openArray[float32],
                     rows, cols: int, eps: float32 = 1e-5'f32): seq[float32] =
  assert x.len == rows * cols
  assert gamma.len == cols and beta.len == cols
  let dX = uploadF32(x)
  let dY = cudaAllocRaw(csize_t(rows * cols * 4))
  let dGamma = uploadF32(gamma)
  let dBeta = uploadF32(beta)
  cudaLayernormRaw(ctx.handle, dX, dY, dGamma, dBeta, cint(rows), cint(cols), cfloat(eps))
  result = downloadF32(dY, rows * cols)
  cudaFreeRaw(dX); cudaFreeRaw(dY); cudaFreeRaw(dGamma); cudaFreeRaw(dBeta)

proc cudaEmbeddingLookup*(ctx: CudaContext, table: openArray[float32], vocab, dim: int,
                           indices: openArray[int32]): seq[float32] =
  let num = indices.len
  let dTable = uploadF32(table)
  let dIdx = uploadI32(indices)
  let dOut = cudaAllocRaw(csize_t(num * dim * 4))
  cudaEmbeddingRaw(ctx.handle, dTable, dIdx, dOut, cint(vocab), cint(dim), cint(num))
  result = downloadF32(dOut, num * dim)
  cudaFreeRaw(dTable); cudaFreeRaw(dIdx); cudaFreeRaw(dOut)