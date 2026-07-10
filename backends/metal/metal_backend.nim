# metal_backend.nim - Metal GPU backend wrapper (chỉ biên dịch trên macOS).
# Trên Linux/Windows module này export các proc "not available" để gpubackend.nim
# vẫn import được bình thường trên mọi OS mà không cần #if trong code gọi.

import std/os

when defined(macosx):
  # Header/.m nằm CÙNG thư mục với chính file này (backends/metal/), nhưng
  # clang chỉ được Nim truyền -I cho thư mục chứa main module (vd. demo/),
  # nên phải tự thêm -I trỏ đúng thư mục này thì mới #include "metal_shim.h"
  # được, bất kể main module nằm ở đâu.
  const metalDir = currentSourcePath().parentDir()
  {.passC: "-I" & metalDir.}
  {.passC: "-fobjc-arc".}
  {.passL: "-framework Metal -framework Foundation".}
  {.compile: "metal_shim.m".}

  # Kernel source là file .metal THẬT (backends/metal/kernels/vecop_matmul.metal),
  # không còn escape thành C-string trong metal_shim.m nữa. staticRead đọc file
  # lúc Nim compile-time (không cần công cụ build ngoài, không cần metallib build-time).
  const kMetalKernelSrc = staticRead(metalDir / "kernels" / "vecop_matmul.metal")

  proc metal_available_c(): cint {.importc: "metal_available", header: "metal_shim.h".}
  proc metal_vecop_c(kernelSrc: cstring, op: cint, a, b: ptr float32, c: ptr float32, n: cint): cint
    {.importc: "metal_vecop", header: "metal_shim.h".}
  proc metal_matmul_c(kernelSrc: cstring, a, b: ptr float32, c: ptr float32, m, k, n: cint): cint
    {.importc: "metal_matmul", header: "metal_shim.h".}
  proc metal_matmul2_c(kernelSrc: cstring,
                        a1, b1: ptr float32, c1: ptr float32, m1, k1, n1: cint,
                        a2, b2: ptr float32, c2: ptr float32, m2, k2, n2: cint): cint
    {.importc: "metal_matmul2", header: "metal_shim.h".}

  proc metal_activation_c(kernelSrc: cstring, op: cint, x: ptr float32, y: ptr float32, n: cint): cint
    {.importc: "metal_activation", header: "metal_shim.h".}
  proc metal_softmax_c(kernelSrc: cstring, x: ptr float32, y: ptr float32, rows, cols: cint): cint
    {.importc: "metal_softmax", header: "metal_shim.h".}
  proc metal_layernorm_c(kernelSrc: cstring, x, gamma, beta: ptr float32, y: ptr float32, rows, cols: cint, eps: float32): cint
    {.importc: "metal_layernorm", header: "metal_shim.h".}
  proc metal_embedding_lookup_c(kernelSrc: cstring, table: ptr float32, indices: ptr int32, y: ptr float32, vocab, dim, num_indices: cint): cint
    {.importc: "metal_embedding_lookup", header: "metal_shim.h".}

  proc metalAvailable*(): bool =
    metal_available_c() != 0

  proc metalVecOp*(op: string, a, b: seq[float32]): seq[float32] =
    let opCode: cint =
      case op
      of "add": 0
      of "sub": 1
      of "mul": 2
      of "div": 3
      else: 0
    let n = a.len
    result = newSeq[float32](n)
    var aVar = a
    var bVar = b
    let ok = metal_vecop_c(kMetalKernelSrc.cstring, opCode, addr aVar[0], addr bVar[0], addr result[0], cint(n))
    if ok == 0:
      raise newException(CatchableError, "metal_vecop failed")

  proc metalMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
    result = newSeq[float32](m * n)
    var aVar = a
    var bVar = b
    let ok = metal_matmul_c(kMetalKernelSrc.cstring, addr aVar[0], addr bVar[0], addr result[0], cint(m), cint(k), cint(n))
    if ok == 0:
      raise newException(CatchableError, "metal_matmul failed")

  proc metalMatmul2*(a1, b1: seq[float32], m1, k1, n1: int,
                      a2, b2: seq[float32], m2, k2, n2: int):
                      tuple[c1, c2: seq[float32]] =
    ## Chạy 2 matmul độc lập trong CÙNG 1 command buffer (1 commit+wait duy
    ## nhất) - dùng khi có 2 phép matmul không phụ thuộc nhau cần chạy song
    ## song (vd. dW/dX trong backward), tránh overhead gọi metalMatmul() 2 lần.
    result.c1 = newSeq[float32](m1 * n1)
    result.c2 = newSeq[float32](m2 * n2)
    var a1v = a1; var b1v = b1
    var a2v = a2; var b2v = b2
    let ok = metal_matmul2_c(kMetalKernelSrc.cstring,
      addr a1v[0], addr b1v[0], addr result.c1[0], cint(m1), cint(k1), cint(n1),
      addr a2v[0], addr b2v[0], addr result.c2[0], cint(m2), cint(k2), cint(n2))
    if ok == 0:
      raise newException(CatchableError, "metal_matmul2 failed")

  proc metalActivation*(op: string, x: seq[float32]): seq[float32] =
    let opCode: cint =
      case op
      of "relu": 0
      of "sigmoid": 1
      of "tanh": 2
      else: 0
    let n = x.len
    result = newSeq[float32](n)
    var xVar = x
    let ok = metal_activation_c(kMetalKernelSrc.cstring, opCode, addr xVar[0], addr result[0], cint(n))
    if ok == 0:
      raise newException(CatchableError, "metal_activation failed")

  proc metalSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
    let n = rows * cols
    result = newSeq[float32](n)
    var xVar = x
    let ok = metal_softmax_c(kMetalKernelSrc.cstring, addr xVar[0], addr result[0], cint(rows), cint(cols))
    if ok == 0:
      raise newException(CatchableError, "metal_softmax failed")

  proc metalLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
    let n = rows * cols
    result = newSeq[float32](n)
    var xVar = x
    var gammaVar = gamma
    var betaVar = beta
    let ok = metal_layernorm_c(kMetalKernelSrc.cstring, addr xVar[0], addr gammaVar[0], addr betaVar[0], addr result[0], cint(rows), cint(cols), eps)
    if ok == 0:
      raise newException(CatchableError, "metal_layernorm failed")

  proc metalEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
    let num = indices.len
    result = newSeq[float32](num * dim)
    var tableVar = table
    var indicesVar = indices
    let ok = metal_embedding_lookup_c(kMetalKernelSrc.cstring, addr tableVar[0], addr indicesVar[0], addr result[0], cint(vocab), cint(dim), cint(num))
    if ok == 0:
      raise newException(CatchableError, "metal_embedding_lookup failed")

else:
  proc metalAvailable*(): bool = false
  proc metalVecOp*(op: string, a, b: seq[float32]): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalMatmul2*(a1, b1: seq[float32], m1, k1, n1: int,
                      a2, b2: seq[float32], m2, k2, n2: int):
                      tuple[c1, c2: seq[float32]] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalActivation*(op: string, x: seq[float32]): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
