# metal_backend.nim - Metal GPU backend wrapper (chỉ biên dịch trên macOS).
import std/os

when defined(macosx):
  const metalDir = currentSourcePath().parentDir()
  {.passC: "-I" & metalDir.}
  {.passC: "-fobjc-arc".}
  {.passL: "-framework Metal -framework Foundation".}
  {.compile: "metal_shim.m".}

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
  proc metal_matmul_q4_c(kernelSrc: cstring, a: ptr float32, wq: ptr uint8,
                          scales, zeros: ptr float32, c: ptr float32,
                          m, k, n, groupSize, nGroupsPerRow: cint): cint
    {.importc: "metal_matmul_q4", header: "metal_shim.h".}
  proc metal_activation_c(kernelSrc: cstring, op: cint, x: ptr float32, y: ptr float32, n: cint): cint
    {.importc: "metal_activation", header: "metal_shim.h".}
  proc metal_softmax_c(kernelSrc: cstring, x: ptr float32, y: ptr float32, rows, cols: cint): cint
    {.importc: "metal_softmax", header: "metal_shim.h".}
  proc metal_layernorm_c(kernelSrc: cstring, x, gamma, beta: ptr float32, y: ptr float32, rows, cols: cint, eps: float32): cint
    {.importc: "metal_layernorm", header: "metal_shim.h".}
  proc metal_embedding_lookup_c(kernelSrc: cstring, table: ptr float32, indices: ptr int32, y: ptr float32, vocab, dim, num_indices: cint): cint
    {.importc: "metal_embedding_lookup", header: "metal_shim.h".}
  proc metal_activation_backward_c(kernelSrc: cstring, op: cint, x, dy: ptr float32, dx: ptr float32, n: cint): cint
    {.importc: "metal_activation_backward", header: "metal_shim.h".}
  proc metal_layernorm_backward_c(kernelSrc: cstring, dy, x, gamma, beta: ptr float32,
                                   dx, dgamma, dbeta: ptr float32, rows, cols: cint, eps: float32): cint
    {.importc: "metal_layernorm_backward", header: "metal_shim.h".}
  
  # === THÊM MỚI: Attention forward + backward ===
  proc metal_attention_fused_c(kernelSrc: cstring, q, k, v: ptr float32, o, s_matrix: ptr float32,
                                B, H, S, D: cint, scale: float32): cint
    {.importc: "metal_attention_fused", header: "metal_shim.h".}
  proc metal_attention_fused_backward_c(kernelSrc: cstring, q, k, v, s_matrix, dy: ptr float32,
                                         dq, dk, dv: ptr float32, B, H, S, D: cint, scale: float32): cint
    {.importc: "metal_attention_fused_backward", header: "metal_shim.h".}

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
    result.c1 = newSeq[float32](m1 * n1)
    result.c2 = newSeq[float32](m2 * n2)
    var a1v = a1; var b1v = b1
    var a2v = a2; var b2v = b2
    let ok = metal_matmul2_c(kMetalKernelSrc.cstring,
      addr a1v[0], addr b1v[0], addr result.c1[0], cint(m1), cint(k1), cint(n1),
      addr a2v[0], addr b2v[0], addr result.c2[0], cint(m2), cint(k2), cint(n2))
    if ok == 0:
      raise newException(CatchableError, "metal_matmul2 failed")

  proc metalMatmulQ4*(a: seq[float32], wq: seq[uint8], scales, zeros: seq[float32],
                       m, k, n, groupSize, nGroupsPerRow: int): seq[float32] =
    ## Matmul truc tiep tren weight int4-asymmetric da pack, khong dequant ra
    ## fp32 tren CPU truoc (xem matmul_q4_kernel trong vecop_matmul.metal va
    ## quant.nim/dequantizeTensorTransposed cho dinh nghia layout chinh xac).
    result = newSeq[float32](m * n)
    var aVar = a
    var wqVar = wq
    var scalesVar = scales
    var zerosVar = zeros
    let ok = metal_matmul_q4_c(kMetalKernelSrc.cstring,
      addr aVar[0], addr wqVar[0], addr scalesVar[0], addr zerosVar[0], addr result[0],
      cint(m), cint(k), cint(n), cint(groupSize), cint(nGroupsPerRow))
    if ok == 0:
      raise newException(CatchableError, "metal_matmul_q4 failed")

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

  proc metalApflu*(x: seq[float32], alpha, beta: float32): seq[float32] =
    if alpha != 0.1'f32 or beta != 1.0'f32:
      raise newException(CatchableError,
        "metalApflu: kernel Metal hien chi ho tro alpha=0.1, beta=1.0")
    let n = x.len
    result = newSeq[float32](n)
    var xVar = x
    let ok = metal_activation_c(kMetalKernelSrc.cstring, 3.cint, addr xVar[0], addr result[0], cint(n))
    if ok == 0:
      raise newException(CatchableError, "metal_activation(apflu) failed")

  proc metalApfluBackward*(x, dy: seq[float32], alpha, beta: float32): seq[float32] =
    if alpha != 0.1'f32 or beta != 1.0'f32:
      raise newException(CatchableError,
        "metalApfluBackward: kernel Metal hien chi ho tro alpha=0.1, beta=1.0")
    let n = x.len
    result = newSeq[float32](n)
    var xVar = x; var dyVar = dy
    let ok = metal_activation_backward_c(kMetalKernelSrc.cstring, 3.cint, addr xVar[0], addr dyVar[0], addr result[0], cint(n))
    if ok == 0:
      raise newException(CatchableError, "metal_activation_backward(apflu) failed")

  proc metalLayernormBackward*(dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
    var dx = newSeq[float32](rows * cols)
    var dgamma = newSeq[float32](cols)
    var dbeta = newSeq[float32](cols)
    var dyVar = dy; var xVar = x; var gammaVar = gamma; var betaVar = beta
    let ok = metal_layernorm_backward_c(kMetalKernelSrc.cstring,
      addr dyVar[0], addr xVar[0], addr gammaVar[0], addr betaVar[0],
      addr dx[0], addr dgamma[0], addr dbeta[0], cint(rows), cint(cols), eps)
    if ok == 0:
      raise newException(CatchableError, "metal_layernorm_backward failed")
    return (dx, dgamma, dbeta)

  # === THÊM MỚI: Attention forward ===
  proc metalAttentionFused*(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
    var o = newSeq[float32](B * H * S * D)
    var sMatrix = newSeq[float32](B * H * S * S)
    var qVar = q; var kVar = k; var vVar = v
    let ok = metal_attention_fused_c(kMetalKernelSrc.cstring,
      addr qVar[0], addr kVar[0], addr vVar[0], addr o[0], addr sMatrix[0],
      cint(B), cint(H), cint(S), cint(D), scale)
    if ok == 0:
      raise newException(CatchableError, "metal_attention_fused failed")
    return (o, sMatrix)

  # === THÊM MỚI: Attention backward ===
  proc metalAttentionFusedBackward*(q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
    var dq = newSeq[float32](B * H * S * D)
    var dk = newSeq[float32](B * H * S * D)
    var dv = newSeq[float32](B * H * S * D)
    var qVar = q; var kVar = k; var vVar = v; var sVar = s_matrix; var dyVar = dy
    let ok = metal_attention_fused_backward_c(kMetalKernelSrc.cstring,
      addr qVar[0], addr kVar[0], addr vVar[0], addr sVar[0], addr dyVar[0],
      addr dq[0], addr dk[0], addr dv[0], cint(B), cint(H), cint(S), cint(D), scale)
    if ok == 0:
      raise newException(CatchableError, "metal_attention_fused_backward failed")
    return (dq, dk, dv)

  # === API resident: gộp nhiều op vào 1 MTLCommandBuffer, chỉ commit+wait
  # 1 lần ở sessionEnd() thay vì mỗi matmul/softmax 1 lần riêng. Dùng cho
  # decode loop (mỗi token) để giảm overhead dispatch trên iGPU yếu.
  type MetalBufferHandle* = pointer

  proc metal_session_begin_c(kernelSrc: cstring): cint {.importc: "metal_session_begin", header: "metal_shim.h".}
  proc metal_upload_c(data: ptr float32, n: cint): MetalBufferHandle {.importc: "metal_upload", header: "metal_shim.h".}
  proc metal_alloc_scratch_c(n: cint): MetalBufferHandle {.importc: "metal_alloc_scratch", header: "metal_shim.h".}
  proc metal_matmul_enc_c(a, b, c: MetalBufferHandle, m, k, n: cint): cint {.importc: "metal_matmul_enc", header: "metal_shim.h".}
  proc metal_softmax_enc_c(x, y: MetalBufferHandle, rows, cols: cint): cint {.importc: "metal_softmax_enc", header: "metal_shim.h".}
  proc metal_session_end_c(): cint {.importc: "metal_session_end", header: "metal_shim.h".}
  proc metal_buffer_read_c(h: MetalBufferHandle, outData: ptr float32, n: cint): cint {.importc: "metal_buffer_read", header: "metal_shim.h".}
  proc metal_buffer_free_c(h: MetalBufferHandle) {.importc: "metal_buffer_free", header: "metal_shim.h".}

  proc metalSessionBegin*(): bool =
    metal_session_begin_c(kMetalKernelSrc.cstring) != 0

  proc metalSessionUpload*(data: seq[float32]): MetalBufferHandle =
    var d = data
    metal_upload_c(addr d[0], cint(d.len))

  proc metalSessionAllocScratch*(n: int): MetalBufferHandle =
    metal_alloc_scratch_c(cint(n))

  proc metalSessionMatmulEnc*(a, b, c: MetalBufferHandle, m, k, n: int): bool =
    metal_matmul_enc_c(a, b, c, cint(m), cint(k), cint(n)) != 0

  proc metalSessionSoftmaxEnc*(x, y: MetalBufferHandle, rows, cols: int): bool =
    metal_softmax_enc_c(x, y, cint(rows), cint(cols)) != 0

  proc metalSessionEnd*(): bool =
    metal_session_end_c() != 0

  proc metalSessionRead*(h: MetalBufferHandle, n: int): seq[float32] =
    result = newSeq[float32](n)
    let ok = metal_buffer_read_c(h, addr result[0], cint(n))
    if ok == 0:
      raise newException(CatchableError, "metal_buffer_read failed")

  proc metalSessionFree*(h: MetalBufferHandle) =
    metal_buffer_free_c(h)

else:
  type MetalBufferHandle* = pointer
  proc metalSessionBegin*(): bool = false
  proc metalSessionUpload*(data: seq[float32]): MetalBufferHandle = nil
  proc metalSessionAllocScratch*(n: int): MetalBufferHandle = nil
  proc metalSessionMatmulEnc*(a, b, c: MetalBufferHandle, m, k, n: int): bool = false
  proc metalSessionSoftmaxEnc*(x, y: MetalBufferHandle, rows, cols: int): bool = false
  proc metalSessionEnd*(): bool = false
  proc metalSessionRead*(h: MetalBufferHandle, n: int): seq[float32] = @[]
  proc metalSessionFree*(h: MetalBufferHandle) = discard

  proc metalAvailable*(): bool = false
  proc metalVecOp*(op: string, a, b: seq[float32]): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalMatmul2*(a1, b1: seq[float32], m1, k1, n1: int,
                      a2, b2: seq[float32], m2, k2, n2: int):
                      tuple[c1, c2: seq[float32]] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalMatmulQ4*(a: seq[float32], wq: seq[uint8], scales, zeros: seq[float32],
                       m, k, n, groupSize, nGroupsPerRow: int): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalActivation*(op: string, x: seq[float32]): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalApflu*(x: seq[float32], alpha, beta: float32): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalApfluBackward*(x, dy: seq[float32], alpha, beta: float32): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalLayernormBackward*(dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalAttentionFused*(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalAttentionFusedBackward*(q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")