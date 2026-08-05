# metal_backend.nim - Metal GPU backend wrapper (chỉ biên dịch trên macOS).
import std/os

when defined(macosx):
  const metalDir = currentSourcePath().parentDir()
  {.passC: "-I" & metalDir.}
  {.passC: "-fobjc-arc".}
  {.passL: "-framework Metal -framework Foundation -framework MetalPerformanceShaders".}
  {.compile: "metal_shim.m".}

  const kMetalKernelSrc = staticRead(metalDir / "kernels" / "vecop_matmul.metal")

  proc metal_available_c(): cint {.importc: "metal_available", header: "metal_shim.h".}
  proc metal_vecop_c(kernelSrc: cstring, op: cint, a, b: ptr float32, c: ptr float32, n: cint): cint
    {.importc: "metal_vecop", header: "metal_shim.h".}
  proc metal_matmul_c(kernelSrc: cstring, a, b: ptr float32, c: ptr float32, m, k, n: cint): cint
    {.importc: "metal_matmul", header: "metal_shim.h".}
  proc metal_matmul_mps_c(a, b: ptr float32, c: ptr float32, m, k, n: cint): cint
    {.importc: "metal_matmul_mps", header: "metal_shim.h".}
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

  # ─────────────────────────────────────────────────────────────────────
  # CÔNG TẮC BẬT/TẮT MPS - sửa trực tiếp giá trị này để chọn đường matmul.
  #   bmUseMps    -> luôn dùng MPSMatrixMultiplication (fallback tự động
  #                  về kernel tay viết nếu máy không hỗ trợ / lỗi).
  #   bmUseKernel -> luôn dùng kernel tay viết matmul_kernel_t{8,16,32}
  #                  (bỏ qua MPS hoàn toàn - dùng khi cần so sánh benchmark
  #                  hoặc nghi ngờ MPS cho kết quả sai trên 1 GPU cụ thể).
  # ─────────────────────────────────────────────────────────────────────
  type BbMatmulBackend = enum
    bmUseMps, bmUseKernel

  const bbMatmulBackend: BbMatmulBackend = bmUseKernel   # <<< SỬA Ở ĐÂY
  # SUA: đổi mặc định từ bmUseMps -> bmUseKernel. Đo thực tế trên Iris Plus
  # 640: MPS chậm HƠN kernel tay viết cho workload decode LLM (16942ms/token
  # so với ~7000-7800ms/token trước đó). Lý do: mỗi bước decode chỉ nhân
  # M=1 (1 token) - thực chất là matrix-VECTOR chứ không phải GEMM lớn, mà
  # MPS được tối ưu cho ma trận lớn/batch lớn. Trên GPU yếu, overhead setup
  # nội bộ của MPS (chọn thuật toán, validate descriptor...) cho M=1 lớn
  # hơn cả phần tính toán thật -> phản tác dụng. Vẫn giữ code MPS lại (bật
  # bmUseMps để so sánh nếu cần, hoặc dùng cho prefill M lớn sau này).

  var gMpsFailedOnce = false  # SUA: nếu MPS lỗi 1 lần (vd. GPU không hỗ trợ),
                              # đừng thử lại mỗi token - tốn 1 lần gọi native
                              # thất bại mỗi bước. Fallback hẳn về kernel tay
                              # viết cho phần đời còn lại của process.

  proc metalMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
    result = newSeq[float32](m * n)
    var aVar = a
    var bVar = b
    # SUA (#3 - dùng MPS): ưu tiên MPSMatrixMultiplication trước - Apple tự
    # chọn thuật toán tối ưu theo đúng GPU đang chạy, thường nhanh hơn đáng
    # kể so với matmul_kernel_t{8,16,32} tay viết, và không cần lo threadgroup
    # limit / tile size gì nữa. Fallback về kernel tay viết (đã có tile động
    # ở #A) nếu MPS không khả dụng trên máy này hoặc lỗi. Có thể tắt hẳn
    # bằng cách đổi bbMatmulBackend ở trên thành bmUseKernel.
    if bbMatmulBackend == bmUseMps and not gMpsFailedOnce:
      let okMps = metal_matmul_mps_c(addr aVar[0], addr bVar[0], addr result[0], cint(m), cint(k), cint(n))
      if okMps != 0:
        return
      gMpsFailedOnce = true
      stderr.write("[metal] MPS matmul khong kha dung/loi - chuyen han ve kernel tay viet cho phan con lai cua phien\n")
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
  proc metal_matmul_nt_enc_c(a, b, c: MetalBufferHandle, m, k, n: cint): cint {.importc: "metal_matmul_nt_enc", header: "metal_shim.h".}
  proc metal_init_c(kernelSrc: cstring): cint {.importc: "metal_init", header: "metal_shim.h".}
  proc metal_upload_bytes_c(data: ptr uint8, n: cint): MetalBufferHandle {.importc: "metal_upload_bytes", header: "metal_shim.h".}
  proc metal_upload_private_c(data: ptr float32, n: cint): MetalBufferHandle {.importc: "metal_upload_private", header: "metal_shim.h".}
  proc metal_upload_bytes_private_c(data: ptr uint8, n: cint): MetalBufferHandle {.importc: "metal_upload_bytes_private", header: "metal_shim.h".}
  proc metal_matmul_q4_enc_c(a, wq, scales, zeros, c: MetalBufferHandle, m, k, n, groupSize, nGroupsPerRow: cint): cint
    {.importc: "metal_matmul_q4_enc", header: "metal_shim.h".}
  proc metal_copy_into_enc_c(src, dst: MetalBufferHandle, dstOffsetElems, n: cint): cint {.importc: "metal_copy_into_enc", header: "metal_shim.h".}
  proc metal_softmax_enc_c(x, y: MetalBufferHandle, rows, cols: cint): cint {.importc: "metal_softmax_enc", header: "metal_shim.h".}
  proc metal_rmsnorm_enc_c(x, gamma, y: MetalBufferHandle, rows, cols: cint, eps: cfloat): cint
    {.importc: "metal_rmsnorm_enc", header: "metal_shim.h".}
  proc metal_layernorm_enc_c(x, gamma, beta, y: MetalBufferHandle, rows, cols: cint, eps: cfloat): cint
    {.importc: "metal_layernorm_enc", header: "metal_shim.h".}
  proc metal_embedding_lookup_enc_c(table, indices, y: MetalBufferHandle, vocab, dim, numIndices: cint): cint
    {.importc: "metal_embedding_lookup_enc", header: "metal_shim.h".}
  proc metal_attention_fused_enc_c(q, k, v, o, sMatrix: MetalBufferHandle, B, H, S, D: cint, scale: cfloat): cint
    {.importc: "metal_attention_fused_enc", header: "metal_shim.h".}
  proc metal_session_end_c(): cint {.importc: "metal_session_end", header: "metal_shim.h".}
  proc metal_buffer_read_c(h: MetalBufferHandle, outData: ptr float32, n: cint): cint {.importc: "metal_buffer_read", header: "metal_shim.h".}
  proc metal_buffer_free_c(h: MetalBufferHandle) {.importc: "metal_buffer_free", header: "metal_shim.h".}

  proc metalInit*(): bool =
    ## Gọi ensureMetalInit() tường minh (dispatch_once bên trong - gọi nhiều
    ## lần vẫn rẻ). PHẢI gọi trước metalUploadBytes/metalSessionUpload nếu
    ## chưa có metalSessionBegin nào chạy trước đó (đúng case resident-upload
    ## lúc load model, trước khi vào decode loop).
    metal_init_c(kMetalKernelSrc.cstring) != 0

  proc metalSessionBegin*(): bool =
    metal_session_begin_c(kMetalKernelSrc.cstring) != 0

  proc metalSessionUpload*(data: seq[float32]): MetalBufferHandle =
    discard metalInit()  # SUA: an toàn khi gọi resident-upload TRƯỚC session begin đầu tiên
    var d = data
    metal_upload_c(addr d[0], cint(d.len))

  proc metalSessionAllocScratch*(n: int): MetalBufferHandle =
    metal_alloc_scratch_c(cint(n))

  proc metalSessionMatmulEnc*(a, b, c: MetalBufferHandle, m, k, n: int): bool =
    metal_matmul_enc_c(a, b, c, cint(m), cint(k), cint(n)) != 0

  proc metalSessionMatmulNTEnc*(a, b, c: MetalBufferHandle, m, k, n: int): bool =
    ## C(m,n) = A(m,k) @ B(n,k)^T - B GIU NGUYEN layout goc [n,k] (khong
    ## transpose CPU truoc). Dung cho attention decode (K/V-cache).
    metal_matmul_nt_enc_c(a, b, c, cint(m), cint(k), cint(n)) != 0

  proc metalUploadBytes*(data: seq[uint8]): MetalBufferHandle =
    discard metalInit()
    var d = data
    metal_upload_bytes_c(addr d[0], cint(d.len))

  proc metalUploadPrivate*(data: seq[float32]): MetalBufferHandle =
    ## Bản Private-storage của metalSessionUpload - CHỈ dùng cho resident
    ## weight fp32 (int8/int4-symmetric/tied-embedding qua beResidentUploadFp32)
    ## mà CPU không đọc/ghi lại sau lúc upload.
    discard metalInit()
    if data.len == 0: return nil
    var d = data
    metal_upload_private_c(addr d[0], cint(d.len))

  proc metalUploadBytesPrivate*(data: seq[uint8]): MetalBufferHandle =
    ## Bản Private-storage của metalUploadBytes - dùng cho wq (Q4 packed) resident.
    discard metalInit()
    if data.len == 0: return nil
    var d = data
    metal_upload_bytes_private_c(addr d[0], cint(d.len))

  proc metalSessionMatmulQ4Enc*(a, wq, scales, zeros, c: MetalBufferHandle,
                                 m, k, n, groupSize, nGroupsPerRow: int): bool =
    metal_matmul_q4_enc_c(a, wq, scales, zeros, c,
      cint(m), cint(k), cint(n), cint(groupSize), cint(nGroupsPerRow)) != 0

  proc metalSessionCopyIntoEnc*(src, dst: MetalBufferHandle, dstOffsetElems, n: int): bool =
    metal_copy_into_enc_c(src, dst, cint(dstOffsetElems), cint(n)) != 0

  proc metalSessionSoftmaxEnc*(x, y: MetalBufferHandle, rows, cols: int): bool =
    metal_softmax_enc_c(x, y, cint(rows), cint(cols)) != 0

  proc metalSessionRmsnormEnc*(x, gamma, y: MetalBufferHandle, rows, cols: int, eps: float32): bool =
    metal_rmsnorm_enc_c(x, gamma, y, cint(rows), cint(cols), cfloat(eps)) != 0

  proc metalSessionLayernormEnc*(x, gamma, beta, y: MetalBufferHandle, rows, cols: int, eps: float32): bool =
    metal_layernorm_enc_c(x, gamma, beta, y, cint(rows), cint(cols), cfloat(eps)) != 0

  proc metalSessionEmbeddingLookupEnc*(table, indices, y: MetalBufferHandle, vocab, dim, numIndices: int): bool =
    metal_embedding_lookup_enc_c(table, indices, y, cint(vocab), cint(dim), cint(numIndices)) != 0

  proc metalSessionAttentionFusedEnc*(q, k, v, o, sMatrix: MetalBufferHandle, B, H, S, D: int, scale: float32): bool =
    metal_attention_fused_enc_c(q, k, v, o, sMatrix, cint(B), cint(H), cint(S), cint(D), cfloat(scale)) != 0

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
  proc metalSessionMatmulNTEnc*(a, b, c: MetalBufferHandle, m, k, n: int): bool = false
  proc metalInit*(): bool = false
  proc metalUploadBytes*(data: seq[uint8]): MetalBufferHandle = nil
  proc metalUploadPrivate*(data: seq[float32]): MetalBufferHandle = nil
  proc metalUploadBytesPrivate*(data: seq[uint8]): MetalBufferHandle = nil
  proc metalSessionMatmulQ4Enc*(a, wq, scales, zeros, c: MetalBufferHandle,
                                 m, k, n, groupSize, nGroupsPerRow: int): bool = false
  proc metalSessionCopyIntoEnc*(src, dst: MetalBufferHandle, dstOffsetElems, n: int): bool = false
  proc metalSessionSoftmaxEnc*(x, y: MetalBufferHandle, rows, cols: int): bool = false
  proc metalSessionRmsnormEnc*(x, gamma, y: MetalBufferHandle, rows, cols: int, eps: float32): bool = false
  proc metalSessionLayernormEnc*(x, gamma, beta, y: MetalBufferHandle, rows, cols: int, eps: float32): bool = false
  proc metalSessionEmbeddingLookupEnc*(table, indices, y: MetalBufferHandle, vocab, dim, numIndices: int): bool = false
  proc metalSessionAttentionFusedEnc*(q, k, v, o, sMatrix: MetalBufferHandle, B, H, S, D: int, scale: float32): bool = false
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