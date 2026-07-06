## backend.nim
## Lớp chọn backend tính toán cho Nimformer: cpu / metal (Apple GPU) / cuda
## (NVIDIA GPU, cuBLAS + kernel .cu tự viết).
##
## VÌ SAO FILE NÀY NHỎ: kiểm tra nimformer.nim cho thấy `ctx: MetalContext`
## được truyền xuyên suốt rất nhiều hàm forward/backward, nhưng THỰC SỰ chỉ
## có Linear.forward/backward gọi ra GPU (qua metalMatmul/metalMatmul2) —
## relu/sigmoid/tanh/softmax/layernorm/embedding trong nimformer.nim đều đã
## là CPU thuần Nim rồi. Nên lớp trừu tượng chỉ cần bọc đúng 2 hàm đó là đủ
## để nimformer.nim chạy được trên cả 3 backend mà KHÔNG cần sửa logic gì
## thêm trong nimformer.nim, chỉ đổi tên kiểu/tên hàm (xem README_BUILD).
##
## CÁCH CHỌN BACKEND (cả 2 như đã chốt):
##   1) Ép cứng lúc compile:  nim c -d:backend=cpu   test_nimformer.nim
##                            nim c -d:backend=metal test_nimformer.nim   (macOS)
##                            nim c -d:backend=cuda  -d:withCuda ...      (xem Makefile target `cuda`)
##   2) Không set (mặc định "auto") -> tự dò lúc RUNTIME: thử CUDA trước
##      (nếu binary có build CUDA), rồi Metal (nếu macOS), cuối cùng CPU.
##   Có thể ép tay lúc runtime bằng cách gọi newBackend("cpu"/"metal"/"cuda")
##   thẳng thay vì dùng default của define lúc compile.

# ── Biên dịch điều kiện: chỉ import metal_ai trên macOS (cần ObjC/Metal.framework),
#    chỉ import cuda_ai khi build với -d:withCuda (cần nvcc + cuBLAS, xem Makefile). ──
when defined(macosx) and not defined(noMetal):
  import metal_ai
  const hasMetal* = true
else:
  const hasMetal* = false

when defined(withCuda):
  import cuda_ai
  const hasCuda* = true
else:
  const hasCuda* = false

# -d:backend=cpu|metal|cuda|auto — mặc định "auto" (tự dò lúc runtime).
const backend* {.strdefine.} = "auto"

type
  BackendKind* = enum
    bkCpu = "cpu"
    bkMetal = "metal"
    bkCuda = "cuda"

  Backend* = object
    case kind*: BackendKind
    of bkMetal:
      when hasMetal:
        mtl*: MetalContext
      else:
        discard
    of bkCuda:
      when hasCuda:
        cu*: CudaContext
      else:
        discard
    of bkCpu:
      discard

# ─────────────────────────────────────────────────────────────
# CPU reference implementation — dùng khi kind == bkCpu, cũng là "sự thật"
# để so sánh sai số với kết quả GPU nếu cần (giống test_nimformer.nim đã
# làm với các dtype lượng tử hoá).
# ─────────────────────────────────────────────────────────────

proc cpuMatmul(a: openArray[float32], M, K: int,
               b: openArray[float32], K2, N: int): seq[float32] =
  assert K == K2
  assert a.len == M * K and b.len == K * N
  result = newSeq[float32](M * N)
  for i in 0 ..< M:
    for k in 0 ..< K:
      let aik = a[i * K + k]
      if aik == 0'f32: continue
      for j in 0 ..< N:
        result[i * N + j] += aik * b[k * N + j]

# ─────────────────────────────────────────────────────────────
# Dò backend khả dụng lúc runtime
# ─────────────────────────────────────────────────────────────

proc resolveBackendKind(want: string): BackendKind =
  case want
  of "cpu": bkCpu
  of "metal":
    when hasMetal: bkMetal
    else: raise newException(ValueError,
      "Backend 'metal' được yêu cầu nhưng binary không build trên macOS " &
      "(hoặc build với -d:noMetal) — xem README_BUILD.md")
  of "cuda":
    when hasCuda: bkCuda
    else: raise newException(ValueError,
      "Backend 'cuda' được yêu cầu nhưng binary chưa build với -d:withCuda " &
      "(xem target `make cuda` trong Makefile)")
  else:
    # "auto" hoặc giá trị lạ -> tự dò: CUDA trước, Metal sau, CPU cuối cùng.
    when hasCuda:
      if cudaDeviceAvailable(): return bkCuda
    when hasMetal:
      if metalDeviceAvailable(): return bkMetal
    bkCpu

proc newBackend*(force: string = backend): Backend =
  ## force: "cpu" | "metal" | "cuda" | "auto". Mặc định lấy từ define lúc
  ## compile (-d:backend=...), truyền tay để ép cứng lúc runtime.
  let kind = resolveBackendKind(force)
  stderr.writeLine "== Backend đã chọn: " & $kind & " =="
  case kind
  of bkCpu: result = Backend(kind: bkCpu)
  of bkMetal:
    when hasMetal:
      result = Backend(kind: bkMetal, mtl: newMetalContext())
    else:
      raise newException(ValueError, "unreachable: hasMetal=false")
  of bkCuda:
    when hasCuda:
      result = Backend(kind: bkCuda, cu: newCudaContext())
    else:
      raise newException(ValueError, "unreachable: hasCuda=false")

proc closeBackend*(b: Backend) =
  case b.kind
  of bkMetal:
    when hasMetal: closeMetalContext(b.mtl)
  of bkCuda:
    when hasCuda: closeCudaContext(b.cu)
  of bkCpu: discard

# ─────────────────────────────────────────────────────────────
# API hợp nhất — thay thế trực tiếp metalMatmul/metalMatmul2 trong
# nimformer.nim (cùng chữ ký, chỉ đổi `ctx: MetalContext` -> `ctx: Backend`).
# ─────────────────────────────────────────────────────────────

proc beMatmul*(ctx: Backend, a: openArray[float32], M, K: int,
                b: openArray[float32], K2, N: int): seq[float32] =
  case ctx.kind
  of bkCpu: cpuMatmul(a, M, K, b, K2, N)
  of bkMetal:
    when hasMetal: metalMatmul(ctx.mtl, a, M, K, b, K2, N)
    else: raise newException(ValueError, "unreachable: hasMetal=false")
  of bkCuda:
    when hasCuda: cudaMatmul(ctx.cu, a, M, K, b, K2, N)
    else: raise newException(ValueError, "unreachable: hasCuda=false")

proc beMatmul2*(ctx: Backend,
                 a1: openArray[float32], M1, K1: int, b1: openArray[float32], K1b, N1: int,
                 a2: openArray[float32], M2, K2: int, b2: openArray[float32], K2b, N2: int):
                 tuple[y1, y2: seq[float32]] =
  ## Metal: gộp 2 dispatch độc lập vào 1 command buffer (metalMatmul2 gốc).
  ## CUDA: cuBLAS + CUDA stream tự pipeline hoá nội bộ tốt, nên chỉ cần gọi
  ## cudaMatmul() 2 lần tuần tự là đủ (không cần gộp thủ công như Metal).
  ## CPU: đơn giản là 2 lần cpuMatmul().
  case ctx.kind
  of bkCpu:
    (cpuMatmul(a1, M1, K1, b1, K1b, N1), cpuMatmul(a2, M2, K2, b2, K2b, N2))
  of bkMetal:
    when hasMetal: metalMatmul2(ctx.mtl, a1, M1, K1, b1, K1b, N1, a2, M2, K2, b2, K2b, N2)
    else: raise newException(ValueError, "unreachable: hasMetal=false")
  of bkCuda:
    when hasCuda:
      (cudaMatmul(ctx.cu, a1, M1, K1, b1, K1b, N1), cudaMatmul(ctx.cu, a2, M2, K2, b2, K2b, N2))
    else: raise newException(ValueError, "unreachable: hasCuda=false")