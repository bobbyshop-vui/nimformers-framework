## metal_ai.nim
## Wrapper Nim gọi Metal qua metal_bridge.h/.m + logic CustomFloat/APF thuần Nim.
##
## Dùng dispatch generic (mtlDispatch), nạp đúng file metal_kernels.metal thật
## (add, matmul, relu/sigmoid/tanh activation, softmax, layernorm,
## embedding_lookup) + kernel customfloat_encode/decode, rồi có wrapper Nim
## cho từng kernel: metalAdd, metalMatmul, metalRelu, metalSigmoid, metalTanh,
## metalSoftmax, metalLayernorm, metalEmbeddingLookup, customfloatEncodeGpu/
## customfloatDecodeGpu, apfCastForTrainingGpu.

import std/[os, strformat, tables]
import customfloat

{.passC: "-fobjc-arc".}
{.passL: "-framework Metal -framework Foundation".}
{.compile: "metal_bridge.m".}

type
  MTLDeviceRef {.importc: "MTLDeviceRef", header: "metal_bridge.h".} = pointer
  MTLQueueRef {.importc: "MTLQueueRef", header: "metal_bridge.h".} = pointer
  MTLBufferRef {.importc: "MTLBufferRef", header: "metal_bridge.h".} = pointer
  MTLLibraryRef {.importc: "MTLLibraryRef", header: "metal_bridge.h".} = pointer
  MTLPipelineRef {.importc: "MTLPipelineRef", header: "metal_bridge.h".} = pointer
  MTLCmdBufRef {.importc: "MTLCmdBufRef", header: "metal_bridge.h".} = pointer
  MTLEncoderRef {.importc: "MTLEncoderRef", header: "metal_bridge.h".} = pointer

proc mtlCreateDevice(): MTLDeviceRef {.importc: "mtl_create_device", header: "metal_bridge.h".}
proc mtlCreateQueue(d: MTLDeviceRef): MTLQueueRef {.importc: "mtl_create_queue", header: "metal_bridge.h".}
proc mtlNewBuffer(d: MTLDeviceRef, length: csize_t): MTLBufferRef {.importc: "mtl_new_buffer", header: "metal_bridge.h".}
proc mtlBufferContents(b: MTLBufferRef): pointer {.importc: "mtl_buffer_contents", header: "metal_bridge.h".}
proc mtlBufferLength(b: MTLBufferRef): csize_t {.importc: "mtl_buffer_length", header: "metal_bridge.h".}
proc mtlCompileLibrary(d: MTLDeviceRef, src: cstring, errOut: ptr cstring): MTLLibraryRef {.importc: "mtl_compile_library", header: "metal_bridge.h".}
proc mtlGetPipeline(d: MTLDeviceRef, lib: MTLLibraryRef, fnName: cstring): MTLPipelineRef {.importc: "mtl_get_pipeline", header: "metal_bridge.h".}
proc mtlDispatchRaw(q: MTLQueueRef, p: MTLPipelineRef, bufs: ptr MTLBufferRef, n: cint,
                     gx, gy, gz, tx, ty, tz: csize_t) {.importc: "mtl_dispatch", header: "metal_bridge.h".}
proc mtlRelease(r: pointer) {.importc: "mtl_release", header: "metal_bridge.h".}

# API gộp nhiều dispatch ĐỘC LẬP vào 1 command buffer (1 wait duy nhất) —
# xem giải thích trong metal_bridge.h. Chỉ dùng khi giữa các dispatch KHÔNG
# có bước xử lý CPU nào cần đọc kết quả GPU giữa chừng.
proc mtlCmdBufCreate(q: MTLQueueRef): MTLCmdBufRef {.importc: "mtl_command_buffer_create", header: "metal_bridge.h".}
proc mtlEncoderCreate(c: MTLCmdBufRef): MTLEncoderRef {.importc: "mtl_encoder_create", header: "metal_bridge.h".}
proc mtlEncoderDispatchRaw(e: MTLEncoderRef, p: MTLPipelineRef, bufs: ptr MTLBufferRef, n: cint,
                           gx, gy, gz, tx, ty, tz: csize_t) {.importc: "mtl_encoder_dispatch", header: "metal_bridge.h".}
proc mtlEncoderEnd(e: MTLEncoderRef) {.importc: "mtl_encoder_end", header: "metal_bridge.h".}
proc mtlCmdBufCommitAndWait(c: MTLCmdBufRef) {.importc: "mtl_command_buffer_commit_and_wait", header: "metal_bridge.h".}

# ─────────────────────────────────────────────────────────────
# Metal kernel source: nạp metal_kernels.metal THẬT (add, matmul,
# relu/sigmoid/tanh, softmax, layernorm, embedding_lookup) + kernel
# customfloat_encode/decode, giống hệt cách _get_pipeline() bản Python
# đọc file rồi nối thêm CUSTOMFLOAT_METAL_KERNEL_SRC.
# staticRead nhúng nội dung lúc COMPILE (không cần file .metal đi kèm
# lúc chạy binary) — nếu muốn đọc lại lúc runtime thay vì compile-time,
# đổi thành: readFile("metal_kernels.metal") trong newMetalContext().
# ─────────────────────────────────────────────────────────────
const customFloatKernelSrc = """
kernel void customfloat_encode(
    device const float* in_buf      [[buffer(0)]],
    device uchar* out_buf           [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    constant uint& exponent_bits    [[buffer(3)]],
    constant uint& mantissa_bits    [[buffer(4)]],
    constant uint& itemsize         [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) return;
    uint bits32 = as_type<uint>(in_buf[id]);
    uint sign   = (bits32 >> 31) & 0x1;
    uint exp32  = (bits32 >> 23) & 0xFF;
    uint mant32 = bits32 & 0x7FFFFF;

    int  bias    = (1 << (exponent_bits - 1)) - 1;
    uint max_exp = (1u << exponent_bits) - 1u;
    int  real_exp = int(exp32) - 127;
    bool is_zero    = (exp32 == 0 && mant32 == 0);
    bool is_inf_nan = (exp32 == 255);

    int new_exp = real_exp + bias;
    ulong mant;
    if (mantissa_bits <= 23) mant = (ulong)(mant32 >> (23 - mantissa_bits));
    else                     mant = (ulong)(mant32) << (mantissa_bits - 23);

    ulong top_shift = (ulong)(exponent_bits + mantissa_bits);
    ulong new_exp_u = (ulong)clamp(new_exp, 0, (int)max_exp);
    ulong packed = (ulong(sign) << top_shift) | (new_exp_u << mantissa_bits) | mant;

    if (is_inf_nan) {
        packed = (ulong(sign) << top_shift) | (ulong(max_exp) << mantissa_bits);
    } else if (is_zero || new_exp <= 0) {
        packed = (ulong(sign) << top_shift);
    } else if (new_exp >= int(max_exp)) {
        packed = (ulong(sign) << top_shift) | (ulong(max_exp) << mantissa_bits);
    }

    for (uint i = 0; i < itemsize; i++) {
        out_buf[id * itemsize + i] = uchar((packed >> (8 * i)) & 0xFF);
    }
}

kernel void customfloat_decode(
    device const uchar* in_buf      [[buffer(0)]],
    device float* out_buf           [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    constant uint& exponent_bits    [[buffer(3)]],
    constant uint& mantissa_bits    [[buffer(4)]],
    constant uint& itemsize         [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) return;
    ulong packed = 0;
    for (uint i = 0; i < itemsize; i++) {
        packed |= (ulong(in_buf[id * itemsize + i]) << (8 * i));
    }
    uint max_exp = (1u << exponent_bits) - 1u;
    int  bias    = (1 << (exponent_bits - 1)) - 1;
    ulong mant_mask = (mantissa_bits == 0) ? 0 : ((1ul << mantissa_bits) - 1);
    ulong top_shift = (ulong)(exponent_bits + mantissa_bits);

    uint  sign = uint((packed >> top_shift) & 1);
    uint  exp  = uint((packed >> mantissa_bits) & max_exp);
    ulong mant = packed & mant_mask;

    bool is_zero    = (exp == 0 && mant == 0);
    bool is_inf_nan = (exp == max_exp);

    int  real_exp = int(exp) - bias;
    uint exp32 = uint(clamp(real_exp + 127, 0, 255));
    uint mant32;
    if (mantissa_bits <= 23) mant32 = uint(mant) << (23 - mantissa_bits);
    else                     mant32 = uint(mant >> (mantissa_bits - 23));

    uint bits32 = (sign << 31) | (exp32 << 23) | mant32;
    if (is_zero)    bits32 = (sign << 31);
    if (is_inf_nan) bits32 = (sign << 31) | (0xFFu << 23);

    out_buf[id] = as_type<float>(bits32);
}
"""

const kernelSrc = staticRead("metal_kernels.metal") & "\n" & customFloatKernelSrc

# ─────────────────────────────────────────────────────────────
# MetalContext — tương đương self.device/self.queue/_shader_pipelines
# ─────────────────────────────────────────────────────────────

type
  MetalContext* = object
    device: MTLDeviceRef
    queue: MTLQueueRef
    library: MTLLibraryRef
    # Cache pipeline theo tên kernel — tương đương self._shader_pipelines (dict)
    # bên bản Python. `ref Table` để cache dùng chung/sống được dù MetalContext
    # được truyền/copy theo giá trị (device/queue/library chỉ là con trỏ nên
    # copy rẻ, nhưng cache PHẢI là 1 vùng nhớ DÙNG CHUNG cho mọi bản copy).
    pipelineCache: ref Table[string, MTLPipelineRef]
    # Buffer pool theo kích thước (byte) — free-list, KHÔNG phải cache dữ liệu.
    # LÝ DO: dù mtl_release() đã gọi đúng sau mỗi dispatch (releaseBufs), RAM
    # tiến trình vẫn phình dần theo từng step vì mtl_new_buffer() alloc MỘT
    # buffer MỚI (MTLResourceStorageModeShared -> backing store là vùng nhớ
    # mmap thật) ở MỌI upload/output của MỌI matmul/add/layernorm/softmax...
    # trong MỌI step — dealloc rồi lại alloc liên tục hàng nghìn buffer nhỏ
    # khiến bộ nhớ ảo của tiến trình (RSS) tăng dần, hiếm khi hệ điều hành trả
    # ngay về free list toàn cục dù object ObjC đã bị giải phóng đúng cách.
    # Giải pháp: giữ lại buffer đã "release" trong 1 free-list theo kích
    # thước, tái dùng ở lần cần buffer cùng size tiếp theo thay vì
    # mtl_new_buffer() mới — số buffer sống thật trong tiến trình sẽ hội tụ
    # về mức đỉnh (peak) rồi PHẲNG, không tăng vô hạn theo số step nữa.
    bufferPool: ref Table[uint64, seq[MTLBufferRef]]

proc newMetalContext*(): MetalContext =
  result.device = mtlCreateDevice()
  if result.device == nil:
    raise newException(IOError, "Không tìm thấy thiết bị Metal (cần macOS + GPU hỗ trợ Metal)")
  result.queue = mtlCreateQueue(result.device)
  var errC: cstring
  result.library = mtlCompileLibrary(result.device, kernelSrc.cstring, addr errC)
  if result.library == nil:
    raise newException(IOError, "Metal compile lỗi: " & $errC)
  new(result.pipelineCache)
  result.pipelineCache[] = initTable[string, MTLPipelineRef]()
  new(result.bufferPool)
  result.bufferPool[] = initTable[uint64, seq[MTLBufferRef]]()

proc poolGet(ctx: MetalContext, length: int): MTLBufferRef =
  ## Lấy buffer sẵn có trong pool đúng size nếu có, không thì mới
  ## mtl_new_buffer() — thay cho gọi mtl_new_buffer trực tiếp ở mọi nơi.
  let key = uint64(length)
  if ctx.bufferPool[].hasKey(key) and ctx.bufferPool[][key].len > 0:
    result = ctx.bufferPool[][key].pop()
  else:
    result = mtlNewBuffer(ctx.device, csize_t(length))

proc poolPut(ctx: MetalContext, buf: MTLBufferRef) =
  ## Thay cho mtl_release trực tiếp: trả buffer về pool theo size (đọc lại
  ## bằng mtl_buffer_length, không cần caller tự nhớ size) để lần dispatch
  ## sau tái dùng thay vì tạo mới.
  if buf == nil: return
  let key = uint64(mtlBufferLength(buf))
  if not ctx.bufferPool[].hasKey(key):
    ctx.bufferPool[][key] = @[]
  ctx.bufferPool[][key].add(buf)

proc closeMetalContext*(ctx: MetalContext) =
  ## Giải phóng THẬT SỰ toàn bộ buffer còn nằm trong pool — gọi 1 lần lúc
  ## chương trình kết thúc (không bắt buộc trong lúc train, chỉ để dọn sạch
  ## trước khi thoát thay vì để OS thu hồi khi process exit).
  for key, bucket in ctx.bufferPool[].mpairs:
    for b in bucket:
      mtlRelease(b)
    bucket.setLen(0)

proc pipeline(ctx: MetalContext, name: string): MTLPipelineRef =
  ## QUAN TRỌNG: mtl_get_pipeline() biên dịch + __bridge_retained 1
  ## MTLComputePipelineState MỚI mỗi lần gọi. TRƯỚC ĐÂY hàm này gọi thẳng
  ## mtl_get_pipeline() mỗi lần dispatch (mỗi matmul/add/layernorm/softmax...
  ## trong MỌI forward+backward) -> tạo pipeline mới liên tục, không bao giờ
  ## release -> đây là nguồn rò rỉ RAM CHÍNH khiến RAM vẫn phình dù buffer
  ## tạm đã được release (pipeline nặng hơn buffer nhiều và bị tạo lại ở tần
  ## suất cao hơn — mỗi dispatch, không phải mỗi buffer). Giờ cache lại theo
  ## tên kernel, chỉ compile 1 LẦN DUY NHẤT cho mỗi kernel trong suốt vòng
  ## đời MetalContext rồi tái dùng — không cần (và không nên) mtl_release
  ## pipeline đã cache vì nó còn sống tới khi chương trình kết thúc.
  if ctx.pipelineCache[].hasKey(name):
    return ctx.pipelineCache[][name]
  result = mtlGetPipeline(ctx.device, ctx.library, name.cstring)
  if result == nil:
    raise newException(IOError, &"Không tìm thấy kernel Metal: {name}")
  ctx.pipelineCache[][name] = result

# helper: alloc buffer GPU rồi copy dữ liệu vào (giống _to_gpu_static)
proc uploadBytes(ctx: MetalContext, data: openArray[uint8]): MTLBufferRef =
  result = ctx.poolGet(data.len)
  if data.len > 0:
    copyMem(mtlBufferContents(result), unsafeAddr data[0], data.len)

proc uploadF32(ctx: MetalContext, data: openArray[float32]): MTLBufferRef =
  result = ctx.poolGet(data.len * 4)
  if data.len > 0:
    copyMem(mtlBufferContents(result), unsafeAddr data[0], data.len * 4)

proc uploadU32(ctx: MetalContext, v: uint32): MTLBufferRef =
  var tmp = v
  result = ctx.poolGet(4)
  copyMem(mtlBufferContents(result), addr tmp, 4)

proc downloadF32(buf: MTLBufferRef, n: int): seq[float32] =
  result = newSeq[float32](n)
  if n > 0:
    copyMem(addr result[0], mtlBufferContents(buf), n * 4)

proc downloadU8(buf: MTLBufferRef, n: int): seq[uint8] =
  result = newSeq[uint8](n)
  if n > 0:
    copyMem(addr result[0], mtlBufferContents(buf), n)

proc dispatch(ctx: MetalContext, pipe: MTLPipelineRef, bufs: var seq[MTLBufferRef],
              gx, gy, gz, tx, ty, tz: int) =
  mtlDispatchRaw(ctx.queue, pipe, addr bufs[0], cint(bufs.len),
                 csize_t(gx), csize_t(gy), csize_t(gz),
                 csize_t(tx), csize_t(ty), csize_t(tz))

# Trả TOÀN BỘ buffer GPU tạm (input lẫn output) về pool sau khi đã dispatch +
# download kết quả về CPU xong — bắt buộc, nếu không mỗi lần gọi metalXxx sẽ
# rò rỉ thêm buffer mới. TRƯỚC ĐÂY hàm này gọi mtl_release() thẳng — vẫn
# đúng về mặt retain-count, nhưng vì mtl_new_buffer() alloc buffer MỚI liên
# tục ở lần dispatch kế tiếp, RSS tiến trình vẫn phình dần do vùng nhớ
# backing (MTLResourceStorageModeShared) bị cấp/hủy lặp lại thay vì tái
# dùng. Giờ trả về ctx.bufferPool bằng poolPut() thay vì mtl_release() thật.
proc releaseBufs(ctx: MetalContext, bufs: openArray[MTLBufferRef]) =
  for b in bufs:
    ctx.poolPut(b)

# ─────────────────────────────────────────────────────────────
# Wrapper cho TOÀN BỘ kernel trong metal_kernels.metal thật
# (add, matmul, relu_activation, sigmoid_activation, tanh_activation,
#  softmax, layernorm, embedding_lookup) — cùng pattern generic-dispatch.
# ─────────────────────────────────────────────────────────────

proc metalAdd*(ctx: MetalContext, a, b: openArray[float32]): seq[float32] =
  assert a.len == b.len
  let n = a.len
  var bufs = @[
    ctx.uploadF32(a),
    ctx.uploadF32(b),
    ctx.poolGet(n * 4),
    ctx.uploadU32(uint32(n)),
  ]
  ctx.dispatch(ctx.pipeline("add"), bufs, n, 1, 1, min(n, 256), 1, 1)
  result = downloadF32(bufs[2], n)
  releaseBufs(ctx, bufs)

proc metalMatmul*(ctx: MetalContext, a: openArray[float32], M, K: int,
                   b: openArray[float32], K2, N: int): seq[float32] =
  assert K == K2
  var bufs = @[
    ctx.uploadF32(a),
    ctx.uploadF32(b),
    ctx.poolGet(M * N * 4),
    ctx.uploadU32(uint32(M)),
    ctx.uploadU32(uint32(N)),
    ctx.uploadU32(uint32(K)),
  ]
  ctx.dispatch(ctx.pipeline("matmul"), bufs, M, N, 1, min(M, 16), min(N, 16), 1)
  result = downloadF32(bufs[2], M * N)
  releaseBufs(ctx, bufs)

proc metalMatmul2*(ctx: MetalContext,
                    a1: openArray[float32], M1, K1: int, b1: openArray[float32], K1b, N1: int,
                    a2: openArray[float32], M2, K2: int, b2: openArray[float32], K2b, N2: int):
                    tuple[y1, y2: seq[float32]] =
  ## Chạy 2 phép matmul ĐỘC LẬP (không cái nào đọc kết quả của cái kia) trong
  ## CÙNG 1 command buffer, chỉ commit+wait MỘT LẦN — thay vì 2 lần dispatch
  ## + 2 lần wait riêng lẻ như gọi metalMatmul() hai lần. Dùng trong
  ## Linear.backward (dW và dX không phụ thuộc lẫn nhau).
  assert K1 == K1b and K2 == K2b
  let out1 = ctx.poolGet(M1 * N1 * 4)
  let out2 = ctx.poolGet(M2 * N2 * 4)
  var bufs1 = @[
    ctx.uploadF32(a1), ctx.uploadF32(b1), out1,
    ctx.uploadU32(uint32(M1)), ctx.uploadU32(uint32(N1)), ctx.uploadU32(uint32(K1)),
  ]
  var bufs2 = @[
    ctx.uploadF32(a2), ctx.uploadF32(b2), out2,
    ctx.uploadU32(uint32(M2)), ctx.uploadU32(uint32(N2)), ctx.uploadU32(uint32(K2)),
  ]
  let pipe = ctx.pipeline("matmul")
  let cmdBuf = mtlCmdBufCreate(ctx.queue)
  let enc = mtlEncoderCreate(cmdBuf)
  mtlEncoderDispatchRaw(enc, pipe, addr bufs1[0], cint(bufs1.len),
                        csize_t(M1), csize_t(N1), csize_t(1),
                        csize_t(min(M1, 16)), csize_t(min(N1, 16)), csize_t(1))
  mtlEncoderDispatchRaw(enc, pipe, addr bufs2[0], cint(bufs2.len),
                        csize_t(M2), csize_t(N2), csize_t(1),
                        csize_t(min(M2, 16)), csize_t(min(N2, 16)), csize_t(1))
  mtlEncoderEnd(enc)
  mtlCmdBufCommitAndWait(cmdBuf)
  result = (downloadF32(out1, M1 * N1), downloadF32(out2, M2 * N2))
  releaseBufs(ctx, bufs1)
  releaseBufs(ctx, bufs2)
  mtlRelease(enc)
  mtlRelease(cmdBuf)

proc metalElementwise(ctx: MetalContext, kernelName: string, x: openArray[float32]): seq[float32] =
  let n = x.len
  var bufs = @[
    ctx.uploadF32(x),
    ctx.poolGet(n * 4),
    ctx.uploadU32(uint32(n)),
  ]
  ctx.dispatch(ctx.pipeline(kernelName), bufs, n, 1, 1, min(n, 256), 1, 1)
  result = downloadF32(bufs[1], n)
  releaseBufs(ctx, bufs)

proc metalRelu*(ctx: MetalContext, x: openArray[float32]): seq[float32] =
  metalElementwise(ctx, "relu_activation", x)

proc metalSigmoid*(ctx: MetalContext, x: openArray[float32]): seq[float32] =
  metalElementwise(ctx, "sigmoid_activation", x)

proc metalTanh*(ctx: MetalContext, x: openArray[float32]): seq[float32] =
  metalElementwise(ctx, "tanh_activation", x)

proc metalSoftmax*(ctx: MetalContext, x: openArray[float32], rows, cols: int): seq[float32] =
  ## x: [rows, cols] dạng phẳng, softmax theo từng hàng (trục cuối)
  assert x.len == rows * cols
  var bufs = @[
    ctx.uploadF32(x),
    ctx.poolGet(rows * cols * 4),
    ctx.uploadU32(uint32(rows)),
    ctx.uploadU32(uint32(cols)),
  ]
  ctx.dispatch(ctx.pipeline("softmax"), bufs, rows, 1, 1, min(rows, 256), 1, 1)
  result = downloadF32(bufs[1], rows * cols)
  releaseBufs(ctx, bufs)

proc uploadF32Val(ctx: MetalContext, v: float32): MTLBufferRef =
  var tmp = v
  result = ctx.poolGet(4)
  copyMem(mtlBufferContents(result), addr tmp, 4)

proc metalLayernorm*(ctx: MetalContext, x: openArray[float32], gamma, beta: openArray[float32],
                      rows, cols: int, eps: float32 = 1e-5'f32): seq[float32] =
  assert x.len == rows * cols
  assert gamma.len == cols and beta.len == cols
  var bufs = @[
    ctx.uploadF32(x),
    ctx.poolGet(rows * cols * 4),
    ctx.uploadF32(gamma),
    ctx.uploadF32(beta),
    ctx.uploadU32(uint32(rows)),
    ctx.uploadU32(uint32(cols)),
    ctx.uploadF32Val(eps),
  ]
  ctx.dispatch(ctx.pipeline("layernorm"), bufs, rows, 1, 1, min(rows, 256), 1, 1)
  result = downloadF32(bufs[1], rows * cols)
  releaseBufs(ctx, bufs)

proc uploadI32(ctx: MetalContext, arr: openArray[int32]): MTLBufferRef =
  result = ctx.poolGet(arr.len * 4)
  if arr.len > 0:
    copyMem(mtlBufferContents(result), unsafeAddr arr[0], arr.len * 4)

proc metalEmbeddingLookup*(ctx: MetalContext, table: openArray[float32], vocab, dim: int,
                            indices: openArray[int32]): seq[float32] =
  ## table: [vocab, dim] phẳng, indices: id cần tra -> trả về [indices.len, dim] phẳng
  let num = indices.len
  var bufs = @[
    ctx.uploadF32(table),
    ctx.uploadI32(indices),
    ctx.poolGet(num * dim * 4),
    ctx.uploadU32(uint32(vocab)),
    ctx.uploadU32(uint32(dim)),
    ctx.uploadU32(uint32(num)),
  ]
  ctx.dispatch(ctx.pipeline("embedding_lookup"), bufs, num, dim, 1, min(num, 16), min(dim, 16), 1)
  result = downloadF32(bufs[2], num * dim)
  releaseBufs(ctx, bufs)

# ─────────────────────────────────────────────────────────────
# Ví dụ: encode/decode CustomFloat trên GPU (Metal-first, APF tự build dtype)
#   tương đương customfloat_encode_dispatch() + apf_cast_for_training() bản Python
# ─────────────────────────────────────────────────────────────

proc customfloatEncodeGpu*(ctx: MetalContext, arr: openArray[float32], cf: CustomFloat): seq[uint8] =
  if not cf.usesUint64:
    return encodeArray(arr, cf)  # fallback CPU, giống nhánh >64 bit trong bản gốc
  let n = arr.len
  var bufs = @[
    ctx.uploadF32(arr),
    ctx.poolGet(n * cf.itemSize),
    ctx.uploadU32(uint32(n)),
    ctx.uploadU32(uint32(cf.exponentBits)),
    ctx.uploadU32(uint32(cf.mantissaBits)),
    ctx.uploadU32(uint32(cf.itemSize)),
  ]
  ctx.dispatch(ctx.pipeline("customfloat_encode"), bufs, n, 1, 1, min(n, 256), 1, 1)
  result = downloadU8(bufs[1], n * cf.itemSize)
  releaseBufs(ctx, bufs)

proc customfloatDecodeGpu*(ctx: MetalContext, buf: openArray[uint8], cf: CustomFloat): seq[float32] =
  if not cf.usesUint64:
    return decodeArray(buf, cf)
  let n = buf.len div cf.itemSize
  var bufs = @[
    ctx.uploadBytes(buf),
    ctx.poolGet(n * 4),
    ctx.uploadU32(uint32(n)),
    ctx.uploadU32(uint32(cf.exponentBits)),
    ctx.uploadU32(uint32(cf.mantissaBits)),
    ctx.uploadU32(uint32(cf.itemSize)),
  ]
  ctx.dispatch(ctx.pipeline("customfloat_decode"), bufs, n, 1, 1, min(n, 256), 1, 1)
  result = downloadF32(bufs[1], n)
  releaseBufs(ctx, bufs)

# apf_cast_for_training trên GPU: build dtype theo tensor (+grad) rồi encode trên Metal
proc apfCastForTrainingGpu*(ctx: MetalContext, arr: openArray[float32],
                             gradArr: openArray[float32] = []): tuple[data: seq[uint8], cf: CustomFloat] =
  let cf = buildCustomDtypeForTensor(arr, gradArr)
  result = (ctx.customfloatEncodeGpu(arr, cf), cf)