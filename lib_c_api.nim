## lib_c_api.nim — Stable C ABI for Nimformers.
##
## This module is the single shared-library entry point for non-Nim callers
## (Python/ctypes, C, C++, Rust, ...).  It exposes:
##
##   * runtime / backend selection and error handling
##   * owned float32 tensors (shape aware)
##   * every math / neural-network kernel from `backend.nim` and
##     `tensor_ops.nim` (elementwise, matmul, attention, norm, reductions,
##     scans, gather, dropout, comparisons, shape movement)
##   * int8/int4/fp8/fp4/custom-float quantization and NIMQ state dicts
##   * the micro-VM bytecode builder + dispatch (the "micro-kernel" path)
##   * resident (GPU-pinned) weight helpers
##
## ABI conventions
## ---------------
## * Integers are `int64` unless documented otherwise; floats are IEEE-754
##   `float32`; booleans are `cint` (0/1).
## * Every function that can fail returns 0 on success and 1 on failure
##   (or NULL / -1 for pointer / length queries).  Call
##   `nimf_last_error()` for the message.
## * Handles are opaque pointers created by `*_create` / `*_upload` and
##   released by the matching `*_destroy` / `*_free` call.
## * All output buffers are caller-allocated and must be at least as large
##   as documented.  API calls copy into them; no Nim-owned pointer is
##   handed out except tensor/quant payload accessors, which stay valid
##   until the handle is destroyed.
## * The library is not thread-safe for concurrent GPU calls on the same
##   context.  The error string is thread-local.

import std/strutils
import nimformers
import backend
import tensor_ops
import customfloat
import quant
import vendor/bybylang/gpubackend as byby
import vendor/bybylang/micro_kernel as micro
import mamba2_ssm_gpu
import vendor/bybylang/backends/cpu/cpu_backend as cpub
import native_hf_tokenizer as hf
import sentencepiece_native as sp
import nim_inference as nfinfer

const
  NIMF_ABI_VERSION* = 2'u32
  NIMF_VERSION_STRING = "nimformers-capi/2"

# =====================================================================
# Handles
# =====================================================================

type
  NimfContext = object
    backend: Backend

  NimfTensor = object
    data: seq[float32]
    shape: seq[int64]

  NimfQuantTensor = object
    qt: QuantTensor

  NimfStateDict = object
    arch: array[5, int]
    items: seq[(string, QuantTensor)]

  NimfVmProgram = object
    prog: micro.VmProgram

  NimfResidentQ4 = object
    weight: byby.ResidentWeightHandle

  NimfResidentF32 = object
    weight: byby.SessionHandle

  NimfTokenizer = object
    tok: hf.HFTokenizer

  NimfSentencePiece = object
    model: sp.SentencePieceModel

  NimfLlama = object
    model: nfinfer.LlamaModelL

# =====================================================================
# Error handling
# =====================================================================

var lastError {.threadvar.}: string

proc fail(msg: string): cint {.inline.} =
  lastError = msg
  1

proc failPtr(msg: string): pointer {.inline.} =
  lastError = msg
  nil

proc failCString(msg: string): cstring {.inline.} =
  lastError = msg
  nil

proc ok(): cint {.inline.} =
  lastError.setLen(0)
  0

proc nimf_abi_version*(): uint32 {.exportc: "nimf_abi_version", cdecl, dynlib.} =
  NIMF_ABI_VERSION

proc nimf_version*(): cstring {.exportc: "nimf_version", cdecl, dynlib.} =
  NIMF_VERSION_STRING.cstring

proc nimf_last_error*(): cstring {.exportc: "nimf_last_error", cdecl, dynlib.} =
  lastError.cstring

proc nimf_clear_error*() {.exportc: "nimf_clear_error", cdecl, dynlib.} =
  lastError.setLen(0)

proc nimf_cpu_isa*(): cint {.exportc: "nimf_cpu_isa", cdecl, dynlib.} =
  ## Numeric CPU ISA: 0=scalar, 1=SSE4, 2=AVX, 3=AVX2, 4=AVX-512, 5=NEON.
  let info = cpub.cpuInfo()
  result = if info.hasNeon: 5 else: cint(info.isa)

var cpuIsaNameBuf {.threadvar.}: string
proc nimf_cpu_isa_name*(): cstring {.exportc: "nimf_cpu_isa_name", cdecl, dynlib.} =
  let info = cpub.cpuInfo()
  cpuIsaNameBuf = if info.hasNeon: "neon" else: info.isaName
  cpuIsaNameBuf.cstring

proc nimf_cpu_features*(): uint32 {.exportc: "nimf_cpu_features", cdecl, dynlib.} =
  ## Bitmask: 1=SSE4 2=AVX 4=AVX2 8=FMA 16=AVX-512 32=NEON.
  let f = cpub.cpuFeatures()
  if f.sse4: result = result or 1'u32
  if f.avx: result = result or 2'u32
  if f.avx2: result = result or 4'u32
  if f.fma: result = result or 8'u32
  if f.avx512: result = result or 16'u32
  if f.neon: result = result or 32'u32

proc nimf_cpu_cache_sizes*(l1, l2, l3: ptr int64): cint
    {.exportc: "nimf_cpu_cache_sizes", cdecl, dynlib.} =
  if l1 == nil or l2 == nil or l3 == nil:
    return fail("null CPU cache output pointer")
  let info = cpub.cpuInfo()
  l1[] = int64(info.l1Bytes)
  l2[] = int64(info.l2Bytes)
  l3[] = int64(info.l3Bytes)
  ok()

proc nimf_cpu_block_sizes*(mc, nc, kc: ptr int64): cint
    {.exportc: "nimf_cpu_block_sizes", cdecl, dynlib.} =
  if mc == nil or nc == nil or kc == nil:
    return fail("null CPU block output pointer")
  let b = cpub.cpuBlockSizes()
  mc[] = int64(b.mc)
  nc[] = int64(b.nc)
  kc[] = int64(b.kc)
  ok()

proc nimf_free_string*(p: cstring) {.exportc: "nimf_free_string", cdecl, dynlib.} =
  if p != nil:
    deallocShared(cast[pointer](p))

proc allocCString(s: string): cstring =
  let p = cast[ptr UncheckedArray[char]](allocShared0(s.len + 1))
  if p == nil: return nil
  for i in 0 ..< s.len: p[i] = s[i]
  p[s.len] = '\0'
  cast[cstring](p)

# =====================================================================
# Small helpers
# =====================================================================

proc checkContext(ctx: pointer): ptr NimfContext {.inline.} =
  if ctx == nil: nil else: cast[ptr NimfContext](ctx)

proc checkTensor(t: pointer): ptr NimfTensor {.inline.} =
  if t == nil: nil else: cast[ptr NimfTensor](t)

proc checkQuant(q: pointer): ptr NimfQuantTensor {.inline.} =
  if q == nil: nil else: cast[ptr NimfQuantTensor](q)

proc checkVm(p: pointer): ptr NimfVmProgram {.inline.} =
  if p == nil: nil else: cast[ptr NimfVmProgram](p)

proc readShape(shape: ptr int64; rank: int64): seq[int] =
  if rank < 0 or rank > 16:
    raise newException(ValueError, "rank must be between 0 and 16")
  if rank > 0 and shape == nil:
    raise newException(ValueError, "null shape pointer")
  result = newSeq[int](int(rank))
  for i in 0 ..< int(rank):
    let d = cast[ptr UncheckedArray[int64]](shape)[i]
    if d < 0 or d > int64(high(int)):
      raise newException(ValueError, "invalid dimension")
    result[i] = int(d)

proc numelOf(shape: openArray[int]): int64 =
  result = 1
  for d in shape:
    if d < 0: raise newException(ValueError, "negative dimension")
    result *= int64(d)

proc numelOf64(shape: seq[int64]): int64 =
  result = 1
  for d in shape:
    if d < 0: raise newException(ValueError, "negative dimension")
    result *= d

proc inputValues(data: ptr cfloat; n: int): seq[float32] =
  if n == 0: return @[]
  if data == nil:
    raise newException(ValueError, "null data pointer")
  result = newSeq[float32](n)
  let src = cast[ptr UncheckedArray[cfloat]](data)
  for i in 0 ..< n: result[i] = float32(src[i])

proc writeValues(dst: ptr cfloat; values: openArray[float32]) =
  if values.len == 0: return
  if dst == nil:
    raise newException(ValueError, "null output pointer")
  let outArr = cast[ptr UncheckedArray[cfloat]](dst)
  for i in 0 ..< values.len: outArr[i] = cfloat(values[i])

proc requireContext(ctxHandle: pointer): Backend =
  if ctxHandle == nil:
    raise newException(ValueError, "null context")
  cast[ptr NimfContext](ctxHandle).backend

proc invalidBuf(buf: pointer; n: int64): bool {.inline.} =
  n > 0 and buf == nil

# =====================================================================
# Runtime / backend
# =====================================================================

proc nimf_context_create*(backendKind: cint): pointer
    {.exportc: "nimf_context_create", cdecl, dynlib.} =
  ## 0 = auto, 1 = Metal, 2 = Vulkan, 3 = CUDA (alias of Vulkan), 4 = CPU.
  try:
    let name = case backendKind
      of 0: "auto"
      of 1: "metal"
      of 2: "vulkan"
      of 3: "cuda"
      of 4: "cpu"
      else:
        return failPtr("backendKind must be 0(auto), 1(Metal), 2(Vulkan), 3(CUDA) or 4(CPU)")
    let ctx = cast[ptr NimfContext](allocShared0(sizeof(NimfContext)))
    if ctx == nil: return failPtr("context allocation failed")
    ctx.backend = newBackend(name)
    discard ok()
    cast[pointer](ctx)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_context_create_name*(backendName: cstring): pointer
    {.exportc: "nimf_context_create_name", cdecl, dynlib.} =
  if backendName == nil: return failPtr("null backend name")
  try:
    let ctx = cast[ptr NimfContext](allocShared0(sizeof(NimfContext)))
    if ctx == nil: return failPtr("context allocation failed")
    ctx.backend = newBackend($backendName)
    discard ok()
    cast[pointer](ctx)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_context_destroy*(handle: pointer): cint
    {.exportc: "nimf_context_destroy", cdecl, dynlib.} =
  if handle == nil: return ok()
  deallocShared(handle)
  ok()

proc nimf_context_backend*(handle: pointer): cint
    {.exportc: "nimf_context_backend", cdecl, dynlib.} =
  ## Returns 1 for Metal, 2 for Vulkan, -1 on error.
  let ctx = checkContext(handle)
  if ctx == nil:
    discard fail("null context")
    return -1
  case ctx.backend.kind
  of bkMetal: result = 1
  of bkCuda: result = 2
  of bkVulkan: result = 2
  of bkCpu: result = 4

proc nimf_context_backend_name*(handle: pointer): cstring
    {.exportc: "nimf_context_backend_name", cdecl, dynlib.} =
  let ctx = checkContext(handle)
  if ctx == nil:
    discard fail("null context")
    return nil
  case ctx.backend.kind
  of bkMetal: result = "metal"
  of bkCuda: result = "vulkan"
  of bkVulkan: result = "vulkan"
  of bkCpu: result = "cpu"

proc nimf_context_sync*(handle: pointer): cint
    {.exportc: "nimf_context_sync", cdecl, dynlib.} =
  ## Best-effort device synchronisation (opens and closes an empty session).
  let ctx = checkContext(handle)
  if ctx == nil: return fail("null context")
  try:
    let be = ctx.backend.kind.toByby()
    if not byby.sessionBegin(be):
      return fail("session begin failed: " & byby.sessionGetLastError(be))
    if not byby.sessionEnd(be):
      return fail("session end failed: " & byby.sessionGetLastError(be))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_context_set_cpu_fallback*(ctx: pointer; allowFallback: cint): cint
    {.exportc: "nimf_context_set_cpu_fallback", cdecl, dynlib.} =
  ## allowFallback != 0 permits silent CPU fallback; 0 forbids it.
  if ctx == nil: return fail("null context")
  backend.setForbidCpuFallback(allowFallback == 0)
  ok()

# =====================================================================
# Owned tensors
# =====================================================================

proc nimf_tensor_create*(data: ptr cfloat; n: int64): pointer
    {.exportc: "nimf_tensor_create", cdecl, dynlib.} =
  if n < 0 or (n > 0 and data == nil):
    return failPtr("invalid tensor pointer or length")
  try:
    let t = cast[ptr NimfTensor](allocShared0(sizeof(NimfTensor)))
    if t == nil: return failPtr("tensor allocation failed")
    t.data = inputValues(data, int(n))
    t.shape = @[n]
    discard ok()
    cast[pointer](t)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_tensor_create_shaped*(data: ptr cfloat; n: int64;
    shape: ptr int64; rank: int64): pointer
    {.exportc: "nimf_tensor_create_shaped", cdecl, dynlib.} =
  try:
    let dims = readShape(shape, rank)
    if numelOf(dims) != n:
      return failPtr("element count does not match shape")
    if n > 0 and data == nil: return failPtr("null data pointer")
    let t = cast[ptr NimfTensor](allocShared0(sizeof(NimfTensor)))
    if t == nil: return failPtr("tensor allocation failed")
    t.data = inputValues(data, int(n))
    t.shape = newSeq[int64](dims.len)
    for i, d in dims: t.shape[i] = int64(d)
    discard ok()
    cast[pointer](t)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_tensor_zeros*(shape: ptr int64; rank: int64; fill: cfloat): pointer
    {.exportc: "nimf_tensor_zeros", cdecl, dynlib.} =
  try:
    let dims = readShape(shape, rank)
    let n = numelOf(dims)
    let t = cast[ptr NimfTensor](allocShared0(sizeof(NimfTensor)))
    if t == nil: return failPtr("tensor allocation failed")
    t.data = newSeq[float32](int(n))
    for i in 0 ..< t.data.len: t.data[i] = float32(fill)
    t.shape = newSeq[int64](dims.len)
    for i, d in dims: t.shape[i] = int64(d)
    discard ok()
    cast[pointer](t)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_tensor_randn*(shape: ptr int64; rank: int64; scale: cfloat): pointer
    {.exportc: "nimf_tensor_randn", cdecl, dynlib.} =
  try:
    let dims = readShape(shape, rank)
    let tmp = randnTensor(dims, float32(scale))
    let t = cast[ptr NimfTensor](allocShared0(sizeof(NimfTensor)))
    if t == nil: return failPtr("tensor allocation failed")
    t.data = tmp.data
    t.shape = newSeq[int64](dims.len)
    for i, d in dims: t.shape[i] = int64(d)
    discard ok()
    cast[pointer](t)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_tensor_clone*(handle: pointer): pointer
    {.exportc: "nimf_tensor_clone", cdecl, dynlib.} =
  let t = checkTensor(handle)
  if t == nil: return failPtr("null tensor")
  try:
    let outT = cast[ptr NimfTensor](allocShared0(sizeof(NimfTensor)))
    if outT == nil: return failPtr("tensor allocation failed")
    outT.data = t.data
    outT.shape = t.shape
    discard ok()
    cast[pointer](outT)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_tensor_destroy*(handle: pointer): cint
    {.exportc: "nimf_tensor_destroy", cdecl, dynlib.} =
  if handle == nil: return ok()
  let t = cast[ptr NimfTensor](handle)
  t.data = @[]
  t.shape = @[]
  deallocShared(handle)
  ok()

proc nimf_tensor_rank*(handle: pointer): int64
    {.exportc: "nimf_tensor_rank", cdecl, dynlib.} =
  let t = checkTensor(handle)
  if t == nil:
    discard fail("null tensor")
    return -1
  int64(t.shape.len)

proc nimf_tensor_numel*(handle: pointer): int64
    {.exportc: "nimf_tensor_numel", cdecl, dynlib.} =
  let t = checkTensor(handle)
  if t == nil:
    discard fail("null tensor")
    return -1
  numelOf64(t.shape)

proc nimf_tensor_shape*(handle: pointer; outShape: ptr int64; cap: int64): int64
    {.exportc: "nimf_tensor_shape", cdecl, dynlib.} =
  ## Writes up to `cap` dimensions and returns the tensor rank.
  let t = checkTensor(handle)
  if t == nil:
    discard fail("null tensor")
    return -1
  let n = min(int(cap), t.shape.len)
  if n > 0 and outShape == nil:
    discard fail("null shape output")
    return -1
  for i in 0 ..< n:
    cast[ptr UncheckedArray[int64]](outShape)[i] = t.shape[i]
  int64(t.shape.len)

proc nimf_tensor_data*(handle: pointer): ptr cfloat
    {.exportc: "nimf_tensor_data", cdecl, dynlib.} =
  let t = checkTensor(handle)
  if t == nil or t.data.len == 0:
    discard fail("null tensor or empty tensor")
    return nil
  discard ok()
  cast[ptr cfloat](addr t.data[0])

proc nimf_tensor_len*(handle: pointer): int64
    {.exportc: "nimf_tensor_len", cdecl, dynlib.} =
  let t = checkTensor(handle)
  if t == nil:
    discard fail("null tensor")
    return -1
  discard ok()
  int64(t.data.len)

proc nimf_tensor_set_data*(handle: pointer; data: ptr cfloat; n: int64): cint
    {.exportc: "nimf_tensor_set_data", cdecl, dynlib.} =
  let t = checkTensor(handle)
  if t == nil: return fail("null tensor")
  if n != numelOf64(t.shape):
    return fail("element count does not match tensor shape")
  try:
    let v = inputValues(data, int(n))
    for i in 0 ..< v.len: t.data[i] = v[i]
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_tensor_fill*(handle: pointer; value: cfloat): cint
    {.exportc: "nimf_tensor_fill", cdecl, dynlib.} =
  let t = checkTensor(handle)
  if t == nil: return fail("null tensor")
  for i in 0 ..< t.data.len: t.data[i] = float32(value)
  ok()

proc nimf_tensor_reshape*(handle: pointer; shape: ptr int64; rank: int64): cint
    {.exportc: "nimf_tensor_reshape", cdecl, dynlib.} =
  let t = checkTensor(handle)
  if t == nil: return fail("null tensor")
  try:
    let dims = readShape(shape, rank)
    if numelOf(dims) != numelOf64(t.shape):
      return fail("new shape has a different element count")
    t.shape = newSeq[int64](dims.len)
    for i, d in dims: t.shape[i] = int64(d)
    ok()
  except CatchableError as e:
    fail(e.msg)

# =====================================================================
# Elementwise math
# =====================================================================

proc binaryOp(ctxHandle: pointer; a, b, outData: ptr cfloat; n: int64;
              op: int): cint =
  if ctxHandle == nil: return fail("null context")
  if n < 0 or invalidBuf(a, n) or invalidBuf(b, n) or invalidBuf(outData, n):
    return fail("invalid data pointer or length")
  try:
    let ctx = requireContext(ctxHandle)
    let av = inputValues(a, int(n))
    let bv = inputValues(b, int(n))
    let v = case op
      of 0: beAdd(ctx, av, bv)
      of 1: beSub(ctx, av, bv)
      of 2: beMul(ctx, av, bv)
      else: beDiv(ctx, av, bv)
    writeValues(outData, v)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_add*(ctx: pointer; a, b, outData: ptr cfloat; n: int64): cint
    {.exportc: "nimf_add", cdecl, dynlib.} = binaryOp(ctx, a, b, outData, n, 0)
proc nimf_sub*(ctx: pointer; a, b, outData: ptr cfloat; n: int64): cint
    {.exportc: "nimf_sub", cdecl, dynlib.} = binaryOp(ctx, a, b, outData, n, 1)
proc nimf_mul*(ctx: pointer; a, b, outData: ptr cfloat; n: int64): cint
    {.exportc: "nimf_mul", cdecl, dynlib.} = binaryOp(ctx, a, b, outData, n, 2)
proc nimf_div*(ctx: pointer; a, b, outData: ptr cfloat; n: int64): cint
    {.exportc: "nimf_div", cdecl, dynlib.} = binaryOp(ctx, a, b, outData, n, 3)

proc unaryOp(ctxHandle: pointer; x, outData: ptr cfloat; n: int64;
             op: int; exponent: cfloat = 1): cint =
  if ctxHandle == nil: return fail("null context")
  if n < 0 or invalidBuf(x, n) or invalidBuf(outData, n):
    return fail("invalid data pointer or length")
  try:
    let ctx = requireContext(ctxHandle)
    let xv = inputValues(x, int(n))
    let values = case op
      of 0: beSqrt(ctx, xv)
      of 1: beExp(ctx, xv)
      of 2: beLog(ctx, xv)
      of 3: bePow(ctx, xv, float32(exponent))
      of 4: beSigmoid(ctx, xv)
      of 5: beTanh(ctx, xv)
      else: beRelu(ctx, xv)
    writeValues(outData, values)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_sqrt*(ctx: pointer; x, outData: ptr cfloat; n: int64): cint
    {.exportc: "nimf_sqrt", cdecl, dynlib.} = unaryOp(ctx, x, outData, n, 0)
proc nimf_exp*(ctx: pointer; x, outData: ptr cfloat; n: int64): cint
    {.exportc: "nimf_exp", cdecl, dynlib.} = unaryOp(ctx, x, outData, n, 1)
proc nimf_log*(ctx: pointer; x, outData: ptr cfloat; n: int64): cint
    {.exportc: "nimf_log", cdecl, dynlib.} = unaryOp(ctx, x, outData, n, 2)
proc nimf_pow*(ctx: pointer; x, outData: ptr cfloat; n: int64; exponent: cfloat): cint
    {.exportc: "nimf_pow", cdecl, dynlib.} = unaryOp(ctx, x, outData, n, 3, exponent)
proc nimf_sigmoid*(ctx: pointer; x, outData: ptr cfloat; n: int64): cint
    {.exportc: "nimf_sigmoid", cdecl, dynlib.} = unaryOp(ctx, x, outData, n, 4)
proc nimf_tanh*(ctx: pointer; x, outData: ptr cfloat; n: int64): cint
    {.exportc: "nimf_tanh", cdecl, dynlib.} = unaryOp(ctx, x, outData, n, 5)
proc nimf_relu*(ctx: pointer; x, outData: ptr cfloat; n: int64): cint
    {.exportc: "nimf_relu", cdecl, dynlib.} = unaryOp(ctx, x, outData, n, 6)

proc nimf_neg*(ctxHandle: pointer; x, outData: ptr cfloat; n: int64): cint
    {.exportc: "nimf_neg", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if n < 0 or invalidBuf(x, n) or invalidBuf(outData, n):
    return fail("invalid data pointer or length")
  try:
    let ctx = requireContext(ctxHandle)
    let xv = inputValues(x, int(n))
    var minusOne = newSeq[float32](int(n))
    for i in 0 ..< minusOne.len: minusOne[i] = -1'f32
    writeValues(outData, beMul(ctx, xv, minusOne))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc scalarOp(ctxHandle: pointer; x, outData: ptr cfloat; n: int64;
              scalar: cfloat; isAdd: bool): cint =
  if ctxHandle == nil: return fail("null context")
  if n < 0 or invalidBuf(x, n) or invalidBuf(outData, n):
    return fail("invalid data pointer or length")
  try:
    let ctx = requireContext(ctxHandle)
    let xv = inputValues(x, int(n))
    var sv = newSeq[float32](int(n))
    for i in 0 ..< sv.len: sv[i] = float32(scalar)
    let v = if isAdd: beAdd(ctx, xv, sv) else: beMul(ctx, xv, sv)
    writeValues(outData, v)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_add_scalar*(ctx: pointer; x, outData: ptr cfloat; n: int64;
    scalar: cfloat): cint {.exportc: "nimf_add_scalar", cdecl, dynlib.} =
  scalarOp(ctx, x, outData, n, scalar, true)

proc nimf_mul_scalar*(ctx: pointer; x, outData: ptr cfloat; n: int64;
    scalar: cfloat): cint {.exportc: "nimf_mul_scalar", cdecl, dynlib.} =
  scalarOp(ctx, x, outData, n, scalar, false)

proc nimf_apflu*(ctxHandle: pointer; x, outData: ptr cfloat; n: int64;
    alpha, beta: cfloat): cint
    {.exportc: "nimf_apflu", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if n < 0 or invalidBuf(x, n) or invalidBuf(outData, n):
    return fail("invalid APFLU arguments")
  try:
    let ctx = requireContext(ctxHandle)
    writeValues(outData, beApflu(ctx, inputValues(x, int(n)),
                                 float32(alpha), float32(beta)))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_apflu_backward*(ctxHandle: pointer; x, dy, outData: ptr cfloat;
    n: int64; alpha, beta: cfloat): cint
    {.exportc: "nimf_apflu_backward", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if n < 0 or invalidBuf(x, n) or invalidBuf(dy, n) or invalidBuf(outData, n):
    return fail("invalid APFLU backward arguments")
  try:
    let ctx = requireContext(ctxHandle)
    writeValues(outData, beApfluBackward(ctx, inputValues(x, int(n)),
      inputValues(dy, int(n)), float32(alpha), float32(beta)))
    ok()
  except CatchableError as e:
    fail(e.msg)

# =====================================================================
# Matrix multiplication
# =====================================================================

proc nimf_matmul*(ctxHandle: pointer; a, b, outData: ptr cfloat;
                  m, k, n: int64): cint
    {.exportc: "nimf_matmul", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if m < 0 or k < 0 or n < 0 or invalidBuf(a, m * k) or
      invalidBuf(b, k * n) or invalidBuf(outData, m * n):
    return fail("invalid matmul dimensions or pointer")
  try:
    let ctx = requireContext(ctxHandle)
    let av = inputValues(a, int(m * k))
    let bv = inputValues(b, int(k * n))
    writeValues(outData, beMatmul(ctx, av, int(m), int(k), bv, int(k), int(n)))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_matmul2*(ctxHandle: pointer; a1, b1, out1: ptr cfloat;
    m1, k1, n1: int64; a2, b2, out2: ptr cfloat; m2, k2, n2: int64): cint
    {.exportc: "nimf_matmul2", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if min([m1, k1, n1, m2, k2, n2]) < 0 or
      invalidBuf(a1, m1 * k1) or invalidBuf(b1, k1 * n1) or invalidBuf(out1, m1 * n1) or
      invalidBuf(a2, m2 * k2) or invalidBuf(b2, k2 * n2) or invalidBuf(out2, m2 * n2):
    return fail("invalid dual-matmul arguments")
  try:
    let ctx = requireContext(ctxHandle)
    let r = beMatmul2(ctx,
      inputValues(a1, int(m1 * k1)), int(m1), int(k1),
      inputValues(b1, int(k1 * n1)), int(k1), int(n1),
      inputValues(a2, int(m2 * k2)), int(m2), int(k2),
      inputValues(b2, int(k2 * n2)), int(k2), int(n2))
    writeValues(out1, r.y1)
    writeValues(out2, r.y2)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_matmul_q4*(ctxHandle: pointer; a: ptr cfloat; aLen: int64;
    packedWeight: ptr uint8; packedLen: int64;
    scales, zeros: ptr cfloat; scaleLen, zeroLen: int64;
    outData: ptr cfloat; m, k, n, groupSize, groupsPerRow: int64): cint
    {.exportc: "nimf_matmul_q4", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if min([aLen, packedLen, scaleLen, zeroLen, m, k, n, groupSize, groupsPerRow]) < 0 or
      (aLen > 0 and a == nil) or (packedLen > 0 and packedWeight == nil) or
      (scaleLen > 0 and scales == nil) or (zeroLen > 0 and zeros == nil) or
      (m * n > 0 and outData == nil):
    return fail("invalid Q4 matmul arguments")
  try:
    let ctx = requireContext(ctxHandle)
    var wq = newSeq[uint8](int(packedLen))
    for i in 0 ..< wq.len: wq[i] = cast[ptr UncheckedArray[uint8]](packedWeight)[i]
    writeValues(outData, beMatmulQ4(ctx, inputValues(a, int(aLen)), wq,
      inputValues(scales, int(scaleLen)), inputValues(zeros, int(zeroLen)),
      int(m), int(k), int(n), int(groupSize), int(groupsPerRow)))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_matmul_nt*(ctxHandle: pointer; a, b, outData: ptr cfloat;
    m, k, n: int64): cint
    {.exportc: "nimf_matmul_nt", cdecl, dynlib.} =
  ## C[M,N] = A[M,K] @ B[N,K]^T.  B keeps its natural [N,K] layout.
  if ctxHandle == nil: return fail("null context")
  if m < 0 or k < 0 or n < 0 or invalidBuf(a, m * k) or
      invalidBuf(b, n * k) or invalidBuf(outData, m * n):
    return fail("invalid matmul-NT dimensions or pointer")
  try:
    let ctx = requireContext(ctxHandle)
    let be = ctx.kind.toByby()
    if not byby.sessionBegin(be):
      return fail("session begin failed: " & byby.sessionGetLastError(be))
    var hA = byby.sessionUpload(be, inputValues(a, int(m * k)))
    var hB = byby.sessionUpload(be, inputValues(b, int(n * k)))
    var hC: byby.SessionHandle
    let matOk = byby.sessionMatmulNT(be, hA, hB, hC, int(m), int(k), int(n))
    if not matOk:
      discard byby.sessionEnd(be)
      byby.sessionFree(be, hA)
      byby.sessionFree(be, hB)
      return fail("sessionMatmulNT failed")
    if not byby.sessionEnd(be):
      byby.sessionFree(be, hA)
      byby.sessionFree(be, hB)
      return fail("session end failed: " & byby.sessionGetLastError(be))
    let values = byby.sessionRead(be, hC, int(m * n))
    byby.sessionFree(be, hA)
    byby.sessionFree(be, hB)
    byby.sessionFree(be, hC)
    writeValues(outData, values)
    ok()
  except CatchableError as e:
    fail(e.msg)

# =====================================================================
# Resident (GPU-pinned) weights
# =====================================================================

proc nimf_resident_q4_upload*(ctxHandle: pointer; packed: ptr uint8;
    packedLen: int64; scales, zeros: ptr cfloat; scaleLen, zeroLen: int64;
    k, n, groupSize, groupsPerRow: int64): pointer
    {.exportc: "nimf_resident_q4_upload", cdecl, dynlib.} =
  if ctxHandle == nil: return failPtr("null context")
  if min([packedLen, scaleLen, zeroLen, k, n, groupSize, groupsPerRow]) < 0 or
      (packedLen > 0 and packed == nil) or (scaleLen > 0 and scales == nil) or
      (zeroLen > 0 and zeros == nil):
    return failPtr("invalid resident Q4 upload arguments")
  try:
    var bytes = newSeq[uint8](int(packedLen))
    for i in 0 ..< bytes.len: bytes[i] = cast[ptr UncheckedArray[uint8]](packed)[i]
    let h = cast[ptr NimfResidentQ4](allocShared0(sizeof(NimfResidentQ4)))
    if h == nil: return failPtr("resident handle allocation failed")
    h.weight = beResidentUploadQ4(requireContext(ctxHandle), bytes,
      inputValues(scales, int(scaleLen)), inputValues(zeros, int(zeroLen)),
      int(k), int(n), int(groupSize), int(groupsPerRow))
    discard ok()
    cast[pointer](h)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_resident_q4_linear*(ctxHandle, weightHandle: pointer; x: ptr cfloat;
    xLen, m: int64; outData: ptr cfloat): cint
    {.exportc: "nimf_resident_q4_linear", cdecl, dynlib.} =
  if ctxHandle == nil or weightHandle == nil: return fail("null handle")
  if xLen < 0 or m < 0 or (xLen > 0 and x == nil):
    return fail("invalid resident Q4 forward arguments")
  try:
    let h = cast[ptr NimfResidentQ4](weightHandle)
    let values = beResidentLinearQ4(requireContext(ctxHandle),
      inputValues(x, int(xLen)), h.weight, int(m))
    if values.len > 0 and outData == nil:
      return fail("null resident Q4 output")
    writeValues(outData, values)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_resident_q4_destroy*(ctxHandle, weightHandle: pointer): cint
    {.exportc: "nimf_resident_q4_destroy", cdecl, dynlib.} =
  if weightHandle == nil: return ok()
  try:
    let h = cast[ptr NimfResidentQ4](weightHandle)
    var w = h.weight
    beResidentFreeQ4(w)
    deallocShared(weightHandle)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_resident_f32_upload*(ctxHandle: pointer; weights: ptr cfloat;
    n: int64): pointer
    {.exportc: "nimf_resident_f32_upload", cdecl, dynlib.} =
  if ctxHandle == nil: return failPtr("null context")
  if n < 0 or (n > 0 and weights == nil):
    return failPtr("invalid resident fp32 upload arguments")
  try:
    let h = cast[ptr NimfResidentF32](allocShared0(sizeof(NimfResidentF32)))
    if h == nil: return failPtr("resident handle allocation failed")
    h.weight = beResidentUploadFp32(requireContext(ctxHandle),
      inputValues(weights, int(n)))
    discard ok()
    cast[pointer](h)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_resident_f32_linear*(ctxHandle, weightHandle: pointer; x: ptr cfloat;
    xLen, m, k, n: int64; outData: ptr cfloat): cint
    {.exportc: "nimf_resident_f32_linear", cdecl, dynlib.} =
  if ctxHandle == nil or weightHandle == nil: return fail("null handle")
  if min([xLen, m, k, n]) < 0 or (xLen > 0 and x == nil):
    return fail("invalid resident fp32 forward arguments")
  try:
    let h = cast[ptr NimfResidentF32](weightHandle)
    let values = beResidentLinearFp32(requireContext(ctxHandle),
      inputValues(x, int(xLen)), h.weight, int(m), int(k), int(n))
    if values.len > 0 and outData == nil:
      return fail("null resident fp32 output")
    writeValues(outData, values)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_resident_f32_destroy*(ctxHandle, weightHandle: pointer): cint
    {.exportc: "nimf_resident_f32_destroy", cdecl, dynlib.} =
  if weightHandle == nil: return ok()
  try:
    let h = cast[ptr NimfResidentF32](weightHandle)
    var w = h.weight
    beResidentFreeFp32(requireContext(ctxHandle), w)
    deallocShared(weightHandle)
    ok()
  except CatchableError as e:
    fail(e.msg)

# =====================================================================
# Neural-network kernels
# =====================================================================

proc nimf_softmax*(ctxHandle: pointer; x, outData: ptr cfloat;
                   rows, cols: int64): cint
    {.exportc: "nimf_softmax", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if rows < 0 or cols < 0 or invalidBuf(x, rows * cols) or
      invalidBuf(outData, rows * cols):
    return fail("invalid softmax arguments")
  try:
    writeValues(outData, beSoftmax(requireContext(ctxHandle),
      inputValues(x, int(rows * cols)), int(rows), int(cols)))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_layernorm*(ctxHandle: pointer; x, gamma, beta, outData: ptr cfloat;
    rows, cols: int64; eps: cfloat): cint
    {.exportc: "nimf_layernorm", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if rows < 0 or cols < 0 or invalidBuf(x, rows * cols) or
      invalidBuf(outData, rows * cols) or (cols > 0 and (gamma == nil or beta == nil)):
    return fail("invalid layernorm arguments")
  try:
    writeValues(outData, beLayernorm(requireContext(ctxHandle),
      inputValues(x, int(rows * cols)), inputValues(gamma, int(cols)),
      inputValues(beta, int(cols)), int(rows), int(cols), float32(eps)))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_add_layernorm*(ctxHandle: pointer; a, b, gamma, beta,
    sumOut, normOut: ptr cfloat; rows, cols: int64; eps: cfloat): cint
    {.exportc: "nimf_add_layernorm", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if rows < 0 or cols < 0 or invalidBuf(a, rows * cols) or
      invalidBuf(b, rows * cols) or invalidBuf(sumOut, rows * cols) or
      invalidBuf(normOut, rows * cols) or
      (cols > 0 and (gamma == nil or beta == nil)):
    return fail("invalid fused add-layernorm arguments")
  try:
    let r = beAddLayernormFused(requireContext(ctxHandle),
      inputValues(a, int(rows * cols)), inputValues(b, int(rows * cols)),
      inputValues(gamma, int(cols)), inputValues(beta, int(cols)),
      int(rows), int(cols), float32(eps))
    writeValues(sumOut, r.summed)
    writeValues(normOut, r.normed)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_layernorm_backward*(ctxHandle: pointer; dy, x, gamma, beta: ptr cfloat;
    rows, cols: int64; eps: cfloat; dx, dgamma, dbeta: ptr cfloat): cint
    {.exportc: "nimf_layernorm_backward", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if min([rows, cols]) < 0 or invalidBuf(dy, rows * cols) or
      invalidBuf(x, rows * cols) or invalidBuf(dx, rows * cols) or
      (cols > 0 and (gamma == nil or beta == nil or
                     dgamma == nil or dbeta == nil)):
    return fail("invalid layernorm backward arguments")
  try:
    let r = beLayernormBackward(requireContext(ctxHandle),
      inputValues(dy, int(rows * cols)), inputValues(x, int(rows * cols)),
      inputValues(gamma, int(cols)), inputValues(beta, int(cols)),
      int(rows), int(cols), float32(eps))
    writeValues(dx, r.dx)
    writeValues(dgamma, r.dgamma)
    writeValues(dbeta, r.dbeta)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_attention*(ctxHandle: pointer; q, k, v, mask: ptr cfloat; maskLen: int64;
    batch, heads, sequenceLen, headDim: int64; scale: cfloat;
    outData, scores: ptr cfloat): cint
    {.exportc: "nimf_attention", cdecl, dynlib.} =
  let tensorLen = batch * heads * sequenceLen * headDim
  let scoreLen = batch * heads * sequenceLen * sequenceLen
  if ctxHandle == nil: return fail("null context")
  if min([batch, heads, sequenceLen, headDim, maskLen]) < 0 or
      invalidBuf(q, tensorLen) or invalidBuf(k, tensorLen) or
      invalidBuf(v, tensorLen) or invalidBuf(outData, tensorLen) or
      invalidBuf(scores, scoreLen) or invalidBuf(mask, maskLen):
    return fail("invalid attention arguments")
  try:
    let r = beAttentionFused(requireContext(ctxHandle),
      inputValues(q, int(tensorLen)), inputValues(k, int(tensorLen)),
      inputValues(v, int(tensorLen)), inputValues(mask, int(maskLen)),
      int(batch), int(heads), int(sequenceLen), int(headDim), float32(scale))
    writeValues(outData, r.o)
    writeValues(scores, r.s_matrix)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_attention_backward*(ctxHandle: pointer; q, k, v, scores, dy: ptr cfloat;
    batch, heads, sequenceLen, headDim: int64; scale: cfloat;
    dq, dk, dv: ptr cfloat): cint
    {.exportc: "nimf_attention_backward", cdecl, dynlib.} =
  let tensorLen = batch * heads * sequenceLen * headDim
  let scoreLen = batch * heads * sequenceLen * sequenceLen
  if ctxHandle == nil: return fail("null context")
  if min([batch, heads, sequenceLen, headDim]) < 0 or
      invalidBuf(q, tensorLen) or invalidBuf(k, tensorLen) or
      invalidBuf(v, tensorLen) or invalidBuf(dy, tensorLen) or
      invalidBuf(dq, tensorLen) or invalidBuf(dk, tensorLen) or
      invalidBuf(dv, tensorLen) or invalidBuf(scores, scoreLen):
    return fail("invalid attention-backward arguments")
  try:
    let r = beAttentionFusedBackward(requireContext(ctxHandle),
      inputValues(q, int(tensorLen)), inputValues(k, int(tensorLen)),
      inputValues(v, int(tensorLen)), inputValues(scores, int(scoreLen)),
      inputValues(dy, int(tensorLen)), int(batch), int(heads),
      int(sequenceLen), int(headDim), float32(scale))
    writeValues(dq, r.dq)
    writeValues(dk, r.dk)
    writeValues(dv, r.dv)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_embedding_lookup*(ctxHandle: pointer; table: ptr cfloat; tableLen: int64;
    indices: ptr int32; indexCount: int64; outData: ptr cfloat;
    vocab, dim: int64): cint
    {.exportc: "nimf_embedding_lookup", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if tableLen < 0 or indexCount < 0 or vocab < 0 or dim < 0 or
      (tableLen > 0 and table == nil) or
      (indexCount > 0 and (indices == nil or outData == nil)):
    return fail("invalid embedding lookup arguments")
  try:
    var ids = newSeq[int32](int(indexCount))
    for i in 0 ..< ids.len: ids[i] = cast[ptr UncheckedArray[int32]](indices)[i]
    writeValues(outData, beEmbeddingLookup(requireContext(ctxHandle),
      inputValues(table, int(tableLen)), ids, int(vocab), int(dim)))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_gather*(ctxHandle: pointer; src: ptr cfloat; srcLen: int64;
    indices: ptr int32; indexCount: int64; outData: ptr cfloat): cint
    {.exportc: "nimf_gather", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if srcLen < 0 or indexCount < 0 or (srcLen > 0 and src == nil) or
      (indexCount > 0 and (indices == nil or outData == nil)):
    return fail("invalid gather arguments")
  try:
    var ids = newSeq[int32](int(indexCount))
    for i in 0 ..< ids.len: ids[i] = cast[ptr UncheckedArray[int32]](indices)[i]
    writeValues(outData, beGatherGpu(requireContext(ctxHandle),
      inputValues(src, int(srcLen)), ids))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_compare*(ctxHandle: pointer; a, b: ptr cfloat; n: int64;
    op: cint; outMask: ptr uint8): cint
    {.exportc: "nimf_compare", cdecl, dynlib.} =
  ## op: 0=gt, 1=lt, 2=eq, 3=ge, 4=le.  outMask receives 0/1 bytes.
  if ctxHandle == nil: return fail("null context")
  if n < 0 or op < 0 or op > 4 or invalidBuf(a, n) or invalidBuf(b, n) or
      invalidBuf(outMask, n):
    return fail("invalid compare arguments (op: 0=gt,1=lt,2=eq,3=ge,4=le)")
  try:
    let mask = beCompareGpuOp(requireContext(ctxHandle),
      inputValues(a, int(n)), inputValues(b, int(n)), int(op))
    for i in 0 ..< mask.len:
      cast[ptr UncheckedArray[uint8]](outMask)[i] = uint8(ord(mask[i]))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_dropout*(ctxHandle: pointer; x, outData: ptr cfloat; n: int64;
    probability: cfloat; training: cint; seed: uint32): cint
    {.exportc: "nimf_dropout", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if n < 0 or probability < 0 or probability >= 1 or
      invalidBuf(x, n) or invalidBuf(outData, n):
    return fail("invalid dropout arguments")
  try:
    writeValues(outData, beDropout(requireContext(ctxHandle),
      inputValues(x, int(n)), float32(probability), training != 0, seed))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_dropout_mask*(ctxHandle: pointer; n: int64; probability: cfloat;
    seed: uint32; outData: ptr cfloat): cint
    {.exportc: "nimf_dropout_mask", cdecl, dynlib.} =
  ## Deterministic inverted-dropout mask (0 or 1/(1-p)) for a given seed.
  if ctxHandle == nil: return fail("null context")
  if n < 0 or probability < 0 or probability >= 1 or invalidBuf(outData, n):
    return fail("invalid dropout-mask arguments")
  try:
    writeValues(outData, beDropoutMaskGpu(requireContext(ctxHandle), int(n),
      float32(probability), seed))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_argmax*(ctxHandle: pointer; x: ptr cfloat; xLen: int64;
    outer, dimSize, inner: int64; outValues: ptr cfloat;
    outIndices: ptr int64): cint
    {.exportc: "nimf_argmax", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if min([xLen, outer, dimSize, inner]) < 0 or invalidBuf(x, xLen) or
      invalidBuf(outValues, outer * inner) or invalidBuf(outIndices, outer * inner):
    return fail("invalid argmax arguments")
  if outer * dimSize * inner != xLen:
    return fail("argmax shapes do not match xLen")
  if dimSize <= 0 and outer * inner > 0:
    return fail("argmax requires a non-empty reduction dimension")
  try:
    let pairs = int(outer * inner)
    var outVals = newSeq[float32](pairs)
    var outIdx = newSeq[int64](pairs)
    # The Metal reduction kernel requires a power-of-two reduction width;
    # fall back to a small host loop for other widths to keep results exact.
    if dimSize > 0 and (dimSize and (dimSize - 1)) == 0:
      let (values, indices) = beArgmaxIndexGpu(requireContext(ctxHandle),
        inputValues(x, int(xLen)), int(outer), int(dimSize), int(inner))
      outVals = values
      for i in 0 ..< indices.len: outIdx[i] = int64(indices[i])
    else:
      let xv = inputValues(x, int(xLen))
      for o in 0 ..< int(outer):
        for i in 0 ..< int(inner):
          var best = xv[(o * int(dimSize)) * int(inner) + i]
          var bestIdx = 0
          for d in 1 ..< int(dimSize):
            let v = xv[(o * int(dimSize) + d) * int(inner) + i]
            if v > best:
              best = v
              bestIdx = d
          outVals[o * int(inner) + i] = best
          outIdx[o * int(inner) + i] = int64(bestIdx)
    writeValues(outValues, outVals)
    for i in 0 ..< outIdx.len:
      cast[ptr UncheckedArray[int64]](outIndices)[i] = outIdx[i]
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_masked_fill*(x: ptr cfloat; mask: ptr uint8; n: int64;
    value: cfloat; outData: ptr cfloat): cint
    {.exportc: "nimf_masked_fill", cdecl, dynlib.} =
  if n < 0 or invalidBuf(x, n) or invalidBuf(mask, n) or invalidBuf(outData, n):
    return fail("invalid masked-fill arguments")
  let src = cast[ptr UncheckedArray[cfloat]](x)
  let msk = cast[ptr UncheckedArray[uint8]](mask)
  let dst = cast[ptr UncheckedArray[cfloat]](outData)
  for i in 0 ..< int(n):
    dst[i] = if msk[i] != 0'u8: cfloat(value) else: src[i]
  ok()

proc nimf_where*(cond: ptr uint8; a, b: ptr cfloat; n: int64;
    outData: ptr cfloat): cint
    {.exportc: "nimf_where", cdecl, dynlib.} =
  if n < 0 or invalidBuf(cond, n) or invalidBuf(a, n) or invalidBuf(b, n) or
      invalidBuf(outData, n):
    return fail("invalid where arguments")
  let c = cast[ptr UncheckedArray[uint8]](cond)
  let av = cast[ptr UncheckedArray[cfloat]](a)
  let bv = cast[ptr UncheckedArray[cfloat]](b)
  let dst = cast[ptr UncheckedArray[cfloat]](outData)
  for i in 0 ..< int(n):
    dst[i] = if c[i] != 0'u8: av[i] else: bv[i]
  ok()

# =====================================================================
# Reductions and scans
# =====================================================================

proc shapeValues(shape: ptr int64; rank: int64): seq[int] =
  readShape(shape, rank)

proc reductionOp(ctxHandle: pointer; x: ptr cfloat; xLen: int64;
    shape: ptr int64; rank, dim: int64; outData: ptr cfloat; op: int): cint =
  if ctxHandle == nil: return fail("null context")
  if xLen < 0 or rank <= 0 or dim < 0 or dim >= rank or shape == nil or
      invalidBuf(x, xLen) or outData == nil:
    return fail("invalid reduction arguments")
  try:
    let dims = shapeValues(shape, rank)
    let xv = inputValues(x, int(xLen))
    let ctx = requireContext(ctxHandle)
    let values = case op
      of 0: beSum(ctx, xv, dims, int(dim))
      of 1: beMean(ctx, xv, dims, int(dim))
      else: beMax(ctx, xv, dims, int(dim))
    writeValues(outData, values)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_sum*(ctx: pointer; x: ptr cfloat; xLen: int64; shape: ptr int64;
    rank, dim: int64; outData: ptr cfloat): cint
    {.exportc: "nimf_sum", cdecl, dynlib.} =
  reductionOp(ctx, x, xLen, shape, rank, dim, outData, 0)
proc nimf_mean*(ctx: pointer; x: ptr cfloat; xLen: int64; shape: ptr int64;
    rank, dim: int64; outData: ptr cfloat): cint
    {.exportc: "nimf_mean", cdecl, dynlib.} =
  reductionOp(ctx, x, xLen, shape, rank, dim, outData, 1)
proc nimf_max*(ctx: pointer; x: ptr cfloat; xLen: int64; shape: ptr int64;
    rank, dim: int64; outData: ptr cfloat): cint
    {.exportc: "nimf_max", cdecl, dynlib.} =
  reductionOp(ctx, x, xLen, shape, rank, dim, outData, 2)

proc scanOp(ctxHandle: pointer; x: ptr cfloat; xLen: int64; shape: ptr int64;
    rank, dim: int64; outData: ptr cfloat; op: int): cint =
  if ctxHandle == nil: return fail("null context")
  if xLen < 0 or rank <= 0 or dim < 0 or dim >= rank or shape == nil or
      invalidBuf(x, xLen) or invalidBuf(outData, xLen):
    return fail("invalid scan arguments")
  try:
    let dims = shapeValues(shape, rank)
    let xv = inputValues(x, int(xLen))
    let ctx = requireContext(ctxHandle)
    let values = if op == 0: beCumsum(ctx, xv, dims, int(dim))
                 else: beCumprod(ctx, xv, dims, int(dim))
    writeValues(outData, values)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_cumsum*(ctx: pointer; x: ptr cfloat; xLen: int64; shape: ptr int64;
    rank, dim: int64; outData: ptr cfloat): cint
    {.exportc: "nimf_cumsum", cdecl, dynlib.} =
  scanOp(ctx, x, xLen, shape, rank, dim, outData, 0)
proc nimf_cumprod*(ctx: pointer; x: ptr cfloat; xLen: int64; shape: ptr int64;
    rank, dim: int64; outData: ptr cfloat): cint
    {.exportc: "nimf_cumprod", cdecl, dynlib.} =
  scanOp(ctx, x, xLen, shape, rank, dim, outData, 1)

proc nimf_affine_scan*(ctxHandle: pointer; aCoef, bCoef: ptr cfloat; n: int64;
    shape: ptr int64; rank, dim: int64; outData: ptr cfloat): cint
    {.exportc: "nimf_affine_scan", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if n < 0 or rank <= 0 or dim < 0 or dim >= rank or shape == nil or
      invalidBuf(aCoef, n) or invalidBuf(bCoef, n) or invalidBuf(outData, n):
    return fail("invalid affine-scan arguments")
  try:
    let dims = shapeValues(shape, rank)
    var inner = 1
    for i in int(dim) + 1 ..< dims.len: inner *= dims[i]
    if inner != 1:
      return fail("affine scan currently supports the last dimension only")
    writeValues(outData, beAffineScan(requireContext(ctxHandle),
      inputValues(aCoef, int(n)), inputValues(bCoef, int(n)),
      dims, int(dim)))
    ok()
  except CatchableError as e:
    fail(e.msg)

# =====================================================================
# Shape / data movement
# =====================================================================

proc nimf_permute*(ctxHandle: pointer; src: ptr cfloat; n: int64;
    newStrides, oldStrides, dimsMap: ptr int32; rank: int64;
    outData: ptr cfloat): cint
    {.exportc: "nimf_permute", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if n < 0 or rank < 0 or invalidBuf(src, n) or invalidBuf(outData, n) or
      (rank > 0 and (newStrides == nil or oldStrides == nil or dimsMap == nil)):
    return fail("invalid permute arguments")
  try:
    var ns = newSeq[int32](int(rank))
    var os = newSeq[int32](int(rank))
    var dm = newSeq[int32](int(rank))
    for i in 0 ..< int(rank):
      ns[i] = cast[ptr UncheckedArray[int32]](newStrides)[i]
      os[i] = cast[ptr UncheckedArray[int32]](oldStrides)[i]
      dm[i] = cast[ptr UncheckedArray[int32]](dimsMap)[i]
    writeValues(outData, bePermuteGpu(requireContext(ctxHandle),
      inputValues(src, int(n)), ns, os, dm, int(n)))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_split_slice*(ctxHandle: pointer; src: ptr cfloat;
    dimTotal, thisSize, offset, inner, n: int64; outData: ptr cfloat): cint
    {.exportc: "nimf_split_slice", cdecl, dynlib.} =
  ## `src` is the full flattened tensor (all outer slices), not just one.
  ## `n` is the output size; outer is derived as n / (thisSize * inner).
  if ctxHandle == nil: return fail("null context")
  if min([dimTotal, thisSize, offset, inner, n]) < 0 or
      offset + thisSize > dimTotal or invalidBuf(outData, n):
    return fail("invalid split slice arguments")
  let unit = thisSize * inner
  if unit > 0 and n mod unit != 0:
    return fail("split slice output size is inconsistent with thisSize*inner")
  let outer = if unit > 0: n div unit else: 0
  let srcLen = outer * dimTotal * inner
  if invalidBuf(src, srcLen):
    return fail("invalid split slice source pointer")
  try:
    writeValues(outData, beSplitSliceGpu(requireContext(ctxHandle),
      inputValues(src, int(srcLen)), int(dimTotal), int(thisSize),
      int(offset), int(inner), int(n)))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_concat*(ctxHandle: pointer; src: ptr cfloat; srcLen: int64;
    srcBase, srcDimSize, srcDimOffset: ptr int32; count, totalDim, inner,
    outLen: int64; outData: ptr cfloat): cint
    {.exportc: "nimf_concat", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if min([srcLen, count, totalDim, inner, outLen]) < 0 or
      invalidBuf(src, srcLen) or
      (count > 0 and (srcBase == nil or srcDimSize == nil or srcDimOffset == nil)) or
      invalidBuf(outData, outLen):
    return fail("invalid concat arguments")
  try:
    var bases = newSeq[int32](int(count))
    var sizes = newSeq[int32](int(count))
    var offsets = newSeq[int32](int(count))
    for i in 0 ..< int(count):
      bases[i] = cast[ptr UncheckedArray[int32]](srcBase)[i]
      sizes[i] = cast[ptr UncheckedArray[int32]](srcDimSize)[i]
      offsets[i] = cast[ptr UncheckedArray[int32]](srcDimOffset)[i]
    writeValues(outData, beConcatGpu(requireContext(ctxHandle),
      inputValues(src, int(srcLen)), bases, sizes, offsets,
      int(totalDim), int(inner), int(outLen)))
    ok()
  except CatchableError as e:
    fail(e.msg)

# =====================================================================
# Quantization
# =====================================================================

const
  QK_FP32_RAW* = 0
  QK_INT8* = 1
  QK_INT4* = 2
  QK_INT4_ASYMMETRIC* = 3
  QK_FP8_E4M3* = 4
  QK_FP8_E5M2* = 5
  QK_CUSTOM* = 6
  QK_AUTO* = 7
  QK_FP4_KM* = 8

proc quantBytesLen(qt: QuantTensor): int {.inline.} =
  if qt.mappedData.owner != nil: qt.mappedData.len else: qt.data.len

proc quantBytesPtr(qt: QuantTensor): pointer =
  if qt.mappedData.owner != nil:
    cast[pointer](qt.mappedData.dataPtr)
  elif qt.data.len > 0:
    cast[pointer](addr qt.data[0])
  else:
    nil

proc nimf_quantize*(x: ptr cfloat; n: int64; shape: ptr int64; rank: int64;
    kind: cint; exponentBits, mantissaBits: cint): pointer
    {.exportc: "nimf_quantize", cdecl, dynlib.} =
  ## Quantize `x` with `kind` (see QK_*).  For QK_CUSTOM, exponentBits and
  ## mantissaBits define the custom float; for the fixed fp8 kinds they are
  ## ignored.  Returns a quant handle or NULL.
  if n < 0 or invalidBuf(x, n): return failPtr("invalid quantize arguments")
  if kind < 0 or kind > 8: return failPtr("invalid quantization kind")
  try:
    let dims = readShape(shape, rank)
    if numelOf(dims) != n:
      return failPtr("element count does not match shape")
    if kind == QK_FP4_KM:
      if dims.len != 2:
        return failPtr("fp4_km quantization requires a 2-D [rows, cols] tensor")
      if dims[1] mod 32 != 0:
        return failPtr("fp4_km quantization requires cols divisible by 32")
    if kind == QK_CUSTOM and (exponentBits != 0 or mantissaBits != 0):
      if exponentBits < 1 or mantissaBits < 0 or
          int(exponentBits) + int(mantissaBits) + 1 > 64:
        return failPtr("custom float needs exponentBits >= 1 and total bits <= 64")
    let cf = if exponentBits > 0 or mantissaBits > 0:
               newCustomFloat(int(exponentBits), int(mantissaBits))
             else:
               FP24
    let q = quantizeTensor(inputValues(x, int(n)), dims, QuantKind(int(kind)), cf)
    let h = cast[ptr NimfQuantTensor](allocShared0(sizeof(NimfQuantTensor)))
    if h == nil: return failPtr("quant handle allocation failed")
    h.qt = q
    discard ok()
    cast[pointer](h)
  except CatchableError as e:
    failPtr(e.msg)
  except Defect as e:
    failPtr("internal quantization error: " & e.msg)

proc nimf_quant_dequantize*(handle: pointer; outData: ptr cfloat;
    capacity: int64): int64
    {.exportc: "nimf_quant_dequantize", cdecl, dynlib.} =
  ## Writes the dequantized tensor and returns its element count, or -1.
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return -1
  try:
    let values = dequantizeTensor(q.qt)
    if capacity < int64(values.len):
      discard fail("output buffer too small")
      return -1
    writeValues(outData, values)
    discard ok()
    int64(values.len)
  except CatchableError as e:
    discard fail(e.msg)
    -1
  except Defect as e:
    discard fail("internal dequantization error: " & e.msg)
    -1

proc nimf_quant_dequantize_transposed*(handle: pointer; outData: ptr cfloat;
    capacity: int64): int64
    {.exportc: "nimf_quant_dequantize_transposed", cdecl, dynlib.} =
  ## Dequantize a 2-D tensor directly into transposed [cols, rows] layout.
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return -1
  try:
    if q.qt.shape.len > 2:
      discard fail("transposed dequantization supports at most 2 dimensions")
      return -1
    let values = dequantizeTensorTransposed(q.qt)
    if capacity < int64(values.len):
      discard fail("output buffer too small")
      return -1
    writeValues(outData, values)
    discard ok()
    int64(values.len)
  except CatchableError as e:
    discard fail(e.msg)
    -1
  except Defect as e:
    discard fail("internal dequantization error: " & e.msg)
    -1

proc nimf_quant_kind*(handle: pointer): cint
    {.exportc: "nimf_quant_kind", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return -1
  cint(ord(q.qt.kind))

proc nimf_quant_rank*(handle: pointer): int64
    {.exportc: "nimf_quant_rank", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return -1
  int64(q.qt.shape.len)

proc nimf_quant_shape*(handle: pointer; outShape: ptr int64;
    cap: int64): int64
    {.exportc: "nimf_quant_shape", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return -1
  let n = min(int(cap), q.qt.shape.len)
  if n > 0 and outShape == nil:
    discard fail("null shape output")
    return -1
  for i in 0 ..< n:
    cast[ptr UncheckedArray[int64]](outShape)[i] = int64(q.qt.shape[i])
  int64(q.qt.shape.len)

proc nimf_quant_numel*(handle: pointer): int64
    {.exportc: "nimf_quant_numel", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return -1
  numelOf(q.qt.shape)

proc nimf_quant_group_size*(handle: pointer): int64
    {.exportc: "nimf_quant_group_size", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return -1
  int64(q.qt.groupSize)

proc nimf_quant_cf_bits*(handle: pointer; exponentBitsOut,
    mantissaBitsOut: ptr cint): cint
    {.exportc: "nimf_quant_cf_bits", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil: return fail("null quant tensor")
  if exponentBitsOut == nil or mantissaBitsOut == nil:
    return fail("null output pointer")
  exponentBitsOut[] = cint(q.qt.cf.exponentBits)
  mantissaBitsOut[] = cint(q.qt.cf.mantissaBits)
  ok()

proc nimf_quant_data_len*(handle: pointer): int64
    {.exportc: "nimf_quant_data_len", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return -1
  int64(quantBytesLen(q.qt))

proc nimf_quant_data*(handle: pointer): ptr uint8
    {.exportc: "nimf_quant_data", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return nil
  cast[ptr uint8](quantBytesPtr(q.qt))

proc nimf_quant_scale_len*(handle: pointer): int64
    {.exportc: "nimf_quant_scale_len", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return -1
  int64(q.qt.scale.len)

proc nimf_quant_scale*(handle: pointer): ptr cfloat
    {.exportc: "nimf_quant_scale", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return nil
  if q.qt.scale.len == 0: return nil
  cast[ptr cfloat](addr q.qt.scale[0])

proc nimf_quant_zero_len*(handle: pointer): int64
    {.exportc: "nimf_quant_zero_len", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return -1
  int64(q.qt.zero_point.len)

proc nimf_quant_zero*(handle: pointer): ptr cfloat
    {.exportc: "nimf_quant_zero", cdecl, dynlib.} =
  let q = checkQuant(handle)
  if q == nil:
    discard fail("null quant tensor")
    return nil
  if q.qt.zero_point.len == 0: return nil
  cast[ptr cfloat](addr q.qt.zero_point[0])

proc nimf_quant_destroy*(handle: pointer): cint
    {.exportc: "nimf_quant_destroy", cdecl, dynlib.} =
  if handle == nil: return ok()
  let q = cast[ptr NimfQuantTensor](handle)
  q.qt.clearPayload()
  q.qt.shape = @[]
  q.qt.scale = @[]
  q.qt.zero_point = @[]
  deallocShared(handle)
  ok()

# =====================================================================
# NIMQ state dicts (.nimq checkpoints)
# =====================================================================

proc nimf_state_dict_load*(path: cstring; mapped: cint): pointer
    {.exportc: "nimf_state_dict_load", cdecl, dynlib.} =
  ## Loads a NIMQ2 state dict.  mapped != 0 keeps tensor payloads in the OS
  ## page cache (recommended for large checkpoints); 0 copies them.
  if path == nil: return failPtr("null path")
  try:
    let loaded = loadQuantStateDict($path, mapped != 0)
    let h = cast[ptr NimfStateDict](allocShared0(sizeof(NimfStateDict)))
    if h == nil: return failPtr("state-dict handle allocation failed")
    h.arch = loaded.arch
    h.items = loaded.sd
    discard ok()
    cast[pointer](h)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_state_dict_create*(): pointer
    {.exportc: "nimf_state_dict_create", cdecl, dynlib.} =
  ## Creates an empty state dict; append tensors with
  ## nimf_state_dict_append(), then persist with nimf_state_dict_save().
  try:
    let h = cast[ptr NimfStateDict](allocShared0(sizeof(NimfStateDict)))
    if h == nil: return failPtr("state-dict handle allocation failed")
    discard ok()
    cast[pointer](h)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_state_dict_append*(handle: pointer; name: cstring;
    quantHandle: pointer): cint
    {.exportc: "nimf_state_dict_append", cdecl, dynlib.} =
  ## Appends a copy of a quant tensor under `name`.
  if handle == nil or quantHandle == nil or name == nil:
    return fail("null state dict, quant handle, or name")
  try:
    let h = cast[ptr NimfStateDict](handle)
    let q = cast[ptr NimfQuantTensor](quantHandle)
    h.items.add(($name, q.qt))
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_state_dict_set_arch*(handle: pointer; arch: ptr int64): cint
    {.exportc: "nimf_state_dict_set_arch", cdecl, dynlib.} =
  if handle == nil or arch == nil: return fail("null state dict or arch array")
  let h = cast[ptr NimfStateDict](handle)
  for i in 0 ..< 5:
    h.arch[i] = int(cast[ptr UncheckedArray[int64]](arch)[i])
  ok()

proc nimf_state_dict_save*(path: cstring; handle: pointer;
    arch: ptr int64): cint
    {.exportc: "nimf_state_dict_save", cdecl, dynlib.} =
  if path == nil or handle == nil or arch == nil:
    return fail("null path, handle, or arch array")
  try:
    var a: array[5, int]
    for i in 0 ..< 5: a[i] = int(cast[ptr UncheckedArray[int64]](arch)[i])
    saveQuantStateDict($path, a, cast[ptr NimfStateDict](handle).items)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_state_dict_count*(handle: pointer): int64
    {.exportc: "nimf_state_dict_count", cdecl, dynlib.} =
  let h = cast[ptr NimfStateDict](handle)
  if h == nil:
    discard fail("null state dict")
    return -1
  int64(h.items.len)

proc nimf_state_dict_arch*(handle: pointer; outArch: ptr int64;
    cap: int64): cint
    {.exportc: "nimf_state_dict_arch", cdecl, dynlib.} =
  let h = cast[ptr NimfStateDict](handle)
  if h == nil: return fail("null state dict")
  if outArch == nil or cap < 5: return fail("arch output needs 5 int64 slots")
  for i in 0 ..< 5:
    cast[ptr UncheckedArray[int64]](outArch)[i] = int64(h.arch[i])
  ok()

proc nimf_state_dict_name*(handle: pointer; index: int64): cstring
    {.exportc: "nimf_state_dict_name", cdecl, dynlib.} =
  ## Returns an allocated string; free it with nimf_free_string().
  let h = cast[ptr NimfStateDict](handle)
  if h == nil: return failCString("null state dict")
  if index < 0 or index >= int64(h.items.len):
    return failCString("state-dict index out of range")
  result = allocCString(h.items[int(index)][0])
  if result == nil: discard fail("string allocation failed")

proc nimf_state_dict_tensor*(handle: pointer; index: int64): pointer
    {.exportc: "nimf_state_dict_tensor", cdecl, dynlib.} =
  ## Returns a NEW quant handle for entry `index`; destroy it separately.
  let h = cast[ptr NimfStateDict](handle)
  if h == nil: return failPtr("null state dict")
  if index < 0 or index >= int64(h.items.len):
    return failPtr("state-dict index out of range")
  try:
    let q = cast[ptr NimfQuantTensor](allocShared0(sizeof(NimfQuantTensor)))
    if q == nil: return failPtr("quant handle allocation failed")
    q.qt = h.items[int(index)][1]
    discard ok()
    cast[pointer](q)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_state_dict_destroy*(handle: pointer): cint
    {.exportc: "nimf_state_dict_destroy", cdecl, dynlib.} =
  if handle == nil: return ok()
  let h = cast[ptr NimfStateDict](handle)
  h.items = @[]
  deallocShared(handle)
  ok()

# =====================================================================
# Custom floats and APF (adaptive precision)
# =====================================================================

proc nimf_customfloat_encode*(ctxHandle: pointer; x: ptr cfloat; n: int64;
    exponentBits, mantissaBits: cint; outBytes: ptr uint8; capacity: int64;
    written: ptr int64): cint
    {.exportc: "nimf_customfloat_encode", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if n < 0 or exponentBits < 1 or mantissaBits < 0 or capacity < 0 or
      int(exponentBits) + int(mantissaBits) + 1 > 64 or
      (n > 0 and x == nil) or (capacity > 0 and outBytes == nil) or
      written == nil:
    return fail("invalid custom-float encode arguments (need exponentBits >= 1, total bits <= 64)")
  try:
    let fmt = newCustomFloat(int(exponentBits), int(mantissaBits))
    let bytes = beCustomfloatEncode(requireContext(ctxHandle),
      inputValues(x, int(n)), fmt)
    if int64(bytes.len) > capacity:
      return fail("custom-float output buffer too small")
    for i in 0 ..< bytes.len:
      cast[ptr UncheckedArray[uint8]](outBytes)[i] = bytes[i]
    written[] = int64(bytes.len)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_customfloat_decode*(ctxHandle: pointer; bytes: ptr uint8;
    byteLen, itemCount: int64; exponentBits, mantissaBits: cint;
    outData: ptr cfloat): cint
    {.exportc: "nimf_customfloat_decode", cdecl, dynlib.} =
  if ctxHandle == nil: return fail("null context")
  if byteLen < 0 or itemCount < 0 or exponentBits < 1 or mantissaBits < 0 or
      int(exponentBits) + int(mantissaBits) + 1 > 64 or
      (byteLen > 0 and bytes == nil) or (itemCount > 0 and outData == nil):
    return fail("invalid custom-float decode arguments (need exponentBits >= 1, total bits <= 64)")
  try:
    var packed = newSeq[uint8](int(byteLen))
    for i in 0 ..< packed.len:
      packed[i] = cast[ptr UncheckedArray[uint8]](bytes)[i]
    let values = beCustomfloatDecode(requireContext(ctxHandle), packed,
      newCustomFloat(int(exponentBits), int(mantissaBits)))
    if values.len != int(itemCount):
      return fail("decoded element count does not match itemCount")
    writeValues(outData, values)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_apf_build_dtype*(x: ptr cfloat; n: int64; grad: ptr cfloat;
    gradLen: int64; relErrorTol: cfloat; expMargin: cint;
    exponentBitsOut, mantissaBitsOut: ptr cint): cint
    {.exportc: "nimf_apf_build_dtype", cdecl, dynlib.} =
  if n < 0 or (n > 0 and x == nil) or gradLen < 0 or
      (gradLen > 0 and grad == nil) or exponentBitsOut == nil or
      mantissaBitsOut == nil:
    return fail("invalid APF dtype arguments")
  try:
    let cf = buildCustomDtypeForTensor(inputValues(x, int(n)),
      inputValues(grad, int(gradLen)), float32(relErrorTol), int(expMargin))
    exponentBitsOut[] = cint(cf.exponentBits)
    mantissaBitsOut[] = cint(cf.mantissaBits)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_apf_cast_training*(x: ptr cfloat; n: int64; grad: ptr cfloat;
    gradLen: int64; relErrorTol: cfloat; expMargin: cint; outBytes: ptr uint8;
    capacity: int64; written: ptr int64; exponentBitsOut,
    mantissaBitsOut: ptr cint): cint
    {.exportc: "nimf_apf_cast_training", cdecl, dynlib.} =
  if n < 0 or (n > 0 and x == nil) or gradLen < 0 or
      (gradLen > 0 and grad == nil) or capacity < 0 or
      (capacity > 0 and outBytes == nil) or written == nil or
      exponentBitsOut == nil or mantissaBitsOut == nil:
    return fail("invalid APF cast arguments")
  try:
    let r = apfCastForTraining(inputValues(x, int(n)),
      inputValues(grad, int(gradLen)), float32(relErrorTol), int(expMargin))
    if int64(r.data.len) > capacity:
      return fail("APF output buffer too small")
    for i in 0 ..< r.data.len:
      cast[ptr UncheckedArray[uint8]](outBytes)[i] = r.data[i]
    written[] = int64(r.data.len)
    exponentBitsOut[] = cint(r.cf.exponentBits)
    mantissaBitsOut[] = cint(r.cf.mantissaBits)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_apf_decode_training*(bytes: ptr uint8; byteLen: int64;
    exponentBits, mantissaBits: cint; outData: ptr cfloat;
    capacity: int64): int64
    {.exportc: "nimf_apf_decode_training", cdecl, dynlib.} =
  if byteLen < 0 or (byteLen > 0 and bytes == nil) or capacity < 0:
    discard fail("invalid APF decode arguments")
    return -1
  try:
    var packed = newSeq[uint8](int(byteLen))
    for i in 0 ..< packed.len:
      packed[i] = cast[ptr UncheckedArray[uint8]](bytes)[i]
    let cf = newCustomFloat(int(exponentBits), int(mantissaBits))
    let values = apfDecodeForTraining(packed, cf)
    if int64(values.len) > capacity:
      discard fail("output buffer too small")
      return -1
    writeValues(outData, values)
    discard ok()
    int64(values.len)
  except CatchableError as e:
    discard fail(e.msg)
    -1

# =====================================================================
# Tokenizers
# =====================================================================

proc nimf_tokenizer_load*(modelDir: cstring): pointer
    {.exportc: "nimf_tokenizer_load", cdecl, dynlib.} =
  ## Load a Hugging Face tokenizer directory (tokenizer.json +
  ## tokenizer_config.json, or tokenizer.model for SentencePiece).
  if modelDir == nil: return failPtr("null tokenizer directory")
  try:
    let tok = hf.newHFTokenizer($modelDir)
    let h = cast[ptr NimfTokenizer](allocShared0(sizeof(NimfTokenizer)))
    if h == nil: return failPtr("tokenizer handle allocation failed")
    h.tok = tok
    discard ok()
    cast[pointer](h)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_tokenizer_destroy*(handle: pointer): cint
    {.exportc: "nimf_tokenizer_destroy", cdecl, dynlib.} =
  if handle == nil: return ok()
  let h = cast[ptr NimfTokenizer](handle)
  h.tok = default(hf.HFTokenizer)   # release refs/tables before freeing memory
  deallocShared(handle)
  ok()

proc nimf_tokenizer_vocab_size*(handle: pointer): int64
    {.exportc: "nimf_tokenizer_vocab_size", cdecl, dynlib.} =
  let h = cast[ptr NimfTokenizer](handle)
  if h == nil:
    discard fail("null tokenizer")
    return -1
  int64(h.tok.vocabSize)

proc nimf_tokenizer_total_vocab_size*(handle: pointer): int64
    {.exportc: "nimf_tokenizer_total_vocab_size", cdecl, dynlib.} =
  let h = cast[ptr NimfTokenizer](handle)
  if h == nil:
    discard fail("null tokenizer")
    return -1
  int64(h.tok.totalVocabSize)

proc nimf_tokenizer_special_ids*(handle: pointer; bosOut, eosOut,
    padOut: ptr int64): cint
    {.exportc: "nimf_tokenizer_special_ids", cdecl, dynlib.} =
  let h = cast[ptr NimfTokenizer](handle)
  if h == nil: return fail("null tokenizer")
  if bosOut == nil or eosOut == nil or padOut == nil:
    return fail("null output pointer")
  bosOut[] = int64(h.tok.bosId)
  eosOut[] = int64(h.tok.eosId)
  padOut[] = int64(h.tok.padId)
  ok()

proc nimf_tokenizer_encode*(handle: pointer; text: cstring; addSpecial: cint;
    outIds: ptr int32; capacity: int64): int64
    {.exportc: "nimf_tokenizer_encode", cdecl, dynlib.} =
  ## Tokenize text; returns the token count or -1 on error/buffer too small.
  let h = cast[ptr NimfTokenizer](handle)
  if h == nil or text == nil:
    discard fail("null tokenizer or text")
    return -1
  try:
    let ids = hf.encode(h.tok, $text, addSpecial != 0)
    if outIds == nil and capacity == 0:
      discard ok()
      return int64(ids.len)
    if int64(ids.len) > capacity:
      discard fail("token output buffer too small")
      return -1
    for i in 0 ..< ids.len:
      cast[ptr UncheckedArray[int32]](outIds)[i] = int32(ids[i])
    discard ok()
    int64(ids.len)
  except CatchableError as e:
    discard fail(e.msg)
    -1

proc nimf_tokenizer_apply_chat*(handle: pointer; userText, systemText: cstring;
    outIds: ptr int32; capacity: int64): int64
    {.exportc: "nimf_tokenizer_apply_chat", cdecl, dynlib.} =
  let h = cast[ptr NimfTokenizer](handle)
  if h == nil or userText == nil:
    discard fail("null tokenizer or text")
    return -1
  try:
    let sys = if systemText == nil: "" else: $systemText
    let ids = hf.applyChatTemplate(h.tok, $userText, sys)
    if outIds == nil and capacity == 0:
      discard ok()
      return int64(ids.len)
    if int64(ids.len) > capacity:
      discard fail("token output buffer too small")
      return -1
    for i in 0 ..< ids.len:
      cast[ptr UncheckedArray[int32]](outIds)[i] = int32(ids[i])
    discard ok()
    int64(ids.len)
  except CatchableError as e:
    discard fail(e.msg)
    -1

proc nimf_tokenizer_decode*(handle: pointer; ids: ptr int32; count: int64;
    skipSpecial: cint): cstring
    {.exportc: "nimf_tokenizer_decode", cdecl, dynlib.} =
  ## Returns an allocated string; free with nimf_free_string().
  let h = cast[ptr NimfTokenizer](handle)
  if h == nil:
    return failCString("null tokenizer")
  if count < 0 or (count > 0 and ids == nil):
    return failCString("invalid token id buffer")
  try:
    var values = newSeq[int](int(count))
    for i in 0 ..< values.len:
      values[i] = int(cast[ptr UncheckedArray[int32]](ids)[i])
    result = allocCString(hf.decode(h.tok, values, skipSpecial != 0))
    if result == nil: discard fail("string allocation failed")
    else: discard ok()
  except CatchableError as e:
    result = failCString(e.msg)

proc nimf_sentencepiece_load*(path: cstring): pointer
    {.exportc: "nimf_sentencepiece_load", cdecl, dynlib.} =
  ## Load a raw SentencePiece `tokenizer.model` file.
  if path == nil: return failPtr("null model path")
  try:
    let model = sp.loadSentencePieceModel($path)
    let h = cast[ptr NimfSentencePiece](allocShared0(sizeof(NimfSentencePiece)))
    if h == nil: return failPtr("SentencePiece handle allocation failed")
    h.model = model
    discard ok()
    cast[pointer](h)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_sentencepiece_destroy*(handle: pointer): cint
    {.exportc: "nimf_sentencepiece_destroy", cdecl, dynlib.} =
  if handle == nil: return ok()
  let h = cast[ptr NimfSentencePiece](handle)
  h.model = nil
  deallocShared(handle)
  ok()

proc nimf_sentencepiece_encode*(handle: pointer; text: cstring;
    outIds: ptr int32; capacity: int64): int64
    {.exportc: "nimf_sentencepiece_encode", cdecl, dynlib.} =
  let h = cast[ptr NimfSentencePiece](handle)
  if h == nil or text == nil:
    discard fail("null SentencePiece model or text")
    return -1
  try:
    let ids = sp.encodeSentencePiece(h.model, $text)
    if outIds == nil and capacity == 0:
      discard ok()
      return int64(ids.len)
    if int64(ids.len) > capacity:
      discard fail("token output buffer too small")
      return -1
    for i in 0 ..< ids.len:
      cast[ptr UncheckedArray[int32]](outIds)[i] = int32(ids[i])
    discard ok()
    int64(ids.len)
  except CatchableError as e:
    discard fail(e.msg)
    -1

proc nimf_sentencepiece_decode*(handle: pointer; ids: ptr int32;
    count: int64): cstring
    {.exportc: "nimf_sentencepiece_decode", cdecl, dynlib.} =
  let h = cast[ptr NimfSentencePiece](handle)
  if h == nil:
    return failCString("null SentencePiece model")
  if count < 0 or (count > 0 and ids == nil):
    return failCString("invalid token id buffer")
  try:
    var values = newSeq[int](int(count))
    for i in 0 ..< values.len:
      values[i] = int(cast[ptr UncheckedArray[int32]](ids)[i])
    result = allocCString(sp.decodeSentencePiece(h.model, values))
    if result == nil: discard fail("string allocation failed")
    else: discard ok()
  except CatchableError as e:
    result = failCString(e.msg)

proc nimf_sentencepiece_piece_id*(handle: pointer; piece: cstring): int64
    {.exportc: "nimf_sentencepiece_piece_id", cdecl, dynlib.} =
  let h = cast[ptr NimfSentencePiece](handle)
  if h == nil or piece == nil:
    discard fail("null SentencePiece model or piece")
    return -2
  int64(sp.pieceId(h.model, $piece))

proc nimf_sentencepiece_normalizer*(handle: pointer): cstring
    {.exportc: "nimf_sentencepiece_normalizer", cdecl, dynlib.} =
  let h = cast[ptr NimfSentencePiece](handle)
  if h == nil:
    return failCString("null SentencePiece model")
  let name = sp.normalizerName(h.model)
  result = allocCString(name)
  if result == nil: discard fail("string allocation failed")

# =====================================================================
# Llama-family inference (NimQ checkpoints)
# =====================================================================

proc nimf_llama_config_load*(configPath: cstring; outInts: ptr int64;
    outFloats: ptr cfloat): cint
    {.exportc: "nimf_llama_config_load", cdecl, dynlib.} =
  ## Reads config.json and writes 7 ints + 2 floats:
  ## vocab, hidden, intermediate, layers, heads, kv_heads, max_positions,
  ## rms_norm_eps, rope_theta.
  if configPath == nil or outInts == nil or outFloats == nil:
    return fail("null config path or output buffer")
  try:
    let cfg = nfinfer.newLlamaLConfig($configPath)
    let ints = cast[ptr UncheckedArray[int64]](outInts)
    ints[0] = int64(cfg.vocabSize)
    ints[1] = int64(cfg.hiddenSize)
    ints[2] = int64(cfg.intermediateSize)
    ints[3] = int64(cfg.nLayers)
    ints[4] = int64(cfg.nHeads)
    ints[5] = int64(cfg.nKVHeads)
    ints[6] = int64(cfg.maxPositionEmbeddings)
    let floats = cast[ptr UncheckedArray[cfloat]](outFloats)
    floats[0] = cfloat(cfg.rmsNormEps)
    floats[1] = cfloat(cfg.ropeTheta)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_llama_load*(ctxHandle: pointer; configPath, weightsPath: cstring;
    uploadResident: cint): pointer
    {.exportc: "nimf_llama_load", cdecl, dynlib.} =
  ## Loads a Llama config + NIMQ weights.  Set uploadResident=1 to pin all
  ## quantized weights on the GPU immediately (faster decode, more VRAM).
  if ctxHandle == nil or configPath == nil or weightsPath == nil:
    return failPtr("null context, config path, or weights path")
  try:
    let cfg = nfinfer.newLlamaLConfig($configPath)
    var model = nfinfer.newLlamaModelL(cfg)
    nfinfer.loadLlamaWeightsL(model, $weightsPath)
    if uploadResident != 0:
      nfinfer.uploadAllResidentWeights(model, requireContext(ctxHandle))
    let h = cast[ptr NimfLlama](allocShared0(sizeof(NimfLlama)))
    if h == nil: return failPtr("Llama handle allocation failed")
    h.model = model
    discard ok()
    cast[pointer](h)
  except CatchableError as e:
    failPtr(e.msg)
  except Defect as e:
    failPtr("internal error while loading model: " & e.msg)

proc nimf_llama_config*(handle: pointer; outInts: ptr int64;
    outFloats: ptr cfloat): cint
    {.exportc: "nimf_llama_config", cdecl, dynlib.} =
  let h = cast[ptr NimfLlama](handle)
  if h == nil: return fail("null Llama model")
  if outInts == nil or outFloats == nil:
    return fail("null output buffer")
  let ints = cast[ptr UncheckedArray[int64]](outInts)
  ints[0] = int64(h.model.config.vocabSize)
  ints[1] = int64(h.model.config.hiddenSize)
  ints[2] = int64(h.model.config.intermediateSize)
  ints[3] = int64(h.model.config.nLayers)
  ints[4] = int64(h.model.config.nHeads)
  ints[5] = int64(h.model.config.nKVHeads)
  ints[6] = int64(h.model.config.maxPositionEmbeddings)
  let floats = cast[ptr UncheckedArray[cfloat]](outFloats)
  floats[0] = cfloat(h.model.config.rmsNormEps)
  floats[1] = cfloat(h.model.config.ropeTheta)
  ok()

proc nimf_llama_generate*(ctxHandle, modelHandle, tokenizerHandle: pointer;
    prompt: cstring; systemText: cstring; maxNewTokens: int64;
    temperature, topP: cfloat; useChatTemplate: cint): cstring
    {.exportc: "nimf_llama_generate", cdecl, dynlib.} =
  ## Generates text.  Returns an allocated string; free with
  ## nimf_free_string().  useChatTemplate=0 tokenizes the prompt verbatim.
  if ctxHandle == nil or modelHandle == nil or tokenizerHandle == nil or prompt == nil:
    return failCString("null context, model, tokenizer, or prompt")
  try:
    let ctx = requireContext(ctxHandle)
    let model = cast[ptr NimfLlama](modelHandle).model
    let tok = cast[ptr NimfTokenizer](tokenizerHandle).tok
    let sysText = if systemText == nil: "" else: $systemText
    let promptIds = if useChatTemplate != 0:
                      hf.applyChatTemplate(tok, $prompt, sysText)
                    else:
                      hf.encode(tok, $prompt, true)
    for id in promptIds:
      if id < 0 or id >= model.config.vocabSize:
        return failCString("prompt token id " & $id & " is outside the model vocab (" &
                           $model.config.vocabSize & "); use a matching tokenizer")
    result = allocCString(nfinfer.generate(model, tok, ctx, promptIds,
      int(maxNewTokens), float32(temperature), float32(topP)))
    if result == nil: discard fail("string allocation failed")
    else: discard ok()
  except CatchableError as e:
    result = failCString(e.msg)
  except Defect as e:
    result = failCString("internal generation error: " & e.msg)

proc nimf_llama_destroy*(handle: pointer): cint
    {.exportc: "nimf_llama_destroy", cdecl, dynlib.} =
  if handle == nil: return ok()
  let h = cast[ptr NimfLlama](handle)
  h.model = default(nfinfer.LlamaModelL)   # release seqs/refs
  deallocShared(handle)
  ok()

# =====================================================================
# Micro-VM: program builder and dispatch
# =====================================================================

proc nimf_vm_create*(): pointer
    {.exportc: "nimf_vm_create", cdecl, dynlib.} =
  try:
    let h = cast[ptr NimfVmProgram](allocShared0(sizeof(NimfVmProgram)))
    if h == nil: return failPtr("VM program allocation failed")
    discard ok()
    cast[pointer](h)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_vm_destroy*(handle: pointer): cint
    {.exportc: "nimf_vm_destroy", cdecl, dynlib.} =
  if handle == nil: return ok()
  let h = cast[ptr NimfVmProgram](handle)
  h.prog.words = @[]
  deallocShared(handle)
  ok()

proc nimf_vm_len*(handle: pointer): int64
    {.exportc: "nimf_vm_len", cdecl, dynlib.} =
  let h = checkVm(handle)
  if h == nil:
    discard fail("null VM program")
    return -1
  int64(h.prog.words.len)

proc nimf_vm_words*(handle: pointer; outWords: ptr uint32;
    cap: int64): int64
    {.exportc: "nimf_vm_words", cdecl, dynlib.} =
  let h = checkVm(handle)
  if h == nil:
    discard fail("null VM program")
    return -1
  let n = min(int(cap), h.prog.words.len)
  if n > 0 and outWords == nil:
    discard fail("null output buffer")
    return -1
  for i in 0 ..< n:
    cast[ptr UncheckedArray[uint32]](outWords)[i] = h.prog.words[i]
  int64(h.prog.words.len)

proc nimf_vm_set_words*(handle: pointer; words: ptr uint32;
    count: int64): cint
    {.exportc: "nimf_vm_set_words", cdecl, dynlib.} =
  let h = checkVm(handle)
  if h == nil: return fail("null VM program")
  if count < 0 or (count > 0 and words == nil):
    return fail("invalid word buffer")
  try:
    var w = newSeq[uint32](int(count))
    for i in 0 ..< w.len:
      w[i] = cast[ptr UncheckedArray[uint32]](words)[i]
    h.prog.words = w
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_vm_emit_word*(handle: pointer; word: uint32): cint
    {.exportc: "nimf_vm_emit_word", cdecl, dynlib.} =
  let h = checkVm(handle)
  if h == nil: return fail("null VM program")
  h.prog.emit(word)
  ok()

proc nimf_vm_emit*(handle: pointer; opcode: uint32; a, b, c, value: uint32): cint
    {.exportc: "nimf_vm_emit", cdecl, dynlib.} =
  ## Generic instruction emitter.  Operand meaning depends on opcode:
  ##   math/U ops:       a=dest, b=src1, c=src2 (value=extra word/float bits)
  ##   load/store:       a=reg,  b=base, c=offset
  ##   load_const:       a=reg,  value=IEEE-754 bits
  ##   add_const/mul:    a=dest, b=src, value=IEEE-754 bits
  ##   for/if:           a=count/cond register
  ##   grid barrier:     a=control index
  ## See `nimformers.vm.Op` in the Python package for symbolic names.
  let h = checkVm(handle)
  if h == nil: return fail("null VM program")
  var p = h.prog
  case opcode
  of micro.OP_NOP: p.emitNop()
  of micro.OP_HALT: p.emitHalt()
  of micro.OP_LOAD_F32: p.emitLoadF32(a, b, c)
  of micro.OP_STORE_F32: p.emitStoreF32(a, b, c)
  of micro.OP_LOAD_U8: p.emitLoadU8(a, b, c)
  of micro.OP_LOAD_CONST: p.emitLoadConstF32(a, cast[float32](value))
  of micro.OP_ADD: p.emitAdd(a, b, c)
  of micro.OP_MUL: p.emitMul(a, b, c)
  of micro.OP_FMA: p.emitFma(a, b, c, value)
  of micro.OP_LUT_U8: p.emitLutU8(a, b)
  of micro.OP_FOR_BEGIN: p.emitForBegin(a)
  of micro.OP_FOR_END: p.emitForEnd()
  of micro.OP_BARRIER_TG: p.emitBarrierTg()
  of micro.OP_TG_LOAD_F32: p.emitTgLoadF32(a, b, c)
  of micro.OP_TG_STORE_F32: p.emitTgStoreF32(a, b, c)
  of micro.OP_SUB: p.emitSub(a, b, c)
  of micro.OP_DIV: p.emitDiv(a, b, c)
  of micro.OP_EXP: p.emitExp(a, b)
  of micro.OP_NEG: p.emitNeg(a, b)
  of micro.OP_SIGMOID: p.emitSigmoid(a, b)
  of micro.OP_SOFTPLUS: p.emitSoftplus(a, b)
  of micro.OP_RSQRT: p.emitRsqrt(a, b)
  of micro.OP_LOAD_F32_IDX: p.emitLoadF32Idx(a, b)
  of micro.OP_STORE_F32_IDX: p.emitStoreF32Idx(a, b)
  of micro.OP_ADD_CONST: p.emitAddConst(a, b, cast[float32](value))
  of micro.OP_MUL_CONST: p.emitMulConst(a, b, cast[float32](value))
  of micro.OP_CMP_LT: p.emitCmpLt(a, b, c)
  of micro.OP_IF_BEGIN: p.emitIfBegin(a)
  of micro.OP_IF_END: p.emitIfEnd()
  of micro.OP_LOAD_LID: p.emitLoadLid(a)
  of micro.OP_LOAD_TGID: p.emitLoadTgid(a)
  of micro.OP_U_TGID: p.emitUTgid(a)
  of micro.OP_U_LID: p.emitULid(a)
  of micro.OP_U_CONST: p.emitUConst(a, value)
  of micro.OP_U_ADD: p.emitUAdd(a, b, c)
  of micro.OP_U_MUL: p.emitUMul(a, b, c)
  of micro.OP_U_ADDK: p.emitUAddK(a, b, value)
  of micro.OP_U_MULK: p.emitUMulK(a, b, value)
  of micro.OP_U_ANDK: p.emitUAndK(a, b, value)
  of micro.OP_U_SHRK: p.emitUShrK(a, b, value)
  of micro.OP_U_SHLK: p.emitUShlK(a, b, value)
  of micro.OP_U_SHR: p.emitUShr(a, b, c)
  of micro.OP_U_LOAD_U8: p.emitULoadU8(a, b, value)
  of micro.OP_U_LOAD_CONSTF: p.emitULoadConstF(a, b, value)
  of micro.OP_U_LOAD_IN: p.emitULoadIn(a, b, value)
  of micro.OP_U_STORE_F32: p.emitUStoreF32(a, b, value)
  of micro.OP_U_TO_R: p.emitUToR(a, b)
  of micro.OP_R_TO_U: p.emitRToU(a, b)
  of micro.OP_U_LOAD_F32: p.emitULoadF32(a, b, value)
  of micro.OP_U_ATOMIC_ADD: p.emitUAtomicAdd(a, b, value)
  of micro.OP_GRID_BARRIER: p.emitGridBarrier(a)
  of micro.OP_GRID_EPILOGUE: p.emitGridEpilogue(a)
  of micro.OP_BASE_F32: p.emitBaseF32(a)
  of micro.OP_BASE_U8: p.emitBaseU8(a)
  of micro.OP_BASE_LUT: p.emitBaseLut(a)
  of micro.OP_BASE_IN: p.emitBaseIn(a)
  of micro.OP_U_LOAD_META: p.emitULoadMeta(a, b, value)
  of micro.OP_VTG_BEGIN: p.emitVtgBegin(a, b, c)
  of micro.OP_VTG_END: p.emitVtgEnd()
  of micro.OP_LOAD_VACT: p.emitLoadVact(a)
  of micro.OP_U_LOAD_U32: p.emitULoadU32(a, b, value)
  else:
    return fail("unsupported VM opcode: 0x" & toHex(opcode.int, 2))
  h.prog = p
  ok()

proc nimf_vm_save*(handle: pointer; path: cstring): cint
    {.exportc: "nimf_vm_save", cdecl, dynlib.} =
  if path == nil: return fail("null path")
  let h = checkVm(handle)
  if h == nil: return fail("null VM program")
  try:
    saveBvm(h.prog, $path)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_vm_load*(path: cstring): pointer
    {.exportc: "nimf_vm_load", cdecl, dynlib.} =
  if path == nil: return failPtr("null path")
  try:
    let prog = loadBvm($path)
    let h = cast[ptr NimfVmProgram](allocShared0(sizeof(NimfVmProgram)))
    if h == nil: return failPtr("VM program allocation failed")
    h.prog = prog
    discard ok()
    cast[pointer](h)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_vm_merge*(handles: ptr pointer; count: int64): pointer
    {.exportc: "nimf_vm_merge", cdecl, dynlib.} =
  if count < 0 or (count > 0 and handles == nil):
    return failPtr("invalid VM handle array")
  try:
    var parts = newSeq[micro.VmProgram](int(count))
    for i in 0 ..< int(count):
      let hp = cast[ptr UncheckedArray[pointer]](handles)[i]
      if hp == nil: return failPtr("null VM program in array")
      parts[i] = cast[ptr NimfVmProgram](hp).prog
    let merged = micro.mergePrograms(parts)
    let h = cast[ptr NimfVmProgram](allocShared0(sizeof(NimfVmProgram)))
    if h == nil: return failPtr("VM program allocation failed")
    h.prog = merged
    discard ok()
    cast[pointer](h)
  except CatchableError as e:
    failPtr(e.msg)

proc nimf_vm_lower_msl*(handle: pointer; kernelName: cstring): cstring
    {.exportc: "nimf_vm_lower_msl", cdecl, dynlib.} =
  ## Returns allocated Metal source; free with nimf_free_string().
  let h = checkVm(handle)
  if h == nil or kernelName == nil:
    return failCString("null VM program or kernel name")
  try:
    result = allocCString(lowerBytecodeToMetal(h.prog, $kernelName))
    if result == nil: discard fail("string allocation failed")
    else: discard ok()
  except CatchableError as e:
    result = failCString(e.msg)

proc nimf_vm_lower_glsl*(handle: pointer; kernelName: cstring): cstring
    {.exportc: "nimf_vm_lower_glsl", cdecl, dynlib.} =
  ## Returns allocated GLSL source; free with nimf_free_string().
  let h = checkVm(handle)
  if h == nil or kernelName == nil:
    return failCString("null VM program or kernel name")
  try:
    result = allocCString(lowerBytecodeToGlsl(h.prog, $kernelName))
    if result == nil: discard fail("string allocation failed")
    else: discard ok()
  except CatchableError as e:
    result = failCString(e.msg)

proc nimf_vm_key*(handle: pointer; tag: cstring): cstring
    {.exportc: "nimf_vm_key", cdecl, dynlib.} =
  ## Deterministic program key (no GPU session needed).
  let h = checkVm(handle)
  if h == nil: return failCString("null VM program")
  try:
    let tagStr = if tag == nil: "" else: $tag
    result = allocCString(programKeyNoSession(h.prog, tagStr))
    if result == nil: discard fail("string allocation failed")
    else: discard ok()
  except CatchableError as e:
    result = failCString(e.msg)

proc nimf_vm_run*(ctxHandle, progHandle: pointer; memory: ptr cfloat;
    memoryLen, nThreads, tgFloats, tgSize: int64): cint
    {.exportc: "nimf_vm_run", cdecl, dynlib.} =
  ## Runs a VM program over `memory` in place.  `tgFloats` is the number of
  ## float32 slots of threadgroup memory; `tgSize` the threads per group
  ## (0 = backend default).
  if ctxHandle == nil or progHandle == nil: return fail("null context or program")
  if memoryLen <= 0 or memory == nil or nThreads <= 0 or tgFloats < 0 or tgSize < 0:
    return fail("invalid micro-VM arguments")
  try:
    let ctx = requireContext(ctxHandle)
    let prog = cast[ptr NimfVmProgram](progHandle).prog
    var mem = inputValues(memory, int(memoryLen))
    let dispatched = runMicroVm(ctx, prog, mem, int(nThreads),
      tgBytes = max(4, int(tgFloats) * 4), tag = "capi", tgSize = int(tgSize))
    if not dispatched: return fail("micro-VM dispatch failed")
    writeValues(memory, mem)
    ok()
  except CatchableError as e:
    fail(e.msg)

proc nimf_micro_vm_dispatch*(ctxHandle: pointer; code: ptr uint32; codeLen: int64;
    memory: ptr cfloat; memoryLen: int64; nThreads, tgFloats, tgSize: int64): cint
    {.exportc: "nimf_micro_vm_dispatch", cdecl, dynlib.} =
  ## Backwards-compatible raw-bytecode dispatch (equivalent to building a
  ## VM program from `code` and calling nimf_vm_run).
  if ctxHandle == nil or code == nil or memory == nil or codeLen <= 0 or
      memoryLen <= 0 or nThreads <= 0 or tgFloats < 0 or tgSize <= 0:
    return fail("invalid micro-VM arguments")
  try:
    var words = newSeq[uint32](int(codeLen))
    for i in 0 ..< words.len:
      words[i] = cast[ptr UncheckedArray[uint32]](code)[i]
    var prog: micro.VmProgram
    prog.words = words
    var mem = inputValues(memory, int(memoryLen))
    let dispatched = runMicroVm(requireContext(ctxHandle), prog, mem, int(nThreads),
      tgBytes = max(4, int(tgFloats) * 4), tag = "capi-raw", tgSize = int(tgSize))
    if not dispatched: return fail("micro-VM dispatch failed")
    writeValues(memory, mem)
    ok()
  except CatchableError as e:
    fail(e.msg)
