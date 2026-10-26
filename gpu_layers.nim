## gpu_layers.nim — các lớp còn lại của đường infer 100% GPU, 1 command buffer / token.
##
## Mọi kernel ở đây là BYTECODE micro-VM (fp4_gemv_vm.nim) đi qua jit_vm
## (lower -> compile JIT 1 lần / shape -> cache). KHÔNG có MSL nhúng trong Nim.
##
##   * GpuLora        : delta LoRA  y += scale * B (A x)   (2 gemv fp32 nhỏ)
##   * GpuLin         : Linear của block A (fp4 resident hoặc fp32) + bias + LoRA
##   * GpuAttnLayer   : block A = LN1 -> QKV -> Linear++ step -> out -> +res
##                                -> LN2 -> FC1+APFLU -> FC2 -> +res
##
## Tất cả encode vào session đang mở; không sessionRead giữa chừng.

import std/[tables, os, strutils]
import backend
import vendor/bybylang/gpubackend as byby
import session_vm
import session_vm_dispatch
import jit_vm
import fp4_gemv_vm
import fp4_gemv_gpu
import quant      # QuantTensor, QuantKind
import nimformer   # Linear (weightQ/useQuant/weight/bias/loraA..)

# ── JIT cache riêng cho kernel infer (compile NGAY, hotThreshold=0) ─────
var gVmCache = jit_vm.initJitCache(jit_vm.JitConfig(
  hotThreshold: 0, enableMetalJit: true, enableVulkanJit: true,
  enableCpuJit: false, maxCacheEntries: 1024))

# Alias used by fp4_gemv_gpu / encode path
var gJitV2* = gVmCache

proc vmKernel*(p: VmProgram; tag: string): string =
  ## Lower bytecode -> Metal (jit_vm) -> compile -> tên pipeline. Gọi NGOÀI
  ## session (lúc prepare) để việc compile không nằm trong hot path.
  ## Program dùng op U-* KHÔNG có đường interpreter -> compile fail = raise ngay
  ## (thay vì rơi vào micro_vm_kernel rồi ra kết quả rác im lặng).
  let kn = jit_vm.dispatchProgram(gVmCache, p, tag)
  if kn.len == 0 or kn == "micro_vm_kernel":
    raise newException(CatchableError,
      "vmKernel[" & tag & "]: JIT Metal THAT BAI (program dung op U-*, khong co " &
      "duong interpreter). Bat BYBY_JIT_DUMP_SOURCE=1 de xem source + loi compile.")
  kn

proc vmKernelForBe*(p: VmProgram; tag: string; be: byby.GpuBackend): string =
  ## Chọn Metal MSL vs Vulkan GLSL theo backend.
  if be == byby.gbVulkan:
    result = jit_vm.kernelNameFor(gJitV2, p, tag)
  else:
    result = jit_vm.dispatchProgramNoSession(gJitV2, p, tag)
  if result.len == 0 or result == "micro_vm_kernel":
    raise newException(CatchableError, "vmKernelForBe[" & tag & "]: JIT that bai")

# ── buffer dummy: kernel JIT luôn khai báo buffer(1..3) -> bind slot trống ─
# Per-backend: một gDummy toàn cục gây nil handle khi process chuyển Metal↔Vulkan
# (bisect/sweep) vì gDummyReady chỉ set 1 lần.
var gDummyMetal: byby.SessionHandle
var gDummyVulkan: byby.SessionHandle
var gDummyMetalReady = false
var gDummyVulkanReady = false
proc ensureDummy*(be: byby.GpuBackend) =
  if be == byby.gbVulkan:
    if not gDummyVulkanReady:
      gDummyVulkan = byby.sessionUploadPrivate(be, newSeq[float32](64))
      gDummyVulkanReady = true
  else:
    if not gDummyMetalReady:
      gDummyMetal = byby.sessionUploadPrivate(be, newSeq[float32](64))
      gDummyMetalReady = true

proc gDummyFor*(be: byby.GpuBackend): byby.SessionHandle =
  ensureDummy(be)
  if be == byby.gbVulkan: gDummyVulkan else: gDummyMetal

# Backward-compat alias: most call sites still say `gDummy` after ensureDummy(be).
template gDummy*: byby.SessionHandle =
  ## Prefer gDummyFor(be). This resolves to Metal dummy if ready, else Vulkan.
  if gDummyMetalReady: gDummyMetal
  elif gDummyVulkanReady: gDummyVulkan
  else: gDummyMetal

proc bindOr(idx: int; h: byby.SessionHandle; offElems: int) =
  ## API chung: route theo h.kind.
  let isVk = h.kind == byby.gbVulkan
  let dummy = gDummyFor(if isVk: byby.gbVulkan else: byby.gbMetal)
  let mbuf = if h.metalHandle != nil: h.metalHandle
             elif dummy.metalHandle != nil: dummy.metalHandle
             else: nil
  discard sessionEncBindVkOrMetal(isVk, idx, mbuf, h.vulkanHandle, offElems)

proc encodeVmKernel*(kname: string; nThreads, tgSize, tgBytes: int;
                     memOut: byby.SessionHandle; memOff: int;
                     wU8, constF, inX: byby.SessionHandle; inOff: int) =
  ## API chung Metal+Vulkan. Layout:
  ##   Metal slots 1..4 = mem / u8 / const / in
  ##   GLSL binding 0..4 = MemF32 / MemU8 / Lut / Input / MemU32
  let isVk = memOut.kind == byby.gbVulkan
  let glslSrc = if isVk: jit_vm.cachedGlslSource(gJitV2, kname) else: ""
  if isVk and glslSrc.len == 0:
    raise newException(CatchableError,
      "encodeVmKernel[" & kname & "]: khong co GLSL source (can kernelNameFor truoc)")
  if not sessionEncPipeline(isVk, kname, glslSrc):
    raise newException(CatchableError, "encodeVmKernel: khong bind duoc pipeline " & kname)
  if isVk:
    bindOr(0, memOut, memOff)
    bindOr(1, wU8, 0)
    bindOr(2, constF, 0)
    bindOr(3, inX, inOff)
    bindOr(4, memOut, memOff)
  else:
    bindOr(1, memOut, memOff)
    bindOr(2, wU8, 0)
    bindOr(3, constF, 0)
    bindOr(4, inX, inOff)
  discard sessionEncTgBytes(isVk, 0, max(tgBytes, 16))
  let maxTG = sessionEncMaxTg(isVk)
  if maxTG > 0 and tgSize > maxTG:
    raise newException(CatchableError,
      "encodeVmKernel[" & kname & "]: can " & $tgSize & " tg nhung max=" & $maxTG)
  if not sessionEncDispatch(isVk, nThreads, tgSize, tgBytes):
    raise newException(CatchableError, "encodeVmKernel[" & kname & "]: dispatch that bai")

# ── kernel dùng chung (không phụ thuộc shape) ───────────────────────────
var gKnBias = ""
var gKnBiasApflu = ""
var gEluPlus1Kn = ""
var gEluPlus1BackwardKn = ""
var gGpuKernelsReady = false
var gGpuKernelsBe: byby.GpuBackend
proc ensureGpuKernels*(be: byby.GpuBackend) =
  ## Compile bias kernels CHO DUNG backend. Neu doi Metal<->Vulkan phai compile lai
  ## (ten kernel + GLSL/MSL cache khac nhau).
  if gGpuKernelsReady and gGpuKernelsBe == be: return
  ensureDummy(be)
  gKnBias = vmKernelForBe(buildBiasActProgram(false), "bias", be)
  gKnBiasApflu = vmKernelForBe(buildBiasActProgram(true), "bias_apflu", be)
  # Elu kernels are prepared lazily; clear so next encode recompiles for this be.
  gEluPlus1Kn = ""
  gEluPlus1BackwardKn = ""
  gGpuKernelsBe = be
  gGpuKernelsReady = true

proc encodeBias*(hMem: byby.SessionHandle; memOff: int; hBias: byby.SessionHandle;
                 n: int; apflu = false) =
  ## mem[memOff ..< memOff+n] = act(mem + bias)
  let tg = min(n, 256)
  encodeVmKernel(if apflu: gKnBiasApflu else: gKnBias, n, tg, 16,
                 hMem, memOff, gDummy, hBias, gDummy, 0)

proc encodeEluPlus1*(ctx: Backend; hInput, hOutput: byby.SessionHandle; n: int) =
  ## GPU JIT feature map for Linear++ training: phi(x)=ELU(x)+1.
  let be = ctx.kind.toByby()
  ensureGpuKernels(be)
  if gEluPlus1Kn.len == 0:
    gEluPlus1Kn = vmKernelForBe(buildEluPlus1Program(max(n, 1)), "lp_elu_plus1", be)
  encodeVmKernel(gEluPlus1Kn, n, min(n, 256), 16,
                 hOutput, 0, gDummy, gDummy, hInput, 0)

proc encodeEluPlus1Backward*(ctx: Backend; hInput, hOutput: byby.SessionHandle; n: int) =
  ## GPU JIT feature-map derivative. hInput layout is [x[n] | dPhi[n]].
  let be = ctx.kind.toByby()
  ensureGpuKernels(be)
  if gEluPlus1BackwardKn.len == 0:
    gEluPlus1BackwardKn = vmKernelForBe(buildEluPlus1BackwardProgram(max(n, 1)), "lp_elu_plus1_bwd", be)
  encodeVmKernel(gEluPlus1BackwardKn, n, min(n, 256), 16,
                 hOutput, 0, gDummy, gDummy, hInput, 0)

# ── LoRA trên GPU ──────────────────────────────────────────────────────
type
  GpuLora* = object
    aBuf*, bBuf*: byby.SessionHandle
    hT*: byby.SessionHandle          ## scratch [r]
    r*, inF*, outF*: int
    knA*, knB*: string
    bSmallK*: bool                   ## knB dùng kernel 1-thread/hàng (r <= 32)
    ready*: bool

var gLoraGpu* = initTable[string, GpuLora]()

proc prepareLoraGpu*(ctx: Backend; name: string; loraA, loraB: seq[float32];
                     r, inF, outF: int; scale: float32): bool =
  ## Upload A [r,inF] và B [outF,r] (fp32, nhỏ) 1 lần. Kernel = 2 gemv bytecode.
  if r <= 0 or loraA.len != r * inF or loraB.len != outF * r: return false
  if name in gLoraGpu and gLoraGpu[name].ready: return true
  let be = ctx.kind.toByby()
  ensureDummy(be)
  var g: GpuLora
  g.r = r; g.inF = inF; g.outF = outF
  g.aBuf = byby.sessionUploadPrivate(be, loraA)
  g.bBuf = byby.sessionUploadPrivate(be, loraB)
  g.hT = byby.sessionUploadPrivate(be, newSeq[float32](max(r, 64)))   # ghi được: kernel ghi vào memF32
  # Vulkan: kernelNameFor + cache GLSL (encodeVmKernel can cachedGlslSource).
  # Metal: vmKernel / dispatch MSL. Truoc day luon vmKernel(Metal) -> Vulkan warmup fail.
  let progA = buildF32GemvProgram(inF, r, 1.0'f32, false)
  g.knA = vmKernelForBe(progA, "lora_a", be)
  # B là [outF, r]: K = r (bé) và N = outF. Với r <= 32, kernel 32-lane ở
  # trên tiêu 5 barrier chỉ cho một dot-product 4..16 phần tử; kernel
  # 1-thread-1-hàng (không reduction) rẻ hơn hẳn.
  if r <= 32:
    let progB = buildF32GemvSmallKProgram(r, outF, scale, true)
    g.knB = vmKernelForBe(progB, "lora_b_sk", be)
    g.bSmallK = true
  else:
    let progB = buildF32GemvProgram(r, outF, scale, true)
    g.knB = vmKernelForBe(progB, "lora_b", be)
    g.bSmallK = false
  g.ready = true
  gLoraGpu[name] = g
  true

var gDisableLora* = false
  ## chan doan (bisect): true = bo qua MOI delta LoRA luc encode -> model goc.

proc encodeLoraDelta*(name: string; hIn: byby.SessionHandle; inOff: int;
                      hOut: byby.SessionHandle; outOff: int): bool =
  ## hOut[outOff ..< +outF] += scale * B (A hIn[inOff ..< +inF])
  if gDisableLora: return false
  if name notin gLoraGpu:
    # Trước đây return false im lặng → LoRA bị bỏ, logits lệch mà không log.
    if getEnv("BYBY_LORA_VERBOSE", "0") != "0":
      stderr.writeLine "[encodeLoraDelta] name missing in gLoraGpu: " & name
    return false
  let g = gLoraGpu[name]
  encodeVmKernel(g.knA, g.r * GemvTg, GemvTg, f32GemvTgBytes(),
                 g.hT, 0, gDummy, g.aBuf, hIn, inOff)
  # sessionDispatch inserts the Metal buffer barrier after each VM dispatch;
  # knB therefore sees the hT values written by knA on Intel as well.
  if g.bSmallK:
    encodeVmKernel(g.knB, g.outF, 64, f32GemvSmallKTgBytes(),
                   hOut, outOff, gDummy, g.bBuf, g.hT, 0)
  else:
    encodeVmKernel(g.knB, g.outF * GemvTg, GemvTg, f32GemvTgBytes(),
                   hOut, outOff, gDummy, g.bBuf, g.hT, 0)
  true

# ── Linear GPU (fp4 resident | fp32) + bias + LoRA ──────────────────────
type
  GpuLin* = object
    name*: string
    isFp4*: bool
    wF32*: byby.SessionHandle
    knF32*: string
    K*, N*: int
    biasH*: byby.SessionHandle
    hasBias*: bool
    hasLora*: bool
    isInt4*: bool
    int4WBuf*: byby.SessionHandle
    ready*: bool

type
  Int4LinCacheGpu = object
    wBuf, scZpBuf: byby.SessionHandle
    knName: string
    N*, groupSize*: int
    ready: bool

var gInt4CacheGpu = initTable[string, Int4LinCacheGpu]()

proc uploadInt4WeightOnce(ctx: Backend; name: string; qt: QuantTensor; K, N: int) =
  if name in gInt4CacheGpu and gInt4CacheGpu[name].ready: return
  let be = ctx.kind.toByby()
  let groupSize = qt.groupSize
  doAssert int4AsymSupported(K, groupSize),
    "uploadInt4WeightOnce " & name & ": K=" & $K & " groupSize=" & $groupSize & " khong ho tro tren GPU"
  var c: Int4LinCacheGpu
  c.wBuf = sessionUploadBytesShared(be, qt.data)
  c.knName = vmKernelForBe(buildInt4AsymGemvProgram(K, N, groupSize),
    "int4_gemv_" & $K & "x" & $N & "_g" & $groupSize, be)
  c.N = N; c.groupSize = groupSize
  # scale + zero_point nối tiếp nhau (khớp buildInt4AsymGemvProgram: [scale(N*nG) | zp(N*nG)])
  var scZp = newSeq[float32](qt.scale.len + qt.zero_point.len)
  for i in 0 ..< qt.scale.len: scZp[i] = qt.scale[i]
  for i in 0 ..< qt.zero_point.len: scZp[qt.scale.len + i] = qt.zero_point[i]
  c.scZpBuf = byby.sessionUploadPrivate(be, scZp)
  c.ready = true
  gInt4CacheGpu[name] = c

proc markFp4Lin*(ctx: Backend; name: string; bias: seq[float32]; K, N: int): GpuLin
proc uploadBiasOrRaise(name: string; bias: seq[float32]; N: int): seq[float32]

var gLinCache* = initTable[string, GpuLin]()   ## 1 nguon su that duy nhat: MOI Linear (R lan A, moi kind) tra qua day.

proc prepareGpuLinForLinear*(ctx: Backend; name: string; lin: var Linear): GpuLin =
  ## Chuẩn bị GPU Linear cho MỌI dạng weight (R lẫn A: fp4-KM resident,
  ## int4-asymmetric resident, hoặc fp32 fallback random-init lúc chưa load).
  ## Dùng CHUNG bytecode-gemv (khong MSL) cho ca 3 nhanh, va LUON ghi ket qua
  ## vao gLinCache[name] - day la NGUON SU THAT DUY NHAT ma encodeLinByName
  ## (dung cho R qua fp4GemvAt) tra cuu. Truoc ban nay co 2 bang rieng
  ## (gFp4CacheGpu/gInt4CacheGpu) va nhanh fallback fp32 KHONG duoc dang ky o
  ## dau ca -> encodeLinByName tra ve false -> crash "chua prepareGpuLinForLinear"
  ## dung luc warmup, dung o block dau tien roi vao nhanh fp32/fallback.
  if name in gLinCache and gLinCache[name].ready:
    return gLinCache[name]
  let be = ctx.kind.toByby()
  ensureGpuKernels(be)
  let K = lin.inF; let N = lin.outF
  if lin.useQuant and lin.weightQ.kind == qkFp4KM:
    if name notin gFp4CacheGpu or not gFp4CacheGpu[name].residentReady:
      let lut = fp4_gemv_gpu.buildFp4KmLutQ(lin.weightQ)
      let gsz = if lin.weightQ.groupSize > 0: lin.weightQ.groupSize else: 32
      # Kernel gemv fp4-KM = bytecode micro-VM -> JIT compile 1 lan / shape (NGOAI session).
      # jitName PHAI la ten pipeline that; rong => sessionPipeline("") that bai luc encode.
      # The paired-row specialization assumes both an even output-row count
      # and the kernel's 32-wide input-group layout. Keep odd/non-multiple
      # shapes on the general implementation instead of failing at JIT time.
      let pairRows = gsz == 32 and K mod 32 == 0 and N mod 2 == 0
      # BYBY_GEMV_VARIANT: 0 = 2 hang, 1 = nap rong 32-bit, 2 = 4 hang/threadgroup (mac dinh).
      # Bien the khong ap dung duoc cho shape nay -> tu ve kernel 2 hang. Do bang bench_gemv_variants.nim.
      let variant = (try: parseInt(getEnv("BYBY_GEMV_VARIANT", "2")) except ValueError: 2)
      let useWide = variant == 1 and pairRows and fp4KmGemvWideSupported(K, N)
      let use4Row = variant == 2 and gsz == 32 and K mod 32 == 0 and N mod 4 == 0
      let gemvRows = if use4Row: 4 elif pairRows: 2 else: 1
      let prog = if use4Row: buildFp4KmGemv4RowProgram(K, N)
                 elif useWide: buildFp4KmGemvWideProgram(K, N)
                 elif pairRows: buildFp4KmGemv2RowProgram(K, N)
                 else: buildFp4GemvProgram(K, N, true, gsz)
      let tag = (if use4Row: "fp4km4_gemv_" elif useWide: "fp4kmw_gemv_"
                 elif pairRows: "fp4km2_gemv_" else: "fp4km_gemv_") & $K & "x" & $N & "_g" & $gsz
      # vmKernelForBe: Vulkan -> kernelNameFor+GLSL cache; Metal -> MSL dispatch
      let kn = vmKernelForBe(prog, tag, be)
      uploadFp4WeightOnce(ctx, name, lin.weightQ.data, lut, N, K, kn,
                          rowsPerTg = gemvRows)
      if be == byby.gbVulkan and name in gFp4CacheGpu:
        var c = gFp4CacheGpu[name]
        c.glslSrc = jit_vm.glslSourceFor(gJitV2, prog, tag)
        gFp4CacheGpu[name] = c
      # Prefill MBLOCK=4: weight doc 1 lan / 4 token (nhu matmul_q4_batched)
      if gsz == 32 and K mod 32 == 0 and name in gFp4CacheGpu:
        let m4prog = buildFp4KmGemmM4Program(K, N)
        let m4tag = "fp4km_m4_" & $K & "x" & $N
        let m4kn = vmKernelForBe(m4prog, m4tag, be)
        var c = gFp4CacheGpu[name]
        c.m4JitName = m4kn
        if be == byby.gbVulkan:
          c.m4GlslSrc = jit_vm.glslSourceFor(gJitV2, m4prog, m4tag)
        gFp4CacheGpu[name] = c
    lin.weightQ.data = @[]   # host trung -> giai phong, GPU da giu ban Shared
    lin.gpuCacheName = name
    result = markFp4Lin(ctx, name, lin.bias.data, K, N)
  elif lin.useQuant and lin.weightQ.kind == qkInt4Asymmetric and int4AsymSupported(K, lin.weightQ.groupSize):
    uploadInt4WeightOnce(ctx, name, lin.weightQ, K, N)
    lin.weightQ.data = @[]   # host trung -> giai phong, GPU da giu ban Shared
    result.name = name; result.K = K; result.N = N
    result.knF32 = gInt4CacheGpu[name].knName
    result.wF32 = gInt4CacheGpu[name].scZpBuf
    result.biasH = byby.sessionUploadPrivate(be, uploadBiasOrRaise(name, lin.bias.data, N))
    result.hasBias = true
    result.ready = true
    result.isFp4 = false
    result.int4WBuf = gInt4CacheGpu[name].wBuf
    result.isInt4 = true
  elif lin.useQuant and lin.weightQ.kind == qkInt4Asymmetric:
    raise newException(ValueError, "prepareGpuLinForLinear " & name & ": weightQ la int4-asymmetric " &
      "nhung K=" & $K & " groupSize=" & $lin.weightQ.groupSize & " khong thoa int4AsymSupported " &
      "(can K%64==0, groupSize=0 hoac luy thua 2 >=64). KHONG fallback fp32.")
  else:
    # FP32 master: block G attn/fuseGate train full. Chi chap nhan weight.len==K*N.
    let expected = K * N
    if expected <= 0 or lin.weight.data.len != expected:
      raise newException(ValueError, "prepareGpuLinForLinear " & name &
        ": lin.useQuant=" & $lin.useQuant &
        (if lin.useQuant: " weightQ.kind=" & $lin.weightQ.kind else: "") &
        " weight.len=" & $lin.weight.data.len & " expected=" & $expected &
        " - khong quant hop le va khong co FP32 master.")
    let prog = buildF32GemvProgram(K, N, 1.0'f32, false)
    let kn = vmKernelForBe(prog, "f32_gemv_" & $K & "x" & $N, be)
    result.name = name; result.K = K; result.N = N
    result.knF32 = kn
    result.wF32 = byby.sessionUpload(be, lin.weight.data)
    result.biasH = byby.sessionUploadPrivate(be, uploadBiasOrRaise(name, lin.bias.data, N))
    result.hasBias = true
    result.ready = true
    result.isFp4 = false
    result.isInt4 = false
  discard prepareLoraGpu(ctx, name, lin.loraA, lin.loraB, lin.loraR, K, N, lin.loraScale)
  gLinCache[name] = result

proc uploadBiasOrRaise(name: string; bias: seq[float32]; N: int): seq[float32] =
  ## Truoc day bias.len != N -> am tham newSeq[float32](N) = 0.
  ## fuseGate.bias init=4; neu bi 0 -> sigmoid(0)=0.5 thay vi ~0.98 -> sidecar
  ## chiem 50% residual -> token rac. outProj/bias-less co the bias.len==0 hop le.
  if bias.len == N:
    return bias
  if bias.len == 0:
    # Linear khong co bias (hoac chua restore). fuseGate BAT BUOC co bias=4.
    if "fuseGate" in name:
      raise newException(ValueError,
        "uploadBiasOrRaise[" & name & "]: fuseGate.bias rong (len=0) nhung can N=" &
        $N & ". Bias bi mat khi restore -> gate=sigmoid(0)=0.5 -> token rac. " &
        "Kiem tra load checkpoint / byName / paramList co fuseGate.bias.")
    return newSeq[float32](N)
  raise newException(ValueError,
    "uploadBiasOrRaise[" & name & "]: bias.len=" & $bias.len & " != N=" & $N &
    " (shape lech, KHONG zero-pad im lang).")

proc markFp4Lin*(ctx: Backend; name: string; bias: seq[float32]; K, N: int): GpuLin =
  ## Weight đã nằm trong gFp4CacheGpu[name] (uploadFp4WeightOnce).
  let be = ctx.kind.toByby()
  ensureGpuKernels(be)
  result.name = name; result.K = K; result.N = N; result.isFp4 = true
  result.biasH = byby.sessionUploadPrivate(be, uploadBiasOrRaise(name, bias, N))
  result.hasBias = true
  result.ready = true

proc encodeLin*(l: GpuLin; hIn: byby.SessionHandle; inOff: int;
                hOut: byby.SessionHandle; outOff: int;
                bias = true; apflu = false)

proc encodeLinByName*(name: string; hIn: byby.SessionHandle; inOff: int;
                      hOut: byby.SessionHandle; outOff: int): bool =
  ## Dung cho R-block (mamba2_fused_step.encodeLayerV2 goi qua fp4GemvAt).
  ## Tra cuu DUY NHAT qua gLinCache (xem prepareGpuLinForLinear) - khong con
  ## doan rieng fp4-vs-int4-vs-fp32 o day nua (do la nguon bug truoc: nhanh
  ## fp32-fallback khong duoc dang ky o bang nao ca).
  if name notin gLinCache or not gLinCache[name].ready: return false
  encodeLin(gLinCache[name], hIn, inOff, hOut, outOff, bias = false)  # R: khong cong bias (giu dung hanh vi cu)
  true

proc encodeLin*(l: GpuLin; hIn: byby.SessionHandle; inOff: int;
                hOut: byby.SessionHandle; outOff: int;
                bias = true; apflu = false) =
  ## y = W x (+ LoRA) (+ bias) (+ APFLU). Thứ tự khớp Linear.forward():
  ## base -> cộng delta LoRA -> cộng bias.
  if l.isFp4:
    if not encodeFp4Gemv(gFp4CacheGpu[l.name], hIn, inOff, hOut, outOff):
      raise newException(CatchableError, "encodeLin: fp4 gemv that bai " & l.name)
  elif l.isInt4:
    encodeVmKernel(l.knF32, l.N * GemvTg, GemvTg, int4GemvTgBytes(),
                   hOut, outOff, l.int4WBuf, l.wF32, hIn, inOff)
  else:
    encodeVmKernel(l.knF32, l.N * GemvTg, GemvTg, f32GemvTgBytes(),
                   hOut, outOff, gDummy, l.wF32, hIn, inOff)
  discard encodeLoraDelta(l.name, hIn, inOff, hOut, outOff)
  if bias and l.hasBias:
    encodeBias(hOut, outOff, l.biasH, l.N, apflu)
  elif apflu:
    raise newException(CatchableError, "encodeLin: apflu can bias kernel (" & l.name & " khong co bias)")

# ── Scratch dùng chung mọi block A ──────────────────────────────────────
type
  AttnScratch* = object
    hAttn*: byby.SessionHandle    ## [hidden]
    hFF*: byby.SessionHandle      ## [ffDim]
    hidden*, ffDim*: int
    ready*: bool

var gAttnScratch* : AttnScratch

proc ensureAttnScratch*(be: byby.GpuBackend; hidden, ffDim: int) =
  if gAttnScratch.ready and gAttnScratch.hidden == hidden and gAttnScratch.ffDim >= ffDim: return
  gAttnScratch.hAttn = byby.sessionAllocScratch(be, hidden)
  if ffDim > 0:
    gAttnScratch.hFF = byby.sessionAllocScratch(be, ffDim)
  gAttnScratch.hidden = hidden
  gAttnScratch.ffDim = ffDim
  gAttnScratch.ready = true

# ── Block A (LinearPlusAttention + FFN) ─────────────────────────────────
type
  GpuAttnLayer* = object
    lay*: LpLayout
    hWork*: byby.SessionHandle          ## [proj | y | z | S] (state nằm trên GPU)
    knStep*: string
    qkv*, outp*, fc1*, fc2*: GpuLin
    gate*: GpuLin
    hGateInput*, hGateLogit*, hGateProb*, hGateComplement*, hGateOnes*: byby.SessionHandle
    ## Snapshot truoc mul in-place (chi cap khi BYBY_G_DIAGNOSTICS): de in mean mamba/attn/mix
    hDiagMamba*, hDiagAttn*: byby.SessionHandle
    gated*: bool
    ln1G*, ln1B*, ln2G*, ln2B*: byby.SessionHandle
    ln1Ones*: byby.SessionHandle        ## gamma=1 (preset "-unit": doc checkpoint train bang trainer cu)
    hidden*, ffDim*, nHeads*: int
    eps*: float32
    ready*: bool

var gAttnLayers* = initTable[int, GpuAttnLayer]()
var gZeroA: byby.SessionHandle
var gZeroALen = 0

proc initAttnLayer*(ctx: Backend; hidden, nHeads, ffDim: int;
                    qkv, outp, fc1, fc2: GpuLin;
                    ln1G, ln1B, ln2G, ln2B: seq[float32]; eps: float32;
                    gate: GpuLin = GpuLin()): GpuAttnLayer =
  let be = ctx.kind.toByby()
  ensureGpuKernels(be)
  if hidden mod nHeads != 0:
    raise newException(ValueError, "initAttnLayer: hidden khong chia het nHeads")
  if hidden mod 64 != 0:
    raise newException(ValueError,
      "initAttnLayer: hidden phai chia het 64 (offset buffer can 256 byte tren Intel)")
  result.lay = lpLayout(hidden, nHeads)
  result.hWork = byby.sessionUploadPrivate(be, newSeq[float32](result.lay.total))
  result.knStep = vmKernelForBe(buildLinearPlusStepProgram(result.lay, 1e-6'f32),
                           "lp_step_" & $nHeads & "_" & $hidden, be)
  result.qkv = qkv; result.outp = outp; result.fc1 = fc1; result.fc2 = fc2
  result.gate = gate
  result.gated = gate.ready
  if result.gated:
    result.hGateInput = byby.sessionAllocScratch(be, 2 * hidden)
    result.hGateLogit = byby.sessionAllocScratch(be, hidden)
    result.hGateProb = byby.sessionAllocScratch(be, hidden)
    result.hGateComplement = byby.sessionAllocScratch(be, hidden)
    var ones = newSeq[float32](hidden)
    for i in 0 ..< hidden: ones[i] = 1'f32
    result.hGateOnes = byby.sessionUploadPrivate(be, ones)
  result.ln1G = byby.sessionUploadPrivate(be, ln1G)
  block:
    var ones = newSeq[float32](ln1G.len)
    for i in 0 ..< ones.len: ones[i] = 1'f32
    result.ln1Ones = byby.sessionUploadPrivate(be, ones)
  result.ln1B = byby.sessionUploadPrivate(be, ln1B)
  result.ln2G = byby.sessionUploadPrivate(be, ln2G)
  result.ln2B = byby.sessionUploadPrivate(be, ln2B)
  result.hidden = hidden; result.nHeads = nHeads; result.ffDim = ffDim
  result.eps = eps
  result.ready = true
  ensureAttnScratch(be, hidden, ffDim)

proc resetAttnLayerState*(ctx: Backend; blockIdx: int): bool =
  ## Xoá z và S (bắt đầu chuỗi mới). PHẢI gọi trong session đang mở.
  if blockIdx notin gAttnLayers: return false
  let be = ctx.kind.toByby()
  let lp = addr gAttnLayers[blockIdx]
  let n = lp.lay.offS + lp.lay.H * lp.lay.D * lp.lay.D - lp.lay.offZ
  if gZeroALen < n:
    gZeroA = byby.sessionUploadPrivate(be, newSeq[float32](n))
    gZeroALen = n
  var hw = lp.hWork
  sessionCopyInto(be, gZeroA, hw, 0, lp.lay.offZ, n)

proc encodeLinearPlusBranch*(ctx: Backend; blockIdx: int;
                             hInput: byby.SessionHandle;
                             hNorm: var byby.SessionHandle;
                             hOut: byby.SessionHandle;
                             useRms = false; unitGamma = false;
                             inputPreNormalized = false) =
  ## Chỉ tính nhánh Linear++ từ hidden đầu vào; không cộng residual/FFN.
  ##
  ## Norm dau vao PHAI giong nhanh R cung khoi: luc train khoi P tinh MOT dau ra
  ## chuan hoa x1 = rms1(x) roi dua cho CA Mamba lan Linear++ (jamba_hybrid.forwardBlock).
  ##   useRms=true  : RMSNorm chi gamma (preset rms-*; la cai train hien tai dung)
  ##   useRms=false : LayerNorm gamma+beta (preset ln-*; khoi 'A' thuan luon the nay)
  ##   unitGamma    : gamma=1 (chi preset "-unit", doc checkpoint cu)
  ## Truoc day ham nay LUON LayerNorm -> sidecar P o infer nhan dau vao khac hang so
  ## so voi luc train (tru mean + beta thay vi chia rms) trong khi nhanh R theo preset.
  doAssert blockIdx in gAttnLayers, "encodeAttnLayer: chua initAttnLayer(" & $blockIdx & ")"
  let lp = addr gAttnLayers[blockIdx]
  template L: untyped = lp[]
  let be = ctx.kind.toByby()
  let lay = L.lay
  if not inputPreNormalized:
    if useRms:
      let gam = if unitGamma: L.ln1Ones else: L.ln1G
      if not sessionRmsnormEnc(be, hInput, gam, hNorm, 1, L.hidden, L.eps):
        raise newException(CatchableError, "encodeLinearPlusBranch: rmsnorm ln1 that bai")
    else:
      if not sessionLayernormEnc(be, hInput, L.ln1G, L.ln1B, hNorm, 1, L.hidden, L.eps):
        raise newException(CatchableError, "encodeLinearPlusBranch: layernorm ln1 that bai")
  encodeLin(L.qkv, hNorm, 0, L.hWork, lay.offProj)               # proj = [Q|K|V]
  encodeVmKernel(L.knStep, lay.H * lay.D, lay.D, 16,             # Linear++ step (state tai cho)
                 L.hWork, 0, gDummy, gDummy, gDummy, 0)
  encodeLin(L.outp, L.hWork, lay.offY, hOut, 0)

proc encodeAttnLayer*(ctx: Backend; blockIdx: int;
                      hRes: byby.SessionHandle;          ## [hidden] vào + ra (in-place)
                      hNorm: var byby.SessionHandle) =   ## [hidden] scratch
  let lp = addr gAttnLayers[blockIdx]
  template L: untyped = lp[]
  let be = ctx.kind.toByby()
  var hA = gAttnScratch.hAttn
  var hF = gAttnScratch.hFF
  encodeLinearPlusBranch(ctx, blockIdx, hRes, hNorm, hA)
  var hDst = hRes
  if not sessionVecOp(be, "add", hRes, hA, hDst, L.hidden):
    raise newException(CatchableError, "encodeAttnLayer: residual attn that bai")

  # ── FFN ──
  if not sessionLayernormEnc(be, hRes, L.ln2G, L.ln2B, hNorm, 1, L.hidden, L.eps):
    raise newException(CatchableError, "encodeAttnLayer: layernorm ln2 that bai")
  encodeLin(L.fc1, hNorm, 0, hF, 0, bias = true, apflu = true)   # APFLU(0.1, 1.0) co bias
  encodeLin(L.fc2, hF, 0, hA, 0)
  if not sessionVecOp(be, "add", hRes, hA, hDst, L.hidden):
    raise newException(CatchableError, "encodeAttnLayer: residual ffn that bai")

proc lpStepSelfCheck*(ctx: Backend; blockIdx: int; T = 4): string =
  ## KIEM TRA 100% GPU - KHONG tinh toan model tren CPU. Chay THANG kernel Linear++
  ## step (cung kernel/buffer/state cua infer) voi [Q|K|V] tu tao, doi chieu voi
  ## dang thuc GIAI TICH cua linear attention:
  ##   (1) K, V co dinh qua T token       => y_t = V               (trong so chuan hoa = 1)
  ##   (2) K co dinh, V xen ke a,b,a,b    => y_t = trung binh chay cua V
  ## Sai offset head (h*D), cong don S/z, barrier giua cac lane hay chuan hoa deu lam lech.
  if blockIdx notin gAttnLayers: return "chua init khoi A " & $blockIdx
  let be = ctx.kind.toByby()
  let lp = addr gAttnLayers[blockIdx]
  let lay = lp.lay
  let C = lay.H * lay.D
  var seed = 987654321'u32
  proc rnd(): float32 =
    seed = seed * 1664525'u32 + 1013904223'u32
    (float32(seed shr 8) / 16777216.0'f32 - 0.5'f32) * 3.0'f32
  var k = newSeq[float32](C)
  var va = newSeq[float32](C)
  var vb = newSeq[float32](C)
  for i in 0 ..< C:
    k[i] = rnd()
    va[i] = rnd()
    vb[i] = rnd()
  var qs = newSeq[seq[float32]](T)
  for t in 0 ..< T:
    qs[t] = newSeq[float32](C)
    for i in 0 ..< C: qs[t][i] = rnd()
  var worst = [0.0, 0.0]
  for mode in 1 .. 2:
    if not sessionBegin(be): return "sessionBegin that bai (reset)"
    discard resetAttnLayerState(ctx, blockIdx)
    if not sessionEnd(be): return "sessionEnd that bai (reset)"
    for t in 0 ..< T:
      var proj = newSeq[float32](3 * C)
      for i in 0 ..< C:
        proj[i] = qs[t][i]
        proj[C + i] = k[i]
        proj[2 * C + i] = (if mode == 1 or t mod 2 == 0: va[i] else: vb[i])
      var hP = sessionUpload(be, proj)
      var hw = lp.hWork
      if not sessionBegin(be):
        sessionFree(be, hP)
        return "sessionBegin that bai"
      discard sessionCopyInto(be, hP, hw, 0, lay.offProj, 3 * C)
      var err = ""
      try:
        encodeVmKernel(lp.knStep, lay.H * lay.D, lay.D, 16, hw, 0, gDummy, gDummy, gDummy, 0)
      except CatchableError as e:
        err = e.msg
      let okEnd = sessionEnd(be)
      sessionFree(be, hP)
      if err.len > 0: return "encode loi: " & err
      if not okEnd: return "sessionEnd that bai: " & sessionGetLastError(be)
      let all = sessionRead(be, hw, lay.offY + C)
      var mx = 0.0
      var md = 0.0
      let na = t div 2 + 1          # so token a trong 0..t
      let nb = (t + 1) div 2        # so token b trong 0..t
      for i in 0 ..< C:
        let want: float64 =
          if mode == 1: float64(va[i])
          else: (float64(na) * float64(va[i]) + float64(nb) * float64(vb[i])) / float64(t + 1)
        md = max(md, abs(float64(all[lay.offY + i]) - want))
        mx = max(mx, abs(want))
      worst[mode - 1] = max(worst[mode - 1], md / max(mx, 1e-12))
  let tol = 2e-3
  result = "kernel Linear++ step (khoi A #" & $blockIdx & ", " & $T & " token): " &
           "K,V co dinh sai=" & $worst[0] &
           " | V xen ke sai=" & $worst[1] &
           (if worst[0] < tol and worst[1] < tol: "  => KHOP dang thuc giai tich"
            else: "  => LECH: kernel Linear++ step tren GPU sai (offset head/cong don state/barrier)")
