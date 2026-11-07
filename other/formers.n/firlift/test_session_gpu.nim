## Cross-check: generated micro-kernels against the real session ops on the GPU.
##
## For every op the same inputs go through (a) the session call (Metal/Vulkan shader)
## and (b) the generated kernel body run as VM bytecode on the same GPU backend.
## The two outputs must agree within a small tolerance (reduction order differs).
## Run: BACKEND=metal|vulkan (default: detected). Skips when only the CPU exists.
import std/[math, os, strutils, tables]
import session_vm
import backend
import mamba2_ssm_gpu
import vendor/bybylang/gpubackend as byby
import ./kernel_gen
import ./kernel_asm
import vendor/bybylang/probe
import quant
import fp4_gemv_gpu
import fp4_gemv_vm
import gpu_layers
import jit_vm

var failures = 0

proc fill(n: int; seed: float; scale = 1.0): seq[float32] =
  for i in 0 ..< n: result.add float32(sin(float(i) * 0.37 + seed) * scale)

var skipping = false

proc agree(name: string; got, want: seq[float32]; tol = 2e-3'f32) =
  if skipping: return
  var worst = 0.0'f32
  var at = -1
  if got.len != want.len:
    echo "FAIL ", name, ": length ", got.len, " vs ", want.len
    inc failures
    return
  for i in 0 ..< got.len:
    let d = abs(got[i] - want[i]) / max(1.0'f32, abs(want[i]))
    if d != d or d > worst:
      worst = d
      at = i
  if at >= 0 and (worst != worst or worst > tol):
    echo "FAIL ", name, ": index ", at, " kernel ", got[at], " session ", want[at], " rel ", worst
    inc failures
  else:
    echo "PASS gpu ", name, " (max rel diff ", worst, ")"

proc runKernel(ctx: Backend; g: Gen; ins: seq[seq[float32]]; outLen: int; tgSize = 0;
               outSizes: seq[int] = @[]): seq[float32] =
  ## Flat memory = inputs then output; returns the output range.
  var sizes: seq[int]
  var mem: seq[float32]
  for d in ins:
    sizes.add d.len
    mem.add d
  let outBase = mem.len
  mem.setLen(outBase + outLen)
  let prog = assemble(g.body, g.ins, g.outs, sizes, (if outSizes.len > 0: outSizes else: @[outLen]), 1e-5'f32)
  # The reference launch is not part of the model trace: keep it out of the probe.
  let probing = firProbeOn
  firProbeOn = false
  discard runMicroVm(ctx, prog, mem, g.threads, 16, "gen", tgSize)
  firProbeOn = probing
  mem[outBase ..< outBase + outLen]

proc bitsToF(v: seq[float32]): seq[float32] =
  ## Session index/mask buffers hold int32/uint32 bits; compare them as numbers.
  result = newSeq[float32](v.len)
  for i in 0 ..< v.len:
    var b: int32
    copyMem(addr b, unsafeAddr v[i], 4)
    result[i] = float32(b)

proc bytesAsFloats(b: seq[uint8]): seq[float32] =
  result = newSeq[float32]((b.len + 3) div 4)
  if b.len > 0: copyMem(addr result[0], unsafeAddr b[0], b.len)

proc maskBytes(v: seq[float32]; n: int): seq[float32] =
  ## The Metal compare mask is one byte per element.
  result = newSeq[float32](n)
  var raw = newSeq[uint8](v.len * 4)
  if v.len > 0: copyMem(addr raw[0], unsafeAddr v[0], v.len * 4)
  for i in 0 ..< n: result[i] = float32(raw[i])

proc skipped(name: string) =
  ## A session op that the backend does not provide is a failure, never a skip.
  skipping = true
  echo "FAIL ", name, ": session op not available on this backend"
  inc failures

proc moreOps(ctx: Backend; be: byby.GpuBackend) =
  template session(body: untyped) =
    skipping = false
    doAssert sessionBegin(be)
    block:
      body
    doAssert sessionEnd(be)

  # asymmetric Q4 matmul against the resident weight path
  block:
    let M = 2
    let K = 128
    let N = 24
    let gs = 32
    let nG = K div gs
    let bpr = K div 2
    var bytes = newSeq[uint8](N*bpr)
    for i in 0 ..< bytes.len: bytes[i] = uint8((i*37 + 11) mod 256)
    let scales = fill(N*nG, 0.4, 0.05)
    var zeros = newSeq[float32](N*nG)
    for i in 0 ..< zeros.len: zeros[i] = 8.0'f32 + float32(i mod 3)
    let a = fill(M*K, 0.8)
    let w = residentUploadQ4(be, bytes, scales, zeros, K, N, gs, nG)
    var yh: SessionHandle
    session:
      let ha = sessionUpload(be, a)
      yh = sessionAllocScratch(be, M*N)
      doAssert sessionMatmulQ4(be, ha, w, yh, M)
    agree("matmul_q4", runKernel(ctx, genMatmulQ4(M, K, N, gs, nG), @[a, bytesAsFloats(bytes), scales, zeros], M*N),
          sessionRead(be, yh, M*N), 5e-3'f32)

  # argmax / compare / split slice / permute
  block:
    let outer = 3
    let dim = 64
    let inner = 2
    let x = fill(outer*dim*inner, 0.3, 3.0)
    var vh, ih: SessionHandle
    session:
      let hx = sessionUpload(be, x)
      vh = sessionAllocScratch(be, outer*inner)
      ih = sessionAllocScratch(be, outer*inner)
      doAssert sessionArgmaxIndexEnc(be, hx, vh, ih, outer, dim, inner)
    let g = genArgmax(outer, dim, inner)
    let mem = runKernel(ctx, g, @[x], 2*outer*inner, 0, @[outer*inner, outer*inner])
    agree("argmax.val", mem[0 ..< outer*inner], sessionRead(be, vh, outer*inner))
    agree("argmax.idx", mem[outer*inner ..< 2*outer*inner], bitsToF(sessionRead(be, ih, outer*inner)))
    let n = 100
    let a = fill(n, 0.1, 2.0)
    let b = fill(n, 0.1, 1.0)
    var mh: SessionHandle
    for op in 0 .. 4:
      session:
        let ha = sessionUpload(be, a)
        let hb = sessionUpload(be, b)
        mh = sessionAllocScratch(be, n)
        doAssert sessionCompareEnc(be, ha, hb, mh, op, n)
      # Metal writes one byte per element, Vulkan one uint32 per element.
      let got = if be == byby.gbVulkan: bitsToF(sessionRead(be, mh, n))
                else: maskBytes(sessionRead(be, mh, (n + 3) div 4), n)
      agree("compare.op" & $op, runKernel(ctx, genCompare(op, n), @[a, b], n), got)
    let dimTotal = 6
    let thisSize = 2
    let offset = 3
    let inn = 3
    let src = fill(2*dimTotal*inn, 0.4)
    var dh: SessionHandle
    session:
      let hs = sessionUpload(be, src)
      dh = sessionAllocScratch(be, 2*thisSize*inn)
      doAssert sessionSplitSliceEnc(be, hs, dh, dimTotal, thisSize, offset, inn, 2*thisSize*inn)
    agree("split_slice", runKernel(ctx, genSplitSlice(dimTotal, thisSize, offset, inn, 2*thisSize*inn), @[src], 2*thisSize*inn),
          sessionRead(be, dh, 2*thisSize*inn))
    let srcP = fill(24, 0.9)
    let ns = @[6'i32, 3, 1]
    let os = @[12'i32, 4, 1]
    let dm = @[2'i32, 0, 1]
    session:
      let hs = sessionUpload(be, srcP)
      let hn = sessionUploadIndices(be, ns)
      let ho = sessionUploadIndices(be, os)
      let hd = sessionUploadIndices(be, dm)
      dh = sessionAllocScratch(be, 24)
      doAssert sessionPermuteEnc(be, hs, dh, hn, ho, hd, 3, 24)
    agree("permute", runKernel(ctx, genPermute(@[6, 3, 1], @[12, 4, 1], @[2, 0, 1], 24), @[srcP], 24),
          sessionRead(be, dh, 24))

  block:
    let n = 200
    var mh: SessionHandle
    session:
      if not sessionDropoutMaskEnc(be, mh, n, 0.3'f32, 777'u32): skipped("dropout_mask")
    agree("dropout_mask", runKernel(ctx, genDropoutMask(n, 0.3, 777'u32), newSeq[seq[float32]](), n),
          sessionRead(be, mh, n))

  # APFLU (the Metal shader supports alpha 0.1 / beta 1.0)
  block:
    let n = 64
    let x = fill(n, 0.2, 2.0)
    var yh: SessionHandle
    session:
      let hx = sessionUpload(be, x)
      yh = sessionAllocScratch(be, n)
      if not sessionApfluEnc(be, hx, yh, n, 0.1'f32, 1.0'f32): skipped("apflu")
    agree("apflu", runKernel(ctx, genApflu(n, 0.1, 1.0), @[x], n), sessionRead(be, yh, n), 5e-3'f32)
    let dyv = fill(n, 0.3, 1.5)
    var dxh: SessionHandle
    session:
      let hx = sessionUpload(be, x)
      let hd = sessionUpload(be, dyv)
      dxh = sessionAllocScratch(be, n)
      if not sessionApfluBackwardEnc(be, hx, hd, dxh, n, 0.1'f32, 1.0'f32): skipped("apflu_backward")
    agree("apflu_backward", runKernel(ctx, genApfluBackward(n, 0.1, 1.0), @[x, dyv], n), sessionRead(be, dxh, n), 5e-3'f32)

  # attention: single-K/V GQA decode, fused causal, flash prefill
  block:
    let H = 8
    let S = 21
    let D = 32
    let q = fill(H*D, 0.1, 0.4)
    let kk = fill(S*D, 0.5, 0.4)
    let vv = fill(S*D, 0.9, 0.4)
    var oh, sh: SessionHandle
    session:
      let hq = sessionUpload(be, q)
      let hk = sessionUpload(be, kk)
      let hv = sessionUpload(be, vv)
      oh = sessionAllocScratch(be, H*D)
      sh = sessionAllocScratch(be, H*S)
      if not sessionAttentionDecodeGqaEnc(be, hq, hk, hv, oh, sh, H, S, D, 0.25'f32): skipped("attention_decode_gqa")
    let mem = runKernel(ctx, genAttentionDecodeGqa(H, S, D, 0.25), @[q, kk, vv], H*D + H*S, 0, @[H*D, H*S])
    agree("attention_decode_gqa.o", mem[0 ..< H*D], sessionRead(be, oh, H*D), 5e-3'f32)
    let B = 1
    let Hh = 2
    let Sf = 12
    let Df = 16
    let n = B*Hh*Sf*Df
    let q2 = fill(n, 0.2, 0.5)
    let k2 = fill(n, 0.6, 0.5)
    let v2 = fill(n, 1.0, 0.5)
    var o2, s2: SessionHandle
    session:
      let hq = sessionUpload(be, q2)
      let hk = sessionUpload(be, k2)
      let hv = sessionUpload(be, v2)
      o2 = sessionAllocScratch(be, n)
      s2 = sessionAllocScratch(be, B*Hh*Sf*Sf)
      if not sessionAttentionFusedEnc(be, hq, hk, hv, o2, s2, B, Hh, Sf, Df, 0.25'f32): skipped("attention_fused")
    let mf = runKernel(ctx, genAttentionFused(B, Hh, Sf, Df, 0.25), @[q2, k2, v2], n + B*Hh*Sf*Sf, 0, @[n, B*Hh*Sf*Sf])
    agree("attention_fused.o", mf[0 ..< n], sessionRead(be, o2, n), 5e-3'f32)
    agree("attention_fused.s", mf[n ..< n + B*Hh*Sf*Sf], sessionRead(be, s2, B*Hh*Sf*Sf), 5e-3'f32)
    let T = 5
    let H3 = 4
    let HKV = 2
    let D3 = 16
    let maxSeq = 16
    let pos = 3
    let q3 = fill(T*H3*D3, 0.3, 0.5)
    let kc = fill(HKV*maxSeq*D3, 0.7, 0.5)
    let vc = fill(HKV*maxSeq*D3, 1.1, 0.5)
    var o3: SessionHandle
    discard sessionFlashAttentionPrepare(be, D3)
    session:
      let hq = sessionUpload(be, q3)
      let hk = sessionUpload(be, kc)
      let hv = sessionUpload(be, vc)
      o3 = sessionAllocScratch(be, T*H3*D3)
      if not sessionFlashAttentionPrefillEnc(be, hq, hk, hv, o3, T, H3, HKV, D3, pos, maxSeq, 0.25'f32): skipped("flash_prefill")
    agree("flash_prefill", runKernel(ctx, genFlashPrefill(T, H3, HKV, D3, maxSeq, 0.25), @[q3, kc, vc, @[float32(pos)]], T*H3*D3),
          sessionRead(be, o3, T*H3*D3), 5e-3'f32)

  # attention backward (accumulates into zero-initialised gradients) and KV scatter
  block:
    let B = 1
    let H = 2
    let S = 8
    let D = 8
    let n = B*H*S*D
    let q = fill(n, 0.2, 0.5)
    let kk = fill(n, 0.6, 0.5)
    let vv = fill(n, 1.0, 0.5)
    let dy = fill(n, 1.4, 0.5)
    var oh, sh, dqh, dkh, dvh: SessionHandle
    session:
      let hq = sessionUpload(be, q)
      let hk = sessionUpload(be, kk)
      let hv = sessionUpload(be, vv)
      let hdy = sessionUpload(be, dy)
      oh = sessionAllocScratch(be, n)
      sh = sessionAllocScratch(be, B*H*S*S)
      doAssert sessionAttentionFusedEnc(be, hq, hk, hv, oh, sh, B, H, S, D, 0.3'f32)
      dqh = sessionUpload(be, newSeq[float32](n))
      dkh = sessionUpload(be, newSeq[float32](n))
      dvh = sessionUpload(be, newSeq[float32](n))
      if not sessionAttentionFusedBackwardEnc(be, hq, hk, hv, sh, hdy, dqh, dkh, dvh, B, H, S, D, 0.3'f32): skipped("attention_fused_backward")
    let probs = sessionRead(be, sh, B*H*S*S)
    let mb = runKernel(ctx, genAttentionFusedBackward(B, H, S, D, 0.3), @[q, kk, vv, probs, dy], 3*n, 0, @[n, n, n])
    agree("attention_fused_backward.dq", mb[0 ..< n], sessionRead(be, dqh, n), 5e-3'f32)
    agree("attention_fused_backward.dk", mb[n ..< 2*n], sessionRead(be, dkh, n), 5e-3'f32)
    agree("attention_fused_backward.dv", mb[2*n ..< 3*n], sessionRead(be, dvh, n), 5e-3'f32)
    let T = 3
    let HKV = 2
    let Dk = 4
    let maxSeq = 6
    let pos = 2
    let src = fill(T*HKV*Dk, 0.9)
    let cache0 = fill(HKV*maxSeq*Dk, 0.1)
    var ch: SessionHandle
    session:
      let hs = sessionUpload(be, src)
      ch = sessionUpload(be, cache0)
      doAssert sessionKvScatterEnc(be, hs, ch, T, HKV, Dk, pos, maxSeq)
    var memK = src
    memK.add float32(pos)
    memK.add cache0
    let gk = genKvScatter(T, HKV, Dk, maxSeq)
    let progK = assemble(gk.body, gk.ins, gk.outs, @[src.len, 1, cache0.len], @[], 1e-5'f32)
    let probing = firProbeOn
    firProbeOn = false
    discard runMicroVm(ctx, progK, memK, gk.threads, 16, "gen", 0)
    firProbeOn = probing
    agree("kv_scatter", memK[src.len + 1 ..< memK.len], sessionRead(be, ch, cache0.len))

  # layernorm backward
  block:
    let rows = 4
    let cols = 32
    let x = fill(rows*cols, 0.1, 2.0)
    let dy = fill(rows*cols, 0.6)
    let gam = fill(cols, 1.4)
    let bet = fill(cols, 0.3)
    var dxh, dgh, dbh: SessionHandle
    session:
      let hdy = sessionUpload(be, dy)
      let hx = sessionUpload(be, x)
      let hg = sessionUpload(be, gam)
      let hb = sessionUpload(be, bet)
      dxh = sessionAllocScratch(be, rows*cols)
      dgh = sessionAllocScratch(be, cols)
      dbh = sessionAllocScratch(be, cols)
      if not sessionLayernormBackwardEnc(be, hdy, hx, hg, hb, dxh, dgh, dbh, rows, cols, 1e-5'f32): skipped("layernorm_backward")
    agree("layernorm_backward.dx", runKernel(ctx, genLayernormBackwardDx(rows, cols, 1e-5), @[dy, x, gam], rows*cols),
          sessionRead(be, dxh, rows*cols), 5e-3'f32)
    let mp = runKernel(ctx, genLayernormBackwardParams(rows, cols, 1e-5), @[dy, x], 2*cols, 0, @[cols, cols])
    agree("layernorm_backward.dgamma", mp[0 ..< cols], sessionRead(be, dgh, cols), 5e-3'f32)
    agree("layernorm_backward.dbeta", mp[cols ..< 2*cols], sessionRead(be, dbh, cols), 5e-3'f32)

proc fp4Ops(ctx: Backend; be: byby.GpuBackend) =
  ## FP4-KM GEMM as the Vulkan session ops: the packed-weight gemv micro-kernel that the runtime really
  ## runs on this device, and the cooperative-matrix kernels where the device has them.
  if be != byby.gbVulkan: return
  template session(body: untyped) =
    skipping = false
    doAssert sessionBegin(be)
    block:
      body
    doAssert sessionEnd(be)
  for (M, K, N) in [(1, 64, 8), (3, 128, 16)]:
    let dense = fill(N * K, 0.2)
    let qt = quantizeFp4KM(dense, @[N, K])
    let lut = buildFp4KmLutQ(qt)
    let x = fill(M * K, 0.5)
    let lay = fp4kmLayout(N, K)
    var blob = newSeq[float32](lay.totalWords)
    copyMem(addr blob[0], unsafeAddr qt.bytes[0], qt.bytes.len)
    for i in 0 ..< 272: blob[lay.lutWord + i] = lut[i]
    let want = runKernel(ctx, genFp4Gemm(M, K, N), @[x, blob], M * N)
    let shape = "." & $M & "x" & $K & "x" & $N
    # The runtime's packed FP4 gemv micro-kernel (4 rows per threadgroup).
    block:
      let prog = buildFp4KmGemv4RowProgram(K, N)
      let tag = "fp4km4_gemv_" & $K & "x" & $N & "_g32"
      let kn = vmKernelForBe(prog, tag, be)
      let cname = "firlift.fp4" & shape
      uploadFp4WeightOnce(ctx, cname, qt.bytes, lut, N, K, kn, rowsPerTg = 4)
      var c = gFp4CacheGpu[cname]
      c.glslSrc = jit_vm.glslSourceFor(gJitV2, prog, tag)
      let hx = sessionUpload(be, x)
      var yh = sessionAllocScratch(be, M * N)
      doAssert sessionBegin(be)
      for row in 0 ..< M:
        doAssert encodeFp4Gemv(c, hx, row * K, yh, row * N)
      doAssert sessionEnd(be)
      agree("fp4_gemv_micro" & shape, want, sessionRead(be, yh, M * N), 2e-2'f32)
    # Cooperative-matrix GEMM session ops; only devices with VK_KHR_cooperative_matrix run them.
    if sessionCooperativeGemmEnabled(be):
      for which in 0 .. 1:
        var ch: SessionHandle
        var ok = false
        session:
          let hx = sessionUpload(be, x)
          let hw = sessionUploadBytesShared(be, qt.bytes)
          let hl = sessionUploadPrivate(be, lut)
          ch = sessionAllocScratch(be, M * N)
          ok = if which == 0: sessionFp4GemmTensorCore(be, hx, hw, hl, ch, M, K, N)
               else: sessionFp4GemmTiledEnc(be, hx, hw, hl, ch, M, K, N)
        let name = (if which == 0: "fp4_gemm_tensorcore" else: "fp4_gemm_tiled") & shape
        if not ok:
          echo "FAIL ", name, ": session op did not run although cooperative GEMM is enabled"
          inc failures
        else:
          agree(name, want, sessionRead(be, ch, M * N), 2e-2'f32)
    else:
      echo "NOTE fp4 cooperative GEMM not enabled on this device (no cooperative matrix); micro-kernel path tested"

proc main() =
  let kind = case getEnv("BACKEND", "auto")
    of "metal": bkMetal
    of "vulkan": bkVulkan
    else:
      case byby.detectBackend()
      of byby.gbMetal: bkMetal
      of byby.gbVulkan: bkVulkan
      else: bkCpu
  if kind == bkCpu:
    echo "SKIP: no GPU backend"
    return
  let ctx = Backend(kind: kind)
  probeStepBegin("decode", 1, 1, 0, 1)
  let be = ctx.kind.toByby()
  echo "backend ", $kind

  template session(body: untyped) =
    skipping = false
    doAssert sessionBegin(be)
    block:
      body
    doAssert sessionEnd(be)

  # rmsnorm / layernorm / softmax
  block:
    let rows = 4
    let cols = 64
    let x = fill(rows*cols, 0.1, 2.0)
    let gam = fill(cols, 0.3)
    let bet = fill(cols, 0.9, 0.5)
    var yh: SessionHandle
    var got, want: seq[float32]
    session:
      let hx = sessionUpload(be, x)
      let hg = sessionUpload(be, gam)
      yh = sessionAllocScratch(be, rows*cols)
      doAssert sessionRmsnormEnc(be, hx, hg, yh, rows, cols, 1e-5'f32)
    want = sessionRead(be, yh, rows*cols)
    got = runKernel(ctx, genRmsnorm(rows, cols, 1e-5), @[x, gam], rows*cols)
    agree("rmsnorm", got, want)
    session:
      let hx = sessionUpload(be, x)
      let hg = sessionUpload(be, gam)
      let hb = sessionUpload(be, bet)
      yh = sessionAllocScratch(be, rows*cols)
      doAssert sessionLayernormEnc(be, hx, hg, hb, yh, rows, cols, 1e-5'f32)
    want = sessionRead(be, yh, rows*cols)
    got = runKernel(ctx, genLayernorm(rows, cols, 1e-5), @[x, gam, bet], rows*cols)
    agree("layernorm", got, want)
    session:
      let hx = sessionUpload(be, x)
      yh = sessionAllocScratch(be, rows*cols)
      doAssert sessionSoftmax(be, hx, yh, rows, cols)
    want = sessionRead(be, yh, rows*cols)
    got = runKernel(ctx, genSoftmax(rows, cols), @[x], rows*cols)
    agree("softmax", got, want)

  # elementwise
  block:
    let n = 257
    let a = fill(n, 0.2, 1.5)
    let b = fill(n, 1.1, 2.0)
    var yh: SessionHandle
    for op in ["add", "mul", "sub"]:
      session:
        let ha = sessionUpload(be, a)
        let hb = sessionUpload(be, b)
        yh = sessionAllocScratch(be, n)
        doAssert sessionVecOp(be, op, ha, hb, yh, n)
      agree("vecop." & op, runKernel(ctx, genVecOp(op, n), @[a, b], n), sessionRead(be, yh, n))
    for op in ["relu", "sigmoid", "tanh"]:
      session:
        let ha = sessionUpload(be, a)
        yh = sessionAllocScratch(be, n)
        doAssert sessionActivation(be, op, ha, yh, n)
      agree("activation." & op, runKernel(ctx, genActivation(op, n), @[a], n), sessionRead(be, yh, n))

  # matmul NT
  block:
    let M = 3
    let K = 96
    let N = 40
    let a = fill(M*K, 0.4)
    let b = fill(N*K, 1.7)
    var yh: SessionHandle
    session:
      let ha = sessionUpload(be, a)
      let hb = sessionUpload(be, b)
      yh = sessionAllocScratch(be, M*N)
      doAssert sessionMatmulNT(be, ha, hb, yh, M, K, N)
    agree("matmul_nt", runKernel(ctx, genMatmulNT(M, K, N), @[a, b], M*N), sessionRead(be, yh, M*N))
    let bn = fill(K*N, 2.1)
    session:
      let ha = sessionUpload(be, a)
      let hb = sessionUpload(be, bn)
      yh = sessionAllocScratch(be, M*N)
      doAssert sessionMatmul(be, ha, hb, yh, M, K, N)
    agree("matmul", runKernel(ctx, genMatmul(M, K, N), @[a, bn], M*N), sessionRead(be, yh, M*N))

  # RoPE (in place on the session side)
  block:
    let T = 2
    let Hn = 4
    let D = 16
    let x = fill(T*Hn*D, 0.5)
    let cosT = fill(T*D div 2, 0.3)
    let sinT = fill(T*D div 2, 1.9)
    var hx: SessionHandle
    session:
      hx = sessionUpload(be, x)
      let hc = sessionUpload(be, cosT)
      let hs = sessionUpload(be, sinT)
      doAssert sessionRope(be, hx, hc, hs, T, Hn, D)
    agree("rope", runKernel(ctx, genRope(T, Hn, D), @[x, cosT, sinT], T*Hn*D),
          sessionRead(be, hx, T*Hn*D))

  # attention decode over a GQA cache
  block:
    let H = 8
    let HKV = 2
    let D = 64
    let maxSeq = 32
    let S = 19
    let q = fill(H*D, 0.2, 0.3)
    let kc = fill(HKV*maxSeq*D, 0.7, 0.5)
    let vc = fill(HKV*maxSeq*D, 1.3, 0.5)
    var oh: SessionHandle
    session:
      let hq = sessionUpload(be, q)
      let hk = sessionUpload(be, kc)
      let hv = sessionUpload(be, vc)
      oh = sessionAllocScratch(be, H*D)
      doAssert sessionAttentionDecodeAllHeadsEnc(be, hq, hk, hv, oh, H, HKV, D, S, maxSeq)
    agree("attention_decode",
          runKernel(ctx, genAttentionDecode(H, HKV, D, maxSeq), @[q, kc, vc, @[float32(S)]], H*D),
          sessionRead(be, oh, H*D), 5e-3'f32)

  # gather and embedding lookup
  block:
    let src = fill(300, 0.8)
    let idx = @[17'i32, 3, 299, 120, 3]
    var yh: SessionHandle
    session:
      let hs = sessionUpload(be, src)
      let hi = sessionUploadIndices(be, idx)
      yh = sessionAllocScratch(be, idx.len)
      doAssert sessionGatherEnc(be, hs, hi, yh, idx.len)
    var fidx: seq[float32]
    for v in idx: fidx.add float32(v)
    agree("gather", runKernel(ctx, genGather(idx.len), @[src, fidx], idx.len), sessionRead(be, yh, idx.len))
    let table = fill(50*32, 0.4)
    let ids = @[7'i32, 0, 49]
    session:
      let ht = sessionUpload(be, table)
      let hi = sessionUploadIndices(be, ids)
      yh = sessionAllocScratch(be, ids.len*32)
      doAssert sessionEmbeddingLookupEnc(be, ht, hi, yh, 50, 32, ids.len)
    var fids: seq[float32]
    for v in ids: fids.add float32(v)
    agree("embedding", runKernel(ctx, genEmbedding(32, ids.len), @[table, fids], ids.len*32),
          sessionRead(be, yh, ids.len*32))

  moreOps(ctx, be)
  fp4Ops(ctx, be)
  probeStepEnd()

  if failures > 0:
    echo failures, " session kernel(s) disagree"
    quit(1)
  echo "all generated kernels match the session ops"

main()
