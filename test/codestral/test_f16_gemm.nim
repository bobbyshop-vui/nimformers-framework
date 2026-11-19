## FP16-weight GEMV/GEMM (Linear++ sidecar, BYBY_LINEARPP_DTYPE=fp16) against an FP32 CPU reference.
##
## Run: nim c -r -d:release --hints:off test/codestral/test_f16_gemm.nim
## (BYBY_BENCH_BACKEND=metal|vulkan, default metal).
import std/[math, strformat, os]
import backend, gpu_layers, fp4_gemv_vm
import vendor/bybylang/gpubackend as byby

proc checkHalfConversion() =
  ## Every finite half survives float -> half -> float, and ties round to even.
  for h in 0 .. 0xFFFF:
    let bits = uint16(h)
    let f = f16BitsToF32(bits)
    if classify(f) == fcNan: continue
    doAssert f32ToF16Bits(f) == bits, &"half 0x{h:04x} does not round-trip"
  doAssert f32ToF16Bits(1.0'f32 + 1.0'f32 / 2048) == 0x3C00'u16          # tie -> even (1.0)
  doAssert f32ToF16Bits(1.0'f32 + 3.0'f32 / 2048) == 0x3C02'u16          # tie -> even (1 + 2^-9)
  doAssert f32ToF16Bits(65520'f32) == 0x7C00'u16                          # rounds to +inf
  doAssert f32ToF16Bits(-1e-9'f32) == 0x8000'u16                          # underflow keeps the sign
  doAssert f16BitsToF32(0x0001'u16) == pow(2'f32, -24'f32)                # smallest subnormal
  echo "PASS half conversion (65536 values, ties to even, overflow, underflow)"

proc run(be: byby.GpuBackend; K, N, M: int; inOff = 0; vecInput = true) =
  var w = newSeq[float32](N * K)
  var x = newSeq[float32](inOff + M * K)
  var rng = 12345'u32
  proc next(): float32 =
    rng = rng * 1664525'u32 + 1013904223'u32
    float32(int(rng shr 8) mod 20001 - 10000) / 10000'f32
  for v in w.mitems: v = next() * 0.05'f32
  for v in x.mitems: v = next()
  let halves = toF16Bits(w)
  # References: the FP32 weights (what fp32 mode computes) and the rounded halves (exact target).
  var want32 = newSeq[float64](M * N)
  var want16 = newSeq[float64](M * N)
  for r in 0 ..< M:
    for n in 0 ..< N:
      var a32, a16 = 0.0
      for k in 0 ..< K:
        let xv = float64(x[inOff + r * K + k])
        a32 += float64(w[n * K + k]) * xv
        a16 += float64(f16BitsToF32(halves[n * K + k])) * xv
      want32[r * N + n] = a32
      want16[r * N + n] = a16
  let hw = byby.sessionUploadPrivatePtr(be, unsafeAddr halves[0], halves.len * sizeof(uint16))
  var hx = byby.sessionUpload(be, x)
  var hy = byby.sessionAllocScratch(be, M * N)
  defer:
    var w2 = hw
    byby.sessionFree(be, w2); byby.sessionFree(be, hx); byby.sessionFree(be, hy)
  let kn = vmKernelForBe(buildF16GemmProgram(K, N, M, vecInput), &"t_f16gemm_{K}x{N}_m{M}_v{vecInput}", be)
  doAssert byby.sessionBegin(be)
  encodeVmKernel(kn, N * GemvTg, GemvTg, f16GemmTgBytes(M), hy, 0, hw, gDummyFor(be), hx, inOff)
  doAssert byby.sessionEnd(be), byby.sessionGetLastError(be)
  let got = byby.sessionRead(be, hy, M * N)
  var err16, err32, mag = 0.0
  for i, v in got:
    doAssert classify(v) notin {fcNan, fcInf, fcNegInf}, &"non-finite output at {i}"
    err16 = max(err16, abs(float64(v) - want16[i]))
    err32 = max(err32, abs(float64(v) - want32[i]))
    mag = max(mag, abs(want32[i]))
  let rel16 = err16 / max(mag, 1e-12)
  let rel32 = err32 / max(mag, 1e-12)
  # FP32 accumulation of the exact half weights: only summation-order rounding remains.
  doAssert rel16 < 2e-5, &"K={K} N={N} M={M}: rel error vs half reference {rel16:.3g}"
  # Against the FP32 weights the half rounding (2^-11 relative per weight) is the error budget.
  doAssert rel32 < 2e-3, &"K={K} N={N} M={M}: rel error vs fp32 reference {rel32:.3g}"
  echo &"PASS f16 gemm K={K} N={N} M={M} inOff={inOff} vec={vecInput} " &
    &"rel(vs half ref)={rel16:.2e} rel(vs fp32 ref)={rel32:.2e}"

when isMainModule:
  checkHalfConversion()
  let ctx = newBackend(getEnv("BYBY_BENCH_BACKEND", "metal"))
  let be = ctx.kind.toByby()
  ensureGpuKernels(be)
  for M in [1, 2, 5, 8, 12, 16, 17, 40]:
    run(be, 512, 96, M)
  run(be, 4096, 256, 1)                   # decode shape (qkvProj/outProj K)
  run(be, 8192, 128, 3)                   # fuseGate K
  run(be, 384, 64, 1, inOff = 64)         # aligned input offset
  if be == byby.gbMetal:
    # Unaligned input offset (scalar loads); Vulkan descriptor offsets must be aligned anyway.
    run(be, 384, 64, 2, inOff = 3, vecInput = false)
  echo "ALL PASS f16 gemm on ", be
