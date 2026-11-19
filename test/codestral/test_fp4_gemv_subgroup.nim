## The subgroup-row FP4-KM decode GEMV (buildFp4KmGemvSubgroupProgram) must reproduce the tuned
## barrier-tree kernel for every geometry, with and without the fused residual add.
##
##   nim c -d:release --mm:arc --path:. --path:test/codestral --path:test/deepseek-inference \
##     --path:test/spectron --path:kernelfortest test/codestral/test_fp4_gemv_subgroup.nim
##   BYBY_BENCH_BACKEND=vulkan test/codestral/test_fp4_gemv_subgroup
## Needs 32-lane subgroups (NVIDIA, Apple GPUs); the runtime only uses a geometry after the same
## comparison passes on the device. BYBY_SG_TEST_STRICT=1 fails on any rejected geometry (use it on
## GPUs with fixed 32-lane subgroups, such as NVIDIA).
import std/[math, strformat, os, tables]
import backend, nimformer, quant, gpu_layers, fp4_gemv_gpu, fp4_gemv_vm
import vendor/bybylang/gpubackend as byby
import vendor/bybylang/micro_kernel/session_vm

proc run(ctx: Backend; k, n: int) =
  let be = ctx.kind.toByby()
  let name = "sg_gemv_test_" & $k & "_" & $n
  var qt = QuantTensor(kind: qkFp4KM, shape: @[n, k], groupSize: 32, scale: @[0.01'f32])
  qt.data = newSeq[uint8](n * (k div 2) + n * (k div 32))
  var rng = 7'u32
  for i in 0 ..< qt.data.len:
    rng = rng * 1664525'u32 + 1013904223'u32
    qt.data[i] = if i < n * (k div 2): uint8(rng shr 16) else: uint8(32 + (rng mod 48))
  var lin = Linear(inF: k, outF: n, useQuant: true, weightQ: qt)
  discard prepareGpuLinForLinear(ctx, name, lin)
  var x = newSeq[float32](k)
  for i in 0 ..< x.len: x[i] = float32(sin(float(i) * 0.13))
  var residual = newSeq[float32](n)
  for i in 0 ..< n: residual[i] = float32(cos(float(i) * 0.7))
  var hx = byby.sessionUpload(be, x)
  var hy = byby.sessionAllocScratch(be, n)
  var hr = byby.sessionUpload(be, residual)
  defer:
    byby.sessionFree(be, hx); byby.sessionFree(be, hy); byby.sessionFree(be, hr)
  let c = gFp4CacheGpu[name]
  proc runKernel(p: VmProgram; tag: string; threads, tg, bytes: int; output: SessionHandle): seq[float32] =
    let kn = vmKernelForBe(p, tag, be)
    doAssert byby.sessionBegin(be)
    encodeVmKernel(kn, threads, tg, bytes, output, 0, c.wBuf, c.lutBuf, hx, 0)
    doAssert byby.sessionEnd(be), byby.sessionGetLastError(be)
    byby.sessionRead(be, output, n)
  let want = runKernel(buildFp4KmGemvTunedProgram(k, n, 32, 4), &"sgref_{k}_{n}",
    (n div 4) * 32, 32, (Fp4KmLutLen + 4 * 32) * 4, hy)
  var mag = 1e-6'f32
  for v in want: mag = max(mag, abs(v))
  var checked, rejected = 0
  for tg in [32, 64, 128, 256]:
    for rows in [1, 2, 4, 8]:
      for unroll in [1, 2, 4]:
        if not fp4KmGemvSubgroupSupported(k, n, tg, rows, unroll): continue
        for lutConst in [false, true]:
          let tag = &"sgtest_{k}_{n}_{tg}_{rows}_{unroll}_{lutConst}"
          let threads = (n div fp4KmGemvSubgroupRowsPerTg(tg, rows)) * tg
          let bytes = fp4KmGemvSubgroupTgBytes(lutConst)
          let got = runKernel(buildFp4KmGemvSubgroupProgram(k, n, tg, rows, unroll, lutConst),
            tag, threads, tg, bytes, hy)
          var err = 0'f32
          for i, v in got:
            doAssert classify(v) notin {fcNan, fcInf, fcNegInf}
            err = max(err, abs(v - want[i]))
          if err / mag >= 1e-4'f32:
            # A GPU that picks its SIMD width per pipeline (Intel) may run this one narrower than
            # 32 lanes; the runtime tuner rejects such a pipeline by the same comparison.
            echo &"REJECT K={k} N={n} tg={tg} rows={rows} unroll={unroll} lutConst={lutConst}: " &
              &"rel={err / mag:.3g} (subgroups narrower than 32 lanes?)"
            inc rejected
            continue
          # Fused residual: out = residual + W x.
          doAssert byby.sessionBegin(be)
          doAssert byby.sessionCopyInto(be, hr, hy, 0, 0, n)
          doAssert byby.sessionEnd(be)
          let acc = runKernel(buildFp4KmGemvSubgroupProgram(k, n, tg, rows, unroll, lutConst,
            accumulate = true), tag & "_acc", threads, tg, bytes, hy)
          var accErr = 0'f32
          for i, v in acc: accErr = max(accErr, abs(v - (want[i] + residual[i])))
          doAssert accErr / (mag + 1) < 1e-4'f32, &"accumulate K={k} N={n} tg={tg} rows={rows}: {accErr}"
          inc checked
  doAssert checked > 0, "no subgroup GEMV geometry matched on this device"
  if getEnv("BYBY_SG_TEST_STRICT", "0") == "1": doAssert rejected == 0
  echo &"PASS fp4 subgroup gemv K={k} N={n}: {checked} geometries match the tuned kernel " &
    &"(plain and fused residual), {rejected} rejected"

when isMainModule:
  let ctx = newBackend(getEnv("BYBY_BENCH_BACKEND", "metal"))
  run(ctx, 4096, 256)
  run(ctx, 8192, 512)
  run(ctx, 2048, 4096)
