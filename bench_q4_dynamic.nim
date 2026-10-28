## Native Q4 session path used by nim_inference. Includes 8096/8192 square
## matrices, non-full workgroups, partial output tiles and partial input batches.
import std/[math, os, monotimes, times, strformat, algorithm]
import backend
import vendor/bybylang/gpubackend as byby
import vendor/bybylang/backends/vulkan/vulkan_driver as vkDrv

proc run(ctx: Backend; K, N, group: int; forceWide = false) =
  let be = ctx.kind.toByby()
  let ng = if group > 0: (K + group - 1) div group else: 1
  var packed = newSeq[uint8](N * (K div 2))
  var scales, zeros = newSeq[float32](N * ng)
  var seed = 91'u32
  for i in 0 ..< packed.len:
    seed = seed * 1664525'u32 + 1013904223'u32
    packed[i] = uint8(seed shr 16)
  for i in 0 ..< scales.len:
    scales[i] = 0.01'f32 + float32(i mod 13) * 0.001'f32
    zeros[i] = 6.5'f32 + float32(i mod 3) * 0.5'f32
  var w = residentUploadQ4(be, packed, scales, zeros, K, N, group, ng)
  defer: residentFreeQ4(w)
  putEnv("BYBY_Q4_TUNED", if forceWide: "wide" else: "1")
  let preparedDecode = residentPrepareQ4(w, 1)
  let preparedPrefill = residentPrepareQ4(w, 8)
  doAssert preparedDecode and preparedPrefill, "Q4 inference autotuner was not prepared"
  for M in [1, 3, 15]:
    var data = newSeq[float32](M * K)
    for i in 0 ..< data.len: data[i] = float32(sin(float(i) * 0.071))
    var input = sessionUpload(be, data)
    var output = sessionAllocScratch(be, M * N)
    var reference: seq[float32]
    var legacyMs = 0.0
    let hasPlan = if M == 1: preparedDecode else: preparedPrefill
    let usesMma = ctx.kind == bkVulkan and M >= 8 and vkDrv.vulkanTensorCoreAvailable()
    let tolerance = if usesMma: 3e-3'f32 else: 1e-4'f32
    for tuned in (if hasPlan: @[false, true] else: @[false]):
      putEnv("BYBY_Q4_TUNED", if tuned: "1" else: "0")
      var times: seq[float]
      for rep in 0 ..< 4:
        let start = getMonoTime()
        doAssert sessionBegin(be)
        doAssert sessionMatmulQ4(be, input, w, output, M)
        doAssert sessionEnd(be)
        if rep > 0: times.add float((getMonoTime() - start).inNanoseconds) / 1e6
      let values = sessionRead(be, output, M * N)
      for v in values: doAssert classify(v) notin {fcNan, fcInf, fcNegInf}
      var err, mag: float32
      if not tuned: reference = values
      else:
        for i, v in values:
          doAssert classify(v) notin {fcNan, fcInf, fcNegInf}
          err = max(err, abs(v - reference[i])); mag = max(mag, abs(reference[i]))
        doAssert err <= max(1e-5'f32, mag * tolerance)
      # Independent FP64 reference checks the quant layout and group tail.
      for m in [0, M - 1]:
        for n in [0, N div 2, N - 1]:
          var expected: float64
          for k in 0 ..< K:
            let byteVal = packed[n * (K div 2) + k div 2]
            let q = (int(byteVal) shr ((k mod 2) * 4)) and 15
            let g = n * ng + (if group > 0: k div group else: 0)
            expected += float64(data[m * K + k]) *
              ((float64(q) - float64(zeros[g])) * float64(scales[g]))
          let cpuTolerance = if usesMma: 0.003 else: 0.0002
          if abs(float64(values[m * N + n]) - expected) >= cpuTolerance * max(1.0, abs(expected)):
            echo "CPU mismatch M=", M, " m=", m, " n=", n, " tuned=", tuned,
              " GPU=", values[m * N + n], " CPU=", expected,
              " scale0=", scales[0], " zero0=", zeros[0]
          doAssert abs(float64(values[m * N + n]) - expected) < cpuTolerance * max(1.0, abs(expected))
      times.sort()
      if not tuned: legacyMs = times[1]
      if not hasPlan:
        echo &"Q4 backend={ctx.kind} K={K} N={N} M={M} group={group} native={times[1]:.3f}ms FP64=PASS"
      elif tuned:
        let relative = err / max(mag, 1e-9'f32)
        echo &"Q4 K={K} N={N} M={M} group={group} legacy={legacyMs:.3f}ms tuned={times[1]:.3f}ms relative={relative:.3g}"
    sessionFree(be, input); sessionFree(be, output)

when isMainModule:
  let ctx = newBackend(getEnv("BYBY_BENCH_BACKEND", "metal"))
  run(ctx, 136, 19, 128, forceWide = true)
  run(ctx, 256, 21, 0, forceWide = true)
  if getEnv("BYBY_Q4_TEST_SMALL", "0") != "1":
    run(ctx, 8096, 8096, 128)
    run(ctx, 8192, 8192, 128)
    run(ctx, 4096, 11008, 128)
    run(ctx, 11008, 4096, 128)
  if ctx.kind == bkMetal:
    echo "PASS Q4 Metal: finite outputs, legacy/tuned parity and FP64 samples"
  else:
    let stats = vkDrv.vulkanQ4TuningStats()
    doAssert stats.preparedPlans > 0 and stats.wideDispatches > 0,
      "Vulkan autotuner must actually dispatch validated wide kernels"
    echo "PASS Q4 Vulkan: legacy/tuned parity, FP64 samples; plans=", stats.preparedPlans,
      " wideDispatches=", stats.wideDispatches
