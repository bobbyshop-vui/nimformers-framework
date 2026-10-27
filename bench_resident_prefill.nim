## Load once, repeat the real prompt to distinguish cold setup from warm prefill.
## nim c -d:release --mm:arc --opt:speed -o:bench_resident_prefill bench_resident_prefill.nim
## BYBY_PREFILL_BENCH_ROUNDS=2 ./bench_resident_prefill [checkpoint] [prompt]
import std/[os, strutils, tables]
import backend, infer_lora
import gpu_layers, fp4_gemv_gpu, fp4_gemv_vm
import vendor/bybylang/gpubackend as byby

when isMainModule:
  let args = commandLineParams()
  let path = if args.len > 0: args[0] else: "jamba_parallel_lora.bin"
  let prompt = if args.len > 1: args[1] else: "[INST] Viet ham fibonacci bang Python[/INST]"
  let ab = getEnv("BYBY_BENCH_AB", "0") == "1"
  let rounds = if ab: 3 else: max(1, parseInt(getEnv("BYBY_PREFILL_BENCH_ROUNDS", "2")))
  let tokens = max(1, parseInt(getEnv("BYBY_BENCH_TOKENS", "8")))
  setInferenceThreadPriority()
  let ctx = newBackend(getEnv("BYBY_BENCH_BACKEND", "metal"))
  var loaded = loadBinaryLoraForInfer(path)
  var tuned = initTable[string, Fp4LinCacheGpu]()
  for r in 1 .. rounds:
    if ab and r == 2:
      for name, c in gFp4CacheGpu: tuned[name] = c
      for name, c in gFp4CacheGpu.mpairs:
        if c.cols mod 32 != 0 or c.rows mod 4 != 0: continue
        c.jitName = vmKernelForBe(buildFp4KmGemv4RowProgram(c.cols, c.rows),
          "ab_legacy_gemv_" & $c.cols & "_" & $c.rows, ctx.kind.toByby())
        c.tgSize = GemvTg; c.rowsPerTg = 4
        c.tgBytes = fp4KmGemvRowsTgBytes(4)
        c.m8JitName = vmKernelForBe(buildFp4KmGemmM8Program(c.cols, c.rows),
          "ab_legacy_m8_" & $c.cols & "_" & $c.rows, ctx.kind.toByby())
        c.m8TgSize = GemvTg
    if ab and r == 3: gFp4CacheGpu = tuned
    let mode = if not ab: "default" elif r == 1: "warmup-new" elif r == 2: "legacy" else: "tuned"
    echo "[bench-prefill] round " & $r & "/" & $rounds & " mode=" & mode
    discard generate(loaded.model, loaded.lora, loaded.tok, prompt, tokens, ctx,
      temperature = 0.7'f32, seed = 1234)
