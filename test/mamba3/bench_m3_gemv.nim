## bench_m3_gemv.nim - FP4-KM decode GEMV variants on the Mamba-3 187M shapes, on the real GPU.
## No model is loaded. Each shape gets enough weight copies to stream past the GPU caches (as in a
## real decode step, where ~100 MB of weights go through per token), every variant is checked
## against the tuned kernel and timed from the command buffer GPU time.
##
##   nim c -d:release -o:test/mamba3/bench_m3_gemv_bin test/mamba3/bench_m3_gemv.nim
##   BYBY_BACKEND=metal test/mamba3/bench_m3_gemv_bin
import std/[os, strutils, strformat, math]
import backend
import vendor/bybylang/gpubackend as byby
import fp4_gemv_vm, fp4_gemv_gpu
import gpu_layers, session_vm

type Variant = object
  label, kn: string
  tg, rows, bytes: int

proc main() =
  let ctx = newBackend(getEnv("BYBY_BACKEND", "auto"))
  let be = ctx.kind.toByby()
  ensureDummy(be)
  let iters = parseInt(getEnv("BYBY_BENCH_ITERS", "48"))
  let shapes = [(768, 3432), (1536, 768), (768, 3072), (768, 128256)]
  for (K, N) in shapes:
    var seed = 7'u32
    proc nextU(): uint32 =
      seed = seed * 1664525'u32 + 1013904223'u32
      seed
    let bytesPer = N * K div 2 + N * (K div 32)
    let copies = clamp(96 * 1024 * 1024 div bytesPer, 1, 64)
    var handles: seq[SessionHandle]
    for c in 0 ..< copies:
      var wq = newSeq[uint8](bytesPer)
      for i in 0 ..< wq.len: wq[i] = uint8(nextU() shr 16)
      handles.add sessionUploadBytesShared(be, wq)
    var lut = newSeq[float32](Fp4KmLutLen)
    for i in 0 ..< 16: lut[i] = float32(i - 8) * 0.125'f32
    for i in 16 ..< Fp4KmLutLen: lut[i] = 0.01'f32 + float32(i and 15) * 0.002'f32
    let hLut = byby.sessionUpload(be, lut)
    let off = parseInt(getEnv("BENCH_OFF", "0"))   # element offset of the input and output views
    var x = newSeq[float32](2 * K + off)
    for i in 0 ..< x.len: x[i] = float32(int(nextU() shr 8) mod 2000 - 1000) / 1000'f32
    let hX = byby.sessionUpload(be, x)
    let hY = byby.sessionAllocScratch(be, N + off)
    let hBw = byby.sessionAllocScratch(be, bytesPer div 4 + 64)
    var variants: seq[Variant]
    let only = getEnv("BENCH_TG", "32")
    for tg in [16, 32, 64]:
      if $tg notin only.split(','): continue
      if K mod (tg * 8) != 0 or N mod 4 != 0: continue
      if be == byby.gbMetal and tg >= 32:
        variants.add Variant(label: "simd4-tg" & $tg, tg: tg, rows: 4,
          bytes: (Fp4KmLutLen + 4 * tg) * 4,
          kn: vmKernelForBe(buildFp4KmGemvSimdProgram(K, N, tg, 4), &"bm_s4_{tg}_{K}x{N}", be))
      for rows in [4, 8, 16]:
        if not fp4KmGemvDecodeSupported(K, N, tg, rows): continue
        let simd = be == byby.gbMetal
        for lc in (if getEnv("BENCH_FAKE", "0") == "1": @[0, 2] else: @[0]):
          variants.add Variant(label: &"dec{rows}{(if lc == 2: \"FAKE\" else: \"\")}-tg{tg}", tg: tg, rows: rows,
            bytes: fp4KmGemvDecodeTgBytes(rows, tg),
            kn: vmKernelForBe(buildFp4KmGemvDecodeProgram(K, N, tg, rows, simd, lutConst = lc == 1, fakeLut = lc == 2),
              &"bm_d{rows}{lc}_{tg}_{K}x{N}", be))
    if getEnv("BENCH_BW", "0") == "1":
      # Pure read bandwidth: every lane sums `per` u32 words with a stride of the whole grid.
      for (tgB, per) in [(256, 8), (256, 16), (128, 16), (64, 16)]:
        let words = bytesPer div 4
        let threads = (words div per) div tgB * tgB
        var bp: VmProgram
        bp.emitUTgid(0); bp.emitULid(1); bp.emitUMulK(2, 0, uint32(tgB)); bp.emitUAdd(2, 2, 1)
        bp.emitUMulK(2, 2, 4)
        bp.emitLoadConstF32(4, 0)
        for i in 0 ..< per:
          bp.emitULoadU32(3, 2, uint32(i * threads * 4)); bp.emitUToR(5, 3); bp.emitAdd(4, 4, 5)
        bp.emitUShrK(2, 2, 2); bp.emitUStoreF32(4, 2, 0); bp.emitHalt()
        variants.add Variant(label: &"bw-tg{tgB}x{per}", tg: tgB, rows: 1, bytes: 16,
          kn: vmKernelForBe(bp, &"bm_bw_{tgB}_{per}_{K}x{N}", be))
        variants[^1].rows = -threads
    echo &"\n== K={K} N={N}: {bytesPer.float / 1048576.0:.2f} MB per copy, {copies} copies, {iters} dispatches =="
    var reference: seq[float32]
    for vi in 0 ..< variants.len:
      let v = variants[vi]
      proc run(n: int): float =
        let g0 = byby.sessionGpuSeconds(be)
        doAssert byby.sessionBegin(be)
        for i in 0 ..< n:
          encodeVmKernel(v.kn, (if v.rows < 0: -v.rows else: (N div v.rows) * v.tg), v.tg, v.bytes, (if v.rows < 0: hBw else: hY), off,
            handles[i mod copies], hLut, hX, off)
        doAssert byby.sessionEnd(be), byby.sessionGetLastError(be)
        byby.sessionGpuSeconds(be) - g0
      discard run(1)
      let outv = byby.sessionRead(be, hY, N + off)[off ..< off + N]
      var note = "reference"
      if v.rows < 0: note = "bandwidth"
      elif reference.len == 0: reference = outv
      else:
        var err, mag = 0.0
        for i in 0 ..< N:
          err = max(err, abs(float(outv[i]) - float(reference[i]))); mag = max(mag, abs(float(reference[i])))
        note = (if err <= max(1e-5, mag * 1e-4): "OK " else: "DIFF ") & &"{err / max(mag, 1e-9):.1e}"
      var best = 1e9
      for _ in 0 ..< 3: best = min(best, run(iters))
      let us = best * 1e6 / float(iters)
      echo &"  {v.label:14s} {us:8.1f} us  {bytesPer.float / (us * 1e-6) / 1e9:6.2f} GB/s  {note}"

main()
