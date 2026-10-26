## bench_gemv_variants.nim — đo + kiểm đúng các biến thể kernel GEMV fp4-KM trên GPU THẬT.
## Chạy vài giây (không nạp model). Dùng đúng cỡ thật của Codestral-7B:
##   in_proj 4096->18560, out_proj 8192->4096, lm_head 4096->32768.
##
##   nim c -d:release --hints:off --mm:arc --opt:speed -o:bench_gemv_variants bench_gemv_variants.nim
##   ./bench_gemv_variants            (BYBY_BENCH_ITERS=40 để đo kỹ hơn)
##
## Mỗi biến thể in: µs/dispatch, GB/s đọc weight, và SAI SỐ so với biến thể 0 (cùng weight, cùng x).
##   variant 0 = kernel 2 hàng hiện tại, 1 = nạp rộng 32-bit, 2 = 4 hàng/threadgroup.
import std/[os, strutils, strformat, times, tables, math]
import backend
import vendor/bybylang/gpubackend as byby
import fp4_gemv_gpu
import fp4_gemv_vm
import session_vm
import gpu_layers

proc buildVariant(v, K, N: int): tuple[prog: VmProgram, tag: string, rows: int, ok: bool] =
  case v
  of 1:
    if not (fp4KmGemvWideSupported(K, N)): return (VmProgram(), "", 0, false)
    (buildFp4KmGemvWideProgram(K, N), "bw_fp4kmw_" & $K & "x" & $N, 2, true)
  of 2:
    if N mod 4 != 0 or K mod 32 != 0: return (VmProgram(), "", 0, false)
    (buildFp4KmGemv4RowProgram(K, N), "bw_fp4km4_" & $K & "x" & $N, 4, true)
  else:
    (buildFp4KmGemv2RowProgram(K, N), "bw_fp4km2_" & $K & "x" & $N, 2, true)

proc runShape(ctx: Backend; K, N, iters: int) =
  let be = ctx.kind.toByby()
  var seed = 99'u32
  proc nextU(): uint32 =
    seed = seed * 1664525'u32 + 1013904223'u32
    seed
  # weight + LUT + x giả lập, DÙNG CHUNG cho mọi biến thể
  var wq = newSeq[uint8]((N * K) div 2 + N * (K div 32))
  for i in 0 ..< wq.len: wq[i] = uint8(nextU() shr 16)
  var lut = newSeq[float32](Fp4KmLutLen)
  for i in 0 ..< 16: lut[i] = float32(i - 8) * 0.125'f32
  for i in 16 ..< Fp4KmLutLen: lut[i] = 0.01'f32 + float32(i and 15) * 0.002'f32
  var x = newSeq[float32](K)
  for i in 0 ..< K: x[i] = float32(int(nextU() shr 8) mod 2000 - 1000) / 1000'f32
  let weightMB = float((N * K) div 2 + N * (K div 32)) / (1024.0 * 1024.0)
  echo &"\n== K={K} N={N}  (weight+scale = {weightMB:.1f} MB) =="
  var refOut: seq[float32]
  var baseUs = 0.0
  for v in 0 .. 2:
    let (prog, tag, rows, ok) = buildVariant(v, K, N)
    if not ok:
      echo &"  variant {v}: bo qua (shape khong ho tro)"
      continue
    let name = &"bench_v{v}_{K}x{N}"
    let kn = gpu_layers.vmKernel(prog, tag)
    uploadFp4WeightOnce(ctx, name, wq, lut, N, K, kn, rowsPerTg = rows)
    var hX = byby.sessionUploadPrivate(be, x)
    var hOut = byby.sessionUploadPrivate(be, newSeq[float32](N))
    # warm-up (JIT compile + cache)
    if byby.sessionBegin(be):
      discard encodeFp4Gemv(gFp4CacheGpu[name], hX, 0, hOut, 0)
      discard byby.sessionEnd(be)
    # kiểm đúng: 1 dispatch rồi đọc kết quả
    if byby.sessionBegin(be):
      discard encodeFp4Gemv(gFp4CacheGpu[name], hX, 0, hOut, 0)
      discard byby.sessionEnd(be)
    let outv = byby.sessionRead(be, hOut, N)
    var errStr = "(chuan)"
    if v == 0: refOut = outv
    elif refOut.len == N:
      var maxErr = 0.0
      var maxRef = 1e-9
      for i in 0 ..< N:
        maxErr = max(maxErr, abs(float(outv[i]) - float(refOut[i])))
        maxRef = max(maxRef, abs(float(refOut[i])))
      errStr = &"sai so tuong doi {maxErr / maxRef:.2e}" & (if maxErr / maxRef < 1e-3: "  OK" else: "  *** LECH ***")
    # đo thời gian: nhiều dispatch trong 1 session
    let t0 = epochTime()
    if byby.sessionBegin(be):
      for _ in 0 ..< iters:
        discard encodeFp4Gemv(gFp4CacheGpu[name], hX, 0, hOut, 0)
      discard byby.sessionEnd(be)
    let us = (epochTime() - t0) * 1.0e6 / float(iters)
    if v == 0: baseUs = us
    let gbs = weightMB / 1024.0 / (us / 1.0e6)
    let speed = if baseUs > 0: baseUs / us else: 1.0
    echo &"  variant {v} ({rows} hang/TG): {us:9.1f} us/dispatch   {gbs:6.2f} GB/s   x{speed:.2f} so voi variant 0   {errStr}"

proc main() =
  let ctx = newBackend("auto")
  if byby.sessionBegin(ctx.kind.toByby()):
    discard byby.sessionEnd(ctx.kind.toByby())
  let iters = parseInt(getEnv("BYBY_BENCH_ITERS", "20"))
  runShape(ctx, 4096, 18560, iters)   # in_proj
  runShape(ctx, 8192, 4096, iters)    # out_proj
  runShape(ctx, 4096, 32768, max(2, iters div 4))  # lm_head
  echo "\nChon BYBY_GEMV_VARIANT = variant nhanh nhat (va OK) khi chay infer-lora."

main()
