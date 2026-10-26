## Test: fp4_gemv_vm (mới, vòng chạy GPU) == bản gốc (unroll host), so BITWISE trên interpreter tham chiếu.
import std/[math, strutils]
import session_vm, vm_interp
import fp4_gemv_vm as nw
import fp4_gemv_vm_orig as old

var seed = 777'u32
proc rnd(): float32 =
  seed = seed * 1664525'u32 + 1013904223'u32
  float32(int(seed shr 8) mod 2000 - 1000) / 1000'f32
proc rbuf(n: int): seq[float32] =
  result = newSeq[float32](n); for i in 0 ..< n: result[i] = rnd()

var fails = 0
proc cmp(name: string; pn, po: VmProgram; nThreads, tg, memN: int; lutN, inN, u8N: int) =
  let m0 = rbuf(memN)
  gLut = rbuf(lutN); gIn = rbuf(inN)
  gU8 = newSeq[uint8](u8N)
  for i in 0 ..< u8N:
    seed = seed * 1664525'u32 + 1013904223'u32
    gU8[i] = uint8(seed shr 16)
  var a = m0; var b = m0
  runGrid(pn, a, nThreads, tg, 4096); runGrid(po, b, nThreads, tg, 4096)
  var diff = 0
  for i in 0 ..< memN:
    if a[i] != b[i] and not (a[i].isNaN and b[i].isNaN): inc diff
  echo (if diff == 0: "OK   " else: "LECH "), name, " (", diff, " lech)"
  if diff != 0: inc fails

for km in [false, true]:
  let K = 128; let N = 5
  let lutN = if km: nw.Fp4KmLutLen else: nw.Fp4LutLen
  cmp("fp4Gemv km=" & $km, nw.buildFp4GemvProgram(K, N, km), old.buildFp4GemvProgram(K, N, km),
      N * nw.GemvTg, nw.GemvTg, 64, lutN, K, N * K + N * (K div 32) + 64)
cmp("f32Gemv acc", nw.buildF32GemvProgram(64, 4, 0.5'f32, true), old.buildF32GemvProgram(64, 4, 0.5'f32, true),
    4 * nw.GemvTg, nw.GemvTg, 4 * 64 + 64, 4 * 64 + 64, 64, 16)
cmp("f32Gemv set", nw.buildF32GemvProgram(64, 4, 1.0'f32, false), old.buildF32GemvProgram(64, 4, 1.0'f32, false),
    4 * nw.GemvTg, nw.GemvTg, 4 * 64 + 64, 4 * 64 + 64, 64, 16)
for gs in [0, 64, 128]:
  let K = 256; let N = 3
  cmp("int4Asym gs=" & $gs, nw.buildInt4AsymGemvProgram(K, N, gs), old.buildInt4AsymGemvProgram(K, N, gs),
      N * nw.GemvTg, nw.GemvTg, 64, 2 * N * 8, K, N * K div 2 + 16)
doAssert fails == 0
echo "ALL OK"
