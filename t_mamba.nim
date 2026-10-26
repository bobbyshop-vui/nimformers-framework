## t_mamba.nim — interpreter THAM CHIẾU (chỉ để test, CPU được phép) chạy bytecode
## nhiều thread có barrier; so (1) kernel forward mới == bản gốc, (2) backward == FD.
import std/[math, strutils]
import session_vm
import mamba2_vm_kernels as nw
import mamba2_old_fwd as old

import vm_interp
proc randBuf(n: int; seed: var uint32): seq[float32] =
  result = newSeq[float32](n)
  for i in 0 ..< n:
    seed = seed * 1664525'u32 + 1013904223'u32
    result[i] = float32(int(seed shr 8) mod 2000 - 1000) / 1000'f32 * 0.8'f32

var seed = 12345'u32
var fails = 0
proc same(name: string; pn, po: VmProgram; n, tg, total: int) =
  var m1 = randBuf(total, seed); var m2 = m1
  runGrid(pn, m1, n, tg); runGrid(po, m2, n, tg)
  var diff = 0
  for i in 0 ..< total:
    if m1[i] != m2[i]: inc diff
  echo (if diff == 0: "OK   " else: "LECH "), name, " (", diff, " phan tu lech)"
  if diff != 0: inc fails

# ── (1) forward mới == bản gốc (bitwise) ──
same("rmsgate C=8", nw.buildRmsGateProgram(8, 1e-5'f32), old.buildRmsGateProgram(8, 1e-5'f32), 8, 8, 64)
same("rmsgate C=1", nw.buildRmsGateProgram(1, 1e-5'f32), old.buildRmsGateProgram(1, 1e-5'f32), 1, 1, 16)
same("rmsgateLarge C=32 tg=8", nw.buildRmsGateLargeProgram(32, 1e-5'f32, 8), old.buildRmsGateLargeProgram(32, 1e-5'f32, 8), 8, 8, 128)
same("rmsgateGrouped C=32 g=2 tg=8", nw.buildRmsGateGroupedAt(32, 2, 8, 1e-5'f32, 0, 32, 64, 96), old.buildRmsGateGroupedAt(32, 2, 8, 1e-5'f32, 0, 32, 64, 96), 16, 8, 128)
for K in 2 .. 5:
  let cDim = 6
  let tot = cDim * (K - 1) + cDim * K + cDim * 3
  same("conv1dInc K=" & $K, nw.buildConv1dIncProgram(cDim, K), old.buildConv1dIncProgram(cDim, K), cDim, cDim, tot)
  let C = 5; let H = 3
  let offConvW = 40; let offConvB = 40 + cDim * K; let offSt = offConvB + cDim
  let offOutSt = offSt + cDim * (K - 1); let offOut = offOutSt + cDim * (K - 1); let tot2 = offOut + 64
  same("fusedSplitConv K=" & $K, nw.buildFusedSplitConvSoftplusProgram(C, cDim, H, K), old.buildFusedSplitConvSoftplusProgram(C, cDim, H, K), 8, 8, tot2 + 400)
  same("splitConvAt K=" & $K,
    nw.buildSplitConvSoftplusAt(C, cDim, H, K, 20, 30, 45, 60, 100, 150, 200),
    old.buildSplitConvSoftplusAt(C, cDim, H, K, 20, 30, 45, 60, 100, 150, 200), 8, 8, 400)

# ── (2) backward == gradient sai phân số (double) ──
let (B, T, H, D, N, G) = (2, 3, 4, 2, 3, 2)
let HD = H * D
let hpg = H div G
let offX = 0; let offB = offX + B*T*HD; let offC = offB + B*T*G*N; let offDt = offC + B*T*G*N
let offALog = offDt + B*T*H; let offDSkip = offALog + H; let offState = offDSkip + H
let offDY = offState + B*(T+1)*HD*N
let offDX = offDY + B*T*HD; let offDB = offDX + B*T*HD; let offDC = offDB + B*T*G*N
let offDDt = offDC + B*T*G*N; let offDALog = offDDt + B*T*H; let offDDSkip = offDALog + H
let total = offDDSkip + H
var mem = newSeq[float32](total)
let rnd = randBuf(total, seed)
for i in 0 ..< offState: mem[i] = rnd[i]
for i in 0 ..< B*T*H: mem[offDt + i] = abs(rnd[offDt + i]) * 0.5'f32 + 0.1'f32
for i in 0 ..< B*T*HD: mem[offDY + i] = rnd[offDY + i]
let s0 = rnd                       # trạng thái đầu t=0 lấy từ rnd

proc ix(b, t, hd, n: int): int = ((b * (T+1) + t) * HD + hd) * N + n
for b in 0 ..< B:
  for hd in 0 ..< HD:
    for n in 0 ..< N: mem[offState + ix(b, 0, hd, n)] = s0[offState + ix(b, 0, hd, n)]

proc fwdLoss(pm: seq[float32]; fillStates: bool; stOut: var seq[float32]): float64 =
  var L = 0.0
  for b in 0 ..< B:
    for hd in 0 ..< HD:
      let h = hd div D; let g = h div hpg
      let A = -exp(float64(pm[offALog + h])); let Dk = float64(pm[offDSkip + h])
      var s = newSeq[float64](N)
      for n in 0 ..< N: s[n] = float64(pm[offState + ix(b, 0, hd, n)])
      for t in 0 ..< T:
        let x = float64(pm[offX + (b*T + t)*HD + hd]); let dt = float64(pm[offDt + (b*T + t)*H + h])
        let dA = exp(dt * A)
        var y = Dk * x
        for n in 0 ..< N:
          let Bv = float64(pm[offB + ((b*T + t)*G + g)*N + n]); let Cv = float64(pm[offC + ((b*T + t)*G + g)*N + n])
          s[n] = s[n] * dA + dt * Bv * x
          y += Cv * s[n]
          if fillStates: stOut[offState + ix(b, t+1, hd, n)] = float32(s[n])
        L += float64(pm[offDY + (b*T + t)*HD + hd]) * y
  L

var withStates = mem
discard fwdLoss(mem, true, withStates)
mem = withStates

let prog = nw.buildMamba2BackwardFullAt(B, T, H, D, N, G, offX, offB, offC, offDt, offALog, offDSkip,
  offState, offDY, offDX, offDB, offDC, offDDt, offDALog, offDDSkip)
# dX cố tình đổ rác để chứng minh kernel tự ghi đè (không cần zero)
for i in 0 ..< B*T*HD: mem[offDX + i] = 777'f32
var dummy = mem
runGrid(prog, mem, B * HD, B * HD)

proc fd(idx: int): float64 =
  let eps = 1e-3
  var a = mem; var b2 = mem
  var d1 = mem; var d2 = mem
  a[idx] += float32(eps); b2[idx] -= float32(eps)
  # dùng double thật cho tham số: tính lại bằng bản sao + eps nhỏ ở float32 vẫn đủ cho tolerance dưới
  (fwdLoss(a, false, d1) - fwdLoss(b2, false, d2)) / (2 * eps)

proc chk(name: string; off, n, anOff: int) =
  var worst = 0.0
  for i in 0 ..< n:
    let f = fd(off + i); let an = float64(mem[anOff + i])
    worst = max(worst, abs(f - an) / max(1.0, abs(f)))
  echo (if worst < 2e-2: "OK   " else: "LECH "), "backward ", name, ": sai so tuong doi lon nhat = ", formatFloat(worst, ffScientific, 2)
  if worst >= 2e-2: inc fails
chk("dX", offX, B*T*HD, offDX)
chk("dB", offB, B*T*G*N, offDB)
chk("dC", offC, B*T*G*N, offDC)
chk("dDt", offDt, B*T*H, offDDt)
chk("dALog", offALog, H, offDALog)
chk("dDSkip", offDSkip, H, offDDSkip)

# ── (3) rms-gate backward == công thức double ──
block:
  let C = 24; let eps = 1e-5'f32
  let tgs = nw.rmsBwdTgSize(C)
  doAssert tgs == 8
  var mm = randBuf(7 * C, seed)
  for i in 0 ..< 3 * C: mm[i] = mm[i] + (if i >= 2*C: 1.0'f32 else: 0'f32)
  let ref0 = mm
  runGrid(nw.buildRmsGateBackwardProgram(C, eps, tgs), mm, tgs, tgs)
  var sumG2 = 0.0; var dot = 0.0
  var gated = newSeq[float64](C); var silu = newSeq[float64](C); var sg = newSeq[float64](C)
  for i in 0 ..< C:
    let z = float64(ref0[C + i]); sg[i] = 1.0 / (1.0 + exp(-z)); silu[i] = z * sg[i]
    gated[i] = float64(ref0[i]) * silu[i]; sumG2 += gated[i] * gated[i]
    dot += float64(ref0[3*C + i]) * float64(ref0[2*C + i]) * gated[i]
  let r = 1.0 / sqrt(sumG2 / C.float64 + float64(eps))
  var worst = 0.0
  for i in 0 ..< C:
    let dnc = float64(ref0[3*C + i]) * float64(ref0[2*C + i])
    let dg = r * dnc - gated[i] * (r*r*r / C.float64) * dot
    let z = float64(ref0[C + i])
    let want = [dg * silu[i], dg * float64(ref0[i]) * (sg[i] * (1 + z * (1 - sg[i]))), float64(ref0[3*C + i]) * gated[i] * r]
    for j in 0 .. 2:
      worst = max(worst, abs(want[j] - float64(mm[(4 + j) * C + i])) / max(1.0, abs(want[j])))
  echo (if worst < 1e-4: "OK   " else: "LECH "), "rmsgate backward: sai so = ", formatFloat(worst, ffScientific, 2)
  if worst >= 1e-4: inc fails

# ── (4) bảng idxH/idxG sinh bằng GPU == công thức ──
for (H, D, G) in [(4, 2, 2), (8, 3, 2), (6, 4, 6), (5, 1, 1)]:
  let HD = H * D; let hpg = max(1, H div G)
  var tb = newSeq[float32](2 * HD)
  runGrid(nw.buildIdxTableProgram(H, D, G), tb, HD, HD)
  var bad = 0
  for tid in 0 ..< HD:
    let h = tid div D
    if int(tb[tid]) != h or int(tb[HD + tid]) != h div hpg: inc bad
  echo (if bad == 0: "OK   " else: "LECH "), "idxTable H=", H, " D=", D, " G=", G
  if bad > 0: inc fails
doAssert fails == 0, "co " & $fails & " kiem tra LECH"
echo "ALL OK"
