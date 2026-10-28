## Kiểm tra pipeline kernel toàn lớp Mamba2 (mamba2_full_vm) trên interpreter tham chiếu:
##  - forward  so với tham chiếu double viết thẳng từ công thức gốc (mamba2_real.nim cũ)
##  - backward so với gradient sai phân số (FD) của loss = sum(dNormedOut * normedOut)
import std/[math, strutils]
import session_vm, vm_interp, mamba2_full_vm
import std/strutils as su2

let (B, T, H, D, N, G, K) = (2, 5, 4, 4, 3, 2, 4)
let C = H * D
let eps = 1e-5'f32
let L = m2Layout(B, T, H, D, N, G, K, C, eps)
let (cDim, P, HD, GN, BT) = (L.cDim, L.P, L.HD, L.GN, L.BT)
let hpg = H div G
var seed = 4242'u32
proc rnd(s = 1.0): float64 =
  seed = seed * 1664525'u32 + 1013904223'u32
  (float64(int(seed shr 8) mod 2000 - 1000) / 1000.0) * s

var buf = newSeq[float32](L.total)
template fill(off, n: int; s: float) =
  for i in 0 ..< n: buf[off + i] = float32(rnd(s))
fill(L.proj, BT * P, 0.8); fill(L.convW, cDim * K, 0.5); fill(L.convB, cDim, 0.3)
fill(L.aLog, H, 0.5); fill(L.dSkip, H, 0.7); fill(L.dtBias, H, 0.3); fill(L.normW, C, 0.6)
for i in 0 ..< C: buf[L.normW + i] += 1.0'f32
fill(L.dNormedOut, BT * C, 1.0)

# ── tham chiếu double: forward đầy đủ, trả về các tensor trung gian ──
type Ref = object
  pre, conv, dt, states, yPre, gated, rms, normedPre, normedOut: seq[float64]
proc fwd(pm: seq[float32]): Ref =
  proc sg(x: float64): float64 = 1.0 / (1.0 + exp(-x))
  var r: Ref
  r.pre = newSeq[float64](BT * cDim); r.conv = r.pre; r.dt = newSeq[float64](BT * H)
  r.states = newSeq[float64](B * (T + 1) * HD * N)
  r.yPre = newSeq[float64](BT * C); r.gated = r.yPre; r.normedPre = r.yPre; r.normedOut = r.yPre
  r.rms = newSeq[float64](BT)
  for b in 0 ..< B:
    for t in 0 ..< T:
      let bt = b * T + t
      for c in 0 ..< cDim:
        var acc = float64(pm[L.convB + c])
        for k in 0 ..< K:
          let tp = t + k - (K - 1)
          if tp >= 0: acc += float64(pm[L.convW + c * K + k]) * float64(pm[L.proj + (b * T + tp) * P + C + c])
        r.pre[bt * cDim + c] = acc; r.conv[bt * cDim + c] = acc * sg(acc)
      for h in 0 ..< H:
        let x = float64(pm[L.proj + bt * P + C + cDim + h]) + float64(pm[L.dtBias + h])
        r.dt[bt * H + h] = (if x > 20: x else: ln(1 + exp(x)))
  for b in 0 ..< B:
    for hd in 0 ..< HD:
      let h = hd div D; let g = h div hpg
      let A = -exp(float64(pm[L.aLog + h]))
      for t in 0 ..< T:
        let bt = b * T + t
        let dtv = r.dt[bt * H + h]; let dA = exp(dtv * A); let x = r.conv[bt * cDim + hd]
        var y = 0.0
        for n in 0 ..< N:
          let bv = r.conv[bt * cDim + C + g * N + n]; let cv = r.conv[bt * cDim + C + GN + g * N + n]
          let sp = r.states[((b * (T + 1) + t) * HD + hd) * N + n]
          let sn = sp * dA + dtv * bv * x
          r.states[((b * (T + 1) + t + 1) * HD + hd) * N + n] = sn
          y += cv * sn
        y += float64(pm[L.dSkip + h]) * x
        r.yPre[bt * C + hd] = y
  for bt in 0 ..< BT:
    var s2 = 0.0
    for c in 0 ..< C:
      let z = float64(pm[L.proj + bt * P + c])
      let gv = r.yPre[bt * C + c] * (z * sg(z)); r.gated[bt * C + c] = gv; s2 += gv * gv
    let rr = 1.0 / sqrt(s2 / float64(C) + float64(eps)); r.rms[bt] = rr
    for c in 0 ..< C:
      r.normedPre[bt * C + c] = r.gated[bt * C + c] * rr
      r.normedOut[bt * C + c] = r.normedPre[bt * C + c] * float64(pm[L.normW + c])
  r

var fails = 0
proc cmpRef(name: string; off: int; want: seq[float64]; tol = 2e-4) =
  var worst = 0.0
  for i in 0 ..< want.len:
    worst = max(worst, abs(want[i] - float64(buf[off + i])) / max(1.0, abs(want[i])))
  echo (if worst < tol: "OK   " else: "LECH "), name, ": sai so ", formatFloat(worst, ffScientific, 2)
  if worst >= tol: inc fails

let tgR = rmsTg(L)
runGrid(buildM2ConvSiluProgram(L), buf, BT * M2Tg, M2Tg)
runGrid(buildM2DtProgram(L), buf, BT * M2Tg, M2Tg)
runGrid(buildM2ScanProgram(L), buf, B * HD, B * HD)
runGrid(buildM2RmsFwdProgram(L), buf, BT * tgR, tgR)
let rf = fwd(buf)
cmpRef("fwd pre", L.pre, rf.pre); cmpRef("fwd conv", L.conv, rf.conv); cmpRef("fwd dt", L.dt, rf.dt)
cmpRef("fwd states", L.states, rf.states); cmpRef("fwd yPre", L.yPre, rf.yPre)
cmpRef("fwd gated", L.gated, rf.gated); cmpRef("fwd rms", L.rms, rf.rms)
cmpRef("fwd normedPre", L.normedPre, rf.normedPre); cmpRef("fwd normedOut", L.normedOut, rf.normedOut)

# ── backward ──
let fwdBuf = buf
runGrid(buildM2RmsBwdProgram(L), buf, BT * tgR, tgR)
runGrid(buildM2ScanBwdProgram(L), buf, B * HD, B * HD)
runGrid(buildM2DtBwdProgram(L), buf, BT * M2Tg, M2Tg)
runGrid(buildM2ConvBwdProgram(L), buf, BT * M2Tg, M2Tg)

proc loss(pm: seq[float32]): float64 =
  let r = fwd(pm)
  for i in 0 ..< BT * C: result += float64(fwdBuf[L.dNormedOut + i]) * r.normedOut[i]
proc fd(idx: int): float64 =
  let e = 2e-3
  var a = fwdBuf; var b2 = fwdBuf
  a[idx] += float32(e); b2[idx] -= float32(e)
  (loss(a) - loss(b2)) / (2 * e)
proc chk(name: string; off, n, anOff: int) =
  var worst = 0.0
  for i in 0 ..< n:
    let f = fd(off + i); let an = float64(buf[anOff + i])
    worst = max(worst, abs(f - an) / max(1.0, abs(f)))
  echo (if worst < 1e-2: "OK   " else: "LECH "), "bwd ", name, ": sai so ", formatFloat(worst, ffScientific, 2)
  if worst >= 1e-2: inc fails
# dProj gồm 3 vùng: z | xBC | dt trên mỗi hàng -> so từng phần tử proj
chk("dProj (z, xBC, dt)", L.proj, BT * P, L.dProj)
chk("dNormW", L.normW, C, L.dNormW)
chk("dALog", L.aLog, H, L.dALog); chk("dDSkip", L.dSkip, H, L.dDSkip)
chk("dDtBias", L.dtBias, H, L.dDtBias)
chk("dConvW", L.convW, cDim * K, L.dConvW); chk("dConvB", L.convB, cDim, L.dConvB)

# axpy
block:
  var m = @[1.0'f32, 2, 3, 10, 20, 30]
  runGrid(buildAxpyProgram(0, 3, 0.5'f32), m, 3, 3)
  doAssert m[0] == 6.0'f32 and m[1] == 12.0'f32 and m[2] == 18.0'f32
  echo "OK   axpy"
# ── kernel train: RMSNorm / Linear++ / softmax-CE / Adam gộp ──
block:
  var seed2 = 777'u32
  proc rnd2(s = 1.0): float64 =
    seed2 = seed2 * 1664525'u32 + 1013904223'u32
    (float64(int(seed2 shr 8) mod 2000 - 1000) / 1000.0) * s
  var bad2 = 0
  proc chk2(name: string; got, want: seq[float32]; tol = 2e-4) =
    var worst = 0.0
    for i in 0 ..< want.len:
      let e = abs(float64(got[i]) - float64(want[i])) / (1.0 + abs(float64(want[i])))
      if e > worst: worst = e
    if worst < tol: echo "OK   ", name, ": sai so ", worst.formatFloat(ffScientific, 2)
    else: echo "FAIL ", name, ": sai so ", worst.formatFloat(ffScientific, 2); inc bad2

  # ───────── RMSNorm ─────────
  block:
    let (rows, C) = (5, 24)
    let eps = 1e-5'f32
    let L = rmsLayout(rows, C, eps)
    var buf = newSeq[float32](L.total)
    for i in 0 ..< rows * C: buf[L.x + i] = float32(rnd2(1.5))
    for i in 0 ..< C: buf[L.gamma + i] = float32(1.0 + rnd2(0.4))
    for i in 0 ..< rows * C: buf[L.dy + i] = float32(rnd2(1.0))
    let tg = rmsRowTg(C)
    runGrid(buildRmsFwdProgram(L), buf, rows * tg, tg)
    runGrid(buildRmsBwdProgram(L), buf, rows * tg, tg)
    var y = newSeq[float32](rows * C); var dx = y; var dg = newSeq[float32](C)
    var gotY = buf[L.y ..< L.y + rows * C]; var gotDx = buf[L.dx ..< L.dx + rows * C]
    var gotDg = buf[L.dGamma ..< L.dGamma + C]
    for r in 0 ..< rows:
      var ms = 0.0
      for c in 0 ..< C: ms += float64(buf[L.x + r*C + c]) ^ 2
      ms /= C.float64
      let rr = 1.0 / sqrt(ms + float64(eps))
      var dot = 0.0
      for c in 0 ..< C:
        y[r*C+c] = float32(float64(buf[L.x + r*C+c]) * rr * float64(buf[L.gamma + c]))
        dot += float64(buf[L.dy + r*C+c]) * float64(buf[L.gamma + c]) * float64(buf[L.x + r*C+c])
      for c in 0 ..< C:
        let x = float64(buf[L.x + r*C+c]); let g = float64(buf[L.gamma + c]); let d = float64(buf[L.dy + r*C+c])
        dx[r*C+c] = float32(rr*g*d - rr*rr*rr/C.float64 * x * dot)
        dg[c] += float32(d * x * rr)
    chk2("rms fwd y", gotY, y); chk2("rms bwd dX", gotDx, dx); chk2("rms bwd dGamma", gotDg, dg)

  # ───────── Linear++ ─────────
  block:
    let (B, T, H, D) = (2, 6, 3, 4)
    let C = H * D
    let L = lpLayout(B, T, H, D)
    var buf = newSeq[float32](L.total)
    for i in 0 ..< L.BT * 3 * C: buf[L.qkv + i] = float32(rnd2(1.2))
    for i in 0 ..< L.BT * C: buf[L.dY + i] = float32(rnd2(1.0))
    let nq = L.BT * 2 * C; let tg = lpPhiTg(L)
    runGrid(buildLpPhiProgram(L), buf, nq, tg)
    let BH = B * H; let BHD = B * H * D
    runGrid(buildLpDenProgram(L), buf, BH, BH)
    runGrid(buildLpFwdProgram(L), buf, BHD, BHD)
    runGrid(buildLpDDenProgram(L), buf, L.BT * H, L.BT * H)
    runGrid(buildLpBwdQProgram(L), buf, BHD, BHD)
    runGrid(buildLpBwdKProgram(L), buf, BHD, BHD)
    runGrid(buildLpBwdVProgram(L), buf, BHD, BHD)
    # tham chiếu double
    proc elu1(x: float64): float64 = (if x > 0: x + 1 else: exp(x))
    proc eluG(x: float64): float64 = (if x > 0: 1.0 else: exp(x))
    let qkv0 = buf[L.qkv ..< L.qkv + L.BT * 3 * C]
    proc fwdRef(qkv: seq[float32]): tuple[y, inv: seq[float64]] =
      result.y = newSeq[float64](L.BT * C); result.inv = newSeq[float64](L.BT * H)
      for b in 0 ..< B:
        for i in 0 ..< T:
          for h in 0 ..< H:
            var den = 0.0; var num = newSeq[float64](D)
            for j in 0 .. i:
              var s = 0.0
              for d in 0 ..< D:
                s += elu1(float64(qkv[(b*T+i)*3*C + h*D + d])) * elu1(float64(qkv[(b*T+j)*3*C + C + h*D + d]))
              den += s
              for d in 0 ..< D: num[d] += s * float64(qkv[(b*T+j)*3*C + 2*C + h*D + d])
            let inv = 1.0 / (den + float64(LpEps))
            result.inv[(b*T+i)*H + h] = inv
            for d in 0 ..< D: result.y[(b*T+i)*C + h*D + d] = num[d] * inv
    let (yr, invr) = fwdRef(qkv0)
    var yRef = newSeq[float32](yr.len); for i in 0 ..< yr.len: yRef[i] = float32(yr[i])
    var iRef = newSeq[float32](invr.len); for i in 0 ..< invr.len: iRef[i] = float32(invr[i])
    chk2("lp fwd y", buf[L.y ..< L.y + L.BT * C], yRef)
    chk2("lp fwd invDen", buf[L.invDen ..< L.invDen + L.BT * H], iRef, 1e-3)
    # FD: loss = Σ dY * y ; kiểm dQKV
    let dY = buf[L.dY ..< L.dY + L.BT * C]
    proc loss(qkv: seq[float32]): float64 =
      let r = fwdRef(qkv)
      for i in 0 ..< r.y.len: result += float64(dY[i]) * r.y[i]
    var fd = newSeq[float32](qkv0.len)
    let h = 1e-3
    for i in 0 ..< qkv0.len:
      var a = qkv0; var m = qkv0
      a[i] += float32(h); m[i] -= float32(h)
      fd[i] = float32((loss(a) - loss(m)) / (2*h))
    chk2("lp bwd dQ", buf[L.dQKV ..< L.dQKV + L.BT*3*C][0 .. 0], fd[0 .. 0], 1e-2) # sanity trivial
    var gq, gk, gv, fq, fk, fv: seq[float32]
    for t in 0 ..< L.BT:
      for c in 0 ..< C:
        gq.add buf[L.dQKV + t*3*C + c]; fq.add fd[t*3*C + c]
        gk.add buf[L.dQKV + t*3*C + C + c]; fk.add fd[t*3*C + C + c]
        gv.add buf[L.dQKV + t*3*C + 2*C + c]; fv.add fd[t*3*C + 2*C + c]
    chk2("lp bwd dQ (FD)", gq, fq, 2e-3); chk2("lp bwd dK (FD)", gk, fk, 2e-3); chk2("lp bwd dV (FD)", gv, fv, 2e-3)

  # ───────── Cross-entropy có mask ─────────
  block:
    let (rows, V) = (5, 48)
    let L = ceLayout(rows, V)
    var buf = newSeq[float32](L.total)
    for i in 0 ..< rows * V: buf[L.logits + i] = float32(rnd2(3.0))
    let tgts = [7, -1, 0, 47, 20]
    var nValid = 0
    for r in 0 ..< rows:
      buf[L.tgt + r] = float32(tgts[r]); (if tgts[r] >= 0: inc nValid)
    let invN = 1.0 / nValid.float64
    buf[L.scal] = float32(invN)
    let tg = rmsRowTg(V)
    runGrid(buildCeProgram(L), buf, rows * tg, tg)
    runGrid(buildCeSumProgram(L), buf, 1, 1)
    var dRef = newSeq[float32](rows * V); var lossRef = 0.0
    for r in 0 ..< rows:
      if tgts[r] < 0: continue
      var m = -1e30
      for v in 0 ..< V: m = max(m, float64(buf[L.logits + r*V + v]))
      var s = 0.0
      for v in 0 ..< V: s += exp(float64(buf[L.logits + r*V + v]) - m)
      for v in 0 ..< V:
        let pr = exp(float64(buf[L.logits + r*V + v]) - m) / s
        dRef[r*V+v] = float32((pr - (if v == tgts[r]: 1.0 else: 0.0)) * invN)
      lossRef += (m + ln(s) - float64(buf[L.logits + r*V + tgts[r]])) * invN
    chk2("ce dLogits", buf[L.dLogits ..< L.dLogits + rows*V], dRef)
    chk2("ce loss", @[buf[L.sum]], @[float32(lossRef)], 1e-5)

  # ───────── Adam gộp ─────────
  block:
    let n = 512
    let L = adamLayout(n)
    var buf = newSeq[float32](L.total)
    for i in 0 ..< n:
      buf[L.p + i] = float32(rnd2(1.0)); buf[L.g + i] = float32(rnd2(0.5))
      buf[L.m + i] = float32(rnd2(0.1)); buf[L.v + i] = float32(abs(rnd2(0.01)))
    let (lr, b1, b2, eps) = (1e-3'f32, 0.9'f32, 0.999'f32, 1e-8'f32)
    let step = 7
    let bc1 = 1.0 - pow(float64(b1), step.float64); let bc2 = 1.0 - pow(float64(b2), step.float64)
    buf[L.scal] = lr; buf[L.scal+1] = float32(1.0/bc1); buf[L.scal+2] = float32(1.0/bc2)
    var pRef = newSeq[float32](n); var mRef = pRef; var vRef = pRef
    for i in 0 ..< n:
      let g = float64(buf[L.g+i])
      let m = float64(b1) * float64(buf[L.m+i]) + (1 - float64(b1)) * g
      let v = float64(b2) * float64(buf[L.v+i]) + (1 - float64(b2)) * g * g
      mRef[i] = float32(m); vRef[i] = float32(v)
      pRef[i] = float32(float64(buf[L.p+i]) - float64(lr) * (m/bc1) / (sqrt(v/bc2) + float64(eps)))
    runGrid(buildAdamProgram(L, b1, b2, eps), buf, n, AdamTg)
    chk2("adam p", buf[L.p ..< L.p + n], pRef, 1e-5)
    chk2("adam m", buf[L.m ..< L.m + n], mRef, 1e-5)
    chk2("adam v", buf[L.v ..< L.v + n], vRef, 1e-5)

  # lặp lại RMS + CE với giới hạn threadgroup nhỏ (GPU Intel chỉ 256; ở đây ép 4) -> nhiều phần tử / lane
  gTgCap = 4
  block:
    let (rows, C) = (5, 24)
    let L = rmsLayout(rows, C, 1e-5'f32)
    var b0 = newSeq[float32](L.total)
    for i in 0 ..< rows * C: b0[L.x + i] = float32(rnd2(1.5)); b0[L.dy + i] = float32(rnd2(1.0))
    for i in 0 ..< C: b0[L.gamma + i] = float32(1.0 + rnd2(0.4))
    var b1 = b0
    gTgCap = 1024
    runGrid(buildRmsFwdProgram(L), b0, rows * rmsRowTg(C), rmsRowTg(C))
    runGrid(buildRmsBwdProgram(L), b0, rows * rmsRowTg(C), rmsRowTg(C))
    gTgCap = 4
    runGrid(buildRmsFwdProgram(L), b1, rows * rmsRowTg(C), rmsRowTg(C))
    runGrid(buildRmsBwdProgram(L), b1, rows * rmsRowTg(C), rmsRowTg(C))
    chk2("rms tg=4 y", b1[L.y ..< L.y + rows*C], b0[L.y ..< L.y + rows*C], 1e-5)
    chk2("rms tg=4 dX", b1[L.dx ..< L.dx + rows*C], b0[L.dx ..< L.dx + rows*C], 1e-5)
  block:
    let (rows, V) = (5, 48)
    let L = ceLayout(rows, V)
    var b0 = newSeq[float32](L.total)
    for i in 0 ..< rows * V: b0[L.logits + i] = float32(rnd2(3.0))
    let tg2 = [7, -1, 0, 47, 20]
    for r in 0 ..< rows: b0[L.tgt + r] = float32(tg2[r])
    b0[L.scal] = 0.25'f32
    var b1 = b0
    gTgCap = 1024
    runGrid(buildCeProgram(L), b0, rows * rmsRowTg(V), rmsRowTg(V))
    gTgCap = 4
    runGrid(buildCeProgram(L), b1, rows * rmsRowTg(V), rmsRowTg(V))
    chk2("ce tg=4 dLogits", b1[L.dLogits ..< L.dLogits + rows*V], b0[L.dLogits ..< L.dLogits + rows*V], 1e-5)
    chk2("ce tg=4 rowLoss", b1[L.rowLoss ..< L.rowLoss + rows], b0[L.rowLoss ..< L.rowLoss + rows], 1e-5)
  gTgCap = 1024
  fails += bad2
doAssert fails == 0, $fails & " kiem tra LECH"
echo "ALL OK"
