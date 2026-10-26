## Kiểm tra kernel train_gpu_vm (RMSNorm + Linear++) trên interpreter tham chiếu (CPU):
##  - forward so với tham chiếu double viết từ công thức gốc
##  - backward so với tham chiếu double (đạo hàm giải tích viết độc lập) VÀ sai phân số (FD)
import std/[math, strutils]
import session_vm, vm_interp, train_gpu_vm

var seed = 777'u32
proc rnd(s = 1.0): float64 =
  seed = seed * 1664525'u32 + 1013904223'u32
  (float64(int(seed shr 8) mod 2000 - 1000) / 1000.0) * s
var bad = 0
proc chk(name: string; got, want: seq[float32]; tol = 2e-4) =
  var worst = 0.0
  for i in 0 ..< want.len:
    let e = abs(float64(got[i]) - float64(want[i])) / (1.0 + abs(float64(want[i])))
    if e > worst: worst = e
  if worst < tol: echo "OK   ", name, ": sai so ", worst.formatFloat(ffScientific, 2)
  else: echo "FAIL ", name, ": sai so ", worst.formatFloat(ffScientific, 2); inc bad

# ───────── RMSNorm ─────────
block:
  let (rows, C) = (5, 24)
  let eps = 1e-5'f32
  let L = rmsLayout(rows, C, eps)
  var buf = newSeq[float32](L.total)
  for i in 0 ..< rows * C: buf[L.x + i] = float32(rnd(1.5))
  for i in 0 ..< C: buf[L.gamma + i] = float32(1.0 + rnd(0.4))
  for i in 0 ..< rows * C: buf[L.dy + i] = float32(rnd(1.0))
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
  chk("rms fwd y", gotY, y); chk("rms bwd dX", gotDx, dx); chk("rms bwd dGamma", gotDg, dg)

# ───────── Linear++ ─────────
block:
  let (B, T, H, D) = (2, 6, 3, 4)
  let C = H * D
  let L = lpLayout(B, T, H, D)
  var buf = newSeq[float32](L.total)
  for i in 0 ..< L.BT * 3 * C: buf[L.qkv + i] = float32(rnd(1.2))
  for i in 0 ..< L.BT * C: buf[L.dY + i] = float32(rnd(1.0))
  let nq = L.BT * 2 * C; let tg = lpPhiTg(L)
  runGrid(buildLpPhiProgram(L), buf, nq, tg)
  runGrid(buildLpFwdProgram(L), buf, L.BT * H, L.BT * H)
  runGrid(buildLpBwdQProgram(L), buf, L.BT * H, L.BT * H)
  runGrid(buildLpBwdKVProgram(L), buf, L.BT * H, L.BT * H)
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
  chk("lp fwd y", buf[L.y ..< L.y + L.BT * C], yRef)
  chk("lp fwd invDen", buf[L.invDen ..< L.invDen + L.BT * H], iRef, 1e-3)
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
  chk("lp bwd dQ", buf[L.dQKV ..< L.dQKV + L.BT*3*C][0 .. 0], fd[0 .. 0], 1e-2) # sanity trivial
  var gq, gk, gv, fq, fk, fv: seq[float32]
  for t in 0 ..< L.BT:
    for c in 0 ..< C:
      gq.add buf[L.dQKV + t*3*C + c]; fq.add fd[t*3*C + c]
      gk.add buf[L.dQKV + t*3*C + C + c]; fk.add fd[t*3*C + C + c]
      gv.add buf[L.dQKV + t*3*C + 2*C + c]; fv.add fd[t*3*C + 2*C + c]
  chk("lp bwd dQ (FD)", gq, fq, 2e-3); chk("lp bwd dK (FD)", gk, fk, 2e-3); chk("lp bwd dV (FD)", gv, fv, 2e-3)
if bad == 0: echo "ALL OK" else: echo "CO LOI: ", bad; quit(1)
