## Linear++ lane scan against the serial den+fwd kernels (nonzero carried S/z,
## batch 2, multi-chunk T).
import std/[math, strformat]
import mamba2_full_vm, prefill_inference_kernels, vm_interp

for D in [32, 64, 128]:
  for T in [2, 15, 16, 17, 40, 128]:
    let L = lpLayout(2, T, 3, D)
    var seed = newSeq[float32](L.total)
    for i in 0 ..< L.BT*3*L.C: seed[L.qkv+i] = sin(float32(i mod 991)*0.17'f32)*0.6'f32
    for i in 0 ..< L.BT*2*L.C: seed[L.phi+i] = 0.3'f32 + abs(cos(float32(i mod 877)*0.11'f32))
    for i in 0 ..< L.B*L.C: seed[L.zF+i] = 0.5'f32 + float32(i mod 7)*0.1'f32
    for i in 0 ..< L.B*L.C*D: seed[L.cF+i] = sin(float32(i)*0.03'f32)*0.2'f32
    var serial = seed
    runGrid(buildLpDenProgram(L), serial, L.B*L.H, tgSize=1, tgFloats=1)
    runGrid(buildLpFwdProgram(L), serial, L.B*L.C, tgSize=1, tgFloats=1)
    var lanes = seed
    runGrid(buildLpLaneProgram(L), lanes, L.B*L.C*LaneScanWidth,
      tgSize=LaneScanWidth, tgFloats=lpLaneTgBytes(T) div 4)
    var err, peak = 0'f32
    # Lane kernel leaves zF as the initial state and writes terminal z to zQ.
    for (off, src, n) in [(L.y, L.y, L.BT*L.C), (L.zQ, L.zF, L.B*L.C), (L.cF, L.cF, L.B*L.C*D)]:
      for i in 0 ..< n:
        doAssert lanes[off+i] == lanes[off+i], &"NaN at {off+i}"
        err = max(err, abs(lanes[off+i]-serial[src+i])); peak = max(peak, abs(serial[src+i]))
    doAssert err <= max(2e-5'f32, peak*2e-6'f32), &"lp D={D} T={T} error={err} peak={peak}"
    echo &"PASS lp lanes D={D} T={T} error={err:.3g}"
