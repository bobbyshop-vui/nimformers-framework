## Register lane scan on a real GPU (BYBY_INFER_BACKEND=metal|vulkan) against
## the host recurrence, including the full Codestral head shape.
import std/[os, math, strformat]
import backend, mamba2_ssm_gpu, mamba2_full_vm, prefill_inference_kernels
import vendor/bybylang/gpubackend as byby

let ctx = newBackend(getEnv("BYBY_INFER_BACKEND", "metal"))
let be = ctx.kind.toByby()
ensureTgCap(ctx)
for (H, D, N, G, C) in [(8, 4, 32, 2, 32), (8, 4, 128, 2, 32), (128, 64, 128, 8, 8192)]:
  for T in [2, 17, 33, 128]:
    if H == 128 and T > 17: continue
    let L = prefixLayout(1, T, H, D, N, G, 4, C, 1e-5'f32, G, true)
    var seed = newSeq[float32](L.fwdEnd)
    for i in 0 ..< T*L.cDim: seed[L.conv+i] = sin(float32(i mod 997)*0.13'f32)*0.4'f32
    for i in 0 ..< T*L.H: seed[L.dt+i] = 0.02'f32 + float32(i mod 13)*0.01'f32
    for h in 0 ..< L.H:
      seed[L.aLog+h] = float32(h mod 9)*0.1'f32
      seed[L.dSkip+h] = 0.5'f32
    for i in 0 ..< L.HD*L.N: seed[L.states+i] = cos(float32(i)*0.1'f32)
    let prog = buildPrefixLaneScanProgram(L)
    let tag = "@test_lanes_" & m2Tag(L)
    let kn = kernelNameFor(ctx, prog, tag)
    var mem = byby.sessionUpload(be, seed)
    let span = L.yPre + T*L.C - L.states
    var readback = byby.sessionAllocScratch(be, span)
    doAssert byby.sessionBegin(be)
    encodeMicroVm(ctx, prog, mem, L.HD*LaneScanWidth, laneScanTgBytes(T), tag,
      LaneScanWidth, kn)
    doAssert byby.sessionCopyInto(be, mem, readback, L.states, 0, span)
    doAssert byby.sessionEnd(be), byby.sessionGetLastError(be)
    let got = byby.sessionRead(be, readback, span)
    byby.sessionFree(be, readback); byby.sessionFree(be, mem)
    var err, peak = 0'f32
    var s = newSeq[float32](L.HD*L.N)
    for i in 0 ..< s.len: s[i] = seed[L.states+i]
    for t in 0 ..< T:
      for hd in 0 ..< L.HD:
        let h = hd div L.D
        let g = h div (L.H div L.G)
        let row = L.conv + t*L.cDim
        let dt = seed[L.dt+t*L.H+h]
        let dA = exp(dt * -exp(seed[L.aLog+h]))
        let x = seed[row+hd]
        var y = 0'f32
        for n in 0 ..< L.N:
          s[hd*L.N+n] = s[hd*L.N+n]*dA + dt*seed[row+L.C+g*L.N+n]*x
          y += seed[row+L.C+L.GN+g*L.N+n]*s[hd*L.N+n]
        let v = got[L.yPre-L.states+t*L.C+hd]
        doAssert v == v, &"NaN y t={t} hd={hd}"
        err = max(err, abs(v - (y + seed[L.dSkip+h]*x))); peak = max(peak, abs(v))
    for i in 0 ..< s.len:
      err = max(err, abs(got[i]-s[i])); err = max(err, abs(got[L.HD*L.N+i]-s[i]))
    let tol = max(2e-5'f32, peak*1e-4'f32)
    doAssert err <= tol, &"H={H} N={N} T={T}: error={err} tolerance={tol}"
    echo &"PASS gpu lane scan H={H} D={D} N={N} T={T} error={err:.3g}"
