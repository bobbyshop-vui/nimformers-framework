## Linear++ lane scan and LoRA-down on a real GPU (BYBY_INFER_BACKEND=metal|vulkan)
## against host references, at the Codestral sidecar/LoRA shapes.
import std/[os, math, strformat]
import backend, gpu_layers, mamba2_ssm_gpu, mamba2_full_vm, prefill_inference_kernels
import vendor/bybylang/gpubackend as byby

let ctx = newBackend(getEnv("BYBY_INFER_BACKEND", "metal"))
let be = ctx.kind.toByby()
ensureTgCap(ctx)
ensureDummy(be)

for (H, D, T) in [(4, 32, 17), (32, 128, 17), (8, 64, 40)]:
  let L = lpLayout(1, T, H, D)
  var seed = newSeq[float32](L.total)
  for i in 0 ..< T*3*L.C: seed[L.qkv+i] = sin(float32(i mod 991)*0.17'f32)*0.6'f32
  for i in 0 ..< T*2*L.C: seed[L.phi+i] = 0.3'f32 + abs(cos(float32(i mod 877)*0.11'f32))
  for i in 0 ..< L.C: seed[L.zF+i] = 0.5'f32 + float32(i mod 7)*0.1'f32
  for i in 0 ..< L.C*D: seed[L.cF+i] = sin(float32(i)*0.03'f32)*0.2'f32
  let prog = buildLpLaneProgram(L)
  let tag = "@test_lp_lanes_" & $H & "_" & $D & "_" & $T
  let kn = kernelNameFor(ctx, prog, tag)
  var mem = byby.sessionUpload(be, seed)
  var back = byby.sessionAllocScratch(be, L.total)
  doAssert byby.sessionBegin(be)
  encodeMicroVm(ctx, prog, mem, L.C*LaneScanWidth, lpLaneTgBytes(T), tag, LaneScanWidth, kn)
  doAssert byby.sessionCopyInto(be, mem, back, 0, 0, L.total)
  doAssert byby.sessionEnd(be), byby.sessionGetLastError(be)
  let got = byby.sessionRead(be, back, L.total)
  byby.sessionFree(be, back); byby.sessionFree(be, mem)
  var err, peak = 0'f32
  for h in 0 ..< H:
    var z = newSeq[float32](D); var S = newSeq[float32](D*D)
    for d in 0 ..< D: z[d] = seed[L.zF+h*D+d]
    for i in 0 ..< D*D: S[i] = seed[L.cF+h*D*D+i]
    for t in 0 ..< T:
      let row = L.phi + t*2*L.C + h*D
      var den = 0'f32
      for d in 0 ..< D:
        z[d] += seed[row+L.C+d]; den += seed[row+d]*z[d]
      for e in 0 ..< D:
        let v = seed[L.qkv+t*3*L.C+2*L.C+h*D+e]
        var num = 0'f32
        for d in 0 ..< D:
          S[e*D+d] += seed[row+L.C+d]*v; num += seed[row+d]*S[e*D+d]
        let y = num/(den+LpEps)
        err = max(err, abs(got[L.y+t*L.C+h*D+e]-y)); peak = max(peak, abs(y))
    for d in 0 ..< D: err = max(err, abs(got[L.zQ+h*D+d]-z[d]))
    for i in 0 ..< D*D: err = max(err, abs(got[L.cF+h*D*D+i]-S[i]))
  doAssert err <= max(2e-5'f32, peak*1e-4'f32), &"lp H={H} D={D} T={T} error={err}"
  echo &"PASS gpu lp lanes H={H} D={D} T={T} error={err:.3g}"

for (K, rank, T) in [(4096, 4, 17), (8192, 4, 17), (512, 16, 33)]:
  var x = newSeq[float32](T*K)
  var a = newSeq[float32](rank*K)
  for i in 0 ..< x.len: x[i] = sin(float32(i mod 613)*0.29'f32)
  for i in 0 ..< a.len: a[i] = cos(float32(i mod 487)*0.41'f32)*0.1'f32
  let kn = vmKernelForBe(buildPrefixLoraDownProgram(K, rank), "test_lora_down_" & $K & "_" & $rank, be)
  var hx = byby.sessionUpload(be, x)
  var ha = byby.sessionUpload(be, a)
  var hout = byby.sessionAllocScratch(be, T*rank)
  doAssert byby.sessionBegin(be)
  encodeVmKernel(kn, T*rank*LoraDownTg, LoraDownTg, LoraDownTg*4, hout, 0, gDummyFor(be), ha, hx, 0)
  doAssert byby.sessionEnd(be), byby.sessionGetLastError(be)
  let got = byby.sessionRead(be, hout, T*rank)
  var err = 0'f64
  for t in 0 ..< T:
    for j in 0 ..< rank:
      var s = 0'f64
      for k in 0 ..< K: s += float64(x[t*K+k])*float64(a[j*K+k])
      err = max(err, abs(s - float64(got[t*rank+j])))
  doAssert err < 2e-3, &"lora K={K} r={rank} T={T} error={err}"
  echo &"PASS gpu lora down K={K} rank={rank} T={T} error={err:.3g}"
