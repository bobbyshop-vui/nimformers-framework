## GPU differential test: parallel Linear++ scan versus the serial VM path.
## Deliberately avoids test_nimformer/Python dependencies so it can run as a
## small standalone Metal/CUDA validation.

import std/[math, strformat]
import backend
import lora
import mamba2_full_vm
import mamba2_ssm_gpu

proc maxDiff(a, b: openArray[float32]): float32 =
  doAssert a.len == b.len
  for i in 0 ..< a.len:
    result = max(result, abs(a[i] - b[i]))

when isMainModule:
  setForbidCpuFallback(true)
  let ctx = newBackend()
  ensureTgCap(ctx)
  initMicroVmJit("linear_plus_scan_gpu_test", ctx.kind.toByby())

  const B = 1
  const T = 7
  const H = 2
  const D = 5
  let L = lpLayout(B, T, H, D)
  var qkv = newSeq[float32](L.BT * 3 * L.C)
  for i in 0 ..< qkv.len:
    qkv[i] = float32(sin(float64(i * 23 + 7) * 0.043) * 0.9)

  let parallelY = linearPlusForwardGpu(ctx, qkv, B, T, H, D)
  var serialBuf = newSeq[float32](L.total)
  serialBuf.blit(L.qkv, qkv, 0, qkv.len)
  let phiTg = lpPhiTg(L)
  discard runMicroVmChain(ctx, [
    (buildLpPhiProgram(L), L.BT * 2 * L.C, phiTg * 4, phiTg, "lp_gpu_ref_phi"),
    (buildLpDenProgram(L), B * H, 4, 0, "lp_gpu_ref_den"),
    (buildLpFwdProgram(L), B * L.C, 4, 0, "lp_gpu_ref_fwd")], serialBuf)
  let serialY = serialBuf[L.y ..< L.y + L.BT * L.C]
  let err = maxDiff(parallelY, serialY)
  if err > 3e-5'f32:
    raise newException(AssertionDefect,
      &"GPU Linear++ parallel scan differs from serial VM recurrence: maxAbs={err:.8g}")
  echo &"PASS GPU Linear++ parallel vs serial (B={B}, T={T}, H={H}, D={D}, maxAbs={err:.3g})"
