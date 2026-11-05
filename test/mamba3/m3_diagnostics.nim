## Small isolated GPU diagnostics for Mamba-3 sidecar kernels.
import std/[math, strformat]
import backend
import session_vm
import m3_inference_kernels
import gpu_layers
import vendor/bybylang/gpubackend as byby

proc eluPlusOne(x: float32): float32 =
  if x < 0'f32: exp(x) else: x + 1'f32

proc linearPlusSelfCheck*(ctx: Backend): string =
  ## Dispatch the production scan kernel against a deterministic CPU reference.
  ## All buffers are private scratch; the live Mamba-3 runtime is untouched.
  let be = ctx.kind.toByby()
  if be notin {byby.gbMetal, byby.gbVulkan}:
    return "[lpcheck] requires a Metal or Vulkan backend"

  const HIDDEN = 64
  const HEADS = 2
  const T = 2
  const D = HIDDEN div HEADS
  let a = arenaLayout(HIDDEN, HIDDEN, HEADS, 8, 16, T, 1)
  let s = stateLayout(a, a.total, HEADS)
  var host = newSeq[float32](s.total)
  # Distinct, bounded nonzero initial S/Z values and Q/K/gate values.
  for h in 0..<HEADS:
    for k in 0..<D:
      host[s.lpZ + h*D + k] = 0.15'f32 + float32((h*D+k) mod 11)*0.013'f32
      for q in 0..<D:
        host[s.lpS + h*D*D + k*D + q] = float32(((h*17+k*3+q*5) mod 19)-9)*0.007'f32
  for t in 0..<T:
    for h in 0..<HEADS:
      for i in 0..<D:
        let row = a.attnProj + t*3*HIDDEN + h*D + i
        host[row] = float32(((t*7+h*5+i*3) mod 17)-8)*0.025'f32
        host[row + HIDDEN] = float32(((t*11+h*3+i*7) mod 19)-9)*0.021'f32
        host[row + 2*HIDDEN] = float32(((t*13+h*7+i*2) mod 15)-7)*0.018'f32

  var expected = host
  var expectedOut = newSeq[float32](T*HIDDEN)
  for t in 0..<T:
    for h in 0..<HEADS:
      for k in 0..<D:
        let row = a.attnProj + t*3*HIDDEN + h*D + k
        let key = eluPlusOne(host[row + HIDDEN])
        let zi = s.lpZ + h*D + k
        expected[zi] = expected[zi] + key
      for q in 0..<D:
        let gate = host[a.attnProj + t*3*HIDDEN + 2*HIDDEN + h*D + q]
        var numerator, denominator = 0'f32
        for k in 0..<D:
          let row = a.attnProj + t*3*HIDDEN + h*D + k
          let key = eluPlusOne(host[row + HIDDEN])
          let query = eluPlusOne(host[row])
          let si = s.lpS + h*D*D + k*D + q
          let zi = s.lpZ + h*D + k
          expected[si] = key*gate + expected[si]
          numerator += query*expected[si]
          denominator += query*expected[zi]
        expectedOut[t*HIDDEN + h*D + q] = numerator/(denominator+1e-6'f32)

  var scratch = byby.sessionUploadPrivate(be, host)
  var opened = false
  try:
    let kernel = vmKernelForBe(linearPlusScan(a, s, HEADS, T),
      "m3_lpcheck_64_2_2", be)
    let carry = vmKernelForBe(carryLinearPlus(a, s),
      "m3_lpcheck_carry_64_2_2", be)
    ensureDummy(be)
    if not sessionBegin(be):
      raise newException(CatchableError, "sessionBegin failed: " & sessionGetLastError(be))
    opened = true
    encodeVmKernel(kernel, HIDDEN*32, 32, 128, scratch, 0,
      gDummyFor(be), gDummyFor(be), scratch, 0)
    encodeVmKernel(carry, HIDDEN, 128, 16, scratch, 0,
      gDummyFor(be), gDummyFor(be), scratch, 0)
    if not sessionEnd(be):
      opened = false
      raise newException(CatchableError, "sessionEnd failed: " & sessionGetLastError(be))
    opened = false
    let got = sessionRead(be, scratch, s.total)
    var outErr, stateErr = 0'f32
    for t in 0..<T:
      for i in 0..<HIDDEN:
        outErr = max(outErr, abs(got[a.attnInner+t*HIDDEN+i] - expectedOut[t*HIDDEN+i]))
    for i in s.lpS..<s.lpZ+HEADS*D:
      stateErr = max(stateErr, abs(got[i]-expected[i]))
    result = &"[lpcheck] T={T} hidden={HIDDEN} heads={HEADS} max_output_error={outErr:.6g} max_state_error={stateErr:.6g}"
  except CatchableError as e:
    result = "[lpcheck] " & e.msg
  finally:
    if opened: discard sessionEnd(be)
    sessionFree(be, scratch)
