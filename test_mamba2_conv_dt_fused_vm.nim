## Verify that independent conv/SiLU and dt/softplus bytecode bodies can be
## safely combined into one physical VM dispatch with their shared launch grid.

import std/[math, strformat]
import mamba2_full_vm
import vm_interp

proc maxDiff(a, b: openArray[float32]; off, n: int): float32 =
  for i in 0 ..< n:
    result = max(result, abs(a[off + i] - b[off + i]))

proc verify(T: int) =
  const B = 1
  const H = 2
  const D = 2
  const N = 2
  const G = 1
  const C = H * D
  const K = 3
  let L = m2Layout(B, T, H, D, N, G, K, C, 1e-5'f32, G)
  var separate = newSeq[float32](L.total)
  var fused = newSeq[float32](L.total)
  for i in 0 ..< L.BT * L.P:
    separate[L.proj + i] = float32(sin(float64(i * 17 + 3) * 0.073) * 0.7)
  for i in 0 ..< L.cDim * K:
    separate[L.convW + i] = float32(cos(float64(i * 11 + 2) * 0.091) * 0.3)
  for i in 0 ..< L.cDim:
    separate[L.convB + i] = float32(i mod 5 - 2) * 0.04'f32
  for i in 0 ..< H:
    separate[L.dtBias + i] = float32(i + 1) * 0.03'f32
  for i in 0 ..< L.total: fused[i] = separate[i]

  let nThreads = L.BT * M2Tg
  runGrid(buildM2ConvSiluProgram(L), separate, nThreads, M2Tg)
  runGrid(buildM2DtProgram(L), separate, nThreads, M2Tg)
  runGrid(buildM2ConvDtProgram(L), fused, nThreads, M2Tg)
  let convErr = maxDiff(separate, fused, L.conv, L.BT * L.cDim)
  let preErr = maxDiff(separate, fused, L.pre, L.BT * L.cDim)
  let dtErr = maxDiff(separate, fused, L.dt, L.BT * H)
  if convErr > 1e-7'f32 or preErr > 1e-7'f32 or dtErr > 1e-7'f32:
    raise newException(AssertionDefect,
      &"fused conv+dt dispatch T={T} mismatch: conv={convErr:.8g}, " &
      &"pre={preErr:.8g}, dt={dtErr:.8g}")
  echo &"PASS fused conv+dt T={T}: conv={convErr:.3g}, pre={preErr:.3g}, dt={dtErr:.3g}"

when isMainModule:
  for T in [1, 3, 7]: verify(T)
