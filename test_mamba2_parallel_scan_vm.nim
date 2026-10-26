## CPU interpreter parity check for the timestep-parallel Mamba2 affine scan.
## This does not need Metal or a GPU; it validates padding, threadgroup
## barriers, affine composition order, and the exact state layout used by BPTT.

import std/[math, strformat]
import mamba2_full_vm
import vm_interp

proc verifyScan(T: int) =
  const B = 2
  const H = 4
  const D = 2
  const N = 3
  const G = 2
  const C = H * D
  const K = 3
  let L = m2Layout(B, T, H, D, N, G, K, C, 1e-5'f32, G)
  var serial = newSeq[float32](L.total)
  for b in 0 ..< B:
    for t in 0 ..< T:
      for c in 0 ..< L.cDim:
        let i = L.conv + (b * T + t) * L.cDim + c
        serial[i] = float32(sin(float64((b + 1) * 17 + (t + 2) * 7 + c) * 0.13) * 0.4)
      for h in 0 ..< H:
        let i = L.dt + (b * T + t) * H + h
        serial[i] = 0.02'f32 + float32((b * 3 + t * 5 + h) mod 11) * 0.017'f32
  for h in 0 ..< H:
    serial[L.aLog + h] = float32((h mod 3) - 1) * 0.2'f32
    serial[L.dSkip + h] = 0.25'f32 + float32(h) * 0.1'f32
  var parallel = serial
  runGrid(buildM2ScanProgram(L), serial, B * L.HD, tgSize = 4)
  let tg = m2ScanTgSize(T)
  doAssert tg > 0
  runGrid(buildM2ScanParallelProgram(L, tg), parallel,
    B * L.HD * tg, tgSize = tg, tgFloats = 4 * tg)
  var yErr = 0'f32
  for i in 0 ..< B * T * C:
    let e = abs(serial[L.yPre + i] - parallel[L.yPre + i])
    if e > yErr: yErr = e
  var stateErr = 0'f32
  for i in 0 ..< B * (T + 1) * L.HD * N:
    stateErr = max(stateErr, abs(serial[L.states + i] - parallel[L.states + i]))
  if yErr > 2e-5'f32 or stateErr > 2e-5'f32:
    raise newException(AssertionDefect,
      &"parallel Mamba2 scan T={T} mismatch: y={yErr:.8g}, state={stateErr:.8g}")
  echo &"PASS T={T} paddedTG={tg} yErr={yErr:.3g} stateErr={stateErr:.3g}"

when isMainModule:
  for T in [2, 3, 7, 8, 128]:
    verifyScan(T)
