## CPU-interpreter parity test for the time-parallel Linear++ scan.
## Compares the complete forward result and terminal recurrent state against
## the existing serial recurrence, including non-power-of-two T and D chunks.

import std/[math, strformat]
import mamba2_full_vm
import fp4_gemv_vm
import vm_interp

proc maxDiff(a, b: openArray[float32]; offA, offB, count: int): float32 =
  for i in 0 ..< count:
    result = max(result, abs(a[offA + i] - b[offB + i]))

proc runLpBackward(buf: var seq[float32]; L: LpLay) =
  runGrid(buildLpDDenProgram(L), buf, L.BT * L.H, 1)
  runGrid(buildLpBwdQProgram(L), buf, L.B * L.C, 1)
  runGrid(buildLpBwdKProgram(L), buf, L.B * L.C, 1)
  runGrid(buildLpBwdVProgram(L), buf, L.B * L.C, 1)

proc verifyLinearPlusScan(T: int) =
  const B = 2
  const H = 2
  const D = 5
  let L = lpLayout(B, T, H, D)
  let tg = lpScanTgSize(T)
  doAssert tg > 0
  let chunk = 2 # Exercise multiple D chunks and the final partial chunk.
  let nChunks = (D + chunk - 1) div chunk
  var serial = newSeq[float32](L.total)
  var parallel = newSeq[float32](L.total)
  for b in 0 ..< B:
    for t in 0 ..< T:
      for c in 0 ..< 3 * L.C:
        let i = L.qkv + (b * T + t) * 3 * L.C + c
        serial[i] = float32(sin(float64((b + 1) * 19 + (t + 3) * 11 + c * 7) * 0.071) * 0.8)
  for i in 0 ..< L.BT * L.C:
    serial[L.dY + i] = float32(cos(float64(i * 13 + 5) * 0.037) * 0.6)
  for i in 0 ..< L.total: parallel[i] = serial[i]

  let phiTg = lpPhiTg(L)
  runGrid(buildLpPhiProgram(L), serial, L.BT * 2 * L.C, phiTg)
  runGrid(buildLpDenProgram(L), serial, B * H, 1)
  runGrid(buildLpFwdProgram(L), serial, B * L.C, 1)

  runGrid(buildLpDenScanParallelProgram(L, tg), parallel,
    B * H * D * tg, tg, tgFloats = 2 * tg)
  runGrid(buildLpStateScanParallelProgram(L, tg, chunk), parallel,
    B * H * D * nChunks * tg, tg, tgFloats = 2 * tg * chunk)
  runGrid(buildLpFinalizeProgram(L), parallel, L.BT * L.C, 1)

  let yErr = maxDiff(serial, parallel, L.y, L.y, L.BT * L.C)
  let phiErr = maxDiff(serial, parallel, L.phi, L.phi, L.BT * 2 * L.C)
  let denErr = maxDiff(serial, parallel, L.invDen, L.invDen, L.BT * H)
  let zErr = maxDiff(serial, parallel, L.zF, L.zF, B * L.C)
  let stateErr = maxDiff(serial, parallel, L.cF, L.cF, B * L.C * D)
  runLpBackward(serial, L)
  runLpBackward(parallel, L)
  let gradErr = maxDiff(serial, parallel, L.dQKV, L.dQKV, L.BT * 3 * L.C)
  if yErr > 4e-5'f32 or phiErr > 4e-5'f32 or denErr > 4e-5'f32 or
      zErr > 4e-5'f32 or stateErr > 4e-5'f32 or gradErr > 2e-4'f32:
    raise newException(AssertionDefect,
      &"Linear++ parallel scan T={T} mismatch: y={yErr:.8g}, invDen={denErr:.8g}, " &
      &"phi={phiErr:.8g}, zF={zErr:.8g}, cF={stateErr:.8g}, grad={gradErr:.8g}; " &
      &"sample y={serial[L.y]:.6g}/{parallel[L.y]:.6g}, " &
      &"inv={serial[L.invDen]:.6g}/{parallel[L.invDen]:.6g}, " &
      &"z={serial[L.zF]:.6g}/{parallel[L.zF]:.6g}, " &
      &"c={serial[L.cF]:.6g}/{parallel[L.cF]:.6g}")
  echo &"PASS Linear++ T={T} paddedTG={tg} y={yErr:.3g} invDen={denErr:.3g} " &
       &"zF={zErr:.3g} cF={stateErr:.3g} grad={gradErr:.3g}"

proc verifyTrainScanAgainstInferenceStep(T: int) =
  ## Compare exactly the two production algorithms: stateless parallel scan
  ## used by finetune versus repeated recurrent one-token steps used by infer.
  const B = 1
  const H = 2
  const D = 5
  let L = mamba2_full_vm.lpLayout(B, T, H, D)
  let stepL = fp4_gemv_vm.lpLayout(L.C, H)
  let stepProgram = buildLinearPlusStepProgram(stepL, LpEps)
  let tg = lpScanTgSize(T)
  let chunk = 2
  let nChunks = (D + chunk - 1) div chunk
  var parallel = newSeq[float32](L.total)
  for t in 0 ..< T:
    for c in 0 ..< 3 * L.C:
      let i = L.qkv + t * 3 * L.C + c
      parallel[i] = float32(cos(float64((t + 2) * 17 + c * 5) * 0.053) * 0.7)

  let phiTg = lpPhiTg(L)
  runGrid(buildLpPhiProgram(L), parallel, L.BT * 2 * L.C, phiTg)
  runGrid(buildLpDenScanParallelProgram(L, tg), parallel,
    B * H * D * tg, tg, tgFloats = 2 * tg)
  runGrid(buildLpStateScanParallelProgram(L, tg, chunk), parallel,
    B * H * D * nChunks * tg, tg, tgFloats = 2 * tg * chunk)
  runGrid(buildLpFinalizeProgram(L), parallel, L.BT * L.C, 1)

  var recurrent = newSeq[float32](stepL.total) # z/S begin at zero, same as a new prompt.
  var yErr = 0'f32
  for t in 0 ..< T:
    for c in 0 ..< 3 * L.C:
      recurrent[stepL.offProj + c] = parallel[L.qkv + t * 3 * L.C + c]
    runGrid(stepProgram, recurrent, H * D, D, tgFloats = 4)
    for c in 0 ..< L.C:
      yErr = max(yErr, abs(recurrent[stepL.offY + c] - parallel[L.y + t * L.C + c]))

  var zErr = 0'f32
  for i in 0 ..< B * L.C:
    zErr = max(zErr, abs(recurrent[stepL.offZ + i] - parallel[L.zF + i]))
  var stateErr = 0'f32
  # Training scan stores the state transposed as [head, value, key] for its
  # output-channel-parallel reduction; the recurrent decode kernel stores
  # [head, key, value]. Compare equivalent matrix entries, not raw offsets.
  for h in 0 ..< H:
    for dKey in 0 ..< D:
      for dValue in 0 ..< D:
        let inferIdx = stepL.offS + h * D * D + dKey * D + dValue
        let trainIdx = L.cF + h * D * D + dValue * D + dKey
        stateErr = max(stateErr, abs(recurrent[inferIdx] - parallel[trainIdx]))
  if yErr > 2e-4'f32 or zErr > 2e-4'f32 or stateErr > 2e-4'f32:
    raise newException(AssertionDefect,
      &"Linear++ train/infer mismatch T={T}: y={yErr:.8g}, z={zErr:.8g}, S={stateErr:.8g}")
  echo &"PASS Linear++ train scan vs inference recurrence T={T}: " &
       &"y={yErr:.3g} z={zErr:.3g} S={stateErr:.3g}"

when isMainModule:
  for T in [1, 2, 3, 7, 8]:
    verifyLinearPlusScan(T)
    verifyTrainScanAgainstInferenceStep(T)
