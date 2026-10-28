## GPU differential check: Vulkan/Metal micro-VM parallel Mamba2 scan against
## the serial recurrence, including the complete recurrent-state layout.

import std/[math, strformat]
import backend
import lora
import mamba2_full_vm
import mamba2_ssm_gpu
import vm_interp

proc verifyOnGpu(ctx: Backend; T: int) =
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

  var gpuSerial = serial
  var gpuParallel = serial
  var refSerial = serial
  var refParallel = serial
  runGrid(buildM2ScanProgram(L), refSerial, B * L.HD, tgSize = 4)
  let tg = m2ScanTgSize(T)
  runGrid(buildM2ScanParallelProgram(L, tg), refParallel,
    B * L.HD * tg, tgSize = tg, tgFloats = 4 * tg)
  doAssert runMicroVm(ctx, buildM2ScanProgram(L), gpuSerial, B * L.HD,
    tgBytes = 4, tag = "m2_scan_serial_gpu_test")
  doAssert tg > 0
  doAssert runMicroVm(ctx, buildM2ScanParallelProgram(L, tg), gpuParallel,
    B * L.HD * tg, tgBytes = 4 * tg * 4,
    tag = "m2_scan_parallel_gpu_test", tgSize = tg)

  var yErr = 0'f32
  for i in 0 ..< B * T * C:
    yErr = max(yErr, abs(gpuSerial[L.yPre + i] - gpuParallel[L.yPre + i]))
  var stateErr = 0'f32
  for i in 0 ..< B * (T + 1) * L.HD * N:
    stateErr = max(stateErr,
      abs(gpuSerial[L.states + i] - gpuParallel[L.states + i]))
  if yErr > 2e-5'f32 or stateErr > 2e-5'f32:
    var serialVsCpuY = 0'f32
    var parallelVsCpuY = 0'f32
    var serialVsCpuState = 0'f32
    var parallelVsCpuState = 0'f32
    var cpuYErr = 0'f32
    var cpuStateErr = 0'f32
    var maxStateIndex = 0
    var maxGpuStateIndex = 0
    var maxGpuStateErr = 0'f32
    for i in 0 ..< B * T * C:
      serialVsCpuY = max(serialVsCpuY, abs(gpuSerial[L.yPre + i] - refSerial[L.yPre + i]))
      parallelVsCpuY = max(parallelVsCpuY, abs(gpuParallel[L.yPre + i] - refParallel[L.yPre + i]))
      cpuYErr = max(cpuYErr, abs(refSerial[L.yPre + i] - refParallel[L.yPre + i]))
    for i in 0 ..< B * (T + 1) * L.HD * N:
      serialVsCpuState = max(serialVsCpuState, abs(gpuSerial[L.states + i] - refSerial[L.states + i]))
      parallelVsCpuState = max(parallelVsCpuState, abs(gpuParallel[L.states + i] - refParallel[L.states + i]))
      let e = abs(refSerial[L.states + i] - refParallel[L.states + i])
      if e > cpuStateErr: cpuStateErr = e; maxStateIndex = i
      let ge = abs(gpuParallel[L.states + i] - refParallel[L.states + i])
      if ge > maxGpuStateErr: maxGpuStateErr = ge; maxGpuStateIndex = i
    stderr.writeLine("[diag] CPU worst state idx=" & $maxStateIndex &
      " serial/parallel=" & $refSerial[L.states + maxStateIndex] & "/" &
      $refParallel[L.states + maxStateIndex] & "; GPU worst idx=" & $maxGpuStateIndex &
      " serial/parallel=" & $gpuSerial[L.states + maxGpuStateIndex] & "/" &
      $gpuParallel[L.states + maxGpuStateIndex] & " CPUref=" &
      $refParallel[L.states + maxGpuStateIndex])
    for t in 0 .. T:
      let i = L.states + t * L.HD * N + 6 * N + 1
      stderr.writeLine("[diag] b0/t" & $t & " hd6/n1 gpu serial=" &
        $gpuSerial[i] & " gpu parallel=" & $gpuParallel[i] &
        " cpu=" & $refSerial[i])
    raise newException(AssertionDefect,
      &"GPU parallel Mamba2 scan T={T} mismatch vs serial: y={yErr:.8g}, state={stateErr:.8g}; " &
      &"GPU-vs-CPU serial y/state={serialVsCpuY:.8g}/{serialVsCpuState:.8g}, " &
      &"parallel y/state={parallelVsCpuY:.8g}/{parallelVsCpuState:.8g}; " &
      &"CPU parallel-vs-serial y/state={cpuYErr:.8g}/{cpuStateErr:.8g}")
  echo &"PASS GPU Mamba2 scan T={T} paddedTG={tg} yErr={yErr:.3g} stateErr={stateErr:.3g}"

when isMainModule:
  setForbidCpuFallback(true)
  let ctx = newBackend()
  ensureTgCap(ctx)
  initMicroVmJit("mamba2_scan_gpu_test", ctx.kind.toByby())
  verifyOnGpu(ctx, 7)
