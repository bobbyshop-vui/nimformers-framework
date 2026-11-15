## GPU check: the simdgroup threadgroup reduction (emitTgSimdReduce) returns the same total as
## the barrier tree for every power-of-two threadgroup size, on whatever simdgroup width the
## device has, and leaves the integer registers of the caller alone (rmsRows keeps its gamma
## offset in U30 across the reduction; an earlier version overwrote U28..U30 and corrupted every
## norm of the Mamba-3 model without this test noticing).
import std/[math, os, strformat]
import backend
import mamba2_ssm_gpu
import mamba2_vm_kernels
import session_vm

const Sentinel = [(20'u32, 4444'u32), (28'u32, 1111'u32), (29'u32, 2222'u32), (30'u32, 3333'u32)]

proc reduceProgram(n: int): VmProgram =
  var p: VmProgram
  p.emitULid(0); p.emitUToR(1, 0)                    # R1 = float(lid)
  p.emitUMulK(2, 0, 1)                               # U2 = lid
  for (reg, value) in Sentinel: p.emitUConst(reg, value)   # live across the reduction
  p.emitULoadF32(5, 2, 0)                            # R5 = in[lid]
  p.emitTgStoreF32(5, 1, 0)
  p.emitBarrierTg()
  p.emitTgTreeReduce(1, n, 4, 5, 6, 10, 11)
  p.emitLoadConstF32(9, 0)
  p.emitTgLoadF32(10, 9, 0)
  p.emitUStoreF32(10, 2, uint32(n))                  # out[n + lid] = total
  p.emitUMulK(3, 0, 4)                               # U3 = 4 * lid
  for k, (reg, value) in Sentinel:
    p.emitUToR(12, reg)
    p.emitUStoreF32(12, 3, uint32(2 * n + k))        # the sentinel the register still holds
  p.emitHalt()
  p

when isMainModule:
  let ctx = newBackend("metal")
  for mode in ["1", "0"]:                            # 1 = simdgroup reduction, 0 = barrier tree
    putEnv("BYBY_SIMD_REDUCE", mode)
    for n in [8, 32, 64, 128, 256]:
      var mem = newSeq[float32](6 * n)
      var want = 0.0
      for i in 0 ..< n:
        mem[i] = float32(sin(float(i) * 0.37) + 0.25)
        want += float(mem[i])
      doAssert runMicroVm(ctx, reduceProgram(n), mem, n, tgBytes = max(16, 4 * n),
        tag = "tg_simd_reduce_" & mode & "_" & $n, tgSize = n)
      var err = 0.0
      for i in 0 ..< n: err = max(err, abs(float(mem[n + i]) - want))
      doAssert err < 1e-4, &"mode={mode} n={n} err={err}"
      for i in 0 ..< n:
        for k, (reg, value) in Sentinel:
          doAssert mem[2 * n + 4 * i + k] == float32(value),
            &"mode={mode} n={n}: U{reg} was clobbered in lane {i} ({mem[2 * n + 4 * i + k]})"
      echo &"PASS tg reduce mode={mode} n={n} err={err:.2e}, U20/U28/U29/U30 intact"
