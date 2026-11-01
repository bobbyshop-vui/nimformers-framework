## mamba2_vm_kernels.nim
## Builds 2 VmPrograms (session_vm.nim) that run the SSM recurrence + RMSNorm
## gate of Mamba2 100% on the GPU. Uses the new opcodes in session_vm.nim
## (EXP, NEG, SIGMOID, RSQRT, LOAD/STORE_F32_IDX, IF_BEGIN/END, CMP_LT...).
## Dispatched through session_vm_dispatch.nim (real metal_enc_*, not emulated).

import std/bitops
import session_vm


# ── Tree reduction in the threadgroup, 100% on the GPU ──────
## Before: `while stride >= 1:` on the HOST, log2(n) steps unrolled at build time.
## Now: 1 FOR_BEGIN/END loop on the GPU; the stride lives in a register and halves each
## iteration via MUL_CONST 0.5 (exact for powers of 2). The barrier is OUTSIDE the IF so
## every thread in the threadgroup reaches it (no deadlock).
## Registers: rStride (changes), rCnt (CONSTANT in the loop: Metal rereads it each iteration),
## rTmp (cmp then addr), rA, rB. idxReg = index within the threadgroup.
proc emitTgTreeReduce*(p: var VmProgram; idxReg: uint32; n: int;
                      rStride, rCnt, rTmp, rA, rB: uint32) =
  doAssert n >= 1 and (n and (n - 1)) == 0, "emitTgTreeReduce: n must be a power of 2"
  p.emitLoadConstF32(rCnt, float32(countTrailingZeroBits(n)))   # log2(n) steps
  p.emitLoadConstF32(rStride, float32(n div 2))
  sessionVmForGpu(p, rCnt):
    p.emitCmpLt(rTmp, idxReg, rStride)                # tid < stride
    sessionVmIfGpu(p, rTmp):
      p.emitAdd(rTmp, idxReg, rStride)                # rTmp = idx + stride
      p.emitTgLoadF32(rA, idxReg, 0)
      p.emitTgLoadF32(rB, rTmp, 0)
      p.emitAdd(rA, rA, rB)
      p.emitTgStoreF32(rA, idxReg, 0)
    p.emitBarrierTg()
    p.emitMulConst(rStride, rStride, 0.5'f32)         # stride /= 2

# ════════════════════════════════════════════════════════════
# 1) Mamba2 SSM step — grid = H*D threads, the n loop runs on the GPU.
# Uses exactly 16 registers, no overflow, no spill.
# The h/d/g index tables are built once at model load (constant for the model's lifetime).
#
# memF32 layout:
#   offX     : xPart[HD]
#   offB     : bMat[G*N]
#   offC     : cMat[G*N]
#   offDt    : dt[H]        (already softplus'd, computed with the existing sessionActivation)
#   offALog  : aLog[H]
#   offDSkip : dSkip[H]
#   offState : ssmState[HD*N]   (read + written in place, persistent)
#   offY     : yPre[HD]         (output)
#   offIdxH/D/G : const index tables, built once
# ════════════════════════════════════════════════════════════
proc buildMamba2StepProgram*(H, D, N, G: int): VmProgram =
  let HD = H * D
  let offX      = 0
  let offB      = HD
  let offC      = offB + G*N
  let offDt     = offC + G*N
  let offALog   = offDt + H
  let offDSkip  = offALog + H
  let offState  = offDSkip + H
  let offY      = offState + HD*N
  # Garbage-token fix (bug #11): this used to declare an EXTRA table
  # `offIdxD` BETWEEN idxH and idxG:
  #     let offIdxD = offIdxH + HD
  #     let offIdxG = offIdxD + HD
  # But buildShapeInfo() in mamba2_ssm_gpu.nim uses ONLY 2 tables
  # (idxH, idxG) ADJACENT, and totalLen = offIdxG + HD. Result:
  # the bytecode read `g` at offset offY+3*HD while the host wrote `g` at
  # offY+2*HD - off by exactly HD elements, and offY+3*HD EQUALS totalLen,
  # i.e. it read OUTSIDE the host-allocated data. On the hot path hScratch is
  # a large shared buffer so it did not crash: nobody had written there (0) ->
  # g = 0 for ALL 8192 threads, EVERY block, EVERY token -> all 128 heads took
  # group 0's B and C. Measured at real scale (H=128 D=64 N=128 G=8):
  # rel-L2 of y = 0.62, of ssmState (carried to the next token) = 0.20,
  # 7168/8192 threads on the wrong group. The idxD table is NOT used anywhere
  # in this program (x comes from offX+tid, stateBase = tid*N)
  # so it is removed outright, not replaced.
  let offIdxH   = offY + HD
  let offIdxG   = offIdxH + HD

  var p: VmProgram
  # reserved for the whole function: R0=tid R1=h R3=g R4=yAcc R9=dA R10=xVal R11=dt
  # scratch in the n loop: R12,R13,R14,R15 (4 registers, enough)
  p.emitAddConst(12, 0, float32(offIdxH)); p.emitLoadF32Idx(1, 12)   # R1 = h
  p.emitAddConst(12, 0, float32(offIdxG)); p.emitLoadF32Idx(3, 12)   # R3 = g

  p.emitAddConst(12, 1, float32(offALog)); p.emitLoadF32Idx(12, 12)
  p.emitExp(12, 12); p.emitNeg(12, 12)                                # R12 = A_h (temp)
  p.emitAddConst(11, 1, float32(offDt));   p.emitLoadF32Idx(11, 11)   # R11 = dt[h]  [long-lived]
  p.emitMul(9, 11, 12); p.emitExp(9, 9)                               # R9 = dA      [long-lived]

  p.emitLoadF32Idx(10, 0)                                             # R10 = xVal   [long-lived]  (offX=0)

  p.emitAddConst(12, 1, float32(offDSkip)); p.emitLoadF32Idx(12, 12)
  p.emitMul(4, 12, 10)                                                # R4 = yAcc init [long-lived]

  p.emitMulConst(2, 3, float32(N))          # R2 = gN         [long-lived]
  p.emitMulConst(6, 0, float32(N))          # R6 = stateBase  [long-lived]
  p.emitLoadConstF32(7, 0.0'f32)             # R7 = n          [long-lived]
  p.emitLoadConstF32(8, float32(N))          # R8 = count FOR  [long-lived]

  sessionVmForGpu(p, 8):
    p.emitAddConst(13, 2, float32(offB)); p.emitAdd(13, 13, 7); p.emitLoadF32Idx(13, 13)  # R13 = bVal
    p.emitAddConst(14, 2, float32(offC)); p.emitAdd(14, 14, 7); p.emitLoadF32Idx(14, 14)  # R14 = cVal
    p.emitAddConst(15, 6, float32(offState)); p.emitAdd(15, 15, 7)                        # R15 = stateIdx
    p.emitLoadF32Idx(12, 15)              # R12 = sOld
    p.emitMul(12, 12, 9)                  # R12 = sOld*dA
    p.emitMul(13, 13, 10)                 # R13 = bVal*xVal  (overwrites, bVal no longer needed)
    p.emitFma(12, 11, 13, 12)             # R12 = sNew = dt*bVal*xVal + sOld*dA
    p.emitStoreF32Idx(12, 15)             # ssmState[stateIdx] = sNew
    p.emitFma(4, 14, 12, 4)               # yAcc += cVal*sNew
    p.emitAddConst(7, 7, 1.0'f32)         # n += 1

  p.emitAddConst(13, 0, float32(offY))
  p.emitStoreF32Idx(4, 13)                # yPre[tid] = yAcc
  p.emitHalt()
  result = p

# ════════════════════════════════════════════════════════════
# 2) Final RMSNorm + gate — 1 threadgroup / token, C threads.
# gated = yPre*silu(z) ; out = gated * rsqrt(mean(gated^2)+eps) * w
# Tree reduction in the threadgroup with IF_BEGIN/END + CMP_LT.
# Requires C to be a power of 2 and <= the device's max threads/threadgroup
# (usually 1024) — a larger C needs several threadgroups + summing partial
# sums with a helper kernel, NOT done here (stated plainly).
# memF32: [ yPre(0..<C) | z(C..<2C) | w(2C..<3C) | out(3C..<4C) ]
# ════════════════════════════════════════════════════════════
proc buildRmsGateProgram*(C: int; eps: float32): VmProgram =
  doAssert (C and (C-1)) == 0, "buildRmsGateProgram: C must be a power of 2 (pad if needed)"
  let offZ = C
  let offW = 2*C
  let offOut = 3*C

  var p: VmProgram
  # R0=tid  R1=gated  [long-lived]
  p.emitLoadF32Idx(1, 0)                              # R1 = yPre[tid]
  p.emitAddConst(2, 0, float32(offZ)); p.emitLoadF32Idx(2, 2)   # R2 = z[tid]
  p.emitNeg(3, 2)                                     # R3 = -z
  p.emitExp(3, 3); p.emitAddConst(3, 3, 1.0'f32)       # R3 = 1+exp(-z)
  p.emitLoadConstF32(5, 1.0'f32)
  p.emitDiv(3, 5, 3)                                   # R3 = sigmoid(z)
  p.emitMul(3, 2, 3)                                   # R3 = silu(z)
  p.emitMul(1, 1, 3)                                   # R1 = gated  [long-lived]

  p.emitMul(4, 1, 1)                                   # R4 = gated^2
  p.emitTgStoreF32(4, 0, 0)
  p.emitBarrierTg()

  p.emitTgTreeReduce(0, C, 6, 7, 8, 9, 10)   # GPU loop, no more host while

  p.emitLoadConstF32(12, 0.0'f32)
  p.emitTgLoadF32(11, 12, 0)         # R11 = tg[0] = sum of gated^2
  p.emitMulConst(11, 11, 1.0'f32/float32(C))
  p.emitAddConst(11, 11, eps)
  p.emitRsqrt(11, 11)                # R11 = rsqrt(mean+eps)

  p.emitAddConst(13, 0, float32(offW)); p.emitLoadF32Idx(13, 13)  # R13 = w[tid]
  p.emitMul(14, 1, 11)               # gated * rsqrt(...)
  p.emitMul(14, 14, 13)              # * w[tid]

  p.emitAddConst(15, 0, float32(offOut))
  p.emitStoreF32Idx(14, 15)
  p.emitHalt()
  result = p

# ════════════════════════════════════════════════════════════
# 3) Incremental conv1d (typically K=4) — grid = cDim threads.
# memF32 layout:
#   offXBC   : xBCPre[cDim]          (new input)
#   offState : convState[cDim*(K-1)] (read + written in place)
#   offW     : convWeight[cDim*K]    (row-major c*K + k)
#   offBias  : convBias[cDim]
#   offOut   : xBCConv[cDim]         (output after silu)
# ════════════════════════════════════════════════════════════
proc buildConv1dIncProgram*(cDim, K: int): VmProgram =
  ## Layout memF32:
  ##   [0, cDim)                 xBCPre
  ##   [cDim, cDim+cDim*(K-1))   convState  index = c*(K-1)+k
  ##   [.. + cDim*K)             convWeight index = c*K+k
  ##   [.. + cDim)               convBias
  ##   [.. + cDim)               out (silu)
  doAssert K >= 2 and K <= 8, "buildConv1dIncProgram: K in [2,8]"
  let offState = cDim
  let offW     = offState + cDim*(K-1)
  let offBias  = offW + cDim*K
  let offOut   = offBias + cDim

  var p: VmProgram
  # R0 = tid = c
  # R1 = acc = bias[c]
  p.emitAddConst(1, 0, float32(offBias))
  p.emitLoadF32Idx(1, 1)

  p.emitLoadConstF32(5, 0.0'f32)                 # k (counted on the GPU)
  p.emitLoadConstF32(6, float32(K-1))     # iteration count
  sessionVmForGpu(p, 6):
    # state addr = offState + tid*(K-1) + k
    p.emitMulConst(3, 0, float32(K-1))
    p.emitAdd(3, 3, 5); p.emitAddConst(3, 3, float32(offState))
    p.emitLoadF32Idx(2, 3)
    # weight addr = offW + tid*K + k
    p.emitMulConst(4, 0, float32(K))
    p.emitAdd(4, 4, 5); p.emitAddConst(4, 4, float32(offW))
    p.emitLoadF32Idx(4, 4)
    p.emitFma(1, 4, 2, 1)
    p.emitAddConst(5, 5, 1.0'f32)            # k += 1

  # xBCPre[c] * weight[..., K-1]
  p.emitLoadF32Idx(2, 0)   # R2 = xBCPre[tid] (offXBC=0)
  p.emitMulConst(4, 0, float32(K))
  p.emitAddConst(4, 4, float32((K-1) + offW))
  p.emitLoadF32Idx(4, 4)
  p.emitFma(1, 4, 2, 1)

  # silu(acc) = acc * sigmoid(acc)
  p.emitNeg(3, 1)
  p.emitExp(3, 3)
  p.emitAddConst(3, 3, 1.0'f32)
  p.emitLoadConstF32(5, 1.0'f32)
  p.emitDiv(3, 5, 3)
  p.emitMul(1, 1, 3)

  p.emitAddConst(6, 0, float32(offOut))
  p.emitStoreF32Idx(1, 6)

  # slide state
  p.emitLoadConstF32(5, 0.0'f32)                 # k (counted on the GPU)
  p.emitLoadConstF32(6, float32(K-2))     # iteration count
  sessionVmForGpu(p, 6):
    p.emitMulConst(3, 0, float32(K-1))
    p.emitAdd(3, 3, 5); p.emitAddConst(3, 3, float32(1 + offState))
    p.emitLoadF32Idx(2, 3)
    p.emitMulConst(4, 0, float32(K-1))
    p.emitAdd(4, 4, 5); p.emitAddConst(4, 4, float32(offState))
    p.emitStoreF32Idx(2, 4)
    p.emitAddConst(5, 5, 1.0'f32)            # k += 1
  # last = xBCPre
  p.emitMulConst(3, 0, float32(K-1))
  p.emitAddConst(3, 3, float32((K-2) + offState))
  p.emitLoadF32Idx(2, 0)
  p.emitStoreF32Idx(2, 3)

  p.emitHalt()
  result = p

# ════════════════════════════════════════════════════════════
# 4) Softplus(dt) + split helper — grid = H threads (dt) or full.
# Simple: softplus on dtPre + dtBias → dt
# mem: [dtPre(0..<H) | dtBias(H..<2H) | dtOut(2H..<3H)]
# ════════════════════════════════════════════════════════════
proc buildSoftplusDtProgram*(H: int): VmProgram =
  var p: VmProgram
  p.emitLoadF32Idx(1, 0)                              # R1 = dtPre[tid]
  p.emitAddConst(2, 0, float32(H)); p.emitLoadF32Idx(2, 2)  # R2 = bias
  p.emitAdd(1, 1, 2)
  p.emitSoftplus(1, 1)
  p.emitAddConst(3, 0, float32(2*H)); p.emitStoreF32Idx(1, 3)
  p.emitHalt()
  result = p

# ════════════════════════════════════════════════════════════
# 5) RMS+gate for large C (chunked): each threadgroup handles 1 chunk
# of size TG=1024 (power of 2). Phase 1: partial sum per chunk
# into memF32[offPartial + chunkId]. Phase 2 (grid=1 TG): reduce
# partials → rsqrt → broadcast scale. Phase 3: scale * gated * w.
# For simplicity and correctness, 2 separate programs + 1 small dispatch.
# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# 6) Split proj [2C+2GN+H] → z[C] | xBC[cDim] | dt[H]
# grid = max(C, cDim, H) threads; each writes only if tid < len
# mem: [ proj(0..<P) | z(P..<P+C) | xBC(P+C..<P+C+cDim) | dt(P+C+cDim..<...) ]
# ════════════════════════════════════════════════════════════
proc buildFusedSplitConvSoftplusProgram*(C, cDim, H, K: int): VmProgram =
  ## NEW - FUSES split(z,xBC,dt) + conv1d(xBC) + softplus(dt) INTO A SINGLE
  ## KERNEL, using only the micro-VM's existing opcodes (NO hand-written .metal).
  ## Modelled on fusedNormQKVRoPE/fusedResidualNormFFN in nim_inference.nim
  ## (the REAL 6.7B model, no lag on the same hardware) - the common rule:
  ## fewer GPU dispatches + separate sessionCopyInto calls per layer.
  ##
  ## SAFE TO FUSE (no threadgroup_barrier needed): the conv1d here is
  ## DEPTHWISE OVER TIME - see the original buildConv1dIncProgram, EVERY access
  ## (state/weight/bias/input) goes through the thread's OWN tid=c, no
  ## thread needs another thread's data. So one thread can, in sequence,
  ## split ITS OWN xBC[tid] value (kept in a register, NO intermediate
  ## RAM write) and use it IMMEDIATELY for conv1d; same for dt+softplus.
  ##
  ## Benefit: 3 dispatches (split, conv, softplus) + 4 sessionCopyInto (moving
  ## xBC/dt between scratch buffers) -> a SINGLE dispatch,
  ## no intermediate copy. Uses at most R0..R7 (the VM has only 16 registers
  ## - R0..R15, see the pack2() 0xF mask in session_vm.nim and "float R[16]"
  ## in jit_vm.nim - going past that OVERFLOWS SILENTLY, giving wrong results).
  ##
  ## Input memory layout (matches how encodeLayerStep calls it):
  ##   [0, P)                    proj (P=C+cDim+H, output of the inProj fp4 gemv)
  ##   [P, P+C)                  outZ (plain copy, used later by rmsnorm-gate)
  ##   [P+C, P+C+cDim*(K-1))     convState (persistent, read - NOT updated
  ##                              here, encodeLayerStep writes it back later)
  ##   [.. + cDim*K)             convWeight (persistent, read only)
  ##   [.. + cDim)               convBias (persistent, read only)
  ##   [.. + cDim)               outXBC (= silu(conv(...)), used by ssm-step)
  ##   [.. + H)                  dtBias (persistent, read only)
  ##   [.. + H)                  outDt (= softplus(...), used by ssm-step)
  doAssert K >= 2 and K <= 8, "buildFusedSplitConvSoftplusProgram: K in [2,8]"
  let P = C + cDim + H
  let offOutZ   = P
  let offConvSt = offOutZ + C
  let offConvW  = offConvSt + cDim*(K-1)
  let offConvB  = offConvW + cDim*K
  let offOutXBC = offConvB + cDim
  let offDtBias = offOutXBC + cDim
  let offOutDt  = offDtBias + H
  let offOutConvSt = offOutDt + H
  ## NEW (fix for the frozen convState bug - see the debug report): an EXTRA output
  ## [offOutConvSt, offOutConvSt + cDim*(K-1)) = the SHIFTED convState, exactly the
  ## "slide state" logic of the original buildConv1dIncProgram (it WAS there, just
  ## dropped when the kernel was fused) - encodeLayerStep must copy this area back to
  ## o.offConvSt AFTER the dispatch, see the matching fix there.

  var p: VmProgram
  # R0 = tid, set once by the VM runtime, used for the whole program - NEVER
  # overwritten by the parts below.

  # ── Part 1: split z (plain copy, used later by rmsnorm-gate) ──
  p.emitLoadConstF32(1, float32(C))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitLoadF32Idx(3, 0)                      # R3 = proj[tid]
    p.emitAddConst(4, 0, float32(offOutZ))
    p.emitStoreF32Idx(3, 4)

  # ── Part 2: split xBC (kept in R7, NO intermediate RAM write) + conv1d
  # right in place (exactly the original buildConv1dIncProgram, only the input moves from memory
  # to register R7) ──
  p.emitLoadConstF32(1, float32(cDim))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitAddConst(3, 0, float32(C))
    p.emitLoadF32Idx(7, 3)                      # R7 = xBC[tid] = proj[C+tid]

    p.emitAddConst(1, 0, float32(offConvB))
    p.emitLoadF32Idx(1, 1)                      # R1 = acc = bias[c]
    p.emitLoadConstF32(8, 0.0'f32)                 # k (counted on the GPU)
    p.emitLoadConstF32(9, float32(K-1))     # iteration count
    sessionVmForGpu(p, 9):
      p.emitMulConst(3, 0, float32(K-1))
      p.emitAdd(3, 3, 8); p.emitAddConst(3, 3, float32(offConvSt))
      p.emitLoadF32Idx(2, 3)                    # R2 = state[c,k]
      p.emitMulConst(4, 0, float32(K))
      p.emitAdd(4, 4, 8); p.emitAddConst(4, 4, float32(offConvW))
      p.emitLoadF32Idx(4, 4)                    # R4 = weight[c,k]
      p.emitFma(1, 4, 2, 1)
      p.emitAddConst(8, 8, 1.0'f32)            # k += 1
    # xBC (R7) * weight[..., K-1]
    p.emitMulConst(4, 0, float32(K))
    p.emitAddConst(4, 4, float32((K-1) + offConvW))
    p.emitLoadF32Idx(4, 4)
    p.emitFma(1, 4, 7, 1)
    # silu(acc) = acc * sigmoid(acc)
    p.emitNeg(3, 1)
    p.emitExp(3, 3)
    p.emitAddConst(3, 3, 1.0'f32)
    p.emitLoadConstF32(5, 1.0'f32)
    p.emitDiv(3, 5, 3)
    p.emitMul(1, 1, 3)
    p.emitAddConst(6, 0, float32(offOutXBC))
    p.emitStoreF32Idx(1, 6)

    # ── NEW: "slide" convState - exactly the original buildConv1dIncProgram (dropped
    # when the kernel was fused, see the offOutConvSt note above). R7 still
    # holds the current xBC[tid] loaded at the start of this block (not touched by the FMA/silu
    # above) - reused directly, no need to reread memory. ──
    p.emitLoadConstF32(8, 0.0'f32)                 # k (counted on the GPU)
    p.emitLoadConstF32(9, float32(K-2))     # iteration count
    sessionVmForGpu(p, 9):
      p.emitMulConst(3, 0, float32(K-1))
      p.emitAdd(3, 3, 8); p.emitAddConst(3, 3, float32(1 + offConvSt))
      p.emitLoadF32Idx(2, 3)
      p.emitMulConst(4, 0, float32(K-1))
      p.emitAdd(4, 4, 8); p.emitAddConst(4, 4, float32(offOutConvSt))
      p.emitStoreF32Idx(2, 4)
      p.emitAddConst(8, 8, 1.0'f32)            # k += 1
    p.emitMulConst(3, 0, float32(K-1))
    p.emitAddConst(3, 3, float32((K-2) + offOutConvSt))
    p.emitStoreF32Idx(7, 3)

  # ── Part 3: split dt (R3) + softplus right in place ──
  p.emitLoadConstF32(1, float32(H))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitAddConst(4, 0, float32(C + cDim))
    p.emitLoadF32Idx(3, 4)                      # R3 = dt[tid] = proj[C+cDim+tid]
    p.emitAddConst(5, 0, float32(offDtBias))
    p.emitLoadF32Idx(5, 5)                      # R5 = dtBias[tid]
    p.emitAdd(3, 3, 5)
    p.emitSoftplus(3, 3)
    p.emitAddConst(4, 0, float32(offOutDt))
    p.emitStoreF32Idx(3, 4)

  p.emitHalt()
  result = p

proc buildSplitProjProgram*(C, cDim, H: int): VmProgram =
  let P = 2*C + 2*(0) # placeholder; real P = 2*C + cDim + H but cDim may be 2*C related
  # Actually cDim = C + 2*G*N typically. Layout from forward:
  # proj = [z(C) | xBC(cDim) | dt(H)]
  let offZ = 0           # we pack: input proj at 0, outputs after
  # mem layout:
  #   proj[0 .. P) where P = C + cDim + H
  #   outZ[P .. P+C)
  #   outXBC[P+C .. P+C+cDim)
  #   outDt[P+C+cDim .. P+C+cDim+H)
  let Ptot = C + cDim + H
  let offOutZ = Ptot
  let offOutXBC = offOutZ + C
  let offOutDt = offOutXBC + cDim
  var p: VmProgram
  # R0=tid
  # if tid < C: outZ[tid] = proj[tid]
  p.emitLoadConstF32(1, float32(C))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitLoadF32Idx(3, 0)  # proj[tid]
    p.emitAddConst(4, 0, float32(offOutZ))
    p.emitStoreF32Idx(3, 4)
  # if tid < cDim: outXBC[tid] = proj[C+tid]
  p.emitLoadConstF32(1, float32(cDim))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitAddConst(3, 0, float32(C))
    p.emitLoadF32Idx(3, 3)
    p.emitAddConst(4, 0, float32(offOutXBC))
    p.emitStoreF32Idx(3, 4)
  # if tid < H: outDt[tid] = proj[C+cDim+tid]
  p.emitLoadConstF32(1, float32(H))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitAddConst(3, 0, float32(C + cDim))
    p.emitLoadF32Idx(3, 3)
    p.emitAddConst(4, 0, float32(offOutDt))
    p.emitStoreF32Idx(3, 4)
  p.emitHalt()
  result = p

# ════════════════════════════════════════════════════════════
# 7) RMS+gate LARGE C (e.g. 8192) — 100% GPU, 1 threadgroup=1024
# Each thread handles elemsPerThread = C/1024 elements:
#   gated, localSumSq, then TG tree reduce → rsqrt → scale written to out.
# mem: [ yPre(0..<C) | z(C..<2C) | w(2C..<3C) | out(3C..<4C) ]
# Requires C % 1024 == 0.
# ════════════════════════════════════════════════════════════
proc buildRmsGateLargeProgram*(C: int; eps: float32; tgSize: int = 1024): VmProgram =
  doAssert C mod tgSize == 0, "buildRmsGateLargeProgram: C must be multiple of tgSize"
  let ept = C div tgSize  # elements per thread
  let offZ = C
  let offW = 2*C
  let offOut = 3*C

  var p: VmProgram
  # R0 = lid (thread in group) — kernel sets R0=global tid; for 1 TG grid, tid=lid
  # localSum in R1
  p.emitLoadConstF32(1, 0.0'f32)

  # loop e = 0 .. ept-1
  # idx = lid + e*tgSize
  p.emitLoadConstF32(8, float32(ept))   # count
  p.emitLoadConstF32(7, 0.0'f32)        # e
  sessionVmForGpu(p, 8):
    # idx = tid + e * tgSize
    p.emitMulConst(9, 7, float32(tgSize))
    p.emitAdd(9, 0, 9)                  # R9 = idx
    # y = yPre[idx]
    p.emitLoadF32Idx(2, 9)
    # z = z[idx]
    p.emitAddConst(3, 9, float32(offZ))
    p.emitLoadF32Idx(3, 3)
    # silu(z) = z * sigmoid(z)
    p.emitNeg(4, 3)
    p.emitExp(4, 4)
    p.emitAddConst(4, 4, 1.0'f32)
    p.emitLoadConstF32(5, 1.0'f32)
    p.emitDiv(4, 5, 4)
    p.emitMul(4, 3, 4)                  # silu
    p.emitMul(2, 2, 4)                  # gated
    # store gated temporarily into out[idx] (reuse)
    p.emitAddConst(6, 9, float32(offOut))
    p.emitStoreF32Idx(2, 6)
    # localSum += gated^2
    p.emitMul(10, 2, 2)
    p.emitAdd(1, 1, 10)
    p.emitAddConst(7, 7, 1.0'f32)       # e += 1

  # TG reduce localSum (R1) → tg[lid]
  p.emitTgStoreF32(1, 0, 0)
  p.emitBarrierTg()

  p.emitTgTreeReduce(0, tgSize, 6, 7, 8, 9, 10)   # GPU loop, no more host while

  # R11 = rsqrt(mean + eps)
  p.emitLoadConstF32(12, 0.0'f32)
  p.emitTgLoadF32(11, 12, 0)
  p.emitMulConst(11, 11, 1.0'f32/float32(C))
  p.emitAddConst(11, 11, eps)
  p.emitRsqrt(11, 11)

  # second loop: out[idx] = gated_stored * r * w[idx]
  p.emitLoadConstF32(8, float32(ept))
  p.emitLoadConstF32(7, 0.0'f32)
  sessionVmForGpu(p, 8):
    p.emitMulConst(9, 7, float32(tgSize))
    p.emitAdd(9, 0, 9)
    p.emitAddConst(2, 9, float32(offOut))
    p.emitLoadF32Idx(2, 2)              # gated
    p.emitAddConst(3, 9, float32(offW))
    p.emitLoadF32Idx(3, 3)              # w
    p.emitMul(2, 2, 11)                 # * r
    p.emitMul(2, 2, 3)                  # * w
    p.emitAddConst(4, 9, float32(offOut))
    p.emitStoreF32Idx(2, 4)
    p.emitAddConst(7, 7, 1.0'f32)

  p.emitHalt()
  result = p

# ════════════════════════════════════════════════════════════
# 9) "IN PLACE" VERSION (new) - split+conv1d+softplus, ssm-step, GROUPED
# gated-rmsnorm, all taking ABSOLUTE OFFSETS and reading/writing DIRECTLY on the resident
# layer buffer (hWork) - no more sessionCopyInto between steps
# (see the full explanation in BAO_CAO.md section 2, and the session_vm.nim/
# jit_vm.nim/micro_vm.metal patch adding OP_LOAD_LID/OP_LOAD_TGID needed by the 3rd function).
#
# The gated RMSNorm here normalises PER GROUP (group_size = C div nGroups),
# the ORIGINAL formula of mamba_ssm (RMSNormGated(..., group_size=d_inner//
# ngroups)) and of vLLM (Mixer2RMSNormGated(intermediate_size, n_groups,...)).
# buildRmsGateProgram/buildRmsGateLargeProgram ABOVE (sections 2/7) normalise
# over ALL of C - the BEHAVIOUR of HF transformers modeling_mamba2.py, where
# n_groups is MISSING in MambaRMSNormGated (confirmed in PR
# huggingface/transformers#35943, fixed for Zamba2; PR #40861 "Support
# n_groups>1 for mamba2" is still open). Calling buildRmsGateGroupedAt with nGroups=1
# reproduces the old function above EXACTLY, so the 2 paths can be compared directly
# (see test_layer0.nim / BYBY_MAMBA_NORM_GROUPS).
# ════════════════════════════════════════════════════════════

proc buildSplitConvSoftplusAt*(C, cDim, H, K: int;
                               offZ, offXBC, offDt,
                               offConvW, offConvB, offConvSt,
                               offDtBias: int): VmProgram =
  ## hWork layout (ABSOLUTE offsets, produced by computeOffV2):
  ##   [0, P)                proj = in_proj(x),  P = C + cDim + H  (read only)
  ##   [offZ,  offZ+C)       z          (written)
  ##   [offXBC, +cDim)       xBCConv after SiLU  (written)
  ##   [offDt,  +H)          dt after softplus   (written)
  ##   [offConvW, +cDim*K)   convWeight (read only, resident)
  ##   [offConvB, +cDim)     convBias   (read only, resident)
  ##   [offConvSt,+cDim*(K-1)) convState (READ + WRITTEN IN PLACE)
  ##   [offDtBias,+H)        dtBias     (read only, resident)
  ## convState is updated in place SAFELY since thread c only uses row c: it reads ALL
  ## K-1 values into registers FIRST and writes back only when done (no more need for
  ## the separate offOutConvSt buffer of the old buildFusedSplitConvSoftplusProgram).
  doAssert K >= 2 and K <= 8, "buildSplitConvSoftplusAt: K must be in [2,8]"
  var p: VmProgram

  # ── Phan 1: z = proj[0..C) ──
  p.emitLoadConstF32(1, float32(C))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitLoadF32Idx(3, 0)                       # R3 = proj[tid]
    p.emitAddConst(4, 0, float32(offZ))
    p.emitStoreF32Idx(3, 4)

  # ── Part 2: xBC = proj[C..C+cDim) -> conv1d -> SiLU, then slide convState ──
  p.emitLoadConstF32(1, float32(cDim))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitAddConst(3, 0, float32(C))
    p.emitLoadF32Idx(7, 3)                       # R7 = xNew (kept for the whole block)
    p.emitMulConst(6, 0, float32(K - 1))         # R6 = tid*(K-1) (kept for the whole block)
    # acc = bias + sum_k w[c,k]*state[c,k] + w[c,K-1]*xNew   (k loop runs on the GPU)
    # No need to "stage" the state into R8..: sliding SEQUENTIALLY in place is still safe because
    # the acc loop finishes first, and the slide loop reads state[k+1] before iteration k+1 overwrites it.
    p.emitAddConst(1, 0, float32(offConvB))
    p.emitLoadF32Idx(1, 1)                       # R1 = acc = bias[c]
    p.emitMulConst(5, 0, float32(K))             # R5 = tid*K
    p.emitLoadConstF32(8, 0.0'f32)               # R8 = k
    p.emitLoadConstF32(9, float32(K - 1))        # R9 = iteration count (constant)
    sessionVmForGpu(p, 9):
      p.emitAdd(3, 6, 8); p.emitAddConst(3, 3, float32(offConvSt))
      p.emitLoadF32Idx(2, 3)                     # R2 = state[c,k]
      p.emitAdd(4, 5, 8); p.emitAddConst(4, 4, float32(offConvW))
      p.emitLoadF32Idx(4, 4)                     # R4 = w[c,k]
      p.emitFma(1, 4, 2, 1)
      p.emitAddConst(8, 8, 1.0'f32)
    p.emitAddConst(4, 5, float32((K - 1) + offConvW))
    p.emitLoadF32Idx(4, 4)
    p.emitFma(1, 4, 7, 1)
    # SiLU(acc)
    p.emitNeg(3, 1); p.emitExp(3, 3); p.emitAddConst(3, 3, 1.0'f32)
    p.emitLoadConstF32(5, 1.0'f32)               # R5 (tid*K) is no longer needed
    p.emitDiv(3, 5, 3)
    p.emitMul(1, 1, 3)
    p.emitAddConst(4, 0, float32(offXBC))
    p.emitStoreF32Idx(1, 4)
    # slide the window: state[k] = state[k+1] (k=0..K-3), state[K-2] = xNew  (GPU loop)
    p.emitLoadConstF32(8, 0.0'f32)
    p.emitLoadConstF32(9, float32(K - 2))
    sessionVmForGpu(p, 9):
      p.emitAdd(3, 6, 8); p.emitAddConst(3, 3, float32(1 + offConvSt))
      p.emitLoadF32Idx(2, 3)                     # R2 = state[k+1]
      p.emitAdd(4, 6, 8); p.emitAddConst(4, 4, float32(offConvSt))
      p.emitStoreF32Idx(2, 4)                    # state[k] = R2
      p.emitAddConst(8, 8, 1.0'f32)
    p.emitAddConst(3, 6, float32((K - 2) + offConvSt))
    p.emitStoreF32Idx(7, 3)

  # ── Phan 3: dt = softplus(proj[C+cDim+tid] + dtBias[tid]) ──
  p.emitLoadConstF32(1, float32(H))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitAddConst(4, 0, float32(C + cDim))
    p.emitLoadF32Idx(3, 4)
    p.emitAddConst(5, 0, float32(offDtBias))
    p.emitLoadF32Idx(5, 5)
    p.emitAdd(3, 3, 5)
    p.emitSoftplus(3, 3)
    p.emitAddConst(4, 0, float32(offDt))
    p.emitStoreF32Idx(3, 4)

  p.emitHalt()
  result = p

proc buildMamba2StepAt*(H, D, N, G: int;
                        offX, offB, offC, offDt, offALog, offDSkip,
                        offState, offY, offIdxH, offIdxG: int): VmProgram =
  ## EXACTLY buildMamba2StepProgram above, with one difference: EVERY offset is a
  ## PARAMETER instead of being hard-coded from 0 - allows reading/writing directly on hWork.
  var p: VmProgram
  p.emitAddConst(12, 0, float32(offIdxH)); p.emitLoadF32Idx(1, 12)   # R1 = h
  p.emitAddConst(12, 0, float32(offIdxG)); p.emitLoadF32Idx(3, 12)   # R3 = g

  p.emitAddConst(12, 1, float32(offALog)); p.emitLoadF32Idx(12, 12)
  p.emitExp(12, 12); p.emitNeg(12, 12)                               # R12 = A_h
  p.emitAddConst(11, 1, float32(offDt)); p.emitLoadF32Idx(11, 11)    # R11 = dt[h]
  p.emitMul(9, 11, 12); p.emitExp(9, 9)                              # R9  = dA

  p.emitAddConst(10, 0, float32(offX)); p.emitLoadF32Idx(10, 10)     # R10 = x[h,d]

  p.emitAddConst(12, 1, float32(offDSkip)); p.emitLoadF32Idx(12, 12)
  p.emitMul(4, 12, 10)                                               # R4 = yAcc = D[h]*x

  p.emitMulConst(2, 3, float32(N))          # R2 = g*N
  p.emitMulConst(6, 0, float32(N))          # R6 = tid*N
  p.emitLoadConstF32(7, 0.0'f32)            # R7 = n
  p.emitLoadConstF32(8, float32(N))         # R8 = count

  sessionVmForGpu(p, 8):
    p.emitAddConst(13, 2, float32(offB)); p.emitAdd(13, 13, 7); p.emitLoadF32Idx(13, 13)
    p.emitAddConst(14, 2, float32(offC)); p.emitAdd(14, 14, 7); p.emitLoadF32Idx(14, 14)
    p.emitAddConst(15, 6, float32(offState)); p.emitAdd(15, 15, 7)
    p.emitLoadF32Idx(12, 15)
    p.emitMul(12, 12, 9)
    p.emitMul(13, 13, 10)
    p.emitFma(12, 11, 13, 12)
    p.emitStoreF32Idx(12, 15)
    p.emitFma(4, 14, 12, 4)
    p.emitAddConst(7, 7, 1.0'f32)

  p.emitAddConst(13, 0, float32(offY))
  p.emitStoreF32Idx(4, 13)
  p.emitHalt()
  result = p

proc buildRmsGateGroupedAt*(C, nGroups, tgSize: int; eps: float32;
                            offY, offZ, offW, offOut: int): VmProgram =
  ## gated = y * SiLU(z); each group g (length L = C/nGroups):
  ##   r_g = rsqrt(mean_{i in g}(gated^2) + eps)
  ##   out[i] = gated[i] * r_{g(i)} * w[i]
  ## Dispatch: nThreads = nGroups*tgSize, threadgroup = tgSize -> EACH
  ## threadgroup handles exactly 1 group. nGroups=1 reproduces EXACTLY
  ## buildRmsGateProgram/buildRmsGateLargeProgram above (normalising over all of C).
  ## Uses OP_LOAD_LID/OP_LOAD_TGID (newly added in session_vm.nim) because the old VM
  ## only had thread_position_in_grid and could not split lid/groupId.
  doAssert nGroups >= 1 and C mod nGroups == 0,
    "buildRmsGateGroupedAt: C must be divisible by nGroups"
  let groupLen = C div nGroups
  doAssert tgSize >= 1 and groupLen mod tgSize == 0,
    "buildRmsGateGroupedAt: groupLen (" & $groupLen & ") must be divisible by tgSize (" & $tgSize & ")"
  doAssert (tgSize and (tgSize - 1)) == 0, "buildRmsGateGroupedAt: tgSize must be a power of 2"
  let ept = groupLen div tgSize

  var p: VmProgram
  p.emitLoadLid(1)                                   # R1 = lid
  p.emitLoadTgid(2)                                  # R2 = groupId
  p.emitMulConst(2, 2, float32(groupLen))            # R2 = groupBase
  p.emitLoadConstF32(3, 0.0'f32)                     # R3 = localSum

  # ── pass 1: gated + localSum(gated^2), stash gated in offOut ──
  p.emitLoadConstF32(8, float32(ept))
  p.emitLoadConstF32(7, 0.0'f32)
  sessionVmForGpu(p, 8):
    p.emitMulConst(9, 7, float32(tgSize))
    p.emitAdd(9, 9, 1)
    p.emitAdd(9, 9, 2)                               # R9 = i (index within C)
    p.emitAddConst(10, 9, float32(offY));  p.emitLoadF32Idx(10, 10)
    p.emitAddConst(11, 9, float32(offZ));  p.emitLoadF32Idx(11, 11)
    p.emitNeg(12, 11); p.emitExp(12, 12); p.emitAddConst(12, 12, 1.0'f32)
    p.emitLoadConstF32(13, 1.0'f32); p.emitDiv(12, 13, 12)
    p.emitMul(12, 11, 12)                            # SiLU(z)
    p.emitMul(10, 10, 12)                            # gated
    p.emitAddConst(13, 9, float32(offOut)); p.emitStoreF32Idx(10, 13)
    p.emitMul(14, 10, 10); p.emitAdd(3, 3, 14)
    p.emitAddConst(7, 7, 1.0'f32)

  # ── tree reduce in the threadgroup (index = lid, NOT the global tid) ──
  p.emitTgStoreF32(3, 1, 0)
  p.emitBarrierTg()
  p.emitTgTreeReduce(1, tgSize, 4, 5, 6, 10, 11)   # GPU loop, no more host while

  p.emitLoadConstF32(4, 0.0'f32)
  p.emitTgLoadF32(15, 4, 0)
  p.emitMulConst(15, 15, 1.0'f32 / float32(groupLen))
  p.emitAddConst(15, 15, eps)
  p.emitRsqrt(15, 15)                                # R15 = r_g (kept until the end)

  # ── pass 2: out = gated * r * w ──
  p.emitLoadConstF32(8, float32(ept))
  p.emitLoadConstF32(7, 0.0'f32)
  sessionVmForGpu(p, 8):
    p.emitMulConst(9, 7, float32(tgSize))
    p.emitAdd(9, 9, 1)
    p.emitAdd(9, 9, 2)
    p.emitAddConst(10, 9, float32(offOut)); p.emitLoadF32Idx(10, 10)
    p.emitAddConst(11, 9, float32(offW));   p.emitLoadF32Idx(11, 11)
    p.emitMul(10, 10, 15)
    p.emitMul(10, 10, 11)
    p.emitAddConst(13, 9, float32(offOut)); p.emitStoreF32Idx(10, 13)
    p.emitAddConst(7, 7, 1.0'f32)

  p.emitHalt()
  result = p

# ═══════════════════════════════════════════════════════════════════════
# SELECTIVE SCAN BACKWARD — 1 DISPATCH FOR ALL T STEPS, 100% GPU
# ═══════════════════════════════════════════════════════════════════════
## The old version had 2 problems: (1) buildMamba2StepBackwardAt needed a CPU LOOP over t, and its
## registers clobbered each other (R12/R13 used for 3 things) and it called an emitAtomicAdd that never existed;
## (2) buildMamba2BackwardFullAt was only a stub (`emitHalt` in the loop) -> the kernel did
## nothing and selectiveScanBackwardGpu SILENTLY returned all zeros. Both are replaced by the kernel
## below (checked against finite-difference gradients - see t_mamba.nim).
##
## Math (each thread = 1 (b,h,d); s_t = s_{t-1}*dA_t + dt_t*B_t*x_t ; y_t = D*x_t + sum_n C_t[n]*s_t[n]):
##   dS_tn   = dS_{t+1,n}*dA_{t+1} + C_t[n]*dY_t
##   dC[t,g,n] += dY_t * s_t[n]                         (atomic - shared by every h,d in group g)
##   dB[t,g,n] += dS_tn*dt_t*x_t                        (atomic)
##   dX[t]     += dS_tn*dt_t*B_t[n]  (+ dY_t*D in the skip pass)
##   dDt[t,h]  += dS_tn*s_{t-1}[n]*dA_t*A + dS_tn*B_t[n]*x_t   (atomic)
##   dALog[h]  += dS_tn*s_{t-1}[n]*dA_t*dt_t*A                 (atomic)
##   dDSkip[h] += dY_t*x_t                                      (atomic)
## dS_n depends only on the same n, so the loops are swapped: n OUTER, t INNER (backwards) -> dS fits in
## 1 register, no per-thread array needed. Addresses go through U[] (uint32) since
## B*(T+1)*HD*N easily exceeds 2^24, the limit where float32 is still exact.
##
## memF32 layout (absolute offsets set by the caller):
##   read : X[B,T,HD] Bm[B,T,G,N] Cm[B,T,G,N] Dt[B,T,H] ALog[H] DSkip[H]
##         State[B,T+1,HD,N] (State[:,0] = initial, State[:,t+1] = after step t)  DY[B,T,HD]
##   write: DX[B,T,HD] (the kernel overwrites it, no zeroing needed)
##   atomic accumulation -> the caller MUST zero first: DB DC DDt DALog DDSkip
## Grid = B*HD threads. h, g are derived inside the kernel (no more idxH/idxG tables
## precomputed on the CPU). Needs MSL 3.0+ (atomic_float) - only runs through the JIT path.
proc buildMamba2BackwardFullAt*(B, T, H, D, N, G: int;
                                offX, offB, offC, offDt, offALog, offDSkip,
                                offState, offDY,
                                offDX, offDB, offDC, offDDt, offDALog, offDDSkip: int;
                                xStr = -1, bcStr = -1;
                                nBeg = 0, nCnt = -1): VmProgram =
  ## nBeg/nCnt: only handles n in [nBeg, nBeg+nCnt) (all of N by default). The skip pass (dX=dY*D, dDSkip)
  ## only runs when nBeg == 0. Split up so each command buffer stays under the GPU watchdog (Intel ~a few seconds);
  ## the calls accumulate atomically so the summed result equals a single call.
  ## xStr/bcStr: per-timestep stride of X/dX and B,C/dB,dC (default HD and G*N =
  ## contiguous layout). Set = cDim to read/write DIRECTLY in the conv buffer [BT,cDim]
  ## (then offB = offConv+C, offC = offConv+C+G*N, offDX/DB/DC point into dXBC on the same row).
  doAssert B >= 1 and T >= 1 and H mod G == 0, "buildMamba2BackwardFullAt: invalid parameters"
  let HD = H * D
  let xS = if xStr < 0: HD else: xStr
  let bcS = if bcStr < 0: G * N else: bcStr
  let hpg = H div G
  let nC = if nCnt < 0: N - nBeg else: nCnt
  doAssert nBeg >= 0 and nC >= 1 and nBeg + nC <= N, "buildMamba2BackwardFullAt: invalid n range"
  var p: VmProgram

  # ── decode tid -> b, hd, h, g (trunc((x+0.5)/m) avoids float rounding errors) ──
  p.emitAddConst(1, 0, 0.5'f32)
  p.emitMulConst(1, 1, 1.0'f32 / float32(HD))
  p.emitRToU(0, 1)                                   # U0 = b
  p.emitUToR(2, 0)
  p.emitMulConst(3, 2, float32(HD))
  p.emitSub(4, 0, 3)                                 # R4 = hd (float, exact)
  p.emitRToU(3, 4)                                   # U3 = hd
  p.emitAddConst(5, 4, 0.5'f32)
  p.emitMulConst(5, 5, 1.0'f32 / float32(D))
  p.emitRToU(4, 5)                                   # U4 = h
  p.emitUToR(6, 4)
  p.emitAddConst(6, 6, 0.5'f32)
  p.emitMulConst(6, 6, 1.0'f32 / float32(hpg))
  p.emitRToU(5, 6)                                   # U5 = g

  # R8 = A_h = -exp(aLog[h]) ; R9 = D[h]
  p.emitULoadF32(8, 4, uint32(offALog)); p.emitExp(8, 8); p.emitNeg(8, 8)
  p.emitULoadF32(9, 4, uint32(offDSkip))

  # ── base pointers (uint32) ──
  p.emitUMulK(6, 0, uint32((T + 1) * HD * N)); p.emitUMulK(7, 3, uint32(N)); p.emitUAdd(6, 6, 7)   # U6 = stateBase
  p.emitUMulK(7, 0, uint32(T * xS)); p.emitUAdd(7, 7, 3)                                        # U7 = xBase
  p.emitUMulK(1, 0, uint32(T * HD)); p.emitUAdd(1, 1, 3)                                        # U1 = dYBase
  p.emitUMulK(8, 0, uint32(T * bcS)); p.emitUMulK(9, 5, uint32(N)); p.emitUAdd(8, 8, 9)         # U8 = bcBase
  p.emitUMulK(9, 0, uint32(T * H)); p.emitUAdd(9, 9, 4)                                         # U9 = dtBase

  # ── pass 1 (skip): dX = dY*D ; dDSkip[h] += dY*x   — the t loop runs on the GPU ──
  if nBeg == 0:
    p.emitUAddK(10, 7, 0)                              # U10 = x ptr
    p.emitUAddK(2, 1, 0)                               # U2 = dY ptr
    p.emitLoadConstF32(13, float32(T))
    sessionVmForGpu(p, 13):
      p.emitULoadF32(10, 10, uint32(offX))
      p.emitULoadF32(11, 2, uint32(offDY))
      p.emitMul(12, 11, 9); p.emitUStoreF32(12, 10, uint32(offDX))
      p.emitMul(12, 11, 10); p.emitUAtomicAdd(12, 4, uint32(offDDSkip))
      p.emitUAddK(10, 10, uint32(xS)); p.emitUAddK(2, 2, uint32(HD))

  # ── pass 2: for n (GPU) { dS=0; for t=T-1..0 (GPU) {...} } ──
  p.emitLoadConstF32(0, float32(nC))                 # R0 = n iteration count (CONSTANT; tid no longer needed)
  p.emitLoadConstF32(12, float32(T))                 # R12 = t iteration count (CONSTANT)
  p.emitUConst(10, uint32(nBeg))                     # U10 = n
  sessionVmForGpu(p, 0):
    p.emitLoadConstF32(15, 0.0'f32)                  # dS = 0
    p.emitUAdd(11, 6, 10); p.emitUAddK(11, 11, uint32((T - 1) * HD * N))   # U11 = sPrev ptr (t=T-1)
    p.emitUAdd(12, 8, 10); p.emitUAddK(12, 12, uint32((T - 1) * bcS))      # U12 = B/C ptr
    p.emitUAddK(13, 7, uint32((T - 1) * xS))                               # U13 = x/dX ptr
    p.emitUAddK(2, 1, uint32((T - 1) * HD))                                # U2 = dY ptr
    p.emitUAddK(14, 9, uint32((T - 1) * H))                                # U14 = dt ptr
    sessionVmForGpu(p, 12):
      p.emitULoadF32(1, 12, uint32(offB))            # R1 = B
      p.emitULoadF32(2, 12, uint32(offC))            # R2 = C
      p.emitULoadF32(3, 13, uint32(offX))            # R3 = x
      p.emitULoadF32(4, 2, uint32(offDY))            # R4 = dY
      p.emitULoadF32(5, 14, uint32(offDt))           # R5 = dt
      p.emitULoadF32(6, 11, uint32(offState))        # R6 = sPrev
      p.emitULoadF32(7, 11, uint32(offState + HD * N))   # R7 = sNew
      p.emitMul(10, 5, 8); p.emitExp(10, 10)         # R10 = dA = exp(dt*A)
      p.emitMul(13, 2, 4); p.emitAdd(15, 15, 13)     # R15 = dS_tn = dS + C*dY
      p.emitMul(13, 4, 7); p.emitUAtomicAdd(13, 12, uint32(offDC))        # dC
      p.emitMul(13, 15, 5); p.emitMul(13, 13, 3); p.emitUAtomicAdd(13, 12, uint32(offDB))   # dB
      p.emitMul(13, 15, 5); p.emitMul(13, 13, 1)     # dS*dt*B
      p.emitULoadF32(14, 13, uint32(offDX)); p.emitAdd(14, 14, 13); p.emitUStoreF32(14, 13, uint32(offDX))
      p.emitMul(13, 15, 6); p.emitMul(13, 13, 10)    # R13 = dS*sPrev*dA
      p.emitMul(14, 13, 8)                           # *A
      p.emitMul(11, 15, 1); p.emitMul(11, 11, 3)     # dS*B*x
      p.emitAdd(14, 14, 11); p.emitUAtomicAdd(14, 14, uint32(offDDt))     # dDt
      p.emitMul(14, 13, 5); p.emitMul(14, 14, 8); p.emitUAtomicAdd(14, 4, uint32(offDALog))  # dALog
      p.emitMul(15, 15, 10)                          # dS *= dA
      p.emitUAddK(11, 11, uint32(0xFFFFFFFF'u32 - uint32(HD * N) + 1))   # ptr -= HD*N
      p.emitUAddK(12, 12, uint32(0xFFFFFFFF'u32 - uint32(bcS) + 1))
      p.emitUAddK(13, 13, uint32(0xFFFFFFFF'u32 - uint32(xS) + 1))
      p.emitUAddK(2, 2, uint32(0xFFFFFFFF'u32 - uint32(HD) + 1))
      p.emitUAddK(14, 14, uint32(0xFFFFFFFF'u32 - uint32(H) + 1))
    p.emitUAddK(10, 10, 1)                           # n += 1
  p.emitHalt()
  result = p

# ═══════════════════════════════════════════════════════════════════════
# RMSNorm-gate BACKWARD — 1 threadgroup, 2 reductions (sum gated², dot) on the GPU
# ═══════════════════════════════════════════════════════════════════════
## Replaces the old backwardRmsGateGpu (CPU: `for v in gated2: sum += v`, `for i in 0..<C:
## dotSum += ...`, `newSeq(C).mapIt(...)` + ~20 separate beMul/beAdd dispatches).
## memF32 layout (offsets from the caller): [yPre | z | w | dN | dYpre | dZ | dNormW], each area C floats
## Grid = tgSize threads, 1 threadgroup, tgSize = a power of 2 dividing C (see rmsBwdTgSize).
## The caller provides tgBytes >= tgSize*4.
proc rmsBwdTgSize*(C: int; cap = 1024): int =
  ## largest power of 2 dividing C, capped by cap (bit trick, no loop)
  min(C and (-C), cap)

proc buildRmsGateBackwardProgram*(C: int; eps: float32; tgSize: int): VmProgram =
  doAssert tgSize >= 1 and (tgSize and (tgSize - 1)) == 0 and C mod tgSize == 0,
    "buildRmsGateBackwardProgram: tgSize must be a power of 2 and divide C"
  let ept = C div tgSize
  let offY = 0; let offZ = C; let offW = 2*C; let offDN = 3*C
  let offDYp = 4*C; let offDZ = 5*C; let offDW = 6*C
  var p: VmProgram
  p.emitLoadConstF32(1, 0.0'f32)                 # sumG2
  p.emitLoadConstF32(2, 0.0'f32)                 # dotSum
  p.emitLoadConstF32(3, float32(ept))            # iteration count (CONSTANT)
  p.emitLoadConstF32(4, 0.0'f32)                 # e
  sessionVmForGpu(p, 3):
    p.emitMulConst(5, 4, float32(tgSize)); p.emitAdd(5, 5, 0)       # R5 = idx = e*tg + lid
    p.emitAddConst(6, 5, float32(offZ)); p.emitLoadF32Idx(6, 6)     # z
    p.emitAddConst(7, 5, float32(offY)); p.emitLoadF32Idx(7, 7)     # yPre
    p.emitNeg(8, 6); p.emitExp(8, 8); p.emitAddConst(8, 8, 1.0'f32)
    p.emitLoadConstF32(9, 1.0'f32); p.emitDiv(8, 9, 8)              # R8 = sigmoid(z)
    p.emitMul(9, 6, 8)                                              # R9 = silu(z)
    p.emitMul(7, 7, 9)                                              # R7 = gated
    p.emitMul(10, 7, 7); p.emitAdd(1, 1, 10)
    p.emitAddConst(11, 5, float32(offDN)); p.emitLoadF32Idx(11, 11)
    p.emitAddConst(12, 5, float32(offW));  p.emitLoadF32Idx(12, 12)
    p.emitMul(11, 11, 12); p.emitMul(11, 11, 7); p.emitAdd(2, 2, 11)  # dot += dN*w*gated
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitLoadConstF32(11, 0.0'f32)
  p.emitTgStoreF32(1, 0, 0); p.emitBarrierTg()
  p.emitTgTreeReduce(0, tgSize, 6, 7, 8, 9, 10)
  p.emitTgLoadF32(13, 11, 0)                     # R13 = sum gated²
  p.emitBarrierTg()
  p.emitTgStoreF32(2, 0, 0); p.emitBarrierTg()
  p.emitTgTreeReduce(0, tgSize, 6, 7, 8, 9, 10)
  p.emitTgLoadF32(14, 11, 0)                     # R14 = dotSum
  p.emitMulConst(13, 13, 1.0'f32 / float32(C)); p.emitAddConst(13, 13, eps); p.emitRsqrt(13, 13)  # r
  p.emitMul(15, 13, 13); p.emitMul(15, 15, 13)
  p.emitMulConst(15, 15, 1.0'f32 / float32(C)); p.emitMul(15, 15, 14)   # R15 = r^3/C * dot
  p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(5, 4, float32(tgSize)); p.emitAdd(5, 5, 0)
    p.emitAddConst(6, 5, float32(offZ)); p.emitLoadF32Idx(6, 6)
    p.emitAddConst(7, 5, float32(offY)); p.emitLoadF32Idx(7, 7)
    p.emitNeg(8, 6); p.emitExp(8, 8); p.emitAddConst(8, 8, 1.0'f32)
    p.emitLoadConstF32(9, 1.0'f32); p.emitDiv(8, 9, 8)              # sig
    p.emitMul(9, 6, 8)                                              # silu
    p.emitMul(10, 7, 9)                                             # gated
    p.emitAddConst(11, 5, float32(offDN)); p.emitLoadF32Idx(11, 11)
    p.emitAddConst(12, 5, float32(offW));  p.emitLoadF32Idx(12, 12)
    p.emitMul(11, 11, 12)                                           # R11 = dNormedC
    p.emitMul(12, 13, 11); p.emitMul(2, 10, 15); p.emitSub(12, 12, 2)   # R12 = dGated
    p.emitMul(1, 12, 9); p.emitAddConst(2, 5, float32(offDYp)); p.emitStoreF32Idx(1, 2)   # dYpre
    p.emitLoadConstF32(2, 1.0'f32); p.emitSub(1, 2, 8); p.emitMul(1, 6, 1)
    p.emitAddConst(1, 1, 1.0'f32); p.emitMul(1, 8, 1)               # R1 = dSilu(z)
    p.emitMul(2, 12, 7); p.emitMul(2, 2, 1)
    p.emitAddConst(1, 5, float32(offDZ)); p.emitStoreF32Idx(2, 1)   # dZ
    p.emitAddConst(1, 5, float32(offDN)); p.emitLoadF32Idx(2, 1); p.emitMul(2, 2, 10); p.emitMul(2, 2, 13)   # dN*gated*r
    p.emitAddConst(1, 5, float32(offDW)); p.emitStoreF32Idx(2, 1)   # dNormW
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitHalt()
  result = p

# ── idxH/idxG index tables generated on the GPU (replaces the `for h/for d` loop on the CPU) ──
## Layout: [idxH(HD) | idxG(HD)], grid = HD threads. h = floor(tid/D) = #{k in 1..H-1 : tid >= k*D},
## g = floor(h/hpg) likewise — uses only CMP_LT/SUB/ADD so it runs both in the interpreter
## and the JIT (no U-* ops, no floor needed).
proc buildIdxTableProgram*(H, D, G: int): VmProgram =
  let HD = H * D
  let hpg = max(1, H div G)
  var p: VmProgram
  p.emitLoadConstF32(1, 0.0'f32)                     # h
  p.emitLoadConstF32(2, 1.0'f32)                     # k
  p.emitLoadConstF32(3, float32(H - 1))              # iteration count (CONSTANT)
  sessionVmForGpu(p, 3):
    p.emitMulConst(4, 2, float32(D)); p.emitCmpLt(5, 0, 4)       # tid < k*D ?
    p.emitLoadConstF32(6, 1.0'f32); p.emitSub(6, 6, 5); p.emitAdd(1, 1, 6)
    p.emitAddConst(2, 2, 1.0'f32)
  p.emitLoadConstF32(7, 0.0'f32)                     # g
  p.emitLoadConstF32(2, 1.0'f32)
  p.emitLoadConstF32(3, float32(G - 1))
  sessionVmForGpu(p, 3):
    p.emitMulConst(4, 2, float32(hpg)); p.emitCmpLt(5, 1, 4)     # h < k*hpg ?
    p.emitLoadConstF32(6, 1.0'f32); p.emitSub(6, 6, 5); p.emitAdd(7, 7, 6)
    p.emitAddConst(2, 2, 1.0'f32)
  p.emitStoreF32(1, 0, 0)                            # idxH[tid] = h
  p.emitAddConst(8, 0, float32(HD)); p.emitStoreF32Idx(7, 8)     # idxG[tid] = g
  p.emitHalt()
  result = p

when isMainModule:
  echo "ssm step words = ", buildMamba2StepProgram(4, 4, 8, 2).words.len
  echo "rmsgate words  = ", buildRmsGateProgram(8, 1e-5'f32).words.len
  echo "conv1d words   = ", buildConv1dIncProgram(16, 4).words.len
  echo "softplus words = ", buildSoftplusDtProgram(8).words.len
  echo "splitconv(at) words = ", buildSplitConvSoftplusAt(8192, 10240, 128, 4,
        20000, 30000, 45000, 46000, 90000, 92000, 95000).words.len
  echo "ssm(at) words       = ", buildMamba2StepAt(128, 64, 128, 8,
        0, 8192, 9216, 10240, 10368, 10496, 10624, 1059200, 1067392, 1075584).words.len
  echo "rmsgroup(at) words  = ", buildRmsGateGroupedAt(8192, 8, 256, 1e-5'f32,
        0, 8192, 16384, 24576).words.len
  echo "backward words      = ", buildMamba2BackwardFullAt(1, 4, 4, 4, 8, 2,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13).words.len
  echo "rmsbwd words         = ", buildRmsGateBackwardProgram(24, 1e-5'f32, 8).words.len
