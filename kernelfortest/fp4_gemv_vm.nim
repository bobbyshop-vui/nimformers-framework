## fp4_gemv_vm.nim — GPU kernels for the infer path, written as micro-VM BYTECODE.
##
## There is NOT a single line of MSL in this file (nor in lora.nim):
## every kernel here is a VmProgram (session_vm.nim) -> jit_vm.dispatchProgram
## lowered + JIT-compiled once per shape -> pipeline cache -> encoded into the session.
##
## Kernel buffer convention after JIT (see jit_vm.lowerBytecodeToMetal):
##   buffer(1) memF32   : output / work (written, may have an offset)
##   buffer(2) memU8    : packed fp4 weights
##   buffer(3) constF32 : LUT (272/256 floats) or fp32 weights (LoRA) or bias
##   buffer(4) inF32    : input x (declared only when the program uses OP_U_LOAD_IN)
##
## Every weight address goes through the integer registers U[] (uint32) — float
## R[] is not used because lm_head 32768x4096/2 = 67M bytes > 2^24.

import std/math
import std/bitops
import session_vm

const
  GemvTg* = 32          ## threads / threadgroup of every gemv kernel (1 output column / threadgroup)
  Fp4KmLutLen* = 272
  Fp4LutLen* = 256

proc alignUp64*(x: int): int {.inline.} = ((x + 63) div 64) * 64

# ── helper: 32-lane reduction in threadgroup memory ──────────
proc emitTgReduce32(p: var VmProgram; accReg: uint32; tgBase: int) =
  ## Sums 32 lanes in tgF32[tgBase .. tgBase+32). The result is at tgF32[tgBase]
  ## (only lane 0 reads it). No simd_sum -> correct for ANY simdWidth (Intel = 16).
  ## Requires: R6 = float(lid) (already set). Uses R7,R8,R9,R10.
  p.emitTgStoreF32(accReg, 6, uint32(tgBase))
  p.emitBarrierTg()
  ## The 5 steps (stride 16,8,4,2,1) are ONE FOR loop on the GPU, the stride lives in R7 and
  ## halves each iteration (no more host-side `for s in [...]` unroll).
  ## R5 = iteration count (CONSTANT in the loop), R8 = cmp then addr. At all 3 call sites R5 is free
  ## (acc in R4/R12, R6 = lane).
  doAssert accReg != 5 and accReg != 7 and accReg != 8 and accReg != 9 and accReg != 10
  p.emitLoadConstF32(5, 5.0'f32)
  p.emitLoadConstF32(7, 16.0'f32)
  sessionVmForGpu(p, 5):
    p.emitCmpLt(8, 6, 7)                        # lid < stride
    sessionVmIfGpu(p, 8):
      p.emitAdd(8, 6, 7)                        # lid + stride
      p.emitTgLoadF32(9, 6, uint32(tgBase))
      p.emitTgLoadF32(10, 8, uint32(tgBase))
      p.emitAdd(9, 9, 10)
      p.emitTgStoreF32(9, 6, uint32(tgBase))
    p.emitBarrierTg()                           # top level: every lane passes through
    p.emitMulConst(7, 7, 0.5'f32)

proc emitLidIsZeroIf(p: var VmProgram) =
  p.emitLoadConstF32(7, 1.0'f32)
  p.emitCmpLt(8, 6, 7)                          # lid < 1  <=> lid == 0
  p.emitIfBegin(8)

# ── 1) FP4 GEMV (M=1) ───────────────────────────────────────────
proc fp4GemvTgBytes*(km: bool): int =
  ((if km: Fp4KmLutLen else: Fp4LutLen) + GemvTg) * 4

proc fp4KmGemv2RowTgBytes*(): int =
  ## LUT + two 32-lane reduction areas, for the 2-output-rows/threadgroup tile.
  (Fp4KmLutLen + 2 * GemvTg) * 4

proc buildFp4GemvProgram*(K, N: int; km: bool; groupSize = 32): VmProgram =
  ## out[n] = sum_k x[k] * dequant(W[n,k]),  n = threadgroup id, 32 lanes split K.
  ## km=true : FP4-KM (E2M1 + E4M3 scale per group of 32, llama.cpp-style nibbles):
  ##           byte t holds element t (low nibble) and t+16 (high nibble);
  ##           merged 272 LUT: [0..15]=E2M1, [16..271]=E4M3 premultiplied by globalScale.
  ## km=false: old format, 1 byte/element, 256 LUT premultiplied by scale.
  if K mod GemvTg != 0:
    raise newException(ValueError, "buildFp4GemvProgram: K=" & $K & " must be divisible by " & $GemvTg)
  if km and not (groupSize == 32 or (groupSize >= 64 and groupSize mod 64 == 0)):
    raise newException(ValueError, "buildFp4GemvProgram: FP4-KM supports groupSize=32 or multiples of 64 (got " & $groupSize & ")")
  if km and K mod groupSize != 0:
    raise newException(ValueError, "buildFp4GemvProgram: K=" & $K & " must be divisible by groupSize=" & $groupSize)
  let lutN = if km: Fp4KmLutLen else: Fp4LutLen
  var p: VmProgram
  p.emitUTgid(0)                                  # U0 = n
  p.emitULid(1)                                   # U1 = lane
  p.emitUToR(6, 1)                                # R6 = float(lane)

  # copy LUT (device) -> threadgroup: lane i, i+32, ...
  p.emitUAddK(8, 1, 0)                            # U8 = lane (running pointer, += 32 per iteration)
  p.emitLoadConstF32(10, float32((lutN + GemvTg - 1) div GemvTg))   # iteration count (CONSTANT)
  p.emitLoadConstF32(12, float32(lutN))
  sessionVmForGpu(p, 10):
    p.emitUToR(11, 8)
    p.emitCmpLt(13, 11, 12)
    sessionVmIfGpu(p, 13):
      p.emitULoadConstF(9, 8, 0)                  # R9 = lut[U8]
      p.emitTgStoreF32(9, 11, 0)                  # tg[R11] = R9
    p.emitUAddK(8, 8, uint32(GemvTg))
  p.emitBarrierTg()

  if km and groupSize != 32:
    ## General group (64, 128, ...). quantizeFp4KM packing: a group of g elements = g/2 bytes,
    ## byte t holds element t (low nibble) and t+g/2 (high nibble). 32 lanes sweep a group in g/32 passes:
    ## pass s covers element s*32+lane -> first half (s < g/64): low nibble of byte s*32+lane;
    ## second half: high nibble of byte (s-g/64)*32+lane. Passes are unrolled on the host (build-time constants).
    let nG = K div groupSize
    let half = groupSize div 2
    let subs = groupSize div 32
    let subHalf = subs div 2
    p.emitUMulK(2, 0, uint32(K div 2))            # U2 = n*(K/2)
    p.emitUAdd(2, 2, 1)                           # + lane
    p.emitUMulK(3, 0, uint32(nG))                 # U3 = n*nG
    p.emitUAddK(3, 3, uint32(N * (K div 2)))      # scale ptr
    p.emitUAddK(5, 1, 0)                          # U5 = x ptr = lane
    p.emitLoadConstF32(5, float32(nG))
    p.emitForBegin(5)
    p.emitULoadU8(7, 3, 0)                        # group scale byte
    p.emitUToR(2, 7)
    p.emitTgLoadF32(2, 2, 16)                     # R2 = E4M3[scale]*gs
    for sub in 0 ..< subs:
      p.emitULoadU8(6, 2, uint32((sub mod subHalf) * 32))
      if sub >= subHalf: p.emitUShrK(6, 6, 4)
      p.emitUAndK(6, 6, 15)                       # nibble
      p.emitUToR(1, 6)
      p.emitTgLoadF32(1, 1, 0)                    # R1 = E2M1[nibble]
      p.emitULoadIn(3, 5, uint32(sub * 32))       # x[k]
      p.emitMul(1, 1, 2)
      p.emitFma(4, 1, 3, 4)                       # acc += w * x
    p.emitUAddK(2, 2, uint32(half))
    p.emitUAddK(3, 3, 1)
    p.emitUAddK(5, 5, uint32(groupSize))
    p.emitForEnd()
  elif km:
    let nG = K div groupSize
    p.emitUMulK(2, 0, uint32(K div 2))            # U2 = n*(K/2)
    p.emitUAndK(6, 1, 15)                         # U6 = lane & 15
    p.emitUAdd(2, 2, 6)                           # nibble ptr
    p.emitUMulK(3, 0, uint32(nG))                 # U3 = n*nG
    p.emitUAddK(3, 3, uint32(N * (K div 2)))      # scale ptr
    p.emitUShrK(4, 1, 4)                          # lane >> 4
    p.emitUShlK(4, 4, 2)                          # U4 = shift (0 or 4)
    p.emitUAddK(5, 1, 0)                          # U5 = x ptr = lane
    p.emitLoadConstF32(5, float32(nG))
    p.emitForBegin(5)
    p.emitULoadU8(6, 2, 0)                        # byte
    p.emitUShr(6, 6, 4)
    p.emitUAndK(6, 6, 15)                         # nibble
    p.emitUToR(1, 6)
    p.emitTgLoadF32(1, 1, 0)                      # R1 = E2M1[nibble]
    p.emitULoadU8(7, 3, 0)                        # scale byte
    p.emitUToR(2, 7)
    p.emitTgLoadF32(2, 2, 16)                     # R2 = E4M3[scale]*gs
    p.emitULoadIn(3, 5, 0)                        # x[k]
    p.emitMul(1, 1, 2)
    p.emitFma(4, 1, 3, 4)                         # acc += w * x
    p.emitUAddK(2, 2, 16)
    p.emitUAddK(3, 3, 1)
    p.emitUAddK(5, 5, 32)
    p.emitForEnd()
  else:
    p.emitUMulK(2, 0, uint32(K))
    p.emitUAdd(2, 2, 1)                           # byte ptr = n*K + lane
    p.emitUAddK(5, 1, 0)
    p.emitLoadConstF32(5, float32(K div GemvTg))
    p.emitForBegin(5)
    p.emitULoadU8(6, 2, 0)
    p.emitUToR(1, 6)
    p.emitTgLoadF32(1, 1, 0)
    p.emitULoadIn(3, 5, 0)
    p.emitFma(4, 1, 3, 4)
    p.emitUAddK(2, 2, 32)
    p.emitUAddK(5, 5, 32)
    p.emitForEnd()

  p.emitUToR(6, 1)                                # R6 = lane (reused for reduce)
  p.emitTgReduce32(4, lutN)
  p.emitLidIsZeroIf()
  p.emitTgLoadF32(9, 6, uint32(lutN))
  p.emitUStoreF32(9, 0, 0)                        # out[n]
  p.emitIfEnd()
  p.emitHalt()
  result = p

proc buildFp4KmGemv2RowProgram*(K, N: int): VmProgram =
  ## FP4-KM group-32 GEMV, two output rows/threadgroup.
  ## Borrowed from N_R0_MXFP4=2 in llama.cpp: each lane reuses x[k]
  ## for two weight rows; two independent reductions use two TG memory areas.
  ## No SIMD-width assumption: the reduction is still the 32-lane tree of the old kernel.
  if K <= 0 or K mod 32 != 0:
    raise newException(ValueError, "buildFp4KmGemv2RowProgram: K must be a positive multiple of 32")
  if N <= 0 or N mod 2 != 0:
    raise newException(ValueError, "buildFp4KmGemv2RowProgram: N must be positive and divisible by 2")
  let nG = K div 32
  var p: VmProgram
  p.emitUTgid(0)                                  # output row pair
  p.emitULid(1)                                   # lane 0..31
  p.emitUToR(6, 1)                                # R6 = lane (for reductions)

  # Cooperatively copy the 272-entry decode LUT into threadgroup memory.
  p.emitUAddK(8, 1, 0)
  p.emitLoadConstF32(10, float32((Fp4KmLutLen + GemvTg - 1) div GemvTg))
  p.emitLoadConstF32(12, float32(Fp4KmLutLen))
  sessionVmForGpu(p, 10):
    p.emitUToR(11, 8)
    p.emitCmpLt(13, 11, 12)
    sessionVmIfGpu(p, 13):
      p.emitULoadConstF(9, 8, 0)
      p.emitTgStoreF32(9, 11, 0)
    p.emitUAddK(8, 8, uint32(GemvTg))
  p.emitBarrierTg()

  # Pair's first output row; row 1 is exactly one row stride later.
  p.emitUMulK(2, 0, 2)                            # U2 = 2 * pair
  p.emitUAndK(3, 1, 15)                           # byte within packed 32-value group
  p.emitUShrK(4, 1, 4); p.emitUShlK(4, 4, 2)      # nibble shift: 0 or 4
  p.emitUMulK(5, 2, uint32(K div 2))              # row0 packed-weight base
  p.emitUAdd(5, 5, 3)
  p.emitUAddK(8, 5, uint32(K div 2))              # row1 packed-weight base
  p.emitUMulK(6, 2, uint32(nG))                   # row0 scale base
  p.emitUAddK(6, 6, uint32(N * (K div 2)))        # scales follow all packed row bytes
  p.emitUAddK(9, 6, uint32(nG))                   # row1 scale base
  p.emitUAddK(7, 1, 0)                            # x index
  p.emitLoadConstF32(4, 0.0'f32)                  # row0 accumulator
  p.emitLoadConstF32(12, 0.0'f32)                 # row1 accumulator
  if nG mod 2 == 0:
    # ── PRELOAD variant, 2 groups/iteration (from llama.cpp: many independent loads in flight) ─────────
    # This kernel is bound by memory LATENCY (measured ~3.6 GB/s = Little's law with ~170
    # hardware threads), so each thread needs more bytes "in flight". Each iteration issues ALL loads
    # (code+scale of 2 rows x 2 groups, and x of 2 groups) BEFORE decoding. The summation order is unchanged
    # (even group then odd group) => bit-identical to the old version.
    # U: A = U10 code0, U11 scale0, U13 code1, U14 scale1 | B = U3 code0, U12 scale0, U15 code1,
    #    B's scale1 is loaded into U10 once group A is done. R3 = x of group A, R9 = x of group B.
    p.emitLoadConstF32(5, float32(nG div 2))      # iteration count (R5 unchanged in the loop)
    p.emitForBegin(5)
    p.emitULoadU8(10, 5, 0)
    p.emitULoadU8(11, 6, 0)
    p.emitULoadU8(13, 8, 0)
    p.emitULoadU8(14, 9, 0)
    p.emitULoadU8(3, 5, 16)
    p.emitULoadU8(12, 6, 1)
    p.emitULoadU8(15, 8, 16)
    p.emitULoadIn(3, 7, 0)
    p.emitULoadIn(9, 7, 32)
    # group A, row 0
    p.emitUShr(10, 10, 4); p.emitUAndK(10, 10, 15)
    p.emitUToR(1, 10); p.emitTgLoadF32(1, 1, 0)
    p.emitUToR(2, 11); p.emitTgLoadF32(2, 2, 16)
    p.emitMul(1, 1, 2); p.emitFma(4, 1, 3, 4)
    # group A, row 1
    p.emitUShr(13, 13, 4); p.emitUAndK(13, 13, 15)
    p.emitUToR(7, 13); p.emitTgLoadF32(7, 7, 0)
    p.emitUToR(8, 14); p.emitTgLoadF32(8, 8, 16)
    p.emitMul(7, 7, 8); p.emitFma(12, 7, 3, 12)
    p.emitULoadU8(10, 9, 1)                       # row 1 scale of group B (U10 is free now)
    # group B, row 0
    p.emitUShr(3, 3, 4); p.emitUAndK(3, 3, 15)
    p.emitUToR(1, 3); p.emitTgLoadF32(1, 1, 0)
    p.emitUToR(2, 12); p.emitTgLoadF32(2, 2, 16)
    p.emitMul(1, 1, 2); p.emitFma(4, 1, 9, 4)
    # group B, row 1
    p.emitUShr(15, 15, 4); p.emitUAndK(15, 15, 15)
    p.emitUToR(7, 15); p.emitTgLoadF32(7, 7, 0)
    p.emitUToR(8, 10); p.emitTgLoadF32(8, 8, 16)
    p.emitMul(7, 7, 8); p.emitFma(12, 7, 9, 12)
    p.emitUAddK(5, 5, 32); p.emitUAddK(8, 8, 32)
    p.emitUAddK(6, 6, 2); p.emitUAddK(9, 9, 2); p.emitUAddK(7, 7, 64)
    p.emitForEnd()
  else:
    p.emitLoadConstF32(5, float32(nG))              # loop count (must stay unchanged in loop)
    p.emitForBegin(5)
    # Load ALL FOUR bytes (code+scale of both rows) BEFORE decoding. This is
    # a latency-bound kernel: 1 load/lane leaves memory idle ~90%. These four
    # independent loads put 4 loads in flight and change NO arithmetic
    # (each lane's summation order is unchanged => bit-identical result).
    # R10/R11 = code+scale of row 0; R13/R14 = code+scale of row 1.
    # NEVER use R12: R12 is row 1's accumulator (initialised with
    # `LoadConstF32(12, 0.0)` before the loop, then `Fma(12, ...)`); loading a byte
    # into R12 would destroy the accumulator on the first iteration.
    p.emitULoadU8(10, 5, 0)                         # row0 code byte
    p.emitULoadU8(11, 6, 0)                         # row0 group scale byte
    p.emitULoadU8(13, 8, 0)                         # row1 code byte
    p.emitULoadU8(14, 9, 0)                         # row1 group scale byte
    # Row 0: decode one nibble and scale, then multiply by the shared x lane.
    p.emitUShr(10, 10, 4); p.emitUAndK(10, 10, 15)
    p.emitUToR(1, 10); p.emitTgLoadF32(1, 1, 0)
    p.emitUToR(2, 11); p.emitTgLoadF32(2, 2, 16)
    p.emitULoadIn(3, 7, 0); p.emitMul(1, 1, 2); p.emitFma(4, 1, 3, 4)

    # Row 1 reuses x, but has its own packed codes and E4M3 group scale.
    p.emitUShr(13, 13, 4); p.emitUAndK(13, 13, 15)
    p.emitUToR(7, 13); p.emitTgLoadF32(7, 7, 0)
    p.emitUToR(8, 14); p.emitTgLoadF32(8, 8, 16)
    p.emitMul(7, 7, 8); p.emitFma(12, 7, 3, 12)

    p.emitUAddK(5, 5, 16); p.emitUAddK(8, 8, 16)
    p.emitUAddK(6, 6, 1); p.emitUAddK(9, 9, 1); p.emitUAddK(7, 7, 32)
    p.emitForEnd()

  p.emitUToR(6, 1)                                # reduction helper expects lane in R6
  p.emitTgReduce32(4, Fp4KmLutLen)
  p.emitTgReduce32(12, Fp4KmLutLen + GemvTg)
  p.emitLidIsZeroIf()
  p.emitTgLoadF32(9, 6, uint32(Fp4KmLutLen))
  p.emitUStoreF32(9, 2, 0)
  p.emitTgLoadF32(9, 6, uint32(Fp4KmLutLen + GemvTg))
  p.emitUAddK(3, 2, 1)
  p.emitUStoreF32(9, 3, 0)
  p.emitIfEnd()
  p.emitHalt()
  result = p

# ── fp4-KM GEMV kernel variants (selected with BYBY_GEMV_VARIANT, see gpu_layers.nim) ────────
# Same weight/LUT/scale layout as buildFp4KmGemv2RowProgram -> shares GPU buffers
# with the prefill GEMM kernel. Layout of one group of 32 values = 16 bytes: the LOW nibble of byte b is
# value b, the HIGH nibble of byte b is value 16+b.

proc emitKmLutCopy(p: var VmProgram) =
  ## Copies the 272-float LUT into tgF32[0..272) (32 lanes cooperate). Uses R10,R11,R12,R13,U8,U9.
  p.emitUAddK(8, 1, 0)
  p.emitLoadConstF32(10, float32((Fp4KmLutLen + GemvTg - 1) div GemvTg))
  p.emitLoadConstF32(12, float32(Fp4KmLutLen))
  sessionVmForGpu(p, 10):
    p.emitUToR(11, 8)
    p.emitCmpLt(13, 11, 12)
    sessionVmIfGpu(p, 13):
      p.emitULoadConstF(9, 8, 0)
      p.emitTgStoreF32(9, 11, 0)
    p.emitUAddK(8, 8, uint32(GemvTg))
  p.emitBarrierTg()

proc fp4KmGemvRowsTgBytes*(rows: int): int = (Fp4KmLutLen + rows * GemvTg) * 4

proc fp4KmGemvWideSupported*(K, N: int): bool =
  K > 0 and K mod 256 == 0 and N > 0 and N mod 2 == 0

proc buildFp4KmGemvWideProgram*(K, N: int): VmProgram =
  ## 2 rows/threadgroup. Each lane loads ONE 32-bit WORD (8 values) per row; 4 lanes = 1 group
  ## of 32 values, 32 lanes = 8 groups / iteration. The E4M3 scale is looked up ONCE / group (old kernel: per value).
  ## Global/SLM load count drops ~3x. Per-group sum: acc += scale * sum(lut[nibble] * x).
  if not fp4KmGemvWideSupported(K, N):
    raise newException(ValueError, "buildFp4KmGemvWideProgram: needs K divisible by 256, N even")
  let nG = K div 32
  let nIter = K div 256
  var p: VmProgram
  p.emitUTgid(0)
  p.emitULid(1)
  p.emitUToR(6, 1)
  p.emitKmLutCopy()
  p.emitUMulK(2, 0, 2)                              # U2 = first row of the pair
  p.emitUShrK(3, 1, 2)                              # U3 = gi = lane/4 (group in the iteration)
  p.emitUAndK(4, 1, 3)                              # U4 = wi = lane%4 (word in the group)
  p.emitUMulK(5, 2, uint32(K div 2))                # row 0 code ptr
  p.emitUShlK(10, 3, 4); p.emitUAdd(5, 5, 10)       # + gi*16
  p.emitUShlK(10, 4, 2); p.emitUAdd(5, 5, 10)       # + wi*4
  p.emitUMulK(6, 2, uint32(nG))                     # row 0 scale ptr
  p.emitUAddK(6, 6, uint32(N * (K div 2)))
  p.emitUAdd(6, 6, 3)                               # + gi
  p.emitUShlK(7, 3, 5)                              # x ptr = gi*32
  p.emitUShlK(10, 4, 2); p.emitUAdd(7, 7, 10)       # + wi*4
  p.emitLoadConstF32(4, 0.0'f32)                    # row 0 acc
  p.emitLoadConstF32(12, 0.0'f32)                   # row 1 acc
  p.emitLoadConstF32(5, float32(nIter))
  p.emitForBegin(5)
  p.emitLoadConstF32(13, 0.0'f32)                   # row 0 group sum
  p.emitLoadConstF32(14, 0.0'f32)                   # row 1 group sum
  p.emitULoadU32(8, 5, 0)                           # 8 nibbles of row 0
  p.emitULoadU32(9, 5, uint32(K div 2))             # 8 nibbles of row 1
  p.emitULoadU8(10, 6, 0)                           # row 0 scale
  p.emitULoadU8(11, 6, uint32(nG))                  # row 1 scale
  for t in 0 ..< 4:
    for hi in 0 ..< 2:
      let sh = uint32(8 * t + 4 * hi)
      let xoff = uint32(t + 16 * hi)
      p.emitUShrK(12, 8, sh); p.emitUAndK(12, 12, 15)
      p.emitUToR(1, 12); p.emitTgLoadF32(1, 1, 0)
      p.emitUShrK(13, 9, sh)                        # U13 = row 1 nibble
      p.emitUAndK(13, 13, 15)
      p.emitUToR(2, 13); p.emitTgLoadF32(2, 2, 0)
      p.emitULoadIn(3, 7, xoff)
      p.emitFma(13, 1, 3, 13)
      p.emitFma(14, 2, 3, 14)
  p.emitUToR(10, 10); p.emitTgLoadF32(10, 10, 16)   # row 0 scale
  p.emitUToR(11, 11); p.emitTgLoadF32(11, 11, 16)   # row 1 scale
  p.emitFma(4, 10, 13, 4)
  p.emitFma(12, 11, 14, 12)
  p.emitUAddK(5, 5, 128); p.emitUAddK(6, 6, 8); p.emitUAddK(7, 7, 256)
  p.emitForEnd()
  p.emitUToR(6, 1)
  p.emitTgReduce32(4, Fp4KmLutLen)
  p.emitTgReduce32(12, Fp4KmLutLen + GemvTg)
  p.emitLidIsZeroIf()
  p.emitTgLoadF32(9, 6, uint32(Fp4KmLutLen))
  p.emitUStoreF32(9, 2, 0)
  p.emitTgLoadF32(9, 6, uint32(Fp4KmLutLen + GemvTg))
  p.emitUAddK(3, 2, 1)
  p.emitUStoreF32(9, 3, 0)
  p.emitIfEnd()
  p.emitHalt()
  result = p

proc buildFp4KmGemv4RowProgram*(K, N: int): VmProgram =
  ## 4 rows/threadgroup, same lane mapping as the 2-row kernel (x loaded once for 4 rows; LUT
  ## and barriers amortised over 4 rows). Loads 8 bytes (code+scale of 4 rows) BEFORE decoding.
  if K <= 0 or K mod 32 != 0 or N <= 0 or N mod 4 != 0:
    raise newException(ValueError, "buildFp4KmGemv4RowProgram: K chia het 32, N chia het 4")
  let nG = K div 32
  const accR = [4'u32, 12'u32, 13'u32, 14'u32]
  var p: VmProgram
  p.emitUTgid(0)
  p.emitULid(1)
  p.emitUToR(6, 1)
  p.emitKmLutCopy()
  p.emitUMulK(2, 0, 4)                              # U2 = first row of the group of 4
  p.emitUAndK(3, 1, 15)
  p.emitUShrK(4, 1, 4); p.emitUShlK(4, 4, 2)        # U4 = shift nibble 0|4
  p.emitUMulK(5, 2, uint32(K div 2)); p.emitUAdd(5, 5, 3)
  p.emitUMulK(6, 2, uint32(nG)); p.emitUAddK(6, 6, uint32(N * (K div 2)))
  p.emitUAddK(7, 1, 0)                              # x ptr = lane
  for r in 0 ..< 4: p.emitLoadConstF32(accR[r], 0.0'f32)
  p.emitLoadConstF32(5, float32(nG))
  p.emitForBegin(5)
  for r in 0 ..< 4:
    p.emitUAddK(3, 5, uint32(r * (K div 2)));  p.emitULoadU8(uint32(8 + r), 3, 0)
    p.emitUAddK(3, 6, uint32(r * nG));         p.emitULoadU8(uint32(12 + r), 3, 0)
  p.emitULoadIn(3, 7, 0)
  for r in 0 ..< 4:
    let cu = uint32(8 + r)
    p.emitUShr(cu, cu, 4); p.emitUAndK(cu, cu, 15)
    p.emitUToR(1, cu); p.emitTgLoadF32(1, 1, 0)
    p.emitUToR(2, uint32(12 + r)); p.emitTgLoadF32(2, 2, 16)
    p.emitMul(1, 1, 2); p.emitFma(accR[r], 1, 3, accR[r])
  p.emitUAddK(5, 5, 16); p.emitUAddK(6, 6, 1); p.emitUAddK(7, 7, 32)
  p.emitForEnd()
  p.emitUToR(6, 1)
  for r in 0 ..< 4: p.emitTgReduce32(accR[r], Fp4KmLutLen + r * GemvTg)
  p.emitLidIsZeroIf()
  for r in 0 ..< 4:
    p.emitTgLoadF32(9, 6, uint32(Fp4KmLutLen + r * GemvTg))
    p.emitUAddK(3, 2, uint32(r))
    p.emitUStoreF32(9, 3, 0)
  p.emitIfEnd()
  p.emitHalt()
  result = p

proc buildFp4KmGemmM8Program*(K, N: int): VmProgram =
  ## Eight input rows share one decoded weight. Save all eight accumulators
  ## before reducing them together, using just one barrier per tree level.
  doAssert K > 0 and K mod 32 == 0 and N > 0
  const accum = [4'u32, 6'u32, 10'u32, 11'u32, 12'u32, 13'u32, 14'u32, 15'u32]
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitKmLutCopy()
  p.emitUMulK(2, 0, uint32(K div 2))
  p.emitUAndK(6, 1, 15); p.emitUAdd(2, 2, 6)
  p.emitUMulK(3, 0, uint32(K div 32))
  p.emitUAddK(3, 3, uint32(N * (K div 2)))
  p.emitUShrK(4, 1, 4); p.emitUShlK(4, 4, 2)
  p.emitUAddK(5, 1, 0)
  for r in accum: p.emitLoadConstF32(r, 0)
  p.emitLoadConstF32(5, float32(K div 32))
  sessionVmForGpu(p, 5):
    p.emitULoadU8(6, 2); p.emitUShr(6, 6, 4); p.emitUAndK(6, 6, 15)
    p.emitUToR(1, 6); p.emitTgLoadF32(1, 1, 0)
    p.emitULoadU8(7, 3); p.emitUToR(2, 7); p.emitTgLoadF32(2, 2, 16)
    p.emitMul(1, 1, 2)
    for m in 0 ..< 8:
      p.emitULoadIn(3, 5, uint32(m * K))
      p.emitFma(accum[m], 1, 3, accum[m])
    p.emitUAddK(2, 2, 16); p.emitUAddK(3, 3, 1); p.emitUAddK(5, 5, 32)
  p.emitUToR(7, 1)
  for m in 0 ..< 8: p.emitTgStoreF32(accum[m], 7, uint32(Fp4KmLutLen + m * GemvTg))
  p.emitBarrierTg()
  p.emitLoadConstF32(5, 5); p.emitLoadConstF32(8, 16)
  sessionVmForGpu(p, 5):
    p.emitCmpLt(9, 7, 8)
    sessionVmIfGpu(p, 9):
      p.emitAdd(10, 7, 8)
      for m in 0 ..< 8:
        p.emitTgLoadF32(1, 7, uint32(Fp4KmLutLen + m * GemvTg))
        p.emitTgLoadF32(2, 10, uint32(Fp4KmLutLen + m * GemvTg))
        p.emitAdd(1, 1, 2); p.emitTgStoreF32(1, 7, uint32(Fp4KmLutLen + m * GemvTg))
    p.emitBarrierTg()
    p.emitMulConst(8, 8, 0.5'f32)
  p.emitLoadConstF32(8, 1); p.emitCmpLt(9, 7, 8)
  sessionVmIfGpu(p, 9):
    for m in 0 ..< 8:
      p.emitTgLoadF32(1, 7, uint32(Fp4KmLutLen + m * GemvTg))
      p.emitUStoreF32(1, 0, uint32(m * N))
  p.emitHalt()
  p

proc emitKmLutCopyDynamic(p: var VmProgram; tg: int) =
  p.emitUAddK(8, 1, 0)
  p.emitLoadConstF32(10, float32((Fp4KmLutLen + tg - 1) div tg))
  p.emitLoadConstF32(12, float32(Fp4KmLutLen))
  sessionVmForGpu(p, 10):
    p.emitUToR(11, 8); p.emitCmpLt(13, 11, 12)
    sessionVmIfGpu(p, 13):
      p.emitULoadConstF(9, 8); p.emitTgStoreF32(9, 11, 0)
    p.emitUAddK(8, 8, uint32(tg))
  p.emitBarrierTg()

proc emitKmReduceDynamic(p: var VmProgram; accum: openArray[uint32]; tg: int;
                         batched: bool; N: int) =
  ## All rows share the barrier at each tree level. Every lane participates,
  ## including Intel GPUs whose SIMD width is smaller than the workgroup.
  p.emitUToR(7, 1)
  for r, reg in accum: p.emitTgStoreF32(reg, 7, uint32(Fp4KmLutLen + r * tg))
  p.emitBarrierTg()
  p.emitLoadConstF32(5, float32(fastLog2(tg)))
  p.emitLoadConstF32(8, float32(tg div 2))
  sessionVmForGpu(p, 5):
    p.emitCmpLt(9, 7, 8)
    sessionVmIfGpu(p, 9):
      p.emitAdd(10, 7, 8)
      for r in 0 ..< accum.len:
        p.emitTgLoadF32(1, 7, uint32(Fp4KmLutLen + r * tg))
        p.emitTgLoadF32(2, 10, uint32(Fp4KmLutLen + r * tg))
        p.emitAdd(1, 1, 2); p.emitTgStoreF32(1, 7, uint32(Fp4KmLutLen + r * tg))
    p.emitBarrierTg(); p.emitMulConst(8, 8, 0.5'f32)
  p.emitLoadConstF32(8, 1); p.emitCmpLt(9, 7, 8)
  sessionVmIfGpu(p, 9):
    for r in 0 ..< accum.len:
      p.emitTgLoadF32(1, 7, uint32(Fp4KmLutLen + r * tg))
      if batched: p.emitUStoreF32(1, 0, uint32(r * N))
      else: p.emitUStoreF32(1, 2, uint32(r))
  p.emitHalt()

proc buildFp4KmGemvTunedProgram*(K, N, tg, rows: int): VmProgram =
  ## Four-byte packed loads cover eight weights/lane. Wider groups split K
  ## over more lanes; rows reuse the activation. No weight repacking required.
  doAssert tg in [32, 64, 128, 256, 512, 1024] and rows in [2, 4]
  doAssert K > 0 and K mod (tg * 8) == 0 and N > 0 and N mod rows == 0
  const acc = [4'u32, 12'u32, 13'u32, 14'u32]
  const partial = [8'u32, 9'u32, 10'u32, 11'u32]
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1); p.emitKmLutCopyDynamic(tg)
  p.emitUMulK(2, 0, uint32(rows))
  p.emitUShrK(3, 1, 2); p.emitUAndK(4, 1, 3)
  p.emitUMulK(5, 2, uint32(K div 2))
  p.emitUShlK(10, 3, 4); p.emitUAdd(5, 5, 10)
  p.emitUShlK(10, 4, 2); p.emitUAdd(5, 5, 10)
  p.emitUMulK(6, 2, uint32(K div 32))
  p.emitUAddK(6, 6, uint32(N * (K div 2))); p.emitUAdd(6, 6, 3)
  p.emitUShlK(7, 3, 5); p.emitUAdd(7, 7, 10)
  for r in 0 ..< rows: p.emitLoadConstF32(acc[r], 0)
  p.emitLoadConstF32(5, float32(K div (tg * 8)))
  sessionVmForGpu(p, 5):
    for r in 0 ..< rows:
      p.emitULoadU32(uint32(8 + r), 5, uint32(r * (K div 2)))
      p.emitULoadU8(uint32(12 + r), 6, uint32(r * (K div 32)))
      p.emitLoadConstF32(partial[r], 0)
    for t in 0 ..< 4:
      for hi in 0 ..< 2:
        p.emitULoadIn(3, 7, uint32(t + 16 * hi))
        for r in 0 ..< rows:
          p.emitUShrK(3, uint32(8 + r), uint32(8 * t + 4 * hi)); p.emitUAndK(3, 3, 15)
          p.emitUToR(1, 3); p.emitTgLoadF32(1, 1, 0)
          p.emitFma(partial[r], 1, 3, partial[r])
    for r in 0 ..< rows:
      p.emitUToR(2, uint32(12 + r)); p.emitTgLoadF32(2, 2, 16)
      p.emitFma(acc[r], 2, partial[r], acc[r])
    p.emitUAddK(5, 5, uint32(tg * 4)); p.emitUAddK(6, 6, uint32(tg div 4))
    p.emitUAddK(7, 7, uint32(tg * 8))
  p.emitKmReduceDynamic(acc.toOpenArray(0, rows - 1), tg, false, N)
  p

proc buildFp4KmGemmTunedProgram*(K, N, tg, batchRows: int): VmProgram =
  ## Batched variant: decode one packed word/scale for all input rows.
  doAssert tg in [32, 64, 128, 256, 512, 1024] and batchRows in [4, 8]
  doAssert K > 0 and K mod (tg * 8) == 0 and N > 0
  const acc = [4'u32, 6'u32, 10'u32, 11'u32, 12'u32, 13'u32, 14'u32, 15'u32]
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1); p.emitKmLutCopyDynamic(tg)
  p.emitUShrK(3, 1, 2); p.emitUAndK(4, 1, 3)
  p.emitUMulK(2, 0, uint32(K div 2))
  p.emitUShlK(6, 3, 4); p.emitUAdd(2, 2, 6)
  p.emitUShlK(6, 4, 2); p.emitUAdd(2, 2, 6)
  p.emitUShlK(5, 3, 5); p.emitUAdd(5, 5, 6)
  p.emitUMulK(3, 0, uint32(K div 32))
  p.emitUAddK(3, 3, uint32(N * (K div 2)))
  p.emitUShrK(6, 1, 2); p.emitUAdd(3, 3, 6)
  for r in 0 ..< batchRows: p.emitLoadConstF32(acc[r], 0)
  p.emitLoadConstF32(5, float32(K div (tg * 8)))
  sessionVmForGpu(p, 5):
    p.emitULoadU32(8, 2); p.emitULoadU8(9, 3)
    p.emitUToR(2, 9); p.emitTgLoadF32(2, 2, 16)
    for t in 0 ..< 4:
      for hi in 0 ..< 2:
        p.emitUShrK(6, 8, uint32(8 * t + 4 * hi)); p.emitUAndK(6, 6, 15)
        p.emitUToR(1, 6); p.emitTgLoadF32(1, 1, 0); p.emitMul(1, 1, 2)
        for r in 0 ..< batchRows:
          p.emitULoadIn(3, 5, uint32(r * K + t + 16 * hi))
          p.emitFma(acc[r], 1, 3, acc[r])
    p.emitUAddK(2, 2, uint32(tg * 4)); p.emitUAddK(3, 3, uint32(tg div 4))
    p.emitUAddK(5, 5, uint32(tg * 8))
  p.emitKmReduceDynamic(acc.toOpenArray(0, batchRows - 1), tg, true, N)
  p

proc buildFp4KmGemmUnrollProgram*(K, N, tg, unroll, rows: int): VmProgram =
  ## Batched FP4-KM tile that keeps `unroll` independent weight words in flight.
  ## The tuned tile issues one u32 load -> 8 LUT loads -> rows*8 FMAs as a single
  ## serial chain per iteration, so every iteration waits a full global-memory
  ## latency before any FMA starts (measured 17-19 GMAC/s, i.e. 4-6% of this
  ## iGPU's FMA peak - batching more rows does not help). Issuing several words
  ## up front lets each word's latency hide under the previous word's FMAs:
  ## measured 1.5x with unroll=2 (unroll=4 loses it again to register pressure).
  ## Same math and same per-accumulator order as buildFp4KmGemmTunedProgram,
  ## so the result is bit-identical (checked by bench_fp4_gemm_tg).
  doAssert unroll in [2, 4] and tg in [32, 64, 128] and rows in [4, 8]
  doAssert K > 0 and K mod (tg * 8 * unroll) == 0 and N > 0
  const accAll = [4'u32, 6'u32, 10'u32, 11'u32, 12'u32, 13'u32, 14'u32, 15'u32]
  const wordReg = [8'u32, 9'u32, 10'u32, 11'u32]      # U: packed weight word
  const byteReg = [12'u32, 13'u32, 14'u32, 15'u32]    # U: group-scale byte
  const scaleReg = [0'u32, 1'u32, 2'u32, 7'u32]       # R: decoded group scale
  var acc: seq[uint32]
  for r in 0 ..< rows: acc.add accAll[r]
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1); p.emitKmLutCopyDynamic(tg)
  p.emitUShrK(3, 1, 2); p.emitUAndK(4, 1, 3)
  p.emitUMulK(2, 0, uint32(K div 2))
  p.emitUShlK(6, 3, 4); p.emitUAdd(2, 2, 6)
  p.emitUShlK(6, 4, 2); p.emitUAdd(2, 2, 6)
  p.emitUShlK(5, 3, 5); p.emitUAdd(5, 5, 6)
  p.emitUMulK(3, 0, uint32(K div 32))
  p.emitUAddK(3, 3, uint32(N * (K div 2)))
  p.emitUShrK(6, 1, 2); p.emitUAdd(3, 3, 6)
  for r in acc: p.emitLoadConstF32(r, 0)
  p.emitLoadConstF32(5, float32(K div (tg * 8 * unroll)))
  sessionVmForGpu(p, 5):
    for u in 0 ..< unroll:
      p.emitULoadU32(wordReg[u], 2, uint32(u * tg * 4))
      p.emitULoadU8(byteReg[u], 3, uint32(u * (tg div 4)))
      p.emitUToR(scaleReg[u], byteReg[u]); p.emitTgLoadF32(scaleReg[u], scaleReg[u], 16)
    for u in 0 ..< unroll:
      for t in 0 ..< 4:
        for hi in 0 ..< 2:
          p.emitUShrK(6, wordReg[u], uint32(8 * t + 4 * hi)); p.emitUAndK(6, 6, 15)
          p.emitUToR(3, 6); p.emitTgLoadF32(3, 3, 0); p.emitMul(3, 3, scaleReg[u])
          for row in 0 ..< rows:
            p.emitULoadIn(8, 5, uint32(row * K + t + 16 * hi + u * (tg * 8)))
            p.emitFma(acc[row], 3, 8, acc[row])
    p.emitUAddK(2, 2, uint32(tg * 8 * unroll div 2))
    p.emitUAddK(3, 3, uint32((tg div 4) * unroll))
    p.emitUAddK(5, 5, uint32(tg * 8 * unroll))
  p.emitKmReduceDynamic(acc.toOpenArray(0, rows - 1), tg, true, N)
  p

proc f32GemmV4TgBytes*(rows, tg: int): int = max((Fp4KmLutLen + rows * tg) * 4, 16)

proc buildFp4KmGemmVecProgram*(K, N, tg, rows: int): VmProgram =
  ## Batched FP4-KM tile whose input loads are 16-byte vector loads.
  ## After the two-word unroll the tile is load-issue limited: every MAC costs
  ## one scalar `inF32[...]` load, i.e. 64 loads per 8 weights for 8 rows, and
  ## the 8 k values a lane owns sit in two aligned 4-float groups ({0..3} and
  ## {16..19} relative to the lane base) - exactly two float4 loads per row.
  ## That is 16 loads instead of 64 for the same 64 MACs.
  ## The 16-float register file forces rows=6 here: 6 accumulators + 4 decoded
  ## weights + the 4-element x group + scale + loop counter = 16. A 7th
  ## accumulator would push the x group out of a 4-aligned register window.
  ## Rounding differs from buildFp4KmGemmTunedProgram (weights are decoded per
  ## 4-wide group instead of interleaved), same arithmetic, rel err < 1e-6.
  doAssert tg in [32, 64, 128] and rows in 1 .. 6
  doAssert K > 0 and K mod (tg * 8) == 0 and N > 0
  const accAll = [4'u32, 6'u32, 10'u32, 11'u32, 12'u32, 13'u32]
  const wreg = [7'u32, 8'u32, 14'u32, 15'u32]   # 4 decoded weights
  const xreg = 0'u32                            # 4-aligned x group: R0..R3
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1); p.emitKmLutCopyDynamic(tg)
  p.emitUShrK(3, 1, 2); p.emitUAndK(4, 1, 3)
  p.emitUMulK(2, 0, uint32(K div 2))
  p.emitUShlK(6, 3, 4); p.emitUAdd(2, 2, 6)
  p.emitUShlK(6, 4, 2); p.emitUAdd(2, 2, 6)
  p.emitUShlK(5, 3, 5); p.emitUAdd(5, 5, 6)
  p.emitUMulK(3, 0, uint32(K div 32))
  p.emitUAddK(3, 3, uint32(N * (K div 2)))
  p.emitUShrK(6, 1, 2); p.emitUAdd(3, 3, 6)
  for r in 0 ..< rows: p.emitLoadConstF32(accAll[r], 0)
  p.emitLoadConstF32(5, float32(K div (tg * 8)))
  sessionVmForGpu(p, 5):
    p.emitULoadU32(8, 2)                             # U8 = packed weight word
    p.emitULoadU8(9, 3)
    p.emitUToR(9, 9); p.emitTgLoadF32(9, 9, 16)      # R9 = group scale
    for hi in 0 ..< 2:
      # decode the 4 weights of this k group once, reuse for all rows
      for t in 0 ..< 4:
        p.emitUShrK(6, 8, uint32(8 * t + 4 * hi)); p.emitUAndK(6, 6, 15)
        p.emitUToR(wreg[t], 6); p.emitTgLoadF32(wreg[t], wreg[t], 0)
        p.emitMul(wreg[t], wreg[t], 9)
      for row in 0 ..< rows:
        p.emitULoadIn4(xreg, 5, uint32(row * K + hi * 16))
        for t in 0 ..< 4:
          p.emitFma(accAll[row], wreg[t], xreg + uint32(t), accAll[row])
    p.emitUAddK(2, 2, uint32(tg * 4))
    p.emitUAddK(3, 3, uint32(tg div 4))
    p.emitUAddK(5, 5, uint32(tg * 8))
  p.emitKmReduceDynamic(accAll.toOpenArray(0, rows - 1), tg, true, N)
  p

proc buildFp4KmGemm2ColProgram*(K, N, tg, rows: int): VmProgram =
  ## Batched FP4-KM tile: `rows` input rows x 2 output columns per threadgroup.
  ## One float4 input load feeds the same 4 k values to both columns, halving
  ## the input-load count per MAC - the dominant cost after the two-word unroll.
  ## Register layout (needs the 32-register file):
  ##   acc[rows][2] = R8..R(8+2*rows-1)   x float4 = R0..R3   wA = R4..R7
  ##   wB = R(8+2*rows)..  scaleA/B, counter follow.
  ## Same arithmetic as buildFp4KmGemmTunedProgram; rounding differs only in
  ## the nibble decode order, so compare numerically (rel err < 1e-6).
  doAssert tg in [32, 64, 128] and rows in [4, 8]
  doAssert K > 0 and K mod (tg * 8) == 0 and N > 0 and N mod 2 == 0
  let nAcc = rows * 2
  let wB = uint32(8 + nAcc)              # 4 weight regs for column B
  let sA = uint32(12 + nAcc)             # scale A
  let sB = uint32(13 + nAcc)             # scale B
  let cnt = uint32(14 + nAcc)            # loop counter
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1); p.emitKmLutCopyDynamic(tg)
  p.emitUShrK(3, 1, 2); p.emitUAndK(4, 1, 3)          # U3=lane>>2, U4=lane&3
  p.emitUMulK(12, 0, 2)                               # U12 = 2*tgid
  p.emitUMulK(2, 0, uint32(K))                        # U2 = (2*tgid)*K/2 bytes (column A)
  p.emitUShlK(6, 3, 4); p.emitUAdd(2, 2, 6)
  p.emitUShlK(6, 4, 2); p.emitUAdd(2, 2, 6)           # + (lane>>2)*16 + (lane&3)*4
  p.emitUAddK(7, 2, uint32(K div 2))                  # U7 = column B weight ptr
  p.emitUShlK(5, 3, 5); p.emitUAdd(5, 5, 6)           # U5 = x offset (lane)
  p.emitUMulK(3, 0, uint32(K div 16))                 # U3 = (2*tgid)*K/32 (scale A)
  p.emitUAddK(3, 3, uint32(N * (K div 2)))
  p.emitUShrK(6, 1, 2); p.emitUAdd(3, 3, 6)           # + lane>>2
  p.emitUAddK(11, 3, uint32(K div 32))                # U11 = column B scale ptr
  for r in 0 ..< nAcc: p.emitLoadConstF32(uint32(8 + r), 0)
  p.emitLoadConstF32(cnt, float32(K div (tg * 8)))
  sessionVmForGpu(p, cnt):
    p.emitULoadU32(8, 2); p.emitULoadU32(9, 7)        # word A, word B
    p.emitULoadU8(10, 3)                              # scale byte A
    p.emitUToR(sA, 10); p.emitTgLoadF32(sA, sA, 16)
    p.emitULoadU8(10, 11)                             # scale byte B
    p.emitUToR(sB, 10); p.emitTgLoadF32(sB, sB, 16)
    for hi in 0 ..< 2:
      for t in 0 ..< 4:
        p.emitUShrK(6, 8, uint32(8 * t + 4 * hi)); p.emitUAndK(6, 6, 15)
        p.emitUToR(4 + uint32(t), 6); p.emitTgLoadF32(4 + uint32(t), 4 + uint32(t), 0)
        p.emitMul(4 + uint32(t), 4 + uint32(t), sA)
        p.emitUShrK(6, 9, uint32(8 * t + 4 * hi)); p.emitUAndK(6, 6, 15)
        p.emitUToR(wB + uint32(t), 6); p.emitTgLoadF32(wB + uint32(t), wB + uint32(t), 0)
        p.emitMul(wB + uint32(t), wB + uint32(t), sB)
      for row in 0 ..< rows:
        p.emitULoadIn4(0, 5, uint32(row * K + hi * 16))
        for t in 0 ..< 4:
          p.emitFma(uint32(8 + row * 2), 4 + uint32(t), uint32(t), uint32(8 + row * 2))
          p.emitFma(uint32(9 + row * 2), wB + uint32(t), uint32(t), uint32(9 + row * 2))
    p.emitUAddK(2, 2, uint32(tg * 4)); p.emitUAddK(7, 7, uint32(tg * 4))
    p.emitUAddK(3, 3, uint32(tg div 4)); p.emitUAddK(11, 11, uint32(tg div 4))
    p.emitUAddK(5, 5, uint32(tg * 8))
  # reduce nAcc accs across tg lanes, then write 2 columns
  p.emitUToR(0, 1)                                    # R0 = lane
  for r in 0 ..< nAcc: p.emitTgStoreF32(uint32(8 + r), 0, uint32(Fp4KmLutLen + r * tg))
  p.emitBarrierTg()
  p.emitLoadConstF32(1, float32(fastLog2(tg)))
  p.emitLoadConstF32(2, float32(tg div 2))
  sessionVmForGpu(p, 1):
    p.emitCmpLt(3, 0, 2)
    sessionVmIfGpu(p, 3):
      p.emitAdd(4, 0, 2)
      for r in 0 ..< nAcc:
        p.emitTgLoadF32(5, 0, uint32(Fp4KmLutLen + r * tg))
        p.emitTgLoadF32(6, 4, uint32(Fp4KmLutLen + r * tg))
        p.emitAdd(5, 5, 6); p.emitTgStoreF32(5, 0, uint32(Fp4KmLutLen + r * tg))
    p.emitBarrierTg(); p.emitMulConst(2, 2, 0.5'f32)
  p.emitLoadConstF32(2, 1)
  p.emitCmpLt(3, 0, 2)
  sessionVmIfGpu(p, 3):
    for row in 0 ..< rows:
      for c in 0 ..< 2:
        p.emitTgLoadF32(5, 0, uint32(Fp4KmLutLen + (row * 2 + c) * tg))
        p.emitUStoreF32(5, 12, uint32(row * N + c))
  p.emitHalt()
  result = p

proc buildFp4KmGemmM4Program*(K, N: int): VmProgram =
  ## FP4-KM group-32 GEMM tile MBLOCK=4: 1 output column / threadgroup, 4 input rows.
  ## Weights read once per column element, reused for 4 tokens (the matmul_q4_batched idea
  ## in nim_inference / Metal Q4). Input layout [4, K], output [4, N] row-major.
  ## Acc: R4/R11/R12/R13. Loop temps: R1/R2/R3. Reduce clobbers R5/R7/R8/R9/R10.
  if K <= 0 or K mod 32 != 0:
    raise newException(ValueError, "buildFp4KmGemmM4Program: K must be a positive multiple of 32")
  if N <= 0:
    raise newException(ValueError, "buildFp4KmGemmM4Program: N must be positive")
  let nG = K div 32
  var p: VmProgram
  p.emitUTgid(0)                                  # n = output column
  p.emitULid(1)                                   # lane 0..31
  p.emitUToR(6, 1)

  # LUT -> TG (272 float)
  p.emitUAddK(8, 1, 0)
  p.emitLoadConstF32(10, float32((Fp4KmLutLen + GemvTg - 1) div GemvTg))
  p.emitLoadConstF32(12, float32(Fp4KmLutLen))
  sessionVmForGpu(p, 10):
    p.emitUToR(11, 8)
    p.emitCmpLt(13, 11, 12)
    sessionVmIfGpu(p, 13):
      p.emitULoadConstF(9, 8, 0)
      p.emitTgStoreF32(9, 11, 0)
    p.emitUAddK(8, 8, uint32(GemvTg))
  p.emitBarrierTg()

  # Weight/scale pointers for column n (same as single-row GEMV group-32)
  p.emitUMulK(2, 0, uint32(K div 2))              # U2 = n*(K/2) + (lane&15) after add
  p.emitUAndK(6, 1, 15)
  p.emitUAdd(2, 2, 6)
  p.emitUMulK(3, 0, uint32(nG))
  p.emitUAddK(3, 3, uint32(N * (K div 2)))        # scale base
  p.emitUShrK(4, 1, 4)
  p.emitUShlK(4, 4, 2)                            # nibble shift 0 or 4
  p.emitUAddK(5, 1, 0)                            # U5 = lane = x row0 ptr
  # Accumulators for 4 input rows
  p.emitLoadConstF32(4, 0.0'f32)
  p.emitLoadConstF32(11, 0.0'f32)
  p.emitLoadConstF32(12, 0.0'f32)
  p.emitLoadConstF32(13, 0.0'f32)
  p.emitLoadConstF32(5, float32(nG))
  p.emitForBegin(5)
  p.emitULoadU8(6, 2, 0)                          # packed byte
  p.emitUShr(6, 6, 4)
  p.emitUAndK(6, 6, 15)
  p.emitUToR(1, 6)
  p.emitTgLoadF32(1, 1, 0)                        # E2M1
  p.emitULoadU8(7, 3, 0)                          # scale byte
  p.emitUToR(2, 7)
  p.emitTgLoadF32(2, 2, 16)                       # E4M3*gs
  p.emitMul(1, 1, 2)                              # R1 = w
  # 4 input rows at offsets 0, K, 2K, 3K from U5
  p.emitULoadIn(3, 5, 0)
  p.emitFma(4, 1, 3, 4)
  p.emitULoadIn(3, 5, uint32(K))
  p.emitFma(11, 1, 3, 11)
  p.emitULoadIn(3, 5, uint32(2 * K))
  p.emitFma(12, 1, 3, 12)
  p.emitULoadIn(3, 5, uint32(3 * K))
  p.emitFma(13, 1, 3, 13)
  p.emitUAddK(2, 2, 16)
  p.emitUAddK(3, 3, 1)
  p.emitUAddK(5, 5, 32)
  p.emitForEnd()

  # Reduce each row, then lane 0 writes out[m*N+n] BEFORE the TG area is reused.
  p.emitUToR(6, 1)
  p.emitTgReduce32(4, Fp4KmLutLen)
  p.emitLidIsZeroIf()
  p.emitTgLoadF32(9, 6, uint32(Fp4KmLutLen))
  p.emitUStoreF32(9, 0, 0)
  p.emitIfEnd()
  p.emitBarrierTg()
  p.emitTgReduce32(11, Fp4KmLutLen)
  p.emitLidIsZeroIf()
  p.emitTgLoadF32(9, 6, uint32(Fp4KmLutLen))
  p.emitUStoreF32(9, 0, uint32(N))
  p.emitIfEnd()
  p.emitBarrierTg()
  p.emitTgReduce32(12, Fp4KmLutLen)
  p.emitLidIsZeroIf()
  p.emitTgLoadF32(9, 6, uint32(Fp4KmLutLen))
  p.emitUStoreF32(9, 0, uint32(2 * N))
  p.emitIfEnd()
  p.emitBarrierTg()
  p.emitTgReduce32(13, Fp4KmLutLen)
  p.emitLidIsZeroIf()
  p.emitTgLoadF32(9, 6, uint32(Fp4KmLutLen))
  p.emitUStoreF32(9, 0, uint32(3 * N))
  p.emitIfEnd()
  p.emitHalt()
  result = p

# ── 2) FP32 GEMV (LoRA A/B, or fp32 Linear without quant) ─
proc f32GemvTgBytes*(): int = GemvTg * 4

proc buildF32GemvProgram*(K, N: int; scale: float32; accumulate: bool): VmProgram =
  ## out[n] (+)= scale * sum_k W[n*K+k] * x[k];  W in buffer(3), x in buffer(4).
  ## Any K (the K mod 32 remainder is handled with a mask).
  var p: VmProgram
  let full = K div GemvTg
  let tail = K mod GemvTg
  p.emitUTgid(0)
  p.emitULid(1)
  p.emitUToR(6, 1)
  p.emitUMulK(2, 0, uint32(K))
  p.emitUAdd(2, 2, 1)                             # W ptr
  p.emitUAddK(5, 1, 0)                            # x ptr
  if full > 0:
    p.emitLoadConstF32(5, float32(full))
    p.emitForBegin(5)
    p.emitULoadConstF(1, 2, 0)
    p.emitULoadIn(3, 5, 0)
    p.emitFma(4, 1, 3, 4)
    p.emitUAddK(2, 2, 32)
    p.emitUAddK(5, 5, 32)
    p.emitForEnd()
  if tail > 0:
    p.emitLoadConstF32(7, float32(tail))
    p.emitCmpLt(8, 6, 7)                          # lane < tail
    p.emitIfBegin(8)
    p.emitULoadConstF(1, 2, 0)
    p.emitULoadIn(3, 5, 0)
    p.emitFma(4, 1, 3, 4)
    p.emitIfEnd()
  p.emitTgReduce32(4, 0)
  p.emitLidIsZeroIf()
  p.emitTgLoadF32(9, 6, 0)
  if scale != 1.0'f32:
    p.emitMulConst(9, 9, scale)
  if accumulate:
    p.emitULoadF32(10, 0, 0)
    p.emitAdd(9, 9, 10)
  p.emitUStoreF32(9, 0, 0)
  p.emitIfEnd()
  p.emitHalt()
  result = p

# ── 2b) small-rank FP32 GEMV, NO tree reduction (used for LoRA B) ──
proc f32GemvSmallKTgBytes*(): int = 16

proc buildF32GemvSmallKProgram*(K, N: int; scale: float32; accumulate: bool): VmProgram =
  ## One thread = one output row, NO reduction.
  ## LoRA B has K = rank (4..16): the 32-lane kernel above pays 5 barriers
  ## for a dot product of a few elements, so it is much slower than K
  ## serial FMAs. Dispatches N threads (1 row / thread) instead of N*GemvTg.
  ## out[n] (+)= scale * sum_k W[n*K+k] * x[k];  W in buffer(3), x in buffer(4).
  if K <= 0 or K > 32:
    raise newException(ValueError,
      "buildF32GemvSmallKProgram: K must be in 1..32 (got " & $K & ")")
  if N <= 0:
    raise newException(ValueError, "buildF32GemvSmallKProgram: N must be positive")
  var p: VmProgram
  p.emitRToU(0, 0)                                # U0 = n = tid (1 row / thread)
  p.emitUMulK(1, 0, uint32(K))                    # U1 = n*K (W row base)
  p.emitUMulK(2, 0, 0)                            # U2 = 0   (x vector base)
  p.emitUAddK(3, 2, uint32(N))                    # U3 = N
  p.emitUToR(6, 0); p.emitUToR(7, 3)              # R6 = n, R7 = N
  p.emitCmpLt(8, 6, 7)                            # n < N
  p.emitIfBegin(8)                                # masks the grid's extra threads
  p.emitLoadConstF32(4, 0.0'f32)                  # acc = 0
  for k in 0 ..< K:
    p.emitULoadConstF(1, 1, uint32(k))            # R1 = W[n*K+k]
    p.emitULoadIn(3, 2, uint32(k))                # R3 = x[k]
    p.emitFma(4, 1, 3, 4)                         # acc += W*x
  if scale != 1.0'f32:
    p.emitMulConst(4, 4, scale)
  if accumulate:
    p.emitULoadF32(5, 0, 0)                       # R5 = out[n]
    p.emitAdd(4, 4, 5)
  p.emitUStoreF32(4, 0, 0)                        # out[n] = acc
  p.emitIfEnd()
  p.emitHalt()
  result = p

# ── 2c) FP32 batched GEMM for PREFILL (unquantized Linear: sidecar block G) ──
proc f32GemmBatchTgBytes*(rows, tg: int): int = max(rows * tg * 4, 16)

proc buildF32GemmBatchProgram*(K, N, tg, rows: int): VmProgram =
  ## out[r*N + n] = sum_k W[n*K + k] * x[r*K + k], r = 0 ..< rows.
  ## W in the constF slot, x in the Input slot, out in the MemF32 slot (like buildF32Gemv).
  ## One threadgroup holds 1 output column and reads each weight only ONCE for all
  ## `rows` rows. Prefill used to fall back to MPS for these fp32 Linears and MPS
  ## reached only ~1.7 GFLOP/s at M=11 on the iGPU (qkvProj 4096x12288 took 654 ms/block),
  ## i.e. ~100x slower than the decode fp32 GEMV.
  doAssert tg in [32, 64, 128, 256], "buildF32GemmBatchProgram: invalid tg"
  doAssert K > 0 and K mod tg == 0, "buildF32GemmBatchProgram: K must be divisible by tg"
  doAssert rows in 1 .. 12 and N > 0, "buildF32GemmBatchProgram: rows 1..12"
  # The first 8 accs use the regs the fp4 m8 kernel uses; the last 4 accs (rows > 8) take
  # regs 0..3. After the accs are written to TG memory they are dead, so the reduce reuses
  # them as temps.
  const accLo = [4'u32, 6'u32, 10'u32, 11'u32, 12'u32, 13'u32, 14'u32, 15'u32]
  const accHi = [0'u32, 1'u32, 2'u32, 3'u32]
  var acc: seq[uint32]
  for i in 0 ..< min(rows, 8): acc.add accLo[i]
  for i in 0 ..< max(0, rows - 8): acc.add accHi[i]
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitUMulK(2, 0, uint32(K)); p.emitUAdd(2, 2, 1)   # U2 = n*K + lane
  p.emitUAddK(3, 1, 0)                               # U3 = lane (x)
  for r in acc: p.emitLoadConstF32(r, 0)
  p.emitLoadConstF32(8, float32(K div tg))
  sessionVmForGpu(p, 8):
    p.emitULoadConstF(5, 2, 0)                       # R5 = w
    for i in 0 ..< rows:
      p.emitULoadIn(7, 3, uint32(i * K))
      p.emitFma(acc[i], 5, 7, acc[i])
    p.emitUAddK(2, 2, uint32(tg)); p.emitUAddK(3, 3, uint32(tg))
  p.emitUToR(9, 1)                                   # R9 = lane (index TG)
  for i in 0 ..< rows: p.emitTgStoreF32(acc[i], 9, uint32(i * tg))
  p.emitBarrierTg()
  p.emitLoadConstF32(5, float32(fastLog2(tg)))
  p.emitLoadConstF32(8, float32(tg div 2))
  sessionVmForGpu(p, 5):
    p.emitCmpLt(7, 9, 8)
    sessionVmIfGpu(p, 7):
      p.emitAdd(10, 9, 8)                            # R10 = lane + tree
      for i in 0 ..< rows:
        p.emitTgLoadF32(4, 9, uint32(i * tg))
        p.emitTgLoadF32(6, 10, uint32(i * tg))
        p.emitAdd(4, 4, 6); p.emitTgStoreF32(4, 9, uint32(i * tg))
    p.emitBarrierTg(); p.emitMulConst(8, 8, 0.5'f32)
  p.emitLoadConstF32(8, 1)
  p.emitCmpLt(7, 9, 8)
  sessionVmIfGpu(p, 7):
    for i in 0 ..< rows:
      p.emitTgLoadF32(4, 9, uint32(i * tg))
      p.emitUStoreF32(4, 0, uint32(i * N))
  p.emitHalt()
  result = p

# ── 3) bias (+ APFLU) element-wise: thread tid = element ────────
proc buildBiasActProgram*(apflu: bool; alpha = 0.1'f32; beta = 1.0'f32): VmProgram =
  ## mem[i] = act(mem[i] + bias[i]); bias in buffer(3). Dispatch exactly n threads.
  ## APFLU: x>0 ? x*(1+alpha*x) : beta*x*exp(x)  (matches cpuApflu in gpubackend.nim)
  var p: VmProgram
  p.emitRToU(0, 0)                                # U0 = tid  (R0 = tid set by the kernel)
  p.emitULoadF32(1, 0, 0)
  p.emitULoadConstF(2, 0, 0)
  p.emitAdd(1, 1, 2)
  if apflu:
    p.emitLoadConstF32(3, 0.0'f32)
    p.emitCmpLt(4, 3, 1)                          # 0 < x
    p.emitMul(5, 1, 4)                            # xp = max(x,0)
    p.emitSub(6, 1, 5)                            # xn = min(x,0)
    p.emitExp(7, 6)
    p.emitMul(7, 6, 7)                            # xn*exp(xn)
    p.emitMulConst(7, 7, beta)
    p.emitMulConst(8, 5, alpha)
    p.emitAddConst(8, 8, 1.0'f32)
    p.emitMul(8, 5, 8)                            # xp*(1+alpha*xp)
    p.emitAdd(1, 8, 7)
  p.emitUStoreF32(1, 0, 0)
  p.emitHalt()
  result = p

# ── 4) Linear++ (linear attention) single step, O(1) state ───────
type
  LpLayout* = object
    C*, H*, D*: int
    offProj*, offY*, offZ*, offS*, total*: int   ## in floats, aligned to 64

proc lpLayout*(C, H: int): LpLayout =
  doAssert C mod H == 0
  result.C = C; result.H = H; result.D = C div H
  result.offProj = 0
  result.offY = alignUp64(3 * C)
  result.offZ = alignUp64(result.offY + C)
  result.offS = alignUp64(result.offZ + H * result.D)
  result.total = alignUp64(result.offS + H * result.D * result.D)

proc emitPhi(p: var VmProgram; dst, src: uint32) =
  ## phi(x) = x>0 ? x+1 : exp(x) = max(x,0) + exp(min(x,0)); uses R14=0, R15 as temp.
  p.emitCmpLt(15, 14, src)
  p.emitMul(dst, src, 15)
  p.emitSub(15, src, dst)
  p.emitExp(15, 15)
  p.emitAdd(dst, dst, 15)

proc buildEluPlus1Program*(N: int): VmProgram =
  ## GPU pointwise phi(x)=ELU(x)+1 used by Linear++ training and inference.
  ## One GPU thread handles one element; input is buffer(4), output buffer(1).
  if N <= 0: raise newException(ValueError, "buildEluPlus1Program: N must be positive")
  var p: VmProgram
  p.emitRToU(0, 0)
  p.emitULoadIn(1, 0, 0)
  p.emitLoadConstF32(2, 0'f32)
  p.emitCmpLt(3, 2, 1)             # positive mask
  p.emitMul(4, 1, 3)               # max(x,0)
  p.emitSub(5, 1, 4)               # min(x,0)
  p.emitExp(6, 5)
  p.emitAdd(4, 4, 6)
  p.emitUStoreF32(4, 0, 0)
  p.emitHalt()
  result = p

proc buildEluPlus1BackwardProgram*(N: int): VmProgram =
  ## GPU pointwise derivative. Input buffer(4) = [x[N] | dPhi[N]], output dX[N].
  if N <= 0: raise newException(ValueError, "buildEluPlus1BackwardProgram: N must be positive")
  var p: VmProgram
  p.emitRToU(0, 0)
  p.emitULoadIn(1, 0, 0)                         # x
  p.emitUAddK(2, 0, uint32(N))
  p.emitULoadIn(3, 2, 0)                         # dPhi
  p.emitLoadConstF32(4, 0'f32)
  p.emitCmpLt(5, 4, 1)                           # x > 0
  p.emitExp(8, 1)
  p.emitLoadConstF32(9, 1'f32)
  p.emitSub(9, 9, 5)                              # 1-mask
  p.emitMul(8, 8, 9)
  p.emitAdd(8, 8, 5)                              # phi'(x), exp(x) for x<=0, 1 otherwise
  p.emitMul(3, 3, 8)
  p.emitUStoreF32(3, 0, 0)
  p.emitHalt()
  result = p

proc buildLinearPlusStepProgram*(lay: LpLayout; eps: float32): VmProgram =
  ## Each threadgroup = 1 head (D lanes, lane = value dim vd).
  ## memF32 buffer = [proj(3C) | y(C) | z(H*D) | S(H*D*D)] (see lpLayout).
  ## S[kd][vd] += phi(k[kd])*v[vd]; z[kd] += phi(k[kd]);
  ## y[vd] = sum_kd phi(q[kd])*S[kd][vd] / (sum_kd phi(q[kd])*z[kd] + eps)
  let C = lay.C; let D = lay.D
  var p: VmProgram
  p.emitLoadConstF32(14, 0.0'f32)
  p.emitUTgid(0)                                  # h
  p.emitULid(1)                                   # vd
  p.emitUMulK(2, 0, uint32(D))                    # U2 = h*D  -> q ptr (offProj = 0)
  p.emitUAddK(3, 2, uint32(C))                    # k ptr
  p.emitUAddK(4, 2, uint32(2 * C))
  p.emitUAdd(4, 4, 1)                             # v[h,vd] ptr
  p.emitULoadF32(11, 4, uint32(lay.offProj))      # R11 = v[h,vd]
  p.emitUAddK(7, 2, uint32(lay.offZ))             # z ptr
  p.emitUMulK(8, 0, uint32(D * D))
  p.emitUAddK(8, 8, uint32(lay.offS))
  p.emitUAdd(8, 8, 1)                             # S ptr = offS + h*D*D + vd
  p.emitLoadConstF32(12, 0.0'f32)                 # num
  p.emitLoadConstF32(13, 0.0'f32)                 # den
  p.emitLoadConstF32(10, float32(D))              # R10 = iteration count (NOT overwritten in the loop body:
  p.emitForBegin(10)                              # the JIT kernel rereads R[count] each iteration)
  p.emitULoadF32(1, 2, 0)                         # q
  p.emitULoadF32(2, 3, 0)                         # k
  p.emitPhi(3, 1)                                 # phi(q)
  p.emitPhi(4, 2)                                 # phi(k)
  p.emitULoadF32(5, 8, 0)                         # S
  p.emitFma(5, 4, 11, 5)                          # S += phi(k)*v
  p.emitUStoreF32(5, 8, 0)
  p.emitFma(12, 3, 5, 12)                         # num += phi(q)*S
  p.emitULoadF32(6, 7, 0)                         # z (old)
  p.emitAdd(6, 6, 4)                              # z + phi(k)
  p.emitFma(13, 3, 6, 13)                         # den += phi(q)*(z+phi(k))
  p.emitUAddK(2, 2, 1)
  p.emitUAddK(3, 3, 1)
  p.emitUAddK(7, 7, 1)
  p.emitUAddK(8, 8, uint32(D))
  p.emitForEnd()
  p.emitAddConst(13, 13, eps)
  p.emitDiv(12, 12, 13)
  p.emitUMulK(9, 0, uint32(D))
  p.emitUAdd(9, 9, 1)
  p.emitUAddK(9, 9, uint32(lay.offY))
  p.emitUStoreF32(12, 9, 0)                       # y[h*D+vd]
  p.emitBarrierTg()                               # every lane has read the old z
  # lane vd updates z[h, vd] += phi(k[h, vd])
  p.emitUMulK(10, 0, uint32(D))
  p.emitUAdd(10, 10, 1)                           # h*D + vd
  p.emitUAddK(11, 10, uint32(C))
  p.emitULoadF32(1, 11, 0)
  p.emitPhi(3, 1)
  p.emitUAddK(12, 10, uint32(lay.offZ))
  p.emitULoadF32(6, 12, 0)
  p.emitAdd(6, 6, 3)
  p.emitUStoreF32(6, 12, 0)
  p.emitHalt()
  result = p

# ── 5) INT4 asymmetric GEMV (block A weights in NIMBIN4 / QLoRA4) ─────
proc int4GemvTgBytes*(): int = GemvTg * 4

proc int4AsymSupported*(K, groupSize: int): bool =
  ## groupSize=0: per-row (1 scale/zp per row); >0: per-group (power of 2, >=64).
  if K mod 64 != 0: return false
  if groupSize == 0: return true
  groupSize >= 64 and (groupSize and (groupSize - 1)) == 0 and K mod groupSize == 0

proc buildInt4AsymGemvProgram*(K, N, groupSize: int): VmProgram =
  ## out[n] = sum_k x[k] * (nib(n,k) - zp[n,g]) * sc[n,g]
  ## Data layout: row n has K/2 bytes, byte b holds element 2b (low nibble) and 2b+1 (high)
  ## - matches dequantizeInt4Asymmetric. buffer(3) = [scale(N*nG) | zp(N*nG)] (fp32),
  ## nG = 1 for per-row. Each lane handles 1 byte (2 elements) / iteration; 32 lanes = 64 elements.
  if not int4AsymSupported(K, groupSize):
    raise newException(ValueError, "buildInt4AsymGemvProgram: K=" & $K & " groupSize=" & $groupSize &
                       " not supported (needs K%64==0, group a power of 2 >= 64)")
  let nG = if groupSize == 0: 1 else: K div groupSize
  ## groupSize is a power of 2 (int4AsymSupported) -> log2 = trailing zero bits, no loop needed
  let log2G = if groupSize > 0: countTrailingZeroBits(groupSize) else: 0
  var p: VmProgram
  p.emitUTgid(0)
  p.emitULid(1)
  p.emitUMulK(2, 0, uint32(K div 2))
  p.emitUAdd(2, 2, 1)                             # byte ptr = n*K/2 + lane
  p.emitUShlK(5, 1, 1)                            # x ptr = 2*lane
  p.emitUMulK(10, 0, uint32(nG))                  # U10 = n*nG
  p.emitLoadConstF32(13, float32(K div 64))       # iteration count (do NOT overwrite R13 in the body)
  p.emitForBegin(13)
  p.emitULoadU8(6, 2, 0)
  p.emitUAndK(7, 6, 15)
  p.emitUToR(1, 7)                                # lo
  p.emitUShrK(7, 6, 4)
  p.emitUToR(2, 7)                                # hi
  p.emitULoadIn(3, 5, 0)                          # x[2b]
  p.emitULoadIn(4, 5, 1)                          # x[2b+1]
  if nG > 1:
    p.emitUShrK(8, 5, uint32(log2G))              # group = k0 >> log2G
    p.emitUAdd(9, 8, 10)
  else:
    p.emitUAddK(9, 10, 0)
  p.emitULoadConstF(5, 9, 0)                      # scale   (R5)
  p.emitULoadConstF(6, 9, uint32(N * nG))         # zero pt (R6)
  p.emitMul(7, 1, 3)
  p.emitFma(7, 2, 4, 7)                           # lo*x0 + hi*x1
  p.emitAdd(8, 3, 4)
  p.emitMul(8, 8, 6)                              # zp*(x0+x1)
  p.emitSub(7, 7, 8)
  p.emitMul(7, 7, 5)                              # * scale
  p.emitAdd(12, 12, 7)                            # acc
  p.emitUAddK(2, 2, 32)
  p.emitUAddK(5, 5, 64)
  p.emitForEnd()
  p.emitUToR(6, 1)                                # R6 = lane for reduce
  p.emitTgReduce32(12, 0)
  p.emitLidIsZeroIf()
  p.emitTgLoadF32(9, 6, 0)
  p.emitUStoreF32(9, 0, 0)
  p.emitIfEnd()
  p.emitHalt()
  result = p

proc fp4KmTgCandidates*(K, rows, maxThreads, maxSharedBytes: int;
                         setting = "auto"): seq[int] =
  ## Reduction geometry is exact: never clamp a compiled kernel at dispatch.
  for tg in [32, 64, 128, 256, 512, 1024]:
    if tg <= maxThreads and K > 0 and K mod (tg * 8) == 0 and
       (Fp4KmLutLen + rows * tg) * 4 <= maxSharedBytes:
      result.add tg
  if setting == "max":
    if result.len > 0: result = @[result[^1]]
  elif setting != "auto" and setting != "0" and setting.len > 0:
    var selected: seq[int]
    for tg in result:
      if $tg == setting: selected.add tg
    result = selected
