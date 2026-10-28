## fp4_gemv_vm.nim — kernel GPU cho đường infer, viết bằng BYTECODE micro-VM.
##
## KHÔNG có một dòng MSL nào trong file này (và không có trong lora.nim):
## mọi kernel ở đây là VmProgram (session_vm.nim) -> jit_vm.dispatchProgram
## lower + compile JIT một lần / shape -> pipeline cache -> encode vào session.
##
## Quy ước buffer của kernel sau JIT (xem jit_vm.lowerBytecodeToMetal):
##   buffer(1) memF32   : output / work (ghi, có thể có offset)
##   buffer(2) memU8    : weight fp4 packed
##   buffer(3) constF32 : LUT (272/256 float) hoặc weight fp32 (LoRA) hoặc bias
##   buffer(4) inF32    : input x (chỉ khai báo khi program dùng OP_U_LOAD_IN)
##
## Mọi địa chỉ weight đi qua thanh ghi số nguyên U[] (uint32) — không dùng
## float R[] vì lm_head 32768x4096/2 = 67M byte > 2^24.

import std/math
import std/bitops
import session_vm

const
  GemvTg* = 32          ## thread / threadgroup của mọi kernel gemv (1 cột output / threadgroup)
  Fp4KmLutLen* = 272
  Fp4LutLen* = 256

proc alignUp64*(x: int): int {.inline.} = ((x + 63) div 64) * 64

# ── helper: reduction 32 lane trong threadgroup memory ──────────
proc emitTgReduce32(p: var VmProgram; accReg: uint32; tgBase: int) =
  ## Cộng 32 lane trong tgF32[tgBase .. tgBase+32). Kết quả nằm ở tgF32[tgBase]
  ## (chỉ lane 0 đọc). Không dùng simd_sum -> đúng với MỌI simdWidth (Intel = 16).
  ## Yêu cầu: R6 = float(lid) (đã set). Dùng R7,R8,R9,R10.
  p.emitTgStoreF32(accReg, 6, uint32(tgBase))
  p.emitBarrierTg()
  ## 5 bước (stride 16,8,4,2,1) là MỘT vòng FOR trên GPU, stride nằm trong R7 và
  ## giảm đôi mỗi vòng (không còn `for s in [...]` unroll trên host).
  ## R5 = số vòng (HẰNG trong vòng), R8 = cmp rồi addr. Tại 3 chỗ gọi, R5 đã rảnh
  ## (acc ở R4/R12, R6 = lane).
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
    p.emitBarrierTg()                           # top-level: mọi lane cùng đi qua
    p.emitMulConst(7, 7, 0.5'f32)

proc emitLidIsZeroIf(p: var VmProgram) =
  p.emitLoadConstF32(7, 1.0'f32)
  p.emitCmpLt(8, 6, 7)                          # lid < 1  <=> lid == 0
  p.emitIfBegin(8)

# ── 1) FP4 GEMV (M=1) ───────────────────────────────────────────
proc fp4GemvTgBytes*(km: bool): int =
  ((if km: Fp4KmLutLen else: Fp4LutLen) + GemvTg) * 4

proc fp4KmGemv2RowTgBytes*(): int =
  ## LUT + hai vung reduction 32-lane, theo tile 2 output rows/threadgroup.
  (Fp4KmLutLen + 2 * GemvTg) * 4

proc buildFp4GemvProgram*(K, N: int; km: bool; groupSize = 32): VmProgram =
  ## out[n] = sum_k x[k] * dequant(W[n,k]),  n = threadgroup id, 32 lane chia K.
  ## km=true : FP4-KM (E2M1 + scale E4M3/nhóm 32, nibble kiểu llama.cpp):
  ##           byte t giữ phần tử t (nibble thấp) và t+16 (nibble cao);
  ##           LUT gộp 272: [0..15]=E2M1, [16..271]=E4M3 đã nhân globalScale.
  ## km=false: định dạng cũ, 1 byte/phần tử, LUT 256 đã nhân scale.
  if K mod GemvTg != 0:
    raise newException(ValueError, "buildFp4GemvProgram: K=" & $K & " phai chia het " & $GemvTg)
  if km and not (groupSize == 32 or (groupSize >= 64 and groupSize mod 64 == 0)):
    raise newException(ValueError, "buildFp4GemvProgram: FP4-KM ho tro groupSize=32 hoac boi so cua 64 (nhan " & $groupSize & ")")
  if km and K mod groupSize != 0:
    raise newException(ValueError, "buildFp4GemvProgram: K=" & $K & " phai chia het groupSize=" & $groupSize)
  let lutN = if km: Fp4KmLutLen else: Fp4LutLen
  var p: VmProgram
  p.emitUTgid(0)                                  # U0 = n
  p.emitULid(1)                                   # U1 = lane
  p.emitUToR(6, 1)                                # R6 = float(lane)

  # copy LUT (device) -> threadgroup: lane i, i+32, ...
  p.emitUAddK(8, 1, 0)                            # U8 = lane (con trỏ chạy, += 32 mỗi vòng)
  p.emitLoadConstF32(10, float32((lutN + GemvTg - 1) div GemvTg))   # số vòng (HẰNG)
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
    ## Nhóm tổng quát (64, 128, ...). Layout đóng gói của quantizeFp4KM: nhóm g phần tử = g/2 byte,
    ## byte t giữ phần tử t (nibble thấp) và t+g/2 (nibble cao). 32 lane quét nhóm theo g/32 lượt:
    ## lượt s phủ phần tử s*32+lane -> nửa đầu (s < g/64): nibble thấp của byte s*32+lane;
    ## nửa sau: nibble cao của byte (s-g/64)*32+lane. Các lượt unroll ở host (hằng số tại build).
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
    p.emitULoadU8(7, 3, 0)                        # scale byte của nhóm
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
    p.emitUShlK(4, 4, 2)                          # U4 = shift (0 hoặc 4)
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

  p.emitUToR(6, 1)                                # R6 = lane (tái dùng cho reduce)
  p.emitTgReduce32(4, lutN)
  p.emitLidIsZeroIf()
  p.emitTgLoadF32(9, 6, uint32(lutN))
  p.emitUStoreF32(9, 0, 0)                        # out[n]
  p.emitIfEnd()
  p.emitHalt()
  result = p

proc buildFp4KmGemv2RowProgram*(K, N: int): VmProgram =
  ## FP4-KM group-32 GEMV, hai hàng output/threadgroup.
  ## Lấy ý tưởng N_R0_MXFP4=2 trong llama.cpp: mỗi lane tái sử dụng x[k]
  ## cho hai hàng weight; hai reduction độc lập dùng hai vùng TG memory.
  ## Không dùng SIMD-width assumption: reduction vẫn là cây 32-lane như kernel cũ.
  if K <= 0 or K mod 32 != 0:
    raise newException(ValueError, "buildFp4KmGemv2RowProgram: K phai la boi so duong cua 32")
  if N <= 0 or N mod 2 != 0:
    raise newException(ValueError, "buildFp4KmGemv2RowProgram: N phai duong va chia het cho 2")
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
    # ── Bản NẠP TRƯỚC 2 nhóm/vòng (học từ llama.cpp: nhiều load độc lập cùng bay) ─────────
    # Kernel này bị chặn bởi ĐỘ TRỄ bộ nhớ (đo ~3.6 GB/s = đúng định luật Little với ~170
    # thread phần cứng), nên cần nhiều byte "đang bay" mỗi thread. Mỗi vòng: phát TẤT CẢ load
    # (mã+scale 2 hàng x 2 nhóm, và x của 2 nhóm) TRƯỚC khi giải mã. Thứ tự cộng giữ nguyên
    # (nhóm chẵn rồi nhóm lẻ) => kết quả giống từng bit bản cũ.
    # U: A = U10 mã0, U11 scale0, U13 mã1, U14 scale1 | B = U3 mã0, U12 scale0, U15 mã1,
    #    scale1 của B nạp vào U10 sau khi nhóm A xong. R3 = x nhóm A, R9 = x nhóm B.
    p.emitLoadConstF32(5, float32(nG div 2))      # số vòng (R5 không đổi trong vòng lặp)
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
    # nhóm A, hàng 0
    p.emitUShr(10, 10, 4); p.emitUAndK(10, 10, 15)
    p.emitUToR(1, 10); p.emitTgLoadF32(1, 1, 0)
    p.emitUToR(2, 11); p.emitTgLoadF32(2, 2, 16)
    p.emitMul(1, 1, 2); p.emitFma(4, 1, 3, 4)
    # nhóm A, hàng 1
    p.emitUShr(13, 13, 4); p.emitUAndK(13, 13, 15)
    p.emitUToR(7, 13); p.emitTgLoadF32(7, 7, 0)
    p.emitUToR(8, 14); p.emitTgLoadF32(8, 8, 16)
    p.emitMul(7, 7, 8); p.emitFma(12, 7, 3, 12)
    p.emitULoadU8(10, 9, 1)                       # scale hàng 1 của nhóm B (U10 đã rảnh)
    # nhóm B, hàng 0
    p.emitUShr(3, 3, 4); p.emitUAndK(3, 3, 15)
    p.emitUToR(1, 3); p.emitTgLoadF32(1, 1, 0)
    p.emitUToR(2, 12); p.emitTgLoadF32(2, 2, 16)
    p.emitMul(1, 1, 2); p.emitFma(4, 1, 9, 4)
    # nhóm B, hàng 1
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
    # Nạp CẢ BỐN byte (code+scale của cả hai hàng) TRƯỚC khi giải mã. Đây là
    # kernel bound-độ-trễ: 1 load/lane đang để bộ nhớ nhàn rỗi ~90%. Bốn load
    # độc lập này cho 4 load cùng bay, và KHÔNG đổi một phép tính nào
    # (thứ tự cộng của từng lane giữ nguyên => kết quả bit-identical).
    # R10/R11 = code+scale hàng 0; R13/R14 = code+scale hàng 1.
    # TUYỆT ĐỐI KHÔNG dùng R12: R12 là accumulator của hàng 1 (khởi tạo
    # `LoadConstF32(12, 0.0)` trước vòng lặp, rồi `Fma(12, ...)`), nạp byte
    # vào R12 sẽ phá accumulator ngay từ iteration đầu.
    p.emitULoadU8(10, 5, 0)                         # row0 byte mã
    p.emitULoadU8(11, 6, 0)                         # row0 byte scale nhóm
    p.emitULoadU8(13, 8, 0)                         # row1 byte mã
    p.emitULoadU8(14, 9, 0)                         # row1 byte scale nhóm
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

# ── Biến thể kernel GEMV fp4-KM (chọn bằng BYBY_GEMV_VARIANT, xem gpu_layers.nim) ────────
# Cùng layout weight/LUT/scale với buildFp4KmGemv2RowProgram -> dùng chung bộ nhớ đệm GPU
# với kernel prefill GEMM. Layout 1 nhóm 32 giá trị = 16 byte: nibble THẤP của byte b là
# giá trị b, nibble CAO của byte b là giá trị 16+b.

proc emitKmLutCopy(p: var VmProgram) =
  ## Chép LUT 272 float vào tgF32[0..272) (cộng tác 32 lane). Dùng R10,R11,R12,R13,U8,U9.
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
  ## 2 hàng/threadgroup. Mỗi lane nạp MỘT WORD 32-bit (8 giá trị) cho mỗi hàng; 4 lane = 1 nhóm
  ## 32 giá trị, 32 lane = 8 nhóm / vòng. Tra scale E4M3 MỘT lần / nhóm (kernel cũ: mỗi giá trị).
  ## Số lệnh nạp global/SLM giảm ~3x. Tổng theo nhóm: acc += scale * sum(lut[nibble] * x).
  if not fp4KmGemvWideSupported(K, N):
    raise newException(ValueError, "buildFp4KmGemvWideProgram: can K chia het 256, N chan")
  let nG = K div 32
  let nIter = K div 256
  var p: VmProgram
  p.emitUTgid(0)
  p.emitULid(1)
  p.emitUToR(6, 1)
  p.emitKmLutCopy()
  p.emitUMulK(2, 0, 2)                              # U2 = hàng đầu của cặp
  p.emitUShrK(3, 1, 2)                              # U3 = gi = lane/4 (nhóm trong vòng)
  p.emitUAndK(4, 1, 3)                              # U4 = wi = lane%4 (word trong nhóm)
  p.emitUMulK(5, 2, uint32(K div 2))                # code ptr hàng 0
  p.emitUShlK(10, 3, 4); p.emitUAdd(5, 5, 10)       # + gi*16
  p.emitUShlK(10, 4, 2); p.emitUAdd(5, 5, 10)       # + wi*4
  p.emitUMulK(6, 2, uint32(nG))                     # scale ptr hàng 0
  p.emitUAddK(6, 6, uint32(N * (K div 2)))
  p.emitUAdd(6, 6, 3)                               # + gi
  p.emitUShlK(7, 3, 5)                              # x ptr = gi*32
  p.emitUShlK(10, 4, 2); p.emitUAdd(7, 7, 10)       # + wi*4
  p.emitLoadConstF32(4, 0.0'f32)                    # acc hàng 0
  p.emitLoadConstF32(12, 0.0'f32)                   # acc hàng 1
  p.emitLoadConstF32(5, float32(nIter))
  p.emitForBegin(5)
  p.emitLoadConstF32(13, 0.0'f32)                   # tổng nhóm hàng 0
  p.emitLoadConstF32(14, 0.0'f32)                   # tổng nhóm hàng 1
  p.emitULoadU32(8, 5, 0)                           # 8 nibble hàng 0
  p.emitULoadU32(9, 5, uint32(K div 2))             # 8 nibble hàng 1
  p.emitULoadU8(10, 6, 0)                           # scale hàng 0
  p.emitULoadU8(11, 6, uint32(nG))                  # scale hàng 1
  for t in 0 ..< 4:
    for hi in 0 ..< 2:
      let sh = uint32(8 * t + 4 * hi)
      let xoff = uint32(t + 16 * hi)
      p.emitUShrK(12, 8, sh); p.emitUAndK(12, 12, 15)
      p.emitUToR(1, 12); p.emitTgLoadF32(1, 1, 0)
      p.emitUShrK(13, 9, sh)                        # U13 = nibble hàng 1
      p.emitUAndK(13, 13, 15)
      p.emitUToR(2, 13); p.emitTgLoadF32(2, 2, 0)
      p.emitULoadIn(3, 7, xoff)
      p.emitFma(13, 1, 3, 13)
      p.emitFma(14, 2, 3, 14)
  p.emitUToR(10, 10); p.emitTgLoadF32(10, 10, 16)   # scale hàng 0
  p.emitUToR(11, 11); p.emitTgLoadF32(11, 11, 16)   # scale hàng 1
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
  ## 4 hàng/threadgroup, cùng ánh xạ lane như kernel 2 hàng (x nạp 1 lần dùng cho 4 hàng; LUT
  ## và barrier amortize trên 4 hàng). Nạp 8 byte (code+scale của 4 hàng) TRƯỚC khi giải mã.
  if K <= 0 or K mod 32 != 0 or N <= 0 or N mod 4 != 0:
    raise newException(ValueError, "buildFp4KmGemv4RowProgram: K chia het 32, N chia het 4")
  let nG = K div 32
  const accR = [4'u32, 12'u32, 13'u32, 14'u32]
  var p: VmProgram
  p.emitUTgid(0)
  p.emitULid(1)
  p.emitUToR(6, 1)
  p.emitKmLutCopy()
  p.emitUMulK(2, 0, 4)                              # U2 = hàng đầu của bộ 4
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
  doAssert tg in [32, 64, 128, 256] and rows in [2, 4]
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
  doAssert tg in [32, 64, 128, 256] and batchRows in [4, 8]
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

proc buildFp4KmGemmM4Program*(K, N: int): VmProgram =
  ## FP4-KM group-32 GEMM tile MBLOCK=4: 1 cột output / threadgroup, 4 hàng input.
  ## Weight đọc 1 lần / phần tử cột, tái dùng cho 4 token (ý tưởng matmul_q4_batched
  ## trong nim_inference / Metal Q4). Input layout [4, K], output [4, N] row-major.
  ## Acc: R4/R11/R12/R13. Temps loop: R1/R2/R3. Reduce clobber R5/R7/R8/R9/R10.
  if K <= 0 or K mod 32 != 0:
    raise newException(ValueError, "buildFp4KmGemmM4Program: K phai la boi so duong cua 32")
  if N <= 0:
    raise newException(ValueError, "buildFp4KmGemmM4Program: N phai duong")
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

  # Reduce từng hàng rồi lane-0 ghi out[m*N+n] TRƯỚC khi tái dùng vùng TG.
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

# ── 2) FP32 GEMV (LoRA A/B, hoặc Linear fp32 nếu không có quant) ─
proc f32GemvTgBytes*(): int = GemvTg * 4

proc buildF32GemvProgram*(K, N: int; scale: float32; accumulate: bool): VmProgram =
  ## out[n] (+)= scale * sum_k W[n*K+k] * x[k];  W ở buffer(3), x ở buffer(4).
  ## K bất kỳ (phần dư K mod 32 xử lý bằng mặt nạ).
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

# ── 2b) FP32 GEMV rank nhỏ, KHÔNG tree-reduction (dùng cho LoRA B) ──
proc f32GemvSmallKTgBytes*(): int = 16

proc buildF32GemvSmallKProgram*(K, N: int; scale: float32; accumulate: bool): VmProgram =
  ## Một thread = một hàng output, KHÔNG reduction.
  ## LoRA B có K = rank (4..16): kernel 32-lane ở trên phải trả 5 barrier
  ## chỉ cho một dot-product vài phần tử, nên nó chậm hơn hẳn K phép FMA
  ## nối tiếp. Dispatch N thread (1 hàng / thread) thay vì N*GemvTg.
  ## out[n] (+)= scale * sum_k W[n*K+k] * x[k];  W ở buffer(3), x ở buffer(4).
  if K <= 0 or K > 32:
    raise newException(ValueError,
      "buildF32GemvSmallKProgram: K phai trong 1..32 (nhan " & $K & ")")
  if N <= 0:
    raise newException(ValueError, "buildF32GemvSmallKProgram: N phai duong")
  var p: VmProgram
  p.emitRToU(0, 0)                                # U0 = n = tid (1 hàng / thread)
  p.emitUMulK(1, 0, uint32(K))                    # U1 = n*K (gốc hàng W)
  p.emitUMulK(2, 0, 0)                            # U2 = 0   (gốc vector x)
  p.emitUAddK(3, 2, uint32(N))                    # U3 = N
  p.emitUToR(6, 0); p.emitUToR(7, 3)              # R6 = n, R7 = N
  p.emitCmpLt(8, 6, 7)                            # n < N
  p.emitIfBegin(8)                                # chặn thread dư của grid
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

# ── 3) bias (+ APFLU) element-wise: thread tid = phần tử ────────
proc buildBiasActProgram*(apflu: bool; alpha = 0.1'f32; beta = 1.0'f32): VmProgram =
  ## mem[i] = act(mem[i] + bias[i]); bias ở buffer(3). Dispatch đúng n thread.
  ## APFLU: x>0 ? x*(1+alpha*x) : beta*x*exp(x)  (khớp cpuApflu trong gpubackend.nim)
  var p: VmProgram
  p.emitRToU(0, 0)                                # U0 = tid  (R0 = tid do kernel gán)
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

# ── 4) Linear++ (attention tuyến tính) 1 bước, state O(1) ───────
type
  LpLayout* = object
    C*, H*, D*: int
    offProj*, offY*, offZ*, offS*, total*: int   ## đơn vị float, đã căn 64

proc lpLayout*(C, H: int): LpLayout =
  doAssert C mod H == 0
  result.C = C; result.H = H; result.D = C div H
  result.offProj = 0
  result.offY = alignUp64(3 * C)
  result.offZ = alignUp64(result.offY + C)
  result.offS = alignUp64(result.offZ + H * result.D)
  result.total = alignUp64(result.offS + H * result.D * result.D)

proc emitPhi(p: var VmProgram; dst, src: uint32) =
  ## phi(x) = x>0 ? x+1 : exp(x) = max(x,0) + exp(min(x,0)); dùng R14=0, R15 tạm.
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
  ## Mỗi threadgroup = 1 head (D lane, lane = chiều value vd).
  ## Buffer memF32 = [proj(3C) | y(C) | z(H*D) | S(H*D*D)] (xem lpLayout).
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
  p.emitLoadConstF32(10, float32(D))              # R10 = số vòng (KHÔNG ghi đè trong thân vòng:
  p.emitForBegin(10)                              # kernel JIT đọc lại R[count] mỗi vòng)
  p.emitULoadF32(1, 2, 0)                         # q
  p.emitULoadF32(2, 3, 0)                         # k
  p.emitPhi(3, 1)                                 # phi(q)
  p.emitPhi(4, 2)                                 # phi(k)
  p.emitULoadF32(5, 8, 0)                         # S
  p.emitFma(5, 4, 11, 5)                          # S += phi(k)*v
  p.emitUStoreF32(5, 8, 0)
  p.emitFma(12, 3, 5, 12)                         # num += phi(q)*S
  p.emitULoadF32(6, 7, 0)                         # z (cũ)
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
  p.emitBarrierTg()                               # mọi lane đã đọc xong z cũ
  # lane vd cập nhật z[h, vd] += phi(k[h, vd])
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

# ── 5) INT4 asymmetric GEMV (weight block A trong NIMBIN4 / QLoRA4) ─────
proc int4GemvTgBytes*(): int = GemvTg * 4

proc int4AsymSupported*(K, groupSize: int): bool =
  ## groupSize=0: per-row (1 scale/zp cho ca hang); >0: per-group (luy thua 2, >=64).
  if K mod 64 != 0: return false
  if groupSize == 0: return true
  groupSize >= 64 and (groupSize and (groupSize - 1)) == 0 and K mod groupSize == 0

proc buildInt4AsymGemvProgram*(K, N, groupSize: int): VmProgram =
  ## out[n] = sum_k x[k] * (nib(n,k) - zp[n,g]) * sc[n,g]
  ## Layout data: hang n co K/2 byte, byte b giu phan tu 2b (nibble thap) va 2b+1 (cao)
  ## - khop dequantizeInt4Asymmetric. buffer(3) = [scale(N*nG) | zp(N*nG)] (fp32),
  ## nG = 1 neu per-row. Moi lane xu ly 1 byte (2 phan tu) / vong; 32 lane = 64 phan tu.
  if not int4AsymSupported(K, groupSize):
    raise newException(ValueError, "buildInt4AsymGemvProgram: K=" & $K & " groupSize=" & $groupSize &
                       " khong ho tro (can K%64==0, group luy thua 2 >= 64)")
  let nG = if groupSize == 0: 1 else: K div groupSize
  ## groupSize là luỹ thừa 2 (int4AsymSupported) -> log2 = số bit 0 đuôi, không cần vòng
  let log2G = if groupSize > 0: countTrailingZeroBits(groupSize) else: 0
  var p: VmProgram
  p.emitUTgid(0)
  p.emitULid(1)
  p.emitUMulK(2, 0, uint32(K div 2))
  p.emitUAdd(2, 2, 1)                             # byte ptr = n*K/2 + lane
  p.emitUShlK(5, 1, 1)                            # x ptr = 2*lane
  p.emitUMulK(10, 0, uint32(nG))                  # U10 = n*nG
  p.emitLoadConstF32(13, float32(K div 64))       # so vong (KHONG ghi de R13 trong than)
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
  p.emitUToR(6, 1)                                # R6 = lane cho reduce
  p.emitTgReduce32(12, 0)
  p.emitLidIsZeroIf()
  p.emitTgLoadF32(9, 6, 0)
  p.emitUStoreF32(9, 0, 0)
  p.emitIfEnd()
  p.emitHalt()
  result = p
