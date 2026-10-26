## mamba2_vm_kernels.nim
## Build 2 VmProgram (session_vm.nim) chạy phần SSM recurrence + RMSNorm
## gate của Mamba2 100% trên GPU. Dùng cùng opcode mới trong session_vm.nim
## (EXP, NEG, SIGMOID, RSQRT, LOAD/STORE_F32_IDX, IF_BEGIN/END, CMP_LT...).
## Dispatch qua session_vm_dispatch.nim (metal_enc_* thật, không giả lập).

import std/bitops
import session_vm


# ── Reduction cây trong threadgroup, chạy 100% trên GPU ──────
## Trước đây: `while stride >= 1:` trên HOST, unroll log2(n) bước lúc build.
## Giờ: 1 vòng FOR_BEGIN/END trên GPU; stride nằm trong thanh ghi, giảm đôi mỗi
## vòng bằng MUL_CONST 0.5 (chính xác với luỹ thừa 2). Barrier nằm NGOÀI IF nên
## mọi thread trong threadgroup cùng gặp nó (không deadlock).
## Thanh ghi: rStride (đổi), rCnt (HẰNG trong vòng: Metal đọc lại mỗi vòng),
## rTmp (cmp rồi addr), rA, rB. idxReg = chỉ số trong threadgroup.
proc emitTgTreeReduce*(p: var VmProgram; idxReg: uint32; n: int;
                      rStride, rCnt, rTmp, rA, rB: uint32) =
  doAssert n >= 1 and (n and (n - 1)) == 0, "emitTgTreeReduce: n phai la luy thua 2"
  p.emitLoadConstF32(rCnt, float32(countTrailingZeroBits(n)))   # log2(n) bước
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
# 1) Mamba2 SSM step — grid = H*D thread, vòng n chạy GPU.
# Dùng đúng 16 thanh ghi, không tràn, không spill.
# Bảng idx h/d/g build 1 lần lúc load model (hằng số suốt đời model).
#
# memF32 layout:
#   offX     : xPart[HD]
#   offB     : bMat[G*N]
#   offC     : cMat[G*N]
#   offDt    : dt[H]        (đã softplus, tính bằng sessionActivation có sẵn)
#   offALog  : aLog[H]
#   offDSkip : dSkip[H]
#   offState : ssmState[HD*N]   (đọc + ghi tại chỗ, persistent)
#   offY     : yPre[HD]         (output)
#   offIdxH/D/G : bảng chỉ số const, build 1 lần
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
  # FIX token rac (bug #11): TRUOC DAY o day khai bao THEM mot bang
  # `offIdxD` xen GIUA idxH va idxG:
  #     let offIdxD = offIdxH + HD
  #     let offIdxG = offIdxD + HD
  # Nhung buildShapeInfo() trong mamba2_ssm_gpu.nim CHI dung 2 bang
  # (idxH, idxG) nam LIEN NHAU, va totalLen = offIdxG + HD. Ket qua:
  # bytecode doc `g` o offset offY+3*HD, trong khi host ghi `g` o
  # offY+2*HD - lech dung HD phan tu, va offY+3*HD CHINH BANG totalLen
  # tuc la doc NGOAI vung du lieu host cap. Tren hot path hScratch la
  # buffer lon dung chung nen khong crash: vung do chua ai ghi (0) ->
  # g = 0 cho CA 8192 thread, MOI block, MOI token -> 128 head deu lay
  # B va C cua group 0. Do o quy mo that (H=128 D=64 N=128 G=8):
  # rel-L2 cua y = 0.62, cua ssmState (mang sang token sau) = 0.20,
  # 7168/8192 thread sai group. Bang idxD KHONG duoc dung o bat ky
  # dau trong chuong trinh nay (x lay o offX+tid, stateBase = tid*N)
  # nen bo han, khong thay the.
  let offIdxH   = offY + HD
  let offIdxG   = offIdxH + HD

  var p: VmProgram
  # reserved suốt hàm: R0=tid R1=h R3=g R4=yAcc R9=dA R10=xVal R11=dt
  # scratch trong vòng n: R12,R13,R14,R15 (4 thanh ghi, đủ)
  p.emitAddConst(12, 0, float32(offIdxH)); p.emitLoadF32Idx(1, 12)   # R1 = h
  p.emitAddConst(12, 0, float32(offIdxG)); p.emitLoadF32Idx(3, 12)   # R3 = g

  p.emitAddConst(12, 1, float32(offALog)); p.emitLoadF32Idx(12, 12)
  p.emitExp(12, 12); p.emitNeg(12, 12)                                # R12 = A_h (tạm)
  p.emitAddConst(11, 1, float32(offDt));   p.emitLoadF32Idx(11, 11)   # R11 = dt[h]  [dài hạn]
  p.emitMul(9, 11, 12); p.emitExp(9, 9)                               # R9 = dA      [dài hạn]

  p.emitLoadF32Idx(10, 0)                                             # R10 = xVal   [dài hạn]  (offX=0)

  p.emitAddConst(12, 1, float32(offDSkip)); p.emitLoadF32Idx(12, 12)
  p.emitMul(4, 12, 10)                                                # R4 = yAcc init [dài hạn]

  p.emitMulConst(2, 3, float32(N))          # R2 = gN         [dài hạn]
  p.emitMulConst(6, 0, float32(N))          # R6 = stateBase  [dài hạn]
  p.emitLoadConstF32(7, 0.0'f32)             # R7 = n          [dài hạn]
  p.emitLoadConstF32(8, float32(N))          # R8 = count FOR  [dài hạn]

  sessionVmForGpu(p, 8):
    p.emitAddConst(13, 2, float32(offB)); p.emitAdd(13, 13, 7); p.emitLoadF32Idx(13, 13)  # R13 = bVal
    p.emitAddConst(14, 2, float32(offC)); p.emitAdd(14, 14, 7); p.emitLoadF32Idx(14, 14)  # R14 = cVal
    p.emitAddConst(15, 6, float32(offState)); p.emitAdd(15, 15, 7)                        # R15 = stateIdx
    p.emitLoadF32Idx(12, 15)              # R12 = sOld
    p.emitMul(12, 12, 9)                  # R12 = sOld*dA
    p.emitMul(13, 13, 10)                 # R13 = bVal*xVal  (đè, bVal dùng xong)
    p.emitFma(12, 11, 13, 12)             # R12 = sNew = dt*bVal*xVal + sOld*dA
    p.emitStoreF32Idx(12, 15)             # ssmState[stateIdx] = sNew
    p.emitFma(4, 14, 12, 4)               # yAcc += cVal*sNew
    p.emitAddConst(7, 7, 1.0'f32)         # n += 1

  p.emitAddConst(13, 0, float32(offY))
  p.emitStoreF32Idx(4, 13)                # yPre[tid] = yAcc
  p.emitHalt()
  result = p

# ════════════════════════════════════════════════════════════
# 2) RMSNorm + gate cuối — 1 threadgroup / token, C thread.
# gated = yPre*silu(z) ; out = gated * rsqrt(mean(gated^2)+eps) * w
# Reduction cây trong threadgroup dùng IF_BEGIN/END + CMP_LT.
# Yêu cầu C là lũy thừa 2 và <= max threads/threadgroup của thiết bị
# (thường 1024) — C lớn hơn cần chia nhiều threadgroup + cộng partial
# sum bằng kernel phụ, CHƯA làm ở đây (nói rõ, không né).
# memF32: [ yPre(0..<C) | z(C..<2C) | w(2C..<3C) | out(3C..<4C) ]
# ════════════════════════════════════════════════════════════
proc buildRmsGateProgram*(C: int; eps: float32): VmProgram =
  doAssert (C and (C-1)) == 0, "buildRmsGateProgram: C phải là lũy thừa 2 (pad nếu cần)"
  let offZ = C
  let offW = 2*C
  let offOut = 3*C

  var p: VmProgram
  # R0=tid  R1=gated  [dài hạn]
  p.emitLoadF32Idx(1, 0)                              # R1 = yPre[tid]
  p.emitAddConst(2, 0, float32(offZ)); p.emitLoadF32Idx(2, 2)   # R2 = z[tid]
  p.emitNeg(3, 2)                                     # R3 = -z
  p.emitExp(3, 3); p.emitAddConst(3, 3, 1.0'f32)       # R3 = 1+exp(-z)
  p.emitLoadConstF32(5, 1.0'f32)
  p.emitDiv(3, 5, 3)                                   # R3 = sigmoid(z)
  p.emitMul(3, 2, 3)                                   # R3 = silu(z)
  p.emitMul(1, 1, 3)                                   # R1 = gated  [dài hạn]

  p.emitMul(4, 1, 1)                                   # R4 = gated^2
  p.emitTgStoreF32(4, 0, 0)
  p.emitBarrierTg()

  p.emitTgTreeReduce(0, C, 6, 7, 8, 9, 10)   # GPU loop, không còn while host

  p.emitLoadConstF32(12, 0.0'f32)
  p.emitTgLoadF32(11, 12, 0)         # R11 = tg[0] = tổng gated^2
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
# 3) Conv1d incremental (K=4 điển hình) — grid = cDim thread.
# memF32 layout:
#   offXBC   : xBCPre[cDim]          (input mới)
#   offState : convState[cDim*(K-1)] (đọc + ghi tại chỗ)
#   offW     : convWeight[cDim*K]    (row-major c*K + k)
#   offBias  : convBias[cDim]
#   offOut   : xBCConv[cDim]         (output sau silu)
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

  p.emitLoadConstF32(5, 0.0'f32)                 # k (đếm trên GPU)
  p.emitLoadConstF32(6, float32(K-1))     # số vòng
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
  p.emitLoadConstF32(5, 0.0'f32)                 # k (đếm trên GPU)
  p.emitLoadConstF32(6, float32(K-2))     # số vòng
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
# 4) Softplus(dt) + split helper — grid = H thread (dt) hoặc full.
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
# 5) RMS+gate cho C lớn (chunked): mỗi threadgroup xử lý 1 chunk
# kích thước TG=1024 (power-of-2). Phase 1: partial sum per chunk
# vào memF32[offPartial + chunkId]. Phase 2 (grid=1 TG): reduce
# partials → rsqrt → broadcast scale. Phase 3: scale * gated * w.
# Để đơn giản và đúng, dùng 2 program riêng + 1 dispatch nhỏ.
# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# 6) Split proj [2C+2GN+H] → z[C] | xBC[cDim] | dt[H]
# grid = max(C, cDim, H) threads; each writes only if tid < len
# mem: [ proj(0..<P) | z(P..<P+C) | xBC(P+C..<P+C+cDim) | dt(P+C+cDim..<...) ]
# ════════════════════════════════════════════════════════════
proc buildFusedSplitConvSoftplusProgram*(C, cDim, H, K: int): VmProgram =
  ## MOI - GOP split(z,xBC,dt) + conv1d(xBC) + softplus(dt) THANH 1 KERNEL
  ## DUY NHAT, dung CHINH bo opcode co san cua micro-VM (KHONG viet .metal
  ## tay). Hoc theo fusedNormQKVRoPE/fusedResidualNormFFN cua nim_inference.nim
  ## (model 6.7B THAT, khong lag tren cung phan cung) - nguyen tac chung:
  ## giam so luong dispatch GPU + sessionCopyInto rieng le moi layer.
  ##
  ## AN TOAN DE GOP (khong can threadgroup_barrier): conv1d o day la
  ## DEPTHWISE THEO THOI GIAN - xem buildConv1dIncProgram goc, MOI truy cap
  ## (state/weight/bias/input) deu qua CHINH SO tid=c cua thread do, khong
  ## thread nao can du lieu cua thread khac. Vay 1 thread co the lam LIEN
  ## TIEP: split gia tri xBC[tid] cua CHINH NO (giu trong register, KHONG
  ## ghi RAM trung gian) roi dung NGAY cho conv1d; tuong tu voi dt+softplus.
  ##
  ## Loi ich: 3 dispatch (split, conv, softplus) + 4 sessionCopyInto (chuyen
  ## xBC/dt qua lai giua cac buffer scratch) -> CON 1 dispatch DUY NHAT,
  ## khong copy trung gian nao. Dung toi da R0..R7 (VM chi co 16 register
  ## - R0..R15, xem pack2() mask 0xF trong session_vm.nim va "float R[16]"
  ## trong jit_vm.nim - vuot qua se TRAN THAM LANG, sai ket qua).
  ##
  ## Memory layout dau vao (khop voi cach encodeLayerStep se goi):
  ##   [0, P)                    proj (P=C+cDim+H, dau ra fp4 gemv inProj)
  ##   [P, P+C)                  outZ (copy thang, dung sau boi rmsnorm-gate)
  ##   [P+C, P+C+cDim*(K-1))     convState (persistent, doc - CHUA cap nhat
  ##                              o day, encodeLayerStep tu ghi lai sau)
  ##   [.. + cDim*K)             convWeight (persistent, chi doc)
  ##   [.. + cDim)               convBias (persistent, chi doc)
  ##   [.. + cDim)               outXBC (= silu(conv(...)), dung boi ssm-step)
  ##   [.. + H)                  dtBias (persistent, chi doc)
  ##   [.. + H)                  outDt (= softplus(...), dung boi ssm-step)
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
  ## MOI (fix bug convState dong bang - xem bao cao debug): output THEM
  ## [offOutConvSt, offOutConvSt + cDim*(K-1)) = convState DA TRUOT, y het
  ## logic "slide state" cua buildConv1dIncProgram goc (da CO san, chi bi
  ## bo sot luc gop kernel) - encodeLayerStep phai copy vung nay ve
  ## o.offConvSt SAU khi dispatch, xem sua tuong ung o do.

  var p: VmProgram
  # R0 = tid, dat 1 lan boi runtime VM, dung suot ca chuong trinh - KHONG
  # bao gio bi cac phan duoi ghi de.

  # ── Phan 1: split z (thuan copy, dung boi rmsnorm-gate ve sau) ──
  p.emitLoadConstF32(1, float32(C))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitLoadF32Idx(3, 0)                      # R3 = proj[tid]
    p.emitAddConst(4, 0, float32(offOutZ))
    p.emitStoreF32Idx(3, 4)

  # ── Phan 2: split xBC (giu o R7, KHONG ghi RAM trung gian) + conv1d
  # NGAY tai cho (y het buildConv1dIncProgram goc, chi doi input tu memory
  # sang register R7) ──
  p.emitLoadConstF32(1, float32(cDim))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitAddConst(3, 0, float32(C))
    p.emitLoadF32Idx(7, 3)                      # R7 = xBC[tid] = proj[C+tid]

    p.emitAddConst(1, 0, float32(offConvB))
    p.emitLoadF32Idx(1, 1)                      # R1 = acc = bias[c]
    p.emitLoadConstF32(8, 0.0'f32)                 # k (đếm trên GPU)
    p.emitLoadConstF32(9, float32(K-1))     # số vòng
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

    # ── MOI: "truot" convState - y het buildConv1dIncProgram goc (bi bo
    # sot luc gop kernel, xem ghi chu o offOutConvSt phia tren). R7 van con
    # giu xBC[tid] hien tai tu luc load o dau block nay (khong bi FMA/silu
    # o tren dung toi) - dung lai thang, khong can doc lai tu memory. ──
    p.emitLoadConstF32(8, 0.0'f32)                 # k (đếm trên GPU)
    p.emitLoadConstF32(9, float32(K-2))     # số vòng
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

  # ── Phan 3: split dt (R3) + softplus NGAY tai cho ──
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
# Mỗi thread xử lý elemsPerThread = C/1024 phần tử:
#   gated, localSumSq, rồi TG tree reduce → rsqrt → scale ghi out.
# mem: [ yPre(0..<C) | z(C..<2C) | w(2C..<3C) | out(3C..<4C) ]
# Yêu cầu C % 1024 == 0.
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

  p.emitTgTreeReduce(0, tgSize, 6, 7, 8, 9, 10)   # GPU loop, không còn while host

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
# 9) BAN "TAI CHO" (moi) - split+conv1d+softplus, ssm-step, gated-rmsnorm
# THEO NHOM, tat ca nhan OFFSET TUYET DOI va doc/ghi TRUC TIEP tren buffer
# layer thuong tru (hWork) - khong con sessionCopyInto nao giua cac buoc
# (xem giai thich day du trong BAO_CAO.md muc 2, va patch session_vm.nim/
# jit_vm.nim/micro_vm.metal them OP_LOAD_LID/OP_LOAD_TGID can cho ham thu 3).
#
# gated RMSNorm o day chuan hoa THEO NHOM (group_size = C div nGroups),
# dung cong thuc GOC cua mamba_ssm (RMSNormGated(..., group_size=d_inner//
# ngroups)) va cua vLLM (Mixer2RMSNormGated(intermediate_size, n_groups,...)).
# Ham buildRmsGateProgram/buildRmsGateLargeProgram o TREN (muc 2/7) chuan hoa
# tren CA C - dung HANH VI CUA HF transformers modeling_mamba2.py, noi
# n_groups bi BO SOT trong MambaRMSNormGated (xac nhan trong PR
# huggingface/transformers#35943, da sua cho Zamba2; PR #40861 "Support
# n_groups>1 for mamba2" con mo). Goi buildRmsGateGroupedAt voi nGroups=1
# se tai tao CHINH XAC ham cu o tren, nen 2 duong co the doi chieu truc tiep
# (xem test_layer0.nim / BYBY_MAMBA_NORM_GROUPS).
# ════════════════════════════════════════════════════════════

proc buildSplitConvSoftplusAt*(C, cDim, H, K: int;
                               offZ, offXBC, offDt,
                               offConvW, offConvB, offConvSt,
                               offDtBias: int): VmProgram =
  ## hWork layout (offset TUYET DOI, do computeOffV2 sinh):
  ##   [0, P)                proj = in_proj(x),  P = C + cDim + H  (chi doc)
  ##   [offZ,  offZ+C)       z          (ghi)
  ##   [offXBC, +cDim)       xBCConv sau SiLU  (ghi)
  ##   [offDt,  +H)          dt sau softplus   (ghi)
  ##   [offConvW, +cDim*K)   convWeight (chi doc, thuong tru)
  ##   [offConvB, +cDim)     convBias   (chi doc, thuong tru)
  ##   [offConvSt,+cDim*(K-1)) convState (DOC + GHI TAI CHO)
  ##   [offDtBias,+H)        dtBias     (chi doc, thuong tru)
  ## convState cap nhat tai cho AN TOAN vi thread c chi dung hang c: doc HET
  ## K-1 gia tri vao thanh ghi TRUOC, tinh xong moi ghi lai (khong con can
  ## buffer offOutConvSt rieng nhu buildFusedSplitConvSoftplusProgram cu).
  doAssert K >= 2 and K <= 8, "buildSplitConvSoftplusAt: K trong [2,8]"
  var p: VmProgram

  # ── Phan 1: z = proj[0..C) ──
  p.emitLoadConstF32(1, float32(C))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitLoadF32Idx(3, 0)                       # R3 = proj[tid]
    p.emitAddConst(4, 0, float32(offZ))
    p.emitStoreF32Idx(3, 4)

  # ── Phan 2: xBC = proj[C..C+cDim) -> conv1d -> SiLU, roi truot convState ──
  p.emitLoadConstF32(1, float32(cDim))
  p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2):
    p.emitAddConst(3, 0, float32(C))
    p.emitLoadF32Idx(7, 3)                       # R7 = xNew (giu suot block)
    p.emitMulConst(6, 0, float32(K - 1))         # R6 = tid*(K-1) (giu suot block)
    # acc = bias + sum_k w[c,k]*state[c,k] + w[c,K-1]*xNew   (vòng k chạy GPU)
    # Không cần "stage" state vào R8..: ghi trượt TUẦN TỰ tại chỗ vẫn an toàn vì
    # vòng acc xong hẳn trước, và vòng trượt đọc state[k+1] trước khi vòng k+1 ghi đè nó.
    p.emitAddConst(1, 0, float32(offConvB))
    p.emitLoadF32Idx(1, 1)                       # R1 = acc = bias[c]
    p.emitMulConst(5, 0, float32(K))             # R5 = tid*K
    p.emitLoadConstF32(8, 0.0'f32)               # R8 = k
    p.emitLoadConstF32(9, float32(K - 1))        # R9 = số vòng (hằng)
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
    p.emitLoadConstF32(5, 1.0'f32)               # R5 (tid*K) da dung xong
    p.emitDiv(3, 5, 3)
    p.emitMul(1, 1, 3)
    p.emitAddConst(4, 0, float32(offXBC))
    p.emitStoreF32Idx(1, 4)
    # truot cua so: state[k] = state[k+1] (k=0..K-3), state[K-2] = xNew  (vòng GPU)
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
  ## Y HET buildMamba2StepProgram o tren, chi khac o cho: MOI offset la
  ## THAM SO thay vi tinh cung tu 0 - cho phep doc/ghi thang tren hWork.
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
  ## gated = y * SiLU(z); moi nhom g (do dai L = C/nGroups):
  ##   r_g = rsqrt(mean_{i in g}(gated^2) + eps)
  ##   out[i] = gated[i] * r_{g(i)} * w[i]
  ## Dispatch: nThreads = nGroups*tgSize, threadgroup = tgSize -> MOI
  ## threadgroup lo DUNG 1 nhom. nGroups=1 tai tao CHINH XAC
  ## buildRmsGateProgram/buildRmsGateLargeProgram o tren (chuan hoa tren ca C).
  ## Dung OP_LOAD_LID/OP_LOAD_TGID (them moi trong session_vm.nim) vi VM cu
  ## chi co thread_position_in_grid, khong tach duoc lid/groupId.
  doAssert nGroups >= 1 and C mod nGroups == 0,
    "buildRmsGateGroupedAt: C phai chia het nGroups"
  let groupLen = C div nGroups
  doAssert tgSize >= 1 and groupLen mod tgSize == 0,
    "buildRmsGateGroupedAt: groupLen (" & $groupLen & ") phai chia het tgSize (" & $tgSize & ")"
  doAssert (tgSize and (tgSize - 1)) == 0, "buildRmsGateGroupedAt: tgSize phai la luy thua 2"
  let ept = groupLen div tgSize

  var p: VmProgram
  p.emitLoadLid(1)                                   # R1 = lid
  p.emitLoadTgid(2)                                  # R2 = groupId
  p.emitMulConst(2, 2, float32(groupLen))            # R2 = groupBase
  p.emitLoadConstF32(3, 0.0'f32)                     # R3 = localSum

  # ── pass 1: gated + localSum(gated^2), stash gated vao offOut ──
  p.emitLoadConstF32(8, float32(ept))
  p.emitLoadConstF32(7, 0.0'f32)
  sessionVmForGpu(p, 8):
    p.emitMulConst(9, 7, float32(tgSize))
    p.emitAdd(9, 9, 1)
    p.emitAdd(9, 9, 2)                               # R9 = i (chi so trong C)
    p.emitAddConst(10, 9, float32(offY));  p.emitLoadF32Idx(10, 10)
    p.emitAddConst(11, 9, float32(offZ));  p.emitLoadF32Idx(11, 11)
    p.emitNeg(12, 11); p.emitExp(12, 12); p.emitAddConst(12, 12, 1.0'f32)
    p.emitLoadConstF32(13, 1.0'f32); p.emitDiv(12, 13, 12)
    p.emitMul(12, 11, 12)                            # SiLU(z)
    p.emitMul(10, 10, 12)                            # gated
    p.emitAddConst(13, 9, float32(offOut)); p.emitStoreF32Idx(10, 13)
    p.emitMul(14, 10, 10); p.emitAdd(3, 3, 14)
    p.emitAddConst(7, 7, 1.0'f32)

  # ── reduce cay trong threadgroup (chi so = lid, KHONG phai tid toan cuc) ──
  p.emitTgStoreF32(3, 1, 0)
  p.emitBarrierTg()
  p.emitTgTreeReduce(1, tgSize, 4, 5, 6, 10, 11)   # GPU loop, không còn while host

  p.emitLoadConstF32(4, 0.0'f32)
  p.emitTgLoadF32(15, 4, 0)
  p.emitMulConst(15, 15, 1.0'f32 / float32(groupLen))
  p.emitAddConst(15, 15, eps)
  p.emitRsqrt(15, 15)                                # R15 = r_g (giu den het)

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
# BACKWARD SELECTIVE SCAN — 1 DISPATCH CHO TOÀN BỘ T BƯỚC, 100% GPU
# ═══════════════════════════════════════════════════════════════════════
## Bản cũ có 2 vấn đề: (1) buildMamba2StepBackwardAt cần VÒNG CPU qua t, lại ghi đè
## thanh ghi lẫn nhau (R12/R13 dùng cho 3 thứ) và gọi emitAtomicAdd chưa từng tồn tại;
## (2) buildMamba2BackwardFullAt chỉ là stub (`emitHalt` trong vòng) -> kernel không
## làm gì, selectiveScanBackwardGpu trả về toàn số 0 IM LẶNG. Cả hai thay bằng kernel
## dưới đây (đã đối chiếu với gradient sai phân số - xem t_mamba.nim).
##
## Toán (mỗi thread = 1 (b,h,d); s_t = s_{t-1}*dA_t + dt_t*B_t*x_t ; y_t = D*x_t + sum_n C_t[n]*s_t[n]):
##   dS_tn   = dS_{t+1,n}*dA_{t+1} + C_t[n]*dY_t
##   dC[t,g,n] += dY_t * s_t[n]                         (atomic - chung cho mọi h,d trong nhóm g)
##   dB[t,g,n] += dS_tn*dt_t*x_t                        (atomic)
##   dX[t]     += dS_tn*dt_t*B_t[n]  (+ dY_t*D ở pass skip)
##   dDt[t,h]  += dS_tn*s_{t-1}[n]*dA_t*A + dS_tn*B_t[n]*x_t   (atomic)
##   dALog[h]  += dS_tn*s_{t-1}[n]*dA_t*dt_t*A                 (atomic)
##   dDSkip[h] += dY_t*x_t                                      (atomic)
## dS_n chỉ phụ thuộc cùng n nên đảo vòng: n NGOÀI, t TRONG (lùi) -> dS nằm gọn trong
## 1 thanh ghi, không cần mảng per-thread. Địa chỉ đi qua U[] (uint32) vì
## B*(T+1)*HD*N dễ vượt 2^24 là mức float32 còn chính xác.
##
## Layout memF32 (offset tuyệt đối do caller đặt):
##   đọc : X[B,T,HD] Bm[B,T,G,N] Cm[B,T,G,N] Dt[B,T,H] ALog[H] DSkip[H]
##         State[B,T+1,HD,N] (State[:,0] = ban đầu, State[:,t+1] = sau bước t)  DY[B,T,HD]
##   ghi : DX[B,T,HD] (kernel tự ghi đè, không cần zero)
##   cộng dồn atomic -> caller PHẢI zero trước: DB DC DDt DALog DDSkip
## Grid = B*HD thread. h, g suy ra ngay trong kernel (không còn bảng idxH/idxG
## precompute trên CPU). Cần MSL 3.0+ (atomic_float) - chỉ chạy qua đường JIT.
proc buildMamba2BackwardFullAt*(B, T, H, D, N, G: int;
                                offX, offB, offC, offDt, offALog, offDSkip,
                                offState, offDY,
                                offDX, offDB, offDC, offDDt, offDALog, offDDSkip: int;
                                xStr = -1, bcStr = -1;
                                nBeg = 0, nCnt = -1): VmProgram =
  ## nBeg/nCnt: chỉ xử lý n in [nBeg, nBeg+nCnt) (mặc định cả N). Pass skip (dX=dY*D, dDSkip)
  ## chỉ chạy khi nBeg == 0. Chia nhỏ để mỗi command buffer < watchdog GPU (Intel ~vài giây);
  ## các lần gọi cộng dồn atomic nên kết quả cộng lại đúng bằng gọi 1 lần.
  ## xStr/bcStr: bước nhảy mỗi timestep của X/dX và B,C/dB,dC (mặc định HD và G*N =
  ## layout liền khối). Đặt = cDim để đọc/ghi THẲNG trong buffer conv [BT,cDim]
  ## (khi đó offB = offConv+C, offC = offConv+C+G*N, offDX/DB/DC trỏ vào dXBC cùng hàng).
  doAssert B >= 1 and T >= 1 and H mod G == 0, "buildMamba2BackwardFullAt: tham so sai"
  let HD = H * D
  let xS = if xStr < 0: HD else: xStr
  let bcS = if bcStr < 0: G * N else: bcStr
  let hpg = H div G
  let nC = if nCnt < 0: N - nBeg else: nCnt
  doAssert nBeg >= 0 and nC >= 1 and nBeg + nC <= N, "buildMamba2BackwardFullAt: khoang n sai"
  var p: VmProgram

  # ── giải mã tid -> b, hd, h, g (trunc((x+0.5)/m) tránh sai số làm tròn float) ──
  p.emitAddConst(1, 0, 0.5'f32)
  p.emitMulConst(1, 1, 1.0'f32 / float32(HD))
  p.emitRToU(0, 1)                                   # U0 = b
  p.emitUToR(2, 0)
  p.emitMulConst(3, 2, float32(HD))
  p.emitSub(4, 0, 3)                                 # R4 = hd (float, chính xác)
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

  # ── con trỏ gốc (uint32) ──
  p.emitUMulK(6, 0, uint32((T + 1) * HD * N)); p.emitUMulK(7, 3, uint32(N)); p.emitUAdd(6, 6, 7)   # U6 = stateBase
  p.emitUMulK(7, 0, uint32(T * xS)); p.emitUAdd(7, 7, 3)                                        # U7 = xBase
  p.emitUMulK(1, 0, uint32(T * HD)); p.emitUAdd(1, 1, 3)                                        # U1 = dYBase
  p.emitUMulK(8, 0, uint32(T * bcS)); p.emitUMulK(9, 5, uint32(N)); p.emitUAdd(8, 8, 9)         # U8 = bcBase
  p.emitUMulK(9, 0, uint32(T * H)); p.emitUAdd(9, 9, 4)                                         # U9 = dtBase

  # ── pass 1 (skip): dX = dY*D ; dDSkip[h] += dY*x   — vòng t chạy GPU ──
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
  p.emitLoadConstF32(0, float32(nC))                 # R0 = số vòng n (HẰNG; tid không còn cần)
  p.emitLoadConstF32(12, float32(T))                 # R12 = số vòng t (HẰNG)
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
# RMSNorm-gate BACKWARD — 1 threadgroup, 2 reduction (sum gated², dot) trên GPU
# ═══════════════════════════════════════════════════════════════════════
## Thay cho backwardRmsGateGpu cũ (CPU: `for v in gated2: sum += v`, `for i in 0..<C:
## dotSum += ...`, `newSeq(C).mapIt(...)` + ~20 dispatch beMul/beAdd rời).
## Layout memF32 (offset do caller): [yPre | z | w | dN | dYpre | dZ | dNormW] mỗi vùng C float
## Grid = tgSize thread, 1 threadgroup, tgSize = luỹ thừa 2 chia hết C (xem rmsBwdTgSize).
## tgBytes caller cấp >= tgSize*4.
proc rmsBwdTgSize*(C: int; cap = 1024): int =
  ## luỹ thừa 2 lớn nhất chia hết C, chặn trên bởi cap (bit trick, không vòng lặp)
  min(C and (-C), cap)

proc buildRmsGateBackwardProgram*(C: int; eps: float32; tgSize: int): VmProgram =
  doAssert tgSize >= 1 and (tgSize and (tgSize - 1)) == 0 and C mod tgSize == 0,
    "buildRmsGateBackwardProgram: tgSize phai la luy thua 2 va chia het C"
  let ept = C div tgSize
  let offY = 0; let offZ = C; let offW = 2*C; let offDN = 3*C
  let offDYp = 4*C; let offDZ = 5*C; let offDW = 6*C
  var p: VmProgram
  p.emitLoadConstF32(1, 0.0'f32)                 # sumG2
  p.emitLoadConstF32(2, 0.0'f32)                 # dotSum
  p.emitLoadConstF32(3, float32(ept))            # số vòng (HẰNG)
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

# ── Bảng chỉ số idxH/idxG sinh bằng GPU (thay cho vòng `for h/for d` trên CPU) ──
## Layout: [idxH(HD) | idxG(HD)], grid = HD thread. h = floor(tid/D) = #{k in 1..H-1 : tid >= k*D},
## g = floor(h/hpg) tương tự — chỉ dùng CMP_LT/SUB/ADD nên chạy được cả ở interpreter
## lẫn JIT (không cần op U-*, không cần floor).
proc buildIdxTableProgram*(H, D, G: int): VmProgram =
  let HD = H * D
  let hpg = max(1, H div G)
  var p: VmProgram
  p.emitLoadConstF32(1, 0.0'f32)                     # h
  p.emitLoadConstF32(2, 1.0'f32)                     # k
  p.emitLoadConstF32(3, float32(H - 1))              # số vòng (HẰNG)
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
