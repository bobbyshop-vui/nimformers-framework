## mamba2_vm_kernels.nim
## Build 2 VmProgram (session_vm.nim) chạy phần SSM recurrence + RMSNorm
## gate của Mamba2 100% trên GPU. Dùng cùng opcode mới trong session_vm.nim
## (EXP, NEG, SIGMOID, RSQRT, LOAD/STORE_F32_IDX, IF_BEGIN/END, CMP_LT...).
## Dispatch qua session_vm_dispatch.nim (metal_enc_* thật, không giả lập).

import session_vm

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

  # reduction cây: log2(C) bước cố định tại BUILD time -> unroll bằng
  # vòng Nim host (KHÔNG phải GPU for) vì số bước biết trước theo C.
  var stride = C div 2
  while stride >= 1:
    p.emitLoadConstF32(6, float32(stride))
    p.emitCmpLt(7, 0, 6)                               # R7 = (tid<stride)
    sessionVmIfGpu(p, 7):
      p.emitAddConst(8, 0, float32(stride))            # R8 = tid+stride
      p.emitTgLoadF32(9, 0, 0)
      p.emitTgLoadF32(10, 8, 0)
      p.emitAdd(9, 9, 10)
      p.emitTgStoreF32(9, 0, 0)
    p.emitBarrierTg()
    stride = stride div 2

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

  for k in 0 ..< (K-1):
    # state addr = offState + tid*(K-1) + k
    p.emitMulConst(3, 0, float32(K-1))
    p.emitAddConst(3, 3, float32(k + offState))
    p.emitLoadF32Idx(2, 3)
    # weight addr = offW + tid*K + k
    p.emitMulConst(4, 0, float32(K))
    p.emitAddConst(4, 4, float32(k + offW))
    p.emitLoadF32Idx(4, 4)
    p.emitFma(1, 4, 2, 1)

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
  for k in 0 ..< (K-2):
    p.emitMulConst(3, 0, float32(K-1))
    p.emitAddConst(3, 3, float32(k+1 + offState))
    p.emitLoadF32Idx(2, 3)
    p.emitMulConst(4, 0, float32(K-1))
    p.emitAddConst(4, 4, float32(k + offState))
    p.emitStoreF32Idx(2, 4)
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
    for k in 0 ..< (K-1):
      p.emitMulConst(3, 0, float32(K-1))
      p.emitAddConst(3, 3, float32(k + offConvSt))
      p.emitLoadF32Idx(2, 3)                    # R2 = state[c,k]
      p.emitMulConst(4, 0, float32(K))
      p.emitAddConst(4, 4, float32(k + offConvW))
      p.emitLoadF32Idx(4, 4)                    # R4 = weight[c,k]
      p.emitFma(1, 4, 2, 1)
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
    for k in 0 ..< (K-2):
      p.emitMulConst(3, 0, float32(K-1))
      p.emitAddConst(3, 3, float32(k+1 + offConvSt))
      p.emitLoadF32Idx(2, 3)
      p.emitMulConst(4, 0, float32(K-1))
      p.emitAddConst(4, 4, float32(k + offOutConvSt))
      p.emitStoreF32Idx(2, 4)
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

  var stride = tgSize div 2
  while stride >= 1:
    p.emitLoadConstF32(6, float32(stride))
    p.emitCmpLt(7, 0, 6)
    sessionVmIfGpu(p, 7):
      p.emitAddConst(8, 0, float32(stride))
      p.emitTgLoadF32(9, 0, 0)
      p.emitTgLoadF32(10, 8, 0)
      p.emitAdd(9, 9, 10)
      p.emitTgStoreF32(9, 0, 0)
    p.emitBarrierTg()
    stride = stride div 2

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
    # doc HET convState vao R8..R(8+K-2) TRUOC khi ghi bat cu gi
    for k in 0 ..< (K - 1):
      p.emitAddConst(3, 6, float32(k + offConvSt))
      p.emitLoadF32Idx(uint32(8 + k), 3)
    # acc = bias + sum_k w[c,k]*state[c,k] + w[c,K-1]*xNew
    p.emitAddConst(1, 0, float32(offConvB))
    p.emitLoadF32Idx(1, 1)
    p.emitMulConst(5, 0, float32(K))             # R5 = tid*K
    for k in 0 ..< (K - 1):
      p.emitAddConst(4, 5, float32(k + offConvW))
      p.emitLoadF32Idx(4, 4)
      p.emitFma(1, 4, uint32(8 + k), 1)
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
    # truot cua so: state[k] = state[k+1] (k=0..K-3), state[K-2] = xNew
    for k in 0 ..< (K - 2):
      p.emitAddConst(3, 6, float32(k + offConvSt))
      p.emitStoreF32Idx(uint32(9 + k), 3)
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
  var stride = tgSize div 2
  while stride >= 1:
    p.emitLoadConstF32(4, float32(stride))
    p.emitCmpLt(5, 1, 4)
    sessionVmIfGpu(p, 5):
      p.emitAddConst(6, 1, float32(stride))
      p.emitTgLoadF32(10, 1, 0)
      p.emitTgLoadF32(11, 6, 0)
      p.emitAdd(10, 10, 11)
      p.emitTgStoreF32(10, 1, 0)
    p.emitBarrierTg()
    stride = stride div 2

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

