## mamba2_full_vm.nim — forward + backward của CẢ lớp Mamba2 (mọi token) bằng micro-kernel.
## Thay cho các vòng for b/t/c/h/d/n trên CPU trong mamba2_real.nim.
##
## Quy ước: 1 buffer float32 duy nhất (layout M2Lay), mỗi stage là 1 kernel.
## Kernel "mỗi token = 1 threadgroup" (TG = M2Tg lane, lane lặp qua kênh bằng vòng FOR
## trên GPU) -> địa chỉ tính bằng U[] (uint32) nên không dính giới hạn 2^24 của float32.
## Gradient cộng dồn dùng OP_U_ATOMIC_ADD (MSL 3.0+, chỉ qua đường JIT) -> caller phải
## để các vùng d* ở trạng thái 0 trước khi chạy (newSeq đã zero-init).
##
## Thứ tự dispatch (cùng buffer, tuần tự trong 1 session):
##   forward : (convSilu + dtSoftplus fused) -> scan -> rmsFwd
##   backward: rmsBwd -> scanBwd -> dtBwd -> convBwd

import std/bitops
import session_vm
import mamba2_vm_kernels

const M2Tg* = 256            ## lane / threadgroup cho kernel conv/dt (<= mọi GPU)

type
  M2Lay* = object
    B*, T*, H*, D*, N*, G*, K*, C*, cDim*, P*, HD*, GN*, BT*, normGroups*: int
    eps*: float32
    # ── forward ──
    proj*, convW*, convB*, aLog*, dSkip*, dtBias*, normW*: int
    pre*, conv*, dt*, states*, yPre*, gated*, rms*, normedPre*, normedOut*: int
    fwdEnd*: int
    # ── backward ──
    dNormedOut*, dYpre*, dProj*, dXBC*, dDt*, dDtBias*, dALog*, dDSkip*: int
    dConvW*, dConvB*, dNormW*: int
    total*: int

proc m2Layout*(B, T, H, D, N, G, K, C: int; eps: float32;
               normGroups = 0): M2Lay =
  ## Mọi offset TUYỆT ĐỐI (float32). proj row = [z(C) | xBC(cDim) | dt(H)], P = C+cDim+H.
  var L: M2Lay
  L.B = B; L.T = T; L.H = H; L.D = D; L.N = N; L.G = G; L.K = K; L.C = C; L.eps = eps
  L.normGroups = if normGroups <= 0: G else: normGroups
  doAssert L.normGroups > 0 and C mod L.normGroups == 0,
    "m2Layout: intermediateSize phai chia het gated RMSNorm groups"
  L.HD = H * D; L.GN = G * N; L.BT = B * T
  L.cDim = C + 2 * L.GN; L.P = C + L.cDim + H
  var o = 0
  template take(f: untyped; n: int) =
    f = o; o += n
  take(L.proj, L.BT * L.P)
  take(L.convW, L.cDim * K); take(L.convB, L.cDim)
  take(L.aLog, H); take(L.dSkip, H); take(L.dtBias, H); take(L.normW, C)
  take(L.pre, L.BT * L.cDim); take(L.conv, L.BT * L.cDim); take(L.dt, L.BT * H)
  take(L.states, B * (T + 1) * L.HD * N)
  take(L.yPre, L.BT * C); take(L.gated, L.BT * C); take(L.rms, L.BT * L.normGroups)
  take(L.normedPre, L.BT * C); take(L.normedOut, L.BT * C)
  L.fwdEnd = o
  take(L.dNormedOut, L.BT * C); take(L.dYpre, L.BT * C); take(L.dProj, L.BT * L.P)
  take(L.dXBC, L.BT * L.cDim); take(L.dDt, L.BT * H); take(L.dDtBias, H)
  take(L.dALog, H); take(L.dDSkip, H)
  take(L.dConvW, L.cDim * K); take(L.dConvB, L.cDim); take(L.dNormW, C)
  L.total = o
  doAssert L.total < (1 shl 31), "m2Layout: buffer qua lon cho dia chi uint32/offset"
  doAssert H mod G == 0 and C == L.HD, "m2Layout: C phai = H*D, H chia het G"
  result = L

var gTgCap* = 1024
  ## Số lane/threadgroup tối đa của GPU (kernel reduction PHẢI dispatch đúng tgSize, không được hạ âm thầm).
  ## Intel Iris Plus 640 chỉ cho 256. mamba2_ssm_gpu.ensureTgCap() đặt giá trị này từ
  ## maxTotalThreadsPerThreadgroup thật (hoặc env BYBY_RMS_TG) TRƯỚC khi build program.

proc m2Tag*(L: M2Lay): string =
  ## Dinh danh layout: moi program m2_* chi phu thuoc cac so nay -> dung lam tag '@...' (hash 1 lan).
  $L.B & "_" & $L.T & "_" & $L.H & "_" & $L.D & "_" & $L.N & "_" & $L.G & "_" & $L.K & "_" &
    $L.C & "_ng" & $L.normGroups & "_" & $L.eps & "_" &
    $min((L.C div L.normGroups) and (-(L.C div L.normGroups)), gTgCap)

proc rmsTg*(L: M2Lay): int =
  let groupLen = L.C div L.normGroups
  min(groupLen and (-groupLen), gTgCap)

proc emitTokenNormGroup(p: var VmProgram; groups: int) =
  ## Kernel grid is [token * groups + group]. Recover both indices without
  ## integer divide/mod opcodes (these grid ids stay exactly representable).
  p.emitUTgid(0)
  p.emitUToR(1, 0)
  p.emitLoadConstF32(2, float32(groups))
  p.emitDiv(1, 1, 2)
  p.emitRToU(2, 1)                  # U2 = token = floor(tgid / groups)
  p.emitUMulK(3, 2, uint32(groups))
  p.emitUToR(4, 0)
  p.emitUToR(5, 3)
  p.emitSub(4, 4, 5)
  p.emitRToU(3, 4)                  # U3 = group = tgid - token*groups

# ── helper emit ──────────────────────────────────────────────
proc emitToken(p: var VmProgram; T: int) =
  ## U0 = token (tgid), U1 = lid, R1 = float(token), R2 = t = token mod T (float)
  p.emitUTgid(0); p.emitULid(1)
  p.emitUToR(1, 0)
  p.emitAddConst(2, 1, 0.5'f32)
  p.emitLoadConstF32(3, float32(T))
  p.emitDiv(2, 2, 3)
  p.emitRToU(2, 2)                      # U2 = b = trunc((token+.5)/T)
  p.emitUToR(2, 2)
  p.emitMulConst(2, 2, float32(T))
  p.emitSub(2, 1, 2)                    # R2 = t

proc emitDPre(p: var VmProgram; offPre, offDXBC: int; ptrU, rOut, ra, rb, rc: uint32) =
  ## rOut = dXBC[ptr]*dSilu(pre[ptr]);  dSilu(x) = s*(1 + x*(1-s)), s = sigmoid(x)
  p.emitULoadF32(ra, ptrU, uint32(offPre))        # pre
  p.emitULoadF32(rb, ptrU, uint32(offDXBC))       # dxbc
  p.emitSigmoid(rc, ra)                           # s
  p.emitLoadConstF32(rOut, 1.0'f32); p.emitSub(rOut, rOut, rc)
  p.emitMul(rOut, ra, rOut); p.emitAddConst(rOut, rOut, 1.0'f32)
  p.emitMul(rOut, rc, rOut); p.emitMul(rOut, rb, rOut)

# ── F1: conv1d causal depthwise + SiLU ───────────────────────
proc buildM2ConvSiluProgram*(L: M2Lay): VmProgram =
  let K = L.K; let P = L.P; let cDim = L.cDim
  var p: VmProgram
  p.emitToken(L.T)
  p.emitUMulK(3, 0, uint32(P))                    # U3 = token*P
  p.emitUMulK(4, 0, uint32(cDim))                 # U4 = token*cDim
  p.emitLoadConstF32(4, float32((cDim + M2Tg - 1) div M2Tg))   # số vòng e (HẰNG)
  p.emitLoadConstF32(5, 0.0'f32)                  # e
  p.emitLoadConstF32(12, float32(K - 1))
  p.emitLoadConstF32(13, float32(K))
  sessionVmForGpu(p, 4):
    p.emitUToR(6, 1)                              # lid
    p.emitMulConst(7, 5, float32(M2Tg)); p.emitAdd(6, 6, 7)       # R6 = c
    p.emitLoadConstF32(7, float32(cDim)); p.emitCmpLt(7, 6, 7)
    sessionVmIfGpu(p, 7):
      p.emitRToU(5, 6)                            # U5 = c
      p.emitULoadF32(8, 5, uint32(L.convB))       # acc = bias[c]
      p.emitUAdd(6, 3, 5); p.emitUAddK(6, 6, uint32(L.C))
      p.emitUAddK(6, 6, uint32(0xFFFFFFFF'u32 - uint32((K - 1) * P) + 1))   # x ptr tại k=0
      p.emitUMulK(7, 5, uint32(K))                # w ptr
      p.emitAddConst(11, 2, 0.0'f32)              # R11 = t + k
      sessionVmForGpu(p, 13):
        p.emitCmpLt(14, 11, 12)                   # t+k < K-1 -> tap ngoài biên
        p.emitLoadConstF32(15, 1.0'f32); p.emitSub(15, 15, 14)
        sessionVmIfGpu(p, 15):
          p.emitULoadF32(9, 6, uint32(L.proj))
          p.emitULoadF32(10, 7, uint32(L.convW))
          p.emitFma(8, 10, 9, 8)
        p.emitUAddK(6, 6, uint32(P)); p.emitUAddK(7, 7, 1)
        p.emitAddConst(11, 11, 1.0'f32)
      p.emitUAdd(8, 4, 5)                         # U8 = token*cDim + c
      p.emitUStoreF32(8, 8, uint32(L.pre))
      p.emitNeg(14, 8); p.emitExp(14, 14); p.emitAddConst(14, 14, 1.0'f32)
      p.emitLoadConstF32(15, 1.0'f32); p.emitDiv(14, 15, 14); p.emitMul(8, 8, 14)
      p.emitUStoreF32(8, 8, uint32(L.conv))
    p.emitAddConst(5, 5, 1.0'f32)
  p.emitHalt()
  result = p

# ── F2: dt = softplus(dtPre + dtBias) ────────────────────────
proc buildM2DtProgram*(L: M2Lay): VmProgram =
  let H = L.H
  var p: VmProgram
  p.emitToken(L.T)
  p.emitUMulK(3, 0, uint32(L.P))
  p.emitUMulK(4, 0, uint32(H))
  p.emitLoadConstF32(4, float32((H + M2Tg - 1) div M2Tg))
  p.emitLoadConstF32(5, 0.0'f32)
  sessionVmForGpu(p, 4):
    p.emitUToR(6, 1); p.emitMulConst(7, 5, float32(M2Tg)); p.emitAdd(6, 6, 7)
    p.emitLoadConstF32(7, float32(H)); p.emitCmpLt(7, 6, 7)
    sessionVmIfGpu(p, 7):
      p.emitRToU(5, 6)
      p.emitUAdd(6, 3, 5)
      p.emitULoadF32(8, 6, uint32(L.proj + L.C + L.cDim))
      p.emitULoadF32(9, 5, uint32(L.dtBias))
      p.emitAdd(8, 8, 9); p.emitSoftplus(8, 8)
      p.emitUAdd(6, 4, 5); p.emitUStoreF32(8, 6, uint32(L.dt))
    p.emitAddConst(5, 5, 1.0'f32)
  p.emitHalt()
  result = p

proc buildM2ConvDtProgram*(L: M2Lay): VmProgram =
  ## Conv/SiLU and dt/softplus read disjoint projections and write disjoint
  ## scratch regions. They share the same token×M2Tg launch geometry, so
  ## concatenate their bytecode bodies into one safe physical dispatch.
  result = buildM2ConvSiluProgram(L)
  doAssert result.words.len > 0 and (result.words[^1] and 0xFF'u32) == OP_HALT
  result.words.setLen(result.words.len - 1)
  let dt = buildM2DtProgram(L)
  result.words.add(dt.words)

# ── F3: selective scan, mọi timestep, 1 thread / (b,h,d) ─────
proc emitDecodeBHD(p: var VmProgram; L: M2Lay) =
  ## R0 = tid. Kết quả: U0=b, U1=hd, U2=h, U3=g.
  let hpg = L.H div L.G
  p.emitAddConst(1, 0, 0.5'f32); p.emitLoadConstF32(2, float32(L.HD)); p.emitDiv(1, 1, 2)
  p.emitRToU(0, 1)                                 # U0 = b
  p.emitUToR(3, 0); p.emitMulConst(3, 3, float32(L.HD)); p.emitSub(4, 0, 3)
  p.emitRToU(1, 4)                                 # U1 = hd
  p.emitAddConst(5, 4, 0.5'f32); p.emitLoadConstF32(2, float32(L.D)); p.emitDiv(5, 5, 2)
  p.emitRToU(2, 5)                                 # U2 = h
  p.emitUToR(6, 2); p.emitAddConst(6, 6, 0.5'f32)
  p.emitLoadConstF32(2, float32(hpg)); p.emitDiv(6, 6, 2)
  p.emitRToU(3, 6)                                 # U3 = g

proc buildM2ScanProgram*(L: M2Lay): VmProgram =
  let T = L.T; let N = L.N; let HD = L.HD; let H = L.H; let cDim = L.cDim
  var p: VmProgram
  p.emitDecodeBHD(L)
  p.emitULoadF32(8, 2, uint32(L.aLog)); p.emitExp(8, 8); p.emitNeg(8, 8)    # R8 = A
  p.emitULoadF32(9, 2, uint32(L.dSkip))                                     # R9 = D[h]
  p.emitUMulK(4, 0, uint32(T * cDim))                                       # U4 = convT
  p.emitUMulK(5, 0, uint32(T * H)); p.emitUAdd(5, 5, 2)                     # U5 = dtT
  p.emitUMulK(6, 0, uint32(T * L.C)); p.emitUAdd(6, 6, 1)                   # U6 = yT
  p.emitUMulK(7, 0, uint32((T + 1) * HD * N)); p.emitUMulK(8, 1, uint32(N)); p.emitUAdd(7, 7, 8)  # U7 = sT
  p.emitUMulK(9, 3, uint32(N))                                              # U9 = g*N
  p.emitLoadConstF32(12, float32(T))
  sessionVmForGpu(p, 12):
    p.emitUAdd(10, 4, 1); p.emitULoadF32(3, 10, uint32(L.conv))            # x
    p.emitULoadF32(1, 5, uint32(L.dt))                                     # dt
    p.emitMul(2, 1, 8); p.emitExp(2, 2)                                    # dA
    p.emitLoadConstF32(4, 0.0'f32)                                         # yAcc
    p.emitUAdd(11, 4, 9)                                                   # bc ptr
    p.emitUAddK(12, 7, 0)                                                  # s ptr
    p.emitLoadConstF32(11, float32(N))
    sessionVmForGpu(p, 11):
      p.emitULoadF32(5, 11, uint32(L.conv + L.C))                          # B
      p.emitULoadF32(6, 11, uint32(L.conv + L.C + L.GN))                   # C
      p.emitULoadF32(7, 12, uint32(L.states))                              # sPrev
      p.emitMul(7, 7, 2)
      p.emitMul(13, 1, 5); p.emitMul(13, 13, 3); p.emitAdd(7, 7, 13)       # sNew
      p.emitUStoreF32(7, 12, uint32(L.states + HD * N))
      p.emitFma(4, 6, 7, 4)
      p.emitUAddK(11, 11, 1); p.emitUAddK(12, 12, 1)
    p.emitFma(4, 9, 3, 4)                                                  # += D*x
    p.emitUStoreF32(4, 6, uint32(L.yPre))
    p.emitUAddK(4, 4, uint32(cDim)); p.emitUAddK(5, 5, uint32(H))
    p.emitUAddK(6, 6, uint32(L.C)); p.emitUAddK(7, 7, uint32(HD * N))
  p.emitHalt()
  result = p

proc m2ScanTgSize*(T: int): int =
  ## Power-of-two lanes for the timestep scan. A zero result means the
  ## sequence is too long for one safe threadgroup and must use the serial
  ## reference kernel (Metal has no grid-wide barrier).
  if T <= 1: return 0
  const maxScanTg = 128 # support the safe 128-token training context; shared scan storage is only 4*tg floats
  let cap = min(gTgCap, maxScanTg)
  var tg = 1
  while tg < T and tg <= cap div 2:
    tg *= 2
  if tg < T or tg > cap: return 0
  result = tg

proc buildM2ScanParallelProgram*(L: M2Lay; tg: int): VmProgram =
  ## Inclusive affine prefix scan over time, one threadgroup per (batch, head,
  ## channel). Each lane is a timestep; N remains a short per-lane loop to
  ## avoid exploding the grid by another dState factor. The affine operator
  ## (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1) is associative, so Hillis-Steele
  ## gives the same recurrence as buildM2ScanProgram, modulo FP32 order.
  ##
  ## Two ping-pong banks hold (a,b) pairs. Padded lanes are the identity
  ## (1,0), and every lane reaches each barrier. Output state[t+1] is written
  ## in the exact layout consumed by the backward kernel.
  doAssert tg >= L.T and (tg and (tg - 1)) == 0 and tg <= gTgCap,
    "Mamba2 parallel scan requires a power-of-two threadgroup >= T"
  let T = L.T
  let N = L.N
  let HD = L.HD
  let cDim = L.cDim
  var p: VmProgram
  p.emitUTgid(0)                    # one group per (b,hd)
  p.emitULid(1)                     # lane = timestep
  p.emitUToR(0, 0)                  # R0 = group/channel id for emitDecodeBHD
  p.emitUToR(14, 1)
  p.emitRToU(4, 14)                 # U4 = t
  p.emitDecodeBHD(L)                # U0=b, U1=hd, U2=h, U3=g

  # Per-lane addresses: convolution row, dt, output state[t+1], yPre.
  p.emitUMulK(5, 0, uint32(T * cDim))
  p.emitUMulK(10, 4, uint32(cDim)); p.emitUAdd(5, 5, 10)       # conv row
  p.emitUMulK(6, 0, uint32(T * L.H))
  p.emitUMulK(10, 4, uint32(L.H)); p.emitUAdd(6, 6, 10)
  p.emitUAdd(6, 6, 2)                                          # dt[h]
  p.emitUMulK(7, 0, uint32((T + 1) * HD * N))
  p.emitUAddK(10, 4, 1); p.emitUMulK(10, 10, uint32(HD * N))
  p.emitUAdd(7, 7, 10); p.emitUMulK(10, 1, uint32(N)); p.emitUAdd(7, 7, 10)
  p.emitUMulK(8, 3, uint32(N))                                 # group*N
  p.emitUMulK(9, 0, uint32(T * L.C))
  p.emitUMulK(10, 4, uint32(L.C)); p.emitUAdd(9, 9, 10)
  p.emitUAdd(9, 9, 1)                                          # yPre[t,hd]
  p.emitUAdd(10, 5, 1)                                         # conv[t,hd]
  p.emitUAdd(14, 5, 8); p.emitUAddK(14, 14, uint32(L.C))       # B row base
  p.emitUAdd(15, 5, 8)
  p.emitUAddK(15, 15, uint32(L.C + L.GN))                     # C row base

  # R7 is the active-lane predicate and is preserved through all N scans.
  p.emitUToR(10, 4); p.emitLoadConstF32(11, float32(T))
  p.emitCmpLt(7, 10, 11)
  p.emitLoadConstF32(4, 0.0'f32)                               # y accumulator
  p.emitUConst(13, 0)                                          # n index
  p.emitLoadConstF32(12, float32(N))
  sessionVmForGpu(p, 12):
    # Neutral affine pair for padding. Valid lanes replace it with
    # (exp(dt*A), dt*B*x). Recompute A_h each n because scan scratch registers
    # are reused below.
    p.emitLoadConstF32(2, 1.0'f32)
    p.emitLoadConstF32(5, 0.0'f32)
    p.emitULoadF32(8, 2, uint32(L.aLog)); p.emitExp(8, 8); p.emitNeg(8, 8)
    sessionVmIfGpu(p, 7):
      p.emitULoadF32(1, 6, uint32(L.dt))
      p.emitMul(2, 1, 8); p.emitExp(2, 2)                     # a_t
      p.emitULoadF32(3, 10, uint32(L.conv))                   # x_t,hd
      p.emitUAdd(11, 14, 13)
      p.emitULoadF32(5, 11, uint32(L.conv))                   # B_t,g,n
      p.emitMul(5, 1, 5); p.emitMul(5, 5, 3)                 # b_t
    p.emitTgStoreF32(2, 14, 0)                                # bank 0: a
    p.emitTgStoreF32(5, 14, uint32(tg))                       # bank 0: b
    p.emitBarrierTg()

    var srcA = 0
    var srcB = tg
    var dstA = 2 * tg
    var dstB = 3 * tg
    var stride = 1
    while stride < tg:
      p.emitUToR(11, 4)                                       # current lane
      p.emitAddConst(10, 11, -float32(stride))                # previous lane
      p.emitTgLoadF32(2, 11, uint32(srcA))
      p.emitTgLoadF32(5, 11, uint32(srcB))
      p.emitLoadConstF32(8, float32(stride))
      p.emitCmpLt(9, 11, 8)
      p.emitLoadConstF32(8, 1.0'f32); p.emitSub(9, 8, 9)    # lane >= stride
      sessionVmIfGpu(p, 9):
        p.emitTgLoadF32(3, 10, uint32(srcA))
        p.emitTgLoadF32(6, 10, uint32(srcB))
        p.emitMul(1, 2, 3)                                    # new a = a * aPrev
        p.emitFma(5, 2, 6, 5)                                 # new b = b + a*bPrev
        p.emitAddConst(2, 1, 0.0'f32)
      p.emitTgStoreF32(2, 14, uint32(dstA))
      p.emitTgStoreF32(5, 14, uint32(dstB))
      p.emitBarrierTg()
      swap(srcA, dstA); swap(srcB, dstB)
      stride *= 2

    # Prefix state is now in the last source bank. Store each valid state and
    # accumulate C*s; the D skip contribution is added after the n loop.
    sessionVmIfGpu(p, 7):
      p.emitUAdd(11, 14, 13); p.emitUAdd(12, 15, 13)
      p.emitUAdd(0, 7, 13)                                    # states[t+1,hd,n]
      p.emitUToR(11, 4)
      p.emitTgLoadF32(5, 11, uint32(srcB))
      p.emitUStoreF32(5, 0, uint32(L.states))
      p.emitULoadF32(6, 12, uint32(L.conv))                   # C_t,g,n
      p.emitFma(4, 6, 5, 4)
    p.emitUAddK(13, 13, 1)

  sessionVmIfGpu(p, 7):
    p.emitULoadF32(3, 10, uint32(L.conv))
    p.emitULoadF32(2, 2, uint32(L.dSkip))
    p.emitFma(4, 2, 3, 4)
    p.emitUStoreF32(4, 9, uint32(L.yPre))
  p.emitHalt()
  result = p

# ── F4: gated RMSNorm, 1 threadgroup / token (tgSize = rmsTg) ─
proc buildM2RmsFwdProgram*(L: M2Lay): VmProgram =
  let C = L.C; let groups = L.normGroups
  let groupLen = C div groups
  let tg = rmsTg(L); let ept = groupLen div tg
  var p: VmProgram
  emitTokenNormGroup(p, groups)
  p.emitULid(1)
  p.emitUToR(14, 1)                                # R14 = lid (float)
  p.emitUMulK(4, 2, uint32(L.P))                   # U4 = token*P   (z nằm đầu hàng proj)
  p.emitUMulK(5, 2, uint32(C))                     # U5 = token*C
  p.emitUMulK(6, 3, uint32(groupLen))
  p.emitUAdd(5, 5, 6)                              # U5 = token*C + groupBase
  p.emitLoadConstF32(1, 0.0'f32)                   # sum gated^2
  p.emitLoadConstF32(3, float32(ept))              # số vòng (HẰNG)
  p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(7, 4, float32(tg)); p.emitAdd(7, 7, 14); p.emitRToU(8, 7)
    p.emitUAdd(9, 6, 8)                                                        # group-local C index
    p.emitUAdd(10, 5, 8)                                                       # token-C index
    p.emitUAdd(11, 4, 9); p.emitULoadF32(6, 11, uint32(L.proj))                # z
    p.emitULoadF32(7, 10, uint32(L.yPre))                                      # y
    p.emitNeg(8, 6); p.emitExp(8, 8); p.emitAddConst(8, 8, 1.0'f32)
    p.emitLoadConstF32(9, 1.0'f32); p.emitDiv(8, 9, 8); p.emitMul(8, 6, 8)     # silu
    p.emitMul(7, 7, 8)                                                         # gated
    p.emitUStoreF32(7, 10, uint32(L.gated))
    p.emitMul(10, 7, 7); p.emitAdd(1, 1, 10)
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitTgStoreF32(1, 14, 0); p.emitBarrierTg()
  p.emitTgTreeReduce(14, tg, 6, 7, 8, 9, 10)
  p.emitLoadConstF32(11, 0.0'f32); p.emitTgLoadF32(13, 11, 0)                  # tổng
  p.emitMulConst(13, 13, 1.0'f32 / float32(groupLen)); p.emitAddConst(13, 13, L.eps)
  p.emitRsqrt(13, 13)                                                          # r
  p.emitUStoreF32(13, 0, uint32(L.rms))
  p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(7, 4, float32(tg)); p.emitAdd(7, 7, 14); p.emitRToU(8, 7)
    p.emitUAdd(9, 6, 8); p.emitUAdd(10, 5, 8)
    p.emitULoadF32(7, 10, uint32(L.gated))
    p.emitMul(7, 7, 13)                                                        # normedPre
    p.emitUStoreF32(7, 10, uint32(L.normedPre))
    p.emitULoadF32(8, 9, uint32(L.normW)); p.emitMul(7, 7, 8)
    p.emitUStoreF32(7, 10, uint32(L.normedOut))
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitHalt()
  result = p

# ── B1: RMSNorm-gate backward theo token (dZ ghi thẳng vào dProj) ─
proc buildM2RmsBwdProgram*(L: M2Lay): VmProgram =
  let groups = L.normGroups
  let groupLen = L.C div groups
  let tg = rmsTg(L); let ept = groupLen div tg
  var p: VmProgram
  emitTokenNormGroup(p, groups)
  p.emitULid(1)
  p.emitUToR(14, 1)
  p.emitUMulK(4, 2, uint32(L.P))
  p.emitUMulK(5, 2, uint32(L.C))
  p.emitUMulK(6, 3, uint32(groupLen))
  p.emitUAdd(5, 5, 6)                              # token*C + groupBase
  p.emitLoadConstF32(1, 0.0'f32); p.emitLoadConstF32(2, 0.0'f32)
  p.emitLoadConstF32(3, float32(ept)); p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(7, 4, float32(tg)); p.emitAdd(7, 7, 14); p.emitRToU(8, 7)
    p.emitUAdd(9, 6, 8)                                                        # norm-weight index
    p.emitUAdd(10, 5, 8)                                                       # token-C index
    p.emitUAdd(11, 4, 9); p.emitULoadF32(6, 11, uint32(L.proj))                # z
    p.emitULoadF32(7, 10, uint32(L.yPre))                                      # y
    p.emitNeg(8, 6); p.emitExp(8, 8); p.emitAddConst(8, 8, 1.0'f32)
    p.emitLoadConstF32(9, 1.0'f32); p.emitDiv(9, 9, 8)                         # sigmoid(z)
    p.emitMul(10, 6, 9)                                                        # silu(z)
    p.emitMul(11, 7, 10)                                                       # gated
    p.emitMul(12, 11, 11); p.emitAdd(1, 1, 12)                                 # sum gated^2
    p.emitULoadF32(12, 10, uint32(L.dNormedOut)); p.emitULoadF32(13, 9, uint32(L.normW))
    p.emitMul(12, 12, 13); p.emitMul(12, 12, 11); p.emitAdd(2, 2, 12)          # dot
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitLoadConstF32(11, 0.0'f32)
  p.emitTgStoreF32(1, 14, 0); p.emitBarrierTg()
  p.emitTgTreeReduce(14, tg, 6, 7, 8, 9, 10)
  p.emitTgLoadF32(13, 11, 0); p.emitBarrierTg()                                # sumG2
  p.emitTgStoreF32(2, 14, 0); p.emitBarrierTg()
  p.emitTgTreeReduce(14, tg, 6, 7, 8, 9, 10)
  p.emitTgLoadF32(15, 11, 0)                                                   # dot
  p.emitMulConst(13, 13, 1.0'f32 / float32(groupLen)); p.emitAddConst(13, 13, L.eps); p.emitRsqrt(13, 13)  # r
  p.emitMul(10, 13, 13); p.emitMul(10, 10, 13)
  p.emitMulConst(10, 10, 1.0'f32 / float32(groupLen)); p.emitMul(15, 10, 15)    # r^3/groupLen*dot
  p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(7, 4, float32(tg)); p.emitAdd(7, 7, 14); p.emitRToU(8, 7)
    p.emitUAdd(9, 6, 8); p.emitUAdd(10, 5, 8)
    p.emitUAdd(11, 4, 9)
    p.emitULoadF32(6, 11, uint32(L.proj)); p.emitULoadF32(7, 10, uint32(L.yPre))
    p.emitNeg(8, 6); p.emitExp(8, 8); p.emitAddConst(8, 8, 1.0'f32)
    p.emitLoadConstF32(9, 1.0'f32); p.emitDiv(9, 9, 8)                         # sigmoid(z)
    p.emitMul(10, 6, 9)                                                        # silu(z)
    p.emitMul(12, 7, 10)                                                       # gated
    p.emitULoadF32(11, 10, uint32(L.dNormedOut)); p.emitULoadF32(1, 9, uint32(L.normW))
    p.emitMul(11, 11, 1)                                                       # dOut * gamma
    p.emitMul(2, 11, 13); p.emitMul(1, 12, 15); p.emitSub(2, 2, 1)             # dGated
    p.emitMul(1, 2, 10); p.emitUStoreF32(1, 10, uint32(L.dYpre))               # dYpre
    p.emitULoadF32(1, 10, uint32(L.dNormedOut)); p.emitMul(1, 1, 12); p.emitMul(1, 1, 13)
    p.emitUAtomicAdd(1, 9, uint32(L.dNormW))
    p.emitLoadConstF32(1, 1.0'f32); p.emitSub(1, 1, 9); p.emitMul(1, 6, 1)
    p.emitAddConst(1, 1, 1.0'f32); p.emitMul(1, 9, 1)                          # silu'(z)
    p.emitMul(1, 1, 2); p.emitMul(1, 1, 7); p.emitUStoreF32(1, 11, uint32(L.dProj)) # dZ
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitHalt()
  result = p

# ── B2: selective-scan backward đọc/ghi thẳng buffer conv/dXBC ─
proc buildM2ScanBwdProgram*(L: M2Lay; nBeg = 0; nCnt = -1): VmProgram =
  buildMamba2BackwardFullAt(L.B, L.T, L.H, L.D, L.N, L.G,
    L.conv, L.conv + L.C, L.conv + L.C + L.GN, L.dt, L.aLog, L.dSkip,
    L.states, L.dYpre,
    L.dXBC, L.dXBC + L.C, L.dXBC + L.C + L.GN, L.dDt, L.dALog, L.dDSkip,
    xStr = L.cDim, bcStr = L.cDim, nBeg = nBeg, nCnt = nCnt)

# ── B3: softplus-dt backward ─────────────────────────────────
proc buildM2DtBwdProgram*(L: M2Lay): VmProgram =
  let H = L.H
  var p: VmProgram
  p.emitToken(L.T)
  p.emitUMulK(3, 0, uint32(L.P)); p.emitUMulK(4, 0, uint32(H))
  p.emitLoadConstF32(4, float32((H + M2Tg - 1) div M2Tg)); p.emitLoadConstF32(5, 0.0'f32)
  sessionVmForGpu(p, 4):
    p.emitUToR(6, 1); p.emitMulConst(7, 5, float32(M2Tg)); p.emitAdd(6, 6, 7)
    p.emitLoadConstF32(7, float32(H)); p.emitCmpLt(7, 6, 7)
    sessionVmIfGpu(p, 7):
      p.emitRToU(5, 6)
      p.emitUAdd(6, 4, 5); p.emitULoadF32(8, 6, uint32(L.dDt))                 # dDt
      p.emitUAdd(6, 3, 5)                                                      # proj/dProj ptr
      p.emitULoadF32(9, 6, uint32(L.proj + L.C + L.cDim))
      p.emitULoadF32(10, 5, uint32(L.dtBias)); p.emitAdd(9, 9, 10)
      p.emitSigmoid(9, 9); p.emitMul(8, 8, 9)                                  # dDtPre
      p.emitUStoreF32(8, 6, uint32(L.dProj + L.C + L.cDim))
      p.emitUAtomicAdd(8, 5, uint32(L.dDtBias))
    p.emitAddConst(5, 5, 1.0'f32)
  p.emitHalt()
  result = p

# ── B4: conv backward (dConvW/dConvB atomic; dX gather -> dProj) ─
proc buildM2ConvBwdProgram*(L: M2Lay): VmProgram =
  let K = L.K; let P = L.P; let cDim = L.cDim; let T = L.T
  let wrapCd = uint32(0xFFFFFFFF'u32 - uint32(cDim) + 1)
  var p: VmProgram
  p.emitToken(L.T)
  p.emitUMulK(3, 0, uint32(P)); p.emitUMulK(4, 0, uint32(cDim))
  p.emitLoadConstF32(4, float32((cDim + M2Tg - 1) div M2Tg)); p.emitLoadConstF32(5, 0.0'f32)
  p.emitLoadConstF32(13, float32(K))
  sessionVmForGpu(p, 4):
    p.emitUToR(6, 1); p.emitMulConst(7, 5, float32(M2Tg)); p.emitAdd(6, 6, 7)
    p.emitLoadConstF32(7, float32(cDim)); p.emitCmpLt(7, 6, 7)
    sessionVmIfGpu(p, 7):
      p.emitRToU(5, 6)                                       # U5 = c
      # ── phần A: grad trọng số từ dPre của CHÍNH (token,c) ──
      p.emitUAdd(6, 4, 5)                                    # U6 = token*cDim + c
      p.emitDPre(L.pre, L.dXBC, 6, 8, 9, 10, 11)             # R8 = dPre
      p.emitUAtomicAdd(8, 5, uint32(L.dConvB))
      p.emitUAdd(7, 3, 5); p.emitUAddK(7, 7, uint32(L.C))
      p.emitUAddK(7, 7, uint32(0xFFFFFFFF'u32 - uint32((K - 1) * P) + 1))   # x ptr k=0
      p.emitUMulK(8, 5, uint32(K))                           # U8 = dConvW index c*K
      p.emitAddConst(11, 2, 0.0'f32); p.emitLoadConstF32(12, float32(K - 1))
      sessionVmForGpu(p, 13):
        p.emitCmpLt(14, 11, 12)
        p.emitLoadConstF32(15, 1.0'f32); p.emitSub(15, 15, 14)
        sessionVmIfGpu(p, 15):
          p.emitULoadF32(9, 7, uint32(L.proj))
          p.emitMul(9, 9, 8)                                 # x * dPre (R8 = dPre, float)
          p.emitUAtomicAdd(9, 8, uint32(L.dConvW))
        p.emitUAddK(7, 7, uint32(P)); p.emitUAddK(8, 8, 1)
        p.emitAddConst(11, 11, 1.0'f32)
      # ── phần B: dX[token,c] = sum_k dPre(token+K-1-k, c) * w[c,k] (gather, không atomic) ──
      p.emitUAdd(6, 4, 5); p.emitUAddK(6, 6, uint32((K - 1) * cDim))
      p.emitUMulK(7, 5, uint32(K))
      p.emitAddConst(11, 2, float32(K - 1))                  # R11 = t + (K-1) - k
      p.emitLoadConstF32(12, float32(T)); p.emitLoadConstF32(9, 0.0'f32)
      sessionVmForGpu(p, 13):
        p.emitCmpLt(14, 11, 12)                              # tt < T
        sessionVmIfGpu(p, 14):
          p.emitDPre(L.pre, L.dXBC, 6, 8, 10, 15, 14)
          p.emitULoadF32(10, 7, uint32(L.convW))
          p.emitFma(9, 8, 10, 9)
        p.emitUAddK(6, 6, wrapCd); p.emitUAddK(7, 7, 1)
        p.emitAddConst(11, 11, -1.0'f32)
      p.emitUAdd(6, 3, 5); p.emitUStoreF32(9, 6, uint32(L.dProj + L.C))
    p.emitAddConst(5, 5, 1.0'f32)
  p.emitHalt()
  result = p

# ── tiện ích: dst[i] += scale * src[i]  (thay vòng `for i: x[i] += delta[i]*s` của LoRA) ──
proc buildAxpyProgram*(offDst, offSrc: int; scale: float32): VmProgram =
  ## grid = n thread, R0 = i. mem: dst ở offDst, src ở offSrc.
  var p: VmProgram
  p.emitAddConst(1, 0, float32(offDst)); p.emitLoadF32Idx(2, 1)
  p.emitAddConst(3, 0, float32(offSrc)); p.emitLoadF32Idx(4, 3)
  p.emitMulConst(4, 4, scale); p.emitAdd(2, 2, 4)
  p.emitStoreF32Idx(2, 1)
  p.emitHalt()
  result = p

proc m2FwdTgBytes*(L: M2Lay): int = rmsTg(L) * 4

# ═════════════════════════════════════════════════════════════
# KERNEL TRAIN: RMSNorm, Linear++ (scan tuyến tính), softmax-CE, Adam gộp
# ═════════════════════════════════════════════════════════════
const
  LpEps* = 1e-6'f32       ## PHẢI khớp LP_EPS trong test_nimformer.nim
  LpPhiTgMax* = 256

# ═════════════════════════════════════════════════════════════
# (1) RMSNorm:  y = x * r * gamma,  r = rsqrt(mean(x²) + eps)
# ═════════════════════════════════════════════════════════════
type
  RmsLay* = object
    rows*, C*: int
    eps*: float32
    x*, gamma*, y*, dy*, dx*, dGamma*: int
    total*: int

proc rmsLayout*(rows, C: int; eps: float32): RmsLay =
  var L: RmsLay
  L.rows = rows; L.C = C; L.eps = eps
  var o = 0
  template take(f: untyped; n: int) =
    f = o; o += n
  take(L.x, rows * C); take(L.gamma, C); take(L.y, rows * C)
  take(L.dy, rows * C); take(L.dx, rows * C); take(L.dGamma, C)
  L.total = o
  result = L

proc rmsRowTg*(C: int): int = min(C and (-C), gTgCap)   ## luỹ thừa 2 lớn nhất chia hết C, <= giới hạn GPU

proc buildRmsFwdProgram*(L: RmsLay): VmProgram =
  ## grid = rows * tg, tgSize = tg. Thread lid xử lý các cột lid, lid+tg, ...
  let C = L.C; let tg = rmsRowTg(C); let ept = C div tg
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitUToR(14, 1)                                # R14 = lid
  p.emitUMulK(2, 0, uint32(C))                     # U2 = row*C
  p.emitLoadConstF32(1, 0.0'f32)                   # sum x²
  p.emitLoadConstF32(3, float32(ept))
  p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(5, 4, float32(tg)); p.emitAdd(5, 5, 14); p.emitRToU(4, 5)   # U4 = col
    p.emitUAdd(5, 2, 4); p.emitULoadF32(6, 5, uint32(L.x))
    p.emitMul(7, 6, 6); p.emitAdd(1, 1, 7)
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitTgStoreF32(1, 14, 0); p.emitBarrierTg()
  p.emitTgTreeReduce(14, tg, 6, 7, 8, 9, 10)
  p.emitLoadConstF32(11, 0.0'f32); p.emitTgLoadF32(13, 11, 0)
  p.emitMulConst(13, 13, 1.0'f32 / float32(C)); p.emitAddConst(13, 13, L.eps)
  p.emitRsqrt(13, 13)                              # r
  p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(5, 4, float32(tg)); p.emitAdd(5, 5, 14); p.emitRToU(4, 5)
    p.emitUAdd(5, 2, 4); p.emitULoadF32(6, 5, uint32(L.x))
    p.emitMul(6, 6, 13)
    p.emitULoadF32(8, 4, uint32(L.gamma)); p.emitMul(6, 6, 8)
    p.emitUStoreF32(6, 5, uint32(L.y))
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitHalt()
  result = p

proc buildRmsBwdProgram*(L: RmsLay): VmProgram =
  ## dX_j = r*g_j*dy_j - (r³/C) * x_j * Σ_c(dy_c*g_c*x_c);  dGamma_c += dy_c*x_c*r
  ## dGamma cộng dồn bằng atomic -> vùng dGamma phải = 0 trước khi chạy.
  let C = L.C; let tg = rmsRowTg(C); let ept = C div tg
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitUToR(14, 1)
  p.emitUMulK(2, 0, uint32(C))
  p.emitLoadConstF32(1, 0.0'f32); p.emitLoadConstF32(2, 0.0'f32)
  p.emitLoadConstF32(3, float32(ept)); p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(5, 4, float32(tg)); p.emitAdd(5, 5, 14); p.emitRToU(4, 5)
    p.emitUAdd(5, 2, 4); p.emitULoadF32(6, 5, uint32(L.x))                 # x
    p.emitMul(7, 6, 6); p.emitAdd(1, 1, 7)                                 # Σx²
    p.emitULoadF32(8, 5, uint32(L.dy)); p.emitULoadF32(9, 4, uint32(L.gamma))
    p.emitMul(8, 8, 9); p.emitMul(8, 8, 6); p.emitAdd(2, 2, 8)             # dot
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitLoadConstF32(11, 0.0'f32)
  p.emitTgStoreF32(1, 14, 0); p.emitBarrierTg()
  p.emitTgTreeReduce(14, tg, 6, 7, 8, 9, 10)
  p.emitTgLoadF32(13, 11, 0); p.emitBarrierTg()                            # Σx²
  p.emitTgStoreF32(2, 14, 0); p.emitBarrierTg()
  p.emitTgTreeReduce(14, tg, 6, 7, 8, 9, 10)
  p.emitTgLoadF32(15, 11, 0)                                               # dot
  p.emitMulConst(13, 13, 1.0'f32 / float32(C)); p.emitAddConst(13, 13, L.eps); p.emitRsqrt(13, 13)  # r
  p.emitMul(10, 13, 13); p.emitMul(10, 10, 13)
  p.emitMulConst(10, 10, 1.0'f32 / float32(C)); p.emitMul(15, 10, 15)      # r³/C*dot
  p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(5, 4, float32(tg)); p.emitAdd(5, 5, 14); p.emitRToU(4, 5)
    p.emitUAdd(5, 2, 4)
    p.emitULoadF32(6, 5, uint32(L.x)); p.emitULoadF32(7, 5, uint32(L.dy))
    p.emitULoadF32(8, 4, uint32(L.gamma))
    p.emitMul(9, 7, 8); p.emitMul(9, 9, 13)                                # r*g*dy
    p.emitMul(10, 6, 15); p.emitSub(9, 9, 10)                              # - x*r³/C*dot
    p.emitUStoreF32(9, 5, uint32(L.dx))
    p.emitMul(9, 7, 6); p.emitMul(9, 9, 13)                                # dy*x*r
    p.emitUAtomicAdd(9, 4, uint32(L.dGamma))
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitHalt()
  result = p

# ═════════════════════════════════════════════════════════════
# (2) Linear++ attention TUYẾN TÍNH theo T (scan), O(T·D²) / head — KHÔNG có ma trận T×T.
#   phi = ELU+1;  S_t = Σ_{j≤t} phi(k_j) ⊗ v_j  (D×D);  z_t = Σ_{j≤t} phi(k_j)
#   y_t = phi(q_t)·S_t / (phi(q_t)·z_t + eps)
#   Backward (g_t = dY_t*invDen_t, dDen_t = -(dY_t·y_t)*invDen_t):
#     dqf_t[d] = Σ_e g_t[e] S_t[d,e] + dDen_t z_t[d]                (scan xuôi)
#     dS_t = Σ_{i≥t} qf_i ⊗ g_i ;  dz_t = Σ_{i≥t} dDen_i qf_i       (scan ngược)
#     dkf_t[d] = Σ_e dS_t[d,e] v_t[e] + dz_t[d] ; dv_t[e] = Σ_d dS_t[d,e] kf_t[d]
# Trạng thái scan nằm trong buffer (mỗi thread một vùng riêng) -> không giới hạn thanh ghi.
# ═════════════════════════════════════════════════════════════
type
  LpLay* = object
    B*, T*, H*, D*, C*, BT*: int
    qkv*, phi*, y*, invDen*, dY*, dDen*, dQKV*: int
    zF*, cF*, rQ*, zQ*, rK*, zK*, cV*: int      ## vùng trạng thái scan (khởi tạo 0)
    total*: int

proc lpLayout*(B, T, H, D: int): LpLay =
  var L: LpLay
  L.B = B; L.T = T; L.H = H; L.D = D; L.C = H * D; L.BT = B * T
  var o = 0
  template take(f: untyped; n: int) =
    f = o; o += n
  let BC = B * L.C
  take(L.qkv, L.BT * 3 * L.C)      # [BT, 3C] = q | k | v  (đầu ra qkvProj)
  take(L.phi, L.BT * 2 * L.C)      # [BT, 2C] = phi(q) | phi(k)
  take(L.y, L.BT * L.C)            # [BT, C]
  take(L.invDen, L.BT * H)         # [BT, H]
  take(L.dY, L.BT * L.C)           # [BT, C]
  take(L.dDen, L.BT * H)           # [BT, H]
  take(L.dQKV, L.BT * 3 * L.C)     # [BT, 3C] gradient ra (dq | dk | dv) — GHI ĐÈ, không cần 0
  take(L.zF, BC); take(L.cF, BC * D)
  take(L.rQ, BC * D); take(L.zQ, BC)
  take(L.rK, BC * D); take(L.zK, BC)
  take(L.cV, BC * D)
  L.total = o
  result = L

proc lpPhiTg*(L: LpLay): int = min((2 * L.C) and (-(2 * L.C)), min(LpPhiTgMax, gTgCap))

const
  LpScanMaxTg* = 64
  LpScanStateChunkMax* = 32

proc lpScanTgSize*(T: int): int =
  ## One lane per timestep for additive inclusive scans. A zero return means
  ## use the serial reference path; padding lanes are the additive identity.
  if T <= 0 or T > LpScanMaxTg: return 0
  var tg = 1
  while tg < T and tg <= gTgCap div 2: tg *= 2
  if tg < T or tg > gTgCap: return 0
  result = tg

proc lpScanStateChunkSize*(D: int): int =
  ## Bound dynamic threadgroup memory for the D×D Linear++ state scan.
  result = min(max(D, 1), LpScanStateChunkMax)

proc emitElu1(p: var VmProgram; rd, rs, rz, rc: uint32) =
  ## rd = elu(rs)+1 = rs>0 ? rs+1 : exp(rs).  rd,rs,rz,rc phải khác nhau.
  p.emitExp(rd, rs)
  p.emitLoadConstF32(rz, 0.0'f32)
  p.emitCmpLt(rc, rz, rs)
  sessionVmIfGpu(p, rc):
    p.emitAddConst(rd, rs, 1.0'f32)

proc emitEluGrad(p: var VmProgram; rd, rs, rz, rc: uint32) =
  ## rd = d/dx[elu(x)+1] = rs>0 ? 1 : exp(rs)
  p.emitExp(rd, rs)
  p.emitLoadConstF32(rz, 0.0'f32)
  p.emitCmpLt(rc, rz, rs)
  sessionVmIfGpu(p, rc):
    p.emitLoadConstF32(rd, 1.0'f32)

proc buildLpPhiProgram*(L: LpLay): VmProgram =
  ## grid = BT * 2C phần tử; tgSize = lpPhiTg. phi[bt*2C + c'] = elu1(qkv[bt*3C + c'])
  let tg = lpPhiTg(L); let c2 = 2 * L.C; let nChunks = c2 div tg
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitUToR(1, 0)
  p.emitAddConst(2, 1, 0.5'f32); p.emitLoadConstF32(3, float32(nChunks)); p.emitDiv(2, 2, 3)
  p.emitRToU(3, 2)                                  # U3 = bt
  p.emitUToR(4, 3); p.emitMulConst(4, 4, float32(nChunks)); p.emitSub(4, 1, 4)
  p.emitRToU(4, 4)                                  # U4 = chunk
  p.emitUMulK(4, 4, uint32(tg)); p.emitUAdd(5, 4, 1)   # U5 = c'
  p.emitUMulK(6, 3, uint32(3 * L.C)); p.emitUAdd(6, 6, 5)
  p.emitUMulK(7, 3, uint32(c2)); p.emitUAdd(7, 7, 5)
  p.emitULoadF32(5, 6, uint32(L.qkv))
  p.emitElu1(6, 5, 7, 8)
  p.emitUStoreF32(6, 7, uint32(L.phi))
  p.emitHalt()
  result = p

proc emitDecodeBH(p: var VmProgram; L: LpLay) =
  ## R0 = tid = b*H + h.  Ra: U0 = bh, U2 = b, U3 = h.
  p.emitAddConst(1, 0, 0.5'f32); p.emitLoadConstF32(2, float32(L.H)); p.emitDiv(1, 1, 2)
  p.emitRToU(2, 1)                                  # U2 = b
  p.emitUToR(3, 2); p.emitMulConst(3, 3, float32(L.H)); p.emitSub(3, 0, 3)
  p.emitRToU(3, 3)                                  # U3 = h
  p.emitRToU(0, 0)                                  # U0 = bh

proc emitDecodeBHX(p: var VmProgram; L: LpLay) =
  ## R0 = tid = (b*H + h)*D + x.  Ra: U0 = tid, U1 = x, U2 = b, U3 = h.
  let D = L.D
  p.emitAddConst(1, 0, 0.5'f32); p.emitLoadConstF32(2, float32(D)); p.emitDiv(1, 1, 2)
  p.emitRToU(4, 1)                                  # U4 = bh
  p.emitUToR(5, 4)                                  # R5 = bh (float, nguyên)
  p.emitMulConst(3, 5, float32(D)); p.emitSub(3, 0, 3)
  p.emitRToU(1, 3)                                  # U1 = x
  p.emitAddConst(4, 5, 0.5'f32); p.emitLoadConstF32(2, float32(L.H)); p.emitDiv(4, 4, 2)
  p.emitRToU(2, 4)                                  # U2 = b
  p.emitUToR(6, 2); p.emitMulConst(6, 6, float32(L.H)); p.emitSub(6, 5, 6)
  p.emitRToU(3, 6)                                  # U3 = h
  p.emitRToU(0, 0)                                  # U0 = tid

proc emitLpDecode(p: var VmProgram; L: LpLay) =
  ## R0 = tid = bt*H + h.  Ra: U0 = bt, U1 = h.
  p.emitAddConst(1, 0, 0.5'f32); p.emitLoadConstF32(2, float32(L.H)); p.emitDiv(1, 1, 2)
  p.emitRToU(0, 1)
  p.emitUToR(2, 0); p.emitMulConst(3, 2, float32(L.H)); p.emitSub(3, 0, 3)
  p.emitRToU(1, 3)

proc negStride(s: int): uint32 = uint32(0) - uint32(s)

proc buildLpDenProgram*(L: LpLay): VmProgram =
  ## 1 thread / (b,h). z += phi(k_t); invDen_t = 1/(phi(q_t)·z + eps). Quét xuôi theo t.
  let C = L.C; let D = L.D
  var p: VmProgram
  p.emitDecodeBH(L)
  p.emitLoadConstF32(7, float32(L.T)); p.emitLoadConstF32(10, float32(D))
  p.emitUMulK(9, 3, uint32(D))                                  # h*D
  p.emitUMulK(4, 2, uint32(L.T * 2 * C)); p.emitUAdd(4, 4, 9)   # U4 = phi row t=0
  p.emitUMulK(5, 0, uint32(D))                                  # U5 = z state base
  p.emitUMulK(6, 2, uint32(L.T * L.H)); p.emitUAdd(6, 6, 3)     # U6 = invDen ptr
  sessionVmForGpu(p, 7):
    p.emitUAddK(10, 4, 0); p.emitUAddK(11, 5, 0)
    sessionVmForGpu(p, 10):
      p.emitULoadF32(11, 10, uint32(L.phi + C)); p.emitULoadF32(12, 11, uint32(L.zF))
      p.emitAdd(12, 12, 11); p.emitUStoreF32(12, 11, uint32(L.zF))
      p.emitUAddK(10, 10, 1); p.emitUAddK(11, 11, 1)
    p.emitLoadConstF32(9, 0.0'f32)
    p.emitUAddK(10, 4, 0); p.emitUAddK(11, 5, 0)
    sessionVmForGpu(p, 10):
      p.emitULoadF32(11, 10, uint32(L.phi)); p.emitULoadF32(12, 11, uint32(L.zF))
      p.emitFma(9, 11, 12, 9); p.emitUAddK(10, 10, 1); p.emitUAddK(11, 11, 1)
    p.emitAddConst(9, 9, LpEps)
    p.emitLoadConstF32(13, 1.0'f32); p.emitDiv(9, 13, 9)
    p.emitUStoreF32(9, 6, uint32(L.invDen))
    p.emitUAddK(4, 4, uint32(2 * C)); p.emitUAddK(6, 6, uint32(L.H))
  p.emitHalt()
  result = p

proc buildLpFwdProgram*(L: LpLay): VmProgram =
  ## 1 thread / (b,h,e): giữ cột S[:,e] (D phần tử) trong buffer; y_t[e] = invDen_t * Σ_d qf_t[d] S_t[d,e].
  let C = L.C; let D = L.D
  var p: VmProgram
  p.emitDecodeBHX(L)
  p.emitLoadConstF32(7, float32(L.T)); p.emitLoadConstF32(10, float32(D))
  p.emitUMulK(9, 3, uint32(D))                                  # h*D
  p.emitUMulK(4, 2, uint32(L.T * 2 * C)); p.emitUAdd(4, 4, 9)   # U4 = phi row t=0
  p.emitUMulK(5, 2, uint32(L.T * 3 * C)); p.emitUAdd(5, 5, 9); p.emitUAdd(5, 5, 1)   # U5 = v elem
  p.emitUMulK(6, 2, uint32(L.T * C)); p.emitUAdd(6, 6, 9); p.emitUAdd(6, 6, 1)       # U6 = y elem
  p.emitUMulK(7, 2, uint32(L.T * L.H)); p.emitUAdd(7, 7, 3)     # U7 = invDen ptr
  p.emitUMulK(8, 0, uint32(D))                                  # U8 = S col base (tid*D)
  sessionVmForGpu(p, 7):
    p.emitULoadF32(9, 5, uint32(L.qkv + 2 * C))                 # v[e]
    p.emitLoadConstF32(8, 0.0'f32)                              # num
    p.emitUAddK(10, 4, 0); p.emitUAddK(11, 8, 0)
    sessionVmForGpu(p, 10):
      p.emitULoadF32(11, 10, uint32(L.phi + C)); p.emitULoadF32(12, 11, uint32(L.cF))
      p.emitFma(12, 11, 9, 12); p.emitUStoreF32(12, 11, uint32(L.cF))
      p.emitULoadF32(11, 10, uint32(L.phi)); p.emitFma(8, 11, 12, 8)
      p.emitUAddK(10, 10, 1); p.emitUAddK(11, 11, 1)
    p.emitULoadF32(13, 7, uint32(L.invDen)); p.emitMul(8, 8, 13)
    p.emitUStoreF32(8, 6, uint32(L.y))
    p.emitUAddK(4, 4, uint32(2 * C)); p.emitUAddK(5, 5, uint32(3 * C))
    p.emitUAddK(6, 6, uint32(C)); p.emitUAddK(7, 7, uint32(L.H))
  p.emitHalt()
  result = p

proc buildLpDenScanParallelProgram*(L: LpLay; tg: int): VmProgram =
  ## Parallel additive prefix for z_t[d] = sum_{j<=t} phi(k_j[d]). One
  ## threadgroup handles one (batch, head, d), lanes are timesteps. It also
  ## writes phi(q/k), atomically reduces q_t·z_t into invDen, and saves z_T.
  doAssert tg >= L.T and (tg and (tg - 1)) == 0 and tg <= gTgCap,
    "Linear++ z scan needs a power-of-two TG >= sequence length"
  let D = L.D
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)

  # Decode group=(b*H+h)*D+d into U3=bh, U5=d, U8=b, U10=h.
  p.emitUToR(0, 0); p.emitLoadConstF32(1, float32(D)); p.emitDiv(2, 0, 1)
  p.emitRToU(3, 2)
  p.emitUToR(4, 3); p.emitMulConst(4, 4, float32(D)); p.emitSub(5, 0, 4)
  p.emitRToU(5, 5)
  p.emitUToR(6, 3); p.emitLoadConstF32(7, float32(L.H)); p.emitDiv(6, 6, 7)
  p.emitRToU(8, 6)
  p.emitUToR(9, 8); p.emitMulConst(9, 9, float32(L.H))
  p.emitUToR(10, 3); p.emitSub(10, 10, 9)
  p.emitRToU(10, 10)

  # Per-lane addresses: qkv/phi row, scalar denominator, terminal z state.
  p.emitUMulK(11, 8, uint32(L.T)); p.emitUAdd(11, 11, 1) # bt
  p.emitUMulK(12, 11, uint32(3 * L.C))
  p.emitUMulK(13, 10, uint32(D)); p.emitUAdd(12, 12, 13); p.emitUAdd(12, 12, 5)
  p.emitUMulK(14, 11, uint32(2 * L.C)); p.emitUAdd(14, 14, 13); p.emitUAdd(14, 14, 5)
  p.emitUMulK(15, 11, uint32(L.H)); p.emitUAdd(15, 15, 10) # invDen index
  p.emitUMulK(0, 3, uint32(D)); p.emitUAdd(0, 0, 5)         # zF index

  # Compute and store phi(q), phi(k); padded lanes contribute zero.
  p.emitLoadConstF32(5, 0'f32); p.emitLoadConstF32(6, 0'f32)
  p.emitUToR(2, 1); p.emitLoadConstF32(3, float32(L.T)); p.emitCmpLt(4, 2, 3)
  sessionVmIfGpu(p, 4):
    p.emitULoadF32(2, 12, uint32(L.qkv))
    p.emitElu1(5, 2, 6, 7)
    p.emitUStoreF32(5, 14, uint32(L.phi))
    p.emitUAddK(9, 12, uint32(L.C)); p.emitULoadF32(2, 9, uint32(L.qkv))
    p.emitElu1(6, 2, 7, 8)
    p.emitUStoreF32(6, 14, uint32(L.phi + L.C))
  p.emitUToR(15, 1) # R15 = immutable lane index for threadgroup-memory addressing
  p.emitTgStoreF32(6, 15, 0)
  p.emitBarrierTg()

  # Hillis-Steele inclusive sum. Every lane reaches every barrier.
  var srcBank = 0
  var dstBank = tg
  var stride = 1
  while stride < tg:
    p.emitUToR(0, 1); p.emitLoadConstF32(1, float32(stride))
    p.emitAddConst(2, 0, -float32(stride)); p.emitRToU(2, 2) # prev lane (unused below stride)
    p.emitTgLoadF32(5, 15, uint32(srcBank))
    p.emitAddConst(0, 0, 1.0'f32) # lane >= stride iff lane+1 > stride
    p.emitLoadConstF32(6, float32(stride)); p.emitCmpLt(7, 6, 0)
    sessionVmIfGpu(p, 7):
      p.emitTgLoadF32(8, 2, uint32(srcBank)); p.emitAdd(5, 5, 8)
    p.emitTgStoreF32(5, 15, uint32(dstBank)); p.emitBarrierTg()
    swap(srcBank, dstBank)
    stride *= 2

  p.emitTgLoadF32(5, 15, uint32(srcBank))
  p.emitUToR(2, 1); p.emitLoadConstF32(3, float32(L.T)); p.emitCmpLt(4, 2, 3)
  sessionVmIfGpu(p, 4):
    p.emitULoadF32(2, 14, uint32(L.phi))
    p.emitMul(6, 2, 5)
    # Keep the unreduced denominator in dDen until all timestep groups finish.
    # The finalize dispatch converts it to invDen, which backward consumes.
    p.emitUAtomicAdd(6, 15, uint32(L.dDen))
    # Among valid lanes, lane+1 > T-1 is exactly lane == T-1.
    p.emitUToR(2, 1); p.emitAddConst(2, 2, 1.0'f32)
    p.emitLoadConstF32(3, float32(L.T - 1)); p.emitCmpLt(4, 3, 2)
    sessionVmIfGpu(p, 4): p.emitUStoreF32(5, 0, uint32(L.zF))
  p.emitHalt()
  result = p

proc buildLpStateScanParallelProgram*(L: LpLay; tg, chunkSize: int): VmProgram =
  ## Parallel additive prefix for each Linear++ state slice S_t[d,e]. One TG
  ## handles (b,h,e,dChunk); each timestep lane scans a bounded D chunk in
  ## ping-pong threadgroup memory, then atomically contributes its q_t[d]*S_t[d,e]
  ## partial sum to y. The final valid lane exports S_T into cF.
  doAssert tg >= L.T and (tg and (tg - 1)) == 0 and tg <= gTgCap
  doAssert chunkSize > 0 and chunkSize <= L.D
  let nChunks = (L.D + chunkSize - 1) div chunkSize
  let tgChunk = tg * chunkSize
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(8)

  # Flattened group=(bh*D+e)*nChunks+chunk; recover base and chunk, then
  # decode base as (bh,e). U6=chunk, U8=lane, U9=lane*chunkSize.
  p.emitUToR(1, 0); p.emitLoadConstF32(2, float32(nChunks)); p.emitDiv(3, 1, 2)
  p.emitRToU(4, 3)
  p.emitUToR(5, 4); p.emitMulConst(5, 5, float32(nChunks)); p.emitSub(6, 1, 5)
  p.emitRToU(6, 6)
  p.emitUToR(0, 4); emitDecodeBHX(p, L) # U0=bh, U1=e, U2=b, U3=h
  p.emitUMulK(7, 6, uint32(chunkSize))
  p.emitUMulK(9, 8, uint32(chunkSize))
  p.emitUToR(9, 9) # R9 = immutable lane base for threadgroup-memory addressing
  p.emitUMulK(10, 2, uint32(L.T)); p.emitUAdd(10, 10, 8) # bt
  p.emitUMulK(11, 3, uint32(L.D))                       # h*D
  p.emitUMulK(12, 10, uint32(2 * L.C)); p.emitUAdd(12, 12, 11); p.emitUAdd(12, 12, 7) # phi q base
  p.emitUMulK(13, 10, uint32(3 * L.C)); p.emitUAddK(13, 13, uint32(2 * L.C))
  p.emitUAdd(13, 13, 11); p.emitUAdd(13, 13, 1) # v base
  p.emitUMulK(14, 4, uint32(L.D * L.D)); p.emitUMulK(5, 1, uint32(L.D))
  p.emitUAdd(14, 14, 5); p.emitUAdd(14, 14, 7) # cF base, [bh,e,d]
  p.emitUMulK(15, 10, uint32(L.C)); p.emitUAdd(15, 15, 11); p.emitUAdd(15, 15, 1) # y base

  # Initial contributions k_phi[t,d] * v[t,e] in bank 0. Invalid/padded
  # lanes and dimensions write the identity 0 but still reach the barrier.
  for j in 0 ..< chunkSize:
    p.emitUAddK(5, 7, uint32(j))
    p.emitLoadConstF32(5, 0'f32)
    p.emitUToR(0, 8); p.emitLoadConstF32(1, float32(L.T)); p.emitCmpLt(2, 0, 1)
    sessionVmIfGpu(p, 2):
      p.emitUToR(0, 7); p.emitAddConst(0, 0, float32(j))
      p.emitLoadConstF32(1, float32(L.D)); p.emitCmpLt(2, 0, 1)
      sessionVmIfGpu(p, 2):
        p.emitULoadF32(3, 12, uint32(L.phi + L.C + j)) # phi(k[t,d])
        p.emitULoadF32(4, 13, uint32(L.qkv))            # v[t,e]
        p.emitMul(5, 3, 4)
    p.emitTgStoreF32(5, 9, uint32(j))
  p.emitBarrierTg()

  # Inclusive scan over time; per-lane D chunk elements are independent.
  var srcBank = 0
  var dstBank = tgChunk
  var stride = 1
  while stride < tg:
    p.emitUToR(0, 8); p.emitAddConst(0, 0, 1.0'f32)
    p.emitLoadConstF32(1, float32(stride)); p.emitCmpLt(2, 1, 0)
    p.emitUToR(3, 8); p.emitAddConst(3, 3, -float32(stride)); p.emitRToU(5, 3)
    p.emitUMulK(5, 5, uint32(chunkSize)); p.emitUToR(5, 5)
    # previous lane's chunk base; convert back to R for TG load addressing.
    for j in 0 ..< chunkSize:
      p.emitTgLoadF32(6, 9, uint32(srcBank + j))
      sessionVmIfGpu(p, 2):
        p.emitTgLoadF32(7, 5, uint32(srcBank + j)); p.emitAdd(6, 6, 7)
      p.emitTgStoreF32(6, 9, uint32(dstBank + j))
    p.emitBarrierTg()
    swap(srcBank, dstBank)
    stride *= 2

  # Reduce d locally per (t,e,chunk), atomically sum chunks into y, and export
  # terminal state from t=T-1. No barrier is inside a lane-varying branch.
  p.emitLoadConstF32(7, 0'f32) # output partial
  for j in 0 ..< chunkSize:
    p.emitUAddK(5, 7, uint32(j)) # scratch d index, recomputed from chunk base below
    p.emitUToR(0, 8); p.emitLoadConstF32(1, float32(L.T)); p.emitCmpLt(2, 0, 1)
    sessionVmIfGpu(p, 2):
      p.emitUToR(0, 7); p.emitAddConst(0, 0, float32(j))
      p.emitLoadConstF32(1, float32(L.D)); p.emitCmpLt(2, 0, 1)
      sessionVmIfGpu(p, 2):
        p.emitTgLoadF32(6, 9, uint32(srcBank + j))
        p.emitULoadF32(8, 12, uint32(L.phi + j))
        p.emitFma(7, 8, 6, 7)
        # Among valid lanes, lane+1 > T-1 is exactly the final real timestep.
        p.emitUToR(0, 8); p.emitAddConst(0, 0, 1.0'f32)
        p.emitLoadConstF32(1, float32(L.T - 1)); p.emitCmpLt(2, 1, 0)
        sessionVmIfGpu(p, 2):
          p.emitUAddK(5, 7, uint32(j))
          p.emitTgLoadF32(6, 9, uint32(srcBank + j))
          p.emitUStoreF32(6, 14, uint32(L.cF + j))
  p.emitUToR(0, 8); p.emitLoadConstF32(1, float32(L.T)); p.emitCmpLt(2, 0, 1)
  sessionVmIfGpu(p, 2): p.emitUAtomicAdd(7, 15, uint32(L.y))
  p.emitHalt()
  result = p

proc buildLpFinalizeProgram*(L: LpLay): VmProgram =
  ## y[t,h,e] = numerator / (denominator + eps), and persist invDen in the
  ## reciprocal form expected by the existing backward kernels. The raw
  ## denominator lives in dDen so the one writer to invDen cannot race readers.
  var p: VmProgram
  p.emitRToU(0, 0) # global element index in [BT,C]
  p.emitUToR(1, 0); p.emitLoadConstF32(2, float32(L.C)); p.emitDiv(3, 1, 2)
  p.emitRToU(1, 3) # bt
  p.emitUToR(2, 0); p.emitUToR(3, 1); p.emitMulConst(3, 3, float32(L.C))
  p.emitSub(2, 2, 3); p.emitLoadConstF32(3, float32(L.D)); p.emitDiv(2, 2, 3)
  p.emitRToU(2, 2) # h
  p.emitUMulK(3, 1, uint32(L.H)); p.emitUAdd(3, 3, 2)
  p.emitULoadF32(4, 3, uint32(L.dDen)); p.emitAddConst(4, 4, LpEps)
  p.emitLoadConstF32(5, 1'f32); p.emitDiv(5, 5, 4)
  # Exactly one channel (e=0) writes each reciprocal denominator.
  p.emitUToR(6, 0); p.emitUToR(7, 1); p.emitMulConst(7, 7, float32(L.C))
  p.emitUToR(8, 2); p.emitMulConst(8, 8, float32(L.D)); p.emitAdd(7, 7, 8)
  p.emitSub(6, 6, 7); p.emitLoadConstF32(7, 1'f32); p.emitCmpLt(8, 6, 7)
  sessionVmIfGpu(p, 8): p.emitUStoreF32(5, 3, uint32(L.invDen))
  p.emitULoadF32(6, 0, uint32(L.y)); p.emitMul(6, 6, 5)
  p.emitUStoreF32(6, 0, uint32(L.y))
  p.emitHalt()
  result = p

proc buildLpDDenProgram*(L: LpLay): VmProgram =
  ## 1 thread / (b,t,h): dDen = -(dY·y) * invDen.
  let C = L.C; let D = L.D
  var p: VmProgram
  p.emitLpDecode(L)
  p.emitLoadConstF32(10, float32(D))
  p.emitUMulK(9, 1, uint32(D))
  p.emitUMulK(6, 0, uint32(C)); p.emitUAdd(6, 6, 9)
  p.emitUMulK(7, 0, uint32(L.H)); p.emitUAdd(7, 7, 1)
  p.emitLoadConstF32(9, 0.0'f32); p.emitUAddK(10, 6, 0)
  sessionVmForGpu(p, 10):
    p.emitULoadF32(11, 10, uint32(L.dY)); p.emitULoadF32(12, 10, uint32(L.y))
    p.emitFma(9, 11, 12, 9); p.emitUAddK(10, 10, 1)
  p.emitULoadF32(8, 7, uint32(L.invDen))
  p.emitNeg(9, 9); p.emitMul(9, 9, 8)
  p.emitUStoreF32(9, 7, uint32(L.dDen))
  p.emitHalt()
  result = p

proc buildLpBwdQProgram*(L: LpLay): VmProgram =
  ## 1 thread / (b,h,d), quét XUÔI: giữ hàng S[d,:] và z[d]; ghi dQ_t[d] (đã nhân elu').
  let C = L.C; let D = L.D
  var p: VmProgram
  p.emitDecodeBHX(L)
  p.emitLoadConstF32(7, float32(L.T)); p.emitLoadConstF32(10, float32(D))
  p.emitUMulK(13, 3, uint32(D))                                 # U13 = h*D
  p.emitUMulK(4, 2, uint32(L.T * 2 * C)); p.emitUAdd(4, 4, 13); p.emitUAdd(4, 4, 1)  # phi elem d
  p.emitUMulK(5, 2, uint32(L.T * 3 * C)); p.emitUAdd(5, 5, 13)                        # v row
  p.emitUMulK(6, 2, uint32(L.T * C)); p.emitUAdd(6, 6, 13)                            # dY row
  p.emitUMulK(7, 2, uint32(L.T * L.H)); p.emitUAdd(7, 7, 3)                           # scalar ptr
  p.emitUMulK(8, 0, uint32(D))                                  # row state base
  p.emitUAddK(9, 0, 0)                                          # z ptr
  p.emitUMulK(12, 2, uint32(L.T * 3 * C)); p.emitUAdd(12, 12, 13); p.emitUAdd(12, 12, 1)  # q/dq elem
  sessionVmForGpu(p, 7):
    p.emitULoadF32(8, 4, uint32(L.phi + C))                     # kf[d]
    p.emitULoadF32(9, 7, uint32(L.invDen))
    p.emitLoadConstF32(13, 0.0'f32)
    p.emitUAddK(10, 5, 0); p.emitUAddK(11, 6, 0); p.emitUAddK(14, 8, 0)
    sessionVmForGpu(p, 10):
      p.emitULoadF32(11, 10, uint32(L.qkv + 2 * C)); p.emitULoadF32(12, 14, uint32(L.rQ))
      p.emitFma(12, 8, 11, 12); p.emitUStoreF32(12, 14, uint32(L.rQ))
      p.emitULoadF32(11, 11, uint32(L.dY)); p.emitFma(13, 11, 12, 13)
      p.emitUAddK(10, 10, 1); p.emitUAddK(11, 11, 1); p.emitUAddK(14, 14, 1)
    p.emitULoadF32(12, 9, uint32(L.zQ)); p.emitAdd(12, 12, 8); p.emitUStoreF32(12, 9, uint32(L.zQ))
    p.emitMul(13, 13, 9)
    p.emitULoadF32(14, 7, uint32(L.dDen)); p.emitFma(13, 14, 12, 13)
    p.emitULoadF32(11, 12, uint32(L.qkv))
    p.emitEluGrad(12, 11, 14, 15)
    p.emitMul(13, 13, 12)
    p.emitUStoreF32(13, 12, uint32(L.dQKV))
    p.emitUAddK(4, 4, uint32(2 * C)); p.emitUAddK(5, 5, uint32(3 * C)); p.emitUAddK(6, 6, uint32(C))
    p.emitUAddK(7, 7, uint32(L.H)); p.emitUAddK(12, 12, uint32(3 * C))
  p.emitHalt()
  result = p

proc buildLpBwdKProgram*(L: LpLay): VmProgram =
  ## 1 thread / (b,h,d), quét NGƯỢC: giữ hàng dS[d,:] và dz[d]; ghi dK_t[d] (đã nhân elu').
  let C = L.C; let D = L.D
  var p: VmProgram
  p.emitDecodeBHX(L)
  p.emitLoadConstF32(7, float32(L.T)); p.emitLoadConstF32(10, float32(D))
  p.emitUMulK(13, 3, uint32(D))
  p.emitUMulK(4, 2, uint32(L.T * 2 * C)); p.emitUAdd(4, 4, 13); p.emitUAdd(4, 4, 1)
  p.emitUAddK(4, 4, uint32((L.T - 1) * 2 * C))
  p.emitUMulK(5, 2, uint32(L.T * 3 * C)); p.emitUAdd(5, 5, 13); p.emitUAddK(5, 5, uint32((L.T - 1) * 3 * C))
  p.emitUMulK(6, 2, uint32(L.T * C)); p.emitUAdd(6, 6, 13); p.emitUAddK(6, 6, uint32((L.T - 1) * C))
  p.emitUMulK(7, 2, uint32(L.T * L.H)); p.emitUAdd(7, 7, 3); p.emitUAddK(7, 7, uint32((L.T - 1) * L.H))
  p.emitUMulK(8, 0, uint32(D))
  p.emitUAddK(9, 0, 0)
  p.emitUMulK(12, 2, uint32(L.T * 3 * C)); p.emitUAdd(12, 12, 13); p.emitUAdd(12, 12, 1)
  p.emitUAddK(12, 12, uint32((L.T - 1) * 3 * C))
  sessionVmForGpu(p, 7):
    p.emitULoadF32(8, 4, uint32(L.phi))                         # qf[d]
    p.emitULoadF32(9, 7, uint32(L.invDen))
    p.emitLoadConstF32(13, 0.0'f32)
    p.emitUAddK(10, 5, 0); p.emitUAddK(11, 6, 0); p.emitUAddK(14, 8, 0)
    sessionVmForGpu(p, 10):
      p.emitULoadF32(11, 11, uint32(L.dY)); p.emitMul(11, 11, 9)       # g[e]
      p.emitULoadF32(12, 14, uint32(L.rK))
      p.emitFma(12, 8, 11, 12); p.emitUStoreF32(12, 14, uint32(L.rK))
      p.emitULoadF32(11, 10, uint32(L.qkv + 2 * C)); p.emitFma(13, 12, 11, 13)
      p.emitUAddK(10, 10, 1); p.emitUAddK(11, 11, 1); p.emitUAddK(14, 14, 1)
    p.emitULoadF32(12, 9, uint32(L.zK)); p.emitULoadF32(14, 7, uint32(L.dDen))
    p.emitFma(12, 14, 8, 12); p.emitUStoreF32(12, 9, uint32(L.zK))
    p.emitAdd(13, 13, 12)
    p.emitULoadF32(11, 12, uint32(L.qkv + C))
    p.emitEluGrad(12, 11, 14, 15)
    p.emitMul(13, 13, 12)
    p.emitUStoreF32(13, 12, uint32(L.dQKV + C))
    p.emitUAddK(4, 4, negStride(2 * C)); p.emitUAddK(5, 5, negStride(3 * C)); p.emitUAddK(6, 6, negStride(C))
    p.emitUAddK(7, 7, negStride(L.H)); p.emitUAddK(12, 12, negStride(3 * C))
  p.emitHalt()
  result = p

proc buildLpBwdVProgram*(L: LpLay): VmProgram =
  ## 1 thread / (b,h,e), quét NGƯỢC: giữ cột dS[:,e]; ghi dV_t[e].
  let C = L.C; let D = L.D
  var p: VmProgram
  p.emitDecodeBHX(L)
  p.emitLoadConstF32(7, float32(L.T)); p.emitLoadConstF32(10, float32(D))
  p.emitUMulK(13, 3, uint32(D))
  p.emitUMulK(4, 2, uint32(L.T * 2 * C)); p.emitUAdd(4, 4, 13); p.emitUAddK(4, 4, uint32((L.T - 1) * 2 * C))
  p.emitUMulK(6, 2, uint32(L.T * C)); p.emitUAdd(6, 6, 13); p.emitUAdd(6, 6, 1)
  p.emitUAddK(6, 6, uint32((L.T - 1) * C))
  p.emitUMulK(7, 2, uint32(L.T * L.H)); p.emitUAdd(7, 7, 3); p.emitUAddK(7, 7, uint32((L.T - 1) * L.H))
  p.emitUMulK(8, 0, uint32(D))
  p.emitUMulK(12, 2, uint32(L.T * 3 * C)); p.emitUAdd(12, 12, 13); p.emitUAdd(12, 12, 1)
  p.emitUAddK(12, 12, uint32((L.T - 1) * 3 * C))
  sessionVmForGpu(p, 7):
    p.emitULoadF32(9, 7, uint32(L.invDen))
    p.emitULoadF32(8, 6, uint32(L.dY)); p.emitMul(8, 8, 9)            # g[e]
    p.emitLoadConstF32(13, 0.0'f32)
    p.emitUAddK(10, 4, 0); p.emitUAddK(14, 8, 0)
    sessionVmForGpu(p, 10):
      p.emitULoadF32(11, 10, uint32(L.phi)); p.emitULoadF32(12, 14, uint32(L.cV))
      p.emitFma(12, 11, 8, 12); p.emitUStoreF32(12, 14, uint32(L.cV))
      p.emitULoadF32(11, 10, uint32(L.phi + C)); p.emitFma(13, 12, 11, 13)
      p.emitUAddK(10, 10, 1); p.emitUAddK(14, 14, 1)
    p.emitUStoreF32(13, 12, uint32(L.dQKV + 2 * C))
    p.emitUAddK(4, 4, negStride(2 * C)); p.emitUAddK(6, 6, negStride(C))
    p.emitUAddK(7, 7, negStride(L.H)); p.emitUAddK(12, 12, negStride(3 * C))
  p.emitHalt()
  result = p

# ═════════════════════════════════════════════════════════════
# (3) Softmax cross-entropy có mask: loss + dLogits trong 1 kernel / hàng
#   target < 0 => bỏ qua (dLogits hàng = 0, loss hàng = 0).
#   dLogits = (softmax - onehot) * invN ;  loss_row = m + ln(Σexp(x-m)) - x_tgt
#   (ln bằng Halley trên GPU vì VM không có opcode LOG)
# ═════════════════════════════════════════════════════════════
type
  CeLay* = object
    rows*, V*: int
    logits*, tgt*, scal*, dLogits*, rowLoss*, total*, sum*: int

proc ceLayout*(rows, V: int): CeLay =
  var L: CeLay
  L.rows = rows; L.V = V
  var o = 0
  template take(f: untyped; n: int) =
    f = o; o += n
  take(L.logits, rows * V); take(L.tgt, rows); take(L.scal, 1)
  take(L.dLogits, rows * V); take(L.rowLoss, rows); take(L.sum, 1)
  L.total = o
  result = L

proc emitTgTreeReduceMax*(p: var VmProgram; idxReg: uint32; n: int;
                          rStride, rCnt, rTmp, rA, rB: uint32) =
  doAssert n >= 1 and (n and (n - 1)) == 0
  p.emitLoadConstF32(rCnt, float32(countTrailingZeroBits(n)))
  p.emitLoadConstF32(rStride, float32(n div 2))
  sessionVmForGpu(p, rCnt):
    p.emitCmpLt(rTmp, idxReg, rStride)
    sessionVmIfGpu(p, rTmp):
      p.emitAdd(rTmp, idxReg, rStride)
      p.emitTgLoadF32(rA, idxReg, 0)
      p.emitTgLoadF32(rB, rTmp, 0)
      p.emitCmpLt(rTmp, rA, rB)
      sessionVmIfGpu(p, rTmp):
        p.emitAddConst(rA, rB, 0.0'f32)
      p.emitTgStoreF32(rA, idxReg, 0)
    p.emitBarrierTg()
    p.emitMulConst(rStride, rStride, 0.5'f32)

proc buildCeProgram*(L: CeLay): VmProgram =
  let V = L.V; let tg = rmsRowTg(V); let ept = V div tg
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitUToR(14, 1)
  p.emitUMulK(2, 0, uint32(V))
  p.emitUConst(3, 0)
  p.emitLoadConstF32(3, float32(ept))
  # ── pass 1: max ──
  p.emitLoadConstF32(1, -1e30'f32); p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(5, 4, float32(tg)); p.emitAdd(5, 5, 14); p.emitRToU(4, 5)
    p.emitUAdd(5, 2, 4); p.emitULoadF32(6, 5, uint32(L.logits))
    p.emitCmpLt(7, 1, 6)
    sessionVmIfGpu(p, 7):
      p.emitAddConst(1, 6, 0.0'f32)
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitTgStoreF32(1, 14, 0); p.emitBarrierTg()
  p.emitTgTreeReduceMax(14, tg, 6, 7, 8, 9, 10)
  p.emitLoadConstF32(11, 0.0'f32); p.emitTgLoadF32(13, 11, 0); p.emitBarrierTg()   # R13 = m
  # ── pass 2: Σ exp(x-m) ──
  p.emitLoadConstF32(1, 0.0'f32); p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(5, 4, float32(tg)); p.emitAdd(5, 5, 14); p.emitRToU(4, 5)
    p.emitUAdd(5, 2, 4); p.emitULoadF32(6, 5, uint32(L.logits))
    p.emitSub(6, 6, 13); p.emitExp(6, 6); p.emitAdd(1, 1, 6)
    p.emitAddConst(4, 4, 1.0'f32)
  p.emitTgStoreF32(1, 14, 0); p.emitBarrierTg()
  p.emitTgTreeReduce(14, tg, 6, 7, 8, 9, 10)
  p.emitTgLoadF32(12, 11, 0)                                   # R12 = S
  p.emitLoadConstF32(8, 1.0'f32); p.emitDiv(10, 8, 12)         # R10 = 1/S
  p.emitUConst(3, 0)
  p.emitULoadF32(15, 0, uint32(L.tgt))                         # R15 = tgt[row]
  p.emitULoadF32(7, 3, uint32(L.scal))                         # R7 = invN
  p.emitLoadConstF32(8, -0.5'f32); p.emitCmpLt(9, 8, 15)       # R9 = valid (tgt >= 0)
  # ── pass 3: dLogits ──
  p.emitLoadConstF32(4, 0.0'f32)
  sessionVmForGpu(p, 3):
    p.emitMulConst(5, 4, float32(tg)); p.emitAdd(5, 5, 14); p.emitRToU(4, 5)
    p.emitUAdd(6, 2, 4); p.emitULoadF32(8, 6, uint32(L.logits))
    p.emitSub(8, 8, 13); p.emitExp(8, 8); p.emitMul(8, 8, 10)  # p
    p.emitAddConst(1, 15, 0.5'f32); p.emitCmpLt(1, 5, 1)       # idx < tgt+.5
    p.emitAddConst(2, 15, -0.5'f32); p.emitCmpLt(2, 2, 5)      # tgt-.5 < idx
    p.emitMul(1, 1, 2)                                         # onehot
    p.emitSub(8, 8, 1); p.emitMul(8, 8, 7); p.emitMul(8, 8, 9)
    p.emitUStoreF32(8, 6, uint32(L.dLogits))
    p.emitAddConst(4, 4, 1.0'f32)
  # ── loss_row = m + ln S - x_tgt  (ln S: Halley, 24 vòng) ──
  p.emitLoadConstF32(1, 0.0'f32); p.emitLoadConstF32(4, 24.0'f32)
  sessionVmForGpu(p, 4):
    p.emitExp(2, 1)                                            # e = exp(y)
    p.emitSub(5, 12, 2); p.emitAdd(6, 12, 2); p.emitDiv(5, 5, 6)
    p.emitMulConst(5, 5, 2.0'f32); p.emitAdd(1, 1, 5)
  p.emitMul(8, 15, 9); p.emitRToU(4, 8)                        # col = valid ? tgt : 0
  p.emitUAdd(5, 2, 4); p.emitULoadF32(6, 5, uint32(L.logits))
  p.emitAdd(1, 1, 13); p.emitSub(1, 1, 6); p.emitMul(1, 1, 9)
  p.emitUStoreF32(1, 0, uint32(L.rowLoss))
  p.emitHalt()
  result = p

proc buildCeSumProgram*(L: CeLay): VmProgram =
  ## 1 thread: sum = Σ rowLoss * invN
  var p: VmProgram
  p.emitUConst(0, 0)
  p.emitLoadConstF32(3, float32(L.rows)); p.emitLoadConstF32(1, 0.0'f32)
  p.emitUConst(1, 0)
  sessionVmForGpu(p, 3):
    p.emitULoadF32(2, 1, uint32(L.rowLoss)); p.emitAdd(1, 1, 2); p.emitUAddK(1, 1, 1)
  p.emitULoadF32(2, 0, uint32(L.scal)); p.emitMul(1, 1, 2)
  p.emitUStoreF32(1, 0, uint32(L.sum))
  p.emitHalt()
  result = p

# ═════════════════════════════════════════════════════════════
# (4) Adam gộp 1 kernel (thay chuỗi ~14 phép be* rời):
#   m = b1 m + (1-b1) g ; v = b2 v + (1-b2) g² ; p -= lr * (m/bc1) / (sqrt(v/bc2) + eps)
#   scal = [lr, 1/bc1, 1/bc2] đọc từ buffer => bytecode KHÔNG đổi theo step (không JIT lại).
# ═════════════════════════════════════════════════════════════
const AdamTg* = 256
type
  AdamLay* = object
    n*: int
    p*, g*, m*, v*, scal*, total*: int

proc adamLayout*(n: int): AdamLay =
  doAssert n mod AdamTg == 0, "adamLayout: n phai chia het " & $AdamTg & " (pad phia host)"
  var L: AdamLay
  L.n = n
  var o = 0
  template take(f: untyped; k: int) =
    f = o; o += k
  take(L.p, n); take(L.g, n); take(L.m, n); take(L.v, n); take(L.scal, 4)
  L.total = o
  result = L

proc buildAdamProgram*(L: AdamLay; b1, b2, eps: float32): VmProgram =
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitUMulK(2, 0, uint32(AdamTg)); p.emitUAdd(2, 2, 1)       # U2 = i
  p.emitUConst(3, 0)
  p.emitULoadF32(1, 2, uint32(L.g))                            # g
  p.emitULoadF32(2, 2, uint32(L.m))                            # m
  p.emitMulConst(2, 2, b1); p.emitMulConst(3, 1, 1.0'f32 - b1); p.emitAdd(2, 2, 3)   # m'
  p.emitUStoreF32(2, 2, uint32(L.m))
  p.emitULoadF32(4, 2, uint32(L.v))
  p.emitMulConst(4, 4, b2); p.emitMul(5, 1, 1); p.emitMulConst(5, 5, 1.0'f32 - b2); p.emitAdd(4, 4, 5)  # v'
  p.emitUStoreF32(4, 2, uint32(L.v))
  p.emitUConst(3, 0)
  p.emitULoadF32(6, 3, uint32(L.scal))                         # lr
  p.emitULoadF32(7, 3, uint32(L.scal + 1))                     # 1/bc1
  p.emitULoadF32(8, 3, uint32(L.scal + 2))                     # 1/bc2
  p.emitMul(2, 2, 7)                                           # mHat
  p.emitMul(4, 4, 8)                                           # vHat
  p.emitAddConst(9, 4, 1e-30'f32); p.emitRsqrt(9, 9); p.emitMul(9, 9, 4)   # sqrt(vHat)
  p.emitAddConst(9, 9, eps)
  p.emitDiv(2, 2, 9); p.emitMul(2, 2, 6)                       # step
  p.emitULoadF32(10, 2, uint32(L.p)); p.emitSub(10, 10, 2)
  p.emitUStoreF32(10, 2, uint32(L.p))
  p.emitHalt()
  result = p
