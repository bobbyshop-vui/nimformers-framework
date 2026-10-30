## Inference-only compact scan, chunk carry and GPU FP4 tiles.
## Training layouts/kernels in mamba2_full_vm remain unchanged.
import session_vm, mamba2_full_vm

proc prefixLayout*(B,T,H,D,N,G,K,C: int; eps: float32; groups: int; compact: bool): M2Lay =
  result = m2Layout(B,T,H,D,N,G,K,C,eps,groups)
  if compact:
    let removed = B*(T-1)*H*D*N
    result.yPre -= removed
    result.gated -= removed
    result.rms -= removed
    result.normedPre -= removed
    result.normedOut -= removed
    result.fwdEnd -= removed
    result.total = result.fwdEnd

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

proc buildPrefixConvProgram*(L: M2Lay; historyOffset = -1): VmProgram =
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
        if historyOffset >= 0:
          sessionVmIfGpu(p, 14):
            p.emitUMulK(9, 5, uint32(K-1))
            p.emitRToU(10,11); p.emitUAdd(9,9,10)
            p.emitULoadF32(9,9,uint32(historyOffset))
            p.emitULoadF32(10,7,uint32(L.convW))
            p.emitFma(8,10,9,8)
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
proc buildPrefixConvDtProgram*(L: M2Lay; historyOffset = -1): VmProgram =
  result = buildPrefixConvProgram(L,historyOffset)
  result.words.setLen(result.words.len-1)
  result.words.add(buildM2DtProgram(L).words)

proc buildPrefixStateTransposeProgram*(L: M2Lay; toCoalesced: bool): VmProgram =
  ## Two compact state slots: normal [b,hd,n] and work [b,n,hd].
  ## The scan then reads/writes adjacent channels with adjacent GPU lanes.
  var p: VmProgram
  p.emitRToU(0, 0)
  p.emitUToR(1, 0); p.emitAddConst(1, 1, 0.5'f32)
  p.emitMulConst(2, 1, 1'f32/float32(L.HD*L.N)); p.emitRToU(1, 2)
  p.emitUMulK(2, 1, uint32(L.HD*L.N))
  p.emitUMulK(3, 2, 0xFFFFFFFF'u32); p.emitUAdd(3, 0, 3)
  p.emitUToR(3, 3); p.emitAddConst(3, 3, 0.5'f32)
  p.emitMulConst(4, 3, 1'f32/float32(L.N)); p.emitRToU(4, 4) # hd
  p.emitUMulK(5, 4, uint32(L.N)); p.emitUMulK(5, 5, 0xFFFFFFFF'u32)
  p.emitUAdd(5, 0, 5); p.emitUMulK(6, 2, 0xFFFFFFFF'u32); p.emitUAdd(5, 5, 6) # n
  p.emitUMulK(5, 5, uint32(L.HD)); p.emitUAdd(5, 5, 4)
  p.emitUAdd(5, 5, 2); p.emitUAdd(5, 5, 2) # b*2*N*HD + n*HD + hd
  p.emitUAdd(0, 0, 2) # b*2*N*HD + hd*N + n
  if toCoalesced:
    p.emitULoadF32(5, 0, uint32(L.states))
    p.emitUStoreF32(5, 5, uint32(L.states+L.HD*L.N))
  else:
    p.emitULoadF32(5, 5, uint32(L.states+L.HD*L.N))
    p.emitUStoreF32(5, 0, uint32(L.states))
  p.emitHalt()
  p

proc buildCompactPrefixScanProgram*(L: M2Lay; coalesced = false): VmProgram =
  let T = L.T; let N = L.N; let HD = L.HD; let H = L.H; let cDim = L.cDim
  var p: VmProgram
  p.emitDecodeBHD(L)
  p.emitULoadF32(8, 2, uint32(L.aLog)); p.emitExp(8, 8); p.emitNeg(8, 8)    # R8 = A
  p.emitULoadF32(9, 2, uint32(L.dSkip))                                     # R9 = D[h]
  p.emitUMulK(4, 0, uint32(T * cDim))                                       # U4 = convT
  p.emitUMulK(5, 0, uint32(T * H)); p.emitUAdd(5, 5, 2)                     # U5 = dtT
  p.emitUMulK(6, 0, uint32(T * L.C)); p.emitUAdd(6, 6, 1)                   # U6 = yT
  p.emitUMulK(7, 0, uint32(2 * HD * N)); p.emitUMulK(8, 1, uint32(N)); p.emitUAdd(7, 7, 8)  # U7 = sT
  if coalesced:
    p.emitUMulK(7, 0, uint32(2*HD*N)); p.emitUAdd(7, 7, 1)
    p.emitUAddK(7, 7, uint32(HD*N)) # work [b,n,hd]
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
      if not coalesced:
        p.emitUStoreF32(7, 12, uint32(L.states + HD * N))
      p.emitUStoreF32(7, 12, uint32(L.states))
      p.emitFma(4, 6, 7, 4)
      p.emitUAddK(11, 11, 1)
      p.emitUAddK(12, 12, uint32(if coalesced: HD else: 1))
    p.emitFma(4, 9, 3, 4)                                                  # += D*x
    p.emitUStoreF32(4, 6, uint32(L.yPre))
    p.emitUAddK(4, 4, uint32(cDim)); p.emitUAddK(5, 5, uint32(H))
    p.emitUAddK(6, 6, uint32(L.C))
  p.emitHalt()
  result = p

proc m2ScanStateLanes*(N, timeLanes: int): int =
  ## Spread the state dimension across lanes in each workgroup. Four lanes
  ## shrink the common N=128 loop from 128 iterations to 32 while keeping
  ## scan scratch below the Vulkan VM's 1024-float limit.
  for lanes in [4, 2]:
    if N mod lanes == 0 and timeLanes * lanes <= gTgCap and
        timeLanes * 4 * lanes <= 1024:
      return lanes
  result = 1

proc buildPrefixParallelScanProgram*(L: M2Lay; tg: int; continuation = false;
                                    stateLanes = 1): VmProgram =
  ## Inclusive affine prefix scan over time, one threadgroup per (batch, head,
  ## channel). Each group partitions N over stateLanes subgroups; each lane
  ## still represents one timestep. The affine operator
  ## (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1) is associative, so Hillis-Steele
  ## gives the same recurrence as buildM2ScanProgram, modulo FP32 order.
  ##
  ## Two ping-pong banks hold (a,b) pairs. Padded lanes are the identity
  ## (1,0), and every lane reaches each barrier. Output state[t+1] is written
  ## in the exact layout consumed by the backward kernel.
  doAssert tg >= L.T and (tg and (tg - 1)) == 0 and tg * stateLanes <= gTgCap,
    "Mamba2 parallel scan requires a power-of-two threadgroup >= T"
  doAssert stateLanes > 0 and L.N mod stateLanes == 0 and
    tg * 4 * stateLanes <= 1024,
    "Mamba2 parallel scan state lanes exceed shared-memory or N limits"
  let T = L.T
  let N = L.N
  let HD = L.HD
  let cDim = L.cDim
  var p: VmProgram
  p.emitUTgid(0)                    # one group per (b,hd)
  p.emitULid(1)                     # local id = nLane*tg + timestep lane
  p.emitUAndK(4, 1, uint32(tg - 1)) # U4 = timestep lane
  var timeShift = 0
  var timeMask = tg - 1
  while timeMask > 0:
    inc timeShift
    timeMask = timeMask shr 1
  p.emitUShrK(16, 1, uint32(timeShift)) # U16 = state lane
  p.emitUMulK(17, 16, uint32(4 * tg))
  p.emitUAdd(17, 17, 4)             # U17 = private scan-bank index
  p.emitUToR(14, 17)                # R14 = shared index base
  p.emitUToR(0, 0)                  # R0 = group/channel id for emitDecodeBHD
  p.emitUAddK(13, 16, 0)            # U13 = first N owned by this lane
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
  p.emitLoadConstF32(12, float32(N div stateLanes))
  sessionVmForGpu(p, 12):
    # Each state lane owns one N residue class, shortening the serial N loop.
    # Neutral affine pairs keep padded time lanes in every group barrier.
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
      p.emitUToR(11, 17)                                      # private lane base
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
      p.emitUToR(11, 17)
      p.emitTgLoadF32(5, 11, uint32(srcB))
      if continuation:
        p.emitTgLoadF32(2,11,uint32(srcA))
        p.emitUAddK(11,4,1); p.emitUMulK(11,11,uint32(HD*N))
        p.emitUMulK(11,11,0xFFFFFFFF'u32); p.emitUAdd(11,0,11)
        p.emitULoadF32(6,11,uint32(L.states)); p.emitFma(5,2,6,5)
      p.emitUStoreF32(5, 0, uint32(L.states))
      p.emitULoadF32(6, 12, uint32(L.conv))                   # C_t,g,n
      p.emitFma(4, 6, 5, 4)
    p.emitUAddK(13, 13, uint32(stateLanes))

  # The N lanes accumulated disjoint C*s partials. Reuse scan bank zero to
  # sum them once per timestep, then add the D skip path exactly once.
  p.emitTgStoreF32(4, 14, 0)
  p.emitBarrierTg()
  p.emitUToR(11, 4)                                           # shared base = t
  p.emitUToR(18, 16)
  p.emitLoadConstF32(20, 1.0'f32)
  p.emitCmpLt(19, 18, 20)                                    # state lane zero
  p.emitMul(19, 19, 7)                                       # and valid time lane
  sessionVmIfGpu(p, 19):
    p.emitTgLoadF32(4, 11, 0)
    for lane in 1 ..< stateLanes:
      p.emitTgLoadF32(20, 11, uint32(lane * 4 * tg))
      p.emitAdd(4, 4, 20)
    p.emitULoadF32(3, 10, uint32(L.conv))
    p.emitULoadF32(2, 2, uint32(L.dSkip))
    p.emitFma(4, 2, 3, 4)
    p.emitUStoreF32(4, 9, uint32(L.yPre))
  p.emitHalt()
  result = p


proc buildPrefixExpandFp4*(K,N,firstRow: int; tg=256): VmProgram =
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitUMulK(0,0,uint32(tg)); p.emitUAdd(0,0,1)
  p.emitUShrK(1,0,4)
  p.emitUAddK(2,1,uint32(N*K div 2+firstRow*K div 32))
  p.emitULoadU8(2,2); p.emitUAddK(2,2,16); p.emitULoadConstF(2,2)
  p.emitUAddK(3,0,uint32(firstRow*K div 2)); p.emitULoadU8(3,3)
  p.emitUAndK(4,3,15); p.emitULoadConstF(3,4)
  p.emitUShrK(3,3,4); p.emitULoadConstF(4,3)
  p.emitMul(3,3,2); p.emitMul(4,4,2)
  p.emitUAndK(0,0,15); p.emitUMulK(1,1,32); p.emitUAdd(0,0,1)
  p.emitUStoreF32(3,0); p.emitUStoreF32(4,0,16); p.emitHalt()
  result=p

proc buildPrefixScatter*(N,tileN,firstCol,tg: int; rank=0; scale=1'f32;
    loraOffset=0; biasOffset = -1): VmProgram =
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitUMulK(0,0,uint32(tg)); p.emitUAdd(0,0,1)
  p.emitUToR(0,0); p.emitAddConst(1,0,0.5'f32)
  p.emitMulConst(1,1,1'f32/float32(tileN)); p.emitRToU(1,1)
  p.emitUAddK(3,1,0) # token
  p.emitUMulK(2,1,uint32(tileN)); p.emitUMulK(2,2,0xFFFFFFFF'u32)
  p.emitUAdd(2,0,2); p.emitUMulK(1,1,uint32(N)); p.emitUAdd(1,1,2)
  p.emitULoadIn(2,0)
  if rank>0:
    p.emitUMulK(4,3,uint32(rank));p.emitUAddK(4,4,uint32(loraOffset))
    p.emitUAddK(5,2,uint32(firstCol));p.emitUMulK(5,5,uint32(rank))
    p.emitLoadConstF32(4,0);p.emitLoadConstF32(8,float32(rank))
    sessionVmForGpu(p,8):
      p.emitULoadIn(5,4);p.emitULoadConstF(6,5);p.emitFma(4,5,6,4)
      p.emitUAddK(4,4,1);p.emitUAddK(5,5,1)
    p.emitMulConst(4,4,scale);p.emitAdd(2,2,4)
  if biasOffset>=0:
    p.emitULoadIn(5,2,uint32(biasOffset));p.emitAdd(2,2,5)
  p.emitUStoreF32(2,1,uint32(firstCol)); p.emitHalt()
  result=p
