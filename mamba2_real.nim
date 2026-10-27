## mamba2_real.nim
## Bản Mamba2 ĐÚNG công thức gốc (khác Mamba2SSD "linear-attention-có-decay-
## đơn-giản-hoá" trong test_nimformer.nim). Khớp kiến trúc HF `Mamba2Mixer`
## mà mistralai/Mamba-Codestral-7B-v0.1 dùng, để weight pretrain map 1-1.
##
## Toán học (mỗi timestep t, head h, channel d trong head, state n):
##   proj            = inProj(x)
##   z, xBC, dt_in   = split(proj)
##   xBC_conv        = SiLU(CausalConv1dDepthwise(xBC))
##   x, B, C         = split(xBC_conv)          # B, C dùng chung theo GROUP
##   A_h             = -exp(A_log[h])
##   dt_t,h          = softplus(dt_in_t,h + dt_bias[h])
##   dA_t,h          = exp(dt_t,h * A_h)
##   s_t             = s_{t-1} * dA_t,h + dt_t,h * B_t,g(h) ⊗ x_t,h,d
##   y_t,h,d         = Σ_n C_t,g(h),n * s_t,h,d,n + D[h] * x_t,h,d
##   gated           = y * SiLU(z)
##   normed          = RMSNorm(gated) * normWeight
##   out             = outProj(normed)
## g(h) = h div (numHeads div nGroups)  (B/C share theo group, kiểu GQA).
##
## HIỆU NĂNG: backward là BPTT TUẦN TỰ qua từng timestep (đúng toán, không
## dùng thuật chunked-SSD song song của paper gốc) - cache state đầy đủ nên
## tốn RAM O(T*H*headDim*dState). Dùng tốt cho verify + finetune seq ngắn/
## batch nhỏ; seq dài trên 7B thật cần viết thêm bản chunked sau.

import std/[math, tables, os, strutils, strformat]
import backend
import nimformer   # Tensor, Linear, newLinear, randnTensor, newTensor
import fp4_gemv_gpu  # fp4GemvGpu - dùng khi fp4InName/fp4OutName != "" (goi tu lora.nim)
import mamba2_ssm_gpu  # mamba2StepGpu - dùng khi useGpuSsm=true
import mamba2_full_vm  # forward/backward cả lớp bằng micro-kernel

type
  Mamba2Real* = object
    hiddenSize*, intermediateSize*, numHeads*, headDim*, nGroups*, dState*, convKernel*: int
    inProj*: Linear
    convWeight*: Tensor    # [convDim, convKernel]
    convBias*: Tensor      # [convDim]
    aLog*: Tensor          # [numHeads]
    dSkip*: Tensor         # [numHeads]  (= "D" trong HF)
    dtBias*: Tensor        # [numHeads]
    normWeight*: Tensor    # [intermediateSize]
    outProj*: Linear
    normEps*: float32

proc convDim*(m: Mamba2Real): int = m.intermediateSize + 2 * m.nGroups * m.dState

proc newMamba2Real*(hiddenSize, intermediateSize, numHeads, dState, nGroups: int,
                     convKernel: int = 4, normEps: float32 = 1e-5'f32;
                     initializeWeights = true): Mamba2Real =
  assert intermediateSize mod numHeads == 0, "intermediateSize phai chia het numHeads"
  assert numHeads mod nGroups == 0, "numHeads phai chia het nGroups"
  result.hiddenSize = hiddenSize
  result.intermediateSize = intermediateSize
  result.numHeads = numHeads
  result.headDim = intermediateSize div numHeads
  result.nGroups = nGroups
  result.dState = dState
  result.convKernel = convKernel
  result.normEps = normEps
  let cDim = intermediateSize + 2 * nGroups * dState
  result.inProj = newLinear(hiddenSize, 2 * intermediateSize + 2 * nGroups * dState + numHeads,
    initializeWeights)
  result.convWeight = randnTensor(@[cDim, convKernel], 1'f32 / sqrt(float32(convKernel)))
  result.convBias = newTensor(@[cDim])
  result.aLog = newTensor(@[numHeads])
  for h in 0 ..< numHeads:
    result.aLog.data[h] = ln(float32(h mod 16 + 1))
  result.dSkip = newTensor(@[numHeads], 1'f32)
  result.dtBias = newTensor(@[numHeads])
  result.normWeight = newTensor(@[intermediateSize], 1'f32)
  result.outProj = newLinear(intermediateSize, hiddenSize, initializeWeights)

proc inProjSsmRowRange*(intermediateSize, nGroups, dState, numHeads: int): (int, int) =
  ## Hàng [lo, hi) của inProj.weight thuộc nhánh SSM (xBC + dt), không gồm z-gate.
  ## Layout out của inProj: [z(C) | xBC(C + 2*G*N) | dt(H)].
  ## Dùng khi LoRA chỉ muốn đụng phần nuôi SSM (tránh poison dt/B/C qua z).
  let C = intermediateSize
  let cDim = C + 2 * nGroups * dState
  (C, C + cDim + numHeads)

proc inProjSsmRowRange*(m: Mamba2Real): (int, int) =
  inProjSsmRowRange(m.intermediateSize, m.nGroups, m.dState, m.numHeads)

type
  Mamba2RealCache* = object
    ## Toàn bộ trung gian forward nằm trong MỘT buffer GPU (layout M2Lay) — backward dùng lại
    ## đúng buffer này, không dựng lại gì trên CPU.
    B*, T*: int
    buf*: seq[float32]
    lay*: M2Lay
    proj*: Tensor

proc sliceTensor(buf: seq[float32]; off, n: int; shape: seq[int]): Tensor =
  ## Chép 1 vùng buffer GPU ra Tensor (copyMem, không tính toán)
  result = Tensor(shape: shape, data: newSeq[float32](n))
  result.data.blit(0, buf, off, n)

proc forwardMamba2Real*(m: Mamba2Real, x: Tensor, ctx: Backend):
                        tuple[y: Tensor, cache: Mamba2RealCache] =
  ## inProj / outProj: GEMM trên GPU (Linear.forward). Phần còn lại (conv1d+SiLU + softplus dt,
  ## selective scan mọi timestep, gated RMSNorm) = 3 micro-kernel chạy liên tiếp trong 1 session
  ## trên cùng 1 buffer; scan dùng affine prefix song song theo T nếu T<=64 và
  ## threadgroup hợp lệ, còn lại dùng recurrence tuần tự. Backward vẫn dùng scan ngược tuần tự.
  ensureTgCap(ctx)   # tgSize reduction phai <= gioi han GPU (Iris 640: 256)
  let B = x.shape[0]; let T = x.shape[1]
  assert x.shape[2] == m.hiddenSize
  let H = m.numHeads; let D = m.headDim; let N = m.dState; let G = m.nGroups
  let cDim = m.convDim
  let K = m.convKernel
  let C = m.intermediateSize

  let L = m2Layout(B, T, H, D, N, G, K, C, m.normEps, G)
  checkVmScratchCapacity(ctx, L.total,
    &"Mamba2Real (B={B}, T={T}, H={H}, D={D}, N={N})")
  let proj = m.inProj.forward(x, ctx)
  var buf = newSeq[float32](L.total)
  buf.blit(L.proj, proj.data, 0, L.BT * L.P)
  buf.blit(L.convW, m.convWeight.data, 0, cDim * K)
  buf.blit(L.convB, m.convBias.data, 0, cDim)
  buf.blit(L.aLog, m.aLog.data, 0, H)
  buf.blit(L.dSkip, m.dSkip.data, 0, H)
  buf.blit(L.dtBias, m.dtBias.data, 0, H)
  buf.blit(L.normW, m.normWeight.data, 0, C)
  let tgR = rmsTg(L)
  discard runMicroVmChain(ctx, [
    (buildM2ConvDtProgram(L), L.BT * M2Tg, 4, M2Tg, "@m2_conv_dt_" & m2Tag(L)),
    mamba2ScanStep(L, ctx),
    (buildM2RmsFwdProgram(L), L.BT * L.normGroups * tgR, tgR * 4, tgR, "@m2_rms_" & m2Tag(L))], buf)
  let normedOut = sliceTensor(buf, L.normedOut, L.BT * C, @[B, T, C])
  let y = m.outProj.forward(normedOut, ctx)
  result = (y, Mamba2RealCache(B: B, T: T, buf: buf, lay: L, proj: proj))

# ─────────────────────────────────────────────────────────────
# Backward - 4 micro-kernel (rms -> scan -> dt -> conv), 1 session.
# ─────────────────────────────────────────────────────────────

proc backwardMamba2Real*(m: Mamba2Real, x: Tensor, cache: Mamba2RealCache,
                          dOut: Tensor, ctx: Backend, prefix: string,
                          grads: var Table[string, Tensor]): Tensor =
  ensureTgCap(ctx)
  let L = cache.lay
  let B = cache.B; let T = cache.T
  let H = m.numHeads; let C = m.intermediateSize
  let cDim = m.convDim
  let K = m.convKernel

  # ---- outProj backward (GEMM GPU) ----
  let normedOut = sliceTensor(cache.buf, L.normedOut, L.BT * C, @[B, T, C])
  let (dNormedOutFull, dOutProjW, dOutProjB) = m.outProj.backward(normedOut, dOut, ctx)
  grads[prefix & ".outProj.weight"] = dOutProjW
  grads[prefix & ".outProj.bias"] = dOutProjB

  # ---- RMSNormGated + selective scan + softplus-dt + conv1d/SiLU backward: 4 kernel GPU ----
  var buf = cache.buf                       # vùng d* trong cache.buf còn 0 -> an toàn cho atomic
  buf.blit(L.dNormedOut, dNormedOutFull.data, 0, L.BT * C)
  let tgR = rmsTg(L)
  discard runMicroVmChain(ctx, [
    (buildM2RmsBwdProgram(L), L.BT * L.normGroups * tgR, tgR * 4, tgR, "@m2_rms_bwd_" & m2Tag(L))], buf)
  # scan backward: chia n thành các đoạn, MỖI đoạn 1 command buffer riêng (watchdog GPU Intel
  # giết command buffer > vài giây). Atomic cộng dồn nên tổng các đoạn == chạy 1 lần.
  var nChunk = 16
  let envC = getEnv("BYBY_SCAN_NCHUNK")
  if envC.len > 0:
    try: nChunk = max(1, parseInt(envC))
    except ValueError: discard
  nChunk = min(nChunk, m.dState)
  let per = (m.dState + nChunk - 1) div nChunk
  var nb = 0
  while nb < m.dState:
    let cnt = min(per, m.dState - nb)
    discard runMicroVmChain(ctx, [
      (buildM2ScanBwdProgram(L, nb, cnt), B * L.HD, 4, 0,
       "@m2_scan_bwd_" & m2Tag(L) & "_n" & $nb & "_" & $cnt)], buf)
    nb += cnt
  discard runMicroVmChain(ctx, [
    (buildM2DtBwdProgram(L), L.BT * M2Tg, 4, M2Tg, "@m2_dt_bwd_" & m2Tag(L)),
    (buildM2ConvBwdProgram(L), L.BT * M2Tg, 4, M2Tg, "@m2_conv_bwd_" & m2Tag(L))], buf)

  grads[prefix & ".normWeight"] = sliceTensor(buf, L.dNormW, C, @[C])
  grads[prefix & ".dSkip"] = sliceTensor(buf, L.dDSkip, H, @[H])
  grads[prefix & ".dtBias"] = sliceTensor(buf, L.dDtBias, H, @[H])
  grads[prefix & ".aLog"] = sliceTensor(buf, L.dALog, H, @[H])
  grads[prefix & ".convWeight"] = sliceTensor(buf, L.dConvW, cDim * K, @[cDim, K])
  grads[prefix & ".convBias"] = sliceTensor(buf, L.dConvB, cDim, @[cDim])

  # ---- inProj backward (GEMM GPU); dProj đã được kernel ghép sẵn [dZ | dXBC | dDtPre] ----
  let dProj = sliceTensor(buf, L.dProj, L.BT * L.P, cache.proj.shape)
  let (dX, dInProjW, dInProjB) = m.inProj.backward(x, dProj, ctx)
  grads[prefix & ".inProj.weight"] = dInProjW
  grads[prefix & ".inProj.bias"] = dInProjB
  result = dX

# ─────────────────────────────────────────────────────────────
# Param registry (giong convention paramList trong jamba_hybrid.nim)
# ─────────────────────────────────────────────────────────────

proc paramListMamba2Real*(m: var Mamba2Real, prefix: string): seq[(string, ptr Tensor)] =
  result = @[]
  result.add((prefix & ".inProj.weight", addr m.inProj.weight))
  result.add((prefix & ".inProj.bias", addr m.inProj.bias))
  result.add((prefix & ".convWeight", addr m.convWeight))
  result.add((prefix & ".convBias", addr m.convBias))
  result.add((prefix & ".aLog", addr m.aLog))
  result.add((prefix & ".dSkip", addr m.dSkip))
  result.add((prefix & ".dtBias", addr m.dtBias))
  result.add((prefix & ".normWeight", addr m.normWeight))
  result.add((prefix & ".outProj.weight", addr m.outProj.weight))
  result.add((prefix & ".outProj.bias", addr m.outProj.bias))

# ─────────────────────────────────────────────────────────────
# Map ten tensor HF (mistralai/Mamba-Codestral-7B-v0.1, mixer cua 1 layer)
# -> ten tham so cua Mamba2Real. Dung voi loadAllHFTensors (pytorch_bin_loader)
# hoac loadAllHFTensorsSharded (sharded_hf_loader) de nap TRUC TIEP weight
# pretrain that vao struct nay - CHUYEN VOI Mamba2SSD cu, cai nay map 1-1
# DUNG NGHIA vi cung 1 cong thuc toan.
# LUU Y: kiem tra lai ten chinh xac qua manifest (pytorch_bin_loader --export
# hoac listStateDictInfo) vi HF co the doi ten giua cac phien ban transformers
# (vd "mixer.dt_bias" vs "mixer.dt_proj.bias", "mixer.D" vs "mixer.dSkip").
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# STATE INCREMENTAL (MOI) - cho INFER 1-token-moi-buoc, O(1)/token thay vi
# O(T) chay lai tu dau (forwardMamba2Real o tren). Cong thuc doi chieu DUNG
# tu source HF that (transformers/models/mamba2/modeling_mamba2.py, ham
# `mamba2_selective_state_update` + phan cap nhat conv_state trong
# `cuda_kernels_forward`/`torch_forward` nhanh "cached_start"), KHONG tu bien:
#   dt[h]      = softplus(dtPre[h] + dtBias[h])
#   dA[h]      = exp(dt[h] * A_h),  A_h = -exp(aLog[h])
#   state[h,d,n] = state[h,d,n]*dA[h] + dt[h]*B[g,n]*x[h,d]
#   y[h,d]     = sum_n( C[g,n] * state[h,d,n] ) + D[h]*x[h,d]
# conv1d: giu convState = K-1 buoc xBCPre GAN NHAT (rolling window), moi
# buoc conv = sum_k convWeight[c,k]*window[k] (window = convState + xBCPre
# moi, K phan tu), roi truot convState (bo phan tu cu nhat, them xBCPre moi).
# CHI dung cho B=1, T=1 (1 chuoi, sinh tung token) - dung y generate().
# ─────────────────────────────────────────────────────────────

type
  Mamba2RealState* = object
    ## convState: [convDim, convKernel-1] - K-1 buoc xBCPre GAN NHAT (truoc
    ## SiLU, truoc khi tach x/B/C) - CHINH XAC thu HF luu trong
    ## `cache_params.conv_states[layer_idx]`.
    ## ssmState: [numHeads, headDim, dState] - CHINH XAC thu HF luu trong
    ## `cache_params.ssm_states[layer_idx]` (goi la recurrent_state).
    convState*: seq[float32]
    ssmState*: seq[float32]

proc newMamba2RealState*(m: Mamba2Real): Mamba2RealState =
  ## Zero-init - dung y het "s_{-1}=0" / "conv_state=0" luc bat dau 1 chuoi
  ## moi (chua co token nao truoc do).
  result.convState = newSeq[float32](m.convDim * (m.convKernel - 1))
  result.ssmState = newSeq[float32](m.numHeads * m.headDim * m.dState)

proc forwardStepMamba2Real*(m: Mamba2Real, x: Tensor, state: var Mamba2RealState,
                             ctx: Backend, fp4InName: string = "", fp4OutName: string = "",
                             useGpuSsm: bool = true): Tensor =
  ## x: [1,1,hiddenSize] - DUNG 1 token moi. Cap nhat state TAI CHO (var),
  ## tra ve y: [1,1,hiddenSize]. KHONG giu cache backward (chi dung infer).
  ## fp4InName/fp4OutName: xem ghi chu cu.
  ## useGpuSsm: true (mac dinh) -> SSM + softplus + (RMS khi C<=1024) qua
  ## microVM (mamba2StepGpu / buildRmsGateProgram / buildConv1dIncProgram).
  ## Interpreter da co du opcode tren Metal/CUDA/Vulkan. Conv1d + RMS lon
  ## van co the fallback host neu kernel chua wire day du; goi site phai
  ## sessionBegin truoc de ghep session.
  assert x.shape == @[1, 1, m.hiddenSize], "forwardStepMamba2Real: chi nhan B=1,T=1"
  let H = m.numHeads; let D = m.headDim; let N = m.dState; let G = m.nGroups
  let hpg = H div G
  let cDim = m.convDim
  let K = m.convKernel
  let C = m.intermediateSize

  var proj: Tensor
  if fp4InName.len > 0:
    let projData = fp4GemvGpu(ctx, fp4InName, x.data, 1)   # 100% GPU, khong materialize
    proj = newTensor(@[1, 1, 2*C + 2*G*N + H])
    proj.data = projData
    # LoRA inProj: delta PHAI cong VAO proj TRUOC conv/SSM (W'=W+BA), khong
    # the cong sau SSM. Linear.forward() da lam viec nay khi khong dung FP4.
    if m.inProj.loraR > 0 and m.inProj.loraA.len == m.inProj.loraR * m.inProj.inF and
       m.inProj.loraB.len == m.inProj.outF * m.inProj.loraR:
      let r = m.inProj.loraR
      let At = transpose(Tensor(shape: @[r, m.inProj.inF], data: m.inProj.loraA))
      let Bt = transpose(Tensor(shape: @[m.inProj.outF, r], data: m.inProj.loraB))
      let xA = beMatmul(ctx, x.data, 1, m.inProj.inF, At.data, m.inProj.inF, r)
      let delta = beMatmul(ctx, xA, 1, r, Bt.data, r, m.inProj.outF)
      axpyGpu(ctx, proj.data, delta, m.inProj.loraScale)   # GPU: proj += scale*delta
  else:
    proj = m.inProj.forward(x, ctx)   # [1,1, 2C+2GN+H] - path CPU/resident (co LoRA neu gan)
  # ---- 100% GPU path (microVM) — useGpuSsm mặc định true, không CPU ----
  var z: seq[float32]
  var xBCPre: seq[float32]
  var dtPre: seq[float32]
  var xBCConv: seq[float32]
  var xPart: seq[float32]
  var bMat: seq[float32]
  var cMat: seq[float32]
  var dt: seq[float32]
  var yPre: seq[float32]
  var normedOut: Tensor

  if not useGpuSsm:
    raise newException(CatchableError,
      "forwardStepMamba2Real: useGpuSsm=false bị CẤM — framework yêu cầu 100% GPU microVM")
  if true:
    # 0) split proj trên GPU
    let spl = splitProjGpu(ctx, C, cDim, H, proj.data)
    z = spl.z
    xBCPre = spl.xBC
    dtPre = spl.dt

    # 1) conv1d
    xBCConv = conv1dIncGpu(ctx, cDim, K, xBCPre, state.convState,
                           m.convWeight.data, m.convBias.data)
    # xBCConv layout đã là [xPart | B | C] — slice zero-copy view qua openArray
    # mamba2StepGpuFromXbc pack trên GPU buffer (không for host)
    dt = softplusDtGpu(ctx, H, dtPre, m.dtBias.data)
    yPre = mamba2StepGpuFromXbc(ctx, H, D, N, G, C, xBCConv, dt,
                                m.aLog.data, m.dSkip.data, state.ssmState)

    # 4) RMS+gate 100% GPU (kể cả C=8192)
    let outData = rmsGateGpu(ctx, C, m.normEps, yPre, z, m.normWeight.data, G)
    normedOut = newTensor(@[1, 1, C])
    normedOut.data = outData
  if fp4OutName.len > 0:
    let outData = fp4GemvGpu(ctx, fp4OutName, normedOut.data, 1)
    result = newTensor(@[1, 1, m.hiddenSize])
    result.data = outData
    # LoRA outProj: activation = normedOut [1,1,C] (C=intermediate, vd 8192),
    # KHONG phai x/ln1 (hidden 4096).
    if m.outProj.loraR > 0 and m.outProj.loraA.len == m.outProj.loraR * m.outProj.inF and
       m.outProj.loraB.len == m.outProj.outF * m.outProj.loraR:
      let r = m.outProj.loraR
      let At = transpose(Tensor(shape: @[r, m.outProj.inF], data: m.outProj.loraA))
      let Bt = transpose(Tensor(shape: @[m.outProj.outF, r], data: m.outProj.loraB))
      let xA = beMatmul(ctx, normedOut.data, 1, m.outProj.inF, At.data, m.outProj.inF, r)
      let delta = beMatmul(ctx, xA, 1, r, Bt.data, r, m.outProj.outF)
      axpyGpu(ctx, result.data, delta, m.outProj.loraScale)   # GPU: out += scale*delta
  else:
    result = m.outProj.forward(normedOut, ctx)

proc hfNameMapMamba2*(layerIdx: int): Table[string, string] =
  result = initTable[string, string]()
  let p = "backbone.layers." & $layerIdx & ".mixer."
  result["inProj.weight"] = p & "in_proj.weight"
  result["inProj.bias"]   = p & "in_proj.bias"      # co the KHONG ton tai (Mamba2 goc thuong bias=False)
  result["convWeight"]    = p & "conv1d.weight"
  result["convBias"]      = p & "conv1d.bias"
  result["aLog"]          = p & "A_log"
  result["dSkip"]         = p & "D"
  result["dtBias"]        = p & "dt_bias"
  result["normWeight"]    = p & "norm.weight"
  result["outProj.weight"] = p & "out_proj.weight"
  result["outProj.bias"]   = p & "out_proj.bias"    # thuong KHONG ton tai
  # FIX token rac (bug #7): pre-norm CUA CA BLOCK (ln1 trong Nim) tuong duong
  # Mamba2Block.norm ben HF, nam NGOAI submodule .mixer nen KHONG dung `p`
  # (trong `p` da co san ".mixer."). Truoc day THIEU HAN key nay -> ln1.gamma
  # khong bao gio duoc nap tu checkpoint, giu nguyen gamma=1.0 (identity mac
  # dinh cua newLayerNorm) cho MOI block, MOI lan chay -> scale sai truoc
  # mixer o ca 64 block -> token rac.
  result["ln1Weight"] = "backbone.layers." & $layerIdx & ".norm.weight"
