## jamba_hybrid.nim
## Kiến trúc hybrid: mỗi vị trí chọn theo `pattern: string`:
##   'M' = Mamba2SSD thuần (SSM, O(T) theo chiều dài chuỗi)
##   'A' = LinearPlusAttention thuần ("Linear++" đã có trong test_nimformer)
##   'S' = TransformerBlockSpectron (gated fusion 2 nhánh, xem test_nimformer)
##   'R' = Mamba2Real; 'P' = Mamba2Real + Linear++ sidecar song song.
##
## Ví dụ pattern "MMAMMAMS" = 8 layer: đa số Mamba, xen 1 layer Attention
## mỗi 3 layer, layer cuối dùng gated fusion - đúng kiểu Jamba gốc (attention
## thưa giữa 1 rừng Mamba) nhưng bạn được tự do chọn tỉ lệ, khác với Jamba
## cố định 1:7. Không viết lại toán Mamba2SSD/LinearPlusAttention - IMPORT
## thẳng từ test_nimformer.nim (module, không chạy main vì có isMainModule).
##
## Toàn bộ tham số train được (weight) quản lý qua 1 REGISTRY chung:
## seq[(name: string, ptrTensor: ptr Tensor)] dựng 1 lần lúc tạo model, để
## trainStep/Adam/checkpoint không phải viết tay từng field theo từng loại
## block như ModelAdamState/trainStep gốc (dài dòng, phải sửa 3 chỗ mỗi khi
## thêm loại block mới). Đổi lại: backward mỗi block phải trả về
## Table[string, Tensor] (grad theo đúng tên tham số) thay vì tuple cứng.

import std/[math, random, strformat, tables, json, os, strutils, sets]
import customfloat
import backend
import rawbin            # checkpoint .bin thô - dung RIENG cho pipeline fine-tune,
                          # KHONG dung .nimq (quant.saveQuantStateDict/loadQuantStateDict)
                          # o day - .nimq chi danh cho nim_inference.nim/test_nimformer.nim
when not defined(codestralInference):
  import pytorch_bin_loader  # HF file loading is training-only; inference has no Python bridge.
                            # KHONG ghi ra file trung gian nao tren dia
import nimformer
import mamba2_ssm_gpu       # RMSNorm fwd/bwd bang micro-kernel GPU
import mamba2_real         # Mamba2Real THẬT (khác Mamba2SSD) - conv1d/A_log/dt/D/RMSNormGated
import test_nimformer   # Mamba2SSD, LinearPlusAttention, TransformerBlockSpectron,
                         # Model/training dependency only; hybrid inference uses the separate module below.
import hybrid_block_kernel # end-to-end logical GPU kernel entrypoints for R/P blocks
import linear_plus_infer
                         # FeedForward, LayerNorm, Embedding, Linear, forwardMamba2,
                         # forwardLinearPlus, backwardMamba2, backwardLinearPlus,
                         # newLinear/newLayerNorm/newFeedForward/newEmbedding,
                         # ApfAdamState, apfAdamStep, softmaxCrossEntropy

type
  BlockKind* = enum
    bkMamba = "M"
    bkAttn = "A"
    bkSpectron = "S"
    bkMambaReal = "R"    # Mamba2 THẬT (mamba2_real.nim) - dùng khi load pretrain Codestral-7B
    bkMambaRealPlusAttn = "P" # Mamba2Real + Linear++ sidecar chạy song song
    bkMambaRealGated = "G" # Mamba2Real + Linear++ sidecar, g=gate Mamba

  RmsNorm* = object
    ## RMSNorm ĐÚNG chuẩn pretrain Mamba2/Codestral (Mamba2RMSNorm của HF):
    ##     y = x / sqrt(mean(x^2) + eps) * gamma     (KHÔNG trừ mean, KHÔNG có beta)
    ## Đây là norm của base: backbone.layers[i].norm VÀ backbone.norm_f trong
    ## HF Mamba2Model đều là RMSNorm. Đường train cũ dùng LayerNorm (có trừ
    ## mean + beta) cho block 'R'/'P' và KHÔNG áp norm_f trước lm_head
    ## (preset "ln-nof-g1") - lệch hẳn phân phối pretrain, LoRA r=4 phải gánh
    ## toàn bộ shift đó. Preset mới ("rms-fin"): norm giống HỆT base, thứ duy
    ## nhất khác base là sidecar attention ở block 'P'.
    gamma*: Tensor      # [dim], khởi tạo = 1 (identity), nạp weight pretrain sau
    eps*: float32

  JambaBlock* = object
    kind*: BlockKind
    ln1*, ln2*: LayerNorm
    rms1*: RmsNorm     # MOI (preset rms-fin): norm đầu block cho 'R'/'P', THAY
                       # ln1 trên đường forward/backward của 2 loại block pretrain.
                       # ln1 vẫn tồn tại cho block legacy 'M'/'A'/'S' (không
                       # pretrain, tự train từ đầu nên LayerNorm không sao).
    ff*: FeedForward
    # chỉ field ứng với kind mới thực sự dùng, field khác giữ nguyên default
    mamba*: Mamba2SSD
    mambaReal*: Mamba2Real
    attn*: LinearPlusAttention
    # attnLn1 ĐÃ BỎ - block P dùng chung ln1 với Mamba (giống Jamba gốc)
    fuseGate*: Linear        # chỉ dùng khi kind == bkSpectron

  NimformerModelJamba* = object
    vocab*, embedDim*, nHeads*, ffMult*: int
    pattern*: string
    embed*: Embedding
    blocks*: seq[JambaBlock]
    outProj*: Linear
    normF*: LayerNorm    # LEGACY - KHÔNG CÒN DÙNG trên đường forward/backward.
                         # Giữ field để không vỡ các module ngoài có thể tham
                         # chiếu (loader cũ). Norm cuối thật sự là rmsF dưới đây.
    rmsF*: RmsNorm       # MOI (preset rms-fin): tương đương backbone.norm_f trong
                         # HF Mamba2Model - RMSNorm áp 1 LẦN sau khi qua hết các
                         # block, TRƯỚC lm_head. Trước đây normF tồn tại nhưng
                         # KHÔNG forward nào gọi -> hidden vào lm_head sai scale.

const
  G_MAX_SIDE_RMS_RATIO* = 0.5'f32
  G_MAX_OUTPUT_RMS_RATIO* = 8'f32
  G_MAX_MAMBA_RMS_RATIO* = 1.5'f32
    ## rms(mambaY)/rms(x) — nhánh Mamba TRƯỚC gate/residual. Linear++ đã bị
    ## siết ≤0.5×; cùng thang mixer R 1.5×; 2.0 quá rộng.
  G_MIN_GATE_MEAN* = 0.75'f32
    ## Trung binh gate (sigmoid) tren toan bo phan tu cua 1 G block phai
    ## >= nguong nay. Gate khoi tao ~0.982 (bias=4); neu trung binh tut
    ## xuong duoi 0.75 nghia la nhanh Linear++ dang chiem uu the tren
    ## residual - guard se raise truoc khi checkpoint.

proc newRmsNorm*(dim: int, eps: float32 = 1e-5'f32): RmsNorm =

  ## eps = 1e-5: đúng mặc định layer_norm_eps của HF Mamba2Config (Codestral).
  result.gamma = newTensor(@[dim])
  for i in 0 ..< dim: result.gamma.data[i] = 1'f32   # identity khi chưa nạp weight
  result.eps = eps

proc forward*(n: RmsNorm, x: Tensor, ctx: Backend): Tensor =
  ## Norm trên chiều cuối. GPU (micro-kernel) khi backend là Metal/CUDA/Vulkan;
  ## vòng lặp fp32 CPU chỉ còn là fallback cho backend cpu.
  let C = x.shape[^1]
  let rows = x.data.len div C
  result = newTensor(x.shape)
  if isGpuCtx(ctx):
    # GPU micro-kernel (train_gpu_vm.buildRmsFwdProgram), CPU chi la fallback backend cpu.
    result.data = rmsNormForwardGpu(ctx, x.data, n.gamma.data, rows, C, n.eps)
    return
  for r in 0 ..< rows:
    let base = r * C
    var ms = 0'f32
    for c in 0 ..< C:
      let v = x.data[base + c]
      ms += v * v
    ms /= float32(C)
    let invRms = 1'f32 / sqrt(ms + n.eps)
    for c in 0 ..< C:
      result.data[base + c] = x.data[base + c] * invRms * n.gamma.data[c]

proc backward*(n: RmsNorm, x, dOut: Tensor, ctx: Backend): tuple[dX, dGamma: Tensor] =
  ## Backward đúng của RMSNorm (tuyến tính theo dOut - quan trọng cho block 'P'
  ## có 2 nhánh song song chia chung 1 norm, xem backwardBlock):
  ##   y = x * r * g,  r = (mean(x^2)+eps)^(-1/2)
  ##   dX_j      = r*g_j*dy_j - (r^3/C) * x_j * sum_c(dy_c*g_c*x_c)
  ##   dGamma_c += sum_rows dy_c * x_c * r
  let C = x.shape[^1]
  let rows = x.data.len div C
  result.dX = newTensor(x.shape)
  result.dGamma = newTensor(n.gamma.shape)   # newTensor zero-init sẵn
  if isGpuCtx(ctx):
    let g = rmsNormBackwardGpu(ctx, x.data, n.gamma.data, dOut.data, rows, C, n.eps)
    result.dX.data = g.dX
    result.dGamma.data = g.dGamma
    return
  for r in 0 ..< rows:
    let base = r * C
    var ms = 0'f32
    for c in 0 ..< C:
      let v = x.data[base + c]
      ms += v * v
    ms /= float32(C)
    let invRms = 1'f32 / sqrt(ms + n.eps)
    var dot = 0'f32
    for c in 0 ..< C:
      let xc = x.data[base + c]
      let dyc = dOut.data[base + c]
      result.dGamma.data[c] += dyc * xc * invRms
      dot += dyc * n.gamma.data[c] * xc
    let coeff = dot * invRms * invRms * invRms / float32(C)
    for c in 0 ..< C:
      result.dX.data[base + c] =
        invRms * n.gamma.data[c] * dOut.data[base + c] - x.data[base + c] * coeff

proc concatLastDim(a, b: Tensor): Tensor =
  ## Bản local (test_nimformer.nim không export concatLastDim/splitLastDim
  ## gốc - copy y hệt logic để khỏi phải sửa file gốc của bạn).
  let B = a.shape[0]; let T = a.shape[1]; let C = a.shape[2]
  result = newTensor(@[B, T, 2 * C])
  for i in 0 ..< B * T:
    for c in 0 ..< C:
      result.data[i * 2 * C + c] = a.data[i * C + c]
      result.data[i * 2 * C + C + c] = b.data[i * C + c]

proc splitLastDim(x: Tensor, C: int): tuple[a, b: Tensor] =
  let B = x.shape[0]; let T = x.shape[1]
  result.a = newTensor(@[B, T, C])
  result.b = newTensor(@[B, T, C])
  for i in 0 ..< B * T:
    for c in 0 ..< C:
      result.a.data[i * C + c] = x.data[i * 2 * C + c]
      result.b.data[i * C + c] = x.data[i * 2 * C + C + c]

proc charToKind(c: char): BlockKind =
  case c
  of 'M', 'm': bkMamba
  of 'A', 'a': bkAttn
  of 'S', 's': bkSpectron
  of 'R', 'r': bkMambaReal
  of 'P', 'p': bkMambaRealPlusAttn
  of 'G', 'g': bkMambaRealGated
  else: raise newException(ValueError, &"Ky tu pattern khong hop le: '{c}' (chi nhan M/A/S/R/P/G)")

proc newJambaBlock(kind: BlockKind, embedDim, nHeads, ffMult: int, useSession: bool,
                    mambaRealDState: int = 64, mambaRealNGroups: int = 1,
                    mambaRealConvKernel: int = 4, mambaRealExpand: int = 2,
                    quantResident: bool = false; initializeWeights = true): JambaBlock =
  result.kind = kind
  result.ln1 = newLayerNorm(embedDim)
  result.ln2 = newLayerNorm(embedDim)
  result.rms1 = newRmsNorm(embedDim)   # dùng cho 'R'/'P' (xem forwardBlock)
  case kind
  of bkMamba:
    result.ff = newFeedForward(embedDim, ffMult)
    result.mamba = newMamba2SSD(embedDim, nHeads, useSession)
  of bkAttn:
    result.ff = newFeedForward(embedDim, ffMult)
    result.attn = newLinearPlusAttention(embedDim, nHeads, useSession)
  of bkSpectron:
    result.ff = newFeedForward(embedDim, ffMult)
    result.attn = newLinearPlusAttention(embedDim, nHeads, useSession)
    result.mamba = newMamba2SSD(embedDim, nHeads, useSession)
    result.fuseGate = newLinear(embedDim * 2, embedDim)
  of bkMambaReal:
    # SUA: intermediateSize = embedDim * mambaRealExpand (Codestral-7B that
    # dung expand=2 -> intermediateSize=8192 khi hiddenSize=4096, xac nhan
    # tu shape THAT: mixer.norm.weight=[8192], mixer.out_proj.weight=[4096,8192]
    # - truoc day hard-code expand=1 la SAI, se lam shape lech het khi nap
    # weight that (load_codestral_into_jamba se bao 'SHAPE LECH' tren moi
    # tensor cua mixer). result.ff/ln2 VAN duoc tao o tren (khong xoa field,
    # tranh vo cac cho khac dang gia dinh field luon ton tai) nhung KHONG
    # duoc DUNG trong forward/backward cua bkMambaReal (xem forwardBlock/
    # backwardBlock ben duoi) - vi Mamba2 that cua HF KHONG co FFN xen giua
    # moi layer (chi norm->mixer->residual), khac Transformer block thuong.
    result.mambaReal = newMamba2Real(embedDim, embedDim * mambaRealExpand, nHeads,
                                      mambaRealDState, mambaRealNGroups, mambaRealConvKernel,
                                      initializeWeights = initializeWeights)
    if quantResident:
      # SUA (fix OOM ~29GB luc DUNG MODEL, truoc ca khi load .nimq): inProj/
      # outProj la 2 ma tran to nhat moi block (~90% tong tham so 1 block
      # Mamba2Real). Che do quant-resident (xem load_codestral_into_jamba.nim
      # + lora.nim mergeInOne/restoreBaseOne) da thiet ke san de 2 tensor nay
      # o trang thai RONG (.data = @[]) luc nghi, CHI dequant tam thoi qua
      # LUT dung 1 block ngay truoc forward/backward roi xoa ngay sau (xem
      # trainStepGraftStreaming) - nhung newMamba2Real() VAN cap phat fp32
      # DAY DU cho ca 2 tensor nay luc tao block (randnTensor/newTensor),
      # roi bi bo phi ngay lap tuc (quant-resident khong bao gio doc gia tri
      # nay). Voi 52 block 'R' (embedDim=4096, intermediateSize=8192) ->
      # ~29GB cap phat VO ICH, du chi ton tai thoang qua - da du OOM-kill
      # tren may RAM han che (vd may dung GPU tich hop, RAM se cung he
      # thong). XOA NGAY o day, TRONG VONG LAP (khong doi newNimformerModelJamba
      # tra ve xong moi xoa) - GC thu hoi tung block 1, dinh (peak) RAM
      # trong luc dung model CHI con ~1 block, khong con 52 block cung luc.
      result.mambaReal.inProj.weight.data = @[]
      result.mambaReal.outProj.weight.data = @[]
  of bkMambaRealPlusAttn, bkMambaRealGated:
    result.mambaReal = newMamba2Real(embedDim, embedDim * mambaRealExpand, nHeads,
                                      mambaRealDState, mambaRealNGroups, mambaRealConvKernel,
                                      initializeWeights = initializeWeights)
    result.attn = newLinearPlusAttention(embedDim, nHeads, useSession, initializeWeights)
    # FIX: DÙNG CHUNG ln1 CHO CẢ MAMBA VÀ ATTENTION (giống Jamba gốc).
    # attnLn1 ĐÃ BỎ - 2 nhánh cùng thấy input đã chuẩn hóa y hệt → tránh lệch scale khi cộng residual.
    # P remains exact Mamba identity at initialization. G adds a near-identity
    # Mamba-first gate below, and must keep the sidecar output projection at zero.
    for i in 0 ..< result.attn.outProj.weight.data.len: result.attn.outProj.weight.data[i] = 0'f32
    for i in 0 ..< result.attn.outProj.bias.data.len: result.attn.outProj.bias.data[i] = 0'f32
    if kind == bkMambaRealGated:
      result.fuseGate = newLinear(embedDim * 2, embedDim, initializeWeights)
      for i in 0 ..< result.fuseGate.weight.data.len: result.fuseGate.weight.data[i] = 0'f32
      for i in 0 ..< result.fuseGate.bias.data.len: result.fuseGate.bias.data[i] = 4'f32
    if quantResident:
      result.mambaReal.inProj.weight.data = @[]
      result.mambaReal.outProj.weight.data = @[]

proc newNimformerModelJamba*(vocab, embedDim, nHeads, ffMult: int, pattern: string,
                              useSession: bool = true, mambaRealDState: int = 64,
                              mambaRealNGroups: int = 1, mambaRealConvKernel: int = 4,
                              mambaRealExpand: int = 2, quantResident: bool = false;
                              initializeWeights = true): NimformerModelJamba =
  assert pattern.len > 0, "pattern rong - can it nhat 1 layer"
  result.vocab = vocab
  result.embedDim = embedDim
  result.nHeads = nHeads
  result.ffMult = ffMult
  result.pattern = pattern
  result.embed = newEmbedding(vocab, embedDim, initializeWeights)
  result.outProj = newLinear(embedDim, vocab, initializeWeights)
  result.normF = newLayerNorm(embedDim)   # LEGACY - không còn dùng (xem struct)
  result.rmsF = newRmsNorm(embedDim)      # MOI (rms-fin): norm_f THẬT trước lm_head
  result.blocks = newSeq[JambaBlock](pattern.len)
  for i, c in pattern:
    result.blocks[i] = newJambaBlock(charToKind(c), embedDim, nHeads, ffMult, useSession,
                                      mambaRealDState, mambaRealNGroups, mambaRealConvKernel,
                                      mambaRealExpand, quantResident, initializeWeights)

# ═══════════════════════════════════════════════════════════════
# FORWARD / BACKWARD theo block. R/P được xử lý bởi block-kernel entrypoints
# trong hybrid_block_kernel.nim; legacy M/A/S giữ đường hiện có bên dưới.
# ═══════════════════════════════════════════════════════════════

proc checkGBlockScale*(blockIndex: int; x, output: Tensor;
                       cache: HybridBlockKernelCache) =
  ## Guard G: Linear++ weighted ≤0.5×, gate mean ≥0.75, VÀ nhánh Mamba
  ## rms(mambaY)/rms(x) ≤ G_MAX_MAMBA_RMS_RATIO. cache.mambaY phải được
  ## forwardPBlockKernel điền (trước residual).
  doAssert x.shape == output.shape and x.data.len == cache.gate.len
  doAssert cache.mambaY.data.len == x.data.len,
    "checkGBlockScale: cache.mambaY missing/wrong size (block kernel must expose mambaY)"
  doAssert cache.linearPlusOut.data.len == x.data.len,
    "checkGBlockScale: cache.linearPlusOut missing/wrong size"
  var inputSq, sideSq, mambaSq, outputSq, gateSum = 0.0
  var inputMax, sideMax, mambaMax, outputMax = 0.0
  for i in 0 ..< x.data.len:
    let gate = cache.gate[i]
    let input = x.data[i]
    let mamba = cache.mambaY.data[i]
    let side = cache.linearPlusOut.data[i]
    let outputValue = output.data[i]
    if gate.classify in {fcNan, fcInf, fcNegInf} or
       input.classify in {fcNan, fcInf, fcNegInf} or
       mamba.classify in {fcNan, fcInf, fcNegInf} or
       side.classify in {fcNan, fcInf, fcNegInf} or
       outputValue.classify in {fcNan, fcInf, fcNegInf}:
      raise newException(CatchableError, "G block produced non-finite branch/gate values")
    if gate < 0'f32 or gate > 1'f32:
      raise newException(CatchableError, "G block sigmoid gate escaped [0,1]")
    inputSq += float64(input) * float64(input)
    mambaSq += float64(mamba) * float64(mamba)
    let weightedSide = (1'f32 - gate) * side
    sideSq += float64(weightedSide) * float64(weightedSide)
    outputSq += float64(outputValue) * float64(outputValue)
    inputMax = max(inputMax, abs(float64(input)))
    mambaMax = max(mambaMax, abs(float64(mamba)))
    sideMax = max(sideMax, abs(float64(weightedSide)))
    outputMax = max(outputMax, abs(float64(outputValue)))
    gateSum += float64(gate)
  let inputRms = sqrt(inputSq / float64(max(1, x.data.len)))
  let mambaRms = sqrt(mambaSq / float64(max(1, x.data.len)))
  let sideRms = sqrt(sideSq / float64(max(1, x.data.len)))
  let outputRms = sqrt(outputSq / float64(max(1, x.data.len)))
  let mambaRmsRatio = mambaRms / max(inputRms, 1e-6)
  let mambaMaxRatio = mambaMax / max(inputMax, 1e-6)
  let sideRmsRatio = sideRms / max(inputRms, 1e-6)
  let outputRmsRatio = outputRms / max(inputRms, 1e-6)
  let sideMaxRatio = sideMax / max(inputMax, 1e-6)
  let outputMaxRatio = outputMax / max(inputMax, 1e-6)
  if mambaRmsRatio > float64(G_MAX_MAMBA_RMS_RATIO):
    raise newException(CatchableError,
      &"G block {blockIndex}: mambaY RMS/input RMS={mambaRmsRatio:.3f} " &
       &"exceeds limit {G_MAX_MAMBA_RMS_RATIO:.2f} (SSM branch poisoning residual)")
  if sideRmsRatio > float64(G_MAX_SIDE_RMS_RATIO):
    raise newException(CatchableError,
      &"G block {blockIndex}: weighted Linear++ RMS/input RMS={sideRmsRatio:.3f} " &
       &"exceeds limit {G_MAX_SIDE_RMS_RATIO:.2f}")
  if sideMaxRatio > 8.0:
    raise newException(CatchableError,
      &"G block {blockIndex}: weighted Linear++ max/input max={sideMaxRatio:.3f} exceeds limit 8.0")
  if outputRmsRatio > float64(G_MAX_OUTPUT_RMS_RATIO) or outputMaxRatio > 16.0:
    raise newException(CatchableError,
      &"G block {blockIndex}: residual output/input RMS={outputRmsRatio:.3f}, " &
       &"max={outputMaxRatio:.3f}; limit is RMS 8.0x / max 16.0x")
  let gateMean = gateSum / float64(max(1, cache.gate.len))
  if gateMean < float64(G_MIN_GATE_MEAN):
    raise newException(CatchableError,
      &"G block {blockIndex}: mean gate {gateMean:.4f} below floor {G_MIN_GATE_MEAN:.2f}; " &
       "Linear++ branch is dominating the residual")
  if getEnv("BYBY_G_DIAGNOSTICS") == "1":
    stderr.writeLine &"[G-check] block={blockIndex} gateMean={gateMean:.4f} " &
      &"mambaRms/input={mambaRmsRatio:.3f} mambaMax/input={mambaMaxRatio:.3f} " &
      &"sideRms/input={sideRmsRatio:.3f} sideMax/input={sideMaxRatio:.3f} " &
      &"outputRms/input={outputRmsRatio:.3f} outputMax/input={outputMaxRatio:.3f}"

proc forwardBlockEx*(blk: JambaBlock, x: Tensor, ctx: Backend;
                     blockIndex = -1):
    tuple[y: Tensor, cache: HybridBlockKernelCache] =
  ## Như forwardBlock nhưng trả cache (có mambaY trước residual cho R/P/G).
  if blk.kind == bkMambaReal:
    return forwardRBlockKernel(blk.mambaReal, x, blk.rms1.gamma, blk.rms1.eps, ctx)
  if blk.kind == bkMambaRealPlusAttn:
    return forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                               blk.rms1.eps, ctx)
  if blk.kind == bkMambaRealGated:
    let fw = forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                 blk.rms1.eps, ctx, fuseGate = unsafeAddr blk.fuseGate)
    checkGBlockScale(blockIndex, x, fw.y, fw.cache)
    return fw
  raise newException(CatchableError,
    "forwardBlockEx: chỉ hỗ trợ R/P/G (cần HybridBlockKernelCache.mambaY)")

proc forwardBlock*(blk: JambaBlock, x: Tensor, ctx: Backend;
                   blockIndex = -1): Tensor =

  if blk.kind == bkMambaReal:
    return forwardRBlockKernel(blk.mambaReal, x, blk.rms1.gamma, blk.rms1.eps, ctx).y
  if blk.kind == bkMambaRealPlusAttn:
    return forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                               blk.rms1.eps, ctx).y
  if blk.kind == bkMambaRealGated:
    let fw = forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                 blk.rms1.eps, ctx, fuseGate = unsafeAddr blk.fuseGate)
    checkGBlockScale(blockIndex, x, fw.y, fw.cache)
    return fw.y
  let x1 = blk.ln1.forward(x, ctx)
  var combined: Tensor
  case blk.kind
  of bkMamba:
    combined = forwardMamba2(blk.mamba, x1, ctx)
  of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
    raise newException(CatchableError, "R/P block missed its block-kernel entrypoint")
  of bkAttn:
    combined = forwardLinearPlus(blk.attn, x1, ctx)
  of bkSpectron:
    let attnOut = forwardLinearPlus(blk.attn, x1, ctx)
    let mambaOut = forwardMamba2(blk.mamba, x1, ctx)
    let fuseIn = concatLastDim(attnOut, mambaOut)   # exported (*) từ test_nimformer
    let gateLogit = blk.fuseGate.forward(fuseIn, ctx)
    let gate = beSigmoid(ctx, gateLogit.data)
    combined = newTensor(attnOut.shape)
    for i in 0 ..< combined.data.len:
      combined.data[i] = gate[i] * attnOut.data[i] + (1'f32 - gate[i]) * mambaOut.data[i]
  let x2 = addT(ctx, x, combined)
  let x3 = blk.ln2.forward(x2, ctx)
  let ffOut = blk.ff.forward(x3, ctx)
  result = addT(ctx, x2, ffOut)

proc backwardBlock*(blk: JambaBlock, x: Tensor, dOut: Tensor, ctx: Backend,
                    prefix: string, grads: var Table[string, Tensor];
                    blockIndex = -1): Tensor =
  if blk.kind == bkMambaReal:
    let fw = forwardRBlockKernel(blk.mambaReal, x, blk.rms1.gamma, blk.rms1.eps, ctx)
    return backwardRBlockKernel(blk.mambaReal, x, blk.rms1.gamma, dOut, fw.cache,
                                blk.rms1.eps, ctx, prefix, grads)
  if blk.kind == bkMambaRealGated:
    let fw = forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                 blk.rms1.eps, ctx, fuseGate = unsafeAddr blk.fuseGate)
    checkGBlockScale(blockIndex, x, fw.y, fw.cache)
    return backwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                dOut, fw.cache, blk.rms1.eps, ctx, prefix, grads,
                                fuseGate = unsafeAddr blk.fuseGate)
  if blk.kind == bkMambaRealPlusAttn:
    let fw = forwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                 blk.rms1.eps, ctx)
    return backwardPBlockKernel(blk.mambaReal, blk.attn, x, blk.rms1.gamma,
                                dOut, fw.cache, blk.rms1.eps, ctx, prefix, grads)
  ## Trả về dX (để lan truyền tiếp lên block trước), đồng thời GHI gradient
  ## của mọi tham số của block này vào `grads` với key = prefix & ".<ten>".
  let x1 = blk.ln1.forward(x, ctx)
  var combined: Tensor
  var attnOut, mambaOut, fuseIn, gateLogit: Tensor
  var gate: seq[float32]
  case blk.kind
  of bkMamba:
    combined = forwardMamba2(blk.mamba, x1, ctx)
  of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
    raise newException(CatchableError, "R/P block missed its block-kernel entrypoint")
  of bkAttn:
    combined = forwardLinearPlus(blk.attn, x1, ctx)
  of bkSpectron:
    attnOut = forwardLinearPlus(blk.attn, x1, ctx)
    mambaOut = forwardMamba2(blk.mamba, x1, ctx)
    fuseIn = concatLastDim(attnOut, mambaOut)
    gateLogit = blk.fuseGate.forward(fuseIn, ctx)
    gate = beSigmoid(ctx, gateLogit.data)
    combined = newTensor(attnOut.shape)
    for i in 0 ..< combined.data.len:
      combined.data[i] = gate[i] * attnOut.data[i] + (1'f32 - gate[i]) * mambaOut.data[i]
  let x2 = addT(ctx, x, combined)

  let x3 = blk.ln2.forward(x2, ctx)
  let dFF = blk.ff.backward(x3, dOut, ctx)
  let dLN2 = blk.ln2.backward(x2, dFF.dX, ctx)
  grads[prefix & ".ln2.gamma"] = dLN2.dGamma
  grads[prefix & ".ln2.beta"] = dLN2.dBeta
  grads[prefix & ".ff.fc1.weight"] = dFF.dFc1W
  grads[prefix & ".ff.fc1.bias"] = dFF.dFc1B
  grads[prefix & ".ff.fc2.weight"] = dFF.dFc2W
  grads[prefix & ".ff.fc2.bias"] = dFF.dFc2B
  var dX2 = dLN2.dX
  for i in 0 ..< dX2.data.len: dX2.data[i] += dOut.data[i]   # residual x2 = x + combined

  var dX1Total: Tensor
  case blk.kind
  of bkMamba:
    let dM = backwardMamba2(blk.mamba, x1, dX2, ctx)
    grads[prefix & ".mamba.inProj.weight"] = dM.dInProjW
    grads[prefix & ".mamba.inProj.bias"] = dM.dInProjB
    grads[prefix & ".mamba.outProj.weight"] = dM.dOutProjW
    grads[prefix & ".mamba.outProj.bias"] = dM.dOutProjB
    dX1Total = dM.dX
  of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
    raise newException(CatchableError, "R/P block missed its block-kernel entrypoint")
  of bkAttn:
    let dA = backwardLinearPlus(blk.attn, x1, dX2, ctx)
    grads[prefix & ".attn.qkvProj.weight"] = dA.dQkvW
    grads[prefix & ".attn.qkvProj.bias"] = dA.dQkvB
    grads[prefix & ".attn.outProj.weight"] = dA.dOutW
    grads[prefix & ".attn.outProj.bias"] = dA.dOutB
    dX1Total = dA.dX
  of bkSpectron:
    var dGate = newSeq[float32](dX2.data.len)
    var dAttnOutPartial = newSeq[float32](dX2.data.len)
    var dMambaOutPartial = newSeq[float32](dX2.data.len)
    for i in 0 ..< dX2.data.len:
      dGate[i] = dX2.data[i] * (attnOut.data[i] - mambaOut.data[i])
      dAttnOutPartial[i] = dX2.data[i] * gate[i]
      dMambaOutPartial[i] = dX2.data[i] * (1'f32 - gate[i])
    var dGateLogit = newTensor(gateLogit.shape)
    for i in 0 ..< dGateLogit.data.len:
      dGateLogit.data[i] = dGate[i] * gate[i] * (1'f32 - gate[i])
    let dFuse = blk.fuseGate.backward(fuseIn, dGateLogit, ctx)
    grads[prefix & ".fuseGate.weight"] = dFuse.dW
    grads[prefix & ".fuseGate.bias"] = dFuse.dB
    let (dAttnOutFuse, dMambaOutFuse) = splitLastDim(dFuse.dX, x.shape[2])
    var dAttnOutTotal = newTensor(attnOut.shape)
    var dMambaOutTotal = newTensor(mambaOut.shape)
    for i in 0 ..< dAttnOutTotal.data.len:
      dAttnOutTotal.data[i] = dAttnOutPartial[i] + dAttnOutFuse.data[i]
      dMambaOutTotal.data[i] = dMambaOutPartial[i] + dMambaOutFuse.data[i]
    let dAttn = backwardLinearPlus(blk.attn, x1, dAttnOutTotal, ctx)
    let dMamba = backwardMamba2(blk.mamba, x1, dMambaOutTotal, ctx)
    grads[prefix & ".attn.qkvProj.weight"] = dAttn.dQkvW
    grads[prefix & ".attn.qkvProj.bias"] = dAttn.dQkvB
    grads[prefix & ".attn.outProj.weight"] = dAttn.dOutW
    grads[prefix & ".attn.outProj.bias"] = dAttn.dOutB
    grads[prefix & ".mamba.inProj.weight"] = dMamba.dInProjW
    grads[prefix & ".mamba.inProj.bias"] = dMamba.dInProjB
    grads[prefix & ".mamba.outProj.weight"] = dMamba.dOutProjW
    grads[prefix & ".mamba.outProj.bias"] = dMamba.dOutProjB
    dX1Total = newTensor(x1.shape)
    for i in 0 ..< dX1Total.data.len:
      dX1Total.data[i] = dAttn.dX.data[i] + dMamba.dX.data[i]

  let dLN1 = blk.ln1.backward(x, dX1Total, ctx)
  grads[prefix & ".ln1.gamma"] = dLN1.dGamma
  grads[prefix & ".ln1.beta"] = dLN1.dBeta
  result = dLN1.dX
  for i in 0 ..< result.data.len: result.data[i] += dX2.data[i]   # residual goc

proc forwardBatch*(m: NimformerModelJamba, idsBatch: seq[seq[int]], ctx: Backend): Tensor =
  var x = m.embed.lookupBatch(idsBatch, ctx)
  for i, blk in m.blocks:
    x = forwardBlock(blk, x, ctx, blockIndex = i)
  # SUA (rms-fin): norm_f (RMSNorm) TRƯỚC lm_head, đúng base HF. Trước đây
  # gọi thẳng outProj lên hidden chưa norm -> sai scale tích lũy 64 block.
  let xNorm = m.rmsF.forward(x, ctx)
  result = m.outProj.forward(xNorm, ctx)

# ═══════════════════════════════════════════════════════════════
# forwardBatchSession: BẢN INFERENCE-ONLY, dùng Session API TOÀN BỘ khi
# backend hỗ trợ (Metal/CUDA/OpenCL), không có backward (infer_lora.nim
# không cần train qua đường này). KHÔNG dùng cho fine-tune (finetune_lora.nim
# vẫn dùng forwardBlock/forwardBatch + backwardBlock ở trên, không đổi).
#   - Block 'M' (Mamba2SSD): forwardMamba2 (test_nimformer.nim) ĐÃ TỰ dùng
#     Session API sẵn khi model.useSession=true - không cần viết lại gì,
#     gọi y hệt forwardBlock ở trên.
#   - Block 'A'/'S' (LinearPlusAttention): forwardLinearPlus gốc KHÔNG dùng
#     session (chỉ có 2 phép Linear qua ctx, còn phần attention cốt lõi
#     luôn chạy CPU - xem comment "đường GPU-session có thể thêm sau theo
#     đúng khuôn Mamba2SSD" ngay tại chỗ khai báo LinearPlusAttention trong
#     test_nimformer.nim). forwardBatchSession dùng forwardLinearPlusSession
#     (MỚI, cũng trong test_nimformer.nim) thay vào - cùng công thức toán,
#     có CPU fallback y hệt nếu session lỗi, chỉ khác tốc độ.
# ═══════════════════════════════════════════════════════════════
proc forwardBlockSession(blk: JambaBlock, x: Tensor, ctx: Backend;
                         blockIndex = -1): Tensor =
  # R/P inference batch path shares the same end-to-end block entrypoint as
  # training. Keep the stateful one-token decode path separate: this sequence
  # kernel starts from zero state and does not export persistent decode state.
  if blk.kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}:
    return forwardBlock(blk, x, ctx, blockIndex = blockIndex)

  # Legacy session path for M/A/S blocks.
  let x1 = blk.ln1.forward(x, ctx)
  var combined: Tensor
  case blk.kind
  of bkMamba:
    combined = forwardMamba2(blk.mamba, x1, ctx)                 # da session-capable san
  of bkMambaReal:
    raise newException(CatchableError, "R block missed the inference block-kernel entrypoint")
  of bkMambaRealPlusAttn, bkMambaRealGated:
    raise newException(CatchableError, "P block missed the inference block-kernel entrypoint")
  of bkAttn:
    combined = forwardLinearPlusSession(blk.attn, x1, ctx)        # MOI - session-based
  of bkSpectron:
    let attnOut = forwardLinearPlusSession(blk.attn, x1, ctx)     # MOI - session-based
    let mambaOut = forwardMamba2(blk.mamba, x1, ctx)              # da session-capable san
    let fuseIn = concatLastDim(attnOut, mambaOut)
    let gateLogit = blk.fuseGate.forward(fuseIn, ctx)
    let gate = beSigmoid(ctx, gateLogit.data)
    combined = newTensor(attnOut.shape)
    for i in 0 ..< combined.data.len:
      combined.data[i] = gate[i] * attnOut.data[i] + (1'f32 - gate[i]) * mambaOut.data[i]
  let x2 = addT(ctx, x, combined)
  let x3 = blk.ln2.forward(x2, ctx)
  let ffOut = blk.ff.forward(x3, ctx)
  result = addT(ctx, x2, ffOut)

proc forwardBatchSession*(m: NimformerModelJamba, idsBatch: seq[seq[int]], ctx: Backend): Tensor =
  ## Dùng cho infer_lora.nim (generation loop) thay vì forwardBatch - CÙNG
  ## công thức toán/kết quả, khác cách thực thi (Session API resident GPU
  ## thay vì mỗi op tự upload/tải xuống riêng lẻ) khi backend hỗ trợ. Nếu
  ## backend không hỗ trợ session hoặc session fail giữa chừng, tự rơi về
  ## CPU đúng như cũ (xem forwardMamba2/forwardLinearPlusSession) - KHÔNG
  ## BAO GIỜ sinh token khác với forwardBatch chỉ vì lỗi session.
  var x = m.embed.lookupBatch(idsBatch, ctx)
  for i, blk in m.blocks:
    x = forwardBlockSession(blk, x, ctx, blockIndex = i)
  # SUA (rms-fin): norm_f trước lm_head cho khớp base + đường train.
  let xNorm = m.rmsF.forward(x, ctx)
  result = m.outProj.forward(xNorm, ctx)

# ═══════════════════════════════════════════════════════════════
# STATE INCREMENTAL (MOI) cho CA MODEL - O(1)/token thay vi forwardBatchSession
# o tren chay lai TOAN BO chuoi tu dau moi token (O(T), cang sinh dai cang
# cham - da xac nhan qua log thuc te ~136s/token). Dung state incremental
# tung block Mamba2Real (xem Mamba2RealState/forwardStepMamba2Real trong
# mamba2_real.nim, doi chieu DUNG cong thuc HF that
# transformers/models/mamba2/modeling_mamba2.py).
#
# GIOI HAN RO RANG (KHONG lam ngo): CHI dung duoc khi pattern model TOAN BO
# la 'R' (Mamba2Real) - khong co block 'A'/'M'/'S' nao ca. Ly do: block
# Attention ('A') can KV-cache rieng (giu K/V cua tung token cu) de attend
# dung qua khu - neu chi forward 1 token don le KHONG co lich su, ket qua
# SAI (khong phai chi cham), khac han Mamba2Real (state that su du de tai
# tao dung ket qua). Chua viet KV-cache cho block 'A' o day - neu pattern
# co block khac 'R', raise loi RO RANG thay vi am tham chay sai.
# ═══════════════════════════════════════════════════════════════

type
  JambaGenState* = object
    ## 1 phan tu/block - CHI co gia tri that (Mamba2RealState) cho block
    ## 'R', block khac (neu co) khong duoc dung qua duong nay (xem raise o
    ## forwardStepJamba).
    mambaStates*: seq[Mamba2RealState]

proc isPureMambaRealPattern*(m: NimformerModelJamba): bool =
  ## Kiem tra TRUOC (khong dung GPU) xem model co THUAN 'R' hay khong -
  ## dieu kien BAT BUOC de dung forwardStepJamba an toan. Goi truoc luc
  ## quyet dinh dung duong incremental hay duong forwardBatchSession cu.
  for blk in m.blocks:
    if blk.kind != bkMambaReal: return false
  return true

type
  HybridGenState* = object
    ## Incremental hybrid: R dung Mamba2RealState; A dung Linear++ state rieng.
    mambaStates*: seq[Mamba2RealState]
    attnStates*: seq[LinearPlusInferState]
    hidden*: int

proc newHybridGenState*(m: NimformerModelJamba): HybridGenState =
  result.hidden = m.embedDim
  result.mambaStates = newSeq[Mamba2RealState](m.blocks.len)
  result.attnStates = newSeq[LinearPlusInferState](m.blocks.len)
  for i, blk in m.blocks:
    if blk.kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}:
      result.mambaStates[i] = newMamba2RealState(blk.mambaReal)
    if blk.kind in {bkAttn, bkMambaRealPlusAttn, bkMambaRealGated}:
      result.attnStates[i] = newLinearPlusInferState(m.embedDim, blk.attn.nHeads)

proc newJambaGenState*(m: NimformerModelJamba): JambaGenState =
  ## RAISE ro rang neu pattern KHONG thuan 'R' - khong am tham tao state
  ## roi de forwardStepJamba tu phat hien sau (fail som, thong bao ro).
  if not isPureMambaRealPattern(m):
    raise newException(ValueError,
      "[JambaGenState] Model co block KHAC 'R' (Attention/Mamba2SSD/Spectron) - " &
      "state incremental CHUA ho tro cho cac block do (can KV-cache rieng, " &
      "chua viet). Dung forwardBatchSession (cham hon nhung DUNG) cho pattern nay.")
  result.mambaStates = newSeq[Mamba2RealState](m.blocks.len)
  for i, blk in m.blocks:
    result.mambaStates[i] = newMamba2RealState(blk.mambaReal)

proc forwardStepJamba*(m: NimformerModelJamba, tokenId: int, state: var JambaGenState,
                        ctx: Backend): Tensor =
  ## DUNG 1 token moi, O(1) (khong phu thuoc do dai chuoi da sinh) - THAY
  ## THE forwardBatchSession(m, @[ids], ctx) trong vong lap generate() khi
  ## pattern thuan 'R' (xem isPureMambaRealPattern). Tra ve logits [1,1,vocab].
  assert m.blocks.len == state.mambaStates.len,
    "[forwardStepJamba] state khong khop so block model (JambaGenState cu?)"
  var x = m.embed.lookupBatch(@[@[tokenId]], ctx)   # [1,1,hiddenSize]
  for i, blk in m.blocks:
    doAssert blk.kind == bkMambaReal,
      "[forwardStepJamba] gap block khong phai 'R' o vi tri " & $i &
      " - khong duoc goi ham nay cho pattern nay (xem isPureMambaRealPattern)."
    let x1 = blk.rms1.forward(x, ctx)   # SUA (rms-fin): RMSNorm thay ln1
    let combined = forwardStepMamba2Real(blk.mambaReal, x1, state.mambaStates[i], ctx)
    x = addT(ctx, x, combined)   # bkMambaReal: KHONG qua ln2/ff, giong het forwardBlock
  # SUA (rms-fin): norm_f trước lm_head.
  let xNorm = m.rmsF.forward(x, ctx)
  result = m.outProj.forward(xNorm, ctx)

proc uploadResidentJamba*(m: var NimformerModelJamba, ctx: Backend,
                           mergedQuantNames: HashSet[string] = initHashSet[string]()) =
  ## THÊM (theo mo hinh uploadAllResidentWeights trong generate.nim/
  ## nim_inference): upload TOAN BO weight len GPU MOT LAN DUY NHAT truoc
  ## khi vao vong lap generate. mamba2_real.nim (inProj/outProj) va
  ## test_nimformer.nim (qkvProj/outProj cua LinearPlusAttention) DA GOI
  ## `.forward()` cua chinh Linear type dung chung voi generate.nim (xem
  ## nimformer.nim) - Linear.forward() DA TU KIEM TRA residentReady/
  ## residentFp32Ready va di duong nhanh (chi upload activation nho moi
  ## buoc, khong dequant/transpose lai weight) - nen KHONG can sua gi trong
  ## mamba2_real.nim/test_nimformer.nim ca, chi can goi uploadResident() 1
  ## lan o day cho tung Linear la moi forward() sau do TU DONG nhanh hon.
  ##
  ## BAT BUOC goi ham nay SAU KHI weight da o dang fp32 DAY DU trong
  ## `.weight.data` (vd sau mergeIn() trong infer_lora.nim - luc dequant-
  ## resident cua inProj/outProj tu .nimq da duoc dien vao t[].data va LoRA
  ## da duoc cong vao permanent cho ca phien generate) - goi som hon (luc
  ## `.weight.data` con rong do quant-resident) se KHONG co gi de upload
  ## (uploadResident tu bo qua neu `l.weight.data.len == 0`, xem nimformer.nim).
  ##
  ## THEM (tiet kiem RAM, sua tiep theo sau bug useQuant o lora.nim):
  ## mergedQuantNames = tap ten tham so ma CALLER (infer_lora.generate) biet
  ## chac la vua duoc mergeIn() TU dequant+cong LoRA rieng cho phien nay
  ## (thuc te la lora.quantBase.keys) - danh dau "ban fp32 nay la CUA
  ## RIENG minh, khong alias/chia se voi bat ky Tensor nao khac". Voi DUNG
  ## nhung ten do, giai phong `.weight.data` NGAY sau khi uploadResident()
  ## xong - neu khong, tham so vua merge se ton tai 2 ban fp32 CUNG LUC (CPU
  ## + GPU-resident); tren GPU tich hop/unified memory (Intel Iris) ca 2 ban
  ## DEU an chung RAM he thong = gap doi RAM that su can, day la nguyen nhan
  ## bi kill (OOM) ngay sau dong log "matmul_q4 dang active". KHONG dung mac
  ## dinh rong (tap rong = khong free gi ca, y het hanh vi CU) de moi caller
  ## khac (neu co, vd test) khong bi anh huong neu khong chu dong truyen vao.
  for i in 0 ..< m.blocks.len:
    let p = &"block{i}"
    case m.blocks[i].kind
    of bkMambaReal:
      uploadResident(m.blocks[i].mambaReal.inProj, ctx)
      if (p & ".mambaReal.inProj.weight") in mergedQuantNames:
        m.blocks[i].mambaReal.inProj.weight.data = @[]
      uploadResident(m.blocks[i].mambaReal.outProj, ctx)
      if (p & ".mambaReal.outProj.weight") in mergedQuantNames:
        m.blocks[i].mambaReal.outProj.weight.data = @[]
    of bkMambaRealPlusAttn, bkMambaRealGated:
      uploadResident(m.blocks[i].mambaReal.inProj, ctx)
      if (p & ".mambaReal.inProj.weight") in mergedQuantNames:
        m.blocks[i].mambaReal.inProj.weight.data = @[]
      uploadResident(m.blocks[i].mambaReal.outProj, ctx)
      if (p & ".mambaReal.outProj.weight") in mergedQuantNames:
        m.blocks[i].mambaReal.outProj.weight.data = @[]
      uploadResident(m.blocks[i].attn.qkvProj, ctx)
      if (p & ".attn.qkvProj.weight") in mergedQuantNames:
        m.blocks[i].attn.qkvProj.weight.data = @[]
      uploadResident(m.blocks[i].attn.outProj, ctx)
      if (p & ".attn.outProj.weight") in mergedQuantNames:
        m.blocks[i].attn.outProj.weight.data = @[]
    of bkMamba:
      uploadResident(m.blocks[i].mamba.inProj, ctx)
      if (p & ".mamba.inProj.weight") in mergedQuantNames:
        m.blocks[i].mamba.inProj.weight.data = @[]
      uploadResident(m.blocks[i].mamba.outProj, ctx)
      if (p & ".mamba.outProj.weight") in mergedQuantNames:
        m.blocks[i].mamba.outProj.weight.data = @[]
    of bkAttn:
      uploadResident(m.blocks[i].attn.qkvProj, ctx)
      if (p & ".attn.qkvProj.weight") in mergedQuantNames:
        m.blocks[i].attn.qkvProj.weight.data = @[]
      uploadResident(m.blocks[i].attn.outProj, ctx)
      if (p & ".attn.outProj.weight") in mergedQuantNames:
        m.blocks[i].attn.outProj.weight.data = @[]
    of bkSpectron:
      uploadResident(m.blocks[i].mamba.inProj, ctx)
      if (p & ".mamba.inProj.weight") in mergedQuantNames:
        m.blocks[i].mamba.inProj.weight.data = @[]
      uploadResident(m.blocks[i].mamba.outProj, ctx)
      if (p & ".mamba.outProj.weight") in mergedQuantNames:
        m.blocks[i].mamba.outProj.weight.data = @[]
      uploadResident(m.blocks[i].attn.qkvProj, ctx)
      if (p & ".attn.qkvProj.weight") in mergedQuantNames:
        m.blocks[i].attn.qkvProj.weight.data = @[]
      uploadResident(m.blocks[i].attn.outProj, ctx)
      if (p & ".attn.outProj.weight") in mergedQuantNames:
        m.blocks[i].attn.outProj.weight.data = @[]
      uploadResident(m.blocks[i].fuseGate, ctx)
      if (p & ".fuseGate.weight") in mergedQuantNames:
        m.blocks[i].fuseGate.weight.data = @[]
  uploadResident(m.outProj, ctx)
  if "outProj.weight" in mergedQuantNames:
    m.outProj.weight.data = @[]

# ═══════════════════════════════════════════════════════════════
# PARAM REGISTRY - generic, không cần sửa tay mỗi khi thêm loại block.
# ═══════════════════════════════════════════════════════════════

proc paramList*(m: var NimformerModelJamba): seq[(string, ptr Tensor)] =
  result = @[("embed.weight", addr m.embed.weight), ("outProj.weight", addr m.outProj.weight),
             ("outProj.bias", addr m.outProj.bias),
             ("rmsF.gamma", addr m.rmsF.gamma)]   # norm_f - checkpoint .bin luu/nap duoc
  for i in 0 ..< m.blocks.len:
    let p = &"block{i}"
    result.add((p & ".ln1.gamma", addr m.blocks[i].ln1.gamma))
    result.add((p & ".ln1.beta", addr m.blocks[i].ln1.beta))
    result.add((p & ".ln2.gamma", addr m.blocks[i].ln2.gamma))
    result.add((p & ".ln2.beta", addr m.blocks[i].ln2.beta))
    if m.blocks[i].kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}:
      # norm THẬT dùng trên đường forward/backward của 'R'/'P' (rms-fin)
      result.add((p & ".rms1.gamma", addr m.blocks[i].rms1.gamma))
    result.add((p & ".ff.fc1.weight", addr m.blocks[i].ff.fc1.weight))
    result.add((p & ".ff.fc1.bias", addr m.blocks[i].ff.fc1.bias))
    result.add((p & ".ff.fc2.weight", addr m.blocks[i].ff.fc2.weight))
    result.add((p & ".ff.fc2.bias", addr m.blocks[i].ff.fc2.bias))
    case m.blocks[i].kind
    of bkMamba:
      result.add((p & ".mamba.inProj.weight", addr m.blocks[i].mamba.inProj.weight))
      result.add((p & ".mamba.inProj.bias", addr m.blocks[i].mamba.inProj.bias))
      result.add((p & ".mamba.outProj.weight", addr m.blocks[i].mamba.outProj.weight))
      result.add((p & ".mamba.outProj.bias", addr m.blocks[i].mamba.outProj.bias))
    of bkMambaReal:
      for (nm, pt) in paramListMamba2Real(m.blocks[i].mambaReal, p & ".mambaReal"):
        result.add((nm, pt))
    of bkMambaRealPlusAttn, bkMambaRealGated:
      for (nm, pt) in paramListMamba2Real(m.blocks[i].mambaReal, p & ".mambaReal"):
        result.add((nm, pt))
      # FIX: attnLn1 ĐÃ BỎ - dùng chung ln1 với Mamba
      result.add((p & ".attn.qkvProj.weight", addr m.blocks[i].attn.qkvProj.weight))
      result.add((p & ".attn.qkvProj.bias", addr m.blocks[i].attn.qkvProj.bias))
      result.add((p & ".attn.outProj.weight", addr m.blocks[i].attn.outProj.weight))
      result.add((p & ".attn.outProj.bias", addr m.blocks[i].attn.outProj.bias))
      if m.blocks[i].kind == bkMambaRealGated:
        result.add((p & ".fuseGate.weight", addr m.blocks[i].fuseGate.weight))
        result.add((p & ".fuseGate.bias", addr m.blocks[i].fuseGate.bias))
    of bkAttn:
      result.add((p & ".attn.qkvProj.weight", addr m.blocks[i].attn.qkvProj.weight))
      result.add((p & ".attn.qkvProj.bias", addr m.blocks[i].attn.qkvProj.bias))
      result.add((p & ".attn.outProj.weight", addr m.blocks[i].attn.outProj.weight))
      result.add((p & ".attn.outProj.bias", addr m.blocks[i].attn.outProj.bias))
    of bkSpectron:
      result.add((p & ".attn.qkvProj.weight", addr m.blocks[i].attn.qkvProj.weight))
      result.add((p & ".attn.qkvProj.bias", addr m.blocks[i].attn.qkvProj.bias))
      result.add((p & ".attn.outProj.weight", addr m.blocks[i].attn.outProj.weight))
      result.add((p & ".attn.outProj.bias", addr m.blocks[i].attn.outProj.bias))
      result.add((p & ".mamba.inProj.weight", addr m.blocks[i].mamba.inProj.weight))
      result.add((p & ".mamba.inProj.bias", addr m.blocks[i].mamba.inProj.bias))
      result.add((p & ".mamba.outProj.weight", addr m.blocks[i].mamba.outProj.weight))
      result.add((p & ".mamba.outProj.bias", addr m.blocks[i].mamba.outProj.bias))
      result.add((p & ".fuseGate.weight", addr m.blocks[i].fuseGate.weight))
      result.add((p & ".fuseGate.bias", addr m.blocks[i].fuseGate.bias))

type
  JambaAdamState* = object
    states*: Table[string, ApfAdamState]

proc paramQuantFlags*(m: var NimformerModelJamba): Table[string, ptr bool] =
  ## BUG FIX: dong bo voi paramList o tren, nhung tra ve con tro TOI THANG
  ## truong `useQuant` cua tung Linear (chi cho ten ".weight" nao thuc su
  ## backed boi 1 Linear - bo qua embed.weight vi Embedding khong co
  ## useQuant, bo qua .bias vi LoRA khong ap dung len bias).
  ##
  ## LY DO CAN: mergeIn/mergeInOne (lora.nim) chi cam duoc `ptr Tensor` (tro
  ## thang vao truong `.weight` NAM BEN TRONG Linear) tu paramList, KHONG
  ## cam duoc ca object Linear chua no - nen truoc ban vá nay, sau khi
  ## mergeIn ghi `t[].data = merged.data` (base + delta LoRA, fp32 THAT),
  ## Linear.useQuant VAN con true. Ke tiep, uploadResident()/Linear.forward()
  ## (nimformer.nim) kiem tra `l.useQuant` TRUOC, thay true nen DEQUANT LAI
  ## tu `l.weightQ` (ban nen GOC, CHUA co LoRA) - bo qua hoan toan t[].data
  ## vua merge. Khong crash, khong warning, chi la LoRA bi "ngam" mat tren
  ## moi tham so quant-resident (dung loai checkpoint dang gap: block khac
  ## 'R', log in "mergeIn + uploadResident + forwardBatchSession").
  ##
  ## mergeIn/mergeInOne dung bang nay de dat useQuant=false NGAY khi ghi
  ## merged.data (bao Linear.forward()/uploadResident() "day la weight fp32
  ## THAT, dung thang, dung dequant lai"); restoreBase/restoreBaseOne dat
  ## lai true khi xoa t[].data ve rong (de lan mergeIn ke tiep - hoac bat ky
  ## uploadResident nao khac chay giua chung - dequant dung tro lai tu
  ## weightQ nhu thiet ke ban dau).
  result = initTable[string, ptr bool]()
  result["outProj.weight"] = addr m.outProj.useQuant
  for i in 0 ..< m.blocks.len:
    let p = &"block{i}"
    result[p & ".ff.fc1.weight"] = addr m.blocks[i].ff.fc1.useQuant
    result[p & ".ff.fc2.weight"] = addr m.blocks[i].ff.fc2.useQuant
    case m.blocks[i].kind
    of bkMamba:
      result[p & ".mamba.inProj.weight"] = addr m.blocks[i].mamba.inProj.useQuant
      result[p & ".mamba.outProj.weight"] = addr m.blocks[i].mamba.outProj.useQuant
    of bkMambaReal:
      result[p & ".mambaReal.inProj.weight"] = addr m.blocks[i].mambaReal.inProj.useQuant
      result[p & ".mambaReal.outProj.weight"] = addr m.blocks[i].mambaReal.outProj.useQuant
    of bkMambaRealPlusAttn, bkMambaRealGated:
      result[p & ".mambaReal.inProj.weight"] = addr m.blocks[i].mambaReal.inProj.useQuant
      result[p & ".mambaReal.outProj.weight"] = addr m.blocks[i].mambaReal.outProj.useQuant
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj.useQuant
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj.useQuant
      if m.blocks[i].kind == bkMambaRealGated:
        result[p & ".fuseGate.weight"] = addr m.blocks[i].fuseGate.useQuant
    of bkAttn:
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj.useQuant
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj.useQuant
    of bkSpectron:
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj.useQuant
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj.useQuant
      result[p & ".mamba.inProj.weight"] = addr m.blocks[i].mamba.inProj.useQuant
      result[p & ".mamba.outProj.weight"] = addr m.blocks[i].mamba.outProj.useQuant
      result[p & ".fuseGate.weight"] = addr m.blocks[i].fuseGate.useQuant

proc paramLinears*(m: var NimformerModelJamba): Table[string, ptr Linear] =
  ## Trả về bảng name → ptr Linear (chỉ các Linear có .weight, đồng bộ
  ## paramQuantFlags / paramList). Dùng để bind quantBase → weightQ/useQuant
  ## (setQuantWeight) trước uploadResident / mergeInUploadStreamingJamba.
  ## Thiếu bước này là nguyên nhân IndexDefect "container is empty" trên
  ## đường quantResident + NIMBIN4 (weight.data bị @[] lúc newJambaBlock,
  ## nhưng weightQ/useQuant chưa được nối từ lora.quantBase).
  result = initTable[string, ptr Linear]()
  result["outProj.weight"] = addr m.outProj
  for i in 0 ..< m.blocks.len:
    let p = &"block{i}"
    result[p & ".ff.fc1.weight"] = addr m.blocks[i].ff.fc1
    result[p & ".ff.fc2.weight"] = addr m.blocks[i].ff.fc2
    case m.blocks[i].kind
    of bkMamba:
      result[p & ".mamba.inProj.weight"] = addr m.blocks[i].mamba.inProj
      result[p & ".mamba.outProj.weight"] = addr m.blocks[i].mamba.outProj
    of bkMambaReal:
      result[p & ".mambaReal.inProj.weight"] = addr m.blocks[i].mambaReal.inProj
      result[p & ".mambaReal.outProj.weight"] = addr m.blocks[i].mambaReal.outProj
    of bkMambaRealPlusAttn, bkMambaRealGated:
      result[p & ".mambaReal.inProj.weight"] = addr m.blocks[i].mambaReal.inProj
      result[p & ".mambaReal.outProj.weight"] = addr m.blocks[i].mambaReal.outProj
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj
      if m.blocks[i].kind == bkMambaRealGated:
        result[p & ".fuseGate.weight"] = addr m.blocks[i].fuseGate
    of bkAttn:
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj
    of bkSpectron:
      result[p & ".attn.qkvProj.weight"] = addr m.blocks[i].attn.qkvProj
      result[p & ".attn.outProj.weight"] = addr m.blocks[i].attn.outProj
      result[p & ".mamba.inProj.weight"] = addr m.blocks[i].mamba.inProj
      result[p & ".mamba.outProj.weight"] = addr m.blocks[i].mamba.outProj
      result[p & ".fuseGate.weight"] = addr m.blocks[i].fuseGate

proc newJambaAdamState*(m: var NimformerModelJamba): JambaAdamState =
  result.states = initTable[string, ApfAdamState]()
  for (name, t) in paramList(m):
    result.states[name] = newApfAdamState(t[].data.len)

proc computeLossAndGrads*(m: var NimformerModelJamba, idsBatch: seq[seq[int]],
                           targetsBatch: seq[seq[int]], ctx: Backend,
                           trainLmHead: bool = true, trainEmbed: bool = true):
                          tuple[loss: float32, grads: Table[string, Tensor]] =
  ## Tách riêng forward+backward (KHÔNG tự update Adam) để lora.nim có thể
  ## tự quyết update tham số nào (LoRA A/B) thay vì update thẳng base weight
  ## như trainStep dưới đây làm cho toàn model.
  var x = m.embed.lookupBatch(idsBatch, ctx)
  var xs: seq[Tensor] = @[x]
  for i, blk in m.blocks:
    x = forwardBlock(blk, x, ctx, blockIndex = i)
    xs.add(x)
  # SUA (rms-fin): norm_f TRƯỚC lm_head ở cả forward lẫn backward.
  let xNorm = m.rmsF.forward(x, ctx)
  let logits = m.outProj.forward(xNorm, ctx)
  let (loss, dLogits) = softmaxCrossEntropy(logits, targetsBatch, ctx)

  var grads = initTable[string, Tensor]()
  var dHeadInput: Tensor
  if trainLmHead:
    let dOutProj = m.outProj.backward(xNorm, dLogits, ctx)
    grads["outProj.weight"] = dOutProj.dW
    grads["outProj.bias"] = dOutProj.dB
    dHeadInput = dOutProj.dX
  else:
    dHeadInput = m.outProj.backwardInputOnly(xNorm, dLogits, ctx)
  let dNormF = m.rmsF.backward(x, dHeadInput, ctx)
  if trainLmHead: grads["rmsF.gamma"] = dNormF.dGamma
  var dX = dNormF.dX
  for i in countdown(m.blocks.len - 1, 0):
    dX = backwardBlock(m.blocks[i], xs[i], dX, ctx, &"block{i}", grads)
  if trainEmbed:
    grads["embed.weight"] = m.embed.backward(idsBatch, dX, ctx)
  result = (loss, grads)

proc trainStep*(m: var NimformerModelJamba, adam: var JambaAdamState,
                idsBatch: seq[seq[int]], targetsBatch: seq[seq[int]],
                ctx: Backend, lr: float32 = 3e-4'f32,
                requantizeEvery: int = 50): float32 =
  let (loss, grads) = computeLossAndGrads(m, idsBatch, targetsBatch, ctx)
  for (name, t) in paramList(m):
    if name in grads:
      var st = adam.states[name]
      discard apfAdamStepFused(ctx, t[], grads[name], st, lr, requantizeEvery = requantizeEvery)
      adam.states[name] = st
    else:
      stderr.writeLine("[CANH BAO] Thieu gradient cho tham so: " & name)
  result = loss

# ═══════════════════════════════════════════════════════════════
# CHECKPOINT: dùng thẳng .bin thô (rawbin.nim), KHÔNG dùng .nimq ở đây.
# .nimq (quant.nim) chỉ dành riêng cho nim_inference.nim/test_nimformer.nim -
# toàn bộ pipeline fine-tune (graft_attention/finetune_lora/infer_lora) đọc
# ghi checkpoint trực tiếp bằng .bin. arch + pattern + step được nhét gọn
# trong phần "meta" của file .bin, không cần sidecar .pattern.json nữa.
# ═══════════════════════════════════════════════════════════════

proc saveCheckpoint*(m: var NimformerModelJamba, path: string, step: int = 0) =
  var tensors: seq[RawBinTensor] = @[]
  for (name, t) in paramList(m):
    tensors.add(RawBinTensor(name: name, shape: t[].shape, data: t[].data))
  var meta = initTable[string, string]()
  meta["vocab"] = $m.vocab
  meta["embedDim"] = $m.embedDim
  meta["nHeads"] = $m.nHeads
  meta["ffMult"] = $m.ffMult
  meta["pattern"] = m.pattern
  meta["step"] = $step
  saveRawBin(path, meta, tensors)
  echo &"[jamba] Da luu checkpoint .bin -> {path} ({tensors.len} tensor, step={step})"

proc loadCheckpoint*(path: string, useSession: bool = true): tuple[model: NimformerModelJamba, step: int] =
  let (meta, tensors) = loadRawBin(path)
  template reqMeta(key: string): string =
    if key notin meta:
      raise newException(IOError, "Checkpoint .bin thieu meta bat buoc '" & key & "': " & path)
    meta[key]
  let vocab = parseInt(reqMeta("vocab"))
  let embedDim = parseInt(reqMeta("embedDim"))
  let nHeads = parseInt(reqMeta("nHeads"))
  let ffMult = parseInt(reqMeta("ffMult"))
  let pattern = reqMeta("pattern")
  let step = (if "step" in meta: parseInt(meta["step"]) else: 0)

  var m = newNimformerModelJamba(vocab, embedDim, nHeads, ffMult, pattern, useSession)
  var lookup = initTable[string, RawBinTensor]()
  for t in tensors:
    lookup[t.name] = t
  for (name, t) in paramList(m):
    if name in lookup:
      t[].data = lookup[name].data
    else:
      stderr.writeLine("[CANH BAO] Checkpoint .bin thieu tham so '" & name & "' - giu khoi tao random")
  result = (m, step)

when not defined(codestralInference):
  # ═══════════════════════════════════════════════════════════════
  # NẠP TRỌNG SỐ TỪ MODEL HF (đã export bằng pytorch_bin_loader.nim) VÀO
  # JAMBA. QUAN TRỌNG: Mamba-1B (HF, kiến trúc mamba_ssm gốc: A_log, D, x_proj,
  # dt_proj, conv1d, in_proj, out_proj...) KHÔNG cùng schema với Mamba2SSD ở
  # đây (SSD/quadratic-form tự viết, tham số hoá khác hẳn). KHÔNG có phép map
  # 1-1 tự động đúng nghĩa toán học giữa 2 kiến trúc này - copy thẳng theo tên
  # gần giống nhau (vd "embeddings.weight" -> "embed.weight") CHỈ hợp lý cho
  # embedding/lm_head (nếu cùng dim), còn phần thân SSD phải huấn luyện lại
  # từ đầu (init random, dùng embedding của HF làm điểm khởi đầu tốt hơn thôi).
  # Hàm dưới đây làm đúng phần "hợp lý tự động" đó, phần còn lại bạn tự thêm
  # theo manifest in ra từ pytorch_bin_loader.
  proc loadEmbeddingFromHF*(m: var NimformerModelJamba, hfBinPath: string,
                             hfEmbedKey: string = "backbone.embedding.weight") =
    ## Doc TRUC TIEP tu file .bin GOC cua HF (pytorch_model.bin/.safetensors,
    ## qua pytorch_bin_loader.loadAllHFTensors) - KHONG can xuat ra file trung
    ## gian nao truoc (khong .nimq, khong ban sao .bin nao khac).
    ## Luu y: "backbone.embedding.weight" (SO IT, khong phai "embeddings") la
    ## key that su cua mamba_ssm/MambaLMHeadModel (Mamba-Chat-2.8B dung kien
    ## truc nay) - kiem tra lai bang manifest neu ban doi model khac.
    let tensors = loadAllHFTensors(hfBinPath)
    for t in tensors:
      if t.name == hfEmbedKey:
        if t.data.len == m.embed.weight.data.len:
          m.embed.weight.data = t.data
          echo &"[jamba] Da nap embedding tu HF ({hfEmbedKey}), {t.data.len} gia tri"
        else:
          stderr.writeLine(&"[CANH BAO] Shape embedding HF ({t.data.len}) khong khop model Jamba ({m.embed.weight.data.len}) - bo qua")
        return
    stderr.writeLine(&"[CANH BAO] Khong thay key '{hfEmbedKey}' trong {hfBinPath}")

  # ═══════════════════════════════════════════════════════════════
  # NẠP TOÀN BỘ khối 'M' (Mamba-only) TRỰC TIẾP TỪ pytorch_model.bin GỐC, không
  # ghi ra bất kỳ file .bin/.nimq trung gian nào - tránh nhân đôi dung lượng
  # đĩa khi checkpoint HF gốc đã nặng vài GB (yêu cầu bắt buộc của project).
  # Chỉ map được các tên tự trùng/khai báo qua `nameMap`; tên nào không khớp
  # giữ nguyên random-init (in cảnh báo tổng hợp, không log từng cái để đỡ rối
  # - xem thêm cảnh báo "KHÔNG map 1-1 tự động" phía trên).
  # ═══════════════════════════════════════════════════════════════
  proc loadMambaOnlyFromHFBin*(hfBinPath: string, vocab, embedDim, nHeads, ffMult, nLayers: int,
                                nameMap: Table[string, string] = initTable[string, string](),
                                useSession: bool = true):
                               tuple[model: NimformerModelJamba, matched, total: int] =
    var pattern = ""
    for _ in 0 ..< nLayers: pattern.add('M')
    var m = newNimformerModelJamba(vocab, embedDim, nHeads, ffMult, pattern, useSession)

    let tensors = loadAllHFTensors(hfBinPath)   # CHI o RAM - khong ghi file nao ra dia
    var lookup = initTable[string, RawBinTensor]()
    for t in tensors:
      lookup[t.name] = t

    var matched = 0
    let plist = paramList(m)
    for (name, t) in plist:
      let hfName = (if name in nameMap: nameMap[name] else: name)
      if hfName in lookup and lookup[hfName].data.len == t[].data.len:
        t[].data = lookup[hfName].data
        inc matched
    echo &"[jamba] Da nap truc tiep tu {hfBinPath}: {matched}/{plist.len} tham so khop ten " &
         "(phan con lai la Mamba2SSD tu viet, KHONG map 1-1 tu dong duoc voi mamba_ssm goc " &
         "- can nameMap hoac train lai tu dau, xem ghi chu dau file)"
    result = (m, matched, plist.len)

# ═══════════════════════════════════════════════════════════════
# NẠP WEIGHT NORM PRETRAIN (preset rms-fin). BẮT BUỘC phải gọi sau khi nạp
# weight mixer - nếu không, rms1/rmsF giữ nguyên gamma=1 (identity), trong
# khi norm weight THẬT của Codestral KHÔNG phải toàn 1.0 -> lại lệch base.
# Map:
#   backbone.layers.{i}.norm.weight  ->  block{i}.rms1.gamma   (block 'R'/'P')
#   backbone.norm_f.weight           ->  rmsF.gamma
# ═══════════════════════════════════════════════════════════════
proc loadNormsFromHfTensors*(m: var NimformerModelJamba, tensors: seq[RawBinTensor]): int =
  ## `tensors` là danh sách tensor HF đã đọc (loadAllHFTensors từ safetensors,
  ## hoặc danh sách dequant từ .nimq - miễn cùng schema tên HF). Trả về số
  ## norm đã nạp. Kỳ vọng = số block 'R'/'P' + 1 (norm_f). In cảnh báo nếu
  ## thiếu - KHÔNG raise (để caller tự quyết), nhưng thiếu norm_f là lỗi nặng.
  var lookup = initTable[string, RawBinTensor]()
  for t in tensors: lookup[t.name] = t
  result = 0
  for i in 0 ..< m.blocks.len:
    if m.blocks[i].kind notin {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}: continue
    let key = "backbone.layers." & $i & ".norm.weight"
    if key in lookup:
      if lookup[key].data.len == m.blocks[i].rms1.gamma.data.len:
        m.blocks[i].rms1.gamma.data = lookup[key].data
        inc result
      else:
        stderr.writeLine(&"[CANH BAO] {key}: shape lech ({lookup[key].data.len} != " &
                         $m.blocks[i].rms1.gamma.data.len & ") - giu gamma=1")
    else:
      stderr.writeLine("[CANH BAO] Thieu " & key & " trong nguon weight - giu gamma=1")
  var foundF = false
  for key in ["backbone.norm_f.weight", "norm_f.weight"]:
    if key in lookup and lookup[key].data.len == m.rmsF.gamma.data.len:
      m.rmsF.gamma.data = lookup[key].data
      inc result
      foundF = true
      break
  if not foundF:
    stderr.writeLine("[CANH BAO] KHONG thay backbone.norm_f.weight - rmsF giu gamma=1, " &
                     "hidden vao lm_head van se lech base. Kiem tra lai nguon weight.")
  echo &"[jamba] rms-fin: da nap {result} norm weight pretrain (rms1 + norm_f)"

const TRAIN_NORM_PRESET* = "rms-fin-gn"
const TRAIN_MAMBA_NORM_IMPL* = "mamba-ssm-postgate-grouped-v1"
  ## Ten preset infer (lora.setGpuNormPreset) tuong ung DUNG voi forward cua trainer nay:
  ## RMSNorm gamma base (syncRmsFromBaseNorms) cho CA nhanh R lan sidecar P cua moi khoi
  ## (forwardBlock: x1 = rms1(x) dung chung), norm_f (rmsF) truoc lm_head, gated-RMSNorm
  ## theo 8 group (Mamba-SSM reference, group_size=8192/8). finetune_lora ghi no vao meta["normPreset"] cua checkpoint,
  ## infer_lora doc lai de dung CHINH preset do - hai ben khong the lech nhau.
  ## DOI cach forward chuan hoa => doi hang so nay (va setGpuNormPreset ben lora.nim).

proc syncRmsFromBaseNorms*(m: var NimformerModelJamba) =
  ## SUA (train/infer preset khop): loadNormsFromHfTensors khong bao gio duoc goi,
  ## nen rms1/rmsF luc train giu gamma=1 trong khi infer (lora.nim) dung
  ## ln1.gamma / normF.gamma = weight pretrain THAT (nap boi load_codestral_*).
  ## Goi ham nay SAU moi lan nap base: chep ln1.gamma -> rms1.gamma (R/P) va
  ## normF.gamma -> rmsF.gamma de train va infer dung CUNG gamma.
  var n = 0
  for i in 0 ..< m.blocks.len:
    if m.blocks[i].kind notin {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}: continue
    if m.blocks[i].ln1.gamma.data.len == m.blocks[i].rms1.gamma.data.len:
      m.blocks[i].rms1.gamma.data = m.blocks[i].ln1.gamma.data
      inc n
  if m.normF.gamma.data.len == m.rmsF.gamma.data.len:
    m.rmsF.gamma.data = m.normF.gamma.data
    inc n
  echo "[jamba] syncRmsFromBaseNorms: dong bo " & $n & " gamma (rms1 + rmsF) tu base"
