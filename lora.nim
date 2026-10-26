## lora.nim
## LoRA (low-rank adapter) cho NimformerModelJamba, + hàm "cấy ghép"
## (graft) thêm block LinearPlusAttention MỚI vào 1 checkpoint hiện tại chỉ
## toàn Mamba2SSD (pattern kiểu "MMMMMM...").
##
## Ý tưởng LoRA giống hệt script Python bạn gửi (LoRALinear: base đóng băng,
## chỉ học A,B hạng thấp): thay vì viết riêng 1 "LoRALinear" tách biệt khỏi
## Mamba2SSD/LinearPlusAttention (phải sửa lại toàn bộ forwardMamba2/
## forwardLinearPlus để nhận Linear thay thế được - động vào quá nhiều),
## ta dùng đúng 1 mẹo toán học:
##   W_eff = W_base + (alpha/r) * B @ A       (B:[outF,r], A:[r,inF])
## rồi GHI ĐÈ TẠM w_base.data = W_eff, chạy y nguyên forwardBlock/
## backwardBlock/computeLossAndGrads đã có (không sửa 1 dòng nào ở
## jamba_hybrid.nim), backward trả ra dW_eff cho tên tham số đó - dùng
## chain rule suy ngược ra dA, dB (base KHÔNG có gradient, không update),
## rồi PHỤC HỒI w_base.data về nguyên bản trước khi sang bước sau.

import std/[math, random, strformat, tables, strutils, times, os, monotimes]

proc monoSecs*(): float =
  ## Dong ho don dieu, KHONG chay khi may ngu (epochTime thi co).
  getMonoTime().ticks.float / 1e9

import nimformer
import backend
import jamba_hybrid   # HybridGenState, JambaGenState, isPureMambaRealPattern, addT, ...
import linear_plus_infer
import mamba2_real     # forwardStepMamba2Real
import mamba2_ssm_gpu
import mamba2_fused_step
import fp4_gemv_gpu    # prepare path: gFp4CacheGpu, fp4GemvGpu, Shared weight
import test_nimformer  # forwardLinearPlusSession (Attention GPU)
import quant           # QuantTensor, dequantizeTensor
import customfloat     # buildDecodeLUT, decodeArrayLUT
import session_vm      # bytecode emit / .bvm / sessionVmFor*
import jit_vm          # HotSpot-style: interpret -> profile -> Metal JIT cache
when not defined(noOneDispatch):
  import onedispatch   # mac dinh BAT (tat luc build: -d:noOneDispatch) : 1 compute dispatch / token cho ca stack R (xem ONE_DISPATCH_PLAN.md)

# Global JIT cache (infer + finetune cung dung)
var gJitCache* = initJitCache(defaultJitConfig())
var gMicroVmBackend* = gbAuto

# Decode profiling (opt-in via BYBY_DECODE_PROFILE=1). Accumulates wall-clock
# time per phase across all tokens so the caller can print a breakdown at the
# end of generation. Zero overhead when disabled (timers are skipped).
var gProfBegin*, gProfEncode*, gProfEnd*, gProfRead*: float64
var gProfTokens*: int
var gProfFirst*: string   ## token dau tach rieng (thuong cham hon han: pipeline/driver lan dau)
var gProfFirstMs*: float64
# BYBY_DECODE_SPLIT=1 (chan doan, CHAM HON): sessionEnd+sessionBegin giua cac phan
# cua 1 token (doan one-dispatch / block G / lm_head) de do thoi gian GPU tung phan.
var gDecSplitOn = getEnv("BYBY_DECODE_SPLIT", "0") != "0"
var gDecSplitActive = false
var gDecSplitMs = initOrderedTable[string, float64]()
var gDecSplitTokens = 0

proc decSplitMark(be: GpuBackend; label: string; t0: var float64) =
  ## Dong command buffer hien tai (cho GPU xong), cong thoi gian vao label, mo buffer moi.
  if not (gDecSplitOn and gDecSplitActive): return
  if not sessionEnd(be):
    raise newException(CatchableError, "BYBY_DECODE_SPLIT: sessionEnd that bai sau " & label &
      ": " & sessionGetLastError(be))
  let t1 = monoSecs()
  gDecSplitMs.mgetOrPut(label, 0.0) += (t1 - t0) * 1000.0
  if not sessionBegin(be):
    raise newException(CatchableError, "BYBY_DECODE_SPLIT: sessionBegin that bai sau " & label)
  t0 = monoSecs()

proc decodeSplitSummary*(): string =
  if gDecSplitTokens == 0 or gDecSplitMs.len == 0: return ""
  var tot = 0.0
  for _, v in gDecSplitMs: tot += v
  result = &"[infer] BYBY_DECODE_SPLIT over {gDecSplitTokens} tokens (bo token dau), ms/tok:"
  for k, v in gDecSplitMs:
    let per = v / float64(gDecSplitTokens)
    result.add &"\n[infer]   {k:<14} {per:>8.1f}ms {100.0 * v / max(tot, 1e-9):>5.1f}%"

proc decodeProfileSummary*(): string =
  ## Human-readable per-token breakdown of the decode hot path. Empty string
  ## when no tokens were profiled. Called by infer_lora.nim after generation.
  ## Trung binh KHONG tinh token dau (in rieng o dong thu 2).
  if gProfTokens <= 1: return gProfFirst
  let n = float64(gProfTokens - 1)
  let perBegin = gProfBegin * 1000.0 / n
  let perEncode = gProfEncode * 1000.0 / n
  let perEnd = gProfEnd * 1000.0 / n
  let perRead = gProfRead * 1000.0 / n
  let total = perBegin + perEncode + perEnd + perRead
  result = &"[infer] decode profile over {gProfTokens - 1} tokens (bo token dau): " &
    &"sessionBegin={perBegin:.1f}ms encodeOneToken={perEncode:.1f}ms " &
    &"sessionEnd={perEnd:.1f}ms sessionRead={perRead:.1f}ms TOTAL={total:.1f}ms/tok\n" & gProfFirst

proc initMicroVmJit*(tag = "default"; selectedBackend = gbAuto) =
  ## Metal compiles per-program MSL; Vulkan uses the shared SPIR-V VM and must
  ## not initialize Metal as a side effect of JIT warmup.
  gMicroVmBackend = selectedBackend
  var cfg = defaultJitConfig()
  if selectedBackend == gbVulkan:
    cfg.enableMetalJit = false
  else:
    if not ensureJitDeviceReady():
      raise newException(CatchableError, "initMicroVmJit: khoi tao GPU that bai")
  gJitCache = initJitCache(cfg)

proc warmupMicroVmJit*(rounds = 12) =
  if gMicroVmBackend == gbVulkan: return
  var prog = buildExampleLutFma()
  for i in 1 .. rounds:
    discard gJitCache.dispatchProgram(prog, "warmup_lut_fma")

type
  LoraAdapter* = object
    r*, inF*, outF*: int
    alpha*: float32
    scale*: float32       ## = alpha / r
    A*: Tensor            ## shape [r, inF]
    B*: Tensor            ## shape [outF, r]

  LoraTrainState* = object
    lr*, wd*: float32
    adamA*, adamB*: Table[string, ApfAdamState]

  JambaLora* = object
    adapters*: Table[string, LoraAdapter]   ## key = tên tham số trong paramList (vd "block2.mamba.inProj.weight")
    backup*: Table[string, seq[float32]]    ## giá trị base gốc (frozen) - phục hồi sau mỗi bước
    quantBase*: Table[string, QuantTensor]  ## MOI: chi co entry cho tham so "quant-resident" (khong
                                             ## giu ban fp32 thuong truc trong RAM - dequant qua LUT
                                             ## MOI LAN mergeIn, xoa ngay sau restoreBase). Rong = mode
                                             ## cu, hanh vi KHONG doi (backward-compatible 100%).

proc newLoraAdapter*(inF, outF, r: int, alpha: float32 = -1'f32): LoraAdapter =
  ## alpha mặc định = 2*r (giống rule-of-thumb phổ biến, cũng là alpha=8,r=4
  ## trong script Python bạn gửi -> alpha=2r).
  ##
  ## CANH BAO FP16: base weight cua Mamba-Chat-2.8B duoc luu/train goc o
  ## fp16 (xem pytorch_bin_loader.nim). Du sau khi doc vao Nim moi thu deu
  ## la fp32 (Tensor cua framework nay LUON fp32, xem nimformer.nim), gia
  ## tri BEN TRONG van mang "dau vet" cua fp16 - nhieu weight rat gan 0
  ## (duoi ~6e-5, gioi han min-normal cua fp16) hoac bi round ve dung 1
  ## trong ~2048 muc gia tri co the co trong khoang do lon cua no. He qua
  ## thuc te cho LoRA:
  ##   - KHONG duoc requantize A/B qua CustomFloat/APF thap-bit (da ep
  ##     requantizeEvery=int.high trong trainStepLora/trainStepGraft o duoi
  ##     - GIU NGUYEN, dung tat no neu ban sua code roi lam gradient LoRA
  ##     bien mat vi bi lam tron ve 0).
  ##   - Init A nho (sqrt(1/inF), nhu duoi) la AN TOAN vi B=0 nen delta ban
  ##     dau = 0 du A co nho, khong cong don voi phan "gan 0" cua base.
  ##   - learning rate LoRA nen o muc 1e-4 hoac thap hon (CFG_LORA_LR trong
  ##     finetune_lora.nim) - lr cao hon de gradient tu base-gan-0 lam A/B
  ##     nhay qua nhanh, mat on dinh (mat tuong tu "loss spike" hay gap khi
  ##     fine-tune model fp16 bang optimizer fp32 ma khong ha lr).
  result.r = r
  result.inF = inF
  result.outF = outF
  result.alpha = if alpha < 0'f32: float32(2 * r) else: alpha
  result.scale = result.alpha / float32(r)
  # A: init nhỏ ngẫu nhiên (kaiming-ish); B: init 0 -> delta ban đầu = 0,
  # model LoRA lúc mới gắn vào giống hệt model gốc (đúng chuẩn LoRA paper).
  result.A = randnTensor(@[r, inF], sqrt(1'f32 / float32(inF)))
  result.B = newTensor(@[outF, r], 0'f32)

proc matmul2D(ctx: Backend, a: Tensor, b: Tensor): Tensor =
  ## a:[M,K] @ b:[K,N] -> [M,N], dùng beMatmul đã có trong backend.nim.
  let M = a.shape[0]; let K = a.shape[1]; let K2 = b.shape[0]; let N = b.shape[1]
  assert K == K2
  result = newTensor(@[M, N])
  result.data = beMatmul(ctx, a.data, M, K, b.data, K2, N)

proc constArr(scale: float32, n: int): seq[float32] =
  ## Mảng hằng số để broadcast 1 scalar vào beMul/beAdd (backend.nim không có
  ## kernel "nhân vô hướng" riêng, chỉ có elementwise 2 mảng cùng cỡ - đây là
  ## cách dùng lại đúng API GPU sẵn có thay vì tự viết vòng for nhân tay).
  ## Việc fill mảng này là O(n) CPU nhưng chỉ là memset, KHÔNG phải phép toán
  ## (không nhân/cộng gì) - phần tính toán thật (nhân, cộng) nằm hết trong
  ## beMul/beAdd bên dưới, chạy trên ctx (GPU).
  result = newSeq[float32](n)
  for i in 0 ..< n: result[i] = scale

proc mergedWeight*(base: Tensor, ad: LoraAdapter, ctx: Backend): Tensor =
  ## W_eff = W_base + scale * B @ A, shape giữ nguyên [outF, inF].
  ## Toàn bộ nhân/cộng ở đây đi qua beMul/beAdd (GPU), KHÔNG còn vòng for
  ## CPU tự cộng dồn từng phần tử như trước.
  let delta = matmul2D(ctx, ad.B, ad.A)   # [outF,r]@[r,inF] -> [outF,inF], da GPU (beMatmul)
  let scaled = beMul(ctx, delta.data, constArr(ad.scale, delta.data.len))
  result = newTensor(base.shape)
  result.data = beAdd(ctx, base.data, scaled)

proc zeroAdapterOutRows*(ad: var LoraAdapter; row0, row1: int) =
  ## Dat 0 cac hang B[row0 ..< row1] (delta LoRA = 0 tren hang out do).
  if ad.r <= 0 or ad.B.data.len != ad.outF * ad.r: return
  let a = max(0, row0)
  let b = min(ad.outF, row1)
  for row in a ..< b:
    let off = row * ad.r
    for k in 0 ..< ad.r:
      ad.B.data[off + k] = 0'f32

proc zeroGradOutRows*(dB: var Tensor; row0, row1: int) =
  if dB.shape.len != 2: return
  let outF = dB.shape[0]
  let r = dB.shape[1]
  if dB.data.len != outF * r: return
  let a = max(0, row0)
  let b = min(outF, row1)
  for row in a ..< b:
    let off = row * r
    for k in 0 ..< r:
      dB.data[off + k] = 0'f32

proc zeroAdamOutRows*(st: var ApfAdamState; outF, r, row0, row1: int) =
  let n = outF * r
  if st.m.len != n or st.v.len != n or r <= 0: return
  let a = max(0, row0)
  let b = min(outF, row1)
  for row in a ..< b:
    let off = row * r
    for k in 0 ..< r:
      st.m[off + k] = 0'f32
      st.v[off + k] = 0'f32

proc adapterOutRowsAbsMax*(ad: LoraAdapter; row0, row1: int): float32 =
  result = 0'f32
  if ad.r <= 0 or ad.B.data.len != ad.outF * ad.r: return
  let a = max(0, row0)
  let b = min(ad.outF, row1)
  for row in a ..< b:
    let off = row * ad.r
    for k in 0 ..< ad.r:
      result = max(result, abs(ad.B.data[off + k]))

proc freezeInProjSsmLora*(m: NimformerModelJamba; lo: var JambaLora;
                          loState: var LoraTrainState) =
  ## G/R: LoRA inProj khong duoc viet B, C, dt. Checkpoint resume cu co the
  ## da lech cac hang nay — zero lai B va Adam m/v tuong ung.
  for i, blk in m.blocks:
    if blk.kind notin {bkMambaReal, bkMambaRealGated, bkMambaRealPlusAttn}:
      continue
    let name = "block" & $i & ".mambaReal.inProj.weight"
    if name notin lo.adapters: continue
    let (row0, row1) = inProjSsmRowRange(blk.mambaReal)
    var ad = lo.adapters[name]
    zeroAdapterOutRows(ad, row0, row1)
    lo.adapters[name] = ad
    if name in loState.adamB:
      var stB = loState.adamB[name]
      zeroAdamOutRows(stB, ad.outF, ad.r, row0, row1)
      loState.adamB[name] = stB

proc gradToAB*(dWeff: Tensor, ad: LoraAdapter, ctx: Backend): tuple[dA, dB: Tensor] =
  ## dWeff: [outF,inF] (gradient theo W_eff, backward tính ra như bình
  ## thường qua Linear.backward). Chain rule:
  ##   dB = scale * dWeff @ A^T     : [outF,inF]@[inF,r] -> [outF,r]
  ##   dA = scale * B^T @ dWeff     : [r,outF]@[outF,inF] -> [r,inF]
  ## Phép nhân với `scale` dùng beMul (GPU) thay vì vòng for CPU.
  let At = transpose(ad.A)   # [inF, r]
  let Bt = transpose(ad.B)   # [r, outF]
  var dB = matmul2D(ctx, dWeff, At)
  var dA = matmul2D(ctx, Bt, dWeff)
  dB.data = beMul(ctx, dB.data, constArr(ad.scale, dB.data.len))
  dA.data = beMul(ctx, dA.data, constArr(ad.scale, dA.data.len))
  result = (dA, dB)

# ═══════════════════════════════════════════════════════════════
# Gắn LoRA vào các tensor weight chỉ định (theo đúng tên trong paramList
# của jamba_hybrid, vd "block0.mamba.inProj.weight",
# "block3.attn.qkvProj.weight" ...). Chỉ gắn được lên tensor 2D (Linear
# weight [outF,inF]) - KHÔNG gắn lên bias/LayerNorm/embedding.
# ═══════════════════════════════════════════════════════════════
proc attachLora*(m: var NimformerModelJamba, targetNames: seq[string],
                  r: int = 4, alpha: float32 = -1'f32): JambaLora =
  result.adapters = initTable[string, LoraAdapter]()
  result.backup = initTable[string, seq[float32]]()
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  for name in targetNames:
    if name notin byName:
      stderr.writeLine("[lora] CANH BAO: khong thay tham so '" & name & "', bo qua")
      continue
    let t = byName[name]
    if t[].shape.len != 2:
      stderr.writeLine("[lora] CANH BAO: '" & name & "' khong phai ma tran 2D, bo qua")
      continue
    let outF = t[].shape[0]; let inF = t[].shape[1]
    result.adapters[name] = newLoraAdapter(inF, outF, r, alpha)
  echo &"[lora] Da gan LoRA (r={r}) vao {result.adapters.len}/{targetNames.len} tham so"

proc attachLoraQuantResident*(m: var NimformerModelJamba, quantWeights: Table[string, QuantTensor],
                               targetNames: seq[string], r: int = 4,
                               alpha: float32 = -1'f32): JambaLora =
  ## Giong attachLora, nhung cho cac tham so KHONG con dang fp32 thuong truc
  ## trong t[].data (vi qua nang RAM voi model 7B - xem giai thich streaming
  ## trong finetune_lora.nim) ma dang nam trong `quantWeights` (tu
  ## load_codestral_into_jamba.loadCodestralQuantResident). Lay shape TU
  ## QuantTensor.shape (khong doc t[].data, vi luc nay t[].data con la rac/
  ## random-init tu newMamba2Real, CHUA duoc dien - se duoc dien MOI LAN
  ## mergeIn qua dequant LUT, khong phai luc attach nay).
  result.adapters = initTable[string, LoraAdapter]()
  result.backup = initTable[string, seq[float32]]()
  result.quantBase = quantWeights
  if targetNames.len == 0:
    raise newException(ValueError,
      "attachLoraQuantResident: targetNames bat buoc; khong tu dong LoRA-hoa moi quant weight")
  var seen = initTable[string, bool]()
  for name in targetNames:
    if name in seen:
      raise newException(ValueError, "attachLoraQuantResident: LoRA target bi lap: " & name)
    seen[name] = true
    if name notin quantWeights:
      raise newException(ValueError,
        "attachLoraQuantResident: LoRA target khong co trong quant weights: " & name)
    let qt = quantWeights[name]
    if qt.shape.len != 2:
      raise newException(ValueError,
        "attachLoraQuantResident: LoRA target quant-resident khong phai ma tran 2D: " & name)
    let outF = qt.shape[0]; let inF = qt.shape[1]
    result.adapters[name] = newLoraAdapter(inF, outF, r, alpha)
  echo &"[lora] Da gan LoRA quant-resident (r={r}) vao {result.adapters.len} target"

proc mergeQuantAdapters*(other: JambaLora, into: var JambaLora) =
  ## Gop them adapter tu attachLora thuong (vd cho block 'A' moi cay) vao
  ## CUNG 1 JambaLora da co adapter quant-resident - de dung 1 vong train
  ## duy nhat cho ca 2 loai. KHONG dung 2 lan attachLora ROI GHI DE ket qua
  ## (se mat quantBase cua lan truoc) - dung ham nay de GOP.
  for name, ad in other.adapters:
    into.adapters[name] = ad
  for name, b in other.backup:
    into.backup[name] = b

proc allProjectionWeightNames*(m: var NimformerModelJamba): seq[string] =
  ## Tiện ích: liệt kê hết các tên weight 2D "gắn LoRA được" (qkvProj/
  ## outProj của attn, inProj/outProj của mamba, fuseGate) - dùng khi muốn
  ## LoRA-hoá TOÀN BỘ model thay vì chọn tay từng block.
  result = @[]
  for (name, t) in paramList(m):
    if t[].shape.len == 2 and (".weight" in name) and
       ("ln1" notin name) and ("ln2" notin name) and (name != "embed.weight"):
      result.add(name)

proc newLoraTrainState*(lora: JambaLora, lr: float32 = 1e-4'f32, wd: float32 = 0.01'f32): LoraTrainState =
  result.lr = lr
  result.wd = wd
  result.adamA = initTable[string, ApfAdamState]()
  result.adamB = initTable[string, ApfAdamState]()
  for name, ad in lora.adapters:
    result.adamA[name] = newApfAdamState(ad.A.data.len)
    result.adamB[name] = newApfAdamState(ad.B.data.len)

var gLutNameCache = initTable[string, array[256, float32]]()  # cache LUT theo ten CustomFloat - build 1 lan/spec (fp4 e2m1 dung chung cho ca 385 tensor, khong build lai moi lan)

proc dequantViaLUT(qt: QuantTensor): seq[float32] =
  ## Nhu quant.dequantizeTensor nhung dung LUT (nhanh hon, xem customfloat.
  ## buildDecodeLUT) khi qt.cf.itemSize==1 (fp4/fp8) - fallback ve
  ## dequantizeTensor thuong (int8/int4/fp32raw hoac itemSize!=1) neu khong
  ## ap dung duoc. KET QUA GIONG HET dequantizeTensor (da verify o test_lut.nim).
  case qt.kind
  of qkFp8E4M3, qkFp8E5M2, qkCustom, qkAuto:
    if qt.cf.itemSize == 1:
      var lut: array[256, float32]
      if qt.cf.name in gLutNameCache:
        lut = gLutNameCache[qt.cf.name]
      else:
        lut = buildDecodeLUT(qt.cf)
        gLutNameCache[qt.cf.name] = lut
      let raw = decodeArrayLUT(qt.data, lut)
      let s = if qt.scale.len > 0: qt.scale[0] else: 1'f32
      if s == 1'f32: return raw
      result = newSeq[float32](raw.len)
      for i in 0 ..< raw.len: result[i] = raw[i] * s
      return result
  else: discard
  result = dequantizeTensor(qt)   # fallback dung cho moi truong hop con lai

proc mergeInOnePlain*(m: var NimformerModelJamba, lora: JambaLora, name: string) =
  ## Ban KHONG-CAN-ADAPTER cua mergeInOne - dung cho tham so quant-resident
  ## nhung KHONG duoc LoRA-hoa (vd lm_head/outProj.weight: dong bang hoan
  ## toan, khong train, chi can dequant TAM THOI luc dung roi xoa - GIONG
  ## HET tinh than mergeInOne cho block 'R', nhung mergeInOne ban goc doc
  ## `lora.adapters[name]` VO DIEU KIEN (se KeyError neu tham so nay khong
  ## co adapter, dung nhu truong hop lm_head). Ham nay CHI dequant qua LUT,
  ## KHONG cong them delta LoRA nao (vi lam gi co adapter ma cong).
  var byName = initTable[string, ptr Tensor]()
  for (nm, t) in paramList(m):
    if nm == name: byName[nm] = t
  if name notin byName: return
  if name notin lora.quantBase: return
  byName[name][].data = dequantViaLUT(lora.quantBase[name])

proc restoreBaseOnePlain*(m: var NimformerModelJamba, lora: JambaLora, name: string) =
  ## Doi xung voi mergeInOnePlain - xoa NGAY sau khi dung xong (giong
  ## restoreBaseOne, nhung khong dong den lora.backup vi khong co gi de
  ## backup - tham so nay chua bao gio bi "merge LoRA" thuc su).
  var byName = initTable[string, ptr Tensor]()
  for (nm, t) in paramList(m):
    if nm == name: byName[nm] = t
  if name notin byName: return
  if name notin lora.quantBase: return
  byName[name][].data = @[]

proc dropZeroLoraAdapters*(lo: var JambaLora): seq[string] =
  ## Bo adapter co A hoac B TOAN 0: delta = scale * B@A = 0 CHINH XAC, nen bo di khong doi
  ## ket qua mot bit nao, chi bot viec (prefill: 2 GEMM LoRA/Linear; decode one-dispatch:
  ## kernel bo qua stage LoRA cua layer do theo co trong bang meta). Tra ve ten da bo.
  ## (Adapter khoi tao B=0 va khong nhan gradient luc train se roi vao truong hop nay.)
  for name, ad in lo.adapters:
    var zeroA = true
    for v in ad.A.data:
      if v != 0'f32:
        zeroA = false
        break
    var zeroB = true
    for v in ad.B.data:
      if v != 0'f32:
        zeroB = false
        break
    if zeroA or zeroB: result.add name
  for name in result: lo.adapters.del(name)

proc mergeLoraIntoQuantBase*(lo: var JambaLora, verbose = true): int =
  ## GOP VINH VIEN LoRA vao weight fp4 (QuantTensor) - TUNG TENSOR MOT:
  ##   dequant(base) -> + scale * B@A -> quantize LAI cung kind/format -> thay vao quantBase,
  ## roi XOA adapter do khoi lo.adapters. Sau do KHONG con lenh LoRA nao tren GPU
  ## (bo ~4 dispatch/block R). Bo nho dinh ~3 ban cua 1 tensor (~1GB), khong dequant ca model.
  ## Tra ve so adapter da gop.
  ## CANH BAO CHAT LUONG (da do tren checkpoint that, block 0/51/58): delta LoRA ~0.1% norm
  ## weight, nho hon buoc luong tu FP4 hang chuc lan -> requantize chi giu ~0-1% delta, trong
  ## khi quantizeFp4KM tinh lai globalScale/scale nhom nen lam tron lai ca tensor (in_proj:
  ## 88% phan tu doi, nhieu ~2.9% |W| - gap ~3 lan chinh delta). Ket qua = mat LoRA + them
  ## nhieu o ca 128 ma tran -> token rac. Chi dung de thi nghiem.
  var names: seq[string]
  for name in lo.adapters.keys: names.add name
  var merged = 0
  for name in names:
    if name notin lo.quantBase: continue
    let base = lo.quantBase[name]
    if base.shape.len != 2: continue
    let ad = lo.adapters[name]
    let outF = base.shape[0]
    let inF = base.shape[1]
    if ad.inF != inF or ad.outF != outF or ad.r <= 0 or
       ad.A.data.len != ad.r * inF or ad.B.data.len != outF * ad.r:
      raise newException(ValueError, "mergeLoraIntoQuantBase: shape adapter khong khop base: " & name)
    var w = dequantViaLUT(base)
    if w.len != outF * inF:
      raise newException(ValueError, "mergeLoraIntoQuantBase: dequant sai kich thuoc: " & name)
    let sc = ad.scale
    for o in 0 ..< outF:
      let row = o * inF
      for r in 0 ..< ad.r:
        let b = ad.B.data[o * ad.r + r] * sc
        if b == 0'f32: continue
        let aRow = r * inF
        for i in 0 ..< inF:
          w[row + i] += b * ad.A.data[aRow + i]
    lo.quantBase[name] = quantizeTensor(w, base.shape, base.kind, customCf = base.cf)
    lo.adapters.del(name)
    inc merged
    if verbose and merged mod 16 == 0:
      echo "[lora-merge] da gop " & $merged & "/" & $names.len & " (" & name & ")"
  if verbose:
    echo "[lora-merge] XONG: " & $merged & " adapter da gop vinh vien vao weight fp4 (quantBase)."
  merged

proc mergeInOne*(m: var NimformerModelJamba, lora: var JambaLora, ctx: Backend, name: string) =
  ## Ban SCOPED cua mergeIn - chi xu ly 1 THAM SO duy nhat (name). Dung cho
  ## streaming train (xem trainStepGraftStreaming trong finetune_lora.nim):
  ## goi mergeInOne ngay TRUOC KHI can weight do (dung cho block hien tai),
  ## restoreBaseOne NGAY SAU KHI dung xong - khong bao gio co qua 1-2 block
  ## quant-resident fp32 cung luc trong RAM (khac mergeIn/restoreBase goc,
  ## xu ly CA LOAT adapters - se dequant ca 52 block cung luc, mat het y
  ## nghia streaming neu dung nham ham do o day).
  var byName = initTable[string, ptr Tensor]()
  for (nm, t) in paramList(m):
    if nm == name: byName[nm] = t
  if name notin byName: return
  let t = byName[name]
  let ad = lora.adapters[name]
  if name in lora.quantBase:
    t[].data = dequantViaLUT(lora.quantBase[name])
  else:
    lora.backup[name] = t[].data
  let merged = mergedWeight(t[], ad, ctx)
  t[].data = merged.data
  if name in lora.quantBase:
    # BUG FIX (xem paramQuantFlags trong jamba_hybrid.nim): t[].data gio la
    # fp32 THAT (base+LoRA), khong phai hang nen nua - tat useQuant de
    # Linear.forward()/uploadResident() dung THANG t[].data, khong dequant
    # nham tu weightQ goc (se lam mat LoRA delta trong im lang).
    let qFlags = paramQuantFlags(m)
    if name in qFlags: qFlags[name][] = false

proc restoreBaseOne*(m: var NimformerModelJamba, lora: JambaLora, name: string) =
  var byName = initTable[string, ptr Tensor]()
  for (nm, t) in paramList(m):
    if nm == name: byName[nm] = t
  if name notin byName: return
  if name in lora.quantBase:
    byName[name][].data = @[]   # giai phong ngay
    # BUG FIX: khoi phuc useQuant=true - lan mergeInOne/uploadResident ke
    # tiep se dequant DUNG tro lai tu weightQ (t[].data dang rong).
    let qFlags = paramQuantFlags(m)
    if name in qFlags: qFlags[name][] = true
  else:
    byName[name][].data = lora.backup[name]

proc bindQuantBaseToLinears*(m: var NimformerModelJamba, lora: var JambaLora) =
  ## FIX IndexDefect "container is empty" trên đường quantResident + NIMBIN4:
  ##
  ## newJambaBlock(quantResident=true) cố ý xóa weight.data = @[] cho
  ## inProj/outProj (tránh allocate ~29GB fp32). QuantTensor thật nằm trong
  ## lora.quantBase (từ loadCodestralQuantResidentFromNimq + newBlock Q4).
  ## Nhưng KHÔNG ai gọi setQuantWeight → Linear.useQuant vẫn false,
  ## weightQ rỗng → uploadResident không upload được gì → forward() rơi
  ## xuống transpose(l.weight) với data rỗng → crash.
  ##
  ## Gọi proc này NGAY SAU khi gán lo.quantBase, TRƯỚC mergeInUploadStreamingJamba
  ## / generate. Bind mọi entry quantBase vào Linear tương ứng (R/A/M/S + outProj).
  ## useQuant=true, weightQ giữ nén; LoRA A/B gắn riêng sau đó trong
  ## mergeInUploadStreamingJamba như thiết kế QLoRA (không merge vật lý).
  let linears = paramLinears(m)
  # SUA (BUG THAT): quantize_lora_bin_4bit.convert() nen MOI tensor "newBlock::*"
  # ke ca tensor 1-D nho (LN gamma/beta, bias cua khoi A). Truoc day 96 tensor
  # nay (12 khoi A x 8) bi BO QUA o day ("96 ten khong map Linear") => khoi A
  # chay voi LN gamma=1/beta=0/bias=0 thay vi gia tri DA TRAIN. Giai nen lai
  # vao dung Tensor cua model.
  var byName = initTable[string, ptr Tensor]()
  for (pname, pt) in paramList(m): byName[pname] = pt
  var bound = 0
  var restoredSmall = 0
  var missing = 0
  var missingNames: seq[string] = @[]
  for name, qt in lora.quantBase.pairs:
    # Guard: bỏ qua QuantTensor rỗng (shape.len == 0) - có thể do checkpoint
    # chưa ghi đầy đủ (vd fuseGate/attn của block G mới init, chưa train).
    if qt.shape.len == 0:
      stderr.writeLine "[lora] bindQuantBaseToLinears: BỎ QUA '" & name & "' (QuantTensor rỗng, shape.len=0)"
      continue
    if name in linears:
      setQuantWeight(linears[name][], qt)
      inc bound
    elif name in byName:
      let dq = dequantizeTensor(qt)
      let t = byName[name]
      var want = 1
      for d in t[].shape: want *= d
      if dq.len == want:
        t[].data = dq
        inc restoredSmall
      else:
        inc missing
        missingNames.add(name & " (dequant=" & $dq.len & " != model=" & $want & ")")
    else:
      # Tên không map được tham số nào của model — bỏ qua, không crash.
      inc missing
      missingNames.add(name)
  echo &"[lora] bindQuantBaseToLinears: {bound} Linear useQuant=true (weightQ từ quantBase), " &
       &"{restoredSmall} tham số nhỏ (LN gamma/beta, bias) giải nén lại vào model, " &
       &"{missing} tên không map được (bỏ qua)."
  for nm in missingNames:
    stderr.writeLine "[lora]   KHONG khoi phuc duoc: " & nm

proc mergeInUploadStreamingJamba*(m: var NimformerModelJamba, lora: var JambaLora, ctx: Backend) =
  ## GAN LoRA RIENG (KHONG merge vat ly) + upload GPU cho toan bo 105 tham
  ## so quant-resident cua model, dung cho duong inference (generate()) khi
  ## pattern co block khac 'R'.
  ##
  ## LICH SU (vi sao KHONG con goi mergeIn()/mergeInOne() o day nua): ban
  ## truoc cua ham nay goi mergeInOne (dequant weightQ qua LUT + mergedWeight
  ## = W_base + scale*B@A ra 1 ma tran fp32 DAY DU) roi upload+free tung
  ## tham so mot. Da giam duoc dinh RAM cua RIENG buoc merge (tu O(105) tham
  ## so cung luc xuong O(1)), nhung VAN bi Killed - vi ban than viec merge
  ## vat ly BUOC weight phai roi khoi dang nen Q4 (~0.5 byte/phan tu) sang
  ## fp32 (~4 byte/phan tu) DE UPLOAD - voi 105 tham so lon cua model 7B,
  ## do la ~3.5GB (nen) -> ~28GB (fp32), BAT KE co free ngay sau do hay
  ## khong, vi luc GPI/Metal can cap phat buffer resident fp32 kich thuoc do
  ## la luc RAM da vuot han. Chinh comment goc o dinh nghia field `weightQ`
  ## trong Linear (nimformer.nim) da canh bao dung diem nay tu truoc: "KHÔNG
  ## bao giờ expand hết ra fp32 một lần rồi giữ mãi (đó là nguyên nhân OOM
  ## trước đó: 6.7B tham số x 4 byte fp32 = ~27GB RAM)" - merge vat ly di
  ## nguoc dung thiet ke goc cua chinh type nay.
  ##
  ## CACH MOI (dung tinh than QLoRA luc suy luan chuan, KHONG merge): base
  ## weight (`weightQ`/`useQuant`) giu NGUYEN VEN dang nen suot doi - KHONG
  ## dequant, KHONG merge, KHONG doi useQuant. Chi gan `loraA`/`loraB`/
  ## `loraR`/`loraScale` (rat nho: [r,inF]+[outF,r] voi r~4-64, thay vi
  ## [outF,inF] day du) truc tiep len Linear. Duong suy luan GPU phai goi
  ## prepareGpuLinForLinear(): base FP4-KM duoc upload packed mot lan vao
  ## Shared GPU cache, con A/B fp32 nho duoc upload private mot lan va fuse
  ## ngay trong cung session. Khong duoc de Linear.forward() tu dong roi vao
  ## dequant CPU cho FP4-KM.
  # THEM (yeu cau debug: in ro block nao duoc gan LoRA - 'r' hay 'a'):
  var attachedCount = 0
  var fullQuantCount = 0   # weight full quant (useQuant) - KHONG phai LoRA
  var fullFp32Count = 0
  var missingCount = 0
  # Khong nested-proc bat `lora` (ARC: cannot capture JambaLora). Chi doc Linear.
  template logWeightSource(name: string, lin: Linear) =
    if lin.useQuant:
      inc fullQuantCount
      echo "[lora]     " & name & ": KHONG co LoRA adapter — FULL weight quant (newBlock/base), useQuant=true"
    elif lin.weight.data.len > 0:
      inc fullFp32Count
      echo "[lora]     " & name & ": KHONG co LoRA adapter — FULL weight fp32 tren Linear"
    else:
      inc missingCount
      echo "[lora]     " & name & ": THIEU weight (khong LoRA, khong useQuant, khong fp32)"

  # FIX Killed:9 / OOM tren GPU tich hop (Iris unified memory):
  # uploadResident Q4 cho ~100+ Linear LON = giu dong thoi ban CPU weightQ +
  # buffer resident GPU (cung an RAM he thong) -> peak vuot han, kernel kill.
  # Khong upload resident nua: resident path tao ban fp32/Q4 rieng cho tung
  # Linear va khong phai la kernel FP4-KM fused. Packed weight chi duoc cache
  # tren GPU tai prepareGpuLinForLinear(), sau do encodeLinByName() su dung
  # truc tiep trong session inference.
  let doUploadResident = false
  echo &"[lora] Bat dau gan LoRA, khong resident upload; GPU packed cache cho {m.blocks.len} block:"
  echo "[lora]   (Ghi chu: block A train FULL — weight nam newBlock::* trong jamba_lora.bin, khong nam adapters.)"
  for i in 0 ..< m.blocks.len:
    let p = &"block{i}"
    let kindLetter = case m.blocks[i].kind
      of bkMambaReal: "R"
      of bkMambaRealPlusAttn: "P"
      of bkMambaRealGated: "G"
      of bkMamba: "M"
      of bkAttn: "A"
      of bkSpectron: "S"
    echo &"[lora]   block{i} = '{kindLetter}'" &
         (if m.blocks[i].kind == bkMambaRealPlusAttn: " (R + Linear++ sidecar)" else: "")
    case m.blocks[i].kind
    of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
      block:
        let name = p & ".mambaReal.inProj.weight"
        if name in lora.adapters:
          # KHONG merge vat ly (mergeInOne) nua - xem Linear.loraA/loraB va
          # forward() trong nimformer.nim: gan adapter RIENG, base (weightQ/
          # useQuant) giu NGUYEN VEN dang nen, khong bao gio bi dequant/merge
          # ra fp32 - tranh dung bloat ~27GB tung gay OOM.
          let ad = lora.adapters[name]
          m.blocks[i].mambaReal.inProj.loraA = ad.A.data
          m.blocks[i].mambaReal.inProj.loraB = ad.B.data
          m.blocks[i].mambaReal.inProj.loraR = ad.r
          m.blocks[i].mambaReal.inProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mambaReal.inProj)
        if doUploadResident:

          uploadResident(m.blocks[i].mambaReal.inProj, ctx)
      block:
        let name = p & ".mambaReal.outProj.weight"
        if name in lora.adapters:
          # KHONG merge vat ly (mergeInOne) nua - xem Linear.loraA/loraB va
          # forward() trong nimformer.nim: gan adapter RIENG, base (weightQ/
          # useQuant) giu NGUYEN VEN dang nen, khong bao gio bi dequant/merge
          # ra fp32 - tranh dung bloat ~27GB tung gay OOM.
          let ad = lora.adapters[name]
          m.blocks[i].mambaReal.outProj.loraA = ad.A.data
          m.blocks[i].mambaReal.outProj.loraB = ad.B.data
          m.blocks[i].mambaReal.outProj.loraR = ad.r
          m.blocks[i].mambaReal.outProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mambaReal.outProj)
        if doUploadResident:
          uploadResident(m.blocks[i].mambaReal.outProj, ctx)
      if m.blocks[i].kind == bkMambaRealPlusAttn:
        block:
          let name = p & ".attn.qkvProj.weight"
          if name in lora.adapters:
            let ad = lora.adapters[name]
            m.blocks[i].attn.qkvProj.loraA = ad.A.data
            m.blocks[i].attn.qkvProj.loraB = ad.B.data
            m.blocks[i].attn.qkvProj.loraR = ad.r
            m.blocks[i].attn.qkvProj.loraScale = ad.scale
            echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
            inc attachedCount
          else:
            logWeightSource(name, m.blocks[i].attn.qkvProj)
          if doUploadResident:
            uploadResident(m.blocks[i].attn.qkvProj, ctx)
        block:
          let name = p & ".attn.outProj.weight"
          if name in lora.adapters:
            let ad = lora.adapters[name]
            m.blocks[i].attn.outProj.loraA = ad.A.data
            m.blocks[i].attn.outProj.loraB = ad.B.data
            m.blocks[i].attn.outProj.loraR = ad.r
            m.blocks[i].attn.outProj.loraScale = ad.scale
            echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
            inc attachedCount
          else:
            logWeightSource(name, m.blocks[i].attn.outProj)
          if doUploadResident:
            uploadResident(m.blocks[i].attn.outProj, ctx)
    of bkMamba:
      block:
        let name = p & ".mamba.inProj.weight"
        if name in lora.adapters:
          # KHONG merge vat ly (mergeInOne) nua - xem Linear.loraA/loraB va
          # forward() trong nimformer.nim: gan adapter RIENG, base (weightQ/
          # useQuant) giu NGUYEN VEN dang nen, khong bao gio bi dequant/merge
          # ra fp32 - tranh dung bloat ~27GB tung gay OOM.
          let ad = lora.adapters[name]
          m.blocks[i].mamba.inProj.loraA = ad.A.data
          m.blocks[i].mamba.inProj.loraB = ad.B.data
          m.blocks[i].mamba.inProj.loraR = ad.r
          m.blocks[i].mamba.inProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mamba.inProj)
        if doUploadResident:

          uploadResident(m.blocks[i].mamba.inProj, ctx)
      block:
        let name = p & ".mamba.outProj.weight"
        if name in lora.adapters:
          # KHONG merge vat ly (mergeInOne) nua - xem Linear.loraA/loraB va
          # forward() trong nimformer.nim: gan adapter RIENG, base (weightQ/
          # useQuant) giu NGUYEN VEN dang nen, khong bao gio bi dequant/merge
          # ra fp32 - tranh dung bloat ~27GB tung gay OOM.
          let ad = lora.adapters[name]
          m.blocks[i].mamba.outProj.loraA = ad.A.data
          m.blocks[i].mamba.outProj.loraB = ad.B.data
          m.blocks[i].mamba.outProj.loraR = ad.r
          m.blocks[i].mamba.outProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mamba.outProj)
        if doUploadResident:

          uploadResident(m.blocks[i].mamba.outProj, ctx)
    of bkAttn:
      block:
        let name = p & ".attn.qkvProj.weight"
        if name in lora.adapters:
          # KHONG merge vat ly (mergeInOne) nua - xem Linear.loraA/loraB va
          # forward() trong nimformer.nim: gan adapter RIENG, base (weightQ/
          # useQuant) giu NGUYEN VEN dang nen, khong bao gio bi dequant/merge
          # ra fp32 - tranh dung bloat ~27GB tung gay OOM.
          let ad = lora.adapters[name]
          m.blocks[i].attn.qkvProj.loraA = ad.A.data
          m.blocks[i].attn.qkvProj.loraB = ad.B.data
          m.blocks[i].attn.qkvProj.loraR = ad.r
          m.blocks[i].attn.qkvProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].attn.qkvProj)
        if doUploadResident:

          uploadResident(m.blocks[i].attn.qkvProj, ctx)
      block:
        let name = p & ".attn.outProj.weight"
        if name in lora.adapters:
          # KHONG merge vat ly (mergeInOne) nua - xem Linear.loraA/loraB va
          # forward() trong nimformer.nim: gan adapter RIENG, base (weightQ/
          # useQuant) giu NGUYEN VEN dang nen, khong bao gio bi dequant/merge
          # ra fp32 - tranh dung bloat ~27GB tung gay OOM.
          let ad = lora.adapters[name]
          m.blocks[i].attn.outProj.loraA = ad.A.data
          m.blocks[i].attn.outProj.loraB = ad.B.data
          m.blocks[i].attn.outProj.loraR = ad.r
          m.blocks[i].attn.outProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].attn.outProj)
        if doUploadResident:

          uploadResident(m.blocks[i].attn.outProj, ctx)
    of bkSpectron:
      block:
        let name = p & ".mamba.inProj.weight"
        if name in lora.adapters:
          # KHONG merge vat ly (mergeInOne) nua - xem Linear.loraA/loraB va
          # forward() trong nimformer.nim: gan adapter RIENG, base (weightQ/
          # useQuant) giu NGUYEN VEN dang nen, khong bao gio bi dequant/merge
          # ra fp32 - tranh dung bloat ~27GB tung gay OOM.
          let ad = lora.adapters[name]
          m.blocks[i].mamba.inProj.loraA = ad.A.data
          m.blocks[i].mamba.inProj.loraB = ad.B.data
          m.blocks[i].mamba.inProj.loraR = ad.r
          m.blocks[i].mamba.inProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mamba.inProj)
        if doUploadResident:

          uploadResident(m.blocks[i].mamba.inProj, ctx)
      block:
        let name = p & ".mamba.outProj.weight"
        if name in lora.adapters:
          # KHONG merge vat ly (mergeInOne) nua - xem Linear.loraA/loraB va
          # forward() trong nimformer.nim: gan adapter RIENG, base (weightQ/
          # useQuant) giu NGUYEN VEN dang nen, khong bao gio bi dequant/merge
          # ra fp32 - tranh dung bloat ~27GB tung gay OOM.
          let ad = lora.adapters[name]
          m.blocks[i].mamba.outProj.loraA = ad.A.data
          m.blocks[i].mamba.outProj.loraB = ad.B.data
          m.blocks[i].mamba.outProj.loraR = ad.r
          m.blocks[i].mamba.outProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].mamba.outProj)
        if doUploadResident:

          uploadResident(m.blocks[i].mamba.outProj, ctx)
      block:
        let name = p & ".attn.qkvProj.weight"
        if name in lora.adapters:
          # KHONG merge vat ly (mergeInOne) nua - xem Linear.loraA/loraB va
          # forward() trong nimformer.nim: gan adapter RIENG, base (weightQ/
          # useQuant) giu NGUYEN VEN dang nen, khong bao gio bi dequant/merge
          # ra fp32 - tranh dung bloat ~27GB tung gay OOM.
          let ad = lora.adapters[name]
          m.blocks[i].attn.qkvProj.loraA = ad.A.data
          m.blocks[i].attn.qkvProj.loraB = ad.B.data
          m.blocks[i].attn.qkvProj.loraR = ad.r
          m.blocks[i].attn.qkvProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].attn.qkvProj)
        if doUploadResident:

          uploadResident(m.blocks[i].attn.qkvProj, ctx)
      block:
        let name = p & ".attn.outProj.weight"
        if name in lora.adapters:
          # KHONG merge vat ly (mergeInOne) nua - xem Linear.loraA/loraB va
          # forward() trong nimformer.nim: gan adapter RIENG, base (weightQ/
          # useQuant) giu NGUYEN VEN dang nen, khong bao gio bi dequant/merge
          # ra fp32 - tranh dung bloat ~27GB tung gay OOM.
          let ad = lora.adapters[name]
          m.blocks[i].attn.outProj.loraA = ad.A.data
          m.blocks[i].attn.outProj.loraB = ad.B.data
          m.blocks[i].attn.outProj.loraR = ad.r
          m.blocks[i].attn.outProj.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].attn.outProj)
        if doUploadResident:

          uploadResident(m.blocks[i].attn.outProj, ctx)
      block:
        let name = p & ".fuseGate.weight"
        if name in lora.adapters:
          # KHONG merge vat ly (mergeInOne) nua - xem Linear.loraA/loraB va
          # forward() trong nimformer.nim: gan adapter RIENG, base (weightQ/
          # useQuant) giu NGUYEN VEN dang nen, khong bao gio bi dequant/merge
          # ra fp32 - tranh dung bloat ~27GB tung gay OOM.
          let ad = lora.adapters[name]
          m.blocks[i].fuseGate.loraA = ad.A.data
          m.blocks[i].fuseGate.loraB = ad.B.data
          m.blocks[i].fuseGate.loraR = ad.r
          m.blocks[i].fuseGate.loraScale = ad.scale
          echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale
          inc attachedCount
        else:
          logWeightSource(name, m.blocks[i].fuseGate)
        if doUploadResident:

          uploadResident(m.blocks[i].fuseGate, ctx)
  block:
    let name = "outProj.weight"
    if name in lora.adapters:
      # Mac dinh KHONG gan LoRA lm_head: checkpoint 105 adapter (outProj.weight)
      # lech trainer hien tai (loraTargets chi R). Bat: BYBY_APPLY_LMHEAD_LORA=1.
      if getEnv("BYBY_APPLY_LMHEAD_LORA") == "1":
        let ad = lora.adapters[name]
        m.outProj.loraA = ad.A.data
        m.outProj.loraB = ad.B.data
        m.outProj.loraR = ad.r
        m.outProj.loraScale = ad.scale
        echo "[lora]     " & name & ": GAN adapter r=" & $ad.r & " scale=" & $ad.scale & " (BYBY_APPLY_LMHEAD_LORA=1)"
        inc attachedCount
      else:
        echo "[lora]     " & name & ": BO QUA LoRA lm_head (mac dinh). Bat: BYBY_APPLY_LMHEAD_LORA=1"
        inc missingCount
    else:
      logWeightSource(name, m.outProj)
    if doUploadResident:

      uploadResident(m.outProj, ctx)
  echo &"[lora] Tong ket: {attachedCount} LoRA adapter | {fullQuantCount} FULL quant (newBlock/base, vd Attention) | {fullFp32Count} FULL fp32 | {missingCount} THIEU weight."
  echo "[lora]   → Attention (block A) nam o cot FULL quant/fp32, KHONG phai 'bo sot adapter'."
  echo "[lora]   → Da BO QUA uploadResident. FP4-KM base + LoRA duoc encode tren GPU trong session."

proc mergeIn*(m: var NimformerModelJamba, lora: var JambaLora, ctx: Backend) =
  ## Trước forward: ghi đè weight base bằng W_eff = base + LoRA delta.
  ## SUA: neu tham so nam trong lora.quantBase (quant-resident, khong giu
  ## fp32 thuong truc) -> DEQUANT MOI QUA LUT ngay tai day (thay vi backup
  ## t[].data cu, vi t[].data dang la rac/khong dung duoc) roi cong LoRA
  ## delta - KHONG luu backup (restoreBase se XOA thay vi phuc hoi, xem duoi).
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  let qFlags = paramQuantFlags(m)
  for name, ad in lora.adapters:
    let t = byName[name]
    if name in lora.quantBase:
      t[].data = dequantViaLUT(lora.quantBase[name])   # base "tuoi" moi lan, khong backup
    else:
      lora.backup[name] = t[].data
    let merged = mergedWeight(t[], ad, ctx)
    t[].data = merged.data
    if name in lora.quantBase and name in qFlags:
      # BUG FIX: xem giai thich day du trong mergeInOne/paramQuantFlags o
      # tren - khong tat cai nay thi uploadResidentJamba() goi ngay sau
      # mergeIn() se dequant lai tu weightQ GOC, xoa sach LoRA vua merge.
      qFlags[name][] = false

proc restoreBase*(m: var NimformerModelJamba, lora: JambaLora) =
  ## Sau backward: trả base về nguyên bản.
  ## SUA: voi tham so quant-resident, KHONG phuc hoi tu backup (khong co) -
  ## XOA thang t[].data (giai phong RAM ngay - day la diem MAU CHOT giai
  ## quyet RAM: khong bao gio co qua 1-2 block fp32 resident cung luc).
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  let qFlags = paramQuantFlags(m)
  for name in lora.adapters.keys:
    if name in lora.quantBase:
      byName[name][].data = @[]   # giai phong - lan mergeIn sau se dequant lai tu dau
      if name in qFlags: qFlags[name][] = true   # BUG FIX: xem mergeIn o tren
    else:
      byName[name][].data = lora.backup[name]

proc trainStepLora*(m: var NimformerModelJamba, lora: var JambaLora,
                     state: var LoraTrainState,
                     idsBatch, targetsBatch: seq[seq[int]], ctx: Backend,
                     requantizeEvery: int = 0): float32 =
  ## 1 bước train CHỈ update A,B của LoRA - mọi tham số khác (base Mamba đã
  ## fine-tune trước đó, embed, outProj, LayerNorm...) giữ NGUYÊN. Đây là
  ## điểm khiến máy yếu vẫn chạy được: số tham số thật sự cập nhật = vài
  ## chục nghìn (r * (inF+outF) mỗi adapter) thay vì toàn bộ model.
  mergeIn(m, lora, ctx)
  let (loss, grads) = computeLossAndGrads(m, idsBatch, targetsBatch, ctx,
                                           trainLmHead = false, trainEmbed = false)
  for name, ad in lora.adapters.mpairs:
    if name notin grads:
      stderr.writeLine("[lora] CANH BAO: thieu grad cho " & name)
      continue
    let (dA, dB) = gradToAB(grads[name], ad, ctx)
    var stA = state.adamA[name]
    var stB = state.adamB[name]
    # requantizeEvery=0 nghia la KHONG ep APF/CustomFloat cho A/B (LoRA rank
    # thap, gia tri nho, ep APF som se giet gradient giong y het canh bao da
    # ghi trong trainStep goc cua test_nimformer.nim) - apfAdamStep coi
    # requantizeEvery<=0 la "khong bao gio requantize", chi Adam thuan fp32.
    discard apfAdamStepFused(ctx, ad.A, dA, stA, state.lr,
                            requantizeEvery = (if requantizeEvery <= 0: int.high else: requantizeEvery))
    discard apfAdamStepFused(ctx, ad.B, dB, stB, state.lr,
                            requantizeEvery = (if requantizeEvery <= 0: int.high else: requantizeEvery))
    state.adamA[name] = stA
    state.adamB[name] = stB
  restoreBase(m, lora)
  result = loss

# MMMM...") -> model mới có thêm block
# LinearPlusAttention/Spectron xen vào, giữ nguyên trọng số Mamba đã train.
# ═══════════════════════════════════════════════════════════════
proc graftBlocks*(old: var NimformerModelJamba, newPattern: string, ctx: Backend): NimformerModelJamba =
  ## `newPattern` phải chứa đúng các ký tự 'M' ở đúng SỐ LƯỢNG và THỨ TỰ
  ## tương đối như `old.pattern` (các 'M' được match tuần tự để copy trọng
  ## số), cộng thêm ký tự 'A'/'S' mới xen vào bất kỳ vị trí nào. Ví dụ:
  ##   old.pattern  = "MMMM"
  ##   newPattern   = "MMAMMSM"   # 4 chu 'M' theo dung thu tu, them 1 'A' + 1 'S'
  ## Các block 'M' trong newPattern được copy y nguyên trọng số (ln1, ln2,
  ## ff, mamba.*) từ old theo đúng thứ tự. Các block 'A'/'S' mới hoàn toàn
  ## random-init - PHẢI train (không LoRA, vì chưa có gì để "thấp hạng
  ## hoá" - train full những block này, ĐỒNG THỜI có thể gắn LoRA lên các
  ## block 'M' đã copy để fine-tune nhẹ phần Mamba cho khớp với Attention
  ## mới, xem attachLora ở trên).
  var oldMIdx = 0
  var mPositionsInNew: seq[int] = @[]
  for i, c in newPattern:
    if c in {'M', 'm'}: mPositionsInNew.add(i)
  let oldMCount = old.pattern.count('M') + old.pattern.count('m')
  assert mPositionsInNew.len == oldMCount,
    &"So block 'M' trong newPattern ({mPositionsInNew.len}) phai khop voi old.pattern ({oldMCount})"

  result = newNimformerModelJamba(old.vocab, old.embedDim, old.nHeads, old.ffMult,
                                   newPattern, useSession = true)
  result.embed.weight = old.embed.weight
  result.outProj.weight = old.outProj.weight
  result.outProj.bias = old.outProj.bias

  for newIdx in mPositionsInNew:
    let oi = oldMIdx
    inc oldMIdx
    result.blocks[newIdx].ln1 = old.blocks[oi].ln1
    result.blocks[newIdx].ln2 = old.blocks[oi].ln2
    result.blocks[newIdx].ff = old.blocks[oi].ff
    result.blocks[newIdx].mamba = old.blocks[oi].mamba
  echo &"[graft] Da copy {oldMIdx} block Mamba tu model cu -> model moi pattern '{newPattern}' " &
       &"({newPattern.len - oldMIdx} block moi (A/S) random-init, can train)"
# ═══════════════════════════════════════════════════════════════
# INFER — 100% GPU, 1 command buffer / token, KHÔNG MSL nhúng trong Nim.
#
# Toàn bộ kernel (gemv fp4-KM / int4-asymmetric / fp32, LoRA, bias+APFLU,
# Linear++ step, split/conv/softplus, SSM step, RMSNorm theo nhóm) là
# BYTECODE micro-VM (session_vm.nim + fp4_gemv_vm.nim), lower + JIT-compile
# qua jit_vm.nim (đúng tinh thần HotSpot/tinygrad MetalCompiler đã ghi ở đầu
# jit_vm.nim: bytecode -> Metal source SINH RA lúc runtime -> newLibraryWithSource
# -> pipeline cache). KHÔNG có kernel .metal nào được viết tay trong Nim.
#
# R (Mamba2Real): mamba2_fused_step.nim (encodeLayerV2, 7 lệnh/layer — giữ
#   nguyên từ bản trước, đã đo ~30 tok/s trước khi bị chậm bởi
#   commitContinue/yield; nay fp4GemvAt tổng quát cho cả int4-asymmetric).
# A (LinearPlusAttention + FFN): gpu_layers.nim (encodeAttnLayer) — MỚI,
#   thay hẳn nhánh forwardLinearPlusInferStep cũ (gọi sai chữ ký hàm — xem
#   lỗi nim check trước khi sửa: nHeads truyền nhầm vào chỗ qkvName) bằng
#   đường GPU thuần, cùng 1 session với R.
#
# Mọi Linear (R in/out_proj, A qkv/out/fc1/fc2, lm_head) đi qua ĐÚNG 1 hàm
# chuẩn bị: gpu_layers.prepareGpuLinForLinear — tự chọn kernel theo
# lin.weightQ.kind (fp4-KM hay int4-asymmetric) hoặc fp32 fallback, và giải
# phóng bản host trùng (Shared/Private GPU đã giữ 1 bản) sau khi upload.
# ═══════════════════════════════════════════════════════════════

import session_vm_dispatch
import gpu_layers
import vendor/bybylang/gpubackend as byby

var gInferStepCount = 0
var gInferVerbose = existsEnv("BYBY_INFER_VERBOSE")
var gGpuGenStatePoisoned = false

proc ilog(msg: string) =
  if not gInferVerbose: return
  stderr.writeLine(msg)
  stderr.flushFile()

proc gNormGroups*(nGroups: int): int =
  ## So nhom cho gated RMSNorm trong mixer R (BAO_CAO.md muc 3.2):
  ##   mac dinh = n_groups cua model  -> dung mamba_ssm / vLLM
  ##   BYBY_MAMBA_NORM_GROUPS=1       -> diagnostic legacy/full-width variant
  let raw = getEnv("BYBY_MAMBA_NORM_GROUPS", "")
  if raw.len > 0:
    try: return max(1, parseInt(raw))
    except ValueError: discard
  nGroups

var gHXPersist, gHNormPersist, gHFinalPersist, gHLogitsPersist: SessionHandle
var gNormFGamma: SessionHandle      ## gamma cua normF (== backbone.norm_f cua HF, RMSNorm, KHONG co beta)
var gNormFOnes: SessionHandle       ## gamma=1 (chi dung khi preset "-unit": xem setGpuNormPreset)
var gLmHead: GpuLin
var gGpuModelReady = false
var gGpuWarmedUp = false

# ── Cau hinh norm cua duong GPU infer ─────────────────────────────────
# LECH THAT giua train va infer (doc code):
#   TRAIN (jamba_hybrid.forwardBlock/forwardBatch + finetune_lora): ln1 cua khoi R
#          = LayerNorm (tru mean + beta), KHONG co normF truoc lm_head.
#   HF Mamba2 goc: RMSNorm cho ln1 va normF = RMSNorm(norm_f) truoc lm_head.
# Checkpoint LoRA + khoi A chi khop voi cau hinh luc train => preset "train".
var gUseNormF* = true
proc setGpuDisableLora*(v: bool) = gDisableLora = v
var gSkipAttn* = false
  ## chan doan (bisect): true = bo qua MOI khoi A (residual di thang) -> chi con Mamba R.
var gNormPresetName* = "hf"

proc setGpuNormPreset*(name: string): bool =
  ## Preset = "<norm>-<final>-<groups>" (khong phan biet hoa/thuong):
  ##   norm   : ln  = LayerNorm (tru mean + beta)   | rms = RMSNorm
  ##   final  : nof = KHONG normF truoc lm_head     | fin = co normF (RMSNorm)
  ##   groups : g1  = full-width gated-RMSNorm | gn = prepared model groups
  ## Bi danh:
  ##   "train" = rms-fin-gn  (trainer hien tai: Mamba-SSM gated-RMSNorm groups)
  ##   "train-legacy" = ln-nof-g1 (trainer cu thoi ky LayerNorm ln1, khong normF)
  ##   "hf"    = rms-fin-g1   (legacy Transformers implementation: full-width reduction)
  ##   "mamba-ssm" = rms-fin-gn (grouped RMSNorm used by mamba_ssm implementations)
  ## Hau to tuy chon "-unit" (vd "rms-fin-g1-unit", "hf-unit"): gamma cua RMSNorm
  ## tien-block (R + sidecar P) va normF = 1. CHI de doc checkpoint train bang trainer
  ## CU - trainer do khong bao gio nap gamma pretrain vao rms1/rmsF (xem
  ## jamba_hybrid.syncRmsFromBaseNorms), nen LoRA + sidecar hoc tren dau vao gamma=1.
  var spec = name.strip().toLowerAscii()
  var unit = false
  if spec.endsWith("-unit"):
    unit = true
    spec = spec[0 ..< spec.len - 5]
  if spec == "train": spec = "rms-fin-gn"
  elif spec == "train-legacy": spec = "ln-nof-g1"
  elif spec == "hf": spec = "rms-fin-g1"
  elif spec == "mamba-ssm": spec = "rms-fin-gn"
  let parts = spec.split('-')
  if parts.len != 3: return false
  if parts[0] notin ["ln", "rms"]: return false
  if parts[1] notin ["nof", "fin"]: return false
  if parts[2] notin ["g1", "gn"]: return false
  if unit and parts[0] != "rms": return false   # "-unit" chi co nghia voi RMSNorm
  gRNormOverride = (if parts[0] == "ln": 1 else: 0)
  gUseNormF = (parts[1] == "fin")
  gRNormGroupsOverride = (if parts[2] == "g1": 1 else: -1)
  gRmsUnitGamma = unit
  gNormPresetName = spec & (if unit: "-unit" else: "")
  result = true

proc normPresetDescription*(m: NimformerModelJamba): string =
  var groups: seq[int] = @[]
  if gRNormGroupsOverride == 1:
    groups = @[1]
  else:
    for blk in m.blocks:
      if blk.kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}:
        let n = gNormGroups(blk.mambaReal.nGroups)
        if n notin groups: groups.add(n)
  result = "R+P-sidecar ln1=" & (if gRNormOverride == 1: "LayerNorm" else: "RMSNorm") &
    (if gRmsUnitGamma: " gamma=1(unit)" else: "") &
    ", normF=" & (if gUseNormF: "bat" else: "tat") &
    ", gated-norm-groups=" & (if groups.len == 0: "n/a" else: $groups)

when not defined(noOneDispatch):
  var gOdActive* = false

  proc odLinHasLora(lin: Linear; K, N: int): bool =
    lin.loraR > 0 and lin.loraA.len == lin.loraR * K and lin.loraB.len == N * lin.loraR

  proc odLinOk(l: Linear): string =
    if not (l.useQuant and l.weightQ.kind == qkFp4KM): return "khong phai fp4-KM"
    if l.weightQ.groupSize != 0 and l.weightQ.groupSize != 32: return "groupSize != 32"
    ""

  proc odTryPrepare(m: var NimformerModelJamba; ctx: Backend): string =
    ## Tra "" neu da pack xong + cai dat (gOdState san sang); khac rong = ly do KHONG eligible.
    ## Moi kiem tra eligibility nam TRUOC khi dung vao weight -> path cu van chay duoc.
    ## Dung con tro (addr) de KHONG deep-copy Linear/Mamba2Real (chua weight hang chuc MB).
    if not isGpuCtx(ctx): return "can GPU Metal"
    if gRNormOverride != -1 or gRNormGroupsOverride != -1 or gRmsUnitGamma:
      return "preset norm override dang bat (LayerNorm / 1 nhom / unit gamma)"
    if m.blocks.len == 0: return "model khong co block"
    # HYBRID: block R (bkMambaReal) -> OD, theo tung DOAN lien tiep (1 dispatch/doan).
    # Block G (bkMambaRealGated) chay path cu xen giua cac doan. Loai khac chua ho tro.
    var odBlocks: seq[int]
    for i in 0 ..< m.blocks.len:
      case m.blocks[i].kind
      of bkMambaReal:
        let w1 = odLinOk(m.blocks[i].mambaReal.inProj)
        if w1.len > 0: return "block " & $i & " inProj " & w1
        let w2 = odLinOk(m.blocks[i].mambaReal.outProj)
        if w2.len > 0: return "block " & $i & " outProj " & w2
        odBlocks.add i
      of bkMambaRealGated: discard
      else:
        return "block " & $i & " kieu " & $m.blocks[i].kind & " chua ho tro (chi R + G)"
    if odBlocks.len == 0: return "khong co block R nao"
    let b0 = odBlocks[0]
    let hidden = m.embedDim
    let C = m.blocks[b0].mambaReal.intermediateSize
    let cDim = m.blocks[b0].mambaReal.convDim
    let H = m.blocks[b0].mambaReal.numHeads
    let P = C + cDim + H
    # Layer co the KHONG co adapter (vd adapter B=0 da bo luc nap): kernel bo qua stage LoRA
    # cua layer do (co theo layer). Rank/scale phai dong nhat giua cac layer CO adapter.
    var rIn, rOut = 0
    var sIn, sOut = 1'f32
    for i in odBlocks:
      if rIn == 0 and odLinHasLora(m.blocks[i].mambaReal.inProj, hidden, P):
        rIn = m.blocks[i].mambaReal.inProj.loraR; sIn = m.blocks[i].mambaReal.inProj.loraScale
      if rOut == 0 and odLinHasLora(m.blocks[i].mambaReal.outProj, C, hidden):
        rOut = m.blocks[i].mambaReal.outProj.loraR; sOut = m.blocks[i].mambaReal.outProj.loraScale
    for i in odBlocks:
      let hi = odLinHasLora(m.blocks[i].mambaReal.inProj, hidden, P)
      let ho = odLinHasLora(m.blocks[i].mambaReal.outProj, C, hidden)
      if (hi and (m.blocks[i].mambaReal.inProj.loraR != rIn or m.blocks[i].mambaReal.inProj.loraScale != sIn)) or
         (ho and (m.blocks[i].mambaReal.outProj.loraR != rOut or m.blocks[i].mambaReal.outProj.loraScale != sOut)):
        return "LoRA rank/scale khong dong nhat tren cac layer (block " & $i & ")"
    try:
      var st = odBegin(ctx, hidden, C, cDim, H, m.blocks[b0].mambaReal.headDim,
        m.blocks[b0].mambaReal.dState, m.blocks[b0].mambaReal.nGroups,
        m.blocks[b0].mambaReal.convKernel, odBlocks.len,
        gNormGroups(m.blocks[b0].mambaReal.nGroups), m.blocks[b0].rms1.eps, rIn, rOut, sIn, sOut)
      st.slotOfBlock = newSeq[int](m.blocks.len)
      for i in 0 ..< m.blocks.len: st.slotOfBlock[i] = -1
      var slot = 0
      for i in odBlocks:
        st.slotOfBlock[i] = slot
        inc slot
      # DOAN R lien tiep (cat boi block G), moi doan = 1 dispatch + 1 buffer trong so rieng.
      # Moi buffer Metal toi da ~2GB -> cat them doan neu qua ~1.5GB.
      let perLayerBytes = odFp4Bytes(hidden, P) + odFp4Bytes(C, hidden) + 512
      let maxLayersPerSeg = max(1, 1_500_000_000 div perLayerBytes)
      var bi = 0
      while bi < m.blocks.len:
        if st.slotOfBlock[bi] < 0:
          inc bi
          continue
        let startSlot = st.slotOfBlock[bi]
        var cnt = 0
        while bi < m.blocks.len and st.slotOfBlock[bi] >= 0 and cnt < maxLayersPerSeg:
          inc cnt
          inc bi
        st.segs.add (start: startSlot, count: cnt)
      for i in odBlocks:
        let blk = addr m.blocks[i]
        let mr = addr blk[].mambaReal
        let lutIn = fp4_gemv_gpu.buildFp4KmLutQ(mr[].inProj.weightQ)
        let lutOut = fp4_gemv_gpu.buildFp4KmLutQ(mr[].outProj.weightQ)
        odAddLayer(st, st.slotOfBlock[i], mr[].aLog.data, mr[].dSkip.data, mr[].dtBias.data, mr[].normWeight.data,
          mr[].convWeight.data, mr[].convBias.data, blk[].ln1.gamma.data,
          mr[].inProj.weightQ.data, lutIn, mr[].outProj.weightQ.data, lutOut,
          (if odLinHasLora(mr[].inProj, hidden, P): mr[].inProj.loraA else: @[]),
          (if odLinHasLora(mr[].inProj, hidden, P): mr[].inProj.loraB else: @[]),
          (if odLinHasLora(mr[].outProj, C, hidden): mr[].outProj.loraA else: @[]),
          (if odLinHasLora(mr[].outProj, C, hidden): mr[].outProj.loraB else: @[]))
      odFinish(ctx, st)
      odInstall(st)
      # KHONG giai phong weight host o day: vong prepareGpuModel ben duoi van nap cache fp4 rieng
      # cho block R (de PREFILL SONG SONG chay GEMM M-block) roi tu bo ban host sau khi upload.
      return ""
    except CatchableError as e:
      return e.msg

proc prepareGpuModel*(m: var NimformerModelJamba, lora: var JambaLora, ctx: Backend) =
  let be = ctx.kind.toByby()
  let hidden = m.embedDim
  when not defined(noOneDispatch):
    if getEnv("BYBY_ONE_DISPATCH", "1") != "0" and not gOdActive:
      let odWhy = odTryPrepare(m, ctx)
      if odWhy.len == 0:
        gOdActive = true
        stderr.writeLine "[one-dispatch] BAT: " & $gOdState.sh.nLayers & " layer R trong " & $gOdState.segs.len &
          " dispatch/token (block G chay path cu xen giua, ntg=" & $gOdState.ntg & ")"
      else:
        stderr.writeLine "[one-dispatch] TAT -> dung path cu: " & odWhy
  for i in 0 ..< m.blocks.len:
    when not defined(noOneDispatch):
      if gOdActive and i < gOdState.slotOfBlock.len and gOdState.slotOfBlock[i] >= 0:
        # Block R do OD so huu state + decode. Van nap cache fp4 rieng cho prefill song song.
        discard prepareGpuLinForLinear(ctx, "block" & $i & ".mambaReal.inProj.weight", m.blocks[i].mambaReal.inProj)
        discard prepareGpuLinForLinear(ctx, "block" & $i & ".mambaReal.outProj.weight", m.blocks[i].mambaReal.outProj)
        continue
    case m.blocks[i].kind
    of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
      let pin = "block" & $i & ".mambaReal.inProj.weight"
      let pout = "block" & $i & ".mambaReal.outProj.weight"
      discard prepareGpuLinForLinear(ctx, pin, m.blocks[i].mambaReal.inProj)
      discard prepareGpuLinForLinear(ctx, pout, m.blocks[i].mambaReal.outProj)
      if i notin gLayersV2:
        let mr = m.blocks[i].mambaReal
        ensureLayerV2(ctx, i,
          mr.numHeads, mr.headDim, mr.dState, mr.nGroups,
          mr.intermediateSize, mr.convDim, mr.convKernel, mr.hiddenSize,
          mr.aLog.data, mr.dSkip.data, mr.dtBias.data, mr.normWeight.data,
          mr.convWeight.data, mr.convBias.data, m.blocks[i].ln1.gamma.data,
          gNormGroups(mr.nGroups), m.blocks[i].rms1.eps,
          ln1Beta = m.blocks[i].ln1.beta.data)
      if m.blocks[i].kind in {bkMambaRealPlusAttn, bkMambaRealGated} and i notin gAttnLayers:
        let pq = "block" & $i & ".attn.qkvProj.weight"
        let po = "block" & $i & ".attn.outProj.weight"
        let qkv = prepareGpuLinForLinear(ctx, pq, m.blocks[i].attn.qkvProj)
        let outp = prepareGpuLinForLinear(ctx, po, m.blocks[i].attn.outProj)
        var fc1, fc2: GpuLin
        # FIX: DÙNG CHUNG ln1 (giống Mamba) cho Attention sidecar - không dùng attnLn1 riêng
        gAttnLayers[i] = initAttnLayer(ctx, hidden, m.blocks[i].attn.nHeads, 0,
          qkv, outp, fc1, fc2,
          m.blocks[i].ln1.gamma.data, m.blocks[i].ln1.beta.data,
          m.blocks[i].ln2.gamma.data, m.blocks[i].ln2.beta.data, m.blocks[i].rms1.eps)
        if m.blocks[i].kind == bkMambaRealGated:
          let gateName = "block" & $i & ".fuseGate.weight"
          gAttnLayers[i].gate = prepareGpuLinForLinear(ctx, gateName, m.blocks[i].fuseGate)
          gAttnLayers[i].gated = true
          gAttnLayers[i].hGateInput = sessionAllocScratch(be, 2 * hidden)
          gAttnLayers[i].hGateLogit = sessionAllocScratch(be, hidden)
          gAttnLayers[i].hGateProb = sessionAllocScratch(be, hidden)
          gAttnLayers[i].hGateComplement = sessionAllocScratch(be, hidden)
          var gateOnes = newSeq[float32](hidden)
          for j in 0 ..< hidden: gateOnes[j] = 1'f32
          gAttnLayers[i].hGateOnes = sessionUploadPrivate(be, gateOnes)
    of bkAttn:
      if i notin gAttnLayers:
        let pq = "block" & $i & ".attn.qkvProj.weight"
        let po = "block" & $i & ".attn.outProj.weight"
        let pf1 = "block" & $i & ".ff.fc1.weight"
        let pf2 = "block" & $i & ".ff.fc2.weight"
        var qkv, outp, fc1, fc2: GpuLin
        qkv = prepareGpuLinForLinear(ctx, pq, m.blocks[i].attn.qkvProj)
        outp = prepareGpuLinForLinear(ctx, po, m.blocks[i].attn.outProj)
        fc1 = prepareGpuLinForLinear(ctx, pf1, m.blocks[i].ff.fc1)
        fc2 = prepareGpuLinForLinear(ctx, pf2, m.blocks[i].ff.fc2)
        gAttnLayers[i] = initAttnLayer(ctx, hidden, m.blocks[i].attn.nHeads, hidden * m.ffMult,
          qkv, outp, fc1, fc2,
          m.blocks[i].ln1.gamma.data, m.blocks[i].ln1.beta.data,
          m.blocks[i].ln2.gamma.data, m.blocks[i].ln2.beta.data, 1e-5'f32)
    of bkMamba, bkSpectron:
      raise newException(ValueError,
        "prepareGpuModel: block M/S chua co duong 100% GPU")

  # ── Buffer persist + normF + lm_head ─────────────────────────────────
  # TRUOC DAY khong ai khoi tao: gHXPersist/gHNormPersist/gHFinalPersist/
  # gHLogitsPersist la handle NIL, gLmHead la GpuLin rong (knF32 = "") ->
  #   - sessionCopyInto(hEmb -> gHXPersist) bao "dst offset vuot qua kich thuoc"
  #   - encodeLin(gLmHead) -> "encodeVmKernel: khong bind duoc pipeline ''"
  # Scratch = Shared, song suot process (giong ban infer cu da chay dung).
  if not gGpuModelReady:
    gHXPersist      = sessionAllocScratch(be, hidden)          # residual, ghi de tai cho qua moi layer
    gHNormPersist   = sessionAllocScratch(be, hidden)          # scratch pre-norm dung chung
    gHFinalPersist  = sessionAllocScratch(be, hidden)          # output normF
    gHLogitsPersist = sessionAllocScratch(be, m.outProj.outF)  # logits [vocab]
    if m.normF.gamma.data.len != hidden:
      raise newException(ValueError,
        "prepareGpuModel: normF.gamma co " & $m.normF.gamma.data.len &
        " phan tu, can " & $hidden)
    gNormFGamma = sessionUploadPrivate(be, m.normF.gamma.data)
    block:
      var ones = newSeq[float32](hidden)
      for i in 0 ..< ones.len: ones[i] = 1'f32
      gNormFOnes = sessionUploadPrivate(be, ones)
    # lm_head di qua CUNG 1 duong chuan bi voi moi Linear khac (fp4-KM / int4-asym,
    # khong fallback fp32). Ten "outProj.weight" khop bindQuantBaseToLinears.
    gLmHead = prepareGpuLinForLinear(ctx, "outProj.weight", m.outProj)
    # Kiem tra handle theo backend that (Vulkan -> vulkanHandle, Metal -> metalHandle).
    # Truoc day chi check metalHandle tren macosx -> backend=vulkan luon fail
    # (metalHandle nil) du buffer Vulkan da cap phat thanh cong.
    proc persistNil(h: SessionHandle): bool =
      # VulkanBufferHandle la object {buf, mem, sizeBytes} - check .buf
      if be == gbVulkan: return h.vulkanHandle.buf == nil
      if be == gbMetal: return h.metalHandle == nil
      false
    if persistNil(gHXPersist) or persistNil(gHNormPersist) or
       persistNil(gHFinalPersist) or persistNil(gHLogitsPersist) or
       persistNil(gNormFGamma):
      let which =
        if be == gbVulkan: "vulkanHandle nil"
        elif be == gbMetal: "metalHandle nil"
        else: "handle nil"
      raise newException(CatchableError,
        "prepareGpuModel: cap phat buffer persist that bai (" & which &
        ", backend=" & $be & ") - het bo nho GPU/unified?")
    gGpuModelReady = true

proc encodeHeadGpu(m: var NimformerModelJamba, ctx: Backend) =
  ## normF (tuy preset) + lm_head. Dung chung cho encodeOneToken va ban trace.
  let be = ctx.kind.toByby()
  let hidden = m.embedDim
  if gUseNormF:
    let gamF = if gRmsUnitGamma: gNormFOnes else: gNormFGamma
    if not sessionRmsnormEnc(be, gHXPersist, gamF, gHFinalPersist, 1, hidden, 1e-5'f32):
      raise newException(CatchableError, "encodeHeadGpu: rmsnorm normF that bai")
    encodeLin(gLmHead, gHFinalPersist, 0, gHLogitsPersist, 0, bias = false)
  else:
    # Khop luc train: lm_head doc THANG residual, khong qua normF.
    encodeLin(gLmHead, gHXPersist, 0, gHLogitsPersist, 0, bias = false)

proc encodeSidecarGpu(ctx: Backend, blockIdx: int) =
  ## Nhanh Linear++ cua khoi P (R + sidecar song song). Norm dau vao GIONG HET nhanh R
  ## cung khoi: luc train ca 2 nhanh doc chung x1 = rms1(x) (jamba_hybrid.forwardBlock),
  ## nen o day cung dung 1 quyet dinh LN/RMS (layerUsesLayerNorm) + 1 co gamma.
  ## Phai goi TRUOC encodeLayerV2 cua khoi do (encodeLayerV2 ghi de residual tai cho).
  encodeLinearPlusBranch(ctx, blockIdx, gHXPersist, gHNormPersist, gAttnScratch.hAttn,
                         useRms = not layerUsesLayerNorm(blockIdx),
                         unitGamma = gRmsUnitGamma)

proc encodeBlockOld(m: var NimformerModelJamba, ctx: Backend, i: int) =
  ## Encode 1 block theo path cu (dung cho moi block khi OD tat, va cho block G khi OD bat).
  case m.blocks[i].kind
  of bkMambaReal:
    encodeLayerV2(ctx, i, gHXPersist, gHNormPersist,
                  "block" & $i & ".mambaReal.inProj.weight",
                  "block" & $i & ".mambaReal.outProj.weight")
  of bkMambaRealPlusAttn, bkMambaRealGated:
    encodeMamba2RealLinearPlusBlock(ctx, i, gHXPersist, gHNormPersist,
      skipLinearPlus = gSkipAttn, unitGamma = gRmsUnitGamma,
      gated = m.blocks[i].kind == bkMambaRealGated)
  of bkAttn:
    if not gSkipAttn:
      encodeAttnLayer(ctx, i, gHXPersist, gHNormPersist)
  of bkMamba, bkSpectron:
    raise newException(ValueError, "encodeOneToken: block M/S chua ho tro")

proc encodeOneToken(m: var NimformerModelJamba, ctx: Backend) =
  ## Encode toan bo 1 buoc forward (R+A theo dung pattern -> normF -> lm_head)
  ## vao session dang mo. Khong sessionRead giua chung.
  let be = ctx.kind.toByby()
  let hidden = m.embedDim
  when not defined(noOneDispatch):
    if gOdActive:
      # Moi DOAN R lien tiep = 1 compute dispatch; block G chay path cu xen giua.
      var i = 0
      var segK = 0
      var tSplit = monoSecs()
      while i < m.blocks.len:
        if gOdState.slotOfBlock[i] >= 0:
          odEncodeSegment(gOdState, segK, gHXPersist)
          decSplitMark(be, "od.seg" & $segK & "(" & $gOdState.segs[segK].count & "L)", tSplit)
          i += gOdState.segs[segK].count
          inc segK
        else:
          encodeBlockOld(m, ctx, i)
          decSplitMark(be, "G.block" & $i, tSplit)
          inc i
      encodeHeadGpu(m, ctx)
      if gDecSplitOn and gDecSplitActive:
        # lm_head: ket thuc o sessionEnd cua caller; do bang buffer rieng cho dong nhat.
        decSplitMark(be, "normF+lm_head", tSplit)
        inc gDecSplitTokens
      return
  for i in 0 ..< m.blocks.len:
    encodeBlockOld(m, ctx, i)
  # normF == backbone.norm_f cua HF Mamba2 = RMSNorm chi co gamma (khop ban infer
  # cu da chay dung, PATCH_MARKER_V4). KHONG con "copy thang neu norm loi": bo qua
  # normF im lang se ra logit sai ma khong ai biet.
  encodeHeadGpu(m, ctx)

proc warmupGpuKernelsOnce(m: var NimformerModelJamba, ctx: Backend) =
  ## Ep JIT-compile + bind PSO lan dau cho MOI loai kernel, trong 1 session
  ## CO LAP (input = embedding cua token 0, x = 0 do gHXPersist chua duoc
  ## copy embedding that vao) roi RESET sach state R/A truoc khi prefill that
  ## bat dau (tranh warmup ban vao trang thai truoc prefill - dung tinh than
  ## ban truoc: "warmup nay ep MOI loai kernel duoc JIT-compile VA dispatch/
  ## bind PSO lan dau O DAY, an toan tuyet doi voi watchdog").
  let be = ctx.kind.toByby()
  let svSkip = gSkipAttn
  let svLora = gDisableLora
  gSkipAttn = false
  gDisableLora = false
  defer:
    gSkipAttn = svSkip
    gDisableLora = svLora
  if sessionBegin(be):
    var warmErr = ""
    try:
      encodeOneToken(m, ctx)
    except CatchableError as e:
      warmErr = e.msg
    if not sessionEnd(be):
      stderr.writeLine "[warmup] session flush loi: " & sessionGetLastError(be) &
        " -> sessionRecoverQueue() truoc buoc tiep theo"
      if not sessionRecoverQueue(be):
        stderr.writeLine "[warmup] sessionRecoverQueue that bai"
    if warmErr.len > 0:
      # KHONG nuot loi: encoder da hong thi cac session reset phia sau cung hong
      # (sinh loi day chuyen "copy_into dst offset"), va forward that se lap lai loi.
      raise newException(CatchableError, "warmup encode loi: " & warmErr)

  if sessionBegin(be):
    for i in 0 ..< m.blocks.len:
      case m.blocks[i].kind
      of bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated:
        if not resetLayerStateV2(ctx, i):
          stderr.writeLine "[warmup] CANH BAO: khong reset duoc state R block " & $i
        if m.blocks[i].kind in {bkMambaRealPlusAttn, bkMambaRealGated} and
           not resetAttnLayerState(ctx, i):
          stderr.writeLine "[warmup] CANH BAO: khong reset duoc state sidecar A block " & $i
      of bkAttn:
        if not resetAttnLayerState(ctx, i):
          stderr.writeLine "[warmup] CANH BAO: khong reset duoc state A block " & $i
      else: discard
    if not sessionEnd(be):
      stderr.writeLine "[warmup] session reset state loi: " & sessionGetLastError(be)
      if not sessionRecoverQueue(be): discard

proc forwardStepJambaGpu*(m: var NimformerModelJamba, tokenId: int, ctx: Backend): Tensor =
  ## MOT session / token cho TOAN BO model (R + A xen ke theo dung pattern),
  ## dung 1 command buffer duy nhat - dung yeu cau "gop thanh 1 kernel tuc 1
  ## session moi token". Khong sessionRead giua layer; chi doc logits 1 lan.
  inc gInferStepCount
  let be = ctx.kind.toByby()
  let hidden = m.embedDim

  if gGpuGenStatePoisoned:
    raise newException(CatchableError,
      "forwardStepJambaGpu: recurrent GPU state khong con dang tin (buoc truoc gap loi/timeout). " &
      "Goi resetGpuGenState() thanh cong truoc khi tiep tuc; khong duoc replay token tren state co the da cap nhat mot phan.")

  let profDecode = getEnv("BYBY_DECODE_PROFILE", "0") != "0"
  let tProfEntry = (if profDecode: monoSecs() else: 0.0)
  if not gGpuWarmedUp:
    warmupGpuKernelsOnce(m, ctx)
    gGpuWarmedUp = true

  var xTensor = m.embed.lookupBatch(@[@[tokenId]], ctx)
  var hEmb = sessionUpload(be, xTensor.data)
  var statefulDispatchStarted = false
  var stepComplete = false
  var tProf0, tProf1, tProf2, tProf3: float64
  try:
    if profDecode: tProf0 = monoSecs()
    if not sessionBegin(be):
      gGpuGenStatePoisoned = true
      discard sessionRecoverQueue(be)
      raise newException(CatchableError,
        "forwardStepJambaGpu: sessionBegin that bai; khong co logits, can reset recurrent state truoc khi thu lai.")
    if profDecode: tProf1 = monoSecs()

    if not sessionCopyInto(be, hEmb, gHXPersist, 0, 0, hidden):
      let copyErr = "copy embedding -> gHXPersist that bai (prepareGpuModel da chay chua?)"
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
      gGpuGenStatePoisoned = true
      raise newException(CatchableError, "forwardStepJambaGpu: " & copyErr)

    try:
      statefulDispatchStarted = true
      gDecSplitActive = gInferStepCount > 0   # bo token dau + warmup
      try:
        encodeOneToken(m, ctx)
      finally:
        gDecSplitActive = false
    except CatchableError as e:
      # encodeOneToken has already queued writes to persistent SSM/Linear++
      # state. Even if sessionEnd later reports success, this token cannot be
      # replayed safely; reset all recurrent state before another attempt.
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
      gGpuGenStatePoisoned = true
      raise newException(CatchableError,
        "forwardStepJambaGpu: encode failed after stateful dispatch; state invalidated, reset required: " & e.msg)
    if profDecode: tProf2 = monoSecs()

    if not sessionEnd(be):
      let err = sessionGetLastError(be)
      discard sessionRecoverQueue(be)
      gGpuGenStatePoisoned = true
      raise newException(CatchableError,
        "forwardStepJambaGpu: GPU command buffer failed (" & err & "). " &
        "Stateful token was NOT replayed because Mamba2/Linear++ state may be partially advanced; " &
        "resetGpuGenState() and restart the sequence. Stateless matmul2 kernels retry internally.")
    if profDecode: tProf3 = monoSecs()
    when not defined(noOneDispatch):
      if gOdActive and getEnv("BYBY_OD_CHECK", "0") != "0":
        let odErr = odCheckErr(gOdState)
        if odErr != 0'u32:
          gGpuGenStatePoisoned = true
          raise newException(CatchableError,
            "one-dispatch: grid barrier het gioi han spin (lastErr=" & $odErr &
            ") - cac threadgroup khong dong thoi resident. Giam BYBY_OD_TG va chay t_grid_barrier.nim.")

    # BYBY_G_DIAGNOSTICS: doc so lieu SAU sessionEnd (doc giua session = rac)
    if getEnv("BYBY_G_DIAGNOSTICS", "0") != "0":
      proc vecStats(name: string; bi: int; v: seq[float32]) =
        if v.len == 0:
          stderr.writeLine "[G_diag] block=" & $bi & " " & name & " EMPTY"
          return
        var sm = 0.0
        var s2 = 0.0
        var mn = 1e30
        var mx = -1e30
        var nanN = 0
        var n0 = 0
        var n1 = 0
        for x0 in v:
          let x = float(x0)
          if x != x or abs(x) > 1e30:
            inc nanN
            continue
          sm += x
          s2 += x * x
          if x < mn: mn = x
          if x > mx: mx = x
          if name == "gate":
            if x < 0.05: inc n0
            if x > 0.95: inc n1
        let n = float(v.len - nanN)
        if n <= 0:
          stderr.writeLine "[G_diag] block=" & $bi & " " & name & " ALL NaN/Inf"
          return
        let mean = sm / n
        let rms = sqrt(s2 / n)
        var line = "[G_diag] block=" & $bi & " " & name &
          " mean=" & $mean & " rms=" & $rms & " min=" & $mn & " max=" & $mx &
          " nan/inf=" & $nanN
        if name == "gate":
          line &= " frac~0(attn)=" & $(float(n0)/n) & " frac~1(mamba)=" & $(float(n1)/n)
        stderr.writeLine line
      for bi in 0 ..< m.blocks.len:
        if m.blocks[bi].kind != bkMambaRealGated: continue
        if bi notin gAttnLayers: continue
        let al = gAttnLayers[bi]
        if not al.gated: continue
        let n = m.embedDim
        stderr.writeLine "[G_diag] ---- block " & $bi & "  mix = g*mamba + (1-g)*attn ; x += mix ----"
        vecStats("gate", bi, sessionRead(be, al.hGateProb, n))
        let hasSnap =
          (be == gbVulkan and al.hDiagMamba.vulkanHandle.buf != nil) or
          (be == gbMetal and al.hDiagMamba.metalHandle != nil)
        if hasSnap:
          vecStats("mamba(before)", bi, sessionRead(be, al.hDiagMamba, n))
          vecStats("attn(before)", bi, sessionRead(be, al.hDiagAttn, n))
        if bi in gLayersV2:
          vecStats("mix(after)", bi, sessionRead(be, gLayersV2[bi].hMix, n))
        vecStats("residual x", bi, sessionRead(be, gHXPersist, n))
    let logits = sessionRead(be, gHLogitsPersist, m.outProj.outF)
    block:
      let dumpPath = getEnv("BYBY_DUMP_LOGITS")   # so sanh path cu vs 1-dispatch: od_compare_logits.py
      if dumpPath.len > 0:
        var df: File
        if open(df, dumpPath, fmAppend):
          if logits.len > 0: discard df.writeBuffer(unsafeAddr logits[0], logits.len * 4)
          df.close()
    if profDecode:
      let tProf4 = monoSecs()
      if gProfTokens == 0:
        let ms = proc(a, b: float64): float64 = (b - a) * 1000.0
        gProfFirstMs = ms(tProfEntry, tProf4)
        gProfFirst = &"[infer] decode token dau: warmup+embed={ms(tProfEntry, tProf0):.1f}ms " &
          &"sessionBegin={ms(tProf0, tProf1):.1f}ms encodeOneToken={ms(tProf1, tProf2):.1f}ms " &
          &"sessionEnd={ms(tProf2, tProf3):.1f}ms sessionRead={ms(tProf3, tProf4):.1f}ms " &
          &"TOTAL={gProfFirstMs:.1f}ms"
      else:
        gProfBegin += tProf1 - tProf0
        gProfEncode += tProf2 - tProf1
        gProfEnd += tProf3 - tProf2
        gProfRead += tProf4 - tProf3
      inc gProfTokens
    result = newTensor(@[1, 1, m.outProj.outF])
    result.data = logits
    inc gInferStepCount
    stepComplete = true
  except CatchableError:
    # Includes failures during the final logits read. Once any stateful
    # kernels were queued, only an explicit recurrent-state reset can make
    # subsequent tokens safe again.
    if statefulDispatchStarted and not stepComplete:
      gGpuGenStatePoisoned = true
    raise
  finally:
    sessionFree(be, hEmb)

type
  BlockStat* = object
    idx*: int
    kind*: char          ## 'R' / 'A' / 'P'
    rms*, maxAbs*, mean*: float32
    nonFinite*: int
    inRms*: float32      ## rms residual TRUOC khoi (de so voi dong gop cua sidecar)
    sideRms*, sideMax*: float32   ## dau ra nhanh Linear++ cua khoi P (truoc khi cong vao residual); -1 = khong co

proc vecStats(v: seq[float32]): tuple[rms, maxAbs, mean: float32, bad: int] =
  var ss = 0.0
  var sm = 0.0
  var mx = 0.0
  var bad = 0
  for x in v:
    if x.classify in {fcNan, fcInf, fcNegInf}:
      inc bad
      continue
    let d = float(x)
    ss += d * d
    sm += d
    if abs(d) > mx: mx = abs(d)
  let n = float(max(1, v.len - bad))
  result = (float32(sqrt(ss / n)), float32(mx), float32(sm / n), bad)

proc forwardStepJambaGpuTrace*(m: var NimformerModelJamba, tokenId: int, ctx: Backend,
                               stats: var seq[BlockStat]): Tensor =
  ## Giong forwardStepJambaGpu (CUNG toan hoc, CUNG state) nhung tach thanh
  ## 1 session / khoi va doc residual sau moi khoi -> do rms/max/mean/NaN.
  ## CHAM hon mot chut (64 lan commit) - CHI dung de chan doan (BYBY_TRACE_BLOCKS=1
  ## hoac trong BYBY_NORM_SWEEP). Cho biet khoi nao lam residual no/NaN.
  let be = ctx.kind.toByby()
  let hidden = m.embedDim
  if gGpuGenStatePoisoned:
    raise newException(CatchableError,
      "forwardStepJambaGpuTrace: recurrent GPU state khong con dang tin; goi resetGpuGenState() truoc.")
  if not gGpuWarmedUp:
    warmupGpuKernelsOnce(m, ctx)
    gGpuWarmedUp = true
  stats.setLen(0)
  var xTensor = m.embed.lookupBatch(@[@[tokenId]], ctx)
  var hEmb = sessionUpload(be, xTensor.data)
  try:
    if not sessionBegin(be):
      raise newException(CatchableError, "trace: sessionBegin (embedding) that bai")
    if not sessionCopyInto(be, hEmb, gHXPersist, 0, 0, hidden):
      discard sessionEnd(be)
      raise newException(CatchableError, "trace: copy embedding that bai")
    if not sessionEnd(be):
      raise newException(CatchableError, "trace: sessionEnd (embedding) that bai")
    var prevRms = vecStats(sessionRead(be, gHXPersist, hidden)).rms
    for i in 0 ..< m.blocks.len:
      if gSkipAttn and m.blocks[i].kind == bkAttn: continue
      if not sessionBegin(be):
        raise newException(CatchableError, "trace: sessionBegin that bai o block " & $i)
      var kindCh = '?'
      var sideRms = -1'f32
      var sideMax = -1'f32
      try:
        case m.blocks[i].kind
        of bkMambaReal:
          kindCh = 'R'
          encodeLayerV2(ctx, i, gHXPersist, gHNormPersist,
                        "block" & $i & ".mambaReal.inProj.weight",
                        "block" & $i & ".mambaReal.outProj.weight")
        of bkMambaRealPlusAttn:
          kindCh = 'P'
          if not gSkipAttn:
            encodeSidecarGpu(ctx, i)
            # Tach session de DOC dau ra sidecar (rms/max) truoc khi no duoc cong vao residual:
            # sidecar >> residual la dau hieu norm/trong so cua nhanh nay lech luc train.
            if not sessionEnd(be):
              raise newException(CatchableError, "trace: sessionEnd (sidecar) that bai")
            let sst = vecStats(sessionRead(be, gAttnScratch.hAttn, hidden))
            sideRms = sst.rms
            sideMax = sst.maxAbs
            if not sessionBegin(be):
              raise newException(CatchableError, "trace: sessionBegin (sau sidecar) that bai o block " & $i)
          encodeLayerV2(ctx, i, gHXPersist, gHNormPersist,
                        "block" & $i & ".mambaReal.inProj.weight",
                        "block" & $i & ".mambaReal.outProj.weight",
                        inputPreNormalized = not gSkipAttn,
                        addResidual = gSkipAttn)
          if not gSkipAttn:
            encodeMamba2RealLinearPlusResidual(ctx, i, gHXPersist,
              gHNormPersist, gAttnScratch.hAttn)
        of bkMambaRealGated:
          kindCh = 'G'
          encodeMamba2RealLinearPlusBlock(ctx, i, gHXPersist, gHNormPersist,
            skipLinearPlus = gSkipAttn, unitGamma = gRmsUnitGamma, gated = true)
        of bkAttn:
          kindCh = 'A'
          encodeAttnLayer(ctx, i, gHXPersist, gHNormPersist)
        of bkMamba, bkSpectron:
          raise newException(ValueError, "trace: block M/S chua ho tro")
      except CatchableError as e:
        discard sessionEnd(be)
        raise newException(CatchableError, "trace: encode loi o block " & $i & ": " & e.msg)
      if not sessionEnd(be):
        raise newException(CatchableError,
          "trace: sessionEnd that bai o block " & $i & ": " & sessionGetLastError(be))
      let xv = sessionRead(be, gHXPersist, hidden)
      let st = vecStats(xv)
      stats.add BlockStat(idx: i, kind: kindCh, rms: st.rms, maxAbs: st.maxAbs,
                          mean: st.mean, nonFinite: st.bad, inRms: prevRms,
                          sideRms: sideRms, sideMax: sideMax)
      prevRms = st.rms
    # head: normF (tuy preset) + lm_head
    if not sessionBegin(be):
      raise newException(CatchableError, "trace: sessionBegin (head) that bai")
    try:
      encodeHeadGpu(m, ctx)
    except CatchableError as e:
      discard sessionEnd(be)
      raise newException(CatchableError, "trace: encode head loi: " & e.msg)
    if not sessionEnd(be):
      raise newException(CatchableError,
        "trace: sessionEnd (head) that bai: " & sessionGetLastError(be))
  except CatchableError:
    gGpuGenStatePoisoned = true
    raise
  finally:
    sessionFree(be, hEmb)
  let logits = sessionRead(be, gHLogitsPersist, m.outProj.outF)
  result = newTensor(@[1, 1, m.outProj.outF])
  result.data = logits

proc resetGpuGenState*(m: var NimformerModelJamba, ctx: Backend) =
  ## Xoa state R (conv/ssm) + A (z/S) - goi 1 lan truoc khi bat dau 1 chuoi
  ## MOI (khac voi warmup: ham nay danh cho luc generate() thuc su bat dau,
  ## sau khi warmup da chay it nhat 1 lan).
  let be = ctx.kind.toByby()
  # Reset is idempotent (writes zeros), so retrying the entire reset after a
  # failed command buffer cannot double-advance recurrent state.
  var lastErr = "sessionBegin failed"
  for attempt in 0 ..< 2:
    if not sessionBegin(be):
      discard sessionRecoverQueue(be)
      lastErr = "sessionBegin failed"
      continue
    var resetOk = true
    for i in 0 ..< m.blocks.len:
      case m.blocks[i].kind
      of bkMambaReal:
        resetOk = resetLayerStateV2(ctx, i) and resetOk
      of bkMambaRealPlusAttn, bkMambaRealGated:
        resetOk = resetLayerStateV2(ctx, i) and resetOk
        resetOk = resetAttnLayerState(ctx, i) and resetOk
      of bkAttn:
        resetOk = resetAttnLayerState(ctx, i) and resetOk
      else: discard
    if not resetOk:
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
      lastErr = "one or more state reset encodes failed"
      continue
    if sessionEnd(be):
      gGpuGenStatePoisoned = false
      return
    lastErr = sessionGetLastError(be)
    discard sessionRecoverQueue(be)
  gGpuGenStatePoisoned = true
  raise newException(CatchableError,
    "resetGpuGenState: could not reset recurrent GPU state after 2 idempotent attempts: " & lastErr)

proc warmupGpuForParallelPrefill*(m: var NimformerModelJamba; ctx: Backend) =
  ## Compile/bind recurrent decode kernels before parallel prefill imports its
  ## terminal states. The normal first-token path warms up and then resets all
  ## states, which would otherwise erase states imported by prefill.
  if not gGpuWarmedUp:
    warmupGpuKernelsOnce(m, ctx)
    gGpuWarmedUp = true
  resetGpuGenState(m, ctx)

proc isGpuTimeoutFailure(msg: string): bool =
  let lower = msg.toLowerAscii()
  result = "timeout" in lower or "timed out" in lower

proc forwardStepJambaGpuRetry*(m: var NimformerModelJamba; tokenId: int;
                               replayTokens: openArray[int]; replayCount: int;
                               ctx: Backend): Tensor =
  ## A recurrent token cannot be blindly replayed after a failed command buffer:
  ## Mamba2/Linear++ may already have advanced some persistent states. On timeout,
  ## reset all states, rebuild the known-good prefix, and only then retry tokenId.
  doAssert replayCount >= 0 and replayCount <= replayTokens.len
  try:
    return forwardStepJambaGpu(m, tokenId, ctx)
  except CatchableError as firstErr:
    if not isGpuTimeoutFailure(firstErr.msg): raise
    stderr.writeLine("[infer] recurrent GPU timeout; reset state, replaying " &
      $replayCount & " committed prefix tokens, then retrying current token once")
    try:
      resetGpuGenState(m, ctx)
      for i in 0 ..< replayCount:
        discard forwardStepJambaGpu(m, replayTokens[i], ctx)
      return forwardStepJambaGpu(m, tokenId, ctx)
    except CatchableError as retryErr:
      raise newException(CatchableError,
        "forwardStepJambaGpuRetry: timeout recovery/replay failed; original=" &
        firstErr.msg & "; recovery=" & retryErr.msg)

proc forwardStepJambaGpuTraceRetry*(m: var NimformerModelJamba; tokenId: int;
                                    replayTokens: openArray[int]; replayCount: int;
                                    ctx: Backend; stats: var seq[BlockStat]): Tensor =
  ## Trace variant with the same sequence-safe retry policy as normal inference.
  doAssert replayCount >= 0 and replayCount <= replayTokens.len
  try:
    return forwardStepJambaGpuTrace(m, tokenId, ctx, stats)
  except CatchableError as firstErr:
    if not isGpuTimeoutFailure(firstErr.msg): raise
    stderr.writeLine("[infer] traced recurrent GPU timeout; reset state, replaying " &
      $replayCount & " committed prefix tokens, then retrying traced token once")
    try:
      resetGpuGenState(m, ctx)
      for i in 0 ..< replayCount:
        discard forwardStepJambaGpu(m, replayTokens[i], ctx)
      stats.setLen(0)
      return forwardStepJambaGpuTrace(m, tokenId, ctx, stats)
    except CatchableError as retryErr:
      raise newException(CatchableError,
        "forwardStepJambaGpuTraceRetry: timeout recovery/replay failed; original=" &
        firstErr.msg & "; recovery=" & retryErr.msg)

proc lpSelfCheck*(ctx: Backend, blockIdx: int): string =
  ## Kiem tra kernel Linear++ step chay 100% tren GPU (xem gpu_layers.lpStepSelfCheck).
  lpStepSelfCheck(ctx, blockIdx)
