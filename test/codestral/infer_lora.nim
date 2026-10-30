## infer_lora.nim
## Inference cho checkpoint đã fine-tune bằng finetune_lora.nim (base Jamba
## đã graft + adapter LoRA lưu riêng trong <path>.lora.json).
##
## 100% GPU: base giữ nén (fp4-KM / int4-asymmetric), LoRA A/B chạy trên GPU
## (bytecode micro-VM JIT, không MSL viết tay), 1 command buffer / token.
## KHÔNG merge fp32, KHÔNG fallback CPU/fp32.

import std/[math, random, strformat, os, strutils, json, tables, times, sequtils, algorithm, sets, terminal]
import nimformer
import backend
import quant
import rawbin
import jamba_hybrid
import lora
when not defined(noOneDispatch):
  from onedispatch import gOdState, odImportState
import session_vm
import jit_vm
import hybrid_block_kernel
import prefill_profile   # BYBY_PREFILL_PROFILE
import prefill_gpu
import inplace_lora_merge
import mamba2_real
import mamba2_full_vm
import mamba2_fused_step
import mamba2_ssm_gpu
import gpu_layers
import fp4_gemv_gpu     # gFp4CacheGpu — sequenceLinearReady cho FP4-KM prefill
import hf_tokenizer      # tokenizer THAT - bat buoc doc lai neu checkpoint tu CFG_INIT_MODE=="codestral"
import test_nimformer   # BPETokenizer: loadBPE/encodeBPE/decodeBPEStr
import load_codestral_into_jamba   # loadCodestralQuantResidentFromNimq - tai tao base tu .nimq
import quantize_lora_bin_4bit

proc configureInferenceDefaults*() =
  ## Apply before GPU initialization/autotuning; explicit environment wins.
  for (name, value) in [("BYBY_INFER_BACKEND", "auto"),
      ("BYBY_GEMV_TG", "auto"), ("BYBY_GPU_AUTOTUNE", "1"),
      ("BYBY_DEBUG_SAMPLING", "0"), ("BYBY_PREFILL_GEMM", "auto"),
      ("BYBY_PREFILL_MPS", "1"),
      # Autotuned scan plus the 8-row x 2-column FP4 prefill tile.
      ("BYBY_PREFILL_SCAN", "auto"),
      ("BYBY_PREFILL_2COL", "1"),
      ("BYBY_PREFILL_FP4_KERNEL", "auto"),
      ("BYBY_PREFILL_RESIDENT", "1")]:
    if not existsEnv(name): putEnv(name, value)
  let mergeMode=getEnv("BYBY_MERGE_LORA","0")
  if mergeMode notin ["0","1","inplace"]:
    raise newException(ValueError,"BYBY_MERGE_LORA must be 0, 1 or inplace (GPU); use 0 to retain separate LoRA")
  if mergeMode!="0" and getEnv("BYBY_MODEL_MMAP","1")=="0":
    raise newException(ValueError,"bounded in-place merge requires BYBY_MODEL_MMAP=1")
  if mergeMode!="0" and getEnv("BYBY_ONE_DISPATCH","0")!="0":
    echo "[infer] In-place merge selects BYBY_ONE_DISPATCH=0 to reuse packed weight buffers."
    putEnv("BYBY_ONE_DISPATCH","0")
when defined(macosx):
  # QOS_CLASS_USER_INITIATED (0x19): CLI inference đang trực tiếp phục vụ
  # yêu cầu của người dùng. Đây là ưu tiên thread CPU; Metal không có API
  # công khai tương đương để cưỡng ép GPU scheduler ưu tiên command buffer.
  type QosClassT = cint
  const QOS_CLASS_USER_INITIATED: QosClassT = 0x19
  proc pthread_set_qos_class_self_np(qos_class: QosClassT, relative_priority: cint): cint
    {.importc: "pthread_set_qos_class_self_np", header: "<pthread/qos.h>".}

  proc setInferenceThreadPriority*() =
    ## Default ON for an interactive inference run. Set
    ## BYBY_MACOS_HIGH_PRIORITY=0 to leave inherited QoS unchanged.
    if getEnv("BYBY_MACOS_HIGH_PRIORITY", "1") != "0":
      let rc = pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0)
      if rc == 0:
        stderr.writeLine "[infer] macOS thread QoS = USER_INITIATED"
      else:
        stderr.writeLine "[infer] macOS QoS update failed (rc=" & $rc & "); using inherited priority"
else:
  proc setInferenceThreadPriority*() = discard

# MOI: doc peak-RSS THAT cua chinh tien trinh (khong fork process con -
# xem giai thich day du trong finetune_lora.nim, ban dau tung dung
# startProcess("/bin/ps",...) va TU NO gay them ap luc RAM luc dang do,
# lam sai lech chinh du lieu dang can dieu tra). Dung de doi chieu: OOM
# kill co phai do CHINH tien trinh nay phinh RAM dan theo thoi gian hay
# do ap luc RAM tu ben ngoai (ung dung khac, he thong) tai 1 thoi diem.
proc rssMB(): string =
  # getrusage reports bytes on macOS, KiB on Linux. Normalize before printing.
  $(peakResidentBytes() div (1024*1024)) & "MiB"

const
  CFG_CHECKPOINT_PATH* = "test/codestral/models/jamba_parallel_lora.bin"   # parallel Mamba+Linear++ checkpoint -
                                             # KHONG con .bin thuc su nao duoc doc/ghi nua.
  CFG_MAX_NEW_TOKENS* = 60
  CFG_TEMPERATURE* = 0.7'f32
  CFG_SEED* = 1234'i64
  # Phai KHOP CHINH XAC voi hang so cung ten trong finetune_lora.nim (dung
  # de tai tao kien truc model tu .nimq - xem loadCheckpointLora ben do).
  CFG_CODESTRAL_VOCAB = 32768
  CFG_CODESTRAL_HIDDEN = 4096
  CFG_CODESTRAL_NHEADS = 128
  CFG_CODESTRAL_DSTATE = 128
  CFG_CODESTRAL_NGROUPS = 8
  CFG_CODESTRAL_CONVKERNEL = 4
  CFG_CODESTRAL_EXPAND = 2
  # FALLBACK (MOI): dung khi KHONG tim thay <path>.lora.json - infer THANG
  # tu base pretrain, KHONG qua LoRA/block Attention da cay (vi khong co gi
  # da train de nap vao cac block do ca). Pattern = TOAN BO 'R' (64 block
  # Mamba2Real thuan, dung HET 64 layer that cua Codestral-Mamba-7B - xem
  # comment CFG_CODESTRAL_PATTERN trong finetune_lora.nim: "Codestral goc
  # KHONG co block Attention nao ca" - day CHINH LA kien truc GOC, khong
  # phai kien truc lai ghep 'RRRRA' dung rieng cho finetune).
  CFG_PURE_MAMBA_PATTERN = "R".repeat(64)
  CFG_HF_DIR_DEFAULT* = "test/codestral/models/hf"
  CFG_NIMQ_PATH_DEFAULT* = "test/codestral/models/codestral_mamba_only.nimq"

type
  AnyTokenizerKind = enum atkBpe, atkHf
  AnyTokenizer = object
    case kind: AnyTokenizerKind
    of atkBpe: bpe: BPETokenizer
    of atkHf: hf: HfTokenizer

proc loadPureMambaForInfer(hfDir, nimqPath: string): tuple[model: NimformerModelJamba, lora: JambaLora, tok: AnyTokenizer] =
  ## KHONG tim thay checkpoint LoRA nao - thay vi bao loi/dung, infer THANG
  ## tren base Mamba-Codestral-7B pretrain (dequant qua LUT tu .nimq).
  echo &"[infer] KHONG thay checkpoint LoRA ({CFG_CHECKPOINT_PATH}.lora.json) - " &
       &"chay Mamba THUAN tu base pretrain ({nimqPath}), pattern='{CFG_PURE_MAMBA_PATTERN}' (64 block 'R', khong Attention)."
  var m = newNimformerModelJamba(CFG_CODESTRAL_VOCAB, CFG_CODESTRAL_HIDDEN, CFG_CODESTRAL_NHEADS,
                                  ffMult = 4, pattern = CFG_PURE_MAMBA_PATTERN, useSession = true,
                                  mambaRealDState = CFG_CODESTRAL_DSTATE,
                                  mambaRealNGroups = CFG_CODESTRAL_NGROUPS,
                                  mambaRealConvKernel = CFG_CODESTRAL_CONVKERNEL,
                                  mambaRealExpand = CFG_CODESTRAL_EXPAND, quantResident = true)
  let quantResident = loadCodestralQuantResidentFromNimq(m, nimqPath)
  var lo: JambaLora
  lo.adapters = initTable[string, LoraAdapter]()   # RONG - khong co LoRA nao ca
  lo.backup = initTable[string, seq[float32]]()
  lo.quantBase = quantResident
  # FIX IndexDefect: bind quantBase → Linear.weightQ/useQuant TRƯỚC generate.
  # (quantResident=true đã xóa weight.data; thiếu bước này → transpose(@[]) crash)
  bindQuantBaseToLinears(m, lo)
  # SUA LAN 2: KHONG eager-dequant het quant-resident nua.
  # forwardStepJambaWithLora (lora.nim) se mergeInOnePlain / restoreBaseOnePlain
  # TUNG BLOCK luc can (on-the-fly) - chi ~2 tensor lon fp32 song cung luc,
  # giong tinh than trainStepGraftStreaming. Duong fallback (forwardBatchSession)
  # van can dequant het o generate() neu pattern khong thuan 'R'.
  echo &"[infer] quant-resident = {quantResident.len} tham so (giu dang nen, dequant on-the-fly luc generate)"
  echo &"[infer] Tokenizer THAT tu {hfDir} ..."
  let hf = newHfTokenizer(hfDir)
  # SUA: TRUOC DAY chi echo vocab ra roi chay tiep. Neu vocab tokenizer khac
  # vocab model thi moi token id >= vocab model se tra cuu SAI HANG embedding
  # (hoac index-out-of-range), va logits/sample cung lech - ra rac 100%. Dung
  # ngay, khong doan.
  if vocabSizeHf(hf) != CFG_CODESTRAL_VOCAB:
    raise newException(ValueError,
      &"[infer] vocab tokenizer ({vocabSizeHf(hf)}) KHAC vocab model " &
      &"({CFG_CODESTRAL_VOCAB}). Dung tokenizer cua CHINH checkpoint nay, " &
      "hoac sua CFG_CODESTRAL_VOCAB cho khop. Khong chay tiep.")
  let tok = AnyTokenizer(kind: atkHf, hf: hf)
  echo &"[infer] Da load base Mamba thuan (KHONG LoRA) tu {nimqPath}, vocab={vocabSizeHf(hf)}"
  result = (m, lo, tok)

proc encodeAny(t: AnyTokenizer, text: string): seq[int] =
  case t.kind
  of atkBpe: encodeBPE(t.bpe, text)
  of atkHf: encodeHf(t.hf, text)

proc encodeAnyNoBos(t: AnyTokenizer, text: string): seq[int] =
  ## Encode KHONG chen BOS - dung de do id cua cac token dinh dang
  ## ("\n", " ", thut dong) khi dung bang mien phat lap.
  case t.kind
  of atkBpe: encodeBPE(t.bpe, text)
  of atkHf: encodeHf(t.hf, text, addEos = false, addBos = false)

proc decodeAny(t: AnyTokenizer, ids: seq[int]): string =
  ## decodeHf gio MAC DINH skipSpecial=true: khong in "<s>"/"[INST]"/
  ## "<SPECIAL_37>" ra man hinh, va khong bao gio de mistral_common raise khi
  ## gap special token (mot so version Tekkenizer raise thay vi bo qua).
  case t.kind
  of atkBpe: decodeBPEStr(t.bpe, ids)
  of atkHf: decodeHf(t.hf, ids)

proc bannedMask(t: AnyTokenizer, vocab: int): seq[bool] =
  ## Chi chan token DIEU KHIEN cua tokenizer, cho phep tat ca token van ban
  ## va khoang trang (ke ca \n / indent). Khong dung bang garbageSpace:
  ## whitespace token hop le co the decode rieng thanh chuoi rong.
  result = newSeq[bool](vocab)
  case t.kind
  of atkHf:
    if t.hf.special.len != vocab:
      raise newException(ValueError,
        "[infer] bang special cua tokenizer (" & $t.hf.special.len &
        ") khac vocab model (" & $vocab & ")")
    for i in 0 ..< vocab:
      result[i] = t.hf.special[i]
    # EOS / stop tokens phai duoc sample de generation ket thuc.
    for id in t.hf.stopIds:
      if id >= 0 and id < vocab: result[id] = false
  of atkBpe:
    discard

proc isStopAny(t: AnyTokenizer, id: int): bool =
  case t.kind
  of atkHf: isStopIdHf(t.hf, id)
  of atkBpe: false

proc pieceAny(t: AnyTokenizer, id: int): string =
  ## ten piece tho (vd "▁", "▁The") - de in chan doan
  case t.kind
  of atkHf: pieceOfHf(t.hf, id)
  of atkBpe: ""

proc buildNoPenalty(t: AnyTokenizer, vocab: int): seq[bool] =
  ## Token dinh dang thuan (\n, space, tab, thut dong) KHONG duoc coi la
  ## "lap lai noi dung". Day dung bai hoc da rut ra trong nim_inference.nim:
  ## voi code, \n lap lai gan nhu moi dong, neu phat no theo repeat-penalty
  ## thi logit cua \n bi de xuong duoi space -> model buoc phai thay xuong
  ## dong bang dau cach du cu phap doi xuong dong (giong co penalize_nl cua
  ## llama.cpp, mac dinh KHONG phat \n).
  result = newSeq[bool](vocab)
  for probe in ["\n", "\n\n", " ", "  ", "    ", "\t", "\n    ", "\n\t", "\r\n"]:
    try:
      for id in encodeAnyNoBos(t, probe):
        if id >= 0 and id < vocab: result[id] = true
    except CatchableError:
      discard

proc loadCheckpointLoraForInfer(path: string): tuple[model: NimformerModelJamba, lora: JambaLora, tok: AnyTokenizer] =
  ## SUA LAN 2 (dong bo voi finetune_lora.nim - KHONG con .bin nao ca):
  ## Base (phan Mamba) TAI TAO tu .nimq (dequant on-the-fly qua LUT, dung
  ## loadCodestralQuantResidentFromNimq - GIONG HET buoc train). Phan
  ## Attention (block moi, train full) NAP TU "newBlockWeights" trong
  ## .lora.json. Adapter LoRA nap tu "adapters" nhu cu. Base va adapter VAN
  ## TRA VE TACH RIENG (giong PeftModel.from_pretrained(base, adapter)) -
  ## cong LoRA delta chi xay ra TAM THOI qua mergeIn/restoreBase moi lan
  ## forward trong generate() ben duoi, KHONG merge vinh vien o day.
  proc shapeNumel(shape: seq[int]): int =
    result = 1
    for dim in shape: result *= dim

  let j = parseFile(path & ".lora.json")
  let nimqPath = j["nimqPath"].getStr
  let pattern = j["pattern"].getStr
  var m = newNimformerModelJamba(CFG_CODESTRAL_VOCAB, CFG_CODESTRAL_HIDDEN, CFG_CODESTRAL_NHEADS,
                                  ffMult = 4, pattern = pattern, useSession = true,
                                  mambaRealDState = CFG_CODESTRAL_DSTATE,
                                  mambaRealNGroups = CFG_CODESTRAL_NGROUPS,
                                  mambaRealConvKernel = CFG_CODESTRAL_CONVKERNEL,
                                  mambaRealExpand = CFG_CODESTRAL_EXPAND, quantResident = true)
  echo &"[infer] Tai tao base tu {nimqPath} (dequant on-the-fly qua LUT, pattern='{pattern}') ..."
  let quantResident = loadCodestralQuantResidentFromNimq(m, nimqPath)

  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  var requiredAdapters = initTable[string, bool]()
  var requiredNewBlocks = initTable[string, bool]()
  for i, c in pattern:
    let prefix = &"block{i}."
    if c in {'R', 'r', 'P', 'p', 'G', 'g'}:
      # 'G' must be here: finetune_lora.nim now saves LoRA adapters for G's
      # pretrained mambaReal weights (same as R/P). Without 'G', loading a G
      # checkpoint raised "JSON checkpoint co LoRA adapter ngoai pattern".
      requiredAdapters[prefix & "mambaReal.inProj.weight"] = true
      requiredAdapters[prefix & "mambaReal.outProj.weight"] = true
    elif c in {'M', 'm'}:
      requiredAdapters[prefix & "mamba.inProj.weight"] = true
      requiredAdapters[prefix & "mamba.outProj.weight"] = true
    if c notin {'R', 'r', 'M', 'm'}:
      for name, _ in byName:
        if name.startsWith(prefix) and
           (m.blocks[i].kind != bkMambaRealPlusAttn or name.startsWith(prefix & "attn.")) and
           (m.blocks[i].kind != bkMambaRealGated or name.startsWith(prefix & "attn.") or
            name.startsWith(prefix & "fuseGate.")):
          requiredNewBlocks[name] = true
  var nAttnW = 0
  var seenNewBlocks = initTable[string, bool]()
  for name, v in j["newBlockWeights"].pairs:
    if name notin requiredNewBlocks or name notin byName:
      raise newException(ValueError, "[infer] JSON checkpoint co newBlock ngoai pattern: " & name)
    if v.kind != JArray or v.len != shapeNumel(byName[name][].shape):
      raise newException(ValueError, "[infer] JSON newBlock sai shape/data: " & name)
    var idx = 0
    for x in v.items: byName[name][].data[idx] = x.getFloat.float32; inc idx
    seenNewBlocks[name] = true
    inc nAttnW
  for name, _ in requiredNewBlocks:
    if name notin seenNewBlocks:
      raise newException(ValueError, "[infer] JSON checkpoint thieu newBlock bat buoc: " & name)
  echo &"[infer] Da nap {nAttnW} tensor Linear++ Attention (block moi, train full) tu .lora.json"

  var lo: JambaLora
  lo.adapters = initTable[string, LoraAdapter]()
  lo.backup = initTable[string, seq[float32]]()
  for name, v in j["adapters"].pairs:
    if name notin requiredAdapters or name notin byName:
      raise newException(ValueError, "[infer] JSON checkpoint co LoRA adapter ngoai pattern: " & name)
    var ad: LoraAdapter
    ad.inF = v["inF"].getInt()
    ad.outF = v["outF"].getInt()
    ad.r = v["r"].getInt()
    if ad.inF <= 0 or ad.outF <= 0 or ad.r <= 0 or
       byName[name][].shape != @[ad.outF, ad.inF] or
       v["A"].kind != JArray or v["B"].kind != JArray or
       v["A"].len != ad.r * ad.inF or v["B"].len != ad.outF * ad.r:
      raise newException(ValueError, "[infer] JSON LoRA sai shape/data: " & name)
    # New checkpoints store alpha per adapter; older JSON stores it only at
    # root. Preserve that value instead of silently assuming alpha=8.
    ad.alpha = v.getOrDefault("alpha").getFloat(
      j.getOrDefault("alpha").getFloat(8.0)).float32
    ad.scale = ad.alpha / float32(ad.r)
    ad.A = newTensor(@[ad.r, ad.inF])
    ad.B = newTensor(@[ad.outF, ad.r])
    var i = 0
    for x in v["A"].items: ad.A.data[i] = float32(x.getFloat()); inc i
    i = 0
    for x in v["B"].items: ad.B.data[i] = float32(x.getFloat()); inc i
    lo.adapters[name] = ad
  for name, _ in requiredAdapters:
    if name notin lo.adapters:
      raise newException(ValueError, "[infer] JSON checkpoint thieu LoRA adapter bat buoc: " & name)
  lo.quantBase = quantResident   # inProj/outProj cua block Mamba - quant-resident,
                                  # mergeIn/restoreBase se tu LUT-dequant/xoa dung luc can.
  # FIX IndexDefect: bind quantBase → Linear.weightQ/useQuant TRƯỚC generate.
  bindQuantBaseToLinears(m, lo)

  # NHU CFG_INIT_MODE trong finetune_lora.nim: neu checkpoint tu che do
  # "codestral" (co sidecar ".hf_tokenizer_dir"), PHAI doc lai dung tokenizer
  # THAT tu chinh HF dir goc - BPE tu che se lam embedding pretrain vo nghia
  # (xem giai thich chi tiet trong hf_tokenizer.nim). KHONG co fallback ngam.
  let hfDirSidecar = path & ".hf_tokenizer_dir"
  var tok: AnyTokenizer
  let step = j["epoch"].getInt()
  if fileExists(hfDirSidecar):
    let hfDir = readFile(hfDirSidecar).strip()
    echo &"[infer] Checkpoint tu che do codestral - doc lai tokenizer THAT tu {hfDir} ..."
    let hfT = newHfTokenizer(hfDir)
    if vocabSizeHf(hfT) != CFG_CODESTRAL_VOCAB:
      raise newException(ValueError,
        &"[infer] vocab tokenizer ({vocabSizeHf(hfT)}) KHAC vocab model " &
        &"({CFG_CODESTRAL_VOCAB}) - xem giai thich trong loadPureMambaForInfer.")
    tok = AnyTokenizer(kind: atkHf, hf: hfT)
    echo &"[infer] Da load {path} (base tai tao tu .nimq) + {lo.adapters.len} adapter LoRA (TACH RIENG, chua merge), " &
         &"epoch={step}, vocab={vocabSizeHf(tok.hf)} (tokenizer that)"
  elif fileExists(path & ".bpe.json"):
    let bpeTok = loadBPE(path & ".bpe.json")
    if vocabSize(bpeTok) != m.vocab:
      raise newException(ValueError,
        &"[infer] BPE vocab ({vocabSize(bpeTok)}) KHAC checkpoint vocab ({m.vocab}); " &
        "khong the dung token ID nay voi embedding/lm_head cua model.")
    tok = AnyTokenizer(kind: atkBpe, bpe: bpeTok)
    echo &"[infer] Da load {path} (base tai tao tu .nimq) + {lo.adapters.len} adapter LoRA (TACH RIENG, chua merge), " &
         &"epoch={step}, vocab={vocabSize(bpeTok)}"
  else:
    # SUA: xem giai thich day du trong loadBinaryLoraForInfer - ca 2 sidecar
    # deu thieu thi tu tao ".hf_tokenizer_dir" bang tay (1 dong text), KHONG
    # can finetune lai, MIEN LA checkpoint train o CFG_INIT_MODE="codestral".
    raise newException(IOError,
      &"[infer] Khong tim thay sidecar tokenizer nao cho checkpoint '{path}' " &
      &"(da thu '{hfDirSidecar}' va '{path}.bpe.json'). Neu train o " &
      "CFG_INIT_MODE=\"codestral\", tu tao sidecar bang 1 lenh (KHONG can " &
      &"finetune lai):\n  echo -n \"/duong/dan/toi/thu-muc-model-HF\" > \"{hfDirSidecar}\"")
  result = (m, lo, tok)

var gCkptNormPreset = ""
  ## meta["normPreset"] cua checkpoint (finetune_lora ghi = preset norm cua forward luc train).
  ## Rong = checkpoint CU (truoc ban co khoa nay) -> generate() tu do preset bang bisect.
var gCkptMambaNormImpl = ""

proc loadBinaryLoraForInfer*(path: string): tuple[model: NimformerModelJamba, lora: JambaLora, tok: AnyTokenizer] =
  ## Hỗ trợ NIMBIN1 (fp32) và NIMBIN4 (QLoRA 4-bit).
  ## SUA (bug RAM): newBlock KHONG con dequant fp32 ngay luc load nua (truoc
  ## day song resident suot doi chuong trinh, dung y het van de m 8GB fp32
  ## ma quant-resident da giai quyet cho base) - gio giu nguyen dang
  ## QuantTensor, gop CHUNG bang voi quantResident (base tu .nimq), de
  ## mergeIn()/restoreBase() (goi trong generate(), xem infer_lora.generate)
  ## tu dequant-on-the-fly + giai phong CA HAI cung 1 co che, giong het
  ## nhau khong phan biet "base" hay "newBlock" nua.
  var meta: Table[string, string]
  var tensors: seq[RawBinTensor]
  var newBlockQuant: Table[string, QuantTensor]
  if isQLora4Bin(path):
    echo "[infer] Phát hiện NIMBIN4 (QLoRA 4-bit) — newBlock giữ nén, dequant on-the-fly lúc generate()..."
    let q = if getEnv("BYBY_MODEL_MMAP", "1") != "0": loadMappedQLora4Bin(path) else: loadQLora4Bin(path)
    meta = q.meta
    tensors = q.fp32Tensors
    newBlockQuant = q.quantWeights
  elif getEnv("BYBY_MODEL_MMAP", "1") != "0":
    let loaded = loadMappedRawBin(path)
    meta = loaded.meta
    tensors = loaded.tensors
  else:
    var reader = openRawBinReader(path)
    defer: reader.closeRawBinReader()
    meta = reader.meta
    while reader.remaining > 0:
      tensors.add(reader.readRawTensor())
  let fmt = meta.getOrDefault("format")
  if fmt != "nimformers-lora-v1" and fmt != "nimformers-lora-v1-qlora4":
    raise newException(ValueError, "Checkpoint khong phai LoRA-only .bin: " & path)
  let pattern = meta["pattern"]
  var nimqPath = meta["nimqPath"]
  # Checkpoints created before the folder reorganization store a base filename
  # (e.g. "codestral_mamba_only.nimq"). Resolve it beside the checkpoint when
  # it is not present in the process working directory.
  if not fileExists(nimqPath):
    let besideCheckpoint = parentDir(path) / nimqPath
    if fileExists(besideCheckpoint): nimqPath = besideCheckpoint
  gCkptNormPreset = meta.getOrDefault("normPreset", "")
  gCkptMambaNormImpl = meta.getOrDefault("mambaNormImpl", "")
  if gCkptMambaNormImpl.len > 0 and gCkptMambaNormImpl != TRAIN_MAMBA_NORM_IMPL:
    raise newException(ValueError,
      &"[infer] Checkpoint Mamba norm implementation '{gCkptMambaNormImpl}' khong duoc ho tro; " &
       &"runtime nay dang dung '{TRAIN_MAMBA_NORM_IMPL}'. Tu choi infer de tranh logits sai.")
  if gCkptNormPreset.len > 0:
    echo "[infer] checkpoint ghi preset norm luc train: " & gCkptNormPreset
    if gCkptMambaNormImpl.len > 0:
      echo "[infer] checkpoint ghi Mamba norm implementation: " & gCkptMambaNormImpl
  else:
    echo "[infer] checkpoint KHONG ghi preset norm (train bang trainer cu) - se do bang bisect."
  # THEM (theo yeu cau debug): in ro pattern block that su cua model - moi
  # ky tu la 1 block (R=Mamba2Real, P=Mamba2Real+Linear++ sum, G=Mamba2Real+
  # gated Linear++/fuseGate, M=Mamba, A=LinearPlusAttention, S=Spectron).
  block:
    var cR, cP, cG, cM, cA, cS = 0
    var upperPattern = ""
    for c in pattern:
      let cu = c.toUpperAscii()
      upperPattern.add(cu)
      case cu
      of 'R': inc cR
      of 'P': inc cP
      of 'G': inc cG
      of 'M': inc cM
      of 'A': inc cA
      of 'S': inc cS
      else: discard
    echo &"[infer] Pattern block ({pattern.len} block): {upperPattern}"
    echo &"[infer] Thanh phan: R(Mamba2Real)={cR} P(Mamba2Real+Linear++ sum)={cP} " &
         &"G(Mamba2Real+gated Linear++ fuseGate)={cG} M(Mamba)={cM} " &
         &"A(LinearPlusAttention)={cA} S(Spectron)={cS}"
  # Validate the tokenizer before model construction or base dequantization.
  let checkpointVocab = parseInt(meta["vocab"])
  let hfDirSidecar = path & ".hf_tokenizer_dir"
  let bpeSidecar = path & ".bpe.json"
  var tok: AnyTokenizer
  if fileExists(hfDirSidecar):
    let hfT = newHfTokenizer(readFile(hfDirSidecar).strip)
    if vocabSizeHf(hfT) != checkpointVocab:
      raise newException(ValueError,
        &"[infer] HF tokenizer vocab ({vocabSizeHf(hfT)}) KHAC checkpoint vocab ({checkpointVocab}); " &
        "token ID co the van nam trong bien nhung co y nghia khac, se tao logits/token rac.")
    tok = AnyTokenizer(kind: atkHf, hf: hfT)
  elif fileExists(bpeSidecar):
    let bpeTok = loadBPE(bpeSidecar)
    if vocabSize(bpeTok) != checkpointVocab:
      raise newException(ValueError,
        &"[infer] BPE tokenizer vocab ({vocabSize(bpeTok)}) KHAC checkpoint vocab ({checkpointVocab}); " &
        "token ID co the van nam trong bien nhung khong con cung y nghia voi embedding/lm_head.")
    tok = AnyTokenizer(kind: atkBpe, bpe: bpeTok)
  else:
    # SUA: TRUOC DAY goi thang loadBPE(bpeSidecar) khi hfDirSidecar khong co -
    # neu checkpoint tu CFG_INIT_MODE=="codestral" (dung tokenizer HF that,
    # KHONG co .bpe.json) thi ca 2 sidecar deu thieu -> crash bang mot
    # IOError "cannot open ...bpe.json" kho hieu, de tuong lam phai finetune
    # lai. THUC RA khong can: sidecar ".hf_tokenizer_dir" chi la 1 dong text
    # chua duong dan toi thu muc HF goc - tu tao duoc bang tay, KHONG can
    # chay lai finetune_lora. (Ban finetune_lora.nim moi da tu ghi sidecar
    # nay SOM, ngay sau khi tokenizer san sang - loi nay chi con gap voi
    # checkpoint cu luu TRUOC ban va cho.)
    raise newException(IOError,
      &"[infer] Khong tim thay sidecar tokenizer nao cho checkpoint '{path}' " &
      &"(da thu '{hfDirSidecar}' va '{bpeSidecar}'). Neu checkpoint nay train " &
      "o CFG_INIT_MODE=\"codestral\" (tokenizer HF that), tu tao sidecar bang " &
      "1 lenh, KHONG can finetune lai:\n" &
      &"  echo -n \"/duong/dan/toi/thu-muc-model-HF\" > \"{hfDirSidecar}\"\n" &
      "(dung DUNG duong dan CFG_HF_DIR luc finetune, vd thu muc Mamba-Codestral-" &
      "7B-v0.1 da tai ve). Neu checkpoint train o CFG_INIT_MODE=\"graft\" (BPE tu " &
      &"train) thi thieu '{bpeSidecar}' that su can finetune lai (BPE khong " &
      "tai tao lai duoc tu con so khac).")
  var m = newNimformerModelJamba(checkpointVocab, parseInt(meta["hidden"]),
                                  parseInt(meta["nHeads"]), parseInt(meta["ffMult"]),
                                  pattern, useSession = true,
                                  mambaRealDState = CFG_CODESTRAL_DSTATE,
                                  mambaRealNGroups = CFG_CODESTRAL_NGROUPS,
                                  mambaRealConvKernel = CFG_CODESTRAL_CONVKERNEL,
                                  mambaRealExpand = CFG_CODESTRAL_EXPAND, quantResident = true,
                                  initializeWeights = false)
  var quantResident = loadCodestralQuantResidentFromNimq(m, nimqPath)
  for name, qt in newBlockQuant:
    quantResident[name] = qt
  echo &"[infer] quant-resident: {quantResident.len} tham so nen tong cong " &
       &"({newBlockQuant.len} tu newBlock QLoRA4 = Attention/block moi da train FULL trong checkpoint, " &
       &"con lai tu base .nimq = Mamba R)."
  echo "[infer]   → newBlock KHONG phai LoRA adapter; do la full weight Attention luu trong cung file .bin."
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  var expectedNewBlocks = initTable[string, bool]()
  for key, _ in meta:
    if key.startsWith("newBlock.") and key.endsWith(".shape"):
      expectedNewBlocks[key[9 ..< key.len - 6]] = true
  if fmt == "nimformers-lora-v1-qlora4" and expectedNewBlocks.len == 0:
    raise newException(ValueError, "[infer] QLoRA4 checkpoint thieu manifest newBlock.shape")
  var requiredNewBlocks = initTable[string, bool]()
  var requiredAdapters = initTable[string, bool]()
  for i, c in pattern:
    let prefix = &"block{i}."
    if c in {'R', 'r', 'P', 'p', 'G', 'g'}:
      # 'G' must be here: finetune_lora.nim now saves LoRA adapters for G's
      # pretrained mambaReal weights (same as R/P). Without 'G', loading a G
      # checkpoint raised "checkpoint co LoRA adapter ngoai pattern".
      requiredAdapters[prefix & "mambaReal.inProj.weight"] = true
      requiredAdapters[prefix & "mambaReal.outProj.weight"] = true
    elif c in {'M', 'm'}:
      requiredAdapters[prefix & "mamba.inProj.weight"] = true
      requiredAdapters[prefix & "mamba.outProj.weight"] = true
    if c notin {'R', 'r', 'M', 'm'}:
      for (name, _) in paramList(m):
        if name.startsWith(prefix) and
           (m.blocks[i].kind != bkMambaRealPlusAttn or name.startsWith(prefix & "attn.")) and
           (m.blocks[i].kind != bkMambaRealGated or name.startsWith(prefix & "attn.") or
            name.startsWith(prefix & "fuseGate.")):
          requiredNewBlocks[name] = true
  for name, _ in requiredNewBlocks:
    if name notin expectedNewBlocks:
      raise newException(ValueError, "[infer] checkpoint manifest thieu newBlock bat buoc: " & name)
  for name, _ in expectedNewBlocks:
    if name notin requiredNewBlocks:
      raise newException(ValueError, "[infer] checkpoint khai bao newBlock ngoai model/pattern: " & name)
    if name notin byName or meta.getOrDefault("newBlock." & name & ".shape") != $byName[name][].shape:
      raise newException(ValueError, "[infer] manifest newBlock sai shape: " & name)
  var seenNewBlocks = initTable[string, bool]()
  for name, qt in newBlockQuant:
    if name notin byName or name notin expectedNewBlocks or
       qt.shape != byName[name][].shape or qt.bytes.len == 0:
      raise newException(ValueError, "[infer] newBlock quant sai shape/data: " & name)
    if name in seenNewBlocks:
      raise newException(ValueError, "[infer] checkpoint trung newBlock quant: " & name)
    seenNewBlocks[name] = true
  var seenAdapterA = initTable[string, bool]()
  var seenAdapterB = initTable[string, bool]()
  var lo: JambaLora
  lo.adapters = initTable[string, LoraAdapter]()
  lo.backup = initTable[string, seq[float32]]()
  for tensor in tensors:
    if tensor.name.startsWith("newBlock::"):
      let name = tensor.name[10 .. ^1]
      if name notin byName or name notin expectedNewBlocks:
        raise newException(ValueError, "[infer] checkpoint co newBlock ngoai model/pattern: " & name)
      if name in seenNewBlocks:
        raise newException(ValueError,
          "[infer] checkpoint co hai nguon newBlock (raw/quant hoac trung ten): " & name)
      var expectedElements = 1
      for dim in tensor.shape: expectedElements *= dim
      if tensor.shape != byName[name][].shape or tensor.floatValues.len != expectedElements:
        raise newException(ValueError, "[infer] newBlock raw sai shape/data: " & name)
      byName[name][].data = tensor.data
      byName[name][].mappedData = tensor.mappedData
      seenNewBlocks[name] = true
    elif tensor.name.startsWith("adapter::"):
      let rest = tensor.name[9 .. ^1]
      let splitAt = rest.rfind("::")
      if splitAt < 0: continue
      let adapterName = rest[0 ..< splitAt]
      let part = rest[splitAt + 2 .. ^1]
      if adapterName notin requiredAdapters:
        raise newException(ValueError, "[infer] checkpoint co LoRA adapter ngoai pattern: " & adapterName)
      if adapterName notin lo.adapters:
        let adIn = parseInt(meta["adapter." & adapterName & ".inF"])
        let adOut = parseInt(meta["adapter." & adapterName & ".outF"])
        let adRank = parseInt(meta["adapter." & adapterName & ".r"])
        let adAlpha = parseFloat(meta["adapter." & adapterName & ".alpha"]).float32
        if adIn <= 0 or adOut <= 0 or adRank <= 0:
          raise newException(ValueError, "[infer] LoRA co inF/outF/r khong hop le: " & adapterName)
        var ad: LoraAdapter
        ad.inF = adIn; ad.outF = adOut; ad.r = adRank
        ad.alpha = adAlpha; ad.scale = adAlpha / float32(adRank)
        ad.A = Tensor(shape: @[adRank, adIn]); ad.B = Tensor(shape: @[adOut, adRank])
        lo.adapters[adapterName] = ad
      if part == "A":
        if adapterName in seenAdapterA:
          raise newException(ValueError, "[infer] checkpoint trung LoRA tensor A: " & adapterName)
        if tensor.shape != lo.adapters[adapterName].A.shape or tensor.floatValues.len != lo.adapters[adapterName].r * lo.adapters[adapterName].inF:
          raise newException(ValueError, "[infer] LoRA A sai shape/data: " & adapterName)
        lo.adapters[adapterName].A.data = tensor.data
        lo.adapters[adapterName].mappedA = tensor.mappedData
        seenAdapterA[adapterName] = true
      elif part == "B":
        if adapterName in seenAdapterB:
          raise newException(ValueError, "[infer] checkpoint trung LoRA tensor B: " & adapterName)
        if tensor.shape != lo.adapters[adapterName].B.shape or tensor.floatValues.len != lo.adapters[adapterName].r * lo.adapters[adapterName].outF:
          raise newException(ValueError, "[infer] LoRA B sai shape/data: " & adapterName)
        lo.adapters[adapterName].B.data = tensor.data
        lo.adapters[adapterName].mappedB = tensor.mappedData
        seenAdapterB[adapterName] = true
      else:
        raise newException(ValueError, "[infer] checkpoint LoRA part la: " & part)
  for name, _ in expectedNewBlocks:
    if name notin seenNewBlocks:
      raise newException(ValueError, "[infer] checkpoint thieu newBlock weight: " & name)
  for name, ad in lo.adapters:
    if name notin seenAdapterA or name notin seenAdapterB:
      raise newException(ValueError, "[infer] checkpoint thieu LoRA A/B: " & name)
  for name, _ in requiredAdapters:
    if name notin lo.adapters:
      raise newException(ValueError, "[infer] checkpoint thieu LoRA adapter bat buoc: " & name)
  for name, _ in lo.adapters:
    if name notin requiredAdapters:
      raise newException(ValueError, "[infer] checkpoint co LoRA adapter ngoai pattern: " & name)
  lo.quantBase = quantResident
  # Adapter A/B toan 0 = delta 0 chinh xac -> bo (khong doi ket qua, bot GEMM LoRA o prefill
  # va stage LoRA trong one-dispatch). BYBY_KEEP_ZERO_LORA=1 de giu lai.
  if getEnv("BYBY_KEEP_ZERO_LORA", "0") == "0":
    let dropped = dropZeroLoraAdapters(lo)
    if dropped.len > 0:
      echo &"[infer] Bo {dropped.len} LoRA adapter co A/B toan 0 (delta = 0, khong doi ket qua); " &
           &"con {lo.adapters.len} adapter co tac dung."
  # The optional bounded merge runs after packed upload, directly into the
  # existing GPU buffers. Never materialize full dense FP32 weight matrices.
  # FIX IndexDefect: bind quantBase → Linear.weightQ/useQuant TRƯỚC generate.
  # (NIMBIN4 + quantResident: weight.data đã @[], phải nối weightQ trước
  # mergeInUploadStreamingJamba / uploadResident, kể cả block A linear.)
  bindQuantBaseToLinears(m, lo)
  echo &"[infer] Da load checkpoint {path}: {lo.adapters.len} LoRA adapter (block R) + " &
       &"{newBlockQuant.len} newBlock quant (Attention train full). " &
       "Base Mamba lay tu .nimq, khong nam trong checkpoint."
  echo "[infer]   ('LoRA-only' = khong luu base Mamba; van co day du weight Attention trong newBlock.)"
  result = (m, lo, tok)

proc isRawBinCheckpoint(path: string): bool =
  if not fileExists(path): return false
  var f: File
  if not open(f, path, fmRead): return false
  var magic = newString(8)
  let got = f.readBuffer(addr magic[0], 8)
  f.close()
  # Chấp nhận cả NIMBIN1 (fp32) và NIMBIN4 (QLoRA 4-bit)
  result = got == 8 and (magic == RAWBIN_MAGIC or magic == "NIMBIN4\0")

# ═══════════════════════════════════════════════════════════════
# SAMPLING - VIET LAI HOAN TOAN.
#
# Ban CU (sampleFromLogits) chi co temperature tren TOAN BO 32768 logit:
# khong top-p/top-k/min-p, khong chan special token, khong guard NaN. Ba
# nguon token rac tu day:
#
#  (a) DUOI PHAN PHOI. ~32k token con lai moi cai p ~ 1e-6, cong lai thuong
#      chiem 0.5-3% khoi luong xac suat. O T=0.7, trung binh moi ~30-60
#      token ban se boc trung 1 token HOAN TOAN ngau nhien. Voi Mamba thi
#      nang hon Transformer nhieu: token rac bi nuot vao STATE hoi quy,
#      khong co cach nao "quen" di - no lam hong toan bo phan sinh ve sau.
#      -> chan bang min-p + top-p.
#
#  (b) SPECIAL TOKEN. tekken danh ~1000 id DAU cho control + <SPECIAL_k>
#      placeholder chua dung bao gio; hang lm_head cua chung gan nhu khong
#      duoc train nen logit la nhieu thuan, nhung VAN nam trong softmax va
#      VAN sample duoc. Cong them lm_head bi quantize FP4 (xem
#      quantize_codestral.nim - Fp32NameHints co "lm_head.bias" nhung CO Y
#      khong co "lm_head.weight") thi sai so luong tu du de day chung len
#      tren. -> chan bang `banned` (tru cac stop token, vi ta MUON model
#      phat duoc stop token de con dung lai).
#
#  (c) NaN. Ban cu: 1 logit NaN -> total=NaN -> r=NaN -> dieu kien
#      "acc >= r" SAI o MOI vong lap -> return probs.len-1 -> LUON ra dung
#      id cuoi vocab (32767), im lang tuyet doi, khong 1 dong log. Voi fp4
#      + Metal tren GPU tich hop thi NaN/Inf khong he hiem.
#      -> guard + dem + canh bao 1 lan.
# ═══════════════════════════════════════════════════════════════

type
  SamplerCfg* = object
    temperature*: float32
    topP*: float32          # 0 hoac >=1 = tat
    minP*: float32          # nguong tuong doi so voi token tot nhat; 0 = tat
    repeatPenalty*: float32 # 1.0 = tat
    repeatWindow*: int

proc envF32(name: string, dflt: float32): float32 =
  let raw = getEnv(name, "")
  if raw.len == 0: return dflt
  try:
    return parseFloat(raw).float32
  except ValueError:
    stderr.writeLine "[infer] " & name & "='" & raw & "' khong phai so - dung mac dinh"
    return dflt

proc envInt(name: string, dflt: int): int =
  let raw = getEnv(name, "")
  if raw.len == 0: return dflt
  try:
    return parseInt(raw)
  except ValueError:
    stderr.writeLine "[infer] " & name & "='" & raw & "' khong phai so - dung mac dinh"
    return dflt

proc defaultSamplerCfg*(temperature: float32): SamplerCfg =
  ## Chinh qua bien moi truong, khong can build lai:
  ##   BYBY_TOP_P=0.9  BYBY_MIN_P=0.05  BYBY_REPEAT_PENALTY=1.1
  ##   BYBY_REPEAT_WINDOW=64
  result.temperature = temperature
  result.topP = envF32("BYBY_TOP_P", 0.9'f32)
  result.minP = envF32("BYBY_MIN_P", 0.05'f32)
  result.repeatPenalty = envF32("BYBY_REPEAT_PENALTY", 1.1'f32)
  result.repeatWindow = envInt("BYBY_REPEAT_WINDOW", 64)

const NegInfF32 = -3.0e38'f32

var gNanWarned = false

proc sampleToken(logits: seq[float32], vocab: int, cfg: SamplerCfg,
                 banned: seq[bool], noPenalty: seq[bool],
                 recent: openArray[int], rng: var Rand): int =
  if vocab <= 0 or logits.len < vocab:
    raise newException(ValueError,
      "[infer] sampleToken: logits.len=" & $logits.len & " < vocab=" & $vocab)
  var maxL = NegInfF32
  var argMaxI = -1
  var nBad = 0
  var adj = newSeq[float32](vocab)
  for i in 0 ..< vocab:
    var v = logits[i]
    if v.classify in {fcNan, fcInf, fcNegInf}:
      adj[i] = NegInfF32
      inc nBad
      continue
    if cfg.repeatPenalty != 1.0'f32 and recent.len > 0:
      if not (noPenalty.len == vocab and noPenalty[i]):
        for rid in recent:
          if rid == i:
            if v > 0'f32: v = v / cfg.repeatPenalty
            else: v = v * cfg.repeatPenalty
            break
    if banned.len == vocab and banned[i]:
      adj[i] = NegInfF32
      continue
    adj[i] = v
    if v > maxL:
      maxL = v
      argMaxI = i
  if nBad > 0 and not gNanWarned:
    gNanWarned = true
    stderr.writeLine "[infer] CANH BAO: " & $nBad & "/" & $vocab & " logit NaN/Inf"
  if argMaxI < 0:
    raise newException(ValueError, "[infer] moi token bi chan/NaN")
  if cfg.temperature <= 0'f32:
    return argMaxI
  var probs = newSeq[float32](vocab)
  var total = 0'f32
  for i in 0 ..< vocab:
    if adj[i] <= NegInfF32:
      probs[i] = 0'f32
    else:
      let e = exp((adj[i] - maxL) / cfg.temperature)
      probs[i] = e
      total += e
  if total <= 0'f32 or total.classify in {fcNan, fcInf}:
    return argMaxI
  var idx = newSeqOfCap[int](256)
  for i in 0 ..< vocab:
    if probs[i] > 0'f32 and probs[i] >= cfg.minP:
      idx.add(i)
  if idx.len == 0: return argMaxI
  if idx.len == 1: return idx[0]
  if cfg.topP > 0'f32 and cfg.topP < 1'f32 and idx.len > 1:
    idx.sort(proc(a, b: int): int = cmp(probs[b], probs[a]))
    var sumAll = 0'f32
    for i in idx: sumAll += probs[i]
    var acc = 0'f32
    var cut = idx.len
    for k in 0 ..< idx.len:
      acc += probs[idx[k]]
      if acc >= cfg.topP * sumAll:
        cut = k + 1
        break
    idx.setLen(max(cut, 1))
  var tot2 = 0'f32
  for i in idx: tot2 += probs[i]
  let r = float32(rng.rand(1.0)) * tot2
  var acc2 = 0'f32
  for i in idx:
    acc2 += probs[i]
    if acc2 >= r: return i
  return idx[^1]


proc quantShapeMatches(lin: Linear; qt: QuantTensor): bool =
  qt.shape.len == 2 and qt.shape[0] == lin.outF and qt.shape[1] == lin.inF and
    qt.bytes.len > 0

proc sequenceLinearReady(lin: Linear; name: string; lora: JambaLora): bool =
  if lin.residentReady or lin.residentFp32Ready: return true
  if lin.useQuant:
    # INT4: beMatmulQ4. FP4-KM: resident GPU + MBLOCK=4 (fp4GemmGpuRows) sau
    # prepareGpuModel — khong dequant host.
    if lin.weightQ.kind == qkInt4Asymmetric and quantShapeMatches(lin, lin.weightQ):
      return true
    if lin.weightQ.kind == qkFp4KM and lin.gpuCacheName.len > 0 and
       lin.gpuCacheName in gFp4CacheGpu and
       gFp4CacheGpu[lin.gpuCacheName].residentReady:
      return true
    if name in lora.quantBase:
      let qt = lora.quantBase[name]
      if qt.kind == qkInt4Asymmetric and quantShapeMatches(lin, qt): return true
      if qt.kind == qkFp4KM and name in gFp4CacheGpu and
         gFp4CacheGpu[name].residentReady:
        return true
    return false
  result = lin.weight.data.len == lin.inF * lin.outF

proc sequenceBlockWeightsReady(m: NimformerModelJamba; lora: JambaLora;
                               blockIdx: int): bool =
  let root = "block" & $blockIdx
  let blk = m.blocks[blockIdx]
  if blk.kind notin {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}: return false
  if not sequenceLinearReady(blk.mambaReal.inProj,
      root & ".mambaReal.inProj.weight", lora): return false
  if not sequenceLinearReady(blk.mambaReal.outProj,
      root & ".mambaReal.outProj.weight", lora): return false
  if blk.kind in {bkMambaRealPlusAttn, bkMambaRealGated}:
    if not sequenceLinearReady(blk.attn.qkvProj,
        root & ".attn.qkvProj.weight", lora): return false
    if not sequenceLinearReady(blk.attn.outProj,
        root & ".attn.outProj.weight", lora): return false
  if blk.kind == bkMambaRealGated and
     not sequenceLinearReady(blk.fuseGate, root & ".fuseGate.weight", lora): return false
  true

proc restoreSequenceWeight(lin: var Linear; name: string;
                           lora: JambaLora): QuantTensor =
  ## prepareGpuModel uploads packed weights to the recurrent GEMV cache and
  ## drops the Linear's host copy. Temporarily borrow the existing quantBase
  ## payload for batched sequence GEMM. Retain the mapping owner as well as
  ## metadata, then restore the prior tensor immediately after the block.
  result = lin.weightQ
  if lin.useQuant and result.bytes.len == 0 and not lin.residentReady and
     not lin.residentFp32Ready and name in lora.quantBase:
    lin.weightQ = lora.quantBase[name]

proc forwardRSequenceBlock(m: var NimformerModelJamba; lora: JambaLora;
                           blockIdx: int; x: Tensor; ctx: Backend):
    tuple[y: Tensor, cache: HybridBlockKernelCache] =
  let root = "block" & $blockIdx & ".mambaReal."
  let oldIn = restoreSequenceWeight(m.blocks[blockIdx].mambaReal.inProj,
    root & "inProj.weight", lora)
  let oldOut = restoreSequenceWeight(m.blocks[blockIdx].mambaReal.outProj,
    root & "outProj.weight", lora)
  try:
    # In inference, prepareGpuModel seeds the recurrent decoder from ln1.gamma.
    # The parallel prefix must use that exact same pretrained gamma; rms1.gamma
    # may still be the constructor's all-ones value in an inference model.
    result = forwardRBlockKernel(m.blocks[blockIdx].mambaReal, x,
      m.blocks[blockIdx].ln1.gamma, m.blocks[blockIdx].rms1.eps, ctx,
      saveBackwardCache = false, exportTerminalState = true)
  finally:
    m.blocks[blockIdx].mambaReal.inProj.weightQ = oldIn
    m.blocks[blockIdx].mambaReal.outProj.weightQ = oldOut

proc forwardPSequenceBlock(m: var NimformerModelJamba; lora: JambaLora;
                           blockIdx: int; x: Tensor; ctx: Backend):
    tuple[y: Tensor, cache: HybridBlockKernelCache] =
  let root = "block" & $blockIdx
  let oldIn = restoreSequenceWeight(m.blocks[blockIdx].mambaReal.inProj,
    root & ".mambaReal.inProj.weight", lora)
  let oldOut = restoreSequenceWeight(m.blocks[blockIdx].mambaReal.outProj,
    root & ".mambaReal.outProj.weight", lora)
  let oldQkv = restoreSequenceWeight(m.blocks[blockIdx].attn.qkvProj,
    root & ".attn.qkvProj.weight", lora)
  let oldAttnOut = restoreSequenceWeight(m.blocks[blockIdx].attn.outProj,
    root & ".attn.outProj.weight", lora)
  var oldGate: QuantTensor
  var gatePtr: ptr Linear = nil
  if m.blocks[blockIdx].kind == bkMambaRealGated:
    oldGate = restoreSequenceWeight(m.blocks[blockIdx].fuseGate,
      root & ".fuseGate.weight", lora)
    gatePtr = addr m.blocks[blockIdx].fuseGate
  try:
    # Match both recurrent branches, which are initialized with ln1.gamma.
    result = forwardPBlockKernel(m.blocks[blockIdx].mambaReal,
      m.blocks[blockIdx].attn, x, m.blocks[blockIdx].ln1.gamma,
      m.blocks[blockIdx].rms1.eps, ctx, saveBackwardCache = false,
      exportTerminalState = true, fuseGate = gatePtr)
  finally:
    m.blocks[blockIdx].mambaReal.inProj.weightQ = oldIn
    m.blocks[blockIdx].mambaReal.outProj.weightQ = oldOut
    m.blocks[blockIdx].attn.qkvProj.weightQ = oldQkv
    m.blocks[blockIdx].attn.outProj.weightQ = oldAttnOut
    if m.blocks[blockIdx].kind == bkMambaRealGated:
      m.blocks[blockIdx].fuseGate.weightQ = oldGate

proc importParallelPrefillState(ctx: Backend; blockIdx: int;
                                cache: HybridBlockKernelCache;
                                attnHeads = 0; attnHeadDim = 0) =
  ## Transfer the scan's terminal Mamba2/Linear++ states into the persistent
  ## buffers used by one-token decode. Linear++ scan stores S as [head,value,key],
  ## while the recurrent kernel consumes [head,key,value], so transpose here.
  when not defined(noOneDispatch):
    # Block R thuoc one-dispatch: state conv/ssm nam trong pack OD -> nhap vao do (khong phai layer cu).
    if gOdActive and blockIdx >= 0 and blockIdx < gOdState.slotOfBlock.len and
       gOdState.slotOfBlock[blockIdx] >= 0:
      let sh = gOdState.sh
      if cache.terminalConvState.len != sh.cDim * (sh.K - 1) or
         cache.terminalSsmState.len != sh.HD * sh.N:
        raise newException(CatchableError,
          "parallel prefill: terminal Mamba2 state shape mismatch (one-dispatch) at block " & $blockIdx)
      let beOd = ctx.kind.toByby()
      var oConv, oSsm: SessionHandle
      var haveOC, haveOS, openOd = false
      try:
        oConv = sessionUpload(beOd, cache.terminalConvState)
        haveOC = true
        oSsm = sessionUpload(beOd, cache.terminalSsmState)
        haveOS = true
        if not sessionBegin(beOd):
          raise newException(CatchableError, "parallel prefill: one-dispatch state-import sessionBegin failed")
        openOd = true
        let okOd = odImportState(gOdState, gOdState.slotOfBlock[blockIdx], oConv, oSsm)
        let endedOd = sessionEnd(beOd)
        openOd = false
        if not okOd or not endedOd:
          discard sessionRecoverQueue(beOd)
          raise newException(CatchableError,
            "parallel prefill: one-dispatch terminal-state import failed at block " & $blockIdx)
      finally:
        if openOd:
          discard sessionEnd(beOd)
          discard sessionRecoverQueue(beOd)
        if haveOC: sessionFree(beOd, oConv)
        if haveOS: sessionFree(beOd, oSsm)
      return
  if blockIdx notin gLayersV2:
    raise newException(CatchableError,
      "parallel prefill: Mamba2 persistent layer is not initialized: " & $blockIdx)
  var layer = gLayersV2[blockIdx]
  let convN = layer.cDim * (layer.K - 1)
  let ssmN = layer.HD * layer.N
  if cache.terminalConvState.len != convN or cache.terminalSsmState.len != ssmN:
    raise newException(CatchableError,
      "parallel prefill: terminal Mamba2 state shape mismatch at block " & $blockIdx)

  var zState, sState: seq[float32]
  var attnLayer: GpuAttnLayer
  if attnHeads > 0:
    if blockIdx notin gAttnLayers:
      raise newException(CatchableError,
        "parallel prefill: Linear++ persistent layer is not initialized: " & $blockIdx)
    attnLayer = gAttnLayers[blockIdx]
    let stateN = attnHeads * attnHeadDim * attnHeadDim
    if attnLayer.lay.H != attnHeads or attnLayer.lay.D != attnHeadDim or
       cache.linearPlusZState.len != attnHeads * attnHeadDim or
       cache.linearPlusScanState.len != stateN:
      raise newException(CatchableError,
        "parallel prefill: terminal Linear++ state shape mismatch at block " & $blockIdx)
    zState = cache.linearPlusZState
    sState = newSeq[float32](stateN)
    for h in 0 ..< attnHeads:
      for value in 0 ..< attnHeadDim:
        for key in 0 ..< attnHeadDim:
          let scanIdx = (h * attnHeadDim + value) * attnHeadDim + key
          let stepIdx = (h * attnHeadDim + key) * attnHeadDim + value
          sState[stepIdx] = cache.linearPlusScanState[scanIdx]

  let be = ctx.kind.toByby()
  var hConv, hSsm: SessionHandle
  var hZ: SessionHandle
  var hS: SessionHandle
  var haveConv, haveSsm, haveZ, haveS = false
  var sessionOpen = false
  try:
    hConv = sessionUpload(be, cache.terminalConvState)
    haveConv = true
    hSsm = sessionUpload(be, cache.terminalSsmState)
    haveSsm = true
    if attnHeads > 0:
      hZ = sessionUpload(be, zState)
      haveZ = true
      hS = sessionUpload(be, sState)
      haveS = true
    if not sessionBegin(be):
      raise newException(CatchableError,
        "parallel prefill: state-import sessionBegin failed")
    sessionOpen = true
    var hMamba = layer.hWork
    var ok = sessionCopyInto(be, hConv, hMamba, 0,
      layer.off.offConvSt, convN)
    ok = sessionCopyInto(be, hSsm, hMamba, 0,
      layer.off.offSsm, ssmN) and ok
    if attnHeads > 0:
      var hAttn = attnLayer.hWork
      ok = sessionCopyInto(be, hZ, hAttn, 0,
        attnLayer.lay.offZ, zState.len) and ok
      ok = sessionCopyInto(be, hS, hAttn, 0,
        attnLayer.lay.offS, sState.len) and ok
    let ended = sessionEnd(be)
    sessionOpen = false
    if not ok or not ended:
      discard sessionRecoverQueue(be)
      raise newException(CatchableError,
        "parallel prefill: GPU terminal-state import failed at block " & $blockIdx)
  finally:
    if sessionOpen:
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
    if haveConv: sessionFree(be, hConv)
    if haveSsm: sessionFree(be, hSsm)
    if haveZ: sessionFree(be, hZ)
    if haveS: sessionFree(be, hS)

proc isGpuTimeoutMessage(msg: string): bool =
  let lower = msg.toLowerAscii()
  "timeout" in lower or "timed out" in lower

proc parallelPrefillFallbackReason(m: NimformerModelJamba; lora: JambaLora;
                                   tokenIds: seq[int]; ctx: Backend): string =
  if getEnv("BYBY_PREFILL_PARALLEL", "1") == "0":
    return "parallel prefill disabled by BYBY_PREFILL_PARALLEL=0"
  if not isGpuCtx(ctx): return "GPU backend is required"
  if tokenIds.len < 2: return "fewer than two prefix tokens"
  # The resident prefix avoids the host-transfer cost of the legacy scan and
  # can batch even short prompts. Keep the old threshold for the old path.
  let defaultMin = if getEnv("BYBY_PREFILL_RESIDENT", "1") != "0" and
                      residentPrefillReason(m, ctx).len == 0: 2 else: 32
  let parMinT = (try: parseInt(getEnv("BYBY_PREFILL_PAR_MIN_T", $defaultMin)) except ValueError: defaultMin)
  if tokenIds.len < parMinT:
    return "prompt " & $tokenIds.len & " token < BYBY_PREFILL_PAR_MIN_T=" & $parMinT
  if getEnv("BYBY_LP_SCAN_SERIAL") == "1": return "Linear++ serial override is enabled"
  if gRNormOverride != 0 or gRNormGroupsOverride == 1 or gRmsUnitGamma:
    return "the selected legacy/LayerNorm preset is not supported by the RMS scan block"
  if gSkipAttn: return "Linear++ sidecar is disabled"
  # The resident inference path chunks long prompts and carries R/P/G states.
  if getEnv("BYBY_PREFILL_RESIDENT", "1") != "0" and residentPrefillReason(m,ctx).len == 0:
    return ""
  if tokenIds.len > 64: return "legacy sequence scan threadgroup limit is 64 tokens"
  if m2ScanTgSize(tokenIds.len) == 0:
    return "Mamba2 parallel scan is unavailable for this sequence length/device"
  var hasP = false
  for i, blk in m.blocks:
    if blk.kind notin {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}:
      return "block " & $i & " is not an R/P sequence block"
    # The recurrent decoder may apply the diagnostic env override to the
    # gated RMSNorm group count. The sequence block kernel currently uses the
    # model's native nGroups; do not mix those two equations in a prefix.
    if gNormGroups(blk.mambaReal.nGroups) != blk.mambaReal.nGroups:
      return "runtime gated-RMSNorm groups differ from the sequence scan at block " & $i
    if blk.kind in {bkMambaRealPlusAttn, bkMambaRealGated}: hasP = true
    if not sequenceBlockWeightsReady(m, lora, i):
      return "sequence weights unavailable in block " & $i
  if hasP and lpScanTgSize(tokenIds.len) == 0:
    return "Linear++ parallel scan is unavailable for this sequence length/device"

proc forwardParallelPrefill(m: var NimformerModelJamba; lora: JambaLora;
                            tokenIds: seq[int]; ctx: Backend) =
  pfReset()
  gPfActive = true
  let pfWall0 = getMonoTime()
  defer:
    gPfActive = false
    if gPfEnabled:
      pfReport(float((getMonoTime() - pfWall0).inNanoseconds) / 1e6)
  if getEnv("BYBY_PREFILL_RESIDENT", "1") != "0" and residentPrefillReason(m, ctx).len == 0:
    let batchSize=envInt("BYBY_PREFILL_BATCH_SIZE",128)
    if batchSize notin 2..128:
      raise newException(ValueError,"BYBY_PREFILL_BATCH_SIZE must be 2..128")
    var offset=0
    while offset < tokenIds.len:
      let count=min(batchSize,tokenIds.len-offset)
      if count == 1:
        discard forwardStepJambaGpuRetry(m,tokenIds[offset],tokenIds,offset,ctx,produceLogits=false)
      else:
        forwardResidentPrefill(m,tokenIds[offset..<offset+count],ctx,continuation=offset>0)
        markGpuPrefillReady()
      offset += count
    return
  pfTime("warmup+reset"):
    warmupGpuForParallelPrefill(m, ctx)
  var x = pfExpr("embed", m.embed.lookupBatch(@[tokenIds], ctx))
  var pCount = 0
  for i in 0 ..< m.blocks.len:
    if m.blocks[i].kind in {bkMambaRealPlusAttn, bkMambaRealGated}:
      let fw = pfExpr("block.P(total)", forwardPSequenceBlock(m, lora, i, x, ctx))
      pfTime("importState"):
        importParallelPrefillState(ctx, i, fw.cache,
          m.blocks[i].attn.nHeads, m.blocks[i].attn.headDim)
      x = fw.y
      inc pCount
    else:
      let fw = pfExpr("block.R(total)", forwardRSequenceBlock(m, lora, i, x, ctx))
      pfTime("importState"):
        importParallelPrefillState(ctx, i, fw.cache)
      x = fw.y
  echo &"[infer] Parallel prefill OK: Mamba2 scan" &
       (if pCount > 0: " + Linear++ scan" else: "") &
       &", T={tokenIds.len}, " &
       &"R/P blocks={m.blocks.len}, P sidecars={pCount}; terminal recurrent states imported."

proc forwardPrefillBatch*(m: var NimformerModelJamba; lora: JambaLora;
                          tokenIds: seq[int]; ctx: Backend) =
  ## Prefill R/P prefixes with sequence-parallel scans, then import each
  ## terminal state into the recurrent decode buffers. Decode remains one-token
  ## causal. Unsupported presets/blocks use the existing fallback; resident R/P/G
  ## prefixes use compact inference scans and GPU MPS tiles in chunks up to 128.
  if tokenIds.len == 0: return
  if isGpuCtx(ctx): ensureTgCap(ctx)
  let fallbackReason = parallelPrefillFallbackReason(m, lora, tokenIds, ctx)
  if fallbackReason.len == 0:
    try:
      forwardParallelPrefill(m, lora, tokenIds, ctx)
      return
    except CatchableError as e:
      # A failure during imports can leave some earlier blocks advanced.
      # Always return the whole model to a clean recurrent state first.
      resetGpuGenState(m, ctx)
      if not isGpuTimeoutMessage(e.msg): raise
      stderr.writeLine("[infer] parallel prefill GPU timeout; resetting state and " &
        "falling back to tokenwise prefill: " & e.msg)
  else:
    echo "[infer] Parallel prefill unavailable (" & fallbackReason & "); using tokenwise prefill."
  var prefixPath = "independent kernels, one session/token"
  when not defined(noOneDispatch):
    if gOdActive: prefixPath = "R one-dispatch segments + gated blocks"
  echo "[infer] recurrent prefill: " & prefixPath & "; no intermediate logits."
  for i, tid in tokenIds:
    if i == 0 or i == tokenIds.high or (i + 1) mod 16 == 0:
      echo "[infer] prefill token " & $(i + 1) & "/" & $tokenIds.len &
        " id=" & $tid
    discard forwardStepJambaGpuRetry(m, tid, tokenIds, i, ctx, produceLogits = false)
  echo "[infer] Prefill xong " & $tokenIds.len & " token -> bat dau sinh."

proc forwardPrefillBatch*(m: var NimformerModelJamba; tokenIds: seq[int];
                          ctx: Backend) =
  ## Norm-sweep diagnostics intentionally do not receive the checkpoint's
  ## quantBase table, so keep their prior recurrent reference path.
  for i, tid in tokenIds:
    discard forwardStepJambaGpuRetry(m, tid, tokenIds, i, ctx, produceLogits = false)

# ═══════════════════════════════════════════════════════════════════
# Decode STREAM dung dau cach + chan doan token / norm preset
# ═══════════════════════════════════════════════════════════════════
type
  StreamDecoder = object
    baseIds: seq[int]    ## token ngu canh (token cuoi cua prompt)
    baseText: string
    emitted: string      ## phan text da in ra

proc initStreamDecoder(tok: AnyTokenizer, lastPromptId: int): StreamDecoder =
  result.baseIds = @[lastPromptId]
  result.baseText = decodeAny(tok, result.baseIds)
  result.emitted = ""

proc streamDelta(sd: var StreamDecoder, tok: AnyTokenizer, genIds: seq[int],
                 final = false): string =
  ## SUA: truoc day moi token decode RIENG (decodeAny(tok, @[id])). Voi
  ## SentencePiece (tokenizer.model.v3) decode([ "▁Hello" ]) = "Hello" - bi BO
  ## dau cach dau => cac tu dinh lien nhau, va token chi la "▁" thanh "" (khong
  ## in gi). Gio decode CA chuoi (token cuoi prompt + cac token da sinh) roi chi
  ## in phan moi => dau cach giua cac tu dung. Byte UTF-8 chua du (U+FFFD o cuoi)
  ## thi cho token sau.
  let full = decodeAny(tok, sd.baseIds & genIds)
  # `baseText` phai la prefix cua `full`. Neu KHONG (tokenizer decode khong
  # don dieu theo prefix), KHONG in lai toan bo `full` - lam vay se lap lai
  # phan da in ra man hinh (nhin nhu rac). Neu CHUA in gi (emitted rong) thi
  # dung `full` nhu cu; neu da in roi thi bo qua token nay de tranh lap.
  var cur: string
  if full.startsWith(sd.baseText):
    cur = full[sd.baseText.len .. ^1]
  elif sd.emitted.len == 0:
    cur = full
  else:
    return ""
  var cp = 0
  let lim = min(cur.len, sd.emitted.len)
  while cp < lim and cur[cp] == sd.emitted[cp]: inc cp
  if cp >= cur.len: return ""
  let delta = cur[cp .. ^1]
  if (not final) and delta.endsWith("\xEF\xBF\xBD"): return ""
  sd.emitted = cur
  result = delta

proc printTopK(lg: seq[float32], vocab: int, banned: seq[bool], tok: AnyTokenizer, k = 5) =
  var idx = toSeq(0 ..< vocab)
  idx.sort(proc(a, b: int): int = cmp(lg[b], lg[a]))
  for r in 0 ..< min(k, idx.len):
    let tid = idx[r]
    let txt = decodeAny(tok, @[tid])
    stderr.writeLine &"      #{r}: id={tid} logit={lg[tid]:.3f} banned={banned[tid]} text={txt.escape}"

proc logitSummary(lg: seq[float32], vocab: int, banned: seq[bool]): string =
  var n = 0
  var sm = 0.0
  var sq = 0.0
  var mx = -1.0e30
  var bad = 0
  for i in 0 ..< min(vocab, lg.len):
    let v = lg[i]
    if v.classify in {fcNan, fcInf, fcNegInf}:
      inc bad
      continue
    if banned.len == vocab and banned[i]: continue
    let d = float(v)
    sm += d
    sq += d * d
    if d > mx: mx = d
    inc n
  let mean = sm / float(max(1, n))
  let sd = sqrt(max(0.0, sq / float(max(1, n)) - mean * mean))
  result = &"mean={mean:.3f} std={sd:.3f} max={mx:.3f} NaN/Inf={bad}"

proc printBlockTable(stats: seq[BlockStat], full: bool) =
  if full:
    stderr.writeLine "      blk kind      rms      max     mean  NaN/Inf  | residual-vao  sidecar rms/max"
    for st in stats:
      stderr.writeLine &"      {st.idx:>3}    {st.kind}  {st.rms:>9.3f} {st.maxAbs:>8.1f} {st.mean:>8.3f}  {st.nonFinite}" &
        (if st.sideRms >= 0: &"  | {st.inRms:>9.3f}  {st.sideRms:>9.3f}/{st.sideMax:.2f}" else: "")
  else:
    var parts: seq[string]
    for st in stats:
      if st.idx in [0, 3, 4, 9, 19, 29, 39, 49, 59, 63] or st.nonFinite > 0:
        parts.add &"b{st.idx}{st.kind}={st.rms:.2f}/{st.maxAbs:.0f}" &
                  (if st.nonFinite > 0: "!NaN" else: "")
    stderr.writeLine "      residual rms/max: " & parts.join(" ")
    # Dong gop cua sidecar Linear++ (khoi P) so voi residual vao khoi: neu sidecar >> residual
    # thi dau vao (norm) cua nhanh nay LECH luc train - day la dau hieu "hybrid ra rac".
    var sp: seq[string]
    for st in stats:
      if st.kind == 'P' and st.sideRms >= 0:
        sp.add &"b{st.idx}={st.sideRms:.3f}/{st.sideMax:.2f}(res {st.inRms:.3f})"
    if sp.len > 0:
      stderr.writeLine "      sidecar Linear++ rms/max (res = residual rms vao khoi): " & sp.join(" ")

proc teacherForced(m: var NimformerModelJamba, tok: AnyTokenizer, promptText, fullText: string,
                   vocab: int, ctx: Backend): tuple[nllPrompt, nllAnswer: float] =
  ## Cho model doc 1 cau MAU dung dinh dang train ("[INST] ..[/INST] tra loi") va do
  ## NLL tung token (teacher forcing). Model tot: NLL tra loi ~1-3; ngau nhien =
  ## ln(32768)=10.4; chi dung phan bo chung (unigram) ~ 7.
  let pIds = encodeAny(tok, promptText)
  let ids = encodeAny(tok, fullText)
  var bnd = pIds.len
  if ids.len <= bnd or ids[0 ..< bnd] != pIds: bnd = max(1, ids.len div 2)
  resetGpuGenState(m, ctx)
  var nllP, nllR = 0.0
  var nP, nR, hitsR = 0
  for t in 0 ..< ids.len - 1:
    let lg = forwardStepJambaGpuRetry(m, ids[t], ids, t, ctx).data
    var mx = -1.0e30
    for i in 0 ..< vocab:
      let v = float(lg[i])
      if v == v and v > mx: mx = v
    var se = 0.0
    var top1 = 0
    var best = -1.0e30
    for i in 0 ..< vocab:
      let v = float(lg[i])
      if v != v: continue
      se += exp(v - mx)
      if v > best:
        best = v
        top1 = i
    let tgt = ids[t + 1]
    let nll = mx + ln(se) - float(lg[tgt])
    var rank = 0
    for i in 0 ..< vocab:
      if lg[i] > lg[tgt]: inc rank
    if t + 1 >= bnd:
      nllR += nll
      inc nR
      if top1 == tgt: inc hitsR
      stderr.writeLine &"      [tf] dung={pieceAny(tok, tgt).escape} doan={pieceAny(tok, top1).escape} " &
                       &"nll={nll:.2f} hang={rank}"
    else:
      if t >= 1:
        nllP += nll
        inc nP
  stderr.writeLine &"      NLL phan hoi={nllP / float(max(1, nP)):.2f} | NLL tra loi={nllR / float(max(1, nR)):.2f} " &
                   &"| doan dung token tra loi {hitsR}/{nR}   (ngau nhien=10.40, phan bo chung~7)"
  result = (nllP / float(max(1, nP)), nllR / float(max(1, nR)))

type SweepMode = object
  name: string
  skipA, noLora: bool
  preset: string

proc runBisect(m: var NimformerModelJamba, tok: AnyTokenizer, vocab: int,
               banned, noPenalty: seq[bool], cfgBase: SamplerCfg, ctx: Backend): string =
  ## BYBY_NORM_SWEEP (opt-in): chia doi de tim cho hong, NGAY TRONG 1 lan
  ## chay (load model rat lau). Moi che do: (1) NLL teacher-forcing tren cau mau
  ## dung dinh dang train, (2) greedy sinh chu tu cau hoi ro rang, (3) residual.
  ##   base-hf        : bo khoi A + tat LoRA = Mamba-Codestral GOC  (phai noi ra chu)
  ##   base+lora-*    : bo khoi A, giu LoRA
  ##   hybrid-*       : day du R + A + LoRA
  ## Hau to cua che do = norm cua trainer da sinh ra checkpoint:
  ##   *-train : rms-fin-gn      trainer HIEN TAI (RMSNorm gamma base + norm_f, da dong bo
  ##                             gamma - jamba_hybrid.syncRmsFromBaseNorms)
  ##   *-unit  : rms-fin-g1-unit trainer CU: rms1/rmsF giu gamma=1 (loadNormsFromHfTensors
  ##                             khong bao gio duoc goi), LoRA + sidecar hoc tren dau vao do
  ##   *-ln    : ln-nof-g1       trainer CU NHAT: LayerNorm + beta, khong norm_f
  ## Tra ve preset de xuat (""= giu mac dinh): preset cua che do hybrid co NLL tra loi thap
  ## nhat, CHI khi no tot ro ret (NLL < 6 va thap hon preset mac dinh >= 1 nat) - de checkpoint
  ## cu van sinh duoc chu ma khong phai chay lai sweep chi de doc ket qua.
  let allModes = @[
    SweepMode(name: "base-hf", skipA: true, noLora: true, preset: "hf"),
    SweepMode(name: "base+lora-train", skipA: true, noLora: false, preset: "rms-fin-g1"),
    SweepMode(name: "base+lora-gn", skipA: true, noLora: false, preset: "rms-fin-gn"),
    SweepMode(name: "base+lora-unit", skipA: true, noLora: false, preset: "rms-fin-g1-unit"),
    SweepMode(name: "base+lora-ln", skipA: true, noLora: false, preset: "ln-nof-g1"),
    SweepMode(name: "hybrid-train", skipA: false, noLora: false, preset: "rms-fin-g1"),
    SweepMode(name: "hybrid-gn", skipA: false, noLora: false, preset: "rms-fin-gn"),
    SweepMode(name: "hybrid-unit", skipA: false, noLora: false, preset: "rms-fin-g1-unit"),
    SweepMode(name: "hybrid-ln", skipA: false, noLora: false, preset: "ln-nof-g1")]
  var hasAttention = false
  var hasLora = false
  for blk in m.blocks:
    case blk.kind
    of bkMambaReal:
      hasLora = hasLora or blk.mambaReal.inProj.loraR > 0 or blk.mambaReal.outProj.loraR > 0
    of bkMambaRealPlusAttn, bkMambaRealGated:
      hasAttention = true
      hasLora = hasLora or blk.mambaReal.inProj.loraR > 0 or blk.mambaReal.outProj.loraR > 0 or
        blk.attn.qkvProj.loraR > 0 or blk.attn.outProj.loraR > 0
    of bkAttn:
      hasAttention = true
      hasLora = hasLora or blk.attn.qkvProj.loraR > 0 or blk.attn.outProj.loraR > 0
    else: discard
  hasLora = hasLora or m.outProj.loraR > 0
  # Mac dinh: TAT CA che do = model goc + (LoRA | LoRA+sidecar) x 3 kieu norm cua trainer
  # (hien tai / cu gamma=1 / cu LayerNorm). Moi che do ~2 phut (load model moi la phan lau):
  # base+lora-* cho biet LoRA (nhanh R) khop kieu norm nao, hybrid-* them sidecar Linear++.
  # Thu hep: BYBY_NORM_SWEEP_LIST=base-hf,hybrid-train
  var modes = allModes
  let only = getEnv("BYBY_NORM_SWEEP_LIST", "")
  if only == "all":
    modes = allModes
  elif only.len > 0:
    let names = only.split(',').mapIt(it.strip())
    modes = allModes.filterIt(it.name in names)
  let nTok = max(1, envInt("BYBY_NORM_SWEEP_TOKENS", 12))
  let qText = "[INST] What is the capital of France?[/INST]"
  let fullText = qText & " The capital of France is Paris."
  var cfgG = cfgBase
  cfgG.temperature = 0'f32
  var rng = initRand(1)
  stderr.writeLine ""
  stderr.writeLine "════════ [bisect] " & $modes.len & " che do, moi che do: NLL cau mau + greedy " & $nTok & " token ════════"
  if not hasAttention or not hasLora:
    stderr.writeLine "      [bisect] model hien tai: A/P=" & (if hasAttention: "co" else: "KHONG CO") &
      ", LoRA=" & (if hasLora: "co" else: "KHONG CO") &
      ". Toggle nao thieu thanh phan se khong tao ra doi chung that."
  var summary: seq[string] = @[]
  var bestPreset = ""
  var bestNll = 1.0e30
  var defaultNll = 1.0e30     # hybrid voi preset mac dinh rms-fin-gn
  for md in modes:
    gSkipAttn = md.skipA
    setGpuDisableLora(md.noLora)
    discard setGpuNormPreset(md.preset)
    let attnMode =
      if not hasAttention: "khong co A/P (toggle no-op)"
      elif md.skipA: "bo A/P"
      else: "co A/P"
    let loraMode =
      if not hasLora: "khong co LoRA (toggle no-op)"
      elif md.noLora: "tat LoRA"
      else: "co LoRA"
    stderr.writeLine "── che do=" & md.name & "  (" & attnMode & ", " & loraMode &
                     ", norm=" & gNormPresetName & ")"
    let t0 = monoSecs()
    try:
      let nl = teacherForced(m, tok, qText, fullText, vocab, ctx)
      summary.add &"{md.name}={nl.nllAnswer:.2f}"
      if not md.skipA and not md.noLora and hasAttention and hasLora:
        if md.preset == "rms-fin-gn": defaultNll = nl.nllAnswer
        if nl.nllAnswer == nl.nllAnswer and nl.nllAnswer < bestNll:   # (chan NaN)
          bestNll = nl.nllAnswer
          bestPreset = md.preset
      # greedy tu cau hoi tieng Anh ro rang
      resetGpuGenState(m, ctx)
      let qIds = encodeAny(tok, qText)
      if qIds.len > 1: forwardPrefillBatch(m, qIds[0 ..< qIds.len - 1], ctx)
      var ids = qIds
      var gen: seq[int] = @[]
      var stats: seq[BlockStat]
      var lg0: seq[float32]
      for k in 0 ..< nTok:
        var logits: Tensor
        if k == 0:
          logits = forwardStepJambaGpuTraceRetry(m, ids[^1], ids, ids.len - 1, ctx, stats)
          lg0 = logits.data
        else:
          logits = forwardStepJambaGpuRetry(m, ids[^1], ids, ids.len - 1, ctx)
        let winStart = max(0, ids.len - cfgG.repeatWindow)
        let nid = sampleToken(logits.data, vocab, cfgG, banned, noPenalty,
                              toOpenArray(ids, winStart, ids.len - 1), rng)
        if isStopAny(tok, nid): break
        ids.add(nid)
        gen.add(nid)
      printBlockTable(stats, false)
      stderr.writeLine "      logits(buoc 0): " & logitSummary(lg0, vocab, banned)
      stderr.writeLine "      top5 buoc 0:"
      var idx = toSeq(0 ..< vocab)
      idx.sort(proc(a, b: int): int = cmp(lg0[b], lg0[a]))
      for r in 0 ..< 5:
        stderr.writeLine &"        #{r}: id={idx[r]} piece={pieceAny(tok, idx[r]).escape} logit={lg0[idx[r]]:.2f}"
      var uniq: seq[int] = @[]
      for g in gen:
        if g notin uniq: uniq.add g
      stderr.writeLine &"      greedy ids={gen}"
      stderr.writeLine &"      greedy text={decodeAny(tok, gen).escape}" &
                       (if uniq.len == 1 and gen.len > 2: "  <-- LAP 1 TOKEN" else: "")
    except CatchableError as e:
      stderr.writeLine "      [bisect] LOI o che do nay: " & e.msg
    stderr.writeLine &"      ({monoSecs() - t0:.0f}s)"
  gSkipAttn = false
  setGpuDisableLora(false)
  if summary.len > 0:
    stderr.writeLine "[bisect] TOM TAT NLL tra loi (thap = tot; ngau nhien=10.4; base-hf = model goc): " &
                     summary.join("  ")
  if bestPreset.len > 0 and bestPreset != "rms-fin-gn" and bestNll < 6.0 and
     defaultNll - bestNll >= 1.0:
    stderr.writeLine &"[bisect] CHECKPOINT KHOP preset '{bestPreset}' (NLL tra loi {bestNll:.2f} so voi " &
      &"{defaultNll:.2f} cua rms-fin-gn): " &
      (if gCkptNormPreset.len == 0:
         "checkpoint nay train bang trainer CU, khong phai trainer hien tai. " &
         "Sinh van ban se tu dung preset nay (tat bang BYBY_NORM_AUTO=0 hoac chon tay bang BYBY_NORM_PRESET)."
       else:
         &"NHUNG checkpoint ghi san preset '{gCkptNormPreset}' (se dung preset do) - sweep va checkpoint " &
         "khong dong y, can xem lai (thu BYBY_NORM_PRESET=" & bestPreset & " de so sanh).")
    result = bestPreset
  elif defaultNll < 1.0e29 and bestNll < 6.0 and bestPreset == "rms-fin-gn":
    stderr.writeLine "[bisect] checkpoint khop preset mac dinh rms-fin-gn (trainer hien tai)."
  elif defaultNll < 1.0e29 and bestNll >= 6.0:
    stderr.writeLine &"[bisect] KHONG preset nao cho hybrid NLL < 6 (tot nhat {bestNll:.2f}): checkpoint khong " &
      "khop kieu norm nao da thu - can train lai bang trainer hien tai (da dong bo gamma, sidecar dung norm cua R)."
  stderr.writeLine "════════ [bisect] xong ════════"
  stderr.writeLine ""

proc generate*(m: var NimformerModelJamba, lora: var JambaLora, tok: AnyTokenizer, prompt: string,
               maxNewTokens: int, ctx: Backend, temperature: float32 = 0'f32,
               seed: int64 = 1234): string =
  configureInferenceDefaults()
  echo "[infer] 100% GPU (R=Mamba2Real fp4/int4 O(1), A=LinearPlusAttention fp4/int4 O(1))."
  mergeInUploadStreamingJamba(m, lora, ctx)
  prepareGpuModel(m, lora, ctx)
  # Inference now owns packed GPU buffers. Keeping the loader's second host
  # copy costs ~3.5 GiB on Codestral and competes with the same iGPU RAM.
  # Legacy sequence forward also uses gpuCacheName, so it needs only metadata.
  var freedPackedBytes = 0
  for name, qt in lora.quantBase.mpairs:
    if name in gFp4CacheGpu and gFp4CacheGpu[name].residentReady and
       ((ctx.kind.toByby() == gbMetal and gFp4CacheGpu[name].wBuf.metalHandle != nil) or
        (ctx.kind.toByby() == gbVulkan and gFp4CacheGpu[name].wBuf.vulkanHandle.buf != nil)):
      freedPackedBytes += qt.bytes.len
      qt.clearPayload()
  if freedPackedBytes > 0:
    echo "[infer] Packed GPU cache ready; released " &
      $(freedPackedBytes div (1024 * 1024)) & " MiB duplicate host weights."
  GC_fullCollect()
  if getEnv("BYBY_MERGE_LORA","0")!="0":
    discard mergeResidentLoraInplace(m,lora,ctx)
  echo &"[infer] GPU san sang. RSS: {rssMB()}"

  let vocabModel = m.vocab
  var ids = encodeAny(tok, prompt)
  if ids.len == 0:
    raise newException(ValueError, "[infer] prompt encode ra 0 token - kiem tra tokenizer.")
  for id in ids:
    if id < 0 or id >= vocabModel:
      raise newException(ValueError,
        &"[infer] tokenizer sinh id={id} ngoai [0,{vocabModel}) - vocab lech.")

  var rng = initRand(seed)
  let banned = bannedMask(tok, vocabModel)
  let noPenalty = buildNoPenalty(tok, vocabModel)
  let cfg = defaultSamplerCfg(temperature)
  var nBanned = 0
  for b in banned:
    if b: inc nBanned
  echo &"[infer] sampler: T={cfg.temperature} top_p={cfg.topP} min_p={cfg.minP} " &
       &"repeat_penalty={cfg.repeatPenalty}/win={cfg.repeatWindow}, " &
       &"chan {nBanned} control token (khong chan whitespace)."

  # ── (1) kiem tra kernel Linear++ step ngay tren GPU (dang thuc giai tich, vai giay) ──
  if getEnv("BYBY_LPCHECK", "0") != "0":
    var firstA = -1
    for i in 0 ..< m.blocks.len:
      if m.blocks[i].kind in {bkAttn, bkMambaRealPlusAttn, bkMambaRealGated}:
        firstA = i
        break
    if firstA >= 0:
      try:
        stderr.writeLine "[lpcheck] " & lpSelfCheck(ctx, firstA)
      except Exception as e:
        stderr.writeLine "[lpcheck] LOI (bo qua, chay tiep): " & e.msg

  # ── (2) chia doi: model goc (bo A, tat LoRA) vs hybrid day du ──────
  var sweepPreset = ""        # preset bisect de xuat (chi khi hybrid khop ro ret 1 kieu norm khac mac dinh)
  if getEnv("BYBY_NORM_SWEEP", "0") != "0":
    try:
      sweepPreset = runBisect(m, tok, vocabModel, banned, noPenalty, cfg, ctx)
    except Exception as e:
      stderr.writeLine "[bisect] LOI: " & e.msg & " - bo qua, chay binh thuong."
    setGpuDisableLora(false)
    gSkipAttn = false

  # ── cau hinh norm cho lan sinh that ───────────────────────────────
  # Thu tu uu tien:
  #   1. BYBY_NORM_PRESET (chon tay)
  #   2. meta["normPreset"] cua checkpoint = preset cua forward luc train (checkpoint MOI)
  #   3. preset bisect de xuat (checkpoint CU khong co khoa tren; tat bang BYBY_NORM_AUTO=0)
  #   4. mac dinh
  # Mac dinh: checkpoint fine-tune MỚI (trainer da dong bo gamma) train với Mamba-SSM
  # group norm theo n_groups -> "rms-fin-gn". Base thuần cũng dùng semantics reference;
  # preset "hf" vẫn giữ cách full-width của Transformers cũ để so sánh checkpoint.
  # (Truoc day doan "VERIFY ... AUTO-CORRECTING" o duoi LUON ep ve rms-fin-g1 nen
  #  BYBY_NORM_PRESET khong co tac dung; da bo.)
  let defaultPreset =
    "rms-fin-gn"
  var presetName = getEnv("BYBY_NORM_PRESET", "")
  var presetWhy = "BYBY_NORM_PRESET"
  if presetName.len == 0:
    if gCkptNormPreset.len > 0:
      presetName = gCkptNormPreset
      presetWhy = "ghi trong checkpoint (= preset luc train)"
    elif sweepPreset.len > 0 and getEnv("BYBY_NORM_AUTO", "1") != "0":
      presetName = sweepPreset
      presetWhy = "bisect de xuat (checkpoint cu)"
    else:
      presetName = defaultPreset
      presetWhy = "mac dinh"
  if not setGpuNormPreset(presetName):
    stderr.writeLine "[infer] preset '" & presetName &
      "' khong hop le (vd: train | hf | mamba-ssm | ln-nof-g1 | rms-fin-g1 | rms-fin-gn | rms-fin-g1-unit) " &
      "- dung default an toan '" & defaultPreset & "'."
    discard setGpuNormPreset(defaultPreset)
    presetWhy = "mac dinh (preset yeu cau khong hop le)"
  if getEnv("BYBY_NORM_SWEEP", "0") != "0":
    stderr.writeLine "[infer] RESTORED norm preset after bisect: " & gNormPresetName
  stderr.writeLine "[infer] FINAL norm preset for generation: " & gNormPresetName & "  (" & presetWhy & ")"
  stderr.writeLine "[infer] Model pattern: " & m.pattern & ", LoRA adapters: " & $lora.adapters.len
  if m.pattern == CFG_PURE_MAMBA_PATTERN and lora.adapters.len == 0 and gNormPresetName == "ln-nof-g1":
    stderr.writeLine "[infer] CANH BAO: preset train khong phu hop voi base Codestral chua fine-tune; " &
      "co the lam logits/token sai scale. Dung BYBY_NORM_PRESET=hf de quay ve Transformers norm."
  if gNormPresetName notin ["rms-fin-gn", "rms-fin-g1"]:
    stderr.writeLine "[infer] LUU Y: preset '" & gNormPresetName & "' khac preset trainer hien tai (rms-fin-gn); " &
      "preset g1/hf la legacy full-width. Checkpoint moi phai dung preset ghi trong metadata."

  # DEBUG: Log norm preset details for diagnosing garbage tokens
  stderr.writeLine "[infer] norm preset = " & gNormPresetName & " (" & normPresetDescription(m) & ")"
  stderr.writeLine "[infer] DEBUG: pattern=" & m.pattern & ", adapters=" & $lora.adapters.len & ", defaultPreset=" & defaultPreset & ", presetName=" & presetName

  var printedText = decodeAny(tok, ids)

  let promptLen = ids.len
  var sd = initStreamDecoder(tok, ids[^1])
  var genIds: seq[int] = @[]
  let traceN = envInt("BYBY_TRACE_TOKENS", 0)
  let traceBlocks = getEnv("BYBY_TRACE_BLOCKS", "0") != "0"

  resetGpuGenState(m, ctx)
  let prefillStart = monoSecs()
  if ids.len > 1:
    echo &"[infer] Prefill batch qua {ids.len - 1} token prompt ..."
    forwardPrefillBatch(m, lora, ids[0 ..< ids.len-1], ctx)
  # Decode owns independent recurrent buffers; prefix/MPS scratch is no longer needed.
  # Keep the arena only for explicit prefill-throughput experiments.
  if getEnv("BYBY_PREFILL_KEEP_SCRATCH", "0") != "1":
    let released = releaseResidentPrefill()
    if released > 0:
      echo "[infer] Released " & $(released div (1024*1024)) & " MiB prefill scratch before native decode (MPS disabled for M=1)."
  let prefillTime = monoSecs() - prefillStart
  echo "[infer] Prefill xong - bat dau decode."
  echo &"[infer] RSS sau prefill: {rssMB()}"

  stdout.write(printedText)
  stdout.flushFile()
  # Notebook and redirected output must contain no terminal cursor controls.
  # BYBY_PROGRESS=0 also disables updates for notebook runners that emulate a TTY.
  let liveProgress = getEnv("BYBY_PROGRESS", "auto") != "0" and
                     isatty(stdout) and isatty(stderr) and getEnv("TERM", "") != "dumb"
  var stoppedBy = ""
  var firstStepTime = 0.0
  let decodeStart = monoSecs()
  for step in 0 ..< maxNewTokens:
    let stepStart = monoSecs()
    var lastLogits: seq[float32]
    if step == 0 and traceBlocks:
      var bstats: seq[BlockStat]
      let lt = forwardStepJambaGpuTraceRetry(m, ids[^1], ids, ids.len - 1, ctx, bstats)
      lastLogits = lt.data
      stderr.writeLine "==== [BYBY_TRACE_BLOCKS] residual sau tung khoi (buoc 0) ===="
      printBlockTable(bstats, true)
    else:
      let logits = forwardStepJambaGpuRetry(m, ids[^1], ids, ids.len - 1, ctx)
      lastLogits = logits.data

    let winStart = max(0, ids.len - cfg.repeatWindow)
    # Mask chi danh cho control token, KHONG chan / phan loai whitespace.
    
    # DEBUG: Log sampling diagnostics. BYBY_DEBUG_SAMPLING=1 in N buoc dau
    # (N = BYBY_DEBUG_SAMPLING_STEPS, mac dinh 5). BYBY_DEBUG_SAMPLING_ALL=1
    # log MOI buoc - dung khi token rac xuat hien sau buoc 5 (truoc day
    # `step < 5` cung nen khong bao gio thay token rac o buoc muon).
    let dbgSampling = getEnv("BYBY_DEBUG_SAMPLING", "0") == "1"
    let dbgSamplingAll = getEnv("BYBY_DEBUG_SAMPLING_ALL", "0") == "1"
    let dbgSamplingSteps = envInt("BYBY_DEBUG_SAMPLING_STEPS", 5)
    if dbgSampling and (dbgSamplingAll or step < dbgSamplingSteps):
      stderr.writeLine &"[DEBUG sample] step={step} vocab={vocabModel} temp={cfg.temperature} topP={cfg.topP} minP={cfg.minP} repPen={cfg.repeatPenalty} win={cfg.repeatWindow}"
      stderr.writeLine &"[DEBUG sample] controlBanned.count={nBanned} noPenalty.count={noPenalty.filterIt(it).len}"
      stderr.writeLine &"[DEBUG sample] recentIds={ids[winStart ..< ids.len]}"
      let logitStats = logitSummary(lastLogits, vocabModel, banned)
      stderr.writeLine &"[DEBUG sample] logits: " & logitStats
      # Also print top 10 logits
      var idx = toSeq(0 ..< vocabModel)
      idx.sort(proc(a, b: int): int = cmp(lastLogits[b], lastLogits[a]))
      for r in 0 ..< min(10, idx.len):
        let tid = idx[r]
        let txt = decodeAny(tok, @[tid])
        stderr.writeLine &"      top#{r}: id={tid} logit={lastLogits[tid]:.3f} banned={banned[tid]} text={txt.escape}"
    
    let nextId = sampleToken(lastLogits, vocabModel, cfg, banned, noPenalty,
                             toOpenArray(ids, winStart, ids.len - 1), rng)

    if step == 0 and getEnv("BYBY_DEBUG_TOP5", "0") != "0":
      stderr.writeLine "==== [top5 logits, buoc 0] vocab=" & $vocabModel & " " &
                       logitSummary(lastLogits, vocabModel, banned) & " ===="
      printTopK(lastLogits, vocabModel, banned, tok, 5)

    if isStopAny(tok, nextId):
      stoppedBy = "stop token id=" & $nextId
      stderr.writeLine &"[infer] gap {stoppedBy} o buoc {step+1} - DUNG (dung y)."
      break

    ids.add(nextId)
    genIds.add(nextId)
    let stepTime = monoSecs() - stepStart
    if step == 0: firstStepTime = stepTime

    if step < traceN:
      let pieceTxt = decodeAny(tok, @[nextId])
      stderr.writeLine &"[tok] #{step+1} id={nextId} piece={pieceAny(tok, nextId).escape} " &
                       &"logit={lastLogits[nextId]:.3f} text={pieceTxt.escape}"

    let delta = streamDelta(sd, tok, genIds)
    if delta.len > 0:
      stdout.write(delta)
      stdout.flushFile()
      printedText.add(delta)

    # Keep terminal-only progress on stderr, outside the generated text stream.
    if liveProgress:
      let curTokPerSec = 1.0 / max(stepTime, 1e-6)
      stderr.write("\e7\e[1;1H\e[K")
      stderr.write(&"[{genIds.len}/{maxNewTokens} tok | {curTokPerSec:.2f} tok/s | {(stepTime*1000):.0f} ms/tok]")
      stderr.write("\e8")
      stderr.flushFile()

  # xa not byte UTF-8 con giu lai
  let tail = streamDelta(sd, tok, genIds, final = true)
  if tail.len > 0:
    stdout.write(tail)
    stdout.flushFile()
    printedText.add(tail)

  stdout.write("\n")
  stdout.flushFile()
  if stoppedBy.len == 0:
    stderr.writeLine &"[infer] het {maxNewTokens} token ma chua gap stop token."
  var distinctGen: seq[int] = @[]
  for g in genIds:
    if g notin distinctGen: distinctGen.add g
  let decodeTime = monoSecs() - decodeStart
  stderr.writeLine &"[infer] tong ket: sinh {genIds.len} token, {distinctGen.len} id khac nhau, " &
                   &"{sd.emitted.len} byte van ban in ra (prompt {promptLen} token)."
  stderr.writeLine "\n== Thống kê tốc độ =="
  stderr.writeLine &"Prefill: {max(0, promptLen - 1)} token trong {prefillTime:.3f}s -> " &
                   &"{(max(0, promptLen - 1).float / max(prefillTime, 1e-9)):.2f} tok/s"
  if genIds.len > 0 and decodeTime > 0:
    stderr.writeLine &"Decode:  {genIds.len} token trong {decodeTime:.3f}s -> " &
                     &"{(genIds.len.float / decodeTime):.2f} tok/s " &
                     &"({(decodeTime * 1000 / genIds.len.float):.1f} ms/token)"
    if genIds.len > 1:
      let steady = decodeTime - firstStepTime
      stderr.writeLine &"         token dau {firstStepTime:.3f}s; {genIds.len - 1} token sau " &
                       &"{(float(genIds.len - 1) / max(steady, 1e-9)):.2f} tok/s " &
                       &"({(steady * 1000 / float(genIds.len - 1)):.1f} ms/token)"
  stderr.writeLine &"Tổng: {(max(0, promptLen - 1) + genIds.len)} token trong " &
                   &"{(prefillTime + decodeTime):.3f}s"
  if getEnv("BYBY_DECODE_PROFILE", "0") != "0":
    let profSummary = decodeProfileSummary()
    if profSummary.len > 0: stderr.writeLine profSummary
  let splitSummary = decodeSplitSummary()
  if splitSummary.len > 0: stderr.writeLine splitSummary

  result = printedText



when isMainModule:
  configureInferenceDefaults()
  echo "[infer] GEMV_TG=" & getEnv("BYBY_GEMV_TG") &
    " autotune=" & getEnv("BYBY_GPU_AUTOTUNE") &
    " debug_sampling=" & getEnv("BYBY_DEBUG_SAMPLING") &
    "; prefill GEMM=" & getEnv("BYBY_PREFILL_GEMM") &
    "; prefill scan=" & getEnv("BYBY_PREFILL_SCAN") &
    "; FP4 2col=" & getEnv("BYBY_PREFILL_2COL") &
    "; FP4 kernel=" & getEnv("BYBY_PREFILL_FP4_KERNEL") &
    "; resident prefill=" & getEnv("BYBY_PREFILL_RESIDENT") &
    "; decode=native (no MPS)"
  setInferenceThreadPriority()
  let args = commandLineParams()
  let ckptPath = if args.len >= 1: args[0] else: CFG_CHECKPOINT_PATH
  # ═══ SUA: BO KHOANG TRANG CUOI PROMPT ═══
  # Ban cu la "[INST] Xin chao[/INST] " (co dau cach cuoi), voi ly do ghi
  # trong comment la "de khop phan phoi luc train". THUC TE LA NGUOC LAI:
  #
  # Luc train, finetune_lora.lineToText tao chuoi "[INST] {p}[/INST] {resp}"
  # roi encode MOT LAN. SentencePiece gan dau cach vao token DAU cua response:
  # chuoi token la  ... "]" , "▁Hello" , ...
  #
  # Luc infer, neu prompt ket thuc bang dau cach thi ta cat DUNG GIUA cho do:
  # ta day vao  ... "]" , "▁"  (mot token khoang trang DON DOC), roi bat model
  # sinh "Hello" KHONG co dau cach dau - mot phan phoi ma no CHUA TUNG THAY
  # trong toan bo qua trinh train. Day la bug tokenize kinh dien.
  #
  # Bo dau cach di thi prompt ket thuc dung o "]" va model tu sinh "▁Hello" -
  # KHOP CHINH XAC voi luc train.
  var prompt = if args.len >= 2: args[1] else: "[INST] Xin chao[/INST]"
  if prompt.len > 0 and prompt[^1] in {' ', '\t', '\n'}:
    let trimmed = prompt.strip(leading = false, trailing = true)
    stderr.writeLine "[infer] CANH BAO: prompt ket thuc bang khoang trang -> " &
      "sinh ra 1 token '▁' don doc, lech phan phoi so voi luc train. " &
      "Da tu dong cat bo. Dat BYBY_KEEP_TRAILING_WS=1 neu ban CO Y giu."
    if getEnv("BYBY_KEEP_TRAILING_WS", "0") != "1":
      prompt = trimmed
  let temperature = if args.len >= 3: parseFloat(args[2]).float32 else: CFG_TEMPERATURE
  # 2 THAM SO MOI (chi dung khi FALLBACK ve Mamba thuan - xem duoi):
  #   ./infer_lora_bin <ckptPath> <prompt> <temperature> <hfDir> <nimqPath>
  let hfDirArg = if args.len >= 4: args[3] else: CFG_HF_DIR_DEFAULT
  let nimqPathArg = if args.len >= 5: args[4] else: CFG_NIMQ_PATH_DEFAULT

  let requestedBackend = getEnv("BYBY_INFER_BACKEND").strip()
  let ctx = if requestedBackend.len > 0: newBackend(requestedBackend)
            else: newBackend()
  # SUA (MOI): neu KHONG tim thay <ckptPath>.lora.json thi KHONG bao loi/
  # dung chuong trinh nua - tu dong chuyen sang infer THANG tren base Mamba
  # pretrain (khong LoRA, khong block Attention da cay - vi khong co gi da
  # train de nap vao). Dung y "co checkpoint fine-tune thi dung no, khong co
  # thi it nhat van chay duoc model goc" thay vi crash hoan toan.
  var m: NimformerModelJamba
  var lo: JambaLora
  var tok: AnyTokenizer
  if isRawBinCheckpoint(ckptPath):
    (m, lo, tok) = loadBinaryLoraForInfer(ckptPath)
  elif fileExists(ckptPath & ".lora.json"):
    (m, lo, tok) = loadCheckpointLoraForInfer(ckptPath)
  elif args.len > 0:
    raise newException(IOError,
      &"Checkpoint duoc chi dinh '{ckptPath}' khong ton tai/khong nhan dang duoc " &
      "(can NIMBIN1/NIMBIN4 hoac '{ckptPath}.lora.json'). Tu choi fallback am tham ve Mamba thuần.")
  else:
    (m, lo, tok) = loadPureMambaForInfer(hfDirArg, nimqPathArg)
  var rBlocks, pBlocks, gBlocks, aBlocks: seq[int] = @[]
  for i, blk in m.blocks:
    case blk.kind
    of bkMambaReal: rBlocks.add(i)
    of bkMambaRealPlusAttn: pBlocks.add(i)
    of bkMambaRealGated: gBlocks.add(i)
    of bkAttn: aBlocks.add(i)
    else: discard
  echo &"[infer] TOPOLOGY: standalone-R={rBlocks.len}, parallel-P={pBlocks.len}, " &
       &"gated-G={gBlocks.len}, standalone-A={aBlocks.len}; " &
       &"total Mamba2Real={rBlocks.len + pBlocks.len + gBlocks.len}."
  if pBlocks.len + gBlocks.len > 0:
    echo &"[infer] P fixed-sum={pBlocks}; G learned-gate={gBlocks}; R-only={rBlocks}"
  elif aBlocks.len > 0:
    echo &"[infer] WARNING: legacy standalone-A replaces R at: {aBlocks}; no parallel P blocks loaded."
  else:
    echo &"[infer] R-only block indices: {rBlocks}; no Linear++ sidecars loaded."
  echo &"[infer] RSS ngay sau khi load xong model (truoc khi suy luan): {rssMB()}"
  echo &"[infer] prompt: {prompt}"
  discard generate(m, lo, tok, prompt, max(1, envInt("BYBY_MAX_NEW_TOKENS", CFG_MAX_NEW_TOKENS)), ctx,
                    temperature = temperature, seed = CFG_SEED)
  echo "─────────────────────────────"
