## finetune_lora.nim
## Fine-tune kiểu LoRA cho NimformerModelJamba. QUAN TRỌNG:
##   - 2 NGUỒN checkpoint đầu vào, chọn 1 qua CFG_INIT_MODE:
##     "graft"     : như trước đây - đọc CFG_OLD_CHECKPOINT (.bin đã graft
##                   sẵn bởi graft_attention.nim, pattern kiểu "MMAMMAMS",
##                   toàn Mamba2SSD 'M' train từ đầu).
##     "codestral" : MỚI - đọc TRỰC TIẾP weight pretrain Mamba-Codestral-7B-v0.1
##                   từ CFG_HF_DIR (thư mục safetensors đã tải về), dựng model
##                   pattern CFG_CODESTRAL_PATTERN gồm block 'R' (Mamba2Real -
##                   đúng công thức SSD thật, xem mamba2_real.nim) nạp thẳng
##                   weight pretrain + block 'A' (LinearPlusAttention mới,
##                   random-init) - qua load_codestral_into_jamba.nim.
##   - Loại block nào là LoRA-hoá (đã pretrain: 'M' hoặc 'R') hay train full
##     (mới cấy: 'A'/'S') đọc TRỰC TIẾP từ `m.pattern` của chính model đã
##     dựng - không lệch giữa 2 mode.
##   - Đóng băng toàn bộ phần đã pretrain ('M'/'R'), chỉ LoRA-hoá (rank thấp)
##     + train FULL các block Attention mới - đúng tinh thần cũ, không đổi.
##   - QUAN TRỌNG: checkpoint base vẫn .bin (rawbin.nim qua jamba_hybrid.
##     saveCheckpoint/loadCheckpoint) - KHÔNG dùng .nimq ở đây.

import std/[math, random, strformat, os, strutils, json, tables, sequtils, parsejson, streams, times, monotimes]
import nimformer
import backend
import quant
import rawbin
import jamba_hybrid
import hybrid_block_kernel   # gLoraGradRequest/LoraGradSpec - backward LoRA-aware
import mamba2_real              # Mamba2Real - can cho block 'R'
import load_codestral_into_jamba # loadCodestralIntoJamba - nap weight HF that vao block 'R'
import hf_tokenizer               # tokenizer THAT (tekken/HF) - bat buoc cho CFG_INIT_MODE=="codestral"
import lora
import session_vm
import jit_vm
import test_nimformer   # tái dùng BPETokenizer: getOrTrainBPE/encodeBPE/decodeBPEStr
import quantize_lora_bin_4bit
import mamba2_ssm_gpu   # softmaxCrossEntropyGpu, apfAdamStepFused
import checkpoint
import prefill_profile   # pfTime/pfReport - dung lai cho BYBY_TRAIN_PROFILE

proc cRename(oldPath, newPath: cstring): cint {.importc: "rename", header: "<stdio.h>".}
proc cLink(oldPath, newPath: cstring): cint {.importc: "link", header: "<unistd.h>".}

# Ctrl+C an toan: lan 1 chi bat co; break SAU Adam+SSD layer hien tai. Lan 2 force quit.
var gStopRequested* = false
var gTrainStepFullyDone* = true

proc onTrainCtrlC() {.noconv.} =
  if gStopRequested:
    try:
      stderr.write("\n[finetune_lora] Ctrl+C lan 2 -> force quit.\n")
      stderr.flushFile()
    except CatchableError: discard
    quit(130)
  gStopRequested = true
  try:
    stderr.write("\n[finetune_lora] Ctrl+C: dung AN TOAN sau Adam+SSD layer hien tai (nhan lan nua de force).\n")
    stderr.flushFile()
  except CatchableError: discard

# ═══════════════════════════════════════════════════════════════
# CONFIG - sửa ở đây (day la config CUNG mac dinh). 6 gia tri quan trong
# nhat (INIT_MODE/OLD_CHECKPOINT/HF_DIR/HF_SOURCE/NIMQ_PATH/CHECKPOINT_PATH)
# CO THE ghi de qua dong lenh (xem parseFinetuneArgs() ngay duoi day) -
# KHONG truyen gi thi dung DUNG config cung nay, khong doi hanh vi cu.
#   ./finetune_lora [initMode] [oldCheckpoint] [hfDir] [hfSource] [nimqPath] [outCheckpointPath]
const
  CFG_INIT_MODE_DEFAULT* = "codestral"                   # "graft" (duong cu) hoac "codestral" (MOI)
  CFG_OLD_CHECKPOINT_DEFAULT* = "mamba_chat_2_8b.bin"    # chi dung khi CFG_INIT_MODE=="graft"
  CFG_HF_DIR_DEFAULT* = "test/codestral/models/hf"  # local Codestral tokenizer/config (or safetensors source)
  CFG_HF_SOURCE_DEFAULT* = "nimq"          # "safetensors" (doc thang ~14GB bf16 goc) hoac
                                     # "nimq" (doc file DA QUANTIZE fp4 ~3.5GB, xem
                                     # quantize_codestral.nim de tao file nay 1 LAN
                                     # tu CFG_HF_DIR truoc, roi co the XOA safetensors
                                     # goc di neu muon tiet kiem o - .nimq van dequantize
                                     # ve fp32 trong RAM luc load, KHONG tiet kiem RAM,
                                     # chi tiet kiem O DIA)
  CFG_NIMQ_PATH_DEFAULT* = "test/codestral/models/codestral_mamba_only.nimq"  # chi dung khi CFG_HF_SOURCE=="nimq" -
                                     # PHAI KHOP voi output cua quantize_codestral.nim /
                                     # graft ARGS (xem Makefile: GRAFT_DEFAULT_OUT dung
                                     # DUNG ten nay de 2 buoc noi thang vao nhau)
  CFG_CHECKPOINT_PATH_DEFAULT* = "test/codestral/models/jamba_parallel_lora.bin"

proc argOrDefault(args: seq[string], i: int, default: string): string =
  ## "-" = sentinel giu nguyen gia tri mac dinh (dung khi muon set tham so
  ## SAU no ma khong doi tham so nay) - giong quy uoc argOr trong Makefile.
  if args.len > i and args[i].len > 0 and args[i] != "-": args[i] else: default

let finetuneArgs = commandLineParams()
let CFG_INIT_MODE* = argOrDefault(finetuneArgs, 0, CFG_INIT_MODE_DEFAULT)
let CFG_OLD_CHECKPOINT* = argOrDefault(finetuneArgs, 1, CFG_OLD_CHECKPOINT_DEFAULT)
let CFG_HF_DIR* = argOrDefault(finetuneArgs, 2, CFG_HF_DIR_DEFAULT)
let CFG_HF_SOURCE* = argOrDefault(finetuneArgs, 3, CFG_HF_SOURCE_DEFAULT)
let CFG_NIMQ_PATH* = argOrDefault(finetuneArgs, 4, CFG_NIMQ_PATH_DEFAULT)
let CFG_CHECKPOINT_PATH* = getEnv("BYBY_CHECKPOINT_PATH", argOrDefault(finetuneArgs, 5, CFG_CHECKPOINT_PATH_DEFAULT))
# THAM SO 7: thu muc scratch activation (spill-to-disk).
# THAM SO 8: thu muc Adam state tren SSD; mac dinh cung SSD voi scratch/model.
#   ./finetune_lora_bin - - - - - - /Volumes/Data 1/scratch /Volumes/Data 1/scratch/.nimformers_adam
# Adam state KHONG bi xoa sau moi tensor - no phai ton tai giua cac step. Chi
# xoa khi training ket thuc (hoac khi bat dau run moi ma oldStep==0).
# THAM SO 7 (MOI): thu muc de ghi file scratch activation (spill-to-disk) -
# MAC DINH la CUNG O voi model (CFG_HF_DIR) vi do la o co san co du GB
# trong (khong phai o he thong "/" thuong gan day) - TRUYEN RIENG neu muon
# 1 o SSD ngoai KHAC voi o chua model, vd:
#   ./finetune_lora_bin - - - - - - "/Volumes/Data 1/scratch"
let CFG_SCRATCH_DIR* = getEnv("BYBY_SCRATCH_DIR", argOrDefault(finetuneArgs, 6, CFG_HF_DIR_DEFAULT))
let CFG_ADAM_DIR* = getEnv("BYBY_ADAM_DIR", argOrDefault(finetuneArgs, 7, CFG_SCRATCH_DIR & "/.nimformers_adam"))
let CFG_ATTN_WEIGHTS_DIR* = getEnv("BYBY_ATTN_WEIGHTS_DIR", argOrDefault(finetuneArgs, 8, CFG_SCRATCH_DIR & "/.nimformers_attnw"))
  # MOI: noi luu tam weight FULL-PRECISION cua block Attention ('A') tren SSD -
  # xem AttnWeightDiskStore. Truoc day cac tham so nay (~3.5GB, 12 block 'A')
  # nam CO DINH trong RAM suot ca qua trinh train (khac han block 'R' co
  # merge/restore qua LUT) - day la nguon RSS lon nhat con lai sau khi da
  # stream hoa Adam state + activation + checkpoint load/save.

proc getCheckpointPath*(): string =
  getEnv("BYBY_CHECKPOINT_PATH", CFG_CHECKPOINT_PATH)

proc getScratchDir*(): string =
  getEnv("BYBY_SCRATCH_DIR", CFG_SCRATCH_DIR)

const
  CFG_CODESTRAL_PATTERN_DEFAULT* = "R".repeat(7) & "G" & "R".repeat(15) & "G" &
                                  "R".repeat(15) & "G" & "R".repeat(15) & "G" & "R".repeat(8)
  CFG_CODESTRAL_VOCAB* = 32768      # XAC NHAN THAT: backbone.embeddings.weight=[32768,4096]
  CFG_CODESTRAL_HIDDEN* = 4096      # XAC NHAN THAT: hidden_size
  CFG_CODESTRAL_NHEADS* = 128       # XAC NHAN THAT: mixer.A_log=[128] (so head SSM cua Mamba2 -
                                     # KHAC so head cua block 'A' LinearPlusAttention moi, dung
                                     # chung gia tri nay cho ca 2 o day cho don gian)
  CFG_CODESTRAL_DSTATE* = 128       # XAC NHAN THAT (suy tu conv_dim=10240=intermediate(8192)+
                                     # 2*nGroups*dState -> nGroups*dState=1024=8*128)
  CFG_CODESTRAL_NGROUPS* = 8        # XAC NHAN THAT (1024/128=8, khop mac dinh HF Mamba2Config)
  CFG_CODESTRAL_CONVKERNEL* = 4     # XAC NHAN THAT: mixer.conv1d.weight=[10240,1,4]
  CFG_CODESTRAL_EXPAND* = 2         # XAC NHAN THAT (SUA - truoc day hard-code sai la 1):
                                     # intermediateSize=8192=4096*2, tu mixer.norm.weight=[8192]
                                     # va mixer.out_proj.weight=[4096,8192]
  CFG_LORA_RANK* = 4
  CFG_LORA_ALPHA* = 8'f32
  CFG_LORA_LR* = 1e-4'f32
  CFG_NEWBLOCK_LR* = 3e-4'f32                    # lr cho block A/S moi (train full)
  CFG_G_SIDECAR_LR* = 5e-5'f32
  CFG_G_GATE_LR* = 2e-5'f32
  CFG_G_GRAD_CLIP_NORM* = 1'f32
  CFG_G_GATE_BIAS_MIN* = 2.0'f32     # gate bias (init=4) khong duoc tut duoi nguong nay
  CFG_G_GATE_BIAS_MAX* = 6.0'f32     # cung khong duoc vuot nguong nay (giu Mamba uu the)
  CFG_G_LR_WARMUP_STEPS* = 200       # G LR scale tu ~0 -> 1.0 trong N buoc dau
  CFG_G_MAX_RESCUE_ATTEMPTS* = 3     # so lan lien tiep 1 khoi G co the rollback truoc khi bao loi
  CFG_G_LR_MIN_SCALE* = 0.05'f32     # san cua per-block LR scale khi phai cuu lien tuc
  # R-block (pure Mamba2Real) rescue. Residual RMS 16× gần như không bắt
  # "hub hub hub" (mode SSM lệch nhưng amplitude vẫn đẹp). Mixer
  # rms(y-x)/rms(x) mới là chỗ siết — gần thang G sidecar 0.5×, không 4×.
  CFG_R_MAX_OUTPUT_RMS_RATIO* = 16.0'f32   # residual out/input (nổ amplitude)
  CFG_R_MAX_OUTPUT_MAX_RATIO* = 32.0'f32
  CFG_R_MAX_MIXER_RMS_RATIO* = 1.5'f32     # rms(y-x)/rms(x) — SSM lấn residual
  CFG_R_MAX_MIXER_MAX_RATIO* = 4.0'f32
  CFG_R_MAX_RESCUE_ATTEMPTS* = 3
  CFG_R_LR_MIN_SCALE* = 0.05'f32
  # G mambaY: G_MAX_MAMBA_RMS_RATIO trong jamba_hybrid.checkGBlockScale
  # (forwardBlock G luôn gọi; cache.mambaY expose từ block kernel).
  # Gate dispersion: if bias min≈max (std ~0) the G block degenerates to
  # constant gate; hard rescue when BYBY_G_GATE_STD_RESCUE=1
  CFG_G_GATE_BIAS_MIN_STD* = 1e-4'f32
  # Per-layer small probe during finetune: after a block update, optionally run
  # a short fixed-prompt generate and flag garbage tokens / large drift.
  # Mặc định BẬT (getEnv default "1"). Tắt bằng BYBY_LAYER_PROBE=0.
  # Chạy mỗi CFG_LAYER_PROBE_EVERY lần update thành công của block (0 = mỗi lần).
  CFG_LAYER_PROBE_EVERY* = 1
  CFG_LAYER_PROBE_MAX_TOKENS* = 16
  CFG_LAYER_PROBE_GARBAGE_FRAC* = 0.25'f32  # P/G / default
  CFG_R_LAYER_PROBE_DRIFT* = 0.08'f32       # R/Mamba: siết hơn — 25% quá rộng với SSM
  CFG_SEQ_LEN_DEFAULT* = 256         # 256 x batch 1. Metal limit auto-scales with RAM
  CFG_BATCH_SIZE* = 1                             # Adam minibatch that; khong co gradient accumulation
  CFG_EPOCHS* = 1
  CFG_CORPUS_PATH* = "test/spectron/databricks-dolly-15k.jsonl"
  CFG_BPE_MERGES* = 500
  CFG_CHECKPOINT_EVERY* = 1   # 1 - luu checkpoint moi training step.
  CFG_LOG_EVERY* = 20
  CFG_KEEP_RESUME_STATE* = true # giu Adam + FP32 block moi tren SSD de resume tiep ma khong reset

let CFG_CODESTRAL_PATTERN* = getEnv("BYBY_CODESTRAL_PATTERN", CFG_CODESTRAL_PATTERN_DEFAULT)

proc envSeqLen(defaultLen: int): int =
  ## Cho phep override CFG_SEQ_LEN qua BYBY_SEQ_LEN de tang cua so train tren
  ## may khoe ma khong build lai. Cua so 128 < p50 (~130 token tren dolly)
  ## nghia la phan lon cua so train bat dau tu giua cau voi SSM state = 0,
  ## lech phan phoi voi infer (prompt day du + state da prefill).
  let raw = getEnv("BYBY_SEQ_LEN", "")
  if raw.len == 0: return defaultLen
  try:
    let v = parseInt(raw)
    if v < 32:
      stderr.writeLine &"[finetune_lora] BYBY_SEQ_LEN=" & raw &
        " qua nho (<32) - dung default " & $defaultLen
      return defaultLen
    echo &"[finetune_lora] BYBY_SEQ_LEN override -> CFG_SEQ_LEN={v} (default {defaultLen})"
    return v
  except ValueError:
    stderr.writeLine &"[finetune_lora] BYBY_SEQ_LEN='" & raw &
      "' khong phai so - dung default " & $defaultLen
    return defaultLen

let CFG_SEQ_LEN* = envSeqLen(CFG_SEQ_LEN_DEFAULT)

# Per-block rescue state for G (gated) AND R (pure Mamba2Real) blocks. Lives at
# module scope so it persists across `trainStepGraftStreaming` calls (each
# training step is a separate call).
#   - gPerBlockLrScale: starts at 1.0; halved on a rescue, slowly recovers on
#     consecutive successes. Floored at CFG_G_LR_MIN_SCALE / CFG_R_LR_MIN_SCALE.
#   - gPerBlockRescueFails: consecutive failures per block; reset to 0 on any
#     successful post-update check, incremented on rescue. Once it reaches
#     CFG_*_MAX_RESCUE_ATTEMPTS, we stop trying and raise.
#   - gPerBlockProbeOk: consecutive successful small probes (for LAYER_PROBE).
var gPerBlockLrScale* = initTable[int, float32]()
var gPerBlockRescueFails* = initTable[int, int]()
var gPerBlockProbeOk* = initTable[int, int]()

# ═══════════════════════════════════════════════════════════════
# DATASET STREAMING: KHONG doc het .jsonl vao RAM (truoc day loadDollyJsonl
# tra ve seq[string] CA FILE, roi makeBatches lai tokenize CA CORPUS thanh
# 1 seq[int] khong lo, roi vet CA DANH SACH batch (seq[tuple[...]]) truoc
# khi train - 3 tang tich tu cung luc. 5 file python_code_dataset_*.jsonl
# dang quarantine: code co the hop le nhung explanation/context/tag bi ghep
# cheo va lap lai nhieu; KHONG dua chung vao CFG_DATASETS. Active corpus la
# Dolly + Nim/Python topic-code mega. Cac dataset duoc chon doc streaming:
# doc TUNG DONG qua iterator `lines()` (Nim doc
# stream, khong load ca file), tokenize TUNG DONG, don vao 1 buffer token
# CO SAN (cap phat 1 LAN, tai su dung nhu pool/ring - kieu allocator co dinh
# tren embedded/ESP32, KHONG malloc/free lap lai moi batch), rut tung batch
# ra khoi buffer do (shift phan con lai ve dau, KHONG cap phat seq moi) roi
# nap tiep tu file khi buffer can. Batch cu (idsB/tgtB) het scope ngay sau
# 1 buoc train (train_step nhan roi bo qua) -> GC thu hoi ngay, khong tich
# tu qua cac epoch.
# ═══════════════════════════════════════════════════════════════

const ResumeActivationMagic = "NIMACT1"
const ResumeStateMagic = "NIMSTATE1"

type
  DatasetSchema* = enum
    dsDolly       # instruction/context/response (hoac output)
    dsTopicCode   # topic/language/code/note (nim_python_code_dataset_mega)

  DatasetSpec* = object
    path*: string
    schema*: DatasetSchema

const CFG_DATASETS*: seq[DatasetSpec] = @[
  # Chi bat nguon da kiem tra schema/nhan. Khong auto-scan JSONL: cac
  # python_code_dataset_1..5 co cap code va explanation/context ghep cheo.
  DatasetSpec(path: "test/spectron/databricks-dolly-15k.jsonl", schema: dsDolly),
  DatasetSpec(path: "test/spectron/nim_python_code_dataset_mega.jsonl", schema: dsTopicCode),
]

proc jsonStringField(obj: JsonNode, key: string): string =
  ## Dataset JSON is external input: treat absent/non-string fields as empty
  ## rather than letting getStr raise JsonKindError and kill a long run.
  if obj.kind != JObject: return ""
  if obj.hasKey(key) and obj[key].kind == JString:
    result = obj[key].getStr()

proc lineToText(schema: DatasetSchema, line: string): string =
  ## "" = dong loi/rong, ben goi bo qua. KHONG raise (1 dong loi khong duoc
  ## lam chet ca vong train dai).
  if line.strip().len == 0: return ""
  try:
    let obj = parseJson(line)
    if obj.kind != JObject: return ""
    case schema
    of dsDolly:
      let ins = jsonStringField(obj, "instruction")
      let ctx = jsonStringField(obj, "context")
      let response = jsonStringField(obj, "response")
      let resp = if response.strip().len > 0: response
                 else: jsonStringField(obj, "output")
      # SUA: dieu kien cu la `ins.len == 0 and resp.len == 0` (AND). Neu resp
      # RONG ma ins co, ban van tao ra vi du "[INST] x[/INST] " + EOS - tuc la
      # DAY MODEL TRA LOI RONG roi dung ngay. Vai tram dong nhu vay du de model
      # hoc thoi quen im lang. (Da dem tren databricks-dolly-15k.jsonl: 0 dong
      # bi dinh - nhung 5 file python_code_dataset_*.jsonl thi chua kiem tra.)
      if ins.strip().len == 0 or resp.strip().len == 0: return ""
      let prompt = if ctx.len > 0: ins & "\n" & ctx else: ins
      # SUA: template CU "<|user|>/<|assistant|>" la ChatML - Codestral
      # (Mistral tokenizer V3, sentencepiece) KHONG dung ChatML ma dung
      # "[INST] ...[/INST] ..." (xem comment CFG_CODESTRAL_PATTERN va
      # hf_tokenizer.nim). "</s>" o cuoi KHONG con nhet vao day nua - eos
      # THAT duoc them qua addEos=true luc goi encodeHf ben duoi (refill).
      result = &"[INST] {prompt}[/INST] {resp}"
    of dsTopicCode:
      let topic = jsonStringField(obj, "topic")
      let lang = jsonStringField(obj, "language")
      let code = jsonStringField(obj, "code")
      if code.strip().len == 0 or topic.strip().len == 0 or lang.strip().len == 0: return ""
      result = &"[INST] Viet code {lang} cho: {topic}[/INST] {code}"
  except JsonParsingError:
    result = ""

iterator streamTextsForever(specs: seq[DatasetSpec]): string =
  var anyFileFound = false
  for s in specs:
    if fileExists(s.path): anyFileFound = true
  if not anyFileFound:
    stderr.writeLine("[data] CANH BAO: khong thay dataset nao trong CFG_DATASETS - dung rong")
  var keepStreaming = anyFileFound
  while keepStreaming:
    var yieldedThisPass = false
    for s in specs:
      if not fileExists(s.path): continue
      for line in lines(s.path):
        let t = lineToText(s.schema, line)
        if t.len > 0:
          yieldedThisPass = true
          yield t
    if not yieldedThisPass:
      stderr.writeLine("[data] Khong co dong JSONL hop le trong cac dataset; dung iterator thay vi lap vo han.")
      keepStreaming = false

const TargetIgnore* = -1
  ## Gia tri target "bo qua vi tri nay" (giong ignore_index cua PyTorch).
  ## CHI softmaxCrossEntropyMasked o duoi hieu gia tri nay -
  ## test_nimformer.softmaxCrossEntropy GOC thi KHONG, nen tuyet doi khong
  ## dua targets da mask vao duong do (xem vong train o cuoi file).

type
  StreamingBatcher* = object
    tok: BPETokenizer
    hfTok: HfTokenizer
    usingHfTok: bool
    seqLen, batchSize: int
    ## `pending` la buffer token DUY NHAT, cap phat 1 lan (newSeqOfCap) roi
    ## tai su dung suot qua trinh train qua setLen(0)/add/shift - giong 1
    ## pool co dinh tren MCU (ESP32-style).
    pending: seq[int]
    ## MOI: docStart[i] = true nghia la pending[i] la token DAU TIEN cua mot
    ## document moi (tuc la BOS cua vi du ke tiep). Dung de mask loss o ranh
    ## gioi - xem nextBatch.
    docStart: seq[bool]
    nMasked: int          # thong ke: tong so vi tri da bi mask
    nEmitted: int         # thong ke: tong so vi tri da sinh ra
    textIter: iterator(): string {.closure.}

const CFG_PENDING_CAPACITY = 1 shl 16   # 65536 token

proc newStreamingBatcher*(specs: seq[DatasetSpec], seqLen, batchSize: int,
                          tok: BPETokenizer, hfTok: HfTokenizer,
                          usingHfTok: bool): StreamingBatcher =
  result.tok = tok
  result.hfTok = hfTok
  result.usingHfTok = usingHfTok
  result.seqLen = seqLen
  result.batchSize = batchSize
  result.pending = newSeqOfCap[int](CFG_PENDING_CAPACITY)
  result.docStart = newSeqOfCap[bool](CFG_PENDING_CAPACITY)
  result.nMasked = 0
  result.nEmitted = 0
  result.textIter = iterator(): string {.closure.} =
    for t in streamTextsForever(specs): yield t

proc refill(b: var StreamingBatcher, need: int) =
  ## Tokenize them dong moi tu textIter, APPEND vao `pending`, dong thoi danh
  ## dau vi tri BAT DAU cua tung document vao `docStart`.
  while b.pending.len < need:
    let t = b.textIter()
    if finished(b.textIter):
      if b.pending.len == 0: return
      break
    # addEos=true: day la VI DU TRAIN DAY DU (prompt+response), can model
    # hoc dung diem dung that (id eos) - KHAC voi luc encode prompt inference
    # (infer_lora.nim, addEos=false mac dinh).
    let ids = if b.usingHfTok: encodeHf(b.hfTok, t, addEos = true) else: encodeBPE(b.tok, t)
    if ids.len == 0: continue
    var first = true
    for x in ids:
      b.pending.add(x)
      b.docStart.add(first)
      first = false

proc nextBatch*(b: var StreamingBatcher): tuple[ids, targets, targetsMasked: seq[seq[int]]] =
  ## ═══════════════════════════════════════════════════════════════════════
  ## HAI SUA so voi ban cu.
  ##
  ## (1) BUOC TRUOT = seqLen, khong phai seqLen+1.
  ##     Ban cu: sample i lay ids = pending[i*33 ..< i*33+32],
  ##             targets = pending[i*33+1 ..< i*33+33], roi base nhay +33.
  ##     Hau qua: token o vi tri i*33+32 la TARGET cua sample i nhung KHONG
  ##     BAO GIO la input cua ai; token i*33+33 la input cua sample i+1 nhung
  ##     KHONG AI du doan no. Tuc la CU 33 TRANSITION THI 1 CAI KHONG BAO GIO
  ##     DUOC HOC, tai moi bien cua so, suot ca qua trinh train.
  ##     Gio: buoc truot = seqLen va GIU LAI 1 token cuoi lam input dau cua
  ##     batch ke tiep -> moi transition deu xuat hien o dau do.
  ##
  ## (2) MASK RANH GIOI DOCUMENT.
  ##     Buffer packed noi cac vi du lai voi nhau, khong quan tam ranh gioi:
  ##        ... </s> <s> [INST] ...
  ##     Ban cu cho TAT CA vi tri vao loss, nen model duoc day ro rang rang
  ##     "sau </s> thi phat <s>" va "sau van ban bat ky thi co the phat <s>".
  ##     Do CHINH LA thu no lam luc infer: tra loi xong -> </s> -> <s> ->
  ##     "[INST] ..." -> mot document moi.
  ##     Tra ve HAI ban targets:
  ##       targets       - THO, moi id deu hop le (dung cho duong trainStepGraft
  ##                       khong streaming, di qua softmaxCrossEntropy GOC cua
  ##                       test_nimformer - file do KHONG bi sua)
  ##       targetsMasked - vi tri ranh gioi = TargetIgnore (dung cho
  ##                       trainStepGraftStreaming + softmaxCrossEntropyMasked
  ##                       dinh nghia ngay trong file nay)
  ##
  ## ─── CAI CHUA SUA O DAY (co y, de ban tu quyet) ────────────────────────
  ## (a) CFG_SEQ_LEN = 32 van giu nguyen vi tang len se an RAM/VRAM ngay, ma
  ##     may dang chay la GPU tich hop. Nhung day la van de LON:
  ##     do trên databricks-dolly-15k.jsonl (15011 dong): do dai vi du
  ##     p50=468 ky tu (~130 token), p90=1720, p99=5316, max=27327.
  ##     Voi cua so 33 token -> trung binh ~4 cua so/vi du, p99 ~45 cua so.
  ##     Nghia la khoang 75% so cua so train KHONG chua ca "[INST]" lan
  ##     "</s>", bat dau tu GIUA CAU, voi SSM state = 0. Ban dang day model
  ##     rang "bat dau tu giua cau voi state rong" la binh thuong - trong khi
  ##     luc infer thi nguoc lai hoan toan. Nen nang len >= 256 khi co the.
  ## (b) MASK PHAN PROMPT ([INST]...[/INST]) chua lam. De lam dung thi phai
  ##     biet ranh gioi prompt/response TINH THEO TOKEN, ma cach duy nhat an
  ##     toan la encode rieng 2 nua roi noi lai - viec do THAY DOI cach
  ##     tokenize ngay tai cho noi (SentencePiece co add_dummy_prefix, encode
  ##     " {resp}" rieng co the ra "▁▁resp" thay vi "▁resp"). Rui ro cao hon
  ##     loi ich, nen de thanh 1 buoc rieng co do lai loss.
  ## ═══════════════════════════════════════════════════════════════════════
  let stride = b.seqLen
  let needTok = b.batchSize * stride + 1     # +1: can token target cuoi cung
  refill(b, needTok)

  var idsB = newSeq[seq[int]](b.batchSize)
  var tgtB = newSeq[seq[int]](b.batchSize)
  var tgtMaskB = newSeq[seq[int]](b.batchSize)
  let avail = if b.pending.len >= 1: (b.pending.len - 1) div stride else: 0
  let n = min(b.batchSize, max(avail, 0))
  var maskedHere = 0
  for i in 0 ..< n:
    let base = i * stride
    idsB[i] = b.pending[base ..< base + b.seqLen]
    var tg = newSeq[int](b.seqLen)
    var tgM = newSeq[int](b.seqLen)
    for t in 0 ..< b.seqLen:
      let tIdx = base + t + 1
      tg[t] = b.pending[tIdx]
      if b.docStart[tIdx]:
        # target la token DAU cua document ke tiep -> khong the suy ra tu ngu
        # canh truoc do, bo khoi loss.
        tgM[t] = TargetIgnore
        inc maskedHere
      else:
        tgM[t] = b.pending[tIdx]
    tgtB[i] = tg
    tgtMaskB[i] = tgM
  idsB.setLen(n); tgtB.setLen(n); tgtMaskB.setLen(n)
  b.nMasked += maskedHere
  b.nEmitted += n * b.seqLen

  let consumed = n * stride
  if consumed > 0:
    # shift-left TAI CHO ca 2 mang (khong tao seq moi). GIU LAI token o vi tri
    # `consumed` - no la target cuoi cua sample cuoi va se lam input dau cua
    # batch ke tiep, nho vay khong transition nao bi bo sot.
    let remain = b.pending.len - consumed
    for i in 0 ..< remain:
      b.pending[i] = b.pending[i + consumed]
      b.docStart[i] = b.docStart[i + consumed]
    b.pending.setLen(remain)
    b.docStart.setLen(remain)
  result = (idsB, tgtB, tgtMaskB)

proc skipSteps*(b: var StreamingBatcher, steps: int) =
  ## Resume dataset TU DUNG vi tri: `epoch` trong checkpoint thuc ra la
  ## so training step, khong phai so vong quet dataset. Vì streamTextsForever()
  ## luon deterministic theo thu tu file/dong, goi nextBatch() `steps` lan
  ## se bo qua DUNG so token da train truoc do. Chi chay khi resume.
  if steps <= 0: return
  for i in 0 ..< steps:
    let (ids, _, _) = b.nextBatch()
    if ids.len == 0:
      raise newException(IOError,
        &"Resume: khong skip duoc toi training step {steps}; dataset ngan/bi thay doi so voi run truoc.")
    if (i + 1) mod 250 == 0:
      stderr.writeLine(&"[resume] da bo qua {i + 1}/{steps} training step de ve dung vi tri dataset")
  stderr.writeLine(&"[resume] dataset cursor = training step {steps} (KHONG train lai cac batch truoc)")

# ═══════════════════════════════════════════════════════════════
# LOSS CO MASK - dinh nghia NGAY TRONG FILE NAY.
# KHONG sua test_nimformer.softmaxCrossEntropy (ban goc giu nguyen 100%),
# va cung khong sua jamba_hybrid. Duong trainStepGraft (khong streaming) van
# dung targets THO khong mask, di qua softmaxCrossEntropy goc nhu cu.
#
# Vi sao can mask: buffer packed noi cac vi du lai voi nhau khong quan tam
# ranh gioi:   ... </s> <s> [INST] ...
# Neu cho TAT CA vi tri vao loss thi model duoc day ro rang hai luat
#   "sau </s> thi phat <s>"  va  "sau van ban bat ky co the phat <s>"
# - dung chinh xac thu no lam luc infer (tra loi xong roi mo mot document
# moi). Cac vi tri co target la token DAU cua document ke tiep duoc gan
# TargetIgnore (khai bao o tren, truoc StreamingBatcher) va bo khoi loss o day.
# ═══════════════════════════════════════════════════════════════
proc softmaxCrossEntropyMasked(logits: Tensor, targets: seq[seq[int]],
                                ctx: Backend): tuple[loss: float32, dLogits: Tensor] =
  ## Giong test_nimformer.softmaxCrossEntropy nhung:
  ##   - bo qua vi tri co target == TargetIgnore (dLogits hang do giu nguyen 0,
  ##     newTensor da zero san), chia loss cho SO VI TRI HOP LE chu khong phai B*T
  ##   - kiem tra bien `tgt`. Ban goc lam `probs[base + tgt]` voi
  ##     base = (b*T+t)*V: neu tgt >= V thi base+tgt VAN NAM TRONG mang phang
  ##     (roi vao hang t+1) o moi vi tri tru hang cuoi -> bounds check cua Nim
  ##     KHONG bat duoc -> gradient cong vao sai vi tri thoi gian, im lang,
  ##     loss van dep. O day raise ngay.
  let B = logits.shape[0]
  let T = logits.shape[1]
  let V = logits.shape[2]
  # 100% GPU: softmax + NLL + dLogits + chia nValid trong 1 micro-kernel / hang
  # (train_gpu_vm.buildCeProgram). Host chi kiem tra id target hop le (so nguyen).
  var tgtF = newSeq[float32](B * T)
  var nValid = 0
  for b in 0 ..< B:
    for t in 0 ..< T:
      let tgt = targets[b][t]
      if tgt == TargetIgnore:
        tgtF[b * T + t] = -1'f32
        continue
      if tgt < 0 or tgt >= V:
        raise newException(IndexDefect,
          "softmaxCrossEntropyMasked: target id=" & $tgt & " ngoai [0," & $V &
          ") tai b=" & $b & " t=" & $t & " - vocab tokenizer va vocab model lech.")
      tgtF[b * T + t] = float32(tgt)
      inc nValid
  if nValid == 0:
    raise newException(ValueError,
      "softmaxCrossEntropyMasked: batch khong co target hop le; tu choi optimizer step de tranh Adam update tren gradient rong/zero")
  var dLogits = newTensor(logits.shape)
  let r = softmaxCrossEntropyGpu(ctx, logits.data, tgtF, B * T, V, nValid)
  dLogits.data = r.dLogits
  result = (r.loss, dLogits)

# ═══════════════════════════════════════════════════════════════
# CHECKPOINT LORA: lưu base Jamba (đã graft, KHÔNG merge LoRA) + A/B từng
# adapter riêng, để lần sau load lại đúng trạng thái LoRA đang train dở.
# ═══════════════════════════════════════════════════════════════
proc writeFloatArrayChunked(f: File, data: seq[float32], chunk: int = 4096) =
  ## Ghi mang float32 thanh JSON array truc tiep ra file, theo chunk de
  ## KHONG bao gio dung 1 chuoi/$JsonNode khang lo trong RAM. Moi so dung
  ## formatFloat(ffDecimal, precision=8) - du chinh xac cho checkpoint fp32
  ## va tuong thich 100% voi parseFile/getFloat ben loadCheckpointLora.
  f.write('[')
  let n = data.len
  var i = 0
  while i < n:
    let endIdx = min(i + chunk, n)
    var buf = newStringOfCap((endIdx - i) * 16)
    for j in i ..< endIdx:
      if j > 0 or i > 0: buf.add(',')
      buf.add(formatFloat(data[j], ffDecimal, 8))
    f.write(buf)
    i = endIdx
  f.write(']')

proc numelShape(shape: seq[int]): int =
  ## Helper cuc bo (khong dung numel() private cua nimformer.nim) - tinh
  ## tong so phan tu tu shape, ON DINH qua lai giua cac buoc train (khac
  ## voi .data.len co the =0 sau khi bi discard). Dung lam "paramLen" tin
  ## cay cho AttnWeightDiskStore.hasW/loadW/saveW.
  result = 1
  for s in shape: result *= s

type
  AttnWeightDiskStore* = object
    ## MOI - doi xung voi AdamDiskStore: weight FULL-PRECISION cua block
    ## Attention ('A', ~3.5GB tho cho 12 block) nam tren SSD, chi 1 block
    ## duoc nap vao RAM dung luc forward/backward/checkpoint-save can toi.
    ## Format file (.attnw): giong het AdamDiskStore ve tinh than - header
    ## nho (magic+n) roi raw float32, KHONG qua JSON/JsonNode.
    root*: string

proc initAttnWeightDiskStore*(root: string): AttnWeightDiskStore =
  createDir(root)
  result.root = root

proc attnSafeName(name: string): string =
  result = newStringOfCap(name.len + 6)
  for c in name:
    if c.isAlphaNumeric or c in {'_', '-', '.'}:
      result.add(c)
    else:
      result.add('_')
  result.add(".attnw")

proc attnPath(store: AttnWeightDiskStore, name: string): string =
  store.root / attnSafeName(name)

const AttnWeightMagic = 0x4E5442594157544D'u64   # "ATTNW"-ish, khac magic cua AdamDiskStore

proc hasW*(store: AttnWeightDiskStore, name: string, paramLen: int): bool =
  let path = attnPath(store, name)
  if not fileExists(path): return false
  var f: File
  if not open(f, path, fmRead): return false
  defer: f.close()
  var magic: uint64
  var n: int64
  if f.readBuffer(addr magic, sizeof(magic)) != sizeof(magic) or
     f.readBuffer(addr n, sizeof(n)) != sizeof(n):
    return false
  result = magic == AttnWeightMagic and n == int64(paramLen)

proc saveW*(store: AttnWeightDiskStore, name: string, data: seq[float32]) =
  let path = attnPath(store, name)
  let tmp = path & ".tmp"
  var f: File
  if not open(f, tmp, fmWrite):
    raise newException(IOError, "Khong mo duoc attn weight temp: " & tmp)
  var magic = AttnWeightMagic
  var n = int64(data.len)
  var ok = true
  try:
    ok = f.writeBuffer(addr magic, sizeof(magic)) == sizeof(magic) and
         f.writeBuffer(addr n, sizeof(n)) == sizeof(n)
    if ok and data.len > 0:
      let nBytes = data.len * sizeof(float32)
      ok = f.writeBuffer(unsafeAddr data[0], nBytes) == nBytes
  finally:
    f.close()
  if not ok:
    # Dia day: KHONG commit file cat cut (hasW chi doc header se tuong la hop le).
    try: removeFile(tmp) except OSError: discard
    raise newException(IOError, "Ghi attn weight thieu byte (dia day?): " & tmp)
  if cRename(tmp.cstring, path.cstring) != 0:
    raise newException(IOError, "Khong the commit attn weight atomic: " & tmp)

proc loadW*(store: AttnWeightDiskStore, name: string, paramLen: int): seq[float32] =
  let path = attnPath(store, name)
  var f: File
  if not open(f, path, fmRead):
    raise newException(IOError, "Khong doc duoc attn weight: " & path)
  defer: f.close()
  var magic: uint64
  var n: int64
  if f.readBuffer(addr magic, sizeof(magic)) != sizeof(magic) or
     f.readBuffer(addr n, sizeof(n)) != sizeof(n):
    raise newException(IOError, "Attn weight header bi loi: " & path)
  if magic != AttnWeightMagic or n != int64(paramLen):
    raise newException(IOError, "Attn weight khong tuong thich (" & $n & " != " & $paramLen & "): " & path)
  result = newSeq[float32](paramLen)
  if paramLen > 0:
    let nBytes = paramLen * sizeof(float32)
    if f.readBuffer(addr result[0], nBytes) != nBytes:
      raise newException(IOError, "Attn weight thieu du lieu: " & path)

proc cleanupAttn*(store: var AttnWeightDiskStore) =
  ## Chi goi sau checkpoint CUOI (giong het adamStore.cleanup).
  try:
    for f in walkFiles(store.root / "*.attnw"):
      removeFile(f)
    for f in walkFiles(store.root / "*.attnw.tmp"):
      removeFile(f)
  except OSError: discard

# Kiem update step 0 (auditStep0LoraBlocks): meta[Step0AuditMetaKey] = "ok" trong .bin
# step 1 = moi block LoRA da duoc xac nhan co update step 0 tinh tu gradient THAT
# (lm_head that, dung batch step 0). Thieu co -> lan chay sau tu kiem + sua.
const Step0AuditMetaKey* = "step0Audit"
var gLoadedCkptMeta*: Table[string, string]   # meta .bin vua nap (loadCheckpointLoraBin)
var gStep0Verified* = false                    # process nay da xac nhan step 0 -> ghi co

proc saveCheckpointLoraBin*(m: var NimformerModelJamba, lora: JambaLora, path: string,
                            epoch: int, newBlockParamNames: seq[string],
                            attnStore: AttnWeightDiskStore) =
  ## LoRA-only checkpoint cho inference / fallback resume — ghi QLoRA 4-bit để nhẹ đĩa.
  ## Base Mamba chỉ tham chiếu qua meta nimqPath, không copy vào .bin.
  ## Adapter A/B giữ fp32; newBlock quantize 4-bit.
  ## Resume train uu tien FP32 Attention + Adam tren SSD; NIMBIN4 la fallback portable.

  var meta = initTable[string, string]()
  meta["format"] = "nimformers-lora-v1-qlora4"
  meta["epoch"] = $epoch
  meta["nimqPath"] = CFG_NIMQ_PATH
  meta["pattern"] = m.pattern
  meta["vocab"] = $m.vocab
  meta["hidden"] = $m.embedDim
  meta["nHeads"] = $m.nHeads
  meta["ffMult"] = $m.ffMult
  meta["quantKind"] = "mixed:qkFp4KM-or-fp32-raw"
  meta["quantGroupSize"] = "mixed:fp4-km-group32-or-fp32-raw"
  meta["trainStepSemantics"] = "step-not-epoch"
  if epoch == 1 and (gStep0Verified or
                     gLoadedCkptMeta.getOrDefault(Step0AuditMetaKey) == "ok"):
    meta[Step0AuditMetaKey] = "ok"
  meta["trainSeqLen"] = $CFG_SEQ_LEN
  meta["trainBatchSize"] = $CFG_BATCH_SIZE
  # Preset norm cua forward luc TRAIN -> infer_lora doc lai va dung CHINH preset nay
  # (checkpoint cu khong co khoa nay: infer tu do bang bisect, xem infer_lora.runBisect).
  meta["normPreset"] = TRAIN_NORM_PRESET
  meta["mambaNormImpl"] = TRAIN_MAMBA_NORM_IMPL
  var fp32Keep: seq[RawBinTensor] = @[]
  var quantList: seq[(string, QuantTensor)] = @[]
  var requiredAdapters = initTable[string, bool]()
  for name, ad in lora.adapters:
    if ad.r <= 0 or ad.alpha <= 0'f32 or ad.alpha.classify in {fcNan, fcInf, fcNegInf} or
       ad.scale.classify in {fcNan, fcInf, fcNegInf} or
       ad.scale != ad.alpha / float32(ad.r):
      raise newException(IOError, "LoRA alpha/scale khong hop le hoac lech nhau: " & name)
  for i, c in m.pattern:
    let prefix = &"block{i}."
    if c in {'R', 'r', 'P', 'p', 'G', 'g'}:
      requiredAdapters[prefix & "mambaReal.inProj.weight"] = true
      requiredAdapters[prefix & "mambaReal.outProj.weight"] = true
    elif c in {'M', 'm'}:
      requiredAdapters[prefix & "mamba.inProj.weight"] = true
      requiredAdapters[prefix & "mamba.outProj.weight"] = true
  for name, _ in requiredAdapters:
    if name notin lora.adapters:
      raise newException(IOError, "Khong export checkpoint thieu LoRA adapter bat buoc: " & name)
  for name, _ in lora.adapters:
    if name notin requiredAdapters:
      raise newException(IOError, "Khong export checkpoint co LoRA ngoai pattern: " & name)
  for name, ad in lora.adapters:
    if ad.r <= 0 or ad.inF <= 0 or ad.outF <= 0 or
       ad.A.shape != @[ad.r, ad.inF] or ad.B.shape != @[ad.outF, ad.r] or
       ad.A.data.len != ad.r * ad.inF or ad.B.data.len != ad.outF * ad.r:
      raise newException(IOError, "Khong export adapter LoRA sai shape/data: " & name)
    for v in ad.A.data:
      if v.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(IOError, "Khong export LoRA A co NaN/Inf: " & name)
    var bNonZero = false
    for v in ad.B.data:
      if v.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(IOError, "Khong export LoRA B co NaN/Inf: " & name)
      if v != 0'f32: bNonZero = true
    if epoch >= 1 and not bNonZero:
      # CAM layer trong: sau >= 1 step train moi adapter phai co B != 0 (B = 0 = block
      # chua tung duoc backward, dung nhu bug 102/128 adapter cu).
      if getEnv("BYBY_FORCE_PARTIAL_COMMIT", "0") == "0":
        raise newException(IOError,
          &"KHONG ghi checkpoint step={epoch}: adapter '{name}' co B = 0 (layer chua duoc train). " &
          "BYBY_FORCE_PARTIAL_COMMIT=1 neu that su muon ghi checkpoint thieu layer.")
      stderr.writeLine &"[ckpt] CANH BAO (BYBY_FORCE_PARTIAL_COMMIT=1): adapter '{name}' B = 0 van ghi."
    meta["adapter." & name & ".inF"] = $ad.inF
    meta["adapter." & name & ".outF"] = $ad.outF
    meta["adapter." & name & ".r"] = $ad.r
    meta["adapter." & name & ".alpha"] = $ad.alpha
    fp32Keep.add(RawBinTensor(name: "adapter::" & name & "::A", shape: ad.A.shape, data: ad.A.data))
    fp32Keep.add(RawBinTensor(name: "adapter::" & name & "::B", shape: ad.B.shape, data: ad.B.data))

  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  for name in newBlockParamNames:
    if name notin byName:
      raise newException(IOError, "Checkpoint thieu newBlock parameter trong model: " & name)
    let p = byName[name]
    meta["newBlock." & name & ".shape"] = $p[].shape
    let expected = numelShape(p[].shape)
    var values: seq[float32]
    if lora.quantBase.len > 0 and attnStore.hasW(name, expected):
      # Train streaming: sidecar SSD la ban CHINH (trainStep nap tu day va ghi lai sau
      # update). p[].data co the con gia tri .bin CU (resume backward bo qua block nay
      # trong process nay; [resume-layer]/COMMIT_ONLY khong chay step) -> KHONG dung no.
      values = attnStore.loadW(name, expected)
    else:
      values = p[].data
      if values.len == 0:
        if not attnStore.hasW(name, expected):
          raise newException(IOError, "Thieu weight block moi trong RAM/SSD: " & name)
        values = attnStore.loadW(name, expected)
    if expected <= 0 or values.len != expected:
      raise newException(IOError,
        &"Khong export newBlock '{name}': shape={p[].shape}, expected={expected}, data={values.len}")
    for v in values:
      if v.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(IOError, "Khong export newBlock co NaN/Inf: " & name)
    # SUA (nghi phạm "rác Airlines AirAir"): trước đây ép int4 asymmetric
    # per-ROW (bỏ qua groupSize) cho weight và per-TENSOR cho bias -> lỗi RMS
    # ~13-15% đúng trên phần được train full. Nay: 1-D (bias/gamma) giữ fp32
    # raw; 2-D chia hết 32 dùng fp4-KM group-32 (ĐÚNG như base .nimq và là nhóm DUY NHẤT mà
    # kernel GEMV GPU giải mã được: buildFp4GemvProgram chỉ hỗ trợ groupSize=32); còn lại fp32 raw.
    #
    # SUA2: attn.qkvProj / attn.outProj / fuseGate.weight được train full từ 0
    # với lr rất nhỏ (5e-5 / 2e-5). Giá trị học được nằm gần mức nhiễu FP4-KM
    # group-32 → quantize rồi infer chạy trên số khác hẳn số master fp32 đã
    # train. Giữ chúng fp32 raw trong mixed checkpoint (meta vẫn
    # quantKind=mixed:qkFp4KM-or-fp32-raw); mọi newBlock khác vẫn FP4-KM.
    let isSensitiveFullTrain =
      name.contains(".attn.") or name.contains(".fuseGate.")
    if (not isSensitiveFullTrain) and
       p[].shape.len == 2 and p[].shape[1] mod 32 == 0:
      let qt = quantizeNewBlock(values, p[].shape, qkFp4KM, 32)
      quantList.add(("newBlock::" & name, qt))
    else:
      fp32Keep.add(RawBinTensor(name: "newBlock::" & name, shape: p[].shape, data: values))
  # Ghi ra file tam, DOC LAI kiem tra du moi adapter/newBlock roi moi thay file that:
  # file .bin chinh khong bao gio la ban ghi do hoac thieu layer.
  let writingPath = path & ".writing"
  try: removeFile(writingPath)
  except OSError: discard
  saveQLora4Bin(writingPath, meta, fp32Keep, quantList)
  let nQuantSaved = quantList.len
  let nFp32Saved = fp32Keep.len
  # Ban ghi da tren dia; giai phong truoc khi doc lai (G fp32 ~1.5GB) de RAM khong nhan doi.
  fp32Keep.setLen(0)
  quantList.setLen(0)
  block verifyWritten:
    let (vMeta, vFp32, vQuant) = loadQLora4Bin(writingPath)
    var problems: seq[string] = @[]
    let vEpoch = vMeta.getOrDefault("epoch")
    if vEpoch != $epoch:
      problems.add(&"meta epoch='{vEpoch}' != {epoch}")
    var got = initTable[string, RawBinTensor]()
    for t in vFp32: got[t.name] = t
    for name, ad in lora.adapters:
      for part in ["A", "B"]:
        let key = "adapter::" & name & "::" & part
        if key notin got:
          problems.add("thieu " & key)
          continue
        let src = if part == "A": ad.A.data else: ad.B.data
        let t = got[key]
        if t.data.len != src.len:
          problems.add(&"{key}: {t.data.len} phan tu != {src.len}")
        elif t.data != src:
          problems.add(key & ": du lieu doc lai khac du lieu ghi")
    for name in newBlockParamNames:
      if ("newBlock::" & name) notin got and name notin vQuant:
        problems.add("thieu newBlock::" & name)
    if problems.len > 0:
      try: removeFile(writingPath)
      except OSError: discard
      raise newException(IOError,
        &"[ckpt] Doc lai checkpoint vua ghi THAT BAI ({problems.len} loi, vd: {problems[0]}) - " &
        &"KHONG thay {path}.")
    echo &"[ckpt] Doc lai OK: {lora.adapters.len} adapter A/B khop tung phan tu, " &
         &"{newBlockParamNames.len} newBlock du."
  # Keep the last committed candidate before replacing the canonical file.
  # If the process dies after this rename but before Adam + resume marker are
  # committed, startup can roll the model back to the step named by the marker.
  if fileExists(path):
    let backupTmp = path & ".prev.tmp"
    let backup = path & ".prev"
    try: removeFile(backupTmp)
    except OSError: discard
    # Same-volume hard link makes each-step checkpoint protection effectively
    # O(1) in disk bandwidth. Fall back to a full copy on filesystems that
    # reject hard links.
    if cLink(path.cstring, backupTmp.cstring) != 0:
      try:
        copyFile(path, backupTmp)
      except OSError as e:
        raise newException(IOError, "Khong the tao checkpoint rollback backup: " & e.msg)
    if cRename(backupTmp.cstring, backup.cstring) != 0:
      try: removeFile(backupTmp)
      except OSError: discard
      raise newException(IOError, "Khong the commit checkpoint rollback backup: " & backupTmp)
  if cRename(writingPath.cstring, path.cstring) != 0:
    raise newException(IOError, "Khong the thay checkpoint bang ban da kiem tra: " & writingPath)
  echo &"[ckpt] Da luu QLoRA4 binary {path}: {lora.adapters.len} adapter (fp32), " &
       &"{nQuantSaved} newBlock (fp4-km), {nFp32Saved} tensor fp32 raw; KHONG luu base Mamba."

proc loadCheckpointLoraBin*(path: string): tuple[model: NimformerModelJamba, lora: JambaLora, epoch: int] =
  ## Hỗ trợ cả .bin fp32 (NIMBIN1) và QLoRA 4-bit (NIMBIN4 — quantize_lora_bin_4bit.nim).
  ## SUA (bug RAM, xem giai thich day du trong infer_lora.loadBinaryLoraForInfer):
  ## newBlock KHONG con dequant fp32 ngay luc load nua - giu QuantTensor, gop
  ## vao lora.quantBase CHUNG voi base tu .nimq, de mergeIn()/restoreBase()
  ## (dung khi train, lora.nim) tu dequant-on-the-fly + giai phong.
  var meta: Table[string, string]
  var tensors: seq[RawBinTensor]
  var newBlockQuant: Table[string, QuantTensor]
  if isQLora4Bin(path):
    echo "[ckpt] Phát hiện NIMBIN4 (QLoRA 4-bit) — newBlock giữ nén, dequant on-the-fly qua quantBase..."
    let q = loadQLora4Bin(path)
    meta = q.meta
    tensors = q.fp32Tensors
    newBlockQuant = q.quantWeights
  else:
    let loaded = loadRawBin(path)
    meta = loaded.meta
    tensors = loaded.tensors
  let fmt = meta.getOrDefault("format")
  if fmt != "nimformers-lora-v1" and fmt != "nimformers-lora-v1-qlora4":
    raise newException(ValueError, "Khong phai LoRA-only checkpoint: " & path)
  gLoadedCkptMeta = meta
  let savedNormPreset = meta.getOrDefault("normPreset")
  let savedMambaNormImpl = meta.getOrDefault("mambaNormImpl")
  if savedNormPreset != TRAIN_NORM_PRESET or savedMambaNormImpl != TRAIN_MAMBA_NORM_IMPL:
    raise newException(ValueError,
      &"[resume] Checkpoint norm khong khop trainer hien tai: normPreset='{savedNormPreset}', " &
       &"mambaNormImpl='{savedMambaNormImpl}', expected '{TRAIN_NORM_PRESET}' / " &
       &"'{TRAIN_MAMBA_NORM_IMPL}'. Tu choi resume de tranh tron optimizer state voi forward math khac.")
  let pattern = meta["pattern"]
  let nimqPath = meta["nimqPath"]
  var m = newNimformerModelJamba(parseInt(meta["vocab"]), parseInt(meta["hidden"]),
                                  parseInt(meta["nHeads"]), parseInt(meta["ffMult"]), pattern,
                                  useSession = true, mambaRealDState = CFG_CODESTRAL_DSTATE,
                                  mambaRealNGroups = CFG_CODESTRAL_NGROUPS,
                                  mambaRealConvKernel = CFG_CODESTRAL_CONVKERNEL,
                                  mambaRealExpand = CFG_CODESTRAL_EXPAND, quantResident = true)
  var quantResident = loadCodestralQuantResidentFromNimq(m, nimqPath)
  syncRmsFromBaseNorms(m)
  for name, qt in newBlockQuant:
    quantResident[name] = qt
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  var expectedNewBlocks = initTable[string, bool]()
  for key, _ in meta:
    if key.startsWith("newBlock.") and key.endsWith(".shape"):
      expectedNewBlocks[key[9 ..< key.len - 6]] = true
  var requiredNewBlocks = initTable[string, bool]()
  var fullBlockIndices: seq[int] = @[]
  for i, c in pattern:
    if c notin {'R', 'r', 'M', 'm'}: fullBlockIndices.add(i)
  for (name, _) in paramList(m):
    for bi in fullBlockIndices:
      let prefix = &"block{bi}."
      if not name.startsWith(prefix): continue
      if m.blocks[bi].kind == bkMambaRealPlusAttn and not name.startsWith(prefix & "attn."):
        continue
      if m.blocks[bi].kind == bkMambaRealGated and
         not (name.startsWith(prefix & "attn.") or name.startsWith(prefix & "fuseGate.")):
        continue
      requiredNewBlocks[name] = true
      break
  if fmt == "nimformers-lora-v1-qlora4" and expectedNewBlocks.len == 0:
    raise newException(ValueError,
      "Checkpoint QLoRA4 khong co manifest newBlock.shape; khong the chung minh checkpoint day du")
  for name, _ in requiredNewBlocks:
    if name notin expectedNewBlocks:
      raise newException(ValueError, "Checkpoint manifest thieu newBlock bat buoc: " & name)
  for name, _ in expectedNewBlocks:
    if name notin requiredNewBlocks:
      raise newException(ValueError, "Checkpoint manifest co newBlock khong thuoc model/pattern: " & name)
    if name notin byName or meta.getOrDefault("newBlock." & name & ".shape") != $byName[name][].shape:
      raise newException(ValueError, "Checkpoint manifest newBlock sai shape: " & name)
  var seenNewBlocks = initTable[string, bool]()
  for name, _ in newBlockQuant:
    if name in seenNewBlocks:
      raise newException(ValueError, "Checkpoint trung newBlock quant: " & name)
    seenNewBlocks[name] = true
  for tensor in tensors:
    if tensor.name.startsWith("newBlock::"):
      let name = tensor.name[10 .. ^1]
      if name in seenNewBlocks:
        raise newException(ValueError,
          "Checkpoint co hai nguon newBlock (raw/quant hoac trung ten): " & name)
      seenNewBlocks[name] = true
  for name, _ in expectedNewBlocks:
    if name notin seenNewBlocks:
      raise newException(ValueError, "Checkpoint thieu newBlock weight: " & name)
  for name, qt in newBlockQuant:
    if name notin byName:
      raise newException(ValueError, "Checkpoint co newBlock khong ton tai trong model: " & name)
    if name notin expectedNewBlocks or qt.shape != byName[name][].shape or qt.data.len == 0:
      raise newException(ValueError, "Checkpoint newBlock quant sai shape/data: " & name)
  var lo: JambaLora
  lo.adapters = initTable[string, LoraAdapter]()
  lo.backup = initTable[string, seq[float32]]()
  var seenAdapterA = initTable[string, bool]()
  var seenAdapterB = initTable[string, bool]()
  for tensor in tensors:
    if tensor.name.startsWith("newBlock::"):
      let name = tensor.name[10 .. ^1]
      if name notin byName or name notin expectedNewBlocks:
        raise newException(ValueError, "Checkpoint co newBlock FP32 khong duoc khai bao/khong ton tai: " & name)
      if tensor.shape != byName[name][].shape or tensor.data.len != numelShape(tensor.shape):
        raise newException(ValueError, "Checkpoint newBlock FP32 sai shape/data: " & name)
      byName[name][].data = tensor.data
    elif tensor.name.startsWith("adapter::"):
      let rest = tensor.name[9 .. ^1]
      let splitAt = rest.rfind("::")
      if splitAt < 0: continue
      let adapterName = rest[0 ..< splitAt]
      let part = rest[splitAt + 2 .. ^1]
      if adapterName notin lo.adapters:
        let adIn = parseInt(meta["adapter." & adapterName & ".inF"])
        let adOut = parseInt(meta["adapter." & adapterName & ".outF"])
        let adRank = parseInt(meta["adapter." & adapterName & ".r"])
        let adAlpha = parseFloat(meta["adapter." & adapterName & ".alpha"]).float32
        if adIn <= 0 or adOut <= 0 or adRank <= 0:
          raise newException(ValueError, "Checkpoint LoRA co inF/outF/r khong hop le: " & adapterName)
        var ad: LoraAdapter
        ad.inF = adIn; ad.outF = adOut; ad.r = adRank
        ad.alpha = adAlpha; ad.scale = adAlpha / float32(adRank)
        ad.A = newTensor(@[adRank, adIn]); ad.B = newTensor(@[adOut, adRank])
        lo.adapters[adapterName] = ad
      if part == "A":
        if adapterName in seenAdapterA:
          raise newException(ValueError, "Checkpoint LoRA trung tensor A: " & adapterName)
        if tensor.shape != @[lo.adapters[adapterName].r, lo.adapters[adapterName].inF] or
           tensor.data.len != lo.adapters[adapterName].r * lo.adapters[adapterName].inF:
          raise newException(ValueError, "Checkpoint LoRA A sai shape/so phan tu: " & adapterName)
        lo.adapters[adapterName].A.data = tensor.data
        seenAdapterA[adapterName] = true
      elif part == "B":
        if adapterName in seenAdapterB:
          raise newException(ValueError, "Checkpoint LoRA trung tensor B: " & adapterName)
        if tensor.shape != @[lo.adapters[adapterName].outF, lo.adapters[adapterName].r] or
           tensor.data.len != lo.adapters[adapterName].outF * lo.adapters[adapterName].r:
          raise newException(ValueError, "Checkpoint LoRA B sai shape/so phan tu: " & adapterName)
        lo.adapters[adapterName].B.data = tensor.data
        seenAdapterB[adapterName] = true
      else:
        raise newException(ValueError, "Checkpoint LoRA co tensor part la: " & part)
  for name, ad in lo.adapters:
    if name notin seenAdapterA or name notin seenAdapterB:
      raise newException(ValueError,
        "Checkpoint LoRA thieu A/B cho adapter '" & name & "' - tu choi resume/export hong")
    if ad.A.data.len != ad.r * ad.inF or ad.B.data.len != ad.outF * ad.r:
      raise newException(ValueError, "Checkpoint LoRA A/B rong hoac sai kich thuoc: " & name)
  var requiredAdapters = initTable[string, bool]()
  for i, c in pattern:
    let prefix = &"block{i}."
    if c in {'R', 'r', 'P', 'p', 'G', 'g'}:
      requiredAdapters[prefix & "mambaReal.inProj.weight"] = true
      requiredAdapters[prefix & "mambaReal.outProj.weight"] = true
    elif c in {'M', 'm'}:
      requiredAdapters[prefix & "mamba.inProj.weight"] = true
      requiredAdapters[prefix & "mamba.outProj.weight"] = true
  for name, _ in requiredAdapters:
    if name notin lo.adapters:
      raise newException(ValueError, "Checkpoint thieu LoRA adapter bat buoc theo pattern: " & name)
  for name, _ in lo.adapters:
    if name notin requiredAdapters:
      raise newException(ValueError, "Checkpoint LoRA adapter khong thuoc pattern: " & name)
  lo.quantBase = quantResident
  bindQuantBaseToLinears(m, lo)
  result = (m, lo, parseInt(meta["epoch"]))

proc saveCheckpointLora(m: var NimformerModelJamba, lora: JambaLora, path: string, epoch: int,
                         newBlockParamNames: seq[string], attnStore: AttnWeightDiskStore) =
  ## SUA LAN 3 (OOM khi luu): KHONG dung %*(seq[float32]) nua - JsonNode
  ## tree cho mang lon (vd 144 tham so block 'A' ~3.5GB thô) phinh gấp 5-10x
  ## trong RAM roi moi $j ra chuoi -> OOM killer dung luc save. Thay bang
  ## ghi text JSON streaming/chunked truc tiep ra file (chi can ~chunk float
  ## trong RAM tai 1 thoi diem). Format van la JSON hop le, loadCheckpointLora
  ## doc lai binh thuong, KHONG doi phia load.
  ## Van KHONG ghi .bin - phan 'R' (Mamba) van tai tao tu .nimq nhu cu.
  ##
  ## SUA THEM (attnStore): tu khi trainStepGraftStreaming discard weight
  ## block 'A' khoi RAM ngay sau moi buoc train (xem AttnWeightDiskStore),
  ## byName[name][].data o DAY co the dang RONG luc ham nay chay (goi sau
  ## 1 buoc train, khong phai ngay sau checkpoint load). Nap TAM tu SSD
  ## dung luc ghi, tra lai trang thai rong ngay sau - KHONG bao gio giu ca
  ## ~3.5GB do o RAM lau hon 1 lan ghi file.
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  var serializedAlpha = CFG_LORA_ALPHA
  var haveSerializedAlpha = false
  for name, ad in lora.adapters:
    if ad.r <= 0 or ad.alpha <= 0'f32 or ad.alpha.classify in {fcNan, fcInf, fcNegInf} or
       ad.scale.classify in {fcNan, fcInf, fcNegInf} or
       ad.scale != ad.alpha / float32(ad.r):
      raise newException(IOError, "LoRA alpha/scale khong hop le hoac lech nhau: " & name)
    if not haveSerializedAlpha:
      serializedAlpha = ad.alpha
      haveSerializedAlpha = true
    elif ad.alpha != serializedAlpha:
      raise newException(IOError,
        "JSON checkpoint chi ho tro alpha chung; adapter co alpha khac nhau: " & name)
  let outPath = path & ".lora.json"
  let f = open(outPath, fmWrite)
  defer: f.close()
  f.write("{\"epoch\":")
  f.write($epoch)
  f.write(",\"rank\":")
  f.write($CFG_LORA_RANK)
  f.write(",\"alpha\":")
  f.write(formatFloat(serializedAlpha, ffDecimal, 8))
  f.write(",\"nimqPath\":")
  f.write(escapeJson(CFG_NIMQ_PATH))
  f.write(",\"pattern\":")
  f.write(escapeJson(m.pattern))
  f.write(",\"adapters\":{")
  var firstAd = true
  for name, ad in lora.adapters:
    if not firstAd: f.write(',')
    firstAd = false
    f.write(escapeJson(name))
    f.write(":{\"A\":")
    writeFloatArrayChunked(f, ad.A.data)
    f.write(",\"B\":")
    writeFloatArrayChunked(f, ad.B.data)
    f.write(",\"inF\":")
    f.write($ad.inF)
    f.write(",\"outF\":")
    f.write($ad.outF)
    f.write(",\"r\":")
    f.write($ad.r)
    f.write(",\"alpha\":")
    f.write(formatFloat(ad.alpha, ffDecimal, 8))
    f.write('}')
  f.write("},\"newBlockWeights\":{")
  var firstNb = true
  for name in newBlockParamNames:
    if name notin byName:
      raise newException(IOError, "[ckpt] DUNG: thieu newBlock parameter trong model: " & name)
    let p = byName[name]
    let wasEmpty = p[].data.len == 0
    if wasEmpty:
      let plen = numelShape(p[].shape)
      if attnStore.hasW(name, plen):
        p[].data = attnStore.loadW(name, plen)
      else:
        raise newException(IOError,
          &"[ckpt] DUNG: '{name}' rong ca RAM lan attnStore - checkpoint se thieu weight")
    if not firstNb: f.write(',')
    firstNb = false
    f.write(escapeJson(name))
    f.write(':')
    writeFloatArrayChunked(f, p[].data)
    if wasEmpty:
      p[].data = @[]   # tra lai trang thai rong nhu truoc khi vao day (duong streaming)
  f.write("}}")
  f.flushFile()
  echo &"[ckpt] Da luu adapter LoRA + {newBlockParamNames.len} tham so block Attention (full, STREAMING) " &
       &"-> {outPath}, epoch={epoch} (KHONG ghi .bin - phan Mamba tai tao tu {CFG_NIMQ_PATH})"

proc loadCheckpointLora*(path: string): tuple[model: NimformerModelJamba, lora: JambaLora, epoch: int] =
  ## SUA (OOM khi resume - bug that, doi xung voi "SUA LAN 3" o
  ## saveCheckpointLora): ham CU dung parseFile (std/json) - dung NGUYEN CAY
  ## JsonNode cho CA FILE, ke ca cac mang float LON (newBlockWeights ~3.5GB
  ## tho cho block Attention, adapters A/B). Day CHINH LA loai loi da duoc
  ## xac nhan va sua o PHIA GHI (comment goc: "JsonNode tree cho mang lon
  ## phinh gap 5-10x RAM roi moi $j ra chuoi -> OOM killer dung luc save")
  ## nhung CHUA TUNG duoc sua o PHIA DOC - nen MOI LAN resume (goi ham nay)
  ## van OOM y het ly do DA duoc sua rieng cho save, giai thich dung hien
  ## tuong nguoi dung gap: epoch 1 xong (streaming train, an toan) nhung
  ## resume tu checkpoint lai OOM ngay (load y nguyen ca file qua JsonNode).
  ##
  ## Fix: tu viet STREAMING parser bang std/parsejson (SAX-style, event-
  ## based, KHONG dung cay JsonNode nao ca) - doc TUNG SO MOT thang vao
  ## seq/tensor dich. File van la JSON hop le y het cu (khong doi format,
  ## tuong thich nguoc voi checkpoint DA CO san cua nguoi dung).
  let fname = path & ".lora.json"
  var p: JsonParser
  p.open(newFileStream(fname, fmRead), fname)
  defer: p.close()

  proc readFloatArray(): seq[float32] =
    if p.kind != jsonArrayStart: raiseParseErr(p, "mong doi mang so (jsonArrayStart)")
    p.next()   # qua khoi '['
    result = @[]
    while p.kind != jsonArrayEnd:
      let v = if p.kind == jsonInt: float(p.getInt()) else: p.getFloat()
      result.add(float32(v))
      p.next()
    p.next()   # qua khoi ']'

  proc nextKey(): string =
    ## Doc 1 "key" cua object HIEN TAI (parser dang dung ngay truoc no -
    ## jsonString hoac jsonObjectEnd, xem quy uoc trong ham chinh ben duoi).
    ## Neu la key: tra ve ten, dua parser toi GIA TRI tuong ung. Neu la het
    ## object (jsonObjectEnd): tra ve "" (KHONG tu next() qua no - nguoi goi
    ## tu lam dieu do sau khi vong lap ket thuc, xem quy uoc duoi).
    if p.kind == jsonObjectEnd: return ""
    if p.kind != jsonString: raiseParseErr(p, "mong doi ten truong (jsonString)")
    result = p.str
    p.next()   # tien toi GIA TRI cua key nay

  var epoch = 0
  var nimqPath = ""
  var pattern = ""
  var checkpointAlpha = CFG_LORA_ALPHA
  var checkpointAlphaFound = false
  var m: NimformerModelJamba
  var modelReady = false
  var byName = initTable[string, ptr Tensor]()
  var quantResident: Table[string, QuantTensor]
  var requiredAdapters = initTable[string, bool]()
  var requiredNewBlocks = initTable[string, bool]()
  var seenNewBlocks = initTable[string, bool]()
  var lo: JambaLora
  lo.adapters = initTable[string, LoraAdapter]()
  lo.backup = initTable[string, seq[float32]]()

  proc ensureModel() =
    ## Dung nimqPath+pattern (da doc truoc do trong file, theo dung thu tu
    ## saveCheckpointLora ghi ra) de dung model NGAY khi bat dau can (luc
    ## gap key "adapters" hoac "newBlockWeights") - cho phep ghi
    ## newBlockWeights THANG vao byName[name][].data (khong qua seq trung
    ## gian), giong het ky thuat ham CU dung khi con doc tu JsonNode.
    if modelReady: return
    doAssert nimqPath.len > 0 and pattern.len > 0,
      "checkpoint JSON thieu nimqPath/pattern TRUOC adapters/newBlockWeights (sai thu tu ghi?)"
    m = newNimformerModelJamba(CFG_CODESTRAL_VOCAB, CFG_CODESTRAL_HIDDEN, CFG_CODESTRAL_NHEADS,
                                ffMult = 4, pattern = pattern, useSession = true,
                                mambaRealDState = CFG_CODESTRAL_DSTATE,
                                mambaRealNGroups = CFG_CODESTRAL_NGROUPS,
                                mambaRealConvKernel = CFG_CODESTRAL_CONVKERNEL,
                                mambaRealExpand = CFG_CODESTRAL_EXPAND, quantResident = true)
    quantResident = loadCodestralQuantResidentFromNimq(m, nimqPath)
    syncRmsFromBaseNorms(m)
    for (name, t) in paramList(m): byName[name] = t
    for i, c in pattern:
      let prefix = &"block{i}."
      if c in {'R', 'r', 'P', 'p', 'G', 'g'}:
        requiredAdapters[prefix & "mambaReal.inProj.weight"] = true
        requiredAdapters[prefix & "mambaReal.outProj.weight"] = true
      elif c in {'M', 'm'}:
        requiredAdapters[prefix & "mamba.inProj.weight"] = true
        requiredAdapters[prefix & "mamba.outProj.weight"] = true
      if c notin {'R', 'r', 'M', 'm', 'G', 'g'}:
        for name, _ in byName:
          if name.startsWith(prefix) and
            (m.blocks[i].kind != bkMambaRealPlusAttn or name.startsWith(prefix & "attn.")) and
            (m.blocks[i].kind != bkMambaRealGated or name.startsWith(prefix & "attn.") or
             name.startsWith(prefix & "fuseGate.")):
            requiredNewBlocks[name] = true
    # Checkpoint data replaces random-init tensors for newly grafted blocks.
    # Release those payloads before reading large arrays from the stream.
    for i, c in pattern:
      if c in {'A', 'a'}:
        let prefix = "block" & $i & "."
        for name, t in byName:
          if name.startsWith(prefix): t[].data = @[]
    modelReady = true

  p.next()
  if p.kind != jsonObjectStart: raiseParseErr(p, "checkpoint phai la 1 JSON object o goc")
  p.next()   # qua khoi '{' cua root
  while true:
    let key = nextKey()
    if key == "": break
    case key
    of "epoch":
      if p.kind != jsonInt: raiseParseErr(p, "epoch phai la so nguyen")
      epoch = int(p.getInt()); p.next()
    of "rank":
      # Rank that su cua tung adapter doc tu shape/metadata rieng ben duoi.
      p.next()
    of "alpha":
      # Adapter scale la alpha/r. Phai resume alpha tu checkpoint, khong lay
      # CFG_LORA_ALPHA cua binary hien tai: config co the da doi sau checkpoint.
      if p.kind notin {jsonInt, jsonFloat}:
        raiseParseErr(p, "alpha phai la so")
      checkpointAlpha = (if p.kind == jsonInt: float32(p.getInt())
                         else: float32(p.getFloat()))
      if checkpointAlpha <= 0'f32 or checkpointAlpha.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(ValueError, "Checkpoint JSON LoRA alpha khong hop le")
      checkpointAlphaFound = true
      p.next()
    of "nimqPath":
      if p.kind != jsonString: raiseParseErr(p, "nimqPath phai la chuoi")
      nimqPath = p.str; p.next()
    of "pattern":
      if p.kind != jsonString: raiseParseErr(p, "pattern phai la chuoi")
      pattern = p.str; p.next()
    of "adapters":
      ensureModel()
      if p.kind != jsonObjectStart: raiseParseErr(p, "adapters phai la object")
      p.next()   # qua khoi '{' cua adapters
      while true:
        let adName = nextKey()
        if adName == "": break
        if adName notin requiredAdapters or adName notin byName:
          raise newException(ValueError, "Checkpoint JSON co LoRA adapter ngoai pattern: " & adName)
        if adName in lo.adapters:
          raise newException(ValueError, "Checkpoint JSON trung LoRA adapter: " & adName)
        if p.kind != jsonObjectStart: raiseParseErr(p, "adapter '" & adName & "' phai la object")
        p.next()   # qua khoi '{' cua adapter nay
        var aData, bData: seq[float32]
        var inF, outF, r: int
        var adapterAlpha = checkpointAlpha
        var adapterAlphaFound = false
        while true:
          let fk = nextKey()
          if fk == "": break
          case fk
          of "A": aData = readFloatArray()
          of "B": bData = readFloatArray()
          of "inF":
            if p.kind != jsonInt: raiseParseErr(p, "inF phai la so nguyen")
            inF = int(p.getInt()); p.next()
          of "outF":
            if p.kind != jsonInt: raiseParseErr(p, "outF phai la so nguyen")
            outF = int(p.getInt()); p.next()
          of "r":
            if p.kind != jsonInt: raiseParseErr(p, "r phai la so nguyen")
            r = int(p.getInt()); p.next()
          of "alpha":
            if p.kind notin {jsonInt, jsonFloat}:
              raiseParseErr(p, "adapter alpha phai la so")
            adapterAlpha = (if p.kind == jsonInt: float32(p.getInt())
                            else: float32(p.getFloat()))
            if adapterAlpha <= 0'f32 or adapterAlpha.classify in {fcNan, fcInf, fcNegInf}:
              raise newException(ValueError, "Checkpoint JSON adapter alpha khong hop le: " & adName)
            adapterAlphaFound = true
            p.next()
          else:
            raiseParseErr(p, "truong la trong adapter '" & adName & "': " & fk)
        var ad: LoraAdapter
        if inF <= 0 or outF <= 0 or r <= 0 or
           byName[adName][].shape != @[outF, inF] or
           aData.len != r * inF or bData.len != outF * r:
          raise newException(ValueError, "Checkpoint JSON LoRA sai shape/data: " & adName)
        ad.inF = inF; ad.outF = outF; ad.r = r
        # Checkpoint cũ không ghi alpha thì giữ hành vi tương thích với config.
        ad.alpha = if adapterAlphaFound: adapterAlpha
                   elif checkpointAlphaFound: checkpointAlpha
                   else: CFG_LORA_ALPHA
        ad.scale = ad.alpha / float32(ad.r)
        ad.A = newTensor(@[ad.r, ad.inF]); ad.A.data = aData
        ad.B = newTensor(@[ad.outF, ad.r]); ad.B.data = bData
        lo.adapters[adName] = ad
      p.next()   # qua khoi '}' cua adapters
    of "newBlockWeights":
      ensureModel()
      if p.kind != jsonObjectStart: raiseParseErr(p, "newBlockWeights phai la object")
      p.next()   # qua khoi '{' cua newBlockWeights
      while true:
        let wName = nextKey()
        if wName == "": break
        let arr = readFloatArray()
        if wName notin requiredNewBlocks or wName notin byName:
          raise newException(ValueError, "Checkpoint JSON co newBlock ngoai pattern: " & wName)
        if wName in seenNewBlocks:
          raise newException(ValueError, "Checkpoint JSON trung newBlock: " & wName)
        let expectedLen = numelShape(byName[wName][].shape)
        if arr.len != expectedLen:
          raise newException(ValueError,
            &"checkpoint newBlockWeights['{wName}'] sai kich thuoc: {arr.len} != {expectedLen}")
        byName[wName][].data = arr
        seenNewBlocks[wName] = true
      p.next()   # qua khoi '}' cua newBlockWeights
    else:
      raiseParseErr(p, "truong goc la trong checkpoint: " & key)

  doAssert modelReady, "checkpoint JSON thieu ca adapters lan newBlockWeights (file hong?)"
  for name, _ in requiredAdapters:
    if name notin lo.adapters:
      raise newException(ValueError, "Checkpoint JSON thieu LoRA adapter bat buoc: " & name)
  for name, _ in requiredNewBlocks:
    if name notin seenNewBlocks:
      raise newException(ValueError, "Checkpoint JSON thieu newBlock bat buoc: " & name)
  lo.quantBase = quantResident
  result = (m, lo, epoch)

type
  ## Adam state backing store: m/v/step cua MOI parameter nam tren SSD,
  ## chi 1 state duoc nap vao RAM moi lan update. File .adam duoc giu lai
  ## giua cac train step, con file .tmp chi la scratch va bi xoa sau khi save.
  AdamDiskStore* = object
    root*: string

proc initAdamDiskStore*(root: string): AdamDiskStore =
  createDir(root)
  result.root = root

proc adamSafeName(name: string): string =
  result = newStringOfCap(name.len + 5)
  for c in name:
    if c.isAlphaNumeric or c in {'_', '-', '.'}:
      result.add(c)
    else:
      result.add('_')
  result.add(".adam")

proc adamPath(store: AdamDiskStore, name: string): string =
  store.root / adamSafeName(name)

proc clearAdamState*(st: var ApfAdamState) =
  ## Dọn ngay m/v trong RAM sau khi save SSD xong.
  st.m.setLen(0)
  st.v.setLen(0)
  st.m = @[]
  st.v = @[]
  st.step = 0

proc has*(store: AdamDiskStore, name: string, paramLen: int): bool =
  let path = adamPath(store, name)
  if not fileExists(path):
    return false
  # Header-only validation used to accept a truncated m/v payload as a valid
  # optimizer snapshot. Check the complete on-disk length before trusting its
  # step; otherwise a crash during write could fail much later during resume.
  let expectedBytes = 28'i64 + int64(paramLen) * 2'i64 * int64(sizeof(float32))
  if getFileSize(path) != expectedBytes:
    return false
  var f: File
  if not open(f, path, fmRead):
    return false
  defer: f.close()
  var magic: uint64
  var version: uint32
  var n: int64
  discard f.readBuffer(addr magic, sizeof(magic))
  discard f.readBuffer(addr version, sizeof(version))
  discard f.readBuffer(addr n, sizeof(n))
  result = magic == 0x425942594144414D'u64 and version == 1'u32 and n == int64(paramLen)

proc init*(store: AdamDiskStore, name: string, paramLen: int) =
  ## Khoi tao state 0 tren SSD. KHONG tao seq[float32] kich thuoc paramLen,
  ## tranh mot spike RAM lon ngay luc tao file moi. Chi dung buffer 1 MiB.
  let path = adamPath(store, name)
  # NO early-return when the file already exists. This proc is the ONLY reset
  # primitive used by resetAdamParameters (resume distrust path) and by the G
  # rescue path in trainStepGraftStreaming - both NEED to overwrite an
  # existing (possibly poisoned) optimizer state with zeros. Callers that
  # merely want a fresh file already guard with `if not store.has(...)`.
  let tmp = path & ".tmp"
  var f: File
  if not open(f, tmp, fmWrite):
    raise newException(IOError, "Khong mo duoc Adam state temp: " & tmp)
  var magic = 0x425942594144414D'u64
  var version = 1'u32
  var n = int64(paramLen)
  var step = 0'i64
  discard f.writeBuffer(addr magic, sizeof(magic))
  discard f.writeBuffer(addr version, sizeof(version))
  discard f.writeBuffer(addr n, sizeof(n))
  discard f.writeBuffer(addr step, sizeof(step))
  let totalBytes = int64(paramLen) * int64(sizeof(float32)) * 2
  var zeroBuf = newSeq[byte](1024 * 1024)
  var left = totalBytes
  while left > 0:
    let chunk = min(left, int64(zeroBuf.len))
    discard f.writeBuffer(addr zeroBuf[0], int(chunk))
    left -= chunk
  f.close()
  zeroBuf.setLen(0)
  zeroBuf = @[]
  if cRename(tmp.cstring, path.cstring) != 0:
    raise newException(IOError, "Khong the commit Adam state atomic: " & tmp)

proc load*(store: AdamDiskStore, name: string, paramLen: int): ApfAdamState =
  if not store.has(name, paramLen):
    store.init(name, paramLen)
  let path = adamPath(store, name)
  var f: File
  if not open(f, path, fmRead):
    raise newException(IOError, "Khong doc duoc Adam state: " & path)
  defer: f.close()
  var magic: uint64
  var version: uint32
  var n: int64
  var step: int64
  if f.readBuffer(addr magic, sizeof(magic)) != sizeof(magic) or
     f.readBuffer(addr version, sizeof(version)) != sizeof(version) or
     f.readBuffer(addr n, sizeof(n)) != sizeof(n) or
     f.readBuffer(addr step, sizeof(step)) != sizeof(step):
    raise newException(IOError, "Adam state header bi loi: " & path)
  if magic != 0x425942594144414D'u64 or version != 1'u32 or n != int64(paramLen):
    raise newException(IOError, "Adam state khong tuong thich: " & path)
  result = newApfAdamState(paramLen)
  result.step = int(step)
  if paramLen > 0:
    let mBytes = paramLen * sizeof(float32)
    if f.readBuffer(addr result.m[0], mBytes) != mBytes:
      raise newException(IOError, "Adam m bi thieu du lieu: " & path)
    if f.readBuffer(addr result.v[0], mBytes) != mBytes:
      raise newException(IOError, "Adam v bi thieu du lieu: " & path)

proc save*(store: AdamDiskStore, name: string, st: ApfAdamState) =
  if st.m.len != st.v.len or st.step < 0:
    raise newException(ValueError, "Adam state co m/v length hoac step khong hop le: " & name)
  let path = adamPath(store, name)
  let tmp = path & ".tmp"
  var f: File
  if not open(f, tmp, fmWrite):
    raise newException(IOError, "Khong mo duoc Adam state temp: " & tmp)
  var magic = 0x425942594144414D'u64
  var version = 1'u32
  var n = int64(st.m.len)
  var step = int64(st.step)
  discard f.writeBuffer(addr magic, sizeof(magic))
  discard f.writeBuffer(addr version, sizeof(version))
  discard f.writeBuffer(addr n, sizeof(n))
  discard f.writeBuffer(addr step, sizeof(step))
  if st.m.len > 0:
    discard f.writeBuffer(unsafeAddr st.m[0], st.m.len * sizeof(float32))
    discard f.writeBuffer(unsafeAddr st.v[0], st.v.len * sizeof(float32))
  f.close()
  let expectedBytes = 28'i64 + int64(st.m.len) * 2'i64 * int64(sizeof(float32))
  if not fileExists(tmp) or getFileSize(tmp) != expectedBytes:
    try: removeFile(tmp)
    except OSError: discard
    raise newException(IOError, "Adam state temp bi thieu/thua du lieu: " & tmp)
  if cRename(tmp.cstring, path.cstring) != 0:
    raise newException(IOError, "Khong the commit Adam state atomic: " & tmp)

proc cleanup*(store: var AdamDiskStore) =
  ## Chi goi khi optimizer state khong con can nua (sau checkpoint cuoi).
  ## Xoa ca .adam va .tmp, tranh phinh SSD sau ca run.
  try:
    for f in walkFiles(store.root / "*.adam"):
      removeFile(f)
    for f in walkFiles(store.root / "*.adam.tmp"):
      removeFile(f)
  except OSError: discard

proc loraAdamKey(name, part: string): string =
  ## Dung chung AdamDiskStore cho LoRA A/B.
  result = "__lora__" & part & "__" & name

proc saveLoraAdamStates*(store: AdamDiskStore, state: LoraTrainState) =
  ## Persist m/v/step cua TAT CA LoRA adapters sau khi checkpoint weight da
  ## ghi thanh cong.
  for name, st in state.adamA:
    store.save(loraAdamKey(name, "A"), st)
  for name, st in state.adamB:
    store.save(loraAdamKey(name, "B"), st)

proc restoreLoraAdamStates*(store: AdamDiskStore, state: var LoraTrainState,
                            lora: JambaLora, expectedStep: int): bool =
  ## Restore persisted LoRA m/v. A mid-step crash leaves a MIXED set: blocks
  ## already processed carry step == expectedStep+1 while untouched blocks
  ## still carry expectedStep. Anything outside that window means the state
  ## cannot be trusted for this step, and the affected adapter is reset.
  if expectedStep < 0: return false
  var allOk = true
  for name, ad in lora.adapters:
    let ka = loraAdamKey(name, "A")
    let kb = loraAdamKey(name, "B")
    var ok = store.has(ka, ad.A.data.len) and store.has(kb, ad.B.data.len)
    if ok:
      let sa = store.load(ka, ad.A.data.len)
      let sb = store.load(kb, ad.B.data.len)
      let aOk = sa.step == expectedStep or sa.step == expectedStep + 1
      let bOk = sb.step == expectedStep or sb.step == expectedStep + 1
      if aOk and bOk:
        state.adamA[name] = sa
        state.adamB[name] = sb
      else:
        ok = false
    if not ok:
      allOk = false
      state.adamA[name] = newApfAdamState(ad.A.data.len)
      state.adamB[name] = newApfAdamState(ad.B.data.len)
  result = allOk

proc resetAdamParameters*(store: AdamDiskStore, model: var NimformerModelJamba,
                          names: seq[string]) =
  ## Xoa/reset TOAN BO state Adam cua block moi neu resume phat hien mismatch.
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(model): byName[name] = t
  for name in names:
    if name notin byName: continue
    let path = adamPath(store, name)
    try: removeFile(path)
    except OSError: discard
    try: removeFile(path & ".tmp")
    except OSError: discard
    store.init(name, numelShape(byName[name][].shape))

proc adamFileStep*(store: AdamDiskStore, name: string, paramLen: int): int =
  ## Doc duy nhat header Adam de kiem tra step, khong load m/v.
  if not store.has(name, paramLen): return -1
  let path = adamPath(store, name)
  var f: File
  if not open(f, path, fmRead): return -1
  defer: f.close()
  var magic: uint64
  var version: uint32
  var n: int64
  var step: int64
  if f.readBuffer(addr magic, sizeof(magic)) != sizeof(magic): return -1
  if f.readBuffer(addr version, sizeof(version)) != sizeof(version): return -1
  if f.readBuffer(addr n, sizeof(n)) != sizeof(n): return -1
  if f.readBuffer(addr step, sizeof(step)) != sizeof(step): return -1
  if magic != 0x425942594144414D'u64 or version != 1'u32 or n != int64(paramLen): return -1
  result = int(step)

proc adamStoreAllAtStep*(store: AdamDiskStore, model: var NimformerModelJamba,
                         names: seq[string], expectedStep: int): bool =
  ## Kiem tra TAT CA Adam state cua block moi co cung step voi checkpoint.
  if expectedStep <= 0: return false
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(model): byName[name] = t
  for name in names:
    if name notin byName: return false
    let plen = numelShape(byName[name][].shape)
    if adamFileStep(store, name, plen) != expectedStep:
      return false
  result = true

proc adamStoreAllAtStepOrNext*(store: AdamDiskStore, model: var NimformerModelJamba,
                               names: seq[string], step: int): bool =
  ## Nhu adamStoreAllAtStep nhung chap nhan step hoac step+1 (step train dang do:
  ## block da update trong step nay o step+1, block chua toi o step) - giong LoRA.
  if step < 0 or names.len == 0: return false
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(model): byName[name] = t
  for name in names:
    if name notin byName: return false
    let s = adamFileStep(store, name, numelShape(byName[name][].shape))
    if s != step and s != step + 1:
      return false
  result = true

proc syncLoraAdamFromStore*(store: AdamDiskStore, state: var LoraTrainState, lora: JambaLora) =
  ## Merge Adam LoRA tu SSD. KHONG de file step=0 cu de len RAM step cao hon
  ## (bug cu: sync de step=0 -> checkpoint fail du train da xong).
  for name, ad in lora.adapters:
    let ka = loraAdamKey(name, "A")
    let kb = loraAdamKey(name, "B")
    if store.has(ka, ad.A.data.len):
      let disk = store.load(ka, ad.A.data.len)
      let ramStep = if name in state.adamA: state.adamA[name].step else: -1
      if disk.step >= ramStep:
        state.adamA[name] = disk
    if store.has(kb, ad.B.data.len):
      let disk = store.load(kb, ad.B.data.len)
      let ramStep = if name in state.adamB: state.adamB[name].step else: -1
      if disk.step >= ramStep:
        state.adamB[name] = disk

proc cloneFloat32Data(src: seq[float32]): seq[float32] =
  ## Dedicated allocation, even under ARC/ORC or shared seq assignment.
  result = newSeq[float32](src.len)
  for j in 0 ..< src.len:
    result[j] = src[j]

proc cloneAdamSnapshot(st: ApfAdamState): ApfAdamState =
  result.step = st.step
  result.m = cloneFloat32Data(st.m)
  result.v = cloneFloat32Data(st.v)

proc adamNeverUpdated(st: ApfAdamState): bool =
  ## m = v = 0 tuyet doi: chua tung nhan gradient nao (moi init / reset).
  for x in st.m:
    if x != 0'f32: return false
  for x in st.v:
    if x != 0'f32: return false
  true

proc forceLoraAdamToStep*(state: var LoraTrainState, lora: JambaLora, step: int): int {.discardable.} =
  ## Ep moi adapter A/B ve dung `step` (giu m/v neu size dung). Dung khi
  ## step train da xong sach nhung mot so file Adam van step=0.
  ## SUA: KHONG ep adapter co Adam B m=v=0 (chua tung co gradient - vd block
  ## chua duoc backward trong step dang do): ep step cho no lam lech
  ## bias-correction o step sau va che giau viec block do chua duoc train.
  ## (Dung B chu khong dung A: o step 1 grad A = B^T.g = 0 hop le vi B=0.)
  ## Tra ve so adapter bi bo qua.
  for name, ad in lora.adapters:
    var sa = if name in state.adamA: state.adamA[name] else: newApfAdamState(ad.A.data.len)
    var sb = if name in state.adamB: state.adamB[name] else: newApfAdamState(ad.B.data.len)
    if sa.m.len != ad.A.data.len or sa.v.len != ad.A.data.len:
      sa = newApfAdamState(ad.A.data.len)
    if sb.m.len != ad.B.data.len or sb.v.len != ad.B.data.len:
      sb = newApfAdamState(ad.B.data.len)
    if step > 0 and adamNeverUpdated(sb):
      inc result
    else:
      sa.step = step
      sb.step = step
    state.adamA[name] = sa
    state.adamB[name] = sb
  if result > 0:
    stderr.writeLine &"[checkpoint] CANH BAO: {result}/{lora.adapters.len} adapter LoRA chua tung co " &
      &"gradient (Adam B m=v=0) - KHONG ep Adam step -> {step} cho chung."

proc forceAdamStoreToStep*(store: AdamDiskStore, model: var NimformerModelJamba,
                           names: seq[string], step: int) =
  ## Ep step file Adam cua block moi (attn/fuseGate full-train) ve dung `step`.
  ## Giu m/v hien co; chi ghi lai header step de dong bo voi training step
  ## (tranh checkpoint-final fail sau resume RESET new-block Adam).
  if names.len == 0: return
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(model): byName[name] = t
  var n = 0
  for name in names:
    if name notin byName: continue
    let plen = numelShape(byName[name][].shape)
    if not store.has(name, plen):
      store.init(name, plen)
    var st = store.load(name, plen)
    st.step = step
    store.save(name, st)
    clearAdamState(st)
    inc n
  stderr.writeLine &"[checkpoint] FORCE: ep {n}/{names.len} new-block Adam step -> {step}"

proc loraAdamStateAllAtStep*(state: LoraTrainState, lora: JambaLora, expectedStep: int): bool =
  ## Kiem tra LoRA A/B da cung step truoc khi commit checkpoint.
  if expectedStep <= 0: return false
  result = true
  var nBad = 0
  for name, _ in lora.adapters:
    if name notin state.adamA or name notin state.adamB:
      if nBad < 8:
        stderr.writeLine &"[checkpoint] LoRA Adam THIEU state: {name}"
      inc nBad
      result = false
      continue
    let sa = state.adamA[name].step
    let sb = state.adamB[name].step
    if sa != expectedStep or sb != expectedStep:
      if nBad < 8:
        stderr.writeLine &"[checkpoint] LoRA Adam lech step: {name} A.step={sa} B.step={sb} expected={expectedStep}"
      inc nBad
      result = false
  if nBad > 8:
    stderr.writeLine &"[checkpoint] ... va {nBad - 8} adapter LoRA lech/thieu them"

proc datasetSignature*(specs: seq[DatasetSpec]): string =
  ## Signature nhe de chan resume: path + kich thuoc + mtime.
  ## Kich thuoc don thuan bo sot truong hop sua corpus nhung giu nguyen so byte.
  ## Khong hash toan bo corpus vi se doc lai hang tram MB/GB truoc moi run.
  result = ""
  for spec in specs:
    if fileExists(spec.path):
      result.add(spec.path & ":" & $getFileSize(spec.path) & ":" &
                 $getLastModificationTime(spec.path).toUnix & ";")
    else:
      result.add(spec.path & ":MISSING;")

proc datasetSignatureSizeOnly(specs: seq[DatasetSpec]): string =
  ## Compatibility check for commits written before mtime was included.
  for spec in specs:
    if fileExists(spec.path):
      result.add(spec.path & ":" & $getFileSize(spec.path) & ";")
    else:
      result.add(spec.path & ":MISSING;")

proc attnStoreAllValid*(store: AttnWeightDiskStore, model: var NimformerModelJamba,
                        names: seq[string]): bool =
  ## FP32 Attention sidecar phai CO DU va dung kich thuoc cho moi tham so.
  ## Thieu 1 file = khong tin ca bo, tranh tron FP32 cu voi Q4 checkpoint moi.
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(model): byName[name] = t
  if names.len == 0: return true
  for name in names:
    if name notin byName: return false
    if not store.hasW(name, numelShape(byName[name][].shape)):
      return false
  result = true

proc writeResumeCommit*(checkpointPath: string, step, seqLen, batchSize: int,
                        pattern: string, datasetSig: string, tokenizerDir: string) =
  ## Commit marker DUOC ghi SAU checkpoint + LoRA Adam state.
  ## Luu them dataset signature + tokenizer path de khong resume vao corpus/tokenizer khac.
  let tmp = checkpointPath & ".resume.commit.tmp"
  # SUA: escapeJson() DA tu boc dau ngoac kep; truoc day boc them 1 lop nua
  # -> "pattern":""RRR..."" la JSON hong, readResumeCommit luon tra false
  # (moi lan resume deu bi coi la khong co commit marker -> bo FP32 sidecar G
  # va reset Adam block G).
  let body = "{\"step\":" & $step &
             ",\"seqLen\":" & $seqLen &
             ",\"batchSize\":" & $batchSize &
             ",\"pattern\":" & escapeJson(pattern) &
             ",\"datasetSig\":" & escapeJson(datasetSig) &
             ",\"tokenizerDir\":" & escapeJson(tokenizerDir) & "}"
  writeFile(tmp, body)
  # POSIX rename replaces the old marker atomically. Never remove the old
  # commit first: a kill in that gap would orphan an otherwise valid checkpoint.
  if cRename(tmp.cstring, (checkpointPath & ".resume.commit").cstring) != 0:
    raise newException(IOError, "Khong the commit resume marker atomic: " & tmp)
  try: removeFile(tmp)
  except OSError: discard

proc readResumeCommit*(checkpointPath: string, stepOut, seqLenOut, batchOut: var int,
                       patternOut, datasetSigOut, tokenizerDirOut: var string): bool =
  let path = checkpointPath & ".resume.commit"
  if not fileExists(path): return false
  try:
    let raw = readFile(path)
    # Marker cu (truoc ban sua writeResumeCommit) co chuoi bi boc ngoac kep
    # 2 lan: ""...""; gia tri ben trong khong chua dau ngoac nen bo 1 lop la du.
    let j = try: parseJson(raw)
            except JsonParsingError: parseJson(raw.replace("\"\"", "\""))
    stepOut = j["step"].getInt().int
    seqLenOut = j["seqLen"].getInt().int
    batchOut = j["batchSize"].getInt().int
    patternOut = j["pattern"].getStr
    if j.hasKey("datasetSig"): datasetSigOut = j["datasetSig"].getStr else: datasetSigOut = ""
    if j.hasKey("tokenizerDir"): tokenizerDirOut = j["tokenizerDir"].getStr else: tokenizerDirOut = ""
    result = true
  except CatchableError:
    result = false

proc markResumeInProgress*(checkpointPath: string, nextStep: int) =
  writeFile(checkpointPath & ".resume.inprogress", $nextStep)

proc clearResumeInProgress*(checkpointPath: string) =
  try: removeFile(checkpointPath & ".resume.inprogress")
  except OSError: discard

proc applyLayerProgressToLora*(lo: var JambaLora, tensors: seq[checkpoint.LayerTensor]): int =
  ## Nap A/B LoRA tu <ckpt>.inprogress/ (ten: adapter::A / adapter::B).
  result = 0
  for t in tensors:
    let sep = t.name.rfind("::")
    if sep < 0: continue
    let adapterName = t.name[0 ..< sep]
    let which = t.name[sep+2 .. ^1]
    if adapterName notin lo.adapters: continue
    if which == "A":
      let need = numelShape(lo.adapters[adapterName].A.shape)
      if need > 0 and t.data.len != need: continue
      lo.adapters[adapterName].A.data = t.data
      if t.shape.len > 0: lo.adapters[adapterName].A.shape = t.shape
      inc result
    elif which == "B":
      let need = numelShape(lo.adapters[adapterName].B.shape)
      if need > 0 and t.data.len != need: continue
      lo.adapters[adapterName].B.data = t.data
      if t.shape.len > 0: lo.adapters[adapterName].B.shape = t.shape
      inc result

proc requireCompleteLayerSnapshot(lo: JambaLora, ckptPath: string, step, li: int)

proc restoreLayerProgressForStep*(lo: var JambaLora, ckptPath: string, step, nBlocks: int,
                                  updateMaxBlock = high(int)): int =
  ## Tra ve lastCompletedLayer (-1 = khong co). Nap LoRA block nBlocks-1 .. cLayer.
  result = -1
  var cStep, cLayer: int
  if not checkpoint.readProgressCursor(ckptPath, cStep, cLayer): return
  if cStep != step: return
  if cLayer < 0 or cLayer >= nBlocks: return
  var nTen = 0
  for li in countdown(nBlocks - 1, cLayer):
    # Fill deliberately only propagates dX through blocks > updateMaxBlock.
    if li > updateMaxBlock: continue
    requireCompleteLayerSnapshot(lo, ckptPath, step, li)
    let tensors = checkpoint.loadLayerProgress(ckptPath, step, li)
    nTen += applyLayerProgressToLora(lo, tensors)
  result = cLayer
  stderr.writeLine("[resume-layer] step=" & $step & " cursor layer=" & $cLayer &
    ": nap " & $nTen & " tensor LoRA (block " & $cLayer & ".." & $(nBlocks-1) &
    "); backward tiep tu block " & $(cLayer - 1) & " xuong.")



# ═══════════════════════════════════════════════════════════════
# 1 buoc train: LoRA cho block Mamba (cu, dong bang) + full Adam cho block
# Attention/Spectron moi cay vao.
# ═══════════════════════════════════════════════════════════════
proc isNewBlockParameter(m: NimformerModelJamba, name: string): bool =
  for i, blk in m.blocks:
    let prefix = &"block{i}."
    if not name.startsWith(prefix): continue
    case blk.kind
    of bkMambaRealPlusAttn:
      return name.startsWith(prefix & "attn.")
    of bkMambaRealGated:
      return name.startsWith(prefix & "attn.") or name.startsWith(prefix & "fuseGate.")
    of bkAttn, bkSpectron:
      return true
    else:
      return false
  false

proc clipGGradient(grad: var Tensor; name: string) =
  var normSq = 0.0
  for value in grad.data:
    if value.classify in {fcNan, fcInf, fcNegInf}:
      raise newException(CatchableError, "G block non-finite gradient in " & name)
    normSq += float64(value) * float64(value)
  let gradNorm = sqrt(normSq)
  if gradNorm > float64(CFG_G_GRAD_CLIP_NORM):
    let scale = float32(float64(CFG_G_GRAD_CLIP_NORM) / gradNorm)
    for i in 0 ..< grad.data.len: grad.data[i] *= scale

proc trainStepGraft(m: var NimformerModelJamba, lo: var JambaLora,
                     loState: var LoraTrainState,
                     adamStore: var AdamDiskStore,
                     newBlockLr: float32,
                     idsBatch, targetsBatch: seq[seq[int]], ctx: Backend): float32 =
  mergeIn(m, lo, ctx)
  let (loss, grads) = computeLossAndGrads(m, idsBatch, targetsBatch, ctx,
                                           trainLmHead = false, trainEmbed = false)

  # -- LoRA (block Mamba cu, dong bang base) --
  for name, ad in lo.adapters.mpairs:
    if name notin grads:
      raise newException(ValueError,
        "LoRA adapter '" & name & "' khong co gradient trong trainStepGraft. Dung train se khong duoc phep tiep tuc am tham.")
    let (dA, dB) = gradToAB(grads[name], ad, ctx)
    var stA = loState.adamA[name]
    var stB = loState.adamB[name]
    discard apfAdamStepFused(ctx, ad.A, dA, stA, loState.lr, requantizeEvery = int.high)
    discard apfAdamStepFused(ctx, ad.B, dB, stB, loState.lr, requantizeEvery = int.high)
    loState.adamA[name] = stA
    loState.adamB[name] = stB

  # -- block moi (Attention/Spectron) train FULL, khong LoRA --
  for (name, t) in paramList(m):
    if name in lo.adapters: continue                 # da xu ly o tren (LoRA)
    if not isNewBlockParameter(m, name): continue
    if name notin grads: continue
    let plen = numelShape(t[].shape)
    if not adamStore.has(name, plen):
      adamStore.init(name, plen)
    var st = adamStore.load(name, plen)
    discard apfAdamStepFused(ctx, t[], grads[name], st, newBlockLr, requantizeEvery = int.high)
    adamStore.save(name, st)
    clearAdamState(st)
    GC_fullCollect()

  restoreBase(m, lo)
  result = loss

# ═══════════════════════════════════════════════════════════════
# BAN STREAMING: dung khi lo.quantBase KHONG rong (model dung nguon
# quant-resident, xem loadCodestralQuantResidentFromNimq) - xu ly TUNG
# BLOCK mot (merge -> forward, sau đó merge lai -> backward -> trich
# LoRA/Adam -> xoa NGAY) thay vi merge/forward/backward CA MODEL cung luc
# roi giu 1 bang grads khong lo cho toi cuoi. Ly do can ban nay: voi 52
# block 'R' cua Codestral-7B, ban thuong (trainStepGraft) se giu DONG THOI:
#   - ~29GB neu dequant het weight fp32 (mergeIn merge CA LOAT)
#   - ~23GB bang grads day du (dInProjW/dOutProjW cho CA 52 block) truoc khi
#     rut xuong LoRA rank-thap
# vuot xa RAM 16GB. Ban streaming nay tai moi thoi diem CHI giu ~1 block
# (~450MB weight tam + vai tram MB cache Mamba2Real) - kiem chung dung bang
# so voi ban thuong tren tensor NHO (xem test_streaming_matches.nim di kem).
# ═══════════════════════════════════════════════════════════════
type Rusage {.importc: "struct rusage", header: "<sys/resource.h>", bycopy.} = object
  ru_maxrss {.importc: "ru_maxrss".}: clong   ## macOS: don vi BYTE (Linux la KB, nhung
                                               ## project nay chi target macOS/Metal nen OK)

proc c_getrusage(who: cint, usage: ptr Rusage): cint {.importc: "getrusage", header: "<sys/resource.h>".}

proc rssMB(): string =
  ## Doc peak-RSS THAT cua chinh tien trinh nay qua getrusage() - goi API C
  ## TRONG CUNG process, KHONG fork/spawn process con (khac ban truoc dung
  ## startProcess("/bin/ps",...) - cai do fork() ca tien trinh dang nam hang
  ## GB RAM MOI LAN log, tu no gay ap luc bo nho va lam nang them dung van
  ## de dang dieu tra - da bo). getrusage cuc re, an toan goi hang tram lan.
  var ru: Rusage
  if c_getrusage(0.cint, addr ru) == 0:
    result = $(int(ru.ru_maxrss) div (1024*1024)) & "MB"
  else:
    result = "?"

type
  DiskActivationStore = object
    ## Thay vi giu `xs: seq[Tensor]` (64 activation) SONG SUOT ca forward+
    ## backward trong RAM, ghi TUNG activation xuong 1 file scratch tren SSD
    ## ngay sau khi tinh xong (forward), CHI giu lai (offset, len, shape) -
    ## vai chuc byte/block - trong RAM. Luc backward can activation nao thi
    ## seek+doc DUNG doan do tu file, dung xong thi thoi (khong giu lai).
    ## Cham hon giu RAM (co I/O disk moi block) nhung doi lai KHONG bao gio
    ## OOM vi phan nay - dung y "chi chiem RAM toi 1 muc, du con lai xuong
    ## SSD" ban yeu cau, ap dung cho cau truc RAM-permanent-trong-1-step lon
    ## nhat (xs) thay vi toan bo model (weight van xu ly rieng qua
    ## quant-resident + mergeInOne/restoreBaseOne nhu cu).
    f: File
    path: string
    writeClosed: bool
    offsets: seq[tuple[pos, count: int64, shape: seq[int]]]

proc openDiskActivationStore(path: string): DiskActivationStore =
  createDir(getScratchDir())
  result.path = path
  if not open(result.f, result.path, fmWrite):
    raise newException(IOError, "Khong mo duoc file scratch SSD: " & result.path)
  result.writeClosed = false
  result.offsets = @[]
  # 8-byte slot at position 0 holds the int64 offset of the header trailer.
  # Written as 0 now; overwritten by finalizeActivationStore after all payload.
  var zero: int64 = 0
  if result.f.writeBuffer(addr zero, sizeof(zero)) != sizeof(zero):
    raise newException(IOError, "openDiskActivationStore: ghi slot header that bai")

proc openResumeActivationStore(path: string): DiskActivationStore =
  if not open(result.f, path, fmRead):
    raise newException(IOError, "Khong mo duoc activation resume: " & path)
  # Read the 8-byte header-offset pointer at position 0.
  var hdrOffset: int64
  if result.f.readBuffer(addr hdrOffset, sizeof(hdrOffset)) != sizeof(hdrOffset) or
     hdrOffset <= 0:
    result.f.close()
    raise newException(IOError, "Activation resume: header offset khong hop le")
  result.f.setFilePos(hdrOffset)
  var magic = newString(ResumeActivationMagic.len)
  if result.f.readBuffer(addr magic[0], magic.len) != magic.len or magic != ResumeActivationMagic:
    result.f.close()
    raise newException(IOError, "Activation resume khong co header hop le")
  var n: int32
  discard result.f.readBuffer(addr n, sizeof(n))
  for _ in 0 ..< int(n):
    var pos, count: int64
    var rank: int32
    discard result.f.readBuffer(addr pos, sizeof(pos))
    discard result.f.readBuffer(addr count, sizeof(count))
    discard result.f.readBuffer(addr rank, sizeof(rank))
    var shape = newSeq[int](int(rank))
    for i in 0 ..< shape.len:
      var dim: int32
      discard result.f.readBuffer(addr dim, sizeof(dim))
      shape[i] = int(dim)
    result.offsets.add((pos, count, shape))
  result.path = path
  result.f.close()
  result.writeClosed = true

proc openForwardResumeActivationStore(path: string, nAct: int, shape: seq[int]): DiskActivationStore =
  ## Mo lai file activation cua 1 forward dang do (chua co header): activation
  ## i nam o 8 + i*bytes (moi activation cung shape [B,T,H]). Giu nAct activation
  ## dau, ghi tiep tu sau do.
  var count = 1
  for d in shape: count *= d
  let bytes = int64(count) * int64(sizeof(float32))
  let need = 8'i64 + int64(nAct) * bytes
  if shape.len == 0 or count <= 0 or nAct <= 0 or getFileSize(path) < need:
    raise newException(IOError,
      &"Forward resume: file activation {path} thieu du lieu ({getFileSize(path)} < {need} byte)")
  if not open(result.f, path, fmReadWriteExisting):
    raise newException(IOError, "Khong mo duoc activation forward resume: " & path)
  result.path = path
  result.writeClosed = false
  result.offsets = @[]
  for i in 0 ..< nAct:
    result.offsets.add((8'i64 + int64(i) * bytes, int64(count), shape))
  result.f.setFilePos(need)

proc writeActivationHeader(s: var DiskActivationStore) =
  # Append header after all payload, then write its offset into the 8-byte slot.
  let hdrOffset = s.f.getFilePos()
  template mustWrite(buf: pointer, n: int) =
    if s.f.writeBuffer(buf, n) != n:
      raise newException(IOError, "writeActivationHeader: ghi that bai (dia day?)")
  mustWrite(ResumeActivationMagic.cstring, ResumeActivationMagic.len)
  var n = int32(s.offsets.len)
  mustWrite(addr n, sizeof(n))
  for item in s.offsets:
    mustWrite(addr item.pos, sizeof(item.pos))
    mustWrite(addr item.count, sizeof(item.count))
    var rank = int32(item.shape.len)
    mustWrite(addr rank, sizeof(rank))
    for dim in item.shape:
      var d = int32(dim)
      mustWrite(addr d, sizeof(d))
  # Patch the pointer slot at position 0.
  s.f.setFilePos(0)
  var ho = hdrOffset
  if s.f.writeBuffer(addr ho, sizeof(ho)) != sizeof(ho):
    raise newException(IOError, "writeActivationHeader: patch offset that bai")

proc pushActivation(s: var DiskActivationStore, t: Tensor) =
  ## Ghi TRUC TIEP xuong disk, KHONG giu ban sao trong RAM sau khi ghi xong
  ## (t van con o scope ngoai goi ham nay trong 1 khoanh khac ngan de
  ## forward tiep, nhung ham nay khong tao them tham chieu nao moi ca).
  let pos = s.f.getFilePos()
  if t.data.len > 0:
    let n = t.data.len * sizeof(float32)
    if s.f.writeBuffer(unsafeAddr t.data[0], n) != n:
      raise newException(IOError, "pushActivation: ghi that bai (dia day?)")
  s.offsets.add((pos, int64(t.data.len), t.shape))

proc finalizeActivationStore(s: var DiskActivationStore) =
  s.f.flushFile()
  writeActivationHeader(s)  # appends header trailer, patches offset slot at pos 0
  s.f.flushFile()

proc resumeStatePath(step: int): string =
  getCheckpointPath() & ".inprogress" / ("step-" & $step & ".state")

proc preparedResumeStatePath(step: int): string =
  resumeStatePath(step) & ".prepared"

proc discardPreparedResumeState(step: int) =
  try: removeFile(preparedResumeStatePath(step))
  except OSError: discard

proc saveResumeState(step, nextLayer: int, phase: string, dX: Tensor,
                     loss: float32, prepared = false) =
  let path = if prepared: preparedResumeStatePath(step) else: resumeStatePath(step)
  createDir(getCheckpointPath() & ".inprogress")
  let tmp = path & ".tmp"
  var f = open(tmp, fmWrite)
  template mustWrite(buf: pointer, n: int) =
    if f.writeBuffer(buf, n) != n:
      # f.close() do finally ben duoi lo (truoc day close 2 lan).
      raise newException(IOError, "saveResumeState: ghi that bai (dia day?)")
  try:
    var magic = ResumeStateMagic
    mustWrite(addr magic[0], magic.len)
    var phaseLen = int32(phase.len)
    mustWrite(addr phaseLen, sizeof(phaseLen))
    if phase.len > 0: mustWrite(addr phase[0], phase.len)
    var s = int32(step)
    var n = int32(nextLayer)
    var l = loss
    mustWrite(addr s, sizeof(s))
    mustWrite(addr n, sizeof(n))
    mustWrite(addr l, sizeof(l))
    var rank = int32(dX.shape.len)
    mustWrite(addr rank, sizeof(rank))
    for dim in dX.shape:
      var d = int32(dim)
      mustWrite(addr d, sizeof(d))
    if dX.data.len > 0:
      mustWrite(unsafeAddr dX.data[0], dX.data.len * sizeof(float32))
  finally:
    f.close()
  if cRename(tmp.cstring, path.cstring) != 0:
    if fileExists(path): removeFile(path)
    moveFile(tmp, path)

proc loadResumeState(step: int, nextLayer: var int, phase: var string,
                     dX: var Tensor, lossOut: var float32, prepared = false): bool =
  let path = if prepared: preparedResumeStatePath(step) else: resumeStatePath(step)
  if not fileExists(path): return false
  var f = open(path, fmRead)
  try:
    var magic = newString(ResumeStateMagic.len)
    if f.readBuffer(addr magic[0], magic.len) != magic.len or magic != ResumeStateMagic:
      return false
    var phaseLen: int32
    if f.readBuffer(addr phaseLen, sizeof(phaseLen)) != sizeof(phaseLen) or phaseLen < 0:
      return false
    phase = newString(int(phaseLen))
    if phaseLen > 0 and f.readBuffer(addr phase[0], int(phaseLen)) != int(phaseLen):
      return false
    var savedStep, savedNext, rank: int32
    var savedLoss: float32
    if f.readBuffer(addr savedStep, sizeof(savedStep)) != sizeof(savedStep) or
       f.readBuffer(addr savedNext, sizeof(savedNext)) != sizeof(savedNext) or
       f.readBuffer(addr savedLoss, sizeof(savedLoss)) != sizeof(savedLoss) or
       f.readBuffer(addr rank, sizeof(rank)) != sizeof(rank) or savedStep != int32(step) or
       rank < 0:
      return false
    var shape = newSeq[int](int(rank))
    var count = 1
    for i in 0 ..< shape.len:
      var dim: int32
      if f.readBuffer(addr dim, sizeof(dim)) != sizeof(dim) or dim < 0: return false
      shape[i] = int(dim)
      count *= shape[i]
    if phase == "forward":
      # Trang thai forward chi luu shape activation (data nam trong file activation).
      dX = Tensor(shape: shape)
      nextLayer = int(savedNext)
      lossOut = savedLoss
      return true
    dX = newTensor(shape)
    if count > 0 and f.readBuffer(addr dX.data[0], count * sizeof(float32)) != count * sizeof(float32):
      return false
    nextLayer = int(savedNext)
    lossOut = savedLoss
    result = true
  finally:
    f.close()

proc promotePreparedResumeState(step: int) =
  let src = preparedResumeStatePath(step)
  if not fileExists(src):
    raise newException(IOError, "[resume] Missing prepared per-layer resume state: " & src)
  if cRename(src.cstring, resumeStatePath(step).cstring) != 0:
    raise newException(IOError, "[resume] Cannot atomically promote " & src)

proc requireCompleteLayerSnapshot(lo: JambaLora, ckptPath: string, step, li: int) =
  ## Verify every A/B pair before trusting a progress cursor. A G-only block
  ## legitimately has zero LoRA tensors but still has a nonempty layer manifest.
  if not checkpoint.layerProgressReadable(ckptPath, step, li):
    raise newException(IOError,
      &"[resume] Missing/corrupt layer snapshot for step {step}, block {li}. " &
      "DO NOT delete .inprogress or restart from the previous checkpoint.")
  let tensors = checkpoint.loadLayerProgress(ckptPath, step, li)
  var expected = 0
  for name, ad in lo.adapters:
    if not name.startsWith("block" & $li & "."): continue
    for which in ["A", "B"]:
      inc expected
      let key = name & "::" & which
      var found = false
      for t in tensors:
        let orig = if which == "A": ad.A else: ad.B
        if t.name == key and t.shape == orig.shape and t.data.len == orig.data.len:
          found = true
          break
      if not found:
        raise newException(IOError, &"[resume] Incomplete LoRA snapshot: {key} step={step} layer={li}")
  if tensors.len != expected:
    raise newException(IOError,
      &"[resume] Unexpected layer tensor count at block {li}: {tensors.len}, expected {expected}")

proc recoverPreparedResumeState(step, nBlocks: int, lo: JambaLora) =
  ## 2-phase layer commit: prepared dX is written BEFORE optimizer updates;
  ## layer weights+cursor are committed after updates; then prepared is promoted.
  ## If killed while optimizer writes were incomplete, refuse to replay the block
  ## (this requires an external rollback; silently replaying doubles Adam steps).
  if not fileExists(preparedResumeStatePath(step)): return
  var nextLayer = -2
  var phase = ""
  var dx: Tensor
  var loss = 0'f32
  if not loadResumeState(step, nextLayer, phase, dx, loss, prepared = true):
    raise newException(IOError, "[resume] Corrupt prepared state; refusing unsafe optimizer replay")
  let finished = if phase == "backward_done": 0 else: nextLayer + 1
  if finished < 0 or finished >= nBlocks or
     (phase != "backward" and phase != "backward_done") or
     (phase == "backward_done" and nextLayer != -1):
    raise newException(IOError, "[resume] Invalid prepared cursor; refusing replay")
  var curStep, curLayer: int
  let hasCursor = checkpoint.readProgressCursor(getCheckpointPath(), curStep, curLayer)
  if hasCursor and (curStep != step or (curLayer != finished and curLayer != finished + 1)):
    raise newException(IOError, "[resume] Prepared state and cursor disagree; refusing replay")
  # Layer files are the authoritative evidence that the post-update LoRA
  # weights were fully written. With none, Adam may already be partly changed.
  requireCompleteLayerSnapshot(lo, getCheckpointPath(), step, finished)
  if not hasCursor or curLayer != finished:
    checkpoint.commitLayerProgressCursor(getCheckpointPath(), step, finished)
  promotePreparedResumeState(step)
  stderr.writeLine &"[resume] Recovered staged block {finished}: LoRA snapshot + cursor + dX consistent."

proc activationResumePath(step: int): string =
  getScratchDir() / (".xs_resume_step_" & $step & ".bin")

proc removeActivationResume*(step: int) =
  try: removeFile(activationResumePath(step))
  except OSError: discard
  try: removeFile(resumeStatePath(step))
  except OSError: discard
  discardPreparedResumeState(step)

proc layerProgressStepState*(step, cursorLayer, nBlocks: int,
                             backwardDone, canResume: var bool): string =
  ## Phan loai step dang do theo cursor layer-progress + resume state:
  ## backwardDone = backward da toi block 0 (moi duoc commit nhu step xong);
  ## canResume = con dX + activation de trainStepGraftStreaming backward tiep.
  ## Tra ve mo ta ngan de log.
  var rsNext = -2
  var rsPhase = ""
  var rsDX: Tensor
  var rsLoss = 0'f32
  let hasRs = loadResumeState(step, rsNext, rsPhase, rsDX, rsLoss)
  # cursor == 0: layer-progress block 0 da ghi = backward xong. KHONG doi phase
  # "backward_done": o block 0 progress duoc ghi TRUOC resume state, crash giua
  # 2 lan ghi de lai phase="backward" next=0 -> replay block 0 se update 2 lan.
  if cursorLayer == 0 and hasRs and rsPhase == "backward" and rsNext == 0:
    # Legacy code recorded rollback of layer 0 using the same cursor as a
    # committed layer 0. Cannot know whether layer 0 was really updated.
    raise newException(IOError,
      "[resume] Ambiguous legacy layer-0 cursor (backward next=0). " &
      "Refusing to commit an incompletely trained step; KEEP .inprogress.")
  backwardDone = cursorLayer == 0
  canResume = hasRs and rsPhase == "backward" and rsNext >= 0 and rsNext < nBlocks and
    fileExists(activationResumePath(step))
  result = &"cursor layer={cursorLayer}, resume phase='{rsPhase}' next={rsNext}, " &
    &"{nBlocks - cursorLayer}/{nBlocks} block da backward"

proc readActivation(s: DiskActivationStore, i: int): Tensor =
  ## Mo RIENG 1 file-handle DOC (khong dung chung handle ghi o tren, tranh
  ## dao lon con tro ghi) - doc dung 1 doan, dong lai NGAY sau khi doc.
  let (pos, count, shape) = s.offsets[i]
  var rf: File
  if not open(rf, s.path, fmRead):
    raise newException(IOError, "Khong doc duoc file scratch SSD: " & s.path)
  rf.setFilePos(pos)
  result = newTensor(shape)
  if count > 0:
    let nRead = rf.readBuffer(addr result.data[0], int(count) * sizeof(float32))
    if nRead != int(count) * sizeof(float32):
      rf.close()
      raise newException(IOError, "Doc thieu du lieu tu file scratch SSD (dia day? file bi cat?)")
  rf.close()

proc closeAndDelete(s: var DiskActivationStore, keep = false) =
  ## "Don ky cang": dong file neu con mo, sau do XOA NGAY scratch.
  ## Defer o trainStepGraftStreaming goi ham nay neu co exception giua step,
  ## nen scratch khong bi bo lai tren SSD.
  if not s.writeClosed:
    try: s.f.close()
    except: discard
    s.writeClosed = true
  if not keep:
    try: removeFile(s.path)
    except: discard

proc dlog(msg: string) =
  ## Log CHAN DOAN, flush ngay lap tuc - de neu process bi kill (OOM) thi
  ## dong log CUOI CUNG con lai tren terminal la dau moi chinh xac nhat de
  ## biet dang chet o buoc nao. rss=... doc qua getrusage() (khong fork).
  stderr.writeLine("[dbg] rss=" & rssMB() & " " & msg)
  stderr.flushFile()

# ── Single-line progress (refresh \r) + throughput ─────────────────────
type
  ProgressMeter* = object
    label*: string
    totalBytes*: int64
    doneBytes*: int64
    totalItems*: int
    doneItems*: int
    t0*: float
    tLast*: float
    lastLineLen*: int

proc progressStart*(label: string; totalItems = 0; totalBytes: int64 = 0): ProgressMeter =
  result.label = label
  result.totalItems = totalItems
  result.totalBytes = totalBytes
  result.t0 = epochTime()
  result.tLast = result.t0

proc progressTick*(p: var ProgressMeter; itemName = ""; itemBytes: int64 = 0;
                   forceNewline = false) =
  ## Cap nhat 1 dong (\r). Hien: [i/N] name  doneMB/totalMB  rate MB/s  ETA.
  ## forceNewline=true: ket thuc meter, xuong dong (goi o cuoi vong lap).
  if itemBytes > 0:
    p.doneBytes += itemBytes
  if itemName.len > 0 or itemBytes > 0:
    inc p.doneItems
  let now = epochTime()
  let dt = max(now - p.t0, 1e-6)
  let rate = float(p.doneBytes) / dt / (1024.0 * 1024.0)  # MB/s
  let doneMB = float(p.doneBytes) / (1024.0 * 1024.0)
  let totalMB = float(p.totalBytes) / (1024.0 * 1024.0)
  var etaStr = "--"
  if p.totalBytes > 0 and p.doneBytes > 0 and rate > 1e-3:
    let remain = float(p.totalBytes - p.doneBytes) / (1024.0 * 1024.0) / rate
    if remain < 60: etaStr = formatFloat(remain, ffDecimal, 0) & "s"
    else: etaStr = formatFloat(remain / 60.0, ffDecimal, 1) & "m"
  elif p.totalItems > 0 and p.doneItems > 0:
    let per = dt / float(p.doneItems)
    let remain = per * float(p.totalItems - p.doneItems)
    if remain < 60: etaStr = formatFloat(remain, ffDecimal, 0) & "s"
    else: etaStr = formatFloat(remain / 60.0, ffDecimal, 1) & "m"
  let shortName =
    if itemName.len > 48: "..." & itemName[^45 .. ^1]
    else: itemName
  var line = "[prog] " & p.label
  if p.totalItems > 0:
    line &= " [" & $p.doneItems & "/" & $p.totalItems & "]"
  if shortName.len > 0:
    line &= " " & shortName
  if p.totalBytes > 0:
    line &= "  " & formatFloat(doneMB, ffDecimal, 1) & "/" &
            formatFloat(totalMB, ffDecimal, 1) & "MB"
  else:
    line &= "  " & formatFloat(doneMB, ffDecimal, 1) & "MB"
  line &= "  " & formatFloat(rate, ffDecimal, 1) & "MB/s  ETA " & etaStr
  line &= "  rss=" & rssMB()
  # Pad with spaces to erase previous longer line, then \r
  if line.len < p.lastLineLen:
    line &= repeat(' ', p.lastLineLen - line.len)
  p.lastLineLen = line.len
  if forceNewline:
    stderr.write("\r" & line & "\n")
  else:
    stderr.write("\r" & line)
  stderr.flushFile()
  p.tLast = now

proc progressEnd*(p: var ProgressMeter) =
  ## Dong ket thuc meter (xuong dong + tong hop).
  progressTick(p, forceNewline = true)

var gLastStepForwardRan*: bool = false

# ── Do thoi gian train ──────────────────────────────────────────────────────
# Moi block in 1 dong "[time] ..." (rat re). BYBY_TRAIN_PROFILE=1: them bang chi
# tiet theo stage (lin.gemm, blk.inProj, ...) tu prefill_profile cho tung block.
let gTrainProfile = getEnv("BYBY_TRAIN_PROFILE", "0") != "0"
# Backward LoRA-aware (mac dinh BAT): kernel tinh thang dA/dB, khong tinh dW day du
# [outF,inF] roi gradToAB. BYBY_LORA_AWARE_BWD=0 -> duong cu (de doi chieu).
let gLoraAwareBwd = getEnv("BYBY_LORA_AWARE_BWD", "1") != "0"

proc dsec(a, b: MonoTime): float = float((b - a).inMilliseconds) / 1000.0

template trainProfBegin() =
  if gTrainProfile:
    gPfEnabled = true
    pfReset()
    gPfActive = true

template trainProfEnd(label: string; t0: MonoTime) =
  if gTrainProfile:
    gPfActive = false
    stderr.writeLine "[train-profile] " & label
    pfReport(float((getMonoTime() - t0).inMilliseconds))

const
  AuditGoodAgree = 0.90   # >= : dau B da luu khop gradient step 0 that -> block dung
  AuditBadAgree = 0.70    # < AuditGoodAgree van tinh lai; > AuditBadAgree (khong phai ~50%) thi canh bao
var gAuditStopBlock* = -1 # audit: block dung dau tien (backward dung tai day), -1 = khong co

proc graftInitForStep0(name: string; cur: seq[float32]): seq[float32] =
  ## Gia tri tham so block G TRUOC step 0 (luc graft, xem jamba_hybrid.newJambaBlock):
  ## attn.outProj = 0, fuseGate.weight = 0, fuseGate.bias = 4. attn.qkvProj co grad
  ## step 0 = 0 (outProj = 0) nen khong doi -> giu nguyen. Gia tri hien tai chi duoc
  ## cach init toi da 1 buoc Adam (LR G * 1/warmup ~ 2.5e-7); xa hon = khong phai
  ## checkpoint step 1 cua graft nay -> raise, khong doan.
  if ".attn.qkvProj." in name:
    return cur
  var init: float32
  if name.endsWith(".attn.outProj.weight") or name.endsWith(".attn.outProj.bias") or
     name.endsWith(".fuseGate.weight"):
    init = 0'f32
  elif name.endsWith(".fuseGate.bias"):
    init = 4'f32
  else:
    raise newException(ValueError, &"[audit] Khong biet gia tri graft truoc step 0 cua '{name}'")
  for v in cur:
    if not (abs(v - init) <= 1e-5'f32):
      raise newException(ValueError,
        &"[audit] '{name}' lech gia tri graft {init} toi {abs(v - init):.3e} (> 1 buoc Adam step 0) - " &
        "khong tai tao duoc dieu kien step 0. Dung.")
  result = newSeq[float32](cur.len)
  for k in 0 ..< result.len: result[k] = init

proc trainStepGraftStreaming*(m: var NimformerModelJamba, lo: var JambaLora,
                               loState: var LoraTrainState,
                               adamStore: var AdamDiskStore,
                               attnStore: var AttnWeightDiskStore,
                               newBlockLr: float32,
                               idsBatch, targetsBatch: seq[seq[int]], ctx: Backend;
                               gLrScale: float32 = 1.0;
                               stepIndex: int = -1;
                               updateMaxBlock: int = high(int);
                               auditRefB: Table[string, seq[float32]] =
                                 initTable[string, seq[float32]]()): float32 =
  ## gLrScale (mac dinh 1.0) nhan vao LR cua MOI tham so block G (ca sidecar
  ## lan fuseGate) - caller (vong train that) truyen warmup factor de tranh
  ## buoc update lon ngay dau. Cac tham so block 'A'/'S' khac van dung
  ## newBlockLr nguyen ven.
  ## updateMaxBlock: block i > updateMaxBlock chi backward de lay dX, KHONG
  ## Adam/rescue/layer-progress (dung boi fillUntrainedLoraBlocks de chay
  ## noi phan backward con thieu cua 1 step dang do).
  ## auditRefB (khong rong = che do kiem step 0, auditStep0LoraBlocks): moi B dang = 0
  ## va block G tinh bang gia tri graft (dieu kien step 0). Tung block tu tren xuong:
  ## dau B da luu (auditRefB) khop -gradient -> block dung, DUNG backward tai day
  ## (gAuditStopBlock); khong khop -> reset Adam roi update lai nhu step 0.
  let auditMode = auditRefB.len > 0
  doAssert isGpuCtx(ctx),
    "trainStepGraftStreaming requires a GPU backend; refusing CPU execution"
  # Allow automatic CPU fallback when GPU runs out of memory, with GPU retry
  # on the next operation. This prevents OOM crashes on memory-constrained GPUs.
  setForbidCpuFallback(false)
  gLastStepForwardRan = false
  dlog "trainStepGraftStreaming: bat dau"
  var resumePhase = ""
  var resumeNextLayer = m.blocks.len - 1
  var resumedDX: Tensor
  var resumedLoss = 0'f32
  var canResumeBackward = false
  var resumeForwardFrom = -1   # >= 0: activation 0..resumeForwardFrom da tren SSD
  # One canonical path per training step so the write side (forward) and the
  # read side (resume) always point at the same file. Legacy/no-stepIndex
  # callers keep the old address-tagged path.
  let actPath = if stepIndex >= 0: activationResumePath(stepIndex)
                else: CFG_SCRATCH_DIR / (".xs_scratch_" & $cast[int](addr m) & ".bin")
  if stepIndex >= 0:
    recoverPreparedResumeState(stepIndex, m.blocks.len, lo)
    let hasState = loadResumeState(stepIndex, resumeNextLayer, resumePhase, resumedDX, resumedLoss)
    var curStep, curLayer: int
    let hasCursor = checkpoint.readProgressCursor(getCheckpointPath(), curStep, curLayer) and
      curStep == stepIndex
    if hasCursor and curLayer == 0:
      # Backward cua step nay DA toi block 0 (layer-progress block 0 da ghi): step
      # da xong, phai commit. Chay/replay lai se update block 0 (hoac ca step) 2 lan.
      raise newException(IOError,
        &"[resume] step {stepIndex} da backward xong toi block 0 nhung chua commit - KHONG chay lai. " &
        "Chay lai voi BYBY_FORCE_CHECKPOINT=1 (mac dinh) de commit step nay.")
    if hasState and resumePhase == "backward" and resumeNextLayer >= 0 and
       resumeNextLayer < m.blocks.len and fileExists(actPath):
      canResumeBackward = true
      # Load the LoRA A/B deltas already applied to the upper blocks before the
      # crash. Their Adam m/v were persisted per layer in adamStore, so
      # restoreLoraAdamStates() at startup already recovered them.
      let restoredLayer = restoreLayerProgressForStep(lo, getCheckpointPath(), stepIndex,
        m.blocks.len, updateMaxBlock)
      # Block > updateMaxBlock chi truyen dX, khong ghi layer-progress: chua toi
      # block updateMaxBlock thi chua co cursor nao (-1).
      let expectedLayer = if resumeNextLayer + 1 > updateMaxBlock: -1 else: resumeNextLayer + 1
      # restoredLayer == resumeNextLayer: block vua bi RESCUE rollback (cursor = i,
      # nextLayer = i) -> replay chinh block do, hop le.
      if resumeNextLayer < m.blocks.len - 1 and restoredLayer != expectedLayer and
         restoredLayer != resumeNextLayer:
        raise newException(IOError,
          &"[resume] Per-layer progress layer {restoredLayer} khong khop voi resumeNextLayer {resumeNextLayer} " &
          &"(ky vong layer {expectedLayer}). GIU .inprogress; kiem tra snapshot truoc khi can thiep.")
      dlog &"resume state: backward layer {resumeNextLayer}, khong chay lai forward"
    elif hasCursor:
      # Da co block update trong step nay nhung khong resume backward duoc: chay
      # lai ca step se update cac block >= cursor LAN 2. Dung.
      raise newException(IOError,
        &"[resume] step {stepIndex}: co layer-progress (cursor layer {curLayer}) nhung khong co " &
        &"dX/activation de backward tiep (phase='{resumePhase}'). KHONG chay lai step (se update " &
        "2 lan). Xoa <ckpt>.inprogress de train lai step.")
    elif hasState and resumePhase == "forward" and resumedDX.shape.len > 0 and
         resumeNextLayer >= 0 and resumeNextLayer <= m.blocks.len and fileExists(actPath):
      resumeForwardFrom = resumeNextLayer
      dlog &"resume state: forward da xong {resumeNextLayer}/{m.blocks.len} block, forward tiep"
  # ten adapter theo tung block (1 block 'R' co 2: inProj.weight, outProj.weight)
  var namesByBlock = initTable[int, seq[string]]()
  for name in lo.adapters.keys:
    if name.startsWith("block"):
      let rest = name["block".len .. ^1]
      let dotIdx = rest.find('.')
      if dotIdx > 0:
        try:
          let bi = parseInt(rest[0 ..< dotIdx])
          namesByBlock.mgetOrPut(bi, @[]).add(name)
        except ValueError: discard
  gTrainStepFullyDone = true
  dlog "trainStepGraftStreaming: namesByBlock xong, bat dau paramList + bootstrap (KHONG phai embed lookup)"

  # Dung 1 LAN cho CA ham (thay vi goi paramList(m) lap lai NHIEU LAN ben
  # trong vong lap block 'A' o duoi - paramList(m) dung lai TOAN BO danh
  # sach tham so model (~250+ entry) MOI LAN goi, goi lap lai cho tung
  # gradient cua tung block 'A' (Attention co ca Q/K/V/O = 4 tham so/block)
  # x 13 block 'A' x 64 lan (ca forward+backward) la rat nhieu lan dung lai
  # danh sach khong can thiet - vua cham vua giu rac tam thoi khong duoc
  # don (khong co GC_fullCollect nao chay cho nhanh block 'A' ca) - chinh
  # la nguyen nhan RSS nhay vot dung tai cac block 'A' (19, 24) thay vi
  # block 'R' (da flat, chung to co che merge/restore cho 'R' dung).
  dlog "paramList(m) bat dau..."
  var paramByName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): paramByName[name] = t
  dlog "paramList xong (" & $paramByName.len & " ten), paramQuantFlags..."
  let quantFlags = paramQuantFlags(m)
  dlog "paramLinears..."
  let linearsByName = paramLinears(m)
  var fullPrecWasQuantized = initTable[string, bool]()

  # MOI - doi xung voi namesByBlock (LoRA) nhung cho tham so FULL-PRECISION
  # (block 'A', khong LoRA-hoa) - dung de nap/xoa qua attnStore dung luc can
  # cho TUNG block, thay vi de nguyen ~3.5GB suot ca buoc train (bug goc gay
  # RSS nhay vot da ghi trong comment o tren).
  var fullPrecNamesByBlock = initTable[int, seq[string]]()
  for name in paramByName.keys:
    if name in lo.adapters or not isNewBlockParameter(m, name): continue
    if name.startsWith("block"):
      let rest = name["block".len .. ^1]
      let dotIdx = rest.find('.')
      if dotIdx > 0:
        try:
          let bi = parseInt(rest[0 ..< dotIdx])
          fullPrecNamesByBlock.mgetOrPut(bi, @[]).add(name)
          if name in quantFlags and name notin fullPrecWasQuantized:
            fullPrecWasQuantized[name] = quantFlags[name][]
        except ValueError: discard
  var nFullPrec = 0
  var totalBootBytes: int64 = 0
  for bi, names in fullPrecNamesByBlock:
    nFullPrec += names.len
    for name in names:
      if name in paramByName:
        totalBootBytes += int64(numelShape(paramByName[name][].shape)) * 4
  dlog "fullPrecNamesByBlock xong: " & $nFullPrec & " tensor ~" &
    $(totalBootBytes div (1024*1024)) & "MB, bootstrap attnStore (1 dong refresh + MB/s)"

  # Bootstrap: lan DAU TIEN goi ham nay trong ca qua trinh train, p[].data
  # dang giu gia tri THAT (tu checkpoint resume hoac init moi) - luu 1 lan
  # xuong SSD. CAC buoc sau, p[].data se luon RONG luc vao day (da bi xoa o
  # cuoi buoc truoc) nen attnStore.hasW se dung, khong ghi de nham gia tri
  # rong len file dung (dung numelShape(p[].shape) - ON DINH, KHAC voi
  # .data.len co the =0 - lam "paramLen" tin cay).
  # LAN DAU co the mat NHIEU PHUT neu volume cham (exFAT/network/USB) -
  # moi tensor G (qkv/out) co the hang tram MB FP32.
  # Progress: 1 dong \r refresh + MB/s + ETA de do hieu suat o dia.
  var bootProg = progressStart("bootstrap→SSD", totalItems = nFullPrec,
                               totalBytes = totalBootBytes)
  for bi, names in fullPrecNamesByBlock:
    for name in names:
      let p = paramByName[name]
      let plen = numelShape(p[].shape)
      let nbytes = int64(plen) * 4
      if not attnStore.hasW(name, plen):
        if p[].data.len == plen:
          attnStore.saveW(name, p[].data)
        elif name in lo.quantBase:
          # Resume fallback: NIMBIN4 giu newBlock weight dang quantized va data=[]
          # trong Linear. Dequant 1 tensor tai day de tao FP32 working copy tren SSD;
          # KHONG dequant ca model.
          let dq = dequantizeTensor(lo.quantBase[name])
          if dq.len != plen:
            raise newException(IOError,
              &"bootstrap attnStore: '{name}' quantized size {dq.len} != shape numel {plen}")
          attnStore.saveW(name, dq)
        else:
          raise newException(IOError,
            &"bootstrap attnStore: '{name}' rong ca RAM lan quantBase (khong co weight de resume)")
        progressTick(bootProg, itemName = name & " WRITE", itemBytes = nbytes)
      else:
        progressTick(bootProg, itemName = name & " skip", itemBytes = nbytes)
  progressEnd(bootProg)
  dlog "bootstrap attnStore xong, bat dau embed.lookupBatch that"

  # ---- FORWARD: tung block, merge truoc (dequant qua LUT neu quant-resident), forward, restore/xoa ngay ----
  var x: Tensor
  var xstore: DiskActivationStore
  # Forward resume: trang thai "forward" next=n nghia la activation 0..n da
  # flush xuong SSD (block 0..n-1 forward xong). Ghi SAU khi flush activation.
  template saveForwardState(nDone: int) =
    if stepIndex >= 0:
      xstore.f.flushFile()
      saveResumeState(stepIndex, nDone, "forward", Tensor(shape: x.shape), 0'f32)
  if canResumeBackward:
    xstore = openResumeActivationStore(actPath)
  elif resumeForwardFrom >= 0:
    xstore = openForwardResumeActivationStore(actPath, resumeForwardFrom + 1, resumedDX.shape)
    x = xstore.readActivation(resumeForwardFrom)
    stderr.writeLine &"[resume] step {stepIndex}: forward tiep tu block {resumeForwardFrom} " &
      &"({resumeForwardFrom}/{m.blocks.len} block da forward)"
  else:
    x = m.embed.lookupBatch(idsBatch, ctx)
    dlog "embed lookup xong, bat dau forward qua " & $m.blocks.len & " block"
    xstore = openDiskActivationStore(actPath)
    xstore.pushActivation(x)
    saveForwardState(0)
  # Keep the scratch file whenever we are inside a checkpoint-able step: a
  # crash during backward must leave the activations on disk for resume. The
  # file is removed explicitly after the step commits.
  defer: xstore.closeAndDelete(keep = stepIndex >= 0)
  if canResumeBackward or resumeForwardFrom >= 0:
    # Activation da luu PHAI thuoc dung batch nay: activation 0 = embed(ids) (copy
    # hang embedding, tat dinh) - lech = batch/seqLen/dataset da doi giua 2 lan chay.
    let x0Now = m.embed.lookupBatch(idsBatch, ctx)
    let x0Saved = xstore.readActivation(0)
    if x0Saved.shape != x0Now.shape or x0Saved.data != x0Now.data:
      if canResumeBackward:
        raise newException(IOError,
          &"[resume] step {stepIndex}: activation/dX da luu thuoc batch KHAC batch hien tai " &
          &"(shape luu={x0Saved.shape}, hien tai={x0Now.shape}) - KHONG backward tiep (gradient se sai). " &
          "Kiem tra BYBY_SEQ_LEN/dataset; xoa <ckpt>.inprogress de train lai step.")
      stderr.writeLine &"[resume] step {stepIndex}: activation forward da luu thuoc batch khac " &
        "-> bo, forward lai tu dau (forward chua update weight nao)."
      xstore.f.close()
      xstore.writeClosed = true
      resumeForwardFrom = -1
      x = x0Now
      xstore = openDiskActivationStore(actPath)
      xstore.pushActivation(x)
      saveForwardState(0)
  if not canResumeBackward:
    gLastStepForwardRan = true
    for i in max(0, resumeForwardFrom) ..< m.blocks.len:
      let tf0 = getMonoTime()
      trainProfBegin()
      if i in namesByBlock:
        dlog &"block {i}: mergeInOne x{namesByBlock[i].len} tham so"
        for name in namesByBlock[i]: mergeInOne(m, lo, ctx, name)
      if i in fullPrecNamesByBlock:
        for name in fullPrecNamesByBlock[i]:
          let p = paramByName[name]
          if name in fullPrecWasQuantized: quantFlags[name][] = false
          if name in linearsByName:
            linearsByName[name][].residentReady = false
            linearsByName[name][].residentFp32Ready = false
          p[].data = attnStore.loadW(name, numelShape(p[].shape))
          if auditMode: p[].data = graftInitForStep0(name, p[].data)
      let tfMerge = getMonoTime()
      x = forwardBlock(m.blocks[i], x, ctx, blockIndex = i)
      let tfFwd = getMonoTime()
      # Bug4: phát hiện NaN/Inf ngay sau mỗi block thay vì chờ đến loss
      for v in x.data:
        if v.classify in {fcNan, fcInf, fcNegInf}:
          raise newException(CatchableError,
            &"Non-finite activation after forward block {i} (kind={m.blocks[i].kind})")
      if i in namesByBlock:
        for name in namesByBlock[i]: restoreBaseOne(m, lo, name)
      if i in fullPrecNamesByBlock:
        for name in fullPrecNamesByBlock[i]:
          paramByName[name][].data = @[]
          if name in fullPrecWasQuantized:
            quantFlags[name][] = fullPrecWasQuantized[name]
      xstore.pushActivation(x)
      saveForwardState(i + 1)
      let tfEnd = getMonoTime()
      stderr.writeLine &"[time] fwd block {i}: merge+load={dsec(tf0, tfMerge):.1f}s " &
        &"forward={dsec(tfMerge, tfFwd):.1f}s spill={dsec(tfFwd, tfEnd):.1f}s total={dsec(tf0, tfEnd):.1f}s"
      trainProfEnd(&"fwd block {i}", tf0)
      GC_fullCollect()
      dlog &"block {i}: forward xong"
      if gStopRequested and stepIndex >= 0:
        gTrainStepFullyDone = false
        stderr.writeLine &"[finetune_lora] Dung an toan sau forward block {i} " &
          "(activation + trang thai forward da tren SSD). Chay lai de forward tiep."
        return 0'f32
    finalizeActivationStore(xstore)
    xstore.f.close()
    xstore.writeClosed = true
    # The final activation is now durable; the backward phase will persist its
    # incoming gradient after the frozen output stack is differentiated.
  dlog "forward toan bo block xong, tinh outProj + loss"
  # lm_head is frozen and remains quant-resident. Forward uses its quantized
  # path directly; backwardInputOnly dequantizes only if its resident data is
  # absent, and computes dX without allocating the unused [vocab, hidden] dW.
  # SUA (preset rms-fin): norm_f (RMSNorm, khop base HF) TRUOC lm_head.
  # Truoc day goi thang outProj.forward(x) len hidden CHUA norm - sai scale
  # tich luy 64 block, mot trong cac nguon sinh logit loan.
  var dX: Tensor
  var loss = 0'f32
  # Loss > ln(vocab) = te hon doan deu ngau nhien: forward/lm_head hong (vd bug
  # lm_head random-init, loss ~35). DUNG truoc khi update/ghi bat ky thu gi - ap
  # ca cho loss/dX doc lai khi resume (step dang do do binary cu tao ra).
  let maxSaneLoss = ln(float32(m.vocab))
  template checkSaneLoss() =
    if loss > maxSaneLoss:
      raise newException(CatchableError,
        &"Loss {loss:.4f} > ln(vocab)={maxSaneLoss:.2f} (te hon doan ngau nhien) - forward/lm_head " &
        "hong. DUNG, KHONG update weight nao. Kiem tra lm_head (bindQuantBaseToLinears) / norm; " &
        "neu la step dang do tu binary cu thi xoa <ckpt>.inprogress.")
  if canResumeBackward:
    dX = resumedDX
    loss = resumedLoss
    checkSaneLoss()
    dlog &"resume: loss={loss}, bat dau backward tu layer {resumeNextLayer}"
  else:
    let xNormed = m.rmsF.forward(x, ctx)
    let logits = m.outProj.forward(xNormed, ctx)
    # SUA: dung ban CO MASK dinh nghia ngay trong file nay. test_nimformer.
    # softmaxCrossEntropy giu NGUYEN BAN GOC, khong dung toi.
    let (l, dLogits) = softmaxCrossEntropyMasked(logits, targetsBatch, ctx)
    loss = l
    if loss.classify in {fcNan, fcInf, fcNegInf}:
      raise newException(CatchableError, "Non-finite training loss; optimizer step cancelled")
    checkSaneLoss()
    dlog &"loss={loss}, bat dau backward"
    # outProj/embed KHONG train trong che do codestral (frozen, giong tinh
    # than adamStore khong dong den) - bo qua dW/dB cua outProj, KHONG goi
    # embed.backward (khong can, khong con lop nao truoc embed de lan dX tiep).
    # rmsF.gamma cung thuoc base -> DONG BANG, chi can dX di tiep xuong duoi.
    let dOutProjInput = m.outProj.backwardInputOnly(xNormed, dLogits, ctx)
    let dNormF = m.rmsF.backward(x, dOutProjInput, ctx)
    dX = dNormF.dX
    if stepIndex >= 0:
      # Persist the gradient entering the top block BEFORE any block work, so
      # a crash inside the backward loop can resume at the exact layer with
      # the correct upstream gradient.
      saveResumeState(stepIndex, m.blocks.len - 1, "backward", dX, loss)
      if getEnv("BYBY_TEST_CRASH_BEFORE_BACKWARD", "0") != "0":
        raise newException(CatchableError, "SIMULATED_CRASH_BEFORE_BACKWARD")
  dlog "bat dau backward"

  # ---- BACKWARD: tung block NGUOC - merge lai, backward, trich LoRA/Adam
  # NGAY vao 1 bang grads RIENG (chi 1 block), roi xoa ----
  let firstBackwardLayer = if canResumeBackward: resumeNextLayer else: m.blocks.len - 1
  # Theo doi block da backward trong step nay (ke ca block xong o lan chay truoc,
  # da duoc xac nhan qua layer-progress khi resume) - cuoi step phai du TAT CA.
  var blockDone = newSeq[bool](m.blocks.len)
  for j in firstBackwardLayer + 1 ..< m.blocks.len: blockDone[j] = true
  var i = firstBackwardLayer
  while i >= 0:
    let tb0 = getMonoTime()
    trainProfBegin()
    if i in namesByBlock:
      dlog &"backward block {i}: mergeInOne"
      for name in namesByBlock[i]: mergeInOne(m, lo, ctx, name)
    if i in fullPrecNamesByBlock:
      # Nap lai tu SSD TRUOC backwardBlock - can gia tri weight THAT de tinh
      # dX/gradient dung (da bi xoa khoi RAM o cuoi forward cua CHINH block
      # nay, xem vong lap FORWARD o tren).
      for name in fullPrecNamesByBlock[i]:
        let p = paramByName[name]
        if name in fullPrecWasQuantized: quantFlags[name][] = false
        if name in linearsByName:
          linearsByName[name][].residentReady = false
          linearsByName[name][].residentFp32Ready = false
        p[].data = attnStore.loadW(name, numelShape(p[].shape))
        if auditMode: p[].data = graftInitForStep0(name, p[].data)

    let tbMerge = getMonoTime()
    let xi = xstore.readActivation(i)   # doc DUNG 1 activation tu SSD, dung xong (backwardBlock) la het scope
    var dXBeforeBlock = dX  # gradient entering i, needed unchanged on retry
    dXBeforeBlock.data = cloneFloat32Data(dX.data)
    var blockGrads = initTable[string, Tensor]()   # rieng cho block nay, KHONG tich luy
    gLoraGradRequest.clear()
    if gLoraAwareBwd:
      for name in namesByBlock.getOrDefault(i, @[]):
        let ad = lo.adapters[name]
        gLoraGradRequest[name] = LoraGradSpec(A: ad.A.data, B: ad.B.data, r: ad.r, scale: ad.scale)
    try:
      dX = backwardBlock(m.blocks[i], xi, dX, ctx, &"block{i}", blockGrads,
                 blockIndex = i)
    finally:
      gLoraGradRequest.clear()
    dlog &"backward block {i}: backwardBlock xong"
    let tbBwd = getMonoTime()
    if i > updateMaxBlock:
      # Chi truyen dX xuong; weight/Adam cua block nay giu nguyen.
      for name in namesByBlock.getOrDefault(i, @[]): restoreBaseOne(m, lo, name)
      for name in fullPrecNamesByBlock.getOrDefault(i, @[]):
        paramByName[name][].data = @[]
        if name in fullPrecWasQuantized: quantFlags[name][] = fullPrecWasQuantized[name]
      if stepIndex >= 0:
        # Resume tiep tu block i-1 voi dX nay (khong co layer-progress cho block nay).
        saveResumeState(stepIndex, i - 1, "backward", dX, loss)
      blockDone[i] = true   # co y chi truyen dX (block > updateMaxBlock)
      stderr.writeLine &"[time] bwd block {i} (chi truyen dX): merge+load={dsec(tb0, tbMerge):.1f}s " &
        &"backwardBlock={dsec(tbMerge, tbBwd):.1f}s total={dsec(tb0, getMonoTime()):.1f}s"
      trainProfEnd(&"bwd block {i} (chi truyen dX)", tb0)
      GC_fullCollect()
      if gStopRequested and stepIndex >= 0:
        gTrainStepFullyDone = false
        stderr.writeLine &"[finetune_lora] Dung an toan sau backward block {i} (chi truyen dX). " &
          "Chay lai de tiep tuc."
        break
      dec i
      continue

    # CAM thieu: moi tham so LoRA va moi tham so block moi (G attn/fuseGate) cua
    # block nay PHAI co gradient - kiem TRUOC khi update bat ky thu gi cua block.
    # (Truoc day tham so block moi thieu grad bi bo qua im lang.)
    # dA/dB tinh TRUOC cho ca block va kiem: huu han; dB khong duoc toan 0 (dB = 0 =
    # block khong nhan duoc gradient -> khong hoc, dung loai loi "layer trong").
    var abGrads = initTable[string, tuple[dA, dB: Tensor]]()
    for name in namesByBlock.getOrDefault(i, @[]):
      var dA, dB: Tensor
      if (name & "::loraA") in blockGrads and (name & "::loraB") in blockGrads:
        # Kernel LoRA-aware da tinh thang dA/dB (khong co dW day du).
        dA = blockGrads[name & "::loraA"]
        dB = blockGrads[name & "::loraB"]
      elif name in blockGrads:
        (dA, dB) = gradToAB(blockGrads[name], lo.adapters[name], ctx)
      else:
        raise newException(ValueError,
          &"LoRA adapter '{name}' khong co gradient trong backward block {i}. " &
           "Dung train se khong duoc phep tiep tuc am tham.")
      let ad0 = lo.adapters[name]
      if dA.shape != ad0.A.shape or dB.shape != ad0.B.shape:
        raise newException(ValueError,
          &"LoRA '{name}' block {i}: grad shape dA={dA.shape} dB={dB.shape} khac A={ad0.A.shape} B={ad0.B.shape}")
      for v in dA.data:
        if v.classify in {fcNan, fcInf, fcNegInf}:
          raise newException(ValueError, &"LoRA '{name}' block {i}: grad A co NaN/Inf - dung, khong update.")
      var dBNonZero = false
      for v in dB.data:
        if v.classify in {fcNan, fcInf, fcNegInf}:
          raise newException(ValueError, &"LoRA '{name}' block {i}: grad B co NaN/Inf - dung, khong update.")
        if v != 0'f32: dBNonZero = true
      if not dBNonZero:
        raise newException(ValueError,
          &"LoRA '{name}' block {i}: grad B TOAN 0 - block khong nhan gradient (dX xuong block = 0?). " &
           "Dung, khong update (se thanh layer trong).")
      abGrads[name] = (dA, dB)
    for name in fullPrecNamesByBlock.getOrDefault(i, @[]):
      if name notin blockGrads:
        raise newException(ValueError,
          &"Tham so block moi '{name}' khong co gradient trong backward block {i} - " &
           "se KHONG duoc update. Dung train.")
      for v in blockGrads[name].data:
        if v.classify in {fcNan, fcInf, fcNegInf}:
          raise newException(ValueError, &"Tham so block moi '{name}' block {i}: grad co NaN/Inf - dung.")

    if auditMode:
      # Adam buoc 1 tu B = 0: B = -lr * g/(|g|+eps) -> dau B da luu phai NGUOC dau
      # gradient step 0 that. So tung phan tu (bo phan tu = 0), lay ty le thap nhat.
      var agree = 1.0
      for name in namesByBlock.getOrDefault(i, @[]):
        if name notin auditRefB:
          raise newException(ValueError, &"[audit] Thieu B da luu cua '{name}'")
        let refB = auditRefB[name]
        let g = abGrads[name].dB.data
        if refB.len != g.len:
          raise newException(ValueError, &"[audit] '{name}': B da luu {refB.len} phan tu, grad {g.len}")
        var nCmp, nSame = 0
        for k in 0 ..< g.len:
          if refB[k] != 0'f32 and g[k] != 0'f32:
            inc nCmp
            if (refB[k] < 0'f32) == (g[k] > 0'f32): inc nSame
        if nCmp == 0:
          raise newException(ValueError, &"[audit] '{name}' block {i}: khong co phan tu nao de so dau")
        agree = min(agree, float(nSame) / float(nCmp))
      stderr.writeLine &"[audit] block {i}: dau B da luu khop gradient step 0 that {agree * 100:.2f}%"
      if agree >= AuditGoodAgree:
        # Block dung: block <= i den tu cung lan tinh dung (lap/train tu tren xuong lien
        # mach) -> giu nguyen, DUNG backward. Khong ghi state: resume se kiem lai block nay.
        for name in namesByBlock.getOrDefault(i, @[]): restoreBaseOne(m, lo, name)
        for name in fullPrecNamesByBlock.getOrDefault(i, @[]):
          paramByName[name][].data = @[]
          if name in fullPrecWasQuantized: quantFlags[name][] = fullPrecWasQuantized[name]
        for j in 0 .. i: blockDone[j] = true
        gAuditStopBlock = i
        stderr.writeLine &"[audit] block {i} DUNG -> giu nguyen block 0..{i}, dung backward."
        trainProfEnd(&"bwd block {i} (audit dung)", tb0)
        GC_fullCollect()
        break
      if agree > AuditBadAgree:
        # Update step 0 DUNG cua block la duy nhat (-lr*sign(g) tren dung batch step 0):
        # khop < nguong = khong phai update do -> van tinh lai (khong raise roi ket mai).
        stderr.writeLine &"[audit] CANH BAO block {i}: khop {agree * 100:.2f}% (khong phai ~50% cua " &
          &"gradient rac, cung khong >= {AuditGoodAgree * 100:.0f}%) - van tinh lai theo gradient that."
      # Block sai: tinh lai update step 0 tu dau (B dang = 0, G dang = gia tri graft).
      for name in namesByBlock.getOrDefault(i, @[]):
        loState.adamA[name] = newApfAdamState(lo.adapters[name].A.data.len)
        loState.adamB[name] = newApfAdamState(lo.adapters[name].B.data.len)
      for name in fullPrecNamesByBlock.getOrDefault(i, @[]):
        adamStore.init(name, numelShape(paramByName[name][].shape))
      stderr.writeLine &"[audit] block {i} SAI -> tinh lai update step 0 (Adam ve 0)."

    # Prepare next dX BEFORE any optimizer write. If the process dies before the
    # layer's .bin/.idx files are complete, startup must stop (not update twice).
    if stepIndex >= 0:
      let preparedPhase = if i == 0: "backward_done" else: "backward"
      saveResumeState(stepIndex, i - 1, preparedPhase, dX, loss, prepared = true)
      if getEnv("BYBY_TEST_CRASH_AFTER_PREPARED_LAYER", "") == $i:
        raise newException(IOError, "SIMULATED_CRASH_AFTER_PREPARED_LAYER_" & $i)
    let tbGrad = getMonoTime()
    # Snapshot pre-update weights for G (gated) and R (pure Mamba2Real) so a
    # failed post-update safety / layer-probe check can be rolled back in-place
    # instead of aborting the whole run. One block at a time -> peak RAM grows
    # by one block's worth, not by all layers.
    # CRITICAL: PHẢI snapshot TRƯỚC apfAdamStepFused (cả LoRA lẫn full-prec),
    # nếu không rescue sẽ rollback nhầm về weight ĐÃ BỊ UPDATE làm hỏng model!
    var gPreUpdate = initTable[string, seq[float32]]()
    var loraPreUpdateA = initTable[string, seq[float32]]()
    var loraPreUpdateB = initTable[string, seq[float32]]()
    var loraPreAdamA = initTable[string, ApfAdamState]()
    var loraPreAdamB = initTable[string, ApfAdamState]()
    let needsRescue = m.blocks[i].kind in {bkMambaRealGated, bkMambaReal, bkMambaRealPlusAttn}
    var rescueTriggered = false  # rollback triggers a same-block retry, never a commit
    let blockLrScale =
      if needsRescue:
        gPerBlockLrScale.getOrDefault(i, 1.0'f32)
      else:
        1.0'f32

    if needsRescue:
      for name in namesByBlock.getOrDefault(i, @[]):
        if name in lo.adapters:
          let ad = lo.adapters[name]
          # Nim seq assignment may share storage. Adam GPU updates m/v in place.
          # A rollback MUST own independent buffers, not aliases to live tensors.
          loraPreUpdateA[name] = cloneFloat32Data(ad.A.data)
          loraPreUpdateB[name] = cloneFloat32Data(ad.B.data)
          loraPreAdamA[name] = cloneAdamSnapshot(loState.adamA[name])
          loraPreAdamB[name] = cloneAdamSnapshot(loState.adamB[name])
      for name in fullPrecNamesByBlock.getOrDefault(i, @[]):
        if name in paramByName:
          let p = paramByName[name]
          let plen = numelShape(p[].shape)
          if p[].data.len == plen:
            var copy = newSeq[float32](plen)
            for k in 0 ..< plen: copy[k] = p[].data[k]
            gPreUpdate[name] = copy

    var layerTensors: seq[checkpoint.LayerTensor] = @[]
    if i in namesByBlock:
      for name in namesByBlock[i]:
        if name notin abGrads:
          raise newException(ValueError,
            &"LoRA adapter '{name}' khong co gradient trong backward block {i}. " &
             "Dung train se khong duoc phep tiep tuc am tham.")
        var ad = lo.adapters[name]
        let (dA, dB) = abGrads[name]   # da tinh + kiem o tren
        var stA = loState.adamA[name]
        var stB = loState.adamB[name]
        let loraLr = loState.lr * blockLrScale
        discard apfAdamStepFused(ctx, ad.A, dA, stA, loraLr, requantizeEvery = int.high)
        discard apfAdamStepFused(ctx, ad.B, dB, stB, loraLr, requantizeEvery = int.high)
        loState.adamA[name] = stA
        loState.adamB[name] = stB
        lo.adapters[name] = ad
        if stepIndex >= 0:
          # Persist LoRA Adam m/v per layer too: without this, a crash mid-step
          # would leave upper layers already advanced while the rest are still
          # at oldStep, and restoreLoraAdamStates would then have no way to
          # recover a consistent per-layer step count.
          adamStore.save(loraAdamKey(name, "A"), stA)
          adamStore.save(loraAdamKey(name, "B"), stB)
          layerTensors.add(checkpoint.LayerTensor(name: name & "::A",
            shape: ad.A.shape, data: ad.A.data))
          layerTensors.add(checkpoint.LayerTensor(name: name & "::B",
            shape: ad.B.shape, data: ad.B.data))
      for name in namesByBlock[i]: restoreBaseOne(m, lo, name)

    # DUNG paramByName da dung san o dau ham, KHONG goi paramList(m) lai.
    # Adam m/v khong nam trong RAM qua cac step: doc 1 state tu SSD, update GPU,
    # ghi lai SSD va giai phong ngay.
    # SUA (bug that, gay IndexDefect "container is empty"): PHAI bo qua
    # nhung tham so DA thuoc lo.adapters (block 'R', xu ly qua LoRA o vong
    # lap tren) - restoreBaseOne() vua chay xong cho cac ten do, lam
    # p[].data VE RONG (@[]) vi day la tham so quant-resident, khong con du
    # lieu that trong RAM giua 2 lan merge. Thieu dieu kien nay khien vong
    # lap duoi co goi apfAdamStepGpu tren p[].data RONG trong khi grad van
    # day du kich thuoc -> index vao mang rong -> crash dung o block 'R'
    # dau tien gap trong backward (dung nhu log thuc te: block 63).
    for name, rawGrad in blockGrads:
      var grad = rawGrad
      if name in lo.adapters: continue
      if not isNewBlockParameter(m, name): continue
      if name notin paramByName: continue
      if m.blocks[i].kind == bkMambaRealGated:
        clipGGradient(grad, name)
      let p = paramByName[name]
      # SUA: dung numelShape(p[].shape) thay vi p[].data.len lam paramLen -
      # .data.len ON DINH ngay LUC NAY (vua nap lai o tren) nhung se VE 0
      # sau khi discard ben duoi; dung shape moi luon dung du p[].data dang
      # rong hay day, giu adamStore.has/init/load nhat quan qua cac buoc.
      let plen = numelShape(p[].shape)
      if not adamStore.has(name, plen):
        adamStore.init(name, plen)
      var st = adamStore.load(name, plen)
      let paramLr =
        if m.blocks[i].kind == bkMambaRealGated:
          (if name.startsWith(&"block{i}.fuseGate."): CFG_G_GATE_LR else: CFG_G_SIDECAR_LR) *
            gLrScale * blockLrScale
        else:
          newBlockLr
      discard apfAdamStepFused(ctx, p[], grad, st, paramLr, requantizeEvery = int.high)
      adamStore.save(name, st)
      clearAdamState(st)
      if name in fullPrecNamesByBlock.getOrDefault(i, @[]):
        # apfAdamStepGpu vua cap nhat p[].data TAI CHO (gia tri weight MOI) -
        # luu lai SSD ngay, roi xoa khoi RAM - KHONG de ~3.5GB nam lai qua
        # block tiep theo (day chinh la phan con thieu so voi block 'R').
        attnStore.saveW(name, p[].data)
        p[].data = @[]
        if name in fullPrecWasQuantized:
          quantFlags[name][] = fullPrecWasQuantized[name]
      GC_fullCollect()

    let tbUpd = getMonoTime()
    # Post-update safety check + rescue for G (gated) and R (pure Mamba) blocks.
    # Runs once per block AFTER the whole Adam loop has written params to SSD.
    # LAYER_PROBE (mặc định bật; tắt bằng BYBY_LAYER_PROBE=0) chạy short probe
    # để bắt activation drift / garbage-token mà pure RMS checks bỏ sót.
    if needsRescue:
      let rescueNames = fullPrecNamesByBlock.getOrDefault(i, @[])
      let rescueLoraNames = namesByBlock.getOrDefault(i, @[])
      for name in rescueNames:
        let p = paramByName[name]
        if name in fullPrecWasQuantized: quantFlags[name][] = false
        if name in linearsByName:
          linearsByName[name][].residentReady = false
          linearsByName[name][].residentFp32Ready = false
        p[].data = attnStore.loadW(name, numelShape(p[].shape))
      # Bug2: merge LoRA trước forward probe để check trên weight thực tế
      for name in rescueLoraNames:
        if name in lo.adapters: mergeInOne(m, lo, ctx, name)
      var gCheckError = ""
      # Output forward (weight SAU update) dung chung cho check resid/mixer va
      # LAYER_PROBE: 2 cho truoc day goi forwardBlock 2 lan cung input + cung
      # weight -> ket qua y het, ton them 1 lan forward ca block.
      var yCheck: Tensor
      try:
        if m.blocks[i].kind == bkMambaRealGated:
          let biasName = &"block{i}.fuseGate.bias"
          if biasName in paramByName:
            let biasTensor = paramByName[biasName][]
            var biasSum = 0.0
            var biasMin = Inf
            var biasMax = NegInf
            var biasSq = 0.0
            for v in biasTensor.data:
              if v.classify in {fcNan, fcInf, fcNegInf}:
                gCheckError = &"G block {i} gate bias non-finite after optimizer update"
                break
              let fv = float64(v)
              biasSum += fv
              biasSq += fv * fv
              if fv < biasMin: biasMin = fv
              if fv > biasMax: biasMax = fv
            if gCheckError.len == 0:
              let n = float64(max(1, biasTensor.data.len))
              let biasMean = biasSum / n
              let biasVar = max(0.0, biasSq / n - biasMean * biasMean)
              let biasStd = sqrt(biasVar)
              if biasMean < float64(CFG_G_GATE_BIAS_MIN) or biasMean > float64(CFG_G_GATE_BIAS_MAX):
                gCheckError = &"G block {i} gate bias mean {biasMean:.3f} outside " &
                  &"[{CFG_G_GATE_BIAS_MIN:.1f}, {CFG_G_GATE_BIAS_MAX:.1f}] after update"
              elif getEnv("BYBY_G_GATE_STD_RESCUE", "0") != "0" and
                   biasStd < float64(CFG_G_GATE_BIAS_MIN_STD):
                gCheckError = &"G block {i} gate bias collapsed (std={biasStd:.2e}, " &
                  &"min={biasMin:.3f} max={biasMax:.3f}); degenerate constant gate"
          # Bug1: kiểm tra output G block thay vì discard
          if gCheckError.len == 0:
            let yG = forwardBlock(m.blocks[i], xi, ctx, blockIndex = i)
            yCheck = yG
            var inSqG, outSqG = 0.0
            var inMaxG, outMaxG = 0.0
            for k in 0 ..< xi.data.len:
              let a = float64(xi.data[k])
              let b = float64(yG.data[k])
              if a.classify in {fcNan, fcInf, fcNegInf} or b.classify in {fcNan, fcInf, fcNegInf}:
                gCheckError = &"G block {i} produced non-finite after update"
                break
              inSqG += a * a
              outSqG += b * b
              inMaxG = max(inMaxG, abs(a))
              outMaxG = max(outMaxG, abs(b))
            if gCheckError.len == 0:
              let nG = float64(max(1, xi.data.len))
              let inRmsG = sqrt(inSqG / nG)
              let outRmsG = sqrt(outSqG / nG)
              let effInRmsG = if i == 0: max(inRmsG, 1.0'f64) else: max(inRmsG, 1e-6)
              let effInMaxG = if i == 0: max(inMaxG, 1.0'f64) else: max(inMaxG, 1e-6)
              let rmsRatioG = outRmsG / effInRmsG
              let maxRatioG = outMaxG / effInMaxG
              if rmsRatioG > float64(CFG_R_MAX_OUTPUT_RMS_RATIO) or
                 maxRatioG > float64(CFG_R_MAX_OUTPUT_MAX_RATIO):
                gCheckError = &"G block {i}: output RMS={rmsRatioG:.3f} " &
                  &"max={maxRatioG:.3f} exceeds limits after update"
        else:
          # R/P block: residual + MIXER. Mixer = y - x (đóng góp của SSM/Linear++
          # vào residual). Residual 16× gần như không bắt hub-hub; mixer 1.5× mới
          # siết nhánh Mamba (gần G sidecar 0.5×, không phải 4×).
          let y = forwardBlock(m.blocks[i], xi, ctx, blockIndex = i)
          yCheck = y
          var inSq, outSq, mixSq = 0.0
          var inMax, outMax, mixMax = 0.0
          for k in 0 ..< xi.data.len:
            let a = float64(xi.data[k])
            let b = float64(y.data[k])
            let d = b - a
            if a.classify in {fcNan, fcInf, fcNegInf} or b.classify in {fcNan, fcInf, fcNegInf}:
              gCheckError = &"R/P block {i} produced non-finite after update"
              break
            inSq += a * a
            outSq += b * b
            mixSq += d * d
            inMax = max(inMax, abs(a))
            outMax = max(outMax, abs(b))
            mixMax = max(mixMax, abs(d))
          if gCheckError.len == 0:
            let n = float64(max(1, xi.data.len))
            let inRms = sqrt(inSq / n)
            let outRms = sqrt(outSq / n)
            let mixRms = sqrt(mixSq / n)
            let effInRms = if i == 0: max(inRms, 1.0'f64) else: max(inRms, 1e-6)
            let effInMax = if i == 0: max(inMax, 1.0'f64) else: max(inMax, 1e-6)
            let rmsRatio = outRms / effInRms
            let maxRatio = outMax / effInMax
            let mixRmsRatio = mixRms / effInRms
            let mixMaxRatio = mixMax / max(effInMax, 1e-6)
            stderr.writeLine &"[finetune_lora] R/P block {i} resid={rmsRatio:.3f} " &
              &"mixer={mixRmsRatio:.3f} (limits resid {CFG_R_MAX_OUTPUT_RMS_RATIO:.1f}/" &
              &"mixer {CFG_R_MAX_MIXER_RMS_RATIO:.1f})"
            if rmsRatio > float64(CFG_R_MAX_OUTPUT_RMS_RATIO) or
               maxRatio > float64(CFG_R_MAX_OUTPUT_MAX_RATIO):
              gCheckError = &"R/P block {i}: residual out/input RMS={rmsRatio:.3f} " &
                &"max={maxRatio:.3f} exceeds limits " &
                &"{CFG_R_MAX_OUTPUT_RMS_RATIO:.1f}/{CFG_R_MAX_OUTPUT_MAX_RATIO:.1f}"
            elif mixRmsRatio > float64(CFG_R_MAX_MIXER_RMS_RATIO) or
                 mixMaxRatio > float64(CFG_R_MAX_MIXER_MAX_RATIO):
              gCheckError = &"R/P block {i}: mixer (y-x)/input RMS={mixRmsRatio:.3f} " &
                &"max={mixMaxRatio:.3f} exceeds limits " &
                &"{CFG_R_MAX_MIXER_RMS_RATIO:.1f}/{CFG_R_MAX_MIXER_MAX_RATIO:.1f} " &
                &"(SSM/LoRA inProj likely poisoning recurrent state)"

        # Bug3+Feature6: LAYER_PROBE thực sự — so sánh output sau update vs xstore (activation gốc).
        # Đặt NGAY TRONG try..except để khi driftRatio vượt ngưỡng thì set gCheckError,
        # kích hoạt cơ chế RESCUE (rollback weight/LoRA, zero Adam, giảm LR) thay vì crash cả run!
        if gCheckError.len == 0 and getEnv("BYBY_LAYER_PROBE", "1") != "0" and i + 1 < m.blocks.len:
          let okCount = gPerBlockProbeOk.getOrDefault(i, 0) + 1
          gPerBlockProbeOk[i] = okCount
          if CFG_LAYER_PROBE_EVERY <= 0 or (okCount mod max(1, CFG_LAYER_PROBE_EVERY)) == 0:
            let xOrig = xstore.readActivation(i + 1)
            let yProbe =
              if yCheck.data.len == xOrig.data.len and yCheck.shape == xOrig.shape: yCheck
              else: forwardBlock(m.blocks[i], xi, ctx, blockIndex = i)
            var driftSq, origSq = 0.0
            for k in 0 ..< xOrig.data.len:
              let orig = float64(xOrig.data[k])
              let newV = float64(yProbe.data[k])
              driftSq += (newV - orig) * (newV - orig)
              origSq += orig * orig
            let driftRatio = sqrt(driftSq / max(origSq, 1e-12))
            # R/Mamba: ngưỡng drift thấp hơn (0.08) — 0.25 quá rộng với SSM recurrent.
            let isMambaish = m.blocks[i].kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated, bkMamba}
            let driftThresh = if isMambaish: float64(CFG_R_LAYER_PROBE_DRIFT)
                              else: float64(CFG_LAYER_PROBE_GARBAGE_FRAC)
            stderr.writeLine &"[finetune_lora] LAYER_PROBE block {i} kind={m.blocks[i].kind} " &
              &"okCount={okCount}: drift={driftRatio:.4f} (threshold={driftThresh:.2f})"
            if driftRatio.classify in {fcNan, fcInf, fcNegInf} or
               driftRatio > driftThresh:
              gCheckError = &"LAYER_PROBE block {i}: activation drift {driftRatio:.4f} > " &
                &"{driftThresh:.2f}; update likely causes token garbage"
      except CatchableError as e:
        gCheckError = e.msg
      # Bug2: restore LoRA sau forward probe
      for name in rescueLoraNames:
        if name in lo.adapters: restoreBaseOne(m, lo, name)
      # Test-only hooks
      if getEnv("BYBY_G_TEST_FORCE_RESCUE", "0") != "0" and m.blocks[i].kind == bkMambaRealGated:
        gCheckError = &"G block {i}: forced test blowup (BYBY_G_TEST_FORCE_RESCUE=1)"
      if getEnv("BYBY_R_TEST_FORCE_RESCUE", "0") != "0" and m.blocks[i].kind in {bkMambaReal, bkMambaRealPlusAttn}:
        gCheckError = &"R block {i}: forced test blowup (BYBY_R_TEST_FORCE_RESCUE=1)"
      let maxAttempts = if m.blocks[i].kind == bkMambaRealGated: CFG_G_MAX_RESCUE_ATTEMPTS
                        else: CFG_R_MAX_RESCUE_ATTEMPTS
      let minScale = if m.blocks[i].kind == bkMambaRealGated: CFG_G_LR_MIN_SCALE
                     else: CFG_R_LR_MIN_SCALE
      let kindTag = if m.blocks[i].kind == bkMambaRealGated: "G"
                    elif m.blocks[i].kind == bkMambaRealPlusAttn: "P"
                    else: "R"
      if gCheckError.len == 0:
        # Update accepted. Recover LR scale slowly; clear fails.
        for name in rescueNames:
          paramByName[name][].data = @[]
          if name in fullPrecWasQuantized: quantFlags[name][] = fullPrecWasQuantized[name]
        gPerBlockRescueFails[i] = 0
        let cur = gPerBlockLrScale.getOrDefault(i, 1.0'f32)
        if cur < 1.0'f32:
          gPerBlockLrScale[i] = min(1.0'f32, cur * 1.1'f32)
      else:
        # RESCUE: roll layer back to pre-update weights, zero Adam, halve LR.
        let fails = gPerBlockRescueFails.getOrDefault(i, 0) + 1
        gPerBlockRescueFails[i] = fails
        let oldScale = gPerBlockLrScale.getOrDefault(i, 1.0'f32)
        let newScale = max(minScale, oldScale * 0.5'f32)
        gPerBlockLrScale[i] = newScale
        for name in rescueNames:
          let p = paramByName[name]
          if name in gPreUpdate:
            p[].data = gPreUpdate[name]
            attnStore.saveW(name, gPreUpdate[name])
            adamStore.init(name, numelShape(p[].shape))
          p[].data = @[]
          if name in fullPrecWasQuantized: quantFlags[name][] = fullPrecWasQuantized[name]
        for name in rescueLoraNames:
          if name in lo.adapters and name in loraPreUpdateA and name in loraPreUpdateB:
            var ad = lo.adapters[name]
            ad.A.data = loraPreUpdateA[name]
            ad.B.data = loraPreUpdateB[name]
            lo.adapters[name] = ad
            let stA = loraPreAdamA[name]
            let stB = loraPreAdamB[name]
            loState.adamA[name] = stA
            loState.adamB[name] = stB
            adamStore.save(loraAdamKey(name, "A"), stA)
            adamStore.save(loraAdamKey(name, "B"), stB)
        # Rolled-back block is NOT completed: no layer-progress record is written.
        if fails > maxAttempts:
          if stepIndex >= 0: discardPreparedResumeState(stepIndex)
          raise newException(CatchableError,
            &"{kindTag} block {i}: {fails} consecutive post-update safety failures " &
            &"(last: {gCheckError}); LR scale={newScale:.3f}. Rescue failed - " &
            "refusing to continue with a layer that cannot be trained stably.")
        stderr.writeLine &"[finetune_lora] RESCUE {kindTag} block {i}: {gCheckError}; " &
          &"rolled back this update, zeroed Adam m/v, LR scale {oldScale:.3f} -> " &
          &"{newScale:.3f} (attempt {fails}/{maxAttempts})."
        rescueTriggered = true
      GC_fullCollect()
    let tbCheck = getMonoTime()
    dlog &"backward block {i}: adam update + SSD spill xong"
    if rescueTriggered:
      # The update was rolled back. Never write layer i as completed or move
      # the progress cursor to i: that would skip i, especially for i == 0.
      # Retry exactly the same block with its original incoming dX and a
      # reduced LR; the inner fails counter bounds repeated rescue attempts.
      if stepIndex >= 0: discardPreparedResumeState(stepIndex)
      dX = dXBeforeBlock
      trainProfEnd(&"bwd block {i} (rescue retry)", tb0)
      GC_fullCollect()
      if gStopRequested:
        gTrainStepFullyDone = false
        stderr.writeLine &"[finetune_lora] RESCUE block {i}: stop before retry; resume at SAME block."
        break
      stderr.writeLine &"[finetune_lora] RESCUE block {i}: retry SAME block (not {i - 1})."
      continue
    if stepIndex >= 0:
      # Two-phase commit: LoRA layer snapshot+cursor is authoritative; prepared
      # dX is promoted to active .state only after the snapshot commits.
      # On restart a leftover .prepared is rolled forward ONLY when that
      # layer snapshot is intact; otherwise training stops without replay.
      if getEnv("BYBY_TEST_CRASH_BEFORE_LAYER_CURSOR", "") == $i:
        raise newException(IOError, "SIMULATED_CRASH_BEFORE_LAYER_CURSOR_" & $i)
      checkpoint.saveLayerProgress(getCheckpointPath(), stepIndex, i, layerTensors)
      if getEnv("BYBY_TEST_CRASH_AFTER_LAYER_CURSOR", "") == $i:
        raise newException(IOError, "SIMULATED_CRASH_AFTER_LAYER_CURSOR_" & $i)
      promotePreparedResumeState(stepIndex)
      if getEnv("BYBY_TEST_CRASH_AFTER_BACKWARD_LAYER", "") == $i:
        raise newException(CatchableError, "SIMULATED_CRASH_AFTER_BACKWARD_LAYER_" & $i)
    blockDone[i] = true
    let tbEnd = getMonoTime()
    stderr.writeLine &"[time] bwd block {i}: merge+load={dsec(tb0, tbMerge):.1f}s " &
      &"backwardBlock={dsec(tbMerge, tbBwd):.1f}s grad+check={dsec(tbBwd, tbGrad):.1f}s " &
      &"adam+SSD={dsec(tbGrad, tbUpd):.1f}s safety+probe={dsec(tbUpd, tbCheck):.1f}s " &
      &"spill={dsec(tbCheck, tbEnd):.1f}s total={dsec(tb0, tbEnd):.1f}s"
    trainProfEnd(&"bwd block {i}", tb0)
    # blockGrads het scope ngay day -> GC thu hoi, KHONG lan sang block ke
    GC_fullCollect()   # SUA: chay cho MOI block, giong nhu forward o tren
    # Ctrl+C: break only after the accepted layer has committed both progress
    # and the next dX; rescue retries never mark a layer completed.
    if gStopRequested:
      gTrainStepFullyDone = false
      stderr.writeLine &"[finetune_lora] Dung an toan sau backward layer {i} " &
        "(resume state + layer progress da tren SSD). Chay lai de tiep tuc."
      break
    dec i

  if gTrainStepFullyDone:
    # CAM thieu: step chi duoc coi la xong khi MOI block da backward.
    var missing: seq[int] = @[]
    for j in 0 ..< m.blocks.len:
      if not blockDone[j]: missing.add j
    if missing.len > 0:
      raise newException(ValueError,
        &"[check] step {stepIndex}: {missing.len} block CHUA backward ({missing}) - KHONG coi step la xong.")
    stderr.writeLine &"[check] step {stepIndex}: du {m.blocks.len}/{m.blocks.len} block da backward."
  # defer closeAndDelete() o dau ham da don scratch SSD ca khi return binh thuong
  # lan khi co exception.
  result = loss

const FillStepIndex* = 0   # lan lap tinh lai step 0

proc fillMarkerPath*(ckptPath: string): string =
  ## Co trong <ckpt>.inprogress/ = dang lap do (noi dung: K).
  checkpoint.progressDir(ckptPath) / "fill-step0.k"

proc auditMarkerPath*(ckptPath: string): string =
  ## Co trong <ckpt>.inprogress/ = dang kiem/sua update step 0 do (auditStep0LoraBlocks).
  checkpoint.progressDir(ckptPath) / "audit-step0"

proc step0RepairInProgress*(ckptPath: string): bool =
  ## Tien do trong <ckpt>.inprogress thuoc lan lap/kiem step 0 (stepIndex = FillStepIndex),
  ## KHONG phai 1 step train thuong.
  fileExists(fillMarkerPath(ckptPath)) or fileExists(auditMarkerPath(ckptPath))

proc fillUntrainedLoraBlocks*(m: var NimformerModelJamba, lo: var JambaLora,
                              loState: var LoraTrainState, adamStore: var AdamDiskStore,
                              attnStore: var AttnWeightDiskStore, newBlockNames: seq[string],
                              bpeTok: BPETokenizer, hfTok: HfTokenizer, usingHfTok: bool,
                              ctx: Backend, ckptStep: int): bool =
  ## LAP block LoRA bi bo trong (B = 0) cua checkpoint da commit dang do (bug
  ## FORCE commit cu: step 0 dung sau backward block 51 nhung van ghi step 1).
  ## Truoc step 0 moi B = 0 -> forward/backward cua step 0 = model goc, nen tam
  ## dat B cua block da train ve 0 + lay lai DUNG batch step 0 la tinh lai dung
  ## gradient; chi ap Adam (step 0 -> 1) cho block 0..K (K = block cao nhat co
  ## B = 0), block > K chi truyen dX roi duoc khoi phuc nguyen ven.
  ## Tra ve true neu da lap (caller phai commit checkpoint).
  var namesByBlock = initTable[int, seq[string]]()
  for name in lo.adapters.keys:
    if name.startsWith("block"):
      let rest = name["block".len .. ^1]
      let dotIdx = rest.find('.')
      if dotIdx > 0:
        namesByBlock.mgetOrPut(parseInt(rest[0 ..< dotIdx]), @[]).add(name)
  proc allZero(xs: seq[float32]): bool =
    for x in xs:
      if x != 0'f32: return false
    true
  var zeroBlock = newSeq[bool](m.blocks.len)
  var k = -1
  for bi in 0 ..< m.blocks.len:
    let names = namesByBlock.getOrDefault(bi, @[])
    zeroBlock[bi] = names.len > 0
    for name in names:
      if not allZero(lo.adapters[name].B.data): zeroBlock[bi] = false
    if zeroBlock[bi]: k = bi
  let ckpt = getCheckpointPath()
  let marker = fillMarkerPath(ckpt)
  if k < 0:
    if fileExists(marker):
      # .bin da chua ket qua lap (crash sau khi ghi .bin, truoc khi don) -> don tien do lap.
      echo "[fill] .bin da du 128 adapter; don tien do lap con sot lai."
      checkpoint.clearLayerProgress(ckpt)
      removeActivationResume(FillStepIndex)
    return false
  for bi in 0 .. k:
    if not zeroBlock[bi]:
      raise newException(ValueError,
        &"[fill] Block {bi} co B != 0 nhung block {k} co B = 0: khong phai step bi ngat tu tren xuong. Dung.")
  if ckptStep != 1:
    raise newException(ValueError,
      &"[fill] Checkpoint step={ckptStep} co block 0..{k} B = 0. Chi tai tao dung duoc step dau " &
      "(step 1, moi B = 0 truoc step) - dung de khong train lech.")
  echo &"[fill] Block 0..{k} chua tung duoc backward (B = 0) -> LAP: chay noi backward step 0 cho block {k}..0, " &
       &"block {k + 1}..{m.blocks.len - 1} giu nguyen"

  # RESUME: lan lap chay voi stepIndex = FillStepIndex nen dung chung co che
  # resume forward/backward + layer-progress cua train thuong. File marker
  # (ghi K) phan biet tien do lap voi tien do 1 step train binh thuong.
  let resuming = fileExists(marker)
  if resuming:
    let savedK = readFile(marker).strip
    if savedK != $k:
      raise newException(IOError,
        &"[fill] Tien do lap do ghi K={savedK} nhung checkpoint hien co B = 0 o block 0..{k}. " &
        &"Xoa {checkpoint.progressDir(ckpt)} de lap lai tu dau.")
  else:
    # Bo moi tien do/activation cu cua step FillStepIndex (khong thuoc lan lap nay).
    checkpoint.clearLayerProgress(ckpt)
    removeActivationResume(FillStepIndex)
    createDir(checkpoint.progressDir(ckpt))
    # Atomic: crash giua chung khong de lai marker rong (lan sau K="" -> raise).
    writeFile(marker & ".tmp", $k)
    if cRename((marker & ".tmp").cstring, marker.cstring) != 0:
      raise newException(IOError, "[fill] Khong ghi duoc marker " & marker)
  if resuming: recoverPreparedResumeState(FillStepIndex, m.blocks.len, lo)
  var curStep, curLayer: int
  let hasCursor = resuming and checkpoint.readProgressCursor(ckpt, curStep, curLayer) and
    curStep == FillStepIndex
  if hasCursor and (curLayer < 0 or curLayer > k):
    raise newException(IOError,
      &"[fill] Cursor lap do o block {curLayer} nam ngoai 0..{k}. Xoa {checkpoint.progressDir(ckpt)} de lap lai.")
  # Block doneFrom..k da lap xong o lan chay truoc: weight (layer-progress) + Adam
  # (SSD, main da nap) cua chung GIU NGUYEN, khong reset/khong update lai.
  let doneFrom = if hasCursor: curLayer else: k + 1
  if resuming:
    echo &"[fill] TIEP TUC lap do: " &
      (if hasCursor: &"block {curLayer}..{k} da lap xong, con block {curLayer - 1}..0"
       else: "chua co block nao lap xong (resume forward/backward tu trang thai SSD)")

  var trainedB = initTable[string, seq[float32]]()
  for bi in k + 1 ..< m.blocks.len:
    for name in namesByBlock.getOrDefault(bi, @[]):
      trainedB[name] = lo.adapters[name].B.data
      lo.adapters[name].B.data = newSeq[float32](lo.adapters[name].B.data.len)
  for bi in 0 ..< doneFrom:
    for name in namesByBlock.getOrDefault(bi, @[]):
      loState.adamA[name] = newApfAdamState(lo.adapters[name].A.data.len)
      loState.adamB[name] = newApfAdamState(lo.adapters[name].B.data.len)
  var byNameP = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byNameP[name] = t
  for name in newBlockNames:
    let rest = name["block".len .. ^1]
    let bi = parseInt(rest[0 ..< rest.find('.')])
    if bi < doneFrom:
      adamStore.init(name, numelShape(byNameP[name][].shape))   # chua tung update -> step 0
    if bi < doneFrom or bi > k:
      # Sidecar FP32 cua block G chua lap trong lan nay (va block > K) PHAI bang .bin:
      # 1 lan lap truoc bi ngat (ke ca binary cu) co the da update no -> update lan 2.
      # Ghi lai tu gia tri .bin (G attn/fuseGate luu fp32 raw nen khong mat gi).
      let p = byNameP[name]
      let plen = numelShape(p[].shape)
      if p[].data.len == plen:
        attnStore.saveW(name, p[].data)
      elif name in lo.quantBase:
        attnStore.saveW(name, dequantizeTensor(lo.quantBase[name]))
      else:
        raise newException(IOError, &"[fill] Khong co gia tri .bin cho '{name}' de dat lai sidecar")

  var loss = 0'f32
  if hasCursor and curLayer == 0:
    # Old binaries could write cursor=0 on a RESCUE rollback. Detect the
    # ambiguous legacy state before treating this as a completed backward.
    var backwardDone, canResume: bool
    discard layerProgressStepState(FillStepIndex, curLayer, m.blocks.len,
      backwardDone, canResume)
    # Backward da toi block 0 o lan truoc (chua kip commit): KHONG chay lai,
    # chi nap weight da lap tu layer-progress.
    discard restoreLayerProgressForStep(lo, ckpt, FillStepIndex, m.blocks.len,
      updateMaxBlock = k)
  else:
    var fillBatcher = newStreamingBatcher(CFG_DATASETS, CFG_SEQ_LEN, CFG_BATCH_SIZE,
                                          bpeTok, hfTok, usingHfTok)
    let (idsB, _, tgtMaskB) = fillBatcher.nextBatch()
    let gLrScale = min(1.0'f32, 1.0'f32 / float32(CFG_G_LR_WARMUP_STEPS))
    loss = trainStepGraftStreaming(m, lo, loState, adamStore, attnStore, CFG_NEWBLOCK_LR,
      idsB, tgtMaskB, ctx, gLrScale, stepIndex = FillStepIndex, updateMaxBlock = k)
    if gStopRequested and not gTrainStepFullyDone:
      stderr.writeLine "[fill] Dung an toan - tien do lap da tren SSD (" & checkpoint.progressDir(ckpt) &
        "). Chay lai de lap tiep, KHONG mat phan da lam."
      quit(130)
  for name, b in trainedB:
    lo.adapters[name].B.data = b
  var stillZero: seq[string] = @[]
  for name, ad in lo.adapters:
    if allZero(ad.B.data): stillZero.add(name)
  if stillZero.len > 0:
    raise newException(ValueError,
      &"[fill] {stillZero.len} adapter van B = 0 sau khi lap (RESCUE rollback?): {stillZero[0]}... " &
      &"KHONG commit. Xem log RESCUE; xoa {checkpoint.progressDir(ckpt)} de lap lai tu dau.")
  echo &"[fill] Lap xong: step 0 loss={loss:.4f}, {lo.adapters.len}/{lo.adapters.len} adapter co B != 0"
  result = true

proc auditStep0Core*(m: var NimformerModelJamba, lo: var JambaLora,
                     loState: var LoraTrainState, adamStore: var AdamDiskStore,
                     attnStore: var AttnWeightDiskStore,
                     idsB, tgtMaskB: seq[seq[int]], ctx: Backend, ckptStep: int): bool =
  ## TU DONG kiem + sua update step 0 cua checkpoint step 1 chua co co
  ## meta[Step0AuditMetaKey] (vd block 51..63 cua lan train 21/10: step 0 tinh qua
  ## lm_head random). Tinh lai DUNG step 0 (moi B = 0, block G = gia tri graft, batch
  ## step 0, lm_head that), backward tu block tren cung xuong: block nao dau B da luu
  ## khong khop gradient that -> reset Adam + update lai; gap block khop dau tien ->
  ## dung (block duoi do giu nguyen). Resume duoc (marker audit + layer progress).
  ## Tra ve true -> caller commit (.bin co co + Adam + commit marker).
  if ckptStep != 1: return false
  if getEnv("BYBY_STEP0_AUDIT", "1") == "0":
    echo "[audit] BYBY_STEP0_AUDIT=0 -> bo qua kiem update step 0."
    return false
  let ckpt = getCheckpointPath()
  let marker = auditMarkerPath(ckpt)
  let resuming = fileExists(marker)
  if resuming: recoverPreparedResumeState(FillStepIndex, m.blocks.len, lo)
  if gLoadedCkptMeta.getOrDefault(Step0AuditMetaKey) == "ok":
    if resuming:
      # Lan truoc da ghi .bin CO co (crash sau saveCheckpointLoraBin, truoc khi don):
      # kiem da xong. Lam tiep se so B MOI voi chinh no -> raise mai. Chi don tien do.
      echo "[audit] .bin da co co xac nhan step 0 - don tien do kiem con sot lai."
      checkpoint.clearLayerProgress(ckpt)
      removeActivationResume(FillStepIndex)
    return false
  for name, ad in lo.adapters:
    var nz = false
    for v in ad.B.data:
      if v != 0'f32:
        nz = true
        break
    if not nz:
      raise newException(ValueError,
        &"[audit] '{name}' con B = 0 - phai lap (fillUntrainedLoraBlocks) truoc khi kiem step 0.")
  if resuming:
    echo "[audit] TIEP TUC kiem/sua update step 0 do (" & marker & ")."
  else:
    echo "[audit] Checkpoint step 1 chua duoc xac nhan update step 0 -> tinh lai step 0 " &
         "(forward 64 block + backward tu block tren cung toi block dung dau tien)."
    checkpoint.clearLayerProgress(ckpt)
    removeActivationResume(FillStepIndex)
    createDir(checkpoint.progressDir(ckpt))
    writeFile(marker & ".tmp", "1")
    if cRename((marker & ".tmp").cstring, marker.cstring) != 0:
      raise newException(IOError, "[audit] Khong ghi duoc marker " & marker)
  var curStep, curLayer: int
  let hasCursor = resuming and checkpoint.readProgressCursor(ckpt, curStep, curLayer) and
    curStep == FillStepIndex
  if hasCursor and (curLayer < 0 or curLayer >= m.blocks.len):
    raise newException(IOError,
      &"[audit] Cursor {curLayer} ngoai 0..{m.blocks.len - 1}. Xoa {checkpoint.progressDir(ckpt)} de kiem lai.")

  # B da luu (.bin) de so dau; roi MOI B = 0 = dieu kien truoc step 0. A khong doi
  # qua step 0 (dA = 0 khi B = 0) nen dung nguyen.
  var refB = initTable[string, seq[float32]]()
  for name in toSeq(lo.adapters.keys):
    refB[name] = lo.adapters[name].B.data
    lo.adapters[name].B.data = newSeq[float32](refB[name].len)
  proc blockOf(name: string): int =
    let rest = name["block".len .. ^1]
    parseInt(rest[0 ..< rest.find('.')])

  gAuditStopBlock = -1
  var loss = 0'f32
  var stopBlock = -1
  if hasCursor and curLayer == 0:
    var backwardDone, canResume: bool
    discard layerProgressStepState(FillStepIndex, curLayer, m.blocks.len,
      backwardDone, canResume)
    # Lan truoc da sua toi block 0 (moi block deu sai), chua kip commit: chi nap ket qua.
    discard restoreLayerProgressForStep(lo, ckpt, FillStepIndex, m.blocks.len)
  else:
    let gLrScale = min(1.0'f32, 1.0'f32 / float32(CFG_G_LR_WARMUP_STEPS))
    loss = trainStepGraftStreaming(m, lo, loState, adamStore, attnStore, CFG_NEWBLOCK_LR,
      idsB, tgtMaskB, ctx, gLrScale, stepIndex = FillStepIndex,
      updateMaxBlock = m.blocks.len - 1, auditRefB = refB)
    if gStopRequested and not gTrainStepFullyDone:
      stderr.writeLine "[audit] Dung an toan - tien do tren SSD (" & checkpoint.progressDir(ckpt) &
        "). Chay lai de tiep tuc, KHONG mat phan da lam."
      quit(130)
    stopBlock = gAuditStopBlock
  # Block <= stopBlock: dung -> tra B da luu. Block > stopBlock: vua sua (B moi).
  var nFixed = 0
  for name, b in refB:
    let bi = blockOf(name)
    if bi <= stopBlock:
      lo.adapters[name].B.data = b
    else:
      let nb = lo.adapters[name].B.data
      var nz = false
      for v in nb:
        if v != 0'f32:
          nz = true
          break
      if not nz:
        raise newException(ValueError,
          &"[audit] '{name}' (block {bi}) van B = 0 sau khi sua (RESCUE rollback?) - KHONG commit. " &
          &"Chay lai de sua tiep block nay.")
      if nb == b:
        raise newException(ValueError,
          &"[audit] '{name}' (block {bi}) bi danh dau sai nhung B moi giong het B cu - KHONG commit.")
      inc nFixed
  if stopBlock == m.blocks.len - 1:
    echo &"[audit] Moi block deu co update step 0 dung (loss step 0={loss:.4f}) - chi ghi co xac nhan."
  else:
    echo &"[audit] Xong: sua block {stopBlock + 1}..{m.blocks.len - 1} ({nFixed} adapter), " &
         (if stopBlock >= 0: &"block 0..{stopBlock} dung (giu nguyen)" else: "khong co block nao dung tu truoc") &
         &", loss step 0={loss:.4f}"
  gStep0Verified = true
  result = true

proc auditStep0LoraBlocks*(m: var NimformerModelJamba, lo: var JambaLora,
                           loState: var LoraTrainState, adamStore: var AdamDiskStore,
                           attnStore: var AttnWeightDiskStore,
                           bpeTok: BPETokenizer, hfTok: HfTokenizer, usingHfTok: bool,
                           ctx: Backend, ckptStep: int): bool =
  ## Nhu auditStep0Core voi DUNG batch step 0 (batcher moi tu dau dataset, nhu lap).
  if ckptStep != 1 or getEnv("BYBY_STEP0_AUDIT", "1") == "0" or
     gLoadedCkptMeta.getOrDefault(Step0AuditMetaKey) == "ok":
    return auditStep0Core(m, lo, loState, adamStore, attnStore, @[], @[], ctx, ckptStep)
  var auditBatcher = newStreamingBatcher(CFG_DATASETS, CFG_SEQ_LEN, CFG_BATCH_SIZE,
                                         bpeTok, hfTok, usingHfTok)
  let (idsB, _, tgtMaskB) = auditBatcher.nextBatch()
  auditStep0Core(m, lo, loState, adamStore, attnStore, idsB, tgtMaskB, ctx, ckptStep)

proc checkLoraCheckpointIntegrity*(m: var NimformerModelJamba, lo: JambaLora, step: int,
                                   newBlockNames: seq[string]): seq[int] =
  ## Kiem tra KY trang thai LoRA + block G vua nap (resume) / vua tao (run moi) TRUOC
  ## khi train/lap. Raise neu co loi lam hong train: thieu adapter, sai kich thuoc,
  ## alpha/scale sai, NaN/Inf, A toan 0 (grad B = g.(A x)^T = 0 -> adapter KHONG BAO
  ## GIO hoc duoc), thieu weight block G. Tra ve cac block co B = 0 o MOI adapter
  ## (block TRONG) - co che bu fillUntrainedLoraBlocks se lap chung.
  var problems: seq[string] = @[]
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  var nAdapters = 0
  for i, c in m.pattern:
    var names: seq[string] = @[]
    if c in {'R', 'r', 'P', 'p', 'G', 'g'}:
      names = @[&"block{i}.mambaReal.inProj.weight", &"block{i}.mambaReal.outProj.weight"]
    elif c in {'M', 'm'}:
      names = @[&"block{i}.mamba.inProj.weight", &"block{i}.mamba.outProj.weight"]
    var blockEmpty = names.len > 0
    var zeroBNames: seq[string] = @[]
    for name in names:
      if name notin lo.adapters:
        problems.add "thieu adapter " & name
        blockEmpty = false
        continue
      inc nAdapters
      let ad = lo.adapters[name]
      if ad.r <= 0 or ad.A.data.len != ad.r * ad.inF or ad.B.data.len != ad.outF * ad.r:
        problems.add &"{name}: A/B sai kich thuoc"
        blockEmpty = false
        continue
      if ad.scale.classify in {fcNan, fcInf, fcNegInf} or ad.scale != ad.alpha / float32(ad.r):
        problems.add &"{name}: alpha/scale khong hop le"
      var aNonZero, bNonZero, aBad, bBad = false
      for v in ad.A.data:
        if v.classify in {fcNan, fcInf, fcNegInf}: aBad = true
        elif v != 0'f32: aNonZero = true
      for v in ad.B.data:
        if v.classify in {fcNan, fcInf, fcNegInf}: bBad = true
        elif v != 0'f32: bNonZero = true
      if aBad: problems.add &"{name}: A co NaN/Inf"
      if bBad: problems.add &"{name}: B co NaN/Inf"
      if not aNonZero:
        problems.add &"{name}: A toan 0 - adapter KHONG BAO GIO hoc duoc"
      if bNonZero or bBad: blockEmpty = false
      if not bNonZero and not bBad: zeroBNames.add name
    if blockEmpty: result.add i
    elif step >= 1 and zeroBNames.len > 0:
      # Nua block trong: co che bu chi lap block trong CA 2 adapter -> se ket o luc luu.
      problems.add &"block {i}: adapter {zeroBNames} co B = 0 nhung adapter con lai da train " &
        "(nua block trong - co che bu khong lap duoc)"
  var nNew = 0
  for name in newBlockNames:
    if name notin byName:
      problems.add "thieu tham so block moi " & name
      continue
    let p = byName[name]
    if p[].data.len == numelShape(p[].shape):
      var bad = false
      for v in p[].data:
        if v.classify in {fcNan, fcInf, fcNegInf}: bad = true
      if bad: problems.add name & ": NaN/Inf"
      inc nNew
    elif name in lo.quantBase:
      inc nNew
    else:
      problems.add name & ": khong co weight (RAM lan quantBase)"
  var emptyDesc = "khong co block trong"
  if result.len > 0:
    var contiguous = true
    for j in 1 ..< result.len:
      if result[j] != result[j - 1] + 1: contiguous = false
    emptyDesc = &"{result.len} block TRONG (B = 0): " &
      (if contiguous: &"{result[0]}..{result[^1]}" else: $result)
  echo &"[check] step={step}: {nAdapters} adapter LoRA, {nNew}/{newBlockNames.len} tham so block moi, " &
       &"{emptyDesc}, {problems.len} loi"
  if problems.len > 0:
    for p in problems[0 ..< min(10, problems.len)]: stderr.writeLine "[check]   " & p
    raise newException(ValueError,
      &"[check] Checkpoint/LoRA hong ({problems.len} loi, vd: {problems[0]}) - KHONG train.")
  if step == 0 and result.len == m.blocks.len:
    echo "[check] Run moi: moi B = 0 la binh thuong o step 0."
  elif result.len > 0:
    echo "[check] -> co che bu (fillUntrainedLoraBlocks) se lap cac block trong truoc khi train tiep."

when isMainModule:
  setControlCHook(onTrainCtrlC)
  if CFG_CODESTRAL_PATTERN.len != 64:
    raise newException(ValueError,
      "BYBY_CODESTRAL_PATTERN must describe exactly 64 Codestral layers")
  for c in CFG_CODESTRAL_PATTERN:
    if c notin {'R', 'r', 'G', 'g'}:
      raise newException(ValueError,
        "BYBY_CODESTRAL_PATTERN only supports R and G; P is disabled for finetuning")
  if 'G' in CFG_CODESTRAL_PATTERN or 'g' in CFG_CODESTRAL_PATTERN:
    echo &"[finetune_lora] G enabled: mambaY RMS≤{G_MAX_MAMBA_RMS_RATIO:.1f}x, " &
      &"weighted side RMS≤{G_MAX_SIDE_RMS_RATIO:.1f}x, " &
      &"gate-mean floor={G_MIN_GATE_MEAN:.2f}, " &
      &"gate bias window=[{CFG_G_GATE_BIAS_MIN:.1f},{CFG_G_GATE_BIAS_MAX:.1f}], " &
      &"sidecar LR={CFG_G_SIDECAR_LR:.1e}, gate LR={CFG_G_GATE_LR:.1e} " &
      &"(warmup {CFG_G_LR_WARMUP_STEPS} steps), per-tensor grad clip={CFG_G_GRAD_CLIP_NORM:.1f}"
  if 'R' in CFG_CODESTRAL_PATTERN or 'r' in CFG_CODESTRAL_PATTERN:
    echo &"[finetune_lora] R rescue enabled: residual RMS/max={CFG_R_MAX_OUTPUT_RMS_RATIO:.1f}/" &
      &"{CFG_R_MAX_OUTPUT_MAX_RATIO:.1f}x, mixer RMS/max={CFG_R_MAX_MIXER_RMS_RATIO:.1f}/" &
      &"{CFG_R_MAX_MIXER_MAX_RATIO:.1f}x, drift≤{CFG_R_LAYER_PROBE_DRIFT:.2f}, " &
      &"max attempts={CFG_R_MAX_RESCUE_ATTEMPTS}, min LR scale={CFG_R_LR_MIN_SCALE:.3f}"
  if getEnv("BYBY_LAYER_PROBE", "1") != "0":
    echo &"[finetune_lora] LAYER_PROBE on (default): every {CFG_LAYER_PROBE_EVERY} accepted updates " &
      &"(R drift≤{CFG_R_LAYER_PROBE_DRIFT:.2f}, other≤{CFG_LAYER_PROBE_GARBAGE_FRAC:.2f}; " &
      "tắt bằng BYBY_LAYER_PROBE=0)"
  if getEnv("BYBY_G_GATE_STD_RESCUE", "0") != "0":
    echo &"[finetune_lora] G gate-std rescue on: min std={CFG_G_GATE_BIAS_MIN_STD:.1e}"

  # newBackend() mac dinh "auto": backend.nim.resolveBackendKind se RAISE loi
  # neu KHONG tim thay GPU nao (CUDA/Metal/OpenCL/Vulkan) tren may - khong bao
  # gio am tham rot ve CPU (gForbidCpuFallback=true mac dinh) - dung yeu cau
  # "phai 100% GPU". Neu muon ep 1 backend cu the: newBackend("cuda"/"metal"/...).
  # Allow CPU fallback for OOM recovery on memory-constrained GPUs.
  setForbidCpuFallback(false)
  let ctx = newBackend()

  # FAIL-FAST: khoi tao GPU + JIT NGAY (truoc khi nap model ~ hang gio). Truoc day
  # initMicroVmJit/warmupMicroVmJit chay SAU khi nap weight -> loi "[metal_jit] gDevice nil"
  # chi lo ra sau ~2 gio. Phai init device TRUOC (metal_available chi do GPU, khong gan gDevice).
  ensureTgCap(ctx)
  initMicroVmJit("finetune_lora", ctx.kind.toByby())
  warmupMicroVmJit(12)

  var m: NimformerModelJamba
  var oldStep = 0
  var quantResidentWeights = initTable[string, QuantTensor]()  # CHI dung TAM de dung lo -
                                                                  # se duoc .clear() ngay sau
                                                                  # khi lo.quantBase nhan xong,
                                                                  # KHONG giu ban copy thua nao
                                                                  # ca chuong trinh (xem duoi).
  var loFromResume: JambaLora
  var resumed = false

  # MOI: AUTO-RESUME - neu <CFG_CHECKPOINT_PATH>.lora.json DA TON TAI (vd tu
  # lan chay truoc bi KILL giua chung do canh tranh RAM he thong - xem hoi
  # thoai), TU DONG doc tiep tu do thay vi train lai TU DAU. May bi kill bao
  # nhieu lan cung khong mat qua 1 epoch dang do (CFG_CHECKPOINT_EVERY quyet
  # dinh tan suat luu). CHI ap dung cho CFG_INIT_MODE=="codestral" (mode
  # "graft" doc .bin rieng, khong lien quan checkpoint LoRA nay).
  if CFG_INIT_MODE == "codestral" and fileExists(getCheckpointPath()):
    echo &"[finetune_lora] TIM THAY checkpoint binary ({getCheckpointPath()}) - " &
         "TU DONG RESUME thay vi train lai tu dau."
    (m, loFromResume, oldStep) = loadCheckpointLoraBin(getCheckpointPath())
    resumed = true
    # A crash can land between atomic checkpoint replacement and the final
    # resume marker. In that case the canonical model may be one step newer
    # than the committed dataset cursor/Adam state. Restore the previous model
    # before any training state is initialized; otherwise weights and cursor
    # silently refer to different steps.
    var recoveryStep = -1
    var recoverySeqLen, recoveryBatch = 0
    var recoveryPattern, recoveryDatasetSig, recoveryTokenizerDir = ""
    let recoveryHasCommit = readResumeCommit(getCheckpointPath(), recoveryStep,
      recoverySeqLen, recoveryBatch, recoveryPattern, recoveryDatasetSig,
      recoveryTokenizerDir)
    let recoveryInProgress = fileExists(getCheckpointPath() & ".resume.inprogress")
    if recoveryInProgress and oldStep > 0 and not recoveryHasCommit:
      raise newException(IOError,
        "[resume] Co checkpoint step>0 va .resume.inprogress nhung khong co commit marker. " &
        "Khong the chung minh model/Adam/dataset cursor cung step; dung an toan de tranh resume lech.")
    if recoveryHasCommit and oldStep != recoveryStep:
      let backupPath = getCheckpointPath() & ".prev"
      if not fileExists(backupPath):
        raise newException(IOError,
          &"[resume] Checkpoint step={oldStep} khong khop commit step={recoveryStep}, " &
           "co .resume.inprogress nhung khong co rollback backup. Dung de tranh train lech state.")
      let (prevModel, prevLora, prevStep) = loadCheckpointLoraBin(backupPath)
      if prevStep != recoveryStep:
        raise newException(IOError,
          &"[resume] Rollback backup step={prevStep} khong khop commit step={recoveryStep}; " &
           "giu nguyen cac file va dung an toan.")
      let restoreTmp = getCheckpointPath() & ".restore.tmp"
      copyFile(backupPath, restoreTmp)
      if cRename(restoreTmp.cstring, getCheckpointPath().cstring) != 0:
        try: removeFile(restoreTmp)
        except OSError: discard
        raise newException(IOError, "[resume] Khong the atomic rollback checkpoint tu .prev")
      m = prevModel
      loFromResume = prevLora
      oldStep = prevStep
      echo &"[resume] Da khoi phuc checkpoint step={oldStep} theo commit marker; Adam/activation sidecar se duoc kiem tra lai."
    if m.pattern != CFG_CODESTRAL_PATTERN:
      raise newException(ValueError,
        &"Checkpoint pattern '{m.pattern}' khong khop parallel pattern '{CFG_CODESTRAL_PATTERN}'. " &
        "Khong the resume checkpoint cu thay A cho R; chon duong checkpoint moi de train sidecar song song.")
    echo &"[finetune_lora] Resume tu epoch={oldStep}, pattern='{m.pattern}' ({m.blocks.len} block)"
    if fileExists(getCheckpointPath() & ".hf_tokenizer_dir"):
      let savedTokDir = readFile(getCheckpointPath() & ".hf_tokenizer_dir").strip
      if savedTokDir.len > 0 and savedTokDir != CFG_HF_DIR:
        raise newException(ValueError,
          &"[resume] Sidecar tokenizer checkpoint='{savedTokDir}' khong khop CFG_HF_DIR='{CFG_HF_DIR}'. Dung ngay, khong train.")

  if not resumed:
   case CFG_INIT_MODE
   of "codestral":
    echo &"[finetune_lora] CFG_INIT_MODE=codestral - dung model MOI, pattern='{CFG_CODESTRAL_PATTERN}'"
    m = newNimformerModelJamba(CFG_CODESTRAL_VOCAB, CFG_CODESTRAL_HIDDEN, CFG_CODESTRAL_NHEADS,
                                ffMult = 4, pattern = CFG_CODESTRAL_PATTERN, useSession = true,
                                mambaRealDState = CFG_CODESTRAL_DSTATE,
                                mambaRealNGroups = CFG_CODESTRAL_NGROUPS,
                                mambaRealConvKernel = CFG_CODESTRAL_CONVKERNEL,
                                mambaRealExpand = CFG_CODESTRAL_EXPAND,
                                quantResident = (CFG_HF_SOURCE == "nimq"))
    echo &"[finetune_lora] Nap weight pretrain (nguon={CFG_HF_SOURCE}) vao block 'R' ..."
    case CFG_HF_SOURCE
    of "nimq":
      # QUANT-RESIDENT: inProj/outProj cua block 'R' GIU DANG NEN (khong
      # dequant het ~29GB fp32 luc load) - dequant TUC THOI qua LUT moi khi
      # can 1 block, xoa ngay sau dung (xem trainStepGraftStreaming). BAT
      # BUOC cho may RAM han che (vd 16GB) - xem hoi thoai giai thich RAM.
      quantResidentWeights = loadCodestralQuantResidentFromNimq(m, CFG_NIMQ_PATH)
      syncRmsFromBaseNorms(m)
    of "safetensors":
      echo "[finetune_lora] CANH BAO: nguon safetensors dequant TOAN BO weight " &
           "ve fp32 ngay luc load (~29GB) - KHONG streaming duoc (chi .nimq moi " &
           "co du lieu nen de dequant-on-demand). Voi may RAM han che, doi sang nguon 'nimq'."
      loadCodestralIntoJamba(m, CFG_HF_DIR)
      syncRmsFromBaseNorms(m)
    else:
      raise newException(ValueError, &"CFG_HF_SOURCE khong hop le: '{CFG_HF_SOURCE}' (chi nhan 'nimq'/'safetensors')")
   of "graft":
    echo &"[finetune_lora] CFG_INIT_MODE=graft - doc checkpoint DA GRAFT SAN: {CFG_OLD_CHECKPOINT}"
    (m, oldStep) = jamba_hybrid.loadCheckpoint(CFG_OLD_CHECKPOINT)
    echo "[finetune_lora] KHONG graft lai (checkpoint nay da la ket qua graft_attention.nim roi - " &
         "xem canh bao dau file, graft 2 lan se sai kien truc)."
   else:
    raise newException(ValueError, &"CFG_INIT_MODE khong hop le: '{CFG_INIT_MODE}' (chi nhan 'codestral'/'graft')")
  echo &"[finetune_lora] pattern='{m.pattern}' ({m.blocks.len} block), step da train truoc={oldStep}"
  if m.pattern.len != m.blocks.len:
    raise newException(ValueError,
      &"[preflight] pattern len={m.pattern.len} nhung model blocks={m.blocks.len}. Dung ngay, khong train.")
  if 'P' in m.pattern or 'p' in m.pattern:
    raise newException(ValueError,
      "[preflight] Block P is disabled for finetuning; choose an R/G checkpoint or start a new run.")
  if CFG_INIT_MODE == "codestral":
    if m.vocab != CFG_CODESTRAL_VOCAB or m.embedDim != CFG_CODESTRAL_HIDDEN or m.nHeads != CFG_CODESTRAL_NHEADS:
      raise newException(ValueError,
        &"[preflight] Model shape khong dung Codestral: vocab={m.vocab}, hidden={m.embedDim}, nHeads={m.nHeads}; " &
         &"expected {CFG_CODESTRAL_VOCAB}/{CFG_CODESTRAL_HIDDEN}/{CFG_CODESTRAL_NHEADS}. Dung ngay, khong train.")

  # Gan LoRA len TAT CA weight 2D cua cac block DA PRETRAIN ('M' cu hoac 'R'
  # moi - ca 2 deu la "da hoc roi, chi finetune nhe") - doc TRUC TIEP tu
  # m.pattern. Block 'A'/'S' (moi cay, chua hoc gi) KHONG gan LoRA, train
  # full o duoi - GIU NGUYEN logic cu, chi mo rong sang 'R'.
  var loraTargets: seq[string] = @[]
  var newBlockIndices: seq[int] = @[]
  for i, c in m.pattern:
    if c in {'M', 'm'}:
      loraTargets.add(&"block{i}.mamba.inProj.weight")
      loraTargets.add(&"block{i}.mamba.outProj.weight")
    elif c in {'R', 'r', 'P', 'p'}:
      loraTargets.add(&"block{i}.mambaReal.inProj.weight")
      loraTargets.add(&"block{i}.mambaReal.outProj.weight")
      if c in {'P', 'p'}: newBlockIndices.add(i)
    elif c in {'G', 'g'}:
      # BUG FIX — root cause of "IndexDefect: container is empty" at the first
      # G block: 'G' was missing from the R/P case, so block G's
      # mambaReal.inProj/outProj.weight never got a LoRA adapter. namesByBlock
      # therefore did not contain them, mergeInOne was never called, and with
      # quantResident the .data stayed @[] (empty). The first Linear.forward
      # inside G then crashed on the empty weight buffer.
      # G keeps the pretrained Mamba2Real weights and trains them via LoRA
      # exactly like R/P; attn. and fuseGate. are the new-block full-precision
      # parameters (see isNewBlockParameter).
      loraTargets.add(&"block{i}.mambaReal.inProj.weight")
      loraTargets.add(&"block{i}.mambaReal.outProj.weight")
      newBlockIndices.add(i)
    else:
      newBlockIndices.add(i)
  var lo: JambaLora
  if resumed:
    # Resume: lo (adapters + quantBase) da doc tu file .lora.json roi, KHONG
    # goi attachLora*/attachLoraQuantResident nua (se tao adapter MOI random-init,
    # de mat sach adapter da train truoc do).
    lo = loFromResume
    # SUA (RO RI THAT, tim ra khi soat lai): sau dong tren, lo.quantBase da
    # co BAN COPY RIENG cua chinh no (Table assignment = deep copy trong
    # Nim). loFromResume.quantBase VAN CON SONG SONG voi ban copy do suot
    # phan doi cua chuong trinh (loFromResume la bien khai bao o scope
    # ngoai, khong tu mat di) - lang phi ~3.5GB (kich thuoc file .nimq nen)
    # KHONG can thiet. Xoa NGAY tai day - lo.quantBase da co ban rieng cua
    # no roi, khong can loFromResume.quantBase nua.
    loFromResume.quantBase.clear()
    loFromResume.adapters.clear()
  elif quantResidentWeights.len > 0:
    # nguon nimq: 'R' block da co dung 1:1 ten trong quantResidentWeights
    # (xem loadCodestralQuantResidentFromNimq) - KHONG dung loraTargets/
    # attachLora thuong (se doc t[].data dang la rac, chua duoc dien).
    lo = attachLoraQuantResident(m, quantResidentWeights, r = CFG_LORA_RANK,
                                 alpha = CFG_LORA_ALPHA, targetNames = loraTargets)
    # SUA (bug loss 35 o run moi): lm_head that chi nam trong quantBase
    # ("outProj.weight"); m.outProj van la newLinear() RANDOM (useQuant=false)
    # cho toi khi bind. Duong resume (loadCheckpointLoraBin) da bind, duong
    # run moi thi khong -> loss/dX cua moi step dau tien tinh qua lm_head ngau
    # nhien (loss ~35 thay vi ~2.8), gradient LoRA vo nghia.
    bindQuantBaseToLinears(m, lo)
    # SUA (RO RI THAT): attachLoraQuantResident() COPY quantResidentWeights
    # vao lo.quantBase (result.quantBase = quantWeights, xem lora.nim) - tu
    # day lo.quantBase la ban CHINH THUC, quantResidentWeights o day chi con
    # la ban THUA, khong con dung lai noi dung cua no o dau nua trong ca
    # chuong trinh (moi cho kiem tra sau nay da doi sang lo.quantBase.len,
    # xem duoi) - xoa NGAY, tranh giu ~3.5GB thua suot phan con lai.
    quantResidentWeights.clear()
  else:
    lo = attachLora(m, loraTargets, r = CFG_LORA_RANK, alpha = CFG_LORA_ALPHA)

  # PREFLIGHT LoRA: tung adapter phai map dung weight 2D, shape A/B phai khop
  # chinh xac. Neu sai thi DUNG truoc batch dau tien, tranh train vao tensor rac.
  block:
    var byNameCheck = initTable[string, ptr Tensor]()
    for (name, t) in paramList(m): byNameCheck[name] = t
    var expectedAdapters = initTable[string, bool]()
    for name in loraTargets: expectedAdapters[name] = true
    for name, _ in expectedAdapters:
      if name notin lo.adapters:
        raise newException(ValueError, "[preflight] Thieu LoRA target bat buoc: " & name)
    for name, ad in lo.adapters:
      if name notin expectedAdapters:
        raise newException(ValueError, "[preflight] Adapter LoRA ngoai target list: " & name)
      if name notin byNameCheck:
        raise newException(ValueError, "[preflight] LoRA target khong ton tai trong model: " & name)
      let baseShape = byNameCheck[name][].shape
      if baseShape.len != 2 or baseShape[0] != ad.outF or baseShape[1] != ad.inF:
        raise newException(ValueError,
          &"[preflight] LoRA '{name}' base shape=" & $baseShape &
           &" nhung adapter={ad.outF}x{ad.inF}, r={ad.r}. Dung ngay, khong train.")
      if ad.A.shape != @[ad.r, ad.inF] or ad.B.shape != @[ad.outF, ad.r] or
         ad.A.data.len != ad.r * ad.inF or ad.B.data.len != ad.outF * ad.r:
        raise newException(ValueError, "[preflight] LoRA tensor shape/data len sai cho " & name)
  echo &"[preflight] LoRA adapter mapping OK: {lo.adapters.len} adapter"
  if "outProj.weight" in lo.quantBase and not m.outProj.useQuant:
    raise newException(ValueError,
      "[preflight] lm_head (outProj.weight) co trong quantBase nhung chua bind vao m.outProj - " &
      "loss se tinh qua lm_head random-init. Dung ngay, khong train.")

  var loState = newLoraTrainState(lo, lr = CFG_LORA_LR)

  # Adam rieng cho tham so cac block moi (A/S) - CHI cac block trong
  # newBlockIndices, khong dong den block M (da LoRA-hoa) hay embed/outProj
  # (giu nguyen, dung tinh than "dong bang toi da, chi hoc phan can hoc").
  var adamStore = initAdamDiskStore(CFG_ADAM_DIR)
  let commitOnlyMode = getEnv("BYBY_COMMIT_ONLY", "0") != "0"
  if not resumed and step0RepairInProgress(getCheckpointPath()):
    # Tien do LAP/KIEM step 0 (tinh tren .bin cu) nhung .bin khong con. Con .prev ->
    # rat co the .bin bi xoa nham: KHONG tu xoa tien do (mat nhieu gio), dung + huong dan.
    if fileExists(getCheckpointPath() & ".prev"):
      raise newException(IOError,
        &"[fill] Con tien do lap/kiem step 0 do ({checkpoint.progressDir(getCheckpointPath())}) nhung " &
        &"khong con {getCheckpointPath()}. Khoi phuc roi chay lai de lam tiep:\n" &
        &"  cp -p {getCheckpointPath()}.prev {getCheckpointPath()}\n" &
        &"Neu that su muon train moi tu dau: xoa {checkpoint.progressDir(getCheckpointPath())} truoc.")
    # Khong con ca .bin lan .prev -> run moi: tien do do KHONG thuoc run nay, de lai thi
    # trainStep step 0 se resume nham vao no. Bo di truoc khi bat cu thu gi doc .inprogress.
    echo "[fill] Khong con checkpoint .bin/.prev nhung con tien do lap do -> bo tien do lap, train moi tu dau."
    checkpoint.clearLayerProgress(getCheckpointPath())
    removeActivationResume(FillStepIndex)
  let hasInprogress = dirExists(checkpoint.progressDir(getCheckpointPath()))
  # Dang lap/kiem step 0 do (fillUntrainedLoraBlocks / auditStep0LoraBlocks bi ngat): tien
  # do nam trong <ckpt>.inprogress voi stepIndex = FillStepIndex, KHONG phai 1 step train thuong.
  var fillInProgress = resumed and step0RepairInProgress(getCheckpointPath())
  if oldStep == 0 and not commitOnlyMode and not hasInprogress:
    # Chi cleanup Adam khi that su run moi (khong con inprogress de recover).
    adamStore.cleanup()
    adamStore = initAdamDiskStore(CFG_ADAM_DIR)
  elif hasInprogress:
    echo "[finetune_lora] Phat hien inprogress — GIU Adam disk (khong cleanup)."
  var attnStore = initAttnWeightDiskStore(CFG_ATTN_WEIGHTS_DIR)
  if oldStep == 0 and not commitOnlyMode and not hasInprogress:
    # Run moi: khong dung weight 'A' cu (neu co) con sot tu run truoc -
    # bootstrap trong trainStepGraftStreaming se ghi lai tu m hien tai.
    attnStore.cleanupAttn()
    attnStore = initAttnWeightDiskStore(CFG_ATTN_WEIGHTS_DIR)
  var newBlockNames: seq[string] = @[]
  for (name, t) in paramList(m):
    for bi in newBlockIndices:
      if name.startsWith(&"block{bi}."):
        if m.blocks[bi].kind == bkMambaRealPlusAttn and not name.startsWith(&"block{bi}.attn."):
          continue
        if m.blocks[bi].kind == bkMambaRealGated and
           not (name.startsWith(&"block{bi}.attn.") or name.startsWith(&"block{bi}.fuseGate.")):
          continue
        let plen = numelShape(t[].shape)
        if not adamStore.has(name, plen):
          adamStore.init(name, plen)
        newBlockNames.add(name)
        break

  # Kiem tra KY LoRA + block G truoc moi thu khac (raise neu hong; bao block trong).
  discard checkLoraCheckpointIntegrity(m, lo, oldStep, newBlockNames)
  # Recover a crash between per-layer cursor commit and resume-state commit.
  # An unfinished optimizer write without a complete layer snapshot FAILS CLOSED.
  if hasInprogress:
    recoverPreparedResumeState(oldStep, m.blocks.len, lo)
    if oldStep != FillStepIndex and step0RepairInProgress(getCheckpointPath()):
      recoverPreparedResumeState(FillStepIndex, m.blocks.len, lo)

  # RESUME INTEGRITY: checkpoint co weight chua du; phai kiem tra Adam + cursor.
  let currentDatasetSig = datasetSignature(CFG_DATASETS)
  var committedStep = -1
  var committedSeqLen = 0
  var committedBatch = 0
  var committedPattern = ""
  var committedDatasetSig = ""
  var committedTokenizerDir = ""
  let hasCommit = readResumeCommit(getCheckpointPath(), committedStep, committedSeqLen, committedBatch, committedPattern, committedDatasetSig, committedTokenizerDir)
  let inProgress = fileExists(getCheckpointPath() & ".resume.inprogress")
  var attnStoreTrusted = false
  if resumed and (oldStep > 0 or hasCommit or inProgress):
    if not hasCommit or committedStep != oldStep:
      echo "[resume] CANH BAO: khong co commit marker dung step -> khong tin resume sidecar FP32."
    elif committedSeqLen != CFG_SEQ_LEN or committedBatch != CFG_BATCH_SIZE or committedPattern != m.pattern:
      raise newException(ValueError,
        &"[resume] Config khong khop checkpoint: step={committedStep}, seqLen={committedSeqLen}, batch={committedBatch}, pattern='{committedPattern}' " &
         &"nhung run hien tai seqLen={CFG_SEQ_LEN}, batch={CFG_BATCH_SIZE}, pattern='{m.pattern}'. KHONG TRAIN.")
    elif committedDatasetSig.len > 0 and committedDatasetSig != currentDatasetSig and
         committedDatasetSig != datasetSignatureSizeOnly(CFG_DATASETS):
      raise newException(ValueError,
        "[resume] Dataset file size/signature da thay doi sau checkpoint. KHONG skip/resume vao vi tri cu; " &
        "giu checkpoint hien tai va dung de tranh train tren corpus lech.")
    elif committedDatasetSig.len > 0 and committedDatasetSig != currentDatasetSig:
      echo "[resume] Commit dung signature legacy path+size; tiep tuc voi canh bao khong co mtime."
    elif CFG_INIT_MODE == "codestral" and committedTokenizerDir.len > 0 and committedTokenizerDir != CFG_HF_DIR:
      raise newException(ValueError,
        &"[resume] Tokenizer directory checkpoint='{committedTokenizerDir}' khong khop run hien tai='{CFG_HF_DIR}'. KHONG TRAIN.")
    else:
      let attnFilesOk = attnStoreAllValid(attnStore, m, newBlockNames)
      if not attnFilesOk:
        echo "[resume] Commit marker dung nhung FP32 Attention sidecar THIEU/SAI shape -> khong tin ca bo sidecar."
      else:
        attnStoreTrusted = true
        if inProgress:
          echo &"[resume] FP32 Attention sidecar OK cho layer resume: step={committedStep}"
        else:
          echo &"[resume] Commit marker + FP32 Attention sidecar OK: step={committedStep}, seqLen={committedSeqLen}, batch={committedBatch}, pattern='{committedPattern}'"

    let loraAdamOk = restoreLoraAdamStates(adamStore, loState, lo, oldStep)
    var newAdamFilesOk = adamStoreAllAtStep(adamStore, m, newBlockNames, oldStep)
    if not newAdamFilesOk and hasInprogress and not fillInProgress:
      # Step train dang do: block G da update trong step nay co Adam oldStep+1 (dung),
      # truoc day bi coi la lech -> RESET Adam ca bo block G moi lan resume.
      newAdamFilesOk = adamStoreAllAtStepOrNext(adamStore, m, newBlockNames, oldStep)
    # If the FP32 sidecar is untrusted, trainStepGraftStreaming will rebuild
    # its weights from the lossy QLoRA checkpoint. Adam moments were computed
    # for the original FP32 masters, so carrying them into the reconstructed
    # weights silently mixes two different optimizer histories.
    let newAdamOk = newAdamFilesOk and
      (newBlockNames.len == 0 or attnStoreTrusted)
    if hasCommit and committedDatasetSig.len == 0:
      echo "[resume] Commit marker cu thieu datasetSig -> chi co the check tuong doi; day la checkpoint legacy."
    if CFG_INIT_MODE == "codestral" and hasCommit and committedTokenizerDir.len == 0:
      echo "[resume] Commit marker cu thieu tokenizerDir -> khong co xac minh duong dan tokenizer legacy."
    if loraAdamOk:
      echo &"[resume] LoRA Adam m/v OK -> tiep tuc tu step {oldStep}"
    else:
      echo "[resume] LoRA Adam m/v THIEU/LECH step -> reset LoRA optimizer, KHONG reset weight."
    if newAdamOk:
      echo &"[resume] New-block Adam m/v OK -> tiep tuc tu step {oldStep}"
    elif fillInProgress and attnStoreTrusted:
      # Block G da lap xong co Adam step 1 that, block chua lap step 0: lech step
      # la DUNG trong luc lap - reset se lam mat Adam cua block da lap.
      echo "[fill] Dang lap do: GIU Adam + FP32 sidecar block moi (block G da lap co state that tren SSD)."
    else:
      echo "[resume] New-block Adam m/v khong duoc tin (thieu/leach step hoac FP32 sidecar khong trusted) -> " &
           "RESET TOAN BO Adam cua block moi; weight tiep tuc tu checkpoint portable."
      resetAdamParameters(adamStore, m, newBlockNames)
  elif oldStep == 0:
    clearResumeInProgress(getCheckpointPath())

  # Per-layer in-progress files describe an *incomplete* training step. They
  # cannot be replayed safely (re-running backward would apply the optimizer a
  # second time), so a leftover directory is reported and dropped; resume
  # always falls back to the last committed `.bin` plus the SSD optimizer and
  # weight stores.
  # inprogress: MAC DINH GIU. Chi xoa SAU khi commit .bin thanh cong.
  # Lech step/cursor -> nap weight + FORCE commit (khong xoa, khong mat train).
  block inspectLayerProgress:
    var pStep, pLayer: int
    if not fillInProgress and checkpoint.readProgressCursor(getCheckpointPath(), pStep, pLayer):
      if pStep == oldStep:
        echo &"[checkpoint] Resume step {pStep} tu layer {pLayer}; giu inprogress, tiep backward."
      else:
        echo &"[checkpoint] inprogress step={pStep} layer={pLayer} (ckpt step={oldStep}) — " &
             "GIU file, se nap LoRA + FORCE commit neu can (khong xoa)."

  var layerProgressUsable = false
  var layerProgressStep, layerProgressLayer: int
  if checkpoint.readProgressCursor(getCheckpointPath(), layerProgressStep, layerProgressLayer):
    layerProgressUsable = true  # luon thu dung; khong clear khi lech
    echo &"[checkpoint] layer-progress usable: step={layerProgressStep} layer={layerProgressLayer}"

  if fillInProgress and not attnStoreTrusted:
    # Weight FP32 cua block G da lap chi nam trong sidecar; sidecar hong thi tien do
    # lap khong con nhat quan -> bo, lap lai tu dau (sidecar bootstrap lai tu .bin).
    echo "[fill] FP32 sidecar block G khong tin duoc -> BO tien do lap do, lap lai tu dau."
    checkpoint.clearLayerProgress(getCheckpointPath())
    removeActivationResume(FillStepIndex)
    fillInProgress = false
  if not attnStoreTrusted and oldStep > 0:
    # Sidecar FP32 co the dang o giua 1 step -> khong dung. Bootstrap tu checkpoint
    # QLoRA4 hien tai; an toan hon la trộn weight/optimizer khong cung version.
    attnStore.cleanupAttn()

  echo &"[finetune_lora] {lo.adapters.len} adapter LoRA, {newBlockNames.len} tham so block moi train full"
  echo &"[finetune_lora] scratch activation (spill-to-disk) se ghi vao: {CFG_SCRATCH_DIR}"
  echo &"[finetune_lora] Adam state swap SSD: {CFG_ADAM_DIR} (m/v/step doc 1 param -> GPU -> ghi SSD -> giai phong RAM)"

  var bpeTok: BPETokenizer
  var hfTok: HfTokenizer
  var usingHfTok = false

  case CFG_INIT_MODE
  of "codestral":
    echo "[finetune_lora] CFG_INIT_MODE=codestral -> BAT BUOC tokenizer THAT " &
         "(khong dung getOrTrainBPE - xem giai thich trong hf_tokenizer.nim), doc tu CFG_HF_DIR ..."
    hfTok = newHfTokenizer(CFG_HF_DIR)   # RAISE ro rang neu khong tim thay tokenizer that
    usingHfTok = true
    # SUA: truoc day chi `echo` canh bao roi TRAIN TIEP. Khong duoc.
    # Neu vocab lech thi tokenizer sinh ra id >= vocab model, va trong
    # softmaxCrossEntropy phep `probs[base + tgt]` voi base=(b*T+t)*V se roi
    # vao HANG THOI GIAN KE TIEP chu khong ra ngoai mang -> bounds check cua
    # Nim khong bat duoc -> gradient cong vao sai cho, IM LANG, loss van dep.
    # Ca run train nhieu gio se ra mot model vo nghia ma khong co dau hieu gi.
    if vocabSizeHf(hfTok) != CFG_CODESTRAL_VOCAB:
      raise newException(ValueError,
        &"[finetune_lora] vocab tokenizer that ({vocabSizeHf(hfTok)}) KHAC " &
        &"CFG_CODESTRAL_VOCAB ({CFG_CODESTRAL_VOCAB}). Sua CFG_CODESTRAL_VOCAB " &
        "cho khop (va sua ca trong infer_lora.nim) roi chay lai. KHONG train tiep.")
    echo &"[finetune_lora] tokenizer that OK, vocab={vocabSizeHf(hfTok)}"
  of "graft":
    echo "[finetune_lora] CFG_INIT_MODE=graft -> BPE tu train tren 1 mau nho dau tien cua " &
         "dataset (KHONG doc ca corpus vao RAM - chi doc it dong dau qua streamTextsForever " &
         "de co du liệu train BPE merges, dung tinh than cu la 'M' random-init tu dau)."
    var corpusForBpe = ""
    block bpeSample:
      for t in streamTextsForever(CFG_DATASETS):
        corpusForBpe.add(t); corpusForBpe.add('\n')
        if corpusForBpe.len > 300_000: break bpeSample
    bpeTok = getOrTrainBPE(corpusForBpe, CFG_BPE_MERGES)
    echo &"[finetune_lora] vocab_size={vocabSize(bpeTok)}"
  else: discard   # da raise o tren, khong toi day duoc

  # SUA (bug sidecar tokenizer): TRUOC DAY 2 dong writeFile(.hf_tokenizer_dir)/
  # saveBPE(.bpe.json) nam o CUOI file, chi chay SAU KHI `while epoch <
  # CFG_EPOCHS` thoat HET (mac dinh CFG_EPOCHS=2000) - neu ban dung train
  # giua chung (Ctrl+C) hay chi chay vai epoch roi qua infer_lora thu, .bin
  # da co (saveCheckpointLoraBin luu dinh ky moi CFG_CHECKPOINT_EVERY epoch)
  # nhung sidecar tokenizer thi CHUA TUNG duoc ghi -> infer_lora.nim khong
  # thay ".hf_tokenizer_dir", roi vao nhanh else tim ".bpe.json" (khong ton
  # tai voi checkpoint codestral) -> crash IOError "cannot open ...bpe.json".
  # GHI NGAY TAI DAY thay vi o cuoi: tokenizer (hfTok/bpeTok) da san sang xong
  # xuoi ngay tren, con vong lap train thi CHUA bat dau - nen MOI checkpoint
  # sinh ra sau nay (ke ca checkpoint dinh ky DAU TIEN) deu co sidecar di kem
  # tu dau, khong con phai cho train xong toan bo moi doc duoc.
  if usingHfTok:
    # KHONG co "save" cho tokenizer that (no la file goc cua HF, khong doi) -
    # chi ghi lai DUONG DAN de infer_lora.nim biet mo lai dung tokenizer nao
    # (xem loadBinaryLoraForInfer/loadCheckpointLoraForInfer trong infer_lora.nim:
    # doc "<ckpt>.hf_tokenizer_dir" neu co).
    writeFile(getCheckpointPath() & ".hf_tokenizer_dir", CFG_HF_DIR)
    echo &"[finetune_lora] Da ghi sidecar tokenizer: {getCheckpointPath()}.hf_tokenizer_dir -> {CFG_HF_DIR}"
  else:
    saveBPE(bpeTok, getCheckpointPath() & ".bpe.json")
    echo &"[finetune_lora] Da ghi sidecar tokenizer: {getCheckpointPath()}.bpe.json"

  # Truoc day threshold la `< 128` nhung default CHINH XAC bang 128 - nen
  # canh bao khong bao gio chay. Nang len 192 de no thuc su xuat hien khi
  # cua so con ngan hon p50~130 token cua dolly, nhac nguoi dung tang len
  # 256 (BYBY_SEQ_LEN=256) neu may cho phep.
  if CFG_SEQ_LEN < 192:
    echo &"[finetune_lora] LUU Y: CFG_SEQ_LEN={CFG_SEQ_LEN}. Do tren " &
         "databricks-dolly-15k.jsonl: vi du dai p50~130 token, p90~490, p99~1500. " &
         &"Voi cua so {CFG_SEQ_LEN} token thi PHAN LON cua so train khong chua ca " &
         "'[INST]' lan '</s>', bat dau tu giua cau voi SSM state = 0 - lech han " &
         "voi luc infer (prompt day du + state da prefill). Nang len >= 256 khi " &
         "RAM/VRAM cho phep."

  var batcher = newStreamingBatcher(CFG_DATASETS, CFG_SEQ_LEN, CFG_BATCH_SIZE,
                                     bpeTok, hfTok, usingHfTok)
  dlog "batcher da tao xong, bat dau vong lap train"

  var epoch = oldStep   # day la training step, KHONG phai full pass dataset
  if epoch > 0:
    batcher.skipSteps(epoch)
  block:
    var cs, cl: int
    if not fillInProgress and checkpoint.readProgressCursor(getCheckpointPath(), cs, cl):
      echo &"[resume-layer] phat hien {getCheckpointPath()}.inprogress cursor step={cs} layer={cl}"
      # SUA (bug "102/128 adapter B=0"): truoc day nhanh nay FORCE commit step
      # cs+1 bat ke cursor dung o layer nao. Step bi ngat (Ctrl+C / crash) sau
      # backward layer 51 -> commit nhu step DA XONG, block 0..50 khong bao gio
      # duoc backward (B=0, Adam m=v=0 nhung step bi ep =1), va voi
      # CFG_EPOCHS=1 vong train khong chay lai nua. Chi commit khi backward da
      # toi block 0; con lai thi de vong train chay tiep step cs - trainStep-
      # GraftStreaming tu nap dX + activation + LoRA da spill va backward tiep.
      var backwardDone, canResumeStep: bool
      let stepDesc = layerProgressStepState(cs, cl, m.blocks.len, backwardDone, canResumeStep)
      # epoch < CFG_EPOCHS: vong train phai THUC SU chay lai step cs, neu khong
      # LoRA dang do nap duoi day se bi checkpoint-final commit nhu step xong.
      canResumeStep = canResumeStep and cs == epoch and epoch < CFG_EPOCHS
      let forcePartial = getEnv("BYBY_FORCE_PARTIAL_COMMIT", "0") != "0"
      if cs != epoch:
        # Tien do cua step KHAC step checkpoint (file sot lai): ap vao se tron LoRA
        # cua 2 step khac nhau roi commit sai step.
        raise newException(IOError,
          &"[resume-layer] inprogress la cua step {cs} nhung checkpoint dang o step {epoch} - " &
          &"KHONG ap/commit (tron 2 step). Xoa {getCheckpointPath()}.inprogress neu khong can.")
      if not backwardDone and canResumeStep and not forcePartial:
        echo &"[resume-layer] step {cs} CHUA XONG ({stepDesc}); KHONG commit - " &
             &"train tiep backward tu block {cl - 1} xuong 0."
      elif not backwardDone and not forcePartial:
        raise newException(IOError,
          &"[resume-layer] step {cs} chua backward xong ({stepDesc}, ckpt step={epoch}, " &
          &"CFG_EPOCHS={CFG_EPOCHS}) nhung khong resume duoc (thieu dX/activation, lech step, hoac " &
          "CFG_EPOCHS khong con step de chay). KHONG commit step nay nhu da xong " &
          &"(block {cl - 1}..0 se khong duoc train). Xoa {getCheckpointPath()}.inprogress " &
          "de train lai step, hoac BYBY_FORCE_PARTIAL_COMMIT=1 neu chap nhan commit dang do.")
      # Nap toan bo LoRA A/B da spill trong inprogress vao RAM
      var nApplied = 0
      for li in countdown(m.blocks.len - 1, 0):
        let tensors = checkpoint.loadLayerProgress(getCheckpointPath(), cs, li)
        if tensors.len > 0:
          nApplied += applyLayerProgressToLora(lo, tensors)
      echo &"[resume-layer] nap {nApplied} tensor LoRA tu inprogress step={cs}"
      if not backwardDone and forcePartial:
        stderr.writeLine &"[resume-layer] CANH BAO: BYBY_FORCE_PARTIAL_COMMIT=1 - commit step {cs} " &
          &"dang do, block {cl - 1}..0 KHONG duoc train trong step nay."
      # Chi commit ngay khi step inprogress da xong het layer (cl==0) hoac user force partial
      let autoCommit = getEnv("BYBY_FORCE_CHECKPOINT", "1") != "0" and
        (backwardDone or forcePartial)
      if autoCommit and nApplied > 0:
        let commitStep = max(1, cs + 1)  # step train cs xong -> marker = cs+1 (giong inc epoch)
        # Neu cursor layer==0 va resume "backward_done" co the coi step cs da xong
        syncLoraAdamFromStore(adamStore, loState, lo)
        forceLoraAdamToStep(loState, lo, commitStep)
        saveLoraAdamStates(adamStore, loState)
        saveCheckpointLoraBin(m, lo, getCheckpointPath(), commitStep, newBlockNames, attnStore)
        writeResumeCommit(getCheckpointPath(), commitStep, CFG_SEQ_LEN, CFG_BATCH_SIZE, m.pattern,
          currentDatasetSig, (if CFG_INIT_MODE == "codestral": CFG_HF_DIR else: ""))
        clearResumeInProgress(getCheckpointPath())
        checkpoint.clearLayerProgress(getCheckpointPath())  # chi xoa SAU commit OK
        try: removeActivationResume(cs)
        except CatchableError: discard
        echo &"[resume-layer] FORCE commit OK step={commitStep} tu inprogress — tiep train neu con epoch."
        # batcher da skipSteps(epoch) o tren; chi bo them so step vua commit.
        if commitStep > epoch:
          batcher.skipSteps(commitStep - epoch)
        epoch = commitStep
  # ── COMMIT-ONLY: khong train; nap LoRA tu .inprogress (neu con) + force Adam step + ghi .bin ──
  if getEnv("BYBY_COMMIT_ONLY", "0") != "0":
    if fillInProgress:
      raise newException(ValueError,
        "[COMMIT_ONLY] Dang lap do block LoRA trong (" & fillMarkerPath(getCheckpointPath()) &
        ") - chay train binh thuong (khong BYBY_COMMIT_ONLY) de lap tiep, KHONG commit dang do.")
    let ckpt = getCheckpointPath()
    let progDir = checkpoint.progressDir(ckpt)
    echo &"[COMMIT_ONLY] Tim LoRA weight tai: {progDir}"
    echo &"[COMMIT_ONLY] Adam dir (chi m/v, KHONG phai weight A/B): {CFG_ADAM_DIR}"
    if dirExists(progDir):
      var nFiles = 0
      for kind, path in walkDir(progDir, relative = false):
        echo &"[COMMIT_ONLY]   {path}"
        inc nFiles
        if nFiles > 40:
          echo "[COMMIT_ONLY]   ..."
          break
      if nFiles == 0:
        echo "[COMMIT_ONLY] Thu muc inprogress rong."
    else:
      echo &"[COMMIT_ONLY] KHONG co thu muc {progDir}"
    var nApplied = 0
    var stepsTried: seq[int] = @[]
    var cs, cl: int
    let hasCursorCO = checkpoint.readProgressCursor(ckpt, cs, cl)
    if hasCursorCO:
      echo &"[COMMIT_ONLY] cursor step={cs} layer={cl}"
      if cs != epoch:
        raise newException(ValueError,
          &"[COMMIT_ONLY] inprogress la cua step {cs} nhung checkpoint dang o step {epoch} - " &
          "KHONG commit (tron 2 step).")
      stepsTried.add(cs)   # CHI step cua cursor; thu muc step-* khac la file sot, khong ap
      var backwardDone, canResumeStep: bool
      let stepDesc = layerProgressStepState(cs, cl, m.blocks.len, backwardDone, canResumeStep)
      if not backwardDone and getEnv("BYBY_FORCE_PARTIAL_COMMIT", "0") == "0":
        raise newException(ValueError,
          &"[COMMIT_ONLY] step {cs} chua backward xong ({stepDesc}). KHONG commit dang do " &
          "(block chua backward se bi coi nhu da train). " &
          (if canResumeStep: "Chay train binh thuong (khong BYBY_COMMIT_ONLY) de backward tiep. "
           else: "") &
          "BYBY_FORCE_PARTIAL_COMMIT=1 neu that su muon commit dang do.")
    if dirExists(progDir):
      for kind, path in walkDir(progDir, relative = false):
        if kind != pcDir: continue
        let base = path.extractFilename
        if base.startsWith("step-") and base != &"step-{cs}":
          echo &"[COMMIT_ONLY] BO QUA {path} (khong phai step cua cursor)"
    for st in stepsTried:
      for li in countdown(m.blocks.len - 1, 0):
        let tensors = checkpoint.loadLayerProgress(ckpt, st, li)
        if tensors.len == 0: continue
        nApplied += applyLayerProgressToLora(lo, tensors)
    echo &"[COMMIT_ONLY] Da nap {nApplied} tensor LoRA tu layer-progress."
    if nApplied == 0 and not fileExists(ckpt) and not fileExists(ckpt & ".prev"):
      raise newException(ValueError,
        "[COMMIT_ONLY] Khong con layer-progress (.bin/.idx theo layer). " &
        "SSD Adam (.nimformers_adam) CHI luu m/v optimizer — KHONG luu weight LoRA A/B. " &
        "Binary cu da clearLayerProgress truoc khi commit fail -> weight step 0 mat. " &
        "Chi con cach train lai 1 step voi binary moi (BYBY_FORCE_CHECKPOINT=1 se commit).")
    # Step cs da backward xong -> commit thanh step cs+1 (giong inc epoch); khong co
    # tien do thi chi ghi lai checkpoint hien tai o dung step cua no.
    let commitStep = if hasCursorCO and nApplied > 0: cs + 1 else: max(1, epoch)
    syncLoraAdamFromStore(adamStore, loState, lo)
    forceLoraAdamToStep(loState, lo, commitStep)
    saveLoraAdamStates(adamStore, loState)
    if newBlockNames.len > 0:
      forceAdamStoreToStep(adamStore, m, newBlockNames, commitStep)
    saveCheckpointLoraBin(m, lo, ckpt, commitStep, newBlockNames, attnStore)
    writeResumeCommit(ckpt, commitStep, CFG_SEQ_LEN, CFG_BATCH_SIZE, m.pattern, currentDatasetSig,
      (if CFG_INIT_MODE == "codestral": CFG_HF_DIR else: ""))
    clearResumeInProgress(ckpt)
    checkpoint.clearLayerProgress(ckpt)
    try: removeActivationResume(0)
    except CatchableError: discard
    if commitStep > 1:
      try: removeActivationResume(commitStep - 1)
      except CatchableError: discard
    echo &"[COMMIT_ONLY] OK — da ghi checkpoint step={commitStep}: {ckpt}"
    quit(0)

  # BAT BUOC lap block LoRA bi bo trong truoc khi train tiep (khong can tool rieng).
  block fillHoles:
    var cs, cl: int
    if not resumed or epoch != oldStep:
      break fillHoles
    if not fillInProgress and checkpoint.readProgressCursor(getCheckpointPath(), cs, cl):
      break fillHoles   # dang co 1 step train thuong do dang: khoi [resume-layer] o tren lo
    if fillUntrainedLoraBlocks(m, lo, loState, adamStore, attnStore, newBlockNames,
                               bpeTok, hfTok, usingHfTok, ctx, epoch):
      if not loraAdamStateAllAtStep(loState, lo, epoch):
        forceLoraAdamToStep(loState, lo, epoch)
      if newBlockNames.len > 0 and not adamStoreAllAtStep(adamStore, m, newBlockNames, epoch):
        forceAdamStoreToStep(adamStore, m, newBlockNames, epoch)
      let ckpt = getCheckpointPath()
      # Backup chi la phu: file co the da bi xoa/doi ten trong luc lap (nhieu gio) -
      # KHONG duoc de loi backup lam mat ket qua lap (bug: copyFile .bin da bi xoa ->
      # OSError -> chet truoc khi ghi .bin).
      for suffix in ["", ".resume.commit"]:
        if fileExists(ckpt & suffix):
          try: copyFile(ckpt & suffix, ckpt & suffix & ".before_fill")
          except OSError as e:
            stderr.writeLine &"[fill] CANH BAO: khong backup duoc {ckpt}{suffix}: {e.msg} - van ghi ket qua lap."
        else:
          stderr.writeLine &"[fill] CANH BAO: {ckpt}{suffix} khong con tren dia - bo qua backup, van ghi ket qua lap."
      saveCheckpointLoraBin(m, lo, ckpt, epoch, newBlockNames, attnStore)
      saveLoraAdamStates(adamStore, loState)
      writeResumeCommit(ckpt, epoch, CFG_SEQ_LEN, CFG_BATCH_SIZE, m.pattern, currentDatasetSig,
        (if CFG_INIT_MODE == "codestral": CFG_HF_DIR else: ""))
      # Chi xoa tien do lap SAU khi .bin + Adam + commit marker da ghi xong.
      checkpoint.clearLayerProgress(ckpt)
      removeActivationResume(FillStepIndex)
      fillInProgress = false
      echo &"[fill] Da ghi {ckpt} step={epoch} (backup: {ckpt}.before_fill)"

  # TU DONG kiem update step 0 (sau khi lap): block co update step 0 sai (vd tinh qua
  # lm_head random) -> tinh lai; ghi co vao .bin de lan sau khong kiem lai.
  block step0Audit:
    var cs, cl: int
    if not resumed or epoch != oldStep:
      break step0Audit
    if not step0RepairInProgress(getCheckpointPath()) and
       checkpoint.readProgressCursor(getCheckpointPath(), cs, cl):
      break step0Audit   # dang co 1 step train thuong do dang
    if auditStep0LoraBlocks(m, lo, loState, adamStore, attnStore,
                            bpeTok, hfTok, usingHfTok, ctx, epoch):
      if not loraAdamStateAllAtStep(loState, lo, epoch):
        forceLoraAdamToStep(loState, lo, epoch)
      if newBlockNames.len > 0 and not adamStoreAllAtStep(adamStore, m, newBlockNames, epoch):
        forceAdamStoreToStep(adamStore, m, newBlockNames, epoch)
      let ckpt = getCheckpointPath()
      # Giu .bin truoc khi sua (hard link, O(1)): .prev KHONG giu duoc vi final save ngay
      # sau se ghi de no. Loi backup khong duoc lam mat ket qua kiem.
      let backup = ckpt & ".before_audit"
      try:
        if fileExists(backup): removeFile(backup)
        if fileExists(ckpt) and cLink(ckpt.cstring, backup.cstring) != 0:
          copyFile(ckpt, backup)
      except OSError as e:
        stderr.writeLine &"[audit] CANH BAO: khong backup duoc {ckpt}: {e.msg} - van ghi ket qua."
      saveCheckpointLoraBin(m, lo, ckpt, epoch, newBlockNames, attnStore)
      saveLoraAdamStates(adamStore, loState)
      writeResumeCommit(ckpt, epoch, CFG_SEQ_LEN, CFG_BATCH_SIZE, m.pattern, currentDatasetSig,
        (if CFG_INIT_MODE == "codestral": CFG_HF_DIR else: ""))
      checkpoint.clearLayerProgress(ckpt)
      removeActivationResume(FillStepIndex)
      fillInProgress = false
      echo &"[audit] Da ghi {ckpt} step={epoch} ({Step0AuditMetaKey}=ok; ban truoc: {backup})"

  var runningLoss = 0'f32
  var prevStepLoss = -1'f32
  while epoch < CFG_EPOCHS:
    markResumeInProgress(getCheckpointPath(), epoch + 1)
    # `idsB`/`tgtB` chi song trong 1 vong lap - het scope la GC thu hoi
    # NGAY, khong tich tu qua cac buoc (khac ban cu giu nguyen ca seq
    # `batches` tu dau den cuoi qua trinh train).
    dlog &"epoch {epoch}: goi nextBatch()"
    let (idsB, tgtRawB, tgtMaskB) = batcher.nextBatch()
    dlog &"epoch {epoch}: nextBatch xong, {idsB.len} sample, goi trainStep"
    if idsB.len == 0:
      echo "[finetune_lora] Dataset rong (khong doc duoc dong nao) - dung."
      quit(1)
    # LR warmup cho block G: scale tu ~0 -> 1.0 trong CFG_G_LR_WARMUP_STEPS buoc.
    let gLrScale = min(1.0'f32, (float32(epoch) + 1.0'f32) / float32(CFG_G_LR_WARMUP_STEPS))
    let loss = if lo.quantBase.len > 0:   # SUA: dung lo.quantBase (nguon su that DUY NHAT
                                            # con lai sau khi da .clear() quantResidentWeights
                                            # o tren) thay vi quantResidentWeights (gio LUON
                                            # rong sau clear - kiem tra no o day se SAI, luon
                                            # roi vao nhanh trainStepGraft khong-streaming).
                 # targets DA MASK ranh gioi -> softmaxCrossEntropyMasked (local)
                 # stepIndex = epoch: cho trainStepGraftStreaming biet day la step nao de
                 # ghi per-layer in-progress (<ckpt>.inprogress/step-<n>/layer-<i>.*).
                 trainStepGraftStreaming(m, lo, loState, adamStore, attnStore,
                   CFG_NEWBLOCK_LR, idsB, tgtMaskB, ctx, gLrScale, stepIndex = epoch)
               else:
                 # duong nay di qua jamba_hybrid.computeLossAndGrads ->
                 # test_nimformer.softmaxCrossEntropy GOC (khong hieu TargetIgnore)
                 # nen PHAI dua targets THO. Duong nay khong duoc mask ranh gioi.
                 trainStepGraft(m, lo, loState, adamStore, CFG_NEWBLOCK_LR, idsB, tgtRawB, ctx)
    runningLoss += loss
    if gStopRequested and not gTrainStepFullyDone:
      # Step dang do: resume/layer progress da tren SSD — KHONG inc epoch, KHONG commit full ckpt.
      stderr.writeLine "[finetune_lora] Thoat an toan (step chua xong). Chay lai make finetune-mamba de resume layer."
      quit(130)
    if epoch == 0 and gTrainStepFullyDone and gLastStepForwardRan:
      # Step 0: loss + gradient tinh tron trong process nay (lm_head da bind, preflight)
      # -> .bin step 1 ghi co xac nhan, lan sau khong can kiem step 0 lai.
      gStep0Verified = true
    inc epoch
    if gStopRequested:
      # Step da xong (backward den layer 0 hoac break sau layer 0). Thu commit roi thoat.
      stderr.writeLine "[finetune_lora] Ctrl+C sau khi step xong — thu commit checkpoint roi thoat."
      syncLoraAdamFromStore(adamStore, loState, lo)
      if loraAdamStateAllAtStep(loState, lo, epoch):
        if newBlockNames.len == 0 or adamStoreAllAtStep(adamStore, m, newBlockNames, epoch):
          if not (CFG_INIT_MODE == "codestral" and newBlockNames.len > 0 and not attnStoreAllValid(attnStore, m, newBlockNames)):
            saveCheckpointLoraBin(m, lo, getCheckpointPath(), epoch, newBlockNames, attnStore)
            saveLoraAdamStates(adamStore, loState)
            writeResumeCommit(getCheckpointPath(), epoch, CFG_SEQ_LEN, CFG_BATCH_SIZE, m.pattern, currentDatasetSig, (if CFG_INIT_MODE == "codestral": CFG_HF_DIR else: ""))
            clearResumeInProgress(getCheckpointPath())
            checkpoint.clearLayerProgress(getCheckpointPath())
            removeActivationResume(epoch - 1)
            stderr.writeLine &"[finetune_lora] Da commit checkpoint step={epoch} truoc khi thoat."
            quit(130)
      stderr.writeLine "[finetune_lora] Khong commit duoc (Adam/sidecar lech); resume layer van con tren SSD. Thoat."
      quit(130)

    # DON DEP sau MOI batch (giong reuse-allocator tren ESP32: chu dong
    # thu hoi rac ngay, khong de tich tu roi mem-spike bat ngo o 1 buoc
    # xa sau). GC_fullCollect nang hon GC_step nhung o day uu tien AN TOAN
    # RAM (may yeu) hon toc do - moi buoc train GPU von da la phan nang
    # nhat, chi phi GC full o day khong dang ke so voi risk OOM.
    GC_fullCollect()

    if epoch mod CFG_LOG_EVERY == 0:
      let maskPct = if batcher.nEmitted > 0:
                      100.0 * float(batcher.nMasked) / float(batcher.nEmitted)
                    else: 0.0
      echo &"[epoch {epoch}/{CFG_EPOCHS}] loss={runningLoss / float32(CFG_LOG_EVERY):.4f} " &
           &"(pending_buf={batcher.pending.len} tok, mask ranh gioi doc={maskPct:.2f}%)"
      runningLoss = 0'f32

    if epoch mod CFG_CHECKPOINT_EVERY == 0:
      # PREFLIGHT TRUOC COMMIT: weight/optimizer/step phai cung mot training step.
      # Streaming da spill Adam LoRA len SSD theo layer — sync RAM tu disk truoc khi check.
      syncLoraAdamFromStore(adamStore, loState, lo)
      # Mac dinh: step train da xong -> KHONG de mat weight vi 1 so Adam file step=0.
      # BYBY_STRICT_ADAM_STEP=1 moi raise. BYBY_FORCE_CHECKPOINT=0 de tat force.
      let strictAdam = getEnv("BYBY_STRICT_ADAM_STEP", "0") != "0"
      let forceCkpt = getEnv("BYBY_FORCE_CHECKPOINT", "1") != "0"
      if not loraAdamStateAllAtStep(loState, lo, epoch):
        if forceCkpt and not strictAdam:
          stderr.writeLine &"[checkpoint] FORCE: ep LoRA Adam step -> {epoch} roi commit (tranh mat step da train)."
          forceLoraAdamToStep(loState, lo, epoch)
          saveLoraAdamStates(adamStore, loState)
        else:
          raise newException(ValueError,
            &"[checkpoint] LoRA Adam step khong dong bo voi training step {epoch}. KHONG commit checkpoint.")
      if newBlockNames.len > 0 and not adamStoreAllAtStep(adamStore, m, newBlockNames, epoch):
        if forceCkpt and not strictAdam:
          stderr.writeLine &"[checkpoint] FORCE: ep new-block Adam step -> {epoch} roi commit"
          forceAdamStoreToStep(adamStore, m, newBlockNames, epoch)
        else:
          raise newException(ValueError,
            &"[checkpoint] New-block Adam state khong dong bo voi training step {epoch}. KHONG commit checkpoint.")
      if CFG_INIT_MODE == "codestral" and newBlockNames.len > 0 and not attnStoreAllValid(attnStore, m, newBlockNames):
        if forceCkpt and not strictAdam:
          stderr.writeLine &"[checkpoint] FORCE: bo qua FP32 Attention sidecar check step={epoch}"
        else:
          raise newException(ValueError,
            &"[checkpoint] FP32 Attention sidecar khong day du tai step {epoch}. KHONG commit checkpoint.")
      # THU TU COMMIT: checkpoint weight -> LoRA Adam -> commit marker -> xoa in-progress.
      saveCheckpointLoraBin(m, lo, getCheckpointPath(), epoch, newBlockNames, attnStore)
      saveLoraAdamStates(adamStore, loState)
      writeResumeCommit(getCheckpointPath(), epoch, CFG_SEQ_LEN, CFG_BATCH_SIZE, m.pattern, currentDatasetSig, (if CFG_INIT_MODE == "codestral": CFG_HF_DIR else: ""))
      clearResumeInProgress(getCheckpointPath())
      # Chi xoa layer progress SAU khi .bin commit OK.
      checkpoint.clearLayerProgress(getCheckpointPath())
      removeActivationResume(epoch - 1)
      GC_fullCollect()

    # BAT BUOC: loss tang so voi step truoc -> DUNG train de kiem tra (step nay
    # da duoc commit o tren / se duoc commit o checkpoint-final ben duoi).
    if prevStepLoss >= 0'f32 and loss > prevStepLoss:
      stderr.writeLine &"[finetune_lora] DUNG: loss TANG {prevStepLoss:.4f} -> {loss:.4f} o step {epoch}. " &
        "Kiem tra du lieu/LR/forward truoc khi train tiep."
      break
    prevStepLoss = loss

  if epoch > 0:
    let strictFinal = getEnv("BYBY_STRICT_ADAM_STEP", "0") != "0"
    let forceFinal = getEnv("BYBY_FORCE_CHECKPOINT", "1") != "0"
    syncLoraAdamFromStore(adamStore, loState, lo)
    if not loraAdamStateAllAtStep(loState, lo, epoch):
      if forceFinal and not strictFinal:
        stderr.writeLine &"[checkpoint-final] FORCE: ep LoRA Adam step -> {epoch}"
        forceLoraAdamToStep(loState, lo, epoch)
        saveLoraAdamStates(adamStore, loState)
      else:
        raise newException(ValueError,
          &"[checkpoint-final] LoRA Adam step khong dong bo voi training step {epoch}. KHONG commit checkpoint.")
    if newBlockNames.len > 0 and not adamStoreAllAtStep(adamStore, m, newBlockNames, epoch):
      if forceFinal and not strictFinal:
        stderr.writeLine &"[checkpoint-final] FORCE: ep new-block Adam step -> {epoch}"
        forceAdamStoreToStep(adamStore, m, newBlockNames, epoch)
      else:
        raise newException(ValueError,
          &"[checkpoint-final] New-block Adam state khong dong bo voi training step {epoch}. KHONG commit checkpoint.")
    if CFG_INIT_MODE == "codestral" and newBlockNames.len > 0 and not attnStoreAllValid(attnStore, m, newBlockNames):
      if forceFinal and not strictFinal:
        stderr.writeLine &"[checkpoint-final] FORCE: bo qua FP32 Attention sidecar check step={epoch}"
      else:
        raise newException(ValueError,
          &"[checkpoint-final] FP32 Attention sidecar khong day du tai step {epoch}. KHONG commit checkpoint.")
  saveCheckpointLoraBin(m, lo, getCheckpointPath(), epoch, newBlockNames, attnStore)
  saveLoraAdamStates(adamStore, loState)
  writeResumeCommit(getCheckpointPath(), epoch, CFG_SEQ_LEN, CFG_BATCH_SIZE, m.pattern, currentDatasetSig, (if CFG_INIT_MODE == "codestral": CFG_HF_DIR else: ""))
  clearResumeInProgress(getCheckpointPath())
  # The `.bin` + commit marker for the final step are durable now, so the
  # per-layer in-progress files and the activation scratch for it are no
  # longer needed either.
  checkpoint.clearLayerProgress(getCheckpointPath())
  removeActivationResume(epoch - 1)
  if not CFG_KEEP_RESUME_STATE:
    adamStore.cleanup()
    attnStore.cleanupAttn()
    echo &"[finetune_lora] Da don Adam disk cache: {CFG_ADAM_DIR}"
    echo &"[finetune_lora] Da don Attn-weight disk cache: {CFG_ATTN_WEIGHTS_DIR}"
  else:
    echo "[finetune_lora] GIU resume state tren SSD: Adam + FP32 Attention. Khong can train lai khi resume."
  # Sidecar tokenizer (.hf_tokenizer_dir / .bpe.json) da duoc ghi SOM, ngay
  # sau khi tokenizer san sang, TRUOC vong lap train (xem phia tren) - KHONG
  # ghi lai o day nua (do la nguyen nhan gay bug "phai train xong moi co").
  echo "[finetune_lora] XONG."
