## infer_lora.nim
## Inference for checkpoints fine-tuned with finetune_lora.nim (grafted Jamba base
## + LoRA adapters stored separately in <path>.lora.json).
##
## 100% GPU: the base stays compressed (fp4-KM / int4-asymmetric), LoRA A/B run on the GPU
## (micro-VM bytecode JIT, no hand-written MSL), 1 command buffer / token.
## NO fp32 merge, NO CPU/fp32 fallback.

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
import fp4_gemv_gpu     # gFp4CacheGpu — sequenceLinearReady for the FP4-KM prefill
import hf_tokenizer      # the REAL tokenizer - must be reloaded when the checkpoint comes from CFG_INIT_MODE=="codestral"
import test_nimformer   # BPETokenizer: loadBPE/encodeBPE/decodeBPEStr
import load_codestral_into_jamba   # loadCodestralQuantResidentFromNimq - rebuilds the base from .nimq
import quantize_lora_bin_4bit

var gMergeLoraExplicit = false   ## BYBY_MERGE_LORA set by hand: a failed merge is then fatal.

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
  # Merging LoRA into the resident base is the default: the decode kernels drop
  # the A/B fusion and the answers match a merged checkpoint. BYBY_MERGE_LORA=0
  # (formers.n: --no-merge-lora) keeps the adapters separate.
  gMergeLoraExplicit = existsEnv("BYBY_MERGE_LORA")
  var mergeMode=getEnv("BYBY_MERGE_LORA","inplace")
  if mergeMode notin ["0","1","inplace"]:
    raise newException(ValueError,"BYBY_MERGE_LORA must be 0, 1 or inplace (GPU); use 0 to retain separate LoRA")
  # The merge rewrites the packed weight the GPU already holds, so it needs the
  # mapped checkpoint and the per-parameter dispatch. Asked for explicitly, a
  # conflict is an error; on by default it simply steps aside.
  if mergeMode!="0" and getEnv("BYBY_MODEL_MMAP","1")=="0":
    if gMergeLoraExplicit:
      raise newException(ValueError,"bounded in-place merge requires BYBY_MODEL_MMAP=1")
    mergeMode="0"
    echo "[infer] BYBY_MODEL_MMAP=0 keeps the LoRA adapters separate (no in-place merge)."
  if mergeMode!="0" and getEnv("BYBY_ONE_DISPATCH","0")!="0":
    if gMergeLoraExplicit:
      echo "[infer] In-place merge selects BYBY_ONE_DISPATCH=0 to reuse packed weight buffers."
      putEnv("BYBY_ONE_DISPATCH","0")
    else:
      mergeMode="0"
      echo "[infer] BYBY_ONE_DISPATCH=1 keeps the LoRA adapters separate (BYBY_MERGE_LORA=1 merges anyway)."
  putEnv("BYBY_MERGE_LORA",mergeMode)
when defined(macosx):
  # QOS_CLASS_USER_INITIATED (0x19): CLI inference directly serves a
  # user request. This is a CPU thread priority; Metal has no equivalent
  # public API to force the GPU scheduler to prioritise command buffers.
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

# Reads the process's own REAL peak RSS (no child process fork -
# see the full explanation in finetune_lora.nim; it originally used
# startProcess("/bin/ps",...) which ITSELF added RAM pressure while measuring,
# skewing the very data being investigated). Used to tell whether an OOM
# kill comes from THIS process growing over time or
# from outside RAM pressure (other apps, the system) at some moment.
proc rssMB(): string =
  # getrusage reports bytes on macOS, KiB on Linux. Normalize before printing.
  $(peakResidentBytes() div (1024*1024)) & "MiB"

const
  CFG_CHECKPOINT_PATH* = "test/codestral/models/jamba_parallel_lora.bin"   # parallel Mamba+Linear++ checkpoint -
                                             # NO real .bin is read/written any more.
  CFG_MAX_NEW_TOKENS* = 60
  CFG_TEMPERATURE* = 0.7'f32
  CFG_SEED* = 1234'i64
  # Must EXACTLY match the constant of the same name in finetune_lora.nim (used
  # to rebuild the model architecture from .nimq - see loadCheckpointLora there).
  CFG_CODESTRAL_VOCAB = 32768
  CFG_CODESTRAL_HIDDEN = 4096
  CFG_CODESTRAL_NHEADS = 128
  CFG_CODESTRAL_DSTATE = 128
  CFG_CODESTRAL_NGROUPS = 8
  CFG_CODESTRAL_CONVKERNEL = 4
  CFG_CODESTRAL_EXPAND = 2
  # FALLBACK (NEW): used when <path>.lora.json is NOT found - infers DIRECTLY
  # from the pretrain base, WITHOUT the LoRA/grafted Attention blocks (since nothing
  # was trained to load into them). Pattern = ALL 'R' (64 plain
  # Mamba2Real blocks, ALL 64 real layers of Codestral-Mamba-7B - see the
  # CFG_CODESTRAL_PATTERN comment in finetune_lora.nim: "the original Codestral
  # has NO Attention blocks at all" - this IS the ORIGINAL architecture, not
  # the hybrid 'RRRRA' architecture used only for fine-tuning).
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
  ## NO LoRA checkpoint found - instead of erroring/stopping, infers DIRECTLY
  ## on the pretrain Mamba-Codestral-7B base (dequantized through the LUT from .nimq).
  echo &"[infer] No LoRA checkpoint found ({CFG_CHECKPOINT_PATH}.lora.json) - " &
       &"running PLAIN Mamba from the pretrain base ({nimqPath}), pattern='{CFG_PURE_MAMBA_PATTERN}' (64 'R' blocks, no Attention)."
  var m = newNimformerModelJamba(CFG_CODESTRAL_VOCAB, CFG_CODESTRAL_HIDDEN, CFG_CODESTRAL_NHEADS,
                                  ffMult = 4, pattern = CFG_PURE_MAMBA_PATTERN, useSession = true,
                                  mambaRealDState = CFG_CODESTRAL_DSTATE,
                                  mambaRealNGroups = CFG_CODESTRAL_NGROUPS,
                                  mambaRealConvKernel = CFG_CODESTRAL_CONVKERNEL,
                                  mambaRealExpand = CFG_CODESTRAL_EXPAND, quantResident = true)
  let quantResident = loadCodestralQuantResidentFromNimq(m, nimqPath)
  var lo: JambaLora
  lo.adapters = initTable[string, LoraAdapter]()   # EMPTY - no LoRA at all
  lo.backup = initTable[string, seq[float32]]()
  lo.quantBase = quantResident
  # IndexDefect fix: bind quantBase → Linear.weightQ/useQuant BEFORE generate.
  # (quantResident=true has cleared weight.data; without this step → transpose(@[]) crash)
  bindQuantBaseToLinears(m, lo)
  # 2nd fix: NO more eager dequant of all quant-resident weights.
  # forwardStepJambaWithLora (lora.nim) does mergeInOnePlain / restoreBaseOnePlain
  # PER BLOCK when needed (on the fly) - only ~2 large fp32 tensors live at once,
  # in the spirit of trainStepGraftStreaming. The fallback path (forwardBatchSession)
  # still needs a full dequant in generate() when the pattern is not pure 'R'.
  echo &"[infer] quant-resident = {quantResident.len} parameters (kept compressed, dequantized on the fly during generate)"
  echo &"[infer] Real tokenizer from {hfDir} ..."
  let hf = newHfTokenizer(hfDir)
  # This used to only echo the vocab and carry on. If the tokenizer vocab differs
  # from the model vocab, every token id >= the model vocab looks up the WRONG embedding
  # row (or goes out of range), and logits/sampling are off too - 100% garbage. Stop
  # right away, no guessing.
  if vocabSizeHf(hf) != CFG_CODESTRAL_VOCAB:
    raise newException(ValueError,
      &"[infer] vocab tokenizer ({vocabSizeHf(hf)}) KHAC vocab model " &
      &"({CFG_CODESTRAL_VOCAB}). Use this checkpoint's OWN tokenizer, " &
      "or fix CFG_CODESTRAL_VOCAB to match. Not continuing.")
  let tok = AnyTokenizer(kind: atkHf, hf: hf)
  echo &"[infer] Loaded the plain Mamba base (NO LoRA) from {nimqPath}, vocab={vocabSizeHf(hf)}"
  result = (m, lo, tok)

proc encodeAny(t: AnyTokenizer, text: string): seq[int] =
  case t.kind
  of atkBpe: encodeBPE(t.bpe, text)
  of atkHf: encodeHf(t.hf, text)

proc encodeAnyNoBos(t: AnyTokenizer, text: string): seq[int] =
  ## Encode WITHOUT inserting BOS - used to measure the ids of formatting tokens
  ## ("\n", " ", indentation) when using the repeat-penalty exemption table.
  case t.kind
  of atkBpe: encodeBPE(t.bpe, text)
  of atkHf: encodeHf(t.hf, text, addEos = false, addBos = false)

proc decodeAny(t: AnyTokenizer, ids: seq[int]): string =
  ## decodeHf now DEFAULTS to skipSpecial=true: "<s>"/"[INST]"/
  ## "<SPECIAL_37>" are not printed, and mistral_common is never left to raise when
  ## it meets a special token (some Tekkenizer versions raise instead of skipping).
  case t.kind
  of atkBpe: decodeBPEStr(t.bpe, ids)
  of atkHf: decodeHf(t.hf, ids)

proc bannedMask(t: AnyTokenizer, vocab: int): seq[bool] =
  ## Bans only the tokenizer's CONTROL tokens, allowing every text token
  ## and whitespace (including \n / indentation). No garbageSpace table:
  ## a valid whitespace token may decode on its own to an empty string.
  result = newSeq[bool](vocab)
  case t.kind
  of atkHf:
    if t.hf.special.len != vocab:
      raise newException(ValueError,
        "[infer] tokenizer special table (" & $t.hf.special.len &
        ") khac vocab model (" & $vocab & ")")
    for i in 0 ..< vocab:
      result[i] = t.hf.special[i]
    # EOS / stop tokens must be sampleable so generation can end.
    for id in t.hf.stopIds:
      if id >= 0 and id < vocab: result[id] = false
  of atkBpe:
    discard

proc isStopAny(t: AnyTokenizer, id: int): bool =
  case t.kind
  of atkHf: isStopIdHf(t.hf, id)
  of atkBpe: false

proc pieceAny(t: AnyTokenizer, id: int): string =
  ## the raw piece name (e.g. "▁", "▁The") - for diagnostic output
  case t.kind
  of atkHf: pieceOfHf(t.hf, id)
  of atkBpe: ""

proc buildNoPenalty(t: AnyTokenizer, vocab: int): seq[bool] =
  ## Pure formatting tokens (\n, space, tab, indentation) are NOT treated as
  ## "repeated content". The lesson learned in nim_inference.nim:
  ## in code, \n repeats on almost every line; penalising it with repeat-penalty
  ## pushes the \n logit below space -> the model is forced to replace line breaks
  ## with spaces even when the syntax needs a newline (like llama.cpp's penalize_nl,
  ## which by default does NOT penalise \n).
  result = newSeq[bool](vocab)
  for probe in ["\n", "\n\n", " ", "  ", "    ", "\t", "\n    ", "\n\t", "\r\n"]:
    try:
      for id in encodeAnyNoBos(t, probe):
        if id >= 0 and id < vocab: result[id] = true
    except CatchableError:
      discard

proc loadCheckpointLoraForInfer(path: string): tuple[model: NimformerModelJamba, lora: JambaLora, tok: AnyTokenizer] =
  ## 2nd fix (in sync with finetune_lora.nim - NO .bin at all any more):
  ## The base (Mamba part) is REBUILT from .nimq (dequantized on the fly through the LUT, using
  ## loadCodestralQuantResidentFromNimq - EXACTLY like training). The
  ## Attention part (new blocks, trained in full) is LOADED FROM "newBlockWeights" in
  ## .lora.json. LoRA adapters are loaded from "adapters" as before. Base and adapters ARE
  ## STILL RETURNED SEPARATELY (like PeftModel.from_pretrained(base, adapter)) -
  ## the LoRA delta is added only TEMPORARILY through mergeIn/restoreBase on each
  ## forward in generate() below, NOT merged permanently here.
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
  echo &"[infer] Rebuilding the base from {nimqPath} (dequantized on the fly through the LUT, pattern='{pattern}') ..."
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
      # checkpoint raised "JSON checkpoint has a LoRA adapter outside the pattern".
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
      raise newException(ValueError, "[infer] JSON checkpoint has a newBlock outside the pattern: " & name)
    if v.kind != JArray or v.len != shapeNumel(byName[name][].shape):
      raise newException(ValueError, "[infer] JSON newBlock has a wrong shape/data: " & name)
    var idx = 0
    for x in v.items: byName[name][].data[idx] = x.getFloat.float32; inc idx
    seenNewBlocks[name] = true
    inc nAttnW
  for name, _ in requiredNewBlocks:
    if name notin seenNewBlocks:
      raise newException(ValueError, "[infer] JSON checkpoint is missing a required newBlock: " & name)
  echo &"[infer] Loaded {nAttnW} Linear++ Attention tensors (new blocks, trained in full) from .lora.json"

  var lo: JambaLora
  lo.adapters = initTable[string, LoraAdapter]()
  lo.backup = initTable[string, seq[float32]]()
  for name, v in j["adapters"].pairs:
    if name notin requiredAdapters or name notin byName:
      raise newException(ValueError, "[infer] JSON checkpoint has a LoRA adapter outside the pattern: " & name)
    var ad: LoraAdapter
    ad.inF = v["inF"].getInt()
    ad.outF = v["outF"].getInt()
    ad.r = v["r"].getInt()
    if ad.inF <= 0 or ad.outF <= 0 or ad.r <= 0 or
       byName[name][].shape != @[ad.outF, ad.inF] or
       v["A"].kind != JArray or v["B"].kind != JArray or
       v["A"].len != ad.r * ad.inF or v["B"].len != ad.outF * ad.r:
      raise newException(ValueError, "[infer] JSON LoRA has a wrong shape/data: " & name)
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
      raise newException(ValueError, "[infer] JSON checkpoint is missing a required LoRA adapter: " & name)
  lo.quantBase = quantResident   # inProj/outProj of the Mamba blocks - quant-resident,
                                  # mergeIn/restoreBase LUT-dequantize/free them when needed.
  # IndexDefect fix: bind quantBase → Linear.weightQ/useQuant BEFORE generate.
  bindQuantBaseToLinears(m, lo)

  # LIKE CFG_INIT_MODE in finetune_lora.nim: if the checkpoint comes from the
  # "codestral" mode (has a ".hf_tokenizer_dir" sidecar), the REAL tokenizer MUST be
  # reloaded from the original HF dir - a self-trained BPE would make the pretrain embedding meaningless
  # (see the detailed explanation in hf_tokenizer.nim). NO silent fallback.
  let hfDirSidecar = path & ".hf_tokenizer_dir"
  var tok: AnyTokenizer
  let step = j["epoch"].getInt()
  if fileExists(hfDirSidecar):
    let hfDir = readFile(hfDirSidecar).strip()
    echo &"[infer] Checkpoint from the codestral mode - reloading the real tokenizer from {hfDir} ..."
    let hfT = newHfTokenizer(hfDir)
    if vocabSizeHf(hfT) != CFG_CODESTRAL_VOCAB:
      raise newException(ValueError,
        &"[infer] vocab tokenizer ({vocabSizeHf(hfT)}) KHAC vocab model " &
        &"({CFG_CODESTRAL_VOCAB}) - see the explanation in loadPureMambaForInfer.")
    tok = AnyTokenizer(kind: atkHf, hf: hfT)
    echo &"[infer] Loaded {path} (base rebuilt from .nimq) + {lo.adapters.len} LoRA adapters (SEPARATE, not merged), " &
         &"epoch={step}, vocab={vocabSizeHf(tok.hf)} (real tokenizer)"
  elif fileExists(path & ".bpe.json"):
    let bpeTok = loadBPE(path & ".bpe.json")
    if vocabSize(bpeTok) != m.vocab:
      raise newException(ValueError,
        &"[infer] BPE vocab ({vocabSize(bpeTok)}) KHAC checkpoint vocab ({m.vocab}); " &
        "these token IDs cannot be used with the model's embedding/lm_head.")
    tok = AnyTokenizer(kind: atkBpe, bpe: bpeTok)
    echo &"[infer] Loaded {path} (base rebuilt from .nimq) + {lo.adapters.len} LoRA adapters (SEPARATE, not merged), " &
         &"epoch={step}, vocab={vocabSize(bpeTok)}"
  else:
    # See the full explanation in loadBinaryLoraForInfer - when both sidecars
    # are missing, create ".hf_tokenizer_dir" by hand (1 line of text), NO
    # re-finetune needed, AS LONG AS the checkpoint was trained with CFG_INIT_MODE="codestral".
    raise newException(IOError,
      &"[infer] No tokenizer sidecar found for checkpoint '{path}' " &
      &"(tried '{hfDirSidecar}' and '{path}.bpe.json'). If trained with " &
      "CFG_INIT_MODE=\"codestral\", create the sidecar with 1 command (NO " &
      &"re-finetune needed):\n  echo -n \"/path/to/the-HF-model-dir\" > \"{hfDirSidecar}\"")
  result = (m, lo, tok)

var gCkptNormPreset = ""
  ## The checkpoint's meta["normPreset"] (finetune_lora writes the forward norm preset used in training).
  ## Empty = OLD checkpoint (before this key existed) -> generate() measures the preset with a bisect.
var gCkptMambaNormImpl = ""

proc loadBinaryLoraForInfer*(path: string): tuple[model: NimformerModelJamba, lora: JambaLora, tok: AnyTokenizer] =
  ## Supports NIMBIN1 (fp32) and NIMBIN4 (QLoRA 4-bit).
  ## (RAM bug fix): newBlock is NO longer dequantized to fp32 at load (it used
  ## to live resident for the whole program, exactly the 8GB fp32 problem
  ## quant-resident solved for the base) - it now stays a
  ## QuantTensor, in the SAME table as quantResident (base from .nimq), so
  ## mergeIn()/restoreBase() (called in generate(), see infer_lora.generate)
  ## dequantize on the fly + free BOTH with the same mechanism, with no
  ## distinction between "base" and "newBlock" any more.
  var meta: Table[string, string]
  var tensors: seq[RawBinTensor]
  var newBlockQuant: Table[string, QuantTensor]
  if isQLora4Bin(path):
    echo "[infer] Detected NIMBIN4 (QLoRA 4-bit) — newBlock stays compressed, dequantized on the fly during generate()..."
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
    raise newException(ValueError, "Checkpoint is not a LoRA-only .bin: " & path)
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
      &"[infer] Checkpoint Mamba norm implementation '{gCkptMambaNormImpl}' is not supported; " &
       &"this runtime uses '{TRAIN_MAMBA_NORM_IMPL}'. Refusing to infer to avoid wrong logits.")
  if gCkptNormPreset.len > 0:
    echo "[infer] checkpoint records the training norm preset: " & gCkptNormPreset
    if gCkptMambaNormImpl.len > 0:
      echo "[infer] checkpoint records the Mamba norm implementation: " & gCkptMambaNormImpl
  else:
    echo "[infer] checkpoint does NOT record a norm preset (trained with an old trainer) - will measure with a bisect."
  # (debugging aid): print the model's actual block pattern - each
  # character is 1 block (R=Mamba2Real, P=Mamba2Real+Linear++ sum, G=Mamba2Real+
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
        "token IDs may still be in range but mean something else, producing garbage logits/tokens.")
    tok = AnyTokenizer(kind: atkHf, hf: hfT)
  elif fileExists(bpeSidecar):
    let bpeTok = loadBPE(bpeSidecar)
    if vocabSize(bpeTok) != checkpointVocab:
      raise newException(ValueError,
        &"[infer] BPE tokenizer vocab ({vocabSize(bpeTok)}) KHAC checkpoint vocab ({checkpointVocab}); " &
        "token IDs may still be in range but no longer mean the same thing to the embedding/lm_head.")
    tok = AnyTokenizer(kind: atkBpe, bpe: bpeTok)
  else:
    # This used to call loadBPE(bpeSidecar) directly when hfDirSidecar was missing -
    # if the checkpoint came from CFG_INIT_MODE=="codestral" (real HF tokenizer,
    # NO .bpe.json) both sidecars were missing -> a crash with a cryptic
    # IOError "cannot open ...bpe.json", suggesting a re-finetune was needed.
    # It is NOT: the ".hf_tokenizer_dir" sidecar is just 1 line of text
    # holding the path to the original HF dir - it can be created by hand, NO need to
    # rerun finetune_lora. (The new finetune_lora.nim writes this sidecar
    # EARLY, right after the tokenizer is ready - this error only happens with
    # old checkpoints saved BEFORE that fix.)
    raise newException(IOError,
      &"[infer] No tokenizer sidecar found for checkpoint '{path}' " &
      &"(tried '{hfDirSidecar}' and '{bpeSidecar}'). If this checkpoint was trained " &
      "with CFG_INIT_MODE=\"codestral\" (real HF tokenizer), create the sidecar with " &
      "1 command, NO re-finetune needed:\n" &
      &"  echo -n \"/duong/dan/toi/thu-muc-model-HF\" > \"{hfDirSidecar}\"\n" &
      "(use EXACTLY the CFG_HF_DIR path from fine-tuning, e.g. the downloaded Mamba-Codestral-" &
      "7B-v0.1 dir). If the checkpoint was trained with CFG_INIT_MODE=\"graft\" (self-trained " &
      &"BPE), a missing '{bpeSidecar}' really does need a re-finetune (the BPE cannot " &
      "be rebuilt from anything else).")
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
  echo &"[infer] quant-resident: {quantResident.len} compressed parameters in total " &
       &"({newBlockQuant.len} from the QLoRA4 newBlock = Attention/new blocks trained in FULL in the checkpoint, " &
       &"the rest from the base .nimq = Mamba R)."
  echo "[infer]   → newBlock is NOT a LoRA adapter; it is the full Attention weights stored in the same .bin."
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
      # checkpoint raised "checkpoint has a LoRA adapter outside the pattern".
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
      raise newException(ValueError, "[infer] checkpoint manifest is missing a required newBlock: " & name)
  for name, _ in expectedNewBlocks:
    if name notin requiredNewBlocks:
      raise newException(ValueError, "[infer] checkpoint declares a newBlock outside the model/pattern: " & name)
    if name notin byName or meta.getOrDefault("newBlock." & name & ".shape") != $byName[name][].shape:
      raise newException(ValueError, "[infer] manifest newBlock has a wrong shape: " & name)
  var seenNewBlocks = initTable[string, bool]()
  for name, qt in newBlockQuant:
    if name notin byName or name notin expectedNewBlocks or
       qt.shape != byName[name][].shape or qt.bytes.len == 0:
      raise newException(ValueError, "[infer] quant newBlock has a wrong shape/data: " & name)
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
        raise newException(ValueError, "[infer] checkpoint has a newBlock outside the model/pattern: " & name)
      if name in seenNewBlocks:
        raise newException(ValueError,
          "[infer] checkpoint has two newBlock sources (raw/quant or duplicate name): " & name)
      var expectedElements = 1
      for dim in tensor.shape: expectedElements *= dim
      if tensor.shape != byName[name][].shape or tensor.floatValues.len != expectedElements:
        raise newException(ValueError, "[infer] raw newBlock has a wrong shape/data: " & name)
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
        raise newException(ValueError, "[infer] checkpoint has a LoRA adapter outside the pattern: " & adapterName)
      if adapterName notin lo.adapters:
        let adIn = parseInt(meta["adapter." & adapterName & ".inF"])
        let adOut = parseInt(meta["adapter." & adapterName & ".outF"])
        let adRank = parseInt(meta["adapter." & adapterName & ".r"])
        let adAlpha = parseFloat(meta["adapter." & adapterName & ".alpha"]).float32
        if adIn <= 0 or adOut <= 0 or adRank <= 0:
          raise newException(ValueError, "[infer] LoRA has an invalid inF/outF/r: " & adapterName)
        var ad: LoraAdapter
        ad.inF = adIn; ad.outF = adOut; ad.r = adRank
        ad.alpha = adAlpha; ad.scale = adAlpha / float32(adRank)
        ad.A = Tensor(shape: @[adRank, adIn]); ad.B = Tensor(shape: @[adOut, adRank])
        lo.adapters[adapterName] = ad
      if part == "A":
        if adapterName in seenAdapterA:
          raise newException(ValueError, "[infer] checkpoint trung LoRA tensor A: " & adapterName)
        if tensor.shape != lo.adapters[adapterName].A.shape or tensor.floatValues.len != lo.adapters[adapterName].r * lo.adapters[adapterName].inF:
          raise newException(ValueError, "[infer] LoRA A has a wrong shape/data: " & adapterName)
        lo.adapters[adapterName].A.data = tensor.data
        lo.adapters[adapterName].mappedA = tensor.mappedData
        seenAdapterA[adapterName] = true
      elif part == "B":
        if adapterName in seenAdapterB:
          raise newException(ValueError, "[infer] checkpoint trung LoRA tensor B: " & adapterName)
        if tensor.shape != lo.adapters[adapterName].B.shape or tensor.floatValues.len != lo.adapters[adapterName].r * lo.adapters[adapterName].outF:
          raise newException(ValueError, "[infer] LoRA B has a wrong shape/data: " & adapterName)
        lo.adapters[adapterName].B.data = tensor.data
        lo.adapters[adapterName].mappedB = tensor.mappedData
        seenAdapterB[adapterName] = true
      else:
        raise newException(ValueError, "[infer] unknown checkpoint LoRA part: " & part)
  for name, _ in expectedNewBlocks:
    if name notin seenNewBlocks:
      raise newException(ValueError, "[infer] checkpoint thieu newBlock weight: " & name)
  for name, ad in lo.adapters:
    if name notin seenAdapterA or name notin seenAdapterB:
      raise newException(ValueError, "[infer] checkpoint thieu LoRA A/B: " & name)
  for name, _ in requiredAdapters:
    if name notin lo.adapters:
      raise newException(ValueError, "[infer] checkpoint is missing a required LoRA adapter: " & name)
  for name, _ in lo.adapters:
    if name notin requiredAdapters:
      raise newException(ValueError, "[infer] checkpoint has a LoRA adapter outside the pattern: " & name)
  lo.quantBase = quantResident
  # An all-zero A/B adapter = an exact zero delta -> dropped (same result, fewer LoRA GEMMs in prefill
  # and LoRA stages in one-dispatch). BYBY_KEEP_ZERO_LORA=1 keeps them.
  if getEnv("BYBY_KEEP_ZERO_LORA", "0") == "0":
    let dropped = dropZeroLoraAdapters(lo)
    if dropped.len > 0:
      echo &"[infer] Dropped {dropped.len} LoRA adapters with all-zero A/B (delta = 0, same result); " &
           &"{lo.adapters.len} effective adapters left."
  # The optional bounded merge runs after packed upload, directly into the
  # existing GPU buffers. Never materialize full dense FP32 weight matrices.
  # IndexDefect fix: bind quantBase → Linear.weightQ/useQuant BEFORE generate.
  # (NIMBIN4 + quantResident: weight.data is already @[], weightQ must be wired before
  # mergeInUploadStreamingJamba / uploadResident, including block A linears.)
  bindQuantBaseToLinears(m, lo)
  echo &"[infer] Loaded checkpoint {path}: {lo.adapters.len} LoRA adapters (R blocks) + " &
       &"{newBlockQuant.len} newBlock quant (Attention train full). " &
       "The Mamba base comes from .nimq, it is not in the checkpoint."
  echo "[infer]   ('LoRA-only' = the Mamba base is not stored; the full Attention weights are still in newBlock.)"
  result = (m, lo, tok)

proc isRawBinCheckpoint(path: string): bool =
  if not fileExists(path): return false
  var f: File
  if not open(f, path, fmRead): return false
  var magic = newString(8)
  let got = f.readBuffer(addr magic[0], 8)
  f.close()
  # Accepts both NIMBIN1 (fp32) and NIMBIN4 (QLoRA 4-bit)
  result = got == 8 and (magic == RAWBIN_MAGIC or magic == "NIMBIN4\0")

# ═══════════════════════════════════════════════════════════════
# SAMPLING - FULLY REWRITTEN.
#
# The OLD version (sampleFromLogits) only had temperature over ALL 32768 logits:
# no top-p/top-k/min-p, no special-token ban, no NaN guard. Three
# sources of garbage tokens from there:
#
#  (a) THE DISTRIBUTION TAIL. The remaining ~32k tokens each have p ~ 1e-6, together usually
#      0.5-3% of the probability mass. At T=0.7, on average every ~30-60
#      tokens you draw a COMPLETELY random token. For Mamba this is
#      much worse than for a Transformer: the garbage token is swallowed into the recurrent STATE,
#      with no way to "forget" it - it ruins all later generation.
#      -> blocked with min-p + top-p.
#
#  (b) SPECIAL TOKENS. tekken reserves the first ~1000 ids for control + never-used <SPECIAL_k>
#      placeholders; their lm_head rows are almost
#      untrained so their logits are pure noise, but they ARE in the softmax and
#      CAN be sampled. On top of that lm_head is FP4-quantized (see
#      quantize_codestral.nim - Fp32NameHints has "lm_head.bias" but DELIBERATELY
#      not "lm_head.weight"), so the quantization error is enough to push them
#      up. -> blocked with `banned` (except the stop tokens, since we WANT the model
#      to be able to emit a stop token and end).
#
#  (c) NaN. Old version: 1 NaN logit -> total=NaN -> r=NaN -> the condition
#      "acc >= r" is FALSE on EVERY iteration -> return probs.len-1 -> ALWAYS the
#      last vocab id (32767), completely silent, not a single log line. With fp4
#      + Metal on an integrated GPU, NaN/Inf is not rare at all.
#      -> guard + count + warn once.
# ═══════════════════════════════════════════════════════════════

type
  SamplerCfg* = object
    temperature*: float32
    topP*: float32          # 0 or >=1 = off
    minP*: float32          # threshold relative to the best token; 0 = off
    repeatPenalty*: float32 # 1.0 = off
    repeatWindow*: int

proc envF32(name: string, dflt: float32): float32 =
  let raw = getEnv(name, "")
  if raw.len == 0: return dflt
  try:
    return parseFloat(raw).float32
  except ValueError:
    stderr.writeLine "[infer] " & name & "='" & raw & "' is not a number - using the default"
    return dflt

proc envInt(name: string, dflt: int): int =
  let raw = getEnv(name, "")
  if raw.len == 0: return dflt
  try:
    return parseInt(raw)
  except ValueError:
    stderr.writeLine "[infer] " & name & "='" & raw & "' is not a number - using the default"
    return dflt

proc defaultSamplerCfg*(temperature: float32): SamplerCfg =
  ## Tuned through environment variables, no rebuild needed:
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
    stderr.writeLine "[infer] WARNING: " & $nBad & "/" & $vocab & " logits are NaN/Inf"
  if argMaxI < 0:
    raise newException(ValueError, "[infer] every token is banned/NaN")
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
    # INT4: beMatmulQ4. FP4-KM: GPU resident + MBLOCK=4 (fp4GemmGpuRows) after
    # prepareGpuModel — no host dequant.
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
    # R blocks in one-dispatch: the conv/ssm state lives in the OD pack -> import into it (not the old layer).
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
                            tokenIds: seq[int]; ctx: Backend; continued = false) =
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
        forwardResidentPrefill(m,tokenIds[offset..<offset+count],ctx,
          continuation=continued or offset>0)
        markGpuPrefillReady()
      offset += count
    return
  # The sequence path below starts from a zero state.
  if continued:
    raise newException(ValueError, "continued prefill needs the resident prefill")
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
                          tokenIds: seq[int]; ctx: Backend; continued = false) =
  ## Prefill R/P prefixes with sequence-parallel scans, then import each
  ## terminal state into the recurrent decode buffers. Decode remains one-token
  ## causal. Unsupported presets/blocks use the existing fallback; resident R/P/G
  ## prefixes use compact inference scans and GPU MPS tiles in chunks up to 128.
  ## continued: the tokens extend the current recurrent state (a later chat
  ## turn) instead of starting a sequence. Only the resident prefill carries
  ## a state in; without it the tokens run through decode steps.
  if tokenIds.len == 0: return
  if isGpuCtx(ctx): ensureTgCap(ctx)
  var fallbackReason = parallelPrefillFallbackReason(m, lora, tokenIds, ctx)
  if continued and fallbackReason.len == 0 and
      (getEnv("BYBY_PREFILL_RESIDENT", "1") == "0" or residentPrefillReason(m, ctx).len > 0):
    fallbackReason = "continued prefill needs the resident prefill"
  if fallbackReason.len == 0:
    try:
      forwardParallelPrefill(m, lora, tokenIds, ctx, continued)
      return
    except CatchableError as e:
      # A reset would drop the state the continued tokens extend.
      if continued: raise
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
  echo "[infer] Prefill done " & $tokenIds.len & " tokens -> starting generation."

proc forwardPrefillBatch*(m: var NimformerModelJamba; tokenIds: seq[int];
                          ctx: Backend) =
  ## Norm-sweep diagnostics intentionally do not receive the checkpoint's
  ## quantBase table, so keep their prior recurrent reference path.
  for i, tid in tokenIds:
    discard forwardStepJambaGpuRetry(m, tid, tokenIds, i, ctx, produceLogits = false)

# ═══════════════════════════════════════════════════════════════════
# STREAM decode with correct spacing + token / norm preset diagnostics
# ═══════════════════════════════════════════════════════════════════
type
  StreamDecoder = object
    baseIds: seq[int]    ## context tokens (the last prompt token)
    baseText: string
    emitted: string      ## text printed so far

proc initStreamDecoder(tok: AnyTokenizer, lastPromptId: int): StreamDecoder =
  result.baseIds = @[lastPromptId]
  result.baseText = decodeAny(tok, result.baseIds)
  result.emitted = ""

proc streamDelta(sd: var StreamDecoder, tok: AnyTokenizer, genIds: seq[int],
                 final = false): string =
  ## Each token used to be decoded ON ITS OWN (decodeAny(tok, @[id])). With
  ## SentencePiece (tokenizer.model.v3) decode([ "▁Hello" ]) = "Hello" - the leading
  ## space is DROPPED => words stick together, and a "▁"-only token becomes "" (nothing
  ## printed). Now the WHOLE sequence (last prompt token + generated tokens) is decoded and only
  ## the new part is printed => correct spacing between words. Incomplete UTF-8 bytes (U+FFFD at the end)
  ## wait for the next token.
  let full = decodeAny(tok, sd.baseIds & genIds)
  # `baseText` must be a prefix of `full`. If it is NOT (the tokenizer decode is not
  # monotonic in the prefix), do NOT reprint the whole `full` - that would repeat
  # what is already on screen (looks like garbage). If NOTHING was printed yet (emitted empty)
  # use `full` as before; if something was printed, skip this token to avoid repetition.
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
    stderr.writeLine "      blk kind      rms      max     mean  NaN/Inf  | residual-in  sidecar rms/max"
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
    # The Linear++ sidecar (P blocks) contribution relative to the block's input residual: if sidecar >> residual
    # the input (norm) of this branch was OFF during training - the sign of "hybrid garbage".
    var sp: seq[string]
    for st in stats:
      if st.kind == 'P' and st.sideRms >= 0:
        sp.add &"b{st.idx}={st.sideRms:.3f}/{st.sideMax:.2f}(res {st.inRms:.3f})"
    if sp.len > 0:
      stderr.writeLine "      sidecar Linear++ rms/max (res = block input residual rms): " & sp.join(" ")

proc teacherForced(m: var NimformerModelJamba, tok: AnyTokenizer, promptText, fullText: string,
                   vocab: int, ctx: Backend): tuple[nllPrompt, nllAnswer: float] =
  ## Lets the model read a SAMPLE sentence in the training format ("[INST] ..[/INST] answer") and measures
  ## per-token NLL (teacher forcing). A good model: answer NLL ~1-3; random =
  ## ln(32768)=10.4; only the shared (unigram) distribution ~ 7.
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
      stderr.writeLine &"      [tf] actual={pieceAny(tok, tgt).escape} guess={pieceAny(tok, top1).escape} " &
                       &"nll={nll:.2f} hang={rank}"
    else:
      if t >= 1:
        nllP += nll
        inc nP
  stderr.writeLine &"      prompt NLL={nllP / float(max(1, nP)):.2f} | answer NLL={nllR / float(max(1, nR)):.2f} " &
                   &"| answer tokens guessed right {hitsR}/{nR}   (random=10.40, unigram~7)"
  result = (nllP / float(max(1, nP)), nllR / float(max(1, nR)))

type SweepMode = object
  name: string
  skipA, noLora: bool
  preset: string

proc runBisect(m: var NimformerModelJamba, tok: AnyTokenizer, vocab: int,
               banned, noPenalty: seq[bool], cfgBase: SamplerCfg, ctx: Backend): string =
  ## BYBY_NORM_SWEEP (opt-in): bisects to find what is broken, WITHIN 1
  ## run (loading the model takes long). Each mode: (1) teacher-forcing NLL on a sample sentence
  ## in the training format, (2) greedy generation from a clear question, (3) residuals.
  ##   base-hf        : drop A blocks + disable LoRA = the ORIGINAL Mamba-Codestral  (must produce text)
  ##   base+lora-*    : drop A blocks, keep LoRA
  ##   hybrid-*       : full R + A + LoRA
  ## Mode suffix = the norm of the trainer that produced the checkpoint:
  ##   *-train : rms-fin-gn      CURRENT trainer (base RMSNorm gamma + norm_f, gamma
  ##                             synced - jamba_hybrid.syncRmsFromBaseNorms)
  ##   *-unit  : rms-fin-g1-unit OLD trainer: rms1/rmsF kept gamma=1 (loadNormsFromHfTensors
  ##                             was never called), LoRA + sidecar learned on that input
  ##   *-ln    : ln-nof-g1       OLDEST trainer: LayerNorm + beta, no norm_f
  ## Returns the suggested preset (""= keep the default): the preset of the hybrid mode with the lowest answer NLL,
  ## ONLY when it is clearly better (NLL < 6 and at least 1 nat below the default preset) - so old checkpoints
  ## still produce text without rerunning the sweep just to read the result.
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
  # Default: ALL modes = original model + (LoRA | LoRA+sidecar) x 3 trainer norm kinds
  # (current / old gamma=1 / old LayerNorm). Each mode ~2 minutes (loading the model is the slow part):
  # base+lora-* shows which norm kind LoRA (the R branch) matches, hybrid-* adds the Linear++ sidecar.
  # Narrow it down: BYBY_NORM_SWEEP_LIST=base-hf,hybrid-train
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
  stderr.writeLine "════════ [bisect] " & $modes.len & " modes, each: sample-sentence NLL + greedy " & $nTok & " tokens ════════"
  if not hasAttention or not hasLora:
    stderr.writeLine "      [bisect] current model: A/P=" & (if hasAttention: "yes" else: "NONE") &
      ", LoRA=" & (if hasLora: "yes" else: "NONE") &
      ". Toggles for missing components do not make a real control."
  var summary: seq[string] = @[]
  var bestPreset = ""
  var bestNll = 1.0e30
  var defaultNll = 1.0e30     # hybrid with the default preset rms-fin-gn
  for md in modes:
    gSkipAttn = md.skipA
    setGpuDisableLora(md.noLora)
    discard setGpuNormPreset(md.preset)
    let attnMode =
      if not hasAttention: "no A/P (toggle no-op)"
      elif md.skipA: "bo A/P"
      else: "co A/P"
    let loraMode =
      if not hasLora: "no LoRA (toggle no-op)"
      elif md.noLora: "LoRA off"
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
      stderr.writeLine "      logits(step 0): " & logitSummary(lg0, vocab, banned)
      stderr.writeLine "      top5 at step 0:"
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
      stderr.writeLine "      [bisect] ERROR in this mode: " & e.msg
    stderr.writeLine &"      ({monoSecs() - t0:.0f}s)"
  gSkipAttn = false
  setGpuDisableLora(false)
  if summary.len > 0:
    stderr.writeLine "[bisect] SUMMARY of answer NLL (low = good; random=10.4; base-hf = original model): " &
                     summary.join("  ")
  if bestPreset.len > 0 and bestPreset != "rms-fin-gn" and bestNll < 6.0 and
     defaultNll - bestNll >= 1.0:
    stderr.writeLine &"[bisect] CHECKPOINT MATCHES preset '{bestPreset}' (answer NLL {bestNll:.2f} vs " &
      &"{defaultNll:.2f} for rms-fin-gn): " &
      (if gCkptNormPreset.len == 0:
         "this checkpoint was trained with an OLD trainer, not the current one. " &
         "Generation will use this preset automatically (disable with BYBY_NORM_AUTO=0 or pick one with BYBY_NORM_PRESET)."
       else:
         &"BUT the checkpoint records preset '{gCkptNormPreset}' (that preset will be used) - the sweep and the checkpoint " &
         "disagree, needs a look (try BYBY_NORM_PRESET=" & bestPreset & " to compare).")
    result = bestPreset
  elif defaultNll < 1.0e29 and bestNll < 6.0 and bestPreset == "rms-fin-gn":
    stderr.writeLine "[bisect] the checkpoint matches the default preset rms-fin-gn (current trainer)."
  elif defaultNll < 1.0e29 and bestNll >= 6.0:
    stderr.writeLine &"[bisect] NO preset gives hybrid NLL < 6 (best {bestNll:.2f}): the checkpoint does not " &
      "match any tried norm kind - retrain with the current trainer (gamma synced, sidecar uses R's norm)."
  stderr.writeLine "════════ [bisect] done ════════"
  stderr.writeLine ""

proc mergeResidentLoraIfEnabled*(m: var NimformerModelJamba; lora: var JambaLora;
                                 ctx: Backend): int =
  ## Folds B@A into the resident FP4-KM base when BYBY_MERGE_LORA is on (the
  ## default). Call it after prepareGpuModel, once the packed weights are on the
  ## GPU. Requested by hand, a failure is fatal; on by default (a machine short
  ## of memory, a parameter the merge cannot encode) it falls back to the
  ## separate adapters, which are still attached to the Linears.
  ## Run configureInferenceDefaults() first: it resolves the mode.
  if getEnv("BYBY_MERGE_LORA","0")=="0" or lora.adapters.len==0:
    return 0
  if gMergeLoraExplicit:
    return mergeResidentLoraInplace(m,lora,ctx)
  try:
    result = mergeResidentLoraInplace(m,lora,ctx)
  except CatchableError as e:
    echo "[lora-merge] skipped, keeping the separate adapters: " & e.msg
    result = 0

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
  discard mergeResidentLoraIfEnabled(m,lora,ctx)
  echo &"[infer] GPU san sang. RSS: {rssMB()}"

  let vocabModel = m.vocab
  var ids = encodeAny(tok, prompt)
  if ids.len == 0:
    raise newException(ValueError, "[infer] the prompt encodes to 0 tokens - check the tokenizer.")
  for id in ids:
    if id < 0 or id >= vocabModel:
      raise newException(ValueError,
        &"[infer] the tokenizer produced id={id} outside [0,{vocabModel}) - vocab mismatch.")

  var rng = initRand(seed)
  let banned = bannedMask(tok, vocabModel)
  let noPenalty = buildNoPenalty(tok, vocabModel)
  let cfg = defaultSamplerCfg(temperature)
  var nBanned = 0
  for b in banned:
    if b: inc nBanned
  echo &"[infer] sampler: T={cfg.temperature} top_p={cfg.topP} min_p={cfg.minP} " &
       &"repeat_penalty={cfg.repeatPenalty}/win={cfg.repeatWindow}, " &
       &"banning {nBanned} control tokens (whitespace not banned)."

  # ── (1) check the Linear++ step kernel right on the GPU (closed form, a few seconds) ──
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
        stderr.writeLine "[lpcheck] ERROR (skipped, continuing): " & e.msg

  # ── (2) bisect: original model (no A, LoRA off) vs full hybrid ──────
  var sweepPreset = ""        # preset suggested by the bisect (only when the hybrid clearly matches a non-default norm kind)
  if getEnv("BYBY_NORM_SWEEP", "0") != "0":
    try:
      sweepPreset = runBisect(m, tok, vocabModel, banned, noPenalty, cfg, ctx)
    except Exception as e:
      stderr.writeLine "[bisect] ERROR: " & e.msg & " - skipped, running normally."
    setGpuDisableLora(false)
    gSkipAttn = false

  # ── norm configuration for the real generation ───────────────────────────────
  # Priority order:
  #   1. BYBY_NORM_PRESET (chosen by hand)
  #   2. the checkpoint's meta["normPreset"] = the forward preset used in training (NEW checkpoints)
  #   3. the preset suggested by the bisect (OLD checkpoints without that key; disable with BYBY_NORM_AUTO=0)
  #   4. the default
  # Default: NEW fine-tune checkpoints (gamma-synced trainer) are trained with the Mamba-SSM
  # group norm over n_groups -> "rms-fin-gn". The plain base also uses the reference semantics;
  # the "hf" preset keeps the old full-width Transformers behaviour for checkpoint comparison.
  # (The "VERIFY ... AUTO-CORRECTING" block below used to ALWAYS force rms-fin-g1 so
  #  BYBY_NORM_PRESET had no effect; it was removed.)
  let defaultPreset =
    "rms-fin-gn"
  var presetName = getEnv("BYBY_NORM_PRESET", "")
  var presetWhy = "BYBY_NORM_PRESET"
  if presetName.len == 0:
    if gCkptNormPreset.len > 0:
      presetName = gCkptNormPreset
      presetWhy = "recorded in the checkpoint (= the training preset)"
    elif sweepPreset.len > 0 and getEnv("BYBY_NORM_AUTO", "1") != "0":
      presetName = sweepPreset
      presetWhy = "suggested by the bisect (old checkpoint)"
    else:
      presetName = defaultPreset
      presetWhy = "mac dinh"
  if not setGpuNormPreset(presetName):
    stderr.writeLine "[infer] preset '" & presetName &
      "' is invalid (e.g.: train | hf | mamba-ssm | ln-nof-g1 | rms-fin-g1 | rms-fin-gn | rms-fin-g1-unit) " &
      "- using the safe default '" & defaultPreset & "'."
    discard setGpuNormPreset(defaultPreset)
    presetWhy = "default (the requested preset is invalid)"
  if getEnv("BYBY_NORM_SWEEP", "0") != "0":
    stderr.writeLine "[infer] RESTORED norm preset after bisect: " & gNormPresetName
  stderr.writeLine "[infer] FINAL norm preset for generation: " & gNormPresetName & "  (" & presetWhy & ")"
  stderr.writeLine "[infer] Model pattern: " & m.pattern & ", LoRA adapters: " & $lora.adapters.len
  if m.pattern == CFG_PURE_MAMBA_PATTERN and lora.adapters.len == 0 and gNormPresetName == "ln-nof-g1":
    stderr.writeLine "[infer] WARNING: the train preset does not suit a non-fine-tuned Codestral base; " &
      "logits/tokens may be at the wrong scale. Use BYBY_NORM_PRESET=hf to go back to the Transformers norm."
  if gNormPresetName notin ["rms-fin-gn", "rms-fin-g1"]:
    stderr.writeLine "[infer] LUU Y: preset '" & gNormPresetName & "' khac preset trainer hien tai (rms-fin-gn); " &
      "the g1/hf presets are legacy full-width. New checkpoints must use the preset recorded in their metadata."

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
  echo "[infer] Prefill done - starting decode."
  echo &"[infer] RSS after prefill: {rssMB()}"

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
      stderr.writeLine "==== [BYBY_TRACE_BLOCKS] residual after each block (step 0) ===="
      printBlockTable(bstats, true)
    else:
      let logits = forwardStepJambaGpuRetry(m, ids[^1], ids, ids.len - 1, ctx)
      lastLogits = logits.data

    let winStart = max(0, ids.len - cfg.repeatWindow)
    # The mask is only for control tokens, whitespace is NOT banned / classified.
    
    # DEBUG: log sampling diagnostics. BYBY_DEBUG_SAMPLING=1 prints the first N steps
    # (N = BYBY_DEBUG_SAMPLING_STEPS, default 5). BYBY_DEBUG_SAMPLING_ALL=1
    # logs EVERY step - use it when garbage tokens appear after step 5 (the old hard-coded
    # `step < 5` meant garbage tokens in later steps were never seen).
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
      stderr.writeLine "==== [top5 logits, step 0] vocab=" & $vocabModel & " " &
                       logitSummary(lastLogits, vocabModel, banned) & " ===="
      printTopK(lastLogits, vocabModel, banned, tok, 5)

    if isStopAny(tok, nextId):
      stoppedBy = "stop token id=" & $nextId
      stderr.writeLine &"[infer] hit {stoppedBy} at step {step+1} - STOPPING (as intended)."
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

  # flush the remaining buffered UTF-8 bytes
  let tail = streamDelta(sd, tok, genIds, final = true)
  if tail.len > 0:
    stdout.write(tail)
    stdout.flushFile()
    printedText.add(tail)

  stdout.write("\n")
  stdout.flushFile()
  if stoppedBy.len == 0:
    stderr.writeLine &"[infer] reached {maxNewTokens} tokens without a stop token."
  var distinctGen: seq[int] = @[]
  for g in genIds:
    if g notin distinctGen: distinctGen.add g
  let decodeTime = monoSecs() - decodeStart
  stderr.writeLine &"[infer] tong ket: sinh {genIds.len} token, {distinctGen.len} id khac nhau, " &
                   &"{sd.emitted.len} byte van ban in ra (prompt {promptLen} token)."
  stderr.writeLine "\n== Speed stats =="
  stderr.writeLine &"Prefill: {max(0, promptLen - 1)} tokens in {prefillTime:.3f}s -> " &
                   &"{(max(0, promptLen - 1).float / max(prefillTime, 1e-9)):.2f} tok/s"
  if genIds.len > 0 and decodeTime > 0:
    stderr.writeLine &"Decode:  {genIds.len} tokens in {decodeTime:.3f}s -> " &
                     &"{(genIds.len.float / decodeTime):.2f} tok/s " &
                     &"({(decodeTime * 1000 / genIds.len.float):.1f} ms/token)"
    if genIds.len > 1:
      let steady = decodeTime - firstStepTime
      stderr.writeLine &"         first token {firstStepTime:.3f}s; {genIds.len - 1} later tokens " &
                       &"{(float(genIds.len - 1) / max(steady, 1e-9)):.2f} tok/s " &
                       &"({(steady * 1000 / float(genIds.len - 1)):.1f} ms/token)"
  stderr.writeLine &"Total: {(max(0, promptLen - 1) + genIds.len)} tokens in " &
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
  # ═══ FIX: STRIP THE TRAILING SPACE OF THE PROMPT ═══
  # The old prompt was "[INST] Xin chao[/INST] " (with a trailing space), justified
  # in a comment as "to match the training distribution". IN FACT IT IS THE OPPOSITE:
  #
  # In training, finetune_lora.lineToText builds "[INST] {p}[/INST] {resp}"
  # and encodes it ONCE. SentencePiece attaches the space to the FIRST token of the response:
  # the token sequence is  ... "]" , "▁Hello" , ...
  #
  # At inference, a prompt ending in a space cuts RIGHT in the middle of that:
  # we feed  ... "]" , "▁"  (a LONE whitespace token), then make the model
  # generate "Hello" WITHOUT a leading space - a distribution it has NEVER SEEN
  # in all of training. A classic tokenization bug.
  #
  # Without the space the prompt ends at "]" and the model generates "▁Hello" itself -
  # EXACTLY matching training.
  var prompt = if args.len >= 2: args[1] else: "[INST] Xin chao[/INST]"
  if prompt.len > 0 and prompt[^1] in {' ', '\t', '\n'}:
    let trimmed = prompt.strip(leading = false, trailing = true)
    stderr.writeLine "[infer] WARNING: the prompt ends with whitespace -> " &
      "that produces a lone '▁' token, off the training distribution. " &
      "Stripped automatically. Set BYBY_KEEP_TRAILING_WS=1 if you DO want to keep it."
    if getEnv("BYBY_KEEP_TRAILING_WS", "0") != "1":
      prompt = trimmed
  let temperature = if args.len >= 3: parseFloat(args[2]).float32 else: CFG_TEMPERATURE
  # 2 NEW PARAMETERS (only used when FALLING BACK to plain Mamba - see below):
  #   ./infer_lora_bin <ckptPath> <prompt> <temperature> <hfDir> <nimqPath>
  let hfDirArg = if args.len >= 4: args[3] else: CFG_HF_DIR_DEFAULT
  let nimqPathArg = if args.len >= 5: args[4] else: CFG_NIMQ_PATH_DEFAULT

  let requestedBackend = getEnv("BYBY_INFER_BACKEND").strip()
  let ctx = if requestedBackend.len > 0: newBackend(requestedBackend)
            else: newBackend()
  # (NEW): when <ckptPath>.lora.json is NOT found, it no longer errors out/
  # stops - it switches automatically to inferring DIRECTLY on the pretrain Mamba
  # base (no LoRA, no grafted Attention blocks - since nothing was
  # trained to load into them). The idea: "use the fine-tuned checkpoint if there is one, otherwise
  # at least run the original model" instead of crashing outright.
  var m: NimformerModelJamba
  var lo: JambaLora
  var tok: AnyTokenizer
  if isRawBinCheckpoint(ckptPath):
    (m, lo, tok) = loadBinaryLoraForInfer(ckptPath)
  elif fileExists(ckptPath & ".lora.json"):
    (m, lo, tok) = loadCheckpointLoraForInfer(ckptPath)
  elif args.len > 0:
    raise newException(IOError,
      &"The given checkpoint '{ckptPath}' does not exist / is not recognised " &
      "(needs NIMBIN1/NIMBIN4 or '{ckptPath}.lora.json'). Refusing to silently fall back to plain Mamba.")
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
  echo &"[infer] RSS right after loading the model (before inference): {rssMB()}"
  echo &"[infer] prompt: {prompt}"
  discard generate(m, lo, tok, prompt, max(1, envInt("BYBY_MAX_NEW_TOKENS", CFG_MAX_NEW_TOKENS)), ctx,
                    temperature = temperature, seed = CFG_SEED)
  echo "─────────────────────────────"
