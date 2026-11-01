## finetune_lora.nim
## LoRA-style fine-tuning for NimformerModelJamba. IMPORTANT:
##   - 2 input checkpoint SOURCES, pick 1 with CFG_INIT_MODE:
##     "graft"     : as before - reads CFG_OLD_CHECKPOINT (a .bin already grafted
##                   by graft_attention.nim, a pattern like "MMAMMAMS",
##                   all Mamba2SSD 'M' trained from scratch).
##     "codestral" : NEW - reads the Mamba-Codestral-7B-v0.1 pretrain weights DIRECTLY
##                   from CFG_HF_DIR (the downloaded safetensors dir), builds a model
##                   with pattern CFG_CODESTRAL_PATTERN made of 'R' blocks (Mamba2Real -
##                   the real SSD formula, see mamba2_real.nim) loaded straight from the
##                   pretrain weights + 'A' blocks (new LinearPlusAttention,
##                   random init) - via load_codestral_into_jamba.nim.
##   - Which block kinds are LoRA-ified (pretrained: 'M' or 'R') or trained in full
##     (newly grafted: 'A'/'S') is read DIRECTLY from the built model's own `m.pattern`
##     - no mismatch between the 2 modes.
##   - Freezes every pretrained part ('M'/'R'), only LoRA-ifies (low rank)
##     + trains the new Attention blocks in FULL - same idea as before, unchanged.
##   - IMPORTANT: the base checkpoint is still .bin (rawbin.nim via jamba_hybrid.
##     saveCheckpoint/loadCheckpoint) - .nimq is NOT used here.

import std/[math, random, strformat, os, strutils, json, tables, sequtils, parsejson, streams, times, monotimes]
import nimformer
import backend
import quant
import rawbin
import jamba_hybrid
import hybrid_block_kernel   # gLoraGradRequest/LoraGradSpec - backward LoRA-aware
import mamba2_real              # Mamba2Real - needed for 'R' blocks
import load_codestral_into_jamba # loadCodestralIntoJamba - loads the real HF weights into 'R' blocks
import hf_tokenizer               # the REAL tokenizer (tekken/HF) - required for CFG_INIT_MODE=="codestral"
import lora
import session_vm
import jit_vm
import test_nimformer   # reuses BPETokenizer: getOrTrainBPE/encodeBPE/decodeBPEStr
import quantize_lora_bin_4bit
import mamba2_ssm_gpu   # softmaxCrossEntropyGpu, apfAdamStepFused
import checkpoint
import prefill_profile   # pfTime/pfReport - reused for BYBY_TRAIN_PROFILE

proc cRename(oldPath, newPath: cstring): cint {.importc: "rename", header: "<stdio.h>".}
proc cLink(oldPath, newPath: cstring): cint {.importc: "link", header: "<unistd.h>".}

# Safe Ctrl+C: the 1st press only sets a flag; break AFTER the current layer's Adam+SSD. The 2nd forces quit.
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
    stderr.write("\n[finetune_lora] Ctrl+C: stopping SAFELY after the current layer's Adam+SSD (press again to force).\n")
    stderr.flushFile()
  except CatchableError: discard

# ═══════════════════════════════════════════════════════════════
# CONFIG - edit here (this is the hard-coded default config). The 6 most important
# values (INIT_MODE/OLD_CHECKPOINT/HF_DIR/HF_SOURCE/NIMQ_PATH/CHECKPOINT_PATH)
# CAN be overridden on the command line (see parseFinetuneArgs() right below) -
# passing nothing uses EXACTLY this hard-coded config, old behaviour unchanged.
#   ./finetune_lora [initMode] [oldCheckpoint] [hfDir] [hfSource] [nimqPath] [outCheckpointPath]
const
  CFG_INIT_MODE_DEFAULT* = "codestral"                   # "graft" (old path) or "codestral" (NEW)
  CFG_OLD_CHECKPOINT_DEFAULT* = "mamba_chat_2_8b.bin"    # only used when CFG_INIT_MODE=="graft"
  CFG_HF_DIR_DEFAULT* = "test/codestral/models/hf"  # local Codestral tokenizer/config (or safetensors source)
  CFG_HF_SOURCE_DEFAULT* = "nimq"          # "safetensors" (reads the original ~14GB bf16 directly) or
                                     # "nimq" (reads the fp4-QUANTIZED file ~3.5GB, see
                                     # quantize_codestral.nim to create it ONCE
                                     # from CFG_HF_DIR first, then the original safetensors can be DELETED
                                     # to save disk - .nimq is still dequantized
                                     # to fp32 in RAM at load, it does NOT save RAM,
                                     # only DISK)
  CFG_NIMQ_PATH_DEFAULT* = "test/codestral/models/codestral_mamba_only.nimq"  # only used when CFG_HF_SOURCE=="nimq" -
                                     # MUST MATCH the output of quantize_codestral.nim /
                                     # graft ARGS (see Makefile: GRAFT_DEFAULT_OUT uses
                                     # EXACTLY this name so the 2 steps chain together)
  CFG_CHECKPOINT_PATH_DEFAULT* = "test/codestral/models/jamba_parallel_lora.bin"

proc argOrDefault(args: seq[string], i: int, default: string): string =
  ## "-" = sentinel keeping the default value (used to set a parameter
  ## AFTER it without changing this one) - like the argOr convention in the Makefile.
  if args.len > i and args[i].len > 0 and args[i] != "-": args[i] else: default

let finetuneArgs = commandLineParams()
let CFG_INIT_MODE* = argOrDefault(finetuneArgs, 0, CFG_INIT_MODE_DEFAULT)
let CFG_OLD_CHECKPOINT* = argOrDefault(finetuneArgs, 1, CFG_OLD_CHECKPOINT_DEFAULT)
let CFG_HF_DIR* = argOrDefault(finetuneArgs, 2, CFG_HF_DIR_DEFAULT)
let CFG_HF_SOURCE* = argOrDefault(finetuneArgs, 3, CFG_HF_SOURCE_DEFAULT)
let CFG_NIMQ_PATH* = argOrDefault(finetuneArgs, 4, CFG_NIMQ_PATH_DEFAULT)
let CFG_CHECKPOINT_PATH* = getEnv("BYBY_CHECKPOINT_PATH", argOrDefault(finetuneArgs, 5, CFG_CHECKPOINT_PATH_DEFAULT))
# PARAMETER 7: activation scratch dir (spill-to-disk).
# PARAMETER 8: Adam state dir on the SSD; defaults to the same SSD as scratch/model.
#   ./finetune_lora_bin - - - - - - /Volumes/Data 1/scratch /Volumes/Data 1/scratch/.nimformers_adam
# The Adam state is NOT deleted after each tensor - it must survive between steps. It is only
# deleted when training ends (or when a new run starts with oldStep==0).
# PARAMETER 7 (NEW): the dir for activation scratch files (spill-to-disk) -
# DEFAULTS to the SAME drive as the model (CFG_HF_DIR) since that drive has enough free
# GB (unlike the often nearly full system drive "/") - PASS IT SEPARATELY to use
# an external SSD DIFFERENT from the model drive, e.g.:
#   ./finetune_lora_bin - - - - - - "/Volumes/Data 1/scratch"
let CFG_SCRATCH_DIR* = getEnv("BYBY_SCRATCH_DIR", argOrDefault(finetuneArgs, 6, CFG_HF_DIR_DEFAULT))
let CFG_ADAM_DIR* = getEnv("BYBY_ADAM_DIR", argOrDefault(finetuneArgs, 7, CFG_SCRATCH_DIR & "/.nimformers_adam"))
let CFG_ATTN_WEIGHTS_DIR* = getEnv("BYBY_ATTN_WEIGHTS_DIR", argOrDefault(finetuneArgs, 8, CFG_SCRATCH_DIR & "/.nimformers_attnw"))
  # NEW: where the FULL-PRECISION weights of the Attention blocks ('A') are kept temporarily on the SSD -
  # see AttnWeightDiskStore. These parameters (~3.5GB, 12 'A' blocks) used to
  # sit PERMANENTLY in RAM for the whole training (unlike 'R' blocks, which
  # merge/restore through the LUT) - the largest remaining RSS source after
  # streaming the Adam state + activations + checkpoint load/save.

proc getCheckpointPath*(): string =
  getEnv("BYBY_CHECKPOINT_PATH", CFG_CHECKPOINT_PATH)

proc getScratchDir*(): string =
  getEnv("BYBY_SCRATCH_DIR", CFG_SCRATCH_DIR)

const
  CFG_CODESTRAL_PATTERN_DEFAULT* = "R".repeat(7) & "G" & "R".repeat(15) & "G" &
                                  "R".repeat(15) & "G" & "R".repeat(15) & "G" & "R".repeat(8)
  CFG_CODESTRAL_VOCAB* = 32768      # CONFIRMED: backbone.embeddings.weight=[32768,4096]
  CFG_CODESTRAL_HIDDEN* = 4096      # CONFIRMED: hidden_size
  CFG_CODESTRAL_NHEADS* = 128       # CONFIRMED: mixer.A_log=[128] (Mamba2's SSM head count -
                                     # DIFFERENT from the head count of the new LinearPlusAttention 'A' blocks,
                                     # the same value is used for both here for simplicity)
  CFG_CODESTRAL_DSTATE* = 128       # CONFIRMED (derived from conv_dim=10240=intermediate(8192)+
                                     # 2*nGroups*dState -> nGroups*dState=1024=8*128)
  CFG_CODESTRAL_NGROUPS* = 8        # CONFIRMED (1024/128=8, matches the HF Mamba2Config default)
  CFG_CODESTRAL_CONVKERNEL* = 4     # CONFIRMED: mixer.conv1d.weight=[10240,1,4]
  CFG_CODESTRAL_EXPAND* = 2         # CONFIRMED (FIXED - it used to be wrongly hard-coded as 1):
                                     # intermediateSize=8192=4096*2, from mixer.norm.weight=[8192]
                                     # and mixer.out_proj.weight=[4096,8192]
  CFG_LORA_RANK* = 4
  CFG_LORA_ALPHA* = 8'f32
  CFG_LORA_LR* = 1e-4'f32
  CFG_NEWBLOCK_LR* = 3e-4'f32                    # lr for the new A/S blocks (trained in full)
  CFG_G_SIDECAR_LR* = 5e-5'f32
  CFG_G_GATE_LR* = 2e-5'f32
  CFG_G_GRAD_CLIP_NORM* = 1'f32
  CFG_G_GATE_BIAS_MIN* = 2.0'f32     # the gate bias (init=4) must not drop below this
  CFG_G_GATE_BIAS_MAX* = 6.0'f32     # nor exceed this (keeps Mamba dominant)
  CFG_G_LR_WARMUP_STEPS* = 200       # the G LR scale goes from ~0 -> 1.0 over the first N steps
  CFG_G_MAX_RESCUE_ATTEMPTS* = 3     # consecutive rollbacks allowed for a G block before erroring
  CFG_G_LR_MIN_SCALE* = 0.05'f32     # floor of the per-block LR scale under repeated rescues
  # R-block (pure Mamba2Real) rescue. A 16× residual RMS barely catches
  # "hub hub hub" (the SSM mode drifts but the amplitude still looks fine). The mixer
  # rms(y-x)/rms(x) is where to tighten — close to the G sidecar's 0.5× scale, not 4×.
  CFG_R_MAX_OUTPUT_RMS_RATIO* = 16.0'f32   # residual out/input (amplitude blow-up)
  CFG_R_MAX_OUTPUT_MAX_RATIO* = 32.0'f32
  CFG_R_MAX_MIXER_RMS_RATIO* = 1.5'f32     # rms(y-x)/rms(x) — the SSM overtaking the residual
  CFG_R_MAX_MIXER_MAX_RATIO* = 4.0'f32
  CFG_R_MAX_RESCUE_ATTEMPTS* = 3
  CFG_R_LR_MIN_SCALE* = 0.05'f32
  # G mambaY: G_MAX_MAMBA_RMS_RATIO in jamba_hybrid.checkGBlockScale
  # (forwardBlock G always calls it; cache.mambaY is exposed by the block kernel).
  # Gate dispersion: if bias min≈max (std ~0) the G block degenerates to
  # constant gate; hard rescue when BYBY_G_GATE_STD_RESCUE=1
  CFG_G_GATE_BIAS_MIN_STD* = 1e-4'f32
  # Per-layer small probe during finetune: after a block update, optionally run
  # a short fixed-prompt generate and flag garbage tokens / large drift.
  # ON by default (getEnv default "1"). Disable with BYBY_LAYER_PROBE=0.
  # Runs every CFG_LAYER_PROBE_EVERY successful updates of a block (0 = every time).
  CFG_LAYER_PROBE_EVERY* = 1
  CFG_LAYER_PROBE_MAX_TOKENS* = 16
  CFG_LAYER_PROBE_GARBAGE_FRAC* = 0.25'f32  # P/G / default
  CFG_R_LAYER_PROBE_DRIFT* = 0.08'f32       # R/Mamba: tighter — 25% is too wide for an SSM
  CFG_SEQ_LEN_DEFAULT* = 256         # 256 x batch 1. Metal limit auto-scales with RAM
  CFG_BATCH_SIZE* = 1                             # a real Adam minibatch; no gradient accumulation
  CFG_EPOCHS* = 1
  CFG_CORPUS_PATH* = "test/spectron/databricks-dolly-15k.jsonl"
  CFG_BPE_MERGES* = 500
  CFG_CHECKPOINT_EVERY* = 1   # 1 - save a checkpoint every training step.
  CFG_LOG_EVERY* = 20
  CFG_KEEP_RESUME_STATE* = true # keep Adam + the FP32 new blocks on the SSD to resume without a reset

let CFG_CODESTRAL_PATTERN* = getEnv("BYBY_CODESTRAL_PATTERN", CFG_CODESTRAL_PATTERN_DEFAULT)

proc envSeqLen(defaultLen: int): int =
  ## Allows overriding CFG_SEQ_LEN with BYBY_SEQ_LEN to widen the training window on
  ## strong machines without a rebuild. A 128 window < p50 (~130 tokens on dolly)
  ## means most training windows start mid-sentence with SSM state = 0,
  ## off the inference distribution (full prompt + a prefilled state).
  let raw = getEnv("BYBY_SEQ_LEN", "")
  if raw.len == 0: return defaultLen
  try:
    let v = parseInt(raw)
    if v < 32:
      stderr.writeLine &"[finetune_lora] BYBY_SEQ_LEN=" & raw &
        " is too small (<32) - using the default " & $defaultLen
      return defaultLen
    echo &"[finetune_lora] BYBY_SEQ_LEN override -> CFG_SEQ_LEN={v} (default {defaultLen})"
    return v
  except ValueError:
    stderr.writeLine &"[finetune_lora] BYBY_SEQ_LEN='" & raw &
      "' is not a number - using the default " & $defaultLen
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
# DATASET STREAMING: the .jsonl is NOT read entirely into RAM (loadDollyJsonl used to
# return seq[string] for the WHOLE FILE, then makeBatches tokenized the WHOLE CORPUS into
# 1 huge seq[int], then built the WHOLE batch LIST (seq[tuple[...]]) before
# training - 3 layers piling up at once. The 5 python_code_dataset_*.jsonl files
# are quarantined: the code may be valid but the explanation/context/tags are
# cross-joined and heavily repeated; they are NOT in CFG_DATASETS. The active corpus is
# Dolly + the Nim/Python topic-code mega set. The chosen datasets are read as a stream:
# LINE BY LINE via the `lines()` iterator (Nim reads it as a
# stream, not the whole file), each line is tokenized, packed into 1 PRE-ALLOCATED token
# buffer (allocated ONCE, reused like a pool/ring - like a fixed allocator
# on embedded/ESP32, NO repeated malloc/free per batch), each batch is drawn
# out of that buffer (the rest shifted to the front, NO new seq allocated), then
# refilled from the file when the buffer needs it. The old batch (idsB/tgtB) goes out of scope right after
# 1 training step (train_step takes it then drops it) -> the GC reclaims it right away, nothing
# accumulates across epochs.
# ═══════════════════════════════════════════════════════════════

const ResumeActivationMagic = "NIMACT1"
const ResumeStateMagic = "NIMSTATE1"

type
  DatasetSchema* = enum
    dsDolly       # instruction/context/response (or output)
    dsTopicCode   # topic/language/code/note (nim_python_code_dataset_mega)

  DatasetSpec* = object
    path*: string
    schema*: DatasetSchema

const CFG_DATASETS*: seq[DatasetSpec] = @[
  # Only sources with a checked schema/label are enabled. No JSONL auto-scan: the
  # python_code_dataset_1..5 files have cross-joined code and explanation/context pairs.
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
  ## "" = a bad/empty line, the caller skips it. Does NOT raise (1 bad line must not
  ## kill a long training loop).
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
      # The old condition was `ins.len == 0 and resp.len == 0` (AND). If resp
      # was EMPTY but ins was not, it still produced an example "[INST] x[/INST] " + EOS - i.e.
      # TEACHING THE MODEL TO ANSWER EMPTY and stop right away. A few hundred such lines are enough for the model
      # to learn to stay silent. (Counted in databricks-dolly-15k.jsonl: 0 lines
      # affected - but the 5 python_code_dataset_*.jsonl files were not checked.)
      if ins.strip().len == 0 or resp.strip().len == 0: return ""
      let prompt = if ctx.len > 0: ins & "\n" & ctx else: ins
      # The OLD "<|user|>/<|assistant|>" template is ChatML - Codestral
      # (Mistral tokenizer V3, sentencepiece) does NOT use ChatML but
      # "[INST] ...[/INST] ..." (see the CFG_CODESTRAL_PATTERN comment and
      # hf_tokenizer.nim). "</s>" is NO longer appended here - the REAL eos
      # is added via addEos=true when calling encodeHf below (refill).
      result = &"[INST] {prompt}[/INST] {resp}"
    of dsTopicCode:
      let topic = jsonStringField(obj, "topic")
      let lang = jsonStringField(obj, "language")
      let code = jsonStringField(obj, "code")
      if code.strip().len == 0 or topic.strip().len == 0 or lang.strip().len == 0: return ""
      result = &"[INST] Write {lang} code for: {topic}[/INST] {code}"
  except JsonParsingError:
    result = ""

iterator streamTextsForever(specs: seq[DatasetSpec]): string =
  var anyFileFound = false
  for s in specs:
    if fileExists(s.path): anyFileFound = true
  if not anyFileFound:
    stderr.writeLine("[data] WARNING: no dataset found in CFG_DATASETS - using empty")
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
      stderr.writeLine("[data] No valid JSONL line in the datasets; using the iterator instead of looping forever.")
      keepStreaming = false

const TargetIgnore* = -1
  ## The "skip this position" target value (like PyTorch's ignore_index).
  ## ONLY softmaxCrossEntropyMasked below understands this value -
  ## the ORIGINAL test_nimformer.softmaxCrossEntropy does NOT, so never
  ## pass masked targets into that path (see the training loop at the end of the file).

type
  StreamingBatcher* = object
    tok: BPETokenizer
    hfTok: HfTokenizer
    usingHfTok: bool
    seqLen, batchSize: int
    ## `pending` is the ONLY token buffer, allocated once (newSeqOfCap) then
    ## reused for the whole training via setLen(0)/add/shift - like a
    ## fixed pool on an MCU (ESP32-style).
    pending: seq[int]
    ## NEW: docStart[i] = true means pending[i] is the FIRST token of a
    ## new document (i.e. the BOS of the next example). Used to mask the loss at the
    ## boundary - see nextBatch.
    docStart: seq[bool]
    nMasked: int          # stats: total number of masked positions
    nEmitted: int         # stats: total number of positions produced
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
  ## Tokenizes more lines from textIter, APPENDS them to `pending`, and marks
  ## the START position of each document in `docStart`.
  while b.pending.len < need:
    let t = b.textIter()
    if finished(b.textIter):
      if b.pending.len == 0: return
      break
    # addEos=true: this is a FULL TRAINING EXAMPLE (prompt+response), the model needs
    # to learn the real stopping point (the eos id) - UNLIKE encoding an inference prompt
    # (infer_lora.nim, addEos=false by default).
    let ids = if b.usingHfTok: encodeHf(b.hfTok, t, addEos = true) else: encodeBPE(b.tok, t)
    if ids.len == 0: continue
    var first = true
    for x in ids:
      b.pending.add(x)
      b.docStart.add(first)
      first = false

proc nextBatch*(b: var StreamingBatcher): tuple[ids, targets, targetsMasked: seq[seq[int]]] =
  ## ═══════════════════════════════════════════════════════════════════════
  ## TWO FIXES compared with the old version.
  ##
  ## (1) THE STRIDE = seqLen, not seqLen+1.
  ##     Old version: sample i took ids = pending[i*33 ..< i*33+32],
  ##             targets = pending[i*33+1 ..< i*33+33], then base jumped +33.
  ##     Result: the token at position i*33+32 was the TARGET of sample i but NEVER
  ##     anyone's input; token i*33+33 was the input of sample i+1 but
  ##     NOBODY predicted it. I.e. ONE IN EVERY 33 TRANSITIONS WAS NEVER
  ##     LEARNED, at every window boundary, for the whole training.
  ##     Now: stride = seqLen and the LAST token is KEPT as the first input of the
  ##     next batch -> every transition appears somewhere.
  ##
  ## (2) DOCUMENT BOUNDARY MASK.
  ##     The packed buffer joins examples together, ignoring boundaries:
  ##        ... </s> <s> [INST] ...
  ##     The old version put ALL positions into the loss, so the model was explicitly taught
  ##     "after </s> emit <s>" and "after any text <s> may be emitted".
  ##     That is EXACTLY what it did at inference: finish the answer -> </s> -> <s> ->
  ##     "[INST] ..." -> a new document.
  ##     Returns TWO target versions:
  ##       targets       - RAW, every id valid (for the non-streaming trainStepGraft
  ##                       path, through the ORIGINAL softmaxCrossEntropy of
  ##                       test_nimformer - that file is NOT changed)
  ##       targetsMasked - boundary positions = TargetIgnore (for
  ##                       trainStepGraftStreaming + softmaxCrossEntropyMasked
  ##                       defined in this file)
  ##
  ## ─── NOT FIXED HERE (on purpose, left for you to decide) ────────────────────────
  ## (a) CFG_SEQ_LEN = 32 is kept because raising it eats RAM/VRAM right away, and
  ##     the machine running it has an integrated GPU. But this is a BIG problem:
  ##     measured on databricks-dolly-15k.jsonl (15011 lines): example lengths
  ##     p50=468 chars (~130 tokens), p90=1720, p99=5316, max=27327.
  ##     With a 33-token window -> on average ~4 windows per example, p99 ~45 windows.
  ##     That means about 75% of training windows contain neither "[INST]" nor
  ##     "</s>", starting MID-SENTENCE with SSM state = 0. You are teaching the model
  ##     that "starting mid-sentence with an empty state" is normal - while
  ##     inference is the exact opposite. Raise it to >= 256 when possible.
  ## (b) MASKING THE PROMPT PART ([INST]...[/INST]) is not done. Doing it right requires
  ##     knowing the prompt/response boundary IN TOKENS, and the only safe way
  ##     is to encode the 2 halves separately and join them - which CHANGES how
  ##     the join point is tokenized (SentencePiece has add_dummy_prefix, encoding
  ##     " {resp}" separately may give "▁▁resp" instead of "▁resp"). The risk outweighs
  ##     the benefit, so it is left as a separate step with a loss measurement.
  ## ═══════════════════════════════════════════════════════════════════════
  let stride = b.seqLen
  let needTok = b.batchSize * stride + 1     # +1: the last target token is needed
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
        # the target is the FIRST token of the next document -> it cannot be inferred from the
        # previous context, dropped from the loss.
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
    # shift-left IN PLACE for both arrays (no new seq). The token at position
    # `consumed` is KEPT - it is the last target of the last sample and becomes the first input of the
    # next batch, so no transition is missed.
    let remain = b.pending.len - consumed
    for i in 0 ..< remain:
      b.pending[i] = b.pending[i + consumed]
      b.docStart[i] = b.docStart[i + consumed]
    b.pending.setLen(remain)
    b.docStart.setLen(remain)
  result = (idsB, tgtB, tgtMaskB)

proc skipSteps*(b: var StreamingBatcher, steps: int) =
  ## Resumes the dataset at EXACTLY the right position: the `epoch` in the checkpoint is actually
  ## the training step count, not the number of dataset passes. Since streamTextsForever()
  ## is always deterministic in file/line order, calling nextBatch() `steps` times
  ## skips EXACTLY the tokens trained before. Only runs on resume.
  if steps <= 0: return
  for i in 0 ..< steps:
    let (ids, _, _) = b.nextBatch()
    if ids.len == 0:
      raise newException(IOError,
        &"Resume: could not skip to training step {steps}; the dataset is shorter/changed compared with the previous run.")
    if (i + 1) mod 250 == 0:
      stderr.writeLine(&"[resume] skipped {i + 1}/{steps} training steps to reach the right dataset position")
  stderr.writeLine(&"[resume] dataset cursor = training step {steps} (earlier batches are NOT trained again)")

# ═══════════════════════════════════════════════════════════════
# MASKED LOSS - defined RIGHT IN THIS FILE.
# test_nimformer.softmaxCrossEntropy is NOT changed (the original is kept 100%),
# and neither is jamba_hybrid. The trainStepGraft path (non-streaming) still
# uses the RAW unmasked targets, through the original softmaxCrossEntropy as before.
#
# Why the mask is needed: the packed buffer joins examples together ignoring
# boundaries:   ... </s> <s> [INST] ...
# If ALL positions go into the loss, the model is explicitly taught two rules
#   "after </s> emit <s>"  and  "after any text <s> may be emitted"
# - exactly what it did at inference (finish the answer, then open a new
# document). Positions whose target is the FIRST token of the next document get
# TargetIgnore (declared above, before StreamingBatcher) and are dropped from the loss here.
# ═══════════════════════════════════════════════════════════════
proc softmaxCrossEntropyMasked(logits: Tensor, targets: seq[seq[int]],
                                ctx: Backend): tuple[loss: float32, dLogits: Tensor] =
  ## Like test_nimformer.softmaxCrossEntropy but:
  ##   - skips positions with target == TargetIgnore (their dLogits rows stay 0,
  ##     newTensor is already zeroed), the loss is divided by the NUMBER OF VALID POSITIONS, not B*T
  ##   - checks the bounds of `tgt`. The original did `probs[base + tgt]` with
  ##     base = (b*T+t)*V: if tgt >= V then base+tgt is STILL INSIDE the flat array
  ##     (landing in row t+1) at every position except the last row -> Nim's bounds check
  ##     does NOT catch it -> the gradient is added at the wrong time position, silently,
  ##     and the loss still looks fine. This raises right away.
  let B = logits.shape[0]
  let T = logits.shape[1]
  let V = logits.shape[2]
  # 100% GPU: softmax + NLL + dLogits + division by nValid in 1 micro-kernel per row
  # (train_gpu_vm.buildCeProgram). The host only checks the target ids are valid (integers).
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
          ") at b=" & $b & " t=" & $t & " - the tokenizer vocab and the model vocab differ.")
      tgtF[b * T + t] = float32(tgt)
      inc nValid
  if nValid == 0:
    raise newException(ValueError,
      "softmaxCrossEntropyMasked: the batch has no valid target; refusing the optimizer step to avoid an Adam update on an empty/zero gradient")
  var dLogits = newTensor(logits.shape)
  let r = softmaxCrossEntropyGpu(ctx, logits.data, tgtF, B * T, V, nValid)
  dLogits.data = r.dLogits
  result = (r.loss, dLogits)

# ═══════════════════════════════════════════════════════════════
# LORA CHECKPOINT: saves the Jamba base (grafted, LoRA NOT merged) + the A/B of each
# adapter separately, so the in-progress LoRA training state can be reloaded later.
# ═══════════════════════════════════════════════════════════════
proc writeFloatArrayChunked(f: File, data: seq[float32], chunk: int = 4096) =
  ## Writes a float32 array as a JSON array straight to the file, in chunks so
  ## a huge string/$JsonNode is NEVER held in RAM. Each number uses
  ## formatFloat(ffDecimal, precision=8) - precise enough for an fp32 checkpoint
  ## and 100% compatible with parseFile/getFloat in loadCheckpointLora.
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
  ## Local helper (not using nimformer.nim's private numel()) - computes
  ## the element count from the shape, STABLE across training steps (unlike
  ## .data.len, which may be 0 after a discard). Used as the trusted "paramLen"
  ## for AttnWeightDiskStore.hasW/loadW/saveW.
  result = 1
  for s in shape: result *= s

type
  AttnWeightDiskStore* = object
    ## NEW - mirrors AdamDiskStore: the FULL-PRECISION weights of the
    ## Attention blocks ('A', ~3.5GB raw for 12 blocks) live on the SSD, only 1 block
    ## is loaded into RAM exactly when forward/backward/checkpoint-save needs it.
    ## File format (.attnw): same spirit as AdamDiskStore - a small header
    ## (magic+n) then raw float32, NO JSON/JsonNode.
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

const AttnWeightMagic = 0x4E5442594157544D'u64   # "ATTNW"-ish, different from the AdamDiskStore magic

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
    raise newException(IOError, "Could not open the attn weight temp file: " & tmp)
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
    # Disk full: do NOT commit a truncated file (hasW only reads the header and would think it valid).
    try: removeFile(tmp) except OSError: discard
    raise newException(IOError, "Attn weight write is missing bytes (disk full?): " & tmp)
  if cRename(tmp.cstring, path.cstring) != 0:
    raise newException(IOError, "Could not commit the attn weight atomically: " & tmp)

proc loadW*(store: AttnWeightDiskStore, name: string, paramLen: int): seq[float32] =
  let path = attnPath(store, name)
  var f: File
  if not open(f, path, fmRead):
    raise newException(IOError, "Could not read the attn weight: " & path)
  defer: f.close()
  var magic: uint64
  var n: int64
  if f.readBuffer(addr magic, sizeof(magic)) != sizeof(magic) or
     f.readBuffer(addr n, sizeof(n)) != sizeof(n):
    raise newException(IOError, "Attn weight header is corrupt: " & path)
  if magic != AttnWeightMagic or n != int64(paramLen):
    raise newException(IOError, "Attn weight is incompatible (" & $n & " != " & $paramLen & "): " & path)
  result = newSeq[float32](paramLen)
  if paramLen > 0:
    let nBytes = paramLen * sizeof(float32)
    if f.readBuffer(addr result[0], nBytes) != nBytes:
      raise newException(IOError, "Attn weight thieu du lieu: " & path)

proc cleanupAttn*(store: var AttnWeightDiskStore) =
  ## Only called after the LAST checkpoint (exactly like adamStore.cleanup).
  try:
    for f in walkFiles(store.root / "*.attnw"):
      removeFile(f)
    for f in walkFiles(store.root / "*.attnw.tmp"):
      removeFile(f)
  except OSError: discard

# Step 0 update check (auditStep0LoraBlocks): meta[Step0AuditMetaKey] = "ok" in the .bin
# step 1 = every LoRA block has been confirmed to have a step 0 update computed from a REAL gradient
# (the real lm_head, the step 0 batch). Missing flag -> the next run checks + fixes it itself.
const Step0AuditMetaKey* = "step0Audit"
var gLoadedCkptMeta*: Table[string, string]   # meta .bin vua nap (loadCheckpointLoraBin)
var gStep0Verified* = false                    # this process has confirmed step 0 -> write the flag

proc saveCheckpointLoraBin*(m: var NimformerModelJamba, lora: JambaLora, path: string,
                            epoch: int, newBlockParamNames: seq[string],
                            attnStore: AttnWeightDiskStore) =
  ## LoRA-only checkpoint for inference / fallback resume — written as QLoRA 4-bit to save disk.
  ## The Mamba base is only referenced through the nimqPath meta, not copied into the .bin.
  ## Adapter A/B stay fp32; newBlock is quantized to 4-bit.
  ## Training resume prefers the FP32 Attention + Adam on the SSD; NIMBIN4 is the portable fallback.

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
  # The forward norm preset used in TRAINING -> infer_lora reads it back and uses EXACTLY this preset
  # (old checkpoints lack this key: infer measures it with a bisect, see infer_lora.runBisect).
  meta["normPreset"] = TRAIN_NORM_PRESET
  meta["mambaNormImpl"] = TRAIN_MAMBA_NORM_IMPL
  var fp32Keep: seq[RawBinTensor] = @[]
  var quantList: seq[(string, QuantTensor)] = @[]
  var requiredAdapters = initTable[string, bool]()
  for name, ad in lora.adapters:
    if ad.r <= 0 or ad.alpha <= 0'f32 or ad.alpha.classify in {fcNan, fcInf, fcNegInf} or
       ad.scale.classify in {fcNan, fcInf, fcNegInf} or
       ad.scale != ad.alpha / float32(ad.r):
      raise newException(IOError, "LoRA alpha/scale is invalid or inconsistent: " & name)
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
      raise newException(IOError, "Not exporting a checkpoint missing a required LoRA adapter: " & name)
  for name, _ in lora.adapters:
    if name notin requiredAdapters:
      raise newException(IOError, "Not exporting a checkpoint with a LoRA outside the pattern: " & name)
  for name, ad in lora.adapters:
    if ad.r <= 0 or ad.inF <= 0 or ad.outF <= 0 or
       ad.A.shape != @[ad.r, ad.inF] or ad.B.shape != @[ad.outF, ad.r] or
       ad.A.data.len != ad.r * ad.inF or ad.B.data.len != ad.outF * ad.r:
      raise newException(IOError, "Not exporting a LoRA adapter with a wrong shape/data: " & name)
    for v in ad.A.data:
      if v.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(IOError, "Not exporting a LoRA A with NaN/Inf: " & name)
    var bNonZero = false
    for v in ad.B.data:
      if v.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(IOError, "Not exporting a LoRA B with NaN/Inf: " & name)
      if v != 0'f32: bNonZero = true
    if epoch >= 1 and not bNonZero:
      # Empty layers are BANNED: after >= 1 training step every adapter must have B != 0 (B = 0 = a block
      # that never got a backward, exactly like the old 102/128 adapters bug).
      if getEnv("BYBY_FORCE_PARTIAL_COMMIT", "0") == "0":
        raise newException(IOError,
          &"NOT writing checkpoint step={epoch}: adapter '{name}' has B = 0 (layer never trained). " &
          "Set BYBY_FORCE_PARTIAL_COMMIT=1 if you really want to write a checkpoint with missing layers.")
      stderr.writeLine &"[ckpt] WARNING (BYBY_FORCE_PARTIAL_COMMIT=1): adapter '{name}' B = 0, writing anyway."
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
      raise newException(IOError, "Checkpoint is missing a newBlock parameter in the model: " & name)
    let p = byName[name]
    meta["newBlock." & name & ".shape"] = $p[].shape
    let expected = numelShape(p[].shape)
    var values: seq[float32]
    if lora.quantBase.len > 0 and attnStore.hasW(name, expected):
      # Streaming training: the SSD sidecar is the MAIN copy (trainStep loads from it and writes back after
      # the update). p[].data may still hold the OLD .bin value (the resume backward skips this block
      # in this process; [resume-layer]/COMMIT_ONLY run no step) -> do NOT use it.
      values = attnStore.loadW(name, expected)
    else:
      values = p[].data
      if values.len == 0:
        if not attnStore.hasW(name, expected):
          raise newException(IOError, "New block weight missing from RAM/SSD: " & name)
        values = attnStore.loadW(name, expected)
    if expected <= 0 or values.len != expected:
      raise newException(IOError,
        &"Not exporting newBlock '{name}': shape={p[].shape}, expected={expected}, data={values.len}")
    for v in values:
      if v.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(IOError, "Not exporting a newBlock with NaN/Inf: " & name)
    # (the "garbage Airlines AirAir" suspect): this used to force int4 asymmetric
    # per-ROW (ignoring groupSize) for weights and per-TENSOR for biases -> an RMS error
    # of ~13-15% exactly on the fully trained part. Now: 1-D (bias/gamma) stays fp32
    # raw; 2-D divisible by 32 uses fp4-KM group-32 (EXACTLY like the base .nimq, and the ONLY group the
    # GPU GEMV kernel can decode: buildFp4GemvProgram only supports groupSize=32); the rest is fp32 raw.
    #
    # (2nd fix): attn.qkvProj / attn.outProj / fuseGate.weight are trained in full from 0
    # with a very small lr (5e-5 / 2e-5). The learned values sit near the FP4-KM
    # group-32 noise level → quantizing them makes inference run on numbers far from the fp32 master
    # that was trained. They are kept fp32 raw in the mixed checkpoint (the meta still says
    # quantKind=mixed:qkFp4KM-or-fp32-raw); every other newBlock stays FP4-KM.
    let isSensitiveFullTrain =
      name.contains(".attn.") or name.contains(".fuseGate.")
    if (not isSensitiveFullTrain) and
       p[].shape.len == 2 and p[].shape[1] mod 32 == 0:
      let qt = quantizeNewBlock(values, p[].shape, qkFp4KM, 32)
      quantList.add(("newBlock::" & name, qt))
    else:
      fp32Keep.add(RawBinTensor(name: "newBlock::" & name, shape: p[].shape, data: values))
  # Write to a temp file, READ IT BACK to check every adapter/newBlock is there, only then replace the real file:
  # the main .bin file is never a broken or incomplete record.
  let writingPath = path & ".writing"
  try: removeFile(writingPath)
  except OSError: discard
  saveQLora4Bin(writingPath, meta, fp32Keep, quantList)
  let nQuantSaved = quantList.len
  let nFp32Saved = fp32Keep.len
  # The record is on disk; free it before reading back (G fp32 ~1.5GB) so RAM does not double.
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
          problems.add(key & ": the data read back differs from the data written")
    for name in newBlockParamNames:
      if ("newBlock::" & name) notin got and name notin vQuant:
        problems.add("thieu newBlock::" & name)
    if problems.len > 0:
      try: removeFile(writingPath)
      except OSError: discard
      raise newException(IOError,
        &"[ckpt] Reading back the just-written checkpoint FAILED ({problems.len} problems, e.g.: {problems[0]}) - " &
        &"NOT replacing {path}.")
    echo &"[ckpt] Read back OK: {lora.adapters.len} adapter A/B match element by element, " &
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
        raise newException(IOError, "Could not create the checkpoint rollback backup: " & e.msg)
    if cRename(backupTmp.cstring, backup.cstring) != 0:
      try: removeFile(backupTmp)
      except OSError: discard
      raise newException(IOError, "Could not commit the checkpoint rollback backup: " & backupTmp)
  if cRename(writingPath.cstring, path.cstring) != 0:
    raise newException(IOError, "Could not replace the checkpoint with the verified copy: " & writingPath)
  echo &"[ckpt] Saved QLoRA4 binary {path}: {lora.adapters.len} adapters (fp32), " &
       &"{nQuantSaved} newBlock (fp4-km), {nFp32Saved} fp32 raw tensors; the Mamba base is NOT saved."

proc loadCheckpointLoraBin*(path: string): tuple[model: NimformerModelJamba, lora: JambaLora, epoch: int] =
  ## Supports both fp32 .bin (NIMBIN1) and QLoRA 4-bit (NIMBIN4 — quantize_lora_bin_4bit.nim).
  ## (RAM bug fix, see the full explanation in infer_lora.loadBinaryLoraForInfer):
  ## newBlock is NO longer dequantized to fp32 at load - it stays a QuantTensor, merged
  ## into lora.quantBase TOGETHER with the base from .nimq, so mergeIn()/restoreBase()
  ## (used during training, lora.nim) dequantize on the fly + free it.
  var meta: Table[string, string]
  var tensors: seq[RawBinTensor]
  var newBlockQuant: Table[string, QuantTensor]
  if isQLora4Bin(path):
    echo "[ckpt] Detected NIMBIN4 (QLoRA 4-bit) — newBlock stays compressed, dequantized on the fly through quantBase..."
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
    raise newException(ValueError, "Not a LoRA-only checkpoint: " & path)
  gLoadedCkptMeta = meta
  let savedNormPreset = meta.getOrDefault("normPreset")
  let savedMambaNormImpl = meta.getOrDefault("mambaNormImpl")
  if savedNormPreset != TRAIN_NORM_PRESET or savedMambaNormImpl != TRAIN_MAMBA_NORM_IMPL:
    raise newException(ValueError,
      &"[resume] Checkpoint norm does not match the current trainer: normPreset='{savedNormPreset}', " &
       &"mambaNormImpl='{savedMambaNormImpl}', expected '{TRAIN_NORM_PRESET}' / " &
       &"'{TRAIN_MAMBA_NORM_IMPL}'. Refusing to resume to avoid mixing optimizer state with different forward math.")
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
      "QLoRA4 checkpoint has no newBlock.shape manifest; cannot prove the checkpoint is complete")
  for name, _ in requiredNewBlocks:
    if name notin expectedNewBlocks:
      raise newException(ValueError, "Checkpoint manifest is missing a required newBlock: " & name)
  for name, _ in expectedNewBlocks:
    if name notin requiredNewBlocks:
      raise newException(ValueError, "Checkpoint manifest has a newBlock outside the model/pattern: " & name)
    if name notin byName or meta.getOrDefault("newBlock." & name & ".shape") != $byName[name][].shape:
      raise newException(ValueError, "Checkpoint manifest newBlock has a wrong shape: " & name)
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
          "Checkpoint has two newBlock sources (raw/quant or duplicate name): " & name)
      seenNewBlocks[name] = true
  for name, _ in expectedNewBlocks:
    if name notin seenNewBlocks:
      raise newException(ValueError, "Checkpoint thieu newBlock weight: " & name)
  for name, qt in newBlockQuant:
    if name notin byName:
      raise newException(ValueError, "Checkpoint has a newBlock that does not exist in the model: " & name)
    if name notin expectedNewBlocks or qt.shape != byName[name][].shape or qt.bytes.len == 0:
      raise newException(ValueError, "Checkpoint quant newBlock has a wrong shape/data: " & name)
  var lo: JambaLora
  lo.adapters = initTable[string, LoraAdapter]()
  lo.backup = initTable[string, seq[float32]]()
  var seenAdapterA = initTable[string, bool]()
  var seenAdapterB = initTable[string, bool]()
  for tensor in tensors:
    if tensor.name.startsWith("newBlock::"):
      let name = tensor.name[10 .. ^1]
      if name notin byName or name notin expectedNewBlocks:
        raise newException(ValueError, "Checkpoint has an FP32 newBlock that is undeclared/does not exist: " & name)
      if tensor.shape != byName[name][].shape or tensor.data.len != numelShape(tensor.shape):
        raise newException(ValueError, "Checkpoint FP32 newBlock has a wrong shape/data: " & name)
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
          raise newException(ValueError, "Checkpoint LoRA has an invalid inF/outF/r: " & adapterName)
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
          raise newException(ValueError, "Checkpoint LoRA A has a wrong shape/element count: " & adapterName)
        lo.adapters[adapterName].A.data = tensor.data
        seenAdapterA[adapterName] = true
      elif part == "B":
        if adapterName in seenAdapterB:
          raise newException(ValueError, "Checkpoint LoRA trung tensor B: " & adapterName)
        if tensor.shape != @[lo.adapters[adapterName].outF, lo.adapters[adapterName].r] or
           tensor.data.len != lo.adapters[adapterName].outF * lo.adapters[adapterName].r:
          raise newException(ValueError, "Checkpoint LoRA B has a wrong shape/element count: " & adapterName)
        lo.adapters[adapterName].B.data = tensor.data
        seenAdapterB[adapterName] = true
      else:
        raise newException(ValueError, "Checkpoint LoRA has an unknown tensor part: " & part)
  for name, ad in lo.adapters:
    if name notin seenAdapterA or name notin seenAdapterB:
      raise newException(ValueError,
        "Checkpoint LoRA is missing A/B for adapter '" & name & "' - refusing a broken resume/export")
    if ad.A.data.len != ad.r * ad.inF or ad.B.data.len != ad.outF * ad.r:
      raise newException(ValueError, "Checkpoint LoRA A/B is empty or the wrong size: " & name)
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
      raise newException(ValueError, "Checkpoint is missing a LoRA adapter required by the pattern: " & name)
  for name, _ in lo.adapters:
    if name notin requiredAdapters:
      raise newException(ValueError, "Checkpoint LoRA adapter is outside the pattern: " & name)
  lo.quantBase = quantResident
  bindQuantBaseToLinears(m, lo)
  result = (m, lo, parseInt(meta["epoch"]))

proc saveCheckpointLora(m: var NimformerModelJamba, lora: JambaLora, path: string, epoch: int,
                         newBlockParamNames: seq[string], attnStore: AttnWeightDiskStore) =
  ## 3rd FIX (OOM on save): %*(seq[float32]) is NO longer used - a JsonNode
  ## tree for large arrays (e.g. the 144 'A' block parameters, ~3.5GB raw) swells 5-10x
  ## in RAM before $j turns it into a string -> the OOM killer struck during save. Replaced by
  ## streaming/chunked JSON text written straight to the file (only ~a chunk of floats
  ## in RAM at a time). The format is still valid JSON, loadCheckpointLora
  ## reads it as usual, the load side is NOT changed.
  ## Still NO .bin written - the 'R' (Mamba) part is still rebuilt from .nimq as before.
  ##
  ## ALSO FIXED (attnStore): since trainStepGraftStreaming discards the 'A' block
  ## weights from RAM right after each training step (see AttnWeightDiskStore),
  ## byName[name][].data HERE may be EMPTY when this function runs (called after
  ## a training step, not right after a checkpoint load). They are loaded TEMPORARILY from the SSD
  ## while writing, and returned to the empty state right after - the whole
  ## ~3.5GB is NEVER held in RAM longer than one file write.
  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t
  var serializedAlpha = CFG_LORA_ALPHA
  var haveSerializedAlpha = false
  for name, ad in lora.adapters:
    if ad.r <= 0 or ad.alpha <= 0'f32 or ad.alpha.classify in {fcNan, fcInf, fcNegInf} or
       ad.scale.classify in {fcNan, fcInf, fcNegInf} or
       ad.scale != ad.alpha / float32(ad.r):
      raise newException(IOError, "LoRA alpha/scale is invalid or inconsistent: " & name)
    if not haveSerializedAlpha:
      serializedAlpha = ad.alpha
      haveSerializedAlpha = true
    elif ad.alpha != serializedAlpha:
      raise newException(IOError,
        "JSON checkpoints only support a shared alpha; adapters have different alphas: " & name)
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
      raise newException(IOError, "[ckpt] STOP: newBlock parameter missing in the model: " & name)
    let p = byName[name]
    let wasEmpty = p[].data.len == 0
    if wasEmpty:
      let plen = numelShape(p[].shape)
      if attnStore.hasW(name, plen):
        p[].data = attnStore.loadW(name, plen)
      else:
        raise newException(IOError,
          &"[ckpt] STOP: '{name}' is empty in both RAM and attnStore - the checkpoint would be missing weights")
    if not firstNb: f.write(',')
    firstNb = false
    f.write(escapeJson(name))
    f.write(':')
    writeFloatArrayChunked(f, p[].data)
    if wasEmpty:
      p[].data = @[]   # back to the empty state from before entering here (the streaming path)
  f.write("}}")
  f.flushFile()
  echo &"[ckpt] Saved the LoRA adapters + {newBlockParamNames.len} Attention block parameters (full, STREAMING) " &
       &"-> {outPath}, epoch={epoch} (NO .bin written - the Mamba part is rebuilt from {CFG_NIMQ_PATH})"

proc loadCheckpointLora*(path: string): tuple[model: NimformerModelJamba, lora: JambaLora, epoch: int] =
  ## (OOM on resume - a real bug, mirroring "3rd FIX" in
  ## saveCheckpointLora): the OLD function used parseFile (std/json) - building the WHOLE
  ## JsonNode tree for the WHOLE FILE, including the LARGE float arrays (newBlockWeights ~3.5GB
  ## raw for the Attention blocks, adapters A/B). This is EXACTLY the kind of error
  ## confirmed and fixed on the WRITE SIDE (original comment: "a JsonNode tree for large arrays
  ## swells 5-10x in RAM before $j turns it into a string -> the OOM killer struck during save")
  ## but it was NEVER fixed on the READ SIDE - so EVERY resume (calling this function)
  ## still OOM'd for exactly the reason already fixed for save, explaining exactly the
  ## symptom seen: epoch 1 finished (streaming training, safe) but
  ## resuming from the checkpoint OOM'd right away (loading the whole file through JsonNode).
  ##
  ## Fix: a hand-written STREAMING parser with std/parsejson (SAX-style, event-
  ## based, NO JsonNode tree at all) - reads ONE NUMBER AT A TIME straight into the
  ## target seq/tensor. The file is still exactly the same valid JSON (format unchanged,
  ## backward compatible with existing checkpoints).
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
    ## Reads 1 "key" of the CURRENT object (the parser is positioned right before it -
    ## a jsonString or jsonObjectEnd, see the convention in the main function below).
    ## If it is a key: returns the name and moves the parser to its VALUE. If it is the end of
    ## the object (jsonObjectEnd): returns "" (it does NOT next() past it - the caller
    ## does that after the loop ends, see the convention below).
    if p.kind == jsonObjectEnd: return ""
    if p.kind != jsonString: raiseParseErr(p, "mong doi ten truong (jsonString)")
    result = p.str
    p.next()   # advance to this key's VALUE

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
    ## Uses nimqPath+pattern (already read earlier in the file, in the order
    ## saveCheckpointLora writes them) to build the model AS SOON AS it is needed (when
    ## the "adapters" or "newBlockWeights" key is reached) - letting
    ## newBlockWeights be written STRAIGHT into byName[name][].data (no intermediate
    ## seq), exactly the technique the OLD function used when reading from a JsonNode.
    if modelReady: return
    doAssert nimqPath.len > 0 and pattern.len > 0,
      "checkpoint JSON is missing nimqPath/pattern BEFORE adapters/newBlockWeights (wrong write order?)"
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
  if p.kind != jsonObjectStart: raiseParseErr(p, "the checkpoint must be a JSON object at the root")
  p.next()   # past the root '{'
  while true:
    let key = nextKey()
    if key == "": break
    case key
    of "epoch":
      if p.kind != jsonInt: raiseParseErr(p, "epoch must be an integer")
      epoch = int(p.getInt()); p.next()
    of "rank":
      # The real rank of each adapter is read from its own shape/metadata below.
      p.next()
    of "alpha":
      # The adapter scale is alpha/r. alpha must be resumed from the checkpoint, not taken
      # from the current binary's CFG_LORA_ALPHA: the config may have changed since the checkpoint.
      if p.kind notin {jsonInt, jsonFloat}:
        raiseParseErr(p, "alpha must be a number")
      checkpointAlpha = (if p.kind == jsonInt: float32(p.getInt())
                         else: float32(p.getFloat()))
      if checkpointAlpha <= 0'f32 or checkpointAlpha.classify in {fcNan, fcInf, fcNegInf}:
        raise newException(ValueError, "Checkpoint JSON LoRA alpha is invalid")
      checkpointAlphaFound = true
      p.next()
    of "nimqPath":
      if p.kind != jsonString: raiseParseErr(p, "nimqPath must be a string")
      nimqPath = p.str; p.next()
    of "pattern":
      if p.kind != jsonString: raiseParseErr(p, "pattern must be a string")
      pattern = p.str; p.next()
    of "adapters":
      ensureModel()
      if p.kind != jsonObjectStart: raiseParseErr(p, "adapters must be an object")
      p.next()   # past the '{' of adapters
      while true:
        let adName = nextKey()
        if adName == "": break
        if adName notin requiredAdapters or adName notin byName:
          raise newException(ValueError, "Checkpoint JSON has a LoRA adapter outside the pattern: " & adName)
        if adName in lo.adapters:
          raise newException(ValueError, "Checkpoint JSON trung LoRA adapter: " & adName)
        if p.kind != jsonObjectStart: raiseParseErr(p, "adapter '" & adName & "' must be an object")
        p.next()   # past the '{' of this adapter
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
            if p.kind != jsonInt: raiseParseErr(p, "inF must be an integer")
            inF = int(p.getInt()); p.next()
          of "outF":
            if p.kind != jsonInt: raiseParseErr(p, "outF must be an integer")
            outF = int(p.getInt()); p.next()
          of "r":
            if p.kind != jsonInt: raiseParseErr(p, "r must be an integer")
            r = int(p.getInt()); p.next()
          of "alpha":
            if p.kind notin {jsonInt, jsonFloat}:
              raiseParseErr(p, "adapter alpha must be a number")
            adapterAlpha = (if p.kind == jsonInt: float32(p.getInt())
                            else: float32(p.getFloat()))
            if adapterAlpha <= 0'f32 or adapterAlpha.classify in {fcNan, fcInf, fcNegInf}:
              raise newException(ValueError, "Checkpoint JSON adapter alpha is invalid: " & adName)
            adapterAlphaFound = true
            p.next()
          else:
            raiseParseErr(p, "unknown field in adapter '" & adName & "': " & fk)
        var ad: LoraAdapter
        if inF <= 0 or outF <= 0 or r <= 0 or
           byName[adName][].shape != @[outF, inF] or
           aData.len != r * inF or bData.len != outF * r:
          raise newException(ValueError, "Checkpoint JSON LoRA has a wrong shape/data: " & adName)
        ad.inF = inF; ad.outF = outF; ad.r = r
        # Old checkpoints without alpha keep the config-compatible behaviour.
        ad.alpha = if adapterAlphaFound: adapterAlpha
                   elif checkpointAlphaFound: checkpointAlpha
                   else: CFG_LORA_ALPHA
        ad.scale = ad.alpha / float32(ad.r)
        ad.A = newTensor(@[ad.r, ad.inF]); ad.A.data = aData
        ad.B = newTensor(@[ad.outF, ad.r]); ad.B.data = bData
        lo.adapters[adName] = ad
      p.next()   # past the '}' of adapters
    of "newBlockWeights":
      ensureModel()
      if p.kind != jsonObjectStart: raiseParseErr(p, "newBlockWeights must be an object")
      p.next()   # past the '{' of newBlockWeights
      while true:
        let wName = nextKey()
        if wName == "": break
        let arr = readFloatArray()
        if wName notin requiredNewBlocks or wName notin byName:
          raise newException(ValueError, "Checkpoint JSON has a newBlock outside the pattern: " & wName)
        if wName in seenNewBlocks:
          raise newException(ValueError, "Checkpoint JSON trung newBlock: " & wName)
        let expectedLen = numelShape(byName[wName][].shape)
        if arr.len != expectedLen:
          raise newException(ValueError,
            &"checkpoint newBlockWeights['{wName}'] has the wrong size: {arr.len} != {expectedLen}")
        byName[wName][].data = arr
        seenNewBlocks[wName] = true
      p.next()   # past the '}' of newBlockWeights
    else:
      raiseParseErr(p, "unknown root field in the checkpoint: " & key)

  doAssert modelReady, "checkpoint JSON is missing both adapters and newBlockWeights (corrupt file?)"
  for name, _ in requiredAdapters:
    if name notin lo.adapters:
      raise newException(ValueError, "Checkpoint JSON is missing a required LoRA adapter: " & name)
  for name, _ in requiredNewBlocks:
    if name notin seenNewBlocks:
      raise newException(ValueError, "Checkpoint JSON is missing a required newBlock: " & name)
  lo.quantBase = quantResident
  result = (m, lo, epoch)

type
  ## Adam state backing store: the m/v/step of EVERY parameter live on the SSD,
  ## only 1 state is loaded into RAM per update. The .adam files are kept
  ## between training steps; the .tmp files are only scratch and are deleted after saving.
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
  ## Frees the m/v in RAM right after the SSD save.
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
  ## Initialises a zero state on the SSD. Does NOT create a paramLen-sized seq[float32],
  ## avoiding a large RAM spike when a new file is created. Only a 1 MiB buffer is used.
  let path = adamPath(store, name)
  # NO early-return when the file already exists. This proc is the ONLY reset
  # primitive used by resetAdamParameters (resume distrust path) and by the G
  # rescue path in trainStepGraftStreaming - both NEED to overwrite an
  # existing (possibly poisoned) optimizer state with zeros. Callers that
  # merely want a fresh file already guard with `if not store.has(...)`.
  let tmp = path & ".tmp"
  var f: File
  if not open(f, tmp, fmWrite):
    raise newException(IOError, "Could not open the Adam state temp file: " & tmp)
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
    raise newException(IOError, "Could not commit the Adam state atomically: " & tmp)

proc load*(store: AdamDiskStore, name: string, paramLen: int): ApfAdamState =
  if not store.has(name, paramLen):
    store.init(name, paramLen)
  let path = adamPath(store, name)
  var f: File
  if not open(f, path, fmRead):
    raise newException(IOError, "Could not read the Adam state: " & path)
  defer: f.close()
  var magic: uint64
  var version: uint32
  var n: int64
  var step: int64
  if f.readBuffer(addr magic, sizeof(magic)) != sizeof(magic) or
     f.readBuffer(addr version, sizeof(version)) != sizeof(version) or
     f.readBuffer(addr n, sizeof(n)) != sizeof(n) or
     f.readBuffer(addr step, sizeof(step)) != sizeof(step):
    raise newException(IOError, "Adam state header is corrupt: " & path)
  if magic != 0x425942594144414D'u64 or version != 1'u32 or n != int64(paramLen):
    raise newException(IOError, "Adam state is incompatible: " & path)
  result = newApfAdamState(paramLen)
  result.step = int(step)
  if paramLen > 0:
    let mBytes = paramLen * sizeof(float32)
    if f.readBuffer(addr result.m[0], mBytes) != mBytes:
      raise newException(IOError, "Adam m is missing data: " & path)
    if f.readBuffer(addr result.v[0], mBytes) != mBytes:
      raise newException(IOError, "Adam v is missing data: " & path)

proc save*(store: AdamDiskStore, name: string, st: ApfAdamState) =
  if st.m.len != st.v.len or st.step < 0:
    raise newException(ValueError, "Adam state has an invalid m/v length or step: " & name)
  let path = adamPath(store, name)
  let tmp = path & ".tmp"
  var f: File
  if not open(f, tmp, fmWrite):
    raise newException(IOError, "Could not open the Adam state temp file: " & tmp)
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
    raise newException(IOError, "Adam state temp has missing/extra data: " & tmp)
  if cRename(tmp.cstring, path.cstring) != 0:
    raise newException(IOError, "Could not commit the Adam state atomically: " & tmp)

proc cleanup*(store: var AdamDiskStore) =
  ## Only called when the optimizer state is no longer needed (after the last checkpoint).
  ## Deletes both .adam and .tmp, so the SSD does not fill up after a run.
  try:
    for f in walkFiles(store.root / "*.adam"):
      removeFile(f)
    for f in walkFiles(store.root / "*.adam.tmp"):
      removeFile(f)
  except OSError: discard

proc loraAdamKey(name, part: string): string =
  ## Reuses AdamDiskStore for LoRA A/B.
  result = "__lora__" & part & "__" & name

proc saveLoraAdamStates*(store: AdamDiskStore, state: LoraTrainState) =
  ## Persists the m/v/step of ALL LoRA adapters after the weight checkpoint has
  ## been written successfully.
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
  ## Deletes/resets ALL the Adam state of the new blocks if resume finds a mismatch.
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
  ## Reads only the Adam header to check the step, without loading m/v.
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
  ## Checks that ALL the Adam states of the new blocks have the same step as the checkpoint.
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
  ## Like adamStoreAllAtStep but accepts step or step+1 (a step in progress:
  ## blocks updated in this step are at step+1, blocks not reached yet at step) - like LoRA.
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
  ## Merges the LoRA Adam state from the SSD. Do NOT let an old step=0 file overwrite a higher step in RAM
  ## (old bug: the sync overwrote with step=0 -> the checkpoint failed even though training had finished).
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
  ## m = v = 0 exactly: never received any gradient (fresh init / reset).
  for x in st.m:
    if x != 0'f32: return false
  for x in st.v:
    if x != 0'f32: return false
  true

proc forceLoraAdamToStep*(state: var LoraTrainState, lora: JambaLora, step: int): int {.discardable.} =
  ## Forces every adapter's A/B to exactly `step` (keeping m/v if the size is right). Used when
  ## the training step finished cleanly but some Adam files are still at step=0.
  ## Does NOT force adapters whose Adam B has m=v=0 (never got a gradient - e.g. a block
  ## not yet backpropagated in the step in progress): forcing the step would skew
  ## the bias correction in the next step and hide that the block was never trained.
  ## (B is used rather than A: at step 1 grad A = B^T.g = 0 is valid since B=0.)
  ## Returns the number of adapters skipped.
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
    stderr.writeLine &"[checkpoint] WARNING: {result}/{lora.adapters.len} LoRA adapters never had a " &
      &"gradient (Adam B m=v=0) - NOT forcing their Adam step -> {step}."

proc forceAdamStoreToStep*(store: AdamDiskStore, model: var NimformerModelJamba,
                           names: seq[string], step: int) =
  ## Forces the Adam file step of the new blocks (fully trained attn/fuseGate) to exactly `step`.
  ## Keeps the existing m/v; only rewrites the step header to sync with the training step
  ## (avoiding a checkpoint-final failure after resume RESET the new-block Adam).
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
  ## Checks that the LoRA A/B are at the same step before committing the checkpoint.
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
    stderr.writeLine &"[checkpoint] ... and {nBad - 8} more mismatched/missing LoRA adapters"

proc datasetSignature*(specs: seq[DatasetSpec]): string =
  ## A light signature to block resume: path + size + mtime.
  ## Size alone misses a corpus edit that keeps the same byte count.
  ## The whole corpus is not hashed since that would reread hundreds of MB/GB before every run.
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
  ## The FP32 Attention sidecar must be COMPLETE and the right size for every parameter.
  ## 1 missing file = the whole set is untrusted, avoiding mixing old FP32 with a new Q4 checkpoint.
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
  ## The commit marker is written AFTER the checkpoint + the LoRA Adam state.
  ## Also stores the dataset signature + tokenizer path so resume never continues on a different corpus/tokenizer.
  let tmp = checkpointPath & ".resume.commit.tmp"
  # escapeJson() ALREADY adds the quotes; this used to add another layer
  # -> "pattern":""RRR..."" was broken JSON, readResumeCommit always returned false
  # (every resume was treated as having no commit marker -> dropping the G FP32 sidecar
  # and resetting the G block Adam).
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
    raise newException(IOError, "Could not commit the resume marker atomically: " & tmp)
  try: removeFile(tmp)
  except OSError: discard

proc readResumeCommit*(checkpointPath: string, stepOut, seqLenOut, batchOut: var int,
                       patternOut, datasetSigOut, tokenizerDirOut: var string): bool =
  let path = checkpointPath & ".resume.commit"
  if not fileExists(path): return false
  try:
    let raw = readFile(path)
    # Old markers (before the writeResumeCommit fix) have strings quoted
    # twice: ""...""; the value inside has no quotes so removing 1 layer is enough.
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
  ## Loads the LoRA A/B from <ckpt>.inprogress/ (names: adapter::A / adapter::B).
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
  ## Returns lastCompletedLayer (-1 = none). Loads LoRA blocks nBlocks-1 .. cLayer.
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
# 1 training step: LoRA for the Mamba blocks (old, frozen) + full Adam for the newly
# grafted Attention/Spectron blocks.
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

  # -- LoRA (the old Mamba blocks, frozen base) --
  for name, ad in lo.adapters.mpairs:
    if name notin grads:
      raise newException(ValueError,
        "LoRA adapter '" & name & "' has no gradient in trainStepGraft. Stopping: training must not silently continue.")
    let (dA, dB) = gradToAB(grads[name], ad, ctx)
    var stA = loState.adamA[name]
    var stB = loState.adamB[name]
    discard apfAdamStepFused(ctx, ad.A, dA, stA, loState.lr, requantizeEvery = int.high)
    discard apfAdamStepFused(ctx, ad.B, dB, stB, loState.lr, requantizeEvery = int.high)
    loState.adamA[name] = stA
    loState.adamB[name] = stB

  # -- the new blocks (Attention/Spectron) are trained in FULL, no LoRA --
  for (name, t) in paramList(m):
    if name in lo.adapters: continue                 # already handled above (LoRA)
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
# STREAMING VERSION: used when lo.quantBase is NOT empty (the model uses the
# quant-resident source, see loadCodestralQuantResidentFromNimq) - processes ONE
# BLOCK at a time (merge -> forward, then merge again -> backward -> extract
# LoRA/Adam -> free RIGHT AWAY) instead of merging/forwarding/backpropagating the WHOLE MODEL at once
# and keeping one huge grads table until the end. Why this version is needed: with the 52
# 'R' blocks of Codestral-7B, the regular version (trainStepGraft) would hold AT THE SAME TIME:
#   - ~29GB if all fp32 weights are dequantized (mergeIn merges EVERYTHING)
#   - ~23GB for the full grads table (dInProjW/dOutProjW for ALL 52 blocks) before
#     reducing them to low-rank LoRA
# far beyond 16GB of RAM. This streaming version only holds ~1 block at any moment
# (~450MB of temporary weights + a few hundred MB of Mamba2Real cache) - verified correct by
# comparing against the regular version on SMALL tensors (see the accompanying test_streaming_matches.nim).
# ═══════════════════════════════════════════════════════════════
type Rusage {.importc: "struct rusage", header: "<sys/resource.h>", bycopy.} = object
  ru_maxrss {.importc: "ru_maxrss".}: clong   ## macOS: in BYTES (Linux uses KB, but
                                               ## this project only targets macOS/Metal so it is fine)

proc c_getrusage(who: cint, usage: ptr Rusage): cint {.importc: "getrusage", header: "<sys/resource.h>".}

proc rssMB(): string =
  ## Reads this process's own REAL peak RSS via getrusage() - a C API call
  ## IN THE SAME process, NO child process fork/spawn (unlike the previous version using
  ## startProcess("/bin/ps",...) - which fork()ed the whole process holding many
  ## GB of RAM ON EVERY LOG, itself adding memory pressure and making the very problem
  ## being investigated worse - removed). getrusage is very cheap, safe to call hundreds of times.
  var ru: Rusage
  if c_getrusage(0.cint, addr ru) == 0:
    result = $(int(ru.ru_maxrss) div (1024*1024)) & "MB"
  else:
    result = "?"

type
  DiskActivationStore = object
    ## Instead of keeping `xs: seq[Tensor]` (64 activations) ALIVE for the whole forward+
    ## backward in RAM, EACH activation is written to a scratch file on the SSD
    ## right after it is computed (forward), keeping ONLY (offset, len, shape) -
    ## a few dozen bytes per block - in RAM. When the backward needs an activation it
    ## seeks+reads EXACTLY that segment from the file, and drops it after use (not kept).
    ## Slower than keeping it in RAM (disk I/O per block) but in exchange it NEVER
    ## OOMs because of this part - the requested "use RAM only up to a level, spill the rest to the
    ## SSD", applied to the largest RAM-permanent-within-1-step structure
    ## (xs) rather than the whole model (weights are still handled separately through
    ## quant-resident + mergeInOne/restoreBaseOne as before).
    f: File
    path: string
    writeClosed: bool
    offsets: seq[tuple[pos, count: int64, shape: seq[int]]]

proc openDiskActivationStore(path: string): DiskActivationStore =
  createDir(getScratchDir())
  result.path = path
  if not open(result.f, result.path, fmWrite):
    raise newException(IOError, "Could not open the SSD scratch file: " & result.path)
  result.writeClosed = false
  result.offsets = @[]
  # 8-byte slot at position 0 holds the int64 offset of the header trailer.
  # Written as 0 now; overwritten by finalizeActivationStore after all payload.
  var zero: int64 = 0
  if result.f.writeBuffer(addr zero, sizeof(zero)) != sizeof(zero):
    raise newException(IOError, "openDiskActivationStore: writing the slot header failed")

proc openResumeActivationStore(path: string): DiskActivationStore =
  if not open(result.f, path, fmRead):
    raise newException(IOError, "Could not open the resume activations: " & path)
  # Read the 8-byte header-offset pointer at position 0.
  var hdrOffset: int64
  if result.f.readBuffer(addr hdrOffset, sizeof(hdrOffset)) != sizeof(hdrOffset) or
     hdrOffset <= 0:
    result.f.close()
    raise newException(IOError, "Resume activations: invalid header offset")
  result.f.setFilePos(hdrOffset)
  var magic = newString(ResumeActivationMagic.len)
  if result.f.readBuffer(addr magic[0], magic.len) != magic.len or magic != ResumeActivationMagic:
    result.f.close()
    raise newException(IOError, "Resume activations have no valid header")
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
  ## Reopens the activation file of an unfinished forward (no header yet): activation
  ## i is at 8 + i*bytes (every activation has the same shape [B,T,H]). Keeps the first nAct
  ## activations and appends after them.
  var count = 1
  for d in shape: count *= d
  let bytes = int64(count) * int64(sizeof(float32))
  let need = 8'i64 + int64(nAct) * bytes
  if shape.len == 0 or count <= 0 or nAct <= 0 or getFileSize(path) < need:
    raise newException(IOError,
      &"Forward resume: file activation {path} thieu du lieu ({getFileSize(path)} < {need} byte)")
  if not open(result.f, path, fmReadWriteExisting):
    raise newException(IOError, "Could not open the forward resume activations: " & path)
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
      raise newException(IOError, "writeActivationHeader: write failed (disk full?)")
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
    raise newException(IOError, "writeActivationHeader: patching the offset failed")

proc pushActivation(s: var DiskActivationStore, t: Tensor) =
  ## Writes DIRECTLY to disk, keeping NO copy in RAM after the write
  ## (t still lives in the caller's outer scope for a short moment to
  ## continue the forward, but this function creates no new reference to it).
  let pos = s.f.getFilePos()
  if t.data.len > 0:
    let n = t.data.len * sizeof(float32)
    if s.f.writeBuffer(unsafeAddr t.data[0], n) != n:
      raise newException(IOError, "pushActivation: write failed (disk full?)")
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
      # f.close() is handled by the finally below (it used to be closed twice).
      raise newException(IOError, "saveResumeState: write failed (disk full?)")
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
      # The forward state only stores the activation shapes (the data is in the activation file).
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
  ## Classifies an unfinished step from the layer-progress cursor + resume state:
  ## backwardDone = the backward has reached block 0 (only committed as a finished step);
  ## canResume = dX + activations remain so trainStepGraftStreaming can continue the backward.
  ## Returns a short description for logging.
  var rsNext = -2
  var rsPhase = ""
  var rsDX: Tensor
  var rsLoss = 0'f32
  let hasRs = loadResumeState(step, rsNext, rsPhase, rsDX, rsLoss)
  # cursor == 0: block 0's layer-progress written = backward done. Do NOT change the phase
  # "backward_done": at block 0 the progress is written BEFORE the resume state, a crash between
  # the 2 writes leaves phase="backward" next=0 -> replaying block 0 would update it twice.
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
    &"{nBlocks - cursorLayer}/{nBlocks} blocks backpropagated"

proc readActivation(s: DiskActivationStore, i: int): Tensor =
  ## Opens a SEPARATE READ file handle (not sharing the write handle above, avoiding
  ## disturbing the write cursor) - reads exactly 1 segment and closes RIGHT AFTER.
  let (pos, count, shape) = s.offsets[i]
  var rf: File
  if not open(rf, s.path, fmRead):
    raise newException(IOError, "Could not read the SSD scratch file: " & s.path)
  rf.setFilePos(pos)
  result = newTensor(shape)
  if count > 0:
    let nRead = rf.readBuffer(addr result.data[0], int(count) * sizeof(float32))
    if nRead != int(count) * sizeof(float32):
      rf.close()
      raise newException(IOError, "Short read from the SSD scratch file (disk full? truncated file?)")
  rf.close()

proc closeAndDelete(s: var DiskActivationStore, keep = false) =
  ## "Thorough cleanup": closes the file if still open, then DELETES the scratch RIGHT AWAY.
  ## The defer in trainStepGraftStreaming calls this if an exception happens mid-step,
  ## so the scratch is not left behind on the SSD.
  if not s.writeClosed:
    try: s.f.close()
    except: discard
    s.writeClosed = true
  if not keep:
    try: removeFile(s.path)
    except: discard

proc dlog(msg: string) =
  ## DIAGNOSTIC log, flushed immediately - so if the process is killed (OOM) the
  ## LAST log line left on the terminal is the most precise clue to
  ## which step it died in. rss=... is read via getrusage() (no fork).
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
  ## Updates 1 line (\r). Shows: [i/N] name  doneMB/totalMB  rate MB/s  ETA.
  ## forceNewline=true: ends the meter with a newline (called at the end of the loop).
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

# ── Training timing ──────────────────────────────────────────────────────
# Each block prints 1 "[time] ..." line (very cheap). BYBY_TRAIN_PROFILE=1: adds a detailed
# per-stage table (lin.gemm, blk.inProj, ...) from prefill_profile for each block.
let gTrainProfile = getEnv("BYBY_TRAIN_PROFILE", "0") != "0"
# LoRA-aware backward (ON by default): the kernel computes dA/dB directly, without the full dW
# [outF,inF] followed by gradToAB. BYBY_LORA_AWARE_BWD=0 -> the old path (for comparison).
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
  AuditGoodAgree = 0.90   # >= : the stored B signs match the real step 0 gradient -> the block is correct
  AuditBadAgree = 0.70    # < AuditGoodAgree is still recomputed; > AuditBadAgree (not ~50%) warns
var gAuditStopBlock* = -1 # audit: the first correct block (the backward stops here), -1 = none

proc graftInitForStep0(name: string; cur: seq[float32]): seq[float32] =
  ## The G block parameter values BEFORE step 0 (at graft time, see jamba_hybrid.newJambaBlock):
  ## attn.outProj = 0, fuseGate.weight = 0, fuseGate.bias = 4. attn.qkvProj has a step 0
  ## grad of 0 (outProj = 0) so it does not change -> kept as is. The current values may only be
  ## at most 1 Adam step from the init (G LR * 1/warmup ~ 2.5e-7); further away = not
  ## step 1 of this graft's checkpoint -> raise, no guessing.
  if ".attn.qkvProj." in name:
    return cur
  var init: float32
  if name.endsWith(".attn.outProj.weight") or name.endsWith(".attn.outProj.bias") or
     name.endsWith(".fuseGate.weight"):
    init = 0'f32
  elif name.endsWith(".fuseGate.bias"):
    init = 4'f32
  else:
    raise newException(ValueError, &"[audit] Unknown pre-step-0 graft value of '{name}'")
  for v in cur:
    if not (abs(v - init) <= 1e-5'f32):
      raise newException(ValueError,
        &"[audit] '{name}' differs from the graft value {init} by {abs(v - init):.3e} (> 1 step-0 Adam step) - " &
        "cannot reproduce the step 0 conditions. Stopping.")
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
  ## gLrScale (default 1.0) multiplies the LR of EVERY G block parameter (both the sidecar
  ## and fuseGate) - the caller (the real training loop) passes a warmup factor to avoid
  ## large updates at the very start. The other 'A'/'S' block parameters still use
  ## newBlockLr unchanged.
  ## updateMaxBlock: blocks i > updateMaxBlock are only backpropagated to get dX, with NO
  ## Adam/rescue/layer-progress (used by fillUntrainedLoraBlocks to run the
  ## missing part of the backward of an unfinished step).
  ## auditRefB (non-empty = step 0 check mode, auditStep0LoraBlocks): every B is currently 0
  ## and the G blocks are at their graft values (the step 0 conditions). Block by block from the top:
  ## the stored B signs (auditRefB) match -gradient -> the block is correct, the backward STOPS here
  ## (gAuditStopBlock); no match -> reset Adam and update again as at step 0.
  let auditMode = auditRefB.len > 0
  doAssert isGpuCtx(ctx),
    "trainStepGraftStreaming requires a GPU backend; refusing CPU execution"
  # Allow automatic CPU fallback when GPU runs out of memory, with GPU retry
  # on the next operation. This prevents OOM crashes on memory-constrained GPUs.
  setForbidCpuFallback(false)
  gLastStepForwardRan = false
  dlog "trainStepGraftStreaming: starting"
  var resumePhase = ""
  var resumeNextLayer = m.blocks.len - 1
  var resumedDX: Tensor
  var resumedLoss = 0'f32
  var canResumeBackward = false
  var resumeForwardFrom = -1   # >= 0: activations 0..resumeForwardFrom are already on the SSD
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
      # This step's backward HAS reached block 0 (block 0's layer-progress is written): the step
      # is done and must be committed. Running/replaying it again would update block 0 (or the whole step) twice.
      raise newException(IOError,
        &"[resume] step {stepIndex} has finished the backward down to block 0 but is not committed - NOT running it again. " &
        "Run again with BYBY_FORCE_CHECKPOINT=1 (the default) to commit this step.")
    if hasState and resumePhase == "backward" and resumeNextLayer >= 0 and
       resumeNextLayer < m.blocks.len and fileExists(actPath):
      canResumeBackward = true
      # Load the LoRA A/B deltas already applied to the upper blocks before the
      # crash. Their Adam m/v were persisted per layer in adamStore, so
      # restoreLoraAdamStates() at startup already recovered them.
      let restoredLayer = restoreLayerProgressForStep(lo, getCheckpointPath(), stepIndex,
        m.blocks.len, updateMaxBlock)
      # Blocks > updateMaxBlock only pass dX through and write no layer-progress: until
      # block updateMaxBlock is reached there is no cursor (-1).
      let expectedLayer = if resumeNextLayer + 1 > updateMaxBlock: -1 else: resumeNextLayer + 1
      # restoredLayer == resumeNextLayer: block vua bi RESCUE rollback (cursor = i,
      # nextLayer = i) -> replaying that same block is valid.
      if resumeNextLayer < m.blocks.len - 1 and restoredLayer != expectedLayer and
         restoredLayer != resumeNextLayer:
        raise newException(IOError,
          &"[resume] Per-layer progress layer {restoredLayer} does not match resumeNextLayer {resumeNextLayer} " &
          &"(expected layer {expectedLayer}). KEEPING .inprogress; check the snapshot before intervening.")
      dlog &"resume state: backward layer {resumeNextLayer}, the forward is not run again"
    elif hasCursor:
      # Some blocks were already updated in this step but the backward cannot resume: running
      # the whole step again would update the blocks >= cursor A 2ND TIME. Stop.
      raise newException(IOError,
        &"[resume] step {stepIndex}: has layer-progress (cursor layer {curLayer}) but no " &
        &"dX/activations to continue the backward (phase='{resumePhase}'). NOT running the step again (it would update " &
        "twice). Delete <ckpt>.inprogress to train the step again.")
    elif hasState and resumePhase == "forward" and resumedDX.shape.len > 0 and
         resumeNextLayer >= 0 and resumeNextLayer <= m.blocks.len and fileExists(actPath):
      resumeForwardFrom = resumeNextLayer
      dlog &"resume state: forward finished {resumeNextLayer}/{m.blocks.len} blocks, continuing the forward"
  # adapter names per block (1 'R' block has 2: inProj.weight, outProj.weight)
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
  dlog "trainStepGraftStreaming: namesByBlock done, starting paramList + bootstrap (NOT the embed lookup)"

  # Built ONCE for the WHOLE function (instead of calling paramList(m) MANY TIMES
  # inside the 'A' block loop below - paramList(m) rebuilds the WHOLE
  # model parameter list (~250+ entries) ON EVERY call, called again for each
  # gradient of each 'A' block (Attention has Q/K/V/O = 4 parameters/block)
  # x 13 'A' blocks x 64 times (forward+backward) is a lot of rebuilding of
  # an unnecessary list - both slow and leaving temporary garbage uncollected
  # (no GC_fullCollect runs in the 'A' block branch at all) - exactly
  # why the RSS jumped right at the 'A' blocks (19, 24) rather than the
  # 'R' blocks (flat, proving the merge/restore mechanism for 'R' works).
  dlog "paramList(m) starting..."
  var paramByName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): paramByName[name] = t
  dlog "paramList done (" & $paramByName.len & " names), paramQuantFlags..."
  let quantFlags = paramQuantFlags(m)
  dlog "paramLinears..."
  let linearsByName = paramLinears(m)
  var fullPrecWasQuantized = initTable[string, bool]()

  # NEW - mirrors namesByBlock (LoRA) but for the FULL-PRECISION parameters
  # ('A' blocks, not LoRA-ified) - used to load/free them through attnStore exactly when needed
  # for EACH block, instead of keeping all ~3.5GB for the whole training step (the original bug causing
  # the RSS jump recorded in the comment above).
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
  dlog "fullPrecNamesByBlock done: " & $nFullPrec & " tensors ~" &
    $(totalBootBytes div (1024*1024)) & "MB, bootstrap attnStore (1 dong refresh + MB/s)"

  # Bootstrap: the FIRST time this function is called in the whole training, p[].data
  # holds the REAL values (from a resume checkpoint or a fresh init) - saved once
  # to the SSD. In LATER steps, p[].data is always EMPTY on entry (freed at
  # the end of the previous step) so attnStore.hasW is true, and an empty value is never written
  # over the right file by mistake (numelShape(p[].shape) is used - STABLE, UNLIKE
  # .data.len which may be 0 - as the trusted "paramLen").
  # The FIRST time may take MANY MINUTES on a slow volume (exFAT/network/USB) -
  # each G tensor (qkv/out) can be hundreds of MB of FP32.
  # Progress: 1 refreshing \r line + MB/s + ETA to measure the disk throughput.
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
          # Resume fallback: NIMBIN4 keeps the newBlock weights quantized with data=[]
          # in the Linear. 1 tensor is dequantized here to create an FP32 working copy on the SSD;
          # the whole model is NOT dequantized.
          let dq = dequantizeTensor(lo.quantBase[name])
          if dq.len != plen:
            raise newException(IOError,
              &"bootstrap attnStore: '{name}' quantized size {dq.len} != shape numel {plen}")
          attnStore.saveW(name, dq)
        else:
          raise newException(IOError,
            &"bootstrap attnStore: '{name}' is empty in both RAM and quantBase (no weight to resume)")
        progressTick(bootProg, itemName = name & " WRITE", itemBytes = nbytes)
      else:
        progressTick(bootProg, itemName = name & " skip", itemBytes = nbytes)
  progressEnd(bootProg)
  dlog "bootstrap attnStore done, starting the real embed.lookupBatch"

  # ---- FORWARD: block by block, merge first (dequant through the LUT if quant-resident), forward, restore/free right away ----
  var x: Tensor
  var xstore: DiskActivationStore
  # Forward resume: the "forward" state next=n means activations 0..n have been
  # flushed to the SSD (blocks 0..n-1 forwarded). Written AFTER the activations are flushed.
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
      &"({resumeForwardFrom}/{m.blocks.len} blocks forwarded)"
  else:
    x = m.embed.lookupBatch(idsBatch, ctx)
    dlog "embed lookup done, starting the forward through " & $m.blocks.len & " blocks"
    xstore = openDiskActivationStore(actPath)
    xstore.pushActivation(x)
    saveForwardState(0)
  # Keep the scratch file whenever we are inside a checkpoint-able step: a
  # crash during backward must leave the activations on disk for resume. The
  # file is removed explicitly after the step commits.
  defer: xstore.closeAndDelete(keep = stepIndex >= 0)
  if canResumeBackward or resumeForwardFrom >= 0:
    # The stored activations MUST belong to this batch: activation 0 = embed(ids) (a copy of
    # embedding rows, deterministic) - a mismatch = the batch/seqLen/dataset changed between the 2 runs.
    let x0Now = m.embed.lookupBatch(idsBatch, ctx)
    let x0Saved = xstore.readActivation(0)
    if x0Saved.shape != x0Now.shape or x0Saved.data != x0Now.data:
      if canResumeBackward:
        raise newException(IOError,
          &"[resume] step {stepIndex}: the stored activations/dX belong to a DIFFERENT batch than the current one " &
          &"(stored shape={x0Saved.shape}, current={x0Now.shape}) - NOT continuing the backward (the gradient would be wrong). " &
          "Check BYBY_SEQ_LEN/the dataset; delete <ckpt>.inprogress to train the step again.")
      stderr.writeLine &"[resume] step {stepIndex}: the stored forward activations belong to a different batch " &
        "-> dropped, forwarding again from the start (the forward has not updated any weight)."
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
      # Bug4: detect NaN/Inf right after each block instead of waiting for the loss
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
      dlog &"block {i}: forward done"
      if gStopRequested and stepIndex >= 0:
        gTrainStepFullyDone = false
        stderr.writeLine &"[finetune_lora] Stopped safely after forward block {i} " &
          "(activations + forward state are on the SSD). Run again to continue the forward."
        return 0'f32
    finalizeActivationStore(xstore)
    xstore.f.close()
    xstore.writeClosed = true
    # The final activation is now durable; the backward phase will persist its
    # incoming gradient after the frozen output stack is differentiated.
  dlog "forward through all blocks done, computing outProj + loss"
  # lm_head is frozen and remains quant-resident. Forward uses its quantized
  # path directly; backwardInputOnly dequantizes only if its resident data is
  # absent, and computes dX without allocating the unused [vocab, hidden] dW.
  # (preset rms-fin): norm_f (RMSNorm, matching the HF base) BEFORE lm_head.
  # This used to call outProj.forward(x) directly on the UN-normalised hidden - a wrong scale
  # accumulated over 64 blocks, one of the sources of chaotic logits.
  var dX: Tensor
  var loss = 0'f32
  # Loss > ln(vocab) = worse than a uniform random guess: the forward/lm_head is broken (e.g. the
  # lm_head random-init bug, loss ~35). STOP before updating/writing anything - this applies
  # to the loss/dX read back on resume too (an unfinished step produced by an old binary).
  let maxSaneLoss = ln(float32(m.vocab))
  template checkSaneLoss() =
    if loss > maxSaneLoss:
      raise newException(CatchableError,
        &"Loss {loss:.4f} > ln(vocab)={maxSaneLoss:.2f} (worse than random guessing) - the forward/lm_head " &
        "is broken. STOPPING, NO weight updated. Check lm_head (bindQuantBaseToLinears) / the norm; " &
        "if this is an unfinished step from an old binary, delete <ckpt>.inprogress.")
  if canResumeBackward:
    dX = resumedDX
    loss = resumedLoss
    checkSaneLoss()
    dlog &"resume: loss={loss}, starting the backward from layer {resumeNextLayer}"
  else:
    let xNormed = m.rmsF.forward(x, ctx)
    let logits = m.outProj.forward(xNormed, ctx)
    # Uses the MASKED version defined right in this file. test_nimformer.
    # softmaxCrossEntropy stays the ORIGINAL version, untouched.
    let (l, dLogits) = softmaxCrossEntropyMasked(logits, targetsBatch, ctx)
    loss = l
    if loss.classify in {fcNan, fcInf, fcNegInf}:
      raise newException(CatchableError, "Non-finite training loss; optimizer step cancelled")
    checkSaneLoss()
    dlog &"loss={loss}, starting the backward"
    # outProj/embed are NOT trained in codestral mode (frozen, in the same spirit
    # as adamStore not touching them) - outProj's dW/dB are skipped, embed.backward is NOT
    # called (not needed, there is no layer before embed to pass dX to).
    # rmsF.gamma is also part of the base -> FROZEN, only dX needs to flow down.
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
  dlog "starting the backward"

  # ---- BACKWARD: block by block IN REVERSE - merge again, backward, extract LoRA/Adam
  # RIGHT AWAY into a SEPARATE grads table (only 1 block), then free ----
  let firstBackwardLayer = if canResumeBackward: resumeNextLayer else: m.blocks.len - 1
  # Tracks the blocks backpropagated in this step (including blocks done in a previous run,
  # confirmed through layer-progress on resume) - at the end of the step ALL must be present.
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
      # Reload from the SSD BEFORE backwardBlock - the REAL weight values are needed to compute
      # the right dX/gradient (they were freed from RAM at the end of THIS block's
      # forward, see the FORWARD loop above).
      for name in fullPrecNamesByBlock[i]:
        let p = paramByName[name]
        if name in fullPrecWasQuantized: quantFlags[name][] = false
        if name in linearsByName:
          linearsByName[name][].residentReady = false
          linearsByName[name][].residentFp32Ready = false
        p[].data = attnStore.loadW(name, numelShape(p[].shape))
        if auditMode: p[].data = graftInitForStep0(name, p[].data)

    let tbMerge = getMonoTime()
    let xi = xstore.readActivation(i)   # reads EXACTLY 1 activation from the SSD, out of scope after backwardBlock
    var dXBeforeBlock = dX  # gradient entering i, needed unchanged on retry
    dXBeforeBlock.data = cloneFloat32Data(dX.data)
    var blockGrads = initTable[string, Tensor]()   # for this block only, NOT accumulated
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
    dlog &"backward block {i}: backwardBlock done"
    let tbBwd = getMonoTime()
    if i > updateMaxBlock:
      # Only pass dX down; this block's weights/Adam stay unchanged.
      for name in namesByBlock.getOrDefault(i, @[]): restoreBaseOne(m, lo, name)
      for name in fullPrecNamesByBlock.getOrDefault(i, @[]):
        paramByName[name][].data = @[]
        if name in fullPrecWasQuantized: quantFlags[name][] = fullPrecWasQuantized[name]
      if stepIndex >= 0:
        # Resume continues from block i-1 with this dX (no layer-progress for this block).
        saveResumeState(stepIndex, i - 1, "backward", dX, loss)
      blockDone[i] = true   # deliberately only passes dX through (block > updateMaxBlock)
      stderr.writeLine &"[time] bwd block {i} (dX pass-through only): merge+load={dsec(tb0, tbMerge):.1f}s " &
        &"backwardBlock={dsec(tbMerge, tbBwd):.1f}s total={dsec(tb0, getMonoTime()):.1f}s"
      trainProfEnd(&"bwd block {i} (dX pass-through only)", tb0)
      GC_fullCollect()
      if gStopRequested and stepIndex >= 0:
        gTrainStepFullyDone = false
        stderr.writeLine &"[finetune_lora] Stopped safely after backward block {i} (dX pass-through only). " &
          "Run again to continue."
        break
      dec i
      continue

    # NO gaps allowed: every LoRA parameter and every new-block parameter (G attn/fuseGate) of
    # this block MUST have a gradient - checked BEFORE updating anything in the block.
    # (Missing grads for new-block parameters used to be skipped silently.)
    # dA/dB are computed FIRST for the whole block and checked: finite; dB must not be all 0 (dB = 0 =
    # the block got no gradient -> it does not learn, exactly the "empty layer" kind of bug).
    var abGrads = initTable[string, tuple[dA, dB: Tensor]]()
    for name in namesByBlock.getOrDefault(i, @[]):
      var dA, dB: Tensor
      if (name & "::loraA") in blockGrads and (name & "::loraB") in blockGrads:
        # The LoRA-aware kernel already computed dA/dB directly (no full dW).
        dA = blockGrads[name & "::loraA"]
        dB = blockGrads[name & "::loraB"]
      elif name in blockGrads:
        (dA, dB) = gradToAB(blockGrads[name], lo.adapters[name], ctx)
      else:
        raise newException(ValueError,
          &"LoRA adapter '{name}' has no gradient in backward block {i}. " &
           "Stopping: training must not silently continue.")
      let ad0 = lo.adapters[name]
      if dA.shape != ad0.A.shape or dB.shape != ad0.B.shape:
        raise newException(ValueError,
          &"LoRA '{name}' block {i}: grad shape dA={dA.shape} dB={dB.shape} differs from A={ad0.A.shape} B={ad0.B.shape}")
      for v in dA.data:
        if v.classify in {fcNan, fcInf, fcNegInf}:
          raise newException(ValueError, &"LoRA '{name}' block {i}: grad A has NaN/Inf - stopping, no update.")
      var dBNonZero = false
      for v in dB.data:
        if v.classify in {fcNan, fcInf, fcNegInf}:
          raise newException(ValueError, &"LoRA '{name}' block {i}: grad B has NaN/Inf - stopping, no update.")
        if v != 0'f32: dBNonZero = true
      if not dBNonZero:
        raise newException(ValueError,
          &"LoRA '{name}' block {i}: grad B is ALL 0 - the block got no gradient (dX into the block = 0?). " &
           "Stopping, no update (it would become an empty layer).")
      abGrads[name] = (dA, dB)
    for name in fullPrecNamesByBlock.getOrDefault(i, @[]):
      if name notin blockGrads:
        raise newException(ValueError,
          &"New-block parameter '{name}' has no gradient in backward block {i} - " &
           "it would NOT be updated. Stopping training.")
      for v in blockGrads[name].data:
        if v.classify in {fcNan, fcInf, fcNegInf}:
          raise newException(ValueError, &"New-block parameter '{name}' block {i}: grad has NaN/Inf - stopping.")

    if auditMode:
      # Adam step 1 from B = 0: B = -lr * g/(|g|+eps) -> the stored B signs must be OPPOSITE to the signs of
      # the real step 0 gradient. Compared element by element (skipping 0 elements), taking the lowest ratio.
      var agree = 1.0
      for name in namesByBlock.getOrDefault(i, @[]):
        if name notin auditRefB:
          raise newException(ValueError, &"[audit] Stored B of '{name}' is missing")
        let refB = auditRefB[name]
        let g = abGrads[name].dB.data
        if refB.len != g.len:
          raise newException(ValueError, &"[audit] '{name}': stored B has {refB.len} elements, grad {g.len}")
        var nCmp, nSame = 0
        for k in 0 ..< g.len:
          if refB[k] != 0'f32 and g[k] != 0'f32:
            inc nCmp
            if (refB[k] < 0'f32) == (g[k] > 0'f32): inc nSame
        if nCmp == 0:
          raise newException(ValueError, &"[audit] '{name}' block {i}: no element to compare signs")
        agree = min(agree, float(nSame) / float(nCmp))
      stderr.writeLine &"[audit] block {i}: the stored B signs match the real step 0 gradient {agree * 100:.2f}%"
      if agree >= AuditGoodAgree:
        # Correct block: blocks <= i come from the same correct computation (filled/trained top-down
        # contiguously) -> kept as is, the backward STOPS. No state written: resume will recheck this block.
        for name in namesByBlock.getOrDefault(i, @[]): restoreBaseOne(m, lo, name)
        for name in fullPrecNamesByBlock.getOrDefault(i, @[]):
          paramByName[name][].data = @[]
          if name in fullPrecWasQuantized: quantFlags[name][] = fullPrecWasQuantized[name]
        for j in 0 .. i: blockDone[j] = true
        gAuditStopBlock = i
        stderr.writeLine &"[audit] block {i} CORRECT -> keeping blocks 0..{i} unchanged, stopping the backward."
        trainProfEnd(&"bwd block {i} (audit correct)", tb0)
        GC_fullCollect()
        break
      if agree > AuditBadAgree:
        # A block's CORRECT step 0 update is unique (-lr*sign(g) on the exact step 0 batch):
        # agreement < the threshold = not that update -> recompute anyway (no raise that would get stuck forever).
        stderr.writeLine &"[audit] WARNING block {i}: {agree * 100:.2f}% agreement (not the ~50% of " &
          &"a garbage gradient, nor >= {AuditGoodAgree * 100:.0f}%) - recomputing from the real gradient anyway."
      # Wrong block: recompute the step 0 update from scratch (B is 0, G is at its graft values).
      for name in namesByBlock.getOrDefault(i, @[]):
        loState.adamA[name] = newApfAdamState(lo.adapters[name].A.data.len)
        loState.adamB[name] = newApfAdamState(lo.adapters[name].B.data.len)
      for name in fullPrecNamesByBlock.getOrDefault(i, @[]):
        adamStore.init(name, numelShape(paramByName[name][].shape))
      stderr.writeLine &"[audit] block {i} WRONG -> recomputing the step 0 update (Adam reset to 0)."

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
    # CRITICAL: the snapshot MUST be taken BEFORE apfAdamStepFused (both LoRA and full-prec),
    # otherwise the rescue would roll back to the ALREADY UPDATED weights that broke the model!
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
          &"LoRA adapter '{name}' has no gradient in backward block {i}. " &
           "Stopping: training must not silently continue.")
        var ad = lo.adapters[name]
        let (dA, dB) = abGrads[name]   # already computed + checked above
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

    # Uses the paramByName built at the top of the function, does NOT call paramList(m) again.
    # The Adam m/v do not stay in RAM across steps: 1 state is read from the SSD, updated on the GPU,
    # written back to the SSD and freed right away.
    # (a real bug, causing the IndexDefect "container is empty"): parameters ALREADY
    # in lo.adapters ('R' blocks, handled through LoRA in the loop
    # above) MUST be skipped - restoreBaseOne() has just run for those names, making
    # p[].data EMPTY (@[]) since they are quant-resident parameters with no
    # real data in RAM between 2 merges. Without this condition the loop
    # below called apfAdamStepGpu on an EMPTY p[].data while the grad was
    # full size -> indexing an empty array -> a crash right at the first 'R' block
    # met in the backward (exactly as in the real log: block 63).
    for name, rawGrad in blockGrads:
      var grad = rawGrad
      if name in lo.adapters: continue
      if not isNewBlockParameter(m, name): continue
      if name notin paramByName: continue
      if m.blocks[i].kind == bkMambaRealGated:
        clipGGradient(grad, name)
      let p = paramByName[name]
      # numelShape(p[].shape) is used instead of p[].data.len as paramLen -
      # .data.len is STABLE RIGHT NOW (just reloaded above) but goes BACK TO 0
      # after the discard below; the shape is always right whether p[].data is
      # empty or full, keeping adamStore.has/init/load consistent across steps.
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
        # apfAdamStepGpu has just updated p[].data IN PLACE (the NEW weight values) -
        # save them to the SSD right away, then free them from RAM - do NOT leave ~3.5GB around for
        # the next block (exactly the missing piece compared with the 'R' blocks).
        attnStore.saveW(name, p[].data)
        p[].data = @[]
        if name in fullPrecWasQuantized:
          quantFlags[name][] = fullPrecWasQuantized[name]
      GC_fullCollect()

    let tbUpd = getMonoTime()
    # Post-update safety check + rescue for G (gated) and R (pure Mamba) blocks.
    # Runs once per block AFTER the whole Adam loop has written params to SSD.
    # LAYER_PROBE (on by default; disable with BYBY_LAYER_PROBE=0) runs a short probe
    # to catch activation drift / garbage tokens that pure RMS checks miss.
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
      # Bug2: merge LoRA before the probe forward so the check runs on the real weights
      for name in rescueLoraNames:
        if name in lo.adapters: mergeInOne(m, lo, ctx, name)
      var gCheckError = ""
      # The forward output (weights AFTER the update) is shared by the resid/mixer check and
      # LAYER_PROBE: these 2 used to call forwardBlock twice with the same input + the same
      # weights -> identical results, costing one extra forward of the whole block.
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
          # Bug1: check the G block output instead of discarding it
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
          # R/P blocks: residual + MIXER. Mixer = y - x (the SSM/Linear++ contribution
          # to the residual). A 16× residual barely catches hub-hub; the 1.5× mixer is what
          # tightens the Mamba branch (close to the G sidecar's 0.5×, not 4×).
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

        # Bug3+Feature6: a real LAYER_PROBE — compares the post-update output with xstore (the original activations).
        # Placed RIGHT INSIDE the try..except so that when driftRatio exceeds the threshold gCheckError is set,
        # triggering the RESCUE mechanism (roll back weights/LoRA, zero Adam, lower the LR) instead of crashing the whole run!
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
            # R/Mamba: a lower drift threshold (0.08) — 0.25 is too wide for a recurrent SSM.
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
      # Bug2: restore LoRA after the probe forward
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
    dlog &"backward block {i}: adam update + SSD spill done"
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
    # blockGrads goes out of scope right here -> the GC reclaims it, it does NOT leak into the next block
    GC_fullCollect()   # runs for EVERY block, like the forward above
    # Ctrl+C: break only after the accepted layer has committed both progress
    # and the next dX; rescue retries never mark a layer completed.
    if gStopRequested:
      gTrainStepFullyDone = false
      stderr.writeLine &"[finetune_lora] Stopped safely after backward layer {i} " &
        "(resume state + layer progress are on the SSD). Run again to continue."
      break
    dec i

  if gTrainStepFullyDone:
    # NO gaps allowed: a step only counts as done when EVERY block has been backpropagated.
    var missing: seq[int] = @[]
    for j in 0 ..< m.blocks.len:
      if not blockDone[j]: missing.add j
    if missing.len > 0:
      raise newException(ValueError,
        &"[check] step {stepIndex}: {missing.len} blocks NOT backpropagated ({missing}) - the step does NOT count as done.")
    stderr.writeLine &"[check] step {stepIndex}: all {m.blocks.len}/{m.blocks.len} blocks backpropagated."
  # the defer closeAndDelete() at the top of the function cleans up the SSD scratch both on a normal return
  # and on an exception.
  result = loss

const FillStepIndex* = 0   # the fill run recomputes step 0

proc fillMarkerPath*(ckptPath: string): string =
  ## Present in <ckpt>.inprogress/ = a fill is in progress (content: K).
  checkpoint.progressDir(ckptPath) / "fill-step0.k"

proc auditMarkerPath*(ckptPath: string): string =
  ## Present in <ckpt>.inprogress/ = the step 0 update is being checked/fixed (auditStep0LoraBlocks).
  checkpoint.progressDir(ckptPath) / "audit-step0"

proc step0RepairInProgress*(ckptPath: string): bool =
  ## The progress in <ckpt>.inprogress belongs to a step 0 fill/check (stepIndex = FillStepIndex),
  ## NOT a regular training step.
  fileExists(fillMarkerPath(ckptPath)) or fileExists(auditMarkerPath(ckptPath))

proc fillUntrainedLoraBlocks*(m: var NimformerModelJamba, lo: var JambaLora,
                              loState: var LoraTrainState, adamStore: var AdamDiskStore,
                              attnStore: var AttnWeightDiskStore, newBlockNames: seq[string],
                              bpeTok: BPETokenizer, hfTok: HfTokenizer, usingHfTok: bool,
                              ctx: Backend, ckptStep: int): bool =
  ## FILLS the empty (B = 0) LoRA blocks of a committed but unfinished checkpoint (the
  ## old FORCE commit bug: step 0 stopped after backward block 51 but still wrote step 1).
  ## Before step 0 every B = 0 -> the step 0 forward/backward = the original model, so temporarily
  ## setting the trained blocks' B back to 0 + taking the EXACT step 0 batch again recomputes the right
  ## gradient; Adam (step 0 -> 1) is only applied to blocks 0..K (K = the highest block with
  ## B = 0), blocks > K only pass dX through and are then restored intact.
  ## Returns true if a fill happened (the caller must commit the checkpoint).
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
      # The .bin already holds the fill result (crash after writing the .bin, before cleanup) -> clean up the fill progress.
      echo "[fill] The .bin already has all 128 adapters; cleaning up leftover fill progress."
      checkpoint.clearLayerProgress(ckpt)
      removeActivationResume(FillStepIndex)
    return false
  for bi in 0 .. k:
    if not zeroBlock[bi]:
      raise newException(ValueError,
        &"[fill] Block {bi} has B != 0 but block {k} has B = 0: not a step interrupted top-down. Stopping.")
  if ckptStep != 1:
    raise newException(ValueError,
      &"[fill] Checkpoint step={ckptStep} has blocks 0..{k} with B = 0. Only the first step can be reproduced correctly " &
      "(step 1, every B = 0 before the step) - stopping to avoid skewed training.")
  echo &"[fill] Blocks 0..{k} were never backpropagated (B = 0) -> FILL: running the remaining step 0 backward for blocks {k}..0, " &
       &"blocks {k + 1}..{m.blocks.len - 1} unchanged"

  # RESUME: the fill runs with stepIndex = FillStepIndex so it shares the
  # forward/backward resume + layer-progress mechanism of regular training. The marker file
  # (holding K) tells fill progress apart from the progress of a regular training step.
  let resuming = fileExists(marker)
  if resuming:
    let savedK = readFile(marker).strip
    if savedK != $k:
      raise newException(IOError,
        &"[fill] The fill progress records K={savedK} but the current checkpoint has B = 0 in blocks 0..{k}. " &
        &"Delete {checkpoint.progressDir(ckpt)} to fill again from scratch.")
  else:
    # Drop all old progress/activations of step FillStepIndex (not part of this fill).
    checkpoint.clearLayerProgress(ckpt)
    removeActivationResume(FillStepIndex)
    createDir(checkpoint.progressDir(ckpt))
    # Atomic: a crash midway does not leave an empty marker (next time K="" -> raise).
    writeFile(marker & ".tmp", $k)
    if cRename((marker & ".tmp").cstring, marker.cstring) != 0:
      raise newException(IOError, "[fill] Could not write the marker " & marker)
  if resuming: recoverPreparedResumeState(FillStepIndex, m.blocks.len, lo)
  var curStep, curLayer: int
  let hasCursor = resuming and checkpoint.readProgressCursor(ckpt, curStep, curLayer) and
    curStep == FillStepIndex
  if hasCursor and (curLayer < 0 or curLayer > k):
    raise newException(IOError,
      &"[fill] The unfinished fill cursor at block {curLayer} is outside 0..{k}. Delete {checkpoint.progressDir(ckpt)} to fill again.")
  # Blocks doneFrom..k were filled in a previous run: their weights (layer-progress) + Adam
  # (SSD, loaded by main) are KEPT, not reset/updated again.
  let doneFrom = if hasCursor: curLayer else: k + 1
  if resuming:
    echo &"[fill] TIEP TUC lap do: " &
      (if hasCursor: &"blocks {curLayer}..{k} already filled, blocks {curLayer - 1}..0 remain"
       else: "no block filled yet (resuming the forward/backward from the SSD state)")

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
      adamStore.init(name, numelShape(byNameP[name][].shape))   # never updated -> step 0
    if bi < doneFrom or bi > k:
      # The FP32 sidecar of G blocks not filled in this run (and blocks > K) MUST equal the .bin:
      # an earlier interrupted fill (including an old binary) may have updated it -> a 2nd update.
      # Rewritten from the .bin values (G attn/fuseGate are stored as fp32 raw so nothing is lost).
      let p = byNameP[name]
      let plen = numelShape(p[].shape)
      if p[].data.len == plen:
        attnStore.saveW(name, p[].data)
      elif name in lo.quantBase:
        attnStore.saveW(name, dequantizeTensor(lo.quantBase[name]))
      else:
        raise newException(IOError, &"[fill] No .bin value for '{name}' to reset the sidecar")

  var loss = 0'f32
  if hasCursor and curLayer == 0:
    # Old binaries could write cursor=0 on a RESCUE rollback. Detect the
    # ambiguous legacy state before treating this as a completed backward.
    var backwardDone, canResume: bool
    discard layerProgressStepState(FillStepIndex, curLayer, m.blocks.len,
      backwardDone, canResume)
    # The backward reached block 0 in a previous run (not committed yet): NOT run again,
    # only the filled weights are loaded from layer-progress.
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
      stderr.writeLine "[fill] Stopped safely - the fill progress is on the SSD (" & checkpoint.progressDir(ckpt) &
        "). Run again to continue the fill, NOTHING done is lost."
      quit(130)
  for name, b in trainedB:
    lo.adapters[name].B.data = b
  var stillZero: seq[string] = @[]
  for name, ad in lo.adapters:
    if allZero(ad.B.data): stillZero.add(name)
  if stillZero.len > 0:
    raise newException(ValueError,
      &"[fill] {stillZero.len} adapters still have B = 0 after the fill (RESCUE rollback?): {stillZero[0]}... " &
      &"NOT committing. See the RESCUE log; delete {checkpoint.progressDir(ckpt)} to fill again from scratch.")
  echo &"[fill] Fill done: step 0 loss={loss:.4f}, {lo.adapters.len}/{lo.adapters.len} adapters have B != 0"
  result = true

proc auditStep0Core*(m: var NimformerModelJamba, lo: var JambaLora,
                     loState: var LoraTrainState, adamStore: var AdamDiskStore,
                     attnStore: var AttnWeightDiskStore,
                     idsB, tgtMaskB: seq[seq[int]], ctx: Backend, ckptStep: int): bool =
  ## AUTOMATICALLY checks + fixes the step 0 update of a step 1 checkpoint without the
  ## meta[Step0AuditMetaKey] flag (e.g. blocks 51..63 of the 21/10 training run: step 0 computed through
  ## a random lm_head). Recomputes the CORRECT step 0 (every B = 0, G blocks = graft values, the step 0
  ## batch, the real lm_head), backward from the top block down: a block whose stored B signs
  ## do not match the real gradient -> reset Adam + update again; at the first matching block ->
  ## stop (blocks below it are kept). Resumable (audit marker + layer progress).
  ## Returns true -> the caller commits (.bin with the flag + Adam + commit marker).
  if ckptStep != 1: return false
  if getEnv("BYBY_STEP0_AUDIT", "1") == "0":
    echo "[audit] BYBY_STEP0_AUDIT=0 -> skipping the step 0 update check."
    return false
  let ckpt = getCheckpointPath()
  let marker = auditMarkerPath(ckpt)
  let resuming = fileExists(marker)
  if resuming: recoverPreparedResumeState(FillStepIndex, m.blocks.len, lo)
  if gLoadedCkptMeta.getOrDefault(Step0AuditMetaKey) == "ok":
    if resuming:
      # A previous run already wrote the .bin WITH the flag (crash after saveCheckpointLoraBin, before cleanup):
      # the check is done. Continuing would compare the NEW B with itself -> raise forever. Only clean up the progress.
      echo "[audit] The .bin already has the step 0 confirmation flag - cleaning up leftover check progress."
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
        &"[audit] '{name}' still has B = 0 - it must be filled (fillUntrainedLoraBlocks) before the step 0 check.")
  if resuming:
    echo "[audit] TIEP TUC kiem/sua update step 0 do (" & marker & ")."
  else:
    echo "[audit] The step 1 checkpoint has no confirmed step 0 update -> recomputing step 0 " &
         "(forward through 64 blocks + backward from the top block to the first correct block)."
    checkpoint.clearLayerProgress(ckpt)
    removeActivationResume(FillStepIndex)
    createDir(checkpoint.progressDir(ckpt))
    writeFile(marker & ".tmp", "1")
    if cRename((marker & ".tmp").cstring, marker.cstring) != 0:
      raise newException(IOError, "[audit] Could not write the marker " & marker)
  var curStep, curLayer: int
  let hasCursor = resuming and checkpoint.readProgressCursor(ckpt, curStep, curLayer) and
    curStep == FillStepIndex
  if hasCursor and (curLayer < 0 or curLayer >= m.blocks.len):
    raise newException(IOError,
      &"[audit] Cursor {curLayer} outside 0..{m.blocks.len - 1}. Delete {checkpoint.progressDir(ckpt)} to check again.")

  # The stored B (.bin) is used to compare signs; then EVERY B = 0 = the pre-step-0 condition. A does not change
  # through step 0 (dA = 0 when B = 0) so it is kept as is.
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
    # A previous run fixed everything down to block 0 (every block was wrong), not committed yet: only load the result.
    discard restoreLayerProgressForStep(lo, ckpt, FillStepIndex, m.blocks.len)
  else:
    let gLrScale = min(1.0'f32, 1.0'f32 / float32(CFG_G_LR_WARMUP_STEPS))
    loss = trainStepGraftStreaming(m, lo, loState, adamStore, attnStore, CFG_NEWBLOCK_LR,
      idsB, tgtMaskB, ctx, gLrScale, stepIndex = FillStepIndex,
      updateMaxBlock = m.blocks.len - 1, auditRefB = refB)
    if gStopRequested and not gTrainStepFullyDone:
      stderr.writeLine "[audit] Stopped safely - the progress is on the SSD (" & checkpoint.progressDir(ckpt) &
        "). Run again to continue, NOTHING done is lost."
      quit(130)
    stopBlock = gAuditStopBlock
  # Blocks <= stopBlock: correct -> restore the stored B. Blocks > stopBlock: just fixed (new B).
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
          &"[audit] '{name}' (block {bi}) still has B = 0 after the fix (RESCUE rollback?) - NOT committing. " &
          &"Run again to continue fixing this block.")
      if nb == b:
        raise newException(ValueError,
          &"[audit] '{name}' (block {bi}) was marked wrong but the new B is identical to the old B - NOT committing.")
      inc nFixed
  if stopBlock == m.blocks.len - 1:
    echo &"[audit] Every block has a correct step 0 update (step 0 loss={loss:.4f}) - only writing the confirmation flag."
  else:
    echo &"[audit] Done: fixed blocks {stopBlock + 1}..{m.blocks.len - 1} ({nFixed} adapters), " &
         (if stopBlock >= 0: &"blocks 0..{stopBlock} correct (kept)" else: "no block was correct beforehand") &
         &", loss step 0={loss:.4f}"
  gStep0Verified = true
  result = true

proc auditStep0LoraBlocks*(m: var NimformerModelJamba, lo: var JambaLora,
                           loState: var LoraTrainState, adamStore: var AdamDiskStore,
                           attnStore: var AttnWeightDiskStore,
                           bpeTok: BPETokenizer, hfTok: HfTokenizer, usingHfTok: bool,
                           ctx: Backend, ckptStep: int): bool =
  ## Like auditStep0Core with EXACTLY the step 0 batch (a new batcher from the start of the dataset, like the fill).
  if ckptStep != 1 or getEnv("BYBY_STEP0_AUDIT", "1") == "0" or
     gLoadedCkptMeta.getOrDefault(Step0AuditMetaKey) == "ok":
    return auditStep0Core(m, lo, loState, adamStore, attnStore, @[], @[], ctx, ckptStep)
  var auditBatcher = newStreamingBatcher(CFG_DATASETS, CFG_SEQ_LEN, CFG_BATCH_SIZE,
                                         bpeTok, hfTok, usingHfTok)
  let (idsB, _, tgtMaskB) = auditBatcher.nextBatch()
  auditStep0Core(m, lo, loState, adamStore, attnStore, idsB, tgtMaskB, ctx, ckptStep)

proc checkLoraCheckpointIntegrity*(m: var NimformerModelJamba, lo: JambaLora, step: int,
                                   newBlockNames: seq[string]): seq[int] =
  ## THOROUGHLY checks the LoRA + G block state just loaded (resume) / just created (new run) BEFORE
  ## training/filling. Raises on any problem that would break training: missing adapters, wrong sizes,
  ## bad alpha/scale, NaN/Inf, an all-zero A (grad B = g.(A x)^T = 0 -> the adapter can NEVER
  ## learn), missing G block weights. Returns the blocks with B = 0 in EVERY adapter
  ## (EMPTY blocks) - the fillUntrainedLoraBlocks mechanism fills them.
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
        problems.add &"{name}: A/B have the wrong size"
        blockEmpty = false
        continue
      if ad.scale.classify in {fcNan, fcInf, fcNegInf} or ad.scale != ad.alpha / float32(ad.r):
        problems.add &"{name}: invalid alpha/scale"
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
        problems.add &"{name}: A is all 0 - the adapter can NEVER learn"
      if bNonZero or bBad: blockEmpty = false
      if not bNonZero and not bBad: zeroBNames.add name
    if blockEmpty: result.add i
    elif step >= 1 and zeroBNames.len > 0:
      # A half-empty block: the fill mechanism only fills blocks empty in BOTH adapters -> it would get stuck at save time.
      problems.add &"block {i}: adapter {zeroBNames} has B = 0 but the other adapter is trained " &
        "(a half-empty block - the fill mechanism cannot fill it)"
  var nNew = 0
  for name in newBlockNames:
    if name notin byName:
      problems.add "missing new-block parameter " & name
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
      problems.add name & ": no weight (neither RAM nor quantBase)"
  var emptyDesc = "no empty blocks"
  if result.len > 0:
    var contiguous = true
    for j in 1 ..< result.len:
      if result[j] != result[j - 1] + 1: contiguous = false
    emptyDesc = &"{result.len} EMPTY blocks (B = 0): " &
      (if contiguous: &"{result[0]}..{result[^1]}" else: $result)
  echo &"[check] step={step}: {nAdapters} LoRA adapters, {nNew}/{newBlockNames.len} new-block parameters, " &
       &"{emptyDesc}, {problems.len} problems"
  if problems.len > 0:
    for p in problems[0 ..< min(10, problems.len)]: stderr.writeLine "[check]   " & p
    raise newException(ValueError,
      &"[check] Checkpoint/LoRA is broken ({problems.len} problems, e.g.: {problems[0]}) - NOT training.")
  if step == 0 and result.len == m.blocks.len:
    echo "[check] New run: every B = 0 is normal at step 0."
  elif result.len > 0:
    echo "[check] -> the fill mechanism (fillUntrainedLoraBlocks) will fill the empty blocks before training continues."

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
      "disable with BYBY_LAYER_PROBE=0)"
  if getEnv("BYBY_G_GATE_STD_RESCUE", "0") != "0":
    echo &"[finetune_lora] G gate-std rescue on: min std={CFG_G_GATE_BIAS_MIN_STD:.1e}"

  # newBackend() defaults to "auto": backend.nim.resolveBackendKind RAISES an error
  # if NO GPU is found (CUDA/Metal/OpenCL/Vulkan) on the machine - it never
  # silently drops to the CPU (gForbidCpuFallback=true by default) - as required:
  # "must be 100% GPU". To force a specific backend: newBackend("cuda"/"metal"/...).
  # Allow CPU fallback for OOM recovery on memory-constrained GPUs.
  setForbidCpuFallback(false)
  let ctx = newBackend()

  # FAIL-FAST: initialise the GPU + JIT RIGHT AWAY (before loading the model, which takes ~hours). Before,
  # initMicroVmJit/warmupMicroVmJit ran AFTER loading the weights -> the "[metal_jit] gDevice nil" error
  # only surfaced after ~2 hours. The device must be initialised FIRST (metal_available only probes the GPU, it does not set gDevice).
  ensureTgCap(ctx)
  initMicroVmJit("finetune_lora", ctx.kind.toByby())
  warmupMicroVmJit(12)

  var m: NimformerModelJamba
  var oldStep = 0
  var quantResidentWeights = initTable[string, QuantTensor]()  # ONLY used TEMPORARILY to build lo -
                                                                  # .clear()ed right after
                                                                  # lo.quantBase takes it,
                                                                  # NO redundant copy is kept
                                                                  # for the whole program (see below).
  var loFromResume: JambaLora
  var resumed = false

  # NEW: AUTO-RESUME - if <CFG_CHECKPOINT_PATH>.lora.json ALREADY EXISTS (e.g. from
  # a previous run KILLED midway by system RAM contention), it
  # AUTOMATICALLY continues from there instead of training again FROM SCRATCH. However many times the machine is killed,
  # at most 1 unfinished epoch is lost (CFG_CHECKPOINT_EVERY decides
  # the save frequency). ONLY applies to CFG_INIT_MODE=="codestral" (the
  # "graft" mode reads its own .bin, unrelated to this LoRA checkpoint).
  if CFG_INIT_MODE == "codestral" and fileExists(getCheckpointPath()):
    echo &"[finetune_lora] FOUND a binary checkpoint ({getCheckpointPath()}) - " &
         "RESUMING AUTOMATICALLY instead of training again from scratch."
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
        "[resume] There is a step>0 checkpoint and .resume.inprogress but no commit marker. " &
        "Cannot prove the model/Adam/dataset cursor are at the same step; stopping safely to avoid a skewed resume.")
    if recoveryHasCommit and oldStep != recoveryStep:
      let backupPath = getCheckpointPath() & ".prev"
      if not fileExists(backupPath):
        raise newException(IOError,
          &"[resume] Checkpoint step={oldStep} does not match commit step={recoveryStep}, " &
           "there is a .resume.inprogress but no rollback backup. Stopping to avoid training on skewed state.")
      let (prevModel, prevLora, prevStep) = loadCheckpointLoraBin(backupPath)
      if prevStep != recoveryStep:
        raise newException(IOError,
          &"[resume] Rollback backup step={prevStep} does not match commit step={recoveryStep}; " &
           "keeping the files and stopping safely.")
      let restoreTmp = getCheckpointPath() & ".restore.tmp"
      copyFile(backupPath, restoreTmp)
      if cRename(restoreTmp.cstring, getCheckpointPath().cstring) != 0:
        try: removeFile(restoreTmp)
        except OSError: discard
        raise newException(IOError, "[resume] Could not atomically roll back the checkpoint from .prev")
      m = prevModel
      loFromResume = prevLora
      oldStep = prevStep
      echo &"[resume] Restored checkpoint step={oldStep} from the commit marker; the Adam/activation sidecars will be rechecked."
    if m.pattern != CFG_CODESTRAL_PATTERN:
      raise newException(ValueError,
        &"Checkpoint pattern '{m.pattern}' does not match the parallel pattern '{CFG_CODESTRAL_PATTERN}'. " &
        "An old checkpoint that replaced R with A cannot be resumed; choose a new checkpoint path to train the parallel sidecar.")
    echo &"[finetune_lora] Resume tu epoch={oldStep}, pattern='{m.pattern}' ({m.blocks.len} block)"
    if fileExists(getCheckpointPath() & ".hf_tokenizer_dir"):
      let savedTokDir = readFile(getCheckpointPath() & ".hf_tokenizer_dir").strip
      if savedTokDir.len > 0 and savedTokDir != CFG_HF_DIR:
        raise newException(ValueError,
          &"[resume] Checkpoint tokenizer sidecar='{savedTokDir}' does not match CFG_HF_DIR='{CFG_HF_DIR}'. Stopping now, not training.")

  if not resumed:
   case CFG_INIT_MODE
   of "codestral":
    echo &"[finetune_lora] CFG_INIT_MODE=codestral - building a NEW model, pattern='{CFG_CODESTRAL_PATTERN}'"
    m = newNimformerModelJamba(CFG_CODESTRAL_VOCAB, CFG_CODESTRAL_HIDDEN, CFG_CODESTRAL_NHEADS,
                                ffMult = 4, pattern = CFG_CODESTRAL_PATTERN, useSession = true,
                                mambaRealDState = CFG_CODESTRAL_DSTATE,
                                mambaRealNGroups = CFG_CODESTRAL_NGROUPS,
                                mambaRealConvKernel = CFG_CODESTRAL_CONVKERNEL,
                                mambaRealExpand = CFG_CODESTRAL_EXPAND,
                                quantResident = (CFG_HF_SOURCE == "nimq"))
    echo &"[finetune_lora] Loading the pretrain weights (source={CFG_HF_SOURCE}) into the 'R' blocks ..."
    case CFG_HF_SOURCE
    of "nimq":
      # QUANT-RESIDENT: inProj/outProj of the 'R' blocks STAY COMPRESSED (the
      # ~29GB of fp32 is not all dequantized at load) - dequantized ON DEMAND through the LUT whenever
      # 1 block is needed, freed right after use (see trainStepGraftStreaming). REQUIRED
      # on RAM-limited machines (e.g. 16GB).
      quantResidentWeights = loadCodestralQuantResidentFromNimq(m, CFG_NIMQ_PATH)
      syncRmsFromBaseNorms(m)
    of "safetensors":
      echo "[finetune_lora] WARNING: the safetensors source dequantizes ALL weights " &
           "to fp32 at load (~29GB) - NO streaming possible (only .nimq " &
           "has compressed data to dequantize on demand). On RAM-limited machines, switch to the 'nimq' source."
      loadCodestralIntoJamba(m, CFG_HF_DIR)
      syncRmsFromBaseNorms(m)
    else:
      raise newException(ValueError, &"Invalid CFG_HF_SOURCE: '{CFG_HF_SOURCE}' (only 'nimq'/'safetensors' accepted)")
   of "graft":
    echo &"[finetune_lora] CFG_INIT_MODE=graft - reading the ALREADY GRAFTED checkpoint: {CFG_OLD_CHECKPOINT}"
    (m, oldStep) = jamba_hybrid.loadCheckpoint(CFG_OLD_CHECKPOINT)
    echo "[finetune_lora] NOT grafting again (this checkpoint is already the output of graft_attention.nim - " &
         "see the warning at the top of the file, grafting twice would break the architecture)."
   else:
    raise newException(ValueError, &"Invalid CFG_INIT_MODE: '{CFG_INIT_MODE}' (only 'codestral'/'graft' accepted)")
  echo &"[finetune_lora] pattern='{m.pattern}' ({m.blocks.len} blocks), steps already trained={oldStep}"
  if m.pattern.len != m.blocks.len:
    raise newException(ValueError,
      &"[preflight] pattern len={m.pattern.len} but model blocks={m.blocks.len}. Stopping now, not training.")
  if 'P' in m.pattern or 'p' in m.pattern:
    raise newException(ValueError,
      "[preflight] Block P is disabled for finetuning; choose an R/G checkpoint or start a new run.")
  if CFG_INIT_MODE == "codestral":
    if m.vocab != CFG_CODESTRAL_VOCAB or m.embedDim != CFG_CODESTRAL_HIDDEN or m.nHeads != CFG_CODESTRAL_NHEADS:
      raise newException(ValueError,
        &"[preflight] The model shape is not Codestral: vocab={m.vocab}, hidden={m.embedDim}, nHeads={m.nHeads}; " &
         &"expected {CFG_CODESTRAL_VOCAB}/{CFG_CODESTRAL_HIDDEN}/{CFG_CODESTRAL_NHEADS}. Stopping now, not training.")

  # Attach LoRA to ALL 2D weights of the PRETRAINED blocks (old 'M' or new
  # 'R' - both are "already learned, only lightly fine-tuned") - read DIRECTLY from
  # m.pattern. 'A'/'S' blocks (newly grafted, nothing learned yet) get NO LoRA, they are trained
  # in full below - the old logic is KEPT, only extended to 'R'.
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
    # Resume: lo (adapters + quantBase) was already read from the .lora.json file, so
    # attachLora*/attachLoraQuantResident are NOT called again (that would create NEW random-init adapters,
    # wiping the previously trained adapters).
    lo = loFromResume
    # (A REAL LEAK, found on review): after the line above, lo.quantBase has
    # its OWN COPY (Table assignment = deep copy in
    # Nim). loFromResume.quantBase STILL LIVES alongside that copy for the
    # rest of the program (loFromResume is declared in an outer scope
    # and does not go away) - wasting ~3.5GB (the size of the compressed .nimq file)
    # for nothing. Cleared RIGHT HERE - lo.quantBase already has its own
    # copy, loFromResume.quantBase is no longer needed.
    loFromResume.quantBase.clear()
    loFromResume.adapters.clear()
  elif quantResidentWeights.len > 0:
    # nimq source: the 'R' blocks already have exact 1:1 names in quantResidentWeights
    # (see loadCodestralQuantResidentFromNimq) - loraTargets/the regular
    # attachLora are NOT used (they would read t[].data, which is garbage, not yet filled).
    lo = attachLoraQuantResident(m, quantResidentWeights, r = CFG_LORA_RANK,
                                 alpha = CFG_LORA_ALPHA, targetNames = loraTargets)
    # (the loss 35 bug in new runs): the real lm_head only lives in quantBase
    # ("outProj.weight"); m.outProj is still a RANDOM newLinear() (useQuant=false)
    # until it is bound. The resume path (loadCheckpointLoraBin) binds it, the new-run
    # path did not -> the loss/dX of every first step went through a random lm_head
    # (loss ~35 instead of ~2.8), making the LoRA gradients meaningless.
    bindQuantBaseToLinears(m, lo)
    # (A REAL LEAK): attachLoraQuantResident() COPIES quantResidentWeights
    # into lo.quantBase (result.quantBase = quantWeights, see lora.nim) - from
    # here on lo.quantBase is the OFFICIAL copy, quantResidentWeights here is just
    # a REDUNDANT copy whose content is never used again anywhere in the
    # program (every later check uses lo.quantBase.len,
    # see below) - cleared RIGHT AWAY, avoiding ~3.5GB of waste for the rest of the run.
    quantResidentWeights.clear()
  else:
    lo = attachLora(m, loraTargets, r = CFG_LORA_RANK, alpha = CFG_LORA_ALPHA)

  # LoRA PREFLIGHT: each adapter must map to the right 2D weight, and the A/B shapes must match
  # exactly. If not, STOP before the first batch, avoiding training into garbage tensors.
  block:
    var byNameCheck = initTable[string, ptr Tensor]()
    for (name, t) in paramList(m): byNameCheck[name] = t
    var expectedAdapters = initTable[string, bool]()
    for name in loraTargets: expectedAdapters[name] = true
    for name, _ in expectedAdapters:
      if name notin lo.adapters:
        raise newException(ValueError, "[preflight] Missing a required LoRA target: " & name)
    for name, ad in lo.adapters:
      if name notin expectedAdapters:
        raise newException(ValueError, "[preflight] Adapter LoRA ngoai target list: " & name)
      if name notin byNameCheck:
        raise newException(ValueError, "[preflight] LoRA target does not exist in the model: " & name)
      let baseShape = byNameCheck[name][].shape
      if baseShape.len != 2 or baseShape[0] != ad.outF or baseShape[1] != ad.inF:
        raise newException(ValueError,
          &"[preflight] LoRA '{name}' base shape=" & $baseShape &
           &" but adapter={ad.outF}x{ad.inF}, r={ad.r}. Stopping now, not training.")
      if ad.A.shape != @[ad.r, ad.inF] or ad.B.shape != @[ad.outF, ad.r] or
         ad.A.data.len != ad.r * ad.inF or ad.B.data.len != ad.outF * ad.r:
        raise newException(ValueError, "[preflight] LoRA tensor shape/data length is wrong for " & name)
  echo &"[preflight] LoRA adapter mapping OK: {lo.adapters.len} adapter"
  if "outProj.weight" in lo.quantBase and not m.outProj.useQuant:
    raise newException(ValueError,
      "[preflight] lm_head (outProj.weight) is in quantBase but not bound to m.outProj - " &
      "the loss would go through a random-init lm_head. Stopping now, not training.")

  var loState = newLoraTrainState(lo, lr = CFG_LORA_LR)

  # Separate Adam for the parameters of the new blocks (A/S) - ONLY the blocks in
  # newBlockIndices, not touching the M blocks (LoRA-ified) or embed/outProj
  # (kept as is, in the spirit of "freeze as much as possible, only learn what must be learned").
  var adamStore = initAdamDiskStore(CFG_ADAM_DIR)
  let commitOnlyMode = getEnv("BYBY_COMMIT_ONLY", "0") != "0"
  if not resumed and step0RepairInProgress(getCheckpointPath()):
    # Step 0 FILL/CHECK progress exists (computed on an old .bin) but the .bin is gone. If .prev remains ->
    # the .bin was most likely deleted by mistake: do NOT delete the progress (hours of work), stop + explain.
    if fileExists(getCheckpointPath() & ".prev"):
      raise newException(IOError,
        &"[fill] Unfinished step 0 fill/check progress remains ({checkpoint.progressDir(getCheckpointPath())}) but " &
        &"{getCheckpointPath()} is gone. Restore it and run again to continue:\n" &
        &"  cp -p {getCheckpointPath()}.prev {getCheckpointPath()}\n" &
        &"If you really want to train anew from scratch: delete {checkpoint.progressDir(getCheckpointPath())} first.")
    # Neither the .bin nor the .prev remains -> a new run: that progress does NOT belong to this run, and if left
    # the step 0 trainStep would resume into it by mistake. Dropped before anything reads .inprogress.
    echo "[fill] No .bin/.prev checkpoint left but unfinished fill progress remains -> dropping the fill progress, training anew from scratch."
    checkpoint.clearLayerProgress(getCheckpointPath())
    removeActivationResume(FillStepIndex)
  let hasInprogress = dirExists(checkpoint.progressDir(getCheckpointPath()))
  # A step 0 fill/check is in progress (fillUntrainedLoraBlocks / auditStep0LoraBlocks interrupted): the
  # progress is in <ckpt>.inprogress with stepIndex = FillStepIndex, NOT a regular training step.
  var fillInProgress = resumed and step0RepairInProgress(getCheckpointPath())
  if oldStep == 0 and not commitOnlyMode and not hasInprogress:
    # Only clean up Adam on a real new run (no inprogress left to recover).
    adamStore.cleanup()
    adamStore = initAdamDiskStore(CFG_ADAM_DIR)
  elif hasInprogress:
    echo "[finetune_lora] Found inprogress — KEEPING the Adam disk state (no cleanup)."
  var attnStore = initAttnWeightDiskStore(CFG_ATTN_WEIGHTS_DIR)
  if oldStep == 0 and not commitOnlyMode and not hasInprogress:
    # New run: old 'A' weights (if any) left over from a previous run are not used -
    # the bootstrap in trainStepGraftStreaming rewrites them from the current m.
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

  # THOROUGHLY checks LoRA + G blocks before anything else (raises if broken; reports empty blocks).
  discard checkLoraCheckpointIntegrity(m, lo, oldStep, newBlockNames)
  # Recover a crash between per-layer cursor commit and resume-state commit.
  # An unfinished optimizer write without a complete layer snapshot FAILS CLOSED.
  if hasInprogress:
    recoverPreparedResumeState(oldStep, m.blocks.len, lo)
    if oldStep != FillStepIndex and step0RepairInProgress(getCheckpointPath()):
      recoverPreparedResumeState(FillStepIndex, m.blocks.len, lo)

  # RESUME INTEGRITY: the checkpoint weights are not enough; Adam + the cursor must be checked.
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
      echo "[resume] WARNING: no commit marker at the right step -> the FP32 resume sidecar is not trusted."
    elif committedSeqLen != CFG_SEQ_LEN or committedBatch != CFG_BATCH_SIZE or committedPattern != m.pattern:
      raise newException(ValueError,
        &"[resume] Config does not match the checkpoint: step={committedStep}, seqLen={committedSeqLen}, batch={committedBatch}, pattern='{committedPattern}' " &
         &"but the current run has seqLen={CFG_SEQ_LEN}, batch={CFG_BATCH_SIZE}, pattern='{m.pattern}'. NOT TRAINING.")
    elif committedDatasetSig.len > 0 and committedDatasetSig != currentDatasetSig and
         committedDatasetSig != datasetSignatureSizeOnly(CFG_DATASETS):
      raise newException(ValueError,
        "[resume] The dataset file size/signature changed after the checkpoint. NOT skipping/resuming to the old position; " &
        "keeping the current checkpoint and stopping to avoid training on a different corpus.")
    elif committedDatasetSig.len > 0 and committedDatasetSig != currentDatasetSig:
      echo "[resume] The commit uses the legacy path+size signature; continuing with a warning that there is no mtime."
    elif CFG_INIT_MODE == "codestral" and committedTokenizerDir.len > 0 and committedTokenizerDir != CFG_HF_DIR:
      raise newException(ValueError,
        &"[resume] Checkpoint tokenizer directory='{committedTokenizerDir}' does not match the current run='{CFG_HF_DIR}'. NOT TRAINING.")
    else:
      let attnFilesOk = attnStoreAllValid(attnStore, m, newBlockNames)
      if not attnFilesOk:
        echo "[resume] The commit marker is right but the FP32 Attention sidecar is MISSING/has a wrong shape -> the whole sidecar set is untrusted."
      else:
        attnStoreTrusted = true
        if inProgress:
          echo &"[resume] FP32 Attention sidecar OK for the layer resume: step={committedStep}"
        else:
          echo &"[resume] Commit marker + FP32 Attention sidecar OK: step={committedStep}, seqLen={committedSeqLen}, batch={committedBatch}, pattern='{committedPattern}'"

    let loraAdamOk = restoreLoraAdamStates(adamStore, loState, lo, oldStep)
    var newAdamFilesOk = adamStoreAllAtStep(adamStore, m, newBlockNames, oldStep)
    if not newAdamFilesOk and hasInprogress and not fillInProgress:
      # A training step in progress: G blocks updated in this step have Adam at oldStep+1 (correct),
      # which used to be treated as a mismatch -> RESETTING the Adam of all G blocks on every resume.
      newAdamFilesOk = adamStoreAllAtStepOrNext(adamStore, m, newBlockNames, oldStep)
    # If the FP32 sidecar is untrusted, trainStepGraftStreaming will rebuild
    # its weights from the lossy QLoRA checkpoint. Adam moments were computed
    # for the original FP32 masters, so carrying them into the reconstructed
    # weights silently mixes two different optimizer histories.
    let newAdamOk = newAdamFilesOk and
      (newBlockNames.len == 0 or attnStoreTrusted)
    if hasCommit and committedDatasetSig.len == 0:
      echo "[resume] Old commit marker without datasetSig -> only a relative check is possible; this is a legacy checkpoint."
    if CFG_INIT_MODE == "codestral" and hasCommit and committedTokenizerDir.len == 0:
      echo "[resume] Old commit marker without tokenizerDir -> the legacy tokenizer path cannot be verified."
    if loraAdamOk:
      echo &"[resume] LoRA Adam m/v OK -> continuing from step {oldStep}"
    else:
      echo "[resume] LoRA Adam m/v MISSING/at the wrong step -> resetting the LoRA optimizer, NOT the weights."
    if newAdamOk:
      echo &"[resume] New-block Adam m/v OK -> continuing from step {oldStep}"
    elif fillInProgress and attnStoreTrusted:
      # Filled G blocks have a real step 1 Adam, unfilled blocks step 0: the step mismatch
      # is EXPECTED during a fill - a reset would lose the Adam of the filled blocks.
      echo "[fill] Fill in progress: KEEPING the new-block Adam + FP32 sidecar (filled G blocks have real state on the SSD)."
    else:
      echo "[resume] New-block Adam m/v is untrusted (missing/wrong step or untrusted FP32 sidecar) -> " &
           "RESETTING ALL new-block Adam; the weights continue from the portable checkpoint."
      resetAdamParameters(adamStore, m, newBlockNames)
  elif oldStep == 0:
    clearResumeInProgress(getCheckpointPath())

  # Per-layer in-progress files describe an *incomplete* training step. They
  # cannot be replayed safely (re-running backward would apply the optimizer a
  # second time), so a leftover directory is reported and dropped; resume
  # always falls back to the last committed `.bin` plus the SSD optimizer and
  # weight stores.
  # inprogress: KEPT BY DEFAULT. Only deleted AFTER the .bin commit succeeds.
  # Step/cursor mismatch -> load the weights + FORCE commit (nothing deleted, no training lost).
  block inspectLayerProgress:
    var pStep, pLayer: int
    if not fillInProgress and checkpoint.readProgressCursor(getCheckpointPath(), pStep, pLayer):
      if pStep == oldStep:
        echo &"[checkpoint] Resuming step {pStep} from layer {pLayer}; keeping inprogress, continuing the backward."
      else:
        echo &"[checkpoint] inprogress step={pStep} layer={pLayer} (ckpt step={oldStep}) — " &
             "KEEPING the files, LoRA will be loaded + FORCE committed if needed (nothing deleted)."

  var layerProgressUsable = false
  var layerProgressStep, layerProgressLayer: int
  if checkpoint.readProgressCursor(getCheckpointPath(), layerProgressStep, layerProgressLayer):
    layerProgressUsable = true  # always try it; not cleared on a mismatch
    echo &"[checkpoint] layer-progress usable: step={layerProgressStep} layer={layerProgressLayer}"

  if fillInProgress and not attnStoreTrusted:
    # The FP32 weights of filled G blocks only live in the sidecar; if the sidecar is broken the fill
    # progress is no longer consistent -> dropped, filled again from scratch (the sidecar is bootstrapped again from the .bin).
    echo "[fill] The G block FP32 sidecar is untrusted -> DROPPING the unfinished fill progress, filling again from scratch."
    checkpoint.clearLayerProgress(getCheckpointPath())
    removeActivationResume(FillStepIndex)
    fillInProgress = false
  if not attnStoreTrusted and oldStep > 0:
    # The FP32 sidecar may be in the middle of a step -> not used. Bootstrapped from the current
    # QLoRA4 checkpoint; safer than mixing weights/optimizer state of different versions.
    attnStore.cleanupAttn()

  echo &"[finetune_lora] {lo.adapters.len} LoRA adapters, {newBlockNames.len} new-block parameters trained in full"
  echo &"[finetune_lora] activation scratch (spill-to-disk) will be written to: {CFG_SCRATCH_DIR}"
  echo &"[finetune_lora] Adam state swapped to the SSD: {CFG_ADAM_DIR} (m/v/step read 1 param -> GPU -> written to SSD -> RAM freed)"

  var bpeTok: BPETokenizer
  var hfTok: HfTokenizer
  var usingHfTok = false

  case CFG_INIT_MODE
  of "codestral":
    echo "[finetune_lora] CFG_INIT_MODE=codestral -> the REAL tokenizer is REQUIRED " &
         "(getOrTrainBPE is not used - see the explanation in hf_tokenizer.nim), reading it from CFG_HF_DIR ..."
    hfTok = newHfTokenizer(CFG_HF_DIR)   # RAISES clearly if the real tokenizer is not found
    usingHfTok = true
    # This used to only `echo` a warning and KEEP TRAINING. Not allowed.
    # If the vocab differs, the tokenizer produces ids >= the model vocab, and in
    # softmaxCrossEntropy `probs[base + tgt]` with base=(b*T+t)*V lands
    # in the NEXT TIME ROW instead of outside the array -> Nim's bounds check
    # does not catch it -> the gradient is added in the wrong place, SILENTLY, and the loss still looks fine.
    # A training run of many hours would produce a meaningless model with no sign of it.
    if vocabSizeHf(hfTok) != CFG_CODESTRAL_VOCAB:
      raise newException(ValueError,
        &"[finetune_lora] the real tokenizer vocab ({vocabSizeHf(hfTok)}) DIFFERS from " &
        &"CFG_CODESTRAL_VOCAB ({CFG_CODESTRAL_VOCAB}). Fix CFG_CODESTRAL_VOCAB " &
        "to match (and also in infer_lora.nim) then run again. NOT training.")
    echo &"[finetune_lora] real tokenizer OK, vocab={vocabSizeHf(hfTok)}"
  of "graft":
    echo "[finetune_lora] CFG_INIT_MODE=graft -> BPE self-trained on a small first sample of the " &
         "dataset (the whole corpus is NOT read into RAM - only the first few lines via streamTextsForever " &
         "to have enough data for the BPE merges, in the old spirit of a random-init 'M' from scratch)."
    var corpusForBpe = ""
    block bpeSample:
      for t in streamTextsForever(CFG_DATASETS):
        corpusForBpe.add(t); corpusForBpe.add('\n')
        if corpusForBpe.len > 300_000: break bpeSample
    bpeTok = getOrTrainBPE(corpusForBpe, CFG_BPE_MERGES)
    echo &"[finetune_lora] vocab_size={vocabSize(bpeTok)}"
  else: discard   # already raised above, cannot get here

  # (tokenizer sidecar bug): the 2 lines writeFile(.hf_tokenizer_dir)/
  # saveBPE(.bpe.json) used to be at the END of the file, only running AFTER `while epoch <
  # CFG_EPOCHS` had fully exited (CFG_EPOCHS=2000 by default) - if training was stopped
  # midway (Ctrl+C) or ran only a few epochs before trying infer_lora, the .bin
  # existed (saveCheckpointLoraBin saves periodically every CFG_CHECKPOINT_EVERY epochs)
  # but the tokenizer sidecar had NEVER been written -> infer_lora.nim did not
  # find ".hf_tokenizer_dir", fell into the else branch looking for ".bpe.json" (which does not
  # exist for codestral checkpoints) -> an IOError crash "cannot open ...bpe.json".
  # WRITTEN RIGHT HERE instead of at the end: the tokenizer (hfTok/bpeTok) is ready
  # just above, and the training loop has NOT started yet - so EVERY checkpoint
  # produced later (including the FIRST periodic one) has its sidecar
  # from the start, no need to wait for the whole training to finish.
  if usingHfTok:
    # There is NO "save" for the real tokenizer (it is the original HF file, unchanged) -
    # only its PATH is written so infer_lora.nim knows which tokenizer to reopen
    # (see loadBinaryLoraForInfer/loadCheckpointLoraForInfer in infer_lora.nim:
    # it reads "<ckpt>.hf_tokenizer_dir" if present).
    writeFile(getCheckpointPath() & ".hf_tokenizer_dir", CFG_HF_DIR)
    echo &"[finetune_lora] Wrote the tokenizer sidecar: {getCheckpointPath()}.hf_tokenizer_dir -> {CFG_HF_DIR}"
  else:
    saveBPE(bpeTok, getCheckpointPath() & ".bpe.json")
    echo &"[finetune_lora] Wrote the tokenizer sidecar: {getCheckpointPath()}.bpe.json"

  # The threshold used to be `< 128` but the default is EXACTLY 128 - so
  # the warning never fired. Raised to 192 so it really shows when
  # the window is shorter than dolly's p50~130 tokens, reminding the user to raise it to
  # 256 (BYBY_SEQ_LEN=256) if the machine allows.
  if CFG_SEQ_LEN < 192:
    echo &"[finetune_lora] NOTE: CFG_SEQ_LEN={CFG_SEQ_LEN}. Measured on " &
         "databricks-dolly-15k.jsonl: example length p50~130 tokens, p90~490, p99~1500. " &
         &"With a {CFG_SEQ_LEN}-token window MOST training windows contain neither " &
         "'[INST]' nor '</s>', starting mid-sentence with SSM state = 0 - far off " &
         "inference (full prompt + a prefilled state). Raise it to >= 256 when " &
         "RAM/VRAM allows."

  var batcher = newStreamingBatcher(CFG_DATASETS, CFG_SEQ_LEN, CFG_BATCH_SIZE,
                                     bpeTok, hfTok, usingHfTok)
  dlog "batcher created, starting the training loop"

  var epoch = oldStep   # this is the training step, NOT a full dataset pass
  if epoch > 0:
    batcher.skipSteps(epoch)
  block:
    var cs, cl: int
    if not fillInProgress and checkpoint.readProgressCursor(getCheckpointPath(), cs, cl):
      echo &"[resume-layer] phat hien {getCheckpointPath()}.inprogress cursor step={cs} layer={cl}"
      # (the "102/128 adapters B=0" bug): this branch used to FORCE commit step
      # cs+1 wherever the cursor stopped. A step interrupted (Ctrl+C / crash) after
      # backward layer 51 -> committed as a FINISHED step, blocks 0..50 were never
      # backpropagated (B=0, Adam m=v=0 but the step forced to 1), and with
      # CFG_EPOCHS=1 the training loop did not run again. Only commit when the backward has
      # reached block 0; otherwise let the training loop continue step cs - trainStep-
      # GraftStreaming loads the spilled dX + activations + LoRA itself and continues the backward.
      var backwardDone, canResumeStep: bool
      let stepDesc = layerProgressStepState(cs, cl, m.blocks.len, backwardDone, canResumeStep)
      # epoch < CFG_EPOCHS: the training loop must REALLY run step cs again, otherwise
      # the unfinished LoRA loaded below would be committed by checkpoint-final as a finished step.
      canResumeStep = canResumeStep and cs == epoch and epoch < CFG_EPOCHS
      let forcePartial = getEnv("BYBY_FORCE_PARTIAL_COMMIT", "0") != "0"
      if cs != epoch:
        # Progress of a step OTHER than the checkpoint step (a leftover file): applying it would mix the LoRA
        # of 2 different steps and commit the wrong step.
        raise newException(IOError,
          &"[resume-layer] inprogress belongs to step {cs} but the checkpoint is at step {epoch} - " &
          &"NOT applying/committing (mixing 2 steps). Delete {getCheckpointPath()}.inprogress if not needed.")
      if not backwardDone and canResumeStep and not forcePartial:
        echo &"[resume-layer] step {cs} NOT FINISHED ({stepDesc}); NOT committing - " &
             &"continuing the backward from block {cl - 1} down to 0."
      elif not backwardDone and not forcePartial:
        raise newException(IOError,
          &"[resume-layer] step {cs} has not finished the backward ({stepDesc}, ckpt step={epoch}, " &
          &"CFG_EPOCHS={CFG_EPOCHS}) but cannot be resumed (missing dX/activations, wrong step, or " &
          "CFG_EPOCHS has no step left to run). NOT committing this step as finished " &
          &"(blocks {cl - 1}..0 would never be trained). Delete {getCheckpointPath()}.inprogress " &
          "to train the step again, or BYBY_FORCE_PARTIAL_COMMIT=1 to accept an unfinished commit.")
      # Load all the LoRA A/B spilled in inprogress into RAM
      var nApplied = 0
      for li in countdown(m.blocks.len - 1, 0):
        let tensors = checkpoint.loadLayerProgress(getCheckpointPath(), cs, li)
        if tensors.len > 0:
          nApplied += applyLayerProgressToLora(lo, tensors)
      echo &"[resume-layer] loaded {nApplied} LoRA tensors from inprogress step={cs}"
      if not backwardDone and forcePartial:
        stderr.writeLine &"[resume-layer] WARNING: BYBY_FORCE_PARTIAL_COMMIT=1 - committing step {cs} " &
          &"unfinished, blocks {cl - 1}..0 are NOT trained in this step."
      # Only commit right away when the inprogress step finished every layer (cl==0) or the user forced a partial commit
      let autoCommit = getEnv("BYBY_FORCE_CHECKPOINT", "1") != "0" and
        (backwardDone or forcePartial)
      if autoCommit and nApplied > 0:
        let commitStep = max(1, cs + 1)  # training step cs done -> marker = cs+1 (like inc epoch)
        # If cursor layer==0 and resume is "backward_done", step cs can be considered done
        syncLoraAdamFromStore(adamStore, loState, lo)
        forceLoraAdamToStep(loState, lo, commitStep)
        saveLoraAdamStates(adamStore, loState)
        saveCheckpointLoraBin(m, lo, getCheckpointPath(), commitStep, newBlockNames, attnStore)
        writeResumeCommit(getCheckpointPath(), commitStep, CFG_SEQ_LEN, CFG_BATCH_SIZE, m.pattern,
          currentDatasetSig, (if CFG_INIT_MODE == "codestral": CFG_HF_DIR else: ""))
        clearResumeInProgress(getCheckpointPath())
        checkpoint.clearLayerProgress(getCheckpointPath())  # only deleted AFTER a successful commit
        try: removeActivationResume(cs)
        except CatchableError: discard
        echo &"[resume-layer] FORCE commit OK step={commitStep} from inprogress — training continues if epochs remain."
        # the batcher already did skipSteps(epoch) above; only skip the steps just committed.
        if commitStep > epoch:
          batcher.skipSteps(commitStep - epoch)
        epoch = commitStep
  # ── COMMIT-ONLY: no training; load LoRA from .inprogress (if any) + force the Adam step + write the .bin ──
  if getEnv("BYBY_COMMIT_ONLY", "0") != "0":
    if fillInProgress:
      raise newException(ValueError,
        "[COMMIT_ONLY] A fill of empty LoRA blocks is in progress (" & fillMarkerPath(getCheckpointPath()) &
        ") - run normal training (without BYBY_COMMIT_ONLY) to continue the fill, NOT committing it unfinished.")
    let ckpt = getCheckpointPath()
    let progDir = checkpoint.progressDir(ckpt)
    echo &"[COMMIT_ONLY] Tim LoRA weight tai: {progDir}"
    echo &"[COMMIT_ONLY] Adam dir (only m/v, NOT the A/B weights): {CFG_ADAM_DIR}"
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
      echo &"[COMMIT_ONLY] NO directory {progDir}"
    var nApplied = 0
    var stepsTried: seq[int] = @[]
    var cs, cl: int
    let hasCursorCO = checkpoint.readProgressCursor(ckpt, cs, cl)
    if hasCursorCO:
      echo &"[COMMIT_ONLY] cursor step={cs} layer={cl}"
      if cs != epoch:
        raise newException(ValueError,
          &"[COMMIT_ONLY] inprogress belongs to step {cs} but the checkpoint is at step {epoch} - " &
          "NOT committing (mixing 2 steps).")
      stepsTried.add(cs)   # ONLY the cursor's step; other step-* dirs are leftovers, not applied
      var backwardDone, canResumeStep: bool
      let stepDesc = layerProgressStepState(cs, cl, m.blocks.len, backwardDone, canResumeStep)
      if not backwardDone and getEnv("BYBY_FORCE_PARTIAL_COMMIT", "0") == "0":
        raise newException(ValueError,
          &"[COMMIT_ONLY] step {cs} has not finished the backward ({stepDesc}). NOT committing it unfinished " &
          "(blocks not backpropagated would count as trained). " &
          (if canResumeStep: "Run normal training (without BYBY_COMMIT_ONLY) to continue the backward. "
           else: "") &
          "BYBY_FORCE_PARTIAL_COMMIT=1 if you really want an unfinished commit.")
    if dirExists(progDir):
      for kind, path in walkDir(progDir, relative = false):
        if kind != pcDir: continue
        let base = path.extractFilename
        if base.startsWith("step-") and base != &"step-{cs}":
          echo &"[COMMIT_ONLY] SKIPPING {path} (not the cursor's step)"
    for st in stepsTried:
      for li in countdown(m.blocks.len - 1, 0):
        let tensors = checkpoint.loadLayerProgress(ckpt, st, li)
        if tensors.len == 0: continue
        nApplied += applyLayerProgressToLora(lo, tensors)
    echo &"[COMMIT_ONLY] Loaded {nApplied} LoRA tensors from layer-progress."
    if nApplied == 0 and not fileExists(ckpt) and not fileExists(ckpt & ".prev"):
      raise newException(ValueError,
        "[COMMIT_ONLY] No layer-progress left (.bin/.idx per layer). " &
        "The SSD Adam (.nimformers_adam) ONLY stores the optimizer m/v — NOT the LoRA A/B weights. " &
        "An old binary ran clearLayerProgress before a failed commit -> the step 0 weights are lost. " &
        "The only way is to train 1 step again with a new binary (BYBY_FORCE_CHECKPOINT=1 will commit).")
    # Step cs finished its backward -> committed as step cs+1 (like inc epoch); with no
    # progress, only the current checkpoint is rewritten at its own step.
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
    echo &"[COMMIT_ONLY] OK — wrote checkpoint step={commitStep}: {ckpt}"
    quit(0)

  # REQUIRED: fill the empty LoRA blocks before training continues (no separate tool needed).
  block fillHoles:
    var cs, cl: int
    if not resumed or epoch != oldStep:
      break fillHoles
    if not fillInProgress and checkpoint.readProgressCursor(getCheckpointPath(), cs, cl):
      break fillHoles   # a regular training step is unfinished: the [resume-layer] block above handles it
    if fillUntrainedLoraBlocks(m, lo, loState, adamStore, attnStore, newBlockNames,
                               bpeTok, hfTok, usingHfTok, ctx, epoch):
      if not loraAdamStateAllAtStep(loState, lo, epoch):
        forceLoraAdamToStep(loState, lo, epoch)
      if newBlockNames.len > 0 and not adamStoreAllAtStep(adamStore, m, newBlockNames, epoch):
        forceAdamStoreToStep(adamStore, m, newBlockNames, epoch)
      let ckpt = getCheckpointPath()
      # The backup is only secondary: the file may have been deleted/renamed during the fill (hours) -
      # a backup error must NOT lose the fill result (bug: copyFile of a deleted .bin ->
      # OSError -> died before writing the .bin).
      for suffix in ["", ".resume.commit"]:
        if fileExists(ckpt & suffix):
          try: copyFile(ckpt & suffix, ckpt & suffix & ".before_fill")
          except OSError as e:
            stderr.writeLine &"[fill] WARNING: could not back up {ckpt}{suffix}: {e.msg} - still writing the fill result."
        else:
          stderr.writeLine &"[fill] WARNING: {ckpt}{suffix} is no longer on disk - skipping the backup, still writing the fill result."
      saveCheckpointLoraBin(m, lo, ckpt, epoch, newBlockNames, attnStore)
      saveLoraAdamStates(adamStore, loState)
      writeResumeCommit(ckpt, epoch, CFG_SEQ_LEN, CFG_BATCH_SIZE, m.pattern, currentDatasetSig,
        (if CFG_INIT_MODE == "codestral": CFG_HF_DIR else: ""))
      # The fill progress is only deleted AFTER the .bin + Adam + commit marker are written.
      checkpoint.clearLayerProgress(ckpt)
      removeActivationResume(FillStepIndex)
      fillInProgress = false
      echo &"[fill] Wrote {ckpt} step={epoch} (backup: {ckpt}.before_fill)"

  # AUTOMATICALLY checks the step 0 update (after the fill): blocks with a wrong step 0 update (e.g. computed through
  # a random lm_head) -> recomputed; the flag is written into the .bin so it is not checked again next time.
  block step0Audit:
    var cs, cl: int
    if not resumed or epoch != oldStep:
      break step0Audit
    if not step0RepairInProgress(getCheckpointPath()) and
       checkpoint.readProgressCursor(getCheckpointPath(), cs, cl):
      break step0Audit   # a regular training step is unfinished
    if auditStep0LoraBlocks(m, lo, loState, adamStore, attnStore,
                            bpeTok, hfTok, usingHfTok, ctx, epoch):
      if not loraAdamStateAllAtStep(loState, lo, epoch):
        forceLoraAdamToStep(loState, lo, epoch)
      if newBlockNames.len > 0 and not adamStoreAllAtStep(adamStore, m, newBlockNames, epoch):
        forceAdamStoreToStep(adamStore, m, newBlockNames, epoch)
      let ckpt = getCheckpointPath()
      # Keep the .bin from before the fix (hard link, O(1)): .prev CANNOT keep it since the final save right
      # after overwrites it. A backup error must not lose the check result.
      let backup = ckpt & ".before_audit"
      try:
        if fileExists(backup): removeFile(backup)
        if fileExists(ckpt) and cLink(ckpt.cstring, backup.cstring) != 0:
          copyFile(ckpt, backup)
      except OSError as e:
        stderr.writeLine &"[audit] WARNING: could not back up {ckpt}: {e.msg} - still writing the result."
      saveCheckpointLoraBin(m, lo, ckpt, epoch, newBlockNames, attnStore)
      saveLoraAdamStates(adamStore, loState)
      writeResumeCommit(ckpt, epoch, CFG_SEQ_LEN, CFG_BATCH_SIZE, m.pattern, currentDatasetSig,
        (if CFG_INIT_MODE == "codestral": CFG_HF_DIR else: ""))
      checkpoint.clearLayerProgress(ckpt)
      removeActivationResume(FillStepIndex)
      fillInProgress = false
      echo &"[audit] Wrote {ckpt} step={epoch} ({Step0AuditMetaKey}=ok; previous version: {backup})"

  var runningLoss = 0'f32
  var prevStepLoss = -1'f32
  while epoch < CFG_EPOCHS:
    markResumeInProgress(getCheckpointPath(), epoch + 1)
    # `idsB`/`tgtB` only live for 1 loop iteration - out of scope the GC reclaims them
    # RIGHT AWAY, nothing accumulates across steps (unlike the old version keeping the whole seq
    # `batches` from start to end of training).
    dlog &"epoch {epoch}: calling nextBatch()"
    let (idsB, tgtRawB, tgtMaskB) = batcher.nextBatch()
    dlog &"epoch {epoch}: nextBatch done, {idsB.len} samples, calling trainStep"
    if idsB.len == 0:
      echo "[finetune_lora] Empty dataset (no line could be read) - stopping."
      quit(1)
    # LR warmup for G blocks: the scale goes from ~0 -> 1.0 over CFG_G_LR_WARMUP_STEPS steps.
    let gLrScale = min(1.0'f32, (float32(epoch) + 1.0'f32) / float32(CFG_G_LR_WARMUP_STEPS))
    let loss = if lo.quantBase.len > 0:   # uses lo.quantBase (the ONLY source of truth
                                            # left after quantResidentWeights was .clear()ed
                                            # above) instead of quantResidentWeights (now ALWAYS
                                            # empty after the clear - checking it here would be WRONG, always
                                            # falling into the non-streaming trainStepGraft branch).
                 # targets with MASKED boundaries -> softmaxCrossEntropyMasked (local)
                 # stepIndex = epoch: tells trainStepGraftStreaming which step this is so it
                 # writes the per-layer in-progress files (<ckpt>.inprogress/step-<n>/layer-<i>.*).
                 trainStepGraftStreaming(m, lo, loState, adamStore, attnStore,
                   CFG_NEWBLOCK_LR, idsB, tgtMaskB, ctx, gLrScale, stepIndex = epoch)
               else:
                 # this path goes through jamba_hybrid.computeLossAndGrads ->
                 # the ORIGINAL test_nimformer.softmaxCrossEntropy (which does not understand TargetIgnore)
                 # so it MUST get the RAW targets. This path does not mask boundaries.
                 trainStepGraft(m, lo, loState, adamStore, CFG_NEWBLOCK_LR, idsB, tgtRawB, ctx)
    runningLoss += loss
    if gStopRequested and not gTrainStepFullyDone:
      # Unfinished step: the resume/layer progress is on the SSD — epoch NOT incremented, the full ckpt NOT committed.
      stderr.writeLine "[finetune_lora] Exiting safely (step unfinished). Run make finetune-mamba again to resume the layer."
      quit(130)
    if epoch == 0 and gTrainStepFullyDone and gLastStepForwardRan:
      # Step 0: the loss + gradient were computed entirely in this process (lm_head bound, preflight)
      # -> the step 1 .bin writes the confirmation flag, step 0 needs no check next time.
      gStep0Verified = true
    inc epoch
    if gStopRequested:
      # The step is done (backward down to layer 0 or a break after layer 0). Try to commit, then exit.
      stderr.writeLine "[finetune_lora] Ctrl+C after the step finished — trying to commit the checkpoint, then exiting."
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
            stderr.writeLine &"[finetune_lora] Committed checkpoint step={epoch} before exiting."
            quit(130)
      stderr.writeLine "[finetune_lora] Could not commit (Adam/sidecar mismatch); the layer resume is still on the SSD. Exiting."
      quit(130)

    # CLEAN UP after EVERY batch (like a reuse allocator on ESP32: actively
    # reclaim garbage right away, never letting it accumulate into a sudden mem spike at some step
    # later). GC_fullCollect is heavier than GC_step but here RAM SAFETY
    # (weak machines) beats speed - each GPU training step is already the heaviest
    # part, the cost of a full GC here is negligible compared with the OOM risk.
    GC_fullCollect()

    if epoch mod CFG_LOG_EVERY == 0:
      let maskPct = if batcher.nEmitted > 0:
                      100.0 * float(batcher.nMasked) / float(batcher.nEmitted)
                    else: 0.0
      echo &"[epoch {epoch}/{CFG_EPOCHS}] loss={runningLoss / float32(CFG_LOG_EVERY):.4f} " &
           &"(pending_buf={batcher.pending.len} tok, mask ranh gioi doc={maskPct:.2f}%)"
      runningLoss = 0'f32

    if epoch mod CFG_CHECKPOINT_EVERY == 0:
      # PREFLIGHT BEFORE COMMIT: weights/optimizer/step must all be at the same training step.
      # Streaming spilled the LoRA Adam to the SSD per layer — sync RAM from disk before checking.
      syncLoraAdamFromStore(adamStore, loState, lo)
      # Default: the training step is done -> do NOT lose the weights because some Adam files are at step=0.
      # Only BYBY_STRICT_ADAM_STEP=1 raises. BYBY_FORCE_CHECKPOINT=0 disables the force.
      let strictAdam = getEnv("BYBY_STRICT_ADAM_STEP", "0") != "0"
      let forceCkpt = getEnv("BYBY_FORCE_CHECKPOINT", "1") != "0"
      if not loraAdamStateAllAtStep(loState, lo, epoch):
        if forceCkpt and not strictAdam:
          stderr.writeLine &"[checkpoint] FORCE: forcing the LoRA Adam step -> {epoch} then committing (avoids losing a trained step)."
          forceLoraAdamToStep(loState, lo, epoch)
          saveLoraAdamStates(adamStore, loState)
        else:
          raise newException(ValueError,
            &"[checkpoint] LoRA Adam step is out of sync with training step {epoch}. NOT committing the checkpoint.")
      if newBlockNames.len > 0 and not adamStoreAllAtStep(adamStore, m, newBlockNames, epoch):
        if forceCkpt and not strictAdam:
          stderr.writeLine &"[checkpoint] FORCE: forcing the new-block Adam step -> {epoch} then committing"
          forceAdamStoreToStep(adamStore, m, newBlockNames, epoch)
        else:
          raise newException(ValueError,
            &"[checkpoint] New-block Adam state is out of sync with training step {epoch}. NOT committing the checkpoint.")
      if CFG_INIT_MODE == "codestral" and newBlockNames.len > 0 and not attnStoreAllValid(attnStore, m, newBlockNames):
        if forceCkpt and not strictAdam:
          stderr.writeLine &"[checkpoint] FORCE: skipping the FP32 Attention sidecar check step={epoch}"
        else:
          raise newException(ValueError,
            &"[checkpoint] FP32 Attention sidecar is incomplete at step {epoch}. NOT committing the checkpoint.")
      # COMMIT ORDER: checkpoint weights -> LoRA Adam -> commit marker -> delete in-progress.
      saveCheckpointLoraBin(m, lo, getCheckpointPath(), epoch, newBlockNames, attnStore)
      saveLoraAdamStates(adamStore, loState)
      writeResumeCommit(getCheckpointPath(), epoch, CFG_SEQ_LEN, CFG_BATCH_SIZE, m.pattern, currentDatasetSig, (if CFG_INIT_MODE == "codestral": CFG_HF_DIR else: ""))
      clearResumeInProgress(getCheckpointPath())
      # Layer progress is only deleted AFTER the .bin commit succeeds.
      checkpoint.clearLayerProgress(getCheckpointPath())
      removeActivationResume(epoch - 1)
      GC_fullCollect()

    # REQUIRED: the loss went up compared with the previous step -> STOP training to check (this step
    # was committed above / will be committed by checkpoint-final below).
    if prevStepLoss >= 0'f32 and loss > prevStepLoss:
      stderr.writeLine &"[finetune_lora] STOP: the loss went UP {prevStepLoss:.4f} -> {loss:.4f} at step {epoch}. " &
        "Check the data/LR/forward before training further."
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
          &"[checkpoint-final] LoRA Adam step is out of sync with training step {epoch}. NOT committing the checkpoint.")
    if newBlockNames.len > 0 and not adamStoreAllAtStep(adamStore, m, newBlockNames, epoch):
      if forceFinal and not strictFinal:
        stderr.writeLine &"[checkpoint-final] FORCE: ep new-block Adam step -> {epoch}"
        forceAdamStoreToStep(adamStore, m, newBlockNames, epoch)
      else:
        raise newException(ValueError,
          &"[checkpoint-final] New-block Adam state is out of sync with training step {epoch}. NOT committing the checkpoint.")
    if CFG_INIT_MODE == "codestral" and newBlockNames.len > 0 and not attnStoreAllValid(attnStore, m, newBlockNames):
      if forceFinal and not strictFinal:
        stderr.writeLine &"[checkpoint-final] FORCE: skipping the FP32 Attention sidecar check step={epoch}"
      else:
        raise newException(ValueError,
          &"[checkpoint-final] FP32 Attention sidecar is incomplete at step {epoch}. NOT committing the checkpoint.")
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
    echo &"[finetune_lora] Cleaned up the Adam disk cache: {CFG_ADAM_DIR}"
    echo &"[finetune_lora] Cleaned up the Attn-weight disk cache: {CFG_ATTN_WEIGHTS_DIR}"
  else:
    echo "[finetune_lora] KEEPING the resume state on the SSD: Adam + FP32 Attention. No retraining needed on resume."
  # The tokenizer sidecar (.hf_tokenizer_dir / .bpe.json) is written EARLY, right
  # after the tokenizer is ready, BEFORE the training loop (see above) - it is NOT
  # written again here (that caused the "you must finish training first" bug).
  echo "[finetune_lora] DONE."
