## CLI for the mmap-backed training path in test_nimformer.
import std/[md5, monotimes, osproc, parseopt, strtabs, tempfiles, terminal, times]
import mapped_weights

const
  PackedPrefix = "__packed."
  PackedOptimizerKey = PackedPrefix & "optimizer"
  PackedTokensPrefix = PackedPrefix & "trained_tokens."
  PackedTokenizerPrefix = PackedPrefix & "tokenizer."
  PackedExpertStepPrefix = PackedPrefix & "expert_step."
  PackedRetryLimit = 2
  LegacyPackedLr = 1e-3'f32
  PackedPeakLr = 1.2e-3'f32
  DefaultHrmCycles = 2
  DefaultHrmLSteps = 2
  DefaultHrmSegments = 2
  DefaultDraftBlock = 16
  PackedFiles =["code-basic.bin", "code-intermediate.bin", "code-advanced.bin",
                 "english.bin", "val-code.bin", "val-english.bin"]
  PackedUsage = """
Usage: test_nimformer --dataset=DIR [options]
  --checkpoint=PATH        Resume or save packed training (default finetune_packed.nimq,
                           or moe/finetune.nimq with --moe-experts/--merge-experts)
  --resume-from=PATH      Use a packed checkpoint if the output does not exist yet
  --init-checkpoint=PATH   Start a new run with existing Spectron weights
  --init-tokenizer=PATH    Tokenizer for the initial weights (default Spectron BPE)
  --backend=metal         Use the specified backend (default from build)
  --steps=6000            Additional optimizer steps (0 saves without training)
  --batch=16 --seq=128    Batch size and tokens per row
  --lr=0.0012 --warmup=150 --decay-steps=500000 --curriculum-steps=4000
  --beta2=0.98          Adam second-moment decay for a new run
  --grad-clip=1.0 --requantize-every=0
  --eval-every=500 --eval-batches=8 --log-every=100 --checkpoint-every=500
  --checkpoint-hook=PATH  Run this executable with the saved path and step
  --hf-repo=REPO --hf-path=PATH
                          Hugging Face mode: upload every saved checkpoint to this model
                          repo path with `hf upload` while training continues; a save made
                          during an upload goes next and older pending saves are skipped
  --hf-token-file=PATH    Write token for Hugging Face mode (default
                          ~/.env_spectron_token_huggingface); asked for and saved there
                          with mode 600 when missing
  --tensor-cores=require  Verify Vulkan cooperative FP16/FP32 GEMM before training
  --moe-experts=2         Create or grow to this total number of experts. With
                          --init-checkpoint on dense weights, every block's trained
                          feed-forward becomes this many experts (sparse upcycling)
  --moe-hidden=512 --moe-heads=8 --moe-layers=4
  --expert-width=7312 --expert-depth=4
  --merge-experts=PATH    Append experts from a compatible packed MoE checkpoint
  --hrm-cycles=2          HRM high-level cycles; HRM is on by default for a new model
                          (0 starts a new model without HRM)
  --hrm-l-steps=2 --hrm-l-layers=HALF --hrm-segments=2
                          Low-level steps per cycle, blocks in the low-level
                          module, and deep-supervision segments per step
  --draft-block=16        Train for parallel block decoding, on by default when --seq is
  --draft-noise=0.5       at least 16 (0 turns it off): in each block of this many inputs,
                          the first stays clean and each later input is replaced by another
                          token with a per-block rate drawn from U(0, noise)
  --eos-data=DIR          Mix complete-document endings into batches and learn EOS
  --eos-rows=N            Clean ending examples per batch (default one quarter)
  --eos-weight=8         Weight assigned to true EOS targets
  --seed=1234
"""

type
  PackedTrainingOptions = object
    dataset, checkpoint, initialWeights, initialTokenizer, backendName: string
    donor, checkpointHook, tensorCoreMode, resumeFrom, eosData: string
    hfRepo, hfPath, hfTokenFile: string
    steps, batch, context, warmup, decaySteps, curriculumSteps, seed: int
    evalEvery, evalBatches, logEvery, saveEvery, requantizeEvery: int
    experts, moeHidden, moeHeads, moeLayers, expertWidth, expertDepth: int
    hrmCycles, hrmLSteps, hrmLLayers, hrmSegments: int
    draftBlock, eosRows: int
    lr, gradClip, beta2, draftNoise, eosWeight: float32
    lrSet, beta2Set, hrmOff, draftBlockSet: bool

  PackedOptimizer = object
    beta2, peakLr, lrScale: float32

  PackedProgress = object
    tokens: int64
    estimated: bool

  PackedDomain = enum packedCode, packedEnglish
  PackedShard = object
    bytes: MappedView[uint8]
    name: string
    tokens: int

  PackedCorpus = object
    streams: seq[PackedShard]
    training: seq[int]
    validation: array[PackedDomain, seq[int]]
    trainTokens, validationTokens: array[PackedDomain, int64]
    tokenizerHash: string
    bits, vocab: int
    legacy: bool

include eos_training

proc packedOptions(): PackedTrainingOptions =
  result = PackedTrainingOptions(
    initialTokenizer: BPE_TOKENIZER_PATH, backendName: backend.backend,
    steps: CFG_N_STEPS, batch: 16, context: CFG_SEQ_LEN, lr: PackedPeakLr,
    warmup: 150, decaySteps: 500_000, curriculumSteps: 4000, seed: 1234,
    beta2: 0.98, gradClip: 1, evalEvery: 500, evalBatches: 8,
    logEvery: CFG_LOG_EVERY, saveEvery: 500, moeHidden: 512, moeHeads: 8, moeLayers: 4,
    expertWidth: 7312, expertDepth: 4, draftNoise: 0.5, eosWeight: 8)
  for kind, key, val in getopt():
    if key == "help" and kind == cmdLongOption:
      echo PackedUsage
      quit(0)
    if kind != cmdLongOption or val.len == 0:
      raise newException(ValueError, "Expected --name=value. See --help.")
    case key
    of "dataset": result.dataset = val
    of "checkpoint": result.checkpoint = val
    of "resume-from": result.resumeFrom = val
    of "checkpoint-hook": result.checkpointHook = val
    of "hf-repo": result.hfRepo = val
    of "hf-path": result.hfPath = val
    of "hf-token-file": result.hfTokenFile = val
    of "init-checkpoint": result.initialWeights = val
    of "init-tokenizer": result.initialTokenizer = val
    of "moe-experts": result.experts = parseInt(val)
    of "moe-hidden": result.moeHidden = parseInt(val)
    of "moe-heads": result.moeHeads = parseInt(val)
    of "moe-layers": result.moeLayers = parseInt(val)
    of "expert-width": result.expertWidth = parseInt(val)
    of "expert-depth": result.expertDepth = parseInt(val)
    of "merge-experts": result.donor = val
    of "hrm-cycles":
      result.hrmCycles = parseInt(val)
      result.hrmOff = result.hrmCycles == 0
    of "hrm-l-steps": result.hrmLSteps = parseInt(val)
    of "hrm-l-layers": result.hrmLLayers = parseInt(val)
    of "hrm-segments": result.hrmSegments = parseInt(val)
    of "draft-block":
      result.draftBlock = parseInt(val)
      result.draftBlockSet = true
    of "draft-noise": result.draftNoise = float32(parseFloat(val))
    of "eos-data": result.eosData = val
    of "eos-rows": result.eosRows = parseInt(val)
    of "eos-weight": result.eosWeight = float32(parseFloat(val))
    of "backend": result.backendName = val
    of "tensor-cores":
      if val notin ["require", "off"]:
        raise newException(ValueError, "--tensor-cores expects require or off")
      result.tensorCoreMode = val
    of "steps": result.steps = parseInt(val)
    of "batch": result.batch = parseInt(val)
    of "seq": result.context = parseInt(val)
    of "warmup": result.warmup = parseInt(val)
    of "decay-steps": result.decaySteps = parseInt(val)
    of "curriculum-steps": result.curriculumSteps = parseInt(val)
    of "seed": result.seed = parseInt(val)
    of "eval-every": result.evalEvery = parseInt(val)
    of "eval-batches": result.evalBatches = parseInt(val)
    of "log-every": result.logEvery = parseInt(val)
    of "checkpoint-every": result.saveEvery = parseInt(val)
    of "requantize-every": result.requantizeEvery = parseInt(val)
    of "beta2":
      result.beta2 = float32(parseFloat(val))
      result.beta2Set = true
    of "lr":
      result.lr = float32(parseFloat(val))
      result.lrSet = true
    of "grad-clip": result.gradClip = float32(parseFloat(val))
    else: raise newException(ValueError, "Unknown option --" & key)
  if result.dataset.len == 0:
    raise newException(ValueError, "--dataset=DIR is required for packed training")
  if result.resumeFrom.len > 0 and result.initialWeights.len > 0:
    raise newException(ValueError, "Choose either --resume-from or --init-checkpoint")
  if result.checkpointHook.len > 0:
    result.checkpointHook = absolutePath(result.checkpointHook)
    if not fileExists(result.checkpointHook):
      raise newException(ValueError, "Checkpoint hook does not exist: " & result.checkpointHook)
  if (result.hfRepo.len > 0) != (result.hfPath.len > 0):
    raise newException(ValueError, "--hf-repo and --hf-path must be given together")
  if result.hfTokenFile.len == 0:
    result.hfTokenFile = getHomeDir() / ".env_spectron_token_huggingface"
  if result.hfRepo.len > 0 and findExe("hf").len == 0:
    raise newException(ValueError, "Hugging Face mode needs the hf CLI: pip install -U huggingface_hub")
  if result.checkpoint.len == 0:
    result.checkpoint = if result.experts > 0 or result.donor.len > 0:
                          "test/spectron/models/moe/finetune.nimq"
                        else: "test/spectron/models/finetune_packed.nimq"
  for value in [result.batch, result.context, result.decaySteps,
                result.evalEvery, result.evalBatches, result.logEvery, result.saveEvery]:
    if value <= 0: raise newException(ValueError, "Step counts and sizes must be positive")
  if result.steps < 0 or result.experts < 0 or result.moeHidden < 1 or result.moeHeads < 1 or
      result.moeHidden mod result.moeHeads != 0 or result.moeLayers < 1 or
      result.expertWidth < 1 or result.expertDepth < 1:
    raise newException(ValueError, "Invalid step count or MoE dimensions")
  if min([result.hrmCycles, result.hrmLSteps, result.hrmLLayers, result.hrmSegments]) < 0:
    raise newException(ValueError, "HRM settings must be positive")
  if result.hrmOff and max([result.hrmLSteps, result.hrmLLayers, result.hrmSegments]) > 0:
    raise newException(ValueError, "--hrm-cycles=0 turns HRM off; remove the other --hrm options")
  if not result.draftBlockSet:
    result.draftBlock = if result.context >= DefaultDraftBlock: DefaultDraftBlock else: 0
  if result.draftBlock < 0 or result.draftBlock == 1 or result.draftBlock > result.context:
    raise newException(ValueError, "--draft-block must be 0 (off) or between 2 and --seq")
  if not finite32(result.draftNoise) or result.draftNoise < 0 or result.draftNoise > 1:
    raise newException(ValueError, "--draft-noise must be in [0, 1]")
  if result.eosRows < 0 or result.eosRows > result.batch or not finite32(result.eosWeight) or
      result.eosWeight < 1:
    raise newException(ValueError, "Invalid EOS row count or loss weight")
  if result.warmup < 0 or result.curriculumSteps < 0 or result.requantizeEvery < 0 or
      result.warmup >= result.decaySteps or
      result.lr.classify notin {fcNormal, fcSubnormal} or result.lr <= 0 or
      result.gradClip.classify in {fcNan, fcInf, fcNegInf} or result.gradClip < 0:
    raise newException(ValueError, "Invalid learning rate, schedule, or gradient clip")
  if not finite32(result.beta2) or result.beta2 < 0 or result.beta2 >= 1:
    raise newException(ValueError, "--beta2 must be in [0, 1)")

proc addShard(corpus: var PackedCorpus, path, name: string, count: int64, context: int,
              domain: PackedDomain, validation: bool) =
  if count <= context or count > int.high or count > (int64.high - 7) div corpus.bits or
      name.len == 0 or name.extractFilename() != name or name in [".", ".."]:
    raise newException(ValueError, "Invalid packed shard: " & name)
  let byteCount = (count * corpus.bits + 7) div 8
  let file = path / name
  if byteCount > int.high or getFileSize(file) != byteCount:
    raise newException(ValueError, "Invalid packed shard size: " & file)
  let i = corpus.streams.len
  if validation:
    corpus.validation[domain].add(i)
    corpus.validationTokens[domain] += count
  else:
    corpus.training.add(i)
    corpus.trainTokens[domain] += count
  corpus.streams.add(PackedShard(
    name: name, tokens: int(count), bytes: mappedView[uint8](mapWeights(file), 0, int(byteCount))))

proc openCorpus(path: string, context: int): PackedCorpus =
  let manifest = parseFile(path / "manifest.json")
  case manifest["format"].getStr()
  of "spectron-u16-v1":
    result.bits = 16
    result.legacy = true
  of "spectron-u8-v2":
    result.bits = manifest["bits_per_token"].getInt()
  else:
    raise newException(ValueError, "Unsupported packed dataset format")
  let tokenizerPath = path / "tokenizer.json"
  result.tokenizerHash = getMD5(readFile(tokenizerPath))
  result.vocab = loadBPE(tokenizerPath).vocabSize
  if result.tokenizerHash != manifest["tokenizer_md5"].getStr() or
      result.vocab != manifest["vocab_size"].getInt() or result.bits notin 8 .. 16 or
      result.vocab notin 256 .. (1 shl result.bits):
    raise newException(ValueError, "Dataset tokenizer does not match its manifest")
  if result.legacy:
    for i, name in PackedFiles:
      let domain = if i in [3, 5]: packedEnglish else: packedCode
      result.addShard(path, name, manifest["tokens"][name].getBiggestInt(), context, domain, i >= 4)
  else:
    for entry in manifest["streams"]:
      let domain = case entry["domain"].getStr()
        of "code": packedCode
        of "english": packedEnglish
        else: raise newException(ValueError, "Invalid corpus domain")
      let split = entry["split"].getStr()
      if split notin ["train", "validation"]:
        raise newException(ValueError, "Invalid corpus split")
      result.addShard(path, entry["file"].getStr(), entry["tokens"].getBiggestInt(),
                      context, domain, split == "validation")
    if result.trainTokens[packedEnglish] != manifest["english_tokens"].getBiggestInt() or
        result.trainTokens[packedCode] != manifest["code_tokens"].getBiggestInt():
      raise newException(ValueError, "Corpus split totals disagree with the manifest")
  for domain in PackedDomain:
    if result.trainTokens[domain] <= 0 or result.validationTokens[domain] <= 0:
      raise newException(ValueError, "Each domain needs training and validation tokens")

proc sampleShard(corpus: PackedCorpus, choices: openArray[int], total: int64, rng: var Rand): int =
  var pick = rng.rand(0'i64 .. total - 1)
  for shard in choices:
    if pick < corpus.streams[shard].tokens: return shard
    pick -= corpus.streams[shard].tokens
  raise newException(ValueError, "Invalid corpus sampling weights")

proc readWindow(corpus: PackedCorpus, stream, context: int, rng: var Rand,
                ids, targets: var seq[int]) =
  let shard = corpus.streams[stream]
  let start = rng.rand(shard.tokens - context - 1)
  let source = shard.bytes.dataPtr
  let mask = (1'u32 shl corpus.bits) - 1
  var bit = int64(start) * corpus.bits
  ids.setLen(context)
  targets.setLen(context)
  var previous = 0
  for t in 0 .. context:
    let offset = int(bit shr 3)
    let shift = int(bit and 7)
    var raw = uint32(source[offset])
    if shift + corpus.bits > 8: raw = raw or (uint32(source[offset + 1]) shl 8)
    if shift + corpus.bits > 16: raw = raw or (uint32(source[offset + 2]) shl 16)
    let id = int((raw shr shift) and mask)
    bit += corpus.bits
    if id >= corpus.vocab:
      raise newException(ValueError, "Invalid token ID in " & shard.name)
    if t > 0:
      ids[t - 1] = previous
      targets[t - 1] = id
    previous = id

proc corruptDrafts(ids: var seq[int], blockSize: int, noise: float32, vocab: int,
                   rng: var Rand): int =
  ## Mimics a pass of parallel block decoding. The window is cut into blocks of
  ## `blockSize` inputs at a random phase. A block's first input plays the last
  ## verified token and stays clean; each later input plays a draft and is
  ## replaced by a different token with probability r, where r ~ U(0, noise) is
  ## drawn per block. Targets keep the true next tokens. Returns the number of
  ## replaced inputs.
  var start = rng.rand(blockSize - 1) - blockSize
  while start < ids.len:
    let rate = rng.rand(float64(noise))
    for i in max(start + 1, 0) ..< min(start + blockSize, ids.len):
      if rng.rand(1.0) < rate:
        ids[i] = (ids[i] + 1 + rng.rand(vocab - 2)) mod vocab
        inc result
    start += blockSize

proc validationLoss(model: NimformerModelSpectron, corpus: PackedCorpus,
                    options: PackedTrainingOptions, domain: PackedDomain, ctx: Backend): float64 =
  var rng = initRand(918277 + ord(domain))
  var ids = newSeq[seq[int]](options.batch)
  var targets = newSeq[seq[int]](options.batch)
  for _ in 1 .. options.evalBatches:
    for row in 0 ..< options.batch:
      let choices = corpus.validation[domain]
      let stream = if choices.len == 1: choices[0]
                   else: corpus.sampleShard(choices, corpus.validationTokens[domain], rng)
      corpus.readWindow(stream, options.context, rng, ids[row], targets[row])
    result += float64(softmaxCrossEntropy(model.forwardBatch(ids, ctx), targets, ctx).loss)
  result /= float64(options.evalBatches)
  if result.classify in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "Non-finite validation loss")

proc learningRate(options: PackedTrainingOptions, peakLr: float32, step: int): float32 =
  if step <= options.warmup:
    return peakLr * (float32(step) / float32(options.warmup))
  let progress = min(1.0, float64(step - options.warmup) /
                      float64(options.decaySteps - options.warmup))
  peakLr * float32(0.1 + 0.45 * (1.0 + cos(PI * progress)))

proc codeLevel(options: PackedTrainingOptions, step: int, rng: var Rand): int =
  let progress = if options.curriculumSteps == 0: 1.0
                 else: min(1.0, float64(step - 1) / float64(options.curriculumSteps))
  let roll = rng.rand(1.0)
  if roll < 0.65 - 0.30 * progress: 0
  elif roll < 0.9 - 0.20 * progress: 1
  else: 2

template visitAdam(adam, globalStep, expertSteps, paramName, state, stateStep, body: untyped) =
  template visit(key, value, stepValue: untyped) =
    block:
      let paramName {.inject.} = key
      let stateStep {.inject, used.} = stepValue
      template state: untyped = value
      body
  for fieldName, field in fieldPairs(adam):
    when field is ApfAdamState:
      visit(fieldName, field, globalStep)
  for i in 0 ..< adam.blocks.len:
    for fieldName, field in fieldPairs(adam.blocks[i]):
      visit("block" & $i & "." & fieldName, field, globalStep)
  if adam.mixture.experts.len > 0:
    for fieldName, field in fieldPairs(adam.mixture):
      when field is ApfAdamState:
        visit("moe." & fieldName, field, globalStep)
    for i in 0 ..< adam.mixture.experts.len:
      if adam.mixture.experts[i] == nil: continue
      for j in 0 ..< adam.mixture.experts[i].layers.len:
        for fieldName, field in fieldPairs(adam.mixture.experts[i].layers[j]):
          visit(expertLayerKey(i, j) & fieldName, field, expertSteps[i])
  for i in 0 ..< adam.blockMoe.len:
    visit("block" & $i & ".moe.routerW", adam.blockMoe[i].routerW, globalStep)
    visit("block" & $i & ".moe.routerB", adam.blockMoe[i].routerB, globalStep)
    for e in 0 ..< adam.blockMoe[i].experts.len:
      for fieldName, field in fieldPairs(adam.blockMoe[i].experts[e]):
        visit("block" & $i & ".moe.ff" & $(e + 1) & "." & fieldName, field, globalStep)

proc savePacked(path: string, model: NimformerModelSpectron, adam: ModelAdamState,
                optimizer: PackedOptimizer, progress: PackedProgress, step: int, tokenizerHash: string) =
  var sd = model.modelToStateDict()
  sd.add((MetaStepKey, quantizeTensor(@[float32(step)], @[1], qkFp32Raw)))
  sd.add((PackedTokenizerPrefix & tokenizerHash & "." & $step,
          quantizeTensor(@[0'f32], @[1], qkFp32Raw)))
  sd.add((PackedOptimizerKey,
          quantizeTensor(@[optimizer.beta2, optimizer.lrScale, optimizer.peakLr], @[3], qkFp32Raw)))
  # Store the token count in the name to preserve the exact integer past 2^24.
  sd.add((PackedTokensPrefix & $progress.tokens,
          quantizeTensor(@[float32(ord(progress.estimated))], @[1], qkFp32Raw)))
  var expertSteps = newSeq[int](adam.mixture.experts.len)
  for i, expert in adam.mixture.experts:
    if expert == nil: continue
    expertSteps[i] = expert.layers[0].upW.step
    sd.add((PackedExpertStepPrefix & $i & "." & $expertSteps[i],
            quantizeTensor(@[0'f32], @[1], qkFp32Raw)))
  visitAdam(adam, step, expertSteps, name, state, stateStep):
    sd.add((PackedPrefix & name & ".m", quantizeTensor(state.m, @[state.m.len], qkFp32Raw)))
    sd.add((PackedPrefix & name & ".v", quantizeTensor(state.v, @[state.v.len], qkFp32Raw)))
  let dir = if path.parentDir.len > 0: path.parentDir else: "."
  createDir(dir)
  let (file, temporary) = createTempFile(".spectron-", ".nimq", dir)
  file.close()
  defer: removeFile(temporary)
  saveQuantStateDict(temporary, model.checkpointArch(), sd)
  moveFile(temporary, path)

proc resumePacked(path, tokenizerHash: string):
    tuple[model: NimformerModelSpectron, adam: ModelAdamState, optimizer: PackedOptimizer,
          progress: PackedProgress, step: int] =
  let (arch, sd, _) = loadQuantStateDict(path, mapped = true)
  var tensors = initTable[string, QuantTensor]()
  let marker = PackedTokenizerPrefix & tokenizerHash & "."
  var savedExpertSteps = initTable[int, int]()
  var step = -1
  result.progress.tokens = -1
  for (name, tensor) in sd:
    if name.startsWith(marker):
      step = parseInt(name[marker.len .. ^1])
    elif name.startsWith(PackedTokensPrefix):
      let tokens = parseBiggestInt(name[PackedTokensPrefix.len .. ^1])
      let flags = dequantizeTensor(tensor)
      if tokens < 0 or flags.len != 1 or flags[0] notin [0'f32, 1'f32]:
        raise newException(ValueError, "Invalid packed training token count")
      result.progress = PackedProgress(tokens: tokens, estimated: flags[0] == 1)
    elif name.startsWith(PackedExpertStepPrefix):
      let parts = name[PackedExpertStepPrefix.len .. ^1].split('.')
      if parts.len != 2:
        raise newException(ValueError, "Invalid expert optimizer step")
      let index = parseInt(parts[0])
      if index in savedExpertSteps:
        raise newException(ValueError, "Duplicate expert optimizer step")
      savedExpertSteps[index] = parseInt(parts[1])
    elif name.startsWith(PackedPrefix):
      tensors[name] = tensor
  if step < 0:
    raise newException(ValueError,
      "Checkpoint is missing packed training state for this tokenizer. Use --init-checkpoint with a new output path to start from its weights.")
  result.model = stateDictToModel(sd, arch[0], arch[1], arch[2], arch[3], arch[4])
  result.adam = newModelAdamState(result.model)
  result.step = step
  # Checkpoints created before beta2 was configurable used Adam's old default.
  result.optimizer = PackedOptimizer(beta2: 0.999, peakLr: LegacyPackedLr, lrScale: 1)
  if PackedOptimizerKey in tensors:
    let settings = dequantizeTensor(tensors[PackedOptimizerKey])
    if settings.len notin 2 .. 3 or not finite32(settings[0]) or settings[0] < 0 or settings[0] >= 1 or
        not finite32(settings[1]) or settings[1] <= 0 or settings[1] > 1 or
        (settings.len == 3 and (not finite32(settings[2]) or settings[2] <= 0)):
      raise newException(ValueError, "Invalid packed optimizer settings")
    result.optimizer.beta2 = settings[0]
    result.optimizer.lrScale = settings[1]
    if settings.len == 3: result.optimizer.peakLr = settings[2]
  var expertSteps = newSeq[int](result.model.mixture.experts.len)
  for index, expertStep in savedExpertSteps:
    if index notin 0 ..< expertSteps.len or expertStep notin 1 .. step:
      raise newException(ValueError, "Invalid expert optimizer progress")
    expertSteps[index] = expertStep
    result.adam.mixture.experts[index] = newExpertAdam(result.model.mixture.experts[index])
  visitAdam(result.adam, step, expertSteps, name, state, stateStep):
    let mKey = PackedPrefix & name & ".m"
    let vKey = PackedPrefix & name & ".v"
    if mKey notin tensors or vKey notin tensors:
      if not name.startsWith("hrmNorm") or result.model.hrmEnabled:
        raise newException(ValueError, "Missing Adam state for " & name)
    else:
      let m = dequantizeTensor(tensors[mKey])
      let v = dequantizeTensor(tensors[vKey])
      if m.len != state.m.len or v.len != state.v.len:
        raise newException(ValueError, "Invalid Adam state for " & name)
      for i in 0 ..< m.len:
        if not finite32(m[i]) or not finite32(v[i]) or v[i] < 0:
          raise newException(ValueError, "Invalid Adam moment in checkpoint for " & name)
      state.m = m
      state.v = v
    state.step = stateStep

proc appendExperts(model: var NimformerModelSpectron, adam: var ModelAdamState,
                   extra: openArray[SpectronExpert]) =
  let count = model.mixture.experts.len + extra.len
  var router = newRouter(model.embed.dim, count)
  for i, value in model.mixture.router.weight.data:
    router.weight.data[i] = value
  for i, value in model.mixture.router.bias.data:
    router.bias.data[i] = value
  model.mixture.router = move(router)
  model.mixture.experts.add(extra)
  adam.mixture.routerW.m.setLen(count * model.embed.dim)
  adam.mixture.routerW.v.setLen(count * model.embed.dim)
  adam.mixture.routerB.m.setLen(count)
  adam.mixture.routerB.v.setLen(count)
  adam.mixture.experts.setLen(count)

proc loadDonorExperts(path, tokenizerHash: string, model: NimformerModelSpectron): seq[SpectronExpert] =
  let (arch, sd, _) = loadQuantStateDict(path, mapped = true)
  let marker = PackedTokenizerPrefix & tokenizerHash & "."
  var matchesTokenizer = false
  for (name, _) in sd:
    if name.startsWith(marker):
      matchesTokenizer = true
      break
  if not matchesTokenizer or arch[1] != model.embed.dim:
    raise newException(ValueError, "Expert donor checkpoint uses a different tokenizer or hidden size")
  let donor = stateDictToModel(sd, arch[0], arch[1], arch[2], arch[3], arch[4])
  if not donor.matchesBpe(model.vocab - ord(model.hasEos)):
    raise newException(ValueError, "Expert donor uses a different tokenizer vocabulary")
  if donor.mixture.experts.len == 0 or donor.mixture.width != model.mixture.width or
      donor.mixture.depth != model.mixture.depth:
    raise newException(ValueError, "Expert donor checkpoint uses a different expert layout")
  result = donor.mixture.experts

proc huggingFaceToken(path: string): string =
  ## Reads the write token from `path`, or asks for it and saves it there.
  if fileExists(path): result = readFile(path).strip()
  if result.len > 0: return
  if not stdin.isatty:
    raise newException(ValueError, "No Hugging Face token in " & path)
  result = readPasswordFromStdin("Hugging Face write token (hidden): ").strip()
  if result.len == 0:
    raise newException(ValueError, "No Hugging Face token entered")
  if path.parentDir.len > 0: createDir(path.parentDir)
  writeFile(path, result & "\n")
  setFilePermissions(path, {fpUserRead, fpUserWrite})
  echo "[hf] Saved the token to ", path

proc trainPacked() =
  let options = packedOptions()
  # Hugging Face mode: the token only reaches the `hf upload` child process.
  var hfEnv: StringTableRef
  var uploader: Process
  var pendingUpload = -1
  let snapshot = options.checkpoint & ".uploading"
  if options.hfRepo.len > 0:
    hfEnv = newStringTable(modeCaseSensitive)
    for key, value in envPairs(): hfEnv[key] = value
    hfEnv["HF_TOKEN"] = huggingFaceToken(options.hfTokenFile)
    echo &"[hf] Uploading saves to {options.hfRepo}/{options.hfPath}"
  proc startUpload(step: int) =
    removeFile(snapshot)
    # The trainer replaces the checkpoint by rename, so the link stays a stable snapshot.
    try: createHardlink(options.checkpoint, snapshot)
    except OSError: copyFile(options.checkpoint, snapshot)
    uploader = startProcess("hf", args = ["upload", options.hfRepo, snapshot, options.hfPath,
      "--quiet", "--commit-message", &"Spectron checkpoint at step {step}"],
      env = hfEnv, options = {poUsePath, poParentStreams})
    echo &"[hf] Upload of step {step} started"
  proc pollUpload(wait = false) =
    if uploader != nil:
      let code = if wait: uploader.waitForExit() else: uploader.peekExitCode()
      if code != -1:
        echo &"[hf] Upload finished with exit code {code}"
        uploader.close()
        uploader = nil
        removeFile(snapshot)
    if uploader == nil and pendingUpload >= 0:
      startUpload(pendingUpload)
      pendingUpload = -1
  defer:
    while uploader != nil or pendingUpload >= 0: pollUpload(wait = true)
  let corpus = openCorpus(options.dataset, options.context)
  randomize(options.seed)
  var model: NimformerModelSpectron
  var adam: ModelAdamState
  var optimizer = PackedOptimizer(beta2: options.beta2, peakLr: options.lr, lrScale: 1)
  var progress: PackedProgress
  var start = 0
  proc saveProgress(step: int) =
    savePacked(options.checkpoint, model, adam, optimizer, progress, step, corpus.tokenizerHash)
    echo &"[packed] Saved {options.checkpoint} at step {step}"
    if hfEnv != nil:
      pendingUpload = step  # An older pending save is skipped.
      pollUpload()
    if options.checkpointHook.len > 0:
      let process = startProcess(options.checkpointHook,
        args = [absolutePath(options.checkpoint), $step], options = {poParentStreams})
      defer: process.close()
      let status = process.waitForExit()
      if status != 0:
        raise newException(IOError, &"Checkpoint hook exited with status {status} at step {step}")
  let resume = if fileExists(options.checkpoint): options.checkpoint else: options.resumeFrom
  if resume.len > 0:
    if options.initialWeights.len != 0:
      raise newException(ValueError, "--init-checkpoint requires a new --checkpoint path")
    (model, adam, optimizer, progress, start) = resumePacked(resume, corpus.tokenizerHash)
    if options.beta2Set and options.beta2 != optimizer.beta2:
      raise newException(ValueError,
        "Resuming Adam requires its saved beta2. Use --init-checkpoint to start a new optimizer.")
    if options.lrSet: optimizer.peakLr = options.lr
    echo &"[packed] Resuming at step {start} with saved Adam moments"
  else:
    if options.initialWeights.len > 0:
      if getMD5(readFile(options.initialTokenizer)) != corpus.tokenizerHash:
        raise newException(ValueError, "Initial checkpoint tokenizer differs from the dataset tokenizer")
      model = loadCheckpoint(options.initialWeights).model
      model.useFinalNorm = true
      echo &"[packed] Initialized weights from {options.initialWeights}"
    elif options.experts > 0:
      model = newNimformerModelSpectron(corpus.vocab, options.moeHidden, options.moeHeads,
                                       options.moeLayers, CFG_FF_MULT)
    else:
      model = newNimformerModelSpectron(corpus.vocab, CFG_EMBED_DIM, CFG_N_HEADS,
                                       CFG_N_LAYERS, CFG_FF_MULT)
    if options.experts > 0 and model.expertCount == 0:
      if options.initialWeights.len > 0:
        # Sparse upcycling: every block's trained feed-forward becomes the experts, so
        # the MoE model starts with exactly the outputs of the dense weights.
        model.upcycleExperts(options.experts)
        echo &"[moe] Upcycled the feed-forward of every block into {options.experts} experts " &
             &"copied from {options.initialWeights}"
      else:
        model.mixture = newMixture(model.embed.dim, options.experts, options.expertWidth, options.expertDepth)
    adam = newModelAdamState(model)
  if not model.matchesBpe(corpus.vocab) or not model.useFinalNorm:
    raise newException(ValueError, "Incompatible Spectron training checkpoint")
  let eosPath = if options.eosData.len > 0: options.eosData
                elif model.hasEos: options.dataset / "eos" else: ""
  var eosCorpus: EosCorpus
  var eosRows = 0
  if eosPath.len > 0:
    eosCorpus = loadEosCorpus(eosPath, corpus)
    if not model.hasEos: extendWithEos(model, adam)
    eosRows = if options.eosRows > 0: options.eosRows else: max(1, options.batch div 4)
    echo &"[eos] token_id={model.eosTokenId} clean_rows={eosRows}/{options.batch} " &
         &"target_weight={options.eosWeight} data={eosPath}"
  # HRM is on by default for a new model. A resumed checkpoint keeps its own core.
  let newModel = resume.len == 0
  let hrmRequested = options.hrmCycles > 0 or options.hrmLSteps > 0 or
                     options.hrmLLayers > 0 or options.hrmSegments > 0
  if options.hrmOff and model.hrmEnabled:
    raise newException(ValueError, "--hrm-cycles=0 cannot remove the HRM core from these weights")
  if hrmRequested or (newModel and not options.hrmOff):
    if not newModel and not model.hrmEnabled:
      raise newException(ValueError,
        "This checkpoint has no HRM core. Use --init-checkpoint with a new --checkpoint path to add one.")
    if not newModel and options.hrmLLayers > 0 and options.hrmLLayers != model.hrm.lLayers:
      raise newException(ValueError, "--hrm-l-layers differs from the checkpoint's HRM split")
    if not model.hrmEnabled:
      model.hrm = HrmConfig(cycles: DefaultHrmCycles, lSteps: DefaultHrmLSteps,
                            segments: DefaultHrmSegments, lLayers: max(1, model.blocks.len div 2))
      # Random blocks start near identity so the recurrence contracts; trained
      # --init-checkpoint weights are kept as they are.
      if options.initialWeights.len == 0:
        model.initHrmResiduals()
        echo &"[hrm] scaled residual branch outputs by {HrmResidualInitScale} for a new HRM core"
    # Cycles, steps and segments only change compute, so a resumed run may adjust them.
    if options.hrmCycles > 0: model.hrm.cycles = options.hrmCycles
    if options.hrmLSteps > 0: model.hrm.lSteps = options.hrmLSteps
    if options.hrmLLayers > 0: model.hrm.lLayers = options.hrmLLayers
    if options.hrmSegments > 0: model.hrm.segments = options.hrmSegments
  elif not model.hrmEnabled and not options.hrmOff:
    echo "[hrm] off: this checkpoint has no HRM core; --init-checkpoint with a new " &
         "--checkpoint path adds one"
  model.validateHrm()
  if model.hrmEnabled:
    let h = model.hrm
    echo &"[hrm] l_layers={h.lLayers} h_layers={model.blocks.len - h.lLayers} cycles={h.cycles} " &
         &"l_steps={h.lSteps} segments={h.segments} " &
         &"block_passes_per_segment={h.cycles * (h.lSteps * h.lLayers + model.blocks.len - h.lLayers)} " &
         "gradient=one-step"
  let existingExperts = model.expertCount
  if options.experts > 0 and options.experts != existingExperts:
    if existingExperts == 0:
      raise newException(ValueError, "Use --init-checkpoint with a new output path to turn dense weights into experts")
    if model.blockExpertCount > 0:
      raise newException(ValueError,
        &"--moe-experts={options.experts} differs from the {existingExperts} experts in every block of this checkpoint")
    if options.experts < existingExperts:
      raise newException(ValueError, "--moe-experts is smaller than the checkpoint's expert count")
    var extra: seq[SpectronExpert]
    for _ in existingExperts ..< options.experts:
      extra.add(newExpert(model.embed.dim, model.mixture.width, model.mixture.depth))
    appendExperts(model, adam, extra)
  if options.donor.len > 0:
    if model.mixture.experts.len == 0 or
        (fileExists(options.checkpoint) and sameFile(options.checkpoint, options.donor)):
      raise newException(ValueError, "Merging experts requires separate target and donor MoE checkpoints")
    let extra = loadDonorExperts(options.donor, corpus.tokenizerHash, model)
    appendExperts(model, adam, extra)
  if model.mixture.experts.len > 0:
    let params = expertParameterCount(model.embed.dim, model.mixture.width, model.mixture.depth)
    echo &"[moe] experts={model.mixture.experts.len} parameters_per_expert={params} routing=top1"
  elif model.blockExpertCount > 0:
    let ff = model.blocks[0].ff
    let params = ff.fc1.weight.data.len + ff.fc1.bias.data.len + ff.fc2.weight.data.len +
                 ff.fc2.bias.data.len
    echo &"[moe] experts={model.blockExpertCount} in each of {model.blocks.len} blocks " &
         &"parameters_per_expert={params} routing=top1"
  if int64(options.batch) > int64.high div int64(options.context):
    raise newException(ValueError, "Packed batch token count is too large")
  let tokensPerStep = int64(options.batch) * int64(options.context)
  if progress.tokens < 0:
    if int64(start) > int64.high div tokensPerStep:
      raise newException(ValueError, "Packed training token count is too large")
    progress = PackedProgress(tokens: int64(start) * tokensPerStep, estimated: start > 0)
  if start > int.high - options.steps or int64(options.steps) > (int64.high - progress.tokens) div tokensPerStep:
    raise newException(ValueError, "Packed training step or token count is too large")
  let codeTokens = corpus.trainTokens[packedCode]
  let englishTokens = corpus.trainTokens[packedEnglish]
  let tokensPerEpoch = codeTokens + englishTokens
  let startTokens = progress.tokens
  let remaining = tokensPerEpoch - startTokens mod tokensPerEpoch
  let finalStep = start + options.steps
  echo &"[packed] code={codeTokens} english={englishTokens} vocab={corpus.vocab} " &
       &"batch={options.batch} context={options.context}"
  echo &"[packed] tokens_per_epoch={tokensPerEpoch} tokens_per_step={tokensPerStep} " &
       &"epoch={startTokens div tokensPerEpoch} tokens_total={startTokens} " &
       &"steps_to_next_epoch={(remaining - 1) div tokensPerStep + 1}"
  let estimate = if progress.estimated: " epoch_estimated=true" else: ""
  if progress.estimated:
    echo "[packed] Historical token count was estimated using the resume batch and context"
  echo &"[packed] beta2={optimizer.beta2:.3f} peak_lr={optimizer.peakLr:.3e} " &
       &"lr_scale={optimizer.lrScale:.4g} requantize_every={options.requantizeEvery}"
  if options.steps == 0:
    saveProgress(start)
    return
  if options.tensorCoreMode == "require":
    putEnv("BYBY_VULKAN_TENSOR_CORES", "force")
  elif options.tensorCoreMode == "off":
    putEnv("BYBY_VULKAN_TENSOR_CORES", "0")
    putEnv("BYBY_FINETUNE_ACCELERATED_GEMM", "0")
  var ctx = newBackend(options.backendName)
  defer: closeBackend(ctx)
  if options.tensorCoreMode == "require":
    ctx.enableTrainingTensorCores()
    echo "[packed] Tensor Core probe passed (Vulkan FP16 inputs, FP32 accumulation)"

  proc evaluate(step: int) =
    let codeLoss = validationLoss(model, corpus, options, packedCode, ctx)
    let englishLoss = validationLoss(model, corpus, options, packedEnglish, ctx)
    echo &"[val] step={step} epoch={progress.tokens div tokensPerEpoch} " &
         &"code_loss={codeLoss:.4f} english_loss={englishLoss:.4f} " &
         &"ppl={exp((codeLoss + englishLoss) / 2.0):.2f}" & estimate
    if eosRows > 0: evalEos(model, eosCorpus, options, step, ctx)

  evaluate(start)
  discard ctx.takeTrainingGemmStats()
  var ids = newSeq[seq[int]](options.batch)
  var targets = newSeq[seq[int]](options.batch)
  var lossSum, seconds, peakGradNorm, balanceSum = 0.0
  var routes = newSeq[int64](model.expertCount)
  var updates, clipped = 0
  var draftReplaced, draftInputs = 0
  var eosTargets = 0
  if options.draftBlock > 0:
    echo &"[draft] block={options.draftBlock} noise={options.draftNoise}: per-block replacement " &
         &"rate ~ U(0, {options.draftNoise}), first input of each block kept clean"
  for step in start + 1 .. finalStep:
    if hfEnv != nil: pollUpload()
    let before = getMonoTime()
    var stats: SpectronTrainingStep
    var lr: float32
    var stepReplaced = 0
    var stepEos = 0
    for attempt in 0 .. PackedRetryLimit:
      var rng = initRand(options.seed.int64 xor (step.int64 shl 1) xor (attempt.int64 shl 48))
      stepReplaced = 0
      stepEos = 0
      for row in 0 ..< options.batch:
        if row < eosRows:
          let domain = PackedDomain((row + step) mod 2)
          stepEos += eosCorpus.train[domain].eosWindow(options.context, rng, ids[row], targets[row])
          continue
        let stream =
          if not corpus.legacy: corpus.sampleShard(corpus.training, tokensPerEpoch, rng)
          elif (row + step) mod 2 == 0: options.codeLevel(step, rng)
          else: 3
        corpus.readWindow(stream, options.context, rng, ids[row], targets[row])
        if options.draftBlock > 0:
          stepReplaced += corruptDrafts(ids[row], options.draftBlock, options.draftNoise,
                                        corpus.vocab, rng)
      lr = options.learningRate(optimizer.peakLr, step) * optimizer.lrScale
      if lr <= 0:
        raise newException(ValueError, "Packed learning rate rounded to zero")
      try:
        stats = checkedTrainStep(model, adam, ids, targets, ctx, lr, optimizer.beta2,
                                 options.gradClip, options.requantizeEvery, options.eosWeight)
        break
      except SpectronNumericalError as error:
        if attempt == PackedRetryLimit or lr * 0.5'f32 == 0:
          if step > start + 1:
            try:
              evaluate(step - 1)
              saveProgress(step - 1)
              stderr.writeLine(&"[packed] Saved last finite update at step {step - 1}")
            except SpectronNumericalError as validationError:
              stderr.writeLine("[packed] Could not validate the last update: " & validationError.msg)
          raise
        optimizer.lrScale *= 0.5'f32
        stderr.writeLine(&"[packed] Rejected step {step}: {error.msg}. Retrying with lr_scale={optimizer.lrScale:.4g}")
    seconds += float64((getMonoTime() - before).inNanoseconds) * 1e-9
    if options.draftBlock > 0:
      draftReplaced += stepReplaced
      draftInputs += (options.batch - eosRows) * options.context
    eosTargets += stepEos
    lossSum += float64(stats.loss)
    balanceSum += stats.balanceLoss
    for i, count in stats.routes:
      routes[i] += count
    peakGradNorm = max(peakGradNorm, stats.gradNorm)
    if options.gradClip > 0 and stats.gradNorm > float64(options.gradClip): inc clipped
    inc updates
    let previousEpoch = progress.tokens div tokensPerEpoch
    progress.tokens += tokensPerStep
    let epoch = progress.tokens div tokensPerEpoch
    if epoch > previousEpoch:
      echo &"[epoch] epoch={epoch} step={step} tokens_total={progress.tokens}" & estimate
    if step == start + 1 or step mod options.logEvery == 0 or step == finalStep:
      let rate = float64(updates) * float64(tokensPerStep) / seconds
      let inEpoch = progress.tokens mod tokensPerEpoch
      echo &"[train] step={step}/{finalStep} epoch={epoch} epoch_tokens={inEpoch}/{tokensPerEpoch} " &
           &"loss={lossSum / float64(updates):.4f} " &
           &"lr={lr:.3e} grad_norm_max={peakGradNorm:.3f} clipped={clipped}/{updates} " &
           &"tok/s={int(round(rate))} tokens_this_run={progress.tokens - startTokens}" & estimate
      if ctx.usesTrainingTensorCores:
        let stats = ctx.takeTrainingGemmStats()
        echo &"[tensor-core] step={step} cooperative_gemm={stats.tensorCore} small_fp32_gemm={stats.fp32}"
      if routes.len > 0:
        echo &"[moe] step={step} expert_tokens={routes} balance_loss={balanceSum / float64(updates):.5f}"
        for count in routes.mitems: count = 0
        balanceSum = 0
      if draftInputs > 0:
        echo &"[draft] step={step} replaced_inputs={draftReplaced}/{draftInputs} " &
             &"({100.0 * float64(draftReplaced) / float64(draftInputs):.2f}%)"
        draftReplaced = 0
        draftInputs = 0
      if eosRows > 0:
        echo &"[eos] step={step} targets={eosTargets} in {updates} updates"
        eosTargets = 0
      lossSum = 0.0
      seconds = 0.0
      peakGradNorm = 0.0
      updates = 0
      clipped = 0
    if step mod options.evalEvery == 0 or step == finalStep:
      evaluate(step)
      discard ctx.takeTrainingGemmStats()
    if step mod options.saveEvery == 0 or step == finalStep:
      saveProgress(step)
