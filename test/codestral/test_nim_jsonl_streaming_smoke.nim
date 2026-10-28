import std/[json, math, os, random, strformat, strutils, tables]
import backend
import finetune_lora
import hf_tokenizer
import jamba_hybrid
import lora
import nimformer
import test_nimformer

proc writeNimExamplesRange(sourcePath, targetPath: string; skip = 0; count = 8) =
  ## Copies `count` Nim-language JSONL rows starting after `skip` matching
  ## rows. Lets us build three DISJOINT splits (pretrain / finetune / heldout)
  ## from the same source without loading it all into RAM.
  var outFile = open(targetPath, fmWrite)
  defer: outFile.close()
  var seen = 0
  var written = 0
  for line in lines(sourcePath):
    let row = parseJson(line)
    if row.kind == JObject and row.hasKey("language") and
       row["language"].getStr().toLowerAscii() == "nim":
      inc seen
      if seen <= skip: continue
      outFile.writeLine(line)
      inc written
      if written >= count: break
  if written == 0:
    raise newException(IOError,
      "No Nim rows in range [" & $skip & ".." & $(skip + count) & ") of " & sourcePath)
  echo &"[smoke] wrote {written} Nim rows (skip={skip}) to {targetPath}"

proc numel(shape: openArray[int]): int =
  result = 1
  for dim in shape: result *= dim

proc cloneFloat32(values: seq[float32]): seq[float32] =
  result = newSeq[float32](values.len)
  for i, value in values: result[i] = value

proc copySharedWeights(source, target: var NimformerModelJamba) =
  var targetByName = initTable[string, ptr Tensor]()
  for (name, tensorPtr) in paramList(target): targetByName[name] = tensorPtr
  for (name, tensorPtr) in paramList(source):
    if name in targetByName:
      targetByName[name][].data = cloneFloat32(tensorPtr[].data)

proc initSmokeModel(pattern: string; vocab: int): NimformerModelJamba =
  newNimformerModelJamba(vocab, 64, 2, 2, pattern, useSession = true,
    mambaRealDState = 4, mambaRealNGroups = 1, mambaRealConvKernel = 3,
    mambaRealExpand = 2)

proc restoreGMasters(m: var NimformerModelJamba; store: AttnWeightDiskStore) =
  for (name, tensorPtr) in paramList(m):
    if not (name.startsWith("block1.attn.") or name.startsWith("block1.fuseGate.")):
      continue
    tensorPtr[].data = store.loadW(name, numel(tensorPtr[].shape))

proc lastLogits(m: var NimformerModelJamba; lo: var JambaLora;
                tokenIds: seq[int]; ctx: Backend): seq[float32] =
  mergeIn(m, lo, ctx)
  try:
    let logits = forwardBatch(m, @[tokenIds], ctx)
    let offset = (tokenIds.len - 1) * m.vocab
    result = logits.data[offset ..< offset + m.vocab]
  finally:
    restoreBase(m, lo)

proc greedyContinuation(m: var NimformerModelJamba; lo: var JambaLora;
                        tok: HfTokenizer; promptIds: seq[int];
                        ctx: Backend; steps = 4): tuple[ids: seq[int], text: string] =
  var context = promptIds
  for _ in 0 ..< steps:
    let logits = lastLogits(m, lo, context, ctx)
    var bestId = 0
    for id in 1 ..< logits.len:
      if logits[id] > logits[bestId]: bestId = id
    result.ids.add(bestId)
    context.add(bestId)
  result.text = decodeHf(tok, result.ids)

proc maskedNll(m: var NimformerModelJamba; lo: var JambaLora;
               ids, targets: seq[seq[int]]; ctx: Backend): float64 =
  mergeIn(m, lo, ctx)
  var logits: Tensor
  try:
    logits = forwardBatch(m, ids, ctx)
  finally:
    restoreBase(m, lo)
  var count = 0
  for row in 0 ..< ids.len:
    for t in 0 ..< targets[row].len:
      let target = targets[row][t]
      if target == TargetIgnore: continue
      let offset = (row * ids[row].len + t) * m.vocab
      var maxLogit = -1.0e300
      for token in 0 ..< m.vocab:
        maxLogit = max(maxLogit, float64(logits.data[offset + token]))
      var expSum = 0.0
      for token in 0 ..< m.vocab:
        expSum += exp(float64(logits.data[offset + token]) - maxLogit)
      result += maxLogit + ln(expSum) - float64(logits.data[offset + target])
      inc count
  if count == 0: raise newException(ValueError, "No targets for smoke NLL")
  result /= float64(count)

when isMainModule:
  setForbidCpuFallback(true)
  let ctx = newBackend("auto")
  let hfDir = getEnv("BYBY_SMOKE_HF_DIR", CFG_HF_DIR_DEFAULT)
  let sourcePath = getEnv("BYBY_SMOKE_NIM_JSONL", "test/spectron/nim_python_code_dataset_mega.jsonl")
  let smokeRoot = getTempDir() / ("nimformers-nim-jsonl-smoke-" & $getCurrentProcessId())
  if dirExists(smokeRoot): removeDir(smokeRoot)
  createDir(smokeRoot)

  # ── Three DISJOINT Nim splits: pretrain (dataA), finetune (dataB), heldout (dataC) ──
  let dataA = smokeRoot / "dataA.jsonl"
  let dataB = smokeRoot / "dataB.jsonl"
  let dataC = smokeRoot / "dataC.jsonl"
  writeNimExamplesRange(sourcePath, dataA, skip = 0, count = 8)
  writeNimExamplesRange(sourcePath, dataB, skip = 8, count = 8)
  writeNimExamplesRange(sourcePath, dataC, skip = 16, count = 4)

  var tokenizer = newHfTokenizer(hfDir)
  echo &"[smoke] tokenizer backend={tokenizer.backend}, vocab={vocabSizeHf(tokenizer)}; kept resident in process"

  proc buildBatches(jsonlPath: string; count: int):
      seq[tuple[ids, targets, targetsMasked: seq[seq[int]]]] =
    var batcher = newStreamingBatcher(
      @[DatasetSpec(path: jsonlPath, schema: dsTopicCode)],
      seqLen = 32, batchSize = 1, tok = default(BPETokenizer),
      hfTok = tokenizer, usingHfTok = true)
    for _ in 0 ..< count:
      let b = batcher.nextBatch()
      if b.ids.len == 0: break
      result.add(b)

  let batchesA = buildBatches(dataA, 4)
  let batchesB = buildBatches(dataB, 4)
  let batchesC = buildBatches(dataC, 2)
  if batchesA.len == 0 or batchesB.len == 0 or batchesC.len == 0:
    raise newException(AssertionDefect, "Not enough Nim rows to build disjoint phases")

  proc assertDisjoint(a, b: seq[tuple[ids, targets, targetsMasked: seq[seq[int]]]];
                      labelA, labelB: string) =
    for ba in a:
      for bb in b:
        if ba.ids == bb.ids:
          raise newException(AssertionDefect,
            "Phase " & labelA & " and " & labelB &
            " share a training batch (not disjoint): " & $ba.ids)
  assertDisjoint(batchesA, batchesB, "pretrain", "finetune")
  assertDisjoint(batchesA, batchesC, "pretrain", "heldout")
  assertDisjoint(batchesB, batchesC, "finetune", "heldout")
  echo &"[smoke] disjoint splits: pretrain={batchesA.len} finetune={batchesB.len} heldout={batchesC.len} batches"

  randomize(44118)

  # ── PHASE 1: pretrain R-only (RR) FROM SCRATCH on dataA ──
  var modelR = initSmokeModel("RR", vocabSizeHf(tokenizer))
  let loraTargets = @[
    "block0.mambaReal.inProj.weight",
    "block0.mambaReal.outProj.weight",
    "block1.mambaReal.inProj.weight",
    "block1.mambaReal.outProj.weight"]
  var adaptersR = attachLora(modelR, loraTargets, r = 2, alpha = 4'f32)
  var stateR = newLoraTrainState(adaptersR, lr = 1e-4'f32)
  var adamR = initAdamDiskStore(smokeRoot / "adam-r")
  var sidecarR = initAttnWeightDiskStore(smokeRoot / "sidecar-r")
  echo "[phase1] train R-only (RR) from scratch on dataA"
  var lossA = 0'f32
  for b in batchesA:
    lossA = trainStepGraftStreaming(modelR, adaptersR, stateR, adamR, sidecarR,
      CFG_NEWBLOCK_LR, b.ids, b.targetsMasked, ctx)
  echo &"[phase1] final loss={lossA:.6f}; R-only LoRA trained"

  # Snapshot MERGED R-only weights (base + LoRA delta) so phase 2 starts from
  # exactly this checkpoint while a fresh LoRA adapter starts at delta=0.
  mergeIn(modelR, adaptersR, ctx)
  var trainedR = initTable[string, seq[float32]]()
  for (name, t) in paramList(modelR):
    trainedR[name] = cloneFloat32(t[].data)
  restoreBase(modelR, adaptersR)

  # ── PHASE 2: finetune R+Linear++ on FRESH dataB, starting from that checkpoint ──
  var modelRG = initSmokeModel("RG", vocabSizeHf(tokenizer))
  block:
    var targetByName = initTable[string, ptr Tensor]()
    for (name, t) in paramList(modelRG): targetByName[name] = t
    for name, data in trainedR:
      if name in targetByName:
        targetByName[name][].data = cloneFloat32(data)
  var adaptersRG = attachLora(modelRG, loraTargets, r = 2, alpha = 4'f32)
  var stateRG = newLoraTrainState(adaptersRG, lr = 1e-4'f32)
  var adamRG = initAdamDiskStore(smokeRoot / "adam-rg")
  var sidecarRG = initAttnWeightDiskStore(smokeRoot / "sidecar-rg")
  echo "[phase2] finetune R+Linear++ on fresh dataB rows"
  var lossB = 0'f32
  for b in batchesB:
    lossB = trainStepGraftStreaming(modelRG, adaptersRG, stateRG, adamRG, sidecarRG,
      CFG_NEWBLOCK_LR, b.ids, b.targetsMasked, ctx)
  restoreGMasters(modelRG, sidecarRG)
  echo &"[phase2] final loss={lossB:.6f}; R+Linear++ finetuned on new data"

  # ── PHASE 3: compare inference of original R-only vs new R+Linear++ on heldout dataC ──
  let evalBatch = batchesC[0]
  let nllR = maskedNll(modelR, adaptersR, evalBatch.ids, evalBatch.targetsMasked, ctx)
  let nllRG = maskedNll(modelRG, adaptersRG, evalBatch.ids, evalBatch.targetsMasked, ctx)

  let promptIds = encodeHf(tokenizer,
    "[INST] Viet code nim cho: Hello World[/INST]", addEos = false)
  let logitsR = lastLogits(modelR, adaptersR, promptIds, ctx)
  let logitsRG = lastLogits(modelRG, adaptersRG, promptIds, ctx)
  let greedyR = greedyContinuation(modelR, adaptersR, tokenizer, promptIds, ctx)
  let greedyRG = greedyContinuation(modelRG, adaptersRG, tokenizer, promptIds, ctx)

  var maxDelta = 0.0
  var deltaSq = 0.0
  var nonFinite = 0
  for i in 0 ..< logitsR.len:
    let d = float64(logitsRG[i] - logitsR[i])
    maxDelta = max(maxDelta, abs(d))
    deltaSq += d * d
    if logitsRG[i].classify in {fcNan, fcInf, fcNegInf}: inc nonFinite
  if nonFinite > 0:
    raise newException(AssertionDefect, "R+Linear++ heldout logits contain NaN/Inf")
  let l2Delta = sqrt(deltaSq)

  echo &"[phase3] heldout NLL: R-only={nllR:.5f}; R+Linear++={nllRG:.5f}; delta={nllRG - nllR:+.5f}"
  echo &"[phase3] prompt logits max|delta|={maxDelta:.6g}; ||delta||2={l2Delta:.6g}"
  echo &"[phase3] greedy R-only: '{greedyR.text.escape}'"
  echo &"[phase3] greedy R+G:    '{greedyRG.text.escape}'"
  echo "PASS - 3-phase: R-only pretrain -> R+Linear++ finetune on fresh data -> heldout inference comparison"

  # ── PHASE 4: per-layer G rescue ────────────────────────────────────────
  # Force the G post-update safety check to fail so the rollback path can be
  # exercised deterministically. With BYBY_G_TEST_FORCE_RESCUE=1 the layer
  # must roll back and halve its LR for CFG_G_MAX_RESCUE_ATTEMPTS steps, then
  # finally raise instead of looping forever.
  block:
    let gIdx = 1                      # "RG" pattern: G block lives at index 1
    gPerBlockLrScale[gIdx] = 1.0'f32
    gPerBlockRescueFails[gIdx] = 0
    putEnv("BYBY_G_TEST_FORCE_RESCUE", "1")
    try:
      var lastScale = 1.0'f32
      for attempt in 1 .. CFG_G_MAX_RESCUE_ATTEMPTS:
        discard trainStepGraftStreaming(modelRG, adaptersRG, stateRG, adamRG,
          sidecarRG, CFG_NEWBLOCK_LR, evalBatch.ids, evalBatch.targetsMasked, ctx)
        if gPerBlockRescueFails[gIdx] != attempt:
          raise newException(AssertionDefect,
            &"rescue counter expected {attempt}, got {gPerBlockRescueFails[gIdx]}")
        let scale = gPerBlockLrScale.getOrDefault(gIdx, 1.0'f32)
        if scale >= lastScale:
          raise newException(AssertionDefect,
            &"rescue did not reduce per-block LR scale: {lastScale:.4f} -> {scale:.4f}")
        lastScale = scale
      var raised = false
      try:
        discard trainStepGraftStreaming(modelRG, adaptersRG, stateRG, adamRG,
          sidecarRG, CFG_NEWBLOCK_LR, evalBatch.ids, evalBatch.targetsMasked, ctx)
      except CatchableError as e:
        raised = ("consecutive post-update safety failures" in e.msg) or
                 ("Rescue failed" in e.msg)
      if not raised:
        raise newException(AssertionDefect,
          "G rescue did not give up after CFG_G_MAX_RESCUE_ATTEMPTS consecutive failures")
      echo &"PASS - G rescue: {CFG_G_MAX_RESCUE_ATTEMPTS} rollbacks with LR 1.0 -> " &
        &"{lastScale:.3f}, then raised on attempt {CFG_G_MAX_RESCUE_ATTEMPTS + 1}"
    finally:
      delEnv("BYBY_G_TEST_FORCE_RESCUE")

  removeDir(smokeRoot)
