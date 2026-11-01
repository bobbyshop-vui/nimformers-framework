## load_codestral_into_jamba.nim
## Loads the REAL pretrained weights of Mamba-Codestral-7B-v0.1 DIRECTLY into the
## Jamba-hybrid model (jamba_hybrid.nim) with 'R' blocks (Mamba2Real - matching the
## formula exactly). The 'A' blocks (LinearPlusAttention) KEEP their random init
## - they are the NEW grafted part, not in the original HF checkpoint,
## to be trained (directly or through LoRA - with lora.nim, unchanged because
## lora.nim is generic over parameter names).
##
## LAYER MAPPING CONVENTION: the 'R' blocks of your pattern get HF weights
## IN ORDER OF APPEARANCE (first 'R' block -> HF layer 0, second 'R'
## block -> HF layer 1, etc.) - the HF layers matching the positions of
## 'A' blocks are SKIPPED (an 'A' is a NEW layer replacing some Mamba layer that was "grafted over").
## E.g. the pattern "RRRARRRA" (8 blocks, 6 'R' + 2 'A') uses HF layers 0..5
## for the 6 'R' blocks and skips HF layers 6,7 (or whichever HF layers) - there is NO
## 1-1 relation between block positions and HF layer positions, only the ORDER is kept.
##
## WARNING: the tensor names "backbone.embeddings.weight"/"lm_head.weight" are not verified
## against the real manifest of Mamba-Codestral-7B-v0.1 (transformers may rename them
## between versions) - the tensor names that could NOT be mapped are PRINTED at the end; adjust
## hfNameMapMamba2/EMBED_NAME/LMHEAD_NAME in this file to match before
## trusting the result.

import std/[tables, strformat, strutils, os]   # os: getEnv for BYBY_ALLOW_RANDOM_EMBED
import nimformer
import jamba_hybrid
import mamba2_real
when not defined(codestralInference):
  import sharded_hf_loader
import rawbin   # RawBinTensor
import std/os
import quant    # loadQuantStateDict, dequantizeTensor - for the .nimq source (fp4 quantized)

const
  EmbedName = "backbone.embeddings.weight"
  LmHeadName = "lm_head.weight"
  NormFName = "backbone.norm_f.weight"   # NEW: see the debug report - the final norm before lm_head

proc findTensor(tensors: seq[RawBinTensor], name: string): int =
  for i, t in tensors:
    if t.name == name: return i
  return -1

proc loadCodestralIntoJambaFromTensors*(m: var NimformerModelJamba, tensors: seq[RawBinTensor]) =
  ## CORE function - takes a list of tensors already decoded to fp32 in RAM,
  ## WHATEVER their source (the original safetensors READ DIRECTLY, or
  ## a quantized fp4 .nimq DEQUANTIZED back) - the 2 wrappers below
  ## (loadCodestralIntoJamba/loadCodestralIntoJambaFromNimq) differ only
  ## IN HOW `tensors` IS OBTAINED, the name mapping/assignment to the model is IDENTICAL.
  echo &"[load_codestral] {tensors.len} tensors. Starting the mapping into the model ..."

  var usedNames = initTable[string, bool]()

  # ── embed + lm_head ──────────────────────────────────────────
  let embedIdx = findTensor(tensors, EmbedName)
  if embedIdx >= 0:
    doAssert tensors[embedIdx].data.len == m.embed.weight.data.len,
      &"embed shape lech: HF={tensors[embedIdx].data.len} model={m.embed.weight.data.len}"
    m.embed.weight.data = tensors[embedIdx].data
    usedNames[EmbedName] = true
    echo &"[load_codestral] OK embed <- {EmbedName}"
  else:
    # CHANGED: it used to just `echo` and continue with a RANDOM EMBEDDING. That is
    # the surest way to generate 100% garbage tokens while the log looks normal
    # and the loss still "converges" (to a meaningless model). The top of this file
    # also says "tensor names not verified" -> a name mismatch is a REAL possibility.
    raise newException(ValueError,
      &"[load_codestral] tensor '{EmbedName}' NOT found in the HF checkpoint. " &
      "A random-init embedding would produce garbage tokens; fix EmbedName from the manifest.")

  let lmHeadIdx = findTensor(tensors, LmHeadName)
  if lmHeadIdx >= 0:
    doAssert tensors[lmHeadIdx].data.len == m.outProj.weight.data.len,
      &"lm_head shape lech: HF={tensors[lmHeadIdx].data.len} model={m.outProj.weight.data.len}"
    m.outProj.weight.data = tensors[lmHeadIdx].data
    usedNames[LmHeadName] = true
    echo "[load_codestral] OK outProj <- lm_head.weight"
  else:
    echo "[load_codestral] WARNING: 'lm_head.weight' not found - the model may use " &
         "tied embeddings (lm_head = embed.weight, no separate tensor) - " &
         "trying to reuse embed.weight for outProj:"
    if embedIdx >= 0 and tensors[embedIdx].data.len == m.outProj.weight.data.len:
      m.outProj.weight.data = tensors[embedIdx].data
      echo "[load_codestral]   -> tied embedding applied OK (shapes match)."
    else:
      # CHANGED: like the embedding case above - a random-init lm_head means
      # COMPLETELY random logits over 32768 tokens. It must not be silent.
      raise newException(ValueError,
        "[load_codestral] 'lm_head.weight' NOT found and tied embeddings CANNOT " &
        "be used either (shape mismatch). Continuing would leave lm_head " &
        "RANDOM-INIT -> random logits over the whole vocab -> 100% garbage tokens. " &
        "Run print_manifest.nim to compare with the real tensor names.")

  # ── normF (== backbone.norm_f, applied once before lm_head) ────────────
  # FIX for garbage tokens (bug #8): this function is SHARED by loadCodestralIntoJamba
  # (original safetensors) AND loadCodestralIntoJambaFromNimq (dequant from .nimq,
  # the finetune_lora.nim path). The earlier normF fix (bug #4) applied ONLY to
  # loadCodestralQuantResidentFromNimq (the infer_lora_bin branch) -> these 2 paths
  # still lacked normF and still ran with gamma=1.
  let normFIdx = findTensor(tensors, NormFName)
  if normFIdx >= 0:
    doAssert tensors[normFIdx].data.len == m.normF.gamma.data.len,
      &"normF shape lech: HF={tensors[normFIdx].data.len} model={m.normF.gamma.data.len}"
    m.normF.gamma.data = tensors[normFIdx].data
    usedNames[NormFName] = true
    echo &"[load_codestral] OK normF <- {NormFName}"
  else:
    raise newException(ValueError,
      "[load_codestral] '" & NormFName & "' NOT found - a random/identity normF would produce garbage tokens.")

  # ── each 'R' block in order of appearance -> HF layers 0,1,2,... ──
  var hfLayerCounter = 0
  var matchedLayers = 0
  var totalLayers = 0
  for bi in 0 ..< m.blocks.len:
    if m.blocks[bi].kind notin {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}: continue
    totalLayers += 1
    let nameMap = hfNameMapMamba2(hfLayerCounter)
    var allOk = true
    var missing: seq[string] = @[]

    proc tryLoad(fieldData: var seq[float32], hfKey: string): bool =
      let idx = findTensor(tensors, nameMap[hfKey])
      if idx >= 0:
        if tensors[idx].data.len == fieldData.len:
          fieldData = tensors[idx].data
          usedNames[nameMap[hfKey]] = true
          return true
        else:
          echo &"[load_codestral] layer {hfLayerCounter} '{hfKey}': SHAPE MISMATCH " &
               &"(HF={tensors[idx].data.len} model={fieldData.len}) - SKIPPED, keeping the random init"
          return false
      else:
        missing.add(nameMap[hfKey])
        return false

    if not tryLoad(m.blocks[bi].mambaReal.inProj.weight.data, "inProj.weight"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.convWeight.data, "convWeight"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.convBias.data, "convBias"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.aLog.data, "aLog"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.dSkip.data, "dSkip"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.dtBias.data, "dtBias"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.normWeight.data, "normWeight"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.outProj.weight.data, "outProj.weight"): allOk = false
    # FIX for garbage tokens (bug #7): the pre-norm OF THE WHOLE BLOCK (ln1 == Mamba2Block.norm in
    # HF, OUTSIDE the mixer submodule - DIFFERENT from 'normWeight' above, which is the norm
    # INSIDE the mixer). It was NOT loaded here before -> ln1.gamma = 1.0 for every
    # block. See hfNameMapMamba2.
    if not tryLoad(m.blocks[bi].ln1.gamma.data, "ln1Weight"): allOk = false
    # inProj.bias / outProj.bias: HF Mamba2 USUALLY has no bias (bias=False)
    # -> NOT an error when missing, only logged as info, the bias stays 0 by default.
    let inBiasIdx = findTensor(tensors, nameMap["inProj.bias"])
    if inBiasIdx >= 0 and tensors[inBiasIdx].data.len == m.blocks[bi].mambaReal.inProj.bias.data.len:
      m.blocks[bi].mambaReal.inProj.bias.data = tensors[inBiasIdx].data
      usedNames[nameMap["inProj.bias"]] = true

    if allOk:
      matchedLayers += 1
      echo &"[load_codestral] block{bi} <- HF mixer layer {hfLayerCounter}: OK (9/9 main tensors match)"
    else:
      echo &"[load_codestral] block{bi} <- HF mixer layer {hfLayerCounter}: THIEU {missing}"
    hfLayerCounter += 1

  echo &"\n[load_codestral] SUMMARY: {matchedLayers}/{totalLayers} 'R' blocks FULLY mapped. " &
       "Blocks not marked 'OK' above KEEP the random init (trained from scratch) - check " &
       "hfNameMapMamba2 if this number is lower than expected."
  if matchedLayers != totalLayers:
    raise newException(ValueError,
      &"[load_codestral] STOP: only {matchedLayers}/{totalLayers} R blocks fully mapped; " &
       "training with random-init weights is not allowed, to avoid garbage tokens.")

  # list the HF tensors NOT used (maybe biases that do not exist in this
  # model - normal - or WRONG name mappings - need checking)
  var unusedSample: seq[string] = @[]
  for t in tensors:
    if t.name notin usedNames:
      unusedSample.add(t.name)
      if unusedSample.len >= 15: break
  if unusedSample.len > 0:
    echo &"[load_codestral] Sample of {unusedSample.len} HF tensors NOT used: {unusedSample}"

proc loadCodestralQuantResidentFromNimq*(m: var NimformerModelJamba, nimqPath: string):
                                          Table[string, QuantTensor] =
  ## "Quant-resident" variant - does NOT dequantize inProj.weight/outProj.weight
  ## (the 2 largest matrices, ~29GB if all 52 blocks were dequantized) - keeps them
  ## COMPRESSED (QuantTensor, ~1 byte/element with fp4), returned in the result
  ## table so lora.nim (attachLoraQuantResident/mergeInOne) dequantizes them ON THE
  ## FLY through the LUT WHENEVER a block is needed, and frees them right after (see
  ## trainStepGraftStreaming in finetune_lora.nim).
  ## The OTHER parameters (embed, lm_head, aLog/dSkip/dtBias/normWeight/
  ## convWeight - all small, ~1.1GB in total) ARE dequantized NORMALLY and assigned
  ## directly to the model like loadCodestralIntoJambaFromNimq (no quant-resident
  ## for this part, not needed).
  echo &"[load_codestral] (quant-resident) Doc {nimqPath} ..."
  var sd = quant.loadQuantStateDict(nimqPath, mapped = getEnv("BYBY_MODEL_MMAP", "1") != "0").sd
  var byName = initTable[string, QuantTensor]()
  for (name, qt) in sd.mitems: byName[name] = move(qt)
  # SUSPECT #2 for the unusually high RSS after load (see the debug report): before,
  # 'sd' (the original seq from loadQuantStateDict) stayed alive UNTIL THIS FUNCTION
  # RETURNED although everything had been copied into byName - so during loading there
  # COULD be 2-3 copies at once of all 129 LARGE tensors (sd + byName +
  # result, each possibly several GB). 'sd' is released RIGHT after everything
  # moved into byName (the move above + the delete here = surely nothing extra kept).
  sd = @[]

  result = initTable[string, QuantTensor]()

  proc shapeNumel(shape: seq[int]): int =
    result = 1
    for d in shape: result *= d

  proc sourceShapeMatches(name: string; source, expected: seq[int]): bool =
    ## HF stores Conv1d weights as [outChannels, inChannels, kernelSize].
    ## Mamba2 here has depthwise Conv1d (inChannels=1), represented internally
    ## as [outChannels, kernelSize]. The singleton axis changes rank, not data.
    if source == expected: return true
    result = name.endsWith(".conv1d.weight") and source.len == 3 and
      expected.len == 2 and source[0] == expected[0] and source[1] == 1 and
      source[2] == expected[1]

  proc requireTensor(name: string, expectedShape: seq[int], missing: var seq[string]) =
    if name notin byName:
      missing.add(name & " (missing)")
    elif not sourceShapeMatches(name, byName[name].shape, expectedShape) or
         byName[name].bytes.len == 0:
      missing.add(name & " (shape=" & $byName[name].shape & ", expected=" &
                  $expectedShape & ", data=" & $byName[name].bytes.len & ")")

  proc dequantSmall(name: string): RawBinTensor =
    let qt = byName[name]
    let values = quant.dequantizeTensor(qt)
    if values.len != shapeNumel(qt.shape):
      raise newException(ValueError,
        &"[load_codestral] .nimq '{name}' decoded into {values.len} elements, " &
        &"but the metadata shape {qt.shape} needs {shapeNumel(qt.shape)}; the file may be corrupt.")
    RawBinTensor(name: name, shape: qt.shape, data: values)

  # embed - normal dequant (needed so lookupBatch can access rows
  # randomly by token id, not a single matmul like lm_head,
  # so it stays resident for simplicity, ~0.54GB).
  var smallTensors: seq[RawBinTensor] = @[]
  var missingRequired: seq[string] = @[]
  requireTensor(EmbedName, m.embed.weight.shape, missingRequired)
  requireTensor(NormFName, m.normF.gamma.shape, missingRequired)
  if EmbedName in byName: smallTensors.add(dequantSmall(EmbedName))
  if NormFName in byName: smallTensors.add(dequantSmall(NormFName))   # NEW
  # lm_head - CHANGED: NO eager dequant any more (saves ~0.54GB of resident RAM
  # during the whole training) - put into `result` (the quant-resident table) under the
  # REAL PARAMETER NAME ("outProj.weight") exactly like the 52 'R' blocks. lm_head is
  # used EXACTLY ONCE per training step (at the end of the forward for the logits, at the start of the backward for
  # dX) - see mergeInOnePlain/restoreBaseOnePlain in lora.nim + the call sites
  # in trainStepGraftStreaming (finetune_lora.nim) around those 2 uses.
  # NO longer put into smallTensors, because it would be eagerly dequantized below.
  if LmHeadName in byName:
    let qt = byName[LmHeadName]
    var expectedLen = 1
    for d in qt.shape: expectedLen *= d
    if qt.shape == m.outProj.weight.shape and qt.bytes.len > 0:
      result["outProj.weight"] = qt
      byName.del(LmHeadName)   # releases the copy in byName, result already holds it
    else:
      missingRequired.add(LmHeadName & " (shape=" & $qt.shape & ")")
  else:
    # Allows valid tied-embedding checkpoints: uses the embedding itself as lm_head.
    # If the shapes do not match it still fails fast, never a random lm_head.
    if EmbedName in byName and byName[EmbedName].shape == m.outProj.weight.shape and
       byName[EmbedName].bytes.len > 0:
      discard
    else:
      missingRequired.add(LmHeadName & " (missing and no valid tied embedding)")

  var hfLayerCounter = 0
  var rBlockIdx: seq[int] = @[]
  for bi in 0 ..< m.blocks.len:
    if m.blocks[bi].kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}: rBlockIdx.add(bi)

  for bi in rBlockIdx:
    let nameMap = hfNameMapMamba2(hfLayerCounter)
    requireTensor(nameMap["inProj.weight"], m.blocks[bi].mambaReal.inProj.weight.shape, missingRequired)
    requireTensor(nameMap["outProj.weight"], m.blocks[bi].mambaReal.outProj.weight.shape, missingRequired)
    requireTensor(nameMap["convWeight"], m.blocks[bi].mambaReal.convWeight.shape, missingRequired)
    requireTensor(nameMap["convBias"], m.blocks[bi].mambaReal.convBias.shape, missingRequired)
    requireTensor(nameMap["aLog"], m.blocks[bi].mambaReal.aLog.shape, missingRequired)
    requireTensor(nameMap["dSkip"], m.blocks[bi].mambaReal.dSkip.shape, missingRequired)
    requireTensor(nameMap["dtBias"], m.blocks[bi].mambaReal.dtBias.shape, missingRequired)
    requireTensor(nameMap["normWeight"], m.blocks[bi].mambaReal.normWeight.shape, missingRequired)
    requireTensor(nameMap["ln1Weight"], m.blocks[bi].ln1.gamma.shape, missingRequired)
    # 2 LARGE matrices - KEPT COMPRESSED, put into result under PARAMLIST NAMES
    # (exactly "block{bi}.mambaReal.inProj.weight" as paramListMamba2Real
    # sets them, NOT the original HF names - so lora.nim looks them up directly by the
    # adapter name with no remapping).
    if nameMap["inProj.weight"] in byName:
      result[&"block{bi}.mambaReal.inProj.weight"] = byName[nameMap["inProj.weight"]]
      byName.del(nameMap["inProj.weight"])   # releases the copy in byName
    else:
      echo &"[load_codestral] WARNING: layer {hfLayerCounter} lacks in_proj.weight in the .nimq"
    if nameMap["outProj.weight"] in byName:
      result[&"block{bi}.mambaReal.outProj.weight"] = byName[nameMap["outProj.weight"]]
      byName.del(nameMap["outProj.weight"])   # releases the copy in byName
    else:
      echo &"[load_codestral] WARNING: layer {hfLayerCounter} lacks out_proj.weight in the .nimq"

    # SMALL parameters - normal dequant as before (convWeight/convBias/aLog/
    # dSkip/dtBias/normWeight - a few tens of KB per block in total, negligible).
    # ln1Weight NEWLY ADDED - FIX for garbage tokens (bug #7): the pre-norm of the whole block
    # (backbone.layers.N.norm.weight, OUTSIDE the mixer) used NOT to be in
    # this list -> ln1.gamma was never really loaded and stayed gamma=1.
    for key in ["convWeight", "convBias", "aLog", "dSkip", "dtBias", "normWeight", "ln1Weight"]:
      if nameMap[key] in byName:
        smallTensors.add(dequantSmall(nameMap[key]))
    hfLayerCounter += 1

  # assigns the SMALL parameters directly to the model (does NOT call loadCodestralIntoJambaFromTensors -
  # that function would report "MISSING inProj.weight/outProj.weight" for EVERY 'R' block because
  # those 2 large tensors are deliberately not in smallTensors - misleading, although
  # functionally harmless).
  proc findRaw(tensors: seq[RawBinTensor], name: string): int =
    for i, t in tensors:
      if t.name == name: return i
    return -1

  let embedIdx = findRaw(smallTensors, EmbedName)
  if embedIdx >= 0 and smallTensors[embedIdx].data.len == shapeNumel(m.embed.weight.shape):
    m.embed.weight.data = smallTensors[embedIdx].data
    echo "[load_codestral] (quant-resident) OK embed"
  let normFIdx = findRaw(smallTensors, NormFName)   # NEW: see the debug report
  if normFIdx >= 0 and smallTensors[normFIdx].data.len == shapeNumel(m.normF.gamma.shape):
    m.normF.gamma.data = smallTensors[normFIdx].data
    echo "[load_codestral] (quant-resident) OK normF (final norm before lm_head)"
  else:
    raise newException(ValueError, "[load_codestral] normF could not be loaded - refusing to run to avoid garbage tokens")
  if "outProj.weight" in result:
    echo "[load_codestral] (quant-resident) OK outProj (lm_head) - kept compressed, dequantized temporarily when used"
  elif embedIdx >= 0 and smallTensors[embedIdx].data.len == shapeNumel(m.outProj.weight.shape):
    # a real tied embedding (no separate lm_head.weight in the .nimq) -
    # in this case it IS dequantized directly (cannot be quant-resident because it
    # shares storage with embed, already dequantized above).
    m.outProj.weight.data = smallTensors[embedIdx].data
    echo "[load_codestral] (quant-resident) OK outProj (tied-embedding)"

  hfLayerCounter = 0
  var ln1MissingCount = 0
  # Counts the fields REALLY assigned — the log "N small parameters dequantized+assigned"
  # used to count only smallTensors.len (the tensors in the file), NOT the model fields
  # that got values. A shape/name mismatch -> the field keeps random/init, silent garbage tokens.
  var smallAssigned = 0
  var smallMissed = 0
  var smallMissNames: seq[string] = @[]
  for bi in rBlockIdx:
    let nameMap = hfNameMapMamba2(hfLayerCounter)
    template assignSmall(field: var seq[float32], key: string) =
      let idx = findRaw(smallTensors, nameMap[key])
      if idx >= 0 and smallTensors[idx].data.len == field.len:
        field = smallTensors[idx].data
        inc smallAssigned
      else:
        inc smallMissed
        if smallMissNames.len < 24:
          let whyStr =
            if idx < 0: "missing"
            else: "shape " & $smallTensors[idx].data.len & " != model " & $field.len
          # Avoids &"..." inside the template (hygiene: key/why undeclared).
          smallMissNames.add("block" & $bi & "." & key & " (" & whyStr & ")")
    assignSmall(m.blocks[bi].mambaReal.convWeight.data, "convWeight")
    assignSmall(m.blocks[bi].mambaReal.convBias.data, "convBias")
    assignSmall(m.blocks[bi].mambaReal.aLog.data, "aLog")
    assignSmall(m.blocks[bi].mambaReal.dSkip.data, "dSkip")
    assignSmall(m.blocks[bi].mambaReal.dtBias.data, "dtBias")
    assignSmall(m.blocks[bi].mambaReal.normWeight.data, "normWeight")
    # FIX for garbage tokens (bug #7): ln1.gamma = the pre-norm OF THE WHOLE BLOCK, applied BEFORE
    # the mixer - different from 'normWeight' above (the norm INSIDE the mixer). It used to be
    # in no assignSmall at all -> gamma=1 for every block. Counted separately
    # (not through assignSmall because that template does not return a bool) to warn
    # in one line instead of spamming 64 lines.
    let ln1Idx = findRaw(smallTensors, nameMap["ln1Weight"])
    if ln1Idx >= 0 and smallTensors[ln1Idx].data.len == m.blocks[bi].ln1.gamma.data.len:
      m.blocks[bi].ln1.gamma.data = smallTensors[ln1Idx].data
      inc smallAssigned
    else:
      ln1MissingCount += 1
      inc smallMissed
    hfLayerCounter += 1

  if ln1MissingCount > 0:
    echo &"[load_codestral] WARNING: {ln1MissingCount}/{rBlockIdx.len} blocks LACK ln1.gamma " &
         "(backbone.layers.N.norm.weight - the pre-norm before the mixer, NOT " &
         "mixer.norm.weight) in the .nimq - these blocks run ln1 with " &
         "gamma=1 (identity, WRONG compared to the checkpoint -> garbage tokens)."
  else:
    echo &"[load_codestral] (quant-resident) OK ln1.gamma for all {rBlockIdx.len} blocks (pre-norm before the mixer)"

  if smallMissed > 0:
    echo &"[load_codestral] WARNING: assignSmall failed for {smallMissed} fields " &
         &"(assigned successfully: {smallAssigned}). Missed fields keep init/random -> garbage tokens."
    for nm in smallMissNames:
      stderr.writeLine "[load_codestral]   miss: " & nm
    # SSM small weights (aLog/dtBias/normWeight/...) with a wrong shape = badly wrong logits.
    # Fail loudly so training/inference never runs with a backbone missing parameters.
    raise newException(ValueError,
      &"[load_codestral] {smallMissed} small-tensor fields could not be assigned " &
      &"(e.g.: {smallMissNames[0 ..< min(smallMissNames.len, 6)]}). " &
      "Fix the .nimq / shape map before training/inference.")

  if missingRequired.len > 0:
    raise newException(ValueError,
      "[load_codestral] .nimq lacks/has wrong REQUIRED weights - not training/inferring, to avoid garbage tokens: " &
      $missingRequired[0 ..< min(missingRequired.len, 12)])

  echo &"[load_codestral] (quant-resident) {result.len} LARGE parameters kept compressed " &
       &"({rBlockIdx.len} 'R' blocks), {smallAssigned} small fields assigned SUCCESSFULLY " &
       &"(smallTensors in the file={smallTensors.len})."

when not defined(codestralInference):
  proc loadCodestralIntoJamba*(m: var NimformerModelJamba, hfDir: string) =
    ## Reads DIRECTLY from the original safetensors (~14GB bf16 on disk, ~28GB once
    ## decoded to fp32 in RAM - see the RAM warning in sharded_hf_loader.nim).
    echo &"[load_codestral] Reading {hfDir} (RAM only) ..."
    let tensors = loadAllHFTensorsSharded(hfDir)
    loadCodestralIntoJambaFromTensors(m, tensors)

proc loadCodestralIntoJambaFromNimq*(m: var NimformerModelJamba, nimqPath: string) =
  ## Reads from a QUANTIZED fp4 FILE (~3.5GB on disk - see quantize_codestral.nim
  ## to create it ONCE from the original safetensors, after which the original safetensors can be
  ## DELETED to save disk) - DEQUANTIZED back to fp32 IN RAM at load (training
  ## still needs fp32 because Mamba2Real.nim only has math on plain float32,
  ## NO kernel training directly on quantized data - the fp4 HERE IS
  ## TO SAVE DISK, NOT to save RAM during training).
  echo &"[load_codestral] Reading {nimqPath} (.nimq, quantized), then dequantizing to fp32 ..."
  let sd = quant.loadQuantStateDict(nimqPath).sd
  var tensors: seq[RawBinTensor] = @[]
  for (name, qt) in sd:
    tensors.add(RawBinTensor(name: name, shape: qt.shape, data: quant.dequantizeTensor(qt)))
  echo &"[load_codestral] Dequantized {tensors.len} tensors from the .nimq"
  loadCodestralIntoJambaFromTensors(m, tensors)
