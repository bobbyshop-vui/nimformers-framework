import std/[math, os, random, strformat, strutils, tables]
import backend
import checkpoint
import finetune_lora
import jamba_hybrid
import lora
import nimformer
import quant
import test_nimformer

proc cloneFloat32(values: seq[float32]): seq[float32] =
  result = newSeq[float32](values.len)
  for i, v in values: result[i] = v

proc cloneLora(lo: JambaLora): JambaLora =
  result = JambaLora(adapters: initTable[string, LoraAdapter](),
                     quantBase: initTable[string, QuantTensor]())
  for k, v in lo.adapters:
    result.adapters[k] = LoraAdapter(
      A: newTensor(v.A.shape),
      B: newTensor(v.B.shape),
      r: v.r,
      inF: v.inF,
      outF: v.outF,
      alpha: v.alpha,
      scale: v.scale
    )
    result.adapters[k].A.data = cloneFloat32(v.A.data)
    result.adapters[k].B.data = cloneFloat32(v.B.data)

proc cloneLoraState(st: LoraTrainState): LoraTrainState =
  result = LoraTrainState(
    lr: st.lr,
    adamA: initTable[string, ApfAdamState](),
    adamB: initTable[string, ApfAdamState]()
  )
  for k, v in st.adamA:
    result.adamA[k] = ApfAdamState(step: v.step, m: cloneFloat32(v.m), v: cloneFloat32(v.v))
  for k, v in st.adamB:
    result.adamB[k] = ApfAdamState(step: v.step, m: cloneFloat32(v.m), v: cloneFloat32(v.v))

proc initTestModel(pattern: string; vocab = 256; dim = 64): NimformerModelJamba =
  newNimformerModelJamba(vocab, dim, 2, 2, pattern, useSession = true,
    mambaRealDState = 4, mambaRealNGroups = 1, mambaRealConvKernel = 3,
    mambaRealExpand = 2)

proc copySharedWeights(source, target: var NimformerModelJamba) =
  var targetByName = initTable[string, ptr Tensor]()
  for (name, tensorPtr) in paramList(target): targetByName[name] = tensorPtr
  for (name, tensorPtr) in paramList(source):
    if name in targetByName:
      targetByName[name][].data = cloneFloat32(tensorPtr[].data)

proc assertTensorsClose(a, b: seq[float32]; label: string; tol = 1e-4'f32) =
  doAssert a.len == b.len, &"{label}: length mismatch {a.len} vs {b.len}"
  var maxDiff = 0.0'f32
  for i in 0 ..< a.len:
    let diff = abs(a[i] - b[i])
    if diff > maxDiff: maxDiff = diff
    doAssert diff.classify notin {fcNan, fcInf, fcNegInf}, &"{label}[{i}] is non-finite"
    doAssert diff <= tol, &"{label}[{i}] diff {diff:.6e} > {tol:.6e} (a={a[i]:.6e}, b={b[i]:.6e})"

proc runTests() =
  echo "=== Test Layer Resume without Forward Reconstruction ==="
  setForbidCpuFallback(true)
  let ctx = newBackend("auto")
  randomize(42)

  let testRoot = getTempDir() / ("nimformers-layer-resume-test-" & $getCurrentProcessId())
  if dirExists(testRoot): removeDir(testRoot)
  createDir(testRoot)
  defer:
    try: removeDir(testRoot)
    except OSError: discard

  # Synthetic batch data
  let seqLen = 16
  let batchSize = 1
  var idsBatch0 = @[newSeq[int](seqLen)]
  var tgtBatch0 = @[newSeq[int](seqLen)]
  for j in 0 ..< seqLen:
    idsBatch0[0][j] = (j * 7 + 13) mod 256
    tgtBatch0[0][j] = ((j + 1) * 7 + 13) mod 256

  var idsBatch1 = @[newSeq[int](seqLen)]
  var tgtBatch1 = @[newSeq[int](seqLen)]
  for j in 0 ..< seqLen:
    idsBatch1[0][j] = (j * 11 + 29) mod 256
    tgtBatch1[0][j] = ((j + 1) * 11 + 29) mod 256

  # ─────────────────────────────────────────────────────────────
  # TEST 1: 2-Layer Model ("RR") - Crash after backward block 1
  # Backward order: block 1 -> block 0. Crash after block 1.
  # ─────────────────────────────────────────────────────────────
  echo "\n--- TEST 1: 2-Layer Model (RR) ---"
  let pattern2 = "RR"
  var mBase2 = initTestModel(pattern2)
  var mTest2 = initTestModel(pattern2)
  copySharedWeights(mBase2, mTest2)

  let targets2 = @[
    "block0.mambaReal.inProj.weight",
    "block0.mambaReal.outProj.weight",
    "block1.mambaReal.inProj.weight",
    "block1.mambaReal.outProj.weight"
  ]
  var loraBase2 = attachLora(mBase2, targets2, r = 2, alpha = 4'f32)
  var loraTest2 = cloneLora(loraBase2)
  var stateBase2 = newLoraTrainState(loraBase2, lr = 1e-3'f32)
  var stateTest2 = cloneLoraState(stateBase2)

  # Isolated paths for baseline and test
  let dirBase2 = testRoot / "base2"
  let dirTest2 = testRoot / "test2"
  createDir(dirBase2)
  createDir(dirTest2)

  let ckptBase2 = dirBase2 / "model.bin"
  let ckptTest2 = dirTest2 / "model.bin"
  let scratchBase2 = dirBase2 / "scratch"
  let scratchTest2 = dirTest2 / "scratch"

  var adamBase2 = initAdamDiskStore(dirBase2 / "adam")
  var sidecarBase2 = initAttnWeightDiskStore(dirBase2 / "sidecar")
  var adamTest2 = initAdamDiskStore(dirTest2 / "adam")
  var sidecarTest2 = initAttnWeightDiskStore(dirTest2 / "sidecar")

  # 1. Run baseline Step 0 uninterrupted
  putEnv("BYBY_CHECKPOINT_PATH", ckptBase2)
  putEnv("BYBY_SCRATCH_DIR", scratchBase2)
  gLastStepForwardRan = false
  let lossBase2 = trainStepGraftStreaming(mBase2, loraBase2, stateBase2, adamBase2, sidecarBase2,
    CFG_NEWBLOCK_LR, idsBatch0, tgtBatch0, ctx, stepIndex = 0)
  checkpoint.clearLayerProgress(ckptBase2)
  removeActivationResume(0)
  doAssert gLastStepForwardRan, "Baseline step 0 MUST have executed forward pass"
  echo &"  [Baseline] Step 0 loss={lossBase2:.6f}"

  # 2. Run test Step 0 with simulated crash after backward layer 1
  putEnv("BYBY_CHECKPOINT_PATH", ckptTest2)
  putEnv("BYBY_SCRATCH_DIR", scratchTest2)
  putEnv("BYBY_TEST_CRASH_AFTER_BACKWARD_LAYER", "1")
  gLastStepForwardRan = false

  var crashed = false
  try:
    discard trainStepGraftStreaming(mTest2, loraTest2, stateTest2, adamTest2, sidecarTest2,
      CFG_NEWBLOCK_LR, idsBatch0, tgtBatch0, ctx, stepIndex = 0)
  except CatchableError as e:
    crashed = true
    echo "  [Test] Caught expected simulated crash: ", e.msg
    doAssert e.msg == "SIMULATED_CRASH_AFTER_BACKWARD_LAYER_1"

  doAssert crashed, "Test step 0 should have crashed after backward layer 1"
  doAssert gLastStepForwardRan, "Initial attempt MUST have run forward pass before crash"

  # Verify on-disk checkpoint artifacts from interrupted step
  let actFile = scratchTest2 / ".xs_resume_step_0.bin"
  let stateFile = ckptTest2 & ".inprogress" / "step-0.state"
  let cursorFile = ckptTest2 & ".inprogress" / "cursor"
  let layer1Bin = ckptTest2 & ".inprogress" / "step-0" / "layer-1.bin"

  doAssert fileExists(actFile), "Activation scratch file MUST exist after crash"
  doAssert fileExists(stateFile), "Resume state file MUST exist after crash"
  doAssert fileExists(cursorFile), "Progress cursor MUST exist after crash"
  doAssert fileExists(layer1Bin), "Layer 1 bin MUST exist after crash"

  var curStep, curLayer: int
  doAssert checkpoint.readProgressCursor(ckptTest2, curStep, curLayer)
  doAssert curStep == 0 and curLayer == 1, &"Cursor should be step=0, layer=1, got {curStep}, {curLayer}"
  echo "  [Test] Layer 1 progress and activation store verified on disk."

  # 3. Resume Step 0 without forward reconstruction!
  putEnv("BYBY_TEST_CRASH_AFTER_BACKWARD_LAYER", "")
  gLastStepForwardRan = false

  # Call trainStepGraftStreaming for step 0 again
  let lossResumed2 = trainStepGraftStreaming(mTest2, loraTest2, stateTest2, adamTest2, sidecarTest2,
    CFG_NEWBLOCK_LR, idsBatch0, tgtBatch0, ctx, stepIndex = 0)

  # CRITICAL ASSERTIONS:
  doAssert not gLastStepForwardRan, "CRITICAL: Layer-level resume MUST NOT rerun forward pass!"
  echo "  [Test] SUCCESS: Forward pass was SKIPPED on layer resume (gLastStepForwardRan == false)!"
  doAssert abs(lossResumed2 - lossBase2) < 1e-4'f32,
    &"Resumed loss {lossResumed2:.6f} != baseline loss {lossBase2:.6f}"
  echo &"  [Test] Resumed step 0 loss={lossResumed2:.6f} matches baseline!"

  # Check that all LoRA adapter weights match baseline exactly
  for targetName in targets2:
    let adBase = loraBase2.adapters[targetName]
    let adTest = loraTest2.adapters[targetName]
    assertTensorsClose(adBase.A.data, adTest.A.data, targetName & "::A")
    assertTensorsClose(adBase.B.data, adTest.B.data, targetName & "::B")
  echo "  [Test] All LoRA weights match baseline perfectly across both blocks!"

  # Commit Step 0 (as done in epoch/step loop)
  checkpoint.clearLayerProgress(ckptTest2)
  removeActivationResume(0)
  doAssert not fileExists(actFile), "Activation scratch should be cleaned on commit"
  doAssert not fileExists(stateFile), "State file should be cleaned on commit"

  # 4. Continue to Step 1 (Verifying epoch-like continuity)
  putEnv("BYBY_CHECKPOINT_PATH", ckptBase2)
  putEnv("BYBY_SCRATCH_DIR", scratchBase2)
  gLastStepForwardRan = false
  let lossStep1Base = trainStepGraftStreaming(mBase2, loraBase2, stateBase2, adamBase2, sidecarBase2,
    CFG_NEWBLOCK_LR, idsBatch1, tgtBatch1, ctx, stepIndex = 1)
  checkpoint.clearLayerProgress(ckptBase2)
  removeActivationResume(1)

  putEnv("BYBY_CHECKPOINT_PATH", ckptTest2)
  putEnv("BYBY_SCRATCH_DIR", scratchTest2)
  gLastStepForwardRan = false
  let lossStep1Test = trainStepGraftStreaming(mTest2, loraTest2, stateTest2, adamTest2, sidecarTest2,
    CFG_NEWBLOCK_LR, idsBatch1, tgtBatch1, ctx, stepIndex = 1)
  checkpoint.clearLayerProgress(ckptTest2)
  removeActivationResume(1)

  doAssert gLastStepForwardRan, "Step 1 MUST run forward pass normally"
  doAssert abs(lossStep1Test - lossStep1Base) < 1e-4'f32,
    &"Step 1 loss {lossStep1Test:.6f} != baseline {lossStep1Base:.6f}"
  for targetName in targets2:
    let adBase = loraBase2.adapters[targetName]
    let adTest = loraTest2.adapters[targetName]
    assertTensorsClose(adBase.A.data, adTest.A.data, targetName & "::A (step 1)")
    assertTensorsClose(adBase.B.data, adTest.B.data, targetName & "::B (step 1)")
  echo &"  [Test] Step 1 loss={lossStep1Test:.6f} matches baseline! Continuous training confirmed."

  # ─────────────────────────────────────────────────────────────
  # TEST 2: 4-Layer Model ("RRRR") - Crash after backward block 2
  # Backward order: block 3 -> block 2 -> crash! (blocks 1, 0 remaining)
  # ─────────────────────────────────────────────────────────────
  echo "\n--- TEST 2: 4-Layer Model (RRRR) ---"
  let pattern4 = "RRRR"
  var mBase4 = initTestModel(pattern4)
  var mTest4 = initTestModel(pattern4)
  copySharedWeights(mBase4, mTest4)

  var targets4: seq[string] = @[]
  for blk in 0 ..< 4:
    targets4.add(&"block{blk}.mambaReal.inProj.weight")
    targets4.add(&"block{blk}.mambaReal.outProj.weight")

  var loraBase4 = attachLora(mBase4, targets4, r = 2, alpha = 4'f32)
  var loraTest4 = cloneLora(loraBase4)
  var stateBase4 = newLoraTrainState(loraBase4, lr = 1e-3'f32)
  var stateTest4 = cloneLoraState(stateBase4)

  let dirBase4 = testRoot / "base4"
  let dirTest4 = testRoot / "test4"
  createDir(dirBase4)
  createDir(dirTest4)

  let ckptBase4 = dirBase4 / "model.bin"
  let ckptTest4 = dirTest4 / "model.bin"
  let scratchBase4 = dirBase4 / "scratch"
  let scratchTest4 = dirTest4 / "scratch"

  var adamBase4 = initAdamDiskStore(dirBase4 / "adam")
  var sidecarBase4 = initAttnWeightDiskStore(dirBase4 / "sidecar")
  var adamTest4 = initAdamDiskStore(dirTest4 / "adam")
  var sidecarTest4 = initAttnWeightDiskStore(dirTest4 / "sidecar")

  # 1. Baseline 4-layer Step 0
  putEnv("BYBY_CHECKPOINT_PATH", ckptBase4)
  putEnv("BYBY_SCRATCH_DIR", scratchBase4)
  gLastStepForwardRan = false
  let lossBase4 = trainStepGraftStreaming(mBase4, loraBase4, stateBase4, adamBase4, sidecarBase4,
    CFG_NEWBLOCK_LR, idsBatch0, tgtBatch0, ctx, stepIndex = 0)
  checkpoint.clearLayerProgress(ckptBase4)
  removeActivationResume(0)
  doAssert gLastStepForwardRan, "Baseline step 0 MUST have executed forward pass"
  echo &"  [Baseline] Step 0 loss={lossBase4:.6f}"

  # 2. Test 4-layer Step 0 - crash after block 2 (so blocks 3 and 2 saved, blocks 1 and 0 remain)
  putEnv("BYBY_CHECKPOINT_PATH", ckptTest4)
  putEnv("BYBY_SCRATCH_DIR", scratchTest4)
  putEnv("BYBY_TEST_CRASH_AFTER_BACKWARD_LAYER", "2")
  gLastStepForwardRan = false

  var crashed4 = false
  try:
    discard trainStepGraftStreaming(mTest4, loraTest4, stateTest4, adamTest4, sidecarTest4,
      CFG_NEWBLOCK_LR, idsBatch0, tgtBatch0, ctx, stepIndex = 0)
  except CatchableError as e:
    crashed4 = true
    echo "  [Test] Caught expected simulated crash: ", e.msg
    doAssert e.msg == "SIMULATED_CRASH_AFTER_BACKWARD_LAYER_2"

  doAssert crashed4, "Test step 0 should have crashed after backward layer 2"
  doAssert checkpoint.readProgressCursor(ckptTest4, curStep, curLayer)
  doAssert curStep == 0 and curLayer == 2, &"Cursor should be step=0, layer=2, got {curStep}, {curLayer}"

  # 3. Resume 4-layer Step 0 from block 1 without forward!
  putEnv("BYBY_TEST_CRASH_AFTER_BACKWARD_LAYER", "")
  gLastStepForwardRan = false

  let lossResumed4 = trainStepGraftStreaming(mTest4, loraTest4, stateTest4, adamTest4, sidecarTest4,
    CFG_NEWBLOCK_LR, idsBatch0, tgtBatch0, ctx, stepIndex = 0)

  doAssert not gLastStepForwardRan, "CRITICAL: Multi-layer resume MUST NOT rerun forward pass!"
  echo "  [Test] SUCCESS: Multi-layer backward resumed directly from block 1 without forward!"
  doAssert abs(lossResumed4 - lossBase4) < 1e-4'f32,
    &"Resumed loss {lossResumed4:.6f} != baseline loss {lossBase4:.6f}"

  for targetName in targets4:
    let adBase = loraBase4.adapters[targetName]
    let adTest = loraTest4.adapters[targetName]
    assertTensorsClose(adBase.A.data, adTest.A.data, targetName & "::A")
    assertTensorsClose(adBase.B.data, adTest.B.data, targetName & "::B")
  echo "  [Test] All 4 blocks matched baseline weights after resume!"

  # Commit Step 0 and clean up
  checkpoint.clearLayerProgress(ckptTest4)
  removeActivationResume(0)

  echo "\n=== ALL TESTS PASSED: Layer-level resume works seamlessly without forward reconstruction! ==="

when isMainModule:
  runTests()
