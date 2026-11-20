## firlift — rebuild model.fir from an fprobe 2 trace.
import ./fuse
import std/[os, strutils, tables, json]
import ./firlift
import ./corelift
import ./sessionlift
import ./steps
import ./sidecar
import ./record
import probe_data

proc compactTraceData(model: var string; tracePath, outDir: string) =
  ## Copy only the byte ranges the model references and merge ranges with equal
  ## contents. Repeated initializers (one per layer) become one copy.
  var lines = model.splitLines()
  var entries: seq[tuple[index, srcOffset, bytes: int, name, fingerprint: string]]
  for i, line in lines:
    if not line.startsWith("init "): continue
    let parts = line.splitWhitespace()
    if parts.len < 5 or parts[2] != "trace.data": continue
    entries.add (i, parseInt(parts[3]), parseInt(parts[4]), parts[1],
      (if parts.len > 5: parts[5] else: ""))
  if entries.len == 0: return
  if not sidecarAvailable(tracePath):
    raise newException(IOError, "missing upload sidecar " & sidecarPath(tracePath))
  var hashes = newSeq[string](entries.len)
  for i, entry in entries:
    hashes[i] = if entry.bytes <= 0: "0" else: sidecarFnv(tracePath, entry.srcOffset, entry.bytes)
  var assigned = initTable[string, int]()
  var unique: seq[int]
  var total = 0
  var rewritten = newSeq[string](entries.len)
  for i, entry in entries:
    let key = $entry.bytes & ":" & hashes[i]
    if not assigned.hasKey(key):
      assigned[key] = total
      unique.add i
      total += entry.bytes
    var line = "init " & entry.name & " trace.data " & $assigned[key] & " " & $entry.bytes
    if entry.fingerprint.len > 0: line.add " " & entry.fingerprint
    rewritten[i] = line
  var target = open(outDir / "trace.data", fmWrite)
  try:
    for i in unique:
      let entry = entries[i]
      sidecarWrite(tracePath, entry.srcOffset, entry.bytes, target)
  finally:
    target.close()
  for i, entry in entries:
    lines[entry.index] = rewritten[i]
  var rebuilt = ""
  for i, line in lines:
    if i > 0: rebuilt.add "\n"
    rebuilt.add line
  model = rebuilt & "\n"

proc usage() =
  echo "usage: firlift (--trace FILE | --run COMMAND...) --config FILE --out DIR [--weights CHECKPOINT]... [--slice A:B | --step N] [--context TOKENS] [--name NAME] [--embedding TENSOR] [--report] [--coverage] [--fuse] [--grid-tg N | --dynamic-grid-tg N] [--core-idiom [--q4]]"
  echo "  --run COMMAND...: record the command's trace in memory through pipes (no .fprobe or .data file); must be last"
  echo "  Mamba-2/Codestral: record with the original .nimq as the first --weights; output is model.fir and config.json only"
  echo "  --slice A:B: zero-based complete session range, B exclusive"
  echo "  --step N: zero-based occurrence of the automatically detected repeating unit"
  echo "  --context TOKENS: cap the lifted graph's KV buffers and position kernels (also writes config.json)"
  quit(2)

proc main() =
  var tracePath, configPath, outDir, name = ""
  var weightsPaths: seq[string]
  var slice, embedding: string
  var step = -1
  var doReport = false
  var useQ4 = false
  var coreIdiom = false
  var coverage = false
  var fuseDispatches = false
  var gridTg = 0
  var contextSize = 0
  var dynamicGrid = false
  var recordRun = false
  var runCommand: seq[string]
  var args = commandLineParams()
  var i = 0
  while i < args.len:
    case args[i]
    of "--weights", "--slice", "--step", "--embedding", "--context":
      let flag = args[i]
      inc i
      if i >= args.len: usage()
      case flag
      of "--weights": weightsPaths.add args[i]
      of "--slice": slice = args[i]
      of "--step": step = parseInt(args[i])
      of "--context": contextSize = parseInt(args[i])
      else: embedding = args[i]
    of "--trace":
      inc i
      if i >= args.len: usage()
      tracePath = args[i]
    of "--config":
      inc i
      if i >= args.len: usage()
      configPath = args[i]
    of "--out":
      inc i
      if i >= args.len: usage()
      outDir = args[i]
    of "--name":
      inc i
      if i >= args.len: usage()
      name = args[i]
    of "--report":
      doReport = true
    of "--q4":
      useQ4 = true
    of "--dynamic-grid-tg":
      inc i
      if i >= args.len: usage()
      gridTg = parseInt(args[i])
      dynamicGrid = true
    of "--grid-tg":
      inc i
      if i >= args.len: usage()
      gridTg = parseInt(args[i])
    of "--fuse":
      fuseDispatches = true
    of "--coverage":
      coverage = true
    of "--core-idiom":
      coreIdiom = true
    of "--run":
      # Everything after --run is the command to record; it is not parsed here.
      runCommand = args[i + 1 .. ^1]
      recordRun = true
      break
    else:
      usage()
    inc i
  if recordRun:
    if tracePath.len > 0: raise newException(ValueError, "--run records in memory; do not pass --trace")
    if runCommand.len == 0: usage()
    tracePath = MemorySource
  if tracePath.len == 0 or configPath.len == 0 or outDir.len == 0: usage()
  if name.len == 0: name = extractFilename(parentDir(configPath))
  if slice.len > 0 and step >= 0: raise newException(ValueError, "choose --slice or --step")
  if contextSize < 0: raise newException(ValueError, "--context must be a positive token count")
  # HF configs written by Python may hold bare Infinity/NaN (Codestral's
  # "time_step_limit": [0.0, Infinity]); they become null, as in the CLI's detection.
  let rawCfgText = readFile(configPath)
  var cfgJson: JsonNode
  var cfgSanitized = false
  try:
    cfgJson = parseJson(rawCfgText)
  except JsonParsingError:
    cfgJson = parseJson(rawCfgText.multiReplace(("-Infinity", "null"), ("Infinity", "null"), ("NaN", "null")))
    cfgSanitized = true
  if contextSize > 0:
    let contextKeys = ["max_position_embeddings", "n_positions", "n_ctx", "seq_length"]
    var declaredContext = high(int)
    var declared = false
    for key in contextKeys:
      if cfgJson.hasKey(key) and cfgJson[key].kind == JInt:
        declaredContext = min(declaredContext, cfgJson[key].getInt())
        declared = true
    if declared and declaredContext <= 0:
      raise newException(ValueError, "--context requires a positive context length in config.json")
    if declared and contextSize > declaredContext:
      raise newException(ValueError, "--context cannot exceed config.json context length " & $declaredContext)
    if declared:
      for key in contextKeys:
        if cfgJson.hasKey(key) and cfgJson[key].kind == JInt:
          cfgJson[key] = newJInt(contextSize)
    else:
      # A recurrent model declares no context length; --context then sets the
      # graph's position capacity.
      cfgJson["max_position_embeddings"] = newJInt(contextSize)
  let modelType = cfgJson{"model_type"}.getStr("").toLowerAscii()
  let architectures = cfgJson{"architectures"}
  let mamba2 = modelType == "mamba2" or (modelType.len == 0 and architectures != nil and
    architectures.kind == JArray and architectures.len > 0 and architectures[0].getStr("") == "Mamba2ForCausalLM")
  let codestral = mamba2 and not coverage
  if codestral:
    if weightsPaths.len == 0 or not weightsPaths[0].endsWith(".nimq"):
      raise newException(ValueError, "Codestral needs the original .nimq as the first --weights")
    if coreIdiom or fuseDispatches or gridTg > 0:
      raise newException(ValueError, "Codestral FIR requires the recorded session and dispatch geometry")
  var parsed: Trace
  if recordRun:
    var recorded = recordCommand(runCommand, mamba2)
    if recorded.exitCode != 0:
      let message = "the recorded command exited with code " & $recorded.exitCode
      if codestral: raise newException(IOError, message)
      stderr.writeLine("firlift: " & message)
    if recorded.trace.len == 0:
      raise newException(IOError, "the command wrote no trace (does it use the core recorder?)")
    echo "recorded in memory: trace ", recorded.trace.len, " bytes, upload data ", recorded.data.len, " bytes"
    useMemory(move recorded.data)
    parsed = parseTrace(recorded.trace)
    parsed.sourcePath = MemorySource
  else:
    parsed = parseTrace(readFile(tracePath))
    parsed.sourcePath = absolutePath(tracePath)
  if codestral:
    if parsed.backend != "vulkan" or parsed.codestralTokens <= 0:
      raise newException(ValueError, "re-record with the current formers-cli on Vulkan for a Codestral graph without sidecars")
    let base = weightsPaths[0]
    if parsed.codestralBase != fingerprintFile(base, 0, getFileSize(base).int):
      raise newException(ValueError, "first --weights differs from the Codestral checkpoint used to record the trace")
    if weightsPaths.len > 1:
      let adapter = weightsPaths[1]
      if weightsPaths.len > 2 or parsed.codestralLora != fingerprintFile(adapter, 0, getFileSize(adapter).int):
        raise newException(ValueError, "auxiliary --weights must be the original LoRA used to record the trace")
      weightsPaths.setLen(1)
    cfgJson["fir_codestral"] = %*{"base": parsed.codestralBase, "lora": parsed.codestralLora,
      "prompt_tokens": parsed.codestralTokens}
  var dims = Dims()
  for key in ["hidden_size", "n_embd", "d_model", "dim"]:
    if cfgJson.hasKey(key) and cfgJson[key].kind == JInt: dims.hidden = cfgJson[key].getInt()
  if cfgJson.hasKey("vocab_size") and cfgJson["vocab_size"].kind == JInt: dims.vocab = cfgJson["vocab_size"].getInt()
  let selected = selectStep(parsed, step, slice, dims)
  let trace = selected.trace
  if selected.sessionCount > 0:
    echo "selected sessions ", selected.firstSession, ":", selected.firstSession+selected.sessionCount,
      " (", selected.dispatches, " dispatches, occurrence ", selected.occurrence, ")"
    if selected.resetCopies > 0:
      echo "state resets left to the graph's initial contents: ", selected.resetCopies, " copies"
  let rewriteConfig = contextSize > 0 or cfgSanitized or codestral
  let cfgText = if rewriteConfig: $cfgJson else: rawCfgText
  if coverage:
    let report = liftSessionFiles(trace, cfgText, name, embedding, true, weightsPaths)
    echo report
    if "FAILED" in report: quit(1)
    quit(0)
  var model = if trace.graphOps().len > 0: buildModel(trace, cfgText, name)
              elif coreIdiom: liftCore(trace, cfgText, name, embedding, useQ4)
              else: liftSessionFiles(trace, cfgText, name, embedding, false, weightsPaths, fuseDispatches, gridTg, dynamicGrid, codestral)
  if droppedSingleRowUnits > 0:
    echo "single-token units left to the decode step: ", droppedSingleRowUnits
  if droppedFailedCalls > 0:
    echo "session calls that did not run (their native fallback follows): ", droppedFailedCalls
  if replacedFp4Fallbacks > 0:
    echo "failed FP4 GEMMs lowered in place of their native fallback tiles: ", replacedFp4Fallbacks
  if gridTg > 0:
    echo "persistent chains: ", fuseStats.chains, " built from ", fuseStats.chained, " dispatches (", gridTg, " threadgroups)"
  if fuseDispatches or gridTg > 0:
    echo "dispatch fusion: ", fuseStats.fused, " of ", fuseStats.pairs, " consecutive raw pairs merged, ", fuseStats.bindable, " share one binding set"
    for reason, count in fuseStats.reasons: echo "  rejected ", count, "x: ", reason
  if codestral:
    for line in model.splitLines():
      if line.startsWith("init ") and line.splitWhitespace()[2] != "checkpoint":
        raise newException(ValueError, "Codestral lift would require an upload sidecar: " & line)
    createDir(outDir)
  else:
    createDir(outDir)
    if " trace.data " in model: compactTraceData(model, tracePath, outDir)
    # Legacy graphs may refer to an auxiliary checkpoint next to model.fir.
    for line in model.splitLines():
      if not line.startsWith("init "): continue
      let parts = line.splitWhitespace()
      if parts.len < 3: continue
      let name = parts[2]
      if name in ["checkpoint", "trace.data"]: continue
      for path in weightsPaths:
        if extractFilename(path) & ".raw" == name:
          let target = outDir / name
          if not fileExists(target):
            try: createSymlink(absolutePath(path), target)
            except CatchableError: copyFile(path, target)
          break
  writeFile(outDir / "model.fir", model)
  if rewriteConfig: writeFile(outDir / "config.json", cfgText & "\n")
  elif configPath != outDir / "config.json": copyFile(configPath, outDir / "config.json")
  if doReport:
    var graphNodes = 0
    for op in trace.ops:
      if op.parent < 0 and op.attrs.hasKey("kind"): inc graphNodes
    echo "trace: backend=", trace.backend, " device=", trace.device, " buffers=", trace.bufs.len,
      " programs=", trace.programs.len, " ops=", trace.ops.len,
      " graph_nodes=", graphNodes
    report(model)
  echo "wrote ", outDir / "model.fir"

try:
  main()
except CatchableError as e:
  stderr.writeLine("firlift: " & e.msg)
  quit(1)
