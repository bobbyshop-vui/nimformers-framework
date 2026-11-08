## firlift — rebuild model.fir from an fprobe 2 trace.
import ./fuse
import std/[os, strutils, tables, json]
import ./firlift
import ./corelift
import ./sessionlift
import ./steps

proc fnvRange(f: File; offset, bytes: int): string =
  if bytes <= 0: return "0"
  f.setFilePos(offset)
  var hash = 0xcbf29ce484222325'u64
  let chunk = min(bytes, 1 shl 20)
  var buffer = newString(chunk)
  var left = bytes
  while left > 0:
    let n = min(chunk, left)
    if f.readBuffer(addr buffer[0], n) != n:
      raise newException(IOError, "truncated probe sidecar")
    for i in 0 ..< n:
      hash = (hash xor uint64(uint8(buffer[i]))) * 0x100000001b3'u64
    left -= n
  toHex(hash, 16)

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
  let source = tracePath & ".data"
  if not fileExists(source):
    raise newException(IOError, "missing upload sidecar " & source)
  var input = open(source, fmRead)
  var hashes = newSeq[string](entries.len)
  for i, entry in entries:
    hashes[i] = fnvRange(input, entry.srcOffset, entry.bytes)
  input.close()
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
  input = open(source, fmRead)
  var target = open(outDir / "trace.data", fmWrite)
  try:
    var buffer = newString(min(total, 1 shl 20))
    if buffer.len == 0: buffer = newString(1)
    for i in unique:
      let entry = entries[i]
      input.setFilePos(entry.srcOffset)
      var left = entry.bytes
      while left > 0:
        let n = min(buffer.len, left)
        if input.readBuffer(addr buffer[0], n) != n:
          raise newException(IOError, "truncated probe sidecar")
        if target.writeBuffer(addr buffer[0], n) != n:
          raise newException(IOError, "cannot write compacted trace.data")
        left -= n
  finally:
    target.close()
    input.close()
  for i, entry in entries:
    lines[entry.index] = rewritten[i]
  var rebuilt = ""
  for i, line in lines:
    if i > 0: rebuilt.add "\n"
    rebuilt.add line
  model = rebuilt & "\n"

proc usage() =
  echo "usage: firlift --trace FILE --config FILE --out DIR [--weights CHECKPOINT]... [--slice A:B | --step N] [--name NAME] [--embedding TENSOR] [--report] [--coverage] [--fuse] [--grid-tg N | --dynamic-grid-tg N] [--core-idiom [--q4]]"
  echo "  --slice A:B: zero-based complete session range, B exclusive"
  echo "  --step N: zero-based occurrence of the automatically detected repeating unit"
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
  var dynamicGrid = false
  var args = commandLineParams()
  var i = 0
  while i < args.len:
    case args[i]
    of "--weights", "--slice", "--step", "--embedding":
      let flag = args[i]
      inc i
      if i >= args.len: usage()
      case flag
      of "--weights": weightsPaths.add args[i]
      of "--slice": slice = args[i]
      of "--step": step = parseInt(args[i])
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
    else:
      usage()
    inc i
  if tracePath.len == 0 or configPath.len == 0 or outDir.len == 0: usage()
  if name.len == 0: name = extractFilename(parentDir(configPath))
  if slice.len > 0 and step >= 0: raise newException(ValueError, "choose --slice or --step")
  var parsed = parseTrace(readFile(tracePath))
  parsed.sourcePath = absolutePath(tracePath)
  let cfgJson = parseJson(readFile(configPath))
  var dims = Dims()
  for key in ["hidden_size", "n_embd", "d_model", "dim"]:
    if cfgJson.hasKey(key) and cfgJson[key].kind == JInt: dims.hidden = cfgJson[key].getInt()
  if cfgJson.hasKey("vocab_size") and cfgJson["vocab_size"].kind == JInt: dims.vocab = cfgJson["vocab_size"].getInt()
  let selected = selectStep(parsed, step, slice, dims)
  let trace = selected.trace
  if selected.sessionCount > 0:
    echo "selected sessions ", selected.firstSession, ":", selected.firstSession+selected.sessionCount,
      " (", selected.dispatches, " dispatches, occurrence ", selected.occurrence, ")"
  let cfgText = readFile(configPath)
  if coverage:
    let report = liftSessionFiles(trace, cfgText, name, embedding, true, weightsPaths)
    echo report
    if "FAILED" in report: quit(1)
    quit(0)
  var model = if trace.graphOps().len > 0: buildModel(trace, cfgText, name)
              elif coreIdiom: liftCore(trace, cfgText, name, embedding, useQ4)
              else: liftSessionFiles(trace, cfgText, name, embedding, false, weightsPaths, fuseDispatches, gridTg, dynamicGrid)
  if gridTg > 0:
    echo "persistent chains: ", fuseStats.chains, " built from ", fuseStats.chained, " dispatches (", gridTg, " threadgroups)"
  if fuseDispatches or gridTg > 0:
    echo "dispatch fusion: ", fuseStats.fused, " of ", fuseStats.pairs, " consecutive raw pairs merged, ", fuseStats.bindable, " share one binding set"
    for reason, count in fuseStats.reasons: echo "  rejected ", count, "x: ", reason
  createDir(outDir)
  if " trace.data " in model:
    compactTraceData(model, tracePath, outDir)
  writeFile(outDir / "model.fir", model)
  # Auxiliary weight files (raw .bin uploads outside the primary checkpoint) are
  # referenced by file name so the model directory is self-contained.
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
  if configPath != outDir / "config.json":
    copyFile(configPath, outDir / "config.json")
  if doReport:
    var graphNodes = 0
    for op in trace.ops:
      if op.parent < 0 and op.attrs.hasKey("kind"): inc graphNodes
    echo "trace: backend=", trace.backend, " buffers=", trace.bufs.len,
      " programs=", trace.programs.len, " ops=", trace.ops.len,
      " graph_nodes=", graphNodes
    report(model)
  echo "wrote ", outDir / "model.fir"

try:
  main()
except CatchableError as e:
  stderr.writeLine("firlift: " & e.msg)
  quit(1)
