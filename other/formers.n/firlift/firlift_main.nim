## firlift — rebuild model.fir from an fprobe 2 trace.
import std/[os, strutils, tables]
import ./firlift
import ./corelift
import ./sessionlift

proc usage() =
  echo "usage: firlift --trace FILE --config FILE --out DIR [--name NAME] [--embedding TENSOR] [--report] [--coverage] [--core-idiom [--q4]]"
  quit(2)

proc main() =
  var tracePath, configPath, outDir, name = ""
  var doReport = false
  var useQ4 = false
  var coreIdiom = false
  var coverage = false
  var args = commandLineParams()
  var i = 0
  while i < args.len:
    case args[i]
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
    of "--coverage":
      coverage = true
    of "--core-idiom":
      coreIdiom = true
    else:
      usage()
    inc i
  if tracePath.len == 0 or configPath.len == 0 or outDir.len == 0: usage()
  if name.len == 0: name = extractFilename(parentDir(configPath))
  let trace = parseTrace(readFile(tracePath))
  let cfgText = readFile(configPath)
  var embedding = ""
  for i in 0 ..< args.len-1:
    if args[i] == "--embedding": embedding = args[i+1]
  if coverage:
    let report = liftSession(trace, cfgText, name, embedding, true)
    echo report
    if "FAILED" in report: quit(1)
    quit(0)
  let model = if trace.graphOps().len > 0: buildModel(trace, cfgText, name)
              elif coreIdiom: liftCore(trace, cfgText, name, embedding, useQ4)
              else: liftSession(trace, cfgText, name, embedding)
  createDir(outDir)
  writeFile(outDir / "model.fir", model)
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

main()
