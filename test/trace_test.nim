## Nim API test for trace.nim (no GPU needed): start/stop, flags, errors.
import std/[os, strutils]
import ../trace
import ../vendor/bybylang/micro_kernel/[session_vm, jit_vm]
import ../vendor/bybylang/probe as fprobe

let dir = getTempDir() / "nf_trace_test_nim"
createDir(dir)

doAssert not traceActive()
let path = dir / "a.fprobe"
traceStart(path, steps = "decode:2", data = false)
doAssert traceActive()
doAssert traceFile() == path
traceStop()
traceStop()  # harmless twice
doAssert not traceActive()
doAssert readFile(path).startsWith("fprobe")

withTrace(dir / "b.fprobe", "decode:1"):
  doAssert traceActive()
doAssert not traceActive()
doAssert fileExists(dir / "b.fprobe")

doAssert getEnv("BYBY_FIR_PROBE_DATA") == ""   # the data flag never leaks out of a trace

var failed = false
try:
  traceStart(dir / "missing" / "c.fprobe")
except CatchableError:
  failed = true
doAssert failed and not traceActive()

failed = false
try:
  traceStart(dir / "d.fprobe", steps = "decode")
except CatchableError:
  failed = true
doAssert failed and not traceActive()

# A kernel compiled BEFORE the trace started must still leave its bytecode in the trace.
var prog: VmProgram
prog.emitULid 0
prog.emitHalt
let key = programKeyNoSession(prog, "trace_before")
traceStart(dir / "e.fprobe", "decode:1", data = false)
doAssert fprobe.probePipelineProgram("jit_" & key) != "none"
doAssert fprobe.probePipelineProgram("vkvm_" & key) != "none"
traceStop()
doAssert "missing_words" notin readFile(dir / "e.fprobe")
removeDir(dir)
echo "trace_test: ok"
