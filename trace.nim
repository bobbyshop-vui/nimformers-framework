## trace.nim - record a model run into a FIR trace (`nf.trace`).
##
## A trace is the file the FIR lifter (`firlift`) turns into `model.fir`.  The recorder lives in
## `vendor/bybylang/probe.nim`; this module is its small public face: start, stop, step markers and
## a `withTrace` block, with the data-capture environment handled for the caller.
##
##   import nimformers
##
##   withTrace("run.fprobe", "decode:2"):
##     for pos, tok in tokens:
##       traceStep(tsDecode, pos, pos, tok):
##         runOneToken(tok)
##
## `import nimformers` re-exports everything here.  BybyLang code uses it with
## `import nim "trace"`.  Python and C reach the same recorder through `nimformers.trace` and the
## `nimf_trace_*` C ABI.

import std/os
import vendor/bybylang/probe as fprobe

type
  TraceStepKind* = enum
    tsPrefill, tsDecode

const
  TraceDataEnv = "BYBY_FIR_PROBE_DATA"
  TraceDataMaxEnv = "BYBY_FIR_PROBE_DATA_MAX"

var
  savedData, savedDataMax: string
  hadData, hadDataMax = false
  traceOwnsEnv = false

proc traceActive*(): bool =
  ## True while a trace is being recorded.
  fprobe.firProbeOn

proc traceFile*(): string =
  ## Path of the trace being recorded ("" when none was started in this process).
  fprobe.probeDebugPath()

proc traceStop*() =
  ## Flush the trace file and close the exact-bytes sidecar (`<path>.data`).  Safe to call twice.
  fprobe.firProbeStop()
  if traceOwnsEnv:
    if hadData: putEnv(TraceDataEnv, savedData) else: delEnv(TraceDataEnv)
    if hadDataMax: putEnv(TraceDataMaxEnv, savedDataMax) else: delEnv(TraceDataMaxEnv)
    traceOwnsEnv = false

proc traceStart*(path: string; steps = ""; data = true; dataMax = 0) =
  ## Start recording to `path` (an existing directory is required).
  ##
  ## * `steps`   which host steps to record, e.g. `"decode:2"` or `"prefill:8,decode:2"`.
  ##             Empty records everything.  Two decode steps are the minimum the lifter needs to
  ##             tell state that repeats from state that changes.
  ## * `data`    also write the exact bytes of every host upload to `<path>.data`
  ##             (needed to lift weights and tables that are not in a checkpoint).
  ## * `dataMax` with `data`, skip uploads larger than this many bytes (0 = unlimited);
  ##             a skipped upload keeps its fingerprint so a checkpoint can still match it.
  ##
  ## Starting while a trace is active stops the previous one first.
  traceStop()
  hadData = existsEnv(TraceDataEnv)
  savedData = getEnv(TraceDataEnv)
  hadDataMax = existsEnv(TraceDataMaxEnv)
  savedDataMax = getEnv(TraceDataMaxEnv)
  traceOwnsEnv = true
  putEnv(TraceDataEnv, if data: "1" else: "0")
  putEnv(TraceDataMaxEnv, $max(dataMax, 0))
  try:
    fprobe.firProbeStart(path, steps)
  except CatchableError:
    traceStop()
    raise

proc traceBegin*(kind: TraceStepKind; index, position, token: int; tokens = 1) =
  ## Open one host step (a prefill chunk or a decode token).  No-op when no trace is active.
  fprobe.probeStepBegin(if kind == tsPrefill: "prefill" else: "decode", index, position, token, tokens)

proc traceEnd*() =
  ## Close the step opened by `traceBegin`.  No-op when no trace is active.
  fprobe.probeStepEnd()

template withTrace*(path: string; steps: string; data: bool; dataMax: int; body: untyped) =
  ## Record `body` and always stop the trace afterwards.
  traceStart(path, steps, data, dataMax)
  try:
    body
  finally:
    traceStop()

template withTrace*(path: string; steps: string; body: untyped) =
  withTrace(path, steps, true, 0, body)

template withTrace*(path: string; body: untyped) =
  withTrace(path, "", true, 0, body)

template traceStep*(kind: TraceStepKind; index, position, token: int; body: untyped) =
  ## Mark `body` as one host step.
  traceBegin(kind, index, position, token)
  try:
    body
  finally:
    traceEnd()
