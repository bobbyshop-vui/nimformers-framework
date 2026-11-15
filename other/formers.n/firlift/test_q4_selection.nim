## Regression: Q4 GEMV lift selection follows backend policy and labels generated fallbacks.
## nim c -r --path:vendor/bybylang other/formers.n/firlift/test_q4_selection.nim
import std/[os, strutils, tables]
import ./firlift
import ./sessionlift

proc configFor(k: int): string =
  "{\"hidden_size\":" & $k & ",\"max_position_embeddings\":128,\"vocab_size\":4}"

type EnvValue = object
  exists: bool
  value: string

proc saveEnv(name: string): EnvValue =
  EnvValue(exists: existsEnv(name), value: getEnv(name))

proc restoreEnv(name: string; value: EnvValue) =
  if value.exists: putEnv(name, value.value)
  else: delEnv(name)

proc setEnv(name, value: string) = putEnv(name, value)

proc liftQ4(backend: string; k = 128; group = 32; nGroups = 4): string =
  var trace: Trace
  trace.backend = backend
  trace.steps = 1
  let a = PortRef(name: "a", buffer: 0, bytes: k * 4)
  let w = PortRef(name: "w", buffer: 1, bytes: 1)
  let c = PortRef(name: "c", buffer: 2, bytes: 4 * 4)
  let upload = Op(id: 0, order: 0, step: 1, impl: "session:sessionUpload",
    parent: -1, outs: @[a])
  trace.ops.add upload
  trace.transfers.add TransferRec(kind: "upload", op: upload.id, order: upload.order, dst: a, n: k)
  var op = Op(id: 1, order: 1, step: 1, impl: "session:sessionMatmulQ4",
    tag: "aten::matmul", parent: -1, ins: @[a, w], outs: @[c])
  for (key, value) in [
      ("M", "1"), ("k", $k), ("n", "4"), ("group", $group),
      ("groupsPerRow", $nGroups)]:
    op.attrs[key] = value
  trace.ops.add op
  let read = Op(id: 2, order: 2, step: 1, impl: "session:sessionRead", parent: -1,
    ins: @[PortRef(name: "h", buffer: c.buffer, bytes: c.bytes)])
  read.attrs["n"] = "4"
  trace.ops.add read
  trace.stepToken = 1
  trace.names.add NameRec(buffer: 1, offset: 0, bytes: 1,
    name: "model.layers.0.q_proj.weight", dtype: "q4r", shape: "(4 " & $k & ")")
  liftSession(trace, configFor(k), "test-model", "")

let saved = [
  saveEnv("FIRLIFT_Q4_GEMV"), saveEnv("FIRLIFT_Q4_LANES"), saveEnv("FIRLIFT_TG"),
  saveEnv("FIRLIFT_SIMD"), saveEnv("FIRLIFT_GENERIC"), saveEnv("FIRLIFT_Q4_WORD_MAX_K")]
try:
  delEnv("FIRLIFT_Q4_GEMV")
  delEnv("FIRLIFT_Q4_LANES")
  delEnv("FIRLIFT_TG")
  delEnv("FIRLIFT_SIMD")
  delEnv("FIRLIFT_GENERIC")
  delEnv("FIRLIFT_Q4_WORD_MAX_K")

  var fir = liftQ4("vulkan")
  doAssert "kernel s.matmul_q4_word_parallel" in fir
  doAssert " q4r model.layers.0.q_proj.weight tensor [4," in fir

  fir = liftQ4("vulkan", k = 11008, group = 128, nGroups = 86)
  doAssert "kernel s.matmul_q4_parallel" in fir
  doAssert "matmul_q4_word_parallel" notin fir

  setEnv("FIRLIFT_Q4_WORD_MAX_K", "11008")
  fir = liftQ4("vulkan", k = 11008, group = 128, nGroups = 86)
  doAssert "kernel s.matmul_q4_word_parallel" in fir

  setEnv("FIRLIFT_Q4_WORD_MAX_K", "0")
  fir = liftQ4("vulkan")
  doAssert "kernel s.matmul_q4_parallel" in fir

  setEnv("FIRLIFT_Q4_GEMV", "word_parallel")
  fir = liftQ4("vulkan")
  doAssert "kernel s.matmul_q4_word_parallel" in fir, "explicit word kernel must override auto threshold"

  setEnv("FIRLIFT_Q4_GEMV", "parallel")
  fir = liftQ4("vulkan")
  doAssert "kernel s.matmul_q4_parallel" in fir

  delEnv("FIRLIFT_Q4_GEMV")
  setEnv("FIRLIFT_SIMD", "1")
  fir = liftQ4("vulkan")
  doAssert "kernel s.matmul_q4_simd" in fir

  delEnv("FIRLIFT_SIMD")
  setEnv("FIRLIFT_GENERIC", "1")
  setEnv("FIRLIFT_TG", "16")
  fir = liftQ4("vulkan")
  doAssert "kernel s.matmul_q4_parallel" in fir

  delEnv("FIRLIFT_TG")
  fir = liftQ4("vulkan")
  doAssert "kernel s.matmul_q4_scalar" in fir

  delEnv("FIRLIFT_GENERIC")
  setEnv("FIRLIFT_Q4_GEMV", "word_parallel")
  fir = liftQ4("vulkan", k = 130, group = 32, nGroups = 5)
  doAssert "kernel s.matmul_q4_scalar" in fir
  doAssert "matmul_q4_word_parallel" notin fir

  delEnv("FIRLIFT_Q4_GEMV")
  fir = liftQ4("metal")
  doAssert "kernel s.matmul_q4_simd" in fir
finally:
  restoreEnv("FIRLIFT_Q4_GEMV", saved[0])
  restoreEnv("FIRLIFT_Q4_LANES", saved[1])
  restoreEnv("FIRLIFT_TG", saved[2])
  restoreEnv("FIRLIFT_SIMD", saved[3])
  restoreEnv("FIRLIFT_GENERIC", saved[4])
  restoreEnv("FIRLIFT_Q4_WORD_MAX_K", saved[5])

echo "PASS Q4 GEMV backend selection, overrides and fallback labels"
