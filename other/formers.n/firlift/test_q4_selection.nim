## Regression: Q4 GEMV lift selection follows backend policy and labels generated fallbacks.
## nim c -r --path:vendor/bybylang other/formers.n/firlift/test_q4_selection.nim
import std/[os, strutils, tables]
import ./firlift
import ./sessionlift

proc configFor(k: int; vocab = 4): string =
  "{\"hidden_size\":" & $k & ",\"max_position_embeddings\":128,\"vocab_size\":" & $vocab & "}"

type EnvValue = object
  exists: bool
  value: string

proc saveEnv(name: string): EnvValue =
  EnvValue(exists: existsEnv(name), value: getEnv(name))

proc restoreEnv(name: string; value: EnvValue) =
  if value.exists: putEnv(name, value.value)
  else: delEnv(name)

proc setEnv(name, value: string) = putEnv(name, value)

proc liftQ4(backend: string; k = 128; group = 32; nGroups = 4; m = 1; n = 4; device = ""): string =
  var trace: Trace
  trace.backend = backend
  trace.device = device
  trace.steps = 1
  let a = PortRef(name: "a", buffer: 0, bytes: m * k * 4)
  let w = PortRef(name: "w", buffer: 1, bytes: 1)
  let c = PortRef(name: "c", buffer: 2, bytes: m * n * 4)
  let upload = Op(id: 0, order: 0, step: 1, impl: "session:sessionUpload",
    parent: -1, outs: @[a])
  trace.ops.add upload
  trace.transfers.add TransferRec(kind: "upload", op: upload.id, order: upload.order, dst: a, n: k)
  var op = Op(id: 1, order: 1, step: 1, impl: "session:sessionMatmulQ4",
    tag: "aten::matmul", parent: -1, ins: @[a, w], outs: @[c])
  for (key, value) in [
      ("M", $m), ("k", $k), ("n", $n), ("group", $group),
      ("groupsPerRow", $nGroups)]:
    op.attrs[key] = value
  trace.ops.add op
  let read = Op(id: 2, order: 2, step: 1, impl: "session:sessionRead", parent: -1,
    ins: @[PortRef(name: "h", buffer: c.buffer, bytes: c.bytes)])
  read.attrs["n"] = $(m * n)
  trace.ops.add read
  trace.stepToken = 1
  trace.names.add NameRec(buffer: 1, offset: 0, bytes: 1,
    name: "model.layers.0.q_proj.weight", dtype: "q4r", shape: "(" & $n & " " & $k & ")")
  liftSession(trace, configFor(k, m * n), "test-model", "")

proc liftFp4(backend, device: string; k = 128; m = 32; n = 64): string =
  ## One FP4-KM prefill GEMM (sessionFp4GemmTensorCore) on a registered weight.
  var trace: Trace
  trace.backend = backend
  trace.device = device
  trace.steps = 1
  let a = PortRef(name: "a", buffer: 0, bytes: m * k * 4)
  let w = PortRef(name: "w", buffer: 1, bytes: n * k div 2 + n * k div 32)
  let lut = PortRef(name: "lut", buffer: 3, bytes: 272 * 4)
  let c = PortRef(name: "c", buffer: 2, bytes: m * n * 4)
  let upload = Op(id: 0, order: 0, step: 1, impl: "session:sessionUpload", parent: -1, outs: @[a])
  trace.ops.add upload
  trace.transfers.add TransferRec(kind: "upload", op: upload.id, order: upload.order, dst: a, n: k)
  var op = Op(id: 1, order: 1, step: 1, impl: "session:sessionFp4GemmTensorCore",
    tag: "aten::matmul", parent: -1, ins: @[a, w, lut], outs: @[c])
  for (key, value) in [("M", $m), ("K", $k), ("N", $n)]:
    op.attrs[key] = value
  trace.ops.add op
  let read = Op(id: 2, order: 2, step: 1, impl: "session:sessionRead", parent: -1,
    ins: @[PortRef(name: "h", buffer: c.buffer, bytes: c.bytes)])
  read.attrs["n"] = $(m * n)
  trace.ops.add read
  trace.stepToken = 1
  trace.names.add NameRec(buffer: 1, offset: 0, bytes: w.bytes,
    name: "backbone.layers.0.mixer.in_proj.weight", dtype: "fp4km", shape: "(" & $n & " " & $k & ")")
  liftSession(trace, configFor(k, m * n), "test-model", "")

let saved = [
  saveEnv("FIRLIFT_Q4_GEMV"), saveEnv("FIRLIFT_Q4_LANES"), saveEnv("FIRLIFT_TG"),
  saveEnv("FIRLIFT_SIMD"), saveEnv("FIRLIFT_GENERIC"), saveEnv("FIRLIFT_Q4_WORD_MAX_K"),
  saveEnv("FIRLIFT_Q4_GEMM"), saveEnv("FIRLIFT_FP4_GEMM")]
try:
  delEnv("FIRLIFT_Q4_GEMV")
  delEnv("FIRLIFT_Q4_LANES")
  delEnv("FIRLIFT_TG")
  delEnv("FIRLIFT_SIMD")
  delEnv("FIRLIFT_GENERIC")
  delEnv("FIRLIFT_Q4_WORD_MAX_K")
  delEnv("FIRLIFT_Q4_GEMM")

  # Vulkan auto: the native decode algorithm as a shape-op micro-kernel.
  var fir = liftQ4("vulkan")
  doAssert "kernel s.matmul_q4_subgroup_micro" in fir
  doAssert " q4r model.layers.0.q_proj.weight tensor [4," in fir

  fir = liftQ4("vulkan", k = 11008, group = 128, nGroups = 86)
  doAssert "kernel s.matmul_q4_subgroup_micro" in fir
  doAssert "matmul_q4_word_parallel" notin fir

  setEnv("FIRLIFT_Q4_GEMV", "subgroup")
  fir = liftQ4("vulkan", k = 11008, group = 128, nGroups = 86)
  doAssert "kernel s.matmul_q4_subgroup_micro" in fir
  var refused = false
  try:
    discard liftQ4("metal")
  except CatchableError:
    refused = true
  doAssert refused, "the subgroup GEMV micro-kernel is Vulkan-only"

  setEnv("FIRLIFT_Q4_GEMV", "word_parallel")
  fir = liftQ4("vulkan", k = 11008, group = 128, nGroups = 86)
  doAssert "kernel s.matmul_q4_word_parallel" in fir
  delEnv("FIRLIFT_Q4_GEMV")

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

  setEnv("FIRLIFT_Q4_GEMM", "tensor_core")
  fir = liftQ4("vulkan", k = 64, group = 32, nGroups = 2, m = 16, n = 64)
  doAssert "kernel s.matmul_q4_tensor_core_micro" in fir
  doAssert "tg=128" in fir
  # Forced on a Metal trace too: the runtime lowers it to simdgroup_matrix on Apple GPUs.
  fir = liftQ4("metal", k = 64, group = 32, nGroups = 2, m = 16, n = 64)
  doAssert "kernel s.matmul_q4_tensor_core_micro" in fir
  refused = false
  try:
    discard liftQ4("cpu", k = 64, group = 32, nGroups = 2, m = 16, n = 64)
  except CatchableError:
    refused = true
  doAssert refused, "the Q4 Tensor Core micro-kernel needs a Vulkan or Metal trace"
  delEnv("FIRLIFT_Q4_GEMM")

  # Q4 prefill GEMM auto: the matrix units of the recorded device (Vulkan cooperative matrix
  # or Apple simdgroup_matrix), the tiled kernel elsewhere.
  fir = liftQ4("vulkan", k = 64, group = 32, nGroups = 2, m = 16, n = 64, device = "coopmatrix")
  doAssert "kernel s.matmul_q4_tensor_core_micro" in fir
  fir = liftQ4("metal", k = 64, group = 32, nGroups = 2, m = 16, n = 64, device = "simdmatrix")
  doAssert "kernel s.matmul_q4_tensor_core_micro" in fir, "Apple GPU trace picks the simdgroup_matrix Tensor Core kernel"
  doAssert "tg=128" in fir
  fir = liftQ4("metal", k = 64, group = 32, nGroups = 2, m = 16, n = 64, device = "plain")
  doAssert "tensor_core" notin fir, "Intel/AMD Metal trace keeps the generic kernels"
  fir = liftQ4("metal", k = 64, group = 32, nGroups = 2, m = 16, n = 64, device = "coopmatrix")
  doAssert "tensor_core" notin fir, "coopmatrix is a Vulkan device feature"
  fir = liftQ4("vulkan", k = 64, group = 32, nGroups = 2, m = 16, n = 64, device = "simdmatrix")
  doAssert "tensor_core" notin fir, "simdmatrix is a Metal device feature"
  setEnv("FIRLIFT_GENERIC", "1")
  fir = liftQ4("metal", k = 64, group = 32, nGroups = 2, m = 16, n = 64, device = "simdmatrix")
  doAssert "tensor_core" notin fir, "FIRLIFT_GENERIC keeps the generic kernels"
  delEnv("FIRLIFT_GENERIC")

  # FP4-KM prefill GEMM: auto follows the recorded device.
  delEnv("FIRLIFT_FP4_GEMM")
  fir = liftFp4("vulkan", "coopmatrix")
  doAssert "kernel s.fp4_gemm_tensor_core_micro" in fir
  doAssert "rows=16:128" in fir, "the Tensor Core kernel declares its row tiles for partial phase runs"
  doAssert "slots=mem:out0,u8:in1,lut:in1,input:in0" in fir
  doAssert "script_abi 2" in fir and "require feature resident_q4" in fir
  doAssert " fp4km backbone.layers.0.mixer.in_proj.weight tensor [64,128]" in fir
  fir = liftFp4("vulkan", "plain")
  doAssert "kernel s.fp4_gemm_generic" in fir and "tensor_core" notin fir
  fir = liftFp4("vulkan", "coopmatrix", k = 96)
  doAssert "kernel s.fp4_gemm_generic" in fir, "K not divisible by 64 has no Tensor Core kernel"
  fir = liftFp4("metal", "plain")
  doAssert "kernel s.fp4_gemm_generic" in fir
  fir = liftFp4("metal", "simdmatrix")
  doAssert "kernel s.fp4_gemm_tensor_core_micro" in fir, "Apple GPU trace picks the simdgroup_matrix Tensor Core kernel"
  doAssert "rows=16:128" in fir and "slots=mem:out0,u8:in1,lut:in1,input:in0" in fir
  fir = liftFp4("metal", "simdmatrix", k = 96)
  doAssert "kernel s.fp4_gemm_generic" in fir, "K not divisible by 64 has no Tensor Core kernel on Metal either"
  fir = liftFp4("metal", "coopmatrix")
  doAssert "kernel s.fp4_gemm_generic" in fir, "coopmatrix is a Vulkan device feature"
  setEnv("FIRLIFT_FP4_GEMM", "coop")
  fir = liftFp4("vulkan", "plain")
  doAssert "kernel s.fp4_gemm_tensor_core_micro" in fir
  fir = liftFp4("metal", "plain")
  doAssert "kernel s.fp4_gemm_tensor_core_micro" in fir, "forced on a Metal trace (the graph then needs an Apple GPU)"
  refused = false
  try:
    discard liftFp4("cpu", "plain")
  except CatchableError:
    refused = true
  doAssert refused, "the FP4 Tensor Core micro-kernel needs a Vulkan or Metal trace"
  setEnv("FIRLIFT_FP4_GEMM", "generic")
  fir = liftFp4("vulkan", "coopmatrix")
  doAssert "kernel s.fp4_gemm_generic" in fir and "tensor_core" notin fir
finally:
  restoreEnv("FIRLIFT_Q4_GEMV", saved[0])
  restoreEnv("FIRLIFT_Q4_LANES", saved[1])
  restoreEnv("FIRLIFT_TG", saved[2])
  restoreEnv("FIRLIFT_SIMD", saved[3])
  restoreEnv("FIRLIFT_GENERIC", saved[4])
  restoreEnv("FIRLIFT_Q4_WORD_MAX_K", saved[5])
  restoreEnv("FIRLIFT_Q4_GEMM", saved[6])
  restoreEnv("FIRLIFT_FP4_GEMM", saved[7])

echo "PASS Q4 GEMV (subgroup micro, word, simd), Q4 and FP4-KM Tensor Core GEMM selection (Vulkan coopmatrix, Metal simdmatrix)"
