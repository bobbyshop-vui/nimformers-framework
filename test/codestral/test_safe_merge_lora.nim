## Tests for guarded in-memory merge; no GPU required for actual test operations.
## Run on the Mac with: nim c -r -d:release --mm:arc --hints:off test_safe_merge_lora.nim
import std/[tables, math, os]
import rawbin, jamba_hybrid, backend
import lora, quant, nimformer

proc makeAdapter(inF, outF: int; bValue: float32): LoraAdapter =
  result.inF = inF
  result.outF = outF
  result.r = 1
  result.alpha = 1'f32
  result.scale = 1'f32
  result.A = newTensor(@[1, inF])
  result.B = newTensor(@[outF, 1])
  for i in 0 ..< inF: result.A.data[i] = 0.5'f32
  for i in 0 ..< outF: result.B.data[i] = bValue

proc makeLora(): JambaLora =
  result.adapters = initTable[string, LoraAdapter]()
  result.quantBase = initTable[string, QuantTensor]()
  result.backup = initTable[string, seq[float32]]()

when isMainModule:
  const key = "block0.mambaReal.inProj.weight"
  let shape = @[32, 32]
  var base = newSeq[float32](32*32)
  for i in 0 ..< base.len: base[i] = float32((i mod 19) - 9) * 0.003'f32

  # FP4-KM must never destroy an adapter when safe guard is used.
  var lo = makeLora()
  lo.quantBase[key] = quantizeFp4KM(base, shape)
  lo.adapters[key] = makeAdapter(32, 32, 0.0001'f32)
  let packedBefore = lo.quantBase[key].data
  doAssert mergeLoraIntoQuantBase(lo, verbose = false) == 0
  doAssert key in lo.adapters
  doAssert key in lo.quantBase
  doAssert lo.quantBase[key].data == packedBefore
  echo "PASS: FP4 safe skip keeps original quant base and adapter"

  # A raw FP32 base can merge without introducing quantization error.
  var hi = makeLora()
  hi.quantBase[key] = quantizeRaw(base, shape)
  hi.adapters[key] = makeAdapter(32, 32, 0.0001'f32)
  doAssert mergeLoraIntoQuantBase(hi, verbose = false) == 1
  doAssert key notin hi.adapters
  let merged = dequantizeTensor(hi.quantBase[key])
  for i in 0 ..< base.len:
    doAssert abs(merged[i] - (base[i] + 0.00005'f32)) < 1.0e-8'f32
  echo "PASS: FP32 merge preserves W + B@A"

  # A mapped adapter must survive attachment without allocating dense A/B.
  let mappedPath = getTempDir() / ("mapped-lora-" & $getCurrentProcessId() & ".bin")
  saveRawBin(mappedPath, initTable[string, string](), @[
    RawBinTensor(name: "adapter::outProj.weight::A", shape: @[1, 32], data: newSeq[float32](32)),
    RawBinTensor(name: "adapter::outProj.weight::B", shape: @[32, 1], data: newSeq[float32](32))])
  let mapped = loadMappedRawBin(mappedPath)
  var ad = makeAdapter(32, 32, 0.0001'f32)
  ad.A.data = @[]; ad.B.data = @[]
  ad.mappedA = mapped.tensors[0].mappedData
  ad.mappedB = mapped.tensors[1].mappedData
  var mappedLo = makeLora()
  mappedLo.adapters["outProj.weight"] = ad
  var model = NimformerModelJamba(outProj: Linear(inF: 32, outF: 32))
  putEnv("BYBY_APPLY_LMHEAD_LORA", "1")
  mergeInUploadStreamingJamba(model, mappedLo, Backend(kind: bkMetal))
  doAssert model.outProj.loraA.len == 0 and model.outProj.loraB.len == 0
  doAssert model.outProj.loraAValues.len == 32 and model.outProj.loraBValues.len == 32
  doAssert model.outProj.mappedLoraA.owner != nil
  doAssert dropZeroLoraAdapters(mappedLo).len == 1
  removeFile(mappedPath)
  echo "PASS: mapped adapter attaches without copies and zero-delta guard reads mapped values"

  # Budget rejection must keep both tensors unchanged.
  var limited = makeLora()
  limited.quantBase[key] = quantizeRaw(base, shape)
  limited.adapters[key] = makeAdapter(32, 32, 0.0001'f32)
  putEnv("BYBY_MERGE_SCRATCH_MIB", "0")
  doAssert mergeLoraIntoQuantBase(limited, verbose = false) == 0
  doAssert key in limited.adapters
  doAssert dequantizeTensor(limited.quantBase[key]) == base
  delEnv("BYBY_MERGE_SCRATCH_MIB")
  echo "PASS: merge scratch budget preserves fused fallback"
