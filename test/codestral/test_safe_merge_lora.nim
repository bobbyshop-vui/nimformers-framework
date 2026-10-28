## Tests for guarded in-memory merge; no GPU required for actual test operations.
## Run on the Mac with: nim c -r -d:release --mm:arc --hints:off test_safe_merge_lora.nim
import std/[tables, math]
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
