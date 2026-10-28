## CPU tests: mapped payloads, ownership, serialization and corrupt files.
import std/[os, tables, math]
import quant, rawbin, quantize_lora_bin_4bit

proc main() =
  let dir = getTempDir() / ("nimformers-mmap-" & $getCurrentProcessId())
  createDir(dir)
  defer: removeDir(dir)
  let arch = [32, 32, 1, 1, 1]
  var data = newSeq[float32](64 * 32)
  for i in 0 ..< data.len: data[i] = sin(float32(i) * 0.07'f32)
  let fp4 = quantizeFp4KM(data, @[64, 32])
  let raw = quantizeRaw(@[1'f32, -2'f32], @[2])
  let path = dir / "base.nimq"
  saveQuantStateDict(path, arch, @[("fp4", fp4), ("raw", raw)])
  var held: QuantTensor
  block:
    let loaded = loadQuantStateDict(path, mapped = true)
    doAssert loaded.arch == arch
    for i, pair in loaded.sd:
      doAssert pair[1].data.len == 0
      doAssert pair[1].mappedData.owner != nil
      let expected = if i == 0: fp4 else: raw
      doAssert @(pair[1].bytes) == expected.data
      doAssert dequantizeTensor(pair[1]) == dequantizeTensor(expected)
    held = loaded.sd[0][1]
  GC_fullCollect()
  doAssert dequantizeTensor(held) == dequantizeTensor(fp4)
  saveQuantStateDict(dir / "roundtrip.nimq", arch, @[("fp4", held)])
  let roundtrip = loadQuantStateDict(dir / "roundtrip.nimq")
  doAssert roundtrip.sd[0][1].data == fp4.data
  held.clearPayload()
  doAssert held.bytes.len == 0 and held.mappedData.owner == nil

  let tensors = @[
    RawBinTensor(name: "adapter::proj::A", shape: @[1, 2], data: @[0.25'f32, -0.5'f32]),
    RawBinTensor(name: "adapter::proj::B", shape: @[2, 1], data: @[1'f32, 2'f32]),
    RawBinTensor(name: "newBlock::norm", shape: @[2], data: @[3'f32, 4'f32]),
    RawBinTensor(name: "newBlock::proj.weight", shape: @[1, 2], data: @[5'f32, 6'f32])]
  var meta = initTable[string, string]()
  meta["format"] = "test"
  saveRawBin(dir / "adapter.bin", meta, tensors)
  let rawLoaded = loadMappedRawBin(dir / "adapter.bin")
  doAssert rawLoaded.meta == meta
  for i, t in rawLoaded.tensors:
    if i != 2:
      doAssert t.data.len == 0 and t.mappedData.owner != nil
      doAssert @(t.mappedData.values) == tensors[i].data
    else: doAssert t.data == tensors[i].data

  saveQLora4Bin(dir / "qlora.bin", meta, tensors, @[("proj", fp4)])
  let qLoaded = loadMappedQLora4Bin(dir / "qlora.bin")
  doAssert qLoaded.quantWeights["proj"].data.len == 0
  doAssert dequantizeTensor(qLoaded.quantWeights["proj"]) == dequantizeTensor(fp4)
  doAssert @(qLoaded.fp32Tensors[0].mappedData.values) == tensors[0].data
  # Mapping is retained even after the reader result is destroyed.
  var adapterView: MappedView[float32]
  block:
    let q = loadMappedQLora4Bin(dir / "qlora.bin")
    adapterView = q.fp32Tensors[1].mappedData
  GC_fullCollect()
  doAssert adapterView.values[1] == 2'f32

  var truncated = readFile(path)
  truncated.setLen(truncated.len - 1)
  writeFile(dir / "truncated.nimq", truncated)
  var rejected = false
  try: discard loadQuantStateDict(dir / "truncated.nimq", mapped = true)
  except IOError: rejected = true
  doAssert rejected
  # A malicious negative count must fail before allocation/pointer arithmetic.
  writeFile(dir / "bad.bin", "NIMBIN1\0\xff\xff\xff\xff")
  rejected = false
  try: discard loadMappedRawBin(dir / "bad.bin")
  except IOError: rejected = true
  doAssert rejected
  echo "PASS: mmap quant + adapters, reader lifetime, roundtrip, truncation, invalid counts"

main()
