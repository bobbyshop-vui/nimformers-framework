## Loader-only memory probe: does not initialize GPU or dequantize weights.
## nim c -d:release --mm:arc --path:. --path:test/codestral -r test/codestral/bench_mmap_load.nim
import std/[os, tables, monotimes, times, strutils]
import quant, quantize_lora_bin_4bit, rawbin

proc main() =
  let basePath = if paramCount() >= 1: paramStr(1) else: "test/codestral/models/codestral_mamba_only.nimq"
  let adapterPath = if paramCount() >= 2: paramStr(2) else: "test/codestral/models/jamba_parallel_lora.bin"
  let start = getMonoTime()
  let base = loadQuantStateDict(basePath, mapped = true)
  var mappedBytes: int64
  for pair in base.sd:
    doAssert pair[1].data.len == 0
    mappedBytes += int64(pair[1].mappedData.len)
  let adapter = loadMappedQLora4Bin(adapterPath)
  var mappedAdapters, mappedSidecars: int
  var ownedRawBytes: int64
  for t in adapter.fp32Tensors:
    if t.mappedData.owner != nil:
      if t.name.startsWith("adapter::"): inc mappedAdapters
      else: inc mappedSidecars
      mappedBytes += int64(t.mappedData.len) * 4
    ownedRawBytes += int64(t.data.len) * 4
  for _, qt in adapter.quantWeights:
    doAssert qt.data.len == 0
    mappedBytes += int64(qt.mappedData.len)
  echo "base tensors=", base.sd.len, " mapped adapters=", mappedAdapters,
    " raw mapped sidecars=", mappedSidecars, " quant sidecars=", adapter.quantWeights.len
  echo "mapped payload MiB=", mappedBytes div (1024 * 1024),
    " owned raw MiB=", ownedRawBytes div (1024 * 1024),
    " elapsed ms=", (getMonoTime() - start).inMilliseconds
main()
