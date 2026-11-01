## sharded_hf_loader.nim
## Complements pytorch_bin_loader.nim: pytorch_bin_loader.loadAllHFTensors()
## takes only ONE file path (a SINGLE .bin or .safetensors). Mamba-Codestral-7B-v0.1
## from HF comes as SHARDED safetensors:
##   model.safetensors.index.json  (map: tensor_name -> shard file name)
##   model-00001-of-00003.safetensors, model-00002-of-00003.safetensors, ...
## (or a single shard model.safetensors without index.json - e.g. a
## repo that repackaged it into one file).
##
## This file does NOT modify the original pytorch_bin_loader.nim - it only WRAPS it: calls
## loadAllHFTensors() for EACH shard in turn and merges them, with the same
## "read into RAM only, no intermediate files" rule as the rest of the project.

import std/[os, json, tables, strformat]
import rawbin
import pytorch_bin_loader

proc loadAllHFTensorsSharded*(hfDir: string): seq[RawBinTensor] =
  ## hfDir = directory containing the downloaded model.safetensors(.index.json)
  ## (e.g. "/Volumes/Data 1/models/Mamba-Codestral-7B-v0.1").
  let indexPath = hfDir / "model.safetensors.index.json"
  var shardFiles: seq[string] = @[]

  if fileExists(indexPath):
    let j = parseJson(readFile(indexPath))
    var seen = initTable[string, bool]()
    for _, v in j["weight_map"].pairs:
      let shard = v.getStr()
      if shard notin seen:
        seen[shard] = true
        shardFiles.add(shard)
    echo &"[sharded_hf_loader] Found index.json: {shardFiles.len} shards"
  else:
    # no index -> assume a single model.safetensors file
    let single = hfDir / "model.safetensors"
    if not fileExists(single):
      raise newException(IOError,
        &"Neither model.safetensors.index.json nor model.safetensors found in {hfDir}")
    shardFiles.add("model.safetensors")
    echo "[sharded_hf_loader] No index.json - using a single shard"

  result = @[]
  for i, shard in shardFiles:
    let full = hfDir / shard
    echo &"[sharded_hf_loader] Doc shard {i+1}/{shardFiles.len}: {shard}"
    let tensors = loadAllHFTensors(full)   # from pytorch_bin_loader.nim, unchanged
    result.add(tensors)
  echo &"[sharded_hf_loader] Loaded {result.len} tensors in total from {shardFiles.len} shards (RAM only)"

when isMainModule:
  # usage: ./sharded_hf_loader_bin "/Volumes/Data 1/models/Mamba-Codestral-7B-v0.1"
  if paramCount() < 1:
    echo "usage: sharded_hf_loader <thu_muc_hf_da_tai_ve>"
    quit(1)
  let tensors = loadAllHFTensorsSharded(paramStr(1))
  var total = 0
  for t in tensors: total += t.data.len
  echo &"[sharded_hf_loader] {tensors.len} tensor, tong {total} gia tri float32 (~{total*4 div (1024*1024)} MB fp32)"