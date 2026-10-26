## sharded_hf_loader.nim
## Bo sung cho pytorch_bin_loader.nim: pytorch_bin_loader.loadAllHFTensors()
## chi nhan 1 FILE path (.bin hoac .safetensors DON). Mamba-Codestral-7B-v0.1
## tu HF ve o dang safetensors CHIA SHARD:
##   model.safetensors.index.json  (map: tensor_name -> ten file shard)
##   model-00001-of-00003.safetensors, model-00002-of-00003.safetensors, ...
## (hoac 1 shard duy nhat model.safetensors, khong co index.json - vi du
## repo nao dong goi lai thanh 1 file).
##
## File nay KHONG sua pytorch_bin_loader.nim goc - chi WRAP no: goi
## loadAllHFTensors() lan luot cho TUNG shard roi gop lai, dung dung co che
## "chi doc vao RAM, khong ghi file trung gian" nhu quy uoc cua project.

import std/[os, json, tables, strformat]
import rawbin
import pytorch_bin_loader

proc loadAllHFTensorsSharded*(hfDir: string): seq[RawBinTensor] =
  ## hfDir = thu muc chua model.safetensors(.index.json) da tai ve
  ## (vd "/Volumes/Data 1/models/Mamba-Codestral-7B-v0.1").
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
    echo &"[sharded_hf_loader] Tim thay index.json: {shardFiles.len} shard"
  else:
    # khong co index -> gia dinh 1 file duy nhat model.safetensors
    let single = hfDir / "model.safetensors"
    if not fileExists(single):
      raise newException(IOError,
        &"Khong thay model.safetensors.index.json lan model.safetensors trong {hfDir}")
    shardFiles.add("model.safetensors")
    echo "[sharded_hf_loader] Khong co index.json - dung 1 shard duy nhat"

  result = @[]
  for i, shard in shardFiles:
    let full = hfDir / shard
    echo &"[sharded_hf_loader] Doc shard {i+1}/{shardFiles.len}: {shard}"
    let tensors = loadAllHFTensors(full)   # tu pytorch_bin_loader.nim, khong doi
    result.add(tensors)
  echo &"[sharded_hf_loader] Da nap TONG {result.len} tensor tu {shardFiles.len} shard (chi trong RAM)"

when isMainModule:
  # cach dung: ./sharded_hf_loader_bin "/Volumes/Data 1/models/Mamba-Codestral-7B-v0.1"
  if paramCount() < 1:
    echo "usage: sharded_hf_loader <thu_muc_hf_da_tai_ve>"
    quit(1)
  let tensors = loadAllHFTensorsSharded(paramStr(1))
  var total = 0
  for t in tensors: total += t.data.len
  echo &"[sharded_hf_loader] {tensors.len} tensor, tong {total} gia tri float32 (~{total*4 div (1024*1024)} MB fp32)"