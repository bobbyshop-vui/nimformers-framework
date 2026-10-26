## quantize_bin_qlora.nim
## Convert OLD raw FP32 NIMBIN1 .bin -> SAME PATH, but as NIMQLR1 QLoRA .bin.
##
## Default policy:
##   * block*.weight 2D tensors -> asymmetric INT4, per-group (group=32)
##   * embedding / lm_head / norms / biases / small state -> FP32
##   * when source is already NIMQLR1, do nothing and exit cleanly
##
## The conversion is streaming: only ONE source tensor is held in RAM at a time.
## The target path is written to <path>.qlora.tmp and atomically renamed over
## the original after all tensors have been written successfully.

import std/[os, strformat, tables, strutils]
import rawbin
import qlorabin
import quant

const DefaultGroupSize = 32

proc numel(shape: seq[int]): int =
  result = 1
  for d in shape: result *= d

proc shouldQuantize(name: string, shape: seq[int]): bool =
  if shape.len != 2: return false
  if not name.startsWith("block"): return false
  if not name.endsWith(".weight"): return false
  numel(shape) >= 4096

proc convertInPlace(path: string, groupSize: int = DefaultGroupSize) =
  if isQLoraBin(path):
    echo &"[qlora] {path} da la NIMQLR1; khong can convert."
    return

  var reader = openRawBinReader(path)
  var meta = reader.meta
  if meta.getOrDefault("format") == "nimformers-lora-v1":
    raise newException(ValueError,
      "File nay la LoRA-only checkpoint cu, base weight nam ngoai .bin. " &
      "Hay convert checkpoint base FP32 NIMBIN1 truoc, hoac dung --merge-base.")

  meta["format"] = "nimformers-qlora-v1"
  meta["quantization"] = "int4-asymmetric-per-group"
  meta["bits"] = "4"
  meta["groupSize"] = $groupSize
  meta["qlora"] = "true"
  meta["sourceFormat"] = "NIMBIN1"

  let tmp = path & ".qlora.tmp"
  if fileExists(tmp): removeFile(tmp)
  var writer = beginQLoraBin(tmp, meta)
  var qCount = 0
  var rawCount = 0
  var qElems: int64 = 0
  var rawElems: int64 = 0
  try:
    while reader.remaining > 0:
      let t = readRawTensor(reader)
      if shouldQuantize(t.name, t.shape):
        let qt = quantizeInt4AsymmetricPerGroup(t.data, t.shape, groupSize)
        writer.writeQLoraQ4Tensor(t.name, qt)
        inc qCount
        qElems += int64(t.data.len)
      else:
        writer.writeQLoraRawTensor(t.name, t.shape, t.data)
        inc rawCount
        rawElems += int64(t.data.len)
    finishQLoraBin(writer)
    reader.closeRawBinReader()
  except:
    reader.closeRawBinReader()
    if writer.stream != nil:
      writer.stream.close()
      writer.stream = nil
    if fileExists(tmp): removeFile(tmp)
    raise

  # Basic size sanity before replacing the source.
  let newSize = getFileSize(tmp)
  if newSize <= 0:
    removeFile(tmp)
    raise newException(IOError, "QLoRA output rong")

  let backup = path & ".fp32.bak"
  if fileExists(backup): removeFile(backup)
  moveFile(path, backup)
  try:
    moveFile(tmp, path)
  except:
    moveFile(backup, path)
    raise
  removeFile(backup)

  echo &"[qlora] DONE: {path}"
  echo &"[qlora] Q4 tensors={qCount}, params quantized={qElems}, raw tensors={rawCount}, raw params={rawElems}"
  echo &"[qlora] size={newSize div (1024*1024)} MiB (group={groupSize}, 4-bit weights)"
  echo "[qlora] Base weights are frozen; fine-tune A/B LoRA only."

when isMainModule:
  if paramCount() < 1:
    echo "usage: quantize_bin_qlora <model.bin> [groupSize]"
    quit(1)
  let groupSize = if paramCount() >= 2: parseInt(paramStr(2)) else: DefaultGroupSize
  doAssert groupSize > 0 and groupSize mod 2 == 0
  convertInPlace(paramStr(1), groupSize)
