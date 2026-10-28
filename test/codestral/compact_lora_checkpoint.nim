## Convert an existing streaming LoRA checkpoint to a compact quantized weight file.
## The original .lora.json is never overwritten.

import std/[os, strformat, tables, json, strutils]
import finetune_lora
import quant
import customfloat
import nimformer
import jamba_hybrid
import lora

const
  OutSuffix = ".compact.nimq"
  MetaSuffix = ".compact.json"

proc writeFloatArray(f: File, values: seq[float32]) =
  f.write("[")
  for i, value in values:
    if i > 0: f.write(",")
    f.write($value)
  f.write("]")

proc compactCheckpoint(input, output: string) =
  if not fileExists(input & ".lora.json"):
    raise newException(IOError, "Khong tim thay checkpoint: " & input & ".lora.json")

  echo &"[compact] Doc checkpoint streaming: {input}.lora.json"
  var model: NimformerModelJamba
  var lora: JambaLora
  var epoch: int
  (model, lora, epoch) = loadCheckpointLora(input)

  var quantized: seq[(string, QuantTensor)] = @[]
  var blockCount = 0
  for (name, tensor) in paramList(model):
    var blockIndex = -1
    if name.startsWith("block"):
      let dot = name.find('.')
      if dot > 5:
        try: blockIndex = parseInt(name[5 ..< dot])
        except ValueError: discard
    if blockIndex < 0 or blockIndex >= model.pattern.len:
      continue
    let pChar = model.pattern[blockIndex].toUpperAscii
    if pChar notin {'A', 'P'}:
      continue
    if pChar == 'P' and not name.contains(".attn."):
      continue
    if tensor[].data.len == 0: continue
    let kind = if ".weight" in name: qkInt8 else: qkFp32Raw
    quantized.add((name, quantizeTensor(tensor[].data, tensor[].shape, kind)))
    inc blockCount

  let arch = [model.vocab, model.embedDim, model.blocks.len, model.nHeads, 4]
  saveQuantStateDict(output, arch, quantized)

  let metaPath = input & MetaSuffix
  let baseNimqPath = getEnv("BYBY_BASE_NIMQ", CFG_NIMQ_PATH_DEFAULT)
  var f = open(metaPath, fmWrite)
  f.write("{\"epoch\":" & $epoch & ",\"baseNimqPath\":" & escapeJson(baseNimqPath) &
      ",\"compactWeightsPath\":" & escapeJson(output) &
      ",\"pattern\":" & escapeJson(model.pattern) & ",\"adapters\":{")
  var first = true
  for name, adapter in lora.adapters:
    if not first: f.write(",")
    first = false
    f.write(escapeJson(name) & ":{\"A\":")
    writeFloatArray(f, adapter.A.data)
    f.write(",\"B\":")
    writeFloatArray(f, adapter.B.data)
    f.write(",\"inF\":" & $adapter.inF & ",\"outF\":" & $adapter.outF &
            ",\"r\":" & $adapter.r & ",\"alpha\":" & $adapter.alpha & "}")
  f.write("},\"compactWeights\":true}")
  f.flushFile()
  f.close()
  echo &"[compact] Da ghi {output} va {metaPath}; {blockCount} tensor block A da quantize int8."
  echo "[compact] Checkpoint goc van duoc giu nguyen."

when isMainModule:
  let input = if paramCount() >= 1: paramStr(1) else: CFG_CHECKPOINT_PATH_DEFAULT
  let output = if paramCount() >= 2: paramStr(2) else: input & OutSuffix
  compactCheckpoint(input, output)
