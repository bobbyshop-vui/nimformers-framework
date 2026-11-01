## merge_lora_to_nimq.nim
## Merges the trained LoRA adapters + Attention (stored in <ckpt>.lora.json)
## PERMANENTLY into the base, then QUANTIZES a SINGLE .nimq (fp4).
##
## NO .bin: does NOT read a .bin as input or write a .bin as output, at ANY
## step. The base (Mamba part) is REBUILT 100% in RAM from the original .nimq + the pattern
## (the same dequant-on-the-fly mechanism as finetune_lora.nim/infer_lora.nim),
## the Attention (new blocks, fully trained) is loaded from ".lora.json" (not in the
## original .nimq), then the LoRA adapters are merged permanently, and finally BOTH
## Attention and Mamba are quantized into one NEW output .nimq. The ONLY inputs needed:
##   - <ckpt>.lora.json   (adapters + trained Attention, produced by finetune_lora.nim)
##   - the original .nimq that <ckpt>.lora.json points to (its "nimqPath" field)
##
## BUILD:
##   nim c -d:release --hints:off -o:merge_lora_to_nimq merge_lora_to_nimq.nim
## USAGE:
##   ./merge_lora_to_nimq jamba_lora.bin jamba_lora_merged.nimq [--graft-quant int8|int4|fp4|fp8]
## (the first argument is the checkpoint name PREFIX - only used to find "<prefix>.lora.json",
##  NOT the path of a real .bin file - that .bin file does NOT EXIST).

import std/[os, strutils, strformat, json, tables]
import quant
import customfloat
import nimformer                   # Tensor, newTensor
import jamba_hybrid
import load_codestral_into_jamba   # loadCodestralQuantResidentFromNimq
import lora
import backend

let FP4_E2M1 = newCustomFloat(2, 1, "fp4_e2m1")

const
  # Must match EXACTLY the constants of the same name in finetune_lora.nim/infer_lora.nim.
  CFG_CODESTRAL_VOCAB = 32768
  CFG_CODESTRAL_HIDDEN = 4096
  CFG_CODESTRAL_NHEADS = 128
  CFG_CODESTRAL_DSTATE = 128
  CFG_CODESTRAL_NGROUPS = 8
  CFG_CODESTRAL_CONVKERNEL = 4
  CFG_CODESTRAL_EXPAND = 2

proc parseGraftQuant(s: string): QuantKind =
  case s.toLowerAscii
  of "int8": qkInt8
  of "int4": qkInt4
  of "fp4":  qkCustom
  of "fp8":  qkFp8E4M3
  else: raise newException(ValueError, &"invalid --graft-quant: {s}")

const PrecisionSensitiveHints = ["embed", "norm", "gamma", "beta", ".bias"]

proc isPrecisionSensitive(name: string): bool =
  let low = name.toLowerAscii
  for h in PrecisionSensitiveHints:
    if h in low: return true
  return false

when isMainModule:
  if paramCount() < 2:
    echo "usage (LEGACY/UNSAFE): merge_lora_to_nimq <tien_to_checkpoint (co <tien_to>.lora.json canh ben)> <out.nimq> [--graft-quant int8|int4|fp4|fp8] [--unsafe-legacy-requant]"
    echo "       NO .bin: the first argument is ONLY used to find <prefix>.lora.json, NO .bin file is read."
    quit(1)

  let ckptPrefix = paramStr(1)
  let outPath = paramStr(2)
  # This tool reads ONLY legacy JSON; it cannot merge the NIMBIN4 checkpoints
  # of the current `finetune_lora.nim`. Fail-closed by default to avoid producing a .nimq
  # with the whole Mamba requantized -> wrong tokens and a lost LoRA delta.
  var unsafeLegacy = false
  for k in 3 .. paramCount():
    if paramStr(k) == "--unsafe-legacy-requant": unsafeLegacy = true
  if not unsafeLegacy:
    stderr.writeLine "[merge] SAFE STOP: this tool only supports legacy .lora.json, " &
      "it does not read current NIMBIN4 .bin checkpoints, and it would requantize Mamba to FP4."
    stderr.writeLine "[merge] NOT creating a merged .nimq because the LoRA delta could be lost " &
      "and the model would generate wrong tokens. BYBY_MERGE_LORA=1 in infer_lora.nim " &
      "has a quality guard and falls back to the fused adapter."
    stderr.writeLine "[merge] If you must run the old JSON tool, " &
      "use --unsafe-legacy-requant (NOT with current .bin files)."
    quit(1)
  if ckptPrefix.toLowerAscii.endsWith(".bin") and fileExists(ckptPrefix):
    stderr.writeLine "[merge] STOP: the input is a .bin checkpoint, " &
      "but this tool CANNOT read .bin. Merging the JSON instead is not allowed."
    quit(1)
  var graftQuantStr = "int8"
  var i = 3
  while i <= paramCount():
    if paramStr(i) == "--graft-quant" and i + 1 <= paramCount():
      graftQuantStr = paramStr(i + 1)
      i += 2
    else:
      i += 1
  let graftKind = parseGraftQuant(graftQuantStr)

  let loraJsonPath = ckptPrefix & ".lora.json"
  if not fileExists(loraJsonPath):
    echo &"[merge] ERROR: {loraJsonPath} not found"
    quit(1)
  let j = parseFile(loraJsonPath)
  let nimqPath = j["nimqPath"].getStr
  let pattern = j["pattern"].getStr

  echo &"[merge] Rebuilding the base from {nimqPath} (dequant on the fly through the LUT, pattern='{pattern}') - NO .bin is read ..."
  var m = newNimformerModelJamba(CFG_CODESTRAL_VOCAB, CFG_CODESTRAL_HIDDEN, CFG_CODESTRAL_NHEADS,
                                  ffMult = 4, pattern = pattern, useSession = true,
                                  mambaRealDState = CFG_CODESTRAL_DSTATE,
                                  mambaRealNGroups = CFG_CODESTRAL_NGROUPS,
                                  mambaRealConvKernel = CFG_CODESTRAL_CONVKERNEL,
                                  mambaRealExpand = CFG_CODESTRAL_EXPAND, quantResident = true)
  let quantResident = loadCodestralQuantResidentFromNimq(m, nimqPath)

  var byName = initTable[string, ptr Tensor]()
  for (name, t) in paramList(m): byName[name] = t

  var nAttnW = 0
  for name, v in j["newBlockWeights"].pairs:
    var idx = 0
    for x in v.items: byName[name][].data[idx] = x.getFloat.float32; inc idx
    inc nAttnW
  echo &"[merge] Loaded {nAttnW} Linear++ Attention tensors (new blocks, fully trained) from {loraJsonPath}"

  var lo: JambaLora
  lo.adapters = initTable[string, LoraAdapter]()
  lo.backup = initTable[string, seq[float32]]()
  for name, v in j["adapters"].pairs:
    var ad: LoraAdapter
    ad.inF = v["inF"].getInt
    ad.outF = v["outF"].getInt
    ad.r = v["r"].getInt
    ad.alpha = v.getOrDefault("alpha").getFloat(8.0).float32
    ad.scale = ad.alpha / float32(ad.r)
    ad.A = newTensor(@[ad.r, ad.inF])
    ad.B = newTensor(@[ad.outF, ad.r])
    var idx = 0
    for x in v["A"].items: ad.A.data[idx] = x.getFloat.float32; inc idx
    idx = 0
    for x in v["B"].items: ad.B.data[idx] = x.getFloat.float32; inc idx
    lo.adapters[name] = ad
  lo.quantBase = quantResident   # inProj/outProj (quant-resident, ~29GB if fully dequantized)
                                  # - mergeIn dequantizes through the LUT when needed and writes directly
                                  # into t[].data (permanent, because this tool does NOT call restoreBase).

  let ctx = newBackend()
  echo &"[merge] Permanently merging {lo.adapters.len} LoRA adapters into the base (W_eff = W_base + (alpha/r)*B@A) ..."
  mergeIn(m, lo, ctx)   # restoreBase is NOT called - keeps the merge permanent

  # FIX: outProj.weight (lm_head) is in lo.quantBase but NOT in
  # lo.adapters (not LoRA-ised), so mergeIn() does NOT dequantize it. It must be dequantized
  # here, otherwise m.outProj.weight.data stays @[] and quantizes to nothing.
  if m.outProj.weight.data.len == 0 and "outProj.weight" in lo.quantBase:
    echo "[merge] Dequantize lm_head (outProj.weight) tu quantBase ..."
    m.outProj.weight.data = quant.dequantizeTensor(lo.quantBase["outProj.weight"])

  echo "[merge] Quantizing the whole model (Mamba + Attention, LoRA merged) -> .nimq ..."
  var sd: seq[(string, QuantTensor)] = @[]
  var nFp4, nFp32, nGraft = 0
  for (name, t) in paramList(m):
    var qt: QuantTensor
    let isAttnNew = ".attn." in name   # Linear++ Attention tensor - NOT in the original HF .nimq
    if isPrecisionSensitive(name):
      qt = quantizeTensor(t[].data, t[].shape, qkFp32Raw)
      inc nFp32
    elif isAttnNew:
      qt = quantizeTensor(t[].data, t[].shape, graftKind, customCf = FP4_E2M1)
      inc nGraft
    else:
      qt = quantizeTensor(t[].data, t[].shape, qkCustom, customCf = FP4_E2M1)
      inc nFp4
    sd.add((name, qt))

  let arch = [CFG_CODESTRAL_VOCAB, CFG_CODESTRAL_HIDDEN, m.blocks.len, CFG_CODESTRAL_NHEADS, CFG_CODESTRAL_EXPAND]
  saveQuantStateDict(outPath, arch, sd)
  echo &"[merge] Wrote {outPath} ({sd.len} tensors, fp4(Mamba)={nFp4} fp32={nFp32} " &
       &"{graftQuantStr}(new Attention)={nGraft}). NO .bin file was created."
  echo "[merge] This file can be used directly by nim_inference.nim, or as CFG_NIMQ_PATH " &
       "for the next finetune_lora.nim run - it already has the Linear++ Attention and the merged LoRA."
