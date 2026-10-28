## merge_lora_to_nimq.nim
## Gop adapter LoRA + Attention (da train, luu trong <ckpt>.lora.json) VINH
## VIEN vao base, roi QUANTIZE ra .nimq (fp4) DUY NHAT.
##
## CAM .bin: KHONG doc .bin o input, KHONG ghi .bin o output, o BAT KY buoc
## nao. Base (phan Mamba) TAI TAO 100% trong RAM tu chinh .nimq goc + pattern
## (dung y het co che dequant-on-the-fly cua finetune_lora.nim/infer_lora.nim),
## Attention (block moi, train full) nap tu ".lora.json" (khong co trong
## .nimq goc), roi merge adapter LoRA vinh vien, cuoi cung quantize CA
## Attention lan Mamba thanh 1 file .nimq output MOI. Input DUY NHAT can co:
##   - <ckpt>.lora.json   (adapter + Attention da train, sinh ra tu finetune_lora.nim)
##   - .nimq goc ma <ckpt>.lora.json tro toi (field "nimqPath" trong chinh no)
##
## CACH BUILD:
##   nim c -d:release --hints:off -o:merge_lora_to_nimq merge_lora_to_nimq.nim
## CACH DUNG:
##   ./merge_lora_to_nimq jamba_lora.bin jamba_lora_merged.nimq [--graft-quant int8|int4|fp4|fp8]
## (tham so dau la TIEN TO ten checkpoint - chi de tim ra "<tien_to>.lora.json",
##  KHONG phai duong dan toi 1 file .bin thuc su - file .bin do KHONG TON TAI).

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
  # Phai KHOP CHINH XAC voi hang so cung ten trong finetune_lora.nim/infer_lora.nim.
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
  else: raise newException(ValueError, &"--graft-quant khong hop le: {s}")

const PrecisionSensitiveHints = ["embed", "norm", "gamma", "beta", ".bias"]

proc isPrecisionSensitive(name: string): bool =
  let low = name.toLowerAscii
  for h in PrecisionSensitiveHints:
    if h in low: return true
  return false

when isMainModule:
  if paramCount() < 2:
    echo "usage (LEGACY/UNSAFE): merge_lora_to_nimq <tien_to_checkpoint (co <tien_to>.lora.json canh ben)> <out.nimq> [--graft-quant int8|int4|fp4|fp8] [--unsafe-legacy-requant]"
    echo "       CAM .bin: tham so dau CHI dung de tim <tien_to>.lora.json, KHONG doc file .bin nao."
    quit(1)

  let ckptPrefix = paramStr(1)
  let outPath = paramStr(2)
  # Tool nay CHI doc JSON legacy; khong the merge checkpoint NIMBIN4
  # cua `finetune_lora.nim` hien tai. Mac dinh fail-closed de tranh tao .nimq
  # bi requant Mamba toan bo -> token sai va mat LoRA delta.
  var unsafeLegacy = false
  for k in 3 .. paramCount():
    if paramStr(k) == "--unsafe-legacy-requant": unsafeLegacy = true
  if not unsafeLegacy:
    stderr.writeLine "[merge] SAFE STOP: tool nay chi ho tro .lora.json legacy, " &
      "khong doc checkpoint NIMBIN4 .bin hien tai; no con requant Mamba ve FP4."
    stderr.writeLine "[merge] KHONG tao merged .nimq vi co nguy co mat delta LoRA " &
      "va lam model sinh token sai. BYBY_MERGE_LORA=1 trong infer_lora.nim " &
      "co quality guard va fallback ve fused adapter."
    stderr.writeLine "[merge] Neu buoc phai chay tool JSON cu, " &
      "dung --unsafe-legacy-requant (KHONG dung voi .bin hien tai)."
    quit(1)
  if ckptPrefix.toLowerAscii.endsWith(".bin") and fileExists(ckptPrefix):
    stderr.writeLine "[merge] STOP: input la checkpoint .bin, " &
      "nhung tool nay KHONG biet doc .bin. Khong duoc merge JSON thay the."
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
    echo &"[merge] LOI: khong tim thay {loraJsonPath}"
    quit(1)
  let j = parseFile(loraJsonPath)
  let nimqPath = j["nimqPath"].getStr
  let pattern = j["pattern"].getStr

  echo &"[merge] Tai tao base tu {nimqPath} (dequant on-the-fly qua LUT, pattern='{pattern}') - KHONG doc .bin nao ..."
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
  echo &"[merge] Da nap {nAttnW} tensor Linear++ Attention (block moi, train full) tu {loraJsonPath}"

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
  lo.quantBase = quantResident   # inProj/outProj (quant-resident, ~29GB neu dequant het)
                                  # - mergeIn se tu dequant qua LUT dung luc can, ghi thang
                                  # vao t[].data (permanent, vi tool nay KHONG goi restoreBase).

  let ctx = newBackend()
  echo &"[merge] Gop vinh vien {lo.adapters.len} adapter LoRA vao base (W_eff = W_base + (alpha/r)*B@A) ..."
  mergeIn(m, lo, ctx)   # KHONG goi restoreBase - giu merged permanent

  # FIX: outProj.weight (lm_head) nam trong lo.quantBase nhung KHONG nam trong
  # lo.adapters (khong LoRA-hoa) nen mergeIn() KHONG dequant no. Phai dequant
  # tai day, neu khong thi m.outProj.weight.data van la @[] va quantize ra rong.
  if m.outProj.weight.data.len == 0 and "outProj.weight" in lo.quantBase:
    echo "[merge] Dequantize lm_head (outProj.weight) tu quantBase ..."
    m.outProj.weight.data = quant.dequantizeTensor(lo.quantBase["outProj.weight"])

  echo "[merge] Quantize toan bo model (Mamba + Attention, da merge LoRA) -> .nimq ..."
  var sd: seq[(string, QuantTensor)] = @[]
  var nFp4, nFp32, nGraft = 0
  for (name, t) in paramList(m):
    var qt: QuantTensor
    let isAttnNew = ".attn." in name   # tensor Linear++ Attention - KHONG co trong .nimq HF goc
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
  echo &"[merge] Da ghi {outPath} ({sd.len} tensor, fp4(Mamba)={nFp4} fp32={nFp32} " &
       &"{graftQuantStr}(Attention-moi)={nGraft}). KHONG co file .bin nao duoc tao ra."
  echo "[merge] File nay dung duoc thang cho nim_inference.nim, hoac lam CFG_NIMQ_PATH " &
       "cho lan finetune_lora.nim tiep theo - da co san Linear++ Attention, da gop LoRA."
