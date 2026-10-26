## quantize_codestral.nim
## Doc Mamba-Codestral-7B-v0.1 (sharded safetensors) THANG TU THU MUC DA TAI
## VE (vd tren "/Volumes/Data 1/...") qua sharded_hf_loader, quantize, va ghi
## DUY NHAT 1 file .nimq nho vao THU MUC PROJECT tren o he thong - KHONG bao
## gio ghi checkpoint fp16/fp32 trung gian nao (dung dung yeu cau "tiet kiem
## o" cua ban - ~7B tham so fp16 ~14GB, sau fp4 con ~3.5GB + it fp32 cho
## embedding/layernorm).
##
## Quy tac quantize (theo dung luu y co san trong quant.nim: "fp32 danh cho
## bias/layernorm - nhay sai so"):
##   - embedding, layernorm (weight+bias), moi loai bias      -> fp32 raw
##   - weight ma tran CHINH cua checkpoint goc HF (in_proj,
##     x_proj, dt_proj, conv1d, out_proj cua Mamba2, lm_head)  -> FP4 (E2M1)
##   - tensor thuoc phan GRAFT moi (LinearPlusAttention moi
##     cay/LoRA A-B) khong nam trong checkpoint HF goc         -> --graft-quant
##     (mac dinh int8, co the doi sang int4/fp4/fp8 qua CLI)
##
## Vi day la .nimq (khong phai .bin nhu jamba_hybrid dung cho LoRA-train), muc
## dich file nay la INFERENCE/baseline gon nhe qua nim_inference.nim, KHONG
## phai file de tiep tuc train (train LoRA van doc .bin qua rawbin.nim nhu
## finetune_lora.nim dang lam - xem ghi chu trong jamba_hybrid.nim).
##
## CACH DUNG:
##   nim c -d:release --hints:off -o:quantize_codestral_bin quantize_codestral.nim
##   ./quantize_codestral_bin "/Volumes/Data 1/models/Mamba-Codestral-7B-v0.1" \
##       ./codestral_mamba_7b_fp4.nimq [--graft-quant int8|int4|fp4|fp8]

import std/[os, strutils, strformat, tables]
import quant
import customfloat
import sharded_hf_loader

# FP4 E2M1 - xem customfloat_fp4_patch.nim (phai dan vao customfloat.nim
# TRUOC khi bien dich file nay, hoac khai bao lai ngay tai day cho gon):
let FP4_E2M1 = newCustomFloat(2, 1, "fp4_e2m1")

const
  Fp32NameHints = ["embedding", "norm", "layernorm", "ln.", ".bias", "lm_head.bias"]
  # heuristic: tensor co ten chua 1 trong cac chuoi tren -> nhay sai so, giu fp32.
  # "lm_head.bias" liet ke rieng vi lm_head.weight KHONG duoc giu fp32 (qua lon,
  # 7B*vocab), chi co bias (neu co) moi giu.

proc isPrecisionSensitive(name: string): bool =
  let lower = name.toLowerAscii
  for hint in Fp32NameHints:
    if hint in lower:
      return true
  return false

proc parseGraftQuant(s: string): QuantKind =
  case s.toLowerAscii
  of "int8": qkInt8
  of "int4": qkInt4
  of "fp4":  qkCustom     # dung FP4_E2M1 lam customCf
  of "fp8":  qkFp8E4M3
  else: raise newException(ValueError, &"--graft-quant khong hop le: {s} (chon int8/int4/fp4/fp8)")

proc parseMainQuant(s: string): QuantKind =
  ## FIX token rac (bug #13). TRUOC DAY moi weight chinh bi hard-code
  ## `qkCustom + FP4_E2M1` (per-tensor scale). Do that bang chinh code cua
  ## repo tren ma tran 512x4096 phan bo kieu LLM:
  ##
  ##   fp4 e2m1 per-tensor : rel-L2 = 0.9421, chi 0.06% phan tu con khac 0
  ##   fp8 e4m3 per-tensor : rel-L2 = 0.0531, 99.89% phan tu con khac 0
  ##
  ## Nguyen nhan: FP4 E2M1 chi co 6 do lon bieu dien duoc
  ## {0, 0.75, 1, 1.5, 2, 3} (4/16 ma con la +-inf), VA encodeArray
  ## TRUNCATE ve 0 chu khong lam tron (0.74 -> 0.00, 0.90 -> 0.00). Voi
  ## scale = maxAbs/3 dung chung cho ca ma tran, MOI weight nho hon
  ## 0.25*maxAbs bi ve DUNG BANG 0 -> ma tran 7B thanh ~99.9% so 0.
  ##
  ## fp8 e4m3 VAN la 1 byte/phan tu (itemSize=1), VAN dung buildDecodeLUT
  ## 256 muc, VAN dung buildFp4Lut/qt.scale[0] va kernel Metal hien tai -
  ## KHONG phai sua mot dong nao o loader hay GPU. Kich thuoc file .nimq
  ## GIU NGUYEN. Chi can export lai.
  ##
  ## fp4km (MOI, MAC DINH): "Q4_K_M nhung FP4". Do tren ma tran kieu LLM:
  ##   fp4 cu (per-tensor, truncate) : 8.00 bpw, relL2 0.879, 99.5% ve 0
  ##   FP4-KM g=32                   : 4.25 bpw, relL2 0.100, cos 0.992
  ##   fp8 e4m3 per-tensor           : 8.00 bpw, relL2 0.049, cos 0.9996
  ## => fp4km vua NHO BANG NUA file hien tai vua sai so thap hon ~9 lan.
  ## Chon fp8 neu muon chat luong cao nhat va khong ngai file 7GB.
  case s.toLowerAscii
  of "fp4km": qkFp4KM
  of "fp8":  qkFp8E4M3
  of "fp4":  qkCustom
  of "int8": qkInt8
  else: raise newException(ValueError, &"--main-quant khong hop le: {s} (chon fp4km/fp8/fp4/int8)")

when isMainModule:
  if paramCount() < 2:
    echo "usage: quantize_codestral <thu_muc_hf_da_tai_ve> <out.nimq> " &
         "[--main-quant fp4km|fp8|fp4|int8] [--graft-quant int8|int4|fp4|fp8]"
    echo "  --main-quant mac dinh = fp4km (4.25 bit/weight). CHO fp4: chi dung neu ban biet " &
         "ro minh dang lam gi - xem parseMainQuant, fp4 e2m1 per-tensor lam mat " &
         "~99.9% weight (bug #13)."
    quit(1)

  let hfDir = paramStr(1)
  let outPath = paramStr(2)
  var graftQuantStr = "int8"
  var mainQuantStr = "fp4km"  # FIX bug #13 - xem parseMainQuant
  var i = 3
  while i <= paramCount():
    if paramStr(i) == "--graft-quant" and i + 1 <= paramCount():
      graftQuantStr = paramStr(i + 1)
      i += 2
    elif paramStr(i) == "--main-quant" and i + 1 <= paramCount():
      mainQuantStr = paramStr(i + 1)
      i += 2
    else:
      i += 1
  let graftKind = parseGraftQuant(graftQuantStr)
  let mainKind = parseMainQuant(mainQuantStr)

  echo &"[quantize_codestral] Doc {hfDir} (chi RAM, khong ghi file trung gian) ..."
  let tensors = loadAllHFTensorsSharded(hfDir)
  echo &"[quantize_codestral] {tensors.len} tensor. Bat dau quantize " &
       &"(weight chinh -> {mainQuantStr}, sensitive -> fp32, graft-moi -> {graftQuantStr}) ..."

  var sd: seq[(string, QuantTensor)] = @[]
  var nFp4, nFp32, nGraft = 0

  for t in tensors:
    # "graft-moi" = tensor KHONG ton tai trong checkpoint HF goc se duoc them
    # SAU BUOC NAY boi graft_attention_bin (LinearPlusAttention qkv/out moi,
    # LoRA A/B) - file .nimq nay CHI chua phan HF goc, nen moi tensor doc tu
    # day deu la "goc", khong co tensor "graft" thuc su - co reserve nhanh
    # phan loai theo ten de neu ban tu them tensor thu cong sau nay (vd merge
    # them 1 .nimq graft rieng) thi logic --graft-quant van dung duoc.
    let isGraftNew = "graft" in t.name.toLowerAscii or "lora_" in t.name.toLowerAscii

    var qt: QuantTensor
    if isPrecisionSensitive(t.name):
      qt = quantizeTensor(t.data, t.shape, qkFp32Raw)
      inc nFp32
    elif isGraftNew:
      qt = quantizeTensor(t.data, t.shape, graftKind, customCf = FP4_E2M1)
      inc nGraft
    elif "lm_head.weight" in t.name.toLowerAscii:
      # lm_head [32768,4096]: fp4km van gay ~0.1 relL2 sai so logit -> dung fp8 (128MB, cos 0.9996)
      qt = quantizeTensor(t.data, t.shape, qkFp8E4M3)
      inc nFp32  # dem chung voi fp32 de log ro
    else:
      # FP4-KM can tensor 2D va cols chia het groupSize (32). Moi weight lon
      # cua Codestral deu thoa (in_proj 18560x4096, out_proj 4096x8192,
      # lm_head 32768x4096, embed 32768x4096). Tensor nao khong thoa thi
      # ROI VE fp8 e4m3 - KHONG bao gio roi ve fp4 per-tensor cu.
      if mainKind == qkFp4KM and (t.shape.len != 2 or t.shape[1] mod 32 != 0):
        qt = quantizeTensor(t.data, t.shape, qkFp8E4M3)
      else:
        qt = quantizeTensor(t.data, t.shape, mainKind, customCf = FP4_E2M1)
      inc nFp4
    sd.add((t.name, qt))

  echo &"[quantize_codestral] main({mainQuantStr})={nFp4} fp32(nhay-sai-so)={nFp32} graft-moi({graftQuantStr})={nGraft}"

  # arch[5]: [vocab, hidden, nLayers, nHeads, ffMult] - Codestral-7B-v0.1
  # (theo config.json cua Mistral): hidden=4096, nLayers=64. vocab/nHeads/ffMult
  # KIEM TRA LAI voi params.json thuc te tai ve truoc khi dung so nay cho
  # nim_inference.nim - dien tam thoi de khong de arch rong (0,0,0,0,0).
  # arch[5]: [vocab, hidden, nLayers, nHeads(SSM), expand] - Codestral-7B-v0.1
  # XAC NHAN THAT tu manifest (print_manifest.nim): vocab=32768, hidden=4096,
  # nLayers=64, nHeads=128 (tu mixer.A_log=[128]), expand=2 (intermediate=8192).
  # LUU Y: arch nay KHONG duoc load_codestral_into_jamba.nim doc lai (loader
  # do map tensor THEO TEN, khong doc arch) - chi de tham khao/dung cho
  # nim_inference.nim (duong inference khac, xem ghi chu trong nim_inference.nim).
  let arch = [32768, 4096, 64, 128, 2]
  saveQuantStateDict(outPath, arch, sd)
  echo &"[quantize_codestral] Da ghi {outPath} ({sd.len} tensor) - dat vao thu muc project cua ban."
  echo "[quantize_codestral] CANH BAO QUAN TRONG: file .nimq nay dung schema tensor-name " &
       "cua HF Mamba-Codestral-7B-v0.1 GOC (kieu mamba_ssm/Mamba2), KHONG khop 1-1 voi " &
       "Mamba2SSD tu viet trong test_nimformer.nim - giong het van de da ghi trong " &
       "jamba_hybrid.loadMambaOnlyFromHFBin. Dung file nay lam INPUT cho buoc map ten " &
       "tensor (nameMap) truoc khi nap vao NimformerModelJamba, KHONG nap thang."