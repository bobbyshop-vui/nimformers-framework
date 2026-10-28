## load_codestral_into_jamba.nim
## Nap TRUC TIEP weight pretrain that cua Mamba-Codestral-7B-v0.1 vao model
## Jamba-hybrid (jamba_hybrid.nim) da co block 'R' (Mamba2Real - khop dung
## cong thuc). Cac block 'A' (LinearPlusAttention) van GIU NGUYEN random-init
## - day la phan "cay them" (graft) MOI, khong nam trong checkpoint HF goc,
## se duoc train (thang hoac qua LoRA - dung lora.nim, khong doi gi ca vi
## lora.nim generic theo ten tham so).
##
## QUY UOC MAP LAYER: cac block 'R' trong pattern cua ban duoc gan weight
## HF theo THU TU XUAT HIEN (block 'R' dau tien -> HF layer 0, block 'R' thu
## 2 -> HF layer 1, v.v...) - BO QUA cac layer HF tuong ung voi vi tri dat
## block 'A' (vi 'A' la layer MOI thay cho 1 layer Mamba nao do bi "cay de").
## Vi du pattern "RRRARRRA" (8 block, 6 'R' + 2 'A') se dung HF layer 0..5
## cho 6 block 'R', bo qua HF layer 6,7 (hoac bat ky layer HF nao) - KHONG co
## quan he 1-1 giua vi tri block va vi tri layer HF, chi giu THU TU.
##
## CANH BAO: chua verify ten tensor "backbone.embeddings.weight"/"lm_head.weight"
## voi manifest that cua Mamba-Codestral-7B-v0.1 (transformers co the doi ten
## giua version) - IN ra danh sach ten tensor KHONG map duoc o cuoi, tu doi
## lai hfNameMapMamba2/EMBED_NAME/LMHEAD_NAME trong file nay cho khop truoc
## khi tin ket qua.

import std/[tables, strformat, strutils, os]   # os: getEnv cho BYBY_ALLOW_RANDOM_EMBED
import nimformer
import jamba_hybrid
import mamba2_real
import sharded_hf_loader
import rawbin   # RawBinTensor
import std/os
import quant    # loadQuantStateDict, dequantizeTensor - cho nguon .nimq (da quantize fp4)

const
  EmbedName = "backbone.embeddings.weight"
  LmHeadName = "lm_head.weight"
  NormFName = "backbone.norm_f.weight"   # MOI: xem bao cao debug - norm cuoi truoc lm_head

proc findTensor(tensors: seq[RawBinTensor], name: string): int =
  for i, t in tensors:
    if t.name == name: return i
  return -1

proc loadCodestralIntoJambaFromTensors*(m: var NimformerModelJamba, tensors: seq[RawBinTensor]) =
  ## Ham LOI - nhan san 1 danh sach tensor da giai nen ve fp32 trong RAM,
  ## KHONG QUAN TAM tensor do den tu dau (safetensors goc DOC THANG, hay
  ## .nimq da quantize fp4 roi DEQUANTIZE lai) - 2 ham wrapper ben duoi
  ## (loadCodestralIntoJamba/loadCodestralIntoJambaFromNimq) chi khac nhau
  ## O BUOC LAY `tensors`, phan map ten/gan vao model GIONG HET nhau.
  echo &"[load_codestral] {tensors.len} tensor. Bat dau map vao model ..."

  var usedNames = initTable[string, bool]()

  # ── embed + lm_head ──────────────────────────────────────────
  let embedIdx = findTensor(tensors, EmbedName)
  if embedIdx >= 0:
    doAssert tensors[embedIdx].data.len == m.embed.weight.data.len,
      &"embed shape lech: HF={tensors[embedIdx].data.len} model={m.embed.weight.data.len}"
    m.embed.weight.data = tensors[embedIdx].data
    usedNames[EmbedName] = true
    echo &"[load_codestral] OK embed <- {EmbedName}"
  else:
    # SUA: truoc day chi `echo` roi chay tiep voi EMBEDDING NGAU NHIEN. Do la
    # cach chac chan nhat de sinh token rac 100% ma log van trong nhu binh
    # thuong va loss van "hoi tu" (ve mot model vo nghia). Dau file nay con
    # ghi ro "chua verify ten tensor" -> kha nang lech ten la THAT.
    raise newException(ValueError,
      &"[load_codestral] KHONG thay tensor '{EmbedName}' trong checkpoint HF. " &
      "Embedding random-init se gay token rac; sua EmbedName theo manifest.")

  let lmHeadIdx = findTensor(tensors, LmHeadName)
  if lmHeadIdx >= 0:
    doAssert tensors[lmHeadIdx].data.len == m.outProj.weight.data.len,
      &"lm_head shape lech: HF={tensors[lmHeadIdx].data.len} model={m.outProj.weight.data.len}"
    m.outProj.weight.data = tensors[lmHeadIdx].data
    usedNames[LmHeadName] = true
    echo "[load_codestral] OK outProj <- lm_head.weight"
  else:
    echo "[load_codestral] CANH BAO: khong thay 'lm_head.weight' - co the model dung " &
         "tied-embedding (lm_head = embed.weight, khong co tensor rieng) - " &
         "thu tam dung lai embed.weight cho outProj:"
    if embedIdx >= 0 and tensors[embedIdx].data.len == m.outProj.weight.data.len:
      m.outProj.weight.data = tensors[embedIdx].data
      echo "[load_codestral]   -> ap dung tied-embedding OK (shape khop)."
    else:
      # SUA: giong truong hop embed o tren - lm_head random-init nghia la
      # logits HOAN TOAN ngau nhien tren 32768 token. Khong duoc am tham.
      raise newException(ValueError,
        "[load_codestral] KHONG thay 'lm_head.weight' va cung KHONG the dung " &
        "tied-embedding (shape khong khop). Neu chay tiep thi lm_head la " &
        "RANDOM-INIT -> logits ngau nhien tren toan bo vocab -> token rac 100%. " &
        "Chay print_manifest.nim de doi chieu ten tensor that.")

  # ── normF (== backbone.norm_f, ap dung 1 lan truoc lm_head) ────────────
  # FIX token rac (bug #8): ham nay dung CHUNG cho ca loadCodestralIntoJamba
  # (safetensors goc) LAN loadCodestralIntoJambaFromNimq (dequant tu .nimq,
  # duong cua finetune_lora.nim). Fix normF truoc day (bug #4) CHI ap dung o
  # loadCodestralQuantResidentFromNimq (nhanh infer_lora_bin) -> 2 duong nay
  # van thieu normF, van chay gamma=1.
  let normFIdx = findTensor(tensors, NormFName)
  if normFIdx >= 0:
    doAssert tensors[normFIdx].data.len == m.normF.gamma.data.len,
      &"normF shape lech: HF={tensors[normFIdx].data.len} model={m.normF.gamma.data.len}"
    m.normF.gamma.data = tensors[normFIdx].data
    usedNames[NormFName] = true
    echo &"[load_codestral] OK normF <- {NormFName}"
  else:
    raise newException(ValueError,
      "[load_codestral] KHONG thay '" & NormFName & "' - normF random/identity se gay token rac.")

  # ── tung block 'R' theo thu tu xuat hien -> HF layer 0,1,2,... ──
  var hfLayerCounter = 0
  var matchedLayers = 0
  var totalLayers = 0
  for bi in 0 ..< m.blocks.len:
    if m.blocks[bi].kind notin {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}: continue
    totalLayers += 1
    let nameMap = hfNameMapMamba2(hfLayerCounter)
    var allOk = true
    var missing: seq[string] = @[]

    proc tryLoad(fieldData: var seq[float32], hfKey: string): bool =
      let idx = findTensor(tensors, nameMap[hfKey])
      if idx >= 0:
        if tensors[idx].data.len == fieldData.len:
          fieldData = tensors[idx].data
          usedNames[nameMap[hfKey]] = true
          return true
        else:
          echo &"[load_codestral] layer {hfLayerCounter} '{hfKey}': SHAPE LECH " &
               &"(HF={tensors[idx].data.len} model={fieldData.len}) - BO QUA, giu random-init"
          return false
      else:
        missing.add(nameMap[hfKey])
        return false

    if not tryLoad(m.blocks[bi].mambaReal.inProj.weight.data, "inProj.weight"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.convWeight.data, "convWeight"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.convBias.data, "convBias"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.aLog.data, "aLog"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.dSkip.data, "dSkip"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.dtBias.data, "dtBias"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.normWeight.data, "normWeight"): allOk = false
    if not tryLoad(m.blocks[bi].mambaReal.outProj.weight.data, "outProj.weight"): allOk = false
    # FIX token rac (bug #7): pre-norm CA BLOCK (ln1 == Mamba2Block.norm ben
    # HF, NGOAI submodule mixer - KHAC voi 'normWeight' o tren la norm BEN
    # TRONG mixer). Truoc day KHONG duoc nap o day -> ln1.gamma = 1.0 cho moi
    # block. Xem hfNameMapMamba2.
    if not tryLoad(m.blocks[bi].ln1.gamma.data, "ln1Weight"): allOk = false
    # inProj.bias / outProj.bias: Mamba2 HF THUONG khong co bias (bias=False)
    # -> KHONG coi la loi neu thieu, chi log info, giu bias=0 mac dinh.
    let inBiasIdx = findTensor(tensors, nameMap["inProj.bias"])
    if inBiasIdx >= 0 and tensors[inBiasIdx].data.len == m.blocks[bi].mambaReal.inProj.bias.data.len:
      m.blocks[bi].mambaReal.inProj.bias.data = tensors[inBiasIdx].data
      usedNames[nameMap["inProj.bias"]] = true

    if allOk:
      matchedLayers += 1
      echo &"[load_codestral] block{bi} <- HF mixer layer {hfLayerCounter}: OK (9/9 tensor chinh khop)"
    else:
      echo &"[load_codestral] block{bi} <- HF mixer layer {hfLayerCounter}: THIEU {missing}"
    hfLayerCounter += 1

  echo &"\n[load_codestral] TOM TAT: {matchedLayers}/{totalLayers} block 'R' map DAY DU. " &
       "Block nao khong 'OK' o tren se GIU random-init (train tu dau) - kiem tra lai " &
       "hfNameMapMamba2 neu so nay thap hon ky vong."
  if matchedLayers != totalLayers:
    raise newException(ValueError,
      &"[load_codestral] DUNG: chi map du {matchedLayers}/{totalLayers} block R; " &
       "khong cho phep train voi weight random-init de tranh token rac.")

  # liet ke tensor HF KHONG dung den (co the la bias khong ton tai o model
  # nay - binh thuong - hoac ten map SAI - can xem lai)
  var unusedSample: seq[string] = @[]
  for t in tensors:
    if t.name notin usedNames:
      unusedSample.add(t.name)
      if unusedSample.len >= 15: break
  if unusedSample.len > 0:
    echo &"[load_codestral] Vi du {unusedSample.len} tensor HF KHONG duoc dung: {unusedSample}"

proc loadCodestralQuantResidentFromNimq*(m: var NimformerModelJamba, nimqPath: string):
                                          Table[string, QuantTensor] =
  ## Ban "quant-resident" - KHONG dequantize inProj.weight/outProj.weight
  ## (2 ma tran lon nhat, ~29GB neu dequant het ca 52 block) - giu NGUYEN
  ## DANG NEN (QuantTensor, ~1 byte/phan tu voi fp4), tra ve trong bang
  ## ket qua de lora.nim (attachLoraQuantResident/mergeInOne) dequant TUC
  ## THOI qua LUT MOI KHI can 1 block, xoa ngay sau dung (xem
  ## trainStepGraftStreaming trong finetune_lora.nim).
  ## Cac tham so KHAC (embed, lm_head, aLog/dSkip/dtBias/normWeight/
  ## convWeight - deu nho, tong ~1.1GB) VAN dequantize BINH THUONG va gan
  ## thang vao model nhu loadCodestralIntoJambaFromNimq (khong quant-resident
  ## cho phan nay, khong can thiet).
  echo &"[load_codestral] (quant-resident) Doc {nimqPath} ..."
  var sd = quant.loadQuantStateDict(nimqPath, mapped = getEnv("BYBY_MODEL_MMAP", "1") != "0").sd
  var byName = initTable[string, QuantTensor]()
  for (name, qt) in sd.mitems: byName[name] = move(qt)
  # NGHI VAN #2 cho RSS-sau-load cao bat thuong (xem bao cao debug): truoc
  # day 'sd' (seq goc tu loadQuantStateDict) van song tiep TOI KHI HAM NAY
  # RETURN du da copy het sang byName roi - nghia la trong luc load co THE
  # ton tai 2-3 ban sao cung luc cua ca 129 tensor LON (sd + byName +
  # result, moi ban co the vai GB). Giai phong 'sd' NGAY sau khi da move
  # het sang byName (move o tren + xoa o day = chac chan khong con giu du).
  sd = @[]

  result = initTable[string, QuantTensor]()

  proc shapeNumel(shape: seq[int]): int =
    result = 1
    for d in shape: result *= d

  proc sourceShapeMatches(name: string; source, expected: seq[int]): bool =
    ## HF stores Conv1d weights as [outChannels, inChannels, kernelSize].
    ## Mamba2 here has depthwise Conv1d (inChannels=1), represented internally
    ## as [outChannels, kernelSize]. The singleton axis changes rank, not data.
    if source == expected: return true
    result = name.endsWith(".conv1d.weight") and source.len == 3 and
      expected.len == 2 and source[0] == expected[0] and source[1] == 1 and
      source[2] == expected[1]

  proc requireTensor(name: string, expectedShape: seq[int], missing: var seq[string]) =
    if name notin byName:
      missing.add(name & " (missing)")
    elif not sourceShapeMatches(name, byName[name].shape, expectedShape) or
         byName[name].bytes.len == 0:
      missing.add(name & " (shape=" & $byName[name].shape & ", expected=" &
                  $expectedShape & ", data=" & $byName[name].bytes.len & ")")

  proc dequantSmall(name: string): RawBinTensor =
    let qt = byName[name]
    let values = quant.dequantizeTensor(qt)
    if values.len != shapeNumel(qt.shape):
      raise newException(ValueError,
        &"[load_codestral] .nimq '{name}' giai nen ra {values.len} phan tu, " &
        &"nhung metadata shape {qt.shape} can {shapeNumel(qt.shape)}; file co the hong.")
    RawBinTensor(name: name, shape: qt.shape, data: values)

  # embed - dequant binh thuong (can cho lookupBatch truy cap tung hang
  # ngau nhien theo token id, khong phai 1 phep matmul don le nhu lm_head
  # nen giu resident luon cho don gian, ~0.54GB).
  var smallTensors: seq[RawBinTensor] = @[]
  var missingRequired: seq[string] = @[]
  requireTensor(EmbedName, m.embed.weight.shape, missingRequired)
  requireTensor(NormFName, m.normF.gamma.shape, missingRequired)
  if EmbedName in byName: smallTensors.add(dequantSmall(EmbedName))
  if NormFName in byName: smallTensors.add(dequantSmall(NormFName))   # MOI
  # lm_head - SUA: KHONG dequant eager nua (tiet kiem ~0.54GB thuong tru RAM
  # suot ca qua trinh train) - dua vao `result` (bang quant-resident) DUNG
  # TEN THAM SO THAT ("outProj.weight") giong het 52 block 'R'. lm_head chi
  # dung DUNG 1 LAN/buoc train (cuoi forward tinh logits, dau backward tinh
  # dX) - xem mergeInOnePlain/restoreBaseOnePlain trong lora.nim + cho goi
  # trong trainStepGraftStreaming (finetune_lora.nim) bao quanh 2 lan dung
  # do. KHONG dua vao smallTensors nua vi se bi dequant eager o duoi.
  if LmHeadName in byName:
    let qt = byName[LmHeadName]
    var expectedLen = 1
    for d in qt.shape: expectedLen *= d
    if qt.shape == m.outProj.weight.shape and qt.bytes.len > 0:
      result["outProj.weight"] = qt
      byName.del(LmHeadName)   # giai phong ban trong byName, result da giu roi
    else:
      missingRequired.add(LmHeadName & " (shape=" & $qt.shape & ")")
  else:
    # Cho phép checkpoint tied-embedding hợp lệ: dùng đúng embed làm lm_head.
    # Nếu shape không khớp thì giữ fail-fast, tuyệt đối không để random lm_head.
    if EmbedName in byName and byName[EmbedName].shape == m.outProj.weight.shape and
       byName[EmbedName].bytes.len > 0:
      discard
    else:
      missingRequired.add(LmHeadName & " (missing and no valid tied embedding)")

  var hfLayerCounter = 0
  var rBlockIdx: seq[int] = @[]
  for bi in 0 ..< m.blocks.len:
    if m.blocks[bi].kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}: rBlockIdx.add(bi)

  for bi in rBlockIdx:
    let nameMap = hfNameMapMamba2(hfLayerCounter)
    requireTensor(nameMap["inProj.weight"], m.blocks[bi].mambaReal.inProj.weight.shape, missingRequired)
    requireTensor(nameMap["outProj.weight"], m.blocks[bi].mambaReal.outProj.weight.shape, missingRequired)
    requireTensor(nameMap["convWeight"], m.blocks[bi].mambaReal.convWeight.shape, missingRequired)
    requireTensor(nameMap["convBias"], m.blocks[bi].mambaReal.convBias.shape, missingRequired)
    requireTensor(nameMap["aLog"], m.blocks[bi].mambaReal.aLog.shape, missingRequired)
    requireTensor(nameMap["dSkip"], m.blocks[bi].mambaReal.dSkip.shape, missingRequired)
    requireTensor(nameMap["dtBias"], m.blocks[bi].mambaReal.dtBias.shape, missingRequired)
    requireTensor(nameMap["normWeight"], m.blocks[bi].mambaReal.normWeight.shape, missingRequired)
    requireTensor(nameMap["ln1Weight"], m.blocks[bi].ln1.gamma.shape, missingRequired)
    # 2 ma tran LON - GIU NGUYEN NEN, dua vao result voi TEN THEO PARAMLIST
    # (khop dung "block{bi}.mambaReal.inProj.weight" ma paramListMamba2Real
    # dat, KHONG phai ten HF goc - de lora.nim tra cuu thang theo ten
    # adapter khong can map lai).
    if nameMap["inProj.weight"] in byName:
      result[&"block{bi}.mambaReal.inProj.weight"] = byName[nameMap["inProj.weight"]]
      byName.del(nameMap["inProj.weight"])   # giai phong ban trong byName
    else:
      echo &"[load_codestral] CANH BAO: layer {hfLayerCounter} thieu in_proj.weight trong .nimq"
    if nameMap["outProj.weight"] in byName:
      result[&"block{bi}.mambaReal.outProj.weight"] = byName[nameMap["outProj.weight"]]
      byName.del(nameMap["outProj.weight"])   # giai phong ban trong byName
    else:
      echo &"[load_codestral] CANH BAO: layer {hfLayerCounter} thieu out_proj.weight trong .nimq"

    # tham so NHO - dequant binh thuong nhu cu (convWeight/convBias/aLog/
    # dSkip/dtBias/normWeight - tong vai chuc KB/block, khong dang ke).
    # ln1Weight MOI THEM - FIX token rac (bug #7): pre-norm ca block
    # (backbone.layers.N.norm.weight, NGOAI mixer) truoc day KHONG nam trong
    # danh sach nay -> ln1.gamma khong bao gio duoc nap that, giu gamma=1.
    for key in ["convWeight", "convBias", "aLog", "dSkip", "dtBias", "normWeight", "ln1Weight"]:
      if nameMap[key] in byName:
        smallTensors.add(dequantSmall(nameMap[key]))
    hfLayerCounter += 1

  # gan truc tiep cac tham so NHO vao model (KHONG goi loadCodestralIntoJambaFromTensors -
  # ham do se bao "THIEU inProj.weight/outProj.weight" cho MOI block 'R' vi
  # co tinh khong dua 2 tensor lon do vao smallTensors - gay hieu lam, tuy
  # khong sai chuc nang).
  proc findRaw(tensors: seq[RawBinTensor], name: string): int =
    for i, t in tensors:
      if t.name == name: return i
    return -1

  let embedIdx = findRaw(smallTensors, EmbedName)
  if embedIdx >= 0 and smallTensors[embedIdx].data.len == shapeNumel(m.embed.weight.shape):
    m.embed.weight.data = smallTensors[embedIdx].data
    echo "[load_codestral] (quant-resident) OK embed"
  let normFIdx = findRaw(smallTensors, NormFName)   # MOI: xem bao cao debug
  if normFIdx >= 0 and smallTensors[normFIdx].data.len == shapeNumel(m.normF.gamma.shape):
    m.normF.gamma.data = smallTensors[normFIdx].data
    echo "[load_codestral] (quant-resident) OK normF (norm cuoi truoc lm_head)"
  else:
    raise newException(ValueError, "[load_codestral] normF khong nap duoc - tu choi chay de tranh token rac")
  if "outProj.weight" in result:
    echo "[load_codestral] (quant-resident) OK outProj (lm_head) - giu nen, dequant tam thoi luc dung"
  elif embedIdx >= 0 and smallTensors[embedIdx].data.len == shapeNumel(m.outProj.weight.shape):
    # tied-embedding that su (khong co lm_head.weight rieng trong file .nimq) -
    # truong hop nay VAN dequant thang (khong quant-resident duoc vi dung
    # chung storage voi embed, da dequant o tren roi).
    m.outProj.weight.data = smallTensors[embedIdx].data
    echo "[load_codestral] (quant-resident) OK outProj (tied-embedding)"

  hfLayerCounter = 0
  var ln1MissingCount = 0
  # Dem THUC TE so field gan duoc — truoc day log "N tham so nho da dequant+gan"
  # chi dem smallTensors.len (so tensor trong file), KHONG dem so field model
  # nhan gia tri. Shape lech / ten lech -> field giu random/init, token rac im lang.
  var smallAssigned = 0
  var smallMissed = 0
  var smallMissNames: seq[string] = @[]
  for bi in rBlockIdx:
    let nameMap = hfNameMapMamba2(hfLayerCounter)
    template assignSmall(field: var seq[float32], key: string) =
      let idx = findRaw(smallTensors, nameMap[key])
      if idx >= 0 and smallTensors[idx].data.len == field.len:
        field = smallTensors[idx].data
        inc smallAssigned
      else:
        inc smallMissed
        if smallMissNames.len < 24:
          let whyStr =
            if idx < 0: "missing"
            else: "shape " & $smallTensors[idx].data.len & " != model " & $field.len
          # Tranh &"..." trong template (hygiene: key/why undeclared).
          smallMissNames.add("block" & $bi & "." & key & " (" & whyStr & ")")
    assignSmall(m.blocks[bi].mambaReal.convWeight.data, "convWeight")
    assignSmall(m.blocks[bi].mambaReal.convBias.data, "convBias")
    assignSmall(m.blocks[bi].mambaReal.aLog.data, "aLog")
    assignSmall(m.blocks[bi].mambaReal.dSkip.data, "dSkip")
    assignSmall(m.blocks[bi].mambaReal.dtBias.data, "dtBias")
    assignSmall(m.blocks[bi].mambaReal.normWeight.data, "normWeight")
    # FIX token rac (bug #7): ln1.gamma = pre-norm CA BLOCK, ap dung TRUOC
    # mixer - khac 'normWeight' o tren (norm BEN TRONG mixer). Truoc day
    # khong nam trong assignSmall nao -> gamma=1 cho moi block. Dem rieng
    # (khong dung assignSmall vi template do khong tra ve bool) de canh bao
    # gop 1 dong thay vi spam 64 dong.
    let ln1Idx = findRaw(smallTensors, nameMap["ln1Weight"])
    if ln1Idx >= 0 and smallTensors[ln1Idx].data.len == m.blocks[bi].ln1.gamma.data.len:
      m.blocks[bi].ln1.gamma.data = smallTensors[ln1Idx].data
      inc smallAssigned
    else:
      ln1MissingCount += 1
      inc smallMissed
    hfLayerCounter += 1

  if ln1MissingCount > 0:
    echo &"[load_codestral] CANH BAO: {ln1MissingCount}/{rBlockIdx.len} block THIEU ln1.gamma " &
         "(backbone.layers.N.norm.weight - pre-norm truoc mixer, KHONG phai " &
         "mixer.norm.weight) trong .nimq - cac block nay dang chay ln1 voi " &
         "gamma=1 (identity, SAI so voi checkpoint -> token rac)."
  else:
    echo &"[load_codestral] (quant-resident) OK ln1.gamma ca {rBlockIdx.len} block (pre-norm truoc mixer)"

  if smallMissed > 0:
    echo &"[load_codestral] CANH BAO: assignSmall that bai {smallMissed} field " &
         &"(gan thanh cong {smallAssigned}). Field miss giu init/random -> token rac."
    for nm in smallMissNames:
      stderr.writeLine "[load_codestral]   miss: " & nm
    # SSM small weights (aLog/dtBias/normWeight/...) lech shape = logits lech nang.
    # Fail-loud de khong train/infer voi backbone bi thieu tham so.
    raise newException(ValueError,
      &"[load_codestral] {smallMissed} small-tensor field khong gan duoc " &
      &"(vi du: {smallMissNames[0 ..< min(smallMissNames.len, 6)]}). " &
      "Sua .nimq / shape map truoc khi train/infer.")

  if missingRequired.len > 0:
    raise newException(ValueError,
      "[load_codestral] .nimq thieu/sai weight BAT BUOC - khong train/infer de tranh token rac: " &
      $missingRequired[0 ..< min(missingRequired.len, 12)])

  echo &"[load_codestral] (quant-resident) {result.len} tham so LON giu dang nen " &
       &"({rBlockIdx.len} block 'R'), {smallAssigned} field nho gan THANH CONG " &
       &"(smallTensors trong file={smallTensors.len})."

proc loadCodestralIntoJamba*(m: var NimformerModelJamba, hfDir: string) =
  ## Doc THANG tu safetensors goc (~14GB bf16 tren dia, ~28GB sau khi giai
  ## nen fp32 trong RAM - xem canh bao RAM trong sharded_hf_loader.nim).
  echo &"[load_codestral] Doc {hfDir} (chi RAM) ..."
  let tensors = loadAllHFTensorsSharded(hfDir)
  loadCodestralIntoJambaFromTensors(m, tensors)

proc loadCodestralIntoJambaFromNimq*(m: var NimformerModelJamba, nimqPath: string) =
  ## Doc tu FILE DA QUANTIZE fp4 (~3.5GB tren dia - xem quantize_codestral.nim
  ## de tao file nay MOT LAN tu safetensors goc, sau do co the XOA safetensors
  ## goc di, tiet kiem o) - DEQUANTIZE lai ve fp32 TRONG RAM luc load (train
  ## van can fp32 vi Mamba2Real.nim chi co ham toan hoc tren float32 thuan,
  ## KHONG co kernel train truc tiep tren du lieu da quantize - fp4 O DAY LA
  ## DE TIET KIEM O DIA, KHONG PHAI tiet kiem RAM luc train).
  echo &"[load_codestral] Doc {nimqPath} (.nimq, da quantize) roi dequantize ve fp32 ..."
  let (_, sd) = quant.loadQuantStateDict(nimqPath)
  var tensors: seq[RawBinTensor] = @[]
  for (name, qt) in sd:
    tensors.add(RawBinTensor(name: name, shape: qt.shape, data: quant.dequantizeTensor(qt)))
  echo &"[load_codestral] Da dequantize {tensors.len} tensor tu .nimq"
  loadCodestralIntoJambaFromTensors(m, tensors)
