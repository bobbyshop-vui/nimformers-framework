## hf_tokenizer.nim
## Tokenizer THẬT cho Mamba-Codestral-7B-v0.1, qua nimpy (giống hệt cách
## pytorch_bin_loader.nim gọi Python cho torch/safetensors - CÙNG 1 kiểu
## dependency, không thêm cơ chế mới).
##
## TẠI SAO FILE NÀY BẮT BUỘC PHẢI CÓ (không phải optional):
## Embedding + lm_head của Codestral-7B đã pretrain gắn CHẶT với 1 bảng id
## token cụ thể (tekken vocab, ~32768 dòng) - hàng thứ 5000 trong embedding
## nghĩa là ĐÚNG 1 token cụ thể mà tekken sinh ra. BPETokenizer tự train từ
## đầu trên vài dòng dolly-15k sinh ra 1 bảng id HOÀN TOÀN KHÁC - dùng nhầm
## sẽ khiến toàn bộ phần train/infer đi qua embedding pretrain trở thành TRA
## CỨU NGẪU NHIÊN. Đây là lý do CFG_INIT_MODE=="codestral" BẮT BUỘC dùng
## module này, không được rơi về getOrTrainBPE.
##
## ═══════════════════════════════════════════════════════════════════════
## BẢN VÁ "TOKEN RÁC" - những gì THÊM so với bản trước:
##
##  (1) eosId/bosId/unkId: TRƯỚC ĐÂY KHÔNG HỀ LỘ RA NGOÀI. Hệ quả:
##      infer_lora.nim không có cách nào dừng khi model phát </s>, nên sau
##      khi trả lời xong nó sinh tiếp <s> [INST] ... = 1 document MỚI. Và vì
##      SentencePiece decode BỎ QUA control token, bạn KHÔNG NHÌN THẤY </s>
##      trong output - chỉ thấy model "lan man". Rất dễ chẩn đoán nhầm sang
##      lỗi model/lỗi train.
##
##  (2) special[]/banned[]/stopIds: bảng đánh dấu token điều khiển. tekken
##      dành ~1000 id ĐẦU cho control + <SPECIAL_k> placeholder CHƯA DÙNG
##      BAO GIỜ; hàng lm_head của chúng gần như không được train nên logit
##      là nhiễu thuần - nhưng vẫn nằm trong softmax và vẫn sample được.
##      Cộng thêm việc lm_head bị quantize FP4 (xem quantize_codestral.nim)
##      thì sai số lượng tử đủ để đẩy chúng lên trên. Giờ chặn ở khâu sample.
##
##  (3) decodeHf(skipSpecial=true) MẶC ĐỊNH: mistral_common Tekkenizer ở 1
##      số version RAISE khi decode trúng special token -> generate crash
##      giữa chừng. Giờ lọc TRƯỚC khi gọi xuống Python, không bao giờ raise.
##
##  (4) encodeHf(addBos=...): trước đây bos=true hard-code.
##
##  (5) backend "hf_tokenizers" + addEos: trước đây ÂM THẦM BỎ QUA (comment
##      cũ tự thừa nhận). Nếu repo thiếu tekken.json/tokenizer.model.v3 mà
##      có tokenizer.json, bạn rơi vào nhánh này và TOÀN BỘ dataset train
##      KHÔNG CÓ MỘT EOS NÀO -> model không bao giờ học dừng. Giờ tự chèn
##      eosId nếu tìm được, RAISE nếu không.
## ═══════════════════════════════════════════════════════════════════════
##
## YÊU CẦU: `pip install mistral-common` (ưu tiên - đúng tokenizer gốc
## "tekken" của Mistral) HOẶC `pip install tokenizers` (fallback).

import std/[os, strformat, strutils]
import nimpy

type
  HfTokenizer* = object
    backend*: string       # "mistral_common" hoac "hf_tokenizers"
    pyTok*: PyObject
    vocab*: int
    bosId*: int            # -1 = khong xac dinh duoc
    eosId*: int
    unkId*: int
    special*: seq[bool]    # special[id]=true -> token DIEU KHIEN, khong in ra
    banned*: seq[bool]     # banned[id]=true -> KHONG duoc sample (= special
                           #   + garbageSpace, tru cac stop token)
    garbageSpace*: seq[bool] # token DUNG LA whitespace-marker (vd "▁") ->
                             # KHONG cho model sample. Khong dung decode()==""
                             # de phan loai vi token prefix-space hop le co the
                             # decode rieng thanh chuoi rong.
    stopIds*: seq[int]     # gap 1 trong cac id nay -> DUNG generate

let pyBuiltins = pyImport("builtins")
proc pyLen(o: PyObject): int = pyBuiltins.len(o).to(int)

proc findTekkenFile(hfDir: string): string =
  for cand in ["tekken.json", "tokenizer.model.v3", "tokenizer.model"]:
    let p = hfDir / cand
    if fileExists(p): return p
  return ""

proc findHfTokenizerJson(hfDir: string): string =
  let p = hfDir / "tokenizer.json"
  if fileExists(p): return p
  return ""

# ───────────────────────────────────────────────────────────────────────
# DO BANG SPECIAL TOKEN
# ───────────────────────────────────────────────────────────────────────

proc rawPiece(tok: HfTokenizer, vocabList: PyObject, id: int): string =
  ## Chuoi "piece" THO cua 1 id (vd "▁hello", "<s>", "[INST]", "<0x0A>").
  ## "" neu khong lay duoc - ben goi phai chiu duoc truong hop nay.
  ## vocabList da duoc lay SAN 1 lan o buildSpecialTables (goi .vocab() moi
  ## id la 1024 lan dung danh sach 32768 phan tu qua nimpy - rat cham).
  try:
    case tok.backend
    of "mistral_common":
      if not vocabList.isNil and id < pyLen(vocabList):
        return vocabList[id].to(string)
    of "hf_tokenizers":
      return tok.pyTok.id_to_token(id).to(string)
    else:
      discard
  except CatchableError:
    discard
  return ""

proc decodeOne(tok: HfTokenizer, id: int): tuple[ok: bool, s: string] =
  ## decode 1 id DUY NHAT. ok=false nghia la Python nem exception (tekken
  ## voi special_token_policy=RAISE lam dung the) - cung duoc coi la special.
  try:
    case tok.backend
    of "mistral_common":
      return (true, tok.pyTok.instruct_tokenizer.tokenizer.decode(@[id]).to(string))
    of "hf_tokenizers":
      return (true, tok.pyTok.decode(@[id], skip_special_tokens = true).to(string))
    else: return (false, "")
  except CatchableError:
    return (false, "")

proc isWhitespaceLikePiece(piece: string): bool =
  ## Piece chi gom khoang trang duoi moi dang ma hoa (SentencePiece "▁",
  ## byte-level BPE "Ġ"/"Ċ", hoac space/tab/newline that).
  ## CAN THIET vi SentencePiece decode(["▁"]) tra ve "" (no strip dummy
  ## prefix) - neu chi dua vao "decode rong => special" thi token space don
  ## se bi ban NHAM, model mat kha nang sinh khoang trang/thut dong trong code.
  if piece.len == 0: return false
  var rest = piece
  for w in ["\u2581", "\u0120", "\u010A", " ", "\t", "\n", "\r"]:
    rest = rest.replace(w, "")
  return rest.len == 0

proc isStandaloneSpacePiece(piece: string): bool =
  ## Chi bat token BAN THAN la dau cach/space-marker.
  ## KHONG bat "▁Hello" / "ĠHello" vi do la token van ban hop le.
  piece == "\u2581" or piece == "\u0120" or piece == " "

proc tokenIdForRawPiece(tok: HfTokenizer, piece: string): int =
  ## Tim ID theo raw vocabulary ma khong quet 32768 ID.
  try:
    case tok.backend
    of "mistral_common":
      return tok.pyTok.instruct_tokenizer.tokenizer.token_to_id(piece).to(int)
    of "hf_tokenizers":
      return tok.pyTok.token_to_id(piece).to(int)
    else:
      discard
  except CatchableError:
    discard
  -1

proc markEncodedIdsForText(tok: var HfTokenizer, text: string) =
  ## Fallback quan trong cho Tekken/V3: encode mot chuoi chi co space
  ## de lay ID that, thay vi hard-code 29473.
  try:
    case tok.backend
    of "mistral_common":
      let ids = tok.pyTok.instruct_tokenizer.tokenizer.encode(text, false, false)
      let n = pyLen(ids)
      for i in 0 ..< n:
        let id = ids[i].to(int)
        if id >= 0 and id < tok.vocab:
          tok.garbageSpace[id] = true
    of "hf_tokenizers":
      let enc = tok.pyTok.encode(text, add_special_tokens = false)
      let ids = enc.ids
      let n = pyLen(ids)
      for i in 0 ..< n:
        let id = ids[i].to(int)
        if id >= 0 and id < tok.vocab:
          tok.garbageSpace[id] = true
    else:
      discard
  except CatchableError:
    discard

proc pieceOfHf*(tok: HfTokenizer, id: int): string =
  ## Ten piece THO cua token (vd "▁", "▁The", "<0x0A>") - de chan doan; ""
  ## neu backend khong ho tro.
  try:
    case tok.backend
    of "mistral_common":
      let t = tok.pyTok.instruct_tokenizer.tokenizer
      try:
        return callMethod(t, "id_to_piece", id).to(string)
      except CatchableError:
        return callMethod(t.getAttr("_model"), "id_to_piece", id).to(string)
    else: discard
  except CatchableError:
    discard
  result = ""

const StopPieceNames = ["</s>", "[INST]", "[/INST]", "<|im_end|>",
                        "<|endoftext|>", "<|eot_id|>"]
  # "[INST]" cung la stop: model hoc tu dataset packed se co xu huong mo luot
  # hoi moi sau khi tra loi xong. Do la diem DUNG, khong phai token de in ra.

proc buildSpecialTables(tok: var HfTokenizer) =
  ## Quet 1 LAN luc khoi tao. Chi quet 1024 id DAU - toan bo khoi control cua
  ## ca tekken (num_special_tokens=1000) lan sentencepiece v3 (~10 id) deu nam
  ## o day; quet ca 32768 id la 32768 loi goi Python, khong dang.
  tok.special = newSeq[bool](tok.vocab)
  tok.banned = newSeq[bool](tok.vocab)
  tok.garbageSpace = newSeq[bool](tok.vocab)
  tok.stopIds = @[]

  # QUAN TRONG: token space rac co the nam ngoai 1024 ID dau.
  # Khong hard-code 29473; lay tu raw vocab va fallback encode(" ").
  for piece in ["\u2581", "\u0120", " "]:
    let id = tokenIdForRawPiece(tok, piece)
    if id >= 0 and id < tok.vocab:
      tok.garbageSpace[id] = true
  markEncodedIdsForText(tok, " ")

  var vocabList: PyObject = nil
  if tok.backend == "mistral_common":
    try:
      vocabList = tok.pyTok.instruct_tokenizer.tokenizer.vocab()
    except CatchableError:
      vocabList = nil

  let limit = min(tok.vocab, 1024)
  for id in 0 ..< limit:
    let piece = rawPiece(tok, vocabList, id)
    if tok.garbageSpace[id]:
      continue                      # standalone-space da tach rieng, khong coi la special
    if isWhitespaceLikePiece(piece):
      continue                      # whitespace co noi dung van duoc phep
    let (ok, s) = decodeOne(tok, id)
    if (not ok) or s.len == 0:
      tok.special[id] = true
    if piece.len > 0:
      for sp in StopPieceNames:
        if piece == sp:
          tok.special[id] = true
          if id notin tok.stopIds: tok.stopIds.add(id)

  # eos luon la stop (ke ca khi khong doc duoc piece string)
  if tok.eosId >= 0 and tok.eosId < tok.vocab:
    tok.special[tok.eosId] = true
    if tok.eosId notin tok.stopIds: tok.stopIds.add(tok.eosId)
  if tok.bosId >= 0 and tok.bosId < tok.vocab: tok.special[tok.bosId] = true
  if tok.unkId >= 0 and tok.unkId < tok.vocab: tok.special[tok.unkId] = true

  # banned = special + standalone-space garbage TRU stop.
  for id in 0 ..< tok.vocab:
    tok.banned[id] = tok.special[id] or tok.garbageSpace[id]
  for id in tok.stopIds: tok.banned[id] = false

  if getEnv("BYBY_ALLOW_SPECIAL", "0") == "1":
    # cua thoat de debug: cho phep sample MOI token (vd de tai hien lai loi cu)
    for id in 0 ..< tok.vocab: tok.banned[id] = false
    echo "[hf_tokenizer] BYBY_ALLOW_SPECIAL=1 - TAT chan special token luc sample (debug)"

  var nSpecial = 0
  for b in tok.special:
    if b: inc nSpecial
  var garbageIds: seq[int] = @[]
  for id, b in tok.garbageSpace:
    if b: garbageIds.add(id)
  echo &"[hf_tokenizer] special={nSpecial} token (quet {limit} id dau), " &
       &"bos={tok.bosId} eos={tok.eosId} unk={tok.unkId}, stopIds={tok.stopIds}"
  echo &"[hf_tokenizer] standalone-space token IDs (KHONG cho sample)=" & $garbageIds
  if tok.eosId < 0:
    echo "[hf_tokenizer] CANH BAO NANG: khong xac dinh duoc eos_id. generate() " &
         "se khong co diem dung theo id (chi con stopIds do tu ten piece). " &
         "Kiem tra lai version mistral_common/tokenizers."

proc mcSpecialIds(inner: PyObject): tuple[bos, eos, unk: int] =
  ## Doc bos_id/eos_id/unk_id cua tokenizer mistral_common. -1 = khong doc
  ## duoc (API doi giua cac version) - ben goi phai chiu duoc truong hop do.
  var b = -1
  var e = -1
  var u = -1
  try:
    b = inner.bos_id.to(int)
  except CatchableError:
    discard
  try:
    e = inner.eos_id.to(int)
  except CatchableError:
    discard
  try:
    u = inner.unk_id.to(int)
  except CatchableError:
    discard
  result = (bos: b, eos: e, unk: u)

proc pyTokenToId(pyTok: PyObject, name: string): int =
  try:
    let v = pyTok.token_to_id(name)
    return v.to(int)
  except CatchableError:
    return -1

proc readSpecialIds(tok: var HfTokenizer) =
  tok.bosId = -1
  tok.eosId = -1
  tok.unkId = -1
  case tok.backend
  of "mistral_common":
    let inner = tok.pyTok.instruct_tokenizer.tokenizer
    let ids = mcSpecialIds(inner)
    tok.bosId = ids.bos
    tok.eosId = ids.eos
    tok.unkId = ids.unk
  of "hf_tokenizers":
    for name in ["<s>", "<|begin_of_text|>"]:
      let v = pyTokenToId(tok.pyTok, name)
      if v >= 0:
        tok.bosId = v
        break
    for name in ["</s>", "<|end_of_text|>", "<|endoftext|>"]:
      let v = pyTokenToId(tok.pyTok, name)
      if v >= 0:
        tok.eosId = v
        break
    tok.unkId = pyTokenToId(tok.pyTok, "<unk>")
  else:
    discard

proc newHfTokenizer*(hfDir: string): HfTokenizer =
  ## Thu mistral_common truoc (dung "tekken" - tokenizer GOC cua Codestral),
  ## fallback sang thu vien `tokenizers` (HF) neu repo co tokenizer.json,
  ## RAISE ro rang neu KHONG tim thay gi ca - KHONG bao gio am tham roi ve
  ## BPE tu che.
  let tekkenPath = findTekkenFile(hfDir)
  if tekkenPath.len > 0:
    try:
      let mc = pyImport("mistral_common.tokens.tokenizers.mistral")
      let tokCls = mc.MistralTokenizer
      let pyTok = tokCls.from_file(tekkenPath)
      let vocabSz = pyLen(pyTok.instruct_tokenizer.tokenizer.vocab())
      echo &"[hf_tokenizer] OK - mistral_common tekken tu {tekkenPath}, vocab={vocabSz}"
      result = HfTokenizer(backend: "mistral_common", pyTok: pyTok, vocab: vocabSz)
      readSpecialIds(result)
      buildSpecialTables(result)
      return result
    except CatchableError as e:
      echo &"[hf_tokenizer] mistral_common that bai ({e.msg}) - thu fallback tokenizers ..."

  let hfJsonPath = findHfTokenizerJson(hfDir)
  if hfJsonPath.len > 0:
    try:
      let tokenizers = pyImport("tokenizers")
      let pyTok = tokenizers.Tokenizer.from_file(hfJsonPath)
      let vocabSz = pyTok.get_vocab_size().to(int)
      echo &"[hf_tokenizer] OK - HF tokenizers tu {hfJsonPath}, vocab={vocabSz}"
      echo "[hf_tokenizer] LUU Y: day la backend FALLBACK, KHONG phai duong that " &
           "cho Codestral (tekken). Kiem tra ky vocab/id truoc khi train that."
      result = HfTokenizer(backend: "hf_tokenizers", pyTok: pyTok, vocab: vocabSz)
      readSpecialIds(result)
      buildSpecialTables(result)
      return result
    except CatchableError as e:
      raise newException(IOError,
        &"[hf_tokenizer] Ca mistral_common lan tokenizers deu that bai ({e.msg}).")

  raise newException(IOError,
    &"[hf_tokenizer] KHONG tim thay tekken.json/tokenizer.model* lan tokenizer.json " &
    &"trong {hfDir}. KHONG the finetune/infer dung nghia neu khong co tokenizer THAT " &
    "cua chinh checkpoint nay - dung getOrTrainBPE (BPE tu che) se lam embedding " &
    "pretrain tro thanh tra cuu ngau nhien (xem giai thich dau file). Tai lai model " &
    "voi day du file tokenizer (huggingface-cli download se tu tai kem neu co).")

# ───────────────────────────────────────────────────────────────────────
# ENCODE / DECODE
# ───────────────────────────────────────────────────────────────────────

proc encodeHf*(tok: HfTokenizer, text: string,
               addEos: bool = false, addBos: bool = true): seq[int] =
  ## `addEos`: dung true cho VI DU TRAIN DAY DU (prompt+response) de model hoc
  ## dung diem dung THAT (id token, KHONG phai nhet chuoi van ban "</s>" vao
  ## `text` - sentencepiece KHONG tu nhan dien "</s>" go tay thanh 1 token dac
  ## biet). Giu false khi encode PROMPT luc inference.
  ##
  ## `addBos`: THEM MOI (truoc day hard-code true). Van mac dinh true de giu
  ## nguyen hanh vi cu o moi cho dang goi.
  ##
  ## CANH BAO VE TEMPLATE (chua sua o day vi sua se lam LECH ca train lan
  ## infer cung luc, phai lam thanh 1 buoc rieng co do lai loss):
  ## "[INST]"/"[/INST]" dang duoc GO TAY vao `text` trong finetune_lora.
  ## lineToText -> chung di qua tokenizer THO nay nhu VAN BAN THUONG, KHONG
  ## thanh control token id 3/4. Train va infer van NHAT QUAN voi nhau nen
  ## khong sinh token rac truc tiep, nhung ban dang vut bo prior instruct ma
  ## base da pretrain. Muon dung that thi phai goi
  ## instruct_tokenizer.encode_instruct(...) chu khong phai .tokenizer.encode().
  case tok.backend
  of "mistral_common":
    let ids = tok.pyTok.instruct_tokenizer.tokenizer.encode(text, addBos, addEos)
    let n = pyLen(ids)
    result = newSeq[int](n)
    for i in 0 ..< n: result[i] = ids[i].to(int)
  of "hf_tokenizers":
    let enc = tok.pyTok.encode(text, add_special_tokens = false)
    let ids = enc.ids
    let n = pyLen(ids)
    result = @[]
    if addBos:
      if tok.bosId < 0:
        raise newException(ValueError,
          "[hf_tokenizer] backend 'hf_tokenizers': addBos=true nhung khong xac dinh " &
          "duoc bos_id. KHONG the encode dung - dung tekken.json/tokenizer.model.v3.")
      result.add(tok.bosId)
    for i in 0 ..< n: result.add(ids[i].to(int))
    if addEos:
      # SUA: truoc day am tham BO QUA addEos o backend nay -> toan bo dataset
      # train KHONG CO EOS -> model khong bao gio hoc dung. Gio raise ro rang.
      if tok.eosId < 0:
        raise newException(ValueError,
          "[hf_tokenizer] backend 'hf_tokenizers': addEos=true nhung khong xac dinh " &
          "duoc eos_id. Neu train tiep, TOAN BO dataset se khong co EOS va model " &
          "KHONG BAO GIO hoc duoc diem dung. Dung tekken.json/tokenizer.model.v3 " &
          "cua chinh checkpoint thay vi tokenizer.json.")
      result.add(tok.eosId)
  else:
    raise newException(ValueError, "encodeHf: backend khong hop le: " & tok.backend)

  # Chot cuoi: khong bao gio de lot id ngoai [0, vocab) ra ngoai. Neu lot,
  # softmaxCrossEntropy se cong gradient vao SAI HANG thoi gian mot cach IM
  # LANG (base+tgt van nam trong mang phang -> bounds check cua Nim khong bat
  # duoc). Xem giai thich day du trong test_nimformer.softmaxCrossEntropy.
  for id in result:
    if id < 0 or id >= tok.vocab:
      raise newException(ValueError,
        &"[hf_tokenizer] tokenizer sinh id={id} NGOAI [0,{tok.vocab}). " &
        "Vocab tokenizer va vocab model KHONG khop - dung ngay, khong train tiep.")

proc decodeHf*(tok: HfTokenizer, ids: seq[int], skipSpecial: bool = true): string =
  ## skipSpecial MAC DINH true:
  ##   - mistral_common Tekkenizer o 1 so version RAISE khi gap special token
  ##     -> loc TRUOC khi goi xuong Python, khong bao gio de no raise.
  ##   - khong in "<s>"/"[INST]"/"<SPECIAL_37>" ra man hinh nguoi dung.
  var clean: seq[int]
  if skipSpecial:
    clean = newSeqOfCap[int](ids.len)
    for id in ids:
      if id < 0 or id >= tok.vocab: continue
      if tok.special.len == tok.vocab and tok.special[id]: continue
      clean.add(id)
  else:
    clean = ids
  if clean.len == 0: return ""
  case tok.backend
  of "mistral_common":
    result = tok.pyTok.instruct_tokenizer.tokenizer.decode(clean).to(string)
  of "hf_tokenizers":
    result = tok.pyTok.decode(clean, skip_special_tokens = skipSpecial).to(string)
  else:
    raise newException(ValueError, "decodeHf: backend khong hop le: " & tok.backend)

proc vocabSizeHf*(tok: HfTokenizer): int = tok.vocab

proc isStopIdHf*(tok: HfTokenizer, id: int): bool =
  for s in tok.stopIds:
    if s == id: return true
  return false
