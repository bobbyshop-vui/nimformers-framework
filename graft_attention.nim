## graft_attention.nim
## Cấy (graft) thêm block LinearPlusAttention/Spectron MỚI vào 1 checkpoint
## Jamba hiện có (thường là checkpoint chỉ toàn Mamba2SSD, pattern kiểu
## "MMMM..."), giữ nguyên trọng số Mamba đã train, chỉ chèn thêm layer
## Attention xen vào theo vị trí bạn muốn.
##
## QUAN TRỌNG: hàm `graftBlocks` THẬT nằm ở `lora.nim` (nhận `ctx: Backend`
## để CHỌN GPU TƯỜNG MINH, dù bản thân việc copy trọng số ở đây chỉ là gán
## con trỏ/mảng - không có phép nhân ma trận nào - ctx được giữ lại CHỦ ĐÍCH
## để graftBlocks() luôn đi kèm 1 Backend đã xác nhận GPU, tránh trường hợp
## ai đó gọi hàm không kèm ctx rồi vô tình có 1 code path "quên" GPU sau
## này). File này CHỈ là entrypoint CLI, không định nghĩa graftBlocks riêng
## nữa (trước đây có 1 bản trùng lặp không dùng ctx ở đây - ĐÃ XÓA, đó là
## bug: 2 hàm cùng tên khác chữ ký, dễ gọi nhầm bản không ép GPU).

import std/[strformat, strutils, os]
import nimformer
import backend
import jamba_hybrid   # NimformerModelJamba, newNimformerModelJamba, loadMambaOnlyFromHFBin
import lora           # graftBlocks*(old, newPattern, ctx: Backend) - BẢN DUY NHẤT, xem lora.nim

when isMainModule:
  # cach dung (1) - graft tu 1 checkpoint Jamba .bin da co san (do jamba_hybrid
  # xuat ra tu 1 lan chay truoc):
  #   ./graft_attention <old.bin> <newPattern vd MMAMMSM> <out.bin>
  #
  # cach dung (2) - graft TRUC TIEP tu pytorch_model.bin GOC cua HF, KHONG can
  # buoc export/xuat file trung gian nao truoc (tranh nhan doi dung luong dia
  # voi checkpoint HF thuong nang vai GB):
  #   ./graft_attention --from-hf <hf.bin> <vocab> <embedDim> <nHeads> <ffMult> \
  #       <nLayers> <newPattern vd MMAMMSM> <out.bin>
  #   (newPattern phai co dung <nLayers> ky tu 'M', xen them 'A'/'S' cho block moi)
  #
  # ctx = newBackend() o duoi mac dinh "auto": neu may KHONG co GPU (CUDA/
  # Metal/OpenCL/TSIC), backend.nim se RAISE loi ngay o day (gForbidCpuFallback
  # = true mac dinh) thay vi am tham chay CPU - dung y "phai 100% GPU".
  let args = commandLineParams()
  let ctx = newBackend()

  if args.len >= 1 and args[0] == "--from-hf":
    if args.len < 9:
      echo "usage: graft_attention --from-hf <hf.bin> <vocab> <embedDim> <nHeads> " &
           "<ffMult> <nLayers> <newPattern> <out.bin>"
      quit(1)
    let hfBinPath = args[1]
    let vocab = parseInt(args[2])
    let embedDim = parseInt(args[3])
    let nHeads = parseInt(args[4])
    let ffMult = parseInt(args[5])
    let nLayers = parseInt(args[6])
    let newPattern = args[7]
    let outPath = args[8]
    echo &"[graft] Doc TRUC TIEP tu {hfBinPath} (khong ghi file trung gian) ..."
    var (old, matched, total) = loadMambaOnlyFromHFBin(hfBinPath, vocab, embedDim, nHeads, ffMult, nLayers)
    echo &"[graft] Da nap {matched}/{total} tham so tu HF, cay '{newPattern}' ..."
    var grafted = graftBlocks(old, newPattern, ctx)
    jamba_hybrid.saveCheckpoint(grafted, outPath, 0)
    echo &"[graft] Da luu model da cay ghep -> {outPath}"
  else:
    if args.len < 3:
      echo "usage: graft_attention <old.bin> <newPattern vd MMAMMSM> <out.bin>"
      echo "   or: graft_attention --from-hf <hf.bin> <vocab> <embedDim> <nHeads> " &
           "<ffMult> <nLayers> <newPattern> <out.bin>"
      quit(1)
    if args[0].toLowerAscii().endsWith(".nimq"):
      echo &"[graft] LOI: '{args[0]}' co duoi .nimq (dinh dang quantize cua " &
           "quant.nim), KHONG phai .bin tho (rawbin.nim) ma graft_attention " &
           "can. Theo quy uoc du an, .nimq CHI dung cho inference (nim_inference.nim, " &
           "test_nimformer.nim), khong dung trong pipeline graft/finetune/infer-lora."
      echo "[graft] Ban co the dang muon finetune truc tiep tu file .nimq nay - " &
           "finetune_lora.nim doc duoc .nimq thang qua CFG_HF_SOURCE=\"nimq\" " &
           "(mac dinh), khong can graft truoc. Chay:"
      echo &"[graft]   make finetune-mamba"
      echo &"[graft]   (hoac: make finetune-mamba ARGS=\"codestral - - nimq {args[0]} <out.bin>\")"
      quit(1)
    var (old, oldStep) = jamba_hybrid.loadCheckpoint(args[0])
    echo &"[graft] Model cu: pattern='{old.pattern}', {old.blocks.len} block, step da train={oldStep}"
    var grafted = graftBlocks(old, args[1], ctx)
    jamba_hybrid.saveCheckpoint(grafted, args[2], oldStep)
    echo &"[graft] Da luu model da cay ghep -> {args[2]}"