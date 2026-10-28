#!/usr/bin/env bash
# finetunemamba_codestral.sh
# Ban sao cua finetunemamba.sh, chinh cho Mamba-Codestral-7B-v0.1:
#   - HF_DIR la 1 THU MUC (sharded safetensors), khong phai 1 file .bin
#   - DA BO buoc graft Attention (theo yeu cau): dung THANG pattern toan 'R'
#     (Mamba2Real, CFG_CODESTRAL_PATTERN hard-code trong finetune_lora.nim -
#     block 'A' Attention van duoc dung nhung la random-init NGAY TRONG
#     finetune_lora.nim/load_codestral_into_jamba.nim, KHONG can "cay" san
#     vao file .nimq nua qua graft_attention.nim).
#   - KHONG con diem 'die' thiet ke cu doi convert .nimq -> .bin thu cong:
#     da xac nhan finetune_lora.nim (CFG_HF_SOURCE mac dinh "nimq") doc
#     THANG file .nimq qua loadCodestralQuantResidentFromNimq trong
#     load_codestral_into_jamba.nim - ham nay TU dequantize + map ten tensor
#     HF Mamba2 -> block 'R', khong can .bin trung gian nao ca.
#   - can bien dich them sharded_hf_loader.nim (xem file di kem) vi loader
#     goc chi doc 1 file
#
# Chay tu project root sau khi tai tokenizer/config vao
# test/codestral/models/hf va dat file NimQ trong test/codestral/models/.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

HF_DIR="${HF_DIR:-$SCRIPT_DIR/models/hf}"
# QUAN TRONG: ten file nay se duoc TRUYEN THANG lam nimqPath cho
# finetune_lora_bin/infer_lora_bin o duoi - KHONG con phu thuoc vao
# CFG_NIMQ_PATH_DEFAULT ("codestral_mamba_7b_fp4.nimq") hard-code trong 2
# file .nim do nua, nen doi ten o day thoai mai, khong so lech.
OLD_MAMBA_NIMQ="${OLD_MAMBA_NIMQ:-$SCRIPT_DIR/models/codestral_mamba_only.nimq}"   # 64 layer toan 'R'
GRAFT_QUANT="${GRAFT_QUANT:-int8}"   # int8|int4|fp4|fp8 cho quantize_codestral_bin
                                     # --graft-quant - GIU LAI cho tuong thich CLI,
                                     # nhung THUC TE khong con anh huong gi vi buoc
                                     # nay chi doc tensor HF goc (khong co tensor
                                     # "graft moi" nao de ap dung quant nay).
LORA_CHECKPOINT="${LORA_CHECKPOINT:-$SCRIPT_DIR/models/jamba_parallel_lora.bin}"
# SUA: template "[INST] ...[/INST] " dung cho Codestral (Mistral tokenizer V3) -
# KHONG phai ChatML "<|user|>/<|assistant|>" (xem finetune_lora.nim/infer_lora.nim).
INFER_PROMPT="${INFER_PROMPT:-[INST] Xin chao[/INST] }"
INFER_TEMPERATURE="${INFER_TEMPERATURE:-0.7}"
# SKIP_TO_FINETUNE=1 (hoac tu dong neu $OLD_MAMBA_NIMQ da ton tai san) -> bo
# qua buoc quantize, di thang vao finetune_lora (dung khi ban DA CO san .nimq
# tu lan chay truoc, khong muon quantize lai).
SKIP_TO_FINETUNE="${SKIP_TO_FINETUNE:-0}"

log() { echo -e "\n\033[1;36m[finetunemamba_codestral]\033[0m $*"; }
die() { echo -e "\033[1;31m[finetunemamba_codestral] LOI:\033[0m $*" >&2; exit 1; }

command -v nim >/dev/null 2>&1 || die "Khong tim thay 'nim' trong PATH."
[[ -d "$HF_DIR" ]] || die "Khong thay thu muc $HF_DIR. Chay ./test/codestral/download_codestral.sh truoc."

# ── Da co .nimq san -> nhay thang vao finetune ─────────────────────────
if [[ "$SKIP_TO_FINETUNE" == "1" || -f "$OLD_MAMBA_NIMQ" ]]; then
  [[ -f "$OLD_MAMBA_NIMQ" ]] || die "SKIP_TO_FINETUNE=1 nhung khong thay $OLD_MAMBA_NIMQ."
  log "Da phat hien $OLD_MAMBA_NIMQ - bo qua quantize, di thang vao finetune."
else
  log "Bien dich quantize_codestral.nim ..."
  nim c --path:. --path:test/codestral --path:test/spectron --path:kernelfortest -d:release --hints:off -o:test/codestral/quantize_codestral_bin test/codestral/quantize_codestral.nim
  log "Quantize $HF_DIR -> $OLD_MAMBA_NIMQ (main-quant mac dinh fp4km, graft=$GRAFT_QUANT) ..."
  ./test/codestral/quantize_codestral_bin "$HF_DIR" "$OLD_MAMBA_NIMQ" --graft-quant "$GRAFT_QUANT"
fi

log "Bien dich finetune_lora.nim ..."
nim c --path:. --path:test/codestral --path:test/spectron --path:kernelfortest -d:release --hints:off -o:test/codestral/finetune_lora_bin test/codestral/finetune_lora.nim
# argv: initMode oldCheckpoint hfDir hfSource nimqPath outCheckpointPath
# ("-" = giu mac dinh cua finetune_lora.nim cho tham so do, xem argOrDefault).
# TRUYEN THANG $OLD_MAMBA_NIMQ o day de finetune_lora_bin KHONG vo tinh doc
# nham file mac dinh cua no (codestral_mamba_7b_fp4.nimq, ten khac $OLD_MAMBA_NIMQ).
log "Finetune LoRA (codestral, doc thang tu $OLD_MAMBA_NIMQ, khong graft Attention) ..."
# TRUYEN THANG "$HF_DIR" (KHONG dung "-") - finetune_lora.nim VAN load
# tokenizer THAT tu day (newHfTokenizer(CFG_HF_DIR)) du trong so doc tu .nimq,
# va CFG_HF_DIR_DEFAULT hard-code trong finetune_lora.nim co the khac $HF_DIR
# neu ban da doi bien moi truong nay.
./test/codestral/finetune_lora_bin codestral - "$HF_DIR" nimq "$OLD_MAMBA_NIMQ" "$LORA_CHECKPOINT"

[[ -f "$LORA_CHECKPOINT" ]] || die "Khong thay $LORA_CHECKPOINT"
log "Bien dich infer_lora.nim ..."
nim c --path:. --path:test/codestral --path:test/spectron --path:kernelfortest -d:release --hints:off -o:test/codestral/infer_lora_bin test/codestral/infer_lora.nim
# argv: ckptPath prompt temperature hfDir nimqPath - truyen nimqPath khop
# $OLD_MAMBA_NIMQ phong khi roi vao nhanh fallback Mamba thuan (xem infer_lora.nim).
# tuong tu: TRUYEN THANG "$HF_DIR" - infer_lora.nim (nhanh fallback Mamba
# thuan) cung load tokenizer that qua newHfTokenizer(hfDir), CFG_HF_DIR_DEFAULT
# rieng cua infer_lora.nim con khac CA finetune_lora.nim ("Data 1" vs "Data 2").
./test/codestral/infer_lora_bin "$LORA_CHECKPOINT" "$INFER_PROMPT" "$INFER_TEMPERATURE" "$HF_DIR" "$OLD_MAMBA_NIMQ"

log "XONG."
