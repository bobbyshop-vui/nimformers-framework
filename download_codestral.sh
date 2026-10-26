#!/usr/bin/env bash
# download_codestral.sh
# Tai mistralai/Mamba-Codestral-7B-v0.1 (an toan tenant) vao o ngoai
# /Volumes/Data 1, KHONG dong cham gi den project folder tren o he thong.
#
# Model nay o dang safetensors CHIA SHARD (model-0000X-of-0000Y.safetensors
# + model.safetensors.index.json) - KHONG phai 1 file .bin duy nhat nhu
# Mamba-Chat-2.8B cu, nen pytorch_bin_loader.nim ban goc (loadAllHFTensors
# nhan 1 path file) KHONG doc thang duoc - xem sharded_hf_loader.nim di kem.
#
# YEU CAU: `pip install -U "huggingface_hub[cli]"` truoc khi chay.

set -euo pipefail

DEST="${1:-/Volumes/Data 2/models/Mamba-Codestral-7B-v0.1}"
REPO="mistralai/Mamba-Codestral-7B-v0.1"

echo "[download_codestral] Tai $REPO -> $DEST"
mkdir -p "$DEST"

huggingface-cli download "$REPO" \
  --local-dir "$DEST" \
  --exclude "*.pt" "consolidated*" \
  # giu lai: *.safetensors, *.json, tokenizer*, params.json

echo "[download_codestral] XONG. Kiem tra shard:"
ls -lh "$DEST" | grep -i safetensors || true

if [[ ! -f "$DEST/model.safetensors.index.json" ]]; then
  echo "[download_codestral] CANH BAO: khong thay model.safetensors.index.json -" \
       "co the model chi co 1 shard duy nhat (model.safetensors) - van OK," \
       "sharded_hf_loader.nim tu xu ly ca 2 truong hop."
fi

echo
echo "Buoc tiep theo (chay tren may ban, KHONG chay trong sandbox nay):"
echo "  1. Copy sharded_hf_loader.nim + quantize_codestral.nim vao project folder"
echo "  2. nim c -d:release --hints:off -o:quantize_codestral_bin quantize_codestral.nim"
echo "  3. ./quantize_codestral_bin \"$DEST\" ./codestral_mamba_7b_fp4.nimq"