#!/usr/bin/env bash
# download_codestral.sh
# Tai mistralai/Mamba-Codestral-7B-v0.1 vao test/codestral/models/hf (hoac thu muc
# truyen vao argv 1) trong project.
#
# Model nay o dang safetensors CHIA SHARD (model-0000X-of-0000Y.safetensors
# + model.safetensors.index.json) - KHONG phai 1 file .bin duy nhat nhu
# Mamba-Chat-2.8B cu, nen pytorch_bin_loader.nim ban goc (loadAllHFTensors
# nhan 1 path file) KHONG doc thang duoc - xem sharded_hf_loader.nim di kem.
#
# YEU CAU: `pip install -U "huggingface_hub[cli]"` truoc khi chay.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/models/hf}"
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
echo "Buoc tiep theo (chay tu project root):"
echo "  nim c --path:. --path:test/codestral --path:test/spectron --path:kernelfortest -d:release --hints:off -o:test/codestral/quantize_codestral_bin test/codestral/quantize_codestral.nim"
echo "  ./test/codestral/quantize_codestral_bin \"$DEST\" test/codestral/models/codestral_mamba_only.nimq"
