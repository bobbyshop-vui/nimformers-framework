#!/usr/bin/env bash
# download_codestral.sh
# Downloads mistralai/Mamba-Codestral-7B-v0.1 into test/codestral/models/hf (or the directory
# given as argv 1) in the project.
#
# This model is SHARDED safetensors (model-0000X-of-0000Y.safetensors
# + model.safetensors.index.json) - NOT a single .bin file like
# the old Mamba-Chat-2.8B, so the original pytorch_bin_loader.nim (loadAllHFTensors
# takes one file path) CANNOT read it directly - see the accompanying sharded_hf_loader.nim.
#
# REQUIRES: `pip install -U "huggingface_hub[cli]"` before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/models/hf}"
REPO="mistralai/Mamba-Codestral-7B-v0.1"

echo "[download_codestral] Tai $REPO -> $DEST"
mkdir -p "$DEST"

huggingface-cli download "$REPO" \
  --local-dir "$DEST" \
  --exclude "*.pt" "consolidated*" \
  # keep: *.safetensors, *.json, tokenizer*, params.json

echo "[download_codestral] DONE. Checking shards:"
ls -lh "$DEST" | grep -i safetensors || true

if [[ ! -f "$DEST/model.safetensors.index.json" ]]; then
  echo "[download_codestral] WARNING: model.safetensors.index.json not found -" \
       "the model may have a single shard only (model.safetensors) - still OK," \
       "sharded_hf_loader.nim handles both cases."
fi

echo
echo "Next step (run from the project root):"
echo "  nim c --path:. --path:test/codestral --path:test/spectron --path:kernelfortest -d:release --hints:off -o:test/codestral/quantize_codestral_bin test/codestral/quantize_codestral.nim"
echo "  ./test/codestral/quantize_codestral_bin \"$DEST\" test/codestral/models/codestral_mamba_only.nimq"
