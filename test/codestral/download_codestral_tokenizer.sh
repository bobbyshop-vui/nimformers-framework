#!/usr/bin/env bash
# Fetch only tokenizer/config assets, preserving existing base and adapter weights.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/models/hf}"
REPO="${CODESTRAL_TOKENIZER_REPO:-bobbydeveloper2014/codestral-v0.1-7b-nimq}"
REVISION="${CODESTRAL_TOKENIZER_REVISION:-148dd26885b5eeef03476683f05f0a4546cd75ba}"
mkdir -p "$DEST"
for file in tokenizer.json tokenizer.model tokenizer.model.v3 tokenizer_config.json special_tokens_map.json config.json params.json generation_config.json; do
  echo "[tokenizer] $file"
  curl -fL --retry 3 --connect-timeout 15 \
    "https://huggingface.co/$REPO/resolve/$REVISION/$file" -o "$DEST/$file.part"
  mv "$DEST/$file.part" "$DEST/$file"
done
echo "[tokenizer] Ready: $DEST"
