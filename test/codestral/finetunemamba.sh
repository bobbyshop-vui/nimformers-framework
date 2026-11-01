#!/usr/bin/env bash
# finetunemamba_codestral.sh
# A copy of finetunemamba.sh adapted for Mamba-Codestral-7B-v0.1:
#   - HF_DIR is a DIRECTORY (sharded safetensors), not a single .bin file
#   - the Attention graft step was REMOVED (on request): uses the all-'R' pattern DIRECTLY
#     (Mamba2Real, CFG_CODESTRAL_PATTERN hard-coded in finetune_lora.nim -
#     'A' Attention blocks are still used but random-initialised RIGHT IN
#     finetune_lora.nim/load_codestral_into_jamba.nim, NO need to "graft" them
#     into the .nimq file beforehand through graft_attention.nim).
#   - NO more 'die' point of the old design waiting for a manual .nimq -> .bin conversion:
#     finetune_lora.nim (CFG_HF_SOURCE defaults to "nimq") is confirmed to read
#     the .nimq file DIRECTLY through loadCodestralQuantResidentFromNimq in
#     load_codestral_into_jamba.nim - that function dequantizes + maps the HF Mamba2
#     tensor names -> 'R' blocks itself, no intermediate .bin needed.
#   - sharded_hf_loader.nim (see the accompanying file) must also be compiled because the
#     original loader reads only one file
#
# Run from the project root after downloading the tokenizer/config into
# test/codestral/models/hf and placing the NimQ file in test/codestral/models/.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

HF_DIR="${HF_DIR:-$SCRIPT_DIR/models/hf}"
# IMPORTANT: this file name is PASSED DIRECTLY as nimqPath to
# finetune_lora_bin/infer_lora_bin below - it NO longer depends on
# the CFG_NIMQ_PATH_DEFAULT ("codestral_mamba_7b_fp4.nimq") hard-coded in those 2
# .nim files, so rename it here freely without mismatches.
OLD_MAMBA_NIMQ="${OLD_MAMBA_NIMQ:-$SCRIPT_DIR/models/codestral_mamba_only.nimq}"   # 64 layer toan 'R'
GRAFT_QUANT="${GRAFT_QUANT:-int8}"   # int8|int4|fp4|fp8 for quantize_codestral_bin
                                     # --graft-quant - KEPT for CLI compatibility,
                                     # but it has NO effect any more because this step
                                     # only reads the original HF tensors (there is no
                                     # "newly grafted" tensor to quantize).
LORA_CHECKPOINT="${LORA_CHECKPOINT:-$SCRIPT_DIR/models/jamba_parallel_lora.bin}"
# CHANGED: the "[INST] ...[/INST] " template is for Codestral (Mistral tokenizer V3) -
# NOT ChatML "<|user|>/<|assistant|>" (see finetune_lora.nim/infer_lora.nim).
INFER_PROMPT="${INFER_PROMPT:-[INST] Xin chao[/INST] }"
INFER_TEMPERATURE="${INFER_TEMPERATURE:-0.7}"
# SKIP_TO_FINETUNE=1 (or automatically when $OLD_MAMBA_NIMQ already exists) -> skips
# the quantize step and goes straight to finetune_lora (when you ALREADY have a .nimq
# from an earlier run and do not want to quantize again).
SKIP_TO_FINETUNE="${SKIP_TO_FINETUNE:-0}"

log() { echo -e "\n\033[1;36m[finetunemamba_codestral]\033[0m $*"; }
die() { echo -e "\033[1;31m[finetunemamba_codestral] ERROR:\033[0m $*" >&2; exit 1; }

command -v nim >/dev/null 2>&1 || die "'nim' not found in PATH."
[[ -d "$HF_DIR" ]] || die "Directory $HF_DIR not found. Run ./test/codestral/download_codestral.sh first."

# ── .nimq already present -> jump straight to finetune ─────────────────────────
if [[ "$SKIP_TO_FINETUNE" == "1" || -f "$OLD_MAMBA_NIMQ" ]]; then
  [[ -f "$OLD_MAMBA_NIMQ" ]] || die "SKIP_TO_FINETUNE=1 but $OLD_MAMBA_NIMQ was not found."
  log "Found $OLD_MAMBA_NIMQ - skipping quantize, going straight to finetune."
else
  log "Bien dich quantize_codestral.nim ..."
  nim c --path:. --path:test/codestral --path:test/spectron --path:kernelfortest -d:release --hints:off -o:test/codestral/quantize_codestral_bin test/codestral/quantize_codestral.nim
  log "Quantize $HF_DIR -> $OLD_MAMBA_NIMQ (main-quant mac dinh fp4km, graft=$GRAFT_QUANT) ..."
  ./test/codestral/quantize_codestral_bin "$HF_DIR" "$OLD_MAMBA_NIMQ" --graft-quant "$GRAFT_QUANT"
fi

log "Bien dich finetune_lora.nim ..."
nim c --path:. --path:test/codestral --path:test/spectron --path:kernelfortest -d:release --hints:off -o:test/codestral/finetune_lora_bin test/codestral/finetune_lora.nim
# argv: initMode oldCheckpoint hfDir hfSource nimqPath outCheckpointPath
# ("-" = keep finetune_lora.nim's default for that parameter, see argOrDefault).
# $OLD_MAMBA_NIMQ is PASSED DIRECTLY here so finetune_lora_bin does not accidentally read
# its own default file (codestral_mamba_7b_fp4.nimq, a different name than $OLD_MAMBA_NIMQ).
log "Finetune LoRA (codestral, reading $OLD_MAMBA_NIMQ directly, no Attention graft) ..."
# "$HF_DIR" is PASSED DIRECTLY (NOT "-") - finetune_lora.nim STILL loads
# the REAL tokenizer from it (newHfTokenizer(CFG_HF_DIR)) even though the weights come from the .nimq,
# and the CFG_HF_DIR_DEFAULT hard-coded in finetune_lora.nim may differ from $HF_DIR
# if you changed that environment variable.
./test/codestral/finetune_lora_bin codestral - "$HF_DIR" nimq "$OLD_MAMBA_NIMQ" "$LORA_CHECKPOINT"

[[ -f "$LORA_CHECKPOINT" ]] || die "$LORA_CHECKPOINT not found"
log "Bien dich infer_lora.nim ..."
nim c --path:. --path:test/codestral --path:test/spectron --path:kernelfortest -d:release --hints:off -o:test/codestral/infer_lora_bin test/codestral/infer_lora.nim
# argv: ckptPath prompt temperature hfDir nimqPath - nimqPath matches
# $OLD_MAMBA_NIMQ in case it falls into the pure-Mamba fallback branch (see infer_lora.nim).
# likewise: "$HF_DIR" is PASSED DIRECTLY - infer_lora.nim (the pure-Mamba fallback
# branch) also loads the real tokenizer through newHfTokenizer(hfDir), and infer_lora.nim's own
# CFG_HF_DIR_DEFAULT differs even from finetune_lora.nim's ("Data 1" vs "Data 2").
./test/codestral/infer_lora_bin "$LORA_CHECKPOINT" "$INFER_PROMPT" "$INFER_TEMPERATURE" "$HF_DIR" "$OLD_MAMBA_NIMQ"

log "DONE."
