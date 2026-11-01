## graft_attention.nim
## Grafts NEW LinearPlusAttention/Spectron blocks into an existing Jamba
## checkpoint (usually an all-Mamba2SSD checkpoint, a pattern like
## "MMMM..."), keeping the trained Mamba weights and only inserting
## Attention layers at the positions you choose.
##
## IMPORTANT: the REAL `graftBlocks` lives in `lora.nim` (it takes `ctx: Backend`
## to choose the GPU EXPLICITLY, even though copying weights here is only
## pointer/array assignment - no matrix multiply - ctx is kept DELIBERATELY
## so graftBlocks() always comes with a Backend confirmed to be a GPU, preventing
## a call without ctx from accidentally creating a code path that "forgets" the GPU
## later). This file is ONLY the CLI entry point and no longer defines its own graftBlocks
## (there used to be a duplicate without ctx here - REMOVED, that was
## a bug: 2 functions with the same name and different signatures, easy to call the one not forcing the GPU).

import std/[strformat, strutils, os]
import nimformer
import backend
import jamba_hybrid   # NimformerModelJamba, newNimformerModelJamba, loadMambaOnlyFromHFBin
import lora           # graftBlocks*(old, newPattern, ctx: Backend) - BẢN DUY NHẤT, xem lora.nim

when isMainModule:
  # usage (1) - graft from an existing Jamba .bin checkpoint (written by jamba_hybrid
  # in an earlier run):
  #   ./graft_attention <old.bin> <newPattern vd MMAMMSM> <out.bin>
  #
  # usage (2) - graft DIRECTLY from the ORIGINAL HF pytorch_model.bin, NO
  # export/intermediate file step first (avoids doubling the disk usage
  # of HF checkpoints that are usually several GB):
  #   ./graft_attention --from-hf <hf.bin> <vocab> <embedDim> <nHeads> <ffMult> \
  #       <nLayers> <newPattern vd MMAMMSM> <out.bin>
  #   (newPattern must have exactly <nLayers> 'M' characters, with 'A'/'S' inserted for new blocks)
  #
  # ctx = newBackend() below defaults to "auto": if the machine has NO GPU (CUDA/
  # Metal/OpenCL/Vulkan), backend.nim RAISES right here (gForbidCpuFallback
  # = true by default) instead of silently running on the CPU - "must be 100% GPU".
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
    echo &"[graft] Reading DIRECTLY from {hfBinPath} (no intermediate file) ..."
    var (old, matched, total) = loadMambaOnlyFromHFBin(hfBinPath, vocab, embedDim, nHeads, ffMult, nLayers)
    echo &"[graft] Loaded {matched}/{total} parameters from HF, grafting '{newPattern}' ..."
    var grafted = graftBlocks(old, newPattern, ctx)
    jamba_hybrid.saveCheckpoint(grafted, outPath, 0)
    echo &"[graft] Saved the grafted model -> {outPath}"
  else:
    if args.len < 3:
      echo "usage: graft_attention <old.bin> <newPattern vd MMAMMSM> <out.bin>"
      echo "   or: graft_attention --from-hf <hf.bin> <vocab> <embedDim> <nHeads> " &
           "<ffMult> <nLayers> <newPattern> <out.bin>"
      quit(1)
    if args[0].toLowerAscii().endsWith(".nimq"):
      echo &"[graft] ERROR: '{args[0]}' has the .nimq extension (the quantized format of " &
           "quant.nim), NOT the raw .bin (rawbin.nim) graft_attention " &
           "needs. By project convention .nimq is ONLY for inference (nim_inference.nim, " &
           "test_nimformer.nim), not for the graft/finetune/infer-lora pipeline."
      echo "[graft] You may want to finetune directly from this .nimq file - " &
           "finetune_lora.nim reads .nimq directly through CFG_HF_SOURCE=\"nimq\" " &
           "(the default), no graft needed first. Run:"
      echo &"[graft]   make finetune-mamba"
      echo &"[graft]   (or: make finetune-mamba ARGS=\"codestral - - nimq {args[0]} <out.bin>\")"
      quit(1)
    var (old, oldStep) = jamba_hybrid.loadCheckpoint(args[0])
    echo &"[graft] Old model: pattern='{old.pattern}', {old.blocks.len} blocks, trained step={oldStep}"
    var grafted = graftBlocks(old, args[1], ctx)
    jamba_hybrid.saveCheckpoint(grafted, args[2], oldStep)
    echo &"[graft] Saved the grafted model -> {args[2]}"