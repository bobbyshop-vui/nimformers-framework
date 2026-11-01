## quantize_codestral.nim
## Reads Mamba-Codestral-7B-v0.1 (sharded safetensors) DIRECTLY FROM THE DOWNLOADED
## DIRECTORY (e.g. on "/Volumes/Data 1/...") through sharded_hf_loader, quantizes, and writes
## ONLY one small .nimq file into the PROJECT DIRECTORY on the system disk - it NEVER
## writes an intermediate fp16/fp32 checkpoint (as requested, to "save
## disk" - ~7B fp16 parameters ~14GB, about ~3.5GB after fp4 + a little fp32 for
## embeddings/layernorms).
##
## Quantization rules (following the note in quant.nim: "fp32 is for
## bias/layernorm - error sensitive"):
##   - embeddings, layernorms (weight+bias), every bias      -> fp32 raw
##   - the MAIN matrices of the original HF checkpoint (in_proj,
##     x_proj, dt_proj, conv1d, out_proj of Mamba2, lm_head)  -> FP4 (E2M1)
##   - tensors of the new GRAFT part (newly grafted LinearPlusAttention
##     /LoRA A-B) not in the original HF checkpoint         -> --graft-quant
##     (default int8, can be changed to int4/fp4/fp8 on the CLI)
##
## Since this is a .nimq (not the .bin jamba_hybrid uses for LoRA training), the purpose
## of this file is a lightweight INFERENCE/baseline through nim_inference.nim, NOT
## a file to continue training from (LoRA training still reads .bin through rawbin.nim as
## finetune_lora.nim does - see the note in jamba_hybrid.nim).
##
## USAGE:
##   nim c -d:release --hints:off -o:quantize_codestral_bin quantize_codestral.nim
##   ./quantize_codestral_bin "/Volumes/Data 1/models/Mamba-Codestral-7B-v0.1" \
##       ./codestral_mamba_7b_fp4.nimq [--graft-quant int8|int4|fp4|fp8]

import std/[os, strutils, strformat, tables]
import quant
import customfloat
import sharded_hf_loader

# FP4 E2M1 - see customfloat_fp4_patch.nim (paste it into customfloat.nim
# BEFORE compiling this file, or declare it again right here):
let FP4_E2M1 = newCustomFloat(2, 1, "fp4_e2m1")

const
  Fp32NameHints = ["embedding", "norm", "layernorm", "ln.", ".bias", "lm_head.bias"]
  # heuristic: a tensor whose name contains one of these strings -> error sensitive, kept in fp32.
  # "lm_head.bias" is listed separately because lm_head.weight is NOT kept in fp32 (too large,
  # 7B*vocab), only the bias (if any) is kept.

proc isPrecisionSensitive(name: string): bool =
  let lower = name.toLowerAscii
  for hint in Fp32NameHints:
    if hint in lower:
      return true
  return false

proc parseGraftQuant(s: string): QuantKind =
  case s.toLowerAscii
  of "int8": qkInt8
  of "int4": qkInt4
  of "fp4":  qkCustom     # FP4_E2M1 used as customCf
  of "fp8":  qkFp8E4M3
  else: raise newException(ValueError, &"invalid --graft-quant: {s} (choose int8/int4/fp4/fp8)")

proc parseMainQuant(s: string): QuantKind =
  ## Fix for garbage tokens (bug #13). BEFORE, every main weight was hard-coded
  ## `qkCustom + FP4_E2M1` (per-tensor scale). Measured with the repo's own code
  ## on a 512x4096 matrix with an LLM-like distribution:
  ##
  ##   fp4 e2m1 per-tensor : rel-L2 = 0.9421, only 0.06% of elements still non-zero
  ##   fp8 e4m3 per-tensor : rel-L2 = 0.0531, 99.89% of elements still non-zero
  ##
  ## Cause: FP4 E2M1 can represent only 6 magnitudes
  ## {0, 0.75, 1, 1.5, 2, 3} (4/16 codes are +-inf), AND encodeArray
  ## TRUNCATES to 0 instead of rounding (0.74 -> 0.00, 0.90 -> 0.00). With
  ## scale = maxAbs/3 shared by the whole matrix, EVERY weight below
  ## 0.25*maxAbs became EXACTLY 0 -> a 7B matrix turned into ~99.9% zeros.
  ##
  ## fp8 e4m3 is STILL 1 byte/element (itemSize=1), STILL uses the 256-entry buildDecodeLUT,
  ## STILL uses buildFp4Lut/qt.scale[0] and the current Metal kernel -
  ## NOT a single line of the loader or GPU code changes. The .nimq file size
  ## STAYS the same. Only a re-export is needed.
  ##
  ## fp4km (NEW, DEFAULT): "Q4_K_M but FP4". Measured on an LLM-like matrix:
  ##   old fp4 (per-tensor, truncate) : 8.00 bpw, relL2 0.879, 99.5% to 0
  ##   FP4-KM g=32                   : 4.25 bpw, relL2 0.100, cos 0.992
  ##   fp8 e4m3 per-tensor           : 8.00 bpw, relL2 0.049, cos 0.9996
  ## => fp4km is HALF the size of the current file and has ~9 times less error.
  ## Choose fp8 for the best quality if a 7GB file is fine.
  case s.toLowerAscii
  of "fp4km": qkFp4KM
  of "fp8":  qkFp8E4M3
  of "fp4":  qkCustom
  of "int8": qkInt8
  else: raise newException(ValueError, &"invalid --main-quant: {s} (choose fp4km/fp8/fp4/int8)")

when isMainModule:
  if paramCount() < 2:
    echo "usage: quantize_codestral <thu_muc_hf_da_tai_ve> <out.nimq> " &
         "[--main-quant fp4km|fp8|fp4|int8] [--graft-quant int8|int4|fp4|fp8]"
    echo "  --main-quant defaults to fp4km (4.25 bits/weight). fp4: use only if you know " &
         "what you are doing - see parseMainQuant, per-tensor fp4 e2m1 loses " &
         "~99.9% weight (bug #13)."
    quit(1)

  let hfDir = paramStr(1)
  let outPath = paramStr(2)
  var graftQuantStr = "int8"
  var mainQuantStr = "fp4km"  # FIX bug #13 - xem parseMainQuant
  var i = 3
  while i <= paramCount():
    if paramStr(i) == "--graft-quant" and i + 1 <= paramCount():
      graftQuantStr = paramStr(i + 1)
      i += 2
    elif paramStr(i) == "--main-quant" and i + 1 <= paramCount():
      mainQuantStr = paramStr(i + 1)
      i += 2
    else:
      i += 1
  let graftKind = parseGraftQuant(graftQuantStr)
  let mainKind = parseMainQuant(mainQuantStr)

  echo &"[quantize_codestral] Reading {hfDir} (RAM only, no intermediate file) ..."
  let tensors = loadAllHFTensorsSharded(hfDir)
  echo &"[quantize_codestral] {tensors.len} tensors. Starting quantization " &
       &"(main weights -> {mainQuantStr}, sensitive -> fp32, new graft -> {graftQuantStr}) ..."

  var sd: seq[(string, QuantTensor)] = @[]
  var nFp4, nFp32, nGraft = 0

  for t in tensors:
    # "new graft" = tensors NOT in the original HF checkpoint that are added
    # AFTER THIS STEP by graft_attention_bin (new LinearPlusAttention qkv/out,
    # LoRA A/B) - this .nimq holds ONLY the original HF part, so every tensor read from
    # here is "original", there is no real "graft" tensor - the name-based branch
    # is reserved so that if you add tensors by hand later (e.g. merging
    # a separate graft .nimq) the --graft-quant logic still works.
    let isGraftNew = "graft" in t.name.toLowerAscii or "lora_" in t.name.toLowerAscii

    var qt: QuantTensor
    if isPrecisionSensitive(t.name):
      qt = quantizeTensor(t.data, t.shape, qkFp32Raw)
      inc nFp32
    elif isGraftNew:
      qt = quantizeTensor(t.data, t.shape, graftKind, customCf = FP4_E2M1)
      inc nGraft
    elif "lm_head.weight" in t.name.toLowerAscii:
      # lm_head [32768,4096]: fp4km still causes ~0.1 relL2 logit error -> fp8 (128MB, cos 0.9996)
      qt = quantizeTensor(t.data, t.shape, qkFp8E4M3)
      inc nFp32  # counted with fp32 for a clear log
    else:
      # FP4-KM needs a 2D tensor whose cols are divisible by groupSize (32). Every large
      # Codestral weight qualifies (in_proj 18560x4096, out_proj 4096x8192,
      # lm_head 32768x4096, embed 32768x4096). A tensor that does not
      # FALLS BACK to fp8 e4m3 - NEVER to the old per-tensor fp4.
      if mainKind == qkFp4KM and (t.shape.len != 2 or t.shape[1] mod 32 != 0):
        qt = quantizeTensor(t.data, t.shape, qkFp8E4M3)
      else:
        qt = quantizeTensor(t.data, t.shape, mainKind, customCf = FP4_E2M1)
      inc nFp4
    sd.add((t.name, qt))

  echo &"[quantize_codestral] main({mainQuantStr})={nFp4} fp32(error-sensitive)={nFp32} new-graft({graftQuantStr})={nGraft}"

  # arch[5]: [vocab, hidden, nLayers, nHeads, ffMult] - Codestral-7B-v0.1
  # (from Mistral's config.json): hidden=4096, nLayers=64. vocab/nHeads/ffMult
  # CHECK AGAIN against the downloaded params.json before using these numbers for
  # nim_inference.nim - filled in for now so arch is not empty (0,0,0,0,0).
  # arch[5]: [vocab, hidden, nLayers, nHeads(SSM), expand] - Codestral-7B-v0.1
  # CONFIRMED from the manifest (print_manifest.nim): vocab=32768, hidden=4096,
  # nLayers=64, nHeads=128 (from mixer.A_log=[128]), expand=2 (intermediate=8192).
  # NOTE: this arch is NOT read back by load_codestral_into_jamba.nim (that loader
  # maps tensors BY NAME and does not read arch) - only for reference/use by
  # nim_inference.nim (a different inference path, see the note in nim_inference.nim).
  let arch = [32768, 4096, 64, 128, 2]
  saveQuantStateDict(outPath, arch, sd)
  echo &"[quantize_codestral] Wrote {outPath} ({sd.len} tensors) - place it in your project directory."
  echo "[quantize_codestral] IMPORTANT WARNING: this .nimq uses the tensor-name schema " &
       "of the ORIGINAL HF Mamba-Codestral-7B-v0.1 (mamba_ssm/Mamba2 style), NOT matching 1-1 the " &
       "hand-written Mamba2SSD in test_nimformer.nim - the same problem noted in " &
       "jamba_hybrid.loadMambaOnlyFromHFBin. Use this file as INPUT for the tensor " &
       "name mapping step (nameMap) before loading into NimformerModelJamba, NOT directly."