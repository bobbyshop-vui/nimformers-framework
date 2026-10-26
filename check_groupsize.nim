import quant
import std/strutils

let (arch, sd) = loadQuantStateDict("model.nimq")
echo "arch: ", arch
var shown = 0
for (name, qt) in sd:
  if name.contains("q_proj.weight") or name.contains("qProj.weight"):
    echo name, " -> kind=", qt.kind, " groupSize=", qt.groupSize, " shape=", qt.shape, " nScales=", qt.scale.len
    inc shown
    if shown >= 3: break
if shown == 0:
  for i, (name, qt) in sd.pairs:
    if i >= 8: break
    echo name, " -> kind=", qt.kind, " groupSize=", qt.groupSize, " shape=", qt.shape

echo "---"
echo "Kiem tra rieng bias co bi quantize int4 khong (KHONG NEN, bias nen la fp32 raw):"
var biasShown = 0
for (name, qt) in sd:
  if name.contains(".bias") and (name.contains("q_proj") or name.contains("k_proj")):
    echo name, " -> kind=", qt.kind, " (kind=0 la QK_FP32_RAW, dung; kind=3 la int4, SAI vi bias khong nen bi nen 4-bit)"
    inc biasShown
    if biasShown >= 2: break