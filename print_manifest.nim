import std/[strutils, algorithm, os]
import sharded_hf_loader

when isMainModule:
  let hfDir = paramStr(1)
  let tensors = loadAllHFTensorsSharded(hfDir)

  echo "\n=== Tensor CHUA 'layers.0.' (mixer cua layer dau tien) ==="
  var layer0Names: seq[string] = @[]
  for t in tensors:
    if "layers.0." in t.name:
      layer0Names.add(t.name & "  shape=" & $t.shape)
  layer0Names.sort()
  for n in layer0Names: echo "  ", n

  echo "\n=== Tensor CHUA 'embed' hoac 'lm_head' hoac khong co 'layers.' ==="
  for t in tensors:
    if "embed" in t.name.toLowerAscii or "lm_head" in t.name.toLowerAscii or "layers." notin t.name:
      echo "  ", t.name, "  shape=", t.shape

  echo "\n=== Dem tong so layer (dua theo 'layers.N.') ==="
  var maxLayer = -1
  for t in tensors:
    let idx = t.name.find("layers.")
    if idx >= 0:
      let rest = t.name[idx + 7 .. ^1]
      let dotIdx = rest.find('.')
      if dotIdx > 0:
        try:
          let n = parseInt(rest[0 ..< dotIdx])
          if n > maxLayer: maxLayer = n
        except: discard
  echo "  So layer = ", maxLayer + 1