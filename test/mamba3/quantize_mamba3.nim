## Convert the official checkpoint using the existing Codestral FP4-KM format.
import std/[os, strutils]
import quant, pytorch_bin_loader, rawbin, m3_config
when isMainModule:
  if paramCount()!=2:
    quit("usage: quantize_mamba3 <pytorch_model.bin|model.safetensors> <output.nimq>",1)
  let cfg = loadMamba3Config(parentDir(paramStr(1)))
  var sd:seq[(string,QuantTensor)]
  var embed: QuantTensor
  var hasEmbed=false
  var hasHead=false
  forEachHFTensor(paramStr(1), proc(t: RawBinTensor) =
    let matrix=t.name.endsWith("in_proj.weight") or t.name.endsWith("out_proj.weight") or t.name=="lm_head.weight" or t.name.endsWith("mlp.fc1.weight") or t.name.endsWith("mlp.fc2.weight")
    let kind=if matrix: qkFp4KM else: qkFp32Raw
    let qt=quantizeTensor(t.data,t.shape,kind)
    sd.add((t.name,qt))
    if t.name=="lm_head.weight": hasHead=true
    if t.name=="backbone.embedding.weight" or t.name=="backbone.embeddings.weight":
      embed=quantizeTensor(t.data,t.shape,qkFp4KM)
      hasEmbed=true
  , mmap = paramStr(1).endsWith(".bin"))
  if not hasHead:
    if not hasEmbed or embed.shape != @[cfg.vocab,cfg.hidden]:
      raise newException(ValueError,"official embedding missing/wrong shape; expected [" &
        $cfg.vocab & "," & $cfg.hidden & "]")
    sd.add(("lm_head.weight",embed))
  createDir(parentDir(paramStr(2)))
  saveQuantStateDict(paramStr(2),[cfg.vocab,cfg.hidden,cfg.layers,cfg.nHeads,cfg.expand],sd)
