## Strict official SISO 187M tensor mapping; frozen large matrices stay quantized.
import std/[tables, strutils, os]
import nimformer, quant, rawbin, m3_hybrid, mamba3_real
when defined(mamba3HfSource):
  import std/strformat
  import ../codestral/pytorch_bin_loader
  import ../codestral/sharded_hf_loader

proc loadMamba3QuantResident*(m: var NimformerModelJamba; byName: Table[string, QuantTensor]): Table[string, QuantTensor] =
  if m.embedDim <= 0 or m.vocab <= 0 or m.blocks.len == 0 or m.nHeads <= 0:
    raise newException(ValueError, "Invalid Mamba-3 model dimensions")
  proc require(name: string; shape: seq[int]): QuantTensor =
    if name notin byName:
      raise newException(ValueError,"Mamba-3 missing tensor: " & name & " expected " & $shape)
    if byName[name].shape!=shape:
      raise newException(ValueError,"Mamba-3 incompatible tensor: " & name &
        " found " & $byName[name].shape & " expected " & $shape)
    byName[name]
  let embedName = if "backbone.embedding.weight" in byName:
      "backbone.embedding.weight"
    elif "backbone.embeddings.weight" in byName:
      "backbone.embeddings.weight"
    else:
      raise newException(ValueError,
        "Mamba-3 missing embedding: expected backbone.embedding.weight (or backbone.embeddings.weight)")
  m.embed.weight.data=dequantizeTensor(require(embedName,m.embed.weight.shape))
  m.normF.gamma.data=dequantizeTensor(require("backbone.norm_f.weight",m.normF.gamma.shape))
  let headName=if "lm_head.weight" in byName: "lm_head.weight" else: embedName
  result["outProj.weight"]=require(headName,m.outProj.weight.shape)
  for i in 0..<m.blocks.len:
    if m.blocks[i].kind notin {bkMambaReal,bkMambaRealPlusAttn,bkMambaRealGated}:
      raise newException(ValueError,"Mamba-3 patterns must preserve all 12 pretrained layers; use R/P/G")
    let mapping=hfNameMapMamba3(i)
    let root="block" & $i
    for (name,pt) in paramListMamba3Real(m.blocks[i].mambaReal,root & ".mambaReal"):
      let local=name[(root & ".mambaReal.").len..^1]
      if local.endsWith(".bias"): continue
      let qt=require(mapping[local],pt[].shape)
      if local in ["inProj.weight","outProj.weight","mlpFc1.weight","mlpFc2.weight"]:
        result[name]=qt; pt[].data = @[]
      else: pt[].data=dequantizeTensor(qt)
    m.blocks[i].ln1.gamma.data=dequantizeTensor(require(mapping["ln1Weight"],@[m.embedDim]))
  m.outProj.weight.data = @[]
  syncRmsFromBaseNorms(m)

proc loadMamba3QuantResidentFromNimq*(m: var NimformerModelJamba; nimqPath: string): Table[string, QuantTensor] =
  let tensors=loadQuantStateDict(nimqPath, mapped = true)
  let arch = tensors.arch
  if m.blocks.len == 0:
    raise newException(ValueError, "Mamba-3 model has no layers")
  let expand = if m.blocks[0].kind in {bkMambaReal, bkMambaRealPlusAttn, bkMambaRealGated}:
      m.blocks[0].mambaReal.intermediateSize div m.embedDim
    else: 0
  if arch != [m.vocab, m.embedDim, m.blocks.len, m.nHeads, expand]:
    raise newException(ValueError,
      "Mamba-3 .nimq architecture metadata does not match model dimensions: " & $arch)
  var byName=initTable[string,QuantTensor]()
  for (name,qt) in tensors.sd: byName[name]=qt
  result = loadMamba3QuantResident(m, byName)

proc mamba3NimqArch*(nimqPath: string): array[5, int] =
  result = loadQuantStateDict(nimqPath, mapped = true).arch

proc newMamba3ModelForNimq*(nimqPath, pattern: string; useSession = true;
                            quantResident = true;
                            initializeWeights = false): NimformerModelJamba =
  let arch = mamba3NimqArch(nimqPath)
  if arch[0] <= 0 or arch[1] <= 0 or arch[2] <= 0 or arch[3] <= 0 or
     arch[4] <= 0 or pattern.len != arch[2]:
    raise newException(ValueError, "Invalid Mamba-3 .nimq architecture metadata: " & $arch)
  result = newNimformerModelJamba(arch[0], arch[1], arch[3], 4, pattern,
    useSession = useSession, mambaRealDState = 128, mambaRealNGroups = 1,
    mambaRealExpand = arch[4], quantResident = quantResident,
    initializeWeights = initializeWeights)

proc loadMamba3IntoJambaFromNimq*(m: var NimformerModelJamba; path: string) =
  let qs=loadMamba3QuantResidentFromNimq(m,path)
  for (name,pt) in paramList(m):
    if name in qs: pt[].data=dequantizeTensor(qs[name])

proc loadMamba3IntoJamba*(m: var NimformerModelJamba; hfDir: string) =
  when defined(mamba3HfSource):
    var tensors: seq[RawBinTensor]
    if dirExists(hfDir):
      let binPath = hfDir / "pytorch_model.bin"
      if fileExists(binPath):
        tensors = loadAllHFTensors(binPath)
      else:
        tensors = loadAllHFTensorsSharded(hfDir)
    elif fileExists(hfDir):
      tensors = loadAllHFTensors(hfDir)
    else:
      raise newException(IOError, "HF checkpoint path not found: " & hfDir)
    var byName = initTable[string, QuantTensor]()
    for t in tensors:
      if t.name in byName:
        raise newException(ValueError, "Duplicate tensor in HF checkpoint: " & t.name)
      byName[t.name] = quantizeRaw(t.data, t.shape)
    let qs = loadMamba3QuantResident(m, byName)
    for (name, pt) in paramList(m):
      if name in qs: pt[].data = dequantizeTensor(qs[name])
  else:
    raise newException(ValueError,
      "Direct HF loading is disabled in this build; compile with -d:mamba3HfSource (requires nimpy and Python torch), or convert the checkpoint to .nimq first")
