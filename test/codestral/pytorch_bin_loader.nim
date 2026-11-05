## pytorch_bin_loader.nim
## Loads the pytorch_model.bin of ayoubkirouane/Mamba-Chat-2.8B (based on
## the original mamba_ssm - MambaLMHeadModel, NOT transformers.AutoModel,
## dim=2560, 64 pure Mamba layers). The original model is stored in FP16
## (`MambaLMHeadModel.from_pretrained(..., dtype=torch.float16)`), so it is handled
## carefully:
##   - torch.load returns fp16 tensors -> they MUST be cast to fp32 before tobytes,
##     otherwise the buffer is 2 bytes/element (fp16) while the Nim code below
##     assumes 4 bytes/element (fp32) -> completely wrong reads or a crash.
##     pyTensorToSeq() below casts to fp32 ON THE NUMPY SIDE (arr.astype("float32"))
##     - NOT tensor.to(torch.float32) (see the technical reason + the specific nimpy error
##     hit, in the comment right at pyTensorToSeq) - do not drop this fp32
##     cast when editing the code.
##   - fp16 has a narrow range (~[-65504, 65504], a large eps) - some tensors such as
##     A_log and dt_proj.bias of the Mamba SSM may hold tiny/huge values
##     after exp/softplus; casting to fp32 keeps the values (only the
##     bit representation changes, lost precision is not recovered) - precision lost
##     when the original model was saved in fp16 cannot be recovered, this only avoids LOSING
##     MORE precision during the conversion.
##
## After this step, see also the "fp16" notes in lora.nim (LoRA A/B init +
## Adam eps must be large enough not to underflow when the base comes from fp16).
##
## Run first:
##   1. Download: huggingface-cli download ayoubkirouane/Mamba-Chat-2.8B --local-dir ./mamba-chat-2.8b
##   2. ./pytorch_bin_loader ./mamba-chat-2.8b/pytorch_model.bin
##      -> only writes the manifest (.manifest.json, very light) to see the real tensor names
##         (backbone.embedding.weight, backbone.layers.N.mixer.*, ...)
##   3. The fine-tune pipeline (graft_attention.nim --from-hf, jamba_hybrid.
##      loadMambaOnlyFromHFBin) reads this original .bin DIRECTLY through
##      loadAllHFTensors() - NO other .bin file needs to be exported,
##      avoiding doubling the disk usage (an original HF checkpoint can be several GB).
##      The `--export out.bin` flag below is ONLY an optional utility (archive/
##      offline inspection), NOT a required step - see the warning in
##      exportBinToRawBin().
##
## REQUIRES: `nimble install nimpy`, and a Python env with torch (confirmed
## installed). NIMPY_PYTHON or PYTHONHOME may need to be set if
## nimpy does not find the Python that has torch.

import std/[json, os, strformat, tables, strutils]
import nimpy
import rawbin   # exports to a RAW .bin (uncompressed) - NOT .nimq (quant.nim)
                # at this step, because this is part of the fine-tune pipeline

let pyBuiltins = pyImport("builtins")
proc pyLen(o: PyObject): int =
  ## NOTE: do NOT use o.len() - "len" in Python is a BUILT-IN FUNCTION
  ## (len(x)), NOT an object method (x.len()) - torch.Size (a tuple)
  ## and many other objects have NO "len" attribute -> o.len() makes nimpy
  ## try a dynamic call into Python and fail at run time with "No
  ## callable attribute: len". Use this function (calling builtins.len(x)) EVERYWHERE
  ## instead of o.len().to(int).
  pyBuiltins.len(o).to(int)

type
  HFTensorInfo* = object
    name*: string
    shape*: seq[int]
    dtype*: string

# ─────────────────────────────────────────────────────────────────
# Calls Python: torch.load(path, map_location="cpu") -> dict[str, Tensor]
# Supports both .bin (pickle) and .safetensors (through the safetensors library if present).
# ─────────────────────────────────────────────────────────────────
proc pyLoadStateDict(path: string; mmap = false): PyObject =
  let torch = pyImport("torch")
  if path.endsWith(".safetensors"):
    let st = pyImport("safetensors.torch")
    result = st.load_file(path)
  else:
    if mmap:
      result = torch.load(path, map_location = "cpu", mmap = true)
    else:
      result = torch.load(path, map_location = "cpu")

proc pyTensorToSeq(t: PyObject): tuple[data: seq[float32], shape: seq[int]] =
  ## .detach().contiguous().float().flatten() -> numpy -> tobytes() -> seq[float32].
  ## CHANGED (Codestral-7B / bf16): Mistral's checkpoint stores dtype bfloat16 -
  ## numpy does NOT SUPPORT bfloat16 (numpy has no such type), so
  ## tensor.numpy() RAISES IMMEDIATELY ("Got unsupported ScalarType
  ## BFloat16"), even before .astype("float32") in the next step (that step
  ## wrongly assumed .numpy() always succeeds and only the dtype needs changing).
  ## It MUST be cast to float32 ON THE TORCH SIDE (torch DOES support bfloat16,
  ## only numpy does not) BEFORE calling .numpy().
  ## IMPORTANT NOTE (still true as noted before): do NOT use tensor.to(torch.float32)
  ## - nimpy DEFINES its own proc named "to" (to convert a
  ## PyObject -> a Nim type, e.g. `.to(int)`), clashing with PyTorch's ".to(dtype)"
  ## method -> Nim/nimpy resolves it WRONGLY. Use ".float()" instead -
  ## PyTorch's OFFICIAL shorthand for ".to(torch.float32)", a completely DIFFERENT
  ## NAME, so nimpy's "to" is not involved, no confusion.
  ## ADDITIONAL NOTE (one more confusion just like ".to"): calling
  ## ".float()" directly on a PyObject is also resolved WRONGLY by Nim/nimpy as Nim's own
  ## builtin `float()` conversion (to Nim's numeric `float`,
  ## not a Python method call) -> "type mismatch: expected float".
  ## Solution: call callMethod("float") EXPLICITLY - forcing nimpy to dispatch
  ## dynamically into Python instead of letting Nim resolve the name first.
  let flat = t.detach().contiguous().callMethod("float").flatten()
  let shapeObj = t.shape
  var shape: seq[int] = @[]
  for i in 0 ..< pyLen(shapeObj):
    shape.add(shapeObj[i].to(int))
  let n = pyLen(flat)
  var data = newSeq[float32](n)
  # A per-element Python->Nim loop is slow for a 1B model (>1e9 parameters for
  # the whole checkpoint) if done 100% through nimpy indexing, so it goes through
  # numpy and takes the buffer directly for speed.
  let arrRaw = flat.numpy()             # now SURELY float32 (.float() above)
  let arr = arrRaw.astype("float32")    # kept (harmless, in case another dtype slips through)
  let buf = arr.tobytes()
  let raw = buf.to(string)
  assert raw.len == n * 4
  copyMem(addr data[0], unsafeAddr raw[0], raw.len)
  result = (data, shape)

proc listStateDictInfo*(path: string): seq[HFTensorInfo] =
  ## Reads only metadata (name + shape + dtype), does NOT pull all the data into RAM.
  ## Use this first to see the manifest, then decide how to map the names.
  let sd = pyLoadStateDict(path)
  result = @[]
  for key in sd.keys():
    let t = sd[key]
    let name = key.to(string)
    var shape: seq[int] = @[]
    let shapeObj = t.shape
    for i in 0 ..< pyLen(shapeObj):
      shape.add(shapeObj[i].to(int))
    # NOTE: t.dtype is a Python object of type 'torch.dtype' (e.g.
    # torch.float16), NOT a str PyObject -> calling
    # ".to(string)" on it directly is rejected by nimpy with the runtime error "Can't
    # convert python obj of type 'torch.dtype' to string" (nimpy's .to(string)
    # only accepts objects that REALLY are Python str, it does not call
    # str() for you). Python's built-in str() (through pyBuiltins) must be called
    # FIRST to turn it into a real str PyObject, then .to(string) works.
    let dtype = pyBuiltins.str(t.dtype).to(string)
    result.add(HFTensorInfo(name: name, shape: shape, dtype: dtype))

proc writeManifest*(infos: seq[HFTensorInfo], outPath: string) =
  var arr = newJArray()
  for info in infos:
    var shapeArr = newJArray()
    for s in info.shape: shapeArr.add(%s)
    arr.add(%*{"name": info.name, "shape": shapeArr, "dtype": info.dtype})
  writeFile(outPath, pretty(arr))
  echo &"[loader] Wrote the manifest of {infos.len} tensors -> {outPath}"

proc loadAllHFTensors*(path: string): seq[RawBinTensor] =
  ## Reads the WHOLE HF checkpoint (.bin/.safetensors) straight into RAM (seq
  ## RawBinTensor), writing NO file to disk. Use it when
  ## you need to build the Nim model directly from the original .bin (e.g.
  ## jamba_hybrid.loadMambaOnlyFromHFBin) without doubling the disk usage
  ## with an intermediate .bin file.
  let sd = pyLoadStateDict(path)
  result = @[]
  var n = 0
  for key in sd.keys():
    let name = key.to(string)
    let t = sd[key]
    let (data, shape) = pyTensorToSeq(t)
    result.add(RawBinTensor(name: name, shape: shape, data: data))
    inc n
    if n mod 20 == 0:
      echo &"[loader] ... {n} tensors read (RAM, nothing written to disk)"

proc forEachHFTensor*(path: string;
                      consume: proc(t: RawBinTensor) {.closure.};
                      mmap = false) =
  ## Converts and releases one tensor at a time so quantizers can handle large
  ## checkpoints without keeping a second fp32 copy of the complete state dict.
  let sd = pyLoadStateDict(path, mmap)
  var n = 0
  for key in sd.keys():
    let name = key.to(string)
    let (data, shape) = pyTensorToSeq(sd[key])
    consume(RawBinTensor(name: name, shape: shape, data: data))
    inc n
    if n mod 20 == 0:
      echo &"[loader] ... {n} tensors converted (streaming)"

proc exportBinToRawBin*(binPath, outBinPath: string, arch: array[5, int] = [0, 0, 0, 0, 0]) =
  ## OPTIONAL UTILITY, NOT A REQUIRED STEP of the fine-tune pipeline.
  ## Writes the whole .bin/.safetensors checkpoint to ANOTHER raw .bin file (rawbin.nim)
  ## - i.e. DOUBLES the disk usage (e.g. a 5.5GB checkpoint -> 5.5GB more).
  ## ONLY use it when you really need a copy readable by pure Nim without
  ## Python (e.g. to archive/inspect offline). For the normal fine-tune pipeline
  ## (graft_attention --from-hf, jamba_hybrid.loadMambaOnlyFromHFBin),
  ## do NOT call this - use loadAllHFTensors (RAM only) instead.
  let tensors = loadAllHFTensors(binPath)
  var meta = initTable[string, string]()
  meta["vocab"] = $arch[0]
  meta["embedDim"] = $arch[1]
  meta["nHeads"] = $arch[2]
  meta["nLayers"] = $arch[3]
  meta["ffMult"] = $arch[4]
  saveRawBin(outBinPath, meta, tensors)
  echo &"[loader] Done: {tensors.len} tensors -> {outBinPath} (WARNING: this file is an EXTRA COPY taking more disk)"

proc loadTensorByName*(path, tensorName: string): tuple[data: seq[float32], shape: seq[int]] =
  ## Reads exactly one tensor by name (saves RAM when only a few tensors are needed, e.g.
  ## the embedding or one layer for comparison/initialisation).
  let sd = pyLoadStateDict(path)
  # NOTE: do NOT use sd.contains(tensorName) - a Python dict has NO
  # ".contains()" method (only the "in"/__contains__ operator, not a method) ->
  # nimpy would try a dynamic call into Python and fail at
  # run time with "No callable attribute: contains". Iterate sd.keys() (a real
  # method, safe) and compare the names instead of .contains().
  var found = false
  for k in sd.keys():
    if k.to(string) == tensorName:
      found = true
      break
  if not found:
    raise newException(KeyError, "Tensor not found: " & tensorName)
  result = pyTensorToSeq(sd[tensorName])

when isMainModule:
  # usage: ./pytorch_bin_loader <path/pytorch_model.bin> [--export out.bin]
  # (--export is OPTIONAL and writes an extra .bin copy - see the warning in
  #  exportBinToRawBin(). The normal fine-tune pipeline does NOT need this flag,
  #  it reads the original .bin directly through loadAllHFTensors/graft_attention --from-hf)
  let args = commandLineParams()
  if args.len < 1:
    echo "usage: pytorch_bin_loader <model.bin|model.safetensors> [--export out.bin]"
    quit(1)
  let path = args[0]
  echo &"[loader] Doc manifest tu {path} ..."
  let infos = listStateDictInfo(path)
  writeManifest(infos, path & ".manifest.json")
  for info in infos[0 ..< min(20, infos.len)]:
    echo &"  {info.name}  shape={info.shape}  dtype={info.dtype}"
  if infos.len > 20:
    echo &"  ... {infos.len - 20} more tensors, see the full list in {path}.manifest.json"
  if args.len >= 3 and args[1] == "--export":
    exportBinToRawBin(path, args[2], [0, 0, 0, 0, 0])