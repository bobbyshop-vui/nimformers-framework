import std/[json, os, strformat]

type
  Mamba3Config* = object
    vocab*, hidden*, intermediate*, layers*: int
    dState*, nHeads*, nGroups*, expand*: int
    normEps*: float32

proc loadMamba3Config*(path: string): Mamba3Config =
  let configPath = if dirExists(path): path / "config.json" else: path
  if not fileExists(configPath):
    raise newException(IOError, "Mamba-3 config.json not found: " & configPath)
  let j = parseFile(configPath)
  let ssm = j["ssm_cfg"]
  if ssm["layer"].getStr != "Mamba3" or ssm["is_mimo"].getBool:
    raise newException(ValueError, "Only Mamba-3 SISO configs are supported: " & configPath)
  if j["attn_layer_idx"].len != 0:
    raise newException(ValueError, "Mamba-3 checkpoints with attention layers are unsupported")
  result.vocab = j["vocab_size"].getInt
  result.hidden = j["d_model"].getInt
  result.intermediate = j["d_intermediate"].getInt
  result.layers = j["n_layer"].getInt
  result.dState = ssm["d_state"].getInt
  result.expand = ssm["expand"].getInt
  result.nGroups = ssm["ngroups"].getInt
  let headDim = ssm["headdim"].getInt
  if result.vocab <= 0 or result.hidden <= 0 or result.intermediate <= 0 or
     result.layers <= 0 or result.dState <= 0 or result.expand <= 0 or
     result.nGroups != 1 or headDim <= 0 or
     result.intermediate != result.hidden * result.expand or
     result.intermediate mod headDim != 0 or result.dState mod 4 != 0:
    raise newException(ValueError, &"Unsupported Mamba-3 SISO dimensions in {configPath}")
  if ssm["rope_fraction"].getFloat != 0.5:
    raise newException(ValueError, "This Mamba-3 implementation requires rope_fraction=0.5")
  result.nHeads = result.intermediate div headDim
  result.normEps = j.getOrDefault("layer_norm_epsilon").getFloat(1e-5).float32
