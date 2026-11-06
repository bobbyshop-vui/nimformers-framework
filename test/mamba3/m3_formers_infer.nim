include infer_lora

# Step API for formers.n conversation mode: the same GPU preparation and decode
# step that generate() uses, exposed so a chat loop can feed tokens turn by turn.

proc formersPrepareChat*(m: var NimformerModelJamba, lora: var JambaLora, ctx: Backend) =
  mergeInUploadStreamingJamba(m, lora, ctx)
  prepareGpuModel(m, lora, ctx)
  if getEnv("BYBY_MERGE_LORA", "0") != "0" and lora.adapters.len > 0:
    discard mergeResidentLoraInplace(m, lora, ctx)
  let preset = getEnv("BYBY_NORM_PRESET", if gCkptNormPreset.len > 0: gCkptNormPreset else: "rms-fin-gn")
  if not setGpuNormPreset(preset):
    raise newException(ValueError, "[formers.n] invalid norm preset: " & preset)
  resetGpuGenState(m, ctx)

proc formersStepChat*(m: var NimformerModelJamba, tokenId: int, history: seq[int], ctx: Backend): seq[float32] =
  ## history already ends with tokenId; returns the logits after it.
  if tokenId < 0 or tokenId >= m.vocab:
    raise newException(ValueError, "[formers.n] token id " & $tokenId & " outside vocab " & $m.vocab)
  forwardStepJambaGpuRetry(m, tokenId, history, history.len - 1, ctx).data
