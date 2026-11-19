include infer_lora

# Step API for formers.n conversation mode: the same GPU preparation and decode
# step that generate() uses, exposed so a chat loop can feed tokens turn by turn.

proc formersPrepareChat*(m: var NimformerModelJamba, lora: var JambaLora, ctx: Backend) =
  mergeInUploadStreamingJamba(m, lora, ctx)
  prepareGpuModel(m, lora, ctx)
  discard mergeResidentLoraAll(m, lora, ctx)
  let preset = getEnv("BYBY_NORM_PRESET", if gCkptNormPreset.len > 0: gCkptNormPreset else: "rms-fin-gn")
  if not setGpuNormPreset(preset):
    raise newException(ValueError, "[formers.n] invalid norm preset: " & preset)
  resetGpuGenState(m, ctx)

proc formersStepChat*(m: var NimformerModelJamba, tokenId: int, history: seq[int], ctx: Backend): seq[float32] =
  ## history already ends with tokenId; returns the logits after it.
  if tokenId < 0 or tokenId >= m.vocab:
    raise newException(ValueError, "[formers.n] token id " & $tokenId & " outside vocab " & $m.vocab)
  forwardStepJambaGpuRetry(m, tokenId, history, history.len - 1, ctx).data

proc chatPrefillOn(): bool =
  ## BYBY_CHAT_PREFILL=0 feeds every chat token as a decode step (the old behaviour), which is the
  ## reference the chunked prefill is compared against.
  getEnv("BYBY_CHAT_PREFILL", "1") != "0"

proc formersFeedChat*(m: var NimformerModelJamba, ids: seq[int], history: seq[int], ctx: Backend): seq[float32] =
  ## Feeds ids (history already ends with ids[^1]) and returns the logits after the last one. On the GPU
  ## the ids before the last go through the prefill scheduler in chunks of fixed lengths (the ones
  ## formersWarmupChat compiled), the few left over through decode steps without logits, and the last id
  ## is a decode step with logits.
  if ids.len == 0:
    raise newException(ValueError, "[formers.n] nothing to feed")
  for id in ids:
    if id < 0 or id >= m.vocab:
      raise newException(ValueError, "[formers.n] token id " & $id & " outside vocab " & $m.vocab)
  let last = ids.len - 1
  var done = 0
  if last > 0 and isGpuCtx(ctx) and chatPrefillOn():
    done = prefillM3Bucketed(m, ids[0 ..< last], ctx)
  let base = history.len - ids.len
  for k in done ..< last:
    discard forwardStepJambaGpuRetry(m, ids[k], history, base + k, ctx, produceLogits = false)
  forwardStepJambaGpuRetry(m, ids[last], history, history.len - 1, ctx).data

proc formersWarmupChat*(m: var NimformerModelJamba, ctx: Backend): seq[int] =
  ## Before the first prompt of a conversation: compiles the prefill chunk lengths and the decode step,
  ## then clears the state, so the first turn pays no kernel setup/JIT. Returns the chunk lengths it ran
  ## (none on the CPU, which has nothing to compile).
  if not isGpuCtx(ctx):
    return
  result = warmM3Resident(m, ctx, prefill = chatPrefillOn())
  resetGpuGenState(m, ctx)
