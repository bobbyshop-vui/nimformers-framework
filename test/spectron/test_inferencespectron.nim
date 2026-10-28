import std/[math, random, strformat, os, strutils, terminal]
import backend
import test_nimformer  # Import toàn bộ module test_nimformer

# =============================================================================
# CẤU HÌNH
# =============================================================================
const
  CHECKPOINT_PATH* = "test/spectron/models/finetune.nimq"
  BPE_TOKENIZER_PATH* = "test/spectron/models/spectron_bpe_tokenizer.json"
  MAX_NEW_TOKENS* = 128
  DEFAULT_TEMPERATURE* = 0.7'f32
  SEED* = 1234

# =============================================================================
# SAMPLING FUNCTION
# =============================================================================
proc sampleFromLogits(logits: seq[float32], temperature: float32, rng: var Rand): int =
  if temperature <= 0'f32:
    var best = 0
    var bestV = logits[0]
    for i in 1 ..< logits.len:
      if logits[i] > bestV:
        bestV = logits[i]; best = i
    return best
  var probs = newSeq[float32](logits.len)
  var maxL = logits[0]
  for v in logits:
    if v > maxL: maxL = v
  var sum = 0'f32
  for i in 0 ..< logits.len:
    probs[i] = exp((logits[i] - maxL) / temperature)
    sum += probs[i]
  let r = float32(rng.rand(1.0)) * sum
  var acc = 0'f32
  for i in 0 ..< probs.len:
    acc += probs[i]
    if acc >= r: return i
  return probs.len - 1

# =============================================================================
# GENERATION WITH HISTORY
# =============================================================================
proc generateWithHistory(
  model: NimformerModelSpectron,
  history: seq[int],
  maxNewTokens: int,
  ctx: Backend,
  temperature: float32,
  seed: int64
): seq[int] =
  var rng = initRand(seed)
  result = history
  for _ in 0 ..< maxNewTokens:
    let logits = model.forward(result, ctx)
    let T = logits.shape[0]
    let V = logits.shape[1]
    let lastLogits = logits.data[(T - 1) * V ..< T * V]
    let nextId = sampleFromLogits(lastLogits, temperature, rng)
    result.add(nextId)
    if nextId == 0:  # EOS token
      break

# =============================================================================
# MAIN CHAT LOOP
# =============================================================================
when isMainModule:
  randomize(SEED)
  let ctx = newBackend()

  # Load tokenizer từ test_nimformer
  if not fileExists(BPE_TOKENIZER_PATH):
    stderr.writeLine("ERROR: Tokenizer file not found: " & BPE_TOKENIZER_PATH)
    quit(1)

  let tok = test_nimformer.loadBPE(BPE_TOKENIZER_PATH)
  echo "Tokenizer loaded: vocab_size = ", tok.vocabSize

  # Load model checkpoint
  if not fileExists(CHECKPOINT_PATH):
    stderr.writeLine("ERROR: Checkpoint file not found: " & CHECKPOINT_PATH)
    quit(1)

  echo "Loading model from checkpoint: ", CHECKPOINT_PATH
  let (model, step) = test_nimformer.loadCheckpoint(CHECKPOINT_PATH)
  echo "Model loaded (trained for ", step, " steps)"
  echo ""
  echo "Type your message and press Enter to chat (Ctrl+C or 'exit' to quit)"
  echo ""

  var conversationHistory: seq[int] = @[]
  var currentSeed = SEED

  while true:
    stdout.write(">>> ")
    stdout.flushFile()

    let userInput = readLine(stdin)

    if userInput == "":
      continue

    if userInput.toLower == "exit" or userInput.toLower == "quit":
      break

    # DÙNG CÁC HÀM TỪ test_nimformer
    let promptIds = test_nimformer.encodeBPE(tok, userInput)

    var fullInput = conversationHistory
    fullInput.add(promptIds)

    echo ""
    echo "You: ", userInput

    let responseIds = model.generateWithHistory(
      fullInput,
      maxNewTokens = MAX_NEW_TOKENS,
      ctx = ctx,
      temperature = DEFAULT_TEMPERATURE,
      seed = currentSeed
    )

    let responseOnlyIds = responseIds[fullInput.len .. ^1]
    let responseText = test_nimformer.decodeBPEStr(tok, responseOnlyIds)

    echo "Assistant: ", responseText
    echo ""

    conversationHistory = responseIds
    currentSeed = currentSeed + 1

  closeBackend(ctx)
  echo ""
  echo "Goodbye!"
