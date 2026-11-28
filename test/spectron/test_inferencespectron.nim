import std/[math, random, os, parseopt, strutils]
import backend
import test_nimformer  # Imports the whole test_nimformer module

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
    let eos = model.eosTokenId
    if nextId == (if eos >= 0: eos else: 0):
      break

# =============================================================================
# MAIN CHAT LOOP
# =============================================================================
when isMainModule:
  var checkpointPath = CHECKPOINT_PATH
  var tokenizerPath = BPE_TOKENIZER_PATH
  for kind, key, val in getopt():
    if kind == cmdLongOption and key == "help":
      quit("Usage: test_inferencespectron [--checkpoint=PATH] [--tokenizer=PATH]", 0)
    if kind != cmdLongOption or val.len == 0:
      quit("Expected --checkpoint=PATH or --tokenizer=PATH", 1)
    case key
    of "checkpoint": checkpointPath = val
    of "tokenizer": tokenizerPath = val
    else: quit("Unknown option --" & key, 1)
  randomize(SEED)
  let ctx = newBackend()

  # Load the tokenizer from test_nimformer
  if not fileExists(tokenizerPath):
    stderr.writeLine("ERROR: Tokenizer file not found: " & tokenizerPath)
    quit(1)

  let tok = test_nimformer.loadBPE(tokenizerPath)
  echo "Tokenizer loaded: vocab_size = ", tok.vocabSize

  # Load model checkpoint
  if not fileExists(checkpointPath):
    stderr.writeLine("ERROR: Checkpoint file not found: " & checkpointPath)
    quit(1)

  echo "Loading model from checkpoint: ", checkpointPath
  let (model, step) = test_nimformer.loadCheckpoint(checkpointPath)
  if not model.matchesBpe(tok.vocabSize):
    quit("Checkpoint vocabulary size differs from the tokenizer", 1)
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
