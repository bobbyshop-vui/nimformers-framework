import std/[math, random, os, parseopt, strutils]
import backend
import test_nimformer  # Imports the whole test_nimformer module
import block_decode

# =============================================================================
# CONFIGURATION
# =============================================================================
const
  CHECKPOINT_PATH* = "test/spectron/models/hrm-eos/finetune.nimq"
  BPE_TOKENIZER_PATH* = "test/spectron/data/scale5b/tokenizer.json"
  MAX_NEW_TOKENS* = 128
  DEFAULT_TEMPERATURE* = 0'f32  # 0 selects confidence-guided parallel block generation
  DEFAULT_BLOCK* = 16
  DEFAULT_CONTEXT* = 128
  DEFAULT_TAU* = 0.9'f32
  SEED* = 1234
  Usage = """
Usage: test_inferencespectron [options]
  --checkpoint=PATH  Checkpoint (default test/spectron/models/hrm-eos/finetune.nimq)
  --tokenizer=PATH   BPE tokenizer (default test/spectron/data/scale5b/tokenizer.json)
  --temperature=0    0 answers with confidence-guided parallel block generation, which
                     gives the greedy reply; above 0 samples one token per forward pass
  --block=16         Positions generated in parallel per block (must divide 128)
  --context=128      Window: the last context-block tokens plus the block
  --tau=0.9          Confidence needed to lock a draft (0.5 < tau <= 1)
"""

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
  ## Samples one token per forward pass over the whole history.
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

proc blockReply(model: NimformerModelSpectron, history: seq[int], blockSize, context: int,
                tau: float32, seed: int64, ctx: Backend, stats: var BlockDecodeStats): seq[int] =
  ## Confidence-guided parallel block generation. Verification keeps the reply
  ## equal to greedy decoding, and the reply ends at a verified EOS.
  result = model.blockDecode(@[history], MAX_NEW_TOKENS, context, blockSize,
                             bpConfidence, tau, seed, ctx, stats)[0]
  if model.eosTokenId < 0:
    # Checkpoints without EOS stop at ID 0, as in the sampled path.
    for i in history.len ..< result.len:
      if result[i] == 0:
        result.setLen(i + 1)
        break

# =============================================================================
# MAIN CHAT LOOP
# =============================================================================
when isMainModule:
  var checkpointPath = CHECKPOINT_PATH
  var tokenizerPath = BPE_TOKENIZER_PATH
  var temperature = DEFAULT_TEMPERATURE
  var blockSize = DEFAULT_BLOCK
  var context = DEFAULT_CONTEXT
  var tau = DEFAULT_TAU
  for kind, key, val in getopt():
    if kind == cmdLongOption and key == "help":
      quit(Usage, 0)
    if kind != cmdLongOption or val.len == 0:
      quit("Expected --name=value. See --help.", 1)
    case key
    of "checkpoint": checkpointPath = val
    of "tokenizer": tokenizerPath = val
    of "temperature": temperature = float32(parseFloat(val))
    of "block": blockSize = parseInt(val)
    of "context": context = parseInt(val)
    of "tau": tau = float32(parseFloat(val))
    else: quit("Unknown option --" & key, 1)
  if not (temperature >= 0'f32) or temperature == Inf.float32:
    quit("--temperature must be 0 or a positive number", 1)
  if blockSize < 1 or MAX_NEW_TOKENS mod blockSize != 0 or context <= blockSize:
    quit("--block must divide " & $MAX_NEW_TOKENS & " and be smaller than --context", 1)
  if not (tau > 0.5'f32 and tau <= 1'f32):
    quit("--tau must be in (0.5, 1]", 1)
  setForbidCpuFallback(true)
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
  var (model, step) = test_nimformer.loadCheckpoint(checkpointPath)
  if not model.matchesBpe(tok.vocabSize):
    quit("Checkpoint vocabulary size differs from the tokenizer", 1)
  echo "Model loaded (trained for ", step, " steps, hrm=", model.hrmEnabled,
       ", experts=", model.expertCount, ", eos=", model.eosTokenId, ")"
  if model.blockExpertCount > 0: model.startRouteTally()
  if temperature > 0'f32:
    echo "Generation: sampling at temperature ", temperature
  else:
    echo "Generation: confidence-guided parallel block generation (block=", blockSize,
         " context=", context, " tau=", tau, ")"
  echo ""
  echo "Type your message and press Enter to chat (Ctrl+C or 'exit' to quit)"
  echo ""

  var conversationHistory: seq[int] = @[]
  var currentSeed = SEED

  while true:
    stdout.write(">>> ")
    stdout.flushFile()

    var userInput: string
    if not stdin.readLine(userInput):
      break

    if userInput == "":
      continue

    if userInput.toLower == "exit" or userInput.toLower == "quit":
      break

    let promptIds = test_nimformer.encodeBPE(tok, userInput)

    var fullInput = conversationHistory
    fullInput.add(promptIds)

    echo ""
    echo "You: ", userInput

    var stats: BlockDecodeStats
    let responseIds =
      if temperature > 0'f32:
        model.generateWithHistory(fullInput, maxNewTokens = MAX_NEW_TOKENS, ctx = ctx,
                                  temperature = temperature, seed = currentSeed)
      else:
        model.blockReply(fullInput, blockSize, context, tau, currentSeed, ctx, stats)

    let responseOnlyIds = responseIds[fullInput.len .. ^1]
    let responseText = test_nimformer.decodeBPEStr(tok, responseOnlyIds)

    echo "Assistant: ", responseText
    if stats.rowPasses > 0:
      echo "[block] tokens=", responseOnlyIds.len, " passes=", stats.rowPasses,
           " tokens_per_pass=",
           formatFloat(float64(responseOnlyIds.len) / float64(stats.rowPasses), ffDecimal, 2)
    if model.routeTally != nil:
      # Rows each block's router sent to each expert for this reply, over every
      # forward pass and every HRM pass of the block.
      for i in 0 ..< model.routeTally[].len:
        template tally: untyped = model.routeTally[][i]
        let total = sum(tally.tokens)
        echo "[moe] block=", i, " expert_tokens=", tally.tokens, " chosen_p=",
             formatFloat(tally.chosenProbability / float64(max(total, 1'i64)), ffDecimal, 4)
        tally = BlockRouteTally()
    echo ""

    conversationHistory = responseIds
    currentSeed = currentSeed + 1

  closeBackend(ctx)
  echo ""
  echo "Goodbye!"
