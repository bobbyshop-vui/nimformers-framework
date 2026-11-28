## Confidence-guided parallel block decoding for Spectron checkpoints.
##
## A block of K positions starts from draft tokens and is decoded in passes. One
## pass runs the model once over the context window plus the K drafts and
## predicts every block position in parallel: position j is read from the logits
## of the input just before it.
##
## Verification keeps the output identical to greedy decoding. Let e be the first
## unverified position. Its prediction only reads verified inputs, so it is exact.
## While the draft at e already equals that prediction, the next prediction also
## reads verified inputs only, so it is exact too; at the first mismatch the draft
## takes the exact prediction. Every pass verifies at least one position, so a
## block takes at most K passes: never more forward calls than greedy decoding,
## and fewer when drafts are already right.
##
## Positions after e follow a policy:
## - `jacobi` replaces every unverified draft with its new prediction.
## - `confidence` locks a draft once its prediction has softmax probability of at
##   least tau and is unchanged over two passes. A locked draft changes only when
##   the model predicts another token with probability at least tau. Confidence
##   only decides which drafts stay put; a locked draft that is wrong is replaced
##   when verification reaches it, so confidence cannot change the output.
##
## The greedy reference uses the same window for every position of a block: the
## last `context - K` tokens before the block plus the block itself. The model is
## causal, so tokens after a position do not change its logits.
import std/[math, monotimes, random, strformat, strutils, times]
import backend, nimformer, test_nimformer
when isMainModule:
  import std/parseopt

type
  BlockPolicy* = enum
    bpJacobi = "jacobi"
    bpConfidence = "confidence"

  BlockDecodeStats* = object
    forwards*: int     ## Batched model calls.
    rowPasses*: int    ## Passes summed over rows and blocks.
    blocks*: int       ## Blocks decoded per row.
    accepted*: int     ## Drafts that already equaled the exact prediction when verified.
    written*: int      ## Positions verified by writing the exact prediction.
    locks*: int        ## Drafts locked by confidence.
    rightLocks*: int   ## Locked drafts that verification accepted.
    wrongLocks*: int   ## Locked drafts that verification replaced.
    blockPasses*: seq[seq[int]]  ## Passes taken by every block, per row.
    seconds*: float64

  Prediction* = tuple[token: int, confidence, margin: float32]

  DraftBlock* = object
    ## Decoding state of one prompt's current block.
    drafts*: seq[int]    ## Current token at every block position.
    locked*: seq[bool]   ## Drafts kept by the confidence policy.
    previous*: seq[int]  ## Prediction of the previous pass, -1 before the first pass.
    verified*: int       ## Positions before this index equal greedy decoding.
    passes*: int         ## Passes run on this block so far.

  TokenText* = proc (id: int): string

proc predict(logits: Tensor, row, position: int): Prediction =
  ## Greedy token, its softmax probability and the gap to the runner-up logit.
  let T = logits.shape[1]
  let V = logits.shape[2]
  let offset = (row * T + position) * V
  var top = -Inf.float32
  var second = -Inf.float32
  for v in 0 ..< V:
    let x = logits.data[offset + v]
    if x > top:
      second = top
      top = x
      result.token = v
    elif x > second:
      second = x
  var total = 0.0
  for v in 0 ..< V: total += exp(float64(logits.data[offset + v] - top))
  result.confidence = float32(1.0 / total)
  result.margin = top - second

proc windowStart(prefixLen, context, blockSize: int): int =
  max(0, prefixLen - (context - blockSize))

proc checkArguments(prompts: seq[seq[int]], newTokens, context, blockSize: int) =
  if prompts.len == 0 or prompts[0].len == 0:
    raise newException(ValueError, "Block decoding needs at least one non-empty prompt")
  for prompt in prompts:
    if prompt.len != prompts[0].len:
      raise newException(ValueError, "Prompts in one batch must have the same token count")
  if blockSize < 1 or context <= blockSize or newTokens < 1 or newTokens mod blockSize != 0:
    raise newException(ValueError,
      "Need 1 <= block < context and a token count that is a multiple of the block")

proc initDraftBlock*(drafts: seq[int]): DraftBlock =
  result = DraftBlock(drafts: drafts, locked: newSeq[bool](drafts.len),
                      previous: newSeq[int](drafts.len))
  for token in result.previous.mitems: token = -1

proc checkTau*(tau: float32) =
  ## Above 1/2 at most one token can reach tau, so a confident conflicting
  ## prediction always names one replacement.
  if not (tau > 0.5'f32 and tau <= 1'f32):
    raise newException(ValueError, "--tau must be in (0.5, 1]")

proc applyPass*(b: var DraftBlock, predictions: openArray[Prediction], policy: BlockPolicy,
                tau: float32, stats: var BlockDecodeStats, eos = -1) =
  ## Updates the block after one pass. `predictions[j]` must come from the inputs
  ## before position j, which include the current drafts before j.
  doAssert predictions.len == b.drafts.len and b.verified < b.drafts.len
  inc stats.rowPasses
  inc b.passes
  # Verification: everything before e is verified, so prediction e is exact.
  var e = b.verified
  while e < b.drafts.len and b.drafts[e] == predictions[e].token:
    inc stats.accepted
    if b.locked[e]: inc stats.rightLocks
    inc e
    if b.drafts[e - 1] == eos:
      b.verified = e
      b.drafts.setLen(e)
      return
  if e < b.drafts.len:
    if b.locked[e]: inc stats.wrongLocks
    b.drafts[e] = predictions[e].token
    inc stats.written
    inc e
    if b.drafts[e - 1] == eos:
      b.verified = e
      b.drafts.setLen(e)
      return
  b.verified = e
  for i in e ..< b.drafts.len:
    let p = predictions[i]
    case policy
    of bpJacobi:
      b.drafts[i] = p.token
    of bpConfidence:
      if b.locked[i]:
        if p.token != b.drafts[i] and p.confidence >= tau: b.drafts[i] = p.token
      else:
        b.drafts[i] = p.token
        if p.confidence >= tau and b.previous[i] == p.token:
          b.locked[i] = true
          inc stats.locks
    b.previous[i] = p.token

proc greedyBlocks*(model: NimformerModelSpectron, prompts: seq[seq[int]],
                   newTokens, context, blockSize: int, ctx: Backend,
                   stats: var BlockDecodeStats, margins: var seq[seq[float32]]): seq[seq[int]] =
  ## Reference greedy decoding: one forward call per position. `margins` gets the
  ## logit gap between the chosen token and the runner-up for every new token.
  checkArguments(prompts, newTokens, context, blockSize)
  let started = getMonoTime()
  result = prompts
  margins = newSeq[seq[float32]](prompts.len)
  let eos = model.eosTokenId
  var stopped = newSeq[bool](prompts.len)
  for blockIndex in 0 ..< newTokens div blockSize:
    let prefixLen = prompts[0].len + blockIndex * blockSize
    let first = windowStart(prefixLen, context, blockSize)
    let base = prefixLen - first
    var generated = newSeq[seq[int]](prompts.len)
    for j in 0 ..< blockSize:
      var windows: seq[seq[int]]
      var activeRows: seq[int]
      for r in 0 ..< prompts.len:
        if stopped[r]: continue
        activeRows.add(r)
        var window = result[r][first ..< prefixLen] & generated[r]
        # Causal model: filler after position j does not change its logits.
        while window.len < base + blockSize: window.add(window[^1])
        windows.add(move(window))
      if activeRows.len == 0: break
      let logits = model.forwardBatch(windows, ctx)
      inc stats.forwards
      for slot, r in activeRows:
        let p = predict(logits, slot, base - 1 + j)
        generated[r].add(p.token)
        margins[r].add(p.margin)
        stopped[r] = p.token == eos
        inc stats.rowPasses
        inc stats.written
    var active = false
    for row in generated:
      if row.len > 0: active = true
    if not active: break
    for r in 0 ..< prompts.len: result[r].add(generated[r])
    inc stats.blocks
  stats.seconds += float64((getMonoTime() - started).inNanoseconds) * 1e-9

proc traceLine(blockIndex, pass: int, b: DraftBlock, tokenText: TokenText): string =
  ## Verified tokens are plain, locked drafts are in {braces}, other drafts start with ?.
  result = &"[trace] block={blockIndex} pass={pass} verified={b.verified}/{b.drafts.len} "
  for i, id in b.drafts:
    let text = tokenText(id).multiReplace(("\n", "\\n"), ("\t", "\\t"))
    result.add(if i < b.verified: "|" & text
               elif b.locked[i]: "|{" & text & "}"
               else: "|?" & text)
  result.add("|")

proc blockDecode*(model: NimformerModelSpectron, prompts: seq[seq[int]],
                  newTokens, context, blockSize: int, policy: BlockPolicy, tau: float32,
                  seed: int64, ctx: Backend, stats: var BlockDecodeStats,
                  tokenText: TokenText = nil, traceBlocks = 0): seq[seq[int]] =
  ## Decodes `newTokens` tokens per prompt in blocks of `blockSize` positions.
  checkArguments(prompts, newTokens, context, blockSize)
  checkTau(tau)
  let started = getMonoTime()
  let rows = prompts.len
  let eos = model.eosTokenId
  let draftVocab = if eos >= 0: eos else: model.vocab
  var stopped = newSeq[bool](rows)
  var rng = initRand(seed)
  result = prompts
  stats.blockPasses.setLen(rows)
  var predictions = newSeq[Prediction](blockSize)
  for blockIndex in 0 ..< newTokens div blockSize:
    let prefixLen = prompts[0].len + blockIndex * blockSize
    let first = windowStart(prefixLen, context, blockSize)
    let base = prefixLen - first
    var blocks = newSeq[DraftBlock](rows)
    var active = false
    for r in 0 ..< rows:
      if stopped[r]: continue
      active = true
      var drafts = newSeq[int](blockSize)
      for token in drafts.mitems: token = rng.rand(draftVocab - 1)
      blocks[r] = initDraftBlock(drafts)
    if not active: break
    var pass = 0
    while true:
      var windows: seq[seq[int]]
      var activeRows: seq[int]
      for r in 0 ..< rows:
        if blocks[r].verified >= blocks[r].drafts.len: continue
        activeRows.add(r)
        windows.add(result[r][first ..< prefixLen] & blocks[r].drafts)
      if activeRows.len == 0: break
      inc pass
      let logits = model.forwardBatch(windows, ctx)
      inc stats.forwards
      for slot, r in activeRows:
        for j in 0 ..< blockSize: predictions[j] = predict(logits, slot, base - 1 + j)
        blocks[r].applyPass(predictions, policy, tau, stats, eos)
        if r == 0 and blockIndex < traceBlocks and tokenText != nil:
          echo traceLine(blockIndex, pass, blocks[r], tokenText)
    for r in 0 ..< rows:
      if stopped[r]: continue
      result[r].add(blocks[r].drafts)
      stopped[r] = blocks[r].drafts[^1] == eos
      stats.blockPasses[r].add(blocks[r].passes)
    inc stats.blocks
  stats.seconds += float64((getMonoTime() - started).inNanoseconds) * 1e-9

type DraftProbe* = object
  ## Counts over block positions 1..K-1 of real text windows.
  total*, agree*, cleanCorrect*, noisyCorrect*: int

proc draftProbe*(model: NimformerModelSpectron, windows: seq[seq[int]], blockSize: int,
                 noise: float32, seed: int64, ctx: Backend): DraftProbe =
  ## Measures how much block predictions depend on wrong drafts, on the same text
  ## for every model. The last `blockSize` tokens of each window form the block.
  ## The clean pass feeds the true block tokens; the noisy pass replaces each
  ## draft input with probability `noise` by a different token. Position 0 reads
  ## no draft, so only positions 1..K-1 are counted. `agree` counts predictions
  ## that the wrong drafts did not change: in block decoding those are the drafts
  ## that verification can accept without another pass.
  let width = windows[0].len
  let base = width - blockSize
  let eos = model.eosTokenId
  let draftVocab = if eos >= 0: eos else: model.vocab
  var rng = initRand(seed)
  var noisyWindows = windows
  for r in 0 ..< windows.len:
    for i in base ..< width - 1:
      if rng.rand(1.0) < float64(noise):
        noisyWindows[r][i] = (windows[r][i] + 1 + rng.rand(draftVocab - 2)) mod draftVocab
  let clean = model.forwardBatch(windows, ctx)
  let noisy = model.forwardBatch(noisyWindows, ctx)
  for r in 0 ..< windows.len:
    for j in 1 ..< blockSize:
      let a = predict(clean, r, base - 1 + j).token
      let b = predict(noisy, r, base - 1 + j).token
      let truth = windows[r][base + j]
      inc result.total
      if a == b: inc result.agree
      if a == truth: inc result.cleanCorrect
      if b == truth: inc result.noisyCorrect

proc firstDifference*(a, b: seq[int]): int =
  ## Index of the first differing token, or -1 when the sequences are equal.
  for i in 0 ..< min(a.len, b.len):
    if a[i] != b[i]: return i
  if a.len != b.len: min(a.len, b.len) else: -1

const
  DefaultTokenizer = "test/spectron/data/scale5b/tokenizer.json"
  DefaultPrompts = [
    "Once upon a time, there was a little girl named Lily. She loved to play in the park " &
      "with her friends. One day,",
    "The United States is a country in North America. It is made up of fifty states and a " &
      "federal district. The capital is",
    "proc add(a, b: int): int =\n  ## Returns the sum of a and b.\n  result = a + b\n\n" &
      "proc sub(a, b: int): int =\n",
    "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a + b\n\n\n" &
      "def multiply(a, b):\n",
    "import os\nimport sys\n\n\ndef main():\n    path = sys.argv[1]\n    if not os.path.exists(path):\n",
    "The history of the city began in the twelfth century, when a small market town grew " &
      "around the river crossing. By the year",
    "function greet(name) {\n  return \"Hello, \" + name + \"!\";\n}\n\nconsole.log(greet(",
    "Tom and his dog went for a walk. The sun was shining and the birds were singing. " &
      "Suddenly, Tom saw"]
  Usage = """
Usage: block_decode --checkpoint=PATH [options]
  --tokenizer=PATH      BPE tokenizer (default test/spectron/data/scale5b/tokenizer.json)
  --block=16            Positions decoded in parallel per block
  --tokens=64           New tokens per prompt (a multiple of --block)
  --context=128         Window: the last context-block tokens plus the block
  --tau=0.9             Confidence needed to lock a draft (0.5 < tau <= 1)
  --policy=confidence   confidence (default), jacobi or both
  --prompt=TEXT         A prompt (repeatable; default: built-in English and code prompts)
  --trace-blocks=1      Print every pass of the first prompt for this many blocks
  --seed=7              Seed for the initial drafts
  --probe-text=PATH     Instead of decoding, cut this text file (repeatable) into
                        windows of --context tokens and measure how often block
                        predictions stay the same when drafts are replaced
  --probe-windows=64    Windows per probe file
  --backend=NAME        Backend (default from the build)
"""

proc printSummary(label: string, stats: BlockDecodeStats, greedy: BlockDecodeStats,
                  rows: int, identical: string) =
  let generated = stats.accepted + stats.written
  let perRow = float64(generated) / float64(stats.rowPasses)
  echo &"[{label}] forwards={stats.forwards} tokens_per_forward=" &
       &"{float64(generated) / float64(rows * stats.forwards):.2f} " &
       &"row_passes={stats.rowPasses} tokens_per_row_pass={perRow:.2f} " &
       &"forward_reduction={float64(greedy.forwards) / float64(stats.forwards):.2f}x " &
       &"accepted_drafts={stats.accepted} written={stats.written} " &
       &"locks={stats.locks} right_locks={stats.rightLocks} wrong_locks={stats.wrongLocks} " &
       &"seconds={stats.seconds:.2f} same_as_greedy={identical}"

when isMainModule:
  var checkpoint, backendName = ""
  var tokenizerPath = DefaultTokenizer
  var blockSize = 16
  var newTokens = 64
  var context = 128
  var tau = 0.9'f32
  var policies = @[bpConfidence]
  var texts: seq[string]
  var traceBlocks = 1
  var seed = 7
  var probeFiles: seq[string]
  var probeWindows = 64
  for kind, key, val in getopt():
    if kind == cmdLongOption and key == "help": quit(Usage, 0)
    if kind != cmdLongOption or val.len == 0: quit("Expected --name=value. See --help.", 1)
    case key
    of "checkpoint": checkpoint = val
    of "tokenizer": tokenizerPath = val
    of "block": blockSize = parseInt(val)
    of "tokens": newTokens = parseInt(val)
    of "context": context = parseInt(val)
    of "tau": tau = float32(parseFloat(val))
    of "policy":
      policies = case val
                 of "both": @[bpJacobi, bpConfidence]
                 of "jacobi": @[bpJacobi]
                 of "confidence": @[bpConfidence]
                 else: quit("--policy expects jacobi, confidence or both", 1)
    of "prompt": texts.add(val.replace("\\n", "\n"))
    of "trace-blocks": traceBlocks = parseInt(val)
    of "seed": seed = parseInt(val)
    of "probe-text": probeFiles.add(val)
    of "probe-windows": probeWindows = parseInt(val)
    of "backend": backendName = val
    else: quit("Unknown option --" & key, 1)
  if checkpoint.len == 0: quit(Usage, 1)
  if texts.len == 0: texts = @DefaultPrompts
  setForbidCpuFallback(true)
  let tok = loadBPE(tokenizerPath)
  let (model, step) = loadCheckpoint(checkpoint)
  if not model.matchesBpe(tok.vocabSize):
    quit("Checkpoint vocabulary differs from the tokenizer", 1)
  if probeFiles.len > 0:
    let ctx = if backendName.len > 0: newBackend(backendName) else: newBackend()
    echo &"[probe] checkpoint={checkpoint} step={step} context={context} block={blockSize}"
    for path in probeFiles:
      let ids = tok.encodeBPE(readFile(path))
      var windows: seq[seq[int]]
      var start = 0
      while start + context <= ids.len and windows.len < probeWindows:
        windows.add(ids[start ..< start + context])
        start += context
      if windows.len == 0: quit("Probe file is shorter than --context tokens: " & path, 1)
      for noise in [0.5'f32, 1'f32]:
        let r = model.draftProbe(windows, blockSize, noise, seed, ctx)
        echo &"[probe] file={path} windows={windows.len} draft_noise={noise} " &
             &"unchanged_predictions={100.0 * float64(r.agree) / float64(r.total):.1f}% " &
             &"clean_top1={100.0 * float64(r.cleanCorrect) / float64(r.total):.1f}% " &
             &"noisy_top1={100.0 * float64(r.noisyCorrect) / float64(r.total):.1f}% " &
             &"positions={r.total}"
    closeBackend(ctx)
    quit(0)
  var prompts: seq[seq[int]]
  for text in texts: prompts.add(tok.encodeBPE(text))
  var shortest = high(int)
  for prompt in prompts: shortest = min(shortest, prompt.len)
  let keep = min(shortest, context - blockSize)
  for prompt in prompts.mitems: prompt = prompt[^keep .. ^1]
  let ctx = if backendName.len > 0: newBackend(backendName) else: newBackend()
  echo &"[block-decode] checkpoint={checkpoint} step={step} vocab={model.vocab} " &
       &"hrm={model.hrmEnabled} experts={model.expertCount} eos={model.eosTokenId} rows={prompts.len} " &
       &"prompt_tokens={keep} new_tokens={newTokens} block={blockSize} context={context} tau={tau}"
  # Compile every window shape once so the timings measure warm kernels.
  block warmup:
    var length = keep
    for _ in 0 ..< newTokens div blockSize:
      let width = length - windowStart(length, context, blockSize) + blockSize
      var windows = newSeq[seq[int]](prompts.len)
      for r in 0 ..< prompts.len: windows[r] = newSeq[int](width)
      discard model.forwardBatch(windows, ctx)
      length += blockSize
  var greedyStats: BlockDecodeStats
  var margins: seq[seq[float32]]
  let greedy = model.greedyBlocks(prompts, newTokens, context, blockSize, ctx, greedyStats, margins)
  echo &"[greedy] forwards={greedyStats.forwards} " &
       &"tokens_per_forward={float64(greedyStats.written) / float64(prompts.len * greedyStats.forwards):.2f} " &
       &"seconds={greedyStats.seconds:.2f}"
  let eos = model.eosTokenId
  let tokenText: TokenText = proc (id: int): string =
    if id == eos: "<EOS>" else: tok.decodeBPEStr(@[id])
  for policy in policies:
    var stats: BlockDecodeStats
    let decoded = model.blockDecode(prompts, newTokens, context, blockSize, policy, tau,
                                    seed, ctx, stats, tokenText, traceBlocks)
    var same = 0
    for r in 0 ..< prompts.len:
      let at = firstDifference(decoded[r], greedy[r])
      if at < 0:
        inc same
      else:
        let index = at - keep
        let margin = if index in 0 ..< margins[r].len: &"{margins[r][index]:.6f}" else: "n/a"
        echo &"[{policy}] row={r} differs from greedy at new token {index} greedy_margin={margin}"
    printSummary($policy, stats, greedyStats, prompts.len, &"{same}/{prompts.len}")
    for r in 0 ..< prompts.len:
      echo &"[passes] {policy} row={r} per_block={stats.blockPasses[r]}"
  for r in 0 ..< prompts.len:
    echo &"[prompt] row={r} " & tok.decodeBPEStr(prompts[r]).multiReplace(("\n", "\\n"))
    let stopped = greedy[r][^1] == eos
    let count = greedy[r].len - keep - ord(stopped)
    echo &"[sample] row={r} tokens={count} eos={stopped} " &
         tok.decodeBPEStr(greedy[r][keep .. ^1]).multiReplace(("\n", "\\n"))
  closeBackend(ctx)
