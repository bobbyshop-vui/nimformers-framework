## Small model-builder API for recording a decoder trace.
##
## The classes below run through the same session API as the production
## runtimes, so one forward call is one recorded step and `firlift` can rebuild
## model.fir from it. Kernels come from the trace lifter's library; the builder
## itself carries no model-specific code.
import std/[math]
import gpubackend as byby
import probe

type
  BuilderConfig* = object
    hidden*, layers*, heads*, kvHeads*, mlp*, vocab*, positions*: int
    eps*: float32

  BLinear* = object
    w*: byby.SessionHandle
    outF*, inF*: int
    name*: string

  BRmsNorm* = object
    gamma*: byby.SessionHandle
    dim*: int
    eps*: float32
    name*: string

  BModel* = object
    be*: byby.GpuBackend
    cfg*: BuilderConfig
    embedTable*: seq[float32]
    layers*: seq[tuple[
      inNorm: BRmsNorm, postNorm: BRmsNorm,
      q, k, v, o, gate, up, down: BLinear]]
    finalNorm*: BRmsNorm
    lmHead*: BLinear
    keyCache*, valueCache: seq[byby.SessionHandle]
    scores*: byby.SessionHandle
    # Reused activation scratch (one set for every layer).
    norm, norm2, qOut, kOut, vOut, attn, oOut, res, gateOut, upOut,
      sig, silu, act, downOut, xOut, normed, logits: byby.SessionHandle

proc uploadF32(be: byby.GpuBackend; values: openArray[float32]): byby.SessionHandle =
  byby.sessionUploadPrivate(be, values)

proc newBLinear*(be: byby.GpuBackend; name: string; outF, inF: int): BLinear =
  result.name = name
  result.outF = outF
  result.inF = inF
  result.w = uploadF32(be, newSeq[float32](outF*inF))
  byby.probeName(result.w, name, "f32", [outF, inF])

proc forward*(l: BLinear; x: byby.SessionHandle; y: var byby.SessionHandle; be: byby.GpuBackend) =
  discard byby.sessionMatmulNT(be, x, l.w, y, 1, l.inF, l.outF)

proc newBRmsNorm*(be: byby.GpuBackend; name: string; dim: int; eps: float32): BRmsNorm =
  result.name = name
  result.dim = dim
  result.eps = eps
  var gamma = newSeq[float32](dim)
  for i in 0 ..< dim: gamma[i] = 1.0'f32
  result.gamma = uploadF32(be, gamma)
  byby.probeName(result.gamma, name, "f32", [dim])

proc forward*(r: BRmsNorm; x: byby.SessionHandle; y: var byby.SessionHandle; be: byby.GpuBackend) =
  discard byby.sessionRmsnormEnc(be, x, r.gamma, y, 1, r.dim, r.eps)

proc linearName(layer: int; part: string): string =
  "model.layers." & $layer & "." & part & ".weight"

proc newBuilderModel*(be: byby.GpuBackend; cfg: BuilderConfig): BModel =
  result.be = be
  result.cfg = cfg
  result.embedTable = newSeq[float32](cfg.vocab*cfg.hidden)
  var seed = 1'f32
  for i in 0 ..< result.embedTable.len:
    seed = seed * 1.0000001'f32 + 0.001'f32
    result.embedTable[i] = seed * 0.01'f32
  result.layers.setLen(cfg.layers)
  for i in 0 ..< cfg.layers:
    result.layers[i].inNorm = newBRmsNorm(be, linearName(i, "input_layernorm"), cfg.hidden, cfg.eps)
    result.layers[i].postNorm = newBRmsNorm(be, linearName(i, "post_attention_layernorm"), cfg.hidden, cfg.eps)
    result.layers[i].q = newBLinear(be, linearName(i, "self_attn.q_proj"), cfg.hidden, cfg.hidden)
    let kv = cfg.kvHeads * (cfg.hidden div cfg.heads)
    result.layers[i].k = newBLinear(be, linearName(i, "self_attn.k_proj"), kv, cfg.hidden)
    result.layers[i].v = newBLinear(be, linearName(i, "self_attn.v_proj"), kv, cfg.hidden)
    result.layers[i].o = newBLinear(be, linearName(i, "self_attn.o_proj"), cfg.hidden, cfg.hidden)
    result.layers[i].gate = newBLinear(be, linearName(i, "mlp.gate_proj"), cfg.mlp, cfg.hidden)
    result.layers[i].up = newBLinear(be, linearName(i, "mlp.up_proj"), cfg.mlp, cfg.hidden)
    result.layers[i].down = newBLinear(be, linearName(i, "mlp.down_proj"), cfg.hidden, cfg.mlp)
    result.keyCache.add byby.sessionAllocScratch(be, cfg.kvHeads*cfg.positions*(cfg.hidden div cfg.heads))
    result.valueCache.add byby.sessionAllocScratch(be, cfg.kvHeads*cfg.positions*(cfg.hidden div cfg.heads))
  result.finalNorm = newBRmsNorm(be, "model.norm.weight", cfg.hidden, cfg.eps)
  result.lmHead = newBLinear(be, "lm_head.weight", cfg.vocab, cfg.hidden)
  result.scores = byby.sessionAllocScratch(be, cfg.heads*cfg.positions + cfg.hidden)
  result.norm = byby.sessionAllocScratch(be, cfg.hidden)
  result.norm2 = byby.sessionAllocScratch(be, cfg.hidden)
  result.qOut = byby.sessionAllocScratch(be, cfg.hidden)
  result.kOut = byby.sessionAllocScratch(be, cfg.kvHeads*(cfg.hidden div cfg.heads))
  result.vOut = byby.sessionAllocScratch(be, cfg.kvHeads*(cfg.hidden div cfg.heads))
  result.attn = byby.sessionAllocScratch(be, cfg.hidden)
  result.oOut = byby.sessionAllocScratch(be, cfg.hidden)
  result.res = byby.sessionAllocScratch(be, cfg.hidden)
  result.gateOut = byby.sessionAllocScratch(be, cfg.mlp)
  result.upOut = byby.sessionAllocScratch(be, cfg.mlp)
  result.sig = byby.sessionAllocScratch(be, cfg.mlp)
  result.silu = byby.sessionAllocScratch(be, cfg.mlp)
  result.act = byby.sessionAllocScratch(be, cfg.mlp)
  result.downOut = byby.sessionAllocScratch(be, cfg.hidden)
  result.xOut = byby.sessionAllocScratch(be, cfg.hidden)
  result.normed = byby.sessionAllocScratch(be, cfg.hidden)
  result.logits = byby.sessionAllocScratch(be, cfg.vocab)

proc forwardLayer(m: var BModel; layer: int; x: byby.SessionHandle; position: int) =
  let be = m.be
  let c = m.cfg
  let headDim = c.hidden div c.heads
  let kvDim = c.kvHeads * headDim
  m.layers[layer].inNorm.forward(x, m.norm, be)
  m.layers[layer].q.forward(m.norm, m.qOut, be)
  m.layers[layer].k.forward(m.norm, m.kOut, be)
  m.layers[layer].v.forward(m.norm, m.vOut, be)
  # The library attention kernel writes the cache row itself; the builder keeps
  # the same writes so its own run has a filled cache.
  for h in 0 ..< c.kvHeads:
    discard byby.sessionCopyInto(be, m.kOut, m.keyCache[layer],
      h*headDim, h*c.positions*headDim + position*headDim, headDim)
    discard byby.sessionCopyInto(be, m.vOut, m.valueCache[layer],
      h*headDim, h*c.positions*headDim + position*headDim, headDim)
  discard byby.sessionAttentionDecodeAllHeadsEnc(be, m.qOut, m.keyCache[layer],
    m.valueCache[layer], m.attn, c.heads, c.kvHeads, headDim, position+1, c.positions)
  m.layers[layer].o.forward(m.attn, m.oOut, be)
  discard byby.sessionVecOp(be, "add", x, m.oOut, m.res, m.cfg.hidden)
  m.layers[layer].postNorm.forward(m.res, m.norm2, be)
  m.layers[layer].gate.forward(m.norm2, m.gateOut, be)
  m.layers[layer].up.forward(m.norm2, m.upOut, be)
  discard byby.sessionActivation(be, "sigmoid", m.gateOut, m.sig, m.cfg.mlp)
  discard byby.sessionVecOp(be, "mul", m.gateOut, m.sig, m.silu, m.cfg.mlp)
  discard byby.sessionVecOp(be, "mul", m.silu, m.upOut, m.act, m.cfg.mlp)
  m.layers[layer].down.forward(m.act, m.downOut, be)
  discard byby.sessionVecOp(be, "add", m.res, m.downOut, m.xOut, m.cfg.hidden)

proc forwardStep*(m: var BModel; token, position: int): seq[float32] =
  let be = m.be
  let c = m.cfg
  var row = newSeq[float32](c.hidden)
  for i in 0 ..< c.hidden:
    row[i] = m.embedTable[token*c.hidden + i]
  if not byby.sessionBegin(be):
    raise newException(ValueError, "builder sessionBegin failed")
  var ran = false
  try:
    let x0 = byby.sessionUpload(be, row)
    var x = x0
    for layer in 0 ..< c.layers:
      m.forwardLayer(layer, x, position)
      x = m.xOut
    m.finalNorm.forward(m.xOut, m.normed, be)
    m.lmHead.forward(m.normed, m.logits, be)
    ran = true
    if not byby.sessionEnd(be):
      raise newException(ValueError, "builder sessionEnd failed")
    result = byby.sessionRead(be, m.logits, c.vocab)
  finally:
    if not ran: discard byby.sessionEnd(be)

proc beginBuilderTrace*(tracePath: string) =
  ## Start the recorder before constructing the model, so weight names and
  ## allocations are part of the trace.
  byby.firProbeStart(tracePath, "")

proc traceModel*(m: var BModel; token, position: int) =
  ## Record exactly one decode step. Call beginBuilderTrace before building.
  if not byby.firProbeOn:
    raise newException(ValueError, "beginBuilderTrace must run before newBuilderModel")
  byby.probeStepBegin("decode", 0, position, token)
  try:
    discard m.forwardStep(token, position)
  finally:
    byby.probeStepEnd()
