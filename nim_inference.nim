## nim_inference.nim
## Inference cho DeepSeek-Coder (LLaMA arch) dùng Metal
## Load model từ .nimq (FP16 weight), tokenizer từ Hugging Face qua nimpy

import std/[math, random, sequtils, strformat, times, options, algorithm]
import nimpy
import customfloat, quant, metal_ai

# ═══════════════════════════════════════════════════════════════
# 1. Tokenizer wrapper (gọi Hugging Face tokenizer qua nimpy)
# ═══════════════════════════════════════════════════════════════
type
  HFTokenizer* = object
    tok: PyObject
    eos_id*, pad_id*, vocab_size*: int
    bos_id*, unk_id*: int

proc newHFTokenizer*(model_path: string): HFTokenizer =
  let transformers = pyImport("transformers")
  result.tok = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=true)
  result.eos_id = result.tok.eos_token_id.to(int)
  result.pad_id = result.tok.pad_token_id.to(int)
  result.bos_id = result.tok.bos_token_id.to(int)
  result.unk_id = result.tok.unk_token_id.to(int)
  result.vocab_size = result.tok.vocab_size.to(int)

proc encode*(tok: HFTokenizer, text: string, add_special_tokens: bool = true): seq[int] =
  let encoded = tok.tok.encode(text, add_special_tokens = add_special_tokens)
  result = encoded.to(seq[int])

proc decode*(tok: HFTokenizer, ids: seq[int], skip_special_tokens: bool = true): string =
  let decoded = tok.tok.decode(ids, skip_special_tokens = skip_special_tokens)
  result = decoded.to(string)

# ═══════════════════════════════════════════════════════════════
# 2. Config
# ═══════════════════════════════════════════════════════════════
type
  LlamaConfig* = object
    vocab_size, hidden_size, intermediate_size: int
    num_hidden_layers, num_attention_heads, num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps, rope_theta: float32
    tie_word_embeddings: bool

proc newLlamaConfig*(model_path: string): LlamaConfig =
  let transformers = pyImport("transformers")
  let config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=true)
  result.vocab_size = config.vocab_size.to(int)
  result.hidden_size = config.hidden_size.to(int)
  result.intermediate_size = config.intermediate_size.to(int)
  result.num_hidden_layers = config.num_hidden_layers.to(int)
  result.num_attention_heads = config.num_attention_heads.to(int)
  result.num_key_value_heads = config.num_key_value_heads.to(int)
  result.max_position_embeddings = config.max_position_embeddings.to(int)
  result.rms_norm_eps = config.rms_norm_eps.to(float32)
  result.rope_theta = config.rope_theta.to(float32)
  result.tie_word_embeddings = config.tie_word_embeddings.to(bool)

# ═══════════════════════════════════════════════════════════════
# 3. Helper functions
# ═══════════════════════════════════════════════════════════════
proc addT*(a, b: seq[float32]): seq[float32] =
  assert a.len == b.len
  result = newSeq[float32](a.len)
  for i in 0 ..< a.len:
    result[i] = a[i] + b[i]

proc softmaxInplace*(arr: var seq[float32]) =
  var mx = arr[0]
  for v in arr:
    if v > mx: mx = v
  var s = 0'f32
  for i in 0 ..< arr.len:
    arr[i] = exp(arr[i] - mx)
    s += arr[i]
  for i in 0 ..< arr.len:
    arr[i] /= s

proc sampleTopP*(logits: seq[float32], temperature: float32 = 0.7, top_p: float32 = 0.9): int =
  var probs = logits
  if temperature > 0:
    for i in 0 ..< probs.len:
      probs[i] = exp(probs[i] / temperature)
    let sum_exp = probs.sum()
    for i in 0 ..< probs.len:
      probs[i] /= sum_exp
  else:
    var max_idx = 0
    var max_val = probs[0]
    for i in 1 ..< probs.len:
      if probs[i] > max_val:
        max_val = probs[i]
        max_idx = i
    return max_idx
  var indices = toSeq(0 ..< probs.len)
  sort(indices, proc(a, b: int): int = cmp(probs[b], probs[a]))
  var cumsum = 0'f32
  var keep = newSeq[bool](probs.len)
  for idx in indices:
    cumsum += probs[idx]
    if cumsum < top_p:
      keep[idx] = true
    else:
      keep[idx] = false
      break
  var filtered_probs: seq[float32] = @[]
  var filtered_indices: seq[int] = @[]
  for i in 0 ..< probs.len:
    if keep[i]:
      filtered_probs.add(probs[i])
      filtered_indices.add(i)
  if filtered_probs.len == 0:
    return indices[0]
  let sum_f = filtered_probs.sum()
  for i in 0 ..< filtered_probs.len:
    filtered_probs[i] /= sum_f
  var r = rand(1.0)
  for i in 0 ..< filtered_probs.len:
    r -= filtered_probs[i]
    if r <= 0:
      return filtered_indices[i]
  return filtered_indices[^1]

# ═══════════════════════════════════════════════════════════════
# 4. Các lớp (LLaMA arch)
# ═══════════════════════════════════════════════════════════════

# ─── Linear ───
type Linear* = object
  weight*: seq[float32]   # [in, out] (đã transpose)
  bias*: seq[float32]
  in_features*, out_features*: int

proc newLinear*(inF, outF: int, weight: seq[float32], bias: seq[float32] = @[]): Linear =
  result.in_features = inF
  result.out_features = outF
  result.weight = weight
  result.bias = bias

proc forwardLinear*(l: Linear, x: seq[float32], ctx: MetalContext): seq[float32] =
  let rows = x.len div l.in_features
  var y = metalMatmul(ctx, x, rows, l.in_features, l.weight, l.in_features, l.out_features)
  if l.bias.len > 0:
    for i in 0 ..< y.len:
      y[i] += l.bias[i mod l.out_features]
  result = y

# ─── RMSNorm ───
type RMSNorm* = object
  weight*: seq[float32]
  eps*: float32

proc newRMSNorm*(dim: int, eps: float32 = 1e-6): RMSNorm =
  result.weight = newSeq[float32](dim)
  for i in 0 ..< dim: result.weight[i] = 1.0
  result.eps = eps

proc forwardRMSNorm*(ln: RMSNorm, x: seq[float32], cols: int): seq[float32] =
  let rows = x.len div cols
  result = newSeq[float32](x.len)
  for r in 0 ..< rows:
    let off = r * cols
    var sq_sum = 0.0
    for c in 0 ..< cols:
      let v = x[off + c]
      sq_sum += v * v
    let rms = sqrt(sq_sum / float32(cols) + ln.eps)
    let inv = 1.0 / rms
    for c in 0 ..< cols:
      result[off + c] = x[off + c] * inv * ln.weight[c]

# ─── SiLU ───
proc silu(x: float32): float32 = x / (1.0 + exp(-x))

proc siluActivation*(x: seq[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len:
    result[i] = silu(x[i])

# ─── RoPE ───
type RoPE* = object
  cos*, sin*: seq[float32]
  max_seq_len*, head_dim*: int

proc newRoPE*(head_dim: int, max_seq_len: int, theta: float32 = 10000.0): RoPE =
  result.head_dim = head_dim
  result.max_seq_len = max_seq_len
  let half = head_dim div 2
  var inv_freq = newSeq[float32](half)
  for i in 0 ..< half:
    inv_freq[i] = 1.0 / (theta ^ (float32(i) / float32(half)))
  result.cos = newSeq[float32](max_seq_len * half)
  result.sin = newSeq[float32](max_seq_len * half)
  for pos in 0 ..< max_seq_len:
    for i in 0 ..< half:
      let angle = float32(pos) * inv_freq[i]
      result.cos[pos * half + i] = cos(angle)
      result.sin[pos * half + i] = sin(angle)

proc applyRoPE*(rope: RoPE, q: seq[float32], k: seq[float32],
                 B, T, H, D: int): tuple[q_rot, k_rot: seq[float32]] =
  let half = D div 2
  var q_rot = newSeq[float32](q.len)
  var k_rot = newSeq[float32](k.len)
  for b in 0 ..< B:
    for h in 0 ..< H:
      let base_q = ((b * T + 0) * H * D) + h * D
      let base_k = ((b * T + 0) * H * D) + h * D
      for t in 0 ..< T:
        let off_cos = t * half
        let q_pos = base_q + t * H * D
        let k_pos = base_k + t * H * D
        for d in 0 ..< half:
          let idx1 = q_pos + d
          let idx2 = q_pos + d + half
          let cos_v = rope.cos[off_cos + d]
          let sin_v = rope.sin[off_cos + d]
          q_rot[idx1] = q[idx1] * cos_v - q[idx2] * sin_v
          q_rot[idx2] = q[idx1] * sin_v + q[idx2] * cos_v
        for d in 0 ..< half:
          let idx1 = k_pos + d
          let idx2 = k_pos + d + half
          let cos_v = rope.cos[off_cos + d]
          let sin_v = rope.sin[off_cos + d]
          k_rot[idx1] = k[idx1] * cos_v - k[idx2] * sin_v
          k_rot[idx2] = k[idx1] * sin_v + k[idx2] * cos_v
  result = (q_rot, k_rot)

# ─── MultiHeadAttention (GQA) ───
type MultiHeadAttention* = object
  n_heads*, n_kv_heads*, head_dim*, hidden_size*: int
  q_proj*, k_proj*, v_proj*, o_proj*: Linear
  rope*: RoPE
  scale*: float32

proc newMultiHeadAttention*(cfg: LlamaConfig, rope: RoPE): MultiHeadAttention =
  result.n_heads = cfg.num_attention_heads
  result.n_kv_heads = cfg.num_key_value_heads
  result.head_dim = cfg.hidden_size div cfg.num_attention_heads
  result.hidden_size = cfg.hidden_size
  result.rope = rope
  result.scale = 1.0 / sqrt(float32(result.head_dim))
  # Các proj sẽ được gán sau khi load weight

proc reshapeForAttention*(x: seq[float32], B, T, H, D: int): seq[float32] =
  result = newSeq[float32](B * T * H * D)
  for b in 0 ..< B:
    for t in 0 ..< T:
      for h in 0 ..< H:
        for d in 0 ..< D:
          let src = (b * T + t) * (H * D) + h * D + d
          let dst = ((b * T + t) * H + h) * D + d
          result[dst] = x[src]

# Chú ý: Hàm forwardMHA này dùng CPU cho attention (có thể dùng Metal sau)
proc forwardMHA*(mha: MultiHeadAttention, x: seq[float32], ctx: MetalContext,
                  B, T, C: int): seq[float32] =
  let head_dim = mha.head_dim
  let n_heads = mha.n_heads
  let n_kv_heads = mha.n_kv_heads
  let hidden = mha.hidden_size

  let q = forwardLinear(mha.q_proj, x, ctx)  # [B*T, hidden]
  let k = forwardLinear(mha.k_proj, x, ctx)
  let v = forwardLinear(mha.v_proj, x, ctx)

  var q_reshaped = reshapeForAttention(q, B, T, n_heads, head_dim)
  var k_reshaped = reshapeForAttention(k, B, T, n_kv_heads, head_dim)
  var v_reshaped = reshapeForAttention(v, B, T, n_kv_heads, head_dim)

  let (q_rot, k_rot) = applyRoPE(mha.rope, q_reshaped, k_reshaped, B, T, n_heads, head_dim)
  let repeat_factor = n_heads div n_kv_heads
  var k_rep = newSeq[float32](B * T * n_heads * head_dim)
  var v_rep = newSeq[float32](B * T * n_heads * head_dim)
  for b in 0 ..< B:
    for t in 0 ..< T:
      for h in 0 ..< n_kv_heads:
        for r in 0 ..< repeat_factor:
          let src_off = ((b * T + t) * n_kv_heads + h) * head_dim
          let dst_off = ((b * T + t) * n_heads + (h * repeat_factor + r)) * head_dim
          for d in 0 ..< head_dim:
            k_rep[dst_off + d] = k_rot[src_off + d]
            v_rep[dst_off + d] = v_reshaped[src_off + d]

  # Attention: dùng CPU
  let BH = B * n_heads
  let dim = head_dim
  var q_flat = newSeq[float32](BH * T * dim)
  var k_flat = newSeq[float32](BH * T * dim)
  var v_flat = newSeq[float32](BH * T * dim)
  for bh in 0 ..< BH:
    for t in 0 ..< T:
      for d in 0 ..< dim:
        let idx_q = (bh * T + t) * dim + d
        q_flat[idx_q] = q_rot[(bh * T + t) * dim + d]
        k_flat[idx_q] = k_rep[(bh * T + t) * dim + d]
        v_flat[idx_q] = v_rep[(bh * T + t) * dim + d]

  var scores = newSeq[float32](BH * T * T)
  for bh in 0 ..< BH:
    let base = bh * T * T
    for i in 0 ..< T:
      for j in 0 ..< T:
        var s = 0.0
        for d in 0 ..< dim:
          let q_idx = (bh * T + i) * dim + d
          let k_idx = (bh * T + j) * dim + d
          s += q_flat[q_idx] * k_flat[k_idx]
        scores[base + i * T + j] = s * mha.scale
  # Causal mask
  for bh in 0 ..< BH:
    let base = bh * T * T
    for i in 0 ..< T:
      for j in i + 1 ..< T:
        scores[base + i * T + j] = -1e9
  # Softmax
  for bh in 0 ..< BH:
    let base = bh * T * T
    for i in 0 ..< T:
      var row = scores[base + i * T ..< base + (i+1) * T]
      softmaxInplace(row)
      scores[base + i * T ..< base + (i+1) * T] = row
  # Output
  var out_flat = newSeq[float32](BH * T * dim)
  for bh in 0 ..< BH:
    let base = bh * T * T
    for i in 0 ..< T:
      for d in 0 ..< dim:
        var s = 0.0
        for j in 0 ..< T:
          s += scores[base + i * T + j] * v_flat[(bh * T + j) * dim + d]
        out_flat[(bh * T + i) * dim + d] = s
  var out = newSeq[float32](B * T * hidden)
  for b in 0 ..< B:
    for t in 0 ..< T:
      for h in 0 ..< n_heads:
        let src = ((b * n_heads + h) * T + t) * dim
        let dst = (b * T + t) * hidden + h * dim
        for d in 0 ..< dim:
          out[dst + d] = out_flat[src + d]
  result = forwardLinear(mha.o_proj, out, ctx)

# ─── SwiGLU ───
type SwiGLU* = object
  gate_proj*, up_proj*, down_proj*: Linear

proc forwardSwiGLU*(ff: SwiGLU, x: seq[float32], ctx: MetalContext): seq[float32] =
  let gate = forwardLinear(ff.gate_proj, x, ctx)
  let up = forwardLinear(ff.up_proj, x, ctx)
  var gate_act = siluActivation(gate)
  var hidden = newSeq[float32](gate_act.len)
  for i in 0 ..< hidden.len:
    hidden[i] = gate_act[i] * up[i]
  result = forwardLinear(ff.down_proj, hidden, ctx)

# ─── TransformerBlock ───
type TransformerBlock* = object
  input_layernorm*, post_attention_layernorm*: RMSNorm
  self_attn*: MultiHeadAttention
  mlp*: SwiGLU

proc forwardBlock*(blk: TransformerBlock, x: seq[float32], ctx: MetalContext,
                    B, T, C: int): seq[float32] =
  let norm1 = forwardRMSNorm(blk.input_layernorm, x, C)
  let attn_out = forwardMHA(blk.self_attn, norm1, ctx, B, T, C)
  let x1 = addT(x, attn_out)
  let norm2 = forwardRMSNorm(blk.post_attention_layernorm, x1, C)
  let ff_out = forwardSwiGLU(blk.mlp, norm2, ctx)
  result = addT(x1, ff_out)

# ─── LlamaModel ───
type LlamaModel* = object
  config*: LlamaConfig
  embed_tokens*: seq[float32]
  layers*: seq[TransformerBlock]
  norm*: RMSNorm
  lm_head*: Linear

proc forwardLlama*(model: LlamaModel, input_ids: seq[int], ctx: MetalContext): seq[float32] =
  let B = 1
  let T = input_ids.len
  let C = model.config.hidden_size
  let vocab = model.config.vocab_size
  var x = newSeq[float32](B * T * C)
  for t, id in input_ids:
    let src = id * C
    let dst = t * C
    for c in 0 ..< C:
      x[dst + c] = model.embed_tokens[src + c]
  for blk in model.layers:
    x = forwardBlock(blk, x, ctx, B, T, C)
  let norm_out = forwardRMSNorm(model.norm, x, C)
  result = forwardLinear(model.lm_head, norm_out, ctx)

# ═══════════════════════════════════════════════════════════════
# 5. Load model từ .nimq
# ═══════════════════════════════════════════════════════════════

proc loadLlamaModel*(path: string, config: LlamaConfig): LlamaModel =
  let (arch, sd) = loadQuantStateDict(path)
  var byName = initTable[string, QuantTensor]()
  for (name, qt) in sd: byName[name] = qt

  proc loadTensor(name: string): seq[float32] =
    if not byName.hasKey(name): return @[]
    dequantizeTensor(byName[name])

  result.config = config
  let hidden = config.hidden_size
  let vocab = config.vocab_size
  let n_heads = config.num_attention_heads
  let n_kv_heads = config.num_key_value_heads
  let n_layers = config.num_hidden_layers
  let intermediate = config.intermediate_size
  let eps = config.rms_norm_eps
  let rope_theta = config.rope_theta

  result.embed_tokens = loadTensor("model.embed_tokens.weight")
  if result.embed_tokens.len == 0:
    result.embed_tokens = loadTensor("embed_tokens.weight")

  var norm_w = loadTensor("model.norm.weight")
  if norm_w.len == 0: norm_w = loadTensor("norm.weight")
  result.norm = newRMSNorm(hidden, eps)
  result.norm.weight = norm_w

  result.layers = newSeq[TransformerBlock](n_layers)
  let rope = newRoPE(hidden div n_heads, config.max_position_embeddings, rope_theta)
  for l in 0 ..< n_layers:
    let prefix = "model.layers." & $l
    var blk: TransformerBlock
    var ln1_w = loadTensor(prefix & ".input_layernorm.weight")
    var ln2_w = loadTensor(prefix & ".post_attention_layernorm.weight")
    if ln1_w.len == 0:
      ln1_w = loadTensor(prefix & ".ln1.weight")
      ln2_w = loadTensor(prefix & ".ln2.weight")
    blk.input_layernorm = newRMSNorm(hidden, eps)
    blk.input_layernorm.weight = ln1_w
    blk.post_attention_layernorm = newRMSNorm(hidden, eps)
    blk.post_attention_layernorm.weight = ln2_w

    proc loadLinear(name_prefix: string, inF, outF: int): Linear =
      var w = loadTensor(name_prefix & ".weight")
      if w.len == 0: w = loadTensor(name_prefix & ".weight")
      var wT = newSeq[float32](inF * outF)
      for o in 0 ..< outF:
        for i in 0 ..< inF:
          wT[i * outF + o] = w[o * inF + i]
      var b = loadTensor(name_prefix & ".bias")
      if b.len == 0: b = newSeq[float32](outF)
      return newLinear(inF, outF, wT, b)

    var attn = newMultiHeadAttention(config, rope)
    attn.q_proj = loadLinear(prefix & ".self_attn.q_proj", hidden, hidden)
    attn.k_proj = loadLinear(prefix & ".self_attn.k_proj", hidden, n_kv_heads * (hidden div n_heads))
    attn.v_proj = loadLinear(prefix & ".self_attn.v_proj", hidden, n_kv_heads * (hidden div n_heads))
    attn.o_proj = loadLinear(prefix & ".self_attn.o_proj", hidden, hidden)
    blk.self_attn = attn

    var gate = loadLinear(prefix & ".mlp.gate_proj", hidden, intermediate)
    var up   = loadLinear(prefix & ".mlp.up_proj", hidden, intermediate)
    var down = loadLinear(prefix & ".mlp.down_proj", intermediate, hidden)
    blk.mlp = SwiGLU(gate_proj: gate, up_proj: up, down_proj: down)
    result.layers[l] = blk

  var lm_w = loadTensor("lm_head.weight")
  if lm_w.len == 0: lm_w = loadTensor("model.lm_head.weight")
  if lm_w.len == 0: lm_w = result.embed_tokens
  let lm_wT = newSeq[float32](hidden * vocab)
  for v in 0 ..< vocab:
    for h in 0 ..< hidden:
      lm_wT[h * vocab + v] = lm_w[v * hidden + h]
  result.lm_head = newLinear(hidden, vocab, lm_wT, newSeq[float32](vocab))

# ═══════════════════════════════════════════════════════════════
# 6. Generation
# ═══════════════════════════════════════════════════════════════

proc generate*(model: LlamaModel, tokenizer: HFTokenizer, ctx: MetalContext,
                prompt: string, max_new_tokens: int = 128,
                temperature: float32 = 0.7, top_p: float32 = 0.9): string =
  var ids = tokenizer.encode(prompt, add_special_tokens=true)
  for _ in 0 ..< max_new_tokens:
    let logits = forwardLlama(model, ids, ctx)
    let last_logits = logits[^tokenizer.vocab_size .. ^1]
    let next_id = sampleTopP(last_logits, temperature, top_p)
    ids.add(next_id)
    if next_id == tokenizer.eos_id: break
  result = tokenizer.decode(ids, skip_special_tokens=true)

# ═══════════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════════

when isMainModule:
  let model_path = "path/to/deepseek-coder"
  echo "Loading config..."
  let config = newLlamaConfig(model_path)
  echo "Loading tokenizer..."
  let tokenizer = newHFTokenizer(model_path)
  echo "Loading model..."
  let ctx = newMetalContext()
  var model = loadLlamaModel("exported/deepseek_6.7b.nimq", config)
  echo "Generating..."
  let prompt = "def fib(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    "
  let output = generate(model, tokenizer, ctx, prompt, max_new_tokens=64)
  echo "Generated:\n", output