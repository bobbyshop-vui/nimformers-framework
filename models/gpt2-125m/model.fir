# Resident GPT-2 decode. Q4 source shapes follow HF Conv1D [input,output];
# the loader transposes to [output,input] before grouped asymmetric int4 quantization.
# wte/lm_head stays tied fp32; --fir-f32 restores all fp32 linears.
fir 1
model gpt2
config config.json
runtime graph
epsilon layer_norm_epsilon
prefix transformer.

# Config checks owned by this model file (a key that is absent passes).
require config model_type gpt2
require config activation_function gelu_new
require config scale_attn_weights true
require config scale_attn_by_inverse_layer_idx false
require config reorder_and_upcast_attn false
require config add_cross_attention false

# Every integer key of config.json is already a symbol (n_embd, n_head, ...).
# The graph runtime needs these names:
sym layers = n_layer
sym positions = n_positions
sym vocab = vocab_size
sym hidden = n_embd
sym nheads = n_head
sym headdim = hidden/nheads
sym mlp = 4*hidden
require sym hidden = nheads*headdim
buf token_id f32 1 input:token
buf position f32 1 input:position
buf x f32 n_embd
buf pos_embed f32 n_embd
buf norm f32 n_embd
buf qkv f32 3*n_embd
buf attn f32 n_embd
buf mix f32 n_embd
buf fc f32 mlp
buf activated f32 mlp
buf scores f32 n_head*n_positions
buf key_cache f32 n_positions*n_embd per_layer
buf value_cache f32 n_positions*n_embd per_layer
buf logits f32 vocab output:logits
weight wte f32 wte.weight tensor [vocab_size,n_embd]
weight wpe f32 wpe.weight tensor [n_positions,n_embd]
weight ln1w f32 h.{i}.ln_1.weight tensor [n_embd]
weight ln1b f32 h.{i}.ln_1.bias tensor [n_embd]
weight qkvw q4 h.{i}.attn.c_attn.weight tensor [n_embd,3*n_embd]
weight qkvb f32 h.{i}.attn.c_attn.bias tensor [3*n_embd]
weight attnw q4 h.{i}.attn.c_proj.weight tensor [n_embd,n_embd]
weight attnb f32 h.{i}.attn.c_proj.bias tensor [n_embd]
weight ln2w f32 h.{i}.ln_2.weight tensor [n_embd]
weight ln2b f32 h.{i}.ln_2.bias tensor [n_embd]
weight fcw q4 h.{i}.mlp.c_fc.weight tensor [n_embd,mlp]
weight fcb f32 h.{i}.mlp.c_fc.bias tensor [mlp]
weight projw q4 h.{i}.mlp.c_proj.weight tensor [mlp,n_embd]
weight projb f32 h.{i}.mlp.c_proj.bias tensor [n_embd]
weight lnfw f32 ln_f.weight tensor [n_embd]
weight lnfb f32 ln_f.bias tensor [n_embd]
# ---- Micro-kernels defined by this model --------------------------------------
# One line = one session_vm emit* call (mnemonic = emit name, lower case, no "emit").
# Registers r are 0..31; @inN / @outN / @buffer are element addresses in the arena.
# R0 holds the lane id on entry (rtou 0 0 copies it to U0).

kernel g2.add in=2 out=1 size=in0,in1,out0 threads=out0.size tg=1 tgbytes=16
  rtou 0 0
  uloadf32 1 0 @in0
  uloadf32 2 0 @in1
  add 1 1 2
  ustoref32 1 0 @out0
endkernel

# gelu_new(x) = x * sigmoid(2*sqrt(2/pi) * (x + 0.044715*x^3))
kernel g2.gelu_new in=1 out=1 size=in0,out0 threads=out0.size tg=1 tgbytes=16
  rtou 0 0
  uloadf32 1 0 @in0
  mul 2 1 1
  mul 2 2 1
  mulconst 2 2 0.044715
  add 2 2 1
  mulconst 2 2 2*sqrt(2/pi)
  sigmoid 2 2
  mul 1 1 2
  ustoref32 1 0 @out0
endkernel

# LayerNorm over one row: mean (R1), variance (R3), then (x-mean)*rsqrt(var+eps)*gamma+beta.
kernel g2.layernorm in=3 out=1 size=in0,in1,in2,out0 threads=1 tg=1 tgbytes=16
  loadconstf32 15 in0.size
  uconst 0 0
  loadconstf32 1 0
  forbegin 15
    uloadf32 2 0 @in0
    add 1 1 2
    uaddk 0 0 1
  forend
  mulconst 1 1 1/in0.size
  uconst 0 0
  loadconstf32 3 0
  forbegin 15
    uloadf32 2 0 @in0
    sub 2 2 1
    mul 2 2 2
    add 3 3 2
    uaddk 0 0 1
  forend
  mulconst 3 3 1/in0.size
  addconst 3 3 eps
  rsqrt 3 3
  uconst 0 0
  forbegin 15
    uloadf32 2 0 @in0
    uloadf32 4 0 @in1
    uloadf32 5 0 @in2
    sub 2 2 1
    mul 2 2 3
    mul 2 2 4
    add 2 2 5
    ustoref32 2 0 @out0
    uaddk 0 0 1
  forend
endkernel

# One lane per head: write this token's K/V into the cache, then causal softmax
# attention over positions 0..position (nothing after the current token is read).
# ins: qkv position key_cache value_cache scores; outs: attn key_cache value_cache scores.
kernel g2.attn_decode in=5 out=4 threads=nheads tg=1 tgbytes=16 expect=in0:3*hidden,in1:1,in2:positions*hidden,in3:positions*hidden,in4:nheads*positions,out0:hidden
  rtou 0 0
  umulk 1 0 headdim
  umulk 2 0 positions
  uconst 3 0
  uloadf32 14 3 @in1
  rtou 4 14
  umulk 4 4 hidden
  uadd 4 4 1
  addconst 14 14 1
  loadconstf32 15 headdim
  uaddk 5 1 0
  forbegin 15
    uloadf32 1 5 @in0+hidden
    uloadf32 2 5 @in0+2*hidden
    ustoref32 1 4 @in2
    ustoref32 2 4 @in3
    uaddk 4 4 1
    uaddk 5 5 1
  forend
  uaddk 4 1 0
  uaddk 6 2 0
  loadconstf32 10 -3e38
  forbegin 14
    uaddk 5 1 0
    loadconstf32 3 0
    forbegin 15
      uloadf32 1 5 @in0
      uloadf32 2 4 @in2
      mul 2 1 2
      add 3 3 2
      uaddk 4 4 1
      uaddk 5 5 1
    forend
    mulconst 3 3 1/sqrt(headdim)
    ustoref32 3 6 @in4
    cmplt 4 10 3
    ifbegin 4
      mulconst 10 3 1
    ifend
    uaddk 4 4 hidden-headdim
    uaddk 6 6 1
  forend
  uaddk 6 2 0
  loadconstf32 11 0
  forbegin 14
    uloadf32 3 6 @in4
    sub 3 3 10
    exp 3 3
    ustoref32 3 6 @in4
    add 11 11 3
    uaddk 6 6 1
  forend
  uaddk 6 2 0
  forbegin 14
    uloadf32 3 6 @in4
    div 3 3 11
    ustoref32 3 6 @in4
    uaddk 6 6 1
  forend
  uaddk 5 1 0
  forbegin 15
    uaddk 4 5 0
    uaddk 6 2 0
    loadconstf32 12 0
    forbegin 14
      uloadf32 1 6 @in4
      uloadf32 2 4 @in3
      mul 2 1 2
      add 12 12 2
      uaddk 4 4 hidden
      uaddk 6 6 1
    forend
    ustoref32 12 5 @out0
    uaddk 5 5 1
  forend
endkernel

node token ext fir.embedding token_id wte -> x
node positional ext fir.embedding position wpe -> pos_embed
node embeddings micro g2.add x pos_embed -> x
repeat layers
node ln1 micro g2.layernorm x ln1w ln1b -> norm
node qkv_linear ext fir.linear norm qkvw -> qkv
node qkv_bias micro g2.add qkv qkvb -> qkv
node attention micro g2.attn_decode qkv position key_cache value_cache scores -> attn key_cache value_cache scores
node attention_linear ext fir.linear attn attnw -> mix
node attention_bias micro g2.add mix attnb -> mix
node attention_residual micro g2.add x mix -> x
node ln2 micro g2.layernorm x ln2w ln2b -> norm
node fc_linear ext fir.linear norm fcw -> fc
node fc_bias micro g2.add fc fcb -> fc
node gelu micro g2.gelu_new fc -> activated
node proj_linear ext fir.linear activated projw -> mix
node proj_bias micro g2.add mix projb -> mix
node mlp_residual micro g2.add x mix -> x
endrepeat
node final_norm micro g2.layernorm x lnfw lnfb -> norm
node head ext fir.linear_t norm wte -> logits
