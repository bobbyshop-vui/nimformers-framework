# FIR version 1: Mamba-3 SISO decode-step graph.
fir 1
model mamba3-siso
config config.json
sym P = 2*inter+2*dstate+3*nheads+rope

# Buffers describe sizes; bindings use the framework arenaLayout offsets.
# Weight descriptors name checkpoint tensors populated by the runtime loader.
buf hidden_in f32 hidden
buf token_id f32 1
buf hidden_norm f32 hidden
buf proj f32 P
buf k f32 nheads*dstate
buf q f32 nheads*dstate
buf coeff f32 nheads*3
buf state f32 inter*dstate per_layer
buf phase f32 nheads*rope per_layer
buf prev_k f32 nheads*dstate per_layer
buf prev_v f32 inter per_layer
buf ssm_out f32 inter
buf mlp_gate f32 4*hidden
buf mlp_out f32 2*hidden
buf residual f32 hidden
buf logits f32 vocab

weight embed f32 backbone.embeddings.weight tensor [vocab,hidden]
weight norm1 f32 backbone.layers.{i}.norm.weight tensor [hidden]
weight in_proj q4 backbone.layers.{i}.mixer.in_proj.weight tensor [P,hidden]
weight b_norm f32 backbone.layers.{i}.mixer.B_norm.weight tensor [dstate]
weight c_norm f32 backbone.layers.{i}.mixer.C_norm.weight tensor [dstate]
weight b_bias f32 backbone.layers.{i}.mixer.B_bias tensor [nheads,dstate]
weight c_bias f32 backbone.layers.{i}.mixer.C_bias tensor [nheads,dstate]
weight dt_bias f32 backbone.layers.{i}.mixer.dt_bias tensor [nheads]
weight D f32 backbone.layers.{i}.mixer.D tensor [nheads]
weight out_proj q4 backbone.layers.{i}.mixer.out_proj.weight tensor [hidden,inter]
weight norm2 f32 backbone.layers.{i}.norm2.weight tensor [hidden]
weight fc1 q4 backbone.layers.{i}.mlp.fc1.weight tensor [4*hidden,hidden]
weight fc2 q4 backbone.layers.{i}.mlp.fc2.weight tensor [hidden,2*hidden]
weight norm_f f32 backbone.norm_f.weight tensor [hidden]
weight lm_head q4 backbone.embeddings.weight tensor [vocab,hidden]

node embed ext m3.embedding token_id embed -> hidden_in
repeat layers
node n1 builtin rms_norm hidden_in norm1 -> hidden_norm
node n2 ext m3.gemv_q4 hidden_norm in_proj -> proj
node n3 micro m3.prepare proj b_norm c_norm b_bias c_bias dt_bias phase -> k q coeff phase
node n4 micro m3.ssm_step proj k q coeff state prev_k prev_v D -> ssm_out state
node carry micro m3.carry k proj -> prev_k prev_v
node n5 ext m3.gemv_q4 ssm_out out_proj -> residual
node n6 builtin add hidden_in residual -> hidden_in
node n7 builtin rms_norm hidden_in norm2 -> hidden_norm
node n8 ext m3.gemv_q4 hidden_norm fc1 -> mlp_gate
node n9 micro m3.silu_mul mlp_gate -> mlp_out
node n10 ext m3.gemv_q4 mlp_out fc2 -> residual
node n11 builtin add hidden_in residual -> hidden_in
endrepeat
node n12 builtin rms_norm hidden_in norm_f -> hidden_norm
node n13 ext m3.gemv_q4 hidden_norm lm_head -> logits
