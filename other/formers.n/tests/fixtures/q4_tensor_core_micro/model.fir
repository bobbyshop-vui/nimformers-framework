# Build-time compatibility check: the actual CLI must accept the bytecode
# emitted by FIRLIFT_Q4_GEMM=tensor_core. No weights or GPU are needed.
fir 1
model q4_tensor_core_micro_check
runtime graph
layout script
script_abi 2
require feature resident_q4
sym layers = num_hidden_layers
sym positions = max_position_embeddings
sym vocab = vocab_size
buf token f32 1 input:token
buf position f32 1 input:position
buf x f32 1024
buf logits f32 1024 output:logits
weight w q4r tensor tensor [64,64]
kernel s.matmul_q4_tensor_core_micro in=2 out=1 threads=128 tg=128 tgbytes=15616 rows=16:128 layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0 tag=aten::matmul
  words 000000aa 00000010 00000040 00000040 00000020 00000002 00000001
endkernel
node q4 micro s.matmul_q4_tensor_core_micro x w -> logits
