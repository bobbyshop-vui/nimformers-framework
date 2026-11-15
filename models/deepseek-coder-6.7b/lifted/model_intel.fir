fir 1
model models
config config.json
runtime graph
layout script
script_abi 2
require feature resident_q4
epsilon rms_norm_eps
require feature weights_bf16
sym hidden = hidden_size
sym layers = num_hidden_layers
sym positions = max_position_embeddings
sym vocab = vocab_size
buf token f32 1 input:token
buf t1038_embedding f32 262144
buf t1042_y f32 262144
buf t1044_c f32 262144
buf t1046_c f32 262144
buf t1048_c f32 262144
buf position f32 1 input:position
buf s740 f32 2097152
buf s741 f32 2097152
buf t1062_c f32 704512
buf t1064_c f32 704512
buf t1066_y f32 704512
buf t1068_c f32 704512
buf s742 f32 2097152
buf s743 f32 2097152
buf s744 f32 2097152
buf s745 f32 2097152
buf s746 f32 2097152
buf s747 f32 2097152
buf s748 f32 2097152
buf s749 f32 2097152
buf s750 f32 2097152
buf s751 f32 2097152
buf s752 f32 2097152
buf s753 f32 2097152
buf s754 f32 2097152
buf s755 f32 2097152
buf s756 f32 2097152
buf s757 f32 2097152
buf s758 f32 2097152
buf s759 f32 2097152
buf s760 f32 2097152
buf s761 f32 2097152
buf s762 f32 2097152
buf s763 f32 2097152
buf s764 f32 2097152
buf s765 f32 2097152
buf s766 f32 2097152
buf s767 f32 2097152
buf s768 f32 2097152
buf s769 f32 2097152
buf s770 f32 2097152
buf s771 f32 2097152
buf s772 f32 2097152
buf s773 f32 2097152
buf s774 f32 2097152
buf s775 f32 2097152
buf s776 f32 2097152
buf s777 f32 2097152
buf s778 f32 2097152
buf s779 f32 2097152
buf s780 f32 2097152
buf s781 f32 2097152
buf s782 f32 2097152
buf s783 f32 2097152
buf s784 f32 2097152
buf s785 f32 2097152
buf s786 f32 2097152
buf s787 f32 2097152
buf s788 f32 2097152
buf s789 f32 2097152
buf s790 f32 2097152
buf s791 f32 2097152
buf s792 f32 2097152
buf s793 f32 2097152
buf s794 f32 2097152
buf s795 f32 2097152
buf s796 f32 2097152
buf s797 f32 2097152
buf s798 f32 2097152
buf s799 f32 2097152
buf s800 f32 2097152
buf s801 f32 2097152
buf s802 f32 2097152
buf s803 f32 2097152
buf s824 f32 4096
buf t2684_embedding f32 65536
buf t2688_y f32 65536
buf t2690_c f32 65536
buf t2692_c f32 65536
buf t2694_c f32 65536
buf t2708_c f32 176128
buf t2710_c f32 176128
buf t2712_y f32 176128
buf t2714_c f32 176128
buf t4284_y f32 4096
buf t4285_c f32 32256
buf t4311_embedding f32 4096
buf t4316_y f32 4096
buf t4317_c f32 4096
buf t4318_c f32 4096
buf t4319_c f32 4096
buf t4333_c f32 11008
buf t4334_c f32 11008
buf t4340_c f32 11008
buf t5403_c f32 32256 output:logits
weight model_embed_tokens_weight_emb f32 model.embed_tokens.weight tensor [32256,4096]
weight model_layers_0_input_layernorm_weight f32 model.layers.0.input_layernorm.weight tensor [4096]
weight model_layers_0_self_attn_q_proj_weight_qr q4r128 model.layers.0.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_0_self_attn_k_proj_weight_qr q4r128 model.layers.0.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_0_self_attn_v_proj_weight_qr q4r128 model.layers.0.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_0_self_attn_o_proj_weight_qr q4r128 model.layers.0.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_0_post_attention_layernorm_weight f32 model.layers.0.post_attention_layernorm.weight tensor [4096]
weight model_layers_0_mlp_gate_proj_weight_qr q4r128 model.layers.0.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_0_mlp_up_proj_weight_qr q4r128 model.layers.0.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_0_mlp_down_proj_weight_qr q4r128 model.layers.0.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_1_input_layernorm_weight f32 model.layers.1.input_layernorm.weight tensor [4096]
weight model_layers_1_self_attn_q_proj_weight_qr q4r128 model.layers.1.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_1_self_attn_k_proj_weight_qr q4r128 model.layers.1.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_1_self_attn_v_proj_weight_qr q4r128 model.layers.1.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_1_self_attn_o_proj_weight_qr q4r128 model.layers.1.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_1_post_attention_layernorm_weight f32 model.layers.1.post_attention_layernorm.weight tensor [4096]
weight model_layers_1_mlp_gate_proj_weight_qr q4r128 model.layers.1.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_1_mlp_up_proj_weight_qr q4r128 model.layers.1.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_1_mlp_down_proj_weight_qr q4r128 model.layers.1.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_2_input_layernorm_weight f32 model.layers.2.input_layernorm.weight tensor [4096]
weight model_layers_2_self_attn_q_proj_weight_qr q4r128 model.layers.2.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_2_self_attn_k_proj_weight_qr q4r128 model.layers.2.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_2_self_attn_v_proj_weight_qr q4r128 model.layers.2.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_2_self_attn_o_proj_weight_qr q4r128 model.layers.2.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_2_post_attention_layernorm_weight f32 model.layers.2.post_attention_layernorm.weight tensor [4096]
weight model_layers_2_mlp_gate_proj_weight_qr q4r128 model.layers.2.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_2_mlp_up_proj_weight_qr q4r128 model.layers.2.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_2_mlp_down_proj_weight_qr q4r128 model.layers.2.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_3_input_layernorm_weight f32 model.layers.3.input_layernorm.weight tensor [4096]
weight model_layers_3_self_attn_q_proj_weight_qr q4r128 model.layers.3.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_3_self_attn_k_proj_weight_qr q4r128 model.layers.3.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_3_self_attn_v_proj_weight_qr q4r128 model.layers.3.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_3_self_attn_o_proj_weight_qr q4r128 model.layers.3.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_3_post_attention_layernorm_weight f32 model.layers.3.post_attention_layernorm.weight tensor [4096]
weight model_layers_3_mlp_gate_proj_weight_qr q4r128 model.layers.3.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_3_mlp_up_proj_weight_qr q4r128 model.layers.3.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_3_mlp_down_proj_weight_qr q4r128 model.layers.3.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_4_input_layernorm_weight f32 model.layers.4.input_layernorm.weight tensor [4096]
weight model_layers_4_self_attn_q_proj_weight_qr q4r128 model.layers.4.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_4_self_attn_k_proj_weight_qr q4r128 model.layers.4.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_4_self_attn_v_proj_weight_qr q4r128 model.layers.4.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_4_self_attn_o_proj_weight_qr q4r128 model.layers.4.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_4_post_attention_layernorm_weight f32 model.layers.4.post_attention_layernorm.weight tensor [4096]
weight model_layers_4_mlp_gate_proj_weight_qr q4r128 model.layers.4.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_4_mlp_up_proj_weight_qr q4r128 model.layers.4.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_4_mlp_down_proj_weight_qr q4r128 model.layers.4.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_5_input_layernorm_weight f32 model.layers.5.input_layernorm.weight tensor [4096]
weight model_layers_5_self_attn_q_proj_weight_qr q4r128 model.layers.5.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_5_self_attn_k_proj_weight_qr q4r128 model.layers.5.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_5_self_attn_v_proj_weight_qr q4r128 model.layers.5.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_5_self_attn_o_proj_weight_qr q4r128 model.layers.5.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_5_post_attention_layernorm_weight f32 model.layers.5.post_attention_layernorm.weight tensor [4096]
weight model_layers_5_mlp_gate_proj_weight_qr q4r128 model.layers.5.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_5_mlp_up_proj_weight_qr q4r128 model.layers.5.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_5_mlp_down_proj_weight_qr q4r128 model.layers.5.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_6_input_layernorm_weight f32 model.layers.6.input_layernorm.weight tensor [4096]
weight model_layers_6_self_attn_q_proj_weight_qr q4r128 model.layers.6.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_6_self_attn_k_proj_weight_qr q4r128 model.layers.6.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_6_self_attn_v_proj_weight_qr q4r128 model.layers.6.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_6_self_attn_o_proj_weight_qr q4r128 model.layers.6.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_6_post_attention_layernorm_weight f32 model.layers.6.post_attention_layernorm.weight tensor [4096]
weight model_layers_6_mlp_gate_proj_weight_qr q4r128 model.layers.6.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_6_mlp_up_proj_weight_qr q4r128 model.layers.6.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_6_mlp_down_proj_weight_qr q4r128 model.layers.6.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_7_input_layernorm_weight f32 model.layers.7.input_layernorm.weight tensor [4096]
weight model_layers_7_self_attn_q_proj_weight_qr q4r128 model.layers.7.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_7_self_attn_k_proj_weight_qr q4r128 model.layers.7.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_7_self_attn_v_proj_weight_qr q4r128 model.layers.7.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_7_self_attn_o_proj_weight_qr q4r128 model.layers.7.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_7_post_attention_layernorm_weight f32 model.layers.7.post_attention_layernorm.weight tensor [4096]
weight model_layers_7_mlp_gate_proj_weight_qr q4r128 model.layers.7.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_7_mlp_up_proj_weight_qr q4r128 model.layers.7.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_7_mlp_down_proj_weight_qr q4r128 model.layers.7.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_8_input_layernorm_weight f32 model.layers.8.input_layernorm.weight tensor [4096]
weight model_layers_8_self_attn_q_proj_weight_qr q4r128 model.layers.8.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_8_self_attn_k_proj_weight_qr q4r128 model.layers.8.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_8_self_attn_v_proj_weight_qr q4r128 model.layers.8.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_8_self_attn_o_proj_weight_qr q4r128 model.layers.8.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_8_post_attention_layernorm_weight f32 model.layers.8.post_attention_layernorm.weight tensor [4096]
weight model_layers_8_mlp_gate_proj_weight_qr q4r128 model.layers.8.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_8_mlp_up_proj_weight_qr q4r128 model.layers.8.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_8_mlp_down_proj_weight_qr q4r128 model.layers.8.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_9_input_layernorm_weight f32 model.layers.9.input_layernorm.weight tensor [4096]
weight model_layers_9_self_attn_q_proj_weight_qr q4r128 model.layers.9.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_9_self_attn_k_proj_weight_qr q4r128 model.layers.9.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_9_self_attn_v_proj_weight_qr q4r128 model.layers.9.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_9_self_attn_o_proj_weight_qr q4r128 model.layers.9.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_9_post_attention_layernorm_weight f32 model.layers.9.post_attention_layernorm.weight tensor [4096]
weight model_layers_9_mlp_gate_proj_weight_qr q4r128 model.layers.9.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_9_mlp_up_proj_weight_qr q4r128 model.layers.9.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_9_mlp_down_proj_weight_qr q4r128 model.layers.9.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_10_input_layernorm_weight f32 model.layers.10.input_layernorm.weight tensor [4096]
weight model_layers_10_self_attn_q_proj_weight_qr q4r128 model.layers.10.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_10_self_attn_k_proj_weight_qr q4r128 model.layers.10.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_10_self_attn_v_proj_weight_qr q4r128 model.layers.10.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_10_self_attn_o_proj_weight_qr q4r128 model.layers.10.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_10_post_attention_layernorm_weight f32 model.layers.10.post_attention_layernorm.weight tensor [4096]
weight model_layers_10_mlp_gate_proj_weight_qr q4r128 model.layers.10.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_10_mlp_up_proj_weight_qr q4r128 model.layers.10.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_10_mlp_down_proj_weight_qr q4r128 model.layers.10.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_11_input_layernorm_weight f32 model.layers.11.input_layernorm.weight tensor [4096]
weight model_layers_11_self_attn_q_proj_weight_qr q4r128 model.layers.11.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_11_self_attn_k_proj_weight_qr q4r128 model.layers.11.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_11_self_attn_v_proj_weight_qr q4r128 model.layers.11.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_11_self_attn_o_proj_weight_qr q4r128 model.layers.11.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_11_post_attention_layernorm_weight f32 model.layers.11.post_attention_layernorm.weight tensor [4096]
weight model_layers_11_mlp_gate_proj_weight_qr q4r128 model.layers.11.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_11_mlp_up_proj_weight_qr q4r128 model.layers.11.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_11_mlp_down_proj_weight_qr q4r128 model.layers.11.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_12_input_layernorm_weight f32 model.layers.12.input_layernorm.weight tensor [4096]
weight model_layers_12_self_attn_q_proj_weight_qr q4r128 model.layers.12.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_12_self_attn_k_proj_weight_qr q4r128 model.layers.12.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_12_self_attn_v_proj_weight_qr q4r128 model.layers.12.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_12_self_attn_o_proj_weight_qr q4r128 model.layers.12.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_12_post_attention_layernorm_weight f32 model.layers.12.post_attention_layernorm.weight tensor [4096]
weight model_layers_12_mlp_gate_proj_weight_qr q4r128 model.layers.12.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_12_mlp_up_proj_weight_qr q4r128 model.layers.12.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_12_mlp_down_proj_weight_qr q4r128 model.layers.12.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_13_input_layernorm_weight f32 model.layers.13.input_layernorm.weight tensor [4096]
weight model_layers_13_self_attn_q_proj_weight_qr q4r128 model.layers.13.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_13_self_attn_k_proj_weight_qr q4r128 model.layers.13.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_13_self_attn_v_proj_weight_qr q4r128 model.layers.13.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_13_self_attn_o_proj_weight_qr q4r128 model.layers.13.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_13_post_attention_layernorm_weight f32 model.layers.13.post_attention_layernorm.weight tensor [4096]
weight model_layers_13_mlp_gate_proj_weight_qr q4r128 model.layers.13.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_13_mlp_up_proj_weight_qr q4r128 model.layers.13.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_13_mlp_down_proj_weight_qr q4r128 model.layers.13.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_14_input_layernorm_weight f32 model.layers.14.input_layernorm.weight tensor [4096]
weight model_layers_14_self_attn_q_proj_weight_qr q4r128 model.layers.14.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_14_self_attn_k_proj_weight_qr q4r128 model.layers.14.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_14_self_attn_v_proj_weight_qr q4r128 model.layers.14.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_14_self_attn_o_proj_weight_qr q4r128 model.layers.14.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_14_post_attention_layernorm_weight f32 model.layers.14.post_attention_layernorm.weight tensor [4096]
weight model_layers_14_mlp_gate_proj_weight_qr q4r128 model.layers.14.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_14_mlp_up_proj_weight_qr q4r128 model.layers.14.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_14_mlp_down_proj_weight_qr q4r128 model.layers.14.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_15_input_layernorm_weight f32 model.layers.15.input_layernorm.weight tensor [4096]
weight model_layers_15_self_attn_q_proj_weight_qr q4r128 model.layers.15.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_15_self_attn_k_proj_weight_qr q4r128 model.layers.15.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_15_self_attn_v_proj_weight_qr q4r128 model.layers.15.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_15_self_attn_o_proj_weight_qr q4r128 model.layers.15.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_15_post_attention_layernorm_weight f32 model.layers.15.post_attention_layernorm.weight tensor [4096]
weight model_layers_15_mlp_gate_proj_weight_qr q4r128 model.layers.15.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_15_mlp_up_proj_weight_qr q4r128 model.layers.15.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_15_mlp_down_proj_weight_qr q4r128 model.layers.15.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_16_input_layernorm_weight f32 model.layers.16.input_layernorm.weight tensor [4096]
weight model_layers_16_self_attn_q_proj_weight_qr q4r128 model.layers.16.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_16_self_attn_k_proj_weight_qr q4r128 model.layers.16.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_16_self_attn_v_proj_weight_qr q4r128 model.layers.16.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_16_self_attn_o_proj_weight_qr q4r128 model.layers.16.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_16_post_attention_layernorm_weight f32 model.layers.16.post_attention_layernorm.weight tensor [4096]
weight model_layers_16_mlp_gate_proj_weight_qr q4r128 model.layers.16.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_16_mlp_up_proj_weight_qr q4r128 model.layers.16.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_16_mlp_down_proj_weight_qr q4r128 model.layers.16.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_17_input_layernorm_weight f32 model.layers.17.input_layernorm.weight tensor [4096]
weight model_layers_17_self_attn_q_proj_weight_qr q4r128 model.layers.17.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_17_self_attn_k_proj_weight_qr q4r128 model.layers.17.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_17_self_attn_v_proj_weight_qr q4r128 model.layers.17.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_17_self_attn_o_proj_weight_qr q4r128 model.layers.17.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_17_post_attention_layernorm_weight f32 model.layers.17.post_attention_layernorm.weight tensor [4096]
weight model_layers_17_mlp_gate_proj_weight_qr q4r128 model.layers.17.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_17_mlp_up_proj_weight_qr q4r128 model.layers.17.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_17_mlp_down_proj_weight_qr q4r128 model.layers.17.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_18_input_layernorm_weight f32 model.layers.18.input_layernorm.weight tensor [4096]
weight model_layers_18_self_attn_q_proj_weight_qr q4r128 model.layers.18.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_18_self_attn_k_proj_weight_qr q4r128 model.layers.18.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_18_self_attn_v_proj_weight_qr q4r128 model.layers.18.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_18_self_attn_o_proj_weight_qr q4r128 model.layers.18.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_18_post_attention_layernorm_weight f32 model.layers.18.post_attention_layernorm.weight tensor [4096]
weight model_layers_18_mlp_gate_proj_weight_qr q4r128 model.layers.18.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_18_mlp_up_proj_weight_qr q4r128 model.layers.18.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_18_mlp_down_proj_weight_qr q4r128 model.layers.18.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_19_input_layernorm_weight f32 model.layers.19.input_layernorm.weight tensor [4096]
weight model_layers_19_self_attn_q_proj_weight_qr q4r128 model.layers.19.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_19_self_attn_k_proj_weight_qr q4r128 model.layers.19.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_19_self_attn_v_proj_weight_qr q4r128 model.layers.19.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_19_self_attn_o_proj_weight_qr q4r128 model.layers.19.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_19_post_attention_layernorm_weight f32 model.layers.19.post_attention_layernorm.weight tensor [4096]
weight model_layers_19_mlp_gate_proj_weight_qr q4r128 model.layers.19.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_19_mlp_up_proj_weight_qr q4r128 model.layers.19.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_19_mlp_down_proj_weight_qr q4r128 model.layers.19.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_20_input_layernorm_weight f32 model.layers.20.input_layernorm.weight tensor [4096]
weight model_layers_20_self_attn_q_proj_weight_qr q4r128 model.layers.20.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_20_self_attn_k_proj_weight_qr q4r128 model.layers.20.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_20_self_attn_v_proj_weight_qr q4r128 model.layers.20.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_20_self_attn_o_proj_weight_qr q4r128 model.layers.20.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_20_post_attention_layernorm_weight f32 model.layers.20.post_attention_layernorm.weight tensor [4096]
weight model_layers_20_mlp_gate_proj_weight_qr q4r128 model.layers.20.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_20_mlp_up_proj_weight_qr q4r128 model.layers.20.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_20_mlp_down_proj_weight_qr q4r128 model.layers.20.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_21_input_layernorm_weight f32 model.layers.21.input_layernorm.weight tensor [4096]
weight model_layers_21_self_attn_q_proj_weight_qr q4r128 model.layers.21.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_21_self_attn_k_proj_weight_qr q4r128 model.layers.21.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_21_self_attn_v_proj_weight_qr q4r128 model.layers.21.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_21_self_attn_o_proj_weight_qr q4r128 model.layers.21.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_21_post_attention_layernorm_weight f32 model.layers.21.post_attention_layernorm.weight tensor [4096]
weight model_layers_21_mlp_gate_proj_weight_qr q4r128 model.layers.21.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_21_mlp_up_proj_weight_qr q4r128 model.layers.21.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_21_mlp_down_proj_weight_qr q4r128 model.layers.21.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_22_input_layernorm_weight f32 model.layers.22.input_layernorm.weight tensor [4096]
weight model_layers_22_self_attn_q_proj_weight_qr q4r128 model.layers.22.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_22_self_attn_k_proj_weight_qr q4r128 model.layers.22.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_22_self_attn_v_proj_weight_qr q4r128 model.layers.22.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_22_self_attn_o_proj_weight_qr q4r128 model.layers.22.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_22_post_attention_layernorm_weight f32 model.layers.22.post_attention_layernorm.weight tensor [4096]
weight model_layers_22_mlp_gate_proj_weight_qr q4r128 model.layers.22.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_22_mlp_up_proj_weight_qr q4r128 model.layers.22.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_22_mlp_down_proj_weight_qr q4r128 model.layers.22.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_23_input_layernorm_weight f32 model.layers.23.input_layernorm.weight tensor [4096]
weight model_layers_23_self_attn_q_proj_weight_qr q4r128 model.layers.23.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_23_self_attn_k_proj_weight_qr q4r128 model.layers.23.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_23_self_attn_v_proj_weight_qr q4r128 model.layers.23.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_23_self_attn_o_proj_weight_qr q4r128 model.layers.23.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_23_post_attention_layernorm_weight f32 model.layers.23.post_attention_layernorm.weight tensor [4096]
weight model_layers_23_mlp_gate_proj_weight_qr q4r128 model.layers.23.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_23_mlp_up_proj_weight_qr q4r128 model.layers.23.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_23_mlp_down_proj_weight_qr q4r128 model.layers.23.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_24_input_layernorm_weight f32 model.layers.24.input_layernorm.weight tensor [4096]
weight model_layers_24_self_attn_q_proj_weight_qr q4r128 model.layers.24.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_24_self_attn_k_proj_weight_qr q4r128 model.layers.24.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_24_self_attn_v_proj_weight_qr q4r128 model.layers.24.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_24_self_attn_o_proj_weight_qr q4r128 model.layers.24.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_24_post_attention_layernorm_weight f32 model.layers.24.post_attention_layernorm.weight tensor [4096]
weight model_layers_24_mlp_gate_proj_weight_qr q4r128 model.layers.24.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_24_mlp_up_proj_weight_qr q4r128 model.layers.24.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_24_mlp_down_proj_weight_qr q4r128 model.layers.24.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_25_input_layernorm_weight f32 model.layers.25.input_layernorm.weight tensor [4096]
weight model_layers_25_self_attn_q_proj_weight_qr q4r128 model.layers.25.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_25_self_attn_k_proj_weight_qr q4r128 model.layers.25.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_25_self_attn_v_proj_weight_qr q4r128 model.layers.25.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_25_self_attn_o_proj_weight_qr q4r128 model.layers.25.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_25_post_attention_layernorm_weight f32 model.layers.25.post_attention_layernorm.weight tensor [4096]
weight model_layers_25_mlp_gate_proj_weight_qr q4r128 model.layers.25.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_25_mlp_up_proj_weight_qr q4r128 model.layers.25.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_25_mlp_down_proj_weight_qr q4r128 model.layers.25.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_26_input_layernorm_weight f32 model.layers.26.input_layernorm.weight tensor [4096]
weight model_layers_26_self_attn_q_proj_weight_qr q4r128 model.layers.26.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_26_self_attn_k_proj_weight_qr q4r128 model.layers.26.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_26_self_attn_v_proj_weight_qr q4r128 model.layers.26.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_26_self_attn_o_proj_weight_qr q4r128 model.layers.26.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_26_post_attention_layernorm_weight f32 model.layers.26.post_attention_layernorm.weight tensor [4096]
weight model_layers_26_mlp_gate_proj_weight_qr q4r128 model.layers.26.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_26_mlp_up_proj_weight_qr q4r128 model.layers.26.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_26_mlp_down_proj_weight_qr q4r128 model.layers.26.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_27_input_layernorm_weight f32 model.layers.27.input_layernorm.weight tensor [4096]
weight model_layers_27_self_attn_q_proj_weight_qr q4r128 model.layers.27.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_27_self_attn_k_proj_weight_qr q4r128 model.layers.27.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_27_self_attn_v_proj_weight_qr q4r128 model.layers.27.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_27_self_attn_o_proj_weight_qr q4r128 model.layers.27.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_27_post_attention_layernorm_weight f32 model.layers.27.post_attention_layernorm.weight tensor [4096]
weight model_layers_27_mlp_gate_proj_weight_qr q4r128 model.layers.27.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_27_mlp_up_proj_weight_qr q4r128 model.layers.27.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_27_mlp_down_proj_weight_qr q4r128 model.layers.27.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_28_input_layernorm_weight f32 model.layers.28.input_layernorm.weight tensor [4096]
weight model_layers_28_self_attn_q_proj_weight_qr q4r128 model.layers.28.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_28_self_attn_k_proj_weight_qr q4r128 model.layers.28.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_28_self_attn_v_proj_weight_qr q4r128 model.layers.28.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_28_self_attn_o_proj_weight_qr q4r128 model.layers.28.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_28_post_attention_layernorm_weight f32 model.layers.28.post_attention_layernorm.weight tensor [4096]
weight model_layers_28_mlp_gate_proj_weight_qr q4r128 model.layers.28.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_28_mlp_up_proj_weight_qr q4r128 model.layers.28.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_28_mlp_down_proj_weight_qr q4r128 model.layers.28.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_29_input_layernorm_weight f32 model.layers.29.input_layernorm.weight tensor [4096]
weight model_layers_29_self_attn_q_proj_weight_qr q4r128 model.layers.29.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_29_self_attn_k_proj_weight_qr q4r128 model.layers.29.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_29_self_attn_v_proj_weight_qr q4r128 model.layers.29.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_29_self_attn_o_proj_weight_qr q4r128 model.layers.29.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_29_post_attention_layernorm_weight f32 model.layers.29.post_attention_layernorm.weight tensor [4096]
weight model_layers_29_mlp_gate_proj_weight_qr q4r128 model.layers.29.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_29_mlp_up_proj_weight_qr q4r128 model.layers.29.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_29_mlp_down_proj_weight_qr q4r128 model.layers.29.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_30_input_layernorm_weight f32 model.layers.30.input_layernorm.weight tensor [4096]
weight model_layers_30_self_attn_q_proj_weight_qr q4r128 model.layers.30.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_30_self_attn_k_proj_weight_qr q4r128 model.layers.30.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_30_self_attn_v_proj_weight_qr q4r128 model.layers.30.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_30_self_attn_o_proj_weight_qr q4r128 model.layers.30.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_30_post_attention_layernorm_weight f32 model.layers.30.post_attention_layernorm.weight tensor [4096]
weight model_layers_30_mlp_gate_proj_weight_qr q4r128 model.layers.30.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_30_mlp_up_proj_weight_qr q4r128 model.layers.30.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_30_mlp_down_proj_weight_qr q4r128 model.layers.30.mlp.down_proj.weight tensor [4096,11008]
weight model_layers_31_input_layernorm_weight f32 model.layers.31.input_layernorm.weight tensor [4096]
weight model_layers_31_self_attn_q_proj_weight_qr q4r128 model.layers.31.self_attn.q_proj.weight tensor [4096,4096]
weight model_layers_31_self_attn_k_proj_weight_qr q4r128 model.layers.31.self_attn.k_proj.weight tensor [4096,4096]
weight model_layers_31_self_attn_v_proj_weight_qr q4r128 model.layers.31.self_attn.v_proj.weight tensor [4096,4096]
weight model_layers_31_self_attn_o_proj_weight_qr q4r128 model.layers.31.self_attn.o_proj.weight tensor [4096,4096]
weight model_layers_31_post_attention_layernorm_weight f32 model.layers.31.post_attention_layernorm.weight tensor [4096]
weight model_layers_31_mlp_gate_proj_weight_qr q4r128 model.layers.31.mlp.gate_proj.weight tensor [11008,4096]
weight model_layers_31_mlp_up_proj_weight_qr q4r128 model.layers.31.mlp.up_proj.weight tensor [11008,4096]
weight model_layers_31_mlp_down_proj_weight_qr q4r128 model.layers.31.mlp.down_proj.weight tensor [4096,11008]
weight model_norm_weight f32 model.norm.weight tensor [4096]
weight lm_head_weight_qr q4r128 lm_head.weight tensor [32256,4096]
kernel s.rmsnorm in=2 out=1 threads=16384 tg=256 tgbytes=1024 expect=in0:262144,in1:4096,out0:262144 rows=1:256 tag=aten::rms_norm
rtou 0 0
loadconstf32 4 0
loadconstf32 7 4096
utgid 1
ulid 2
loadlid 1
umulk 3 1 4096
uaddk 4 2 0
loadconstf32 8 16
forbegin 8
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
mul 2 2 2
add 4 4 2
ifend 
uaddk 4 4 256
forend 
tgstoref32 4 1 0
barriertg 
loadconstf32 8 128
cmplt 10 1 8
ifbegin 10
addconst 9 1 128
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 64
cmplt 10 1 8
ifbegin 10
addconst 9 1 64
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 32
cmplt 10 1 8
ifbegin 10
addconst 9 1 32
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 16
cmplt 10 1 8
ifbegin 10
addconst 9 1 16
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 8
cmplt 10 1 8
ifbegin 10
addconst 9 1 8
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 4
cmplt 10 1 8
ifbegin 10
addconst 9 1 4
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 2
cmplt 10 1 8
ifbegin 10
addconst 9 1 2
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 1
cmplt 10 1 8
ifbegin 10
addconst 9 1 1
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 9 0
tgloadf32 8 9 0
mulconst 8 8 0.000244140625
addconst 8 8 0.000001
rsqrt 8 8
uaddk 4 2 0
loadconstf32 10 16
forbegin 10
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
mul 2 2 8
uloadf32 3 4 @in1
mul 2 2 3
ustoref32 2 5 @out0
ifend 
uaddk 4 4 256
forend 
endkernel
kernel s.matmul_q4_tiled_fma in=2 out=1 threads=4096 tg=128 tgbytes=25600 rows=64:4096 layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0 tag=aten::matmul
words 00000090 00000180 00000281 01000113 42000000 01003c8f 49078227 08004390 0100628f 49044221 490b9c23 08038490 00008586 00000080 00006686 00000040 00004787 0000000f 00004888 00000004 00000f82 00000000 0001ef86 00000004 00000982 00000000 00000113 00000000 00000213 00000000 00000313 00000000 00000413 00000000 00000513 00000000 00000613 00000000 00000713 00000000 00000813 00000000 00000913 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000f13 00000000 00001013 00000000 00001113 00000000 00001213 00000000 00001313 00000000 00001413 00000000 00001513 00000000 00001613 00000000 00001713 00000000 00001813 00000000 00001913 00000000 00001a13 00000000 00001b13 00000000 00001c13 00000000 00001d13 00000000 00001e13 00000000 00001f13 00000000 00800013 00000000 00800113 00000000 00800213 00000000 00800313 00000000 00800413 00000000 00800513 00000000 00800613 00000000 00800713 00000000 00800813 00000000 00800913 00000000 00800a13 00000000 00800b13 00000000 00800c13 00000000 00800d13 00000000 00800e13 00000000 00800f13 00000000 00801013 00000000 00801113 00000000 00801213 00000000 00801313 00000000 00801413 00000000 00801513 00000000 00801613 00000000 00801713 00000000 00801813 00000000 00801913 00000000 00801a13 00000000 00801b13 00000000 00801c13 00000000 00801d13 00000000 00801e13 00000000 00801f13 00000000 01000013 00000000 01800013 43000000 01800040 00004a85 00000000 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000080 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000100 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000180 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000000 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00200000 01015f8c 00220000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000080 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00200000 01015f8c 00220000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000100 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00200000 01015f8c 00220000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000180 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00200000 01015f8c 00220000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00000050 00010a86 00000008 0101598f 0000ea86 00000008 00014a85 00000880 01015a8f 01800113 42000000 01800140 09034960 00000000 09034a60 00000001 09034b60 00000002 09034c60 00000003 09034d60 00000004 09034e60 00000005 09034f60 00000006 09035060 00000007 09033160 00000000 09033260 00000001 09033360 00000002 09033460 00000003 09033560 00000004 09033660 00000005 09033760 00000006 09033860 00000007 48262122 00000001 482a2222 00000002 482e2322 00000003 48322422 00000004 48362522 00000005 483a2622 00000006 483e2722 00000007 48422822 00000008 48264922 00000009 482a4a22 0000000a 482e4b22 0000000b 48324c22 0000000c 48364d22 0000000d 483a4e22 0000000e 483e4f22 0000000f 48425022 00000010 48267122 00000011 482a7222 00000012 482e7322 00000013 48327422 00000014 48367522 00000015 483a7622 00000016 483e7722 00000017 48427822 00000018 48269922 00000019 482a9a22 0000001a 482e9b22 0000001b 48329c22 0000001c 48369d22 0000001d 483a9e22 0000001e 483e9f22 0000001f 48c28022 00000020 48a6a122 00000021 48aaa222 00000022 48aea322 00000023 48b2a422 00000024 48b6a522 00000025 48baa622 00000026 48bea722 00000027 48c2a822 00000028 48a6c922 00000029 48aaca22 0000002a 48aecb22 0000002b 48b2cc22 0000002c 48b6cd22 0000002d 48bace22 0000002e 48becf22 0000002f 48c2d022 00000030 48a6f122 00000031 48aaf222 00000032 48aef322 00000033 48b2f422 00000034 48b6f522 00000035 48baf622 00000036 48bef722 00000037 48c2f822 00000038 48a71922 00000039 48ab1a22 0000003a 48af1b22 0000003b 48b31c22 0000003c 48b71d22 0000003d 48bb1e22 0000003e 48bf1f22 0000003f 49430022 00000040 0903392a 42880000 09035a2a 43040000 00000041 00000050 00012985 00000020 00000041 00010a86 00000008 00194a83 0000eb86 00000008 00156b83 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001818e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001828e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001838e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001848e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001858e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001868e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001878e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001888e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001898e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018a8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018b8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018c8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018d8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018e8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018f8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001908e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001918e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001928e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001938e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001948e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001958e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001968e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001978e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001988e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001998e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019a8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019b8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019c8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019d8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019e8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019f8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081808e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081818e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081828e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081838e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081848e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081858e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081868e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081878e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081888e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081898e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818a8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818b8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818c8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818d8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818e8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818f8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081908e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081918e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081928e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081938e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081948e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081958e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081968e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081978e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081988e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081998e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819a8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819b8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819c8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819d8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819e8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819f8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0101808e 00000000 00000073 00000001
endkernel
kernel s.rope in=2 out=1 threads=131072 tg=64 tgbytes=16 expect=in0:262144,in1:1,out0:262144 rows=1:2048
rtou 0 0
uconst 1 0
uandk 2 0 63
ushrk 3 0 6
umulk 5 3 128
uadd 5 5 2
uloadf32 1 1 @in1
utor 10 3
addconst 10 10 0.5
mulconst 10 10 0.03125
rtou 4 10
utor 9 4
add 1 1 9
mulconst 1 1 0.25
utor 9 2
mulconst 9 9 -0.17988946039016
exp 2 9
mul 3 1 2
mulconst 11 3 0.1591549430919
addconst 11 11 0.5
rtou 6 11
utor 11 6
mulconst 13 11 6.28125
sub 12 3 13
mulconst 13 11 0.0019353071795864769
sub 12 12 13
mul 14 12 12
mulconst 15 12 1
mulconst 4 12 1
loadconstf32 16 1
loadconstf32 5 1
mul 15 15 14
mulconst 15 15 -0.16666666666667
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.5
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.05
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.08333333333333
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.02380952380952
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.03333333333333
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.01388888888889
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.01785714285714
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00909090909091
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.01111111111111
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00641025641026
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00757575757576
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.0047619047619
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00549450549451
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00367647058824
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00416666666667
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00292397660819
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00326797385621
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00238095238095
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00263157894737
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00197628458498
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.0021645021645
add 5 5 16
uloadf32 6 5 @in0
uloadf32 7 5 @in0+64
mul 8 6 5
mul 3 7 4
sub 8 8 3
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
ustoref32 8 5 @out0+64
endkernel
kernel s.kv_scatter in=3 out=1 threads=262144 tg=64 tgbytes=16 expect=in0:262144,in1:1,in2:2097152,out0:2097152 rows=1:4096
rtou 0 0
uconst 1 0
uloadf32 1 1 @in1
rtou 2 1
utor 3 0
addconst 3 3 0.5
mulconst 3 3 0.0078125
rtou 3 3
utor 4 0
utor 5 3
mulconst 5 5 128
sub 4 4 5
rtou 6 4
utor 6 3
addconst 6 6 0.5
mulconst 6 6 0.03125
rtou 4 6
utor 7 3
utor 8 4
mulconst 8 8 32
sub 7 7 8
rtou 5 7
uadd 7 2 4
umulk 5 5 512
uadd 7 7 5
umulk 7 7 128
uadd 7 7 6
uloadf32 2 0 @in0
ustoref32 2 7 @in2
endkernel
kernel s.flash_prefill in=4 out=1 threads=16384 tg=128 tgbytes=25600 expect=in0:262144,in1:2097152,in2:2097152,in3:1,out0:262144 rows=16:4096 tag=aten::scaled_dot_product_attention
rtou 0 0
uconst 12 0
utgid 1
ulid 2
loadconstf32 1 32
utor 2 1
div 3 2 1
rtou 3 3
utor 3 3
mul 3 3 1
sub 2 2 3
rtou 4 2
ushrk 5 2 3
uandk 6 2 7
umulk 7 3 16
utor 6 7
uadd 7 7 5
utor 2 4
loadconstf32 1 1
div 2 2 1
rtou 11 2
umulk 8 11 65536
uloadf32 1 12 @in3
utor 2 7
add 5 1 2
addconst 5 5 1
add 6 6 1
addconst 6 6 16
addconst 1 1 64
cmplt 3 1 6
ifbegin 3
mulconst 6 1 1
ifend 
addconst 7 6 15
mulconst 7 7 0.0625
loadconstf32 1 64
cmplt 4 2 1
utor 9 6
umulk 11 5 128
utor 20 11
add 24 20 9
addconst 25 24 2048
umulk 11 6 128
utor 21 11
addconst 21 21 2048
umulk 11 5 16
utor 26 11
addconst 26 26 6144
mulconst 27 9 16
addconst 27 27 4096
utor 28 5
add 28 28 27
umulk 9 7 32
uadd 9 9 4
umulk 9 9 128
uadd 9 9 6
loadconstf32 49 0
loadconstf32 50 0
loadconstf32 51 0
loadconstf32 52 0
loadconstf32 53 0
loadconstf32 54 0
loadconstf32 55 0
loadconstf32 56 0
loadconstf32 57 0
loadconstf32 58 0
loadconstf32 59 0
loadconstf32 60 0
loadconstf32 61 0
loadconstf32 62 0
loadconstf32 63 0
loadconstf32 64 0
ifbegin 4
uloadf32 49 9 @in0+0
uloadf32 50 9 @in0+8
uloadf32 51 9 @in0+16
uloadf32 52 9 @in0+24
uloadf32 53 9 @in0+32
uloadf32 54 9 @in0+40
uloadf32 55 9 @in0+48
uloadf32 56 9 @in0+56
uloadf32 57 9 @in0+64
uloadf32 58 9 @in0+72
uloadf32 59 9 @in0+80
uloadf32 60 9 @in0+88
uloadf32 61 9 @in0+96
uloadf32 62 9 @in0+104
uloadf32 63 9 @in0+112
uloadf32 64 9 @in0+120
ifend 
tgstoref32 49 24 0
tgstoref32 50 24 8
tgstoref32 51 24 16
tgstoref32 52 24 24
tgstoref32 53 24 32
tgstoref32 54 24 40
tgstoref32 55 24 48
tgstoref32 56 24 56
tgstoref32 57 24 64
tgstoref32 58 24 72
tgstoref32 59 24 80
tgstoref32 60 24 88
tgstoref32 61 24 96
tgstoref32 62 24 104
tgstoref32 63 24 112
tgstoref32 64 24 120
loadconstf32 33 0
loadconstf32 34 0
loadconstf32 35 0
loadconstf32 36 0
loadconstf32 37 0
loadconstf32 38 0
loadconstf32 39 0
loadconstf32 40 0
loadconstf32 41 0
loadconstf32 42 0
loadconstf32 43 0
loadconstf32 44 0
loadconstf32 45 0
loadconstf32 46 0
loadconstf32 47 0
loadconstf32 48 0
loadconstf32 10 -1000000000000000019884624838656.0
loadconstf32 11 0
uconst 10 0
loadconstf32 8 0
forbegin 7
barriertg 
uadd 11 10 5
utor 2 11
cmplt 3 2 6
umulk 11 11 128
uadd 11 11 8
uadd 11 11 6
loadconstf32 49 0
loadconstf32 50 0
loadconstf32 51 0
loadconstf32 52 0
loadconstf32 53 0
loadconstf32 54 0
loadconstf32 55 0
loadconstf32 56 0
loadconstf32 57 0
loadconstf32 58 0
loadconstf32 59 0
loadconstf32 60 0
loadconstf32 61 0
loadconstf32 62 0
loadconstf32 63 0
loadconstf32 64 0
ifbegin 3
uloadf32 49 11 @in1+0
uloadf32 50 11 @in1+8
uloadf32 51 11 @in1+16
uloadf32 52 11 @in1+24
uloadf32 53 11 @in1+32
uloadf32 54 11 @in1+40
uloadf32 55 11 @in1+48
uloadf32 56 11 @in1+56
uloadf32 57 11 @in1+64
uloadf32 58 11 @in1+72
uloadf32 59 11 @in1+80
uloadf32 60 11 @in1+88
uloadf32 61 11 @in1+96
uloadf32 62 11 @in1+104
uloadf32 63 11 @in1+112
uloadf32 64 11 @in1+120
ifend 
tgstoref32 49 25 0
tgstoref32 50 25 8
tgstoref32 51 25 16
tgstoref32 52 25 24
tgstoref32 53 25 32
tgstoref32 54 25 40
tgstoref32 55 25 48
tgstoref32 56 25 56
tgstoref32 57 25 64
tgstoref32 58 25 72
tgstoref32 59 25 80
tgstoref32 60 25 88
tgstoref32 61 25 96
tgstoref32 62 25 104
tgstoref32 63 25 112
tgstoref32 64 25 120
loadconstf32 49 0
loadconstf32 50 0
loadconstf32 51 0
loadconstf32 52 0
loadconstf32 53 0
loadconstf32 54 0
loadconstf32 55 0
loadconstf32 56 0
loadconstf32 57 0
loadconstf32 58 0
loadconstf32 59 0
loadconstf32 60 0
loadconstf32 61 0
loadconstf32 62 0
loadconstf32 63 0
loadconstf32 64 0
ifbegin 3
uloadf32 49 11 @in2+0
uloadf32 50 11 @in2+8
uloadf32 51 11 @in2+16
uloadf32 52 11 @in2+24
uloadf32 53 11 @in2+32
uloadf32 54 11 @in2+40
uloadf32 55 11 @in2+48
uloadf32 56 11 @in2+56
uloadf32 57 11 @in2+64
uloadf32 58 11 @in2+72
uloadf32 59 11 @in2+80
uloadf32 60 11 @in2+88
uloadf32 61 11 @in2+96
uloadf32 62 11 @in2+104
uloadf32 63 11 @in2+112
uloadf32 64 11 @in2+120
ifend 
tgstoref32 49 28 0
tgstoref32 50 28 128
tgstoref32 51 28 256
tgstoref32 52 28 384
tgstoref32 53 28 512
tgstoref32 54 28 640
tgstoref32 55 28 768
tgstoref32 56 28 896
tgstoref32 57 28 1024
tgstoref32 58 28 1152
tgstoref32 59 28 1280
tgstoref32 60 28 1408
tgstoref32 61 28 1536
tgstoref32 62 28 1664
tgstoref32 63 28 1792
tgstoref32 64 28 1920
barriertg 
loadconstf32 15 0
loadconstf32 16 0
mulconst 22 20 1
mulconst 23 21 1
loadconstf32 65 32
forbegin 65
tgloadf32 49 22 0
tgloadf32 50 22 1
tgloadf32 51 22 2
tgloadf32 52 22 3
tgloadf32 53 23 0
tgloadf32 54 23 1
tgloadf32 55 23 2
tgloadf32 56 23 3
tgloadf32 57 23 1024
tgloadf32 58 23 1025
tgloadf32 59 23 1026
tgloadf32 60 23 1027
fma 15 49 53 15
fma 16 49 57 16
fma 15 50 54 15
fma 16 50 58 16
fma 15 51 55 15
fma 16 51 59 16
fma 15 52 56 15
fma 16 52 60 16
addconst 22 22 4
addconst 23 23 4
forend 
add 2 8 9
loadconstf32 18 -1000000000000000019884624838656.0
cmplt 3 2 5
ifbegin 3
mulconst 18 15 0.08838834613562
ifend 
addconst 2 2 8
loadconstf32 19 -1000000000000000019884624838656.0
cmplt 3 2 5
ifbegin 3
mulconst 19 16 0.08838834613562
ifend 
add 2 26 9
tgstoref32 18 2 0
tgstoref32 19 2 8
barriertg 
tgloadf32 49 26 0
tgloadf32 50 26 1
tgloadf32 51 26 2
tgloadf32 52 26 3
tgloadf32 53 26 4
tgloadf32 54 26 5
tgloadf32 55 26 6
tgloadf32 56 26 7
tgloadf32 57 26 8
tgloadf32 58 26 9
tgloadf32 59 26 10
tgloadf32 60 26 11
tgloadf32 61 26 12
tgloadf32 62 26 13
tgloadf32 63 26 14
tgloadf32 64 26 15
mulconst 12 10 1
cmplt 3 12 49
ifbegin 3
mulconst 12 49 1
ifend 
cmplt 3 12 50
ifbegin 3
mulconst 12 50 1
ifend 
cmplt 3 12 51
ifbegin 3
mulconst 12 51 1
ifend 
cmplt 3 12 52
ifbegin 3
mulconst 12 52 1
ifend 
cmplt 3 12 53
ifbegin 3
mulconst 12 53 1
ifend 
cmplt 3 12 54
ifbegin 3
mulconst 12 54 1
ifend 
cmplt 3 12 55
ifbegin 3
mulconst 12 55 1
ifend 
cmplt 3 12 56
ifbegin 3
mulconst 12 56 1
ifend 
cmplt 3 12 57
ifbegin 3
mulconst 12 57 1
ifend 
cmplt 3 12 58
ifbegin 3
mulconst 12 58 1
ifend 
cmplt 3 12 59
ifbegin 3
mulconst 12 59 1
ifend 
cmplt 3 12 60
ifbegin 3
mulconst 12 60 1
ifend 
cmplt 3 12 61
ifbegin 3
mulconst 12 61 1
ifend 
cmplt 3 12 62
ifbegin 3
mulconst 12 62 1
ifend 
cmplt 3 12 63
ifbegin 3
mulconst 12 63 1
ifend 
cmplt 3 12 64
ifbegin 3
mulconst 12 64 1
ifend 
sub 13 10 12
exp 13 13
loadconstf32 14 0
loadconstf32 1 -500000000000000009942312419328.0
cmplt 3 1 49
sub 49 49 12
exp 49 49
mul 49 49 3
add 14 14 49
cmplt 3 1 50
sub 50 50 12
exp 50 50
mul 50 50 3
add 14 14 50
cmplt 3 1 51
sub 51 51 12
exp 51 51
mul 51 51 3
add 14 14 51
cmplt 3 1 52
sub 52 52 12
exp 52 52
mul 52 52 3
add 14 14 52
cmplt 3 1 53
sub 53 53 12
exp 53 53
mul 53 53 3
add 14 14 53
cmplt 3 1 54
sub 54 54 12
exp 54 54
mul 54 54 3
add 14 14 54
cmplt 3 1 55
sub 55 55 12
exp 55 55
mul 55 55 3
add 14 14 55
cmplt 3 1 56
sub 56 56 12
exp 56 56
mul 56 56 3
add 14 14 56
cmplt 3 1 57
sub 57 57 12
exp 57 57
mul 57 57 3
add 14 14 57
cmplt 3 1 58
sub 58 58 12
exp 58 58
mul 58 58 3
add 14 14 58
cmplt 3 1 59
sub 59 59 12
exp 59 59
mul 59 59 3
add 14 14 59
cmplt 3 1 60
sub 60 60 12
exp 60 60
mul 60 60 3
add 14 14 60
cmplt 3 1 61
sub 61 61 12
exp 61 61
mul 61 61 3
add 14 14 61
cmplt 3 1 62
sub 62 62 12
exp 62 62
mul 62 62 3
add 14 14 62
cmplt 3 1 63
sub 63 63 12
exp 63 63
mul 63 63 3
add 14 14 63
cmplt 3 1 64
sub 64 64 12
exp 64 64
mul 64 64 3
add 14 14 64
mul 11 11 13
add 11 11 14
mulconst 10 12 1
mul 33 33 13
tgloadf32 29 27 0
tgloadf32 30 27 1
tgloadf32 31 27 2
tgloadf32 32 27 3
fma 33 49 29 33
fma 33 50 30 33
fma 33 51 31 33
fma 33 52 32 33
tgloadf32 29 27 4
tgloadf32 30 27 5
tgloadf32 31 27 6
tgloadf32 32 27 7
fma 33 53 29 33
fma 33 54 30 33
fma 33 55 31 33
fma 33 56 32 33
tgloadf32 29 27 8
tgloadf32 30 27 9
tgloadf32 31 27 10
tgloadf32 32 27 11
fma 33 57 29 33
fma 33 58 30 33
fma 33 59 31 33
fma 33 60 32 33
tgloadf32 29 27 12
tgloadf32 30 27 13
tgloadf32 31 27 14
tgloadf32 32 27 15
fma 33 61 29 33
fma 33 62 30 33
fma 33 63 31 33
fma 33 64 32 33
mul 34 34 13
tgloadf32 29 27 128
tgloadf32 30 27 129
tgloadf32 31 27 130
tgloadf32 32 27 131
fma 34 49 29 34
fma 34 50 30 34
fma 34 51 31 34
fma 34 52 32 34
tgloadf32 29 27 132
tgloadf32 30 27 133
tgloadf32 31 27 134
tgloadf32 32 27 135
fma 34 53 29 34
fma 34 54 30 34
fma 34 55 31 34
fma 34 56 32 34
tgloadf32 29 27 136
tgloadf32 30 27 137
tgloadf32 31 27 138
tgloadf32 32 27 139
fma 34 57 29 34
fma 34 58 30 34
fma 34 59 31 34
fma 34 60 32 34
tgloadf32 29 27 140
tgloadf32 30 27 141
tgloadf32 31 27 142
tgloadf32 32 27 143
fma 34 61 29 34
fma 34 62 30 34
fma 34 63 31 34
fma 34 64 32 34
mul 35 35 13
tgloadf32 29 27 256
tgloadf32 30 27 257
tgloadf32 31 27 258
tgloadf32 32 27 259
fma 35 49 29 35
fma 35 50 30 35
fma 35 51 31 35
fma 35 52 32 35
tgloadf32 29 27 260
tgloadf32 30 27 261
tgloadf32 31 27 262
tgloadf32 32 27 263
fma 35 53 29 35
fma 35 54 30 35
fma 35 55 31 35
fma 35 56 32 35
tgloadf32 29 27 264
tgloadf32 30 27 265
tgloadf32 31 27 266
tgloadf32 32 27 267
fma 35 57 29 35
fma 35 58 30 35
fma 35 59 31 35
fma 35 60 32 35
tgloadf32 29 27 268
tgloadf32 30 27 269
tgloadf32 31 27 270
tgloadf32 32 27 271
fma 35 61 29 35
fma 35 62 30 35
fma 35 63 31 35
fma 35 64 32 35
mul 36 36 13
tgloadf32 29 27 384
tgloadf32 30 27 385
tgloadf32 31 27 386
tgloadf32 32 27 387
fma 36 49 29 36
fma 36 50 30 36
fma 36 51 31 36
fma 36 52 32 36
tgloadf32 29 27 388
tgloadf32 30 27 389
tgloadf32 31 27 390
tgloadf32 32 27 391
fma 36 53 29 36
fma 36 54 30 36
fma 36 55 31 36
fma 36 56 32 36
tgloadf32 29 27 392
tgloadf32 30 27 393
tgloadf32 31 27 394
tgloadf32 32 27 395
fma 36 57 29 36
fma 36 58 30 36
fma 36 59 31 36
fma 36 60 32 36
tgloadf32 29 27 396
tgloadf32 30 27 397
tgloadf32 31 27 398
tgloadf32 32 27 399
fma 36 61 29 36
fma 36 62 30 36
fma 36 63 31 36
fma 36 64 32 36
mul 37 37 13
tgloadf32 29 27 512
tgloadf32 30 27 513
tgloadf32 31 27 514
tgloadf32 32 27 515
fma 37 49 29 37
fma 37 50 30 37
fma 37 51 31 37
fma 37 52 32 37
tgloadf32 29 27 516
tgloadf32 30 27 517
tgloadf32 31 27 518
tgloadf32 32 27 519
fma 37 53 29 37
fma 37 54 30 37
fma 37 55 31 37
fma 37 56 32 37
tgloadf32 29 27 520
tgloadf32 30 27 521
tgloadf32 31 27 522
tgloadf32 32 27 523
fma 37 57 29 37
fma 37 58 30 37
fma 37 59 31 37
fma 37 60 32 37
tgloadf32 29 27 524
tgloadf32 30 27 525
tgloadf32 31 27 526
tgloadf32 32 27 527
fma 37 61 29 37
fma 37 62 30 37
fma 37 63 31 37
fma 37 64 32 37
mul 38 38 13
tgloadf32 29 27 640
tgloadf32 30 27 641
tgloadf32 31 27 642
tgloadf32 32 27 643
fma 38 49 29 38
fma 38 50 30 38
fma 38 51 31 38
fma 38 52 32 38
tgloadf32 29 27 644
tgloadf32 30 27 645
tgloadf32 31 27 646
tgloadf32 32 27 647
fma 38 53 29 38
fma 38 54 30 38
fma 38 55 31 38
fma 38 56 32 38
tgloadf32 29 27 648
tgloadf32 30 27 649
tgloadf32 31 27 650
tgloadf32 32 27 651
fma 38 57 29 38
fma 38 58 30 38
fma 38 59 31 38
fma 38 60 32 38
tgloadf32 29 27 652
tgloadf32 30 27 653
tgloadf32 31 27 654
tgloadf32 32 27 655
fma 38 61 29 38
fma 38 62 30 38
fma 38 63 31 38
fma 38 64 32 38
mul 39 39 13
tgloadf32 29 27 768
tgloadf32 30 27 769
tgloadf32 31 27 770
tgloadf32 32 27 771
fma 39 49 29 39
fma 39 50 30 39
fma 39 51 31 39
fma 39 52 32 39
tgloadf32 29 27 772
tgloadf32 30 27 773
tgloadf32 31 27 774
tgloadf32 32 27 775
fma 39 53 29 39
fma 39 54 30 39
fma 39 55 31 39
fma 39 56 32 39
tgloadf32 29 27 776
tgloadf32 30 27 777
tgloadf32 31 27 778
tgloadf32 32 27 779
fma 39 57 29 39
fma 39 58 30 39
fma 39 59 31 39
fma 39 60 32 39
tgloadf32 29 27 780
tgloadf32 30 27 781
tgloadf32 31 27 782
tgloadf32 32 27 783
fma 39 61 29 39
fma 39 62 30 39
fma 39 63 31 39
fma 39 64 32 39
mul 40 40 13
tgloadf32 29 27 896
tgloadf32 30 27 897
tgloadf32 31 27 898
tgloadf32 32 27 899
fma 40 49 29 40
fma 40 50 30 40
fma 40 51 31 40
fma 40 52 32 40
tgloadf32 29 27 900
tgloadf32 30 27 901
tgloadf32 31 27 902
tgloadf32 32 27 903
fma 40 53 29 40
fma 40 54 30 40
fma 40 55 31 40
fma 40 56 32 40
tgloadf32 29 27 904
tgloadf32 30 27 905
tgloadf32 31 27 906
tgloadf32 32 27 907
fma 40 57 29 40
fma 40 58 30 40
fma 40 59 31 40
fma 40 60 32 40
tgloadf32 29 27 908
tgloadf32 30 27 909
tgloadf32 31 27 910
tgloadf32 32 27 911
fma 40 61 29 40
fma 40 62 30 40
fma 40 63 31 40
fma 40 64 32 40
mul 41 41 13
tgloadf32 29 27 1024
tgloadf32 30 27 1025
tgloadf32 31 27 1026
tgloadf32 32 27 1027
fma 41 49 29 41
fma 41 50 30 41
fma 41 51 31 41
fma 41 52 32 41
tgloadf32 29 27 1028
tgloadf32 30 27 1029
tgloadf32 31 27 1030
tgloadf32 32 27 1031
fma 41 53 29 41
fma 41 54 30 41
fma 41 55 31 41
fma 41 56 32 41
tgloadf32 29 27 1032
tgloadf32 30 27 1033
tgloadf32 31 27 1034
tgloadf32 32 27 1035
fma 41 57 29 41
fma 41 58 30 41
fma 41 59 31 41
fma 41 60 32 41
tgloadf32 29 27 1036
tgloadf32 30 27 1037
tgloadf32 31 27 1038
tgloadf32 32 27 1039
fma 41 61 29 41
fma 41 62 30 41
fma 41 63 31 41
fma 41 64 32 41
mul 42 42 13
tgloadf32 29 27 1152
tgloadf32 30 27 1153
tgloadf32 31 27 1154
tgloadf32 32 27 1155
fma 42 49 29 42
fma 42 50 30 42
fma 42 51 31 42
fma 42 52 32 42
tgloadf32 29 27 1156
tgloadf32 30 27 1157
tgloadf32 31 27 1158
tgloadf32 32 27 1159
fma 42 53 29 42
fma 42 54 30 42
fma 42 55 31 42
fma 42 56 32 42
tgloadf32 29 27 1160
tgloadf32 30 27 1161
tgloadf32 31 27 1162
tgloadf32 32 27 1163
fma 42 57 29 42
fma 42 58 30 42
fma 42 59 31 42
fma 42 60 32 42
tgloadf32 29 27 1164
tgloadf32 30 27 1165
tgloadf32 31 27 1166
tgloadf32 32 27 1167
fma 42 61 29 42
fma 42 62 30 42
fma 42 63 31 42
fma 42 64 32 42
mul 43 43 13
tgloadf32 29 27 1280
tgloadf32 30 27 1281
tgloadf32 31 27 1282
tgloadf32 32 27 1283
fma 43 49 29 43
fma 43 50 30 43
fma 43 51 31 43
fma 43 52 32 43
tgloadf32 29 27 1284
tgloadf32 30 27 1285
tgloadf32 31 27 1286
tgloadf32 32 27 1287
fma 43 53 29 43
fma 43 54 30 43
fma 43 55 31 43
fma 43 56 32 43
tgloadf32 29 27 1288
tgloadf32 30 27 1289
tgloadf32 31 27 1290
tgloadf32 32 27 1291
fma 43 57 29 43
fma 43 58 30 43
fma 43 59 31 43
fma 43 60 32 43
tgloadf32 29 27 1292
tgloadf32 30 27 1293
tgloadf32 31 27 1294
tgloadf32 32 27 1295
fma 43 61 29 43
fma 43 62 30 43
fma 43 63 31 43
fma 43 64 32 43
mul 44 44 13
tgloadf32 29 27 1408
tgloadf32 30 27 1409
tgloadf32 31 27 1410
tgloadf32 32 27 1411
fma 44 49 29 44
fma 44 50 30 44
fma 44 51 31 44
fma 44 52 32 44
tgloadf32 29 27 1412
tgloadf32 30 27 1413
tgloadf32 31 27 1414
tgloadf32 32 27 1415
fma 44 53 29 44
fma 44 54 30 44
fma 44 55 31 44
fma 44 56 32 44
tgloadf32 29 27 1416
tgloadf32 30 27 1417
tgloadf32 31 27 1418
tgloadf32 32 27 1419
fma 44 57 29 44
fma 44 58 30 44
fma 44 59 31 44
fma 44 60 32 44
tgloadf32 29 27 1420
tgloadf32 30 27 1421
tgloadf32 31 27 1422
tgloadf32 32 27 1423
fma 44 61 29 44
fma 44 62 30 44
fma 44 63 31 44
fma 44 64 32 44
mul 45 45 13
tgloadf32 29 27 1536
tgloadf32 30 27 1537
tgloadf32 31 27 1538
tgloadf32 32 27 1539
fma 45 49 29 45
fma 45 50 30 45
fma 45 51 31 45
fma 45 52 32 45
tgloadf32 29 27 1540
tgloadf32 30 27 1541
tgloadf32 31 27 1542
tgloadf32 32 27 1543
fma 45 53 29 45
fma 45 54 30 45
fma 45 55 31 45
fma 45 56 32 45
tgloadf32 29 27 1544
tgloadf32 30 27 1545
tgloadf32 31 27 1546
tgloadf32 32 27 1547
fma 45 57 29 45
fma 45 58 30 45
fma 45 59 31 45
fma 45 60 32 45
tgloadf32 29 27 1548
tgloadf32 30 27 1549
tgloadf32 31 27 1550
tgloadf32 32 27 1551
fma 45 61 29 45
fma 45 62 30 45
fma 45 63 31 45
fma 45 64 32 45
mul 46 46 13
tgloadf32 29 27 1664
tgloadf32 30 27 1665
tgloadf32 31 27 1666
tgloadf32 32 27 1667
fma 46 49 29 46
fma 46 50 30 46
fma 46 51 31 46
fma 46 52 32 46
tgloadf32 29 27 1668
tgloadf32 30 27 1669
tgloadf32 31 27 1670
tgloadf32 32 27 1671
fma 46 53 29 46
fma 46 54 30 46
fma 46 55 31 46
fma 46 56 32 46
tgloadf32 29 27 1672
tgloadf32 30 27 1673
tgloadf32 31 27 1674
tgloadf32 32 27 1675
fma 46 57 29 46
fma 46 58 30 46
fma 46 59 31 46
fma 46 60 32 46
tgloadf32 29 27 1676
tgloadf32 30 27 1677
tgloadf32 31 27 1678
tgloadf32 32 27 1679
fma 46 61 29 46
fma 46 62 30 46
fma 46 63 31 46
fma 46 64 32 46
mul 47 47 13
tgloadf32 29 27 1792
tgloadf32 30 27 1793
tgloadf32 31 27 1794
tgloadf32 32 27 1795
fma 47 49 29 47
fma 47 50 30 47
fma 47 51 31 47
fma 47 52 32 47
tgloadf32 29 27 1796
tgloadf32 30 27 1797
tgloadf32 31 27 1798
tgloadf32 32 27 1799
fma 47 53 29 47
fma 47 54 30 47
fma 47 55 31 47
fma 47 56 32 47
tgloadf32 29 27 1800
tgloadf32 30 27 1801
tgloadf32 31 27 1802
tgloadf32 32 27 1803
fma 47 57 29 47
fma 47 58 30 47
fma 47 59 31 47
fma 47 60 32 47
tgloadf32 29 27 1804
tgloadf32 30 27 1805
tgloadf32 31 27 1806
tgloadf32 32 27 1807
fma 47 61 29 47
fma 47 62 30 47
fma 47 63 31 47
fma 47 64 32 47
mul 48 48 13
tgloadf32 29 27 1920
tgloadf32 30 27 1921
tgloadf32 31 27 1922
tgloadf32 32 27 1923
fma 48 49 29 48
fma 48 50 30 48
fma 48 51 31 48
fma 48 52 32 48
tgloadf32 29 27 1924
tgloadf32 30 27 1925
tgloadf32 31 27 1926
tgloadf32 32 27 1927
fma 48 53 29 48
fma 48 54 30 48
fma 48 55 31 48
fma 48 56 32 48
tgloadf32 29 27 1928
tgloadf32 30 27 1929
tgloadf32 31 27 1930
tgloadf32 32 27 1931
fma 48 57 29 48
fma 48 58 30 48
fma 48 59 31 48
fma 48 60 32 48
tgloadf32 29 27 1932
tgloadf32 30 27 1933
tgloadf32 31 27 1934
tgloadf32 32 27 1935
fma 48 61 29 48
fma 48 62 30 48
fma 48 63 31 48
fma 48 64 32 48
uaddk 10 10 16
addconst 8 8 16
forend 
ifbegin 4
loadconstf32 1 0.000000000000000000000000000001
cmplt 3 11 1
ifbegin 3
mulconst 11 1 1
ifend 
loadconstf32 1 1
div 1 1 11
mul 33 33 1
ustoref32 33 9 @out0+0
mul 34 34 1
ustoref32 34 9 @out0+8
mul 35 35 1
ustoref32 35 9 @out0+16
mul 36 36 1
ustoref32 36 9 @out0+24
mul 37 37 1
ustoref32 37 9 @out0+32
mul 38 38 1
ustoref32 38 9 @out0+40
mul 39 39 1
ustoref32 39 9 @out0+48
mul 40 40 1
ustoref32 40 9 @out0+56
mul 41 41 1
ustoref32 41 9 @out0+64
mul 42 42 1
ustoref32 42 9 @out0+72
mul 43 43 1
ustoref32 43 9 @out0+80
mul 44 44 1
ustoref32 44 9 @out0+88
mul 45 45 1
ustoref32 45 9 @out0+96
mul 46 46 1
ustoref32 46 9 @out0+104
mul 47 47 1
ustoref32 47 9 @out0+112
mul 48 48 1
ustoref32 48 9 @out0+120
ifend 
endkernel
kernel s.vec_add in=2 out=1 threads=262144 tg=64 tgbytes=16 expect=in0:262144,in1:262144,out0:262144 rows=1:4096 tag=aten::add
rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
add 1 1 2
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_tiled_fma_2 in=2 out=1 threads=11008 tg=128 tgbytes=25600 rows=64:11008 layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0 tag=aten::matmul
words 00000090 00000180 00000281 01000113 42ac0000 01003c8f 49078227 08004390 0100628f 49044221 490b9c23 08038490 00008586 00000080 00006686 00000040 00004787 0000000f 00004888 00000004 00000f82 00000000 0001ef86 00000004 00000982 00000000 00000113 00000000 00000213 00000000 00000313 00000000 00000413 00000000 00000513 00000000 00000613 00000000 00000713 00000000 00000813 00000000 00000913 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000f13 00000000 00001013 00000000 00001113 00000000 00001213 00000000 00001313 00000000 00001413 00000000 00001513 00000000 00001613 00000000 00001713 00000000 00001813 00000000 00001913 00000000 00001a13 00000000 00001b13 00000000 00001c13 00000000 00001d13 00000000 00001e13 00000000 00001f13 00000000 00800013 00000000 00800113 00000000 00800213 00000000 00800313 00000000 00800413 00000000 00800513 00000000 00800613 00000000 00800713 00000000 00800813 00000000 00800913 00000000 00800a13 00000000 00800b13 00000000 00800c13 00000000 00800d13 00000000 00800e13 00000000 00800f13 00000000 00801013 00000000 00801113 00000000 00801213 00000000 00801313 00000000 00801413 00000000 00801513 00000000 00801613 00000000 00801713 00000000 00801813 00000000 00801913 00000000 00801a13 00000000 00801b13 00000000 00801c13 00000000 00801d13 00000000 00801e13 00000000 00801f13 00000000 01000013 00000000 01800013 43000000 01800040 00004a85 00000000 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000080 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000100 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000180 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000000 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 462c0000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000080 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 462c0000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000100 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 462c0000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000180 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 462c0000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00000050 00010a86 00000008 0101598f 0000ea86 00000008 00014a85 00000880 01015a8f 01800113 42000000 01800140 09034960 00000000 09034a60 00000001 09034b60 00000002 09034c60 00000003 09034d60 00000004 09034e60 00000005 09034f60 00000006 09035060 00000007 09033160 00000000 09033260 00000001 09033360 00000002 09033460 00000003 09033560 00000004 09033660 00000005 09033760 00000006 09033860 00000007 48262122 00000001 482a2222 00000002 482e2322 00000003 48322422 00000004 48362522 00000005 483a2622 00000006 483e2722 00000007 48422822 00000008 48264922 00000009 482a4a22 0000000a 482e4b22 0000000b 48324c22 0000000c 48364d22 0000000d 483a4e22 0000000e 483e4f22 0000000f 48425022 00000010 48267122 00000011 482a7222 00000012 482e7322 00000013 48327422 00000014 48367522 00000015 483a7622 00000016 483e7722 00000017 48427822 00000018 48269922 00000019 482a9a22 0000001a 482e9b22 0000001b 48329c22 0000001c 48369d22 0000001d 483a9e22 0000001e 483e9f22 0000001f 48c28022 00000020 48a6a122 00000021 48aaa222 00000022 48aea322 00000023 48b2a422 00000024 48b6a522 00000025 48baa622 00000026 48bea722 00000027 48c2a822 00000028 48a6c922 00000029 48aaca22 0000002a 48aecb22 0000002b 48b2cc22 0000002c 48b6cd22 0000002d 48bace22 0000002e 48becf22 0000002f 48c2d022 00000030 48a6f122 00000031 48aaf222 00000032 48aef322 00000033 48b2f422 00000034 48b6f522 00000035 48baf622 00000036 48bef722 00000037 48c2f822 00000038 48a71922 00000039 48ab1a22 0000003a 48af1b22 0000003b 48b31c22 0000003c 48b71d22 0000003d 48bb1e22 0000003e 48bf1f22 0000003f 49430022 00000040 0903392a 42880000 09035a2a 43040000 00000041 00000050 00012985 00000020 00000041 00010a86 00000008 00194a83 0000eb86 00000008 00156b83 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001818e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001828e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001838e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001848e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001858e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001868e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001878e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001888e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001898e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018a8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018b8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018c8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018d8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018e8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018f8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001908e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001918e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001928e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001938e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001948e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001958e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001968e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001978e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001988e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001998e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019a8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019b8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019c8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019d8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019e8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019f8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081808e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081818e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081828e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081838e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081848e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081858e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081868e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081878e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081888e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081898e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818a8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818b8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818c8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818d8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818e8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818f8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081908e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081918e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081928e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081938e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081948e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081958e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081968e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081978e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081988e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081998e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819a8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819b8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819c8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819d8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819e8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819f8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0101808e 00000000 00000073 00000001
endkernel
kernel s.act_sigmoid in=1 out=1 threads=704512 tg=64 tgbytes=16 expect=in0:704512,out0:704512 rows=1:11008 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 1 1
ustoref32 1 0 @out0
endkernel
kernel s.vec_mul in=2 out=1 threads=704512 tg=64 tgbytes=16 expect=in0:704512,in1:704512,out0:704512 rows=1:11008 tag=aten::mul
rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
mul 1 1 2
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_tiled_fma_3 in=2 out=1 threads=4096 tg=128 tgbytes=25600 rows=64:4096 layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0 tag=aten::matmul
words 00000090 00000180 00000281 01000113 42000000 01003c8f 49078227 08004390 0100628f 49044221 490b9c23 08038490 00008586 00000080 00006686 00000040 00004787 0000000f 00004888 00000004 00000f82 00000000 0001ef86 00000004 00000982 00000000 00000113 00000000 00000213 00000000 00000313 00000000 00000413 00000000 00000513 00000000 00000613 00000000 00000713 00000000 00000813 00000000 00000913 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000f13 00000000 00001013 00000000 00001113 00000000 00001213 00000000 00001313 00000000 00001413 00000000 00001513 00000000 00001613 00000000 00001713 00000000 00001813 00000000 00001913 00000000 00001a13 00000000 00001b13 00000000 00001c13 00000000 00001d13 00000000 00001e13 00000000 00001f13 00000000 00800013 00000000 00800113 00000000 00800213 00000000 00800313 00000000 00800413 00000000 00800513 00000000 00800613 00000000 00800713 00000000 00800813 00000000 00800913 00000000 00800a13 00000000 00800b13 00000000 00800c13 00000000 00800d13 00000000 00800e13 00000000 00800f13 00000000 00801013 00000000 00801113 00000000 00801213 00000000 00801313 00000000 00801413 00000000 00801513 00000000 00801613 00000000 00801713 00000000 00801813 00000000 00801913 00000000 00801a13 00000000 00801b13 00000000 00801c13 00000000 00801d13 00000000 00801e13 00000000 00801f13 00000000 01000013 00000000 01800013 43ac0000 01800040 00004a85 00000000 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00002b00 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000080 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00002b00 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000100 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00002b00 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000180 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 42800000 497b9d70 01001d72 00014a86 00002b00 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000000 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00001580 00316b83 003d6b83 00016e9d 00000000 00014a86 00000056 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000080 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00001580 00316b83 003d6b83 00016e9d 00000000 00014a86 00000056 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000100 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00001580 00316b83 003d6b83 00016e9d 00000000 00014a86 00000056 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000180 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00001580 00316b83 003d6b83 00016e9d 00000000 00014a86 00000056 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00000050 00010a86 00000008 0101598f 0000ea86 00000008 00014a85 00000880 01015a8f 01800113 42000000 01800140 09034960 00000000 09034a60 00000001 09034b60 00000002 09034c60 00000003 09034d60 00000004 09034e60 00000005 09034f60 00000006 09035060 00000007 09033160 00000000 09033260 00000001 09033360 00000002 09033460 00000003 09033560 00000004 09033660 00000005 09033760 00000006 09033860 00000007 48262122 00000001 482a2222 00000002 482e2322 00000003 48322422 00000004 48362522 00000005 483a2622 00000006 483e2722 00000007 48422822 00000008 48264922 00000009 482a4a22 0000000a 482e4b22 0000000b 48324c22 0000000c 48364d22 0000000d 483a4e22 0000000e 483e4f22 0000000f 48425022 00000010 48267122 00000011 482a7222 00000012 482e7322 00000013 48327422 00000014 48367522 00000015 483a7622 00000016 483e7722 00000017 48427822 00000018 48269922 00000019 482a9a22 0000001a 482e9b22 0000001b 48329c22 0000001c 48369d22 0000001d 483a9e22 0000001e 483e9f22 0000001f 48c28022 00000020 48a6a122 00000021 48aaa222 00000022 48aea322 00000023 48b2a422 00000024 48b6a522 00000025 48baa622 00000026 48bea722 00000027 48c2a822 00000028 48a6c922 00000029 48aaca22 0000002a 48aecb22 0000002b 48b2cc22 0000002c 48b6cd22 0000002d 48bace22 0000002e 48becf22 0000002f 48c2d022 00000030 48a6f122 00000031 48aaf222 00000032 48aef322 00000033 48b2f422 00000034 48b6f522 00000035 48baf622 00000036 48bef722 00000037 48c2f822 00000038 48a71922 00000039 48ab1a22 0000003a 48af1b22 0000003b 48b31c22 0000003c 48b71d22 0000003d 48bb1e22 0000003e 48bf1f22 0000003f 49430022 00000040 0903392a 42880000 09035a2a 43040000 00000041 00000050 00012985 00000020 00000041 00010a86 00000008 00194a83 0000eb86 00000008 00156b83 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001818e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001828e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001838e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001848e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001858e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001868e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001878e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001888e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001898e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018a8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018b8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018c8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018d8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018e8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018f8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001908e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001918e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001928e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001938e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001948e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001958e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001968e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001978e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001988e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001998e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019a8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019b8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019c8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019d8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019e8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019f8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081808e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081818e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081828e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081838e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081848e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081858e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081868e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081878e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081888e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081898e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818a8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818b8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818c8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818d8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818e8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818f8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081908e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081918e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081928e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081938e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081948e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081958e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081968e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081978e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081988e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081998e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819a8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819b8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819c8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819d8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819e8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819f8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 42800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0101808e 00000000 00000073 00000001
endkernel
kernel s.silu_mul in=2 out=1 threads=704512 tg=64 tgbytes=16 expect=in0:704512,in1:704512,out0:704512 rows=1:11008 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 2 1
mul 1 1 2
uloadf32 3 0 @in1
mul 1 1 3
ustoref32 1 0 @out0
endkernel
kernel s.rmsnorm_2 in=2 out=1 threads=4096 tg=256 tgbytes=1024 expect=in0:65536,in1:4096,out0:65536 rows=1:256 tag=aten::rms_norm
rtou 0 0
loadconstf32 4 0
loadconstf32 7 4096
utgid 1
ulid 2
loadlid 1
umulk 3 1 4096
uaddk 4 2 0
loadconstf32 8 16
forbegin 8
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
mul 2 2 2
add 4 4 2
ifend 
uaddk 4 4 256
forend 
tgstoref32 4 1 0
barriertg 
loadconstf32 8 128
cmplt 10 1 8
ifbegin 10
addconst 9 1 128
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 64
cmplt 10 1 8
ifbegin 10
addconst 9 1 64
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 32
cmplt 10 1 8
ifbegin 10
addconst 9 1 32
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 16
cmplt 10 1 8
ifbegin 10
addconst 9 1 16
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 8
cmplt 10 1 8
ifbegin 10
addconst 9 1 8
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 4
cmplt 10 1 8
ifbegin 10
addconst 9 1 4
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 2
cmplt 10 1 8
ifbegin 10
addconst 9 1 2
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 1
cmplt 10 1 8
ifbegin 10
addconst 9 1 1
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 9 0
tgloadf32 8 9 0
mulconst 8 8 0.000244140625
addconst 8 8 0.000001
rsqrt 8 8
uaddk 4 2 0
loadconstf32 10 16
forbegin 10
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
mul 2 2 8
uloadf32 3 4 @in1
mul 2 2 3
ustoref32 2 5 @out0
ifend 
uaddk 4 4 256
forend 
endkernel
kernel s.matmul_q4_tiled_fma_4 in=2 out=1 threads=4096 tg=128 tgbytes=25600 rows=64:4096 layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0 tag=aten::matmul
words 00000090 00000180 00000281 01000113 42000000 01003c8f 49078227 08004390 0100628f 49044221 490b9c23 08038490 00008586 00000080 00006686 00000040 00004787 0000000f 00004888 00000004 00000f82 00000000 0001ef86 00000004 00000982 00000000 00000113 00000000 00000213 00000000 00000313 00000000 00000413 00000000 00000513 00000000 00000613 00000000 00000713 00000000 00000813 00000000 00000913 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000f13 00000000 00001013 00000000 00001113 00000000 00001213 00000000 00001313 00000000 00001413 00000000 00001513 00000000 00001613 00000000 00001713 00000000 00001813 00000000 00001913 00000000 00001a13 00000000 00001b13 00000000 00001c13 00000000 00001d13 00000000 00001e13 00000000 00001f13 00000000 00800013 00000000 00800113 00000000 00800213 00000000 00800313 00000000 00800413 00000000 00800513 00000000 00800613 00000000 00800713 00000000 00800813 00000000 00800913 00000000 00800a13 00000000 00800b13 00000000 00800c13 00000000 00800d13 00000000 00800e13 00000000 00800f13 00000000 00801013 00000000 00801113 00000000 00801213 00000000 00801313 00000000 00801413 00000000 00801513 00000000 00801613 00000000 00801713 00000000 00801813 00000000 00801913 00000000 00801a13 00000000 00801b13 00000000 00801c13 00000000 00801d13 00000000 00801e13 00000000 00801f13 00000000 01000013 00000000 01800013 43000000 01800040 00004a85 00000000 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000080 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000100 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000180 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000000 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00200000 01015f8c 00220000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000080 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00200000 01015f8c 00220000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000100 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00200000 01015f8c 00220000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000180 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00200000 01015f8c 00220000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00000050 00010a86 00000008 0101598f 0000ea86 00000008 00014a85 00000880 01015a8f 01800113 42000000 01800140 09034960 00000000 09034a60 00000001 09034b60 00000002 09034c60 00000003 09034d60 00000004 09034e60 00000005 09034f60 00000006 09035060 00000007 09033160 00000000 09033260 00000001 09033360 00000002 09033460 00000003 09033560 00000004 09033660 00000005 09033760 00000006 09033860 00000007 48262122 00000001 482a2222 00000002 482e2322 00000003 48322422 00000004 48362522 00000005 483a2622 00000006 483e2722 00000007 48422822 00000008 48264922 00000009 482a4a22 0000000a 482e4b22 0000000b 48324c22 0000000c 48364d22 0000000d 483a4e22 0000000e 483e4f22 0000000f 48425022 00000010 48267122 00000011 482a7222 00000012 482e7322 00000013 48327422 00000014 48367522 00000015 483a7622 00000016 483e7722 00000017 48427822 00000018 48269922 00000019 482a9a22 0000001a 482e9b22 0000001b 48329c22 0000001c 48369d22 0000001d 483a9e22 0000001e 483e9f22 0000001f 48c28022 00000020 48a6a122 00000021 48aaa222 00000022 48aea322 00000023 48b2a422 00000024 48b6a522 00000025 48baa622 00000026 48bea722 00000027 48c2a822 00000028 48a6c922 00000029 48aaca22 0000002a 48aecb22 0000002b 48b2cc22 0000002c 48b6cd22 0000002d 48bace22 0000002e 48becf22 0000002f 48c2d022 00000030 48a6f122 00000031 48aaf222 00000032 48aef322 00000033 48b2f422 00000034 48b6f522 00000035 48baf622 00000036 48bef722 00000037 48c2f822 00000038 48a71922 00000039 48ab1a22 0000003a 48af1b22 0000003b 48b31c22 0000003c 48b71d22 0000003d 48bb1e22 0000003e 48bf1f22 0000003f 49430022 00000040 0903392a 42880000 09035a2a 43040000 00000041 00000050 00012985 00000020 00000041 00010a86 00000008 00194a83 0000eb86 00000008 00156b83 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001818e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001828e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001838e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001848e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001858e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001868e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001878e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001888e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001898e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018a8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018b8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018c8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018d8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018e8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018f8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001908e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001918e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001928e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001938e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001948e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001958e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001968e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001978e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001988e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001998e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019a8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019b8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019c8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019d8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019e8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019f8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081808e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081818e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081828e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081838e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081848e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081858e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081868e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081878e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081888e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081898e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818a8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818b8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818c8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818d8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818e8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818f8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081908e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081918e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081928e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081938e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081948e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081958e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081968e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081978e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081988e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081998e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819a8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819b8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819c8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819d8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819e8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819f8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0101808e 00000000 00000073 00000001
endkernel
kernel s.rope_2 in=2 out=1 threads=32768 tg=64 tgbytes=16 expect=in0:65536,in1:1,out0:65536 rows=1:2048
rtou 0 0
uconst 1 0
uandk 2 0 63
ushrk 3 0 6
umulk 5 3 128
uadd 5 5 2
uloadf32 1 1 @in1
utor 10 3
addconst 10 10 0.5
mulconst 10 10 0.03125
rtou 4 10
utor 9 4
add 1 1 9
mulconst 1 1 0.25
utor 9 2
mulconst 9 9 -0.17988946039016
exp 2 9
mul 3 1 2
mulconst 11 3 0.1591549430919
addconst 11 11 0.5
rtou 6 11
utor 11 6
mulconst 13 11 6.28125
sub 12 3 13
mulconst 13 11 0.0019353071795864769
sub 12 12 13
mul 14 12 12
mulconst 15 12 1
mulconst 4 12 1
loadconstf32 16 1
loadconstf32 5 1
mul 15 15 14
mulconst 15 15 -0.16666666666667
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.5
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.05
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.08333333333333
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.02380952380952
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.03333333333333
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.01388888888889
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.01785714285714
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00909090909091
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.01111111111111
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00641025641026
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00757575757576
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.0047619047619
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00549450549451
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00367647058824
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00416666666667
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00292397660819
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00326797385621
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00238095238095
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00263157894737
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00197628458498
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.0021645021645
add 5 5 16
uloadf32 6 5 @in0
uloadf32 7 5 @in0+64
mul 8 6 5
mul 3 7 4
sub 8 8 3
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
ustoref32 8 5 @out0+64
endkernel
kernel s.kv_scatter_2 in=3 out=1 threads=65536 tg=64 tgbytes=16 expect=in0:65536,in1:1,in2:2097152,out0:2097152 rows=1:4096
rtou 0 0
uconst 1 0
uloadf32 1 1 @in1
rtou 2 1
utor 3 0
addconst 3 3 0.5
mulconst 3 3 0.0078125
rtou 3 3
utor 4 0
utor 5 3
mulconst 5 5 128
sub 4 4 5
rtou 6 4
utor 6 3
addconst 6 6 0.5
mulconst 6 6 0.03125
rtou 4 6
utor 7 3
utor 8 4
mulconst 8 8 32
sub 7 7 8
rtou 5 7
uadd 7 2 4
umulk 5 5 512
uadd 7 7 5
umulk 7 7 128
uadd 7 7 6
uloadf32 2 0 @in0
ustoref32 2 7 @in2
endkernel
kernel s.flash_prefill_2 in=4 out=1 threads=4096 tg=128 tgbytes=25600 expect=in0:65536,in1:2097152,in2:2097152,in3:1,out0:65536 rows=16:4096 tag=aten::scaled_dot_product_attention
rtou 0 0
uconst 12 0
utgid 1
ulid 2
loadconstf32 1 32
utor 2 1
div 3 2 1
rtou 3 3
utor 3 3
mul 3 3 1
sub 2 2 3
rtou 4 2
ushrk 5 2 3
uandk 6 2 7
umulk 7 3 16
utor 6 7
uadd 7 7 5
utor 2 4
loadconstf32 1 1
div 2 2 1
rtou 11 2
umulk 8 11 65536
uloadf32 1 12 @in3
utor 2 7
add 5 1 2
addconst 5 5 1
add 6 6 1
addconst 6 6 16
addconst 1 1 16
cmplt 3 1 6
ifbegin 3
mulconst 6 1 1
ifend 
addconst 7 6 15
mulconst 7 7 0.0625
loadconstf32 1 16
cmplt 4 2 1
utor 9 6
umulk 11 5 128
utor 20 11
add 24 20 9
addconst 25 24 2048
umulk 11 6 128
utor 21 11
addconst 21 21 2048
umulk 11 5 16
utor 26 11
addconst 26 26 6144
mulconst 27 9 16
addconst 27 27 4096
utor 28 5
add 28 28 27
umulk 9 7 32
uadd 9 9 4
umulk 9 9 128
uadd 9 9 6
loadconstf32 49 0
loadconstf32 50 0
loadconstf32 51 0
loadconstf32 52 0
loadconstf32 53 0
loadconstf32 54 0
loadconstf32 55 0
loadconstf32 56 0
loadconstf32 57 0
loadconstf32 58 0
loadconstf32 59 0
loadconstf32 60 0
loadconstf32 61 0
loadconstf32 62 0
loadconstf32 63 0
loadconstf32 64 0
ifbegin 4
uloadf32 49 9 @in0+0
uloadf32 50 9 @in0+8
uloadf32 51 9 @in0+16
uloadf32 52 9 @in0+24
uloadf32 53 9 @in0+32
uloadf32 54 9 @in0+40
uloadf32 55 9 @in0+48
uloadf32 56 9 @in0+56
uloadf32 57 9 @in0+64
uloadf32 58 9 @in0+72
uloadf32 59 9 @in0+80
uloadf32 60 9 @in0+88
uloadf32 61 9 @in0+96
uloadf32 62 9 @in0+104
uloadf32 63 9 @in0+112
uloadf32 64 9 @in0+120
ifend 
tgstoref32 49 24 0
tgstoref32 50 24 8
tgstoref32 51 24 16
tgstoref32 52 24 24
tgstoref32 53 24 32
tgstoref32 54 24 40
tgstoref32 55 24 48
tgstoref32 56 24 56
tgstoref32 57 24 64
tgstoref32 58 24 72
tgstoref32 59 24 80
tgstoref32 60 24 88
tgstoref32 61 24 96
tgstoref32 62 24 104
tgstoref32 63 24 112
tgstoref32 64 24 120
loadconstf32 33 0
loadconstf32 34 0
loadconstf32 35 0
loadconstf32 36 0
loadconstf32 37 0
loadconstf32 38 0
loadconstf32 39 0
loadconstf32 40 0
loadconstf32 41 0
loadconstf32 42 0
loadconstf32 43 0
loadconstf32 44 0
loadconstf32 45 0
loadconstf32 46 0
loadconstf32 47 0
loadconstf32 48 0
loadconstf32 10 -1000000000000000019884624838656.0
loadconstf32 11 0
uconst 10 0
loadconstf32 8 0
forbegin 7
barriertg 
uadd 11 10 5
utor 2 11
cmplt 3 2 6
umulk 11 11 128
uadd 11 11 8
uadd 11 11 6
loadconstf32 49 0
loadconstf32 50 0
loadconstf32 51 0
loadconstf32 52 0
loadconstf32 53 0
loadconstf32 54 0
loadconstf32 55 0
loadconstf32 56 0
loadconstf32 57 0
loadconstf32 58 0
loadconstf32 59 0
loadconstf32 60 0
loadconstf32 61 0
loadconstf32 62 0
loadconstf32 63 0
loadconstf32 64 0
ifbegin 3
uloadf32 49 11 @in1+0
uloadf32 50 11 @in1+8
uloadf32 51 11 @in1+16
uloadf32 52 11 @in1+24
uloadf32 53 11 @in1+32
uloadf32 54 11 @in1+40
uloadf32 55 11 @in1+48
uloadf32 56 11 @in1+56
uloadf32 57 11 @in1+64
uloadf32 58 11 @in1+72
uloadf32 59 11 @in1+80
uloadf32 60 11 @in1+88
uloadf32 61 11 @in1+96
uloadf32 62 11 @in1+104
uloadf32 63 11 @in1+112
uloadf32 64 11 @in1+120
ifend 
tgstoref32 49 25 0
tgstoref32 50 25 8
tgstoref32 51 25 16
tgstoref32 52 25 24
tgstoref32 53 25 32
tgstoref32 54 25 40
tgstoref32 55 25 48
tgstoref32 56 25 56
tgstoref32 57 25 64
tgstoref32 58 25 72
tgstoref32 59 25 80
tgstoref32 60 25 88
tgstoref32 61 25 96
tgstoref32 62 25 104
tgstoref32 63 25 112
tgstoref32 64 25 120
loadconstf32 49 0
loadconstf32 50 0
loadconstf32 51 0
loadconstf32 52 0
loadconstf32 53 0
loadconstf32 54 0
loadconstf32 55 0
loadconstf32 56 0
loadconstf32 57 0
loadconstf32 58 0
loadconstf32 59 0
loadconstf32 60 0
loadconstf32 61 0
loadconstf32 62 0
loadconstf32 63 0
loadconstf32 64 0
ifbegin 3
uloadf32 49 11 @in2+0
uloadf32 50 11 @in2+8
uloadf32 51 11 @in2+16
uloadf32 52 11 @in2+24
uloadf32 53 11 @in2+32
uloadf32 54 11 @in2+40
uloadf32 55 11 @in2+48
uloadf32 56 11 @in2+56
uloadf32 57 11 @in2+64
uloadf32 58 11 @in2+72
uloadf32 59 11 @in2+80
uloadf32 60 11 @in2+88
uloadf32 61 11 @in2+96
uloadf32 62 11 @in2+104
uloadf32 63 11 @in2+112
uloadf32 64 11 @in2+120
ifend 
tgstoref32 49 28 0
tgstoref32 50 28 128
tgstoref32 51 28 256
tgstoref32 52 28 384
tgstoref32 53 28 512
tgstoref32 54 28 640
tgstoref32 55 28 768
tgstoref32 56 28 896
tgstoref32 57 28 1024
tgstoref32 58 28 1152
tgstoref32 59 28 1280
tgstoref32 60 28 1408
tgstoref32 61 28 1536
tgstoref32 62 28 1664
tgstoref32 63 28 1792
tgstoref32 64 28 1920
barriertg 
loadconstf32 15 0
loadconstf32 16 0
mulconst 22 20 1
mulconst 23 21 1
loadconstf32 65 32
forbegin 65
tgloadf32 49 22 0
tgloadf32 50 22 1
tgloadf32 51 22 2
tgloadf32 52 22 3
tgloadf32 53 23 0
tgloadf32 54 23 1
tgloadf32 55 23 2
tgloadf32 56 23 3
tgloadf32 57 23 1024
tgloadf32 58 23 1025
tgloadf32 59 23 1026
tgloadf32 60 23 1027
fma 15 49 53 15
fma 16 49 57 16
fma 15 50 54 15
fma 16 50 58 16
fma 15 51 55 15
fma 16 51 59 16
fma 15 52 56 15
fma 16 52 60 16
addconst 22 22 4
addconst 23 23 4
forend 
add 2 8 9
loadconstf32 18 -1000000000000000019884624838656.0
cmplt 3 2 5
ifbegin 3
mulconst 18 15 0.08838834613562
ifend 
addconst 2 2 8
loadconstf32 19 -1000000000000000019884624838656.0
cmplt 3 2 5
ifbegin 3
mulconst 19 16 0.08838834613562
ifend 
add 2 26 9
tgstoref32 18 2 0
tgstoref32 19 2 8
barriertg 
tgloadf32 49 26 0
tgloadf32 50 26 1
tgloadf32 51 26 2
tgloadf32 52 26 3
tgloadf32 53 26 4
tgloadf32 54 26 5
tgloadf32 55 26 6
tgloadf32 56 26 7
tgloadf32 57 26 8
tgloadf32 58 26 9
tgloadf32 59 26 10
tgloadf32 60 26 11
tgloadf32 61 26 12
tgloadf32 62 26 13
tgloadf32 63 26 14
tgloadf32 64 26 15
mulconst 12 10 1
cmplt 3 12 49
ifbegin 3
mulconst 12 49 1
ifend 
cmplt 3 12 50
ifbegin 3
mulconst 12 50 1
ifend 
cmplt 3 12 51
ifbegin 3
mulconst 12 51 1
ifend 
cmplt 3 12 52
ifbegin 3
mulconst 12 52 1
ifend 
cmplt 3 12 53
ifbegin 3
mulconst 12 53 1
ifend 
cmplt 3 12 54
ifbegin 3
mulconst 12 54 1
ifend 
cmplt 3 12 55
ifbegin 3
mulconst 12 55 1
ifend 
cmplt 3 12 56
ifbegin 3
mulconst 12 56 1
ifend 
cmplt 3 12 57
ifbegin 3
mulconst 12 57 1
ifend 
cmplt 3 12 58
ifbegin 3
mulconst 12 58 1
ifend 
cmplt 3 12 59
ifbegin 3
mulconst 12 59 1
ifend 
cmplt 3 12 60
ifbegin 3
mulconst 12 60 1
ifend 
cmplt 3 12 61
ifbegin 3
mulconst 12 61 1
ifend 
cmplt 3 12 62
ifbegin 3
mulconst 12 62 1
ifend 
cmplt 3 12 63
ifbegin 3
mulconst 12 63 1
ifend 
cmplt 3 12 64
ifbegin 3
mulconst 12 64 1
ifend 
sub 13 10 12
exp 13 13
loadconstf32 14 0
loadconstf32 1 -500000000000000009942312419328.0
cmplt 3 1 49
sub 49 49 12
exp 49 49
mul 49 49 3
add 14 14 49
cmplt 3 1 50
sub 50 50 12
exp 50 50
mul 50 50 3
add 14 14 50
cmplt 3 1 51
sub 51 51 12
exp 51 51
mul 51 51 3
add 14 14 51
cmplt 3 1 52
sub 52 52 12
exp 52 52
mul 52 52 3
add 14 14 52
cmplt 3 1 53
sub 53 53 12
exp 53 53
mul 53 53 3
add 14 14 53
cmplt 3 1 54
sub 54 54 12
exp 54 54
mul 54 54 3
add 14 14 54
cmplt 3 1 55
sub 55 55 12
exp 55 55
mul 55 55 3
add 14 14 55
cmplt 3 1 56
sub 56 56 12
exp 56 56
mul 56 56 3
add 14 14 56
cmplt 3 1 57
sub 57 57 12
exp 57 57
mul 57 57 3
add 14 14 57
cmplt 3 1 58
sub 58 58 12
exp 58 58
mul 58 58 3
add 14 14 58
cmplt 3 1 59
sub 59 59 12
exp 59 59
mul 59 59 3
add 14 14 59
cmplt 3 1 60
sub 60 60 12
exp 60 60
mul 60 60 3
add 14 14 60
cmplt 3 1 61
sub 61 61 12
exp 61 61
mul 61 61 3
add 14 14 61
cmplt 3 1 62
sub 62 62 12
exp 62 62
mul 62 62 3
add 14 14 62
cmplt 3 1 63
sub 63 63 12
exp 63 63
mul 63 63 3
add 14 14 63
cmplt 3 1 64
sub 64 64 12
exp 64 64
mul 64 64 3
add 14 14 64
mul 11 11 13
add 11 11 14
mulconst 10 12 1
mul 33 33 13
tgloadf32 29 27 0
tgloadf32 30 27 1
tgloadf32 31 27 2
tgloadf32 32 27 3
fma 33 49 29 33
fma 33 50 30 33
fma 33 51 31 33
fma 33 52 32 33
tgloadf32 29 27 4
tgloadf32 30 27 5
tgloadf32 31 27 6
tgloadf32 32 27 7
fma 33 53 29 33
fma 33 54 30 33
fma 33 55 31 33
fma 33 56 32 33
tgloadf32 29 27 8
tgloadf32 30 27 9
tgloadf32 31 27 10
tgloadf32 32 27 11
fma 33 57 29 33
fma 33 58 30 33
fma 33 59 31 33
fma 33 60 32 33
tgloadf32 29 27 12
tgloadf32 30 27 13
tgloadf32 31 27 14
tgloadf32 32 27 15
fma 33 61 29 33
fma 33 62 30 33
fma 33 63 31 33
fma 33 64 32 33
mul 34 34 13
tgloadf32 29 27 128
tgloadf32 30 27 129
tgloadf32 31 27 130
tgloadf32 32 27 131
fma 34 49 29 34
fma 34 50 30 34
fma 34 51 31 34
fma 34 52 32 34
tgloadf32 29 27 132
tgloadf32 30 27 133
tgloadf32 31 27 134
tgloadf32 32 27 135
fma 34 53 29 34
fma 34 54 30 34
fma 34 55 31 34
fma 34 56 32 34
tgloadf32 29 27 136
tgloadf32 30 27 137
tgloadf32 31 27 138
tgloadf32 32 27 139
fma 34 57 29 34
fma 34 58 30 34
fma 34 59 31 34
fma 34 60 32 34
tgloadf32 29 27 140
tgloadf32 30 27 141
tgloadf32 31 27 142
tgloadf32 32 27 143
fma 34 61 29 34
fma 34 62 30 34
fma 34 63 31 34
fma 34 64 32 34
mul 35 35 13
tgloadf32 29 27 256
tgloadf32 30 27 257
tgloadf32 31 27 258
tgloadf32 32 27 259
fma 35 49 29 35
fma 35 50 30 35
fma 35 51 31 35
fma 35 52 32 35
tgloadf32 29 27 260
tgloadf32 30 27 261
tgloadf32 31 27 262
tgloadf32 32 27 263
fma 35 53 29 35
fma 35 54 30 35
fma 35 55 31 35
fma 35 56 32 35
tgloadf32 29 27 264
tgloadf32 30 27 265
tgloadf32 31 27 266
tgloadf32 32 27 267
fma 35 57 29 35
fma 35 58 30 35
fma 35 59 31 35
fma 35 60 32 35
tgloadf32 29 27 268
tgloadf32 30 27 269
tgloadf32 31 27 270
tgloadf32 32 27 271
fma 35 61 29 35
fma 35 62 30 35
fma 35 63 31 35
fma 35 64 32 35
mul 36 36 13
tgloadf32 29 27 384
tgloadf32 30 27 385
tgloadf32 31 27 386
tgloadf32 32 27 387
fma 36 49 29 36
fma 36 50 30 36
fma 36 51 31 36
fma 36 52 32 36
tgloadf32 29 27 388
tgloadf32 30 27 389
tgloadf32 31 27 390
tgloadf32 32 27 391
fma 36 53 29 36
fma 36 54 30 36
fma 36 55 31 36
fma 36 56 32 36
tgloadf32 29 27 392
tgloadf32 30 27 393
tgloadf32 31 27 394
tgloadf32 32 27 395
fma 36 57 29 36
fma 36 58 30 36
fma 36 59 31 36
fma 36 60 32 36
tgloadf32 29 27 396
tgloadf32 30 27 397
tgloadf32 31 27 398
tgloadf32 32 27 399
fma 36 61 29 36
fma 36 62 30 36
fma 36 63 31 36
fma 36 64 32 36
mul 37 37 13
tgloadf32 29 27 512
tgloadf32 30 27 513
tgloadf32 31 27 514
tgloadf32 32 27 515
fma 37 49 29 37
fma 37 50 30 37
fma 37 51 31 37
fma 37 52 32 37
tgloadf32 29 27 516
tgloadf32 30 27 517
tgloadf32 31 27 518
tgloadf32 32 27 519
fma 37 53 29 37
fma 37 54 30 37
fma 37 55 31 37
fma 37 56 32 37
tgloadf32 29 27 520
tgloadf32 30 27 521
tgloadf32 31 27 522
tgloadf32 32 27 523
fma 37 57 29 37
fma 37 58 30 37
fma 37 59 31 37
fma 37 60 32 37
tgloadf32 29 27 524
tgloadf32 30 27 525
tgloadf32 31 27 526
tgloadf32 32 27 527
fma 37 61 29 37
fma 37 62 30 37
fma 37 63 31 37
fma 37 64 32 37
mul 38 38 13
tgloadf32 29 27 640
tgloadf32 30 27 641
tgloadf32 31 27 642
tgloadf32 32 27 643
fma 38 49 29 38
fma 38 50 30 38
fma 38 51 31 38
fma 38 52 32 38
tgloadf32 29 27 644
tgloadf32 30 27 645
tgloadf32 31 27 646
tgloadf32 32 27 647
fma 38 53 29 38
fma 38 54 30 38
fma 38 55 31 38
fma 38 56 32 38
tgloadf32 29 27 648
tgloadf32 30 27 649
tgloadf32 31 27 650
tgloadf32 32 27 651
fma 38 57 29 38
fma 38 58 30 38
fma 38 59 31 38
fma 38 60 32 38
tgloadf32 29 27 652
tgloadf32 30 27 653
tgloadf32 31 27 654
tgloadf32 32 27 655
fma 38 61 29 38
fma 38 62 30 38
fma 38 63 31 38
fma 38 64 32 38
mul 39 39 13
tgloadf32 29 27 768
tgloadf32 30 27 769
tgloadf32 31 27 770
tgloadf32 32 27 771
fma 39 49 29 39
fma 39 50 30 39
fma 39 51 31 39
fma 39 52 32 39
tgloadf32 29 27 772
tgloadf32 30 27 773
tgloadf32 31 27 774
tgloadf32 32 27 775
fma 39 53 29 39
fma 39 54 30 39
fma 39 55 31 39
fma 39 56 32 39
tgloadf32 29 27 776
tgloadf32 30 27 777
tgloadf32 31 27 778
tgloadf32 32 27 779
fma 39 57 29 39
fma 39 58 30 39
fma 39 59 31 39
fma 39 60 32 39
tgloadf32 29 27 780
tgloadf32 30 27 781
tgloadf32 31 27 782
tgloadf32 32 27 783
fma 39 61 29 39
fma 39 62 30 39
fma 39 63 31 39
fma 39 64 32 39
mul 40 40 13
tgloadf32 29 27 896
tgloadf32 30 27 897
tgloadf32 31 27 898
tgloadf32 32 27 899
fma 40 49 29 40
fma 40 50 30 40
fma 40 51 31 40
fma 40 52 32 40
tgloadf32 29 27 900
tgloadf32 30 27 901
tgloadf32 31 27 902
tgloadf32 32 27 903
fma 40 53 29 40
fma 40 54 30 40
fma 40 55 31 40
fma 40 56 32 40
tgloadf32 29 27 904
tgloadf32 30 27 905
tgloadf32 31 27 906
tgloadf32 32 27 907
fma 40 57 29 40
fma 40 58 30 40
fma 40 59 31 40
fma 40 60 32 40
tgloadf32 29 27 908
tgloadf32 30 27 909
tgloadf32 31 27 910
tgloadf32 32 27 911
fma 40 61 29 40
fma 40 62 30 40
fma 40 63 31 40
fma 40 64 32 40
mul 41 41 13
tgloadf32 29 27 1024
tgloadf32 30 27 1025
tgloadf32 31 27 1026
tgloadf32 32 27 1027
fma 41 49 29 41
fma 41 50 30 41
fma 41 51 31 41
fma 41 52 32 41
tgloadf32 29 27 1028
tgloadf32 30 27 1029
tgloadf32 31 27 1030
tgloadf32 32 27 1031
fma 41 53 29 41
fma 41 54 30 41
fma 41 55 31 41
fma 41 56 32 41
tgloadf32 29 27 1032
tgloadf32 30 27 1033
tgloadf32 31 27 1034
tgloadf32 32 27 1035
fma 41 57 29 41
fma 41 58 30 41
fma 41 59 31 41
fma 41 60 32 41
tgloadf32 29 27 1036
tgloadf32 30 27 1037
tgloadf32 31 27 1038
tgloadf32 32 27 1039
fma 41 61 29 41
fma 41 62 30 41
fma 41 63 31 41
fma 41 64 32 41
mul 42 42 13
tgloadf32 29 27 1152
tgloadf32 30 27 1153
tgloadf32 31 27 1154
tgloadf32 32 27 1155
fma 42 49 29 42
fma 42 50 30 42
fma 42 51 31 42
fma 42 52 32 42
tgloadf32 29 27 1156
tgloadf32 30 27 1157
tgloadf32 31 27 1158
tgloadf32 32 27 1159
fma 42 53 29 42
fma 42 54 30 42
fma 42 55 31 42
fma 42 56 32 42
tgloadf32 29 27 1160
tgloadf32 30 27 1161
tgloadf32 31 27 1162
tgloadf32 32 27 1163
fma 42 57 29 42
fma 42 58 30 42
fma 42 59 31 42
fma 42 60 32 42
tgloadf32 29 27 1164
tgloadf32 30 27 1165
tgloadf32 31 27 1166
tgloadf32 32 27 1167
fma 42 61 29 42
fma 42 62 30 42
fma 42 63 31 42
fma 42 64 32 42
mul 43 43 13
tgloadf32 29 27 1280
tgloadf32 30 27 1281
tgloadf32 31 27 1282
tgloadf32 32 27 1283
fma 43 49 29 43
fma 43 50 30 43
fma 43 51 31 43
fma 43 52 32 43
tgloadf32 29 27 1284
tgloadf32 30 27 1285
tgloadf32 31 27 1286
tgloadf32 32 27 1287
fma 43 53 29 43
fma 43 54 30 43
fma 43 55 31 43
fma 43 56 32 43
tgloadf32 29 27 1288
tgloadf32 30 27 1289
tgloadf32 31 27 1290
tgloadf32 32 27 1291
fma 43 57 29 43
fma 43 58 30 43
fma 43 59 31 43
fma 43 60 32 43
tgloadf32 29 27 1292
tgloadf32 30 27 1293
tgloadf32 31 27 1294
tgloadf32 32 27 1295
fma 43 61 29 43
fma 43 62 30 43
fma 43 63 31 43
fma 43 64 32 43
mul 44 44 13
tgloadf32 29 27 1408
tgloadf32 30 27 1409
tgloadf32 31 27 1410
tgloadf32 32 27 1411
fma 44 49 29 44
fma 44 50 30 44
fma 44 51 31 44
fma 44 52 32 44
tgloadf32 29 27 1412
tgloadf32 30 27 1413
tgloadf32 31 27 1414
tgloadf32 32 27 1415
fma 44 53 29 44
fma 44 54 30 44
fma 44 55 31 44
fma 44 56 32 44
tgloadf32 29 27 1416
tgloadf32 30 27 1417
tgloadf32 31 27 1418
tgloadf32 32 27 1419
fma 44 57 29 44
fma 44 58 30 44
fma 44 59 31 44
fma 44 60 32 44
tgloadf32 29 27 1420
tgloadf32 30 27 1421
tgloadf32 31 27 1422
tgloadf32 32 27 1423
fma 44 61 29 44
fma 44 62 30 44
fma 44 63 31 44
fma 44 64 32 44
mul 45 45 13
tgloadf32 29 27 1536
tgloadf32 30 27 1537
tgloadf32 31 27 1538
tgloadf32 32 27 1539
fma 45 49 29 45
fma 45 50 30 45
fma 45 51 31 45
fma 45 52 32 45
tgloadf32 29 27 1540
tgloadf32 30 27 1541
tgloadf32 31 27 1542
tgloadf32 32 27 1543
fma 45 53 29 45
fma 45 54 30 45
fma 45 55 31 45
fma 45 56 32 45
tgloadf32 29 27 1544
tgloadf32 30 27 1545
tgloadf32 31 27 1546
tgloadf32 32 27 1547
fma 45 57 29 45
fma 45 58 30 45
fma 45 59 31 45
fma 45 60 32 45
tgloadf32 29 27 1548
tgloadf32 30 27 1549
tgloadf32 31 27 1550
tgloadf32 32 27 1551
fma 45 61 29 45
fma 45 62 30 45
fma 45 63 31 45
fma 45 64 32 45
mul 46 46 13
tgloadf32 29 27 1664
tgloadf32 30 27 1665
tgloadf32 31 27 1666
tgloadf32 32 27 1667
fma 46 49 29 46
fma 46 50 30 46
fma 46 51 31 46
fma 46 52 32 46
tgloadf32 29 27 1668
tgloadf32 30 27 1669
tgloadf32 31 27 1670
tgloadf32 32 27 1671
fma 46 53 29 46
fma 46 54 30 46
fma 46 55 31 46
fma 46 56 32 46
tgloadf32 29 27 1672
tgloadf32 30 27 1673
tgloadf32 31 27 1674
tgloadf32 32 27 1675
fma 46 57 29 46
fma 46 58 30 46
fma 46 59 31 46
fma 46 60 32 46
tgloadf32 29 27 1676
tgloadf32 30 27 1677
tgloadf32 31 27 1678
tgloadf32 32 27 1679
fma 46 61 29 46
fma 46 62 30 46
fma 46 63 31 46
fma 46 64 32 46
mul 47 47 13
tgloadf32 29 27 1792
tgloadf32 30 27 1793
tgloadf32 31 27 1794
tgloadf32 32 27 1795
fma 47 49 29 47
fma 47 50 30 47
fma 47 51 31 47
fma 47 52 32 47
tgloadf32 29 27 1796
tgloadf32 30 27 1797
tgloadf32 31 27 1798
tgloadf32 32 27 1799
fma 47 53 29 47
fma 47 54 30 47
fma 47 55 31 47
fma 47 56 32 47
tgloadf32 29 27 1800
tgloadf32 30 27 1801
tgloadf32 31 27 1802
tgloadf32 32 27 1803
fma 47 57 29 47
fma 47 58 30 47
fma 47 59 31 47
fma 47 60 32 47
tgloadf32 29 27 1804
tgloadf32 30 27 1805
tgloadf32 31 27 1806
tgloadf32 32 27 1807
fma 47 61 29 47
fma 47 62 30 47
fma 47 63 31 47
fma 47 64 32 47
mul 48 48 13
tgloadf32 29 27 1920
tgloadf32 30 27 1921
tgloadf32 31 27 1922
tgloadf32 32 27 1923
fma 48 49 29 48
fma 48 50 30 48
fma 48 51 31 48
fma 48 52 32 48
tgloadf32 29 27 1924
tgloadf32 30 27 1925
tgloadf32 31 27 1926
tgloadf32 32 27 1927
fma 48 53 29 48
fma 48 54 30 48
fma 48 55 31 48
fma 48 56 32 48
tgloadf32 29 27 1928
tgloadf32 30 27 1929
tgloadf32 31 27 1930
tgloadf32 32 27 1931
fma 48 57 29 48
fma 48 58 30 48
fma 48 59 31 48
fma 48 60 32 48
tgloadf32 29 27 1932
tgloadf32 30 27 1933
tgloadf32 31 27 1934
tgloadf32 32 27 1935
fma 48 61 29 48
fma 48 62 30 48
fma 48 63 31 48
fma 48 64 32 48
uaddk 10 10 16
addconst 8 8 16
forend 
ifbegin 4
loadconstf32 1 0.000000000000000000000000000001
cmplt 3 11 1
ifbegin 3
mulconst 11 1 1
ifend 
loadconstf32 1 1
div 1 1 11
mul 33 33 1
ustoref32 33 9 @out0+0
mul 34 34 1
ustoref32 34 9 @out0+8
mul 35 35 1
ustoref32 35 9 @out0+16
mul 36 36 1
ustoref32 36 9 @out0+24
mul 37 37 1
ustoref32 37 9 @out0+32
mul 38 38 1
ustoref32 38 9 @out0+40
mul 39 39 1
ustoref32 39 9 @out0+48
mul 40 40 1
ustoref32 40 9 @out0+56
mul 41 41 1
ustoref32 41 9 @out0+64
mul 42 42 1
ustoref32 42 9 @out0+72
mul 43 43 1
ustoref32 43 9 @out0+80
mul 44 44 1
ustoref32 44 9 @out0+88
mul 45 45 1
ustoref32 45 9 @out0+96
mul 46 46 1
ustoref32 46 9 @out0+104
mul 47 47 1
ustoref32 47 9 @out0+112
mul 48 48 1
ustoref32 48 9 @out0+120
ifend 
endkernel
kernel s.vec_add_2 in=2 out=1 threads=65536 tg=64 tgbytes=16 expect=in0:65536,in1:65536,out0:65536 rows=1:4096 tag=aten::add
rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
add 1 1 2
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_tiled_fma_5 in=2 out=1 threads=11008 tg=128 tgbytes=25600 rows=64:11008 layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0 tag=aten::matmul
words 00000090 00000180 00000281 01000113 42ac0000 01003c8f 49078227 08004390 0100628f 49044221 490b9c23 08038490 00008586 00000080 00006686 00000040 00004787 0000000f 00004888 00000004 00000f82 00000000 0001ef86 00000004 00000982 00000000 00000113 00000000 00000213 00000000 00000313 00000000 00000413 00000000 00000513 00000000 00000613 00000000 00000713 00000000 00000813 00000000 00000913 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000f13 00000000 00001013 00000000 00001113 00000000 00001213 00000000 00001313 00000000 00001413 00000000 00001513 00000000 00001613 00000000 00001713 00000000 00001813 00000000 00001913 00000000 00001a13 00000000 00001b13 00000000 00001c13 00000000 00001d13 00000000 00001e13 00000000 00001f13 00000000 00800013 00000000 00800113 00000000 00800213 00000000 00800313 00000000 00800413 00000000 00800513 00000000 00800613 00000000 00800713 00000000 00800813 00000000 00800913 00000000 00800a13 00000000 00800b13 00000000 00800c13 00000000 00800d13 00000000 00800e13 00000000 00800f13 00000000 00801013 00000000 00801113 00000000 00801213 00000000 00801313 00000000 00801413 00000000 00801513 00000000 00801613 00000000 00801713 00000000 00801813 00000000 00801913 00000000 00801a13 00000000 00801b13 00000000 00801c13 00000000 00801d13 00000000 00801e13 00000000 00801f13 00000000 01000013 00000000 01800013 43000000 01800040 00004a85 00000000 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000080 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000100 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000180 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00001000 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000000 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 462c0000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000080 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 462c0000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000100 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 462c0000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000180 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 462c0000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00000800 00316b83 003d6b83 00016e9d 00000000 00014a86 00000020 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00000050 00010a86 00000008 0101598f 0000ea86 00000008 00014a85 00000880 01015a8f 01800113 42000000 01800140 09034960 00000000 09034a60 00000001 09034b60 00000002 09034c60 00000003 09034d60 00000004 09034e60 00000005 09034f60 00000006 09035060 00000007 09033160 00000000 09033260 00000001 09033360 00000002 09033460 00000003 09033560 00000004 09033660 00000005 09033760 00000006 09033860 00000007 48262122 00000001 482a2222 00000002 482e2322 00000003 48322422 00000004 48362522 00000005 483a2622 00000006 483e2722 00000007 48422822 00000008 48264922 00000009 482a4a22 0000000a 482e4b22 0000000b 48324c22 0000000c 48364d22 0000000d 483a4e22 0000000e 483e4f22 0000000f 48425022 00000010 48267122 00000011 482a7222 00000012 482e7322 00000013 48327422 00000014 48367522 00000015 483a7622 00000016 483e7722 00000017 48427822 00000018 48269922 00000019 482a9a22 0000001a 482e9b22 0000001b 48329c22 0000001c 48369d22 0000001d 483a9e22 0000001e 483e9f22 0000001f 48c28022 00000020 48a6a122 00000021 48aaa222 00000022 48aea322 00000023 48b2a422 00000024 48b6a522 00000025 48baa622 00000026 48bea722 00000027 48c2a822 00000028 48a6c922 00000029 48aaca22 0000002a 48aecb22 0000002b 48b2cc22 0000002c 48b6cd22 0000002d 48bace22 0000002e 48becf22 0000002f 48c2d022 00000030 48a6f122 00000031 48aaf222 00000032 48aef322 00000033 48b2f422 00000034 48b6f522 00000035 48baf622 00000036 48bef722 00000037 48c2f822 00000038 48a71922 00000039 48ab1a22 0000003a 48af1b22 0000003b 48b31c22 0000003c 48b71d22 0000003d 48bb1e22 0000003e 48bf1f22 0000003f 49430022 00000040 0903392a 42880000 09035a2a 43040000 00000041 00000050 00012985 00000020 00000041 00010a86 00000008 00194a83 0000eb86 00000008 00156b83 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001818e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001828e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001838e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001848e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001858e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001868e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001878e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001888e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001898e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018a8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018b8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018c8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018d8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018e8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00018f8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001908e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001918e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001928e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001938e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001948e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001958e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001968e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001978e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001988e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0001998e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019a8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019b8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019c8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019d8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019e8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00019f8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081808e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081818e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081828e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081838e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081848e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081858e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081868e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081878e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081888e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081898e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818a8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818b8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818c8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818d8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818e8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00818f8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081908e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081918e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081928e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081938e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081948e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081958e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081968e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081978e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081988e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0081998e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819a8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819b8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819c8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819d8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819e8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 00819f8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 462c0000 497b9b70 496fbd21 01001d72 00018c86 00002b00 00358c83 0101808e 00000000 00000073 00000001
endkernel
kernel s.act_sigmoid_2 in=1 out=1 threads=176128 tg=64 tgbytes=16 expect=in0:176128,out0:176128 rows=1:11008 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 1 1
ustoref32 1 0 @out0
endkernel
kernel s.vec_mul_2 in=2 out=1 threads=176128 tg=64 tgbytes=16 expect=in0:176128,in1:176128,out0:176128 rows=1:11008 tag=aten::mul
rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
mul 1 1 2
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_tiled_fma_6 in=2 out=1 threads=4096 tg=128 tgbytes=25600 rows=64:4096 layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0 tag=aten::matmul
words 00000090 00000180 00000281 01000113 42000000 01003c8f 49078227 08004390 0100628f 49044221 490b9c23 08038490 00008586 00000080 00006686 00000040 00004787 0000000f 00004888 00000004 00000f82 00000000 0001ef86 00000004 00000982 00000000 00000113 00000000 00000213 00000000 00000313 00000000 00000413 00000000 00000513 00000000 00000613 00000000 00000713 00000000 00000813 00000000 00000913 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000f13 00000000 00001013 00000000 00001113 00000000 00001213 00000000 00001313 00000000 00001413 00000000 00001513 00000000 00001613 00000000 00001713 00000000 00001813 00000000 00001913 00000000 00001a13 00000000 00001b13 00000000 00001c13 00000000 00001d13 00000000 00001e13 00000000 00001f13 00000000 00800013 00000000 00800113 00000000 00800213 00000000 00800313 00000000 00800413 00000000 00800513 00000000 00800613 00000000 00800713 00000000 00800813 00000000 00800913 00000000 00800a13 00000000 00800b13 00000000 00800c13 00000000 00800d13 00000000 00800e13 00000000 00800f13 00000000 00801013 00000000 00801113 00000000 00801213 00000000 00801313 00000000 00801413 00000000 00801513 00000000 00801613 00000000 00801713 00000000 00801813 00000000 00801913 00000000 00801a13 00000000 00801b13 00000000 00801c13 00000000 00801d13 00000000 00801e13 00000000 00801f13 00000000 01000013 00000000 01800013 43ac0000 01800040 00004a85 00000000 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00002b00 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000080 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00002b00 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000100 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00002b00 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000180 00014b88 00000003 00014c87 00000007 00018a86 00000110 002d4a83 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 002cca83 01015c8f 01001e13 41800000 497b9d70 01001d72 00014a86 00002b00 00254a83 00018c86 00000004 00314a83 0101418d 00000000 0101428d 00000001 0101438d 00000002 0101448d 00000003 00000073 09036161 00000000 09036261 00000044 09036361 00000088 09036461 000000cc 00004a85 00000000 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00001580 00316b83 003d6b83 00016e9d 00000000 00014a86 00000056 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000080 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00001580 00316b83 003d6b83 00016e9d 00000000 00014a86 00000056 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000100 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00001580 00316b83 003d6b83 00016e9d 00000000 00014a86 00000056 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00004a85 00000180 00014b88 00000002 00014c87 00000003 00018a86 00000420 002d4a83 00014a85 00000880 01015b8f 01000113 00000000 01000213 00000000 01000313 00000000 01000413 00000000 01000513 00000000 01000613 00000000 01000713 00000000 01000813 00000000 002caa83 01015c8f 01001e13 45800000 497b9d70 01001d72 00018d86 00000008 0025ad83 0001ab88 00000001 00014c86 00001580 00316b83 003d6b83 00016e9d 00000000 00014a86 00000056 0001ad88 00000007 00354a83 01015e8c 00560000 01015f8c 005b6000 0001cb88 00000000 00016b87 0000000f 0101618f 497c2123 49782121 0001cb88 00000004 00016b87 0000000f 0101628f 497c4223 49784221 0001cb88 00000008 00016b87 0000000f 0101638f 497c6323 49786321 0001cb88 0000000c 00016b87 0000000f 0101648f 497c8423 49788421 0001cb88 00000010 00016b87 0000000f 0101658f 497ca523 4978a521 0001cb88 00000014 00016b87 0000000f 0101668f 497cc623 4978c621 0001cb88 00000018 00016b87 0000000f 0101678f 497ce723 4978e721 0001cb88 0000001c 0101688f 497d0823 49790821 00000073 09036161 00000000 09036261 00000084 09036361 00000108 09036461 0000018c 09036561 00000210 09036661 00000294 09036761 00000318 09036861 0000039c 00000050 00010a86 00000008 0101598f 0000ea86 00000008 00014a85 00000880 01015a8f 01800113 42000000 01800140 09034960 00000000 09034a60 00000001 09034b60 00000002 09034c60 00000003 09034d60 00000004 09034e60 00000005 09034f60 00000006 09035060 00000007 09033160 00000000 09033260 00000001 09033360 00000002 09033460 00000003 09033560 00000004 09033660 00000005 09033760 00000006 09033860 00000007 48262122 00000001 482a2222 00000002 482e2322 00000003 48322422 00000004 48362522 00000005 483a2622 00000006 483e2722 00000007 48422822 00000008 48264922 00000009 482a4a22 0000000a 482e4b22 0000000b 48324c22 0000000c 48364d22 0000000d 483a4e22 0000000e 483e4f22 0000000f 48425022 00000010 48267122 00000011 482a7222 00000012 482e7322 00000013 48327422 00000014 48367522 00000015 483a7622 00000016 483e7722 00000017 48427822 00000018 48269922 00000019 482a9a22 0000001a 482e9b22 0000001b 48329c22 0000001c 48369d22 0000001d 483a9e22 0000001e 483e9f22 0000001f 48c28022 00000020 48a6a122 00000021 48aaa222 00000022 48aea322 00000023 48b2a422 00000024 48b6a522 00000025 48baa622 00000026 48bea722 00000027 48c2a822 00000028 48a6c922 00000029 48aaca22 0000002a 48aecb22 0000002b 48b2cc22 0000002c 48b6cd22 0000002d 48bace22 0000002e 48becf22 0000002f 48c2d022 00000030 48a6f122 00000031 48aaf222 00000032 48aef322 00000033 48b2f422 00000034 48b6f522 00000035 48baf622 00000036 48bef722 00000037 48c2f822 00000038 48a71922 00000039 48ab1a22 0000003a 48af1b22 0000003b 48b31c22 0000003c 48b71d22 0000003d 48bb1e22 0000003e 48bf1f22 0000003f 49430022 00000040 0903392a 42880000 09035a2a 43040000 00000041 00000050 00012985 00000020 00000041 00010a86 00000008 00194a83 0000eb86 00000008 00156b83 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001818e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001828e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001838e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001848e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001858e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001868e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001878e 00000000 00000073 00014c85 00000000 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001888e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001898e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018a8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018b8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018c8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018d8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018e8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00018f8e 00000000 00000073 00014c85 00000001 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001908e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001918e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001928e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001938e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001948e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001958e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001968e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001978e 00000000 00000073 00014c85 00000002 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001988e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0001998e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019a8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019b8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019c8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019d8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019e8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00019f8e 00000000 00000073 00014c85 00000003 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081808e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081818e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081828e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081838e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081848e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081858e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081868e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081878e 00000000 00000073 00014c85 00000004 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081888e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081898e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818a8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818b8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818c8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818d8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818e8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00818f8e 00000000 00000073 00014c85 00000005 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081908e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081918e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081928e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081938e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081948e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081958e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081968e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081978e 00000000 00000073 00014c85 00000006 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081988e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000000 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0081998e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000001 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819a8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000002 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819b8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000003 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819c8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000004 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819d8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000005 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819e8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000006 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 00819f8e 00000000 00000073 00014c85 00000007 01019c8f 01001e13 41800000 497b9d70 00016d85 00000007 0101bc8f 01001e13 45800000 497b9b70 496fbd21 01001d72 00018c86 00001000 00358c83 0101808e 00000000 00000073 00000001
endkernel
kernel s.silu_mul_2 in=2 out=1 threads=176128 tg=64 tgbytes=16 expect=in0:176128,in1:176128,out0:176128 rows=1:11008 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 2 1
mul 1 1 2
uloadf32 3 0 @in1
mul 1 1 3
ustoref32 1 0 @out0
endkernel
kernel s.rmsnorm_3 in=2 out=1 threads=256 tg=256 tgbytes=1024 expect=in0:4096,in1:4096,out0:4096 rows=1:256 tag=aten::rms_norm
rtou 0 0
loadconstf32 4 0
loadconstf32 7 4096
utgid 1
ulid 2
loadlid 1
umulk 3 1 4096
uaddk 4 2 0
loadconstf32 8 16
forbegin 8
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
mul 2 2 2
add 4 4 2
ifend 
uaddk 4 4 256
forend 
tgstoref32 4 1 0
barriertg 
loadconstf32 8 128
cmplt 10 1 8
ifbegin 10
addconst 9 1 128
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 64
cmplt 10 1 8
ifbegin 10
addconst 9 1 64
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 32
cmplt 10 1 8
ifbegin 10
addconst 9 1 32
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 16
cmplt 10 1 8
ifbegin 10
addconst 9 1 16
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 8
cmplt 10 1 8
ifbegin 10
addconst 9 1 8
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 4
cmplt 10 1 8
ifbegin 10
addconst 9 1 4
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 2
cmplt 10 1 8
ifbegin 10
addconst 9 1 2
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 1
cmplt 10 1 8
ifbegin 10
addconst 9 1 1
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 9 0
tgloadf32 8 9 0
mulconst 8 8 0.000244140625
addconst 8 8 0.000001
rsqrt 8 8
uaddk 4 2 0
loadconstf32 10 16
forbegin 10
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
mul 2 2 8
uloadf32 3 4 @in1
mul 2 2 3
ustoref32 2 5 @out0
ifend 
uaddk 4 4 256
forend 
endkernel
kernel s.matmul_q4_simd in=2 out=1 threads=258048 tg=64 tgbytes=16 layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0 tag=aten::matmul
words 00000090 00000180 000002a1 000004a3 000003a4 00006686 00000004 00008586 00000008 00000882 00000000 00010886 00000004 00004786 00000008 00000113 00000000 00000213 00000000 00000313 00000000 00000413 00000000 00001613 45800000 00001813 46fc0000 00001913 3f800000 0000d78f 0062f470 00001472 00001a13 42000000 00001a40 0000f58f 005ab470 00001472 0000e58d 00000000 0000e68d 00000001 0000e78d 00000002 0000e88d 00000003 0000e98d 00000004 0000ea8d 00000005 0000eb8d 00000006 0000ec8d 00000007 0018ae20 0020ef20 003dce20 00292f20 00317020 0041ef20 003dcd20 0000f085 00000000 00020f88 00000007 00020e88 00000001 0000cd85 00000000 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00fc0000 0001938c 010bc000 00367321 004e1023 00424122 00000001 0000cd85 00000001 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00fc0000 0001938c 010bc000 00367321 004e1023 00424222 00000002 0000cd85 00000002 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00fc0000 0001938c 010bc000 00367321 004e1023 00424322 00000003 0000cd85 00000003 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00fc0000 0001938c 010bc000 00367321 004e1023 00424422 00000004 00000073 0014e783 00000041 0000558f 0066b470 000030a0 00001472 0000cd85 00000000 0001b08e 00000000 00000073 000050a0 00001472 0000cd85 00000001 0001b08e 00000000 00000073 000070a0 00001472 0000cd85 00000002 0001b08e 00000000 00000073 000090a0 00001472 0000cd85 00000003 0001b08e 00000000 00000073 00000073 00000001
endkernel
kernel s.matmul_q4_simd_2 in=2 out=1 threads=32768 tg=64 tgbytes=16 layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0 tag=aten::matmul
words 00000090 00000180 000002a1 000004a3 000003a4 00006686 00000004 00008586 00000008 00000882 00000000 00010886 00000004 00004786 00000008 00000113 00000000 00000213 00000000 00000313 00000000 00000413 00000000 00001613 45800000 00001813 45800000 00001913 3f800000 0000d78f 0062f470 00001472 00001a13 42000000 00001a40 0000f58f 005ab470 00001472 0000e58d 00000000 0000e68d 00000001 0000e78d 00000002 0000e88d 00000003 0000e98d 00000004 0000ea8d 00000005 0000eb8d 00000006 0000ec8d 00000007 0018ae20 0020ef20 003dce20 00292f20 00317020 0041ef20 003dcd20 0000f085 00000000 00020f88 00000007 00020e88 00000001 0000cd85 00000000 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00200000 0001938c 00220000 00367321 004e1023 00424122 00000001 0000cd85 00000001 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00200000 0001938c 00220000 00367321 004e1023 00424222 00000002 0000cd85 00000002 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00200000 0001938c 00220000 00367321 004e1023 00424322 00000003 0000cd85 00000003 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00200000 0001938c 00220000 00367321 004e1023 00424422 00000004 00000073 0014e783 00000041 0000558f 0066b470 000030a0 00001472 0000cd85 00000000 0001b08e 00000000 00000073 000050a0 00001472 0000cd85 00000001 0001b08e 00000000 00000073 000070a0 00001472 0000cd85 00000002 0001b08e 00000000 00000073 000090a0 00001472 0000cd85 00000003 0001b08e 00000000 00000073 00000073 00000001
endkernel
kernel s.attention in=4 out=1 threads=4096 tg=128 tgbytes=2560 expect=in0:4096,in1:2097152,in2:2097152,in3:1,out0:4096 tag=aten::scaled_dot_product_attention
rtou 0 0
loadconstf32 13 0
loadconstf32 14 1
loadconstf32 15 -300000000000000012135895401846682943488.0
uconst 11 0
utgid 1
ulid 3
loadlid 1
uloadf32 2 11 @in3
addconst 2 2 1
ushrk 2 1 0
umulk 4 1 128
umulk 5 2 65536
rtou 9 2
uaddk 9 9 127
ushrk 9 9 7
utor 10 9
mulconst 4 15 1
loadconstf32 5 0
uaddk 6 3 0
loadconstf32 3 0
loadconstf32 16 4
forbegin 16
utor 3 6
cmplt 9 3 2
ifbegin 9
umulk 7 6 128
uadd 7 7 5
uaddk 8 4 0
loadconstf32 6 0
loadconstf32 17 128
forbegin 17
uloadf32 7 8 @in0
uloadf32 8 7 @in1
mul 7 7 8
add 6 6 7
uaddk 8 8 1
uaddk 7 7 1
forend 
tgstoref32 6 3 0
cmplt 9 4 6
ifbegin 9
mulconst 4 6 1
ifend 
ifend 
uaddk 6 6 128
forend 
tgstoref32 4 1 512
barriertg 
loadconstf32 16 64
cmplt 18 1 16
ifbegin 18
addconst 17 1 64
tgloadf32 19 17 512
tgloadf32 20 1 512
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 32
cmplt 18 1 16
ifbegin 18
addconst 17 1 32
tgloadf32 19 17 512
tgloadf32 20 1 512
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 16
cmplt 18 1 16
ifbegin 18
addconst 17 1 16
tgloadf32 19 17 512
tgloadf32 20 1 512
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 8
cmplt 18 1 16
ifbegin 18
addconst 17 1 8
tgloadf32 19 17 512
tgloadf32 20 1 512
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 4
cmplt 18 1 16
ifbegin 18
addconst 17 1 4
tgloadf32 19 17 512
tgloadf32 20 1 512
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 2
cmplt 18 1 16
ifbegin 18
addconst 17 1 2
tgloadf32 19 17 512
tgloadf32 20 1 512
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 1
cmplt 18 1 16
ifbegin 18
addconst 17 1 1
tgloadf32 19 17 512
tgloadf32 20 1 512
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 512
ifend 
barriertg 
tgloadf32 11 13 512
barriertg 
uaddk 6 3 0
loadconstf32 16 4
forbegin 16
utor 3 6
cmplt 9 3 2
ifbegin 9
tgloadf32 6 3 0
sub 6 6 11
exp 6 6
tgstoref32 6 3 0
add 5 5 6
ifend 
uaddk 6 6 128
forend 
tgstoref32 5 1 512
barriertg 
loadconstf32 16 64
cmplt 18 1 16
ifbegin 18
addconst 17 1 64
tgloadf32 19 17 512
tgloadf32 20 1 512
add 20 20 19
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 32
cmplt 18 1 16
ifbegin 18
addconst 17 1 32
tgloadf32 19 17 512
tgloadf32 20 1 512
add 20 20 19
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 16
cmplt 18 1 16
ifbegin 18
addconst 17 1 16
tgloadf32 19 17 512
tgloadf32 20 1 512
add 20 20 19
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 8
cmplt 18 1 16
ifbegin 18
addconst 17 1 8
tgloadf32 19 17 512
tgloadf32 20 1 512
add 20 20 19
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 4
cmplt 18 1 16
ifbegin 18
addconst 17 1 4
tgloadf32 19 17 512
tgloadf32 20 1 512
add 20 20 19
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 2
cmplt 18 1 16
ifbegin 18
addconst 17 1 2
tgloadf32 19 17 512
tgloadf32 20 1 512
add 20 20 19
tgstoref32 20 1 512
ifend 
barriertg 
loadconstf32 16 1
cmplt 18 1 16
ifbegin 18
addconst 17 1 1
tgloadf32 19 17 512
tgloadf32 20 1 512
add 20 20 19
tgstoref32 20 1 512
ifend 
barriertg 
tgloadf32 12 13 512
barriertg 
loadconstf32 16 128
cmplt 9 1 16
ifbegin 9
loadconstf32 6 0
loadconstf32 3 0
uaddk 7 5 0
uadd 7 7 3
forbegin 2
tgloadf32 7 3 0
uloadf32 8 7 @in2
fma 6 7 8 6
addconst 3 3 1
uaddk 7 7 128
forend 
div 6 6 12
uadd 10 4 3
ustoref32 6 10 @out0
ifend 
endkernel
kernel s.add_rmsnorm in=3 out=2 threads=256 tg=256 tgbytes=1024 expect=in0:4096,in1:4096,in2:4096,out0:4096,out1:4096 tag=aten::add
rtou 0 0
loadconstf32 4 0
loadconstf32 7 4096
utgid 1
ulid 2
loadlid 1
umulk 3 1 4096
uaddk 4 2 0
loadconstf32 8 16
forbegin 8
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
uloadf32 3 5 @in1
add 2 2 3
ustoref32 2 5 @out0
mul 2 2 2
add 4 4 2
ifend 
uaddk 4 4 256
forend 
tgstoref32 4 1 0
barriertg 
loadconstf32 8 128
cmplt 10 1 8
ifbegin 10
addconst 9 1 128
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 64
cmplt 10 1 8
ifbegin 10
addconst 9 1 64
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 32
cmplt 10 1 8
ifbegin 10
addconst 9 1 32
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 16
cmplt 10 1 8
ifbegin 10
addconst 9 1 16
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 8
cmplt 10 1 8
ifbegin 10
addconst 9 1 8
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 4
cmplt 10 1 8
ifbegin 10
addconst 9 1 4
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 2
cmplt 10 1 8
ifbegin 10
addconst 9 1 2
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 1
cmplt 10 1 8
ifbegin 10
addconst 9 1 1
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 9 0
tgloadf32 8 9 0
mulconst 8 8 0.000244140625
addconst 8 8 0.000001
rsqrt 8 8
uaddk 4 2 0
loadconstf32 10 16
forbegin 10
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @out0
mul 2 2 8
uloadf32 3 4 @in2
mul 2 2 3
ustoref32 2 5 @out1
ifend 
uaddk 4 4 256
forend 
endkernel
kernel s.matmul_q4_simd_3 in=2 out=1 threads=88064 tg=64 tgbytes=16 layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0 tag=aten::matmul
words 00000090 00000180 000002a1 000004a3 000003a4 00006686 00000004 00008586 00000008 00000882 00000000 00010886 00000004 00004786 00000008 00000113 00000000 00000213 00000000 00000313 00000000 00000413 00000000 00001613 45800000 00001813 462c0000 00001913 3f800000 0000d78f 0062f470 00001472 00001a13 42000000 00001a40 0000f58f 005ab470 00001472 0000e58d 00000000 0000e68d 00000001 0000e78d 00000002 0000e88d 00000003 0000e98d 00000004 0000ea8d 00000005 0000eb8d 00000006 0000ec8d 00000007 0018ae20 0020ef20 003dce20 00292f20 00317020 0041ef20 003dcd20 0000f085 00000000 00020f88 00000007 00020e88 00000001 0000cd85 00000000 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00560000 0001938c 005b6000 00367321 004e1023 00424122 00000001 0000cd85 00000001 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00560000 0001938c 005b6000 00367321 004e1023 00424222 00000002 0000cd85 00000002 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00560000 0001938c 005b6000 00367321 004e1023 00424322 00000003 0000cd85 00000003 0001a986 00000800 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000020 003d8c83 0001928c 00560000 0001938c 005b6000 00367321 004e1023 00424422 00000004 00000073 0014e783 00000041 0000558f 0066b470 000030a0 00001472 0000cd85 00000000 0001b08e 00000000 00000073 000050a0 00001472 0000cd85 00000001 0001b08e 00000000 00000073 000070a0 00001472 0000cd85 00000002 0001b08e 00000000 00000073 000090a0 00001472 0000cd85 00000003 0001b08e 00000000 00000073 00000073 00000001
endkernel
kernel s.silu_mul_3 in=2 out=1 threads=11008 tg=64 tgbytes=16 expect=in0:11008,in1:11008,out0:11008 rows=1:11008 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 2 1
mul 1 1 2
uloadf32 3 0 @in1
mul 1 1 3
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_simd_4 in=2 out=1 threads=32768 tg=64 tgbytes=16 layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0 tag=aten::matmul
words 00000090 00000180 000002a1 000004a3 000003a4 00006686 00000004 00008586 00000008 00000882 00000000 00010886 00000004 00004786 00000008 00000113 00000000 00000213 00000000 00000313 00000000 00000413 00000000 00001613 462c0000 00001813 45800000 00001913 3f800000 0000d78f 0062f470 00001472 00001a13 43000000 00001a40 0000f58f 005ab470 00001472 0000e58d 00000000 0000e68d 00000001 0000e78d 00000002 0000e88d 00000003 0000e98d 00000004 0000ea8d 00000005 0000eb8d 00000006 0000ec8d 00000007 0018ae20 0020ef20 003dce20 00292f20 00317020 0041ef20 003dcd20 0000f085 00000000 00020f88 00000007 00020e88 00000001 0000cd85 00000000 0001a986 00001580 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000056 003d8c83 0001928c 00560000 0001938c 005b6000 00367321 004e1023 00424122 00000001 0000cd85 00000001 0001a986 00001580 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000056 003d8c83 0001928c 00560000 0001938c 005b6000 00367321 004e1023 00424222 00000002 0000cd85 00000002 0001a986 00001580 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000056 003d8c83 0001928c 00560000 0001938c 005b6000 00367321 004e1023 00424322 00000003 0000cd85 00000003 0001a986 00001580 00212983 00392983 00012a9d 00000000 00014b88 0000001c 0001718f 00459021 00014b88 00000018 00016b87 0000000f 0001718f 00457022 00000010 00014b88 00000014 00016b87 0000000f 0001718f 00455022 00000010 00014b88 00000010 00016b87 0000000f 0001718f 00453022 00000010 00014b88 0000000c 00016b87 0000000f 0001718f 00451022 00000010 00014b88 00000008 00016b87 0000000f 0001718f 0044f022 00000010 00014b88 00000004 00016b87 0000000f 0001718f 0044d022 00000010 00014b88 00000000 00016b87 0000000f 0001718f 0044b022 00000010 0001ac86 00000056 003d8c83 0001928c 00560000 0001938c 005b6000 00367321 004e1023 00424422 00000004 00000073 0014e783 00000041 0000558f 0066b470 000030a0 00001472 0000cd85 00000000 0001b08e 00000000 00000073 000050a0 00001472 0000cd85 00000001 0001b08e 00000000 00000073 000070a0 00001472 0000cd85 00000002 0001b08e 00000000 00000073 000090a0 00001472 0000cd85 00000003 0001b08e 00000000 00000073 00000073 00000001
endkernel
kernel s.side in=4 out=2 threads=4096 tg=64 tgbytes=16 expect=in0:4096,in1:1,in2:4096,in3:1,out0:4096,out1:4096
rtou 0 0
utor 16 0
loadconstf32 17 2048
cmplt 17 16 17
loadconstf32 18 0
cmplt 18 16 18
sub 17 17 18
ifbegin 17
uconst 1 0
uandk 2 0 63
ushrk 3 0 6
umulk 5 3 128
uadd 5 5 2
uloadf32 1 1 @in1
mulconst 1 1 0.25
utor 9 2
mulconst 9 9 -0.17988946039016
exp 2 9
mul 3 1 2
mulconst 10 3 0.1591549430919
addconst 10 10 0.5
rtou 6 10
utor 10 6
mulconst 12 10 6.28125
sub 11 3 12
mulconst 12 10 0.0019353071795864769
sub 11 11 12
mul 13 11 11
mulconst 14 11 1
mulconst 4 11 1
loadconstf32 15 1
loadconstf32 5 1
mul 14 14 13
mulconst 14 14 -0.16666666666667
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.5
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.05
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.08333333333333
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.02380952380952
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.03333333333333
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.01388888888889
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.01785714285714
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00909090909091
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.01111111111111
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00641025641026
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00757575757576
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.0047619047619
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00549450549451
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00367647058824
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00416666666667
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00292397660819
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00326797385621
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00238095238095
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00263157894737
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00197628458498
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.0021645021645
add 5 5 15
uloadf32 6 5 @in0
uloadf32 7 5 @in0+64
mul 8 6 5
mul 3 7 4
sub 8 8 3
mulconst 8 8 0.08838834613562
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
mulconst 8 8 0.08838834613562
ustoref32 8 5 @out0+64
loadconstf32 16 -1
ifend
loadconstf32 17 4096
cmplt 17 16 17
loadconstf32 18 2048
cmplt 18 16 18
sub 17 17 18
ifbegin 17
uaddk 0 0 4294965248
uconst 1 0
uandk 2 0 63
ushrk 3 0 6
umulk 5 3 128
uadd 5 5 2
uloadf32 1 1 @in3
mulconst 1 1 0.25
utor 9 2
mulconst 9 9 -0.17988946039016
exp 2 9
mul 3 1 2
mulconst 10 3 0.1591549430919
addconst 10 10 0.5
rtou 6 10
utor 10 6
mulconst 12 10 6.28125
sub 11 3 12
mulconst 12 10 0.0019353071795864769
sub 11 11 12
mul 13 11 11
mulconst 14 11 1
mulconst 4 11 1
loadconstf32 15 1
loadconstf32 5 1
mul 14 14 13
mulconst 14 14 -0.16666666666667
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.5
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.05
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.08333333333333
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.02380952380952
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.03333333333333
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.01388888888889
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.01785714285714
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00909090909091
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.01111111111111
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00641025641026
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00757575757576
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.0047619047619
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00549450549451
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00367647058824
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00416666666667
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00292397660819
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00326797385621
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00238095238095
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00263157894737
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00197628458498
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.0021645021645
add 5 5 15
uloadf32 6 5 @in2
uloadf32 7 5 @in2+64
mul 8 6 5
mul 3 7 4
sub 8 8 3
ustoref32 8 5 @out1
mul 8 7 5
mul 3 6 4
add 8 8 3
ustoref32 8 5 @out1+64
loadconstf32 16 -1
ifend
endkernel
kernel s.side_2 in=6 out=2 threads=8192 tg=64 tgbytes=16 expect=in0:4096,in1:1,in2:2097152,in3:4096,in4:1,in5:2097152,out0:2097152,out1:2097152
rtou 0 0
utor 9 0
loadconstf32 10 4096
cmplt 10 9 10
loadconstf32 11 0
cmplt 11 9 11
sub 10 10 11
ifbegin 10
uconst 1 0
uloadf32 1 1 @in1
rtou 2 1
utor 3 0
addconst 3 3 0.5
mulconst 3 3 0.0078125
rtou 3 3
utor 4 0
utor 5 3
mulconst 5 5 128
sub 4 4 5
rtou 6 4
utor 6 3
addconst 6 6 0.5
mulconst 6 6 0.03125
rtou 4 6
utor 7 3
utor 8 4
mulconst 8 8 32
sub 7 7 8
rtou 5 7
uadd 7 2 4
umulk 5 5 512
uadd 7 7 5
umulk 7 7 128
uadd 7 7 6
uloadf32 2 0 @in0
ustoref32 2 7 @in2
loadconstf32 9 -1
ifend
loadconstf32 10 8192
cmplt 10 9 10
loadconstf32 11 4096
cmplt 11 9 11
sub 10 10 11
ifbegin 10
uaddk 0 0 4294963200
uconst 1 0
uloadf32 1 1 @in4
rtou 2 1
utor 3 0
addconst 3 3 0.5
mulconst 3 3 0.0078125
rtou 3 3
utor 4 0
utor 5 3
mulconst 5 5 128
sub 4 4 5
rtou 6 4
utor 6 3
addconst 6 6 0.5
mulconst 6 6 0.03125
rtou 4 6
utor 7 3
utor 8 4
mulconst 8 8 32
sub 7 7 8
rtou 5 7
uadd 7 2 4
umulk 5 5 512
uadd 7 7 5
umulk 7 7 128
uadd 7 7 6
uloadf32 2 0 @in3
ustoref32 2 7 @in5
loadconstf32 9 -1
ifend
endkernel
phase chunk1 rows=64 partial
node embedding_p1 ext fir.embedding token model_embed_tokens_weight_emb -> t1038_embedding
node rmsnorm_0 micro s.rmsnorm t1038_embedding model_layers_0_input_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_0 micro s.matmul_q4_tiled_fma t1042_y model_layers_0_self_attn_q_proj_weight_qr -> t1044_c
node matmul_q4_tiled_fma_1 micro s.matmul_q4_tiled_fma t1042_y model_layers_0_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_2 micro s.matmul_q4_tiled_fma t1042_y model_layers_0_self_attn_v_proj_weight_qr -> t1048_c
node rope_0 micro s.rope t1044_c position -> t1044_c
node rope_1 micro s.rope t1046_c position -> t1046_c
node kv_scatter_0 micro s.kv_scatter t1046_c position s740 -> s740
node kv_scatter_1 micro s.kv_scatter t1048_c position s741 -> s741
node flash_prefill_0 micro s.flash_prefill t1044_c s740 s741 position -> t1048_c
node matmul_q4_tiled_fma_3 micro s.matmul_q4_tiled_fma t1048_c model_layers_0_self_attn_o_proj_weight_qr -> t1044_c
node vec_add_0 micro s.vec_add t1038_embedding t1044_c -> t1048_c
node rmsnorm_1 micro s.rmsnorm t1048_c model_layers_0_post_attention_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_4 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_0_mlp_gate_proj_weight_qr -> t1062_c
node matmul_q4_tiled_fma_5 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_0_mlp_up_proj_weight_qr -> t1064_c
node act_sigmoid_0 micro s.act_sigmoid t1062_c -> t1066_y
node vec_mul_0 micro s.vec_mul t1062_c t1066_y -> t1068_c
node vec_mul_1 micro s.vec_mul t1068_c t1064_c -> t1066_y
node matmul_q4_tiled_fma_6 micro s.matmul_q4_tiled_fma_3 t1066_y model_layers_0_mlp_down_proj_weight_qr -> t1044_c
node vec_add_1 micro s.vec_add t1048_c t1044_c -> t1038_embedding
node rmsnorm_2 micro s.rmsnorm t1038_embedding model_layers_1_input_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_7 micro s.matmul_q4_tiled_fma t1044_c model_layers_1_self_attn_q_proj_weight_qr -> t1048_c
node matmul_q4_tiled_fma_8 micro s.matmul_q4_tiled_fma t1044_c model_layers_1_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_9 micro s.matmul_q4_tiled_fma t1044_c model_layers_1_self_attn_v_proj_weight_qr -> t1042_y
node rope_2 micro s.rope t1048_c position -> t1048_c
node rope_3 micro s.rope t1046_c position -> t1046_c
node kv_scatter_2 micro s.kv_scatter t1046_c position s742 -> s742
node kv_scatter_3 micro s.kv_scatter t1042_y position s743 -> s743
node flash_prefill_1 micro s.flash_prefill t1048_c s742 s743 position -> t1042_y
node matmul_q4_tiled_fma_10 micro s.matmul_q4_tiled_fma t1042_y model_layers_1_self_attn_o_proj_weight_qr -> t1048_c
node vec_add_2 micro s.vec_add t1038_embedding t1048_c -> t1042_y
node rmsnorm_3 micro s.rmsnorm t1042_y model_layers_1_post_attention_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_11 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_1_mlp_gate_proj_weight_qr -> t1066_y
node matmul_q4_tiled_fma_12 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_1_mlp_up_proj_weight_qr -> t1068_c
node act_sigmoid_1 micro s.act_sigmoid t1066_y -> t1064_c
node vec_mul_2 micro s.vec_mul t1066_y t1064_c -> t1062_c
node vec_mul_3 micro s.vec_mul t1062_c t1068_c -> t1064_c
node matmul_q4_tiled_fma_13 micro s.matmul_q4_tiled_fma_3 t1064_c model_layers_1_mlp_down_proj_weight_qr -> t1048_c
node vec_add_3 micro s.vec_add t1042_y t1048_c -> t1038_embedding
node rmsnorm_4 micro s.rmsnorm t1038_embedding model_layers_2_input_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_14 micro s.matmul_q4_tiled_fma t1048_c model_layers_2_self_attn_q_proj_weight_qr -> t1042_y
node matmul_q4_tiled_fma_15 micro s.matmul_q4_tiled_fma t1048_c model_layers_2_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_16 micro s.matmul_q4_tiled_fma t1048_c model_layers_2_self_attn_v_proj_weight_qr -> t1044_c
node rope_4 micro s.rope t1042_y position -> t1042_y
node rope_5 micro s.rope t1046_c position -> t1046_c
node kv_scatter_4 micro s.kv_scatter t1046_c position s744 -> s744
node kv_scatter_5 micro s.kv_scatter t1044_c position s745 -> s745
node flash_prefill_2 micro s.flash_prefill t1042_y s744 s745 position -> t1044_c
node matmul_q4_tiled_fma_17 micro s.matmul_q4_tiled_fma t1044_c model_layers_2_self_attn_o_proj_weight_qr -> t1042_y
node vec_add_4 micro s.vec_add t1038_embedding t1042_y -> t1044_c
node rmsnorm_5 micro s.rmsnorm t1044_c model_layers_2_post_attention_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_18 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_2_mlp_gate_proj_weight_qr -> t1064_c
node matmul_q4_tiled_fma_19 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_2_mlp_up_proj_weight_qr -> t1062_c
node act_sigmoid_2 micro s.act_sigmoid t1064_c -> t1068_c
node vec_mul_4 micro s.vec_mul t1064_c t1068_c -> t1066_y
node vec_mul_5 micro s.vec_mul t1066_y t1062_c -> t1068_c
node matmul_q4_tiled_fma_20 micro s.matmul_q4_tiled_fma_3 t1068_c model_layers_2_mlp_down_proj_weight_qr -> t1042_y
node vec_add_5 micro s.vec_add t1044_c t1042_y -> t1038_embedding
node rmsnorm_6 micro s.rmsnorm t1038_embedding model_layers_3_input_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_21 micro s.matmul_q4_tiled_fma t1042_y model_layers_3_self_attn_q_proj_weight_qr -> t1044_c
node matmul_q4_tiled_fma_22 micro s.matmul_q4_tiled_fma t1042_y model_layers_3_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_23 micro s.matmul_q4_tiled_fma t1042_y model_layers_3_self_attn_v_proj_weight_qr -> t1048_c
node rope_6 micro s.rope t1044_c position -> t1044_c
node rope_7 micro s.rope t1046_c position -> t1046_c
node kv_scatter_6 micro s.kv_scatter t1046_c position s746 -> s746
node kv_scatter_7 micro s.kv_scatter t1048_c position s747 -> s747
node flash_prefill_3 micro s.flash_prefill t1044_c s746 s747 position -> t1048_c
node matmul_q4_tiled_fma_24 micro s.matmul_q4_tiled_fma t1048_c model_layers_3_self_attn_o_proj_weight_qr -> t1044_c
node vec_add_6 micro s.vec_add t1038_embedding t1044_c -> t1048_c
node rmsnorm_7 micro s.rmsnorm t1048_c model_layers_3_post_attention_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_25 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_3_mlp_gate_proj_weight_qr -> t1068_c
node matmul_q4_tiled_fma_26 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_3_mlp_up_proj_weight_qr -> t1066_y
node act_sigmoid_3 micro s.act_sigmoid t1068_c -> t1062_c
node vec_mul_6 micro s.vec_mul t1068_c t1062_c -> t1064_c
node vec_mul_7 micro s.vec_mul t1064_c t1066_y -> t1062_c
node matmul_q4_tiled_fma_27 micro s.matmul_q4_tiled_fma_3 t1062_c model_layers_3_mlp_down_proj_weight_qr -> t1044_c
node vec_add_7 micro s.vec_add t1048_c t1044_c -> t1038_embedding
node rmsnorm_8 micro s.rmsnorm t1038_embedding model_layers_4_input_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_28 micro s.matmul_q4_tiled_fma t1044_c model_layers_4_self_attn_q_proj_weight_qr -> t1048_c
node matmul_q4_tiled_fma_29 micro s.matmul_q4_tiled_fma t1044_c model_layers_4_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_30 micro s.matmul_q4_tiled_fma t1044_c model_layers_4_self_attn_v_proj_weight_qr -> t1042_y
node rope_8 micro s.rope t1048_c position -> t1048_c
node rope_9 micro s.rope t1046_c position -> t1046_c
node kv_scatter_8 micro s.kv_scatter t1046_c position s748 -> s748
node kv_scatter_9 micro s.kv_scatter t1042_y position s749 -> s749
node flash_prefill_4 micro s.flash_prefill t1048_c s748 s749 position -> t1042_y
node matmul_q4_tiled_fma_31 micro s.matmul_q4_tiled_fma t1042_y model_layers_4_self_attn_o_proj_weight_qr -> t1048_c
node vec_add_8 micro s.vec_add t1038_embedding t1048_c -> t1042_y
node rmsnorm_9 micro s.rmsnorm t1042_y model_layers_4_post_attention_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_32 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_4_mlp_gate_proj_weight_qr -> t1062_c
node matmul_q4_tiled_fma_33 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_4_mlp_up_proj_weight_qr -> t1064_c
node act_sigmoid_4 micro s.act_sigmoid t1062_c -> t1066_y
node vec_mul_8 micro s.vec_mul t1062_c t1066_y -> t1068_c
node vec_mul_9 micro s.vec_mul t1068_c t1064_c -> t1066_y
node matmul_q4_tiled_fma_34 micro s.matmul_q4_tiled_fma_3 t1066_y model_layers_4_mlp_down_proj_weight_qr -> t1048_c
node vec_add_9 micro s.vec_add t1042_y t1048_c -> t1038_embedding
node rmsnorm_10 micro s.rmsnorm t1038_embedding model_layers_5_input_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_35 micro s.matmul_q4_tiled_fma t1048_c model_layers_5_self_attn_q_proj_weight_qr -> t1042_y
node matmul_q4_tiled_fma_36 micro s.matmul_q4_tiled_fma t1048_c model_layers_5_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_37 micro s.matmul_q4_tiled_fma t1048_c model_layers_5_self_attn_v_proj_weight_qr -> t1044_c
node rope_10 micro s.rope t1042_y position -> t1042_y
node rope_11 micro s.rope t1046_c position -> t1046_c
node kv_scatter_10 micro s.kv_scatter t1046_c position s750 -> s750
node kv_scatter_11 micro s.kv_scatter t1044_c position s751 -> s751
node flash_prefill_5 micro s.flash_prefill t1042_y s750 s751 position -> t1044_c
node matmul_q4_tiled_fma_38 micro s.matmul_q4_tiled_fma t1044_c model_layers_5_self_attn_o_proj_weight_qr -> t1042_y
node vec_add_10 micro s.vec_add t1038_embedding t1042_y -> t1044_c
node rmsnorm_11 micro s.rmsnorm t1044_c model_layers_5_post_attention_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_39 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_5_mlp_gate_proj_weight_qr -> t1066_y
node matmul_q4_tiled_fma_40 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_5_mlp_up_proj_weight_qr -> t1068_c
node act_sigmoid_5 micro s.act_sigmoid t1066_y -> t1064_c
node vec_mul_10 micro s.vec_mul t1066_y t1064_c -> t1062_c
node vec_mul_11 micro s.vec_mul t1062_c t1068_c -> t1064_c
node matmul_q4_tiled_fma_41 micro s.matmul_q4_tiled_fma_3 t1064_c model_layers_5_mlp_down_proj_weight_qr -> t1042_y
node vec_add_11 micro s.vec_add t1044_c t1042_y -> t1038_embedding
node rmsnorm_12 micro s.rmsnorm t1038_embedding model_layers_6_input_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_42 micro s.matmul_q4_tiled_fma t1042_y model_layers_6_self_attn_q_proj_weight_qr -> t1044_c
node matmul_q4_tiled_fma_43 micro s.matmul_q4_tiled_fma t1042_y model_layers_6_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_44 micro s.matmul_q4_tiled_fma t1042_y model_layers_6_self_attn_v_proj_weight_qr -> t1048_c
node rope_12 micro s.rope t1044_c position -> t1044_c
node rope_13 micro s.rope t1046_c position -> t1046_c
node kv_scatter_12 micro s.kv_scatter t1046_c position s752 -> s752
node kv_scatter_13 micro s.kv_scatter t1048_c position s753 -> s753
node flash_prefill_6 micro s.flash_prefill t1044_c s752 s753 position -> t1048_c
node matmul_q4_tiled_fma_45 micro s.matmul_q4_tiled_fma t1048_c model_layers_6_self_attn_o_proj_weight_qr -> t1044_c
node vec_add_12 micro s.vec_add t1038_embedding t1044_c -> t1048_c
node rmsnorm_13 micro s.rmsnorm t1048_c model_layers_6_post_attention_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_46 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_6_mlp_gate_proj_weight_qr -> t1064_c
node matmul_q4_tiled_fma_47 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_6_mlp_up_proj_weight_qr -> t1062_c
node act_sigmoid_6 micro s.act_sigmoid t1064_c -> t1068_c
node vec_mul_12 micro s.vec_mul t1064_c t1068_c -> t1066_y
node vec_mul_13 micro s.vec_mul t1066_y t1062_c -> t1068_c
node matmul_q4_tiled_fma_48 micro s.matmul_q4_tiled_fma_3 t1068_c model_layers_6_mlp_down_proj_weight_qr -> t1044_c
node vec_add_13 micro s.vec_add t1048_c t1044_c -> t1038_embedding
node rmsnorm_14 micro s.rmsnorm t1038_embedding model_layers_7_input_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_49 micro s.matmul_q4_tiled_fma t1044_c model_layers_7_self_attn_q_proj_weight_qr -> t1048_c
node matmul_q4_tiled_fma_50 micro s.matmul_q4_tiled_fma t1044_c model_layers_7_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_51 micro s.matmul_q4_tiled_fma t1044_c model_layers_7_self_attn_v_proj_weight_qr -> t1042_y
node rope_14 micro s.rope t1048_c position -> t1048_c
node rope_15 micro s.rope t1046_c position -> t1046_c
node kv_scatter_14 micro s.kv_scatter t1046_c position s754 -> s754
node kv_scatter_15 micro s.kv_scatter t1042_y position s755 -> s755
node flash_prefill_7 micro s.flash_prefill t1048_c s754 s755 position -> t1042_y
node matmul_q4_tiled_fma_52 micro s.matmul_q4_tiled_fma t1042_y model_layers_7_self_attn_o_proj_weight_qr -> t1048_c
node vec_add_14 micro s.vec_add t1038_embedding t1048_c -> t1042_y
node rmsnorm_15 micro s.rmsnorm t1042_y model_layers_7_post_attention_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_53 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_7_mlp_gate_proj_weight_qr -> t1068_c
node matmul_q4_tiled_fma_54 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_7_mlp_up_proj_weight_qr -> t1066_y
node act_sigmoid_7 micro s.act_sigmoid t1068_c -> t1062_c
node vec_mul_14 micro s.vec_mul t1068_c t1062_c -> t1064_c
node vec_mul_15 micro s.vec_mul t1064_c t1066_y -> t1062_c
node matmul_q4_tiled_fma_55 micro s.matmul_q4_tiled_fma_3 t1062_c model_layers_7_mlp_down_proj_weight_qr -> t1048_c
node vec_add_15 micro s.vec_add t1042_y t1048_c -> t1038_embedding
node rmsnorm_16 micro s.rmsnorm t1038_embedding model_layers_8_input_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_56 micro s.matmul_q4_tiled_fma t1048_c model_layers_8_self_attn_q_proj_weight_qr -> t1042_y
node matmul_q4_tiled_fma_57 micro s.matmul_q4_tiled_fma t1048_c model_layers_8_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_58 micro s.matmul_q4_tiled_fma t1048_c model_layers_8_self_attn_v_proj_weight_qr -> t1044_c
node rope_16 micro s.rope t1042_y position -> t1042_y
node rope_17 micro s.rope t1046_c position -> t1046_c
node kv_scatter_16 micro s.kv_scatter t1046_c position s756 -> s756
node kv_scatter_17 micro s.kv_scatter t1044_c position s757 -> s757
node flash_prefill_8 micro s.flash_prefill t1042_y s756 s757 position -> t1044_c
node matmul_q4_tiled_fma_59 micro s.matmul_q4_tiled_fma t1044_c model_layers_8_self_attn_o_proj_weight_qr -> t1042_y
node vec_add_16 micro s.vec_add t1038_embedding t1042_y -> t1044_c
node rmsnorm_17 micro s.rmsnorm t1044_c model_layers_8_post_attention_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_60 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_8_mlp_gate_proj_weight_qr -> t1062_c
node matmul_q4_tiled_fma_61 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_8_mlp_up_proj_weight_qr -> t1064_c
node act_sigmoid_8 micro s.act_sigmoid t1062_c -> t1066_y
node vec_mul_16 micro s.vec_mul t1062_c t1066_y -> t1068_c
node vec_mul_17 micro s.vec_mul t1068_c t1064_c -> t1066_y
node matmul_q4_tiled_fma_62 micro s.matmul_q4_tiled_fma_3 t1066_y model_layers_8_mlp_down_proj_weight_qr -> t1042_y
node vec_add_17 micro s.vec_add t1044_c t1042_y -> t1038_embedding
node rmsnorm_18 micro s.rmsnorm t1038_embedding model_layers_9_input_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_63 micro s.matmul_q4_tiled_fma t1042_y model_layers_9_self_attn_q_proj_weight_qr -> t1044_c
node matmul_q4_tiled_fma_64 micro s.matmul_q4_tiled_fma t1042_y model_layers_9_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_65 micro s.matmul_q4_tiled_fma t1042_y model_layers_9_self_attn_v_proj_weight_qr -> t1048_c
node rope_18 micro s.rope t1044_c position -> t1044_c
node rope_19 micro s.rope t1046_c position -> t1046_c
node kv_scatter_18 micro s.kv_scatter t1046_c position s758 -> s758
node kv_scatter_19 micro s.kv_scatter t1048_c position s759 -> s759
node flash_prefill_9 micro s.flash_prefill t1044_c s758 s759 position -> t1048_c
node matmul_q4_tiled_fma_66 micro s.matmul_q4_tiled_fma t1048_c model_layers_9_self_attn_o_proj_weight_qr -> t1044_c
node vec_add_18 micro s.vec_add t1038_embedding t1044_c -> t1048_c
node rmsnorm_19 micro s.rmsnorm t1048_c model_layers_9_post_attention_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_67 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_9_mlp_gate_proj_weight_qr -> t1066_y
node matmul_q4_tiled_fma_68 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_9_mlp_up_proj_weight_qr -> t1068_c
node act_sigmoid_9 micro s.act_sigmoid t1066_y -> t1064_c
node vec_mul_18 micro s.vec_mul t1066_y t1064_c -> t1062_c
node vec_mul_19 micro s.vec_mul t1062_c t1068_c -> t1064_c
node matmul_q4_tiled_fma_69 micro s.matmul_q4_tiled_fma_3 t1064_c model_layers_9_mlp_down_proj_weight_qr -> t1044_c
node vec_add_19 micro s.vec_add t1048_c t1044_c -> t1038_embedding
node rmsnorm_20 micro s.rmsnorm t1038_embedding model_layers_10_input_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_70 micro s.matmul_q4_tiled_fma t1044_c model_layers_10_self_attn_q_proj_weight_qr -> t1048_c
node matmul_q4_tiled_fma_71 micro s.matmul_q4_tiled_fma t1044_c model_layers_10_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_72 micro s.matmul_q4_tiled_fma t1044_c model_layers_10_self_attn_v_proj_weight_qr -> t1042_y
node rope_20 micro s.rope t1048_c position -> t1048_c
node rope_21 micro s.rope t1046_c position -> t1046_c
node kv_scatter_20 micro s.kv_scatter t1046_c position s760 -> s760
node kv_scatter_21 micro s.kv_scatter t1042_y position s761 -> s761
node flash_prefill_10 micro s.flash_prefill t1048_c s760 s761 position -> t1042_y
node matmul_q4_tiled_fma_73 micro s.matmul_q4_tiled_fma t1042_y model_layers_10_self_attn_o_proj_weight_qr -> t1048_c
node vec_add_20 micro s.vec_add t1038_embedding t1048_c -> t1042_y
node rmsnorm_21 micro s.rmsnorm t1042_y model_layers_10_post_attention_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_74 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_10_mlp_gate_proj_weight_qr -> t1064_c
node matmul_q4_tiled_fma_75 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_10_mlp_up_proj_weight_qr -> t1062_c
node act_sigmoid_10 micro s.act_sigmoid t1064_c -> t1068_c
node vec_mul_20 micro s.vec_mul t1064_c t1068_c -> t1066_y
node vec_mul_21 micro s.vec_mul t1066_y t1062_c -> t1068_c
node matmul_q4_tiled_fma_76 micro s.matmul_q4_tiled_fma_3 t1068_c model_layers_10_mlp_down_proj_weight_qr -> t1048_c
node vec_add_21 micro s.vec_add t1042_y t1048_c -> t1038_embedding
node rmsnorm_22 micro s.rmsnorm t1038_embedding model_layers_11_input_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_77 micro s.matmul_q4_tiled_fma t1048_c model_layers_11_self_attn_q_proj_weight_qr -> t1042_y
node matmul_q4_tiled_fma_78 micro s.matmul_q4_tiled_fma t1048_c model_layers_11_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_79 micro s.matmul_q4_tiled_fma t1048_c model_layers_11_self_attn_v_proj_weight_qr -> t1044_c
node rope_22 micro s.rope t1042_y position -> t1042_y
node rope_23 micro s.rope t1046_c position -> t1046_c
node kv_scatter_22 micro s.kv_scatter t1046_c position s762 -> s762
node kv_scatter_23 micro s.kv_scatter t1044_c position s763 -> s763
node flash_prefill_11 micro s.flash_prefill t1042_y s762 s763 position -> t1044_c
node matmul_q4_tiled_fma_80 micro s.matmul_q4_tiled_fma t1044_c model_layers_11_self_attn_o_proj_weight_qr -> t1042_y
node vec_add_22 micro s.vec_add t1038_embedding t1042_y -> t1044_c
node rmsnorm_23 micro s.rmsnorm t1044_c model_layers_11_post_attention_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_81 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_11_mlp_gate_proj_weight_qr -> t1068_c
node matmul_q4_tiled_fma_82 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_11_mlp_up_proj_weight_qr -> t1066_y
node act_sigmoid_11 micro s.act_sigmoid t1068_c -> t1062_c
node vec_mul_22 micro s.vec_mul t1068_c t1062_c -> t1064_c
node vec_mul_23 micro s.vec_mul t1064_c t1066_y -> t1062_c
node matmul_q4_tiled_fma_83 micro s.matmul_q4_tiled_fma_3 t1062_c model_layers_11_mlp_down_proj_weight_qr -> t1042_y
node vec_add_23 micro s.vec_add t1044_c t1042_y -> t1038_embedding
node rmsnorm_24 micro s.rmsnorm t1038_embedding model_layers_12_input_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_84 micro s.matmul_q4_tiled_fma t1042_y model_layers_12_self_attn_q_proj_weight_qr -> t1044_c
node matmul_q4_tiled_fma_85 micro s.matmul_q4_tiled_fma t1042_y model_layers_12_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_86 micro s.matmul_q4_tiled_fma t1042_y model_layers_12_self_attn_v_proj_weight_qr -> t1048_c
node rope_24 micro s.rope t1044_c position -> t1044_c
node rope_25 micro s.rope t1046_c position -> t1046_c
node kv_scatter_24 micro s.kv_scatter t1046_c position s764 -> s764
node kv_scatter_25 micro s.kv_scatter t1048_c position s765 -> s765
node flash_prefill_12 micro s.flash_prefill t1044_c s764 s765 position -> t1048_c
node matmul_q4_tiled_fma_87 micro s.matmul_q4_tiled_fma t1048_c model_layers_12_self_attn_o_proj_weight_qr -> t1044_c
node vec_add_24 micro s.vec_add t1038_embedding t1044_c -> t1048_c
node rmsnorm_25 micro s.rmsnorm t1048_c model_layers_12_post_attention_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_88 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_12_mlp_gate_proj_weight_qr -> t1062_c
node matmul_q4_tiled_fma_89 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_12_mlp_up_proj_weight_qr -> t1064_c
node act_sigmoid_12 micro s.act_sigmoid t1062_c -> t1066_y
node vec_mul_24 micro s.vec_mul t1062_c t1066_y -> t1068_c
node vec_mul_25 micro s.vec_mul t1068_c t1064_c -> t1066_y
node matmul_q4_tiled_fma_90 micro s.matmul_q4_tiled_fma_3 t1066_y model_layers_12_mlp_down_proj_weight_qr -> t1044_c
node vec_add_25 micro s.vec_add t1048_c t1044_c -> t1038_embedding
node rmsnorm_26 micro s.rmsnorm t1038_embedding model_layers_13_input_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_91 micro s.matmul_q4_tiled_fma t1044_c model_layers_13_self_attn_q_proj_weight_qr -> t1048_c
node matmul_q4_tiled_fma_92 micro s.matmul_q4_tiled_fma t1044_c model_layers_13_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_93 micro s.matmul_q4_tiled_fma t1044_c model_layers_13_self_attn_v_proj_weight_qr -> t1042_y
node rope_26 micro s.rope t1048_c position -> t1048_c
node rope_27 micro s.rope t1046_c position -> t1046_c
node kv_scatter_26 micro s.kv_scatter t1046_c position s766 -> s766
node kv_scatter_27 micro s.kv_scatter t1042_y position s767 -> s767
node flash_prefill_13 micro s.flash_prefill t1048_c s766 s767 position -> t1042_y
node matmul_q4_tiled_fma_94 micro s.matmul_q4_tiled_fma t1042_y model_layers_13_self_attn_o_proj_weight_qr -> t1048_c
node vec_add_26 micro s.vec_add t1038_embedding t1048_c -> t1042_y
node rmsnorm_27 micro s.rmsnorm t1042_y model_layers_13_post_attention_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_95 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_13_mlp_gate_proj_weight_qr -> t1066_y
node matmul_q4_tiled_fma_96 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_13_mlp_up_proj_weight_qr -> t1068_c
node act_sigmoid_13 micro s.act_sigmoid t1066_y -> t1064_c
node vec_mul_26 micro s.vec_mul t1066_y t1064_c -> t1062_c
node vec_mul_27 micro s.vec_mul t1062_c t1068_c -> t1064_c
node matmul_q4_tiled_fma_97 micro s.matmul_q4_tiled_fma_3 t1064_c model_layers_13_mlp_down_proj_weight_qr -> t1048_c
node vec_add_27 micro s.vec_add t1042_y t1048_c -> t1038_embedding
node rmsnorm_28 micro s.rmsnorm t1038_embedding model_layers_14_input_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_98 micro s.matmul_q4_tiled_fma t1048_c model_layers_14_self_attn_q_proj_weight_qr -> t1042_y
node matmul_q4_tiled_fma_99 micro s.matmul_q4_tiled_fma t1048_c model_layers_14_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_100 micro s.matmul_q4_tiled_fma t1048_c model_layers_14_self_attn_v_proj_weight_qr -> t1044_c
node rope_28 micro s.rope t1042_y position -> t1042_y
node rope_29 micro s.rope t1046_c position -> t1046_c
node kv_scatter_28 micro s.kv_scatter t1046_c position s768 -> s768
node kv_scatter_29 micro s.kv_scatter t1044_c position s769 -> s769
node flash_prefill_14 micro s.flash_prefill t1042_y s768 s769 position -> t1044_c
node matmul_q4_tiled_fma_101 micro s.matmul_q4_tiled_fma t1044_c model_layers_14_self_attn_o_proj_weight_qr -> t1042_y
node vec_add_28 micro s.vec_add t1038_embedding t1042_y -> t1044_c
node rmsnorm_29 micro s.rmsnorm t1044_c model_layers_14_post_attention_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_102 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_14_mlp_gate_proj_weight_qr -> t1064_c
node matmul_q4_tiled_fma_103 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_14_mlp_up_proj_weight_qr -> t1062_c
node act_sigmoid_14 micro s.act_sigmoid t1064_c -> t1068_c
node vec_mul_28 micro s.vec_mul t1064_c t1068_c -> t1066_y
node vec_mul_29 micro s.vec_mul t1066_y t1062_c -> t1068_c
node matmul_q4_tiled_fma_104 micro s.matmul_q4_tiled_fma_3 t1068_c model_layers_14_mlp_down_proj_weight_qr -> t1042_y
node vec_add_29 micro s.vec_add t1044_c t1042_y -> t1038_embedding
node rmsnorm_30 micro s.rmsnorm t1038_embedding model_layers_15_input_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_105 micro s.matmul_q4_tiled_fma t1042_y model_layers_15_self_attn_q_proj_weight_qr -> t1044_c
node matmul_q4_tiled_fma_106 micro s.matmul_q4_tiled_fma t1042_y model_layers_15_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_107 micro s.matmul_q4_tiled_fma t1042_y model_layers_15_self_attn_v_proj_weight_qr -> t1048_c
node rope_30 micro s.rope t1044_c position -> t1044_c
node rope_31 micro s.rope t1046_c position -> t1046_c
node kv_scatter_30 micro s.kv_scatter t1046_c position s770 -> s770
node kv_scatter_31 micro s.kv_scatter t1048_c position s771 -> s771
node flash_prefill_15 micro s.flash_prefill t1044_c s770 s771 position -> t1048_c
node matmul_q4_tiled_fma_108 micro s.matmul_q4_tiled_fma t1048_c model_layers_15_self_attn_o_proj_weight_qr -> t1044_c
node vec_add_30 micro s.vec_add t1038_embedding t1044_c -> t1048_c
node rmsnorm_31 micro s.rmsnorm t1048_c model_layers_15_post_attention_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_109 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_15_mlp_gate_proj_weight_qr -> t1068_c
node matmul_q4_tiled_fma_110 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_15_mlp_up_proj_weight_qr -> t1066_y
node act_sigmoid_15 micro s.act_sigmoid t1068_c -> t1062_c
node vec_mul_30 micro s.vec_mul t1068_c t1062_c -> t1064_c
node vec_mul_31 micro s.vec_mul t1064_c t1066_y -> t1062_c
node matmul_q4_tiled_fma_111 micro s.matmul_q4_tiled_fma_3 t1062_c model_layers_15_mlp_down_proj_weight_qr -> t1044_c
node vec_add_31 micro s.vec_add t1048_c t1044_c -> t1038_embedding
node rmsnorm_32 micro s.rmsnorm t1038_embedding model_layers_16_input_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_112 micro s.matmul_q4_tiled_fma t1044_c model_layers_16_self_attn_q_proj_weight_qr -> t1048_c
node matmul_q4_tiled_fma_113 micro s.matmul_q4_tiled_fma t1044_c model_layers_16_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_114 micro s.matmul_q4_tiled_fma t1044_c model_layers_16_self_attn_v_proj_weight_qr -> t1042_y
node rope_32 micro s.rope t1048_c position -> t1048_c
node rope_33 micro s.rope t1046_c position -> t1046_c
node kv_scatter_32 micro s.kv_scatter t1046_c position s772 -> s772
node kv_scatter_33 micro s.kv_scatter t1042_y position s773 -> s773
node flash_prefill_16 micro s.flash_prefill t1048_c s772 s773 position -> t1042_y
node matmul_q4_tiled_fma_115 micro s.matmul_q4_tiled_fma t1042_y model_layers_16_self_attn_o_proj_weight_qr -> t1048_c
node vec_add_32 micro s.vec_add t1038_embedding t1048_c -> t1042_y
node rmsnorm_33 micro s.rmsnorm t1042_y model_layers_16_post_attention_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_116 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_16_mlp_gate_proj_weight_qr -> t1062_c
node matmul_q4_tiled_fma_117 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_16_mlp_up_proj_weight_qr -> t1064_c
node act_sigmoid_16 micro s.act_sigmoid t1062_c -> t1066_y
node vec_mul_32 micro s.vec_mul t1062_c t1066_y -> t1068_c
node vec_mul_33 micro s.vec_mul t1068_c t1064_c -> t1066_y
node matmul_q4_tiled_fma_118 micro s.matmul_q4_tiled_fma_3 t1066_y model_layers_16_mlp_down_proj_weight_qr -> t1048_c
node vec_add_33 micro s.vec_add t1042_y t1048_c -> t1038_embedding
node rmsnorm_34 micro s.rmsnorm t1038_embedding model_layers_17_input_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_119 micro s.matmul_q4_tiled_fma t1048_c model_layers_17_self_attn_q_proj_weight_qr -> t1042_y
node matmul_q4_tiled_fma_120 micro s.matmul_q4_tiled_fma t1048_c model_layers_17_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_121 micro s.matmul_q4_tiled_fma t1048_c model_layers_17_self_attn_v_proj_weight_qr -> t1044_c
node rope_34 micro s.rope t1042_y position -> t1042_y
node rope_35 micro s.rope t1046_c position -> t1046_c
node kv_scatter_34 micro s.kv_scatter t1046_c position s774 -> s774
node kv_scatter_35 micro s.kv_scatter t1044_c position s775 -> s775
node flash_prefill_17 micro s.flash_prefill t1042_y s774 s775 position -> t1044_c
node matmul_q4_tiled_fma_122 micro s.matmul_q4_tiled_fma t1044_c model_layers_17_self_attn_o_proj_weight_qr -> t1042_y
node vec_add_34 micro s.vec_add t1038_embedding t1042_y -> t1044_c
node rmsnorm_35 micro s.rmsnorm t1044_c model_layers_17_post_attention_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_123 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_17_mlp_gate_proj_weight_qr -> t1066_y
node matmul_q4_tiled_fma_124 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_17_mlp_up_proj_weight_qr -> t1068_c
node act_sigmoid_17 micro s.act_sigmoid t1066_y -> t1064_c
node vec_mul_34 micro s.vec_mul t1066_y t1064_c -> t1062_c
node vec_mul_35 micro s.vec_mul t1062_c t1068_c -> t1064_c
node matmul_q4_tiled_fma_125 micro s.matmul_q4_tiled_fma_3 t1064_c model_layers_17_mlp_down_proj_weight_qr -> t1042_y
node vec_add_35 micro s.vec_add t1044_c t1042_y -> t1038_embedding
node rmsnorm_36 micro s.rmsnorm t1038_embedding model_layers_18_input_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_126 micro s.matmul_q4_tiled_fma t1042_y model_layers_18_self_attn_q_proj_weight_qr -> t1044_c
node matmul_q4_tiled_fma_127 micro s.matmul_q4_tiled_fma t1042_y model_layers_18_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_128 micro s.matmul_q4_tiled_fma t1042_y model_layers_18_self_attn_v_proj_weight_qr -> t1048_c
node rope_36 micro s.rope t1044_c position -> t1044_c
node rope_37 micro s.rope t1046_c position -> t1046_c
node kv_scatter_36 micro s.kv_scatter t1046_c position s776 -> s776
node kv_scatter_37 micro s.kv_scatter t1048_c position s777 -> s777
node flash_prefill_18 micro s.flash_prefill t1044_c s776 s777 position -> t1048_c
node matmul_q4_tiled_fma_129 micro s.matmul_q4_tiled_fma t1048_c model_layers_18_self_attn_o_proj_weight_qr -> t1044_c
node vec_add_36 micro s.vec_add t1038_embedding t1044_c -> t1048_c
node rmsnorm_37 micro s.rmsnorm t1048_c model_layers_18_post_attention_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_130 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_18_mlp_gate_proj_weight_qr -> t1064_c
node matmul_q4_tiled_fma_131 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_18_mlp_up_proj_weight_qr -> t1062_c
node act_sigmoid_18 micro s.act_sigmoid t1064_c -> t1068_c
node vec_mul_36 micro s.vec_mul t1064_c t1068_c -> t1066_y
node vec_mul_37 micro s.vec_mul t1066_y t1062_c -> t1068_c
node matmul_q4_tiled_fma_132 micro s.matmul_q4_tiled_fma_3 t1068_c model_layers_18_mlp_down_proj_weight_qr -> t1044_c
node vec_add_37 micro s.vec_add t1048_c t1044_c -> t1038_embedding
node rmsnorm_38 micro s.rmsnorm t1038_embedding model_layers_19_input_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_133 micro s.matmul_q4_tiled_fma t1044_c model_layers_19_self_attn_q_proj_weight_qr -> t1048_c
node matmul_q4_tiled_fma_134 micro s.matmul_q4_tiled_fma t1044_c model_layers_19_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_135 micro s.matmul_q4_tiled_fma t1044_c model_layers_19_self_attn_v_proj_weight_qr -> t1042_y
node rope_38 micro s.rope t1048_c position -> t1048_c
node rope_39 micro s.rope t1046_c position -> t1046_c
node kv_scatter_38 micro s.kv_scatter t1046_c position s778 -> s778
node kv_scatter_39 micro s.kv_scatter t1042_y position s779 -> s779
node flash_prefill_19 micro s.flash_prefill t1048_c s778 s779 position -> t1042_y
node matmul_q4_tiled_fma_136 micro s.matmul_q4_tiled_fma t1042_y model_layers_19_self_attn_o_proj_weight_qr -> t1048_c
node vec_add_38 micro s.vec_add t1038_embedding t1048_c -> t1042_y
node rmsnorm_39 micro s.rmsnorm t1042_y model_layers_19_post_attention_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_137 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_19_mlp_gate_proj_weight_qr -> t1068_c
node matmul_q4_tiled_fma_138 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_19_mlp_up_proj_weight_qr -> t1066_y
node act_sigmoid_19 micro s.act_sigmoid t1068_c -> t1062_c
node vec_mul_38 micro s.vec_mul t1068_c t1062_c -> t1064_c
node vec_mul_39 micro s.vec_mul t1064_c t1066_y -> t1062_c
node matmul_q4_tiled_fma_139 micro s.matmul_q4_tiled_fma_3 t1062_c model_layers_19_mlp_down_proj_weight_qr -> t1048_c
node vec_add_39 micro s.vec_add t1042_y t1048_c -> t1038_embedding
node rmsnorm_40 micro s.rmsnorm t1038_embedding model_layers_20_input_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_140 micro s.matmul_q4_tiled_fma t1048_c model_layers_20_self_attn_q_proj_weight_qr -> t1042_y
node matmul_q4_tiled_fma_141 micro s.matmul_q4_tiled_fma t1048_c model_layers_20_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_142 micro s.matmul_q4_tiled_fma t1048_c model_layers_20_self_attn_v_proj_weight_qr -> t1044_c
node rope_40 micro s.rope t1042_y position -> t1042_y
node rope_41 micro s.rope t1046_c position -> t1046_c
node kv_scatter_40 micro s.kv_scatter t1046_c position s780 -> s780
node kv_scatter_41 micro s.kv_scatter t1044_c position s781 -> s781
node flash_prefill_20 micro s.flash_prefill t1042_y s780 s781 position -> t1044_c
node matmul_q4_tiled_fma_143 micro s.matmul_q4_tiled_fma t1044_c model_layers_20_self_attn_o_proj_weight_qr -> t1042_y
node vec_add_40 micro s.vec_add t1038_embedding t1042_y -> t1044_c
node rmsnorm_41 micro s.rmsnorm t1044_c model_layers_20_post_attention_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_144 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_20_mlp_gate_proj_weight_qr -> t1062_c
node matmul_q4_tiled_fma_145 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_20_mlp_up_proj_weight_qr -> t1064_c
node act_sigmoid_20 micro s.act_sigmoid t1062_c -> t1066_y
node vec_mul_40 micro s.vec_mul t1062_c t1066_y -> t1068_c
node vec_mul_41 micro s.vec_mul t1068_c t1064_c -> t1066_y
node matmul_q4_tiled_fma_146 micro s.matmul_q4_tiled_fma_3 t1066_y model_layers_20_mlp_down_proj_weight_qr -> t1042_y
node vec_add_41 micro s.vec_add t1044_c t1042_y -> t1038_embedding
node rmsnorm_42 micro s.rmsnorm t1038_embedding model_layers_21_input_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_147 micro s.matmul_q4_tiled_fma t1042_y model_layers_21_self_attn_q_proj_weight_qr -> t1044_c
node matmul_q4_tiled_fma_148 micro s.matmul_q4_tiled_fma t1042_y model_layers_21_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_149 micro s.matmul_q4_tiled_fma t1042_y model_layers_21_self_attn_v_proj_weight_qr -> t1048_c
node rope_42 micro s.rope t1044_c position -> t1044_c
node rope_43 micro s.rope t1046_c position -> t1046_c
node kv_scatter_42 micro s.kv_scatter t1046_c position s782 -> s782
node kv_scatter_43 micro s.kv_scatter t1048_c position s783 -> s783
node flash_prefill_21 micro s.flash_prefill t1044_c s782 s783 position -> t1048_c
node matmul_q4_tiled_fma_150 micro s.matmul_q4_tiled_fma t1048_c model_layers_21_self_attn_o_proj_weight_qr -> t1044_c
node vec_add_42 micro s.vec_add t1038_embedding t1044_c -> t1048_c
node rmsnorm_43 micro s.rmsnorm t1048_c model_layers_21_post_attention_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_151 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_21_mlp_gate_proj_weight_qr -> t1066_y
node matmul_q4_tiled_fma_152 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_21_mlp_up_proj_weight_qr -> t1068_c
node act_sigmoid_21 micro s.act_sigmoid t1066_y -> t1064_c
node vec_mul_42 micro s.vec_mul t1066_y t1064_c -> t1062_c
node vec_mul_43 micro s.vec_mul t1062_c t1068_c -> t1064_c
node matmul_q4_tiled_fma_153 micro s.matmul_q4_tiled_fma_3 t1064_c model_layers_21_mlp_down_proj_weight_qr -> t1044_c
node vec_add_43 micro s.vec_add t1048_c t1044_c -> t1038_embedding
node rmsnorm_44 micro s.rmsnorm t1038_embedding model_layers_22_input_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_154 micro s.matmul_q4_tiled_fma t1044_c model_layers_22_self_attn_q_proj_weight_qr -> t1048_c
node matmul_q4_tiled_fma_155 micro s.matmul_q4_tiled_fma t1044_c model_layers_22_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_156 micro s.matmul_q4_tiled_fma t1044_c model_layers_22_self_attn_v_proj_weight_qr -> t1042_y
node rope_44 micro s.rope t1048_c position -> t1048_c
node rope_45 micro s.rope t1046_c position -> t1046_c
node kv_scatter_44 micro s.kv_scatter t1046_c position s784 -> s784
node kv_scatter_45 micro s.kv_scatter t1042_y position s785 -> s785
node flash_prefill_22 micro s.flash_prefill t1048_c s784 s785 position -> t1042_y
node matmul_q4_tiled_fma_157 micro s.matmul_q4_tiled_fma t1042_y model_layers_22_self_attn_o_proj_weight_qr -> t1048_c
node vec_add_44 micro s.vec_add t1038_embedding t1048_c -> t1042_y
node rmsnorm_45 micro s.rmsnorm t1042_y model_layers_22_post_attention_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_158 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_22_mlp_gate_proj_weight_qr -> t1064_c
node matmul_q4_tiled_fma_159 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_22_mlp_up_proj_weight_qr -> t1062_c
node act_sigmoid_22 micro s.act_sigmoid t1064_c -> t1068_c
node vec_mul_44 micro s.vec_mul t1064_c t1068_c -> t1066_y
node vec_mul_45 micro s.vec_mul t1066_y t1062_c -> t1068_c
node matmul_q4_tiled_fma_160 micro s.matmul_q4_tiled_fma_3 t1068_c model_layers_22_mlp_down_proj_weight_qr -> t1048_c
node vec_add_45 micro s.vec_add t1042_y t1048_c -> t1038_embedding
node rmsnorm_46 micro s.rmsnorm t1038_embedding model_layers_23_input_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_161 micro s.matmul_q4_tiled_fma t1048_c model_layers_23_self_attn_q_proj_weight_qr -> t1042_y
node matmul_q4_tiled_fma_162 micro s.matmul_q4_tiled_fma t1048_c model_layers_23_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_163 micro s.matmul_q4_tiled_fma t1048_c model_layers_23_self_attn_v_proj_weight_qr -> t1044_c
node rope_46 micro s.rope t1042_y position -> t1042_y
node rope_47 micro s.rope t1046_c position -> t1046_c
node kv_scatter_46 micro s.kv_scatter t1046_c position s786 -> s786
node kv_scatter_47 micro s.kv_scatter t1044_c position s787 -> s787
node flash_prefill_23 micro s.flash_prefill t1042_y s786 s787 position -> t1044_c
node matmul_q4_tiled_fma_164 micro s.matmul_q4_tiled_fma t1044_c model_layers_23_self_attn_o_proj_weight_qr -> t1042_y
node vec_add_46 micro s.vec_add t1038_embedding t1042_y -> t1044_c
node rmsnorm_47 micro s.rmsnorm t1044_c model_layers_23_post_attention_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_165 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_23_mlp_gate_proj_weight_qr -> t1068_c
node matmul_q4_tiled_fma_166 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_23_mlp_up_proj_weight_qr -> t1066_y
node act_sigmoid_23 micro s.act_sigmoid t1068_c -> t1062_c
node vec_mul_46 micro s.vec_mul t1068_c t1062_c -> t1064_c
node vec_mul_47 micro s.vec_mul t1064_c t1066_y -> t1062_c
node matmul_q4_tiled_fma_167 micro s.matmul_q4_tiled_fma_3 t1062_c model_layers_23_mlp_down_proj_weight_qr -> t1042_y
node vec_add_47 micro s.vec_add t1044_c t1042_y -> t1038_embedding
node rmsnorm_48 micro s.rmsnorm t1038_embedding model_layers_24_input_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_168 micro s.matmul_q4_tiled_fma t1042_y model_layers_24_self_attn_q_proj_weight_qr -> t1044_c
node matmul_q4_tiled_fma_169 micro s.matmul_q4_tiled_fma t1042_y model_layers_24_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_170 micro s.matmul_q4_tiled_fma t1042_y model_layers_24_self_attn_v_proj_weight_qr -> t1048_c
node rope_48 micro s.rope t1044_c position -> t1044_c
node rope_49 micro s.rope t1046_c position -> t1046_c
node kv_scatter_48 micro s.kv_scatter t1046_c position s788 -> s788
node kv_scatter_49 micro s.kv_scatter t1048_c position s789 -> s789
node flash_prefill_24 micro s.flash_prefill t1044_c s788 s789 position -> t1048_c
node matmul_q4_tiled_fma_171 micro s.matmul_q4_tiled_fma t1048_c model_layers_24_self_attn_o_proj_weight_qr -> t1044_c
node vec_add_48 micro s.vec_add t1038_embedding t1044_c -> t1048_c
node rmsnorm_49 micro s.rmsnorm t1048_c model_layers_24_post_attention_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_172 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_24_mlp_gate_proj_weight_qr -> t1062_c
node matmul_q4_tiled_fma_173 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_24_mlp_up_proj_weight_qr -> t1064_c
node act_sigmoid_24 micro s.act_sigmoid t1062_c -> t1066_y
node vec_mul_48 micro s.vec_mul t1062_c t1066_y -> t1068_c
node vec_mul_49 micro s.vec_mul t1068_c t1064_c -> t1066_y
node matmul_q4_tiled_fma_174 micro s.matmul_q4_tiled_fma_3 t1066_y model_layers_24_mlp_down_proj_weight_qr -> t1044_c
node vec_add_49 micro s.vec_add t1048_c t1044_c -> t1038_embedding
node rmsnorm_50 micro s.rmsnorm t1038_embedding model_layers_25_input_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_175 micro s.matmul_q4_tiled_fma t1044_c model_layers_25_self_attn_q_proj_weight_qr -> t1048_c
node matmul_q4_tiled_fma_176 micro s.matmul_q4_tiled_fma t1044_c model_layers_25_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_177 micro s.matmul_q4_tiled_fma t1044_c model_layers_25_self_attn_v_proj_weight_qr -> t1042_y
node rope_50 micro s.rope t1048_c position -> t1048_c
node rope_51 micro s.rope t1046_c position -> t1046_c
node kv_scatter_50 micro s.kv_scatter t1046_c position s790 -> s790
node kv_scatter_51 micro s.kv_scatter t1042_y position s791 -> s791
node flash_prefill_25 micro s.flash_prefill t1048_c s790 s791 position -> t1042_y
node matmul_q4_tiled_fma_178 micro s.matmul_q4_tiled_fma t1042_y model_layers_25_self_attn_o_proj_weight_qr -> t1048_c
node vec_add_50 micro s.vec_add t1038_embedding t1048_c -> t1042_y
node rmsnorm_51 micro s.rmsnorm t1042_y model_layers_25_post_attention_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_179 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_25_mlp_gate_proj_weight_qr -> t1066_y
node matmul_q4_tiled_fma_180 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_25_mlp_up_proj_weight_qr -> t1068_c
node act_sigmoid_25 micro s.act_sigmoid t1066_y -> t1064_c
node vec_mul_50 micro s.vec_mul t1066_y t1064_c -> t1062_c
node vec_mul_51 micro s.vec_mul t1062_c t1068_c -> t1064_c
node matmul_q4_tiled_fma_181 micro s.matmul_q4_tiled_fma_3 t1064_c model_layers_25_mlp_down_proj_weight_qr -> t1048_c
node vec_add_51 micro s.vec_add t1042_y t1048_c -> t1038_embedding
node rmsnorm_52 micro s.rmsnorm t1038_embedding model_layers_26_input_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_182 micro s.matmul_q4_tiled_fma t1048_c model_layers_26_self_attn_q_proj_weight_qr -> t1042_y
node matmul_q4_tiled_fma_183 micro s.matmul_q4_tiled_fma t1048_c model_layers_26_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_184 micro s.matmul_q4_tiled_fma t1048_c model_layers_26_self_attn_v_proj_weight_qr -> t1044_c
node rope_52 micro s.rope t1042_y position -> t1042_y
node rope_53 micro s.rope t1046_c position -> t1046_c
node kv_scatter_52 micro s.kv_scatter t1046_c position s792 -> s792
node kv_scatter_53 micro s.kv_scatter t1044_c position s793 -> s793
node flash_prefill_26 micro s.flash_prefill t1042_y s792 s793 position -> t1044_c
node matmul_q4_tiled_fma_185 micro s.matmul_q4_tiled_fma t1044_c model_layers_26_self_attn_o_proj_weight_qr -> t1042_y
node vec_add_52 micro s.vec_add t1038_embedding t1042_y -> t1044_c
node rmsnorm_53 micro s.rmsnorm t1044_c model_layers_26_post_attention_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_186 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_26_mlp_gate_proj_weight_qr -> t1064_c
node matmul_q4_tiled_fma_187 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_26_mlp_up_proj_weight_qr -> t1062_c
node act_sigmoid_26 micro s.act_sigmoid t1064_c -> t1068_c
node vec_mul_52 micro s.vec_mul t1064_c t1068_c -> t1066_y
node vec_mul_53 micro s.vec_mul t1066_y t1062_c -> t1068_c
node matmul_q4_tiled_fma_188 micro s.matmul_q4_tiled_fma_3 t1068_c model_layers_26_mlp_down_proj_weight_qr -> t1042_y
node vec_add_53 micro s.vec_add t1044_c t1042_y -> t1038_embedding
node rmsnorm_54 micro s.rmsnorm t1038_embedding model_layers_27_input_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_189 micro s.matmul_q4_tiled_fma t1042_y model_layers_27_self_attn_q_proj_weight_qr -> t1044_c
node matmul_q4_tiled_fma_190 micro s.matmul_q4_tiled_fma t1042_y model_layers_27_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_191 micro s.matmul_q4_tiled_fma t1042_y model_layers_27_self_attn_v_proj_weight_qr -> t1048_c
node rope_54 micro s.rope t1044_c position -> t1044_c
node rope_55 micro s.rope t1046_c position -> t1046_c
node kv_scatter_54 micro s.kv_scatter t1046_c position s794 -> s794
node kv_scatter_55 micro s.kv_scatter t1048_c position s795 -> s795
node flash_prefill_27 micro s.flash_prefill t1044_c s794 s795 position -> t1048_c
node matmul_q4_tiled_fma_192 micro s.matmul_q4_tiled_fma t1048_c model_layers_27_self_attn_o_proj_weight_qr -> t1044_c
node vec_add_54 micro s.vec_add t1038_embedding t1044_c -> t1048_c
node rmsnorm_55 micro s.rmsnorm t1048_c model_layers_27_post_attention_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_193 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_27_mlp_gate_proj_weight_qr -> t1068_c
node matmul_q4_tiled_fma_194 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_27_mlp_up_proj_weight_qr -> t1066_y
node act_sigmoid_27 micro s.act_sigmoid t1068_c -> t1062_c
node vec_mul_54 micro s.vec_mul t1068_c t1062_c -> t1064_c
node vec_mul_55 micro s.vec_mul t1064_c t1066_y -> t1062_c
node matmul_q4_tiled_fma_195 micro s.matmul_q4_tiled_fma_3 t1062_c model_layers_27_mlp_down_proj_weight_qr -> t1044_c
node vec_add_55 micro s.vec_add t1048_c t1044_c -> t1038_embedding
node rmsnorm_56 micro s.rmsnorm t1038_embedding model_layers_28_input_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_196 micro s.matmul_q4_tiled_fma t1044_c model_layers_28_self_attn_q_proj_weight_qr -> t1048_c
node matmul_q4_tiled_fma_197 micro s.matmul_q4_tiled_fma t1044_c model_layers_28_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_198 micro s.matmul_q4_tiled_fma t1044_c model_layers_28_self_attn_v_proj_weight_qr -> t1042_y
node rope_56 micro s.rope t1048_c position -> t1048_c
node rope_57 micro s.rope t1046_c position -> t1046_c
node kv_scatter_56 micro s.kv_scatter t1046_c position s796 -> s796
node kv_scatter_57 micro s.kv_scatter t1042_y position s797 -> s797
node flash_prefill_28 micro s.flash_prefill t1048_c s796 s797 position -> t1042_y
node matmul_q4_tiled_fma_199 micro s.matmul_q4_tiled_fma t1042_y model_layers_28_self_attn_o_proj_weight_qr -> t1048_c
node vec_add_56 micro s.vec_add t1038_embedding t1048_c -> t1042_y
node rmsnorm_57 micro s.rmsnorm t1042_y model_layers_28_post_attention_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_200 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_28_mlp_gate_proj_weight_qr -> t1062_c
node matmul_q4_tiled_fma_201 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_28_mlp_up_proj_weight_qr -> t1064_c
node act_sigmoid_28 micro s.act_sigmoid t1062_c -> t1066_y
node vec_mul_56 micro s.vec_mul t1062_c t1066_y -> t1068_c
node vec_mul_57 micro s.vec_mul t1068_c t1064_c -> t1066_y
node matmul_q4_tiled_fma_202 micro s.matmul_q4_tiled_fma_3 t1066_y model_layers_28_mlp_down_proj_weight_qr -> t1048_c
node vec_add_57 micro s.vec_add t1042_y t1048_c -> t1038_embedding
node rmsnorm_58 micro s.rmsnorm t1038_embedding model_layers_29_input_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_203 micro s.matmul_q4_tiled_fma t1048_c model_layers_29_self_attn_q_proj_weight_qr -> t1042_y
node matmul_q4_tiled_fma_204 micro s.matmul_q4_tiled_fma t1048_c model_layers_29_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_205 micro s.matmul_q4_tiled_fma t1048_c model_layers_29_self_attn_v_proj_weight_qr -> t1044_c
node rope_58 micro s.rope t1042_y position -> t1042_y
node rope_59 micro s.rope t1046_c position -> t1046_c
node kv_scatter_58 micro s.kv_scatter t1046_c position s798 -> s798
node kv_scatter_59 micro s.kv_scatter t1044_c position s799 -> s799
node flash_prefill_29 micro s.flash_prefill t1042_y s798 s799 position -> t1044_c
node matmul_q4_tiled_fma_206 micro s.matmul_q4_tiled_fma t1044_c model_layers_29_self_attn_o_proj_weight_qr -> t1042_y
node vec_add_58 micro s.vec_add t1038_embedding t1042_y -> t1044_c
node rmsnorm_59 micro s.rmsnorm t1044_c model_layers_29_post_attention_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_207 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_29_mlp_gate_proj_weight_qr -> t1066_y
node matmul_q4_tiled_fma_208 micro s.matmul_q4_tiled_fma_2 t1042_y model_layers_29_mlp_up_proj_weight_qr -> t1068_c
node act_sigmoid_29 micro s.act_sigmoid t1066_y -> t1064_c
node vec_mul_58 micro s.vec_mul t1066_y t1064_c -> t1062_c
node vec_mul_59 micro s.vec_mul t1062_c t1068_c -> t1064_c
node matmul_q4_tiled_fma_209 micro s.matmul_q4_tiled_fma_3 t1064_c model_layers_29_mlp_down_proj_weight_qr -> t1042_y
node vec_add_59 micro s.vec_add t1044_c t1042_y -> t1038_embedding
node rmsnorm_60 micro s.rmsnorm t1038_embedding model_layers_30_input_layernorm_weight -> t1042_y
node matmul_q4_tiled_fma_210 micro s.matmul_q4_tiled_fma t1042_y model_layers_30_self_attn_q_proj_weight_qr -> t1044_c
node matmul_q4_tiled_fma_211 micro s.matmul_q4_tiled_fma t1042_y model_layers_30_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_212 micro s.matmul_q4_tiled_fma t1042_y model_layers_30_self_attn_v_proj_weight_qr -> t1048_c
node rope_60 micro s.rope t1044_c position -> t1044_c
node rope_61 micro s.rope t1046_c position -> t1046_c
node kv_scatter_60 micro s.kv_scatter t1046_c position s800 -> s800
node kv_scatter_61 micro s.kv_scatter t1048_c position s801 -> s801
node flash_prefill_30 micro s.flash_prefill t1044_c s800 s801 position -> t1048_c
node matmul_q4_tiled_fma_213 micro s.matmul_q4_tiled_fma t1048_c model_layers_30_self_attn_o_proj_weight_qr -> t1044_c
node vec_add_60 micro s.vec_add t1038_embedding t1044_c -> t1048_c
node rmsnorm_61 micro s.rmsnorm t1048_c model_layers_30_post_attention_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_214 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_30_mlp_gate_proj_weight_qr -> t1064_c
node matmul_q4_tiled_fma_215 micro s.matmul_q4_tiled_fma_2 t1044_c model_layers_30_mlp_up_proj_weight_qr -> t1062_c
node act_sigmoid_30 micro s.act_sigmoid t1064_c -> t1068_c
node vec_mul_60 micro s.vec_mul t1064_c t1068_c -> t1066_y
node vec_mul_61 micro s.vec_mul t1066_y t1062_c -> t1068_c
node matmul_q4_tiled_fma_216 micro s.matmul_q4_tiled_fma_3 t1068_c model_layers_30_mlp_down_proj_weight_qr -> t1044_c
node vec_add_61 micro s.vec_add t1048_c t1044_c -> t1038_embedding
node rmsnorm_62 micro s.rmsnorm t1038_embedding model_layers_31_input_layernorm_weight -> t1044_c
node matmul_q4_tiled_fma_217 micro s.matmul_q4_tiled_fma t1044_c model_layers_31_self_attn_q_proj_weight_qr -> t1048_c
node matmul_q4_tiled_fma_218 micro s.matmul_q4_tiled_fma t1044_c model_layers_31_self_attn_k_proj_weight_qr -> t1046_c
node matmul_q4_tiled_fma_219 micro s.matmul_q4_tiled_fma t1044_c model_layers_31_self_attn_v_proj_weight_qr -> t1042_y
node rope_62 micro s.rope t1048_c position -> t1048_c
node rope_63 micro s.rope t1046_c position -> t1046_c
node kv_scatter_62 micro s.kv_scatter t1046_c position s802 -> s802
node kv_scatter_63 micro s.kv_scatter t1042_y position s803 -> s803
node flash_prefill_31 micro s.flash_prefill t1048_c s802 s803 position -> t1042_y
node matmul_q4_tiled_fma_220 micro s.matmul_q4_tiled_fma t1042_y model_layers_31_self_attn_o_proj_weight_qr -> t1048_c
node vec_add_62 micro s.vec_add t1038_embedding t1048_c -> t1042_y
node rmsnorm_63 micro s.rmsnorm t1042_y model_layers_31_post_attention_layernorm_weight -> t1048_c
node matmul_q4_tiled_fma_221 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_31_mlp_gate_proj_weight_qr -> t1068_c
node matmul_q4_tiled_fma_222 micro s.matmul_q4_tiled_fma_2 t1048_c model_layers_31_mlp_up_proj_weight_qr -> t1066_y
node silu_mul_0 micro s.silu_mul t1068_c t1066_y -> t1062_c
node matmul_q4_tiled_fma_223 micro s.matmul_q4_tiled_fma_3 t1062_c model_layers_31_mlp_down_proj_weight_qr -> t1048_c
node vec_add_63 micro s.vec_add t1042_y t1048_c -> t1038_embedding
node last_row_0 session sessionCopyInto:srcOff=0,dstOff=0,n=4096,lastRowOf=64 t1038_embedding -> s824
phase chunk2 rows=16 last=t4285_c partial
node embedding_p2 ext fir.embedding token model_embed_tokens_weight_emb -> t2684_embedding
node rmsnorm_64 micro s.rmsnorm_2 t2684_embedding model_layers_0_input_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_224 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_0_self_attn_q_proj_weight_qr -> t2690_c
node matmul_q4_tiled_fma_225 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_0_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_226 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_0_self_attn_v_proj_weight_qr -> t2694_c
node rope_64 micro s.rope_2 t2690_c position -> t2690_c
node rope_65 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_64 micro s.kv_scatter_2 t2692_c position s740 -> s740
node kv_scatter_65 micro s.kv_scatter_2 t2694_c position s741 -> s741
node flash_prefill_32 micro s.flash_prefill_2 t2690_c s740 s741 position -> t2694_c
node matmul_q4_tiled_fma_227 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_0_self_attn_o_proj_weight_qr -> t2690_c
node vec_add_64 micro s.vec_add_2 t2684_embedding t2690_c -> t2694_c
node rmsnorm_65 micro s.rmsnorm_2 t2694_c model_layers_0_post_attention_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_228 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_0_mlp_gate_proj_weight_qr -> t2708_c
node matmul_q4_tiled_fma_229 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_0_mlp_up_proj_weight_qr -> t2710_c
node act_sigmoid_31 micro s.act_sigmoid_2 t2708_c -> t2712_y
node vec_mul_62 micro s.vec_mul_2 t2708_c t2712_y -> t2714_c
node vec_mul_63 micro s.vec_mul_2 t2714_c t2710_c -> t2712_y
node matmul_q4_tiled_fma_230 micro s.matmul_q4_tiled_fma_6 t2712_y model_layers_0_mlp_down_proj_weight_qr -> t2690_c
node vec_add_65 micro s.vec_add_2 t2694_c t2690_c -> t2684_embedding
node rmsnorm_66 micro s.rmsnorm_2 t2684_embedding model_layers_1_input_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_231 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_1_self_attn_q_proj_weight_qr -> t2694_c
node matmul_q4_tiled_fma_232 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_1_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_233 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_1_self_attn_v_proj_weight_qr -> t2688_y
node rope_66 micro s.rope_2 t2694_c position -> t2694_c
node rope_67 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_66 micro s.kv_scatter_2 t2692_c position s742 -> s742
node kv_scatter_67 micro s.kv_scatter_2 t2688_y position s743 -> s743
node flash_prefill_33 micro s.flash_prefill_2 t2694_c s742 s743 position -> t2688_y
node matmul_q4_tiled_fma_234 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_1_self_attn_o_proj_weight_qr -> t2694_c
node vec_add_66 micro s.vec_add_2 t2684_embedding t2694_c -> t2688_y
node rmsnorm_67 micro s.rmsnorm_2 t2688_y model_layers_1_post_attention_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_235 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_1_mlp_gate_proj_weight_qr -> t2712_y
node matmul_q4_tiled_fma_236 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_1_mlp_up_proj_weight_qr -> t2714_c
node act_sigmoid_32 micro s.act_sigmoid_2 t2712_y -> t2710_c
node vec_mul_64 micro s.vec_mul_2 t2712_y t2710_c -> t2708_c
node vec_mul_65 micro s.vec_mul_2 t2708_c t2714_c -> t2710_c
node matmul_q4_tiled_fma_237 micro s.matmul_q4_tiled_fma_6 t2710_c model_layers_1_mlp_down_proj_weight_qr -> t2694_c
node vec_add_67 micro s.vec_add_2 t2688_y t2694_c -> t2684_embedding
node rmsnorm_68 micro s.rmsnorm_2 t2684_embedding model_layers_2_input_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_238 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_2_self_attn_q_proj_weight_qr -> t2688_y
node matmul_q4_tiled_fma_239 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_2_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_240 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_2_self_attn_v_proj_weight_qr -> t2690_c
node rope_68 micro s.rope_2 t2688_y position -> t2688_y
node rope_69 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_68 micro s.kv_scatter_2 t2692_c position s744 -> s744
node kv_scatter_69 micro s.kv_scatter_2 t2690_c position s745 -> s745
node flash_prefill_34 micro s.flash_prefill_2 t2688_y s744 s745 position -> t2690_c
node matmul_q4_tiled_fma_241 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_2_self_attn_o_proj_weight_qr -> t2688_y
node vec_add_68 micro s.vec_add_2 t2684_embedding t2688_y -> t2690_c
node rmsnorm_69 micro s.rmsnorm_2 t2690_c model_layers_2_post_attention_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_242 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_2_mlp_gate_proj_weight_qr -> t2710_c
node matmul_q4_tiled_fma_243 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_2_mlp_up_proj_weight_qr -> t2708_c
node act_sigmoid_33 micro s.act_sigmoid_2 t2710_c -> t2714_c
node vec_mul_66 micro s.vec_mul_2 t2710_c t2714_c -> t2712_y
node vec_mul_67 micro s.vec_mul_2 t2712_y t2708_c -> t2714_c
node matmul_q4_tiled_fma_244 micro s.matmul_q4_tiled_fma_6 t2714_c model_layers_2_mlp_down_proj_weight_qr -> t2688_y
node vec_add_69 micro s.vec_add_2 t2690_c t2688_y -> t2684_embedding
node rmsnorm_70 micro s.rmsnorm_2 t2684_embedding model_layers_3_input_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_245 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_3_self_attn_q_proj_weight_qr -> t2690_c
node matmul_q4_tiled_fma_246 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_3_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_247 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_3_self_attn_v_proj_weight_qr -> t2694_c
node rope_70 micro s.rope_2 t2690_c position -> t2690_c
node rope_71 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_70 micro s.kv_scatter_2 t2692_c position s746 -> s746
node kv_scatter_71 micro s.kv_scatter_2 t2694_c position s747 -> s747
node flash_prefill_35 micro s.flash_prefill_2 t2690_c s746 s747 position -> t2694_c
node matmul_q4_tiled_fma_248 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_3_self_attn_o_proj_weight_qr -> t2690_c
node vec_add_70 micro s.vec_add_2 t2684_embedding t2690_c -> t2694_c
node rmsnorm_71 micro s.rmsnorm_2 t2694_c model_layers_3_post_attention_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_249 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_3_mlp_gate_proj_weight_qr -> t2714_c
node matmul_q4_tiled_fma_250 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_3_mlp_up_proj_weight_qr -> t2712_y
node act_sigmoid_34 micro s.act_sigmoid_2 t2714_c -> t2708_c
node vec_mul_68 micro s.vec_mul_2 t2714_c t2708_c -> t2710_c
node vec_mul_69 micro s.vec_mul_2 t2710_c t2712_y -> t2708_c
node matmul_q4_tiled_fma_251 micro s.matmul_q4_tiled_fma_6 t2708_c model_layers_3_mlp_down_proj_weight_qr -> t2690_c
node vec_add_71 micro s.vec_add_2 t2694_c t2690_c -> t2684_embedding
node rmsnorm_72 micro s.rmsnorm_2 t2684_embedding model_layers_4_input_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_252 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_4_self_attn_q_proj_weight_qr -> t2694_c
node matmul_q4_tiled_fma_253 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_4_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_254 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_4_self_attn_v_proj_weight_qr -> t2688_y
node rope_72 micro s.rope_2 t2694_c position -> t2694_c
node rope_73 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_72 micro s.kv_scatter_2 t2692_c position s748 -> s748
node kv_scatter_73 micro s.kv_scatter_2 t2688_y position s749 -> s749
node flash_prefill_36 micro s.flash_prefill_2 t2694_c s748 s749 position -> t2688_y
node matmul_q4_tiled_fma_255 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_4_self_attn_o_proj_weight_qr -> t2694_c
node vec_add_72 micro s.vec_add_2 t2684_embedding t2694_c -> t2688_y
node rmsnorm_73 micro s.rmsnorm_2 t2688_y model_layers_4_post_attention_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_256 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_4_mlp_gate_proj_weight_qr -> t2708_c
node matmul_q4_tiled_fma_257 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_4_mlp_up_proj_weight_qr -> t2710_c
node act_sigmoid_35 micro s.act_sigmoid_2 t2708_c -> t2712_y
node vec_mul_70 micro s.vec_mul_2 t2708_c t2712_y -> t2714_c
node vec_mul_71 micro s.vec_mul_2 t2714_c t2710_c -> t2712_y
node matmul_q4_tiled_fma_258 micro s.matmul_q4_tiled_fma_6 t2712_y model_layers_4_mlp_down_proj_weight_qr -> t2694_c
node vec_add_73 micro s.vec_add_2 t2688_y t2694_c -> t2684_embedding
node rmsnorm_74 micro s.rmsnorm_2 t2684_embedding model_layers_5_input_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_259 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_5_self_attn_q_proj_weight_qr -> t2688_y
node matmul_q4_tiled_fma_260 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_5_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_261 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_5_self_attn_v_proj_weight_qr -> t2690_c
node rope_74 micro s.rope_2 t2688_y position -> t2688_y
node rope_75 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_74 micro s.kv_scatter_2 t2692_c position s750 -> s750
node kv_scatter_75 micro s.kv_scatter_2 t2690_c position s751 -> s751
node flash_prefill_37 micro s.flash_prefill_2 t2688_y s750 s751 position -> t2690_c
node matmul_q4_tiled_fma_262 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_5_self_attn_o_proj_weight_qr -> t2688_y
node vec_add_74 micro s.vec_add_2 t2684_embedding t2688_y -> t2690_c
node rmsnorm_75 micro s.rmsnorm_2 t2690_c model_layers_5_post_attention_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_263 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_5_mlp_gate_proj_weight_qr -> t2712_y
node matmul_q4_tiled_fma_264 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_5_mlp_up_proj_weight_qr -> t2714_c
node act_sigmoid_36 micro s.act_sigmoid_2 t2712_y -> t2710_c
node vec_mul_72 micro s.vec_mul_2 t2712_y t2710_c -> t2708_c
node vec_mul_73 micro s.vec_mul_2 t2708_c t2714_c -> t2710_c
node matmul_q4_tiled_fma_265 micro s.matmul_q4_tiled_fma_6 t2710_c model_layers_5_mlp_down_proj_weight_qr -> t2688_y
node vec_add_75 micro s.vec_add_2 t2690_c t2688_y -> t2684_embedding
node rmsnorm_76 micro s.rmsnorm_2 t2684_embedding model_layers_6_input_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_266 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_6_self_attn_q_proj_weight_qr -> t2690_c
node matmul_q4_tiled_fma_267 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_6_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_268 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_6_self_attn_v_proj_weight_qr -> t2694_c
node rope_76 micro s.rope_2 t2690_c position -> t2690_c
node rope_77 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_76 micro s.kv_scatter_2 t2692_c position s752 -> s752
node kv_scatter_77 micro s.kv_scatter_2 t2694_c position s753 -> s753
node flash_prefill_38 micro s.flash_prefill_2 t2690_c s752 s753 position -> t2694_c
node matmul_q4_tiled_fma_269 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_6_self_attn_o_proj_weight_qr -> t2690_c
node vec_add_76 micro s.vec_add_2 t2684_embedding t2690_c -> t2694_c
node rmsnorm_77 micro s.rmsnorm_2 t2694_c model_layers_6_post_attention_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_270 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_6_mlp_gate_proj_weight_qr -> t2710_c
node matmul_q4_tiled_fma_271 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_6_mlp_up_proj_weight_qr -> t2708_c
node act_sigmoid_37 micro s.act_sigmoid_2 t2710_c -> t2714_c
node vec_mul_74 micro s.vec_mul_2 t2710_c t2714_c -> t2712_y
node vec_mul_75 micro s.vec_mul_2 t2712_y t2708_c -> t2714_c
node matmul_q4_tiled_fma_272 micro s.matmul_q4_tiled_fma_6 t2714_c model_layers_6_mlp_down_proj_weight_qr -> t2690_c
node vec_add_77 micro s.vec_add_2 t2694_c t2690_c -> t2684_embedding
node rmsnorm_78 micro s.rmsnorm_2 t2684_embedding model_layers_7_input_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_273 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_7_self_attn_q_proj_weight_qr -> t2694_c
node matmul_q4_tiled_fma_274 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_7_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_275 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_7_self_attn_v_proj_weight_qr -> t2688_y
node rope_78 micro s.rope_2 t2694_c position -> t2694_c
node rope_79 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_78 micro s.kv_scatter_2 t2692_c position s754 -> s754
node kv_scatter_79 micro s.kv_scatter_2 t2688_y position s755 -> s755
node flash_prefill_39 micro s.flash_prefill_2 t2694_c s754 s755 position -> t2688_y
node matmul_q4_tiled_fma_276 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_7_self_attn_o_proj_weight_qr -> t2694_c
node vec_add_78 micro s.vec_add_2 t2684_embedding t2694_c -> t2688_y
node rmsnorm_79 micro s.rmsnorm_2 t2688_y model_layers_7_post_attention_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_277 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_7_mlp_gate_proj_weight_qr -> t2714_c
node matmul_q4_tiled_fma_278 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_7_mlp_up_proj_weight_qr -> t2712_y
node act_sigmoid_38 micro s.act_sigmoid_2 t2714_c -> t2708_c
node vec_mul_76 micro s.vec_mul_2 t2714_c t2708_c -> t2710_c
node vec_mul_77 micro s.vec_mul_2 t2710_c t2712_y -> t2708_c
node matmul_q4_tiled_fma_279 micro s.matmul_q4_tiled_fma_6 t2708_c model_layers_7_mlp_down_proj_weight_qr -> t2694_c
node vec_add_79 micro s.vec_add_2 t2688_y t2694_c -> t2684_embedding
node rmsnorm_80 micro s.rmsnorm_2 t2684_embedding model_layers_8_input_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_280 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_8_self_attn_q_proj_weight_qr -> t2688_y
node matmul_q4_tiled_fma_281 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_8_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_282 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_8_self_attn_v_proj_weight_qr -> t2690_c
node rope_80 micro s.rope_2 t2688_y position -> t2688_y
node rope_81 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_80 micro s.kv_scatter_2 t2692_c position s756 -> s756
node kv_scatter_81 micro s.kv_scatter_2 t2690_c position s757 -> s757
node flash_prefill_40 micro s.flash_prefill_2 t2688_y s756 s757 position -> t2690_c
node matmul_q4_tiled_fma_283 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_8_self_attn_o_proj_weight_qr -> t2688_y
node vec_add_80 micro s.vec_add_2 t2684_embedding t2688_y -> t2690_c
node rmsnorm_81 micro s.rmsnorm_2 t2690_c model_layers_8_post_attention_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_284 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_8_mlp_gate_proj_weight_qr -> t2708_c
node matmul_q4_tiled_fma_285 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_8_mlp_up_proj_weight_qr -> t2710_c
node act_sigmoid_39 micro s.act_sigmoid_2 t2708_c -> t2712_y
node vec_mul_78 micro s.vec_mul_2 t2708_c t2712_y -> t2714_c
node vec_mul_79 micro s.vec_mul_2 t2714_c t2710_c -> t2712_y
node matmul_q4_tiled_fma_286 micro s.matmul_q4_tiled_fma_6 t2712_y model_layers_8_mlp_down_proj_weight_qr -> t2688_y
node vec_add_81 micro s.vec_add_2 t2690_c t2688_y -> t2684_embedding
node rmsnorm_82 micro s.rmsnorm_2 t2684_embedding model_layers_9_input_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_287 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_9_self_attn_q_proj_weight_qr -> t2690_c
node matmul_q4_tiled_fma_288 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_9_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_289 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_9_self_attn_v_proj_weight_qr -> t2694_c
node rope_82 micro s.rope_2 t2690_c position -> t2690_c
node rope_83 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_82 micro s.kv_scatter_2 t2692_c position s758 -> s758
node kv_scatter_83 micro s.kv_scatter_2 t2694_c position s759 -> s759
node flash_prefill_41 micro s.flash_prefill_2 t2690_c s758 s759 position -> t2694_c
node matmul_q4_tiled_fma_290 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_9_self_attn_o_proj_weight_qr -> t2690_c
node vec_add_82 micro s.vec_add_2 t2684_embedding t2690_c -> t2694_c
node rmsnorm_83 micro s.rmsnorm_2 t2694_c model_layers_9_post_attention_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_291 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_9_mlp_gate_proj_weight_qr -> t2712_y
node matmul_q4_tiled_fma_292 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_9_mlp_up_proj_weight_qr -> t2714_c
node act_sigmoid_40 micro s.act_sigmoid_2 t2712_y -> t2710_c
node vec_mul_80 micro s.vec_mul_2 t2712_y t2710_c -> t2708_c
node vec_mul_81 micro s.vec_mul_2 t2708_c t2714_c -> t2710_c
node matmul_q4_tiled_fma_293 micro s.matmul_q4_tiled_fma_6 t2710_c model_layers_9_mlp_down_proj_weight_qr -> t2690_c
node vec_add_83 micro s.vec_add_2 t2694_c t2690_c -> t2684_embedding
node rmsnorm_84 micro s.rmsnorm_2 t2684_embedding model_layers_10_input_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_294 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_10_self_attn_q_proj_weight_qr -> t2694_c
node matmul_q4_tiled_fma_295 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_10_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_296 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_10_self_attn_v_proj_weight_qr -> t2688_y
node rope_84 micro s.rope_2 t2694_c position -> t2694_c
node rope_85 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_84 micro s.kv_scatter_2 t2692_c position s760 -> s760
node kv_scatter_85 micro s.kv_scatter_2 t2688_y position s761 -> s761
node flash_prefill_42 micro s.flash_prefill_2 t2694_c s760 s761 position -> t2688_y
node matmul_q4_tiled_fma_297 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_10_self_attn_o_proj_weight_qr -> t2694_c
node vec_add_84 micro s.vec_add_2 t2684_embedding t2694_c -> t2688_y
node rmsnorm_85 micro s.rmsnorm_2 t2688_y model_layers_10_post_attention_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_298 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_10_mlp_gate_proj_weight_qr -> t2710_c
node matmul_q4_tiled_fma_299 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_10_mlp_up_proj_weight_qr -> t2708_c
node act_sigmoid_41 micro s.act_sigmoid_2 t2710_c -> t2714_c
node vec_mul_82 micro s.vec_mul_2 t2710_c t2714_c -> t2712_y
node vec_mul_83 micro s.vec_mul_2 t2712_y t2708_c -> t2714_c
node matmul_q4_tiled_fma_300 micro s.matmul_q4_tiled_fma_6 t2714_c model_layers_10_mlp_down_proj_weight_qr -> t2694_c
node vec_add_85 micro s.vec_add_2 t2688_y t2694_c -> t2684_embedding
node rmsnorm_86 micro s.rmsnorm_2 t2684_embedding model_layers_11_input_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_301 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_11_self_attn_q_proj_weight_qr -> t2688_y
node matmul_q4_tiled_fma_302 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_11_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_303 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_11_self_attn_v_proj_weight_qr -> t2690_c
node rope_86 micro s.rope_2 t2688_y position -> t2688_y
node rope_87 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_86 micro s.kv_scatter_2 t2692_c position s762 -> s762
node kv_scatter_87 micro s.kv_scatter_2 t2690_c position s763 -> s763
node flash_prefill_43 micro s.flash_prefill_2 t2688_y s762 s763 position -> t2690_c
node matmul_q4_tiled_fma_304 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_11_self_attn_o_proj_weight_qr -> t2688_y
node vec_add_86 micro s.vec_add_2 t2684_embedding t2688_y -> t2690_c
node rmsnorm_87 micro s.rmsnorm_2 t2690_c model_layers_11_post_attention_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_305 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_11_mlp_gate_proj_weight_qr -> t2714_c
node matmul_q4_tiled_fma_306 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_11_mlp_up_proj_weight_qr -> t2712_y
node act_sigmoid_42 micro s.act_sigmoid_2 t2714_c -> t2708_c
node vec_mul_84 micro s.vec_mul_2 t2714_c t2708_c -> t2710_c
node vec_mul_85 micro s.vec_mul_2 t2710_c t2712_y -> t2708_c
node matmul_q4_tiled_fma_307 micro s.matmul_q4_tiled_fma_6 t2708_c model_layers_11_mlp_down_proj_weight_qr -> t2688_y
node vec_add_87 micro s.vec_add_2 t2690_c t2688_y -> t2684_embedding
node rmsnorm_88 micro s.rmsnorm_2 t2684_embedding model_layers_12_input_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_308 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_12_self_attn_q_proj_weight_qr -> t2690_c
node matmul_q4_tiled_fma_309 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_12_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_310 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_12_self_attn_v_proj_weight_qr -> t2694_c
node rope_88 micro s.rope_2 t2690_c position -> t2690_c
node rope_89 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_88 micro s.kv_scatter_2 t2692_c position s764 -> s764
node kv_scatter_89 micro s.kv_scatter_2 t2694_c position s765 -> s765
node flash_prefill_44 micro s.flash_prefill_2 t2690_c s764 s765 position -> t2694_c
node matmul_q4_tiled_fma_311 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_12_self_attn_o_proj_weight_qr -> t2690_c
node vec_add_88 micro s.vec_add_2 t2684_embedding t2690_c -> t2694_c
node rmsnorm_89 micro s.rmsnorm_2 t2694_c model_layers_12_post_attention_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_312 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_12_mlp_gate_proj_weight_qr -> t2708_c
node matmul_q4_tiled_fma_313 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_12_mlp_up_proj_weight_qr -> t2710_c
node act_sigmoid_43 micro s.act_sigmoid_2 t2708_c -> t2712_y
node vec_mul_86 micro s.vec_mul_2 t2708_c t2712_y -> t2714_c
node vec_mul_87 micro s.vec_mul_2 t2714_c t2710_c -> t2712_y
node matmul_q4_tiled_fma_314 micro s.matmul_q4_tiled_fma_6 t2712_y model_layers_12_mlp_down_proj_weight_qr -> t2690_c
node vec_add_89 micro s.vec_add_2 t2694_c t2690_c -> t2684_embedding
node rmsnorm_90 micro s.rmsnorm_2 t2684_embedding model_layers_13_input_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_315 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_13_self_attn_q_proj_weight_qr -> t2694_c
node matmul_q4_tiled_fma_316 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_13_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_317 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_13_self_attn_v_proj_weight_qr -> t2688_y
node rope_90 micro s.rope_2 t2694_c position -> t2694_c
node rope_91 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_90 micro s.kv_scatter_2 t2692_c position s766 -> s766
node kv_scatter_91 micro s.kv_scatter_2 t2688_y position s767 -> s767
node flash_prefill_45 micro s.flash_prefill_2 t2694_c s766 s767 position -> t2688_y
node matmul_q4_tiled_fma_318 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_13_self_attn_o_proj_weight_qr -> t2694_c
node vec_add_90 micro s.vec_add_2 t2684_embedding t2694_c -> t2688_y
node rmsnorm_91 micro s.rmsnorm_2 t2688_y model_layers_13_post_attention_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_319 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_13_mlp_gate_proj_weight_qr -> t2712_y
node matmul_q4_tiled_fma_320 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_13_mlp_up_proj_weight_qr -> t2714_c
node act_sigmoid_44 micro s.act_sigmoid_2 t2712_y -> t2710_c
node vec_mul_88 micro s.vec_mul_2 t2712_y t2710_c -> t2708_c
node vec_mul_89 micro s.vec_mul_2 t2708_c t2714_c -> t2710_c
node matmul_q4_tiled_fma_321 micro s.matmul_q4_tiled_fma_6 t2710_c model_layers_13_mlp_down_proj_weight_qr -> t2694_c
node vec_add_91 micro s.vec_add_2 t2688_y t2694_c -> t2684_embedding
node rmsnorm_92 micro s.rmsnorm_2 t2684_embedding model_layers_14_input_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_322 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_14_self_attn_q_proj_weight_qr -> t2688_y
node matmul_q4_tiled_fma_323 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_14_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_324 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_14_self_attn_v_proj_weight_qr -> t2690_c
node rope_92 micro s.rope_2 t2688_y position -> t2688_y
node rope_93 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_92 micro s.kv_scatter_2 t2692_c position s768 -> s768
node kv_scatter_93 micro s.kv_scatter_2 t2690_c position s769 -> s769
node flash_prefill_46 micro s.flash_prefill_2 t2688_y s768 s769 position -> t2690_c
node matmul_q4_tiled_fma_325 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_14_self_attn_o_proj_weight_qr -> t2688_y
node vec_add_92 micro s.vec_add_2 t2684_embedding t2688_y -> t2690_c
node rmsnorm_93 micro s.rmsnorm_2 t2690_c model_layers_14_post_attention_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_326 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_14_mlp_gate_proj_weight_qr -> t2710_c
node matmul_q4_tiled_fma_327 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_14_mlp_up_proj_weight_qr -> t2708_c
node act_sigmoid_45 micro s.act_sigmoid_2 t2710_c -> t2714_c
node vec_mul_90 micro s.vec_mul_2 t2710_c t2714_c -> t2712_y
node vec_mul_91 micro s.vec_mul_2 t2712_y t2708_c -> t2714_c
node matmul_q4_tiled_fma_328 micro s.matmul_q4_tiled_fma_6 t2714_c model_layers_14_mlp_down_proj_weight_qr -> t2688_y
node vec_add_93 micro s.vec_add_2 t2690_c t2688_y -> t2684_embedding
node rmsnorm_94 micro s.rmsnorm_2 t2684_embedding model_layers_15_input_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_329 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_15_self_attn_q_proj_weight_qr -> t2690_c
node matmul_q4_tiled_fma_330 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_15_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_331 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_15_self_attn_v_proj_weight_qr -> t2694_c
node rope_94 micro s.rope_2 t2690_c position -> t2690_c
node rope_95 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_94 micro s.kv_scatter_2 t2692_c position s770 -> s770
node kv_scatter_95 micro s.kv_scatter_2 t2694_c position s771 -> s771
node flash_prefill_47 micro s.flash_prefill_2 t2690_c s770 s771 position -> t2694_c
node matmul_q4_tiled_fma_332 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_15_self_attn_o_proj_weight_qr -> t2690_c
node vec_add_94 micro s.vec_add_2 t2684_embedding t2690_c -> t2694_c
node rmsnorm_95 micro s.rmsnorm_2 t2694_c model_layers_15_post_attention_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_333 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_15_mlp_gate_proj_weight_qr -> t2714_c
node matmul_q4_tiled_fma_334 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_15_mlp_up_proj_weight_qr -> t2712_y
node act_sigmoid_46 micro s.act_sigmoid_2 t2714_c -> t2708_c
node vec_mul_92 micro s.vec_mul_2 t2714_c t2708_c -> t2710_c
node vec_mul_93 micro s.vec_mul_2 t2710_c t2712_y -> t2708_c
node matmul_q4_tiled_fma_335 micro s.matmul_q4_tiled_fma_6 t2708_c model_layers_15_mlp_down_proj_weight_qr -> t2690_c
node vec_add_95 micro s.vec_add_2 t2694_c t2690_c -> t2684_embedding
node rmsnorm_96 micro s.rmsnorm_2 t2684_embedding model_layers_16_input_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_336 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_16_self_attn_q_proj_weight_qr -> t2694_c
node matmul_q4_tiled_fma_337 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_16_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_338 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_16_self_attn_v_proj_weight_qr -> t2688_y
node rope_96 micro s.rope_2 t2694_c position -> t2694_c
node rope_97 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_96 micro s.kv_scatter_2 t2692_c position s772 -> s772
node kv_scatter_97 micro s.kv_scatter_2 t2688_y position s773 -> s773
node flash_prefill_48 micro s.flash_prefill_2 t2694_c s772 s773 position -> t2688_y
node matmul_q4_tiled_fma_339 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_16_self_attn_o_proj_weight_qr -> t2694_c
node vec_add_96 micro s.vec_add_2 t2684_embedding t2694_c -> t2688_y
node rmsnorm_97 micro s.rmsnorm_2 t2688_y model_layers_16_post_attention_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_340 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_16_mlp_gate_proj_weight_qr -> t2708_c
node matmul_q4_tiled_fma_341 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_16_mlp_up_proj_weight_qr -> t2710_c
node act_sigmoid_47 micro s.act_sigmoid_2 t2708_c -> t2712_y
node vec_mul_94 micro s.vec_mul_2 t2708_c t2712_y -> t2714_c
node vec_mul_95 micro s.vec_mul_2 t2714_c t2710_c -> t2712_y
node matmul_q4_tiled_fma_342 micro s.matmul_q4_tiled_fma_6 t2712_y model_layers_16_mlp_down_proj_weight_qr -> t2694_c
node vec_add_97 micro s.vec_add_2 t2688_y t2694_c -> t2684_embedding
node rmsnorm_98 micro s.rmsnorm_2 t2684_embedding model_layers_17_input_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_343 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_17_self_attn_q_proj_weight_qr -> t2688_y
node matmul_q4_tiled_fma_344 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_17_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_345 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_17_self_attn_v_proj_weight_qr -> t2690_c
node rope_98 micro s.rope_2 t2688_y position -> t2688_y
node rope_99 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_98 micro s.kv_scatter_2 t2692_c position s774 -> s774
node kv_scatter_99 micro s.kv_scatter_2 t2690_c position s775 -> s775
node flash_prefill_49 micro s.flash_prefill_2 t2688_y s774 s775 position -> t2690_c
node matmul_q4_tiled_fma_346 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_17_self_attn_o_proj_weight_qr -> t2688_y
node vec_add_98 micro s.vec_add_2 t2684_embedding t2688_y -> t2690_c
node rmsnorm_99 micro s.rmsnorm_2 t2690_c model_layers_17_post_attention_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_347 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_17_mlp_gate_proj_weight_qr -> t2712_y
node matmul_q4_tiled_fma_348 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_17_mlp_up_proj_weight_qr -> t2714_c
node act_sigmoid_48 micro s.act_sigmoid_2 t2712_y -> t2710_c
node vec_mul_96 micro s.vec_mul_2 t2712_y t2710_c -> t2708_c
node vec_mul_97 micro s.vec_mul_2 t2708_c t2714_c -> t2710_c
node matmul_q4_tiled_fma_349 micro s.matmul_q4_tiled_fma_6 t2710_c model_layers_17_mlp_down_proj_weight_qr -> t2688_y
node vec_add_99 micro s.vec_add_2 t2690_c t2688_y -> t2684_embedding
node rmsnorm_100 micro s.rmsnorm_2 t2684_embedding model_layers_18_input_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_350 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_18_self_attn_q_proj_weight_qr -> t2690_c
node matmul_q4_tiled_fma_351 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_18_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_352 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_18_self_attn_v_proj_weight_qr -> t2694_c
node rope_100 micro s.rope_2 t2690_c position -> t2690_c
node rope_101 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_100 micro s.kv_scatter_2 t2692_c position s776 -> s776
node kv_scatter_101 micro s.kv_scatter_2 t2694_c position s777 -> s777
node flash_prefill_50 micro s.flash_prefill_2 t2690_c s776 s777 position -> t2694_c
node matmul_q4_tiled_fma_353 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_18_self_attn_o_proj_weight_qr -> t2690_c
node vec_add_100 micro s.vec_add_2 t2684_embedding t2690_c -> t2694_c
node rmsnorm_101 micro s.rmsnorm_2 t2694_c model_layers_18_post_attention_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_354 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_18_mlp_gate_proj_weight_qr -> t2710_c
node matmul_q4_tiled_fma_355 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_18_mlp_up_proj_weight_qr -> t2708_c
node act_sigmoid_49 micro s.act_sigmoid_2 t2710_c -> t2714_c
node vec_mul_98 micro s.vec_mul_2 t2710_c t2714_c -> t2712_y
node vec_mul_99 micro s.vec_mul_2 t2712_y t2708_c -> t2714_c
node matmul_q4_tiled_fma_356 micro s.matmul_q4_tiled_fma_6 t2714_c model_layers_18_mlp_down_proj_weight_qr -> t2690_c
node vec_add_101 micro s.vec_add_2 t2694_c t2690_c -> t2684_embedding
node rmsnorm_102 micro s.rmsnorm_2 t2684_embedding model_layers_19_input_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_357 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_19_self_attn_q_proj_weight_qr -> t2694_c
node matmul_q4_tiled_fma_358 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_19_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_359 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_19_self_attn_v_proj_weight_qr -> t2688_y
node rope_102 micro s.rope_2 t2694_c position -> t2694_c
node rope_103 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_102 micro s.kv_scatter_2 t2692_c position s778 -> s778
node kv_scatter_103 micro s.kv_scatter_2 t2688_y position s779 -> s779
node flash_prefill_51 micro s.flash_prefill_2 t2694_c s778 s779 position -> t2688_y
node matmul_q4_tiled_fma_360 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_19_self_attn_o_proj_weight_qr -> t2694_c
node vec_add_102 micro s.vec_add_2 t2684_embedding t2694_c -> t2688_y
node rmsnorm_103 micro s.rmsnorm_2 t2688_y model_layers_19_post_attention_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_361 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_19_mlp_gate_proj_weight_qr -> t2714_c
node matmul_q4_tiled_fma_362 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_19_mlp_up_proj_weight_qr -> t2712_y
node act_sigmoid_50 micro s.act_sigmoid_2 t2714_c -> t2708_c
node vec_mul_100 micro s.vec_mul_2 t2714_c t2708_c -> t2710_c
node vec_mul_101 micro s.vec_mul_2 t2710_c t2712_y -> t2708_c
node matmul_q4_tiled_fma_363 micro s.matmul_q4_tiled_fma_6 t2708_c model_layers_19_mlp_down_proj_weight_qr -> t2694_c
node vec_add_103 micro s.vec_add_2 t2688_y t2694_c -> t2684_embedding
node rmsnorm_104 micro s.rmsnorm_2 t2684_embedding model_layers_20_input_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_364 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_20_self_attn_q_proj_weight_qr -> t2688_y
node matmul_q4_tiled_fma_365 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_20_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_366 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_20_self_attn_v_proj_weight_qr -> t2690_c
node rope_104 micro s.rope_2 t2688_y position -> t2688_y
node rope_105 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_104 micro s.kv_scatter_2 t2692_c position s780 -> s780
node kv_scatter_105 micro s.kv_scatter_2 t2690_c position s781 -> s781
node flash_prefill_52 micro s.flash_prefill_2 t2688_y s780 s781 position -> t2690_c
node matmul_q4_tiled_fma_367 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_20_self_attn_o_proj_weight_qr -> t2688_y
node vec_add_104 micro s.vec_add_2 t2684_embedding t2688_y -> t2690_c
node rmsnorm_105 micro s.rmsnorm_2 t2690_c model_layers_20_post_attention_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_368 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_20_mlp_gate_proj_weight_qr -> t2708_c
node matmul_q4_tiled_fma_369 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_20_mlp_up_proj_weight_qr -> t2710_c
node act_sigmoid_51 micro s.act_sigmoid_2 t2708_c -> t2712_y
node vec_mul_102 micro s.vec_mul_2 t2708_c t2712_y -> t2714_c
node vec_mul_103 micro s.vec_mul_2 t2714_c t2710_c -> t2712_y
node matmul_q4_tiled_fma_370 micro s.matmul_q4_tiled_fma_6 t2712_y model_layers_20_mlp_down_proj_weight_qr -> t2688_y
node vec_add_105 micro s.vec_add_2 t2690_c t2688_y -> t2684_embedding
node rmsnorm_106 micro s.rmsnorm_2 t2684_embedding model_layers_21_input_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_371 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_21_self_attn_q_proj_weight_qr -> t2690_c
node matmul_q4_tiled_fma_372 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_21_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_373 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_21_self_attn_v_proj_weight_qr -> t2694_c
node rope_106 micro s.rope_2 t2690_c position -> t2690_c
node rope_107 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_106 micro s.kv_scatter_2 t2692_c position s782 -> s782
node kv_scatter_107 micro s.kv_scatter_2 t2694_c position s783 -> s783
node flash_prefill_53 micro s.flash_prefill_2 t2690_c s782 s783 position -> t2694_c
node matmul_q4_tiled_fma_374 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_21_self_attn_o_proj_weight_qr -> t2690_c
node vec_add_106 micro s.vec_add_2 t2684_embedding t2690_c -> t2694_c
node rmsnorm_107 micro s.rmsnorm_2 t2694_c model_layers_21_post_attention_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_375 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_21_mlp_gate_proj_weight_qr -> t2712_y
node matmul_q4_tiled_fma_376 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_21_mlp_up_proj_weight_qr -> t2714_c
node act_sigmoid_52 micro s.act_sigmoid_2 t2712_y -> t2710_c
node vec_mul_104 micro s.vec_mul_2 t2712_y t2710_c -> t2708_c
node vec_mul_105 micro s.vec_mul_2 t2708_c t2714_c -> t2710_c
node matmul_q4_tiled_fma_377 micro s.matmul_q4_tiled_fma_6 t2710_c model_layers_21_mlp_down_proj_weight_qr -> t2690_c
node vec_add_107 micro s.vec_add_2 t2694_c t2690_c -> t2684_embedding
node rmsnorm_108 micro s.rmsnorm_2 t2684_embedding model_layers_22_input_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_378 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_22_self_attn_q_proj_weight_qr -> t2694_c
node matmul_q4_tiled_fma_379 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_22_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_380 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_22_self_attn_v_proj_weight_qr -> t2688_y
node rope_108 micro s.rope_2 t2694_c position -> t2694_c
node rope_109 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_108 micro s.kv_scatter_2 t2692_c position s784 -> s784
node kv_scatter_109 micro s.kv_scatter_2 t2688_y position s785 -> s785
node flash_prefill_54 micro s.flash_prefill_2 t2694_c s784 s785 position -> t2688_y
node matmul_q4_tiled_fma_381 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_22_self_attn_o_proj_weight_qr -> t2694_c
node vec_add_108 micro s.vec_add_2 t2684_embedding t2694_c -> t2688_y
node rmsnorm_109 micro s.rmsnorm_2 t2688_y model_layers_22_post_attention_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_382 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_22_mlp_gate_proj_weight_qr -> t2710_c
node matmul_q4_tiled_fma_383 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_22_mlp_up_proj_weight_qr -> t2708_c
node act_sigmoid_53 micro s.act_sigmoid_2 t2710_c -> t2714_c
node vec_mul_106 micro s.vec_mul_2 t2710_c t2714_c -> t2712_y
node vec_mul_107 micro s.vec_mul_2 t2712_y t2708_c -> t2714_c
node matmul_q4_tiled_fma_384 micro s.matmul_q4_tiled_fma_6 t2714_c model_layers_22_mlp_down_proj_weight_qr -> t2694_c
node vec_add_109 micro s.vec_add_2 t2688_y t2694_c -> t2684_embedding
node rmsnorm_110 micro s.rmsnorm_2 t2684_embedding model_layers_23_input_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_385 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_23_self_attn_q_proj_weight_qr -> t2688_y
node matmul_q4_tiled_fma_386 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_23_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_387 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_23_self_attn_v_proj_weight_qr -> t2690_c
node rope_110 micro s.rope_2 t2688_y position -> t2688_y
node rope_111 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_110 micro s.kv_scatter_2 t2692_c position s786 -> s786
node kv_scatter_111 micro s.kv_scatter_2 t2690_c position s787 -> s787
node flash_prefill_55 micro s.flash_prefill_2 t2688_y s786 s787 position -> t2690_c
node matmul_q4_tiled_fma_388 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_23_self_attn_o_proj_weight_qr -> t2688_y
node vec_add_110 micro s.vec_add_2 t2684_embedding t2688_y -> t2690_c
node rmsnorm_111 micro s.rmsnorm_2 t2690_c model_layers_23_post_attention_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_389 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_23_mlp_gate_proj_weight_qr -> t2714_c
node matmul_q4_tiled_fma_390 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_23_mlp_up_proj_weight_qr -> t2712_y
node act_sigmoid_54 micro s.act_sigmoid_2 t2714_c -> t2708_c
node vec_mul_108 micro s.vec_mul_2 t2714_c t2708_c -> t2710_c
node vec_mul_109 micro s.vec_mul_2 t2710_c t2712_y -> t2708_c
node matmul_q4_tiled_fma_391 micro s.matmul_q4_tiled_fma_6 t2708_c model_layers_23_mlp_down_proj_weight_qr -> t2688_y
node vec_add_111 micro s.vec_add_2 t2690_c t2688_y -> t2684_embedding
node rmsnorm_112 micro s.rmsnorm_2 t2684_embedding model_layers_24_input_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_392 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_24_self_attn_q_proj_weight_qr -> t2690_c
node matmul_q4_tiled_fma_393 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_24_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_394 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_24_self_attn_v_proj_weight_qr -> t2694_c
node rope_112 micro s.rope_2 t2690_c position -> t2690_c
node rope_113 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_112 micro s.kv_scatter_2 t2692_c position s788 -> s788
node kv_scatter_113 micro s.kv_scatter_2 t2694_c position s789 -> s789
node flash_prefill_56 micro s.flash_prefill_2 t2690_c s788 s789 position -> t2694_c
node matmul_q4_tiled_fma_395 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_24_self_attn_o_proj_weight_qr -> t2690_c
node vec_add_112 micro s.vec_add_2 t2684_embedding t2690_c -> t2694_c
node rmsnorm_113 micro s.rmsnorm_2 t2694_c model_layers_24_post_attention_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_396 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_24_mlp_gate_proj_weight_qr -> t2708_c
node matmul_q4_tiled_fma_397 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_24_mlp_up_proj_weight_qr -> t2710_c
node act_sigmoid_55 micro s.act_sigmoid_2 t2708_c -> t2712_y
node vec_mul_110 micro s.vec_mul_2 t2708_c t2712_y -> t2714_c
node vec_mul_111 micro s.vec_mul_2 t2714_c t2710_c -> t2712_y
node matmul_q4_tiled_fma_398 micro s.matmul_q4_tiled_fma_6 t2712_y model_layers_24_mlp_down_proj_weight_qr -> t2690_c
node vec_add_113 micro s.vec_add_2 t2694_c t2690_c -> t2684_embedding
node rmsnorm_114 micro s.rmsnorm_2 t2684_embedding model_layers_25_input_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_399 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_25_self_attn_q_proj_weight_qr -> t2694_c
node matmul_q4_tiled_fma_400 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_25_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_401 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_25_self_attn_v_proj_weight_qr -> t2688_y
node rope_114 micro s.rope_2 t2694_c position -> t2694_c
node rope_115 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_114 micro s.kv_scatter_2 t2692_c position s790 -> s790
node kv_scatter_115 micro s.kv_scatter_2 t2688_y position s791 -> s791
node flash_prefill_57 micro s.flash_prefill_2 t2694_c s790 s791 position -> t2688_y
node matmul_q4_tiled_fma_402 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_25_self_attn_o_proj_weight_qr -> t2694_c
node vec_add_114 micro s.vec_add_2 t2684_embedding t2694_c -> t2688_y
node rmsnorm_115 micro s.rmsnorm_2 t2688_y model_layers_25_post_attention_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_403 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_25_mlp_gate_proj_weight_qr -> t2712_y
node matmul_q4_tiled_fma_404 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_25_mlp_up_proj_weight_qr -> t2714_c
node act_sigmoid_56 micro s.act_sigmoid_2 t2712_y -> t2710_c
node vec_mul_112 micro s.vec_mul_2 t2712_y t2710_c -> t2708_c
node vec_mul_113 micro s.vec_mul_2 t2708_c t2714_c -> t2710_c
node matmul_q4_tiled_fma_405 micro s.matmul_q4_tiled_fma_6 t2710_c model_layers_25_mlp_down_proj_weight_qr -> t2694_c
node vec_add_115 micro s.vec_add_2 t2688_y t2694_c -> t2684_embedding
node rmsnorm_116 micro s.rmsnorm_2 t2684_embedding model_layers_26_input_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_406 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_26_self_attn_q_proj_weight_qr -> t2688_y
node matmul_q4_tiled_fma_407 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_26_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_408 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_26_self_attn_v_proj_weight_qr -> t2690_c
node rope_116 micro s.rope_2 t2688_y position -> t2688_y
node rope_117 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_116 micro s.kv_scatter_2 t2692_c position s792 -> s792
node kv_scatter_117 micro s.kv_scatter_2 t2690_c position s793 -> s793
node flash_prefill_58 micro s.flash_prefill_2 t2688_y s792 s793 position -> t2690_c
node matmul_q4_tiled_fma_409 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_26_self_attn_o_proj_weight_qr -> t2688_y
node vec_add_116 micro s.vec_add_2 t2684_embedding t2688_y -> t2690_c
node rmsnorm_117 micro s.rmsnorm_2 t2690_c model_layers_26_post_attention_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_410 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_26_mlp_gate_proj_weight_qr -> t2710_c
node matmul_q4_tiled_fma_411 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_26_mlp_up_proj_weight_qr -> t2708_c
node act_sigmoid_57 micro s.act_sigmoid_2 t2710_c -> t2714_c
node vec_mul_114 micro s.vec_mul_2 t2710_c t2714_c -> t2712_y
node vec_mul_115 micro s.vec_mul_2 t2712_y t2708_c -> t2714_c
node matmul_q4_tiled_fma_412 micro s.matmul_q4_tiled_fma_6 t2714_c model_layers_26_mlp_down_proj_weight_qr -> t2688_y
node vec_add_117 micro s.vec_add_2 t2690_c t2688_y -> t2684_embedding
node rmsnorm_118 micro s.rmsnorm_2 t2684_embedding model_layers_27_input_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_413 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_27_self_attn_q_proj_weight_qr -> t2690_c
node matmul_q4_tiled_fma_414 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_27_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_415 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_27_self_attn_v_proj_weight_qr -> t2694_c
node rope_118 micro s.rope_2 t2690_c position -> t2690_c
node rope_119 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_118 micro s.kv_scatter_2 t2692_c position s794 -> s794
node kv_scatter_119 micro s.kv_scatter_2 t2694_c position s795 -> s795
node flash_prefill_59 micro s.flash_prefill_2 t2690_c s794 s795 position -> t2694_c
node matmul_q4_tiled_fma_416 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_27_self_attn_o_proj_weight_qr -> t2690_c
node vec_add_118 micro s.vec_add_2 t2684_embedding t2690_c -> t2694_c
node rmsnorm_119 micro s.rmsnorm_2 t2694_c model_layers_27_post_attention_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_417 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_27_mlp_gate_proj_weight_qr -> t2714_c
node matmul_q4_tiled_fma_418 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_27_mlp_up_proj_weight_qr -> t2712_y
node act_sigmoid_58 micro s.act_sigmoid_2 t2714_c -> t2708_c
node vec_mul_116 micro s.vec_mul_2 t2714_c t2708_c -> t2710_c
node vec_mul_117 micro s.vec_mul_2 t2710_c t2712_y -> t2708_c
node matmul_q4_tiled_fma_419 micro s.matmul_q4_tiled_fma_6 t2708_c model_layers_27_mlp_down_proj_weight_qr -> t2690_c
node vec_add_119 micro s.vec_add_2 t2694_c t2690_c -> t2684_embedding
node rmsnorm_120 micro s.rmsnorm_2 t2684_embedding model_layers_28_input_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_420 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_28_self_attn_q_proj_weight_qr -> t2694_c
node matmul_q4_tiled_fma_421 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_28_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_422 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_28_self_attn_v_proj_weight_qr -> t2688_y
node rope_120 micro s.rope_2 t2694_c position -> t2694_c
node rope_121 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_120 micro s.kv_scatter_2 t2692_c position s796 -> s796
node kv_scatter_121 micro s.kv_scatter_2 t2688_y position s797 -> s797
node flash_prefill_60 micro s.flash_prefill_2 t2694_c s796 s797 position -> t2688_y
node matmul_q4_tiled_fma_423 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_28_self_attn_o_proj_weight_qr -> t2694_c
node vec_add_120 micro s.vec_add_2 t2684_embedding t2694_c -> t2688_y
node rmsnorm_121 micro s.rmsnorm_2 t2688_y model_layers_28_post_attention_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_424 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_28_mlp_gate_proj_weight_qr -> t2708_c
node matmul_q4_tiled_fma_425 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_28_mlp_up_proj_weight_qr -> t2710_c
node act_sigmoid_59 micro s.act_sigmoid_2 t2708_c -> t2712_y
node vec_mul_118 micro s.vec_mul_2 t2708_c t2712_y -> t2714_c
node vec_mul_119 micro s.vec_mul_2 t2714_c t2710_c -> t2712_y
node matmul_q4_tiled_fma_426 micro s.matmul_q4_tiled_fma_6 t2712_y model_layers_28_mlp_down_proj_weight_qr -> t2694_c
node vec_add_121 micro s.vec_add_2 t2688_y t2694_c -> t2684_embedding
node rmsnorm_122 micro s.rmsnorm_2 t2684_embedding model_layers_29_input_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_427 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_29_self_attn_q_proj_weight_qr -> t2688_y
node matmul_q4_tiled_fma_428 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_29_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_429 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_29_self_attn_v_proj_weight_qr -> t2690_c
node rope_122 micro s.rope_2 t2688_y position -> t2688_y
node rope_123 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_122 micro s.kv_scatter_2 t2692_c position s798 -> s798
node kv_scatter_123 micro s.kv_scatter_2 t2690_c position s799 -> s799
node flash_prefill_61 micro s.flash_prefill_2 t2688_y s798 s799 position -> t2690_c
node matmul_q4_tiled_fma_430 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_29_self_attn_o_proj_weight_qr -> t2688_y
node vec_add_122 micro s.vec_add_2 t2684_embedding t2688_y -> t2690_c
node rmsnorm_123 micro s.rmsnorm_2 t2690_c model_layers_29_post_attention_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_431 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_29_mlp_gate_proj_weight_qr -> t2712_y
node matmul_q4_tiled_fma_432 micro s.matmul_q4_tiled_fma_5 t2688_y model_layers_29_mlp_up_proj_weight_qr -> t2714_c
node silu_mul_1 micro s.silu_mul_2 t2712_y t2714_c -> t2710_c
node matmul_q4_tiled_fma_433 micro s.matmul_q4_tiled_fma_6 t2710_c model_layers_29_mlp_down_proj_weight_qr -> t2688_y
node vec_add_123 micro s.vec_add_2 t2690_c t2688_y -> t2684_embedding
node rmsnorm_124 micro s.rmsnorm_2 t2684_embedding model_layers_30_input_layernorm_weight -> t2688_y
node matmul_q4_tiled_fma_434 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_30_self_attn_q_proj_weight_qr -> t2690_c
node matmul_q4_tiled_fma_435 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_30_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_436 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_30_self_attn_v_proj_weight_qr -> t2694_c
node rope_124 micro s.rope_2 t2690_c position -> t2690_c
node rope_125 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_124 micro s.kv_scatter_2 t2692_c position s800 -> s800
node kv_scatter_125 micro s.kv_scatter_2 t2694_c position s801 -> s801
node flash_prefill_62 micro s.flash_prefill_2 t2690_c s800 s801 position -> t2694_c
node matmul_q4_tiled_fma_437 micro s.matmul_q4_tiled_fma_4 t2694_c model_layers_30_self_attn_o_proj_weight_qr -> t2690_c
node vec_add_124 micro s.vec_add_2 t2684_embedding t2690_c -> t2694_c
node rmsnorm_125 micro s.rmsnorm_2 t2694_c model_layers_30_post_attention_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_438 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_30_mlp_gate_proj_weight_qr -> t2710_c
node matmul_q4_tiled_fma_439 micro s.matmul_q4_tiled_fma_5 t2690_c model_layers_30_mlp_up_proj_weight_qr -> t2714_c
node act_sigmoid_60 micro s.act_sigmoid_2 t2710_c -> t2712_y
node vec_mul_120 micro s.vec_mul_2 t2710_c t2712_y -> t2708_c
node vec_mul_121 micro s.vec_mul_2 t2708_c t2714_c -> t2712_y
node matmul_q4_tiled_fma_440 micro s.matmul_q4_tiled_fma_6 t2712_y model_layers_30_mlp_down_proj_weight_qr -> t2690_c
node vec_add_125 micro s.vec_add_2 t2694_c t2690_c -> t2684_embedding
node rmsnorm_126 micro s.rmsnorm_2 t2684_embedding model_layers_31_input_layernorm_weight -> t2690_c
node matmul_q4_tiled_fma_441 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_31_self_attn_q_proj_weight_qr -> t2694_c
node matmul_q4_tiled_fma_442 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_31_self_attn_k_proj_weight_qr -> t2692_c
node matmul_q4_tiled_fma_443 micro s.matmul_q4_tiled_fma_4 t2690_c model_layers_31_self_attn_v_proj_weight_qr -> t2688_y
node rope_126 micro s.rope_2 t2694_c position -> t2694_c
node rope_127 micro s.rope_2 t2692_c position -> t2692_c
node kv_scatter_126 micro s.kv_scatter_2 t2692_c position s802 -> s802
node kv_scatter_127 micro s.kv_scatter_2 t2688_y position s803 -> s803
node flash_prefill_63 micro s.flash_prefill_2 t2694_c s802 s803 position -> t2688_y
node matmul_q4_tiled_fma_444 micro s.matmul_q4_tiled_fma_4 t2688_y model_layers_31_self_attn_o_proj_weight_qr -> t2694_c
node vec_add_126 micro s.vec_add_2 t2684_embedding t2694_c -> t2688_y
node rmsnorm_127 micro s.rmsnorm_2 t2688_y model_layers_31_post_attention_layernorm_weight -> t2694_c
node matmul_q4_tiled_fma_445 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_31_mlp_gate_proj_weight_qr -> t2712_y
node matmul_q4_tiled_fma_446 micro s.matmul_q4_tiled_fma_5 t2694_c model_layers_31_mlp_up_proj_weight_qr -> t2708_c
node silu_mul_2 micro s.silu_mul_2 t2712_y t2708_c -> t2714_c
node matmul_q4_tiled_fma_447 micro s.matmul_q4_tiled_fma_6 t2714_c model_layers_31_mlp_down_proj_weight_qr -> t2694_c
node vec_add_127 micro s.vec_add_2 t2688_y t2694_c -> t2684_embedding
node last_row_1 session sessionCopyInto:srcOff=0,dstOff=0,n=4096,lastRowOf=16 t2684_embedding -> s824
node rmsnorm_128 micro s.rmsnorm_3 s824 model_norm_weight -> t4284_y
node matmul_q4_simd_0 micro s.matmul_q4_simd t4284_y lm_head_weight_qr -> t4285_c
phase decode
node embedding ext fir.embedding token model_embed_tokens_weight_emb -> t4311_embedding
node rmsnorm_129 micro s.rmsnorm_3 t4311_embedding model_layers_0_input_layernorm_weight -> t4316_y
node matmul_q4_simd_1 micro s.matmul_q4_simd_2 t4316_y model_layers_0_self_attn_q_proj_weight_qr -> t4317_c
node matmul_q4_simd_2 micro s.matmul_q4_simd_2 t4316_y model_layers_0_self_attn_k_proj_weight_qr -> t4318_c
node matmul_q4_simd_3 micro s.matmul_q4_simd_2 t4316_y model_layers_0_self_attn_v_proj_weight_qr -> t4319_c
node rope_128_side micro s.side t4317_c position t4318_c position -> t4317_c t4318_c
node kv_scatter_128_side micro s.side_2 t4318_c position s740 t4319_c position s741 -> s740 s741
node attention_0 micro s.attention t4317_c s740 s741 position -> t4319_c
node matmul_q4_simd_4 micro s.matmul_q4_simd_2 t4319_c model_layers_0_self_attn_o_proj_weight_qr -> t4317_c
node add_rmsnorm_0 micro s.add_rmsnorm t4311_embedding t4317_c model_layers_0_post_attention_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_5 micro s.matmul_q4_simd_3 t4318_c model_layers_0_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_6 micro s.matmul_q4_simd_3 t4318_c model_layers_0_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_3 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_7 micro s.matmul_q4_simd_4 t4340_c model_layers_0_mlp_down_proj_weight_qr -> t4318_c
node add_rmsnorm_1 micro s.add_rmsnorm t4319_c t4318_c model_layers_1_input_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_8 micro s.matmul_q4_simd_2 t4311_embedding model_layers_1_self_attn_q_proj_weight_qr -> t4318_c
node matmul_q4_simd_9 micro s.matmul_q4_simd_2 t4311_embedding model_layers_1_self_attn_k_proj_weight_qr -> t4319_c
node matmul_q4_simd_10 micro s.matmul_q4_simd_2 t4311_embedding model_layers_1_self_attn_v_proj_weight_qr -> t4316_y
node rope_130_side micro s.side t4318_c position t4319_c position -> t4318_c t4319_c
node kv_scatter_130_side micro s.side_2 t4319_c position s742 t4316_y position s743 -> s742 s743
node attention_1 micro s.attention t4318_c s742 s743 position -> t4316_y
node matmul_q4_simd_11 micro s.matmul_q4_simd_2 t4316_y model_layers_1_self_attn_o_proj_weight_qr -> t4318_c
node add_rmsnorm_2 micro s.add_rmsnorm t4317_c t4318_c model_layers_1_post_attention_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_12 micro s.matmul_q4_simd_3 t4319_c model_layers_1_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_13 micro s.matmul_q4_simd_3 t4319_c model_layers_1_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_4 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_14 micro s.matmul_q4_simd_4 t4333_c model_layers_1_mlp_down_proj_weight_qr -> t4319_c
node add_rmsnorm_3 micro s.add_rmsnorm t4316_y t4319_c model_layers_2_input_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_15 micro s.matmul_q4_simd_2 t4317_c model_layers_2_self_attn_q_proj_weight_qr -> t4319_c
node matmul_q4_simd_16 micro s.matmul_q4_simd_2 t4317_c model_layers_2_self_attn_k_proj_weight_qr -> t4316_y
node matmul_q4_simd_17 micro s.matmul_q4_simd_2 t4317_c model_layers_2_self_attn_v_proj_weight_qr -> t4311_embedding
node rope_132_side micro s.side t4319_c position t4316_y position -> t4319_c t4316_y
node kv_scatter_132_side micro s.side_2 t4316_y position s744 t4311_embedding position s745 -> s744 s745
node attention_2 micro s.attention t4319_c s744 s745 position -> t4311_embedding
node matmul_q4_simd_18 micro s.matmul_q4_simd_2 t4311_embedding model_layers_2_self_attn_o_proj_weight_qr -> t4319_c
node add_rmsnorm_4 micro s.add_rmsnorm t4318_c t4319_c model_layers_2_post_attention_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_19 micro s.matmul_q4_simd_3 t4316_y model_layers_2_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_20 micro s.matmul_q4_simd_3 t4316_y model_layers_2_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_5 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_21 micro s.matmul_q4_simd_4 t4340_c model_layers_2_mlp_down_proj_weight_qr -> t4316_y
node add_rmsnorm_5 micro s.add_rmsnorm t4311_embedding t4316_y model_layers_3_input_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_22 micro s.matmul_q4_simd_2 t4318_c model_layers_3_self_attn_q_proj_weight_qr -> t4316_y
node matmul_q4_simd_23 micro s.matmul_q4_simd_2 t4318_c model_layers_3_self_attn_k_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_24 micro s.matmul_q4_simd_2 t4318_c model_layers_3_self_attn_v_proj_weight_qr -> t4317_c
node rope_134_side micro s.side t4316_y position t4311_embedding position -> t4316_y t4311_embedding
node kv_scatter_134_side micro s.side_2 t4311_embedding position s746 t4317_c position s747 -> s746 s747
node attention_3 micro s.attention t4316_y s746 s747 position -> t4317_c
node matmul_q4_simd_25 micro s.matmul_q4_simd_2 t4317_c model_layers_3_self_attn_o_proj_weight_qr -> t4316_y
node add_rmsnorm_6 micro s.add_rmsnorm t4319_c t4316_y model_layers_3_post_attention_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_26 micro s.matmul_q4_simd_3 t4311_embedding model_layers_3_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_27 micro s.matmul_q4_simd_3 t4311_embedding model_layers_3_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_6 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_28 micro s.matmul_q4_simd_4 t4333_c model_layers_3_mlp_down_proj_weight_qr -> t4311_embedding
node add_rmsnorm_7 micro s.add_rmsnorm t4317_c t4311_embedding model_layers_4_input_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_29 micro s.matmul_q4_simd_2 t4319_c model_layers_4_self_attn_q_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_30 micro s.matmul_q4_simd_2 t4319_c model_layers_4_self_attn_k_proj_weight_qr -> t4317_c
node matmul_q4_simd_31 micro s.matmul_q4_simd_2 t4319_c model_layers_4_self_attn_v_proj_weight_qr -> t4318_c
node rope_136_side micro s.side t4311_embedding position t4317_c position -> t4311_embedding t4317_c
node kv_scatter_136_side micro s.side_2 t4317_c position s748 t4318_c position s749 -> s748 s749
node attention_4 micro s.attention t4311_embedding s748 s749 position -> t4318_c
node matmul_q4_simd_32 micro s.matmul_q4_simd_2 t4318_c model_layers_4_self_attn_o_proj_weight_qr -> t4311_embedding
node add_rmsnorm_8 micro s.add_rmsnorm t4316_y t4311_embedding model_layers_4_post_attention_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_33 micro s.matmul_q4_simd_3 t4317_c model_layers_4_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_34 micro s.matmul_q4_simd_3 t4317_c model_layers_4_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_7 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_35 micro s.matmul_q4_simd_4 t4340_c model_layers_4_mlp_down_proj_weight_qr -> t4317_c
node add_rmsnorm_9 micro s.add_rmsnorm t4318_c t4317_c model_layers_5_input_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_36 micro s.matmul_q4_simd_2 t4316_y model_layers_5_self_attn_q_proj_weight_qr -> t4317_c
node matmul_q4_simd_37 micro s.matmul_q4_simd_2 t4316_y model_layers_5_self_attn_k_proj_weight_qr -> t4318_c
node matmul_q4_simd_38 micro s.matmul_q4_simd_2 t4316_y model_layers_5_self_attn_v_proj_weight_qr -> t4319_c
node rope_138_side micro s.side t4317_c position t4318_c position -> t4317_c t4318_c
node kv_scatter_138_side micro s.side_2 t4318_c position s750 t4319_c position s751 -> s750 s751
node attention_5 micro s.attention t4317_c s750 s751 position -> t4319_c
node matmul_q4_simd_39 micro s.matmul_q4_simd_2 t4319_c model_layers_5_self_attn_o_proj_weight_qr -> t4317_c
node add_rmsnorm_10 micro s.add_rmsnorm t4311_embedding t4317_c model_layers_5_post_attention_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_40 micro s.matmul_q4_simd_3 t4318_c model_layers_5_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_41 micro s.matmul_q4_simd_3 t4318_c model_layers_5_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_8 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_42 micro s.matmul_q4_simd_4 t4333_c model_layers_5_mlp_down_proj_weight_qr -> t4318_c
node add_rmsnorm_11 micro s.add_rmsnorm t4319_c t4318_c model_layers_6_input_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_43 micro s.matmul_q4_simd_2 t4311_embedding model_layers_6_self_attn_q_proj_weight_qr -> t4318_c
node matmul_q4_simd_44 micro s.matmul_q4_simd_2 t4311_embedding model_layers_6_self_attn_k_proj_weight_qr -> t4319_c
node matmul_q4_simd_45 micro s.matmul_q4_simd_2 t4311_embedding model_layers_6_self_attn_v_proj_weight_qr -> t4316_y
node rope_140_side micro s.side t4318_c position t4319_c position -> t4318_c t4319_c
node kv_scatter_140_side micro s.side_2 t4319_c position s752 t4316_y position s753 -> s752 s753
node attention_6 micro s.attention t4318_c s752 s753 position -> t4316_y
node matmul_q4_simd_46 micro s.matmul_q4_simd_2 t4316_y model_layers_6_self_attn_o_proj_weight_qr -> t4318_c
node add_rmsnorm_12 micro s.add_rmsnorm t4317_c t4318_c model_layers_6_post_attention_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_47 micro s.matmul_q4_simd_3 t4319_c model_layers_6_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_48 micro s.matmul_q4_simd_3 t4319_c model_layers_6_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_9 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_49 micro s.matmul_q4_simd_4 t4340_c model_layers_6_mlp_down_proj_weight_qr -> t4319_c
node add_rmsnorm_13 micro s.add_rmsnorm t4316_y t4319_c model_layers_7_input_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_50 micro s.matmul_q4_simd_2 t4317_c model_layers_7_self_attn_q_proj_weight_qr -> t4319_c
node matmul_q4_simd_51 micro s.matmul_q4_simd_2 t4317_c model_layers_7_self_attn_k_proj_weight_qr -> t4316_y
node matmul_q4_simd_52 micro s.matmul_q4_simd_2 t4317_c model_layers_7_self_attn_v_proj_weight_qr -> t4311_embedding
node rope_142_side micro s.side t4319_c position t4316_y position -> t4319_c t4316_y
node kv_scatter_142_side micro s.side_2 t4316_y position s754 t4311_embedding position s755 -> s754 s755
node attention_7 micro s.attention t4319_c s754 s755 position -> t4311_embedding
node matmul_q4_simd_53 micro s.matmul_q4_simd_2 t4311_embedding model_layers_7_self_attn_o_proj_weight_qr -> t4319_c
node add_rmsnorm_14 micro s.add_rmsnorm t4318_c t4319_c model_layers_7_post_attention_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_54 micro s.matmul_q4_simd_3 t4316_y model_layers_7_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_55 micro s.matmul_q4_simd_3 t4316_y model_layers_7_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_10 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_56 micro s.matmul_q4_simd_4 t4333_c model_layers_7_mlp_down_proj_weight_qr -> t4316_y
node add_rmsnorm_15 micro s.add_rmsnorm t4311_embedding t4316_y model_layers_8_input_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_57 micro s.matmul_q4_simd_2 t4318_c model_layers_8_self_attn_q_proj_weight_qr -> t4316_y
node matmul_q4_simd_58 micro s.matmul_q4_simd_2 t4318_c model_layers_8_self_attn_k_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_59 micro s.matmul_q4_simd_2 t4318_c model_layers_8_self_attn_v_proj_weight_qr -> t4317_c
node rope_144_side micro s.side t4316_y position t4311_embedding position -> t4316_y t4311_embedding
node kv_scatter_144_side micro s.side_2 t4311_embedding position s756 t4317_c position s757 -> s756 s757
node attention_8 micro s.attention t4316_y s756 s757 position -> t4317_c
node matmul_q4_simd_60 micro s.matmul_q4_simd_2 t4317_c model_layers_8_self_attn_o_proj_weight_qr -> t4316_y
node add_rmsnorm_16 micro s.add_rmsnorm t4319_c t4316_y model_layers_8_post_attention_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_61 micro s.matmul_q4_simd_3 t4311_embedding model_layers_8_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_62 micro s.matmul_q4_simd_3 t4311_embedding model_layers_8_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_11 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_63 micro s.matmul_q4_simd_4 t4340_c model_layers_8_mlp_down_proj_weight_qr -> t4311_embedding
node add_rmsnorm_17 micro s.add_rmsnorm t4317_c t4311_embedding model_layers_9_input_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_64 micro s.matmul_q4_simd_2 t4319_c model_layers_9_self_attn_q_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_65 micro s.matmul_q4_simd_2 t4319_c model_layers_9_self_attn_k_proj_weight_qr -> t4317_c
node matmul_q4_simd_66 micro s.matmul_q4_simd_2 t4319_c model_layers_9_self_attn_v_proj_weight_qr -> t4318_c
node rope_146_side micro s.side t4311_embedding position t4317_c position -> t4311_embedding t4317_c
node kv_scatter_146_side micro s.side_2 t4317_c position s758 t4318_c position s759 -> s758 s759
node attention_9 micro s.attention t4311_embedding s758 s759 position -> t4318_c
node matmul_q4_simd_67 micro s.matmul_q4_simd_2 t4318_c model_layers_9_self_attn_o_proj_weight_qr -> t4311_embedding
node add_rmsnorm_18 micro s.add_rmsnorm t4316_y t4311_embedding model_layers_9_post_attention_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_68 micro s.matmul_q4_simd_3 t4317_c model_layers_9_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_69 micro s.matmul_q4_simd_3 t4317_c model_layers_9_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_12 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_70 micro s.matmul_q4_simd_4 t4333_c model_layers_9_mlp_down_proj_weight_qr -> t4317_c
node add_rmsnorm_19 micro s.add_rmsnorm t4318_c t4317_c model_layers_10_input_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_71 micro s.matmul_q4_simd_2 t4316_y model_layers_10_self_attn_q_proj_weight_qr -> t4317_c
node matmul_q4_simd_72 micro s.matmul_q4_simd_2 t4316_y model_layers_10_self_attn_k_proj_weight_qr -> t4318_c
node matmul_q4_simd_73 micro s.matmul_q4_simd_2 t4316_y model_layers_10_self_attn_v_proj_weight_qr -> t4319_c
node rope_148_side micro s.side t4317_c position t4318_c position -> t4317_c t4318_c
node kv_scatter_148_side micro s.side_2 t4318_c position s760 t4319_c position s761 -> s760 s761
node attention_10 micro s.attention t4317_c s760 s761 position -> t4319_c
node matmul_q4_simd_74 micro s.matmul_q4_simd_2 t4319_c model_layers_10_self_attn_o_proj_weight_qr -> t4317_c
node add_rmsnorm_20 micro s.add_rmsnorm t4311_embedding t4317_c model_layers_10_post_attention_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_75 micro s.matmul_q4_simd_3 t4318_c model_layers_10_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_76 micro s.matmul_q4_simd_3 t4318_c model_layers_10_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_13 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_77 micro s.matmul_q4_simd_4 t4340_c model_layers_10_mlp_down_proj_weight_qr -> t4318_c
node add_rmsnorm_21 micro s.add_rmsnorm t4319_c t4318_c model_layers_11_input_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_78 micro s.matmul_q4_simd_2 t4311_embedding model_layers_11_self_attn_q_proj_weight_qr -> t4318_c
node matmul_q4_simd_79 micro s.matmul_q4_simd_2 t4311_embedding model_layers_11_self_attn_k_proj_weight_qr -> t4319_c
node matmul_q4_simd_80 micro s.matmul_q4_simd_2 t4311_embedding model_layers_11_self_attn_v_proj_weight_qr -> t4316_y
node rope_150_side micro s.side t4318_c position t4319_c position -> t4318_c t4319_c
node kv_scatter_150_side micro s.side_2 t4319_c position s762 t4316_y position s763 -> s762 s763
node attention_11 micro s.attention t4318_c s762 s763 position -> t4316_y
node matmul_q4_simd_81 micro s.matmul_q4_simd_2 t4316_y model_layers_11_self_attn_o_proj_weight_qr -> t4318_c
node add_rmsnorm_22 micro s.add_rmsnorm t4317_c t4318_c model_layers_11_post_attention_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_82 micro s.matmul_q4_simd_3 t4319_c model_layers_11_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_83 micro s.matmul_q4_simd_3 t4319_c model_layers_11_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_14 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_84 micro s.matmul_q4_simd_4 t4333_c model_layers_11_mlp_down_proj_weight_qr -> t4319_c
node add_rmsnorm_23 micro s.add_rmsnorm t4316_y t4319_c model_layers_12_input_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_85 micro s.matmul_q4_simd_2 t4317_c model_layers_12_self_attn_q_proj_weight_qr -> t4319_c
node matmul_q4_simd_86 micro s.matmul_q4_simd_2 t4317_c model_layers_12_self_attn_k_proj_weight_qr -> t4316_y
node matmul_q4_simd_87 micro s.matmul_q4_simd_2 t4317_c model_layers_12_self_attn_v_proj_weight_qr -> t4311_embedding
node rope_152_side micro s.side t4319_c position t4316_y position -> t4319_c t4316_y
node kv_scatter_152_side micro s.side_2 t4316_y position s764 t4311_embedding position s765 -> s764 s765
node attention_12 micro s.attention t4319_c s764 s765 position -> t4311_embedding
node matmul_q4_simd_88 micro s.matmul_q4_simd_2 t4311_embedding model_layers_12_self_attn_o_proj_weight_qr -> t4319_c
node add_rmsnorm_24 micro s.add_rmsnorm t4318_c t4319_c model_layers_12_post_attention_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_89 micro s.matmul_q4_simd_3 t4316_y model_layers_12_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_90 micro s.matmul_q4_simd_3 t4316_y model_layers_12_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_15 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_91 micro s.matmul_q4_simd_4 t4340_c model_layers_12_mlp_down_proj_weight_qr -> t4316_y
node add_rmsnorm_25 micro s.add_rmsnorm t4311_embedding t4316_y model_layers_13_input_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_92 micro s.matmul_q4_simd_2 t4318_c model_layers_13_self_attn_q_proj_weight_qr -> t4316_y
node matmul_q4_simd_93 micro s.matmul_q4_simd_2 t4318_c model_layers_13_self_attn_k_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_94 micro s.matmul_q4_simd_2 t4318_c model_layers_13_self_attn_v_proj_weight_qr -> t4317_c
node rope_154_side micro s.side t4316_y position t4311_embedding position -> t4316_y t4311_embedding
node kv_scatter_154_side micro s.side_2 t4311_embedding position s766 t4317_c position s767 -> s766 s767
node attention_13 micro s.attention t4316_y s766 s767 position -> t4317_c
node matmul_q4_simd_95 micro s.matmul_q4_simd_2 t4317_c model_layers_13_self_attn_o_proj_weight_qr -> t4316_y
node add_rmsnorm_26 micro s.add_rmsnorm t4319_c t4316_y model_layers_13_post_attention_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_96 micro s.matmul_q4_simd_3 t4311_embedding model_layers_13_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_97 micro s.matmul_q4_simd_3 t4311_embedding model_layers_13_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_16 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_98 micro s.matmul_q4_simd_4 t4333_c model_layers_13_mlp_down_proj_weight_qr -> t4311_embedding
node add_rmsnorm_27 micro s.add_rmsnorm t4317_c t4311_embedding model_layers_14_input_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_99 micro s.matmul_q4_simd_2 t4319_c model_layers_14_self_attn_q_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_100 micro s.matmul_q4_simd_2 t4319_c model_layers_14_self_attn_k_proj_weight_qr -> t4317_c
node matmul_q4_simd_101 micro s.matmul_q4_simd_2 t4319_c model_layers_14_self_attn_v_proj_weight_qr -> t4318_c
node rope_156_side micro s.side t4311_embedding position t4317_c position -> t4311_embedding t4317_c
node kv_scatter_156_side micro s.side_2 t4317_c position s768 t4318_c position s769 -> s768 s769
node attention_14 micro s.attention t4311_embedding s768 s769 position -> t4318_c
node matmul_q4_simd_102 micro s.matmul_q4_simd_2 t4318_c model_layers_14_self_attn_o_proj_weight_qr -> t4311_embedding
node add_rmsnorm_28 micro s.add_rmsnorm t4316_y t4311_embedding model_layers_14_post_attention_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_103 micro s.matmul_q4_simd_3 t4317_c model_layers_14_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_104 micro s.matmul_q4_simd_3 t4317_c model_layers_14_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_17 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_105 micro s.matmul_q4_simd_4 t4340_c model_layers_14_mlp_down_proj_weight_qr -> t4317_c
node add_rmsnorm_29 micro s.add_rmsnorm t4318_c t4317_c model_layers_15_input_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_106 micro s.matmul_q4_simd_2 t4316_y model_layers_15_self_attn_q_proj_weight_qr -> t4317_c
node matmul_q4_simd_107 micro s.matmul_q4_simd_2 t4316_y model_layers_15_self_attn_k_proj_weight_qr -> t4318_c
node matmul_q4_simd_108 micro s.matmul_q4_simd_2 t4316_y model_layers_15_self_attn_v_proj_weight_qr -> t4319_c
node rope_158_side micro s.side t4317_c position t4318_c position -> t4317_c t4318_c
node kv_scatter_158_side micro s.side_2 t4318_c position s770 t4319_c position s771 -> s770 s771
node attention_15 micro s.attention t4317_c s770 s771 position -> t4319_c
node matmul_q4_simd_109 micro s.matmul_q4_simd_2 t4319_c model_layers_15_self_attn_o_proj_weight_qr -> t4317_c
node add_rmsnorm_30 micro s.add_rmsnorm t4311_embedding t4317_c model_layers_15_post_attention_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_110 micro s.matmul_q4_simd_3 t4318_c model_layers_15_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_111 micro s.matmul_q4_simd_3 t4318_c model_layers_15_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_18 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_112 micro s.matmul_q4_simd_4 t4333_c model_layers_15_mlp_down_proj_weight_qr -> t4318_c
node add_rmsnorm_31 micro s.add_rmsnorm t4319_c t4318_c model_layers_16_input_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_113 micro s.matmul_q4_simd_2 t4311_embedding model_layers_16_self_attn_q_proj_weight_qr -> t4318_c
node matmul_q4_simd_114 micro s.matmul_q4_simd_2 t4311_embedding model_layers_16_self_attn_k_proj_weight_qr -> t4319_c
node matmul_q4_simd_115 micro s.matmul_q4_simd_2 t4311_embedding model_layers_16_self_attn_v_proj_weight_qr -> t4316_y
node rope_160_side micro s.side t4318_c position t4319_c position -> t4318_c t4319_c
node kv_scatter_160_side micro s.side_2 t4319_c position s772 t4316_y position s773 -> s772 s773
node attention_16 micro s.attention t4318_c s772 s773 position -> t4316_y
node matmul_q4_simd_116 micro s.matmul_q4_simd_2 t4316_y model_layers_16_self_attn_o_proj_weight_qr -> t4318_c
node add_rmsnorm_32 micro s.add_rmsnorm t4317_c t4318_c model_layers_16_post_attention_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_117 micro s.matmul_q4_simd_3 t4319_c model_layers_16_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_118 micro s.matmul_q4_simd_3 t4319_c model_layers_16_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_19 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_119 micro s.matmul_q4_simd_4 t4340_c model_layers_16_mlp_down_proj_weight_qr -> t4319_c
node add_rmsnorm_33 micro s.add_rmsnorm t4316_y t4319_c model_layers_17_input_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_120 micro s.matmul_q4_simd_2 t4317_c model_layers_17_self_attn_q_proj_weight_qr -> t4319_c
node matmul_q4_simd_121 micro s.matmul_q4_simd_2 t4317_c model_layers_17_self_attn_k_proj_weight_qr -> t4316_y
node matmul_q4_simd_122 micro s.matmul_q4_simd_2 t4317_c model_layers_17_self_attn_v_proj_weight_qr -> t4311_embedding
node rope_162_side micro s.side t4319_c position t4316_y position -> t4319_c t4316_y
node kv_scatter_162_side micro s.side_2 t4316_y position s774 t4311_embedding position s775 -> s774 s775
node attention_17 micro s.attention t4319_c s774 s775 position -> t4311_embedding
node matmul_q4_simd_123 micro s.matmul_q4_simd_2 t4311_embedding model_layers_17_self_attn_o_proj_weight_qr -> t4319_c
node add_rmsnorm_34 micro s.add_rmsnorm t4318_c t4319_c model_layers_17_post_attention_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_124 micro s.matmul_q4_simd_3 t4316_y model_layers_17_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_125 micro s.matmul_q4_simd_3 t4316_y model_layers_17_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_20 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_126 micro s.matmul_q4_simd_4 t4333_c model_layers_17_mlp_down_proj_weight_qr -> t4316_y
node add_rmsnorm_35 micro s.add_rmsnorm t4311_embedding t4316_y model_layers_18_input_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_127 micro s.matmul_q4_simd_2 t4318_c model_layers_18_self_attn_q_proj_weight_qr -> t4316_y
node matmul_q4_simd_128 micro s.matmul_q4_simd_2 t4318_c model_layers_18_self_attn_k_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_129 micro s.matmul_q4_simd_2 t4318_c model_layers_18_self_attn_v_proj_weight_qr -> t4317_c
node rope_164_side micro s.side t4316_y position t4311_embedding position -> t4316_y t4311_embedding
node kv_scatter_164_side micro s.side_2 t4311_embedding position s776 t4317_c position s777 -> s776 s777
node attention_18 micro s.attention t4316_y s776 s777 position -> t4317_c
node matmul_q4_simd_130 micro s.matmul_q4_simd_2 t4317_c model_layers_18_self_attn_o_proj_weight_qr -> t4316_y
node add_rmsnorm_36 micro s.add_rmsnorm t4319_c t4316_y model_layers_18_post_attention_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_131 micro s.matmul_q4_simd_3 t4311_embedding model_layers_18_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_132 micro s.matmul_q4_simd_3 t4311_embedding model_layers_18_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_21 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_133 micro s.matmul_q4_simd_4 t4340_c model_layers_18_mlp_down_proj_weight_qr -> t4311_embedding
node add_rmsnorm_37 micro s.add_rmsnorm t4317_c t4311_embedding model_layers_19_input_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_134 micro s.matmul_q4_simd_2 t4319_c model_layers_19_self_attn_q_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_135 micro s.matmul_q4_simd_2 t4319_c model_layers_19_self_attn_k_proj_weight_qr -> t4317_c
node matmul_q4_simd_136 micro s.matmul_q4_simd_2 t4319_c model_layers_19_self_attn_v_proj_weight_qr -> t4318_c
node rope_166_side micro s.side t4311_embedding position t4317_c position -> t4311_embedding t4317_c
node kv_scatter_166_side micro s.side_2 t4317_c position s778 t4318_c position s779 -> s778 s779
node attention_19 micro s.attention t4311_embedding s778 s779 position -> t4318_c
node matmul_q4_simd_137 micro s.matmul_q4_simd_2 t4318_c model_layers_19_self_attn_o_proj_weight_qr -> t4311_embedding
node add_rmsnorm_38 micro s.add_rmsnorm t4316_y t4311_embedding model_layers_19_post_attention_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_138 micro s.matmul_q4_simd_3 t4317_c model_layers_19_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_139 micro s.matmul_q4_simd_3 t4317_c model_layers_19_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_22 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_140 micro s.matmul_q4_simd_4 t4333_c model_layers_19_mlp_down_proj_weight_qr -> t4317_c
node add_rmsnorm_39 micro s.add_rmsnorm t4318_c t4317_c model_layers_20_input_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_141 micro s.matmul_q4_simd_2 t4316_y model_layers_20_self_attn_q_proj_weight_qr -> t4317_c
node matmul_q4_simd_142 micro s.matmul_q4_simd_2 t4316_y model_layers_20_self_attn_k_proj_weight_qr -> t4318_c
node matmul_q4_simd_143 micro s.matmul_q4_simd_2 t4316_y model_layers_20_self_attn_v_proj_weight_qr -> t4319_c
node rope_168_side micro s.side t4317_c position t4318_c position -> t4317_c t4318_c
node kv_scatter_168_side micro s.side_2 t4318_c position s780 t4319_c position s781 -> s780 s781
node attention_20 micro s.attention t4317_c s780 s781 position -> t4319_c
node matmul_q4_simd_144 micro s.matmul_q4_simd_2 t4319_c model_layers_20_self_attn_o_proj_weight_qr -> t4317_c
node add_rmsnorm_40 micro s.add_rmsnorm t4311_embedding t4317_c model_layers_20_post_attention_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_145 micro s.matmul_q4_simd_3 t4318_c model_layers_20_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_146 micro s.matmul_q4_simd_3 t4318_c model_layers_20_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_23 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_147 micro s.matmul_q4_simd_4 t4340_c model_layers_20_mlp_down_proj_weight_qr -> t4318_c
node add_rmsnorm_41 micro s.add_rmsnorm t4319_c t4318_c model_layers_21_input_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_148 micro s.matmul_q4_simd_2 t4311_embedding model_layers_21_self_attn_q_proj_weight_qr -> t4318_c
node matmul_q4_simd_149 micro s.matmul_q4_simd_2 t4311_embedding model_layers_21_self_attn_k_proj_weight_qr -> t4319_c
node matmul_q4_simd_150 micro s.matmul_q4_simd_2 t4311_embedding model_layers_21_self_attn_v_proj_weight_qr -> t4316_y
node rope_170_side micro s.side t4318_c position t4319_c position -> t4318_c t4319_c
node kv_scatter_170_side micro s.side_2 t4319_c position s782 t4316_y position s783 -> s782 s783
node attention_21 micro s.attention t4318_c s782 s783 position -> t4316_y
node matmul_q4_simd_151 micro s.matmul_q4_simd_2 t4316_y model_layers_21_self_attn_o_proj_weight_qr -> t4318_c
node add_rmsnorm_42 micro s.add_rmsnorm t4317_c t4318_c model_layers_21_post_attention_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_152 micro s.matmul_q4_simd_3 t4319_c model_layers_21_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_153 micro s.matmul_q4_simd_3 t4319_c model_layers_21_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_24 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_154 micro s.matmul_q4_simd_4 t4333_c model_layers_21_mlp_down_proj_weight_qr -> t4319_c
node add_rmsnorm_43 micro s.add_rmsnorm t4316_y t4319_c model_layers_22_input_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_155 micro s.matmul_q4_simd_2 t4317_c model_layers_22_self_attn_q_proj_weight_qr -> t4319_c
node matmul_q4_simd_156 micro s.matmul_q4_simd_2 t4317_c model_layers_22_self_attn_k_proj_weight_qr -> t4316_y
node matmul_q4_simd_157 micro s.matmul_q4_simd_2 t4317_c model_layers_22_self_attn_v_proj_weight_qr -> t4311_embedding
node rope_172_side micro s.side t4319_c position t4316_y position -> t4319_c t4316_y
node kv_scatter_172_side micro s.side_2 t4316_y position s784 t4311_embedding position s785 -> s784 s785
node attention_22 micro s.attention t4319_c s784 s785 position -> t4311_embedding
node matmul_q4_simd_158 micro s.matmul_q4_simd_2 t4311_embedding model_layers_22_self_attn_o_proj_weight_qr -> t4319_c
node add_rmsnorm_44 micro s.add_rmsnorm t4318_c t4319_c model_layers_22_post_attention_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_159 micro s.matmul_q4_simd_3 t4316_y model_layers_22_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_160 micro s.matmul_q4_simd_3 t4316_y model_layers_22_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_25 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_161 micro s.matmul_q4_simd_4 t4340_c model_layers_22_mlp_down_proj_weight_qr -> t4316_y
node add_rmsnorm_45 micro s.add_rmsnorm t4311_embedding t4316_y model_layers_23_input_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_162 micro s.matmul_q4_simd_2 t4318_c model_layers_23_self_attn_q_proj_weight_qr -> t4316_y
node matmul_q4_simd_163 micro s.matmul_q4_simd_2 t4318_c model_layers_23_self_attn_k_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_164 micro s.matmul_q4_simd_2 t4318_c model_layers_23_self_attn_v_proj_weight_qr -> t4317_c
node rope_174_side micro s.side t4316_y position t4311_embedding position -> t4316_y t4311_embedding
node kv_scatter_174_side micro s.side_2 t4311_embedding position s786 t4317_c position s787 -> s786 s787
node attention_23 micro s.attention t4316_y s786 s787 position -> t4317_c
node matmul_q4_simd_165 micro s.matmul_q4_simd_2 t4317_c model_layers_23_self_attn_o_proj_weight_qr -> t4316_y
node add_rmsnorm_46 micro s.add_rmsnorm t4319_c t4316_y model_layers_23_post_attention_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_166 micro s.matmul_q4_simd_3 t4311_embedding model_layers_23_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_167 micro s.matmul_q4_simd_3 t4311_embedding model_layers_23_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_26 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_168 micro s.matmul_q4_simd_4 t4333_c model_layers_23_mlp_down_proj_weight_qr -> t4311_embedding
node add_rmsnorm_47 micro s.add_rmsnorm t4317_c t4311_embedding model_layers_24_input_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_169 micro s.matmul_q4_simd_2 t4319_c model_layers_24_self_attn_q_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_170 micro s.matmul_q4_simd_2 t4319_c model_layers_24_self_attn_k_proj_weight_qr -> t4317_c
node matmul_q4_simd_171 micro s.matmul_q4_simd_2 t4319_c model_layers_24_self_attn_v_proj_weight_qr -> t4318_c
node rope_176_side micro s.side t4311_embedding position t4317_c position -> t4311_embedding t4317_c
node kv_scatter_176_side micro s.side_2 t4317_c position s788 t4318_c position s789 -> s788 s789
node attention_24 micro s.attention t4311_embedding s788 s789 position -> t4318_c
node matmul_q4_simd_172 micro s.matmul_q4_simd_2 t4318_c model_layers_24_self_attn_o_proj_weight_qr -> t4311_embedding
node add_rmsnorm_48 micro s.add_rmsnorm t4316_y t4311_embedding model_layers_24_post_attention_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_173 micro s.matmul_q4_simd_3 t4317_c model_layers_24_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_174 micro s.matmul_q4_simd_3 t4317_c model_layers_24_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_27 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_175 micro s.matmul_q4_simd_4 t4340_c model_layers_24_mlp_down_proj_weight_qr -> t4317_c
node add_rmsnorm_49 micro s.add_rmsnorm t4318_c t4317_c model_layers_25_input_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_176 micro s.matmul_q4_simd_2 t4316_y model_layers_25_self_attn_q_proj_weight_qr -> t4317_c
node matmul_q4_simd_177 micro s.matmul_q4_simd_2 t4316_y model_layers_25_self_attn_k_proj_weight_qr -> t4318_c
node matmul_q4_simd_178 micro s.matmul_q4_simd_2 t4316_y model_layers_25_self_attn_v_proj_weight_qr -> t4319_c
node rope_178_side micro s.side t4317_c position t4318_c position -> t4317_c t4318_c
node kv_scatter_178_side micro s.side_2 t4318_c position s790 t4319_c position s791 -> s790 s791
node attention_25 micro s.attention t4317_c s790 s791 position -> t4319_c
node matmul_q4_simd_179 micro s.matmul_q4_simd_2 t4319_c model_layers_25_self_attn_o_proj_weight_qr -> t4317_c
node add_rmsnorm_50 micro s.add_rmsnorm t4311_embedding t4317_c model_layers_25_post_attention_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_180 micro s.matmul_q4_simd_3 t4318_c model_layers_25_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_181 micro s.matmul_q4_simd_3 t4318_c model_layers_25_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_28 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_182 micro s.matmul_q4_simd_4 t4333_c model_layers_25_mlp_down_proj_weight_qr -> t4318_c
node add_rmsnorm_51 micro s.add_rmsnorm t4319_c t4318_c model_layers_26_input_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_183 micro s.matmul_q4_simd_2 t4311_embedding model_layers_26_self_attn_q_proj_weight_qr -> t4318_c
node matmul_q4_simd_184 micro s.matmul_q4_simd_2 t4311_embedding model_layers_26_self_attn_k_proj_weight_qr -> t4319_c
node matmul_q4_simd_185 micro s.matmul_q4_simd_2 t4311_embedding model_layers_26_self_attn_v_proj_weight_qr -> t4316_y
node rope_180_side micro s.side t4318_c position t4319_c position -> t4318_c t4319_c
node kv_scatter_180_side micro s.side_2 t4319_c position s792 t4316_y position s793 -> s792 s793
node attention_26 micro s.attention t4318_c s792 s793 position -> t4316_y
node matmul_q4_simd_186 micro s.matmul_q4_simd_2 t4316_y model_layers_26_self_attn_o_proj_weight_qr -> t4318_c
node add_rmsnorm_52 micro s.add_rmsnorm t4317_c t4318_c model_layers_26_post_attention_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_187 micro s.matmul_q4_simd_3 t4319_c model_layers_26_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_188 micro s.matmul_q4_simd_3 t4319_c model_layers_26_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_29 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_189 micro s.matmul_q4_simd_4 t4340_c model_layers_26_mlp_down_proj_weight_qr -> t4319_c
node add_rmsnorm_53 micro s.add_rmsnorm t4316_y t4319_c model_layers_27_input_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_190 micro s.matmul_q4_simd_2 t4317_c model_layers_27_self_attn_q_proj_weight_qr -> t4319_c
node matmul_q4_simd_191 micro s.matmul_q4_simd_2 t4317_c model_layers_27_self_attn_k_proj_weight_qr -> t4316_y
node matmul_q4_simd_192 micro s.matmul_q4_simd_2 t4317_c model_layers_27_self_attn_v_proj_weight_qr -> t4311_embedding
node rope_182_side micro s.side t4319_c position t4316_y position -> t4319_c t4316_y
node kv_scatter_182_side micro s.side_2 t4316_y position s794 t4311_embedding position s795 -> s794 s795
node attention_27 micro s.attention t4319_c s794 s795 position -> t4311_embedding
node matmul_q4_simd_193 micro s.matmul_q4_simd_2 t4311_embedding model_layers_27_self_attn_o_proj_weight_qr -> t4319_c
node add_rmsnorm_54 micro s.add_rmsnorm t4318_c t4319_c model_layers_27_post_attention_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_194 micro s.matmul_q4_simd_3 t4316_y model_layers_27_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_195 micro s.matmul_q4_simd_3 t4316_y model_layers_27_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_30 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_196 micro s.matmul_q4_simd_4 t4333_c model_layers_27_mlp_down_proj_weight_qr -> t4316_y
node add_rmsnorm_55 micro s.add_rmsnorm t4311_embedding t4316_y model_layers_28_input_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_197 micro s.matmul_q4_simd_2 t4318_c model_layers_28_self_attn_q_proj_weight_qr -> t4316_y
node matmul_q4_simd_198 micro s.matmul_q4_simd_2 t4318_c model_layers_28_self_attn_k_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_199 micro s.matmul_q4_simd_2 t4318_c model_layers_28_self_attn_v_proj_weight_qr -> t4317_c
node rope_184_side micro s.side t4316_y position t4311_embedding position -> t4316_y t4311_embedding
node kv_scatter_184_side micro s.side_2 t4311_embedding position s796 t4317_c position s797 -> s796 s797
node attention_28 micro s.attention t4316_y s796 s797 position -> t4317_c
node matmul_q4_simd_200 micro s.matmul_q4_simd_2 t4317_c model_layers_28_self_attn_o_proj_weight_qr -> t4316_y
node add_rmsnorm_56 micro s.add_rmsnorm t4319_c t4316_y model_layers_28_post_attention_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_201 micro s.matmul_q4_simd_3 t4311_embedding model_layers_28_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_202 micro s.matmul_q4_simd_3 t4311_embedding model_layers_28_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_31 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_203 micro s.matmul_q4_simd_4 t4340_c model_layers_28_mlp_down_proj_weight_qr -> t4311_embedding
node add_rmsnorm_57 micro s.add_rmsnorm t4317_c t4311_embedding model_layers_29_input_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_204 micro s.matmul_q4_simd_2 t4319_c model_layers_29_self_attn_q_proj_weight_qr -> t4311_embedding
node matmul_q4_simd_205 micro s.matmul_q4_simd_2 t4319_c model_layers_29_self_attn_k_proj_weight_qr -> t4317_c
node matmul_q4_simd_206 micro s.matmul_q4_simd_2 t4319_c model_layers_29_self_attn_v_proj_weight_qr -> t4318_c
node rope_186_side micro s.side t4311_embedding position t4317_c position -> t4311_embedding t4317_c
node kv_scatter_186_side micro s.side_2 t4317_c position s798 t4318_c position s799 -> s798 s799
node attention_29 micro s.attention t4311_embedding s798 s799 position -> t4318_c
node matmul_q4_simd_207 micro s.matmul_q4_simd_2 t4318_c model_layers_29_self_attn_o_proj_weight_qr -> t4311_embedding
node add_rmsnorm_58 micro s.add_rmsnorm t4316_y t4311_embedding model_layers_29_post_attention_layernorm_weight -> t4318_c t4317_c
node matmul_q4_simd_208 micro s.matmul_q4_simd_3 t4317_c model_layers_29_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_209 micro s.matmul_q4_simd_3 t4317_c model_layers_29_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_32 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_210 micro s.matmul_q4_simd_4 t4333_c model_layers_29_mlp_down_proj_weight_qr -> t4317_c
node add_rmsnorm_59 micro s.add_rmsnorm t4318_c t4317_c model_layers_30_input_layernorm_weight -> t4311_embedding t4316_y
node matmul_q4_simd_211 micro s.matmul_q4_simd_2 t4316_y model_layers_30_self_attn_q_proj_weight_qr -> t4317_c
node matmul_q4_simd_212 micro s.matmul_q4_simd_2 t4316_y model_layers_30_self_attn_k_proj_weight_qr -> t4318_c
node matmul_q4_simd_213 micro s.matmul_q4_simd_2 t4316_y model_layers_30_self_attn_v_proj_weight_qr -> t4319_c
node rope_188_side micro s.side t4317_c position t4318_c position -> t4317_c t4318_c
node kv_scatter_188_side micro s.side_2 t4318_c position s800 t4319_c position s801 -> s800 s801
node attention_30 micro s.attention t4317_c s800 s801 position -> t4319_c
node matmul_q4_simd_214 micro s.matmul_q4_simd_2 t4319_c model_layers_30_self_attn_o_proj_weight_qr -> t4317_c
node add_rmsnorm_60 micro s.add_rmsnorm t4311_embedding t4317_c model_layers_30_post_attention_layernorm_weight -> t4319_c t4318_c
node matmul_q4_simd_215 micro s.matmul_q4_simd_3 t4318_c model_layers_30_mlp_gate_proj_weight_qr -> t4333_c
node matmul_q4_simd_216 micro s.matmul_q4_simd_3 t4318_c model_layers_30_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_33 micro s.silu_mul_3 t4333_c t4334_c -> t4340_c
node matmul_q4_simd_217 micro s.matmul_q4_simd_4 t4340_c model_layers_30_mlp_down_proj_weight_qr -> t4318_c
node add_rmsnorm_61 micro s.add_rmsnorm t4319_c t4318_c model_layers_31_input_layernorm_weight -> t4317_c t4311_embedding
node matmul_q4_simd_218 micro s.matmul_q4_simd_2 t4311_embedding model_layers_31_self_attn_q_proj_weight_qr -> t4318_c
node matmul_q4_simd_219 micro s.matmul_q4_simd_2 t4311_embedding model_layers_31_self_attn_k_proj_weight_qr -> t4319_c
node matmul_q4_simd_220 micro s.matmul_q4_simd_2 t4311_embedding model_layers_31_self_attn_v_proj_weight_qr -> t4316_y
node rope_190_side micro s.side t4318_c position t4319_c position -> t4318_c t4319_c
node kv_scatter_190_side micro s.side_2 t4319_c position s802 t4316_y position s803 -> s802 s803
node attention_31 micro s.attention t4318_c s802 s803 position -> t4316_y
node matmul_q4_simd_221 micro s.matmul_q4_simd_2 t4316_y model_layers_31_self_attn_o_proj_weight_qr -> t4318_c
node add_rmsnorm_62 micro s.add_rmsnorm t4317_c t4318_c model_layers_31_post_attention_layernorm_weight -> t4316_y t4319_c
node matmul_q4_simd_222 micro s.matmul_q4_simd_3 t4319_c model_layers_31_mlp_gate_proj_weight_qr -> t4340_c
node matmul_q4_simd_223 micro s.matmul_q4_simd_3 t4319_c model_layers_31_mlp_up_proj_weight_qr -> t4334_c
node silu_mul_34 micro s.silu_mul_3 t4340_c t4334_c -> t4333_c
node matmul_q4_simd_224 micro s.matmul_q4_simd_4 t4333_c model_layers_31_mlp_down_proj_weight_qr -> t4319_c
node add_rmsnorm_63 micro s.add_rmsnorm t4316_y t4319_c model_norm_weight -> t4318_c t4317_c
node matmul_q4_simd_225 micro s.matmul_q4_simd t4317_c lm_head_weight_qr -> t5403_c
