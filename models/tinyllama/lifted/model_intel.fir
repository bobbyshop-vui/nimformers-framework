fir 1
model tinyllama
config config.json
runtime graph
layout script
script_abi 1
epsilon rms_norm_eps
require feature weights_bf16
sym hidden = hidden_size
sym layers = num_hidden_layers
sym positions = max_position_embeddings
sym vocab = vocab_size
buf token f32 1 input:token
buf t718_embedding f32 131072
buf t722_y f32 131072
buf t724_c f32 131072
buf t726_c f32 16384
buf t728_c f32 16384
buf position f32 1 input:position
buf s510 f32 524288
buf s511 f32 524288
buf t742_c f32 360448
buf t744_c f32 360448
buf t746_y f32 360448
buf t748_c f32 360448
buf s512 f32 524288
buf s513 f32 524288
buf s514 f32 524288
buf s515 f32 524288
buf s516 f32 524288
buf s517 f32 524288
buf s518 f32 524288
buf s519 f32 524288
buf s520 f32 524288
buf s521 f32 524288
buf s522 f32 524288
buf s523 f32 524288
buf s524 f32 524288
buf s525 f32 524288
buf s526 f32 524288
buf s527 f32 524288
buf s528 f32 524288
buf s529 f32 524288
buf s530 f32 524288
buf s531 f32 524288
buf s532 f32 524288
buf s533 f32 524288
buf s534 f32 524288
buf s535 f32 524288
buf s536 f32 524288
buf s537 f32 524288
buf s538 f32 524288
buf s539 f32 524288
buf s540 f32 524288
buf s541 f32 524288
buf s542 f32 524288
buf s543 f32 524288
buf s544 f32 524288
buf s545 f32 524288
buf s546 f32 524288
buf s547 f32 524288
buf s548 f32 524288
buf s549 f32 524288
buf s550 f32 524288
buf s551 f32 524288
buf s552 f32 524288
buf s553 f32 524288
buf s694 f32 2048
buf t1818_embedding f32 917504
buf t1822_y f32 917504
buf t1824_c f32 917504
buf t1826_c f32 114688
buf t1828_c f32 114688
buf t1842_c f32 2523136
buf t1844_c f32 2523136
buf t1850_c f32 2523136
buf t1899_c f32 2523136
buf t2930_y f32 2048
buf t2931_c f32 32000
buf t2957_embedding f32 2048
buf t2962_y f32 2048
buf t2963_c f32 2048
buf t2964_c f32 256
buf t2965_c f32 256
buf t2978_y f32 2048
buf t2979_c f32 5632
buf t2980_c f32 5632
buf t2986_c f32 5632
buf t3709_c f32 32000 output:logits
weight model_embed_tokens_weight_emb f32 model.embed_tokens.weight tensor [32000,2048]
weight model_layers_0_input_layernorm_weight f32 model.layers.0.input_layernorm.weight tensor [2048]
weight model_layers_0_self_attn_q_proj_weight_q q4b128 model.layers.0.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_0_self_attn_k_proj_weight_q q4b128 model.layers.0.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_0_self_attn_v_proj_weight_q q4b128 model.layers.0.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_0_self_attn_o_proj_weight_q q4b128 model.layers.0.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_0_post_attention_layernorm_weight f32 model.layers.0.post_attention_layernorm.weight tensor [2048]
weight model_layers_0_mlp_gate_proj_weight_q q4b128 model.layers.0.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_0_mlp_up_proj_weight_q q4b128 model.layers.0.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_0_mlp_down_proj_weight_q q4b128 model.layers.0.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_1_input_layernorm_weight f32 model.layers.1.input_layernorm.weight tensor [2048]
weight model_layers_1_self_attn_q_proj_weight_q q4b128 model.layers.1.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_1_self_attn_k_proj_weight_q q4b128 model.layers.1.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_1_self_attn_v_proj_weight_q q4b128 model.layers.1.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_1_self_attn_o_proj_weight_q q4b128 model.layers.1.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_1_post_attention_layernorm_weight f32 model.layers.1.post_attention_layernorm.weight tensor [2048]
weight model_layers_1_mlp_gate_proj_weight_q q4b128 model.layers.1.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_1_mlp_up_proj_weight_q q4b128 model.layers.1.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_1_mlp_down_proj_weight_q q4b128 model.layers.1.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_2_input_layernorm_weight f32 model.layers.2.input_layernorm.weight tensor [2048]
weight model_layers_2_self_attn_q_proj_weight_q q4b128 model.layers.2.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_2_self_attn_k_proj_weight_q q4b128 model.layers.2.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_2_self_attn_v_proj_weight_q q4b128 model.layers.2.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_2_self_attn_o_proj_weight_q q4b128 model.layers.2.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_2_post_attention_layernorm_weight f32 model.layers.2.post_attention_layernorm.weight tensor [2048]
weight model_layers_2_mlp_gate_proj_weight_q q4b128 model.layers.2.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_2_mlp_up_proj_weight_q q4b128 model.layers.2.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_2_mlp_down_proj_weight_q q4b128 model.layers.2.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_3_input_layernorm_weight f32 model.layers.3.input_layernorm.weight tensor [2048]
weight model_layers_3_self_attn_q_proj_weight_q q4b128 model.layers.3.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_3_self_attn_k_proj_weight_q q4b128 model.layers.3.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_3_self_attn_v_proj_weight_q q4b128 model.layers.3.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_3_self_attn_o_proj_weight_q q4b128 model.layers.3.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_3_post_attention_layernorm_weight f32 model.layers.3.post_attention_layernorm.weight tensor [2048]
weight model_layers_3_mlp_gate_proj_weight_q q4b128 model.layers.3.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_3_mlp_up_proj_weight_q q4b128 model.layers.3.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_3_mlp_down_proj_weight_q q4b128 model.layers.3.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_4_input_layernorm_weight f32 model.layers.4.input_layernorm.weight tensor [2048]
weight model_layers_4_self_attn_q_proj_weight_q q4b128 model.layers.4.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_4_self_attn_k_proj_weight_q q4b128 model.layers.4.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_4_self_attn_v_proj_weight_q q4b128 model.layers.4.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_4_self_attn_o_proj_weight_q q4b128 model.layers.4.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_4_post_attention_layernorm_weight f32 model.layers.4.post_attention_layernorm.weight tensor [2048]
weight model_layers_4_mlp_gate_proj_weight_q q4b128 model.layers.4.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_4_mlp_up_proj_weight_q q4b128 model.layers.4.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_4_mlp_down_proj_weight_q q4b128 model.layers.4.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_5_input_layernorm_weight f32 model.layers.5.input_layernorm.weight tensor [2048]
weight model_layers_5_self_attn_q_proj_weight_q q4b128 model.layers.5.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_5_self_attn_k_proj_weight_q q4b128 model.layers.5.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_5_self_attn_v_proj_weight_q q4b128 model.layers.5.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_5_self_attn_o_proj_weight_q q4b128 model.layers.5.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_5_post_attention_layernorm_weight f32 model.layers.5.post_attention_layernorm.weight tensor [2048]
weight model_layers_5_mlp_gate_proj_weight_q q4b128 model.layers.5.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_5_mlp_up_proj_weight_q q4b128 model.layers.5.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_5_mlp_down_proj_weight_q q4b128 model.layers.5.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_6_input_layernorm_weight f32 model.layers.6.input_layernorm.weight tensor [2048]
weight model_layers_6_self_attn_q_proj_weight_q q4b128 model.layers.6.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_6_self_attn_k_proj_weight_q q4b128 model.layers.6.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_6_self_attn_v_proj_weight_q q4b128 model.layers.6.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_6_self_attn_o_proj_weight_q q4b128 model.layers.6.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_6_post_attention_layernorm_weight f32 model.layers.6.post_attention_layernorm.weight tensor [2048]
weight model_layers_6_mlp_gate_proj_weight_q q4b128 model.layers.6.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_6_mlp_up_proj_weight_q q4b128 model.layers.6.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_6_mlp_down_proj_weight_q q4b128 model.layers.6.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_7_input_layernorm_weight f32 model.layers.7.input_layernorm.weight tensor [2048]
weight model_layers_7_self_attn_q_proj_weight_q q4b128 model.layers.7.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_7_self_attn_k_proj_weight_q q4b128 model.layers.7.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_7_self_attn_v_proj_weight_q q4b128 model.layers.7.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_7_self_attn_o_proj_weight_q q4b128 model.layers.7.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_7_post_attention_layernorm_weight f32 model.layers.7.post_attention_layernorm.weight tensor [2048]
weight model_layers_7_mlp_gate_proj_weight_q q4b128 model.layers.7.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_7_mlp_up_proj_weight_q q4b128 model.layers.7.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_7_mlp_down_proj_weight_q q4b128 model.layers.7.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_8_input_layernorm_weight f32 model.layers.8.input_layernorm.weight tensor [2048]
weight model_layers_8_self_attn_q_proj_weight_q q4b128 model.layers.8.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_8_self_attn_k_proj_weight_q q4b128 model.layers.8.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_8_self_attn_v_proj_weight_q q4b128 model.layers.8.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_8_self_attn_o_proj_weight_q q4b128 model.layers.8.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_8_post_attention_layernorm_weight f32 model.layers.8.post_attention_layernorm.weight tensor [2048]
weight model_layers_8_mlp_gate_proj_weight_q q4b128 model.layers.8.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_8_mlp_up_proj_weight_q q4b128 model.layers.8.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_8_mlp_down_proj_weight_q q4b128 model.layers.8.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_9_input_layernorm_weight f32 model.layers.9.input_layernorm.weight tensor [2048]
weight model_layers_9_self_attn_q_proj_weight_q q4b128 model.layers.9.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_9_self_attn_k_proj_weight_q q4b128 model.layers.9.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_9_self_attn_v_proj_weight_q q4b128 model.layers.9.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_9_self_attn_o_proj_weight_q q4b128 model.layers.9.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_9_post_attention_layernorm_weight f32 model.layers.9.post_attention_layernorm.weight tensor [2048]
weight model_layers_9_mlp_gate_proj_weight_q q4b128 model.layers.9.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_9_mlp_up_proj_weight_q q4b128 model.layers.9.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_9_mlp_down_proj_weight_q q4b128 model.layers.9.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_10_input_layernorm_weight f32 model.layers.10.input_layernorm.weight tensor [2048]
weight model_layers_10_self_attn_q_proj_weight_q q4b128 model.layers.10.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_10_self_attn_k_proj_weight_q q4b128 model.layers.10.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_10_self_attn_v_proj_weight_q q4b128 model.layers.10.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_10_self_attn_o_proj_weight_q q4b128 model.layers.10.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_10_post_attention_layernorm_weight f32 model.layers.10.post_attention_layernorm.weight tensor [2048]
weight model_layers_10_mlp_gate_proj_weight_q q4b128 model.layers.10.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_10_mlp_up_proj_weight_q q4b128 model.layers.10.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_10_mlp_down_proj_weight_q q4b128 model.layers.10.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_11_input_layernorm_weight f32 model.layers.11.input_layernorm.weight tensor [2048]
weight model_layers_11_self_attn_q_proj_weight_q q4b128 model.layers.11.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_11_self_attn_k_proj_weight_q q4b128 model.layers.11.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_11_self_attn_v_proj_weight_q q4b128 model.layers.11.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_11_self_attn_o_proj_weight_q q4b128 model.layers.11.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_11_post_attention_layernorm_weight f32 model.layers.11.post_attention_layernorm.weight tensor [2048]
weight model_layers_11_mlp_gate_proj_weight_q q4b128 model.layers.11.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_11_mlp_up_proj_weight_q q4b128 model.layers.11.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_11_mlp_down_proj_weight_q q4b128 model.layers.11.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_12_input_layernorm_weight f32 model.layers.12.input_layernorm.weight tensor [2048]
weight model_layers_12_self_attn_q_proj_weight_q q4b128 model.layers.12.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_12_self_attn_k_proj_weight_q q4b128 model.layers.12.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_12_self_attn_v_proj_weight_q q4b128 model.layers.12.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_12_self_attn_o_proj_weight_q q4b128 model.layers.12.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_12_post_attention_layernorm_weight f32 model.layers.12.post_attention_layernorm.weight tensor [2048]
weight model_layers_12_mlp_gate_proj_weight_q q4b128 model.layers.12.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_12_mlp_up_proj_weight_q q4b128 model.layers.12.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_12_mlp_down_proj_weight_q q4b128 model.layers.12.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_13_input_layernorm_weight f32 model.layers.13.input_layernorm.weight tensor [2048]
weight model_layers_13_self_attn_q_proj_weight_q q4b128 model.layers.13.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_13_self_attn_k_proj_weight_q q4b128 model.layers.13.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_13_self_attn_v_proj_weight_q q4b128 model.layers.13.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_13_self_attn_o_proj_weight_q q4b128 model.layers.13.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_13_post_attention_layernorm_weight f32 model.layers.13.post_attention_layernorm.weight tensor [2048]
weight model_layers_13_mlp_gate_proj_weight_q q4b128 model.layers.13.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_13_mlp_up_proj_weight_q q4b128 model.layers.13.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_13_mlp_down_proj_weight_q q4b128 model.layers.13.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_14_input_layernorm_weight f32 model.layers.14.input_layernorm.weight tensor [2048]
weight model_layers_14_self_attn_q_proj_weight_q q4b128 model.layers.14.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_14_self_attn_k_proj_weight_q q4b128 model.layers.14.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_14_self_attn_v_proj_weight_q q4b128 model.layers.14.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_14_self_attn_o_proj_weight_q q4b128 model.layers.14.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_14_post_attention_layernorm_weight f32 model.layers.14.post_attention_layernorm.weight tensor [2048]
weight model_layers_14_mlp_gate_proj_weight_q q4b128 model.layers.14.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_14_mlp_up_proj_weight_q q4b128 model.layers.14.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_14_mlp_down_proj_weight_q q4b128 model.layers.14.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_15_input_layernorm_weight f32 model.layers.15.input_layernorm.weight tensor [2048]
weight model_layers_15_self_attn_q_proj_weight_q q4b128 model.layers.15.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_15_self_attn_k_proj_weight_q q4b128 model.layers.15.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_15_self_attn_v_proj_weight_q q4b128 model.layers.15.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_15_self_attn_o_proj_weight_q q4b128 model.layers.15.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_15_post_attention_layernorm_weight f32 model.layers.15.post_attention_layernorm.weight tensor [2048]
weight model_layers_15_mlp_gate_proj_weight_q q4b128 model.layers.15.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_15_mlp_up_proj_weight_q q4b128 model.layers.15.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_15_mlp_down_proj_weight_q q4b128 model.layers.15.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_16_input_layernorm_weight f32 model.layers.16.input_layernorm.weight tensor [2048]
weight model_layers_16_self_attn_q_proj_weight_q q4b128 model.layers.16.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_16_self_attn_k_proj_weight_q q4b128 model.layers.16.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_16_self_attn_v_proj_weight_q q4b128 model.layers.16.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_16_self_attn_o_proj_weight_q q4b128 model.layers.16.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_16_post_attention_layernorm_weight f32 model.layers.16.post_attention_layernorm.weight tensor [2048]
weight model_layers_16_mlp_gate_proj_weight_q q4b128 model.layers.16.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_16_mlp_up_proj_weight_q q4b128 model.layers.16.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_16_mlp_down_proj_weight_q q4b128 model.layers.16.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_17_input_layernorm_weight f32 model.layers.17.input_layernorm.weight tensor [2048]
weight model_layers_17_self_attn_q_proj_weight_q q4b128 model.layers.17.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_17_self_attn_k_proj_weight_q q4b128 model.layers.17.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_17_self_attn_v_proj_weight_q q4b128 model.layers.17.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_17_self_attn_o_proj_weight_q q4b128 model.layers.17.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_17_post_attention_layernorm_weight f32 model.layers.17.post_attention_layernorm.weight tensor [2048]
weight model_layers_17_mlp_gate_proj_weight_q q4b128 model.layers.17.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_17_mlp_up_proj_weight_q q4b128 model.layers.17.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_17_mlp_down_proj_weight_q q4b128 model.layers.17.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_18_input_layernorm_weight f32 model.layers.18.input_layernorm.weight tensor [2048]
weight model_layers_18_self_attn_q_proj_weight_q q4b128 model.layers.18.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_18_self_attn_k_proj_weight_q q4b128 model.layers.18.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_18_self_attn_v_proj_weight_q q4b128 model.layers.18.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_18_self_attn_o_proj_weight_q q4b128 model.layers.18.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_18_post_attention_layernorm_weight f32 model.layers.18.post_attention_layernorm.weight tensor [2048]
weight model_layers_18_mlp_gate_proj_weight_q q4b128 model.layers.18.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_18_mlp_up_proj_weight_q q4b128 model.layers.18.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_18_mlp_down_proj_weight_q q4b128 model.layers.18.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_19_input_layernorm_weight f32 model.layers.19.input_layernorm.weight tensor [2048]
weight model_layers_19_self_attn_q_proj_weight_q q4b128 model.layers.19.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_19_self_attn_k_proj_weight_q q4b128 model.layers.19.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_19_self_attn_v_proj_weight_q q4b128 model.layers.19.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_19_self_attn_o_proj_weight_q q4b128 model.layers.19.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_19_post_attention_layernorm_weight f32 model.layers.19.post_attention_layernorm.weight tensor [2048]
weight model_layers_19_mlp_gate_proj_weight_q q4b128 model.layers.19.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_19_mlp_up_proj_weight_q q4b128 model.layers.19.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_19_mlp_down_proj_weight_q q4b128 model.layers.19.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_20_input_layernorm_weight f32 model.layers.20.input_layernorm.weight tensor [2048]
weight model_layers_20_self_attn_q_proj_weight_q q4b128 model.layers.20.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_20_self_attn_k_proj_weight_q q4b128 model.layers.20.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_20_self_attn_v_proj_weight_q q4b128 model.layers.20.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_20_self_attn_o_proj_weight_q q4b128 model.layers.20.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_20_post_attention_layernorm_weight f32 model.layers.20.post_attention_layernorm.weight tensor [2048]
weight model_layers_20_mlp_gate_proj_weight_q q4b128 model.layers.20.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_20_mlp_up_proj_weight_q q4b128 model.layers.20.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_20_mlp_down_proj_weight_q q4b128 model.layers.20.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_21_input_layernorm_weight f32 model.layers.21.input_layernorm.weight tensor [2048]
weight model_layers_21_self_attn_q_proj_weight_q q4b128 model.layers.21.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_21_self_attn_k_proj_weight_q q4b128 model.layers.21.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_21_self_attn_v_proj_weight_q q4b128 model.layers.21.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_21_self_attn_o_proj_weight_q q4b128 model.layers.21.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_21_post_attention_layernorm_weight f32 model.layers.21.post_attention_layernorm.weight tensor [2048]
weight model_layers_21_mlp_gate_proj_weight_q q4b128 model.layers.21.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_21_mlp_up_proj_weight_q q4b128 model.layers.21.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_21_mlp_down_proj_weight_q q4b128 model.layers.21.mlp.down_proj.weight tensor [2048,5632]
weight model_norm_weight f32 model.norm.weight tensor [2048]
weight lm_head_weight_q q4b128 lm_head.weight tensor [32000,2048]
kernel s.rmsnorm in=2 out=1 threads=16384 tg=256 tgbytes=1024 expect=in0:131072,in1:2048,out0:131072 rows=1:256 tag=aten::rms_norm
rtou 0 0
loadconstf32 4 0
loadconstf32 7 2048
utgid 1
ulid 2
loadlid 1
umulk 3 1 2048
uaddk 4 2 0
loadconstf32 8 8
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
mulconst 8 8 0.00048828125
addconst 8 8 0.00000999999975
rsqrt 8 8
uaddk 4 2 0
loadconstf32 10 8
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
kernel s.matmul_q4_tiled_fma in=2 out=1 threads=2048 tg=128 tgbytes=25600 expect=in0:131072,in1:589824,out0:131072 rows=64:2048 tag=aten::matmul
rtou 0 0
utgid 1
ulid 2
loadconstf32 65 16
utor 92 1
div 66 92 65
rtou 3 66
utor 66 3
mul 66 66 65
sub 92 92 66
rtou 4 92
umulk 5 4 128
umulk 6 3 64
uandk 7 2 15
ushrk 8 2 4
uconst 15 @in1
umulk 15 15 4
uconst 9 0
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 5 0
loadconstf32 6 0
loadconstf32 7 0
loadconstf32 8 0
loadconstf32 9 0
loadconstf32 10 0
loadconstf32 11 0
loadconstf32 12 0
loadconstf32 13 0
loadconstf32 14 0
loadconstf32 15 0
loadconstf32 16 0
loadconstf32 17 0
loadconstf32 18 0
loadconstf32 19 0
loadconstf32 20 0
loadconstf32 21 0
loadconstf32 22 0
loadconstf32 23 0
loadconstf32 24 0
loadconstf32 25 0
loadconstf32 26 0
loadconstf32 27 0
loadconstf32 28 0
loadconstf32 29 0
loadconstf32 30 0
loadconstf32 31 0
loadconstf32 32 0
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
loadconstf32 96 64
forbegin 96
uaddk 10 2 0
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 128
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 256
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 384
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 0
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+524288
uloadf32 95 10 @in1+557056
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 128
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+524288
uloadf32 95 10 @in1+557056
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 256
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+524288
uloadf32 95 10 @in1+557056
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 384
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+524288
uloadf32 95 10 @in1+557056
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
barriertg 
umulk 10 8 8
utor 89 10
umulk 10 7 8
uaddk 10 10 2176
utor 90 10
loadconstf32 97 32
forbegin 97
tgloadf32 73 90 0
tgloadf32 74 90 1
tgloadf32 75 90 2
tgloadf32 76 90 3
tgloadf32 77 90 4
tgloadf32 78 90 5
tgloadf32 79 90 6
tgloadf32 80 90 7
tgloadf32 81 89 0
tgloadf32 82 89 1
tgloadf32 83 89 2
tgloadf32 84 89 3
tgloadf32 85 89 4
tgloadf32 86 89 5
tgloadf32 87 89 6
tgloadf32 88 89 7
fma 1 81 73 1
fma 2 81 74 2
fma 3 81 75 3
fma 4 81 76 4
fma 5 81 77 5
fma 6 81 78 6
fma 7 81 79 7
fma 8 81 80 8
fma 9 82 73 9
fma 10 82 74 10
fma 11 82 75 11
fma 12 82 76 12
fma 13 82 77 13
fma 14 82 78 14
fma 15 82 79 15
fma 16 82 80 16
fma 17 83 73 17
fma 18 83 74 18
fma 19 83 75 19
fma 20 83 76 20
fma 21 83 77 21
fma 22 83 78 22
fma 23 83 79 23
fma 24 83 80 24
fma 25 84 73 25
fma 26 84 74 26
fma 27 84 75 27
fma 28 84 76 28
fma 29 84 77 29
fma 30 84 78 30
fma 31 84 79 31
fma 32 84 80 32
fma 33 85 73 33
fma 34 85 74 34
fma 35 85 75 35
fma 36 85 76 36
fma 37 85 77 37
fma 38 85 78 38
fma 39 85 79 39
fma 40 85 80 40
fma 41 86 73 41
fma 42 86 74 42
fma 43 86 75 43
fma 44 86 76 44
fma 45 86 77 45
fma 46 86 78 46
fma 47 86 79 47
fma 48 86 80 48
fma 49 87 73 49
fma 50 87 74 50
fma 51 87 75 51
fma 52 87 76 52
fma 53 87 77 53
fma 54 87 78 54
fma 55 87 79 55
fma 56 87 80 56
fma 57 88 73 57
fma 58 88 74 58
fma 59 88 75 59
fma 60 88 76 60
fma 61 88 77 61
fma 62 88 78 62
fma 63 88 79 63
fma 64 88 80 64
addconst 89 89 68
addconst 90 90 132
forend 
barriertg 
uaddk 9 9 32
forend 
umulk 10 8 8
uadd 10 10 6
umulk 11 7 8
uadd 11 11 5
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 1 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 2 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 3 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 4 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 5 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 6 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 7 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 8 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 9 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 10 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 11 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 12 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 13 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 14 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 15 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 16 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 17 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 18 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 19 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 20 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 21 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 22 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 23 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 24 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 25 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 26 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 27 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 28 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 29 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 30 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 31 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 32 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 33 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 34 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 35 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 36 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 37 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 38 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 39 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 40 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 41 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 42 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 43 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 44 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 45 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 46 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 47 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 48 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 49 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 50 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 51 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 52 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 53 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 54 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 55 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 56 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 57 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 58 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 59 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 60 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 61 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 62 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 63 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 64 12 @out0
ifend 
endkernel
kernel s.matmul_q4_tiled_fma_2 in=2 out=1 threads=256 tg=128 tgbytes=25600 expect=in0:131072,in1:73728,out0:16384 rows=64:256 tag=aten::matmul
rtou 0 0
utgid 1
ulid 2
loadconstf32 65 2
utor 92 1
div 66 92 65
rtou 3 66
utor 66 3
mul 66 66 65
sub 92 92 66
rtou 4 92
umulk 5 4 128
umulk 6 3 64
uandk 7 2 15
ushrk 8 2 4
uconst 15 @in1
umulk 15 15 4
uconst 9 0
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 5 0
loadconstf32 6 0
loadconstf32 7 0
loadconstf32 8 0
loadconstf32 9 0
loadconstf32 10 0
loadconstf32 11 0
loadconstf32 12 0
loadconstf32 13 0
loadconstf32 14 0
loadconstf32 15 0
loadconstf32 16 0
loadconstf32 17 0
loadconstf32 18 0
loadconstf32 19 0
loadconstf32 20 0
loadconstf32 21 0
loadconstf32 22 0
loadconstf32 23 0
loadconstf32 24 0
loadconstf32 25 0
loadconstf32 26 0
loadconstf32 27 0
loadconstf32 28 0
loadconstf32 29 0
loadconstf32 30 0
loadconstf32 31 0
loadconstf32 32 0
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
loadconstf32 96 64
forbegin 96
uaddk 10 2 0
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 128
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 256
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 384
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 0
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 256
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+65536
uloadf32 95 10 @in1+69632
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 128
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 256
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+65536
uloadf32 95 10 @in1+69632
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 256
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 256
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+65536
uloadf32 95 10 @in1+69632
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 384
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 256
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+65536
uloadf32 95 10 @in1+69632
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
barriertg 
umulk 10 8 8
utor 89 10
umulk 10 7 8
uaddk 10 10 2176
utor 90 10
loadconstf32 97 32
forbegin 97
tgloadf32 73 90 0
tgloadf32 74 90 1
tgloadf32 75 90 2
tgloadf32 76 90 3
tgloadf32 77 90 4
tgloadf32 78 90 5
tgloadf32 79 90 6
tgloadf32 80 90 7
tgloadf32 81 89 0
tgloadf32 82 89 1
tgloadf32 83 89 2
tgloadf32 84 89 3
tgloadf32 85 89 4
tgloadf32 86 89 5
tgloadf32 87 89 6
tgloadf32 88 89 7
fma 1 81 73 1
fma 2 81 74 2
fma 3 81 75 3
fma 4 81 76 4
fma 5 81 77 5
fma 6 81 78 6
fma 7 81 79 7
fma 8 81 80 8
fma 9 82 73 9
fma 10 82 74 10
fma 11 82 75 11
fma 12 82 76 12
fma 13 82 77 13
fma 14 82 78 14
fma 15 82 79 15
fma 16 82 80 16
fma 17 83 73 17
fma 18 83 74 18
fma 19 83 75 19
fma 20 83 76 20
fma 21 83 77 21
fma 22 83 78 22
fma 23 83 79 23
fma 24 83 80 24
fma 25 84 73 25
fma 26 84 74 26
fma 27 84 75 27
fma 28 84 76 28
fma 29 84 77 29
fma 30 84 78 30
fma 31 84 79 31
fma 32 84 80 32
fma 33 85 73 33
fma 34 85 74 34
fma 35 85 75 35
fma 36 85 76 36
fma 37 85 77 37
fma 38 85 78 38
fma 39 85 79 39
fma 40 85 80 40
fma 41 86 73 41
fma 42 86 74 42
fma 43 86 75 43
fma 44 86 76 44
fma 45 86 77 45
fma 46 86 78 46
fma 47 86 79 47
fma 48 86 80 48
fma 49 87 73 49
fma 50 87 74 50
fma 51 87 75 51
fma 52 87 76 52
fma 53 87 77 53
fma 54 87 78 54
fma 55 87 79 55
fma 56 87 80 56
fma 57 88 73 57
fma 58 88 74 58
fma 59 88 75 59
fma 60 88 76 60
fma 61 88 77 61
fma 62 88 78 62
fma 63 88 79 63
fma 64 88 80 64
addconst 89 89 68
addconst 90 90 132
forend 
barriertg 
uaddk 9 9 32
forend 
umulk 10 8 8
uadd 10 10 6
umulk 11 7 8
uadd 11 11 5
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 1 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 2 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 3 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 4 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 5 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 6 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 7 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 8 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 9 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 10 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 11 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 12 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 13 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 14 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 15 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 16 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 17 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 18 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 19 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 20 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 21 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 22 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 23 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 24 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 25 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 26 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 27 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 28 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 29 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 30 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 31 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 32 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 33 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 34 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 35 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 36 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 37 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 38 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 39 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 40 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 41 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 42 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 43 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 44 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 45 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 46 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 47 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 48 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 49 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 50 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 51 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 52 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 53 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 54 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 55 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 56 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 57 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 58 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 59 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 60 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 61 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 62 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 63 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 64 12 @out0
ifend 
endkernel
kernel s.rope in=2 out=1 threads=65536 tg=64 tgbytes=16 expect=in0:131072,in1:1,out0:131072 rows=1:1024
rtou 0 0
uconst 1 0
uandk 2 0 31
ushrk 3 0 5
umulk 5 3 64
uadd 5 5 2
uloadf32 1 1 @in1
utor 10 3
addconst 10 10 0.5
mulconst 10 10 0.03125
rtou 4 10
utor 9 4
add 1 1 9
utor 9 2
mulconst 9 9 -0.28782313662426
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
uloadf32 7 5 @in0+32
mul 8 6 5
mul 3 7 4
sub 8 8 3
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
ustoref32 8 5 @out0+32
endkernel
kernel s.rope_2 in=2 out=1 threads=8192 tg=64 tgbytes=16 expect=in0:16384,in1:1,out0:16384 rows=1:128
rtou 0 0
uconst 1 0
uandk 2 0 31
ushrk 3 0 5
umulk 5 3 64
uadd 5 5 2
uloadf32 1 1 @in1
utor 10 3
addconst 10 10 0.5
mulconst 10 10 0.25
rtou 4 10
utor 9 4
add 1 1 9
utor 9 2
mulconst 9 9 -0.28782313662426
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
uloadf32 7 5 @in0+32
mul 8 6 5
mul 3 7 4
sub 8 8 3
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
ustoref32 8 5 @out0+32
endkernel
kernel s.kv_scatter in=3 out=1 threads=16384 tg=64 tgbytes=16 expect=in0:16384,in1:1,in2:524288,out0:524288 rows=1:256
rtou 0 0
uconst 1 0
uloadf32 1 1 @in1
rtou 2 1
utor 3 0
addconst 3 3 0.5
mulconst 3 3 0.015625
rtou 3 3
utor 4 0
utor 5 3
mulconst 5 5 64
sub 4 4 5
rtou 6 4
utor 6 3
addconst 6 6 0.5
mulconst 6 6 0.25
rtou 4 6
utor 7 3
utor 8 4
mulconst 8 8 4
sub 7 7 8
rtou 5 7
uadd 7 2 4
umulk 5 5 2048
uadd 7 7 5
umulk 7 7 64
uadd 7 7 6
uloadf32 2 0 @in0
ustoref32 2 7 @in2
endkernel
kernel s.flash_prefill in=4 out=1 threads=16384 tg=128 tgbytes=13312 expect=in0:131072,in1:524288,in2:524288,in3:1,out0:131072 rows=16:4096 tag=aten::scaled_dot_product_attention
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
loadconstf32 1 8
div 2 2 1
rtou 11 2
umulk 8 11 131072
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
umulk 11 5 64
utor 20 11
add 24 20 9
addconst 25 24 1024
umulk 11 6 64
utor 21 11
addconst 21 21 1024
umulk 11 5 16
utor 26 11
addconst 26 26 3072
mulconst 27 9 16
addconst 27 27 2048
utor 28 5
add 28 28 27
umulk 9 7 32
uadd 9 9 4
umulk 9 9 64
uadd 9 9 6
loadconstf32 41 0
loadconstf32 42 0
loadconstf32 43 0
loadconstf32 44 0
loadconstf32 45 0
loadconstf32 46 0
loadconstf32 47 0
loadconstf32 48 0
ifbegin 4
uloadf32 41 9 @in0+0
uloadf32 42 9 @in0+8
uloadf32 43 9 @in0+16
uloadf32 44 9 @in0+24
uloadf32 45 9 @in0+32
uloadf32 46 9 @in0+40
uloadf32 47 9 @in0+48
uloadf32 48 9 @in0+56
ifend 
tgstoref32 41 24 0
tgstoref32 42 24 8
tgstoref32 43 24 16
tgstoref32 44 24 24
tgstoref32 45 24 32
tgstoref32 46 24 40
tgstoref32 47 24 48
tgstoref32 48 24 56
loadconstf32 33 0
loadconstf32 34 0
loadconstf32 35 0
loadconstf32 36 0
loadconstf32 37 0
loadconstf32 38 0
loadconstf32 39 0
loadconstf32 40 0
loadconstf32 10 -1000000000000000019884624838656.0
loadconstf32 11 0
uconst 10 0
loadconstf32 8 0
forbegin 7
barriertg 
uadd 11 10 5
utor 2 11
cmplt 3 2 6
umulk 11 11 64
uadd 11 11 8
uadd 11 11 6
loadconstf32 41 0
loadconstf32 42 0
loadconstf32 43 0
loadconstf32 44 0
loadconstf32 45 0
loadconstf32 46 0
loadconstf32 47 0
loadconstf32 48 0
ifbegin 3
uloadf32 41 11 @in1+0
uloadf32 42 11 @in1+8
uloadf32 43 11 @in1+16
uloadf32 44 11 @in1+24
uloadf32 45 11 @in1+32
uloadf32 46 11 @in1+40
uloadf32 47 11 @in1+48
uloadf32 48 11 @in1+56
ifend 
tgstoref32 41 25 0
tgstoref32 42 25 8
tgstoref32 43 25 16
tgstoref32 44 25 24
tgstoref32 45 25 32
tgstoref32 46 25 40
tgstoref32 47 25 48
tgstoref32 48 25 56
loadconstf32 41 0
loadconstf32 42 0
loadconstf32 43 0
loadconstf32 44 0
loadconstf32 45 0
loadconstf32 46 0
loadconstf32 47 0
loadconstf32 48 0
ifbegin 3
uloadf32 41 11 @in2+0
uloadf32 42 11 @in2+8
uloadf32 43 11 @in2+16
uloadf32 44 11 @in2+24
uloadf32 45 11 @in2+32
uloadf32 46 11 @in2+40
uloadf32 47 11 @in2+48
uloadf32 48 11 @in2+56
ifend 
tgstoref32 41 28 0
tgstoref32 42 28 128
tgstoref32 43 28 256
tgstoref32 44 28 384
tgstoref32 45 28 512
tgstoref32 46 28 640
tgstoref32 47 28 768
tgstoref32 48 28 896
barriertg 
loadconstf32 15 0
loadconstf32 16 0
mulconst 22 20 1
mulconst 23 21 1
loadconstf32 57 16
forbegin 57
tgloadf32 41 22 0
tgloadf32 42 22 1
tgloadf32 43 22 2
tgloadf32 44 22 3
tgloadf32 45 23 0
tgloadf32 46 23 1
tgloadf32 47 23 2
tgloadf32 48 23 3
tgloadf32 49 23 512
tgloadf32 50 23 513
tgloadf32 51 23 514
tgloadf32 52 23 515
fma 15 41 45 15
fma 16 41 49 16
fma 15 42 46 15
fma 16 42 50 16
fma 15 43 47 15
fma 16 43 51 16
fma 15 44 48 15
fma 16 44 52 16
addconst 22 22 4
addconst 23 23 4
forend 
add 2 8 9
loadconstf32 18 -1000000000000000019884624838656.0
cmplt 3 2 5
ifbegin 3
mulconst 18 15 0.125
ifend 
addconst 2 2 8
loadconstf32 19 -1000000000000000019884624838656.0
cmplt 3 2 5
ifbegin 3
mulconst 19 16 0.125
ifend 
add 2 26 9
tgstoref32 18 2 0
tgstoref32 19 2 8
barriertg 
tgloadf32 41 26 0
tgloadf32 42 26 1
tgloadf32 43 26 2
tgloadf32 44 26 3
tgloadf32 45 26 4
tgloadf32 46 26 5
tgloadf32 47 26 6
tgloadf32 48 26 7
tgloadf32 49 26 8
tgloadf32 50 26 9
tgloadf32 51 26 10
tgloadf32 52 26 11
tgloadf32 53 26 12
tgloadf32 54 26 13
tgloadf32 55 26 14
tgloadf32 56 26 15
mulconst 12 10 1
cmplt 3 12 41
ifbegin 3
mulconst 12 41 1
ifend 
cmplt 3 12 42
ifbegin 3
mulconst 12 42 1
ifend 
cmplt 3 12 43
ifbegin 3
mulconst 12 43 1
ifend 
cmplt 3 12 44
ifbegin 3
mulconst 12 44 1
ifend 
cmplt 3 12 45
ifbegin 3
mulconst 12 45 1
ifend 
cmplt 3 12 46
ifbegin 3
mulconst 12 46 1
ifend 
cmplt 3 12 47
ifbegin 3
mulconst 12 47 1
ifend 
cmplt 3 12 48
ifbegin 3
mulconst 12 48 1
ifend 
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
sub 13 10 12
exp 13 13
loadconstf32 14 0
loadconstf32 1 -500000000000000009942312419328.0
cmplt 3 1 41
sub 41 41 12
exp 41 41
mul 41 41 3
add 14 14 41
cmplt 3 1 42
sub 42 42 12
exp 42 42
mul 42 42 3
add 14 14 42
cmplt 3 1 43
sub 43 43 12
exp 43 43
mul 43 43 3
add 14 14 43
cmplt 3 1 44
sub 44 44 12
exp 44 44
mul 44 44 3
add 14 14 44
cmplt 3 1 45
sub 45 45 12
exp 45 45
mul 45 45 3
add 14 14 45
cmplt 3 1 46
sub 46 46 12
exp 46 46
mul 46 46 3
add 14 14 46
cmplt 3 1 47
sub 47 47 12
exp 47 47
mul 47 47 3
add 14 14 47
cmplt 3 1 48
sub 48 48 12
exp 48 48
mul 48 48 3
add 14 14 48
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
mul 11 11 13
add 11 11 14
mulconst 10 12 1
mul 33 33 13
tgloadf32 29 27 0
tgloadf32 30 27 1
tgloadf32 31 27 2
tgloadf32 32 27 3
fma 33 41 29 33
fma 33 42 30 33
fma 33 43 31 33
fma 33 44 32 33
tgloadf32 29 27 4
tgloadf32 30 27 5
tgloadf32 31 27 6
tgloadf32 32 27 7
fma 33 45 29 33
fma 33 46 30 33
fma 33 47 31 33
fma 33 48 32 33
tgloadf32 29 27 8
tgloadf32 30 27 9
tgloadf32 31 27 10
tgloadf32 32 27 11
fma 33 49 29 33
fma 33 50 30 33
fma 33 51 31 33
fma 33 52 32 33
tgloadf32 29 27 12
tgloadf32 30 27 13
tgloadf32 31 27 14
tgloadf32 32 27 15
fma 33 53 29 33
fma 33 54 30 33
fma 33 55 31 33
fma 33 56 32 33
mul 34 34 13
tgloadf32 29 27 128
tgloadf32 30 27 129
tgloadf32 31 27 130
tgloadf32 32 27 131
fma 34 41 29 34
fma 34 42 30 34
fma 34 43 31 34
fma 34 44 32 34
tgloadf32 29 27 132
tgloadf32 30 27 133
tgloadf32 31 27 134
tgloadf32 32 27 135
fma 34 45 29 34
fma 34 46 30 34
fma 34 47 31 34
fma 34 48 32 34
tgloadf32 29 27 136
tgloadf32 30 27 137
tgloadf32 31 27 138
tgloadf32 32 27 139
fma 34 49 29 34
fma 34 50 30 34
fma 34 51 31 34
fma 34 52 32 34
tgloadf32 29 27 140
tgloadf32 30 27 141
tgloadf32 31 27 142
tgloadf32 32 27 143
fma 34 53 29 34
fma 34 54 30 34
fma 34 55 31 34
fma 34 56 32 34
mul 35 35 13
tgloadf32 29 27 256
tgloadf32 30 27 257
tgloadf32 31 27 258
tgloadf32 32 27 259
fma 35 41 29 35
fma 35 42 30 35
fma 35 43 31 35
fma 35 44 32 35
tgloadf32 29 27 260
tgloadf32 30 27 261
tgloadf32 31 27 262
tgloadf32 32 27 263
fma 35 45 29 35
fma 35 46 30 35
fma 35 47 31 35
fma 35 48 32 35
tgloadf32 29 27 264
tgloadf32 30 27 265
tgloadf32 31 27 266
tgloadf32 32 27 267
fma 35 49 29 35
fma 35 50 30 35
fma 35 51 31 35
fma 35 52 32 35
tgloadf32 29 27 268
tgloadf32 30 27 269
tgloadf32 31 27 270
tgloadf32 32 27 271
fma 35 53 29 35
fma 35 54 30 35
fma 35 55 31 35
fma 35 56 32 35
mul 36 36 13
tgloadf32 29 27 384
tgloadf32 30 27 385
tgloadf32 31 27 386
tgloadf32 32 27 387
fma 36 41 29 36
fma 36 42 30 36
fma 36 43 31 36
fma 36 44 32 36
tgloadf32 29 27 388
tgloadf32 30 27 389
tgloadf32 31 27 390
tgloadf32 32 27 391
fma 36 45 29 36
fma 36 46 30 36
fma 36 47 31 36
fma 36 48 32 36
tgloadf32 29 27 392
tgloadf32 30 27 393
tgloadf32 31 27 394
tgloadf32 32 27 395
fma 36 49 29 36
fma 36 50 30 36
fma 36 51 31 36
fma 36 52 32 36
tgloadf32 29 27 396
tgloadf32 30 27 397
tgloadf32 31 27 398
tgloadf32 32 27 399
fma 36 53 29 36
fma 36 54 30 36
fma 36 55 31 36
fma 36 56 32 36
mul 37 37 13
tgloadf32 29 27 512
tgloadf32 30 27 513
tgloadf32 31 27 514
tgloadf32 32 27 515
fma 37 41 29 37
fma 37 42 30 37
fma 37 43 31 37
fma 37 44 32 37
tgloadf32 29 27 516
tgloadf32 30 27 517
tgloadf32 31 27 518
tgloadf32 32 27 519
fma 37 45 29 37
fma 37 46 30 37
fma 37 47 31 37
fma 37 48 32 37
tgloadf32 29 27 520
tgloadf32 30 27 521
tgloadf32 31 27 522
tgloadf32 32 27 523
fma 37 49 29 37
fma 37 50 30 37
fma 37 51 31 37
fma 37 52 32 37
tgloadf32 29 27 524
tgloadf32 30 27 525
tgloadf32 31 27 526
tgloadf32 32 27 527
fma 37 53 29 37
fma 37 54 30 37
fma 37 55 31 37
fma 37 56 32 37
mul 38 38 13
tgloadf32 29 27 640
tgloadf32 30 27 641
tgloadf32 31 27 642
tgloadf32 32 27 643
fma 38 41 29 38
fma 38 42 30 38
fma 38 43 31 38
fma 38 44 32 38
tgloadf32 29 27 644
tgloadf32 30 27 645
tgloadf32 31 27 646
tgloadf32 32 27 647
fma 38 45 29 38
fma 38 46 30 38
fma 38 47 31 38
fma 38 48 32 38
tgloadf32 29 27 648
tgloadf32 30 27 649
tgloadf32 31 27 650
tgloadf32 32 27 651
fma 38 49 29 38
fma 38 50 30 38
fma 38 51 31 38
fma 38 52 32 38
tgloadf32 29 27 652
tgloadf32 30 27 653
tgloadf32 31 27 654
tgloadf32 32 27 655
fma 38 53 29 38
fma 38 54 30 38
fma 38 55 31 38
fma 38 56 32 38
mul 39 39 13
tgloadf32 29 27 768
tgloadf32 30 27 769
tgloadf32 31 27 770
tgloadf32 32 27 771
fma 39 41 29 39
fma 39 42 30 39
fma 39 43 31 39
fma 39 44 32 39
tgloadf32 29 27 772
tgloadf32 30 27 773
tgloadf32 31 27 774
tgloadf32 32 27 775
fma 39 45 29 39
fma 39 46 30 39
fma 39 47 31 39
fma 39 48 32 39
tgloadf32 29 27 776
tgloadf32 30 27 777
tgloadf32 31 27 778
tgloadf32 32 27 779
fma 39 49 29 39
fma 39 50 30 39
fma 39 51 31 39
fma 39 52 32 39
tgloadf32 29 27 780
tgloadf32 30 27 781
tgloadf32 31 27 782
tgloadf32 32 27 783
fma 39 53 29 39
fma 39 54 30 39
fma 39 55 31 39
fma 39 56 32 39
mul 40 40 13
tgloadf32 29 27 896
tgloadf32 30 27 897
tgloadf32 31 27 898
tgloadf32 32 27 899
fma 40 41 29 40
fma 40 42 30 40
fma 40 43 31 40
fma 40 44 32 40
tgloadf32 29 27 900
tgloadf32 30 27 901
tgloadf32 31 27 902
tgloadf32 32 27 903
fma 40 45 29 40
fma 40 46 30 40
fma 40 47 31 40
fma 40 48 32 40
tgloadf32 29 27 904
tgloadf32 30 27 905
tgloadf32 31 27 906
tgloadf32 32 27 907
fma 40 49 29 40
fma 40 50 30 40
fma 40 51 31 40
fma 40 52 32 40
tgloadf32 29 27 908
tgloadf32 30 27 909
tgloadf32 31 27 910
tgloadf32 32 27 911
fma 40 53 29 40
fma 40 54 30 40
fma 40 55 31 40
fma 40 56 32 40
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
ifend 
endkernel
kernel s.vec_add in=2 out=1 threads=131072 tg=64 tgbytes=16 expect=in0:131072,in1:131072,out0:131072 rows=1:2048 tag=aten::add
rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
add 1 1 2
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_tiled_fma_3 in=2 out=1 threads=5632 tg=128 tgbytes=25600 expect=in0:131072,in1:1622016,out0:360448 rows=64:5632 tag=aten::matmul
rtou 0 0
utgid 1
ulid 2
loadconstf32 65 44
utor 92 1
div 66 92 65
rtou 3 66
utor 66 3
mul 66 66 65
sub 92 92 66
rtou 4 92
umulk 5 4 128
umulk 6 3 64
uandk 7 2 15
ushrk 8 2 4
uconst 15 @in1
umulk 15 15 4
uconst 9 0
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 5 0
loadconstf32 6 0
loadconstf32 7 0
loadconstf32 8 0
loadconstf32 9 0
loadconstf32 10 0
loadconstf32 11 0
loadconstf32 12 0
loadconstf32 13 0
loadconstf32 14 0
loadconstf32 15 0
loadconstf32 16 0
loadconstf32 17 0
loadconstf32 18 0
loadconstf32 19 0
loadconstf32 20 0
loadconstf32 21 0
loadconstf32 22 0
loadconstf32 23 0
loadconstf32 24 0
loadconstf32 25 0
loadconstf32 26 0
loadconstf32 27 0
loadconstf32 28 0
loadconstf32 29 0
loadconstf32 30 0
loadconstf32 31 0
loadconstf32 32 0
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
loadconstf32 96 64
forbegin 96
uaddk 10 2 0
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 128
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 256
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 384
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 0
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 5632
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 128
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 5632
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 256
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 5632
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 384
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 5632
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
barriertg 
umulk 10 8 8
utor 89 10
umulk 10 7 8
uaddk 10 10 2176
utor 90 10
loadconstf32 97 32
forbegin 97
tgloadf32 73 90 0
tgloadf32 74 90 1
tgloadf32 75 90 2
tgloadf32 76 90 3
tgloadf32 77 90 4
tgloadf32 78 90 5
tgloadf32 79 90 6
tgloadf32 80 90 7
tgloadf32 81 89 0
tgloadf32 82 89 1
tgloadf32 83 89 2
tgloadf32 84 89 3
tgloadf32 85 89 4
tgloadf32 86 89 5
tgloadf32 87 89 6
tgloadf32 88 89 7
fma 1 81 73 1
fma 2 81 74 2
fma 3 81 75 3
fma 4 81 76 4
fma 5 81 77 5
fma 6 81 78 6
fma 7 81 79 7
fma 8 81 80 8
fma 9 82 73 9
fma 10 82 74 10
fma 11 82 75 11
fma 12 82 76 12
fma 13 82 77 13
fma 14 82 78 14
fma 15 82 79 15
fma 16 82 80 16
fma 17 83 73 17
fma 18 83 74 18
fma 19 83 75 19
fma 20 83 76 20
fma 21 83 77 21
fma 22 83 78 22
fma 23 83 79 23
fma 24 83 80 24
fma 25 84 73 25
fma 26 84 74 26
fma 27 84 75 27
fma 28 84 76 28
fma 29 84 77 29
fma 30 84 78 30
fma 31 84 79 31
fma 32 84 80 32
fma 33 85 73 33
fma 34 85 74 34
fma 35 85 75 35
fma 36 85 76 36
fma 37 85 77 37
fma 38 85 78 38
fma 39 85 79 39
fma 40 85 80 40
fma 41 86 73 41
fma 42 86 74 42
fma 43 86 75 43
fma 44 86 76 44
fma 45 86 77 45
fma 46 86 78 46
fma 47 86 79 47
fma 48 86 80 48
fma 49 87 73 49
fma 50 87 74 50
fma 51 87 75 51
fma 52 87 76 52
fma 53 87 77 53
fma 54 87 78 54
fma 55 87 79 55
fma 56 87 80 56
fma 57 88 73 57
fma 58 88 74 58
fma 59 88 75 59
fma 60 88 76 60
fma 61 88 77 61
fma 62 88 78 62
fma 63 88 79 63
fma 64 88 80 64
addconst 89 89 68
addconst 90 90 132
forend 
barriertg 
uaddk 9 9 32
forend 
umulk 10 8 8
uadd 10 10 6
umulk 11 7 8
uadd 11 11 5
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 1 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 2 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 3 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 4 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 5 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 6 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 7 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 8 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 9 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 10 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 11 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 12 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 13 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 14 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 15 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 16 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 17 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 18 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 19 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 20 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 21 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 22 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 23 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 24 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 25 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 26 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 27 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 28 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 29 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 30 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 31 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 32 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 33 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 34 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 35 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 36 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 37 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 38 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 39 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 40 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 41 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 42 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 43 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 44 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 45 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 46 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 47 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 48 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 49 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 50 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 51 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 52 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 53 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 54 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 55 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 56 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 57 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 58 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 59 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 60 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 61 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 62 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 63 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 64 12 @out0
ifend 
endkernel
kernel s.act_sigmoid in=1 out=1 threads=360448 tg=64 tgbytes=16 expect=in0:360448,out0:360448 rows=1:5632 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 1 1
ustoref32 1 0 @out0
endkernel
kernel s.vec_mul in=2 out=1 threads=360448 tg=64 tgbytes=16 expect=in0:360448,in1:360448,out0:360448 rows=1:5632 tag=aten::mul
rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
mul 1 1 2
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_tiled_fma_4 in=2 out=1 threads=2048 tg=128 tgbytes=25600 expect=in0:360448,in1:1622016,out0:131072 rows=64:2048 tag=aten::matmul
rtou 0 0
utgid 1
ulid 2
loadconstf32 65 16
utor 92 1
div 66 92 65
rtou 3 66
utor 66 3
mul 66 66 65
sub 92 92 66
rtou 4 92
umulk 5 4 128
umulk 6 3 64
uandk 7 2 15
ushrk 8 2 4
uconst 15 @in1
umulk 15 15 4
uconst 9 0
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 5 0
loadconstf32 6 0
loadconstf32 7 0
loadconstf32 8 0
loadconstf32 9 0
loadconstf32 10 0
loadconstf32 11 0
loadconstf32 12 0
loadconstf32 13 0
loadconstf32 14 0
loadconstf32 15 0
loadconstf32 16 0
loadconstf32 17 0
loadconstf32 18 0
loadconstf32 19 0
loadconstf32 20 0
loadconstf32 21 0
loadconstf32 22 0
loadconstf32 23 0
loadconstf32 24 0
loadconstf32 25 0
loadconstf32 26 0
loadconstf32 27 0
loadconstf32 28 0
loadconstf32 29 0
loadconstf32 30 0
loadconstf32 31 0
loadconstf32 32 0
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
loadconstf32 96 176
forbegin 96
uaddk 10 2 0
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 5632
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 128
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 5632
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 256
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 5632
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 384
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 64
cmplt 93 92 94
ifbegin 93
umulk 10 10 5632
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 0
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 2816
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 44
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 128
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 2816
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 44
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 256
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 2816
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 44
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 384
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 2816
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 44
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
barriertg 
umulk 10 8 8
utor 89 10
umulk 10 7 8
uaddk 10 10 2176
utor 90 10
loadconstf32 97 32
forbegin 97
tgloadf32 73 90 0
tgloadf32 74 90 1
tgloadf32 75 90 2
tgloadf32 76 90 3
tgloadf32 77 90 4
tgloadf32 78 90 5
tgloadf32 79 90 6
tgloadf32 80 90 7
tgloadf32 81 89 0
tgloadf32 82 89 1
tgloadf32 83 89 2
tgloadf32 84 89 3
tgloadf32 85 89 4
tgloadf32 86 89 5
tgloadf32 87 89 6
tgloadf32 88 89 7
fma 1 81 73 1
fma 2 81 74 2
fma 3 81 75 3
fma 4 81 76 4
fma 5 81 77 5
fma 6 81 78 6
fma 7 81 79 7
fma 8 81 80 8
fma 9 82 73 9
fma 10 82 74 10
fma 11 82 75 11
fma 12 82 76 12
fma 13 82 77 13
fma 14 82 78 14
fma 15 82 79 15
fma 16 82 80 16
fma 17 83 73 17
fma 18 83 74 18
fma 19 83 75 19
fma 20 83 76 20
fma 21 83 77 21
fma 22 83 78 22
fma 23 83 79 23
fma 24 83 80 24
fma 25 84 73 25
fma 26 84 74 26
fma 27 84 75 27
fma 28 84 76 28
fma 29 84 77 29
fma 30 84 78 30
fma 31 84 79 31
fma 32 84 80 32
fma 33 85 73 33
fma 34 85 74 34
fma 35 85 75 35
fma 36 85 76 36
fma 37 85 77 37
fma 38 85 78 38
fma 39 85 79 39
fma 40 85 80 40
fma 41 86 73 41
fma 42 86 74 42
fma 43 86 75 43
fma 44 86 76 44
fma 45 86 77 45
fma 46 86 78 46
fma 47 86 79 47
fma 48 86 80 48
fma 49 87 73 49
fma 50 87 74 50
fma 51 87 75 51
fma 52 87 76 52
fma 53 87 77 53
fma 54 87 78 54
fma 55 87 79 55
fma 56 87 80 56
fma 57 88 73 57
fma 58 88 74 58
fma 59 88 75 59
fma 60 88 76 60
fma 61 88 77 61
fma 62 88 78 62
fma 63 88 79 63
fma 64 88 80 64
addconst 89 89 68
addconst 90 90 132
forend 
barriertg 
uaddk 9 9 32
forend 
umulk 10 8 8
uadd 10 10 6
umulk 11 7 8
uadd 11 11 5
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 1 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 2 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 3 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 4 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 5 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 6 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 7 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 8 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 9 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 10 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 11 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 12 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 13 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 14 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 15 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 16 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 17 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 18 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 19 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 20 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 21 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 22 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 23 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 24 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 25 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 26 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 27 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 28 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 29 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 30 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 31 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 32 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 33 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 34 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 35 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 36 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 37 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 38 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 39 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 40 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 41 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 42 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 43 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 44 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 45 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 46 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 47 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 48 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 49 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 50 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 51 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 52 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 53 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 54 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 55 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 56 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 57 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 58 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 59 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 60 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 61 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 62 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 63 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 64
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 64 12 @out0
ifend 
endkernel
kernel s.silu_mul in=2 out=1 threads=360448 tg=64 tgbytes=16 expect=in0:360448,in1:360448,out0:360448 rows=1:5632 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 2 1
mul 1 1 2
uloadf32 3 0 @in1
mul 1 1 3
ustoref32 1 0 @out0
endkernel
kernel s.rmsnorm_2 in=2 out=1 threads=114688 tg=256 tgbytes=1024 expect=in0:917504,in1:2048,out0:917504 rows=1:256 tag=aten::rms_norm
rtou 0 0
loadconstf32 4 0
loadconstf32 7 2048
utgid 1
ulid 2
loadlid 1
umulk 3 1 2048
uaddk 4 2 0
loadconstf32 8 8
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
mulconst 8 8 0.00048828125
addconst 8 8 0.00000999999975
rsqrt 8 8
uaddk 4 2 0
loadconstf32 10 8
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
kernel s.matmul_q4_tiled_fma_5 in=2 out=1 threads=14336 tg=128 tgbytes=25600 expect=in0:917504,in1:589824,out0:917504 rows=64:2048 tag=aten::matmul
rtou 0 0
utgid 1
ulid 2
loadconstf32 65 16
utor 92 1
div 66 92 65
rtou 3 66
utor 66 3
mul 66 66 65
sub 92 92 66
rtou 4 92
umulk 5 4 128
umulk 6 3 64
uandk 7 2 15
ushrk 8 2 4
uconst 15 @in1
umulk 15 15 4
uconst 9 0
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 5 0
loadconstf32 6 0
loadconstf32 7 0
loadconstf32 8 0
loadconstf32 9 0
loadconstf32 10 0
loadconstf32 11 0
loadconstf32 12 0
loadconstf32 13 0
loadconstf32 14 0
loadconstf32 15 0
loadconstf32 16 0
loadconstf32 17 0
loadconstf32 18 0
loadconstf32 19 0
loadconstf32 20 0
loadconstf32 21 0
loadconstf32 22 0
loadconstf32 23 0
loadconstf32 24 0
loadconstf32 25 0
loadconstf32 26 0
loadconstf32 27 0
loadconstf32 28 0
loadconstf32 29 0
loadconstf32 30 0
loadconstf32 31 0
loadconstf32 32 0
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
loadconstf32 96 64
forbegin 96
uaddk 10 2 0
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 128
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 256
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 384
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 0
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+524288
uloadf32 95 10 @in1+557056
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 128
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+524288
uloadf32 95 10 @in1+557056
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 256
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+524288
uloadf32 95 10 @in1+557056
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 384
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+524288
uloadf32 95 10 @in1+557056
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
barriertg 
umulk 10 8 8
utor 89 10
umulk 10 7 8
uaddk 10 10 2176
utor 90 10
loadconstf32 97 32
forbegin 97
tgloadf32 73 90 0
tgloadf32 74 90 1
tgloadf32 75 90 2
tgloadf32 76 90 3
tgloadf32 77 90 4
tgloadf32 78 90 5
tgloadf32 79 90 6
tgloadf32 80 90 7
tgloadf32 81 89 0
tgloadf32 82 89 1
tgloadf32 83 89 2
tgloadf32 84 89 3
tgloadf32 85 89 4
tgloadf32 86 89 5
tgloadf32 87 89 6
tgloadf32 88 89 7
fma 1 81 73 1
fma 2 81 74 2
fma 3 81 75 3
fma 4 81 76 4
fma 5 81 77 5
fma 6 81 78 6
fma 7 81 79 7
fma 8 81 80 8
fma 9 82 73 9
fma 10 82 74 10
fma 11 82 75 11
fma 12 82 76 12
fma 13 82 77 13
fma 14 82 78 14
fma 15 82 79 15
fma 16 82 80 16
fma 17 83 73 17
fma 18 83 74 18
fma 19 83 75 19
fma 20 83 76 20
fma 21 83 77 21
fma 22 83 78 22
fma 23 83 79 23
fma 24 83 80 24
fma 25 84 73 25
fma 26 84 74 26
fma 27 84 75 27
fma 28 84 76 28
fma 29 84 77 29
fma 30 84 78 30
fma 31 84 79 31
fma 32 84 80 32
fma 33 85 73 33
fma 34 85 74 34
fma 35 85 75 35
fma 36 85 76 36
fma 37 85 77 37
fma 38 85 78 38
fma 39 85 79 39
fma 40 85 80 40
fma 41 86 73 41
fma 42 86 74 42
fma 43 86 75 43
fma 44 86 76 44
fma 45 86 77 45
fma 46 86 78 46
fma 47 86 79 47
fma 48 86 80 48
fma 49 87 73 49
fma 50 87 74 50
fma 51 87 75 51
fma 52 87 76 52
fma 53 87 77 53
fma 54 87 78 54
fma 55 87 79 55
fma 56 87 80 56
fma 57 88 73 57
fma 58 88 74 58
fma 59 88 75 59
fma 60 88 76 60
fma 61 88 77 61
fma 62 88 78 62
fma 63 88 79 63
fma 64 88 80 64
addconst 89 89 68
addconst 90 90 132
forend 
barriertg 
uaddk 9 9 32
forend 
umulk 10 8 8
uadd 10 10 6
umulk 11 7 8
uadd 11 11 5
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 1 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 2 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 3 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 4 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 5 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 6 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 7 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 8 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 9 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 10 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 11 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 12 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 13 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 14 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 15 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 16 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 17 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 18 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 19 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 20 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 21 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 22 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 23 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 24 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 25 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 26 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 27 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 28 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 29 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 30 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 31 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 32 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 33 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 34 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 35 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 36 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 37 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 38 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 39 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 40 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 41 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 42 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 43 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 44 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 45 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 46 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 47 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 48 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 49 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 50 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 51 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 52 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 53 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 54 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 55 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 56 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 57 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 58 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 59 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 60 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 61 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 62 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 63 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 64 12 @out0
ifend 
endkernel
kernel s.matmul_q4_tiled_fma_6 in=2 out=1 threads=1792 tg=128 tgbytes=25600 expect=in0:917504,in1:73728,out0:114688 rows=64:256 tag=aten::matmul
rtou 0 0
utgid 1
ulid 2
loadconstf32 65 2
utor 92 1
div 66 92 65
rtou 3 66
utor 66 3
mul 66 66 65
sub 92 92 66
rtou 4 92
umulk 5 4 128
umulk 6 3 64
uandk 7 2 15
ushrk 8 2 4
uconst 15 @in1
umulk 15 15 4
uconst 9 0
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 5 0
loadconstf32 6 0
loadconstf32 7 0
loadconstf32 8 0
loadconstf32 9 0
loadconstf32 10 0
loadconstf32 11 0
loadconstf32 12 0
loadconstf32 13 0
loadconstf32 14 0
loadconstf32 15 0
loadconstf32 16 0
loadconstf32 17 0
loadconstf32 18 0
loadconstf32 19 0
loadconstf32 20 0
loadconstf32 21 0
loadconstf32 22 0
loadconstf32 23 0
loadconstf32 24 0
loadconstf32 25 0
loadconstf32 26 0
loadconstf32 27 0
loadconstf32 28 0
loadconstf32 29 0
loadconstf32 30 0
loadconstf32 31 0
loadconstf32 32 0
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
loadconstf32 96 64
forbegin 96
uaddk 10 2 0
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 128
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 256
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 384
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 0
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 256
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+65536
uloadf32 95 10 @in1+69632
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 128
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 256
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+65536
uloadf32 95 10 @in1+69632
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 256
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 256
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+65536
uloadf32 95 10 @in1+69632
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 384
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 256
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+65536
uloadf32 95 10 @in1+69632
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
barriertg 
umulk 10 8 8
utor 89 10
umulk 10 7 8
uaddk 10 10 2176
utor 90 10
loadconstf32 97 32
forbegin 97
tgloadf32 73 90 0
tgloadf32 74 90 1
tgloadf32 75 90 2
tgloadf32 76 90 3
tgloadf32 77 90 4
tgloadf32 78 90 5
tgloadf32 79 90 6
tgloadf32 80 90 7
tgloadf32 81 89 0
tgloadf32 82 89 1
tgloadf32 83 89 2
tgloadf32 84 89 3
tgloadf32 85 89 4
tgloadf32 86 89 5
tgloadf32 87 89 6
tgloadf32 88 89 7
fma 1 81 73 1
fma 2 81 74 2
fma 3 81 75 3
fma 4 81 76 4
fma 5 81 77 5
fma 6 81 78 6
fma 7 81 79 7
fma 8 81 80 8
fma 9 82 73 9
fma 10 82 74 10
fma 11 82 75 11
fma 12 82 76 12
fma 13 82 77 13
fma 14 82 78 14
fma 15 82 79 15
fma 16 82 80 16
fma 17 83 73 17
fma 18 83 74 18
fma 19 83 75 19
fma 20 83 76 20
fma 21 83 77 21
fma 22 83 78 22
fma 23 83 79 23
fma 24 83 80 24
fma 25 84 73 25
fma 26 84 74 26
fma 27 84 75 27
fma 28 84 76 28
fma 29 84 77 29
fma 30 84 78 30
fma 31 84 79 31
fma 32 84 80 32
fma 33 85 73 33
fma 34 85 74 34
fma 35 85 75 35
fma 36 85 76 36
fma 37 85 77 37
fma 38 85 78 38
fma 39 85 79 39
fma 40 85 80 40
fma 41 86 73 41
fma 42 86 74 42
fma 43 86 75 43
fma 44 86 76 44
fma 45 86 77 45
fma 46 86 78 46
fma 47 86 79 47
fma 48 86 80 48
fma 49 87 73 49
fma 50 87 74 50
fma 51 87 75 51
fma 52 87 76 52
fma 53 87 77 53
fma 54 87 78 54
fma 55 87 79 55
fma 56 87 80 56
fma 57 88 73 57
fma 58 88 74 58
fma 59 88 75 59
fma 60 88 76 60
fma 61 88 77 61
fma 62 88 78 62
fma 63 88 79 63
fma 64 88 80 64
addconst 89 89 68
addconst 90 90 132
forend 
barriertg 
uaddk 9 9 32
forend 
umulk 10 8 8
uadd 10 10 6
umulk 11 7 8
uadd 11 11 5
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 1 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 2 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 3 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 4 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 5 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 6 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 7 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 8 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 9 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 10 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 11 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 12 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 13 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 14 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 15 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 16 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 17 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 18 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 19 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 20 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 21 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 22 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 23 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 24 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 25 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 26 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 27 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 28 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 29 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 30 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 31 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 32 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 33 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 34 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 35 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 36 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 37 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 38 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 39 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 40 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 41 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 42 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 43 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 44 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 45 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 46 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 47 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 48 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 49 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 50 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 51 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 52 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 53 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 54 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 55 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 56 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 57 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 58 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 59 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 60 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 61 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 62 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 63 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 256
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 256
uadd 12 12 13
ustoref32 64 12 @out0
ifend 
endkernel
kernel s.rope_3 in=2 out=1 threads=458752 tg=64 tgbytes=16 expect=in0:917504,in1:1,out0:917504 rows=1:1024
rtou 0 0
uconst 1 0
uandk 2 0 31
ushrk 3 0 5
umulk 5 3 64
uadd 5 5 2
uloadf32 1 1 @in1
utor 10 3
addconst 10 10 0.5
mulconst 10 10 0.03125
rtou 4 10
utor 9 4
add 1 1 9
utor 9 2
mulconst 9 9 -0.28782313662426
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
uloadf32 7 5 @in0+32
mul 8 6 5
mul 3 7 4
sub 8 8 3
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
ustoref32 8 5 @out0+32
endkernel
kernel s.rope_4 in=2 out=1 threads=57344 tg=64 tgbytes=16 expect=in0:114688,in1:1,out0:114688 rows=1:128
rtou 0 0
uconst 1 0
uandk 2 0 31
ushrk 3 0 5
umulk 5 3 64
uadd 5 5 2
uloadf32 1 1 @in1
utor 10 3
addconst 10 10 0.5
mulconst 10 10 0.25
rtou 4 10
utor 9 4
add 1 1 9
utor 9 2
mulconst 9 9 -0.28782313662426
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
uloadf32 7 5 @in0+32
mul 8 6 5
mul 3 7 4
sub 8 8 3
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
ustoref32 8 5 @out0+32
endkernel
kernel s.kv_scatter_2 in=3 out=1 threads=114688 tg=64 tgbytes=16 expect=in0:114688,in1:1,in2:524288,out0:524288 rows=1:256
rtou 0 0
uconst 1 0
uloadf32 1 1 @in1
rtou 2 1
utor 3 0
addconst 3 3 0.5
mulconst 3 3 0.015625
rtou 3 3
utor 4 0
utor 5 3
mulconst 5 5 64
sub 4 4 5
rtou 6 4
utor 6 3
addconst 6 6 0.5
mulconst 6 6 0.25
rtou 4 6
utor 7 3
utor 8 4
mulconst 8 8 4
sub 7 7 8
rtou 5 7
uadd 7 2 4
umulk 5 5 2048
uadd 7 7 5
umulk 7 7 64
uadd 7 7 6
uloadf32 2 0 @in0
ustoref32 2 7 @in2
endkernel
kernel s.flash_prefill_2 in=4 out=1 threads=114688 tg=128 tgbytes=13312 expect=in0:917504,in1:524288,in2:524288,in3:1,out0:917504 rows=16:4096 tag=aten::scaled_dot_product_attention
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
loadconstf32 1 8
div 2 2 1
rtou 11 2
umulk 8 11 131072
uloadf32 1 12 @in3
utor 2 7
add 5 1 2
addconst 5 5 1
add 6 6 1
addconst 6 6 16
addconst 1 1 448
cmplt 3 1 6
ifbegin 3
mulconst 6 1 1
ifend 
addconst 7 6 15
mulconst 7 7 0.0625
loadconstf32 1 448
cmplt 4 2 1
utor 9 6
umulk 11 5 64
utor 20 11
add 24 20 9
addconst 25 24 1024
umulk 11 6 64
utor 21 11
addconst 21 21 1024
umulk 11 5 16
utor 26 11
addconst 26 26 3072
mulconst 27 9 16
addconst 27 27 2048
utor 28 5
add 28 28 27
umulk 9 7 32
uadd 9 9 4
umulk 9 9 64
uadd 9 9 6
loadconstf32 41 0
loadconstf32 42 0
loadconstf32 43 0
loadconstf32 44 0
loadconstf32 45 0
loadconstf32 46 0
loadconstf32 47 0
loadconstf32 48 0
ifbegin 4
uloadf32 41 9 @in0+0
uloadf32 42 9 @in0+8
uloadf32 43 9 @in0+16
uloadf32 44 9 @in0+24
uloadf32 45 9 @in0+32
uloadf32 46 9 @in0+40
uloadf32 47 9 @in0+48
uloadf32 48 9 @in0+56
ifend 
tgstoref32 41 24 0
tgstoref32 42 24 8
tgstoref32 43 24 16
tgstoref32 44 24 24
tgstoref32 45 24 32
tgstoref32 46 24 40
tgstoref32 47 24 48
tgstoref32 48 24 56
loadconstf32 33 0
loadconstf32 34 0
loadconstf32 35 0
loadconstf32 36 0
loadconstf32 37 0
loadconstf32 38 0
loadconstf32 39 0
loadconstf32 40 0
loadconstf32 10 -1000000000000000019884624838656.0
loadconstf32 11 0
uconst 10 0
loadconstf32 8 0
forbegin 7
barriertg 
uadd 11 10 5
utor 2 11
cmplt 3 2 6
umulk 11 11 64
uadd 11 11 8
uadd 11 11 6
loadconstf32 41 0
loadconstf32 42 0
loadconstf32 43 0
loadconstf32 44 0
loadconstf32 45 0
loadconstf32 46 0
loadconstf32 47 0
loadconstf32 48 0
ifbegin 3
uloadf32 41 11 @in1+0
uloadf32 42 11 @in1+8
uloadf32 43 11 @in1+16
uloadf32 44 11 @in1+24
uloadf32 45 11 @in1+32
uloadf32 46 11 @in1+40
uloadf32 47 11 @in1+48
uloadf32 48 11 @in1+56
ifend 
tgstoref32 41 25 0
tgstoref32 42 25 8
tgstoref32 43 25 16
tgstoref32 44 25 24
tgstoref32 45 25 32
tgstoref32 46 25 40
tgstoref32 47 25 48
tgstoref32 48 25 56
loadconstf32 41 0
loadconstf32 42 0
loadconstf32 43 0
loadconstf32 44 0
loadconstf32 45 0
loadconstf32 46 0
loadconstf32 47 0
loadconstf32 48 0
ifbegin 3
uloadf32 41 11 @in2+0
uloadf32 42 11 @in2+8
uloadf32 43 11 @in2+16
uloadf32 44 11 @in2+24
uloadf32 45 11 @in2+32
uloadf32 46 11 @in2+40
uloadf32 47 11 @in2+48
uloadf32 48 11 @in2+56
ifend 
tgstoref32 41 28 0
tgstoref32 42 28 128
tgstoref32 43 28 256
tgstoref32 44 28 384
tgstoref32 45 28 512
tgstoref32 46 28 640
tgstoref32 47 28 768
tgstoref32 48 28 896
barriertg 
loadconstf32 15 0
loadconstf32 16 0
mulconst 22 20 1
mulconst 23 21 1
loadconstf32 57 16
forbegin 57
tgloadf32 41 22 0
tgloadf32 42 22 1
tgloadf32 43 22 2
tgloadf32 44 22 3
tgloadf32 45 23 0
tgloadf32 46 23 1
tgloadf32 47 23 2
tgloadf32 48 23 3
tgloadf32 49 23 512
tgloadf32 50 23 513
tgloadf32 51 23 514
tgloadf32 52 23 515
fma 15 41 45 15
fma 16 41 49 16
fma 15 42 46 15
fma 16 42 50 16
fma 15 43 47 15
fma 16 43 51 16
fma 15 44 48 15
fma 16 44 52 16
addconst 22 22 4
addconst 23 23 4
forend 
add 2 8 9
loadconstf32 18 -1000000000000000019884624838656.0
cmplt 3 2 5
ifbegin 3
mulconst 18 15 0.125
ifend 
addconst 2 2 8
loadconstf32 19 -1000000000000000019884624838656.0
cmplt 3 2 5
ifbegin 3
mulconst 19 16 0.125
ifend 
add 2 26 9
tgstoref32 18 2 0
tgstoref32 19 2 8
barriertg 
tgloadf32 41 26 0
tgloadf32 42 26 1
tgloadf32 43 26 2
tgloadf32 44 26 3
tgloadf32 45 26 4
tgloadf32 46 26 5
tgloadf32 47 26 6
tgloadf32 48 26 7
tgloadf32 49 26 8
tgloadf32 50 26 9
tgloadf32 51 26 10
tgloadf32 52 26 11
tgloadf32 53 26 12
tgloadf32 54 26 13
tgloadf32 55 26 14
tgloadf32 56 26 15
mulconst 12 10 1
cmplt 3 12 41
ifbegin 3
mulconst 12 41 1
ifend 
cmplt 3 12 42
ifbegin 3
mulconst 12 42 1
ifend 
cmplt 3 12 43
ifbegin 3
mulconst 12 43 1
ifend 
cmplt 3 12 44
ifbegin 3
mulconst 12 44 1
ifend 
cmplt 3 12 45
ifbegin 3
mulconst 12 45 1
ifend 
cmplt 3 12 46
ifbegin 3
mulconst 12 46 1
ifend 
cmplt 3 12 47
ifbegin 3
mulconst 12 47 1
ifend 
cmplt 3 12 48
ifbegin 3
mulconst 12 48 1
ifend 
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
sub 13 10 12
exp 13 13
loadconstf32 14 0
loadconstf32 1 -500000000000000009942312419328.0
cmplt 3 1 41
sub 41 41 12
exp 41 41
mul 41 41 3
add 14 14 41
cmplt 3 1 42
sub 42 42 12
exp 42 42
mul 42 42 3
add 14 14 42
cmplt 3 1 43
sub 43 43 12
exp 43 43
mul 43 43 3
add 14 14 43
cmplt 3 1 44
sub 44 44 12
exp 44 44
mul 44 44 3
add 14 14 44
cmplt 3 1 45
sub 45 45 12
exp 45 45
mul 45 45 3
add 14 14 45
cmplt 3 1 46
sub 46 46 12
exp 46 46
mul 46 46 3
add 14 14 46
cmplt 3 1 47
sub 47 47 12
exp 47 47
mul 47 47 3
add 14 14 47
cmplt 3 1 48
sub 48 48 12
exp 48 48
mul 48 48 3
add 14 14 48
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
mul 11 11 13
add 11 11 14
mulconst 10 12 1
mul 33 33 13
tgloadf32 29 27 0
tgloadf32 30 27 1
tgloadf32 31 27 2
tgloadf32 32 27 3
fma 33 41 29 33
fma 33 42 30 33
fma 33 43 31 33
fma 33 44 32 33
tgloadf32 29 27 4
tgloadf32 30 27 5
tgloadf32 31 27 6
tgloadf32 32 27 7
fma 33 45 29 33
fma 33 46 30 33
fma 33 47 31 33
fma 33 48 32 33
tgloadf32 29 27 8
tgloadf32 30 27 9
tgloadf32 31 27 10
tgloadf32 32 27 11
fma 33 49 29 33
fma 33 50 30 33
fma 33 51 31 33
fma 33 52 32 33
tgloadf32 29 27 12
tgloadf32 30 27 13
tgloadf32 31 27 14
tgloadf32 32 27 15
fma 33 53 29 33
fma 33 54 30 33
fma 33 55 31 33
fma 33 56 32 33
mul 34 34 13
tgloadf32 29 27 128
tgloadf32 30 27 129
tgloadf32 31 27 130
tgloadf32 32 27 131
fma 34 41 29 34
fma 34 42 30 34
fma 34 43 31 34
fma 34 44 32 34
tgloadf32 29 27 132
tgloadf32 30 27 133
tgloadf32 31 27 134
tgloadf32 32 27 135
fma 34 45 29 34
fma 34 46 30 34
fma 34 47 31 34
fma 34 48 32 34
tgloadf32 29 27 136
tgloadf32 30 27 137
tgloadf32 31 27 138
tgloadf32 32 27 139
fma 34 49 29 34
fma 34 50 30 34
fma 34 51 31 34
fma 34 52 32 34
tgloadf32 29 27 140
tgloadf32 30 27 141
tgloadf32 31 27 142
tgloadf32 32 27 143
fma 34 53 29 34
fma 34 54 30 34
fma 34 55 31 34
fma 34 56 32 34
mul 35 35 13
tgloadf32 29 27 256
tgloadf32 30 27 257
tgloadf32 31 27 258
tgloadf32 32 27 259
fma 35 41 29 35
fma 35 42 30 35
fma 35 43 31 35
fma 35 44 32 35
tgloadf32 29 27 260
tgloadf32 30 27 261
tgloadf32 31 27 262
tgloadf32 32 27 263
fma 35 45 29 35
fma 35 46 30 35
fma 35 47 31 35
fma 35 48 32 35
tgloadf32 29 27 264
tgloadf32 30 27 265
tgloadf32 31 27 266
tgloadf32 32 27 267
fma 35 49 29 35
fma 35 50 30 35
fma 35 51 31 35
fma 35 52 32 35
tgloadf32 29 27 268
tgloadf32 30 27 269
tgloadf32 31 27 270
tgloadf32 32 27 271
fma 35 53 29 35
fma 35 54 30 35
fma 35 55 31 35
fma 35 56 32 35
mul 36 36 13
tgloadf32 29 27 384
tgloadf32 30 27 385
tgloadf32 31 27 386
tgloadf32 32 27 387
fma 36 41 29 36
fma 36 42 30 36
fma 36 43 31 36
fma 36 44 32 36
tgloadf32 29 27 388
tgloadf32 30 27 389
tgloadf32 31 27 390
tgloadf32 32 27 391
fma 36 45 29 36
fma 36 46 30 36
fma 36 47 31 36
fma 36 48 32 36
tgloadf32 29 27 392
tgloadf32 30 27 393
tgloadf32 31 27 394
tgloadf32 32 27 395
fma 36 49 29 36
fma 36 50 30 36
fma 36 51 31 36
fma 36 52 32 36
tgloadf32 29 27 396
tgloadf32 30 27 397
tgloadf32 31 27 398
tgloadf32 32 27 399
fma 36 53 29 36
fma 36 54 30 36
fma 36 55 31 36
fma 36 56 32 36
mul 37 37 13
tgloadf32 29 27 512
tgloadf32 30 27 513
tgloadf32 31 27 514
tgloadf32 32 27 515
fma 37 41 29 37
fma 37 42 30 37
fma 37 43 31 37
fma 37 44 32 37
tgloadf32 29 27 516
tgloadf32 30 27 517
tgloadf32 31 27 518
tgloadf32 32 27 519
fma 37 45 29 37
fma 37 46 30 37
fma 37 47 31 37
fma 37 48 32 37
tgloadf32 29 27 520
tgloadf32 30 27 521
tgloadf32 31 27 522
tgloadf32 32 27 523
fma 37 49 29 37
fma 37 50 30 37
fma 37 51 31 37
fma 37 52 32 37
tgloadf32 29 27 524
tgloadf32 30 27 525
tgloadf32 31 27 526
tgloadf32 32 27 527
fma 37 53 29 37
fma 37 54 30 37
fma 37 55 31 37
fma 37 56 32 37
mul 38 38 13
tgloadf32 29 27 640
tgloadf32 30 27 641
tgloadf32 31 27 642
tgloadf32 32 27 643
fma 38 41 29 38
fma 38 42 30 38
fma 38 43 31 38
fma 38 44 32 38
tgloadf32 29 27 644
tgloadf32 30 27 645
tgloadf32 31 27 646
tgloadf32 32 27 647
fma 38 45 29 38
fma 38 46 30 38
fma 38 47 31 38
fma 38 48 32 38
tgloadf32 29 27 648
tgloadf32 30 27 649
tgloadf32 31 27 650
tgloadf32 32 27 651
fma 38 49 29 38
fma 38 50 30 38
fma 38 51 31 38
fma 38 52 32 38
tgloadf32 29 27 652
tgloadf32 30 27 653
tgloadf32 31 27 654
tgloadf32 32 27 655
fma 38 53 29 38
fma 38 54 30 38
fma 38 55 31 38
fma 38 56 32 38
mul 39 39 13
tgloadf32 29 27 768
tgloadf32 30 27 769
tgloadf32 31 27 770
tgloadf32 32 27 771
fma 39 41 29 39
fma 39 42 30 39
fma 39 43 31 39
fma 39 44 32 39
tgloadf32 29 27 772
tgloadf32 30 27 773
tgloadf32 31 27 774
tgloadf32 32 27 775
fma 39 45 29 39
fma 39 46 30 39
fma 39 47 31 39
fma 39 48 32 39
tgloadf32 29 27 776
tgloadf32 30 27 777
tgloadf32 31 27 778
tgloadf32 32 27 779
fma 39 49 29 39
fma 39 50 30 39
fma 39 51 31 39
fma 39 52 32 39
tgloadf32 29 27 780
tgloadf32 30 27 781
tgloadf32 31 27 782
tgloadf32 32 27 783
fma 39 53 29 39
fma 39 54 30 39
fma 39 55 31 39
fma 39 56 32 39
mul 40 40 13
tgloadf32 29 27 896
tgloadf32 30 27 897
tgloadf32 31 27 898
tgloadf32 32 27 899
fma 40 41 29 40
fma 40 42 30 40
fma 40 43 31 40
fma 40 44 32 40
tgloadf32 29 27 900
tgloadf32 30 27 901
tgloadf32 31 27 902
tgloadf32 32 27 903
fma 40 45 29 40
fma 40 46 30 40
fma 40 47 31 40
fma 40 48 32 40
tgloadf32 29 27 904
tgloadf32 30 27 905
tgloadf32 31 27 906
tgloadf32 32 27 907
fma 40 49 29 40
fma 40 50 30 40
fma 40 51 31 40
fma 40 52 32 40
tgloadf32 29 27 908
tgloadf32 30 27 909
tgloadf32 31 27 910
tgloadf32 32 27 911
fma 40 53 29 40
fma 40 54 30 40
fma 40 55 31 40
fma 40 56 32 40
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
ifend 
endkernel
kernel s.vec_add_2 in=2 out=1 threads=917504 tg=64 tgbytes=16 expect=in0:917504,in1:917504,out0:917504 rows=1:2048 tag=aten::add
rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
add 1 1 2
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_tiled_fma_7 in=2 out=1 threads=39424 tg=128 tgbytes=25600 expect=in0:917504,in1:1622016,out0:2523136 rows=64:5632 tag=aten::matmul
rtou 0 0
utgid 1
ulid 2
loadconstf32 65 44
utor 92 1
div 66 92 65
rtou 3 66
utor 66 3
mul 66 66 65
sub 92 92 66
rtou 4 92
umulk 5 4 128
umulk 6 3 64
uandk 7 2 15
ushrk 8 2 4
uconst 15 @in1
umulk 15 15 4
uconst 9 0
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 5 0
loadconstf32 6 0
loadconstf32 7 0
loadconstf32 8 0
loadconstf32 9 0
loadconstf32 10 0
loadconstf32 11 0
loadconstf32 12 0
loadconstf32 13 0
loadconstf32 14 0
loadconstf32 15 0
loadconstf32 16 0
loadconstf32 17 0
loadconstf32 18 0
loadconstf32 19 0
loadconstf32 20 0
loadconstf32 21 0
loadconstf32 22 0
loadconstf32 23 0
loadconstf32 24 0
loadconstf32 25 0
loadconstf32 26 0
loadconstf32 27 0
loadconstf32 28 0
loadconstf32 29 0
loadconstf32 30 0
loadconstf32 31 0
loadconstf32 32 0
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
loadconstf32 96 64
forbegin 96
uaddk 10 2 0
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 128
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 256
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 384
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 2048
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 0
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 5632
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 128
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 5632
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 256
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 5632
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 384
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 5632
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 1024
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 16
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
barriertg 
umulk 10 8 8
utor 89 10
umulk 10 7 8
uaddk 10 10 2176
utor 90 10
loadconstf32 97 32
forbegin 97
tgloadf32 73 90 0
tgloadf32 74 90 1
tgloadf32 75 90 2
tgloadf32 76 90 3
tgloadf32 77 90 4
tgloadf32 78 90 5
tgloadf32 79 90 6
tgloadf32 80 90 7
tgloadf32 81 89 0
tgloadf32 82 89 1
tgloadf32 83 89 2
tgloadf32 84 89 3
tgloadf32 85 89 4
tgloadf32 86 89 5
tgloadf32 87 89 6
tgloadf32 88 89 7
fma 1 81 73 1
fma 2 81 74 2
fma 3 81 75 3
fma 4 81 76 4
fma 5 81 77 5
fma 6 81 78 6
fma 7 81 79 7
fma 8 81 80 8
fma 9 82 73 9
fma 10 82 74 10
fma 11 82 75 11
fma 12 82 76 12
fma 13 82 77 13
fma 14 82 78 14
fma 15 82 79 15
fma 16 82 80 16
fma 17 83 73 17
fma 18 83 74 18
fma 19 83 75 19
fma 20 83 76 20
fma 21 83 77 21
fma 22 83 78 22
fma 23 83 79 23
fma 24 83 80 24
fma 25 84 73 25
fma 26 84 74 26
fma 27 84 75 27
fma 28 84 76 28
fma 29 84 77 29
fma 30 84 78 30
fma 31 84 79 31
fma 32 84 80 32
fma 33 85 73 33
fma 34 85 74 34
fma 35 85 75 35
fma 36 85 76 36
fma 37 85 77 37
fma 38 85 78 38
fma 39 85 79 39
fma 40 85 80 40
fma 41 86 73 41
fma 42 86 74 42
fma 43 86 75 43
fma 44 86 76 44
fma 45 86 77 45
fma 46 86 78 46
fma 47 86 79 47
fma 48 86 80 48
fma 49 87 73 49
fma 50 87 74 50
fma 51 87 75 51
fma 52 87 76 52
fma 53 87 77 53
fma 54 87 78 54
fma 55 87 79 55
fma 56 87 80 56
fma 57 88 73 57
fma 58 88 74 58
fma 59 88 75 59
fma 60 88 76 60
fma 61 88 77 61
fma 62 88 78 62
fma 63 88 79 63
fma 64 88 80 64
addconst 89 89 68
addconst 90 90 132
forend 
barriertg 
uaddk 9 9 32
forend 
umulk 10 8 8
uadd 10 10 6
umulk 11 7 8
uadd 11 11 5
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 1 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 2 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 3 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 4 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 5 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 6 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 7 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 8 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 9 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 10 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 11 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 12 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 13 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 14 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 15 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 16 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 17 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 18 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 19 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 20 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 21 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 22 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 23 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 24 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 25 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 26 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 27 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 28 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 29 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 30 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 31 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 32 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 33 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 34 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 35 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 36 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 37 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 38 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 39 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 40 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 41 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 42 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 43 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 44 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 45 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 46 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 47 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 48 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 49 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 50 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 51 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 52 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 53 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 54 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 55 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 56 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 57 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 58 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 59 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 60 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 61 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 62 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 63 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 5632
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 5632
uadd 12 12 13
ustoref32 64 12 @out0
ifend 
endkernel
kernel s.silu_mul_2 in=2 out=1 threads=2523136 tg=64 tgbytes=16 expect=in0:2523136,in1:2523136,out0:2523136 rows=1:5632 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 2 1
mul 1 1 2
uloadf32 3 0 @in1
mul 1 1 3
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_tiled_fma_8 in=2 out=1 threads=14336 tg=128 tgbytes=25600 expect=in0:2523136,in1:1622016,out0:917504 rows=64:2048 tag=aten::matmul
rtou 0 0
utgid 1
ulid 2
loadconstf32 65 16
utor 92 1
div 66 92 65
rtou 3 66
utor 66 3
mul 66 66 65
sub 92 92 66
rtou 4 92
umulk 5 4 128
umulk 6 3 64
uandk 7 2 15
ushrk 8 2 4
uconst 15 @in1
umulk 15 15 4
uconst 9 0
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 5 0
loadconstf32 6 0
loadconstf32 7 0
loadconstf32 8 0
loadconstf32 9 0
loadconstf32 10 0
loadconstf32 11 0
loadconstf32 12 0
loadconstf32 13 0
loadconstf32 14 0
loadconstf32 15 0
loadconstf32 16 0
loadconstf32 17 0
loadconstf32 18 0
loadconstf32 19 0
loadconstf32 20 0
loadconstf32 21 0
loadconstf32 22 0
loadconstf32 23 0
loadconstf32 24 0
loadconstf32 25 0
loadconstf32 26 0
loadconstf32 27 0
loadconstf32 28 0
loadconstf32 29 0
loadconstf32 30 0
loadconstf32 31 0
loadconstf32 32 0
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
loadconstf32 96 176
forbegin 96
uaddk 10 2 0
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 5632
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 128
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 5632
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 256
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 5632
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 384
ushrk 11 10 3
uandk 12 10 7
umulk 10 12 272
uadd 10 10 11
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
uadd 10 6 11
utor 92 10
loadconstf32 94 448
cmplt 93 92 94
ifbegin 93
umulk 10 10 5632
uadd 10 10 9
umulk 12 12 4
uadd 10 10 12
uloadf32 65 10 @in0+0
uloadf32 66 10 @in0+1
uloadf32 67 10 @in0+2
uloadf32 68 10 @in0+3
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 68
tgstoref32 67 91 136
tgstoref32 68 91 204
uaddk 10 2 0
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 2816
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 44
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 128
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 2816
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 44
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 256
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 2816
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 44
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
uaddk 10 2 384
ushrk 11 10 2
uandk 12 10 3
umulk 10 12 1056
uadd 10 10 11
uaddk 10 10 2176
utor 91 10
loadconstf32 65 0
loadconstf32 66 0
loadconstf32 67 0
loadconstf32 68 0
loadconstf32 69 0
loadconstf32 70 0
loadconstf32 71 0
loadconstf32 72 0
uadd 10 5 11
utor 92 10
loadconstf32 94 2048
cmplt 93 92 94
ifbegin 93
umulk 13 12 8
uadd 13 13 9
ushrk 11 13 1
umulk 12 10 2816
uadd 11 11 12
uadd 11 11 15
uloadu32 14 11 0
umulk 10 10 44
ushrk 13 13 7
uadd 10 10 13
uloadf32 94 10 @in1+1441792
uloadf32 95 10 @in1+1531904
ushrk 11 14 0
uandk 11 11 15
utor 65 11
sub 65 65 95
mul 65 65 94
ushrk 11 14 4
uandk 11 11 15
utor 66 11
sub 66 66 95
mul 66 66 94
ushrk 11 14 8
uandk 11 11 15
utor 67 11
sub 67 67 95
mul 67 67 94
ushrk 11 14 12
uandk 11 11 15
utor 68 11
sub 68 68 95
mul 68 68 94
ushrk 11 14 16
uandk 11 11 15
utor 69 11
sub 69 69 95
mul 69 69 94
ushrk 11 14 20
uandk 11 11 15
utor 70 11
sub 70 70 95
mul 70 70 94
ushrk 11 14 24
uandk 11 11 15
utor 71 11
sub 71 71 95
mul 71 71 94
ushrk 11 14 28
utor 72 11
sub 72 72 95
mul 72 72 94
ifend 
tgstoref32 65 91 0
tgstoref32 66 91 132
tgstoref32 67 91 264
tgstoref32 68 91 396
tgstoref32 69 91 528
tgstoref32 70 91 660
tgstoref32 71 91 792
tgstoref32 72 91 924
barriertg 
umulk 10 8 8
utor 89 10
umulk 10 7 8
uaddk 10 10 2176
utor 90 10
loadconstf32 97 32
forbegin 97
tgloadf32 73 90 0
tgloadf32 74 90 1
tgloadf32 75 90 2
tgloadf32 76 90 3
tgloadf32 77 90 4
tgloadf32 78 90 5
tgloadf32 79 90 6
tgloadf32 80 90 7
tgloadf32 81 89 0
tgloadf32 82 89 1
tgloadf32 83 89 2
tgloadf32 84 89 3
tgloadf32 85 89 4
tgloadf32 86 89 5
tgloadf32 87 89 6
tgloadf32 88 89 7
fma 1 81 73 1
fma 2 81 74 2
fma 3 81 75 3
fma 4 81 76 4
fma 5 81 77 5
fma 6 81 78 6
fma 7 81 79 7
fma 8 81 80 8
fma 9 82 73 9
fma 10 82 74 10
fma 11 82 75 11
fma 12 82 76 12
fma 13 82 77 13
fma 14 82 78 14
fma 15 82 79 15
fma 16 82 80 16
fma 17 83 73 17
fma 18 83 74 18
fma 19 83 75 19
fma 20 83 76 20
fma 21 83 77 21
fma 22 83 78 22
fma 23 83 79 23
fma 24 83 80 24
fma 25 84 73 25
fma 26 84 74 26
fma 27 84 75 27
fma 28 84 76 28
fma 29 84 77 29
fma 30 84 78 30
fma 31 84 79 31
fma 32 84 80 32
fma 33 85 73 33
fma 34 85 74 34
fma 35 85 75 35
fma 36 85 76 36
fma 37 85 77 37
fma 38 85 78 38
fma 39 85 79 39
fma 40 85 80 40
fma 41 86 73 41
fma 42 86 74 42
fma 43 86 75 43
fma 44 86 76 44
fma 45 86 77 45
fma 46 86 78 46
fma 47 86 79 47
fma 48 86 80 48
fma 49 87 73 49
fma 50 87 74 50
fma 51 87 75 51
fma 52 87 76 52
fma 53 87 77 53
fma 54 87 78 54
fma 55 87 79 55
fma 56 87 80 56
fma 57 88 73 57
fma 58 88 74 58
fma 59 88 75 59
fma 60 88 76 60
fma 61 88 77 61
fma 62 88 78 62
fma 63 88 79 63
fma 64 88 80 64
addconst 89 89 68
addconst 90 90 132
forend 
barriertg 
uaddk 9 9 32
forend 
umulk 10 8 8
uadd 10 10 6
umulk 11 7 8
uadd 11 11 5
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 1 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 2 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 3 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 4 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 5 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 6 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 7 12 @out0
ifend 
uaddk 12 10 0
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 8 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 9 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 10 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 11 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 12 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 13 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 14 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 15 12 @out0
ifend 
uaddk 12 10 1
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 16 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 17 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 18 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 19 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 20 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 21 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 22 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 23 12 @out0
ifend 
uaddk 12 10 2
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 24 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 25 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 26 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 27 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 28 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 29 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 30 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 31 12 @out0
ifend 
uaddk 12 10 3
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 32 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 33 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 34 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 35 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 36 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 37 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 38 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 39 12 @out0
ifend 
uaddk 12 10 4
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 40 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 41 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 42 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 43 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 44 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 45 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 46 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 47 12 @out0
ifend 
uaddk 12 10 5
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 48 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 49 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 50 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 51 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 52 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 53 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 54 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 55 12 @out0
ifend 
uaddk 12 10 6
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 56 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 0
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 57 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 1
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 58 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 2
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 59 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 3
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 60 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 4
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 61 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 5
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 62 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 6
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 63 12 @out0
ifend 
uaddk 12 10 7
utor 92 12
loadconstf32 94 448
cmplt 93 92 94
uaddk 13 11 7
utor 92 13
loadconstf32 94 2048
cmplt 91 92 94
mul 93 93 91
ifbegin 93
umulk 12 12 2048
uadd 12 12 13
ustoref32 64 12 @out0
ifend 
endkernel
kernel s.act_sigmoid_2 in=1 out=1 threads=2523136 tg=64 tgbytes=16 expect=in0:2523136,out0:2523136 rows=1:5632 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 1 1
ustoref32 1 0 @out0
endkernel
kernel s.vec_mul_2 in=2 out=1 threads=2523136 tg=64 tgbytes=16 expect=in0:2523136,in1:2523136,out0:2523136 rows=1:5632 tag=aten::mul
rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
mul 1 1 2
ustoref32 1 0 @out0
endkernel
kernel s.rmsnorm_3 in=2 out=1 threads=256 tg=256 tgbytes=1024 expect=in0:2048,in1:2048,out0:2048 rows=1:256 tag=aten::rms_norm
rtou 0 0
loadconstf32 4 0
loadconstf32 7 2048
utgid 1
ulid 2
loadlid 1
umulk 3 1 2048
uaddk 4 2 0
loadconstf32 8 8
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
mulconst 8 8 0.00048828125
addconst 8 8 0.00000999999975
rsqrt 8 8
uaddk 4 2 0
loadconstf32 10 8
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
kernel s.matmul_q4_simd in=2 out=1 threads=256000 tg=64 tgbytes=16 expect=in0:2048,in1:9216000,out0:32000 tag=aten::matmul
rtou 0 0
utgid 1
simdlane 2
simdw 4
simdgid 3
umulk 6 3 4
umulk 5 4 8
uconst 8 @in1
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 2048
loadconstf32 24 32000
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 16
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in0+0
uloadf32 6 7 @in0+1
uloadf32 7 7 @in0+2
uloadf32 8 7 @in0+3
uloadf32 9 7 @in0+4
uloadf32 10 7 @in0+5
uloadf32 11 7 @in0+6
uloadf32 12 7 @in0+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+8192000
uloadf32 19 12 @in1+8704000
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+8192000
uloadf32 19 12 @in1+8704000
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+8192000
uloadf32 19 12 @in1+8704000
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+8192000
uloadf32 19 12 @in1+8704000
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend 
uadd 7 7 5
forend 
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out0
ifend 
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out0
ifend 
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out0
ifend 
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out0
ifend 
ifend 
endkernel
kernel s.matmul_q4_simd_2 in=2 out=1 threads=16384 tg=64 tgbytes=16 expect=in0:2048,in1:589824,out0:2048 tag=aten::matmul
rtou 0 0
utgid 1
simdlane 2
simdw 4
simdgid 3
umulk 6 3 4
umulk 5 4 8
uconst 8 @in1
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 2048
loadconstf32 24 2048
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 16
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in0+0
uloadf32 6 7 @in0+1
uloadf32 7 7 @in0+2
uloadf32 8 7 @in0+3
uloadf32 9 7 @in0+4
uloadf32 10 7 @in0+5
uloadf32 11 7 @in0+6
uloadf32 12 7 @in0+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend 
uadd 7 7 5
forend 
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out0
ifend 
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out0
ifend 
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out0
ifend 
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out0
ifend 
ifend 
endkernel
kernel s.attention in=4 out=1 threads=4096 tg=128 tgbytes=8704 expect=in0:2048,in1:524288,in2:524288,in3:1,out0:2048 tag=aten::scaled_dot_product_attention
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
ushrk 2 1 3
umulk 4 1 64
umulk 5 2 131072
rtou 9 2
uaddk 9 9 127
ushrk 9 9 7
utor 10 9
mulconst 4 15 1
loadconstf32 5 0
uaddk 6 3 0
loadconstf32 3 0
loadconstf32 16 16
forbegin 16
utor 3 6
cmplt 9 3 2
ifbegin 9
umulk 7 6 64
uadd 7 7 5
uaddk 8 4 0
loadconstf32 6 0
loadconstf32 17 64
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
tgstoref32 4 1 2048
barriertg 
loadconstf32 16 64
cmplt 18 1 16
ifbegin 18
addconst 17 1 64
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 32
cmplt 18 1 16
ifbegin 18
addconst 17 1 32
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 16
cmplt 18 1 16
ifbegin 18
addconst 17 1 16
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 8
cmplt 18 1 16
ifbegin 18
addconst 17 1 8
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 4
cmplt 18 1 16
ifbegin 18
addconst 17 1 4
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 2
cmplt 18 1 16
ifbegin 18
addconst 17 1 2
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 1
cmplt 18 1 16
ifbegin 18
addconst 17 1 1
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
tgloadf32 11 13 2048
barriertg 
uaddk 6 3 0
loadconstf32 16 16
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
tgstoref32 5 1 2048
barriertg 
loadconstf32 16 64
cmplt 18 1 16
ifbegin 18
addconst 17 1 64
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 32
cmplt 18 1 16
ifbegin 18
addconst 17 1 32
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 16
cmplt 18 1 16
ifbegin 18
addconst 17 1 16
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 8
cmplt 18 1 16
ifbegin 18
addconst 17 1 8
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 4
cmplt 18 1 16
ifbegin 18
addconst 17 1 4
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 2
cmplt 18 1 16
ifbegin 18
addconst 17 1 2
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 1
cmplt 18 1 16
ifbegin 18
addconst 17 1 1
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
tgloadf32 12 13 2048
barriertg 
loadconstf32 16 64
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
uaddk 7 7 64
forend 
div 6 6 12
uadd 10 4 3
ustoref32 6 10 @out0
ifend 
endkernel
kernel s.add_rmsnorm in=3 out=2 threads=256 tg=256 tgbytes=1024 expect=in0:2048,in1:2048,in2:2048,out0:2048,out1:2048 tag=aten::add
rtou 0 0
loadconstf32 4 0
loadconstf32 7 2048
utgid 1
ulid 2
loadlid 1
umulk 3 1 2048
uaddk 4 2 0
loadconstf32 8 8
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
mulconst 8 8 0.00048828125
addconst 8 8 0.00000999999975
rsqrt 8 8
uaddk 4 2 0
loadconstf32 10 8
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
kernel s.silu_mul_3 in=2 out=1 threads=5632 tg=64 tgbytes=16 expect=in0:5632,in1:5632,out0:5632 rows=1:5632 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 2 1
mul 1 1 2
uloadf32 3 0 @in1
mul 1 1 3
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_simd_5 in=2 out=1 threads=16384 tg=64 tgbytes=16 expect=in0:5632,in1:1622016,out0:2048 tag=aten::matmul
rtou 0 0
utgid 1
simdlane 2
simdw 4
simdgid 3
umulk 6 3 4
umulk 5 4 8
uconst 8 @in1
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 5632
loadconstf32 24 2048
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 64
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in0+0
uloadf32 6 7 @in0+1
uloadf32 7 7 @in0+2
uloadf32 8 7 @in0+3
uloadf32 9 7 @in0+4
uloadf32 10 7 @in0+5
uloadf32 11 7 @in0+6
uloadf32 12 7 @in0+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 2816
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 44
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 2816
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 44
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 2816
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 44
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 2816
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 44
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend 
uadd 7 7 5
forend 
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out0
ifend 
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out0
ifend 
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out0
ifend 
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out0
ifend 
ifend 
endkernel
kernel s.side in=6 out=3 threads=20480 tg=64 tgbytes=16 expect=in0:2048,in1:589824,in2:2048,in3:73728,in4:2048,in5:73728,out0:2048,out1:256,out2:256
rtou 0 0
utor 27 0
loadconstf32 28 16384
cmplt 28 27 28
loadconstf32 29 0
cmplt 29 27 29
sub 28 28 29
ifbegin 28
utgid 1
simdlane 2
simdw 4
simdgid 3
umulk 6 3 4
umulk 5 4 8
uconst 8 @in1
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 2048
loadconstf32 24 2048
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 16
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in0+0
uloadf32 6 7 @in0+1
uloadf32 7 7 @in0+2
uloadf32 8 7 @in0+3
uloadf32 9 7 @in0+4
uloadf32 10 7 @in0+5
uloadf32 11 7 @in0+6
uloadf32 12 7 @in0+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend
uadd 7 7 5
forend
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out0
ifend
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out0
ifend
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out0
ifend
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out0
ifend
ifend
loadconstf32 27 -1
ifend
loadconstf32 28 18432
cmplt 28 27 28
loadconstf32 29 16384
cmplt 29 27 29
sub 28 28 29
ifbegin 28
uaddk 0 0 4294950912
utgid 1
uaddk 1 1 4294967040
simdlane 2
simdw 4
simdgid 3
simdw 17
utor 27 3
utor 28 17
loadconstf32 29 16384
div 29 29 28
sub 27 27 29
rtou 3 27
umulk 6 3 4
umulk 5 4 8
uconst 8 @in3
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 2048
loadconstf32 24 256
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 16
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in2+0
uloadf32 6 7 @in2+1
uloadf32 7 7 @in2+2
uloadf32 8 7 @in2+3
uloadf32 9 7 @in2+4
uloadf32 10 7 @in2+5
uloadf32 11 7 @in2+6
uloadf32 12 7 @in2+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in3+65536
uloadf32 19 12 @in3+69632
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in3+65536
uloadf32 19 12 @in3+69632
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in3+65536
uloadf32 19 12 @in3+69632
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in3+65536
uloadf32 19 12 @in3+69632
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend
uadd 7 7 5
forend
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out1
ifend
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out1
ifend
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out1
ifend
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out1
ifend
ifend
loadconstf32 27 -1
ifend
loadconstf32 28 20480
cmplt 28 27 28
loadconstf32 29 18432
cmplt 29 27 29
sub 28 28 29
ifbegin 28
uaddk 0 0 4294948864
utgid 1
uaddk 1 1 4294967008
simdlane 2
simdw 4
simdgid 3
simdw 17
utor 27 3
utor 28 17
loadconstf32 29 18432
div 29 29 28
sub 27 27 29
rtou 3 27
umulk 6 3 4
umulk 5 4 8
uconst 8 @in5
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 2048
loadconstf32 24 256
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 16
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in4+0
uloadf32 6 7 @in4+1
uloadf32 7 7 @in4+2
uloadf32 8 7 @in4+3
uloadf32 9 7 @in4+4
uloadf32 10 7 @in4+5
uloadf32 11 7 @in4+6
uloadf32 12 7 @in4+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in5+65536
uloadf32 19 12 @in5+69632
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in5+65536
uloadf32 19 12 @in5+69632
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in5+65536
uloadf32 19 12 @in5+69632
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in5+65536
uloadf32 19 12 @in5+69632
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend
uadd 7 7 5
forend
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out2
ifend
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out2
ifend
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out2
ifend
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out2
ifend
ifend
loadconstf32 27 -1
ifend
endkernel
kernel s.side_2 in=4 out=2 threads=1152 tg=64 tgbytes=16 expect=in0:2048,in1:1,in2:256,in3:1,out0:2048,out1:256
rtou 0 0
utor 16 0
loadconstf32 17 1024
cmplt 17 16 17
loadconstf32 18 0
cmplt 18 16 18
sub 17 17 18
ifbegin 17
uconst 1 0
uandk 2 0 31
ushrk 3 0 5
umulk 5 3 64
uadd 5 5 2
uloadf32 1 1 @in1
utor 9 2
mulconst 9 9 -0.28782313662426
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
uloadf32 7 5 @in0+32
mul 8 6 5
mul 3 7 4
sub 8 8 3
mulconst 8 8 0.125
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
mulconst 8 8 0.125
ustoref32 8 5 @out0+32
loadconstf32 16 -1
ifend
loadconstf32 17 1152
cmplt 17 16 17
loadconstf32 18 1024
cmplt 18 16 18
sub 17 17 18
ifbegin 17
uaddk 0 0 4294966272
uconst 1 0
uandk 2 0 31
ushrk 3 0 5
umulk 5 3 64
uadd 5 5 2
uloadf32 1 1 @in3
utor 9 2
mulconst 9 9 -0.28782313662426
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
uloadf32 7 5 @in2+32
mul 8 6 5
mul 3 7 4
sub 8 8 3
ustoref32 8 5 @out1
mul 8 7 5
mul 3 6 4
add 8 8 3
ustoref32 8 5 @out1+32
loadconstf32 16 -1
ifend
endkernel
kernel s.side_3 in=6 out=2 threads=512 tg=64 tgbytes=16 expect=in0:256,in1:1,in2:524288,in3:256,in4:1,in5:524288,out0:524288,out1:524288
rtou 0 0
utor 9 0
loadconstf32 10 256
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
mulconst 3 3 0.015625
rtou 3 3
utor 4 0
utor 5 3
mulconst 5 5 64
sub 4 4 5
rtou 6 4
utor 6 3
addconst 6 6 0.5
mulconst 6 6 0.25
rtou 4 6
utor 7 3
utor 8 4
mulconst 8 8 4
sub 7 7 8
rtou 5 7
uadd 7 2 4
umulk 5 5 2048
uadd 7 7 5
umulk 7 7 64
uadd 7 7 6
uloadf32 2 0 @in0
ustoref32 2 7 @in2
loadconstf32 9 -1
ifend
loadconstf32 10 512
cmplt 10 9 10
loadconstf32 11 256
cmplt 11 9 11
sub 10 10 11
ifbegin 10
uaddk 0 0 4294967040
uconst 1 0
uloadf32 1 1 @in4
rtou 2 1
utor 3 0
addconst 3 3 0.5
mulconst 3 3 0.015625
rtou 3 3
utor 4 0
utor 5 3
mulconst 5 5 64
sub 4 4 5
rtou 6 4
utor 6 3
addconst 6 6 0.5
mulconst 6 6 0.25
rtou 4 6
utor 7 3
utor 8 4
mulconst 8 8 4
sub 7 7 8
rtou 5 7
uadd 7 2 4
umulk 5 5 2048
uadd 7 7 5
umulk 7 7 64
uadd 7 7 6
uloadf32 2 0 @in3
ustoref32 2 7 @in5
loadconstf32 9 -1
ifend
endkernel
kernel s.side_4 in=4 out=2 threads=90112 tg=64 tgbytes=16 expect=in0:2048,in1:1622016,in2:2048,in3:1622016,out0:5632,out1:5632
rtou 0 0
utor 27 0
loadconstf32 28 45056
cmplt 28 27 28
loadconstf32 29 0
cmplt 29 27 29
sub 28 28 29
ifbegin 28
utgid 1
simdlane 2
simdw 4
simdgid 3
umulk 6 3 4
umulk 5 4 8
uconst 8 @in1
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 2048
loadconstf32 24 5632
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 16
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in0+0
uloadf32 6 7 @in0+1
uloadf32 7 7 @in0+2
uloadf32 8 7 @in0+3
uloadf32 9 7 @in0+4
uloadf32 10 7 @in0+5
uloadf32 11 7 @in0+6
uloadf32 12 7 @in0+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend
uadd 7 7 5
forend
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out0
ifend
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out0
ifend
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out0
ifend
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out0
ifend
ifend
loadconstf32 27 -1
ifend
loadconstf32 28 90112
cmplt 28 27 28
loadconstf32 29 45056
cmplt 29 27 29
sub 28 28 29
ifbegin 28
uaddk 0 0 4294922240
utgid 1
uaddk 1 1 4294966592
simdlane 2
simdw 4
simdgid 3
simdw 17
utor 27 3
utor 28 17
loadconstf32 29 45056
div 29 29 28
sub 27 27 29
rtou 3 27
umulk 6 3 4
umulk 5 4 8
uconst 8 @in3
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 2048
loadconstf32 24 5632
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 16
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in2+0
uloadf32 6 7 @in2+1
uloadf32 7 7 @in2+2
uloadf32 8 7 @in2+3
uloadf32 9 7 @in2+4
uloadf32 10 7 @in2+5
uloadf32 11 7 @in2+6
uloadf32 12 7 @in2+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in3+1441792
uloadf32 19 12 @in3+1531904
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in3+1441792
uloadf32 19 12 @in3+1531904
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in3+1441792
uloadf32 19 12 @in3+1531904
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in3+1441792
uloadf32 19 12 @in3+1531904
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend
uadd 7 7 5
forend
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out1
ifend
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out1
ifend
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out1
ifend
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out1
ifend
ifend
loadconstf32 27 -1
ifend
endkernel
phase chunk1 rows=64 partial
node embedding_p1 ext fir.embedding token model_embed_tokens_weight_emb -> t718_embedding
node rmsnorm_0 micro s.rmsnorm t718_embedding model_layers_0_input_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_0 micro s.matmul_q4_tiled_fma t722_y model_layers_0_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_tiled_fma_1 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_0_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_tiled_fma_2 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_0_self_attn_v_proj_weight_q -> t728_c
node rope_0 micro s.rope t724_c position -> t724_c
node rope_1 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_0 micro s.kv_scatter t726_c position s510 -> s510
node kv_scatter_1 micro s.kv_scatter t728_c position s511 -> s511
node flash_prefill_0 micro s.flash_prefill t724_c s510 s511 position -> t722_y
node matmul_q4_tiled_fma_3 micro s.matmul_q4_tiled_fma t722_y model_layers_0_self_attn_o_proj_weight_q -> t724_c
node vec_add_0 micro s.vec_add t718_embedding t724_c -> t722_y
node rmsnorm_1 micro s.rmsnorm t722_y model_layers_0_post_attention_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_4 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_0_mlp_gate_proj_weight_q -> t742_c
node matmul_q4_tiled_fma_5 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_0_mlp_up_proj_weight_q -> t744_c
node act_sigmoid_0 micro s.act_sigmoid t742_c -> t746_y
node vec_mul_0 micro s.vec_mul t742_c t746_y -> t748_c
node vec_mul_1 micro s.vec_mul t748_c t744_c -> t746_y
node matmul_q4_tiled_fma_6 micro s.matmul_q4_tiled_fma_4 t746_y model_layers_0_mlp_down_proj_weight_q -> t724_c
node vec_add_1 micro s.vec_add t722_y t724_c -> t718_embedding
node rmsnorm_2 micro s.rmsnorm t718_embedding model_layers_1_input_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_7 micro s.matmul_q4_tiled_fma t724_c model_layers_1_self_attn_q_proj_weight_q -> t722_y
node matmul_q4_tiled_fma_8 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_1_self_attn_k_proj_weight_q -> t728_c
node matmul_q4_tiled_fma_9 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_1_self_attn_v_proj_weight_q -> t726_c
node rope_2 micro s.rope t722_y position -> t722_y
node rope_3 micro s.rope_2 t728_c position -> t728_c
node kv_scatter_2 micro s.kv_scatter t728_c position s512 -> s512
node kv_scatter_3 micro s.kv_scatter t726_c position s513 -> s513
node flash_prefill_1 micro s.flash_prefill t722_y s512 s513 position -> t724_c
node matmul_q4_tiled_fma_10 micro s.matmul_q4_tiled_fma t724_c model_layers_1_self_attn_o_proj_weight_q -> t722_y
node vec_add_2 micro s.vec_add t718_embedding t722_y -> t724_c
node rmsnorm_3 micro s.rmsnorm t724_c model_layers_1_post_attention_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_11 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_1_mlp_gate_proj_weight_q -> t746_y
node matmul_q4_tiled_fma_12 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_1_mlp_up_proj_weight_q -> t748_c
node act_sigmoid_1 micro s.act_sigmoid t746_y -> t744_c
node vec_mul_2 micro s.vec_mul t746_y t744_c -> t742_c
node vec_mul_3 micro s.vec_mul t742_c t748_c -> t744_c
node matmul_q4_tiled_fma_13 micro s.matmul_q4_tiled_fma_4 t744_c model_layers_1_mlp_down_proj_weight_q -> t722_y
node vec_add_3 micro s.vec_add t724_c t722_y -> t718_embedding
node rmsnorm_4 micro s.rmsnorm t718_embedding model_layers_2_input_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_14 micro s.matmul_q4_tiled_fma t722_y model_layers_2_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_tiled_fma_15 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_2_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_tiled_fma_16 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_2_self_attn_v_proj_weight_q -> t728_c
node rope_4 micro s.rope t724_c position -> t724_c
node rope_5 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_4 micro s.kv_scatter t726_c position s514 -> s514
node kv_scatter_5 micro s.kv_scatter t728_c position s515 -> s515
node flash_prefill_2 micro s.flash_prefill t724_c s514 s515 position -> t722_y
node matmul_q4_tiled_fma_17 micro s.matmul_q4_tiled_fma t722_y model_layers_2_self_attn_o_proj_weight_q -> t724_c
node vec_add_4 micro s.vec_add t718_embedding t724_c -> t722_y
node rmsnorm_5 micro s.rmsnorm t722_y model_layers_2_post_attention_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_18 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_2_mlp_gate_proj_weight_q -> t744_c
node matmul_q4_tiled_fma_19 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_2_mlp_up_proj_weight_q -> t742_c
node act_sigmoid_2 micro s.act_sigmoid t744_c -> t748_c
node vec_mul_4 micro s.vec_mul t744_c t748_c -> t746_y
node vec_mul_5 micro s.vec_mul t746_y t742_c -> t748_c
node matmul_q4_tiled_fma_20 micro s.matmul_q4_tiled_fma_4 t748_c model_layers_2_mlp_down_proj_weight_q -> t724_c
node vec_add_5 micro s.vec_add t722_y t724_c -> t718_embedding
node rmsnorm_6 micro s.rmsnorm t718_embedding model_layers_3_input_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_21 micro s.matmul_q4_tiled_fma t724_c model_layers_3_self_attn_q_proj_weight_q -> t722_y
node matmul_q4_tiled_fma_22 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_3_self_attn_k_proj_weight_q -> t728_c
node matmul_q4_tiled_fma_23 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_3_self_attn_v_proj_weight_q -> t726_c
node rope_6 micro s.rope t722_y position -> t722_y
node rope_7 micro s.rope_2 t728_c position -> t728_c
node kv_scatter_6 micro s.kv_scatter t728_c position s516 -> s516
node kv_scatter_7 micro s.kv_scatter t726_c position s517 -> s517
node flash_prefill_3 micro s.flash_prefill t722_y s516 s517 position -> t724_c
node matmul_q4_tiled_fma_24 micro s.matmul_q4_tiled_fma t724_c model_layers_3_self_attn_o_proj_weight_q -> t722_y
node vec_add_6 micro s.vec_add t718_embedding t722_y -> t724_c
node rmsnorm_7 micro s.rmsnorm t724_c model_layers_3_post_attention_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_25 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_3_mlp_gate_proj_weight_q -> t748_c
node matmul_q4_tiled_fma_26 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_3_mlp_up_proj_weight_q -> t746_y
node act_sigmoid_3 micro s.act_sigmoid t748_c -> t742_c
node vec_mul_6 micro s.vec_mul t748_c t742_c -> t744_c
node vec_mul_7 micro s.vec_mul t744_c t746_y -> t742_c
node matmul_q4_tiled_fma_27 micro s.matmul_q4_tiled_fma_4 t742_c model_layers_3_mlp_down_proj_weight_q -> t722_y
node vec_add_7 micro s.vec_add t724_c t722_y -> t718_embedding
node rmsnorm_8 micro s.rmsnorm t718_embedding model_layers_4_input_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_28 micro s.matmul_q4_tiled_fma t722_y model_layers_4_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_tiled_fma_29 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_4_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_tiled_fma_30 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_4_self_attn_v_proj_weight_q -> t728_c
node rope_8 micro s.rope t724_c position -> t724_c
node rope_9 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_8 micro s.kv_scatter t726_c position s518 -> s518
node kv_scatter_9 micro s.kv_scatter t728_c position s519 -> s519
node flash_prefill_4 micro s.flash_prefill t724_c s518 s519 position -> t722_y
node matmul_q4_tiled_fma_31 micro s.matmul_q4_tiled_fma t722_y model_layers_4_self_attn_o_proj_weight_q -> t724_c
node vec_add_8 micro s.vec_add t718_embedding t724_c -> t722_y
node rmsnorm_9 micro s.rmsnorm t722_y model_layers_4_post_attention_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_32 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_4_mlp_gate_proj_weight_q -> t742_c
node matmul_q4_tiled_fma_33 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_4_mlp_up_proj_weight_q -> t744_c
node act_sigmoid_4 micro s.act_sigmoid t742_c -> t746_y
node vec_mul_8 micro s.vec_mul t742_c t746_y -> t748_c
node vec_mul_9 micro s.vec_mul t748_c t744_c -> t746_y
node matmul_q4_tiled_fma_34 micro s.matmul_q4_tiled_fma_4 t746_y model_layers_4_mlp_down_proj_weight_q -> t724_c
node vec_add_9 micro s.vec_add t722_y t724_c -> t718_embedding
node rmsnorm_10 micro s.rmsnorm t718_embedding model_layers_5_input_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_35 micro s.matmul_q4_tiled_fma t724_c model_layers_5_self_attn_q_proj_weight_q -> t722_y
node matmul_q4_tiled_fma_36 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_5_self_attn_k_proj_weight_q -> t728_c
node matmul_q4_tiled_fma_37 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_5_self_attn_v_proj_weight_q -> t726_c
node rope_10 micro s.rope t722_y position -> t722_y
node rope_11 micro s.rope_2 t728_c position -> t728_c
node kv_scatter_10 micro s.kv_scatter t728_c position s520 -> s520
node kv_scatter_11 micro s.kv_scatter t726_c position s521 -> s521
node flash_prefill_5 micro s.flash_prefill t722_y s520 s521 position -> t724_c
node matmul_q4_tiled_fma_38 micro s.matmul_q4_tiled_fma t724_c model_layers_5_self_attn_o_proj_weight_q -> t722_y
node vec_add_10 micro s.vec_add t718_embedding t722_y -> t724_c
node rmsnorm_11 micro s.rmsnorm t724_c model_layers_5_post_attention_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_39 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_5_mlp_gate_proj_weight_q -> t746_y
node matmul_q4_tiled_fma_40 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_5_mlp_up_proj_weight_q -> t748_c
node act_sigmoid_5 micro s.act_sigmoid t746_y -> t744_c
node vec_mul_10 micro s.vec_mul t746_y t744_c -> t742_c
node vec_mul_11 micro s.vec_mul t742_c t748_c -> t744_c
node matmul_q4_tiled_fma_41 micro s.matmul_q4_tiled_fma_4 t744_c model_layers_5_mlp_down_proj_weight_q -> t722_y
node vec_add_11 micro s.vec_add t724_c t722_y -> t718_embedding
node rmsnorm_12 micro s.rmsnorm t718_embedding model_layers_6_input_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_42 micro s.matmul_q4_tiled_fma t722_y model_layers_6_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_tiled_fma_43 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_6_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_tiled_fma_44 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_6_self_attn_v_proj_weight_q -> t728_c
node rope_12 micro s.rope t724_c position -> t724_c
node rope_13 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_12 micro s.kv_scatter t726_c position s522 -> s522
node kv_scatter_13 micro s.kv_scatter t728_c position s523 -> s523
node flash_prefill_6 micro s.flash_prefill t724_c s522 s523 position -> t722_y
node matmul_q4_tiled_fma_45 micro s.matmul_q4_tiled_fma t722_y model_layers_6_self_attn_o_proj_weight_q -> t724_c
node vec_add_12 micro s.vec_add t718_embedding t724_c -> t722_y
node rmsnorm_13 micro s.rmsnorm t722_y model_layers_6_post_attention_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_46 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_6_mlp_gate_proj_weight_q -> t744_c
node matmul_q4_tiled_fma_47 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_6_mlp_up_proj_weight_q -> t742_c
node act_sigmoid_6 micro s.act_sigmoid t744_c -> t748_c
node vec_mul_12 micro s.vec_mul t744_c t748_c -> t746_y
node vec_mul_13 micro s.vec_mul t746_y t742_c -> t748_c
node matmul_q4_tiled_fma_48 micro s.matmul_q4_tiled_fma_4 t748_c model_layers_6_mlp_down_proj_weight_q -> t724_c
node vec_add_13 micro s.vec_add t722_y t724_c -> t718_embedding
node rmsnorm_14 micro s.rmsnorm t718_embedding model_layers_7_input_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_49 micro s.matmul_q4_tiled_fma t724_c model_layers_7_self_attn_q_proj_weight_q -> t722_y
node matmul_q4_tiled_fma_50 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_7_self_attn_k_proj_weight_q -> t728_c
node matmul_q4_tiled_fma_51 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_7_self_attn_v_proj_weight_q -> t726_c
node rope_14 micro s.rope t722_y position -> t722_y
node rope_15 micro s.rope_2 t728_c position -> t728_c
node kv_scatter_14 micro s.kv_scatter t728_c position s524 -> s524
node kv_scatter_15 micro s.kv_scatter t726_c position s525 -> s525
node flash_prefill_7 micro s.flash_prefill t722_y s524 s525 position -> t724_c
node matmul_q4_tiled_fma_52 micro s.matmul_q4_tiled_fma t724_c model_layers_7_self_attn_o_proj_weight_q -> t722_y
node vec_add_14 micro s.vec_add t718_embedding t722_y -> t724_c
node rmsnorm_15 micro s.rmsnorm t724_c model_layers_7_post_attention_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_53 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_7_mlp_gate_proj_weight_q -> t748_c
node matmul_q4_tiled_fma_54 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_7_mlp_up_proj_weight_q -> t746_y
node act_sigmoid_7 micro s.act_sigmoid t748_c -> t742_c
node vec_mul_14 micro s.vec_mul t748_c t742_c -> t744_c
node vec_mul_15 micro s.vec_mul t744_c t746_y -> t742_c
node matmul_q4_tiled_fma_55 micro s.matmul_q4_tiled_fma_4 t742_c model_layers_7_mlp_down_proj_weight_q -> t722_y
node vec_add_15 micro s.vec_add t724_c t722_y -> t718_embedding
node rmsnorm_16 micro s.rmsnorm t718_embedding model_layers_8_input_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_56 micro s.matmul_q4_tiled_fma t722_y model_layers_8_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_tiled_fma_57 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_8_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_tiled_fma_58 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_8_self_attn_v_proj_weight_q -> t728_c
node rope_16 micro s.rope t724_c position -> t724_c
node rope_17 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_16 micro s.kv_scatter t726_c position s526 -> s526
node kv_scatter_17 micro s.kv_scatter t728_c position s527 -> s527
node flash_prefill_8 micro s.flash_prefill t724_c s526 s527 position -> t722_y
node matmul_q4_tiled_fma_59 micro s.matmul_q4_tiled_fma t722_y model_layers_8_self_attn_o_proj_weight_q -> t724_c
node vec_add_16 micro s.vec_add t718_embedding t724_c -> t722_y
node rmsnorm_17 micro s.rmsnorm t722_y model_layers_8_post_attention_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_60 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_8_mlp_gate_proj_weight_q -> t742_c
node matmul_q4_tiled_fma_61 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_8_mlp_up_proj_weight_q -> t744_c
node act_sigmoid_8 micro s.act_sigmoid t742_c -> t746_y
node vec_mul_16 micro s.vec_mul t742_c t746_y -> t748_c
node vec_mul_17 micro s.vec_mul t748_c t744_c -> t746_y
node matmul_q4_tiled_fma_62 micro s.matmul_q4_tiled_fma_4 t746_y model_layers_8_mlp_down_proj_weight_q -> t724_c
node vec_add_17 micro s.vec_add t722_y t724_c -> t718_embedding
node rmsnorm_18 micro s.rmsnorm t718_embedding model_layers_9_input_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_63 micro s.matmul_q4_tiled_fma t724_c model_layers_9_self_attn_q_proj_weight_q -> t722_y
node matmul_q4_tiled_fma_64 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_9_self_attn_k_proj_weight_q -> t728_c
node matmul_q4_tiled_fma_65 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_9_self_attn_v_proj_weight_q -> t726_c
node rope_18 micro s.rope t722_y position -> t722_y
node rope_19 micro s.rope_2 t728_c position -> t728_c
node kv_scatter_18 micro s.kv_scatter t728_c position s528 -> s528
node kv_scatter_19 micro s.kv_scatter t726_c position s529 -> s529
node flash_prefill_9 micro s.flash_prefill t722_y s528 s529 position -> t724_c
node matmul_q4_tiled_fma_66 micro s.matmul_q4_tiled_fma t724_c model_layers_9_self_attn_o_proj_weight_q -> t722_y
node vec_add_18 micro s.vec_add t718_embedding t722_y -> t724_c
node rmsnorm_19 micro s.rmsnorm t724_c model_layers_9_post_attention_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_67 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_9_mlp_gate_proj_weight_q -> t746_y
node matmul_q4_tiled_fma_68 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_9_mlp_up_proj_weight_q -> t748_c
node act_sigmoid_9 micro s.act_sigmoid t746_y -> t744_c
node vec_mul_18 micro s.vec_mul t746_y t744_c -> t742_c
node vec_mul_19 micro s.vec_mul t742_c t748_c -> t744_c
node matmul_q4_tiled_fma_69 micro s.matmul_q4_tiled_fma_4 t744_c model_layers_9_mlp_down_proj_weight_q -> t722_y
node vec_add_19 micro s.vec_add t724_c t722_y -> t718_embedding
node rmsnorm_20 micro s.rmsnorm t718_embedding model_layers_10_input_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_70 micro s.matmul_q4_tiled_fma t722_y model_layers_10_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_tiled_fma_71 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_10_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_tiled_fma_72 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_10_self_attn_v_proj_weight_q -> t728_c
node rope_20 micro s.rope t724_c position -> t724_c
node rope_21 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_20 micro s.kv_scatter t726_c position s530 -> s530
node kv_scatter_21 micro s.kv_scatter t728_c position s531 -> s531
node flash_prefill_10 micro s.flash_prefill t724_c s530 s531 position -> t722_y
node matmul_q4_tiled_fma_73 micro s.matmul_q4_tiled_fma t722_y model_layers_10_self_attn_o_proj_weight_q -> t724_c
node vec_add_20 micro s.vec_add t718_embedding t724_c -> t722_y
node rmsnorm_21 micro s.rmsnorm t722_y model_layers_10_post_attention_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_74 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_10_mlp_gate_proj_weight_q -> t744_c
node matmul_q4_tiled_fma_75 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_10_mlp_up_proj_weight_q -> t742_c
node silu_mul_0 micro s.silu_mul t744_c t742_c -> t748_c
node matmul_q4_tiled_fma_76 micro s.matmul_q4_tiled_fma_4 t748_c model_layers_10_mlp_down_proj_weight_q -> t724_c
node vec_add_21 micro s.vec_add t722_y t724_c -> t718_embedding
node rmsnorm_22 micro s.rmsnorm t718_embedding model_layers_11_input_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_77 micro s.matmul_q4_tiled_fma t724_c model_layers_11_self_attn_q_proj_weight_q -> t722_y
node matmul_q4_tiled_fma_78 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_11_self_attn_k_proj_weight_q -> t728_c
node matmul_q4_tiled_fma_79 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_11_self_attn_v_proj_weight_q -> t726_c
node rope_22 micro s.rope t722_y position -> t722_y
node rope_23 micro s.rope_2 t728_c position -> t728_c
node kv_scatter_22 micro s.kv_scatter t728_c position s532 -> s532
node kv_scatter_23 micro s.kv_scatter t726_c position s533 -> s533
node flash_prefill_11 micro s.flash_prefill t722_y s532 s533 position -> t724_c
node matmul_q4_tiled_fma_80 micro s.matmul_q4_tiled_fma t724_c model_layers_11_self_attn_o_proj_weight_q -> t722_y
node vec_add_22 micro s.vec_add t718_embedding t722_y -> t724_c
node rmsnorm_23 micro s.rmsnorm t724_c model_layers_11_post_attention_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_81 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_11_mlp_gate_proj_weight_q -> t748_c
node matmul_q4_tiled_fma_82 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_11_mlp_up_proj_weight_q -> t742_c
node silu_mul_1 micro s.silu_mul t748_c t742_c -> t744_c
node matmul_q4_tiled_fma_83 micro s.matmul_q4_tiled_fma_4 t744_c model_layers_11_mlp_down_proj_weight_q -> t722_y
node vec_add_23 micro s.vec_add t724_c t722_y -> t718_embedding
node rmsnorm_24 micro s.rmsnorm t718_embedding model_layers_12_input_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_84 micro s.matmul_q4_tiled_fma t722_y model_layers_12_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_tiled_fma_85 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_12_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_tiled_fma_86 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_12_self_attn_v_proj_weight_q -> t728_c
node rope_24 micro s.rope t724_c position -> t724_c
node rope_25 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_24 micro s.kv_scatter t726_c position s534 -> s534
node kv_scatter_25 micro s.kv_scatter t728_c position s535 -> s535
node flash_prefill_12 micro s.flash_prefill t724_c s534 s535 position -> t722_y
node matmul_q4_tiled_fma_87 micro s.matmul_q4_tiled_fma t722_y model_layers_12_self_attn_o_proj_weight_q -> t724_c
node vec_add_24 micro s.vec_add t718_embedding t724_c -> t722_y
node rmsnorm_25 micro s.rmsnorm t722_y model_layers_12_post_attention_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_88 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_12_mlp_gate_proj_weight_q -> t744_c
node matmul_q4_tiled_fma_89 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_12_mlp_up_proj_weight_q -> t742_c
node silu_mul_2 micro s.silu_mul t744_c t742_c -> t748_c
node matmul_q4_tiled_fma_90 micro s.matmul_q4_tiled_fma_4 t748_c model_layers_12_mlp_down_proj_weight_q -> t724_c
node vec_add_25 micro s.vec_add t722_y t724_c -> t718_embedding
node rmsnorm_26 micro s.rmsnorm t718_embedding model_layers_13_input_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_91 micro s.matmul_q4_tiled_fma t724_c model_layers_13_self_attn_q_proj_weight_q -> t722_y
node matmul_q4_tiled_fma_92 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_13_self_attn_k_proj_weight_q -> t728_c
node matmul_q4_tiled_fma_93 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_13_self_attn_v_proj_weight_q -> t726_c
node rope_26 micro s.rope t722_y position -> t722_y
node rope_27 micro s.rope_2 t728_c position -> t728_c
node kv_scatter_26 micro s.kv_scatter t728_c position s536 -> s536
node kv_scatter_27 micro s.kv_scatter t726_c position s537 -> s537
node flash_prefill_13 micro s.flash_prefill t722_y s536 s537 position -> t724_c
node matmul_q4_tiled_fma_94 micro s.matmul_q4_tiled_fma t724_c model_layers_13_self_attn_o_proj_weight_q -> t722_y
node vec_add_26 micro s.vec_add t718_embedding t722_y -> t724_c
node rmsnorm_27 micro s.rmsnorm t724_c model_layers_13_post_attention_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_95 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_13_mlp_gate_proj_weight_q -> t748_c
node matmul_q4_tiled_fma_96 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_13_mlp_up_proj_weight_q -> t742_c
node silu_mul_3 micro s.silu_mul t748_c t742_c -> t744_c
node matmul_q4_tiled_fma_97 micro s.matmul_q4_tiled_fma_4 t744_c model_layers_13_mlp_down_proj_weight_q -> t722_y
node vec_add_27 micro s.vec_add t724_c t722_y -> t718_embedding
node rmsnorm_28 micro s.rmsnorm t718_embedding model_layers_14_input_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_98 micro s.matmul_q4_tiled_fma t722_y model_layers_14_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_tiled_fma_99 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_14_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_tiled_fma_100 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_14_self_attn_v_proj_weight_q -> t728_c
node rope_28 micro s.rope t724_c position -> t724_c
node rope_29 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_28 micro s.kv_scatter t726_c position s538 -> s538
node kv_scatter_29 micro s.kv_scatter t728_c position s539 -> s539
node flash_prefill_14 micro s.flash_prefill t724_c s538 s539 position -> t722_y
node matmul_q4_tiled_fma_101 micro s.matmul_q4_tiled_fma t722_y model_layers_14_self_attn_o_proj_weight_q -> t724_c
node vec_add_28 micro s.vec_add t718_embedding t724_c -> t722_y
node rmsnorm_29 micro s.rmsnorm t722_y model_layers_14_post_attention_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_102 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_14_mlp_gate_proj_weight_q -> t744_c
node matmul_q4_tiled_fma_103 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_14_mlp_up_proj_weight_q -> t742_c
node silu_mul_4 micro s.silu_mul t744_c t742_c -> t748_c
node matmul_q4_tiled_fma_104 micro s.matmul_q4_tiled_fma_4 t748_c model_layers_14_mlp_down_proj_weight_q -> t724_c
node vec_add_29 micro s.vec_add t722_y t724_c -> t718_embedding
node rmsnorm_30 micro s.rmsnorm t718_embedding model_layers_15_input_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_105 micro s.matmul_q4_tiled_fma t724_c model_layers_15_self_attn_q_proj_weight_q -> t722_y
node matmul_q4_tiled_fma_106 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_15_self_attn_k_proj_weight_q -> t728_c
node matmul_q4_tiled_fma_107 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_15_self_attn_v_proj_weight_q -> t726_c
node rope_30 micro s.rope t722_y position -> t722_y
node rope_31 micro s.rope_2 t728_c position -> t728_c
node kv_scatter_30 micro s.kv_scatter t728_c position s540 -> s540
node kv_scatter_31 micro s.kv_scatter t726_c position s541 -> s541
node flash_prefill_15 micro s.flash_prefill t722_y s540 s541 position -> t724_c
node matmul_q4_tiled_fma_108 micro s.matmul_q4_tiled_fma t724_c model_layers_15_self_attn_o_proj_weight_q -> t722_y
node vec_add_30 micro s.vec_add t718_embedding t722_y -> t724_c
node rmsnorm_31 micro s.rmsnorm t724_c model_layers_15_post_attention_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_109 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_15_mlp_gate_proj_weight_q -> t748_c
node matmul_q4_tiled_fma_110 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_15_mlp_up_proj_weight_q -> t742_c
node silu_mul_5 micro s.silu_mul t748_c t742_c -> t744_c
node matmul_q4_tiled_fma_111 micro s.matmul_q4_tiled_fma_4 t744_c model_layers_15_mlp_down_proj_weight_q -> t722_y
node vec_add_31 micro s.vec_add t724_c t722_y -> t718_embedding
node rmsnorm_32 micro s.rmsnorm t718_embedding model_layers_16_input_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_112 micro s.matmul_q4_tiled_fma t722_y model_layers_16_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_tiled_fma_113 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_16_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_tiled_fma_114 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_16_self_attn_v_proj_weight_q -> t728_c
node rope_32 micro s.rope t724_c position -> t724_c
node rope_33 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_32 micro s.kv_scatter t726_c position s542 -> s542
node kv_scatter_33 micro s.kv_scatter t728_c position s543 -> s543
node flash_prefill_16 micro s.flash_prefill t724_c s542 s543 position -> t722_y
node matmul_q4_tiled_fma_115 micro s.matmul_q4_tiled_fma t722_y model_layers_16_self_attn_o_proj_weight_q -> t724_c
node vec_add_32 micro s.vec_add t718_embedding t724_c -> t722_y
node rmsnorm_33 micro s.rmsnorm t722_y model_layers_16_post_attention_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_116 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_16_mlp_gate_proj_weight_q -> t744_c
node matmul_q4_tiled_fma_117 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_16_mlp_up_proj_weight_q -> t742_c
node act_sigmoid_10 micro s.act_sigmoid t744_c -> t748_c
node vec_mul_20 micro s.vec_mul t744_c t748_c -> t746_y
node vec_mul_21 micro s.vec_mul t746_y t742_c -> t748_c
node matmul_q4_tiled_fma_118 micro s.matmul_q4_tiled_fma_4 t748_c model_layers_16_mlp_down_proj_weight_q -> t724_c
node vec_add_33 micro s.vec_add t722_y t724_c -> t718_embedding
node rmsnorm_34 micro s.rmsnorm t718_embedding model_layers_17_input_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_119 micro s.matmul_q4_tiled_fma t724_c model_layers_17_self_attn_q_proj_weight_q -> t722_y
node matmul_q4_tiled_fma_120 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_17_self_attn_k_proj_weight_q -> t728_c
node matmul_q4_tiled_fma_121 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_17_self_attn_v_proj_weight_q -> t726_c
node rope_34 micro s.rope t722_y position -> t722_y
node rope_35 micro s.rope_2 t728_c position -> t728_c
node kv_scatter_34 micro s.kv_scatter t728_c position s544 -> s544
node kv_scatter_35 micro s.kv_scatter t726_c position s545 -> s545
node flash_prefill_17 micro s.flash_prefill t722_y s544 s545 position -> t724_c
node matmul_q4_tiled_fma_122 micro s.matmul_q4_tiled_fma t724_c model_layers_17_self_attn_o_proj_weight_q -> t722_y
node vec_add_34 micro s.vec_add t718_embedding t722_y -> t724_c
node rmsnorm_35 micro s.rmsnorm t724_c model_layers_17_post_attention_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_123 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_17_mlp_gate_proj_weight_q -> t748_c
node matmul_q4_tiled_fma_124 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_17_mlp_up_proj_weight_q -> t746_y
node act_sigmoid_11 micro s.act_sigmoid t748_c -> t742_c
node vec_mul_22 micro s.vec_mul t748_c t742_c -> t744_c
node vec_mul_23 micro s.vec_mul t744_c t746_y -> t742_c
node matmul_q4_tiled_fma_125 micro s.matmul_q4_tiled_fma_4 t742_c model_layers_17_mlp_down_proj_weight_q -> t722_y
node vec_add_35 micro s.vec_add t724_c t722_y -> t718_embedding
node rmsnorm_36 micro s.rmsnorm t718_embedding model_layers_18_input_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_126 micro s.matmul_q4_tiled_fma t722_y model_layers_18_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_tiled_fma_127 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_18_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_tiled_fma_128 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_18_self_attn_v_proj_weight_q -> t728_c
node rope_36 micro s.rope t724_c position -> t724_c
node rope_37 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_36 micro s.kv_scatter t726_c position s546 -> s546
node kv_scatter_37 micro s.kv_scatter t728_c position s547 -> s547
node flash_prefill_18 micro s.flash_prefill t724_c s546 s547 position -> t722_y
node matmul_q4_tiled_fma_129 micro s.matmul_q4_tiled_fma t722_y model_layers_18_self_attn_o_proj_weight_q -> t724_c
node vec_add_36 micro s.vec_add t718_embedding t724_c -> t722_y
node rmsnorm_37 micro s.rmsnorm t722_y model_layers_18_post_attention_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_130 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_18_mlp_gate_proj_weight_q -> t742_c
node matmul_q4_tiled_fma_131 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_18_mlp_up_proj_weight_q -> t744_c
node act_sigmoid_12 micro s.act_sigmoid t742_c -> t746_y
node vec_mul_24 micro s.vec_mul t742_c t746_y -> t748_c
node vec_mul_25 micro s.vec_mul t748_c t744_c -> t746_y
node matmul_q4_tiled_fma_132 micro s.matmul_q4_tiled_fma_4 t746_y model_layers_18_mlp_down_proj_weight_q -> t724_c
node vec_add_37 micro s.vec_add t722_y t724_c -> t718_embedding
node rmsnorm_38 micro s.rmsnorm t718_embedding model_layers_19_input_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_133 micro s.matmul_q4_tiled_fma t724_c model_layers_19_self_attn_q_proj_weight_q -> t722_y
node matmul_q4_tiled_fma_134 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_19_self_attn_k_proj_weight_q -> t728_c
node matmul_q4_tiled_fma_135 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_19_self_attn_v_proj_weight_q -> t726_c
node rope_38 micro s.rope t722_y position -> t722_y
node rope_39 micro s.rope_2 t728_c position -> t728_c
node kv_scatter_38 micro s.kv_scatter t728_c position s548 -> s548
node kv_scatter_39 micro s.kv_scatter t726_c position s549 -> s549
node flash_prefill_19 micro s.flash_prefill t722_y s548 s549 position -> t724_c
node matmul_q4_tiled_fma_136 micro s.matmul_q4_tiled_fma t724_c model_layers_19_self_attn_o_proj_weight_q -> t722_y
node vec_add_38 micro s.vec_add t718_embedding t722_y -> t724_c
node rmsnorm_39 micro s.rmsnorm t724_c model_layers_19_post_attention_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_137 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_19_mlp_gate_proj_weight_q -> t746_y
node matmul_q4_tiled_fma_138 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_19_mlp_up_proj_weight_q -> t748_c
node silu_mul_6 micro s.silu_mul t746_y t748_c -> t744_c
node matmul_q4_tiled_fma_139 micro s.matmul_q4_tiled_fma_4 t744_c model_layers_19_mlp_down_proj_weight_q -> t722_y
node vec_add_39 micro s.vec_add t724_c t722_y -> t718_embedding
node rmsnorm_40 micro s.rmsnorm t718_embedding model_layers_20_input_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_140 micro s.matmul_q4_tiled_fma t722_y model_layers_20_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_tiled_fma_141 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_20_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_tiled_fma_142 micro s.matmul_q4_tiled_fma_2 t722_y model_layers_20_self_attn_v_proj_weight_q -> t728_c
node rope_40 micro s.rope t724_c position -> t724_c
node rope_41 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_40 micro s.kv_scatter t726_c position s550 -> s550
node kv_scatter_41 micro s.kv_scatter t728_c position s551 -> s551
node flash_prefill_20 micro s.flash_prefill t724_c s550 s551 position -> t722_y
node matmul_q4_tiled_fma_143 micro s.matmul_q4_tiled_fma t722_y model_layers_20_self_attn_o_proj_weight_q -> t724_c
node vec_add_40 micro s.vec_add t718_embedding t724_c -> t722_y
node rmsnorm_41 micro s.rmsnorm t722_y model_layers_20_post_attention_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_144 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_20_mlp_gate_proj_weight_q -> t744_c
node matmul_q4_tiled_fma_145 micro s.matmul_q4_tiled_fma_3 t724_c model_layers_20_mlp_up_proj_weight_q -> t748_c
node silu_mul_7 micro s.silu_mul t744_c t748_c -> t746_y
node matmul_q4_tiled_fma_146 micro s.matmul_q4_tiled_fma_4 t746_y model_layers_20_mlp_down_proj_weight_q -> t724_c
node vec_add_41 micro s.vec_add t722_y t724_c -> t718_embedding
node rmsnorm_42 micro s.rmsnorm t718_embedding model_layers_21_input_layernorm_weight -> t724_c
node matmul_q4_tiled_fma_147 micro s.matmul_q4_tiled_fma t724_c model_layers_21_self_attn_q_proj_weight_q -> t722_y
node matmul_q4_tiled_fma_148 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_21_self_attn_k_proj_weight_q -> t728_c
node matmul_q4_tiled_fma_149 micro s.matmul_q4_tiled_fma_2 t724_c model_layers_21_self_attn_v_proj_weight_q -> t726_c
node rope_42 micro s.rope t722_y position -> t722_y
node rope_43 micro s.rope_2 t728_c position -> t728_c
node kv_scatter_42 micro s.kv_scatter t728_c position s552 -> s552
node kv_scatter_43 micro s.kv_scatter t726_c position s553 -> s553
node flash_prefill_21 micro s.flash_prefill t722_y s552 s553 position -> t724_c
node matmul_q4_tiled_fma_150 micro s.matmul_q4_tiled_fma t724_c model_layers_21_self_attn_o_proj_weight_q -> t722_y
node vec_add_42 micro s.vec_add t718_embedding t722_y -> t724_c
node rmsnorm_43 micro s.rmsnorm t724_c model_layers_21_post_attention_layernorm_weight -> t722_y
node matmul_q4_tiled_fma_151 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_21_mlp_gate_proj_weight_q -> t746_y
node matmul_q4_tiled_fma_152 micro s.matmul_q4_tiled_fma_3 t722_y model_layers_21_mlp_up_proj_weight_q -> t748_c
node silu_mul_8 micro s.silu_mul t746_y t748_c -> t744_c
node matmul_q4_tiled_fma_153 micro s.matmul_q4_tiled_fma_4 t744_c model_layers_21_mlp_down_proj_weight_q -> t722_y
node vec_add_43 micro s.vec_add t724_c t722_y -> t718_embedding
node last_row_0 session sessionCopyInto:srcOff=0,dstOff=0,n=2048,lastRowOf=64 t718_embedding -> s694
phase chunk2 rows=448 last=t2931_c partial
node embedding_p2 ext fir.embedding token model_embed_tokens_weight_emb -> t1818_embedding
node rmsnorm_44 micro s.rmsnorm_2 t1818_embedding model_layers_0_input_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_154 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_0_self_attn_q_proj_weight_q -> t1824_c
node matmul_q4_tiled_fma_155 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_0_self_attn_k_proj_weight_q -> t1826_c
node matmul_q4_tiled_fma_156 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_0_self_attn_v_proj_weight_q -> t1828_c
node rope_44 micro s.rope_3 t1824_c position -> t1824_c
node rope_45 micro s.rope_4 t1826_c position -> t1826_c
node kv_scatter_44 micro s.kv_scatter_2 t1826_c position s510 -> s510
node kv_scatter_45 micro s.kv_scatter_2 t1828_c position s511 -> s511
node flash_prefill_22 micro s.flash_prefill_2 t1824_c s510 s511 position -> t1822_y
node matmul_q4_tiled_fma_157 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_0_self_attn_o_proj_weight_q -> t1824_c
node vec_add_44 micro s.vec_add_2 t1818_embedding t1824_c -> t1822_y
node rmsnorm_45 micro s.rmsnorm_2 t1822_y model_layers_0_post_attention_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_158 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_0_mlp_gate_proj_weight_q -> t1842_c
node matmul_q4_tiled_fma_159 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_0_mlp_up_proj_weight_q -> t1844_c
node silu_mul_9 micro s.silu_mul_2 t1842_c t1844_c -> t1850_c
node matmul_q4_tiled_fma_160 micro s.matmul_q4_tiled_fma_8 t1850_c model_layers_0_mlp_down_proj_weight_q -> t1824_c
node vec_add_45 micro s.vec_add_2 t1822_y t1824_c -> t1818_embedding
node rmsnorm_46 micro s.rmsnorm_2 t1818_embedding model_layers_1_input_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_161 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_1_self_attn_q_proj_weight_q -> t1822_y
node matmul_q4_tiled_fma_162 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_1_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_tiled_fma_163 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_1_self_attn_v_proj_weight_q -> t1826_c
node rope_46 micro s.rope_3 t1822_y position -> t1822_y
node rope_47 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_46 micro s.kv_scatter_2 t1828_c position s512 -> s512
node kv_scatter_47 micro s.kv_scatter_2 t1826_c position s513 -> s513
node flash_prefill_23 micro s.flash_prefill_2 t1822_y s512 s513 position -> t1824_c
node matmul_q4_tiled_fma_164 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_1_self_attn_o_proj_weight_q -> t1822_y
node vec_add_46 micro s.vec_add_2 t1818_embedding t1822_y -> t1824_c
node rmsnorm_47 micro s.rmsnorm_2 t1824_c model_layers_1_post_attention_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_165 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_1_mlp_gate_proj_weight_q -> t1850_c
node matmul_q4_tiled_fma_166 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_1_mlp_up_proj_weight_q -> t1844_c
node act_sigmoid_13 micro s.act_sigmoid_2 t1850_c -> t1842_c
node vec_mul_26 micro s.vec_mul_2 t1850_c t1842_c -> t1899_c
node vec_mul_27 micro s.vec_mul_2 t1899_c t1844_c -> t1842_c
node matmul_q4_tiled_fma_167 micro s.matmul_q4_tiled_fma_8 t1842_c model_layers_1_mlp_down_proj_weight_q -> t1822_y
node vec_add_47 micro s.vec_add_2 t1824_c t1822_y -> t1818_embedding
node rmsnorm_48 micro s.rmsnorm_2 t1818_embedding model_layers_2_input_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_168 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_2_self_attn_q_proj_weight_q -> t1824_c
node matmul_q4_tiled_fma_169 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_2_self_attn_k_proj_weight_q -> t1826_c
node matmul_q4_tiled_fma_170 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_2_self_attn_v_proj_weight_q -> t1828_c
node rope_48 micro s.rope_3 t1824_c position -> t1824_c
node rope_49 micro s.rope_4 t1826_c position -> t1826_c
node kv_scatter_48 micro s.kv_scatter_2 t1826_c position s514 -> s514
node kv_scatter_49 micro s.kv_scatter_2 t1828_c position s515 -> s515
node flash_prefill_24 micro s.flash_prefill_2 t1824_c s514 s515 position -> t1822_y
node matmul_q4_tiled_fma_171 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_2_self_attn_o_proj_weight_q -> t1824_c
node vec_add_48 micro s.vec_add_2 t1818_embedding t1824_c -> t1822_y
node rmsnorm_49 micro s.rmsnorm_2 t1822_y model_layers_2_post_attention_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_172 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_2_mlp_gate_proj_weight_q -> t1842_c
node matmul_q4_tiled_fma_173 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_2_mlp_up_proj_weight_q -> t1899_c
node act_sigmoid_14 micro s.act_sigmoid_2 t1842_c -> t1844_c
node vec_mul_28 micro s.vec_mul_2 t1842_c t1844_c -> t1850_c
node vec_mul_29 micro s.vec_mul_2 t1850_c t1899_c -> t1844_c
node matmul_q4_tiled_fma_174 micro s.matmul_q4_tiled_fma_8 t1844_c model_layers_2_mlp_down_proj_weight_q -> t1824_c
node vec_add_49 micro s.vec_add_2 t1822_y t1824_c -> t1818_embedding
node rmsnorm_50 micro s.rmsnorm_2 t1818_embedding model_layers_3_input_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_175 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_3_self_attn_q_proj_weight_q -> t1822_y
node matmul_q4_tiled_fma_176 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_3_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_tiled_fma_177 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_3_self_attn_v_proj_weight_q -> t1826_c
node rope_50 micro s.rope_3 t1822_y position -> t1822_y
node rope_51 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_50 micro s.kv_scatter_2 t1828_c position s516 -> s516
node kv_scatter_51 micro s.kv_scatter_2 t1826_c position s517 -> s517
node flash_prefill_25 micro s.flash_prefill_2 t1822_y s516 s517 position -> t1824_c
node matmul_q4_tiled_fma_178 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_3_self_attn_o_proj_weight_q -> t1822_y
node vec_add_50 micro s.vec_add_2 t1818_embedding t1822_y -> t1824_c
node rmsnorm_51 micro s.rmsnorm_2 t1824_c model_layers_3_post_attention_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_179 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_3_mlp_gate_proj_weight_q -> t1844_c
node matmul_q4_tiled_fma_180 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_3_mlp_up_proj_weight_q -> t1850_c
node act_sigmoid_15 micro s.act_sigmoid_2 t1844_c -> t1899_c
node vec_mul_30 micro s.vec_mul_2 t1844_c t1899_c -> t1842_c
node vec_mul_31 micro s.vec_mul_2 t1842_c t1850_c -> t1899_c
node matmul_q4_tiled_fma_181 micro s.matmul_q4_tiled_fma_8 t1899_c model_layers_3_mlp_down_proj_weight_q -> t1822_y
node vec_add_51 micro s.vec_add_2 t1824_c t1822_y -> t1818_embedding
node rmsnorm_52 micro s.rmsnorm_2 t1818_embedding model_layers_4_input_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_182 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_4_self_attn_q_proj_weight_q -> t1824_c
node matmul_q4_tiled_fma_183 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_4_self_attn_k_proj_weight_q -> t1826_c
node matmul_q4_tiled_fma_184 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_4_self_attn_v_proj_weight_q -> t1828_c
node rope_52 micro s.rope_3 t1824_c position -> t1824_c
node rope_53 micro s.rope_4 t1826_c position -> t1826_c
node kv_scatter_52 micro s.kv_scatter_2 t1826_c position s518 -> s518
node kv_scatter_53 micro s.kv_scatter_2 t1828_c position s519 -> s519
node flash_prefill_26 micro s.flash_prefill_2 t1824_c s518 s519 position -> t1822_y
node matmul_q4_tiled_fma_185 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_4_self_attn_o_proj_weight_q -> t1824_c
node vec_add_52 micro s.vec_add_2 t1818_embedding t1824_c -> t1822_y
node rmsnorm_53 micro s.rmsnorm_2 t1822_y model_layers_4_post_attention_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_186 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_4_mlp_gate_proj_weight_q -> t1899_c
node matmul_q4_tiled_fma_187 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_4_mlp_up_proj_weight_q -> t1842_c
node act_sigmoid_16 micro s.act_sigmoid_2 t1899_c -> t1850_c
node vec_mul_32 micro s.vec_mul_2 t1899_c t1850_c -> t1844_c
node vec_mul_33 micro s.vec_mul_2 t1844_c t1842_c -> t1850_c
node matmul_q4_tiled_fma_188 micro s.matmul_q4_tiled_fma_8 t1850_c model_layers_4_mlp_down_proj_weight_q -> t1824_c
node vec_add_53 micro s.vec_add_2 t1822_y t1824_c -> t1818_embedding
node rmsnorm_54 micro s.rmsnorm_2 t1818_embedding model_layers_5_input_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_189 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_5_self_attn_q_proj_weight_q -> t1822_y
node matmul_q4_tiled_fma_190 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_5_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_tiled_fma_191 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_5_self_attn_v_proj_weight_q -> t1826_c
node rope_54 micro s.rope_3 t1822_y position -> t1822_y
node rope_55 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_54 micro s.kv_scatter_2 t1828_c position s520 -> s520
node kv_scatter_55 micro s.kv_scatter_2 t1826_c position s521 -> s521
node flash_prefill_27 micro s.flash_prefill_2 t1822_y s520 s521 position -> t1824_c
node matmul_q4_tiled_fma_192 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_5_self_attn_o_proj_weight_q -> t1822_y
node vec_add_54 micro s.vec_add_2 t1818_embedding t1822_y -> t1824_c
node rmsnorm_55 micro s.rmsnorm_2 t1824_c model_layers_5_post_attention_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_193 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_5_mlp_gate_proj_weight_q -> t1850_c
node matmul_q4_tiled_fma_194 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_5_mlp_up_proj_weight_q -> t1844_c
node silu_mul_10 micro s.silu_mul_2 t1850_c t1844_c -> t1842_c
node matmul_q4_tiled_fma_195 micro s.matmul_q4_tiled_fma_8 t1842_c model_layers_5_mlp_down_proj_weight_q -> t1822_y
node vec_add_55 micro s.vec_add_2 t1824_c t1822_y -> t1818_embedding
node rmsnorm_56 micro s.rmsnorm_2 t1818_embedding model_layers_6_input_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_196 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_6_self_attn_q_proj_weight_q -> t1824_c
node matmul_q4_tiled_fma_197 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_6_self_attn_k_proj_weight_q -> t1826_c
node matmul_q4_tiled_fma_198 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_6_self_attn_v_proj_weight_q -> t1828_c
node rope_56 micro s.rope_3 t1824_c position -> t1824_c
node rope_57 micro s.rope_4 t1826_c position -> t1826_c
node kv_scatter_56 micro s.kv_scatter_2 t1826_c position s522 -> s522
node kv_scatter_57 micro s.kv_scatter_2 t1828_c position s523 -> s523
node flash_prefill_28 micro s.flash_prefill_2 t1824_c s522 s523 position -> t1822_y
node matmul_q4_tiled_fma_199 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_6_self_attn_o_proj_weight_q -> t1824_c
node vec_add_56 micro s.vec_add_2 t1818_embedding t1824_c -> t1822_y
node rmsnorm_57 micro s.rmsnorm_2 t1822_y model_layers_6_post_attention_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_200 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_6_mlp_gate_proj_weight_q -> t1842_c
node matmul_q4_tiled_fma_201 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_6_mlp_up_proj_weight_q -> t1844_c
node silu_mul_11 micro s.silu_mul_2 t1842_c t1844_c -> t1850_c
node matmul_q4_tiled_fma_202 micro s.matmul_q4_tiled_fma_8 t1850_c model_layers_6_mlp_down_proj_weight_q -> t1824_c
node vec_add_57 micro s.vec_add_2 t1822_y t1824_c -> t1818_embedding
node rmsnorm_58 micro s.rmsnorm_2 t1818_embedding model_layers_7_input_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_203 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_7_self_attn_q_proj_weight_q -> t1822_y
node matmul_q4_tiled_fma_204 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_7_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_tiled_fma_205 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_7_self_attn_v_proj_weight_q -> t1826_c
node rope_58 micro s.rope_3 t1822_y position -> t1822_y
node rope_59 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_58 micro s.kv_scatter_2 t1828_c position s524 -> s524
node kv_scatter_59 micro s.kv_scatter_2 t1826_c position s525 -> s525
node flash_prefill_29 micro s.flash_prefill_2 t1822_y s524 s525 position -> t1824_c
node matmul_q4_tiled_fma_206 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_7_self_attn_o_proj_weight_q -> t1822_y
node vec_add_58 micro s.vec_add_2 t1818_embedding t1822_y -> t1824_c
node rmsnorm_59 micro s.rmsnorm_2 t1824_c model_layers_7_post_attention_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_207 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_7_mlp_gate_proj_weight_q -> t1850_c
node matmul_q4_tiled_fma_208 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_7_mlp_up_proj_weight_q -> t1844_c
node silu_mul_12 micro s.silu_mul_2 t1850_c t1844_c -> t1842_c
node matmul_q4_tiled_fma_209 micro s.matmul_q4_tiled_fma_8 t1842_c model_layers_7_mlp_down_proj_weight_q -> t1822_y
node vec_add_59 micro s.vec_add_2 t1824_c t1822_y -> t1818_embedding
node rmsnorm_60 micro s.rmsnorm_2 t1818_embedding model_layers_8_input_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_210 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_8_self_attn_q_proj_weight_q -> t1824_c
node matmul_q4_tiled_fma_211 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_8_self_attn_k_proj_weight_q -> t1826_c
node matmul_q4_tiled_fma_212 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_8_self_attn_v_proj_weight_q -> t1828_c
node rope_60 micro s.rope_3 t1824_c position -> t1824_c
node rope_61 micro s.rope_4 t1826_c position -> t1826_c
node kv_scatter_60 micro s.kv_scatter_2 t1826_c position s526 -> s526
node kv_scatter_61 micro s.kv_scatter_2 t1828_c position s527 -> s527
node flash_prefill_30 micro s.flash_prefill_2 t1824_c s526 s527 position -> t1822_y
node matmul_q4_tiled_fma_213 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_8_self_attn_o_proj_weight_q -> t1824_c
node vec_add_60 micro s.vec_add_2 t1818_embedding t1824_c -> t1822_y
node rmsnorm_61 micro s.rmsnorm_2 t1822_y model_layers_8_post_attention_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_214 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_8_mlp_gate_proj_weight_q -> t1842_c
node matmul_q4_tiled_fma_215 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_8_mlp_up_proj_weight_q -> t1844_c
node act_sigmoid_17 micro s.act_sigmoid_2 t1842_c -> t1850_c
node vec_mul_34 micro s.vec_mul_2 t1842_c t1850_c -> t1899_c
node vec_mul_35 micro s.vec_mul_2 t1899_c t1844_c -> t1850_c
node matmul_q4_tiled_fma_216 micro s.matmul_q4_tiled_fma_8 t1850_c model_layers_8_mlp_down_proj_weight_q -> t1824_c
node vec_add_61 micro s.vec_add_2 t1822_y t1824_c -> t1818_embedding
node rmsnorm_62 micro s.rmsnorm_2 t1818_embedding model_layers_9_input_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_217 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_9_self_attn_q_proj_weight_q -> t1822_y
node matmul_q4_tiled_fma_218 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_9_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_tiled_fma_219 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_9_self_attn_v_proj_weight_q -> t1826_c
node rope_62 micro s.rope_3 t1822_y position -> t1822_y
node rope_63 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_62 micro s.kv_scatter_2 t1828_c position s528 -> s528
node kv_scatter_63 micro s.kv_scatter_2 t1826_c position s529 -> s529
node flash_prefill_31 micro s.flash_prefill_2 t1822_y s528 s529 position -> t1824_c
node matmul_q4_tiled_fma_220 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_9_self_attn_o_proj_weight_q -> t1822_y
node vec_add_62 micro s.vec_add_2 t1818_embedding t1822_y -> t1824_c
node rmsnorm_63 micro s.rmsnorm_2 t1824_c model_layers_9_post_attention_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_221 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_9_mlp_gate_proj_weight_q -> t1850_c
node matmul_q4_tiled_fma_222 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_9_mlp_up_proj_weight_q -> t1899_c
node act_sigmoid_18 micro s.act_sigmoid_2 t1850_c -> t1844_c
node vec_mul_36 micro s.vec_mul_2 t1850_c t1844_c -> t1842_c
node vec_mul_37 micro s.vec_mul_2 t1842_c t1899_c -> t1844_c
node matmul_q4_tiled_fma_223 micro s.matmul_q4_tiled_fma_8 t1844_c model_layers_9_mlp_down_proj_weight_q -> t1822_y
node vec_add_63 micro s.vec_add_2 t1824_c t1822_y -> t1818_embedding
node rmsnorm_64 micro s.rmsnorm_2 t1818_embedding model_layers_10_input_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_224 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_10_self_attn_q_proj_weight_q -> t1824_c
node matmul_q4_tiled_fma_225 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_10_self_attn_k_proj_weight_q -> t1826_c
node matmul_q4_tiled_fma_226 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_10_self_attn_v_proj_weight_q -> t1828_c
node rope_64 micro s.rope_3 t1824_c position -> t1824_c
node rope_65 micro s.rope_4 t1826_c position -> t1826_c
node kv_scatter_64 micro s.kv_scatter_2 t1826_c position s530 -> s530
node kv_scatter_65 micro s.kv_scatter_2 t1828_c position s531 -> s531
node flash_prefill_32 micro s.flash_prefill_2 t1824_c s530 s531 position -> t1822_y
node matmul_q4_tiled_fma_227 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_10_self_attn_o_proj_weight_q -> t1824_c
node vec_add_64 micro s.vec_add_2 t1818_embedding t1824_c -> t1822_y
node rmsnorm_65 micro s.rmsnorm_2 t1822_y model_layers_10_post_attention_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_228 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_10_mlp_gate_proj_weight_q -> t1844_c
node matmul_q4_tiled_fma_229 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_10_mlp_up_proj_weight_q -> t1842_c
node act_sigmoid_19 micro s.act_sigmoid_2 t1844_c -> t1899_c
node vec_mul_38 micro s.vec_mul_2 t1844_c t1899_c -> t1850_c
node vec_mul_39 micro s.vec_mul_2 t1850_c t1842_c -> t1899_c
node matmul_q4_tiled_fma_230 micro s.matmul_q4_tiled_fma_8 t1899_c model_layers_10_mlp_down_proj_weight_q -> t1824_c
node vec_add_65 micro s.vec_add_2 t1822_y t1824_c -> t1818_embedding
node rmsnorm_66 micro s.rmsnorm_2 t1818_embedding model_layers_11_input_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_231 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_11_self_attn_q_proj_weight_q -> t1822_y
node matmul_q4_tiled_fma_232 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_11_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_tiled_fma_233 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_11_self_attn_v_proj_weight_q -> t1826_c
node rope_66 micro s.rope_3 t1822_y position -> t1822_y
node rope_67 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_66 micro s.kv_scatter_2 t1828_c position s532 -> s532
node kv_scatter_67 micro s.kv_scatter_2 t1826_c position s533 -> s533
node flash_prefill_33 micro s.flash_prefill_2 t1822_y s532 s533 position -> t1824_c
node matmul_q4_tiled_fma_234 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_11_self_attn_o_proj_weight_q -> t1822_y
node vec_add_66 micro s.vec_add_2 t1818_embedding t1822_y -> t1824_c
node rmsnorm_67 micro s.rmsnorm_2 t1824_c model_layers_11_post_attention_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_235 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_11_mlp_gate_proj_weight_q -> t1899_c
node matmul_q4_tiled_fma_236 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_11_mlp_up_proj_weight_q -> t1850_c
node act_sigmoid_20 micro s.act_sigmoid_2 t1899_c -> t1842_c
node vec_mul_40 micro s.vec_mul_2 t1899_c t1842_c -> t1844_c
node vec_mul_41 micro s.vec_mul_2 t1844_c t1850_c -> t1842_c
node matmul_q4_tiled_fma_237 micro s.matmul_q4_tiled_fma_8 t1842_c model_layers_11_mlp_down_proj_weight_q -> t1822_y
node vec_add_67 micro s.vec_add_2 t1824_c t1822_y -> t1818_embedding
node rmsnorm_68 micro s.rmsnorm_2 t1818_embedding model_layers_12_input_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_238 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_12_self_attn_q_proj_weight_q -> t1824_c
node matmul_q4_tiled_fma_239 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_12_self_attn_k_proj_weight_q -> t1826_c
node matmul_q4_tiled_fma_240 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_12_self_attn_v_proj_weight_q -> t1828_c
node rope_68 micro s.rope_3 t1824_c position -> t1824_c
node rope_69 micro s.rope_4 t1826_c position -> t1826_c
node kv_scatter_68 micro s.kv_scatter_2 t1826_c position s534 -> s534
node kv_scatter_69 micro s.kv_scatter_2 t1828_c position s535 -> s535
node flash_prefill_34 micro s.flash_prefill_2 t1824_c s534 s535 position -> t1822_y
node matmul_q4_tiled_fma_241 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_12_self_attn_o_proj_weight_q -> t1824_c
node vec_add_68 micro s.vec_add_2 t1818_embedding t1824_c -> t1822_y
node rmsnorm_69 micro s.rmsnorm_2 t1822_y model_layers_12_post_attention_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_242 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_12_mlp_gate_proj_weight_q -> t1842_c
node matmul_q4_tiled_fma_243 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_12_mlp_up_proj_weight_q -> t1844_c
node act_sigmoid_21 micro s.act_sigmoid_2 t1842_c -> t1850_c
node vec_mul_42 micro s.vec_mul_2 t1842_c t1850_c -> t1899_c
node vec_mul_43 micro s.vec_mul_2 t1899_c t1844_c -> t1850_c
node matmul_q4_tiled_fma_244 micro s.matmul_q4_tiled_fma_8 t1850_c model_layers_12_mlp_down_proj_weight_q -> t1824_c
node vec_add_69 micro s.vec_add_2 t1822_y t1824_c -> t1818_embedding
node rmsnorm_70 micro s.rmsnorm_2 t1818_embedding model_layers_13_input_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_245 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_13_self_attn_q_proj_weight_q -> t1822_y
node matmul_q4_tiled_fma_246 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_13_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_tiled_fma_247 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_13_self_attn_v_proj_weight_q -> t1826_c
node rope_70 micro s.rope_3 t1822_y position -> t1822_y
node rope_71 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_70 micro s.kv_scatter_2 t1828_c position s536 -> s536
node kv_scatter_71 micro s.kv_scatter_2 t1826_c position s537 -> s537
node flash_prefill_35 micro s.flash_prefill_2 t1822_y s536 s537 position -> t1824_c
node matmul_q4_tiled_fma_248 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_13_self_attn_o_proj_weight_q -> t1822_y
node vec_add_70 micro s.vec_add_2 t1818_embedding t1822_y -> t1824_c
node rmsnorm_71 micro s.rmsnorm_2 t1824_c model_layers_13_post_attention_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_249 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_13_mlp_gate_proj_weight_q -> t1850_c
node matmul_q4_tiled_fma_250 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_13_mlp_up_proj_weight_q -> t1899_c
node act_sigmoid_22 micro s.act_sigmoid_2 t1850_c -> t1844_c
node vec_mul_44 micro s.vec_mul_2 t1850_c t1844_c -> t1842_c
node vec_mul_45 micro s.vec_mul_2 t1842_c t1899_c -> t1844_c
node matmul_q4_tiled_fma_251 micro s.matmul_q4_tiled_fma_8 t1844_c model_layers_13_mlp_down_proj_weight_q -> t1822_y
node vec_add_71 micro s.vec_add_2 t1824_c t1822_y -> t1818_embedding
node rmsnorm_72 micro s.rmsnorm_2 t1818_embedding model_layers_14_input_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_252 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_14_self_attn_q_proj_weight_q -> t1824_c
node matmul_q4_tiled_fma_253 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_14_self_attn_k_proj_weight_q -> t1826_c
node matmul_q4_tiled_fma_254 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_14_self_attn_v_proj_weight_q -> t1828_c
node rope_72 micro s.rope_3 t1824_c position -> t1824_c
node rope_73 micro s.rope_4 t1826_c position -> t1826_c
node kv_scatter_72 micro s.kv_scatter_2 t1826_c position s538 -> s538
node kv_scatter_73 micro s.kv_scatter_2 t1828_c position s539 -> s539
node flash_prefill_36 micro s.flash_prefill_2 t1824_c s538 s539 position -> t1822_y
node matmul_q4_tiled_fma_255 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_14_self_attn_o_proj_weight_q -> t1824_c
node vec_add_72 micro s.vec_add_2 t1818_embedding t1824_c -> t1822_y
node rmsnorm_73 micro s.rmsnorm_2 t1822_y model_layers_14_post_attention_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_256 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_14_mlp_gate_proj_weight_q -> t1844_c
node matmul_q4_tiled_fma_257 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_14_mlp_up_proj_weight_q -> t1842_c
node act_sigmoid_23 micro s.act_sigmoid_2 t1844_c -> t1899_c
node vec_mul_46 micro s.vec_mul_2 t1844_c t1899_c -> t1850_c
node vec_mul_47 micro s.vec_mul_2 t1850_c t1842_c -> t1899_c
node matmul_q4_tiled_fma_258 micro s.matmul_q4_tiled_fma_8 t1899_c model_layers_14_mlp_down_proj_weight_q -> t1824_c
node vec_add_73 micro s.vec_add_2 t1822_y t1824_c -> t1818_embedding
node rmsnorm_74 micro s.rmsnorm_2 t1818_embedding model_layers_15_input_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_259 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_15_self_attn_q_proj_weight_q -> t1822_y
node matmul_q4_tiled_fma_260 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_15_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_tiled_fma_261 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_15_self_attn_v_proj_weight_q -> t1826_c
node rope_74 micro s.rope_3 t1822_y position -> t1822_y
node rope_75 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_74 micro s.kv_scatter_2 t1828_c position s540 -> s540
node kv_scatter_75 micro s.kv_scatter_2 t1826_c position s541 -> s541
node flash_prefill_37 micro s.flash_prefill_2 t1822_y s540 s541 position -> t1824_c
node matmul_q4_tiled_fma_262 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_15_self_attn_o_proj_weight_q -> t1822_y
node vec_add_74 micro s.vec_add_2 t1818_embedding t1822_y -> t1824_c
node rmsnorm_75 micro s.rmsnorm_2 t1824_c model_layers_15_post_attention_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_263 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_15_mlp_gate_proj_weight_q -> t1899_c
node matmul_q4_tiled_fma_264 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_15_mlp_up_proj_weight_q -> t1850_c
node act_sigmoid_24 micro s.act_sigmoid_2 t1899_c -> t1842_c
node vec_mul_48 micro s.vec_mul_2 t1899_c t1842_c -> t1844_c
node vec_mul_49 micro s.vec_mul_2 t1844_c t1850_c -> t1842_c
node matmul_q4_tiled_fma_265 micro s.matmul_q4_tiled_fma_8 t1842_c model_layers_15_mlp_down_proj_weight_q -> t1822_y
node vec_add_75 micro s.vec_add_2 t1824_c t1822_y -> t1818_embedding
node rmsnorm_76 micro s.rmsnorm_2 t1818_embedding model_layers_16_input_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_266 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_16_self_attn_q_proj_weight_q -> t1824_c
node matmul_q4_tiled_fma_267 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_16_self_attn_k_proj_weight_q -> t1826_c
node matmul_q4_tiled_fma_268 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_16_self_attn_v_proj_weight_q -> t1828_c
node rope_76 micro s.rope_3 t1824_c position -> t1824_c
node rope_77 micro s.rope_4 t1826_c position -> t1826_c
node kv_scatter_76 micro s.kv_scatter_2 t1826_c position s542 -> s542
node kv_scatter_77 micro s.kv_scatter_2 t1828_c position s543 -> s543
node flash_prefill_38 micro s.flash_prefill_2 t1824_c s542 s543 position -> t1822_y
node matmul_q4_tiled_fma_269 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_16_self_attn_o_proj_weight_q -> t1824_c
node vec_add_76 micro s.vec_add_2 t1818_embedding t1824_c -> t1822_y
node rmsnorm_77 micro s.rmsnorm_2 t1822_y model_layers_16_post_attention_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_270 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_16_mlp_gate_proj_weight_q -> t1842_c
node matmul_q4_tiled_fma_271 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_16_mlp_up_proj_weight_q -> t1844_c
node act_sigmoid_25 micro s.act_sigmoid_2 t1842_c -> t1850_c
node vec_mul_50 micro s.vec_mul_2 t1842_c t1850_c -> t1899_c
node vec_mul_51 micro s.vec_mul_2 t1899_c t1844_c -> t1850_c
node matmul_q4_tiled_fma_272 micro s.matmul_q4_tiled_fma_8 t1850_c model_layers_16_mlp_down_proj_weight_q -> t1824_c
node vec_add_77 micro s.vec_add_2 t1822_y t1824_c -> t1818_embedding
node rmsnorm_78 micro s.rmsnorm_2 t1818_embedding model_layers_17_input_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_273 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_17_self_attn_q_proj_weight_q -> t1822_y
node matmul_q4_tiled_fma_274 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_17_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_tiled_fma_275 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_17_self_attn_v_proj_weight_q -> t1826_c
node rope_78 micro s.rope_3 t1822_y position -> t1822_y
node rope_79 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_78 micro s.kv_scatter_2 t1828_c position s544 -> s544
node kv_scatter_79 micro s.kv_scatter_2 t1826_c position s545 -> s545
node flash_prefill_39 micro s.flash_prefill_2 t1822_y s544 s545 position -> t1824_c
node matmul_q4_tiled_fma_276 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_17_self_attn_o_proj_weight_q -> t1822_y
node vec_add_78 micro s.vec_add_2 t1818_embedding t1822_y -> t1824_c
node rmsnorm_79 micro s.rmsnorm_2 t1824_c model_layers_17_post_attention_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_277 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_17_mlp_gate_proj_weight_q -> t1850_c
node matmul_q4_tiled_fma_278 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_17_mlp_up_proj_weight_q -> t1899_c
node act_sigmoid_26 micro s.act_sigmoid_2 t1850_c -> t1844_c
node vec_mul_52 micro s.vec_mul_2 t1850_c t1844_c -> t1842_c
node vec_mul_53 micro s.vec_mul_2 t1842_c t1899_c -> t1844_c
node matmul_q4_tiled_fma_279 micro s.matmul_q4_tiled_fma_8 t1844_c model_layers_17_mlp_down_proj_weight_q -> t1822_y
node vec_add_79 micro s.vec_add_2 t1824_c t1822_y -> t1818_embedding
node rmsnorm_80 micro s.rmsnorm_2 t1818_embedding model_layers_18_input_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_280 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_18_self_attn_q_proj_weight_q -> t1824_c
node matmul_q4_tiled_fma_281 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_18_self_attn_k_proj_weight_q -> t1826_c
node matmul_q4_tiled_fma_282 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_18_self_attn_v_proj_weight_q -> t1828_c
node rope_80 micro s.rope_3 t1824_c position -> t1824_c
node rope_81 micro s.rope_4 t1826_c position -> t1826_c
node kv_scatter_80 micro s.kv_scatter_2 t1826_c position s546 -> s546
node kv_scatter_81 micro s.kv_scatter_2 t1828_c position s547 -> s547
node flash_prefill_40 micro s.flash_prefill_2 t1824_c s546 s547 position -> t1822_y
node matmul_q4_tiled_fma_283 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_18_self_attn_o_proj_weight_q -> t1824_c
node vec_add_80 micro s.vec_add_2 t1818_embedding t1824_c -> t1822_y
node rmsnorm_81 micro s.rmsnorm_2 t1822_y model_layers_18_post_attention_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_284 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_18_mlp_gate_proj_weight_q -> t1844_c
node matmul_q4_tiled_fma_285 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_18_mlp_up_proj_weight_q -> t1842_c
node act_sigmoid_27 micro s.act_sigmoid_2 t1844_c -> t1899_c
node vec_mul_54 micro s.vec_mul_2 t1844_c t1899_c -> t1850_c
node vec_mul_55 micro s.vec_mul_2 t1850_c t1842_c -> t1899_c
node matmul_q4_tiled_fma_286 micro s.matmul_q4_tiled_fma_8 t1899_c model_layers_18_mlp_down_proj_weight_q -> t1824_c
node vec_add_81 micro s.vec_add_2 t1822_y t1824_c -> t1818_embedding
node rmsnorm_82 micro s.rmsnorm_2 t1818_embedding model_layers_19_input_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_287 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_19_self_attn_q_proj_weight_q -> t1822_y
node matmul_q4_tiled_fma_288 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_19_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_tiled_fma_289 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_19_self_attn_v_proj_weight_q -> t1826_c
node rope_82 micro s.rope_3 t1822_y position -> t1822_y
node rope_83 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_82 micro s.kv_scatter_2 t1828_c position s548 -> s548
node kv_scatter_83 micro s.kv_scatter_2 t1826_c position s549 -> s549
node flash_prefill_41 micro s.flash_prefill_2 t1822_y s548 s549 position -> t1824_c
node matmul_q4_tiled_fma_290 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_19_self_attn_o_proj_weight_q -> t1822_y
node vec_add_82 micro s.vec_add_2 t1818_embedding t1822_y -> t1824_c
node rmsnorm_83 micro s.rmsnorm_2 t1824_c model_layers_19_post_attention_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_291 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_19_mlp_gate_proj_weight_q -> t1899_c
node matmul_q4_tiled_fma_292 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_19_mlp_up_proj_weight_q -> t1850_c
node act_sigmoid_28 micro s.act_sigmoid_2 t1899_c -> t1842_c
node vec_mul_56 micro s.vec_mul_2 t1899_c t1842_c -> t1844_c
node vec_mul_57 micro s.vec_mul_2 t1844_c t1850_c -> t1842_c
node matmul_q4_tiled_fma_293 micro s.matmul_q4_tiled_fma_8 t1842_c model_layers_19_mlp_down_proj_weight_q -> t1822_y
node vec_add_83 micro s.vec_add_2 t1824_c t1822_y -> t1818_embedding
node rmsnorm_84 micro s.rmsnorm_2 t1818_embedding model_layers_20_input_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_294 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_20_self_attn_q_proj_weight_q -> t1824_c
node matmul_q4_tiled_fma_295 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_20_self_attn_k_proj_weight_q -> t1826_c
node matmul_q4_tiled_fma_296 micro s.matmul_q4_tiled_fma_6 t1822_y model_layers_20_self_attn_v_proj_weight_q -> t1828_c
node rope_84 micro s.rope_3 t1824_c position -> t1824_c
node rope_85 micro s.rope_4 t1826_c position -> t1826_c
node kv_scatter_84 micro s.kv_scatter_2 t1826_c position s550 -> s550
node kv_scatter_85 micro s.kv_scatter_2 t1828_c position s551 -> s551
node flash_prefill_42 micro s.flash_prefill_2 t1824_c s550 s551 position -> t1822_y
node matmul_q4_tiled_fma_297 micro s.matmul_q4_tiled_fma_5 t1822_y model_layers_20_self_attn_o_proj_weight_q -> t1824_c
node vec_add_84 micro s.vec_add_2 t1818_embedding t1824_c -> t1822_y
node rmsnorm_85 micro s.rmsnorm_2 t1822_y model_layers_20_post_attention_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_298 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_20_mlp_gate_proj_weight_q -> t1842_c
node matmul_q4_tiled_fma_299 micro s.matmul_q4_tiled_fma_7 t1824_c model_layers_20_mlp_up_proj_weight_q -> t1844_c
node act_sigmoid_29 micro s.act_sigmoid_2 t1842_c -> t1850_c
node vec_mul_58 micro s.vec_mul_2 t1842_c t1850_c -> t1899_c
node vec_mul_59 micro s.vec_mul_2 t1899_c t1844_c -> t1850_c
node matmul_q4_tiled_fma_300 micro s.matmul_q4_tiled_fma_8 t1850_c model_layers_20_mlp_down_proj_weight_q -> t1824_c
node vec_add_85 micro s.vec_add_2 t1822_y t1824_c -> t1818_embedding
node rmsnorm_86 micro s.rmsnorm_2 t1818_embedding model_layers_21_input_layernorm_weight -> t1824_c
node matmul_q4_tiled_fma_301 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_21_self_attn_q_proj_weight_q -> t1822_y
node matmul_q4_tiled_fma_302 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_21_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_tiled_fma_303 micro s.matmul_q4_tiled_fma_6 t1824_c model_layers_21_self_attn_v_proj_weight_q -> t1826_c
node rope_86 micro s.rope_3 t1822_y position -> t1822_y
node rope_87 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_86 micro s.kv_scatter_2 t1828_c position s552 -> s552
node kv_scatter_87 micro s.kv_scatter_2 t1826_c position s553 -> s553
node flash_prefill_43 micro s.flash_prefill_2 t1822_y s552 s553 position -> t1824_c
node matmul_q4_tiled_fma_304 micro s.matmul_q4_tiled_fma_5 t1824_c model_layers_21_self_attn_o_proj_weight_q -> t1822_y
node vec_add_86 micro s.vec_add_2 t1818_embedding t1822_y -> t1824_c
node rmsnorm_87 micro s.rmsnorm_2 t1824_c model_layers_21_post_attention_layernorm_weight -> t1822_y
node matmul_q4_tiled_fma_305 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_21_mlp_gate_proj_weight_q -> t1850_c
node matmul_q4_tiled_fma_306 micro s.matmul_q4_tiled_fma_7 t1822_y model_layers_21_mlp_up_proj_weight_q -> t1899_c
node act_sigmoid_30 micro s.act_sigmoid_2 t1850_c -> t1844_c
node vec_mul_60 micro s.vec_mul_2 t1850_c t1844_c -> t1842_c
node vec_mul_61 micro s.vec_mul_2 t1842_c t1899_c -> t1844_c
node matmul_q4_tiled_fma_307 micro s.matmul_q4_tiled_fma_8 t1844_c model_layers_21_mlp_down_proj_weight_q -> t1822_y
node vec_add_87 micro s.vec_add_2 t1824_c t1822_y -> t1818_embedding
node last_row_1 session sessionCopyInto:srcOff=0,dstOff=0,n=2048,lastRowOf=448 t1818_embedding -> s694
node rmsnorm_88 micro s.rmsnorm_3 s694 model_norm_weight -> t2930_y
node matmul_q4_simd_0 micro s.matmul_q4_simd t2930_y lm_head_weight_q -> t2931_c
phase decode
node embedding ext fir.embedding token model_embed_tokens_weight_emb -> t2957_embedding
node rmsnorm_89 micro s.rmsnorm_3 t2957_embedding model_layers_0_input_layernorm_weight -> t2962_y
node matmul_q4_simd_1_side micro s.side t2962_y model_layers_0_self_attn_q_proj_weight_q t2962_y model_layers_0_self_attn_k_proj_weight_q t2962_y model_layers_0_self_attn_v_proj_weight_q -> t2963_c t2964_c t2965_c
node rope_88_side micro s.side_2 t2963_c position t2964_c position -> t2963_c t2964_c
node kv_scatter_88_side micro s.side_3 t2964_c position s510 t2965_c position s511 -> s510 s511
node attention_0 micro s.attention t2963_c s510 s511 position -> t2962_y
node matmul_q4_simd_4 micro s.matmul_q4_simd_2 t2962_y model_layers_0_self_attn_o_proj_weight_q -> t2963_c
node add_rmsnorm_0 micro s.add_rmsnorm t2957_embedding t2963_c model_layers_0_post_attention_layernorm_weight -> t2962_y t2978_y
node matmul_q4_simd_5_side micro s.side_4 t2978_y model_layers_0_mlp_gate_proj_weight_q t2978_y model_layers_0_mlp_up_proj_weight_q -> t2979_c t2980_c
node silu_mul_13 micro s.silu_mul_3 t2979_c t2980_c -> t2986_c
node matmul_q4_simd_7 micro s.matmul_q4_simd_5 t2986_c model_layers_0_mlp_down_proj_weight_q -> t2978_y
node add_rmsnorm_1 micro s.add_rmsnorm t2962_y t2978_y model_layers_1_input_layernorm_weight -> t2963_c t2957_embedding
node matmul_q4_simd_8_side micro s.side t2957_embedding model_layers_1_self_attn_q_proj_weight_q t2957_embedding model_layers_1_self_attn_k_proj_weight_q t2957_embedding model_layers_1_self_attn_v_proj_weight_q -> t2978_y t2965_c t2964_c
node rope_90_side micro s.side_2 t2978_y position t2965_c position -> t2978_y t2965_c
node kv_scatter_90_side micro s.side_3 t2965_c position s512 t2964_c position s513 -> s512 s513
node attention_1 micro s.attention t2978_y s512 s513 position -> t2957_embedding
node matmul_q4_simd_11 micro s.matmul_q4_simd_2 t2957_embedding model_layers_1_self_attn_o_proj_weight_q -> t2978_y
node add_rmsnorm_2 micro s.add_rmsnorm t2963_c t2978_y model_layers_1_post_attention_layernorm_weight -> t2957_embedding t2962_y
node matmul_q4_simd_12_side micro s.side_4 t2962_y model_layers_1_mlp_gate_proj_weight_q t2962_y model_layers_1_mlp_up_proj_weight_q -> t2986_c t2980_c
node silu_mul_14 micro s.silu_mul_3 t2986_c t2980_c -> t2979_c
node matmul_q4_simd_14 micro s.matmul_q4_simd_5 t2979_c model_layers_1_mlp_down_proj_weight_q -> t2962_y
node add_rmsnorm_3 micro s.add_rmsnorm t2957_embedding t2962_y model_layers_2_input_layernorm_weight -> t2978_y t2963_c
node matmul_q4_simd_15_side micro s.side t2963_c model_layers_2_self_attn_q_proj_weight_q t2963_c model_layers_2_self_attn_k_proj_weight_q t2963_c model_layers_2_self_attn_v_proj_weight_q -> t2962_y t2964_c t2965_c
node rope_92_side micro s.side_2 t2962_y position t2964_c position -> t2962_y t2964_c
node kv_scatter_92_side micro s.side_3 t2964_c position s514 t2965_c position s515 -> s514 s515
node attention_2 micro s.attention t2962_y s514 s515 position -> t2963_c
node matmul_q4_simd_18 micro s.matmul_q4_simd_2 t2963_c model_layers_2_self_attn_o_proj_weight_q -> t2962_y
node add_rmsnorm_4 micro s.add_rmsnorm t2978_y t2962_y model_layers_2_post_attention_layernorm_weight -> t2963_c t2957_embedding
node matmul_q4_simd_19_side micro s.side_4 t2957_embedding model_layers_2_mlp_gate_proj_weight_q t2957_embedding model_layers_2_mlp_up_proj_weight_q -> t2979_c t2980_c
node silu_mul_15 micro s.silu_mul_3 t2979_c t2980_c -> t2986_c
node matmul_q4_simd_21 micro s.matmul_q4_simd_5 t2986_c model_layers_2_mlp_down_proj_weight_q -> t2957_embedding
node add_rmsnorm_5 micro s.add_rmsnorm t2963_c t2957_embedding model_layers_3_input_layernorm_weight -> t2962_y t2978_y
node matmul_q4_simd_22_side micro s.side t2978_y model_layers_3_self_attn_q_proj_weight_q t2978_y model_layers_3_self_attn_k_proj_weight_q t2978_y model_layers_3_self_attn_v_proj_weight_q -> t2957_embedding t2965_c t2964_c
node rope_94_side micro s.side_2 t2957_embedding position t2965_c position -> t2957_embedding t2965_c
node kv_scatter_94_side micro s.side_3 t2965_c position s516 t2964_c position s517 -> s516 s517
node attention_3 micro s.attention t2957_embedding s516 s517 position -> t2978_y
node matmul_q4_simd_25 micro s.matmul_q4_simd_2 t2978_y model_layers_3_self_attn_o_proj_weight_q -> t2957_embedding
node add_rmsnorm_6 micro s.add_rmsnorm t2962_y t2957_embedding model_layers_3_post_attention_layernorm_weight -> t2978_y t2963_c
node matmul_q4_simd_26_side micro s.side_4 t2963_c model_layers_3_mlp_gate_proj_weight_q t2963_c model_layers_3_mlp_up_proj_weight_q -> t2986_c t2980_c
node silu_mul_16 micro s.silu_mul_3 t2986_c t2980_c -> t2979_c
node matmul_q4_simd_28 micro s.matmul_q4_simd_5 t2979_c model_layers_3_mlp_down_proj_weight_q -> t2963_c
node add_rmsnorm_7 micro s.add_rmsnorm t2978_y t2963_c model_layers_4_input_layernorm_weight -> t2957_embedding t2962_y
node matmul_q4_simd_29_side micro s.side t2962_y model_layers_4_self_attn_q_proj_weight_q t2962_y model_layers_4_self_attn_k_proj_weight_q t2962_y model_layers_4_self_attn_v_proj_weight_q -> t2963_c t2964_c t2965_c
node rope_96_side micro s.side_2 t2963_c position t2964_c position -> t2963_c t2964_c
node kv_scatter_96_side micro s.side_3 t2964_c position s518 t2965_c position s519 -> s518 s519
node attention_4 micro s.attention t2963_c s518 s519 position -> t2962_y
node matmul_q4_simd_32 micro s.matmul_q4_simd_2 t2962_y model_layers_4_self_attn_o_proj_weight_q -> t2963_c
node add_rmsnorm_8 micro s.add_rmsnorm t2957_embedding t2963_c model_layers_4_post_attention_layernorm_weight -> t2962_y t2978_y
node matmul_q4_simd_33_side micro s.side_4 t2978_y model_layers_4_mlp_gate_proj_weight_q t2978_y model_layers_4_mlp_up_proj_weight_q -> t2979_c t2980_c
node silu_mul_17 micro s.silu_mul_3 t2979_c t2980_c -> t2986_c
node matmul_q4_simd_35 micro s.matmul_q4_simd_5 t2986_c model_layers_4_mlp_down_proj_weight_q -> t2978_y
node add_rmsnorm_9 micro s.add_rmsnorm t2962_y t2978_y model_layers_5_input_layernorm_weight -> t2963_c t2957_embedding
node matmul_q4_simd_36_side micro s.side t2957_embedding model_layers_5_self_attn_q_proj_weight_q t2957_embedding model_layers_5_self_attn_k_proj_weight_q t2957_embedding model_layers_5_self_attn_v_proj_weight_q -> t2978_y t2965_c t2964_c
node rope_98_side micro s.side_2 t2978_y position t2965_c position -> t2978_y t2965_c
node kv_scatter_98_side micro s.side_3 t2965_c position s520 t2964_c position s521 -> s520 s521
node attention_5 micro s.attention t2978_y s520 s521 position -> t2957_embedding
node matmul_q4_simd_39 micro s.matmul_q4_simd_2 t2957_embedding model_layers_5_self_attn_o_proj_weight_q -> t2978_y
node add_rmsnorm_10 micro s.add_rmsnorm t2963_c t2978_y model_layers_5_post_attention_layernorm_weight -> t2957_embedding t2962_y
node matmul_q4_simd_40_side micro s.side_4 t2962_y model_layers_5_mlp_gate_proj_weight_q t2962_y model_layers_5_mlp_up_proj_weight_q -> t2986_c t2980_c
node silu_mul_18 micro s.silu_mul_3 t2986_c t2980_c -> t2979_c
node matmul_q4_simd_42 micro s.matmul_q4_simd_5 t2979_c model_layers_5_mlp_down_proj_weight_q -> t2962_y
node add_rmsnorm_11 micro s.add_rmsnorm t2957_embedding t2962_y model_layers_6_input_layernorm_weight -> t2978_y t2963_c
node matmul_q4_simd_43_side micro s.side t2963_c model_layers_6_self_attn_q_proj_weight_q t2963_c model_layers_6_self_attn_k_proj_weight_q t2963_c model_layers_6_self_attn_v_proj_weight_q -> t2962_y t2964_c t2965_c
node rope_100_side micro s.side_2 t2962_y position t2964_c position -> t2962_y t2964_c
node kv_scatter_100_side micro s.side_3 t2964_c position s522 t2965_c position s523 -> s522 s523
node attention_6 micro s.attention t2962_y s522 s523 position -> t2963_c
node matmul_q4_simd_46 micro s.matmul_q4_simd_2 t2963_c model_layers_6_self_attn_o_proj_weight_q -> t2962_y
node add_rmsnorm_12 micro s.add_rmsnorm t2978_y t2962_y model_layers_6_post_attention_layernorm_weight -> t2963_c t2957_embedding
node matmul_q4_simd_47_side micro s.side_4 t2957_embedding model_layers_6_mlp_gate_proj_weight_q t2957_embedding model_layers_6_mlp_up_proj_weight_q -> t2979_c t2980_c
node silu_mul_19 micro s.silu_mul_3 t2979_c t2980_c -> t2986_c
node matmul_q4_simd_49 micro s.matmul_q4_simd_5 t2986_c model_layers_6_mlp_down_proj_weight_q -> t2957_embedding
node add_rmsnorm_13 micro s.add_rmsnorm t2963_c t2957_embedding model_layers_7_input_layernorm_weight -> t2962_y t2978_y
node matmul_q4_simd_50_side micro s.side t2978_y model_layers_7_self_attn_q_proj_weight_q t2978_y model_layers_7_self_attn_k_proj_weight_q t2978_y model_layers_7_self_attn_v_proj_weight_q -> t2957_embedding t2965_c t2964_c
node rope_102_side micro s.side_2 t2957_embedding position t2965_c position -> t2957_embedding t2965_c
node kv_scatter_102_side micro s.side_3 t2965_c position s524 t2964_c position s525 -> s524 s525
node attention_7 micro s.attention t2957_embedding s524 s525 position -> t2978_y
node matmul_q4_simd_53 micro s.matmul_q4_simd_2 t2978_y model_layers_7_self_attn_o_proj_weight_q -> t2957_embedding
node add_rmsnorm_14 micro s.add_rmsnorm t2962_y t2957_embedding model_layers_7_post_attention_layernorm_weight -> t2978_y t2963_c
node matmul_q4_simd_54_side micro s.side_4 t2963_c model_layers_7_mlp_gate_proj_weight_q t2963_c model_layers_7_mlp_up_proj_weight_q -> t2986_c t2980_c
node silu_mul_20 micro s.silu_mul_3 t2986_c t2980_c -> t2979_c
node matmul_q4_simd_56 micro s.matmul_q4_simd_5 t2979_c model_layers_7_mlp_down_proj_weight_q -> t2963_c
node add_rmsnorm_15 micro s.add_rmsnorm t2978_y t2963_c model_layers_8_input_layernorm_weight -> t2957_embedding t2962_y
node matmul_q4_simd_57_side micro s.side t2962_y model_layers_8_self_attn_q_proj_weight_q t2962_y model_layers_8_self_attn_k_proj_weight_q t2962_y model_layers_8_self_attn_v_proj_weight_q -> t2963_c t2964_c t2965_c
node rope_104_side micro s.side_2 t2963_c position t2964_c position -> t2963_c t2964_c
node kv_scatter_104_side micro s.side_3 t2964_c position s526 t2965_c position s527 -> s526 s527
node attention_8 micro s.attention t2963_c s526 s527 position -> t2962_y
node matmul_q4_simd_60 micro s.matmul_q4_simd_2 t2962_y model_layers_8_self_attn_o_proj_weight_q -> t2963_c
node add_rmsnorm_16 micro s.add_rmsnorm t2957_embedding t2963_c model_layers_8_post_attention_layernorm_weight -> t2962_y t2978_y
node matmul_q4_simd_61_side micro s.side_4 t2978_y model_layers_8_mlp_gate_proj_weight_q t2978_y model_layers_8_mlp_up_proj_weight_q -> t2979_c t2980_c
node silu_mul_21 micro s.silu_mul_3 t2979_c t2980_c -> t2986_c
node matmul_q4_simd_63 micro s.matmul_q4_simd_5 t2986_c model_layers_8_mlp_down_proj_weight_q -> t2978_y
node add_rmsnorm_17 micro s.add_rmsnorm t2962_y t2978_y model_layers_9_input_layernorm_weight -> t2963_c t2957_embedding
node matmul_q4_simd_64_side micro s.side t2957_embedding model_layers_9_self_attn_q_proj_weight_q t2957_embedding model_layers_9_self_attn_k_proj_weight_q t2957_embedding model_layers_9_self_attn_v_proj_weight_q -> t2978_y t2965_c t2964_c
node rope_106_side micro s.side_2 t2978_y position t2965_c position -> t2978_y t2965_c
node kv_scatter_106_side micro s.side_3 t2965_c position s528 t2964_c position s529 -> s528 s529
node attention_9 micro s.attention t2978_y s528 s529 position -> t2957_embedding
node matmul_q4_simd_67 micro s.matmul_q4_simd_2 t2957_embedding model_layers_9_self_attn_o_proj_weight_q -> t2978_y
node add_rmsnorm_18 micro s.add_rmsnorm t2963_c t2978_y model_layers_9_post_attention_layernorm_weight -> t2957_embedding t2962_y
node matmul_q4_simd_68_side micro s.side_4 t2962_y model_layers_9_mlp_gate_proj_weight_q t2962_y model_layers_9_mlp_up_proj_weight_q -> t2986_c t2980_c
node silu_mul_22 micro s.silu_mul_3 t2986_c t2980_c -> t2979_c
node matmul_q4_simd_70 micro s.matmul_q4_simd_5 t2979_c model_layers_9_mlp_down_proj_weight_q -> t2962_y
node add_rmsnorm_19 micro s.add_rmsnorm t2957_embedding t2962_y model_layers_10_input_layernorm_weight -> t2978_y t2963_c
node matmul_q4_simd_71_side micro s.side t2963_c model_layers_10_self_attn_q_proj_weight_q t2963_c model_layers_10_self_attn_k_proj_weight_q t2963_c model_layers_10_self_attn_v_proj_weight_q -> t2962_y t2964_c t2965_c
node rope_108_side micro s.side_2 t2962_y position t2964_c position -> t2962_y t2964_c
node kv_scatter_108_side micro s.side_3 t2964_c position s530 t2965_c position s531 -> s530 s531
node attention_10 micro s.attention t2962_y s530 s531 position -> t2963_c
node matmul_q4_simd_74 micro s.matmul_q4_simd_2 t2963_c model_layers_10_self_attn_o_proj_weight_q -> t2962_y
node add_rmsnorm_20 micro s.add_rmsnorm t2978_y t2962_y model_layers_10_post_attention_layernorm_weight -> t2963_c t2957_embedding
node matmul_q4_simd_75_side micro s.side_4 t2957_embedding model_layers_10_mlp_gate_proj_weight_q t2957_embedding model_layers_10_mlp_up_proj_weight_q -> t2979_c t2980_c
node silu_mul_23 micro s.silu_mul_3 t2979_c t2980_c -> t2986_c
node matmul_q4_simd_77 micro s.matmul_q4_simd_5 t2986_c model_layers_10_mlp_down_proj_weight_q -> t2957_embedding
node add_rmsnorm_21 micro s.add_rmsnorm t2963_c t2957_embedding model_layers_11_input_layernorm_weight -> t2962_y t2978_y
node matmul_q4_simd_78_side micro s.side t2978_y model_layers_11_self_attn_q_proj_weight_q t2978_y model_layers_11_self_attn_k_proj_weight_q t2978_y model_layers_11_self_attn_v_proj_weight_q -> t2957_embedding t2965_c t2964_c
node rope_110_side micro s.side_2 t2957_embedding position t2965_c position -> t2957_embedding t2965_c
node kv_scatter_110_side micro s.side_3 t2965_c position s532 t2964_c position s533 -> s532 s533
node attention_11 micro s.attention t2957_embedding s532 s533 position -> t2978_y
node matmul_q4_simd_81 micro s.matmul_q4_simd_2 t2978_y model_layers_11_self_attn_o_proj_weight_q -> t2957_embedding
node add_rmsnorm_22 micro s.add_rmsnorm t2962_y t2957_embedding model_layers_11_post_attention_layernorm_weight -> t2978_y t2963_c
node matmul_q4_simd_82_side micro s.side_4 t2963_c model_layers_11_mlp_gate_proj_weight_q t2963_c model_layers_11_mlp_up_proj_weight_q -> t2986_c t2980_c
node silu_mul_24 micro s.silu_mul_3 t2986_c t2980_c -> t2979_c
node matmul_q4_simd_84 micro s.matmul_q4_simd_5 t2979_c model_layers_11_mlp_down_proj_weight_q -> t2963_c
node add_rmsnorm_23 micro s.add_rmsnorm t2978_y t2963_c model_layers_12_input_layernorm_weight -> t2957_embedding t2962_y
node matmul_q4_simd_85_side micro s.side t2962_y model_layers_12_self_attn_q_proj_weight_q t2962_y model_layers_12_self_attn_k_proj_weight_q t2962_y model_layers_12_self_attn_v_proj_weight_q -> t2963_c t2964_c t2965_c
node rope_112_side micro s.side_2 t2963_c position t2964_c position -> t2963_c t2964_c
node kv_scatter_112_side micro s.side_3 t2964_c position s534 t2965_c position s535 -> s534 s535
node attention_12 micro s.attention t2963_c s534 s535 position -> t2962_y
node matmul_q4_simd_88 micro s.matmul_q4_simd_2 t2962_y model_layers_12_self_attn_o_proj_weight_q -> t2963_c
node add_rmsnorm_24 micro s.add_rmsnorm t2957_embedding t2963_c model_layers_12_post_attention_layernorm_weight -> t2962_y t2978_y
node matmul_q4_simd_89_side micro s.side_4 t2978_y model_layers_12_mlp_gate_proj_weight_q t2978_y model_layers_12_mlp_up_proj_weight_q -> t2979_c t2980_c
node silu_mul_25 micro s.silu_mul_3 t2979_c t2980_c -> t2986_c
node matmul_q4_simd_91 micro s.matmul_q4_simd_5 t2986_c model_layers_12_mlp_down_proj_weight_q -> t2978_y
node add_rmsnorm_25 micro s.add_rmsnorm t2962_y t2978_y model_layers_13_input_layernorm_weight -> t2963_c t2957_embedding
node matmul_q4_simd_92_side micro s.side t2957_embedding model_layers_13_self_attn_q_proj_weight_q t2957_embedding model_layers_13_self_attn_k_proj_weight_q t2957_embedding model_layers_13_self_attn_v_proj_weight_q -> t2978_y t2965_c t2964_c
node rope_114_side micro s.side_2 t2978_y position t2965_c position -> t2978_y t2965_c
node kv_scatter_114_side micro s.side_3 t2965_c position s536 t2964_c position s537 -> s536 s537
node attention_13 micro s.attention t2978_y s536 s537 position -> t2957_embedding
node matmul_q4_simd_95 micro s.matmul_q4_simd_2 t2957_embedding model_layers_13_self_attn_o_proj_weight_q -> t2978_y
node add_rmsnorm_26 micro s.add_rmsnorm t2963_c t2978_y model_layers_13_post_attention_layernorm_weight -> t2957_embedding t2962_y
node matmul_q4_simd_96_side micro s.side_4 t2962_y model_layers_13_mlp_gate_proj_weight_q t2962_y model_layers_13_mlp_up_proj_weight_q -> t2986_c t2980_c
node silu_mul_26 micro s.silu_mul_3 t2986_c t2980_c -> t2979_c
node matmul_q4_simd_98 micro s.matmul_q4_simd_5 t2979_c model_layers_13_mlp_down_proj_weight_q -> t2962_y
node add_rmsnorm_27 micro s.add_rmsnorm t2957_embedding t2962_y model_layers_14_input_layernorm_weight -> t2978_y t2963_c
node matmul_q4_simd_99_side micro s.side t2963_c model_layers_14_self_attn_q_proj_weight_q t2963_c model_layers_14_self_attn_k_proj_weight_q t2963_c model_layers_14_self_attn_v_proj_weight_q -> t2962_y t2964_c t2965_c
node rope_116_side micro s.side_2 t2962_y position t2964_c position -> t2962_y t2964_c
node kv_scatter_116_side micro s.side_3 t2964_c position s538 t2965_c position s539 -> s538 s539
node attention_14 micro s.attention t2962_y s538 s539 position -> t2963_c
node matmul_q4_simd_102 micro s.matmul_q4_simd_2 t2963_c model_layers_14_self_attn_o_proj_weight_q -> t2962_y
node add_rmsnorm_28 micro s.add_rmsnorm t2978_y t2962_y model_layers_14_post_attention_layernorm_weight -> t2963_c t2957_embedding
node matmul_q4_simd_103_side micro s.side_4 t2957_embedding model_layers_14_mlp_gate_proj_weight_q t2957_embedding model_layers_14_mlp_up_proj_weight_q -> t2979_c t2980_c
node silu_mul_27 micro s.silu_mul_3 t2979_c t2980_c -> t2986_c
node matmul_q4_simd_105 micro s.matmul_q4_simd_5 t2986_c model_layers_14_mlp_down_proj_weight_q -> t2957_embedding
node add_rmsnorm_29 micro s.add_rmsnorm t2963_c t2957_embedding model_layers_15_input_layernorm_weight -> t2962_y t2978_y
node matmul_q4_simd_106_side micro s.side t2978_y model_layers_15_self_attn_q_proj_weight_q t2978_y model_layers_15_self_attn_k_proj_weight_q t2978_y model_layers_15_self_attn_v_proj_weight_q -> t2957_embedding t2965_c t2964_c
node rope_118_side micro s.side_2 t2957_embedding position t2965_c position -> t2957_embedding t2965_c
node kv_scatter_118_side micro s.side_3 t2965_c position s540 t2964_c position s541 -> s540 s541
node attention_15 micro s.attention t2957_embedding s540 s541 position -> t2978_y
node matmul_q4_simd_109 micro s.matmul_q4_simd_2 t2978_y model_layers_15_self_attn_o_proj_weight_q -> t2957_embedding
node add_rmsnorm_30 micro s.add_rmsnorm t2962_y t2957_embedding model_layers_15_post_attention_layernorm_weight -> t2978_y t2963_c
node matmul_q4_simd_110_side micro s.side_4 t2963_c model_layers_15_mlp_gate_proj_weight_q t2963_c model_layers_15_mlp_up_proj_weight_q -> t2986_c t2980_c
node silu_mul_28 micro s.silu_mul_3 t2986_c t2980_c -> t2979_c
node matmul_q4_simd_112 micro s.matmul_q4_simd_5 t2979_c model_layers_15_mlp_down_proj_weight_q -> t2963_c
node add_rmsnorm_31 micro s.add_rmsnorm t2978_y t2963_c model_layers_16_input_layernorm_weight -> t2957_embedding t2962_y
node matmul_q4_simd_113_side micro s.side t2962_y model_layers_16_self_attn_q_proj_weight_q t2962_y model_layers_16_self_attn_k_proj_weight_q t2962_y model_layers_16_self_attn_v_proj_weight_q -> t2963_c t2964_c t2965_c
node rope_120_side micro s.side_2 t2963_c position t2964_c position -> t2963_c t2964_c
node kv_scatter_120_side micro s.side_3 t2964_c position s542 t2965_c position s543 -> s542 s543
node attention_16 micro s.attention t2963_c s542 s543 position -> t2962_y
node matmul_q4_simd_116 micro s.matmul_q4_simd_2 t2962_y model_layers_16_self_attn_o_proj_weight_q -> t2963_c
node add_rmsnorm_32 micro s.add_rmsnorm t2957_embedding t2963_c model_layers_16_post_attention_layernorm_weight -> t2962_y t2978_y
node matmul_q4_simd_117_side micro s.side_4 t2978_y model_layers_16_mlp_gate_proj_weight_q t2978_y model_layers_16_mlp_up_proj_weight_q -> t2979_c t2980_c
node silu_mul_29 micro s.silu_mul_3 t2979_c t2980_c -> t2986_c
node matmul_q4_simd_119 micro s.matmul_q4_simd_5 t2986_c model_layers_16_mlp_down_proj_weight_q -> t2978_y
node add_rmsnorm_33 micro s.add_rmsnorm t2962_y t2978_y model_layers_17_input_layernorm_weight -> t2963_c t2957_embedding
node matmul_q4_simd_120_side micro s.side t2957_embedding model_layers_17_self_attn_q_proj_weight_q t2957_embedding model_layers_17_self_attn_k_proj_weight_q t2957_embedding model_layers_17_self_attn_v_proj_weight_q -> t2978_y t2965_c t2964_c
node rope_122_side micro s.side_2 t2978_y position t2965_c position -> t2978_y t2965_c
node kv_scatter_122_side micro s.side_3 t2965_c position s544 t2964_c position s545 -> s544 s545
node attention_17 micro s.attention t2978_y s544 s545 position -> t2957_embedding
node matmul_q4_simd_123 micro s.matmul_q4_simd_2 t2957_embedding model_layers_17_self_attn_o_proj_weight_q -> t2978_y
node add_rmsnorm_34 micro s.add_rmsnorm t2963_c t2978_y model_layers_17_post_attention_layernorm_weight -> t2957_embedding t2962_y
node matmul_q4_simd_124_side micro s.side_4 t2962_y model_layers_17_mlp_gate_proj_weight_q t2962_y model_layers_17_mlp_up_proj_weight_q -> t2986_c t2980_c
node silu_mul_30 micro s.silu_mul_3 t2986_c t2980_c -> t2979_c
node matmul_q4_simd_126 micro s.matmul_q4_simd_5 t2979_c model_layers_17_mlp_down_proj_weight_q -> t2962_y
node add_rmsnorm_35 micro s.add_rmsnorm t2957_embedding t2962_y model_layers_18_input_layernorm_weight -> t2978_y t2963_c
node matmul_q4_simd_127_side micro s.side t2963_c model_layers_18_self_attn_q_proj_weight_q t2963_c model_layers_18_self_attn_k_proj_weight_q t2963_c model_layers_18_self_attn_v_proj_weight_q -> t2962_y t2964_c t2965_c
node rope_124_side micro s.side_2 t2962_y position t2964_c position -> t2962_y t2964_c
node kv_scatter_124_side micro s.side_3 t2964_c position s546 t2965_c position s547 -> s546 s547
node attention_18 micro s.attention t2962_y s546 s547 position -> t2963_c
node matmul_q4_simd_130 micro s.matmul_q4_simd_2 t2963_c model_layers_18_self_attn_o_proj_weight_q -> t2962_y
node add_rmsnorm_36 micro s.add_rmsnorm t2978_y t2962_y model_layers_18_post_attention_layernorm_weight -> t2963_c t2957_embedding
node matmul_q4_simd_131_side micro s.side_4 t2957_embedding model_layers_18_mlp_gate_proj_weight_q t2957_embedding model_layers_18_mlp_up_proj_weight_q -> t2979_c t2980_c
node silu_mul_31 micro s.silu_mul_3 t2979_c t2980_c -> t2986_c
node matmul_q4_simd_133 micro s.matmul_q4_simd_5 t2986_c model_layers_18_mlp_down_proj_weight_q -> t2957_embedding
node add_rmsnorm_37 micro s.add_rmsnorm t2963_c t2957_embedding model_layers_19_input_layernorm_weight -> t2962_y t2978_y
node matmul_q4_simd_134_side micro s.side t2978_y model_layers_19_self_attn_q_proj_weight_q t2978_y model_layers_19_self_attn_k_proj_weight_q t2978_y model_layers_19_self_attn_v_proj_weight_q -> t2957_embedding t2965_c t2964_c
node rope_126_side micro s.side_2 t2957_embedding position t2965_c position -> t2957_embedding t2965_c
node kv_scatter_126_side micro s.side_3 t2965_c position s548 t2964_c position s549 -> s548 s549
node attention_19 micro s.attention t2957_embedding s548 s549 position -> t2978_y
node matmul_q4_simd_137 micro s.matmul_q4_simd_2 t2978_y model_layers_19_self_attn_o_proj_weight_q -> t2957_embedding
node add_rmsnorm_38 micro s.add_rmsnorm t2962_y t2957_embedding model_layers_19_post_attention_layernorm_weight -> t2978_y t2963_c
node matmul_q4_simd_138_side micro s.side_4 t2963_c model_layers_19_mlp_gate_proj_weight_q t2963_c model_layers_19_mlp_up_proj_weight_q -> t2986_c t2980_c
node silu_mul_32 micro s.silu_mul_3 t2986_c t2980_c -> t2979_c
node matmul_q4_simd_140 micro s.matmul_q4_simd_5 t2979_c model_layers_19_mlp_down_proj_weight_q -> t2963_c
node add_rmsnorm_39 micro s.add_rmsnorm t2978_y t2963_c model_layers_20_input_layernorm_weight -> t2957_embedding t2962_y
node matmul_q4_simd_141_side micro s.side t2962_y model_layers_20_self_attn_q_proj_weight_q t2962_y model_layers_20_self_attn_k_proj_weight_q t2962_y model_layers_20_self_attn_v_proj_weight_q -> t2963_c t2964_c t2965_c
node rope_128_side micro s.side_2 t2963_c position t2964_c position -> t2963_c t2964_c
node kv_scatter_128_side micro s.side_3 t2964_c position s550 t2965_c position s551 -> s550 s551
node attention_20 micro s.attention t2963_c s550 s551 position -> t2962_y
node matmul_q4_simd_144 micro s.matmul_q4_simd_2 t2962_y model_layers_20_self_attn_o_proj_weight_q -> t2963_c
node add_rmsnorm_40 micro s.add_rmsnorm t2957_embedding t2963_c model_layers_20_post_attention_layernorm_weight -> t2962_y t2978_y
node matmul_q4_simd_145_side micro s.side_4 t2978_y model_layers_20_mlp_gate_proj_weight_q t2978_y model_layers_20_mlp_up_proj_weight_q -> t2979_c t2980_c
node silu_mul_33 micro s.silu_mul_3 t2979_c t2980_c -> t2986_c
node matmul_q4_simd_147 micro s.matmul_q4_simd_5 t2986_c model_layers_20_mlp_down_proj_weight_q -> t2978_y
node add_rmsnorm_41 micro s.add_rmsnorm t2962_y t2978_y model_layers_21_input_layernorm_weight -> t2963_c t2957_embedding
node matmul_q4_simd_148_side micro s.side t2957_embedding model_layers_21_self_attn_q_proj_weight_q t2957_embedding model_layers_21_self_attn_k_proj_weight_q t2957_embedding model_layers_21_self_attn_v_proj_weight_q -> t2978_y t2965_c t2964_c
node rope_130_side micro s.side_2 t2978_y position t2965_c position -> t2978_y t2965_c
node kv_scatter_130_side micro s.side_3 t2965_c position s552 t2964_c position s553 -> s552 s553
node attention_21 micro s.attention t2978_y s552 s553 position -> t2957_embedding
node matmul_q4_simd_151 micro s.matmul_q4_simd_2 t2957_embedding model_layers_21_self_attn_o_proj_weight_q -> t2978_y
node add_rmsnorm_42 micro s.add_rmsnorm t2963_c t2978_y model_layers_21_post_attention_layernorm_weight -> t2957_embedding t2962_y
node matmul_q4_simd_152_side micro s.side_4 t2962_y model_layers_21_mlp_gate_proj_weight_q t2962_y model_layers_21_mlp_up_proj_weight_q -> t2986_c t2980_c
node silu_mul_34 micro s.silu_mul_3 t2986_c t2980_c -> t2979_c
node matmul_q4_simd_154 micro s.matmul_q4_simd_5 t2979_c model_layers_21_mlp_down_proj_weight_q -> t2962_y
node add_rmsnorm_43 micro s.add_rmsnorm t2957_embedding t2962_y model_norm_weight -> t2978_y t2963_c
node matmul_q4_simd_155 micro s.matmul_q4_simd t2963_c lm_head_weight_q -> t3709_c
