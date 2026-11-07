"""Llama-style decode with RMSNorm, RoPE, grouped queries, and SwiGLU."""

from .. import nn
from ..tensor import (arange, rotary_embedding, scaled_dot_product_attention)
from .gpt2 import INPUTS

CONFIG = {"vocab_size": 8, "n_positions": 4, "hidden_size": 4, "intermediate_size": 8,
          "num_attention_heads": 2, "num_key_value_heads": 1, "num_hidden_layers": 2,
          "rms_norm_eps": 1e-6, "layer_norm_epsilon": 1e-5, "use_rope": True}


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c["hidden_size"]
        self.heads, self.kvheads = c["num_attention_heads"], c["num_key_value_heads"]
        self.d, self.positions, self.use_rope = h//self.heads, c["n_positions"], c["use_rope"]
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, self.kvheads*self.d, bias=False)
        self.v_proj = nn.Linear(h, self.kvheads*self.d, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.register_state("key_cache", (self.kvheads, self.positions, self.d))
        self.register_state("value_cache", (self.kvheads, self.positions, self.d))
    def forward(self, x, position):
        b, seq, h = x.shape
        q = self.q_proj(x).reshape(b, seq, self.heads, self.d).transpose(1, 2)
        k = self.k_proj(x).reshape(b, seq, self.kvheads, self.d).transpose(1, 2)
        v = self.v_proj(x).reshape(b, seq, self.kvheads, self.d).transpose(1, 2)
        if self.use_rope:
            q, k = rotary_embedding(q, position), rotary_embedding(k, position)
        k = self.key_cache.update(k.reshape(self.kvheads, self.d), position, dim=1)
        v = self.value_cache.update(v.reshape(self.kvheads, self.d), position, dim=1)
        mask = arange(self.positions) <= position.reshape(())
        y = scaled_dot_product_attention(q, k.reshape(1, *k.shape), v.reshape(1, *v.shape),
                                         attn_mask=mask, enable_gqa=True)
        return self.o_proj(y.transpose(1, 2).reshape(b, seq, h))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        h, m = c["hidden_size"], c["intermediate_size"]
        self.gate_proj, self.up_proj = nn.Linear(h,m,bias=False), nn.Linear(h,m,bias=False)
        self.down_proj = nn.Linear(m,h,bias=False)
    def forward(self, x): return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.self_attn = Attention(c)
        self.post_attention_layernorm = nn.RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.mlp = MLP(c)
    def forward(self, x, position):
        x = x + self.self_attn(self.input_layernorm(x), position)
        return x + self.mlp(self.post_attention_layernorm(x))


class Llama(nn.Module):
    def __init__(self, config=CONFIG):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.model.layers = nn.ModuleList(Block(config) for _ in range(config["num_hidden_layers"]))
        self.model.norm = nn.RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
    def forward(self, token, position):
        x = self.model.embed_tokens(token)
        for block in self.model.layers: x = block(x, position)
        return self.lm_head(self.model.norm(x))
