"""GPT-2 decode with learned positions, MHA, GELU, and a tied F32 head."""

from .. import nn
from ..tensor import (TensorSpec, Dim, arange, scaled_dot_product_attention)

CONFIG = {"vocab_size": 8, "n_positions": 4, "n_embd": 4, "n_head": 2,
          "n_layer": 2, "layer_norm_epsilon": 1e-5}
INPUTS = (TensorSpec((Dim("batch", 1), Dim("seq", 1)), "token"),
          TensorSpec((1, 1), "position"))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config["n_embd"]
        self.heads, self.width = config["n_head"], h
        self.c_attn = nn.Linear(h, 3*h, layout="io")
        self.c_proj = nn.Linear(h, h, layout="io")
        # [n_positions, n_embd] caches: row = position, column = head*head_width + d.
        self.positions = config["n_positions"]
        self.register_state("key_cache", (self.positions, h))
        self.register_state("value_cache", (self.positions, h))

    def forward(self, x, position):
        b, seq, h = x.shape
        q, k, v = self.c_attn(x).chunk(3, -1)
        def heads(y): return y.reshape(b, seq, self.heads, h//self.heads).transpose(1, 2)
        q = heads(q)
        k = self.key_cache.update(k.reshape(h), position, dim=0)
        v = self.value_cache.update(v.reshape(h), position, dim=0)
        def cached(c): return c.reshape(1, self.positions, self.heads, h//self.heads).transpose(1, 2)
        mask = arange(self.positions) <= position.reshape(())
        y = scaled_dot_product_attention(q, cached(k), cached(v), attn_mask=mask)
        return self.c_proj(y.transpose(1, 2).reshape(b, seq, h))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config["n_embd"]
        self.c_fc, self.c_proj = nn.Linear(h, 4*h, layout="io"), nn.Linear(4*h, h, layout="io")
    def forward(self, x): return self.c_proj(self.c_fc(x).gelu())


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(config["n_embd"]), Attention(config)
        self.ln_2, self.mlp = nn.LayerNorm(config["n_embd"]), MLP(config)
    def forward(self, x, position):
        x = x + self.attn(self.ln_1(x), position)
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    def __init__(self, config=CONFIG):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.transformer.wpe = nn.Embedding(config["n_positions"], config["n_embd"])
        self.transformer.h = nn.ModuleList(Block(config) for _ in range(config["n_layer"]))
        self.transformer.ln_f = nn.LayerNorm(config["n_embd"])
    def forward(self, token, position):
        x = self.transformer.wte(token) + self.transformer.wpe(position)
        for block in self.transformer.h: x = block(x, position)
        return self.transformer.wte.project(self.transformer.ln_f(x))
