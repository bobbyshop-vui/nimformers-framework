"""Metadata-only trace of the checkpoint-compatible TinyLlama 1.1B decode graph."""

import json
from pathlib import Path

from .. import nn
from ..emitter import export
from ..tensor import TensorSpec, arange, rotary_embedding, scaled_dot_product_attention
from ..tracer import trace


ROOT = Path(__file__).resolve().parents[6]
CONFIG_PATH = ROOT / "models" / "tinyllama" / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
INPUTS = (TensorSpec((1, 1), "token"), TensorSpec((1, 1), "position"))


class LlamaAttention(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        width = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_width = width // self.heads
        self.positions = config["max_position_embeddings"]
        self.rope_theta = float(config["rope_theta"])
        kv_width = self.kv_heads * self.head_width
        attn = prefix + ".self_attn"
        self.q_proj = nn.Linear(width, width, name=attn + ".q_proj", bias=False, layout="oi")
        self.k_proj = nn.Linear(width, kv_width, name=attn + ".k_proj", bias=False, layout="oi")
        self.v_proj = nn.Linear(width, kv_width, name=attn + ".v_proj", bias=False, layout="oi")
        self.o_proj = nn.Linear(width, width, name=attn + ".o_proj", bias=False, layout="oi")
        self.register_state("key_cache", (self.kv_heads, self.positions, self.head_width))
        self.register_state("value_cache", (self.kv_heads, self.positions, self.head_width))

    def forward(self, x, position):
        batch, seq, width = x.shape
        q = self.q_proj(x).reshape(batch, seq, self.heads, self.head_width).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq, self.kv_heads, self.head_width).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq, self.kv_heads, self.head_width).transpose(1, 2)
        q = rotary_embedding(q, position, base=self.rope_theta)
        k = rotary_embedding(k, position, base=self.rope_theta)
        keys = self.key_cache.update(k.reshape(self.kv_heads, self.head_width), position, dim=1)
        values = self.value_cache.update(v.reshape(self.kv_heads, self.head_width), position, dim=1)
        mask = arange(self.positions) <= position.reshape(())
        attended = scaled_dot_product_attention(
            q, keys.reshape(1, *keys.shape), values.reshape(1, *values.shape),
            attn_mask=mask, enable_gqa=True)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, seq, width))


class LlamaMLP(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        width, inner = config["hidden_size"], config["intermediate_size"]
        mlp = prefix + ".mlp"
        self.gate_proj = nn.Linear(width, inner, name=mlp + ".gate_proj", bias=False, layout="oi")
        self.up_proj = nn.Linear(width, inner, name=mlp + ".up_proj", bias=False, layout="oi")
        self.down_proj = nn.Linear(inner, width, name=mlp + ".down_proj", bias=False, layout="oi")

    def forward(self, x):
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        prefix = "model.layers." + str(layer)
        width, eps = config["hidden_size"], config["rms_norm_eps"]
        self.input_layernorm = nn.RMSNorm(width, eps, name=prefix + ".input_layernorm")
        self.self_attn = LlamaAttention(config, prefix)
        self.post_attention_layernorm = nn.RMSNorm(width, eps, name=prefix + ".post_attention_layernorm")
        self.mlp = LlamaMLP(config, prefix)

    def forward(self, x, position):
        x = x + self.self_attn(self.input_layernorm(x), position)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaForCausalLM(nn.Module):
    def __init__(self, config=CONFIG):
        super().__init__()
        width = config["hidden_size"]
        self.embed_tokens = nn.Embedding(config["vocab_size"], width, name="model.embed_tokens")
        self.layers = nn.ModuleList(LlamaBlock(config, i) for i in range(config["num_hidden_layers"]))
        self.norm = nn.RMSNorm(width, config["rms_norm_eps"], name="model.norm")
        self.lm_head = nn.Linear(width, config["vocab_size"], name="lm_head", bias=False, layout="oi")

    def forward(self, token, position):
        x = self.embed_tokens(token)
        for block in self.layers:
            x = block(x, position)
        return self.lm_head(self.norm(x))


REQUIRE_CONFIG = {"model_type": "llama", "hidden_act": "silu", "attention_bias": False,
                  "tie_word_embeddings": False}


def build(optimize=True):
    return trace(LlamaForCausalLM(CONFIG), INPUTS, config=CONFIG, name="tinyllama", optimize=optimize)


def export_model(directory, optimize=True, tags=False):
    graph = build(optimize)
    return export(graph, directory, require_config=REQUIRE_CONFIG, tags=tags), graph


def main(argv=None):
    import argparse
    import time
    from ..report import report
    parser = argparse.ArgumentParser(prog="python3 -m formers.fir.examples.tinyllama",
                                     description="Emit the TinyLlama decode graph as FIR (no weights are read).")
    parser.add_argument("outdir")
    parser.add_argument("--report", action="store_true", help="print node/kernel/instruction statistics")
    parser.add_argument("--tags", action="store_true", help="add op identity tags (tag=aten::...) to kernel headers")
    parser.add_argument("--no-optimize", action="store_true", help="emit generic kernels without pattern passes")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    path, graph = export_model(args.outdir, not args.no_optimize, args.tags)
    print(f"{path} ({time.perf_counter() - start:.2f}s)")
    if args.report:
        print(report(path.read_text(encoding="utf-8"), graph))


if __name__ == "__main__":
    main()
