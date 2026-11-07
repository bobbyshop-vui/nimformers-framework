"""Metadata-only trace of the checkpoint-compatible GPT-2 125M decode graph."""

import json
from pathlib import Path

from .. import nn
from ..emitter import export
from ..tensor import TensorSpec, arange, scaled_dot_product_attention
from ..tracer import trace


ROOT = Path(__file__).resolve().parents[6]
CONFIG_PATH = ROOT / "models" / "gpt2-125m" / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
INPUTS = (TensorSpec((1, 1), "token"), TensorSpec((1, 1), "position"))


class GPT2Attention(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        width, heads = config["n_embd"], config["n_head"]
        self.heads, self.width = heads, width
        self.positions = config["n_positions"]
        self.c_attn = nn.Linear(width, 3 * width, name=prefix + ".attn.c_attn",
                                layout="io", dtype="q4")
        self.c_proj = nn.Linear(width, width, name=prefix + ".attn.c_proj",
                                layout="io", dtype="q4")
        # One [n_positions, n_embd] row-major cache per layer, as in the runner's
        # hand-written GPT-2 file: row = position, column = head * head_width + d.
        self.register_state("key_cache", (self.positions, width))
        self.register_state("value_cache", (self.positions, width))

    def forward(self, x, position):
        batch, seq, width = x.shape
        q, k, v = self.c_attn(x).chunk(3, -1)
        head_width = width // self.heads

        def split_heads(value):
            return value.reshape(batch, seq, self.heads, head_width).transpose(1, 2)

        q = split_heads(q)
        keys = self.key_cache.update(k.reshape(width), position, dim=0)
        values = self.value_cache.update(v.reshape(width), position, dim=0)

        def cache_heads(cache):
            return cache.reshape(1, self.positions, self.heads, head_width).transpose(1, 2)

        mask = arange(self.positions) <= position.reshape(())
        attended = scaled_dot_product_attention(
            q, cache_heads(keys), cache_heads(values), attn_mask=mask
        )
        return self.c_proj(attended.transpose(1, 2).reshape(batch, seq, width))


class GPT2MLP(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        width = config["n_embd"]
        self.c_fc = nn.Linear(width, 4 * width, name=prefix + ".mlp.c_fc",
                              layout="io", dtype="q4")
        self.c_proj = nn.Linear(4 * width, width, name=prefix + ".mlp.c_proj",
                                layout="io", dtype="q4")

    def forward(self, x):
        return self.c_proj(self.c_fc(x).gelu())


class GPT2Block(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        prefix = "transformer.h." + str(layer)
        width = config["n_embd"]
        epsilon = config["layer_norm_epsilon"]
        self.ln_1 = nn.LayerNorm(width, name=prefix + ".ln_1", eps=epsilon)
        self.attn = GPT2Attention(config, prefix)
        self.ln_2 = nn.LayerNorm(width, name=prefix + ".ln_2", eps=epsilon)
        self.mlp = GPT2MLP(config, prefix)

    def forward(self, x, position):
        x = x + self.attn(self.ln_1(x), position)
        return x + self.mlp(self.ln_2(x))


class GPT2LMHead(nn.Module):
    def __init__(self, config=CONFIG):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(
            config["vocab_size"], config["n_embd"], name="transformer.wte"
        )
        self.transformer.wpe = nn.Embedding(
            config["n_positions"], config["n_embd"], name="transformer.wpe"
        )
        self.transformer.h = nn.ModuleList(
            GPT2Block(config, i) for i in range(config["n_layer"])
        )
        self.transformer.ln_f = nn.LayerNorm(
            config["n_embd"], name="transformer.ln_f",
            eps=config["layer_norm_epsilon"]
        )

    def forward(self, token, position):
        x = self.transformer.wte(token) + self.transformer.wpe(position)
        for block in self.transformer.h:
            x = block(x, position)
        return self.transformer.wte.project(self.transformer.ln_f(x))


# Config checks of the hand-written models/gpt2-125m/model.fir (absent keys pass).
REQUIRE_CONFIG = {"model_type": "gpt2", "activation_function": "gelu_new", "scale_attn_weights": True,
                  "scale_attn_by_inverse_layer_idx": False, "reorder_and_upcast_attn": False,
                  "add_cross_attention": False}
HAND_WRITTEN = ROOT / "models" / "gpt2-125m" / "model.fir"


def build(optimize=True):
    return trace(GPT2LMHead(CONFIG), INPUTS, config=CONFIG, name="gpt2_125m", optimize=optimize)


def export_model(directory, optimize=True, tags=False):
    graph = build(optimize)
    return export(graph, directory, prefix="transformer.", require_config=REQUIRE_CONFIG, tags=tags), graph


def main(argv=None):
    import argparse
    import time
    from ..report import report
    parser = argparse.ArgumentParser(prog="python3 -m formers.fir.examples.gpt2_125m",
                                     description="Emit the GPT-2 125M decode graph as FIR (no weights are read).")
    parser.add_argument("outdir")
    parser.add_argument("--report", action="store_true", help="print node/kernel/instruction statistics")
    parser.add_argument("--tags", action="store_true", help="add op identity tags (tag=aten::...) to kernel headers")
    parser.add_argument("--no-optimize", action="store_true", help="emit generic kernels without pattern passes")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    path, graph = export_model(args.outdir, not args.no_optimize, args.tags)
    elapsed = time.perf_counter() - start
    print(f"{path} ({elapsed:.2f}s)")
    if args.report:
        print(report(path.read_text(encoding="utf-8"), graph))
        if HAND_WRITTEN.exists():
            print("hand-written reference:")
            print(report(HAND_WRITTEN.read_text(encoding="utf-8"), config=CONFIG))


if __name__ == "__main__":
    main()
