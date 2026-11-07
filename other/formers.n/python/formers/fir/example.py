"""Export a tiny GPT-2-style embedding/MLP residual block without weight data.

From other/formers.n/python:
    python3 -m formers.fir.example /tmp/fir-script-example
"""

import sys

from . import Embedding, GELU, LayerNorm, Linear, Module, TensorSpec, export, script


CONFIG = {"vocab_size": 16, "n_positions": 8, "layer_norm_epsilon": 1e-5}
INPUTS = (TensorSpec(1, "token"), TensorSpec(1, "position"))


class TinyBlock(Module):
    def __init__(self):
        self.tokens = Embedding(16, 4, "wte")
        self.positions = Embedding(8, 4, "wpe")
        self.norm = LayerNorm(4, "ln_1")
        self.fc = Linear(4, 16, "mlp.c_fc")
        self.gelu = GELU()
        self.proj = Linear(16, 4, "mlp.c_proj")

    def forward(self, token, position):
        x = self.tokens(token) + self.positions(position)
        y = self.proj(self.gelu(self.fc(self.norm(x))))
        return self.tokens.project(x + y)


if __name__ == "__main__":
    graph = script(TinyBlock(), INPUTS, config=CONFIG, name="tiny_gpt2_block")
    print(export(graph, sys.argv[1]))
