"""Run a converted op on the FIR VM: wrap it as a one-token decoder and compare with torch.

The FIR runner feeds only a token and a position and reads ``logits``. So the op under test sees a
row of an embedding table (the test data) and its flattened result is returned as the logits.
"""
import json
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

import nfgraph2torch as nf2t
from formers import fir


class LMWrap(nn.Module):
    def __init__(self, fn, data):
        super().__init__()
        self.fn, self.shape = fn, tuple(data.shape)
        rows = fn(data).numel()                 # the runner needs vocab_size embedding rows
        table = torch.zeros(rows, data.numel())
        table[0] = data.reshape(-1)
        self.table = nn.Parameter(table)

    def forward(self, token, position):
        x = F.embedding(token, self.table).reshape(self.shape)
        return self.fn(x).reshape(-1) + position.float() * 0.0


def run_on_vm(fn, data, rtol=2e-4, atol=5e-5):
    module = LMWrap(fn, data).eval()
    expected = module(torch.tensor([0]), torch.tensor([0])).detach()
    config = {"vocab_size": expected.numel(), "n_positions": 8, "model_type": "nfgraph2torch"}
    inputs = (torch.tensor([0]), torch.tensor([0]))
    graph = nf2t.from_torch(module, inputs, config=config,
                            inputs=[nf2t.Input("token"), nf2t.Input("position")])
    with tempfile.TemporaryDirectory() as d:
        path = nf2t.export(module, inputs, d, config=config,
                           inputs=[nf2t.Input("token"), nf2t.Input("position")])
        weights = {k: v.detach().flatten().tolist() for k, v in module.state_dict().items()}
        model = fir.FirModel(path, weights=weights)
        got = torch.tensor(model.step(0), dtype=torch.float32)
        text = open(path).read()
    torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
    return text
