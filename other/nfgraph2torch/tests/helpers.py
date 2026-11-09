"""Shared checks: convert a torch callable and compare the FIR reference result with torch."""
import warnings

import torch
import torch.nn as nn

import nfgraph2torch as nf2t

warnings.filterwarnings("ignore")


class Fn(nn.Module):
    """Wrap a function of tensors as a module (optionally with parameters)."""

    def __init__(self, fn, **params):
        super().__init__()
        self.fn = fn
        self.params = bool(params)
        for k, v in params.items():
            setattr(self, k, nn.Parameter(v))

    def forward(self, *xs):
        return self.fn(self, *xs) if self.params else self.fn(*xs)


def convert_and_compare(module, *inputs, rtol=2e-4, atol=2e-5, optimize=True):
    module = module.eval()
    expected = module(*inputs)
    expected = expected[0] if isinstance(expected, (tuple, list)) else expected
    graph = nf2t.from_torch(module, inputs, optimize=optimize)
    got, shape = nf2t.evaluate(graph, module, inputs)
    assert tuple(expected.shape) == tuple(shape), (tuple(expected.shape), shape)
    got = torch.tensor(got, dtype=torch.float32).reshape(shape)
    torch.testing.assert_close(got, expected.float(), rtol=rtol, atol=atol)
    return graph
