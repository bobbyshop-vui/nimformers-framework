"""torch fn -> FIR (model.fir) -> torch again, compared with the original."""
import tempfile

import torch

import nfgraph2torch as nf2t
from lm_harness import LMWrap


def roundtrip(fn, data, rtol=5e-4, atol=1e-4):
    module = LMWrap(fn, data).eval()
    expected = module(torch.tensor([0]), torch.tensor([0])).detach()
    config = {"vocab_size": expected.numel(), "n_positions": 8, "model_type": "nfgraph2torch"}
    inputs = (torch.tensor([0]), torch.tensor([0]))
    roles = [nf2t.Input("token"), nf2t.Input("position")]
    with tempfile.TemporaryDirectory() as d:
        path = nf2t.export(module, inputs, d, config=config, inputs=roles)
        back = nf2t.to_torch(path, weights=dict(module.state_dict()))
        with torch.no_grad():
            got = back(inputs[0], inputs[1])
    torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
    return back
