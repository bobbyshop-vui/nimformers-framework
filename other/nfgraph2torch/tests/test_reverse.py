"""model.fir -> torch: every micro-kernel op round-trips (torch -> FIR -> torch)."""
import pytest
import torch

import nfgraph2torch as nf2t
from roundtrip import roundtrip
from test_vm import CASES
from resnet import ResNet18
from lm_harness import LMWrap


@pytest.mark.parametrize("name", sorted(CASES))
def test_roundtrip(name):
    fn, data = CASES[name]
    roundtrip(fn, data)


def _resnet_lm(width, size, classes):
    """ResNet fed from an embedding row: token t selects image t, the logits are the class scores."""
    net = ResNet18(classes=classes, width=width).eval()
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(); m.running_var.uniform_(0.5, 1.5)
            m.weight.data.normal_(1, 0.1); m.bias.data.normal_()
    images = torch.randn(classes, 3 * size * size)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.images = torch.nn.Parameter(images)

        def forward(self, token, position):
            x = torch.nn.functional.embedding(token, self.images).reshape(1, 3, size, size)
            return self.net(x).reshape(-1) + position.float() * 0.0
    return Net().eval()


def test_resnet18_roundtrip(tmp_path):
    torch.manual_seed(0)
    module = _resnet_lm(width=4, size=16, classes=6)
    inputs = (torch.tensor([3]), torch.tensor([0]))
    expected = module(*inputs).detach()
    config = {"vocab_size": 6, "n_positions": 8, "model_type": "resnet18"}
    roles = [nf2t.Input("token"), nf2t.Input("position")]
    path = nf2t.export(module, inputs, tmp_path, config=config, inputs=roles)
    back = nf2t.to_torch(path, weights=dict(module.state_dict()))
    with torch.no_grad():
        got = back(*inputs)
    torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)
