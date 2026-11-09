"""A ResNet-18 converts to a FIR graph and matches torch."""
import torch

import nfgraph2torch as nf2t
from resnet import ResNet18


def _randomise_bn(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_()
            mod.running_var.uniform_(0.5, 1.5)
            mod.weight.data.normal_(1, 0.1)
            mod.bias.data.normal_()


def test_resnet18_small_matches_torch():
    torch.manual_seed(0)
    m = ResNet18(classes=10, width=8).eval()
    _randomise_bn(m)
    x = torch.randn(1, 3, 32, 32)
    graph = nf2t.from_torch(m, (x,))
    got, shape = nf2t.evaluate(graph, m, (x,))
    torch.testing.assert_close(torch.tensor(got).reshape(shape), m(x), rtol=1e-4, atol=1e-4)


def test_resnet18_full_size_converts():
    m = ResNet18().eval()
    graph = nf2t.from_torch(m, (torch.randn(1, 3, 224, 224),))
    assert tuple(graph.values[graph.output].shape) == (1, 1000)
