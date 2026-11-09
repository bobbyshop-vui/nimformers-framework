"""Micro-kernel ops compiled to model.fir and executed on the FIR VM match torch."""
import pytest
import torch
import torch.nn.functional as F

from lm_harness import run_on_vm

torch.manual_seed(1)
X = torch.randn(3, 5) * 3
UNIT = torch.rand(3, 5) + 0.5
IDX = torch.tensor([[0, 2, 1, 4, 3], [4, 4, 0, 1, 2], [3, 2, 1, 0, 0]])
PERM = torch.tensor([[0, 2, 1, 4, 3], [4, 3, 0, 1, 2], [3, 2, 1, 0, 4]])

CASES = {
    "trunc": (lambda x: torch.trunc(x), X),
    "floor": (lambda x: torch.floor(x), X),
    "ceil": (lambda x: torch.ceil(x), X),
    "round": (lambda x: torch.round(x), X),
    "round_ties": (lambda x: torch.round(x), torch.tensor([[0.5, 1.5, 2.5, -0.5, -1.5], [3.5, 4.5, -2.5, 0.25, 0.75],
                                                           [10.5, 11.5, -3.5, 2.0, -7.0]])),
    "fmod": (lambda x: torch.fmod(x, 1.7), X),
    "remainder": (lambda x: torch.remainder(x, 1.7), X),
    "floor_divide": (lambda x: torch.floor_divide(x, 1.7), X),
    "erf": (lambda x: torch.erf(x), X),
    "prod": (lambda x: x.prod(1), UNIT),
    "prod_dim0": (lambda x: x.prod(0), UNIT),
    "cumprod": (lambda x: x.cumprod(1), UNIT),
    "cumprod_dim0": (lambda x: x.cumprod(0), UNIT),
    "sort": (lambda x: x.sort(1).values, X),
    "sort_desc": (lambda x: x.sort(1, descending=True).values, X),
    "argsort": (lambda x: x.argsort(1).float(), X),
    "topk": (lambda x: x.topk(3, 1).values, X),
    "topk_indices": (lambda x: x.topk(2, 1).indices.float(), X),
    "scatter_add": (lambda x: torch.zeros(3, 5).scatter_add(1, IDX, x), X),
    "scatter": (lambda x: torch.zeros(3, 5).scatter(1, PERM, x), X),
    "scatter_value": (lambda x: x.scatter(1, IDX[:, :2], 7.0), X),
    "index_put": (lambda x: x.t().index_put((IDX[0, :2],), torch.ones(2, 3)), X),
    "norm": (lambda x: torch.linalg.vector_norm(x, 2, 1), X),
    "pad_reflect": (lambda x: F.pad(x[None], (1, 2), "reflect")[0], X),
    "pad_replicate": (lambda x: F.pad(x[None], (2, 1), "replicate")[0], X),
    "gather": (lambda x: x.gather(1, IDX), X),
    "conv2d": (lambda x: F.conv2d(x, torch.arange(54.0).reshape(3, 2, 3, 3) / 30 - 0.9, None, 1, 1),
               torch.randn(1, 2, 5, 5)),
    "conv2d_grouped": (lambda x: F.conv2d(x, torch.arange(8.0).reshape(4, 1, 2, 1) / 4 - 1, None, 1, 0, 1, 2),
                       torch.randn(1, 2, 4, 3)),
    "max_pool2d": (lambda x: F.max_pool2d(x, 2), torch.randn(1, 2, 4, 4)),
    "avg_pool2d": (lambda x: F.avg_pool2d(x, 3, 2, 1), torch.randn(1, 2, 5, 5)),
    "softmax_cumsum": (lambda x: x.softmax(1).cumsum(1), X),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_on_vm(name):
    fn, data = CASES[name]
    run_on_vm(fn, data, rtol=5e-4, atol=1e-4)
