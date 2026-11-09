"""One conversion per ATen op family: torch result == FIR reference result."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import Fn, convert_and_compare

torch.manual_seed(0)
X = torch.randn(3, 5)
Y = torch.randn(3, 5)
POS = torch.rand(3, 5) + 0.5
UNIT = torch.rand(3, 5) * 1.8 - 0.9
IDX = torch.tensor([[0, 2, 1, 4, 3], [4, 4, 0, 1, 2], [3, 2, 1, 0, 0]])
IMG = torch.randn(1, 4, 8, 8)
SEQ = torch.randn(1, 4, 12)

UNARY = {
    "abs": torch.abs, "neg": torch.neg, "exp": torch.exp, "sin": torch.sin, "cos": torch.cos,
    "tanh": torch.tanh, "sigmoid": torch.sigmoid, "relu": torch.relu, "floor": torch.floor,
    "ceil": torch.ceil, "trunc": torch.trunc, "sign": torch.sign, "reciprocal": torch.reciprocal,
    "expm1": torch.expm1, "erf": torch.erf, "silu": F.silu, "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"), "softplus": F.softplus,
    "leaky_relu": F.leaky_relu, "elu": F.elu, "hardtanh": F.hardtanh, "mish": F.mish,
    "hardswish": F.hardswish, "hardsigmoid": F.hardsigmoid, "relu6": F.relu6, "square": torch.square,
    "round": torch.round, "sinh": torch.sinh, "cosh": torch.cosh, "tan": torch.tan,
    "isnan": lambda x: torch.isnan(x).float(), "isinf": lambda x: torch.isinf(x).float(),
}
POSITIVE = {"log": torch.log, "sqrt": torch.sqrt, "rsqrt": torch.rsqrt, "log1p": torch.log1p,
            "log2": torch.log2, "log10": torch.log10}
BINARY = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "div": lambda a, b: a / (b.abs() + 0.5), "pow": lambda a, b: a.abs().pow(b), "maximum": torch.maximum,
    "minimum": torch.minimum, "where": lambda a, b: torch.where(a > b, a, b), "rsub": lambda a, b: 2.0 - a,
    "add_alpha": lambda a, b: torch.add(a, b, alpha=2), "lt": lambda a, b: (a < b).float(),
    "eq": lambda a, b: (a == a).float(), "clamp": lambda a, b: torch.clamp(a, -0.3, 0.4),
    "fmod": lambda a, b: torch.fmod(a, 0.7), "remainder": lambda a, b: torch.remainder(a, 0.7),
    "floor_div": lambda a, b: torch.floor_divide(a, 0.7), "mul_scalar": lambda a, b: a * 3.5,
    "addcmul": lambda a, b: torch.addcmul(a, a, b, value=0.5),
    "lerp": lambda a, b: torch.lerp(a, b, 0.25),
}
REDUCE = {
    "sum": lambda x: x.sum(), "sum_dim": lambda x: x.sum(1), "sum_keep": lambda x: x.sum(0, keepdim=True),
    "mean": lambda x: x.mean(1), "amax": lambda x: x.amax(1), "amin": lambda x: x.amin(0),
    "max_dim": lambda x: x.max(1).values, "argmax": lambda x: x.argmax(1).float(),
    "argmin": lambda x: x.argmin(0).float(), "var": lambda x: x.var(1), "std": lambda x: x.std(1),
    "cumsum": lambda x: x.cumsum(1), "logsumexp": lambda x: x.logsumexp(1),
    "softmax": lambda x: x.softmax(1), "log_softmax": lambda x: x.log_softmax(0),
    "prod": lambda x: x.prod(1), "prod_all": lambda x: x.prod(), "cumprod": lambda x: x.cumprod(1),
    "any": lambda x: (x > 0.5).any(1).float(), "all": lambda x: (x > -3).all(1).float(),
    "norm": lambda x: torch.linalg.vector_norm(x, 2, 1),
}
SHAPE = {
    "view": lambda x: x.view(5, 3), "reshape": lambda x: x.reshape(15), "permute": lambda x: x.permute(1, 0),
    "transpose": lambda x: x.transpose(0, 1), "squeeze": lambda x: x.unsqueeze(1).squeeze(1),
    "expand": lambda x: x[:1].expand(4, 5), "slice": lambda x: x[:, 1:4], "slice_step": lambda x: x[:, ::2],
    "select": lambda x: x[1], "narrow": lambda x: x.narrow(1, 1, 3), "flatten": lambda x: x.flatten(),
    "cat": lambda x: torch.cat([x, x * 2], 1), "stack": lambda x: torch.stack([x, x + 1], 0),
    "split": lambda x: torch.split(x, 2, 1)[1], "chunk": lambda x: torch.chunk(x, 2, 1)[0],
    "repeat": lambda x: x.repeat(2, 1), "roll": lambda x: x.roll(1, 1), "flip": lambda x: x.flip(1),
    "tril": lambda x: x.tril(), "triu": lambda x: x.triu(1), "unbind": lambda x: x.unbind(0)[2],
    "pad": lambda x: F.pad(x, (1, 2), value=0.5), "pad_reflect": lambda x: F.pad(x[None], (1, 2), "reflect")[0],
    "masked_fill": lambda x: x.masked_fill(x > 0, 0.25), "diag_embed": lambda x: x * torch.eye(3, 5),
    "arange": lambda x: x + torch.arange(5.0), "full": lambda x: x + torch.full((3, 5), 2.0),
    "zeros_like": lambda x: x + torch.zeros_like(x), "ones": lambda x: x * torch.ones(5),
    "to_double": lambda x: x.double().float(), "clone": lambda x: x.clone() + 1,
}
INDEXING = {
    "gather": lambda x: x.gather(1, IDX), "index_select": lambda x: x.index_select(1, IDX[0]),
    "embedding": lambda x: F.embedding(IDX[0], x.t()), "take_along_dim": lambda x: torch.take_along_dim(x, IDX, 1),
    "index_tensor": lambda x: x.t()[IDX[0]], "scatter_add": lambda x: torch.zeros(3, 5).scatter_add(1, IDX, x),
    "scatter": lambda x: torch.zeros(3, 5).scatter(1, IDX[:, :1].expand(3, 5) * 0 + torch.arange(5), x),
    "sort": lambda x: x.sort(1).values, "sort_desc": lambda x: x.sort(1, descending=True).indices.float(),
    "topk": lambda x: x.topk(2, 1).values, "argsort": lambda x: x.argsort(1).float(),
    "index_put": lambda x: x.index_put((IDX[0, :2],), torch.ones(2, 5)),
    "select_scatter": lambda x: torch.select_scatter(x, torch.ones(5), 0, 1),
    "slice_scatter": lambda x: torch.slice_scatter(x, torch.ones(3, 2), 1, 1, 3),
    "one_hot": lambda x: F.one_hot(IDX[0], 5).float() + x[0],
    "nonzero_free_cumsum": lambda x: torch.cumsum((x > 0).float(), 1),
}


def _run(fn, *inputs, **kw):
    return convert_and_compare(Fn(fn), *inputs, **kw)


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary(name):
    _run(UNARY[name], UNIT if name in ("tan",) else X, atol=3e-5)


@pytest.mark.parametrize("name", sorted(POSITIVE))
def test_positive_unary(name):
    _run(POSITIVE[name], POS, atol=3e-5)


@pytest.mark.parametrize("name", sorted(BINARY))
def test_binary(name):
    _run(BINARY[name], X, Y, atol=3e-5)


@pytest.mark.parametrize("name", sorted(REDUCE))
def test_reduce(name):
    _run(REDUCE[name], X if name not in ("prod", "prod_all", "cumprod") else UNIT + 1.0, atol=5e-5)


@pytest.mark.parametrize("name", sorted(SHAPE))
def test_shape(name):
    _run(SHAPE[name], X)


@pytest.mark.parametrize("name", sorted(INDEXING))
def test_indexing(name):
    _run(INDEXING[name], X)


W2 = torch.randn(6, 4, 3, 3)
WG = torch.randn(4, 2, 3, 3)
BG = torch.randn(4)
W1 = torch.randn(5, 4, 3)
WINDOW = {
    "conv2d": (lambda t: F.conv2d(t, W2, None, 1, 1), IMG),
    "conv2d_grouped_strided": (lambda t: F.conv2d(t, WG, BG, 2, 0, 1, 2), IMG),
    "conv2d_dilated": (lambda t: F.conv2d(t, W2, None, 1, 2, 2), IMG),
    "conv1d": (lambda t: F.conv1d(t, W1, None, 1, 1), SEQ),
    "max_pool2d": (lambda t: F.max_pool2d(t, 2), IMG),
    "max_pool2d_padded": (lambda t: F.max_pool2d(t, 3, 2, 1), IMG),
    "avg_pool2d": (lambda t: F.avg_pool2d(t, 3, 2, 1), IMG),
    "adaptive_avg_pool2d": (lambda t: F.adaptive_avg_pool2d(t, (2, 2)), IMG),
    "upsample_nearest_scale": (lambda t: F.interpolate(t, scale_factor=2, mode="nearest"), IMG),
    "upsample_nearest_size": (lambda t: F.interpolate(t, size=(5, 12), mode="nearest"), IMG),
}


@pytest.mark.parametrize("name", sorted(WINDOW))
def test_window(name):
    fn, data = WINDOW[name]
    convert_and_compare(Fn(fn), data, atol=5e-5)
