"""``torch.nn`` with torch constructor signatures over the fir modules."""

from ... import nn as _fir
from ...nn import (Module, Parameter, Buffer, ModuleList, ModuleDict, ParameterList, ParameterDict, Sequential,
                   Identity, State)
import importlib as _importlib
_T = _importlib.import_module("...tensor", __package__)
from . import functional as F
from . import functional, init


class Linear(_fir.Linear):
    """torch.nn.Linear: weight ``[out_features, in_features]``."""
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, layout="oi")


class Embedding(_fir.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        if max_norm is not None: raise _T.Unsupported("Embedding max_norm renormalizes rows at lookup")
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)


class LayerNorm(_fir.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias)


class RMSNorm(_fir.RMSNorm):
    def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
        shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        if len(shape) != 1: raise _T.Unsupported("RMSNorm over more than the last dimension")
        super().__init__(shape[0], eps=eps, elementwise_affine=elementwise_affine)


class Conv1d(Module):
    """torch.nn.Conv1d (zeros padding, dilation 1): a causal FIR conv over padded input, sliced and strided."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode="zeros", device=None, dtype=None):
        super().__init__()
        kernel_size = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else kernel_size
        if padding_mode != "zeros": raise _T.Unsupported("Conv1d padding_mode %r" % padding_mode)
        if (dilation if isinstance(dilation, int) else dilation[0]) != 1: raise _T.Unsupported("Conv1d dilation != 1")
        if in_channels % groups or out_channels % groups: raise ValueError("groups must divide the channels")
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, (kernel_size,)
        self.stride = stride if isinstance(stride, tuple) else (stride,)
        self.padding = padding if isinstance(padding, (str, tuple)) else (padding,)
        self.dilation, self.groups, self.padding_mode = (1,), groups, padding_mode
        self.weight = Parameter((out_channels, in_channels // groups, kernel_size))
        self.bias = Parameter((out_channels,)) if bias else None
    def forward(self, x):
        padding = self.padding if isinstance(self.padding, str) else self.padding[0]
        return F.conv1d(x, self.weight, self.bias, self.stride[0], padding, 1, self.groups)


class Dropout(_fir.Dropout):
    pass


class _Activation(Module):
    fn = None
    def __init__(self, *args, inplace=False, **kwargs):
        super().__init__()
        self.args, self.kwargs = args, kwargs
    def forward(self, x): return type(self).fn(x, *self.args, **self.kwargs)


def _activation(name, fn, doc=""):
    return type(name, (_Activation,), {"fn": staticmethod(fn), "__doc__": doc})


class GELU(Module):
    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate
    def forward(self, x): return F.gelu(x, approximate=self.approximate)


SiLU = _activation("SiLU", lambda x: F.silu(x))
ReLU = _activation("ReLU", lambda x: F.relu(x))
ReLU6 = _activation("ReLU6", lambda x: F.relu6(x))
Tanh = _activation("Tanh", lambda x: F.tanh(x))
Sigmoid = _activation("Sigmoid", lambda x: F.sigmoid(x))
Mish = _activation("Mish", lambda x: F.mish(x))
Hardswish = _activation("Hardswish", lambda x: F.hardswish(x))
Hardsigmoid = _activation("Hardsigmoid", lambda x: F.hardsigmoid(x))
LogSigmoid = _activation("LogSigmoid", lambda x: F.logsigmoid(x))
Softsign = _activation("Softsign", lambda x: F.softsign(x))
Tanhshrink = _activation("Tanhshrink", lambda x: F.tanhshrink(x))


class Softplus(Module):
    def __init__(self, beta=1.0, threshold=20.0):
        super().__init__()
        self.beta, self.threshold = beta, threshold
    def forward(self, x): return F.softplus(x, self.beta, self.threshold)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x): return F.leaky_relu(x, self.negative_slope)


class ELU(Module):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__()
        self.alpha = alpha
    def forward(self, x): return F.elu(x, self.alpha)


class CELU(ELU):
    def forward(self, x): return F.celu(x, self.alpha)


class SELU(Module):
    def __init__(self, inplace=False): super().__init__()
    def forward(self, x): return F.selu(x)


class Hardtanh(Module):
    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False):
        super().__init__()
        self.min_val, self.max_val = min_val, max_val
    def forward(self, x): return F.hardtanh(x, self.min_val, self.max_val)


class Threshold(Module):
    def __init__(self, threshold, value, inplace=False):
        super().__init__()
        self.threshold, self.value = threshold, value
    def forward(self, x): return F.threshold(x, self.threshold, self.value)


class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
        super().__init__()
        self.weight = Parameter((num_parameters,))
    def forward(self, x):
        w = self.weight if self.weight.shape[0] == 1 or x.ndim < 2 else \
            self.weight.reshape((-1,) + (1,) * (x.ndim - 2))
        return _T.where(x > 0, x, x * w)


class GLU(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x): return F.glu(x, self.dim)


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim
    def forward(self, x): return F.softmax(x, self.dim)


class LogSoftmax(Softmax):
    def forward(self, x): return F.log_softmax(x, self.dim)


class Softmin(Softmax):
    def forward(self, x): return F.softmax(-x, self.dim)


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim
    def forward(self, x): return x.flatten(self.start_dim, self.end_dim)


class Unflatten(Module):
    def __init__(self, dim, unflattened_size):
        super().__init__()
        self.dim, self.unflattened_size = dim, unflattened_size
    def forward(self, x): return x.unflatten(self.dim, self.unflattened_size)


class CosineSimilarity(Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim, self.eps = dim, eps
    def forward(self, a, b): return F.cosine_similarity(a, b, self.dim, self.eps)


class PairwiseDistance(Module):
    def __init__(self, p=2.0, eps=1e-6, keepdim=False):
        super().__init__()
        self.p, self.eps, self.keepdim = p, eps, keepdim
    def forward(self, a, b): return F.pairwise_distance(a, b, self.p, self.eps, self.keepdim)


class utils:
    """torch.nn.utils subset."""
    class parametrize:
        @staticmethod
        def is_parametrized(module, name=None): return False
