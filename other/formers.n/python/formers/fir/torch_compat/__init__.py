"""A ``torch``-shaped namespace over ``formers.fir`` (stdlib only).

A model file written against torch traces after changing only its imports::

    from formers.fir import torch_compat as torch
    from formers.fir.torch_compat import nn
    from formers.fir.torch_compat.nn import functional as F

Tensor functions build traced operations inside ``fir.trace`` and evaluate on
Python values outside it (construction-time constants such as ``inv_freq``).
dtype objects (``torch.float32``, ``torch.bfloat16``, ``torch.int64``,
``torch.bool``, ...) follow torch promotion; values are computed in f32.
Random tensors outside a trace are shape-only placeholders (enough for
``nn.Parameter(torch.randn(n))``); inside a trace they raise ``Unsupported``.
"""

import contextlib
import functools
import math as _math

from .. import dtypes as _D
import importlib as _importlib
_T = _importlib.import_module("..tensor", __package__)
from ..tensor import (ProxyTensor as Tensor, Size, Device as device, Unsupported, tensor, as_tensor, arange,
                      linspace, full, zeros, ones, empty, eye, zeros_like, ones_like, full_like, empty_like,
                      cat, concat, concatenate, stack, where, matmul, mm, bmm, mv, einsum, outer, ger, cumsum,
                      cumprod, flip, roll, tril, triu, softmax, log_softmax, sum, mean, amax, amin, argmax,
                      argmin, var, std, var_mean, logsumexp, norm, prod, topk, sort, clamp, clip, clamp_min,
                      clamp_max, maximum, minimum, abs, neg, exp, log, sqrt, rsqrt, sin, cos, tan, tanh,
                      sigmoid, relu, pow, add, sub, subtract, mul, multiply, div, divide, true_divide,
                      floor_divide, remainder, fmod, reciprocal, square, sign, sgn, floor, ceil, round, trunc,
                      erf, exp2, expm1, log1p, log2, log10, sinh, cosh, lerp, reshape, transpose, permute,
                      flatten, unflatten, squeeze, unsqueeze, unbind, narrow, select, movedim, swapaxes,
                      broadcast_to, broadcast_tensors, broadcast_shapes, index_select, gather, masked_fill,
                      masked_select, repeat_interleave, diff, diagonal, nonzero, addmm, baddbmm, bilinear,
                      logical_and, logical_or, logical_xor, logical_not, bitwise_and, bitwise_or, bitwise_xor,
                      bitwise_not, chunk, split, rotary_embedding)
from ..dtypes import DType as dtype, finfo, iinfo, promote_types
from . import nn
from .nn import functional

_builtin_bool, _builtin_int, _builtin_float = bool, int, float

float32 = float = _D.float32
float16 = half = _D.float16
bfloat16 = _D.bfloat16
float64 = double = _D.float64
int64 = long = _D.int64
int32 = int = _D.int32
int16 = short = _D.int16
int8 = _D.int8
uint8 = _D.uint8
bool = _D.bool_

FloatTensor = LongTensor = IntTensor = BoolTensor = HalfTensor = BFloat16Tensor = DoubleTensor = Tensor
strided = "strided"
contiguous_format = preserve_format = channels_last = "memory_format"
__version__ = "2.8.0+fir"


def max(input, dim=None, keepdim=False, *, out=None): return _T.max(input, dim, keepdim)
def min(input, dim=None, keepdim=False, *, out=None): return _T.min(input, dim, keepdim)
def any(input, dim=None, keepdim=False): return _T.any(input, dim, keepdim)
def all(input, dim=None, keepdim=False): return _T.all(input, dim, keepdim)
def argsort(input, dim=-1, descending=False, stable=False): return _T.sort(input, dim, descending)[1]
def take_along_dim(input, indices, dim=None): return _T.gather(input, dim, indices)
def eq(a, b): return _T.binary("eq", a, b)
def ne(a, b): return _T.binary("ne", a, b)
def lt(a, b): return _T.binary("lt", a, b)
def le(a, b): return _T.binary("le", a, b)
def gt(a, b): return _T.binary("gt", a, b)
def ge(a, b): return _T.binary("ge", a, b)
greater, greater_equal, less, less_equal, not_equal = gt, ge, lt, le, ne
def isnan(x): return x.isnan()
def isinf(x): return x.isinf()
def isfinite(x): return x.isfinite()
def numel(x): return x.numel()
def is_tensor(x): return isinstance(x, Tensor)
def is_floating_point(x): return x.is_floating_point()
def is_complex(x): return False
def result_type(a, b): return _D.result_type(_T._dtype_operand(a), _T._dtype_operand(b))
def get_default_dtype(): return _D.float32
def can_cast(from_, to): return _D.promote_types(from_, to) is _D.dtype(to)


def set_default_dtype(d):
    if _D.dtype(d) is not _D.float32: raise Unsupported("FIR computes in f32; the default dtype stays float32")


def manual_seed(seed): return None
def seed(): return 0


def _random(*size, generator=None, dtype=None, device=None, requires_grad=False, out=None, **kwargs):
    from ..tracer import active_trace
    if active_trace() is not None:
        raise Unsupported("random tensors inside a trace need an RNG, which FIR does not have")
    return _T.Uninitialized(_T._sizes(size), dtype)


rand = randn = _random


def randint(low=0, high=None, size=None, *, dtype=None, **kwargs):
    if size is None: raise TypeError("randint needs a size")
    return _random(*size, dtype=dtype or _D.int64)


def rand_like(x, **kwargs): return _random(*x.shape, dtype=kwargs.get("dtype") or x.dtype)
randn_like = rand_like


def multinomial(*args, **kwargs): raise Unsupported("multinomial sampling needs an RNG; sample on the host")
def bernoulli(*args, **kwargs): raise Unsupported("bernoulli needs an RNG")


def polar(*args, **kwargs): raise Unsupported("complex tensors are not represented; write RoPE with rotate_half")
view_as_complex = view_as_real = polar


class _GradMode(contextlib.ContextDecorator):
    """no_grad / enable_grad / inference_mode / set_grad_enabled: no autograd, so no-ops."""
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def __call__(self, fn=None):
        if fn is None: return self
        @functools.wraps(fn)
        def wrapper(*args, **kwargs): return fn(*args, **kwargs)
        return wrapper


no_grad = enable_grad = inference_mode = set_grad_enabled = _GradMode


def is_grad_enabled(): return False
def is_inference_mode_enabled(): return True


class autocast(_GradMode):
    def __init__(self, device_type="cpu", dtype=None, enabled=True, cache_enabled=None):
        if enabled and dtype is not None and _D.dtype(dtype) is not _D.float32:
            raise Unsupported("autocast to %s: FIR computes in f32" % _D.dtype(dtype))


def compile(model=None, **kwargs):
    if model is None: return lambda m: m
    return model


class _Namespace:
    def __init__(self, **items): self.__dict__.update(items)


jit = _Namespace(is_tracing=lambda: False, is_scripting=lambda: False, script=lambda f, *a, **k: f,
                 ignore=lambda f=None, **k: (f if f is not None else (lambda g: g)), unused=lambda f: f,
                 Final=object, export=lambda f: f)
compiler = _Namespace(is_compiling=lambda: False, is_dynamo_compiling=lambda: False, disable=lambda f=None, **k: (f if f is not None else (lambda g: g)))
cuda = _Namespace(is_available=lambda: False, device_count=lambda: 0, is_bf16_supported=lambda: False,
                  current_device=lambda: 0, synchronize=lambda *a: None, empty_cache=lambda: None)
backends = _Namespace(mps=_Namespace(is_available=lambda: False), cuda=_Namespace(matmul=_Namespace(allow_tf32=False)),
                      cudnn=_Namespace(allow_tf32=False, enabled=False))
mps = _Namespace(is_available=lambda: False)
fx = _Namespace(wrap=lambda f: f)
linalg = _Namespace(vector_norm=lambda x, ord=2, dim=None, keepdim=False, dtype=None: norm(x, ord, dim, keepdim),
                    norm=lambda x, ord=None, dim=None, keepdim=False, dtype=None: norm(x, 2 if ord is None else ord, dim, keepdim),
                    matmul=matmul)
special = _Namespace(expit=sigmoid, erf=erf, logsumexp=logsumexp, softmax=softmax, log_softmax=log_softmax)
return_types = _Namespace(max=_T.MaxResult, min=_T.MinResult, sort=_T.SortResult, topk=_T.SortResult)
del contextlib
