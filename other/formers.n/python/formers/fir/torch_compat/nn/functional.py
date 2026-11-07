"""``torch.nn.functional`` over the fir tensor operations (torch argument order and defaults)."""

import math

import importlib as _importlib
T = _importlib.import_module("...tensor", __package__)
from ... import nn as _fir
from ... import dtypes as D
from ...tensor import Unsupported, where, maximum, minimum, cat, zeros, full


def linear(input, weight, bias=None):
    """``input @ weight.T + bias``; ``refuse_linear`` lowers a checkpoint weight to ``linear_t``."""
    y = T.matmul(input, weight.T if weight.ndim == 2 else weight)
    return y + bias if bias is not None else y


def bilinear(input1, input2, weight, bias=None): return T.bilinear(input1, input2, weight, bias)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    if max_norm is not None: raise Unsupported("embedding max_norm")
    return _fir.embedding(input, weight)


def one_hot(tensor, num_classes=-1):
    if num_classes < 0:
        num_classes = int(max(T._require_static(tensor, "one_hot without num_classes"))) + 1
    return T.cast(T.arange(num_classes) == tensor.unsqueeze(-1), D.int64)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return _fir.layer_norm(input, normalized_shape, weight, bias, eps)


def rms_norm(input, normalized_shape, weight=None, eps=None):
    return _fir.rms_norm(input, normalized_shape, weight, eps)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    n, c = input.shape[0], input.shape[1]
    x = input.reshape(n, num_groups, -1)
    m = x.mean(-1, keepdim=True)
    v = ((x - m) * (x - m)).mean(-1, keepdim=True)
    y = ((x - m) * T.rsqrt(v + eps)).reshape(tuple(input.shape))
    shape = (c,) + (1,) * (input.ndim - 2)
    if weight is not None: y = y * weight.reshape(shape)
    if bias is not None: y = y + bias.reshape(shape)
    return y


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    if training: raise Unsupported("batch_norm in training mode")
    shape = (-1,) + (1,) * (input.ndim - 2)
    y = (input - running_mean.reshape(shape)) * T.rsqrt(running_var.reshape(shape) + eps)
    if weight is not None: y = y * weight.reshape(shape)
    if bias is not None: y = y + bias.reshape(shape)
    return y


def instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True,
                  momentum=0.1, eps=1e-5):
    dims = tuple(range(2, input.ndim))
    m = input.mean(dims, keepdim=True)
    v = ((input - m) * (input - m)).mean(dims, keepdim=True)
    y = (input - m) * T.rsqrt(v + eps)
    shape = (-1,) + (1,) * (input.ndim - 2)
    if weight is not None: y = y * weight.reshape(shape)
    if bias is not None: y = y + bias.reshape(shape)
    return y


def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    return input / maximum(T.norm(input, p, dim, keepdim=True), eps)


def _implicit_dim(x, dim):
    if dim is not None: return dim
    return 0 if x.ndim in (0, 1, 3) else 1


def softmax(input, dim=None, _stacklevel=3, dtype=None): return T.softmax(input, _implicit_dim(input, dim), dtype)
def log_softmax(input, dim=None, _stacklevel=3, dtype=None): return T.log_softmax(input, _implicit_dim(input, dim), dtype)
def softmin(input, dim=None, _stacklevel=3, dtype=None): return T.softmax(-input, _implicit_dim(input, dim), dtype)


def gelu(input, approximate="none"):
    """torch default: exact (erf) GELU; ``approximate="tanh"`` is HF ``gelu_new``."""
    return T.gelu(input, approximate)


def silu(input, inplace=False): return T.silu(input)
def relu(input, inplace=False): return T.relu(input)
def relu6(input, inplace=False): return minimum(T.relu(input), 6.0)
def sigmoid(input): return T.sigmoid(input)
def tanh(input): return T.tanh(input)
def softplus(input, beta=1.0, threshold=20.0): return T.softplus(input, beta, threshold)
def mish(input, inplace=False): return input * T.tanh(T.softplus(input))
def hardswish(input, inplace=False): return input * minimum(maximum(input + 3.0, 0.0), 6.0) / 6.0
def hardsigmoid(input, inplace=False): return minimum(maximum(input + 3.0, 0.0), 6.0) / 6.0
def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False): return minimum(maximum(input, min_val), max_val)
def leaky_relu(input, negative_slope=0.01, inplace=False): return where(input > 0, input, input * negative_slope)
def elu(input, alpha=1.0, inplace=False): return where(input > 0, input, (T.exp(input) - 1.0) * alpha)
def celu(input, alpha=1.0, inplace=False): return maximum(input, 0.0) + minimum((T.exp(input / alpha) - 1.0) * alpha, 0.0)


def selu(input, inplace=False):
    alpha, scale = 1.6732632423543772, 1.0507009873554805
    return elu(input, alpha) * scale


def logsigmoid(input): return -T.softplus(-input)
def softsign(input): return input / (T.abs(input) + 1.0)
def tanhshrink(input): return input - T.tanh(input)
def threshold(input, threshold, value, inplace=False): return where(input > threshold, input, value)
def silu_(input): return T.silu(input)


def glu(input, dim=-1):
    a, b = input.chunk(2, dim)
    return a * T.sigmoid(b)


def dropout(input, p=0.5, training=True, inplace=False):
    """Identity at inference; ``training=True`` with ``p > 0`` needs an RNG and raises."""
    if training and p > 0:
        raise Unsupported("dropout with training=True needs an RNG; pass training=False (HF passes self.training)")
    return input


alpha_dropout = feature_alpha_dropout = dropout1d = dropout2d = dropout3d = dropout


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                                 enable_gqa=False):
    return T.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                          is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """torch conv1d (zeros padding, dilation 1) from the causal FIR convolution.

    With left/right padding ``pl``/``pr`` the torch output ``o[s]`` equals the causal
    convolution of ``cat(zeros(max(0, pl-K+1)), x, zeros(pr))`` at ``t = s + K-1 + a - pl``.
    """
    stride = stride[0] if isinstance(stride, (tuple, list)) else stride
    dilation = dilation[0] if isinstance(dilation, (tuple, list)) else dilation
    if dilation != 1: raise Unsupported("conv1d dilation != 1")
    unbatched = input.ndim == 2
    x = input.unsqueeze(0) if unbatched else input
    k = weight.shape[-1]
    length = x.shape[-1]
    if padding == "valid": pl = pr = 0
    elif padding == "same":
        if stride != 1: raise ValueError("padding='same' requires stride 1")
        pl = (k - 1) // 2
        pr = k - 1 - pl
    else:
        pl = pr = padding[0] if isinstance(padding, (tuple, list)) else int(padding)
    a = max(0, pl - (k - 1))
    parts = []
    if a: parts.append(zeros((x.shape[0], x.shape[1], a)))
    parts.append(x)
    if pr: parts.append(zeros((x.shape[0], x.shape[1], pr)))
    xp = cat(parts, -1) if len(parts) > 1 else x
    y = _fir.causal_conv1d(xp, weight, bias, groups)
    out = (length + pl + pr - k) // stride + 1
    if out <= 0: raise ValueError("conv1d output would be empty")
    start = k - 1 + a - pl
    y = y[:, :, start:start + (out - 1) * stride + 1:stride]
    return y.squeeze(0) if unbatched else y


def pad(input, pad, mode="constant", value=None):
    """F.pad for constant, reflect and replicate modes (static widths)."""
    x = input
    for i in range(len(pad) // 2):
        dim = x.ndim - 1 - i
        before, after = pad[2 * i], pad[2 * i + 1]
        if before < 0 or after < 0:
            ix = [slice(None)] * x.ndim
            ix[dim] = slice(-before if before < 0 else 0, x.shape[dim] + after if after < 0 else None)
            x = x[tuple(ix)]
            before, after = max(before, 0), max(after, 0)
        if not before and not after: continue
        shape = list(x.shape)
        def block(n, side):
            if mode == "constant":
                s = list(shape); s[dim] = n
                return full(tuple(s), 0.0 if value is None else value, dtype=x.dtype)
            ix = [slice(None)] * x.ndim
            if mode == "replicate":
                ix[dim] = slice(0, 1) if side == "before" else slice(shape[dim] - 1, shape[dim])
                s = list(shape); s[dim] = n
                return x[tuple(ix)].expand(tuple(s))
            if mode == "reflect":
                ix[dim] = slice(n, 0, -1) if side == "before" else slice(shape[dim] - 2, shape[dim] - 2 - n, -1)
                return x[tuple(ix)]
            if mode == "circular":
                ix[dim] = slice(shape[dim] - n, shape[dim]) if side == "before" else slice(0, n)
                return x[tuple(ix)]
            raise Unsupported("pad mode %r" % mode)
        parts = ([block(before, "before")] if before else []) + [x] + ([block(after, "after")] if after else [])
        x = cat(parts, dim)
    return x


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    dot = (x1 * x2).sum(dim)
    return dot / maximum(T.sqrt((x1 * x1).sum(dim)) * T.sqrt((x2 * x2).sum(dim)), eps)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    return T.norm(x1 - x2 + eps, p, -1, keepdim)


def interpolate(*args, **kwargs): raise Unsupported("interpolate: write nearest resizing as expand + reshape")
def grid_sample(*args, **kwargs): raise Unsupported("grid_sample (data-dependent gather)")
def cross_entropy(*args, **kwargs): raise Unsupported("losses are training-only")
nll_loss = mse_loss = cross_entropy
