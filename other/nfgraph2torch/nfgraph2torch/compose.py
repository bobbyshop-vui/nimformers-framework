"""ATen ops that are compositions of FIR operations (no kernel of their own).

They run on top of ``micro.trunc`` (the single rounding primitive) and plain FIR tensor math, so they
fuse into ordinary pointwise kernels. Accuracy notes are on each function.
"""
from __future__ import annotations

import math

from formers.fir import torch_compat as T

from . import aten
from . import micro

F = T.nn.functional
H = aten.HANDLERS


def floor(x):
    t = micro.trunc(x)
    return T.where(x < t, t - 1.0, t)


def ceil(x):
    t = micro.trunc(x)
    return T.where(t < x, t + 1.0, t)


def round_half_even(x):
    f = floor(x)
    d = x - f
    odd = (f - 2.0 * floor(f * 0.5)) > 0.5
    up = (d > 0.5) | ((d == 0.5) & odd)
    return T.where(up, f + 1.0, f)


def erf(x):
    """Abramowitz-Stegun 7.1.26, absolute error <= 1.5e-7."""
    ax = T.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    r = 1.0 - poly * T.exp(-(ax * ax))
    return T.where(x < 0.0, -r, r)


def fmod(x, y):
    return x - y * micro.trunc(x / y)


def remainder(x, y):
    return x - y * floor(x / y)


H["floor"] = lambda a: floor(a.self)
H["ceil"] = lambda a: ceil(a.self)
H["round"] = lambda a: round_half_even(a.self)
H["erf"] = lambda a: erf(a.self)
H["fmod"] = lambda a: fmod(a.self, a.other)
H["remainder"] = lambda a: remainder(a.self, a.other)
H["floor_divide"] = lambda a: floor(a.self / a.other)
H["isinf"] = lambda a: T.abs(a.self) > 3.4028234663852886e38
H["isnan"] = lambda a: a.self != a.self       # NaN compares unequal only where the VM keeps IEEE semantics


@aten.handler("div.Tensor_mode", "div.Scalar_mode")
def _div_mode(a):
    q = a.self / a.other
    mode = a.rounding_mode
    if mode == "trunc":
        return micro.trunc(q)
    if mode == "floor":
        return floor(q)
    return q


# --------------------------------------------------------------------------------------------
# norms
# --------------------------------------------------------------------------------------------
@aten.handler("linalg_vector_norm")
def _vector_norm(a):
    x, p = a.self, a["ord"]
    dim = tuple(a["dim"]) if a["dim"] is not None else None
    keep = a["keepdim"]
    if p == 2:
        return T.sqrt(T.sum(x * x, dim, keep))
    if p == 1:
        return T.sum(T.abs(x), dim, keep)
    if p == math.inf:
        return T.amax(T.abs(x), dim, keep)
    if p == -math.inf:
        return T.amin(T.abs(x), dim, keep)
    if p == 0:
        return T.sum((x != 0.0), dim, keep)
    return T.pow(T.sum(T.pow(T.abs(x), p), dim, keep), 1.0 / p)


# --------------------------------------------------------------------------------------------
# padding
# --------------------------------------------------------------------------------------------
def _take(x, dim, start, stop):
    ix = [slice(None)] * x.ndim
    ix[dim] = slice(start, stop)
    return x[tuple(ix)]


def pad_dim(x, dim, left, right, mode):
    parts = []
    n = x.shape[dim]
    if left:
        if mode == "reflect":
            parts.append(T.flip(_take(x, dim, 1, left + 1), (dim,)))
        else:
            parts.append(_take(x, dim, 0, 1).expand(*[left if d == dim else s for d, s in enumerate(x.shape)]))
    parts.append(x)
    if right:
        if mode == "reflect":
            parts.append(T.flip(_take(x, dim, n - right - 1, n - 1), (dim,)))
        else:
            parts.append(_take(x, dim, n - 1, n).expand(*[right if d == dim else s for d, s in enumerate(x.shape)]))
    return T.cat(parts, dim) if len(parts) > 1 else x


def pad_mode(x, pad, mode):
    """torch pad list (last dim first: left, right, top, bottom, ...)."""
    for i in range(0, len(pad), 2):
        dim = x.ndim - 1 - i // 2
        x = pad_dim(x, dim, pad[i], pad[i + 1], mode)
    return x


for _name, _mode in (("reflection_pad1d", "reflect"), ("reflection_pad2d", "reflect"), ("reflection_pad3d", "reflect"),
                     ("replication_pad1d", "replicate"), ("replication_pad2d", "replicate"),
                     ("replication_pad3d", "replicate")):
    H[_name] = (lambda mode: lambda a: pad_mode(a.self, list(a.padding), mode))(_mode)


# --------------------------------------------------------------------------------------------
# scatter family and indexing
# --------------------------------------------------------------------------------------------
@aten.handler("slice_scatter")
def _slice_scatter(a):
    x, src, dim = a.self, a.src, a["dim"] % a.self.ndim
    start = a["start"] or 0
    end = x.shape[dim] if a["end"] is None or a["end"] >= 2 ** 62 else a["end"]
    if a["step"] not in (None, 1):
        raise ValueError("slice_scatter with step != 1")
    parts = [p for p in (_take(x, dim, 0, start), src, _take(x, dim, end, x.shape[dim]))
             if p.shape[dim] > 0]
    return T.cat(parts, dim)


def _single_index(indices):
    live = [(d, i) for d, i in enumerate(indices) if i is not None]
    if len(live) != 1:
        raise ValueError("advanced indexing with %d index tensors; only one is lowered" % len(live))
    return live[0]


@aten.handler("index.Tensor")
def _index(a):
    x = a.self
    d, idx = _single_index(list(a.indices))
    flat = T.index_select(x, d, idx.reshape(-1))
    shape = tuple(x.shape[:d]) + tuple(idx.shape) + tuple(x.shape[d + 1:])
    return flat.reshape(shape)


@aten.handler("index_put")
def _index_put(a):
    x = a.self
    d, idx = _single_index(list(a.indices))
    k = idx.shape[0] if idx.ndim == 1 else None
    if k is None:
        raise ValueError("index_put with a multi-dimensional index")
    full = tuple(x.shape[:d]) + (k,) + tuple(x.shape[d + 1:])
    values = a["values"]
    values = values.expand(*full) if tuple(values.shape) != full else values
    shape = [1] * x.ndim
    shape[d] = k
    index = idx.reshape(tuple(shape)).expand(*full)
    return micro.scatter(x, d, index, values, add=bool(a["accumulate"]))


@aten.handler("scatter_reduce.two")
def _scatter_reduce(a):
    if a.reduce != "sum":
        raise ValueError("scatter_reduce(%s); only sum is lowered" % a.reduce)
    return micro.scatter(a.self, a.dim, a.index, a.src, add=True)


@aten.handler("_to_copy", "to")
def _to_copy(a):
    d = aten._fir_dtype(a["dtype"])
    if d is None or d.name in ("float64", "f64"):
        return a.self                      # FIR computes in f32; f64 is a logical tag only
    return a.self.to(d)
