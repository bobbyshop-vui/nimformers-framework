"""Convolution and pooling micro-kernels (N-d, NC<spatial> layout).

One FIR operation, ``window``, covers them all: it reduces a sliding window of the input, either
against a weight (``mode="conv"``: sum of x*w over the input channels of the group and the kernel) or
on its own (``mode="pool"``: sum, mean or max over the kernel). The lowering describes the window
with strided views and hands the reduction to ``codegen.gen_reduce``, so the generated micro-kernel
is a lane over the output with three nested loops over the window; no gather, no bounds checks.
Padding is materialised first (zeros for conv and avg-pool, -F32max for max-pool).
"""
from __future__ import annotations

import itertools
import math

from formers import fir
from formers.fir import codegen as C
from formers.fir import views as V
from formers.fir import torch_compat as T

from . import aten

F = T.nn.functional
NEG_LARGE = -3.0e38


def _tuple(v, n):
    if isinstance(v, (list, tuple)):
        return tuple(v) if len(v) == n else tuple(v) * n
    return (v,) * n


def out_size(n, k, stride, dilation):
    return (n - dilation * (k - 1) - 1) // stride + 1


def _geometry(attrs, x_shape):
    s = attrs["nsp"]
    sp = x_shape[2:]
    out_sp = tuple(out_size(sp[i], attrs["kernel"][i], attrs["stride"][i], attrs["dilation"][i]) for i in range(s))
    return s, sp, out_sp


def _window_views(attrs, xview, wview, x_shape, w_shape):
    """Strided views of x (and w) over the space (N, Co, *out, Cig, *kernel)."""
    s, sp, out_sp = _geometry(attrs, x_shape)
    k, stride, dil, groups = attrs["kernel"], attrs["stride"], attrs["dilation"], attrs["groups"]
    co = attrs["co"]
    cig = x_shape[1] // groups if attrs["mode"] == "conv" else 1
    xd = xview.dims
    if any(d[2] != 1 for d in xd):
        raise C.LoweringError("window input is a grouped view; materialize it first")
    xdims = [(x_shape[0], xd[0][1], 1)]
    if attrs["mode"] == "conv":
        xdims.append((co, xd[1][1] * cig, co // groups) if groups > 1 else (co, 0, 1))
    else:
        xdims.append((x_shape[1], xd[1][1], 1))
    xdims += [(out_sp[i], stride[i] * xd[2 + i][1], 1) for i in range(s)]
    xdims.append((cig, xd[1][1], 1))
    xdims += [(k[i], dil[i] * xd[2 + i][1], 1) for i in range(s)]
    xv = V.View(xview.offset, tuple(xdims))
    if attrs["mode"] != "conv":
        # pool: (N, C, *out, 1, *k) -> drop the unit channel dim
        xv = V.View(xview.offset, tuple(d for i, d in enumerate(xdims) if i != 2 + s))
        return xv, None
    wd = wview.dims
    wdims = [(x_shape[0], 0, 1), (co, wd[0][1], 1)] + [(out_sp[i], 0, 1) for i in range(s)]
    wdims.append((cig, wd[1][1], 1))
    wdims += [(k[i], wd[2 + i][1], 1) for i in range(s)]
    return xv, V.View(wview.offset, tuple(wdims))


def _lower(em, node, ctx, kb):
    g = em.graph
    a = node.attrs
    x = em.operand(node.inputs[0], ctx, "mem")
    shape = g.values[node.output].shape
    out = C.Mem(ctx.port_out(em.output_buffer(node.output)), V.identity(shape))
    x_shape = g.values[node.inputs[0]].shape
    s = a["nsp"]
    w = em.operand(node.inputs[1], ctx, "mem") if a["mode"] == "conv" else None
    w_shape = g.values[node.inputs[1]].shape if w is not None else None
    xv, wv = _window_views(a, x.view, w.view if w is not None else None, x_shape, w_shape)
    space = tuple(d[0] for d in xv.dims)
    n_out = 2 + s
    inner = list(range(n_out, len(space))) if a["mode"] != "conv" else list(range(n_out, len(space)))
    operands = [C.Mem(x.port, xv)] + ([C.Mem(w.port, wv)] if w is not None else [])
    if a["mode"] == "conv":
        tree, kind = ("mul", (("input", 0), ("input", 1))), "sum"
    else:
        tree, kind = ("input", 0), a["kind"]
    threads = C.gen_reduce(kb, kind, tree, operands, out, space, inner, False)
    return ("conv" if a["mode"] == "conv" else "pool_" + a["kind"]), threads


def _reference(ins, shapes, out_shape, a):
    s = a["nsp"]
    x, x_shape = ins[0], shapes[0]
    w = ins[1] if a["mode"] == "conv" else None
    k, stride, dil, groups = a["kernel"], a["stride"], a["dilation"], a["groups"]
    n, c = x_shape[0], x_shape[1]
    sp = x_shape[2:]
    out_sp = out_shape[2:]
    co = out_shape[1]
    cig = c // groups
    xs = _strides(x_shape)
    result = []
    for b in range(n):
        for o in range(co):
            for pos in itertools.product(*(range(v) for v in out_sp)):
                values = []
                chans = range((o // (co // groups)) * cig, (o // (co // groups) + 1) * cig) \
                    if a["mode"] == "conv" else [o]
                for ci_i, ci in enumerate(chans):
                    for kp in itertools.product(*(range(v) for v in k)):
                        at = [pos[i] * stride[i] + kp[i] * dil[i] for i in range(s)]
                        xv = x[b * xs[0] + ci * xs[1] + sum(at[i] * xs[2 + i] for i in range(s))]
                        if w is not None:
                            ws = _strides(shapes[1])
                            xv *= w[o * ws[0] + ci_i * ws[1] + sum(kp[i] * ws[2 + i] for i in range(s))]
                        values.append(xv)
                if a["mode"] == "conv" or a["kind"] == "sum":
                    result.append(sum(values))
                elif a["kind"] == "mean":
                    result.append(sum(values) / len(values))
                else:
                    result.append(max(values))
    return result


def _strides(shape):
    out, step = [], 1
    for d in reversed(shape):
        out.append(step)
        step *= d
    return list(reversed(out))


fir.register_op("window", reference=_reference, lower=_lower, replace=True)


# --------------------------------------------------------------------------------------------
# front end
# --------------------------------------------------------------------------------------------
def _pad(x, padding, value):
    """Pad every spatial dim symmetrically; `padding` is per spatial dim."""
    if not any(padding):
        return x
    flat = []
    for p in reversed(padding):
        flat += [p, p]
    return F.pad(x, tuple(flat), "constant", value)


def conv(x, w, bias, stride, padding, dilation, groups):
    nsp = x.ndim - 2
    stride, padding, dilation = _tuple(stride, nsp), _tuple(padding, nsp), _tuple(dilation, nsp)
    x = _pad(x, padding, 0.0)
    k = tuple(w.shape[2:])
    attrs = {"mode": "conv", "nsp": nsp, "kernel": k, "stride": stride, "dilation": dilation,
             "groups": int(groups), "co": w.shape[0]}
    out_sp = tuple(out_size(x.shape[2 + i], k[i], stride[i], dilation[i]) for i in range(nsp))
    y = fir.custom_op("window", (x, w), (x.shape[0], w.shape[0]) + out_sp, attrs)
    if bias is not None:
        y = y + bias.reshape((1, -1) + (1,) * nsp)
    return y


def pool(x, kind, kernel, stride, padding, dilation=1):
    nsp = x.ndim - 2
    kernel, padding, dilation = _tuple(kernel, nsp), _tuple(padding, nsp), _tuple(dilation, nsp)
    stride = _tuple(stride, nsp) if stride else kernel
    x = _pad(x, padding, NEG_LARGE if kind == "max" else 0.0)
    attrs = {"mode": "pool", "kind": kind, "nsp": nsp, "kernel": kernel, "stride": stride,
             "dilation": dilation, "groups": 1, "co": x.shape[1]}
    out_sp = tuple(out_size(x.shape[2 + i], kernel[i], stride[i], dilation[i]) for i in range(nsp))
    return fir.custom_op("window", (x,), (x.shape[0], x.shape[1]) + out_sp, attrs)


# --------------------------------------------------------------------------------------------
# ATen handlers
# --------------------------------------------------------------------------------------------
@aten.handler("convolution")
def _convolution(a):
    if a.transposed:
        raise ValueError("transposed convolution is not lowered")
    return conv(a.input, a.weight, a["bias"], a.stride, a.padding, a.dilation, a.groups)


def _check_floor_mode(ceil_mode):
    if ceil_mode:
        raise ValueError("ceil_mode pooling is not lowered")


@aten.handler("max_pool2d_with_indices", "max_pool3d_with_indices")
def _max_pool(a):
    _check_floor_mode(a["ceil_mode"])
    return pool(a.self, "max", a.kernel_size, a["stride"], a["padding"], a["dilation"]), None


@aten.handler("avg_pool2d", "avg_pool3d")
def _avg_pool(a):
    _check_floor_mode(a["ceil_mode"])
    if a["divisor_override"] is not None:
        raise ValueError("avg_pool divisor_override is not lowered")
    if not a["count_include_pad"] and any(_tuple(a["padding"], a.self.ndim - 2)):
        raise ValueError("avg_pool with count_include_pad=False and padding is not lowered")
    return pool(a.self, "mean", a.kernel_size, a["stride"], a["padding"])


@aten.handler("_adaptive_avg_pool2d", "_adaptive_avg_pool3d", "adaptive_avg_pool2d")
def _adaptive_avg_pool(a):
    x = a.self
    nsp = x.ndim - 2
    size = _tuple(a.output_size, nsp)
    kernel = []
    for i in range(nsp):
        n = x.shape[2 + i]
        if n % size[i]:
            raise ValueError("adaptive pooling from %d to %d does not divide evenly" % (n, size[i]))
        kernel.append(n // size[i])
    return pool(x, "mean", tuple(kernel), tuple(kernel), 0)


def _nearest_indices(n_in, n_out, scale):
    s = (1.0 / scale) if scale else n_in / n_out
    return [min(int(math.floor(i * s)), n_in - 1) for i in range(n_out)]


@aten.handler("upsample_nearest2d.vec", "upsample_nearest2d")
def _upsample_nearest2d(a):
    x = a.get("input") if a.get("input") is not None else a["self"]
    size = a.get("output_size")
    scales = a.get("scale_factors")
    if size is None:
        if not scales:
            raise ValueError("upsample_nearest2d needs an output size or scale factors")
        size = [int(math.floor(x.shape[2 + i] * scales[i])) for i in range(2)]
    ih = _nearest_indices(x.shape[2], size[0], scales[0] if scales else None)
    iw = _nearest_indices(x.shape[3], size[1], scales[1] if scales else None)
    rows = T.index_select(x, 2, T.tensor(ih, dtype=T.int64))
    return T.index_select(rows, 3, T.tensor(iw, dtype=T.int64))
