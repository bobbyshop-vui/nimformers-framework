"""Micro-kernels for ATen ops that the FIR tracer has no kernel for.

Each operation here has three parts, all registered with ``formers.fir``:

* a Python **reference** (flat lists; used by ``fir.evaluate`` and the tests),
* a **micro-kernel generator** built on ``formers.fir.codegen.Kernel`` (one ``kernel`` block in
  ``model.fir``), and
* a front-end function that records the node (``trunc``, ``prod``, ``cumprod``, ``scatter``,
  ``sort_values``, ``sort_indices``).

Operations that are only a composition of existing FIR ops (``floor``, ``fmod``, ``erf`` ...) are
written in Python on top of ``trunc`` and live in ``compose.py``.
"""
from __future__ import annotations

import math

from formers import fir
from formers.fir import codegen as C
from formers.fir import custom
from formers.fir import views as V

from . import aten


# --------------------------------------------------------------------------------------------
# small shape helpers (references)
# --------------------------------------------------------------------------------------------
def _strides(shape):
    out, step = [], 1
    for d in reversed(shape):
        out.append(step)
        step *= d
    return list(reversed(out))


def _rows(shape, dim):
    """(outer index list, stride of dim, size of dim) to walk 1-D lines of ``dim``."""
    st = _strides(shape)
    n = shape[dim]
    outer = [()]
    for d, s in enumerate(shape):
        if d != dim:
            outer = [o + (i,) for o in outer for i in range(s)]
    base = []
    for o in outer:
        it = iter(o)
        base.append(sum(next(it) * st[d] for d in range(len(shape)) if d != dim))
    return base, st[dim], n


# --------------------------------------------------------------------------------------------
# trunc: the one rounding primitive (float -> integer-valued float, toward zero)
# --------------------------------------------------------------------------------------------
def _trunc_emit(k, args):
    """Scalar tree op: r = trunc(r). Uses the VM float->uint conversion on |x|, restores the sign."""
    r = args[0]
    s, t, sign, zero, neg, big, small = k.r(), k.r(), k.const(1), k.const(0), k.r(), k.const(8388608.0), k.r()
    k.copy(s, r)
    k.emit("cmplt", neg, s, zero)

    def flip():
        k.emit("neg", s, s)
        k.const(-1, sign)
    k.cond(neg, flip)
    u = k.u()
    k.emit("rtou", u, s)
    k.emit("utor", t, u)
    k.free_u(u)
    k.emit("mul", t, t, sign)
    k.emit("cmplt", small, s, big)          # |x| < 2**23; larger floats are already integers
    k.cond(small, lambda: k.copy(r, t))
    k.free_r(s, t, sign, zero, neg, big, small)
    return r


custom.register_tree_op("ext_trunc", _trunc_emit)


def _lower_pointwise(kind):
    def lower(em, node, ctx, kb):
        g = em.graph
        shape = g.values[node.output].shape
        operands = [em.operand(k, ctx) for k in node.inputs]
        out = C.Mem(ctx.port_out(em.output_buffer(node.output)), V.identity(shape))
        tree = ("ext_" + kind, tuple(("input", i) for i in range(len(node.inputs))))
        return kind, C.gen_pointwise(kb, tree, operands, [out], shape)
    return lower


fir.register_op("trunc", reference=lambda ins, shapes, out, attrs: [float(math.trunc(v)) if abs(v) < 2 ** 52 else v
                                                                    for v in ins[0]],
                lower=_lower_pointwise("trunc"), replace=True)


def trunc(x):
    return fir.custom_op("trunc", (x,), x.shape)


# --------------------------------------------------------------------------------------------
# prod / cumprod along one dim
# --------------------------------------------------------------------------------------------
def _gen_prod(k, x, out, shape, dim):
    inner_ids = [dim]
    kept = [d for i, d in enumerate(shape) if i != dim]
    red = [shape[dim]]
    xacc = C.access(x, shape, inner_ids)
    oview = V.reshape(out.view, tuple(kept)) if out.view.shape != tuple(kept) else out.view
    if oview is None:
        raise C.LoweringError("prod output view is not strided")
    oacc = C.access(C.Mem(out.port, oview), tuple(kept))
    sizes = C.plan(kept, [xacc, oacc], "outer")
    inner = C.plan(red, [xacc], "inner")
    threads, regs, rest = k.lanes(sizes, [xacc, oacc])

    def row(regs):
        acc = k.const(1)

        def body(r):
            v = k.load(xacc, r)
            k.emit("mul", acc, acc, v)
            k.free_r(v)
        k.nest(inner, list(range(len(inner))), [xacc], regs, body, "inner")
        k.store(oacc, regs, acc)
        k.free_r(acc)
    k.nest(sizes, rest, [xacc, oacc], regs, row)
    return threads


def _lower_prod(em, node, ctx, kb):
    g = em.graph
    x = em.operand(node.inputs[0], ctx, "mem")
    out = C.Mem(ctx.port_out(em.output_buffer(node.output)), V.identity(g.values[node.output].shape))
    return "prod", _gen_prod(kb, x, out, g.values[node.inputs[0]].shape, node.attrs["dim"])


def _ref_prod(ins, shapes, out, attrs):
    shape, dim = shapes[0], attrs["dim"]
    base, step, n = _rows(shape, dim)
    return [math.prod(ins[0][b + i * step] for i in range(n)) for b in base]


fir.register_op("prod", reference=_ref_prod, lower=_lower_prod, replace=True)


def _gen_cumprod(k, x, out, shape, dim):
    accs = [C.access(x, shape, [dim]), C.access(out, shape, [dim])]
    outer = [d for i, d in enumerate(shape) if i != dim]
    sizes = C.plan(outer, accs, "outer")
    inner = C.plan([shape[dim]], accs, "inner")
    threads, regs, rest = k.lanes(sizes, accs)

    def row(regs):
        acc = k.const(1)

        def body(r):
            v = k.load(accs[0], r)
            k.emit("mul", acc, acc, v)
            k.store(accs[1], r, acc)
            k.free_r(v)
        k.nest(inner, list(range(len(inner))), accs, regs, body, "inner")
        k.free_r(acc)
    k.nest(sizes, rest, accs, regs, row)
    return threads


def _lower_cumprod(em, node, ctx, kb):
    g = em.graph
    shape = g.values[node.output].shape
    x = em.operand(node.inputs[0], ctx, "mem")
    out = C.Mem(ctx.port_out(em.output_buffer(node.output)), V.identity(shape))
    return "cumprod", _gen_cumprod(kb, x, out, shape, node.attrs["dim"])


def _ref_cumprod(ins, shapes, out, attrs):
    shape, dim = shapes[0], attrs["dim"]
    result = list(ins[0])
    base, step, n = _rows(shape, dim)
    for b in base:
        acc = 1.0
        for i in range(n):
            acc *= ins[0][b + i * step]
            result[b + i * step] = acc
    return result


fir.register_op("cumprod", reference=_ref_cumprod, lower=_lower_cumprod, replace=True)


def prod(x, dim=None, keepdim=False):
    if dim is None:
        x = x.reshape(-1)
        dim = 0
    dim = dim % x.ndim
    kept = tuple(d for i, d in enumerate(x.shape) if i != dim)
    result = fir.custom_op("prod", (x,), kept, {"dim": dim})
    return result.unsqueeze(dim) if keepdim else result


def cumprod(x, dim):
    return fir.custom_op("cumprod", (x,), x.shape, {"dim": dim % x.ndim})


# --------------------------------------------------------------------------------------------
# scatter / scatter_add (also index_put): out = self; out[..., index[i], ...] (+)= src[i]
# --------------------------------------------------------------------------------------------
def _gen_scatter(k, base, index, src, out, shape, ishape, dim, add):
    # 1. out = base (single thread, serial copy)
    ca, co = C.access(base, shape), C.access(C.Mem(out.port, out.view), shape)
    sizes = C.plan(shape, [ca, co], "outer")

    def copy(r):
        v = k.load(ca, r)
        k.store(co, r, v)
        k.free_r(v)
    k.nest(sizes, list(range(len(sizes))), [ca, co], {}, copy)
    # 2. scatter, one index element at a time
    iacc = C.access(index, ishape)
    sacc = C.access(src, ishape) if not isinstance(src, C.Const) else None
    ostrides = [s for _, s, _ in out.view.dims]
    oacc = C.Acc("mem", out.port, out.view.offset, [(0 if d == dim else ostrides[d], 1) for d in range(len(ishape))])
    live = [a for a in (iacc, sacc, oacc) if a is not None]
    isizes = C.plan(ishape, live, "outer")
    stride_dim = ostrides[dim]

    def body(r):
        i = k.load(iacc, r)
        dyn = k.u()
        k.emit("rtou", dyn, i)
        k.emit("umulk", dyn, dyn, stride_dim)
        target = k.u()
        k.emit("uadd", target, r.get(id(oacc)) if id(oacc) in r else k.zero_u(), dyn)
        v = k.const(src.value) if sacc is None else k.load(sacc, r)
        if add:
            cur = k.r()
            k.emit("uloadf32", cur, target, C.address(out.port, out.view.offset))
            k.emit("add", v, v, cur)
            k.free_r(cur)
        k.emit("ustoref32", v, target, C.address(out.port, out.view.offset))
        k.free_r(i, v)
        k.free_u(dyn, target)
    k.nest(isizes, list(range(len(isizes))), live, {}, body)
    return 1


def _lower_scatter(add):
    def lower(em, node, ctx, kb):
        g = em.graph
        shape = g.values[node.output].shape
        base = em.operand(node.inputs[0], ctx, "mem")
        index = em.operand(node.inputs[1], ctx, "mem")
        src = em.operand(node.inputs[2], ctx) if len(node.inputs) > 2 else C.Const(node.attrs["value"])
        out = C.Mem(ctx.port_out(em.output_buffer(node.output)), V.identity(shape))
        return ("scatter_add" if add else "scatter"), _gen_scatter(
            kb, base, index, src, out, shape, g.values[node.inputs[1]].shape, node.attrs["dim"], add)
    return lower


def _ref_scatter(add):
    def ref(ins, shapes, out, attrs):
        shape, ishape, dim = shapes[0], shapes[1], attrs["dim"]
        result = list(ins[0])
        st, ist = _strides(shape), _strides(ishape)
        sshape = shapes[2] if len(shapes) > 2 else None
        sst = _strides(sshape) if sshape else None
        for flat in range(math.prod(ishape)):
            coord, rest = [], flat
            for s in ist:
                coord.append(rest // s)
                rest %= s
            target = sum((int(ins[1][flat]) if d == dim else c) * st[d] for d, c in enumerate(coord))
            value = ins[2][sum(c * s for c, s in zip(coord, sst))] if sshape else attrs["value"]
            result[target] = result[target] + value if add else value
        return result
    return ref


fir.register_op("scatter", reference=_ref_scatter(False), lower=_lower_scatter(False), replace=True)
fir.register_op("scatter_add", reference=_ref_scatter(True), lower=_lower_scatter(True), replace=True)


def scatter(x, dim, index, src, add=False):
    dim = dim % x.ndim
    name = "scatter_add" if add else "scatter"
    if isinstance(src, (int, float)):
        return fir.custom_op(name, (x, index), x.shape, {"dim": dim, "value": float(src)})
    if tuple(src.shape) != tuple(index.shape):          # torch reads only the leading block of src
        ix = tuple(slice(0, n) for n in index.shape)
        src = src[ix]
    return fir.custom_op(name, (x, index, src), x.shape, {"dim": dim})


# --------------------------------------------------------------------------------------------
# sort / argsort / topk by counting ranks (O(n^2) per line; n is the size of the sorted dim)
# --------------------------------------------------------------------------------------------
def _gen_sort(k, x, out, shape, dim, descending, indices):
    xi = C.access(x, shape, [dim])
    xj = C.access(x, shape, [dim])
    outer = [d for i, d in enumerate(shape) if i != dim]
    o_strides = [s for _, s, _ in out.view.dims]
    oacc = C.Acc("mem", out.port, out.view.offset, [(o_strides[d], 1) for d in range(len(shape)) if d != dim])
    sizes = C.plan(outer, [xi, xj, oacc], "outer")
    inner_i = C.plan([shape[dim]], [xi], "inner")
    inner_j = C.plan([shape[dim]], [xj], "inner")
    threads, regs, rest = k.lanes(sizes, [xi, xj, oacc])
    stride_dim = o_strides[dim]

    def row(regs):
        icount = k.const(-1)

        def per_i(ri):
            k.emit("addconst", icount, icount, 1)
            vi = k.load(xi, ri)
            rank, jcount = k.const(0), k.const(-1)

            def per_j(rj):
                k.emit("addconst", jcount, jcount, 1)
                vj = k.load(xj, rj)
                lt, gt, before, tmp = k.r(), k.r(), k.r(), k.r()
                a, b = (vi, vj) if descending else (vj, vi)
                k.emit("cmplt", lt, a, b)             # j sorts strictly before i
                k.emit("cmplt", gt, b, a)             # j sorts strictly after i
                k.emit("cmplt", before, jcount, icount)
                k.emit("add", tmp, lt, gt)
                k.emit("neg", tmp, tmp)
                k.emit("addconst", tmp, tmp, 1)       # tmp = 1 when equal
                k.emit("mul", tmp, tmp, before)       # equal and earlier
                k.emit("add", rank, rank, lt)
                k.emit("add", rank, rank, tmp)
                k.free_r(vj, lt, gt, before, tmp)
            k.nest(inner_j, list(range(len(inner_j))), [xj], ri, per_j, "inner")
            dyn = k.u()
            k.emit("rtou", dyn, rank)
            k.emit("umulk", dyn, dyn, stride_dim)
            base = regs.get(id(oacc)) if id(oacc) in regs else k.zero_u()
            target = k.u()
            k.emit("uadd", target, base, dyn)
            k.emit("ustoref32", icount if indices else vi, target, C.address(out.port, out.view.offset))
            k.free_u(dyn, target)
            k.free_r(vi, rank, jcount)
        k.nest(inner_i, list(range(len(inner_i))), [xi], regs, per_i, "inner")
        k.free_r(icount)
    k.nest(sizes, rest, [xi, xj, oacc], regs, row)
    return threads


def _lower_sort(indices):
    def lower(em, node, ctx, kb):
        g = em.graph
        shape = g.values[node.output].shape
        x = em.operand(node.inputs[0], ctx, "mem")
        out = C.Mem(ctx.port_out(em.output_buffer(node.output)), V.identity(shape))
        return ("sort_indices" if indices else "sort_values"), _gen_sort(
            kb, x, out, shape, node.attrs["dim"], node.attrs["descending"], indices)
    return lower


def _ref_sort(indices):
    def ref(ins, shapes, out, attrs):
        shape, dim = shapes[0], attrs["dim"]
        result = list(ins[0])
        base, step, n = _rows(shape, dim)
        for b in base:
            line = [(ins[0][b + i * step], i) for i in range(n)]
            line.sort(key=lambda p: (-p[0] if attrs["descending"] else p[0], p[1]))
            for r, (v, i) in enumerate(line):
                result[b + r * step] = float(i) if indices else v
        return result
    return ref


fir.register_op("sort_values", reference=_ref_sort(False), lower=_lower_sort(False), replace=True)
fir.register_op("sort_indices", reference=_ref_sort(True), lower=_lower_sort(True), replace=True)


def sort(x, dim=-1, descending=False):
    dim = dim % x.ndim
    attrs = {"dim": dim, "descending": bool(descending)}
    return (fir.custom_op("sort_values", (x,), x.shape, attrs),
            fir.custom_op("sort_indices", (x,), x.shape, attrs, dtype="i64"))


def topk(x, k, dim=-1, largest=True):
    values, indices = sort(x, dim, descending=largest)
    dim = dim % x.ndim
    ix = tuple(slice(0, k) if d == dim else slice(None) for d in range(x.ndim))
    return values[ix], indices[ix]


# --------------------------------------------------------------------------------------------
# ATen handlers for the operations above
# --------------------------------------------------------------------------------------------
M = aten.MICRO
M["prod"] = lambda a: prod(a.self, a["dim"] if "dim" in a else None, a["keepdim"] if "keepdim" in a else False)
M["cumprod"] = lambda a: cumprod(a.self, a.dim)
M["scatter"] = lambda a: scatter(a.self, a.dim, a.index, a.get("src") if a.get("src") is not None else a.get("value"))
M["scatter_add"] = lambda a: scatter(a.self, a.dim, a.index, a.src, add=True)
M["sort"] = lambda a: sort(a.self, a["dim"], a["descending"])
M["topk"] = lambda a: topk(a.self, a.k, a["dim"], a["largest"])
M["argsort"] = lambda a: sort(a.self, a["dim"], a["descending"])[1]
M["trunc"] = lambda a: trunc(a.self)
