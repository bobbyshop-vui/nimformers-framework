"""Re-fusion of decomposed torch/HF idioms into the operations the pattern passes expect.

HF modeling code writes RMSNorm, RoPE, linear layers, activations and causal masks
as compositions of primitive tensor operations. These passes run after
``fold_views`` and before ``fuse_attention``; each rewrites one idiom into the
operation that the hand-written FIR examples trace directly, so the rest of the
pipeline (attention + RoPE fusion, residual + norm fusion, pointwise fusion)
produces the same compact kernels:

* ``refuse_additive_masks``: ``scores + min * (arange(P) > position)`` (and the
  ``where(arange(P) > position, min, scores)`` form) becomes
  ``where(arange(P) <= position, scores, -big)``, the cache mask of
  ``attention_decode``.
* ``refuse_rotary``: ``q * cos + cat(-q2, q1) * sin`` (HF ``rotate_half``), where
  ``cos``/``sin`` are computed from the position input and a default
  ``inv_freq = base ** (-2i / D)`` table, becomes ``rotary(q, position)``
  (times ``attention_scaling`` when it is not 1). The angles are checked
  numerically by evaluating the cos/sin subgraph at several positions, so any way
  of computing them (matmul, outer product, einsum, cat or repeat) matches.
* ``refuse_linear``: ``x @ W.T`` with a checkpoint matrix ``W [out, in]`` becomes
  ``linear_t``; ``x @ W`` with ``W [in, out]`` and ``x`` of rank >= 2 becomes
  ``linear`` (HF GPT-2 ``Conv1D``/``addmm``). A linear layer applied to
  ``x.view(-1, in)`` whose (bias-added) result is viewed back to
  ``x.shape[:-1] + (out,)`` is applied to ``x`` directly, so the bias add
  fuses with its consumer as in the hand-written layer.
* ``refuse_norms``: ``x * rsqrt(mean(x*x, -1, keepdim) + eps)`` (or ``x**2``,
  ``x / sqrt(...)``) becomes ``rmsnorm``; a following ``* weight`` with a
  checkpoint vector becomes the norm weight.
* ``refuse_activations``: a smooth single-input pointwise subgraph that equals
  ``gelu_new``, exact ``gelu``, ``gelu_quick``, ``silu``, ``sigmoid``, ``tanh`` or
  ``softplus`` at a set of probe points (to 1e-9) becomes that activation (HF
  ``NewGELUActivation``, ``FastGELUActivation``, ``QuickGELUActivation``,
  ``x * sigmoid(x)``).
"""

import math
from dataclasses import replace

from .graph import Node, Value, ELEMENTWISE, producers, consumers, layer_scope, eliminate_dead_code
from .patterns import const_value
from .reference import apply_op, run_node
from .views import VIEW_NODE_OPS


def _fresh(graph, base):
    key, i = base, 0
    while key in graph.values:
        i += 1
        key = "%s_%d" % (base, i)
    return key


def _add_value(graph, key, shape, constant=None, dtype="f32"):
    graph.values[key] = Value(math.prod(shape), shape=tuple(shape), dtype=dtype,
                              constant=tuple(constant) if constant is not None else None)
    return key


def _position(graph):
    return next((k for k, v in graph.values.items() if v.role == "position" and v.size == 1), None)


def _replace(graph, old, new_nodes):
    """Replace node ``old`` by ``new_nodes`` (the last one must produce old.output)."""
    out = []
    for n in graph.nodes:
        if n is old: out.extend(new_nodes)
        else: out.append(n)
    graph.nodes = out


# ---- masks ---------------------------------------------------------------------------

def _peel_views(graph, prods, k):
    for _ in range(16):
        n = prods.get(k)
        if n is None or n.op not in ("reshape", "expand", "slice"): return k
        k = n.inputs[0]
    return k


def _range_compare(graph, prods, k):
    """(arange, position, negated op) when ``k`` is ``arange(P) > position`` or ``position < arange(P)``."""
    n = prods.get(k)
    if n is None or n.op not in ("gt", "lt"): return None
    a, b = n.inputs
    if n.op == "lt": a, b = b, a
    ar = graph.values[a]
    if ar.constant is None or ar.constant != tuple(float(i) for i in range(ar.size)) or len(ar.shape) != 1:
        return None
    if graph.values[b].size != 1: return None
    return n, a, b


def refuse_additive_masks(graph):
    """Additive or masked_fill causal masks over the cache become ``where(arange <= position, s, -big)``."""
    prods, cons = producers(graph), consumers(graph)
    rewrites = []
    for n in graph.nodes:
        if n.op == "add":
            for s, m in (n.inputs, n.inputs[::-1]):
                base = _peel_views(graph, prods, m)
                mul = prods.get(base)
                cmp_ = None
                if mul is not None and mul.op == "mul":
                    for c, g in (mul.inputs, mul.inputs[::-1]):
                        cv = const_value(graph, c)
                        if cv is not None and cv <= -1e30:
                            cmp_ = _range_compare(graph, prods, _peel_views(graph, prods, g))
                            if cmp_: break
                elif mul is not None and mul.op == "where":
                    c, a, b = mul.inputs
                    ca, cb = const_value(graph, a), const_value(graph, b)
                    if ca is not None and ca <= -1e30 and cb == 0:
                        cmp_ = _range_compare(graph, prods, _peel_views(graph, prods, c))
                if cmp_:
                    rewrites.append((n, s, cmp_))
                    break
        elif n.op == "where":
            c, a, b = n.inputs
            ca = const_value(graph, a)
            if ca is not None and ca <= -1e30:
                cmp_ = _range_compare(graph, prods, _peel_views(graph, prods, c))
                if cmp_: rewrites.append((n, b, cmp_))
    for n, scores, (compare, ar, pos) in rewrites:
        mask = _add_value(graph, _fresh(graph, n.output + "_mask"), graph.values[compare.output].shape, dtype="bool")
        neg = _add_value(graph, _fresh(graph, n.output + "_neg"), (), (-3.4028234663852886e38,))
        le = Node("le", (ar, pos), mask, {}, n.scope)
        sel = Node("where", (mask, scores, neg), n.output, {}, n.scope)
        _replace(graph, n, [le, sel])
    if rewrites: eliminate_dead_code(graph)


# ---- rotary -----------------------------------------------------------------------------

def _ancestors(graph, prods, k, limit=400):
    """Nodes computing ``k`` in program order, or None unless every leaf is a constant or the position."""
    order, seen, stack = [], set(), [k]
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x)
        v = graph.values[x]
        if v.constant is not None: continue
        if v.role:
            if v.role != "position" or v.size != 1: return None
            continue
        n = prods.get(x)
        if n is None or n.effectful or n.extra or v.weight or v.state: return None
        order.append(n)
        if len(order) > limit: return None
        stack.extend(n.inputs)
    index = {id(n): i for i, n in enumerate(graph.nodes)}
    return sorted(order, key=lambda n: index[id(n)])


def _evaluate_at(graph, nodes, position_key, p, wanted):
    needed = {k for n in nodes for k in n.inputs} | set(wanted)
    values = {k: list(graph.values[k].constant) for k in needed if graph.values[k].constant is not None}
    values[position_key] = [float(p)]
    for n in nodes:
        for name, data in run_node(n, graph, values).items():
            values[name] = data
    return [values[k] for k in wanted]


def _row_vector(flat, shape, width):
    """The last-dim vector of a tensor constant across all leading dims, else None."""
    if not shape or shape[-1] != width: return None
    rows = len(flat) // width
    first = flat[:width]
    for r in range(1, rows):
        if any(abs(a - b) > 1e-12 * max(1.0, abs(b)) for a, b in zip(flat[r*width:(r+1)*width], first)): return None
    return first


def _rotate_half_source(graph, prods, k, width):
    """Q when ``k`` is ``cat((-Q[..., half:], Q[..., :half]), -1)``."""
    n = prods.get(k)
    if n is None or n.op != "cat" or len(n.inputs) != 2: return None
    shape = graph.values[k].shape
    if n.attrs["dim"] != len(shape) - 1 or shape[-1] != width or width % 2: return None
    half = width // 2
    neg = prods.get(n.inputs[0])
    if neg is None or neg.op != "neg": return None
    second, first = prods.get(neg.inputs[0]), prods.get(n.inputs[1])
    if second is None or first is None or second.op != "slice" or first.op != "slice": return None
    if second.inputs[0] != first.inputs[0]: return None
    q = first.inputs[0]
    qshape = graph.values[q].shape
    if qshape != shape: return None
    def full(spec, last):
        return all(s == (0, d, 1) for s, d in zip(spec[:-1], qshape[:-1])) and spec[-1] == last
    if not full(first.attrs["index"], (0, half, 1)) or not full(second.attrs["index"], (half, width, 1)): return None
    return q


def _fit_rope(c0, s0, c1, s1, half):
    scale = c0[0]
    if scale == 0 or any(abs(c - scale) > 1e-9 * abs(scale) for c in c0) or any(abs(s) > 1e-9 * abs(scale) for s in s0):
        return None
    freqs = [math.atan2(s / scale, c / scale) for c, s in zip(c1, s1)]
    if any(abs(freqs[i] - freqs[i + half]) > 1e-9 for i in range(half)): return None
    if freqs[0] <= 0 or abs(freqs[0] - 1.0) > 1e-9: return None
    if half == 1: return scale, 10000.0, freqs
    if freqs[1] <= 0: return None
    base = freqs[1] ** (-half)
    for i in range(half):
        want = base ** (-i / half)
        if abs(freqs[i] - want) > 1e-7 * want: return None
    return scale, base, freqs


def refuse_rotary(graph):
    """HF ``q * cos + rotate_half(q) * sin`` with default RoPE frequencies becomes ``rotary``."""
    position = _position(graph)
    if position is None: return
    prods, cons = producers(graph), consumers(graph)
    rewrites, cache = [], {}
    for n in graph.nodes:
        if n.op != "add": continue
        a, b = (prods.get(k) for k in n.inputs)
        if a is None or b is None or a.op != "mul" or b.op != "mul": continue
        width = graph.values[n.output].shape[-1] if graph.values[n.output].shape else 0
        match = None
        for qmul, rmul in ((a, b), (b, a)):
            for r, s in (rmul.inputs, rmul.inputs[::-1]):
                q = _rotate_half_source(graph, prods, r, width)
                if q is None: continue
                if q in qmul.inputs:
                    c = qmul.inputs[1] if qmul.inputs[0] == q else qmul.inputs[0]
                    match = (q, c, s)
                    break
            if match: break
        if match is None: continue
        q, c, s = match
        if graph.values[n.output].shape != graph.values[q].shape: continue
        key = (c, s)
        if key not in cache:
            cache[key] = None
            nodes_c, nodes_s = _ancestors(graph, prods, c), _ancestors(graph, prods, s)
            if nodes_c is not None and nodes_s is not None and (nodes_c or nodes_s):
                nodes = sorted({id(x): x for x in nodes_c + nodes_s}.values(), key=graph.nodes.index)
                try:
                    samples = {p: _evaluate_at(graph, nodes, position, p, (c, s)) for p in (0, 1, 7)}
                    rows = {p: [_row_vector(v, graph.values[k].shape, width) for v, k in zip(vals, (c, s))]
                            for p, vals in samples.items()}
                    if all(r is not None for vals in rows.values() for r in vals):
                        fit = _fit_rope(rows[0][0], rows[0][1], rows[1][0], rows[1][1], width // 2)
                        if fit is not None:
                            scale, base, freqs = fit
                            c7, s7 = rows[7]
                            ok = all(abs(c7[d] - scale * math.cos(7 * f)) <= 1e-7 * abs(scale) and
                                     abs(s7[d] - scale * math.sin(7 * f)) <= 1e-7 * abs(scale)
                                     for d, f in enumerate(freqs))
                            if ok: cache[key] = (scale, base)
                except (ValueError, ZeroDivisionError, OverflowError, KeyError, IndexError):
                    cache[key] = None
        if cache[key] is None: continue
        rewrites.append((n, q, cache[key]))
    for n, q, (scale, base) in rewrites:
        pos = _add_value(graph, _fresh(graph, n.output + "_position"), ())
        nodes = [Node("reshape", (position,), pos, {}, n.scope)]
        if abs(scale - 1.0) <= 1e-12:
            nodes.append(Node("rotary", (q, pos), n.output, {"base": float(base)}, n.scope))
        else:
            rot = _add_value(graph, _fresh(graph, n.output + "_rotary"), graph.values[n.output].shape)
            factor = _add_value(graph, _fresh(graph, n.output + "_scale"), (), (float(scale),))
            nodes.append(Node("rotary", (q, pos), rot, {"base": float(base)}, n.scope))
            nodes.append(Node("mul", (rot, factor), n.output, {}, n.scope))
        _replace(graph, n, nodes)
    if rewrites: eliminate_dead_code(graph)


# ---- linear --------------------------------------------------------------------------------

def refuse_linear(graph):
    """``x @ W.T`` (W ``[out,in]`` checkpoint) -> ``linear_t``; ``x @ W`` (W ``[in,out]``, rank-2+ x) -> ``linear``."""
    prods = producers(graph)
    out, changed = [], False
    for n in graph.nodes:
        if n.op == "matmul":
            x, w = n.inputs
            xv, wv = graph.values[x], graph.values[w]
            p = prods.get(w)
            if (p is not None and p.op == "permute" and p.attrs["dims"] == (1, 0)
                    and graph.values[p.inputs[0]].weight is not None and len(graph.values[p.inputs[0]].shape) == 2
                    and xv.shape and xv.shape[-1] == wv.shape[0]):
                n = Node("linear_t", (x, p.inputs[0]), n.output, {}, n.scope)
                changed = True
            elif (wv.weight is not None and len(wv.shape) == 2 and len(xv.shape) >= 2
                    and xv.shape[-1] == wv.shape[0] and not xv.weight):
                n = Node("linear", (x, w), n.output, {}, n.scope)
                changed = True
        out.append(n)
    graph.nodes = out
    if changed: eliminate_dead_code(graph)
    _unflatten_linear(graph)


def _unflatten_linear(graph):
    """``reshape(linear(reshape(x)) [+ bias]) -> x.shape[:-1] + (out,)`` becomes ``linear(x) [+ bias]``.

    When the linear input is not such a view (attention already consumed it), the
    output view moves between the linear layer and the bias add, so the add can
    fuse with the residual add that follows.
    """
    prods, cons = producers(graph), consumers(graph)
    rewrite, inserted, drop = {}, {}, set()
    for r in graph.nodes:
        if r.op != "reshape" or id(r) in drop: continue
        a = r.inputs[0]
        p = prods.get(a)
        add, lin = None, p
        if p is not None and p.op == "add":
            for l, b in (p.inputs, p.inputs[::-1]):
                ln = prods.get(l)
                bv = graph.values[b]
                if (ln is not None and ln.op in ("linear", "linear_t") and len(bv.shape) == 1
                        and bv.shape[0] == graph.values[l].shape[-1] and _single(cons, graph, l, p)):
                    add, lin, bias = p, ln, b
                    break
            else:
                continue
        if lin is None or lin.op not in ("linear", "linear_t") or not _single(cons, graph, a, r): continue
        lout, out = lin.output, r.output
        ls, os_ = graph.values[lout].shape, graph.values[out].shape
        if not ls or not os_ or os_[-1] != ls[-1]: continue
        r0 = prods.get(lin.inputs[0])
        xs = graph.values[r0.inputs[0]].shape if r0 is not None and r0.op == "reshape" else None
        if xs and xs[-1] == graph.values[lin.inputs[0]].shape[-1] and os_ == xs[:-1] + (ls[-1],):
            # The linear layer applies to the unflattened input directly.
            graph.values[lout] = replace(graph.values[lout], shape=os_)
            rewrite[id(lin)] = replace(lin, inputs=(r0.inputs[0],) + lin.inputs[1:])
            view = lout
        elif add is None:
            continue
        else:
            # Keep the linear layer; move the view between it and the bias add.
            view = lout + "_view"
            graph.values[view] = replace(graph.values[out])
            inserted[id(lin)] = Node("reshape", (lout,), view, {}, r.scope)
        if add is not None:
            rewrite[id(add)] = replace(add, inputs=tuple(view if k == lout else bias for k in add.inputs), output=out)
        else:
            rewrite[id(lin)] = replace(rewrite[id(lin)], output=out)
        drop.add(id(r))
    if not drop: return
    nodes = []
    for n in graph.nodes:
        if id(n) in drop: continue
        nodes.append(rewrite.get(id(n), n))
        if id(n) in inserted: nodes.append(inserted[id(n)])
    graph.nodes = nodes
    eliminate_dead_code(graph)


# ---- RMSNorm ----------------------------------------------------------------------------------

def _single(cons, graph, k, user):
    return [id(u) for u in cons.get(k, [])] == [id(user)] and not graph.is_output(k)


def _match_rms(graph, prods, cons, n):
    """x * rsqrt(mean(x*x, -1, keepdim=True) + eps) or x / sqrt(...): returns (x, eps, removed)."""
    if n.op not in ("mul", "div"): return None
    pairs = (n.inputs, n.inputs[::-1]) if n.op == "mul" else (n.inputs,)
    for x, r in pairs:
        rn = prods.get(r)
        if rn is None or not _single(cons, graph, r, n): continue
        if (n.op == "mul" and rn.op != "rsqrt") or (n.op == "div" and rn.op != "sqrt"): continue
        an = prods.get(rn.inputs[0])
        if an is None or an.op != "add" or not _single(cons, graph, rn.inputs[0], rn): continue
        for m, e in (an.inputs, an.inputs[::-1]):
            eps = const_value(graph, e)
            mn = prods.get(m)
            if eps is None or eps <= 0 or mn is None or mn.op != "mean" or not _single(cons, graph, m, an): continue
            xshape = graph.values[x].shape
            if not xshape or mn.attrs["dims"] != (len(xshape) - 1,) or not mn.attrs["keepdim"]: continue
            sq = prods.get(mn.inputs[0])
            if sq is None or not _single(cons, graph, mn.inputs[0], mn): continue
            if sq.op == "mul" and sq.inputs == (x, x): pass
            elif sq.op == "pow" and sq.inputs[0] == x and const_value(graph, sq.inputs[1]) == 2: pass
            else: continue
            if graph.values[n.output].shape != xshape: continue
            return x, eps, (sq, mn, an, rn)
    return None


def refuse_norms(graph):
    """Decomposed RMSNorm (with or without f32 round trips) -> ``rmsnorm`` [+ weight]."""
    prods, cons = producers(graph), consumers(graph)
    changed = False
    out = []
    for n in graph.nodes:
        m = _match_rms(graph, prods, cons, n)
        if m is not None:
            x, eps, _ = m
            n = Node("rmsnorm", (x,), n.output, {"width": graph.values[x].shape[-1], "eps": eps}, n.scope)
            changed = True
        out.append(n)
    graph.nodes = out
    if not changed: return
    eliminate_dead_code(graph)
    prods, cons = producers(graph), consumers(graph)
    out, dead = [], set()
    for n in graph.nodes:
        if n.op == "mul":
            for r, w in (n.inputs, n.inputs[::-1]):
                rn = prods.get(r)
                wv = graph.values[w]
                if (rn is not None and rn.op == "rmsnorm" and len(rn.inputs) == 1 and _single(cons, graph, r, n)
                        and wv.weight is not None and wv.shape == (rn.attrs["width"],)
                        and graph.values[n.output].shape == graph.values[r].shape
                        and layer_scope(rn.scope) == layer_scope(n.scope)):
                    dead.add(id(rn))
                    n = Node("rmsnorm", (rn.inputs[0], w), n.output, dict(rn.attrs), rn.scope)
                    break
        out.append(n)
    graph.nodes = [n for n in out if id(n) not in dead]
    eliminate_dead_code(graph)


# ---- activations --------------------------------------------------------------------------------

SMOOTH = {"add", "sub", "mul", "div", "pow", "neg", "exp", "log", "sqrt", "rsqrt", "tanh", "sigmoid",
          "silu", "softplus", "gelu_new", "gelu", "gelu_quick", "sin", "cos"}
ACTIVATIONS = ("gelu_new", "gelu", "gelu_quick", "silu", "sigmoid", "tanh", "softplus")
PROBES = (-6.0, -3.0, -1.7, -1.0, -0.4, -0.05, 0.0, 0.03, 0.5, 1.2, 2.2, 4.5, 7.0)


def _region(graph, prods, cons, root):
    """Single-use smooth pointwise producers of ``root``; returns (nodes, leaves)."""
    region = {id(root): root}
    changed = True
    while changed:
        changed = False
        for node in list(region.values()):
            for k in node.inputs:
                p = prods.get(k)
                if p is None or id(p) in region or p.op not in SMOOTH or p.effectful or p.extra: continue
                if graph.is_output(k) or layer_scope(p.scope) != layer_scope(root.scope): continue
                users = cons.get(k, [])
                if len(users) != 1 or id(users[0]) not in region: continue
                region[id(p)] = p
                changed = True
    outputs = {n.output for n in region.values()}
    leaves = {k for n in region.values() for k in n.inputs if k not in outputs}
    return region, leaves


def _activation_of(graph, region, root, x):
    nodes = {n.output: n for n in region.values()}
    def value(k, probe, memo):
        if k == x: return probe
        if k in memo: return memo[k]
        cv = const_value(graph, k)
        if cv is not None and k not in nodes: return cv
        n = nodes[k]
        r = apply_op(n.op, [value(i, probe, memo) for i in n.inputs])
        if isinstance(r, complex) or r != r: raise ValueError("non-real value")
        memo[k] = r
        return r
    try:
        got = [value(root.output, p, {}) for p in PROBES]
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    for act in ACTIVATIONS:
        want = [apply_op(act, [p]) for p in PROBES]
        if all(abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in zip(got, want)):
            return act
    return None


def refuse_activations(graph):
    """Decomposed activation formulas (NewGELU, FastGELU, QuickGELU, x*sigmoid(x), ...) -> one op."""
    prods, cons = producers(graph), consumers(graph)
    absorbed, replaced = set(), {}
    for root in reversed(graph.nodes):
        if id(root) in absorbed or root.op not in SMOOTH or root.effectful: continue
        region, leaves = _region(graph, prods, cons, root)
        if len(region) < 2: continue
        tensors = [k for k in leaves if const_value(graph, k) is None]
        if len(tensors) != 1: continue
        if any(graph.values[k].constant is not None and const_value(graph, k) is None for k in leaves): continue
        x = tensors[0]
        if graph.values[x].shape != graph.values[root.output].shape: continue
        act = _activation_of(graph, region, root, x)
        if act is None: continue
        absorbed.update(region)
        replaced[id(root)] = Node(act, (x,), root.output, {}, root.scope)
    if not replaced: return
    graph.nodes = [replaced.get(id(n), n) for n in graph.nodes]
    eliminate_dead_code(graph)
