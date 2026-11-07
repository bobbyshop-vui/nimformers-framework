"""Graph passes that recognise higher-level patterns in traced graphs.

Every pass rewrites matched subgraphs into one fused operation whose reference
semantics live in ``reference.py`` and whose loop-based micro-kernel lives in
``codegen.py``. Unmatched subgraphs are left unchanged and keep the generic
lowering. All passes are deterministic: they visit nodes in program order and
produce stable names.

Recognised patterns:

* ``fuse_attention``: causal decode attention over K/V caches. Matches
  ``softmax(mask(scale * q @ k^T)) @ v`` where K/V are indexed state writes at
  ``position`` and the mask is ``arange(n_positions) <= position``. Multi-head,
  grouped-query (slice/cat or expand/reshape) and shared-KV layouts are accepted.
* ``fuse_rotary_attention``: RoPE producers of q and of the cached k are folded
  into the attention kernel.
* ``push_slice_into_conv``: a convolution whose output is sliced along time only
  computes the selected time steps.
* ``fuse_conv_state``: streaming depthwise causal convolution with a history
  state (``cat(state, x)``, ``state.update(window[..., 1:])``, ``conv1d``).
* ``fuse_pointwise``: elementwise DAGs (activations, SwiGLU gates, bias adds,
  residual adds) become one expression tree.
* ``fuse_state_update_reduction``: a full state update followed by a reduction
  of the updated state, the selective-scan (SSM) decode step.
* ``fuse_reduction_prologue``: elementwise producers of a reduction.
* ``fuse_norms``: elementwise producers of LayerNorm/RMSNorm (residual + norm),
  keeping the residual sum as a second output when it is used later.
"""

from dataclasses import replace
import math

from .graph import (Node, ELEMENTWISE, producers, consumers, layer_scope, eliminate_dead_code)
from . import views as V
from .scalar import unravel, offset as ravel


TRANSCENDENTAL = {"exp", "log", "sigmoid", "silu", "softplus", "gelu_new", "gelu", "gelu_quick",
                  "tanh", "sin", "cos", "rsqrt", "sqrt", "div", "pow"}
MAX_TREE_PORTS = 8
MAX_TREE_OPS = 40


def const_value(graph, k):
    """The common value of a uniform constant, else None."""
    v = graph.values.get(k)
    if v is None or v.constant is None or not v.constant:
        return None
    first = v.constant[0]
    return first if all(x == first for x in v.constant) else None


def _position_role(graph, prods, k):
    """True when ``k`` is the scalar runtime position, possibly reshaped."""
    seen = 0
    while seen < 16:
        v = graph.values[k]
        if v.role == "position" and v.size == 1:
            return True
        node = prods.get(k)
        if node is None or node.op not in ("reshape", "permute", "slice", "expand") or v.size != 1:
            return False
        k = node.inputs[0]
        seen += 1
    return False


def _position_name(graph):
    return next((k for k, v in graph.values.items() if v.role == "position"), None)


def _fit(table, counts):
    """Fit offset + sum(stride_i * index_i) to a dense integer table and verify it."""
    import itertools
    origin = table[(0,) * len(counts)]
    strides = []
    for axis, n in enumerate(counts):
        if n == 1:
            strides.append(0)
            continue
        one = [0] * len(counts)
        one[axis] = 1
        strides.append(table[tuple(one)] - origin)
    if origin < 0 or any(s < 0 for s in strides):
        return None
    for ix in itertools.product(*(range(n) for n in counts)):
        if table[ix] != origin + sum(s * i for s, i in zip(strides, ix)):
            return None
    return (origin,) + tuple(strides)


def _view_closure(graph, cons, start, sinks):
    """View-only nodes reachable from ``start`` must all end in ``sinks`` (nodes)."""
    sink_ids = {id(n) for n in sinks}
    seen, stack = set(), [start]
    while stack:
        k = stack.pop()
        for n in cons.get(k, []):
            if n.output in seen or id(n) in sink_ids:
                continue
            if n.op not in V.VIEW_NODE_OPS or graph.is_output(n.output):
                return None
            seen.add(n.output)
            stack.append(n.output)
    return seen


def _unit_batch(dims, count):
    """Split resolved dims into (head dim, rest) where only one batch dim is non-unit."""
    batch, tail = dims[:-2], dims[-2:]
    heads = [d for d in batch if d[0] != 1]
    if len(heads) > 1:
        return None
    head = heads[0] if heads else (1, 0, 1)
    if head[0] != count:
        return None
    return head, tail


def _cache_side(graph, prods, cons, base, sinks):
    """Validate a K or V cache operand: an indexed state write at ``position``."""
    write = prods.get(base)
    if write is None or write.op != "state_write" or len(write.inputs) != 3:
        return None
    prev, new, index = write.inputs
    read = prods.get(prev)
    if read is None or read.op != "state_read" or [id(u) for u in cons.get(prev, [])] != [id(write)]:
        return None
    if not _position_role(graph, prods, index):
        return None
    storage = write.writes[0]
    if read.inputs[0] != storage:
        return None
    closure = _view_closure(graph, cons, base, sinks)
    if closure is None:
        return None
    return write, read, storage, closure


def _cache_maps(graph, prods, storage, ds, view_dims, view_offset, kvh, width, new_value):
    """Compute cache (o, s_kv, s_d, s_j) and new-row maps for one cache operand."""
    shape = graph.values[storage].shape
    st = V.row_major(shape)
    head, dim, slot = view_dims
    if slot[0] != shape[ds] or slot[1] != st[ds] or slot[2] != 1 or dim[2] != 1:
        return None
    s_kv = head[1] if kvh > 1 else 0
    s_d = dim[1]
    rest = math.prod(shape) // shape[ds]
    if kvh * width != rest:
        return None
    new_shape = graph.values[new_value].shape
    if math.prod(new_shape) != rest:
        return None
    nbase, nview = V.resolve(graph, new_value, prods)
    if nview is None:
        return None
    seen, table = set(), {}
    for h in range(kvh):
        for d in range(width):
            flat = view_offset + h * s_kv + d * s_d
            ix = unravel(flat, shape)
            if ix[ds] != 0 or flat >= math.prod(shape):
                return None
            seen.add(flat)
            rest_ix = ix[:ds] + ix[ds + 1:]
            new_flat = ravel(rest_ix, shape[:ds] + shape[ds + 1:])
            table[(h, d)] = nview.at(unravel(new_flat, new_shape))
    if len(seen) != rest:
        return None
    new_map = _fit(table, (kvh, width))
    if new_map is None or view_offset < 0 or s_kv < 0 or s_d < 0:
        return None
    return (view_offset, s_kv, s_d, st[ds]), nbase, new_map


def _output_chain(graph, prods, cons, out):
    """Follow single-use reshape/permute consumers; return (final value, chain)."""
    chain = []
    k = out
    while True:
        users = cons.get(k, [])
        if len(users) != 1 or graph.is_output(k):
            break
        n = users[0]
        if n.op not in ("reshape", "permute") or n.inputs[0] != k:
            break
        chain.append(n)
        k = n.output
    return k, chain


def _match_attention(graph, S, prods, cons, order):
    single = lambda k: len(cons.get(k, [])) == 1 and not graph.is_output(k)
    shape = graph.values[S.output].shape
    if len(shape) < 2 or S.attrs["dim"] != len(shape) - 1 or shape[-2] != 1:
        return None
    removed = [S]
    k, mask, scale = S.inputs[0], None, 1.0
    for _ in range(3):
        n = prods.get(k)
        if n is None or not single(k):
            return None
        if n.op == "where" and mask is None:
            c, a, b = n.inputs
            neg = const_value(graph, b)
            if neg is None or neg > -1e30:
                return None
            mask, k = c, a
        elif n.op in ("mul", "div"):
            x, y = n.inputs
            cv = const_value(graph, y)
            if cv is None and n.op == "mul":
                x, cv = y, const_value(graph, x)
            if cv is None or cv == 0:
                return None
            scale *= cv if n.op == "mul" else 1 / cv
            k = x
        elif n.op == "matmul":
            break
        else:
            return None
        removed.append(n)
    qk = prods.get(k)
    if qk is None or qk.op != "matmul" or mask is None:
        return None
    removed.append(qk)
    # Probabilities: optional where(mask, p, 0), then p @ v.
    p = S.output
    users = cons.get(p, [])
    if len(users) != 1:
        return None
    n = users[0]
    if n.op == "where":
        if n.inputs[0] != mask or n.inputs[1] != p or const_value(graph, n.inputs[2]) != 0 or not single(n.output):
            return None
        removed.append(n)
        p = n.output
        n = cons[p][0]
    if n.op != "matmul" or n.inputs[0] != p:
        return None
    pv = n
    removed.append(pv)
    # The mask must be arange(n_positions) <= position along the last axis.
    mnode = prods.get(mask)
    mshape = graph.values[mask].shape
    positions = shape[-1]
    if mnode is None or not mshape or mshape[-1] != positions or any(d != 1 for d in mshape[:-1]):
        return None
    if mnode.op == "le":
        ar, pos = mnode.inputs
    elif mnode.op == "ge":
        pos, ar = mnode.inputs
    else:
        return None
    if graph.values[ar].constant != tuple(float(i) for i in range(positions)):
        return None
    if not _position_role(graph, prods, pos):
        return None
    batch = shape[:-2]
    heads = math.prod(batch)
    q, kt, v = qk.inputs[0], qk.inputs[1], pv.inputs[1]
    width = graph.values[q].shape[-1]
    vdim = graph.values[pv.output].shape[-1]
    qb, qv = V.resolve(graph, q, prods)
    kb, kv_ = V.resolve(graph, kt, prods)
    vb, vv = V.resolve(graph, v, prods)
    if qv is None or kv_ is None or vv is None:
        return None
    qv = V.broadcast_to(qv, batch + (1, width))
    kv_ = V.broadcast_to(kv_, batch + (width, positions))
    vv = V.broadcast_to(vv, batch + (positions, vdim))
    if qv is None or kv_ is None or vv is None:
        return None
    qsplit, ksplit, vsplit = (_unit_batch(x.dims, heads) for x in (qv, kv_, vv))
    if qsplit is None or ksplit is None or vsplit is None:
        return None
    qhead, (_, qd) = qsplit
    if qhead[2] != 1 or qd[2] != 1:
        return None

    def groups(head):
        if heads == 1 or head[1] == 0:
            return heads
        return head[2]
    group = groups(ksplit[0])
    if group != groups(vsplit[0]) or heads % group:
        return None
    kvh = heads // group
    kside = _cache_side(graph, prods, cons, kb, [qk])
    vside = _cache_side(graph, prods, cons, vb, [pv])
    if kside is None or vside is None or kside[0] is vside[0]:
        return None
    kwrite, kread, kstorage, kclosure = kside
    vwrite, vread, vstorage, vclosure = vside
    if kstorage == vstorage:
        return None
    kd, kj = ksplit[1]
    vj, ve = vsplit[1]
    kmaps = _cache_maps(graph, prods, kstorage, kwrite.attrs["dim"], (ksplit[0], kd, kj),
                        kv_.offset, kvh, width, kwrite.inputs[1])
    vmaps = _cache_maps(graph, prods, vstorage, vwrite.attrs["dim"], (vsplit[0], ve, vj),
                        vv.offset, kvh, vdim, vwrite.inputs[1])
    if kmaps is None or vmaps is None:
        return None
    # Output: absorb a pure reshape/permute chain when it is a bijection.
    final, chain = _output_chain(graph, prods, cons, pv.output)
    out_shape = graph.values[pv.output].shape
    if chain:
        fb, fview = V.resolve(graph, final, prods, stop={pv.output})
        size = graph.values[final].size
        if fb != pv.output or fview is None:
            final, chain = pv.output, []
        else:
            inverse = {}
            fshape = graph.values[final].shape
            for f in range(size):
                inverse[fview.at(unravel(f, fshape))] = f
            if len(inverse) != size:
                final, chain = pv.output, []
    table = {}
    for h in range(heads):
        for e in range(vdim):
            flat = h * vdim + e
            table[(h, e)] = inverse[flat] if chain else flat
    out_map = _fit(table, (heads, vdim))
    if out_map is None:
        return None
    attrs = {"heads": heads, "group": group, "head_dim": width, "value_dim": vdim,
             "positions": positions, "scale": scale,
             "q": (qv.offset, qhead[1] if heads > 1 else 0, qd[1] if width > 1 else 0),
             "k_new": kmaps[2][:3], "v_new": vmaps[2][:3],
             "k_cache": kmaps[0], "v_cache": vmaps[0], "out": out_map}
    position = _position_name(graph)
    node = Node("attention_decode", (qb, kmaps[1], vmaps[1], position, kstorage, vstorage), final,
                attrs, S.scope, writes=(kstorage, vstorage))
    dead = removed + chain + [kwrite, kread, vwrite, vread]
    return node, dead


def fuse_attention(graph):
    """Rewrite causal decode attention over K/V caches into ``attention_decode``."""
    changed = True
    while changed:
        changed = False
        prods, cons = producers(graph), consumers(graph)
        order = {n.output: i for i, n in enumerate(graph.nodes)}
        for S in graph.nodes:
            if S.op != "softmax":
                continue
            match = _match_attention(graph, S, prods, cons, order)
            if match is None:
                continue
            node, dead = match
            dead_ids = {id(n) for n in dead}
            anchor = max(order[n.output] for n in dead if n.output in order and not n.effectful)
            out = []
            for i, n in enumerate(graph.nodes):
                if i == anchor:
                    out.append(node)
                elif id(n) not in dead_ids:
                    out.append(n)
            graph.nodes = out
            eliminate_dead_code(graph)
            changed = True
            break


def _rotary_source(graph, prods, cons, value, user, width):
    """If ``value`` is a single-use decode RoPE output, return (input, base, rows map)."""
    node = prods.get(value)
    if node is None or node.op != "rotary" or [id(u) for u in cons.get(value, [])] != [id(user)]:
        return None
    x, pos = node.inputs
    if not _position_role(graph, prods, pos) or graph.values[x].shape[-1] != width:
        return None
    return node


def _compose_rows(graph, prods, rope, rows, width, row_map):
    """Map (row, d) through a contiguous rotary output back to its input base."""
    o, s_row, s_d = row_map
    if s_d != 1 and width > 1:
        return None
    x = rope.inputs[0]
    shape = graph.values[x].shape
    xb, xv = V.resolve(graph, x, prods)
    if xv is None:
        return None
    table = {}
    for r in range(rows):
        start = o + r * s_row
        if start % width:
            return None
        for d in range(width):
            table[(r, d)] = xv.at(unravel(start + d, shape))
    fitted = _fit(table, (rows, width))
    if fitted is None:
        return None
    return xb, fitted


def fuse_rotary_attention(graph):
    """Fold decode RoPE of q and of the new cache row k into ``attention_decode``."""
    prods, cons = producers(graph), consumers(graph)
    out, dead = [], set()
    for n in graph.nodes:
        if n.op != "attention_decode" or n.attrs.get("rope"):
            out.append(n)
            continue
        a = dict(n.attrs)
        inputs = list(n.inputs)
        width = a["head_dim"]
        rope = {}
        q = _rotary_source(graph, prods, cons, inputs[0], n, width) if inputs[0] != inputs[1] else None
        k = _rotary_source(graph, prods, cons, inputs[1], n, width) if inputs[0] != inputs[1] else None
        if q is not None and k is not None and q.attrs["base"] != k.attrs["base"]:
            q = k = None
        for which, node, port, rows in (("q", q, 0, a["heads"]), ("k", k, 1, a["heads"] // a["group"])):
            if node is None:
                continue
            key = "q" if which == "q" else "k_new"
            composed = _compose_rows(graph, prods, node, rows, width, a[key])
            if composed is None:
                continue
            inputs[port] = composed[0]
            a[key] = composed[1]
            rope[which] = True
            rope["base"] = node.attrs["base"]
            dead.add(id(node))
        if rope:
            a["rope"] = rope
            n = replace(n, inputs=tuple(inputs), attrs=a)
        out.append(n)
    graph.nodes = [n for n in out if id(n) not in dead]
    eliminate_dead_code(graph)


def push_slice_into_conv(graph):
    """Compute only the time steps a slice keeps from a causal convolution."""
    cons = consumers(graph)
    out = []
    replaced = {}
    for n in graph.nodes:
        if n.op == "conv1d" and "times" not in n.attrs:
            users = cons.get(n.output, [])
            if len(users) == 1 and users[0].op == "slice" and not graph.is_output(n.output):
                sl = users[0]
                spec = sl.attrs["index"]
                shape = graph.values[n.output].shape
                if all(isinstance(s, tuple) and s == (0, d, 1) for s, d in zip(spec[:2], shape[:2])):
                    t = spec[2]
                    times = (t,) if isinstance(t, int) else tuple(range(*t))
                    new_shape = shape[:2] + (len(times),)
                    graph.values[n.output] = replace(graph.values[n.output], shape=new_shape,
                                                     size=math.prod(new_shape))
                    n = replace(n, attrs=dict(n.attrs, times=times))
                    replaced[id(sl)] = Node("reshape", (n.output,), sl.output, {}, sl.scope)
        out.append(replaced.get(id(n), n))
    graph.nodes = out
    from .graph import fold_views
    fold_views(graph)


def fuse_conv_state(graph):
    """Fuse a streaming depthwise conv: state history + new sample -> one step kernel."""
    prods, cons = producers(graph), consumers(graph)
    for conv in list(graph.nodes):
        if conv.op != "conv1d" or conv.attrs.get("times") is None:
            continue
        K = conv.attrs["kernel_size"]
        oshape = graph.values[conv.output].shape
        if conv.attrs["times"] != (K - 1,) or K < 2:
            continue
        x = conv.inputs[0]
        xshape = graph.values[x].shape
        C = xshape[1]
        if (conv.attrs["groups"] != C or oshape[1] != C or xshape[0] != 1 or xshape[2] != K
                or graph.values[conv.inputs[1]].shape != (C, 1, K)):
            continue
        # The window may be reshaped between cat and conv.
        window = x
        chain = []
        node = prods.get(window)
        while node is not None and node.op == "reshape" and len(cons.get(window, [])) == 1:
            chain.append(node)
            window = node.inputs[0]
            node = prods.get(window)
        cat = node
        if cat is None or cat.op != "cat" or len(cat.inputs) != 2:
            continue
        wshape = graph.values[window].shape
        if cat.attrs["dim"] != len(wshape) - 1 or math.prod(wshape[:-1]) != C or wshape[-1] != K:
            continue
        hist, sample = cat.inputs
        read = prods.get(hist)
        if read is None or read.op != "state_read":
            continue
        storage = read.inputs[0]
        sshape = graph.values[storage].shape
        if math.prod(sshape) != C * (K - 1) or graph.values[hist].shape[-1] != K - 1:
            continue
        if graph.values[sample].shape[-1] != 1:
            continue
        # The update keeps window[..., 1:].
        chain_ids = {id(c) for c in chain}
        users = [u for u in cons.get(window, []) if id(u) not in chain_ids]
        if (chain and len(users) != 1) or (not chain and len(users) != 2):
            continue
        writer = None
        for u in cons.get(hist, []):
            if u.op == "state_write" and u.inputs[0] == hist and len(u.inputs) == 2:
                writer = u
        if writer is None or {id(u) for u in cons.get(hist, [])} != {id(cat), id(writer)}:
            continue
        new_state = writer.inputs[1]
        nb, nview = V.resolve(graph, new_state, prods)
        if nb != window or nview is None:
            continue
        expect = V.slice_view(V.identity(wshape), tuple((0, d, 1) for d in wshape[:-1]) + ((1, K, 1),))
        if V.reshape(expect, sshape) != nview:
            continue
        if cons.get(writer.output):
            continue
        closure = _view_closure(graph, cons, window, [conv, writer] + chain)
        if closure is None:
            continue
        bias = len(conv.inputs) == 3
        position = _position_name(graph)
        inputs = (sample, conv.inputs[1]) + ((conv.inputs[2],) if bias else ()) + (storage, position)
        attrs = {"kernel_size": K, "channels": C, "bias": bias, "state_port": 3 if bias else 2}
        fused = Node("conv1d_step", inputs, conv.output, attrs, conv.scope, writes=(storage,))
        dead = {id(read), id(cat), id(writer), id(conv)} | {id(n) for n in chain}
        graph.nodes = [fused if n is conv else n for n in graph.nodes if id(n) not in dead or n is conv]
        eliminate_dead_code(graph)
        prods, cons = producers(graph), consumers(graph)


def _tree_of(node):
    if node.op == "fused":
        return node.attrs["tree"], list(node.inputs)
    return (node.op, tuple(("input", i) for i in range(len(node.inputs)))), list(node.inputs)


def _remap(tree, mapping):
    if tree[0] == "input":
        return mapping[tree[1]]
    if tree[0] in ("const", "state", "updated"):
        return tree
    return (tree[0], tuple(_remap(c, mapping) for c in tree[1]))


def tree_ops(tree):
    if tree[0] in ("input", "const", "state", "updated"):
        return []
    return [tree[0]] + [op for c in tree[1] for op in tree_ops(c)]


def _pointwise(node):
    return (node.op in ELEMENTWISE or node.op == "fused") and not node.effectful and not node.extra


def _inline(graph, node, ports, tree, prods, cons):
    """Inline single-use pointwise producers into ``tree`` over ``ports``.

    Returns (tree, ports, inlined producer nodes).
    """
    out_size = graph.values[node.output].size
    inlined_nodes = []
    changed = True
    while changed:
        changed = False
        for i, k in enumerate(ports):
            p = prods.get(k)
            if p is None or not _pointwise(p):
                continue
            if [id(u) for u in cons.get(k, [])] != [id(node)] or graph.is_output(k) \
                    or layer_scope(p.scope) != layer_scope(node.scope):
                continue
            ptree, pports = _tree_of(p)
            if graph.values[k].size != out_size and any(op in TRANSCENDENTAL for op in tree_ops(ptree)):
                continue
            new_ports = [x for j, x in enumerate(ports) if j != i]
            mapping = {}
            for j, x in enumerate(pports):
                if x not in new_ports:
                    new_ports.append(x)
                mapping[j] = ("input", new_ports.index(x))
            if len(new_ports) > MAX_TREE_PORTS:
                continue
            shift = {j: ("input", new_ports.index(x)) for j, x in enumerate(ports) if j != i}
            shift[i] = _remap(ptree, mapping)
            candidate = _remap(tree, shift)
            if len(tree_ops(candidate)) > MAX_TREE_OPS:
                continue
            tree, ports = candidate, new_ports
            inlined_nodes.append(p)
            changed = True
            break
    return tree, ports, inlined_nodes


def _literalize(graph, tree, ports):
    """Replace uniform scalar constant ports by literal leaves and drop them."""
    keep, mapping = [], {}
    for i, k in enumerate(ports):
        cv = const_value(graph, k)
        if cv is not None:
            mapping[i] = ("const", float(cv))
        else:
            if k not in keep:
                keep.append(k)
            mapping[i] = ("input", keep.index(k))
    return _remap(tree, mapping), keep


def fuse_pointwise(graph):
    """Fuse elementwise DAGs with single-use intermediates into expression trees."""
    nodes = list(graph.nodes)
    prods = {k: n for n in nodes for k in n.outputs}
    cons = consumers(graph)
    removed = set()
    for idx, n in enumerate(nodes):
        if not _pointwise(n):
            continue
        tree, ports = _tree_of(n)
        tree2, ports2, inlined = _inline(graph, n, ports, tree, prods, cons)
        tree2, ports2 = _literalize(graph, tree2, ports2)
        if (tree2 == tree and ports2 == ports) or not ports2:
            continue
        fused = replace(n, op="fused", inputs=tuple(ports2), attrs={"tree": tree2})
        nodes[idx] = fused
        prods[n.output] = fused
        for k in dict.fromkeys(n.inputs):
            cons[k] = [u for u in cons.get(k, []) if u is not n]
        for p in inlined:
            removed.add(id(p))
            for k in dict.fromkeys(p.inputs):
                cons[k] = [u for u in cons.get(k, []) if u is not p]
        for k in dict.fromkeys(ports2):
            cons.setdefault(k, []).append(fused)
    graph.nodes = [n for n in nodes if id(n) not in removed]
    eliminate_dead_code(graph)


def fuse_reduction_prologue(graph):
    """Fold a single-use pointwise producer into sum/mean/max."""
    prods, cons = producers(graph), consumers(graph)
    out, dead = [], set()
    for n in graph.nodes:
        if n.op in ("sum", "mean", "max") and "tree" not in n.attrs:
            k = n.inputs[0]
            p = prods.get(k)
            in_shape = graph.values[k].shape
            if (p is not None and _pointwise(p) and cons.get(k, []) == [n] and not graph.is_output(k)
                    and layer_scope(p.scope) == layer_scope(n.scope)):
                tree, ports = _tree_of(p)
                tree, ports = _literalize(graph, tree, ports)
                if ports:
                    n = replace(n, inputs=tuple(ports), attrs=dict(n.attrs, tree=tree, in_shape=in_shape))
                    dead.add(id(p))
            elif "tree" not in n.attrs:
                n = replace(n, attrs=dict(n.attrs, tree=("input", 0), in_shape=in_shape))
        out.append(n)
    graph.nodes = [n for n in out if id(n) not in dead]
    eliminate_dead_code(graph)


def fuse_state_update_reduction(graph):
    """SSM decode step: state <- f(state, inputs); y <- reduce(g(state_new, inputs))."""
    prods, cons = producers(graph), consumers(graph)
    for red in list(graph.nodes):
        if red.op not in ("sum", "mean") or "tree" in red.attrs:
            continue
        m = prods.get(red.inputs[0])
        if m is None or not _pointwise(m) or cons.get(m.output, []) != [red]:
            continue
        mtree, mports = _tree_of(m)
        writer = None
        for k in mports:
            p = prods.get(k)
            if p is not None and p.op == "state_write" and len(p.inputs) == 2:
                writer, updated_name = p, k
                value = p.inputs[1]
            elif p is not None and _pointwise(p):
                for u in cons.get(k, []):
                    if u.op == "state_write" and len(u.inputs) == 2 and u.inputs[1] == k:
                        writer, updated_name, value = u, k, k
        if writer is None:
            continue
        storage = writer.writes[0]
        sshape = graph.values[storage].shape
        update = prods.get(value)
        read = prods.get(writer.inputs[0])
        if update is None or not _pointwise(update) or read is None or read.op != "state_read":
            continue
        if graph.values[value].shape != sshape or graph.values[m.output].shape != sshape:
            continue
        scopes = {layer_scope(x.scope) for x in (red, m, update, writer)}
        if len(scopes) != 1:
            continue
        rk = read.output
        if set(id(u) for u in cons.get(rk, [])) != {id(update), id(writer)}:
            continue
        users_value = cons.get(value, [])
        users_written = cons.get(writer.output, [])
        if updated_name == value:
            if set(id(u) for u in users_value) != {id(writer), id(m)} or users_written:
                continue
        else:
            if [id(u) for u in users_value] != [id(writer)] or [id(u) for u in users_written] != [id(m)]:
                continue
        utree, uports = _tree_of(update)
        ports = []
        def port(k):
            if k not in ports:
                ports.append(k)
            return ports.index(k)
        umap = {}
        for j, k in enumerate(uports):
            umap[j] = ("state", port(storage)) if k == rk else ("input", port(k))
        if rk not in uports:
            continue
        rmap = {}
        for j, k in enumerate(mports):
            rmap[j] = ("updated",) if k == updated_name else ("input", port(k))
        new_update = _remap(utree, umap)
        new_reduce = _remap(mtree, rmap)
        position = _position_name(graph)
        if position is None or len(ports) + 1 > MAX_TREE_PORTS + 1:
            continue
        state_port = ports.index(storage)
        ports.append(position)
        attrs = {"update": new_update, "reduce": new_reduce, "dims": red.attrs["dims"],
                 "keepdim": red.attrs["keepdim"], "in_shape": sshape, "state_port": state_port,
                 "reset_port": len(ports) - 1, "kind": red.op}
        fused = Node("state_update_reduce", tuple(ports), red.output, attrs, red.scope, writes=(storage,))
        dead = {id(read), id(update), id(writer), id(m), id(red)}
        graph.nodes = [fused if x is red else x for x in graph.nodes if id(x) not in dead or x is red]
        eliminate_dead_code(graph)
        prods, cons = producers(graph), consumers(graph)


def fuse_norms(graph):
    """Fold residual/bias producers into LayerNorm/RMSNorm (keeping the sum if reused)."""
    prods, cons = producers(graph), consumers(graph)
    order = {id(n): i for i, n in enumerate(graph.nodes)}
    out, dead = [], set()
    for n in graph.nodes:
        if n.op not in ("layernorm", "rmsnorm"):
            out.append(n)
            continue
        x = n.inputs[0]
        p = prods.get(x)
        affine = list(n.inputs[1:])
        attrs = {"kind": n.op, "width": n.attrs["width"], "eps": n.attrs["eps"],
                 "weight": len(affine) >= 1, "bias": len(affine) >= 2}
        if p is not None and _pointwise(p) and layer_scope(p.scope) == layer_scope(n.scope) \
                and graph.values[x].shape == graph.values[n.output].shape and not graph.is_output(x):
            users = cons.get(x, [])
            later = all(u is n or order[id(u)] > order[id(n)] for u in users)
            tree, ports = _tree_of(p)
            tree, ports = _literalize(graph, tree, ports)
            if later and ports and len(ports) + len(affine) <= MAX_TREE_PORTS:
                side = len(users) > 1
                attrs.update(tree=tree, ports=len(ports))
                n = Node("norm", tuple(ports) + tuple(affine), n.output, attrs, n.scope,
                         extra=(x,) if side else ())
                dead.add(id(p))
        out.append(n)
    graph.nodes = [n for n in out if id(n) not in dead]
    eliminate_dead_code(graph)
