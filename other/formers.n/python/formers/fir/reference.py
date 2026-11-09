"""Pure-Python reference evaluator for traced and optimized graphs; no weight loading.

Primitive operations use the per-element scalar plans in ``scalar.py``. Fused
operations created by the optimization passes have direct implementations here,
written independently of the VM lowering, so raw and optimized graphs can be
compared step by step.
"""

import math
import random
from .scalar import plans, offset, unravel, broadcast_index, coordinates
from . import custom


def _sigmoid(x):
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def _softplus(x):
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def apply_op(op, args):
    if op == "add": return args[0]+args[1]
    if op == "sub": return args[0]-args[1]
    if op == "mul": return args[0]*args[1]
    if op == "div": return args[0]/args[1]
    if op == "pow": return args[0]**args[1]
    if op == "maximum": return max(args[0], args[1])
    if op == "minimum": return min(args[0], args[1])
    if op == "neg": return -args[0]
    if op == "lt": return float(args[0] < args[1])
    if op == "le": return float(args[0] <= args[1])
    if op == "gt": return float(args[0] > args[1])
    if op == "ge": return float(args[0] >= args[1])
    if op == "eq": return float(args[0] == args[1])
    if op == "ne": return float(args[0] != args[1])
    if op == "where": return args[1] if args[0] else args[2]
    x = args[0]
    if op == "relu": return max(0.0, x)
    if op == "sigmoid": return _sigmoid(x)
    if op == "silu": return x * _sigmoid(x)
    if op == "softplus": return _softplus(x)
    if op == "gelu_new": return 0.5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3)))
    if op == "gelu": return 0.5*x*(1+math.erf(x/math.sqrt(2)))
    if op == "gelu_quick": return x * _sigmoid(1.702*x)
    if op == "rsqrt": return 1/math.sqrt(x)
    if op == "abs": return abs(x)
    if op in ("exp", "log", "sqrt", "tanh", "sin", "cos"): return getattr(math, op)(x)
    raise ValueError("reference op missing: " + op)


def eval_expr(e, inputs):
    op = e[0]
    if op == "const": return e[1]
    if op == "load": return inputs[e[1]][e[2]]
    if op == "gather":
        idx = eval_expr(e[2], inputs)
        if int(idx) != idx or not 0 <= idx < e[5]: raise IndexError("embedding index out of range")
        return inputs[e[1]][int(idx)*e[3]+e[4]]
    args = [eval_expr(x, inputs) for x in e[1]]
    if op == "sum_expr": return sum(args)
    if op == "max_expr": return max(args)
    return apply_op(op, args)


def eval_tree(tree, leaf):
    """Evaluate a fused tree; ``leaf(kind, port)`` supplies leaf values."""
    kind = tree[0]
    if kind in ("input", "state", "updated"): return leaf(kind, tree[1] if len(tree) > 1 else None)
    if kind == "const": return tree[1]
    return apply_op(kind, [eval_tree(child, leaf) for child in tree[1]])


def _broadcast_leaf(inputs, shapes, ix):
    def leaf(kind, port):
        shape = shapes[port]
        return inputs[port][offset(broadcast_index(ix, shape), shape)]
    return leaf


def _norm(node, graph, inputs, shapes):
    attrs = node.attrs
    out_shape = graph.values[node.output].shape
    width = attrs["width"]
    tree = attrs.get("tree", ("input", 0))
    ports = attrs.get("ports", 1)
    xs = [eval_tree(tree, _broadcast_leaf(inputs, shapes, ix)) for ix in coordinates(out_shape)]
    weight = inputs[ports] if attrs.get("weight", len(inputs) > ports) else None
    bias = inputs[ports + 1] if attrs.get("bias", len(inputs) > ports + 1) else None
    out = []
    for r in range(0, len(xs), width):
        row = xs[r:r+width]
        if attrs.get("kind", node.op) == "layernorm":
            mean = sum(row)/width
            centered = [x-mean for x in row]
        else:
            centered = row
        inv = 1/math.sqrt(sum(x*x for x in centered)/width + attrs["eps"])
        for j, x in enumerate(centered):
            y = x*inv
            if weight is not None: y *= weight[j]
            if bias is not None: y += bias[j]
            out.append(y)
    result = {node.output: out}
    if node.extra: result[node.extra[0]] = xs
    return result


def _attention(node, graph, inputs, states):
    a = node.attrs
    q, knew, vnew, position = inputs[0], inputs[1], inputs[2], inputs[3]
    kc, vc = list(inputs[4]), list(inputs[5])
    pos = int(position[0])
    H, G, D, Dv = a["heads"], a["group"], a["head_dim"], a["value_dim"]
    kvh = H // G
    rot = a.get("rope")

    def rope(values, base_offset, s_d, angle_pos):
        half = D // 2
        out = []
        for d in range(D):
            i = d % half
            theta = angle_pos * rot["base"] ** (-i / half)
            x = values[base_offset + d*s_d]
            pair = values[base_offset + ((d + half) % D)*s_d]
            other = -pair if d < half else pair
            out.append(x*math.cos(theta) + other*math.sin(theta))
        return out

    ko, ks, kd = a["k_new"]
    vo, vs, vd = a["v_new"]
    kco, kcs, kcd, kcj = a["k_cache"]
    vco, vcs, vcd, vcj = a["v_cache"]
    for h in range(kvh):
        row = rope(knew, ko + h*ks, kd, pos) if rot and rot.get("k") else [knew[ko + h*ks + d*kd] for d in range(D)]
        for d in range(D):
            kc[kco + h*kcs + d*kcd + pos*kcj] = row[d]
        for e in range(Dv):
            vc[vco + h*vcs + e*vcd + pos*vcj] = vnew[vo + h*vs + e*vd]
    qo, qs, qd = a["q"]
    oo, os_, oe = a["out"]
    out = [0.0] * graph.values[node.output].size
    for h in range(H):
        kv = h // G
        qrow = rope(q, qo + h*qs, qd, pos) if rot and rot.get("q") else [q[qo + h*qs + d*qd] for d in range(D)]
        scores = [a["scale"] * sum(qrow[d]*kc[kco + kv*kcs + d*kcd + j*kcj] for d in range(D))
                  for j in range(pos + 1)]
        m = max(scores)
        weights = [math.exp(s - m) for s in scores]
        total = sum(weights)
        for e in range(Dv):
            out[oo + h*os_ + e*oe] = sum(w*vc[vco + kv*vcs + e*vcd + j*vcj] for j, w in enumerate(weights)) / total
    return {node.output: out, node.writes[0]: kc, node.writes[1]: vc}


def _reduce_tree(node, graph, inputs, shapes):
    a = node.attrs
    in_shape, dims = a["in_shape"], a["dims"]
    out_shape = graph.values[node.output].shape
    kept = [d for d in range(len(in_shape)) if d not in dims]
    groups = {}
    for ix in coordinates(in_shape):
        x = eval_tree(a["tree"], _broadcast_leaf(inputs, shapes, ix))
        key = tuple(ix[d] for d in kept)
        groups.setdefault(key, []).append(x)
    out = []
    for key in coordinates(tuple(in_shape[d] for d in kept)):
        xs = groups[key]
        out.append(max(xs) if node.op == "max" else sum(xs)/(len(xs) if node.op == "mean" else 1))
    assert len(out) == graph.values[node.output].size
    return {node.output: out}


def _state_update_reduce(node, graph, inputs, shapes):
    a = node.attrs
    in_shape, dims = a["in_shape"], a["dims"]
    storage = list(inputs[a["state_port"]])
    kept = [d for d in range(len(in_shape)) if d not in dims]
    groups = {}
    for ix in coordinates(in_shape):
        flat = offset(ix, in_shape)
        def leaf(kind, port, ix=ix):
            if kind == "state": return inputs[port][flat]
            shape = shapes[port]
            return inputs[port][offset(broadcast_index(ix, shape), shape)]
        updated = eval_tree(a["update"], leaf)
        storage[flat] = updated
        def rleaf(kind, port, ix=ix, updated=updated):
            return updated if kind == "updated" else leaf(kind, port)
        key = tuple(ix[d] for d in kept)
        groups.setdefault(key, []).append(eval_tree(a["reduce"], rleaf))
    out = [sum(groups[key]) for key in coordinates(tuple(in_shape[d] for d in kept))]
    return {node.output: out, node.writes[0]: storage}


def _conv_step(node, graph, inputs):
    a = node.attrs
    K, C = a["kernel_size"], a["channels"]
    x, weight = inputs[0], inputs[1]
    bias = inputs[2] if a["bias"] else None
    state = list(inputs[a["state_port"]])
    out = []
    for c in range(C):
        window = state[c*(K-1):(c+1)*(K-1)] + [x[c]]
        y = sum(window[j]*weight[c*K + j] for j in range(K))
        out.append(y + (bias[c] if bias else 0.0))
        state[c*(K-1):(c+1)*(K-1)] = window[1:]
    return {node.output: out, node.writes[0]: state}


def run_node(node, graph, values):
    """Evaluate one node and return {value or storage name: flat list}."""
    inputs = [values[k] for k in node.inputs]
    shapes = [graph.values[k].shape for k in node.inputs]
    if node.op in custom.OPS:
        return custom.run_reference(node, graph, inputs, shapes)
    if node.op == "norm" or (node.op in ("layernorm", "rmsnorm") and node.extra):
        return _norm(node, graph, inputs, shapes)
    if node.op == "attention_decode":
        return _attention(node, graph, inputs, None)
    if node.op in ("sum", "mean", "max") and "tree" in node.attrs:
        return _reduce_tree(node, graph, inputs, shapes)
    if node.op == "state_update_reduce":
        return _state_update_reduce(node, graph, inputs, shapes)
    if node.op == "conv1d_step":
        return _conv_step(node, graph, inputs)
    result = [eval_expr(e, inputs) for e in plans(node, graph)]
    out = {node.output: result}
    for storage in node.writes:
        out[storage] = result
    return out


def evaluate_node(node, graph, values):
    return run_node(node, graph, values)[node.output]


def evaluate(graph, inputs, parameters=None, states=None, return_states=False, all_outputs=False):
    """Execute flat input/parameter lists. State dictionaries persist between calls.

    A position input equal to zero starts a new sequence and clears all states.
    ``all_outputs=True`` returns a list with every graph output (``graph.outputs``).
    """
    values = {}
    parameters, states = parameters or {}, states if states is not None else {}
    if inputs.get("position") in ([0], [0.0]):
        states.clear()
    for k, v in graph.values.items():
        if v.weight:
            if v.weight.name not in parameters: raise ValueError("missing reference parameter: " + v.weight.name)
            values[k] = list(parameters[v.weight.name])
        elif v.constant is not None: values[k] = list(v.constant)
        elif v.state: values[k] = list(states.get(v.state, [0.0]*v.size))
        elif v.role: values[k] = list(inputs[k])
        if k in values and len(values[k]) != v.size: raise ValueError("reference input size mismatch: " + k)
    for n in graph.nodes:
        for name, data in run_node(n, graph, values).items():
            values[name] = data
            if graph.values[name].state:
                states[graph.values[name].state] = list(data)
    result = [values[k] for k in graph.output_names()] if all_outputs else values[graph.output]
    return (result, states) if return_states else result


def random_parameters(graph, seed=0, scale=0.2):
    rng = random.Random(seed)
    return {v.weight.name: [rng.uniform(-scale, scale) for _ in range(v.size)]
            for v in sorted(graph.values.values(), key=lambda v: v.weight.name if v.weight else "")
            if v.weight}


def compare_graphs(original, optimized, input_steps, parameters=None, *, rel_tol=1e-6, abs_tol=1e-7):
    """Compare explicit synthetic fixtures, including persistent state and resets.

    Parameters are supplied as flat lists by the caller; no checkpoint is opened.
    Return the maximum absolute difference and raise on the first mismatch.
    """
    left_states, right_states, largest = {}, {}, 0.0
    for step, inputs in enumerate(input_steps):
        left = evaluate(original, inputs, parameters, left_states, all_outputs=True)
        right = evaluate(optimized, inputs, parameters, right_states, all_outputs=True)
        if len(left) != len(right): raise AssertionError(f"output counts differ at step {step}")
        pairs = [("output" if i == 0 else "output%d" % i, a, b) for i, (a, b) in enumerate(zip(left, right))]
        if left_states.keys() != right_states.keys():
            raise AssertionError(f"state names differ at step {step}")
        pairs += [(key, left_states[key], right_states[key]) for key in left_states]
        for key, a, b in pairs:
            if len(a) != len(b): raise AssertionError(f"{key} size differs at step {step}")
            for i, (x, y) in enumerate(zip(a, b)):
                if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol):
                    raise AssertionError(f"{key}[{i}] differs at step {step}: {x} != {y}")
                largest = max(largest, abs(x-y))
    return largest
