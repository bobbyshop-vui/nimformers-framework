"""Scalar expression plans for reference evaluation and portable VM lowering."""

import itertools
import math


def coordinates(shape): return itertools.product(*(range(d) for d in shape))


def offset(index, shape):
    result = 0
    for i, d in zip(index, shape): result = result*d + i
    return result


def unravel(i, shape):
    out = []
    for d in reversed(shape):
        out.append(i % d)
        i //= d
    return tuple(reversed(out))


def broadcast_index(index, shape):
    return tuple(0 if d == 1 else c for c, d in zip(index[len(index)-len(shape):] if shape else (), shape))


def literal(x): return ("const", float(x))
def expr(op, *args): return (op, tuple(args))
def total(xs): return ("sum_expr", tuple(xs))
def maximum(xs): return ("max_expr", tuple(xs))


BINARY = ("add", "sub", "mul", "div", "pow", "maximum", "minimum",
          "lt", "le", "gt", "ge", "eq", "ne", "where")
UNARY = ("neg", "gelu_new", "gelu", "gelu_quick", "relu", "silu", "tanh", "sigmoid", "softplus",
         "exp", "log", "sqrt", "rsqrt", "abs", "sin", "cos")


def tree_expr(tree, leaf):
    """Expand a fused expression tree; ``leaf(port)`` returns the input expression."""
    if tree[0] == "input": return leaf(tree[1])
    if tree[0] == "const": return literal(tree[1])
    return expr(tree[0], *(tree_expr(child, leaf) for child in tree[1]))


def plans(node, graph):
    """Yield one expression per flattened output element, preserving row-major layout."""
    shapes = [graph.values[k].shape for k in node.inputs]
    outshape = graph.values[node.output].shape
    op, attrs = node.op, node.attrs

    def load(p, ix):
        return ("load", p, offset(ix, shapes[p]))
    def bload(p, ix): return load(p, broadcast_index(ix, shapes[p]))

    for ix in coordinates(outshape):
        if op in BINARY:
            yield expr(op, *(bload(p, ix) for p in range(len(shapes))))
        elif op in UNARY:
            yield expr(op, load(0, ix))
        elif op == "expand": yield bload(0, ix)
        elif op == "fused":
            yield tree_expr(attrs["tree"], lambda p: bload(p, ix))
        elif op == "state_read":
            y = load(0, ix)
            if "reset_port" in attrs:
                y = expr("where", expr("eq", ("load", attrs["reset_port"], 0), literal(0)), literal(0), y)
            yield y
        elif op == "reshape": yield ("load", 0, offset(ix, outshape))
        elif op == "permute":
            src = [0] * len(ix)
            for j, d in enumerate(attrs["dims"]): src[d] = ix[j]
            yield load(0, src)
        elif op == "slice":
            src, j = [], 0
            for s in attrs["index"]:
                if isinstance(s, int): src.append(s)
                else:
                    src.append(s[0]+ix[j]*s[2])
                    j += 1
            yield load(0, src)
        elif op == "cat":
            dim, n = attrs["dim"], ix[attrs["dim"]]
            for p, shape in enumerate(shapes):
                if n < shape[dim]:
                    src = list(ix)
                    src[dim] = n
                    yield load(p, src)
                    break
                n -= shape[dim]
        elif op in ("sum", "mean", "max"):
            dims = attrs["dims"]
            src, j = [], 0
            for d in range(len(shapes[0])):
                if d in dims:
                    src.append(0)
                    if attrs["keepdim"]: j += 1
                else:
                    src.append(ix[j])
                    j += 1
            terms = []
            for rc in coordinates(tuple(shapes[0][d] for d in dims)):
                for d, c in zip(dims, rc): src[d] = c
                terms.append(load(0, src))
            result = maximum(terms) if op == "max" else total(terms)
            yield expr("mul", result, literal(1/len(terms))) if op == "mean" else result
        elif op == "cumsum":
            dim = attrs["dim"]
            yield total(load(0, ix[:dim]+(j,)+ix[dim+1:]) for j in range(ix[dim]+1))
        elif op == "softmax":
            dim = attrs["dim"]
            row = [load(0, ix[:dim]+(j,)+ix[dim+1:]) for j in range(shapes[0][dim])]
            m = maximum(row)
            exps = [expr("exp", expr("sub", x, m)) for x in row]
            yield expr("div", exps[ix[dim]], total(exps))
        elif op in ("layernorm", "rmsnorm"):
            width = attrs["width"]
            flat = offset(ix, outshape)
            base = flat // width * width
            row = [("load", 0, base+j) for j in range(width)]
            avg = expr("mul", total(row), literal(1/width)) if op == "layernorm" else literal(0)
            centered = [expr("sub", x, avg) for x in row]
            variance = expr("mul", total(expr("mul", x, x) for x in centered), literal(1/width))
            y = expr("mul", centered[flat % width], expr("rsqrt", expr("add", variance, literal(attrs["eps"]))))
            if len(shapes) > 1: y = expr("mul", y, ("load", 1, flat % width))
            if len(shapes) > 2: y = expr("add", y, ("load", 2, flat % width))
            yield y
        elif op == "matmul":
            a, b = shapes
            aa = (1,)+a if len(a) == 1 else a
            bb = b+(1,) if len(b) == 1 else b
            full = ix
            if len(a) == 1: full = ix[:-1]+(0,)+ix[-1:] if len(b) > 1 else (0, 0)
            if len(b) == 1 and len(a) > 1: full = ix+(0,)
            batch, m, n = full[:-2], full[-2], full[-1]
            terms = []
            for j in range(aa[-1]):
                ia = broadcast_index(batch, aa[:-2]) + (m, j)
                ib = broadcast_index(batch, bb[:-2]) + (j, n)
                if len(a) == 1: ia = (j,)
                if len(b) == 1: ib = (j,)
                terms.append(expr("mul", load(0, ia), load(1, ib)))
            yield total(terms)
        elif op in ("linear", "linear_t"):
            width = shapes[0][-1]
            yield total(expr("mul", load(0, ix[:-1]+(j,)),
                             load(1, (ix[-1], j) if op == "linear_t" else (j, ix[-1]))) for j in range(width))
        elif op == "embedding":
            yield ("gather", 1, load(0, ix[:-1]), shapes[1][1], ix[-1], shapes[1][0])
        elif op == "index_select":
            d = attrs["dim"]
            terms = []
            for j in range(shapes[0][d]):
                src = ix[:d]+(j,)+ix[d+1:]
                terms.append(expr("where", expr("eq", load(1, (ix[d],)), literal(j)), load(0, src), literal(0)))
            yield total(terms)
        elif op == "state_write":
            if len(shapes) == 2: yield load(1, ix)
            else:
                d = attrs["dim"]
                yield expr("where", expr("eq", ("load", 2, 0), literal(ix[d])),
                           load(1, ix[:d]+ix[d+1:]), load(0, ix))
        elif op == "conv1d":
            b, c, pos = ix
            if "times" in attrs: pos = attrs["times"][pos]
            terms = []
            channels = shapes[1][1]
            group = c // (outshape[1] // attrs["groups"])
            for channel in range(channels):
                for j in range(attrs["kernel_size"]):
                    t = pos - attrs["kernel_size"]+1+j
                    if t >= 0:
                        terms.append(expr("mul", load(0, (b,group*channels+channel,t)), load(1, (c,channel,j))))
            y = total(terms)
            yield expr("add", y, load(2, (c,))) if len(shapes) == 3 else y
        elif op == "rotary":
            # HF rotate_half convention; pair first and second half of the head.
            half = outshape[-1]//2
            d = ix[-1]
            pair = ix[:-1]+((d+half) % (2*half),)
            angle = expr("mul", bload(1, ix[:-1]), literal(attrs["base"] ** (-(d % half)/half)))
            other = load(0, pair)
            if d < half: other = expr("neg", other)
            yield expr("add", expr("mul", load(0, ix), expr("cos", angle)), expr("mul", other, expr("sin", angle)))
        else:
            raise ValueError("unsupported traced operation %s; add a scalar plan and VM lowering" % op)
