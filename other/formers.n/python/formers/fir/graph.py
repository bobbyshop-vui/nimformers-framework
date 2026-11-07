"""Typed SSA metadata and ordered, effect-aware graph transformations."""

from dataclasses import dataclass, field, replace
import re


@dataclass(frozen=True)
class Weight:
    name: str
    shape: tuple
    dtype: str = "f32"


@dataclass(frozen=True)
class Value:
    size: int
    role: str = ""
    weight: Weight = None
    shape: tuple = ()
    dtype: str = "f32"
    constant: tuple = None
    state: str = ""


@dataclass(frozen=True)
class Node:
    """One SSA operation.

    ``output`` is the primary result. ``extra`` lists additional results produced
    by fused operations (for example the residual sum of a fused residual+norm).
    ``writes`` names persistent state storages that the operation updates.
    """
    op: str
    inputs: tuple
    output: str
    attrs: dict = field(default_factory=dict)
    scope: str = ""
    extra: tuple = ()
    writes: tuple = ()

    @property
    def outputs(self):
        return (self.output,) + tuple(self.extra)

    @property
    def effectful(self):
        return bool(self.writes)


@dataclass
class Graph:
    name: str
    config: dict
    values: dict = field(default_factory=dict)
    nodes: list = field(default_factory=list)
    output: str = ""
    symbols: dict = field(default_factory=dict)
    passes: list = field(default_factory=list)
    hints: dict = field(default_factory=dict)
    outputs: tuple = ()

    def output_names(self):
        """Every graph output; ``output`` is the first (the decode logits)."""
        return (self.output,) + tuple(k for k in self.outputs if k != self.output)

    def is_output(self, k):
        return k == self.output or k in self.outputs


VIEW_OPS = {"reshape", "permute", "slice", "cat"}


def producers(graph):
    return {k: n for n in graph.nodes for k in n.outputs}


def consumers(graph):
    out = {}
    for n in graph.nodes:
        for k in dict.fromkeys(n.inputs):
            out.setdefault(k, []).append(n)
    return out


def layer_scope(scope):
    m = re.match(r"^(.*?\.\d+)(?:\.|$)", scope)
    return m.group(1) if m else ""


def eliminate_dead_code(graph):
    live = set(graph.output_names())
    kept = []
    for node in reversed(graph.nodes):
        if node.effectful or any(k in live for k in node.outputs):
            kept.append(node)
            live.update(node.inputs)
            live.update(node.writes)
            live.update(node.outputs)
    graph.nodes = list(reversed(kept))
    graph.values = {k: v for k, v in graph.values.items() if k in live or v.role or v.state}


def constant_fold(graph):
    """Fold pure constant subgraphs with the same evaluator as the reference path."""
    from .reference import evaluate_node
    constants = {k: list(v.constant) for k, v in graph.values.items() if v.constant is not None}
    kept = []
    for n in graph.nodes:
        if not n.effectful and not n.extra and n.inputs and all(k in constants for k in n.inputs):
            result = evaluate_node(n, graph, constants)
            constants[n.output] = result
            graph.values[n.output] = replace(graph.values[n.output], constant=tuple(result))
        else:
            kept.append(n)
    graph.nodes = kept


def _redirect(graph, aliases):
    def root(k):
        while k in aliases:
            k = aliases[k]
        return k
    graph.nodes = [replace(n, inputs=tuple(root(k) for k in n.inputs)) for n in graph.nodes]
    graph.output = root(graph.output)
    graph.outputs = tuple(dict.fromkeys(root(k) for k in graph.outputs))


def common_subexpressions(graph):
    seen, aliases, kept = {}, {}, []
    for n in graph.nodes:
        ins = tuple(aliases.get(k, k) for k in n.inputs)
        n = replace(n, inputs=ins)
        # States are mutable across writes; never merge state-dependent reads.
        stateful = any(graph.values[k].state for k in ins)
        key = (n.op, ins, repr(sorted(n.attrs.items())), n.scope)
        mergeable = not n.effectful and not n.extra and not stateful
        if mergeable and key in seen:
            aliases[n.output] = seen[key]
        else:
            kept.append(n)
            if mergeable:
                seen[key] = n.output
    graph.nodes = kept
    _redirect(graph, aliases)
    eliminate_dead_code(graph)


ELEMENTWISE = {"add", "sub", "mul", "div", "pow", "neg", "gelu_new", "gelu", "gelu_quick", "relu",
               "silu", "tanh", "sigmoid", "softplus", "exp", "log", "sqrt", "rsqrt", "abs",
               "maximum", "minimum", "sin", "cos", "lt", "le", "gt", "ge", "eq", "ne", "where"}


def fold_views(graph):
    """Remove identity reshapes/slices, merge reshape chains and inverse permutations."""
    aliases, kept = {}, []
    producers_ = {n.output: n for n in graph.nodes}
    for n in graph.nodes:
        src = aliases.get(n.inputs[0], n.inputs[0]) if n.inputs else None
        p = producers_.get(src) if src else None
        if n.op == "reshape" and graph.values[src].shape == graph.values[n.output].shape:
            aliases[n.output] = src
        elif n.op == "reshape" and p is not None and p.op == "reshape":
            n = replace(n, inputs=p.inputs)
            producers_[n.output] = n
            kept.append(n)
        elif n.op == "slice" and all(isinstance(s, tuple) and s == (0, d, 1) for s, d in
                                     zip(n.attrs["index"], graph.values[src].shape)):
            aliases[n.output] = src
        elif n.op == "permute" and p is not None and p.op == "permute":
            order = tuple(p.attrs["dims"][d] for d in n.attrs["dims"])
            if order == tuple(range(len(order))):
                aliases[n.output] = p.inputs[0]
            else:
                n = replace(n, inputs=p.inputs, attrs={"dims": order})
                producers_[n.output] = n
                kept.append(n)
        elif n.op == "permute" and n.attrs["dims"] == tuple(range(len(n.attrs["dims"]))):
            aliases[n.output] = src
        else:
            kept.append(n)
    graph.nodes = kept
    _redirect(graph, aliases)
    eliminate_dead_code(graph)


def select_q4(graph):
    """Record eligible resident layouts; never reinterpret checkpoint payloads."""
    graph.hints["q4_candidates"] = sorted({graph.values[n.inputs[1]].weight.name for n in graph.nodes
                                         if n.op == "linear" and graph.values[n.inputs[1]].weight is not None})
    graph.hints["q4_requires_conversion"] = sorted({graph.values[n.inputs[1]].weight.name
                                                   for n in graph.nodes if n.op == "linear_t"
                                                   and graph.values[n.inputs[1]].weight is not None})


def deduplicate_kernels(graph):
    graph.hints["kernel_signatures"] = sorted({repr((n.op, tuple(graph.values[k].shape for k in n.inputs),
                                                    graph.values[n.output].shape, sorted(n.attrs.items())))
                                               for n in graph.nodes})


from .patterns import (fuse_attention, fuse_conv_state, push_slice_into_conv, fuse_pointwise,  # noqa: E402
                       fuse_rotary_attention, fuse_reduction_prologue, fuse_state_update_reduction,
                       fuse_norms)

from .refusion import (refuse_additive_masks, refuse_rotary, refuse_linear, refuse_norms,  # noqa: E402
                       refuse_activations)

PASSES = (eliminate_dead_code, constant_fold, common_subexpressions, fold_views,
          refuse_additive_masks, refuse_rotary, refuse_linear, refuse_norms, refuse_activations,
          fuse_attention, fuse_rotary_attention, push_slice_into_conv, fuse_conv_state,
          fuse_pointwise, fuse_state_update_reduction, fuse_reduction_prologue, fuse_norms,
          select_q4, deduplicate_kernels)

# Backwards-compatible name for the pointwise fusion pass.
fuse_elementwise = fuse_pointwise


def optimize_graph(graph):
    for fn in PASSES:
        fn(graph)
        graph.passes.append(fn.__name__)
    eliminate_dead_code(graph)
    return graph
