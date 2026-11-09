"""Registry for operations that live outside this package.

A front end (for example ``nfgraph2torch``) can add an operation that the built-in tracer does not
know: it registers a Python reference (used by ``evaluate``, constant folding and tests) and a
micro-kernel generator (used by ``emit``), then records nodes with ``custom_op``::

    from formers import fir
    fir.register_op("prod", reference=ref, lower=gen)
    y = fir.custom_op("prod", (x,), shape=(4,), attrs={"dim": 1})

``reference(inputs, shapes, out_shape, attrs)`` takes flat lists and returns the flat output.
``lower(emission, node, ctx, kernel)`` writes the kernel with the helpers of ``codegen`` and returns
``(family, threads)``; ``family`` names the kernel in the emitted FIR.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CustomOp:
    name: str
    reference: object
    lower: object


OPS = {}

# Scalar tree operations added by a front end: ``EXT_TREE_OPS[kind](kernel, args) -> register``.
# ``args`` are the registers of the operands (all owned by the callee, which may modify args[0] in
# place and must return the result register); the kernel frees args[1:] afterwards.
EXT_TREE_OPS = {}


def register_tree_op(kind, emit):
    """Teach the pointwise kernel builder a new scalar operation (used inside ``custom`` lowerings)."""
    EXT_TREE_OPS[kind] = emit


def register_op(name, *, reference, lower, replace=False):
    if name in OPS and not replace:
        raise ValueError("operation already registered: " + name)
    OPS[name] = CustomOp(name, reference, lower)


def custom_op(name, inputs, shape, attrs=None, dtype="f32"):
    """Record ``name`` on the active trace and return the result proxy."""
    if name not in OPS:
        raise KeyError("unregistered custom operation: " + name)
    if not inputs:
        raise ValueError("a custom operation needs at least one tensor input")
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    from .tensor import _proxy, _trace_of
    trace = _trace_of(*inputs)
    proxies = [_proxy(t, trace) for t in inputs]      # Parameters become checkpoint weights
    return trace.record(name, tuple(t.name for t in proxies), shape, dict(attrs or {}), dtype=dtype)


def run_reference(node, graph, inputs, shapes):
    op = OPS[node.op]
    out_shape = graph.values[node.output].shape
    result = op.reference(inputs, shapes, out_shape, node.attrs)
    if len(result) != math.prod(out_shape):
        raise ValueError("%s reference returned %d values for shape %s" % (node.op, len(result), out_shape))
    return {node.output: [float(v) for v in result]}
