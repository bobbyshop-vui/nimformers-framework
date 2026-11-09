"""torch module -> formers.fir graph -> model.fir.

1. ``torch.export`` captures the module as an ATen graph, then ``run_decompositions`` lowers it to
   Core ATen (about 160 ops; hundreds of higher-level torch ops disappear here).
2. Each node is replayed on the ``formers.fir`` tracer through ``aten.handler``. ATen ops that FIR
   already has become ordinary FIR kernels; the rest become micro-kernels (``micro.py``).
3. ``fir.export`` writes ``model.fir`` and ``config.json``.

Parameters become checkpoint descriptors named like the module's ``state_dict`` keys; their values
are never read. Buffers and constants are embedded as FIR constants.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Optional

from formers import fir
from formers.fir import nn as fnn
from formers.fir import torch_compat as T
from formers.fir.tensor import ProxyTensor, TensorSpec

from . import aten


class UnsupportedOp(ValueError):
    """An ATen operation with no FIR kernel and no micro-kernel; names the op and the node."""


@dataclass(frozen=True)
class Input:
    """Describes one user input of the torch module.

    ``role`` is ``"token"`` or ``"position"`` for the two inputs the FIR runner feeds, or ``""``
    for a plain tensor. ``dtype`` overrides the dtype taken from the example tensor.
    """
    role: str = ""
    dtype: Optional[str] = None


def _fir_dtype(torch_dtype):
    from formers.fir import dtypes
    return dtypes.dtype(str(torch_dtype).replace("torch.", "")).fir if torch_dtype is not None else None


def capture(module, example_inputs, *, kwargs=None, strict=False, decompositions=None):
    """``torch.export`` + decomposition to Core ATen. Returns the ExportedProgram."""
    import torch
    module = module.eval()
    with torch.no_grad():
        program = torch.export.export(module, tuple(example_inputs), kwargs=kwargs or None, strict=strict)
        program = program.run_decompositions(decompositions) if decompositions is not None \
            else program.run_decompositions()
    return program


def _constants(program):
    """Buffers and constant tensors of the program by placeholder name."""
    import torch
    values = {}
    sig = program.graph_signature
    for spec in sig.input_specs:
        kind = spec.kind.name
        if kind == "BUFFER":
            values[spec.arg.name] = program.state_dict[spec.target]
        elif kind == "CONSTANT_TENSOR":
            values[spec.arg.name] = program.constants[spec.target]
    return values


def _tensor_constant(value):
    return T.tensor(value.detach().cpu().tolist(), dtype=_fir_dtype(value.dtype))


def _map(value, env):
    if isinstance(value, (tuple, list)):
        return type(value)(_map(v, env) for v in value)
    if isinstance(value, dict):
        return {k: _map(v, env) for k, v in value.items()}
    if hasattr(value, "op") and hasattr(value, "target"):
        return env[value]
    return value


def _adapter(program, specs):
    import torch
    sig = program.graph_signature
    consts = _constants(program)
    params, by_target = {}, {}
    for spec in sig.input_specs:
        if spec.kind.name == "PARAMETER":
            t = program.state_dict[spec.target]
            p = fnn.Parameter(tuple(t.shape), _fir_dtype(t.dtype) or "f32", name=spec.target)
            params[spec.arg.name] = by_target[spec.target] = p

    class Adapter(fnn.Module):
        def __init__(self):
            super().__init__()

        def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
            yield from by_target.items()

        def forward(self, *inputs):
            env, user = {}, iter(inputs)
            user_names = [s.arg.name for s in sig.input_specs if s.kind.name == "USER_INPUT"]
            for node in program.graph.nodes:
                if node.op == "placeholder":
                    if node.name in params:
                        env[node] = params[node.name]
                    elif node.name in consts:
                        env[node] = _tensor_constant(consts[node.name])
                    elif node.name in user_names:
                        env[node] = next(user)
                    else:
                        raise UnsupportedOp("unrecognised placeholder %s" % node.name)
                elif node.op == "call_function":
                    args, kwargs = _map(node.args, env), _map(node.kwargs, env)
                    try:
                        env[node] = aten.replay(node.target, args, kwargs)
                    except UnsupportedOp:
                        raise
                    except Exception as error:
                        raise UnsupportedOp("%s (node %s): %s: %s" % (
                            node.target, node.name, type(error).__name__, error)) from error
                elif node.op == "output":
                    return _map(node.args[0], env)
                else:
                    raise UnsupportedOp("unsupported graph node kind: %s" % node.op)
    return Adapter()


def _specs(program, example_inputs, inputs):
    sig = program.graph_signature
    names = [s.arg.name for s in sig.input_specs if s.kind.name == "USER_INPUT"]
    inputs = list(inputs) if inputs is not None else [Input()] * len(names)
    if len(inputs) != len(names):
        raise ValueError("%d Input descriptions for %d user inputs (%s)" % (len(inputs), len(names), names))
    specs = []
    for name, example, desc in zip(names, example_inputs, inputs):
        if isinstance(example, TensorSpec):
            specs.append(example)
            continue
        dtype = desc.dtype or _fir_dtype(example.dtype) or "f32"
        specs.append(TensorSpec(tuple(example.shape), desc.role, dtype))
    return tuple(specs)


def from_torch(module, example_inputs, *, inputs=None, kwargs=None, config=None, name="from_torch",
               optimize=True, strict=False, decompositions=None):
    """Convert a torch module into a ``formers.fir`` graph.

    ``example_inputs`` are the tensors passed to ``module(*example_inputs)``; ``inputs`` is an
    optional list of :class:`Input` (role/dtype per user input). ``config`` is a dict or an object
    with ``to_dict()`` and is written to ``config.json`` by :func:`export`.
    """
    if config is None and hasattr(getattr(module, "config", None), "to_dict"):
        config = module.config.to_dict()
    program = capture(module, example_inputs, kwargs=kwargs, strict=strict, decompositions=decompositions)
    specs = _specs(program, example_inputs, inputs)
    return fir.trace(_adapter(program, specs), specs, config=config, name=name, optimize=optimize)


def export(module, example_inputs, directory, *, report=False, **options):
    """``from_torch`` + write ``model.fir`` and ``config.json``; returns the path of ``model.fir``."""
    graph = from_torch(module, example_inputs, **options)
    return fir.export(graph, directory, report=report)


def evaluate(graph, module, example_inputs):
    """Run the converted graph with the Python reference evaluator on ``example_inputs``.

    Weights come from ``module.state_dict()``. Returns ``(flat_values, shape)`` of the first output;
    compare it with ``module(*example_inputs)`` to check a conversion.
    """
    parameters = {k: v.detach().flatten().tolist() for k, v in module.state_dict().items()}
    inputs = {"input%d" % i: x.detach().flatten().tolist() for i, x in enumerate(example_inputs)}
    values = fir.evaluate(graph, inputs, parameters)
    return values, tuple(graph.values[graph.output].shape)
