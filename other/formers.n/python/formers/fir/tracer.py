"""Metadata tracing of static Python module control flow into the shared SSA IR."""

from contextvars import ContextVar
from dataclasses import replace
import inspect
import math
import re
from .graph import Graph, Node, Value, Weight, optimize_graph
from .tensor import ProxyTensor, TensorSpec, Dim, Unsupported
from .nn import Module, Parameter, Embedding, Linear, LayerNorm, GELU
from . import dtypes as D

_ACTIVE = ContextVar("fir_trace", default=None)


def active_trace(): return _ACTIVE.get()


def current_trace():
    t = active_trace()
    if t is None: raise RuntimeError("tensor operations require an active fir.trace")
    return t


class _Trace:
    eager = False

    def __init__(self, graph, model):
        self.graph, self.scope = graph, ""
        self.imported = {}
        self.weights, self.parameter_names, self.module_names, self.state_names = {}, {}, {}, {}
        self.state_versions = {}
        self.state_storage = {}
        if isinstance(model, Module):
            self.parameter_names = {id(p): p.name or k for k, p in model.named_parameters()}
            self.module_names = {id(m): k for k, m in model.named_modules()}
            for path, m in model.named_modules():
                for key, state in m._states.items():
                    self.state_names[id(state)] = path + ("." if path else "") + key

    def parameter(self, p):
        name = p.name or self.parameter_names.get(id(p))
        if not name: raise ValueError("unregistered Parameter: assign it to a Module attribute")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name): raise ValueError("invalid checkpoint tensor name: " + name)
        descriptor = Weight(name, p.shape, p.dtype)
        if name in self.weights:
            key = self.weights[name]
            if self.graph.values[key].weight != descriptor: raise ValueError("conflicting descriptor for checkpoint tensor " + name)
        else:
            key = "w" + str(len(self.weights))
            self.weights[name] = key
            self.graph.values[key] = Value(math.prod(p.shape), weight=descriptor, shape=p.shape, dtype=p.dtype)
        return ProxyTensor(self, key)

    def state(self, state):
        if id(state) not in self.state_names: raise ValueError("state must be registered on the traced module")
        if id(state) in self.state_versions: return ProxyTensor(self, self.state_versions[id(state)])
        name = "state_" + str(len([v for v in self.graph.values.values() if v.state]))
        shape = self.resolve(state.shape)
        self.graph.values[name] = Value(math.prod(shape), shape=shape, state=self.state_names[id(state)])
        self.state_storage[id(state)] = name
        result = self.record("state_read", (name,), shape)
        self.state_versions[id(state)] = result.name
        return result

    def resolve(self, shape):
        out = []
        for dim in shape:
            if isinstance(dim, Dim):
                key, value = dim.name, dim.value
            elif isinstance(dim, str): key, value = dim, None
            else: key, value = "", dim
            if key:
                value = value if value is not None else self.graph.symbols.get(key, self.graph.config.get(key))
                if value is None: raise ValueError("bind symbolic dimension %r in config or Dim(name, value)" % key)
                if key in self.graph.config and (type(self.graph.config[key]) is not int or self.graph.config[key] != value):
                    raise ValueError("symbolic dimension conflicts with config: " + key)
                if key in self.graph.symbols and self.graph.symbols[key] != value: raise ValueError("conflicting symbolic dimension: " + key)
                self.graph.symbols[key] = value
            if type(value) is not int or value <= 0: raise ValueError("dimensions must specialize to positive integers")
            out.append(value)
        return tuple(out)

    def constant(self, data, dtype="f32"):
        def flatten(x):
            if isinstance(x, (tuple, list)):
                if not x: raise ValueError("empty constants are not supported")
                children = [flatten(y) for y in x]
                if any(s != children[0][0] for s, _ in children): raise ValueError("ragged constant")
                return (len(x),) + children[0][0], [v for _, ys in children for v in ys]
            if not isinstance(x, (int, float, bool)) or not math.isfinite(x): raise ValueError("constants must be finite numbers")
            return (), [float(x)]
        shape, values = flatten(data)
        return self.constant_flat(values, shape, dtype)

    def constant_flat(self, values, shape, dtype="f32"):
        key = "c" + str(len(self.graph.values))
        self.graph.values[key] = Value(len(values), shape=shape, dtype=dtype, constant=tuple(values))
        return ProxyTensor(self, key)

    def import_constant(self, x):
        """A constant of this trace holding the values of an eager (construction-time) tensor."""
        key = (id(x._trace), x.name)
        if key not in self.imported:
            from .tensor import static_values
            v = x._trace.graph.values[x.name]
            proxy = self.constant_flat(static_values(x), v.shape, v.dtype)
            self.imported[key] = (x._trace, proxy.name)
        return ProxyTensor(self, self.imported[key][1], x._dtype)

    def record(self, op, inputs, shape, attrs=None, dtype="f32", writes=()):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        name = "v" + str(len(self.graph.nodes))
        self.graph.values[name] = Value(math.prod(shape), shape=shape, dtype=dtype)
        self.graph.nodes.append(Node(op, tuple(inputs), name, attrs or {}, self.scope, writes=tuple(writes)))
        return ProxyTensor(self, name)


class EagerTrace(_Trace):
    """Construction-time evaluation: every operation is folded into a constant at once."""
    eager = True

    def __init__(self):
        super().__init__(Graph("eager", {}), None)
        self.counter = 0

    def parameter(self, p):
        raise RuntimeError("Parameter values are read only inside fir.trace (checkpoint payloads are never loaded)")

    def state(self, state):
        raise RuntimeError("states are read and written only inside fir.trace")

    def record(self, op, inputs, shape, attrs=None, dtype="f32", writes=()):
        if writes: raise RuntimeError("state writes need an active fir.trace")
        from .reference import run_node
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.counter += 1
        name = "e" + str(self.counter)
        self.graph.values[name] = Value(math.prod(shape), shape=shape, dtype=dtype)
        node = Node(op, tuple(inputs), name, attrs or {})
        values = {k: list(self.graph.values[k].constant) for k in node.inputs}
        result = run_node(node, self.graph, values)[name]
        self.graph.values[name] = replace(self.graph.values[name], constant=tuple(float(v) for v in result))
        return ProxyTensor(self, name)


def _graph_config(config):
    if config is None: return {}
    if hasattr(config, "to_dict"): config = config.to_dict()
    return dict(config)


def _outputs(result, trace):
    """Proxies returned by forward, in order: tensors, tuples/lists, dicts and ModelOutput-like objects."""
    found = []
    def walk(x, depth=0):
        if depth > 8: return
        if isinstance(x, ProxyTensor):
            found.append(x)
        elif isinstance(x, (tuple, list)):
            for y in x: walk(y, depth + 1)
        elif isinstance(x, dict):
            for y in x.values(): walk(y, depth + 1)
        elif hasattr(x, "__dict__") and not isinstance(x, (Module, type)) and type(x).__module__ != "builtins":
            for y in vars(x).values():
                if isinstance(y, (ProxyTensor, tuple, list, dict)): walk(y, depth + 1)
    walk(result)
    names = []
    for x in found:
        if x._trace is not trace:
            if not x._trace.eager: raise TypeError("forward must return one proxy from this trace")
            x = trace.import_constant(x)
        if x.name not in names: names.append(x.name)
    return names


def trace(model, example_inputs=(), *, config=None, name="scripted", optimize=True):
    """Trace one specialization of ``model(*example_inputs)``.

    ``example_inputs`` is a sequence (positional arguments) or a mapping (keyword
    arguments). ``TensorSpec`` entries become graph inputs (keys: the role, the
    keyword name or ``input<i>``); other entries (ints, flags, ``None``, config
    objects) are passed to forward unchanged. Forward may return a tensor, or a
    tuple, list, dict or ModelOutput-like object of tensors: the first tensor is
    ``graph.output`` and all of them are ``graph.outputs``.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name): raise ValueError("model name must be an identifier")
    config = _graph_config(config)
    eps_key = next((k for k in ("layer_norm_epsilon", "rms_norm_eps", "layer_norm_eps", "norm_epsilon", "eps") if k in config), None)
    eps = config[eps_key] if eps_key else config.setdefault("layer_norm_epsilon", 1e-5)
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or not 1e-38 <= eps <= 3e38:
        raise ValueError("the normalization epsilon must be positive and representable in f32")
    graph = Graph(name, config)
    t = _Trace(graph, model)
    if isinstance(example_inputs, TensorSpec): example_inputs = (example_inputs,)
    keyword = isinstance(example_inputs, dict)
    entries = list(example_inputs.items()) if keyword else list(enumerate(example_inputs))
    if not entries and not keyword:
        raise TypeError("example inputs must be TensorSpecs (a sequence or a mapping of forward arguments)")
    args, kwargs = [], {}
    for i, spec in entries:
        if isinstance(spec, TensorSpec):
            key = spec.role or (i if keyword else "input" + str(i))
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key): raise ValueError("input names must be identifiers")
            if key in graph.values: raise ValueError("duplicate input role")
            shape = t.resolve(spec.shape)
            graph.values[key] = Value(math.prod(shape), role=spec.role or "argument", shape=shape, dtype=spec.dtype)
            value = ProxyTensor(t, key)
        elif not keyword and not isinstance(spec, (int, float, bool, str, type(None))) and not hasattr(spec, "__dict__"):
            raise TypeError("example inputs must be TensorSpecs or static Python values")
        else:
            value = spec
        if keyword: kwargs[i] = value
        else: args.append(value)
    if not any(v.role for v in graph.values.values()):
        raise TypeError("example inputs must include at least one TensorSpec")
    token = _ACTIVE.set(t)
    try: result = model(*args, **kwargs)
    finally: _ACTIVE.reset(token)
    names = _outputs(result, t)
    if not names: raise TypeError("forward must return one proxy from this trace")
    graph.output, graph.outputs = names[0], tuple(names)
    if optimize: optimize_graph(graph)
    return graph


def script(model, example_inputs, **kwargs):
    """Trace one specialization, including loops and branches over static values."""
    return trace(model, example_inputs, **kwargs)
