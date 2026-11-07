"""Optional torch.fx adapter. Torch is imported only when ``from_torch`` is called.

``from_torch`` replays a ``torch.fx.GraphModule`` (or ``symbolic_trace(module)``)
on fir tensors. Parameters become checkpoint descriptors (shape and dtype only;
their payloads are never read); buffers (``inv_freq``, causal masks) are read as
constants. Targets resolve by name against ``formers.fir.torch_compat``:

* ``call_function``: Python operators, ``torch.*``, ``torch.nn.functional.*``
  and ``aten.*`` overloads (with aten argument orders for ``_softmax``,
  ``embedding``, ``slice``, ``rsub``, ``_to_copy``, ``div.Tensor_mode``,
  ``addmm`` and the view/copy aliases);
* ``call_method``: the tensor method of the same name;
* ``call_module``: Linear, Embedding, LayerNorm, RMSNorm, Conv1d, Dropout,
  Identity, Flatten, GELU and the parameter-free activation modules.

Every other target raises ``ValueError`` naming it.
"""

import operator

from . import nn
from .tensor import TensorSpec, ProxyTensor
from .tracer import trace


def _map(value, env):
    if isinstance(value, (tuple, list)):
        return type(value)(_map(v, env) for v in value)
    if isinstance(value, dict):
        return {k: _map(v, env) for k, v in value.items()}
    if isinstance(value, slice):
        return slice(_map(value.start, env), _map(value.stop, env), _map(value.step, env))
    if hasattr(value, "op") and hasattr(value, "target"):
        return env[value]
    return value


def _fir_dtype(torch_dtype):
    from . import dtypes
    return dtypes.dtype(str(torch_dtype).replace("torch.", "")) if torch_dtype is not None else None


def _aten_slice(x, dim=0, start=None, end=None, step=1):
    ix = [slice(None)] * x.ndim
    ix[dim] = slice(start, None if end is not None and end >= 2 ** 62 else end, step)
    return x[tuple(ix)]


def _aten_table(torch_compat):
    T, F = torch_compat, torch_compat.nn.functional

    def to_copy(x, dtype=None, **kwargs): return x.to(_fir_dtype(dtype)) if dtype is not None else x
    def rsub(a, b, alpha=1): return b - a * alpha
    def softmax(x, dim, half_to_float=False): return T.softmax(x, dim)
    def log_softmax(x, dim, half_to_float=False): return T.log_softmax(x, dim)
    def identity(x, *args, **kwargs): return x
    def dtype_kwargs(fn):
        def call(*args, dtype=None, layout=None, device=None, pin_memory=None, **kwargs):
            return fn(*args, dtype=_fir_dtype(dtype), **kwargs)
        return call

    return {
        "_softmax": softmax, "_log_softmax": log_softmax, "softmax": softmax, "log_softmax": log_softmax,
        "embedding": lambda weight, indices, padding_idx=-1, *a: F.embedding(indices, weight),
        "slice": _aten_slice, "rsub": rsub, "_to_copy": to_copy,
        "view": lambda x, shape: x.reshape(shape), "_unsafe_view": lambda x, shape: x.reshape(shape),
        "reshape": lambda x, shape: x.reshape(shape), "t": lambda x: x.t(),
        "expand": lambda x, size, implicit=False: x.expand(size),
        "clone": identity, "alias": identity, "detach": identity, "lift_fresh_copy": identity,
        "contiguous": identity, "dropout": lambda x, p=0.5, train=False: F.dropout(x, p, train),
        "native_dropout": None, "native_layer_norm": None,
        "addmm": lambda bias, a, b, beta=1, alpha=1: T.addmm(bias, a, b, beta=beta, alpha=alpha),
        "mean": lambda x, dim=None, keepdim=False, dtype=None: T.mean(x, dim, keepdim),
        "sum": lambda x, dim=None, keepdim=False, dtype=None: T.sum(x, dim, keepdim),
        "full": dtype_kwargs(T.full), "arange": dtype_kwargs(T.arange), "zeros": dtype_kwargs(T.zeros),
        "ones": dtype_kwargs(T.ones), "scalar_tensor": dtype_kwargs(T.tensor),
        "gelu": lambda x, approximate="none": F.gelu(x, approximate),
        "split_with_sizes": lambda x, sizes, dim=0: T.split(x, list(sizes), dim),
        "where": lambda c, a=None, b=None: T.where(c) if a is None else T.where(c, a, b),
        "scaled_dot_product_attention": F.scaled_dot_product_attention,
        "linear": F.linear, "silu": F.silu, "relu": F.relu,
    }


def _resolve_function(target, torch_compat, aten):
    if target is getattr:
        return getattr
    if getattr(operator, getattr(target, "__name__", ""), None) is target:
        return target
    text = str(target)
    if text.startswith("aten."):
        name = text.split(".")[1]
        fn = aten.get(name, getattr(torch_compat, name, None)) if name in aten else getattr(torch_compat, name, None)
        if fn is None: fn = getattr(torch_compat.nn.functional, name, None)
        if fn is None: raise ValueError("unsupported aten target %s; add a conversion in torch_adapter.py" % text)
        return fn
    name = getattr(target, "__name__", "")
    module = getattr(target, "__module__", "") or ""
    F = torch_compat.nn.functional
    spaces = (F, torch_compat) if ("functional" in module or "_nn" in module) else (torch_compat, F)
    for space in spaces:
        fn = getattr(space, name, None)
        if callable(fn) and not isinstance(fn, type): return fn
    raise ValueError("unsupported FX target %s.%s; add a conversion in torch_adapter.py" % (module, name or text))


def _leaf(kind, path, m, P, torch_compat):
    F = torch_compat.nn.functional
    tnn = torch_compat.nn
    def param(key): return P.get(path + "." + key) if getattr(m, key, None) is not None else None
    if kind == "Linear":
        return lambda x: F.linear(x, P[path + ".weight"], param("bias"))
    if kind == "Embedding":
        return lambda x: F.embedding(x, P[path + ".weight"])
    if kind == "LayerNorm":
        return lambda x: F.layer_norm(x, tuple(m.normalized_shape), param("weight"), param("bias"), m.eps)
    if kind == "RMSNorm":
        return lambda x: F.rms_norm(x, tuple(m.normalized_shape), param("weight"), m.eps)
    if kind == "Conv1d":
        if getattr(m, "padding_mode", "zeros") != "zeros": raise ValueError("Conv1d padding_mode %r" % m.padding_mode)
        return lambda x: F.conv1d(x, P[path + ".weight"], param("bias"), m.stride, m.padding, m.dilation, m.groups)
    if kind in ("Dropout", "Identity"):
        return lambda x, *a, **k: x
    if kind == "GELU":
        return lambda x: F.gelu(x, m.approximate)
    if kind == "Flatten":
        return lambda x: x.flatten(m.start_dim, m.end_dim)
    if kind == "Softmax":
        return lambda x: F.softmax(x, m.dim)
    simple = {"ReLU": F.relu, "SiLU": F.silu, "Tanh": F.tanh, "Sigmoid": F.sigmoid, "Mish": F.mish,
              "ReLU6": F.relu6, "Hardswish": F.hardswish, "Hardsigmoid": F.hardsigmoid, "LogSigmoid": F.logsigmoid,
              "Softsign": F.softsign, "Tanhshrink": F.tanhshrink, "SELU": F.selu}
    if kind in simple:
        return simple[kind]
    if kind == "LeakyReLU": return lambda x: F.leaky_relu(x, m.negative_slope)
    if kind == "ELU": return lambda x: F.elu(x, m.alpha)
    if kind == "Softplus": return lambda x: F.softplus(x, m.beta, m.threshold)
    if kind == "Hardtanh": return lambda x: F.hardtanh(x, m.min_val, m.max_val)
    return None


def from_torch(module, example_inputs, *, config=None, name="from_torch", optimize=True):
    """Trace an FX graph of ``module`` into a fir graph; unsupported targets fail by name."""
    try:
        import torch
        import torch.fx
    except ImportError as error:
        raise ImportError("fir.from_torch requires the optional torch package") from error
    from . import torch_compat
    fx = module if isinstance(module, torch.fx.GraphModule) else torch.fx.symbolic_trace(module)
    if config is None and hasattr(getattr(module, "config", None), "to_dict"):
        config = module.config.to_dict()
    aten = _aten_table(torch_compat)

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.source_parameters = {k: nn.Parameter(tuple(p.shape), _fir_dtype(p.dtype) or "f32", name=k)
                                      for k, p in fx.named_parameters()}
            self.source_buffers = dict(fx.named_buffers())
            self.leaves = {}
            for path, m in fx.named_modules():
                if path:
                    leaf = _leaf(type(m).__name__, path, m, self.source_parameters, torch_compat)
                    if leaf is not None: self.leaves[path] = leaf

        def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
            yield from self.source_parameters.items()

        def attribute(self, target):
            if target in self.source_parameters: return self.source_parameters[target]
            if target in self.source_buffers:
                b = self.source_buffers[target]
                return torch_compat.tensor(b.detach().cpu().tolist(), dtype=_fir_dtype(b.dtype))
            value = fx
            for part in target.split("."): value = getattr(value, part)
            if hasattr(value, "tolist") and hasattr(value, "dtype"):
                return torch_compat.tensor(value.detach().cpu().tolist(), dtype=_fir_dtype(value.dtype))
            return value

        def forward(self, *inputs):
            env, inputs = {}, iter(inputs)
            for n in fx.graph.nodes:
                args, kwargs = _map(n.args, env), _map(n.kwargs, env)
                if n.op == "placeholder":
                    result = next(inputs)
                elif n.op == "get_attr":
                    result = self.attribute(n.target)
                elif n.op == "call_module":
                    if n.target not in self.leaves:
                        raise ValueError("unsupported FX module %s (%s)" % (n.target, type(fx.get_submodule(n.target)).__name__
                                                                              if hasattr(fx, "get_submodule") else "?"))
                    result = self.leaves[n.target](*args, **kwargs)
                elif n.op == "call_method":
                    if "dtype" in kwargs: kwargs["dtype"] = _fir_dtype(kwargs["dtype"])
                    args = [_fir_dtype(a) if "dtype" in type(a).__name__ else a for a in args]
                    if not isinstance(args[0], ProxyTensor) and not hasattr(args[0], n.target):
                        raise ValueError("unsupported FX method %s on %s" % (n.target, type(args[0]).__name__))
                    result = getattr(args[0], n.target)(*args[1:], **kwargs)
                elif n.op == "call_function":
                    fn = _resolve_function(n.target, torch_compat, aten)
                    if "dtype" in kwargs: kwargs["dtype"] = _fir_dtype(kwargs["dtype"])
                    result = fn(*args, **kwargs)
                elif n.op == "output":
                    return args[0]
                else:
                    raise ValueError("unsupported FX node kind: " + n.op)
                env[n] = result

    specs = tuple(x if isinstance(x, TensorSpec) else TensorSpec(tuple(x.shape)) for x in example_inputs)
    return trace(Adapter(), specs, config=config, name=name, optimize=optimize)
