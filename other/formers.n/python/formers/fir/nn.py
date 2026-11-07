"""Torch-style metadata modules with checkpoint-compatible parameter names."""

from collections import OrderedDict, namedtuple
import math
from .tensor import _trace_of, _proxy, ProxyTensor, Size, CPU, Uninitialized, Unsupported
from . import dtypes as D


class Parameter:
    """A checkpoint tensor described by shape and storage dtype; payloads are never read.

    ``Parameter((out, in))`` gives the shape directly; ``Parameter(tensor)`` (the
    torch form, e.g. ``Parameter(torch.ones(n))``) takes the shape of the tensor
    and ignores its values, because parameter values always come from the checkpoint.
    """

    requires_grad = True
    grad = None

    def __init__(self, shape=None, dtype="f32", name=None, requires_grad=True):
        if isinstance(dtype, bool):
            dtype, requires_grad = "f32", dtype
        if shape is not None and not isinstance(shape, (tuple, list)) and hasattr(shape, "shape"):
            shape = tuple(shape.shape)
        self.shape = Size(tuple(shape) if shape is not None else ())
        if not self.shape or any(type(d) is not int or d <= 0 for d in self.shape):
            raise ValueError("parameter dimensions must be positive integers")
        dtype = "f32" if dtype in (None, D.float32, D.float16, D.bfloat16) else dtype
        if dtype not in ("f32", "q4"): raise ValueError("parameter dtype must be f32 or q4")
        self.dtype, self.name = dtype, name
        self.requires_grad = requires_grad

    def _tensor(self): return _trace_of().parameter(self)
    ndim = property(lambda self: len(self.shape))
    device = property(lambda self: CPU)
    data = property(lambda self: self)
    def size(self, dim=None): return self.shape if dim is None else self.shape[dim]
    def dim(self): return len(self.shape)
    def numel(self): return math.prod(self.shape)
    def element_size(self): return 4
    def is_floating_point(self): return True
    def requires_grad_(self, requires_grad=True): self.requires_grad = requires_grad; return self
    # Initializers are no-ops: values come from the checkpoint.
    def normal_(self, *args, **kwargs): return self
    uniform_ = zero_ = fill_ = copy_ = normal_
    def __repr__(self): return "Parameter(shape=%s, dtype=%s%s)" % (list(self.shape), self.dtype, ", name=%r" % self.name if self.name else "")
    def __matmul__(self, x): return self._tensor() @ x
    def __rmatmul__(self, x): return x @ self._tensor()
    def __add__(self, x): return self._tensor() + x
    def __radd__(self, x): return x + self._tensor()
    def __sub__(self, x): return self._tensor() - x
    def __mul__(self, x): return self._tensor() * x
    def __rmul__(self, x): return x * self._tensor()
    def __truediv__(self, x): return self._tensor() / x
    def __rsub__(self, x): return x - self._tensor()
    def __rtruediv__(self, x): return x / self._tensor()
    def __pow__(self, x): return self._tensor() ** x
    def __neg__(self): return -self._tensor()
    def __getitem__(self, ix): return self._tensor()[ix]
    def reshape(self, *shape): return self._tensor().reshape(*shape)
    view = reshape
    def transpose(self, a, b): return self._tensor().transpose(a, b)
    def permute(self, *dims): return self._tensor().permute(*dims)
    def chunk(self, chunks, dim=0): return self._tensor().chunk(chunks, dim)
    def split(self, sizes, dim=0): return self._tensor().split(sizes, dim)
    @property
    def T(self): return self._tensor().T

    def __getattr__(self, name):
        # Any other tensor method (exp, softplus, sum, unsqueeze, ...) applies to
        # the parameter's traced value; outside a trace it is a missing attribute.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            tensor = self._tensor()
        except Exception:
            raise AttributeError(name) from None
        return getattr(tensor, name)


class State:
    def __init__(self, shape): self.shape = tuple(shape)
    def read(self): return _trace_of().state(self)
    def update(self, value, index=None, dim=0):
        t = _trace_of(value, index)
        state = t.state(self)
        value = _proxy(value, t)
        attrs = {"dim": dim % state.ndim}
        inputs = [state.name, value.name]
        if index is None:
            if tuple(value.shape) != tuple(state.shape): raise ValueError("state update shape mismatch")
        else:
            index = _proxy(index, t)
            dim %= state.ndim
            expected = tuple(state.shape[:dim] + state.shape[dim+1:])
            if tuple(value.shape) != expected or index.numel() != 1:
                raise ValueError("indexed state update requires one complete state slice and scalar index")
            inputs.append(index.name)
        out = t.record("state_write", inputs, state.shape, attrs, writes=(t.state_storage[id(self)],))
        # Fresh SSA snapshots preserve read-before-write and write-before-read semantics.
        t.state_versions[id(self)] = out.name
        return out


IncompatibleKeys = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])


class Buffer:
    """torch.nn.Buffer: assigning one to a module attribute registers a buffer."""
    def __init__(self, data=None, *, persistent=True):
        self.data, self.persistent = data, persistent


def _is_buffer_value(value):
    return value is None or isinstance(value, (ProxyTensor, Uninitialized))


class Module:
    """torch.nn.Module surface: registries, buffers, containers traversal, train/eval flags.

    Modules start in eval mode (``training`` is False) because fir traces inference;
    ``train()`` sets the flag for code that branches on it.
    """

    dump_patches = False

    def __init__(self, *args, **kwargs):
        self._ensure_registry()

    def _ensure_registry(self):
        if "_parameters" not in self.__dict__:
            object.__setattr__(self, "_parameters", OrderedDict())
            object.__setattr__(self, "_modules", OrderedDict())
            object.__setattr__(self, "_states", OrderedDict())
            object.__setattr__(self, "_buffers", OrderedDict())
            object.__setattr__(self, "_non_persistent", set())
            object.__setattr__(self, "training", False)

    def __setattr__(self, name, value):
        self._ensure_registry()
        if isinstance(value, Buffer):
            self.register_buffer(name, value.data, persistent=value.persistent)
            return
        if name in self._buffers and _is_buffer_value(value):
            self._buffers[name] = value
            object.__setattr__(self, name, value)
            return
        for registry in (self._parameters, self._modules, self._states, self._buffers):
            registry.pop(name, None)
        self._non_persistent.discard(name)
        if isinstance(value, Parameter): self._parameters[name] = value
        elif isinstance(value, Module): self._modules[name] = value
        elif isinstance(value, State): self._states[name] = value
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        self._ensure_registry()
        for registry in (self._parameters, self._modules, self._states, self._buffers):
            registry.pop(name, None)
        self._non_persistent.discard(name)
        object.__delattr__(self, name)

    # ---- registration ----------------------------------------------------------
    def register_buffer(self, name, tensor, persistent=True):
        """Constant tensor attribute (``inv_freq``, masks). Traced reads see its value as a graph constant."""
        self._ensure_registry()
        if not isinstance(name, str) or not name or "." in name: raise KeyError("buffer name must be a non-empty string without dots")
        if name in self.__dict__ and name not in self._buffers: raise KeyError("attribute %r already exists" % name)
        for registry in (self._parameters, self._modules, self._states): registry.pop(name, None)
        if not _is_buffer_value(tensor):
            raise TypeError("buffer %r must be a tensor or None (got %s)" % (name, type(tensor).__name__))
        self._buffers[name] = tensor
        if persistent: self._non_persistent.discard(name)
        else: self._non_persistent.add(name)
        object.__setattr__(self, name, tensor)

    def register_parameter(self, name, param):
        if param is None:
            object.__setattr__(self, name, None)
            return
        setattr(self, name, param)

    def add_module(self, name, module):
        if module is None:
            object.__setattr__(self, name, None)
            return
        setattr(self, name, module)

    register_module = add_module

    def register_state(self, name, shape):
        if not name.isidentifier() or hasattr(self, name): raise ValueError("state needs a fresh attribute name")
        state = State(shape)
        setattr(self, name, state)
        return state

    # ---- traversal ----------------------------------------------------------------
    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        self._ensure_registry()
        if isinstance(memo, str): memo, prefix = None, memo
        memo = set() if memo is None else memo
        if remove_duplicate:
            if id(self) in memo: return
            memo.add(id(self))
        yield prefix, self
        for key, child in self._modules.items():
            if child is None: continue
            yield from child.named_modules(memo, prefix + ("." if prefix else "") + key, remove_duplicate)

    def modules(self): return (m for _, m in self.named_modules())

    def named_children(self):
        self._ensure_registry()
        return ((k, m) for k, m in self._modules.items() if m is not None)

    def children(self): return (m for _, m in self.named_children())

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
        for path, module in modules:
            for key, value in module._parameters.items():
                if not remove_duplicate or id(value) not in seen:
                    seen.add(id(value))
                    yield path + ("." if path else "") + key, value

    def parameters(self, recurse=True):
        return (p for _, p in self.named_parameters(recurse=recurse))

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
        for path, module in modules:
            for key, value in module._buffers.items():
                if value is None or (remove_duplicate and id(value) in seen): continue
                seen.add(id(value))
                yield path + ("." if path else "") + key, value

    def buffers(self, recurse=True):
        return (b for _, b in self.named_buffers(recurse=recurse))

    def get_submodule(self, target):
        module = self
        for part in target.split(".") if target else []:
            module = module._modules.get(part) if isinstance(module, Module) else None
            if module is None: raise AttributeError("no submodule %r" % target)
        return module

    def get_parameter(self, target):
        path, _, name = target.rpartition(".")
        module = self.get_submodule(path)
        if name not in module._parameters: raise AttributeError("no parameter %r" % target)
        return module._parameters[name]

    def get_buffer(self, target):
        path, _, name = target.rpartition(".")
        module = self.get_submodule(path)
        if name not in module._buffers: raise AttributeError("no buffer %r" % target)
        return module._buffers[name]

    # ---- state dict (metadata) -------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Parameters (shape descriptors) and persistent buffers, keyed like torch."""
        out = OrderedDict() if destination is None else destination
        for path, module in self.named_modules(prefix=prefix.rstrip("."), remove_duplicate=False):
            dot = path + ("." if path else "")
            for key, value in module._parameters.items():
                out[dot + key] = value
            for key, value in module._buffers.items():
                if value is not None and key not in module._non_persistent:
                    out[dot + key] = value
        return out

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Check names and shapes against a state dict (tensors, shape tuples or objects with ``shape``).

        Parameter payloads are never read. Persistent buffers take the given values
        when they are fir tensors or nested Python lists.
        """
        expected = self.state_dict()
        missing = [k for k in expected if k not in state_dict]
        unexpected = [k for k in state_dict if k not in expected]
        errors = []
        for key, target in expected.items():
            if key not in state_dict: continue
            given = state_dict[key]
            shape = tuple(given) if isinstance(given, (tuple, Size)) and all(isinstance(d, int) for d in given) \
                else tuple(getattr(given, "shape", ()) or _list_shape(given))
            if tuple(target.shape) != shape:
                errors.append("size mismatch for %s: checkpoint %s, model %s" % (key, list(shape), list(target.shape)))
        if strict and (missing or unexpected or errors):
            parts = errors + (["missing keys: " + ", ".join(missing)] if missing else []) + \
                    (["unexpected keys: " + ", ".join(unexpected)] if unexpected else [])
            raise RuntimeError("Error(s) in loading state_dict:\n  " + "\n  ".join(parts))
        if errors: raise RuntimeError("Error(s) in loading state_dict:\n  " + "\n  ".join(errors))
        from .tensor import tensor as make_tensor
        for path, module in self.named_modules(remove_duplicate=False):
            dot = path + ("." if path else "")
            for key in list(module._buffers):
                value = state_dict.get(dot + key)
                if key in module._non_persistent or value is None: continue
                if isinstance(value, list): value = make_tensor(value, dtype=module._buffers[key].dtype)
                if isinstance(value, ProxyTensor) and value._trace.eager: setattr(module, key, value)
        return IncompatibleKeys(missing, unexpected)

    # ---- modes and no-op device/dtype moves ----------------------------------------------
    def train(self, mode=True):
        for m in self.modules(): object.__setattr__(m, "training", bool(mode))
        return self

    def eval(self): return self.train(False)

    def apply(self, fn):
        for child in self.children(): child.apply(fn)
        fn(self)
        return self

    def requires_grad_(self, requires_grad=True):
        for p in self.parameters(): p.requires_grad = requires_grad
        return self

    def zero_grad(self, set_to_none=True): pass
    def to(self, *args, **kwargs): return self
    def float(self): return self
    def half(self): return self
    def bfloat16(self): return self
    def double(self): return self
    def cuda(self, device=None): return self
    def cpu(self): return self
    def type(self, dst_type): return self
    def extra_repr(self): return ""
    def share_memory(self): return self

    @property
    def device(self): return CPU

    @property
    def dtype(self): return D.float32

    def __call__(self, *args, **kwargs):
        from .tracer import active_trace
        t = active_trace()
        if t is None: return self.forward(*args, **kwargs)
        old = t.scope
        t.scope = t.module_names.get(id(self), old)
        try:
            return self.forward(*args, **kwargs)
        finally:
            t.scope = old

    def forward(self, *args, **kwargs): raise NotImplementedError


def _list_shape(x):
    shape = []
    while isinstance(x, list):
        shape.append(len(x))
        x = x[0] if x else None
    return tuple(shape)


class ModuleList(Module):
    def __init__(self, modules=()):
        super().__init__()
        for m in modules: self.append(m)
    def append(self, module):
        if not isinstance(module, Module): raise TypeError("ModuleList entries must be modules")
        setattr(self, str(len(self._modules)), module)
        return self
    def extend(self, modules):
        for m in modules: self.append(m)
        return self
    def insert(self, index, module):
        items = list(self._modules.values())
        items.insert(index, module)
        for k in list(self._modules): delattr(self, k)
        for m in items: self.append(m)
    def __len__(self): return len(self._modules)
    def __iter__(self): return iter(self._modules.values())
    def __getitem__(self, index):
        items = list(self._modules.values())
        return ModuleList(items[index]) if isinstance(index, slice) else items[index]
    def __setitem__(self, index, module):
        key = list(self._modules)[index]
        setattr(self, key, module)
    def __iadd__(self, modules): return self.extend(modules)


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None: self.update(modules)
    def __getitem__(self, key): return self._modules[key]
    def __setitem__(self, key, module): setattr(self, key, module)
    def __delitem__(self, key): delattr(self, key)
    def __contains__(self, key): return key in self._modules
    def __len__(self): return len(self._modules)
    def __iter__(self): return iter(self._modules)
    def keys(self): return self._modules.keys()
    def values(self): return self._modules.values()
    def items(self): return self._modules.items()
    def update(self, modules):
        for k, m in (modules.items() if hasattr(modules, "items") else modules): self[k] = m
    def pop(self, key):
        m = self._modules[key]
        delattr(self, key)
        return m


class ParameterList(Module):
    def __init__(self, values=None):
        super().__init__()
        for p in values or (): self.append(p)
    def append(self, value):
        setattr(self, str(len(self._parameters)), value if isinstance(value, Parameter) else Parameter(value))
        return self
    def extend(self, values):
        for v in values: self.append(v)
        return self
    def __len__(self): return len(self._parameters)
    def __iter__(self): return iter(self._parameters.values())
    def __getitem__(self, index): return list(self._parameters.values())[index]


class ParameterDict(Module):
    def __init__(self, parameters=None):
        super().__init__()
        for k, v in (parameters or {}).items(): self[k] = v
    def __getitem__(self, key): return self._parameters[key]
    def __setitem__(self, key, value): setattr(self, key, value if isinstance(value, Parameter) else Parameter(value))
    def __contains__(self, key): return key in self._parameters
    def __len__(self): return len(self._parameters)
    def __iter__(self): return iter(self._parameters)
    def keys(self): return self._parameters.keys()
    def values(self): return self._parameters.values()
    def items(self): return self._parameters.items()


class Sequential(ModuleList):
    def __init__(self, *modules):
        Module.__init__(self)
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            for k, m in modules[0].items(): setattr(self, k, m)
        else:
            for m in modules: self.append(m)
    def forward(self, x):
        for module in self: x = module(x)
        return x


def _linear(x, weight, transpose=False):
    t = _trace_of(x)
    w = t.parameter(weight)
    x = _proxy(x, t)
    in_dim, out_dim = weight.shape[::-1] if transpose else weight.shape
    if x.shape[-1] != in_dim: raise ValueError("linear input width mismatch")
    return t.record("linear_t" if transpose else "linear", (x.name, w.name), x.shape[:-1] + (out_dim,),
                    dtype=x.dtype.fir if x.dtype.is_floating_point else "f32")


class Linear(Module):
    def __init__(self, in_features, out_features, name=None, bias=True, dtype="f32", layout=None):
        super().__init__()
        # Explicit legacy names use HF GPT-2 Conv1D orientation; unnamed layers use nn.Linear orientation.
        layout = layout or ("io" if name is not None else "oi")
        if layout not in ("io", "oi") or (layout == "oi" and dtype == "q4"):
            raise ValueError("linear supports io f32/q4 or oi f32 weights")
        self.in_features, self.out_features = in_features, out_features
        shape = (in_features, out_features) if layout == "io" else (out_features, in_features)
        self.weight = Parameter(shape, dtype, name + ".weight" if name else None)
        self.bias = Parameter((out_features,), name=name + ".bias" if name else None) if bias else None
        self.transpose = layout == "oi"
    def forward(self, x):
        y = _linear(x, self.weight, self.transpose)
        return y + self.bias if self.bias is not None else y


class Embedding(Module):
    def __init__(self, count, width, name=None, padding_idx=None):
        super().__init__()
        self.num_embeddings, self.embedding_dim, self.padding_idx = count, width, padding_idx
        self.weight = Parameter((count, width), name=name + ".weight" if name else None)
    def forward(self, index):
        return embedding(index, self.weight)
    def project(self, x): return _linear(x, self.weight, True)


def embedding(index, weight):
    """Row gather; the token/position role lowers to ext fir.embedding."""
    t = _trace_of(index)
    index = _proxy(index, t)
    if not isinstance(weight, Parameter):
        from .tensor import index_select
        w = _proxy(weight, t)
        return index_select(w, 0, index.reshape(-1)).reshape(tuple(index.shape) + (w.shape[1],))
    value = t.graph.values[index.name]
    if value.role in ("token", "position"):
        key = "vocab_size" if value.role == "token" else next((k for k in ("n_positions", "max_position_embeddings", "n_ctx") if k in t.graph.config), "n_positions")
        if key in t.graph.config and weight.shape[0] != t.graph.config[key]:
            raise ValueError("embedding rows must match " + key)
    w = t.parameter(weight)
    return t.record("embedding", (index.name, w.name), tuple(index.shape) + (weight.shape[1],))


def _norm_eps(t, eps):
    if eps is not None: return eps
    return next((t.graph.config[k] for k in ("layer_norm_epsilon", "layer_norm_eps", "norm_epsilon", "eps") if k in t.graph.config), 1e-5)


class LayerNorm(Module):
    def __init__(self, width, name=None, eps=None, elementwise_affine=True, bias=True):
        super().__init__()
        self.width = (width,) if isinstance(width, int) else tuple(width)
        self.normalized_shape = self.width
        self.eps = eps
        self.weight = Parameter(self.width, name=name + ".weight" if name else None) if elementwise_affine else None
        self.bias = Parameter(self.width, name=name + ".bias" if name else None) if elementwise_affine and bias else None
    def forward(self, x):
        return layer_norm(x, self.width, self.weight, self.bias, self.eps)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=None):
    width = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if tuple(x.shape[-len(width):]) != width: raise ValueError("LayerNorm width mismatch")
    t = _trace_of(x, weight, bias)
    x = _proxy(x, t)
    if bias is not None and weight is None:
        weight = _trace_of(x).constant_flat([1.0] * math.prod(width), width)
    ins = [x.name] + [_proxy(p, t).name for p in (weight, bias) if p is not None]
    return t.record("layernorm", ins, x.shape, {"width": math.prod(width), "eps": _norm_eps(t, eps)}, dtype=x.dtype.fir)


class RMSNorm(Module):
    def __init__(self, width, eps=1e-6, name=None, elementwise_affine=True):
        super().__init__()
        self.width, self.eps = width, eps
        self.normalized_shape = (width,)
        self.weight = Parameter((width,), name=name + ".weight" if name else None) if elementwise_affine else None
    def forward(self, x):
        return rms_norm(x, self.width, self.weight, self.eps)


def rms_norm(x, normalized_shape, weight=None, eps=None):
    width = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if tuple(x.shape[-len(width):]) != width: raise ValueError("RMSNorm width mismatch")
    t = _trace_of(x, weight)
    x = _proxy(x, t)
    if eps is None: eps = D.finfo(x.dtype if x.dtype.is_floating_point else D.float32).eps
    ins = [x.name] + ([_proxy(weight, t).name] if weight is not None else [])
    return t.record("rmsnorm", ins, x.shape, {"width": math.prod(width), "eps": eps}, dtype=x.dtype.fir)


class Conv1d(Module):
    """Stride-one causal NCL convolution, including grouped/depthwise kernels."""
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True):
        super().__init__()
        if (any(type(d) is not int or d < 1 for d in (in_channels, out_channels, kernel_size, groups))
                or in_channels % groups or out_channels % groups):
            raise ValueError("Conv1d needs positive dimensions and groups dividing input/output channels")
        self.channels, self.out_channels = in_channels, out_channels
        self.in_channels = in_channels
        self.kernel_size, self.groups = kernel_size, groups
        self.weight = Parameter((out_channels, in_channels//groups, kernel_size))
        self.bias = Parameter((out_channels,)) if bias else None
    def forward(self, x):
        return causal_conv1d(x, self.weight, self.bias, self.groups)


def causal_conv1d(x, weight, bias=None, groups=1):
    """Stride-one causal convolution: output t sees inputs t-K+1..t (zeros before 0)."""
    t = _trace_of(x, weight, bias)
    x = _proxy(x, t)
    w = _proxy(weight, t)
    out_channels, per_group, kernel = w.shape
    if x.ndim != 3 or x.shape[1] != per_group * groups: raise ValueError("Conv1d expects [batch,channels,length]")
    ins = [x.name, w.name]
    if bias is not None: ins.append(_proxy(bias, t).name)
    return t.record("conv1d", ins, (x.shape[0], out_channels, x.shape[2]),
                    {"kernel_size": kernel, "groups": groups})


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if not 0 <= p <= 1: raise ValueError("dropout probability out of range")
        self.p = p
    def forward(self, x):
        if self.training and self.p > 0:
            raise Unsupported("dropout in training mode needs an RNG; call eval() before tracing")
        return x


class Identity(Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, x): return x


class GELU(Module):
    def forward(self, x): return x.gelu()
