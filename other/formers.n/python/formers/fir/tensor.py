"""Symbolic tensor proxies, shape rules, and functional operators (stdlib only).

``ProxyTensor`` implements the torch ``Tensor`` surface used by inference code
on top of a small set of traced operations. Methods without a FIR operation of
their own are compositions of traced operations; operations that need integer
rounding or data-dependent shapes are evaluated at trace time when their inputs
are constants and raise ``Unsupported`` otherwise.

Outside ``fir.trace`` the same API evaluates eagerly on Python values (a private
eager trace that folds every operation into a constant), so construction-time
code such as ``inv_freq = 1 / base ** (arange(0, d, 2).float() / d)`` and
``tril(ones(n, n))`` produces constants that become graph constants when a
traced operation reads them.
"""

from collections import namedtuple
from dataclasses import dataclass
import math
import builtins
from itertools import zip_longest

from . import dtypes as D


class Unsupported(NotImplementedError):
    """The operation has no FIR lowering for runtime (non-constant) tensors."""


@dataclass(frozen=True)
class Dim:
    """A named dimension specialized by trace config or TensorSpec bindings."""
    name: str
    value: int = None


@dataclass(frozen=True, init=False)
class TensorSpec:
    """Example input: shape, runner role (``token``, ``position`` or ``""``) and dtype."""
    shape: tuple
    role: str
    dtype: str

    def __init__(self, shape, role="", dtype="f32"):
        shape = (shape,) if isinstance(shape, (int, str, Dim)) else tuple(shape)
        if builtins.any(isinstance(d, int) and d <= 0 for d in shape):
            raise ValueError("dimensions must be positive")
        if role not in ("", "token", "position"):
            raise ValueError("unknown input role: " + role)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "dtype", D.dtype(dtype).fir)

    @property
    def size(self):
        return math.prod(self.shape)


class Size(tuple):
    """torch.Size: a tuple of ints with ``numel()``."""
    def numel(self): return math.prod(self)
    def __getitem__(self, ix):
        r = tuple.__getitem__(self, ix)
        return Size(r) if isinstance(ix, slice) else r
    def __add__(self, other): return Size(tuple(self) + tuple(other))
    def __radd__(self, other): return Size(tuple(other) + tuple(self))
    def __repr__(self): return "Size(%s)" % list(self)


class Device:
    """torch.device; tracing is device independent, every device is accepted."""
    def __init__(self, type="cpu", index=None):
        if isinstance(type, Device): type, index = type.type, type.index
        type = str(type)
        if ":" in type: type, index = type.split(":")[0], int(type.split(":")[1])
        self.type, self.index = type, index
    def __repr__(self): return "device(type=%r%s)" % (self.type, "" if self.index is None else ", index=%d" % self.index)
    def __str__(self): return self.type if self.index is None else "%s:%d" % (self.type, self.index)
    def __eq__(self, other):
        if isinstance(other, str): other = Device(other)
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)
    def __hash__(self): return hash((self.type, self.index))


CPU = Device("cpu")
MaxResult = namedtuple("max", ["values", "indices"])
MinResult = namedtuple("min", ["values", "indices"])
VarMean = namedtuple("var_mean", ["var", "mean"])


def broadcast(a, b):
    out = []
    for x, y in zip_longest(reversed(a), reversed(b), fillvalue=1):
        if x != y and x != 1 and y != 1:
            raise ValueError("broadcast requires compatible shapes (equal element counts are insufficient): %s, %s" % (a, b))
        out.append(builtins.max(x, y))
    return tuple(reversed(out))


def _is_tensor(x):
    from .nn import Parameter
    return isinstance(x, (ProxyTensor, Parameter, Uninitialized))


def _trace_of(*xs):
    from .tracer import active_trace, EagerTrace
    traces = {}
    for x in xs:
        if isinstance(x, ProxyTensor):
            traces[id(x._trace)] = x._trace
    real = [t for t in traces.values() if not t.eager]
    if len(real) > 1:
        raise ValueError("tensors belong to different traces")
    if real: return real[0]
    active = active_trace()
    if active is not None: return active
    if traces: return next(iter(traces.values()))
    return EagerTrace()


def _proxy(x, trace):
    from .nn import Parameter
    if isinstance(x, ProxyTensor):
        if x._trace is trace: return x
        if x._trace.eager: return trace.import_constant(x)
        raise ValueError("tensors belong to different traces")
    if isinstance(x, Parameter):
        return trace.parameter(x)
    if isinstance(x, Uninitialized):
        return trace.constant_flat([0.0] * math.prod(x.shape), tuple(x.shape), x.dtype.fir)
    if isinstance(x, D.DType) or x is None:
        raise TypeError("expected a tensor or number, got %r" % (x,))
    return trace.constant(_finite_data(x), _infer_dtype(x).fir)


def _finite_data(x):
    if isinstance(x, (list, tuple)): return [_finite_data(y) for y in x]
    if _is_tensor(x): return x.tolist()
    return D.finite(x)


def _infer_dtype(data):
    """torch.tensor dtype inference: bool, int64 or float32."""
    kinds = set()
    def walk(x):
        if isinstance(x, (list, tuple)):
            for y in x: walk(y)
        elif _is_tensor(x): kinds.add(D.dtype(x.dtype).category)
        else: kinds.add(D.of_scalar(x).category)
    walk(data)
    top = builtins.max(kinds) if kinds else 2
    return (D.bool_, D.int64, D.float32)[top]


def static_values(x):
    """Flat Python values of a tensor whose value is known at trace time, else None."""
    if not isinstance(x, ProxyTensor): return None
    graph = x._trace.graph
    v = graph.values[x.name]
    if v.constant is not None: return list(v.constant)
    from .reference import run_node
    producers = {n.output: n for n in graph.nodes}
    memo = {}
    def value(k, depth=0):
        if k in memo: return memo[k]
        val = graph.values[k]
        if val.constant is not None: memo[k] = list(val.constant); return memo[k]
        node = producers.get(k)
        if node is None or node.effectful or node.extra or depth > 200: return None
        args = {}
        for i in node.inputs:
            r = value(i, depth + 1)
            if r is None: return None
            args[i] = r
        memo[k] = run_node(node, graph, args)[k]
        return memo[k]
    return value(x.name)


def _static_tensor(values, shape, dtype, like=()):
    t = _trace_of(*like)
    return t.constant_flat([float(D.finite(v)) for v in values], tuple(shape), D.dtype(dtype).fir)


def _require_static(x, what):
    values = static_values(x) if isinstance(x, ProxyTensor) else None
    if values is None:
        raise Unsupported(what + " of a runtime tensor has no FIR operation (it needs integer rounding or a "
                          "data-dependent shape); it is evaluated at trace time for constant tensors only")
    return values


def _nest(values, shape):
    if not shape: return values[0]
    step = math.prod(shape[1:])
    return [_nest(values[i*step:(i+1)*step], shape[1:]) for i in range(shape[0])]


def tensor(data, dtype=None, device=None, requires_grad=False):
    """Constant tensor from nested Python values (torch.tensor dtype inference)."""
    if isinstance(data, ProxyTensor): return data.to(dtype) if dtype is not None else data
    target = D.dtype(dtype) if dtype is not None else _infer_dtype(data)
    t = _trace_of()
    x = t.constant(_finite_data(data), target.fir)
    if target.category < 2 and _infer_dtype(data).category == 2:
        values = static_values(x)
        cast = (lambda v: float(v != 0)) if target is D.bool_ else (lambda v: float(math.trunc(v)))
        x = t.constant_flat([cast(v) for v in values], x.shape, target.fir)
    elif target is D.bool_:
        x = t.constant_flat([float(v != 0) for v in static_values(x)], x.shape, "bool")
    return x


as_tensor = tensor


def _result_dtype(op, a, b):
    if op in ("lt", "le", "gt", "ge", "eq", "ne"): return D.bool_
    r = D.result_type(*(_dtype_operand(x) for x in (a, b)))
    if op == "div" and r.category < 2: return D.DEFAULT_FLOAT
    return r


class _Operand:
    __slots__ = ("dtype", "ndim")
    def __init__(self, dtype, ndim): self.dtype, self.ndim = dtype, ndim


def _dtype_operand(x):
    if isinstance(x, (bool, int, float)): return x
    if _is_tensor(x): return _Operand(D.dtype(x.dtype), len(x.shape))
    if isinstance(x, (list, tuple)): return _Operand(_infer_dtype(x), 1)
    raise TypeError("unsupported operand %r" % (x,))


def binary(op, a, b):
    t = _trace_of(a, b)
    kind = _result_dtype(op, a, b)
    a, b = _proxy(a, t), _proxy(b, t)
    return t.record(op, (a.name, b.name), broadcast(a.shape, b.shape), dtype=kind.fir)


FLOAT_UNARY = {"gelu_new", "gelu", "gelu_quick", "silu", "tanh", "sigmoid", "softplus", "exp", "log",
               "sqrt", "rsqrt", "sin", "cos"}


def unary(op, x):
    x = _proxy(x, _trace_of(x))
    d = x.dtype
    if op in FLOAT_UNARY and not d.is_floating_point: d = D.DEFAULT_FLOAT
    return x._trace.record(op, (x.name,), x.shape, dtype=d.fir)


def _dim(d, ndim):
    if ndim == 0 and d in (0, -1): return 0
    if not -ndim <= d < ndim: raise IndexError("dimension %d out of range for rank %d" % (d, ndim))
    return d % ndim


def _sizes(args):
    if len(args) == 1 and isinstance(args[0], (tuple, list)): return tuple(args[0])
    return tuple(args)


def _cast_dtype_args(args, kwargs):
    """Parse Tensor.to(...) arguments; returns the target dtype or None."""
    target = kwargs.get("dtype")
    for a in args:
        if isinstance(a, ProxyTensor): target = a.dtype
        elif isinstance(a, (D.DType,)) or a in (float, int, bool): target = a
        elif isinstance(a, str):
            try: target = D.dtype(a)
            except TypeError: Device(a)
        elif isinstance(a, Device) or a is None: pass
        else:
            try: target = D.dtype(a)
            except TypeError: raise TypeError("unsupported Tensor.to argument %r" % (a,)) from None
    return D.dtype(target) if target is not None else None


VIEW_PRODUCERS = ("reshape", "permute", "slice", "expand")


class ProxyTensor:
    """A traced (or eager constant) tensor with the torch ``Tensor`` API."""

    __array_priority__ = 1000
    requires_grad = False
    grad = None
    is_cuda = False
    is_sparse = False
    is_quantized = False
    is_meta = False

    def __init__(self, trace, name, dtype=None):
        self._trace, self.name, self._dtype = trace, name, dtype

    __hash__ = object.__hash__

    def _value(self): return self._trace.graph.values[self.name]

    def __repr__(self):
        return "ProxyTensor(%s, shape=%s, dtype=%s)" % (self.name, list(self.shape), self.dtype)

    # ---- metadata ---------------------------------------------------------
    @property
    def shape(self): return Size(self._value().shape)

    @property
    def dtype(self): return D.dtype(self._dtype or self._value().dtype)

    @property
    def device(self): return CPU

    @property
    def ndim(self): return len(self._value().shape)

    @property
    def data(self): return self

    @property
    def layout(self): return "strided"

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[_dim(dim, self.ndim)]

    def dim(self): return self.ndim
    ndimension = dim
    def numel(self): return self._value().size
    nelement = numel
    def element_size(self): return self.dtype.itemsize
    def is_floating_point(self): return self.dtype.is_floating_point
    def is_complex(self): return False
    def is_contiguous(self, memory_format=None): return True
    def get_device(self): return -1
    def __len__(self):
        if not self.ndim: raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    # ---- static (trace-time) values ----------------------------------------
    def tolist(self):
        values = _require_static(self, "tolist()")
        if not self.dtype.is_floating_point: values = [int(v) if self.dtype is not D.bool_ else bool(v) for v in values]
        return _nest(values, self.shape) if self.ndim else values[0]

    def item(self):
        if self.numel() != 1: raise ValueError("item() needs exactly one element")
        values = _require_static(self, "item()")
        v = values[0]
        if self.dtype is D.bool_: return bool(v)
        return int(v) if not self.dtype.is_floating_point else v

    def __bool__(self):
        values = static_values(self) if self.numel() == 1 else None
        if values is None:
            raise TypeError("data-dependent Python control flow/comparisons require where or a static config branch")
        return bool(values[0])

    def __int__(self): return int(self.item())
    def __float__(self): return float(self.item())
    def __index__(self):
        if self.dtype.is_floating_point: raise TypeError("only integer tensors can be used as an index")
        return int(self.item())

    def __iter__(self):
        if not self.ndim: raise TypeError("iteration over a 0-d tensor")
        return iter(self.unbind(0))

    def numpy(self): raise Unsupported("numpy() needs tensor payloads; FIR traces metadata only")

    # ---- operators ----------------------------------------------------------
    def __add__(self, x): return binary("add", self, x)
    def __radd__(self, x): return binary("add", x, self)
    def __sub__(self, x): return binary("sub", self, x)
    def __rsub__(self, x): return binary("sub", x, self)
    def __mul__(self, x): return binary("mul", self, x)
    def __rmul__(self, x): return binary("mul", x, self)
    def __truediv__(self, x): return binary("div", self, x)
    def __rtruediv__(self, x): return binary("div", x, self)
    def __pow__(self, x): return pow(self, x)
    def __rpow__(self, x): return pow(x, self)
    def __neg__(self): return unary("neg", self)
    def __pos__(self): return self
    def __abs__(self): return abs(self)
    def __matmul__(self, x): return matmul(self, x)
    def __rmatmul__(self, x): return matmul(x, self)
    def __lt__(self, x): return binary("lt", self, x)
    def __le__(self, x): return binary("le", self, x)
    def __gt__(self, x): return binary("gt", self, x)
    def __ge__(self, x): return binary("ge", self, x)
    def __eq__(self, x): return binary("eq", self, x)
    def __ne__(self, x): return binary("ne", self, x)
    def __and__(self, x): return logical("and", self, x)
    def __rand__(self, x): return logical("and", x, self)
    def __or__(self, x): return logical("or", self, x)
    def __ror__(self, x): return logical("or", x, self)
    def __xor__(self, x): return logical("xor", self, x)
    def __rxor__(self, x): return logical("xor", x, self)
    def __invert__(self): return logical_not(self) if self.dtype is D.bool_ else bitwise_not(self)
    def __floordiv__(self, x): return div(self, x, rounding_mode="floor")
    def __rfloordiv__(self, x): return div(x, self, rounding_mode="floor")
    def __mod__(self, x): return remainder(self, x)
    def __rmod__(self, x): return remainder(x, self)
    def __lshift__(self, x): return _static_binary(self, x, lambda a, b: int(a) << int(b), "<<")
    def __rshift__(self, x): return _static_binary(self, x, lambda a, b: int(a) >> int(b), ">>")

    def __setitem__(self, index, value):
        self._inplace(index_put_basic(self, index, value), "__setitem__")

    # ---- views ----------------------------------------------------------------
    def reshape(self, *shape):
        shape = _sizes(shape)
        if not shape:
            if self.numel() != 1: raise ValueError("reshape element count mismatch")
            return self._trace.record("reshape", (self.name,), (), dtype=self._fir())
        shape = list(shape)
        if shape.count(-1) > 1:
            raise ValueError("only one inferred reshape dimension is allowed")
        if -1 in shape:
            known = math.prod(d for d in shape if d != -1)
            if known <= 0 or self.numel() % known:
                raise ValueError("reshape element count mismatch")
            shape[shape.index(-1)] = self.numel() // known
        if builtins.any(type(d) is not int or d <= 0 for d in shape) or math.prod(shape) != self.numel():
            raise ValueError("reshape element count mismatch")
        return self._trace.record("reshape", (self.name,), tuple(shape), dtype=self._fir())

    def view(self, *shape):
        if len(shape) == 1 and (isinstance(shape[0], D.DType) or shape[0] in (float, int)):
            raise Unsupported("view(dtype) reinterprets bits; FIR has no bit casts")
        return self.reshape(*shape)

    def view_as(self, other): return self.reshape(tuple(other.shape))
    reshape_as = view_as

    def _fir(self): return self.dtype.fir

    def permute(self, *dims):
        dims = tuple(_dim(d, self.ndim) for d in _sizes(dims))
        if sorted(dims) != list(range(self.ndim)):
            raise ValueError("permute needs each dimension once")
        return self._trace.record("permute", (self.name,), tuple(self.shape[d] for d in dims), {"dims": dims},
                                  dtype=self._fir())

    def transpose(self, a, b):
        dims = list(range(self.ndim))
        a, b = _dim(a, self.ndim), _dim(b, self.ndim)
        dims[a], dims[b] = dims[b], dims[a]
        return self.permute(dims)

    swapaxes = swapdims = transpose

    def t(self):
        if self.ndim > 2: raise ValueError("t() expects a tensor with <= 2 dimensions")
        return self.transpose(0, 1) if self.ndim == 2 else self

    def movedim(self, source, destination):
        src = (source,) if isinstance(source, int) else tuple(source)
        dst = (destination,) if isinstance(destination, int) else tuple(destination)
        src = [_dim(d, self.ndim) for d in src]
        dst = [_dim(d, self.ndim) for d in dst]
        order = [None] * self.ndim
        for s, d in zip(src, dst): order[d] = s
        rest = iter(d for d in range(self.ndim) if d not in src)
        return self.permute([o if o is not None else next(rest) for o in order])

    moveaxis = movedim

    @property
    def T(self):
        return self.permute(tuple(reversed(range(self.ndim))))

    @property
    def mT(self):
        if self.ndim < 2: raise ValueError("mT needs at least two dimensions")
        return self.transpose(-2, -1)

    mH = mT

    @property
    def H(self): return self.T

    def __getitem__(self, index): return getitem(self, index)

    def split(self, sizes, dim=0):
        dim = _dim(dim, self.ndim)
        if isinstance(sizes, int):
            if sizes <= 0: raise ValueError("split size must be positive")
            sizes = [builtins.min(sizes, self.shape[dim]-i) for i in range(0, self.shape[dim], sizes)]
        sizes = [int(s) for s in sizes]
        if builtins.sum(sizes) != self.shape[dim] or builtins.any(s <= 0 for s in sizes):
            raise ValueError("split sizes must cover dimension")
        out, start = [], 0
        for size in sizes:
            ix = [slice(None)] * self.ndim
            ix[dim] = slice(start, start+size)
            out.append(self[tuple(ix)])
            start += size
        return tuple(out)

    split_with_sizes = split

    def chunk(self, chunks, dim=0):
        if chunks <= 0: raise ValueError("chunks must be positive")
        return self.split((self.shape[_dim(dim, self.ndim)]+chunks-1)//chunks, dim)

    def tensor_split(self, sections, dim=0):
        n = self.shape[_dim(dim, self.ndim)]
        if isinstance(sections, int):
            q, r = divmod(n, sections)
            sizes = [q + (1 if i < r else 0) for i in range(sections)]
        else:
            bounds = [0] + list(sections) + [n]
            sizes = [b - a for a, b in zip(bounds, bounds[1:])]
        return self.split([s for s in sizes if s > 0], dim)

    def unbind(self, dim=0):
        dim = _dim(dim, self.ndim)
        return tuple(self.select(dim, i) for i in range(self.shape[dim]))

    def select(self, dim, index):
        ix = [slice(None)] * self.ndim
        ix[_dim(dim, self.ndim)] = int(index)
        return self[tuple(ix)]

    def narrow(self, dim, start, length):
        dim = _dim(dim, self.ndim)
        start = int(start) % self.shape[dim] if int(start) < 0 else int(start)
        ix = [slice(None)] * self.ndim
        ix[dim] = slice(start, start + int(length))
        return self[tuple(ix)]

    def unsqueeze(self, dim):
        dim = dim % (self.ndim + 1)
        return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

    def squeeze(self, dim=None):
        if dim is None: return self.reshape(tuple(d for d in self.shape if d != 1) or ())
        dims = {_dim(d, self.ndim) for d in ((dim,) if isinstance(dim, int) else dim)}
        shape = tuple(s for i, s in enumerate(self.shape) if not (i in dims and s == 1))
        return self if shape == self.shape else self.reshape(shape)

    def flatten(self, start_dim=0, end_dim=-1):
        if not self.ndim: return self.reshape(1)
        a, b = _dim(start_dim, self.ndim), _dim(end_dim, self.ndim)
        if a > b: raise ValueError("flatten start_dim must not exceed end_dim")
        shape = self.shape[:a] + (math.prod(self.shape[a:b+1]),) + self.shape[b+1:]
        return self if shape == self.shape else self.reshape(shape)

    def unflatten(self, dim, sizes):
        dim = _dim(dim, self.ndim)
        sizes = list(sizes)
        if -1 in sizes:
            sizes[sizes.index(-1)] = self.shape[dim] // math.prod(s for s in sizes if s != -1)
        if math.prod(sizes) != self.shape[dim]: raise ValueError("unflatten sizes must multiply to the dimension")
        return self.reshape(self.shape[:dim] + tuple(sizes) + self.shape[dim+1:])

    def ravel(self): return self.reshape(-1)

    def expand(self, *sizes):
        sizes = _sizes(sizes)
        if len(sizes) < self.ndim: raise ValueError("expand needs at least the tensor rank")
        lead = len(sizes) - self.ndim
        shape = []
        for i, size in enumerate(sizes):
            have = self.shape[i - lead] if i >= lead else 1
            size = have if size == -1 else int(size)
            if have != 1 and size != have: raise ValueError("expand can only broadcast size-1 dimensions")
            shape.append(size)
        if tuple(shape) == self.shape: return self
        return self._trace.record("expand", (self.name,), tuple(shape), dtype=self._fir())

    def expand_as(self, other): return self.expand(tuple(other.shape))
    def broadcast_to(self, shape): return self.expand(tuple(shape))

    def repeat(self, *sizes): return repeat(self, _sizes(sizes))
    def tile(self, *dims):
        dims = _sizes(dims)
        return repeat(self, (1,) * (self.ndim - len(dims)) + tuple(dims))
    def repeat_interleave(self, repeats, dim=None, output_size=None): return repeat_interleave(self, repeats, dim)

    def contiguous(self, memory_format=None): return self
    def clone(self, memory_format=None):
        # A new proxy: in-place methods rebind it without changing the source tensor.
        return ProxyTensor(self._trace, self.name, self._dtype)
    def detach(self): return self
    def requires_grad_(self, requires_grad=True): return self
    def cpu(self): return self
    def cuda(self, *args, **kwargs): return self
    def pin_memory(self): return self
    def share_memory_(self): return self

    # ---- dtype ---------------------------------------------------------------
    def to(self, *args, **kwargs):
        target = _cast_dtype_args(args, kwargs)
        return self if target is None else cast(self, target)

    def type(self, dtype=None, non_blocking=False):
        if dtype is None: return "torch." + {"float32": "FloatTensor", "float16": "HalfTensor", "bfloat16": "BFloat16Tensor",
                                             "float64": "DoubleTensor", "int64": "LongTensor", "int32": "IntTensor",
                                             "bool": "BoolTensor"}.get(self.dtype.name, "Tensor")
        if isinstance(dtype, str) and dtype.endswith("Tensor"):
            dtype = {"FloatTensor": "float32", "HalfTensor": "float16", "BFloat16Tensor": "bfloat16", "DoubleTensor": "float64",
                     "LongTensor": "int64", "IntTensor": "int32", "BoolTensor": "bool"}[dtype.split(".")[-1]]
        return cast(self, dtype)

    def type_as(self, other): return cast(self, other.dtype)
    def float(self, memory_format=None): return cast(self, D.float32)
    def half(self, memory_format=None): return cast(self, D.float16)
    def bfloat16(self, memory_format=None): return cast(self, D.bfloat16)
    def double(self, memory_format=None): return cast(self, D.float64)
    def int(self, memory_format=None): return cast(self, D.int32)
    def long(self, memory_format=None): return cast(self, D.int64)
    def short(self, memory_format=None): return cast(self, D.int16)
    def char(self, memory_format=None): return cast(self, D.int8)
    def byte(self, memory_format=None): return cast(self, D.uint8)
    def bool(self, memory_format=None): return cast(self, D.bool_)

    def new_zeros(self, *size, dtype=None, device=None, requires_grad=False):
        return zeros(_sizes(size), dtype=dtype or self.dtype)
    def new_ones(self, *size, dtype=None, device=None, requires_grad=False):
        return ones(_sizes(size), dtype=dtype or self.dtype)
    def new_empty(self, *size, dtype=None, device=None, requires_grad=False):
        return zeros(_sizes(size), dtype=dtype or self.dtype)
    def new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False):
        return full(size, fill_value, dtype=dtype or self.dtype)
    def new_tensor(self, data, dtype=None, device=None, requires_grad=False):
        return tensor(data, dtype=dtype or self.dtype)

    # ---- arithmetic method forms ---------------------------------------------------
    def add(self, other, alpha=1): return add(self, other, alpha=alpha)
    def sub(self, other, alpha=1): return sub(self, other, alpha=alpha)
    subtract = sub
    def mul(self, other): return binary("mul", self, other)
    multiply = mul
    def div(self, other, rounding_mode=None): return div(self, other, rounding_mode=rounding_mode)
    divide = true_divide = div
    def floor_divide(self, other): return div(self, other, rounding_mode="floor")
    def remainder(self, other): return remainder(self, other)
    def fmod(self, other): return fmod(self, other)
    def pow(self, exponent): return pow(self, exponent)
    def float_power(self, exponent): return pow(self.double(), exponent)
    def neg(self): return unary("neg", self)
    negative = neg
    def abs(self): return abs(self)
    absolute = abs
    def reciprocal(self): return reciprocal(self)
    def square(self): return binary("mul", self, self)
    def sign(self): return sign(self)
    sgn = sign
    def matmul(self, other): return matmul(self, other)
    def mm(self, other): return matmul(self, other)
    def bmm(self, other): return matmul(self, other)
    def mv(self, other): return matmul(self, other)
    def dot(self, other): return matmul(self, other)
    def outer(self, other): return outer(self, other)
    def lerp(self, end, weight): return lerp(self, end, weight)
    def addcmul(self, a, b, value=1): return self + a * b * value
    def addcdiv(self, a, b, value=1): return self + a / b * value
    def clamp(self, min=None, max=None): return clamp(self, min, max)
    clip = clamp
    def clamp_min(self, min): return clamp(self, min, None)
    def clamp_max(self, max): return clamp(self, None, max)
    def maximum(self, other): return maximum(self, other)
    def minimum(self, other): return minimum(self, other)
    def fmax(self, other): return maximum(self, other)
    def fmin(self, other): return minimum(self, other)
    def eq(self, other): return binary("eq", self, other)
    def ne(self, other): return binary("ne", self, other)
    def lt(self, other): return binary("lt", self, other)
    def le(self, other): return binary("le", self, other)
    def gt(self, other): return binary("gt", self, other)
    def ge(self, other): return binary("ge", self, other)
    greater, greater_equal, less, less_equal, not_equal = gt, ge, lt, le, ne
    def equal(self, other):
        a, b = _require_static(self, "equal()"), _require_static(_proxy(other, self._trace), "equal()")
        return tuple(self.shape) == tuple(other.shape) and a == b
    def allclose(self, other, rtol=1e-5, atol=1e-8, equal_nan=False):
        a, b = _require_static(self, "allclose()"), _require_static(_proxy(other, self._trace), "allclose()")
        return builtins.all(builtins.abs(x - y) <= atol + rtol * builtins.abs(y) for x, y in zip(a, b))
    def logical_and(self, other): return logical("and", self, other)
    def logical_or(self, other): return logical("or", self, other)
    def logical_xor(self, other): return logical("xor", self, other)
    def logical_not(self): return logical_not(self)
    def bitwise_and(self, other): return logical("and", self, other)
    def bitwise_or(self, other): return logical("or", self, other)
    def bitwise_xor(self, other): return logical("xor", self, other)
    def bitwise_not(self): return self.__invert__()
    def where(self, condition, other): return where(condition, self, other)
    def masked_fill(self, mask, value): return masked_fill(self, mask, value)
    def isnan(self):
        values = static_values(self)
        if values is None: raise Unsupported("isnan of a runtime tensor: FIR values are always finite")
        return _static_tensor([0.0] * len(values), self.shape, "bool", (self,))
    def isinf(self): return self.isnan()
    def isfinite(self): return logical_not(self.isnan())
    def nan_to_num(self, nan=0.0, posinf=None, neginf=None): return self

    # ---- pointwise math -----------------------------------------------------------
    def exp(self): return exp(self)
    def exp2(self): return exp2(self)
    def expm1(self): return expm1(self)
    def log(self): return log(self)
    def log2(self): return log2(self)
    def log10(self): return log10(self)
    def log1p(self): return log1p(self)
    def sqrt(self): return sqrt(self)
    def rsqrt(self): return rsqrt(self)
    def sin(self): return sin(self)
    def cos(self): return cos(self)
    def tan(self): return tan(self)
    def sinh(self): return sinh(self)
    def cosh(self): return cosh(self)
    def tanh(self): return tanh(self)
    def sigmoid(self): return sigmoid(self)
    def relu(self): return relu(self)
    def silu(self): return silu(self)
    def softplus(self, beta=1, threshold=20): return softplus(self, beta, threshold)
    def gelu(self, approximate="tanh"): return gelu(self, approximate)
    def floor(self): return floor(self)
    def ceil(self): return ceil(self)
    def round(self, decimals=0): return round(self, decimals)
    def trunc(self): return trunc(self)
    fix = trunc
    def frac(self): return self - trunc(self)
    def erf(self): return erf(self)
    def atan(self): return _static_unary(self, math.atan, "atan", D.float32)
    def asin(self): return _static_unary(self, math.asin, "asin", D.float32)
    def acos(self): return _static_unary(self, math.acos, "acos", D.float32)

    # ---- reductions ---------------------------------------------------------------
    def sum(self, dim=None, keepdim=False, dtype=None): return sum(self, dim, keepdim, dtype=dtype)
    def mean(self, dim=None, keepdim=False, dtype=None): return mean(self, dim, keepdim, dtype=dtype)
    def max(self, dim=None, keepdim=False): return max(self, dim, keepdim)
    def min(self, dim=None, keepdim=False): return min(self, dim, keepdim)
    def amax(self, dim=(), keepdim=False): return amax(self, dim, keepdim)
    def amin(self, dim=(), keepdim=False): return amin(self, dim, keepdim)
    def argmax(self, dim=None, keepdim=False): return argmax(self, dim, keepdim)
    def argmin(self, dim=None, keepdim=False): return argmin(self, dim, keepdim)
    def var(self, dim=None, unbiased=None, keepdim=False, correction=None): return var(self, dim, unbiased, keepdim, correction=correction)
    def std(self, dim=None, unbiased=None, keepdim=False, correction=None): return std(self, dim, unbiased, keepdim, correction=correction)
    def norm(self, p="fro", dim=None, keepdim=False, dtype=None): return norm(self, p, dim, keepdim)
    def logsumexp(self, dim, keepdim=False): return logsumexp(self, dim, keepdim)
    def any(self, dim=None, keepdim=False): return any(self, dim, keepdim)
    def all(self, dim=None, keepdim=False): return all(self, dim, keepdim)
    def prod(self, dim=None, keepdim=False, dtype=None): return prod(self, dim, keepdim)
    def count_nonzero(self, dim=None): return sum(self != 0, dim)
    def softmax(self, dim=-1, dtype=None): return softmax(self, dim, dtype=dtype)
    def log_softmax(self, dim=-1, dtype=None): return log_softmax(self, dim, dtype=dtype)
    def cumsum(self, dim=0, dtype=None): return cumsum(self, dim, dtype=dtype)
    def cumprod(self, dim=0, dtype=None): return cumprod(self, dim)
    def topk(self, k, dim=-1, largest=True, sorted=True): return topk(self, k, dim, largest, sorted)
    def sort(self, dim=-1, descending=False, stable=False): return sort(self, dim, descending)
    def argsort(self, dim=-1, descending=False, stable=False): return sort(self, dim, descending)[1]
    def index_select(self, dim, index): return index_select(self, dim, index)
    def gather(self, dim, index): return gather(self, dim, index)
    def take_along_dim(self, indices, dim=None): return gather(self, dim, indices)
    def tril(self, diagonal=0): return tril(self, diagonal)
    def triu(self, diagonal=0): return triu(self, diagonal)
    def flip(self, *dims): return flip(self, _sizes(dims))
    def fliplr(self): return flip(self, (1,))
    def flipud(self): return flip(self, (0,))
    def roll(self, shifts, dims=None): return roll(self, shifts, dims)
    def diff(self, n=1, dim=-1): return diff(self, n, dim)
    def diagonal(self, offset=0, dim1=0, dim2=1): return diagonal(self, offset, dim1, dim2)
    def nonzero(self, as_tuple=False): return nonzero(self, as_tuple)

    # ---- in place (functionalized by rebinding this proxy) ------------------------
    def _inplace(self, result, name):
        graph = self._trace.graph
        producer = next((n for n in reversed(graph.nodes) if n.output == self.name), None)
        if producer is not None and producer.op in VIEW_PRODUCERS:
            raise Unsupported("%s on a view writes through to its base tensor; FIR values are SSA, use the out-of-place form" % name)
        if builtins.any(n.op in VIEW_PRODUCERS and n.inputs and n.inputs[0] == self.name for n in graph.nodes):
            raise Unsupported("%s on a tensor that has views would not update the views; use the out-of-place form" % name)
        if self._value().state or self._value().weight or (producer is not None and producer.op == "state_read"):
            raise Unsupported("%s on a state or weight; use State.update" % name)
        result = _retag(result, self.dtype) if result.dtype != self.dtype else result
        if tuple(result.shape) != tuple(self.shape): result = result.expand(tuple(self.shape))
        self.name, self._dtype = result.name, result._dtype
        return self

    def __getattr__(self, name):
        if name.endswith("_") and not name.startswith("_") and name[:-1] in INPLACE:
            base = name[:-1]
            def method(*args, **kwargs):
                if base == "copy": value = args[0]
                elif base == "fill": value = full_like(self, args[0])
                elif base == "zero": value = zeros_like(self)
                elif base in ("normal", "uniform"): return self
                else: value = getattr(self, base)(*args, **kwargs)
                if base == "copy" and not isinstance(value, ProxyTensor): value = tensor(value)
                return self._inplace(_proxy(value, self._trace) if isinstance(value, ProxyTensor) else value, name)
            return method
        raise AttributeError("%r has no attribute %r" % (type(self).__name__, name))


INPLACE = {"add", "sub", "mul", "div", "pow", "neg", "abs", "clamp", "clamp_min", "clamp_max", "masked_fill",
           "exp", "log", "sqrt", "rsqrt", "sigmoid", "tanh", "relu", "silu", "copy", "fill", "zero", "normal",
           "uniform", "floor", "ceil", "round", "trunc", "reciprocal", "square", "unsqueeze", "squeeze", "t",
           "transpose", "float", "lerp", "addcmul", "addcdiv", "where", "tril", "triu", "sin", "cos"}


class Uninitialized:
    """``torch.empty`` outside a trace: shape and dtype only (reads as zeros)."""
    def __init__(self, shape, dtype=None):
        self.shape, self.dtype = Size(shape), D.dtype(dtype) or D.float32
    ndim = property(lambda self: len(self.shape))
    device = property(lambda self: CPU)
    def size(self, dim=None): return self.shape if dim is None else self.shape[dim]
    def dim(self): return self.ndim
    def numel(self): return math.prod(self.shape)
    def _real(self): return zeros(self.shape, dtype=self.dtype)
    def tolist(self): return self._real().tolist()
    def normal_(self, *a, **k): return self
    uniform_ = zero_ = fill_ = normal_
    def __getattr__(self, name):
        if name.startswith("_"): raise AttributeError(name)
        return getattr(self._real(), name)
    def __add__(self, x): return self._real() + x
    def __radd__(self, x): return x + self._real()
    def __mul__(self, x): return self._real() * x
    def __rmul__(self, x): return x * self._real()


# ---- casts --------------------------------------------------------------------------

def cast(x, dtype):
    """Change the logical dtype; values stay f32 (integer casts of runtime floats are rejected)."""
    target = D.dtype(dtype)
    if not isinstance(x, ProxyTensor): x = _proxy(x, _trace_of(x))
    source = x.dtype
    if source is target: return x
    if target is D.bool_:
        if source is D.bool_: return x
        r = binary("ne", x, 0)
        return r
    if target.category == 1 and source.category == 2:
        values = static_values(x)
        if values is None:
            if not _integral_runtime(x):
                raise Unsupported("casting a runtime float tensor to %s needs truncation, which FIR does not have" % target)
            return _retag(x, target)
        return _static_tensor([math.trunc(v) for v in values], x.shape, target, (x,))
    return ProxyTensor(x._trace, x.name, target.fir)


def _retag(x, dtype):
    """Same values, new logical dtype (no conversion)."""
    return ProxyTensor(x._trace, x.name, D.dtype(dtype).fir)


def _integral_runtime(x):
    """Token and position inputs (and views of them) hold integers."""
    graph = x._trace.graph
    producers = {n.output: n for n in graph.nodes}
    k = x.name
    for _ in range(32):
        v = graph.values[k]
        if v.role in ("token", "position"): return True
        n = producers.get(k)
        if n is None or n.op not in VIEW_PRODUCERS: return False
        k = n.inputs[0]
    return False


# ---- indexing -----------------------------------------------------------------------

def _normalize_index(x, index):
    index = index if isinstance(index, tuple) else (index,)
    if builtins.sum(i is Ellipsis for i in index) > 1:
        raise IndexError("only one ellipsis allowed")
    consumed = builtins.sum(1 for i in index if i is not None and i is not Ellipsis)
    if builtins.any(i is Ellipsis for i in index):
        j = next(i for i, v in enumerate(index) if v is Ellipsis)
        index = index[:j] + (slice(None),) * (x.ndim - consumed) + index[j+1:]
    consumed = builtins.sum(1 for i in index if i is not None)
    if consumed > x.ndim: raise IndexError("too many indices for tensor of rank %d" % x.ndim)
    return index + (slice(None),) * (x.ndim - consumed)


def _is_bool_index(i):
    return isinstance(i, bool) or (isinstance(i, ProxyTensor) and i.dtype is D.bool_)


def getitem(x, index):
    """Basic indexing (int, slice incl. negative steps, None, Ellipsis) plus one advanced index."""
    if isinstance(index, ProxyTensor) and index.dtype is D.bool_ and tuple(index.shape) == tuple(x.shape):
        values = _require_static(index, "boolean mask indexing")
        flat = x.reshape(-1) if x.ndim != 1 else x
        return index_select(flat, 0, [i for i, v in enumerate(values) if v])
    index = _normalize_index(x, index)
    spec, shape, new_axes, advanced = [], [], [], None
    out_dim = 0
    dims = iter(range(x.ndim))
    for ix in index:
        if ix is None:
            new_axes.append(out_dim)
            out_dim += 1
            continue
        d = next(dims)
        size = x.shape[d]
        if isinstance(ix, ProxyTensor) and ix.ndim == 0 and not ix.dtype.is_floating_point and static_values(ix) is not None:
            ix = int(ix.item())
        if isinstance(ix, bool):
            raise Unsupported("indexing with a Python bool")
        if isinstance(ix, int) or (hasattr(ix, "__index__") and not isinstance(ix, (ProxyTensor, list, tuple, slice))):
            ix = int(ix)
            ix = ix + size if ix < 0 else ix
            if not 0 <= ix < size: raise IndexError("tensor index out of range")
            spec.append(ix)
        elif isinstance(ix, slice):
            start, stop, step = (None if v is None else int(v) for v in (ix.start, ix.stop, ix.step))
            start, stop, step = slice(start, stop, step).indices(size)
            count = len(range(start, stop, step))
            if not count: raise ValueError("empty slices are not supported by FIR buffers")
            spec.append((start, stop, step))
            shape.append(count)
            out_dim += 1
        elif isinstance(ix, (ProxyTensor, list, tuple)):
            if advanced is not None:
                raise Unsupported("indexing with more than one index tensor (broadcast advanced indexing)")
            if isinstance(ix, ProxyTensor) and ix.dtype is D.bool_:
                values = _require_static(ix, "boolean mask indexing")
                ix = [i for i, v in enumerate(values) if v]
            spec.append((0, size, 1))
            shape.append(size)
            advanced = (out_dim, ix)
            out_dim += 1
        else:
            raise TypeError("unsupported index %r" % (ix,))
    y = x
    if builtins.any(not (isinstance(s, tuple) and s == (0, d, 1)) for s, d in zip(spec, x.shape)):
        y = x._trace.record("slice", (x.name,), tuple(shape), {"index": tuple(spec)}, dtype=x._fir())
    if advanced is not None:
        pos, ix = advanced
        t = _trace_of(y, ix)
        y, idx = _proxy(y, t), _proxy(ix, t)
        flat = idx.reshape(-1) if idx.ndim != 1 else idx
        if idx.ndim == 0: flat = idx.reshape(1)
        z = index_select(y, pos, flat)
        if idx.ndim != 1:
            z = z.reshape(tuple(z.shape[:pos]) + tuple(idx.shape) + tuple(z.shape[pos+1:]))
            new_axes = [a if a <= pos else a + idx.ndim - 1 for a in new_axes]
        y = z
    if new_axes:
        out = list(y.shape)
        for a in new_axes: out.insert(a, 1)
        y = y.reshape(tuple(out))
    return y


def index_put_basic(x, index, value):
    """Functional ``x[index] = value`` for static int/slice (step 1) indices or a bool mask."""
    t = _trace_of(x, value)
    x = _proxy(x, t)
    if isinstance(index, ProxyTensor) and index.dtype is D.bool_:
        return where(index, value, x)
    index = _normalize_index(x, index)
    if builtins.any(i is None for i in index): raise Unsupported("assignment with None indices")
    mask_rows, region, pads = [], [], []
    for d, (ix, size) in enumerate(zip(index, x.shape)):
        if isinstance(ix, int):
            ix = ix % size
            region.append(1); pads.append((ix, size - ix - 1)); mask_rows.append(range(ix, ix + 1))
        elif isinstance(ix, slice):
            start, stop, step = ix.indices(size)
            if step != 1: raise Unsupported("assignment through a strided slice")
            region.append(stop - start); pads.append((start, size - stop)); mask_rows.append(range(start, stop))
        else:
            raise Unsupported("assignment with tensor indices (scatter)")
    v = _proxy(value, t)
    squeezed = [i for i, ix in enumerate(index) if isinstance(ix, int)]
    vshape = list(broadcast(tuple(s for i, s in enumerate(region) if i not in squeezed), v.shape))
    v = v.expand(tuple(vshape)) if tuple(v.shape) != tuple(vshape) else v
    v = v.reshape(tuple(region))
    for d, (before, after) in enumerate(pads):
        parts = []
        if before: parts.append(zeros(tuple(v.shape[:d]) + (before,) + tuple(v.shape[d+1:])))
        parts.append(v)
        if after: parts.append(zeros(tuple(v.shape[:d]) + (after,) + tuple(v.shape[d+1:])))
        if len(parts) > 1: v = cat(parts, d)
    from .scalar import coordinates
    flags = [float(builtins.all(c in r for c, r in zip(ix, mask_rows))) for ix in coordinates(tuple(x.shape))]
    mask = t.constant_flat(flags, tuple(x.shape), "bool")
    return _retag(where(mask, v, x), x.dtype)


# ---- functional operators -----------------------------------------------------------

def reduce(op, x, dim=None, keepdim=False, dtype=None):
    if not isinstance(x, ProxyTensor): x = _proxy(x, _trace_of(x))
    if dim is None or dim == ():
        dims = tuple(range(x.ndim))
    else:
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
    dims = tuple(sorted(_dim(d, x.ndim) for d in dims))
    if len(set(dims)) != len(dims): raise ValueError("duplicate reduction dimension")
    if not x.ndim:
        return x if dtype is None else cast(x, dtype)
    shape = tuple(1 if i in dims else d for i, d in enumerate(x.shape) if keepdim or i not in dims)
    kind = x.dtype
    if op == "sum" and kind.category < 2: kind = D.int64
    if op == "mean" and kind.category < 2: kind = D.float32
    if dtype is not None: kind = D.dtype(dtype)
    return x._trace.record(op, (x.name,), shape, {"dims": dims, "keepdim": keepdim}, dtype=kind.fir)


def matmul(a, b):
    t = _trace_of(a, b)
    kind = D.result_type(_dtype_operand(a), _dtype_operand(b))
    a, b = _proxy(a, t), _proxy(b, t)
    if not a.ndim or not b.ndim: raise ValueError("matmul requires rank >= 1")
    aa = (1,) + a.shape if a.ndim == 1 else a.shape
    bb = b.shape + (1,) if b.ndim == 1 else b.shape
    if aa[-1] != bb[-2]: raise ValueError("matmul contracted dimension mismatch")
    shape = broadcast(aa[:-2], bb[:-2]) + (aa[-2], bb[-1])
    if a.ndim == 1: shape = shape[:-2] + shape[-1:]
    if b.ndim == 1: shape = shape[:-1]
    return t.record("matmul", (a.name, b.name), shape, dtype=kind.fir)


mm = bmm = mv = matmul


def cat(tensors, dim=0, *, axis=None):
    tensors = tuple(tensors)
    if axis is not None: dim = axis
    if not tensors: raise ValueError("cat needs tensors")
    t = _trace_of(*tensors)
    kind = D.result_type(*(_dtype_operand(x) for x in tensors))
    tensors = tuple(_proxy(x, t) for x in tensors)
    dim = _dim(dim, tensors[0].ndim)
    shape = list(tensors[0].shape)
    for x in tensors[1:]:
        if x.ndim != len(shape) or builtins.any(x.shape[i] != shape[i] for i in range(len(shape)) if i != dim):
            raise ValueError("cat dimensions mismatch")
        shape[dim] += x.shape[dim]
    if len(tensors) == 1: return _retag(tensors[0], kind)
    return t.record("cat", tuple(x.name for x in tensors), tuple(shape), {"dim": dim}, dtype=kind.fir)


concat = concatenate = cat


def stack(tensors, dim=0):
    tensors = tuple(tensors)
    if not tensors: raise ValueError("stack needs tensors")
    ndim = len(tensors[0].shape) + 1
    dim = dim % ndim
    return cat([x.unsqueeze(dim) if isinstance(x, ProxyTensor) else _proxy(x, _trace_of(*tensors)).unsqueeze(dim)
                for x in tensors], dim)


def where(condition, a=None, b=None):
    if a is None and b is None:
        return nonzero(condition, as_tuple=True)
    t = _trace_of(condition, a, b)
    kind = D.result_type(_dtype_operand(a), _dtype_operand(b))
    c, a, b = (_proxy(D.finite(x) if isinstance(x, float) else x, t) for x in (condition, a, b))
    return t.record("where", (c.name, a.name, b.name), broadcast(c.shape, broadcast(a.shape, b.shape)), dtype=kind.fir)


def masked_fill(x, mask, value):
    if isinstance(value, ProxyTensor) and value.ndim: raise ValueError("masked_fill value must be a scalar")
    return _retag(where(mask, value, x), x.dtype) if isinstance(x, ProxyTensor) else where(mask, value, x)


def softmax(x, dim=-1, dtype=None, _stacklevel=None):
    if not isinstance(x, ProxyTensor): x = _proxy(x, _trace_of(x))
    kind = D.dtype(dtype) if dtype is not None else (x.dtype if x.dtype.is_floating_point else D.float32)
    return x._trace.record("softmax", (x.name,), x.shape, {"dim": _dim(dim, x.ndim)}, dtype=kind.fir)


def log_softmax(x, dim=-1, dtype=None, _stacklevel=None):
    if dtype is not None: x = cast(x, dtype)
    return x - logsumexp(x, dim, keepdim=True)


def cumsum(x, dim=0, dtype=None):
    if not isinstance(x, ProxyTensor): x = _proxy(x, _trace_of(x))
    kind = D.dtype(dtype) if dtype is not None else (D.int64 if x.dtype.category < 2 else x.dtype)
    return x._trace.record("cumsum", (x.name,), x.shape, {"dim": _dim(dim, x.ndim)}, dtype=kind.fir)


def cumprod(x, dim=0, dtype=None):
    values = static_values(x)
    if values is None:
        raise Unsupported("cumprod of a runtime tensor (exp(cumsum(log(x))) holds for positive x only)")
    out = list(values)
    dim = _dim(dim, x.ndim)
    from .scalar import coordinates, offset
    shape = tuple(x.shape)
    for ix in coordinates(shape):
        if ix[dim]:
            prev = ix[:dim] + (ix[dim] - 1,) + ix[dim+1:]
            out[offset(ix, shape)] *= out[offset(prev, shape)]
    return _static_tensor(out, shape, x.dtype, (x,))


def scan(step, xs, initial, dim=0):
    """Static sequential scan. Returns (stacked outputs, final carry)."""
    dim %= xs.ndim
    carry, outputs = initial, []
    for i in range(xs.shape[dim]):
        ix = [slice(None)] * xs.ndim
        ix[dim] = i
        carry = step(carry, xs[tuple(ix)])
        outputs.append(carry.reshape((1,) + tuple(carry.shape)))
    return cat(outputs, 0), carry


def index_select(x, dim, index):
    t = _trace_of(x, index)
    x, index = _proxy(x, t), _proxy(index, t)
    if index.ndim == 0: index = index.reshape(1)
    if index.ndim != 1: raise ValueError("index_select requires a vector index")
    dim = _dim(dim, x.ndim)
    shape = list(x.shape)
    shape[dim] = index.numel()
    return t.record("index_select", (x.name, index.name), tuple(shape), {"dim": dim}, dtype=x._fir())


def gather(x, dim, index):
    """torch.gather: per-element gather, supported when the index is a constant."""
    values = static_values(index) if isinstance(index, ProxyTensor) else None
    if values is None and isinstance(index, (list, tuple)):
        index = tensor(index)
        values = static_values(index)
    if values is None:
        raise Unsupported("gather with a runtime index tensor (per-element dynamic gather) has no FIR operation")
    dim = _dim(dim, x.ndim)
    from .scalar import coordinates, offset
    shape = tuple(index.shape)
    flat_index = []
    for ix, v in zip(coordinates(shape), values):
        src = ix[:dim] + (int(v),) + ix[dim+1:]
        flat_index.append(offset(src, tuple(x.shape)))
    return index_select(x.reshape(-1), 0, flat_index).reshape(shape)


def rotary_embedding(x, position, base=10000.0):
    if x.shape[-1] % 2: raise ValueError("RoPE head dimension must be even")
    t = _trace_of(x, position)
    x, position = _proxy(x, t), _proxy(position, t)
    if broadcast(x.shape[:-1], position.shape) != x.shape[:-1]:
        raise ValueError("RoPE position must broadcast to the head/sequence dimensions")
    if base <= 0: raise ValueError("RoPE base must be positive")
    return t.record("rotary", (x.name, position.name), x.shape, {"base": float(base)}, dtype=x._fir())


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                                 enable_gqa=False):
    """torch SDPA semantics: bool masks select, float masks add; GQA by explicit head slices."""
    if dropout_p:
        raise Unsupported("scaled_dot_product_attention with dropout_p > 0 needs an RNG; pass dropout_p=0.0 for inference")
    if builtins.min(q.ndim, k.ndim, v.ndim) < 3: raise ValueError("attention expects [..., heads, sequence, width]")
    if (k.shape[-2] != v.shape[-2] or q.shape[-1] != k.shape[-1]
            or k.shape[-3] != v.shape[-3]): raise ValueError("attention shape mismatch")
    if q.shape[-3] != k.shape[-3]:
        if not enable_gqa or q.shape[-3] % k.shape[-3] or k.shape[-3] != v.shape[-3]:
            raise ValueError("grouped-query attention requires divisible query/KV heads")
        group = q.shape[-3] // k.shape[-3]
        # Static slices keep grouped-query semantics explicit in the shared IR.
        def expand(x):
            parts = []
            for h in range(x.shape[-3]):
                ix = [slice(None)] * x.ndim
                ix[-3] = slice(h, h+1)
                parts.extend([x[tuple(ix)]] * group)
            return cat(parts, -3)
        k, v = expand(k), expand(v)
    scores = (q @ k.transpose(-1, -2)) * (scale if scale is not None else 1/math.sqrt(q.shape[-1]))
    masks = []
    if is_causal:
        nq, nk = q.shape[-2], k.shape[-2]
        mask = tensor([[j <= i for j in range(nk)] for i in range(nq)], "bool")
        scores = where(mask, scores, -3e38)
        masks.append(mask)
    if attn_mask is not None:
        is_bool = getattr(attn_mask, "dtype", None) == "bool"
        scores = where(attn_mask, scores, -3e38) if is_bool else scores + attn_mask
        if is_bool: masks.append(attn_mask)
    probabilities = softmax(scores, -1)
    for mask in masks:
        probabilities = where(mask, probabilities, 0)
    return probabilities @ v


def arange(start=0, end=None, step=1, *, out=None, dtype=None, device=None, layout=None,
           requires_grad=False, pin_memory=False):
    """torch.arange: ``arange(end)``, ``arange(start, end)``, ``arange(start, end, step)``."""
    if end is None: start, end = 0, start
    bounds = [b.item() if isinstance(b, ProxyTensor) else b for b in (start, end, step)]
    start, end, step = bounds
    if step == 0: raise ValueError("arange step must be nonzero")
    count = builtins.max(0, math.ceil((end - start) / step))
    if not count: raise ValueError("empty arange is not supported by FIR buffers")
    floating = builtins.any(isinstance(v, float) for v in bounds)
    kind = D.dtype(dtype) if dtype is not None else (D.float32 if floating else D.int64)
    values = [start + i * step for i in range(count)]
    if kind.category < 2: values = [math.trunc(v) for v in values]
    return _trace_of().constant_flat([float(v) for v in values], (count,), kind.fir)


def range_(start=0, end=None, step=1, *, dtype=None, device=None, requires_grad=False):
    """torch.range (end inclusive, deprecated in torch)."""
    if end is None: start, end = 0, start
    return arange(start, end + step / 2 if isinstance(step, float) or isinstance(end, float) else end + 1, step,
                  dtype=dtype or D.float32)


def linspace(start, end, steps, *, dtype=None, device=None, requires_grad=False):
    if steps == 1: values = [float(start)]
    else: values = [start + (end - start) * i / (steps - 1) for i in range(steps)]
    return _trace_of().constant_flat(values, (steps,), (D.dtype(dtype) or D.float32).fir)


def full(size, fill_value, *, dtype=None, device=None, layout=None, requires_grad=False, out=None):
    size = (size,) if isinstance(size, int) else tuple(size)
    if isinstance(fill_value, ProxyTensor):
        value = fill_value if dtype is None else cast(fill_value, dtype)
        return value.reshape(()).expand(size) if size else value.reshape(())
    kind = D.dtype(dtype) if dtype is not None else D.of_scalar(fill_value)
    value = D.finite(fill_value)
    if kind.category == 1: value = math.trunc(value)
    if kind is D.bool_: value = float(bool(value))
    return _trace_of().constant_flat([float(value)] * math.prod(size), size, kind.fir)


def zeros(*size, dtype=None, device=None, layout=None, requires_grad=False, out=None):
    return full(_sizes(size), 0.0, dtype=dtype or D.float32)


def ones(*size, dtype=None, device=None, layout=None, requires_grad=False, out=None):
    return full(_sizes(size), 1.0, dtype=dtype or D.float32)


def empty(*size, dtype=None, device=None, layout=None, requires_grad=False, pin_memory=False, memory_format=None, out=None):
    from .tracer import active_trace
    if active_trace() is None: return Uninitialized(_sizes(size), dtype)
    return zeros(*size, dtype=dtype)


def eye(n, m=None, *, dtype=None, device=None, requires_grad=False):
    m = n if m is None else m
    return _trace_of().constant_flat([float(i == j) for i in range(n) for j in range(m)], (n, m),
                                     (D.dtype(dtype) or D.float32).fir)


def zeros_like(x, *, dtype=None, device=None, requires_grad=False, memory_format=None):
    return full(tuple(x.shape), 0.0, dtype=dtype or x.dtype)


def ones_like(x, *, dtype=None, device=None, requires_grad=False, memory_format=None):
    return full(tuple(x.shape), 1.0, dtype=dtype or x.dtype)


def full_like(x, fill_value, *, dtype=None, device=None, requires_grad=False, memory_format=None):
    return full(tuple(x.shape), fill_value, dtype=dtype or x.dtype)


empty_like = zeros_like


for _name in ("silu", "relu", "tanh", "sigmoid", "exp", "log", "sqrt", "rsqrt", "sin", "cos"):
    def _make(name):
        return lambda x: unary(name, x)
    globals()[_name] = _make(_name)


def softplus(x, beta=1, threshold=20):
    """VM softplus; beta != 1 is softplus(beta*x)/beta (threshold is not modelled: exact softplus everywhere)."""
    if beta == 1: return unary("softplus", x)
    return unary("softplus", x * beta) / beta


_GELU = {"tanh": "gelu_new", "none": "gelu", "erf": "gelu", "quick": "gelu_quick"}


def gelu(x, approximate="tanh"):
    """GELU: ``tanh`` (GPT-2 gelu_new), ``none``/``erf`` (exact) or ``quick`` (x*sigmoid(1.702x))."""
    if approximate not in _GELU: raise ValueError("gelu approximate must be tanh, none, erf or quick")
    return unary(_GELU[approximate], x)


def gelu_new(x): return unary("gelu_new", x)


def abs(x):
    if isinstance(x, (int, float)): return builtins.abs(x)
    return unary("abs", x)


def neg(x): return unary("neg", x)


def maximum(a, b): return binary("maximum", a, b)


def minimum(a, b): return binary("minimum", a, b)


def add(a, b, alpha=1):
    return binary("add", a, b if alpha == 1 else b * alpha)


def sub(a, b, alpha=1):
    return binary("sub", a, b if alpha == 1 else b * alpha)


subtract = sub


def mul(a, b): return binary("mul", a, b)


multiply = mul


def pow(a, b):
    if not _is_tensor(a) and not _is_tensor(b): return a ** b
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        if b == 0.5: return sqrt(a)
        if b == -0.5: return rsqrt(a)
        if b == 1: return a if isinstance(a, ProxyTensor) else _proxy(a, _trace_of(a))
    if isinstance(a, (int, float)) and not isinstance(a, bool) and _is_tensor(b):
        if a <= 0: raise Unsupported("scalar ** tensor needs a positive base (exp(b * log(a)))")
        return exp(b * math.log(a))
    return binary("pow", a, b)


def div(a, b, rounding_mode=None):
    if rounding_mode is None: return binary("div", a, b)
    if rounding_mode not in ("floor", "trunc"): raise ValueError("rounding_mode must be None, 'floor' or 'trunc'")
    fn = math.floor if rounding_mode == "floor" else math.trunc
    def apply(x, y):
        if y == 0: raise ZeroDivisionError("division by zero in a constant tensor")
        return fn(x / y)
    return _static_binary(a, b, apply, "div(rounding_mode=%r)" % rounding_mode, keep_int=True)


true_divide = divide = div


def floor_divide(a, b): return div(a, b, rounding_mode="floor")


def remainder(a, b):
    return _static_binary(a, b, lambda x, y: x - math.floor(x / y) * y, "remainder (%)", keep_int=True)


def fmod(a, b):
    return _static_binary(a, b, lambda x, y: math.fmod(x, y), "fmod", keep_int=True)


def _static_binary(a, b, fn, what, keep_int=False):
    t = _trace_of(a, b)
    kind = D.result_type(_dtype_operand(a), _dtype_operand(b))
    pa, pb = _proxy(a, t), _proxy(b, t)
    va, vb = static_values(pa), static_values(pb)
    if va is None or vb is None:
        raise Unsupported(what + " of a runtime tensor needs a rounding op, which FIR does not have; "
                          "it is evaluated at trace time for constant tensors only")
    shape = broadcast(pa.shape, pb.shape)
    from .scalar import coordinates, offset, broadcast_index
    out = []
    for ix in coordinates(shape):
        x = va[offset(broadcast_index(ix, pa.shape), pa.shape)]
        y = vb[offset(broadcast_index(ix, pb.shape), pb.shape)]
        out.append(fn(x, y))
    return t.constant_flat([float(v) for v in out], shape, kind.fir)


def _static_unary(x, fn, what, dtype=None):
    t = _trace_of(x)
    x = _proxy(x, t)
    values = static_values(x)
    if values is None:
        raise Unsupported(what + " of a runtime tensor has no FIR operation; constant tensors are folded at trace time")
    return t.constant_flat([float(fn(v)) for v in values], tuple(x.shape), (D.dtype(dtype) or x.dtype).fir)


def floor(x): return x if not x.dtype.is_floating_point else _static_unary(x, math.floor, "floor")
def ceil(x): return x if not x.dtype.is_floating_point else _static_unary(x, math.ceil, "ceil")
def trunc(x): return x if not x.dtype.is_floating_point else _static_unary(x, math.trunc, "trunc")


def round(x, decimals=0):
    if not _is_tensor(x): return builtins.round(x, decimals)
    scale = 10 ** decimals
    def half_even(v): return builtins.round(v * scale) / scale
    return x if not x.dtype.is_floating_point else _static_unary(x, half_even, "round")


def erf(x): return _static_unary(x, math.erf, "erf (use gelu(approximate='none') for exact GELU)", D.float32)


def reciprocal(x): return binary("div", 1.0, x)


def square(x): return binary("mul", x, x)


def sign(x):
    return _retag(where(x > 0, 1.0, 0.0) - where(x < 0, 1.0, 0.0), x.dtype)


sgn = sign


def exp2(x): return exp(x * math.log(2.0))
def expm1(x): return exp(x) - 1
def log2(x): return log(x) * (1 / math.log(2.0))
def log10(x): return log(x) * (1 / math.log(10.0))
def log1p(x): return log(x + 1)
def tan(x): return sin(x) / cos(x)
def sinh(x): return (exp(x) - exp(-x)) * 0.5
def cosh(x): return (exp(x) + exp(-x)) * 0.5


def clamp(x, min=None, max=None):
    if min is None and max is None: raise ValueError("clamp needs min or max")
    y = x
    if min is not None: y = maximum(y, D.finite(min) if isinstance(min, float) else min)
    if max is not None: y = minimum(y, D.finite(max) if isinstance(max, float) else max)
    return y


clip = clamp


def clamp_min(x, min): return clamp(x, min, None)
def clamp_max(x, max): return clamp(x, None, max)


def lerp(a, b, weight): return a + weight * (b - a)


def outer(a, b): return a.reshape(-1, 1) * b.reshape(1, -1)


ger = outer


def logical(op, a, b):
    """Bool ``&``, ``|``, ``^`` on 0/1 masks; integer bitwise ops on constant tensors."""
    da, db = (_dtype_operand(x) for x in (a, b))
    kinds = [x.dtype if isinstance(x, _Operand) else D.of_scalar(x) for x in (da, db)]
    if builtins.all(k is D.bool_ for k in kinds):
        if op == "and": r = binary("mul", a, b)
        elif op == "or": r = binary("maximum", a, b)
        else: r = binary("ne", a, b)
        return cast(r, D.bool_) if r.dtype is not D.bool_ else r
    fn = {"and": lambda x, y: int(x) & int(y), "or": lambda x, y: int(x) | int(y), "xor": lambda x, y: int(x) ^ int(y)}[op]
    if builtins.any(k.category == 2 for k in kinds): raise TypeError("bitwise %s is not defined for floating point tensors" % op)
    return _static_binary(a, b, fn, "integer bitwise " + op)


def logical_and(a, b): return logical("and", cast(a, D.bool_), cast(b, D.bool_))
def logical_or(a, b): return logical("or", cast(a, D.bool_), cast(b, D.bool_))
def logical_xor(a, b): return logical("xor", cast(a, D.bool_), cast(b, D.bool_))


def logical_not(x):
    r = binary("eq", x, 0)
    return r


def bitwise_not(x):
    if x.dtype is D.bool_: return logical_not(x)
    return _static_unary(x, lambda v: ~int(v), "integer bitwise not")


bitwise_and, bitwise_or, bitwise_xor = (lambda a, b: logical("and", a, b), lambda a, b: logical("or", a, b),
                                        lambda a, b: logical("xor", a, b))


def reshape(x, *shape): return x.reshape(*shape)
def transpose(x, a, b): return x.transpose(a, b)
def permute(x, *dims): return x.permute(*dims)
def chunk(x, chunks, dim=0): return x.chunk(chunks, dim)
def split(x, sizes, dim=0): return x.split(sizes, dim)
def flatten(x, start_dim=0, end_dim=-1): return x.flatten(start_dim, end_dim)
def unflatten(x, dim, sizes): return x.unflatten(dim, sizes)
def squeeze(x, dim=None): return x.squeeze(dim)
def unsqueeze(x, dim): return x.unsqueeze(dim)
def unbind(x, dim=0): return x.unbind(dim)
def narrow(x, dim, start, length): return x.narrow(dim, start, length)
def select(x, dim, index): return x.select(dim, index)
def movedim(x, source, destination): return x.movedim(source, destination)
def swapaxes(x, a, b): return x.transpose(a, b)
def broadcast_to(x, shape): return x.expand(tuple(shape))
def masked_select(x, mask): return getitem(x, mask)


def broadcast_tensors(*tensors):
    shape = ()
    for x in tensors: shape = broadcast(shape, tuple(x.shape))
    return tuple(x.expand(shape) for x in tensors)


def broadcast_shapes(*shapes):
    out = ()
    for s in shapes: out = broadcast(out, (s,) if isinstance(s, int) else tuple(s))
    return Size(out)


def sum(x, dim=None, keepdim=False, dtype=None): return reduce("sum", x, dim, keepdim, dtype)
def mean(x, dim=None, keepdim=False, dtype=None): return reduce("mean", x, dim, keepdim, dtype)


def amax(x, dim=(), keepdim=False): return reduce("max", x, None if dim == () else dim, keepdim)


def amin(x, dim=(), keepdim=False):
    return _retag(neg(reduce("max", neg(x), None if dim == () else dim, keepdim)), x.dtype)


def _arg_reduce(x, dim, keepdim, largest):
    if dim is None:
        flat = x.reshape(-1)
        r = _arg_reduce(flat, 0, False, largest)
        return r.reshape([1] * x.ndim) if keepdim else r
    dim = _dim(dim, x.ndim)
    n = x.shape[dim]
    best = amax(x, dim, keepdim=True) if largest else amin(x, dim, keepdim=True)
    position = _trace_of(x).constant_flat([-float(i) for i in range(n)],
                                         (n,) + (1,) * (x.ndim - dim - 1), "f32")
    # First index on ties: maximize -index over the positions holding the extreme value.
    candidate = where(x == best, position, -float(n))
    index = neg(reduce("max", candidate, dim, keepdim))
    return _retag(index, D.int64)


def argmax(x, dim=None, keepdim=False): return _arg_reduce(x, dim, keepdim, True)
def argmin(x, dim=None, keepdim=False): return _arg_reduce(x, dim, keepdim, False)


def max(x, dim=None, keepdim=False, *, out=None):
    """torch.max: full reduction, ``(values, indices)`` along ``dim``, or elementwise with a tensor."""
    if _is_tensor(dim) or isinstance(dim, (float,)): return maximum(x, dim)
    if dim is None: return reduce("max", x, None, False)
    return MaxResult(reduce("max", x, dim, keepdim), argmax(x, dim, keepdim))


def min(x, dim=None, keepdim=False, *, out=None):
    if _is_tensor(dim) or isinstance(dim, (float,)): return minimum(x, dim)
    if dim is None: return amin(x)
    return MinResult(amin(x, dim, keepdim), argmin(x, dim, keepdim))


def var(x, dim=None, unbiased=None, keepdim=False, *, correction=None):
    if isinstance(unbiased, bool) is False and unbiased is not None and correction is None:
        correction = unbiased
    if correction is None: correction = 1 if unbiased in (None, True) else 0
    dims = tuple(range(x.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    n = math.prod(x.shape[_dim(d, x.ndim)] for d in dims)
    centered = x - mean(x, dims, keepdim=True)
    total = sum(centered * centered, dims, keepdim)
    return total / builtins.max(n - correction, 0) if n - correction > 0 else total * float("nan")


def std(x, dim=None, unbiased=None, keepdim=False, *, correction=None):
    return sqrt(var(x, dim, unbiased, keepdim, correction=correction))


def var_mean(x, dim=None, unbiased=None, keepdim=False, *, correction=None):
    return VarMean(var(x, dim, unbiased, keepdim, correction=correction), mean(x, dim, keepdim))


def logsumexp(x, dim, keepdim=False):
    m = amax(x, dim, keepdim=True)
    r = m + log(sum(exp(x - m), dim, keepdim=True))
    return r if keepdim else r.squeeze(dim)


def norm(x, p="fro", dim=None, keepdim=False, dtype=None):
    if p in ("fro", 2, 2.0): return sqrt(sum(x * x, dim, keepdim))
    if p in (1, 1.0): return sum(abs(x), dim, keepdim)
    if p in (float("inf"), "inf"): return amax(abs(x), () if dim is None else dim, keepdim)
    if isinstance(p, (int, float)) and p > 0: return sum(abs(x) ** p, dim, keepdim) ** (1 / p)
    raise Unsupported("norm with p=%r" % (p,))


def any(x, dim=None, keepdim=False):
    return cast(reduce("max", cast(x, D.bool_), dim, keepdim), D.bool_)


def all(x, dim=None, keepdim=False):
    return cast(amin(cast(x, D.bool_), () if dim is None else dim, keepdim), D.bool_)


def prod(x, dim=None, keepdim=False, dtype=None):
    values = _require_static(x, "prod")
    dims = tuple(range(x.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    dims = {_dim(d, x.ndim) for d in dims}
    from .scalar import coordinates, offset
    kept = tuple(s for i, s in enumerate(x.shape) if i not in dims)
    out = {}
    for ix, v in zip(coordinates(tuple(x.shape)), values):
        key = tuple(c for i, c in enumerate(ix) if i not in dims)
        out[key] = out.get(key, 1.0) * v
    shape = tuple(1 if i in dims else s for i, s in enumerate(x.shape)) if keepdim else kept
    return _static_tensor([out[k] for k in coordinates(kept)], shape, x.dtype, (x,))


def topk(x, k, dim=-1, largest=True, sorted=True):
    values = _require_static(x, "topk")
    return _static_sort(x, values, dim, largest, k)


def sort(x, dim=-1, descending=False, stable=False):
    values = _require_static(x, "sort")
    return _static_sort(x, values, dim, descending, None)


SortResult = namedtuple("sort", ["values", "indices"])


def _static_sort(x, values, dim, descending, k):
    dim = _dim(dim, x.ndim)
    from .scalar import coordinates, offset
    shape = tuple(x.shape)
    n = shape[dim]
    k = n if k is None else k
    out_shape = shape[:dim] + (k,) + shape[dim+1:]
    vals, idxs = [0.0] * math.prod(out_shape), [0.0] * math.prod(out_shape)
    rest = shape[:dim] + (1,) + shape[dim+1:]
    for ix in coordinates(rest):
        row = [(values[offset(ix[:dim] + (j,) + ix[dim+1:], shape)], j) for j in range(n)]
        row.sort(key=lambda p: (-p[0], p[1]) if descending else (p[0], p[1]))
        for j, (v, i) in enumerate(row[:k]):
            o = offset(ix[:dim] + (j,) + ix[dim+1:], out_shape)
            vals[o], idxs[o] = v, float(i)
    return SortResult(_static_tensor(vals, out_shape, x.dtype, (x,)), _static_tensor(idxs, out_shape, D.int64, (x,)))


def nonzero(x, as_tuple=False):
    values = _require_static(x, "nonzero (data-dependent output shape)")
    from .scalar import coordinates
    hits = [ix for ix, v in zip(coordinates(tuple(x.shape)), values) if v]
    if not hits: raise ValueError("nonzero of an all-zero constant gives an empty tensor (unsupported)")
    if as_tuple:
        return tuple(_static_tensor([ix[d] for ix in hits], (len(hits),), D.int64, (x,)) for d in range(x.ndim))
    return _static_tensor([c for ix in hits for c in ix], (len(hits), x.ndim), D.int64, (x,))


def repeat(x, sizes):
    sizes = tuple(int(s) for s in sizes)
    if len(sizes) < x.ndim: raise ValueError("repeat needs at least the tensor rank")
    x = x.reshape((1,) * (len(sizes) - x.ndim) + tuple(x.shape)) if len(sizes) > x.ndim else x
    if builtins.all(s == 1 for s in sizes): return x
    inter = tuple(v for s in x.shape for v in (1, s))
    expanded = tuple(v for r, s in zip(sizes, x.shape) for v in (r, s))
    return x.reshape(inter).expand(expanded).reshape(tuple(r * s for r, s in zip(sizes, x.shape)))


def repeat_interleave(x, repeats, dim=None, *, output_size=None):
    if not isinstance(x, ProxyTensor): x = _proxy(x, _trace_of(x))
    if dim is None: x, dim = x.reshape(-1), 0
    dim = _dim(dim, x.ndim)
    if isinstance(repeats, ProxyTensor):
        values = _require_static(repeats, "repeat_interleave with tensor repeats")
        reps = [int(v) for v in values]
        if len(reps) == 1: repeats = reps[0]
        else:
            index = [i for i, r in enumerate(reps) for _ in range(r)]
            return index_select(x, dim, index)
    repeats = int(repeats)
    if repeats == 1: return x
    shape = tuple(x.shape)
    return (x.unsqueeze(dim + 1).expand(shape[:dim+1] + (repeats,) + shape[dim+1:])
            .reshape(shape[:dim] + (shape[dim] * repeats,) + shape[dim+1:]))


def _triangle(x, diagonal, lower):
    rows, cols = x.shape[-2], x.shape[-1]
    keep = [float((j - i <= diagonal) if lower else (j - i >= diagonal)) for i in range(rows) for j in range(cols)]
    mask = _trace_of(x).constant_flat(keep, (rows, cols), "bool")
    return _retag(where(mask, x, 0), x.dtype)


def tril(x, diagonal=0): return _triangle(x, diagonal, True)
def triu(x, diagonal=0): return _triangle(x, diagonal, False)


def flip(x, dims):
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    ix = [slice(None)] * x.ndim
    for d in dims: ix[_dim(d, x.ndim)] = slice(None, None, -1)
    return x[tuple(ix)]


def roll(x, shifts, dims=None):
    if dims is None:
        return roll(x.reshape(-1), shifts, 0).reshape(tuple(x.shape))
    shifts = (shifts,) if isinstance(shifts, int) else tuple(shifts)
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    for s, d in zip(shifts, dims):
        d = _dim(d, x.ndim)
        n = x.shape[d]
        s %= n
        if not s: continue
        head = [slice(None)] * x.ndim
        tail = [slice(None)] * x.ndim
        head[d], tail[d] = slice(n - s, n), slice(0, n - s)
        x = cat((x[tuple(head)], x[tuple(tail)]), d)
    return x


def diff(x, n=1, dim=-1):
    for _ in range(n):
        d = _dim(dim, x.ndim)
        a, b = [slice(None)] * x.ndim, [slice(None)] * x.ndim
        a[d], b[d] = slice(1, None), slice(None, -1)
        x = x[tuple(a)] - x[tuple(b)]
    return x


def diagonal(x, offset=0, dim1=0, dim2=1):
    d1, d2 = _dim(dim1, x.ndim), _dim(dim2, x.ndim)
    rest = [d for d in range(x.ndim) if d not in (d1, d2)]
    y = x.permute(rest + [d1, d2])
    rows, cols = y.shape[-2], y.shape[-1]
    pairs = [(i, i + offset) for i in range(rows) if 0 <= i + offset < cols]
    flat = y.reshape(tuple(y.shape[:-2]) + (rows * cols,))
    return index_select(flat, -1, [i * cols + j for i, j in pairs])


def einsum(equation, *operands):
    """torch.einsum without repeated indices inside one operand; pairwise contractions use matmul."""
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)): operands = tuple(operands[0])
    equation = equation.replace(" ", "")
    if "->" in equation: lhs, out = equation.split("->")
    else: lhs, out = equation, None
    terms = lhs.split(",")
    if len(terms) != len(operands): raise ValueError("einsum operand count mismatch")
    free = iter(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in equation)
    ell = None
    subs = []
    for term, x in zip(terms, operands):
        if "..." in term:
            n = x.ndim - (len(term) - 3)
            if ell is None: ell = "".join(next(free) for _ in range(n))
            if n > len(ell): ell = "".join(next(free) for _ in range(n - len(ell))) + ell
            term = term.replace("...", ell[len(ell) - n:])
        if len(set(term)) != len(term): raise Unsupported("einsum with a repeated index inside one operand (diagonal)")
        if len(term) != x.ndim: raise ValueError("einsum subscript rank mismatch")
        subs.append(term)
    if out is None:
        counts = {}
        for s in subs:
            for c in s: counts[c] = counts.get(c, 0) + 1
        out = (ell or "") + "".join(sorted(c for c in counts if counts[c] == 1 and (ell is None or c not in ell)))
    elif "..." in out:
        out = out.replace("...", ell or "")
    sizes = {}
    for s, x in zip(subs, operands):
        for c, n in zip(s, x.shape):
            if sizes.get(c, n) != n and 1 not in (sizes[c], n): raise ValueError("einsum size mismatch for " + c)
            sizes[c] = builtins.max(sizes.get(c, 1), n)
    items = list(zip(subs, operands))
    def needed(exclude):
        keep = set(out)
        for j, (s, _) in enumerate(items):
            if j not in exclude: keep |= set(s)
        return keep
    # Sum out indices that appear in one operand only and nowhere else.
    for j, (s, x) in enumerate(items):
        keep = needed({j})
        drop = [i for i, c in enumerate(s) if c not in keep]
        if drop:
            x = sum(x, drop)
            s = "".join(c for c in s if c not in [s[i] for i in drop])
            items[j] = (s, x)
    while len(items) > 1:
        (sa, a), (sb, b) = items[0], items[1]
        keep = needed({0, 1})
        batch = [c for c in sa if c in sb and c in keep]
        contract = [c for c in sa if c in sb and c not in keep]
        left = [c for c in sa if c not in sb]
        right = [c for c in sb if c not in sa]
        a = a.permute([sa.index(c) for c in batch + left + contract]) if sa else a
        b = b.permute([sb.index(c) for c in batch + contract + right]) if sb else b
        size = lambda cs, x, off: [x.shape[off + i] for i in range(len(cs))]
        bshape = tuple(builtins.max(p, q) for p, q in zip(size(batch, a, 0), size(batch, b, 0)))
        la, kc, rb = (math.prod(sizes[c] for c in cs) for cs in (left, contract, right))
        a3 = a.reshape(tuple(a.shape[:len(batch)]) + (la, kc))
        b3 = b.reshape(tuple(b.shape[:len(batch)]) + (kc, rb))
        y = matmul(a3, b3).reshape(bshape + tuple(sizes[c] for c in left + right))
        items = [("".join(batch + left + right), y)] + items[2:]
    s, y = items[0]
    extra = [i for i, c in enumerate(s) if c not in out]
    if extra:
        y = sum(y, extra)
        s = "".join(c for c in s if c in out)
    return y.permute([s.index(c) for c in out]) if len(out) > 1 else y


def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    product = matmul(mat1, mat2)
    if alpha != 1: product = product * alpha
    return product + (input if beta == 1 else input * beta) if beta else product


def baddbmm(input, batch1, batch2, *, beta=1, alpha=1):
    return addmm(input, batch1, batch2, beta=beta, alpha=alpha)


def bilinear(x1, x2, weight, bias=None):
    y = einsum("...i,oij,...j->...o", x1, weight, x2)
    return y + bias if bias is not None else y
