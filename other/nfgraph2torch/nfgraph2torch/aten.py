"""ATen operation -> formers.fir tracer calls.

``replay(target, args, kwargs)`` runs one Core ATen node on ``formers.fir`` tensors. Arguments are
bound to the operator's schema names, so every handler reads ``a.self``, ``a.dim`` ... exactly as in
``native_functions.yaml``. Resolution order for an operation ``aten::NAME.OVERLOAD``:

1. ``HANDLERS["NAME.OVERLOAD"]``, then ``HANDLERS["NAME"]``  (explicit mapping, this file);
2. a micro-kernel operation of ``micro.py`` registered under NAME (``MICRO``);
3. the same-named function of ``formers.fir.torch`` / ``torch.nn.functional`` / the tensor method,
   called with the schema arguments in order (FIR mirrors the torch API).

If none exists the op is reported by ``coverage`` and ``UnsupportedOp`` is raised.
"""
from __future__ import annotations

import math
import operator

from formers.fir import torch_compat as T
from formers.fir.tensor import ProxyTensor

F = T.nn.functional
HANDLERS = {}
MICRO = {}          # filled by micro.py: op name -> callable(a)


class Args(dict):
    """Schema-bound arguments with attribute access (``a.self``, ``a.dim``)."""
    __getattr__ = dict.__getitem__


def handler(*names):
    def register(fn):
        for n in names:
            HANDLERS[n] = fn
        return fn
    return register


def _fir_dtype(torch_dtype):
    from formers.fir import dtypes
    return dtypes.dtype(str(torch_dtype).replace("torch.", "")) if torch_dtype is not None else None


def bind(target, args, kwargs):
    """Bind ``args``/``kwargs`` of an OpOverload call to its schema; fills defaults."""
    schema = target._schema
    bound = Args()
    positional = list(args)
    for arg in schema.arguments:
        if arg.kwarg_only:
            if arg.name in kwargs:
                bound[arg.name] = kwargs[arg.name]
            else:
                bound[arg.name] = arg.default_value if arg.has_default_value() else None
        elif positional:
            bound[arg.name] = positional.pop(0)
        elif arg.name in kwargs:
            bound[arg.name] = kwargs[arg.name]
        else:
            bound[arg.name] = arg.default_value if arg.has_default_value() else None
    return bound


def op_name(target):
    """('add', 'Tensor') for aten::add.Tensor."""
    name = target._schema.name.split("::")[1]
    return name, target._overloadname


def replay(target, args, kwargs):
    if target is operator.getitem:
        return args[0][args[1]]
    if not hasattr(target, "_schema"):
        raise ValueError("unsupported call target %r" % (target,))
    name, overload = op_name(target)
    a = bind(target, args, kwargs)
    fn = MICRO.get(name) or HANDLERS.get(name + "." + overload) or HANDLERS.get(name)
    if fn is not None:
        return fn(a)
    return _generic(name, target, a)


def _generic(name, target, a):
    """Same-named torch-API function, called with the schema arguments in order."""
    first = next(iter(a.values()), None)
    schema = target._schema
    pos, kw = [], {}
    for arg in schema.arguments:
        value = a[arg.name]
        if arg.name in ("layout", "device", "pin_memory", "memory_format", "out", "non_blocking"):
            continue
        if arg.name == "dtype":
            value = _fir_dtype(value)
        (kw if arg.kwarg_only else pos).append(None)
        (kw.__setitem__(arg.name, value) if arg.kwarg_only else pos.__setitem__(-1, value))
    kw = {k: v for k, v in kw.items() if not (k == "dtype" and v is None)}
    for space in (T, F):
        fn = getattr(space, name, None)
        if callable(fn) and not isinstance(fn, type):
            return fn(*pos, **kw)
    if isinstance(first, ProxyTensor) and hasattr(first, name):
        return getattr(first, name)(*pos[1:], **kw)
    raise ValueError("no FIR mapping for aten::%s.%s" % (name, target._overloadname))


def known(name, overload=""):
    """True when ``replay`` has an explicit mapping for the op (used by ``coverage``)."""
    return bool(HANDLERS.get(name + "." + overload) or HANDLERS.get(name) or MICRO.get(name))


# --------------------------------------------------------------------------------------------
# creation and identity
# --------------------------------------------------------------------------------------------
def _identity(a):
    return a.self


for _n in ("clone", "alias", "detach", "lift_fresh_copy", "contiguous", "lift_fresh", "copy_",
           "view_as_real_copy", "alias_copy", "detach_copy", "fake_quantize_per_tensor_affine"):
    HANDLERS[_n] = _identity


@handler("_assert_tensor_metadata", "_assert_scalar", "_assert_async", "sym_constrain_range_for_size", "sym_constrain_range")
def _ignored(a):
    return None


@handler("copy")
def _copy(a):
    return a.src


@handler("_to_copy", "to")
def _to_copy(a):
    d = _fir_dtype(a["dtype"])
    return a.self.to(d) if d is not None else a.self


@handler("arange.start_step", "arange.start", "arange")
def _arange(a):
    start = a.get("start", 0) if hasattr(a, "get") else 0
    end = a["end"]
    step = a["step"] if "step" in a and a["step"] is not None else 1
    return T.arange(start if start is not None else 0, end, step, dtype=_fir_dtype(a["dtype"]))


@handler("full")
def _full(a):
    return T.full(tuple(a.size), a.fill_value, dtype=_fir_dtype(a["dtype"]))


@handler("zeros")
def _zeros(a):
    return T.zeros(tuple(a.size), dtype=_fir_dtype(a["dtype"]))


@handler("ones")
def _ones(a):
    return T.ones(tuple(a.size), dtype=_fir_dtype(a["dtype"]))


@handler("empty.memory_format", "empty_strided")
def _empty(a):
    return T.zeros(tuple(a.size), dtype=_fir_dtype(a["dtype"]))


@handler("scalar_tensor")
def _scalar_tensor(a):
    return T.tensor(a.s, dtype=_fir_dtype(a["dtype"]))


@handler("full_like")
def _full_like(a):
    return T.full_like(a.self, a.fill_value, dtype=_fir_dtype(a["dtype"]))


@handler("zeros_like")
def _zeros_like(a):
    return T.zeros_like(a.self, dtype=_fir_dtype(a["dtype"]))


@handler("ones_like")
def _ones_like(a):
    return T.ones_like(a.self, dtype=_fir_dtype(a["dtype"]))


@handler("fill.Scalar", "fill.Tensor")
def _fill(a):
    return T.full_like(a.self, a.value)


# --------------------------------------------------------------------------------------------
# arithmetic
# --------------------------------------------------------------------------------------------
@handler("add.Tensor", "add.Scalar", "add")
def _add(a):
    return T.add(a.self, a.other, alpha=a["alpha"]) if a["alpha"] not in (None, 1) else a.self + a.other


@handler("sub.Tensor", "sub.Scalar", "sub")
def _sub(a):
    return T.sub(a.self, a.other, alpha=a["alpha"]) if a["alpha"] not in (None, 1) else a.self - a.other


@handler("rsub.Tensor", "rsub.Scalar", "rsub")
def _rsub(a):
    alpha = a["alpha"] if a["alpha"] is not None else 1
    return a.other - a.self * alpha


@handler("mul.Tensor", "mul.Scalar", "mul")
def _mul(a):
    return a.self * a.other


@handler("div.Tensor", "div.Scalar")
def _div(a):
    return a.self / a.other


@handler("div.Tensor_mode", "div.Scalar_mode")
def _div_mode(a):
    return T.div(a.self, a.other, rounding_mode=a.rounding_mode)


@handler("floor_divide")
def _floor_divide(a):
    return T.floor_divide(a.self, a.other)


@handler("pow.Tensor_Scalar", "pow.Tensor_Tensor", "pow.Scalar")
def _pow(a):
    return T.pow(a.self if "self" in a else a.exponent, a.exponent)


@handler("neg")
def _neg(a):
    return -a.self


@handler("clamp")
def _clamp(a):
    return T.clamp(a.self, a["min"], a["max"])


@handler("clamp.Tensor")
def _clamp_t(a):
    return T.clamp(a.self, a["min"], a["max"])


@handler("maximum")
def _maximum(a):
    return T.maximum(a.self, a.other)


@handler("minimum")
def _minimum(a):
    return T.minimum(a.self, a.other)


@handler("where.self", "where.ScalarSelf", "where.ScalarOther", "where.Scalar")
def _where(a):
    return T.where(a.condition, a.self, a.other)


@handler("addcmul")
def _addcmul(a):
    return a.self + a.value * (a.tensor1 * a.tensor2)


@handler("addcdiv")
def _addcdiv(a):
    return a.self + a.value * (a.tensor1 / a.tensor2)


@handler("lerp.Scalar", "lerp.Tensor")
def _lerp(a):
    return a.self + a.weight * (a.end - a.self)


# comparison and logic -----------------------------------------------------------------------
for _name, _fn in (("eq", operator.eq), ("ne", operator.ne), ("lt", operator.lt), ("le", operator.le),
                   ("gt", operator.gt), ("ge", operator.ge)):
    HANDLERS[_name] = (lambda fn: lambda a: fn(a.self, a.other))(_fn)

for _name, _fn in (("bitwise_and", T.bitwise_and), ("bitwise_or", T.bitwise_or), ("bitwise_xor", T.bitwise_xor),
                   ("logical_and", T.logical_and), ("logical_or", T.logical_or), ("logical_xor", T.logical_xor)):
    HANDLERS[_name] = (lambda fn: lambda a: fn(a.self, a.other))(_fn)

HANDLERS["bitwise_not"] = lambda a: T.bitwise_not(a.self)
HANDLERS["logical_not"] = lambda a: T.logical_not(a.self)


# --------------------------------------------------------------------------------------------
# unary math: same-named torch functions (arguments are just `self`)
# --------------------------------------------------------------------------------------------
for _n in ("abs", "exp", "log", "sqrt", "rsqrt", "sin", "cos", "tan", "tanh", "sigmoid", "sign", "reciprocal",
           "expm1", "log1p", "log2", "log10", "exp2", "isfinite", "relu", "sinh", "cosh", "square", "sgn"):
    HANDLERS[_n] = (lambda n: lambda a: getattr(T, n)(a.self))(_n)

HANDLERS["silu"] = lambda a: F.silu(a.self)
HANDLERS["gelu"] = lambda a: F.gelu(a.self, a["approximate"] or "none")
HANDLERS["relu6"] = lambda a: F.relu6(a.self)
HANDLERS["leaky_relu"] = lambda a: F.leaky_relu(a.self, a.negative_slope)
HANDLERS["elu"] = lambda a: F.elu(a.self, a["alpha"])
HANDLERS["softplus"] = lambda a: F.softplus(a.self, a.beta, a.threshold)
HANDLERS["hardtanh"] = lambda a: F.hardtanh(a.self, a.min_val, a.max_val)
HANDLERS["hardswish"] = lambda a: F.hardswish(a.self)
HANDLERS["hardsigmoid"] = lambda a: F.hardsigmoid(a.self)
HANDLERS["mish"] = lambda a: F.mish(a.self)
HANDLERS["silu_"] = HANDLERS["silu"]
HANDLERS["glu"] = lambda a: F.glu(a.self, a.dim)


# --------------------------------------------------------------------------------------------
# reductions
# --------------------------------------------------------------------------------------------
def _dims(dim):
    return None if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0) else dim


@handler("sum.dim_IntList", "sum")
def _sum(a):
    return T.sum(a.self, _dims(a["dim"]) if "dim" in a else None, a["keepdim"] if "keepdim" in a else False)


@handler("mean.dim", "mean")
def _mean(a):
    return T.mean(a.self, _dims(a["dim"]) if "dim" in a else None, a["keepdim"] if "keepdim" in a else False)


@handler("amax")
def _amax(a):
    return T.amax(a.self, _dims(a["dim"]), a["keepdim"])


@handler("amin")
def _amin(a):
    return T.amin(a.self, _dims(a["dim"]), a["keepdim"])


@handler("max.dim")
def _max_dim(a):
    return T.max(a.self, a.dim, a["keepdim"])


@handler("min.dim")
def _min_dim(a):
    return T.min(a.self, a.dim, a["keepdim"])


@handler("max")
def _max_all(a):
    return T.max(a.self)


@handler("min")
def _min_all(a):
    return T.min(a.self)


@handler("argmax")
def _argmax(a):
    return T.argmax(a.self, a["dim"], a["keepdim"])


@handler("argmin")
def _argmin(a):
    return T.argmin(a.self, a["dim"], a["keepdim"])


@handler("var.correction", "var.dim")
def _var(a):
    correction = a["correction"] if "correction" in a and a["correction"] is not None else 1
    return T.var(a.self, _dims(a["dim"]), correction=correction, keepdim=a["keepdim"])


@handler("var_mean.correction")
def _var_mean(a):
    correction = a["correction"] if a["correction"] is not None else 1
    dim = _dims(a["dim"])
    return T.var(a.self, dim, correction=correction, keepdim=a["keepdim"]), T.mean(a.self, dim, a["keepdim"])


@handler("any.dim", "any")
def _any(a):
    return T.any(a.self) if "dim" not in a else T.any(a.self, a.dim, a["keepdim"])


@handler("all.dim", "all")
def _all(a):
    return T.all(a.self) if "dim" not in a else T.all(a.self, a.dim, a["keepdim"])


@handler("cumsum")
def _cumsum(a):
    return T.cumsum(a.self, a.dim)


@handler("logsumexp")
def _logsumexp(a):
    return T.logsumexp(a.self, a.dim, a["keepdim"])


@handler("softmax.int", "_softmax")
def _softmax(a):
    return T.softmax(a.self, a.dim)


@handler("log_softmax.int", "_log_softmax")
def _log_softmax(a):
    return T.log_softmax(a.self, a.dim)


# --------------------------------------------------------------------------------------------
# shape and views
# --------------------------------------------------------------------------------------------
@handler("view", "reshape", "_unsafe_view", "_reshape_alias")
def _view(a):
    return a.self.reshape(tuple(a["size"] if "size" in a else a["shape"]))


@handler("permute")
def _permute(a):
    return a.self.permute(*a.dims)


@handler("transpose.int")
def _transpose(a):
    return a.self.transpose(a.dim0, a.dim1)


@handler("t")
def _t(a):
    return a.self.t()


@handler("squeeze.dim", "squeeze.dims", "squeeze")
def _squeeze(a):
    d = a["dim"] if "dim" in a else None
    if d is None:
        return a.self.squeeze()
    for x in (d if isinstance(d, (list, tuple)) else [d]):
        pass
    return a.self.squeeze(d) if not isinstance(d, (list, tuple)) else _squeeze_many(a.self, d)


def _squeeze_many(x, dims):
    dims = sorted((d % x.ndim for d in dims), reverse=True)
    for d in dims:
        if x.shape[d] == 1:
            x = x.squeeze(d)
    return x


@handler("unsqueeze")
def _unsqueeze(a):
    return a.self.unsqueeze(a.dim)


@handler("expand")
def _expand(a):
    return a.self.expand(*tuple(a["size"]))


@handler("flatten.using_ints")
def _flatten(a):
    return a.self.flatten(a.start_dim, a.end_dim)


@handler("unflatten.int")
def _unflatten(a):
    return a.self.unflatten(a.dim, tuple(a.sizes))


@handler("slice.Tensor")
def _slice(a):
    ix = [slice(None)] * a.self.ndim
    end = a["end"]
    ix[a["dim"]] = slice(a["start"], None if end is not None and end >= 2 ** 62 else end, a["step"] or 1)
    return a.self[tuple(ix)]


@handler("select.int")
def _select(a):
    return a.self.select(a.dim, a.index)


@handler("narrow")
def _narrow(a):
    return a.self.narrow(a.dim, a.start, a.length)


@handler("split_with_sizes")
def _split_with_sizes(a):
    return T.split(a.self, list(a.split_sizes), a["dim"])


@handler("split.Tensor")
def _split(a):
    return T.split(a.self, a.split_size, a["dim"])


@handler("chunk")
def _chunk(a):
    return T.chunk(a.self, a.chunks, a["dim"])


@handler("unbind.int")
def _unbind(a):
    return T.unbind(a.self, a["dim"])


@handler("cat")
def _cat(a):
    return T.cat(list(a.tensors), a["dim"])


@handler("stack")
def _stack(a):
    return T.stack(list(a.tensors), a["dim"])


@handler("repeat")
def _repeat(a):
    return a.self.repeat(*tuple(a.repeats))


@handler("roll")
def _roll(a):
    return T.roll(a.self, tuple(a.shifts), tuple(a.dims))


@handler("flip")
def _flip(a):
    return T.flip(a.self, tuple(a.dims))


@handler("tril")
def _tril(a):
    return T.tril(a.self, a["diagonal"])


@handler("triu")
def _triu(a):
    return T.triu(a.self, a["diagonal"])


@handler("index_select")
def _index_select(a):
    return T.index_select(a.self, a.dim, a.index)


@handler("gather")
def _gather(a):
    return T.gather(a.self, a.dim, a.index)


@handler("embedding")
def _embedding(a):
    return F.embedding(a.indices, a.weight)


@handler("masked_fill.Scalar", "masked_fill.Tensor")
def _masked_fill(a):
    return T.masked_fill(a.self, a.mask, a.value)


# --------------------------------------------------------------------------------------------
# linear algebra and neural-network layers
# --------------------------------------------------------------------------------------------
@handler("mm")
def _mm(a):
    return T.matmul(a.self, a.mat2)


@handler("bmm")
def _bmm(a):
    return T.matmul(a.self, a.mat2)


@handler("matmul")
def _matmul(a):
    return T.matmul(a.self, a.other)


@handler("addmm")
def _addmm(a):
    return T.addmm(a.self, a.mat1, a.mat2, beta=a["beta"], alpha=a["alpha"])


@handler("baddbmm")
def _baddbmm(a):
    return T.baddbmm(a.self, a.batch1, a.batch2, beta=a["beta"], alpha=a["alpha"])


@handler("linear")
def _linear(a):
    return F.linear(a.input, a.weight, a["bias"])


@handler("native_layer_norm")
def _native_layer_norm(a):
    out = F.layer_norm(a.input, tuple(a.normalized_shape), a["weight"], a["bias"], a.eps)
    return out, None, None


@handler("native_group_norm")
def _native_group_norm(a):
    n, c = a.input.shape[0], a.input.shape[1]
    out = F.group_norm(a.input, a.group, a["weight"], a["bias"], a.eps)
    return out, None, None


@handler("_native_batch_norm_legit_no_training")
def _bn_legit_no_training(a):
    out = F.batch_norm(a.input, a.running_mean, a.running_var, a["weight"], a["bias"], False, a.momentum, a.eps)
    return out, None, None


@handler("_native_batch_norm_legit")
def _bn_legit(a):
    out = F.batch_norm(a.input, a["running_mean"], a["running_var"], a["weight"], a["bias"], False, 0.0, a.eps)
    return out, None, None


@handler("native_batch_norm")
def _native_bn(a):
    out = F.batch_norm(a.input, a["running_mean"], a["running_var"], a["weight"], a["bias"], False, 0.0, a.eps)
    return out, None, None


@handler("native_dropout")
def _native_dropout(a):
    return a.input, None


@handler("dropout", "feature_dropout")
def _dropout(a):
    return a.input


@handler("scaled_dot_product_attention")
def _sdpa(a):
    return F.scaled_dot_product_attention(a.query, a.key, a.value, a["attn_mask"], a["dropout_p"] or 0.0,
                                          a["is_causal"], scale=a["scale"])


@handler("constant_pad_nd")
def _constant_pad(a):
    return F.pad(a.self, tuple(a.pad), "constant", a["value"])
