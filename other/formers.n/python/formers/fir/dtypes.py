"""Torch-shaped dtype objects and promotion rules for traced tensors (stdlib only).

FIR buffers hold f32 values. Logical dtypes are tracked on traced values so that
torch-style code (``x.dtype == torch.bfloat16``, ``torch.finfo(x.dtype).min``,
promotion of int/bool tensors) behaves like PyTorch; the values themselves are
computed in f32, integers are exact below 2**24 and bools are stored as 0/1.
"""

import math


class DType:
    """A dtype tag. Compares equal to itself, its FIR name and its torch name."""

    __slots__ = ("name", "fir", "category", "rank", "itemsize", "is_signed")

    def __init__(self, name, fir, category, rank, itemsize, is_signed=True):
        self.name, self.fir, self.category, self.rank = name, fir, category, rank
        self.itemsize, self.is_signed = itemsize, is_signed

    @property
    def is_floating_point(self): return self.category == 2

    @property
    def is_complex(self): return False

    def to_real(self): return self

    def __repr__(self): return "torch." + self.name

    __str__ = __repr__

    def __eq__(self, other):
        if isinstance(other, DType): return self is other
        if isinstance(other, str): return other in (self.fir, self.name, "torch." + self.name) or ALIASES.get(other) is self
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self): return hash(self.name)

    def __reduce__(self): return (dtype, (self.fir,))


# category: 0 bool, 1 integer, 2 floating point
bool_ = DType("bool", "bool", 0, 0, 1, False)
uint8 = DType("uint8", "u8", 1, 1, 1, False)
int8 = DType("int8", "i8", 1, 1, 1)
int16 = DType("int16", "i16", 1, 2, 2)
int32 = DType("int32", "i32", 1, 3, 4)
int64 = DType("int64", "i64", 1, 4, 8)
float16 = DType("float16", "f16", 2, 1, 2)
bfloat16 = DType("bfloat16", "bf16", 2, 1, 2)
float32 = DType("float32", "f32", 2, 2, 4)
float64 = DType("float64", "f64", 2, 3, 8)

ALL = (bool_, uint8, int8, int16, int32, int64, float16, bfloat16, float32, float64)
BY_FIR = {d.fir: d for d in ALL}
BY_FIR["q4"] = float32  # q4 checkpoint storage is dequantized to f32 values
ALIASES = {"float": float32, "half": float16, "double": float64, "long": int64, "int": int32,
           "short": int16, "bool": bool_, "float32": float32, "float16": float16,
           "bfloat16": bfloat16, "float64": float64, "int64": int64, "int32": int32,
           "int16": int16, "int8": int8, "uint8": uint8}

DEFAULT_FLOAT = float32
DEFAULT_INT = int64


def dtype(value):
    """Normalize a DType, FIR dtype string, torch name or Python type to a DType."""
    if value is None: return None
    if isinstance(value, DType): return value
    if value is float: return DEFAULT_FLOAT
    if value is int: return DEFAULT_INT
    if value is bool: return bool_
    if isinstance(value, str):
        key = value[6:] if value.startswith("torch.") else value
        if key in BY_FIR: return BY_FIR[key]
        if key in ALIASES: return ALIASES[key]
    name = getattr(value, "name", None) or str(value)
    if isinstance(name, str) and name != value:
        try: return dtype(name)
        except TypeError: pass
    raise TypeError("unknown dtype: %r" % (value,))


def of_scalar(x):
    if isinstance(x, bool): return bool_
    if isinstance(x, int): return DEFAULT_INT
    return DEFAULT_FLOAT


def promote_types(a, b):
    """torch.promote_types for the supported dtypes."""
    a, b = dtype(a), dtype(b)
    if a is b: return a
    if a.category != b.category: return a if a.category > b.category else b
    if a.category == 2:
        if {a, b} == {float16, bfloat16}: return float32
        return a if a.rank > b.rank else b
    if a.category == 1:
        if uint8 in (a, b):
            other = b if a is uint8 else a
            return int16 if other is int8 else other
        return a if a.rank > b.rank else b
    return bool_


def result_type(*operands):
    """torch.result_type over tensors (objects with ``dtype`` and ``ndim``) and Python scalars.

    Dimensioned tensors determine the dtype within their category; zero-dim tensors
    and Python scalars only raise the category (scalars to the default dtype).
    """
    dims, zero, scalars = [], [], []
    for x in operands:
        d = getattr(x, "dtype", None)
        if d is not None and not isinstance(x, (bool, int, float)):
            d = dtype(d)
            (dims if getattr(x, "ndim", 1) else zero).append(d)
        elif isinstance(x, (bool, int, float)):
            scalars.append(of_scalar(x))
    result = None
    for d in dims:
        result = d if result is None else promote_types(result, d)
    for group, default in ((zero, False), (scalars, True)):
        best = None
        for d in group:
            best = d if best is None else promote_types(best, d)
        if best is None: continue
        if result is None: result = best
        elif best.category > result.category:
            result = (DEFAULT_FLOAT if best.category == 2 else DEFAULT_INT) if default else best
    return result or DEFAULT_FLOAT


class finfo:
    """torch.finfo for float16, bfloat16, float32 and float64."""
    _TABLE = {
        "float16": (16, 65504.0, 2.0 ** -10, 2.0 ** -14, 0.001),
        "bfloat16": (16, 3.3895313892515355e38, 2.0 ** -7, 1.1754943508222875e-38, 0.01),
        "float32": (32, 3.4028234663852886e38, 2.0 ** -23, 1.1754943508222875e-38, 1e-6),
        "float64": (64, 1.7976931348623157e308, 2.0 ** -52, 2.2250738585072014e-308, 1e-15),
    }

    def __init__(self, type=None):
        d = dtype(type) if type is not None else DEFAULT_FLOAT
        if not d.is_floating_point: raise TypeError("finfo requires a floating point dtype")
        self.dtype = d
        self.bits, self.max, self.eps, self.tiny, self.resolution = self._TABLE[d.name]
        self.min = -self.max
        self.smallest_normal = self.tiny

    def __repr__(self): return "finfo(dtype=%s, min=%g, max=%g, eps=%g)" % (self.dtype.name, self.min, self.max, self.eps)


class iinfo:
    """torch.iinfo for the integer dtypes."""
    def __init__(self, type):
        d = dtype(type)
        if d.category != 1: raise TypeError("iinfo requires an integer dtype")
        self.dtype, self.bits = d, d.itemsize * 8
        self.min = -(1 << (self.bits - 1)) if d.is_signed else 0
        self.max = (1 << (self.bits - 1)) - 1 if d.is_signed else (1 << self.bits) - 1


F32_MAX = finfo.__dict__["_TABLE"]["float32"][1]


def finite(x):
    """Map Python infinities to the f32 extremes (FIR constants must be finite)."""
    if isinstance(x, float) and math.isinf(x):
        return F32_MAX if x > 0 else -F32_MAX
    return x
