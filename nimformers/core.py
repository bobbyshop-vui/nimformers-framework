"""High-level NumPy API for the Nimformers C ABI."""
from __future__ import annotations

import math
import os
import warnings
from typing import Optional, Sequence

import numpy as np

from ._lib import F32, I32, I64, U8, get_library, NimformersError
from .model import LlamaConfig, LlamaModel
from .quant import KIND_IDS, QuantTensor
from .state import StateDict
from .tensor import Tensor
from .tokenizer import SentencePieceTokenizer, Tokenizer
from .vm import VMProgram

__all__ = ["Context", "Nimformers", "ResidentF32Weights", "ResidentQ4Weights"]


_BACKENDS = {
    "auto": 0,
    "metal": 1,
    "vulkan": 2,
    "cuda": 3,   # legacy alias, redirected to Vulkan by the Nim backend
    "opencl": 2,  # legacy alias for Vulkan
    "cpu": 4,    # runtime-dispatched SIMD CPU backend (no GPU required)
}
_COMPARE_OPS = {"gt": 0, "lt": 1, "eq": 2, "ge": 3, "le": 4}
_CPU_ISA_NAMES = {0: "scalar", 1: "sse4", 2: "avx", 3: "avx2", 4: "avx512", 5: "neon"}


# ---------------------------------------------------------------------
# numpy helpers
# ---------------------------------------------------------------------

def _as_f32(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32)


def _f32ptr(arr: np.ndarray):
    return arr.ctypes.data_as(F32)


def _i64ptr(arr: np.ndarray):
    return arr.ctypes.data_as(I64)


def _i32ptr(arr: np.ndarray):
    return arr.ctypes.data_as(I32)


def _u8ptr(arr: np.ndarray):
    return arr.ctypes.data_as(U8)


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise np.AxisError(axis, ndim) if hasattr(np, "AxisError") else ValueError(
            f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis


def _product(values) -> int:
    result = 1
    for v in values:
        result *= int(v)
    return result


def _q4_params(weight):
    """Return (n, k, group_size, scales, zeros, groups_per_row) for a weight.

    The int4 kernel indexes ``scales[n * groups_per_row + group]`` and requires
    a non-empty scale buffer for every output row, so a single per-tensor scale
    is expanded to one entry per row.
    """
    n, k = weight.shape
    gs = weight.group_size
    scales = weight.scales(copy=False)
    zeros = weight.zeros(copy=False)
    if zeros.size != scales.size:
        raise ValueError("quantized weight has mismatched scales/zeros lengths")
    if gs > 0:
        gpr = (k + gs - 1) // gs
        if scales.size != n * gpr:
            raise ValueError(
                f"grouped Q4 weight needs {n * gpr} scales, got {scales.size}")
        return n, k, gs, scales, zeros, gpr
    if scales.size <= 1:
        scale = float(scales[0]) if scales.size else 1.0
        zero = float(zeros[0]) if zeros.size else 0.0
        scales = np.full(n, scale, dtype=np.float32)
        zeros = np.full(n, zero, dtype=np.float32)
        return n, k, 0, scales, zeros, 1
    if n > 0 and scales.size % n != 0:
        raise ValueError("per-row Q4 scales must be divisible by the row count")
    return n, k, 0, scales, zeros, max(1, scales.size // n)


def _strides(shape) -> list[int]:
    result = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        result[i] = acc
        acc *= int(shape[i])
    return result


# ---------------------------------------------------------------------
# resident weight handles
# ---------------------------------------------------------------------

class ResidentF32Weights:
    """A float32 weight matrix uploaded to GPU memory once."""

    def __init__(self, ctx: "Context", weights):
        self.ctx = ctx
        arr = _as_f32(weights)
        if arr.ndim != 2:
            raise ValueError("resident weights must be a 2-D [K, N] array")
        self.shape = arr.shape
        self.handle = ctx.lib.nimf_resident_f32_upload(
            ctx._ctx, _f32ptr(arr), arr.size)
        if not self.handle:
            ctx._raise()

    def linear(self, x) -> np.ndarray:
        k, n = self.shape
        arr = _as_f32(x)
        if arr.ndim < 1 or arr.shape[-1] != k:
            raise ValueError(f"input last dimension must be {k}")
        rows = arr.size // k if k else 0
        out = np.empty(rows * n, dtype=np.float32)
        status = self.ctx.lib.nimf_resident_f32_linear(
            self.ctx._ctx, self.handle, _f32ptr(arr), arr.size, rows, k, n,
            _f32ptr(out))
        if status != 0:
            self.ctx._raise()
        return out.reshape(arr.shape[:-1] + (n,))

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.ctx.lib.nimf_resident_f32_destroy(self.ctx._ctx, self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


class ResidentQ4Weights:
    """A packed int4 weight matrix uploaded to GPU memory once."""

    def __init__(self, ctx: "Context", weight: QuantTensor):
        if not isinstance(weight, QuantTensor):
            raise TypeError("resident_q4 expects a QuantTensor")
        if len(weight.shape) != 2:
            raise ValueError("resident Q4 weights must be 2-D [N, K]")
        if weight.kind != "int4_asym":
            raise ValueError(
                "resident_q4 requires int4_asym weights (the GPU kernel uses "
                "asymmetric (nibble - zero) * scale decoding)")
        n, k, gs, scales, zeros, gpr = _q4_params(weight)
        self.ctx = ctx
        self.k, self.n = k, n
        packed = weight.data(copy=False)
        self.handle = ctx.lib.nimf_resident_q4_upload(
            ctx._ctx, _u8ptr(packed), packed.size, _f32ptr(scales),
            _f32ptr(zeros), scales.size, zeros.size, k, n, gs, gpr)
        if not self.handle:
            ctx._raise()

    def linear(self, x) -> np.ndarray:
        arr = _as_f32(x)
        if arr.ndim < 1 or arr.shape[-1] != self.k:
            raise ValueError(f"input last dimension must be {self.k}")
        rows = arr.size // self.k if self.k else 0
        out = np.empty(rows * self.n, dtype=np.float32)
        status = self.ctx.lib.nimf_resident_q4_linear(
            self.ctx._ctx, self.handle, _f32ptr(arr), arr.size, rows,
            _f32ptr(out))
        if status != 0:
            self.ctx._raise()
        return out.reshape(arr.shape[:-1] + (self.n,))

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.ctx.lib.nimf_resident_q4_destroy(self.ctx._ctx, self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# context
# ---------------------------------------------------------------------

class Context:
    """A Nimformers backend context.

    Parameters
    ----------
    backend:
        ``"auto"`` (default), ``"metal"``, ``"vulkan"``, ``"cuda"``
        (legacy alias for Vulkan), or ``"cpu"`` (SIMD CPU backend).
    library:
        Optional path to ``libnimformers.dylib`` / ``.so`` / ``.dll``.
    forbid_cpu_fallback:
        ``True`` makes missing GPU kernels raise instead of silently
        running on the CPU.  ``None`` keeps the library default.

    Use it as a context manager::

        with Context() as ctx:
            y = ctx.matmul(a, b)
    """

    _BACKEND_NAMES = _BACKENDS

    def __init__(self, backend="auto", library=None, *,
                 forbid_cpu_fallback: Optional[bool] = None):
        # Backwards compatibility with the old `Nimformers("/path/lib.dylib")`.
        if library is None and isinstance(backend, (str, os.PathLike)):
            text = str(backend)
            if text not in _BACKENDS:
                if (os.path.sep in text or text.endswith((".dylib", ".so", ".dll"))):
                    library, backend = text, "auto"
                else:
                    raise ValueError(
                        f"backend must be one of {sorted(_BACKENDS)}, got {text!r}")
        if backend not in _BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(_BACKENDS)}, got {backend!r}")

        self.lib = get_library(library)
        self._ctx = self.lib.nimf_context_create(_BACKENDS[backend])
        if not self._ctx:
            self._raise()
        if forbid_cpu_fallback is not None:
            self.set_cpu_fallback(not forbid_cpu_fallback)

    # -- identity / lifetime -------------------------------------------
    @property
    def abi_version(self) -> int:
        return int(self.lib.nimf_abi_version())

    @property
    def ctx(self):
        """Raw context pointer (backwards-compatible alias for ``_ctx``)."""
        return self._ctx

    @property
    def backend(self) -> str:
        if not self._ctx:
            raise NimformersError("context is closed")
        name = self.lib.nimf_context_backend_name(self._ctx)
        return name.decode() if name else "unknown"

    @property
    def backend_id(self) -> int:
        return int(self.lib.nimf_context_backend(self._ctx))

    def sync(self) -> None:
        """Best-effort device synchronisation."""
        self._check(self.lib.nimf_context_sync(self._ctx))

    def set_cpu_fallback(self, allow: bool) -> None:
        self._check(self.lib.nimf_context_set_cpu_fallback(
            self._ctx, 1 if allow else 0))

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self.lib.nimf_context_destroy(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        try:
            return f"Context(backend={self.backend!r}, abi={self.abi_version})"
        except Exception:
            return "Context(<closed>)"

    # -- errors ---------------------------------------------------------
    def _raise(self):
        msg = self.lib.nimf_last_error()
        raise NimformersError(
            msg.decode("utf-8", "replace") if msg else "Nimformers C API call failed")

    def _check(self, status: int):
        if status != 0:
            self._raise()

    def _check_ptr(self, handle):
        if not handle:
            self._raise()
        return handle

    # -- tensors --------------------------------------------------------
    def tensor(self, value) -> Tensor:
        """Create an owned tensor from any array-like value."""
        arr = _as_f32(value)
        dims = np.ascontiguousarray(arr.shape, dtype=np.int64)
        handle = self.lib.nimf_tensor_create_shaped(
            _f32ptr(arr), arr.size, _i64ptr(dims), dims.size)
        return Tensor(self.lib, self._check_ptr(handle), arr.shape)

    def tensor_from_shape(self, shape, fill: float = 0.0) -> Tensor:
        dims = np.ascontiguousarray(shape, dtype=np.int64)
        handle = self.lib.nimf_tensor_zeros(
            _i64ptr(dims), dims.size, float(fill))
        return Tensor(self.lib, self._check_ptr(handle), tuple(int(d) for d in dims))

    def zeros(self, shape, fill: float = 0.0) -> Tensor:
        return self.tensor_from_shape(shape, fill)

    def randn(self, shape, scale: float = 1.0) -> Tensor:
        dims = np.ascontiguousarray(shape, dtype=np.int64)
        handle = self.lib.nimf_tensor_randn(
            _i64ptr(dims), dims.size, float(scale))
        return Tensor(self.lib, self._check_ptr(handle), tuple(int(d) for d in dims))

    # -- elementwise ----------------------------------------------------
    def _binary(self, name: str, a, b) -> np.ndarray:
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        if aa.shape != bb.shape:
            aa, bb = np.broadcast_arrays(aa, bb)
        aa = np.ascontiguousarray(aa)
        bb = np.ascontiguousarray(bb)
        out = np.empty(aa.shape, dtype=np.float32)
        status = getattr(self.lib, name)(
            self._ctx, _f32ptr(aa), _f32ptr(bb), _f32ptr(out), out.size)
        self._check(status)
        return out

    def add(self, a, b) -> np.ndarray:
        return self._binary("nimf_add", a, b)

    def sub(self, a, b) -> np.ndarray:
        return self._binary("nimf_sub", a, b)

    def mul(self, a, b) -> np.ndarray:
        return self._binary("nimf_mul", a, b)

    def div(self, a, b) -> np.ndarray:
        return self._binary("nimf_div", a, b)

    def _unary(self, name: str, x) -> np.ndarray:
        arr = _as_f32(x)
        out = np.empty_like(arr)
        self._check(getattr(self.lib, name)(
            self._ctx, _f32ptr(arr), _f32ptr(out), out.size))
        return out.reshape(arr.shape)

    def relu(self, x) -> np.ndarray:
        return self._unary("nimf_relu", x)

    def sqrt(self, x) -> np.ndarray:
        return self._unary("nimf_sqrt", x)

    def exp(self, x) -> np.ndarray:
        return self._unary("nimf_exp", x)

    def log(self, x) -> np.ndarray:
        return self._unary("nimf_log", x)

    def sigmoid(self, x) -> np.ndarray:
        return self._unary("nimf_sigmoid", x)

    def tanh(self, x) -> np.ndarray:
        return self._unary("nimf_tanh", x)

    def neg(self, x) -> np.ndarray:
        return self._unary("nimf_neg", x)

    def pow(self, x, exponent: float) -> np.ndarray:
        arr = _as_f32(x)
        out = np.empty_like(arr)
        self._check(self.lib.nimf_pow(
            self._ctx, _f32ptr(arr), _f32ptr(out), out.size, float(exponent)))
        return out.reshape(arr.shape)

    def add_scalar(self, x, scalar: float) -> np.ndarray:
        arr = _as_f32(x)
        out = np.empty_like(arr)
        self._check(self.lib.nimf_add_scalar(
            self._ctx, _f32ptr(arr), _f32ptr(out), out.size, float(scalar)))
        return out.reshape(arr.shape)

    def mul_scalar(self, x, scalar: float) -> np.ndarray:
        arr = _as_f32(x)
        out = np.empty_like(arr)
        self._check(self.lib.nimf_mul_scalar(
            self._ctx, _f32ptr(arr), _f32ptr(out), out.size, float(scalar)))
        return out.reshape(arr.shape)

    def apflu(self, x, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
        arr = _as_f32(x)
        out = np.empty_like(arr)
        self._check(self.lib.nimf_apflu(
            self._ctx, _f32ptr(arr), _f32ptr(out), out.size,
            float(alpha), float(beta)))
        return out.reshape(arr.shape)

    def apflu_backward(self, x, dy, alpha: float = 0.1,
                       beta: float = 0.1) -> np.ndarray:
        arr = _as_f32(x)
        grad = _as_f32(dy)
        if arr.shape != grad.shape:
            raise ValueError("x and dy must have the same shape")
        out = np.empty_like(arr)
        self._check(self.lib.nimf_apflu_backward(
            self._ctx, _f32ptr(arr), _f32ptr(grad), _f32ptr(out), out.size,
            float(alpha), float(beta)))
        return out.reshape(arr.shape)

    # -- matrix multiplication -----------------------------------------
    def _matmul2d(self, a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
        m, k = a.shape
        k2, n = b.shape
        if k != k2:
            raise ValueError(f"matmul shape mismatch: {a.shape} @ {b.shape}")
        self._check(self.lib.nimf_matmul(
            self._ctx, _f32ptr(a), _f32ptr(b), _f32ptr(out), m, k, n))

    def matmul(self, a, b) -> np.ndarray:
        """Matrix multiply with NumPy-style batch broadcasting."""
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        if aa.ndim < 2 or bb.ndim < 2:
            raise ValueError("matmul expects arrays with at least 2 dimensions")
        if aa.ndim == 2 and bb.ndim == 2:
            out = np.empty((aa.shape[0], bb.shape[1]), dtype=np.float32)
            self._matmul2d(np.ascontiguousarray(aa), np.ascontiguousarray(bb), out)
            return out
        batch = np.broadcast_shapes(aa.shape[:-2], bb.shape[:-2])
        a_b = np.broadcast_to(aa, batch + aa.shape[-2:])
        b_b = np.broadcast_to(bb, batch + bb.shape[-2:])
        out = np.empty(batch + (aa.shape[-2], bb.shape[-1]), dtype=np.float32)
        for idx in np.ndindex(batch):
            self._matmul2d(np.ascontiguousarray(a_b[idx]),
                           np.ascontiguousarray(b_b[idx]), out[idx])
        return out

    def _matmul_nt2d(self, a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
        m, k = a.shape
        n, k2 = b.shape
        if k != k2:
            raise ValueError(f"matmul_nt shape mismatch: {a.shape} @ {b.shape}.T")
        self._check(self.lib.nimf_matmul_nt(
            self._ctx, _f32ptr(a), _f32ptr(b), _f32ptr(out), m, k, n))

    def matmul_nt(self, a, b) -> np.ndarray:
        """``a @ b.T`` where ``b`` keeps its natural ``[..., N, K]`` layout."""
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        if aa.ndim < 2 or bb.ndim < 2:
            raise ValueError("matmul_nt expects arrays with at least 2 dimensions")
        if aa.ndim == 2 and bb.ndim == 2:
            out = np.empty((aa.shape[0], bb.shape[0]), dtype=np.float32)
            self._matmul_nt2d(np.ascontiguousarray(aa),
                              np.ascontiguousarray(bb), out)
            return out
        batch = np.broadcast_shapes(aa.shape[:-2], bb.shape[:-2])
        a_b = np.broadcast_to(aa, batch + aa.shape[-2:])
        b_b = np.broadcast_to(bb, batch + bb.shape[-2:])
        out = np.empty(batch + (aa.shape[-2], bb.shape[-2]), dtype=np.float32)
        for idx in np.ndindex(batch):
            self._matmul_nt2d(np.ascontiguousarray(a_b[idx]),
                              np.ascontiguousarray(b_b[idx]), out[idx])
        return out

    def matmul2(self, a1, b1, a2, b2):
        """Twin matmuls sharing one GPU sync: returns ``(a1@b1, a2@b2)``."""
        x1 = np.ascontiguousarray(a1, dtype=np.float32)
        y1 = np.ascontiguousarray(b1, dtype=np.float32)
        x2 = np.ascontiguousarray(a2, dtype=np.float32)
        y2 = np.ascontiguousarray(b2, dtype=np.float32)
        if x1.ndim != 2 or y1.ndim != 2 or x2.ndim != 2 or y2.ndim != 2:
            raise ValueError("matmul2 expects four 2-D arrays")
        o1 = np.empty((x1.shape[0], y1.shape[1]), dtype=np.float32)
        o2 = np.empty((x2.shape[0], y2.shape[1]), dtype=np.float32)
        self._check(self.lib.nimf_matmul2(
            self._ctx,
            _f32ptr(x1), _f32ptr(y1), _f32ptr(o1),
            x1.shape[0], x1.shape[1], y1.shape[1],
            _f32ptr(x2), _f32ptr(y2), _f32ptr(o2),
            x2.shape[0], x2.shape[1], y2.shape[1]))
        return o1, o2

    def matmul_q4(self, a, packed_weight, scales, zeros, *, k: int, n: int,
                  group_size: int = 0, groups_per_row: int = 1) -> np.ndarray:
        """Int4 weight matmul.

        ``packed_weight`` is ``[N, K]`` nibble-packed (two values per byte),
        ``scales``/``zeros`` are per-row-group quantization parameters.
        """
        aa = np.asarray(a, dtype=np.float32)
        if aa.ndim != 2 or aa.shape[1] != k:
            raise ValueError(f"a must be [m, {k}]")
        packed = np.ascontiguousarray(packed_weight, dtype=np.uint8).reshape(-1)
        sc = np.ascontiguousarray(scales, dtype=np.float32).reshape(-1)
        ze = np.ascontiguousarray(zeros, dtype=np.float32).reshape(-1)
        out = np.empty((aa.shape[0], n), dtype=np.float32)
        aa = np.ascontiguousarray(aa)
        self._check(self.lib.nimf_matmul_q4(
            self._ctx, _f32ptr(aa), aa.size, _u8ptr(packed), packed.size,
            _f32ptr(sc), _f32ptr(ze), sc.size, ze.size, _f32ptr(out),
            aa.shape[0], k, n, group_size, groups_per_row))
        return out

    def linear_q4(self, x, weight: QuantTensor) -> np.ndarray:
        """``x @ weight`` using a quantized ``[N, K]`` weight tensor."""
        if not isinstance(weight, QuantTensor):
            raise TypeError("linear_q4 expects a QuantTensor")
        if len(weight.shape) != 2:
            raise ValueError("quantized weight must be 2-D [N, K]")
        # The int4 matmul kernel decodes (nibble - zero_point) * scale, which
        # exactly matches int4_asym (per-tensor or per-group).  Symmetric int4
        # stores two's-complement nibbles, and FP4-KM interleaves per-group
        # scale bytes; both must be dequantized before multiplying.
        if weight.kind != "int4_asym":
            return self.matmul_nt(np.asarray(x, dtype=np.float32),
                                  weight.dequantize())
        n, k, gs, scales, zeros, gpr = _q4_params(weight)
        return self.matmul_q4(
            x, weight.data(copy=False), scales, zeros, k=k, n=n,
            group_size=gs, groups_per_row=gpr)

    def resident_f32(self, weights) -> ResidentF32Weights:
        return ResidentF32Weights(self, weights)

    def resident_q4(self, weight: QuantTensor) -> ResidentQ4Weights:
        return ResidentQ4Weights(self, weight)

    # -- neural-network kernels ----------------------------------------
    def softmax(self, x, axis: int = -1) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError("softmax needs at least one dimension")
        axis = _normalize_axis(axis, arr.ndim)
        moved = np.moveaxis(arr, axis, -1)
        cols = moved.shape[-1]
        flat = np.ascontiguousarray(moved).reshape(-1, cols)
        out = np.empty_like(flat)
        self._check(self.lib.nimf_softmax(
            self._ctx, _f32ptr(flat), _f32ptr(out), flat.shape[0], cols))
        return np.moveaxis(out.reshape(moved.shape), -1, axis)

    def layernorm(self, x, gamma, beta, eps: float = 1e-5,
                  axis: int = -1) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError("layernorm needs at least one dimension")
        axis = _normalize_axis(axis, arr.ndim)
        g = _as_f32(gamma).reshape(-1)
        b = _as_f32(beta).reshape(-1)
        cols = arr.shape[axis]
        if g.size != cols or b.size != cols:
            raise ValueError(
                f"gamma and beta must have {cols} elements, got {g.size}/{b.size}")
        moved = np.moveaxis(arr, axis, -1)
        flat = np.ascontiguousarray(moved).reshape(-1, cols)
        out = np.empty_like(flat)
        self._check(self.lib.nimf_layernorm(
            self._ctx, _f32ptr(flat), _f32ptr(g), _f32ptr(b),
            _f32ptr(out), flat.shape[0], cols, float(eps)))
        return np.moveaxis(out.reshape(moved.shape), -1, axis)

    def add_layernorm(self, a, b, gamma, beta, eps: float = 1e-5,
                      axis: int = -1):
        """Fused ``(a + b)`` and ``layernorm(a + b)``; returns both arrays."""
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        if aa.shape != bb.shape:
            raise ValueError("a and b must have the same shape")
        if aa.ndim == 0:
            raise ValueError("add_layernorm needs at least one dimension")
        axis = _normalize_axis(axis, aa.ndim)
        g = _as_f32(gamma).reshape(-1)
        be = _as_f32(beta).reshape(-1)
        cols = aa.shape[axis]
        if g.size != cols or be.size != cols:
            raise ValueError(f"gamma and beta must have {cols} elements")
        a_m = np.moveaxis(aa, axis, -1)
        b_m = np.moveaxis(bb, axis, -1)
        flat_a = np.ascontiguousarray(a_m).reshape(-1, cols)
        flat_b = np.ascontiguousarray(b_m).reshape(-1, cols)
        summed = np.empty_like(flat_a)
        normed = np.empty_like(flat_a)
        self._check(self.lib.nimf_add_layernorm(
            self._ctx, _f32ptr(flat_a), _f32ptr(flat_b), _f32ptr(g),
            _f32ptr(be), _f32ptr(summed), _f32ptr(normed),
            flat_a.shape[0], cols, float(eps)))
        shape = a_m.shape
        return (np.moveaxis(summed.reshape(shape), -1, axis),
                np.moveaxis(normed.reshape(shape), -1, axis))

    def layernorm_backward(self, dy, x, gamma, beta, eps: float = 1e-5,
                           axis: int = -1):
        """Returns ``(dx, dgamma, dbeta)``."""
        d = np.asarray(dy, dtype=np.float32)
        xx = np.asarray(x, dtype=np.float32)
        if d.shape != xx.shape:
            raise ValueError("dy and x must have the same shape")
        if xx.ndim == 0:
            raise ValueError("layernorm_backward needs at least one dimension")
        axis = _normalize_axis(axis, xx.ndim)
        g = _as_f32(gamma).reshape(-1)
        b = _as_f32(beta).reshape(-1)
        cols = xx.shape[axis]
        if g.size != cols or b.size != cols:
            raise ValueError(f"gamma and beta must have {cols} elements")
        d_m = np.moveaxis(d, axis, -1)
        x_m = np.moveaxis(xx, axis, -1)
        flat_d = np.ascontiguousarray(d_m).reshape(-1, cols)
        flat_x = np.ascontiguousarray(x_m).reshape(-1, cols)
        dx = np.empty_like(flat_x)
        dg = np.empty(cols, dtype=np.float32)
        db = np.empty(cols, dtype=np.float32)
        self._check(self.lib.nimf_layernorm_backward(
            self._ctx, _f32ptr(flat_d), _f32ptr(flat_x), _f32ptr(g),
            _f32ptr(b), flat_x.shape[0], cols, float(eps),
            _f32ptr(dx), _f32ptr(dg), _f32ptr(db)))
        return np.moveaxis(dx.reshape(x_m.shape), -1, axis), dg, db

    def attention(self, q, k, v, mask=None, scale: Optional[float] = None):
        """Fused causal self-attention.

        ``q``/``k``/``v`` are ``[B, H, S, D]``.  Returns ``(out, scores)``
        where ``scores`` is ``[B, H, S, S]``.  The current Metal/Vulkan
        kernels are causal; ``mask`` is accepted for API compatibility but
        is not applied by them.
        """
        qa = np.asarray(q, dtype=np.float32)
        ka = np.asarray(k, dtype=np.float32)
        va = np.asarray(v, dtype=np.float32)
        if qa.ndim != 4:
            raise ValueError("q must be [batch, heads, seq, head_dim]")
        if qa.shape != ka.shape or qa.shape != va.shape:
            raise ValueError("q, k and v must have the same shape")
        B, H, S, D = qa.shape
        if mask is not None:
            warnings.warn(
                "Nimformers fused attention is causal; the mask argument is "
                "ignored by the current Metal/Vulkan kernels",
                RuntimeWarning, stacklevel=2)
        if scale is None:
            scale = 1.0 / math.sqrt(D) if D else 1.0
        qf = np.ascontiguousarray(qa).reshape(-1)
        kf = np.ascontiguousarray(ka).reshape(-1)
        vf = np.ascontiguousarray(va).reshape(-1)
        mf = (np.zeros(0, dtype=np.float32) if mask is None
              else np.ascontiguousarray(mask, dtype=np.float32).reshape(-1))
        out = np.empty_like(qf)
        scores = np.empty(B * H * S * S, dtype=np.float32)
        self._check(self.lib.nimf_attention(
            self._ctx, _f32ptr(qf), _f32ptr(kf), _f32ptr(vf),
            _f32ptr(mf), mf.size, B, H, S, D, float(scale),
            _f32ptr(out), _f32ptr(scores)))
        return out.reshape(qa.shape), scores.reshape(B, H, S, S)

    def attention_backward(self, q, k, v, scores, dy,
                           scale: Optional[float] = None):
        """Returns ``(dq, dk, dv)``."""
        qa = np.asarray(q, dtype=np.float32)
        ka = np.asarray(k, dtype=np.float32)
        va = np.asarray(v, dtype=np.float32)
        sa = np.asarray(scores, dtype=np.float32)
        da = np.asarray(dy, dtype=np.float32)
        if qa.ndim != 4 or qa.shape != ka.shape or qa.shape != va.shape:
            raise ValueError("q, k and v must be [B, H, S, D] with equal shapes")
        if da.shape != qa.shape:
            raise ValueError("dy must match q")
        B, H, S, D = qa.shape
        if sa.size != B * H * S * S:
            raise ValueError(f"scores must have {B * H * S * S} elements")
        if scale is None:
            scale = 1.0 / math.sqrt(D) if D else 1.0
        qf = np.ascontiguousarray(qa).reshape(-1)
        kf = np.ascontiguousarray(ka).reshape(-1)
        vf = np.ascontiguousarray(va).reshape(-1)
        sf = np.ascontiguousarray(sa).reshape(-1)
        df = np.ascontiguousarray(da).reshape(-1)
        dq = np.empty_like(qf)
        dk = np.empty_like(qf)
        dv = np.empty_like(qf)
        self._check(self.lib.nimf_attention_backward(
            self._ctx, _f32ptr(qf), _f32ptr(kf), _f32ptr(vf),
            _f32ptr(sf), _f32ptr(df), B, H, S, D, float(scale),
            _f32ptr(dq), _f32ptr(dk), _f32ptr(dv)))
        return dq.reshape(qa.shape), dk.reshape(qa.shape), dv.reshape(qa.shape)

    def embedding_lookup(self, table, indices) -> np.ndarray:
        tab = _as_f32(table)
        if tab.ndim != 2:
            raise ValueError("embedding table must be [vocab, dim]")
        idx = np.ascontiguousarray(indices, dtype=np.int32).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= tab.shape[0]):
            raise IndexError("embedding index out of range")
        out = np.empty((idx.size, tab.shape[1]), dtype=np.float32)
        self._check(self.lib.nimf_embedding_lookup(
            self._ctx, _f32ptr(tab), tab.size, _i32ptr(idx), idx.size,
            _f32ptr(out), tab.shape[0], tab.shape[1]))
        return out.reshape(np.shape(indices) + (tab.shape[1],))

    def gather(self, x, indices) -> np.ndarray:
        src = _as_f32(x).reshape(-1)
        idx = np.ascontiguousarray(indices, dtype=np.int32).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= src.size):
            raise IndexError("gather index out of range")
        out = np.empty(idx.size, dtype=np.float32)
        self._check(self.lib.nimf_gather(
            self._ctx, _f32ptr(src), src.size, _i32ptr(idx), idx.size,
            _f32ptr(out)))
        return out.reshape(np.shape(indices))

    def compare(self, a, b, op: str = "gt") -> np.ndarray:
        if isinstance(op, str):
            if op not in _COMPARE_OPS:
                raise ValueError(f"op must be one of {sorted(_COMPARE_OPS)}")
            op_id = _COMPARE_OPS[op]
        else:
            op_id = int(op)
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        if aa.shape != bb.shape:
            aa, bb = np.broadcast_arrays(aa, bb)
        aa = np.ascontiguousarray(aa)
        bb = np.ascontiguousarray(bb)
        out = np.empty(aa.shape, dtype=np.uint8)
        self._check(self.lib.nimf_compare(
            self._ctx, _f32ptr(aa), _f32ptr(bb), aa.size, op_id, _u8ptr(out)))
        return out.astype(bool)

    def dropout(self, x, p: float = 0.5, *, training: bool = True,
                seed: int = 0) -> np.ndarray:
        arr = _as_f32(x)
        out = np.empty_like(arr)
        self._check(self.lib.nimf_dropout(
            self._ctx, _f32ptr(arr), _f32ptr(out), arr.size, float(p),
            1 if training else 0, int(seed) & 0xFFFFFFFF))
        return out.reshape(arr.shape)

    def dropout_mask(self, n: int, p: float = 0.5, *, seed: int = 0) -> np.ndarray:
        """Deterministic inverted-dropout mask of length ``n``."""
        if n < 0:
            raise ValueError("n must be non-negative")
        out = np.empty(max(int(n), 1), dtype=np.float32)
        self._check(self.lib.nimf_dropout_mask(
            self._ctx, int(n), float(p), int(seed) & 0xFFFFFFFF, _f32ptr(out)))
        return out[:int(n)]

    def argmax(self, x, axis: int = -1) -> np.ndarray:
        """Index of the maximum along ``axis`` (int64)."""
        return self.argmax_with_values(x, axis=axis)[1]

    def argmax_with_values(self, x, axis: int = -1):
        arr = np.asarray(x, dtype=np.float32)
        axis = _normalize_axis(axis, arr.ndim)
        dim = arr.shape[axis]
        if dim == 0:
            raise ValueError("argmax requires a non-empty dimension")
        outer = _product(arr.shape[:axis])
        inner = _product(arr.shape[axis + 1:])
        out_shape = arr.shape[:axis] + arr.shape[axis + 1:]
        vals = np.empty(max(outer * inner, 1), dtype=np.float32)
        idx = np.empty(max(outer * inner, 1), dtype=np.int64)
        self._check(self.lib.nimf_argmax(
            self._ctx, _f32ptr(np.ascontiguousarray(arr)), arr.size,
            outer, dim, inner, _f32ptr(vals), _i64ptr(idx)))
        count = outer * inner
        return vals[:count].reshape(out_shape), idx[:count].reshape(out_shape)

    def masked_fill(self, x, mask, value: float) -> np.ndarray:
        arr = _as_f32(x)
        m = np.ascontiguousarray(mask).astype(bool).reshape(arr.shape)
        out = np.empty_like(arr)
        self._check(self.lib.nimf_masked_fill(
            _f32ptr(arr), _u8ptr(m.view(np.uint8)), arr.size, float(value),
            _f32ptr(out)))
        return out.reshape(arr.shape)

    def where(self, cond, a, b) -> np.ndarray:
        c = np.ascontiguousarray(cond).astype(bool)
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        c, aa, bb = np.broadcast_arrays(c, aa, bb)
        c = np.ascontiguousarray(c)
        aa = np.ascontiguousarray(aa)
        bb = np.ascontiguousarray(bb)
        out = np.empty(aa.shape, dtype=np.float32)
        self._check(self.lib.nimf_where(
            _u8ptr(c.view(np.uint8)), _f32ptr(aa), _f32ptr(bb),
            aa.size, _f32ptr(out)))
        return out

    # -- reductions / scans --------------------------------------------
    def _reduce(self, name: str, x, axis: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        axis = _normalize_axis(axis, arr.ndim)
        shape = np.ascontiguousarray(arr.shape, dtype=np.int64)
        out_shape = arr.shape[:axis] + arr.shape[axis + 1:]
        out = np.empty(max(_product(out_shape), 1), dtype=np.float32)
        self._check(getattr(self.lib, name)(
            self._ctx, _f32ptr(np.ascontiguousarray(arr)), arr.size,
            _i64ptr(shape), shape.size, axis, _f32ptr(out)))
        return out[:_product(out_shape)].reshape(out_shape)

    def sum(self, x, axis: int = -1) -> np.ndarray:
        return self._reduce("nimf_sum", x, axis)

    def mean(self, x, axis: int = -1) -> np.ndarray:
        return self._reduce("nimf_mean", x, axis)

    def max(self, x, axis: int = -1) -> np.ndarray:
        return self._reduce("nimf_max", x, axis)

    def _scan(self, name: str, x, axis: int) -> np.ndarray:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        axis = _normalize_axis(axis, arr.ndim)
        shape = np.ascontiguousarray(arr.shape, dtype=np.int64)
        out = np.empty(arr.shape, dtype=np.float32)
        self._check(getattr(self.lib, name)(
            self._ctx, _f32ptr(arr), arr.size,
            _i64ptr(shape), shape.size, axis, _f32ptr(out)))
        return out

    def cumsum(self, x, axis: int = -1) -> np.ndarray:
        return self._scan("nimf_cumsum", x, axis)

    def cumprod(self, x, axis: int = -1) -> np.ndarray:
        return self._scan("nimf_cumprod", x, axis)

    def affine_scan(self, a, b, axis: int = -1) -> np.ndarray:
        """Inclusive scan ``h[t] = a[t] * h[t-1] + b[t]`` (``h[-1] = 0``)."""
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        if aa.shape != bb.shape:
            raise ValueError("a and b must have the same shape")
        axis = _normalize_axis(axis, aa.ndim)
        # The Nim kernel scans the last dimension; move the requested axis to
        # the end and restore it afterwards.
        moved_a = np.moveaxis(aa, axis, -1)
        moved_b = np.moveaxis(bb, axis, -1)
        shape = np.ascontiguousarray(moved_a.shape, dtype=np.int64)
        out = np.empty(moved_a.shape, dtype=np.float32)
        self._check(self.lib.nimf_affine_scan(
            self._ctx, _f32ptr(np.ascontiguousarray(moved_a)),
            _f32ptr(np.ascontiguousarray(moved_b)), moved_a.size, _i64ptr(shape),
            shape.size, moved_a.ndim - 1, _f32ptr(out)))
        return np.moveaxis(out, -1, axis)

    # -- shape movement -------------------------------------------------
    def permute(self, x, dims) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        dims = [int(d) for d in dims]
        if len(dims) != arr.ndim:
            raise ValueError("permute needs one dimension index per axis")
        new_shape = tuple(arr.shape[d] for d in dims)
        old_strides = np.ascontiguousarray(_strides(arr.shape), dtype=np.int32)
        new_strides = np.ascontiguousarray(_strides(new_shape), dtype=np.int32)
        dims_map = np.ascontiguousarray(dims, dtype=np.int32)
        out = np.empty(new_shape, dtype=np.float32)
        self._check(self.lib.nimf_permute(
            self._ctx, _f32ptr(np.ascontiguousarray(arr)), arr.size,
            _i32ptr(new_strides), _i32ptr(old_strides), _i32ptr(dims_map),
            len(dims), _f32ptr(out)))
        return out

    def split_slice(self, x, axis: int, offset: int, size: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        axis = _normalize_axis(axis, arr.ndim)
        dim_total = arr.shape[axis]
        if offset < 0 or size < 0 or offset + size > dim_total:
            raise ValueError("invalid split slice range")
        outer = _product(arr.shape[:axis])
        inner = _product(arr.shape[axis + 1:])
        new_shape = arr.shape[:axis] + (size,) + arr.shape[axis + 1:]
        out = np.empty(_product(new_shape), dtype=np.float32)
        self._check(self.lib.nimf_split_slice(
            self._ctx, _f32ptr(np.ascontiguousarray(arr)), dim_total, size,
            offset, inner, out.size, _f32ptr(out)))
        return out.reshape(new_shape)

    def split(self, x, split_size: int, axis: int = 0):
        """Split into chunks of at most ``split_size`` along ``axis``."""
        arr = np.asarray(x, dtype=np.float32)
        axis = _normalize_axis(axis, arr.ndim)
        total = arr.shape[axis]
        if split_size <= 0:
            raise ValueError("split_size must be positive")
        return [self.split_slice(arr, axis, off, min(split_size, total - off))
                for off in range(0, total, split_size)]

    def stack(self, arrays: Sequence, axis: int = 0) -> np.ndarray:
        """Stack arrays along a new axis (uses the concat kernel)."""
        items = [_as_f32(a) for a in arrays]
        if not items:
            raise ValueError("stack needs at least one array")
        rank = items[0].ndim
        for item in items:
            if item.shape != items[0].shape:
                raise ValueError("stack requires arrays with identical shapes")
        axis = _normalize_axis(axis, rank + 1)
        return self.concat([np.expand_dims(a, axis) for a in items], axis=axis)

    def concat(self, arrays: Sequence, axis: int = 0) -> np.ndarray:
        items = [np.asarray(a, dtype=np.float32) for a in arrays]
        if not items:
            raise ValueError("concat needs at least one array")
        base_shape = list(items[0].shape)
        axis = _normalize_axis(axis, len(base_shape))
        for item in items:
            if item.ndim != len(base_shape):
                raise ValueError("all inputs must have the same rank")
            for i, dim in enumerate(item.shape):
                if i != axis and dim != base_shape[i]:
                    raise ValueError("all dimensions except axis must match")
        total_dim = sum(item.shape[axis] for item in items)
        new_shape = list(base_shape)
        new_shape[axis] = total_dim
        inner = _product(base_shape[axis + 1:])
        flat_src = np.concatenate([np.ascontiguousarray(it).reshape(-1)
                                   for it in items]).astype(np.float32, copy=False)
        bases, sizes, offsets = [], [], []
        position = 0
        dim_offset = 0
        for item in items:
            bases.append(position)
            sizes.append(item.shape[axis])
            offsets.append(dim_offset)
            position += item.size
            dim_offset += item.shape[axis]
        out = np.empty(_product(new_shape), dtype=np.float32)
        self._check(self.lib.nimf_concat(
            self._ctx, _f32ptr(flat_src), flat_src.size,
            _i32ptr(np.ascontiguousarray(bases, dtype=np.int32)),
            _i32ptr(np.ascontiguousarray(sizes, dtype=np.int32)),
            _i32ptr(np.ascontiguousarray(offsets, dtype=np.int32)),
            len(items), total_dim, inner, out.size, _f32ptr(out)))
        return out.reshape(new_shape)

    # -- quantization ---------------------------------------------------
    def quantize(self, x, kind: str = "int8", *, exponent_bits: int = 0,
                 mantissa_bits: int = 0) -> QuantTensor:
        if kind not in KIND_IDS:
            raise ValueError(f"unknown quantization kind {kind!r}")
        arr = np.ascontiguousarray(x, dtype=np.float32)
        shape = np.ascontiguousarray(arr.shape, dtype=np.int64)
        handle = self.lib.nimf_quantize(
            _f32ptr(arr), arr.size, _i64ptr(shape), shape.size,
            KIND_IDS[kind], int(exponent_bits), int(mantissa_bits))
        return QuantTensor(self.lib, self._check_ptr(handle))

    def load_state_dict(self, path, mapped: bool = True) -> StateDict:
        return StateDict.load(self.lib, path, mapped=mapped)

    def create_state_dict(self, arch=(0, 0, 0, 0, 0)) -> StateDict:
        return StateDict.create(self.lib, arch=arch)

    # -- tokenizers -----------------------------------------------------
    def tokenizer(self, model_dir) -> Tokenizer:
        """Load a Hugging Face tokenizer directory."""
        return Tokenizer.load(self.lib, model_dir)

    def sentencepiece(self, model_path) -> SentencePieceTokenizer:
        """Load a raw SentencePiece ``tokenizer.model`` file."""
        return SentencePieceTokenizer.load(self.lib, model_path)

    # -- Llama inference ------------------------------------------------
    def llama_config(self, config_path) -> LlamaConfig:
        return LlamaConfig.load(self.lib, config_path)

    def load_llama(self, config_path, weights_path,
                   upload_resident: bool = True) -> LlamaModel:
        """Load a Llama-style NIMQ checkpoint (slow; multi-GB)."""
        return LlamaModel.load(self, config_path, weights_path,
                               upload_resident=upload_resident)

    # -- custom float / APF --------------------------------------------
    def customfloat_encode(self, x, exponent_bits: int,
                           mantissa_bits: int) -> np.ndarray:
        arr = _as_f32(x)
        # encodeArray uses a whole number of bytes per item (at most 8 for
        # total_bits <= 64), so 8 bytes/element is a safe upper bound.
        out = np.empty(max(arr.size * 8, 1), dtype=np.uint8)
        written = np.zeros(1, dtype=np.int64)
        self._check(self.lib.nimf_customfloat_encode(
            self._ctx, _f32ptr(arr), arr.size, int(exponent_bits),
            int(mantissa_bits), _u8ptr(out), out.size, _i64ptr(written)))
        return out[:int(written[0])]

    def customfloat_decode(self, data, item_count: int, exponent_bits: int,
                           mantissa_bits: int) -> np.ndarray:
        raw = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
        out = np.empty(item_count, dtype=np.float32)
        self._check(self.lib.nimf_customfloat_decode(
            self._ctx, _u8ptr(raw), raw.size, item_count,
            int(exponent_bits), int(mantissa_bits), _f32ptr(out)))
        return out

    def apf_build_dtype(self, x, grad=None, rel_error_tol: float = 1e-3,
                        exp_margin: int = 1):
        arr = _as_f32(x)
        g = np.zeros(0, dtype=np.float32) if grad is None else _as_f32(grad).reshape(-1)
        import ctypes as _ct
        e = _ct.c_int()
        m = _ct.c_int()
        self._check(self.lib.nimf_apf_build_dtype(
            _f32ptr(arr), arr.size, _f32ptr(g), g.size, float(rel_error_tol),
            int(exp_margin), _ct.byref(e), _ct.byref(m)))
        return int(e.value), int(m.value)

    def apf_cast_training(self, x, grad=None, rel_error_tol: float = 1e-3,
                          exp_margin: int = 1):
        """Returns ``(bytes, (exponent_bits, mantissa_bits))``."""
        arr = _as_f32(x)
        g = np.zeros(0, dtype=np.float32) if grad is None else _as_f32(grad).reshape(-1)
        out = np.empty(max(arr.size * 8, 1), dtype=np.uint8)
        written = np.zeros(1, dtype=np.int64)
        import ctypes as _ct
        e = _ct.c_int()
        m = _ct.c_int()
        self._check(self.lib.nimf_apf_cast_training(
            _f32ptr(arr), arr.size, _f32ptr(g), g.size, float(rel_error_tol),
            int(exp_margin), _u8ptr(out), out.size, _i64ptr(written),
            _ct.byref(e), _ct.byref(m)))
        return out[:int(written[0])], (int(e.value), int(m.value))

    def apf_decode_training(self, data, exponent_bits: int,
                            mantissa_bits: int, item_count: int) -> np.ndarray:
        raw = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
        out = np.empty(item_count, dtype=np.float32)
        got = int(self.lib.nimf_apf_decode_training(
            _u8ptr(raw), raw.size, int(exponent_bits), int(mantissa_bits),
            _f32ptr(out), item_count))
        if got < 0:
            self._raise()
        return out[:got]

    # -- micro-VM -------------------------------------------------------
    def vm_program(self) -> VMProgram:
        return VMProgram(self.lib)

    def vm_run(self, program: VMProgram, memory, n_threads: Optional[int] = None,
               tg_floats: int = 0, tg_size: int = 0) -> np.ndarray:
        return program.run(self, memory, n_threads=n_threads,
                           tg_floats=tg_floats, tg_size=tg_size)

    def micro_vm_dispatch(self, words, memory, *, n_threads: Optional[int] = None,
                          tg_floats: int = 0, tg_size: int = 0) -> np.ndarray:
        """Raw-bytecode dispatch; prefer :meth:`vm_run` with a VMProgram."""
        code = np.ascontiguousarray(words, dtype=np.uint32).reshape(-1)
        program = self.vm_program()
        program.set_words(code)
        try:
            return program.run(self, memory, n_threads=n_threads,
                               tg_floats=tg_floats, tg_size=tg_size)
        finally:
            program.close()

    # -- misc -----------------------------------------------------------
    @property
    def version(self) -> str:
        value = self.lib.nimf_version()
        return value.decode() if value else ""


def cpu_info() -> dict:
    """Return the detected CPU ISA, feature bits, cache sizes and matmul tiles.

    This does not require a :class:`Context`; it reports the process-wide
    CPU backend configuration (the C constructor detects it once).
    """
    lib = get_library()
    isa = int(lib.nimf_cpu_isa())
    raw_name = lib.nimf_cpu_isa_name()
    features = int(lib.nimf_cpu_features())
    l1 = np.zeros(1, dtype=np.int64)
    l2 = np.zeros(1, dtype=np.int64)
    l3 = np.zeros(1, dtype=np.int64)
    lib.nimf_cpu_cache_sizes(
        l1.ctypes.data_as(I64), l2.ctypes.data_as(I64), l3.ctypes.data_as(I64))
    mc = np.zeros(1, dtype=np.int64)
    nc = np.zeros(1, dtype=np.int64)
    kc = np.zeros(1, dtype=np.int64)
    lib.nimf_cpu_block_sizes(
        mc.ctypes.data_as(I64), nc.ctypes.data_as(I64), kc.ctypes.data_as(I64))
    return {
        "isa": isa,
        "isa_name": raw_name.decode() if raw_name else _CPU_ISA_NAMES.get(isa, "unknown"),
        "features": features,
        "has_sse4": bool(features & 1),
        "has_avx": bool(features & 2),
        "has_avx2": bool(features & 4),
        "has_fma": bool(features & 8),
        "has_avx512": bool(features & 16),
        "has_neon": bool(features & 32),
        "l1_bytes": int(l1[0]),
        "l2_bytes": int(l2[0]),
        "l3_bytes": int(l3[0]),
        "block_mc": int(mc[0]),
        "block_nc": int(nc[0]),
        "block_kc": int(kc[0]),
    }


# Backwards-compatible name used by the original ctypes wrapper.
Nimformers = Context
