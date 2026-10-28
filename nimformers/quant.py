"""Quantized tensor handles (int8/int4/fp8/fp4/custom float)."""
from __future__ import annotations

import ctypes as ct

import numpy as np

from ._lib import F32, I64, NimformersError

__all__ = ["QuantTensor", "QuantKind"]

# Must match the QK_* constants in lib_c_api.nim / QuantKind in quant.nim.
KIND_NAMES = (
    "fp32",
    "int8",
    "int4",
    "int4_asym",
    "fp8_e4m3",
    "fp8_e5m2",
    "custom",
    "auto",
    "fp4_km",
)
KIND_IDS = {name: i for i, name in enumerate(KIND_NAMES)}


class QuantKind:
    """Symbolic quantization kinds accepted by :meth:`Context.quantize`."""

    FP32 = "fp32"
    INT8 = "int8"
    INT4 = "int4"
    INT4_ASYM = "int4_asym"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    CUSTOM = "custom"
    AUTO = "auto"
    FP4_KM = "fp4_km"


class QuantTensor:
    """An owned quantized tensor produced by the Nimformers library."""

    def __init__(self, lib, handle):
        if not handle:
            raise NimformersError("failed to create quantized tensor")
        self.lib = lib
        self.handle = handle

    # -- metadata -------------------------------------------------------
    @property
    def kind_id(self) -> int:
        k = int(self.lib.nimf_quant_kind(self.handle))
        if k < 0:
            self._raise()
        return k

    @property
    def kind(self) -> str:
        return KIND_NAMES[self.kind_id]

    @property
    def shape(self) -> tuple[int, ...]:
        rank = int(self.lib.nimf_quant_rank(self.handle))
        if rank < 0:
            self._raise()
        if rank == 0:
            return ()
        out = np.empty(rank, dtype=np.int64)
        got = int(self.lib.nimf_quant_shape(
            self.handle, out.ctypes.data_as(I64), rank))
        if got < 0:
            self._raise()
        return tuple(int(d) for d in out[:got])

    @property
    def numel(self) -> int:
        n = int(self.lib.nimf_quant_numel(self.handle))
        if n < 0:
            self._raise()
        return n

    @property
    def group_size(self) -> int:
        g = int(self.lib.nimf_quant_group_size(self.handle))
        if g < 0:
            self._raise()
        return g

    @property
    def cf_bits(self) -> tuple[int, int]:
        e, m = ct.c_int(), ct.c_int()
        if self.lib.nimf_quant_cf_bits(self.handle, ct.byref(e), ct.byref(m)) != 0:
            self._raise()
        return int(e.value), int(m.value)

    @property
    def exponent_bits(self) -> int:
        return self.cf_bits[0]

    @property
    def mantissa_bits(self) -> int:
        return self.cf_bits[1]

    @property
    def nbytes(self) -> int:
        n = int(self.lib.nimf_quant_data_len(self.handle))
        if n < 0:
            self._raise()
        return n

    # -- payload views --------------------------------------------------
    def _float_array(self, getter, len_getter, copy: bool) -> np.ndarray:
        n = int(len_getter(self.handle))
        if n < 0:
            self._raise()
        if n == 0:
            return np.empty(0, dtype=np.float32)
        ptr = getter(self.handle)
        if not ptr:
            self._raise()
        view = np.ctypeslib.as_array(ptr, shape=(n,))
        return view.copy() if copy else view

    def data(self, copy: bool = True) -> np.ndarray:
        """Raw packed payload bytes (uint8)."""
        n = int(self.lib.nimf_quant_data_len(self.handle))
        if n < 0:
            self._raise()
        if n == 0:
            return np.empty(0, dtype=np.uint8)
        ptr = self.lib.nimf_quant_data(self.handle)
        if not ptr:
            self._raise()
        view = np.ctypeslib.as_array(ptr, shape=(n,))
        return view.copy() if copy else view

    def scales(self, copy: bool = True) -> np.ndarray:
        return self._float_array(
            self.lib.nimf_quant_scale, self.lib.nimf_quant_scale_len, copy)

    def zeros(self, copy: bool = True) -> np.ndarray:
        return self._float_array(
            self.lib.nimf_quant_zero, self.lib.nimf_quant_zero_len, copy)

    # -- conversion -----------------------------------------------------
    def dequantize(self, copy: bool = True) -> np.ndarray:
        """Dequantize to a float32 array with this tensor's shape."""
        n = self.numel
        out = np.empty(n, dtype=np.float32)
        got = int(self.lib.nimf_quant_dequantize(
            self.handle, out.ctypes.data_as(F32), n))
        if got < 0:
            self._raise()
        return out.reshape(self.shape)

    def dequantize_transposed(self) -> np.ndarray:
        """Dequantize a 2-D tensor directly into transposed ``[cols, rows]``."""
        shape = self.shape
        if len(shape) > 2:
            raise ValueError("transposed dequantization supports at most 2-D")
        out_shape = tuple(reversed(shape)) if len(shape) == 2 else shape
        n = self.numel
        out = np.empty(n, dtype=np.float32)
        got = int(self.lib.nimf_quant_dequantize_transposed(
            self.handle, out.ctypes.data_as(F32), n))
        if got < 0:
            self._raise()
        return out.reshape(out_shape)

    def numpy(self, copy: bool = True) -> np.ndarray:
        """Alias for :meth:`dequantize`."""
        return self.dequantize(copy=copy)

    # -- lifetime -------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.nimf_quant_destroy(self.handle)
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

    def _raise(self):
        msg = self.lib.nimf_last_error()
        raise NimformersError(
            msg.decode("utf-8", "replace") if msg else "quantization failed")

    def __repr__(self) -> str:
        try:
            return (f"QuantTensor(kind={self.kind!r}, shape={self.shape}, "
                    f"bytes={self.nbytes}, group_size={self.group_size})")
        except Exception:
            return "QuantTensor(<closed>)"
