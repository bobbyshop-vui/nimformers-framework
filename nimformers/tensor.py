"""Owned Nimformers tensor handles with a NumPy view."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._lib import F32, I64, NimformersError

__all__ = ["Tensor"]


class Tensor:
    """A float32 tensor allocated and owned by the Nimformers library.

    The common path is to let :class:`nimformers.Context` create tensors::

        t = ctx.tensor(np.zeros((2, 3), dtype=np.float32))
        t = ctx.zeros((2, 3))
        t = ctx.randn((4, 8), scale=0.02)

    ``numpy()`` returns a copy by default, so the result stays valid after
    ``close()``.  Pass ``copy=False`` for a zero-copy view whose lifetime is
    tied to the tensor handle.
    """

    def __init__(self, lib, handle, shape: Optional[Sequence[int]] = None):
        if not handle:
            raise NimformersError("failed to create Nimformers tensor")
        self.lib = lib
        self.handle = handle
        if shape is not None:
            self._shape_hint = tuple(int(d) for d in shape)
        else:
            self._shape_hint = None

    # -- introspection --------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        rank = int(self.lib.nimf_tensor_rank(self.handle))
        if rank < 0:
            self._raise()
        if rank == 0:
            return ()
        out = np.empty(rank, dtype=np.int64)
        got = int(self.lib.nimf_tensor_shape(
            self.handle, out.ctypes.data_as(I64), rank))
        if got < 0:
            self._raise()
        return tuple(int(d) for d in out[:got])

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        n = int(self.lib.nimf_tensor_numel(self.handle))
        if n < 0:
            self._raise()
        return n

    @property
    def dtype(self):
        return np.float32

    @property
    def ndim(self) -> int:
        return self.rank

    @property
    def size(self) -> int:
        return self.numel

    def numpy(self, copy: bool = True) -> np.ndarray:
        """Return the tensor data as a NumPy array."""
        n = int(self.lib.nimf_tensor_len(self.handle))
        if n < 0:
            self._raise()
        shape = self.shape
        if n == 0:
            return np.empty(shape, dtype=np.float32)
        ptr = self.lib.nimf_tensor_data(self.handle)
        if not ptr:
            self._raise()
        view = np.ctypeslib.as_array(ptr, shape=(n,)).reshape(shape)
        return view.copy() if copy else view

    def __array__(self, dtype=None, copy=None):
        arr = self.numpy(copy=True if copy is None else bool(copy))
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    # -- mutation -------------------------------------------------------
    def fill(self, value: float) -> "Tensor":
        self._check(self.lib.nimf_tensor_fill(self.handle, float(value)))
        return self

    def set_data(self, value) -> "Tensor":
        arr = np.ascontiguousarray(value, dtype=np.float32)
        if arr.size != self.numel:
            raise ValueError(
                f"expected {self.numel} values, got {arr.size}")
        self._check(self.lib.nimf_tensor_set_data(
            self.handle, arr.ctypes.data_as(F32), arr.size))
        return self

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        dims = np.ascontiguousarray(shape, dtype=np.int64)
        self._check(self.lib.nimf_tensor_reshape(
            self.handle, dims.ctypes.data_as(I64), len(dims)))
        return self

    def view(self, *shape) -> "Tensor":
        """Alias for :meth:`reshape` (shares the underlying data)."""
        return self.reshape(*shape)

    def clone(self) -> "Tensor":
        handle = self.lib.nimf_tensor_clone(self.handle)
        if not handle:
            self._raise()
        return Tensor(self.lib, handle, self.shape)

    # -- lifetime -------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.nimf_tensor_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> "Tensor":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self):  # pragma: no cover - interpreter shutdown
        try:
            self.close()
        except Exception:
            pass

    # -- helpers --------------------------------------------------------
    def _raise(self):
        msg = self.lib.nimf_last_error()
        raise NimformersError(
            msg.decode("utf-8", "replace") if msg else "tensor operation failed")

    def _check(self, status: int):
        if status != 0:
            self._raise()

    def __repr__(self) -> str:
        try:
            return f"Tensor(shape={self.shape}, dtype=float32)"
        except Exception:
            return "Tensor(<closed>)"
