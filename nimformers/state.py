"""NIMQ state-dict loading/saving."""
from __future__ import annotations

import ctypes as ct
from typing import Iterator, Sequence

import numpy as np

from ._lib import I64, NimformersError
from .quant import QuantTensor

__all__ = ["StateDict"]


class StateDict:
    """A NIMQ2 checkpoint: an architecture tuple plus named quantized tensors.

    Load (memory-mapped by default)::

        sd = ctx.load_state_dict("model.nimq")
        for name in sd:
            print(name, sd[name].shape)

    Build and save::

        sd = ctx.create_state_dict(arch=(32, 4096, 11008, 32, 32000))
        sd["model.layers.0.self_attn.q_proj.weight"] = qt
        sd.save("model.nimq")
    """

    def __init__(self, lib, handle, *, owned: bool = True):
        if not handle:
            raise NimformersError("failed to open Nimformers state dict")
        self.lib = lib
        self.handle = handle
        self._owned = owned
        self._cache: dict[int, QuantTensor] = {}

    # -- construction ---------------------------------------------------
    @classmethod
    def load(cls, lib, path, mapped: bool = True) -> "StateDict":
        handle = lib.nimf_state_dict_load(str(path).encode(), int(mapped))
        if not handle:
            msg = lib.nimf_last_error()
            raise NimformersError(
                msg.decode("utf-8", "replace") if msg else "state dict load failed")
        return cls(lib, handle)

    @classmethod
    def create(cls, lib, arch: Sequence[int] = (0, 0, 0, 0, 0)) -> "StateDict":
        handle = lib.nimf_state_dict_create()
        if not handle:
            msg = lib.nimf_last_error()
            raise NimformersError(
                msg.decode("utf-8", "replace") if msg else "state dict create failed")
        sd = cls(lib, handle)
        sd.arch = arch
        return sd

    # -- metadata -------------------------------------------------------
    @property
    def arch(self) -> tuple[int, ...]:
        out = np.empty(5, dtype=np.int64)
        if self.lib.nimf_state_dict_arch(
                self.handle, out.ctypes.data_as(I64), 5) != 0:
            self._raise()
        return tuple(int(v) for v in out)

    @arch.setter
    def arch(self, value: Sequence[int]) -> None:
        arch = np.ascontiguousarray(value, dtype=np.int64)
        if arch.size != 5:
            raise ValueError("arch must contain exactly 5 integers")
        arch = arch.reshape(5)
        if self.lib.nimf_state_dict_set_arch(
                self.handle, arch.ctypes.data_as(I64)) != 0:
            self._raise()

    @property
    def names(self) -> list[str]:
        count = len(self)
        result = []
        for i in range(count):
            ptr = self.lib.nimf_state_dict_name(self.handle, i)
            if not ptr:
                self._raise()
            try:
                result.append(ct.string_at(ptr).decode("utf-8", "replace"))
            finally:
                self.lib.nimf_free_string(ptr)
        return result

    def keys(self) -> list[str]:
        return self.names

    # -- element access -------------------------------------------------
    def _index(self, key) -> int:
        if isinstance(key, (int, np.integer)):
            idx = int(key)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("state-dict index out of range")
            return idx
        for i, name in enumerate(self.names):
            if name == key:
                return i
        raise KeyError(key)

    def __getitem__(self, key) -> QuantTensor:
        idx = self._index(key)
        cached = self._cache.get(idx)
        if cached is not None and cached.handle:
            return cached
        handle = self.lib.nimf_state_dict_tensor(self.handle, idx)
        if not handle:
            self._raise()
        qt = QuantTensor(self.lib, handle)
        self._cache[idx] = qt
        return qt

    def __setitem__(self, name: str, quant: QuantTensor) -> None:
        self.append(name, quant)

    def append(self, name: str, quant: QuantTensor) -> None:
        if not isinstance(quant, QuantTensor):
            raise TypeError("state-dict values must be QuantTensor instances")
        status = self.lib.nimf_state_dict_append(
            self.handle, str(name).encode(), quant.handle)
        if status != 0:
            self._raise()

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def values(self) -> list[QuantTensor]:
        return [self[i] for i in range(len(self))]

    def items(self):
        return [(name, self[name]) for name in self.names]

    def to_dict(self) -> dict[str, np.ndarray]:
        """Dequantize every tensor; convenient for inspection of small models."""
        return {name: self[name].dequantize() for name in self.names}

    # -- persistence ----------------------------------------------------
    def save(self, path) -> None:
        arch = np.ascontiguousarray(self.arch, dtype=np.int64)
        status = self.lib.nimf_state_dict_save(
            str(path).encode(), self.handle, arch.ctypes.data_as(I64))
        if status != 0:
            self._raise()

    # -- lifetime -------------------------------------------------------
    def close(self) -> None:
        for qt in self._cache.values():
            try:
                qt.close()
            except Exception:
                pass
        self._cache.clear()
        if getattr(self, "handle", None) and self._owned:
            self.lib.nimf_state_dict_destroy(self.handle)
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

    def __len__(self) -> int:
        n = int(self.lib.nimf_state_dict_count(self.handle))
        if n < 0:
            self._raise()
        return n

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __contains__(self, key) -> bool:
        try:
            self._index(key)
            return True
        except (KeyError, IndexError):
            return False

    def _raise(self):
        msg = self.lib.nimf_last_error()
        raise NimformersError(
            msg.decode("utf-8", "replace") if msg else "state-dict operation failed")

    def __repr__(self) -> str:
        try:
            return f"StateDict(arch={self.arch}, tensors={len(self)})"
        except Exception:
            return "StateDict(<closed>)"
