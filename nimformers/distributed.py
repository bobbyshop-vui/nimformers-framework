"""Python binding for the native Nim/BybyLang device scheduler."""
from __future__ import annotations

import ctypes
import json
import math
import os
import threading
from operator import index
from pathlib import Path

import numpy as np

from ._lib import F32, U32, NimformersError, get_library, last_error
from .vm import VMProgram

__all__ = ["ProcessGroup", "init_process_group", "destroy_process_group"]


def _json(value) -> bytes:
    return json.dumps(value, allow_nan=False).encode("utf-8")


def _ratios(values):
    return None if values is None else [float(value) for value in values]


def _size(value) -> int:
    value = index(value)
    if not 0 <= value <= (1 << 63) - 1:
        raise ValueError("size or extent is out of range")
    return value


class ProcessGroup:
    """Own a native local process group and its learned workload weights."""

    def __init__(self, devices, *, ratios=None, library=None, worker_path=None):
        self.devices = (devices,) if isinstance(devices, str) else tuple(devices)
        self._lock = threading.Lock()
        self._handle = None
        self._lib = get_library(library)
        if not hasattr(self._lib, "nimf_parallel_create"):
            raise NimformersError("native device parallelism requires make build-shared")
        selected = worker_path or os.environ.get("NIMFORMERS_WORKER") or None
        if selected is None:
            name = "nimformers-worker.exe" if os.name == "nt" else "nimformers-worker"
            sibling = Path(self._lib._name).with_name(name)
            if sibling.is_file():
                selected = sibling.resolve()
        self._worker = None if selected is None else os.fsencode(selected)
        weights = _ratios(ratios)
        self._handle = self._lib.nimf_parallel_create(
            _json(self.devices), None if weights is None else _json(weights))
        if not self._handle:
            self._raise()

    def _raise(self):
        raise NimformersError(last_error(self._lib) or "native device-parallel operation failed")

    def _require_handle(self):
        if not self._handle:
            raise NimformersError("process group is closed")
        return self._handle

    def _read_json(self, ptr):
        if not ptr:
            self._raise()
        try:
            return json.loads(ctypes.string_at(ptr))
        finally:
            self._lib.nimf_free_string(ptr)

    def partition(self, length: int, *, ratios=None) -> tuple[int, ...]:
        """Preview counts from configured, explicit, or initial weights."""
        weights = _ratios(ratios)
        with self._lock:
            ptr = self._lib.nimf_parallel_partition(
                self._require_handle(), _size(length), None if weights is None else _json(weights))
            return tuple(self._read_json(ptr))

    def vm_run(self, program, memory, *, threads_per_item=None, tg_floats: int = 0,
               tg_size: int = 0, ratios=None) -> np.ndarray:
        """Run shard-local micro-VM bytecode over independent items along axis 0."""
        data = np.asarray(memory, dtype=np.float32)
        if data.ndim == 0:
            raise ValueError("memory must have an item axis")
        data = np.ascontiguousarray(data)
        words = program.words if isinstance(program, VMProgram) else np.asarray(list(program), dtype=np.uint32)
        if words.ndim != 1 or not words.size:
            raise ValueError("expected a nonempty sequence of VM program words")
        words = np.ascontiguousarray(words)
        size = math.prod(data.shape[1:])
        threads = size if threads_per_item is None else _size(threads_per_item)
        result = np.empty_like(data)
        weights = _ratios(ratios)
        with self._lock:
            status = self._lib.nimf_parallel_vm_run(
                self._require_handle(), words.ctypes.data_as(U32), words.size,
                data.ctypes.data_as(F32), data.size, size, threads, _size(tg_floats),
                _size(tg_size), None if weights is None else _json(weights), self._worker,
                result.ctypes.data_as(F32))
            if status != 0:
                self._raise()
        return result

    def generate_batch(self, cli_path, base_args, prompts, *, token_ids=False,
                       max_tokens=128, temperature=0.8, env=None, ratios=None) -> list[str]:
        """Run independent prompts with the native formers batch scheduler."""
        request = _json({
            "executable": os.fspath(cli_path), "args": list(base_args), "prompts": list(prompts),
            "token_ids": token_ids, "max_tokens": _size(max_tokens),
            "temperature": float(temperature), "ratios": _ratios(ratios),
            "env": {key: None if value is None else str(value) for key, value in (env or {}).items()},
        })
        with self._lock:
            return self._read_json(self._lib.nimf_parallel_generate(self._require_handle(), request))

    def close(self) -> None:
        """Release the native process group."""
        with self._lock:
            if self._handle:
                self._lib.nimf_parallel_destroy(self._handle)
                self._handle = None

    def __enter__(self) -> ProcessGroup:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self):  # pragma: no cover
        if getattr(self, "_handle", None):
            self.close()


def init_process_group(devices, *, ratios=None, library=None, worker_path=None) -> ProcessGroup:
    """Create a local group for the requested CPU and GPU ordinals."""
    return ProcessGroup(devices, ratios=ratios, library=library, worker_path=worker_path)


def destroy_process_group(group: ProcessGroup) -> None:
    """Release a group created by init_process_group."""
    group.close()
