"""Resident buffers and stateful sessions for micro-kernel models.

``Context.vm_run`` uploads a new arena for every call. A model needs weights and state that stay on
the GPU between dispatches, so this module exposes named resident buffers and a session that encodes
many dispatches before one submit. Each dispatch binds up to four buffers to the micro-VM domains
``mem`` (read/write float32), ``u8`` (bytes), ``lut`` and ``input`` (read-only float32); addresses in
the program are relative to the bound buffer (plus the bind offset).
"""
from __future__ import annotations

import ctypes as ct
import weakref
from typing import Optional, Tuple

import numpy as np

from ._lib import F32, V, NimformersError

__all__ = ["Buffer", "Session"]

DOMAINS = ("mem", "u8", "lut", "input")

# Sessions alive in this process; ``nf.trace`` re-registers their kernels when a recording starts.
_LIVE_SESSIONS: "weakref.WeakSet[Session]" = weakref.WeakSet()


def _raise(lib):
    msg = lib.nimf_last_error()
    raise NimformersError(msg.decode("utf-8", "replace") if msg else "session operation failed")


class Buffer:
    """A float32 buffer resident on the GPU."""

    def __init__(self, ctx, data=None, size: Optional[int] = None):
        self.ctx = ctx
        if data is not None:
            arr = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
            self.size = arr.size
            ptr = arr.ctypes.data_as(F32)
        else:
            if size is None:
                raise ValueError("size is required without data")
            arr, ptr, self.size = None, None, int(size)
        self.handle = ctx.lib.nimf_buf_alloc(ctx._ctx, ptr, self.size)
        if not self.handle:
            _raise(ctx.lib)

    def read(self, offset: int = 0, n: Optional[int] = None) -> np.ndarray:
        """Copy back n floats; only valid while no session is open."""
        n = self.size - offset if n is None else n
        out = np.empty(n, dtype=np.float32)
        if self.ctx.lib.nimf_buf_read(self.ctx._ctx, self.handle, offset, n, out.ctypes.data_as(F32)) != 0:
            _raise(self.ctx.lib)
        return out

    def close(self):
        if getattr(self, "handle", None):
            self.ctx.lib.nimf_buf_free(self.ctx._ctx, self.handle)
            self.handle = None

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


class Session:
    """One command buffer: ``with Session(ctx) as s: s.dispatch(...)`` submits on exit."""

    def __init__(self, ctx):
        self.ctx = ctx
        self._kernels = {}
        self._programs = {}  # key -> (program, tag), kept so a later trace can see their bytecode
        _LIVE_SESSIONS.add(self)

    def _jit(self, program, tag: str) -> str:
        ptr = self.ctx.lib.nimf_prog_compile(self.ctx._ctx, program.handle, tag.encode())
        if not ptr:
            _raise(self.ctx.lib)
        try:
            return ct.string_at(ptr).decode()
        finally:
            self.ctx.lib.nimf_free_string(ptr)

    def compile(self, program, tag: str = "py") -> str:
        """JIT a program (call before the session opens); cached per program object."""
        key = (id(program), tag)
        if key not in self._kernels:
            self._kernels[key] = self._jit(program, tag)
            self._programs[key] = (program, tag)
        return self._kernels[key]

    def _reregister(self) -> None:
        """Compile every known program again so a recording that started later sees its bytecode."""
        for key, (program, tag) in self._programs.items():
            self._kernels[key] = self._jit(program, tag)

    def __enter__(self):
        if self.ctx.lib.nimf_sess_begin(self.ctx._ctx) != 0:
            _raise(self.ctx.lib)
        return self

    def __exit__(self, exc_type, *_):
        status = self.ctx.lib.nimf_sess_end(self.ctx._ctx)
        if status != 0 and exc_type is None:
            _raise(self.ctx.lib)

    def set(self, buf: Buffer, offset: int, values):
        arr = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        if self.ctx.lib.nimf_sess_set(self.ctx._ctx, buf.handle, offset, arr.ctypes.data_as(F32), arr.size) != 0:
            _raise(self.ctx.lib)

    def step(self, token: int, position: int, kind: str = "decode"):
        """Context manager marking one decode/prefill step for the FIR recorder.

        A trace of this driver then carries the same ``step`` records as the Nim decode loops.
        """
        return _Step(self, kind, token, position)

    def set_step_inputs(self, buf: Buffer, token_offset: int, position_offset: int,
                        token: int, position: int):
        """Write the step's token and position scalars into ``buf`` (annotated for lifting)."""
        if self.ctx.lib.nimf_sess_set_step_inputs(self.ctx._ctx, buf.handle, token_offset,
                                                  position_offset, token, position) != 0:
            _raise(self.ctx.lib)

    def dispatch(self, kernel: str, *, threads: int, tg_size: int, tg_bytes: int = 16,
                 mem: Optional[Tuple[Buffer, int]] = None, u8=None, lut=None, input=None):
        """Encode one dispatch. Each binding is ``(Buffer, float_offset)`` or ``Buffer``."""
        bound = []
        for b in (mem, u8, lut, input):
            if b is None:
                bound.append((None, 0))
            elif isinstance(b, Buffer):
                bound.append((b, 0))
            else:
                bound.append(b)
        handles = (V * 4)(*[(b.handle if b is not None else None) for b, _ in bound])
        offsets = (ct.c_int64 * 4)(*[int(o) for _, o in bound])
        if self.ctx.lib.nimf_sess_dispatch(self.ctx._ctx, kernel.encode(), handles, offsets,
                                           threads, tg_size, tg_bytes) != 0:
            _raise(self.ctx.lib)


class _Step:
    def __init__(self, sess, kind, token, position):
        self.sess, self.kind, self.token, self.position = sess, kind, token, position

    def __enter__(self):
        lib = self.sess.ctx.lib
        if lib.nimf_probe_step(1, 1 if self.kind == "decode" else 0, self.position, self.position, self.token) != 0:
            _raise(lib)
        return self

    def __exit__(self, exc_type, *_):
        lib = self.sess.ctx.lib
        status = lib.nimf_probe_step(0, 0, 0, 0, 0)
        if status != 0 and exc_type is None:
            _raise(lib)
