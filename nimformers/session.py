"""Resident buffers and stateful sessions for native operations and micro-kernels.

``Context.vm_run`` uploads a new arena for every call. A model needs weights and state that stay on
the GPU between dispatches, so this module exposes named resident buffers and a session that encodes
many dispatches before one submit. Each dispatch binds up to four buffers to the micro-VM domains
``mem`` (read/write float32), ``u8`` (bytes), ``lut`` and ``input`` (read-only float32); addresses in
the program are relative to the bound buffer (plus the bind offset).
"""
from __future__ import annotations

import ctypes as ct
import math
import weakref
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from ._lib import F32, V, NimformersError

if TYPE_CHECKING:
    from .core import ResidentQ4Weights

__all__ = ["Buffer", "Session"]

DOMAINS = ("mem", "u8", "lut", "input")

# Sessions alive in this process; ``nf.trace`` re-registers their kernels when a recording starts.
_LIVE_SESSIONS: "weakref.WeakSet[Session]" = weakref.WeakSet()


def _raise(lib):
    msg = lib.nimf_last_error()
    raise NimformersError(msg.decode("utf-8", "replace") if msg else "session operation failed")


class Buffer:
    """A float32 buffer resident on the context's backend."""

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

    def __enter__(self) -> Buffer:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


class Session:
    """Encode native operations or micro-kernel dispatches and submit on exit."""

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

    def _encode(self, symbol: str, *args) -> None:
        fn = getattr(self.ctx.lib, symbol, None)
        if fn is None:
            raise NimformersError(f"missing {symbol}; rebuild with `make build-for-python`")
        if fn(self.ctx._ctx, *args) != 0:
            _raise(self.ctx.lib)

    def copy_into(self, src: Buffer, dst: Buffer, n: int, *,
                  src_offset: int = 0, dst_offset: int = 0) -> None:
        """Copy a range of floats between resident buffers within this session."""
        self._encode("nimf_sess_copy", src.handle, dst.handle, src_offset, dst_offset, n)

    def matmul(self, a: Buffer, b: Buffer, out: Buffer, *,
               m: int, k: int, n: int, transpose_b: bool = False) -> None:
        """Write an (m, n) product to out. With transpose_b, b is stored as (n, k)."""
        self._encode("nimf_sess_matmul", a.handle, b.handle, out.handle, m, k, n,
                     int(transpose_b))

    def matmul_q4(self, x: Buffer, weight: ResidentQ4Weights, out: Buffer, *,
                  rows: int) -> None:
        """Multiply (rows, k) activations by an already resident int4_asym weight."""
        self._encode("nimf_sess_matmul_q4", x.handle, weight.handle, out.handle, rows)

    def softmax(self, x: Buffer, out: Buffer, *, rows: int, cols: int) -> None:
        """Normalize each row of a row-major matrix."""
        self._encode("nimf_sess_softmax", x.handle, out.handle, rows, cols)

    def layernorm(self, x: Buffer, gamma: Buffer, beta: Buffer, out: Buffer, *,
                  rows: int, cols: int, eps: float = 1e-5) -> None:
        """Apply row-wise LayerNorm with column-wise gamma and beta."""
        if beta.handle is None:
            raise NimformersError("LayerNorm requires an open beta buffer")
        self._encode("nimf_sess_norm", x.handle, gamma.handle, beta.handle, out.handle,
                     rows, cols, eps)

    def rmsnorm(self, x: Buffer, gamma: Buffer, out: Buffer, *,
                rows: int, cols: int, eps: float = 1e-5) -> None:
        """Apply row-wise RMSNorm with column-wise gamma."""
        self._encode("nimf_sess_norm", x.handle, gamma.handle, None, out.handle,
                     rows, cols, eps)

    def attention(self, q: Buffer, k: Buffer, v: Buffer, out: Buffer, scores: Buffer, *,
                  batch: int, heads: int, steps: int, dim: int,
                  scale: Optional[float] = None) -> None:
        """Run causal attention on (batch, heads, steps, dim) inputs."""
        if dim <= 0:
            raise ValueError("attention head dimension must be positive")
        scale = 1.0 / math.sqrt(dim) if scale is None else scale
        self._encode("nimf_sess_attention", q.handle, k.handle, v.handle, out.handle,
                     scores.handle, batch, heads, steps, dim, scale)

    def vec_op(self, op: str, a: Buffer, b: Buffer, out: Buffer) -> None:
        """Apply add, sub, mul or div to the first out.size elements."""
        self._encode("nimf_sess_vec", a.handle, b.handle, out.handle, op.encode(), out.size)

    def activation(self, op: str, x: Buffer, out: Buffer, *, p: float = 0.0) -> None:
        """Apply relu, sigmoid, tanh, exp, log, sqrt or pow (exponent p)."""
        self._encode("nimf_sess_activation", x.handle, out.handle, op.encode(), out.size, p)

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
