"""Micro-VM bytecode builder and dispatch.

The micro-VM is Nimformers' low-level GPU kernel path: you emit a small
register-machine program (f32 registers ``R0..R15`` and u32 registers
``U0..U15``), and the backend lowers it to Metal/MSL or Vulkan/GLSL and
dispatches it over a flat float32 memory buffer.

Example: double every element in a buffer::

    from nimformers import Context, Op

    with Context() as ctx:
        prog = ctx.vm_program()
        prog.load_f32(1, 0, 0)     # R1 = mem[R0 + 0]  (R0 = thread id)
        prog.add(1, 1, 1)          # R1 = R1 + R1
        prog.store_f32(1, 0, 0)    # mem[R0 + 0] = R1
        prog.halt()
        data = ctx.vm_run(prog, np.ones(64, dtype=np.float32), tg_size=64)
"""
from __future__ import annotations

import ctypes as ct
import struct
from enum import IntEnum
from typing import Iterable, Optional

import numpy as np

from ._lib import F32, U32, V, NimformersError

__all__ = ["Op", "VMProgram"]


class Op(IntEnum):
    """Opcodes understood by :meth:`VMProgram.emit` (mirrors session_vm.nim)."""

    NOP = 0x00
    HALT = 0x01

    LOAD_F32 = 0x10
    STORE_F32 = 0x11
    LOAD_U8 = 0x12
    LOAD_CONST = 0x13
    LOAD_F32_IDX = 0x14
    STORE_F32_IDX = 0x15
    LOAD_LID = 0x16
    LOAD_TGID = 0x17

    ADD = 0x20
    MUL = 0x21
    FMA = 0x22
    SUB = 0x23
    EXP = 0x24
    NEG = 0x25
    SIGMOID = 0x26
    DIV = 0x27
    SOFTPLUS = 0x28
    RSQRT = 0x29
    ADD_CONST = 0x2A
    MUL_CONST = 0x2B

    LUT_U8 = 0x30
    FOR_BEGIN = 0x40
    FOR_END = 0x41
    BARRIER_TG = 0x50
    TG_LOAD_F32 = 0x60
    TG_STORE_F32 = 0x61

    CMP_LT = 0x70
    IF_BEGIN = 0x72
    IF_END = 0x73

    U_TGID = 0x80
    U_LID = 0x81
    U_CONST = 0x82
    U_ADD = 0x83
    U_MUL = 0x84
    U_ADDK = 0x85
    U_MULK = 0x86
    U_ANDK = 0x87
    U_SHRK = 0x88
    U_SHLK = 0x89
    U_SHR = 0x8A
    U_LOAD_U8 = 0x8B
    U_LOAD_CONSTF = 0x8C
    U_LOAD_IN = 0x8D
    U_STORE_F32 = 0x8E
    U_TO_R = 0x8F
    R_TO_U = 0x90
    U_LOAD_F32 = 0x91
    U_ATOMIC_ADD = 0x92
    GRID_BARRIER = 0x93
    BASE_F32 = 0x94
    BASE_U8 = 0x95
    BASE_LUT = 0x96
    BASE_IN = 0x97
    U_LOAD_META = 0x98
    VTG_BEGIN = 0x99
    VTG_END = 0x9A
    LOAD_VACT = 0x9B
    GRID_EPILOGUE = 0x9C
    U_LOAD_U32 = 0x9D


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


class VMProgram:
    """A micro-VM bytecode program owned by the Nimformers library."""

    def __init__(self, lib, handle=None):
        self.lib = lib
        if handle is None:
            handle = lib.nimf_vm_create()
            if not handle:
                self._raise()
        self.handle = handle

    # -- construction ---------------------------------------------------
    @classmethod
    def load(cls, lib, path) -> "VMProgram":
        handle = lib.nimf_vm_load(str(path).encode())
        if not handle:
            msg = lib.nimf_last_error()
            raise NimformersError(
                msg.decode("utf-8", "replace") if msg else "failed to load .bvm")
        return cls(lib, handle)

    @classmethod
    def merge(cls, lib, *programs: "VMProgram") -> "VMProgram":
        if not programs:
            return cls(lib)
        handle_arr = (V * len(programs))(*[p.handle for p in programs])
        handle = lib.nimf_vm_merge(handle_arr, len(programs))
        if not handle:
            msg = lib.nimf_last_error()
            raise NimformersError(
                msg.decode("utf-8", "replace") if msg else "failed to merge programs")
        return cls(lib, handle)

    # -- emission -------------------------------------------------------
    def emit(self, op, a: int = 0, b: int = 0, c: int = 0,
             value: int = 0, *, f32: Optional[float] = None) -> "VMProgram":
        """Emit one instruction.

        ``op`` may be an :class:`Op` or its integer value.  For the float
        immediate variants (``LOAD_CONST``, ``ADD_CONST``, ``MUL_CONST``)
        pass ``f32=`` instead of ``value=``.
        """
        if f32 is not None:
            value = _f32_bits(f32)
        status = self.lib.nimf_vm_emit(
            self.handle, int(op) & 0xFFFFFFFF, int(a) & 0xFFFFFFFF,
            int(b) & 0xFFFFFFFF, int(c) & 0xFFFFFFFF, int(value) & 0xFFFFFFFF)
        if status != 0:
            self._raise()
        return self

    def emit_word(self, word: int) -> "VMProgram":
        status = self.lib.nimf_vm_emit_word(self.handle, int(word) & 0xFFFFFFFF)
        if status != 0:
            self._raise()
        return self

    # Named helpers for the most common instructions.
    def nop(self): return self.emit(Op.NOP)
    def halt(self): return self.emit(Op.HALT)
    def load_f32(self, rd, rb, offset=0): return self.emit(Op.LOAD_F32, rd, rb, offset)
    def store_f32(self, rs, rb, offset=0): return self.emit(Op.STORE_F32, rs, rb, offset)
    def load_u8(self, rd, rb, offset=0): return self.emit(Op.LOAD_U8, rd, rb, offset)
    def load_const(self, rd, value): return self.emit(Op.LOAD_CONST, rd, f32=value)
    def add(self, rd, ra, rb): return self.emit(Op.ADD, rd, ra, rb)
    def sub(self, rd, ra, rb): return self.emit(Op.SUB, rd, ra, rb)
    def mul(self, rd, ra, rb): return self.emit(Op.MUL, rd, ra, rb)
    def div(self, rd, ra, rb): return self.emit(Op.DIV, rd, ra, rb)
    def fma(self, rd, ra, rb, rc): return self.emit(Op.FMA, rd, ra, rb, rc)
    def exp(self, rd, rb): return self.emit(Op.EXP, rd, rb)
    def neg(self, rd, rb): return self.emit(Op.NEG, rd, rb)
    def sigmoid(self, rd, rb): return self.emit(Op.SIGMOID, rd, rb)
    def softplus(self, rd, rb): return self.emit(Op.SOFTPLUS, rd, rb)
    def rsqrt(self, rd, rb): return self.emit(Op.RSQRT, rd, rb)
    def add_const(self, rd, rb, value): return self.emit(Op.ADD_CONST, rd, rb, f32=value)
    def mul_const(self, rd, rb, value): return self.emit(Op.MUL_CONST, rd, rb, f32=value)
    def lut_u8(self, rd, ri): return self.emit(Op.LUT_U8, rd, ri)
    def for_begin(self, count_reg): return self.emit(Op.FOR_BEGIN, count_reg)
    def for_end(self): return self.emit(Op.FOR_END)
    def barrier_tg(self): return self.emit(Op.BARRIER_TG)
    def tg_load_f32(self, rd, rb, offset=0): return self.emit(Op.TG_LOAD_F32, rd, rb, offset)
    def tg_store_f32(self, rs, rb, offset=0): return self.emit(Op.TG_STORE_F32, rs, rb, offset)
    def cmp_lt(self, rd, ra, rb): return self.emit(Op.CMP_LT, rd, ra, rb)
    def if_begin(self, cond): return self.emit(Op.IF_BEGIN, cond)
    def if_end(self): return self.emit(Op.IF_END)
    def load_lid(self, rd): return self.emit(Op.LOAD_LID, rd)
    def load_tgid(self, rd): return self.emit(Op.LOAD_TGID, rd)
    def grid_barrier(self, ctrl): return self.emit(Op.GRID_BARRIER, ctrl)
    def grid_epilogue(self, ctrl): return self.emit(Op.GRID_EPILOGUE, ctrl)
    def vtg_begin(self, mode, n_virt, param=0): return self.emit(Op.VTG_BEGIN, mode, n_virt, param)
    def vtg_end(self): return self.emit(Op.VTG_END)
    def load_vact(self, rd): return self.emit(Op.LOAD_VACT, rd)

    # -- data -----------------------------------------------------------
    @property
    def words(self) -> np.ndarray:
        n = int(self.lib.nimf_vm_len(self.handle))
        if n < 0:
            self._raise()
        if n == 0:
            return np.empty(0, dtype=np.uint32)
        out = np.empty(n, dtype=np.uint32)
        got = int(self.lib.nimf_vm_words(
            self.handle, out.ctypes.data_as(U32), n))
        if got < 0:
            self._raise()
        return out[:got]

    def set_words(self, words: Iterable[int]) -> "VMProgram":
        arr = np.ascontiguousarray(list(words), dtype=np.uint32)
        status = self.lib.nimf_vm_set_words(
            self.handle, arr.ctypes.data_as(U32), arr.size)
        if status != 0:
            self._raise()
        return self

    # -- execution ------------------------------------------------------
    def run(self, ctx, memory, n_threads: Optional[int] = None,
            tg_floats: int = 0, tg_size: int = 0) -> np.ndarray:
        """Dispatch the program over ``memory`` (float32) and return the result.

        If ``memory`` is a C-contiguous float32 ndarray it is updated in
        place.  ``tg_size`` forces the threads-per-threadgroup; 0 lets the
        backend choose.  ``tg_floats`` is the number of float32 slots of
        threadgroup scratch memory used by the kernel.
        """
        mem = np.ascontiguousarray(memory, dtype=np.float32)
        if mem.ndim == 0:
            raise ValueError("memory must be at least 1-D")
        flat = mem.reshape(-1)
        threads = flat.size if n_threads is None else int(n_threads)
        if threads <= 0:
            raise ValueError("n_threads must be positive")
        status = self.lib.nimf_vm_run(
            ctx._ctx, self.handle, flat.ctypes.data_as(F32), flat.size,
            threads, int(tg_floats), int(tg_size))
        if status != 0:
            self._raise()
        result = flat.reshape(mem.shape)
        if isinstance(memory, np.ndarray) and memory is not mem:
            try:
                np.copyto(memory, result)
            except (ValueError, TypeError):
                pass
        return result

    # -- persistence / lowering ----------------------------------------
    def save(self, path) -> None:
        if self.lib.nimf_vm_save(self.handle, str(path).encode()) != 0:
            self._raise()

    def lower_msl(self, kernel_name: str = "vm_kernel") -> str:
        return self._take_string(self.lib.nimf_vm_lower_msl(
            self.handle, kernel_name.encode()))

    def lower_glsl(self, kernel_name: str = "vm_kernel") -> str:
        return self._take_string(self.lib.nimf_vm_lower_glsl(
            self.handle, kernel_name.encode()))

    def key(self, tag: str = "") -> str:
        return self._take_string(self.lib.nimf_vm_key(
            self.handle, tag.encode()))

    # -- lifetime -------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.nimf_vm_destroy(self.handle)
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
        n = int(self.lib.nimf_vm_len(self.handle))
        return n if n >= 0 else 0

    # -- helpers --------------------------------------------------------
    def _take_string(self, ptr) -> str:
        if not ptr:
            self._raise()
        try:
            return ct.string_at(ptr).decode("utf-8", "replace")
        finally:
            self.lib.nimf_free_string(ptr)

    def _raise(self):
        msg = self.lib.nimf_last_error()
        raise NimformersError(
            msg.decode("utf-8", "replace") if msg else "micro-VM operation failed")

    def __repr__(self) -> str:
        try:
            return f"VMProgram(words={len(self)})"
        except Exception:
            return "VMProgram(<closed>)"
