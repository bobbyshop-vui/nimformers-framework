"""FORGE-500M-Chat decode model written as a Python class on top of micro-kernels.

FORGE mixes three kinds of blocks: ARG (selective SSM + sliding-window GQA, blended by a
learned gate), global GQA attention (layers 5/11/17) and two FFN kinds (dense SwiGLU on odd
layers, hierarchical top-2 MoE "HSE" on even layers).  Every compute step is a hand-emitted
micro-VM program dispatched on stateful Metal buffers through ``Session``.  One token = one
session step.  The model is meant to be traced (``BYBY_FIR_PROBE``) and lifted unchanged.

Layout
  * ``arena``  (domain ``mem``): activations, router state, rope tables, conv/SSM states, KV caches.
  * weights    (domain ``input``): one resident f32 ``Buffer`` per tensor (bf16 -> f32).
  * experts are packed per layer (d_ff padded 210 -> 256 with zeros) so the chosen expert is a
    data-dependent address.

Usage:  python forge_vm.py MODEL_DIR [--tokens 1,100,200] [--new 3] [--check]
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np

from nimformers import Buffer, Context, Op, Session

MAXSEQ = 64  # KV-cache capacity (= attention threadgroup width = local window)
H = 1024
NG = 16  # DGN groups
GS = H // NG
FPAD = 256  # padded expert hidden width


# --------------------------------------------------------------------------- weights
class _Tag:
    def __init__(self, name):
        self.name = name


def _rebuild(storage, offset, size, stride, *rest):
    return ("T", storage, offset, tuple(size), tuple(stride))


class _SafeUnpickler(pickle.Unpickler):
    """Allowlist-only unpickler: tensor plumbing is the only thing it will resolve."""

    def find_class(self, mod, name):
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild
        if mod == "torch" and name.endswith("Storage"):
            return _Tag(name)
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"blocked global {mod}.{name}")

    def persistent_load(self, pid):
        return ("S", pid[1].name, str(pid[2]), pid[4])


class PtWeights:
    """Reads a bf16 ``torch.save`` state dict without torch (zip + restricted pickle)."""

    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        pk = [n for n in self.z.namelist() if n.endswith("data.pkl")][0]
        self.prefix = pk[: -len("data.pkl")]
        self.sd = _SafeUnpickler(self.z.open(pk)).load()

    def key(self, name):
        return self.sd[name][1][2]

    def get(self, name) -> np.ndarray:
        _, (_, stype, key, _n), offset, size, stride = self.sd[name]
        assert stype == "BFloat16Storage", stype
        raw = np.frombuffer(self.z.read(f"{self.prefix}data/{key}"), dtype=np.uint16)
        n = int(np.prod(size)) if size else 1
        raw = raw[offset: offset + n]
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(size)


def bf16_round(a):
    u = np.asarray(a, dtype=np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def softplus(x):
    return np.logaddexp(0.0, x).astype(np.float32)


# --------------------------------------------------------------------------- emit helpers
def u_gid(p, ud, tmp, tg):
    p.emit(Op.U_TGID, ud)
    p.emit(Op.U_MULK, ud, ud, value=tg)
    p.emit(Op.U_LID, tmp)
    p.emit(Op.U_ADD, ud, ud, tmp)


def tg_reduce(p, T, val, lidf, zero, r1, r2, r3, r4, mode):
    """Reduce ``val`` over the threadgroup (sum or max); every lane gets the result."""
    p.emit(Op.TG_STORE_F32, val, lidf, c=0)
    p.emit(Op.BARRIER_TG)
    s = T // 2
    while s >= 1:
        p.emit(Op.LOAD_CONST, r4, f32=float(s))
        p.emit(Op.CMP_LT, r3, lidf, r4)
        p.emit(Op.IF_BEGIN, r3)
        p.emit(Op.TG_LOAD_F32, r1, lidf, c=0)
        p.emit(Op.TG_LOAD_F32, r2, lidf, c=s)
        if mode == "sum":
            p.emit(Op.ADD, r1, r1, r2)
        else:  # max(a, b) = a + (a < b) * (b - a)
            p.emit(Op.CMP_LT, r3, r1, r2)
            p.emit(Op.SUB, r2, r2, r1)
            p.emit(Op.FMA, r1, r3, r2, value=r1)
        p.emit(Op.TG_STORE_F32, r1, lidf, c=0)
        p.emit(Op.IF_END)
        p.emit(Op.BARRIER_TG)
        s //= 2
    p.emit(Op.TG_LOAD_F32, val, zero, c=0)
    p.emit(Op.BARRIER_TG)


class Arena:
    """Static layout of the mem arena (float offsets)."""

    def __init__(self, cfg, mha_layers, arg_layers):
        off = 0

        def take(n):
            nonlocal off
            o = off
            off += (n + 63) // 64 * 64
            return o

        self.V = cfg["vocab_size"]
        self.TOK = take(1)
        self.POS = take(1)
        self.X = take(H)
        self.HN = take(H)
        # ARG scratch
        self.XZ = take(2 * H)
        self.XC = take(H)
        self.XDBL = take(144)
        self.DT = take(H)
        self.Y = take(H)
        self.REC = take(H)
        self.LQKV = take(768)
        self.LATT = take(512)
        self.LOC = take(H)
        self.GATE = take(1)
        # MHA scratch
        self.QKV = take(1536)
        self.ATT = take(H)
        # FFN scratch
        self.G = take(2560)
        self.RL1 = take(4)
        self.RL2 = take(8)
        self.DOM = take(1)
        self.SEL = [take(1), take(1)]
        self.W = [take(1), take(1)]
        self.EG = take(FPAD)
        # rope tables
        self.COS = take(MAXSEQ * 32)
        self.SIN = take(MAXSEQ * 32)
        # per-layer persistent state
        self.CS = {i: take(3 * H) for i in arg_layers}
        self.HS = {i: take(H * 48) for i in arg_layers}
        self.LKV = {i: take(2 * 2 * MAXSEQ * 64) for i in arg_layers}
        self.KV = {i: take(2 * 4 * MAXSEQ * 64) for i in mha_layers}
        self.size = off


# --------------------------------------------------------------------------- kernels
class Kernels:
    def __init__(self, ctx, sess, A):
        self.ctx, self.s, self.A = ctx, sess, A
        self.cache = {}

    def prog(self, tag, build):
        if tag not in self.cache:
            p = self.ctx.vm_program()
            build(p)
            p.emit(Op.HALT)
            self.cache[tag] = self.s.compile(p, tag)
        return self.cache[tag]

    # token embedding gather: X = embed[TOK]
    def embed(self):
        A = self.A

        def b(p):
            u_gid(p, 0, 1, 256)
            p.emit(Op.U_CONST, 2, value=0)
            p.emit(Op.U_LOAD_F32, 0, 2, value=A.TOK)
            p.emit(Op.R_TO_U, 3, 0)
            p.emit(Op.U_MULK, 3, 3, value=H)
            p.emit(Op.U_ADD, 3, 3, 0)
            p.emit(Op.U_LOAD_IN, 1, 3, value=0)
            p.emit(Op.U_STORE_F32, 1, 0, value=A.X)

        return self.prog("embed", b), H, 256, 16

    # differential group norm: one threadgroup per group, weight buffer = [w(1024) | b(1024)]
    def dgn(self, x_off, o_off, eps, tag):
        def b(p):
            p.emit(Op.U_TGID, 0)
            p.emit(Op.U_LID, 1)
            p.emit(Op.LOAD_LID, 20)
            p.emit(Op.LOAD_CONST, 21, f32=0.0)
            p.emit(Op.U_MULK, 2, 0, value=GS)
            p.emit(Op.U_ADD, 2, 2, 1)
            p.emit(Op.U_LOAD_F32, 5, 2, value=x_off)
            p.emit(Op.MUL, 6, 5, 5)
            tg_reduce(p, GS, 6, 20, 21, 13, 14, 15, 16, "sum")
            p.emit(Op.MUL_CONST, 6, 6, f32=1.0 / GS)
            p.emit(Op.ADD_CONST, 6, 6, f32=eps)
            p.emit(Op.RSQRT, 12, 6)
            p.emit(Op.MUL, 5, 5, 12)
            p.emit(Op.U_LOAD_IN, 7, 2, value=0)
            p.emit(Op.MUL, 5, 5, 7)
            p.emit(Op.U_ADDK, 3, 2, value=H)
            p.emit(Op.U_LOAD_IN, 8, 3, value=0)
            p.emit(Op.ADD, 5, 5, 8)
            p.emit(Op.U_STORE_F32, 5, 2, value=o_off)

        return self.prog(tag, b), NG * GS, GS, 4 * GS

    # y = W x (thread group per output row); optional bias / softplus / accumulate / expert select
    # / swiglu pair / output scale.  With bias the weight buffer is [W (N*K) | b (N)].
    def matvec(self, N, K, x_off, y_off, tag, *, acc=False, expert_stride=None, sel_off=None,
               dual=False, T=256, x_lut=False, bias=False, softplus_out=False, scale_off=None):
        iters = K // T
        assert iters * T == K, (N, K, T)

        def b(p):
            p.emit(Op.U_TGID, 0)
            p.emit(Op.U_LID, 1)
            p.emit(Op.LOAD_LID, 20)
            p.emit(Op.LOAD_CONST, 21, f32=0.0)
            p.emit(Op.LOAD_CONST, 22, f32=1.0)
            p.emit(Op.U_MULK, 2, 0, value=K)
            p.emit(Op.U_ADD, 2, 2, 1)
            if expert_stride is not None:
                p.emit(Op.U_CONST, 4, value=0)
                p.emit(Op.U_LOAD_F32, 3, 4, value=sel_off)
                p.emit(Op.R_TO_U, 5, 3)
                p.emit(Op.U_MULK, 5, 5, value=expert_stride)
                p.emit(Op.U_ADD, 2, 2, 5)
            if dual:
                p.emit(Op.U_ADDK, 6, 2, value=N * K)
            p.emit(Op.U_ADDK, 7, 1, value=0)
            p.emit(Op.LOAD_CONST, 10, f32=0.0)
            p.emit(Op.LOAD_CONST, 11, f32=0.0)
            p.emit(Op.LOAD_CONST, 8, f32=float(iters))
            p.emit(Op.FOR_BEGIN, 8)
            p.emit(Op.U_LOAD_CONSTF if x_lut else Op.U_LOAD_F32, 5, 7, value=x_off)
            p.emit(Op.U_LOAD_IN, 4, 2, value=0)
            p.emit(Op.FMA, 10, 4, 5, value=10)
            p.emit(Op.U_ADDK, 2, 2, value=T)
            if dual:
                p.emit(Op.U_LOAD_IN, 4, 6, value=0)
                p.emit(Op.FMA, 11, 4, 5, value=11)
                p.emit(Op.U_ADDK, 6, 6, value=T)
            p.emit(Op.U_ADDK, 7, 7, value=T)
            p.emit(Op.FOR_END)
            tg_reduce(p, T, 10, 20, 21, 13, 14, 15, 16, "sum")
            if dual:
                tg_reduce(p, T, 11, 20, 21, 13, 14, 15, 16, "sum")
                p.emit(Op.SIGMOID, 12, 10)
                p.emit(Op.MUL, 10, 10, 12)
                p.emit(Op.MUL, 10, 10, 11)
            p.emit(Op.CMP_LT, 13, 20, 22)  # lane 0 only
            p.emit(Op.IF_BEGIN, 13)
            if bias:
                p.emit(Op.U_ADDK, 8, 0, value=N * K)
                p.emit(Op.U_LOAD_IN, 14, 8, value=0)
                p.emit(Op.ADD, 10, 10, 14)
            if softplus_out:
                p.emit(Op.SOFTPLUS, 10, 10)
            if scale_off is not None:
                p.emit(Op.U_CONST, 9, value=0)
                p.emit(Op.U_LOAD_F32, 15, 9, value=scale_off)
                p.emit(Op.MUL, 10, 10, 15)
            if acc:
                p.emit(Op.U_LOAD_F32, 14, 0, value=y_off)
                p.emit(Op.ADD, 10, 10, 14)
            p.emit(Op.U_STORE_F32, 10, 0, value=y_off)
            p.emit(Op.IF_END)

        return self.prog(tag, b), N * T, T, 4 * T

    # causal depthwise conv (k=4) + silu with a 3-token history; weight = [w(1024*4) | bias(1024)]
    def conv(self, layer):
        A = self.A
        cs = A.CS[layer]

        def b(p):
            u_gid(p, 0, 1, 256)
            p.emit(Op.U_LOAD_F32, 1, 0, value=A.XZ)
            p.emit(Op.U_LOAD_F32, 2, 0, value=cs)
            p.emit(Op.U_LOAD_F32, 3, 0, value=cs + H)
            p.emit(Op.U_LOAD_F32, 4, 0, value=cs + 2 * H)
            p.emit(Op.U_MULK, 2, 0, value=4)
            p.emit(Op.U_LOAD_IN, 5, 2, value=0)
            p.emit(Op.U_ADDK, 3, 2, value=1)
            p.emit(Op.U_LOAD_IN, 6, 3, value=0)
            p.emit(Op.U_ADDK, 3, 2, value=2)
            p.emit(Op.U_LOAD_IN, 7, 3, value=0)
            p.emit(Op.U_ADDK, 3, 2, value=3)
            p.emit(Op.U_LOAD_IN, 8, 3, value=0)
            p.emit(Op.U_ADDK, 3, 0, value=4 * H)
            p.emit(Op.U_LOAD_IN, 9, 3, value=0)
            p.emit(Op.FMA, 9, 5, 2, value=9)
            p.emit(Op.FMA, 9, 6, 3, value=9)
            p.emit(Op.FMA, 9, 7, 4, value=9)
            p.emit(Op.FMA, 9, 8, 1, value=9)
            p.emit(Op.SIGMOID, 10, 9)
            p.emit(Op.MUL, 10, 10, 9)
            p.emit(Op.U_STORE_F32, 10, 0, value=A.XC)
            p.emit(Op.U_STORE_F32, 3, 0, value=cs)
            p.emit(Op.U_STORE_F32, 4, 0, value=cs + H)
            p.emit(Op.U_STORE_F32, 1, 0, value=cs + 2 * H)

        return self.prog(f"conv{layer}", b), H, 256, 16

    # one selective-scan step per channel; weight = [D(1024) | A(48)]
    def ssm(self, layer):
        A = self.A
        hs = A.HS[layer]

        def b(p):
            u_gid(p, 0, 1, 256)
            p.emit(Op.U_LOAD_F32, 1, 0, value=A.DT)
            p.emit(Op.U_LOAD_F32, 2, 0, value=A.XC)
            p.emit(Op.MUL, 3, 1, 2)
            p.emit(Op.U_MULK, 2, 0, value=48)
            p.emit(Op.U_CONST, 3, value=0)
            p.emit(Op.U_CONST, 4, value=H)
            p.emit(Op.LOAD_CONST, 10, f32=0.0)
            p.emit(Op.LOAD_CONST, 11, f32=48.0)
            p.emit(Op.FOR_BEGIN, 11)
            p.emit(Op.U_LOAD_IN, 4, 4, value=0)
            p.emit(Op.U_LOAD_F32, 5, 3, value=A.XDBL + 48)
            p.emit(Op.U_LOAD_F32, 6, 3, value=A.XDBL + 96)
            p.emit(Op.U_LOAD_F32, 7, 2, value=hs)
            p.emit(Op.MUL, 8, 1, 4)
            p.emit(Op.EXP, 8, 8)
            p.emit(Op.MUL, 9, 3, 5)
            p.emit(Op.FMA, 7, 8, 7, value=9)
            p.emit(Op.U_STORE_F32, 7, 2, value=hs)
            p.emit(Op.FMA, 10, 7, 6, value=10)
            p.emit(Op.U_ADDK, 2, 2, value=1)
            p.emit(Op.U_ADDK, 3, 3, value=1)
            p.emit(Op.U_ADDK, 4, 4, value=1)
            p.emit(Op.FOR_END)
            p.emit(Op.U_LOAD_IN, 12, 0, value=0)
            p.emit(Op.FMA, 10, 2, 12, value=10)
            p.emit(Op.U_LOAD_F32, 13, 0, value=A.XZ + H)
            p.emit(Op.SIGMOID, 14, 13)
            p.emit(Op.MUL, 14, 14, 13)
            p.emit(Op.MUL, 10, 10, 14)
            p.emit(Op.U_STORE_F32, 10, 0, value=A.Y)

        return self.prog(f"ssm{layer}", b), H, 256, 16

    # X += a*local + (1-a)*recurrent, a = sigmoid(gate)
    def blend(self):
        A = self.A

        def b(p):
            u_gid(p, 0, 1, 256)
            p.emit(Op.U_CONST, 2, value=0)
            p.emit(Op.U_LOAD_F32, 1, 2, value=A.GATE)
            p.emit(Op.SIGMOID, 1, 1)
            p.emit(Op.U_LOAD_F32, 2, 0, value=A.LOC)
            p.emit(Op.U_LOAD_F32, 3, 0, value=A.REC)
            p.emit(Op.SUB, 4, 2, 3)
            p.emit(Op.FMA, 5, 1, 4, value=3)
            p.emit(Op.U_LOAD_F32, 6, 0, value=A.X)
            p.emit(Op.ADD, 6, 6, 5)
            p.emit(Op.U_STORE_F32, 6, 0, value=A.X)

        return self.prog("blend", b), H, 256, 16

    # in-place rotate-half rope over `heads` heads of width 64 at the current position
    def rope(self, qkv_off, heads, tag):
        A = self.A
        H2 = 32

        def b(p):
            u_gid(p, 0, 9, 64)
            p.emit(Op.U_SHRK, 1, 0, value=H2.bit_length() - 1)
            p.emit(Op.U_MULK, 1, 1, value=64)
            p.emit(Op.U_ANDK, 2, 0, value=H2 - 1)
            p.emit(Op.U_ADD, 3, 1, 2)
            p.emit(Op.U_CONST, 4, value=0)
            p.emit(Op.U_LOAD_F32, 0, 4, value=A.POS)
            p.emit(Op.R_TO_U, 5, 0)
            p.emit(Op.U_MULK, 5, 5, value=H2)
            p.emit(Op.U_ADD, 5, 5, 2)
            p.emit(Op.U_LOAD_F32, 1, 5, value=A.COS)
            p.emit(Op.U_LOAD_F32, 2, 5, value=A.SIN)
            p.emit(Op.U_LOAD_F32, 3, 3, value=qkv_off)
            p.emit(Op.U_LOAD_F32, 4, 3, value=qkv_off + H2)
            p.emit(Op.MUL, 5, 3, 1)
            p.emit(Op.MUL, 6, 4, 2)
            p.emit(Op.SUB, 5, 5, 6)
            p.emit(Op.MUL, 7, 4, 1)
            p.emit(Op.MUL, 8, 3, 2)
            p.emit(Op.ADD, 7, 7, 8)
            p.emit(Op.U_STORE_F32, 5, 3, value=qkv_off)
            p.emit(Op.U_STORE_F32, 7, 3, value=qkv_off + H2)

        return self.prog(tag, b), heads * H2, 64, 16

    # append this token's k/v (at qkv_off + nh*64) to a [k|v][nkv][MAXSEQ][64] cache
    def kv_write(self, kv, nh, nkv, qkv_off, tag):
        A = self.A
        D = 64
        per = nkv * D

        def b(p):
            u_gid(p, 0, 9, 64)
            p.emit(Op.U_LOAD_F32, 1, 0, value=qkv_off + nh * D)
            p.emit(Op.U_SHRK, 2, 0, value=per.bit_length() - 1)
            p.emit(Op.U_MULK, 2, 2, value=nkv * MAXSEQ * D)
            p.emit(Op.U_SHRK, 3, 0, value=D.bit_length() - 1)
            p.emit(Op.U_ANDK, 3, 3, value=nkv - 1)
            p.emit(Op.U_MULK, 3, 3, value=MAXSEQ * D)
            p.emit(Op.U_ADD, 2, 2, 3)
            p.emit(Op.U_ANDK, 4, 0, value=D - 1)
            p.emit(Op.U_ADD, 2, 2, 4)
            p.emit(Op.U_CONST, 5, value=0)
            p.emit(Op.U_LOAD_F32, 0, 5, value=A.POS)
            p.emit(Op.R_TO_U, 6, 0)
            p.emit(Op.U_MULK, 6, 6, value=D)
            p.emit(Op.U_ADD, 2, 2, 6)
            p.emit(Op.U_STORE_F32, 1, 2, value=kv)

        return self.prog(tag, b), 2 * per, 64, 16

    # causal attention, one threadgroup per q head, lane t owns key position t (head dim 64).
    # MAXSEQ == 64, so the 64-wide local window never truncates anything.
    def attention(self, kv, nh, nkv, q_off, att_off, tag):
        A = self.A
        T = MAXSEQ
        D = 64
        group = nh // nkv
        vbase = kv + nkv * MAXSEQ * D

        def b(p):
            p.emit(Op.U_TGID, 0)
            p.emit(Op.U_LID, 1)
            p.emit(Op.LOAD_LID, 20)
            p.emit(Op.LOAD_CONST, 21, f32=0.0)
            p.emit(Op.LOAD_CONST, 22, f32=1.0)
            p.emit(Op.U_CONST, 2, value=0)
            p.emit(Op.U_LOAD_F32, 0, 2, value=A.POS)
            p.emit(Op.ADD_CONST, 23, 0, f32=1.0)  # n = pos + 1
            p.emit(Op.U_MULK, 3, 0, value=D)  # q base
            p.emit(Op.U_SHRK, 4, 0, value=group.bit_length() - 1)
            p.emit(Op.U_MULK, 4, 4, value=MAXSEQ * D)  # kv head base
            p.emit(Op.U_MULK, 5, 1, value=D)
            p.emit(Op.U_ADD, 5, 5, 4)  # k row of lane t
            p.emit(Op.LOAD_CONST, 10, f32=0.0)
            p.emit(Op.LOAD_CONST, 11, f32=float(D))
            p.emit(Op.FOR_BEGIN, 11)
            p.emit(Op.U_LOAD_F32, 1, 3, value=q_off)
            p.emit(Op.U_LOAD_F32, 2, 5, value=kv)
            p.emit(Op.FMA, 10, 1, 2, value=10)
            p.emit(Op.U_ADDK, 3, 3, value=1)
            p.emit(Op.U_ADDK, 5, 5, value=1)
            p.emit(Op.FOR_END)
            p.emit(Op.MUL_CONST, 10, 10, f32=1.0 / (D ** 0.5))
            # mask lanes t >= n
            p.emit(Op.CMP_LT, 12, 20, 23)
            p.emit(Op.SUB, 14, 22, 12)
            p.emit(Op.LOAD_CONST, 13, f32=-1.0e30)
            p.emit(Op.MUL, 10, 10, 12)
            p.emit(Op.FMA, 10, 14, 13, value=10)
            p.emit(Op.ADD_CONST, 24, 10, f32=0.0)
            tg_reduce(p, T, 24, 20, 21, 15, 16, 17, 18, "max")
            p.emit(Op.SUB, 10, 10, 24)
            p.emit(Op.EXP, 10, 10)
            p.emit(Op.ADD_CONST, 25, 10, f32=0.0)
            tg_reduce(p, T, 25, 20, 21, 15, 16, 17, 18, "sum")
            p.emit(Op.DIV, 10, 10, 25)
            p.emit(Op.TG_STORE_F32, 10, 20, c=2 * T)
            p.emit(Op.BARRIER_TG)
            # weighted V sum: lane owns column d = lane
            p.emit(Op.U_TGID, 0)
            p.emit(Op.U_LID, 1)
            p.emit(Op.U_SHRK, 4, 0, value=group.bit_length() - 1)
            p.emit(Op.U_MULK, 4, 4, value=MAXSEQ * D)
            p.emit(Op.U_ADD, 6, 4, 1)  # v row 0, column lane
            p.emit(Op.LOAD_CONST, 10, f32=0.0)
            p.emit(Op.LOAD_CONST, 26, f32=0.0)
            p.emit(Op.ADD_CONST, 27, 23, f32=0.0)
            p.emit(Op.FOR_BEGIN, 27)
            p.emit(Op.TG_LOAD_F32, 1, 26, c=2 * T)
            p.emit(Op.U_LOAD_F32, 2, 6, value=vbase)
            p.emit(Op.FMA, 10, 1, 2, value=10)
            p.emit(Op.U_ADDK, 6, 6, value=D)
            p.emit(Op.ADD, 26, 26, 22)
            p.emit(Op.FOR_END)
            p.emit(Op.U_MULK, 7, 0, value=D)
            p.emit(Op.U_ADD, 7, 7, 1)
            p.emit(Op.U_STORE_F32, 10, 7, value=att_off)

        return self.prog(tag, b), nh * T, T, 4 * 3 * T

    # domain = argmax(tier-1 logits) over 4 domains (first index wins ties)
    def argmax4(self):
        A = self.A

        def b(p):
            p.emit(Op.U_CONST, 0, value=0)
            for j in range(4):
                p.emit(Op.U_LOAD_F32, j, 0, value=A.RL1 + j)
            p.emit(Op.ADD_CONST, 6, 0, f32=0.0)  # best
            p.emit(Op.LOAD_CONST, 7, f32=0.0)  # idx
            for j in range(1, 4):
                p.emit(Op.CMP_LT, 8, 6, j)
                p.emit(Op.SUB, 9, j, 6)
                p.emit(Op.FMA, 6, 8, 9, value=6)
                p.emit(Op.LOAD_CONST, 10, f32=float(j))
                p.emit(Op.SUB, 10, 10, 7)
                p.emit(Op.FMA, 7, 8, 10, value=7)
            p.emit(Op.U_STORE_F32, 7, 0, value=A.DOM)

        return self.prog("argmax4", b), 1, 1, 16

    # top-2 of the six tier-2 logits of the chosen domain -> global expert ids + mixing weights
    def top2(self):
        A = self.A

        def b(p):
            p.emit(Op.U_CONST, 0, value=0)
            for j in range(6):
                p.emit(Op.U_LOAD_F32, j, 0, value=A.RL2 + j)
            p.emit(Op.LOAD_CONST, 20, f32=1.0)
            p.emit(Op.ADD_CONST, 6, 0, f32=0.0)  # b1
            p.emit(Op.LOAD_CONST, 7, f32=0.0)  # i1
            p.emit(Op.LOAD_CONST, 8, f32=-1.0e30)  # b2
            p.emit(Op.LOAD_CONST, 9, f32=0.0)  # i2
            for j in range(1, 6):
                p.emit(Op.LOAD_CONST, 10, f32=float(j))
                p.emit(Op.CMP_LT, 11, 6, j)  # c1: b1 < l
                p.emit(Op.CMP_LT, 12, 8, j)  # c2: b2 < l
                p.emit(Op.SUB, 13, 20, 11)
                p.emit(Op.MUL, 13, 13, 12)  # m2 = (1 - c1) * c2
                p.emit(Op.SUB, 14, 6, 8)
                p.emit(Op.SUB, 15, j, 8)
                p.emit(Op.FMA, 16, 11, 14, value=8)
                p.emit(Op.FMA, 16, 13, 15, value=16)  # new b2
                p.emit(Op.SUB, 17, 7, 9)
                p.emit(Op.SUB, 18, 10, 9)
                p.emit(Op.FMA, 19, 11, 17, value=9)
                p.emit(Op.FMA, 19, 13, 18, value=19)  # new i2
                p.emit(Op.SUB, 14, j, 6)
                p.emit(Op.FMA, 6, 11, 14, value=6)  # new b1
                p.emit(Op.SUB, 17, 10, 7)
                p.emit(Op.FMA, 7, 11, 17, value=7)  # new i1
                p.emit(Op.ADD_CONST, 8, 16, f32=0.0)
                p.emit(Op.ADD_CONST, 9, 19, f32=0.0)
            p.emit(Op.SUB, 14, 6, 8)
            p.emit(Op.SIGMOID, 14, 14)  # w1 = softmax weight of the best expert among the two
            p.emit(Op.SUB, 15, 20, 14)  # w2
            p.emit(Op.U_LOAD_F32, 16, 0, value=A.DOM)
            p.emit(Op.MUL_CONST, 16, 16, f32=6.0)
            p.emit(Op.ADD, 17, 16, 7)
            p.emit(Op.ADD, 18, 16, 9)
            p.emit(Op.U_STORE_F32, 17, 0, value=A.SEL[0])
            p.emit(Op.U_STORE_F32, 18, 0, value=A.SEL[1])
            p.emit(Op.U_STORE_F32, 14, 0, value=A.W[0])
            p.emit(Op.U_STORE_F32, 15, 0, value=A.W[1])

        return self.prog("top2", b), 1, 1, 16


# --------------------------------------------------------------------------- model
class Forge:
    def __init__(self, ctx, model_dir):
        self.ctx = ctx
        self.dir = Path(model_dir)
        self.cfg = cfg = json.loads((self.dir / "model_config.json").read_text())
        self.L = cfg["n_layers"]
        self.mha = set(cfg["mha_layer_indices"])
        self.hse = set(cfg["hse_ffn_layer_indices"])
        arg = [i for i in range(self.L) if i not in self.mha]
        self.A = A = Arena(cfg, sorted(self.mha), arg)
        self.pt = PtWeights(self.dir / "model_bf16.pt")
        self.sess = Session(ctx)
        self.K = Kernels(ctx, self.sess, A)
        self._load()
        self._compile()
        self.pos = 0

    def _buf(self, arr):
        return Buffer(self.ctx, np.ascontiguousarray(arr, dtype=np.float32).reshape(-1))

    def _norm(self, prefix):
        g = self.pt.get
        return self._buf(np.concatenate([g(prefix + ".weight"), g(prefix + ".bias")]))

    def _load(self):
        g, cfg = self.pt.get, self.cfg
        self.embed = self._buf(g("embed_tokens.weight"))
        if self.pt.key("lm_head.weight") == self.pt.key("embed_tokens.weight"):
            self.lm_head = self.embed
        else:
            self.lm_head = self._buf(g("lm_head.weight"))
        self.final_norm = self._norm("norm_final")
        self.layers = []
        for i in range(self.L):
            q = f"layers.{i}."
            s, f = q + "seq_mixer.", q + "ffn."
            w = {"seq_norm": self._norm(s + "norm"), "ffn_norm": self._norm(f + "norm")}
            if i in self.mha:
                w["qkv"] = self._buf(np.concatenate([g(s + "q_proj.weight"), g(s + "k_proj.weight"),
                                                     g(s + "v_proj.weight")]))
                w["o"] = self._buf(g(s + "o_proj.weight"))
            else:
                w["in_proj"] = self._buf(g(s + "in_proj.weight"))
                w["conv"] = self._buf(np.concatenate([g(s + "conv1d.weight").reshape(-1),
                                                      g(s + "conv1d.bias")]))
                w["x_proj"] = self._buf(g(s + "x_proj.weight"))
                w["dt_proj"] = self._buf(np.concatenate([g(s + "dt_proj.weight").reshape(-1),
                                                         g(s + "dt_proj.bias")]))
                a_state = np.repeat(-softplus(g(s + "nu")), 2)
                w["ssm"] = self._buf(np.concatenate([g(s + "D"), a_state]))
                w["out_proj"] = self._buf(g(s + "out_proj.weight"))
                w["lqkv"] = self._buf(np.concatenate([g(s + "local_q.weight"), g(s + "local_k.weight"),
                                                      g(s + "local_v.weight")]))
                w["local_o"] = self._buf(g(s + "local_o.weight"))
                w["gate"] = self._buf(np.concatenate([g(s + "gate_proj.weight").reshape(-1),
                                                      g(s + "gate_proj.bias")]))
            if i in self.hse:
                w["r1"] = self._buf(g(f + "tier1_router.weight"))
                w["r2"] = self._buf(np.concatenate([g(f + f"tier2_routers.{d}.weight") for d in range(4)]))
                ne = cfg["hse_n_domains"] * cfg["hse_n_experts_per_domain"]
                gu = np.zeros((ne, 2, FPAD, H), dtype=np.float32)
                dn = np.zeros((ne, H, FPAD), dtype=np.float32)
                for e in range(ne):
                    gu[e, 0, :210] = g(f + f"experts.{e}.gate.weight")
                    gu[e, 1, :210] = g(f + f"experts.{e}.up.weight")
                    dn[e, :, :210] = g(f + f"experts.{e}.down.weight")
                w["gate_up"] = self._buf(gu)
                w["down"] = self._buf(dn)
                del gu, dn
            else:
                w["gate_up"] = self._buf(np.concatenate([g(f + "gate.weight"), g(f + "up.weight")]))
                w["down"] = self._buf(g(f + "down.weight"))
            self.layers.append(w)
            print(f"loaded layer {i}", flush=True)
        # host-side rope tables (the VM has no sin/cos); cos/sin are bf16-rounded like the model's
        base = np.float32(cfg["rope_theta"])
        inv = (1.0 / (base ** (np.arange(0, 64, 2, dtype=np.float32) / np.float32(64)))).astype(np.float32)
        ang = np.outer(np.arange(MAXSEQ, dtype=np.float32), inv)
        A = self.A
        init = np.zeros(A.size, dtype=np.float32)
        init[A.COS:A.COS + MAXSEQ * 32] = bf16_round(np.cos(ang)).reshape(-1)
        init[A.SIN:A.SIN + MAXSEQ * 32] = bf16_round(np.sin(ang)).reshape(-1)
        self.arena = Buffer(self.ctx, init)
        self.logits = Buffer(self.ctx, size=A.V)

    def _compile(self):
        A, K, cfg = self.A, self.K, self.cfg
        eps = cfg["norm_eps"]
        plan = []

        def add(spec, weight, **bind):
            kname, threads, tg, tgb = spec
            plan.append((kname, threads, tg, tgb, weight, bind))

        add(K.embed(), self.embed)
        for i, w in enumerate(self.layers):
            add(K.dgn(A.X, A.HN, eps, "dgn_seq"), w["seq_norm"])
            if i in self.mha:
                add(K.matvec(1536, H, A.HN, A.QKV, "mha_qkv"), w["qkv"])
                add(K.rope(A.QKV, 20, "rope_mha"), None)
                add(K.kv_write(A.KV[i], 16, 4, A.QKV, f"kvw_mha{i}"), None)
                add(K.attention(A.KV[i], 16, 4, A.QKV, A.ATT, f"attn_mha{i}"), None)
                add(K.matvec(H, H, A.ATT, A.X, "mha_o", acc=True), w["o"])
            else:
                add(K.matvec(2 * H, H, A.HN, A.XZ, "in_proj"), w["in_proj"])
                add(K.conv(i), w["conv"])
                add(K.matvec(144, H, A.XC, A.XDBL, "x_proj"), w["x_proj"])
                add(K.matvec(H, 48, A.XDBL, A.DT, "dt_proj", T=16, bias=True, softplus_out=True),
                    w["dt_proj"])
                add(K.ssm(i), w["ssm"])
                add(K.matvec(H, H, A.Y, A.REC, "out_proj"), w["out_proj"])
                add(K.matvec(768, H, A.HN, A.LQKV, "lqkv"), w["lqkv"])
                add(K.rope(A.LQKV, 10, "rope_local"), None)
                add(K.kv_write(A.LKV[i], 8, 2, A.LQKV, f"kvw_loc{i}"), None)
                add(K.attention(A.LKV[i], 8, 2, A.LQKV, A.LATT, f"attn_loc{i}"), None)
                add(K.matvec(H, 512, A.LATT, A.LOC, "local_o"), w["local_o"])
                add(K.matvec(1, H, A.HN, A.GATE, "gate", bias=True), w["gate"])
                add(K.blend(), None)
            add(K.dgn(A.X, A.HN, eps, "dgn_ffn"), w["ffn_norm"])
            if i in self.hse:
                add(K.matvec(4, H, A.HN, A.RL1, "router1"), w["r1"])
                add(K.argmax4(), None)
                add(K.matvec(6, H, A.HN, A.RL2, "router2", expert_stride=6 * H, sel_off=A.DOM), w["r2"])
                add(K.top2(), None)
                for s in range(2):
                    add(K.matvec(FPAD, H, A.HN, A.EG, f"gate_up_e{s}", expert_stride=2 * FPAD * H,
                                 sel_off=A.SEL[s], dual=True), w["gate_up"])
                    add(K.matvec(H, FPAD, A.EG, A.X, f"down_e{s}", acc=True, expert_stride=H * FPAD,
                                 sel_off=A.SEL[s], scale_off=A.W[s]), w["down"])
            else:
                add(K.matvec(2560, H, A.HN, A.G, "dense_gu", dual=True), w["gate_up"])
                add(K.matvec(H, 2560, A.G, A.X, "dense_down", acc=True), w["down"])
        add(K.dgn(A.X, A.HN, eps, "dgn_final"), self.final_norm)
        # logits get their own exactly-vocab-sized buffer (mem); the hidden row is read through
        # the read-only lut domain of the arena.
        add(K.matvec(A.V, H, A.HN, 0, "lm_head", x_lut=True), self.lm_head,
            mem=self.logits, lut=self.arena)
        self.plan = plan

    def step(self, token):
        """Run one decode step; returns the logits for ``token`` at the current position."""
        A = self.A
        assert self.pos < MAXSEQ
        with self.sess.step(token, self.pos):
            with self.sess:
                self.sess.set_step_inputs(self.arena, A.TOK, A.POS, token, self.pos)
                for kname, threads, tg, tgb, weight, bind in self.plan:
                    bind = {"mem": self.arena, **bind}
                    self.sess.dispatch(kname, threads=threads, tg_size=tg, tg_bytes=tgb,
                                       input=weight, **bind)
            logits = self.logits.read(0, self.A.V)
        self.pos += 1
        return logits


# --------------------------------------------------------------------------- numpy reference
def reference(model_dir, tokens):
    """Independent float32 numpy forward (no micro-kernels); returns logits per position."""
    d = Path(model_dir)
    cfg = json.loads((d / "model_config.json").read_text())
    pt = PtWeights(d / "model_bf16.pt")
    g = pt.get
    L, eps = cfg["n_layers"], cfg["norm_eps"]
    mha, hse = set(cfg["mha_layer_indices"]), set(cfg["hse_ffn_layer_indices"])
    inv = (1.0 / (np.float32(cfg["rope_theta"]) ** (np.arange(0, 64, 2, dtype=np.float32) / np.float32(64)))
           ).astype(np.float32)

    def dgn(x, p):
        xg = x.reshape(NG, GS)
        r = 1.0 / np.sqrt((xg * xg).mean(1, keepdims=True) + eps)
        return (xg * r).reshape(-1) * g(p + ".weight") + g(p + ".bias")

    def silu(x):
        return x / (1.0 + np.exp(-x))

    def rope(a, cos, sin):  # a: [heads, 64]
        lo, hi = a[:, :32], a[:, 32:]
        return np.concatenate([lo * cos - hi * sin, hi * cos + lo * sin], -1)

    def attend(q, ks, vs, nkv):
        K, V = np.stack(ks), np.stack(vs)  # [n, nkv, 64]
        nh = q.shape[0]
        out = np.zeros((nh, 64), dtype=np.float32)
        for hd in range(nh):
            kv = hd // (nh // nkv)
            s = K[:, kv, :] @ q[hd] / 8.0
            p = np.exp(s - s.max())
            p /= p.sum()
            out[hd] = p @ V[:, kv, :]
        return out.reshape(-1)

    emb, lm = g("embed_tokens.weight"), g("lm_head.weight")
    conv_hist = {i: np.zeros((3, H), dtype=np.float32) for i in range(L)}
    ssm_h = {i: np.zeros((H, 48), dtype=np.float32) for i in range(L)}
    kc = {i: [] for i in range(L)}
    vc = {i: [] for i in range(L)}
    out = []
    for pos, t in enumerate(tokens):
        x = emb[t].copy()
        ang = pos * inv
        cos, sin = bf16_round(np.cos(ang)), bf16_round(np.sin(ang))
        for i in range(L):
            q = f"layers.{i}."
            s, f = q + "seq_mixer.", q + "ffn."
            h = dgn(x, s + "norm")
            if i in mha:
                qq = (g(s + "q_proj.weight") @ h).reshape(16, 64)
                kk = (g(s + "k_proj.weight") @ h).reshape(4, 64)
                vv = (g(s + "v_proj.weight") @ h).reshape(4, 64)
                qq, kk = rope(qq, cos, sin), rope(kk, cos, sin)
                kc[i].append(kk)
                vc[i].append(vv)
                x = x + g(s + "o_proj.weight") @ attend(qq, kc[i], vc[i], 4)
            else:
                xz = g(s + "in_proj.weight") @ h
                xi, z = xz[:H], xz[H:]
                win = np.concatenate([conv_hist[i], xi[None]], 0)  # [4, H]: x_{t-3..t}
                cw = g(s + "conv1d.weight").reshape(H, 4)
                xc = silu((cw * win.T).sum(1) + g(s + "conv1d.bias"))
                conv_hist[i] = win[1:]
                xdbl = g(s + "x_proj.weight") @ xc
                dt = softplus(g(s + "dt_proj.weight") @ xdbl[:48] + g(s + "dt_proj.bias"))
                B, C = xdbl[48:96], xdbl[96:144]
                A_ = np.repeat(-softplus(g(s + "nu")), 2)
                ssm_h[i] = np.exp(dt[:, None] * A_[None]) * ssm_h[i] + (dt * xc)[:, None] * B[None]
                y = ((ssm_h[i] * C[None]).sum(1) + xc * g(s + "D")) * silu(z)
                rec = g(s + "out_proj.weight") @ y
                qq = (g(s + "local_q.weight") @ h).reshape(8, 64)
                kk = (g(s + "local_k.weight") @ h).reshape(2, 64)
                vv = (g(s + "local_v.weight") @ h).reshape(2, 64)
                qq, kk = rope(qq, cos, sin), rope(kk, cos, sin)
                kc[i].append(kk)
                vc[i].append(vv)
                loc = g(s + "local_o.weight") @ attend(qq, kc[i][-65:], vc[i][-65:], 2)
                a = 1.0 / (1.0 + np.exp(-(g(s + "gate_proj.weight") @ h + g(s + "gate_proj.bias"))))
                x = x + a * loc + (1 - a) * rec
            h = dgn(x, f + "norm")
            if i in hse:
                dom = int(np.argmax(g(f + "tier1_router.weight") @ h))
                l2 = g(f + f"tier2_routers.{dom}.weight") @ h
                p2 = np.exp(l2 - l2.max())
                p2 /= p2.sum()
                top = np.argsort(-p2, kind="stable")[:2]
                w = p2[top] / (p2[top].sum() + 1e-9)
                acc = np.zeros(H, dtype=np.float32)
                for wk, e in zip(w, top):
                    ge = dom * 6 + int(e)
                    gg = g(f + f"experts.{ge}.gate.weight") @ h
                    uu = g(f + f"experts.{ge}.up.weight") @ h
                    acc += wk * (g(f + f"experts.{ge}.down.weight") @ (silu(gg) * uu))
                x = x + acc
            else:
                gg, uu = g(f + "gate.weight") @ h, g(f + "up.weight") @ h
                x = x + g(f + "down.weight") @ (silu(gg) * uu)
        out.append(lm @ dgn(x, "norm_final"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--tokens", default="1,100,200")
    ap.add_argument("--new", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    prompt = [int(t) for t in a.tokens.split(",")]
    with Context() as ctx:
        m = Forge(ctx, a.model_dir)
        seq, logits_all = list(prompt), []
        nxt = None
        for i in range(len(prompt) + a.new):
            tok = seq[i] if i < len(seq) else nxt
            if i >= len(seq):
                seq.append(tok)
            t0 = time.time()
            lg = m.step(tok)
            logits_all.append(lg)
            nxt = int(np.argmax(lg))
            print(f"pos {i} token {tok} -> next {nxt}  ({time.time() - t0:.2f}s)", flush=True)
        print("tokens:", seq + [nxt])
        if a.check:
            ref = reference(a.model_dir, seq)
            for i, (r, g) in enumerate(zip(ref, logits_all)):
                err = float(np.abs(r - g).max())
                print(f"pos {i}: max|dlogit|={err:.3e} ref_argmax={int(np.argmax(r))} vm_argmax={int(np.argmax(g))}")


if __name__ == "__main__":
    main()
