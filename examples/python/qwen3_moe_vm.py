"""Qwen3-MoE decode model written as a Python class on top of micro-kernels.

Every compute step is a hand-emitted micro-VM program (``VMProgram``) dispatched on
stateful Metal buffers through ``Session``.  One token = one session.  The model is
meant to be traced (``BYBY_FIR_PROBE``) and lifted to ``model.fir`` unchanged.

Layout
  * ``arena``  (domain ``mem``): activations, router state, rope tables, KV caches.
  * weights    (domain ``input``): one resident f32 ``Buffer`` per tensor (bf16 -> f32).
  * experts are packed per layer so the chosen expert is a data-dependent address.

Usage:  python qwen3_moe_vm.py MODEL_DIR [--tokens 9707,11,1246] [--new 3] [--check]
"""
from __future__ import annotations

import argparse
import json
import mmap
import struct
import time
from pathlib import Path

import numpy as np

from nimformers import Buffer, Context, Op, Session

MAXSEQ = 64  # KV-cache capacity (also the attention threadgroup width)


# --------------------------------------------------------------------------- weights
class Safetensors:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.meta = json.loads(self.f.read(n))
        self.base = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def get(self, name) -> np.ndarray:
        m = self.meta[name]
        assert m["dtype"] == "BF16", m["dtype"]
        a, b = m["data_offsets"]
        raw = np.frombuffer(self.mm, dtype=np.uint16, count=(b - a) // 2, offset=self.base + a)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])


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

    def __init__(self, cfg, layers):
        self.H = cfg["hidden_size"]
        off = 0

        def take(n):
            nonlocal off
            o = off
            off += (n + 63) // 64 * 64
            return o

        self.TOK = take(1)
        self.POS = take(1)
        self.SEL = take(1)
        self.RL = take(4)
        self.X = take(self.H)
        self.HN = take(self.H)
        self.NH = cfg["num_attention_heads"]
        self.NKV = cfg["num_key_value_heads"]
        self.D = cfg["head_dim"]
        self.QKV = take((self.NH + 2 * self.NKV) * self.D)
        self.ATT = take(self.NH * self.D)
        self.G = take(cfg["moe_intermediate_size"])
        self.V = cfg["vocab_size"]
        self.COS = take(MAXSEQ * (self.D // 2))
        self.SIN = take(MAXSEQ * (self.D // 2))
        self.KV = [take(2 * self.NKV * MAXSEQ * self.D) for _ in range(layers)]
        self.size = off


# --------------------------------------------------------------------------- kernels
class Kernels:
    def __init__(self, ctx, sess, cfg, A):
        self.ctx, self.s, self.cfg, self.A = ctx, sess, cfg, A
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
            p.emit(Op.U_MULK, 3, 3, value=A.H)
            p.emit(Op.U_ADD, 3, 3, 0)
            p.emit(Op.U_LOAD_IN, 1, 3, value=0)
            p.emit(Op.U_STORE_F32, 1, 0, value=A.X)

        return self.prog("embed", b), A.H, 256, 16

    # rmsnorm over `rows` rows of width n; weight is a [n] input vector
    def rmsnorm(self, rows, n, T, x_off, o_off, eps, tag):
        iters = n // T
        assert iters * T == n

        def b(p):
            p.emit(Op.U_TGID, 0)
            p.emit(Op.U_LID, 1)
            p.emit(Op.LOAD_LID, 20)
            p.emit(Op.LOAD_CONST, 21, f32=0.0)
            p.emit(Op.LOAD_CONST, 11, f32=float(iters))
            for pass_ in range(2):
                p.emit(Op.U_MULK, 2, 0, value=n)
                p.emit(Op.U_ADD, 2, 2, 1)
                if pass_ == 0:
                    p.emit(Op.LOAD_CONST, 10, f32=0.0)
                else:
                    p.emit(Op.U_ADDK, 3, 1, value=0)
                p.emit(Op.FOR_BEGIN, 11)
                p.emit(Op.U_LOAD_F32, 5, 2, value=x_off)
                if pass_ == 0:
                    p.emit(Op.FMA, 10, 5, 5, value=10)
                else:
                    p.emit(Op.U_LOAD_IN, 6, 3, value=0)
                    p.emit(Op.MUL, 5, 5, 12)
                    p.emit(Op.MUL, 5, 5, 6)
                    p.emit(Op.U_STORE_F32, 5, 2, value=o_off)
                    p.emit(Op.U_ADDK, 3, 3, value=T)
                p.emit(Op.U_ADDK, 2, 2, value=T)
                p.emit(Op.FOR_END)
                if pass_ == 0:
                    tg_reduce(p, T, 10, 20, 21, 13, 14, 15, 16, "sum")
                    p.emit(Op.MUL_CONST, 10, 10, f32=1.0 / n)
                    p.emit(Op.ADD_CONST, 10, 10, f32=eps)
                    p.emit(Op.RSQRT, 12, 10)
                    p.emit(Op.U_LID, 1)

        return self.prog(tag, b), rows * T, T, 4 * T

    # y = W x (thread group per output row); optional accumulate / expert select / swiglu pair
    def matvec(self, N, K, x_off, y_off, tag, *, acc=False, expert_stride=None, dual=False, T=256,
               x_lut=False):
        A = self.A
        iters = K // T
        assert iters * T == K

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
                p.emit(Op.U_LOAD_F32, 3, 4, value=A.SEL)
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
            if acc:
                p.emit(Op.U_LOAD_F32, 14, 0, value=y_off)
                p.emit(Op.ADD, 10, 10, 14)
            p.emit(Op.U_STORE_F32, 10, 0, value=y_off)
            p.emit(Op.IF_END)

        return self.prog(tag, b), N * T, T, 4 * T

    # in-place rotate-half rope over q and k heads at the current position
    def rope(self):
        A = self.A
        H2 = A.D // 2
        heads = A.NH + A.NKV

        def b(p):
            u_gid(p, 0, 9, 256)
            p.emit(Op.U_SHRK, 1, 0, value=H2.bit_length() - 1)
            p.emit(Op.U_MULK, 1, 1, value=A.D)
            p.emit(Op.U_ANDK, 2, 0, value=H2 - 1)
            p.emit(Op.U_ADD, 3, 1, 2)
            p.emit(Op.U_CONST, 4, value=0)
            p.emit(Op.U_LOAD_F32, 0, 4, value=A.POS)
            p.emit(Op.R_TO_U, 5, 0)
            p.emit(Op.U_MULK, 5, 5, value=H2)
            p.emit(Op.U_ADD, 5, 5, 2)
            p.emit(Op.U_LOAD_F32, 1, 5, value=A.COS)
            p.emit(Op.U_LOAD_F32, 2, 5, value=A.SIN)
            p.emit(Op.U_LOAD_F32, 3, 3, value=A.QKV)
            p.emit(Op.U_LOAD_F32, 4, 3, value=A.QKV + H2)
            p.emit(Op.MUL, 5, 3, 1)
            p.emit(Op.MUL, 6, 4, 2)
            p.emit(Op.SUB, 5, 5, 6)
            p.emit(Op.MUL, 7, 4, 1)
            p.emit(Op.MUL, 8, 3, 2)
            p.emit(Op.ADD, 7, 7, 8)
            p.emit(Op.U_STORE_F32, 5, 3, value=A.QKV)
            p.emit(Op.U_STORE_F32, 7, 3, value=A.QKV + H2)

        return self.prog("rope", b), heads * H2, 256, 16

    # append this token's k/v to the layer cache
    def kv_write(self, layer):
        A = self.A
        per = A.NKV * A.D  # 1024: size of k (or v) part

        def b(p):
            u_gid(p, 0, 9, 256)
            p.emit(Op.U_LOAD_F32, 1, 0, value=A.QKV + A.NH * A.D)
            p.emit(Op.U_SHRK, 2, 0, value=per.bit_length() - 1)
            p.emit(Op.U_MULK, 2, 2, value=A.NKV * MAXSEQ * A.D)
            p.emit(Op.U_SHRK, 3, 0, value=A.D.bit_length() - 1)
            p.emit(Op.U_ANDK, 3, 3, value=A.NKV - 1)
            p.emit(Op.U_MULK, 3, 3, value=MAXSEQ * A.D)
            p.emit(Op.U_ADD, 2, 2, 3)
            p.emit(Op.U_ANDK, 4, 0, value=A.D - 1)
            p.emit(Op.U_ADD, 2, 2, 4)
            p.emit(Op.U_CONST, 5, value=0)
            p.emit(Op.U_LOAD_F32, 0, 5, value=A.POS)
            p.emit(Op.R_TO_U, 6, 0)
            p.emit(Op.U_MULK, 6, 6, value=A.D)
            p.emit(Op.U_ADD, 2, 2, 6)
            p.emit(Op.U_STORE_F32, 1, 2, value=A.KV[layer])

        return self.prog(f"kvw{layer}", b), 2 * per, 256, 16

    # causal attention, one threadgroup per q head, lane t owns key position t
    def attention(self, layer):
        A = self.A
        T = MAXSEQ
        D = A.D
        group = A.NH // A.NKV
        kv = A.KV[layer]
        vbase = kv + A.NKV * MAXSEQ * D

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
            p.emit(Op.U_LOAD_F32, 1, 3, value=A.QKV)
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
            # weighted V sum: lane owns d = lane and lane + 64
            p.emit(Op.U_TGID, 0)
            p.emit(Op.U_LID, 1)
            p.emit(Op.U_SHRK, 4, 0, value=group.bit_length() - 1)
            p.emit(Op.U_MULK, 4, 4, value=MAXSEQ * D)
            p.emit(Op.U_ADD, 6, 4, 1)  # v row 0, column lane
            p.emit(Op.LOAD_CONST, 10, f32=0.0)
            p.emit(Op.LOAD_CONST, 11, f32=0.0)
            p.emit(Op.LOAD_CONST, 26, f32=0.0)
            p.emit(Op.ADD_CONST, 27, 23, f32=0.0)
            p.emit(Op.FOR_BEGIN, 27)
            p.emit(Op.TG_LOAD_F32, 1, 26, c=2 * T)
            p.emit(Op.U_LOAD_F32, 2, 6, value=vbase)
            p.emit(Op.U_LOAD_F32, 3, 6, value=vbase + T)
            p.emit(Op.FMA, 10, 1, 2, value=10)
            p.emit(Op.FMA, 11, 1, 3, value=11)
            p.emit(Op.U_ADDK, 6, 6, value=D)
            p.emit(Op.ADD, 26, 26, 22)
            p.emit(Op.FOR_END)
            p.emit(Op.U_MULK, 7, 0, value=D)
            p.emit(Op.U_ADD, 7, 7, 1)
            p.emit(Op.U_STORE_F32, 10, 7, value=A.ATT)
            p.emit(Op.U_STORE_F32, 11, 7, value=A.ATT + T)

        assert D == 2 * T
        return self.prog(f"attn{layer}", b), A.NH * T, T, 4 * 3 * T

    # sel = argmax(router logits) (3 experts)
    def argmax3(self):
        A = self.A

        def b(p):
            p.emit(Op.U_CONST, 0, value=0)
            p.emit(Op.U_LOAD_F32, 0, 0, value=A.RL)
            p.emit(Op.U_LOAD_F32, 1, 0, value=A.RL + 1)
            p.emit(Op.U_LOAD_F32, 2, 0, value=A.RL + 2)
            p.emit(Op.CMP_LT, 5, 0, 1)
            p.emit(Op.SUB, 3, 1, 0)
            p.emit(Op.FMA, 4, 5, 3, value=0)  # best of first two
            p.emit(Op.CMP_LT, 6, 4, 2)
            p.emit(Op.LOAD_CONST, 7, f32=2.0)
            p.emit(Op.SUB, 7, 7, 5)
            p.emit(Op.FMA, 8, 6, 7, value=5)
            p.emit(Op.U_STORE_F32, 8, 0, value=A.SEL)

        return self.prog("argmax3", b), 1, 1, 16


# --------------------------------------------------------------------------- model
class Qwen3MoE:
    def __init__(self, ctx, model_dir):
        self.ctx = ctx
        self.dir = Path(model_dir)
        self.cfg = cfg = json.loads((self.dir / "config.json").read_text())
        assert cfg["num_experts"] == 3 and cfg["num_experts_per_tok"] == 1
        self.L = cfg["num_hidden_layers"]
        self.A = A = Arena(cfg, self.L)
        self.st = Safetensors(self.dir / "model.safetensors")
        self.sess = Session(ctx)
        self.K = Kernels(ctx, self.sess, cfg, A)
        self._load()
        self._compile()
        self.pos = 0

    def _buf(self, arr):
        return Buffer(self.ctx, np.ascontiguousarray(arr, dtype=np.float32).reshape(-1))

    def _load(self):
        cfg, A, st = self.cfg, self.A, self.st
        pre = "model.layers.%d."
        self.embed = self._buf(st.get("model.embed_tokens.weight"))
        self.lm_head = self._buf(st.get("lm_head.weight"))
        self.final_norm = self._buf(st.get("model.norm.weight"))
        self.layers = []
        for i in range(self.L):
            q = pre % i
            w = {}
            for k in ("input_layernorm", "post_attention_layernorm"):
                w[k] = self._buf(st.get(q + k + ".weight"))
            for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
                w[k] = self._buf(st.get(q + "self_attn." + k + ".weight"))
            w["q_norm"] = self._buf(st.get(q + "self_attn.q_norm.weight"))
            w["k_norm"] = self._buf(st.get(q + "self_attn.k_norm.weight"))
            w["gate"] = self._buf(st.get(q + "mlp.gate.weight"))
            ne = cfg["num_experts"]
            gu = np.stack([np.stack([st.get(q + f"mlp.experts.{e}.gate_proj.weight"),
                                     st.get(q + f"mlp.experts.{e}.up_proj.weight")]) for e in range(ne)])
            w["gate_up"] = self._buf(gu)
            del gu
            dn = np.stack([st.get(q + f"mlp.experts.{e}.down_proj.weight") for e in range(ne)])
            w["down"] = self._buf(dn)
            del dn
            self.layers.append(w)
            print(f"loaded layer {i}", flush=True)
        # host-side rope tables (the VM has no sin/cos)
        D2 = A.D // 2
        inv = 1.0 / (cfg["rope_theta"] ** (np.arange(D2, dtype=np.float64) / D2))
        ang = np.arange(MAXSEQ, dtype=np.float64)[:, None] * inv[None, :]
        init = np.zeros(A.size, dtype=np.float32)
        init[A.COS:A.COS + MAXSEQ * D2] = np.cos(ang).astype(np.float32).reshape(-1)
        init[A.SIN:A.SIN + MAXSEQ * D2] = np.sin(ang).astype(np.float32).reshape(-1)
        self.arena = Buffer(self.ctx, init)
        self.logits = Buffer(self.ctx, size=A.V)

    def _compile(self):
        A, K, cfg = self.A, self.K, self.cfg
        eps = cfg["rms_norm_eps"]
        H, I = A.H, cfg["moe_intermediate_size"]
        qd, kd = A.NH * A.D, A.NKV * A.D
        plan = []

        def add(spec, weight, **bind):
            kname, threads, tg, tgb = spec
            plan.append((kname, threads, tg, tgb, weight, bind))

        add(K.embed(), self.embed)
        for i, w in enumerate(self.layers):
            add(K.rmsnorm(1, H, 256, A.X, A.HN, eps, "norm_h"), w["input_layernorm"])
            add(K.matvec(qd, H, A.HN, A.QKV, "q_proj"), w["q_proj"])
            add(K.matvec(kd, H, A.HN, A.QKV + qd, "k_proj"), w["k_proj"])
            add(K.matvec(kd, H, A.HN, A.QKV + qd + kd, "v_proj"), w["v_proj"])
            add(K.rmsnorm(A.NH, A.D, 128, A.QKV, A.QKV, eps, "q_norm"), w["q_norm"])
            add(K.rmsnorm(A.NKV, A.D, 128, A.QKV + qd, A.QKV + qd, eps, "k_norm"), w["k_norm"])
            add(K.rope(), None)
            add(K.kv_write(i), None)
            add(K.attention(i), None)
            add(K.matvec(H, qd, A.ATT, A.X, "o_proj", acc=True), w["o_proj"])
            add(K.rmsnorm(1, H, 256, A.X, A.HN, eps, "norm_h"), w["post_attention_layernorm"])
            add(K.matvec(3, H, A.HN, A.RL, "router"), w["gate"])
            add(K.argmax3(), None)
            add(K.matvec(I, H, A.HN, A.G, "gate_up", expert_stride=2 * I * H, dual=True), w["gate_up"])
            add(K.matvec(H, I, A.G, A.X, "down", acc=True, expert_stride=H * I), w["down"])
        add(K.rmsnorm(1, H, 256, A.X, A.HN, eps, "norm_h"), self.final_norm)
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
    cfg = json.loads((d / "config.json").read_text())
    st = Safetensors(d / "model.safetensors")
    NH, NKV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]

    def rms(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

    D2 = D // 2
    inv = 1.0 / (cfg["rope_theta"] ** (np.arange(D2) / D2))
    emb, lm, fn = st.get("model.embed_tokens.weight"), st.get("lm_head.weight"), st.get("model.norm.weight")
    kc = [[] for _ in range(cfg["num_hidden_layers"])]
    vc = [[] for _ in range(cfg["num_hidden_layers"])]
    out = []
    for pos, t in enumerate(tokens):
        x = emb[t].copy()
        cos, sin = np.cos(pos * inv).astype(np.float32), np.sin(pos * inv).astype(np.float32)
        for i in range(cfg["num_hidden_layers"]):
            q = f"model.layers.{i}."
            h = rms(x, st.get(q + "input_layernorm.weight"))
            qq = (st.get(q + "self_attn.q_proj.weight") @ h).reshape(NH, D)
            kk = (st.get(q + "self_attn.k_proj.weight") @ h).reshape(NKV, D)
            vv = (st.get(q + "self_attn.v_proj.weight") @ h).reshape(NKV, D)
            qq = rms(qq, st.get(q + "self_attn.q_norm.weight"))
            kk = rms(kk, st.get(q + "self_attn.k_norm.weight"))

            def rope(a):
                lo, hi = a[:, :D2], a[:, D2:]
                return np.concatenate([lo * cos - hi * sin, hi * cos + lo * sin], -1)

            qq, kk = rope(qq), rope(kk)
            kc[i].append(kk)
            vc[i].append(vv)
            K, V = np.stack(kc[i]), np.stack(vc[i])  # [n, NKV, D]
            att = np.zeros((NH, D), dtype=np.float32)
            for hd in range(NH):
                g = hd // (NH // NKV)
                s = K[:, g, :] @ qq[hd] / np.sqrt(D)
                p = np.exp(s - s.max())
                p /= p.sum()
                att[hd] = p @ V[:, g, :]
            x = x + st.get(q + "self_attn.o_proj.weight") @ att.reshape(-1)
            h = rms(x, st.get(q + "post_attention_layernorm.weight"))
            e = int(np.argmax(st.get(q + "mlp.gate.weight") @ h))
            g = st.get(q + f"mlp.experts.{e}.gate_proj.weight") @ h
            u = st.get(q + f"mlp.experts.{e}.up_proj.weight") @ h
            x = x + st.get(q + f"mlp.experts.{e}.down_proj.weight") @ (g / (1 + np.exp(-g)) * u)
        out.append(lm @ rms(x, fn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--tokens", default="9707,11,1246")
    ap.add_argument("--new", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    prompt = [int(t) for t in a.tokens.split(",")]
    with Context() as ctx:
        m = Qwen3MoE(ctx, a.model_dir)
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
