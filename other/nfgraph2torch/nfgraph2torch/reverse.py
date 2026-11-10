"""model.fir -> torch (direction 2: nimformers -> torch).

``to_torch("dir/model.fir", weights)`` returns a ``torch.nn.Module`` that computes what the FIR runner
computes, using nothing but elementary ATen operations:

* ``ext`` nodes (``fir.embedding``, ``fir.linear``, ``fir.linear_t``) become ``index_select`` and a
  ``torch.matmul``.
* ``micro`` nodes are micro-kernels, i.e. a small bytecode program run by one thread per output
  element. They are executed by a *vectorised interpreter*: every VM register is a tensor with one
  entry per thread, ``add``/``mul``/``exp``... are the matching ATen op, ``uloadf32`` is an index
  (gather), ``ustoref32`` is an ``index_put``, ``ifbegin`` is a mask (``where``) and ``forbegin`` is
  a Python loop around the same tensor ops. So any kernel the FIR VM can run, torch can run; the cost
  is speed, not coverage: a convolution is one tensor op per multiply-accumulate of the window.

Weights are ``nn.Parameter`` objects named like the checkpoint tensors. The activation arena is a
flat float tensor, addressed exactly like the FIR runner addresses it.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn

from formers.fir import vm

MASK = 0xFFFFFFFF


class _Executor:
    """Runs one assembled kernel for all its threads at once."""

    def __init__(self, program, device):
        self.p = program
        self.device = device

    def run(self, mem):
        p, dev = self.p, self.device
        n = p.threads
        ops, jumps = p.ops, p.jumps
        R = [torch.zeros(n, dtype=torch.float32, device=dev) for _ in range(32)]
        U = [torch.zeros(n, dtype=torch.int64, device=dev) for _ in range(32)]
        R[0] = torch.arange(n, dtype=torch.float32, device=dev)
        mask = None                      # None = every thread active
        stack = []                       # ("if", saved) | ("for", start, reg, iteration, saved)
        pc, last = 0, len(ops)

        def put(regs, i, value):
            regs[i] = value if mask is None else torch.where(mask, value, regs[i])

        while pc < last:
            op = ops[pc]
            m = op[0]
            pc += 1
            if m == "uloadf32":
                addr = (U[op[2]] + op[3]) & MASK
                if mask is None:
                    R[op[1]] = mem[addr]
                else:
                    R[op[1]] = torch.where(mask, mem[torch.where(mask, addr, 0)], R[op[1]])
            elif m == "ustoref32":
                addr = (U[op[2]] + op[3]) & MASK
                value = R[op[1]]
                if mask is not None:
                    addr, value = addr[mask], value[mask]
                mem.index_put_((addr,), value)
            elif m == "add":
                put(R, op[1], R[op[2]] + R[op[3]])
            elif m == "sub":
                put(R, op[1], R[op[2]] - R[op[3]])
            elif m == "mul":
                put(R, op[1], R[op[2]] * R[op[3]])
            elif m == "div":
                put(R, op[1], R[op[2]] / R[op[3]])
            elif m == "fma":
                put(R, op[1], R[op[2]] * R[op[3]] + R[op[4] & 31])
            elif m == "mulconst":
                put(R, op[1], R[op[2]] * op[3])
            elif m == "addconst":
                put(R, op[1], R[op[2]] + op[3])
            elif m == "loadconstf32":
                put(R, op[1], torch.full((n,), op[2], dtype=torch.float32, device=dev))
            elif m == "neg":
                put(R, op[1], -R[op[2]])
            elif m == "exp":
                put(R, op[1], torch.exp(R[op[2]]))
            elif m == "sigmoid":
                put(R, op[1], torch.sigmoid(R[op[2]]))
            elif m == "softplus":
                x = R[op[2]]
                put(R, op[1], torch.where(x < 80, torch.log1p(torch.exp(torch.clamp(x, max=80))), x))
            elif m == "rsqrt":
                put(R, op[1], torch.rsqrt(R[op[2]]))
            elif m == "cmplt":
                put(R, op[1], (R[op[2]] < R[op[3]]).float())
            elif m == "rtou":
                put(U, op[1], torch.clamp(R[op[2]], min=0).to(torch.int64) & MASK)
            elif m == "utor":
                put(R, op[1], U[op[2]].float())
            elif m == "uconst":
                put(U, op[1], torch.full((n,), op[2] & MASK, dtype=torch.int64, device=dev))
            elif m == "uaddk":
                put(U, op[1], (U[op[2]] + op[3]) & MASK)
            elif m == "umulk":
                put(U, op[1], (U[op[2]] * op[3]) & MASK)
            elif m == "uadd":
                put(U, op[1], (U[op[2]] + U[op[3]]) & MASK)
            elif m == "umul":
                put(U, op[1], (U[op[2]] * U[op[3]]) & MASK)
            elif m == "uandk":
                put(U, op[1], U[op[2]] & op[3])
            elif m == "ushrk":
                put(U, op[1], U[op[2]] >> op[3])
            elif m == "ushlk":
                put(U, op[1], (U[op[2]] << op[3]) & MASK)
            elif m == "ifbegin":
                cond = R[op[1]] != 0
                inner = cond if mask is None else (mask & cond)
                stack.append(("if", mask))
                if not bool(inner.any()):
                    pc = jumps[pc - 1]
                    mask = stack.pop()[1]
                else:
                    mask = inner
            elif m == "ifend":
                mask = stack.pop()[1]
            elif m == "forbegin":
                count = R[op[1]].to(torch.int64)
                live = count > 0 if mask is None else (mask & (count > 0))
                if not bool(live.any()):
                    pc = jumps[pc - 1]
                else:
                    stack.append(("for", pc, op[1], 0, mask))
                    mask = live
            elif m == "forend":
                _, start, reg, it, saved = stack.pop()
                it += 1
                count = R[reg].to(torch.int64)
                again = count > it if saved is None else (saved & (count > it))
                if bool(again.any()):
                    stack.append(("for", start, reg, it, saved))
                    mask = again
                    pc = start
                else:
                    mask = saved
            elif m == "nop":
                pass
            else:
                raise vm.FirVmError("%s: mnemonic %s is not interpretable" % (p.where, m))


class FirTorch(nn.Module):
    """A FIR graph (``runtime graph``) as a torch module: ``forward(token, position) -> logits``."""

    def __init__(self, source, config=None, weights=None):
        super().__init__()
        text, config = _read(source, config)
        weights = {k: _flat(v) for k, v in (weights or {}).items()}
        layers = vm.parse(text, config)["symbols"]["layers"]
        sizes = {}
        for b in vm.parse(text, config)["buffers"].values():
            if b["kind"] == "weight":
                for layer in range(layers if "{i}" in b["tensor"] else 1):
                    sizes[b["tensor"].replace("{i}", str(layer))] = b["elements"]
        # FirModel parses and assembles the kernels (addresses, jumps); its own copy of the weights
        # is a placeholder, the real ones are the Parameters below.
        ref = vm.FirModel(text, config=config, weights={t: [0.0] * n for t, n in sizes.items()})
        self.ref, self.layers = ref, ref.layers
        prog, bufs = ref.prog, ref.prog["buffers"]
        self.params = nn.ParameterDict()
        self._param_key = {}
        for name, b in bufs.items():
            if b["kind"] != "weight":
                continue
            for layer in range(self.layers if "{i}" in b["tensor"] else 1):
                tensor = b["tensor"].replace("{i}", str(layer))
                source_name = ref.resolve(weights, tensor) if weights else tensor
                value = torch.tensor(weights[source_name], dtype=torch.float32) if weights \
                    else torch.zeros(b["elements"])
                key = tensor.replace(".", "__")
                self.params[key] = nn.Parameter(value.reshape(b["dims"]), requires_grad=False)
                self._param_key[(name, layer)] = key
        self.arena = len(ref.mem)
        self.exec = {key: _Executor(program, "cpu") for key, program in ref.programs.items()}

    def _weight(self, name, layer):
        return self.params[self._param_key[(name, layer if (name, layer) in self._param_key else 0)]]

    def forward(self, token, position=None):
        ref, prog = self.ref, self.ref.prog
        bufs, roles = prog["buffers"], ref.roles
        mem = torch.zeros(self.arena)
        token = torch.as_tensor(token).reshape(-1)[0]
        position = torch.zeros((), dtype=torch.long) if position is None else torch.as_tensor(position).reshape(-1)[0]
        mem[ref.memory[roles["token"]]] = token.float()
        mem[ref.memory[roles["position"]]] = position.float()
        for (name, layer), key in self._param_key.items():       # weights that live in the arena
            if name in ref.memory:
                base = ref.binding(name, layer)
                mem[base:base + bufs[name]["elements"]] = self.params[key].reshape(-1)
        nodes = prog["nodes"]
        i = 0
        while i < len(nodes):
            if nodes[i]["per_layer"]:
                stop = i
                while stop < len(nodes) and nodes[stop]["per_layer"]:
                    stop += 1
                for layer in range(self.layers):
                    for node in nodes[i:stop]:
                        self._node(node, layer, mem, token, position)
                i = stop
            else:
                self._node(nodes[i], 0, mem, token, position)
                i += 1
        base = ref.memory[roles["logits"]]
        return mem[base:base + prog["symbols"]["vocab"]].clone()

    def _node(self, node, layer, mem, token, position):
        if node["kind"] == "micro":
            self.exec[(node["name"], layer)].run(mem)
            return
        ref, bufs = self.ref, self.ref.prog["buffers"]
        xin, wname = node["inputs"][:2]
        out = node["outputs"][0]
        n = bufs[out]["elements"]
        ob = ref.binding(out, layer)
        weight = self._weight(wname, layer)
        if node["op"] == "fir.embedding":
            index = token if bufs[xin]["role"] == "token" else position
            mem[ob:ob + n] = weight.index_select(0, index.reshape(1).long()).reshape(-1)
            return
        total = bufs[xin]["elements"]
        xb = ref.binding(xin, layer)
        k = weight.shape[0] if node["op"] == "fir.linear" else weight.shape[1]
        x = mem[xb:xb + total].reshape(total // k, k)
        if node["op"] == "fir.linear":        # weight is [in, out]
            y = torch.matmul(x, weight).reshape(-1)
        else:                                  # fir.linear_t: weight is [out, in]
            y = torch.matmul(x, weight.t()).reshape(-1)
        mem[ob:ob + n] = y


def _flat(v):
    return v.detach().reshape(-1).tolist() if hasattr(v, "detach") else list(v)


def _read(source, config):
    path = Path(source) if not (isinstance(source, str) and "\n" in source) else None
    if path is not None:
        text = path.read_text(encoding="utf-8")
        if config is None:
            m = re.search(r"^config\s+(\S+)", text, re.M)
            config = json.loads((path.parent / (m.group(1) if m else "config.json")).read_text())
        return text, config
    return source, config or {}


def to_torch(source, weights=None, config=None):
    """``model.fir`` (path or text) -> ``FirTorch`` module. ``weights``: checkpoint name -> tensor."""
    return FirTorch(source, config=config, weights=weights)
