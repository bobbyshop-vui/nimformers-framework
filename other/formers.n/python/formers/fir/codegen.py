"""Loop-based session_vm micro-kernel generation.

Kernels iterate with runtime loops (``forbegin``/``forend``) and integer offset
registers, so their size depends on tensor rank, never on element counts. One
dimension of the iteration space is spread across threads (one lane per
element of that dimension); the remaining dimensions are serial loops.

Operands are strided views over flat arena buffers (``views.View``), so reshape,
permute, slice, expand and grouped concatenations are read in place without
copies. Uniform constants become literals and affine constants (``arange``) are
computed from loop counters.

Register conventions: R0 holds the lane id on entry and is not used afterwards;
R1..R31 are allocated. U0..U31 are allocated. Loop counts live in dedicated R
registers that are never written inside the loop (the JIT re-reads the count at
every iteration, the CPU VM snapshots it at loop entry). Loop nesting is
limited to 8 by the CPU VM.
"""

import math

from . import views as V
from . import custom


class LoweringError(ValueError):
    pass


MAX_LOOP_DEPTH = 8
WRAPPED = ("uaddk", "umulk", "uconst")
NEG_LARGE = -3.0e38


def fnum(x):
    """Exact decimal literal for a VM float constant."""
    x = float(x)
    if not math.isfinite(x):
        raise LoweringError("non-finite VM constant; use a finite masking sentinel")
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return repr(x)


class Mem:
    """A strided arena operand: ``port`` names the kernel address (in0, out1, ...)."""
    def __init__(self, port, view):
        self.port, self.view = port, view


class Const:
    def __init__(self, value):
        self.value = float(value)


class Affine:
    """value = offset + sum(coef[d] * index[d]) over the operand's logical dims."""
    def __init__(self, offset, coefs, shape):
        self.offset, self.coefs, self.shape = float(offset), tuple(coefs), tuple(shape)


class Acc:
    """An operand access over an iteration space split into outer and inner dims."""
    def __init__(self, kind, port, imm, outer, inner=(), offset=0.0):
        self.kind, self.port, self.imm = kind, port, imm
        self.outer, self.inner = [tuple(x) for x in outer], [tuple(x) for x in inner]
        self.offset = offset
        self.dynamic = None


def address(port, imm):
    return "@" + port + ("+" + str(imm) if imm else "")


def dims_for(operand, space):
    """Per-space-dim (stride, group) for an operand broadcast to ``space``."""
    if isinstance(operand, Mem):
        view = V.broadcast_to(operand.view, space)
        if view is None and operand.view.size == 1:
            # A single element broadcasts to any space regardless of its rank.
            view = V.View(operand.view.offset, tuple((n, 0, 1) for n in space))
        if view is None:
            raise LoweringError("operand view %s does not broadcast to %s" % (operand.view.shape, space))
        return view.offset, [(s, g) for _, s, g in view.dims]
    if isinstance(operand, Affine):
        lead = len(space) - len(operand.shape)
        out = []
        for i, size in enumerate(space):
            if i < lead:
                out.append((0, 1))
            else:
                d = operand.shape[i - lead]
                out.append((int(operand.coefs[i - lead]) if d == size and d != 1 else 0, 1))
        return 0, out
    return 0, [(0, 1)] * len(space)


def access(operand, space, split=None):
    """Bind ``operand`` to a space; ``split`` lists the inner (reduction) dims."""
    if isinstance(operand, Const):
        return None
    imm, dims = dims_for(operand, space)
    inner_ids = list(split or ())
    outer = [d for i, d in enumerate(dims) if i not in inner_ids]
    inner = [dims[i] for i in inner_ids]
    if isinstance(operand, Mem):
        return Acc("mem", operand.port, imm, outer, inner)
    return Acc("affine", None, 0, outer, inner, offset=operand.offset)


def plan(sizes, accs, part):
    """Split grouped dims, drop unit dims and coalesce; mutates acc.<part>."""
    sizes = list(sizes)
    accs = [a for a in accs if a is not None]
    rows = [list(getattr(a, part)) for a in accs]
    d = 0
    while d < len(sizes):
        groups = {r[d][1] for r in rows if r[d][1] != 1 and r[d][0] != 0}
        if len(groups) > 1:
            raise LoweringError("mixed grouped views in one dimension; materialize one operand")
        if groups:
            g = groups.pop()
            if sizes[d] % g:
                raise LoweringError("grouped view does not divide its dimension")
            sizes[d:d + 1] = [sizes[d] // g, g]
            for r in rows:
                s, gg = r[d]
                r[d:d + 1] = [(s, 1), (0, 1)] if gg == g and s != 0 else [(s * g, 1), (s, 1)]
            d += 2
        else:
            for r in rows:
                if r[d][0] == 0:
                    r[d] = (0, 1)
            d += 1
    keep = [i for i, n in enumerate(sizes) if n != 1]
    sizes = [sizes[i] for i in keep]
    rows = [[r[i][0] for i in keep] for r in rows]
    i = 0
    while i + 1 < len(sizes):
        if all(r[i] == r[i + 1] * sizes[i + 1] for r in rows):
            sizes[i:i + 2] = [sizes[i] * sizes[i + 1]]
            for r in rows:
                r[i:i + 2] = [r[i + 1]]
        else:
            i += 1
    for a, r in zip(accs, rows):
        setattr(a, part, r)
    return sizes


class Kernel:
    """Text builder with register allocation and structured loops."""

    def __init__(self):
        self.lines = []
        self.depth = 0
        self.loops = 0
        self.rfree = list(range(1, 32))
        self.ufree = list(range(0, 32))
        self._zero = None
        self._lane = None

    def emit(self, *args):
        if args[0] in WRAPPED:
            # Integer immediates are uint32; a negative stride is its two's complement
            # (U registers and addresses wrap modulo 2**32 in every VM backend).
            args = args[:-1] + ((args[-1] + (1 << 32) if isinstance(args[-1], int) and args[-1] < 0 else args[-1]),)
        self.lines.append("  " * (self.depth + 1) + " ".join(str(a) for a in args))

    def r(self):
        if not self.rfree:
            raise LoweringError("kernel exceeds 31 float registers; split the expression")
        return self.rfree.pop(0)

    def u(self):
        if not self.ufree:
            raise LoweringError("kernel exceeds 32 integer registers; reduce operand count or rank")
        return self.ufree.pop(0)

    def free_r(self, *rs):
        for r in rs:
            if r is not None and r not in self.rfree:
                self.rfree.append(r)
        self.rfree.sort()

    def free_u(self, *us):
        for u in us:
            if u is not None and u not in self.ufree and u != self._zero and u != self._lane:
                self.ufree.append(u)
        self.ufree.sort()

    def const(self, x, reg=None):
        reg = self.r() if reg is None else reg
        self.emit("loadconstf32", reg, fnum(x) if not isinstance(x, str) else x)
        return reg

    def zero_u(self):
        if self._zero is None:
            self._zero = self.u()
            self.emit("uconst", self._zero, 0)
        return self._zero

    def lane(self):
        if self._lane is None:
            self._lane = self.u()
            self.emit("rtou", self._lane, 0)
        return self._lane

    def loop(self, count, body):
        """Run ``body()`` count times; ``count`` is an int or an R register."""
        own = not isinstance(count, tuple)
        if own:
            if count <= 0:
                return
            reg = self.const(count)
        else:
            reg = count[0]
        self.loops += 1
        if self.loops > MAX_LOOP_DEPTH:
            raise LoweringError("loop nesting exceeds the VM limit of 8")
        self.emit("forbegin", reg)
        self.depth += 1
        body()
        self.depth -= 1
        self.emit("forend")
        self.loops -= 1
        if own:
            self.free_r(reg)

    def cond(self, reg, body):
        self.emit("ifbegin", reg)
        self.depth += 1
        body()
        self.depth -= 1
        self.emit("ifend")

    def copy(self, dst, src):
        self.emit("mulconst", dst, src, 1)

    # ---- loads and stores -------------------------------------------------
    def load(self, acc, regs, reg=None):
        reg = self.r() if reg is None else reg
        u = regs.get(id(acc))
        if acc.kind == "affine":
            if u is None:
                self.const(acc.offset, reg)
            else:
                self.emit("utor", reg, u)
                if acc.offset:
                    self.emit("addconst", reg, reg, fnum(acc.offset))
            return reg
        self.emit("uloadf32", reg, self.zero_u() if u is None else u, address(acc.port, acc.imm))
        return reg

    def store(self, acc, regs, reg):
        u = regs.get(id(acc))
        self.emit("ustoref32", reg, self.zero_u() if u is None else u, address(acc.port, acc.imm))

    # ---- iteration --------------------------------------------------------
    def lanes(self, sizes, accs, part="outer"):
        """Spread the largest dim over threads; return (threads, regs, loop dims)."""
        accs = [a for a in accs if a is not None]
        regs = {}
        for a in accs:
            if a.dynamic is not None:
                regs[id(a)] = a.dynamic
        if not sizes:
            return 1, regs, []
        lane_dim = max(range(len(sizes)), key=lambda i: (sizes[i], -i))
        lane = self.lane()
        shared = {}
        for a in accs:
            s = getattr(a, part)[lane_dim]
            if s == 0:
                continue
            if s == 1 and a.dynamic is None:
                regs[id(a)] = lane
                continue
            key = (s, a.dynamic)
            if key not in shared:
                u = self.u()
                self.emit("umulk", u, lane, s)
                if a.dynamic is not None:
                    self.emit("uadd", u, u, a.dynamic)
                shared[key] = u
            regs[id(a)] = shared[key]
        rest = [i for i in range(len(sizes)) if i != lane_dim]
        return sizes[lane_dim], regs, rest

    def nest(self, sizes, dims, accs, regs, body, part="outer"):
        """Nested loops over ``dims`` (indexes into sizes and acc.<part>)."""
        accs = [a for a in accs if a is not None]
        if not dims:
            body(regs)
            return
        d = dims[0]
        local = dict(regs)
        mine, shared = [], {}
        for a in accs:
            s = getattr(a, part)[d]
            if s == 0:
                continue
            base = regs.get(id(a))
            key = (base, s)
            if key not in shared:
                u = self.u()
                if base is None:
                    self.emit("uconst", u, 0)
                else:
                    self.emit("uaddk", u, base, 0)
                shared[key] = u
                mine.append((u, s))
            local[id(a)] = shared[key]

        def step():
            self.nest(sizes, dims[1:], accs, local, body, part)
            for u, s in mine:
                self.emit("uaddk", u, u, s)
        self.loop(sizes[d], step)
        self.free_u(*(u for u, _ in mine))

    # ---- expressions ------------------------------------------------------
    def tree(self, tree, leaf):
        """Compile a fused expression tree; ``leaf(kind, port)`` returns a register."""
        kind = tree[0]
        if kind in ("input", "state", "updated"):
            return leaf(kind, tree[1] if len(tree) > 1 else None)
        if kind == "const":
            return self.const(tree[1])
        if kind == "pow":
            base, exponent = tree[1]
            if exponent[0] != "const" or exponent[1] != int(exponent[1]) or abs(exponent[1]) > 32:
                raise LoweringError("pow requires a constant integer exponent in [-32,32]; express "
                                    "positive-base real powers as exp(exponent*log(base))")
            n = int(exponent[1])
            if n == 0:
                return self.const(1)
            r = self.tree(base, leaf)
            s = self.r()
            self.copy(s, r)
            for _ in range(abs(n) - 1):
                self.emit("mul", s, s, r)
            self.free_r(r)
            if n < 0:
                one = self.const(1)
                self.emit("div", s, one, s)
                self.free_r(one)
            return s
        args = [self.tree(c, leaf) for c in tree[1]]
        return self.op(kind, args)

    def op(self, kind, args):
        r = args[0]
        if kind in ("add", "sub", "mul", "div"):
            self.emit(kind, r, r, args[1])
        elif kind in ("maximum", "minimum"):
            c = self.r()
            a, b = args
            self.emit("cmplt", c, a, b) if kind == "maximum" else self.emit("cmplt", c, b, a)
            self.cond(c, lambda: self.copy(r, b))
            self.free_r(c)
        elif kind in ("lt", "gt", "le", "ge"):
            a, b = args
            if kind in ("gt", "le"):
                a, b = b, a
            c = self.r()
            self.emit("cmplt", c, a, b)
            if kind in ("le", "ge"):
                self.emit("neg", c, c)
                self.emit("addconst", c, c, 1)
            self.copy(r, c)
            self.free_r(c)
        elif kind in ("eq", "ne"):
            a, b = args
            c, d = self.r(), self.r()
            self.emit("cmplt", c, a, b)
            self.emit("cmplt", d, b, a)
            self.emit("add", c, c, d)
            if kind == "eq":
                self.emit("neg", c, c)
                self.emit("addconst", c, c, 1)
            self.copy(r, c)
            self.free_r(c, d)
        elif kind == "where":
            c, a, b = args
            out = self.r()
            self.copy(out, b)
            self.cond(c, lambda: self.copy(out, a))
            self.free_r(*args)
            return out
        elif kind in ("exp", "neg", "sigmoid", "rsqrt", "softplus"):
            self.emit(kind, r, r)
        elif kind == "sqrt":
            s = self.const(1)
            self.emit("rsqrt", r, r)
            self.emit("div", r, s, r)
            self.free_r(s)
        elif kind == "relu":
            z, c = self.const(0), self.r()
            self.emit("cmplt", c, r, z)
            self.cond(c, lambda: self.copy(r, z))
            self.free_r(z, c)
        elif kind == "abs":
            z, c = self.const(0), self.r()
            self.emit("cmplt", c, r, z)
            self.cond(c, lambda: self.emit("neg", r, r))
            self.free_r(z, c)
        elif kind == "silu":
            s = self.r()
            self.emit("sigmoid", s, r)
            self.emit("mul", r, r, s)
            self.free_r(s)
        elif kind == "gelu_quick":
            s = self.r()
            self.emit("mulconst", s, r, 1.702)
            self.emit("sigmoid", s, s)
            self.emit("mul", r, r, s)
            self.free_r(s)
        elif kind == "tanh":
            self.emit("mulconst", r, r, 2)
            self.emit("sigmoid", r, r)
            self.emit("mulconst", r, r, 2)
            self.emit("addconst", r, r, -1)
        elif kind == "gelu_new":
            s = self.r()
            self.emit("mul", s, r, r)
            self.emit("mul", s, s, r)
            self.emit("mulconst", s, s, 0.044715)
            self.emit("add", s, s, r)
            self.emit("mulconst", s, s, "2*sqrt(2/pi)")
            self.emit("sigmoid", s, s)
            self.emit("mul", r, r, s)
            self.free_r(s)
        elif kind == "gelu":
            self._gelu_erf(r)
        elif kind == "log":
            self._log(r)
        elif kind in ("sin", "cos"):
            s, c = self.sincos(r)
            self.copy(r, s if kind == "sin" else c)
            self.free_r(s, c)
        elif kind in custom.EXT_TREE_OPS:
            r = custom.EXT_TREE_OPS[kind](self, args)
        else:
            raise LoweringError("missing VM composition for " + kind)
        self.free_r(*args[1:])
        return r

    def _gelu_erf(self, r):
        # x * Phi(x); Phi via Abramowitz-Stegun 7.1.26 (|error| <= 1.5e-7).
        z, t, p, c = self.r(), self.r(), self.r(), self.r()
        self.copy(z, r)
        zero = self.const(0)
        self.emit("cmplt", c, z, zero)
        self.cond(c, lambda: self.emit("neg", z, z))
        self.emit("mulconst", z, z, repr(1 / math.sqrt(2)))
        self.emit("mulconst", t, z, 0.3275911)
        self.emit("addconst", t, t, 1)
        one = self.const(1)
        self.emit("div", t, one, t)
        self.const(1.061405429, p)
        for a in (-1.453152027, 1.421413741, -0.284496736, 0.254829592):
            self.emit("mul", p, p, t)
            self.emit("addconst", p, p, a)
        self.emit("mul", p, p, t)
        self.emit("mul", z, z, z)
        self.emit("neg", z, z)
        self.emit("exp", z, z)
        self.emit("mul", p, p, z)
        self.emit("mulconst", p, p, 0.5)
        # p = 0.5*erfc(|x|/sqrt2); Phi = 1 - p for x >= 0, p for x < 0.
        self.emit("neg", t, p)
        self.emit("addconst", t, t, 1)
        self.cond(c, lambda: self.copy(t, p))
        self.emit("mul", r, r, t)
        self.free_r(z, t, p, c, zero, one)

    def _log(self, r):
        # Eight square roots bring positive finite f32 into a small interval.
        s, one = self.r(), self.const(1)
        for _ in range(8):
            self.emit("rsqrt", s, r)
            self.emit("div", r, one, s)
        self.emit("addconst", s, r, 1)
        self.emit("addconst", r, r, -1)
        self.emit("div", r, r, s)
        z2, power, acc = self.r(), self.r(), self.r()
        self.emit("mul", z2, r, r)
        self.copy(power, r)
        self.copy(acc, r)
        for j in range(1, 12):
            self.emit("mul", power, power, z2)
            self.emit("mulconst", s, power, repr(1 / (2 * j + 1)))
            self.emit("add", acc, acc, s)
        self.emit("mulconst", r, acc, 512)
        self.free_r(s, one, z2, power, acc)

    def sincos(self, angle):
        """Return new registers (sin, cos) of ``angle`` (preserved)."""
        x, sign, z, c = self.r(), self.const(1), self.r(), self.r()
        self.copy(x, angle)
        zero = self.const(0)
        self.emit("cmplt", c, x, zero)

        def flip():
            self.emit("neg", x, x)
            self.const(-1, sign)
        self.cond(c, flip)
        # Reduce |angle| to [-pi, pi]: x - 2*pi*floor(x/(2*pi) + 1/2).
        u = self.u()
        self.emit("mulconst", z, x, repr(1 / (2 * math.pi)))
        self.emit("addconst", z, z, 0.5)
        self.emit("rtou", u, z)
        self.emit("utor", z, u)
        self.free_u(u)
        self.emit("mulconst", z, z, repr(2 * math.pi))
        self.emit("sub", x, x, z)
        z2, term_s, term_c, out_s, out_c = self.r(), self.r(), self.r(), self.r(), self.r()
        self.emit("mul", z2, x, x)
        self.copy(term_s, x)
        self.copy(out_s, x)
        self.const(1, term_c)
        self.const(1, out_c)
        for j in range(1, 12):
            self.emit("mul", term_s, term_s, z2)
            self.emit("mulconst", term_s, term_s, repr(-1 / ((2 * j) * (2 * j + 1))))
            self.emit("add", out_s, out_s, term_s)
            self.emit("mul", term_c, term_c, z2)
            self.emit("mulconst", term_c, term_c, repr(-1 / ((2 * j - 1) * (2 * j))))
            self.emit("add", out_c, out_c, term_c)
        self.emit("mul", out_s, out_s, sign)
        self.free_r(x, sign, z, c, zero, z2, term_s, term_c)
        return out_s, out_c

    def text(self, name, ins, outs, threads, size=None):
        head = f"kernel {name} in={ins} out={outs}"
        if size:
            head += " size=" + ",".join(size)
        head += f" threads={threads} tg=1 tgbytes=16"
        return head + "\n" + "\n".join(self.lines) + "\nendkernel\n"


# ---- generators -------------------------------------------------------------
# Each generator receives the kernel builder and operands and returns the
# thread count. Operands for node inputs are Mem/Const/Affine; outputs are Mem.


def gen_pointwise(k, tree, operands, outputs, space):
    """outputs[i] = tree(operands) over ``space``; ``outputs`` are Mem views."""
    accs = [access(op, space) for op in operands]
    outs = [access(o, space) for o in outputs]
    sizes = plan(space, accs + outs, "outer")
    threads, regs, rest = k.lanes(sizes, accs + outs)

    def body(regs):
        def leaf(kind, port):
            op = operands[port]
            if isinstance(op, Const):
                return k.const(op.value)
            return k.load(accs[port], regs)
        r = k.tree(tree, leaf)
        for o in outs:
            k.store(o, regs, r)
        k.free_r(r)
    k.nest(sizes, rest, accs + outs, regs, body)
    return threads


def _reduce_split(space, dims):
    return [i for i in range(len(space)) if i in dims]


def gen_reduce(k, kind, tree, operands, out, in_shape, dims, keepdim):
    """out = reduce_kind over ``dims`` of tree(operands) evaluated on ``in_shape``."""
    inner_ids = _reduce_split(in_shape, dims)
    kept = [d for i, d in enumerate(in_shape) if i not in inner_ids]
    red = [in_shape[i] for i in inner_ids]
    accs = [access(op, in_shape, inner_ids) for op in operands]
    oview = V.reshape(out.view, tuple(kept)) if out.view.shape != tuple(kept) else out.view
    if oview is None:
        raise LoweringError("reduction output view is not strided")
    oacc = access(Mem(out.port, oview), tuple(kept))
    sizes = plan(kept, accs + [oacc], "outer")
    inner = plan(red, accs, "inner")
    threads, regs, rest = k.lanes(sizes, accs + [oacc])
    count = math.prod(red)

    def outer(regs):
        acc = k.const(NEG_LARGE if kind == "max" else 0)

        def body(iregs):
            def leaf(_, port):
                op = operands[port]
                return k.const(op.value) if isinstance(op, Const) else k.load(accs[port], iregs)
            x = k.tree(tree, leaf)
            if kind == "max":
                c = k.r()
                k.emit("cmplt", c, acc, x)
                k.cond(c, lambda: k.copy(acc, x))
                k.free_r(c)
            else:
                k.emit("add", acc, acc, x)
            k.free_r(x)
        k.nest(inner, list(range(len(inner))), accs, regs, body, "inner")
        if kind == "mean":
            k.emit("mulconst", acc, acc, repr(1 / count))
        k.store(oacc, regs, acc)
        k.free_r(acc)
    k.nest(sizes, rest, accs + [oacc], regs, outer)
    return threads


def gen_softmax(k, x, out, shape, dim):
    accs = [access(x, shape, [dim]), access(out, shape, [dim])]
    outer = [d for i, d in enumerate(shape) if i != dim]
    sizes = plan(outer, accs, "outer")
    inner = plan([shape[dim]], accs, "inner")
    threads, regs, rest = k.lanes(sizes, accs)
    xa, oa = accs

    def row(regs):
        m, total = k.const(NEG_LARGE), k.const(0)

        def pass_max(r):
            v = k.load(xa, r)
            c = k.r()
            k.emit("cmplt", c, m, v)
            k.cond(c, lambda: k.copy(m, v))
            k.free_r(v, c)

        def pass_exp(r):
            v = k.load(xa, r)
            k.emit("sub", v, v, m)
            k.emit("exp", v, v)
            k.store(oa, r, v)
            k.emit("add", total, total, v)
            k.free_r(v)

        def pass_norm(r):
            v = k.load(oa, r)
            k.emit("div", v, v, total)
            k.store(oa, r, v)
            k.free_r(v)
        for fn in (pass_max, pass_exp, pass_norm):
            k.nest(inner, list(range(len(inner))), accs, regs, fn, "inner")
        k.free_r(m, total)
    k.nest(sizes, rest, accs, regs, row)
    return threads


def gen_norm(k, kind, tree, operands, weight, bias, out, side, shape, width, eps):
    """LayerNorm/RMSNorm over trailing ``width`` elements of tree(operands)."""
    tail, prod = 0, 1
    while prod < width:
        tail += 1
        prod *= shape[-tail]
    if prod != width:
        raise LoweringError("norm width must cover trailing dimensions")
    inner_ids = list(range(len(shape) - tail, len(shape)))
    tmp = side if side is not None else out
    accs = [access(op, shape, inner_ids) for op in operands]
    reread = side is None and tree == ("input", 0) and isinstance(operands[0], Mem)
    wa = access(weight, shape, inner_ids) if weight is not None else None
    ba = access(bias, shape, inner_ids) if bias is not None else None
    oa = access(out, shape, inner_ids)
    ta = access(tmp, shape, inner_ids) if side is not None else oa
    if reread:
        ta = accs[0]
    every = accs + [wa, ba, oa] + ([ta] if side is not None else [])
    outer = shape[:len(shape) - tail]
    sizes = plan(outer, every, "outer")
    inner = plan(shape[len(shape) - tail:], every, "inner")
    threads, regs, rest = k.lanes(sizes, every)
    dims = list(range(len(inner)))

    def row(regs):
        total = k.const(0)

        def first(r):
            def leaf(_, port):
                op = operands[port]
                return k.const(op.value) if isinstance(op, Const) else k.load(accs[port], r)
            x = k.tree(tree, leaf)
            if not reread:
                k.store(ta, r, x)
            if kind == "rmsnorm":
                k.emit("mul", x, x, x)
            k.emit("add", total, total, x)
            k.free_r(x)
        k.nest(inner, dims, every, regs, first, "inner")
        k.emit("mulconst", total, total, "1/" + str(width))
        mean = None
        if kind == "layernorm":
            mean, var = total, k.const(0)

            def second(r):
                x = k.load(ta, r)
                k.emit("sub", x, x, mean)
                k.emit("mul", x, x, x)
                k.emit("add", var, var, x)
                k.free_r(x)
            k.nest(inner, dims, every, regs, second, "inner")
            k.emit("mulconst", var, var, "1/" + str(width))
        else:
            var = total
        k.emit("addconst", var, var, eps)
        k.emit("rsqrt", var, var)

        def third(r):
            x = k.load(ta, r)
            if mean is not None:
                k.emit("sub", x, x, mean)
            k.emit("mul", x, x, var)
            if wa is not None:
                w = k.load(wa, r)
                k.emit("mul", x, x, w)
                k.free_r(w)
            if ba is not None:
                b = k.load(ba, r)
                k.emit("add", x, x, b)
                k.free_r(b)
            k.store(oa, r, x)
            k.free_r(x)
        k.nest(inner, dims, every, regs, third, "inner")
        k.free_r(total, var)
    k.nest(sizes, rest, every, regs, row)
    return threads


def gen_matmul(k, a, b, out, ashape, bshape, oshape, a_is_weight_t=False):
    """Batched matmul with broadcasting: out[..., m, n] = sum_k a[..., m, k] b[..., k, n]."""
    aa = (1,) + ashape if len(ashape) == 1 else ashape
    bb = bshape + (1,) if len(bshape) == 1 else bshape
    batch = V.broadcast_shape(aa[:-2], bb[:-2])
    del a_is_weight_t
    m, n, kdim = aa[-2], bb[-1], aa[-1]
    space = batch + (m, n, kdim)

    def bind(op, shape, kind):
        if isinstance(op, Const):
            return None
        if isinstance(op, Mem):
            view = op.view
            if len(op.view.shape) == 1:
                view = V.reshape(view, shape)
            view = V.broadcast_to(view, batch + shape[-2:])
            if view is None:
                raise LoweringError("matmul operand does not broadcast")
            dims = [(s, g) for _, s, g in view.dims]
            imm = view.offset
        else:
            raise LoweringError("matmul operands must be arena tensors or uniform constants")
        bd = dims[:len(batch)]
        r0, r1 = dims[len(batch):]
        if kind == "a":
            full = bd + [r0, (0, 1), r1]
        else:
            full = bd + [(0, 1), r1, r0]
        return Acc("mem", op.port, imm, full[:-1], [full[-1]])
    acc_a = bind(a, aa, "a")
    acc_b = bind(b, bb, "b")
    oview = V.reshape(out.view, batch + (m, n))
    oacc = Acc("mem", out.port, oview.offset, [(s, g) for _, s, g in oview.dims], [(0, 1)])
    accs = [x for x in (acc_a, acc_b, oacc) if x is not None]
    sizes = plan(batch + (m, n), accs, "outer")
    inner = plan([kdim], [x for x in (acc_a, acc_b) if x is not None], "inner")
    threads, regs, rest = k.lanes(sizes, accs)

    def cell(regs):
        acc = k.const(0)

        def body(r):
            x = k.const(a.value) if acc_a is None else k.load(acc_a, r)
            y = k.const(b.value) if acc_b is None else k.load(acc_b, r)
            k.emit("mul", x, x, y)
            k.emit("add", acc, acc, x)
            k.free_r(x, y)
        k.nest(inner, list(range(len(inner))), [x for x in (acc_a, acc_b) if x is not None], regs, body, "inner")
        k.store(oacc, regs, acc)
        k.free_r(acc)
    k.nest(sizes, rest, accs, regs, cell)
    return threads


def gen_cumsum(k, x, out, shape, dim):
    accs = [access(x, shape, [dim]), access(out, shape, [dim])]
    outer = [d for i, d in enumerate(shape) if i != dim]
    sizes = plan(outer, accs, "outer")
    inner = plan([shape[dim]], accs, "inner")
    threads, regs, rest = k.lanes(sizes, accs)

    def row(regs):
        acc = k.const(0)

        def body(r):
            v = k.load(accs[0], r)
            k.emit("add", acc, acc, v)
            k.store(accs[1], r, acc)
            k.free_r(v)
        k.nest(inner, list(range(len(inner))), accs, regs, body, "inner")
        k.free_r(acc)
    k.nest(sizes, rest, accs, regs, row)
    return threads


def gen_gather(k, source, index, out, dim):
    """out[..., i, ...] = source[..., index[i], ...] (dynamic offsets, no bounds check)."""
    if not isinstance(source, Mem) or not isinstance(index, Mem):
        raise LoweringError("gather needs arena source and index tensors")
    sview, oview, iview = source.view, out.view, index.view
    if sview.dims[dim][2] != 1 or iview.dims[0][2] != 1:
        raise LoweringError("gather over a grouped view")
    count = oview.dims[dim][0]
    lane = k.lane()
    iu = k.u()
    k.emit("umulk", iu, lane, iview.dims[0][1])
    idx = k.r()
    k.emit("uloadf32", idx, iu, address(index.port, iview.offset))
    k.free_u(iu)
    dyn = k.u()
    k.emit("rtou", dyn, idx)
    k.emit("umulk", dyn, dyn, sview.dims[dim][1])
    k.free_r(idx)
    ou = k.u()
    k.emit("umulk", ou, lane, oview.dims[dim][1])
    rest = tuple(d for i, d in enumerate(oview.shape) if i != dim)
    sacc = Acc("mem", source.port, sview.offset, [(s, g) for i, (_, s, g) in enumerate(sview.dims) if i != dim])
    oacc = Acc("mem", out.port, oview.offset, [(s, g) for i, (_, s, g) in enumerate(oview.dims) if i != dim])
    sizes = plan(rest, [sacc, oacc], "outer")
    regs = {id(sacc): dyn, id(oacc): ou}

    def body(r):
        v = k.load(sacc, r)
        k.store(oacc, r, v)
        k.free_r(v)
    k.nest(sizes, list(range(len(sizes))), [sacc, oacc], regs, body)
    return count


def gen_state_reset(k, position, storage, shape):
    """Zero ``storage`` when position == 0 (new sequence); lanes over the largest dim."""
    acc = access(storage, shape)
    sizes = plan(shape, [acc], "outer")
    pacc = access(position, (1,))
    p = k.load(pacc, {})
    c = k.r()
    k.emit("addconst", c, p, -0.5)
    zero = k.const(0)
    k.emit("cmplt", c, c, zero)
    k.free_r(p)
    threads, regs, rest = k.lanes(sizes, [acc])

    def body(r):
        k.store(acc, r, zero)
    k.cond(c, lambda: k.nest(sizes, rest, [acc], regs, body))
    return threads


def gen_indexed_write(k, value, index, storage, sshape, dim):
    """storage[..., index, ...] = value (in place, one slice)."""
    vshape = sshape[:dim] + sshape[dim + 1:]
    sview = storage.view
    stride = sview.dims[dim][1]
    rest = V.View(sview.offset, tuple(d for i, d in enumerate(sview.dims) if i != dim))
    iacc = access(index, (1,))
    idx = k.load(iacc, {})
    dyn = k.u()
    k.emit("rtou", dyn, idx)
    k.emit("umulk", dyn, dyn, stride)
    k.free_r(idx)
    vacc = access(value, vshape)
    sacc = access(Mem(storage.port, rest), vshape)
    sacc.dynamic = dyn
    sizes = plan(vshape, [vacc, sacc], "outer")
    threads, regs, rest_dims = k.lanes(sizes, [vacc, sacc])

    def body(r):
        v = k.load(vacc, r) if vacc is not None else k.const(value.value)
        k.store(sacc, r, v)
        k.free_r(v)
    k.nest(sizes, rest_dims, [vacc, sacc], regs, body)
    return threads


def gen_conv1d(k, x, w, bias, out, xshape, wshape, oshape, groups, times):
    """Causal stride-one convolution with left zero padding."""
    B, Cin, L = xshape
    Cout, cpg, K = wshape
    out_per_group = Cout // groups
    T = len(times)
    lane = k.lane()
    # Lanes over output channels; batch and the selected times are serial.
    wv = w.view
    xv = x.view
    ov = out.view
    bv = bias.view if bias is not None else None
    ulane = {}
    for name, stride in (("w", wv.dims[0][1]), ("o", ov.dims[1][1]), ("b", bv.dims[0][1] if bv else 0)):
        if stride:
            u = k.u()
            k.emit("umulk", u, lane, stride)
            ulane[name] = u
    # Input channel base: (c // out_per_group) * cpg * stride_c.
    group = k.u()
    if groups == 1:
        k.emit("uconst", group, 0)
    else:
        f = k.r()
        k.emit("utor", f, lane)
        k.emit("addconst", f, f, 0.5)
        k.emit("mulconst", f, f, repr(1 / out_per_group))
        k.emit("rtou", group, f)
        k.free_r(f)
        k.emit("umulk", group, group, cpg * xv.dims[1][1])
    for b in range(B):
        for ti, t in enumerate(times):
            acc = k.const(0)
            lo = max(0, K - 1 - t)
            taps = K - lo

            def channel(chan_u):
                def tap(tap_x, tap_w):
                    xr = k.r()
                    k.emit("uloadf32", xr, tap_x, address(x.port, xv.offset + b * xv.dims[0][1]
                                                         + (t - K + 1 + lo) * xv.dims[2][1]))
                    wr = k.r()
                    k.emit("uloadf32", wr, tap_w, address(w.port, wv.offset + lo * wv.dims[2][1]))
                    k.emit("mul", xr, xr, wr)
                    k.emit("add", acc, acc, xr)
                    k.free_r(xr, wr)
                    k.emit("uaddk", tap_x, tap_x, xv.dims[2][1])
                    k.emit("uaddk", tap_w, tap_w, wv.dims[2][1])
                tx, tw = k.u(), k.u()
                k.emit("uaddk", tx, chan_u[0], 0)
                k.emit("uaddk", tw, chan_u[1], 0)
                k.loop(taps, lambda: tap(tx, tw))
                k.free_u(tx, tw)
                k.emit("uaddk", chan_u[0], chan_u[0], xv.dims[1][1])
                k.emit("uaddk", chan_u[1], chan_u[1], wv.dims[1][1])
            cx, cw = k.u(), k.u()
            k.emit("uaddk", cx, group, 0)
            if "w" in ulane:
                k.emit("uaddk", cw, ulane["w"], 0)
            else:
                k.emit("uconst", cw, 0)
            k.loop(cpg, lambda: channel((cx, cw)))
            k.free_u(cx, cw)
            if bias is not None:
                br = k.r()
                k.emit("uloadf32", br, ulane.get("b", k.zero_u()), address(bias.port, bv.offset))
                k.emit("add", acc, acc, br)
                k.free_r(br)
            k.emit("ustoref32", acc, ulane.get("o", k.zero_u()),
                   address(out.port, ov.offset + b * ov.dims[0][1] + ti * ov.dims[2][1]))
            k.free_r(acc)
    return Cout


def gen_rotary(k, x, position, out, shape, base):
    """HF rotate_half RoPE: rows over shape[:-1], pairs (i, i + D/2)."""
    D = shape[-1]
    rows = shape[:-1]
    last = len(shape) - 1
    xa = access(x, shape, [last])
    oa = access(out, shape, [last])
    if xa.inner[0][1] != 1 or oa.inner[0][1] != 1:
        raise LoweringError("RoPE head dimension view is grouped")
    xs, os_ = xa.inner[0][0], oa.inner[0][0]
    pa = access(position, rows) if not isinstance(position, Const) else None
    accs = [a for a in (xa, oa, pa) if a is not None]
    sizes = plan(rows, accs, "outer")
    threads, regs, rest = k.lanes(sizes, accs)

    def row(regs):
        pos = k.const(position.value) if pa is None else k.load(pa, regs)
        emit_rope_row(k, pos, D, base, (xa.port, xa.imm, xs, regs.get(id(xa))),
                      (oa.port, oa.imm, os_, regs.get(id(oa))))
        k.free_r(pos)
    k.nest(sizes, rest, accs, regs, row)
    return threads


def emit_rope_row(k, pos, D, base, src, dst, extra_dst=None):
    """Rotate one head row: dst[i] = x[i]*cos - x[i+h]*sin; dst[i+h] = x[i+h]*cos + x[i]*sin."""
    half = D // 2
    sport, simm, sstride, sbase = src
    dport, dimm, dstride, dbase = dst
    su, du = k.u(), k.u()
    zero = k.zero_u()
    k.emit("uaddk", su, zero if sbase is None else sbase, 0)
    k.emit("uaddk", du, zero if dbase is None else dbase, 0)
    fi = k.const(0)
    scale = -math.log(base) / half

    def body():
        f = k.r()
        k.emit("mulconst", f, fi, repr(scale))
        k.emit("exp", f, f)
        k.emit("mul", f, f, pos)
        s, c = k.sincos(f)
        k.free_r(f)
        x1, x2 = k.r(), k.r()
        k.emit("uloadf32", x1, su, address(sport, simm))
        k.emit("uloadf32", x2, su, address(sport, simm + half * sstride))
        y1, t = k.r(), k.r()
        k.emit("mul", y1, x1, c)
        k.emit("mul", t, x2, s)
        k.emit("sub", y1, y1, t)
        k.emit("ustoref32", y1, du, address(dport, dimm))
        k.emit("mul", y1, x2, c)
        k.emit("mul", t, x1, s)
        k.emit("add", y1, y1, t)
        k.emit("ustoref32", y1, du, address(dport, dimm + half * dstride))
        k.free_r(x1, x2, y1, t, s, c)
        k.emit("addconst", fi, fi, 1)
        k.emit("uaddk", su, su, sstride)
        k.emit("uaddk", du, du, dstride)
    k.loop(half, body)
    k.free_r(fi)
    k.free_u(su, du)


def gen_attention(k, a, ports):
    """Decode attention: one lane per KV head, query heads of the group in a loop.

    ports: q, k_new, v_new, position, k_cache, v_cache, scratch, out (port names).
    Scratch holds scores [heads * positions] and, with q RoPE, rotated q rows.
    """
    H, G, D, Dv, P = a["heads"], a["group"], a["head_dim"], a["value_dim"], a["positions"]
    rope = a.get("rope") or {}
    qo, qs, qd = a["q"]
    ko, ks, kd = a["k_new"]
    vo, vs, vd = a["v_new"]
    kco, kcs, kcd, kcj = a["k_cache"]
    vco, vcs, vcd, vcj = a["v_cache"]
    oo, osh, ose = a["out"]
    qrot = H * P
    lane = k.lane()
    zero = k.zero_u()
    pos = k.r()
    k.emit("uloadf32", pos, zero, address(ports["position"], 0))
    upos = k.u()
    k.emit("rtou", upos, pos)
    count = k.r()
    k.emit("addconst", count, pos, 1)
    # 1. Write this token's K/V row of the lane's KV head at slot `position`.
    kc = k.u()
    k.emit("umulk", kc, upos, kcj)
    tmp = k.u()
    if kcs:
        k.emit("umulk", tmp, lane, kcs)
        k.emit("uadd", kc, kc, tmp)
    kn = k.u()
    k.emit("umulk", kn, lane, ks) if ks else k.emit("uconst", kn, 0)
    if rope.get("k"):
        emit_rope_row(k, pos, D, rope["base"], (ports["k_new"], ko, kd, kn), (ports["k_cache"], kco, kcd, kc))
    else:
        def kcopy():
            r = k.r()
            k.emit("uloadf32", r, kn, address(ports["k_new"], ko))
            k.emit("ustoref32", r, kc, address(ports["k_cache"], kco))
            k.free_r(r)
            k.emit("uaddk", kn, kn, kd)
            k.emit("uaddk", kc, kc, kcd)
        k.loop(D, kcopy)
    vc, vn = kc, kn
    k.emit("umulk", vc, upos, vcj)
    if vcs:
        k.emit("umulk", tmp, lane, vcs)
        k.emit("uadd", vc, vc, tmp)
    k.emit("umulk", vn, lane, vs) if vs else k.emit("uconst", vn, 0)

    def vcopy():
        r = k.r()
        k.emit("uloadf32", r, vn, address(ports["v_new"], vo))
        k.emit("ustoref32", r, vc, address(ports["v_cache"], vco))
        k.free_r(r)
        k.emit("uaddk", vn, vn, vd)
        k.emit("uaddk", vc, vc, vcd)
    k.loop(Dv, vcopy)
    k.free_u(vc, vn, upos)
    # Cache row bases of this KV head.
    kbase, vbase = k.u(), k.u()
    k.emit("umulk", kbase, lane, kcs) if kcs else k.emit("uconst", kbase, 0)
    k.emit("umulk", vbase, lane, vcs) if vcs else k.emit("uconst", vbase, 0)
    # Per query head of the group: q offset, scores offset, out offset.
    if G == 1:
        head = lane
    else:
        head = k.u()
        k.emit("umulk", head, lane, G)
    qh, sh, oh = k.u(), k.u(), k.u()
    k.emit("umulk", qh, head, qs) if qs else k.emit("uconst", qh, 0)
    k.emit("umulk", sh, head, P)
    k.emit("umulk", oh, head, osh) if osh else k.emit("uconst", oh, 0)
    if rope.get("q"):
        rh = k.u()
        k.emit("umulk", rh, head, D)
    k.free_u(head, tmp)

    def query():
        if rope.get("q"):
            emit_rope_row(k, pos, D, rope["base"], (ports["q"], qo, qd, qh), (ports["scratch"], qrot, 1, rh))
            qport, qimm, qstep, qbase = ports["scratch"], qrot, 1, rh
        else:
            qport, qimm, qstep, qbase = ports["q"], qo, qd, qh
        # Scores and running maximum over slots 0..position.
        m = k.const(NEG_LARGE)
        kj, sj = k.u(), k.u()
        k.emit("uaddk", kj, kbase, 0)
        k.emit("uaddk", sj, sh, 0)

        def slot():
            qi, kd_ = k.u(), k.u()
            k.emit("uaddk", qi, qbase, 0)
            k.emit("uaddk", kd_, kj, 0)
            dot = k.const(0)

            def dim():
                x, y = k.r(), k.r()
                k.emit("uloadf32", x, qi, address(qport, qimm))
                k.emit("uloadf32", y, kd_, address(ports["k_cache"], kco))
                k.emit("mul", x, x, y)
                k.emit("add", dot, dot, x)
                k.free_r(x, y)
                k.emit("uaddk", qi, qi, qstep)
                k.emit("uaddk", kd_, kd_, kcd)
            k.loop(D, dim)
            k.free_u(qi, kd_)
            k.emit("mulconst", dot, dot, repr(float(a["scale"])) if not math.isclose(a["scale"], 1 / math.sqrt(D), rel_tol=1e-15)
                   else "1/sqrt(" + str(D) + ")")
            k.emit("ustoref32", dot, sj, address(ports["scratch"], 0))
            c = k.r()
            k.emit("cmplt", c, m, dot)
            k.cond(c, lambda: k.copy(m, dot))
            k.free_r(c, dot)
            k.emit("uaddk", kj, kj, kcj)
            k.emit("uaddk", sj, sj, 1)
        k.loop((count,), slot)
        # Exponentials and their sum.
        total = k.const(0)
        k.emit("uaddk", sj, sh, 0)

        def expo():
            x = k.r()
            k.emit("uloadf32", x, sj, address(ports["scratch"], 0))
            k.emit("sub", x, x, m)
            k.emit("exp", x, x)
            k.emit("ustoref32", x, sj, address(ports["scratch"], 0))
            k.emit("add", total, total, x)
            k.free_r(x)
            k.emit("uaddk", sj, sj, 1)
        k.loop((count,), expo)
        k.free_r(m)
        k.free_u(kj)
        # Weighted values, normalized once per output element.
        ve, oe = k.u(), k.u()
        k.emit("uaddk", ve, vbase, 0)
        k.emit("uaddk", oe, oh, 0)

        def column():
            acc = k.const(0)
            vj = k.u()
            k.emit("uaddk", vj, ve, 0)
            k.emit("uaddk", sj, sh, 0)

            def slot_v():
                x, y = k.r(), k.r()
                k.emit("uloadf32", x, sj, address(ports["scratch"], 0))
                k.emit("uloadf32", y, vj, address(ports["v_cache"], vco))
                k.emit("mul", x, x, y)
                k.emit("add", acc, acc, x)
                k.free_r(x, y)
                k.emit("uaddk", vj, vj, vcj)
                k.emit("uaddk", sj, sj, 1)
            k.loop((count,), slot_v)
            k.free_u(vj)
            k.emit("div", acc, acc, total)
            k.emit("ustoref32", acc, oe, address(ports["out"], oo))
            k.free_r(acc)
            k.emit("uaddk", ve, ve, vcd)
            k.emit("uaddk", oe, oe, ose)
        k.loop(Dv, column)
        k.free_u(ve, oe, sj)
        k.free_r(total)
        # Next query head of the group.
        if G == 1:
            return
        if qs:
            k.emit("uaddk", qh, qh, qs)
        k.emit("uaddk", sh, sh, P)
        if osh:
            k.emit("uaddk", oh, oh, osh)
        if rope.get("q"):
            k.emit("uaddk", rh, rh, D)
    if G == 1:
        query()
    else:
        k.loop(G, query)
    return H // G


def gen_conv_step(k, a, ports, views):
    """Depthwise streaming conv: one lane per channel; history state shifts in place."""
    K, C = a["kernel_size"], a["channels"]
    lane = k.lane()
    zero = k.zero_u()
    xview, wview, sview, oview = views["x"], views["w"], views["state"], views["out"]
    bview = views.get("bias")
    p = k.r()
    k.emit("uloadf32", p, zero, address(ports["position"], 0))
    reset = k.r()
    k.emit("addconst", reset, p, -0.5)
    z = k.const(0)
    k.emit("cmplt", reset, reset, z)
    k.free_r(p)

    def lane_reg(stride):
        if stride == 0:
            return zero
        if stride == 1:
            return lane
        u = k.u()
        k.emit("umulk", u, lane, stride)
        return u
    srow = lane_reg(sview.dims[0][1])
    wrow = lane_reg(wview.dims[0][1])
    xrow = lane_reg(xview.dims[0][1])
    orow = lane_reg(oview.dims[0][1])
    ss, ws = sview.dims[1][1], wview.dims[1][1]

    def clear():
        u = k.u()
        k.emit("uaddk", u, srow, 0)

        def body():
            k.emit("ustoref32", z, u, address(ports["state"], sview.offset))
            k.emit("uaddk", u, u, ss)
        k.loop(K - 1, body)
        k.free_u(u)
    k.cond(reset, clear)
    acc = k.const(0)
    if bview is not None:
        br = k.r()
        k.emit("uloadf32", br, lane_reg(bview.dims[0][1]), address(ports["bias"], bview.offset))
        k.emit("add", acc, acc, br)
        k.free_r(br)
    su, wu = k.u(), k.u()
    k.emit("uaddk", su, srow, 0)
    k.emit("uaddk", wu, wrow, 0)

    def tap():
        h, w = k.r(), k.r()
        k.emit("uloadf32", h, su, address(ports["state"], sview.offset))
        k.emit("uloadf32", w, wu, address(ports["w"], wview.offset))
        k.emit("mul", h, h, w)
        k.emit("add", acc, acc, h)
        k.free_r(h, w)
        k.emit("uaddk", su, su, ss)
        k.emit("uaddk", wu, wu, ws)
    k.loop(K - 1, tap)
    x, w = k.r(), k.r()
    k.emit("uloadf32", x, xrow, address(ports["x"], xview.offset))
    k.emit("uloadf32", w, wu, address(ports["w"], wview.offset))
    k.emit("mul", w, w, x)
    k.emit("add", acc, acc, w)
    k.free_r(w)
    k.emit("ustoref32", acc, orow, address(ports["out"], oview.offset))
    # Shift the history left by one and append the new sample.
    k.emit("uaddk", su, srow, 0)
    if K > 2:
        def shift():
            h = k.r()
            k.emit("uloadf32", h, su, address(ports["state"], sview.offset + ss))
            k.emit("ustoref32", h, su, address(ports["state"], sview.offset))
            k.free_r(h)
            k.emit("uaddk", su, su, ss)
        k.loop(K - 2, shift)
    k.emit("ustoref32", x, su, address(ports["state"], sview.offset))
    k.free_r(x, acc, reset, z)
    return C


def gen_state_update_reduce(k, a, operands, out, storage, position):
    """state <- update(state, ...); out <- reduce(reduce_tree(state_new, ...))."""
    in_shape, dims = a["in_shape"], a["dims"]
    inner_ids = [i for i in range(len(in_shape)) if i in dims]
    kept = [d for i, d in enumerate(in_shape) if i not in inner_ids]
    red = [in_shape[i] for i in inner_ids]
    accs = [None if isinstance(op, Const) or i == a["reset_port"] else access(op, in_shape, inner_ids)
            for i, op in enumerate(operands)]
    sacc = accs[a["state_port"]]
    oview = V.reshape(out.view, tuple(kept))
    oacc = access(Mem(out.port, oview), tuple(kept))
    live = [x for x in accs if x is not None]
    sizes = plan(kept, live + [oacc], "outer")
    inner = plan(red, live, "inner")
    pa = access(position, (1,))
    p = k.load(pa, {})
    reset = k.r()
    k.emit("addconst", reset, p, -0.5)
    z = k.const(0)
    k.emit("cmplt", reset, reset, z)
    k.free_r(p)
    threads, regs, rest = k.lanes(sizes, live + [oacc])
    count = math.prod(red)

    def row(regs):
        acc = k.const(0)

        def body(r):
            def leaf(kind, port):
                op = operands[port]
                if isinstance(op, Const):
                    return k.const(op.value)
                v = k.load(accs[port], r)
                if kind == "state":
                    k.cond(reset, lambda: k.copy(v, z))
                return v
            updated = k.tree(a["update"], leaf)
            k.store(sacc, r, updated)

            def rleaf(kind, port):
                if kind == "updated":
                    t = k.r()
                    k.copy(t, updated)
                    return t
                return leaf(kind, port)
            y = k.tree(a["reduce"], rleaf)
            k.emit("add", acc, acc, y)
            k.free_r(y, updated)
        k.nest(inner, list(range(len(inner))), live, regs, body, "inner")
        if a["kind"] == "mean":
            k.emit("mulconst", acc, acc, repr(1 / count))
        k.store(oacc, regs, acc)
        k.free_r(acc)
    k.nest(sizes, rest, live + [oacc], regs, row)
    k.free_r(reset, z)
    return threads


def gen_constant(k, values, out):
    """Materialize an arbitrary constant array (written every step)."""
    zero = k.zero_u()
    r = k.r()
    for i, x in enumerate(values):
        k.const(x, r)
        k.emit("ustoref32", r, zero, address(out.port, out.view.offset + i))
    k.free_r(r)
    return 1
