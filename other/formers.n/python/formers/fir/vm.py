"""Pure-Python execution of FIR graph files for numerical validation.

``FirModel`` parses a ``runtime graph`` FIR file the way the runner does
(``fir_parse``/``fir_graph``/``fir_kernel``): symbols from integer config keys,
``sym``/``require`` lines, buffer bindings in one f32 arena (per-layer buffers
and ``{i}`` weights get one slice per layer), kernels assembled from mnemonic
lines, and nodes executed in order with ``repeat layers`` blocks run per layer.

Micro-kernels run on an interpreter that mirrors ``vm_interp.nim``: R0 holds
the thread id on entry, float registers are rounded to f32 after every
operation, integer registers wrap at 32 bits and ``forbegin`` re-reads its count
register at every iteration. Each dispatch is bounded to 2**24 steps and every
load/store is checked against the buffers bound to the node's ports (stores
only to output ports), so an out-of-buffer access is an error rather than
silent corruption.

Resident externals (``fir.embedding``, ``fir.linear``, ``fir.linear_t``) are
computed in Python. ``q4`` weights are used as f32: the runner's int4
quantization is not modelled, so q4 files compare like ``--fir-f32``.
"""

import json
import math
import re
from array import array
from pathlib import Path


MAX_STEPS = 1 << 24
MAX_LOOP_DEPTH = 8
MASK = 0xFFFFFFFF

SIGNATURES = {
    "nop": "", "forend": "", "ifend": "",
    "forbegin": "r", "ifbegin": "r",
    "add": "rrr", "sub": "rrr", "mul": "rrr", "div": "rrr", "cmplt": "rrr", "uadd": "rrr", "umul": "rrr",
    "fma": "rrrr",
    "exp": "rr", "neg": "rr", "sigmoid": "rr", "softplus": "rr", "rsqrt": "rr", "rtou": "rr", "utor": "rr",
    "uconst": "rk",
    "uaddk": "rrk", "umulk": "rrk", "uandk": "rrk", "ushrk": "rrk", "ushlk": "rrk",
    "uloadf32": "rrk", "ustoref32": "rrk",
    "addconst": "rrf", "mulconst": "rrf",
    "loadconstf32": "rf",
}
# Accepted by the FIR assembler but not interpretable here (thread-group,
# atomic and legacy R-indexed memory operations): a kernel using one fails.
UNSUPPORTED = ("barriertg", "loadlid", "loadtgid", "utgid", "ulid", "uatomicadd", "loadf32", "storef32",
               "tgloadf32", "tgstoref32")


class FirVmError(ValueError):
    pass


_F = array("f", [0.0])


def f32(x):
    _F[0] = x
    return _F[0]


def _exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


# ---- expressions ---------------------------------------------------------------

def eval_int(expr, syms):
    """Integer expression semantics of ``firEvalExpr`` (+, -, *, integer /)."""
    s = expr.strip()
    if not s:
        return 0
    if s in syms:
        return syms[s]
    for ops in (("+", "-"), ("*", "/")):
        depth = 0
        for i in range(len(s) - 1, -1, -1):
            c = s[i]
            if c == ")":
                depth += 1
            elif c == "(":
                depth -= 1
            elif depth == 0 and c in ops and (i > 0 or ops[0] == "*"):
                left, right = eval_int(s[:i], syms), eval_int(s[i + 1:], syms)
                if c == "+":
                    return left + right
                if c == "-":
                    return left - right
                if c == "*":
                    return left * right
                if right == 0:
                    raise FirVmError("division by zero in FIR expression: " + s)
                return int(left / right)
    if s.startswith("(") and s.endswith(")"):
        return eval_int(s[1:-1], syms)
    try:
        return int(s)
    except ValueError:
        raise FirVmError("unknown FIR symbol or expression: " + s) from None


class _Scan:
    """Float expression grammar of ``fkParse`` (kernel arguments and options)."""

    def __init__(self, syms, address):
        self.syms, self.address = syms, address

    def eval(self, text):
        self.text, self.pos = text, 0
        value = self.parse(0)
        if self.pos != len(text):
            raise FirVmError("trailing characters in '%s'" % text)
        return value

    def parse(self, level):
        t = self.text
        if level < 2:
            value = self.parse(level + 1)
            ops = "+-" if level == 0 else "*/"
            while self.pos < len(t) and t[self.pos] in ops:
                op = t[self.pos]
                self.pos += 1
                right = self.parse(level + 1)
                if op == "+":
                    value += right
                elif op == "-":
                    value -= right
                elif op == "*":
                    value *= right
                else:
                    if right == 0:
                        raise FirVmError("division by zero in '%s'" % t)
                    value /= right
            return value
        if self.pos >= len(t):
            raise FirVmError("unexpected end of expression '%s'" % t)
        c = t[self.pos]
        if c == "(":
            self.pos += 1
            value = self.parse(0)
            if self.pos >= len(t) or t[self.pos] != ")":
                raise FirVmError("missing ')' in '%s'" % t)
            self.pos += 1
            return value
        if c == "-":
            self.pos += 1
            return -self.parse(2)
        if c.isdigit() or c == ".":
            m = re.compile(r"[0-9.]+(?:[eE][+-]?[0-9]+)?").match(t, self.pos)
            self.pos = m.end()
            return float(m.group(0))
        m = re.compile(r"[A-Za-z0-9_.]+").match(t, self.pos + (c == "@"))
        if m is None:
            raise FirVmError("unexpected character '%s' in '%s'" % (c, t))
        name = m.group(0)
        self.pos = m.end()
        if c == "@":
            return float(self.address(name))
        if self.pos < len(t) and t[self.pos] == "(":
            if name != "sqrt":
                raise FirVmError("unknown function " + name)
            self.pos += 1
            inner = self.parse(0)
            if self.pos >= len(t) or t[self.pos] != ")" or inner < 0:
                raise FirVmError("invalid sqrt argument in '%s'" % t)
            self.pos += 1
            return math.sqrt(inner)
        if name not in self.syms:
            raise FirVmError("unknown name '%s' in '%s'" % (name, t))
        return self.syms[name]


# ---- parsing -------------------------------------------------------------------

def parse(text, config):
    """Parse FIR text with a config dict; return a dict program description."""
    prog = {"symbols": {}, "epsilon": float(config.get("layer_norm_epsilon", 1e-5)), "prefixes": [],
            "buffers": {}, "kernels": {}, "nodes": [], "layout": "classic", "model": "", "runtime": ""}
    syms = prog["symbols"]
    for key, value in config.items():
        if type(value) is int and value > 0:
            syms[key] = value
    in_repeat, open_kernel = False, None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        w = line.split()

        def fail(message):
            raise FirVmError("FIR line %d: %s" % (number, message))
        if open_kernel is not None and w[0] != "endkernel":
            open_kernel["body"].append((w[0].lower(), w[1:], number))
            continue
        head = w[0]
        if head == "fir":
            if w != ["fir", "1"]:
                fail("expected 'fir 1'")
        elif head == "model":
            prog["model"] = w[1]
        elif head == "config":
            pass
        elif head == "runtime":
            prog["runtime"] = w[1]
        elif head == "layout":
            prog["layout"] = w[1]
        elif head == "prefix":
            prog["prefixes"].extend(w[1:])
        elif head == "epsilon":
            value = config.get(w[1])
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not value > 0 or not math.isfinite(value):
                fail("epsilon key %s must name a positive config value" % w[1])
            prog["epsilon"] = f32(value)
        elif head == "require":
            if len(w) >= 4 and w[1] == "config":
                value = config.get(w[2])
                actual = "" if value is None else value if isinstance(value, str) else json.dumps(value)
                if actual and actual != w[3]:
                    fail("config %s is %s, model requires %s" % (w[2], actual, w[3]))
            elif len(w) >= 5 and w[1] == "sym" and "=" in w[3:]:
                eq = w.index("=")
                if eval_int("".join(w[2:eq]), syms) != eval_int("".join(w[eq + 1:]), syms):
                    fail("requirement failed: " + " ".join(w[2:]))
            else:
                fail("malformed require")
        elif head == "sym":
            if len(w) < 4 or w[2] != "=":
                fail("malformed symbol")
            syms[w[1]] = eval_int("".join(w[3:]), syms)
        elif head == "buf":
            b = {"name": w[1], "dtype": w[2], "kind": "buf", "per_layer": False, "role": None, "tensor": ""}
            b["elements"] = eval_int(w[3], syms)
            b["dims"] = [b["elements"]]
            for token in w[4:]:
                if token == "per_layer":
                    b["per_layer"] = True
                elif token.startswith(("input:", "output:")):
                    b["role"] = token.split(":", 1)[1]
            if b["dtype"] != "f32" or b["elements"] <= 0 or w[1] in prog["buffers"]:
                fail("invalid or duplicate buffer " + w[1])
            prog["buffers"][w[1]] = b
        elif head == "weight":
            if len(w) < 6 or w[4] != "tensor" or not (w[-1].startswith("[") and w[-1].endswith("]")):
                fail("malformed weight")
            dims = [eval_int(d, syms) for d in w[-1][1:-1].split(",")]
            if any(d <= 0 for d in dims) or w[2] not in ("f32", "q4") or w[1] in prog["buffers"]:
                fail("invalid weight " + w[1])
            prog["buffers"][w[1]] = {"name": w[1], "dtype": w[2], "kind": "weight", "per_layer": False,
                                     "role": None, "tensor": w[3], "dims": dims,
                                     "elements": math.prod(dims)}
        elif head == "repeat":
            if in_repeat or w[1:] != ["layers"]:
                fail("FIR supports one layer repeat scope")
            in_repeat = True
        elif head == "endrepeat":
            if not in_repeat:
                fail("unexpected endrepeat")
            in_repeat = False
        elif head == "kernel":
            if in_repeat or w[1] in prog["kernels"]:
                fail("invalid or duplicate kernel")
            k = {"name": w[1], "ins": 0, "outs": 0, "threads": "1", "tg": "1", "tgbytes": "16",
                 "size": [], "expect": [], "body": []}
            for field in w[2:]:
                key, _, value = field.partition("=")
                if key in ("in", "out"):
                    k[key + "s"] = int(value)
                elif key in ("threads", "tg", "tgbytes"):
                    k[key] = value
                elif key in ("size", "expect"):
                    k[key] = value.split(",")
                elif key == "tag":
                    k["tag"] = value
                else:
                    fail("unknown kernel option " + key)
            prog["kernels"][w[1]] = open_kernel = k
        elif head == "endkernel":
            if open_kernel is None:
                fail("unexpected endkernel")
            open_kernel = None
        elif head == "node":
            if "->" not in w or w.index("->") < 4 or w[-1] == "->":
                fail("node requires inputs -> outputs")
            arrow = w.index("->")
            if any(n["name"] == w[1] for n in prog["nodes"]):
                fail("duplicate node " + w[1])
            prog["nodes"].append({"name": w[1], "kind": w[2], "op": w[3], "inputs": w[4:arrow],
                                  "outputs": w[arrow + 1:], "per_layer": in_repeat, "line": number})
        else:
            fail("unknown declaration '%s'" % head)
    if in_repeat or open_kernel is not None or not prog["model"]:
        raise FirVmError("FIR requires a version, model, and closed repeat scopes")
    if prog["runtime"] != "graph":
        raise FirVmError("the Python FIR VM runs 'runtime graph' files")
    for name in ("layers", "positions", "vocab"):
        if name not in syms:
            raise FirVmError("FIR graph needs symbol '%s'" % name)
    for role in ("token", "position", "logits"):
        if sum(b["role"] == role for b in prog["buffers"].values()) != 1:
            raise FirVmError("FIR graph needs one buffer with role " + role)
    return prog


# ---- kernels -------------------------------------------------------------------

class Program:
    """One assembled kernel bound to arena addresses."""

    def __init__(self, ops, jumps, threads, loads, stores, where):
        self.ops, self.jumps, self.threads = ops, jumps, threads
        self.loads, self.stores, self.where = loads, stores, where


def assemble(kernel, prog, node, memory, layer_memory):
    ins, outs = node["inputs"], node["outputs"]
    where = "%s (node %s)" % (kernel["name"], node["name"])
    if len(ins) != kernel["ins"] or len(outs) != kernel["outs"]:
        raise FirVmError("FIR kernel %s: expects %d inputs and %d outputs" % (where, kernel["ins"], kernel["outs"]))
    buffers = prog["buffers"]
    syms = {k: float(v) for k, v in prog["symbols"].items()}
    syms["eps"] = float(prog["epsilon"])
    syms["pi"] = math.pi
    for name in ins + outs:
        if name not in buffers:
            raise FirVmError("FIR kernel %s: unknown buffer %s" % (where, name))
        if name not in layer_memory:
            raise FirVmError("FIR kernel %s: missing memory binding: %s" % (where, name))
    for i, name in enumerate(ins):
        syms["in%d.size" % i] = float(buffers[name]["elements"])
    for i, name in enumerate(outs):
        syms["out%d.size" % i] = float(buffers[name]["elements"])
        if buffers[name]["kind"] == "weight":
            raise FirVmError("FIR kernel %s: output must be an activation: %s" % (where, name))

    def address(name):
        target = name
        if re.fullmatch(r"in\d", name) and int(name[2]) < len(ins):
            target = ins[int(name[2])]
        elif re.fullmatch(r"out\d", name) and int(name[3]) < len(outs):
            target = outs[int(name[3])]
        if target not in layer_memory:
            raise FirVmError("FIR kernel %s: no memory binding for @%s" % (where, name))
        return layer_memory[target]
    scan = _Scan(syms, address)
    sizes = [syms[p + ".size"] for p in kernel["size"]]
    if len(set(sizes)) > 1:
        raise FirVmError("FIR kernel %s: buffer sizes differ" % where)
    for item in kernel["expect"]:
        port, _, expr = item.partition(":")
        if syms.get(port + ".size") != scan.eval(expr):
            raise FirVmError("FIR kernel %s: %s size mismatch" % (where, port))
    ops, stack, jumps, depth = [], [], {}, 0
    for mnemonic, args, number in kernel["body"]:
        if mnemonic in UNSUPPORTED:
            raise FirVmError("FIR kernel %s line %d: %s is not supported by the Python VM" % (kernel["name"], number, mnemonic))
        sig = SIGNATURES.get(mnemonic)
        if sig is None:
            raise FirVmError("FIR kernel %s line %d: unknown mnemonic '%s'" % (kernel["name"], number, mnemonic))
        if len(args) != len(sig):
            raise FirVmError("FIR kernel %s line %d: '%s' takes %d arguments" % (kernel["name"], number, mnemonic, len(sig)))
        values = []
        for kind, text in zip(sig, args):
            v = scan.eval(text)
            if kind == "f":
                values.append(f32(v))
            else:
                if v < 0 or v > MASK or v != math.floor(v):
                    raise FirVmError("FIR kernel %s line %d: expected a non-negative integer" % (kernel["name"], number))
                if kind == "r" and v > 31:
                    raise FirVmError("FIR kernel %s line %d: register out of range" % (kernel["name"], number))
                values.append(int(v))
        pc = len(ops)
        if mnemonic in ("forbegin", "ifbegin"):
            stack.append((mnemonic, pc))
            if mnemonic == "forbegin":
                depth += 1
                if sum(1 for m, _ in stack if m == "forbegin") > MAX_LOOP_DEPTH:
                    raise FirVmError("FIR kernel %s: loop nesting exceeds %d" % (kernel["name"], MAX_LOOP_DEPTH))
        elif mnemonic in ("forend", "ifend"):
            want = "forbegin" if mnemonic == "forend" else "ifbegin"
            if not stack or stack[-1][0] != want:
                raise FirVmError("FIR kernel %s line %d: unbalanced %s" % (kernel["name"], number, mnemonic))
            _, start = stack.pop()
            jumps[start] = pc + 1
            jumps[pc] = start + 1
        ops.append((mnemonic, *values))
    if stack:
        raise FirVmError("FIR kernel %s: unclosed scope" % kernel["name"])
    threads = int(scan.eval(kernel["threads"]))

    def ranges(names):
        return sorted({(layer_memory[n], layer_memory[n] + buffers[n]["elements"]) for n in names})
    return Program(ops, jumps, threads, ranges(ins + outs), ranges(outs), where)


def _inside(ranges, address):
    for lo, hi in ranges:
        if lo <= address < hi:
            return True
    return False


def run_program(p, mem):
    """Execute all threads of one dispatch in thread order (tg=1, no barriers)."""
    steps = 0
    ops, jumps = p.ops, p.jumps
    loads, stores = p.loads, p.stores
    n = len(ops)
    for tid in range(p.threads):
        R = [0.0] * 32
        U = [0] * 32
        R[0] = float(tid)
        loops = []
        pc = 0
        while pc < n:
            steps += 1
            if steps > MAX_STEPS:
                raise FirVmError("%s exceeds %d VM steps per dispatch" % (p.where, MAX_STEPS))
            op = ops[pc]
            m = op[0]
            pc += 1
            if m == "uloadf32":
                address = (U[op[2]] + op[3]) & MASK
                if not _inside(loads, address):
                    raise FirVmError("%s: load at %d outside its port buffers" % (p.where, address))
                R[op[1]] = mem[address]
            elif m == "ustoref32":
                address = (U[op[2]] + op[3]) & MASK
                if not _inside(stores, address):
                    raise FirVmError("%s: store at %d outside its output buffers" % (p.where, address))
                mem[address] = R[op[1]]
            elif m == "add":
                R[op[1]] = f32(R[op[2]] + R[op[3]])
            elif m == "sub":
                R[op[1]] = f32(R[op[2]] - R[op[3]])
            elif m == "mul":
                R[op[1]] = f32(R[op[2]] * R[op[3]])
            elif m == "div":
                b = R[op[3]]
                a = R[op[2]]
                R[op[1]] = f32(a / b) if b != 0 else (math.nan if a == 0 or a != a else math.copysign(math.inf, a) * math.copysign(1, b))
            elif m == "mulconst":
                R[op[1]] = f32(R[op[2]] * op[3])
            elif m == "addconst":
                R[op[1]] = f32(R[op[2]] + op[3])
            elif m == "loadconstf32":
                R[op[1]] = op[2]
            elif m == "uaddk":
                U[op[1]] = (U[op[2]] + op[3]) & MASK
            elif m == "umulk":
                U[op[1]] = (U[op[2]] * op[3]) & MASK
            elif m == "uadd":
                U[op[1]] = (U[op[2]] + U[op[3]]) & MASK
            elif m == "umul":
                U[op[1]] = (U[op[2]] * U[op[3]]) & MASK
            elif m == "uconst":
                U[op[1]] = op[2] & MASK
            elif m == "forbegin":
                if int(R[op[1]]) <= 0:
                    pc = jumps[pc - 1]
                else:
                    loops.append([pc, op[1], 0])
            elif m == "forend":
                top = loops[-1]
                top[2] += 1
                if top[2] < int(R[top[1]]):
                    pc = top[0]
                else:
                    loops.pop()
            elif m == "ifbegin":
                if R[op[1]] == 0.0:
                    pc = jumps[pc - 1]
            elif m == "ifend":
                pass
            elif m == "cmplt":
                R[op[1]] = 1.0 if R[op[2]] < R[op[3]] else 0.0
            elif m == "exp":
                R[op[1]] = f32(_exp(R[op[2]]))
            elif m == "neg":
                R[op[1]] = -R[op[2]]
            elif m == "sigmoid":
                R[op[1]] = f32(1.0 / (1.0 + _exp(-R[op[2]])))
            elif m == "softplus":
                x = R[op[2]]
                R[op[1]] = f32(math.log1p(_exp(x))) if x < 80 else x
            elif m == "rsqrt":
                x = R[op[2]]
                R[op[1]] = f32(1.0 / math.sqrt(x)) if x > 0 else (math.inf if x == 0 else math.nan)
            elif m == "rtou":
                x = R[op[2]]
                if not x >= 0 or x > MASK:
                    raise FirVmError("%s: rtou of %r (negative or out of range)" % (p.where, x))
                U[op[1]] = int(x)
            elif m == "utor":
                R[op[1]] = f32(float(U[op[2]]))
            elif m == "fma":
                R[op[1]] = f32(R[op[2]] * R[op[3]] + R[op[4] & 31])
            elif m == "uandk":
                U[op[1]] = U[op[2]] & op[3]
            elif m == "ushrk":
                U[op[1]] = U[op[2]] >> op[3]
            elif m == "ushlk":
                U[op[1]] = (U[op[2]] << op[3]) & MASK
            elif m == "nop":
                pass
            else:
                raise FirVmError("%s: mnemonic %s is not interpretable" % (p.where, m))
    return steps


# ---- graph ---------------------------------------------------------------------

class FirModel:
    """Resident-arena execution of a FIR graph with checkpoint weights by name."""

    def __init__(self, path_or_text, config=None, weights=None):
        text = path_or_text
        if isinstance(path_or_text, Path) or (isinstance(path_or_text, str) and "\n" not in path_or_text):
            path = Path(path_or_text)
            text = path.read_text(encoding="utf-8")
            if config is None:
                m = re.search(r"^config\s+(\S+)", text, re.M)
                config = json.loads((path.parent / (m.group(1) if m else "config.json")).read_text())
        self.config = config
        self.prog = prog = parse(text, config)
        self.layers = prog["symbols"]["layers"]
        buffers = prog["buffers"]
        memory, offset = {}, 0
        for name, b in buffers.items():
            arena = b["kind"] != "weight" or len(b["dims"]) == 1 or (
                prog["layout"] == "script" and b["dtype"] == "f32" and len(b["dims"]) != 2)
            if arena:
                memory[name] = offset
                offset += b["elements"] * (self.layers if b["per_layer"] or "{i}" in b["tensor"] else 1)
        self.memory = memory
        self.mem = array("f", bytes(4 * offset))
        self.resident = {}
        self.position = 0
        self.dispatches = 0
        self.steps = 0
        weights = weights or {}
        for name, b in buffers.items():
            if b["kind"] != "weight":
                continue
            for layer in range(self.layers if "{i}" in b["tensor"] else 1):
                tensor = b["tensor"].replace("{i}", str(layer))
                source = self.resolve(weights, tensor)
                values = weights[source]
                if len(values) != b["elements"]:
                    raise FirVmError("FIR graph invalid weight element count: " + tensor)
                if name in memory:
                    base = self.binding(name, layer)
                    self.mem[base:base + len(values)] = array("f", values)
                else:
                    self.resident[tensor] = array("f", values)
        self.programs = {}
        for node in prog["nodes"]:
            if node["kind"] == "micro":
                if node["op"] not in prog["kernels"]:
                    raise FirVmError("FIR node %s: unknown kernel %s" % (node["name"], node["op"]))
                for layer in range(self.layers if node["per_layer"] else 1):
                    self.programs[(node["name"], layer)] = assemble(
                        prog["kernels"][node["op"]], prog, node, memory, self.layer_memory(layer))
            elif node["kind"] == "ext":
                if node["op"] not in ("fir.embedding", "fir.linear", "fir.linear_t"):
                    raise FirVmError("FIR graph unsupported resident external: " + node["op"])
                self.validate_external(node)
            else:
                raise FirVmError("FIR graph executor supports micro and ext nodes: " + node["name"])
        roles = {b["role"]: n for n, b in buffers.items() if b["role"]}
        self.roles = roles

    def resolve(self, weights, tensor):
        if tensor in weights:
            return tensor
        for prefix in self.prog["prefixes"]:
            if prefix + tensor in weights:
                return prefix + tensor
        raise FirVmError("FIR graph checkpoint lacks tensor: " + tensor)

    def binding(self, name, layer):
        b = self.prog["buffers"][name]
        base = self.memory[name]
        if b["per_layer"] or "{i}" in b["tensor"]:
            base += layer * b["elements"]
        return base

    def layer_memory(self, layer):
        return {name: self.binding(name, layer) for name in self.memory}

    def validate_external(self, node):
        buffers = self.prog["buffers"]
        x, w = (buffers[k] for k in node["inputs"][:2])
        y = buffers[node["outputs"][0]]
        if (w["kind"] != "weight" or len(w["dims"]) != 2 or x["kind"] == "weight" or y["kind"] == "weight"
                or (node["op"] != "fir.linear" and w["dtype"] != "f32")):
            raise FirVmError("FIR graph external requires matching weight dtype and f32 activations: " + node["name"])
        if node["op"] == "fir.embedding":
            if x["elements"] != 1 or w["dims"][1] != y["elements"]:
                raise FirVmError("FIR graph embedding shape mismatch: " + node["name"])
        else:
            inner = w["dims"][0] if node["op"] == "fir.linear" else w["dims"][1]
            rows = x["elements"] // inner if inner else 0
            expected = [inner, y["elements"] // rows] if node["op"] == "fir.linear" and rows else \
                [y["elements"] // rows, inner] if rows else None
            if rows < 1 or rows * inner != x["elements"] or w["dims"] != expected:
                raise FirVmError("FIR graph matrix shape mismatch: " + node["name"])

    def reset(self):
        """New sequence; like firGraphReset, buffers are not cleared."""
        self.position = 0

    def run_node(self, node, layer, token):
        if node["kind"] == "micro":
            self.dispatches += 1
            self.steps += run_program(self.programs[(node["name"], layer)], self.mem)
            return
        buffers = self.prog["buffers"]
        xin, wname = node["inputs"][:2]
        out = node["outputs"][0]
        w = buffers[wname]
        weight = self.resident[w["tensor"].replace("{i}", str(layer))]
        n = buffers[out]["elements"]
        ob = self.binding(out, layer)
        if node["op"] == "fir.embedding":
            role = buffers[xin]["role"]
            if role not in ("token", "position"):
                raise FirVmError("FIR graph embedding input needs role token or position: " + xin)
            index = token if role == "token" else self.position
            if not 0 <= index < w["dims"][0]:
                raise FirVmError("FIR graph embedding index outside weight")
            self.mem[ob:ob + n] = weight[index * n:(index + 1) * n]
            return
        total = buffers[xin]["elements"]
        xb = self.binding(xin, layer)
        x = list(self.mem[xb:xb + total])
        y = array("f", bytes(4 * n))
        k = w["dims"][0] if node["op"] == "fir.linear" else w["dims"][1]
        rows = total // k
        cols = n // rows
        for r in range(rows):
            xr = x[r * k:(r + 1) * k]
            for j in range(cols):
                if node["op"] == "fir.linear":
                    y[r * cols + j] = math.fsum(xr[i] * weight[i * cols + j] for i in range(k))
                else:
                    y[r * cols + j] = math.fsum(xr[i] * weight[j * k + i] for i in range(k))
        self.mem[ob:ob + n] = y

    def step(self, token):
        """One decode step; returns the logits as a list of floats."""
        syms = self.prog["symbols"]
        if not 0 <= token < syms["vocab"]:
            raise FirVmError("FIR graph token outside vocabulary")
        if self.position >= syms["positions"]:
            raise FirVmError("FIR graph context exceeds n_positions")
        self.mem[self.memory[self.roles["token"]]] = float(token)
        self.mem[self.memory[self.roles["position"]]] = float(self.position)
        nodes = self.prog["nodes"]
        i = 0
        while i < len(nodes):
            if nodes[i]["per_layer"]:
                stop = i
                while stop < len(nodes) and nodes[stop]["per_layer"]:
                    stop += 1
                for layer in range(self.layers):
                    for node in nodes[i:stop]:
                        self.run_node(node, layer, token)
                i = stop
            else:
                self.run_node(nodes[i], 0, token)
                i += 1
        base = self.memory[self.roles["logits"]]
        self.position += 1
        return list(self.mem[base:base + syms["vocab"]])
