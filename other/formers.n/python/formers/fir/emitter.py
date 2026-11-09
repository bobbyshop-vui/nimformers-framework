"""Emit FIR v1 text for the generic resident graph runner (``runtime graph``).

Emission works on the optimized SSA graph:

1. ``extract_repeats`` folds one homogeneous layer stack into ``repeat layers``.
2. State versions are analysed per storage. When no stale version is read after
   a write, all versions alias the persistent buffer and writes happen in place.
3. View chains (reshape/permute/slice/expand/strided cat) are resolved into
   strided reads of their base buffer; copies are emitted only where a buffer is
   required (resident externals, graph output, repeat carry, irregular cats).
4. Each remaining operation becomes one resident external or one loop-based
   micro-kernel (``codegen.py``). Identical kernel bodies are shared.

Names are derived from module scopes and checkpoint names, so re-exports of an
unchanged model are byte-identical and small model edits give small diffs.
"""

import copy
import json
import re
from dataclasses import replace
from pathlib import Path

from . import views as V
from . import codegen as C
from . import custom
from .codegen import LoweringError
from .graph import ELEMENTWISE, Node
from .repeat import extract_repeats


LAYER_KEYS = ("n_layer", "num_hidden_layers", "n_layers", "num_layers")
EPSILON_KEYS = ("layer_norm_epsilon", "rms_norm_eps", "layer_norm_eps", "norm_epsilon", "eps")
POSITION_KEYS = ("n_positions", "max_position_embeddings", "n_ctx")


def position_key(config):
    """Config key that holds the context length (the first present positive integer key)."""
    return next((k for k in POSITION_KEYS if type(config.get(k)) is int and config[k] > 0), "n_positions")
VIEW_NODE_OPS = V.VIEW_NODE_OPS
_OP_TAGS = {"attention_decode": "attention", "layernorm": "norm", "rmsnorm": "norm", "norm": "norm",
            "state_update_reduce": "ssm", "conv1d_step": "conv", "linear_t": "linear",
            "embedding": "embed", "state_read": "state", "state_write": "state"}


def _affine(value):
    """Return (offset, coefs) when a constant is offset + sum(coef*index), coefs >= 0 ints."""
    shape, data = value.shape, value.constant
    if not shape:
        return None
    origin = data[0]
    strides = V.row_major(shape)
    coefs = []
    for d, n in enumerate(shape):
        c = data[strides[d]] - origin if n > 1 else 0.0
        if c != int(c) or c < 0:
            return None
        coefs.append(int(c))
    for i, x in enumerate(data):
        idx = []
        rest = i
        for s in strides:
            idx.append(rest // s)
            rest %= s
        if x != origin + sum(c * j for c, j in zip(coefs, idx)):
            return None
    if any(abs(x) >= 2 ** 24 for x in data):
        return None
    return origin, coefs


class _Namer:
    """Deterministic, readable, unique identifiers."""

    RESERVED = re.compile(r"^(in|out)\d$")

    def __init__(self):
        self.used = set()

    def take(self, candidates):
        for c in candidates:
            c = re.sub(r"[^A-Za-z0-9_]", "_", c).strip("_") or "v"
            if c[0].isdigit():
                c = "v_" + c
            if self.RESERVED.match(c):
                c = "b_" + c
            if c not in self.used:
                self.used.add(c)
                return c
        base = c
        i = 2
        while f"{base}_{i}" in self.used:
            i += 1
        self.used.add(f"{base}_{i}")
        return f"{base}_{i}"


def _scope_parts(scope):
    parts = [p for p in scope.split(".") if p]
    # Drop numeric layer indexes; the repeat template is layer 0.
    return [p for p in parts if not p.isdigit()]


def _suffixes(parts, tag, minimum=1):
    out = []
    for n in range(minimum, len(parts) + 1):
        out.append("_".join(parts[-n:] + ([tag] if tag else [])))
    if not parts:
        out.append(tag)
    return out


# Op identity tags for kernel headers (`tag=` is metadata only; it lets a recorder or a lifter map a
# micro-kernel back to the PyTorch op it implements). Fused kernels list their ops joined by `+`.
KERNEL_TAGS = {
    "add": "aten::add", "add_2": "aten::add", "sub": "aten::sub", "mul": "aten::mul", "div": "aten::div",
    "layernorm": "aten::native_layer_norm", "rmsnorm": "aten::rms_norm",
    "add_layernorm": "aten::add+aten::native_layer_norm", "add_rmsnorm": "aten::add+aten::rms_norm",
    "attention": "aten::scaled_dot_product_attention", "gelu_new": "aten::gelu",
    "add_gelu_new": "aten::add+aten::gelu", "silu_mul": "aten::silu+aten::mul",
    "rotary": "fir::rotary_embedding", "conv_step": "aten::conv1d", "ssm_step": "fir::ssm_step",
}


def kernel_tag(family):
    """Tag for a kernel family; generic families are named after their graph op, so `aten::<family>`."""
    tag = KERNEL_TAGS.get(family)
    if tag is None:
        tag = "aten::" + "".join(c if (c.isalnum() or c == "_") else "_" for c in family)
    return tag


class _Emission:
    def __init__(self, graph, prefix, require_config, tags=False):
        self.tags = tags
        # A private value table: emission may add temporaries, never to the caller's graph.
        self.graph = copy.copy(graph)
        self.graph.values = dict(graph.values)
        self.prefix = prefix or ""
        # A sequence names config keys whose current values become requirements;
        # a mapping gives explicit values (also for keys the config omits).
        items = require_config.items() if isinstance(require_config, dict) else (
            (key, graph.config.get(key)) for key in (require_config or ()))
        self.require_config = tuple(items)
        self.errors = []

    # ---- names and aliases ---------------------------------------------------
    def root(self, k):
        seen = set()
        while k in self.aliases:
            if k in seen:
                raise LoweringError("cyclic alias")
            seen.add(k)
            k = self.aliases[k]
        return k

    def run(self):
        g = self.graph
        self.validate()
        try:
            schedule, aliases, templates, layer_states, layers = extract_repeats(g)
        except ValueError as error:
            raise LoweringError(str(error)) from error
        self.aliases, self.templates, self.layer_states, self.layers = dict(aliases), templates, layer_states, layers
        self.roles = {v.role: k for k, v in g.values.items() if v.role}
        # Rewrite every scheduled node to root names.
        self.schedule = []
        for n, repeated in schedule:
            self.schedule.append((self.rename(n), repeated))
        self.output = self.root(g.output)
        self.split_carry()
        self.index = {id(n): i for i, (n, _) in enumerate(self.schedule)}
        self.prods = {k: n for n, _ in self.schedule for k in n.outputs}
        self.cons = {}
        for n, _ in self.schedule:
            for k in dict.fromkeys(n.inputs):
                self.cons.setdefault(k, []).append(n)
        self.analyse_states()
        self.analyse_views()
        return self.build()

    def split_carry(self):
        """Write the repeat carry through a temporary when the last layer node reads it non-pointwise.

        The last node of the layer body stores into the carried activation. A
        pointwise node (or a norm, one row per lane) reading the carry at the
        same element is safe in place; any other reader (convolution windows,
        matmul, views) would see partially overwritten input.
        """
        g = self.graph
        last = max((i for i, (_, rep) in enumerate(self.schedule) if rep), default=None)
        if last is None:
            return
        node = self.schedule[last][0]
        carry = node.output
        prods = {k: n for n, _ in self.schedule for k in n.outputs}
        reads = [k for k in node.inputs if k == carry or V.resolve(g, k, prods)[0] == carry]
        if not reads:
            return
        direct = all(k == carry for k in reads)
        if direct and (node.op in ELEMENTWISE or node.op in ("fused", "layernorm", "rmsnorm", "norm")):
            return
        temp = carry + "_next"
        while temp in g.values:
            temp += "_"
        g.values[temp] = replace(g.values[carry])
        moved = replace(node, output=temp)
        self.schedule[last] = (moved, True)
        self.schedule.insert(last + 1, (Node("copy", (temp,), carry, {}, node.scope), True))

    def rename(self, n):
        r = self.root
        return replace(n, inputs=tuple(r(k) for k in n.inputs), output=r(n.output),
                       extra=tuple(r(k) for k in n.extra), writes=tuple(r(k) for k in n.writes))

    def validate(self):
        g = self.graph
        errors = []
        roles = {v.role: k for k, v in g.values.items() if v.role}
        arguments = sorted(k for k, v in g.values.items() if v.role == "argument")
        if arguments:
            errors.append("the runner ABI has token and position inputs only; inputs %s have no runner role "
                          "(trace and fir.evaluate accept them)" % ", ".join(arguments))
        elif set(roles) != {"token", "position"} or any(g.values[k].size != 1 for k in roles.values()):
            errors.append("generic FIR decode requires one scalar token and position input; use trace/reference for batched graphs")
        extra = [k for k in g.output_names()[1:]]
        if extra:
            errors.append("the runner ABI exports one logits output; the graph has %d outputs "
                          "(fir.evaluate(..., all_outputs=True) returns all of them)" % (len(extra) + 1))
        if g.output not in g.values or g.values[g.output].size != g.config.get("vocab_size"):
            errors.append("decode output must contain vocab_size logits")
        if any(type(g.config.get(k)) is not int or g.config[k] <= 0 for k in (position_key(g.config), "vocab_size")):
            errors.append("export config requires a positive context length (%s) and vocab_size" % "/".join(POSITION_KEYS))
        if g.output in g.values and (g.values[g.output].weight or g.values[g.output].role):
            errors.append("decode logits must be produced by an operation in an activation buffer")
        if "ssm_cfg" in g.config:
            errors.append("flatten ssm_cfg dimensions into top-level config keys for the generic FIR parser")
        if errors:
            raise LoweringError("Cannot export FIR:\n- " + "\n- ".join(errors))

    # ---- state versions ------------------------------------------------------
    def analyse_states(self):
        g = self.graph
        self.state_mode = {}      # storage -> "inplace" | "copy" | "native"
        self.skip = set()          # node ids not emitted
        self.value_alias = {}      # value -> buffer it is written into
        chains = {}
        for n, _ in self.schedule:
            if n.op == "state_read":
                chains.setdefault(n.inputs[0], {"read": None, "writes": [], "native": []})["read"] = n
            elif n.op == "state_write":
                chains.setdefault(n.writes[0], {"read": None, "writes": [], "native": []})["writes"].append(n)
            elif n.writes:
                for s in n.writes:
                    chains.setdefault(s, {"read": None, "writes": [], "native": []})["native"].append(n)
        for storage, chain in chains.items():
            if chain["native"]:
                if chain["read"] or chain["writes"]:
                    raise LoweringError("state %s mixes fused and unfused updates" % g.values[storage].state)
                self.state_mode[storage] = "native"
                continue
            read, writes = chain["read"], chain["writes"]
            versions = [read.output] if read else []
            ok = True
            for w in writes:
                v = w.inputs[0]
                wi = self.index[id(w)]
                for u in self.uses(v):
                    if u is not w and self.index[id(u)] > wi:
                        ok = False
                versions.append(w.output)
            self.state_mode[storage] = "inplace" if ok else "copy"
            if not ok:
                continue
            for v in versions:
                self.value_alias[v] = storage
            if read is not None:
                others = [u for u in self.uses(read.output) if not (u.op == "state_write" and u.inputs[0] == read.output)]
                first = writes[0] if writes else None
                # A read is emitted as a reset-at-position-0 kernel unless the
                # first write replaces the whole state before anything reads it.
                if not (others or first is None or len(first.inputs) == 3):
                    self.skip.add(id(read))
            for w in writes:
                if len(w.inputs) == 2:
                    value = w.inputs[1]
                    p = self.prods.get(value)
                    prev_users = [u for u in self.uses(w.inputs[0]) if u is not w]
                    if (p is not None and p.op not in VIEW_NODE_OPS and p.op not in ("embedding",)
                            and not p.writes and [id(u) for u in self.cons.get(value, [])] == [id(w)]
                            and all(self.index[id(u)] < self.index[id(p)] for u in prev_users)
                            and g.values[value].shape == g.values[storage].shape and value not in self.value_alias
                            and not g.values[value].role and value != self.output):
                        self.value_alias[value] = storage
                        self.skip.add(id(w))

    def uses(self, k):
        """Consumers of ``k``, following view nodes (their readers read ``k``)."""
        out, todo, seen = [], [k], set()
        while todo:
            v = todo.pop()
            for u in self.cons.get(v, []):
                if id(u) in seen:
                    continue
                seen.add(id(u))
                out.append(u)
                if u.op in VIEW_NODE_OPS:
                    todo.append(u.output)
        return out

    def buffer(self, k):
        """Physical buffer for value k (after state aliasing)."""
        return self.value_alias.get(k, k)

    # ---- views ---------------------------------------------------------------
    def analyse_views(self):
        g = self.graph
        self.virtual = {}
        self.view_alias = {}   # needed view that is its whole base buffer -> base
        prods = self.prods
        for n, _ in self.schedule:
            if n.op not in VIEW_NODE_OPS:
                continue
            base, view = V.resolve(g, n.output, prods, stop=set(self.value_alias))
            if base == n.output:
                continue
            self.virtual[n.output] = (self.buffer(base), view)
        # Values that must exist as buffers.
        self.materialize = set()
        carry_values = set()
        last_repeat = None
        for n, rep in self.schedule:
            if rep:
                last_repeat = n
        if last_repeat is not None:
            carry_values.add(last_repeat.output)
        for k in list(self.virtual):
            base, view = self.virtual[k]
            alias = view.contiguous and view.offset == 0 and view.size == g.values[base].size if base in g.values else False
            need = (k == self.output or k in carry_values
                    or any(self.external(u) for u in self.cons.get(k, []))
                    or any(u.op == "state_write" and u.inputs[0] == k for u in self.cons.get(k, [])))
            if need and not (alias and k != self.output and k not in carry_values):
                self.materialize.add(k)
            elif need:
                self.view_alias[k] = base

    def external(self, n):
        g = self.graph
        if n.op == "embedding":
            x = g.values[n.inputs[0]]
            return x.role in ("token", "position") and x.size == 1 and bool(g.values[n.inputs[1]].weight)
        if n.op in ("linear", "linear_t"):
            x = g.values[n.inputs[0]]
            w = g.values[n.inputs[1]]
            return w.weight is not None and len(w.shape) == 2 and x.size == x.shape[-1]
        return False

    def physical(self, k):
        """Buffer holding value k (resolving identity aliases of views)."""
        if k in self.virtual and k not in self.materialize and k in self.view_alias:
            return self.view_alias[k]
        return self.buffer(k)

    # ---- emission ------------------------------------------------------------
    def build(self):
        g = self.graph
        self.namer = _Namer()
        self.names = {}
        self.kernels = {}      # body -> name
        self.kernel_texts = []
        self.kernel_info = []
        self.node_lines = []
        self.node_namer = _Namer()
        self.buf_order = []
        self.buf_info = {}
        self.scratch = 0
        self.epsilon_key, self.eps = self.pick_epsilon()
        # The runner requires both input roles even when the graph does not read them.
        self.name_of(self.roles["token"])
        self.name_of(self.roles["position"])
        # Constants that need buffers are initialized first.
        repeating = False
        body_nodes = []
        for n, repeated in self.schedule:
            if id(n) in self.skip:
                continue
            if n.op in VIEW_NODE_OPS and n.output in self.virtual and n.output not in self.materialize:
                continue
            if repeated != repeating:
                body_nodes.append("repeat layers" if repeated else "endrepeat")
                repeating = repeated
            try:
                body_nodes.extend(self.lower(n, repeated))
            except (ValueError, NotImplementedError) as error:
                self.errors.append(f"{n.scope or g.name}: {n.op} ({n.output}): {error}")
        if repeating:
            body_nodes.append("endrepeat")
        if self.errors:
            raise LoweringError("Cannot lower traced operations:\n- " + "\n- ".join(self.errors))
        if self.scratch:
            self.declare_scratch()
        init_nodes = self.node_lines
        lines = self.header()
        lines += [self.buf_info[k] for k in self.buf_order]
        text = "\n".join(lines) + "\n\n" + "".join(self.kernel_texts) + "\n" + "\n".join(init_nodes + body_nodes) + "\n"
        return text

    def pick_epsilon(self):
        g = self.graph
        eps = {n.attrs["eps"] for n, _ in self.schedule if n.op in ("layernorm", "rmsnorm", "norm")}
        if len(eps) == 1:
            value = eps.pop()
            for key in EPSILON_KEYS:
                if g.config.get(key) == value:
                    return key, value
        key = next((k for k in EPSILON_KEYS if k in g.config), "layer_norm_epsilon")
        return key, g.config[key]

    def header(self):
        g = self.graph
        lines = ["fir 1", "model " + g.name, "config config.json", "runtime graph", "layout script",
                 "epsilon " + self.epsilon_key]
        if self.prefix:
            lines.append("prefix " + self.prefix)
        for key, value in self.require_config:
            if value is None:
                raise LoweringError("require_config key is absent from config: " + key)
            text = ("true" if value else "false") if isinstance(value, bool) else (value if isinstance(value, str) else json.dumps(value))
            if not text or any(c.isspace() for c in text):
                raise LoweringError("require_config value must be one word: " + key)
            lines.append(f"require config {key} {text}")
        layer_key = next((key for key in LAYER_KEYS if g.config.get(key) == self.layers), None)
        lines += ["sym layers = " + (layer_key or str(self.layers)), "sym positions = " + position_key(g.config),
                  "sym vocab = vocab_size"]
        # Kernels are specialized to these dimensions; a different config must fail at load.
        for key, value in g.config.items():
            if type(value) is int and value > 0 and not key.endswith("_id") and key.isidentifier():
                lines.append(f"require sym {key} = {value}")
        reserved = {"layers": self.layers, "positions": g.config[position_key(g.config)], "vocab": g.config["vocab_size"]}
        for key, value in g.symbols.items():
            if key in reserved:
                if value != reserved[key]:
                    raise LoweringError(f"symbol {key}={value} conflicts with runtime value {reserved[key]}")
                continue
            if not key.isidentifier():
                raise LoweringError("symbol name must be an identifier: " + key)
            if key in g.config:
                if g.config[key] != value:
                    raise LoweringError(f"symbol {key}={value} differs from config value {g.config[key]}")
                continue
            lines.append(f"sym {key} = {value}")
        return lines

    # ---- buffers -------------------------------------------------------------
    def name_of(self, k):
        """Declare (once) and return the FIR name of buffer value ``k``."""
        if k in self.names:
            return self.names[k]
        g = self.graph
        v = g.values[k]
        if v.weight:
            raise LoweringError("weights are declared through weight_name")
        if v.role:
            name = self.namer.take([v.role])
            role = " input:" + v.role
        elif v.state:
            parts = _scope_parts(v.state)
            name = self.namer.take(_suffixes(parts, ""))
            role = ""
        elif k == self.output:
            name = self.namer.take(["logits"])
            role = " output:logits"
        else:
            p = self.prods.get(k)
            tag = "residual" if p is not None and k in p.extra else _OP_TAGS.get(p.op, p.op) if p else "value"
            name = self.namer.take(_suffixes(_scope_parts(p.scope if p else ""), tag))
            role = ""
        if k == self.output and v.role:
            raise LoweringError("decode logits must be produced by an operation")
        per_layer = " per_layer" if k in self.layer_states else ""
        self.names[k] = name
        self.buf_order.append(k)
        self.buf_info[k] = f"buf {name} f32 {v.size}{per_layer}{role}"
        return name

    def weight_name(self, k, resident):
        """Declare and return the resident (rank-2) or arena binding of weight ``k``."""
        g = self.graph
        v = g.values[k]
        w = v.weight
        source = self.templates.get(k, w.name)
        if self.prefix and source.startswith(self.prefix):
            source = source[len(self.prefix):]
        key = (k, resident or len(w.shape) != 2)
        if key in self.names:
            return self.names[key]
        parts = [p for p in source.replace("{i}", "").split(".") if p]
        if resident:
            if len(w.shape) != 2:
                raise LoweringError(f"resident weight {w.name} requires rank 2, got {w.shape}")
            name = self.namer.take(_suffixes(parts, "", 2) if len(parts) > 1 else parts)
            line = f"weight {name} {w.dtype} {source} tensor [{','.join(map(str, w.shape))}]"
        else:
            if w.dtype != "f32":
                raise LoweringError(f"VM weight {w.name} requires f32; q4 is supported only by fir.linear")
            if len(w.shape) != 2:
                name = self.namer.take(_suffixes(parts, "", 2) if len(parts) > 1 else parts)
                line = f"weight {name} f32 {source} tensor [{','.join(map(str, w.shape))}]"
            else:
                flat = source + ".fir_flat"
                names = {x.weight.name for x in g.values.values() if x.weight}
                if any(flat.replace("{i}", str(i)) in names or (self.prefix + flat).replace("{i}", str(i)) in names
                       for i in range(self.layers)):
                    raise LoweringError("flattened checkpoint name conflicts with a source parameter: " + flat)
                name = self.namer.take([s + "_flat" for s in (_suffixes(parts, "", 2) if len(parts) > 1 else parts)])
                shape = ",".join(map(str, w.shape))
                line = (f"# checkpoint-map {source} [{shape}] -> {flat} [{v.size}]\n"
                        "# offline reshape required; row-major flatten; axis permutation=identity\n"
                        f"weight {name} f32 {flat} tensor [{v.size}]")
        self.names[key] = name
        self.buf_order.append(key)
        self.buf_info[key] = line
        return name

    def declare_scratch(self):
        name = self.namer.take(["scores"])
        self.scratch_name_final = name
        self.buf_order.append("__scratch__")
        self.buf_info["__scratch__"] = f"buf {name} f32 {self.scratch}"

    # ---- operands ------------------------------------------------------------
    def operand(self, k, ctx, mode="any"):
        """Kernel operand for value ``k``: Const, Affine or Mem with a port.

        ``mode`` restricts constants: "no_affine" reads non-uniform constants
        from a buffer, "mem" reads every constant from a buffer.
        """
        g = self.graph
        v = g.values[k]
        if v.constant is not None:
            first = v.constant[0]
            if mode != "mem" and all(x == first for x in v.constant):
                return C.Const(first)
            fit = _affine(v) if mode == "any" else None
            if fit is not None:
                return C.Affine(fit[0], fit[1], v.shape)
            return C.Mem(ctx.port_in(self.constant_buffer(k)), V.identity(v.shape))
        if k in self.virtual and k not in self.materialize:
            base, view = self.virtual[k]
            return C.Mem(ctx.port_in(self.buffer_name(base)), view)
        return C.Mem(ctx.port_in(self.buffer_name(self.physical(k))), V.identity(v.shape))

    def buffer_name(self, k):
        g = self.graph
        if g.values[k].weight:
            return self.weight_name(k, False)
        return self.name_of(k)

    def constant_buffer(self, k):
        """FIR name of the buffer holding constant ``k`` (written by an init node)."""
        if k in self.names:
            return self.names[k]
        return self.name_of_const(k)

    def name_of_const(self, k):
        g = self.graph
        v = g.values[k]
        name = self.namer.take(["constant_" + k])
        self.names[k] = name
        self.buf_order.append(k)
        self.buf_info[k] = f"buf {name} f32 {v.size}"
        kb = C.Kernel()
        out = C.Mem("out0", V.identity(v.shape))
        threads = C.gen_constant(kb, v.constant, out)
        kname = self.kernel(kb, "constant", 1, 1, threads)
        self.node_lines.append(f"node {self.node_namer.take(['init_' + name])} micro {kname} "
                               f"{self.name_of(self.roles['token'])} -> {name}")
        return name

    # ---- kernels ---------------------------------------------------------------
    def kernel(self, kb, family, ins, outs, threads, size=None, out0=None):
        if out0 is not None and threads == out0 and threads > 1:
            # A size expression lets same-shaped bodies of different widths share a kernel.
            threads = "out0.size"
        body = kb.text("@", ins, outs, threads, size)
        body = body[body.index(" in="):]
        if body in self.kernels:
            return self.kernels[body]
        name = "script." + family
        taken = set(self.kernels.values())
        i = 2
        while name in taken:
            name = f"script.{family}_{i}"
            i += 1
        self.kernels[body] = name
        text = "kernel " + name + body
        if self.tags:
            text = text.replace("\n", " tag=" + kernel_tag(family) + "\n", 1)  # header line only
        self.kernel_texts.append(text)
        self.kernel_info.append((name, family, len(kb.lines)))
        return name

    def lower(self, n, repeated):
        g = self.graph
        if self.external(n):
            return [self.lower_external(n)]
        ctx = _Ports(self)
        kb = C.Kernel()
        family, threads = self.generate(n, ctx, kb)
        out0 = self.buffer_size(ctx.outs[0]) if ctx.outs else None
        kname = self.kernel(kb, family, len(ctx.ins), len(ctx.outs), threads, out0=out0)
        node_name = self.node_namer.take(_suffixes(_scope_parts(n.scope), _OP_TAGS.get(n.op, n.op)))
        return [f"node {node_name} micro {kname} {' '.join(ctx.ins)} -> {' '.join(ctx.outs)}"]

    def buffer_size(self, name):
        for key, fir_name in self.names.items():
            if fir_name == name and not isinstance(key, tuple):
                return self.graph.values[key].size
        return None

    def lower_external(self, n):
        g = self.graph
        x, w = n.inputs
        xv, wv, yv = g.values[x], g.values[w], g.values[n.output]
        if xv.weight:
            raise LoweringError("resident external requires an activation input")
        if n.op != "linear" and wv.weight.dtype != "f32":
            raise LoweringError(n.op + " requires f32 checkpoint weights")
        if n.op == "embedding":
            if wv.shape[1] != yv.size:
                raise LoweringError("embedding output width mismatch")
        elif wv.shape != ((xv.size, yv.size) if n.op == "linear" else (yv.size, xv.size)):
            raise LoweringError("resident linear checkpoint shape mismatch")
        xin = self.name_of(self.physical(x)) if not xv.role else self.name_of(x)
        wname = self.weight_name(w, True)
        out = self.output_buffer(n.output)
        node_name = self.node_namer.take(_suffixes(_scope_parts(n.scope), _OP_TAGS.get(n.op, n.op)))
        return f"node {node_name} ext fir.{n.op} {xin} {wname} -> {out}"

    def output_buffer(self, k):
        return self.name_of(self.buffer(k))

    def eps_literal(self, value):
        return "eps" if value == self.eps else C.fnum(value)

    def generate(self, n, ctx, kb):
        """Dispatch one scheduled node to a kernel generator; return (family, threads)."""
        g = self.graph
        op = n.op
        shape = g.values[n.output].shape
        out = lambda k=n.output: C.Mem(ctx.port_out(self.output_buffer(k)), V.identity(g.values[k].shape))
        if op in VIEW_NODE_OPS:
            if n.output in self.virtual:
                base, view = self.virtual[n.output]
                src = C.Mem(ctx.port_in(self.buffer_name(base)), view)
                threads = C.gen_pointwise(kb, ("input", 0), [src], [out()], shape)
                return "copy", threads
            if op == "cat":
                return "cat", self.gen_cat(n, ctx, kb, out())
            src = self.operand(n.inputs[0], ctx, "mem")
            view = V.apply(n, g, [src.view])
            if view is not None:
                threads = C.gen_pointwise(kb, ("input", 0), [C.Mem(src.port, view)], [out()], shape)
                return "copy", threads
            if op != "reshape":
                raise LoweringError("view is not strided over its input")
            # Row-major order is preserved: copy over the input space.
            ishape = g.values[n.inputs[0]].shape
            o = out()
            threads = C.gen_pointwise(kb, ("input", 0), [src], [C.Mem(o.port, V.identity(ishape))], ishape)
            return "copy", threads
        if op == "copy":
            threads = C.gen_pointwise(kb, ("input", 0), [self.operand(n.inputs[0], ctx)], [out()], shape)
            return "copy", threads
        if op in ELEMENTWISE or op == "fused":
            tree = n.attrs["tree"] if op == "fused" else (op, tuple(("input", i) for i in range(len(n.inputs))))
            operands = [self.operand(k, ctx) for k in n.inputs]
            threads = C.gen_pointwise(kb, tree, operands, [out()], shape)
            return _family(tree), threads
        if op in ("layernorm", "rmsnorm", "norm"):
            a = n.attrs
            kind = a.get("kind", op)
            ports = a.get("ports", 1)
            tree = a.get("tree", ("input", 0))
            ins = [self.operand(k, ctx) for k in n.inputs]
            has_w = a.get("weight", len(n.inputs) > 1)
            has_b = a.get("bias", len(n.inputs) > 2)
            weight = ins[ports] if has_w else None
            bias = ins[ports + 1] if has_b else None
            o = out()
            side = out(n.extra[0]) if n.extra else None
            threads = C.gen_norm(kb, kind, tree, ins[:ports], weight, bias, o, side, shape, a["width"],
                                 self.eps_literal(a["eps"]))
            family = ("add_" if ports > 1 or tree != ("input", 0) else "") + kind
            return family, threads
        if op == "softmax":
            threads = C.gen_softmax(kb, self.operand(n.inputs[0], ctx, "mem"), out(), shape, n.attrs["dim"])
            return "softmax", threads
        if op in ("sum", "mean", "max"):
            a = n.attrs
            in_shape = a.get("in_shape", g.values[n.inputs[0]].shape)
            tree = a.get("tree", ("input", 0))
            operands = [self.operand(k, ctx) for k in n.inputs]
            threads = C.gen_reduce(kb, op, tree, operands, out(), in_shape, a["dims"], a["keepdim"])
            return op, threads
        if op == "matmul":
            a, b = (self.operand(k, ctx, "no_affine") for k in n.inputs)
            threads = C.gen_matmul(kb, a, b, out(), g.values[n.inputs[0]].shape, g.values[n.inputs[1]].shape, shape)
            return "matmul", threads
        if op in ("linear", "linear_t"):
            x, w = (self.operand(k, ctx, "mem") for k in n.inputs)
            ws = g.values[n.inputs[1]].shape
            if op == "linear_t":
                w = C.Mem(w.port, V.permute(w.view, (1, 0)))
                ws = ws[::-1]
            threads = C.gen_matmul(kb, x, w, out(), g.values[n.inputs[0]].shape, ws, shape)
            return "linear", threads
        if op == "cumsum":
            threads = C.gen_cumsum(kb, self.operand(n.inputs[0], ctx, "mem"), out(), shape, n.attrs["dim"])
            return "cumsum", threads
        if op == "index_select":
            src, idx = (self.operand(k, ctx, "mem") for k in n.inputs)
            threads = C.gen_gather(kb, src, idx, out(), n.attrs["dim"])
            return "gather", threads
        if op == "embedding":
            idx, table = n.inputs
            iv = g.values[idx]
            src = self.operand(table, ctx, "mem")
            index = self.operand(idx, ctx, "mem")
            if not isinstance(index, C.Mem):
                raise LoweringError("embedding with a constant index; use a slice of the weight")
            index = C.Mem(index.port, V.reshape(index.view, (iv.size,)))
            o = out()
            o = C.Mem(o.port, V.reshape(o.view, (iv.size, g.values[table].shape[1])))
            threads = C.gen_gather(kb, src, index, o, 0)
            return "gather", threads
        if op == "conv1d":
            ins = [self.operand(k, ctx, "mem") for k in n.inputs]
            xshape = g.values[n.inputs[0]].shape
            times = n.attrs.get("times", tuple(range(xshape[2])))
            threads = C.gen_conv1d(kb, ins[0], ins[1], ins[2] if len(ins) > 2 else None, out(), xshape,
                                   g.values[n.inputs[1]].shape, shape, n.attrs["groups"], times)
            return "conv1d", threads
        if op == "rotary":
            x, p = self.operand(n.inputs[0], ctx, "mem"), self.operand(n.inputs[1], ctx, "no_affine")
            threads = C.gen_rotary(kb, x, p, out(), shape, n.attrs["base"])
            return "rotary", threads
        if op == "attention_decode":
            return "attention", self.gen_attention(n, ctx, kb)
        if op == "conv1d_step":
            return "conv_step", self.gen_conv_step(n, ctx, kb)
        if op == "state_update_reduce":
            operands = [self.operand(k, ctx) for k in n.inputs]
            storage = n.writes[0]
            sname = self.name_of(storage)
            # The state is loaded and stored through its input port (same arena
            # address); it is also listed as an output to record the write.
            operands[n.attrs["state_port"]] = C.Mem(ctx.port_in(sname), V.identity(g.values[storage].shape))
            o = out()
            ctx.port_out(sname)
            threads = C.gen_state_update_reduce(kb, n.attrs, operands, o, None, operands[n.attrs["reset_port"]])
            return "ssm_step", threads
        if op == "state_read":
            storage = n.inputs[0]
            mode = self.state_mode.get(storage)
            pos = self.operand(self.roles["position"], ctx)
            if mode == "inplace":
                sname = self.name_of(storage)
                threads = C.gen_state_reset(kb, pos, C.Mem(ctx.port_out(sname), V.identity(g.values[storage].shape)),
                                            g.values[storage].shape)
                return "state_reset", threads
            src = self.operand(storage, ctx)
            tree = ("where", (("lt", (("input", 1), ("const", 0.5))), ("const", 0.0), ("input", 0)))
            threads = C.gen_pointwise(kb, tree, [src, pos], [out()], shape)
            return "state_read", threads
        if op == "state_write":
            return self.gen_state_write(n, ctx, kb)
        if op in custom.OPS:
            return custom.OPS[op].lower(self, n, ctx, kb)
        raise LoweringError("unsupported traced operation %s; add a kernel generator in codegen.py" % op)

    def gen_state_write(self, n, ctx, kb):
        g = self.graph
        storage = n.writes[0]
        sshape = g.values[storage].shape
        mode = self.state_mode.get(storage)
        sname = self.name_of(storage)
        if mode == "inplace":
            if len(n.inputs) == 3:
                value, index = self.operand(n.inputs[1], ctx), self.operand(n.inputs[2], ctx)
                threads = C.gen_indexed_write(kb, value, index, C.Mem(ctx.port_out(sname), V.identity(sshape)),
                                              sshape, n.attrs["dim"])
                return "state_write", threads
            value = self.operand(n.inputs[1], ctx)
            threads = C.gen_pointwise(kb, ("input", 0), [value],
                                      [C.Mem(ctx.port_out(sname), V.identity(sshape))], sshape)
            return "state_copy", threads
        outs = [C.Mem(ctx.port_out(self.output_buffer(n.output)), V.identity(sshape)),
                C.Mem(ctx.port_out(sname), V.identity(sshape))]
        if len(n.inputs) == 3:
            prev, value, index = (self.operand(k, ctx) for k in n.inputs)
            dim = n.attrs["dim"]
            coefs = [1 if i == dim else 0 for i in range(len(sshape))]
            iota = C.Affine(0, coefs, sshape)
            if isinstance(value, C.Mem):
                vdims = list(value.view.dims)
                vdims.insert(dim, (sshape[dim], 0, 1))
                value = C.Mem(value.port, V.View(value.view.offset, tuple(vdims)))
            tree = ("where", (("eq", (("input", 3), ("input", 2))), ("input", 1), ("input", 0)))
            threads = C.gen_pointwise(kb, tree, [prev, value, index, iota], outs, sshape)
            return "state_select", threads
        value = self.operand(n.inputs[1], ctx)
        threads = C.gen_pointwise(kb, ("input", 0), [value], outs, sshape)
        return "state_copy", threads

    def gen_cat(self, n, ctx, kb, o):
        """Irregular concatenation: one serial copy loop nest per part (single thread)."""
        g = self.graph
        dim = n.attrs["dim"]
        shape = g.values[n.output].shape
        start = 0
        for k in n.inputs:
            part = self.operand(k, ctx)
            pshape = g.values[k].shape
            spec = tuple((start, start + pshape[d], 1) if d == dim else (0, s, 1) for d, s in enumerate(shape))
            target = C.Mem(o.port, V.slice_view(o.view, spec))
            start += pshape[dim]
            ai, ao = C.access(part, pshape), C.access(target, pshape)
            live = [a for a in (ai, ao) if a is not None]
            sizes = C.plan(pshape, live, "outer")

            def body(r, part=part, ai=ai, ao=ao):
                v = kb.load(ai, r) if ai is not None else kb.const(part.value)
                kb.store(ao, r, v)
                kb.free_r(v)
            kb.nest(sizes, list(range(len(sizes))), live, {}, body)
        return 1

    def gen_attention(self, n, ctx, kb):
        g = self.graph
        a = n.attrs
        q, knew, vnew, position, kstore, vstore = n.inputs
        need = a["heads"] * a["positions"] + (a["heads"] * a["head_dim"] if (a.get("rope") or {}).get("q") else 0)
        self.scratch = max(self.scratch, need)
        ports = {"q": ctx.port_in(self.buffer_name(self.physical(q))),
                 "k_new": ctx.port_in(self.buffer_name(self.physical(knew))),
                 "v_new": ctx.port_in(self.buffer_name(self.physical(vnew))),
                 "position": ctx.port_in(self.name_of(position)),
                 "k_cache": ctx.port_in(self.name_of(kstore)),
                 "v_cache": ctx.port_in(self.name_of(vstore)),
                 "scratch": ctx.port_in(ctx.scratch())}
        ports["out"] = ctx.port_out(self.output_buffer(n.output))
        ctx.port_out(self.name_of(kstore))
        ctx.port_out(self.name_of(vstore))
        ctx.port_out(ctx.scratch())
        for key in ("q", "k_new", "v_new"):
            src = {"q": q, "k_new": knew, "v_new": vnew}[key]
            if src in self.virtual and src not in self.materialize:
                raise LoweringError("attention operand must be a buffer")
        return C.gen_attention(kb, a, ports)

    def gen_conv_step(self, n, ctx, kb):
        g = self.graph
        a = n.attrs
        Cn, K = a["channels"], a["kernel_size"]
        x = self.operand(n.inputs[0], ctx, "mem")
        w = self.operand(n.inputs[1], ctx, "mem")
        bias = self.operand(n.inputs[2], ctx, "mem") if a["bias"] else None
        storage = n.writes[0]
        sname = self.name_of(storage)
        views = {"x": V.reshape(x.view, (Cn,)), "w": V.reshape(w.view, (Cn, K)),
                 "state": V.identity((Cn, K - 1)), "out": V.identity((Cn,))}
        if bias is not None:
            views["bias"] = V.reshape(bias.view, (Cn,))
        if any(v is None for v in views.values()):
            raise LoweringError("conv step operands are not strided")
        ports = {"x": x.port, "w": w.port, "position": ctx.port_in(self.name_of(self.roles["position"])),
                 "state": ctx.port_in(sname)}
        if bias is not None:
            ports["bias"] = bias.port
        ports["out"] = ctx.port_out(self.output_buffer(n.output))
        ctx.port_out(sname)
        return C.gen_conv_step(kb, a, ports, views)


def _family(tree):
    ops = []

    def walk(t):
        if t[0] in ("input", "const", "state", "updated"):
            return
        ops.append(t[0])
        for c in t[1]:
            walk(c)
    walk(tree)
    if not ops:
        return "copy"
    seen = []
    for o in reversed(ops):
        if o not in seen:
            seen.append(o)
    return "_".join(seen[:3])


class _Ports:
    """Assigns in/out port names for one kernel node."""

    def __init__(self, emission):
        self.e = emission
        self.ins, self.outs = [], []

    def port_in(self, name):
        if name not in self.ins:
            if len(self.ins) >= 10:
                raise LoweringError("kernel needs more than 10 input buffers; split the operation")
            self.ins.append(name)
        return "in" + str(self.ins.index(name))

    def port_out(self, name):
        if name not in self.outs:
            if len(self.outs) >= 10:
                raise LoweringError("kernel needs more than 10 output buffers; split the operation")
            self.outs.append(name)
        return "out" + str(self.outs.index(name))

    def scratch(self):
        return "__scratch__"


def emit(graph, *, prefix=None, require_config=(), tags=False):
    """Serialize a decode specialization; raise LoweringError before writing anything.

    ``tags=True`` adds a `tag=` op identity to every kernel header (script layout only)."""
    e = _Emission(graph, prefix, require_config, tags)
    text = e.run()
    if e.scratch:
        text = text.replace("__scratch__", e.scratch_name_final)
    return text


def export(graph, directory, *, example_inputs=None, config=None, name=None, optimize=None,
           prefix=None, require_config=(), report=False, tags=False):
    """Write validated FIR text and metadata config, without checkpoint tensors.

    ``graph`` is a traced Graph or a Module/callable (then ``example_inputs`` is
    required and tracing uses ``config``/``name``). ``optimize=True`` applies the
    optimization passes to a copy of an unoptimized graph; ``optimize=False``
    requires an unoptimized graph and emits it with generic kernels only;
    ``None`` emits the graph as given. ``prefix`` is stripped from checkpoint
    names and written as a ``prefix`` line (the runner tries the plain name,
    then the prefixed one). ``require_config`` is a list of config keys or a
    mapping key -> value, written as ``require config`` lines.
    Returns the path of the written model.fir.
    """
    import copy
    from .graph import Graph, optimize_graph
    if not isinstance(graph, Graph):
        from .tracer import trace
        if example_inputs is None:
            raise TypeError("export of a module needs example_inputs")
        graph = trace(graph, example_inputs, config=config, name=name or "scripted",
                      optimize=optimize is not False)
    elif optimize and not graph.passes:
        graph = optimize_graph(copy.deepcopy(graph))
    elif optimize is False and graph.passes:
        raise ValueError("export(optimize=False) needs an unoptimized graph; trace with optimize=False")
    text = emit(graph, prefix=prefix, require_config=require_config, tags=tags)
    config_text = json.dumps(graph.config, indent=2, allow_nan=False) + "\n"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(config_text, encoding="utf-8")
    path = directory / "model.fir"
    path.write_text(text, encoding="utf-8")
    if report:
        from .report import report as make_report
        print(make_report(text, graph))
    return path
