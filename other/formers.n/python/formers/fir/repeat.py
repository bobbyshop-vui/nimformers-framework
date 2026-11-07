"""Conservative structural hashing of homogeneous, single-carry module stacks."""

import hashlib
import re


def _state_template(name, prefix, base, layer):
    """Layer-independent state name: the layer prefix, or else the ``.<layer>.`` segments.

    States owned by a module outside the layer (an HF ``StaticCache`` holds
    ``past_key_values.layers.<i>.keys``) are matched through their index segment;
    a name whose template differs between layers keeps the stack unrolled.
    """
    if prefix in name: return name.replace(prefix, base+".{i}.")
    return re.sub(r"(?:(?<=\.)|^)%d(?=\.|$)" % layer, "{i}", name)


def extract_repeats(graph):
    """Return schedule, aliases, weight templates, per-layer states and repeat count.

    Only one contiguous stack with one activation carry is representable by FIR v1.
    Other dependencies must be identical immutable weights/constants or input roles.
    """
    groups = {}
    for i, n in enumerate(graph.nodes):
        m = re.match(r"^(.*?)\.(\d+)(?:\.|$)", n.scope)
        if m:
            base, layer = m.group(1), int(m.group(2))
            groups.setdefault(base, {}).setdefault(layer, []).append(i)
    candidates = []
    for base, layers in groups.items():
        if len(layers) < 2 or sorted(layers) != list(range(len(layers))): continue
        segments = [layers[i] for i in range(len(layers))]
        if any(xs != list(range(xs[0], xs[-1]+1)) for xs in segments): continue
        if any(a[-1]+1 != b[0] for a, b in zip(segments, segments[1:])): continue
        signatures, externals = [], []
        for layer, indexes in enumerate(segments):
            local, external, sig = {}, [], []
            prefix = base+"."+str(layer)+"."
            def key(k):
                if k in local: return ("local", local[k])
                v = graph.values[k]
                if v.weight: return ("weight", v.weight.name.replace(prefix, base+".{i}."), v.weight.shape, v.weight.dtype)
                if v.state: return ("state", _state_template(v.state, prefix, base, layer), v.shape)
                if v.constant is not None: return ("const", v.shape, v.constant)
                if v.role: return ("role", k, v.shape)
                if k not in external: external.append(k)
                return ("carry", external.index(k), v.shape)
            for j in indexes:
                n = graph.nodes[j]
                sig.append((n.op, tuple(key(k) for k in n.inputs),
                            tuple(graph.values[k].shape for k in n.outputs),
                            repr(sorted(n.attrs.items())), tuple(key(k) for k in n.writes)))
                for k in n.outputs:
                    local[k] = len(local)
            signatures.append(hashlib.sha256(repr(sig).encode()).digest())
            externals.append(external)
        if len(set(signatures)) != 1 or any(len(x) != 1 for x in externals): continue
        # Each block must pass its last output directly to the following block.
        if any(externals[i+1][0] != graph.nodes[segments[i][-1]].output for i in range(len(segments)-1)): continue
        # No intermediate may escape the repeat, and the carry may only enter as input.
        all_indexes = {j for xs in segments for j in xs}
        outputs = {k for j in all_indexes for k in graph.nodes[j].outputs}
        last = graph.nodes[segments[-1][-1]].output
        outside_uses = {k for j, n in enumerate(graph.nodes) if j not in all_indexes
                        for k in n.inputs + n.writes}
        # A later layer's checkpoint/state cannot be redirected to layer zero
        # when an operation outside the repeat still needs its concrete identity.
        later_storage = {k for xs in segments[1:] for j in xs for k in graph.nodes[j].inputs + graph.nodes[j].writes
                         if graph.values[k].weight or graph.values[k].state}
        if later_storage & outside_uses: continue
        carry = externals[0][0]
        after_uses = {k for n in graph.nodes[segments[-1][-1]+1:] for k in n.inputs}
        # The in-place repeat carry must not overwrite a pre-stack residual that
        # is still needed after the stack or returned as the graph output.
        if carry in after_uses or graph.is_output(carry): continue
        if (outputs & (outside_uses | set(graph.output_names()))) - {last}: continue
        candidates.append((base, segments, externals[0][0]))
    if not candidates: return [(n, False) for n in graph.nodes], {}, {}, set(), 1
    if len(candidates) > 1:
        raise ValueError("FIR supports one repeat stack; combine homogeneous stacks or export separately")
    base, segments, carry = candidates[0]
    aliases, templates, states = {}, {}, set()
    first = segments[0]
    for layer, indexes in enumerate(segments):
        for j0, j in zip(first, indexes):
            a, b = graph.nodes[j0], graph.nodes[j]
            for k0, k in zip(a.outputs, b.outputs):
                aliases[k] = k0
            for k0, k in zip(a.inputs + a.writes, b.inputs + b.writes):
                v = graph.values[k]
                if v.weight:
                    aliases[k] = k0
                    templates[k0] = graph.values[k0].weight.name.replace(base+".0.", base+".{i}.")
                elif v.state:
                    aliases[k] = k0
                    states.add(k0)
    aliases[graph.nodes[first[-1]].output] = carry
    # Preserve every internal use of the carried input; store the result only at block end.
    # The last node's output overwrites carry after all earlier nodes have finished.
    omitted = {j for xs in segments[1:] for j in xs}
    schedule = [(n, j in first) for j, n in enumerate(graph.nodes) if j not in omitted]
    aliases = {k: v for k, v in aliases.items() if k != v}
    return schedule, aliases, templates, states, len(segments)
