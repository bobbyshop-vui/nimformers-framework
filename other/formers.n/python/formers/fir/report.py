"""Static statistics of FIR files and traced graphs (``--report``)."""

import json
import re
from collections import Counter
from pathlib import Path


def fir_stats(text, config=None):
    """Counts of buffers, weights, nodes and per-kernel instructions of FIR text."""
    from .vm import parse
    if config is None:
        config = {}
    prog = parse(text, config)
    layers = prog["symbols"]["layers"]
    buffers = prog["buffers"]
    weights = [b for b in buffers.values() if b["kind"] == "weight"]
    activations = [b for b in buffers.values() if b["kind"] != "weight"]
    arena = sum(b["elements"] * (layers if b["per_layer"] else 1) for b in activations)
    kernels = {}
    for name, k in prog["kernels"].items():
        depth = best = loops = conditions = 0
        for mnemonic, _, _ in k["body"]:
            if mnemonic == "forbegin":
                depth += 1
                loops += 1
                best = max(best, depth)
            elif mnemonic == "forend":
                depth -= 1
            elif mnemonic == "ifbegin":
                conditions += 1
        kernels[name] = {"instructions": len(k["body"]), "loops": loops, "max_loop_depth": best,
                         "conditions": conditions, "in": k["ins"], "out": k["outs"], "threads": k["threads"],
                         "uses": sum(n["op"] == name for n in prog["nodes"])}
    nodes = prog["nodes"]
    per_token = Counter()
    for n in nodes:
        per_token[n["kind"]] += layers if n["per_layer"] else 1
    return {"model": prog["model"], "buffers": len(activations), "weights": len(weights),
            "declarations": len(buffers), "arena_floats": arena, "nodes": len(nodes),
            "micro_nodes": sum(n["kind"] == "micro" for n in nodes),
            "ext_nodes": sum(n["kind"] == "ext" for n in nodes),
            "repeat_nodes": sum(n["per_layer"] for n in nodes),
            "dispatches_per_token": dict(per_token), "kernels": kernels,
            "kernel_instructions": sum(k["instructions"] for k in kernels.values()),
            "lines": len(text.splitlines())}


def graph_stats(graph):
    ops = Counter(n.op for n in graph.nodes)
    return {"graph_nodes": len(graph.nodes), "graph_ops": dict(sorted(ops.items())), "passes": list(graph.passes)}


def report(text, graph=None, config=None):
    """Human-readable report of FIR text (and optionally its source graph)."""
    if config is None and graph is not None:
        config = graph.config
    s = fir_stats(text, config)
    lines = [f"FIR {s['model']}: {s['lines']} lines, {s['declarations']} declarations "
             f"({s['buffers']} buffers, {s['weights']} weights), arena {s['arena_floats']} floats",
             f"nodes {s['nodes']} ({s['micro_nodes']} micro, {s['ext_nodes']} ext, {s['repeat_nodes']} in repeat); "
             f"per token: " + ", ".join(f"{v} {k}" for k, v in sorted(s["dispatches_per_token"].items())),
             f"kernels {len(s['kernels'])}, {s['kernel_instructions']} instructions"]
    width = max([len(k) for k in s["kernels"]] + [6])
    lines.append(f"  {'kernel':<{width}}  instr  loops  depth  in  out  uses  threads")
    for name, k in s["kernels"].items():
        lines.append(f"  {name:<{width}}  {k['instructions']:>5}  {k['loops']:>5}  {k['max_loop_depth']:>5}  "
                     f"{k['in']:>2}  {k['out']:>3}  {k['uses']:>4}  {k['threads']}")
    if graph is not None:
        g = graph_stats(graph)
        lines.append(f"graph {g['graph_nodes']} nodes: " + ", ".join(f"{op} {c}" for op, c in g["graph_ops"].items()))
    return "\n".join(lines)


def main(argv=None):
    """python3 -m formers.fir.report MODEL.fir [config.json] [--json]"""
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if not args:
        raise SystemExit("usage: python3 -m formers.fir.report MODEL.fir [config.json] [--json]")
    path = Path(args[0])
    text = path.read_text(encoding="utf-8")
    if len(args) > 1:
        config_path = Path(args[1])
    else:
        m = re.search(r"^config\s+(\S+)", text, re.M)
        config_path = path.parent / (m.group(1) if m else "config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(json.dumps(fir_stats(text, config), indent=2) if as_json else report(text, config=config))


if __name__ == "__main__":
    main()
