"""Which ATen ops of a model convert, and how."""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

from . import aten


@dataclass
class CoverageReport:
    ops: dict = field(default_factory=dict)        # "aten::name.overload" -> count
    mapped: dict = field(default_factory=dict)     # explicit handler
    micro: dict = field(default_factory=dict)      # micro-kernel
    generic: dict = field(default_factory=dict)    # resolved by the same-named torch function
    unsupported: dict = field(default_factory=dict)

    @property
    def ok(self):
        return not self.unsupported

    def __str__(self):
        lines = ["ATen ops: %d distinct, %d nodes" % (len(self.ops), sum(self.ops.values()))]
        for title, table in (("mapped", self.mapped), ("micro-kernel", self.micro),
                             ("generic (same-named torch function)", self.generic),
                             ("UNSUPPORTED", self.unsupported)):
            if table:
                lines.append("%s: %s" % (title, ", ".join("%s x%d" % (k, v) for k, v in sorted(table.items()))))
        return "\n".join(lines)


def coverage(module_or_program, example_inputs=None, **options):
    """Count the Core ATen ops of a module (or ExportedProgram) and say how each one converts."""
    from .lower import capture
    program = module_or_program if hasattr(module_or_program, "graph_signature") else \
        capture(module_or_program, example_inputs, **options)
    report = CoverageReport()
    counts = collections.Counter()
    for node in program.graph.nodes:
        if node.op != "call_function" or not hasattr(node.target, "_schema"):
            continue
        name, overload = aten.op_name(node.target)
        key = "%s.%s" % (name, overload)
        counts[key] += 1
        if aten.MICRO.get(name):
            table = report.micro
        elif aten.HANDLERS.get(key) or aten.HANDLERS.get(name):
            table = report.mapped
        elif any(callable(getattr(space, name, None)) for space in (aten.T, aten.F)):
            table = report.generic
        else:
            table = report.unsupported
        table[key] = table.get(key, 0) + 1
    report.ops = dict(counts)
    return report
