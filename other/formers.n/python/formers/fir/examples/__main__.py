"""python3 -m formers.fir.examples OUTDIR [--report]

Exports the tiny GPT-2, Llama and Mamba examples (optimized) into OUTDIR/<name>.
"""

import sys
from pathlib import Path
from .. import trace, export
from ..report import report
from . import gpt2, llama, mamba

args = [a for a in sys.argv[1:] if a != "--report"]
if len(args) != 1:
    raise SystemExit("usage: python3 -m formers.fir.examples OUTDIR [--report]")
for name, source, model in (("gpt2", gpt2, gpt2.GPT2), ("llama", llama, llama.Llama), ("mamba", mamba, mamba.Mamba)):
    graph = trace(model(), source.INPUTS, config=source.CONFIG, name="tiny_" + name)
    path = export(graph, Path(args[0]) / name)
    print(path)
    if "--report" in sys.argv:
        print(report(path.read_text(encoding="utf-8"), graph))
