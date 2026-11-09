"""nfgraph2torch - convert between torch graphs and nimformers FIR.

Direction 1 (this release): ``torch graph -> nimformers``.

    import nfgraph2torch as nf2t

    graph = nf2t.from_torch(model, (example_input,))       # a formers.fir graph
    nf2t.export(model, (example_input,), "out_dir")         # out_dir/model.fir + config.json

A torch module is captured with ``torch.export``, decomposed to Core ATen, and replayed op by op on
the ``formers.fir`` tracer. ATen operations that FIR already has become ordinary FIR kernels;
operations FIR lacks are lowered to micro-kernels registered by ``nfgraph2torch.micro``.
"""
import os
import sys


def _locate_formers():
    """Put the ``formers`` package on ``sys.path`` (it lives in other/formers.n/python)."""
    try:
        import formers.fir  # noqa: F401
        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.environ.get("NFGRAPH2TORCH_FORMERS", ""),
                  os.path.join(here, "..", "..", "formers.n", "python")]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "formers")):
            sys.path.insert(0, os.path.abspath(c))
            return
    raise ImportError("cannot find the formers package; set NFGRAPH2TORCH_FORMERS to "
                      "other/formers.n/python")


_locate_formers()

from . import micro, compose, window  # noqa: E402  (registers the micro-kernels and compositions)
from .lower import from_torch, export, evaluate, Input, UnsupportedOp  # noqa: E402
from .coverage import coverage, CoverageReport  # noqa: E402

__all__ = ["from_torch", "export", "evaluate", "Input", "UnsupportedOp", "coverage", "CoverageReport", "micro"]
