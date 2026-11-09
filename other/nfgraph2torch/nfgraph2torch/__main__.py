"""Command line: ``python -m nfgraph2torch MODULE:FACTORY OUT_DIR``.

``MODULE:FACTORY`` names a Python callable (``package.module:make_model``) that returns
``(torch_module, example_inputs)``. The converted ``model.fir`` and ``config.json`` are written to
``OUT_DIR``. ``--coverage`` prints which ATen ops the module needs and how each one is lowered.
"""
import argparse
import importlib
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nfgraph2torch", description=__doc__.split("\n\n")[0])
    ap.add_argument("factory", help="module:callable returning (module, example_inputs)")
    ap.add_argument("out", nargs="?", help="output directory for model.fir")
    ap.add_argument("--coverage", action="store_true", help="report ATen op coverage and exit")
    args = ap.parse_args(argv)
    mod, _, name = args.factory.partition(":")
    if not name:
        ap.error("factory must look like package.module:callable")
    sys.path.insert(0, ".")
    module, example = getattr(importlib.import_module(mod), name)()
    import nfgraph2torch as nf2t
    if args.coverage:
        print(nf2t.coverage(module, example))
        return 0
    if not args.out:
        ap.error("OUT_DIR is required unless --coverage is given")
    print(nf2t.export(module, example, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
