#!/usr/bin/env python3
"""Lift one training step of test/spectron/test_nimformer.nim from a recorded trace.

The trace is recorded without step markers:
    BYBY_FIR_PROBE=/path/nf.fprobe test_nimformer
Usage: test_nimformer_trace.py TRACE [FIRLIFT_BINARY]

The recording holds many training steps. One step contains exactly one `ce_<n>` (cross-entropy) launch, so the test keeps the
header, the events up to the second one, and the `words` section, and runs
`firlift --coverage` on that slice. It checks that every session op in the slice lowers to a
generated micro-kernel and reports how many raw micro-VM launches (disp lines) remain unlifted.
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EVENT_PREFIXES = ("op ", "disp ", "upload ", "copy ", "note ", "barrier ")


def slice_one_step(lines):
    words = next(i for i, l in enumerate(lines) if l.startswith("words "))
    ce = [i for i, l in enumerate(lines) if l.startswith("disp ") and re.search(r"_ce_[0-9]", l)]
    if len(ce) < 2:
        sys.exit("FAIL trace holds fewer than two training steps (need two `ce_<n>` launches)")
    return lines[:ce[1]] + lines[words:]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    trace = sys.argv[1]
    firlift = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "firlift")
    with open(trace) as f:
        lines = f.read().split("\n")
    step = slice_one_step(lines)
    raw = [l for l in step if l.startswith("disp ")]
    kernels = {}
    for l in raw:
        m = re.search(r'pipe="[^"]*?_([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)"', l)
        name = re.sub(r"^jit_[0-9A-F]+_", "", re.search(r'pipe="([^"]*)"', l).group(1)) if m else "?"
        kernels[name] = kernels.get(name, 0) + 1
    with tempfile.TemporaryDirectory() as tmp:
        sliced = os.path.join(tmp, "step.fprobe")
        with open(sliced, "w") as f:
            f.write("\n".join(step))
        proc = subprocess.run(
            [firlift, "--trace", sliced, "--config", os.path.join(HERE, "coverage_config.json"),
             "--out", os.path.join(tmp, "out"), "--coverage"],
            capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit("FAIL firlift --coverage rejected the session ops of one training step")
    print("raw micro-VM launches not lifted (arena-only trace, no ports):", len(raw))
    for name, n in sorted(kernels.items(), key=lambda kv: -kv[1]):
        print("  %5d  %s" % (n, name))
    print("PASS session ops of one nimformer training step lower to micro-kernels")


if __name__ == "__main__":
    main()
