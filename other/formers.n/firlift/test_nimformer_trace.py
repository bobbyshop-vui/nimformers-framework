#!/usr/bin/env python3
"""Check automatic unit selection and lowering of a core-only trace.

Usage: test_nimformer_trace.py TRACE [FIRLIFT_BINARY]
No runtime is launched. Missing raw byte extents or upload identities are errors,
never ignored dispatches. The generic lifter selects a repeating session unit.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    firlift = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "firlift")
    proc = subprocess.run(
        [firlift, "--trace", sys.argv[1], "--config", os.path.join(HERE, "coverage_config.json"),
         "--out", os.path.dirname(os.path.abspath(sys.argv[1])), "--coverage"],
        capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode:
        sys.exit(proc.returncode)
    print("PASS selected core trace unit lowers without ignored raw dispatches")


if __name__ == "__main__":
    main()
