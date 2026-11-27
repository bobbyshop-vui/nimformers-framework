"""Explorer benchmark on classic OEIS entries: give the first terms, predict the last stored one.

Run `make -C test/math_spectronnexus bench-oeis` after `make oeis-data`. A direction is "right"
when its far term matches the stored OEIS term modulo 1e9+7, and "wrong" when it does not.
"""
import json
import os
import subprocess
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
BIN = ROOT + "/nimcache/math_spectronnexus"
P = 1_000_000_007

# (A-number, offset, expectation): "f" = has structure in reach, "m" = multiplicative, "n" = no structure expected.
CASES = [
    ("A000045", 0, "f"), ("A000032", 0, "f"), ("A000129", 0, "f"), ("A001045", 0, "f"), ("A000079", 0, "f"),
    ("A000244", 0, "f"), ("A000217", 0, "f"), ("A000290", 0, "f"), ("A000578", 0, "f"), ("A000292", 0, "f"),
    ("A000124", 0, "f"), ("A002620", 0, "f"), ("A004526", 0, "f"), ("A000142", 0, "f"), ("A001147", 0, "f"),
    ("A000165", 0, "f"), ("A000166", 0, "f"), ("A000085", 0, "f"), ("A000108", 0, "f"), ("A001006", 0, "f"),
    ("A000984", 0, "f"), ("A001003", 0, "f"), ("A006318", 0, "f"), ("A000172", 0, "f"), ("A005258", 0, "f"),
    ("A005259", 0, "f"), ("A002893", 0, "f"), ("A000262", 0, "f"), ("A000110", 0, "f"), ("A000111", 0, "f"),
    ("A000081", 1, "f"), ("A000041", 0, "f"), ("A000009", 0, "f"), ("A000219", 0, "f"), ("A000670", 0, "f"),
    ("A000058", 0, "f"), ("A003095", 0, "f"), ("A000120", 0, "f"), ("A002487", 0, "f"), ("A001511", 1, "f"),
    ("A007814", 1, "f"), ("A006046", 0, "f"), ("A006257", 0, "f"), ("A001316", 0, "f"), ("A000123", 0, "f"),
    ("A005187", 0, "f"), ("A005130", 0, "f"), ("A000255", 0, "f"), ("A000073", 0, "f"), ("A000930", 0, "f"),
    ("A001333", 0, "f"), ("A002426", 0, "f"), ("A001850", 0, "f"), ("A005043", 0, "f"), ("A001764", 0, "f"),
    ("A002212", 0, "f"), ("A000005", 1, "m"), ("A000010", 1, "m"), ("A000203", 1, "m"), ("A008683", 1, "m"),
    ("A000040", 1, "n"), ("A000796", 1, "n"), ("A001113", 1, "n"), ("A000088", 0, "n"), ("A000001", 0, "n"),
    ("A000112", 0, "n"), ("A000798", 0, "n"), ("A000793", 0, "n"), ("A002113", 0, "n"), ("A000002", 1, "n"),
]

stored = {}
with open(ROOT + "/data/oeis/stripped") as f:
    wanted = {c[0] for c in CASES}
    for line in f:
        if line[:7] in wanted:
            stored[line[:7]] = [int(x) for x in line[8:].strip().strip(",").split(",")]

tally = {"right": 0, "wrong": 0, "no_target": 0, "missed": 0, "false_positive": 0, "quiet": 0}
for anumber, offset, kind in CASES:
    terms = stored[anumber]
    given = min(30, len(terms) - 4)
    if given < 8:
        given = len(terms) - 1
    target = offset + len(terms) - 1
    problem = {"terms": terms[:given], "offset": offset, "max_directions": 1,
               "target": {"index": target, "modulus": P}}
    run = subprocess.run([BIN, "explore", "--backend=metal"], input=json.dumps(problem),
                         capture_output=True, text=True)
    if not run.stdout.strip():
        tally.setdefault("error", 0)
        tally["error"] += 1
        print(f"{anumber} {kind} given={given} error: {run.stderr.strip()[-160:]}")
        continue
    reply = json.loads(run.stdout)
    directions = reply["directions"]
    found = [d for d in directions if d["hypothesis"] not in ("periodic_residues", "multiplicative")] or \
            [d for d in directions if d["hypothesis"] == "multiplicative"]
    label = found[0]["sequence"] + ": " + found[0]["hypothesis"] if found else "-"
    value = reply.get("target", {}).get("value")
    expected = terms[-1] % P
    if kind == "n":
        outcome = "false_positive" if found and value is not None and value != expected else (
            "quiet" if not found else "right" if value == expected else "no_target")
    elif not found:
        outcome = "missed"
    elif value is None:
        outcome = "no_target"
    else:
        outcome = "right" if value == expected else "wrong"
    tally[outcome] += 1
    print(f"{anumber} {kind} given={given} target={target} {outcome:14} {label}")
print(json.dumps(tally))
