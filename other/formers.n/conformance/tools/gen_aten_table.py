"""Build the full ATen op table from PyTorch's native_functions.yaml and cross-check it with formers.fir.

Usage (system python3 is enough; no torch needed):
    PYTHONPATH=other/formers.n/python python3 other/formers.n/conformance/tools/gen_aten_table.py \
        --yaml /tmp/pytorch-src/aten/src/ATen/native/native_functions.yaml \
        --out other/formers.n/conformance/aten_ops.json \
        --doc docs/fir-script-aten-coverage.md

Every entry of the yaml (one `- func:` per overload) is classified:
  covered_by_functional  out=/inplace variants: the traced graph is functional, so the base op decides
  out_of_scope           training/backward, optimizer, RNG, sparse, quantized, backend specific (with a reason)
  traced                 a frontend entry (torch namespace, aten table, F namespace or Tensor method) exists
  missing                in scope for inference but no frontend entry: the work list
The table is data, not a claim of numerical equivalence; numerical checks live in conformance tests.
"""

import argparse
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

FUNC_RE = re.compile(r"^- func:\s*([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_]+))?\((.*)\)\s*->\s*(.*)$")

# Reasons are matched in this order; the first match wins.
OUT_OF_SCOPE_RULES = [
    ("training", re.compile(r"(^|_)(backward|double_backward)(_|$)|_backward$|^_.*_backward|^batch_norm_backward")),
    ("optimizer", re.compile(r"^_?(fused_)?(adam|adamw|sgd|lamb|adagrad|rmsprop)|^_foreach_|^_amp_|^_fused_")),
    ("rng", re.compile(r"^(rand|randn|randint|randperm|bernoulli|multinomial|normal|poisson|random|uniform|"
                        r"exponential|geometric|cauchy|log_normal|rrelu_with_noise|dropout|alpha_dropout|"
                        r"feature_dropout|feature_alpha_dropout|_fused_dropout|native_dropout|fractional_max_pool|"
                        r"rand_like|randn_like|randint_like|_standard_gamma|_sample_dirichlet|binomial)")),
    ("distributed", re.compile(r"^(_?c10d|all_reduce|broadcast_coalesced|_nccl)")),
    ("sparse", re.compile(r"sparse|^_?to_sparse|coalesce|_coo_|_csr_|_csc_|_bsr_|_bsc_|^indices$|^values$|^crow_|^col_indices|^ccol_|^row_indices|dense_dim")),
    ("quantized", re.compile(r"^(quantize|dequantize|q_|fake_quantize|_fake_quantize|int_repr|_make_per|choose_qparams|fbgemm|quantized_|_saturate_weight_to_fp16|_empty_affine_quantized|_empty_per_channel)")),
    ("backend_specific", re.compile(r"^(cudnn|miopen|mkldnn|mps|_mps|_cudnn|_cufft|cufft|_nested|nested|xnnpack|_xnnpack|vulkan|_vulkan|slow_conv|thnn|_slow_conv|_cslt|_triton|_pin|pin_memory|is_pinned|_cuda|cuda|_sparse_semi|xla|_test_|_debug|_assert_|_print|_validate|_check|_fw_primal|_make_dual|_unpack_dual|_new_zeros_with_same_feature_meta|_has_same_storage|_version|record_stream|_cufft)")),
    ("meta_or_system", re.compile(r"^(retain_grad|requires_grad_|is_leaf|_backward|set_data|data|output_nr|_rowwise_prune|resize|set_|storage_offset|_storage_|is_set_to|is_same_size|_debug_has_internal_overlap|_use_cudnn|_has_compatible_shallow_copy_type|_fused_moving_avg_obs|lift|lift_fresh|_lazy|_to_cpu|_functional_|_to_copy_out)")),
    ("complex_or_fft", re.compile(r"^(fft_|_fft_|view_as_complex|view_as_real|polar|complex$|real$|imag$|angle$|conj|resolve_conj|resolve_neg|is_complex|_conj|stft|istft)")),
]


def parse_yaml(path):
    entries = []
    current = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.startswith("- func:"):
            m = FUNC_RE.match(raw)
            if not m:
                raise SystemExit("cannot parse: " + raw)
            current = OrderedDict(name=m.group(1), overload=m.group(2) or "", args=m.group(3).strip(),
                                  returns=m.group(4).strip(), tags=[], variants=[], python_module="",
                                  structured=False, has_cpu_dispatch=False)
            entries.append(current)
        elif current is not None and raw.startswith("  "):
            line = raw.strip()
            if line.startswith("tags:"):
                current["tags"] = [t.strip() for t in line[5:].split(",") if t.strip()]
            elif line.startswith("variants:"):
                current["variants"] = [t.strip() for t in line[9:].split(",") if t.strip()]
            elif line.startswith("python_module:"):
                current["python_module"] = line.split(":", 1)[1].strip()
            elif line.startswith("structured_delegate:") or line.startswith("structured:"):
                current["structured"] = True
            elif re.match(r"^(CPU|CompositeExplicitAutograd|CompositeImplicitAutograd)\b", line):
                current["has_cpu_dispatch"] = True
    return entries


def base_name(name):
    """add_ -> add; add -> add."""
    return name[:-1] if name.endswith("_") and not name.endswith("__") else name


def frontend_resolvers():
    """Return a function name -> (where) | None using only formers.fir (the same lookup torch_adapter uses)."""
    from formers.fir import torch_compat, torch_adapter, tensor
    aten = torch_adapter._aten_table(torch_compat)
    F = getattr(torch_compat.nn, "functional", None)
    proxy = getattr(tensor, "ProxyTensor", None)
    modules = [("aten_table", aten), ("torch", torch_compat), ("F", F)]
    nn = getattr(torch_compat, "nn", None)

    def resolve(name):
        for label, mod in modules:
            if mod is None:
                continue
            if isinstance(mod, dict):
                if name in mod:
                    return label
            elif hasattr(mod, name):
                return label
        if proxy is not None and hasattr(proxy, name):
            return "Tensor"
        return None
    return resolve


def classify(entry, resolve, coverage):
    name, overload = entry["name"], entry["overload"]
    base = base_name(name)
    if overload == "out" or overload.endswith("_out") or overload.endswith("out"):
        if overload in ("out",) or overload.endswith("_out"):
            return "covered_by_functional", "out variant of " + base
    if name.endswith("_") and not name.startswith("__"):
        return "covered_by_functional", "in-place variant of " + base
    for reason, rx in OUT_OF_SCOPE_RULES:
        if rx.search(base):
            return "out_of_scope", reason
    if entry["python_module"] in ("sparse", "nested"):
        return "out_of_scope", entry["python_module"]
    if "nondeterministic_seeded" in entry["tags"]:
        return "out_of_scope", "rng"
    key = "aten." + name + ("." + overload if overload else "")
    if key in coverage:
        st = coverage[key]["status"]
        return ("traced" if st in ("supported", "pattern", "generic") else "missing"), "coverage_table:" + st
    where = resolve(base)
    if where:
        return "traced", "frontend:" + where
    return "missing", "no frontend entry"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--doc", required=True)
    args = ap.parse_args()
    from formers.fir import coverage_table
    coverage = coverage_table.COVERAGE
    resolve = frontend_resolvers()
    entries = parse_yaml(args.yaml)
    rows = []
    for e in entries:
        status, reason = classify(e, resolve, coverage)
        rows.append(OrderedDict(op=e["name"] + ("." + e["overload"] if e["overload"] else ""),
                                status=status, reason=reason, core="core" in e["tags"],
                                tags=e["tags"], module=e["python_module"], args=e["args"], returns=e["returns"]))
    Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    by = Counter(r["status"] for r in rows)
    core = [r for r in rows if r["core"]]
    core_by = Counter(r["status"] for r in core)
    reasons = Counter(r["reason"] for r in rows if r["status"] == "out_of_scope")
    missing = [r for r in rows if r["status"] == "missing"]
    missing_base = sorted({base_name(r["op"].split(".")[0]) for r in missing})
    lines = ["# ATen op coverage (native_functions.yaml)", "",
             "Generated by `other/formers.n/conformance/tools/gen_aten_table.py` from PyTorch's `native_functions.yaml` "
             "and `formers.fir`. The full table is `other/formers.n/conformance/aten_ops.json`.", "",
             "| Status | Overloads |", "| --- | ---: |"]
    for k in ("traced", "covered_by_functional", "out_of_scope", "missing"):
        lines.append("| %s | %d |" % (k, by.get(k, 0)))
    lines.append("| total | %d |" % len(rows))
    lines += ["", "Core ATen (`tags: core`): %d overloads: %s." % (
        len(core), ", ".join("%s %d" % (k, core_by.get(k, 0)) for k in ("traced", "covered_by_functional", "out_of_scope", "missing"))), ""]
    lines += ["## Out of scope for inference", "", "| Reason | Overloads |", "| --- | ---: |"]
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append("| %s | %d |" % (k, v))
    lines += ["", "## Core ATen ops still missing", ""]
    for r in core:
        if r["status"] == "missing":
            lines.append("- `%s`" % r["op"])
    lines += ["", "## Missing in-scope ops by base name (%d)" % len(missing_base), ""]
    lines.append(", ".join("`%s`" % n for n in missing_base))
    Path(args.doc).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(dict(by), "core:", dict(core_by), "missing base names:", len(missing_base))


if __name__ == "__main__":
    sys.exit(main())
