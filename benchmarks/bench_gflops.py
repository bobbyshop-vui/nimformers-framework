#!/usr/bin/env python3
"""Measure FP32 matmul GFLOPS on the Nimformers CPU, Metal, and Vulkan backends.

By default a failed GPU kernel is an error/SKIP, not a silent CPU fallback, so
the reported GFLOPS belongs to the requested backend.  Use
`--allow-cpu-fallback` only when you explicitly want fallback numbers.

Usage:
    python3 benchmarks/bench_gflops.py
    python3 benchmarks/bench_gflops.py --backends cpu,metal --sizes 256,512,1024
    python3 benchmarks/bench_gflops.py --iters 10 --warmup 2
    python3 benchmarks/bench_gflops.py --csv results.csv

GFLOPS is reported as 2*M*N*K / seconds / 1e9.  Each backend is opened in a
separate Context; unavailable backends are reported as SKIP, not failures.
Build the library first with `make build-for-python`.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Allow running directly from a source checkout.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nimformers import Context, NimformersError, cpu_info  # noqa: E402


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_sizes(text: str) -> list[int]:
    sizes = []
    for item in parse_csv_list(text):
        value = int(item)
        if value < 2:
            raise argparse.ArgumentTypeError("matrix sizes must be >= 2")
        sizes.append(value)
    return sizes


def bench_matmul(ctx: Context, size: int, iters: int, warmup: int,
                 tile: int = 0) -> dict:
    rng = np.random.default_rng(1234)
    a = rng.standard_normal((size, size), dtype=np.float32)
    b = rng.standard_normal((size, size), dtype=np.float32)
    use_tiles = 0 < tile < size

    if use_tiles:
        # A single 8096^3 dispatch can exceed the Metal/MoltenVK command-buffer
        # watchdog on integrated GPUs.  Compute the same 2*N^3 FLOPs as a grid
        # of tile x tile output blocks; each dispatch is short enough to finish.
        c_out = np.empty((size, size), dtype=np.float32)

        def run_once():
            for i0 in range(0, size, tile):
                i1 = min(i0 + tile, size)
                for j0 in range(0, size, tile):
                    j1 = min(j0 + tile, size)
                    c_out[i0:i1, j0:j1] = ctx.matmul(
                        a[i0:i1, :], b[:, j0:j1])
            return c_out
    else:
        def run_once():
            return ctx.matmul(a, b)

    # Keep the output alive and force a real call during warmup (JIT compile,
    # kernel selection, allocator caches, etc.).
    c = None
    for _ in range(warmup):
        c = run_once()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        c = run_once()
        t1 = time.perf_counter()
        samples.append(t1 - t0)

    if c is None or c.shape != (size, size):
        raise RuntimeError(f"unexpected matmul shape {None if c is None else c.shape}")

    # A sink read keeps lazy implementations honest and mirrors real use.
    sink = float(c[0, 0]) + float(c[-1, -1])
    if not np.isfinite(sink):
        raise RuntimeError("matmul produced non-finite values")

    best = min(samples)
    median = statistics.median(samples)
    flops = 2.0 * size * size * size
    return {
        "size": size,
        "tile": tile if use_tiles else 0,
        "iters": iters,
        "best_s": best,
        "median_s": median,
        "best_gflops": flops / best / 1e9,
        "median_gflops": flops / median / 1e9,
    }


def backend_note(ctx: Context) -> str:
    if ctx.backend == "cpu":
        info = cpu_info()
        return f"isa={info['isa_name']} l3={info['l3_bytes'] // 1024}KiB " \
               f"tiles={info['block_mc']}x{info['block_nc']}x{info['block_kc']}"
    return ""


def run_backend(name: str, sizes: list[int], iters: int, warmup: int,
                allow_cpu_fallback: bool, tile: int) -> list[dict]:
    try:
        ctx = Context(backend=name, forbid_cpu_fallback=not allow_cpu_fallback)
    except (NimformersError, ValueError, OSError) as exc:
        print(f"[{name:6s}] SKIP: {exc}")
        return []

    rows: list[dict] = []
    try:
        note = backend_note(ctx)
        print(f"[{name:6s}] backend={ctx.backend} abi={ctx.abi_version}"
              + (f" ({note})" if note else ""))
        for size in sizes:
            try:
                row = bench_matmul(ctx, size, iters, warmup, tile)
            except Exception as exc:  # keep other sizes/backends running
                print(f"[{name:6s}] {size:5d}^3 SKIP: {exc}")
                continue
            row["backend"] = name
            row["runtime_backend"] = ctx.backend
            rows.append(row)
            print(
                f"[{name:6s}] {size:5d}^3  "
                f"best {row['best_gflops']:8.2f} GFLOPS  "
                f"median {row['median_gflops']:8.2f} GFLOPS  "
                f"({row['median_s'] * 1e3:8.3f} ms)"
            )
    finally:
        ctx.close()
    return rows


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("\nNo benchmark results.")
        return
    print("\n=== Peak FP32 matmul GFLOPS (best sample per cell) ===")
    header = f"{'size':>6} | " + " | ".join(f"{b:>10}" for b in
                                             ["cpu", "metal", "vulkan"])
    print(header)
    print("-" * len(header))
    by_cell = {(r["backend"], r["size"]): r for r in rows}
    sizes = sorted({r["size"] for r in rows})
    for size in sizes:
        cells = []
        for backend in ("cpu", "metal", "vulkan"):
            row = by_cell.get((backend, size))
            cells.append(f"{row['best_gflops']:10.2f}" if row else f"{'-':>10}")
        print(f"{size:6d} | " + " | ".join(cells))

    print("\n=== Per-backend best ===")
    for backend in ("cpu", "metal", "vulkan"):
        best = max((r for r in rows if r["backend"] == backend),
                   key=lambda r: r["best_gflops"], default=None)
        if best:
            print(f"{backend:6s}: {best['best_gflops']:8.2f} GFLOPS "
                  f"at {best['size']}^3 ({best['median_s'] * 1e3:.3f} ms median)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="cpu,metal,vulkan",
                        help="comma-separated list (default: cpu,metal,vulkan)")
    parser.add_argument("--sizes", default="128,256,512,1024",
                        help="comma-separated square matrix sizes")
    parser.add_argument("--iters", type=int, default=5,
                        help="timed iterations per size (default: 5)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warmup iterations per size (default: 1)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="write raw results to this CSV file")
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="allow a failed GPU op to be measured as CPU (not recommended)")
    parser.add_argument("--tile", type=int, default=0,
                        help="if >0 and < size, compute size^3 as tiled matmuls "
                             "(useful for sizes that exceed the GPU watchdog)")
    parser.add_argument("--gpu-8096", action="store_true",
                        help="preset: Metal+Vulkan, 8096^3, tile 1024, 1 iteration")
    args = parser.parse_args(argv)

    if args.gpu_8096:
        args.backends = "metal,vulkan"
        args.sizes = "8096"
        args.tile = 1024
        args.iters = 1
        args.warmup = 0

    if args.iters < 1 or args.warmup < 0:
        parser.error("--iters must be >= 1 and --warmup >= 0")
    backends = parse_csv_list(args.backends)
    sizes = parse_sizes(args.sizes)
    if not backends:
        parser.error("--backends cannot be empty")

    print(f"FP32 square matmul; iters={args.iters}, warmup={args.warmup}, "
          f"sizes={sizes}")
    all_rows: list[dict] = []
    for backend in backends:
        all_rows.extend(run_backend(backend, sizes, args.iters, args.warmup,
                                    args.allow_cpu_fallback, args.tile))
        print()

    print_summary(all_rows)

    if args.csv is not None:
        fields = ["backend", "runtime_backend", "size", "tile", "iters", "best_s",
                  "median_s", "best_gflops", "median_gflops"]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {args.csv}")

    return 0 if all_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
