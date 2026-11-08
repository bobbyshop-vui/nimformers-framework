#!/usr/bin/env bash
# Build and run the BybyLang integration tests.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BYBY="$ROOT/vendor/bybylang/bybylang"
CACHE="$ROOT/nimcache/bybylang-test"

cd "$ROOT"
mkdir -p "$CACHE"
export BYBYLANG_NIMCACHE="$CACHE"

if [ ! -x "$BYBY" ]; then
    echo "[bybylang-test] building bybylang..."
    make -C vendor/bybylang build
fi

for name in syntax_test native_gpu_test cpu_test capi_test trace_test; do
    echo ""
    echo "=== [bybylang-test] $name ==="
    "$BYBY" "test/bybylang_test/$name.bybylang" \
        --aot="test/bybylang_test/${name}_bin"
    "test/bybylang_test/${name}_bin"
done

echo ""
echo "=== [bybylang-test] cpu_kernel_regression ==="
"${NIM:-nim}" c -r --hints:off --mm:arc --opt:speed \
    --path="$ROOT/vendor/bybylang" \
    --nimcache="$CACHE/cpu-kernel-regression" \
    -o:"$CACHE/cpu_kernel_regression" \
    "$HERE/cpu_kernel_regression.nim"

echo ""
echo "[bybylang-test] all tests passed"
