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

for name in syntax_test native_gpu_test cpu_test capi_test trace_test diag_test; do
    echo ""
    echo "=== [bybylang-test] $name ==="
    "$BYBY" "test/bybylang_test/$name.bybylang" \
        --aot="test/bybylang_test/${name}_bin"
    "test/bybylang_test/${name}_bin"
done

echo ""
echo "=== [bybylang-test] diagnostics ==="
# warn / traceback / error print the BybyLang position on stderr.
out="$(test/bybylang_test/diag_test_bin 2>&1)"
for want in "diag_test.bybylang:6: warning: this is only a warning" \
            "traceback (most recent call last):" "diag_test.bybylang:3 <- traceback" \
            "diag_test: ok"; do
    case "$out" in *"$want"*) ;; *) echo "missing in diag_test output: $want"; echo "$out"; exit 1;; esac
done
# A syntax error is reported on the .bybylang line with a caret, exits 1, and shows no Nim trace.
BAD="$CACHE/diag-bad"
mkdir -p "$BAD"
printf 'x = (1 + 2\nprint x\n' > "$BAD/paren.bybylang"
printf 'if true\n  print 1\n' > "$BAD/colon.bybylang"
for pair in "paren:paren.bybylang:1:11:expected: ')'" "colon:colon.bybylang:1:8:expected: ':'"; do
    IFS=: read -r name file line col msg <<<"$pair"
    set +e
    out="$("$BYBY" "$BAD/$file" --aot="$BAD/$name" 2>&1)"
    code=$?
    set -e
    [ "$code" -eq 1 ] || { echo "$name: expected exit 1, got $code"; echo "$out"; exit 1; }
    case "$out" in *"$file:$line:$col: error: $msg"*) ;; *) echo "$name: wrong position/message"; echo "$out"; exit 1;; esac
    case "$out" in *"^"*) ;; *) echo "$name: no caret line"; echo "$out"; exit 1;; esac
    case "$out" in *"unhandled exception"*|*"generateNimCode"*) echo "$name: Nim trace leaked"; echo "$out"; exit 1;; esac
done
echo "diagnostics: ok"

echo ""
echo "=== [bybylang-test] cpu_kernel_regression ==="
"${NIM:-nim}" c -r --hints:off --mm:arc --opt:speed \
    --path="$ROOT/vendor/bybylang" \
    --nimcache="$CACHE/cpu-kernel-regression" \
    -o:"$CACHE/cpu_kernel_regression" \
    "$HERE/cpu_kernel_regression.nim"

echo ""
echo "[bybylang-test] all tests passed"
