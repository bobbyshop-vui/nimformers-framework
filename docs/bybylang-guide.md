
# Use BybyLang with Nimformers

[English](bybylang-guide.md) | [Tiếng Việt](bybylang-guide.vi.md)

There are three ways to use the repository runtime:

- Write `.bybylang` with the native `gpu ...` DSL.
- Import Nim modules from `.bybylang` and call the framework API or C ABI.
- Import `vendor/bybylang/gpubackend` from Nim to call `gpu*` functions directly.

All use the vendored runtime in `vendor/bybylang/`. See the
[BybyLang API](bybylang-api.md) for signatures and
[GPU sessions](gpu-session-api.md)/[micro-VM](microkernel-api.md) for deeper APIs.


## Build the compiler and run an existing example

From the repository root:

```sh
make -C vendor/bybylang build
mkdir -p nimcache/bybylang-demo
BYBYLANG_NIMCACHE="$PWD/nimcache/bybylang-demo" \
  ./vendor/bybylang/bybylang test/bybylang_test/native_gpu_test.bybylang \
  --aot=test/bybylang_test/native_gpu_test_bin
./test/bybylang_test/native_gpu_test_bin
```

`--aot` creates an executable and a generated Nim file beside it. AOT compilation
does not execute its operations; the last command runs them.
`BYBYLANG_NIMCACHE` selects the compiler cache, not the output path.
Inspect CLI options with `./vendor/bybylang/bybylang --help`.

Run both native DSL and C ABI tests:

```sh
make bybylang-test
```

The runner builds the compiler if the executable is missing. After compiler
source changes, run `make -C vendor/bybylang build` yourself: the runner does
not check staleness. Expected messages include `PASS native add`,
`PASS native micro kernel`, `PASS matmul`, and finally
`[bybylang-test] all tests passed`.


## Write a first GPU script

Save as `test/bybylang_test/hello.bybylang`:

```bybylang
gpu backend is "auto"

gpu array A = [1, 2, 3, 4]
gpu array B = [10, 20, 30, 40]
gpu add A, B -> C size 4
print C

gpu array X = [1, 2, 3, 4, 5, 6]
gpu array W = [1, 0, 0, 1, 1, 1]
gpu matmul X, W -> Y m 2 k 3 n 2
print Y
```

Build and run from the repository root:

```sh
./vendor/bybylang/bybylang test/bybylang_test/hello.bybylang \
  --aot=test/bybylang_test/hello_bin
./test/bybylang_test/hello_bin
```

Addition produces `[11,22,33,44]`; matmul produces `[4,5,10,11]`, representing
`[[4,5],[10,11]]`. Arrays are flat row-major data; `m=2,k=3,n=2` describes
`[2,3] @ [3,2]`. The DSL does not infer shape from six input elements.
Change the backend line to `gpu backend is "metal"` or `"vulkan"` explicitly.


## Sessions and micro-kernels

Group supported operations in a session:

```bybylang
gpu backend is "auto"
gpu array A = [-1, 2, -3, 4]
gpu array B = [0, 1, 0, 1]
session:
  gpu add A, B -> C size 4
  gpu relu C -> D
print D
```

The result is `[0,3,0,5]`. Session encoders depend on the backend; see the
[GPU Session API](gpu-session-api.md) for immediate/fallback paths.

A standalone micro-kernel doubling four elements:

```bybylang
gpu backend is "auto"
gpu array X = [1, 2, 3, 4]
micro kernel double X -> OUT threads 4 tg 4 out 4:
  emitULid 0
  emitULoadIn 1 0 0
  emitAdd 1 1 1
  emitUStoreF32 1 0 0
  emitHalt
print OUT
```

`threads 4` and `tg 4` launch exactly one threadgroup. `emitULid 0` puts
local ID in U0, so this indexing only handles one group. Multiple groups
need a global index derived from group and local IDs. The final two zeros
in the load/store instructions are address register and offset; do not omit them.


## Call the C ABI from `.bybylang`

Save this in `test/bybylang_test/abi_hello.bybylang` to use the search paths
from `test/bybylang_test/nim.cfg`:

```bybylang
import nim "../../lib_c_api"

ctx = nimf_context_create(0)
if ctx == nil:
  print $nimf_last_error()
  quit(1)

A = @[1'f32, 2, 3, 4]
B = @[10'f32, 20, 30, 40]
C = newSeq[float32](4)
if nimf_add(ctx, addr A[0], addr B[0], addr C[0], 4) != 0:
  print $nimf_last_error()
  discard nimf_context_destroy(ctx)
  quit(1)
print C
discard nimf_context_destroy(ctx)
```

```sh
./vendor/bybylang/bybylang test/bybylang_test/abi_hello.bybylang \
  --aot=test/bybylang_test/abi_hello_bin
./test/bybylang_test/abi_hello_bin
```

Here `import nim` compiles `lib_c_api.nim` into the AOT program; it does not
load `libnimformers.dylib` with dynamic FFI as Python does. Symbols and
signatures share the C ABI; see the [public header](../include/nimformers.h).
`nimf_context_create(0)` selects auto; status `0` means success. Release
handles with their matching destroy functions. Read errors from `nimf_last_error()`.

Imports resolve through the compiler and the generated Nim file's location.
These examples keep source and output beside each other in
`test/bybylang_test/` to retain relative imports and `nim.cfg`. Update
imports/search paths when moving scripts elsewhere.


## Call the GPU API from Nim

Save as `byby_demo.nim` at the repository root:

```nim
import vendor/bybylang/gpubackend

setForbidCpuFallback(true)
let be = detectBackend()
let a = @[1'f32, 2, 3, 4, 5, 6]
let b = @[1'f32, 0, 0, 1, 1, 1]
echo gpuMatmul(be, a, b, 2, 3, 2)
```

```sh
nim c -r -d:release --path:. byby_demo.nim
```

Framework models use `Backend` and `be*` functions from `backend.nim`.
Direct BybyLang calls use `GpuBackend` and `gpu*`; the types are not
interchangeable. Low-level BybyLang retains some CPU reference/fallback
paths; `setForbidCpuFallback(true)` makes an operation's GPU fallback an error.


## Troubleshooting

- AOT cannot find `lib_c_api`, `fp4_gemv_gpu`, or model modules:
  check relative imports and `nim.cfg` search paths.
- Old compiler behavior after source changes: rebuild the vendored compiler.
- Metal/Vulkan unavailable: try the binary in a terminal with GPU access;
  check the driver/runtime before changing DSL code.
- A parser misreads `=` in a trailing Nim comment: put the comment on its
  own line. Assignment parsing includes a branch based on `=`.

Full test scripts are in [test/bybylang_test](../test/bybylang_test/README.md).
