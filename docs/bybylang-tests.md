# BybyLang integration test reference

[English](bybylang-tests.md) | [Tiếng Việt](bybylang-tests.vi.md)

This page explains how to run, read, and extend the two `.bybylang` programs
in `test/bybylang_test`. It documents only the APIs those tests currently use;
see the [BybyLang API](bybylang-api.md) and
[language reference](bybylang-language.md) for the complete language surface.


## Test files

- [`native_gpu_test.bybylang`](../test/bybylang_test/native_gpu_test.bybylang)
  uses native `gpu ...` commands and a `micro kernel ...:` block. It tests
  vector addition, matrix multiplication, softmax, ReLU, and micro-VM dispatch.
- [`capi_test.bybylang`](../test/bybylang_test/capi_test.bybylang) imports
  `../../lib_c_api` as a Nim module. It tests the ABI version, backend context,
  addition, matmul, softmax, LayerNorm, INT8 quantization, micro-VM dispatch,
  and fused attention.
- [`run.sh`](../test/bybylang_test/run.sh) AOT-compiles and executes both files.
- [`nim.cfg`](../test/bybylang_test/nim.cfg) supplies the repository search paths
  required by generated Nim code that imports `lib_c_api.nim`.

Generated `native_gpu_test_bin`, `capi_test_bin`, and their `.nim` sources are
build artifacts. They are ignored by Git and can be regenerated at any time.


## Requirements and quick run

Run from the repository root:

```sh
make bybylang-test
```

The tests require Nim, the vendored BybyLang compiler, and a Metal or Vulkan
GPU. Backend `auto` tries Metal first on macOS and then Vulkan. The legacy name
`cuda` is an alias for Vulkan. Although the parser accepts `cpu`, the native
micro-kernel requires Metal or Vulkan, so the complete suite is not a CPU test.

The runner builds `vendor/bybylang/bybylang` only when the compiler executable
is missing. Rebuild it explicitly after changing compiler source:

```sh
make -C vendor/bybylang build
make bybylang-test
```

A successful run ends with:

```text
=== native BybyLang GPU test PASSED ===
=== BybyLang C ABI test PASSED ===
[bybylang-test] all tests passed
```


## Compile and run one file manually

The runner is equivalent to the following commands for the native test:

```sh
mkdir -p nimcache/bybylang-test
BYBYLANG_NIMCACHE="$PWD/nimcache/bybylang-test" \
  ./vendor/bybylang/bybylang \
  test/bybylang_test/native_gpu_test.bybylang \
  --aot=test/bybylang_test/native_gpu_test_bin
./test/bybylang_test/native_gpu_test_bin
```

For the C ABI test:

```sh
BYBYLANG_NIMCACHE="$PWD/nimcache/bybylang-test" \
  ./vendor/bybylang/bybylang \
  test/bybylang_test/capi_test.bybylang \
  --aot=test/bybylang_test/capi_test_bin
./test/bybylang_test/capi_test_bin
```

`--aot=PATH` emits `PATH.nim`, invokes `nim c -d:release`, and creates the
native executable at `PATH`. Compilation does not run the test. The final
command executes it. `BYBYLANG_NIMCACHE` controls Nim's cache directory, not
the generated source or executable location.

The output path matters for `capi_test.bybylang`: keeping generated output in
`test/bybylang_test` lets Nim find that directory's `nim.cfg`, while the import
`../../lib_c_api` resolves back to the repository root.


## How native GPU DSL lines are translated

The central part of `native_gpu_test.bybylang` is:

```bybylang
gpu backend is "auto"
gpu array X = [1, 2, 3, 4]
gpu array Y = [10, 20, 30, 40]
gpu array P = [1, 2, 3, 4, 5, 6]
gpu array Q = [1, 0, 0, 1, 1, 1]

gpu add X, Y -> Z size 4
gpu matmul P, Q -> R m 2 k 3 n 2
gpu softmax X -> SM rows 1 cols 4
gpu relu X -> RELU
```

The compiler generates direct calls to the vendored GPU runtime:

- `gpu backend is "auto"` sets `gpuBackendSelected = parseBackend("auto")`.
  Accepted names are `auto`, `metal`, `vulkan`, `cuda`, and `cpu`.
- `gpu array NAME = [...]` creates a flat `seq[float32]`. Arrays do not carry
  tensor shapes.
- `gpu add X, Y -> Z size 4` calls `gpuOp("add", ...)`. The `size` value is
  descriptive in the current generator; it is not passed to `gpuOp`.
- `gpu matmul P, Q -> R m 2 k 3 n 2` treats `P` as row-major `[2,3]`, `Q` as
  row-major `[3,2]`, and returns a flat `[2,2]` result.
- `gpu softmax X -> SM rows 1 cols 4` normalizes each of the specified rows.
  The input must contain at least `rows * cols` values.
- `gpu relu X -> RELU` calls `gpuRelu`; the positive test input makes `RELU`
  equal to `X`.

When the source says `auto`, `BYBYLANG_BACKEND` can override it while the
compiler is generating Nim code:

```sh
BYBYLANG_BACKEND=vulkan \
  ./vendor/bybylang/bybylang \
  test/bybylang_test/native_gpu_test.bybylang \
  --aot=test/bybylang_test/native_gpu_test_bin
```

The chosen override is embedded in the generated program. Setting the variable
only when launching an already-built binary does not rewrite that program.


## Native `micro kernel` API used by the test

The test doubles four input values:

```bybylang
micro kernel byby_double X -> DOUBLED threads 4 tg 4 out 4:
  emitULid 0
  emitULoadIn 1 0 0
  emitAdd 1 1 1
  emitUStoreF32 1 0 0
  emitHalt
```

The header has this form:

`micro kernel NAME INPUT -> OUTPUT threads N tg N out N:`

- `threads` is the total number of dispatched threads.
- `tg` is the number of threads per threadgroup.
- `out` is the number of `float32` elements read back into `OUTPUT`.
- The indented body accepts exported `emit...` procedures from
  `vendor/bybylang/micro_kernel/session_vm.nim`.

The five instructions used here are:

- `emitULid 0`: put the local thread ID into unsigned register `U0`.
- `emitULoadIn 1 0 0`: load `input[U0 + 0]` into float register `R1`.
- `emitAdd 1 1 1`: compute `R1 = R1 + R1`.
- `emitUStoreF32 1 0 0`: store `R1` into `output[U0 + 0]`.
- `emitHalt`: stop the program for this thread.

This example deliberately dispatches one group (`threads == tg == 4`), so a
local ID is also a global index. For multiple threadgroups, derive a global
index from `emitUTgid`, the group size, and `emitULid`; otherwise each group
writes the same output positions. Register IDs are limited to the micro-VM's
16 float registers `R0..R15` and 16 unsigned registers `U0..U15`.


## Why `import nim` can call the C ABI

The C ABI test starts with:

```bybylang
import nim "../../lib_c_api"
```

BybyLang places this as a real Nim `import` at the top of generated source.
It compiles `lib_c_api.nim` into the AOT executable; it does not dynamically
load `libnimformers.dylib`. The test then uses Nim expressions such as `@[]`,
`newSeq`, `addr`, `nil`, `discard`, and numeric suffixes like `1'f32` directly.
Lines that are not special BybyLang commands are largely emitted as Nim code.


## Runtime and error API

The test uses these calls:

- `nimf_abi_version(): uint32` returns the numeric ABI version.
- `nimf_context_create(backendKind): pointer` creates a context. Values are
  `0=auto`, `1=Metal`, `2=Vulkan`, and `3=CUDA aliasing Vulkan`. It returns
  `nil` on error.
- `nimf_context_backend_name(ctx): cstring` returns `metal` or `vulkan` for a
  valid context.
- `nimf_last_error(): cstring` returns the last error for the current thread.
- `nimf_context_destroy(ctx): cint` releases the context; a `nil` handle is
  accepted.

Functions returning `cint` use `0` for success and nonzero for failure. Pointer
constructors return `nil` on failure. `nimf_quant_dequantize` is different: it
returns the number of values written or `-1` on failure. Read
`nimf_last_error()` immediately after a failed operation.


## Tensor C ABI used by the test

All tensor buffers below are contiguous flat `float32` sequences:

- `nimf_add(ctx, a, b, out, n)` adds `n` values element by element. The test
  passes addresses with `addr A[0]` and preallocates `out` with `newSeq`.
- `nimf_matmul(ctx, a, b, out, m, k, n)` computes row-major
  `A[m,k] @ B[k,n] -> out[m,n]`. The test uses `(m,k,n) = (2,3,2)`.
- `nimf_softmax(ctx, x, out, rows, cols)` treats `x` as `[rows,cols]` and
  normalizes each row. The test checks that one row sums to approximately one.
- `nimf_layernorm(ctx, x, gamma, beta, out, rows, cols, eps)` requires
  `x/out` length `rows*cols` and `gamma/beta` length `cols`. The test uses
  unit gamma, zero beta, and verifies the output mean is near zero.

These procedures copy host sequences into the selected backend and copy the
result back into the caller's preallocated output sequence.


## Quantization C ABI used by the test

The call
`nimf_quantize(x, n, shape, rank, kind, exponentBits, mantissaBits)` returns an
owned quantized-tensor handle. The test passes shape `[4]`, rank `1`, and kind
`1`, which is `QK_INT8`. The exponent and mantissa arguments are `0` because
they only configure `QK_CUSTOM`.

`numel(shape)` must equal `n`. The test calls
`nimf_quant_dequantize(handle, out, capacity)` and expects the return value `4`.
Finally, `nimf_quant_destroy(handle)` releases the quantized tensor. Every
successful quantize call must have a matching destroy call.


## Micro-VM C ABI used by the test

The C ABI test builds the same doubling operation with numeric opcodes:

```bybylang
prog = nimf_vm_create()
discard nimf_vm_emit(prog, 0x10'u32, 1, 0, 0, 0)
discard nimf_vm_emit(prog, 0x20'u32, 1, 1, 1, 0)
discard nimf_vm_emit(prog, 0x11'u32, 1, 0, 0, 0)
discard nimf_vm_emit(prog, 0x01'u32, 0, 0, 0, 0)

VMEM = @[1'f32, 2, 3, 4]
discard nimf_vm_run(ctx, prog, addr VMEM[0], 4, 4, 0, 4)
discard nimf_vm_destroy(prog)
```

- `nimf_vm_create()` allocates an empty program.
- `nimf_vm_emit(program, opcode, a, b, c, value)` appends one typed
  instruction. Here `0x10` loads `memory[R0]` into `R1`, `0x20` adds
  `R1 + R1`, `0x11` stores `R1` to `memory[R0]`, and `0x01` halts.
- The generated GPU kernel initializes float register `R0` with the global
  thread ID, so four threads process four elements.
- `nimf_vm_run(ctx, program, memory, memoryLen, nThreads, tgFloats, tgSize)`
  modifies `memory` in place. The test uses four elements, four threads, no
  requested threadgroup scratch floats, and a group size of four.
- `nimf_vm_destroy(program)` releases the program.

Check every emit/run return status in application code. The test discards emit
statuses only because it uses known-valid constants and separately validates
the final result.


## Fused attention C ABI used by the test

`nimf_attention` has the following logical layout:

- `q`, `k`, `v`, and `out`: flat `[batch, heads, sequenceLen, headDim]`.
- `scores`: flat `[batch, heads, sequenceLen, sequenceLen]`.
- `mask` plus `maskLen`: an optional flat mask. `nil, 0` means no mask.
- `scale`: the score multiplier, commonly `1 / sqrt(headDim)`.

The test uses `(batch, heads, sequenceLen, headDim) = (1,1,4,4)`, so Q/K/V and
output each contain 16 floats and scores contains 16 floats. It passes scale
`0.5`, then checks that the first output is not NaN. This is a smoke test for
dispatch and shape wiring, not a numerical reference test for every value.


## Ownership rules

The test creates three owned handles and releases each with its matching API:

- context: `nimf_context_create` -> `nimf_context_destroy`
- quantized tensor: `nimf_quantize` -> `nimf_quant_destroy`
- VM program: `nimf_vm_create` -> `nimf_vm_destroy`

Sequences created by `newSeq` are managed by Nim. Do not destroy them through
the C ABI. Keep input and output sequences alive while their `addr seq[0]`
pointers are in use, and never take element zero from an empty sequence.


## Add another `.bybylang` test

1. Put the source in `test/bybylang_test` so it inherits `nim.cfg`.
2. Make failures call `quit(1)`; `run.sh` uses `set -e` and stops on a nonzero
   exit status.
3. Compile it manually first with a distinct `--aot` output name.
4. Add its base name to the loop in `run.sh` if it should run under
   `make bybylang-test`.
5. Rebuild the compiler manually if you changed `vendor/bybylang` source.

Avoid `=` in a trailing comment on a raw Nim line. The current parser may treat
any non-command line containing `=` as an assignment; put such comments on a
separate line.


## Related documentation

- [BybyLang walkthrough](bybylang-guide.md)
- [BybyLang GPU API and DSL](bybylang-api.md)
- [BybyLang language reference](bybylang-language.md)
- [Python and C ABI](python-api.md)
- [Micro-VM / JIT API](microkernel-api.md)

