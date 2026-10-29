
# BybyLang GPU API and DSL

[English](bybylang-api.md) | [Tiếng Việt](bybylang-api.vi.md)

BybyLang is vendored under `vendor/bybylang`. Nim callers can import `vendor/bybylang/bybylang` for the DSL compiler or `vendor/bybylang/gpubackend` for its GPU operations. The framework facade re-exports selected backend/session symbols through `backend.nim` and `nimformers.nim`.

See the [BybyLang walkthrough](bybylang-guide.md) for complete examples: build the compiler, compile/run a script, call the C ABI, and
manage source/output search paths. `make bybylang-test` runs the native DSL
and imported C ABI examples together.


## Backend selection

```nim
type GpuBackend* = enum
  gbAuto, gbCpu, gbCuda, gbMetal, gbVulkan

proc parseBackend*(s: string): GpuBackend
proc normalizeBackend*(b: GpuBackend): GpuBackend
proc detectBackend*(): GpuBackend
proc setForbidCpuFallback*(forbid: bool)
```

`parseBackend("cuda")` maps to Vulkan. The framework-level `newBackend("cpu")` is unsupported; direct BybyLang GPU functions have CPU reference/fallback behavior for operations where implemented. `setForbidCpuFallback(true)` turns operation fallback into a hard error.


## Immediate GPU operations

Functions take `backend: GpuBackend`; input and output arrays are flattened row-major data.

```nim
gpuMatmul(backend, a, b, m, k, n)
gpuMatmul2(backend, a1, b1, m1, k1, n1, a2, b2, m2, k2, n2)
gpuMatmulQ4(backend, a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
gpuOp(op, backend, a, b) # add, sub, mul, div
gpuRelu(backend, x); gpuSigmoid(backend, x); gpuTanh(backend, x)
gpuSqrt(backend, x); gpuExp(backend, x); gpuLog(backend, x); gpuPow(backend, x, p)
gpuSoftmax(backend, x, rows, cols)
gpuLayernorm(backend, x, gamma, beta, rows, cols, eps)
gpuLayernormBackward(backend, dy, x, gamma, beta, rows, cols, eps)
gpuEmbeddingLookup(backend, table, indices, vocab, dim)
gpuAttentionFused(backend, q, k, v, mask, B, H, S, D, scale)
gpuAttentionFusedBackward(backend, q, k, v, sMatrix, dy, B, H, S, D, scale)
gpuApflu(backend, x, alpha, beta); gpuApfluBackward(backend, x, dy, alpha, beta)
gpuFusedAddAct(backend, a, b, act)
gpuGather(backend, src, indices)
gpuArgmaxIndex(backend, x, outer, dimSize, inner)
gpuCompare(backend, a, b, op) # op: 0 gt, 1 lt, 2 eq, 3 ge, 4 le
gpuDropoutMask(backend, n, p, seed)
gpuPermute(backend, src, newStrides, oldStrides, dimsMap, n)
gpuSplitSlice(backend, src, dimTotal, thisSize, offset, inner, n)
gpuConcat(backend, flatSrc, srcBase, srcDimSize, srcDimOffset, numTensors, totalDim, inner)
```

Q4 resident buffers and additional operations are documented in [Backend API](backend-api.md) and [GPU Session API](gpu-session-api.md).


## BybyLang source syntax

The parser reads `.bybylang` files and emits a Nim AOT program. Imports resolve relative to the current directory or the importing file's directory. `import nim` imports a Nim module into the generated AOT module.

```bybylang
gpu backend is "auto"
import nim "math"

gpu array A = [1, 2, 3, 4]
gpu array B = [10, 20, 30, 40]
gpu add A, B -> C size 4
print C
print sqrt(16.0)
```

Common GPU DSL statements include `gpu array`, `gpu add|sub|mul|div`, `gpu relu|sigmoid|tanh|apflu`, `gpu softmax`, `gpu matmul`, `gpu layernorm`, `gpu embedding`, `gpu attention`, `gpu gather|compare|fused_add_act`, and unary `gpu sqrt|exp|log|pow`. Dimensions and optional parameters follow each operation after `->`; use `vendor/bybylang/demo/demo_gpu.bybylang` as the runnable syntax reference.


## Session block

Group operations into one command buffer using:

```bybylang
session:
  gpu add A, B -> C size 4
  gpu relu C -> D
```

Supported operations depend on backend. The compiler falls back to immediate operations for unsupported session encoders where a fallback exists. See [GPU Session API](gpu-session-api.md).


## Native micro-kernel block

Micro-kernel bytecode can be declared directly in `.bybylang`:

```bybylang
micro kernel double A -> OUT threads 4 tg 4 out 4:
  emitRToU 0 0
  emitULoadIn 1 0 0
  emitLoadConstF32 2 2.0'f32
  emitMul 1 1 2
  emitUStoreF32 1 0 0
  emitHalt
```

`A` is a float input array. `threads` is dispatch size, `tg` is threadgroup size, and `out` is output element count. The block compiles the `emit*` lines to `VmProgram` bytecode and runs it via the selected GPU session. See [Micro-VM / JIT API](microkernel-api.md).


## AOT build and run

```sh
# From the Nimformers repository root:
make -C vendor/bybylang build
./vendor/bybylang/bybylang vendor/bybylang/demo/demo_gpu.bybylang \
  --aot=vendor/bybylang/demo/demo_gpu_out
./vendor/bybylang/demo/demo_gpu_out
```

`--aot` writes and compiles a Nim source file; execute the generated binary
separately. `BYBYLANG_NIMCACHE` selects the compiler cache directory. Examples
importing `lib_c_api` also need the framework's module search paths; the test
directory supplies them in `test/bybylang_test/nim.cfg`. An `import nim` of
`lib_c_api` compiles the Nim ABI implementation into the binary and does not
dynamically load the Python shared library.

On macOS where the build/run session needs to remain awake, prefix the relevant commands with `caffeinate -i`. For the GPU process, ensure the backend can access the device; a sandboxed/headless launch may not expose Metal.
