
# BybyLang GPU API and DSL

[English](bybylang-api.md) | [Tiếng Việt](bybylang-api.vi.md)

BybyLang is vendored under `vendor/bybylang`. Nim callers can import `vendor/bybylang/bybylang` for the DSL compiler or `vendor/bybylang/gpubackend` for its GPU operations. The framework facade re-exports selected backend/session symbols through `backend.nim` and `nimformers.nim`.

The walkthrough covers building the compiler, running a script and calling the
framework C ABI. The Python package calls the same C ABI through the shared
library. [BybyLang walkthrough](bybylang-guide.md) · [Python API](python-api.md)


## Backend selection

```nim
type GpuBackend* = enum
  gbAuto, gbCpu, gbCuda, gbMetal, gbVulkan

proc parseBackend*(s: string): GpuBackend
proc normalizeBackend*(b: GpuBackend): GpuBackend
proc detectBackend*(): GpuBackend
proc setForbidCpuFallback*(forbid: bool)
```

`parseBackend("cuda")` maps to Vulkan. Both `parseBackend("cpu")` and the
framework's `newBackend("cpu")` select CPU explicitly. With `auto`, detection
tries Metal on macOS, then Vulkan, then CPU. For an operation with a CPU
fallback, `setForbidCpuFallback(true)` makes a GPU failure raise an error.


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


## Native command replay

Use the Nim session API for fixed-shape steps whose buffer handles can stay
live across runs. Replay submits the same commands against those handles, so a
host write can change the input values between steps.

1. Check `sessionCanRecord(backend)`. Allocate persistent inputs, outputs and
   scratch before `sessionBeginRecording(backend)`.
2. Encode the step and call `sessionEnd(backend)`. Check its return value and
   the ID from `sessionLastRecording(backend)`. An ID of `-1` means the commands
   executed but could not be retained after a successful `sessionEnd`. Staging
   uploads, deferred frees and internal scratch can prevent retention.
3. Use `sessionHostBuffer` and `sessionHostWrite` to supply changing input. For
   host readback, encode a copy to the mapped output and record
   `sessionHostReadBarrier(backend)` before ending the session. Call
   `sessionReplay(backend, id)` outside an active session and check its return
   value before reading with `sessionHostRead`.
4. Release the recording with `sessionFreeRecording(backend, id)` before
   freeing any of its buffers.

Vulkan keeps a reusable command buffer and its descriptor pools. Metal can use
an indirect command buffer or issue the recorded dispatches on a plain encoder.
`sessionRecordingNote(backend, id)` reports the Metal route or the reason for
refusal. `BYBY_METAL_REPLAY_ICB=0` selects the plain encoder. The Metal replay
test shows the buffer lifecycle and compares replay with a fresh encode.
[Replay test](../vendor/bybylang/tests/t_metal_session_replay.nim)

## Prefill session operations

- `sessionFp4GemmTensorCore` uses Vulkan cooperative matrices for packed FP4-KM
  weights. `sessionFp4GemmTiledEnc` uses register tiles on Vulkan.
  `sessionFp4GemmMetal` uses register tiles or Apple `simdgroup_matrix` when
  requested and supported. Allocate the Metal output handle before encoding.
- `sessionF16GemmTile(backend, a, w, c, M, K, N, tensorCore)` expects IEEE FP16
  weights in `[N, K]` order and computes `c = a @ w.T`. Vulkan uses cooperative
  matrices when `tensorCore` is true and register tiles otherwise. For Vulkan
  FP4-KM and FP16 routes, call `sessionTileGemmReady` outside a session before
  first use to run the parity check. An encoder returning `false` leaves the
  choice of a fallback to the caller.
- `sessionMamba2SsdPrepare(backend, N, D, Q)` prepares chunked SSD prefill on
  Vulkan outside an active session. It returns `"coop"`, `"scalar"`, or `""`
  for an unavailable route. `sessionMamba2SsdPrefillEnc` then encodes five
  dispatches against the supplied `float32` arena, including state carry.
- `sessionCompileSeconds()` is a process-wide running total for shader
  compilation and pipeline creation on cache misses. Take the difference
  before and after prefill to report its setup time separately.

The tile programs are built by `buildFp4TileGemmProgram` and
`buildF16TileGemmProgram` in the VM instruction module. `tile_gemm.nim` lowers
those programs for the selected GPU. [Session signatures](../vendor/bybylang/gpubackend.nim) ·
[Tile GEMM lowering](../vendor/bybylang/micro_kernel/tile_gemm.nim)


## Native micro-kernel block

Micro-kernel bytecode can be declared directly in `.bybylang`:

```bybylang
micro kernel double A -> OUT threads 4 tg 4 out 4:
  emitULid 0
  emitULoadIn 1 0 0
  emitLoadConstF32 2 2.0'f32
  emitMul 1 1 2
  emitUStoreF32 1 0 0
  emitHalt
```

`A` is a float input array. `threads` sets the dispatch size, `tg` sets the
threadgroup size, and `out` sets the output element count. `emitULid` uses the
local index within this single group. Use a group offset for multiple groups.
The compiler builds `VmProgram` bytecode for GPU lowering or execution by the
CPU VM. [Micro-VM / JIT API](microkernel-api.md)


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

## Tracing

`trace.nim` records a run for the FIR lifter. A BybyLang script imports it like any Nim module and calls
the same functions as Nim code:

```bybylang
import nim "../../trace"

traceStart("run.fprobe", "decode:2", true, 0)   # path, steps, data, dataMax
# ... build and run the model ...
traceStop()
```

Start recording before the operations you want to capture. Registered kernel
bytecode survives across traces, so pipelines can be compiled beforehand.
`traceBegin` and `traceEnd` mark steps. `traceActive`, `traceFile`, `withTrace`
and `traceStep` are also available.

For Codestral Vulkan, use `firlift --run` with the native CLI to record through
an in-memory pipe. The lift uses the original checkpoint and LoRA at graph load
and writes `model.fir` and `config.json`. The `codestral_prompt` marker bounds
the operations that can enter the replay. [Trace API](trace-api.md) ·
[Codestral FIR lift](fir-script-lift.md#codestral-on-vulkan)
