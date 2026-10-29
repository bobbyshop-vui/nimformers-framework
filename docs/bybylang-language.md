# BybyLang

[English](bybylang-language.md) | [Tiếng Việt](bybylang-language.vi.md)

This is the framework-tracked bilingual edition of the [vendored BybyLang README](../vendor/bybylang/readme.md). The multi-group micro-kernel example computes a global index from group/local IDs.

A Nim-based DSL, AOT-compiled to Nim source then built into a native binary. Its `gpu ...` commands generate tensor operations; `session:` batches supported operations; and `micro kernel` defines a native BybyLang VM kernel that is lowered to Metal or Vulkan GPU code. Generic control flow and hardware-simulation commands are also available.


## Build & run

```bash
make -C vendor/bybylang build
make -C vendor/bybylang test
# Or, from vendor/bybylang itself:
make build
make test
# equivalent to the build and AOT run:
nim c -d:release -o:bybylang bybylang.nim
./bybylang demo/demo_gpu.bybylang --aot=demo/demo_gpu_out
./demo/demo_gpu_out
```

`--aot=<path>` generates `<path>.nim`, compiles it with Nim, and produces the binary `<path>`.

---


## 1. General syntax

- One command per line in a `.bybylang` file. Indentation (spaces/tabs) determines the block for `if/elif/else/while/for`.
- `#` at the start of a line is a comment.
- `import name` or `import "path/name.bybylang"` — recursive import, `.bybylang` extension auto-appended, cycle-protected via absolute path.
- `print <expr>` → generates `echo <expr>`.
- `function NAME` opens a definition; the body is whatever's indented deeper than the `function` line (same dedent rule as `if/while/for`), no closing marker needed. Invoke it with `call NAME` or just a line containing only `NAME`.
- `if cond:` / `elif cond:` / `else:` / `while cond:` / `for x in range(a, b):` — translated directly to the equivalent Nim construct, arbitrarily nestable.
- `mode is N` (N = 1..4) → prints "Mode 1: Low-level" / "Mode 2: Mid-level" / "Mode 3: High-level" / "Mode 4: Web-level" (any other N → "Unknown mode").


## 2. `gpu ...` commands


### Select backend

```
gpu backend is "auto"      # auto | cpu | cuda (Vulkan alias) | metal | vulkan
```

`auto` probes Metal on macOS, then Vulkan, then CPU. `cuda` is a legacy backend name that maps to Vulkan; CUDA/cuBLAS is not implemented in this version. `opencl` is not a current `GpuBackend` option.

`gForbidCpuFallback` defaults to `false`. Call `setForbidCpuFallback(true)` when you need GPU-only execution to fail fast instead of allowing operation-level CPU fallback. Backend availability and operation coverage vary; check boolean results from session encoders and surface errors from `sessionEnd`.


### Declare an array

```
gpu array A = [1, 2, 3, 4]
```

Generates `var A: seq[float32] = @[1, 2, 3, 4].mapIt(it.float32)`. **Must be declared before use** — codegen has no hoisting; it translates lines strictly in file order.


### Basic binary ops

```
gpu add A, B -> C
gpu sub A, B -> C
gpu mul A, B -> C
gpu div A, B -> C
```

A trailing `size N` is descriptive only and doesn't affect codegen.


### Matmul

```
gpu matmul A, B -> C m M k K n N
```
A: [M×K], B: [K×N] → C: [M×N].

`gpuMatmul2` (two independent matmuls in one backend call) is available to Nim callers, but has no `.bybylang` syntax. Call `gpuMatmul2(...)` directly from Nim.


### Activations

```
gpu relu X -> Y
gpu sigmoid X -> Y
gpu tanh X -> Y
gpu apflu X -> Y alpha A beta B      # alpha/beta optional; defaults to alpha=0.1, beta=1.0 if omitted
```

Both `apflu_backward` and `fused_add_act` have `.bybylang` syntax and Nim APIs (`gpuApfluBackward`, `gpuFusedAddAct`).


### Softmax / LayerNorm

```
gpu softmax X -> Y rows R cols C
gpu layernorm X, GAMMA, BETA -> Y rows R cols C eps E
gpu layernorm_backward DY, X, GAMMA, BETA -> DX, DGAMMA, DBETA rows R cols C eps E
```

### Embedding

```
gpu embedding TABLE, INDICES -> Y vocab V dim D
```
`INDICES` is auto-cast to `int32`.


### Attention (fused)

```
gpu attention Q, K, V -> O, S_MATRIX B H S D scale SCALE
gpu attention_backward Q, K, V, S_MATRIX, DY -> DQ, DK, DV B H S D scale SCALE
```
B=batch, H=heads, S=seq_len, D=head_dim.

### RoPE

```
gpu rope X, COS, SIN -> Y seqlen T heads H dim D
```
Rotates `X` in place using `COS`/`SIN` tables. Use this inside `session:`; there is no immediate `gpu rope` implementation. If the backend cannot begin a session, generated code warns and returns the original input unchanged, so do not treat that fallback as a RoPE result.


### Other DSL operations

The parser also supports `sqrt`, `exp`, `log`, `pow`, `gather`, `compare`, `fused_add_act`, `apflu_backward`, `permute`, `split_slice`, `concat`, `argmax_index`, and `dropout_mask`. Their exact operand and shape syntax is demonstrated in [`demo/demo_gpu.bybylang`](../vendor/bybylang/demo/demo_gpu.bybylang). `compare` returns a boolean mask; gather and embedding indices are cast to `int32`.


### Generic op (fallback)

```
gpu <other_op> A, B -> C
```
Generates `gpuOp("<other_op>", backend, A, B)` — requires a matching branch inside `gpuOp` to actually do something.

---


## 3. `session:` blocks — batching multiple `gpu ...` calls

```
session:
  gpu add A, B -> C size 4
  gpu mul C, A -> D size 4
  gpu relu D -> E
  gpu rope Q, COS, SIN -> Q2 seqlen 4 heads 1 dim 4
```

Wraps the enclosed `gpu ...` lines with the Session API (`sessionBegin` / `sessionUpload` / `session<Op>` / `sessionEnd` / `sessionRead` / `sessionFree` in `gpubackend.nim`) so intermediate results stay on the GPU across ops instead of round-tripping through `seq[float32]` after every single call.

- Ops with a session-encoded path include `add`/`sub`/`mul`/`div`, activations, `softmax`, `matmul`, `layernorm`, `rope`, `fused_add_act`, `apflu`/`apflu_backward`, `gather`, `compare`, `dropout_mask`, and `split_slice`. Other operations such as `embedding`, `attention`, `attention_backward`, `layernorm_backward`, `permute`, `concat`, and `argmax_index` run through their immediate APIs when placed in the block; those calls do not become resident session ops.
- Unlike outside a session, a trailing `size N` **does** affect codegen here — it's used as the element-count argument passed to the session op, not just descriptive.
- If `sessionBegin()` fails, the generated block falls back to immediate calls for supported fallback operations. `gpu rope` has no immediate equivalent; fallback warns and leaves its output equal to its input (RoPE is skipped).
- Non-`gpu` lines inside a `session:` block (assignments, `print`, etc.) run normally, outside the session's GPU batching.

---


## 4. Nim GPU and Session APIs

Import the backend directly from a Nim application. With the project root as the working directory:

```nim
import vendor/bybylang/gpubackend

let backend = parseBackend("vulkan") # also "metal", "auto", "cpu"
let a = @[1'f32, 2, 3, 4]
let b = @[10'f32, 20, 30, 40]
let c = gpuOp("add", backend, a, b)
echo c
```

For the DSL compiler facade and VM helpers, import `vendor/bybylang/bybylang`; it re-exports `gpubackend`, `micro_kernel`, and the Metal/Vulkan modules. For backend-only code, `vendor/bybylang/gpubackend` keeps the import surface smaller.

The immediate `gpu*` API accepts/returns CPU `seq`s, so each call may transfer data. Available families include `gpuMatmul`, `gpuMatmul2`, `gpuMatmulQ4`, vector ops (`gpuRelu`, `gpuSigmoid`, `gpuTanh`, `gpuFusedAddAct`), `gpuSoftmax`, `gpuLayernorm` and backward, fused attention and backward, embedding lookup, RoPE-related session operations, and utility ops (`gpuGather`, `gpuCompare`, `gpuPermute`, `gpuSplitSlice`, `gpuConcat`, `gpuArgmaxIndex`, `gpuDropoutMask`). See the exported signatures in `gpubackend.nim` for exact dimensions and argument order.

For a chain of GPU operations, use `SessionHandle` so intermediate tensors stay on the device. Always check encoder return values and end the session before reading results:

```nim
import vendor/bybylang/gpubackend

let backend = parseBackend("vulkan")
if not sessionBegin(backend):
  raise newException(IOError, "GPU sessions are unavailable")
var a = sessionUpload(backend, @[1'f32, 2, 3, 4])
var b = sessionUpload(backend, @[10'f32, 20, 30, 40])
var c: SessionHandle
if not sessionVecOp(backend, "add", a, b, c, 4):
  raise newException(IOError, "add encode failed")
if not sessionEnd(backend):
  raise newException(IOError, sessionGetLastError(backend))
echo sessionRead(backend, c, 4)
sessionFree(backend, a)
sessionFree(backend, b)
sessionFree(backend, c)
```

Core lifecycle calls: `sessionBegin`, `sessionUpload`/`sessionUploadBorrowed`/`sessionUploadPrivate`, `sessionAllocScratch`, `sessionEnd`, `sessionRead`, and `sessionFree`. Encode APIs cover vector/activation ops, matmul and Q4 matmul, softmax, layernorm/RMSNorm, RoPE, attention, copy/scale, gather/compare, dropout, slicing, and embedding where implemented by the chosen backend. `sessionCommitContinue` can commit queued work and continue encoding without an explicit CPU readback; `sessionRecoverQueue` is for recovery after a failed session. `sessionAllocHeap` is currently a Metal-only heap optimization and returns `nil` on unsupported backends; check it before suballocating.

For quantized Q4 weights, upload once and reuse the returned resident handle across sessions:

```nim
var q4 = residentUploadQ4(backend, packedWeights, scales, zeros,
                          k = 4096, n = 4096, groupSize = 32,
                          nGroupsPerRow = 128)
discard residentPrepareQ4(q4, M = 1) # optional autotune; false means no tuned plan
if not sessionBegin(backend): raise newException(IOError, "session unavailable")
var x = sessionUpload(backend, activations)
var y: SessionHandle
if not sessionMatmulQ4(backend, x, q4, y, M = 1):
  raise newException(IOError, "Q4 matmul encode failed")
if not sessionEnd(backend): raise newException(IOError, sessionGetLastError(backend))
let result = sessionRead(backend, y, 4096)
sessionFree(backend, x)
sessionFree(backend, y)
residentFreeQ4(q4) # call only when unloading the weights
```

The old `cu*` resident-tensor names are compatibility stubs in this revision; they do not provide CUDA/cuBLAS execution. Use the common Session API above. The legacy backend string `cuda` resolves to Vulkan.


## 5. Native BybyLang micro-kernels

Write a `micro kernel` block directly in `.bybylang`; no Nim helper module or external micro-kernel import is required. The compiler builds a `VmProgram`, lowers it to Metal or Vulkan code, dispatches it in a GPU session, and reads back the output:

```text
gpu backend is "vulkan"
gpu array INPUT = [1, 2, 3, 4, 5, 6, 7, 8]

micro kernel scale INPUT -> OUTPUT threads 8 tg 4 out 8:
  emitUTgid 0
  emitUMulK 0 0 4
  emitULid 1
  emitUAdd 0 0 1
  emitULoadIn 1 0 0
  emitLoadConstF32 2 2.0'f32
  emitMul 1 1 2
  emitUStoreF32 1 0 0
  emitHalt

print OUTPUT
```

The VM instruction API is exported by `vendor/bybylang/micro_kernel` and re-exported by `vendor/bybylang/bybylang`: create a `VmProgram` and append instructions with `emit*` procedures. Instructions include float/integer loads and stores, arithmetic, FMA, loops and conditionals, thread/threadgroup IDs, threadgroup memory and barriers, atomics, and virtual-threadgroup helpers. Integer registers are 32-bit; float registers are `float32`. Each instruction takes register indices and offsets as documented at its declaration in `micro_kernel/session_vm.nim`.

The DSL header is `micro kernel NAME INPUT -> OUTPUT threads N tg N out N:` and its indented body accepts `emit...` instruction calls. `threads` is the dispatch thread count, `tg` the threadgroup size, and `out` the output element count. Choose counts and guard memory accesses so every dispatched thread stays in bounds. This path currently requires Metal or Vulkan; it raises for CPU/unsupported backends. For more complex kernels, import `vendor/bybylang/bybylang`, create the `VmProgram` in Nim, and call `executeMicroKernel(backend, kernelName, program, input, outputElems, threads, tgSize)` directly.

---


## 6. Low-level hardware-simulation commands

These **do** have real DSL syntax, parsed in `genBlock` (`bybylang.nim` ~line 668–714) and lowered to calls that operate on a simulated RAM (1024 ints) / BUS (seq[string]) / 32 Pins:

```
apu tran "chip1" with 101010          # -> apuTran("chip1", 101010)
apu mem write RAM0 with 42            # -> apuMem("write", "RAM0", "42")
apu mem read RAM0 with 0              # -> apuMem("read", "RAM0", "0")
apu core run                          # -> apuCore(1, "run")  (mode is always hardcoded to 1)
apu pin 3 is high                     # -> apuPin(3, "high")
bit send 1010                         # -> bitSend("1010")
bit recv                              # -> bitRecv()
mem map "device0"                     # -> memMap("device0")
mem push RAM0 with 99                 # -> memPush("RAM0", "99")
tran pulse pin 3 width 10ns           # -> tranPulse(3, "10ns")
```

Parser quirks worth knowing:
- `apu mem <action> <target> with <value>` — `target` is read positionally as the 4th word (`left[3]`) and quotes are stripped; `action` should be `write` or `read`.
- `apu core ...` ignores everything after `apu core` — it always emits `apuCore(1, "run")`.
- `apu pin <n> is <state>` reads `n` from word index 2 and `state` from word index 4 — extra or missing words will misparse silently.
- `tran pulse pin <n> width <w>` reads `n` from word index 3 and `w` as the **last** word on the line.

`apuTran`, `apuMem`, `apuCore`, `apuPin`, `bitSend`, `bitRecv`, `memMap`, `memPush`, `tranPulse` show up as "declared but not used" warnings only when compiling `bybylang.nim` itself (the compiler for the DSL) — that warning is irrelevant to whether the DSL syntax works. The AOT-generated output (`--aot=...nim`) re-declares its own copies of these same procs (see `bybylang.nim` ~line 775 onward) so the compiled program can actually call them at runtime.

---


## 7. Backends

| Backend | File | Notes |
|---|---|---|
| Metal | `backends/metal/metal_backend.nim`, `backends/metal/kernels/` | macOS GPU; native buffer, pipeline, and session support |
| Vulkan | `backends/vulkan/vulkan_driver.nim`, `backends/vulkan/kernels/` | Vulkan GPU backend for NVIDIA, AMD, and Intel devices where drivers/features support the operation |
| CPU | reference operations in `gpubackend.nim` | fallback or explicit CPU implementation where implemented; can be forbidden with `setForbidCpuFallback(true)` |

`cuda` is accepted as a compatibility spelling and normalized to Vulkan. It does not load CUDA Runtime or cuBLAS. OpenCL is not implemented as a current selectable backend.


## 8. Minimal example

```
gpu backend is "vulkan"
gpu array A = [1, 2, 3, 4]
gpu array B = [5, 6, 7, 8]
gpu add A, B -> C
print C
```
