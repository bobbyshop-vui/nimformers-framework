# BybyLang

A Nim-based DSL, AOT-compiled to Nim source then built into a native binary. Its core is a set of `gpu ...` commands for running tensor ops on CUDA / Metal / OpenCL / TSIC-IR / CPU, plus generic control flow and a set of low-level hardware-simulation commands.

## Build & run

```bash
make build #build library don't run that test code
make test
# equivalent to:
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
- `function NAME` ... a line containing only `NAME` closes the definition; call it with `call NAME`.
- `if cond:` / `elif cond:` / `else:` / `while cond:` / `for x in range(a, b):` — translated directly to the equivalent Nim construct, arbitrarily nestable.
- `mode is N` (N = 1..4) → prints "Mode 1: Low-level" / "Mode 2: Mid-level" / "Mode 3: High-level" / "Mode 4: Web-level" (any other N → "Unknown mode").

## 2. `gpu ...` commands

### Select backend

```
gpu backend is "auto"      # auto | cpu | cuda | metal | opencl | tsic
```

`auto` probes in order: CUDA (NVIDIA) → Metal (macOS) → OpenCL → plain CPU.
By default, **silent CPU fallback is forbidden** (`gForbidCpuFallback = true`): if a specific GPU backend is requested and it's unavailable or fails, the program raises instead of silently falling back to CPU (so you never accidentally train on CPU without noticing).

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

```
gpu matmul2 A1, B1, A2, B2 -> C1, C2 m1 M k1 K n1 N m2 M k2 K n2 N
```
Two independent matmuls fused into a single call.

### Activations

```
gpu relu X -> Y
gpu sigmoid X -> Y
gpu tanh X -> Y
gpu apflu X -> Y alpha A beta B      # defaults: alpha=0.1, beta=1.0 if omitted
gpu apflu_backward X, DY -> DX alpha A beta B   # alpha/beta optional, same defaults
```

### Fused add + activation

```
gpu fused_add_act A, B -> C act "relu"    # "relu" | "sigmoid" | "tanh" | "none"
```

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

### Generic op (fallback)

```
gpu <other_op> A, B -> C
```
Generates `gpuOp("<other_op>", backend, A, B)` — requires a matching branch inside `gpuOp` to actually do something.

---

## 3. GPU Resident Tensor API (CUDA) — called directly from Nim

Defined at the end of `gpubackend.nim`. **Not** exposed through `.bybylang` syntax — call it from plain Nim code to avoid a CPU↔GPU round trip on every op:

```nim
let ta = cuUpload(dataA)          # upload seq[float32] to the GPU once
let tb = cuUpload(dataB)
let tc = cuAddR(ta, tb)           # result STAYS on the GPU
let out = cuDownload(tc)          # only download when you actually need the result
cuFree(ta); cuFree(tb); cuFree(tc)
```

Available: `cuUpload`, `cuUploadIndices`, `cuDownload`, `cuFree`, `cuMatmulR`, `cuAddR`, `cuSubR`, `cuMulR`, `cuDivR`, `cuReluR`, `cuSigmoidR`, `cuTanhR`, `cuSoftmaxR`, `cuLayernormR`, `cuEmbeddingLookupR`.

> ⚠️ **Note:** these resident-tensor calls exist but the forward pass generated from `.bybylang` (`genGpuLine`) does not call them — every `gpu ...` DSL command still uploads/downloads through `seq[float32]` on each call (per-op round trip), not the resident chain.

---

## 4. Low-level hardware-simulation commands

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

## 5. Backends

| Backend | File | Notes |
|---|---|---|
| CUDA | `backends/cuda/cuda_driver.nim`, `cuda_runtime.nim` | Direct Driver API + cuBLAS, PTX JIT via `cuModuleLoadDataEx`, persistent context/module cache |
| Metal | `backends/metal/metal_backend.nim` (+ `.metal` kernels) | macOS GPU, buffer/pipeline cache |
| OpenCL | `backends/opencl/opencl_api.nim` | any other GPU/CPU |
| TSIC-IR | `tsic_ir.nim` | intermediate IR, lowerable to PTX / MSL / OpenCL C / GLSL |
| CPU | inside `gpubackend.nim` (`cpuRelu`, `cpuMatmul`, ...) | pure-Nim fallback, always correct but slow |

## 6. Minimal example

```
gpu backend is "tsic"
gpu array A = [1, 2, 3, 4]
gpu array B = [5, 6, 7, 8]
gpu add A, B -> C
print C
```