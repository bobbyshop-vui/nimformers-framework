# FIR from a core trace

The recorder (`vendor/bybylang/probe.nim`) and the Nim lifter
(`other/formers.n/firlift/`) rebuild a runnable `model.fir` from one recorded
decode step. No Python is on the trace path.

## Record

`BYBY_FIR_PROBE=<path>` enables the recorder; `BYBY_FIR_PROBE_STEPS=decode:2` selects
steps. With the recorder off, every hook is one boolean test. The output is
`fprobe 2`, structure only: buffers, programs, steps, ops and host transfers. No
tensor payload is written. A step is flushed by `probeStepEnd`; a run without
step markers has no output file, so use a runtime that marks its decode steps
(the FIR runner and `nim_inference` do).

```
fprobe 2
backend metal device="unspecified" build="nim-2.2.10" launch="session"
buf b4 bytes=8192 origin=upload name="model.layers.0.input_layernorm.weight" dtype=f32 shape=(2048)
prog p3 kind=vm words=312 hash=56c12e3f82b9c1e8
step decode index=5 position=5 token=8241 tokens=1
op o12 tag=aten::rms_norm in=(x=b4@0:8192,gamma=b9@0:8192) out=(y=b11@0:8192) attrs(rows=1,cols=2048,eps=h3727c5ac) impl=session:sessionRmsnormEnc parent=none prog=none
disp d12 prog=p3 nThreads=32 tgSize=1 tgbytes=16 pipe="jit_..." op=o12
end step
words p3
 00000090 00000191 ...
end
```

- `op` is the semantic tier: ATen tag, named input/output ports
  (`buffer@byteOffset:byteCount`), scalar attributes, implementation
  (`session:<proc>`, `micro:<hash>`, `ext:<op>`), parent op and program.
- `disp` is one micro-kernel launch (CPU VM, Metal JIT or Vulkan dispatch).
- `name` records the checkpoint tensor at upload, so weight names come from the
  load sites. `-scales`/`-zeros` buffers are derived from a q4 tensor.
- Buffer byte extents are set at allocation/upload for CPU, Metal, Vulkan and
  q4 weights. The backend field is the canonical name (`cpu`, `metal`,
  `vulkan`).
- A path containing a newline is rejected, so a log line can never become an
  output file name.

## Lift

```
other/formers.n/firlift/firlift --trace t.fprobe --config config.json --out DIR \
  [--name NAME] [--q4] [--report]
```

The lifter:

1. reads one step and keeps the semantic ops;
2. builds a value per port interval and a buffer per name;
3. folds the repeated layer blocks into `repeat layers`;
4. turns `model.layers.N` weight names into `{i}` templates;
5. replaces concrete counts with config symbols (`hidden`, `layers`, `positions`,
   `vocab`, `nheads`, `nkvheads`, `headdim`, `mlp`, `3*hidden`, ...);
6. writes each recorded micro program back as FIR kernel text, rewriting address
   immediates as `@port + offset`;
7. prints `--report` counts (nodes, micro, ext, kernels, instructions) and writes
   `model.fir` plus a copy of `config.json`.

Two lift modes:

- **Graph trace** (a model already running through the FIR graph executor):
  the recorded node names, ports, weights and kernel programs are used
  directly. GPT-2 125M is a byte-structure round trip: 30 buffers, 19 nodes
  (12 micro, 7 ext), 4 kernels, 123 instructions, and 24 greedy tokens equal to
  the hand-written file.
- **Decoder trace** (the native Llama runtime on TinyLlama): the lifter
  recognises embedding upload, RMSNorm, q/k/v projections, the decode attention
  call, residual adds and the SwiGLU gate. It fuses RoPE + KV-cache write +
  GQA attention into the library attention kernel, drops the staging copies,
  coalesces the per-head KV handles into per-layer caches and fuses the SwiLU
  gate into `script.silu_mul`. The result matches the Python reference export's
  structure (15–16 nodes, 4–5 kernels, 364–445 instructions) and its 24 tokens.

`--q4` emits `q4t` weights for the per-layer `[out,in]` linears. The runner
quantizes them at load and uses the resident Q4 GEMV. Measured on TinyLlama
(Intel Iris Plus 640, Metal): 1.3 tok/s f32 vs 4.0 tok/s q4t for the same graph.
The q4 model is a faster, lossy variant; the default f32 emission is the exact
reference.

## Micro-kernel library

`other/formers.n/firlift/kernel_library.nim` holds the bodies the lifter emits:

| Kernel | Tag | Shape/role |
| --- | --- | --- |
| `script.rmsnorm` | `aten::rms_norm` | 256-lane threadgroup per row; tree reduction |
| `script.add` | `aten::add` | pointwise residual/bias |
| `script.silu_mul` | `aten::silu+aten::mul` | SwiGLU gate, one launch |
| `script.attention` | `aten::scaled_dot_product_attention` | RoPE + KV write + GQA decode; hidden=2048, heads=32, kv=4, head_dim=64, positions=2048 |
| `script.add_rmsnorm` | `aten::add+aten::rms_norm` | library variant kept for models that need 2 outputs |

`other/formers.n/firlift/test_library.nim` assembles the library bodies and runs
them on the production CPU VM against independent float formulas (tolerance
`1e-4`). It covers rmsnorm (serial and 256-lane), add, mul, sigmoid and silu_mul.

## Builder API

`other/formers.n/firlift/builder.nim` provides `BuilderConfig`, `BLinear`,
`BRmsNorm`, `newBuilderModel`, `forwardStep`, `beginBuilderTrace` and
`traceModel`. A model is built from the same library classes, one forward call
records one step, and `firlift` emits the file. `other/formers.n/firlift/builder_test`
(the compile target is shown below) records a 1-layer 2048-wide decoder, lifts
it and passes `--fir-check`.

## Limits

- A raw trace is lifted by the session lifter (below). `--core-idiom` keeps the older
  decoder-idiom lifter, whose attention body is specialized to the TinyLlama dimensions.
- Raw micro-VM launches that the runtime records without ports (`disp` lines on an arena, as the
  Mamba runtimes produce) are not lifted by this stage: they carry no tensor extents. Mamba's
  session SSD op (`sessionMamba2SsdPrefillEnc`) is lowered.
- Host computation between GPU sessions is invisible to the recorder. The one idiom handled is
  the native Llama runtime's final RMSNorm (see the session lifter).
- `q4t` is script-layout only and applies to `fir.linear_t`; `fir.linear`
  keeps `q4` ([in,out]).
- Ext linears still stage the arena input through a scratch handle; removing
  those copies needs an input-offset session API.

## Commands

```sh
make -C other/formers.n build
other/formers.n/formers-cli --models models/gpt2-125m --fir /tmp/gpt2-lifted/model.fir \
  -p "The capital of France is" -n 24 --temp 0 --backend cpu

# record a TinyLlama decode step with the native runtime
BYBY_FIR_PROBE=/tmp/tl.fprobe BYBY_FIR_PROBE_STEPS=decode:1 \
  ./nim_inference models/tinyllama models/tinyllama/model.nimq metal "Hi" 2 0 1.0
other/formers.n/firlift/firlift --trace /tmp/tl.fprobe \
  --config models/tinyllama/config.json --out /tmp/tl --report
other/formers.n/formers-cli --fir-check /tmp/tl/model.fir

# raw core trace -> generated micro-kernels (every session op)
other/formers.n/firlift/firlift --trace /tmp/tl.fprobe \
  --config models/tinyllama/config.json --out /tmp/tl-sess --report
other/formers.n/formers-cli --models models/tinyllama -m models/tinyllama/model.safetensors \
  --fir /tmp/tl-sess/model.fir -p "The capital of France is" -n 24 --temp 0 --backend metal

# builder
nim c -d:release --path:vendor/bybylang --nimcache:nimcache/builder-test \
  -o:/tmp/builder_test other/formers.n/firlift/builder_test.nim 2>/dev/null || true
```

## Session-op kernels

`other/formers.n/firlift/kernel_gen.nim` generates one micro-kernel body per session
op, from the dimensions of the call and with the call's argument order as port order.
The bodies use only `session_vm` mnemonics, so the same text runs on the CPU VM, the
Metal JIT and the Vulkan VM. A kernel that updates a buffer in place (KV cache, SSM
state, accumulated gradients) writes through its input port and lists that buffer as an
output too.

| Session op | Generator | Ports |
| --- | --- | --- |
| `sessionVecOp` | `genVecOp` | a, b -> c |
| `sessionActivation` | `genActivation` (relu, silu, gelu variants, tanh, sigmoid, pow, ...) | x -> y |
| `sessionScale` | `genScale` | x -> x |
| `sessionSoftmax` | `genSoftmax` | x -> y |
| `sessionRmsnormEnc` / `sessionLayernormEnc` | `genRmsnorm` / `genLayernorm` | x, gamma[, beta] -> y |
| `sessionRope` (one token, tables from the host) | `genRopePos` | x, position -> x |
| `sessionRope` (tables in the graph, several tokens) | `genRope` | x, cos, sin -> x |
| `sessionMatmulNT` / `sessionMatmul` (+ `...Accelerated`, `...Fp16Coop`, `...TiledEnc`) | `genMatmulNT` / `genMatmul` | a, b -> c |
| `sessionMatmulQ4` | `genMatmulQ4` (`packed`: one `q4b` weight) | a, w -> c |
| `sessionFp4GemmTiledEnc` / `sessionFp4GemmTensorCore` | `genFp4Gemm` (`fp4km` weight; the table is part of it) | a, w -> c |
| `sessionGatherEnc` / `sessionEmbeddingLookupEnc` | `genGather` / `genEmbedding` | table, indices -> y |
| `sessionKvScatterEnc` (and the per-head `sessionCopyInto` run it replaces) | `genKvScatter` | src, position, cache -> cache |
| `sessionCopyInto` | `genCopyOffset` | src -> dst (partial) |
| `sessionAttentionDecodeAllHeadsEnc` | `genAttentionDecode` (count = position + 1) | q, K, V, position -> o |
| `sessionAttentionDecodeGqaEnc` | `genAttentionDecodeGqa` | q, K, V -> o, scores |
| `sessionAttentionFusedEnc` / `...BackwardEnc` | `genAttentionFused` / `genAttentionFusedBackward` | see generator |
| `sessionFlashAttentionPrefillEnc` | `genFlashPrefill` | q, K, V, position -> o |
| `sessionMamba2SsdPrefillEnc` | `genMamba2Ssd` + `rebasePorts` (regions of one arena) | arena -> arena |
| `sessionCompareEnc`, `sessionArgmaxIndexEnc`, `sessionSplitSliceEnc`, `sessionDropoutMaskEnc` | `genCompare`, `genArgmax`, `genSplitSlice`, `genDropoutMask` | as the call |
| `sessionPermuteEnc` | `genPermute` (the recorder keeps the short stride tables) | src -> dst |
| `sessionApfluEnc` / `...BackwardEnc` | `genApflu` / `genApfluBackward` | x[, dy] -> y |
| `sessionLayernormBackwardEnc` | `genLayernormBackwardDx` + `genLayernormBackwardParams` | dy, x, gamma -> dx; dy, x -> dgamma, dbeta |

Opt-in threadgroup-parallel variants `genMatmulQ4Par` / `genMatmulNTPar` split the K
reduction over T lanes with a tree reduction (`FIRLIFT_TG=16|32|64`). On Intel Iris Plus
640 they were slower per token than the lane-per-column kernels, so they are off by default.

`test_library` checks every generator on the CPU VM against a scalar reference.
`test_session_gpu` runs each generated body on the GPU backend and compares it with
the real session call on the same inputs.

## Session lifter

`firlift` on a trace that has no graph nodes runs `sessionlift.nim`:

1. takes the last recorded step; every `session*` op becomes one node running the
   generated kernel (deduplicated by text and port sizes, named `s.<op>`);
2. buffers are named by the trace buffer they stand for; a write creates a fresh FIR
   buffer, except state updated in place (the KV cache keeps one buffer per layer and is
   sized `kv_heads * positions * head_dim` from the config);
3. weights come from the registered names: Q4 matrices become `q4b<group>` (the group of
   the call, so a native 128-group `.nimq` checkpoint is quantized with 128), f32 matrices
   used by a kernel become `f32a`, vectors stay `f32`;
4. host uploads are classified by their consumer: the first upload of the step is the token
   embedding (`fir.embedding`), RoPE cos/sin tables become a position-driven RoPE kernel
   (frequencies from `rope_theta` / `rope_scaling`), anything else is host data and the ops
   reading it are dropped;
5. dynamic scalars (KV write row, attention length) are checked against the traced
   position and read from the `position` buffer;
6. nodes whose results never reach the logits are dropped (the per-head KV copies kept for
   prefill).

The native Llama runtime computes its final RMSNorm on the host (GPU read, two vector
multiplies by uploaded data, upload). The lifter recognises that round trip — a GPU read,
then an upload straight after a read of host-derived data — and replaces it with one
`rmsnorm` over the checkpoint's final norm weight.

TinyLlama 1.1B (Q4, group 128, 22 layers) lifts to 334 nodes and 14 kernels (581
instructions) after the peephole fusions (SwiGLU gate, residual add + RMSNorm, RoPE + query
scale; `FIRLIFT_NOFUSE=1` turns them off; 443 nodes). On Metal it continues "The capital of France is" with `Paris.` followed by a coherent
list, like the native runtime's first token, at roughly 2-3 tok/s against 9 tok/s native (measured on a loaded machine; fused and
unfused graphs were indistinguishable, so the lane-per-column matmul kernels dominate).

`firlift --coverage` lowers every op of a trace on its own, tolerates unnamed weights and
host inputs, and fails when an op has no mapping. `make fir-lift-test` runs it on the trace
that `test_session_gpu` records, which exercises every op family the Metal backend has.
