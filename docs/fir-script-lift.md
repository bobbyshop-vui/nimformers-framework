# FIR from a core trace

The recorder (`vendor/bybylang/probe.nim`) and the Nim lifter
(`other/formers.n/firlift/`) rebuild a runnable `model.fir` from one recorded
decode step. No Python is on the trace path.

> Recording a trace from code (Python, Nim, BybyLang, C) is covered by the [Trace API](trace-api.md); this page covers what the lifter does with it.

## Record

`BYBY_FIR_PROBE=<path>` enables the recorder; `BYBY_FIR_PROBE_STEPS=decode:2` selects
steps. With the recorder off, every hook is one boolean test. The output is
`fprobe 2`: buffers, programs, optional steps, ops and host transfers. Snapshots
are flushed periodically and at exit, including runs without step markers.
Core CPU/Metal/Vulkan uploads record a versioned sampled fingerprint (size plus
up to five 64-byte blocks). Set `BYBY_FIR_PROBE_DATA=1` to also stream exact host
upload bytes to `<trace>.data`. This can use checkpoint-sized disk space. It
preserves unnamed LUTs, adapters and composite arenas without model hooks;
fingerprints alone cannot reconstruct these auxiliary values. The recorder adds no
GPU readback of its own. A runtime that rewrites device buffers after uploading
them (the Mamba-3 in-place LoRA merge) reads them back once, only while the
recorder is on, and records them as a fresh upload. Keep the trace and its
sidecar together.
`BYBY_FIR_PROBE_DATA_MAX=0` (the default) leaves capture unlimited; a positive
byte cap omits larger payloads while retaining their fingerprints.

### Record without files

`BYBY_FIR_PROBE=fd:T[,D]` sends the recorder's output down inherited pipe write ends instead
of files: the finished trace text goes to descriptor `T` once, when the recorder stops, and
the exact upload bytes (with `BYBY_FIR_PROBE_DATA=1`) stream to descriptor `D`. A payload
that is uploaded again is sent once and referenced by the same `capture=` offset. Mid-run
snapshots are not written in this mode. `firlift --run` sets this up: it creates both
pipes, starts the command with the environment above, reads the pipes until the command
ends and lifts from memory. No `.fprobe` or `.data` file exists at any point; the only
files are `model.fir` and the compacted `trace.data` in `--out`.

```sh
BYBY_FIR_PROBE_STEPS=prefill:1,decode:1 \
other/formers.n/firlift/firlift --config CONFIG --weights model.nimq --out DIR --report \
  --run other/formers.n/formers-cli --models MODELS -m model.nimq --raw -p "$PROMPT" -n 4 --temp 0
```

`--run` and everything after it is the command (put it last); `--trace` is not allowed with
it. `BYBY_FIR_PROBE_DATA` defaults to `1` there. The upload bytes are held in memory until
the lift ends, so they count against RAM: set `BYBY_FIR_PROBE_DATA_MAX` to leave out
checkpoint-sized uploads that `--weights` can match by fingerprint. The command must be built
with the pipe-aware recorder; an older binary treats `fd:T,D` as a file name and writes
`fd:T,D` and `fd:T,D.data` into the working directory.

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
- Optional `name` records preserve names supplied by an existing caller.
  Unmodified runtimes need no naming hooks: the raw path uses core upload
  fingerprints and `--weights` instead. `-scales`/`-zeros` named buffers may be
  derived from a q4 tensor.
- Buffer byte extents are set at allocation/upload for CPU, Metal, Vulkan and
  q4 weights. The backend field is the canonical name (`cpu`, `metal`,
  `vulkan`).
- A path containing a newline is rejected, so a log line can never become an
  output file name.

## Lift

```
other/formers.n/firlift/firlift --trace t.fprobe --config config.json --out DIR \
  [--weights model.nimq]... [--slice A:B | --step N] [--context TOKENS] [--embedding TENSOR] [--report]
```

`--context TOKENS` lowers context-sized KV buffers and position-dependent kernels to that
capacity and writes the matching context into the output `config.json`. Use it when the source
model declares a much larger maximum context than the runner needs (for example, lift DeepSeek
with `--context 512` before running it with `-c 512`). The requested capacity cannot exceed
the source config's context length.

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
- Raw dispatches with program words and complete slot extents are lifted as
  described below. Historical zero-sized or opaque uploads cannot be recovered
  retrospectively. Missing identities or initialization data fail explicitly.
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
The Q4 GEMV `word_parallel` variant is subgroup-independent and loads packed weights as aligned
32-bit words before reducing across the threadgroup. It supports K divisible by 8 and a power-of-two
quantization group size of at least 8 that divides K, with `nGroups == K / groupSize` and a useful
threadgroup width of at least two lanes for the selected lane limit. Other shapes retain the byte-load
parallel kernel. Vulkan auto selection uses this variant when supported. No measured speedup is
implied.

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
`test_steps` builds synthetic traces and checks which earlier passes are replayed with the
selected decode step: the state clear that is left to the graph's initial contents (and the
cases where it must stay: a write before it, a non-zero source, a session that also computes)
and the single-token units that the decode step covers.

## Unmarked traces and raw slot kernels

Unmarked traces are selected by repeating sequences of complete session
signatures (semantic op kinds and dispatch pipeline kinds, without JIT hash
prefixes). Units may span multiple sessions. The default chooses the first
occurrence of the repetition with the greatest repeated work; incomplete trailing
sessions are excluded. `--step N` selects another occurrence. `--slice A:B`
selects zero-based complete sessions, with B exclusive. Ambiguous nonrepeating
recordings require an explicit slice. Preceding host uploads and trailing reads
are included. No model name, kernel kind, layer count or fixed dispatch count is
used by the selector.

Each raw launch emits a `micro` node and a kernel with `layout=slots`, for example:

```text
kernel raw0 in=2 out=1 threads=4 tg=4 tgbytes=16 layout=slots slots=mem:in0,input:in1
words 00000081 0000018d 00000000 0000018e 00000000 00000001
endkernel
```

The hex words, including HALT, are copied verbatim. Static instruction scanning
classifies reads/writes of `mem`, `u8`, `lut`, `input` and synchronization storage.
Programs with equal words, geometry and domain ports share one definition.
`view NAME BASE OFFSET COUNT` preserves word-aligned subrange aliases; offsets
and counts are float words. Mutable physical buffers stay in place across steps.
Mixed semantic session ops use these same physical buffers. Elimination is
conservative: all writes, including persistent state writes, are retained.
Existing kernels keep their arena/script assembly behavior.
Read-only raw weights used exclusively by slot kernels have separate resident
allocations, loaded one buffer at a time. They are excluded from the shared
activation/state arena; raw weights read by generated arena kernels stay there.
The CPU VM also executes independent buffer views, so tests cover this storage
path without a GPU. This avoids making a multi-GB checkpoint one Metal buffer.

`--weights` indexes NIMQ2/NIMQ3, raw-FP32 NIMBIN or single-file safetensors headers
without loading the checkpoint. Repeat the option for auxiliary weight files;
the first file is the primary checkpoint and embedding source. Raw uploads match
payload size and sampled fingerprint; a unique size is a legacy fallback, while
ambiguous sizes fail. The runner's embedding loader supports `.nimq` and
safetensors; NIMBIN files are auxiliary raw payload sources. Payload samples are read
only when a fingerprint is available. `weight NAME raw TENSOR tensor [WORDS]`
plus `init NAME checkpoint OFFSET BYTES [FINGERPRINT]` loads the original packed
bytes, without dequantizing or requantizing them. Unmatched auxiliary uploads
require captured bytes (`init NAME trace.data OFFSET BYTES [FINGERPRINT]`);
short recorded tables may use `initwords`. File extents and recorded fingerprints
are checked before execution. The lifter copies a needed sidecar as `DIR/trace.data`
and links auxiliary checkpoint files as `BASENAME.raw` (copying if a symlink is unavailable).
Allocation padding beyond the uploaded prefix is zero-filled; kernels must not
depend on uninitialized storage.

A single hidden-sized per-step upload becomes `fir.embedding`. Its tensor is
selected by the checkpoint shape `[vocab,hidden]`, or `--embedding NAME` when
ambiguous. The graph exposes the runner's token, position and logits roles. Native
NIMQ embeddings are loaded selectively, one tensor at a time. Unrecognized host
uploads inside later selected units are rejected; startup captures are replayed
as initializers. Arbitrary host calculations and control flow have no general
reconstruction rule. State initializers restore on
`firGraphReset`; this does not reconstruct initialization performed only by GPU
work outside the selected unit.

The logits role goes on the buffer or view that the vocabulary-sized read covers. When that read
covers only the first `vocab` floats of a longer scratch buffer (the Mamba-3 runtime reads 128,256
floats from a 128,260-float scratch), the role sits on a `view` that no node names. The lifter emits
it anyway, as `view NAME BASE 0 VOCAB output:logits`, and the runner copies the vocabulary through it.

The production CPU VM verifies separate slots, a resident raw weight outside
the arena, shared subranges, three raw kernels mixed with copy/add session ops,
state across steps and reset, plus
per-token embedding uploads with an arbitrary checkpoint tensor name and a logits read over the
first vocabulary-sized part of a longer scratch buffer. The
fixture compares output bits with the original recorded CPU runs. GPU slot
execution, full-model tokens and performance have not been verified here.
CPU slot execution rejects virtual threadgroups, device grid control, and a
distinct MemU32 atomic buffer (it must alias MemF32); it retains the existing VM
thread/step limits. Unaligned byte views are rejected.
Recorded program constants and geometry remain fixed: automatic selection rejects
changes to either across repeating units. Position-dependent program regeneration
or host-side shape changes need a dynamic lowering rule. An explicit slice is
only a fixed-launch replay in that case. The selector is a heuristic, so inspect its
reported range or use an override.

The supplied legacy `cs2.fprobe` was recorded before the core upload hook knew
buffer sizes, so every packed weight binding has zero bytes and no fingerprint;
it cannot become a runnable model. Record again with the updated core:

```sh
BYBY_MODEL_MMAP=1 BYBY_MERGE_LORA=0 BYBY_ONE_DISPATCH=0 \
BYBY_FIR_PROBE="$TRACE" BYBY_FIR_PROBE_DATA=1 BYBY_FIR_PROBE_DATA_MAX=16777216 \
  timeout 420 ./infer_lora_bin test/codestral/models/jamba_parallel_lora.bin \
  "hi" 0 test/codestral/models/hf test/codestral/models/codestral_mamba_only.nimq
```

`BYBY_FIR_PROBE_DATA_MAX` caps the exact-bytes sidecar at 16 MiB per upload:
state arenas (4.8-11 MB) and adapters are kept, checkpoint-sized packed weights
are only fingerprinted. The cap is a recorder-only option; with the probe off it
is a single environment lookup that is never executed.

Lift with the primary NIMQ checkpoint and, when the model stores trained tensors
outside it, one or more auxiliary checkpoints. Every `--weights` file is indexed
(NIMQ2/3, safetensors or raw-fp32 NIMBIN), and each raw upload is matched by
exact byte size plus sampled content fingerprint. Ambiguous or missing matches
fails with the buffer name; nothing is guessed:

```sh
other/formers.n/firlift/firlift --trace "$TRACE" --config "$CONFIG" \
  --weights test/codestral/models/codestral_mamba_only.nimq \
  --weights test/codestral/models/jamba_parallel_lora.bin \
  --out models/codestral-fir --report
other/formers.n/formers-cli --fir-check models/codestral-fir/model.fir
```

The output directory is self-contained apart from explicit symlinks:

```text
model.fir            lifted graph (slot kernels + exact session-replay nodes)
config.json          copied from --config
trace.data           only the referenced sidecar ranges, deduplicated
jamba_parallel_lora.bin.raw   symlink to the auxiliary checkpoint
model.nimq           symlink to the primary checkpoint
tokenizer.model      SentencePiece model, tokenizer_config.json
```

The `.raw` suffix keeps the auxiliary checkpoint from being mistaken for a LoRA
adapter by model discovery. The tokenizer directory deliberately contains only
`tokenizer.model` and `tokenizer_config.json`: the full HuggingFace directory's
`tokenizer.json` maps `[INST]`/`[/INST]` to added special tokens, while the
native baseline tokenizes them as ordinary BPE pieces.

## Measured Codestral result

`firlift` selects session `243:244` (605 dispatches). The lift contains 960
buffers, 605 raw slot nodes, 158 exact session-replay nodes, one embedding node
and 16 kernel definitions. `--fir-check` passes, the model loads on Metal and
generates at about 0.25-1.0 tok/s.

Decode replay itself matches the native runtime: the first three greedy tokens
are identical (`781, 781, 29560`, the same top-5 order). The step-0 logits differ
by 0.2-0.7 (native `781:8.655`, lifted `781:9.396`) because the native prompt
uses its batched T=8 prefill path while a decode-only graph feeds the prompt
through decode kernels. At token 4 that small state difference flips a 0.18-logit
margin (`17057 "INST"` vs `3713 "INT"`), after which the text diverges from the
baseline. Exact 60-token equality therefore requires replaying the prefill unit
(the 16 prefill group sessions in the trace) before the decode unit; that is not
implemented. The decode-only graph must not be claimed token-equivalent.

The previous all-zero-weight failure and the missing auxiliary data are gone:
the fresh trace records weight sizes, full-upload fingerprints and exact bytes
for every upload below the cap.

## Tracing the Llama runtime (fused decode)

`nim_inference` marks every prefill chunk and decode token as a probe step, gives each
quantized linear and RMSNorm weight its checkpoint tensor name at upload, and records the
fused Metal ops (`sessionRopeKvEnc`, `sessionKvAppendEnc`). While the recorder is on, the
fused matmul, the GPU token pipeline and the decode graph are skipped, so the trace holds
the plain ATen-level ops. Do not pass `--weights` (it selects the raw-slot lift).

```sh
BYBY_FIR_PROBE=/tmp/tl.fprobe BYBY_FIR_PROBE_STEPS=decode:1 \
  other/formers.n/formers-cli --models models/tinyllama -p "Hi" -n 2 --temp 0 --no-progress --backend metal
other/formers.n/firlift/firlift --trace /tmp/tl.fprobe --config models/tinyllama/config.json --out /tmp/tl --report
other/formers.n/formers-cli --models models/tinyllama -m models/tinyllama/model.safetensors \
  --fir /tmp/tl/model.fir --raw -p "The capital of France is" -n 14 --temp 0 --backend metal
```

### Kernels that follow the Metal ones

The lifter emits kernels built the way the Metal ones are:

| Op | Generator | Structure |
| --- | --- | --- |
| Q4 GEMV (M = 1) | `genMatmulQ4Simd` | one simdgroup per 4 rows, `simd_sum` |
| Q4 GEMV (M = 1, Vulkan auto / explicit) | `genMatmulQ4WordPar` | aligned 32-bit packed-weight loads and a threadgroup tree reduction; falls back to byte-load parallel for unsupported K/group shapes or a one-lane reduction |
| Q4 scalar / parallel GEMM | `genMatmulQ4`, `genMatmulQ4Par` | one lane per output with serial K reduction, or a threadgroup split over K; each has separate multiply-add and fused FMA forms |
| Q4 GEMM (M >= 16, Intel/AMD default) | `genMatmulQ4Gemm` | the `matmul_q4_gemm*_kernel` Metal kernels: an output tile per threadgroup, K step 32 through threadgroup memory, a register tile per thread; no `simdgroup_matrix`. `FIRLIFT_GEMM_TILE=32x64\|64x64\|32x128\|64x128` (default 64x128) picks the output tile and `FIRLIFT_GEMM_REG=4x4\|8x4\|4x8\|8x8` (default 8x8) the register tile, matching native Metal's default. A 64-row tile decodes each weight once per 64 rows; use the overrides to tune other devices |
| Q4 GEMM (M > 1, `FIRLIFT_TARGET=apple`) | `genMatmulQ4Mm` | 32 x 32 tile, 4 simdgroups, 8 x 8 `simdgroup_matrix` (the `matmul_q4_mm_sg` kernel; Metal only) |
| RMSNorm, add + RMSNorm | `genRmsnormPar`, `genAddRmsnormPar` | one threadgroup per row |
| decode attention | `genAttentionDecodePar` | one threadgroup per head, scores in threadgroup memory |
| flash prefill attention | `genFlashPrefillTiled` | the `attention_prefill_flash_kernel` Metal kernel: one threadgroup per 16 query rows of a head (128 threads, 8 per row), query rows staged once and keys/values 16 at a time in threadgroup memory, online softmax. Needs `head_dim % 8 == 0` and `head_dim <= 128`, otherwise one threadgroup per (row, head). `FIRLIFT_FLASH=row` selects the per-row kernel |
| RoPE | `genRopePosPar` | one thread per (row, head, frequency) |
| KV scatter | `genKvScatterPar` | one thread per element |
| f32 GEMV, `matmul_nt` (M = 1) | `genMatmulNTSimd` | one simdgroup per 4 rows, `simd_sum` |
| f32 GEMM, `matmul_nt` (M > 1, `FIRLIFT_TARGET=apple`) | `genMatmulNTMm` | 32 x 32 tile, 4 simdgroups, 8 x 8 `simdgroup_matrix` |

Choose the Q4 implementation at lift time. `FIRLIFT_Q4_GEMV` accepts `auto`, `scalar`, `scalar_fma`
(`fma` is an alias), `parallel`, `parallel_fma`, `simd` or `word_parallel`. Vulkan `auto` uses
`word_parallel` for supported shapes with K up to 4096 by default; larger shapes retain the byte-load
kernel. The limit reflects local Iris measurements: word loads were 11–12% faster at K=4096 and 32%
slower at K=11008. Set `FIRLIFT_Q4_WORD_MAX_K` to tune the auto limit per device (`0` disables the
word kernel); explicit `word_parallel` ignores this limit and uses the same shape checks on any
target. Metal keeps its SIMD path and generic kernels keep their existing selection.
`FIRLIFT_Q4_GEMM` accepts `auto`, `scalar`,
`scalar_fma`, `parallel`, `parallel_fma`, `tiled_fma` (`tiled` and `fma` are aliases), or `matrix`
(`mm` is an alias). The scalar forms use one lane per output and serial K reduction; the parallel forms
split K across lanes; tiled FMA uses the register-blocked Q4 GEMM; matrix uses `simdgroup_matrix`.
`auto` keeps target-aware selection. The chosen form is written into the generated `.fir`; re-lift to
change it. Kernel names include the selected form, such as `matmul_q4_tiled_fma` or
`matmul_q4_parallel_fma` or `matmul_q4_word_parallel`. If a tiled or matrix generator cannot handle
the weight shape, it emits its lane-parallel fallback, and the kernel name records the fallback that
was actually generated. Unsupported `word_parallel` shapes use the byte-load generator and record its
actual `parallel` or `scalar` variant.

Use `FIRLIFT_Q4_LANES=16|32|64` (default 32) to select the threadgroup width for explicitly selected
parallel forms; when unset it falls back to `FIRLIFT_TG`. The Q4 GEMV SIMD tuner remains available per
device: `FIRLIFT_Q4_ROWS` (1, 2, 4, 8 rows per simdgroup, default 4), `FIRLIFT_Q4_SG` (1, 2, 4, 8
simdgroups per threadgroup, default 2) and `FIRLIFT_Q4_PACKS` (1 or 2 eight-weight packs per lane step,
default 1). GEMV `simd` falls back to the lane-parallel form if its shape constraints do not hold.

For a direct runtime comparison, `BYBY_FIR_Q4_BACKEND=native` routes `q4r` slot nodes through the
same resident `sessionMatmulQ4` dispatcher as native inference. The default `micro` mode keeps the
kernel selected and serialized by the lifter. Native mode loads the packed weights, scales and zero
points into separate resident Q4 buffers and skips JIT for only the Q4 slot nodes it replaces; the
other FIR kernels remain unchanged. The same graph file works in either mode.

To compare the previous byte-load parallel kernel, `word_parallel`, and native Q4 GEMV on Vulkan,
run this benchmark from the repository root:

```sh
BACKEND=vulkan nim c -r -d:release --hints:off --path:vendor/bybylang \
  --nimcache:/tmp/firlift-q4-bench-cache -o:/tmp/firlift-q4-bench \
  other/formers.n/firlift/bench_q4_gemv.nim
```

It checks both generated kernels against native Q4 before reporting median dispatch times. Uploads and
JIT warmup happen before samples; each timed sample includes `sessionBegin`/`sessionEnd` around four
dispatches. The output reports timings and numerical error for the tested shapes; results depend on
the Vulkan device and driver.

`FIRLIFT_GENERIC=1` selects the lane-per-output kernels that also run on the CPU VM. Intel and AMD
Metal devices cannot create `simdgroup_matrix` pipelines. `FIRLIFT_Q4_GEMM=matrix` therefore requires
`FIRLIFT_TARGET=apple` with `FIRLIFT_GENERIC` unset; the Vulkan JIT rejects a FIR kernel containing
those Metal-only instructions.
Use `scalar`, `parallel`, `parallel_fma`, `word_parallel` (Q4 GEMV), or `tiled_fma` for Vulkan and
other targets.

A graph lifted from a Vulkan run keeps the default kernels except the simdgroup decode matmuls
(`genMatmulQ4Simd`, `genMatmulNTSimd`). Those follow Metal, where a simdgroup is a contiguous run of at
least 16 threads of the grid; Vulkan leaves the subgroup width to the driver, and on MoltenVK the
simdgroup Q4 GEMV returns wrong sums. For a Vulkan trace `auto` uses `word_parallel` for supported Q4
GEMV shapes up to `FIRLIFT_Q4_WORD_MAX_K`, falling back to the byte-load reduction for larger or
unsupported shapes; the f32 GEMV and other reductions continue to use threadgroup memory.
`FIRLIFT_SIMD=1` explicitly requests simdgroup kernels for a Vulkan trace, and `FIRLIFT_SIMD=0`
disables them on targets that otherwise select them.

### Registers

A kernel uses as many registers as it needs: register operands are 8-bit fields (up to 256 float and
256 integer registers), the low 5 bits where the 5-bit fields were and the 3 high bits in the top of
the instruction word, so programs written with registers 0..31 keep their exact words. No engine fixes
the count: the CPU interpreter allocates what the program names, and the Metal/GLSL lowering declares
that many. Programs with more than 32 registers get separate locals (`R_0`, `R_1`, ...) instead of an
array: the Intel shader compiler keeps a large register array in private memory, which made the 8 x 8
accumulator tile of the GEMM about twice as slow. Kernels with up to 32 registers keep their array and
their generated source.

The Metal lowering reads or writes four consecutive floats through one index register (offsets
`k .. k+3` of `tgloadf32`, `tgstoref32`, `uloadf32` or `ustoref32`) as one `packed_float4` access, so
generators emit their loads as runs of four. `BYBY_JIT_VEC4=0` keeps one access per float.

### Loading: native weights and one pipeline per kernel

A lifted graph loads its weights from the native `.nimq` checkpoint as well as from
`model.safetensors`. From `.nimq`, a packed asymmetric Q4 tensor whose group size and shape match a
`q4b` weight is copied as stored (nibble bytes, scales, zero points): nothing is dequantized and
quantized again, and the graph computes with the very values the native runtime uses. With no `-m`,
the `.nimq` in the model directory is picked. Generation speed is the same for both files; only the
load differs (TinyLlama: 8.6 s of CPU instead of 35.5 s).

On Metal the runner assembles every kernel position independent: the program holds no buffer
address, it reads the arena address of each port from a small per-node table (bound through the
input domain) and adds it to every access. All nodes of a kernel then share one program, so one
pipeline is compiled per kernel instead of one per node. A TinyLlama graph has about a thousand
nodes and 37 kernels: the whole run went from about 7 minutes (shader compilation) to about 30
seconds on Intel Iris Plus 640. `BYBY_FIR_RELOC=0` keeps one program per node; the CPU and Vulkan
backends keep absolute addresses.

### Buffer reuse

The lifter gives every op output a buffer of its own, which for a multi-row phase is one activation
per op: a TinyLlama lift with a 448-row phase declared 964 buffers, 1.87 GiB, and did not load. A
temporary only has to live from the node that writes it to the last node that reads it, so later
temporaries of the same size take over its buffer (the same lift: 77 buffers, 145 MiB). Only buffers
that one phase writes before it reads them qualify; state (the KV cache), graph inputs and outputs,
buffers with initial data and anything two phases share keep their own buffer. `FIRLIFT_NOREUSE=1`
keeps one buffer per output.

### Fewer launches per token

Every dispatch has a fixed cost (about 60 microseconds on Intel Iris Plus 640), so the decode step is
bound by its launch count as much as by its arithmetic. After dead-node elimination the lifter merges
consecutive decode kernels that do not depend on each other (no kernel reads or writes a buffer another
one writes) and share a threadgroup size into one launch: each kernel gets its own thread range and sees
its own thread, threadgroup and simdgroup numbering (`fuseSideBySide`). For a Llama layer that is
q/k/v, the two RoPE kernels, the two KV scatters and gate/up: 15 launches become 10.
`FIRLIFT_NOSIDE=1` keeps one launch per kernel.

### Prompt tails and the prefill head

A kernel whose threads are ordered by row declares it on its kernel line: `rows=B:T` means every `B`
rows take `T` threads and a thread touches only its own rows (and state of earlier rows). When every
node of a phase is the token embedding or such a kernel, the phase is declared `partial`: the runner may
run it over fewer rows by launching only the threads of those rows, so the tail of a prompt takes one
phase run instead of one decode step per token. The tiled GEMM is ordered by tile row, so a tail costs
whole tiles (like the native runtime, which runs its last chunk at the tile size).

When the recorded prefill step ends in a vocabulary-sized read, the lifter keeps the head (last-row
copy, final norm, output projection) in the phase and declares `last=BUFFER`: the phase then returns
the logits after its last row and the prompt needs no extra decode step. The last-row copy is emitted
as `sessionCopyInto:...,lastRowOf=R`, which the runner resolves to the last row it really ran.

```
kernel s.matmul_q4 in=2 out=1 threads=2048 tg=128 tgbytes=25600 expect=... rows=64:2048 tag=aten::matmul
phase chunk1 rows=64 last=t1797_c partial
```

The native runtime splits a long prompt into chunks (64 rows first, then what fits its time budget
per command buffer). Every token-row upload of the recorded prefill step becomes a phase of its own
(`chunk1 rows=64`, `chunk2 rows=256`, ...), so record `prefill:2` on a prompt of 64 + N tokens to get an
N-row phase. The runner uses one partial run of the smallest phase that holds what is left of the
prompt, and inside a multi-row phase it commits the command buffer and keeps encoding (no wait) every
512 row-nodes: a 256-row phase in a single command buffer is cancelled by the Intel driver.

`BYBY_FIR_PROFILE=1` adds a line with the mean encode, GPU wait and logits read time of a decode step;
`BYBY_FIR_PROFILE=2` waits for the GPU after every node and prints the time per kernel and phase (it
slows the run; use it to see where a phase spends its time).

A recorded prefill step before the decode step becomes the graph's multi-row phase (`phase chunk1`),
which the runner repeats over the prompt in chunks of that many rows. The runtime prefills in chunks of
64 tokens, so trace a 64-token prompt:

```sh
BYBY_FIR_PROBE=/tmp/t64.fprobe BYBY_FIR_PROBE_STEPS=prefill:1,decode:1 \
  other/formers.n/formers-cli --models models/tinyllama -p "$(yes a | head -46 | tr '\n' ' ')" -n 2 --temp 0 --no-progress --backend metal
FIRLIFT_TARGET=apple other/formers.n/firlift/firlift --trace /tmp/t64.fprobe --config models/tinyllama/config.json --out /tmp/t64a
```

The lifted TinyLlama graphs are `models/tinyllama/lifted/model.fir` (Apple GPU) and
`models/tinyllama/lifted/model_intel.fir`:

```sh
other/formers.n/formers-cli_arm64 --models models/tinyllama -m models/tinyllama/model.safetensors \
  --fir models/tinyllama/lifted/model.fir --raw -p "The capital of France is" -n 64 --temp 0 --backend metal
```

`model_intel.fir` holds a 64-row and a 448-row phase. It was lifted from a 512-token prompt with the
native chunk budget raised so the second chunk takes the rest of the prompt:

```sh
BYBY_UBATCH=512 BYBY_PREFILL_CB_MS=450 BYBY_FIR_PROBE_STEPS=prefill:2,decode:1 BYBY_FIR_PROBE_DATA=0 \
other/formers.n/firlift/firlift --config models/tinyllama/config.json --out "$OUT" --report \
  --run other/formers.n/formers-cli --models models/tinyllama -m models/tinyllama/model.nimq \
    --raw -p "$PROMPT_512_TOKENS" -n 4 --temp 0 --backend metal --no-progress
other/formers.n/formers-cli --models models/tinyllama -m models/tinyllama/model.nimq \
  --fir models/tinyllama/lifted/model_intel.fir --raw -p "$PROMPT" -n 64 --temp 0 --backend metal -c 2048
```

Intel Iris Plus 640, Metal, 486-token prompt: the graph prefills at 62.9 tok/s and decodes at
10.5 tok/s; the native runtime (`BYBY_UBATCH=512 BYBY_PREFILL_CB_MS=450`) prefills at 52.8-59.5 tok/s
and decodes at 10.5-11.3 tok/s, with the same text.

## Resident Q4 weights (`script_abi 2`)

A `q4b` weight lives in the graph's arena, and one arena is one device buffer: a model whose packed
weights exceed the largest buffer of the device (2 GiB on an Intel Iris Plus 640) cannot load that way.
A graph that declares `script_abi 2` and `require feature resident_q4` may declare weights `q4r` /
`q4r128` instead. The bytes are the same (nibbles, then the group scales, then the group zero points) but
every such weight gets a device buffer of its own, filled straight from the packed tensor of the `.nimq`
checkpoint, so the graph computes with the very values the native runtime uses.

A kernel reaches a resident weight through its bindings, not through an arena address. It is declared
`layout=slots slots=mem:out0,u8:in1,lut:in1,input:in0`: the mem domain is the output, the u8 and lut
domains are the weight (nibbles through `uloadu8`/`uloadu32`, scales and zero points through the constant
loads) and the input domain is the activation. Everything else stays where it was: activations, state and
small weights share the arena, the other kernels run in place on it with one compiled pipeline per kernel,
and a slot kernel binds its arena ports at their arena offsets. The context still bounds the arena (the
KV caches are part of it); the runner reports an arena that does not fit one device buffer.

Lift a Q4 model for a Vulkan device (Colab T4) the same way, from a pipe and without `--weights`, so the
lift is the semantic one (weights named after their checkpoint tensors, every Q4 matrix a resident `q4r`
weight, one arena for activations and KV caches):

```sh
BYBY_FIR_PROBE_STEPS=prefill:2,decode:1 BYBY_FIR_PROBE_DATA=0 \
other/formers.n/firlift/firlift --config "$MODELS/config.json" --context 512 --out "$OUT" --report \
  --run other/formers.n/formers-cli --models "$MODELS" -m "$MODELS/model.nimq" --raw --add-special-tokens \
  -p "$PROMPT" -n 4 --temp 0 --backend vulkan --no-progress --no-warmup -c 512
```

`--weights` together with recorded upload data selects the raw lift instead: a data graph in which every
buffer has an allocation of its own. A data graph also accepts `q4b`/`f32a`/`fp4km` weights now (each is a
buffer of its own in the arena encoding and arena-layout kernels stage it), but it stages every port of every
kernel, so it is the slow path.

`BYBY_FIR_Q4_BACKEND=native` is a diagnostic switch: resident Q4 nodes then dispatch through the
backend's `sessionMatmulQ4` instead of their lifted kernel, which tells a slow Q4 kernel from a slow rest.

DeepSeek Coder 6.7B (32 layers, hidden 4096, 224 resident matrices) lifted for a 512-token context is
`models/deepseek-coder-6.7b/lifted/model_intel.fir`:

```sh
other/formers.n/formers-cli --models test/deepseek-inference/models \
  --fir models/deepseek-coder-6.7b/lifted/model_intel.fir \
  --raw --add-special-tokens -p "$PROMPT" -n 64 --temp 0 --backend metal -c 512
```

Intel Iris Plus 640, Metal, 106-token prompt: the graph prefills at 6.6 tok/s and decodes at 1.7 tok/s;
the native runtime prefills at 5.2 tok/s and decodes at 1.4-1.7 tok/s, with the same text.

## Mamba-3 with a merged LoRA (the gated model)

`test/mamba3` merges the LoRA adapters of the `RRRRRRRRRRRG` checkpoint into the FP4 weights on the GPU
before it generates. That rewrites the packed weights and the LUT after the upload the recorder saw, so
with the recorder on (`BYBY_FIR_PROBE`, `BYBY_FIR_PROBE_DATA=1`) the merge reads the merged buffers back
once and records them as a fresh upload (`recordMergedFp4Weights`; with the recorder off it is one boolean
test and no readback). The lifted graph holds the merged tensors, so it runs from the same `.nimq` as the
native run and needs no adapter file.

Record the native run the graph should cover with the warm-up off (`--no-warmup` in chat mode): the lifter
replays every earlier pass that wrote state the decode step reads, so a warm-up pass over token 0 would
become a phase too. Then lift and run. `--run COMMAND...` records and lifts in one step without writing a
trace or sidecar file (see "Record without files"); with a trace file already on the disk use
`--trace "$TRACE"` in its place:

```sh
other/formers.n/firlift/firlift \
  --config test/mamba3/models/mamba3-siso-187m/config.json \
  --weights test/mamba3/models/mamba3_siso_187m.nimq --out "$OUT" --report \
  --run other/formers.n/formers-cli --models test/mamba3/models \
    -m test/mamba3/models/mamba3_siso_187m.nimq --lora "$LORA" --raw -p "$PROMPT" -n 4 --temp 0 --backend metal
other/formers.n/formers-cli --models test/mamba3/models/mamba3-siso-187m \
  -m test/mamba3/models/mamba3_siso_187m.nimq --fir "$OUT/model.fir" \
  --raw --add-special-tokens -p "What is the capital of France?" -n 32 --temp 0 --backend metal
```

`--add-special-tokens` makes the FIR path prepend BOS the way the native runtime does. `ready: warmup N s`
runs the prompt and two decode steps once (so every kernel is compiled), restores the initial state with
`firGraphReset` and starts the timed run; `--no-warmup` skips it. The run prints the prompt speed
(`FIR graph prefill (K multi-row phases): P tokens in S s`) and the decode speed
(`FIR graph decode: N steps in S s`); both leave the kernel setup/JIT out.

The lifter reports what it did with the trace, for example for a chat turn of 176 tokens:

```text
selected sessions 113:114 (205 dispatches, occurrence 0)
state resets left to the graph's initial contents: 2 copies
single-token units left to the decode step: 7
report: nodes=1738 micro=1734 ext=4 kernels=72 instructions=72
```

- **Prompt phases.** Each token-row upload before the selected decode step starts a unit that becomes
  `phase chunkN rows=R`. A one-shot run (`--raw -p`) prefills all but the last token in one batch, so an
  8-token prompt gives `rows=7`. A chat turn goes through the scheduler in chunks of 128, 32 and 8 rows;
  a turn of 169 to 176 tokens uses each length once. The runner (`fmFirFeedPrompt`) builds a prompt out of
  them: the largest phase that leaves at least one token, then the rest as decode steps, the last token
  being the decode step with logits. A longer turn repeats a chunk length; the repeat becomes another phase
  of the same size, which the runner never picks and which only enlarges the file.
- **State clears are not replayed.** The native runtime clears its recurrent state between prompts by
  copying an uploaded zero buffer over it (`sessionCopyInto` in a session that launches no kernel). The
  graph already starts from the uploaded zeros and `firGraphReset` restores them. Left inside a phase, the
  copy would wipe the state each time the phase is used, so a prompt that uses a phase twice (two 8-row
  chunks, say) would continue from a zeroed state and produce garbage. The lifter leaves the copy out when
  its source is a still-zero upload, the cleared range was uploaded as zeros, and no replayed op wrote that
  buffer before the copy; any other copy is replayed.
- **Single-token units are not phases.** The chat scheduler feeds the fewer than 8 tokens left over as
  decode steps without logits. A phase has at least two rows and the decode step makes the same state
  update, so the lifter leaves these units out and the runner feeds the remaining tokens through the decode
  step (which also computes logits that are not used).

On the 187M gated model the lifted graph gives the same text as the native runtime for a 20-token prompt,
a 171-token prompt and a 201-token prompt (54 to 64 generated tokens, `--temp 0`). Compare the two with
`--raw` on both sides (the native runtime otherwise wraps the prompt in its chat template) and with no
trailing whitespace on the prompt: the graph path keeps a trailing space as one more token, the native
runtime drops it, and the two then continue differently. Speeds against the native runtime on the same GPU
are in [`test/mamba3/benchmark.md`](../test/mamba3/benchmark.md).
