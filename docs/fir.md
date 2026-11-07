# FIR (formers IR)

[English](fir.md) | [Tiếng Việt](fir.vi.md)

FIR is a small text file that ships with a model and describes its decode graph: which
**micro-kernels** run, which **backend kernels** to call, and the tensor names and
dimensions (taken from the model's `config.json`). It does not add a second VM or
assembler. Micro-kernels are the framework's own (`session_vm` bytecode, the BybyLang
`micro kernel` DSL, JIT in `jit_vm`), referenced by name from a registry. A model can
therefore choose or add kernels without the framework hard-coding that architecture.

The first model is Mamba-3 SISO 1.5B: [`models/mamba3-1.5b/model.fir`](../models/mamba3-1.5b/model.fir).
The full format and the kernel ABI are in [`models/FIR.md`](../models/FIR.md).
For Python forward-pass tracing and the small exporter prototype, see
[`FIR script`](fir-script.md).

The optional header directive `layout classic|script` controls graph weight
placement. A `layout script` file may declare `q4t` weights for `[out,in]`
`fir.linear_t` matrices, and arena-resident matrices for micro-kernels: `f32a` (f32),
`q4b` / `q4b<G>` (packed asymmetric Q4, group 32 or G) and `fp4km` (FP4-KM, group 32);
`classic` rejects all of these dtypes. The trace lifter is in
[`FIR from a core trace`](fir-script-lift.md). Omitting it selects `classic`: only rank-1 weights receive `memF32`
arena bindings. With `layout script`, all non-matrix (rank != 2) F32 weights use
the arena, retaining their original checkpoint name and declared logical shape.
The loader checks the complete shape and copies values in row-major order.
Rank-2 matrices keep their resident storage for external linear and embedding
calls. Unknown layout values are rejected. The Python exporter writes `layout script`.

`script_abi 1` opts a `layout script` file into the versioned script contract.
`require feature weights_bf16` and `require feature weights_f16` declare required
checkpoint conversion capabilities. Layout is resolved before these directives,
so it may appear later in the file. Unknown ABI versions and required features
fail during parsing, before weight upload. Both directives require `layout script`;
script files without `script_abi` retain the legacy instruction contract.
ABI 1 requires immutable float loop-count registers inside their loops and shift
counts in 0..31. See [the script contract](../models/FIR.md) for loading rules and
the instruction whitelist.

## Node kinds

| Kind | Meaning | Example |
| --- | --- | --- |
| `micro` | a registered micro-kernel (VM bytecode, JIT-compiled on GPU) | `node scan micro m3.ssm_step ...` |
| `ext` | a backend kernel by name (Q4 GEMV, embedding) | `node project ext m3.gemv_q4 ...` |
| `builtin` | small ops (`rms_norm`, `add`) | `node r builtin add ...` |

## Use

```sh
make formers-cli
# check a .fir without loading weights: resolves every name, builds every micro program
other/formers.n/formers-cli --fir-check models/mamba3-1.5b/model.fir
# run Mamba-3 1.5B with its micro-kernels chosen by the FIR file
other/formers.n/formers-cli --models test/mamba3/models \
  -m test/mamba3/models/mamba3_siso_1.5b.nimq \
  --config test/mamba3/models/mamba3-siso-1.5b/config.json \
  --tokenizer test/mamba3/models/mamba3-siso-1.5b \
  --no-lora -p 'def fib(n):' -n 24 --temp 0 --backend metal \
  --fir models/mamba3-1.5b/model.fir
```

With `--fir`, formers.n prints `FIR: provider used for N programs`. Without it the
runtime keeps its direct builders; output is the same.

## Tests

```sh
make -C other/formers.n fir-test    # or: make formers-fir-test
```

It parses the 1.5B file, builds every micro program with the 1.5B dimensions, runs the
Mamba-3 prepare/scan/carry kernels on the CPU reference interpreter against a plain-float
reference (tolerance `1e-4`), and compares the FIR provider's programs with the direct
builders.

## Verified and not verified

- Verified: Mamba-3 1.5B (base, no LoRA), greedy 24 tokens, Metal and Vulkan (MoltenVK)
  on Intel Iris Plus 640; FIR and direct runs give identical text.
- Not verified: other GPUs, LoRA/Linear++ runs, long generations. `BYBY_ONE_DISPATCH=1` is
  incompatible with `--fir`.
- FIR chooses the registered operations for fixed activation roles. It does not reorder
  the schedule or replace the resident weight/projection implementation; a new
  architecture with a different schedule also needs a runtime adapter.

## GPT-2 small (124M)

[`models/gpt2-125m/model.fir`](../models/gpt2-125m/model.fir) describes the entire
decode step: token and position embeddings, affine layernorm, fused QKV Conv1D,
causal multi-head attention with per-layer K/V, attention projection and residual,
second layernorm, MLP with `gelu_new`, residual, final layernorm and tied `wte` head.
All dimensions and epsilon come from the adjacent GPT-2 `config.json`.

```sh
make formers-cli
other/formers.n/formers-cli --fir-check models/gpt2-125m/model.fir
other/formers.n/formers-cli --models models/gpt2-125m \
  --fir models/gpt2-125m/model.fir --raw -p "Hello, my name is" -n 32 --temp 0 --backend cpu
# GPU check: use the identical prompt and sampling settings
other/formers.n/formers-cli --models models/gpt2-125m \
  --fir models/gpt2-125m/model.fir --raw -p "Hello, my name is" -n 32 --temp 0 --backend metal
# Use --backend vulkan for Vulkan instead.
```

**Nothing about this model is built into formers.n.** A model directory is a
self-contained graph model when its `model.fir` (or the file given with `--fir`)
declares `runtime graph`; formers.n then routes any such directory to the generic FIR
graph runner, whatever the architecture. Everything model-specific comes from the file:

| Declaration | Meaning |
|---|---|
| `layout classic` / `layout script` | graph weight placement; default `classic` (rank-1 arena weights), `script` adds non-matrix F32 weights |
| `runtime graph` | the file carries its own graph, weights and kernels |
| `sym NAME = EXPR` | symbols; every positive integer key of `config.json` is already one. The runner needs `layers`, `positions` and `vocab` |
| `epsilon KEY` | float constant taken from a config key (`eps` in kernels) |
| `require config KEY VALUE` | a present config value must equal VALUE (absent passes) |
| `require sym A = B` | integer expressions over symbols must match |
| `prefix P` | extra checkpoint tensor-name prefix tried after the plain name |
| `buf NAME f32 N input:token` / `input:position` / `output:logits` | the runtime I/O contract |
| `weight NAME dtype TENSOR tensor [dims]` | checkpoint tensor (`{i}` = layer), `f32`, `q4`, or script-only `q4t` (shape `[out,in]` for `fir.linear_t`), `f32a`, `q4b`/`q4b<G>`, `fp4km` (shape `[out,in]`, stored in the arena for micro-kernels, see below); shape `[in,out]` for `fir.linear` |
| `node N ext fir.embedding IDX TABLE -> OUT` | row lookup (index from the `input:` role) |
| `node N ext fir.linear X W -> Y` | `Y = X*W`; `q4` weights run resident Q4, `f32` weights an F32 matmul |
| `node N ext fir.linear_t X W -> Y` | `Y = X*W^T` for a `[out,in]` weight (tied heads share the embedding table) |
| `node N micro NAME ... -> ...` | a kernel defined in the file (below) or a framework kernel |

The runner needs a safetensors checkpoint with F32 tensors, `config.json` and a
HF `tokenizer.json` (`bos_token_id` / `eos_token_id` from the config). It is a
completion runner; `-cnv` chat renders the chat template from the model folder (`chat_template.jinja` or
`tokenizer_config.json`). Partial offload is rejected, and
exceeding `positions` fails explicitly. The checkpoint is checked against the declared
weights (name, dtype, shape) before anything is loaded. `--fir-check` validates the
file without weights. `--fir-f32` runs `q4` weights in fp32 (correctness reference).

**Kernels live in the model file.** This model needs operations the framework does not have
(affine layernorm, `gelu_new`, bias add, cached causal attention), so
`model.fir` defines them itself in `kernel ... endkernel` blocks. Each body line is one
`session_vm` `emit*` call: the mnemonic is the emit name in lower case without
`emit` (`emitULoadF32` -> `uloadf32`), registers are `0..31`, and arguments are
numeric expressions over the model symbols, `eps`, `pi`, `inN.size` / `outN.size`
and buffer addresses `@inN`, `@outN`, `@buffer`:

```
kernel g2.add in=2 out=1 size=in0,in1,out0 threads=out0.size tg=1 tgbytes=16
  rtou 0 0
  uloadf32 1 0 @in0
  uloadf32 2 0 @in1
  add 1 1 2
  ustoref32 1 0 @out0
endkernel
```

Options: `in=`/`out=` arity, `threads=`/`tg=`/`tgbytes=` geometry, `size=` buffers that
must have equal element counts, `expect=PORT:EXPR,...` exact element counts, and
`tag=NAME` (`layout script` only) an op identity such as `aten::linear` or
`aten::add+aten::gelu` for fused kernels. A tag is metadata: it does not change execution,
it is carried to the kernel record and written into probe logs so a recorder can map a
micro-kernel back to the op it implements; `layout classic` files reject it. The
parser and `--fir-check` reject unknown mnemonics, wrong argument counts, registers
above 31, unbalanced `forbegin`/`ifbegin`, unknown names and shape mismatches. A kernel
defined in the file wins over a framework registry kernel of the same name (Mamba-3
keeps using the built-in `m3.*` kernels). The text is only a front end for `emit*`;
FIR adds no bytecode and no interpreter.

**Execution.** `fir_graph.bybylang` walks graph nodes in file order and repeats the
layer scope per layer. Linear layers are resident: `c_attn`, attention `c_proj`,
`c_fc` and MLP `c_proj` are quantized at load (grouped asymmetric int4, group 32) and
stay on the GPU; `--fir-f32` keeps them fp32 as the correctness reference. The tied
`wte` head is fp32. One decode step is one GPU session with the FIR kernels and the
resident linears encoded in order; K/V and scratch stay resident, and only the logits
are read back. Metal runs compiled MSL, Vulkan the existing GLSL JIT, CPU the
framework's production `cpu_backend.vmRun`. Context overflow fails explicitly;
positions are never wrapped. Prefill runs token by token through the same path.

`make formers-fir-test` includes the model-folder test `models/gpt2-125m/test_model.bybylang` (also
`make -C other/formers.n fir-graph-test`): seeded random micro fixtures through
`vm_interp.runGrid`, poison-filled future KV slots, two tiny two-layer full graphs
through the production CPU backend against independent float formulas, selective
safetensors loading/routing and context/truncation rejection. Tolerance is `1e-4`.
The real-model check is separate and requires the existing local weights:

```sh
python3 models/gpt2-125m/check_hf.py --backend cpu
# Also runs the same comparison on GPU, outside the sandbox:
python3 models/gpt2-125m/check_hf.py --backend metal
```

Decode on Intel Iris Plus 640 in chat mode (see [benchmark](../models/gpt2-125m/benchmark.md)):
Q4 8.57 tok/s on Metal and 11.40 tok/s on Vulkan; `--fir-f32` 10.78 tok/s on Metal and
8.54 tok/s on Vulkan. The fp32 path matches all 32 greedy token IDs of offline
Hugging Face on CPU and Metal.
