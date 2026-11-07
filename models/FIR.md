# Formers IR (FIR)

FIR is a thin description of a model graph: kernel names, tensor names, and
symbolic dimensions from the adjacent `config.json`. It reuses the framework's
micro-kernel builders and backend API. FIR has no assembler, bytecode interpreter,
or implementation of the Mamba recurrence.

## Text format

Files start with `fir 1`, `model name`, and `config config.json`. `#` starts a
comment. `sym P = 2*inter+2*dstate+3*nheads+rope` defines a dimension; integer
expressions support `+`, `-`, `*`, `/`, and parentheses.

The optional header directive `layout classic|script` is stored on `FirProgram`.
When absent, the parser selects `classic`; unknown values are rejected. In the
graph runner, `classic` binds only rank-1 weights in the F32 arena (`memF32`).
`layout script` additionally binds non-matrix (rank != 2) F32 weights there,
including higher-rank tensors under their original checkpoint names and logical
shapes. Their payloads are copied in row-major order after the loader checks the
complete declared shape. Rank-2 weights retain resident matrix storage for
external linear and embedding calls. `q4t` weights use the same resident path
without the `[in,out]` transpose. The Python exporter emits `layout script`.

### Script ABI and capabilities

```text
layout script
script_abi 1
require feature weights_bf16
require feature weights_f16
```

`script_abi 1` selects the versioned runner contract. `FirProgram.scriptAbi` is
1 when declared and 0 when omitted. Existing script files without this directive
keep their legacy instruction validation. The runner supports the required
features `weights_bf16` and `weights_f16`; each `require feature NAME` asserts
runner support and is retained in `FirProgram.requiredFeatures`. Features do not
select a storage dtype or change a weight declaration. Repeated feature
requirements are idempotent. The ABI directive occurs at most once.

Both directives require `layout script`. The parser resolves the final layout
outside kernel bodies before checking gated declarations, including when layout
appears later in the file. Unknown ABI versions, malformed directives and unknown
required features fail before checkpoint loading. The graph loader also checks
the contract for programs constructed directly by a caller.

### Script checkpoint loading

With `layout script`, selected safetensors tensors accept F32, BF16 and F16 source
storage independently of the FIR `f32` or `q4` destination declaration. BF16
conversion shifts the 16 source bits into the upper half of the F32 word. F16
conversion constructs IEEE F32 bits exactly, including signed zero, subnormals,
infinities and NaN sign/payload bits. No finite value needs rounding. All payloads
are little-endian. Activations, state and converted weights use F32 storage;
`q4` still requests the existing transpose and load-time Q4 quantization of the
converted F32 input.

The script loader checks every selected tensor's full declared shape, supported
source dtype and exact byte count before backend preparation or weight upload.
It resolves plain names before declared prefixes and expands each `{i}` layer
name. Conversion covers rank-1 arena weights, higher-rank script arena weights,
resident matrices, embeddings, tied heads, flattened checkpoint copies and Q4
inputs. After preflight, it reads, converts and uploads one tensor at a time.
Unsupported dtypes report the source dtype and tensor name. Required feature
declarations assert capabilities; conversion is available to all `layout script`
files, including those that omit the ABI directive.

Default and explicit `layout classic` keep the F32-only checkpoint reader,
placement, bytecode and error handling. A selected non-F32 tensor retains the
error `safetensors shape/dtype mismatch: NAME`.

### Instruction whitelist and conformance

The FIR text whitelist is `nop`, `loadconstf32`, `add`, `sub`, `mul`, `div`, `fma`,
`neg`, `exp`, `rsqrt`, `sigmoid`, `softplus`, `addconst`, `mulconst`, `cmplt`,
`ifbegin`, `ifend`, `forbegin`, `forend`, `loadlid`, `loadtgid`, `ulid`, `utgid`,
`uconst`, `uadd`, `umul`, `uaddk`, `umulk`, `uandk`, `ushrk`, `ushlk`, `ushr` and
`uloadu8`/`uloadu32` (the last three only with `layout script`), `rtou`,
`utor`, `loadf32`, `storef32`, `uloadf32`, `ustoref32`, `tgloadf32`, `tgstoref32`,
`barriertg` and `uatomicadd`. Assembly appends HALT; `halt`, grid barriers and
other framework emit helpers are outside this text whitelist.

ABI 1 validation rejects writes to any active float loop-count register, including
writes inside nested conditionals/loops, and restricts immediate shift counts to
0..31. Classic and unversioned script assembly retain their existing validation.
All three use the same bytecode instructions and backend implementations.

- Loop counts: production CPU snapshots the integer trip count on entry;
  JIT, Python and the test interpreter re-read the register. A fixture that sets
  its count from 3 to 0 inside the loop executes three times on CPU and once in
  the test interpreter. ABI 1 rejects that kernel.
- Effective U addresses: CPU widens U+immediate before adding and bounds-checks
  the arena; JIT and Python use uint32 wrap. CPU rejects `0xffffffff + 1` as an
  arena address. Valid portable addresses fit without wrapping the final sum;
  uint32 register arithmetic still wraps.
- Shifts: CPU masks immediate counts with 31; Python uses the supplied count.
  The CPU fixture checks both shifts by 32; ABI 1 rejects those operands.
- FMA: production CPU uses `fmaf`, and JIT uses fused multiply-add. Python
  evaluates in double precision then rounds; the test interpreter evaluates a
  float expression. The CPU cancellation fixture gives exactly `-2^-46` for
  `(1+2^-23)*(1-2^-23)-1`.
- Softplus: CPU/JIT evaluate `log(1+exp(x))`; Python uses a stable branch for
  large positive inputs. The CPU fixture returns infinity at 100.

`make -C other/formers.n fir-test` runs the production CPU conformance fixtures,
whitelist checks, bytecode comparisons and synthetic mixed-dtype loading tests.

The initial config adapter reads Mamba dimensions: `hidden`, `inter`, `layers`,
`vocab`, `dstate`, `headdim`, `nheads`, and `rope`. `inter` uses an explicit
`d_intermediate` or defaults to `hidden*ssm_cfg.expand`. `nheads=inter/headdim`;
`rope=dstate/4`. RMS epsilon comes from `layer_norm_epsilon` (default `1e-5`),
and is a float field rather than an integer dimension.

`buf name f32 count` describes an activation or state. `per_layer` marks
persistent state belonging to each layer. `weight name dtype tensor.name tensor
[dims]` describes a checkpoint tensor; parsing allocates no tensor memory.
`dtype` is `f32` or `q4`; `layout script` also accepts `q4t` for a `[out,in]`
matrix consumed by `fir.linear_t`. `q4t` is quantized in place (no transpose) at
load and runs the resident Q4 GEMV.

`layout script` also has dtypes for 2-D `[out,in]` matrices that micro-kernels read
directly from the arena (a kernel port is the whole blob):

| dtype | Arena contents (float words) |
| --- | --- |
| `f32a` | the matrix, row-major |
| `q4b`, `q4b<G>` | nibble bytes `[out, ceil(in/2)]` padded to words, then `out*ceil(in/G)` scales, then as many zero points; low nibble = even k; value `(nib - zero) * scale`; `q4b` is group 32, `q4b128` group 128 |
| `fp4km` | nibble+scale bytes (`out*in/2` + `out*in/32`) padded to words, then 272 floats: 16 FP4 values and 256 E4M3 scale values times the tensor scale; byte j of a 32-group holds element j (low) and j+16 (high) |

Kernels read the byte region with `uloadu8`/`uloadu32` (script layout only), the same
bytes the float arena holds.
Weight shapes and activation element counts are resolved and validated without loading weights.
The example uses a tied embedding/language-head tensor.

Nodes use `node id micro|ext|builtin kernel.name inputs -> outputs`, for example:

```text
node prepare micro m3.prepare proj b_norm c_norm b_bias c_bias dt_bias phase -> k q coeff phase
node scan micro m3.ssm_step proj k q coeff state prev_k prev_v D -> ssm_out state
node carry micro m3.carry k proj -> prev_k prev_v
node project ext m3.gemv_q4 ssm_out out_proj -> residual
node residual builtin add hidden_in residual -> hidden_in
```

`repeat layers` / `endrepeat` marks a layer-scoped subgraph. The parser retains
that scope on each node; a runtime caller iterates it and resolves `{i}` in weight
names. FIR does not expand or execute the full graph itself.

## Kernel registry and bindings

`firDefaultRegistry()` in `other/formers.n/src/formers/ir/fir_exec.bybylang`
registers actual framework builders:

- `m3.prepare`: `prepareSiso`, one 256-lane group per head, 1024 shared bytes.
- `m3.ssm_step`: `scanSiso`, one 32-lane group per intermediate channel,
  128 shared bytes. It updates the SSM state and emits the gated inner output.
- `m3.carry`: `carrySiso`, saving the current K/V for the next decode step.
- `m3.silu_mul`: `siluMlp`, emitting the existing MLP gating program.
- Small builtins `rms_norm` and `add` reuse `rmsRows` and `residual`.

These procs live in `test/mamba3/m3_inference_kernels.nim`. The node buffer names
for Mamba micro nodes are an explicit ABI, checked along with their sizes.
`firMambaBindings(model)` calls the existing `arenaLayout(hidden, inter, nheads,
dstate, vocab, 1, 1)` and `stateLayout(arena, arena.total, 0)`. It describes one
layer's decode step (`T=1`), not all layers' state allocation.

`FirBindings.memory` maps activation/state names to float element offsets in
`memF32`. `FirBindings.constants` maps small parameter names to offsets in the
separate float parameter buffer bound as the VM LUT/const buffer. The returned
`params` is an `M3Params` passed directly to the framework builders. `norm_f`
has its own constant offset. State includes phase, previous K/V, and SSM values;
K and Q each contain `nheads*dstate` floats. A caller can supply its own
`FirBindings` with actual per-layer addresses and matching name maps. These
programs use static offsets (`dynamicOffsets=false`); rebuild for new addresses.

`firMicroKernel(node, model, registry, bindings)` returns a `FirKernel` containing
`program`, `threads`, `tgSize`, and `tgBytes`. `firMicroProgram` returns only its
`VmProgram`; `firMicroKey` returns the registry name, suitable as a JIT tag (the
JIT key also hashes the program). GPU callers can pass the program to
`jit_vm.dispatchProgram`, bind the arena and parameters, and launch the returned
geometry. See [the existing API](../docs/microkernel-api.md).

External entries contain callable adapters and their backend symbol names:
`m3.gemv_q4` invokes `backend.beMatmulQ4` with `M=1` (the backend's grouped Q4
format), and `m3.embedding` invokes `backend.beEmbeddingLookup`. Resolve with
`firExternal(node, registry)`, then call its `call` with `FirExtArgs` containing
the backend and loaded buffers. A runtime may override a registration for its
resident/FP4 implementation with `firRegisterExt`; FIR does not load or repack
weights. Validation checks registered callbacks, arity, and matrix dimensions;
it does not dispatch a backend or validate weight payloads.

New architectures add framework kernel builders with `firRegisterMicro` (and
appropriate config/binding adapters), or use BybyLang's existing `micro kernel
NAME ... :` DSL in source code. FIR files reference registered names rather than
embedding a second assembly language. Unknown micro, ext, and builtin names fail
with the node name. GPU runtime integration, graph scheduling, weight loading,
and per-layer storage allocation remain the caller's responsibility.

## Lightweight checks

```sh
make -C other/formers.n fir-test
other/formers.n/formers-cli --fir-check models/mamba3-1.5b/model.fir
```

The checker resolves every node and actually builds all micro/builtin programs
using the 1.5B dimensions, rejecting empty/Halt-only programs. It never allocates
a 1.5B model or loads weights. Tests execute the registered prepare/SSM/carry
programs with the root `vm_interp.runGrid`, the framework arena layout, and a
separate plain-float reference (tolerance `1e-4`). Tree reduction is emitted as
ordinary loops, loads, stores, and threadgroup barriers supported by that
interpreter; these kernels require no grid barrier or virtual-thread opcodes.

The requested tiny dimensions are overdetermined: hidden 32 and expand 2 imply
intermediate 64, but two heads of width eight imply intermediate 16. Tests cover
both the explicit `inter=16`, two-head kernel fixture and a consistent
`inter=64`, eight-head expand-2 fixture. No GPU, backend dispatch, or full model
inference is covered by these CPU tests.

## Resident Mamba-3 inference

Pass `--fir models/mamba3-1.5b/model.fir` to `other/formers.n/formers-cli`
with the usual Mamba-3 model/config/tokenizer options. The runner validates the
file and installs an `M3ProgramProvider` for generation, then removes it even
if generation fails. `FIR: provider used for N programs` is printed in quiet
mode too; zero means generation never reached resident program preparation.
Without `--fir`, the runtime keeps its direct builders.

The provider resolves norm1, prepare, scan, carry, norm2, MLP, residual and final
norm through graph activation bindings and the micro/builtin registry. Runtime
requests supply the actual arena, layer state, parameter offsets, batch length,
and dynamic-offset flag. Dispatch geometry and compile tags match the original
resident path; installing/removing a provider invalidates cached runtime programs.
Missing roles, unsupported micro/builtin scheduling roles, inconsistent duplicate
residual operations, dimensions and epsilon mismatches fail explicitly.
`BYBY_ONE_DISPATCH=1` is incompatible with a program provider.

This is a program-provider integration for the existing resident scheduler.
The FIR file chooses registered operations for its fixed activation roles; it
does not reorder execution or replace the resident weight/projection/embedding
implementation. External nodes remain validated by `--fir-check`. Linear++/gate
sidecar programs, bias, LoRA and projection programs still use runtime builders.
New architectures that need a different schedule also need a runtime adapter.

`make -C other/formers.n fir-test` additionally compares all eight provider roles
with direct builders (48 bytecode/dispatch comparisons at T=1,7,128 and two layer
offsets), including a custom registry operation selected by the graph.

## Graph models (`runtime graph`): config, registry and executor

A model file with `runtime graph` is run by the generic graph executor. Config
integers become symbols automatically; `sym`, `epsilon`, `require`, `prefix` and the
`input:`/`output:` buffer roles (see `docs/fir.md`) let the file declare everything
that is model-specific, including its own config checks. `models/gpt2-125m/model.fir`
is the worked example. Linear weights are `[input,output]`; biases are explicit
nodes; the head can reference the embedding table with `fir.linear_t`.

`fir_graph_kernels.bybylang` registers only the ext entries `fir.embedding`, `fir.linear` and `fir.linear_t`. The micro-kernels (`g2.layernorm`, `g2.gelu_new`,
`g2.add`, `g2.attn_decode`) are defined in `model.fir` itself with `kernel ... endkernel`
blocks (see `docs/fir.md`): one `session_vm` `emit*` call per line, parsed by
`fir_kernel.bybylang` into a `VmProgram`. `firMicroKernel` resolves a model-defined
kernel before a registry kernel. Rank-1 weights live in the F32 activation arena
for VM access; `layout script` also places higher-rank non-matrix F32 weights
there. `firGraphBindings` assigns offsets, multiplying `per_layer` buffers and
weights whose tensor names contain `{i}` by `layers`; each layer gets separate
weight and K/V slices. `position` is the current decode index; attention writes
the new K/V, computes stable softmax over
`[0,position]` and emits the concatenated heads. Scratch `scores` is reused between
layers. FIR has no VM or interpreter of its own.

`fir_graph.bybylang` loads graph-selected safetensors tensors through the
Nim-side header reader, converts script BF16/F16 sources to F32, and executes
graph order, repeating the layer scope.
Micro programs use `jit_vm.dispatchProgramNoSession` and existing Metal/Vulkan
JIT or production CPU VM. `vm_interp.runGrid` is used only by tests. Ext entries
invoke backend embedding/matmul APIs. The CLI routes any directory whose `model.fir` declares `runtime graph` to this
executor, native HF BPE and the shared sampler; `--fir-check` builds all programs
without loading weights. No token/position dimensions are hardcoded.

```sh
make formers-cli
make formers-fir-test
other/formers.n/formers-cli --models models/gpt2-125m \
  --fir models/gpt2-125m/model.fir --raw -p "Hello, my name is" -n 32 --temp 0 --backend cpu
other/formers.n/formers-cli --models models/gpt2-125m -cnv --temp 0.8 --backend metal   # chat
```

Linear layers are Q4 resident by default (`--fir-f32` selects the fp32 reference) and
one decode step is one GPU session. `q4t` keeps Llama-style `[out,in]` weights in
the same resident Q4 form. See [`docs/fir.md`](../docs/fir.md) for measured
speed, Q4 quality and the offline HF comparison commands.

## Trace lifter

`docs/fir-script-lift.md` describes `fprobe 2` and the compiler that rebuilds
`model.fir` from a recorded decode step (`other/formers.n/firlift/`). The graph
runner can run the result directly: the GPT-2 125M round trip reproduces the
hand-written file's structure and 24 greedy tokens, and the TinyLlama core trace
lifts to a 16-node graph whose 24 tokens equal the Python reference export.
