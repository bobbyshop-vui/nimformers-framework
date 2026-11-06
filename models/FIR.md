# Formers IR (FIR)

FIR is a thin description of a model graph: kernel names, tensor names, and
symbolic dimensions from the adjacent `config.json`. It reuses the framework's
micro-kernel builders and backend API. FIR has no assembler, bytecode interpreter,
or implementation of the Mamba recurrence.

## Text format

Files start with `fir 1`, `model name`, and `config config.json`. `#` starts a
comment. `sym P = 2*inter+2*dstate+3*nheads+rope` defines a dimension; integer
expressions support `+`, `-`, `*`, `/`, and parentheses.

The initial config adapter reads Mamba dimensions: `hidden`, `inter`, `layers`,
`vocab`, `dstate`, `headdim`, `nheads`, and `rope`. `inter` uses an explicit
`d_intermediate` or defaults to `hidden*ssm_cfg.expand`. `nheads=inter/headdim`;
`rope=dstate/4`. RMS epsilon comes from `layer_norm_epsilon` (default `1e-5`),
and is a float field rather than an integer dimension.

`buf name f32 count` describes an activation or state. `per_layer` marks
persistent state belonging to each layer. `weight name dtype tensor.name tensor
[rows,cols]` describes a checkpoint tensor; it allocates no memory. Weight shapes
and activation element counts are resolved and validated without loading weights.
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
