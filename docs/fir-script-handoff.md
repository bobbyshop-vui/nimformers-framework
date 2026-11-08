# FIR from the core: handoff

State of the work in the working tree. The last commit is `fcea8ef`; nothing
below is committed.

Read first: `docs/fir-script-lift.md` (recorder, lifter, library, builder),
`docs/fir-script.md` (Python reference system), `docs/fir-script-core-probe.md`
(where launches cross the runtime), `docs/fir-script-roadmap.md` (runtime
roadmap), `models/FIR.md` and `docs/fir.md` (FIR format and runner).

## Raw-dispatch update (CPU verification)

The generic path now retains `disp` events and selects repeating complete
session units without markers; `--slice A:B` and `--step N` override selection.
Recorded programs use per-kernel `layout=slots` with verbatim hex words, shared
buffer views and conservative state writes. Semantic session ops and raw nodes
share physical buffers. Core upload instrumentation covers direct Metal/Vulkan
driver uploads and mmap wrapping as well as CPU session uploads. Fingerprints
support `--weights` payload matching; `BYBY_FIR_PROBE_DATA=1` preserves exact
auxiliary upload bytes in a sidecar when no checkpoint tensor can identify them.
Repeated `--weights` arguments also index auxiliary raw-FP32 NIMBIN checkpoints;
the first argument remains the primary checkpoint/embedding source.
The generic graph loader supports raw initializers and selective native NIMQ
embedding loading. No traced model/runtime file needs a hook.
Raw weights consumed exclusively by slot kernels have separate resident
allocations, loaded individually, rather than occupying a checkpoint-sized
shared arena. The CPU round trip also exercises an independent raw weight buffer.
The graph CLI has an explicit `--add-special-tokens` option. The native-baseline
comparison uses a SentencePiece-only tokenizer directory; the full HF directory's
JSON tokenizer gives different IDs for wrapped prompts. Both raw and wrapped
`hi` prompts are checked without loading a model.

`make fir-lift-test` includes a CPU recorder/lifter/runner round trip that checks
separate U8/LUT/input domains, subrange aliases, mixed ops, persistent state and
reset bit-for-bit. It also checks unmarked embedding uploads, a unit spanning two
sessions, and selection of one 605-dispatch session from `cs2.fprobe` when present.
GPU session tests remain the existing tests; unavailable backends skip normally.

The supplied legacy Codestral recording is insufficient for a runnable model:
the selector chooses `243:244`, but 141 needed packed-weight bindings have no byte
extent. LUT, adapter and composite-arena payloads are also absent. The lifter
rejects this information loss; it does not infer identities from layer order or
invent random/zero weights. Re-record with the updated core and upload data
enabled. The fresh `cs3.fprobe` now exports with both the primary NIMQ and auxiliary
`jamba_parallel_lora.bin`. The reproduced output is byte-identical to
`models/codestral-fir/model.fir`: 960 buffers, 763 micro nodes, one embedding
node and 24 kernel definitions (16 raw slot definitions). Its program check and CPU metadata/fingerprint preflight
pass. Twelve large auxiliary uploads omitted by the capture cap require that
second checkpoint; the primary-only command fails loudly.
Metal slot execution and Codestral generation: the model loads and runs on the
Intel Iris Plus 640. The lifter now also lifts the ops that produced the state the
selected unit reads (a conservative backward data-flow slice over earlier ops, plus
the host uploads of per-run buffers released later). For the Codestral recording this
adds the 16-session batched T=8 prefill, emitted as `phase chunk1 rows=8` before
`phase decode`. Phases are generic: any name, any row count, any number of them (each
token-row upload of the replayed ops starts its own phase); its token-row embedding input becomes a multi-row `fir.embedding`
and host-cleared (all-zero) scratch uploads become zero-initialised buffers. The
runner consumes the prompt with the largest phase whose rows still fit (before the last
token), then decodes the remaining tokens, always including the last. `--fir-check`
passes (2189 nodes). Exact 60-token equality with `infer_lora_bin` has not been
measured yet: it needs the 7B model run on Metal. The graph/native samplers still
differ in formatting-token penalty exemptions, banned-token masks and secondary
stop IDs.

## Done

| Area | What exists | Verified |
| --- | --- | --- |
| Recorder | `vendor/bybylang/probe.nim`: `fprobe 2`, one canonical `op` per launch (`tag`, named ports, attrs, `impl=session:…`/`micro:<hash>`/`ext:…`), `disp` per micro launch, CPU/Metal/Vulkan byte extents, backend names, weight names from load sites, safe output path | GPT-2 and TinyLlama traces; recorder off by default |
| Graph lifter | `other/formers.n/firlift/`: bytecode disassembler, port-relative addresses, repeat folding, weight templates, config symbols, report | GPT-2 125M round trip: 30 buffers, 19 nodes (12 micro, 7 ext), 4 kernels, 123 instructions; 24 greedy tokens equal the hand-written file |
| Core lifter | decoder trace mode: embedding upload, RMSNorm, q/k/v/o, fused RoPE+KV+GQA attention, residual/add, SwiGLU fusion, final norm, head | TinyLlama: 16 nodes (7 micro, 9 ext), 4 kernels, 445 instructions; 24 tokens equal the Python reference export |
| Micro-kernel library | `firlift/kernel_library.nim`: parallel 256-lane RMSNorm, add, silu_mul, attention (fused), add_rmsnorm; tags and roles | `firlift/test_library` runs the bodies on the production CPU VM against scalar references (1e-4) |
| Builder API | `firlift/builder.nim`: `BuilderConfig`, `BLinear`, `BRmsNorm`, `newBuilderModel`, `forwardStep`, `beginBuilderTrace`, `traceModel` | `firlift/builder_test` records, lifts and passes `--fir-check` |
| q4 for `[out,in]` linears | `q4t` weight dtype (script layout only); `fir.linear_t` uses resident Q4 GEMV | TinyLlama q4t: 4.0 tok/s vs 1.3 tok/s f32 on Metal; lossy output |
| Runner | `layout script`, `script_abi 1`, features, BF16/F16 loading, kernel tags, `fir.linear_t` q4t, unchanged classic path | `make build test fir-test`, all `models/*/model.fir`, 102 Python tests |
| Session lifter | `firlift/sessionlift.nim` + `kernel_gen.nim`: in raw replay the four elementwise/norm ops present in a decode unit (`sessionRmsnormEnc`, `sessionVecOp`, `sessionActivation`, `sessionCopyInto`) are emitted as exact session-replay nodes that call the backend API on staging buffers; every other `session*` op becomes a generated micro-kernel (rmsnorm, layernorm, matmul NT/NN/Q4/FP4-KM, RoPE from position, KV scatter, attention decode/GQA/fused/flash/backward, SSD, small ops); host embedding, RoPE tables and the host final norm are recognised | 74 CPU-VM kernel tests, `test_session_gpu` on Metal (all kernels match the real ops), `firlift --coverage` on the recorded op trace, TinyLlama (Q4 group 128) lifts to 443 nodes / 15 kernels and generates on Metal |
| Runner dtypes | `f32a`, `q4b`/`q4b<G>`, `fp4km` arena matrices; `ushr`, `uloadu8`, `uloadu32` (script layout only) | `make fir-test`, `make test`, `--fir-check models/*` |
| Python reference | `other/formers.n/python/formers/fir/` (unchanged); GPT-2/TinyLlama exports | 102 tests; used only as the independent reference |

The TinyLlama f32 graph, its q4t variant and the GPT-2 graph all pass
`formers-cli --fir-check`. The f32 TinyLlama graph produces the same 24 tokens
as the Python reference export. The native runtime's q4 baseline produces a
different first token (`Paris.` vs `Paris,`), and the Python reference export
shows the same difference, so it is a checkpoint/quantization difference rather
than a lifter disagreement.

## Not done

| Area | Missing |
| --- | --- |
| Legacy raw recordings | missing byte extents/identities/auxiliary bytes cannot be reconstructed; new complete slot traces are supported by the CPU-tested path above |
| Speed | the lifted TinyLlama runs about 2-3 tok/s on Metal (native 9); peephole fusion cut nodes 443 -> 334 without a measurable change, the lane-per-column matmuls dominate; the threadgroup-parallel matmuls were slower on Intel Iris Plus 640 and are opt-in (`FIRLIFT_TG`) |
| FP4-KM weights | the `fp4km` dtype and `genFp4Gemm` are tested on the CPU VM only; no Vulkan trace with FP4 weights was available |
| Session-op coverage | MoE, conv/pool, sort/top-k and vision/audio semantic calls are not in the library/session-op rules; raw bytecode bypasses that library, within the limits above |
| Ext staging | `fir.linear`/`fir.linear_t` still copy the arena input into a scratch handle; an input-offset session API would remove two copies per linear |
| Numeric q4 | q4t is group 32 from the load-time F32/BF16 conversion; matching the native `.nimq` q4 (group 128) requires a verified export path |
| Native core | the old `other/formers.n/native/` port was removed; its corpus and documents are gone. The reusable ideas (fgir, report, VM) were not retained |
| Python front-end gaps | scatter, dynamic shapes, sort/top-k, runtime integer arithmetic, RNG; see `docs/fir-script-coverage.md` |

## Rules

- `layout classic` (and a missing `layout`) keeps the old bytes and behavior.
  Every extension is behind `layout script`/`script_abi`/features. `q4t` is
  script-only.
- No Python in the trace or lift path; Python is the independent reference in
  tests.
- No model-specific code in BybyLang. The lifter recognises operation idioms,
  not model names.
- Core capture lives in `vendor/bybylang`: `probe.nim`, `probe_data.nim`,
  `probe_weights.nim`, backend upload paths and `gpubackend.nim`.
  Slot analysis is in `micro_kernel/slot_layout.nim`; CPU slot execution also
  extends `backends/cpu/cpu_backend.nim`, `cpu_vm.c` and `cpu_kernels.h`.
  Traced model/runtime files do not need changes.
- Comments, help and docs in English; user chat in Vietnamese.

## How to run

```sh
make -C other/formers.n build
make -C other/formers.n test
make -C other/formers.n fir-test
other/formers.n/formers-cli --fir-check models/*/model.fir
cd other/formers.n/python && python3 -B -m unittest

# recorder -> lifter -> runner (GPT-2)
BYBY_FIR_PROBE=/tmp/gpt2.fprobe BYBY_FIR_PROBE_STEPS=decode:2 \
  other/formers.n/formers-cli --models models/gpt2-125m -p "The capital of France is" \
  -n 3 --temp 0 --no-progress --backend cpu
other/formers.n/firlift/firlift --trace /tmp/gpt2.fprobe \
  --config models/gpt2-125m/config.json --out /tmp/gpt2-lifted --report
other/formers.n/formers-cli --models models/gpt2-125m --fir /tmp/gpt2-lifted/model.fir \
  -p "The capital of France is" -n 24 --temp 0 --no-progress --backend cpu

# TinyLlama core trace (native runtime)
BYBY_FIR_PROBE=/tmp/tl.fprobe BYBY_FIR_PROBE_STEPS=decode:1 \
  ./nim_inference models/tinyllama models/tinyllama/model.nimq metal "Hi" 2 0 1.0
other/formers.n/firlift/firlift --trace /tmp/tl.fprobe \
  --config models/tinyllama/config.json --out /tmp/tl --report [--q4]
other/formers.n/formers-cli --models models/tinyllama -m models/tinyllama/model.safetensors \
  --fir /tmp/tl/model.fir -p "The capital of France is" -n 24 --temp 0 --backend metal

make -C other/formers.n fir-lift-test
```

## Suggested commits

1. Recorder and runner extensions (`vendor/bybylang/probe.nim`,
   `gpubackend.nim`, `cpu_backend.nim`, `jit_vm.nim`, `fir_graph*.bybylang`,
   `fir_parse.bybylang`, tests).
2. Lifter, library and builder (`other/formers.n/firlift/`, Makefile target).
3. Weight names and `q4t` (`nimformer.nim`, `nim_inference.nim`, docs).
4. Docs.

## Known oddities

- The Python `vm.parse` fails on `models/mamba3-1.5b/model.fir`
  (`unknown FIR symbol or expression: inter`); the native port used it as an
  error fixture. The FIR runner accepts the file.
- The native runtime prints `out of memory` after a probed TinyLlama generation
  on this machine even though the trace and tokens complete; the trace file is
  valid and the command's model output is produced.
- `nim_inference`'s final RMSNorm is a host reduction (two `sessionVecOp` muls);
  the lifter replaces it with one `script.rmsnorm` and keeps the head input
  `normed`.
