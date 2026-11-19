# Codestral performance handoff

## Objective

The latest user feedback in the T4 log was that inference is still too slow.
Continue optimizing Codestral Mamba/Linear++ prefill and decode while preserving
correct tokens, micro-VM kernels, LoRA merge, and the lifted FIR execution path.
The current request first requires importing Claude's project memory into Codex.

## Verified starting point

- The checkout was clean on `main`, HEAD `66cca0c`, at import time.
- Recent commits include whole-prompt ubatching in `infer_lora` (`d1dbb48`),
  separate compile/autotune timing (`e1f160f`), and FP16 sidecar tiled GEMMs
  (`66cca0c`). Do not repeat these changes without checking their call sites.
- The pasted T4 build and FIR check succeeded; this is a performance task,
  not an unresolved compiler failure.
- The T4 log uses Vulkan cooperative matrix FP16/FP32, 64 model layers, with
  Linear++ attention blocks at indices 7, 23, 39, and 55.
- Normal run: prompt eval 247 tokens / 1.727 s = 143.00 tok/s;
  decode 200 tokens / 9.217 s = 21.70 tok/s.
- Diagnostic run: prompt eval 247 / 2.990 s = 82.60 tok/s;
  decode 16 / 0.701 s = 22.83 tok/s. Diagnostic stage synchronizations add
  overhead, so those prefill numbers are not comparable to the normal run.

## Continued work in Codex

The initial diagnosis identified the CLI's fixed prefill pieces (128, 64, 32,
16, and a small tail) as a likely source of repeated dispatch/weight-read cost.
The inferred roughly 150 ms fixed cost per piece is a hypothesis from the
supplied logs, not a new measurement or proof of the bottleneck.

`other/formers.n/src/formers/models/mamba2.bybylang` now defaults native resident
prefill to `BYBY_UBATCH` (512) and feeds the longest batchable prompt prefix to
`forwardPrefillBatch` in one call. The backend divides that call into bounded
ubatches as needed and carries recurrent state between them. A final prefix
remainder below `BYBY_PREFILL_MIN_ROWS` stays on decode; the prompt's last token
still goes through decode to produce logits. FIR recording keeps the old fixed
chunk/tail plan (128/64/32/16/8/5 by default) to preserve lifted phases. Warmup
visits a representative native size or all FIR sizes. New native lengths can
compile shaders after ready; setup time remains excluded from throughput.

Build completed successfully on this Mac for both the default and ARM64 CLI,
including the Makefile's FIR fixture check. A local benchmark or 7B inference
was not run. The T4 speed and token-parity effects therefore remain unverified;
use the existing Colab build/update flow and run the benchmark once on T4.

The 21.70 tok/s decode result remains an open GPU optimization target. The CLI
now reuses its token-embedding buffer and sampler workspace across decode steps,
avoiding a per-token embedding Tensor and repeated vocab-sized sampling arrays.
The token-selection math is unchanged. The GPU path still enables decode command
replay, T4 subgroup-row FP4 GEMV autotune, and fused R-block residual updates by
default. Do not change the decode kernels without a T4 profile: first confirm
the normal log says decode replay is on, get a clean normal-run throughput
baseline, then run a separate `BYBY_DECODE_PROFILE=1` pass and inspect its
sessionBegin/encode/sessionEnd/read breakdown. Keep the normal run's tok/s as
the performance result; profiling adds timing overhead. The host-side changes
compiled successfully, but their impact on T4 throughput and token parity has
not been measured.

The transcript also begins reading `test/codestral/prefill_gpu.nim`, including
lane-scan parity probes, scan selection, and tiled GEMM parity. Its final output
ends mid-read of `chooseScanMode`; it does not establish a new bug in that code.

## Files to inspect

- `other/formers.n/src/formers/models/mamba2.bybylang`: batching, warmup,
  feed/state carry, and CLI timing.
- `test/codestral/infer_lora.nim`: `forwardPrefillBatch`, ubatch defaults.
- `test/codestral/prefill_gpu.nim`: resident prefill bounds, GEMM dispatch,
  shape-dependent scan compilation and state handoff.
- `test/codestral/prefill_inference_kernels.nim`, `kernelfortest/`, and
  `vendor/bybylang/`: micro-VM implementations and backend lowering.
- `other/formers.n/src/formers/models/fir_model.bybylang` and FIR lifter tests:
  keep recorded prefill phases compatible.

## Validation boundaries

Do not load the 7B model or launch a heavy lift/benchmark on this Mac. For T4
validation, first compare generated tokens with the baseline, then record
prefill throughput from a normal run (diagnostic profiling adds synchronization
overhead and is not comparable). Provide only a run cell when needed; do not
edit notebooks or include git/build steps in the run cell. Preserve all final
lifted model and trace data files.

## Official Codex context reference

OpenAI documents `AGENTS.md` as persistent repository guidance and recommends
separate context files for material decisions:
[Iterating Development Workflows with Codex](https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex).
