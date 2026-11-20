# Project guidance

## Imported project memory
- Claude's project memory was imported on 2026-11-19. Read
  `docs/agent-memory/README.md` and `docs/agent-memory/handoff.md` when resuming work.
- Original notes are preserved in `docs/agent-memory/claude/`, indexed by
  `MEMORY.md`. Treat old progress reports as historical; verify current code.
- Current user instructions take precedence. Claude-specific agent names,
  model routing, and expired run state are reference material only.

## Communication and scope
- Speak Vietnamese, using "tôi" and "bạn". Keep code comments, logs, and help
  in English; maintain localized documents in the language of each file.
- Finish the requested scope; report remaining validation honestly.
- Do not edit `.ipynb` notebooks. Provide requested cells in chat. Colab run
  cells must not include git/update/build steps already covered by other cells.
- Finish the active Codestral/Mamba/FIR work before changing Python bindings,
  `lib_c_api.nim`, or the shared-library API unless explicitly requested.

## Execution and validation
- Do not load/train Codestral 7B or run heavy inference for validation.
  Prefer compilation, existing evidence, and small numerical harnesses.
- At most one heavy model/lift process across all agents; propagate this rule
  to agents that might execute models. Never rerun a completed benchmark just
  to extract its output.
- Show build/test output directly. Do not redirect it to log files or pipe it
  through grep/tail/head/awk/sed/cut. Read source files directly; use required
  source-search tooling without filtering build/test output.
- Trace in the Nim core via `firlift --run` pipes/RAM, not Python or on-disk
  `.fprobe` files. Codestral lifts may write only `model.fir` and `config.json`;
  reuse the original checkpoint and LoRA at runtime. Never create
  `trace.data` or a LoRA `.raw` file for Codestral. Preserve existing artifacts.
- Remove only identified temporary artifacts when no longer needed; never
  delete user checkpoints, model files, or unrelated caches.

## Architecture and benchmarks
- Keep formers.n orchestration in `other/formers.n` as `.bybylang` code.
- Codestral/Mamba-3 kernels must use Nim micro-VM programs and shared lowering
  to Metal/GLSL/CPU, including tensor-core paths; no handwritten backend kernels.
- Codestral FIR trace/lift/run must emit exactly the same greedy token IDs as
  native inference for the same model, prompt, and decode settings. Replay the
  exact native session operation when arithmetic differs; successful graph
  construction alone does not establish token parity.
- Keep LoRA merge enabled; do not use `--no-merge-lora` or `BYBY_MERGE_LORA=0`.
- Benchmark comparisons use GPU results and exclude cold JIT/setup/autotune.
  Keep one current result per machine/backend; show setup separately.

## Git preferences when commit/push is requested
- Commit directly to main; no PR. Push bybylang before the parent repository.
- Include relevant existing tracked edits, including notebooks, but exclude
  `test/deepseek-inference/benchmark.md`, build binaries, and generated artifacts.
- Use short neutral commit titles; Codex co-author email is `noreply@openai.com`.
