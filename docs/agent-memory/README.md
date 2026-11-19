# Imported Claude project memory

Imported on 2026-11-19 for the nimformers-framework project. The repository's
root `AGENTS.md` points here so future Codex sessions can recover the context.
This is repository-local persistent context, not an account-wide memory import.

## Sources

- Original project memory directory:
  `/Users/macos/.claude/projects/-Volumes-Data-Laptrinh-App-nimformers-framework/memory/`.
  Every Markdown file is preserved byte-for-byte under `claude/`.
- User-provided T4 build/benchmark output:
  `/Users/macos/.codex/attachments/e2b00ad2-f1a6-446c-913a-80b49e2ef2ed/Pasted text.txt`.
- User-provided continuation transcript:
  `/Users/macos/.codex/attachments/509b5275-718a-4421-ae11-6e21a017fc4b/Pasted text.txt`.

The attachments remain at their original paths; their relevant conclusions are
captured in `handoff.md`. `claude/MEMORY.md` is the original memory index.

## Interpretation

The root guidance distills active preferences and resolves conflicting old notes:
do not edit notebooks, but include existing relevant notebook edits when a commit
is requested. A memory about committing does not itself request a new commit.
Historical progress reports (especially `codestral-fir-work-paused.md`) must be
checked against HEAD before acting on them.

Claude's `codex:codex-rescue`, MCP session routing, and old model defaults belong
to its former execution environment. They are preserved for provenance, not
translated into executable requirements for this Codex session. Old cleanup
examples likewise do not authorize deleting arbitrary files or caches.

For memory and source provenance, see `import-manifest.json`, which records source
paths and SHA-256 checksums. No credentials or model data are part of this import.
