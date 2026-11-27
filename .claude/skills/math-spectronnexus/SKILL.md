---
name: math-spectronnexus
description: Solves exact discrete math problems with the Math SpectronNexus MCP server. Use for integer sequences, counting problems, recurrence discovery, large-index terms, modular sums, OEIS matches, and checking mathematical conjectures.
---

# Math SpectronNexus

Use the tools provided by the `math-spectronnexus` MCP server. Read `test/math_spectronnexus/SKILL.md` from the repository root for tool arguments, supported methods, and limits.

- Call `mcp__math-spectronnexus__explore_sequence` to discover structure or test proposed formulas. Supply `terms` with their starting `offset`, or `generator`, `start`, and `count`. Use `claims` to ask for counterexamples.
- Call `mcp__math-spectronnexus__research_sequence` for a specific far term, including a far term of a sequence of partial sums. Supply the same sequence input and `question.index`, with `question.modulus` for a modular answer. For example, use `question: {"index": 10500000, "modulus": 2187}` with the first 40 Motzkin partial sums and `offset: 0`.
- Use `calculate` to generate small cases, `oeis_lookup` to identify a sequence, and `search_counterexample` to test a proposed identity on a finite grid. For a known recurrence modulo a supported prime, use `recurrence_term`.
- Report `answer` with `status` and the relevant `steps`. For `no_answer`, inspect the steps for the failed route and supply more terms or an alternative formulation where possible. For `extrapolation_from_conjectured_direction`, explain that the far term depends on the inferred rule and give a proof or independent check when available.

If the MCP tools are unavailable, check `/mcp` and the Local MCP section in `test/math_spectronnexus/README.md`.
