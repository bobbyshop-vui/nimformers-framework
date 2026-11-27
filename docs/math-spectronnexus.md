# Math SpectronNexus

[English](math-spectronnexus.md) | [Tiếng Việt](math-spectronnexus.vi.md)

Math SpectronNexus is an experimental exact-math solver under
`test/math_spectronnexus`. One native Nim program exposes its operations through
the command line and a local Model Context Protocol (MCP) server. The server
publishes 21 tools. Depending on the operation, replies contain an exact result,
verification details, a witness, or the sequence of tools used.

## What it can do

- **Exact arithmetic:** arbitrary-size integers and fractions, expressions,
  matrices, determinants, roots, factorials, binomial coefficients, and
  arithmetic functions.
- **Sequences:** compute terms, test candidate formulas, search for polynomial,
  linear or holonomic recurrences, algebraic generating functions,
  multiplicative structure and periodic residues. Search can also examine
  transformed sequences such as differences and partial sums.
- **Far terms:** `research_sequence` plans how to answer a question at a distant
  index. It can request more terms from a generator, explore the sequence,
  evaluate modulo a modulus or its prime-power factors, use CRT, or give up
  when its available routes fail. The reply includes `answer`, `status`,
  `direction`, and `steps`.
- **Finite-field and polynomial work:** matrix rank and linear systems,
  polynomial GCD and factorization, and Chinese remainder reconstruction.
- **Combinatorics and graphs:** linear extensions of a partial order, maximum
  flow with a minimum-cut certificate, and weighted walk counts.
- **Identification and retrieval:** offline OEIS lookup, numerical constant
  recognition, high-precision constant digits, and search for proved theorems
  in Lean Mathlib.

The full tool list, input schemas, supported ranges, CLI examples, training
commands and benchmark results are in the
[SpectronNexus README](../test/math_spectronnexus/README.md). The language-model
workflow and detailed tool guidance are in its [SKILL.md](../test/math_spectronnexus/SKILL.md).

## Sequence workflow

Use `calculate` to generate small cases. Send at least eight terms and their
starting index to `explore_sequence` to test claims and look for structure. Use
`research_sequence` when you want a value at a specific large index. For a
computed generator, provide `generator`, `start`, and `count`. For supplied
values, provide `terms` and `offset`. Put the requested `index` and optional
`modulus` in `question`.

Inspect `status` before presenting a far-term result. In particular,
`extrapolation_from_conjectured_direction` means the solver evaluated a rule
fitted to the supplied terms and checked against unseen terms. That check is
evidence, not a mathematical proof. State this qualification unless the rule
has been proved independently. `no_answer` means the available routes could
not produce an answer.

## Run locally

Build and start the MCP server from the repository root:

```sh
make -C test/math_spectronnexus build
NEXUS="$PWD/test/math_spectronnexus/nimcache/math_spectronnexus"
"$NEXUS" mcp --backend=metal
```

To register it with Claude Code for this checkout, run:

```sh
claude mcp add --transport stdio --scope local math-spectronnexus -- \
  "$PWD/test/math_spectronnexus/nimcache/math_spectronnexus" mcp --backend=metal \
  "--program-checkpoint=$PWD/test/math_spectronnexus/models/sequence_program_curriculum.nimq" \
  "--explorer-checkpoint=$PWD/test/math_spectronnexus/models/sequence_explorer.nimq" \
  "--planner-checkpoint=$PWD/test/math_spectronnexus/models/research_planner.nimq"
```

The project skill is `/math-spectronnexus`. Check the connection in Claude
Code with `/mcp`. The MCP setup and the meaning of each checkpoint option are
also documented in the [local MCP section](../test/math_spectronnexus/README.md#local-mcp).

## Experimental results

The solver, learned sequence explorer and route planner are experimental
research code. Benchmark success measures the listed finite test sets and is
not a guarantee for arbitrary inputs. Training is optional. Tool availability
and exact limits depend on each operation; use the linked README as the source
for current limits.
