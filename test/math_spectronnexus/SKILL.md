---
name: math-spectronnexus
description: Use the math-spectronnexus MCP tools for exact discrete mathematics and research-style math
  problems. Trigger on integer sequences and counting problems ("find a formula", "guess the recurrence",
  "what is a(10^18) mod p", "tìm công thức dãy", "số hạng thứ n"), on identifying a sequence in the OEIS,
  on recognizing a real constant in closed form, on testing an identity or inequality before proving it,
  on CRT, exact roots, finite field linear algebra, polynomial gcd or factorization mod p, max flow,
  linear extensions, and on looking up proved theorems in Lean Mathlib. Also use it for
  FrontierMath-style problems that end in one exact number.
---

# Math SpectronNexus

Exact math tools: exact integers, fractions or prime fields, most with a certificate.
Compute with them instead of trusting an unchecked formula of your own.

## Sequence or counting problems

1. Compute 20-40 small terms (minimum 8) with `calculate` loops (`sum`, `prod`,
   `count`, `binomial`, `totient`, `mobius`, `sigma`, `powmod`, `isprime`).
   Note the first index as `offset`.
2. `oeis_lookup` the terms (6+). A match is evidence; check the definition.
3. `explore_sequence` with `terms`, `offset` and your guesses as `claims`
   (expressions in `n`, refuted with a counterexample). It finds polynomials,
   linear/holonomic recurrences, algebraic generating functions,
   multiplicative, binary (a(2n), a(2n+1)) and self-referential relations,
   also after transforms. `target: {index, modulus}` or `{index, exact: true}`
   gives one far term. `search.order` names the order used, and a `program`
   field, when present, lists the tool chain that drove the search.
4. `research_sequence` with `terms` or `generator` and `question: {index,
   modulus?}` plans the far-term route (direct, CRT over factors, exact,
   many primes) and checks extra primes. Read `answer`, `status`, `direction`,
   `solution`.
5. A direction held on unseen terms modulo three primes but is not proved.
   Say so; prove it or cite `search_theorems` when certainty matters.

## Reading status

- `extrapolation_from_conjectured_direction`: checked, not proved.
- Evidence `holds_with_few_spare_terms` or `fits_every_term_and_predicts_integers`:
  short-data fit; add terms and rerun when possible.
- `extrapolation_from_oeis_terms`, `oeis_stored_value`: valid only if the OEIS
  entry is your sequence with the same offset. `oeis_candidates` disagree; check.
- `no_answer`: more terms, another offset, or a related quantity.
- `target.status: not_evaluated` states why.

## Other tools

- `calculate`: exact scalars, matrices, loops; `prime` for a prime field (p ≤ 65521).
  Values grow past 64 bits by themselves (BigInt, Karatsuba): factorial up to
  10^6, big `mod`/`gcd`/`div`/`powmod`/`isprime`, exact det of fraction matrices.
- `search_counterexample`: test `predicate` or `left` = `right` on a grid,
  `variables: {n: [1, 200]}`, up to 2,000,000 cases.
- `chinese_remainder`: up to 100,000 residues, moduli of any size; `symmetric`
  for the value nearest 0.
- `nth_root`: real k-th root to `digits` with a bracket, or all roots mod p.
- `identify_constant`: closed form from 40+ digits, optional `basis`
  (`pi^2`, `log2`, `zeta3`, ...) or `algebraic_degree`.
- `constant_digits`: up to 3,000 digits of pi, e, gamma, catalan, zetaN, logN, sqrtN expressions.
- `recurrence_term`, `weighted_walk_count`: huge indices (100,000 digits) mod p.
- `finite_field_rank`, `finite_field_linear_system`, `matrix_determinant`,
  `matrix_inverse`, `matrix_multiply`: linear algebra with witnesses.
- `finite_field_polynomial_gcd`, `factor_polynomial`: coefficients ascending.
- `max_flow` (min-cut certificate), `count_linear_extensions`.
- `search_theorems`: Mathlib by Loogle `pattern` (`"Nat.choose _ _ = _"`) or
  plain `query`; network.

`include_certificate: false` keeps replies short.

## Limits

8-5,000 terms. Holonomic order ≤ 5, degree ≤ 6. `index` and `modulus` may be
decimal strings of any size; the tool climbs tiers (residues, exact, big
Kitamasa, CRT over prime power parts with Pollard rho and ECM) and names
one in `tier`. Exact holonomic terms reach 10^7; mod a prime in [2^20, 2^62) any
term below the prime. Self-referential targets and Euler/binomial paths
reach 2,000,000 mod a prime < 2^62 (20,000 otherwise, 2,000 exact). Past the
modulus: order one mod p ≤ 4·10^12 or any p^k (2^64, 10^30), a leading
coefficient with no root mod p (derangements), Berlekamp–Massey for
p ≤ 65,536, periods, the Lucas property, constant terms CT[P^n Q] and
b(n) − μ b(n−1) = CT[P^n Q] (Motzkin, Fine mod any p^k ≤ 2^21) and binomial
sums such as Apéry (p^k ≤ 2^22). Fitted
forms are checked, not proved; `via` says so. Otherwise `no_answer`, never
a guess.
Values over 200,000 digits come back summarized; ask for a modulus.
