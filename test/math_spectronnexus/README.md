# Math SpectronNexus

Math SpectronNexus chooses and applies discrete mathematical operations. Its
results contain an answer, a worked explanation, and a checked certificate.
The native solver serves the same operations through the CLI and local MCP.

## Mathematical tools

- **Finite field rank** checks an echelon witness E and matrices P and Q.
  It verifies P·A = E and P·Q = I modulo the prime and counts the pivots of E.
  The rank tool accepts matrices up to 14×14 over a prime up to 257.
- **Integral maximum flow** verifies capacities and conservation and checks
  that the flow value equals the capacity of a cut. The exact teacher finds
  a shortest residual path with one search per augmentation. The tool
  accepts up to 14 vertices and capacity up to 4096 per directed pair.
- **Finite field linear systems** finds the rank, a particular solution and a
  full kernel basis, or a left-kernel witness y with yᵀA = 0 and yᵀb ≠ 0.
  The tool accepts matrices up to 64×64 over a prime up to 65,521.
- **Matrix arithmetic** multiplies two integer matrices with optional prime
  modulus, and computes determinants and inverses modulo a prime. Matrix
  dimensions are limited to 64. Integer products use int64 while every entry
  is within ±1,000,000. Past that they use big integers, with entries given as
  decimal strings of any length. A singular inverse request returns a nonzero
  kernel vector.
- **Polynomial GCD over a finite field** returns the monic gcd d, quotients
  f/d and g/d, and Bézout coefficients s and t with s·f + t·g = d. It
  accepts degree up to 511 over a prime up to 65,521. Coefficients go in
  ascending degree order, and an empty array represents zero.
- **Linear extension counts** counts the total orders of a poset on up to 20
  elements. It factors interleaved or ordered blocks and uses subset dynamic
  programming for the remaining blocks. The checker verifies the partition
  and counts each remaining block in the reverse order. Counts are decimal
  strings, which preserve values such as 20! in JSON clients.
- **Finite field factorization** handles nonzero polynomials up to degree 128.
  It checks irreducibility using Frobenius powers and polynomial GCDs, and
  reconstructs the input from the factors, their multiplicities and the
  leading coefficient.
- **Large-index recurrence** accepts an existing linear recurrence of order
  up to 512 or recovers the shortest one from up to 4,096 samples. For
  inference, give a promised order bound and at least twice as many samples
  as the bound.
- **Weighted walk counts** evaluates a transfer matrix up to 128×128. It
  reconstructs a recurrence for the requested matrix entry and checks the
  result against independent matrix exponentiation. Both this tool and the
  recurrence tool accept nonnegative indices of up to 100,000 decimal digits.
  The index is turned into bits 30 at a time by long division on a big
  integer. A 5,000-digit index takes 0.02 s.

Vertices and pivot columns start at zero. The policy learns rank, flow and
polynomial GCD within its state width. For the polynomial policy that means
degree up to 13 and prime up to 257. The exact solver handles the full tool
limits.

## Build and train

From the repository root on macOS:

```sh
make -C test/math_spectronnexus train ARGS="--backend=metal --tokens=1000000000"
```

The default budget is **1,000,000,000 structural input tokens**. One policy
state is a fixed 256 slot input consisting of a grid, three vectors, and
state controls. That includes zero slots for unused rows and columns. The
budget is 3,906,250 complete states used for model updates. The counters
separate supervised states, RLVR states, and sampled rollout states.
Rollouts that do not contribute to an update are excluded from the budget.

The generator draws problems from a seed as training runs. The trainer
keeps one batch, a rollout group, and a bounded pool of hard problems in
memory. The default run starts with 100,000,000 supervised tokens, then
attempts one RLVR group every four optimizer steps. Supervised batches
continue between groups.

- **Supervised training** follows exact Gaussian elimination, Edmonds–Karp,
  and Euclidean polynomial division. Polynomial cases include planted
  common factors, repeated roots, and long Euclidean remainder chains.
- **RLVR** samples several action sequences for the same problem. Each verified
  certificate earns a reward with a small bonus for lower work. The update
  uses each reward minus the mean reward of the other sequences in its group.
  If all attempts fail, the exact solution goes into a supervised batch.
- **Adversarial hard mining** tracks pass rates by variant and revisits
  misses after transformations that preserve the answer. For polynomials
  it replaces f by f + a·g or exchanges the inputs. It also samples fresh
  examples.

The checkpoint stores the policy, Adam moments, token counters, seed, and
mining pool in a NIMQ file. Run the same command to resume. Earlier
checkpoints can resume with the expanded problem mix. Use a separate
checkpoint path for an independent run.

```sh
make -C test/math_spectronnexus train \
  ARGS="--backend=metal --tokens=262144 --warmup-tokens=32768 --batch-states=64 --rollout-steps=32 --rl-every=1 --log-every=1 --checkpoint=/tmp/math-nexus-small.nimq"
```

Training defaults to Metal on macOS. Metal runs the matrix multiplications
for the policy layers and their gradients. A Metal failure stops training.
The exact solver, action distribution and Adam update run on the host.
Set `--backend=vulkan` or `--backend=cpu` to choose another backend explicitly.
Set `--rl-every=0` to run supervised training or
`--no-hard-mining` to use uniform problem selection for RLVR. Token counts
must be multiples of 256.

## Train for quality on Metal

```sh
make -C test/math_spectronnexus train-quality ARGS="--backend=metal --tokens=1000000000"
```

The profile uses 384 hidden units, a 200 million token supervised warmup
for the default budget, and eight rollouts per RLVR problem. Four problems
share each RLVR batch by default. Change that with `--rl-problems`. Forced
moves and zero weight decisions are excluded from the GPU batch. They
still count as states processed by the solver.

The policy scores valid pivot and Euclidean moves. For flow, it scores
edges along shortest paths in the residual network. Distances and
bottleneck capacities are computed once per augmentation. The flow teacher
chooses the widest shortest path, and the reward favors verified solutions
that take less work. Hard mining revisits verified problems when sampled
trajectories differ in work by at least 10%. Older quality checkpoints
pick up these rules on resume.

Every 500 updates and at completion, validation measures greedy policy
pass@1 on 10,000 distinct problems, 1,000 per variant. The best checkpoint
is selected by verified solves, then by its least successful variant, then
by fewer actions. Confirmation uses another 10,000 distinct problems with
no overlap with validation. Both scores count certified first attempts
in the policy mode shown by `policy_mode`. Use `--raw-policy` to measure
the same checkpoint with unrestricted actions, or `--constrained` to
apply the new rules to an older checkpoint for evaluation.

The latest checkpoint is `models/math_policy_quality.nimq` and the best
checkpoint is `models/math_policy_quality.best.nimq` in this folder. Repeat
the training command to resume. Evaluate the saved best with:

```sh
make -C test/math_spectronnexus eval ARGS="--backend=metal --eval-per-variant=1000"
make -C test/math_spectronnexus eval ARGS="--backend=metal --eval-per-variant=1000 --confirmation"
```

## Research track

The research track mixes four mathematical areas. Finite-field linear algebra
uses matrices with hidden rank and difficult pivots. Graph theory and
combinatorics use dense flow networks with relabelled vertices. Algebra uses
long polynomial Euclidean chains. Discrete dynamics uses sparse or dense
recurrences and weighted walks that require a transfer-matrix reduction.
The policy receives structured state and proposes its next mathematical
operation. Exact transitions, certificate checks, RLVR, and adversarial
mining train those choices across this mixture. Factorization is also
available as an exact tool.

For recurrences, each action tests one candidate order with an exact Hankel
fit over the prime field. A nonsingular fit that misses a sample proves the
minimal order is higher, so lower orders leave the action mask. Every
system above the minimal order is singular. The search ends at the first
fit that reproduces all samples, which is the Berlekamp–Massey recurrence.
The policy learns which order to test next. The teacher bisects between the
proven floor and the lowest singular order above it. Research rollouts
allow 128 actions, the same as the quality profile.

Validation and confirmation use separate sets over primes reserved for
evaluation. The generated tasks are original and their score measures this
track only. The curriculum has four internal stages. It starts with small
instances, then adds longer operation chains, harder parameters, and
adversarially combined structures. The last stage is research-inspired
synthetic work, not a reproduction of mathematicians' research problems.
This is not an official FrontierMath tier assignment or a claim of benchmark
equivalence. FrontierMath spans hundreds of novel problems,
with Tier 4 framed as research projects. It evaluates whether a solver can
discover a method and return a checkable answer. Its benchmark problems must
stay out of training data.
[Epoch AI's tier criteria and benchmark description](https://epoch.ai/frontiermath/tiers-1-4/about).

From the repository root:

```sh
make -C test/math_spectronnexus data ARGS="--research --samples=3 --seed=42"
make -C test/math_spectronnexus train-research ARGS="--backend=metal --tokens=1000000000"
make -C test/math_spectronnexus eval ARGS="--research --backend=metal --confirmation"
```

Research checkpoints use `models/math_policy_research.nimq` and
`models/math_policy_research.best.nimq` inside this folder. The research
policy supports primes up to 257 and orders up to 14. The exact tools support
primes up to 65,521 and can run before training.

```sh
NEXUS="$PWD/test/math_spectronnexus/nimcache/math_spectronnexus"
printf '%s\n' '{"kind":"recurrence_term","prime":101,"order_bound":3,"samples":[0,1,1,2,3,5],"index":"100000000000000000000"}' |
  "$NEXUS" solve --exact
printf '%s\n' '{"kind":"weighted_walk_count","prime":101,"matrix":[[1,1],[1,0]],"source":0,"sink":1,"length":"100000000000000000000"}' |
  "$NEXUS" solve --exact
printf '%s\n' '{"kind":"factor_polynomial","prime":2,"polynomial":[1,0,0,0,1]}' |
  "$NEXUS" solve --exact
```

## Solve from the command line

```sh
make -C test/math_spectronnexus data ARGS="--samples=3 --seed=42"

NEXUS="$PWD/test/math_spectronnexus/nimcache/math_spectronnexus"
printf '%s\n' '{"kind":"finite_field_polynomial_gcd","prime":7,"f":[-1,0,0,1],"g":[-1,0,1]}' |
  "$NEXUS" solve --exact
```

The JSON contains `answer`, `solution`, `certificate`, `verified`, and
`method`. For rank, flow, polynomial GCD and matrix arithmetic, `method_steps` records the row
operations, augmenting paths, or Euclidean operations actually executed.
The policy can choose these operations directly. Exact arithmetic computes
the answer and checks the certificate.

With a checkpoint, `solve` first tries the policy. A second attempt restricts
its choices to mathematically viable moves and reports `method=policy_guided`.
An exact solve after both attempts reports `method=exact_fallback`.
A quality checkpoint's first attempt uses its saved constraints and reports
`method=policy_constrained` when it certifies. Use `--policy-only` for a
single policy attempt or `--exact` for the exact solver.

## General calculator

The `calc` command accepts infix expressions and parentheses. The `calculate`
MCP tool and JSON `solve` support the same syntax and named variables.

```sh
./test/math_spectronnexus/nimcache/math_spectronnexus calc '(7+5)/8'
printf '%s\n' '{"kind":"calculate","variables":{"A":[[1,2],[3,4]],"B":[[5],[6]]},"expression":"inv(A)*B"}' | ./test/math_spectronnexus/nimcache/math_spectronnexus solve
```

Use `+`, `-`, `*`, `/`, `^` and `%`, or functions such as `sqrt`, `gcd`,
`lcm`, `abs` and `factorial`. Matrix functions include `transpose`, `trace`,
`det`, `inv`, `rank`, `rref`, `solve`, `eye`, `hadamard` and `kronecker`.
Matrix division `A/B` means `A*inv(B)`. `solve(A,B)` computes `X` in
`A*X=B` for an invertible square `A`. Set `prime` for arithmetic over
that prime field. JSON expressions can also use an operation node, for
example `{"op":"add","args":[1,2]}`.

Loops bind a variable: `sum(k,1,n,expr)`, `prod(k,1,n,expr)` and
`count(k,1,n,condition)`. `root(x,k)` returns an exact rational k-th root, or
the smallest root in the prime field. Number theory functions are `binomial(n,k)`,
`powmod(a,e,m)`, `isprime(n)`, `mobius(n)`, `totient(n)` and `sigma(k,n)`.
`div(a,b)` is floor division. Comparisons `eq`, `ne`, `lt`, `le`, `gt` and
`ge` return 1 or 0, and they combine with `and`, `or`, `not`, `if(c,a,b)`,
`min` and `max`. One expression may run up to 50,000,000 evaluations, and a
loop may cover up to 10,000,000 values.

Numbers use exact fractions. Use decimal literals in the expression or
quoted decimals or fractions in JSON variables. Matrix dimensions are limited
to 64. A square root returns a rational value when one exists, or a root in
the selected prime field.

Exact arithmetic works in tiers, so nothing stops at 64 bits:

1. A fraction starts with an int64 numerator and denominator. Every sum and
   product checks for overflow.
2. An operation that would overflow is redone with arbitrary precision
   integers (base 10^9 limbs) and schoolbook multiplication.
3. Once both factors have 48 limbs or more (about 430 digits), products switch
   to Karatsuba. Long division is Knuth's algorithm D.
4. A result that fits in int64 again drops back to the first tier.

`mod`, `gcd`, `lcm`, `div`, `powmod`, `abs`, comparisons and `root(x,k)`
accept integers of any size. `isprime` is deterministic Miller–Rabin in
int64 and a 20-base strong probable prime test above. `factorial(n)` is
exact up to n = 1,000,000 with a balanced product tree. `binomial(n,k)` with
n up to 30,000,000 multiplies the prime powers p^e that Legendre's formula
gives, in a product tree. binomial(2·10^6, 10^6) takes 1.2 s. Past that n it
steps through min(k, n−k) ≤ 200,000 factors. `x^k` takes a scalar exponent up to
10^6 and a matrix exponent up to 4,096. `totient`, `mobius` and `sigma`
factor any positive int64 with trial division, Miller–Rabin and Pollard rho
(Brent). A value is bounded only by 10,000,000 decimal digits.

On this machine (Intel MacBook), the exact determinant of the 50 × 50 Hilbert
matrix (1,467 digit denominator) takes 0.7 s. factorial(100000), which has
456,574 digits, takes 0.65 s. All 25 big-number checks match Python's
integers and fractions.

## Helper tools for a language model

Three tools target two common weaknesses of a language model working on
competition and research problems. It may trust a formula it has not checked,
or lose track of a long calculation. The intended loop is that the model
computes small cases, for example with a brute-force count in `calculate`. It
then sends the terms and its formula claims to `explore_sequence`, reads the
reported directions, and decides what to prove or compute next.

- **`explore_sequence`** takes `terms` with an `offset`, or a calculator
  `generator` in `n` with `start` and `count`. Each claim in `claims` is checked
  exactly and refuted with the first failing index. The search tests
  polynomials, linear recurrences (Berlekamp–Massey), holonomic recurrences
  of order up to 5 with coefficient degree up to 6, algebraic generating
  functions of degree 2 or 3,
  multiplicativity, and periodic residues. It also applies up to two
  transforms: differences, partial sums, ratios, division or multiplication
  by n!, the binomial and inverse binomial transforms, Möbius inversion, and
  the inverse Euler transform. A direction is fitted on early terms and must
  hold on later terms modulo three primes. Coefficients are reconstructed as
  rationals. A `target` with `index` and `modulus` (or `exact: true`) evaluates
  a far term from the first direction. It uses Kitamasa for linear
  recurrences and iteration for holonomic ones, and inverts the transforms
  of a transformed direction. Every direction and target is a conjecture
  until proved.

  The index and the modulus may be decimal strings of any size. A target
  climbs tiers until one works, and `target.tier` names the one used:

  1. Residues modulo M, in int64, for M up to 2^62. Residue code only adds
     two residues and multiplies with an overflow-free mulMod.
  2. The exact value, then reduced modulo M. It is also the fallback when a
     composite M shares a factor with a leading coefficient.
  3. Kitamasa with big integer residues, for a linear recurrence at a huge
     index and a huge modulus, such as Fibonacci(10^100) mod 10^60+7.
  4. CRT over the prime power parts of the modulus, of any size: trial
     division below 10^5, then Pollard rho (Brent, with batched gcds) on big
     integers until every piece fits int64, where Pollard rho on int64
     finishes. A modulus of four primes near 10^6 (about 10^24) splits in
     0.3 s. When Pollard rho stalls, Lenstra's elliptic curve method takes
     over: Montgomery curves with Suyama's parametrisation, stage 1 to B1
     and a baby-step giant-step stage 2 to 50·B1, with B1 from 2,000 to
     250,000 within 40 s. (10^16+61)(10^17+3) splits in 5 s and a product of
     two primes near 10^20 and 10^21 in 28 s. A part up to 2^62 takes the
     residue routes. A larger part, such
     as 2^64 or the 5^30 inside 10^30, takes the p-adic route of an
     order-one relation with big integer residues.

  Inside the residue tier, these routes reach past the iteration windows:

  - **Hypergeometric terms.** These follow an order-one relation
    a(m+1) = R(m)·a(m). Modulo a prime p ≤ 4·10^12 they reach any int64 index
    when R splits into linear factors over Q. a(N) is then a(m0) times
    products of arithmetic progressions Π(u·k + v). Wilson's theorem gives
    each product's unit part, since every full block of p terms contributes
    −1. The terms divisible by p form another progression with the same
    step, and the recursion on them tracks the p-adic valuation. This
    generalises Lucas' theorem: a leading coefficient that vanishes mod p
    causes no trouble. Above p = 10^7 the unit parts of long progressions
    come from the square root product below. Catalan(10^18) mod 999983
    takes 0.1 s, and Catalan(10^17) mod 10^9+7 1.6 s.

    The same recursion works modulo a prime power q = p^k (k ≥ 2, p up to
    5·10^7). By Gauss's generalisation of Wilson's theorem, q consecutive
    terms multiply to −1, or to 1 for 2^k with k ≥ 3. The rest of a
    progression splits as t = s + p·j. The classes with s ≠ hit hold units
    c_s + u·p·j, whose product is F(u·p·j) with F(X) = Π(c_s + X). If a
    polynomial g(y) has its y^i coefficient divisible by p^i, its terms
    of degree k and up vanish modulo q. Such polynomials stay that way
    under products and integer shifts y → y + m. So the product
    P_m(y) = Π_{j<m} g(y + j) is kept to degree < k and doubles as
    P_2m(y) = P_m(y)·P_m(y + m). That costs O(k²) per bit of the count,
    whatever the size of q. Catalan(10^18) takes 0.1 s mod 10^9, 0.5 s
    mod 2^61, 0.3 s mod 3^39 and 1.9 s mod 2^64 (big integers). The exact
    values up to n = 3,000 match modulo prime powers up to 2^62, as do
    those of C(3n, n), whose step u = 3 is divisible by p = 3.
  - **Unit leading coefficient.** When the leading coefficient L(n) of a
    holonomic relation has no root modulo p (checked by
    gcd(L, x^p − x) over F_p), every step can be taken modulo p^k, and the
    step matrices repeat. The block matrix B(x) = diag(M(x), L(x)) at
    x = x0 + s + p·y gives one row of p steps, G(y) = Π_s B(x0 + s + p·y).
    Its entries are polynomials in y whose y^i coefficients are divisible
    by p^i, so they are kept to degree < k. J rows double as
    P_2m(y) = P_m(y + m)·P_m(y); for k = 1 that is G(0)^J. One division by
    the product of L(x) ends it. Derangements, involutions and
    Π(k² + 1) reach any index: derangements(10^18) mod 10^9+7 takes 2.1 s
    (the row is one square root product), mod 2^40 0.1 s.
  - **Recurrences of the residues.** Modulo a prime up to 65,536, a sequence
    is often linear recurrent; Touchard's congruence for Bell numbers is one
    case. Berlekamp–Massey finds the shortest recurrence on up to 16,384
    computed terms. It is used only if it holds on at least a quarter of
    them beyond the 2L terms that fix it. Kitamasa then jumps to any index.
    Bell(10^18) mod 13 and derangements(10^15) mod 101 come from it. It
    applies to transformed, self-referential and holonomic directions.
  - **Eventual periods.** Modulo a composite number up to 65,536 there is no
    field for Berlekamp–Massey. The residues are tested for an eventual
    period instead: a pre-period, then two full periods and a margin.
    Bell(10^18) mod 64 and derangements(10^15) mod 1000 come from it.
  - **Lucas property.** Many holonomic sequences satisfy
    a(np + j) ≡ a(n)·a(j) mod p for every prime p: central binomials,
    Apéry and Franel numbers. The tool checks this on 360 exact terms modulo
    the primes 5 to 31 (at least 1,000 checks, offset 0). If it holds,
    a(N) mod q is the product of a over the base-q digits of N, each digit
    term below q. The result's `via` says this route was checked, not
    proved. Apéry(10^18) mod 1000003 takes 1.1 s.
  - **Constant terms.** Ballot-type sequences such as Motzkin, Riordan and
    Schröder numbers lack the Lucas property, but many are constant terms
    a(n + s) = CT[P(x)^n Q(x)] with Laurent polynomials P and Q. The tool
    fits P, Q on x^-4..x^4 and the shift s ≤ 2 on 64 terms of the
    relation, with 24 spare equations, and checks the fit modulo three
    primes. P has three terms a/x + b + c x (|a|, c ≤ 4, |b| ≤ 6), or four
    to five terms with ends ±1 or ±2 and inner terms in [−2, 2]. Modulo a
    prime p, P^p = P(x^p), so a_R(n p + j) = a_S(n) with S = Λ_p(P^j R):
    the Cartier operator Λ_p keeps the coefficients at multiples of p. S
    keeps a bounded support, so each base-p digit needs a few coefficients
    of A^α·B^β, with A and B the low and high ends of P. One engine finds
    them from C·F' = E·F, a recurrence whose leading factor C0·(t + 1)
    first vanishes at t + 1 = p. A walk that must pass it runs modulo p²,
    where the single factor p of the running denominator divides out
    exactly. Below p = 2^31 a residue is one int64. A prime from 2^31 to
    2^43 (8.8·10^12) keeps a residue as a pair of base-p digits: a product
    of the two low digits splits by a floating-point quotient corrected in
    int64, and the exact products, below 2^110, come back through four NTT
    primes (2^115) and Garner's mixed radix over the two digit lists. So a
    P of four or five terms passes t = p up to 2^43 as well. The square
    root product carries long stretches.
    Motzkin(10^18) mod 10^9+7 takes 2.7 s, mod 10000019 0.5 s.
    A five-term P that is the square of a three-term P3, such as
    P = (1/x + 1 + x)^2 for Riordan(2n), is evaluated as CT[P3^(2n) Q]. Its
    walks stay below p, so any prime up to 4·10^12 works: Riordan(2n) at
    10^18 takes 1.4 s mod 1000003 and 22 s mod 2147483659.

    A first-order inhomogeneous form b(n) − μ·b(n − 1) = CT[P^n Q], with P
    of three terms, covers Fine's numbers (2F(n) + F(n − 1) = C(n)).
    b(N) = μ^N·(b(0) − Q_0 + S(N)), where S(N) sums CT[Q·y^m] over m ≤ N
    for y = P/μ. The base-p digits of N carry two states, one for the
    prefix and one for the prefix minus one. Each digit d needs the
    partial geometric sum T_d(y) = 1 + y + ··· + y^d. Its coefficients
    come from one joint walk that never crosses p. Fine(5·10^6) mod 1000003
    takes 3.4 s, and Fine(10^18) mod 10^9+7 14 s.

    Modulo a prime power p^k (k ≥ 2), P^(p^k) ≡ P(x^p)^(p^(k−1)) mod p^k
    (Rowland and Yassawi). With h = p^(k−1) and n = m + p^k·t, the state
    starts as Λ_p(Q·P^m). Each base-p digit j of t multiplies it by P^(h·j)
    and applies Λ_p again. The state is a polynomial of degree about
    h·deg P, so this route takes p^k ≤ 2^21. Motzkin(10^6) mod 1024
    takes 0.1 s. Motzkin(10^18) mod 10^9 takes 63 s, most of it in the
    5^9 part. The `via` field names every fitted form and says it is
    checked, not proved.

    The inhomogeneous form b(n) − μ·b(n − 1) = c(n) = CT[P^n Q] works
    modulo p^k for every P of the fit and every μ. Denominators p^d in Q
    are cleared by working modulo p^(k+d) and dividing at the end.
    - μ a p-adic unit: b(n) = μ^n·(b(0) − Q_0 + S(n)) with S(n) the sum of
      CT[Q·Y^m] over m ≤ n, Y = P/μ. With Z = Y^h and m = r + p^k·t, each
      term is CT[Λ_p(Q·Y^r)·Z^t]. The sum keeps two states: A for the full
      blocks t < T and B for the last one. A base-p digit j of T maps them
      to A' = Λ_p(A·H_p) and B' = Λ_p(A·H_j + B·Z^j), with
      H_j = 1 + Z + ··· + Z^(j−1), since Z^p = Z(x^p) mod p^k. The sum is
      CT[B]. Motzkin partial sums (b^2 ≠ 4ac) at 10^18 take 3.5 s mod 3^9.
    - p divides the denominator of μ, as for Fine's numbers modulo 2^k:
      ν = 1/μ is divisible by p, and
      b(n) = −Σ_{j ≤ L} ν^j·c(n + j) + ν^L·b(n + L). Once p^k divides ν^L,
      this is one homogeneous constant term,
      −CT[P^(n+1)·Q·Σ_{j ≤ L} ν^j P^(j−1)]. Fine(10^18) mod 2^16 takes
      5.5 s.
    - p divides the numerator of μ: b(n) is
      CT[P^(n−L+1)·Q·Σ_{j<L} μ^j P^(L−1−j)] in the same way.
  - **Binomial sums.** Apéry and Franel numbers have the Lucas property
    modulo p only, and their constant terms need two or three variables.
    The tool also fits a(N) = c·S(N + s) with S(n) = Σ_k w^k Π C(T, B)^e.
    The products have up to four factors from C(n,k), C(n+k,k), C(2k,k),
    C(n,2k), C(n−k,k), C(2n−2k,n−k), C(n+k,2k) and Cat(k), with weights
    w from ±1 to 27. The fit needs 40 terms of the relation and holds
    modulo two primes. Granville's theorem gives a binomial modulo p^k
    from the windows X_j = ⌊X/p^j⌋ mod p^k of its top, bottom and rest:
    C(T, B) ≡ (±1)^(e_(k−1))·p^(e_0)·Π_j (T_j!)_p / ((B_j!)_p (R_j!)_p),
    with e_0 the carries of B + R and (x!)_p the product of the units up
    to x. Each factor depends on one window. The window of a form
    a·n + b·k + c is a·n_j + b·k_j + carry_j, so a digit DP over k keeps
    the last k − 1 digits of k and one carry per form. A form that ends
    negative leaves a carry, which removes the k out of range. This works
    for p^k ≤ 2^22. Apéry(10^18) mod 5^9 takes 10 s and mod 3^7 0.02 s.
    The same DP sums Fine's form modulo odd prime powers: with
    b^2 = 4ac, P = (a/x)(1 + t·x)^2, so every CT[P^m x^j] is one binomial
    C(2m, m − j), and the sum over m ≤ n is a binomial sum with the factor
    C(n − m, 0). Fine(10^18) mod 3^13 takes 33 s.

  - **Square root product.** For a holonomic relation of any order past
    10^6 steps, modulo a prime from 2^20 to 2^62, the step matrices are
    multiplied in O(√N log N) by the baby-step giant-step method of Bostan,
    Gaudry and Schost. With P_d(x) = M(x+d−1)···M(x), the samples
    P_d(x0 + i·v), for i = 0..D·d, fix a polynomial of degree D·d in i.
    Doubling d takes two Lagrange shifts of those samples, each one NTT
    convolution, and a pointwise product P_2d(x) = P_d(x+d)·P_d(x). The
    giant steps P_v(x0 + i·v) then cover the range. The index must stay
    below the modulus, or a leading coefficient vanishes. Motzkin(10^8)
    mod 10^9+7 takes 1.2 s, Apéry(5·10^7) mod 998244353 0.8 s, and
    Motzkin(2·10^7) mod 10^12+39 1.3 s.

  Between the iteration window and the prime, a failed iteration (a leading
  coefficient that vanishes modulo a small prime) falls through to the
  routes past the prime instead of stopping.

  What remains out of reach returns `no_answer` with the reason, never a
  guess. Most of it is the cost of the mathematics, not a missing route:
  - past the prime, a holonomic relation whose leading coefficient has a
    root modulo p, without the Lucas property, a constant-term form or a
    binomial sum of the menu. No general method is known: the residues
    form a p-automatic sequence at best, and finding it needs the
    representation;
  - constant terms modulo p^k above 2^21 and binomial sums above 2^22. The
    automaton of such a sequence modulo p^k has about p^(k−1) states, so
    the state alone outgrows memory;
  - an order-one relation modulo a prime above 4·10^12 at an index past
    the prime. That needs a product of about p factors, and the best
    known method costs about √p; computing n! mod p fast is open;
  - a prime factor of the modulus above 2^63 at an index past 10^7, or a
    cofactor whose prime factors are all beyond the elliptic curve method
    (about 25 digits and up). Splitting it is integer factorisation.

  A squarefree modulus such as 10007·10009 reaches both small-prime routes
  through the CRT tier.

  Exact holonomic and linear terms use binary splitting after 2,000 steps.
  One step maps the window of the last r terms to M(n)·w / L(n). The products
  of the M(n) and of the L(n) are built as a balanced tree, so the big
  products are Karatsuba sized. The single exact division at the end divides
  only the top limbs, since the quotient is much shorter than the divisor.
  One multiplication then confirms the result. Exact iteration reaches index
  10^7, bounded also by the 10,000,000 digit limit. Catalan(200,000), with
  120,404 digits, takes 5.4 s. The 300,000th factorial takes 3.8 s.

  A transformed direction with rational coefficients also gives exact values. Its residues modulo
  primes below 2^31 are joined by CRT until one more prime agrees, so
  p(1200) from the inverse Euler transform takes 0.1 s. A fit with residue
  coefficients can extend the transformed sequence and undo the transforms
  at one of the primes used for the fit. Modulo m < 2^31,
  undoing the inverse Euler transform solves n·a(n) = Σ c(k)·a(n−k) online.
  A CDQ split over power-of-two blocks hands each finished half to one
  cyclic convolution of the block's length. Wrapped terms land below the
  outputs it needs. The weights' transforms are shared by every block of a
  length. The convolutions use NTTs with Montgomery products, joined by
  Garner's CRT: three NTT primes for a modulus below 2^31, six below 2^62.
  The butterfly loops run without bound and overflow checks, about 20%
  faster. The whole costs O(n log² n). Paths through it reach index 2,000,000;
  p(5·10^5) mod 10^9+7 takes 5.8 s, and p(10^5) mod 10^12+39 2.4 s.

  Modulo a prime between 2·10^6 and 2^62, a binomial transform is one
  convolution of exponential generating functions: B(x) = e^(±x)·S(x).
  Paths through it also reach 2,000,000. Otherwise both transforms stay
  O(n²) and reach index 20,000. `max_directions` counts formula and multiplicative
  directions. A periodic residue pattern is reported without ending the
  search.

  A multiplicative direction evaluates far terms by factoring the index into
  coprime prime powers and multiplying their values. When each prime power
  appears in the supplied terms, those values suffice. Two fitted models
  can supply values for missing prime powers. In the first, the divisor sum
  g = Σ_{d|m} a(d) is a polynomial for m ≥ 2, checked on unseen terms, and
  a(p^e) = g(p^e) − g(p^(e−1)) with g(1) = 1. This covers φ (g(m) = m) and
  μ (g(m) = 0). In the second, a completely multiplicative function has a(p)
  equal to a polynomial in p, checked on spare primes, and a(p^e) = a(p)^e.
  This covers Liouville's λ. Any int64 index is factored by trial division,
  Miller–Rabin and Pollard rho. Prime powers inside the data are read from it,
  and the rest come from the model. φ(10^18 + 9) and φ(2^62 − 1) are exact. φ(10^12) exact, φ(999999999989) and μ(999999999989)
  mod 10^9+7, and μ and λ at 10^12 − 1 all match Python.

  When the search has yet to find a general formula for far terms, it
  tries more relations on the original terms. This includes multiplicative
  directions supported by the supplied prime-power values.

  The first is a binary recursive relation. Here a(2n) and a(2n+1) are each
  a fixed combination of a(n), a(n+1), the previous term, n and 1, fitted
  separately for each parity from small sets upward. Recovered relations:
  - digit sums (A000120);
  - Stern's sequence: a(2n) = a(n), a(2n+1) = a(n) + a(n+1);
  - the ruler function and 2-adic valuation;
  - the Josephus problem;
  - binary partitions;
  - Gould's sequence;
  - A006046 and A005187.

  Without the previous-term feature a far term needs about log₂ n steps.
  Stern's a(10^18) mod 998244353 matches Python.

  The second is a self-referential relation, in which a(n+1) is a fixed
  rational combination of transforms of a(0..n):
  - sums: Σ C(n,k)·a(k), Σ a(k)·a(n−k) and Σ C(n,k)·a(k)·a(n−k);
  - Euler products: [xⁿ] Π(1−x^k)^(−a(k)) and [x^(n+1)] Π(1−x^k)^(−a(k−1));
  - simple terms: a(n), n·a(n), a(n−1) and 1;
  - quadratic terms: a(n)², a(n)·a(n−1) and a(n−1)², for maps such as
    Sylvester's a(n+1) = a(n)² − a(n) + 1.

  These relations cover sequences with no polynomial-coefficient
  recurrence. It tries one transform first, then one transform with the
  simple terms, then everything. Each fit may start at n = 0, 1 or 2, so a
  few early terms can serve as initial values. Here n counts from the first
  supplied term. The same three-prime check on unseen terms applies.

  Recovered relations:
  - Bell numbers: a(n+1) = Σ C(n,k)·a(k);
  - unlabeled rooted trees (A000081, the Euler product with a(k−1));
  - Euler zigzag numbers: a(n+1) = ½ Σ C(n,k)·a(k)·a(n−k) from n = 1.

  On 30 random six-digit numbers it reports nothing. A target modulo any
  number up to 9·10^9 is iterated in O(n) per term, up to index 20,000.
  Modulo a prime below 2^62 and above the index, the sums and products are
  online convolutions instead. The same CDQ blocks serve a·a, s·s and the
  Euler products. When one factor is fully known, as 1/k! is for the
  binomial sums, a semi-online engine does half the work. These targets
  reach index 2,000,000: Bell(10^6) mod 10^9+7 takes 12.7 s, and Bell(2·10^5)
  mod 10^12+39 5.2 s.
  Pascal rows keep binomials valid for composite moduli. The Euler products
  divide by n, so they need n invertible. An exact target, up to index 2,000,
  joins residues modulo primes below 2^31 by CRT until one more prime agrees.
  The relations: B(300) mod 10^9+7, B(300) mod 1000, all 454 digits of B(300),
  rooted trees at n = 1000 and zigzag E(500) mod 998244353 all match
  independent Python computations.

  Last come two short-data fits for shapes the regular tests skip because
  there are too few terms. Each must be the only relation of its shape and
  hold modulo all three primes. Its evidence says it rests on few spare
  terms.
  - A self-referential relation from 8 terms. Only one transform alone, or
    one transform with a(n) and 1, or the quadratic map, is tried. It keeps
    one spare and two unseen equations. Sylvester's map is found from 8
    terms and A003095's from 10.
  - A holonomic relation may use every equation to fit, with no unseen
    term. It must then predict integers for 60 further terms, iterated
    exactly. A relation forced through too few equations almost never does
    that. A constant leading coefficient makes the integer test empty, so
    it still needs two spare equations. The Apéry numbers for ζ(3) give
    their order-2, degree-3 relation from 14 terms, and a(500) mod 10^9+7
    matches Python.
  - The same holonomic fit also runs on one-step transforms that stay exact
    integers: ratio, difference, partial sum, and multiplication or
    division by n!. A transform often lowers the shape but costs a term.
    From ten terms of 1/det(H_n) for the Hilbert matrices (A005249), the
    ratio gives (n+1)²·b(n+1) = 4(2n+1)(2n+3)·b(n). Many primes and CRT then
    rebuild all 1466 digits of 1/det(H_50), equal to exact fraction
    elimination of the 50×50 matrix.

  A holonomic fit is rejected when its leading coefficient vanishes past the
  first two equations at a nonzero term. Sparse sequences admit such fits.
  They hold on every given term and fail later: for binary partitions one
  extrapolated a wrong value.

  `make -C test/math_spectronnexus bench-oeis` measures the explorer on 70
  classic OEIS entries. It runs after `make oeis-data`. Each entry gives
  its first terms, up to 30, and predicts the last stored term modulo
  10^9+7.

  | Explorer | Right | Wrong | Found, no far term | Missed | False positive |
  |---|---|---|---|---|---|
  | Before | 48 | 1 | 4 | 6 + 2 too short | 0 |
  | Now | 60 | 0 | 1 | 0 | 0 |

  - All 60 entries with structure, including the four multiplicative
    functions, give the right far term. The Apéry numbers get 14 terms,
    Sylvester's sequence 8 and A003095 10. These are all but the last
    stored term, and the short-data fits find them.
  - The ten entries without structure (primes, digits of π and e, graphs,
    groups and others) report nothing. The exception is the palindromes,
    where a residue-coefficient fit after a transform gives no far term.
- **`search_counterexample`** evaluates a `predicate`, or `left` and `right`,
  on every point of an integer grid of up to four variables and 2,000,000 cases.
  It returns up to five counterexamples. Use it to test an inequality,
  identity or divisibility claim before trying to prove it.
- **`chinese_remainder`** combines up to 100,000 residues modulo pairwise
  coprime moduli into one exact integer of any size. Moduli up to 9·10^9 use
  Garner's algorithm one prime at a time. Larger moduli, given as decimal
  strings, use the extended Euclidean algorithm on big integers. 3,000
  congruences that rebuild a 9,543-digit value take 0.16 s. Several
  `explore_sequence` targets with different moduli can recover an exact huge
  value.
- **`nth_root`** takes the k-th root of an integer or fraction `p/q` of any
  size, with k up to 100,000. Without a prime it returns the real root
  truncated to `digits` places (at most 100,000). It also says whether the root
  is exact, and gives the exact root such as `9/7` when it is. The certificate
  is the bracket y^k ≤ X < (y+1)^k for the scaled value. With `prime` (up to
  9·10^9) it lists every y with y^k ≡ value and their count gcd(k, p−1),
  using a primitive root and a baby-step giant-step logarithm. The 200th root
  of 2 to 100 digits, or of a 1,000-digit integer to 300 digits, takes under
  one second. Integer roots use Newton iteration with exact long division.
  The calculator's `root(x,k)` gives an exact rational root or a root in the
  prime field.

The explorer search order comes from a small MLP with 128 inputs, 128 hidden
units and 16 actions. It sees the sequence features and the tests already
done, then chooses the next hypothesis or transform. After 12 guided tests an
exhaustive breadth-first search covers the rest within `max_tests`. The
policy changes which tests run first, and every direction is still verified. `search.order` reports
`mlp_policy_then_exhaustive` when this policy is loaded. When a Spectron program
checkpoint is present the tool chain below drives the whole search instead, and
`search.order` reports `spectron_tool_chain`.
The policy trains
with supervision on the shortest test route to a verified formula or
multiplicative direction at the lowest transform depth. The training mix has
eleven basic families and nine hard families of 48 terms each, half and
half by default. Hard families include Apéry-like binomial sums up to
order-3 recurrences with degree-6 coefficients, order-3 holonomic
recurrences, and algebraic generating functions of degree 3. Others need
two transforms: Euler transforms of divisor sums, binomial or factorial
scalings of Euler transforms, and divisor sums of recurrences. For hard
examples the generator records the planted transform path. The teacher
uses it only to order paths within one depth, so routes stay minimal, and
the policy never sees it. Training takes a few minutes.

```sh
make -C test/math_spectronnexus train-explorer ARGS="--backend=metal"
```

Evaluation uses the MCP budget of 300 tests and fresh examples. On 300
basic sequences both orders find every direction, and mean tests fall from
9.30 to 1.01. On 360 hard sequences (40 per family) the default order
finds 315 and the policy finds 355. Mean tests fall from 94.29 to 5.38. The
default order finds none of the Euler transforms of divisor sums within the
budget. The policy finds all 40. The remaining five misses are
sum C(n,k)^3·C(n+k,k). Its smallest relation has order 5, coefficient
degree 6 and 42 unknowns, so it needs at least 53 terms. With 80 terms the
explorer finds it. Its coefficients exceed rational reconstruction modulo
one prime, so the relation is kept modulo the three detection primes. A
target modulo one of those primes still evaluates. When the early
equations leave several relations, the fit extends to every equation
except the last four. Then the canonical reduced echelon vector is chosen,
which is the same relation modulo every prime. The checkpoint is
`models/sequence_explorer.nimq`. `solve`, `explore` and `mcp` load it when it
exists. Pass `--explorer-checkpoint=path` to choose another file.

```sh
NEXUS="$PWD/test/math_spectronnexus/nimcache/math_spectronnexus"
printf '%s\n' '{"terms":[1,1,2,5,14,42,132,429,1430,4862,16796,58786,208012,742900],"claims":["2^n/(n+1)"],"target":{"index":1000000,"modulus":1000000007}}' |
  "$NEXUS" explore
printf '%s\n' '{"generator":"sum(k,0,n,binomial(n,k)^2)","start":0,"count":30}' |
  "$NEXUS" explore
```

## The tool policy

`explore_sequence` is driven by a Spectron policy: Linear++ attention with a
Mamba2SSD branch in every layer, from `test/spectron`. It never produces free
text. Each step it reads the engine's own 128 features for the sequence and for
the node the walker stands on - magnitudes, curvature, linear complexity, the
polynomial degree of the residues, the transforms already on the path - each one
bucketed into a token, together with a short fingerprint (the first 16 terms
modulo 1009), the tools that are still legal where the engine stands, the
transform depth, how much work each transform still holds below it, and the last
four tool calls with the engine's verdict on each. It then names one tool from a
fixed alphabet - the six search hypotheses, the nine transforms, pop, and, for
the holonomic fit, an order 1 to 5 and a coefficient degree 0 to 6 - and the
exact engine runs and checks that step before the next one is asked for. A step
into a node is never followed by a step straight back out: the walker tests where
it stands, or goes on, so no two calls can circle without reaching a new test.

A named shape is a hint and not a filter. The engine tries that order and degree
first and, when they do not fit, runs the same shape scan the plain order runs,
so naming the wrong shape costs a little time and never a direction; the state
then says `scan` instead of `found`, which teaches the policy what its name was
worth. The policy can name order and degree pairs the plain scan never tries,
such as a degree 0 coefficient block, so the alphabet it picks from is the plain
one plus those.

The tool chain is the search. There is no separate scan behind the policy:
`search.order` reports `spectron_tool_chain`, and `program` returns the chain
that ran, for example `["difference","P-recursive(r=2,d=1)"]` - the difference
transform followed by a holonomic fit of order 2 and degree 1. Every direction
is still fitted and checked by the same exact code, and a tool call that fails
is recorded as `miss` in the next state, so the policy sees what the engine
learned. Without a program checkpoint, `explore_sequence` falls back to the
smaller MLP ordering policy and the exhaustive order as before.

Training is supervised one decision at a time. The generator supplies 64
visible terms and computes 32 more. The teacher searches for a route that
computes the last of those extra terms modulo 1,000,000,007. A route contributes
training samples only after the engine replays it and matches that answer.
The loss supervises the tool or shape choice at the final `ASK` token.

The trainer defaults to 12,000 distinct sequences of 64 terms, with an 80%
chance of drawing from the hard families for each sequence. The hard pool
includes order 4–5 holonomic recurrences with coefficient degrees 4–6. It
also includes order 6–10 linear recurrences hidden under two transforms,
such as factorial weighting followed by a binomial transform. The holdout is
generated first with its own seed, and the train pool excludes its visible
prefixes.

The trainer groups each verified route by difficulty. Advanced routes use two
transforms, a linear recurrence of order at least six, or a holonomic fit with
at least 25 coefficient slots. Intermediate routes include a single transform,
an algebraic fit, or a recurrence order or coefficient degree above two. The
remaining routes are basic.

The sampling mix ramps linearly from **50% basic, 35% intermediate, 15% advanced**
to **10% basic, 25% intermediate, 65% advanced** over one run with the same
optimizer. When a level has no samples, the trainer proportionally distributes
its share among the available levels. Each level is shuffled before a pass.

```sh
make -C test/math_spectronnexus train-program \
  ARGS="--backend=metal --steps=4500 --batch-states=8 --pool=12000 --hard-fraction=0.8 \
  --lr=0.0003 --eval-size=128 --eval-seed=24263 --init-checkpoint=models/sequence_program.nimq \
  --checkpoint=models/sequence_program_curriculum.nimq"
```

This command starts with the saved weights and a fresh Adam optimizer. The
new checkpoint records the cumulative training steps. Without
`--init-checkpoint`, training initializes a new model.

The holdout reports `correct`, `wrong`, `no_answer`, and mean tests for the
policy and plain scan. Both orders search for one strong direction under the
same test budget, then evaluate the target. Use `eval-program` with the same
holdout size, seed, and test budget to compare checkpoints.

```sh
make -C test/math_spectronnexus eval-program \
  ARGS="--backend=metal --eval-size=128 --eval-seed=24263 --max-tests=300 --checkpoint=models/sequence_program.nimq"
```

The default inference checkpoint is `models/sequence_program.nimq`.
`explore`, `solve` and `mcp` load it when it exists. Pass `--program-checkpoint=path`
to choose another file. The policy drives `explore_sequence` only: `research_sequence` plans its
routes with the plain explorer, so a policy checkpoint cannot change a
planner's answer.

Training runs on the GPU backends only, one session per batch item and head in
every layer, so the step cost follows `batch * heads * layers`. On the
integrated Iris Plus 640 a step costs about 1.3 s at batch 8, hidden 64, heads 4,
two layers with the state of 180 tokens and vocabulary 1631.

The earlier `models/sequence_program.nimq` checkpoint came from 900 steps with
7,896 teacher samples from 3,000 generated sequences of 48 terms. Loss went
from 8.40 to 0.55 in 1,137.3 s of training. On 64 fresh
holdout sequences the policy and the plain scan find the same directions, and
the policy does it in a sixth of the tests.

| order | found | mean tests | mean calls |
| --- | --- | --- | --- |
| policy `spectron_tool_chain` | 64/64 | 7.83 | 17.2 |
| plain scan | 64/64 | 50.89 | - |

A second set of 40 fresh sequences, measured outside the trainer with the MCP
budget, agrees: 39/40 for both orders, 14.57 tests on average for the policy
against 49.02 for the plain scan, 31.6 calls on average, three sequences past 60
tests, and the whole set in 88.9 s. Both orders answer the same far terms -
Catalan(1000) mod 10^9+7 is 110961515, Fibonacci(500) 523068127 and Motzkin(301)
626852939, each in 3 tests through `spectron_tool_chain`. The run benchmark of 81
questions stays at 81 right, 0 wrong, 0 no answer, in 62.44 s.

## Constant recognition

Many hard answers end as a real number whose closed form is unknown. Two tools
cover that step.

- **`constant_digits`** evaluates an expression in named constants to up to
  3,000 digits. The constants are `pi`, `e`, `gamma`, `catalan`, `phi`,
  `logN`, `zetaN` and `sqrtN`. The operations are `+ - * /`, integer powers,
  `sqrt`, `root(x,k)`, `log`, `exp` and `zeta(n)`. Values use fixed-point
  BigInt arithmetic with 15 guard digits:
  - π by Machin's formula;
  - logarithms by atanh series;
  - ζ(n) and Catalan's constant by the Cohen, Rodriguez Villegas and
    Zagier acceleration of alternating series;
  - γ by Brent–McMillan.

  At 150 digits every constant matches independent Python references: AGM
  for π, Apéry's series for ζ(3), Euler–Maclaurin for ζ(5) and γ, and
  Ramanujan's series for Catalan's constant.
- **`identify_constant`** finds an integer relation among the value and a
  basis with exact integral LLL (Cohen, Algorithm 2.6.7). It returns a
  closed form such as `(7*zeta3 - 2*pi^2*log2)/8`, or a candidate minimal
  polynomial with `algebraic_degree`. Without a basis it tries 14 common
  bases in turn: rationals, powers of π, logarithms, ζ values, Catalan's
  constant, square roots, e and γ. A relation must hold to the supplied
  precision. It must also leave at least `min_excess` (default 10) digits
  between the lattice precision and the total digits of its coefficients,
  so a chance fit is rejected. On 60 random digits all 14 bases report no
  relation. The result is strong evidence, not a proof.

## Tool-chain planner

`research_sequence` answers a question about a sequence: the value at a far
index, exactly or modulo M. The planner covers any int64 index and modulus.
An index or modulus past 64 bits, given as a decimal string, goes straight to
the explorer's target tiers. `tier` then reports the tier used. A second light MLP (64 features, 128 hidden
units, 9 actions) plans which exact tool to call next:

- explore the sequence;
- request more terms from a generator;
- evaluate the target modulo M directly;
- evaluate modulo each prime factor of M and combine with CRT;
- iterate exactly;
- evaluate modulo many primes and combine with CRT, starting from a digit
  estimate and adding primes until one more prime leaves the value unchanged;
- check a candidate against an extra prime;
- answer, or give up.

Its features are structured: the question type, the factors of M, the
direction found and its transforms, which tools failed, the candidate
answer, and checks so far. It never writes text. A self-referential
relation sets the holonomic feature, since both are iterated term by term.
The checkpoint therefore keeps its 64 inputs.

Training starts with supervision from a teacher. The teacher tries each
route on a copy of the state and takes the cheapest one that gives the
true value. Reinforcement learning follows, with four rollouts per task
and rewards relative to the group mean. A right answer earns 1 and a wrong
one costs 1. Giving up earns 0, and every tool call costs 0.02. Tasks are
far terms of the basic and hard explorer families, asked modulo primes,
modulo composite numbers such as 210 or 6·998244353, or exactly. The true
value is computed independently.

`--far-fraction` (default 0.4) sets the share of far tasks in the training
pool and the held-out set. Far tasks cover what the older checkpoint never
saw:

- multiplicative functions (φ, σ, τ, μ, λ) at indices up to 10^15;
- linear recurrences at huge indices;
- hypergeometric and holonomic terms modulo wide moduli such as 2^61-1,
  10^18+9 or 10^12;
- long exact terms.

```sh
make -C test/math_spectronnexus train-planner ARGS="--backend=metal"
```

The default run has 600 training tasks, 1,500 supervised steps and 300
RL iterations. It takes a few minutes on Metal. On 200 held-out tasks:

| Planner | Right | Wrong | Gave up | Mean tool calls |
|---|---|---|---|---|
| Fixed order | 187 | 0 | 13 | 4.68 |
| Lookahead teacher | 187 | 0 | 13 | 3.40 |
| Trained MLP | 187 | 0 | 13 | 3.42 |

The MLP decides without lookahead and still matches the teacher's route
cost. It skips routes that would fail, for example a direct evaluation
modulo 210 through an inverse Euler transform. RL kept that level but did
not improve on it, because the teacher is already near optimal on this
mix. In the 13 tasks no route produced a value, and the planner gave up
rather than answer wrongly. The table predates the stabilising many-prime
step. That step fixed q(3000) mod 30030 for partitions into distinct parts,
whose growth exp(c·√n) the digit estimate undershoots. The checkpoint is
`models/research_planner.nimq`. Training and inference run on Metal or
Vulkan. The CPU backend is refused.

### Far-term benchmark

`research_benchmark.nim` asks `research_sequence` 81 far-term questions over
MCP. Each question gives 40 terms. They fall into these groups:

- six multiplicative functions at indices up to 10^15;
- linear recurrences up to index 10^100 and modulus 10^60+7;
- holonomic sequences such as Catalan, Motzkin, Franel, Apéry and
  derangements, exactly and modulo 2^30, 10^12 or primes;
- partitions through the inverse Euler transform;
- Bell numbers;
- Stern's sequence and popcount;
- small prime moduli at huge indices:
  - Catalan(10^18) mod 999983;
  - Catalan(10^12) mod 10007·10009;
  - C(2n, n) at 10^15;
  - Bell(10^18) mod 13;
  - derangements(10^15) mod 101;
- holonomic terms past 10^6 steps modulo large primes:
  - Motzkin(10^7) and Motzkin(10^8) mod 10^9+7;
  - Apéry(5·10^7) mod 998244353;
- the binomial transform of the partition numbers at 10^5;
- indices past the prime: Catalan and C(2n, n) at 10^17 mod 10^9+7,
  Apéry(10^18) mod 1000003, Franel(10^15) mod 999983;
- online self-referential targets: Bell(10^6), zigzag and rooted trees at
  3·10^4 mod 10^9+7;
- a wide prime, 10^12+39: p(10^5), Motzkin(2·10^7), Bell(2·10^5);
- composite moduli: Bell(10^18) mod 64, derangements(10^15) mod 1000;
- constant terms past the prime: Motzkin(10^18) mod 1000003 and mod
  10000019, Schröder(10^18) mod 1000003, Riordan(10^15) mod 10000019,
  Riordan(2n) at 10^18 mod 1000003, Motzkin(5·10^6) mod 1000003, and
  Fine(5·10^6) mod 1000003 and Fine(10^7) mod 65537;
- prime powers: Catalan(10^18) mod 10^9 and mod 2^24, Motzkin(10^6) mod
  1024, Motzkin(3·10^5) mod 3^7, Catalan(10^7) mod 3^39, C(2n, n) at
  2·10^7 mod 2^61, Catalan(5·10^6) mod 1009^6, past 2^62
  Catalan(1.2·10^7) mod 2^64 and Catalan(1.05·10^7) mod 10^30, and the
  inhomogeneous forms Fine(1.05·10^7) mod 2^10 and the Motzkin partial
  sums at 1.05·10^7 mod 3^7;
- unit leading coefficients past the prime: derangements(10^18) mod
  10^9+7 and mod 2^40, involutions(10^18) mod 1000003, and Π(k² + 1) at
  10^18 mod 10^9+7;
- binomial sums modulo prime powers: Apéry(1.05·10^7) mod 3^10,
  Franel(1.05·10^7) mod 2^10 and Fine(1.05·10^7) mod 3^9;
- a modulus that needs Pollard rho on big integers: Catalan(10^18) modulo
  the product of four primes near 10^6;
- a modulus that needs the elliptic curve method: Catalan(2·10^7) mod
  (10^16+61)(10^17+3).

The truth comes from independent code in the same file: closed forms,
Lucas' theorem, Touchard's congruence as a polynomial power modulo
x^p − x − 1, and cycle detection on the state of D(n) = n·D(n−1) + (−1)^n.
Far binomials use factorials at chosen points and Lucas' theorem, Apéry and
Franel use Gessel's Lucas property, Bell numbers Dobinski's formula modulo a
prime, zigzag numbers Seidel's triangle, rooted trees their divisor-sum
recurrence, and composite moduli a brute-force period search. Constant
terms walk the carries over the base-p digits of the index, with each
coefficient of (a/x + b + c x)^m from the multinomial sum. Their identities
are checked against exact terms from each sequence's own recurrence, also
modulo 7, 11 and 13, where the walk crosses several digits. Fine's
numbers are summed term by term from 2F(n) + F(n − 1) = C(n). Prime power
binomials use Legendre's valuation and a table of unit factorials, and
the moduli past 2^61 a direct product of (n + i)/i with the factors p
counted apart (uint64 wraparound for 2^64, big integers for 5^30).
Derangements past p start from D(m0) = (−1)^m0 at the last multiple of p,
and modulo 2^40 only the last 160 terms of Σ (−1)^m n!/m! count.
Involutions take one 2×2 product over a period of p, then a power, and
Π(k² + 1) over a period of residues is 4 for p ≡ 3 mod 4. Apéry, Franel
and Fine modulo p^k are summed term by term with the factors p counted
apart, and the four-prime modulus joins Lucas' theorem for each prime.
Fine's numbers modulo 2^10 come from the alternating ballot sum
F(n) = Σ_j (−1)^j (C(2n−j, n) − C(2n−j, n+1)), and the Motzkin partial
sums from Σ_k C(n+1, 2k+1)·Catalan(k), each binomial updated from the last
with its factors p counted apart. The ECM modulus joins a direct Catalan
product modulo each prime.
Recurrences are iterated term by term. Long holonomic ranges keep one
common denominator and invert once, and their polynomial coefficients
advance by forward differences. The truth code shares only the BigInt type
with the tool. The benchmark is a compiled program, built with AVX2 and FMA
on x86_64. The 81 truths take about 63 s, mostly the long checking
ranges. Each question has a timeout (default 120 s). A question that hits
it counts as no answer, and the server restarts.

```sh
make -C test/math_spectronnexus bench-research
make -C test/math_spectronnexus bench-research ARGS="--planner-checkpoint=models/research_planner.next.nimq"
```

With the current checkpoint on an Intel Mac (Metal):

| Questions | Right | Wrong | No answer | Tool calls | Time |
|---|---|---|---|---|---|
| 81 | 81 | 0 | 0 | 230 | 67.1 s |

A question can be marked as a gap, meaning the tools cannot reach it yet.
None is marked now. Catalan(10^12) mod 10007, p(10^5) and Bell(10^5) mod
101 were gaps until the hypergeometric, NTT Euler and recurrence routes
above. These changes made the multiplicative questions work and keep failures fast:

- **Digit estimate.** A sequence whose log|a(n)| / log n stops rising now
  gets a polynomial digit estimate. The n log n fit had asked for ~10^11
  primes at index 10^12.
- **Divisor sums.** A Möbius-inverted polynomial, such as τ (b = 1) or σ
  (b = n), is evaluated at any int64 index as the sum over the divisors of
  the index.
- **Named routes.** When a residue route rests on an extra conjecture
  (the Lucas property, a fitted constant-term form), `solution` names it
  instead of saying the relation was iterated.
- **Failure breaks.** The many-primes route stops after 16 primes in a row
  that give no residue, so a sequence with no route ends in `no_answer`
  quickly instead of hanging.

Without `models/research_planner.nimq` the tool follows a fixed order:
direct, factors, exact, many primes. It checks a many-prime answer
against one extra prime before answering. When serving, a trained MLP that
gives up while a target route is still untried falls back to that fixed
order. The checkpoint never saw multiplicative far terms: for exact
φ(10^12) it gave up straight after exploring.

```sh
NEXUS="$PWD/test/math_spectronnexus/nimcache/math_spectronnexus"
printf '%s\n' '{"terms":[1,1,2,3,5,7,11,15,22,30,42,56,77,101,135,176,231,297,385,490],"question":{"index":1000,"modulus":210}}' |
  "$NEXUS" research
printf '%s\n' '{"kind":"identify_constant","value":"0.8224670334241132182362075833230125946094749506033992188"}' |
  "$NEXUS" solve --exact
```

Through MCP the tool name replaces `kind`.

The server carries its own skill. `SKILL.md` in this folder is compiled into
the binary. Its body is sent as the server `instructions` in the
`initialize` reply, so a client that adds the server also gets the workflow:
- compute small cases;
- identify them in the OEIS;
- explore;
- research the far term;
- report conjectures as conjectures.

It also covers how to read each `status` and a table of the tools. The same
text is the MCP prompt `math_spectronnexus_guide`. `SKILL.md` keeps its front
matter, so it also works as a Claude Code skill file. Edit it and rebuild to
change what clients receive. Keep the body under 4,096 characters, since
Claude Code cuts server instructions there. It is 4,020 now.

A successful answer carries `solution`, the method as plain sentences built
from the steps actually run. It names the relation found, with its formula,
how many equations fitted it and how many unseen ones confirmed it. For a
short-data fit it states the integer check and a caution, and for a
multiplicative function the prime power model. Then it
says how the far term was computed and checked, and gives the answer. When
the OEIS data is present, `oeis` names up to three entries that contain the
supplied terms.

If no route works, the OEIS can supply more terms. The planner takes every
entry whose stored terms start with the supplied ones. It keeps only the
terms all of these entries agree on. If those reach the index, it answers
from them with status `oeis_stored_value`. Otherwise it plans again with the
longer list. An answer found that way has status
`extrapolation_from_oeis_terms`. With 9 Franel terms the explorer finds no
relation. The 18 terms shared by the entries starting with them give the
order-2 relation, and a(3000) mod 10^9+7. If the entries disagree at the
index, nothing is chosen. `oeis_candidates` lists each entry's value
instead. For example, the first ten Bell numbers begin many entries,
including set partitions into at most 9 or 10 blocks, and these entries
differ at n = 20.

## Retrieval: OEIS and Mathlib

The solver computes and checks; it does not store mathematical knowledge.
Two read-only tools fetch it instead.

- **`oeis_lookup`** searches a local copy of the OEIS term lists and names.
  Fetch the data once (about 41 MB download, 120 MB unpacked, CC BY-SA 4.0,
  kept out of git):

  ```sh
  make -C test/math_spectronnexus oeis-data
  ```

  The files are memory-mapped and scanned with Boyer–Moore–Horspool. A
  command-line lookup takes under half a second, including startup. It tries the terms as given, then absolute values
  and division by their common factor. A query longer than the stored line
  matches a 12-term prefix and must agree with every stored term. Dropping
  the first one or two terms is the last fallback. Each match gives the
  A-number, name, position, stored continuation and links. `anumber` returns
  one entry. `SPECTRON_OEIS_DIR` points to the data elsewhere.
- **`search_theorems`** searches Lean Mathlib over the network and stores
  nothing. `pattern` goes to Loogle, which matches formula structure such as
  `Nat.choose (2 * _) _` or `?a * ?b = ?b * ?a`. `query` goes to LeanSearch,
  which ranks statements by meaning. `with_source` reads each declaration
  from the Mathlib source on GitHub. Every hit is a statement that Lean has
  checked. For "Bell numbers satisfy a recurrence with binomial
  coefficients" the top hit is `Nat.bell_succ`, which states
  B(n+1) = Σ C(n,i)·B(n−i).

```sh
NEXUS="$PWD/test/math_spectronnexus/nimcache/math_spectronnexus"
printf '%s\n' '{"kind":"oeis_lookup","terms":[1,1,2,5,15,52,203,877,4140,21147]}' | "$NEXUS" solve --exact
printf '%s\n' '{"kind":"search_theorems","query":"sum of binomial coefficients is a power of two","pattern":"Nat.choose _ _","max_results":3}' |
  "$NEXUS" solve --exact
```

## Local MCP

Build the binary and register it as a local stdio server. The twenty-one
native tools are available before training. Each accepts `include_certificate=false`
to keep the answer and verification summary while omitting the full witness.
The binary stays resident for the session.

```sh
make -C test/math_spectronnexus build
NEXUS="$PWD/test/math_spectronnexus/nimcache/math_spectronnexus"

codex mcp add math-spectronnexus -- "$NEXUS" mcp
claude mcp add --transport stdio --scope local math-spectronnexus -- "$NEXUS" mcp
```

### Claude Code

Run the setup from the repository root. `--scope local` registers the server
for your Claude Code sessions in this checkout. Check it with
`claude mcp get math-spectronnexus` and open `/mcp` in Claude Code to see the
connection and tools. Start a new session after rebuilding the server so it
launches the new binary.

The [project skill](../../.claude/skills/math-spectronnexus/SKILL.md) is available
as `/math-spectronnexus`. It guides Claude to `explore_sequence` for finding
structure or testing a formula and `research_sequence` for answering a specific
far-term question. The MCP server also exposes the detailed [tool guide](SKILL.md)
in its instructions and as the `math_spectronnexus_guide` prompt.

For a smoke test, ask Claude to use `research_sequence` to compute the sum of
the Motzkin numbers from index 0 through 10,500,000 modulo 2187. Claude should
generate the first 40 partial sums and send them as `terms` with `offset: 0`
and `question: {"index": 10500000, "modulus": 2187}`. The expected answer is
175, independently checked by `research_benchmark.nim`. The status
`extrapolation_from_conjectured_direction` means the tool computed the value
using a rule inferred from the supplied terms.

### Other MCP clients

For a client that reads an `mcpServers` object, use the same absolute binary
path in its settings. On Windows use the `.exe` filename.

```json
{
  "mcpServers": {
    "math-spectronnexus": {
      "command": "/absolute/path/to/test/math_spectronnexus/nimcache/math_spectronnexus",
      "args": ["mcp"]
    }
  }
}
```

To use a trained sequence program, append
`--program-checkpoint=/absolute/path/sequence_program_curriculum.nimq` after
`mcp`. It guides `explore_sequence`. Use
`--planner-checkpoint=/absolute/path/research_planner.nimq` to select the
planner for `research_sequence`. On macOS, pass `--backend=metal` for GPU
inference. The separate `--checkpoint=/absolute/path/math_policy.nimq`
option selects the policy for the general solver, which checks its
certificate before accepting a result.

Consult the client setup instructions for configuration scope and reload
behavior. [OpenAI Docs for Codex MCP](https://learn.chatgpt.com/docs/extend/mcp?surface=cli),
[Claude Code MCP docs](https://code.claude.com/docs/en/mcp#option-3-add-a-local-stdio-server),
[MCP stdio specification](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports#stdio).
