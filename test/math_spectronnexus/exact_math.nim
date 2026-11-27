## Shared native operations for the command line and the local MCP server.
import std/json
import big_integer, calculator, discrete_math, factorization_math, linear_system, matrix_math,
  polynomial_math, poset_count, recurrence_math, root_math, sequence_explorer, high_precision, constant_math,
  research_planner, oeis_lookup, theorem_search

proc canUsePolicy*(problem: JsonNode): bool =
  case problem{"kind"}.getStr()
  of "finite_field_rank", "max_flow": true
  of "finite_field_polynomial_gcd":
    let f = problem{"f"}
    let g = problem{"g"}
    not f.isNil and f.kind == JArray and f.len <= MaxOrder and
      not g.isNil and g.kind == JArray and g.len <= MaxOrder and
      problem{"prime"}.getBiggestInt() in 2 .. MaxPrime
  of "recurrence_term":
    problem{"order_bound"}.getBiggestInt() in 1 .. MaxOrder and
      problem{"prime"}.getBiggestInt() in 2 .. MaxPrime and not problem.hasKey("coefficients")
  of "weighted_walk_count":
    let matrix = problem{"matrix"}
    not matrix.isNil and matrix.kind == JArray and matrix.len in 1 .. MaxOrder and
      problem{"prime"}.getBiggestInt() in 2 .. MaxPrime
  else: false

proc solveExact*(problem: JsonNode): JsonNode =
  case problem{"kind"}.getStr()
  of "finite_field_rank", "max_flow":
    resultJson(runExpert(parseProblem(problem), recordMethod = true), "exact")
  of "finite_field_polynomial_gcd":
    solvePolynomial(parsePolynomialProblem(problem))
  of "finite_field_linear_system":
    solveSystem(problem)
  of "factor_polynomial":
    solveFactorization(problem)
  of "recurrence_term":
    solveRecurrence(parseRecurrence(problem))
  of "weighted_walk_count":
    solveWalk(problem)
  of "calculate":
    calculateExact(problem)
  of "matrix_multiply":
    solveMatrixProduct(problem)
  of "matrix_determinant":
    solveSquareMatrix(problem, inverse = false)
  of "matrix_inverse":
    solveSquareMatrix(problem, inverse = true)
  of "count_linear_extensions":
    solvePoset(problem)
  of "explore_sequence":
    exploreSequence(problem)
  of "search_counterexample":
    searchCounterexample(problem)
  of "chinese_remainder":
    solveChineseRemainder(problem)
  of "nth_root":
    solveNthRoot(problem)
  of "identify_constant":
    solveIdentifyConstant(problem)
  of "constant_digits":
    solveConstantDigits(problem)
  of "research_sequence":
    researchSequence(problem)
  of "oeis_lookup":
    solveOeisLookup(problem)
  of "search_theorems":
    solveSearchTheorems(problem)
  else:
    raise newException(ValueError, "Unknown mathematical problem kind")
