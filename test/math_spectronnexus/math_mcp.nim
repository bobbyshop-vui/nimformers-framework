## Local MCP tool server. The native solver stays resident for stdio requests.
import std/[json, strutils]
import discrete_math, linear_system, polynomial_math, poset_count, prime_field
import factorization_math, recurrence_math

const
  SupportedProtocols = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
  StructuredProtocol = "2025-06-18"
  MaxMessageBytes = 64 * 1_048_576
  MaxTerms = 5_000          ## MaxExplorerTerms in sequence_explorer.nim.
  MaxCongruences = 100_000
  SkillSource = staticRead("SKILL.md")
  GuidePrompt = "math_spectronnexus_guide"

type SolveHandler* = proc(problem: JsonNode): JsonNode {.closure.}

proc skillBody(): string =
  ## SKILL.md without its front matter. Every client that adds the server receives it as
  ## the server instructions, so the skill comes with the tools.
  result = SkillSource
  if result.startsWith("---"):
    let close = result.find("\n---", 3)
    if close >= 0: result = result[close + 4 .. ^1]
  result = result.strip()

proc skillDescription(): string =
  ## The front matter description, joined across its indented continuation lines.
  var inside = false
  for line in SkillSource.splitLines():
    if line.startsWith("description:"):
      inside = true
      result = line["description:".len .. ^1].strip()
    elif inside and line.startsWith(" "):
      result.add(" " & line.strip())
    elif inside:
      break
  if result.len == 0: result = "How to use the Math SpectronNexus tools."

proc integerSchema(minimum, maximum: int): JsonNode =
  %*{"type": "integer", "minimum": minimum, "maximum": maximum}

proc arraySchema(items: JsonNode, minimum, maximum: int): JsonNode =
  %*{"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}

proc tool(name, description: string, properties: JsonNode, required: openArray[string],
          openWorld = false): JsonNode =
  properties["include_certificate"] = %*{"type": "boolean", "default": true,
    "description": "Return the full witness. Set false to keep the answer and verification summary."}
  %*{"name": name, "description": description,
     "inputSchema": {"type": "object", "properties": properties,
                     "required": @required, "additionalProperties": false},
     "annotations": {"readOnlyHint": true, "idempotentHint": true, "openWorldHint": openWorld}}

proc catalog(): JsonNode =
  let integer = %*{"type": "integer"}
  let smallMatrix = arraySchema(arraySchema(integer, 1, MaxOrder), 1, MaxOrder)
  let systemMatrix = arraySchema(arraySchema(integer, 1, SystemMaxOrder), 1, SystemMaxOrder)
  let coefficients = arraySchema(integer, 0, MaxPolynomialDegree + 1)
  let index = %*{"anyOf": [{"type": "integer", "minimum": 0},
                           {"type": "string", "pattern": "^[0-9]+$", "maxLength": MaxIndexDigits}]}
  let recurrenceCoefficients = arraySchema(integer, 1, MaxRecurrenceOrder)
  let pair = arraySchema(integerSchema(0, MaxPosetOrder - 1), 2, 2)
  let triple = arraySchema(integer, 3, 3)
  result = newJArray()
  result.add(tool("finite_field_rank",
    "Compute the rank of a rectangular integer matrix modulo a small prime and return a row-equivalence certificate.",
    %*{"prime": integerSchema(2, MaxPrime), "matrix": smallMatrix}, ["prime", "matrix"]))
  result.add(tool("max_flow",
    "Find the maximum integral flow in a directed graph and certify it with a minimum cut. Edges are [from, to, capacity]. Vertices start at 0.",
    %*{"nodes": integerSchema(2, MaxOrder), "source": integerSchema(0, MaxOrder - 1),
       "sink": integerSchema(0, MaxOrder - 1), "edges": arraySchema(triple, 0, MaxOrder * MaxOrder * MaxCapacity)},
    ["nodes", "source", "sink", "edges"]))
  result.add(tool("finite_field_linear_system",
    "Solve A*x = b over a prime field. Return rank, a particular solution, a kernel basis, or a left-kernel witness of inconsistency.",
    %*{"prime": integerSchema(2, ExactMaxPrime), "matrix": systemMatrix,
       "rhs": arraySchema(integer, 1, SystemMaxOrder)},
    ["prime", "matrix", "rhs"]))
  result.add(tool("calculate",
    "Compute exact scalar or matrix expressions. Use +, -, *, /, ^, %, parentheses, and calls such as det(A), inv(A), transpose(A), rank(A), rref(A), trace(A), sqrt(4/9), gcd(12,18), or solve(A,B). Loops: sum(k,1,n,expr), prod(k,1,n,expr), count(k,1,n,condition). Also root(x,k) for an exact rational k-th root or a root in the prime field, binomial(n,k), powmod(a,e,m), isprime(n), mobius(n), totient(n), sigma(k,n), div(a,b) for floor division, eq/ne/lt/le/gt/ge, and/or/not, if(c,a,b), min, max. Matrices can be named in variables. A prime switches to modular arithmetic.",
    %*{"expression": {"description": "Infix expression string or an AST with op and args.",
                     "anyOf": [{"type": "string"}, {"type": "object"}, {"type": "integer"}, {"type": "array"}]},
       "variables": {"type": "object", "description": "Named exact scalars or matrices as arrays of rows."},
       "prime": integerSchema(2, ExactMaxPrime)}, ["expression"]))
  let bigEntry = %*{"anyOf": [{"type": "integer"}, {"type": "string", "pattern": "^[+-]?[0-9]+$"}]}
  let bigMatrix = arraySchema(arraySchema(bigEntry, 1, SystemMaxOrder), 1, SystemMaxOrder)
  result.add(tool("matrix_multiply",
    "Multiply two matrices with exact integers of any size (decimal strings for large entries), or modulo a prime.",
    %*{"left": bigMatrix, "right": bigMatrix, "prime": integerSchema(2, ExactMaxPrime)},
    ["left", "right"]))
  result.add(tool("matrix_determinant",
    "Compute the determinant of a square matrix over a prime field, with elimination steps.",
    %*{"matrix": systemMatrix, "prime": integerSchema(2, ExactMaxPrime)}, ["matrix", "prime"]))
  result.add(tool("matrix_inverse",
    "Invert a square matrix over a prime field. A singular matrix returns a nonzero kernel witness and no inverse.",
    %*{"matrix": systemMatrix, "prime": integerSchema(2, ExactMaxPrime)}, ["matrix", "prime"]))
  result.add(tool("finite_field_polynomial_gcd",
    "Find the monic gcd of f and g over a prime field with a Bézout certificate. Coefficients are in ascending degree order. [] denotes zero.",
    %*{"prime": integerSchema(2, ExactMaxPrime), "f": coefficients, "g": coefficients},
    ["prime", "f", "g"]))
  result.add(tool("count_linear_extensions",
    "Count total orders compatible with a poset. Relations are [before, after] pairs with zero-based vertices. The exact count is a decimal string.",
    %*{"nodes": integerSchema(0, MaxPosetOrder), "relations": arraySchema(pair, 0, MaxPosetOrder * MaxPosetOrder)},
    ["nodes", "relations"]))
  result.add(tool("factor_polynomial",
    "Factor a nonzero polynomial over a prime field. Return monic irreducible factors and multiplicities. List coefficients in ascending degree order.",
    %*{"prime": integerSchema(2, ExactMaxPrime),
       "polynomial": arraySchema(integer, 1, MaxFactorDegree + 1)}, ["prime", "polynomial"]))
  result.add(tool("recurrence_term",
    "Evaluate a large-index term over a prime field. Supply coefficients newest-lag first with initial terms, or samples with a promised order_bound. Supply at least 2 * order_bound samples.",
    %*{"prime": integerSchema(2, ExactMaxPrime), "index": index,
       "coefficients": recurrenceCoefficients, "initial": recurrenceCoefficients,
       "samples": arraySchema(integer, 2, MaxRecurrenceSamples),
       "order_bound": integerSchema(1, MaxRecurrenceOrder)}, ["prime", "index"]))
  result.add(tool("weighted_walk_count",
    "Count weighted directed walks of a given length modulo a prime. matrix[i][j] is the edge weight from i to j. Source and sink are zero-based.",
    %*{"prime": integerSchema(2, ExactMaxPrime), "length": index,
       "matrix": arraySchema(arraySchema(integer, 1, MaxWalkOrder), 1, MaxWalkOrder),
       "source": integerSchema(0, MaxWalkOrder - 1), "sink": integerSchema(0, MaxWalkOrder - 1)},
    ["prime", "length", "matrix", "source", "sink"]))
  let bigInteger = %*{"anyOf": [{"type": "integer"}, {"type": "string", "pattern": "^[+-]?[0-9]+$"}]}
  let expression = %*{"type": "string", "description": "Calculator syntax, for example sum(k,0,n,binomial(n,k)^2)."}
  result.add(tool("explore_sequence",
    "Find and check structure in an integer sequence you computed, for example brute-force counts for small n. Refutes your formula claims with an exact counterexample. Searches polynomials, linear and holonomic (P-recursive) recurrences, algebraic generating functions, multiplicativity and periodic residues, also after transforms such as differences, partial sums, ratios, factorial scaling, binomial, Mobius and inverse Euler transforms. Every reported direction is fitted on early terms and confirmed on unseen terms modulo three primes. It is a conjecture, not a proof. A target evaluates a far term exactly or modulo a number from the first direction on the original sequence.",
    %*{"terms": arraySchema(bigInteger, 8, MaxTerms), "offset": %*{"type": "integer", "description": "Index of the first term, default 0."},
       "generator": expression, "start": %*{"type": "integer"}, "count": integerSchema(8, MaxTerms),
       "claims": arraySchema(expression, 0, 16),
       "target": %*{"type": "object", "properties": {"index": index, "modulus": bigInteger, "exact": {"type": "boolean"}}},
       "moduli": arraySchema(integerSchema(2, 1_000_000_000), 1, 16),
       "max_tests": integerSchema(1, 5000), "max_directions": integerSchema(1, 8)}, []))
  result.add(tool("search_counterexample",
    "Test a conjecture on every point of an integer grid with exact arithmetic. Give predicate (nonzero means true) or left and right expressions, and variables mapping names to [first, last]. Returns counterexamples with their assignment. Passing a finite check is evidence, not a proof.",
    %*{"predicate": expression, "left": expression, "right": expression,
       "variables": {"type": "object", "additionalProperties": arraySchema(integer, 2, 2)},
       "constants": {"type": "object"}, "prime": integerSchema(2, ExactMaxPrime),
       "max_cases": integerSchema(1, 2_000_000)}, ["variables"]))
  result.add(tool("chinese_remainder",
    "Combine congruences x = r_i mod m_i for pairwise coprime moduli into one exact integer of any size. Residues and moduli may be decimal strings of any length.",
    %*{"residues": arraySchema(bigInteger, 1, MaxCongruences), "moduli": arraySchema(bigInteger, 1, MaxCongruences),
       "symmetric": {"type": "boolean", "description": "Return the representative closest to zero."}},
    ["residues", "moduli"]))
  result.add(tool("nth_root",
    "Take the k-th root of an exact integer or fraction p/q of any size. Without a prime, return the real root truncated to the requested digits, whether it is exact, and a bracket y^k <= X < (y+1)^k certifying the digits. With a prime, list every y with y^k = value mod prime and their count.",
    %*{"value": {"anyOf": [{"type": "integer"}, {"type": "string", "pattern": "^[+-]?[0-9]+(/[+-]?[0-9]+)?$"}]},
       "degree": integerSchema(1, 100_000),
       "digits": integerSchema(0, 100_000), "prime": integerSchema(2, 9_000_000_000),
       "max_roots": integerSchema(1, 1_000)},
    ["value", "degree"]))
  result.add(tool("identify_constant",
    "Recognize a real number from its decimal digits. Integer relation search (exact integral LLL) finds x as a rational combination of constants such as pi, pi^2, log2, zeta3, catalan, gamma, e, sqrtN, or finds an integer polynomial with root x (algebraic_degree). With no basis it tries a cascade of common bases. Supply at least 20 and preferably 40 or more correct digits. A relation reports how many digits it holds to and how much it beats chance. It is evidence, not a proof.",
    %*{"value": {"type": "string", "description": "Decimal digits, for example \"1.2020569031595942853997381615\"."},
       "basis": arraySchema(%*{"type": "string"}, 1, 14),
       "algebraic_degree": integerSchema(1, 13), "digits": integerSchema(12, 420),
       "min_excess": {"type": "number", "description": "Required margin in digits between precision and coefficient size, default 10."}},
    ["value"]))
  result.add(tool("constant_digits",
    "Evaluate an expression in named constants to many digits: pi, e, gamma, catalan, phi, logN, zetaN, sqrtN, with + - * / ^integer, sqrt(x), root(x,k), log(x), exp(x), zeta(n). Use it to test a closed form against computed digits.",
    %*{"expression": {"type": "string"}, "digits": integerSchema(1, 3_000)}, ["expression"]))
  result.add(tool("research_sequence",
    "Answer a question about an integer sequence with a planned chain of exact tools: explore structure, request more terms from a generator, then evaluate the far term directly modulo M, modulo the prime factors of M with CRT, by exact iteration, or with many primes and CRT, and check the result against extra primes. A learned planner picks the steps. Returns the answer, every step taken, and the direction it relies on.",
    %*{"terms": arraySchema(bigInteger, 8, MaxTerms), "offset": %*{"type": "integer"},
       "generator": expression, "start": %*{"type": "integer"}, "count": integerSchema(8, MaxTerms),
       "question": {"type": "object", "properties": {"index": index, "modulus": bigInteger},
         "required": ["index"]}},
    ["question"]))
  result.add(tool("oeis_lookup",
    "Identify an integer sequence in a local offline copy of the OEIS. Give at least 3 and preferably 6 or more terms; entries containing them are returned with the A-number, name, the stored continuation (next_terms), and links. It also tries absolute values, division by the common factor, and dropping the first terms. Or give anumber (for example \"A000110\") for an entry's name and stored terms. A match is evidence about the sequence, not a proof.",
    %*{"terms": arraySchema(bigInteger, 3, MaxTerms),
       "anumber": {"type": "string", "pattern": "^[Aa]?[0-9]{1,6}$"},
       "max_results": integerSchema(1, 20)}, []))
  result.add(tool("search_theorems",
    "Search Lean Mathlib for proved theorems and lemmas, read over the network. pattern uses Loogle syntax to match formula structure, for example \"Nat.choose _ _ = _\", \"Real.pi\", \"?a * ?b = ?b * ?a\" or \"|- tsum _ = _\". query is plain language ranked by meaning (LeanSearch), for example \"sum of binomial coefficients is a power of two\". Returns the formal statement, module, documentation link, and an informal description or the proof when available. with_source also reads each declaration from the Mathlib source on GitHub.",
    %*{"query": {"type": "string", "maxLength": 500}, "pattern": {"type": "string", "maxLength": 500},
       "max_results": integerSchema(1, 20), "with_source": {"type": "boolean"}}, [],
    openWorld = true))

proc reply(id, payload: JsonNode) =
  stdout.writeLine($( %*{"jsonrpc": "2.0", "id": id, "result": payload} ))
  stdout.flushFile()

proc rpcError(id: JsonNode, code: int, message: string) =
  stdout.writeLine($( %*{"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}} ))
  stdout.flushFile()

proc toolError(id: JsonNode, message: string) =
  reply(id, %*{"content": [{"type": "text", "text": message}], "isError": true})

proc callTool(id, params, tools: JsonNode, solver: SolveHandler, protocol: string) =
  let toolName = params{"name"}
  if toolName.isNil or toolName.kind != JString:
    rpcError(id, -32602, "Expected a tool name")
    return
  let name = toolName.getStr()
  var declared: JsonNode
  for item in tools:
    if item["name"].getStr() == name:
      declared = item
      break
  if declared.isNil:
    rpcError(id, -32602, "Unknown math tool")
    return
  let args = params{"arguments"}
  if args.isNil or args.kind != JObject:
    toolError(id, "Expected a tool arguments object")
    return
  let schema = declared["inputSchema"]
  for key in schema["required"]:
    if not args.hasKey(key.getStr()):
      toolError(id, "Missing argument " & key.getStr())
      return
  let wantProof = args{"include_certificate"}
  if not wantProof.isNil and wantProof.kind != JBool:
    toolError(id, "include_certificate must be boolean")
    return
  let problem = %*{"kind": name}
  for key, value in args:
    if not schema["properties"].hasKey(key):
      toolError(id, "Unknown argument " & key)
      return
    if key != "include_certificate": problem[key] = value
  try:
    let solved = solver(problem)
    if not wantProof.getBool(true) and solved.hasKey("certificate"):
      solved.delete("certificate")
    let payload = %*{"content": [{"type": "text", "text": $solved}], "isError": false}
    if protocol >= StructuredProtocol: payload["structuredContent"] = solved
    reply(id, payload)
  except CatchableError as error:
    toolError(id, error.msg)
  except Defect as defect:
    # An internal bug, such as a 64 bit overflow, fails this call and keeps the server alive.
    toolError(id, "Internal error (" & $defect.name & "): " & defect.msg & ". Please report the input.")

proc serveMcp*(solver: SolveHandler) =
  let tools = catalog()
  var protocol = ""
  var ready = false
  var line: string
  while stdin.readLine(line):
    if line.len > MaxMessageBytes:
      rpcError(newJNull(), -32600, "MCP message exceeds one MiB")
      continue
    if line.strip().len == 0: continue
    var request: JsonNode
    try:
      request = parseJson(line)
    except JsonParsingError:
      rpcError(newJNull(), -32700, "Invalid JSON")
      continue
    let rpc = request{"jsonrpc"}
    let rpcMethod = request{"method"}
    let id = request{"id"}
    let validId = not id.isNil and id.kind in {JString, JInt}
    if request.kind != JObject or rpc.getStr() != "2.0" or
        rpcMethod.isNil or rpcMethod.kind != JString or (not id.isNil and not validId):
      rpcError(if validId: id else: newJNull(), -32600, "Invalid JSON-RPC request")
      continue

    let name = rpcMethod.getStr()
    if id.isNil:
      if name == "notifications/initialized" and protocol.len > 0: ready = true
      continue
    case name
    of "ping":
      reply(id, newJObject())
    of "initialize":
      let version = request{"params", "protocolVersion"}
      if protocol.len > 0 or version.isNil or version.kind != JString:
        rpcError(id, -32602, "Expected a single initialize request with protocolVersion")
        continue
      let requested = version.getStr()
      protocol = if requested in SupportedProtocols: requested else: SupportedProtocols[^1]
      reply(id, %*{"protocolVersion": protocol, "capabilities": {"tools": {}, "prompts": {}},
        "serverInfo": {"name": "math-spectronnexus", "version": "0.3.0"},
        "instructions": skillBody()})
    else:
      if not ready:
        rpcError(id, -32002, "MCP session is not initialized")
      elif name == "tools/list":
        reply(id, %*{"tools": tools})
      elif name == "tools/call":
        callTool(id, request{"params"}, tools, solver, protocol)
      elif name == "prompts/list":
        reply(id, %*{"prompts": [{"name": GuidePrompt, "title": "Math SpectronNexus skill",
                                  "description": skillDescription()}]})
      elif name == "prompts/get":
        if request{"params", "name"}.getStr() != GuidePrompt:
          rpcError(id, -32602, "Unknown prompt")
        else:
          reply(id, %*{"description": skillDescription(),
            "messages": [{"role": "user", "content": {"type": "text", "text": skillBody()}}]})
      else:
        rpcError(id, -32601, "Method not found")
