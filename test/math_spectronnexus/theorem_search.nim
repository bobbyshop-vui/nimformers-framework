## Online theorem search over Lean Mathlib, read over the network and never stored.
## Loogle matches the structure of a formula or type pattern; LeanSearch ranks
## statements by meaning for a plain-language query. Every hit is a statement
## that Lean has checked, so it is safe to cite, but its hypotheses must still
## fit the problem at hand.
import std/[json, osproc, streams, strutils, uri]

const
  LoogleEndpoint = "https://loogle.lean-lang.org/json?q="
  LeanSearchEndpoint = "https://leansearch.net/search"
  DocsBase = "https://leanprover-community.github.io/mathlib4_docs/"
  SourceBase = "https://github.com/leanprover-community/mathlib4/blob/master/"
  RawBase = "https://raw.githubusercontent.com/leanprover-community/mathlib4/master/"
  DefaultResults = 5
  MaxResults = 20
  MaxQueryLength = 500
  MaxProofChars = 1_500
  MaxSourceLines = 40
  RequestSeconds = "30"

proc fetch(args: openArray[string]): string =
  ## One HTTPS request through curl with an argument list (no shell).
  var command = @["-sS", "-f", "-L", "--max-time", RequestSeconds]
  for a in args: command.add(a)
  let process = startProcess("curl", args = command, options = {poUsePath, poStdErrToStdOut})
  result = process.outputStream.readAll()
  let code = process.waitForExit()
  process.close()
  if code != 0:
    raise newException(IOError, "network request failed (curl exit " & $code & "): " & result.strip())

proc modulePath(module: string): string = module.replace('.', '/')

proc links(entry: JsonNode, module, name: string) =
  if module.len == 0: return
  entry["module"] = %module
  entry["docs_url"] = %(DocsBase & modulePath(module) & ".html#" & name)
  if module.startsWith("Mathlib."):
    entry["source_url"] = %(SourceBase & modulePath(module) & ".lean")

proc clip(text: string, limit: int): string =
  if text.len <= limit: text else: text[0 ..< limit] & " …"

proc searchLoogle(pattern: string, limit: int): JsonNode =
  let reply = parseJson(fetch([LoogleEndpoint & encodeUrl(pattern, usePlus = false)]))
  result = newJArray()
  if reply.hasKey("error"):
    var message = reply["error"].getStr()
    if reply.hasKey("suggestions") and reply["suggestions"].len > 0:
      message.add(" Suggestions: " & $reply["suggestions"])
    raise newException(ValueError, "Loogle: " & message)
  for hit in reply{"hits"}.getElems():
    if result.len >= limit: break
    let name = hit{"name"}.getStr()
    var entry = %*{"engine": "loogle", "name": name, "statement": hit{"type"}.getStr().strip()}
    let doc = hit{"doc"}
    if not doc.isNil and doc.kind == JString: entry["doc"] = %doc.getStr().strip()
    entry.links(hit{"module"}.getStr(), name)
    result.add(entry)

proc searchLeanSearch(query: string, limit: int): JsonNode =
  let body = $(%*{"query": [query], "num_results": limit})
  let reply = parseJson(fetch(["-X", "POST", "-H", "Content-Type: application/json", "--data-binary", body,
                               LeanSearchEndpoint]))
  result = newJArray()
  if reply.kind != JArray or reply.len == 0: return
  for item in reply[0].getElems():
    if result.len >= limit: break
    let hit = item{"result"}
    if hit.isNil: continue
    var parts: seq[string]
    for p in hit{"name"}.getElems(): parts.add(p.getStr())
    var moduleParts: seq[string]
    for p in hit{"module_name"}.getElems(): moduleParts.add(p.getStr())
    let name = parts.join(".")
    var entry = %*{"engine": "leansearch", "name": name, "kind": hit{"kind"}.getStr(),
                   "statement": hit{"type"}.getStr().strip(),
                   "distance": item{"distance"}.getFloat()}
    for key in ["informal_name", "informal_description", "docstring"]:
      let value = hit{key}
      if not value.isNil and value.kind == JString and value.getStr().len > 0: entry[key] = value
    let proof = hit{"value"}
    if not proof.isNil and proof.kind == JString: entry["proof"] = %clip(proof.getStr().strip(), MaxProofChars)
    entry.links(moduleParts.join("."), name)
    result.add(entry)

proc declarationSource(module, name: string): string =
  ## The declaration's lines from the Mathlib file, read over the network.
  if not module.startsWith("Mathlib."): return ""
  let text = fetch([RawBase & modulePath(module) & ".lean"])
  let short = name.split('.')[^1]
  let lines = text.splitLines()
  for i, line in lines:
    let s = line.strip()
    for keyword in ["theorem ", "lemma ", "def ", "abbrev ", "instance "]:
      let at = s.find(keyword & short)
      if at < 0: continue
      let after = at + keyword.len + short.len
      if after < s.len and s[after] notin {' ', ':', '(', '{', '[', '\t'}: continue
      var chunk: seq[string]
      for j in i ..< min(lines.len, i + MaxSourceLines):
        if j > i and lines[j].strip().len == 0: break
        chunk.add(lines[j])
      return chunk.join("\n")
  ""

proc solveSearchTheorems*(problem: JsonNode): JsonNode =
  let query = problem{"query"}.getStr().strip()
  let pattern = problem{"pattern"}.getStr().strip()
  if query.len == 0 and pattern.len == 0:
    raise newException(ValueError, "Supply query (plain language) or pattern (a Lean formula pattern), or both")
  if query.len > MaxQueryLength or pattern.len > MaxQueryLength:
    raise newException(ValueError, "query and pattern are limited to " & $MaxQueryLength & " characters")
  let limit = int(problem{"max_results"}.getBiggestInt(DefaultResults))
  if limit notin 1 .. MaxResults: raise newException(ValueError, "max_results must be 1.." & $MaxResults)
  let withSource = problem{"with_source"}.getBool(false)
  var results = newJArray()
  var errors = newJArray()
  if pattern.len > 0:
    try:
      for r in searchLoogle(pattern, limit): results.add(r)
    except CatchableError as error:
      errors.add(%*{"engine": "loogle", "error": error.msg})
  if query.len > 0:
    try:
      for r in searchLeanSearch(query, limit): results.add(r)
    except CatchableError as error:
      errors.add(%*{"engine": "leansearch", "error": error.msg})
  if withSource:
    for entry in results:
      if entry.hasKey("proof"): continue
      try:
        let source = declarationSource(entry{"module"}.getStr(), entry["name"].getStr())
        if source.len > 0: entry["source"] = %source
      except CatchableError as error:
        entry["source_error"] = %error.msg
  result = %*{"results": results, "count": results.len,
              "status": (if results.len > 0: "found" elif errors.len > 0: "network_error" else: "no_match")}
  if errors.len > 0: result["errors"] = errors
  result["note"] = %("Each statement is proved in Lean's Mathlib, so it can be cited, but check that its " &
    "hypotheses and types (for example natural versus real numbers) fit your problem. Results are read " &
    "over the network from Loogle, LeanSearch and GitHub; nothing is stored locally.")
