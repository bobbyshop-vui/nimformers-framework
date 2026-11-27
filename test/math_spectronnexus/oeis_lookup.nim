## Offline OEIS lookup. `make oeis-data` fetches the stripped term lists and the
## names (CC BY-SA 4.0) into data/oeis; this module memory-maps them and finds
## every entry whose stored terms contain the query, also after simple
## normalisations. A match is evidence about which sequence the terms come
## from. It is not a proof.
import std/[algorithm, json, memfiles, os, strutils]
import big_integer

const
  MaxRawHits = 5_000
  DefaultResults = 5
  MaxResults = 20
  NextTermsShown = 12
  MinQueryTerms = 3
  MaxQueryTerms = 5_000
  PrefixWindow = 12        ## Stored lines hold about 260 characters, so long queries search a prefix.
  MaxAlignedEntries = 64

type
  OeisData = object
    loaded: bool
    dir: string
    stripped, names: MemFile
  Variant = tuple[label: string, terms: seq[BigInt], rank: int, full: seq[BigInt]]
  AlignedEntry* = tuple[anumber: int, name: string, terms: seq[BigInt]]
  Hit = object
    anumber: int
    position: int          ## Index of the first matching term among the stored terms.
    variant: Variant
    lineStart: int

var oeis: OeisData

proc oeisDirectory*(): string =
  ## SPECTRON_OEIS_DIR, else data/oeis beside the build folder, else under the working directory.
  let fromEnv = getEnv("SPECTRON_OEIS_DIR")
  if fromEnv.len > 0: return fromEnv
  let besideBuild = getAppDir().parentDir / "data" / "oeis"
  if fileExists(besideBuild / "stripped"): return besideBuild
  getCurrentDir() / "data" / "oeis"

proc ensureLoaded() =
  if oeis.loaded: return
  let dir = oeisDirectory()
  if not fileExists(dir / "stripped") or not fileExists(dir / "names"):
    raise newException(ValueError, "The offline OEIS data is missing in " & dir &
      ". Run `make -C test/math_spectronnexus oeis-data` once, or set SPECTRON_OEIS_DIR.")
  oeis.stripped = memfiles.open(dir / "stripped")
  oeis.names = memfiles.open(dir / "names")
  oeis.dir = dir
  oeis.loaded = true

template bytes(m: MemFile): ptr UncheckedArray[char] = cast[ptr UncheckedArray[char]](m.mem)

proc lineStartBefore(m: MemFile, pos: int): int =
  result = pos
  while result > 0 and m.bytes[result - 1] != '\n': dec result

proc nextLineStart(m: MemFile, pos: int): int =
  result = pos
  while result < m.size and m.bytes[result] != '\n': inc result
  if result < m.size: inc result

proc lineText(m: MemFile, start: int): string =
  var stop = start
  while stop < m.size and m.bytes[stop] != '\n': inc stop
  result = newString(stop - start)
  if result.len > 0: copyMem(addr result[0], addr m.bytes[start], result.len)

proc anumberAt(m: MemFile, start: int): int =
  ## The number of the entry whose line starts at start, or -1 for a comment line.
  if start + 7 > m.size or m.bytes[start] != 'A': return -1
  result = 0
  for i in 1 .. 6:
    let c = m.bytes[start + i]
    if c notin Digits: return -1
    result = result * 10 + (ord(c) - ord('0'))

proc findLine(m: MemFile, anumber: int): int =
  ## Binary search over the sorted lines of a names or stripped file. Returns -1 if absent.
  var lo = 0
  var hi = m.size
  while lo < hi:
    let start = m.lineStartBefore((lo + hi) div 2)
    let n = m.anumberAt(start)
    if n == anumber: return start
    if n < anumber: lo = m.nextLineStart(start)
    else: hi = start
  -1

proc formatAnumber(n: int): string = "A" & align($n, 6, '0')

proc parseAnumber(text: string): int =
  var t = text.strip()
  if t.len > 0 and t[0] in {'A', 'a'}: t = t[1 .. ^1]
  if t.len == 0 or t.len > 6 or not t.allCharsInSet(Digits):
    raise newException(ValueError, "anumber must look like A000110")
  parseInt(t)

proc entryName(anumber: int): string =
  let start = oeis.names.findLine(anumber)
  if start < 0: return ""
  let line = oeis.names.lineText(start)
  if line.len > 8: line[8 .. ^1] else: ""

proc storedTerms(lineStart: int): seq[string] =
  ## The stored terms on a stripped line "A000110 ,1,1,2,5,...,".
  let line = oeis.stripped.lineText(lineStart)
  let comma = line.find(',')
  if comma < 0: return
  for part in line[comma + 1 .. ^1].split(','):
    if part.len > 0: result.add(part)

proc searchAll(pattern: string, limit: int): seq[int] =
  ## Boyer-Moore-Horspool over the mapped stripped file.
  let m = oeis.stripped
  let n = m.size
  let k = pattern.len
  if k == 0 or k > n: return
  var shift: array[256, int]
  for i in 0 .. 255: shift[i] = k
  for i in 0 ..< k - 1: shift[ord(pattern[i])] = k - 1 - i
  var pos = 0
  let text = m.bytes
  while pos <= n - k:
    var j = k - 1
    while j >= 0 and text[pos + j] == pattern[j]: dec j
    if j < 0:
      result.add(pos)
      if result.len >= limit: return
      pos += 1
    else:
      pos += shift[ord(text[pos + k - 1])]

proc patternOf(terms: seq[BigInt]): string =
  result = ","
  for t in terms: result.add($t & ",")

proc variantsOf(terms: seq[BigInt]): seq[Variant] =
  result.add(("exact", terms, 0, @[]))
  var anyNegative = false
  for t in terms:
    if t.isNegative: anyNegative = true
  if anyNegative:
    var absolute: seq[BigInt]
    for t in terms: absolute.add(if t.isNegative: -t else: t)
    result.add(("absolute_values", absolute, 1, @[]))
  var g = big(0)
  for t in terms:
    if not t.isZero: g = gcdBig(g, if t.isNegative: -t else: t)
  if not g.isZero and big(1) < g:
    var scaled: seq[BigInt]
    for t in terms: scaled.add(divMod(t, g).quotient)
    result.add(("divided_by_" & $g, scaled, 2, @[]))
  if terms.len > PrefixWindow:
    # The entry may store fewer terms than the query: match a prefix, then require
    # every overlapping stored term to agree with the query.
    result.add(("prefix_then_all_stored_terms_agree", terms[0 ..< PrefixWindow], 3, terms))
  for drop in 1 .. 2:
    if terms.len - drop >= 6:
      result.add(("without_first_" & $drop & "_terms", terms[drop .. ^1], 3 + drop, @[]))

proc consistent(stored: seq[string], position: int, full: seq[BigInt]): bool =
  for i, t in full:
    let j = position + i
    if j >= stored.len: break
    if stored[j] != $t: return false
  true

proc termPosition(lineStart, matchPos: int): int =
  let m = oeis.stripped
  var i = lineStart
  while i < matchPos and m.bytes[i] != ',': inc i
  for p in i ..< matchPos:
    if m.bytes[p] == ',': inc result

proc dataModified(): string =
  ## The "Last Modified" header line of the stripped file.
  var start = 0
  for _ in 0 ..< 4:
    let line = oeis.stripped.lineText(start)
    if line.startsWith("# Last Modified:"): return line[16 .. ^1].strip()
    start = oeis.stripped.nextLineStart(start)
  ""

proc entryJson(anumber, lineStart: int): JsonNode =
  let tag = formatAnumber(anumber)
  %*{"anumber": tag, "name": entryName(anumber), "url": "https://oeis.org/" & tag,
     "bfile": "https://oeis.org/" & tag & "/b" & tag[1 .. ^1] & ".txt",
     "stored_terms": storedTerms(lineStart).len}

proc lookupTerms*(terms: seq[BigInt], maxResults = DefaultResults): JsonNode =
  ## Entries containing the terms, best first: plainer normalisation, earlier match, older entry.
  ensureLoaded()
  var hits: seq[Hit]
  var seen: seq[int]
  var capped = false
  for variant in variantsOf(terms):
    if variant.rank >= 3 and hits.len > 0: break   # Dropping terms is a fallback only.
    let positions = searchAll(patternOf(variant.terms), MaxRawHits)
    if positions.len >= MaxRawHits: capped = true
    for pos in positions:
      let start = oeis.stripped.lineStartBefore(pos)
      let anumber = oeis.stripped.anumberAt(start)
      if anumber < 0 or anumber in seen: continue
      let position = termPosition(start, pos)
      if variant.full.len > 0 and not consistent(storedTerms(start), position, variant.full): continue
      seen.add(anumber)
      hits.add(Hit(anumber: anumber, position: position, variant: variant, lineStart: start))
  hits.sort(proc(a, b: Hit): int =
    result = cmp(a.variant.rank, b.variant.rank)
    if result == 0: result = cmp(ord(a.position > 3), ord(b.position > 3))
    if result == 0: result = cmp(a.anumber, b.anumber))
  var matches = newJArray()
  for hit in hits[0 ..< min(hits.len, maxResults)]:
    var entry = entryJson(hit.anumber, hit.lineStart)
    let stored = storedTerms(hit.lineStart)
    let after = hit.position + hit.variant.terms.len
    entry["match"] = %*{"normalisation": hit.variant.label, "first_matching_stored_term": hit.position}
    entry["next_terms"] = %stored[min(after, stored.len) ..< min(after + NextTermsShown, stored.len)]
    matches.add(entry)
  result = %*{"query_terms": terms.len, "matches": matches, "entries_found": hits.len,
              "entries_found_is_lower_bound": capped,
              "data": {"source": "OEIS stripped and names files, local copy", "modified": dataModified(),
                       "license": "CC BY-SA 4.0, https://oeis.org/LICENSE"},
              "status": (if hits.len == 0: "no_match" else: "lookup_evidence")}
  var digits = 0
  for t in terms: digits += ($t).len
  if hits.len > 0 and digits < 12:
    result["warning"] = %"The query carries few digits, so matches may be coincidental. Supply more terms."
  result["note"] = %("A match means these numbers occur in the entry. It is evidence, not a proof: check the " &
    "entry's definition and offset against your problem, then test a formula with explore_sequence or " &
    "research_sequence. next_terms are the entry's stored continuation after the matched window.")

proc lookupEntry*(anumber: int): JsonNode =
  ensureLoaded()
  let start = oeis.stripped.findLine(anumber)
  if start < 0: raise newException(ValueError, formatAnumber(anumber) & " is not in the local OEIS data")
  result = entryJson(anumber, start)
  result["terms"] = %storedTerms(start)
  result["status"] = %"entry"

proc alignedEntries*(terms: seq[BigInt]): tuple[entries: seq[AlignedEntry], capped: bool] =
  ## Every entry whose stored terms start with the query (as far as they are stored).
  ensureLoaded()
  let window = if terms.len > PrefixWindow: terms[0 ..< PrefixWindow] else: terms
  for pos in searchAll(patternOf(window), MaxRawHits):
    let start = oeis.stripped.lineStartBefore(pos)
    let anumber = oeis.stripped.anumberAt(start)
    if anumber < 0 or termPosition(start, pos) != 0: continue
    let stored = storedTerms(start)
    if not consistent(stored, 0, terms): continue
    if result.entries.len >= MaxAlignedEntries:
      result.capped = true
      return
    var values: seq[BigInt]
    for t in stored: values.add(parseBig(t))
    result.entries.add((anumber, entryName(anumber), values))

proc formatEntry*(anumber: int): string = formatAnumber(anumber)

proc oeisAvailable*(): bool =
  try:
    ensureLoaded()
    true
  except ValueError, OSError, IOError:
    false

proc solveOeisLookup*(problem: JsonNode): JsonNode =
  let count = int(problem{"max_results"}.getBiggestInt(DefaultResults))
  if count notin 1 .. MaxResults: raise newException(ValueError, "max_results must be 1.." & $MaxResults)
  if problem.hasKey("anumber"):
    return lookupEntry(parseAnumber(problem["anumber"].getStr()))
  let node = problem{"terms"}
  if node.isNil or node.kind != JArray: raise newException(ValueError, "Supply terms (an array) or anumber")
  if node.len notin MinQueryTerms .. MaxQueryTerms:
    raise newException(ValueError, "Supply between " & $MinQueryTerms & " and " & $MaxQueryTerms & " terms")
  var terms: seq[BigInt]
  for t in node: terms.add(toBig(t, "term"))
  lookupTerms(terms, count)
