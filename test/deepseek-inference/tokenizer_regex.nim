## Small pure Nim regular-expression matcher for tokenizer pre-tokenization.
## Unicode classes, alternation, greedy/lazy repeats, groups and lookahead are
## evaluated without PCRE/Oniguruma. Captures and backreferences are not used.
import std/[strutils, unicode]
import tokenizer_unicode

type
  RxKind = enum rxClass, rxSequence, rxAlternative, rxRepeat, rxLook, rxStart, rxEnd
  TokenizerRegex* = ref object
    kind: RxKind
    children: seq[TokenizerRegex]
    ranges: seq[(int, int)]
    properties: seq[(string, bool)]
    inverted, insensitive, positive, lazy: bool
    minimum, maximum: int
  RxParser = object
    chars: seq[Rune]
    pos: int
    insensitive: bool
  RegexMatch* = tuple[first, last: int] # byte offsets, exclusive last

proc bad(message: string) {.noreturn.} =
  raise newException(ValueError, "[native tokenizer regex] " & message)
proc peek(p: RxParser): int =
  if p.pos < p.chars.len: int(p.chars[p.pos]) else: -1
proc take(p: var RxParser): int =
  result = p.peek()
  if result < 0: bad("unexpected end of pattern")
  inc p.pos
proc literal(cp: int, ci: bool): TokenizerRegex =
  TokenizerRegex(kind: rxClass, ranges: @[(cp, cp)], insensitive: ci)

proc escaped(p: var RxParser): TokenizerRegex =
  let ch = p.take()
  result = TokenizerRegex(kind: rxClass, insensitive: p.insensitive)
  case ch
  of ord('s'), ord('S'): result.properties = @[("s", ch == ord('S'))]
  of ord('w'), ord('W'): result.properties = @[("w", ch == ord('W'))]
  of ord('d'), ord('D'): result.properties = @[("Nd", ch == ord('D'))]
  of ord('p'), ord('P'):
    if p.take() != ord('{'): bad("expected Unicode property braces")
    var name = ""
    while p.peek() != ord('}'):
      name.add($Rune(p.take()))
    inc p.pos
    # Validate eagerly, before any model weights are loaded.
    discard unicodeCategory(65, name)
    result.properties = @[(name, ch == ord('P'))]
  of ord('n'): result.ranges = @[(10, 10)]
  of ord('r'): result.ranges = @[(13, 13)]
  of ord('t'): result.ranges = @[(9, 9)]
  of ord('f'): result.ranges = @[(12, 12)]
  of ord('v'): result.ranges = @[(11, 11)]
  of ord('A'): result.kind = rxStart
  of ord('z'), ord('Z'): result.kind = rxEnd
  else:
    if ch in ord('0')..ord('9'): bad("backreferences are unsupported")
    result.ranges = @[(ch, ch)]

proc alternative(p: var RxParser): TokenizerRegex
proc atom(p: var RxParser): TokenizerRegex =
  let ch = p.take()
  case ch
  of ord('('):
    var look = false
    var positive = true
    let oldCi = p.insensitive
    if p.peek() == ord('?'):
      inc p.pos
      case p.take()
      of ord(':'): discard
      of ord('='): look = true
      of ord('!'): look = true; positive = false
      of ord('i'):
        p.insensitive = true
        if p.take() != ord(':'): bad("expected (?i:...) group")
      else: bad("unsupported group modifier")
    result = p.alternative()
    if p.take() != ord(')'): bad("unclosed group")
    p.insensitive = oldCi
    if look: result = TokenizerRegex(kind: rxLook, children: @[result], positive: positive)
  of ord('['):
    result = TokenizerRegex(kind: rxClass, insensitive: p.insensitive)
    if p.peek() == ord('^'): result.inverted = true; inc p.pos
    while p.peek() != ord(']'):
      if p.peek() < 0: bad("unclosed character class")
      var item: TokenizerRegex
      if p.peek() == ord('\\'): inc p.pos; item = p.escaped()
      else: item = literal(p.take(), p.insensitive)
      if p.peek() == ord('-') and p.pos+1 < p.chars.len and int(p.chars[p.pos+1]) != ord(']'):
        inc p.pos
        var last: TokenizerRegex
        if p.peek() == ord('\\'): inc p.pos; last = p.escaped()
        else: last = literal(p.take(), p.insensitive)
        if item.ranges.len != 1 or last.ranges.len != 1 or item.properties.len > 0 or last.properties.len > 0:
          bad("character-class ranges require literal endpoints")
        result.ranges.add((item.ranges[0][0], last.ranges[0][0]))
      else:
        result.ranges.add(item.ranges)
        result.properties.add(item.properties)
    inc p.pos
  of ord('\\'): result = p.escaped()
  of ord('.'):
    result = TokenizerRegex(kind: rxClass, inverted: true, ranges: @[(10, 10)])
  of ord('^'): result = TokenizerRegex(kind: rxStart)
  of ord('$'): result = TokenizerRegex(kind: rxEnd)
  else: result = literal(ch, p.insensitive)

proc repetition(p: var RxParser): TokenizerRegex =
  result = p.atom()
  var lo = -1
  var hi = -1
  case p.peek()
  of ord('?'): lo = 0; hi = 1; inc p.pos
  of ord('*'): lo = 0; inc p.pos
  of ord('+'): lo = 1; inc p.pos
  of ord('{'):
    inc p.pos
    var a = ""
    while p.peek() in ord('0')..ord('9'): a.add(char(p.take()))
    if a.len == 0: bad("invalid repetition bound")
    lo = parseInt(a)
    hi = lo
    if p.peek() == ord(','):
      inc p.pos
      var b = ""
      while p.peek() in ord('0')..ord('9'): b.add(char(p.take()))
      hi = if b.len == 0: -1 else: parseInt(b)
    if p.take() != ord('}'): bad("unclosed repetition")
    if hi >= 0 and hi < lo: bad("invalid repetition bounds")
  else: discard
  if lo >= 0:
    result = TokenizerRegex(kind: rxRepeat, children: @[result], minimum: lo, maximum: hi)
    if p.peek() == ord('?'): result.lazy = true; inc p.pos

proc sequence(p: var RxParser): TokenizerRegex =
  result = TokenizerRegex(kind: rxSequence)
  while p.peek() >= 0 and p.peek() notin [ord('|'), ord(')')]:
    result.children.add(p.repetition())

proc alternative(p: var RxParser): TokenizerRegex =
  result = TokenizerRegex(kind: rxAlternative, children: @[p.sequence()])
  while p.peek() == ord('|'):
    inc p.pos
    result.children.add(p.sequence())

proc compileTokenizerRegex*(pattern: string): TokenizerRegex =
  var p = RxParser(chars: pattern.toRunes())
  result = p.alternative()
  if p.pos != p.chars.len: bad("unexpected closing parenthesis")

proc matches(node: TokenizerRegex, cp: int): bool =
  var yes = false
  for r in node.ranges:
    if cp in r[0]..r[1]: yes = true; break
    if node.insensitive:
      let lower = int(toLower(Rune(cp)))
      if lower in int(toLower(Rune(r[0])))..int(toLower(Rune(r[1]))): yes = true; break
  if not yes:
    for prop in node.properties:
      if unicodeCategory(cp, prop[0]) != prop[1]: yes = true; break
  yes != node.inverted

proc addUnique(target: var seq[int], value: int) =
  if value notin target: target.add(value)

proc endpoints(node: TokenizerRegex, text: seq[RuneSpan], start: int): seq[int] =
  case node.kind
  of rxClass:
    if start < text.len and node.matches(text[start].cp): result.add(start+1)
  of rxStart:
    if start == 0: result.add(start)
  of rxEnd:
    if start == text.len: result.add(start)
  of rxLook:
    let found = node.children[0].endpoints(text, start).len > 0
    if found == node.positive: result.add(start)
  of rxAlternative:
    for child in node.children:
      for finish in child.endpoints(text, start): result.addUnique(finish)
  of rxSequence:
    result = @[start]
    for child in node.children:
      var next: seq[int]
      for at in result:
        for finish in child.endpoints(text, at): next.addUnique(finish)
      result = move(next)
      if result.len == 0: break
  of rxRepeat:
    let body = node.children[0]
    let limit = if node.maximum < 0: text.len-start else: min(node.maximum, text.len-start)
    if body.kind == rxClass:
      var finish = start
      while finish < start+limit and body.matches(text[finish].cp): inc finish
      if finish-start >= node.minimum:
        if node.lazy:
          for at in start+node.minimum .. finish: result.add(at)
        else:
          for at in countdown(finish, start+node.minimum): result.add(at)
    else:
      var layers = @[@[start]]
      for count in 0..<limit:
        var next: seq[int]
        for at in layers[^1]:
          for finish in body.endpoints(text, at):
            if finish > at: next.addUnique(finish)
        if next.len == 0: break
        layers.add(move(next))
      if layers.len > node.minimum:
        if node.lazy:
          for count in node.minimum..<layers.len:
            for finish in layers[count]: result.addUnique(finish)
        else:
          for count in countdown(layers.len-1, node.minimum):
            for finish in layers[count]: result.addUnique(finish)

proc findTokenizerMatches*(pattern: TokenizerRegex, text: string): seq[RegexMatch] =
  let chars = runeSpans(text)
  var at = 0
  while at < chars.len:
    let ends = pattern.endpoints(chars, at)
    var finish = -1
    for candidate in ends:
      if candidate > at: finish = candidate; break
    if finish < 0: inc at
    else:
      result.add((chars[at].start, if finish == chars.len: text.len else: chars[finish].start))
      at = finish
