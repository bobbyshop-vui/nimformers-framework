## Native, sandboxed chat-template renderer. Supports text, expressions, set,
## if/elif/else and for with Jinja's loop scoping and whitespace control.
## Unsupported syntax is rejected instead of replacing the model's template.
import std/[json, strutils, tables, unicode]

type
  TemplateEnvironment* = Table[string, JsonNode]
  Tag = object
    kind, text: string
  NodeKind = enum nText, nOutput, nSet, nIf, nFor
  ChatTemplate* = ref object
    kind: NodeKind
    text, variable: string
    children: seq[ChatTemplate]
    conditions: seq[string]
    branches: seq[seq[ChatTemplate]]
  ExprToken = object
    kind, value: string
  ExprParser = object
    tokens: seq[ExprToken]
    pos: int
    env: TemplateEnvironment

proc bad(message: string) {.noreturn.} =
  raise newException(ValueError, "[native chat template] " & message)
proc stringValue(v: JsonNode): string =
  if v.isNil: return ""
  case v.kind
  of JString: v.getStr()
  of JBool: (if v.getBool(): "True" else: "False")
  of JNull: "None"
  else: $v
proc truth(v: JsonNode): bool =
  if v.isNil: return false
  case v.kind
  of JNull: false
  of JBool: v.getBool()
  of JString: v.getStr().len > 0
  of JArray, JObject: v.len > 0
  of JInt: v.getInt() != 0
  of JFloat: v.getFloat() != 0

proc lex(expression: string): seq[ExprToken] =
  var at = 0
  while at < expression.len:
    let ch = expression[at]
    if ch in {' ', '\t', '\r', '\n'}: inc at; continue
    if ch in {'\'', '"'}:
      let quote = ch
      inc at
      var value = ""
      var closed = false
      while at < expression.len:
        let c = expression[at]; inc at
        if c == quote: closed = true; break
        if c != '\\': value.add(c); continue
        if at >= expression.len: bad("unterminated string escape")
        let escaped = expression[at]; inc at
        case escaped
        of 'n': value.add('\n')
        of 'r': value.add('\r')
        of 't': value.add('\t')
        of 'b': value.add('\b')
        of 'f': value.add('\f')
        of '\\', '\'', '"': value.add(escaped)
        of 'u', 'x':
          let count = if escaped == 'u': 4 else: 2
          if at+count > expression.len: bad("truncated character escape")
          let cp = parseHexInt(expression[at..<at+count]); at += count
          if escaped == 'x': value.add(char(cp))
          else: value.add($Rune(cp))
        else: value.add('\\'); value.add(escaped)
      if not closed: bad("unterminated string")
      result.add(ExprToken(kind: "string", value: value))
    elif ch in {'a'..'z', 'A'..'Z', '_'}:
      let first = at
      while at < expression.len and expression[at] in {'a'..'z','A'..'Z','0'..'9','_'}: inc at
      result.add(ExprToken(kind: "name", value: expression[first..<at]))
    elif ch in {'0'..'9'}:
      let first = at
      while at < expression.len and expression[at] in {'0'..'9'}: inc at
      result.add(ExprToken(kind: "number", value: expression[first..<at]))
    else:
      var op = $ch
      inc at
      if at < expression.len and op & expression[at] in ["==","!=","<=",">="]:
        op.add(expression[at]); inc at
      if op notin ["==","!=","<=",">=","<",">","+","-","~","(",")","[","]",".",",","|",":","%"]:
        bad("unsupported expression operator: " & op)
      result.add(ExprToken(kind: "op", value: op))

proc peek(p: ExprParser): string =
  if p.pos < p.tokens.len: p.tokens[p.pos].value else: ""
proc take(p: var ExprParser): ExprToken =
  if p.pos >= p.tokens.len: bad("unexpected end of expression")
  result = p.tokens[p.pos]; inc p.pos
proc expect(p: var ExprParser, value: string) =
  if p.take().value != value: bad("expected " & value)
proc expr(p: var ExprParser): JsonNode
proc arguments(p: var ExprParser): seq[JsonNode] =
  p.expect("(")
  while p.peek() != ")":
    result.add(p.expr())
    if p.peek() != ",": break
    inc p.pos
  p.expect(")")
proc access(value, key: JsonNode): JsonNode =
  result = newJNull()
  if value.isNil: return
  if value.kind == JObject and key.kind == JString:
    result = value.getOrDefault(key.getStr())
    if result.isNil: result = newJNull()
  elif value.kind == JArray and key.kind == JInt:
    var i = key.getInt()
    if i < 0: i += value.len
    if i >= 0 and i < value.len: result = value[i]

proc primary(p: var ExprParser): JsonNode =
  let token = p.take()
  if token.kind == "string": result = %token.value
  elif token.kind == "number": result = %parseInt(token.value)
  elif token.value == "(": result = p.expr(); p.expect(")")
  elif token.kind == "name":
    case token.value
    of "true", "True": result = %true
    of "false", "False": result = %false
    of "none", "None": result = newJNull()
    else:
      if p.peek() == "(":
        let args = p.arguments()
        if token.value == "raise_exception":
          bad(if args.len > 0: stringValue(args[0]) else: "template rejected messages")
        else: bad("unsupported template function: " & token.value)
      result = p.env.getOrDefault(token.value, newJNull())
  else: bad("unsupported expression term: " & token.value)
  while true:
    if p.peek() == "[":
      inc p.pos
      let key = p.expr()
      p.expect("]")
      result = access(result, key)
    elif p.peek() == ".":
      inc p.pos
      let name = p.take().value
      if p.peek() == "(":
        let args = p.arguments()
        case name
        of "strip": result = %strutils.strip(stringValue(result))
        of "lower": result = %unicode.toLower(stringValue(result))
        of "upper": result = %unicode.toUpper(stringValue(result))
        of "get":
          if args.len < 1: bad("get requires a key")
          let found = access(result, args[0])
          result = if found.kind == JNull and args.len > 1: args[1] else: found
        else: bad("unsupported template method: " & name)
      else: result = access(result, %name)
    elif p.peek() == "|":
      inc p.pos
      let name = p.take().value
      let args = if p.peek() == "(": p.arguments() else: newSeq[JsonNode]()
      case name
      of "trim": result = %strutils.strip(stringValue(result))
      of "lower": result = %unicode.toLower(stringValue(result))
      of "upper": result = %unicode.toUpper(stringValue(result))
      of "length":
        result = %(if result.kind == JString: result.getStr().runeLen else: result.len)
      of "string": result = %stringValue(result)
      of "tojson": result = %($result)
      of "default":
        if result.kind == JNull or (args.len > 1 and truth(args[1]) and not truth(result)):
          result = if args.len > 0: args[0] else: %""
      of "replace":
        if args.len != 2: bad("replace requires two arguments")
        result = %stringValue(result).replace(stringValue(args[0]), stringValue(args[1]))
      else: bad("unsupported template filter: " & name)
    else: break

proc unary(p: var ExprParser): JsonNode =
  if p.peek() == "not": inc p.pos; result = %(not truth(p.unary()))
  elif p.peek() == "-": inc p.pos; result = %(-p.unary().getInt())
  else: result = p.primary()
proc sum(p: var ExprParser): JsonNode =
  result = p.unary()
  while p.peek() in ["+", "~", "-", "%"]:
    let op = p.take().value
    let right = p.unary()
    if op == "%": result = %(result.getInt() mod right.getInt())
    elif op == "-": result = %(result.getInt()-right.getInt())
    elif op == "+" and result.kind == JInt and right.kind == JInt:
      result = %(result.getInt()+right.getInt())
    else: result = %(stringValue(result) & stringValue(right))
proc compare(p: var ExprParser): JsonNode =
  result = p.sum()
  while p.peek() in ["==","!=","<",">","<=",">=","in","is"]:
    let op = p.take().value
    if op == "is":
      var negative = false
      if p.peek() == "not": inc p.pos; negative = true
      let test = p.take().value
      var yes: bool
      case test
      of "defined": yes = result.kind != JNull
      of "none", "None": yes = result.kind == JNull
      of "string": yes = result.kind == JString
      else: bad("unsupported template test: " & test)
      result = %(yes != negative)
      continue
    let right = p.sum()
    var yes = false
    case op
    of "==": yes = result == right
    of "!=": yes = result != right
    of "in":
      if right.kind == JString: yes = stringValue(result) in right.getStr()
      elif right.kind == JObject: yes = right.hasKey(stringValue(result))
      elif right.kind == JArray:
        for value in right: yes = yes or value == result
    else:
      let ordering = if result.kind == JInt and right.kind == JInt:
                       cmp(result.getInt(), right.getInt())
                     else: cmp(stringValue(result), stringValue(right))
      case op
      of "<": yes = ordering < 0
      of ">": yes = ordering > 0
      of "<=": yes = ordering <= 0
      of ">=": yes = ordering >= 0
      else: discard
    result = %yes
proc conjunction(p: var ExprParser): JsonNode =
  result = p.compare()
  while p.peek() == "and":
    inc p.pos
    let right = p.compare()
    if truth(result): result = right
proc expr(p: var ExprParser): JsonNode =
  result = p.conjunction()
  while p.peek() == "or":
    inc p.pos
    let right = p.conjunction()
    if not truth(result): result = right
proc evaluate(expression: string, env: TemplateEnvironment): JsonNode =
  var p = ExprParser(tokens: lex(expression), env: env)
  result = p.expr()
  if p.pos != p.tokens.len: bad("unsupported expression: " & expression)

proc tags(source: string): seq[Tag] =
  var at = 0
  var pending = ""
  while at < source.len:
    if at+1 >= source.len or source[at] != '{' or source[at+1] notin {'{','%','#'}:
      pending.add(source[at]); inc at; continue
    let kind = source[at+1]
    at += 2
    var trimLeft = false
    var keepLeft = false
    if at < source.len and source[at] in {'-', '+'}:
      trimLeft = source[at] == '-'; keepLeft = source[at] == '+'; inc at
    if trimLeft:
      pending = strutils.strip(pending, leading = false)
    elif kind != '{' and not keepLeft:
      # lstrip_blocks=True removes indentation before a statement/comment tag.
      var cut = pending.len
      while cut > 0 and pending[cut-1] in {' ', '\t'}: dec cut
      if cut == 0 or pending[cut-1] == '\n': pending.setLen(cut)
    if pending.len > 0: result.add(Tag(kind: "text", text: pending)); pending = ""
    let first = at
    let closing = if kind == '{': "}}" elif kind == '%': "%}" else: "#}"
    # Delimiters inside a string literal are ordinary template content.
    var quote = '\0'
    while at+1 < source.len:
      if quote != '\0':
        if source[at] == '\\': at += 2; continue
        if source[at] == quote: quote = '\0'
      elif source[at] in {'\'', '"'} and kind != '#': quote = source[at]
      elif source[at..<at+2] == closing: break
      inc at
    if at+1 >= source.len: bad("unterminated template tag")
    var finish = at
    var trimRight = false
    var keepRight = false
    if finish > first and source[finish-1] in {'-', '+'}:
      trimRight = source[finish-1] == '-'; keepRight = source[finish-1] == '+'; dec finish
    if kind != '#': result.add(Tag(kind: if kind == '{': "output" else: "statement", text: strutils.strip(source[first..<finish])))
    at += 2
    if trimRight:
      while at < source.len and source[at] in {' ', '\t', '\r', '\n'}: inc at
    elif kind != '{' and not keepRight:
      # trim_blocks=True consumes one newline immediately after a block tag.
      if at < source.len and source[at] == '\r': inc at
      if at < source.len and source[at] == '\n': inc at
  if pending.len > 0: result.add(Tag(kind: "text", text: pending))

proc keyword(text: string): string =
  let stop = text.find(' ')
  if stop < 0: text else: text[0..<stop]
proc body(parts: seq[Tag], at: var int, stops: openArray[string]): seq[ChatTemplate] =
  while at < parts.len:
    let tag = parts[at]
    let command = keyword(tag.text)
    if tag.kind == "statement" and command in stops: return
    inc at
    if tag.kind == "text": result.add(ChatTemplate(kind: nText, text: tag.text))
    elif tag.kind == "output": result.add(ChatTemplate(kind: nOutput, text: tag.text))
    elif command == "set":
      let assignment = tag.text[4..^1]
      let eq = assignment.find('=')
      if eq < 0: bad("set requires an assignment")
      let variable = strutils.strip(assignment[0..<eq])
      if variable.len == 0 or '.' in variable or '[' in variable: bad("unsupported assignment target: " & variable)
      result.add(ChatTemplate(kind: nSet, variable: variable, text: strutils.strip(assignment[eq+1..^1])))
    elif command == "for":
      let clause = tag.text[4..^1]
      let split = clause.find(" in ")
      if split < 0: bad("for requires 'in'")
      let node = ChatTemplate(kind: nFor, variable: strutils.strip(clause[0..<split]), text: strutils.strip(clause[split+4..^1]))
      node.children = body(parts, at, ["endfor"])
      if at >= parts.len: bad("unclosed for")
      inc at
      result.add(node)
    elif command == "if":
      let node = ChatTemplate(kind: nIf)
      var condition = tag.text[3..^1]
      while true:
        node.conditions.add(condition)
        node.branches.add(body(parts, at, ["elif", "else", "endif"]))
        if at >= parts.len: bad("unclosed if")
        let ending = parts[at].text
        inc at
        if keyword(ending) == "endif": break
        if keyword(ending) == "else":
          node.conditions.add("")
          node.branches.add(body(parts, at, ["endif"]))
          if at >= parts.len: bad("unclosed else")
          inc at
          break
        condition = ending[5..^1]
      result.add(node)
    else: bad("unsupported template statement: " & tag.text)

proc compileChatTemplate*(source: string): ChatTemplate =
  # Jinja's default newline_sequence and keep_trailing_newline=False.
  var normalized = source.replace("\r\n", "\n").replace("\r", "\n")
  if normalized.endsWith("\n"): normalized.setLen(normalized.len-1)
  let parts = tags(normalized)
  var at = 0
  result = ChatTemplate(kind: nFor, variable: "", children: body(parts, at, []))

proc clone(env: TemplateEnvironment): TemplateEnvironment =
  result = initTable[string, JsonNode]()
  for key, value in env: result[key] = value
proc render(nodes: seq[ChatTemplate], env: var TemplateEnvironment): string =
  for node in nodes:
    case node.kind
    of nText: result.add(node.text)
    of nOutput: result.add(stringValue(evaluate(node.text, env)))
    of nSet: env[node.variable] = evaluate(node.text, env)
    of nIf:
      for i, condition in node.conditions:
        if condition.len == 0 or truth(evaluate(condition, env)):
          result.add(render(node.branches[i], env))
          break
    of nFor:
      let items = evaluate(node.text, env)
      if items.kind != JArray: bad("for requires an array")
      for i in 0..<items.len:
        let item = items[i]
        var local = clone(env)
        local[node.variable] = item
        local["loop"] = %*{"index0": i, "index": i+1, "first": i == 0, "last": i == items.len-1, "length": items.len}
        result.add(render(node.children, local))
proc renderChatTemplate*(program: ChatTemplate, env: TemplateEnvironment): string =
  var local = clone(env)
  render(program.children, local)
