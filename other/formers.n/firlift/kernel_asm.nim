## Test-time assembler: kernel body text -> session_vm program (resolves @inN/@outN to flat offsets).
import std/[strutils, tables]
import session_vm

type
  Syms = Table[string, float64]

proc parseExpr(text: string; pos: var int; syms: Syms; level: int): float64

proc parseFactor(text: string; pos: var int; syms: Syms): float64 =
  if pos >= text.len: raise newException(ValueError, "expression expected")
  let c = text[pos]
  if c == '(':
    inc pos
    result = parseExpr(text, pos, syms, 0)
    if pos >= text.len or text[pos] != ')': raise newException(ValueError, "missing )")
    inc pos
    return
  if c == '-':
    inc pos
    return -parseFactor(text, pos, syms)
  if c == '+':
    inc pos
    return parseFactor(text, pos, syms)
  if c.isDigit or c == '.':
    let start = pos
    while pos < text.len and (text[pos].isDigit or text[pos] in {'.', 'e', 'E'}):
      inc pos
    return parseFloat(text[start ..< pos])
  if c.isAlphaAscii or c == '_':
    let start = pos
    while pos < text.len and (text[pos].isAlphaNumeric or text[pos] in {'_', '.'}): inc pos
    let name = text[start ..< pos]
    if not syms.hasKey(name): raise newException(ValueError, "unknown symbol " & name)
    return syms[name]
  raise newException(ValueError, "unexpected character " & $c)

proc parseExpr(text: string; pos: var int; syms: Syms; level: int): float64 =
  if level == 0:
    result = parseExpr(text, pos, syms, 1)
    while pos < text.len and text[pos] in {'+', '-'}:
      let op = text[pos]
      inc pos
      let rhs = parseExpr(text, pos, syms, 1)
      if op == '+': result += rhs else: result -= rhs
  elif level == 1:
    result = parseFactor(text, pos, syms)
    while pos < text.len and text[pos] in {'*', '/'}:
      let op = text[pos]
      inc pos
      let rhs = parseFactor(text, pos, syms)
      if op == '*': result *= rhs else: result /= rhs
  else:
    result = parseFactor(text, pos, syms)

proc evalExpr(text: string; syms: Syms): float64 =
  var pos = 0
  result = parseExpr(text, pos, syms, 0)
  if pos != text.len: raise newException(ValueError, "trailing text in " & text)

type
  Layout = object
    inBase, outBase: seq[int]
    inSize, outSize: seq[int]
    syms: Syms

proc portOffset(l: Layout; port: string; extra: string): uint32 =
  var base = 0
  var size = 0
  if port.startsWith("in"):
    let i = parseInt(port[2 .. ^1])
    base = l.inBase[i]; size = l.inSize[i]
  else:
    let i = parseInt(port[3 .. ^1])
    base = l.outBase[i]; size = l.outSize[i]
  let rel = if extra.len == 0: 0.0 else: evalExpr(extra, l.syms)
  if rel < 0: raise newException(ValueError, "negative kernel offset")
  uint32(base + int(rel))

proc resolveAddress(arg: string; l: Layout): uint32 =
  if not arg.startsWith("@"): return uint32(evalExpr(arg, l.syms))
  var port = ""
  var i = 1
  while i < arg.len and (arg[i].isAlphaNumeric or arg[i] == '_'):
    port.add arg[i]
    inc i
  let extra = if i < arg.len: arg[i .. ^1] else: ""
  portOffset(l, port, extra)

proc assemble*(body: string; ins, outs: int; inSize, outSize: seq[int]; eps: float32): VmProgram =
  var syms = initTable[string, float64]()
  for i, n in inSize: syms["in" & $i & ".size"] = float64(n)
  for i, n in outSize: syms["out" & $i & ".size"] = float64(n)
  syms["eps"] = float64(eps)
  var l = Layout(syms: syms, inSize: inSize, outSize: outSize)
  var base = 0
  for n in inSize:
    l.inBase.add base
    base += n
  for n in outSize:
    l.outBase.add base
    base += n
  for raw in body.splitLines():
    let line = raw.strip()
    if line.len == 0 or line.startsWith("#"): continue
    let w = line.splitWhitespace()
    let m = w[0]
    let a = w[1 .. ^1]
    case m
    of "rtou": result.emitRToU(parseInt(a[0]).uint32, parseInt(a[1]).uint32)
    of "utor": result.emitUToR(parseInt(a[0]).uint32, parseInt(a[1]).uint32)
    of "uloadf32": result.emitULoadF32(parseInt(a[0]).uint32, parseInt(a[1]).uint32, resolveAddress(a[2], l))
    of "ustoref32": result.emitUStoreF32(parseInt(a[0]).uint32, parseInt(a[1]).uint32, resolveAddress(a[2], l))
    of "uconst": result.emitUConst(parseInt(a[0]).uint32, resolveAddress(a[1], l))
    of "uadd": result.emitUAdd(parseInt(a[0]).uint32, parseInt(a[1]).uint32, parseInt(a[2]).uint32)
    of "umul": result.emitUMul(parseInt(a[0]).uint32, parseInt(a[1]).uint32, parseInt(a[2]).uint32)
    of "uaddk": result.emitUAddK(parseInt(a[0]).uint32, parseInt(a[1]).uint32, uint32(evalExpr(a[2], syms)))
    of "loadlid": result.emitLoadLid(parseInt(a[0]).uint32)
    of "loadtgid": result.emitLoadTgid(parseInt(a[0]).uint32)
    of "utgid": result.emitUTgid(parseInt(a[0]).uint32)
    of "ulid": result.emitULid(parseInt(a[0]).uint32)
    of "simdlane": result.emitSimdLane(parseInt(a[0]).uint32)
    of "simdid": result.emitSimdId(parseInt(a[0]).uint32)
    of "simdw": result.emitSimdW(parseInt(a[0]).uint32)
    of "simdgid": result.emitSimdGid(parseInt(a[0]).uint32)
    of "simdsum": result.emitSimdSum(parseInt(a[0]).uint32, parseInt(a[1]).uint32)
    of "tgloadf32": result.emitTgLoadF32(parseInt(a[0]).uint32, parseInt(a[1]).uint32, resolveAddress(a[2], l))
    of "tgstoref32": result.emitTgStoreF32(parseInt(a[0]).uint32, parseInt(a[1]).uint32, resolveAddress(a[2], l))
    of "ushr": result.emitUShr(parseInt(a[0]).uint32, parseInt(a[1]).uint32, parseInt(a[2]).uint32)
    of "ushrk": result.emitUShrK(parseInt(a[0]).uint32, parseInt(a[1]).uint32, uint32(evalExpr(a[2], syms)))
    of "ushlk": result.emitUShlK(parseInt(a[0]).uint32, parseInt(a[1]).uint32, uint32(evalExpr(a[2], syms)))
    of "uandk": result.emitUAndK(parseInt(a[0]).uint32, parseInt(a[1]).uint32, uint32(evalExpr(a[2], syms)))
    of "uloadu8": result.emitULoadU8(parseInt(a[0]).uint32, parseInt(a[1]).uint32, uint32(evalExpr(a[2], syms)))
    of "uloadu32": result.emitULoadU32(parseInt(a[0]).uint32, parseInt(a[1]).uint32, uint32(evalExpr(a[2], syms)))
    of "umulk": result.emitUMulK(parseInt(a[0]).uint32, parseInt(a[1]).uint32, uint32(evalExpr(a[2], syms)))
    of "loadconstf32": result.emitLoadConstF32(parseInt(a[0]).uint32, float32(evalExpr(a[1], syms)))
    of "addconst": result.emitAddConst(parseInt(a[0]).uint32, parseInt(a[1]).uint32, float32(evalExpr(a[2], syms)))
    of "mulconst": result.emitMulConst(parseInt(a[0]).uint32, parseInt(a[1]).uint32, float32(evalExpr(a[2], syms)))
    of "fma": result.emitFma(parseInt(a[0]).uint32, parseInt(a[1]).uint32, parseInt(a[2]).uint32, parseInt(a[3]).uint32)
    of "add": result.emitAdd(parseInt(a[0]).uint32, parseInt(a[1]).uint32, parseInt(a[2]).uint32)
    of "sub": result.emitSub(parseInt(a[0]).uint32, parseInt(a[1]).uint32, parseInt(a[2]).uint32)
    of "mul": result.emitMul(parseInt(a[0]).uint32, parseInt(a[1]).uint32, parseInt(a[2]).uint32)
    of "div": result.emitDiv(parseInt(a[0]).uint32, parseInt(a[1]).uint32, parseInt(a[2]).uint32)
    of "exp": result.emitExp(parseInt(a[0]).uint32, parseInt(a[1]).uint32)
    of "neg": result.emitNeg(parseInt(a[0]).uint32, parseInt(a[1]).uint32)
    of "sigmoid": result.emitSigmoid(parseInt(a[0]).uint32, parseInt(a[1]).uint32)
    of "softplus": result.emitSoftplus(parseInt(a[0]).uint32, parseInt(a[1]).uint32)
    of "rsqrt": result.emitRsqrt(parseInt(a[0]).uint32, parseInt(a[1]).uint32)
    of "cmplt": result.emitCmpLt(parseInt(a[0]).uint32, parseInt(a[1]).uint32, parseInt(a[2]).uint32)
    of "ifbegin": result.emitIfBegin(parseInt(a[0]).uint32)
    of "ifend": result.emitIfEnd()
    of "forbegin": result.emitForBegin(parseInt(a[0]).uint32)
    of "forend": result.emitForEnd()
    of "barriertg": result.emitBarrierTg()
    else: raise newException(ValueError, "test assembler does not cover " & m)
  result.emitHalt()

