# bybylang.nim - BybyLang AOT executable + Nim code generation + auto compile release
# Hỗ trợ cơ chế function: define function bằng "function NAME" ... kết thúc bằng một dòng chỉ chứa NAME
import strutils, os, osproc, tables, sequtils, times

# --------------------------
# Traceback system - LUÔN BẬT, IN RA TERMINAL
# --------------------------
type
  TraceLevel* = enum
    tlDebug, tlInfo, tlWarn, tlError

var gTraceEnabled = true
var gTraceLevel = tlInfo

proc currentTime(): string =
  let t = getTime()
  return format(t, "HH:mm:ss")

proc trace*(level: TraceLevel, msg: string, file: string = "", line: int = 0) =
  if not gTraceEnabled: return
  if level < gTraceLevel: return
  
  let prefix = case level
    of tlDebug: "[DEBUG]"
    of tlInfo: "[INFO]"
    of tlWarn: "[WARN]"
    of tlError: "[ERROR]"
  
  let loc = if file.len > 0: file else: "bybylang.nim"
  let ln = if line > 0: line else: instantiationInfo().line
  
  echo prefix & " " & currentTime() & " " & loc & ":" & $ln & " - " & msg

proc traceDebug*(msg: string) = trace(tlDebug, msg)
proc traceInfo*(msg: string) = trace(tlInfo, msg)
proc traceWarn*(msg: string) = trace(tlWarn, msg)
proc traceError*(msg: string) = trace(tlError, msg)

# --------------------------
# Helpers
# --------------------------
proc stripQuotes(s: string): string =
  if s.len >= 2 and s[0] == '"' and s[^1] == '"':
    return s[1..^2]
  else:
    return s
    
proc parseIntSafe(s: string): int =
  try:
    return parseInt(s)
  except:
    return 0

# --------------------------
# Types
# --------------------------
type
  Mode = enum
    Low, Mid, High

  Token = object
    sym: string
    text: string
    indent: int
    
# --------------------------
# RAM / Bus / Pins giả lập
# --------------------------
const RAM_SIZE = 1024
var RAM: array[0..RAM_SIZE-1, int]
var BUS: seq[string] = @[]
var Pins: array[0..31, bool]

var ignoreErrors = false
var quietMode = false

# function table lưu body token
var funcTable = initTable[string, seq[Token]]()
proc tokenizeLine(line: string): Token =
  var tok: Token
  tok.indent = line.len - line.strip(chars={' ', '\t'}).len
  let clean = line.strip()

  if clean.len == 0:
    tok.sym = "empty"
    tok.text = ""
  elif clean.startsWith("#"):
    tok.sym = "comment"
    tok.text = clean
  elif clean.startsWith("function "):
    tok.sym = "function"
    tok.text = clean.replace("function ", "")
  elif clean.startsWith("import "):
    tok.sym = "import"
    tok.text = clean["import ".len..^1].strip()
  elif clean.startsWith("print "):
    tok.sym = "print"
    tok.text = clean
  else:
    tok.sym = "other"
    tok.text = clean

  return tok
# --------------------------
# Hệ thống import: "import tenfile" hoặc "import \"path/ten.bybylang\""
# Nạp đệ quy, chống import vòng lặp bằng `visited` (đường dẫn tuyệt đối).
# Token import được thay bằng toàn bộ token của file được import (đặt trước
# vị trí import), không sinh trực tiếp thành dòng Nim nào.
# --------------------------
proc resolveImports(filename: string, visited: var seq[string]): seq[Token] =
  result = @[]
  let absPath = absolutePath(filename)
  if absPath in visited:
    traceDebug("Skipping already imported: " & filename)
    return
  visited.add(absPath)

  if not fileExists(filename):
    traceError("Import file not found: " & filename)
    echo "[ERROR] import: file not found: ", filename
    return

  traceInfo("Resolving imports for: " & filename)
  for line in lines(filename):
    let t = tokenizeLine(line)
    if t.sym == "import":
      var target = stripQuotes(t.text)
      if not target.endsWith(".bybylang"):
        target &= ".bybylang"
      if not fileExists(target):
        target = filename.parentDir() / target
      result.add(resolveImports(target, visited))
    else:
      result.add(t)
  traceDebug("Resolved " & $result.len & " tokens from " & filename)

# Đọc file .bybylang gốc + toàn bộ file được import thành 1 danh sách tokens
proc tokenizeFile(filename: string): seq[Token] =
  traceInfo("Tokenizing file: " & filename)
  var visited: seq[string] = @[]
  let result = resolveImports(filename, visited)
  traceInfo("Total tokens: " & $result.len)
  return result

# --------------------------
# Hardware-level functions
# --------------------------
proc apuTran(name: string, payload: string) =
  traceDebug("apuTran: " & name & " -> " & payload)
  BUS.add(payload)
  if not quietMode:
    echo "[APU-TRAN] ", name, " -> ", payload

proc apuMem(action: string, target: string, value: string) =
  traceDebug("apuMem: " & action & " " & target & " " & value)
  let ramAddr = parseIntSafe(target.replace("RAM",""))
  if ramAddr < 0 or ramAddr >= RAM_SIZE:
    traceError("Invalid RAM address: " & $ramAddr)
    if not ignoreErrors:
      echo "[ERROR] Invalid RAM address: ", ramAddr
      quit(1)
    return
  if action == "write":
    RAM[ramAddr] = parseIntSafe(value)
    if not quietMode:
      echo "[APU-MEM] RAM[", ramAddr, "] <- ", value
  elif action == "read":
    if not quietMode:
      echo "[APU-MEM] RAM[", ramAddr, "] -> ", RAM[ramAddr]

proc apuCore(mode: int, code: string) =
  traceDebug("apuCore: mode=" & $mode & " code=" & code)
  if not quietMode:
    echo "[APU-CORE] Mode: ", mode, ", running: ", code

proc apuPin(pin: int, state: string) =
  traceDebug("apuPin: pin=" & $pin & " state=" & state)
  if pin < 0 or pin > 31:
    traceError("Invalid pin: " & $pin)
    if not ignoreErrors:
      echo "[ERROR] Invalid pin: ", pin
      quit(1)
    return
  Pins[pin] = (state == "high")
  if not quietMode:
    echo "[APU-PIN] pin ", pin, " set ", state

proc bitSend(bits: string) =
  traceDebug("bitSend: " & bits)
  BUS.add(bits)
  if not quietMode:
    echo "[BIT-SEND] ", bits

proc bitRecv() =
  traceDebug("bitRecv")
  if BUS.len > 0:
    let b = BUS[0]
    delete(BUS, 0)
    if not quietMode:
      echo "[BIT-RECV] ", b
  else:
    if not quietMode:
      echo "[BIT-RECV] empty"

proc memMap(target: string) =
  traceDebug("memMap: " & target)
  if not quietMode:
    echo "[MEM-MAP] ", target

proc memPush(target: string, value: string) =
  traceDebug("memPush: " & target & " <- " & value)
  if not quietMode:
    echo "[MEM-PUSH] ", target, " <- ", value

proc tranPulse(pin: int, width: string) =
  traceDebug("tranPulse: pin=" & $pin & " width=" & width)
  if not quietMode:
    echo "[TRAN-PULSE] pin ", pin, " width ", width

# --------------------------
# Generate Nim code + compile to binary release
# --------------------------
# --------------------------
# Sinh code cho lệnh "gpu ..."
# --------------------------
# Cú pháp hỗ trợ:
#   gpu backend is "auto"          # "auto" | "cuda" | "metal" | "opencl" | "cpu" | "tsic"
#   gpu array X = [1, 2, 3, 4]
#   gpu add A, B -> C size 4       # "size N" tuỳ chọn, chỉ mang tính mô tả
#   gpu sub A, B -> C
#   gpu mul A, B -> C
#   gpu div A, B -> C
#   gpu matmul A, B -> C m M k K n N
#   gpu relu X -> Y
#   gpu sigmoid X -> Y
#   gpu tanh X -> Y
#   gpu apflu X -> Y alpha A beta B
#   gpu softmax X -> Y rows R cols C
#   gpu layernorm X, GAMMA, BETA -> Y rows R cols C eps E
#   gpu layernorm_backward DY, X, GAMMA, BETA -> DX, DGAMMA, DBETA rows R cols C eps E
#   gpu embedding TABLE, INDICES -> Y vocab V dim D
#   gpu attention Q, K, V -> O, S_MATRIX B H S D scale SCALE
#   gpu attention_backward Q, K, V, S_MATRIX, DY -> DQ, DK, DV B H S D scale SCALE
proc genGpuLine(line: string): seq[string] =
  traceDebug("Parsing GPU line: " & line)
  result = @[]
  let rest = line[3..^1].strip()

  if rest.startsWith("backend is"):
    let raw = rest["backend is".len..^1].strip()
    let val = stripQuotes(raw)
    traceInfo("Setting GPU backend: " & val)
    result.add("gpuBackendSelected = parseBackend(\"" & val & "\")")

  elif rest.startsWith("array "):
    let body = rest["array ".len..^1].strip()
    let parts = body.split("=", 1)
    if parts.len == 2:
      let name = parts[0].strip()
      let valsRaw = parts[1].strip()
      traceDebug("Creating GPU array: " & name & " = " & valsRaw)
      result.add("var " & name & ": seq[float32] = @" & valsRaw & ".mapIt(it.float32)")
    else:
      traceWarn("Invalid gpu array syntax: " & line)
      result.add("# [WARN] cú pháp 'gpu array' không hợp lệ: " & line)

  else:
    let arrowParts = rest.split("->")
    if arrowParts.len == 2:
      let lhs = arrowParts[0].strip()
      let rhs = arrowParts[1].strip()
      let lhsWords = lhs.split()
      if lhsWords.len >= 1:
        let opName = lhsWords[0].toLowerAscii()
        let operandsStr = lhs[opName.len..^1].strip()
        let operands = operandsStr.split(",")
        traceDebug("GPU op: " & opName & " with " & $operands.len & " operands")
        
        # Xử lý opName đặc biệt
        if opName == "layernorm_backward":
          if operands.len >= 4:
            let dy = operands[0].strip()
            let x = operands[1].strip()
            let gamma = operands[2].strip()
            let beta = operands[3].strip()
            let rhsParts = rhs.splitWhitespace()
            if rhsParts.len >= 9:
              let targetDx = rhsParts[0].replace(",", "").strip()
              let targetDgamma = rhsParts[1].replace(",", "").strip()
              let targetDbeta = rhsParts[2].replace(",", "").strip()
              var rows = "0"; var cols = "0"; var eps = "0.0"
              for i in 0..<rhsParts.len:
                if rhsParts[i] == "rows" and i+1 < rhsParts.len:
                  rows = rhsParts[i+1]
                elif rhsParts[i] == "cols" and i+1 < rhsParts.len:
                  cols = rhsParts[i+1]
                elif rhsParts[i] == "eps" and i+1 < rhsParts.len:
                  eps = rhsParts[i+1]
              if rows != "0" and cols != "0" and eps != "0.0":
                result.add("let lnBwd = gpuLayernormBackward(gpuBackendSelected, " & dy & ", " & x & ", " & gamma & ", " & beta & ", " & rows & ", " & cols & ", " & eps & ")")
                result.add("var " & targetDx & " = lnBwd.dx")
                result.add("var " & targetDgamma & " = lnBwd.dgamma")
                result.add("var " & targetDbeta & " = lnBwd.dbeta")
              else:
                traceWarn("Invalid layernorm_backward syntax: " & line)
                result.add("# [WARN] cú pháp 'gpu layernorm_backward' cần: gpu layernorm_backward DY, X, GAMMA, BETA -> DX, DGAMMA, DBETA rows R cols C eps E : " & line)
            else:
              traceWarn("Invalid layernorm_backward syntax: " & line)
              result.add("# [WARN] cú pháp 'gpu layernorm_backward' cần: gpu layernorm_backward DY, X, GAMMA, BETA -> DX, DGAMMA, DBETA rows R cols C eps E : " & line)
          else:
            traceWarn("layernorm_backward needs 4 operands: " & line)
            result.add("# [WARN] cú pháp 'gpu layernorm_backward' cần 4 toán hạng: " & line)
            
        elif opName == "attention":
          if operands.len >= 3:
            let q = operands[0].strip()
            let k = operands[1].strip()
            let v = operands[2].strip()
            let rhsParts = rhs.splitWhitespace()
            if rhsParts.len >= 11:
              let targetO = rhsParts[0].replace(",", "").strip()
              let targetS = rhsParts[1].replace(",", "").strip()
              var B = "0"; var H = "0"; var S = "0"; var D = "0"; var scale = "0.0"
              for i in 0..<rhsParts.len:
                if rhsParts[i] == "B" and i+1 < rhsParts.len:
                  B = rhsParts[i+1]
                elif rhsParts[i] == "H" and i+1 < rhsParts.len:
                  H = rhsParts[i+1]
                elif rhsParts[i] == "S" and i+1 < rhsParts.len:
                  S = rhsParts[i+1]
                elif rhsParts[i] == "D" and i+1 < rhsParts.len:
                  D = rhsParts[i+1]
                elif rhsParts[i] == "scale" and i+1 < rhsParts.len:
                  scale = rhsParts[i+1]
              if B != "0" and H != "0" and S != "0" and D != "0" and scale != "0.0":
                result.add("let attnRes = gpuAttentionFused(gpuBackendSelected, " & q & ", " & k & ", " & v & ", @[], " & B & ", " & H & ", " & S & ", " & D & ", " & scale & ")")
                result.add("var " & targetO & " = attnRes.o")
                result.add("var " & targetS & " = attnRes.s_matrix")
              else:
                traceWarn("Invalid attention syntax: " & line)
                result.add("# [WARN] cú pháp 'gpu attention' cần: gpu attention Q, K, V -> O, S_MATRIX B H S D scale SCALE : " & line)
            else:
              traceWarn("Invalid attention syntax: " & line)
              result.add("# [WARN] cú pháp 'gpu attention' cần: gpu attention Q, K, V -> O, S_MATRIX B H S D scale SCALE : " & line)
          else:
            traceWarn("attention needs 3 operands: " & line)
            result.add("# [WARN] cú pháp 'gpu attention' cần 3 toán hạng Q, K, V: " & line)
            
        elif opName == "attention_backward":
          if operands.len >= 5:
            let q = operands[0].strip()
            let k = operands[1].strip()
            let v = operands[2].strip()
            let s_matrix = operands[3].strip()
            let dy = operands[4].strip()
            let rhsParts = rhs.splitWhitespace()
            if rhsParts.len >= 11:
              let targetDq = rhsParts[0].replace(",", "").strip()
              let targetDk = rhsParts[1].replace(",", "").strip()
              let targetDv = rhsParts[2].replace(",", "").strip()
              var B = "0"; var H = "0"; var S = "0"; var D = "0"; var scale = "0.0"
              for i in 0..<rhsParts.len:
                if rhsParts[i] == "B" and i+1 < rhsParts.len:
                  B = rhsParts[i+1]
                elif rhsParts[i] == "H" and i+1 < rhsParts.len:
                  H = rhsParts[i+1]
                elif rhsParts[i] == "S" and i+1 < rhsParts.len:
                  S = rhsParts[i+1]
                elif rhsParts[i] == "D" and i+1 < rhsParts.len:
                  D = rhsParts[i+1]
                elif rhsParts[i] == "scale" and i+1 < rhsParts.len:
                  scale = rhsParts[i+1]
              if B != "0" and H != "0" and S != "0" and D != "0" and scale != "0.0":
                result.add("let attnBwd = gpuAttentionFusedBackward(gpuBackendSelected, " & q & ", " & k & ", " & v & ", " & s_matrix & ", " & dy & ", " & B & ", " & H & ", " & S & ", " & D & ", " & scale & ")")
                result.add("var " & targetDq & " = attnBwd.dq")
                result.add("var " & targetDk & " = attnBwd.dk")
                result.add("var " & targetDv & " = attnBwd.dv")
              else:
                traceWarn("Invalid attention_backward syntax: " & line)
                result.add("# [WARN] cú pháp 'gpu attention_backward' cần: gpu attention_backward Q, K, V, S_MATRIX, DY -> DQ, DK, DV B H S D scale SCALE : " & line)
            else:
              traceWarn("Invalid attention_backward syntax: " & line)
              result.add("# [WARN] cú pháp 'gpu attention_backward' cần: gpu attention_backward Q, K, V, S_MATRIX, DY -> DQ, DK, DV B H S D scale SCALE : " & line)
          else:
            traceWarn("attention_backward needs 5 operands: " & line)
            result.add("# [WARN] cú pháp 'gpu attention_backward' cần 5 toán hạng: " & line)
            
        elif opName == "apflu":
          if operands.len >= 1:
            let x = operands[0].strip()
            let rhsParts = rhs.splitWhitespace()
            if rhsParts.len >= 5 and rhsParts[1] == "alpha" and rhsParts[3] == "beta":
              let target = rhsParts[0]
              let alpha = rhsParts[2]
              let beta = rhsParts[4]
              result.add("var " & target & " = gpuApflu(gpuBackendSelected, " & x & ", " & alpha & ", " & beta & ")")
            else:
              result.add("var " & rhsParts[0] & " = gpuApflu(gpuBackendSelected, " & x & ", 0.1'f32, 1.0'f32)")
          else:
            traceWarn("apflu needs 1 operand: " & line)
            result.add("# [WARN] cú pháp 'gpu apflu' cần 1 toán hạng: " & line)
            
        else:
          if operands.len >= 2:
            let a = operands[0].strip()
            let b = operands[1].strip()

            if opName == "matmul":
              let rhsParts = rhs.splitWhitespace()
              if rhsParts.len >= 7 and rhsParts[1] == "m" and rhsParts[3] == "k" and rhsParts[5] == "n":
                let target = rhsParts[0]
                let m = rhsParts[2]; let k = rhsParts[4]; let n = rhsParts[6]
                result.add("var " & target & " = gpuMatmul(gpuBackendSelected, " & a & ", " & b & ", " & m & ", " & k & ", " & n & ")")
              else:
                traceWarn("Invalid matmul syntax: " & line)
                result.add("# [WARN] cú pháp 'gpu matmul' cần: gpu matmul A, B -> C m M k K n N : " & line)
            elif opName == "layernorm":
              if operands.len >= 3:
                let gamma = operands[1].strip()
                let beta = operands[2].strip()
                let rhsParts = rhs.splitWhitespace()
                if rhsParts.len >= 7 and rhsParts[1] == "rows" and rhsParts[3] == "cols" and rhsParts[5] == "eps":
                  let target = rhsParts[0]
                  let rows = rhsParts[2]; let cols = rhsParts[4]; let eps = rhsParts[6]
                  result.add("var " & target & " = gpuLayernorm(gpuBackendSelected, " & a & ", " & gamma & ", " & beta & ", " & rows & ", " & cols & ", " & eps & ")")
                else:
                  traceWarn("Invalid layernorm syntax: " & line)
                  result.add("# [WARN] cú pháp 'gpu layernorm' cần: gpu layernorm X, GAMMA, BETA -> Y rows R cols C eps E : " & line)
              else:
                traceWarn("layernorm needs 3 operands: " & line)
                result.add("# [WARN] cú pháp 'gpu layernorm' cần 3 toán hạng: " & line)
            elif opName == "embedding":
              let rhsParts = rhs.splitWhitespace()
              if rhsParts.len >= 5 and rhsParts[1] == "vocab" and rhsParts[3] == "dim":
                let target = rhsParts[0]
                let vocab = rhsParts[2]; let dim = rhsParts[4]
                result.add("var " & target & " = gpuEmbeddingLookup(gpuBackendSelected, " & a & ", " & b & ".mapIt(int32(it)), " & vocab & ", " & dim & ")")
              else:
                traceWarn("Invalid embedding syntax: " & line)
                result.add("# [WARN] cú pháp 'gpu embedding' cần: gpu embedding TABLE, INDICES -> Y vocab V dim D : " & line)
            else:
              var target = rhs
              if "size" in rhs:
                target = rhs.split("size")[0].strip()
              result.add("var " & target & " = gpuOp(\"" & opName & "\", gpuBackendSelected, " & a & ", " & b & ")")
          elif operands.len == 1:
            let x = operands[0].strip()
            case opName
            of "relu", "sigmoid", "tanh":
              let target = rhs.splitWhitespace()[0]
              let fn = if opName == "relu": "gpuRelu"
                       elif opName == "sigmoid": "gpuSigmoid"
                       else: "gpuTanh"
              result.add("var " & target & " = " & fn & "(gpuBackendSelected, " & x & ")")
            of "softmax":
              let rhsParts = rhs.splitWhitespace()
              if rhsParts.len >= 5 and rhsParts[1] == "rows" and rhsParts[3] == "cols":
                let target = rhsParts[0]
                let rows = rhsParts[2]; let cols = rhsParts[4]
                result.add("var " & target & " = gpuSoftmax(gpuBackendSelected, " & x & ", " & rows & ", " & cols & ")")
              else:
                traceWarn("Invalid softmax syntax: " & line)
                result.add("# [WARN] cú pháp 'gpu softmax' cần: gpu softmax X -> Y rows R cols C : " & line)
            else:
              traceWarn("Unknown unary op: " & opName)
              result.add("# [WARN] cú pháp 'gpu " & opName & "' cần 2 toán hạng: " & line)
          else:
            traceWarn("No operands for op: " & opName)
            result.add("# [WARN] cú pháp 'gpu " & opName & "' cần toán hạng: " & line)
      else:
        traceWarn("No words in LHS: " & lhs)
        result.add("# [WARN] cú pháp gpu không hợp lệ: " & line)
    else:
      traceWarn("No arrow found: " & rest)
      result.add("# [WARN] cú pháp gpu không hợp lệ (thiếu '->'): " & line)
# --------------------------
# Sinh code đệ quy cho một khối lệnh
# --------------------------
# Hỗ trợ lồng nhau ở ĐỘ SÂU BẤT KỲ: if trong if, while trong if, for trong while,
# gpu-call trong if-trong-for, v.v. Mỗi khi gặp một statement mở khối mới
# (if/elif/else/while/for) thì gọi lại chính nó (đệ quy) để sinh phần thân,
# thay vì phải chép lặp code xử lý riêng cho từng cấp như bản cũ.
proc genBlock(tokens: seq[Token], idx: var int, indent: string,
              funcNames: seq[string], localVars: var seq[string]): seq[string]

# Dùng chung cho if/elif/else/while/for: emit dòng header, rồi hoặc đệ quy vào
# thân khối (nếu dòng kế tiếp thụt sâu hơn) hoặc emit "discard" (khối rỗng).
proc genHeaderedBlock(tokens: seq[Token], idx: var int, indent: string, headerLine: string,
                       blockIndent: int, funcNames: seq[string], localVars: var seq[string]): seq[string] =
  result = @[indent & headerLine]
  idx.inc
  if idx < tokens.len and tokens[idx].indent > blockIndent:
    result.add(genBlock(tokens, idx, indent & "  ", funcNames, localVars))
  else:
    result.add(indent & "  discard")
proc genBlock(tokens: seq[Token], idx: var int, indent: string,
              funcNames: seq[string], localVars: var seq[string]): seq[string] =
  result = @[]
  if idx >= tokens.len: return

  let blockIndent = tokens[idx].indent

  while idx < tokens.len and tokens[idx].indent >= blockIndent:
    let tk = tokens[idx]

    if tk.indent > blockIndent:
      idx.inc
      continue

    case tk.sym
    of "empty":
      idx.inc

    of "comment":
      idx.inc

    of "import":
      idx.inc

    of "print":
      traceDebug("Generating print: " & tk.text)
      result.add(indent & "echo " & tk.text.replace("print", "").strip())
      idx.inc

    of "function":
      let fBase = tk.indent
      idx.inc
      while idx < tokens.len and tokens[idx].indent > fBase:
        idx.inc

    else:
      let line = tk.text.strip()

      if line.len == 0:
        idx.inc

      elif line.startsWith("#"):
        idx.inc

      elif line.startsWith("if ") or line.startsWith("elif ") or line == "else:":
        traceDebug("Generating conditional: " & line)
        result.add(genHeaderedBlock(tokens, idx, indent, line, blockIndent, funcNames, localVars))

      elif line.startsWith("while "):
        traceDebug("Generating while: " & line)
        result.add(genHeaderedBlock(tokens, idx, indent, line, blockIndent, funcNames, localVars))

      elif line.startsWith("for "):
        var forLine = line
        if "range(" in forLine:
          let inside = forLine.split("range(")[1].split(")")[0]
          let parts = inside.split(",")
          if parts.len == 2:
            forLine = forLine.replace("range(" & inside & ")",
                                       parts[0].strip() & ".." & parts[1].strip())
        traceDebug("Generating for: " & forLine)
        result.add(genHeaderedBlock(tokens, idx, indent, forLine, blockIndent, funcNames, localVars))

      elif line.startsWith("gpu "):
        for l in genGpuLine(line):
          result.add(indent & l)
        idx.inc

      elif line.startsWith("call "):
        let parts = line.split()
        if parts.len >= 2:
          let fname = parts[1]
          if fname in funcNames:
            result.add(indent & fname & "()")
          else:
            traceWarn("Function not found: " & fname)
            result.add(indent & "# [WARN] function not found: " & fname)
        idx.inc

      elif line in funcNames:
        result.add(indent & line & "()")
        idx.inc

      elif line.startsWith("mode is"):
        let parts = line.split()
        if parts.len >= 3:
          let m = parseIntSafe(parts[2])
          case m
          of 1: result.add(indent & "echo \"Mode 1: Low-level\"")
          of 2: result.add(indent & "echo \"Mode 2: Mid-level\"")
          of 3: result.add(indent & "echo \"Mode 3: High-level\"")
          of 4: result.add(indent & "echo \"Mode 4: Web-level\"")
          else: result.add(indent & "echo \"Unknown mode: " & $m & "\"")
        idx.inc

      elif line.startsWith("apu tran"):
        let parts = line.split("with")
        let name = stripQuotes(parts[0].split()[2].strip())
        let payload = parts[1].strip()
        result.add(indent & "apuTran(\"" & name & "\", " & payload & ")")
        idx.inc

      elif line.startsWith("apu mem"):
        let parts = line.split("with")
        let left = parts[0].split()
        let action = left[2]
        let target = stripQuotes(left[3])
        let value = parts[1].strip()
        result.add(indent & "apuMem(\"" & action & "\", \"" & target & "\", \"" & value & "\")")
        idx.inc

      elif line.startsWith("apu core"):
        result.add(indent & "apuCore(1, \"run\")")
        idx.inc

      elif line.startsWith("apu pin"):
        let words = line.split()
        result.add(indent & "apuPin(" & words[2] & ", \"" & words[4] & "\")")
        idx.inc

      elif line.startsWith("bit send"):
        result.add(indent & "bitSend(\"" & line.split()[2] & "\")")
        idx.inc

      elif line.startsWith("bit recv"):
        result.add(indent & "bitRecv()")
        idx.inc

      elif line.startsWith("mem map"):
        result.add(indent & "memMap(\"" & stripQuotes(line.split()[2]) & "\")")
        idx.inc

      elif line.startsWith("mem push"):
        let parts = line.split("with")
        result.add(indent & "memPush(\"" & stripQuotes(parts[0].split()[2]) & "\", \"" & parts[1].strip() & "\")")
        idx.inc

      elif line.startsWith("tran pulse"):
        let words = line.split()
        result.add(indent & "tranPulse(" & words[3] & ", \"" & words[^1] & "\")")
        idx.inc

      elif line.contains("="):
        let parts = line.split("=")
        if parts.len >= 2:
          let left = parts[0].strip()
          let right = parts[1..^1].join("=").strip()
          if left notin localVars:
            localVars.add(left)
            result.add(indent & "var " & left & " = " & right)
          else:
            result.add(indent & left & " = " & right)
        idx.inc

      else:
        result.add(indent & line)
        idx.inc
# --------------------------
# Generate Nim code + compile to binary release
# --------------------------
proc generateNimCode(tokens: seq[Token], outFile: string) =
  traceInfo("Generating Nim code for: " & outFile)
  var funcBodiesLocal = initTable[string, seq[Token]]()
  var idx = 0

  # --- tách thân hàm bằng indent ---
  while idx < tokens.len:
    let t = tokens[idx]
    if t.sym == "function":
      let fname = t.text.strip()
      let baseIndent = t.indent
      var body: seq[Token] = @[]
      idx.inc
      while idx < tokens.len and tokens[idx].indent > baseIndent:
        body.add(tokens[idx])
        idx.inc
      funcBodiesLocal[fname] = body
      traceDebug("Found function: " & fname & " with " & $body.len & " body tokens")
    else:
      idx.inc

  # --- khởi tạo file ---
  var nimFile = outFile
  if not nimFile.endsWith(".nim"): nimFile &= ".nim"
  traceInfo("Writing to: " & nimFile)

  var code = newSeq[string]()
  code.add("import strutils, sequtils")
  code.add("import gpubackend")
  code.add("const RAM_SIZE = 1024")
  code.add("var RAM: array[0..RAM_SIZE-1, int]")
  code.add("var BUS: seq[string] = @[]")
  code.add("var Pins: array[0..31, bool]")
  code.add("")
  code.add("proc stripQuotes(s: string): string =")
  code.add("  if s.len >= 2 and s[0] == '\"' and s[^1] == '\"':")
  code.add("    return s[1..^2]")
  code.add("  else:")
  code.add("    return s")
  code.add("")
  # --- proc HW ---
  code.add("proc apuTran(name: string, payload: string) =")
  code.add("  BUS.add(payload)")
  code.add("  echo \"[APU-TRAN] \", name, \" -> \", payload")
  code.add("")
  code.add("proc apuMem(action: string, target: string, value: string) =")
  code.add("  var ramAddr = parseInt(target.replace(\"RAM\", \"\"))")
  code.add("  if action == \"write\":")
  code.add("    RAM[ramAddr] = parseInt(value)")
  code.add("  elif action == \"read\":")
  code.add("    echo \"[APU-MEM] RAM[\", ramAddr, \"] -> \", RAM[ramAddr]")
  code.add("")
  code.add("proc apuCore(mode: int, code: string) =")
  code.add("  echo \"[APU-CORE] Mode:\", mode, \" run:\", code")
  code.add("")
  code.add("proc apuPin(pin: int, state: string) =")
  code.add("  Pins[pin] = (state == \"high\")")
  code.add("")
  code.add("proc bitSend(bits: string) =")
  code.add("  BUS.add(bits)")
  code.add("")
  code.add("proc bitRecv() =")
  code.add("  if BUS.len > 0:")
  code.add("    echo BUS[0]")
  code.add("    delete(BUS, 0)")
  code.add("  else:")
  code.add("    echo \"[BIT-RECV] empty\"")
  code.add("")
  code.add("proc memMap(target: string) =")
  code.add("  echo \"[MEM-MAP] \", target")
  code.add("")
  code.add("proc memPush(target: string, value: string) =")
  code.add("  echo \"[MEM-PUSH] \", target, \" <- \", value")
  code.add("")
  code.add("proc tranPulse(pin: int, width: string) =")
  code.add("  echo \"[TRAN-PULSE] pin \", pin, \" width \", width")
  code.add("")

  # --- thu thập tên hàm ---
  var funcNames: seq[string] = @[]
  for k, _ in funcBodiesLocal:
    funcNames.add(k)

  # --- 1. Sinh tất cả proc trước ---
  traceInfo("Generating " & $funcBodiesLocal.len & " functions")
  for k, v in funcBodiesLocal:
    code.add("")
    code.add("proc " & k & "() =")
    var localVars: seq[string] = @[]
    var bidx = 0
    let bodyLines = genBlock(v, bidx, "  ", funcNames, localVars)
    if bodyLines.len == 0:
      code.add("  discard")
    else:
      for l in bodyLines: code.add(l)

  # --- 2. Sinh top-level code ---
  code.add("")
  var topVars: seq[string] = @[]
  var tidx = 0
  if tokens.len > 0:
    traceInfo("Generating top-level code")
    let topLines = genBlock(tokens, tidx, "", funcNames, topVars)
    for l in topLines: code.add(l)

  writeFile(nimFile, code.join("\n"))
  traceInfo("Generated " & $code.len & " lines of Nim code")
  echo "[INFO] Generated Nim code to ", nimFile
  
  let cmd = "nim c -d:release --path:\"" & getAppDir() & "\" -o:\"" & outFile & "\" \"" & nimFile & "\""
  traceInfo("Compiling with: " & cmd)
  let res = execProcess(cmd)
  echo res
  echo "[INFO] Built executable: ", outFile

# --------------------------
# Main
# --------------------------
proc main() =
  var inputFile = ""
  var args = commandLineParams()
  var aotFile = ""
  
  for i, a in args:
    if a == "--ignore-errors":
      ignoreErrors = true
    elif a == "--quiet":
      quietMode = true
    elif a.startsWith("--aot="):
      aotFile = a.split('=')[1]
    elif a == "--help" or a == "-h":
      echo "Usage: ./bybylang <file.bybylang> [options]"
      echo ""
      echo "Options:"
      echo "  --aot=FILE          Compile to executable FILE"
      echo "  --ignore-errors     Ignore non-fatal errors"
      echo "  --quiet             Suppress informational output"
      echo "  --help, -h          Show this help"
      quit(0)
    elif i == 0 or inputFile == "":
      inputFile = a

  if inputFile == "":
    echo "Usage: ./bybylang <file.bybylang> [--ignore-errors] [--quiet] [--aot=output]"
    quit(1)

  if not fileExists(inputFile):
    echo "[ERROR] File not found: ", inputFile
    quit(1)

  traceInfo("Starting BybyLang with file: " & inputFile)
  traceDebug("Arguments: " & $args)

  let tokens = tokenizeFile(inputFile)
  traceInfo("Tokenization complete, " & $tokens.len & " tokens")

  if aotFile != "":
    traceInfo("Generating AOT executable: " & aotFile)
    generateNimCode(tokens, aotFile)
  else:
    traceInfo("No AOT output specified")

  traceInfo("BybyLang finished")

main()