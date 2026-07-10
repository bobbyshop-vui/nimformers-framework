# bybylang.nim - BybyLang AOT executable + Nim code generation + auto compile release
# Hỗ trợ cơ chế function: define function bằng "function NAME" ... kết thúc bằng một dòng chỉ chứa NAME
import strutils, os, osproc, tables, sequtils

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

# --------------------------
# Lexer đơn giản
# --------------------------

proc tokenizeLine(line: string): Token =
  var tok: Token
  # Đếm số khoảng trắng đầu dòng để xác định cấp indent
  tok.indent = line.len - line.strip(chars={' ', '\t'}).len

  # Loại bỏ khoảng trắng đầu cuối để xử lý cú pháp
  let clean = line.strip()

  if clean.len == 0:
    tok.sym = "empty"
    tok.text = ""
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
    return
  visited.add(absPath)

  if not fileExists(filename):
    echo "[ERROR] import: file not found: ", filename
    return

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

# Đọc file .bybylang gốc + toàn bộ file được import thành 1 danh sách tokens
proc tokenizeFile(filename: string): seq[Token] =
  var visited: seq[string] = @[]
  return resolveImports(filename, visited)
# --------------------------
# Hardware-level functions
# --------------------------
proc apuTran(name: string, payload: string) =
  BUS.add(payload)
  if not quietMode:
    echo "[APU-TRAN] ", name, " -> ", payload

proc apuMem(action: string, target: string, value: string) =
  let ramAddr = parseIntSafe(target.replace("RAM",""))
  if ramAddr < 0 or ramAddr >= RAM_SIZE:
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
  if not quietMode:
    echo "[APU-CORE] Mode: ", mode, ", running: ", code

proc apuPin(pin: int, state: string) =
  if pin < 0 or pin > 31:
    if not ignoreErrors:
      echo "[ERROR] Invalid pin: ", pin
      quit(1)
    return
  Pins[pin] = (state == "high")
  if not quietMode:
    echo "[APU-PIN] pin ", pin, " set ", state

proc bitSend(bits: string) =
  BUS.add(bits)
  if not quietMode:
    echo "[BIT-SEND] ", bits

proc bitRecv() =
  if BUS.len > 0:
    let b = BUS[0]
    delete(BUS, 0)
    if not quietMode:
      echo "[BIT-RECV] ", b
  else:
    if not quietMode:
      echo "[BIT-RECV] empty"

proc memMap(target: string) =
  if not quietMode:
    echo "[MEM-MAP] ", target

proc memPush(target: string, value: string) =
  if not quietMode:
    echo "[MEM-PUSH] ", target, " <- ", value

proc tranPulse(pin: int, width: string) =
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
#   gpu softmax X -> Y rows R cols C
#   gpu layernorm X, GAMMA, BETA -> Y rows R cols C eps E
#   gpu embedding TABLE, INDICES -> Y vocab V dim D
# Tất cả các lệnh trên đều đi qua cùng bộ kernel gpubackend.nim dùng chung với
# nimformer (backend.nim) -> thêm kernel mới ở gpubackend là tự động dùng được
# ở cả hai nơi, không cần viết riêng cho từng bên.
proc genGpuLine(line: string): seq[string] =
  result = @[]
  let rest = line[3..^1].strip()  # bỏ "gpu"

  if rest.startsWith("backend is"):
    let raw = rest["backend is".len..^1].strip()
    let val = stripQuotes(raw)
    result.add("gpuBackendSelected = parseBackend(\"" & val & "\")")

  elif rest.startsWith("array "):
    let body = rest["array ".len..^1].strip()
    let parts = body.split("=", 1)
    if parts.len == 2:
      let name = parts[0].strip()
      let valsRaw = parts[1].strip()
      result.add("var " & name & ": seq[float32] = @" & valsRaw & ".mapIt(it.float32)")
    else:
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
        if operands.len >= 2:
          let a = operands[0].strip()
          let b = operands[1].strip()

          if opName == "matmul":
            # Cú pháp: gpu matmul A, B -> C m M k K n N
            # C(m x n) = A(m x k) * B(k x n), row-major. m/k/n bắt buộc (khác
            # với "size" của vecop chỉ mang tính mô tả) vì matmul cần biết
            # hình dạng ma trận để chia việc đúng trên GPU.
            let rp = rhs.splitWhitespace()
            if rp.len >= 7 and rp[1] == "m" and rp[3] == "k" and rp[5] == "n":
              let target = rp[0]
              result.add("var " & target & " = gpuMatmul(gpuBackendSelected, " & a & ", " & b &
                          ", " & rp[2] & ", " & rp[4] & ", " & rp[6] & ")")
            else:
              result.add("# [WARN] cú pháp 'gpu matmul' cần: gpu matmul A, B -> C m M k K n N : " & line)
          elif opName == "layernorm":
            # Cú pháp: gpu layernorm X, GAMMA, BETA -> Y rows R cols C eps E
            if operands.len >= 3:
              let gamma = operands[1].strip()
              let beta = operands[2].strip()
              let rp = rhs.splitWhitespace()
              if rp.len >= 7 and rp[1] == "rows" and rp[3] == "cols" and rp[5] == "eps":
                let target = rp[0]
                result.add("var " & target & " = gpuLayernorm(gpuBackendSelected, " & a & ", " & gamma &
                            ", " & beta & ", " & rp[2] & ", " & rp[4] & ", " & rp[6] & ")")
              else:
                result.add("# [WARN] cú pháp 'gpu layernorm' cần: gpu layernorm X, GAMMA, BETA -> Y rows R cols C eps E : " & line)
            else:
              result.add("# [WARN] cú pháp 'gpu layernorm' cần 3 toán hạng: " & line)
          # Trong genGpuLine, phần xử lý embedding:
          elif opName == "embedding":
            let rp = rhs.splitWhitespace()
            if rp.len >= 5 and rp[1] == "vocab" and rp[3] == "dim":
              let target = rp[0]
              # Chuyển đổi indices thành seq[int32] nếu cần
              result.add("var " & target & " = gpuEmbeddingLookup(gpuBackendSelected, " & a & 
                        ", " & b & ".mapIt(int32(it)), " & rp[2] & ", " & rp[4] & ")")
            else:
              result.add("# [WARN] cú pháp 'gpu embedding' cần: gpu embedding TABLE, INDICES -> Y vocab V dim D : " & line)
          else:
            var target = rhs
            if "size" in rhs:
              target = rhs.split("size")[0].strip()
            result.add("var " & target & " = gpuOp(\"" & opName & "\", gpuBackendSelected, " & a & ", " & b & ")")
        elif operands.len == 1:
          # Toán tử 1 toán hạng: relu/sigmoid/tanh (elementwise) hoặc softmax
          # (cần thêm rows/cols vì softmax chuẩn hoá theo từng hàng, khác vecop).
          let x = operands[0].strip()
          case opName
          of "relu", "sigmoid", "tanh":
            let target = rhs.splitWhitespace()[0]
            let fn = if opName == "relu": "gpuRelu"
                     elif opName == "sigmoid": "gpuSigmoid"
                     else: "gpuTanh"
            result.add("var " & target & " = " & fn & "(gpuBackendSelected, " & x & ")")
          of "softmax":
            # Cú pháp: gpu softmax X -> Y rows R cols C
            let rp = rhs.splitWhitespace()
            if rp.len >= 5 and rp[1] == "rows" and rp[3] == "cols":
              let target = rp[0]
              result.add("var " & target & " = gpuSoftmax(gpuBackendSelected, " & x & ", " & rp[2] & ", " & rp[4] & ")")
            else:
              result.add("# [WARN] cú pháp 'gpu softmax' cần: gpu softmax X -> Y rows R cols C : " & line)
          else:
            result.add("# [WARN] cú pháp 'gpu " & opName & "' cần 2 toán hạng: " & line)
        else:
          result.add("# [WARN] cú pháp 'gpu " & opName & "' cần toán hạng: " & line)
      else:
        result.add("# [WARN] cú pháp gpu không hợp lệ: " & line)
    else:
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
# Trước đây 3 nhánh if/while/for chép y hệt đoạn kiểm tra này -> giờ chỉ 1 chỗ.
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
      # Indent lệch bất thường (đáng lẽ phải được khối con xử lý) -> bỏ qua an toàn
      idx.inc
      continue

    case tk.sym
    of "empty":
      idx.inc

    of "import":
      idx.inc  # đã được resolveImports xử lý trước đó, còn sót thì bỏ qua

    of "print":
      result.add(indent & "echo " & tk.text.replace("print", "").strip())
      idx.inc

    of "function":
      # Định nghĩa hàm chỉ hợp lệ ở top-level và đã được sinh ở bước 1;
      # nếu gặp lồng trong khối khác thì bỏ qua toàn bộ thân của nó.
      let fBase = tk.indent
      idx.inc
      while idx < tokens.len and tokens[idx].indent > fBase:
        idx.inc

    else:
      let line = tk.text.strip()

      if line.len == 0:
        idx.inc

      elif line.startsWith("if ") or line.startsWith("elif ") or line == "else:":
        result.add(genHeaderedBlock(tokens, idx, indent, line, blockIndent, funcNames, localVars))

      elif line.startsWith("while "):
        result.add(genHeaderedBlock(tokens, idx, indent, line, blockIndent, funcNames, localVars))

      elif line.startsWith("for "):
        var forLine = line
        if "range(" in forLine:
          let inside = forLine.split("range(")[1].split(")")[0]
          let parts = inside.split(",")
          if parts.len == 2:
            forLine = forLine.replace("range(" & inside & ")",
                                       parts[0].strip() & ".." & parts[1].strip())
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
    else:
      idx.inc

  # --- khởi tạo file ---
  var nimFile = outFile
  if not nimFile.endsWith(".nim"): nimFile &= ".nim"

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

  # --- 1. Sinh tất cả proc trước (đệ quy, hỗ trợ lồng nhau vô hạn cấp) ---
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

  # --- 2. Sinh top-level code (đệ quy, hỗ trợ lồng nhau vô hạn cấp) ---
  code.add("")
  var topVars: seq[string] = @[]
  var tidx = 0
  if tokens.len > 0:
    let topLines = genBlock(tokens, tidx, "", funcNames, topVars)
    for l in topLines: code.add(l)

  writeFile(nimFile, code.join("\n"))
  echo "[INFO] Generated Nim code to ", nimFile
  # --path: trỏ về thư mục chứa bybylang (nơi có gpubackend.nim + backends/)
  # để `import gpubackend` trong file sinh ra luôn tìm thấy module, bất kể
  # người dùng chạy ./bybylang từ thư mục nào hay --aot=... trỏ ra thư mục khác.
  
  # Sửa lỗi trên Windows: 
  # 1. Không dùng --run (vì đang compile, không chạy)
  # 2. Dùng đường dẫn đúng cho Windows
  let cmd = "nim c -d:release --path:\"" & getAppDir() & "\" -o:\"" & outFile & "\" \"" & nimFile & "\""
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
  discard initTable[string, seq[Token]]()
  for i, a in args:
    if a == "--ignore-errors":
      ignoreErrors = true
    elif a == "--quiet":
      quietMode = true
    elif a.startsWith("--aot="):
      aotFile = a.split('=')[1]
    elif i == 0 or inputFile == "":
      inputFile = a

  if inputFile == "":
    echo "Usage: ./bybylang <file.bybylang> [--ignore-errors] [--quiet] [--aot=output]"
    quit(1)

  if not fileExists(inputFile):
    echo "[ERROR] File not found: ", inputFile
    quit(1)

  # --- Sửa tại đây ---
  let tokens = tokenizeFile(inputFile)

  if aotFile != "":
    generateNimCode(tokens, aotFile)

main()