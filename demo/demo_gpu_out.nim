import strutils, sequtils
import gpubackend
const RAM_SIZE = 1024
var RAM: array[0..RAM_SIZE-1, int]
var BUS: seq[string] = @[]
var Pins: array[0..31, bool]

proc stripQuotes(s: string): string =
  if s.len >= 2 and s[0] == '"' and s[^1] == '"':
    return s[1..^2]
  else:
    return s

proc apuTran(name: string, payload: string) =
  BUS.add(payload)
  echo "[APU-TRAN] ", name, " -> ", payload

proc apuMem(action: string, target: string, value: string) =
  var ramAddr = parseInt(target.replace("RAM", ""))
  if action == "write":
    RAM[ramAddr] = parseInt(value)
  elif action == "read":
    echo "[APU-MEM] RAM[", ramAddr, "] -> ", RAM[ramAddr]

proc apuCore(mode: int, code: string) =
  echo "[APU-CORE] Mode:", mode, " run:", code

proc apuPin(pin: int, state: string) =
  Pins[pin] = (state == "high")

proc bitSend(bits: string) =
  BUS.add(bits)

proc bitRecv() =
  if BUS.len > 0:
    echo BUS[0]
    delete(BUS, 0)
  else:
    echo "[BIT-RECV] empty"

proc memMap(target: string) =
  echo "[MEM-MAP] ", target

proc memPush(target: string, value: string) =
  echo "[MEM-PUSH] ", target, " <- ", value

proc tranPulse(pin: int, width: string) =
  echo "[TRAN-PULSE] pin ", pin, " width ", width


echo "Mode 3: High-level"
gpuBackendSelected = parseBackend("opencl")
var A: seq[float32] = @[1, 2, 3, 4, 5, 6, 7, 8].mapIt(it.float32)
var B: seq[float32] = @[10, 20, 30, 40, 50, 60, 70, 80].mapIt(it.float32)
var E: seq[float32] = @[10, 20, 30, 40, 50, 60, 70, 80, 100, 110, 120, 130, 134, 12122121123123123, 123123123].mapIt(it.float32)
var O: seq[float32] = @[10, 20, 30, 40, 50, 60, 70, 80, 100, 110, 120, 130, 134, 12122121123123123, 123123123].mapIt(it.float32)
var C = gpuOp("add", gpuBackendSelected, A, B)
var D = gpuOp("mul", gpuBackendSelected, A, B)
var G = gpuOp("mul", gpuBackendSelected, E, O)
var H = gpuOp("sub", gpuBackendSelected, A, B)
var I = gpuOp("div", gpuBackendSelected, A, B)
var J = gpuRelu(gpuBackendSelected, A)
var K = gpuSigmoid(gpuBackendSelected, A)
var L = gpuTanh(gpuBackendSelected, A)
var M = gpuSoftmax(gpuBackendSelected, A, 1, 8)
var N = gpuMatmul(gpuBackendSelected, A, B, 8, 8, 1)
echo "GPU add result:"
echo C
echo "GPU mul result:"
echo D
echo "GPU sub result:"
echo H
echo "GPU div result:"
echo I
echo "GPU relu result:"
echo J
echo "GPU sigmoid result:"
echo K
echo "GPU tanh result:"
echo L
echo "GPU softmax result:"
echo M
echo "GPU matmul result:"
echo N
echo G
var x = 5
if x > 0:
  echo "x duong"
  if x > 3:
    echo "x lon hon 3"
    if x == 5:
      echo "x dung bang 5 - if long 3 cap hoat dong!"
    else:
      echo "x khac 5"
  else:
    echo "x tu 1 den 3"
elif x == 0:
  echo "x bang 0"
else:
  echo "x am"
for i in 1..4:
  if i == 2:
    echo "gap so 2 trong vong lap"
  else:
    echo "khong phai so 2"
# Các phép toán GPU nâng cao
var LAYERNORM_RES = gpuLayernorm(gpuBackendSelected, A, B, C, 8, 1, 1e-5)
# Tạo indices riêng cho embedding (dùng biến thường, không phải gpu array)
var indices = [0, 1, 2, 3, 4, 5, 6, 7]
var EMBED_RES = gpuEmbeddingLookup(gpuBackendSelected, E, indices.mapIt(int32(it)), 8, 8)
echo "GPU layernorm result:"
echo LAYERNORM_RES
echo "GPU embedding result:"
echo EMBED_RES