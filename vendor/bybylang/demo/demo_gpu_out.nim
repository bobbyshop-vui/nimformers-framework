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
gpuBackendSelected = parseBackend("tsic")
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
var SigR = gpuSigmoid(gpuBackendSelected, A)
var L = gpuTanh(gpuBackendSelected, A)
var APFLURES = gpuApflu(gpuBackendSelected, A, 0.1, 1.0)
var APFLURESB = gpuApflu(gpuBackendSelected, B, 0.1'f32, 1.0'f32)
var M = gpuSoftmax(gpuBackendSelected, A, 1, 8)
var N = gpuMatmul(gpuBackendSelected, A, B, 1, A.len, 1)
var LAYERNORMRES = gpuLayernorm(gpuBackendSelected, A, B, C, 8, 1, 1e-5)
var DY: seq[float32] = @[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].mapIt(it.float32)
let lnBwd = gpuLayernormBackward(gpuBackendSelected, DY, A, B, C, 8, 1, 1e-5)
var DX = lnBwd.dx
var DGAMMA = lnBwd.dgamma
var DBETA = lnBwd.dbeta
var EMB_TABLE: seq[float32] = @[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64].mapIt(it.float32)
var indices = [0, 1, 2, 3, 4, 5, 6, 7]
var EMBEDRES = gpuEmbeddingLookup(gpuBackendSelected, EMB_TABLE, indices.mapIt(int32(it)), 8, 8)
var Q: seq[float32] = @[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].mapIt(it.float32)
var MatKey: seq[float32] = @[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6].mapIt(it.float32)
var V: seq[float32] = @[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32].mapIt(it.float32)
let attnRes = gpuAttentionFused(gpuBackendSelected, Q, MatKey, V, @[], 1, 1, 4, 4, 0.7071)
var ATTENTIONO = attnRes.o
var ATTENTIONS = attnRes.s_matrix
var DYATTN: seq[float32] = @[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16].mapIt(it.float32)
let attnBwd = gpuAttentionFusedBackward(gpuBackendSelected, Q, MatKey, V, ATTENTIONS, DYATTN, 1, 1, 4, 4, 0.7071)
var DQ = attnBwd.dq
var DK = attnBwd.dk
var DV = attnBwd.dv
var SESSC: seq[float32] = @[]
var SESSD: seq[float32] = @[]
var SESSE: seq[float32] = @[]
var SESSM: seq[float32] = @[]
block:
  var sessOk = sessionBegin(gpuBackendSelected)
  if sessOk:
    var sh1 = sessionUpload(gpuBackendSelected, A)
    var sh2 = sessionUpload(gpuBackendSelected, B)
    var sh3: SessionHandle
    var sh4: SessionHandle
    var sh5: SessionHandle
    var sh6: SessionHandle
    discard sessionVecOp(gpuBackendSelected, "add", sh1, sh2, sh3, 8)
    discard sessionVecOp(gpuBackendSelected, "mul", sh3, sh1, sh4, 8)
    discard sessionActivation(gpuBackendSelected, "relu", sh4, sh5, 8)
    discard sessionSoftmax(gpuBackendSelected, sh1, sh6, 1, 8)
    discard sessionEnd(gpuBackendSelected)
    SESSC = sessionRead(gpuBackendSelected, sh3, 8)
    SESSD = sessionRead(gpuBackendSelected, sh4, 8)
    SESSE = sessionRead(gpuBackendSelected, sh5, 8)
    SESSM = sessionRead(gpuBackendSelected, sh6, 1 * 8)
    sessionFree(gpuBackendSelected, sh3)
    sessionFree(gpuBackendSelected, sh4)
    sessionFree(gpuBackendSelected, sh5)
    sessionFree(gpuBackendSelected, sh6)
    sessionFree(gpuBackendSelected, sh2)
    sessionFree(gpuBackendSelected, sh1)
  else:
    echo "[WARN] session không khởi tạo được, fallback sang immediate"
    SESSC = gpuOp("add", gpuBackendSelected, A, B)
    SESSD = gpuOp("mul", gpuBackendSelected, SESSC, A)
    SESSE = gpuRelu(gpuBackendSelected, SESSD)
    SESSM = gpuSoftmax(gpuBackendSelected, A, 1, 8)
echo "=== GPU VECTOR OPERATIONS ==="
echo "GPU add result:"
echo C
echo "GPU mul result (A*B):"
echo D
echo "GPU mul result (E*O):"
echo G
echo "GPU sub result:"
echo H
echo "GPU div result:"
echo I
echo "=== GPU ACTIVATIONS ==="
echo "GPU relu result:"
echo J
echo "GPU sigmoid result:"
echo SigR
echo "GPU tanh result:"
echo L
echo "GPU apflu result (A):"
echo APFLURES
echo "GPU apflu result (B):"
echo APFLURESB
echo "=== GPU SOFTMAX ==="
echo "GPU softmax result:"
echo M
echo "=== GPU MATMUL ==="
echo "GPU matmul result:"
echo N
echo "=== GPU LAYERNORM ==="
echo "GPU layernorm result:"
echo LAYERNORMRES
echo "GPU layernorm backward DX:"
echo DX
echo "GPU layernorm backward DGAMMA:"
echo DGAMMA
echo "GPU layernorm backward DBETA:"
echo DBETA
echo "=== GPU EMBEDDING ==="
echo "GPU embedding result:"
echo EMBEDRES
echo "=== GPU ATTENTION ==="
echo "GPU attention output:"
echo ATTENTIONO
echo "GPU attention score matrix:"
echo ATTENTIONS
echo "GPU attention backward DQ:"
echo DQ
echo "GPU attention backward DK:"
echo DK
echo "GPU attention backward DV:"
echo DV
echo "=== SESSION RESULTS ==="
echo "GPU session add result (A+B):"
echo SESSC
echo "GPU session mul result (SESSC*A):"
echo SESSD
echo "GPU session relu result (SESSD):"
echo SESSE
echo "GPU session softmax result (A):"
echo SESSM
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