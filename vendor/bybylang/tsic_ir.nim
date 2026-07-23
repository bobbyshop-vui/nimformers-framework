# tsic_ir.nim - TSIC: intermediate representation (IR) trung gian cho BybyLang.
import std/[strutils, math]
import backends/cuda/cuda_driver
import backends/cuda/cuda_runtime
import backends/opencl/opencl_api
import backends/metal/metal_backend

type
  TsicOpcode* = enum
    tLoad, tStore, tAdd, tSub, tMul, tDiv, tRet,
    tFusedAddAct

  TsicOperand* = object
    isReg*: bool
    name*: string

  TsicInstr* = object
    op*: TsicOpcode
    dst*: TsicOperand
    src1*: TsicOperand
    src2*: TsicOperand
    imm*: int

  TsicProgram* = object
    name*: string
    instrs*: seq[TsicInstr]

type TsicBackend* = enum
  tbUnknown, tbCuda, tbMetal, tbOpenCL

var gTsicBackend = tbUnknown

proc tsicInit*(): TsicBackend =
  if gTsicBackend != tbUnknown:
    return gTsicBackend
  if cudaAvailable():
    gTsicBackend = tbCuda
  elif defined(macosx) and metalAvailable():
    gTsicBackend = tbMetal
  elif openclAvailable():
    gTsicBackend = tbOpenCL
  else:
    gTsicBackend = tbUnknown
  return gTsicBackend

proc reg(n: string): TsicOperand = TsicOperand(isReg: true, name: n)
proc buf(n: string): TsicOperand = TsicOperand(isReg: false, name: n)

proc binOpcode(op: string): TsicOpcode =
  case op
  of "add": tAdd
  of "sub": tSub
  of "mul": tMul
  of "div": tDiv
  else: tAdd

proc genTsicVecOp*(op: string): TsicProgram =
  result.name = "vecop_" & op
  result.instrs = @[
    TsicInstr(op: tLoad, dst: reg("r0"), src1: buf("a")),
    TsicInstr(op: tLoad, dst: reg("r1"), src1: buf("b")),
    TsicInstr(op: binOpcode(op), dst: reg("r2"), src1: reg("r0"), src2: reg("r1")),
    TsicInstr(op: tStore, dst: buf("c"), src1: reg("r2")),
    TsicInstr(op: tRet),
  ]

proc symbolOf(o: TsicOpcode): string =
  case o
  of tAdd: "+"
  of tSub: "-"
  of tMul: "*"
  of tDiv: "/"
  else: ""

proc tsicEmitPTX*(prog: TsicProgram): string =
  var op = tAdd
  for i in prog.instrs:
    if i.op in {tAdd, tSub, tMul, tDiv}: op = i.op
  let instr =
    case op
    of tAdd: "fma.rn.f32 \t%f3, %f2, 1.0, %f1;"
    of tSub: "sub.f32 \t%f3, %f2, %f1;"
    of tMul: "mul.f32 \t%f3, %f2, %f1;"
    of tDiv: "div.rn.f32 \t%f3, %f2, %f1;"
    else: "fma.rn.f32 \t%f3, %f2, 1.0, %f1;"
  result = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry """ & prog.name & """(
    .param .u64 vecop_param_0,
    .param .u64 vecop_param_1,
    .param .u64 vecop_param_2,
    .param .u32 vecop_param_3
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<11>;

    ld.param.u64 %rd1, [vecop_param_0];
    ld.param.u64 %rd2, [vecop_param_1];
    ld.param.u64 %rd3, [vecop_param_2];
    ld.param.u32 %r2, [vecop_param_3];
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r1, %r4, %r3, %r5;
    setp.ge.s32 %p1, %r1, %r2;
    @%p1 bra BB0_2;

    cvta.to.global.u64 %rd4, %rd1;
    mul.wide.s32 %rd5, %r1, 4;
    add.s64 %rd6, %rd4, %rd5;
    cvta.to.global.u64 %rd7, %rd2;
    add.s64 %rd8, %rd7, %rd5;
    ld.global.nc.f32 %f1, [%rd8];
    ld.global.nc.f32 %f2, [%rd6];
    """ & instr & """
    cvta.to.global.u64 %rd9, %rd3;
    add.s64 %rd10, %rd9, %rd5;
    st.global.f32 [%rd10], %f3;

BB0_2:
    ret;
}
"""

proc tsicEmitMSL*(prog: TsicProgram): string =
  var opSym = "+"
  for i in prog.instrs:
    if i.op in {tAdd, tSub, tMul, tDiv}: opSym = symbolOf(i.op)
  result = """
#include <metal_stdlib>
using namespace metal;
kernel void """ & prog.name & """(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* c [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    c[id] = a[id] """ & opSym & """ b[id];
}
"""

proc tsicEmitOpenCLC*(prog: TsicProgram): string =
  var opSym = "+"
  for i in prog.instrs:
    if i.op in {tAdd, tSub, tMul, tDiv}: opSym = symbolOf(i.op)
  result = """
__kernel void """ & prog.name & """(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] """ & opSym & """ b[i];
}
"""

proc tsicEmitGLSL*(prog: TsicProgram): string =
  var opSym = "+"
  for i in prog.instrs:
    if i.op in {tAdd, tSub, tMul, tDiv}: opSym = symbolOf(i.op)
  result = """
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) readonly buffer B { float b[]; };
layout(std430, binding = 2) writeonly buffer C { float c[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    c[i] = a[i] """ & opSym & """ b[i];
}
"""

proc opcodeBits(o: TsicOpcode): string =
  case o
  of tLoad:  "0000"
  of tStore: "0001"
  of tAdd:   "0010"
  of tSub:   "0011"
  of tMul:   "0100"
  of tDiv:   "0101"
  of tRet:   "0110"
  of tFusedAddAct: "0111"

proc bufIndex(name: string): int =
  case name
  of "a": 0
  of "b": 1
  of "c": 2
  else: 0

proc operandBits(o: TsicOperand): string =
  let idx =
    if o.isReg:
      try: parseInt(o.name[1..^1]) except: 0
    else:
      bufIndex(o.name)
  let regBit = if o.isReg: "1" else: "0"
  result = regBit & toBin(idx, 4)

proc tsicEmitBinary*(prog: TsicProgram): string =
  var lines: seq[string] = @[]
  for i in prog.instrs:
    let dstBits = if i.dst.name.len > 0: operandBits(i.dst) else: "00000"
    let s1Bits  = if i.src1.name.len > 0: operandBits(i.src1) else: "00000"
    let s2Bits  = if i.src2.name.len > 0: operandBits(i.src2) else: "00000"
    lines.add(opcodeBits(i.op) & " " & dstBits & " " & s1Bits & " " & s2Bits)
  result = lines.join("\n")

proc tsicAvailable*(): bool =
  let backend = tsicInit()
  result = backend != tbUnknown

proc tsicVecOp*(op: string, a, b: seq[float32]): seq[float32] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaVecOp(op, a, b)
  of tbMetal:
    return metalVecOp(op, a, b)
  of tbOpenCL:
    return openclVecOp(op, a, b)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

type
  TsicMatmulProgram* = object
    name*: string
    m*, k*, n*: int

proc genTsicMatmul*(m, k, n: int): TsicMatmulProgram =
  TsicMatmulProgram(name: "matmul_naive", m: m, k: k, n: n)

proc tsicEmitMatmulMSL*(prog: TsicMatmulProgram): string =
  result = """
#include <metal_stdlib>
using namespace metal;
kernel void """ & prog.name & """(device const float* a [[buffer(0)]],
                                   device const float* b [[buffer(1)]],
                                   device float* c [[buffer(2)]],
                                   constant int& M [[buffer(3)]],
                                   constant int& K [[buffer(4)]],
                                   constant int& N [[buffer(5)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= (uint)N || gid.y >= (uint)M) return;
    float sum = 0.0;
    for (int p = 0; p < K; p++) sum += a[gid.y * (uint)K + p] * b[(uint)p * (uint)N + gid.x];
    c[gid.y * (uint)N + gid.x] = sum;
}
"""

proc tsicEmitMatmulOpenCLC*(prog: TsicMatmulProgram): string =
  result = """
__kernel void """ & prog.name & """(__global const float* a, __global const float* b, __global float* c,
                            const int M, const int K, const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int p = 0; p < K; p++) sum += a[row * K + p] * b[p * N + col];
    c[row * N + col] = sum;
}
"""

proc tsicEmitMatmulGLSL*(prog: TsicMatmulProgram): string =
  result = """
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) readonly buffer B { float b[]; };
layout(std430, binding = 2) writeonly buffer C { float c[]; };
uniform int M; uniform int K; uniform int N;
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    if (row >= uint(M) || col >= uint(N)) return;
    float sum = 0.0;
    for (int p = 0; p < K; p++) sum += a[row * uint(K) + uint(p)] * b[uint(p) * uint(N) + col];
    c[row * uint(N) + col] = sum;
}
"""

proc tsicMatmulOp*(a, b: seq[float32], m, k, n: int): seq[float32] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaMatmulF32(a, b, m, k, n)
  of tbMetal:
    return metalMatmul(a, b, m, k, n)
  of tbOpenCL:
    return openclMatmul(a, b, m, k, n)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

proc tsicRelu*(x: seq[float32]): seq[float32] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaActivation("relu", x)
  of tbMetal:
    return metalActivation("relu", x)
  of tbOpenCL:
    return openclActivation("relu", x)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

proc tsicSigmoid*(x: seq[float32]): seq[float32] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaActivation("sigmoid", x)
  of tbMetal:
    return metalActivation("sigmoid", x)
  of tbOpenCL:
    return openclActivation("sigmoid", x)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

proc tsicTanh*(x: seq[float32]): seq[float32] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaActivation("tanh", x)
  of tbMetal:
    return metalActivation("tanh", x)
  of tbOpenCL:
    return openclActivation("tanh", x)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

proc tsicSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaSoftmax(x, rows, cols)
  of tbMetal:
    return metalSoftmax(x, rows, cols)
  of tbOpenCL:
    return openclSoftmax(x, rows, cols)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

proc tsicLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaLayernorm(x, gamma, beta, rows, cols, eps)
  of tbMetal:
    return metalLayernorm(x, gamma, beta, rows, cols, eps)
  of tbOpenCL:
    return openclLayernorm(x, gamma, beta, rows, cols, eps)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

proc tsicEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaEmbeddingLookup(table, indices, vocab, dim)
  of tbMetal:
    return metalEmbeddingLookup(table, indices, vocab, dim)
  of tbOpenCL:
    return openclEmbeddingLookup(table, indices, vocab, dim)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

# ============================================================
# SỬA: CUDA APFLU - GỌI KERNEL THẬT
# ============================================================
proc tsicApflu*(x: seq[float32], alpha, beta: float32): seq[float32] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaApflu(x, alpha, beta)
  of tbMetal:
    return metalApflu(x, alpha, beta)
  of tbOpenCL:
    return openclApflu(x, alpha, beta)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

# ============================================================
# SỬA: CUDA APFLU BACKWARD - GỌI KERNEL THẬT
# ============================================================
proc tsicApfluBackward*(x, dy: seq[float32], alpha, beta: float32): seq[float32] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaApfluBackward(x, dy, alpha, beta)
  of tbMetal:
    return metalApfluBackward(x, dy, alpha, beta)
  of tbOpenCL:
    return openclApfluBackward(x, dy, alpha, beta)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

# ============================================================
# SỬA: CUDA LAYERNORM BACKWARD - GỌI KERNEL THẬT
# ============================================================
proc tsicLayernormBackward*(dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
  of tbMetal:
    return metalLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
  of tbOpenCL:
    return openclLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

# ============================================================
# ATTENTION - ĐÃ GỌI CUDA ĐÚNG
# ============================================================
proc tsicAttentionFused*(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaAttentionFused(q, k, v, mask, B, H, S, D, scale)
  of tbMetal:
    return metalAttentionFused(q, k, v, mask, B, H, S, D, scale)
  of tbOpenCL:
    return openclAttentionFused(q, k, v, mask, B, H, S, D, scale)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

proc tsicAttentionFusedBackward*(q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  let backend = tsicInit()
  case backend
  of tbCuda:
    return cudaAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
  of tbMetal:
    return metalAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
  of tbOpenCL:
    return openclAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
  else:
    raise newException(CatchableError, "TSIC: No GPU backend available")

proc actImmOf(act: string): int =
  case act
  of "none": 0
  of "relu": 1
  of "sigmoid": 2
  of "tanh": 3
  else:
    raise newException(ValueError, "actImmOf: unknown activation: " & act)

proc genTsicFusedAddAct*(act: string): TsicProgram =
  result.name = "fused_add_" & act
  result.instrs = @[
    TsicInstr(op: tLoad, dst: reg("r0"), src1: buf("a")),
    TsicInstr(op: tLoad, dst: reg("r1"), src1: buf("b")),
    TsicInstr(op: tFusedAddAct, dst: reg("r2"), src1: reg("r0"), src2: reg("r1"), imm: actImmOf(act)),
    TsicInstr(op: tStore, dst: buf("c"), src1: reg("r2")),
    TsicInstr(op: tRet),
  ]

proc fusedActOf(prog: TsicProgram): int =
  result = 0
  for i in prog.instrs:
    if i.op == tFusedAddAct:
      return i.imm

proc tsicEmitFusedAddActPTX*(prog: TsicProgram): string =
  let act = fusedActOf(prog)
  let bodyInstr =
    case act
    of 0: "fma.rn.f32 \t%f3, %f2, 1.0, %f1;"
    of 1: "add.f32 \t%f3, %f2, %f1;\n    max.f32 \t%f3, %f3, %f0;"
    else:
      raise newException(ValueError, "tsicEmitFusedAddActPTX: activation " & $act & " chua co PTX")
  result = """
.version 7.5
.target sm_75
.address_size 64

.visible .entry """ & prog.name & """(
    .param .u64 fused_param_0,
    .param .u64 fused_param_1,
    .param .u64 fused_param_2,
    .param .u32 fused_param_3
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<11>;

    mov.f32 %f0, 0f00000000;
    ld.param.u64 %rd1, [fused_param_0];
    ld.param.u64 %rd2, [fused_param_1];
    ld.param.u64 %rd3, [fused_param_2];
    ld.param.u32 %r2, [fused_param_3];
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r1, %r4, %r3, %r5;
    setp.ge.s32 %p1, %r1, %r2;
    @%p1 bra BB0_2;

    cvta.to.global.u64 %rd4, %rd1;
    mul.wide.s32 %rd5, %r1, 4;
    add.s64 %rd6, %rd4, %rd5;
    cvta.to.global.u64 %rd7, %rd2;
    add.s64 %rd8, %rd7, %rd5;
    ld.global.nc.f32 %f1, [%rd6];
    ld.global.nc.f32 %f2, [%rd8];
    """ & bodyInstr & """
    cvta.to.global.u64 %rd9, %rd3;
    add.s64 %rd10, %rd9, %rd5;
    st.global.f32 [%rd10], %f3;

BB0_2:
    ret;
}
"""

proc tsicEmitFusedAddActMSL*(prog: TsicProgram): string =
  let act = fusedActOf(prog)
  let actExpr =
    case act
    of 0: "s"
    of 1: "max(s, 0.0f)"
    of 2: "1.0f / (1.0f + exp(-s))"
    of 3: "tanh(s)"
    else:
      raise newException(ValueError, "tsicEmitFusedAddActMSL: unknown imm " & $act)
  result = """
#include <metal_stdlib>
using namespace metal;
kernel void """ & prog.name & """(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* c [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    float s = a[id] + b[id];
    c[id] = """ & actExpr & """;
}
"""

proc tsicEmitFusedAddActOpenCLC*(prog: TsicProgram): string =
  let act = fusedActOf(prog)
  let actExpr =
    case act
    of 0: "s"
    of 1: "fmax(s, 0.0f)"
    of 2: "1.0f / (1.0f + exp(-s))"
    of 3: "tanh(s)"
    else:
      raise newException(ValueError, "tsicEmitFusedAddActOpenCLC: unknown imm " & $act)
  result = """
__kernel void """ & prog.name & """(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    float s = a[i] + b[i];
    c[i] = """ & actExpr & """;
}
"""