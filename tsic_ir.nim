# tsic_ir.nim - TSIC: intermediate representation (IR) trung gian cho BybyLang.
#
# TSIC KHÔNG phải driver GPU thật (giống libcuda/OpenCL/Metal ở 3 module kia).
# TSIC là một "assembly ảo" nhỏ (giống PTX/SPIR-V ở quy mô tối thiểu): mỗi phép
# elementwise (add/sub/mul/div) được biểu diễn thành seq[TsicInstr], rồi từ đó
# "lower" (hạ tầng) sang:
#   - PTX text          (tsicEmitPTX)   -> nạp bằng cuModuleLoadData như cuda_driver.nim
#   - Metal Shading Lang (tsicEmitMSL)  -> nạp bằng newLibraryWithSource như metal_shim.m
#   - OpenCL C           (tsicEmitOpenCLC) -> nạp bằng clBuildProgram như opencl_api.nim
#   - GLSL compute shader (tsicEmitGLSL)   -> dùng cho backend OpenGL tự viết
#   - Nhị phân 0/1 tự định nghĩa (tsicEmitBinary) -> làm điểm khởi đầu để viết
#     driver cho GPU tự nghiên cứu/sản xuất: chỉ cần đọc seq[TsicInstr] (hoặc
#     bản encode nhị phân) rồi nạp vào ISA thật của phần cứng đó, tự do đổi
#     bảng mã opcode/operand bên dưới cho khớp thiết kế của bạn.
#
# Viết backend GPU mới từ TSIC: chỉ cần 1 proc `proc myGpuVecOp(op: string, a, b: seq[float32]): seq[float32]`
# gọi `genTsicVecOp(op)` rồi tsicEmitXXX(...) ra định dạng ISA của bạn, nạp và
# chạy trên driver riêng. Không cần đụng vào bybylang.nim hay gpubackend.nim,
# chỉ thêm 1 case gbXXX mới trỏ vào proc đó.
import std/[strutils, math]
import backends/cuda/cuda_driver
import backends/cuda/cuda_runtime
import backends/opencl/opencl_api
import backends/metal/metal_backend

type
  TsicOpcode* = enum
    tLoad, tStore, tAdd, tSub, tMul, tDiv, tRet

  TsicOperand* = object
    isReg*: bool   # true = thanh ghi ảo (r0, r1, ...) ; false = buffer vào/ra (a, b, c)
    name*: string

  TsicInstr* = object
    op*: TsicOpcode
    dst*: TsicOperand
    src1*: TsicOperand
    src2*: TsicOperand   # rỗng với load/store/ret

  TsicProgram* = object
    name*: string
    instrs*: seq[TsicInstr]

proc reg(n: string): TsicOperand = TsicOperand(isReg: true, name: n)
proc buf(n: string): TsicOperand = TsicOperand(isReg: false, name: n)

proc binOpcode(op: string): TsicOpcode =
  case op
  of "add": tAdd
  of "sub": tSub
  of "mul": tMul
  of "div": tDiv
  else: tAdd

# --------------------------------------------------------------------------
# Sinh IR cho 1 kernel elementwise vecop(a, b) -> c
# --------------------------------------------------------------------------
proc genTsicVecOp*(op: string): TsicProgram =
  result.name = "vecop_" & op
  result.instrs = @[
    TsicInstr(op: tLoad,           dst: reg("r0"), src1: buf("a")),
    TsicInstr(op: tLoad,           dst: reg("r1"), src1: buf("b")),
    TsicInstr(op: binOpcode(op),   dst: reg("r2"), src1: reg("r0"), src2: reg("r1")),
    TsicInstr(op: tStore,          dst: buf("c"),  src1: reg("r2")),
    TsicInstr(op: tRet),
  ]

proc symbolOf(o: TsicOpcode): string =
  case o
  of tAdd: "+"
  of tSub: "-"
  of tMul: "*"
  of tDiv: "/"
  else: ""

# --------------------------------------------------------------------------
# Lower -> PTX (tương thích cuModuleLoadData trong cuda_driver.nim)
# --------------------------------------------------------------------------
proc tsicEmitPTX*(prog: TsicProgram): string =
  var op = tAdd
  for i in prog.instrs:
    if i.op in {tAdd, tSub, tMul, tDiv}: op = i.op
  let instr =
    case op
    of tAdd: "add.f32 \t%f3, %f2, %f1;"
    of tSub: "sub.f32 \t%f3, %f2, %f1;"
    of tMul: "mul.f32 \t%f3, %f2, %f1;"
    of tDiv: "div.rn.f32 \t%f3, %f2, %f1;"
    else: "add.f32 \t%f3, %f2, %f1;"
  result = """
.version 7.0
.target sm_50
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
    ld.global.f32 %f1, [%rd8];
    ld.global.f32 %f2, [%rd6];
    """ & instr & """
    cvta.to.global.u64 %rd9, %rd3;
    add.s64 %rd10, %rd9, %rd5;
    st.global.f32 [%rd10], %f3;

BB0_2:
    ret;
}
"""

# --------------------------------------------------------------------------
# Lower -> Metal Shading Language (biên dịch runtime như metal_shim.m)
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Lower -> OpenCL C (biên dịch runtime bằng clBuildProgram)
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Lower -> GLSL compute shader (dùng cho backend OpenGL tự viết)
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Lower -> nhị phân 0/1 tự định nghĩa (điểm khởi đầu cho ISA của GPU tự làm).
# Mã hoá mặc định (đổi thoải mái cho khớp phần cứng thật):
#   opcode: 4 bit | dst: 1 bit (reg/buf) + 4 bit index | src1: 5 bit | src2: 5 bit
# --------------------------------------------------------------------------
proc opcodeBits(o: TsicOpcode): string =
  case o
  of tLoad:  "0000"
  of tStore: "0001"
  of tAdd:   "0010"
  of tSub:   "0011"
  of tMul:   "0100"
  of tDiv:   "0101"
  of tRet:   "0110"

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

# --------------------------------------------------------------------------
# Thực thi tham chiếu (software) trên CPU khi chưa có driver phần cứng riêng
# cho backend "tsic". Diễn giải trực tiếp seq[TsicInstr] để đảm bảo TSIC luôn
# chạy được ngay cả khi chưa viết xong backend GPU tự chế.
# --------------------------------------------------------------------------
proc tsicAvailable*(): bool =
  if cudaAvailable(): return true
  when defined(macosx):
    if metalAvailable(): return true
  if openclAvailable(): return true
  return false

proc tsicVecOp*(op: string, a, b: seq[float32]): seq[float32] =
  if cudaAvailable():
    return cudaVecOp(op, a, b)
  when defined(macosx):
    if metalAvailable():
      return metalVecOp(op, a, b)
  if openclAvailable():
    return openclVecOp(op, a, b)
  raise newException(CatchableError, "TSIC Error: No GPU backend (CUDA/Metal/OpenCL) is available, and CPU is forbidden on TSIC!")

# --------------------------------------------------------------------------
# TSIC matmul: C(m x n) = A(m x k) * B(k x n), row-major.
# Biểu diễn riêng khỏi TsicProgram (không phải elementwise theo từng phần tử
# độc lập) nhưng cùng triết lý: chỉ là mô tả trung gian, lower sang MSL/OpenCL C/
# GLSL để chạy thật, hoặc tự viết thêm lowering sang ISA GPU riêng.
# Muốn Tensor Core thật trên CUDA thì dùng cuda_runtime.nim (cublasSgemm với
# CUBLAS_TENSOR_OP_MATH) - PTX tay wmma.mma.sync không được sinh ở đây vì rất
# kén kiến trúc GPU, dễ sai; cuBLAS là đường an toàn và tổng quát hơn.
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
  if cudaAvailable():
    return cudaMatmulF32(a, b, m, k, n)
  when defined(macosx):
    if metalAvailable():
      return metalMatmul(a, b, m, k, n)
  if openclAvailable():
    return openclMatmul(a, b, m, k, n)
  raise newException(CatchableError, "TSIC Error: No GPU backend (CUDA/Metal/OpenCL) is available, and CPU is forbidden on TSIC!")

# --------------------------------------------------------------------------
# TSIC operations for complete backend support (no CPU fallback)
# --------------------------------------------------------------------------
proc tsicRelu*(x: seq[float32]): seq[float32] =
  if cudaAvailable():
    return cudaActivation("relu", x)
  when defined(macosx):
    if metalAvailable():
      return metalActivation("relu", x)
  if openclAvailable():
    return openclActivation("relu", x)
  raise newException(CatchableError, "TSIC Error: No GPU backend (CUDA/Metal/OpenCL) is available, and CPU is forbidden on TSIC!")

proc tsicSigmoid*(x: seq[float32]): seq[float32] =
  if cudaAvailable():
    return cudaActivation("sigmoid", x)
  when defined(macosx):
    if metalAvailable():
      return metalActivation("sigmoid", x)
  if openclAvailable():
    return openclActivation("sigmoid", x)
  raise newException(CatchableError, "TSIC Error: No GPU backend (CUDA/Metal/OpenCL) is available, and CPU is forbidden on TSIC!")

proc tsicTanh*(x: seq[float32]): seq[float32] =
  if cudaAvailable():
    return cudaActivation("tanh", x)
  when defined(macosx):
    if metalAvailable():
      return metalActivation("tanh", x)
  if openclAvailable():
    return openclActivation("tanh", x)
  raise newException(CatchableError, "TSIC Error: No GPU backend (CUDA/Metal/OpenCL) is available, and CPU is forbidden on TSIC!")

proc tsicSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
  if cudaAvailable():
    return cudaSoftmax(x, rows, cols)
  when defined(macosx):
    if metalAvailable():
      return metalSoftmax(x, rows, cols)
  if openclAvailable():
    return openclSoftmax(x, rows, cols)
  raise newException(CatchableError, "TSIC Error: No GPU backend (CUDA/Metal/OpenCL) is available, and CPU is forbidden on TSIC!")

proc tsicLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
  if cudaAvailable():
    return cudaLayernorm(x, gamma, beta, rows, cols, eps)
  when defined(macosx):
    if metalAvailable():
      return metalLayernorm(x, gamma, beta, rows, cols, eps)
  if openclAvailable():
    return openclLayernorm(x, gamma, beta, rows, cols, eps)
  raise newException(CatchableError, "TSIC Error: No GPU backend (CUDA/Metal/OpenCL) is available, and CPU is forbidden on TSIC!")

proc tsicEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
  if cudaAvailable():
    return cudaEmbeddingLookup(table, indices, vocab, dim)
  when defined(macosx):
    if metalAvailable():
      return metalEmbeddingLookup(table, indices, vocab, dim)
  if openclAvailable():
    return openclEmbeddingLookup(table, indices, vocab, dim)
  raise newException(CatchableError, "TSIC Error: No GPU backend (CUDA/Metal/OpenCL) is available, and CPU is forbidden on TSIC!")

