## Interpreter THAM CHIẾU cho bytecode micro-VM (CHỈ dùng trong test).
import std/math
import session_vm

type Th = object
  pc: int
  R: array[32, float32]
  U: array[32, uint32]
  stack: seq[tuple[pc, reg, it: int]]
  done, wait: bool

proc skipBlock(p: VmProgram; pc: var int; openOp, closeOp: uint32) =
  var d = 1
  while d > 0:
    let o = p.words[pc] and 0xFF'u32
    inc pc
    if opHasWord(o): inc pc
    if o == openOp: inc d
    elif o == closeOp: dec d

var gU8*: seq[uint8]
var gLut*: seq[float32]
var gIn*: seq[float32]

proc stepUntilBarrier(t: var Th; p: VmProgram; mem: var seq[float32];
                      tg: var seq[float32]; tid, lid, tgid: int) =
  t.R[0] = float32(tid)
  while not t.done and not t.wait:
    if t.pc >= p.words.len: t.done = true; break
    let w = p.words[t.pc]; inc t.pc
    let op = w and 0xFF'u32
    let a = int((w shr 8) and 0x1F); let b = int((w shr 13) and 0x1F); let c = int((w shr 18) and 0x1F)
    var k = 0'u32
    if opHasWord(op): k = p.words[t.pc]; inc t.pc
    template R: untyped = t.R
    template U: untyped = t.U
    case op
    of OP_NOP: discard
    of OP_HALT: t.done = true
    of OP_LOAD_F32: R[a] = mem[int(R[b]) + int(k)]
    of OP_STORE_F32: mem[int(R[b]) + int(k)] = R[a]
    of OP_LOAD_CONST: R[a] = cast[float32](k)
    of OP_ADD: R[a] = R[b] + R[c]
    of OP_MUL: R[a] = R[b] * R[c]
    of OP_FMA: R[a] = R[b] * R[c] + R[int(k and 0x1F)]
    of OP_SUB: R[a] = R[b] - R[c]
    of OP_DIV: R[a] = R[b] / R[c]
    of OP_EXP: R[a] = exp(R[b])
    of OP_NEG: R[a] = -R[b]
    of OP_SIGMOID: R[a] = 1'f32 / (1'f32 + exp(-R[b]))
    of OP_SOFTPLUS: R[a] = ln(1'f32 + exp(R[b]))
    of OP_RSQRT: R[a] = 1'f32 / sqrt(R[b])
    of OP_LOAD_F32_IDX: R[a] = mem[int(R[b])]
    of OP_STORE_F32_IDX: mem[int(R[b])] = R[a]
    of OP_ADD_CONST: R[a] = R[b] + cast[float32](k)
    of OP_MUL_CONST: R[a] = R[b] * cast[float32](k)
    of OP_CMP_LT: R[a] = (if R[b] < R[c]: 1'f32 else: 0'f32)
    of OP_IF_BEGIN:
      if R[a] == 0'f32: skipBlock(p, t.pc, OP_IF_BEGIN, OP_IF_END)
    of OP_IF_END: discard
    of OP_LOAD_LID: R[a] = float32(lid)
    of OP_LOAD_TGID: R[a] = float32(tgid)
    of OP_BARRIER_TG: t.wait = true
    of OP_TG_LOAD_F32: R[a] = tg[int(R[b]) + int(k)]
    of OP_TG_STORE_F32: tg[int(R[b]) + int(k)] = R[a]
    of OP_FOR_BEGIN:
      if int(R[a]) <= 0: skipBlock(p, t.pc, OP_FOR_BEGIN, OP_FOR_END)
      else: t.stack.add (t.pc, a, 0)
    of OP_FOR_END:
      inc t.stack[^1].it
      if t.stack[^1].it < int(R[t.stack[^1].reg]): t.pc = t.stack[^1].pc
      else: discard t.stack.pop
    of OP_U_CONST: U[a] = k
    of OP_U_ADD: U[a] = U[b] + U[c]
    of OP_U_ADDK: U[a] = U[b] + k
    of OP_U_MULK: U[a] = U[b] * k
    of OP_U_LOAD_F32: R[a] = mem[int(U[b] + k)]
    of OP_U_STORE_F32: mem[int(U[b] + k)] = R[a]
    of OP_U_TGID: U[a] = uint32(tgid)
    of OP_U_LID: U[a] = uint32(lid)
    of OP_U_MUL: U[a] = U[b] * U[c]
    of OP_U_ANDK: U[a] = U[b] and k
    of OP_U_SHRK: U[a] = U[b] shr k
    of OP_U_SHLK: U[a] = U[b] shl k
    of OP_U_SHR: U[a] = U[b] shr U[c]
    of OP_U_LOAD_U8: U[a] = uint32(gU8[int(U[b] + k)])
    of OP_U_LOAD_U32:
      let ad = int(U[b] + k)
      doAssert ad mod 4 == 0, "OP_U_LOAD_U32: dia chi khong chia het 4"
      U[a] = uint32(gU8[ad]) or (uint32(gU8[ad + 1]) shl 8) or (uint32(gU8[ad + 2]) shl 16) or (uint32(gU8[ad + 3]) shl 24)
    of OP_U_LOAD_CONSTF: R[a] = gLut[int(U[b] + k)]
    of OP_U_LOAD_IN: R[a] = gIn[int(U[b] + k)]
    of OP_LUT_U8: R[a] = gLut[int(uint32(R[b]) and 255'u32)]
    of OP_U_ATOMIC_ADD: mem[int(U[b] + k)] += R[a]
    of OP_U_TO_R: R[a] = float32(U[b])
    of OP_R_TO_U: U[a] = uint32(R[b])
    else: doAssert false, "op chua ho tro trong interpreter test: " & $op

proc runGrid*(p: VmProgram; mem: var seq[float32]; nThreads, tgSize: int; tgFloats = 4096) =
  let nTg = nThreads div tgSize
  for g in 0 ..< nTg:
    var tg = newSeq[float32](tgFloats)
    var ths = newSeq[Th](tgSize)
    while true:
      var alive = 0
      for l in 0 ..< tgSize:
        if not ths[l].done:
          stepUntilBarrier(ths[l], p, mem, tg, g * tgSize + l, l, g)
          if not ths[l].done: inc alive
      if alive == 0: break
      for l in 0 ..< tgSize: ths[l].wait = false

