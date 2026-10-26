import std/[strutils, sequtils]
import session_vm, fp4_gemv_vm
import jit_vm as nw
import jit_vm_orig as old

# --- 1) lowering y hệt bản gốc trên nhiều program thật
let progs = @[buildExampleLutFma(), buildFp4GemvProgram(4096, 1024, false),
  buildFp4GemvProgram(4096, 1024, true), buildF32GemvProgram(64, 32, 0.5'f32, true),
  buildBiasActProgram(true), buildEluPlus1Program(100), buildEluPlus1BackwardProgram(100),
  buildInt4AsymGemvProgram(512, 64, 64)]
var bad = 0
for i, p in progs:
  if nw.lowerBytecodeToMetal(p, "k") != old.lowerBytecodeToMetal(p, "k"):
    inc bad; echo "LECH program #", i
echo "lowering: ", progs.len - bad, "/", progs.len, " giong het ban goc"

# --- 2) tag sanitize == bản gốc (key khác nhau phần hash nên chỉ so đuôi)
let t = "a-b c.é_9"
doAssert nw.programKey(progs[0], t).split('_', 1)[1] == old.programKey(progs[0], t).split('_', 1)[1]

# --- 3) kiểm bytecode hash bằng interpreter tham chiếu (test only, CPU được)
proc refRun(p: VmProgram; mem: var seq[float32]) =
  var R: array[16, float32]; var U: array[16, uint32]
  var pc = 0
  let mu8 = cast[ptr UncheckedArray[uint8]](addr mem[0])
  var stack: seq[(int, int)]   # (pc đầu thân, còn lại)
  while pc < p.words.len:
    let w = p.words[pc]; inc pc
    let op = w and 0xFF; let a = int((w shr 8) and 0xF); let b = int((w shr 12) and 0xF); let c = int((w shr 16) and 0xF)
    var k = 0'u32
    if opHasWord(op): k = p.words[pc]; inc pc
    case op
    of OP_HALT: return
    of OP_LOAD_F32: R[a] = mem[int(R[b]) + int(k)]
    of OP_STORE_F32: mem[int(R[b]) + int(k)] = R[a]
    of OP_U_CONST: U[a] = k
    of OP_U_ADD: U[a] = U[b] + U[c]
    of OP_U_ADDK: U[a] = U[b] + k
    of OP_U_MULK: U[a] = U[b] * k
    of OP_U_ANDK: U[a] = U[b] and k
    of OP_U_SHRK: U[a] = U[b] shr k
    of OP_U_LOAD_U8: U[a] = uint32(mu8[int(U[b] + k)])
    of OP_U_TO_R: R[a] = float32(U[b])
    of OP_FOR_BEGIN:
      if int(R[a]) <= 0:
        var d = 1
        while d > 0:
          let o = p.words[pc] and 0xFF; inc pc
          if opHasWord(o): inc pc
          if o == OP_FOR_BEGIN: inc d
          if o == OP_FOR_END: dec d
      else: stack.add (pc, int(R[a]))
    of OP_FOR_END:
      dec stack[^1][1]
      if stack[^1][1] > 0: pc = stack[^1][0] else: discard stack.pop
    else: doAssert false, "op la " & $op
  discard

proc expectHash(words: seq[uint32]): uint64 =   # công thức mong đợi (độc lập)
  var h1 = 0x811C9DC5'u32; var h2 = 0x9E3779B1'u32
  for byt in cast[seq[uint8]](newSeq[uint8](0)): discard
  var bytes = newSeq[uint8](words.len * 4)
  if words.len > 0: copyMem(addr bytes[0], unsafeAddr words[0], bytes.len)
  for x in bytes:
    h1 = (h1 + x) * 0x01000193'u32
    h2 = (h2 + x) * 0x85EBCA6B'u32
  uint64(h1) or (uint64(h2) shl 32)

for p in progs[0..3]:
  let words = p.words
  var mem = newSeq[float32](8 + words.len)
  mem[0] = float32(words.len * 4)
  copyMem(addr mem[8], unsafeAddr words[0], words.len * 4)
  refRun(nw.buildHashProgram(), mem)
  let h1 = uint64(uint32(mem[1])) or (uint64(uint32(mem[2])) shl 16)
  let h2 = uint64(uint32(mem[3])) or (uint64(uint32(mem[4])) shl 16)
  doAssert ((h2 shl 32) or h1) == expectHash(words), "hash kernel sai"
echo "hash micro-kernel bytecode: dung (4 program)"

# --- 4) merge: giữ operand == 1, chỉ bỏ HALT thật
var x, y: VmProgram
x.emitUConst(0, 1'u32); x.emitHalt()      # operand = 1 trùng opcode HALT
y.emitUAddK(0, 0, 1'u32); y.emitHalt()
let m = nw.mergePrograms([x, y])
let om = old.mergePrograms([x, y])
echo "merge moi words=", m.words.len, " (ky vong 5)  ban cu words=", om.words.len, " (mat operand)"
doAssert m.words.len == 5
echo "ALL OK"
