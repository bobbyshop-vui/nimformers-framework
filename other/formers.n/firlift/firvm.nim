## session_vm bytecode disassembly for the trace lifter.
##
## The FIR kernel text is a one-to-one front end for the same emit* calls, so a
## recorded VmProgram can be written back as kernel lines. Address words are
## rewritten relative to the op's ports; that makes the same kernel from two
## layer instances compare equal.
import std/[strutils, math]

const
  OpNop* = 0x00'u32
  OpHalt* = 0x01'u32
  OpLoadF32* = 0x10'u32
  OpStoreF32* = 0x11'u32
  OpLoadU8* = 0x12'u32
  OpLoadConst* = 0x13'u32
  OpLoadF32Idx* = 0x14'u32
  OpStoreF32Idx* = 0x15'u32
  OpLoadLid* = 0x16'u32
  OpLoadTgid* = 0x17'u32
  OpAdd* = 0x20'u32
  OpMul* = 0x21'u32
  OpFma* = 0x22'u32
  OpSub* = 0x23'u32
  OpExp* = 0x24'u32
  OpNeg* = 0x25'u32
  OpSigmoid* = 0x26'u32
  OpDiv* = 0x27'u32
  OpSoftplus* = 0x28'u32
  OpRsqrt* = 0x29'u32
  OpAddConst* = 0x2A'u32
  OpMulConst* = 0x2B'u32
  OpCmpLt* = 0x70'u32
  OpIfBegin* = 0x72'u32
  OpIfEnd* = 0x73'u32
  OpForBegin* = 0x40'u32
  OpForEnd* = 0x41'u32
  OpBarrierTg* = 0x50'u32
  OpTgLoadF32* = 0x60'u32
  OpTgStoreF32* = 0x61'u32
  OpUTgid* = 0x80'u32
  OpULid* = 0x81'u32
  OpUConst* = 0x82'u32
  OpUAdd* = 0x83'u32
  OpUMul* = 0x84'u32
  OpUAddK* = 0x85'u32
  OpUMulK* = 0x86'u32
  OpUAndK* = 0x87'u32
  OpUShrK* = 0x88'u32
  OpUShlK* = 0x89'u32
  OpUShr* = 0x8A'u32
  OpULoadU8* = 0x8B'u32
  OpULoadConstF* = 0x8C'u32
  OpULoadIn* = 0x8D'u32
  OpUStoreF32* = 0x8E'u32
  OpUToR* = 0x8F'u32
  OpRToU* = 0x90'u32
  OpULoadF32* = 0x91'u32
  OpUAtomicAdd* = 0x92'u32
  OpGridBarrier* = 0x93'u32
  OpBaseF32* = 0x94'u32
  OpBaseU8* = 0x95'u32
  OpBaseLut* = 0x96'u32
  OpBaseIn* = 0x97'u32
  OpULoadMeta* = 0x98'u32
  OpVtgBegin* = 0x99'u32
  OpVtgEnd* = 0x9A'u32
  OpLoadVact* = 0x9B'u32
  OpGridEpilogue* = 0x9C'u32
  OpULoadU32* = 0x9D'u32
  OpULoadIn4* = 0x9E'u32

type
  LiftPort* = object
    name*: string
    base*: int          ## first element of the port in the arena (or 0 for
                        ## resident weights)
    elements*: int

  LiftKernel* = object
    name*, tag*, text*: string
    threads*, tg*, tgBytes*: int
    ins*, outs*: int
    portSizes*: seq[int]
    tagConflict*: bool

proc fmtFloat(v: float32): string =
  if v.classify in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "non-finite float in kernel constant")
  $v

proc addrExpr*(value: uint32; ports: seq[LiftPort]): string =
  ## Prefer a symbolic `@port + offset` when the address lies inside a port.
  let v = int(value)
  for port in ports:
    if port.elements > 0 and v >= port.base and v < port.base + port.elements:
      let rel = v - port.base
      result = "@" & port.name
      if rel != 0:
        result.add "+" & $rel
      return
  return $value

proc uConstExpr*(value: uint32; ports: seq[LiftPort]): string =
  ## A nonzero uconst immediate that equals a port base is a buffer address.
  ## Zero is the loop/counter idiom (`uconst 0 0`) and stays numeric.
  let v = int(value)
  if v == 0: return "0"
  for port in ports:
    if port.elements > 0 and v == port.base:
      return "@" & port.name
  return $value

proc floatOf(w: uint32): float32 =
  copyMem(addr result, unsafeAddr w, 4)

proc needsWord*(op: uint32): bool =
  case op
  of OpLoadF32, OpStoreF32, OpLoadU8, OpLoadConst, OpFma, OpAddConst, OpMulConst,
     OpTgLoadF32, OpTgStoreF32, OpUConst, OpUAddK, OpUMulK, OpUAndK, OpUShrK,
     OpUShlK, OpULoadU8, OpULoadU32, OpULoadConstF, OpULoadIn, OpULoadIn4,
     OpUStoreF32, OpULoadF32, OpUAtomicAdd, OpGridBarrier, OpGridEpilogue,
     OpVtgBegin, OpULoadMeta:
    true
  else:
    false

proc arityWord*(op: uint32): int =
  case op
  of OpVtgBegin: 2
  else: 1

proc disassemble*(words: openArray[uint32]; ports: seq[LiftPort]): string =
  ## Text lines without the `kernel`/`endkernel` wrapper.
  var i = 0
  var lines: seq[string]
  while i < words.len:
    let w = words[i]
    let op = w and 0xFF'u32
    let a = (w shr 8) and 0x1F'u32
    let b = (w shr 13) and 0x1F'u32
    let c = (w shr 18) and 0x1F'u32
    inc i
    var args: seq[string]
    var mnemonic = ""
    proc push(x: string) = args.add x
    proc pushU(x: uint32) = args.add $x
    case op
    of OpNop: mnemonic = "nop"
    of OpHalt:
      if i != words.len:
        raise newException(ValueError, "halt before the end of the program")
      mnemonic = ""
    of OpAdd: mnemonic = "add"; pushU a; pushU b; pushU c
    of OpMul: mnemonic = "mul"; pushU a; pushU b; pushU c
    of OpSub: mnemonic = "sub"; pushU a; pushU b; pushU c
    of OpDiv: mnemonic = "div"; pushU a; pushU b; pushU c
    of OpExp: mnemonic = "exp"; pushU a; pushU b
    of OpNeg: mnemonic = "neg"; pushU a; pushU b
    of OpSigmoid: mnemonic = "sigmoid"; pushU a; pushU b
    of OpSoftplus: mnemonic = "softplus"; pushU a; pushU b
    of OpRsqrt: mnemonic = "rsqrt"; pushU a; pushU b
    of OpCmpLt: mnemonic = "cmplt"; pushU a; pushU b; pushU c
    of OpIfBegin: mnemonic = "ifbegin"; pushU a
    of OpIfEnd: mnemonic = "ifend"
    of OpForBegin: mnemonic = "forbegin"; pushU a
    of OpForEnd: mnemonic = "forend"
    of OpBarrierTg: mnemonic = "barriertg"
    of OpLoadLid: mnemonic = "loadlid"; pushU a
    of OpLoadTgid: mnemonic = "loadtgid"; pushU a
    of OpUTgid: mnemonic = "utgid"; pushU a
    of OpULid: mnemonic = "ulid"; pushU a
    of OpUAdd: mnemonic = "uadd"; pushU a; pushU b; pushU c
    of OpUMul: mnemonic = "umul"; pushU a; pushU b; pushU c
    of OpUShr: mnemonic = "ushr"; pushU a; pushU b; pushU c
    of OpUToR: mnemonic = "utor"; pushU a; pushU b
    of OpRToU: mnemonic = "rtou"; pushU a; pushU b
    of OpLoadF32Idx, OpStoreF32Idx:
      raise newException(ValueError, "dynamic indexed load/store is not expressible in FIR kernel text")
    of OpUConst:
      let value = words[i]; inc i
      mnemonic = "uconst"; pushU a; push(uConstExpr(value, ports))
    of OpUAddK, OpUMulK, OpUAndK, OpUShrK, OpUShlK:
      let value = words[i]; inc i
      mnemonic = case op
        of OpUAddK: "uaddk"
        of OpUMulK: "umulk"
        of OpUAndK: "uandk"
        of OpUShrK: "ushrk"
        else: "ushlk"
      pushU a; pushU b; pushU value
    of OpLoadF32, OpStoreF32:
      let value = words[i]; inc i
      mnemonic = if op == OpLoadF32: "loadf32" else: "storef32"
      pushU a; pushU b; push(addrExpr(value, ports))
    of OpLoadU8:
      raise newException(ValueError, "loadu8 is outside the FIR kernel whitelist")
    of OpLoadConst:
      let value = words[i]; inc i
      mnemonic = "loadconstf32"; pushU a; push(fmtFloat(floatOf(value)))
    of OpAddConst, OpMulConst:
      let value = words[i]; inc i
      mnemonic = if op == OpAddConst: "addconst" else: "mulconst"
      pushU a; pushU b; push(fmtFloat(floatOf(value)))
    of OpFma:
      let value = words[i]; inc i
      mnemonic = "fma"; pushU a; pushU b; pushU c; pushU (value and 0x1F)
    of OpTgLoadF32, OpTgStoreF32:
      let value = words[i]; inc i
      mnemonic = if op == OpTgLoadF32: "tgloadf32" else: "tgstoref32"
      pushU a; pushU b; push(addrExpr(value, ports))
    of OpULoadF32, OpUStoreF32, OpUAtomicAdd:
      let value = words[i]; inc i
      mnemonic = case op
        of OpULoadF32: "uloadf32"
        of OpUStoreF32: "ustoref32"
        else: "uatomicadd"
      pushU a; pushU b; push(addrExpr(value, ports))
    of OpULoadConstF, OpULoadIn, OpULoadIn4, OpULoadU8, OpULoadU32, OpULoadMeta:
      let value = words[i]; inc i
      mnemonic = case op
        of OpULoadConstF: "uloadconstf"
        of OpULoadIn: "uloadin"
        of OpULoadIn4: "uloadin4"
        of OpULoadU8: "uloadu8"
        of OpULoadU32: "uloadu32"
        else: "uloadmeta"
      pushU a; pushU b; push(addrExpr(value, ports))
    of OpGridBarrier:
      let value = words[i]; inc i
      mnemonic = "gridbarrier"; pushU value
    of OpGridEpilogue:
      let value = words[i]; inc i
      mnemonic = "gridepilogue"; pushU value
    of OpBaseF32: mnemonic = "basef32"; pushU a
    of OpBaseU8: mnemonic = "baseu8"; pushU a
    of OpBaseLut: mnemonic = "baselut"; pushU a
    of OpBaseIn: mnemonic = "basein"; pushU a
    of OpLoadVact: mnemonic = "loadvact"; pushU a
    of OpVtgBegin:
      let nVirt = words[i]; let param = words[i+1]; i += 2
      mnemonic = "vtgbegin"; pushU a; pushU nVirt; pushU param
    of OpVtgEnd: mnemonic = "vtgend"
    else:
      raise newException(ValueError, "unsupported session_vm opcode 0x" & toHex(op, 2) &
        " at word " & $(i-1))
    if mnemonic.len > 0:
      if args.len > 0:
        lines.add mnemonic & " " & args.join(" ")
      else:
        lines.add mnemonic
  result = lines.join("\n")
