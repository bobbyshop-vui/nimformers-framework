## Dispatch fusion for slot-layout session_vm kernels.
##
## Two consecutive dispatches A and B can become one dispatch `A; barriertg; B`
## when every threadgroup of the fused launch only needs data that its own
## threadgroup produced. The VM has no grid-wide barrier, so this is the only
## sound merge: `OP_BARRIER_TG` orders the threads inside one threadgroup and
## the proof below shows that no other threadgroup can interfere.
##
## The proof is exact, not heuristic. Both programs are interpreted for every
## (threadgroup, lane) with a scalar abstract machine that tracks the integer
## pipeline concretely (addresses, loop counts, guards) and marks everything
## that comes from memory as unknown. The mem-domain footprint of A and B is
## recorded per threadgroup, then:
##   * a footprint address touched by both kernels, where either one writes it,
##     must be touched by one single threadgroup only (same in A and in B);
##   * an unknown address, an unknown loop count or an unknown branch rejects;
##   * B reading a register before writing it rejects (A would leak into it);
##   * the interpreter has a step budget; running out rejects.
## Anything the analysis cannot decide is reported as a rejection reason and
## the two dispatches stay separate.
import std/[tables, sets, math, strutils]
import ./firvm

type
  FuseKernel* = object
    words*: seq[uint32]
    threads*, tg*, tgBytes*: int
    bindings*: seq[(string, string)]   ## (domain, buffer) in slot order

  Footprint = object
    acc: Table[uint64, int32]          ## mem address -> the one threadgroup touching it, -1 = several
    wr, rd: HashSet[uint64]
    initR, initU: uint64               ## registers read before the kernel wrote them

  FusePlan* = object
    restoreR*, restoreU*: uint64       ## registers B reads at their launch value, reset before B

  FuseStats* = object
    pairs*, fused*, bindable*: int      ## bindable: consecutive pairs sharing one binding set
    chains*, chained*: int              ## persistent chains built / dispatches they replaced
    reasons*: CountTable[string]

  Reject = object of CatchableError

const DefaultBudget* = 40_000_000

var fuseStats*: FuseStats

proc reject(msg: string) {.noreturn.} =
  raise newException(Reject, msg)

type OpInfo = object
  at: int
  op, a, b, c: uint32
  imm: uint32
  match: int                           ## matching end for loops and branches

proc decode(words: seq[uint32]): seq[OpInfo] =
  var i = 0
  var scopes: seq[int]
  while i < words.len:
    let w = words[i]
    var info = OpInfo(at: i, op: w and 0xFF'u32, a: (w shr 8) and 0x1F'u32,
      b: (w shr 13) and 0x1F'u32, c: (w shr 18) and 0x1F'u32, match: -1)
    inc i
    if info.op == OpVtgBegin or info.op == OpGridBarrier or info.op == OpGridEpilogue or
       info.op == OpVtgEnd or info.op == OpLoadVact or
       (info.op >= OpBaseF32 and info.op <= OpBaseIn):
      reject("persistent-kernel opcode 0x" & toHex(info.op, 2))
    if needsWord(info.op):
      if i >= words.len: reject("truncated immediate")
      info.imm = words[i]
      inc i
    result.add info
    case info.op
    of OpForBegin, OpIfBegin: scopes.add result.high
    of OpForEnd, OpIfEnd:
      if scopes.len == 0: reject("unbalanced scope")
      result[scopes.pop()].match = result.high
    of OpHalt:
      if i != words.len: reject("halt before the end")
    else: discard
  if scopes.len != 0: reject("unbalanced scope")
  if result.len == 0 or result[^1].op != OpHalt: reject("program does not end in halt")

type
  State = enum stKnown, stUnknown

proc fbits(v: float32): uint32 =
  copyMem(addr result, unsafeAddr v, 4)

proc ffrom(w: uint32): float32 =
  copyMem(addr result, unsafeAddr w, 4)

proc footprintOf(prog: seq[OpInfo]; threads, tg: int; budget: var int): Footprint =
  var fp: Footprint
  let groups = (threads + tg - 1) div tg
  for g in 0 ..< groups:
    for lane in 0 ..< tg:
      var r: array[64, float32]
      var rs: array[64, State]
      var u: array[64, uint32]
      var us: array[64, State]
      var wrR, wrU: uint64             # written by this run
      r[0] = float32(g * tg + lane)    # every VM layout starts with R0 = thread id, all else zero
      var loops: seq[(int, int)]       # (begin op index, iterations left)
      var pc = 0
      proc rr(k: uint32): float32 =
        if (wrR and (1'u64 shl k)) == 0: fp.initR = fp.initR or (1'u64 shl k)
        r[k]
      proc ru(k: uint32): uint32 =
        if (wrU and (1'u64 shl k)) == 0: fp.initU = fp.initU or (1'u64 shl k)
        u[k]
      proc needInit(k: uint32) = discard rr(k)
      proc needInitU(k: uint32) = discard ru(k)
      proc setR(k: uint32; v: float32; st: State) =
        r[k] = v; rs[k] = st; wrR = wrR or (1'u64 shl k)
      proc setU(k: uint32; v: uint32; st: State) =
        u[k] = v; us[k] = st; wrU = wrU or (1'u64 shl k)
      proc bin(op: OpInfo; f: proc(x, y: float32): float32 {.nimcall.}) =
        let x = rr(op.b); let y = rr(op.c)
        if rs[op.b] == stKnown and rs[op.c] == stKnown: setR(op.a, f(x, y), stKnown)
        else: setR(op.a, 0, stUnknown)
      proc un(op: OpInfo; f: proc(x: float32): float32 {.nimcall.}) =
        let x = rr(op.b)
        if rs[op.b] == stKnown: setR(op.a, f(x), stKnown) else: setR(op.a, 0, stUnknown)
      proc touch(key: uint64; write: bool; read = false) =
        let cur = fp.acc.getOrDefault(key, -2'i32)
        if cur == -2: fp.acc[key] = int32(g)
        elif cur != int32(g): fp.acc[key] = -1
        if write: fp.wr.incl key
        if read: fp.rd.incl key
      proc uaddr(op: OpInfo; write, read: bool) =
        needInitU(op.b)
        if us[op.b] != stKnown: reject("data-dependent mem address")
        touch(uint64(u[op.b] + op.imm), write, read)
      proc raddr(op: OpInfo; off: uint32; write: bool) =
        # (a load also reads: see the callers)
        let x = rr(op.b)
        if rs[op.b] != stKnown: reject("data-dependent mem address")
        touch(uint64(uint32(x) + off), write, not write)
      while pc < prog.len:
        dec budget
        if budget < 0: reject("analysis budget exceeded")
        let op = prog[pc]
        var next = pc + 1
        case op.op
        of OpNop, OpBarrierTg: discard
        of OpHalt: break
        of OpLoadConst: setR(op.a, ffrom(op.imm), stKnown)
        of OpLoadLid: setR(op.a, float32(lane), stKnown)
        of OpLoadTgid: setR(op.a, float32(g), stKnown)
        of OpAdd: bin(op, proc(x, y: float32): float32 {.nimcall.} = x + y)
        of OpSub: bin(op, proc(x, y: float32): float32 {.nimcall.} = x - y)
        of OpMul: bin(op, proc(x, y: float32): float32 {.nimcall.} = x * y)
        of OpDiv: bin(op, proc(x, y: float32): float32 {.nimcall.} = x / y)
        of OpCmpLt: bin(op, proc(x, y: float32): float32 {.nimcall.} = (if x < y: 1'f32 else: 0'f32))
        of OpFma:
          let rc = op.imm and 0x1F'u32
          let x = rr(op.b); let y = rr(op.c); let z = rr(rc)
          if rs[op.b] == stKnown and rs[op.c] == stKnown and rs[rc] == stKnown:
            setR(op.a, x * y + z, stKnown)
          else: setR(op.a, 0, stUnknown)
        of OpExp: un(op, proc(x: float32): float32 {.nimcall.} = exp(x))
        of OpNeg: un(op, proc(x: float32): float32 {.nimcall.} = -x)
        of OpSigmoid: un(op, proc(x: float32): float32 {.nimcall.} = 1'f32 / (1'f32 + exp(-x)))
        of OpSoftplus: un(op, proc(x: float32): float32 {.nimcall.} = ln(1'f32 + exp(x)))
        of OpRsqrt: un(op, proc(x: float32): float32 {.nimcall.} = 1'f32 / sqrt(x))
        of OpAddConst:
          let x = rr(op.b)
          if rs[op.b] == stKnown: setR(op.a, x + ffrom(op.imm), stKnown) else: setR(op.a, 0, stUnknown)
        of OpMulConst:
          let x = rr(op.b)
          if rs[op.b] == stKnown: setR(op.a, x * ffrom(op.imm), stKnown) else: setR(op.a, 0, stUnknown)
        of OpIfBegin:
          let c = rr(op.a)
          if rs[op.a] != stKnown: reject("data-dependent branch")
          if c == 0'f32: next = op.match + 1
        of OpIfEnd: discard
        of OpForBegin:
          let c = rr(op.a)
          if rs[op.a] != stKnown: reject("data-dependent loop count")
          let n = if c >= 1'f32 and c < 2147483647'f32: int(c) else: 0
          if n <= 0: next = op.match + 1
          else: loops.add (pc, n)
        of OpForEnd:
          var top = loops[^1]
          dec top[1]
          if top[1] > 0:
            loops[^1] = top
            next = top[0] + 1
          else:
            discard loops.pop()
        of OpTgLoadF32:
          needInit(op.b); setR(op.a, 0, stUnknown)
        of OpTgStoreF32:
          needInit(op.a); needInit(op.b)
        of OpLoadF32:
          raddr(op, op.imm, false); setR(op.a, 0, stUnknown)
        of OpStoreF32:
          needInit(op.a); raddr(op, op.imm, true)
        of OpLoadF32Idx:
          raddr(op, 0, false); setR(op.a, 0, stUnknown)
        of OpStoreF32Idx:
          needInit(op.a); raddr(op, 0, true)
        of OpLoadU8:
          needInit(op.b); setR(op.a, 0, stUnknown)
        of OpUTgid: setU(op.a, uint32(g), stKnown)
        of OpULid: setU(op.a, uint32(lane), stKnown)
        of OpUConst: setU(op.a, op.imm, stKnown)
        of OpUAdd, OpUMul, OpUShr:
          let x = ru(op.b); let y = ru(op.c)
          if us[op.b] == stKnown and us[op.c] == stKnown:
            let v = if op.op == OpUAdd: x + y
                    elif op.op == OpUMul: x * y
                    elif y < 32: x shr y
                    else: 0'u32
            setU(op.a, v, stKnown)
          else: setU(op.a, 0, stUnknown)
        of OpUAddK, OpUMulK, OpUAndK, OpUShrK, OpUShlK:
          let x = ru(op.b)
          if us[op.b] == stKnown:
            let k = op.imm
            let v = case op.op
              of OpUAddK: x + k
              of OpUMulK: x * k
              of OpUAndK: x and k
              of OpUShrK: (if k < 32: x shr k else: 0'u32)
              else: (if k < 32: x shl k else: 0'u32)
            setU(op.a, v, stKnown)
          else: setU(op.a, 0, stUnknown)
        of OpULoadU8, OpULoadU32, OpULoadMeta:
          needInitU(op.b); setU(op.a, 0, stUnknown)
        of OpULoadConstF, OpULoadIn:
          needInitU(op.b); setR(op.a, 0, stUnknown)
        of OpULoadIn4:
          needInitU(op.b)
          for k in 0'u32 .. 3'u32: setR(op.a + k, 0, stUnknown)
        of OpULoadF32:
          uaddr(op, false, true); setR(op.a, 0, stUnknown)
        of OpUStoreF32:
          needInit(op.a); uaddr(op, true, false)
        of OpUAtomicAdd:
          needInit(op.a); uaddr(op, true, true)
        of OpUToR:
          let x = ru(op.b)
          if us[op.b] == stKnown: setR(op.a, float32(x), stKnown) else: setR(op.a, 0, stUnknown)
        of OpRToU:
          let x = rr(op.b)
          if rs[op.b] == stKnown: setU(op.a, uint32(x), stKnown) else: setU(op.a, 0, stUnknown)
        else:
          reject("unsupported opcode 0x" & toHex(op.op, 2))
        pc = next
  result = fp

proc rootOf(roots: Table[string, string]; name: string): string =
  roots.getOrDefault(name, name)

proc bindingsOk(a, b: FuseKernel; roots: Table[string, string]) =
  ## One buffer per domain, and no aliasing between the written mem buffer and
  ## any other binding of the fused launch.
  var dom = initTable[string, string]()
  for (d, n) in a.bindings & b.bindings:
    if d in dom and dom[d] != n: reject("different buffers in domain " & d)
    dom[d] = n
  if "mem" in dom:
    let m = rootOf(roots, dom["mem"])
    for d, n in dom:
      if d != "mem" and rootOf(roots, n) == m: reject("mem buffer aliases " & d & " buffer")

proc bindingReason*(a, b: FuseKernel; roots: Table[string, string]): string =
  ## Empty when both kernels can share one set of domain bindings (a prerequisite
  ## of any single-dispatch merge, persistent or not).
  try:
    bindingsOk(a, b, roots)
    ""
  except Reject as e: e.msg

proc fusable*(a, b: FuseKernel; roots: Table[string, string]; plan: var FusePlan;
              budget = DefaultBudget): string =
  ## Empty when `A; barriertg; B` is exactly equivalent to the two dispatches,
  ## otherwise the reason it is not.
  try:
    if a.tg != b.tg: reject("different threadgroup size")
    if a.tg <= 0 or a.threads <= 0: reject("empty launch")
    bindingsOk(a, b, roots)
    var left = budget
    let pa = decode(a.words)
    let pb = decode(b.words)
    block regs:
      var used = 0
      for o in pa & pb:
        used = max(used, int(max(o.a, max(o.b, o.c))))
        if o.op == OpFma: used = max(used, int(o.imm and 0x1F'u32))
        if o.op == OpULoadIn4: used = max(used, int(o.a) + 3)
      if used + 2 > 31: reject("no free registers for the fused launch")
    let fa = footprintOf(pa, a.threads, a.tg, left)
    let fb = footprintOf(pb, b.threads, b.tg, left)
    plan = FusePlan(restoreR: fb.initR, restoreU: fb.initU)
    for key, ga in fa.acc:
      let gb = fb.acc.getOrDefault(key, -2'i32)
      if gb == -2: continue
      if key notin fa.wr and key notin fb.wr: continue
      if ga == -1 or gb == -1 or ga != gb: reject("cross-threadgroup data flow")
    return ""
  except Reject as e:
    return e.msg

proc pack(op: uint32; a = 0'u32; b = 0'u32; c = 0'u32): uint32 =
  op or (a shl 8) or (b shl 13) or (c shl 18)

proc maxReg(prog: seq[OpInfo]): int =
  for o in prog:
    result = max(result, int(max(o.a, max(o.b, o.c))))
    if o.op == OpFma: result = max(result, int(o.imm and 0x1F'u32))
    if o.op == OpULoadIn4: result = max(result, int(o.a) + 3)

proc groupsOf(k: FuseKernel): int = (k.threads + k.tg - 1) div k.tg

proc fusedThreads*(a, b: FuseKernel): int = max(groupsOf(a), groupsOf(b)) * a.tg

proc concatWords*(a, b: FuseKernel; plan: FusePlan): seq[uint32] =
  ## A, a threadgroup barrier, then B in one launch of the larger grid. A kernel
  ## with fewer threadgroups runs under a uniform `if tgid < groups` guard, so
  ## every thread still reaches the barrier. Registers B reads at their launch
  ## value (R0 = thread id, everything else zero) are reset after the barrier.
  let total = max(groupsOf(a), groupsOf(b))
  let used = max(maxReg(decode(a.words)), maxReg(decode(b.words)))
  let x = uint32(used + 1)
  let y = uint32(used + 2)
  proc restore(k: FuseKernel; plan: FusePlan): seq[uint32] =
    for r in 0'u32 .. 31'u32:
      if (plan.restoreU and (1'u64 shl r)) != 0: result.add [pack(OpUConst, r), 0'u32]
      if (plan.restoreR and (1'u64 shl r)) != 0:
        if r == 0:
          result.add pack(OpUTgid, x)
          result.add [pack(OpUMulK, x, x), uint32(k.tg)]
          result.add pack(OpULid, y)
          result.add pack(OpUAdd, x, x, y)
          result.add pack(OpUToR, 0, x)
        else:
          result.add [pack(OpLoadConst, r), 0'u32]
  proc body(k: FuseKernel; plan: FusePlan): seq[uint32] =
    let inner = restore(k, plan) & k.words[0 ..< k.words.high]
    if groupsOf(k) == total: return inner
    result = @[pack(OpLoadTgid, x), pack(OpLoadConst, y), fbits(float32(groupsOf(k))),
      pack(OpCmpLt, x, x, y), pack(OpIfBegin, x)]
    result.add inner
    result.add pack(OpIfEnd, 0)
  result = body(a, FusePlan())
  result.add OpBarrierTg
  result.add body(b, plan)
  result.add OpHalt

proc record*(reason: string) =
  inc fuseStats.pairs
  if reason.len == 0: inc fuseStats.fused
  else: fuseStats.reasons.inc reason

proc idempotent*(k: FuseKernel; budget = DefaultBudget): bool =
  ## True when running any threadgroup of the kernel twice gives the same memory: no address is
  ## both read and written and nothing is accumulated atomically. Proven by the same exact
  ## interpretation as the fusion check; the input domain (an alias of mem) must stay unused.
  try:
    let prog = decode(k.words)
    for o in prog:
      if o.op == OpULoadIn or o.op == OpULoadIn4 or o.op == OpUAtomicAdd: return false
    var left = budget
    let fp = footprintOf(prog, k.threads, k.tg, left)
    for key in fp.wr:
      if key in fp.rd: return false
    return true
  except Reject:
    return false

# ── Persistent chains ─────────────────────────────────────────────────────────
# One launch of `ntg` physical threadgroups (256 lanes) runs a whole chain of dispatches. Each stage
# iterates its own virtual threadgroups (sub-groups of its own tg size) and stages are separated by the
# software grid barrier, so any dependency between stages is honoured. It needs every physical
# threadgroup resident at once, so `ntg` is a lift-time setting (grid_barrier_probe finds the limit).

const PhysicalTg* = 256

type
  PersistentStage* = object
    kernel*: FuseKernel
    rebase*: seq[(string, int)]        ## (domain, input index) the stage addresses

  Persistent* = object
    lines*: seq[string]                ## kernel body: `words ...` and `rebase DOMAIN inK` lines
    tgBytes*: int

proc hexLine(words: seq[uint32]): string =
  result = "words"
  for w in words: result.add " " & toHex(w, 8).toLowerAscii

proc persistentChain*(stages: seq[PersistentStage]; ntg, ctrl: int): (Persistent, string) =
  ## The chain program, or the reason it cannot be built.
  try:
    if stages.len < 2: reject("a chain needs two stages")
    var words: seq[uint32]
    var lines: seq[string]
    var tgBytes = 16
    proc flush() =
      if words.len > 0:
        lines.add hexLine(words)
        words.setLen(0)
    for i, stage in stages:
      let k = stage.kernel
      if k.tg <= 0 or k.tg > PhysicalTg or PhysicalTg mod k.tg != 0:
        reject("threadgroup size " & $k.tg & " does not divide " & $PhysicalTg)
      let prog = decode(k.words)
      var hasBarrier = false
      for o in prog:
        if o.op == OpBarrierTg: hasBarrier = true
      let nVirt = groupsOf(k)
      let subs = PhysicalTg div k.tg
      if hasBarrier and nVirt mod (subs * ntg) != 0 and not idempotent(k):
        reject("stage with a threadgroup barrier needs a virtual group count divisible by " & $(subs * ntg))
      let stride = (k.tgBytes + 3) div 4
      if stride >= (1 shl 20): reject("threadgroup memory too large")
      tgBytes = max(tgBytes, subs * stride * 4)
      if tgBytes > 32768: reject("threadgroup memory above 32 KiB")
      let used = maxReg(prog)
      if not hasBarrier and used + 1 > 31: reject("no free register for the activity guard")
      flush()
      for (domain, index) in stage.rebase:
        lines.add "rebase " & domain & " in" & $index
      words.add pack(OpVtgBegin, 1)
      words.add uint32(nVirt)
      words.add (uint32(k.tg) shl 20) or uint32(stride)
      if not hasBarrier:
        let x = uint32(used + 1)
        words.add pack(OpLoadVact, x)
        words.add pack(OpIfBegin, x)
      words.add k.words[0 ..< k.words.high]
      if not hasBarrier: words.add pack(OpIfEnd, 0)
      words.add pack(OpVtgEnd, 0)
      if i < stages.high:
        words.add pack(OpGridBarrier)
        words.add uint32(ctrl)
    words.add pack(OpGridEpilogue)
    words.add uint32(ctrl)
    words.add OpHalt
    flush()
    return (Persistent(lines: lines, tgBytes: tgBytes), "")
  except Reject as e:
    return (Persistent(), e.msg)
