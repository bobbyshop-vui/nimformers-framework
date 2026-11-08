## Dispatch fusion: the analysis verdicts and, for every accepted pair, bit-exact
## equivalence of the fused program with the two sequential dispatches on the CPU VM.
import std/[tables, strutils]
import session_vm
import backends/cpu/cpu_backend as cpu
import ./fuse

const Tg = 4
const Groups = 8
const N = Tg * Groups

type Body = proc(p: var VmProgram)

proc build(body: Body): seq[uint32] =
  var p: VmProgram
  body(p)
  p.emitHalt()
  p.words

proc kern(words: seq[uint32]; threads = N; tg = Tg; mem = "buf"): FuseKernel =
  FuseKernel(words: words, threads: threads, tg: tg, tgBytes: 0, bindings: @[("mem", mem)])

proc idx(p: var VmProgram) =               # U3 = tgid*Tg + lid
  p.emitUTgid(0); p.emitUMulK(1, 0, Tg.uint32); p.emitULid(2); p.emitUAdd(3, 1, 2)

proc scale(src, dst: uint32; k: float32): Body =
  result = proc(p: var VmProgram) =
    idx(p); p.emitULoadF32(0, 3, src); p.emitMulConst(1, 0, k); p.emitUStoreF32(1, 3, dst)

proc shifted(src, dst: uint32): Body =    # reads the neighbouring threadgroup's element
  result = proc(p: var VmProgram) =
    idx(p); p.emitUAddK(4, 3, Tg.uint32); p.emitULoadF32(0, 4, src); p.emitUStoreF32(0, 3, dst)

proc run(words: seq[uint32]; mem: var seq[float32]; threads = N) =
  if not cpu.vmRun(words, mem, newSeq[float32](), threads, 4, Tg):
    raise newException(ValueError, "CPU VM rejected a program")

proc fresh(): seq[float32] =
  result = newSeq[float32](4 * N + Tg)
  for i in 0 ..< result.len: result[i] = float32(i mod 17) * 0.5'f32 + 1

proc expectFuse(name: string; a, b: Body; threadsB = N; threadsA = N) =
  let ka = kern(build(a), threads = threadsA); let kb = kern(build(b), threads = threadsB)
  var plan: FusePlan
  let why = fusable(ka, kb, initTable[string, string](), plan)
  if why.len > 0: raise newException(ValueError, name & ": expected fusion, got: " & why)
  var seqMem = fresh()
  run(ka.words, seqMem, threadsA); run(kb.words, seqMem, threadsB)
  var fusedMem = fresh()
  run(concatWords(ka, kb, plan), fusedMem, fusedThreads(ka, kb))
  if seqMem != fusedMem: raise newException(ValueError, name & ": fused result differs from the dispatches")
  echo "PASS fuse ", name

proc expectReject(name, reason: string; ka, kb: FuseKernel;
                  roots = initTable[string, string]()) =
  var plan: FusePlan
  let why = fusable(ka, kb, roots, plan)
  if reason notin why:
    raise newException(ValueError, name & ": expected rejection '" & reason & "', got '" & why & "'")
  echo "PASS fuse reject ", name

proc main() =
  expectFuse("producer-consumer in one threadgroup", scale(0, N.uint32, 2), scale(N.uint32, 2 * N.uint32, 3))
  expectFuse("independent buffers", scale(0, N.uint32, 2), scale(0, 2 * N.uint32, 5))
  expectFuse("loop with known count", proc(p: var VmProgram) =
      idx(p); p.emitLoadConstF32(5, 3.0); p.emitLoadConstF32(6, 0.0)
      p.emitForBegin(5); p.emitAddConst(6, 6, 2.0); p.emitForEnd()
      p.emitUStoreF32(6, 3, N.uint32),
    scale(N.uint32, 2 * N.uint32, 0.5))
  expectFuse("guarded store", proc(p: var VmProgram) =
      idx(p); p.emitUToR(7, 2); p.emitLoadConstF32(8, 2.0); p.emitCmpLt(9, 7, 8)
      p.emitIfBegin(9); p.emitLoadConstF32(0, 9.0); p.emitUStoreF32(0, 3, N.uint32); p.emitIfEnd(),
    scale(N.uint32, 2 * N.uint32, 1.5))
  expectFuse("fewer threadgroups in B", scale(0, N.uint32, 2), scale(N.uint32, 2 * N.uint32, 3), N div 2)
  expectFuse("fewer threadgroups in A", scale(0, N.uint32, 2), scale(N.uint32, 2 * N.uint32, 3), N, N div 2)
  let a = kern(build(scale(0, N.uint32, 2)))
  expectReject("neighbouring threadgroup", "cross-threadgroup", a,
    kern(build(shifted(N.uint32, 2 * N.uint32))))
  expectReject("write-after-read across groups", "cross-threadgroup", kern(build(shifted(0, N.uint32))),
    kern(build(scale(2 * N.uint32, 0, 2))))
  var other = a; other.tg = 8
  expectReject("threadgroup size", "size", a, other)
  var diffBuf = a; diffBuf.bindings = @[("mem", "other")]
  expectReject("buffer binding", "different buffers", a, diffBuf)
  var aliased = a; aliased.bindings = @[("mem", "buf"), ("input", "view")]
  expectReject("alias", "aliases", a, aliased, {"view": "buf"}.toTable)
  expectFuse("reads launch-value registers", scale(0, N.uint32, 2), proc(p: var VmProgram) =
      idx(p); p.emitAddConst(5, 0, 1.0); p.emitAddConst(0, 6, 2.0); p.emitUStoreF32(5, 3, 3 * N.uint32))
  expectReject("data-dependent address", "data-dependent", a, kern(build(proc(p: var VmProgram) =
    idx(p); p.emitULoadU32(5, 3, 0); p.emitULoadF32(0, 5, 0); p.emitUStoreF32(0, 3, 3 * N.uint32))))
  expectReject("data-dependent loop", "data-dependent", a, kern(build(proc(p: var VmProgram) =
    idx(p); p.emitULoadF32(5, 3, 0); p.emitForBegin(5); p.emitForEnd())))
  expectReject("persistent opcode", "persistent", a, kern(build(proc(p: var VmProgram) =
    p.emitGridBarrier(0))))
  var plan: FusePlan
  if "budget" notin fusable(a, a, initTable[string, string](), plan, 10):
    raise newException(ValueError, "budget exhaustion must reject")
  echo "PASS fuse reject budget"
  # Idempotence (a threadgroup may be run twice): plain copy yes, in-place update no.
  if not idempotent(kern(build(scale(0, N.uint32, 2)))): raise newException(ValueError, "copy must be idempotent")
  if idempotent(kern(build(scale(0, 0, 2)))): raise newException(ValueError, "in-place scale must not be idempotent")
  echo "PASS fuse idempotent"
main()
