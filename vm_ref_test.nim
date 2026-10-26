## vm_ref_test.nim — kiểm chứng kernel bytecode TRÊN CPU (không cần Metal).
##
## Interpreter ở đây mô phỏng ĐÚNG ngữ nghĩa của kernel do
## jit_vm.lowerBytecodeToMetal sinh ra (kể cả chi tiết: FOR đọc lại R[count] mỗi
## vòng; barrier chỉ hợp lệ ở mức top-level; U[] là uint32 wrap-around).
## Chạy: nim c -r vm_ref_test.nim
import std/[math, random, strformat]
import session_vm, fp4_gemv_vm

type
  Thread = object
    R: array[16, float32]
    U: array[16, uint32]
    pc: int
    stack: seq[tuple[start: int, i: int, reg: int]]
    done: bool

  Env = object
    memF32: seq[float32]
    memU8: seq[uint8]
    constF32: seq[float32]
    inF32: seq[float32]
    tg: seq[float32]

proc matchEnd(w: seq[uint32]; startPc: int; isFor: bool): int =
  ## trả về pc của FOR_END/IF_END khớp với block bắt đầu ở startPc
  var depth = 1
  var pc = startPc + 1
  while pc < w.len:
    let op = w[pc] and 0xFF'u32
    if op == OP_FOR_BEGIN or op == OP_IF_BEGIN: inc depth
    elif op == OP_FOR_END or op == OP_IF_END:
      dec depth
      if depth == 0: return pc
    if opHasWord(op): inc pc
    inc pc
  raise newException(ValueError, "block khong dong")

proc runThread(t: var Thread; w: seq[uint32]; env: var Env;
               tid, lid, tgid: int; atTopBarrier: var bool) =
  ## chạy tới barrier top-level kế tiếp (atTopBarrier=true) hoặc hết chương trình
  atTopBarrier = false
  template A(x: uint32): int = int((x shr 8) and 0xF)
  template B(x: uint32): int = int((x shr 12) and 0xF)
  template C(x: uint32): int = int((x shr 16) and 0xF)
  template nextWord(): uint32 =
    let v = w[t.pc]
    inc t.pc
    v
  while t.pc < w.len and not t.done:
    let x = w[t.pc]
    inc t.pc
    let op = x and 0xFF'u32
    let a = A(x); let b = B(x); let c = C(x)
    case op
    of OP_NOP: discard
    of OP_HALT: t.done = true; return
    of OP_LOAD_CONST: t.R[a] = cast[float32](nextWord())
    of OP_ADD: t.R[a] = t.R[b] + t.R[c]
    of OP_SUB: t.R[a] = t.R[b] - t.R[c]
    of OP_MUL: t.R[a] = t.R[b] * t.R[c]
    of OP_DIV: t.R[a] = t.R[b] / t.R[c]
    of OP_FMA:
      let rc = int(nextWord() and 0xF)
      t.R[a] = t.R[b] * t.R[c] + t.R[rc]
    of OP_EXP: t.R[a] = exp(t.R[b])
    of OP_ADD_CONST: t.R[a] = t.R[b] + cast[float32](nextWord())
    of OP_MUL_CONST: t.R[a] = t.R[b] * cast[float32](nextWord())
    of OP_CMP_LT: t.R[a] = (if t.R[b] < t.R[c]: 1'f32 else: 0'f32)
    of OP_TG_LOAD_F32:
      let off = int(nextWord())
      t.R[a] = env.tg[int(t.R[b]) + off]
    of OP_TG_STORE_F32:
      let off = int(nextWord())
      env.tg[int(t.R[b]) + off] = t.R[a]
    of OP_BARRIER_TG:
      # Barrier HOP LE o top-level cua FOR: so vong lap la hang so nen MOI lane
      # deu di qua barrier cung so lan -> khong phan ky (dung nhu emitTgReduce32
      # phat ra). Nguoc lai, barrier trong IF la PHAN KY (mot so lane nhay qua
      # body) -> deadlock tren GPU that. t.stack chua ca FOR (reg >= 0) lan IF
      # (reg = -1), nen chi can chan khi con bat ky IF nao dang mo.
      for s in t.stack:
        if s.reg < 0:
          raise newException(ValueError, "barrier trong IF - khong hop le")
      atTopBarrier = true
      return
    of OP_IF_BEGIN:
      if t.R[a] == 0'f32:
        t.pc = matchEnd(w, t.pc - 1, false) + 1
      else:
        t.stack.add (start: t.pc - 1, i: -1, reg: -1)   # đánh dấu IF (để check barrier)
    of OP_IF_END:
      discard t.stack.pop()
    of OP_FOR_BEGIN:
      if not (0 < int(t.R[a])):
        t.pc = matchEnd(w, t.pc - 1, true) + 1
      else:
        t.stack.add (start: t.pc, i: 0, reg: a)
    of OP_FOR_END:
      var fr = t.stack[^1]
      inc fr.i
      if fr.i < int(t.R[fr.reg]):            # ĐỌC LẠI R[count] như kernel đã lower
        t.stack[^1] = fr
        t.pc = fr.start
      else:
        discard t.stack.pop()
    of OP_U_TGID: t.U[a] = uint32(tgid)
    of OP_U_LID: t.U[a] = uint32(lid)
    of OP_U_CONST: t.U[a] = nextWord()
    of OP_U_ADD: t.U[a] = t.U[b] + t.U[c]
    of OP_U_MUL: t.U[a] = t.U[b] * t.U[c]
    of OP_U_SHR: t.U[a] = t.U[b] shr t.U[c]
    of OP_U_ADDK: t.U[a] = t.U[b] + nextWord()
    of OP_U_MULK: t.U[a] = t.U[b] * nextWord()
    of OP_U_ANDK: t.U[a] = t.U[b] and nextWord()
    of OP_U_SHRK: t.U[a] = t.U[b] shr nextWord()
    of OP_U_SHLK: t.U[a] = t.U[b] shl nextWord()
    of OP_U_LOAD_U8: t.U[a] = uint32(env.memU8[int(t.U[b] + nextWord())])
    of OP_U_LOAD_CONSTF: t.R[a] = env.constF32[int(t.U[b] + nextWord())]
    of OP_U_LOAD_IN: t.R[a] = env.inF32[int(t.U[b] + nextWord())]
    of OP_U_STORE_F32: env.memF32[int(t.U[b] + nextWord())] = t.R[a]
    of OP_U_LOAD_F32: t.R[a] = env.memF32[int(t.U[b] + nextWord())]
    of OP_U_TO_R: t.R[a] = float32(t.U[b])
    of OP_R_TO_U: t.U[a] = uint32(t.R[b])
    else:
      raise newException(ValueError, "op chua mo phong: 0x" & $op)
  t.done = true

proc dispatch(w: seq[uint32]; env: var Env; nThreads, tgSize, tgFloats: int) =
  ## mô phỏng dispatchThreads: threadgroup chạy tuần tự, các lane đồng bộ tại barrier
  let nTg = (nThreads + tgSize - 1) div tgSize
  for g in 0 ..< nTg:
    env.tg = newSeq[float32](tgFloats)
    for i in 0 ..< tgFloats: env.tg[i] = float32.high     # bắt bug đọc tg chưa ghi
    var ths = newSeq[Thread](tgSize)
    for l in 0 ..< tgSize:
      ths[l].R[0] = float32(g * tgSize + l)
      if g * tgSize + l >= nThreads: ths[l].done = true   # dispatchThreads: tg cuoi khong deu
    while true:
      var anyBarrier = false
      var allDone = true
      for l in 0 ..< tgSize:
        if ths[l].done: continue
        var atB = false
        runThread(ths[l], w, env, g * tgSize + l, l, g, atB)
        if atB: anyBarrier = true
        if not ths[l].done: allDone = false
      if allDone and not anyBarrier: break
      if not anyBarrier: break

proc relErr(a, b: float32): float32 = abs(a - b) / max(1e-6'f32, abs(b))

var okAll = true
proc check(name: string; got, want: seq[float32]; tol = 1e-4'f32) =
  var worst = 0'f32
  for i in 0 ..< want.len:
    worst = max(worst, relErr(got[i], want[i]))
  let ok = worst <= tol
  if not ok: okAll = false
  echo &"[{(if ok: \"PASS\" else: \"FAIL\")}] {name}: max rel err = {worst:.3e} ({want.len} phan tu)"

randomize(7)
proc rnd(): float32 = float32(rand(2.0) - 1.0)

# ── Test 1: FP4-KM gemv ─────────────────────────────────────────
block:
  let K = 256; let N = 7; let G = 32; let nG = K div G
  var lut = newSeq[float32](272)
  for i in 0 ..< 272: lut[i] = rnd()
  var w = newSeq[uint8](N * (K div 2) + N * nG)
  var wDeq = newSeq[float32](N * K)
  for n in 0 ..< N:
    for j in 0 ..< nG:
      let sc = uint8(rand(255))
      w[N * (K div 2) + n * nG + j] = sc
      for t in 0 ..< G:
        let nib = uint8(rand(15))
        let byteIdx = n * (K div 2) + j * 16 + (t and 15)
        if t < 16: w[byteIdx] = (w[byteIdx] and 0xF0'u8) or nib
        else: w[byteIdx] = (w[byteIdx] and 0x0F'u8) or (nib shl 4)
        wDeq[n * K + j * G + t] = lut[int(nib)] * lut[16 + int(sc)]
  var x = newSeq[float32](K)
  for i in 0 ..< K: x[i] = rnd()
  var want = newSeq[float32](N)
  for n in 0 ..< N:
    var s = 0.0'f64
    for k in 0 ..< K: s += float64(wDeq[n * K + k]) * float64(x[k])
    want[n] = float32(s)
  let prog = buildFp4GemvProgram(K, N, true)
  var env = Env(memF32: newSeq[float32](N), memU8: w, constF32: lut, inF32: x)
  dispatch(prog.words, env, N * GemvTg, GemvTg, fp4GemvTgBytes(true) div 4)
  check("fp4-KM gemv K=256 N=7", env.memF32, want)

# ── Test 2: FP4 cũ (1 byte / phần tử) ───────────────────────────
block:
  let K = 96; let N = 5
  var lut = newSeq[float32](256)
  for i in 0 ..< 256: lut[i] = rnd()
  var w = newSeq[uint8](N * K)
  for i in 0 ..< w.len: w[i] = uint8(rand(255))
  var x = newSeq[float32](K)
  for i in 0 ..< K: x[i] = rnd()
  var want = newSeq[float32](N)
  for n in 0 ..< N:
    var s = 0.0'f64
    for k in 0 ..< K: s += float64(lut[int(w[n * K + k])]) * float64(x[k])
    want[n] = float32(s)
  let prog = buildFp4GemvProgram(K, N, false)
  var env = Env(memF32: newSeq[float32](N), memU8: w, constF32: lut, inF32: x)
  dispatch(prog.words, env, N * GemvTg, GemvTg, fp4GemvTgBytes(false) div 4)
  check("fp4-legacy gemv K=96 N=5", env.memF32, want)

# ── Test 3: FP32 gemv (LoRA) có tail + scale + accumulate ───────
block:
  for (K, acc) in [(50, false), (50, true), (128, true), (8, false)]:
    let N = 6; let scale = 0.25'f32
    var W = newSeq[float32](N * K)
    for i in 0 ..< W.len: W[i] = rnd()
    var x = newSeq[float32](K)
    for i in 0 ..< K: x[i] = rnd()
    var init = newSeq[float32](N)
    for i in 0 ..< N: init[i] = (if acc: rnd() else: 0'f32)
    var want = newSeq[float32](N)
    for n in 0 ..< N:
      var s = 0.0'f64
      for k in 0 ..< K: s += float64(W[n * K + k]) * float64(x[k])
      want[n] = float32(s) * scale + init[n]
    let prog = buildF32GemvProgram(K, N, scale, acc)
    var env = Env(memF32: init, constF32: W, inF32: x)
    dispatch(prog.words, env, N * GemvTg, GemvTg, f32GemvTgBytes() div 4)
    check(&"f32 gemv (LoRA) K={K} acc={acc}", env.memF32, want)

# ── Test 3b: FP32 gemv rank nhỏ (LoRA B): 1 thread/hàng, KHÔNG reduce ──
# Đây là kernel dùng cho B của LoRA khi K = rank <= 32. Không dùng TG memory
# và không tree-reduce, nên so sánh trực tiếp với oracle f64.
block:
  for K in [1, 4, 16, 32]:
    let N = 7; let scale = 0.5'f32
    var W = newSeq[float32](N * K)
    for i in 0 ..< W.len: W[i] = rnd()
    var x = newSeq[float32](K)
    for i in 0 ..< K: x[i] = rnd()
    var init = newSeq[float32](N)
    for i in 0 ..< N: init[i] = rnd()
    var want = newSeq[float32](N)
    for n in 0 ..< N:
      var s = 0.0'f64
      for k in 0 ..< K: s += float64(W[n * K + k]) * float64(x[k])
      want[n] = float32(s) * scale + init[n]
    let prog = buildF32GemvSmallKProgram(K, N, scale, true)
    var env = Env(memF32: init, constF32: W, inF32: x)
    # N thread = N hàng; tgSize bất kỳ vì kernel không chạm TG memory.
    dispatch(prog.words, env, N, 64, f32GemvSmallKTgBytes() div 4)
    check(&"f32 gemv smallK K={K} N={N}", env.memF32, want)

# ── Test 4: bias + APFLU ────────────────────────────────────────
block:
  let n = 100
  var mem = newSeq[float32](n)
  var bias = newSeq[float32](n)
  var want = newSeq[float32](n)
  for i in 0 ..< n:
    mem[i] = rnd() * 4; bias[i] = rnd()
    let v = mem[i] + bias[i]
    want[i] = if v > 0: v * (1'f32 + 0.1'f32 * v) else: 1'f32 * v * exp(v)
  let prog = buildBiasActProgram(true)
  var env = Env(memF32: mem, constF32: bias)
  dispatch(prog.words, env, n, 32, 1)
  check("bias+apflu n=100", env.memF32, want)

# ── Test 5: Linear++ step, 4 token liên tiếp (kiểm state) ───────
block:
  let C = 64; let H = 4; let D = C div H
  let lay = lpLayout(C, H)
  var mem = newSeq[float32](lay.total)
  # CPU reference (sao chép đúng công thức forwardLinearPlusInferStep)
  var sRef = newSeq[float32](H * D * D)
  var zRef = newSeq[float32](H * D)
  proc phi(x: float32): float32 = (if x > 0: x + 1 else: exp(x))
  let prog = buildLinearPlusStepProgram(lay, 1e-6'f32)
  var worst = 0'f32
  for step in 0 ..< 4:
    var proj = newSeq[float32](3 * C)
    for i in 0 ..< proj.len: proj[i] = rnd() * 2
    for i in 0 ..< proj.len: mem[lay.offProj + i] = proj[i]
    var yRef = newSeq[float32](C)
    for h in 0 ..< H:
      let base = h * D
      var kf = newSeq[float32](D); var qf = newSeq[float32](D)
      for d in 0 ..< D:
        qf[d] = phi(proj[base + d]); kf[d] = phi(proj[C + base + d])
        zRef[base + d] += kf[d]
      for kd in 0 ..< D:
        for vd in 0 ..< D:
          sRef[h * D * D + kd * D + vd] += kf[kd] * proj[2 * C + base + vd]
      var den = 0'f32
      for kd in 0 ..< D: den += qf[kd] * zRef[base + kd]
      for vd in 0 ..< D:
        var num = 0'f32
        for kd in 0 ..< D: num += qf[kd] * sRef[h * D * D + kd * D + vd]
        yRef[base + vd] = num / (den + 1e-6'f32)
    var env = Env(memF32: mem)
    dispatch(prog.words, env, H * D, D, 1)
    mem = env.memF32
    for i in 0 ..< C: worst = max(worst, relErr(mem[lay.offY + i], yRef[i]))
  let ok = worst < 1e-3'f32
  if not ok: okAll = false
  echo &"[{(if ok: \"PASS\" else: \"FAIL\")}] Linear++ step 4 token: max rel err = {worst:.3e}"


# ── Test 7: INT4 asymmetric gemv (per-row, per-group, K khong luy thua 2) ──
block:
  for (K, G) in [(256, 0), (256, 128), (192, 0), (512, 64)]:
    let N = 5
    let nG = if G == 0: 1 else: K div G
    var scale = newSeq[float32](N * nG)
    var zp = newSeq[float32](N * nG)
    for i in 0 ..< scale.len:
      scale[i] = rnd() * 0.2 + 0.3
      zp[i] = float32(rand(15))
    var data = newSeq[uint8](N * (K div 2))
    var wDeq = newSeq[float32](N * K)
    for n in 0 ..< N:
      for c in 0 ..< K:
        let nib = uint8(rand(15))
        let bi = n * (K div 2) + c div 2
        if c mod 2 == 0: data[bi] = (data[bi] and 0xF0'u8) or nib
        else: data[bi] = (data[bi] and 0x0F'u8) or (nib shl 4)
        let g = if G == 0: 0 else: c div G
        wDeq[n * K + c] = (float32(nib) - zp[n * nG + g]) * scale[n * nG + g]
    var x = newSeq[float32](K)
    for i in 0 ..< K: x[i] = rnd()
    var want = newSeq[float32](N)
    for n in 0 ..< N:
      var sacc = 0.0'f64
      for k in 0 ..< K: sacc += float64(wDeq[n * K + k]) * float64(x[k])
      want[n] = float32(sacc)
    let prog = buildInt4AsymGemvProgram(K, N, G)
    var env = Env(memF32: newSeq[float32](N), memU8: data, constF32: scale & zp, inF32: x)
    dispatch(prog.words, env, N * GemvTg, GemvTg, int4GemvTgBytes() div 4)
    check(&"int4-asym gemv K={K} G={G}", env.memF32, want, 2e-4'f32)

# ── Test 6: địa chỉ > 2^24 (bug float-index cũ) ─────────────────
block:
  let K = 4096; let N = 8192      # N*K/2 = 16.7M -> scale ptr bắt đầu đúng ngưỡng 2^24
  let nG = K div 32
  var lut = newSeq[float32](272)
  for i in 0 ..< 272: lut[i] = (if i < 16: float32(i) else: 0'f32)
  lut[16 + 1] = 1'f32                       # scale byte 1 -> 1.0
  var w = newSeq[uint8](N * (K div 2) + N * nG)
  for i in N * (K div 2) ..< w.len: w[i] = 1'u8
  # chỉ đặt weight cho cột n = N-1: nibble = 1 cho mọi phần tử -> weight = 1.0
  let n = N - 1
  for j in 0 ..< nG:
    for b in 0 ..< 16: w[n * (K div 2) + j * 16 + b] = 0x11'u8
  var x = newSeq[float32](K)
  for i in 0 ..< K: x[i] = 1'f32
  let prog = buildFp4GemvProgram(K, N, true)
  var env = Env(memF32: newSeq[float32](N), memU8: w, constF32: lut, inF32: x)
  # chỉ mô phỏng threadgroup cuối (n = N-1) để test nhanh
  var sub = prog
  # dispatch 1 threadgroup nhưng ép tgid = N-1 bằng cách chạy N threadgroup rỗng là quá chậm -> chạy tay:
  env.tg = newSeq[float32](fp4GemvTgBytes(true) div 4)
  var ths = newSeq[Thread](GemvTg)
  var done = false
  while not done:
    var anyB = false
    var allDone = true
    for l in 0 ..< GemvTg:
      if ths[l].done: continue
      var atB = false
      runThread(ths[l], sub.words, env, n * GemvTg + l, l, n, atB)
      if atB: anyB = true
      if not ths[l].done: allDone = false
    done = (allDone and not anyB) or not anyB
  let got = env.memF32[n]
  let ok = abs(got - float32(K)) < 1e-2'f32
  if not ok: okAll = false
  echo &"[{(if ok: \"PASS\" else: \"FAIL\")}] dia chi > 2^24 (cot N-1): got={got} want={K}"

if okAll: echo "TAT CA TEST PASS" else:
  echo "CO TEST FAIL"
  quit(1)
