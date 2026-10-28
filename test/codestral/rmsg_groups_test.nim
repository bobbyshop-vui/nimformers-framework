## vm_ref_test.nim — kiểm chứng kernel bytecode TRÊN CPU (không cần Metal).
##
## Interpreter ở đây mô phỏng ĐÚNG ngữ nghĩa của kernel do
## jit_vm.lowerBytecodeToMetal sinh ra (kể cả chi tiết: FOR đọc lại R[count] mỗi
## vòng; barrier chỉ hợp lệ ở mức top-level; U[] là uint32 wrap-around).
## Chạy: nim c -r vm_ref_test.nim
import std/[math, random, strformat]
import session_vm, fp4_gemv_vm, mamba2_vm_kernels, mamba2_full_vm, quant, fp4_gemv_gpu

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
      # A threadgroup barrier inside a uniform-count GPU loop is valid (the
      # tree reduction uses exactly this pattern). It is unsafe only under a
      # per-lane IF, where some lanes could skip the barrier.
      for frame in t.stack:
        if frame.reg < 0:
          raise newException(ValueError, "barrier trong IF khong dong nhat")
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
    of OP_LOAD_LID: t.R[a] = float32(lid)
    of OP_LOAD_TGID: t.R[a] = float32(tgid)
    of OP_NEG: t.R[a] = -t.R[b]
    of OP_RSQRT: t.R[a] = 1'f32 / sqrt(t.R[b])
    of OP_LOAD_F32_IDX: t.R[a] = env.memF32[int(t.R[b])]
    of OP_STORE_F32_IDX: env.memF32[int(t.R[b])] = t.R[a]
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
    of OP_U_LOAD_U32:
      let off = int(t.U[b] + nextWord())
      t.U[a] = uint32(env.memU8[off]) or (uint32(env.memU8[off + 1]) shl 8) or
        (uint32(env.memU8[off + 2]) shl 16) or (uint32(env.memU8[off + 3]) shl 24)
    of OP_U_LOAD_CONSTF: t.R[a] = env.constF32[int(t.U[b] + nextWord())]
    of OP_U_LOAD_IN: t.R[a] = env.inF32[int(t.U[b] + nextWord())]
    of OP_U_STORE_F32: env.memF32[int(t.U[b] + nextWord())] = t.R[a]
    of OP_U_LOAD_F32: t.R[a] = env.memF32[int(t.U[b] + nextWord())]
    of OP_U_ATOMIC_ADD:
      let off = int(nextWord())
      env.memF32[int(t.U[b]) + off] += t.R[a]
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


proc silu(z: float32): float32 = z / (1'f32 + exp(-z))

proc refGatedNorm(y, z, w: seq[float32]; C, nGroups: int; eps: float32): seq[float32] =
  result = newSeq[float32](C)
  let L = C div nGroups
  for g in 0 ..< nGroups:
    var ss = 0.0'f64
    for i in g * L ..< (g + 1) * L:
      let gv = y[i] * silu(z[i])
      ss += float64(gv) * float64(gv)
    let r = 1.0 / sqrt(ss / float64(L) + float64(eps))
    for i in g * L ..< (g + 1) * L:
      result[i] = float32(float64(y[i] * silu(z[i])) * r) * w[i]

proc lpPhiRef(x: float32): float32 =
  if x > 0'f32: x + 1'f32 else: exp(x)

proc lpStepRef(qkv: seq[float32]; C, H, D: int; eps: float32;
               stateZ, stateS: var seq[float32]): seq[float32] =
  result = newSeq[float32](C)
  for h in 0 ..< H:
    let base = h * D
    let sBase = h * D * D
    var q = newSeq[float32](D)
    var k = newSeq[float32](D)
    for d in 0 ..< D:
      q[d] = lpPhiRef(qkv[base + d])
      k[d] = lpPhiRef(qkv[C + base + d])
      stateZ[base + d] += k[d]
    for dk in 0 ..< D:
      for dv in 0 ..< D:
        stateS[sBase + dk * D + dv] += k[dk] * qkv[2 * C + base + dv]
    var den = 0'f32
    for dk in 0 ..< D: den += q[dk] * stateZ[base + dk]
    for dv in 0 ..< D:
      var num = 0'f32
      for dk in 0 ..< D: num += q[dk] * stateS[sBase + dk * D + dv]
      result[base + dv] = num / (den + eps)

proc runLinearPlusStepCheck() =
  let C = 16; let H = 4; let D = C div H; let eps = 1e-6'f32
  let lay = lpLayout(C, H)
  let prog = buildLinearPlusStepProgram(lay, eps)
  var env: Env
  env.memF32 = newSeq[float32](lay.total)
  var stateZ = newSeq[float32](C)
  var stateS = newSeq[float32](H * D * D)
  var worst = 0'f32
  for step in 0 ..< 4:
    var qkv = newSeq[float32](3 * C)
    for i in 0 ..< qkv.len:
      qkv[i] = float32(sin(float64((step + 1) * (i + 3)) * 0.113) * 0.9)
      env.memF32[lay.offProj + i] = qkv[i]
    let want = lpStepRef(qkv, C, H, D, eps, stateZ, stateS)
    dispatch(prog.words, env, H * D, D, 4)
    for i in 0 ..< C:
      let got = env.memF32[lay.offY + i]
      worst = max(worst, abs(got - want[i]) / max(1'f32, abs(want[i])))
    for i in 0 ..< stateZ.len:
      worst = max(worst, abs(env.memF32[lay.offZ + i] - stateZ[i]) / max(1'f32, abs(stateZ[i])))
    for i in 0 ..< stateS.len:
      worst = max(worst, abs(env.memF32[lay.offS + i] - stateS[i]) / max(1'f32, abs(stateS[i])))
  echo &"linear++-step H={H} D={D} steps=4: max rel err = {worst:.2e}"
  if worst >= 3e-5: quit(1)

proc cloneF32(v: seq[float32]): seq[float32] =
  result = newSeq[float32](v.len)
  for i, x in v: result[i] = x

proc gatedNormObjective(y, z, gamma, dy: seq[float32]; C, groups: int;
                        eps: float32): float64 =
  result = 0.0
  let BT = y.len div C
  for t in 0 ..< BT:
    let yy = y[t * C ..< (t + 1) * C]
    let zz = z[t * C ..< (t + 1) * C]
    let normed = refGatedNorm(yy, zz, gamma, C, groups, eps)
    for c in 0 ..< C:
      result += float64(normed[c]) * float64(dy[t * C + c])

proc runM2TrainingNormFiniteDiff() =
  let B = 1; let T = 2; let H = 2; let D = 8; let N = 2; let G = 2; let K = 2
  let C = H * D; let groups = G; let eps = 1e-5'f32
  let L = m2Layout(B, T, H, D, N, G, K, C, eps, groups)
  let tg = rmsTg(L)
  let BT = B * T
  var y = newSeq[float32](BT * C)
  var z = newSeq[float32](BT * C)
  var gamma = newSeq[float32](C)
  var dy = newSeq[float32](BT * C)
  for i in 0 ..< y.len:
    y[i] = float32(sin(float64(i + 3) * 0.37) * 0.8)
    z[i] = float32(cos(float64(i + 2) * 0.29) * 0.7)
    dy[i] = float32(sin(float64(i + 5) * 0.17) * 0.6)
  for i in 0 ..< C: gamma[i] = 0.7'f32 + float32(i) * 0.025'f32

  var fwd: Env
  fwd.memF32 = newSeq[float32](L.total)
  for t in 0 ..< BT:
    for c in 0 ..< C:
      fwd.memF32[L.proj + t * L.P + c] = z[t * C + c]
      fwd.memF32[L.yPre + t * C + c] = y[t * C + c]
  for c in 0 ..< C: fwd.memF32[L.normW + c] = gamma[c]
  let fwdProg = buildM2RmsFwdProgram(L)
  dispatch(fwdProg.words, fwd, BT * groups * tg, tg, tg)
  var worstFwd = 0'f32
  for t in 0 ..< BT:
    let want = refGatedNorm(y[t * C ..< (t + 1) * C], z[t * C ..< (t + 1) * C],
                            gamma, C, groups, eps)
    for c in 0 ..< C:
      let got = fwd.memF32[L.normedOut + t * C + c]
      worstFwd = max(worstFwd, abs(got - want[c]) / max(1'f32, abs(want[c])))

  var bwd: Env
  bwd.memF32 = newSeq[float32](L.total)
  for t in 0 ..< BT:
    for c in 0 ..< C:
      bwd.memF32[L.proj + t * L.P + c] = z[t * C + c]
      bwd.memF32[L.yPre + t * C + c] = y[t * C + c]
      bwd.memF32[L.dNormedOut + t * C + c] = dy[t * C + c]
  for c in 0 ..< C: bwd.memF32[L.normW + c] = gamma[c]
  let bwdProg = buildM2RmsBwdProgram(L)
  dispatch(bwdProg.words, bwd, BT * groups * tg, tg, tg)

  let delta = 1e-3'f32
  var worstGrad = 0'f32
  for i in 0 ..< y.len:
    var yup = cloneF32(y); var ydn = cloneF32(y)
    yup[i] += delta; ydn[i] -= delta
    let fdY = float32((gatedNormObjective(yup, z, gamma, dy, C, groups, eps) -
                       gatedNormObjective(ydn, z, gamma, dy, C, groups, eps)) / (2.0 * float64(delta)))
    let gotY = bwd.memF32[L.dYpre + i]
    worstGrad = max(worstGrad, abs(fdY - gotY) / max(0.1'f32, abs(fdY)))

    var zup = cloneF32(z); var zdn = cloneF32(z)
    zup[i] += delta; zdn[i] -= delta
    let fdZ = float32((gatedNormObjective(y, zup, gamma, dy, C, groups, eps) -
                       gatedNormObjective(y, zdn, gamma, dy, C, groups, eps)) / (2.0 * float64(delta)))
    let gotZ = bwd.memF32[L.dProj + (i div C) * L.P + (i mod C)]
    worstGrad = max(worstGrad, abs(fdZ - gotZ) / max(0.1'f32, abs(fdZ)))

  for c in 0 ..< C:
    var wup = cloneF32(gamma); var wdn = cloneF32(gamma)
    wup[c] += delta; wdn[c] -= delta
    let fdW = float32((gatedNormObjective(y, z, wup, dy, C, groups, eps) -
                       gatedNormObjective(y, z, wdn, dy, C, groups, eps)) / (2.0 * float64(delta)))
    let gotW = bwd.memF32[L.dNormW + c]
    worstGrad = max(worstGrad, abs(fdW - gotW) / max(0.1'f32, abs(fdW)))

  echo &"mamba2-full-rms groups={groups}: fwd rel err={worstFwd:.2e}, finite-diff grad err={worstGrad:.2e}"
  if worstFwd >= 3e-5 or worstGrad >= 8e-3: quit(1)

proc runCase(C, nGroups, tg: int) =
  let eps = 1e-5'f32
  var y = newSeq[float32](C)
  var z = newSeq[float32](C)
  var w = newSeq[float32](C)
  for i in 0 ..< C:
    y[i] = float32(rand(4.0) - 2.0) * (if i mod 97 == 0: 6'f32 else: 1'f32)
    z[i] = float32(rand(4.0) - 2.0)
    w[i] = 0.5'f32 + float32(rand(1.0))
  let offY = 0; let offZ = C; let offW = 2 * C; let offOut = 3 * C
  var env: Env
  env.memF32 = newSeq[float32](4 * C)
  for i in 0 ..< C:
    env.memF32[offY + i] = y[i]; env.memF32[offZ + i] = z[i]; env.memF32[offW + i] = w[i]
  let prog = buildRmsGateGroupedAt(C, nGroups, tg, eps, offY, offZ, offW, offOut)
  dispatch(prog.words, env, nGroups * tg, tg, tg)
  let want = refGatedNorm(y, z, w, C, nGroups, eps)
  let got = env.memF32[offOut ..< offOut + C]
  var worst = 0'f32
  for i in 0 ..< C: worst = max(worst, abs(got[i] - want[i]) / max(1e-3'f32, abs(want[i])))
  echo &"rmsg C={C} groups={nGroups} tg={tg}: max rel err = {worst:.2e}  {(if worst < 2e-4: \"OK\" else: \"SAI\")}"
  if worst >= 2e-4: quit(1)

proc runFp4Km2RowCheck() =
  ## Generated two-row microkernel versus exact dequantize+dot reference.
  for (K, N) in [(32, 2), (64, 6), (96, 8)]:
    var source = newSeq[float32](N * K)
    var x = newSeq[float32](K)
    for i in 0 ..< source.len:
      source[i] = float32(sin(float64(i * 13 + 5) * 0.071) * 2.5 + cos(float64(i * 7) * 0.019))
    for i in 0 ..< K:
      x[i] = float32(cos(float64(i * 11 + 2) * 0.037) * 0.8)
    let qt = quantizeFp4KM(source, @[N, K], groupSize = 32)
    let lut = buildFp4KmLutQ(qt)
    let prog = buildFp4KmGemv2RowProgram(K, N)
    var env: Env
    env.memF32 = newSeq[float32](N)
    env.memU8 = qt.data
    env.constF32 = lut
    env.inF32 = x
    dispatch(prog.words, env, (N div 2) * GemvTg, GemvTg,
             Fp4KmLutLen + 2 * GemvTg)
    var envOld: Env
    envOld.memF32 = newSeq[float32](N)
    envOld.memU8 = qt.data; envOld.constF32 = lut; envOld.inF32 = x
    let oldProg = buildFp4GemvProgram(K, N, true, 32)
    dispatch(oldProg.words, envOld, N * GemvTg, GemvTg, Fp4KmLutLen + GemvTg)
    let deq = dequantizeFp4KM(qt)
    var worst = 0'f32
    var oldWorst = 0'f32
    for row in 0 ..< N:
      var want = 0'f64
      for col in 0 ..< K:
        want += float64(deq[row * K + col]) * float64(x[col])
      worst = max(worst, abs(env.memF32[row] - float32(want)) / max(1e-3'f32, abs(float32(want))))
      oldWorst = max(oldWorst, abs(envOld.memF32[row] - float32(want)) /
                     max(1e-3'f32, abs(float32(want))))
    echo &"fp4km-2row K={K} N={N}: rel err new={worst:.2e}, old={oldWorst:.2e}"
    if worst >= 2e-4 or oldWorst >= 2e-4: quit(1)

proc runDynamicFp4Check() =
  const K = 8192
  const N = 4
  doAssert fp4KmTgCandidates(K, 4, 1024, 65536, "max") == @[1024]
  doAssert fp4KmTgCandidates(4096, 4, 1024, 65536, "max") == @[512]
  doAssert fp4KmTgCandidates(K, 8, 1024, 32768, "max") == @[512]
  doAssert fp4KmTgCandidates(K, 4, 128, 65536, "max") == @[128]
  doAssert fp4KmTgCandidates(K, 4, 1024, 65536, "2048").len == 0
  var source = newSeq[float32](N * K)
  var x = newSeq[float32](8 * K)
  for i in 0 ..< source.len: source[i] = float32(sin(float64(i) * 0.017) * 0.03)
  for i in 0 ..< x.len: x[i] = float32(cos(float64(i) * 0.013))
  let qt = quantizeFp4KM(source, @[N, K])
  let decoded = dequantizeTensor(qt)
  let lut = buildFp4KmLutQ(qt)
  for batch in [1, 8]:
    var expected = newSeq[float32](batch * N)
    for b in 0 ..< batch:
      for n in 0 ..< N:
        var dot = 0.0
        for k in 0 ..< K: dot += float64(x[b * K + k]) * float64(decoded[n * K + k])
        expected[b * N + n] = float32(dot)
    for tg in [32, 64, 128, 256, 512, 1024]:
      let rows = if batch == 1: 4 else: 8
      let prog = if batch == 1: buildFp4KmGemvTunedProgram(K, N, tg, rows)
                 else: buildFp4KmGemmTunedProgram(K, N, tg, rows)
      var env = Env(memF32: newSeq[float32](batch * N), memU8: qt.data,
                    constF32: lut, inF32: x)
      let groups = if batch == 1: N div rows else: N
      dispatch(prog.words, env, groups * tg, tg, Fp4KmLutLen + rows * tg)
      for i, want in expected:
        doAssert abs(env.memF32[i] - want) < max(1e-5'f32, abs(want) * 1e-4'f32)
  echo "PASS: FP4 GEMV/GEMM groups 32..1024 match dequantized dot; device/shared limits"

randomize(7)
runCase(512, 8, 32)    # cach cu: n_groups=8
runCase(512, 1, 64)    # cach TRAIN: 1 nhom tren toan C
runCase(512, 1, 32)
runCase(2048, 1, 256)  # gan kich thuoc that (C=8192 -> tg=256)
runCase(2048, 8, 128)
runFp4Km2RowCheck()
runDynamicFp4Check()
runLinearPlusStepCheck()
runM2TrainingNormFiniteDiff()
echo "TAT CA OK"
