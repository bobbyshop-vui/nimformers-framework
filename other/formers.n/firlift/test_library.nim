## Numeric test for the trace lifter's micro-kernel library.
##
## Each kernel body is assembled with the same mnemonic set the FIR kernel
## parser accepts and executed on the production CPU VM. A plain Nim loop is the
## independent reference; the tolerance is 1e-4.
import std/[math, strutils, tables]
import session_vm
import backends/cpu/cpu_backend as cpu
import ./kernel_library
import ./kernel_gen
import ./kernel_asm

proc near(a, b: float32; tol = 1e-4'f32): bool = abs(a - b) <= tol

proc runCase(name, body: string; ins, outs: int; inData: seq[seq[float32]];
             expected: seq[float32]; threads: int; tgFloats = 4; tgSize = 1;
             eps = 1e-5'f32) =
  var inSize, outSize: seq[int]
  for d in inData: inSize.add d.len
  outSize.add expected.len
  var mem: seq[float32]
  for d in inData:
    for v in d: mem.add v
  let total = mem.len
  mem.setLen(total + expected.len)
  let program = assemble(body, ins, outs, inSize, outSize, eps)
  if not cpu.vmRun(program.words, mem, newSeq[float32](), threads, tgFloats, tgSize):
    raise newException(ValueError, name & ": CPU VM rejected the program")
  for i in 0 ..< expected.len:
    if not near(mem[total + i], expected[i]):
      raise newException(ValueError, name & " mismatch at " & $i & ": got " &
        $mem[total + i] & " want " & $expected[i])
  echo "PASS library kernel ", name

proc chk(name: string; g: Gen; inData: seq[seq[float32]]; expected: seq[float32]; tol = 1e-4'f32) =
  var inSize: seq[int]
  var mem: seq[float32]
  for d in inData:
    inSize.add d.len
    for v in d: mem.add v
  let total = mem.len
  mem.setLen(total + expected.len)
  let program = assemble(g.body, g.ins, g.outs, inSize, @[expected.len], 1e-5'f32)
  if not cpu.vmRun(program.words, mem, newSeq[float32](), g.threads, max(4, g.tgBytes div 4), max(1, g.tg)):
    raise newException(ValueError, name & ": CPU VM rejected the program")
  for i in 0 ..< expected.len:
    if not near(mem[total + i], expected[i], tol):
      raise newException(ValueError, name & " mismatch at " & $i & ": got " & $mem[total + i] & " want " & $expected[i])
  echo "PASS session kernel ", name

proc main() =
  const n = 8
  var x, gamma, y: seq[float32]
  for i in 0 ..< n:
    x.add float32(sin(float(i) * 0.7) * 1.5)
    gamma.add float32(0.5 + 0.25*float(i))
  var mean = 0'f32
  for v in x: mean += v*v
  mean = mean / float32(n)
  let inv = 1.0'f32 / sqrt(mean + 1e-5'f32)
  y = newSeq[float32](n)
  for i in 0 ..< n: y[i] = x[i] * inv * gamma[i]
  runCase("script.rmsnorm", RmsnormBody, 2, 1, @[x, gamma], y, 1)
  runCase("script.rmsnorm_par", RmsnormParBody, 2, 1, @[x, gamma], y, 256, 256, 256)

  var a, b: seq[float32]
  for i in 0 ..< n:
    a.add float32(cos(float(i) * 0.3))
    b.add float32(sin(float(i) * 0.4))
  var r = newSeq[float32](n)
  for i in 0 ..< n: r[i] = a[i] + b[i]
  runCase("script.add", AddBody, 2, 1, @[a, b], r, n)
  for i in 0 ..< n: r[i] = a[i] * b[i]
  runCase("script.mul", MulBody, 2, 1, @[a, b], r, n)
  for i in 0 ..< n: r[i] = 1.0'f32 / (1.0'f32 + exp(-a[i]))
  runCase("script.sigmoid", SigmoidBody, 1, 1, @[a], r, n)
  for i in 0 ..< n: r[i] = b[i] * (1.0'f32 / (1.0'f32 + exp(-b[i]))) * a[i]
  runCase("script.silu_mul", SiluMulBody, 2, 1, @[a, b], r, n)

proc fill(n: int; seed: float; scale = 1.0): seq[float32] =
  for i in 0 ..< n:
    result.add float32(sin(float(i) * 0.9 + seed) * scale)

proc runMem(g: Gen; ins: seq[seq[float32]]; outLens: seq[int]): seq[float32] =
  ## Flat memory after the run: inputs, then outputs.
  var sizes: seq[int]
  var mem: seq[float32]
  for d in ins:
    sizes.add d.len
    mem.add d
  let total = mem.len
  var outTotal = 0
  for n in outLens: outTotal += n
  mem.setLen(total + outTotal)
  let program = assemble(g.body, g.ins, g.outs, sizes, outLens, 1e-5'f32)
  if not cpu.vmRun(program.words, mem, newSeq[float32](), g.threads, 4, 1):
    raise newException(ValueError, "CPU VM rejected the program")
  mem

proc chkN(name: string; g: Gen; ins: seq[seq[float32]]; expected: seq[seq[float32]]; tol = 1e-4'f32) =
  var lens: seq[int]
  var base = 0
  for d in ins: base += d.len
  for e in expected: lens.add e.len
  let mem = runMem(g, ins, lens)
  var at = base
  for oi, e in expected:
    for i in 0 ..< e.len:
      if not near(mem[at + i], e[i], tol):
        raise newException(ValueError, name & " output " & $oi & " mismatch at " & $i &
          ": got " & $mem[at + i] & " want " & $e[i])
    at += e.len
  echo "PASS session kernel ", name

proc bytesAsFloats(b: seq[uint8]): seq[float32] =
  result = newSeq[float32]((b.len + 3) div 4)
  if b.len > 0: copyMem(addr result[0], unsafeAddr b[0], b.len)

proc sessionTests2() =
  block:
    let g = Gen(body: "uloadf32 1 2 @in1+4\nustoref32 1 2 @out0\nuloadf32 3 2 @in0\nuloadf32 4 2 @in1", threads: 1, ins: 2, outs: 1)
    let r = g.rebasePorts(@[0, 10], @[3])
    doAssert r.body == "uloadf32 1 2 @in1+14\nustoref32 1 2 @out0+3\nuloadf32 3 2 @in0\nuloadf32 4 2 @in1+10", r.body
    echo "PASS rebase_ports"
  chk("copy", genCopy(5), @[fill(5, 0.3)], fill(5, 0.3))

  # FP4-KM GEMM: group 32, byte j = element j (low) / j+16 (high), scale = lut[16 + byte]
  for (M, K, N) in [(1, 32, 3), (2, 64, 5)]:
    let nG = K div 32
    let lay = fp4kmLayout(N, K)
    var bytes = newSeq[uint8](lay.blobWords * 4)
    var nibs = newSeq[int](N*K)
    var sbytes = newSeq[int](N*nG)
    for i in 0 ..< N*K: nibs[i] = (i*5 + 3) mod 16
    for i in 0 ..< N*nG: sbytes[i] = (i*37 + 11) mod 256
    for n in 0 ..< N:
      for g in 0 ..< nG:
        for j in 0 ..< 16:
          let lo = nibs[n*K + g*32 + j]
          let hi = nibs[n*K + g*32 + j + 16]
          bytes[n*(K div 2) + g*16 + j] = uint8(lo or (hi shl 4))
        bytes[N*(K div 2) + n*nG + g] = uint8(sbytes[n*nG + g])
    var lut = newSeq[float32](272)
    for i in 0 ..< 16: lut[i] = float32(float(i - 7) * 0.375)
    for i in 0 ..< 256: lut[16 + i] = float32(float(i) * 0.0078125 + 0.01)
    var blob = bytesAsFloats(bytes)
    blob.add lut
    let a = fill(M*K, 0.35, 1.0)
    var want = newSeq[float32](M*N)
    for m in 0 ..< M:
      for n in 0 ..< N:
        var acc = 0.0
        for kk in 0 ..< K:
          let g = kk div 32
          let v = lut[nibs[n*K + kk]] * lut[16 + sbytes[n*nG + g]]
          acc += float(a[m*K + kk]) * float(v)
        want[m*N + n] = float32(acc)
    chk("fp4_gemm M" & $M & " K" & $K, genFp4Gemm(M, K, N), @[a, blob], want, 2e-3'f32)

  # fused launches
  block:
    let n = 12
    let a = fill(n, 0.2)
    let b = fill(n, 0.9)
    let gamma = fill(n, 1.3, 0.5)
    var sum = newSeq[float32](n)
    var ms = 0.0
    for i in 0 ..< n:
      sum[i] = a[i] + b[i]
      ms += float(sum[i]) * float(sum[i])
    let inv = 1.0 / sqrt(ms / float(n) + 1e-5)
    var normed = newSeq[float32](n)
    for i in 0 ..< n: normed[i] = float32(float(sum[i]) * inv * float(gamma[i]))
    chkN("add_rmsnorm", genAddRmsnorm(n, 1e-5), @[a, b, gamma], @[sum, normed])
    var want = newSeq[float32](n)
    for i in 0 ..< n:
      let x = float(a[i])
      want[i] = float32(x / (1.0 + exp(-x)) * float(b[i]))
    chk("silu_mul", genSiluMul(n), @[a, b], want)
    let Hn = 2
    let D = 8
    let xs = fill(Hn*D, 0.4)
    var plain = xs
    var scaled = xs
    var s1 = newSeq[float32](Hn*D)
    for h in 0 ..< Hn:
      for i in 0 ..< D div 2:
        let ang = 5.0 / pow(100.0, float(i) / float(D div 2))
        let x1 = float(xs[h*D + i])
        let x2 = float(xs[h*D + i + D div 2])
        s1[h*D + i] = float32((x1*cos(ang) - x2*sin(ang)) * 0.125)
        s1[h*D + i + D div 2] = float32((x2*cos(ang) + x1*sin(ang)) * 0.125)
    chk("rope_pos_scaled", genRopePos(Hn, D, pow(100.0, -1.0 / float(D div 2)), 1.0, 0.125),
      @[xs, @[5.0'f32]], s1, 2e-3'f32)

  # copy with offsets: only the copied range of out0 is written
  block:
    let src = fill(8, 0.3)
    var want = newSeq[float32](10)
    for i in 0 ..< 3: want[5 + i] = src[2 + i]
    chk("copy_offset", genCopyOffset(3, 2, 5), @[src], want)

  # RoPE from a position scalar: theta^(-i/half) frequencies, sin/cos by series
  for (pos, theta, scale) in [(0.0, 10000.0, 1.0), (1.0, 10000.0, 1.0), (137.0, 10000.0, 1.0),
                              (2047.0, 10000.0, 1.0), (300.0, 500000.0, 1.0), (96.0, 10000.0, 0.25)]:
    let Hn = 3
    let D = 8
    let half = D div 2
    let xs = fill(Hn*D, 0.4, 1.0)
    var want = xs
    for h in 0 ..< Hn:
      for i in 0 ..< half:
        let angle = pos * scale / pow(theta, float(i) / float(half))
        let x1 = float(xs[h*D + i])
        let x2 = float(xs[h*D + i + half])
        want[h*D + i] = float32(x1*cos(angle) - x2*sin(angle))
        want[h*D + i + half] = float32(x2*cos(angle) + x1*sin(angle))
    chk("rope_pos " & $pos & " theta " & $theta, genRopePos(Hn, D, pow(theta, -1.0 / float(half)), scale),
      @[xs, @[float32(pos)]], want, 2e-3'f32)

  # multi-row RoPE: row t uses position + t, rows laid out [T, Hn, D]
  for (pos, T, theta, scale) in [(0.0, 3, 10000.0, 1.0), (5.0, 4, 10000.0, 1.0), (96.0, 2, 500000.0, 0.25)]:
    let Hn = 3
    let D = 8
    let half = D div 2
    let xs = fill(T*Hn*D, 0.4, 1.0)
    var want = xs
    for t in 0 ..< T:
      for h in 0 ..< Hn:
        for i in 0 ..< half:
          let angle = (pos + float(t)) * scale / pow(theta, float(i) / float(half))
          let base = (t*Hn + h)*D
          let x1 = float(xs[base + i])
          let x2 = float(xs[base + i + half])
          want[base + i] = float32(x1*cos(angle) - x2*sin(angle))
          want[base + i + half] = float32(x2*cos(angle) + x1*sin(angle))
    chk("rope_pos_rows T=" & $T & " pos " & $pos, genRopePosRows(T, Hn, D, pow(theta, -1.0 / float(half)), scale),
      @[xs, @[float32(pos)]], want, 2e-3'f32)

  # asymmetric Q4 matmul: low nibble = even k, high = odd k
  for (K, gs) in [(10, 4), (9, 4), (6, 0), (8, 8)]:
    let M = 2
    let N = 3
    let bpr = (K + 1) div 2
    let g = if gs > 0: gs else: K
    let nG = (K + g - 1) div g
    var q = newSeq[int](N*K)
    var bytes = newSeq[uint8](N*bpr)
    for i in 0 ..< N*K: q[i] = (i*7 + 3) mod 16
    for n in 0 ..< N:
      for kk in 0 ..< K:
        let b = n*bpr + kk div 2
        bytes[b] = bytes[b] or uint8(q[n*K + kk] shl (if kk mod 2 == 0: 0 else: 4))
    let scales = fill(N*nG, 0.4, 0.3)
    let zeros = fill(N*nG, 1.1, 2.0)
    let a = fill(M*K, 0.8)
    var want = newSeq[float32](M*N)
    for m in 0 ..< M:
      for n in 0 ..< N:
        var s = 0.0'f32
        for kk in 0 ..< K:
          let gi = n*nG + kk div g
          s += a[m*K + kk] * ((float32(q[n*K + kk]) - zeros[gi]) * scales[gi])
        want[m*N + n] = s
    chk("matmul_q4.K" & $K & ".gs" & $gs, genMatmulQ4(M, K, N, gs, nG),
        @[a, bytesAsFloats(bytes), scales, zeros], want, 1e-3'f32)

  # packed q4b layout: [nibble bytes padded to words][scales][zeros] in one port
  block:
    let M = 2
    let K = 40
    let N = 3
    let gs = 32
    let nG = (K + gs - 1) div gs
    let bpr = (K + 1) div 2
    var q = newSeq[int](N*K)
    var bytes = newSeq[uint8](N*bpr)
    for i in 0 ..< N*K: q[i] = (i*5 + 1) mod 16
    for n in 0 ..< N:
      for kk in 0 ..< K:
        let b = n*bpr + kk div 2
        bytes[b] = bytes[b] or uint8(q[n*K + kk] shl (if kk mod 2 == 0: 0 else: 4))
    let scales = fill(N*nG, 0.4, 0.3)
    let zeros = fill(N*nG, 1.1, 2.0)
    let a = fill(M*K, 0.8)
    var want = newSeq[float32](M*N)
    for m in 0 ..< M:
      for n in 0 ..< N:
        var s = 0.0'f32
        for kk in 0 ..< K:
          let gi = n*nG + kk div gs
          s += a[m*K + kk] * ((float32(q[n*K + kk]) - zeros[gi]) * scales[gi])
        want[m*N + n] = s
    chk("matmul_q4.packed", genMatmulQ4(M, K, N, gs, nG, 0, true),
        @[a, bytesAsFloats(bytes) & scales & zeros], want, 1e-3'f32)

  # threadgroup-parallel Q4 matmul: same packed layout, T lanes per output column
  for (M, K, N, gs, T) in [(1, 128, 5, 32, 8), (2, 128, 3, 64, 32), (1, 256, 4, 128, 32), (3, 64, 2, 16, 4)]:
    let nG = K div gs
    let bpr = K div 2
    var q = newSeq[int](N*K)
    var bytes = newSeq[uint8](N*bpr)
    for i in 0 ..< N*K: q[i] = (i*7 + 2) mod 16
    for n in 0 ..< N:
      for kk in 0 ..< K:
        let b = n*bpr + kk div 2
        bytes[b] = bytes[b] or uint8(q[n*K + kk] shl (if kk mod 2 == 0: 0 else: 4))
    let scales = fill(N*nG, 0.4, 0.3)
    let zeros = fill(N*nG, 1.1, 2.0)
    let a = fill(M*K, 0.8)
    var want = newSeq[float32](M*N)
    for m in 0 ..< M:
      for n in 0 ..< N:
        var s = 0.0'f32
        for kk in 0 ..< K:
          let gi = n*nG + kk div gs
          s += a[m*K + kk] * ((float32(q[n*K + kk]) - zeros[gi]) * scales[gi])
        want[m*N + n] = s
    let g = genMatmulQ4Par(M, K, N, gs, nG, T)
    doAssert g.tg >= 2, "parallel variant not selected"
    chk("matmul_q4_par M" & $M & " K" & $K & " gs" & $gs & " tg" & $g.tg, g,
        @[a, bytesAsFloats(bytes) & scales & zeros], want, 1e-3'f32)

  # argmax keeps the first maximum
  block:
    let outer = 2
    let dim = 5
    let inner = 3
    var x = fill(outer*dim*inner, 0.5)
    x[1*inner + 0] = 9.0'f32
    x[3*inner + 0] = 9.0'f32      # tie, first wins
    var val = newSeq[float32](outer*inner)
    var idx = newSeq[float32](outer*inner)
    for o in 0 ..< outer:
      for i in 0 ..< inner:
        var best = x[(o*dim)*inner + i]
        var bi = 0
        for d in 1 ..< dim:
          let v = x[(o*dim + d)*inner + i]
          if v > best:
            best = v
            bi = d
        val[o*inner + i] = best
        idx[o*inner + i] = float32(bi)
    chkN("argmax", genArgmax(outer, dim, inner), @[x], @[val, idx])

  # compare
  block:
    let a = @[1.0'f32, 2, 3, 2, 5]
    let b = @[2.0'f32, 2, 1, 2, 4]
    for op in 0 .. 4:
      var want = newSeq[float32](a.len)
      for i in 0 ..< a.len:
        let r = case op
          of 0: a[i] > b[i]
          of 1: a[i] < b[i]
          of 2: a[i] == b[i]
          of 3: a[i] >= b[i]
          else: a[i] <= b[i]
        want[i] = if r: 1.0 else: 0.0
      chk("compare.op" & $op, genCompare(op, a.len), @[a, b], want)

  # split slice, concat, permute
  block:
    let dimTotal = 6
    let thisSize = 2
    let offset = 3
    let inner = 3
    let outer = 2
    let src = fill(outer*dimTotal*inner, 0.2)
    var want = newSeq[float32](outer*thisSize*inner)
    for o in 0 ..< outer:
      for d in 0 ..< thisSize:
        for i in 0 ..< inner:
          want[(o*thisSize + d)*inner + i] = src[(o*dimTotal + offset + d)*inner + i]
    chk("split_slice", genSplitSlice(dimTotal, thisSize, offset, inner, want.len), @[src], want)
    let fa = fill(outer*2*inner, 0.7)
    let fb = fill(outer*3*inner, 1.4)
    var cat = newSeq[float32](outer*5*inner)
    for o in 0 ..< outer:
      for d in 0 ..< 2:
        for i in 0 ..< inner: cat[(o*5 + d)*inner + i] = fa[(o*2 + d)*inner + i]
      for d in 0 ..< 3:
        for i in 0 ..< inner: cat[(o*5 + 2 + d)*inner + i] = fb[(o*3 + d)*inner + i]
    chk("concat", genConcat(@[0, fa.len], @[2, 3], @[0, 2], 5, inner, cat.len), @[fa & fb], cat)
    let srcP = fill(24, 0.9)
    let newStrides = @[6, 3, 1]
    let oldStrides = @[12, 4, 1]
    let dimsMap = @[2, 0, 1]
    var perm = newSeq[float32](24)
    for lin in 0 ..< 24:
      var rem = lin
      var old = 0
      for i in 0 ..< 3:
        let ix = rem div newStrides[i]
        rem = rem mod newStrides[i]
        old += ix * oldStrides[dimsMap[i]]
      perm[lin] = srcP[old]
    chk("permute", genPermute(newStrides, oldStrides, dimsMap, 24), @[srcP], perm)

  # dropout mask (xorshift32 reference)
  block:
    let n = 16
    let p = 0.3
    let seed = 12345'u32
    var want = newSeq[float32](n)
    for id in 0 ..< n:
      var st = seed xor (uint32(id) * 2654435761'u32 + 1'u32)
      for _ in 0 ..< 2:
        st = st xor (st shl 13)
        st = st xor (st shr 17)
        st = st xor (st shl 5)
      let u = float(st shr 8) * (1.0 / 16777216.0)
      want[id] = if u < p: 0.0'f32 else: float32(1.0 / (1.0 - p))
    chk("dropout_mask", genDropoutMask(n, p, seed), newSeq[seq[float32]](), want)

  # APFLU forward and backward
  block:
    let n = 9
    let x = fill(n, 0.35, 2.0)
    let dy = fill(n, 1.2)
    var f = newSeq[float32](n)
    var b = newSeq[float32](n)
    for i in 0 ..< n:
      let v = x[i]
      f[i] = if v > 0: v * (1 + 0.1'f32*v) else: 1.0'f32 * v * exp(v)
      b[i] = if v > 0: dy[i] * (1 + 2*0.1'f32*v) else: dy[i] * 1.0'f32 * exp(v) * (1 + v)
    chk("apflu", genApflu(n, 0.1, 1.0), @[x], f, 1e-3'f32)
    chk("apflu_backward", genApfluBackward(n, 0.1, 1.0), @[x, dy], b, 1e-3'f32)

  # layernorm backward
  block:
    let rows = 3
    let cols = 5
    let x = fill(rows*cols, 0.1, 2.0)
    let dy = fill(rows*cols, 0.9)
    let gamma = fill(cols, 1.7)
    var dx = newSeq[float32](rows*cols)
    var dg = newSeq[float32](cols)
    var db = newSeq[float32](cols)
    for r in 0 ..< rows:
      var mean = 0.0'f32
      for c in 0 ..< cols: mean += x[r*cols + c]
      mean /= float32(cols)
      var v = 0.0'f32
      for c in 0 ..< cols: v += (x[r*cols + c] - mean)*(x[r*cols + c] - mean)
      v /= float32(cols)
      let inv = 1.0'f32 / sqrt(v + 1e-5'f32)
      var s1 = 0.0'f32
      var s2 = 0.0'f32
      for c in 0 ..< cols:
        let nrm = (x[r*cols + c] - mean) * inv
        dg[c] += dy[r*cols + c] * nrm
        db[c] += dy[r*cols + c]
        let gg = dy[r*cols + c] * gamma[c] * inv
        s1 += gg
        s2 += gg * (x[r*cols + c] - mean)
      s2 *= -inv*inv / float32(cols)
      for c in 0 ..< cols:
        dx[r*cols + c] = dy[r*cols + c]*gamma[c]*inv - s1/float32(cols) + (x[r*cols + c] - mean)*s2
    chk("layernorm_backward.dx", genLayernormBackwardDx(rows, cols, 1e-5), @[dy, x, gamma], dx, 1e-3'f32)
    chkN("layernorm_backward.params", genLayernormBackwardParams(rows, cols, 1e-5), @[dy, x], @[dg, db], 1e-3'f32)

  # single-K/V GQA decode
  block:
    let H = 4
    let S = 5
    let D = 4
    let sc = 0.5'f32
    let q = fill(H*D, 0.2)
    let kk = fill(S*D, 0.6)
    let vv = fill(S*D, 1.1)
    var o = newSeq[float32](H*D)
    var scores = newSeq[float32](H*S)
    for h in 0 ..< H:
      var mx = -1e30'f32
      var raw = newSeq[float32](S)
      for s in 0 ..< S:
        var d0 = 0.0'f32
        for d in 0 ..< D: d0 += q[h*D + d]*kk[s*D + d]
        raw[s] = d0*sc
        mx = max(mx, raw[s])
      var tot = 0.0'f32
      for s in 0 ..< S:
        scores[h*S + s] = exp(raw[s] - mx)
        tot += scores[h*S + s]
      for d in 0 ..< D:
        var a = 0.0'f32
        for s in 0 ..< S: a += scores[h*S + s]*vv[s*D + d]
        o[h*D + d] = a / tot
    chkN("attention_decode_gqa", genAttentionDecodeGqa(H, S, D, sc), @[q, kk, vv], @[o, scores], 1e-3'f32)

  # fused causal attention forward and backward
  block:
    let B = 1
    let H = 2
    let S = 4
    let D = 3
    let sc = 0.5'f32
    let BH = B*H
    let q = fill(BH*S*D, 0.1)
    let kk = fill(BH*S*D, 0.8)
    let vv = fill(BH*S*D, 1.5)
    let dy = fill(BH*S*D, 2.2)
    var o = newSeq[float32](BH*S*D)
    var sm = newSeq[float32](BH*S*S)
    for bh in 0 ..< BH:
      for ti in 0 ..< S:
        var mx = -1e30'f32
        for tj in 0 .. ti:
          var d0 = 0.0'f32
          for d in 0 ..< D: d0 += q[(bh*S + ti)*D + d]*kk[(bh*S + tj)*D + d]
          sm[(bh*S + ti)*S + tj] = d0*sc
          mx = max(mx, sm[(bh*S + ti)*S + tj])
        var tot = 0.0'f32
        for tj in 0 .. ti:
          sm[(bh*S + ti)*S + tj] = exp(sm[(bh*S + ti)*S + tj] - mx)
          tot += sm[(bh*S + ti)*S + tj]
        for tj in 0 .. ti: sm[(bh*S + ti)*S + tj] /= tot
        for d in 0 ..< D:
          var a = 0.0'f32
          for tj in 0 .. ti: a += sm[(bh*S + ti)*S + tj]*vv[(bh*S + tj)*D + d]
          o[(bh*S + ti)*D + d] = a
    chkN("attention_fused", genAttentionFused(B, H, S, D, sc), @[q, kk, vv], @[o, sm], 1e-3'f32)
    var dq = newSeq[float32](BH*S*D)
    var dk = newSeq[float32](BH*S*D)
    var dv = newSeq[float32](BH*S*D)
    for bh in 0 ..< BH:
      for ti in 0 ..< S:
        var dS = newSeq[float32](ti + 1)
        for tj in 0 .. ti:
          let w = sm[(bh*S + ti)*S + tj]
          var dot = 0.0'f32
          for d in 0 ..< D:
            dv[(bh*S + tj)*D + d] += w*dy[(bh*S + ti)*D + d]
            dot += dy[(bh*S + ti)*D + d]*vv[(bh*S + tj)*D + d]
          dS[tj] = dot
        var dsum = 0.0'f32
        for tj in 0 .. ti: dsum += sm[(bh*S + ti)*S + tj]*dS[tj]
        for tj in 0 .. ti:
          let ds = sm[(bh*S + ti)*S + tj]*(dS[tj] - dsum)*sc
          for d in 0 ..< D:
            dq[(bh*S + ti)*D + d] += ds*kk[(bh*S + tj)*D + d]
            dk[(bh*S + tj)*D + d] += ds*q[(bh*S + ti)*D + d]
    chkN("attention_fused_backward", genAttentionFusedBackward(B, H, S, D, sc),
         @[q, kk, vv, sm, dy], @[dq, dk, dv], 1e-3'f32)

  # causal flash prefill over a GQA cache
  block:
    let T = 3
    let H = 4
    let HKV = 2
    let D = 4
    let maxSeq = 8
    let pos = 2
    let sc = 0.5'f32
    let q = fill(T*H*D, 0.3)
    let kc = fill(HKV*maxSeq*D, 0.9)
    let vc = fill(HKV*maxSeq*D, 1.6)
    var o = newSeq[float32](T*H*D)
    let rep = H div HKV
    for t in 0 ..< T:
      for h in 0 ..< H:
        let base = (h div rep)*maxSeq*D
        let n = pos + t + 1
        var raw = newSeq[float32](n)
        var mx = -1e30'f32
        for key in 0 ..< n:
          var s = 0.0'f32
          for d in 0 ..< D: s += q[(t*H + h)*D + d]*kc[base + key*D + d]
          raw[key] = s*sc
          mx = max(mx, raw[key])
        var tot = 0.0'f32
        for key in 0 ..< n:
          raw[key] = exp(raw[key] - mx)
          tot += raw[key]
        for d in 0 ..< D:
          var a = 0.0'f32
          for key in 0 ..< n: a += raw[key]*vc[base + key*D + d]
          o[(t*H + h)*D + d] = a / tot
    chk("flash_prefill", genFlashPrefill(T, H, HKV, D, maxSeq, sc), @[q, kc, vc, @[float32(pos)]], o, 1e-3'f32)

  # Mamba-2 state-space scan
  for cont in 0 .. 1:
    let T = 5
    let H = 4
    let G = 2
    let N = 3
    let D = 2
    let C = H*D
    let cDim = C + 2*G*N
    let conv = fill(T*cDim, 0.25, 1.0)
    var dt = fill(T*H, 0.6, 0.4)
    for i in 0 ..< dt.len: dt[i] = abs(dt[i]) + 0.05'f32
    let aLog = fill(H, 0.1, 0.5)
    let dSkip = fill(H, 1.3)
    var states = newSeq[float32](2*C*N)
    for i in 0 ..< C*N: states[i] = float32(sin(float(i)*0.2))
    var cur = newSeq[float32](C*N)
    if cont == 1:
      for i in 0 ..< C*N: cur[i] = states[i]
    var y = newSeq[float32](T*C)
    for t in 0 ..< T:
      for h in 0 ..< H:
        let gi = h div (H div G)
        let decay = exp(dt[t*H + h] * -exp(aLog[h]))
        for d in 0 ..< D:
          let x = conv[t*cDim + h*D + d]
          var acc = 0.0'f32
          for n in 0 ..< N:
            let si = (h*D + d)*N + n
            cur[si] = cur[si]*decay + conv[t*cDim + C + gi*N + n]*dt[t*H + h]*x
            acc += conv[t*cDim + C + G*N + gi*N + n]*cur[si]
          y[t*C + h*D + d] = acc + dSkip[h]*x
    let g = genMamba2Ssd(T, H, G, N, D, cont)
    let mem = runMem(g, @[conv, dt, aLog, dSkip, states], @[T*C])
    var base = conv.len + dt.len + aLog.len + dSkip.len
    for i in 0 ..< C*N:
      if not near(mem[base + C*N + i], cur[i], 1e-3'f32):
        raise newException(ValueError, "mamba2_ssd final state mismatch at " & $i)
    base += states.len
    for i in 0 ..< y.len:
      if not near(mem[base + i], y[i], 1e-3'f32):
        raise newException(ValueError, "mamba2_ssd output mismatch at " & $i & " got " & $mem[base + i] & " want " & $y[i])
    echo "PASS session kernel mamba2_ssd.cont" & $cont

proc sessionTests() =
  # elementwise ops
  let n = 9
  let a = fill(n, 0.1, 1.5)
  let b = fill(n, 0.7, 2.0)
  for op in ["add", "sub", "mul", "div", "max", "min"]:
    var want = newSeq[float32](n)
    for i in 0 ..< n:
      want[i] = case op
        of "add": a[i] + b[i]
        of "sub": a[i] - b[i]
        of "mul": a[i] * b[i]
        of "div": a[i] / b[i]
        of "max": max(a[i], b[i])
        else: min(a[i], b[i])
    chk("vecop." & op, genVecOp(op, n), @[a, b], want, 1e-3'f32)
  var pos = newSeq[float32](n)
  for i in 0 ..< n: pos[i] = float32(0.3 + 0.35*float(i))
  for op in ["relu", "sigmoid", "tanh", "silu", "gelu", "gelu_new", "gelu_quick", "exp",
             "sqrt", "log", "softplus", "neg", "abs"]:
    var want = newSeq[float32](n)
    let src = if op in ["sqrt", "log"]: pos else: a
    for i in 0 ..< n:
      let x = float(src[i])
      want[i] = float32(case op
        of "relu": max(x, 0.0)
        of "sigmoid": 1.0 / (1.0 + exp(-x))
        of "tanh": tanh(x)
        of "silu": x / (1.0 + exp(-x))
        of "gelu": 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
        of "gelu_new": 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x*x*x)))
        of "gelu_quick": x / (1.0 + exp(-1.702 * x))
        of "exp": exp(x)
        of "sqrt": sqrt(x)
        of "log": ln(x)
        of "softplus": ln(1.0 + exp(x))
        of "neg": -x
        else: abs(x))
    chk("activation." & op, genActivation(op, n), @[src], want, 2e-3'f32)
  block:
    var want = newSeq[float32](n)
    for i in 0 ..< n: want[i] = a[i]*a[i]*a[i]
    chk("activation.pow3", genActivation("pow", n, 3.0), @[a], want, 1e-3'f32)
    for i in 0 ..< n: want[i] = a[i] * 0.125'f32
    chk("scale", genScale(n, 0.125), @[a], want)

  # row ops
  let rows = 3
  let cols = 6
  let x = fill(rows*cols, 0.2, 2.0)
  let gamma = fill(cols, 1.1, 1.0)
  let beta = fill(cols, 2.3, 0.5)
  var sm = newSeq[float32](rows*cols)
  var rn = newSeq[float32](rows*cols)
  var ln1 = newSeq[float32](rows*cols)
  for r in 0 ..< rows:
    var mx = -1e30
    var mean = 0.0
    var ms = 0.0
    for c in 0 ..< cols:
      let v = float(x[r*cols + c])
      mx = max(mx, v)
      mean += v
      ms += v*v
    mean /= float(cols)
    ms /= float(cols)
    var den = 0.0
    var varr = 0.0
    for c in 0 ..< cols:
      let v = float(x[r*cols + c])
      den += exp(v - mx)
      varr += (v - mean)*(v - mean)
    varr /= float(cols)
    for c in 0 ..< cols:
      let v = float(x[r*cols + c])
      sm[r*cols + c] = float32(exp(v - mx) / den)
      rn[r*cols + c] = float32(v / sqrt(ms + 1e-5) * float(gamma[c]))
      ln1[r*cols + c] = float32((v - mean) / sqrt(varr + 1e-5) * float(gamma[c]) + float(beta[c]))
  chk("softmax", genSoftmax(rows, cols), @[x], sm)
  chk("rmsnorm.rows", genRmsnorm(rows, cols, 1e-5), @[x, gamma], rn)
  chk("layernorm", genLayernorm(rows, cols, 1e-5), @[x, gamma, beta], ln1, 1e-3'f32)

  # RoPE with [T, D/2] tables
  block:
    let T = 2
    let Hn = 3
    let D = 8
    let half = D div 2
    let xs = fill(T*Hn*D, 0.4, 1.0)
    let cosT = fill(T*half, 0.5, 1.0)
    let sinT = fill(T*half, 1.5, 1.0)
    var want = xs
    for t in 0 ..< T:
      for h in 0 ..< Hn:
        for i in 0 ..< half:
          let base = (t*Hn + h)*D
          let x1 = xs[base + i]
          let x2 = xs[base + i + half]
          let c = cosT[t*half + i]
          let s = sinT[t*half + i]
          want[base + i] = x1*c - x2*s
          want[base + i + half] = x2*c + x1*s
    chk("rope", genRope(T, Hn, D), @[xs, cosT, sinT], want)

  # matmuls
  block:
    let M = 3
    let K = 5
    let N = 4
    let am = fill(M*K, 0.3)
    let bnt = fill(N*K, 0.8)
    var want = newSeq[float32](M*N)
    for m in 0 ..< M:
      for j in 0 ..< N:
        var s = 0.0'f32
        for q in 0 ..< K: s += am[m*K + q] * bnt[j*K + q]
        want[m*N + j] = s
    chk("matmul_nt", genMatmulNT(M, K, N), @[am, bnt], want)
    chk("matmul_nt.offset", genMatmulNT(M, K, N, 2), @[am, @[0.0'f32, 0.0'f32] & bnt], want)
    for (pm, pk, pn) in [(2, 64, 3), (1, 96, 5)]:
      let pa = fill(pm*pk, 0.3)
      let pb = fill(pn*pk, 0.8)
      var pw = newSeq[float32](pm*pn)
      for m in 0 ..< pm:
        for j in 0 ..< pn:
          var s = 0.0'f32
          for q in 0 ..< pk: s += pa[m*pk + q] * pb[j*pk + q]
          pw[m*pn + j] = s
      let pg = genMatmulNTPar(pm, pk, pn)
      doAssert pg.tg >= 2
      chk("matmul_nt_par K" & $pk & " tg" & $pg.tg, pg, @[pa, pb], pw, 1e-3'f32)
    let bnn = fill(K*N, 1.9)
    for m in 0 ..< M:
      for j in 0 ..< N:
        var s = 0.0'f32
        for q in 0 ..< K: s += am[m*K + q] * bnn[q*N + j]
        want[m*N + j] = s
    chk("matmul", genMatmul(M, K, N), @[am, bnn], want)

  # gather and embedding
  block:
    let src = fill(10, 0.6)
    let idx = @[3'f32, 0, 9, 3, 7]
    var want = newSeq[float32](idx.len)
    for i in 0 ..< idx.len: want[i] = src[int(idx[i])]
    chk("gather", genGather(idx.len), @[src, idx], want)
    let table = fill(5*4, 0.2)
    let ids = @[4'f32, 1, 1]
    var emb = newSeq[float32](ids.len*4)
    for i in 0 ..< ids.len:
      for d in 0 ..< 4: emb[i*4 + d] = table[int(ids[i])*4 + d]
    chk("embedding", genEmbedding(4, ids.len), @[table, ids], emb)

  # attention decode: GQA, positions below the cache capacity
  block:
    let H = 4
    let HKV = 2
    let D = 4
    let maxSeq = 6
    let S = 5
    let q = fill(H*D, 0.1)
    let kc = fill(HKV*maxSeq*D, 0.5)
    let vc = fill(HKV*maxSeq*D, 1.3)
    var want = newSeq[float32](H*D)
    let nRep = H div HKV
    for h in 0 ..< H:
      let hk = h div nRep
      var sc = newSeq[float64](S)
      var mx = -1e30
      for s in 0 ..< S:
        for d in 0 ..< D: sc[s] += float(q[h*D + d]) * float(kc[(hk*maxSeq + s)*D + d])
        mx = max(mx, sc[s])
      var den = 0.0
      for s in 0 ..< S:
        sc[s] = exp(sc[s] - mx)
        den += sc[s]
      for d in 0 ..< D:
        var o = 0.0
        for s in 0 ..< S: o += sc[s] / den * float(vc[(hk*maxSeq + s)*D + d])
        want[h*D + d] = float32(o)
    chk("attention_decode", genAttentionDecode(H, HKV, D, maxSeq), @[q, kc, vc, @[float32(S)]], want)

  # KV scatter writes into its cache port
  block:
    let T = 2
    let HKV = 2
    let D = 3
    let maxSeq = 5
    let pos = 1
    let src = fill(T*HKV*D, 0.9)
    var cache = newSeq[float32](HKV*maxSeq*D)
    let g = genKvScatter(T, HKV, D, maxSeq)
    var mem = src
    mem.add float32(pos)
    mem.add cache
    let program = assemble(g.body, g.ins, g.outs, @[src.len, 1, cache.len], @[], 1e-5'f32)
    if not cpu.vmRun(program.words, mem, newSeq[float32](), g.threads, 4, 1):
      raise newException(ValueError, "kv_scatter: CPU VM rejected the program")
    for t in 0 ..< T:
      for h in 0 ..< HKV:
        for d in 0 ..< D:
          cache[(h*maxSeq + pos + t)*D + d] = src[(t*HKV + h)*D + d]
    for i in 0 ..< cache.len:
      if not near(mem[src.len + 1 + i], cache[i]):
        raise newException(ValueError, "kv_scatter mismatch at " & $i)
    echo "PASS session kernel kv_scatter"

sessionTests2()
sessionTests()
main()
