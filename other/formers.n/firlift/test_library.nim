## Numeric test for the trace lifter's micro-kernel library.
##
## Each kernel body is assembled with the same mnemonic set the FIR kernel
## parser accepts and executed on the production CPU VM. A plain Nim loop is the
## independent reference; the tolerance is 1e-4.
import std/[math, strutils, tables, os, osproc, streams]
import session_vm
import jit_vm
import backends/cpu/cpu_backend as cpu
import ./kernel_library
import ./kernel_gen
import ./kernel_asm
when defined(macosx):
  import backends/metal/metal_backend

proc near(a, b: float32; tol = 1e-4'f32): bool = abs(a - b) <= tol

var gMetalCompilerFound = -1   # -1 untried, 0 missing, 1 `xcrun -sdk macosx metal` works

proc mslCompileCheck(source, tag: string) =
  ## Compiles a generated MSL kernel offline (MSL 2.3, what simdgroup_matrix needs) when the
  ## Metal toolchain is installed, like the glslc check of the GLSL kernels; skipped otherwise.
  if gMetalCompilerFound < 0:
    gMetalCompilerFound = 0
    if findExe("xcrun").len > 0:
      let probe = execCmdEx("xcrun -sdk macosx -f metal")
      if probe.exitCode == 0: gMetalCompilerFound = 1
    if gMetalCompilerFound == 0:
      echo "INFO MSL offline compile: no `xcrun -sdk macosx metal`; compiling through the Metal framework instead"
  if gMetalCompilerFound == 0:
    when defined(macosx):
      # The Metal framework compiler (newLibraryWithSource, MSL 2.3) of this machine's GPU.
      let check = metalLibraryCompileCheck(source)
      if not check.ok:
        raise newException(ValueError, "the Metal framework compiler rejected the simdgroup_matrix micro shader " & tag & ": " & check.error)
      echo "PASS MSL framework compile ", tag
    return
  let path = getTempDir() / ("fir_simd_" & $getCurrentProcessId() & ".metal")
  let air = path & ".air"
  writeFile(path, source)
  let compile = startProcess("xcrun", args = ["-sdk", "macosx", "metal", "-std=macos-metal2.3", "-c", path, "-o", air],
    options = {poUsePath, poStdErrToStdOut})
  let output = compile.outputStream.readAll()
  let code = compile.waitForExit()
  compile.close()
  if fileExists(path): removeFile(path)
  if fileExists(air): removeFile(air)
  if code != 0:
    raise newException(ValueError, "the Metal compiler rejected the simdgroup_matrix micro shader " & tag & ": " & output)
  echo "PASS MSL offline compile ", tag

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
    raise newException(ValueError, name & ": CPU VM rejected the program: " & cpu.cpuLastErrorMessage())
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
    raise newException(ValueError, name & ": CPU VM rejected the program: " & cpu.cpuLastErrorMessage())
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
  if not cpu.vmRun(program.words, mem, newSeq[float32](), g.threads, max(4, g.tgBytes div 4), max(1, g.tg)):
    raise newException(ValueError, "CPU VM rejected the program: " & cpu.cpuLastErrorMessage())
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

proc chkResidentQ4(name: string; g: Gen; a, packed, expected: seq[float32]; tol = 1e-3'f32) =
  let program = assemble(g.body, 2, 1, @[a.len, packed.len], @[expected.len], 1e-5'f32, slotLayout = true)
  var output = newSeq[float32](expected.len)
  var views: array[5, cpu.CpuVmView]
  views[0] = cpu.CpuVmView(base: cast[ptr float32](addr output[0]), elements: output.len, allocationElements: output.len)
  views[1] = cpu.CpuVmView(base: cast[ptr float32](unsafeAddr packed[0]), elements: packed.len, allocationElements: packed.len)
  views[2] = views[1]
  views[3] = cpu.CpuVmView(base: cast[ptr float32](unsafeAddr a[0]), elements: a.len, allocationElements: a.len)
  if not cpu.vmRunBound(program.words, views, g.threads, max(4, g.tgBytes div 4), max(1, g.tg)):
    raise newException(ValueError, name & ": resident CPU VM rejected the program: " & cpu.cpuLastErrorMessage())
  for i in 0 ..< expected.len:
    if not near(output[i], expected[i], tol):
      raise newException(ValueError, name & " resident mismatch at " & $i & ": got " & $output[i] & " want " & $expected[i])
  echo "PASS resident Q4 kernel ", name

proc fp4kmCase(M, K, N: int; seed: int): tuple[a, blob, want: seq[float32]] =
  ## A random-looking FP4-KM problem in the fp4kmLayout blob and its scalar FP32 reference:
  ## byte j of a 32-value group holds element j (low nibble) and j+16 (high nibble),
  ## weight = lut[nibble] * lut[16 + scale byte].
  let nG = K div 32
  let lay = fp4kmLayout(N, K)
  var bytes = newSeq[uint8](lay.blobWords * 4)
  var nibs = newSeq[int](N*K)
  var sbytes = newSeq[int](N*nG)
  for n in 0 ..< N:
    for kk in 0 ..< K:
      nibs[n*K + kk] = (n*11 + kk*7 + kk div 3 + seed) mod 16
  for i in 0 ..< N*nG: sbytes[i] = (i*37 + 11 + 5*seed) mod 256
  for n in 0 ..< N:
    for g in 0 ..< nG:
      for j in 0 ..< 16:
        let lo = nibs[n*K + g*32 + j]
        let hi = nibs[n*K + g*32 + j + 16]
        bytes[n*(K div 2) + g*16 + j] = uint8(lo or (hi shl 4))
      bytes[N*(K div 2) + n*nG + g] = uint8(sbytes[n*nG + g])
  var lut = newSeq[float32](272)
  const fp4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
  for i in 0 ..< 16: lut[i] = float32(if i < 8: fp4[i] else: -fp4[i - 8])
  for i in 0 ..< 256: lut[16 + i] = float32(float(i) * 0.0009765625 + 0.003)
  result.blob = bytesAsFloats(bytes)
  doAssert result.blob.len == lay.lutWord
  result.blob.add lut
  result.a = newSeq[float32](M*K)
  for i in 0 ..< result.a.len: result.a[i] = float32((i*13 + 7*seed) mod 41 - 20) / 31
  result.want = newSeq[float32](M*N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0
      for kk in 0 ..< K:
        let w = lut[nibs[n*K + kk]] * lut[16 + sbytes[n*nG + kk div 32]]
        acc += float(result.a[m*K + kk]) * float(w)
      result.want[m*N + n] = float32(acc)

proc chkResidentFp4(name: string; g: Gen; a, blob, expected: seq[float32]; rel = 1e-3) =
  ## Runs a slot-layout FP4-KM kernel on the CPU VM: out -> slot 0, blob -> slots 1 and 2,
  ## activations -> slot 3. Each output must be within `rel` of the reference (relative to
  ## max(1, |reference|)).
  let program = assemble(g.body, 2, 1, @[a.len, blob.len], @[expected.len], 1e-5'f32, slotLayout = true)
  var output = newSeq[float32](expected.len)
  var views: array[5, cpu.CpuVmView]
  views[0] = cpu.CpuVmView(base: cast[ptr float32](addr output[0]), elements: output.len, allocationElements: output.len)
  views[1] = cpu.CpuVmView(base: cast[ptr float32](unsafeAddr blob[0]), elements: blob.len, allocationElements: blob.len)
  views[2] = views[1]
  views[3] = cpu.CpuVmView(base: cast[ptr float32](unsafeAddr a[0]), elements: a.len, allocationElements: a.len)
  if not cpu.vmRunBound(program.words, views, g.threads, max(4, g.tgBytes div 4), max(1, g.tg)):
    raise newException(ValueError, name & ": resident CPU VM rejected the program: " & cpu.cpuLastErrorMessage())
  for i in 0 ..< expected.len:
    if abs(float(output[i]) - float(expected[i])) > rel * max(1.0, abs(float(expected[i]))):
      raise newException(ValueError, name & " resident mismatch at " & $i & ": got " & $output[i] & " want " & $expected[i])
  echo "PASS resident FP4 kernel ", name

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
    chkN("add_rmsnorm_par", genAddRmsnormPar(n, 1e-5), @[a, b, gamma], @[sum, normed])
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

  # per-element RoPE (Metal-shaped): one thread per (row, head, frequency)
  for (pos, T, theta, scale) in [(0.0, 1, 10000.0, 1.0), (137.0, 1, 10000.0, 1.0), (5.0, 4, 10000.0, 1.0),
                                 (96.0, 3, 500000.0, 0.25)]:
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
    chk("rope_pos_par T=" & $T & " pos " & $pos, genRopePosPar(T, Hn, D, pow(theta, -1.0 / float(half)), scale),
      @[xs, @[float32(pos)]], want, 5e-3'f32)

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
    chkResidentQ4("scalar packed", genMatmulQ4(M, K, N, gs, nG, 0, true, false, true),
        a, bytesAsFloats(bytes) & scales & zeros, want)

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
    let resident = genMatmulQ4Par(M, K, N, gs, nG, T, false, true)
    chkResidentQ4("parallel K" & $K & " tg" & $resident.tg, resident,
        a, bytesAsFloats(bytes) & scales & zeros, want)

  # word-parallel Q4 GEMV: one output per threadgroup, 8 nibbles per uloadu32.
  # Use odd N and nonuniform activations / per-group quantization parameters so
  # swapped nibble order, row stride, and group correction errors are visible.
  for (K, gs, T) in [(256, 32, 16), (256, 64, 32), (256, 128, 64), (256, 256, 16),
                     (384, 32, 32), (384, 64, 64), (384, 128, 16),
                     (384, 32, 24),
                     (4096, 32, 64), (4096, 64, 16), (4096, 128, 32), (4096, 256, 64),
                     (11008, 128, 32)]:
    let N = 5
    let nG = (K + gs - 1) div gs
    let bpr = (K + 1) div 2
    var q = newSeq[int](N*K)
    var bytes = newSeq[uint8](N*bpr)
    for n in 0 ..< N:
      for kk in 0 ..< K:
        # Distinct neighboring low/high nibbles and rows.
        q[n*K + kk] = (n*11 + kk*7 + kk div 3 + 2) mod 16
        let b = n*bpr + kk div 2
        bytes[b] = bytes[b] or uint8(q[n*K + kk] shl (if kk mod 2 == 0: 0 else: 4))
    let scales = fill(N*nG, 0.17, 0.11)
    let zeros = fill(N*nG, 6.3, 2.7)
    let a = fill(K, 0.43, 1.2)
    var want = newSeq[float32](N)
    for n in 0 ..< N:
      var sum = 0.0'f32
      for kk in 0 ..< K:
        let gi = n*nG + kk div gs
        sum += a[kk] * ((float32(q[n*K + kk]) - zeros[gi]) * scales[gi])
      want[n] = sum
    let packed = bytesAsFloats(bytes) & scales & zeros
    let g = genMatmulQ4WordPar(K, N, gs, nG, T)
    doAssert g.tg >= 2 and (g.tg and (g.tg - 1)) == 0 and g.tg <= T,
      "word-parallel Q4 threadgroup width is invalid for K=" & $K & " gs=" & $gs & " T=" & $T
    doAssert "uloadu32 " in g.body, "word-parallel Q4 must load packed words"
    doAssert "simd" notin g.body, "word-parallel Q4 must not use simd instructions"
    chk("matmul_q4_wordpar.K" & $K & ".gs" & $gs & ".T" & $T, g, @[a, packed], want, 2e-3'f32)
    # Exercise resident weight loads on representative partial and full groups.
    if (K == 384 and gs == 256) or (K == 4096 and gs == 128):
      let resident = genMatmulQ4WordPar(K, N, gs, nG, T, true)
      chkResidentQ4("wordpar resident K" & $K & " gs" & $gs, resident, a, packed, want, 2e-3'f32)

  # Unsupported word layouts must retain the generic correct path. These include
  # a short odd-K tail and a group size that cannot be decomposed into 8-nibble words.
  for (K, N, gs, T) in [(13, 3, 6, 16), (30, 3, 10, 32), (384, 3, 256, 32)]:
    let nG = (K + gs - 1) div gs
    let bpr = (K + 1) div 2
    var q = newSeq[int](N*K)
    var bytes = newSeq[uint8](N*bpr)
    for n in 0 ..< N:
      for kk in 0 ..< K:
        q[n*K + kk] = (n*9 + kk*5 + 1) mod 16
        let b = n*bpr + kk div 2
        bytes[b] = bytes[b] or uint8(q[n*K + kk] shl (if kk mod 2 == 0: 0 else: 4))
    let scales = fill(N*nG, 0.29, 0.13)
    let zeros = fill(N*nG, 7.1, 1.8)
    let a = fill(K, 0.91, 0.8)
    var want = newSeq[float32](N)
    for n in 0 ..< N:
      for kk in 0 ..< K:
        let gi = n*nG + kk div gs
        want[n] += a[kk] * ((float32(q[n*K + kk]) - zeros[gi]) * scales[gi])
    let g = genMatmulQ4WordPar(K, N, gs, nG, T)
    chk("matmul_q4_wordpar_fallback.K" & $K & ".gs" & $gs, g,
        @[a, bytesAsFloats(bytes) & scales & zeros], want, 2e-3'f32)

  # simdgroup Q4 GEMV (the Metal matmul_q4_simd structure): lanes stride over K, simd_sum combines
  for (K, N, gs, rows, sg) in [(256, 8, 128, 4, 2), (512, 4, 32, 4, 1), (128, 6, 64, 2, 2), (768, 4, 128, 4, 1), (3072, 4, 128, 4, 1)]:
    let nG = K div gs
    let bpr = K div 2
    var q = newSeq[int](N*K)
    var bytes = newSeq[uint8](N*bpr)
    for i in 0 ..< N*K: q[i] = (i*5 + 3) mod 16
    for n in 0 ..< N:
      for kk in 0 ..< K:
        let b = n*bpr + kk div 2
        bytes[b] = bytes[b] or uint8(q[n*K + kk] shl (if kk mod 2 == 0: 0 else: 4))
    let scales = fill(N*nG, 0.4, 0.3)
    let zeros = fill(N*nG, 1.1, 2.0)
    let a = fill(K, 0.8)
    var want = newSeq[float32](N)
    for n in 0 ..< N:
      var s = 0.0'f32
      for kk in 0 ..< K:
        let gi = n*nG + kk div gs
        s += a[kk] * ((float32(q[n*K + kk]) - zeros[gi]) * scales[gi])
      want[n] = s
    for packs in [1, 2]:
      if K mod (8 * packs) != 0 or gs < 8 * packs: continue
      let g = genMatmulQ4Simd(K, N, gs, nG, rows, sg, packs)
      doAssert g.tg == sg * 32, "simd variant not selected"
      chk("matmul_q4_simd K" & $K & " gs" & $gs & " rows" & $rows & " sg" & $sg & " packs" & $packs, g,
          @[a, bytesAsFloats(bytes) & scales & zeros], want, 1e-3'f32)
      let resident = genMatmulQ4Simd(K, N, gs, nG, rows, sg, packs, true)
      chkResidentQ4("simd K" & $K & " packs" & $packs, resident,
          a, bytesAsFloats(bytes) & scales & zeros, want)

  # simdgroup_matrix Q4 GEMM (the Metal matmul_q4_mm_sg structure): full and partial tiles
  for (M, K, N, gs) in [(32, 64, 32, 32), (40, 64, 40, 32), (3, 96, 5, 32), (33, 128, 70, 128)]:
    let nG = K div gs
    let bpr = K div 2
    var q = newSeq[int](N*K)
    var bytes = newSeq[uint8](N*bpr)
    for i in 0 ..< N*K: q[i] = (i*7 + 1) mod 16
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
        var sum = 0.0'f32
        for kk in 0 ..< K:
          let gi = n*nG + kk div gs
          sum += a[m*K + kk] * ((float32(q[n*K + kk]) - zeros[gi]) * scales[gi])
        want[m*N + n] = sum
    let g = genMatmulQ4Mm(M, K, N, gs, nG)
    doAssert g.tg == 128, "mm variant not selected"
    chk("matmul_q4_mm M" & $M & " K" & $K & " N" & $N, g,
        @[a, bytesAsFloats(bytes) & scales & zeros], want, 1e-3'f32)
    let resident = genMatmulQ4Mm(M, K, N, gs, nG, true)
    chkResidentQ4("matrix M" & $M & " K" & $K, resident,
        a, bytesAsFloats(bytes) & scales & zeros, want)

  # tiled FP32 Q4 GEMM (the Metal matmul_q4_gemm_kernel structure): full and partial tiles
  for (M, K, N, gs) in [(32, 64, 64, 32), (40, 64, 70, 32), (3, 96, 5, 32), (65, 128, 130, 128), (16, 64, 33, 64)]:
    let nG = K div gs
    let bpr = K div 2
    var q = newSeq[int](N*K)
    var bytes = newSeq[uint8](N*bpr)
    for i in 0 ..< N*K: q[i] = (i*7 + 1) mod 16
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
        var sum = 0.0'f32
        for kk in 0 ..< K:
          let gi = n*nG + kk div gs
          sum += a[m*K + kk] * ((float32(q[n*K + kk]) - zeros[gi]) * scales[gi])
        want[m*N + n] = sum
    for (bm, bn, tm, tn) in [(32, 64, 4, 4), (64, 64, 8, 4), (32, 128, 4, 8), (64, 128, 8, 8), (64, 64, 4, 4)]:
      let g = genMatmulQ4Gemm(M, K, N, gs, nG, bm, bn, tm, tn)
      doAssert g.tg == (bm div tm) * (bn div tn), "gemm variant not selected"
      chk("matmul_q4_gemm " & $bm & "x" & $bn & " reg " & $tm & "x" & $tn & " M" & $M & " K" & $K & " N" & $N, g,
          @[a, bytesAsFloats(bytes) & scales & zeros], want, 1e-3'f32)
      let resident = genMatmulQ4Gemm(M, K, N, gs, nG, bm, bn, tm, tn, true)
      chkResidentQ4("tiled M" & $M & " K" & $K & " N" & $N & " tile " & $bm & "x" & $bn,
          resident, a, bytesAsFloats(bytes) & scales & zeros, want)

  # Vulkan cooperative-matrix Q4 GEMM is a custom micro-VM op. The production
  # Vulkan JIT lowers it to KHR_cooperative_matrix; the CPU VM executes the
  # same instruction as a scalar reference for bytecode/layout parity.
  block:
    let M = 16
    let K = 64
    let N = 64
    let gs = 32
    let nG = K div gs
    let bpr = K div 2
    var q = newSeq[int](N*K)
    var bytes = newSeq[uint8](N*bpr)
    for n in 0 ..< N:
      for kk in 0 ..< K:
        q[n*K + kk] = (n*11 + kk*7 + kk div 3 + 2) mod 16
        let b = n*bpr + kk div 2
        bytes[b] = bytes[b] or uint8(q[n*K + kk] shl (if kk mod 2 == 0: 0 else: 4))
    let scales = fill(N*nG, 0.17, 0.11)
    let zeros = fill(N*nG, 6.3, 2.7)
    var a = newSeq[float32](M*K)
    for i in 0 ..< a.len: a[i] = float32((i*13 mod 41) - 20) / 31
    var want = newSeq[float32](M*N)
    for m in 0 ..< M:
      for n in 0 ..< N:
        var sum = 0.0'f32
        for kk in 0 ..< K:
          let gi = n*nG + kk div gs
          sum += a[m*K + kk] * ((float32(q[n*K + kk]) - zeros[gi]) * scales[gi])
        want[m*N + n] = sum
    let packed = bytesAsFloats(bytes) & scales & zeros
    let g = genMatmulQ4TensorCore(M, K, N, gs, nG)
    doAssert g.threads == 128 and g.tg == 128 and "q4coop " in g.body
    chkResidentQ4("tensor_core_micro M16 K64 N64", g, a, packed, want)
    # Row tiles follow the device and the rows of a dispatch: the tallest tile whose grid
    # still gives each SM/CU two workgroups, never more than twice the rows it holds.
    doAssert q4CoopRowTile(16, 64) == 16
    doAssert q4CoopRowTile(64, 4096) == 32            # unknown device: 64 units
    doAssert q4CoopRowTile(64, 11008) == 64
    doAssert q4CoopRowTile(64, 4096, 40) == 32        # Tesla T4: 40 SMs
    doAssert q4CoopRowTile(64, 11008, 40) == 64
    doAssert q4CoopRowTile(20, 11008, 40) == 32       # a short last block
    doAssert q4CoopRowTile(64, 4096, 16) == 64        # a small GPU fills with tall tiles
    doAssert q4CoopRowTile(64, 4096, 40, 4) == 16     # more workgroups per SM requested
    doAssert q4CoopRowTile(512, 4096, 40) == 64
    doAssert genMatmulQ4TensorCore(64, 4096, 4096, 128, 32).threads == 2 * 64 * 128
    doAssert genMatmulQ4TensorCore(64, 4096, 11008, 128, 32).threads == 172 * 128

    # The CPU reference does not depend on the grid the runtime picks.
    block:
      let M2 = 40
      let N2 = 80
      var bytes2 = newSeq[uint8](N2*bpr)
      var q2 = newSeq[int](N2*K)
      for n in 0 ..< N2:
        for kk in 0 ..< K:
          q2[n*K + kk] = (n*5 + kk*3 + kk div 5 + 1) mod 16
          let b = n*bpr + kk div 2
          bytes2[b] = bytes2[b] or uint8(q2[n*K + kk] shl (if kk mod 2 == 0: 0 else: 4))
      let scales2 = fill(N2*nG, 0.41, 0.07)
      let zeros2 = fill(N2*nG, 4.9, 1.3)
      var a2 = newSeq[float32](M2*K)
      for i in 0 ..< a2.len: a2[i] = float32((i*7 mod 29) - 14) / 23
      var want2 = newSeq[float32](M2*N2)
      for m in 0 ..< M2:
        for n in 0 ..< N2:
          var sum = 0.0'f32
          for kk in 0 ..< K:
            let gi = n*nG + kk div gs
            sum += a2[m*K + kk] * ((float32(q2[n*K + kk]) - zeros2[gi]) * scales2[gi])
          want2[m*N2 + n] = sum
      let packed2 = bytesAsFloats(bytes2) & scales2 & zeros2
      var g2 = genMatmulQ4TensorCore(M2, K, N2, gs, nG)
      for tile in Q4CoopRowTiles:
        g2.threads = ((M2 + tile - 1) div tile) * 2 * 128
        chkResidentQ4("tensor_core_micro M40 K64 N80 grid for tile " & $tile, g2, a2, packed2, want2)

    let glslc = findExe("glslc")
    for shape in [(M, K, N, gs, nG, packed.len), (64, 4096, 4096, 128, 32, 0), (64, 11008, 4096, 128, 86, 0),
                  (64, 4096, 11008, 128, 32, 0), (49, 4096, 11008, 128, 32, 0)]:
      let (sm, sk, sn, sgs, sng, spacked) = shape
      let sg = genMatmulQ4TensorCore(sm, sk, sn, sgs, sng)
      let weightFloats = if spacked > 0: spacked else: (sn * (sk div 2) + 3) div 4 + 2 * sn * sng
      let program = assemble(sg.body, 2, 1, @[sm * sk, weightFloats], @[sm * sn], 1e-5'f32, slotLayout = true)
      doAssert ("const uint BM = " & $q4CoopRowTile(sm, sn) & "u;") in lowerBytecodeToGlsl(program, "q4_tc_default")
      # Metal lowers the same op to MSL simdgroup_matrix (Apple GPUs).
      doAssert ("constant constexpr uint BM = " & $q4CoopRowTile(sm, sn) & "u;") in lowerBytecodeToMetal(program, "q4_tc_metal")
      for tile in Q4CoopRowTiles:
        let msl = lowerCoopGemmMetal(program, "q4_simd_source_test", tile)
        let tag = "M" & $sm & " K" & $sk & " N" & $sn & " tile " & $tile
        doAssert msl == lowerQ4CoopMetal(program, "q4_simd_source_test", tile), tag
        doAssert "kernel void q4_simd_source_test(" in msl, tag
        doAssert "#include <metal_simdgroup_matrix>" in msl and "simdgroup_float8x8" in msl, tag
        doAssert "simdgroup_load(" in msl and "simdgroup_multiply_accumulate(" in msl and "simdgroup_store(" in msl, tag
        doAssert ("constant constexpr uint BM = " & $tile & "u;") in msl, tag
        doAssert ("constant constexpr uint Q4_M = " & $sm & "u;") in msl and ("constant constexpr uint Q4_K = " & $sk & "u;") in msl, tag
        doAssert ("constant constexpr uint Q4_N = " & $sn & "u;") in msl and ("constant constexpr uint Q4_GROUP_SIZE = " & $sgs & "u;") in msl, tag
        doAssert ("constant constexpr uint Q4_N_GROUPS = " & $sng & "u;") in msl and "@" notin msl, tag
        mslCompileCheck(msl, "Q4 " & tag)
        echo "PASS Q4 simdgroup_matrix micro shader lowering ", tag
      for (tile, frags) in [(16, 1), (32, 1), (64, 1), (16, 2), (32, 2), (64, 2)]:
        let shader = lowerQ4CoopGlsl(program, "q4_tensor_core_source_test", tile, frags)
        let tag = "M" & $sm & " K" & $sk & " N" & $sn & " tile " & $tile & " fragments " & $frags
        doAssert ("const uint FRAGS = " & $frags & "u;") in shader, tag
        doAssert "GL_KHR_cooperative_matrix" in shader
        doAssert "coopMatMulAdd" in shader
        doAssert "sessionMatmulQ4" notin shader
        doAssert ("const uint BM = " & $tile & "u;") in shader, tag
        echo "PASS Q4 Tensor Core micro shader lowering ", tag
        if glslc.len > 0:
          let shaderPath = getTempDir() / ("fir_q4_tc_" & $getCurrentProcessId() & ".comp")
          let spvPath = shaderPath & ".spv"
          writeFile(shaderPath, shader)
          let compile = startProcess(glslc, args = ["--target-env=vulkan1.3", "-o", spvPath, shaderPath],
            options = {poUsePath, poStdErrToStdOut})
          let compileOutput = compile.outputStream.readAll()
          let compileCode = compile.waitForExit()
          compile.close()
          if fileExists(shaderPath): removeFile(shaderPath)
          if fileExists(spvPath): removeFile(spvPath)
          if compileCode != 0:
            raise newException(ValueError, "glslc rejected the Q4 Tensor Core micro shader " & tag & ": " & compileOutput)
          echo "PASS Q4 Tensor Core micro shader glslc compile ", tag

  # Vulkan cooperative-matrix FP4-KM GEMM (Codestral Mamba-2 weights) is a custom micro-VM op
  # too: same tiles, slots and grid as the Q4 one. The CPU VM runs it as a scalar reference.
  block:
    for (M, K, N) in [(16, 64, 64), (40, 128, 80), (3, 64, 17), (64, 256, 130)]:
      let (a, blob, want) = fp4kmCase(M, K, N, M + N)
      var g = genFp4GemmTensorCore(M, K, N)
      let tag = "M" & $M & " K" & $K & " N" & $N
      doAssert g.tg == 128 and ("fp4coop " & $M & " " & $K & " " & $N & " 32 " & $(K div 32)) == g.body, tag
      doAssert g.threads == ((M + q4CoopRowTile(M, N) - 1) div q4CoopRowTile(M, N)) * ((N + 63) div 64) * 128, tag
      chkResidentFp4("tensor_core_micro " & tag, g, a, blob, want)
      # The CPU reference does not depend on the grid the runtime picks.
      for tile in Q4CoopRowTiles:
        g.threads = ((M + tile - 1) div tile) * ((N + 63) div 64) * 128
        chkResidentFp4("tensor_core_micro " & tag & " grid for tile " & $tile, g, a, blob, want)
    # The same FP4-KM blob through the generic lane-per-column kernel gives the same product.
    block:
      let (a, blob, want) = fp4kmCase(2, 64, 5, 3)
      chk("fp4_gemm generic on the tensor core case", genFp4Gemm(2, 64, 5), @[a, blob], want, 2e-3'f32)
    var rejected = 0
    for (M, K, N) in [(0, 64, 64), (4, 96, 64), (4, 64, 8)]:
      try:
        discard genFp4GemmTensorCore(M, K, N)
      except GenError:
        inc rejected
    doAssert rejected == 3, "FP4 Tensor Core shape checks"
    # Codestral Mamba-2 7B prefill shapes, M = 64 rows: in_proj K=4096 N=18560, out_proj K=8192 N=4096.
    # Unknown device (64 units): in_proj takes 64-row tiles (290 column tiles already fill it),
    # out_proj 32-row tiles (2 x 64 workgroups); a Tesla T4 (40 SMs) picks the same tiles.
    doAssert q4CoopRowTile(64, 18560) == 64 and q4CoopRowTile(64, 4096) == 32
    doAssert q4CoopRowTile(64, 18560, 40) == 64 and q4CoopRowTile(64, 4096, 40) == 32
    doAssert genFp4GemmTensorCore(64, 4096, 18560).threads == 290 * 128
    doAssert genFp4GemmTensorCore(64, 8192, 4096).threads == 2 * 64 * 128
    doAssert genFp4GemmTensorCore(64, 4096, 18560).tg == 128
    doAssert genFp4GemmTensorCore(64, 8192, 4096).tg == 128
    echo "PASS FP4 Tensor Core grid for Codestral in_proj (290 x 128 threads) and out_proj (128 x 128 threads)"

    let glslc = findExe("glslc")
    for (sm, sk, sn) in [(16, 64, 64), (64, 4096, 18560), (64, 8192, 4096), (49, 4096, 18560), (3, 64, 17)]:
      let sg = genFp4GemmTensorCore(sm, sk, sn)
      let blobFloats = fp4kmLayout(sn, sk).totalWords
      let program = assemble(sg.body, 2, 1, @[sm * sk, blobFloats], @[sm * sn], 1e-5'f32, slotLayout = true)
      doAssert program.words.len == 7 and (program.words[0] and 0xFF'u32) == OP_FP4_COOP_GEMM
      doAssert program.words[1 .. 5] == @[uint32(sm), uint32(sk), uint32(sn), 32'u32, uint32(sk div 32)]
      doAssert isCoopGemmProgram(program) and not isQ4CoopProgram(program)
      let shape = coopGemmShape(program)
      doAssert shape.M == sm and shape.K == sk and shape.N == sn and shape.groupSize == 32 and shape.nGroups == sk div 32
      doAssert opImmediateWords(OP_FP4_COOP_GEMM) == 5
      let defaultShader = lowerBytecodeToGlsl(program, "fp4_tc_default")
      doAssert ("const uint BM = " & $q4CoopRowTile(sm, sn) & "u;") in defaultShader
      doAssert ("const uint LUT_BASE = " & $fp4CoopLutWord(sk, sn) & "u;") in defaultShader
      doAssert ("const uint SCALE_BASE = " & $(sn * (sk div 2)) & "u;") in defaultShader
      # The legacy shape op expands to the tile GEMM micro-kernel program the lowering walks.
      doAssert fp4TileProgram(program).words == buildFp4TileGemmProgram(sm, sk, sn, sn * (sk div 2), fp4CoopLutWord(sk, sn)).words
      # Metal lowers the same op to MSL simdgroup_matrix (Apple GPUs) or register tiles (other GPUs).
      let metalDefault = lowerBytecodeToMetal(program, "fp4_tc_metal")
      doAssert ("constant constexpr uint BM = " & $q4CoopRowTile(sm, sn) & "u;") in metalDefault
      for tile in Q4CoopRowTiles:
        let tiledMsl = lowerFp4TiledMetal(program, "fp4_tiled_source_test", tile)
        let tiledGlsl = lowerFp4TiledGlsl(program, "fp4_tiled_source_test", tile)
        let tag = "M" & $sm & " K" & $sk & " N" & $sn & " tile " & $tile
        doAssert "simdgroup" notin tiledMsl and "coopmat" notin tiledGlsl, tag
        doAssert ("constant constexpr uint TM = " & $(tile div 8) & "u;") in tiledMsl, tag
        mslCompileCheck(tiledMsl, "FP4 register tiles " & tag)
        echo "PASS FP4 register-tiled micro shader lowering (MSL) ", tag
      for tile in Q4CoopRowTiles:
        let msl = lowerCoopGemmMetal(program, "fp4_simd_source_test", tile)
        let tag = "M" & $sm & " K" & $sk & " N" & $sn & " tile " & $tile
        doAssert msl == lowerFp4CoopMetal(program, "fp4_simd_source_test", tile), tag
        doAssert "kernel void fp4_simd_source_test(" in msl, tag
        doAssert "#include <metal_simdgroup_matrix>" in msl and "simdgroup_float8x8" in msl, tag
        doAssert "simdgroup_load(" in msl and "simdgroup_multiply_accumulate(" in msl and "simdgroup_store(" in msl, tag
        doAssert ("constant constexpr uint BM = " & $tile & "u;") in msl, tag
        doAssert ("constant constexpr uint T_M = " & $sm & "u;") in msl and ("constant constexpr uint T_K = " & $sk & "u;") in msl, tag
        doAssert ("constant constexpr uint T_N = " & $sn & "u;") in msl and ("constant constexpr uint T_NG = " & $(sk div 32) & "u;") in msl, tag
        doAssert "@" notin msl and "sessionFp4Gemm" notin msl, tag
        mslCompileCheck(msl, "FP4 " & tag)
        echo "PASS FP4 simdgroup_matrix micro shader lowering ", tag
      for (tile, frags) in [(16, 1), (32, 1), (64, 1), (16, 2), (32, 2), (64, 2)]:
        let shader = lowerCoopGemmGlsl(program, "fp4_tensor_core_source_test", tile, frags)
        let tag = "M" & $sm & " K" & $sk & " N" & $sn & " tile " & $tile & " fragments " & $frags
        doAssert shader == lowerFp4CoopGlsl(program, "fp4_tensor_core_source_test", tile, frags), tag
        doAssert ("const uint FRAGS = " & $frags & "u;") in shader, tag
        doAssert ("const uint BM = " & $tile & "u;") in shader, tag
        doAssert ("const uint T_K = " & $sk & "u;") in shader and ("const uint T_N = " & $sn & "u;") in shader, tag
        doAssert "GL_KHR_cooperative_matrix" in shader
        doAssert "coopMatMulAdd" in shader
        doAssert "sessionFp4Gemm" notin shader and "@" notin shader, tag
        echo "PASS FP4 Tensor Core micro shader lowering ", tag
        if glslc.len > 0:
          let shaderPath = getTempDir() / ("fir_fp4_tc_" & $getCurrentProcessId() & ".comp")
          let spvPath = shaderPath & ".spv"
          writeFile(shaderPath, shader)
          let compile = startProcess(glslc, args = ["--target-env=vulkan1.3", "-o", spvPath, shaderPath],
            options = {poUsePath, poStdErrToStdOut})
          let compileOutput = compile.outputStream.readAll()
          let compileCode = compile.waitForExit()
          compile.close()
          if fileExists(shaderPath): removeFile(shaderPath)
          if fileExists(spvPath): removeFile(spvPath)
          if compileCode != 0:
            raise newException(ValueError, "glslc rejected the FP4 Tensor Core micro shader " & tag & ": " & compileOutput)
          echo "PASS FP4 Tensor Core micro shader glslc compile ", tag

  # Vulkan Q4 decode GEMV with the native decode algorithm is a shape-op micro-kernel too.
  # The CPU VM runs the same instruction as a scalar reference; the Vulkan JIT lowers it to
  # the subgroup or shared-memory shader the runtime picks per device.
  block:
    for (M, K, N, gs) in [(1, 64, 40, 32), (2, 72, 24, 8), (1, 128, 33, 64)]:
      let nG = K div gs
      let bpr = K div 2
      var q = newSeq[int](N*K)
      var bytes = newSeq[uint8](N*bpr)
      for n in 0 ..< N:
        for kk in 0 ..< K:
          q[n*K + kk] = (n*7 + kk*5 + kk div 7 + 3) mod 16
          let b = n*bpr + kk div 2
          bytes[b] = bytes[b] or uint8(q[n*K + kk] shl (if kk mod 2 == 0: 0 else: 4))
      let scales = fill(N*nG, 0.23, 0.05)
      let zeros = fill(N*nG, 7.1, 1.9)
      var a = newSeq[float32](M*K)
      for i in 0 ..< a.len: a[i] = float32((i*11 mod 37) - 18) / 27
      var want = newSeq[float32](M*N)
      for m in 0 ..< M:
        for n in 0 ..< N:
          var sum = 0.0'f32
          for kk in 0 ..< K:
            let gi = n*nG + kk div gs
            sum += a[m*K + kk] * ((float32(q[n*K + kk]) - zeros[gi]) * scales[gi])
          want[m*N + n] = sum
      let packed = bytesAsFloats(bytes) & scales & zeros
      var g = genMatmulQ4SubgroupMicro(M, K, N, gs, nG)
      let tag = "M" & $M & " K" & $K & " N" & $N & " group " & $gs
      doAssert g.tg == 128 and "q4gemv " in g.body, tag
      doAssert g.threads == q4GemvThreadCount(M, N, Q4GemvDefaultRows), tag
      for rows in Q4GemvRowChoices:
        g.threads = q4GemvThreadCount(M, N, rows)
        chkResidentQ4("subgroup_micro GEMV " & tag & " grid for rows " & $rows, g, a, packed, want)
    doAssert q4GemvRowsPerGroup(2) == 8
    doAssert q4GemvThreadCount(1, 4096, 2) == 512 * 128
    doAssert q4GemvThreadCount(1, 11008, 4) == 688 * 128
    doAssert q4GemvThreadCount(3, 40, 1) == 3 * 10 * 128
    doAssert not q4GemvSupported(130, 32, 5) and not q4GemvSupported(64, 4, 16)

    let glslc = findExe("glslc")
    for (sk, sn, sgs) in [(4096, 4096, 128), (11008, 4096, 128), (4096, 11008, 64), (4096, 32256, 32), (72, 24, 8)]:
      let sng = sk div sgs
      let sg = genMatmulQ4SubgroupMicro(1, sk, sn, sgs, sng)
      let weightFloats = (sn * (sk div 2) + 3) div 4 + 2 * sn * sng
      let program = assemble(sg.body, 2, 1, @[sk, weightFloats], @[sn], 1e-5'f32, slotLayout = true)
      doAssert "#define SHARED_REDUCE 1" in lowerBytecodeToGlsl(program, "q4_gemv_default")
      for rows in Q4GemvRowChoices:
        for sharedReduce in [true, false]:
          let shader = lowerQ4GemvGlsl(program, "q4_gemv_source_test", rows, sharedReduce)
          let tag = "K" & $sk & " N" & $sn & " group " & $sgs & " rows " & $rows & (if sharedReduce: " shared" else: " subgroup")
          doAssert ("#define ROWS " & $rows) in shader, tag
          doAssert ("#define VEC " & (if sk mod 32 == 0 and sgs mod 32 == 0: "4" else: "1")) in shader, tag
          doAssert "sessionMatmulQ4" notin shader
          if glslc.len > 0:
            let shaderPath = getTempDir() / ("fir_q4_gemv_" & $getCurrentProcessId() & ".comp")
            let spvPath = shaderPath & ".spv"
            writeFile(shaderPath, shader)
            let compile = startProcess(glslc, args = ["--target-env=vulkan1.1", "-o", spvPath, shaderPath],
              options = {poUsePath, poStdErrToStdOut})
            let compileOutput = compile.outputStream.readAll()
            let compileCode = compile.waitForExit()
            compile.close()
            if fileExists(shaderPath): removeFile(shaderPath)
            if fileExists(spvPath): removeFile(spvPath)
            if compileCode != 0:
              raise newException(ValueError, "glslc rejected the Q4 GEMV micro shader " & tag & ": " & compileOutput)
          echo "PASS Q4 subgroup GEMV micro shader ", tag

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
    if (H and (H - 1)) == 0:
      chk("flash_prefill_par", genAttentionDecodePar(H, HKV, D, maxSeq, 32, T, sc), @[q, kc, vc, @[float32(pos)]], o, 1e-3'f32)

  # tiled flash prefill: 16 query rows per threadgroup, keys streamed 16 at a time
  for (T, H, HKV, D, maxSeq, pos) in [(3, 4, 2, 8, 24, 2), (16, 4, 2, 8, 40, 0), (20, 4, 4, 16, 64, 5),
                                      (37, 6, 2, 8, 80, 30), (33, 2, 1, 24, 48, 15)]:
    let sc = 0.35'f32
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
    let g = genFlashPrefillTiled(T, H, HKV, D, maxSeq, sc)
    doAssert g.tg == 128 and g.threads == ((T + 15) div 16)*H*128, "tiled flash prefill not selected"
    chk("flash_prefill_tiled T" & $T & " H" & $H & " D" & $D & " pos" & $pos, g, @[q, kc, vc, @[float32(pos)]], o, 1e-3'f32)

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
  chk("rmsnorm_par.rows", genRmsnormPar(rows, cols, 1e-5), @[x, gamma], rn)
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
      if pm == 1:
        let sg = genMatmulNTSimd(pk, pn, 1)
        chk("matmul_nt_simd K" & $pk & " N" & $pn, sg, @[pa, pb], pw, 1e-3'f32)
    for (pm, pk, pn) in [(32, 64, 32), (40, 64, 40), (3, 96, 5), (33, 128, 70)]:
      let pa = fill(pm*pk, 0.3)
      let pb = fill(pn*pk, 0.8)
      var pw = newSeq[float32](pm*pn)
      for m in 0 ..< pm:
        for j in 0 ..< pn:
          var s = 0.0'f32
          for q in 0 ..< pk: s += pa[m*pk + q] * pb[j*pk + q]
          pw[m*pn + j] = s
      let mg = genMatmulNTMm(pm, pk, pn)
      doAssert mg.tg == 128, "mm variant not selected"
      chk("matmul_nt_mm M" & $pm & " K" & $pk & " N" & $pn, mg, @[pa, pb], pw, 1e-3'f32)
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
    # threadgroup version: in3 is the position (S - 1); also a longer cache than lanes
    chk("attention_decode_par", genAttentionDecodePar(H, HKV, D, maxSeq, 32), @[q, kc, vc, @[float32(S - 1)]], want)

  # KV scatter writes into its cache port
  block:
    let T = 2
    let HKV = 2
    let D = 3
    let maxSeq = 5
    let pos = 1
    let src = fill(T*HKV*D, 0.9)
    var cache = newSeq[float32](HKV*maxSeq*D)
    for par in [false, true]:
      let g = if par: genKvScatterPar(T, HKV, D, maxSeq) else: genKvScatter(T, HKV, D, maxSeq)
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

proc sideBySideTests() =
  ## fuseSideBySide: the fused launch must write what the kernels write when launched one by one.
  type Part = object
    g: Gen
    hiR, hiU: int
    data: seq[seq[float32]]
    outSizes: seq[int]
  proc part(g: Gen; data: seq[seq[float32]]; outSizes: seq[int]): Part =
    Part(g: g, hiR: genHiR, hiU: genHiU, data: data, outSizes: outSizes)
  proc launch(body: string; ins, outs, threads, tg, tgBytes: int; data: seq[seq[float32]];
              outSizes: seq[int]): seq[float32] =
    var inSize: seq[int]
    var mem: seq[float32]
    for d in data:
      inSize.add d.len
      for v in d: mem.add v
    let total = mem.len
    mem.setLen(total + sum(outSizes))
    let program = assemble(body, ins, outs, inSize, outSizes, 1e-5'f32)
    if not cpu.vmRun(program.words, mem, newSeq[float32](), threads, max(4, tgBytes div 4), tg):
      raise newException(ValueError, "side_by_side: CPU VM rejected the program: " & cpu.cpuLastErrorMessage())
    mem[total .. ^1]
  proc check(name: string; parts: seq[Part]; tg: int) =
    var bodies: seq[string]
    var threads, ins, outs, hiR, hiU, outSizes: seq[int]
    var data: seq[seq[float32]]
    var want: seq[float32]
    var tgBytes = 16
    for p in parts:
      want.add launch(p.g.body, p.g.ins, p.g.outs, p.g.threads, tg, p.g.tgBytes, p.data, p.outSizes)
      bodies.add p.g.body
      threads.add p.g.threads
      ins.add p.g.ins
      outs.add p.g.outs
      hiR.add p.hiR
      hiU.add p.hiU
      outSizes.add p.outSizes
      data.add p.data
      tgBytes = max(tgBytes, p.g.tgBytes)
    let body = fuseSideBySide(bodies, threads, ins, outs, hiR, hiU, tg)
    doAssert body.len > 0, name & ": the kernels were not fused"
    let got = launch(body, sum(ins), sum(outs), sum(threads), tg, tgBytes, data, outSizes)
    doAssert got.len == want.len
    for i in 0 ..< want.len:
      if not near(got[i], want[i]):
        raise newException(ValueError, "side_by_side " & name & " mismatch at " & $i & ": got " & $got[i] & " want " & $want[i])
    echo "PASS session kernel side_by_side ", name
  check("vecop+activation", @[
    part(genVecOp("add", 96), @[fill(96, 0.3), fill(96, 1.1)], @[96]),
    part(genActivation("sigmoid", 64), @[fill(64, 0.7, 2.0)], @[64])], 32)
  check("rmsnorm_par x2 (threadgroup ids)", @[
    part(genRmsnormPar(2, 64, 1e-5), @[fill(128, 0.2, 2.0), fill(64, 0.9)], @[128]),
    part(genRmsnormPar(2, 64, 1e-5), @[fill(128, 1.4, 0.5), fill(64, 0.1)], @[128])], 256)
  block:
    let a = genMatmulNTSimd(96, 8)
    let pa = part(a, @[fill(96, 0.4), fill(8 * 96, 1.7)], @[8])
    let b = genMatmulNTSimd(96, 16)
    let pb = part(b, @[fill(96, 0.8), fill(16 * 96, 0.3)], @[16])
    doAssert a.tg == b.tg and a.tg > 0
    check("matmul_nt_simd x2 (simdgroup ids)", @[pa, pb], a.tg)

sideBySideTests()
sessionTests2()
sessionTests()
main()
