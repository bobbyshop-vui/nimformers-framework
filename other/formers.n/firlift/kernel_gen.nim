## Micro-kernel generators: one VM program per session op.
##
## Every generator returns a kernel body (session_vm mnemonics, `@inN`/`@outN`
## port addresses) that computes what the matching `session*` call computes on
## Metal/Vulkan, with the same port order as the call's arguments. Dimensions are
## baked into the text, so the kernel is exact for any shape (no model constants).
## The lane id is `U0`; `R0` is reserved. Float literals are plain decimals.
##
## Lowering is lane-per-output with serial inner loops: correct on every VM
## backend (CPU, Metal, Vulkan). Threadgroup-parallel variants are an
## optimization layered on the same ports.

import std/[strutils, math]

type
  KB* = ref object
    lines*: seq[string]
    rbusy, ubusy: array[32, bool]

  GenError* = object of ValueError

proc fnum*(x: float): string =
  if x == floor(x) and abs(x) < 1e15: return $int64(x)
  result = formatFloat(x, ffDecimal, 14)
  result = result.strip(leading = false, chars = {'0'})
  if result.endsWith("."): result.add "0"

proc newKB*(): KB =
  result = KB()
  result.rbusy[0] = true
  result.ubusy[0] = true
  result.lines.add "rtou 0 0"

proc e*(k: KB; op: string; a: varargs[string, `$`]) =
  k.lines.add op & " " & a.join(" ")

proc r*(k: KB): int =
  for i in 1 .. 31:
    if not k.rbusy[i]:
      k.rbusy[i] = true
      return i
  raise newException(GenError, "kernel exceeds 31 float registers")

proc u*(k: KB): int =
  for i in 1 .. 31:
    if not k.ubusy[i]:
      k.ubusy[i] = true
      return i
  raise newException(GenError, "kernel exceeds 31 integer registers")

proc fr*(k: KB; regs: varargs[int]) =
  for x in regs: k.rbusy[x] = false

proc fu*(k: KB; regs: varargs[int]) =
  for x in regs: k.ubusy[x] = false

proc cf*(k: KB; v: float): int =
  result = k.r()
  k.e("loadconstf32", result, fnum(v))

proc cu*(k: KB; v: int): int =
  result = k.u()
  k.e("uconst", result, v)

proc forR*(k: KB; count: int; body: proc()) =
  k.e("forbegin", count)
  body()
  k.e("forend")

proc forN*(k: KB; n: int; body: proc()) =
  if n <= 0: return
  let c = k.cf(float(n))
  k.forR(c, body)
  k.fr(c)

proc cond*(k: KB; reg: int; body: proc()) =
  k.e("ifbegin", reg)
  body()
  k.e("ifend")

proc cp*(k: KB; dst, src: int) =
  k.e("mulconst", dst, src, 1)

proc text*(k: KB): string = k.lines.join("\n")

const NegLarge = -3.0e38

# ---- scalar functions on a register ---------------------------------------

proc unary*(k: KB; kind: string; x: int) =
  ## In place: R[x] = f(R[x]).
  case kind
  of "neg", "exp", "sigmoid", "rsqrt", "softplus": k.e(kind, x, x)
  of "relu":
    let z = k.cf(0)
    let c = k.r()
    k.e("cmplt", c, x, z)
    k.cond(c, proc() = k.cp(x, z))
    k.fr(z, c)
  of "abs":
    let z = k.cf(0)
    let c = k.r()
    k.e("cmplt", c, x, z)
    k.cond(c, proc() = k.e("neg", x, x))
    k.fr(z, c)
  of "silu":
    let s = k.r()
    k.e("sigmoid", s, x)
    k.e("mul", x, x, s)
    k.fr(s)
  of "gelu_quick":
    let s = k.r()
    k.e("mulconst", s, x, "1.702")
    k.e("sigmoid", s, s)
    k.e("mul", x, x, s)
    k.fr(s)
  of "tanh":
    k.e("mulconst", x, x, 2)
    k.e("sigmoid", x, x)
    k.e("mulconst", x, x, 2)
    k.e("addconst", x, x, -1)
  of "gelu_new":
    let s = k.r()
    k.e("mul", s, x, x)
    k.e("mul", s, s, x)
    k.e("mulconst", s, s, "0.044715")
    k.e("add", s, s, x)
    k.e("mulconst", s, s, "1.5957691216057308")
    k.e("sigmoid", s, s)
    k.e("mul", x, x, s)
    k.fr(s)
  of "gelu":
    # x * Phi(x); Phi from Abramowitz-Stegun 7.1.26 (|error| <= 1.5e-7).
    let z = k.r()
    let t = k.r()
    let p = k.r()
    let c = k.r()
    let zero = k.cf(0)
    let one = k.cf(1)
    k.cp(z, x)
    k.e("cmplt", c, z, zero)
    k.cond(c, proc() = k.e("neg", z, z))
    k.e("mulconst", z, z, "0.7071067811865476")
    k.e("mulconst", t, z, "0.3275911")
    k.e("addconst", t, t, 1)
    k.e("div", t, one, t)
    k.e("loadconstf32", p, "1.061405429")
    for a in ["-1.453152027", "1.421413741", "-0.284496736", "0.254829592"]:
      k.e("mul", p, p, t)
      k.e("addconst", p, p, a)
    k.e("mul", p, p, t)
    k.e("mul", z, z, z)
    k.e("neg", z, z)
    k.e("exp", z, z)
    k.e("mul", p, p, z)
    k.e("mulconst", p, p, "0.5")
    k.e("neg", t, p)
    k.e("addconst", t, t, 1)
    k.cond(c, proc() = k.cp(t, p))
    k.e("mul", x, x, t)
    k.fr(z, t, p, c, zero, one)
  of "sqrt":
    let one = k.cf(1)
    k.e("rsqrt", x, x)
    k.e("div", x, one, x)
    k.fr(one)
  of "log":
    # Eight square roots bring a positive float into a small interval, then atanh series.
    let s = k.r()
    let one = k.cf(1)
    for _ in 0 ..< 8:
      k.e("rsqrt", s, x)
      k.e("div", x, one, s)
    k.e("addconst", s, x, 1)
    k.e("addconst", x, x, -1)
    k.e("div", x, x, s)
    let z2 = k.r()
    let power = k.r()
    let acc = k.r()
    k.e("mul", z2, x, x)
    k.cp(power, x)
    k.cp(acc, x)
    for j in 1 ..< 12:
      k.e("mul", power, power, z2)
      k.e("mulconst", s, power, fnum(1.0 / float(2*j + 1)))
      k.e("add", acc, acc, s)
    k.e("mulconst", x, acc, 512)
    k.fr(s, one, z2, power, acc)
  else:
    raise newException(GenError, "no VM composition for unary op " & kind)

proc binary*(k: KB; kind: string; a, b: int) =
  ## In place on `a`.
  case kind
  of "add", "sub", "mul", "div": k.e(kind, a, a, b)
  of "max":
    let c = k.r()
    k.e("cmplt", c, a, b)
    k.cond(c, proc() = k.cp(a, b))
    k.fr(c)
  of "min":
    let c = k.r()
    k.e("cmplt", c, b, a)
    k.cond(c, proc() = k.cp(a, b))
    k.fr(c)
  else:
    raise newException(GenError, "no VM composition for binary op " & kind)

# ---- session op generators ------------------------------------------------
# Thread counts are returned through `Gen.threads`.

type
  Gen* = object
    body*: string
    threads*: int      ## VM thread count (one lane per output element or row)
    tg*: int           ## threadgroup size the kernel needs (0: any)
    tgBytes*: int      ## threadgroup memory the kernel needs
    ins*, outs*: int   ## port counts

proc done(k: KB; threads, ins, outs: int): Gen =
  Gen(body: k.text, threads: threads, ins: ins, outs: outs)

proc genVecOp*(op: string; n: int): Gen =
  ## sessionVecOp(op, a, b, c, n): c = a (op) b, elementwise. ports in0=a in1=b out0=c.
  let k = newKB()
  let a = k.r()
  let b = k.r()
  k.e("uloadf32", a, 0, "@in0")
  k.e("uloadf32", b, 0, "@in1")
  k.binary(op, a, b)
  k.e("ustoref32", a, 0, "@out0")
  k.done(n, 2, 1)

proc genActivation*(op: string; n: int; p = 0.0): Gen =
  ## sessionActivation(op, x, y, n[, p]): y = f(x). ports in0=x out0=y.
  let k = newKB()
  let x = k.r()
  k.e("uloadf32", x, 0, "@in0")
  if op == "pow":
    if p != floor(p) or abs(p) > 32:
      raise newException(GenError, "pow needs an integer exponent in [-32,32]")
    let n = int(p)
    if n == 0:
      k.e("loadconstf32", x, 1)
    else:
      let base = k.r()
      k.cp(base, x)
      for _ in 1 ..< abs(n): k.e("mul", x, x, base)
      if n < 0:
        let one = k.cf(1)
        k.e("div", x, one, x)
        k.fr(one)
      k.fr(base)
  else:
    k.unary(op, x)
  k.e("ustoref32", x, 0, "@out0")
  k.done(n, 1, 1)

proc genScale*(n: int; s: float): Gen =
  ## sessionScale(x, s, n): y = x * s. ports in0=x out0=y.
  let k = newKB()
  let x = k.r()
  k.e("uloadf32", x, 0, "@in0")
  k.e("mulconst", x, x, fnum(s))
  k.e("ustoref32", x, 0, "@out0")
  k.done(n, 1, 1)

proc genSoftmax*(rows, cols: int): Gen =
  ## sessionSoftmax(x, y, rows, cols): row-wise softmax. ports in0=x out0=y; lane = row.
  let k = newKB()
  let base = k.u()
  k.e("umulk", base, 0, cols)
  let m = k.cf(NegLarge)
  let total = k.cf(0)
  let i = k.u()
  let v = k.r()
  let c = k.r()
  k.e("uaddk", i, base, 0)
  k.forN(cols, proc() =
    k.e("uloadf32", v, i, "@in0")
    k.e("cmplt", c, m, v)
    k.cond(c, proc() = k.cp(m, v))
    k.e("uaddk", i, i, 1))
  k.e("uaddk", i, base, 0)
  k.forN(cols, proc() =
    k.e("uloadf32", v, i, "@in0")
    k.e("sub", v, v, m)
    k.e("exp", v, v)
    k.e("ustoref32", v, i, "@out0")
    k.e("add", total, total, v)
    k.e("uaddk", i, i, 1))
  k.e("uaddk", i, base, 0)
  k.forN(cols, proc() =
    k.e("uloadf32", v, i, "@out0")
    k.e("div", v, v, total)
    k.e("ustoref32", v, i, "@out0")
    k.e("uaddk", i, i, 1))
  k.done(rows, 1, 1)

proc genNorm(rows, cols: int; eps: float; layer, hasBeta: bool): Gen =
  ## ports: in0=x in1=gamma [in2=beta] out0=y; lane = row.
  let k = newKB()
  let base = k.u()
  k.e("umulk", base, 0, cols)
  let i = k.u()
  let j = k.u()
  let v = k.r()
  let w = k.r()
  let mean = k.cf(0)
  let sq = k.cf(0)
  k.e("uaddk", i, base, 0)
  k.forN(cols, proc() =
    k.e("uloadf32", v, i, "@in0")
    if layer: k.e("add", mean, mean, v)
    k.e("mul", v, v, v)
    k.e("add", sq, sq, v)
    k.e("uaddk", i, i, 1))
  let inv = k.r()
  if layer:
    k.e("mulconst", mean, mean, fnum(1.0 / float(cols)))
    k.e("mulconst", sq, sq, fnum(1.0 / float(cols)))
    k.e("mul", v, mean, mean)
    k.e("sub", sq, sq, v)         # variance = E[x^2] - mean^2
  else:
    k.e("mulconst", sq, sq, fnum(1.0 / float(cols)))
  k.e("addconst", inv, sq, fnum(eps))
  k.e("rsqrt", inv, inv)
  k.e("uaddk", i, base, 0)
  k.e("uconst", j, 0)
  k.forN(cols, proc() =
    k.e("uloadf32", v, i, "@in0")
    if layer: k.e("sub", v, v, mean)
    k.e("mul", v, v, inv)
    k.e("uloadf32", w, j, "@in1")
    k.e("mul", v, v, w)
    if hasBeta:
      k.e("uloadf32", w, j, "@in2")
      k.e("add", v, v, w)
    k.e("ustoref32", v, i, "@out0")
    k.e("uaddk", i, i, 1)
    k.e("uaddk", j, j, 1))
  k.done(rows, (if hasBeta: 3 else: 2), 1)

proc genRmsnorm*(rows, cols: int; eps: float): Gen =
  ## sessionRmsnormEnc(x, gamma, y, rows, cols, eps).
  genNorm(rows, cols, eps, false, false)

proc genAddRmsnorm*(cols: int; eps: float): Gen =
  ## sessionVecOp add followed by sessionRmsnormEnc (rows = 1) in one launch:
  ## out0 = a + b, out1 = rmsnorm(out0) * gamma. ports in0=a in1=b in2=gamma.
  let k = newKB()
  let i = k.u()
  let v = k.r()
  let w = k.r()
  let sq = k.cf(0)
  k.e("uconst", i, 0)
  k.forN(cols, proc() =
    k.e("uloadf32", v, i, "@in0")
    k.e("uloadf32", w, i, "@in1")
    k.e("add", v, v, w)
    k.e("ustoref32", v, i, "@out0")
    k.e("mul", v, v, v)
    k.e("add", sq, sq, v)
    k.e("uaddk", i, i, 1))
  let inv = k.r()
  k.e("mulconst", sq, sq, fnum(1.0 / float(cols)))
  k.e("addconst", inv, sq, fnum(eps))
  k.e("rsqrt", inv, inv)
  k.e("uconst", i, 0)
  k.forN(cols, proc() =
    k.e("uloadf32", v, i, "@out0")
    k.e("mul", v, v, inv)
    k.e("uloadf32", w, i, "@in2")
    k.e("mul", v, v, w)
    k.e("ustoref32", v, i, "@out1")
    k.e("uaddk", i, i, 1))
  k.done(1, 3, 2)

proc genSiluMul*(n: int): Gen =
  ## sigmoid, mul, mul of the SwiGLU gate in one launch: out0 = x * sigmoid(x) * u.
  ## ports in0=x (gate) in1=u (up).
  let k = newKB()
  let x = k.r()
  let s = k.r()
  let u = k.r()
  k.e("uloadf32", x, 0, "@in0")
  k.e("sigmoid", s, x)
  k.e("mul", x, x, s)
  k.e("uloadf32", u, 0, "@in1")
  k.e("mul", x, x, u)
  k.e("ustoref32", x, 0, "@out0")
  k.done(n, 2, 1)

proc genLayernorm*(rows, cols: int; eps: float; hasBeta = true): Gen =
  ## sessionLayernormEnc(x, gamma, beta, y, rows, cols, eps).
  genNorm(rows, cols, eps, true, hasBeta)

proc genRope*(T, Hn, D: int): Gen =
  ## sessionRope(x, cosT, sinT, T, Hn, D): rotate-half RoPE with [T, D/2] tables.
  ## ports in0=x in1=cos in2=sin out0=y (same shape as x); lane = head.
  let half = D div 2
  let k = newKB()
  let t = k.u()
  let i = k.u()
  let tab = k.u()
  let xb = k.u()
  let c = k.r()
  let s = k.r()
  let x1 = k.r()
  let x2 = k.r()
  let o = k.r()
  k.e("uconst", t, 0)
  k.e("uconst", tab, 0)
  k.e("umulk", xb, 0, D)          # (t*Hn + h)*D with t = 0
  k.forN(T, proc() =
    k.e("uconst", i, 0)
    k.forN(half, proc() =
      let a = k.u()
      k.e("uadd", a, xb, i)
      let b = k.u()
      k.e("uadd", b, tab, i)
      k.e("uloadf32", c, b, "@in1")
      k.e("uloadf32", s, b, "@in2")
      k.e("uloadf32", x1, a, "@in0")
      k.e("uloadf32", x2, a, "@in0+" & $half)
      k.e("mul", o, x1, c)
      k.e("mul", x1, x1, s)
      k.e("mul", x2, x2, s)
      k.e("sub", o, o, x2)        # x1*c - x2*s
      k.e("ustoref32", o, a, "@out0")
      # second half: x2*c + x1*s (x1 already multiplied by s, x2 by s; recompute)
      k.e("uloadf32", o, a, "@in0+" & $half)
      k.e("mul", o, o, c)
      k.e("add", o, o, x1)
      k.e("ustoref32", o, a, "@out0+" & $half)
      k.fu(a, b)
      k.e("uaddk", i, i, 1))
    k.e("uaddk", tab, tab, half)
    k.e("uaddk", xb, xb, Hn * D))
  k.done(Hn, 3, 1)

proc genMatmulNT*(M, K, N: int; bOffset = 0): Gen =
  ## sessionMatmulNT(a, b, c, M, K, N, bOffset): C[m,n] = sum_k A[m,k] * B[n,k].
  ## ports in0=A in1=B out0=C; lane = n.
  let k = newKB()
  let bi = k.u()
  let ai = k.u()
  let ci = k.u()
  let bBase = k.u()
  let acc = k.r()
  let x = k.r()
  let y = k.r()
  k.e("umulk", bBase, 0, K)
  k.e("uconst", ai, 0)
  k.e("uaddk", ci, 0, 0)
  k.forN(M, proc() =
    k.e("uaddk", bi, bBase, 0)
    k.e("loadconstf32", acc, 0)
    k.forN(K, proc() =
      k.e("uloadf32", x, ai, "@in0")
      k.e("uloadf32", y, bi, "@in1" & (if bOffset != 0: "+" & $bOffset else: ""))
      k.e("mul", x, x, y)
      k.e("add", acc, acc, x)
      k.e("uaddk", ai, ai, 1)
      k.e("uaddk", bi, bi, 1))
    k.e("ustoref32", acc, ci, "@out0")
    k.e("uaddk", ci, ci, N))
  k.done(N, 2, 1)

proc genMatmul*(M, K, N: int): Gen =
  ## sessionMatmul(a, b, c, M, K, N): C[m,n] = sum_k A[m,k] * B[k,n]. lane = n.
  let k = newKB()
  let bi = k.u()
  let ai = k.u()
  let ci = k.u()
  let acc = k.r()
  let x = k.r()
  let y = k.r()
  k.e("uconst", ai, 0)
  k.e("uaddk", ci, 0, 0)
  k.forN(M, proc() =
    k.e("uaddk", bi, 0, 0)
    k.e("loadconstf32", acc, 0)
    k.forN(K, proc() =
      k.e("uloadf32", x, ai, "@in0")
      k.e("uloadf32", y, bi, "@in1")
      k.e("mul", x, x, y)
      k.e("add", acc, acc, x)
      k.e("uaddk", ai, ai, 1)
      k.e("uaddk", bi, bi, N))
    k.e("ustoref32", acc, ci, "@out0")
    k.e("uaddk", ci, ci, N))
  k.done(N, 2, 1)

proc genGather*(n: int): Gen =
  ## sessionGatherEnc(src, idx, dst, n): dst[i] = src[idx[i]]. ports in0=src in1=idx out0=dst.
  let k = newKB()
  let ix = k.r()
  let iu = k.u()
  let v = k.r()
  k.e("uloadf32", ix, 0, "@in1")
  k.e("rtou", iu, ix)
  k.e("uloadf32", v, iu, "@in0")
  k.e("ustoref32", v, 0, "@out0")
  k.done(n, 2, 1)

proc genEmbedding*(dim, numIndices: int): Gen =
  ## sessionEmbeddingLookupEnc(table, indices, y, vocab, dim, numIndices):
  ## y[i, :] = table[indices[i], :]. ports in0=table in1=indices out0=y; lane = index.
  let k = newKB()
  let ix = k.r()
  let row = k.u()
  let dst = k.u()
  let v = k.r()
  k.e("uloadf32", ix, 0, "@in1")
  k.e("rtou", row, ix)
  k.e("umulk", row, row, dim)
  k.e("umulk", dst, 0, dim)
  k.forN(dim, proc() =
    k.e("uloadf32", v, row, "@in0")
    k.e("ustoref32", v, dst, "@out0")
    k.e("uaddk", row, row, 1)
    k.e("uaddk", dst, dst, 1))
  k.done(numIndices, 2, 1)

proc genKvScatter*(T, HKV, D, maxSeqLen: int): Gen =
  ## sessionKvScatterEnc(src, dst, T, HKV, D, pos, maxSeqLen):
  ## dst[h, pos + t, :] = src[t, h, :]. ports in0=src in1=pos (scalar) in2=dst (written); lane = h.
  let k = newKB()
  let p = k.r()
  let pu = k.u()
  let s = k.u()
  let d = k.u()
  let v = k.r()
  let zero = k.cu(0)
  k.e("uloadf32", p, zero, "@in1")
  k.e("rtou", pu, p)
  k.e("umulk", pu, pu, D)
  k.e("umulk", d, 0, maxSeqLen * D)
  k.e("uadd", d, d, pu)
  k.e("umulk", s, 0, D)
  k.forN(T, proc() =
    k.forN(D, proc() =
      k.e("uloadf32", v, s, "@in0")
      k.e("ustoref32", v, d, "@in2")
      k.e("uaddk", s, s, 1)
      k.e("uaddk", d, d, 1))
    k.e("uaddk", s, s, (HKV - 1) * D))
  k.done(HKV, 3, 0)

proc genAttentionDecode*(H, HKV, D, maxSeqLen: int; countFromPosition = false): Gen =
  ## sessionAttentionDecodeAllHeadsEnc(q, kAll, vAll, o, H, HKV, D, S, maxSeqLen).
  ## q is already scaled; scores = q.K[s], softmax over s < S, o = sum p*V.
  ## ports in0=q [H*D] in1=K [HKV*maxSeqLen*D] in2=V in3=S (scalar) out0=o [H*D]; lane = kv head,
  ## each lane serves the H/HKV query heads that share it.
  if HKV <= 0 or H mod HKV != 0:
    raise newException(GenError, "attention needs H divisible by HKV")
  let nRep = H div HKV
  let k = newKB()
  let sCnt = k.r()
  let qb = k.u()
  let ob = k.u()
  let kvb = k.u()
  let m = k.r()
  let l = k.r()
  let dot = k.r()
  let qv = k.r()
  let kv = k.r()
  let c = k.r()
  let ks = k.u()
  let qi = k.u()
  let ki = k.u()
  let oi = k.u()
  let vi = k.u()
  let zero = k.cu(0)
  k.e("uloadf32", sCnt, zero, "@in3")
  if countFromPosition:
    # in3 holds the token position; the cache holds position + 1 valid rows.
    k.e("addconst", sCnt, sCnt, 1)
  k.e("umulk", qb, 0, nRep * D)
  k.e("uaddk", ob, qb, 0)
  k.e("umulk", kvb, 0, maxSeqLen * D)

  proc scoreOf() =
    # dot = q . K[s]; qi/ki advance D; caller sets ks (row base).
    k.e("loadconstf32", dot, 0)
    k.e("uaddk", ki, ks, 0)
    k.e("uaddk", qi, qb, 0)
    k.forN(D, proc() =
      k.e("uloadf32", qv, qi, "@in0")
      k.e("uloadf32", kv, ki, "@in1")
      k.e("mul", qv, qv, kv)
      k.e("add", dot, dot, qv)
      k.e("uaddk", qi, qi, 1)
      k.e("uaddk", ki, ki, 1))

  k.forN(nRep, proc() =
    k.e("loadconstf32", m, fnum(NegLarge))
    k.e("loadconstf32", l, 0)
    k.e("uaddk", ks, kvb, 0)
    k.forR(sCnt, proc() =
      scoreOf()
      k.e("cmplt", c, m, dot)
      k.cond(c, proc() = k.cp(m, dot))
      k.e("uaddk", ks, ks, D))
    k.e("uaddk", oi, ob, 0)
    k.forN(D, proc() =
      k.e("loadconstf32", qv, 0)
      k.e("ustoref32", qv, oi, "@out0")
      k.e("uaddk", oi, oi, 1))
    k.e("uaddk", ks, kvb, 0)
    k.forR(sCnt, proc() =
      scoreOf()
      k.e("sub", dot, dot, m)
      k.e("exp", dot, dot)
      k.e("add", l, l, dot)
      k.e("uaddk", vi, ks, 0)
      k.e("uaddk", oi, ob, 0)
      k.forN(D, proc() =
        k.e("uloadf32", kv, vi, "@in2")
        k.e("mul", kv, kv, dot)
        k.e("uloadf32", qv, oi, "@out0")
        k.e("add", qv, qv, kv)
        k.e("ustoref32", qv, oi, "@out0")
        k.e("uaddk", vi, vi, 1)
        k.e("uaddk", oi, oi, 1))
      k.e("uaddk", ks, ks, D))
    k.e("uaddk", oi, ob, 0)
    k.forN(D, proc() =
      k.e("uloadf32", qv, oi, "@out0")
      k.e("div", qv, qv, l)
      k.e("ustoref32", qv, oi, "@out0")
      k.e("uaddk", oi, oi, 1))
    k.e("uaddk", qb, qb, D)
    k.e("uaddk", ob, ob, D))
  k.done(HKV, 4, 1)

proc rebasePorts*(g: Gen; inOffsets, outOffsets: seq[int]): Gen =
  ## Port `@inK` / `@outK` addresses of `g` shifted by a float offset: lets several tensors that
  ## live at different offsets of one arena buffer be addressed through one buffer name
  ## (`@in2+4` becomes `@in2+(4 + inOffsets[2])`).
  result = g
  var outText = ""
  var i = 0
  let t = g.body
  while i < t.len:
    if t[i] == '@' and (t.continuesWith("in", i+1) or t.continuesWith("out", i+1)):
      let isIn = t.continuesWith("in", i+1)
      var j = i + (if isIn: 3 else: 4)
      var digits = ""
      while j < t.len and t[j] in {'0'..'9'}:
        digits.add t[j]
        inc j
      if digits.len == 0:
        outText.add t[i]
        inc i
        continue
      let idx = parseInt(digits)
      let offs = if isIn: inOffsets else: outOffsets
      var extra = if idx < offs.len: offs[idx] else: 0
      if j < t.len and t[j] == '+':
        var k = j + 1
        var num = ""
        while k < t.len and t[k] in {'0'..'9'}:
          num.add t[k]
          inc k
        if num.len > 0:
          extra += parseInt(num)
          j = k
      outText.add (if isIn: "@in" else: "@out") & digits
      if extra != 0: outText.add "+" & $extra
      i = j
    else:
      outText.add t[i]
      inc i
  result.body = outText

# ---- more session ops -----------------------------------------------------

proc ad(port: string; off = 0): string =
  "@" & port & (if off != 0: "+" & $off else: "")

proc genCopy*(n: int): Gen =
  ## sessionCopyInto(src, dst, srcOff, dstOff, n): out0 = in0[0 ..< n]. ports in0 -> out0.
  let k = newKB()
  let v = k.r()
  k.e("uloadf32", v, 0, "@in0")
  k.e("ustoref32", v, 0, "@out0")
  k.done(n, 1, 1)

proc genCopyOffset*(n, srcOffset, dstOffset: int): Gen =
  ## sessionCopyInto(src, dst, srcOffsetElems, dstOffsetElems, n): out0[dstOff + i] = in0[srcOff + i].
  ## Only the copied range of out0 is written; the rest keeps its contents.
  let k = newKB()
  let v = k.r()
  k.e("uloadf32", v, 0, "@in0" & (if srcOffset != 0: "+" & $srcOffset else: ""))
  k.e("ustoref32", v, 0, "@out0" & (if dstOffset != 0: "+" & $dstOffset else: ""))
  k.done(n, 1, 1)

proc sincosOf*(k: KB; angle, outS, outC: int) =
  ## sin and cos of a non-negative angle held in R[angle] (preserved). Cody-Waite reduction to
  ## [-pi, pi] (the high part has 8 mantissa bits, so z * hi is exact for z < 2^15), then
  ## Taylor series.
  let z = k.r()
  let x = k.r()
  let t = k.r()
  let z2 = k.r()
  let ts = k.r()
  let tc = k.r()
  let u = k.u()
  k.e("mulconst", z, angle, fnum(1.0 / (2.0 * PI)))
  k.e("addconst", z, z, "0.5")
  k.e("rtou", u, z)
  k.e("utor", z, u)
  k.e("mulconst", t, z, "6.28125")
  k.e("sub", x, angle, t)
  k.e("mulconst", t, z, "0.0019353071795864769")
  k.e("sub", x, x, t)
  k.e("mul", z2, x, x)
  k.cp(ts, x)
  k.cp(outS, x)
  k.e("loadconstf32", tc, 1)
  k.e("loadconstf32", outC, 1)
  for j in 1 .. 11:
    k.e("mul", ts, ts, z2)
    k.e("mulconst", ts, ts, fnum(-1.0 / float((2 * j) * (2 * j + 1))))
    k.e("add", outS, outS, ts)
    k.e("mul", tc, tc, z2)
    k.e("mulconst", tc, tc, fnum(-1.0 / float((2 * j - 1) * (2 * j))))
    k.e("add", outC, outC, tc)
  k.fr(z, x, t, z2, ts, tc)
  k.fu(u)

proc genRopePos*(Hn, D: int; ratio: float; posScale = 1.0; outScale = 1.0): Gen =
  ## Rotate-half RoPE for one token whose position is a scalar port:
  ## angle_i = position * posScale * ratio^i with ratio = theta^(-1/(D/2)).
  ## ports in0=x [Hn*D] in1=position (scalar) out0=y (may alias x); lane = head.
  let half = D div 2
  let k = newKB()
  let zero = k.cu(0)
  let pos = k.r()
  let freq = k.r()
  let ang = k.r()
  let sn = k.r()
  let cs = k.r()
  let x1 = k.r()
  let x2 = k.r()
  let o = k.r()
  let xb = k.u()
  k.e("uloadf32", pos, zero, "@in1")
  if posScale != 1.0: k.e("mulconst", pos, pos, fnum(posScale))
  k.e("loadconstf32", freq, 1)
  k.e("umulk", xb, 0, D)
  k.forN(half, proc() =
    k.e("mul", ang, pos, freq)
    k.sincosOf(ang, sn, cs)
    k.e("uloadf32", x1, xb, "@in0")
    k.e("uloadf32", x2, xb, "@in0+" & $half)
    k.e("mul", o, x1, cs)
    k.e("mul", ang, x2, sn)
    k.e("sub", o, o, ang)
    if outScale != 1.0: k.e("mulconst", o, o, fnum(outScale))
    k.e("ustoref32", o, xb, "@out0")
    k.e("mul", o, x2, cs)
    k.e("mul", ang, x1, sn)
    k.e("add", o, o, ang)
    if outScale != 1.0: k.e("mulconst", o, o, fnum(outScale))
    k.e("ustoref32", o, xb, "@out0+" & $half)
    k.e("mulconst", freq, freq, fnum(ratio))
    k.e("uaddk", xb, xb, 1))
  k.done(Hn, 2, 1)

proc genRopePosRows*(T, Hn, D: int; ratio: float; posScale = 1.0; outScale = 1.0): Gen =
  ## Rotate-half RoPE over T consecutive tokens, x laid out [T, Hn, D]; row t sits at position
  ## `position + t` (scalar port in1 holds the first position). ports in0=x in1=position out0=y
  ## (may alias x); lane = head.
  let half = D div 2
  let k = newKB()
  let zero = k.cu(0)
  let pos0 = k.r()
  let pos = k.r()
  let freq = k.r()
  let ang = k.r()
  let sn = k.r()
  let cs = k.r()
  let x1 = k.r()
  let x2 = k.r()
  let o = k.r()
  let xb = k.u()
  k.e("uloadf32", pos0, zero, "@in1")
  if posScale != 1.0: k.e("mulconst", pos0, pos0, fnum(posScale))
  k.e("umulk", xb, 0, D)
  k.forN(T, proc() =
    k.e("loadconstf32", freq, 1)
    k.e("mulconst", pos, pos0, 1)
    k.forN(half, proc() =
      k.e("mul", ang, pos, freq)
      k.sincosOf(ang, sn, cs)
      k.e("uloadf32", x1, xb, "@in0")
      k.e("uloadf32", x2, xb, "@in0+" & $half)
      k.e("mul", o, x1, cs)
      k.e("mul", ang, x2, sn)
      k.e("sub", o, o, ang)
      if outScale != 1.0: k.e("mulconst", o, o, fnum(outScale))
      k.e("ustoref32", o, xb, "@out0")
      k.e("mul", o, x2, cs)
      k.e("mul", ang, x1, sn)
      k.e("add", o, o, ang)
      if outScale != 1.0: k.e("mulconst", o, o, fnum(outScale))
      k.e("ustoref32", o, xb, "@out0+" & $half)
      k.e("mulconst", freq, freq, fnum(ratio))
      k.e("uaddk", xb, xb, 1))
    k.e("addconst", pos0, pos0, fnum(posScale))
    k.e("uaddk", xb, xb, Hn*D - half))
  k.done(Hn, 2, 1)

proc ilog2(n: int): int =
  while (1 shl (result + 1)) <= n: inc result

proc divFloorU(k: KB; dst, src: int; n: int) =
  ## U[dst] = U[src] / n for small values, through float (exact while src < 2^20).
  let r = k.r()
  k.e("utor", r, src)
  k.e("addconst", r, r, "0.5")
  k.e("mulconst", r, r, fnum(1.0 / float(n)))
  k.e("rtou", dst, r)

proc modU(k: KB; dst, src, q: int; n: int) =
  ## U[dst] = U[src] - U[q] * n, through float (small values).
  let a = k.r()
  let b = k.r()
  k.e("utor", a, src)
  k.e("utor", b, q)
  k.e("mulconst", b, b, fnum(float(n)))
  k.e("sub", a, a, b)
  k.e("rtou", dst, a)

proc genRopePosPar*(T, Hn, D: int; ratio: float; posScale = 1.0; outScale = 1.0): Gen =
  ## Rotate-half RoPE with one thread per (row, head, frequency) pair, the shape of the Metal
  ## rope kernels. Row t sits at position + t; x is [T, Hn, D]. Needs D/2 to be a power of two.
  ## ports in0=x in1=position (scalar) out0=y (may alias x).
  let half = D div 2
  let k = newKB()
  let zero = k.cu(0)
  let pos = k.r()
  let freq = k.r()
  let ang = k.r()
  let sn = k.r()
  let cs = k.r()
  let x1 = k.r()
  let x2 = k.r()
  let o = k.r()
  let fi = k.r()
  let i = k.u()
  let q = k.u()
  let t = k.u()
  let xb = k.u()
  k.e("uandk", i, 0, half - 1)
  k.e("ushrk", q, 0, ilog2(half))
  k.e("umulk", xb, q, D)
  k.e("uadd", xb, xb, i)
  k.e("uloadf32", pos, zero, "@in1")
  if T > 1:
    k.divFloorU(t, q, Hn)
    k.e("utor", fi, t)
    k.e("add", pos, pos, fi)
  if posScale != 1.0: k.e("mulconst", pos, pos, fnum(posScale))
  k.e("utor", fi, i)
  k.e("mulconst", fi, fi, fnum(ln(ratio)))
  k.e("exp", freq, fi)
  k.e("mul", ang, pos, freq)
  k.sincosOf(ang, sn, cs)
  k.e("uloadf32", x1, xb, "@in0")
  k.e("uloadf32", x2, xb, "@in0+" & $half)
  k.e("mul", o, x1, cs)
  k.e("mul", ang, x2, sn)
  k.e("sub", o, o, ang)
  if outScale != 1.0: k.e("mulconst", o, o, fnum(outScale))
  k.e("ustoref32", o, xb, "@out0")
  k.e("mul", o, x2, cs)
  k.e("mul", ang, x1, sn)
  k.e("add", o, o, ang)
  if outScale != 1.0: k.e("mulconst", o, o, fnum(outScale))
  k.e("ustoref32", o, xb, "@out0+" & $half)
  k.done(T * Hn * half, 2, 1)

proc genKvScatterPar*(T, HKV, D, maxSeqLen: int): Gen =
  ## KV scatter with one thread per element: dst[h, pos + t, d] = src[t, h, d].
  ## ports in0=src in1=pos (scalar) in2=dst (written).
  let k = newKB()
  let zero = k.cu(0)
  let p = k.r()
  let v = k.r()
  let pu = k.u()
  let q = k.u()
  let t = k.u()
  let h = k.u()
  let d = k.u()
  let a = k.u()
  k.e("uloadf32", p, zero, "@in1")
  k.e("rtou", pu, p)
  k.divFloorU(q, 0, D)
  k.modU(d, 0, q, D)
  k.divFloorU(t, q, HKV)
  k.modU(h, q, t, HKV)
  k.e("uadd", a, pu, t)
  k.e("umulk", h, h, maxSeqLen)
  k.e("uadd", a, a, h)
  k.e("umulk", a, a, D)
  k.e("uadd", a, a, d)
  k.e("uloadf32", v, 0, "@in0")
  k.e("ustoref32", v, a, "@in2")
  k.done(T * HKV * D, 3, 0)

proc genMatmulQ4*(M, K, N, groupSize, nGroups: int; wqByteOffset = 0; packed = false): Gen =
  ## sessionMatmulQ4(a, w, c, M): C[m,n] = sum_k A[m,k] * (nib(n,k) - zero[n,g]) * scale[n,g],
  ## nibble low = even k, high = odd k, g = k / groupSize.
  ## ports in0=A [M,K] in1=packed weights (bytes, [N, ceil(K/2)]) in2=scales [N*nGroups]
  ## in3=zeros [N*nGroups] out0=C [M,N]; lane = n. The byte region of in1 is read through
  ## `uloadu8` (needs the script layout).
  ## packed = true: one weight port in1 holds [nibble bytes padded to words][scales][zeros] (the q4b
  ## weight dtype), so the node has two inputs (A, W).
  let gs = if groupSize > 0: groupSize else: K
  if gs mod 2 != 0:
    raise newException(GenError, "q4 group size must be even")
  let bytesPerRow = (K + 1) div 2
  let fullGroups = K div gs
  let rem = K - fullGroups * gs
  let wqF = (N * bytesPerRow + 3) div 4
  let scalesAddr = if packed: "@in1+" & $wqF else: "@in2"
  let zerosAddr = if packed: "@in1+" & $(wqF + N * nGroups) else: "@in3"
  let k = newKB()
  let wqb = k.u()
  let rowb = k.u()
  let soff = k.u()
  let ai = k.u()
  let ci = k.u()
  let bi = k.u()
  let si = k.u()
  let bu = k.u()
  let nib = k.u()
  let acc = k.r()
  let sc = k.r()
  let zp = k.r()
  let a0 = k.r()
  let w = k.r()
  k.e("uconst", wqb, "@in1")
  k.e("umulk", wqb, wqb, 4)
  if wqByteOffset != 0: k.e("uaddk", wqb, wqb, wqByteOffset)
  k.e("umulk", rowb, 0, bytesPerRow)
  k.e("uadd", rowb, rowb, wqb)
  k.e("umulk", soff, 0, nGroups)
  k.e("uconst", ai, 0)
  k.e("uaddk", ci, 0, 0)

  proc pair(withHigh: bool) =
    k.e("uloadu8", bu, bi, 0)
    k.e("uandk", nib, bu, 15)
    k.e("utor", w, nib)
    k.e("sub", w, w, zp)
    k.e("mul", w, w, sc)
    k.e("uloadf32", a0, ai, "@in0")
    k.e("mul", a0, a0, w)
    k.e("add", acc, acc, a0)
    if withHigh:
      k.e("ushrk", nib, bu, 4)
      k.e("utor", w, nib)
      k.e("sub", w, w, zp)
      k.e("mul", w, w, sc)
      k.e("uloadf32", a0, ai, "@in0+1")
      k.e("mul", a0, a0, w)
      k.e("add", acc, acc, a0)
      k.e("uaddk", ai, ai, 2)
    else:
      k.e("uaddk", ai, ai, 1)
    k.e("uaddk", bi, bi, 1)

  proc loadGroup() =
    k.e("uloadf32", sc, si, scalesAddr)
    k.e("uloadf32", zp, si, zerosAddr)

  k.forN(M, proc() =
    k.e("uaddk", bi, rowb, 0)
    k.e("uaddk", si, soff, 0)
    k.e("loadconstf32", acc, 0)
    k.forN(fullGroups, proc() =
      loadGroup()
      k.forN(gs div 2, proc() = pair(true))
      k.e("uaddk", si, si, 1))
    if rem > 0:
      loadGroup()
      k.forN(rem div 2, proc() = pair(true))
      if rem mod 2 == 1: pair(false)
    k.e("ustoref32", acc, ci, "@out0")
    k.e("uaddk", ci, ci, N))
  k.done(N, (if packed: 2 else: 4), 1)

proc fp4kmLayout*(N, K: int): tuple[blobWords, lutWord, totalWords: int] =
  ## Words of the `fp4km` blob: [nibble bytes N*K/2][scale bytes N*K/32] padded to words, then the
  ## 272-float decode table (16 FP4 values, then 256 E4M3 scale values times the tensor scale).
  let bytes = N * (K div 2) + N * (K div 32)
  let w = (bytes + 3) div 4
  (w, w, w + 272)

proc genFp4Gemm*(M, K, N: int): Gen =
  ## sessionFp4GemmTiledEnc / sessionFp4GemmTensorCore(a, w, lut, c, M, K, N): FP4-KM weights
  ## W[N,K], group 32; byte j of a group holds element j (low nibble) and j+16 (high nibble);
  ## value = lut[nibble] * lut[16 + scaleByte].
  ## ports in0=A [M,K] in1=blob (see fp4kmLayout; the lut is its tail) out0=C [M,N]; lane = n.
  if K mod 32 != 0: raise newException(GenError, "FP4-KM needs K divisible by 32")
  let nG = K div 32
  let lay = fp4kmLayout(N, K)
  let lutAddr = "@in1+" & $lay.lutWord
  let scaleLutAddr = "@in1+" & $(lay.lutWord + 16)
  let k = newKB()
  let wqb = k.u()
  let rowb = k.u()
  let sb = k.u()
  let bi = k.u()
  let ai = k.u()
  let ci = k.u()
  let bu = k.u()
  let nib = k.u()
  let sc = k.r()
  let acc = k.r()
  let a0 = k.r()
  let w = k.r()
  k.e("uconst", wqb, "@in1")
  k.e("umulk", wqb, wqb, 4)
  k.e("umulk", rowb, 0, K div 2)
  k.e("uadd", rowb, rowb, wqb)
  k.e("umulk", sb, 0, nG)
  k.e("uaddk", sb, sb, N * (K div 2))
  k.e("uadd", sb, sb, wqb)
  k.e("uconst", ai, 0)
  k.e("uaddk", ci, 0, 0)
  k.forN(M, proc() =
    k.e("uaddk", bi, rowb, 0)
    k.e("loadconstf32", acc, 0)
    # scale byte pointer: sb advances per group, restored per row below
    let sp = k.u()
    k.e("uaddk", sp, sb, 0)
    k.forN(nG, proc() =
      k.e("uloadu8", bu, sp, 0)
      k.e("uloadf32", sc, bu, scaleLutAddr)
      k.forN(16, proc() =
        k.e("uloadu8", bu, bi, 0)
        k.e("uandk", nib, bu, 15)
        k.e("uloadf32", w, nib, lutAddr)
        k.e("mul", w, w, sc)
        k.e("uloadf32", a0, ai, "@in0")
        k.e("mul", a0, a0, w)
        k.e("add", acc, acc, a0)
        k.e("ushrk", nib, bu, 4)
        k.e("uloadf32", w, nib, lutAddr)
        k.e("mul", w, w, sc)
        k.e("uloadf32", a0, ai, "@in0+16")
        k.e("mul", a0, a0, w)
        k.e("add", acc, acc, a0)
        k.e("uaddk", ai, ai, 1)
        k.e("uaddk", bi, bi, 1))
      k.e("uaddk", ai, ai, 16)
      k.e("uaddk", sp, sp, 1))
    k.fu(sp)
    k.e("ustoref32", acc, ci, "@out0")
    k.e("uaddk", ci, ci, N))
  k.done(N, 2, 1)

# ---- threadgroup-parallel matmuls -----------------------------------------
# One threadgroup of T lanes per output column: the lanes split the K reduction (coalesced reads
# along K) and a tree reduction in threadgroup memory adds the partial sums.

proc tgSum(k: KB; lf, acc, T: int) =
  ## Sum `acc` over the T lanes of the group; the total ends in threadgroup slot 0 (every lane
  ## must call this). `lf` holds the float lane id.
  k.e("tgstoref32", acc, lf, 0)
  k.e("barriertg")
  let lim = k.r()
  let idx = k.r()
  let c = k.r()
  let x = k.r()
  let y = k.r()
  var step = T div 2
  while step >= 1:
    let s = step
    k.e("loadconstf32", lim, s)
    k.e("cmplt", c, lf, lim)
    k.cond(c, proc() =
      k.e("addconst", idx, lf, s)
      k.e("tgloadf32", x, idx, 0)
      k.e("tgloadf32", y, lf, 0)
      k.e("add", y, y, x)
      k.e("tgstoref32", y, lf, 0))
    k.e("barriertg")
    step = step div 2
  k.fr(lim, idx, c, x, y)

proc storeLaneZero(k: KB; lf, zeroF, T, ci: int; outPort: string) =
  ## Lane 0 writes tg slot 0 to out[ci]; all lanes then pass a barrier so the slot can be reused.
  let t = k.r()
  let one = k.cf(1)
  let c = k.r()
  k.e("tgloadf32", t, zeroF, 0)
  k.e("cmplt", c, lf, one)
  k.cond(c, proc() = k.e("ustoref32", t, ci, outPort))
  k.e("barriertg")
  k.fr(t, one, c)

proc parWidth(n: int; limit = 32): int =
  ## Largest power of two <= limit that divides n (1 when n is odd).
  var t = limit
  while t > 1 and n mod t != 0: t = t div 2
  t

proc genMatmulNTPar*(M, K, N: int; T = 32): Gen =
  ## genMatmulNT with a threadgroup of `t` lanes per output column (t divides K). ports as genMatmulNT.
  let t = parWidth(K, T)
  if t < 2: return genMatmulNT(M, K, N)
  let k = newKB()
  let ng = k.u()
  let ul = k.u()
  let ai = k.u()
  let bi = k.u()
  let bBase = k.u()
  let ci = k.u()
  let lf = k.r()
  let zeroF = k.cf(0)
  let acc = k.r()
  let x = k.r()
  let y = k.r()
  k.e("utgid", ng)
  k.e("ulid", ul)
  k.e("loadlid", lf)
  k.e("uaddk", ai, ul, 0)
  k.e("umulk", bBase, ng, K)
  k.e("uadd", bBase, bBase, ul)
  k.e("uaddk", ci, ng, 0)
  k.forN(M, proc() =
    k.e("uaddk", bi, bBase, 0)
    k.e("loadconstf32", acc, 0)
    k.forN(K div t, proc() =
      k.e("uloadf32", x, ai, "@in0")
      k.e("uloadf32", y, bi, "@in1")
      k.e("mul", x, x, y)
      k.e("add", acc, acc, x)
      k.e("uaddk", ai, ai, t)
      k.e("uaddk", bi, bi, t))
    k.tgSum(lf, acc, t)
    k.storeLaneZero(lf, zeroF, t, ci, "@out0")
    k.e("uaddk", ci, ci, N))
  result = Gen(body: k.text, threads: N * t, ins: 2, outs: 1, tg: t, tgBytes: max(16, t * 4))

proc genMatmulQ4Par*(M, K, N, groupSize, nGroups: int; T = 32): Gen =
  ## genMatmulQ4(packed = true) with a threadgroup of `t` lanes per output column. Needs
  ## K divisible by the group size and an even group; ports in0=A in1=packed weights out0=C.
  let gs = if groupSize > 0: groupSize else: K
  let pairs = gs div 2
  let t = if gs mod 2 == 0 and K mod gs == 0: parWidth(pairs, T) else: 1
  if t < 2: return genMatmulQ4(M, K, N, groupSize, nGroups, 0, true)
  let bytesPerRow = (K + 1) div 2
  let fullGroups = K div gs
  let wqF = (N * bytesPerRow + 3) div 4
  let scalesAddr = "@in1+" & $wqF
  let zerosAddr = "@in1+" & $(wqF + N * nGroups)
  let k = newKB()
  let ng = k.u()
  let ul = k.u()
  let wqb = k.u()
  let rowb = k.u()
  let si = k.u()
  let bi = k.u()
  let ai = k.u()
  let ci = k.u()
  let bu = k.u()
  let nib = k.u()
  let lf = k.r()
  let zeroF = k.cf(0)
  let acc = k.r()
  let sc = k.r()
  let zp = k.r()
  let a0 = k.r()
  let w = k.r()
  k.e("utgid", ng)
  k.e("ulid", ul)
  k.e("loadlid", lf)
  k.e("uconst", wqb, "@in1")
  k.e("umulk", wqb, wqb, 4)
  k.e("umulk", rowb, ng, bytesPerRow)
  k.e("uadd", rowb, rowb, wqb)
  k.e("uaddk", ci, ng, 0)
  k.e("uconst", ai, 0)
  k.e("uaddk", ai, ul, 0)
  k.e("uadd", ai, ai, ul)            # 2 * lane: the first element of this lane's pair
  k.forN(M, proc() =
    k.e("uadd", bi, rowb, ul)
    k.e("umulk", si, ng, nGroups)
    k.e("loadconstf32", acc, 0)
    k.forN(fullGroups, proc() =
      k.e("uloadf32", sc, si, scalesAddr)
      k.e("uloadf32", zp, si, zerosAddr)
      k.forN(pairs div t, proc() =
        k.e("uloadu8", bu, bi, 0)
        k.e("uandk", nib, bu, 15)
        k.e("utor", w, nib)
        k.e("sub", w, w, zp)
        k.e("mul", w, w, sc)
        k.e("uloadf32", a0, ai, "@in0")
        k.e("mul", a0, a0, w)
        k.e("add", acc, acc, a0)
        k.e("ushrk", nib, bu, 4)
        k.e("utor", w, nib)
        k.e("sub", w, w, zp)
        k.e("mul", w, w, sc)
        k.e("uloadf32", a0, ai, "@in0+1")
        k.e("mul", a0, a0, w)
        k.e("add", acc, acc, a0)
        k.e("uaddk", ai, ai, 2 * t)
        k.e("uaddk", bi, bi, t))
      # the lane pointers advanced pairs/t * t = pairs bytes (2*pairs = gs elements): next group
      k.e("uaddk", si, si, 1))
    k.tgSum(lf, acc, t)
    k.storeLaneZero(lf, zeroF, t, ci, "@out0")
    k.e("uaddk", ci, ci, N)
    # ai/bi continue into the next row of A; bi restarts at the row base each m
    )
  result = Gen(body: k.text, threads: N * t, ins: 2, outs: 1, tg: t, tgBytes: max(16, t * 4))

proc genMatmulQ4Simd*(K, N, groupSize, nGroups: int; rows = 4; sg = 2; packs = 1): Gen =
  ## Decode (M=1) Q4 GEMV built the way the Metal kernel matmul_q4_simd is: one simdgroup per
  ## `rows` output rows, a lane takes 8 weights (one aligned 4-byte pack) per step and the lanes
  ## stride over K, the dot is the same fma chain, the group is corrected once per step with
  ##   sum = fma(scale, dot - zero * sum(a), sum)
  ## and the lanes are combined with simd_sum. `packs` is how many such 8-weight packs a lane takes
  ## per step (the tuner knob next to rows and sg). Same ports as genMatmulQ4Par (packed q4b layout).
  ## Needs K % 8 == 0, a power-of-two group that divides K, and N % rows == 0.
  let gs = if groupSize > 0: groupSize else: K
  if K mod (8 * packs) != 0 or K mod gs != 0 or (gs and (gs - 1)) != 0 or N mod rows != 0 or gs < 8 * packs:
    return genMatmulQ4Par(1, K, N, groupSize, nGroups)
  var gShift = 0
  while (1 shl gShift) < gs: inc gShift
  let rowBytes = K div 2
  let wqF = (N * rowBytes + 3) div 4
  let scalesAddr = "@in1+" & $wqF
  let zerosAddr = "@in1+" & $(wqF + N * nGroups)
  let stepsK = (K + 128 * packs - 1) div (128 * packs)            # at least 16 lanes x 8 weights per step (width is read at run time)
  # The step count is rounded up to a power of two: the surplus steps are skipped by the k < K guard.
  # On Intel Iris (Metal) a non-power-of-two trip count made this loop return wrong sums.
  var pow2Steps = 1
  while pow2Steps < stepsK: pow2Steps *= 2
  let k = newKB()
  let tg = k.u()
  let lane = k.u()
  let sid = k.u()
  let sw = k.u()
  let stepU = k.u()
  let row0 = k.u()
  let kk = k.u()
  let wqb = k.u()
  let ad = k.u()
  let pk = k.u()
  let tmp = k.u()
  let gi = k.u()
  let nr = k.u()
  let kh = k.u()
  let gb = k.u()
  let kt = k.u()
  k.e("utgid", tg)
  k.e("simdlane", lane)
  k.e("simdw", sw)
  k.e("simdgid", sid)                 # grid-wide simdgroup index, whatever the simdgroup width is
  k.e("umulk", row0, sid, rows)
  k.e("umulk", stepU, sw, 8 * packs)
  k.e("uconst", wqb, "@in1")
  k.e("umulk", wqb, wqb, 4)
  k.e("umulk", kk, lane, 8 * packs)
  let sums = newSeq[int](rows)
  var sumR = sums
  for r in 0 ..< rows: sumR[r] = k.cf(0)
  let acts = newSeq[int](8)
  var a = acts
  for j in 0 ..< 8: a[j] = k.r()
  let asum = k.r()
  let t0 = k.r()
  let t1 = k.r()
  let d = k.r()
  let q = k.r()
  let sc = k.r()
  let zp = k.r()
  let c = k.r()
  let kf = k.r()
  let kLimit = k.cf(float(K))
  let rowf = k.r()
  let nLimit = k.cf(float(N))
  let one = k.cf(1)
  k.e("utor", rowf, row0)
  k.e("cmplt", c, rowf, nLimit)
  k.cond(c, proc() =
    k.forN(pow2Steps, proc() =
      k.e("utor", kf, kk)
      k.e("cmplt", c, kf, kLimit)
      k.cond(c, proc() =
        for pp in 0 ..< packs:
          for j in 0 ..< 8:
            k.e("uloadf32", a[j], kk, "@in0+" & $(j + 8 * pp))
          # act.sum = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7))
          k.e("add", t0, a[0], a[1])
          k.e("add", t1, a[2], a[3])
          k.e("add", t0, t0, t1)
          k.e("add", t1, a[4], a[5])
          k.e("add", d, a[6], a[7])
          k.e("add", t1, t1, d)
          k.e("add", asum, t0, t1)
          k.e("uaddk", kt, kk, 8 * pp)
          k.e("ushrk", gb, kt, gShift)
          k.e("ushrk", kh, kt, 1)
          for r in 0 ..< rows:
            k.e("uaddk", nr, row0, r)
            k.e("umulk", ad, nr, rowBytes)
            k.e("uadd", ad, ad, wqb)
            k.e("uadd", ad, ad, kh)
            k.e("uloadu32", pk, ad, 0)
            # d = fma(a0,q0, fma(a1,q1, ... fma(a6,q6, a7*q7)))
            for j in countdown(7, 0):
              k.e("ushrk", tmp, pk, 4 * j)
              if j < 7: k.e("uandk", tmp, tmp, 15)
              k.e("utor", q, tmp)
              if j == 7: k.e("mul", d, a[7], q)
              else: k.e("fma", d, a[j], q, d)
            k.e("umulk", gi, nr, nGroups)
            k.e("uadd", gi, gi, gb)
            k.e("uloadf32", sc, gi, scalesAddr)
            k.e("uloadf32", zp, gi, zerosAddr)
            k.e("mul", zp, zp, asum)
            k.e("sub", d, d, zp)
            k.e("fma", sumR[r], sc, d, sumR[r]))
      k.e("uadd", kk, kk, stepU))
    k.e("utor", kf, lane)
    k.e("cmplt", c, kf, one)
    for r in 0 ..< rows:
      k.e("simdsum", d, sumR[r])
      k.cond(c, proc() =
        k.e("uaddk", nr, row0, r)
        k.e("ustoref32", d, nr, "@out0")))
  result = Gen(body: k.text, threads: ((N div rows + sg - 1) div sg) * sg * 32, ins: 2, outs: 1,
    tg: sg * 32, tgBytes: 16)

proc tgReduceAt(k: KB; lf, acc, T, off: int; isMax: bool) =
  ## Like tgSum, but over threadgroup slots [off, off+T) and optionally a maximum. The result is
  ## left in slot `off`; every lane must call it.
  k.e("tgstoref32", acc, lf, off)
  k.e("barriertg")
  let lim = k.r()
  let idx = k.r()
  let c = k.r()
  let x = k.r()
  let y = k.r()
  var step = T div 2
  while step >= 1:
    let st = step
    k.e("loadconstf32", lim, st)
    k.e("cmplt", c, lf, lim)
    k.cond(c, proc() =
      k.e("addconst", idx, lf, st)
      k.e("tgloadf32", x, idx, off)
      k.e("tgloadf32", y, lf, off)
      if isMax:
        k.e("cmplt", c, y, x)
        k.cond(c, proc() = k.cp(y, x))
      else:
        k.e("add", y, y, x)
      k.e("tgstoref32", y, lf, off))
    k.e("barriertg")
    step = step div 2
  k.fr(lim, idx, c, x, y)

proc pow2Ceil(n: int): int =
  result = 1
  while result < n: result *= 2

proc genNormPar(rows, cols: int; eps: float; addInput: bool; T = 256): Gen =
  ## RMSNorm with one threadgroup of T lanes per row (the structure of the Metal rmsnorm kernels):
  ## lanes stride over the row, the sum of squares is reduced through threadgroup memory.
  ## addInput: ports in0=a in1=b in2=gamma, out0=a+b, out1=norm(a+b)*gamma (rows must be 1).
  ## otherwise: in0=x in1=gamma out0=y.
  let iters = pow2Ceil((cols + T - 1) div T)       # surplus steps are skipped by the i < cols guard
  let k = newKB()
  let row = k.u()
  let lane = k.u()
  let base = k.u()
  let i = k.u()
  let a = k.u()
  let lf = k.r()
  let v = k.r()
  let w = k.r()
  let sq = k.cf(0)
  let c = k.r()
  let jf = k.r()
  let lim = k.cf(float(cols))
  k.e("utgid", row)
  k.e("ulid", lane)
  k.e("loadlid", lf)
  k.e("umulk", base, row, cols)
  proc pass(body: proc()) =
    k.e("uaddk", i, lane, 0)
    k.forN(iters, proc() =
      k.e("utor", jf, i)
      k.e("cmplt", c, jf, lim)
      k.cond(c, proc() =
        k.e("uadd", a, base, i)
        body())
      k.e("uaddk", i, i, T))
  pass(proc() =
    if addInput:
      k.e("uloadf32", v, a, "@in0")
      k.e("uloadf32", w, a, "@in1")
      k.e("add", v, v, w)
      k.e("ustoref32", v, a, "@out0")
    else:
      k.e("uloadf32", v, a, "@in0")
    k.e("mul", v, v, v)
    k.e("add", sq, sq, v))
  k.tgSum(lf, sq, T)
  let inv = k.r()
  let z = k.cf(0)
  k.e("tgloadf32", inv, z, 0)
  k.e("mulconst", inv, inv, fnum(1.0 / float(cols)))
  k.e("addconst", inv, inv, fnum(eps))
  k.e("rsqrt", inv, inv)
  pass(proc() =
    k.e("uloadf32", v, a, (if addInput: "@out0" else: "@in0"))
    k.e("mul", v, v, inv)
    k.e("uloadf32", w, i, (if addInput: "@in2" else: "@in1"))
    k.e("mul", v, v, w)
    k.e("ustoref32", v, a, (if addInput: "@out1" else: "@out0")))
  result = Gen(body: k.text, threads: rows * T, ins: (if addInput: 3 else: 2), outs: (if addInput: 2 else: 1),
    tg: T, tgBytes: max(16, T * 4))

proc genRmsnormPar*(rows, cols: int; eps: float): Gen = genNormPar(rows, cols, eps, false)
proc genAddRmsnormPar*(cols: int; eps: float): Gen = genNormPar(1, cols, eps, true)

proc genFlashPrefill*(T, H, HKV, D, maxSeqLen: int; scale: float): Gen

proc genAttentionDecodePar*(H, HKV, D, maxSeqLen: int; T = 128; flashT = 0; scale = 1.0): Gen =
  ## Decode attention with one threadgroup of T lanes per query head, as the Metal decode kernels
  ## do: lanes take the keys s = lane, lane+T, ... (score = q . K[s], kept in threadgroup memory),
  ## the max and the sum of exp are reduced across the group, then lane d < D accumulates
  ## sum_s p[s] * V[s][d]. Ports as genAttentionDecode with countFromPosition (in3 = position).
  ## flashT > 0: the prefill form, one threadgroup per (row t, head h) over `flashT` rows, row t sees
  ## position + t + 1 keys and the scores are multiplied by `scale` (q and o are [flashT, H, D]).
  let nRep = if HKV > 0: H div HKV else: 0
  if HKV <= 0 or H mod HKV != 0 or (nRep and (nRep - 1)) != 0 or D > T or
     (maxSeqLen + T) * 4 > 24000 or (flashT > 0 and (H and (H - 1)) != 0):
    if flashT > 0: return genFlashPrefill(flashT, H, HKV, D, maxSeqLen, scale)
    return genAttentionDecode(H, HKV, D, maxSeqLen, true)
  var nShift = 0
  while (1 shl nShift) < nRep: inc nShift
  var tShift = 0
  while (1 shl tShift) < T: inc tShift
  var hShift = 0
  while (1 shl hShift) < H: inc hShift
  let k = newKB()
  let h = k.u()
  let kvh = k.u()
  let lane = k.u()
  let qb = k.u()
  let kvb = k.u()
  let sU = k.u()
  let ki = k.u()
  let qi = k.u()
  let stepsU = k.u()
  let tmp = k.u()
  let lf = k.r()
  let sCnt = k.r()
  let sF = k.r()
  let m = k.r()
  let l = k.r()
  let dot = k.r()
  let qv = k.r()
  let kv = k.r()
  let c = k.r()
  let itf = k.r()
  let gm = k.r()
  let gl = k.r()
  let z = k.cf(0)
  let one = k.cf(1)
  let negl = k.cf(NegLarge)
  let zeroU = k.cu(0)
  k.e("utgid", h)
  k.e("ulid", lane)
  k.e("loadlid", lf)
  k.e("uloadf32", sCnt, zeroU, "@in3")
  k.e("addconst", sCnt, sCnt, 1)
  if flashT > 0:
    # tgid = t * H + h: the row adds t keys, the kv head comes from h, q/o rows start at tgid * D
    k.e("ushrk", tmp, h, hShift)
    k.e("utor", dot, tmp)
    k.e("add", sCnt, sCnt, dot)
    k.e("uandk", kvh, h, H - 1)
    k.e("ushrk", kvh, kvh, nShift)
  else:
    k.e("ushrk", kvh, h, nShift)
  k.e("umulk", qb, h, D)
  k.e("umulk", kvb, kvh, maxSeqLen * D)
  k.e("rtou", stepsU, sCnt)
  k.e("uaddk", stepsU, stepsU, T - 1)
  k.e("ushrk", stepsU, stepsU, tShift)
  k.e("utor", itf, stepsU)
  var iters = 1                                        # compile-time bound for the pow2 loop trip
  while iters < (maxSeqLen + T - 1) div T: iters *= 2
  k.cp(m, negl)
  k.e("loadconstf32", l, 0)
  # phase 1: scores
  k.e("uaddk", sU, lane, 0)
  k.e("loadconstf32", sF, 0)
  k.forN(iters, proc() =
    k.e("utor", sF, sU)
    k.e("cmplt", c, sF, sCnt)
    k.cond(c, proc() =
      k.e("umulk", ki, sU, D)
      k.e("uadd", ki, ki, kvb)
      k.e("uaddk", qi, qb, 0)
      k.e("loadconstf32", dot, 0)
      k.forN(D, proc() =
        k.e("uloadf32", qv, qi, "@in0")
        k.e("uloadf32", kv, ki, "@in1")
        k.e("mul", qv, qv, kv)
        k.e("add", dot, dot, qv)
        k.e("uaddk", qi, qi, 1)
        k.e("uaddk", ki, ki, 1))
      if flashT > 0: k.e("mulconst", dot, dot, fnum(scale))
      k.e("tgstoref32", dot, sF, 0)
      k.e("cmplt", c, m, dot)
      k.cond(c, proc() = k.cp(m, dot)))
    k.e("uaddk", sU, sU, T))
  k.tgReduceAt(lf, m, T, maxSeqLen, true)
  k.e("tgloadf32", gm, z, maxSeqLen)
  k.e("barriertg")
  # phase 2: p = exp(score - max), sum
  k.e("uaddk", sU, lane, 0)
  k.forN(iters, proc() =
    k.e("utor", sF, sU)
    k.e("cmplt", c, sF, sCnt)
    k.cond(c, proc() =
      k.e("tgloadf32", dot, sF, 0)
      k.e("sub", dot, dot, gm)
      k.e("exp", dot, dot)
      k.e("tgstoref32", dot, sF, 0)
      k.e("add", l, l, dot))
    k.e("uaddk", sU, sU, T))
  k.tgReduceAt(lf, l, T, maxSeqLen, false)
  k.e("tgloadf32", gl, z, maxSeqLen)
  k.e("barriertg")
  # phase 3: o[d] = sum_s p[s] * V[s][d] / sum
  let dLim = k.cf(float(D))
  k.e("cmplt", c, lf, dLim)
  k.cond(c, proc() =
    k.e("loadconstf32", dot, 0)
    k.e("loadconstf32", sF, 0)
    k.e("uaddk", ki, kvb, 0)
    k.e("uadd", ki, ki, lane)
    k.forR(sCnt, proc() =
      k.e("tgloadf32", qv, sF, 0)
      k.e("uloadf32", kv, ki, "@in2")
      k.e("fma", dot, qv, kv, dot)
      k.e("addconst", sF, sF, 1)
      k.e("uaddk", ki, ki, D))
    k.e("div", dot, dot, gl)
    k.e("uadd", tmp, qb, lane)
    k.e("ustoref32", dot, tmp, "@out0"))
  result = Gen(body: k.text, threads: (if flashT > 0: flashT else: 1) * H * T, ins: 4, outs: 1, tg: T,
    tgBytes: (maxSeqLen + T) * 4)

proc genMatmulQ4Mm*(M, K, N, groupSize, nGroups: int): Gen =
  ## Prefill Q4 GEMM built like the Metal kernel matmul_q4_mm_sg (simdgroup_matrix): a 32 x 32
  ## output tile per threadgroup, K walked in steps of 32, the activation tile and the dequantised
  ## weight tile staged in threadgroup memory, four simdgroups (2 x 2) multiplying 16 x 16 blocks
  ## with 8 x 8 simdgroup matrices. Same ports as genMatmulQ4Par. Needs K % 32 == 0 and a
  ## power-of-two group of at least 8 that divides K; otherwise the lane-per-column kernel.
  let gs = if groupSize > 0: groupSize else: K
  if K mod 32 != 0 or K mod gs != 0 or (gs and (gs - 1)) != 0 or gs < 8:
    return genMatmulQ4Par(M, K, N, groupSize, nGroups)
  var gShift = 0
  while (1 shl gShift) < gs: inc gShift
  const LD = 36
  const WS = 32 * LD
  let rowBytes = K div 2
  let wqF = (N * rowBytes + 3) div 4
  let scalesAddr = "@in1+" & $wqF
  let zerosAddr = "@in1+" & $(wqF + N * nGroups)
  let nTN = (N + 31) div 32
  let nTM = (M + 31) div 32
  let k = newKB()
  let tg = k.u()
  let lid = k.u()
  let sg = k.u()
  let tm = k.u()
  let tn = k.u()
  let n0 = k.u()
  let m0 = k.u()
  let lr = k.u()
  let lc = k.u()
  let sm = k.u()
  let sn = k.u()
  let k0 = k.u()
  let gk = k.u()
  let t1 = k.u()
  let t2 = k.u()
  let xoff0 = k.u()
  let xoff1 = k.u()
  let woff0 = k.u()
  let woff1 = k.u()
  let rowW = k.u()
  let wqb = k.u()
  let gb = k.u()
  let pk = k.u()
  let xi = k.r()
  let tf = k.r()
  let tq = k.r()
  let tilesN = k.cf(float(nTN))
  let mf = k.cf(float(M))
  let nf = k.cf(float(N))
  let one = k.cf(1)
  let c = k.r()
  let c2 = k.r()
  let sc = k.r()
  let zp = k.r()
  var v: seq[int]
  for j in 0 ..< 8: v.add k.r()
  k.e("utgid", tg)
  k.e("ulid", lid)
  k.e("simdid", sg)
  k.e("utor", tf, tg)
  k.e("div", tq, tf, tilesN)
  k.e("rtou", tm, tq)                    # truncating: tile row = tg / nTN
  k.e("utor", tq, tm)
  k.e("mul", tq, tq, tilesN)
  k.e("sub", tf, tf, tq)
  k.e("rtou", tn, tf)                    # tile column = tg - tm * nTN
  k.e("umulk", n0, tn, 32)
  k.e("umulk", m0, tm, 32)
  k.e("ushrk", lr, lid, 2)
  k.e("uandk", lc, lid, 3)
  k.e("umulk", lc, lc, 8)
  k.e("ushrk", sm, sg, 1)
  k.e("uandk", sn, sg, 1)
  k.e("umulk", t1, lr, LD)
  k.e("uadd", t1, t1, lc)
  k.e("utor", xi, t1)
  k.e("umulk", xoff0, sm, 16 * LD)
  k.e("uaddk", xoff1, xoff0, 8 * LD)
  k.e("umulk", woff0, sn, 16 * LD)
  k.e("uaddk", woff0, woff0, WS)
  k.e("uaddk", woff1, woff0, 8 * LD)
  k.e("uconst", wqb, "@in1")
  k.e("umulk", wqb, wqb, 4)
  k.e("uaddk", t1, n0, 0)
  k.e("uadd", t1, t1, lr)                # n = n0 + lr
  k.e("umulk", rowW, t1, rowBytes)
  k.e("uadd", rowW, rowW, wqb)
  k.e("uconst", k0, 0)
  for i in 0 ..< 4: k.e("mmzero", i)
  k.forN(K div 32, proc() =
    k.e("uadd", gk, k0, lc)
    # activation tile row lr (zeros past M)
    for j in 0 ..< 8: k.e("loadconstf32", v[j], 0)
    k.e("uadd", t1, m0, lr)
    k.e("utor", tf, t1)
    k.e("cmplt", c, tf, mf)
    k.cond(c, proc() =
      k.e("umulk", t1, t1, K)
      k.e("uadd", t1, t1, gk)
      for j in 0 ..< 8: k.e("uloadf32", v[j], t1, "@in0+" & $j))
    for j in 0 ..< 8: k.e("tgstoref32", v[j], xi, j)
    # dequantised weight tile row lr (zeros past N)
    for j in 0 ..< 8: k.e("loadconstf32", v[j], 0)
    k.e("uadd", t1, n0, lr)
    k.e("utor", tf, t1)
    k.e("cmplt", c, tf, nf)
    k.cond(c, proc() =
      k.e("ushrk", t2, gk, 1)
      k.e("uadd", t2, t2, rowW)
      k.e("uloadu32", pk, t2, 0)
      k.e("umulk", t1, t1, nGroups)
      k.e("ushrk", gb, gk, gShift)
      k.e("uadd", t1, t1, gb)
      k.e("uloadf32", sc, t1, scalesAddr)
      k.e("uloadf32", zp, t1, zerosAddr)
      for j in 0 ..< 8:
        k.e("ushrk", t2, pk, 4 * j)
        if j < 7: k.e("uandk", t2, t2, 15)
        k.e("utor", v[j], t2)
        k.e("sub", v[j], v[j], zp)
        k.e("mul", v[j], v[j], sc))
    for j in 0 ..< 8: k.e("tgstoref32", v[j], xi, WS + j)
    k.e("barriertg")
    for kk in 0 ..< 4:
      k.e("uaddk", t1, xoff0, kk * 8)
      k.e("mmload", 4, t1, LD)
      k.e("uaddk", t1, xoff1, kk * 8)
      k.e("mmload", 5, t1, LD)
      k.e("uaddk", t1, woff0, kk * 8)
      k.e("mmloadt", 6, t1, LD)
      k.e("uaddk", t1, woff1, kk * 8)
      k.e("mmloadt", 7, t1, LD)
      for i in 0 ..< 2:
        for j in 0 ..< 2:
          k.e("mmmac", i * 2 + j, 4 + i, 6 + j)
    k.e("barriertg")
    k.e("uaddk", k0, k0, 32))
  # epilogue: a full tile goes straight to memory, a partial one through threadgroup memory
  k.e("utor", tf, m0)
  k.e("addconst", tf, tf, 31)
  k.e("cmplt", c, tf, mf)
  k.e("utor", tf, n0)
  k.e("addconst", tf, tf, 31)
  k.e("cmplt", c2, tf, nf)
  k.e("mul", c, c, c2)
  k.cond(c, proc() =
    k.e("uconst", t1, "@out0")
    k.e("umulk", t2, sm, 16)
    k.e("uadd", t2, t2, m0)
    k.e("umulk", t2, t2, N)
    k.e("uadd", t1, t1, t2)
    k.e("umulk", t2, sn, 16)
    k.e("uadd", t2, t2, n0)
    k.e("uadd", t1, t1, t2)
    for i in 0 ..< 2:
      for j in 0 ..< 2:
        k.e("uaddk", t2, t1, i * 8 * N + j * 8)
        k.e("mmstoremem", i * 2 + j, t2, N))
  k.e("sub", c, one, c)
  k.cond(c, proc() =
    for i in 0 ..< 2:
      for j in 0 ..< 2:
        k.e("umulk", t2, sn, 16)
        k.e("uadd", t2, t2, xoff0)
        k.e("uaddk", t2, t2, i * 8 * LD + j * 8)
        k.e("mmstoretg", i * 2 + j, t2, LD)
    k.e("barriertg")
    for it in 0 ..< 8:
      # element (r, col) of the tile: r = lid / 32 + 4 * it, col = lid % 32
      k.e("ushrk", t1, lid, 5)
      k.e("uaddk", t1, t1, 4 * it)          # r
      k.e("uandk", t2, lid, 31)             # col
      k.e("uadd", gk, m0, t1)               # global row
      k.e("uadd", gb, n0, t2)               # global col
      k.e("utor", tf, gk)
      k.e("cmplt", c, tf, mf)
      k.e("utor", tf, gb)
      k.e("cmplt", c2, tf, nf)
      k.e("mul", c, c, c2)
      k.cond(c, proc() =
        k.e("umulk", t1, t1, LD)
        k.e("uadd", t1, t1, t2)
        k.e("utor", tf, t1)
        k.e("tgloadf32", sc, tf, 0)
        k.e("umulk", gk, gk, N)
        k.e("uadd", gk, gk, gb)
        k.e("ustoref32", sc, gk, "@out0")))
  result = Gen(body: k.text, threads: nTM * nTN * 128, ins: 2, outs: 1, tg: 128, tgBytes: 2 * WS * 4)

proc genMatmulNTSimd*(K, N: int; rows = 4; sg = 2): Gen =
  ## f32 GEMV C[1,N] = A[1,K] * B[N,K]^T like the Metal simdgroup GEMV: one simdgroup per `rows`
  ## output rows, a lane takes 4 floats per step with an fma chain, lanes combined by simd_sum.
  ## Needs K % 4 == 0 and N % rows == 0; otherwise the threadgroup kernel.
  if K mod 4 != 0 or N mod rows != 0:
    return genMatmulNTPar(1, K, N)
  var pow2Steps = 1
  while pow2Steps < (K + 63) div 64: pow2Steps *= 2     # see genMatmulQ4Simd: Intel needs 2^n trips
  let k = newKB()
  let lane = k.u()
  let sid = k.u()
  let sw = k.u()
  let stepU = k.u()
  let row0 = k.u()
  let kk = k.u()
  let bi = k.u()
  let nr = k.u()
  var sumR: seq[int]
  for r in 0 ..< rows: sumR.add k.cf(0)
  var a: seq[int]
  var b: seq[int]
  for j in 0 ..< 4:
    a.add k.r()
    b.add k.r()
  let c = k.r()
  let kf = k.r()
  let kLimit = k.cf(float(K))
  let rowf = k.r()
  let nLimit = k.cf(float(N))
  let one = k.cf(1)
  let d = k.r()
  k.e("simdlane", lane)
  k.e("simdw", sw)
  k.e("simdgid", sid)
  k.e("umulk", row0, sid, rows)
  k.e("umulk", stepU, sw, 4)
  k.e("umulk", kk, lane, 4)
  k.e("utor", rowf, row0)
  k.e("cmplt", c, rowf, nLimit)
  k.cond(c, proc() =
    k.forN(pow2Steps, proc() =
      k.e("utor", kf, kk)
      k.e("cmplt", c, kf, kLimit)
      k.cond(c, proc() =
        for j in 0 ..< 4:
          k.e("uloadf32", a[j], kk, "@in0+" & $j)
        for r in 0 ..< rows:
          k.e("uaddk", nr, row0, r)
          k.e("umulk", bi, nr, K)
          k.e("uadd", bi, bi, kk)
          for j in 0 ..< 4:
            k.e("uloadf32", b[j], bi, "@in1+" & $j)
            k.e("fma", sumR[r], a[j], b[j], sumR[r]))
      k.e("uadd", kk, kk, stepU))
    k.e("utor", kf, lane)
    k.e("cmplt", c, kf, one)
    for r in 0 ..< rows:
      k.e("simdsum", d, sumR[r])
      k.cond(c, proc() =
        k.e("uaddk", nr, row0, r)
        k.e("ustoref32", d, nr, "@out0")))
  result = Gen(body: k.text, threads: ((N div rows + sg - 1) div sg) * sg * 32, ins: 2, outs: 1,
    tg: sg * 32, tgBytes: 16)

proc genMatmulNTMm*(M, K, N: int): Gen =
  ## f32 GEMM C[M,N] = A[M,K] * B[N,K]^T built like the Metal simdgroup GEMM: 32 x 32 output tile
  ## per threadgroup, K in steps of 32, both tiles staged in threadgroup memory, four simdgroups
  ## (2 x 2) multiplying 16 x 16 blocks with 8 x 8 simdgroup matrices. Same ports as genMatmulNT
  ## (in0=A in1=B out0=C). Needs K % 32 == 0; otherwise the lane-per-column kernel.
  if K mod 32 != 0:
    return genMatmulNTPar(M, K, N)
  const LD = 36
  const WS = 32 * LD
  let nTN = (N + 31) div 32
  let nTM = (M + 31) div 32
  let k = newKB()
  let tg = k.u()
  let lid = k.u()
  let sg = k.u()
  let tm = k.u()
  let tn = k.u()
  let n0 = k.u()
  let m0 = k.u()
  let lr = k.u()
  let lc = k.u()
  let sm = k.u()
  let sn = k.u()
  let k0 = k.u()
  let gk = k.u()
  let t1 = k.u()
  let t2 = k.u()
  let xoff0 = k.u()
  let xoff1 = k.u()
  let woff0 = k.u()
  let woff1 = k.u()
  let rowW = k.u()
  let gb = k.u()
  let xi = k.r()
  let tf = k.r()
  let tq = k.r()
  let tilesN = k.cf(float(nTN))
  let mf = k.cf(float(M))
  let nf = k.cf(float(N))
  let one = k.cf(1)
  let c = k.r()
  let c2 = k.r()
  let sc = k.r()
  var v: seq[int]
  for j in 0 ..< 8: v.add k.r()
  k.e("utgid", tg)
  k.e("ulid", lid)
  k.e("simdid", sg)
  k.e("utor", tf, tg)
  k.e("div", tq, tf, tilesN)
  k.e("rtou", tm, tq)                    # truncating: tile row = tg / nTN
  k.e("utor", tq, tm)
  k.e("mul", tq, tq, tilesN)
  k.e("sub", tf, tf, tq)
  k.e("rtou", tn, tf)                    # tile column = tg - tm * nTN
  k.e("umulk", n0, tn, 32)
  k.e("umulk", m0, tm, 32)
  k.e("ushrk", lr, lid, 2)
  k.e("uandk", lc, lid, 3)
  k.e("umulk", lc, lc, 8)
  k.e("ushrk", sm, sg, 1)
  k.e("uandk", sn, sg, 1)
  k.e("umulk", t1, lr, LD)
  k.e("uadd", t1, t1, lc)
  k.e("utor", xi, t1)
  k.e("umulk", xoff0, sm, 16 * LD)
  k.e("uaddk", xoff1, xoff0, 8 * LD)
  k.e("umulk", woff0, sn, 16 * LD)
  k.e("uaddk", woff0, woff0, WS)
  k.e("uaddk", woff1, woff0, 8 * LD)
  k.e("uconst", k0, 0)
  for i in 0 ..< 4: k.e("mmzero", i)
  k.forN(K div 32, proc() =
    k.e("uadd", gk, k0, lc)
    # activation tile row lr (zeros past M)
    for j in 0 ..< 8: k.e("loadconstf32", v[j], 0)
    k.e("uadd", t1, m0, lr)
    k.e("utor", tf, t1)
    k.e("cmplt", c, tf, mf)
    k.cond(c, proc() =
      k.e("umulk", t1, t1, K)
      k.e("uadd", t1, t1, gk)
      for j in 0 ..< 8: k.e("uloadf32", v[j], t1, "@in0+" & $j))
    for j in 0 ..< 8: k.e("tgstoref32", v[j], xi, j)
    # weight tile row lr (zeros past N)
    for j in 0 ..< 8: k.e("loadconstf32", v[j], 0)
    k.e("uadd", t1, n0, lr)
    k.e("utor", tf, t1)
    k.e("cmplt", c, tf, nf)
    k.cond(c, proc() =
      k.e("umulk", t1, t1, K)
      k.e("uadd", t1, t1, gk)
      for j in 0 ..< 8: k.e("uloadf32", v[j], t1, "@in1+" & $j))
    for j in 0 ..< 8: k.e("tgstoref32", v[j], xi, WS + j)
    k.e("barriertg")
    for kk in 0 ..< 4:
      k.e("uaddk", t1, xoff0, kk * 8)
      k.e("mmload", 4, t1, LD)
      k.e("uaddk", t1, xoff1, kk * 8)
      k.e("mmload", 5, t1, LD)
      k.e("uaddk", t1, woff0, kk * 8)
      k.e("mmloadt", 6, t1, LD)
      k.e("uaddk", t1, woff1, kk * 8)
      k.e("mmloadt", 7, t1, LD)
      for i in 0 ..< 2:
        for j in 0 ..< 2:
          k.e("mmmac", i * 2 + j, 4 + i, 6 + j)
    k.e("barriertg")
    k.e("uaddk", k0, k0, 32))
  # epilogue: a full tile goes straight to memory, a partial one through threadgroup memory
  k.e("utor", tf, m0)
  k.e("addconst", tf, tf, 31)
  k.e("cmplt", c, tf, mf)
  k.e("utor", tf, n0)
  k.e("addconst", tf, tf, 31)
  k.e("cmplt", c2, tf, nf)
  k.e("mul", c, c, c2)
  k.cond(c, proc() =
    k.e("uconst", t1, "@out0")
    k.e("umulk", t2, sm, 16)
    k.e("uadd", t2, t2, m0)
    k.e("umulk", t2, t2, N)
    k.e("uadd", t1, t1, t2)
    k.e("umulk", t2, sn, 16)
    k.e("uadd", t2, t2, n0)
    k.e("uadd", t1, t1, t2)
    for i in 0 ..< 2:
      for j in 0 ..< 2:
        k.e("uaddk", t2, t1, i * 8 * N + j * 8)
        k.e("mmstoremem", i * 2 + j, t2, N))
  k.e("sub", c, one, c)
  k.cond(c, proc() =
    for i in 0 ..< 2:
      for j in 0 ..< 2:
        k.e("umulk", t2, sn, 16)
        k.e("uadd", t2, t2, xoff0)
        k.e("uaddk", t2, t2, i * 8 * LD + j * 8)
        k.e("mmstoretg", i * 2 + j, t2, LD)
    k.e("barriertg")
    for it in 0 ..< 8:
      # element (r, col) of the tile: r = lid / 32 + 4 * it, col = lid % 32
      k.e("ushrk", t1, lid, 5)
      k.e("uaddk", t1, t1, 4 * it)          # r
      k.e("uandk", t2, lid, 31)             # col
      k.e("uadd", gk, m0, t1)               # global row
      k.e("uadd", gb, n0, t2)               # global col
      k.e("utor", tf, gk)
      k.e("cmplt", c, tf, mf)
      k.e("utor", tf, gb)
      k.e("cmplt", c2, tf, nf)
      k.e("mul", c, c, c2)
      k.cond(c, proc() =
        k.e("umulk", t1, t1, LD)
        k.e("uadd", t1, t1, t2)
        k.e("utor", tf, t1)
        k.e("tgloadf32", sc, tf, 0)
        k.e("umulk", gk, gk, N)
        k.e("uadd", gk, gk, gb)
        k.e("ustoref32", sc, gk, "@out0")))
  result = Gen(body: k.text, threads: nTM * nTN * 128, ins: 2, outs: 1, tg: 128, tgBytes: 2 * WS * 4)

proc genArgmax*(outer, dimSize, inner: int): Gen =
  ## sessionArgmaxIndexEnc(x, val, idx, outer, dimSize, inner): first maximum along dim.
  ## ports in0=x out0=val [outer*inner] out1=idx [outer*inner] (index as float); lane = outer row.
  let k = newKB()
  let xb = k.u()
  let ob = k.u()
  let xi = k.u()
  let best = k.r()
  let cur = k.r()
  let idx = k.r()
  let v = k.r()
  let c = k.r()
  k.e("umulk", xb, 0, dimSize * inner)
  k.e("umulk", ob, 0, inner)
  k.forN(inner, proc() =
    k.e("uaddk", xi, xb, 0)
    k.e("uloadf32", best, xi, "@in0")
    k.e("loadconstf32", idx, 0)
    k.e("loadconstf32", cur, 0)
    k.forN(dimSize - 1, proc() =
      k.e("uaddk", xi, xi, inner)
      k.e("addconst", cur, cur, 1)
      k.e("uloadf32", v, xi, "@in0")
      k.e("cmplt", c, best, v)
      k.cond(c, proc() =
        k.cp(best, v)
        k.cp(idx, cur)))
    k.e("ustoref32", best, ob, "@out0")
    k.e("ustoref32", idx, ob, "@out1")
    k.e("uaddk", ob, ob, 1)
    k.e("uaddk", xb, xb, 1))
  k.done(outer, 1, 2)

proc genCompare*(op, n: int): Gen =
  ## sessionCompareEnc(a, b, mask, op, n): op 0 >, 1 <, 2 ==, 3 >=, 4 <=; mask 1.0 / 0.0.
  let k = newKB()
  let a = k.r()
  let b = k.r()
  let c = k.r()
  let d = k.r()
  k.e("uloadf32", a, 0, "@in0")
  k.e("uloadf32", b, 0, "@in1")
  case op
  of 0: k.e("cmplt", c, b, a)
  of 1: k.e("cmplt", c, a, b)
  of 2, 3, 4:
    # ==: 1 - (a<b) - (b<a); >=: 1 - (a<b); <=: 1 - (b<a)
    k.e("loadconstf32", c, 0)
    if op != 4:
      k.e("cmplt", d, a, b)
      k.e("add", c, c, d)
    if op != 3:
      k.e("cmplt", d, b, a)
      k.e("add", c, c, d)
    k.e("neg", c, c)
    k.e("addconst", c, c, 1)
  else:
    raise newException(GenError, "unknown compare op " & $op)
  k.e("ustoref32", c, 0, "@out0")
  k.done(n, 2, 1)

proc genSplitSlice*(dimTotal, thisSize, offset, inner, n: int): Gen =
  ## sessionSplitSliceEnc(src, dst, dimTotal, thisSize, offset, inner, n):
  ## dst[o, d, i] = src[o, offset + d, i]. ports in0 -> out0; lane = outer index.
  let run = thisSize * inner
  let k = newKB()
  let s = k.u()
  let d = k.u()
  let v = k.r()
  k.e("umulk", s, 0, dimTotal * inner)
  k.e("uaddk", s, s, offset * inner)
  k.e("umulk", d, 0, run)
  k.forN(run, proc() =
    k.e("uloadf32", v, s, "@in0")
    k.e("ustoref32", v, d, "@out0")
    k.e("uaddk", s, s, 1)
    k.e("uaddk", d, d, 1))
  k.done(n div run, 1, 1)

proc genConcat*(srcBase, srcDimSize, srcDimOffset: seq[int]; totalDim, inner, n: int): Gen =
  ## gpuConcat: dst[o, dOff_t + d, i] = flat[srcBase_t + (o*size_t + d)*inner + i].
  ## ports in0 = the flat concatenation of the sources -> out0; lane = outer index.
  let k = newKB()
  let s = k.u()
  let d = k.u()
  let v = k.r()
  for t in 0 ..< srcBase.len:
    let run = srcDimSize[t] * inner
    k.e("umulk", s, 0, run)
    k.e("uaddk", s, s, srcBase[t])
    k.e("umulk", d, 0, totalDim * inner)
    k.e("uaddk", d, d, srcDimOffset[t] * inner)
    k.forN(run, proc() =
      k.e("uloadf32", v, s, "@in0")
      k.e("ustoref32", v, d, "@out0")
      k.e("uaddk", s, s, 1)
      k.e("uaddk", d, d, 1))
  k.done(n div (totalDim * inner), 1, 1)

proc genPermute*(newStrides, oldStrides, dimsMap: seq[int]; n: int): Gen =
  ## sessionPermuteEnc(src, dst, newStrides, oldStrides, dimsMap, ndim, n): the tables are
  ## baked in. dst[linear] = src[sum_i idx_i * oldStrides[dimsMap[i]]] with idx from newStrides.
  ## ports in0 -> out0; lane = the leading output index.
  let nd = newStrides.len
  var size = newSeq[int](nd)
  var sstr = newSeq[int](nd)
  for i in 0 ..< nd:
    size[i] = if i == 0: n div newStrides[0] else: newStrides[i-1] div newStrides[i]
    sstr[i] = oldStrides[dimsMap[i]]
  let k = newKB()
  let v = k.r()
  let s0 = k.u()
  let d0 = k.u()
  k.e("umulk", s0, 0, sstr[0])
  k.e("umulk", d0, 0, newStrides[0])
  proc level(i, su, du: int) =
    if i == nd:
      k.e("uloadf32", v, su, "@in0")
      k.e("ustoref32", v, du, "@out0")
      return
    let s2 = k.u()
    let d2 = k.u()
    k.e("uaddk", s2, su, 0)
    k.e("uaddk", d2, du, 0)
    k.forN(size[i], proc() =
      level(i + 1, s2, d2)
      k.e("uaddk", s2, s2, sstr[i])
      k.e("uaddk", d2, d2, newStrides[i]))
    k.fu(s2, d2)
  level(1, s0, d0)
  k.done(size[0], 1, 1)

proc xorU(k: KB; dst, a, b: int) =
  ## dst = a xor b over 32 bits (the VM has no register xor).
  let i = k.u()
  let ba = k.u()
  let bb = k.u()
  let p2 = k.u()
  let t = k.u()
  k.e("uconst", dst, 0)
  k.e("uconst", i, 0)
  k.e("uconst", p2, 1)
  k.forN(32, proc() =
    k.e("ushr", ba, a, i)
    k.e("uandk", ba, ba, 1)
    k.e("ushr", bb, b, i)
    k.e("uandk", bb, bb, 1)
    k.e("uadd", ba, ba, bb)
    k.e("uandk", ba, ba, 1)
    k.e("umul", t, ba, p2)
    k.e("uadd", dst, dst, t)
    k.e("uaddk", i, i, 1)
    k.e("uadd", p2, p2, p2))
  k.fu(i, ba, bb, p2, t)

proc genDropoutMask*(n: int; p: float; seed: uint32): Gen =
  ## sessionDropoutMaskEnc(mask, n, p, seed): mask[id] = (u < p) ? 0 : 1/(1-p), u from two
  ## xorshift32 rounds seeded by seed ^ (id*2654435761 + 1). ports: out0 only.
  let scale = if p < 1.0: 1.0 / (1.0 - p) else: 0.0
  let k = newKB()
  let st = k.u()
  let t = k.u()
  let sd = k.u()
  let tmp = k.u()
  k.e("umulk", t, 0, 2654435761'u32)
  k.e("uaddk", t, t, 1)
  k.e("uconst", sd, seed)
  k.xorU(st, sd, t)
  for _ in 0 ..< 2:
    k.e("ushlk", t, st, 13)
    k.xorU(tmp, st, t)
    k.e("ushrk", t, tmp, 17)
    k.xorU(st, tmp, t)
    k.e("ushlk", t, st, 5)
    k.xorU(tmp, st, t)
    k.e("uaddk", st, tmp, 0)
  k.e("ushrk", st, st, 8)
  let u = k.r()
  let c = k.r()
  let pr = k.cf(p)
  k.e("utor", u, st)
  k.e("mulconst", u, u, fnum(1.0 / 16777216.0))
  k.e("cmplt", c, u, pr)
  k.e("neg", c, c)
  k.e("addconst", c, c, 1)
  k.e("mulconst", c, c, fnum(scale))
  k.e("ustoref32", c, 0, "@out0")
  k.done(n, 0, 1)

proc genApflu*(n: int; alpha, beta: float): Gen =
  ## sessionApfluEnc: y = v > 0 ? v*(1 + alpha*v) : beta*v*exp(v). ports in0 -> out0.
  let k = newKB()
  let v = k.r()
  let o = k.r()
  let t = k.r()
  let z = k.cf(0)
  let c = k.r()
  k.e("uloadf32", v, 0, "@in0")
  k.e("exp", t, v)
  k.e("mul", o, v, t)
  k.e("mulconst", o, o, fnum(beta))
  k.e("cmplt", c, z, v)
  k.cond(c, proc() =
    k.e("mulconst", t, v, fnum(alpha))
    k.e("addconst", t, t, 1)
    k.e("mul", o, v, t))
  k.e("ustoref32", o, 0, "@out0")
  k.done(n, 1, 1)

proc genApfluBackward*(n: int; alpha, beta: float): Gen =
  ## sessionApfluBackwardEnc: dx = v > 0 ? dy*(1 + 2*alpha*v) : dy*beta*exp(v)*(1 + v).
  ## ports in0=x in1=dy -> out0.
  let k = newKB()
  let v = k.r()
  let dy = k.r()
  let o = k.r()
  let t = k.r()
  let z = k.cf(0)
  let c = k.r()
  k.e("uloadf32", v, 0, "@in0")
  k.e("uloadf32", dy, 0, "@in1")
  k.e("exp", t, v)
  k.e("mul", o, dy, t)
  k.e("mulconst", o, o, fnum(beta))
  k.e("addconst", t, v, 1)
  k.e("mul", o, o, t)
  k.e("cmplt", c, z, v)
  k.cond(c, proc() =
    k.e("mulconst", t, v, fnum(2.0 * alpha))
    k.e("addconst", t, t, 1)
    k.e("mul", o, dy, t))
  k.e("ustoref32", o, 0, "@out0")
  k.done(n, 2, 1)

proc rowStats(k: KB; base, cols: int; mean, inv: int; eps: float) =
  ## mean and 1/sqrt(var + eps) of the row starting at U[base] of in1 (x).
  let i = k.u()
  let v = k.r()
  let vr = k.cf(0)
  k.e("loadconstf32", mean, 0)
  k.e("uaddk", i, base, 0)
  k.forN(cols, proc() =
    k.e("uloadf32", v, i, "@in1")
    k.e("add", mean, mean, v)
    k.e("uaddk", i, i, 1))
  k.e("mulconst", mean, mean, fnum(1.0 / float(cols)))
  k.e("uaddk", i, base, 0)
  k.forN(cols, proc() =
    k.e("uloadf32", v, i, "@in1")
    k.e("sub", v, v, mean)
    k.e("mul", v, v, v)
    k.e("add", vr, vr, v)
    k.e("uaddk", i, i, 1))
  k.e("mulconst", vr, vr, fnum(1.0 / float(cols)))
  k.e("addconst", inv, vr, fnum(eps))
  k.e("rsqrt", inv, inv)
  k.fu(i)
  k.fr(v, vr)

proc genLayernormBackwardDx*(rows, cols: int; eps: float): Gen =
  ## sessionLayernormBackwardEnc, dx part. ports in0=dy in1=x in2=gamma -> out0=dx; lane = row.
  let k = newKB()
  let base = k.u()
  let i = k.u()
  let j = k.u()
  let mean = k.r()
  let inv = k.r()
  let s1 = k.cf(0)
  let s2 = k.cf(0)
  let dr = k.r()
  let g = k.r()
  let x = k.r()
  let t = k.r()
  k.e("umulk", base, 0, cols)
  k.rowStats(base, cols, mean, inv, eps)
  k.e("uaddk", i, base, 0)
  k.e("uconst", j, 0)
  k.forN(cols, proc() =
    k.e("uloadf32", dr, i, "@in0")
    k.e("uloadf32", g, j, "@in2")
    k.e("mul", g, g, dr)
    k.e("mul", g, g, inv)
    k.e("add", s1, s1, g)
    k.e("uloadf32", x, i, "@in1")
    k.e("sub", x, x, mean)
    k.e("mul", x, x, g)
    k.e("add", s2, s2, x)
    k.e("uaddk", i, i, 1)
    k.e("uaddk", j, j, 1))
  k.e("mul", t, inv, inv)
  k.e("neg", t, t)
  k.e("mulconst", t, t, fnum(1.0 / float(cols)))
  k.e("mul", s2, s2, t)
  k.e("mulconst", s1, s1, fnum(1.0 / float(cols)))
  k.e("uaddk", i, base, 0)
  k.e("uconst", j, 0)
  k.forN(cols, proc() =
    k.e("uloadf32", dr, i, "@in0")
    k.e("uloadf32", g, j, "@in2")
    k.e("mul", g, g, dr)
    k.e("mul", g, g, inv)
    k.e("sub", g, g, s1)
    k.e("uloadf32", x, i, "@in1")
    k.e("sub", x, x, mean)
    k.e("mul", x, x, s2)
    k.e("add", g, g, x)
    k.e("ustoref32", g, i, "@out0")
    k.e("uaddk", i, i, 1)
    k.e("uaddk", j, j, 1))
  k.done(rows, 3, 1)

proc genLayernormBackwardParams*(rows, cols: int; eps: float): Gen =
  ## sessionLayernormBackwardEnc, dgamma/dbeta part. ports in0=dy in1=x -> out0=dgamma out1=dbeta;
  ## lane = column (row statistics are recomputed per row).
  let k = newKB()
  let base = k.u()
  let ci = k.u()
  let mean = k.r()
  let inv = k.r()
  let dg = k.cf(0)
  let db = k.cf(0)
  let x = k.r()
  let dr = k.r()
  k.e("uconst", base, 0)
  k.e("uaddk", ci, 0, 0)
  k.forN(rows, proc() =
    k.rowStats(base, cols, mean, inv, eps)
    k.e("uloadf32", dr, ci, "@in0")
    k.e("uloadf32", x, ci, "@in1")
    k.e("sub", x, x, mean)
    k.e("mul", x, x, inv)
    k.e("mul", x, x, dr)
    k.e("add", dg, dg, x)
    k.e("add", db, db, dr)
    k.e("uaddk", ci, ci, cols)
    k.e("uaddk", base, base, cols))
  k.e("ustoref32", dg, 0, "@out0")
  k.e("ustoref32", db, 0, "@out1")
  k.done(cols, 2, 2)

proc genAttentionDecodeGqa*(H, S, D: int; scale: float): Gen =
  ## sessionAttentionDecodeGqaEnc(q, k, v, o, scores, H, S, D, scale): all heads share one
  ## K/V [S, D]. ports in0=q [H*D] in1=K in2=V -> out0=o [H*D], out1=scores [H*S] (the
  ## unnormalised exp(score - max), as the shader leaves it); lane = head.
  let k = newKB()
  let qb = k.u()
  let sb = k.u()
  let qi = k.u()
  let ki = k.u()
  let si = k.u()
  let vi = k.u()
  let oi = k.u()
  let mx = k.cf(-1e30)
  let tot = k.cf(0)
  let dot = k.r()
  let qv = k.r()
  let kv = k.r()
  let c = k.r()
  k.e("umulk", qb, 0, D)
  k.e("umulk", sb, 0, S)
  k.e("uconst", ki, 0)
  k.e("uaddk", si, sb, 0)
  k.forN(S, proc() =
    k.e("loadconstf32", dot, 0)
    k.e("uaddk", qi, qb, 0)
    k.forN(D, proc() =
      k.e("uloadf32", qv, qi, "@in0")
      k.e("uloadf32", kv, ki, "@in1")
      k.e("mul", qv, qv, kv)
      k.e("add", dot, dot, qv)
      k.e("uaddk", qi, qi, 1)
      k.e("uaddk", ki, ki, 1))
    k.e("mulconst", dot, dot, fnum(scale))
    k.e("ustoref32", dot, si, "@out1")
    k.e("cmplt", c, mx, dot)
    k.cond(c, proc() = k.cp(mx, dot))
    k.e("uaddk", si, si, 1))
  k.e("uaddk", si, sb, 0)
  k.forN(S, proc() =
    k.e("uloadf32", dot, si, "@out1")
    k.e("sub", dot, dot, mx)
    k.e("exp", dot, dot)
    k.e("ustoref32", dot, si, "@out1")
    k.e("add", tot, tot, dot)
    k.e("uaddk", si, si, 1))
  k.e("uaddk", oi, qb, 0)
  k.forN(D, proc() =
    k.e("loadconstf32", qv, 0)
    k.e("ustoref32", qv, oi, "@out0")
    k.e("uaddk", oi, oi, 1))
  k.e("uaddk", si, sb, 0)
  k.e("uconst", vi, 0)
  k.forN(S, proc() =
    k.e("uloadf32", dot, si, "@out1")
    k.e("uaddk", oi, qb, 0)
    k.forN(D, proc() =
      k.e("uloadf32", kv, vi, "@in2")
      k.e("mul", kv, kv, dot)
      k.e("uloadf32", qv, oi, "@out0")
      k.e("add", qv, qv, kv)
      k.e("ustoref32", qv, oi, "@out0")
      k.e("uaddk", vi, vi, 1)
      k.e("uaddk", oi, oi, 1))
    k.e("uaddk", si, si, 1))
  k.e("uaddk", oi, qb, 0)
  k.forN(D, proc() =
    k.e("uloadf32", qv, oi, "@out0")
    k.e("div", qv, qv, tot)
    k.e("ustoref32", qv, oi, "@out0")
    k.e("uaddk", oi, oi, 1))
  k.done(H, 3, 2)

proc genAttentionFused*(B, H, S, D: int; scale: float): Gen =
  ## sessionAttentionFusedEnc(q, k, v, o, sMatrix, B, H, S, D, scale): causal attention over
  ## [B*H, S, D]; sMatrix [B*H, S, S] receives the softmax probabilities (zero above the
  ## diagonal). ports in0=q in1=k in2=v -> out0=o out1=sMatrix; lane = query row ti.
  let BH = B * H
  let k = newKB()
  let tiR = k.r()
  let cnt = k.r()
  let rest = k.r()
  let baseI = k.u()
  let baseS = k.u()
  let qb = k.u()
  let row = k.u()
  let qi = k.u()
  let ki = k.u()
  let si = k.u()
  let oi = k.u()
  let kb = k.u()
  let mx = k.r()
  let tot = k.r()
  let dot = k.r()
  let qv = k.r()
  let kv = k.r()
  let c = k.r()
  let acc = k.r()
  k.e("utor", tiR, 0)
  k.e("mulconst", cnt, tiR, 1)
  k.e("addconst", cnt, cnt, 1)                       # ti + 1
  k.e("mulconst", rest, tiR, -1)
  k.e("addconst", rest, rest, S - 1)                 # S - 1 - ti
  k.e("uconst", baseI, 0)
  k.e("uconst", baseS, 0)
  k.forN(BH, proc() =
    k.e("umulk", qb, 0, D)
    k.e("uadd", qb, qb, baseI)
    k.e("umulk", row, 0, S)
    k.e("uadd", row, row, baseS)
    k.e("loadconstf32", mx, fnum(-1e30))
    k.e("loadconstf32", tot, 0)
    k.e("uaddk", si, row, 0)
    k.e("uaddk", kb, baseI, 0)
    k.forR(cnt, proc() =
      k.e("loadconstf32", dot, 0)
      k.e("uaddk", qi, qb, 0)
      k.e("uaddk", ki, kb, 0)
      k.forN(D, proc() =
        k.e("uloadf32", qv, qi, "@in0")
        k.e("uloadf32", kv, ki, "@in1")
        k.e("mul", qv, qv, kv)
        k.e("add", dot, dot, qv)
        k.e("uaddk", qi, qi, 1)
        k.e("uaddk", ki, ki, 1))
      k.e("mulconst", dot, dot, fnum(scale))
      k.e("ustoref32", dot, si, "@out1")
      k.e("cmplt", c, mx, dot)
      k.cond(c, proc() = k.cp(mx, dot))
      k.e("uaddk", si, si, 1)
      k.e("uaddk", kb, kb, D))
    k.e("uaddk", si, row, 0)
    k.forR(cnt, proc() =
      k.e("uloadf32", dot, si, "@out1")
      k.e("sub", dot, dot, mx)
      k.e("exp", dot, dot)
      k.e("ustoref32", dot, si, "@out1")
      k.e("add", tot, tot, dot)
      k.e("uaddk", si, si, 1))
    k.e("uaddk", si, row, 0)
    k.forR(cnt, proc() =
      k.e("uloadf32", dot, si, "@out1")
      k.e("div", dot, dot, tot)
      k.e("ustoref32", dot, si, "@out1")
      k.e("uaddk", si, si, 1))
    k.e("loadconstf32", dot, 0)
    k.forR(rest, proc() =                            # entries above the diagonal
      k.e("ustoref32", dot, si, "@out1")
      k.e("uaddk", si, si, 1))
    k.e("uaddk", oi, qb, 0)
    k.e("uaddk", kb, baseI, 0)                       # V column d of row 0
    k.forN(D, proc() =
      k.e("loadconstf32", acc, 0)
      k.e("uaddk", si, row, 0)
      k.e("uaddk", ki, kb, 0)
      k.forR(cnt, proc() =
        k.e("uloadf32", dot, si, "@out1")
        k.e("uloadf32", kv, ki, "@in2")
        k.e("mul", dot, dot, kv)
        k.e("add", acc, acc, dot)
        k.e("uaddk", si, si, 1)
        k.e("uaddk", ki, ki, D))
      k.e("ustoref32", acc, oi, "@out0")
      k.e("uaddk", oi, oi, 1)
      k.e("uaddk", kb, kb, 1))
    k.e("uaddk", baseI, baseI, S * D)
    k.e("uaddk", baseS, baseS, S * S))
  k.done(S, 3, 2)

proc genAttentionFusedBackward*(B, H, S, D: int; scale: float): Gen =
  ## sessionAttentionFusedBackwardEnc(q, k, v, sMatrix, dy, dq, dk, dv, ...): accumulates
  ## (+=) into dq/dk/dv exactly like the shader. ports in0=q in1=k in2=v in3=sMatrix in4=dy
  ## -> out0=dq out1=dk out2=dv; lane = batch*head (rows run serially, so no atomics).
  let k = newKB()
  let baseI = k.u()
  let rowS = k.u()
  let qb = k.u()
  let wi = k.u()
  let vi = k.u()
  let dyb = k.u()
  let di = k.u()
  let qi = k.u()
  let cnt = k.r()
  let dotSum = k.r()
  let dS = k.r()
  let w = k.r()
  let x = k.r()
  let y = k.r()
  let sc = k.r()
  k.e("umulk", baseI, 0, S * D)
  k.e("umulk", rowS, 0, S * S)
  k.e("uaddk", qb, baseI, 0)
  k.e("loadconstf32", cnt, 1)

  proc dyDotV() =
    # dS = dy[ti] . v[tj] with dyb/di advanced past the row
    k.e("loadconstf32", dS, 0)
    k.e("uaddk", dyb, qb, 0)
    k.e("uaddk", di, vi, 0)
    k.forN(D, proc() =
      k.e("uloadf32", x, dyb, "@in4")
      k.e("uloadf32", y, di, "@in2")
      k.e("mul", y, y, x)
      k.e("add", dS, dS, y)
      k.e("uaddk", dyb, dyb, 1)
      k.e("uaddk", di, di, 1))

  k.forN(S, proc() =
    k.e("loadconstf32", dotSum, 0)
    k.e("uaddk", wi, rowS, 0)
    k.e("uaddk", vi, baseI, 0)
    k.forR(cnt, proc() =                              # pass 1: dv and the softmax dot
      k.e("uloadf32", w, wi, "@in3")
      dyDotV()
      k.e("mul", sc, dS, w)
      k.e("add", dotSum, dotSum, sc)
      k.e("uaddk", dyb, qb, 0)
      k.e("uaddk", di, vi, 0)
      k.forN(D, proc() =
        k.e("uloadf32", x, dyb, "@in4")
        k.e("mul", x, x, w)
        k.e("uloadf32", y, di, "@out2")
        k.e("add", y, y, x)
        k.e("ustoref32", y, di, "@out2")
        k.e("uaddk", dyb, dyb, 1)
        k.e("uaddk", di, di, 1))
      k.e("uaddk", wi, wi, 1)
      k.e("uaddk", vi, vi, D))
    k.e("uaddk", wi, rowS, 0)
    k.e("uaddk", vi, baseI, 0)
    k.forR(cnt, proc() =                              # pass 2: dq and dk
      k.e("uloadf32", w, wi, "@in3")
      dyDotV()
      k.e("sub", dS, dS, dotSum)
      k.e("mul", dS, dS, w)
      k.e("mulconst", dS, dS, fnum(scale))
      k.e("uaddk", qi, qb, 0)
      k.e("uaddk", di, vi, 0)
      k.forN(D, proc() =
        k.e("uloadf32", x, di, "@in1")
        k.e("mul", x, x, dS)
        k.e("uloadf32", y, qi, "@out0")
        k.e("add", y, y, x)
        k.e("ustoref32", y, qi, "@out0")
        k.e("uloadf32", x, qi, "@in0")
        k.e("mul", x, x, dS)
        k.e("uloadf32", y, di, "@out1")
        k.e("add", y, y, x)
        k.e("ustoref32", y, di, "@out1")
        k.e("uaddk", qi, qi, 1)
        k.e("uaddk", di, di, 1))
      k.e("uaddk", wi, wi, 1)
      k.e("uaddk", vi, vi, D))
    k.e("addconst", cnt, cnt, 1)
    k.e("uaddk", rowS, rowS, S)
    k.e("uaddk", qb, qb, D))
  k.done(B * H, 5, 3)

proc genFlashPrefill*(T, H, HKV, D, maxSeqLen: int; scale: float): Gen =
  ## sessionFlashAttentionPrefillEnc(q, kAll, vAll, o, T, H, HKV, D, pos, maxSeqLen, scale):
  ## o[t, h, :] = softmax(q[t,h,:] . K[hkv, 0..pos+t] * scale) . V; q/o are [T, H, D], the cache
  ## [HKV, maxSeqLen, D]. ports in0=q in1=K in2=V in3=pos (scalar) -> out0=o; lane = kv head,
  ## serving the H/HKV query heads that share it.
  if HKV <= 0 or H mod HKV != 0:
    raise newException(GenError, "flash prefill needs H divisible by HKV")
  let nRep = H div HKV
  let k = newKB()
  let zero = k.cu(0)
  let cnt = k.r()
  let qb = k.u()
  let kvb = k.u()
  let ks = k.u()
  let qi = k.u()
  let ki = k.u()
  let oi = k.u()
  let tq = k.u()
  let m = k.r()
  let l = k.r()
  let dot = k.r()
  let qv = k.r()
  let kv = k.r()
  let c = k.r()
  k.e("uloadf32", cnt, zero, "@in3")
  k.e("addconst", cnt, cnt, 1)                       # keys for row t = pos + t + 1
  k.e("umulk", kvb, 0, maxSeqLen * D)
  k.e("umulk", tq, 0, nRep * D)                      # (t = 0, h = lane * nRep) * D
  proc score() =
    k.e("loadconstf32", dot, 0)
    k.e("uaddk", ki, ks, 0)
    k.e("uaddk", qi, qb, 0)
    k.forN(D, proc() =
      k.e("uloadf32", qv, qi, "@in0")
      k.e("uloadf32", kv, ki, "@in1")
      k.e("mul", qv, qv, kv)
      k.e("add", dot, dot, qv)
      k.e("uaddk", qi, qi, 1)
      k.e("uaddk", ki, ki, 1))
    k.e("mulconst", dot, dot, fnum(scale))
  k.forN(T, proc() =
    k.e("uaddk", qb, tq, 0)
    k.forN(nRep, proc() =
      k.e("loadconstf32", m, fnum(-1e30))
      k.e("loadconstf32", l, 0)
      k.e("uaddk", ks, kvb, 0)
      k.forR(cnt, proc() =
        score()
        k.e("cmplt", c, m, dot)
        k.cond(c, proc() = k.cp(m, dot))
        k.e("uaddk", ks, ks, D))
      k.e("uaddk", oi, qb, 0)
      k.forN(D, proc() =
        k.e("loadconstf32", qv, 0)
        k.e("ustoref32", qv, oi, "@out0")
        k.e("uaddk", oi, oi, 1))
      k.e("uaddk", ks, kvb, 0)
      k.forR(cnt, proc() =
        score()
        k.e("sub", dot, dot, m)
        k.e("exp", dot, dot)
        k.e("add", l, l, dot)
        k.e("uaddk", ki, ks, 0)
        k.e("uaddk", oi, qb, 0)
        k.forN(D, proc() =
          k.e("uloadf32", kv, ki, "@in2")
          k.e("mul", kv, kv, dot)
          k.e("uloadf32", qv, oi, "@out0")
          k.e("add", qv, qv, kv)
          k.e("ustoref32", qv, oi, "@out0")
          k.e("uaddk", ki, ki, 1)
          k.e("uaddk", oi, oi, 1))
        k.e("uaddk", ks, ks, D))
      k.e("uaddk", oi, qb, 0)
      k.forN(D, proc() =
        k.e("uloadf32", qv, oi, "@out0")
        k.e("div", qv, qv, l)
        k.e("ustoref32", qv, oi, "@out0")
        k.e("uaddk", oi, oi, 1))
      k.e("uaddk", qb, qb, D))
    k.e("addconst", cnt, cnt, 1)
    k.e("uaddk", tq, tq, H * D))
  k.done(HKV, 4, 1)

proc genMamba2Ssd*(T, H, G, N, D, cont: int): Gen =
  ## sessionMamba2SsdPrefillEnc as a serial state-space scan (the chunked SSD pass computes
  ## the same recurrence): s[h,d,n] = exp(dt*A) s + dt * B[t,g,n] * x[t,h,d];
  ## y[t,h,d] = sum_n C[t,g,n] s[h,d,n] + dskip[h] * x[t,h,d], A = -exp(aLog[h]).
  ## ports in0=conv rows [T, cDim] = [x(C) | B(G*N) | C(G*N)] (post conv+SiLU)
  ## in1=dt [T, H] (post softplus) in2=aLog [H] in3=dSkip [H] in4=states [2, H*D, N]
  ## (slot 0 = initial state when cont, slot 1 = final state, written) -> out0=yPre [T, H*D];
  ## lane = group g, serving its H/G heads.
  if G <= 0 or H mod G != 0:
    raise newException(GenError, "ssd needs H divisible by G")
  let hpg = H div G
  let C = H * D
  let cDim = C + 2 * G * N
  let k = newKB()
  let zero = k.cu(0)
  let rowI = k.u()      # conv row base of token t
  let tH = k.u()        # t * H
  let tC = k.u()        # t * C
  let hh = k.u()        # head index (loop)
  let xb = k.u()
  let bb = k.u()
  let cb = k.u()
  let st = k.u()
  let st0 = k.u()
  let yo = k.u()
  let dtI = k.u()
  let ai = k.u()
  let bi = k.u()
  let ci = k.u()
  let si = k.u()
  let sj = k.u()
  let a = k.r()
  let dt = k.r()
  let decay = k.r()
  let dsk = k.r()
  let x = k.r()
  let bv = k.r()
  let s = k.r()
  let acc = k.r()
  # slot 1 <- slot 0 (carried state) or zero, for this group's heads
  k.e("umulk", hh, 0, hpg)
  k.e("umulk", st, hh, D * N)
  k.e("uaddk", si, st, C * N)
  k.e("uaddk", sj, st, 0)
  k.forN(hpg * D * N, proc() =
    if cont != 0:
      k.e("uloadf32", s, sj, "@in4")
    else:
      k.e("loadconstf32", s, 0)
    k.e("ustoref32", s, si, "@in4")
    k.e("uaddk", si, si, 1)
    k.e("uaddk", sj, sj, 1))
  k.e("uconst", rowI, 0)
  k.e("uconst", tH, 0)
  k.e("uconst", tC, 0)
  k.forN(T, proc() =
    k.e("umulk", hh, 0, hpg)                         # first head of the group
    k.e("uaddk", ai, hh, 0)
    k.e("uadd", dtI, hh, tH)
    k.e("umulk", st, hh, D * N)
    k.e("uaddk", st, st, C * N)
    k.e("umulk", xb, hh, D)
    k.e("uadd", xb, xb, rowI)
    k.e("umulk", bb, 0, N)
    k.e("uaddk", bb, bb, C)
    k.e("uadd", bb, bb, rowI)
    k.e("umulk", cb, 0, N)
    k.e("uaddk", cb, cb, C + G * N)
    k.e("uadd", cb, cb, rowI)
    k.e("umulk", yo, hh, D)
    k.e("uadd", yo, yo, tC)
    k.forN(hpg, proc() =
      k.e("uloadf32", a, ai, "@in2")
      k.e("exp", a, a)
      k.e("neg", a, a)
      k.e("uloadf32", dt, dtI, "@in1")
      k.e("mul", decay, dt, a)
      k.e("exp", decay, decay)
      k.e("uloadf32", dsk, ai, "@in3")
      k.forN(D, proc() =
        k.e("uloadf32", x, xb, "@in0")
        k.e("loadconstf32", acc, 0)
        k.e("uaddk", bi, bb, 0)
        k.e("uaddk", ci, cb, 0)
        k.forN(N, proc() =
          k.e("uloadf32", s, st, "@in4")
          k.e("mul", s, s, decay)
          k.e("uloadf32", bv, bi, "@in0")
          k.e("mul", bv, bv, dt)
          k.e("mul", bv, bv, x)
          k.e("add", s, s, bv)
          k.e("ustoref32", s, st, "@in4")
          k.e("uloadf32", bv, ci, "@in0")
          k.e("mul", bv, bv, s)
          k.e("add", acc, acc, bv)
          k.e("uaddk", st, st, 1)
          k.e("uaddk", bi, bi, 1)
          k.e("uaddk", ci, ci, 1))
        k.e("mul", x, x, dsk)
        k.e("add", acc, acc, x)
        k.e("ustoref32", acc, yo, "@out0")
        k.e("uaddk", yo, yo, 1)
        k.e("uaddk", xb, xb, 1))
      k.e("uaddk", ai, ai, 1)
      k.e("uaddk", dtI, dtI, 1))
    k.e("uaddk", rowI, rowI, cDim)
    k.e("uaddk", tH, tH, H)
    k.e("uaddk", tC, tC, C))
  discard zero
  k.done(G, 5, 1)
