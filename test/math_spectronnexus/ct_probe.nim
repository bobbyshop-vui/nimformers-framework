include sequence_explorer

# 1. The digit-pair walk against the int64 walk modulo p^2 < 2^62, where both apply.
for (p, poly) in [(1_000_003'i64, @[1'i64, 2, 3, 2, 1]), (1_000_003'i64, @[1'i64, -1, 2, 1, 1]),
                  (65_537'i64, @[2'i64, 1, 0, -2, 1]), (999_983'i64, @[1'i64, 1, 1, 2])]:
  for alpha in [p - 1, p div 2 + 17, (3 * p) div 4 + 5]:
    let total = int64(poly.len - 1) * alpha
    var wanted: seq[int64]
    for t in [p - 3, p - 1, p, p + 1, p + 2, min(total div 2, 2 * p - 2), total div 2 - 1]:
      if t >= 0 and t <= total div 2 + 1 and t < 2 * p: wanted.add(t)
    let narrow = powerProductCoefficients(poly, alpha, @[1'i64], 0, wanted, p)
    let wide = powerProductCoefficientsWide(poly, alpha, @[1'i64], 0, wanted, p)
    var bad = 0
    for t in wanted:
      if narrow[t] != wide[t]: inc bad
    echo "p=", p, " P=", poly, " alpha=", alpha, " coefficients=", wanted.len, " mismatches=", bad

# 2. The five-term walk past t = p above 2^31 against the square route, P = (1/x + 1 + x)^2.
proc fiveTermWalk(laurent: seq[int64], left: int64, start: Laurent, n: BigInt, q: int64): int64 =
  var r = start
  var rest = n
  while not rest.isZero:
    let (quotient, j) = divSmall(rest, q)
    r = cartierProduct(r, laurent, j, @[1'i64], 0, -left * j, q)
    rest = quotient
  constantOf(r)

var weights = newSeq[Rational](9)
for i in 0 ..< 9: weights[i] = (0'i64, 1'i64)
weights[4] = (1'i64, 1'i64)
weights[3] = (-1'i64, 1'i64)
let square = ConstantTermForm(ok: true, shift: 0, left: 2, laurent: @[1'i64, 2, 3, 2, 1], weights: weights)
for (p, n) in [(2_147_483_659'i64, big(2_147_483_658'i64)), (2_147_483_659'i64, big(1_000_000_000_000_000_000'i64)),
               (4_294_967_311'i64, big(4_000_000_000'i64))]:
  let start: Laurent = (-4'i64, weights.mapIt(ratMod(it, p)))
  let t0 = epochTime()
  let five = fiveTermWalk(square.laurent, 2, start, n, p)
  let t1 = epochTime()
  let three = constantTermMod(square, n, p)
  echo "Riordan(2n) n=", n, " mod ", p, ": five-term walk ", five, " (", t1 - t0, " s), square route ", three,
    " (", epochTime() - t1, " s), equal=", five == three
