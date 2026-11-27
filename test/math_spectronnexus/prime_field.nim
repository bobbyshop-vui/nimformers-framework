## Exact arithmetic for the bounded prime fields used by the native tools.
import std/json

const ExactMaxPrime* = 65_521

func isPrime*(value: int): bool =
  if value < 2: return false
  if value mod 2 == 0: return value == 2
  var divisor = 3
  while divisor <= value div divisor:
    if value mod divisor == 0: return false
    divisor += 2
  true

proc integer*(node: JsonNode, label: string): int64 =
  if node.isNil or node.kind != JInt:
    raise newException(ValueError, "Expected integer " & label)
  node.getBiggestInt()

proc readPrime*(objectNode: JsonNode, bound = ExactMaxPrime): int =
  let value = integer(objectNode{"prime"}, "prime")
  if value < 2 or value > int64(bound) or not isPrime(int(value)):
    raise newException(ValueError, "prime must be prime and at most " & $bound)
  int(value)

func residue*(value: int64, p: int): int {.inline.} =
  let remainder = int(value mod int64(p))
  if remainder < 0: remainder + p else: remainder

func productMod*(a, b, p: int): int {.inline.} =
  int((int64(a) * int64(b)) mod int64(p))

func subtractProduct*(a, b, c, p: int): int {.inline.} =
  residue(int64(a) - int64(b) * int64(c), p)

func reciprocal*(value, p: int): int =
  doAssert value != 0
  var exponent = p - 2
  var base = value
  result = 1
  while exponent > 0:
    if (exponent and 1) != 0: result = productMod(result, base, p)
    base = productMod(base, base, p)
    exponent = exponent shr 1
