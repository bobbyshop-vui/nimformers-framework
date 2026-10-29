import std/math
import customfloat

# Narrow ranges may still need many exponent bits with the fixed bias.
# These occur in Adam-trained bias and LayerNorm tensors at APF boundaries.
for values in [@[0.001'f32, -0.002'f32], @[0.99'f32, 0.999'f32],
               @[1024'f32, -2048'f32], @[1e-30'f32, -1e30'f32],
               @[0'f32, -0'f32], @[1'f32, 1'f32]]:
  let cf = buildCustomDtypeForTensor(values)
  let decoded = decodeArray(encodeArray(values, cf), cf)
  for i, value in values:
    doAssert decoded[i].classify notin {fcNan, fcInf, fcNegInf}
    if value == 0:
      doAssert decoded[i] == 0
    else:
      doAssert abs((decoded[i] - value) / value) <= 1e-3'f32,
        "APF lost a finite value: " & $value & " -> " & $decoded[i] & " (" & cf.name & ")"
echo "PASS: APF exponent range preserves small/large and near-unit tensors"
