import nimformers_shared

var ctx = newContext("cpu")
var group = newProcessGroup(["cpu"])
try:
  doAssert abiVersion() == 2
  echo "backend: ", backendName(ctx)
  echo add(ctx, [1'f32, 2, 3], [10'f32, 20, 30])
  echo "partition: ", partition(group, 12)
finally:
  close(group)
  close(ctx)
