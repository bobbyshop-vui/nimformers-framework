import nimformers

var beforeCalls = 0
var afterCalls = 0

proc addInputOffset(ctx: Backend; input: Tensor): Tensor =
  inc beforeCalls
  result = input
  for value in result.data.mitems:
    value += 0.1'f32

proc scaleOutput(ctx: Backend; input, output: Tensor): Tensor =
  inc afterCalls
  result = output
  for value in result.data.mitems:
    value *= 0.5'f32

proc main() =
  let hookId = registerForwardHook(ForwardApiAll, before = addInputOffset,
    after = scaleOutput)
  defer: discard removeForwardHook(hookId)

  let ctx = newBackend("cpu")
  defer: closeBackend(ctx)
  var layer = newLinear(4, 2)
  let output = layer.forward(newTensor(@[1, 4], 1'f32), ctx)
  doAssert beforeCalls == 1
  doAssert afterCalls == 1
  echo "APIs with hook support: ", forwardApiNames()
  echo "linear output shape: ", output.shape

main()
