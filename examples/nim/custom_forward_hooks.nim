import nimformers

var beforeCalls = 0
var afterCalls = 0

proc addInputOffset(api: string; ctx: Backend; inputs: var seq[Tensor]) =
  inc beforeCalls
  for input in inputs.mitems:
    for value in input.data.mitems:
      value += 0.1'f32

proc scaleOutput(api: string; ctx: Backend; inputs: openArray[Tensor];
    output: Tensor): Tensor =
  inc afterCalls
  result = output
  for value in result.data.mitems:
    value *= 0.5'f32

proc main() =
  let ctx = newBackend("cpu")
  defer: closeBackend(ctx)
  var linear = newLinear(4, 2)
  let input = newTensor(@[1, 4], 1'f32)

  let linearHookId = registerApiHook("linear", before = addInputOffset,
    after = scaleOutput)
  let addHookId = registerApiHook("addT", before = addInputOffset,
    after = scaleOutput)
  defer: discard removeApiHook(linearHookId)
  defer: discard removeApiHook(addHookId)

  let output = linear.forward(input, ctx)
  let residual = addT(ctx, input, input)
  doAssert beforeCalls == 2
  doAssert afterCalls == 2
  echo "APIs with hook support: ", hookApiNames()
  echo "linear output shape: ", output.shape
  echo "tensor add output shape: ", residual.shape

main()
