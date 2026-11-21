import nimformer
import backend
import customfloat
import quant
import tensor_ops

# Define a custom forward hook
proc customForwardHook(ctx: Backend, input: Tensor): Tensor =
  ## Example hook: add a bias to the input
  let bias = newTensor(@[input.shape[^1]], 0.1'f32)
  return addT(ctx, input, bias)

# Create a Linear layer with the custom hook
var linear = newLinear(128, 128)
linear.preForwardHook = customForwardHook

# Create a test input tensor
let input = randnTensor(@[1, 128], 1.0'f32)

# Forward pass through the layer
let output = linear.forward(input, newBackend())

# Verify the output
echo "Output shape: ", $output.shape
for i in 0 ..< output.data.len:
  echo output.data[i]
