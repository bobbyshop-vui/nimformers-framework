# Nim forward hooks

[English](forward-hooks.md) | [Tiếng Việt](forward-hooks.vi.md)

`nimformers` provides before and after hooks for every high-level model forward
API. Register a hook for one API or use `ForwardApiAll` (`"*"`) to observe all
of them.

```nim
import nimformers

proc addInputOffset(ctx: Backend; input: Tensor): Tensor =
  result = input
  for value in result.data.mitems:
    value += 0.1'f32

proc scaleOutput(ctx: Backend; input, output: Tensor): Tensor =
  result = output
  for value in result.data.mitems:
    value *= 0.5'f32

proc main() =
  let hookId = registerForwardHook("linear", before = addInputOffset,
    after = scaleOutput)
  defer: discard removeForwardHook(hookId)
  echo forwardApiNames()

main()
```

`before` receives `(ctx, input)` and returns the input tensor used by the
forward pass. `after` receives `(ctx, input, output)` and returns the final
output tensor. Hooks run in registration order; nested model forwards also
invoke hooks registered for their component APIs. `forwardApiNames()` lists
the supported selectors: `linear`, `layer_norm`, `causal_self_attention`,
`feed_forward`, `transformer_block`, `embedding`, `nimformer_model_batch`, and
`nimformer_model`.

Use `registerForwardHook("*")` or `registerForwardHook(ForwardApiAll, ...)` for
all selectors. Hook registrations are process-wide. Set them up before
inference and avoid changing the registry while forwards are running. Existing
`Linear.preForwardHook` remains supported and runs after registered `linear`
before-hooks.

Callbacks use Nim's `nimcall` convention, so define them as ordinary procs
without captured closure state. Hooks do not provide matching backward
callbacks; output transformations that affect gradients need a corresponding
custom backward implementation before they are used for training.
