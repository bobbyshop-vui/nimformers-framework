# Nim API hooks

[English](forward-hooks.md) | [Tiếng Việt](forward-hooks.vi.md)

Nimformers hooks span four surfaces: Tensor/model values, typed arguments and
results for instrumented Nim APIs, before/after observation across framework
and BybyLang calls, and replacement of generated micro-kernel bytecode.

## Change Tensor and model calls

`registerApiHook` installs process-wide hooks for model forwards and
Tensor-valued operations, including nested calls:

```nim
import nimformers

proc addInputOffset(api: string; ctx: Backend; inputs: var seq[Tensor]) =
  for input in inputs.mitems:
    for value in input.data.mitems:
      value += 0.1'f32

proc scaleOutput(api: string; ctx: Backend; inputs: openArray[Tensor];
    output: Tensor): Tensor =
  result = output
  for value in result.data.mitems:
    value *= 0.5'f32

let hookId = registerApiHook("addT", before = addInputOffset,
  after = scaleOutput)
defer: discard removeApiHook(hookId)
```

The `before` callback receives the API name, backend, and every Tensor operand.
It can replace or modify those operands in place, while keeping the operand
count unchanged. The `after` callback receives the inputs and returns the
output Tensor used by the caller. These callbacks use Nim's closure calling
convention and can capture state. Hook selectors are case-insensitive.

Use `registerApiHook("*")` to hook all names returned by `hookApiNames()`.
Nested calls each produce their own callback. `registerForwardHook` remains
available for the original `(ctx, input)` / `(ctx, input, output)` callbacks;
its wildcard selects only module forwards. `Linear.preForwardHook` is also
retained.

Hooks run in registration order and are process-wide. Register them before
parallel inference and do not mutate the registry while calls are running.
Hooks do not provide backward-gradient callbacks; use a custom backward
implementation when a forward hook changes values that affect training.

## Observe framework and BybyLang APIs

`registerBybyApiHook` observes public APIs across the framework, including
model/Tensor calls, backend operations, BybyLang sessions, micro-VM bytecode
builders, JIT/lowering, and distributed helpers. Pass a procedure name or
`"*"` to receive before/after events for all instrumented APIs:

```nim
import nimformers

proc inspectApi(api: string; phase: BybyApiHookPhase; metadata: string) =
  echo phase, " ", api, " ", metadata

proc main() =
  let hookId = registerBybyApiHook("*", inspectApi)
  defer: discard removeBybyApiHook(hookId)

main()
```

Metadata summarizes scalar arguments and collection lengths. After events also
include output summaries and changed `var` arguments; large tensors and device
buffers are not copied into the callback. This hook observes calls. Use
`registerBybyApiValueHook` to change typed BybyLang arguments/results:

```nim
import nimformers

proc rewriteRead(api: string; phase: BybyApiHookPhase;
    call: var BybyApiCall) =
  case phase
  of bybyApiBefore:
    if api == "sessionRead":
      apiArg[int](call.args[2]) = 2
  of bybyApiAfter:
    if api == "sessionRead":
      apiArg[seq[float32]](call.output)[0] = 42'f32

proc main() =
  let hookId = registerBybyApiValueHook("sessionRead", rewriteRead)
  defer: discard removeBybyApiValueHook(hookId)
  var buffer = sessionUpload(gbCpu, @[1'f32, 2'f32])
  defer: sessionFree(gbCpu, buffer)
  let values = sessionRead(gbCpu, buffer, 1)
  doAssert values == @[42'f32, 2'f32]

main()
```

Use `apiArg[T]` with the exact Nim type shown by `BybyApiArg.typeName` to read
or mutate an argument in the before phase and the return value in the after
phase. `call.output` is available in the after phase. A selector of `"*"`
applies one callback to every instrumented public Nim API; check `api` and
`typeName` before calling `apiArg[T]` because the arguments differ across
APIs. A `var` argument is copied back to its caller after the hook. Hooks run
in registration order and should be installed before concurrent inference.
Nim APIs returning `var T` remain observable, but their returned alias cannot
be replaced by a typed value hook without breaking Nim's aliasing contract.
`registerApiHook` remains the Tensor-specific path; the micro-kernel hook below
replaces generated compute.

## Hook a procedure in your own Nim module

Import `nimformers` and add `{.apiHooked.}` to each runtime procedure that
should participate in the same registry. This lets a library expose hookable
APIs without writing its own callback plumbing:

```nim
import nimformers

proc scaleValue*(value: float32): float32 {.apiHooked.} =
  value * 2'f32
```

`registerBybyApiValueHook("*", callback)` then receives this procedure along
with the framework APIs. Compile-time macros, templates, and raw imported C
declarations do not have a Nim runtime body to wrap; annotate a Nim wrapper
around an imported function when it needs hooks.

### Import a hook plugin

A plugin can be an ordinary Nim module compiled into the host application. It
can expose its own `{.apiHooked.}` procedures and install callbacks for those
procedures or framework APIs. The example plugin also changes a low-level
BybyLang `sessionRead` call:

```sh
make hook-plugin-test
```

See [`token_hook_plugin.nim`](../examples/nim/plugins/token_hook_plugin.nim)
and [`test_hook_plugin.nim`](../test/hooks/test_hook_plugin.nim). The test
checks a plugin-owned API's input and result hooks, then changes the requested
read length and returned data from `sessionRead`. This is an in-process Nim
module plugin; it is compiled with the host and does not use a separate
`dlopen`/`LoadLibrary` plugin ABI.

The plugin exports `install()` and `uninstall(id)`. Installation registers a
wildcard typed hook, and its callback filters by API name before accessing
typed arguments. For `pluginScale(3)`, it changes the input to `4`, lets the
plugin procedure double it to `8`, then changes the result to `18`. For
`sessionRead`, it changes the requested length from one to two and then changes
the first returned value. Import the module in the host and install it before
calling either API:

```nim
import nimformers
import token_hook_plugin

let pluginHook = install()
defer: discard uninstall(pluginHook)
doAssert pluginScale(3'f32) == 18'f32
```

The hook registry is process-wide. Remove the returned hook ID when the plugin
is unloaded or no longer needed; do not add/remove hooks while other threads
are executing hooked calls.

## Replace a generated BybyLang micro-kernel

BybyLang micro-kernels can be rewritten before CPU execution or GPU lowering,
including a program already built and about to enter JIT or dispatch. Select by
kernel tag; DSL kernels use the `bybylang_<name>` tag. Assign another
`VmProgram` in the callback to replace its bytecode:

```nim
import nimformers

var replacementProgram: VmProgram # Build this with the session_vm emit API.

proc useReplacement(tag: string; program: var VmProgram) =
  if tag == "bybylang_scale":
    program = replacementProgram

proc main() =
  let hookId = registerMicroKernelHook("bybylang_scale", useReplacement)
  defer: discard removeMicroKernelHook(hookId)
  # Run code that calls the generated "bybylang_scale" kernel here.

main()
```

The replacement must preserve the original buffer layout, output size, launch
dimensions, and backend-supported instruction set. Hooks run before JIT
compilation and dispatch, so the replacement bytecode is what gets compiled
and executed. Micro-kernel callbacks use Nim's closure calling convention.
The selector "*" replaces every tagged micro-kernel; use a specific tag when
the replacement is valid for only one kernel. A VmProgram already rewritten
for a tag is not rewritten again as it passes through lowering and dispatch.
