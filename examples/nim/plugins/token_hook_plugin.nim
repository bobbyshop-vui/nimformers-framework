## Example extension module: its own Nim API and a low-level BybyLang hook.
## Import this module in the host process, then call install() before using
## either the plugin API or the hooked framework API.

import nimformers

proc pluginScale*(value: float32): float32 {.apiHooked.} =
  ## A plugin-owned runtime API that participates in Nimformers' hook registry.
  value * 2'f32

proc rewritePluginCalls(api: string; phase: BybyApiHookPhase;
                        call: var BybyApiCall) =
  case api
  of "pluginScale":
    case phase
    of bybyApiBefore:
      let value = apiArg[float32](call.args[0])
      apiArg[float32](call.args[0]) = value + 1'f32
    of bybyApiAfter:
      let value = apiArg[float32](call.output)
      apiArg[float32](call.output) = value + 10'f32
  of "sessionRead":
    case phase
    of bybyApiBefore:
      # sessionRead(backend, handle, count): ask the runtime for two values.
      apiArg[int](call.args[2]) = 2
    of bybyApiAfter:
      apiArg[seq[float32]](call.output)[0] = 42'f32
  else:
    discard

proc install*(): BybyApiValueHookId =
  ## Install the plugin callback. The wildcard lets one plugin handle APIs
  ## owned by both this module and the framework.
  registerBybyApiValueHook("*", rewritePluginCalls)

proc uninstall*(hookId: BybyApiValueHookId): bool =
  removeBybyApiValueHook(hookId)
