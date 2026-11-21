## Smoke test for an importable Nim hook plugin.

import nimformers
import token_hook_plugin

proc main() =
  let hookId = install()
  defer: doAssert uninstall(hookId)

  # The plugin hook changes 3 -> 4 before the plugin body, then 8 -> 18 after it.
  doAssert pluginScale(3'f32) == 18'f32

  # Also exercise the same plugin against a low-level BybyLang session API.
  var buffer = sessionUpload(gbCpu, @[1'f32, 2'f32])
  defer: sessionFree(gbCpu, buffer)
  let values = sessionRead(gbCpu, buffer, 1)
  doAssert values == @[42'f32, 2'f32]

  echo "Nim hook plugin passed: custom API + BybyLang sessionRead"

main()
