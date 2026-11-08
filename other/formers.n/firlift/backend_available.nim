## Read-only device availability check; never dispatch a kernel or load a model.
import std/os
import backends/metal/metal_backend
import backends/vulkan/vulkan_driver

let name = if paramCount() > 0: paramStr(1) else: ""
let available = case name
  of "metal": metalAvailable()
  of "vulkan": vulkanAvailable()
  else: false
if not available:
  echo "SKIP session backend ", name, ": device unavailable"
  quit(1)
