## Routing regression: CUDA is no longer a framework execution backend.
## This test needs no GPU because explicit backend selection is lazy.

import backend
import vendor/bybylang/gpubackend as byby

let selected = newBackend("cuda")
doAssert selected.kind == bkVulkan,
  "newBackend(\"cuda\") must redirect to Vulkan"
doAssert toByby(bkCuda) == byby.gbVulkan,
  "legacy bkCuda enum must never map to the CUDA driver"
doAssert byby.parseBackend("cuda") == byby.gbVulkan,
  "BybyLang string parsing must redirect CUDA to Vulkan"
echo "PASS - all CUDA backend selection routes resolve to Vulkan"
