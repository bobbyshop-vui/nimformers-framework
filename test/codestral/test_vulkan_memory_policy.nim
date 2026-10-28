## Host-side regression for resident-weight policy; no Vulkan device required.
import vendor/bybylang/backends/vulkan/vulkan_caps

for mode in ["auto", "1", "0", "invalid"]:
  for nonDiscrete in [0'u32, 1'u32, 3'u32, 4'u32]:
    doAssert not useDeviceLocalWeights(nonDiscrete, mode)
doAssert useDeviceLocalWeights(2, "auto")
doAssert useDeviceLocalWeights(2, "1")
doAssert not useDeviceLocalWeights(2, "0")
doAssert not useDeviceLocalWeights(2, "invalid")
doAssert deviceClass(1) == "iGPU" and deviceClass(2) == "dGPU"
echo "PASS: only dGPU uses private/device-local weights, even with DEVICE_LOCAL=1"
