## Regression for the Iris 640 failure: reject the full B=4,T=256 Mamba2
## training cache before allocation/upload, while allowing B=1,T=128.
import std/strutils
import backend
import mamba2_full_vm
import mamba2_ssm_gpu

let metal = Backend(kind: bkMetal)
let tooLarge = m2Layout(4, 256, 128, 64, 128, 8, 4, 8192, 1e-5'f32, 8)
doAssert tooLarge.total == 1_198_125_824
var rejected = false
try:
  checkVmScratchCapacity(metal, tooLarge.total, "regression B4T256")
except CatchableError as e:
  rejected = "4570." in e.msg and "768 MiB" in e.msg
doAssert rejected, "4.46 GiB Mamba2 cache must fail with the Metal memory diagnostic"

let safe = m2Layout(1, 128, 128, 64, 128, 8, 4, 8192, 1e-5'f32, 8)
doAssert safe.total == 150_394_624
checkVmScratchCapacity(metal, safe.total, "regression B1T128")
echo "PASS - Mamba2 scratch budget rejects 4.46 GiB and accepts 574 MiB"
