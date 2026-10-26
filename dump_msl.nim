import std/[strutils]
import mamba2_vm_kernels, jit_vm, session_vm

let pSSM = buildMamba2StepProgram(128, 64, 128, 8)
echo "=== SSM STEP (lines 1..50) ==="
let srcSSM = lowerBytecodeToMetal(pSSM, "test_ssm")
let linesSSM = srcSSM.splitLines
for i in 0 ..< min(50, linesSSM.len):
  echo linesSSM[i]

echo "\n=== RMSL ==="
let pRMS = buildRmsGateLargeProgram(8192, 1e-5, 1024)
let srcRMS = lowerBytecodeToMetal(pRMS, "test_rms")
let linesRMS = srcRMS.splitLines
for i in 0 ..< min(50, linesRMS.len):
  echo linesRMS[i]
echo "..."
for i in max(0, linesRMS.len - 30) ..< linesRMS.len:
  echo linesRMS[i]

echo "\n=== Fused Split Conv ==="
let pConv = buildFusedSplitConvSoftplusProgram(8192, 10240, 128, 4)
let srcConv = lowerBytecodeToMetal(pConv, "test_conv")
let linesConv = srcConv.splitLines
for i in 0 ..< min(40, linesConv.len):
  echo linesConv[i]
