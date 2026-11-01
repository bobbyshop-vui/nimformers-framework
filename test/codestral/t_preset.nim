## t_preset.nim - checks the norm-selection LOGIC of the infer path (runs on the CPU, no GPU needed).
##   nim c -r t_preset.nim
## Checks: setGpuNormPreset (aliases, the "-unit" suffix), the global flags read by encodeLayerV2 /
## encodeLinearPlusBranch / encodeHeadGpu, and layerUsesLayerNorm (the single source of truth
## for both the R branch AND the P sidecar).
import std/strutils
import lora
import mamba2_fused_step

var failed = 0
proc check(name: string; cond: bool) =
  if cond: echo "[PASS] ", name
  else:
    echo "[FAIL] ", name
    inc failed

# ── current trainer preset: RMSNorm with base gamma + norm_f ──
check("rms-fin-g1 is valid", setGpuNormPreset("rms-fin-g1"))
check("rms-fin-g1: R+sidecar RMSNorm", gRNormOverride == 0 and not layerUsesLayerNorm(4))
check("rms-fin-g1: normF on, 1 group, base gamma", gUseNormF and gRNormGroupsOverride == 1 and not gRmsUnitGamma)
check("rms-fin-g1: name", gNormPresetName == "rms-fin-g1")

# ── alias ──
check("hf = rms-fin-g1", setGpuNormPreset("HF") and gNormPresetName == "rms-fin-g1" and gRNormOverride == 0)
check("train = ln-nof-g1 (LayerNorm, no normF)",
      setGpuNormPreset("train") and gNormPresetName == "ln-nof-g1" and gRNormOverride == 1 and
      not gUseNormF and layerUsesLayerNorm(4) and not gRmsUnitGamma)
check("mamba-ssm = rms-fin-gn", setGpuNormPreset("mamba-ssm") and gNormPresetName == "rms-fin-gn" and
      gRNormGroupsOverride == -1)

# ── the -unit suffix: gamma = 1 (checkpoints trained with the old trainer) ──
check("rms-fin-g1-unit is valid", setGpuNormPreset("rms-fin-g1-unit"))
check("rms-fin-g1-unit: RMSNorm + gamma=1 + normF", gRNormOverride == 0 and gRmsUnitGamma and gUseNormF and
      gNormPresetName == "rms-fin-g1-unit")
check("hf-unit = rms-fin-g1-unit", setGpuNormPreset("hf-unit") and gNormPresetName == "rms-fin-g1-unit" and gRmsUnitGamma)
check("-unit only with RMSNorm: ln-nof-g1-unit is rejected", not setGpuNormPreset("ln-nof-g1-unit"))
check("train-unit is rejected", not setGpuNormPreset("train-unit"))

# ── changing the preset must RESET the unit flag (sweeps switch modes repeatedly) ──
discard setGpuNormPreset("rms-fin-g1-unit")
discard setGpuNormPreset("hf")
check("hf after unit: gamma back to base", not gRmsUnitGamma)
discard setGpuNormPreset("rms-fin-g1-unit")
discard setGpuNormPreset("ln-nof-g1")
check("ln after unit: gamma back to base", not gRmsUnitGamma and layerUsesLayerNorm(0))

# ── invalid presets ──
check("invalid preset is rejected", not setGpuNormPreset("foo") and not setGpuNormPreset("rms-fin") and
      not setGpuNormPreset("rms-fin-g1-foo"))

if failed == 0: echo "ALL TESTS PASS"
else:
  echo failed, " TEST FAIL"
  quit(1)
