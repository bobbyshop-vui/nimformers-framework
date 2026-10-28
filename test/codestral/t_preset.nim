## t_preset.nim - kiem tra LOGIC chon norm cua duong infer (chay duoc tren CPU, khong can GPU).
##   nim c -r t_preset.nim
## Kiem: setGpuNormPreset (alias, hau to "-unit"), cac co toan cuc ma encodeLayerV2 /
## encodeLinearPlusBranch / encodeHeadGpu doc, va layerUsesLayerNorm (nguon su that duy nhat
## cho nhanh R LAN sidecar P).
import std/strutils
import lora
import mamba2_fused_step

var failed = 0
proc check(name: string; cond: bool) =
  if cond: echo "[PASS] ", name
  else:
    echo "[FAIL] ", name
    inc failed

# ── preset hien tai cua trainer: RMSNorm gamma base + norm_f ──
check("rms-fin-g1 hop le", setGpuNormPreset("rms-fin-g1"))
check("rms-fin-g1: R+sidecar RMSNorm", gRNormOverride == 0 and not layerUsesLayerNorm(4))
check("rms-fin-g1: normF bat, 1 nhom, gamma base", gUseNormF and gRNormGroupsOverride == 1 and not gRmsUnitGamma)
check("rms-fin-g1: ten", gNormPresetName == "rms-fin-g1")

# ── alias ──
check("hf = rms-fin-g1", setGpuNormPreset("HF") and gNormPresetName == "rms-fin-g1" and gRNormOverride == 0)
check("train = ln-nof-g1 (LayerNorm, khong normF)",
      setGpuNormPreset("train") and gNormPresetName == "ln-nof-g1" and gRNormOverride == 1 and
      not gUseNormF and layerUsesLayerNorm(4) and not gRmsUnitGamma)
check("mamba-ssm = rms-fin-gn", setGpuNormPreset("mamba-ssm") and gNormPresetName == "rms-fin-gn" and
      gRNormGroupsOverride == -1)

# ── hau to -unit: gamma = 1 (checkpoint train bang trainer cu) ──
check("rms-fin-g1-unit hop le", setGpuNormPreset("rms-fin-g1-unit"))
check("rms-fin-g1-unit: RMSNorm + gamma=1 + normF", gRNormOverride == 0 and gRmsUnitGamma and gUseNormF and
      gNormPresetName == "rms-fin-g1-unit")
check("hf-unit = rms-fin-g1-unit", setGpuNormPreset("hf-unit") and gNormPresetName == "rms-fin-g1-unit" and gRmsUnitGamma)
check("-unit chi voi RMSNorm: ln-nof-g1-unit bi tu choi", not setGpuNormPreset("ln-nof-g1-unit"))
check("train-unit bi tu choi", not setGpuNormPreset("train-unit"))

# ── doi preset phai RESET co unit (sweep chuyen che do lien tuc) ──
discard setGpuNormPreset("rms-fin-g1-unit")
discard setGpuNormPreset("hf")
check("hf sau unit: gamma tro ve base", not gRmsUnitGamma)
discard setGpuNormPreset("rms-fin-g1-unit")
discard setGpuNormPreset("ln-nof-g1")
check("ln sau unit: gamma tro ve base", not gRmsUnitGamma and layerUsesLayerNorm(0))

# ── preset sai ──
check("preset sai bi tu choi", not setGpuNormPreset("foo") and not setGpuNormPreset("rms-fin") and
      not setGpuNormPreset("rms-fin-g1-foo"))

if failed == 0: echo "TAT CA TEST PASS"
else:
  echo failed, " TEST FAIL"
  quit(1)
