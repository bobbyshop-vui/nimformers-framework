import std/[strformat, math, tables]
import backend
import nimformer
import jamba_hybrid
import mamba2_real
import mamba2_ssm_gpu
import lora
import quant
import fp4_gemv_gpu
import load_codestral_into_jamba

when isMainModule:
  let ctx = newBackend("metal")
  let be = ctx.kind.toByby()
  echo "Backend initialized: ", ctx.kind

  # Load model structure
  var m = newNimformerModelJamba(32768, 4096, 128, ffMult = 4, pattern = "R",
                                useSession = true, mambaRealDState = 128,
                                mambaRealNGroups = 8, mambaRealConvKernel = 4,
                                mambaRealExpand = 2, quantResident = true)
  echo "Model created, loading quant-resident from codestral_mamba_7b_fp4.nimq..."
  let quantWeights = loadCodestralQuantResidentFromNimq(m, "codestral_mamba_7b_fp4.nimq")
  echo "Loaded quant resident: ", quantWeights.len, " tensors."

  # Inspect layer 0 norm weight vs ln1
  echo "m.blocks[0].ln1.gamma first 5: ", m.blocks[0].ln1.gamma.data[0 ..< 5]
  echo "m.normF.gamma first 5: ", m.normF.gamma.data[0 ..< 5]
