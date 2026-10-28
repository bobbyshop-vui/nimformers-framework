## Config parsing regressions; importing inference does not load a model or Python.
import std/[json, os]
import nim_inference

proc main() =
  let folder = getTempDir() / ("nim-native-llama-config-" & $getCurrentProcessId())
  createDir(folder)
  defer: removeDir(folder)
  let path = folder/"config.json"
  var data = %*{"vocab_size":32256, "hidden_size":4096, "intermediate_size":11008,
    "num_hidden_layers":32, "num_attention_heads":32, "num_key_value_heads":8,
    "max_position_embeddings":16384, "rms_norm_eps":1e-6, "rope_theta":100000,
    "rope_scaling":{"type":"linear", "factor":4.0}}
  writeFile(path, $data)
  let cfg = newLlamaLConfig(folder)
  doAssert cfg.vocabSize == 32256 and cfg.hiddenSize == 4096
  doAssert cfg.nLayers == 32 and cfg.nHeads == 32 and cfg.nKVHeads == 8
  doAssert cfg.maxPositionEmbeddings == 16384 and cfg.rmsNormEps == 1e-6'f32
  doAssert cfg.ropeTheta == 100000'f32 and cfg.ropeScalingFactor == 4'f32
  doAssert cfg.ropeScalingType == "linear"
  data.delete("num_key_value_heads")
  data["rope_scaling"] = newJNull()
  writeFile(path, $data)
  let mha = newLlamaLConfig(path)
  doAssert mha.nKVHeads == mha.nHeads and mha.ropeScalingFactor == 1'f32
  data["rope_scaling"] = %*{"rope_type":"dynamic", "factor":2.0}
  writeFile(path, $data)
  doAssert newLlamaLConfig(folder).ropeScalingType == "dynamic"
  data.delete("num_attention_heads")
  writeFile(path, $data)
  var rejected = false
  try: discard newLlamaLConfig(folder)
  except ValueError: rejected = true
  doAssert rejected
  echo "PASS: native config, GQA/MHA, RoPE scaling and invalid metadata"
main()
