## Build one decoder with the builder API, record one step, lift it and check it.
## The scratch directory is outside the repository.
import std/[os, json]
import ./firlift
import ./corelift
import ./builder
import gpubackend as byby

proc main() =
  let outDir = "/tmp/fir-builder"
  createDir(outDir)
  let cfgText = """{"hidden_size":2048,"num_hidden_layers":1,"num_attention_heads":32,"num_key_value_heads":4,"intermediate_size":5632,"max_position_embeddings":2048,"vocab_size":64,"rms_norm_eps":1e-5,"torch_dtype":"float32"}"""
  writeFile(outDir / "config.json", cfgText)
  let cfg = BuilderConfig(hidden: 2048, layers: 1, heads: 32, kvHeads: 4,
    mlp: 5632, vocab: 64, positions: 2048, eps: 1e-5'f32)
  beginBuilderTrace(outDir / "builder.fprobe")
  var m = newBuilderModel(byby.gbCpu, cfg)
  traceModel(m, 1, 0)
  let trace = parseTrace(readFile(outDir / "builder.fprobe"))
  let model = liftCore(trace, cfgText, "builder", "")
  writeFile(outDir / "model.fir", model)
  echo "builder trace: ops=", trace.ops.len, " names=", trace.names.len
  report(model)

main()
