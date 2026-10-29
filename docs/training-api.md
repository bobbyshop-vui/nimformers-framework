
# Spectron Training API

[English](training-api.md) | [Tiếng Việt](training-api.vi.md)

The Spectron trainer is [`test/spectron/test_nimformer.nim`](../test/spectron/test_nimformer.nim).
See the [detailed Spectron walkthrough](spectron-guide.md) for build versus run,
configuration, JSONL schema, checkpoints, loss, and a standalone Nim example.
The [Make command guide](make-commands.md) covers every target. Run training
from the repository root:

```sh
make run
```

It has no training command-line options. Edit the `CFG_*` constants near the
top of the source to change model size, sequence length, batch size, step
count, learning rate, or dataset sample limit. `make run-metal` selects Metal
on macOS. The legacy `run-cpu` target fails at runtime because CPU model
execution is no longer supported. `make build` only builds the trainer;
`make run-spectron` builds and runs the separate interactive inference program.


## Datasets and model files

The trainer reads `test/spectron/databricks-dolly-15k.jsonl` and all six code datasets in `test/spectron/`: `nim_python_code_dataset_mega.jsonl` and `python_code_dataset_1.jsonl` through `python_code_dataset_5.jsonl`. It scans each code dataset and uses reservoir sampling to retain up to 4,000 valid examples per file, keeping memory bounded while allowing rows from throughout each file into training. The byte-level BPE training slice samples from each source. The Dolly path is read as raw text, including JSONL syntax. `make download-datasets` downloads the six code files, not Dolly. Missing sources are skipped; if every source is empty/missing, the trainer uses a repeated demo string.

The trained tokenizer and checkpoint are stored under `test/spectron/models/`:

- `test/spectron/models/spectron_bpe_tokenizer.json`
- `test/spectron/models/finetune.nimq`

Training resumes from the checkpoint if it exists. The checkpoint stores model weights and completed step count; Adam optimizer moments restart on resume.

`CFG_N_STEPS` is the number of additional steps per launch. New models have a
final LayerNorm before the output projection. Old checkpoints without its
tensors preserve their original inference behavior on load; the training main
explicitly enables final LayerNorm before continuing training. The next save
includes its parameters. Logs report current-batch `loss` and the mean over
batches since the previous log; no validation loss is computed by this demo.


## Byte-level BPE

```nim
proc getOrTrainBPE*(corpusText: string, numMerges: int): BPETokenizer
proc encodeBPE*(tok: BPETokenizer, s: string): seq[int]
proc decodeBPEStr*(tok: BPETokenizer, ids: seq[int]): string
```

This tokenizer is trained by the Spectron program and saved at `test/spectron/models/spectron_bpe_tokenizer.json`. It is separate from the Hugging Face tokenizer used by DeepSeek and Codestral.


## Model, training, and checkpoints

```nim
proc newNimformerModelSpectron*(vocab, embedDim, nHeads, nLayers, ffMult: int,
                                 useSession: bool = true): NimformerModelSpectron
proc forward*(m: NimformerModelSpectron, ids: seq[int], ctx: Backend): Tensor
proc forwardBatch*(m: NimformerModelSpectron, idsBatch: seq[seq[int]],
                   ctx: Backend): Tensor
proc softmaxCrossEntropy*(logits: Tensor, targets: seq[seq[int]], ctx: Backend):
    tuple[loss: float32, dLogits: Tensor]
proc newModelAdamState*(m: NimformerModelSpectron): ModelAdamState
proc trainStep*(m: var NimformerModelSpectron, adam: var ModelAdamState,
                idsBatch, targetsBatch: seq[seq[int]], ctx: Backend,
                lr: float32 = 3e-4'f32,
                requantizeEvery: int = 50): float32
proc saveCheckpoint*(m: NimformerModelSpectron, path: string, step: int = 0)
proc loadCheckpoint*(path: string): tuple[model: NimformerModelSpectron, step: int]
```

The source also exports the Linear++ attention and Spectron block forward/backward helpers for code that builds or trains individual layers.

`forward` returns `[T,V]`; `forwardBatch` returns `[B,T,V]`. Input and target
batches have matching `[B][T]` dimensions, with target tokens shifted one
position forward. `trainStep` returns mean token cross-entropy before updating
the weights. `ModelAdamState` must be kept across steps within the process.


## Codestral LoRA data

The Codestral trainer is [`test/codestral/finetune_lora.nim`](../test/codestral/finetune_lora.nim). It reads the Dolly dataset and `test/spectron/nim_python_code_dataset_mega.jsonl`; its paths are configured in `CFG_DATASETS` in that source. Its weights, adapter checkpoint, and tokenizer files live under `test/codestral/models/`.
