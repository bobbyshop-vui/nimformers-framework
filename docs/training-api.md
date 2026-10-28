# Spectron Training API

The Spectron trainer is [`test/spectron/test_nimformer.nim`](../test/spectron/test_nimformer.nim). Run it from the repository root:

```sh
make run
```

It has no training command-line options. Edit the `CFG_*` constants near the top of the source to change model size, sequence length, batch size, step count, learning rate, or dataset sample limit. `make run-metal` selects the Metal backend on macOS; `make run-cpu` selects CPU.

## Datasets and model files

The trainer reads `test/spectron/databricks-dolly-15k.jsonl` and all six code datasets in `test/spectron/`: `nim_python_code_dataset_mega.jsonl` and `python_code_dataset_1.jsonl` through `python_code_dataset_5.jsonl`. It scans each code dataset and uses reservoir sampling to retain up to 4,000 valid examples per file, keeping memory bounded while allowing rows from throughout each file into training. The byte-level BPE training slice samples from each source.

The trained tokenizer and checkpoint are stored under `test/spectron/models/`:

- `test/spectron/models/spectron_bpe_tokenizer.json`
- `test/spectron/models/finetune.nimq`

Training resumes from the checkpoint if it exists. The checkpoint stores model weights and completed step count; Adam optimizer moments restart on resume.

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
proc newModelAdamState*(m: NimformerModelSpectron): ModelAdamState
proc trainStep*(m: var NimformerModelSpectron, adam: var ModelAdamState,
                idsBatch, targetsBatch: seq[seq[int]], ctx: Backend,
                lr: float32 = 3e-4'f32,
                requantizeEvery: int = 50): float32
proc saveCheckpoint*(m: NimformerModelSpectron, path: string, step: int = 0)
proc loadCheckpoint*(path: string): tuple[model: NimformerModelSpectron, step: int]
```

The source also exports the Linear++ attention and Spectron block forward/backward helpers for code that builds or trains individual layers.

## Codestral LoRA data

The Codestral trainer is [`test/codestral/finetune_lora.nim`](../test/codestral/finetune_lora.nim). It reads the Dolly dataset and `test/spectron/nim_python_code_dataset_mega.jsonl`; its paths are configured in `CFG_DATASETS` in that source. Its weights, adapter checkpoint, and tokenizer files live under `test/codestral/models/`.
