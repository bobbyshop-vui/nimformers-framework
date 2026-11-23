
# LoRA Fine-Tuning and Resume APIs (`test/codestral/finetune_lora.nim`)

[English](finetune-lora-api.md) | [Tiếng Việt](finetune-lora-api.vi.md)

This module contains the training utilities for `NimformerModelJamba` and `JambaLora`. The main training entry point is built/run as a program; import its exported types/functions only when writing a custom trainer. Core adapter operations are in [LoRA API](lora-api.md).

## RAM-only Codestral fine-tuning

RAM-only is the default for `make finetune-mamba`. It keeps activations, Adam moments, and full-precision trainable attention weights in memory. It does not write activation scratch, Adam sidecars, or per-layer resume snapshots. A portable LoRA `.bin` checkpoint is written after each completed training step (`CFG_CHECKPOINT_EVERY = 1`). The code calls each step an epoch in its logs; it is one streamed batch, not a full pass through the corpus.

```sh
make finetune-codestral-vulkan  # select Vulkan; cooperative matrices are used when supported
make finetune-codestral-metal   # select Metal's register-blocked SIMD GEMM

# Select the backend directly through make:
make finetune-mamba FINETUNE_BACKEND=auto FINETUNE_RAM_ONLY=1

# Keep the older disk-backed activation/Adam/layer-resume mode:
make finetune-mamba FINETUNE_RAM_ONLY=0
```

Equivalent runtime controls are `BYBY_FINETUNE_RAM_ONLY=1|0`, `BYBY_FINETUNE_BACKEND=auto|vulkan|metal`, and `BYBY_FINETUNE_ACCELERATED_GEMM=1|0`. The make targets enable accelerated GEMM by default. On Vulkan, the NT GEMM uses cooperative FP16 matrix inputs with FP32 accumulation when the selected device and shape support them; otherwise BybyLang uses its exact FP32 route. `BYBY_VULKAN_TENSOR_CORES=0` disables the cooperative route. On Metal, the NT GEMM uses the register-blocked SIMD kernel; `BYBY_METAL_GEMM_RB=0` disables that kernel. Metal does not use the Vulkan Tensor Core path.

RAM-only mode resumes from the latest completed `.bin` weights if the program is restarted, but it starts Adam from zero and ignores existing layer-resume state. An interrupted step is discarded; the last completed checkpoint remains usable. It can require substantially more system memory because optimizer moments and trainable full-precision weights remain resident. The `.bin` checkpoint and tokenizer reference sidecar are still written to disk.


## Streaming batches

```nim
type
  DatasetSchema* = enum
    dsDolly, dsTopicCode
  DatasetSpec* = object
    path*: string
    schema*: DatasetSchema
  StreamingBatcher* = object

const CFG_DATASETS*: seq[DatasetSpec]
const TargetIgnore* = -1

proc newStreamingBatcher*(specs: seq[DatasetSpec], seqLen, batchSize: int,
                          tok: BPETokenizer, hfTok: HfTokenizer,
                          usingHfTok: bool): StreamingBatcher
proc nextBatch*(b: var StreamingBatcher):
  tuple[ids, targets, targetsMasked: seq[seq[int]]]
proc skipSteps*(b: var StreamingBatcher, steps: int)
```

`DatasetSchema` selects the expected JSONL fields. `TargetIgnore` marks labels excluded from masked loss. `nextBatch` streams/tokenizes examples into a reusable token buffer rather than loading the entire corpus into RAM.


## Checkpoint and resume

```nim
proc saveCheckpointLoraBin*(m: var NimformerModelJamba, lora: JambaLora,
    path: string, epoch: int, newBlockParamNames: seq[string],
    attnStore: AttnWeightDiskStore)
proc loadCheckpointLoraBin*(path: string):
    tuple[model: NimformerModelJamba, lora: JambaLora, epoch: int]
proc loadCheckpointLora*(path: string):
    tuple[model: NimformerModelJamba, lora: JambaLora, epoch: int]

proc writeResumeCommit*(checkpointPath: string, step, seqLen, batchSize: int,
                        pattern, datasetSig, tokenizerDir: string)
proc readResumeCommit*(checkpointPath: string,
    stepOut, seqLenOut, batchOut: var int,
    patternOut, datasetSigOut, tokenizerDirOut: var string): bool
proc markResumeInProgress*(ckptPath: string, nextStep: int)
proc clearResumeInProgress*(ckptPath: string)
proc restoreLayerProgressForStep*(lo: var JambaLora, ckptPath: string,
                                  step, nBlocks: int,
                                  updateMaxBlock = high(int)): int
```

LoRA checkpoints use `.bin` or the supported QLoRA 4-bit variant. Metadata records model pattern, quantization path, step/epoch, and normalization settings; incompatible normalization metadata is rejected on resume. Layer-by-layer transient progress is documented in [Checkpoint and Binary Formats](checkpoint-formats.md).


## Optimizer state stores

```nim
type AdamDiskStore* = object
  root*: string
  ramOnly*: bool

proc initAdamDiskStore*(root: string, ramOnly = false): AdamDiskStore
proc has*(store: AdamDiskStore, name: string, paramLen: int): bool
proc init*(store: AdamDiskStore, name: string, paramLen: int)
proc load*(store: AdamDiskStore, name: string, paramLen: int): ApfAdamState
proc save*(store: AdamDiskStore, name: string, st: ApfAdamState)
proc cleanup*(store: var AdamDiskStore)
proc clearAdamState*(st: var ApfAdamState)
```

`AdamDiskStore` can use either disk or RAM. Disk mode stores Adam moments (`m`, `v`, `step`) in files so training does not keep every optimizer tensor resident. RAM-only mode keeps them in memory and never creates the store directory. Both modes validate parameter lengths before loading state.


## Training step and integrity helpers

`trainStepGraftStreaming` trains new/grafted blocks while storing activations and full-precision parameters either in RAM or on disk. Additional exported resume/integrity helpers include `saveLoraAdamStates`, `restoreLoraAdamStates`, `resetAdamParameters`, `adamFileStep`, `adamStoreAllAtStep`, `adamStoreAllAtStepOrNext`, `syncLoraAdamFromStore`, `forceLoraAdamToStep`, `forceAdamStoreToStep`, `loraAdamStateAllAtStep`, `datasetSignature`, `attnStoreAllValid`, `applyLayerProgressToLora`, `layerProgressStepState`, `fillUntrainedLoraBlocks`, `auditStep0Core`, `auditStep0LoraBlocks`, and `checkLoraCheckpointIntegrity`.

These helpers support the training program's recovery protocol and are lower-level than the normal `trainStepLora`/checkpoint workflow. Their implementation and version-specific parameters are in `test/codestral/finetune_lora.nim`.
