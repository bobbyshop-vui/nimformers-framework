
# LoRA Fine-Tuning and Resume APIs (`test/codestral/finetune_lora.nim`)

[English](finetune-lora-api.md) | [Tiếng Việt](finetune-lora-api.vi.md)

This module contains the disk-backed training/resume utilities for `NimformerModelJamba` and `JambaLora`. The main training entry point is built/run as a program; import its exported types/functions only when writing a custom trainer. Core adapter operations are in [LoRA API](lora-api.md).


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


## Disk-backed optimizer state

```nim
type AdamDiskStore* = object
  root*: string

proc initAdamDiskStore*(root: string): AdamDiskStore
proc has*(store: AdamDiskStore, name: string, paramLen: int): bool
proc init*(store: AdamDiskStore, name: string, paramLen: int)
proc load*(store: AdamDiskStore, name: string, paramLen: int): ApfAdamState
proc save*(store: AdamDiskStore, name: string, st: ApfAdamState)
proc cleanup*(store: var AdamDiskStore)
proc clearAdamState*(st: var ApfAdamState)
```

Adam moments (`m`, `v`, `step`) live on disk so training large models does not require keeping every optimizer tensor in RAM. The store validates payload length before accepting a snapshot.


## Training step and integrity helpers

`trainStepGraftStreaming` trains new/grafted blocks while streaming activation and Adam state to disk. Additional exported resume/integrity helpers include `saveLoraAdamStates`, `restoreLoraAdamStates`, `resetAdamParameters`, `adamFileStep`, `adamStoreAllAtStep`, `adamStoreAllAtStepOrNext`, `syncLoraAdamFromStore`, `forceLoraAdamToStep`, `forceAdamStoreToStep`, `loraAdamStateAllAtStep`, `datasetSignature`, `attnStoreAllValid`, `applyLayerProgressToLora`, `layerProgressStepState`, `fillUntrainedLoraBlocks`, `auditStep0Core`, `auditStep0LoraBlocks`, and `checkLoraCheckpointIntegrity`.

These helpers support the training program's recovery protocol and are lower-level than the normal `trainStepLora`/checkpoint workflow. Their implementation and version-specific parameters are in `test/codestral/finetune_lora.nim`.
