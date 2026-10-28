# LoRA API

The LoRA implementation is in `test/codestral/lora.nim` and targets `NimformerModelJamba`. Training and command-line inference are implemented in `test/codestral/finetune_lora.nim` and `test/codestral/infer_lora.nim`.

## Adapter types and construction

```nim
type
  LoraAdapter* = object
    r*, inF*, outF*: int
    alpha*, scale*: float32
    A*, B*: Tensor
  LoraTrainState* = object
    lr*, wd*: float32
    adamA*, adamB*: Table[string, ApfAdamState]
  JambaLora* = object
    adapters*: Table[string, LoraAdapter]
    backup*: Table[string, seq[float32]]
    quantBase*: Table[string, QuantTensor]

proc newLoraAdapter*(inF, outF, r: int, alpha: float32 = -1'f32): LoraAdapter
proc newLoraTrainState*(lora: JambaLora, lr: float32 = 1e-4'f32,
                        wd: float32 = 0.01'f32): LoraTrainState
```

`A` has shape `[r, inF]`, `B` has shape `[outF, r]`, and the merged update is `(alpha / r) * B @ A`. A negative `alpha` selects the implementation default (`2*r`). The base parameter is restored after each LoRA training step.

## Attach, train, merge, and restore

```nim
proc attachLora*(m: var NimformerModelJamba, targetNames: seq[string],
                 r: int = 4, alpha: float32 = -1'f32): JambaLora
proc attachLoraQuantResident*(m: var NimformerModelJamba,
                              quantWeights: Table[string, QuantTensor],
                              targetNames: seq[string], r: int = 4,
                              alpha: float32 = -1'f32): JambaLora
proc trainStepLora*(m: var NimformerModelJamba, lora: var JambaLora,
                    state: var LoraTrainState,
                    idsBatch, targetsBatch: seq[seq[int]], ctx: Backend,
                    requantizeEvery: int = 0): float32

proc mergeIn*(m: var NimformerModelJamba, lora: var JambaLora,
              ctx: Backend, name: string)
proc restoreBase*(m: var NimformerModelJamba, lora: JambaLora)
proc mergeLoraIntoQuantBase*(lo: var JambaLora, verbose = true): int
proc mergeQuantAdapters*(other: JambaLora, into: var JambaLora)
proc dropZeroLoraAdapters*(lo: var JambaLora): seq[string]
```

`attachLora` selects named projection weights. Call `mergeIn` before base-model forward/backward and restore the frozen base with `restoreBase` after the step. `trainStepLora` performs this lifecycle and updates adapter A/B only. Quant-resident adapters use `attachLoraQuantResident` to avoid keeping another full precision copy of the quantized base in host RAM.

Other parameter and merge helpers:

```nim
proc mergedWeight*(base: Tensor, ad: LoraAdapter, ctx: Backend): Tensor
proc gradToAB*(dWeff: Tensor, ad: LoraAdapter, ctx: Backend): tuple[dA, dB: Tensor]
proc allProjectionWeightNames*(m: var NimformerModelJamba): seq[string]
proc bindQuantBaseToLinears*(m: var NimformerModelJamba, lora: var JambaLora)
proc mergeInUploadStreamingJamba*(m: var NimformerModelJamba,
                                  lora: var JambaLora, ctx: Backend)
proc graftBlocks*(old: var NimformerModelJamba, newPattern: string,
                  ctx: Backend): NimformerModelJamba
```

`graftBlocks` adds new `A` (attention) or `S` (Spectron) blocks into a block pattern while copying the existing `M` blocks in order.

## GPU inference helpers

```nim
proc prepareGpuModel*(m: var NimformerModelJamba,
                      lora: var JambaLora, ctx: Backend)
proc forwardStepJambaGpu*(m: var NimformerModelJamba, tokenId: int,
                          ctx: Backend, produceLogits: bool = true): Tensor
proc forwardStepJambaGpuRetry*(m: var NimformerModelJamba, tokenId: int,
                               replayTokens: openArray[int], replayCount: int,
                               ctx: Backend,
                               produceLogits: bool = true): Tensor
proc resetGpuGenState*(m: var NimformerModelJamba, ctx: Backend)
proc warmupGpuForParallelPrefill*(m: var NimformerModelJamba, ctx: Backend)
proc generate*(m: var NimformerModelJamba, lora: var JambaLora,
               tok: AnyTokenizer, prompt: string, maxNewTokens: int,
               ctx: Backend, temperature: float32 = 0'f32,
               seed: int64 = 1234): string
```

Decode profiling helpers are `decodeProfileSummary()` and `decodeSplitSummary()`. `initMicroVmJit` initializes the bytecode/JIT cache and `warmupMicroVmJit` warms it up. `setGpuDisableLora`, `setGpuNormPreset`, `normPresetDescription`, `lpSelfCheck`, and `graftBlocks` support advanced model preparation/debugging; see their source doc comments for constraints.

## Inference CLI

`test/codestral/infer_lora.nim` is the executable entry point for `.bin`/LoRA checkpoints. It exposes public helpers `loadBinaryLoraForInfer`, `defaultSamplerCfg`, `forwardPrefillBatch`, and `generate`; command-line options are printed by `./infer_lora_bin --help` after building that target. The CLI and its environment variables can change more frequently than the library API.
