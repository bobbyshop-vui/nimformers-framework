# Checkpoint and Binary Formats

The project currently has three distinct model-data formats. Use the format expected by the script/module you are running; `.nimq` and `.bin` are not interchangeable.

## `.nimq` quantized state dict

API in `quant.nim`:

```nim
proc saveQuantStateDict*(path: string, arch: array[5, int],
                         sd: seq[(string, QuantTensor)])
proc loadQuantStateDict*(path: string):
  tuple[arch: array[5, int], sd: seq[(string, QuantTensor)]]
```

The architecture tuple stores `[vocab, embedDim, nHeads, nLayers, ffMult]`. Each named tensor stores its quantization kind, shape, scales/zero-points, group size, CustomFloat metadata, and packed bytes. The high-level quantizers and `.nimq` exporter are documented in [Quantization and `.nimq`](quantization-nimq.md).

## Raw `.bin` state dict

API in `test/codestral/rawbin.nim`:

```nim
type RawBinTensor* = object
  name*: string
  shape*: seq[int]
  data*: seq[float32]

proc saveRawBin*(path: string, meta: Table[string, string],
                 tensors: seq[RawBinTensor])
proc loadRawBin*(path: string):
  tuple[meta: Table[string, string], tensors: seq[RawBinTensor]]

proc beginRawBin*(path: string, meta: Table[string, string],
                  tensorCount: int = 0): RawBinWriter
proc writeRawTensor*(writer: var RawBinWriter, tensor: RawBinTensor)
proc finishRawBin*(writer: var RawBinWriter)
proc openRawBinReader*(path: string): RawBinReader
proc readRawTensor*(reader: var RawBinReader): RawBinTensor
proc closeRawBinReader*(reader: var RawBinReader)
```

Raw `.bin` stores fp32 tensor payloads without quantization. Streaming writer/reader APIs avoid holding the entire state dict in memory. The format starts with `NIMBIN1\0`, followed by length-prefixed metadata and tensor names/shapes, then little-endian fp32 data.

## `.qlora.bin`

API in `test/codestral/qlorabin.nim`:

```nim
type
  QLoraTensorKind* = enum
    qlkRaw, qlkQ4Asym
  QLoraBinTensor* = object
    name*: string
    kind*: QLoraTensorKind
    shape*: seq[int]
    raw*: seq[float32]
    qt*: QuantTensor
  QLoraBinWriter* = object
  QLoraBinReader* = object

proc beginQLoraBin*(path: string, meta: Table[string, string],
                    tensorCount: int = 0): QLoraBinWriter
proc writeQLoraRawTensor*(writer: var QLoraBinWriter, name: string,
                          shape: seq[int], data: seq[float32])
proc writeQLoraQ4Tensor*(writer: var QLoraBinWriter, name: string,
                         qt: QuantTensor)
proc writeQLoraTensor*(writer: var QLoraBinWriter, t: QLoraBinTensor)
proc finishQLoraBin*(writer: var QLoraBinWriter)
proc saveQLoraBin*(path: string, meta: Table[string, string],
                   tensors: seq[QLoraBinTensor])
proc openQLoraBinReader*(path: string): QLoraBinReader
proc readQLoraTensor*(reader: var QLoraBinReader): QLoraBinTensor
proc closeQLoraBinReader*(reader: var QLoraBinReader)
proc isQLoraBin*(path: string): bool
```

Use `isQLoraBin` to identify this container. The format magic is `NIMQLR1\0`; tensors may be stored raw or in the supported Q4 representation.

## Training progress checkpoints

`test/codestral/checkpoint.nim` stores in-progress per-layer data beside a completed checkpoint as `<checkpoint>.inprogress/`. Public functions:

```nim
type LayerTensor* = object
  name*: string
  shape*: seq[int]
  data*: seq[float32]

proc progressDir*(ckptPath: string): string
proc clearLayerProgress*(ckptPath: string)
proc hasLayerProgress*(ckptPath: string, step: int): bool
proc commitLayerProgressCursor*(ckptPath: string, step, layer: int)
proc readProgressCursor*(ckptPath: string, stepOut, layerOut: var int): bool
proc saveLayerProgress*(ckptPath: string, step, layer: int,
                        tensors: seq[LayerTensor])
proc loadLayerProgress*(ckptPath: string, step, layer: int): seq[LayerTensor]
proc layerProgressReadable*(ckptPath: string, step, layer: int): bool
```

Layer progress files are temporary resume state. A committed final `.bin` remains the durable checkpoint; clear progress after a successful commit.
