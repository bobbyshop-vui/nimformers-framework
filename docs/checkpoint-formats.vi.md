
# Định dạng checkpoint và binary

[English](checkpoint-formats.md) | [Tiếng Việt](checkpoint-formats.vi.md)

Project hiện dùng ba định dạng dữ liệu model khác nhau. Chọn đúng định dạng
mà script/module yêu cầu; `.nimq` và `.bin` không thể thay thế cho nhau.


## State dict lượng tử hóa `.nimq`

API trong `quant.nim`:

```nim
proc saveQuantStateDict*(path: string, arch: array[5, int],
                         sd: seq[(string, QuantTensor)])
proc loadQuantStateDict*(path: string):
  tuple[arch: array[5, int], sd: seq[(string, QuantTensor)]]
```

Tuple kiến trúc lưu `[vocab,embedDim,nHeads,nLayers,ffMult]`. Mỗi tensor có
tên lưu kiểu lượng tử hóa, shape, scale/zero-point, kích thước group,
metadata CustomFloat và bytes đóng gói. Quantizer cấp cao và exporter
`.nimq` được mô tả ở [lượng tử hóa và `.nimq`](quantization-nimq.vi.md).


## State dict `.bin` raw

API trong `test/codestral/rawbin.nim`:

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

`.bin` raw lưu payload fp32 không lượng tử hóa. API reader/writer stream
tránh giữ toàn state dict trong RAM. Magic mở đầu là `NIMBIN1\0`, tiếp theo
là metadata và tên/shape tensor có tiền tố độ dài, rồi dữ liệu fp32 little-endian.

## `.qlora.bin`

API trong `test/codestral/qlorabin.nim`:

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

Dùng `isQLoraBin` để nhận diện container. Magic là `NIMQLR1\0`; tensor có
thể lưu raw hoặc dạng Q4 được hỗ trợ.


## Checkpoint tiến độ training

`test/codestral/checkpoint.nim` lưu dữ liệu từng layer đang xử lý cạnh
checkpoint hoàn chỉnh tại `<checkpoint>.inprogress/`. Các hàm công khai:

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

File tiến độ layer là trạng thái resume tạm. File `.bin` cuối đã commit
mới là checkpoint bền vững; xóa tiến độ tạm sau commit thành công.
