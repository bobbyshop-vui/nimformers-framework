## quantize_lora_bin_4bit.nim
## Converts a LoRA .bin checkpoint (raw fp32, rawbin) into a 4-bit QLoRA version
## to reduce disk/RAM size (~8GB fp32 -> ~1–1.5GB depending on the number of Attention blocks).
##
## Conventions:
##   - LoRA adapters A/B: KEPT in fp32 (low rank, small, need precision for training/merging).
##   - newBlock::* (fully trained Attention block weights): quantized to 4 bits
##     (default qkInt4Asymmetric per-group 128, close to Q4_K_M).
##   - Meta + a new format magic "NIMBIN4\0" so the loader recognises it and dequantizes when needed.
##   - CAN OVERWRITE: pass the same source/destination path (writes a .tmp file and renames safely).
##
## Usage:
##   nim c -d:release quantize_lora_bin_4bit.nim
##   ./quantize_lora_bin_4bit jamba_lora.bin              # OVERWRITES the same file
##   ./quantize_lora_bin_4bit jamba_lora.bin jamba_lora.bin
##   ./quantize_lora_bin_4bit jamba_lora.bin out_qlora4.bin  # to another file
##
## Once the 4-bit file exists:
##   - finetune_lora / infer_lora have a load path for the NIMBIN4 magic
##     (CHANGED: newBlock stays compressed as a QuantTensor, merged into lora.quantBase
##     together with the base .nimq - dequant on the fly + release through mergeIn()/
##     restoreBase() in lora.nim, NO longer fully dequantized to fp32 at load
##     time; adapters A/B stay fp32 as before, loaded with loadQLora4Bin instead of
##     loadQLora4BinDequant - the old function still exists but nobody calls it).
##   - For even lighter training/inference: keep the base Mamba quant-resident through a .nimq
##     like the existing codestral pipeline (already available).

import std/[os, strformat, strutils, tables, math, streams]

proc cRename(oldPath, newPath: cstring): cint {.importc: "rename", header: "<stdio.h>".}
import rawbin
import quant
import customfloat

const
  MAGIC_FP32* = "NIMBIN1\0"
  MAGIC_QLORA4* = "NIMBIN4\0"
  DefaultGroupSize = 128

# ── Raw QuantTensor read/write on a stream (compatible with the quant.nim layout) ─────

proc writeI32(s: Stream, v: int32) =
  var x = v
  s.writeData(addr x, 4)

proc writeI64(s: Stream, v: int64) =
  var x = v
  s.writeData(addr x, 8)

proc writeStrLP(s: Stream, str: string) =
  writeI32(s, int32(str.len))
  if str.len > 0:
    s.writeData(unsafeAddr str[0], str.len)

proc writeQuantTensorRaw(s: Stream, qt: QuantTensor) =
  var kindByte = uint8(qt.kind)
  s.writeData(addr kindByte, 1)
  writeI32(s, int32(qt.shape.len))
  for dim in qt.shape: writeI32(s, int32(dim))
  writeI32(s, int32(qt.scale.len))
  if qt.scale.len > 0:
    s.writeData(unsafeAddr qt.scale[0], qt.scale.len * 4)
  # zero_point has as many elements as scale (quant.nim layout)
  var zp = qt.zero_point
  if zp.len != qt.scale.len:
    zp = newSeq[float32](qt.scale.len)
  if zp.len > 0:
    s.writeData(unsafeAddr zp[0], zp.len * 4)
  writeI32(s, int32(qt.groupSize))
  writeI32(s, int32(max(qt.cf.exponentBits, 1)))
  writeI32(s, int32(qt.cf.mantissaBits))
  writeI32(s, int32(qt.bytes.len))
  if qt.bytes.len > 0:
    s.writeData(unsafeAddr qt.bytes[0], qt.bytes.len)

proc quantizeNewBlock*(data: seq[float32], shape: seq[int],
                      kind: QuantKind, groupSize: int): QuantTensor =
  ## Picks the best quantization path available in quant.nim.
  case kind
  of qkInt4Asymmetric:
    # per-tensor asymmetric first; if 2D and groupSize>0 the per-group path is used
    # (the current quantizeInt4Asymmetric is per-tensor; per-group lives in the
    # dequantize path / export_to_nimq). For simplicity + stability: quantize
    # per row when 2D (one scale per row) — much better than per-tensor.
    if shape.len == 2 and shape[0] > 0 and shape[1] > 0:
      # Layout matches dequantizeInt4Asymmetric (per-row): each row is padded to
      # bytesPerRow = ceil(cols/2), scale/zp per row.
      let rows = shape[0]
      let cols = shape[1]
      let bytesPerRow = (cols + 1) div 2
      var scales = newSeq[float32](rows)
      var zps = newSeq[float32](rows)
      var packed = newSeq[uint8](rows * bytesPerRow)
      for r in 0 ..< rows:
        var minV = float32.high
        var maxV = -float32.high
        let off = r * cols
        for c in 0 ..< cols:
          let v = data[off + c]
          if v < minV: minV = v
          if v > maxV: maxV = v
        let rng = maxV - minV
        let sc = if rng < 1e-12'f32: 1'f32 else: rng / 15'f32
        let zp = if rng < 1e-12'f32: 0'f32 else: -minV / sc
        scales[r] = sc
        zps[r] = zp
        let rowByteOff = r * bytesPerRow
        for c in 0 ..< cols:
          let q = clamp(int(round(data[off + c] / sc + zp)), 0, 15)
          let nibble = uint8(q and 0x0F)
          let byteIdx = rowByteOff + c div 2
          if c mod 2 == 0:
            packed[byteIdx] = (packed[byteIdx] and 0xF0'u8) or nibble
          else:
            packed[byteIdx] = (packed[byteIdx] and 0x0F'u8) or (nibble shl 4)
      result.kind = qkInt4Asymmetric
      result.shape = shape
      result.scale = scales
      result.zero_point = zps
      result.groupSize = 0   # per-row (scale.len == rows)
      result.data = packed
    else:
      result = quantizeInt4Asymmetric(data, shape)
  of qkInt4:
    result = quantizeInt4(data, shape)
  of qkFp4KM:
    # cols must be divisible by groupSize
    if shape.len == 2 and shape[1] mod groupSize == 0:
      result = quantizeFp4KM(data, shape, groupSize = groupSize)
    else:
      echo "[qlora4] qkFp4KM requires cols divisible by groupSize; falling back to int4 asymmetric"
      result = quantizeNewBlock(data, shape, qkInt4Asymmetric, groupSize)
  else:
    result = quantizeInt4Asymmetric(data, shape)

proc saveQLora4Bin*(path: string, meta: Table[string, string],
                    fp32Tensors: seq[RawBinTensor],
                    quantTensors: seq[(string, QuantTensor)]) =
  ## NIMBIN4 magic:
  ##   meta (like NIMBIN1) + nFp32 + [fp32 tensors] + nQuant + [quant tensors]
  ##
  ## IMPORTANT FOR TRAINING/RESUME: writes to .tmp and renames after closing.
  ## Never write straight into the target checkpoint: if the process is killed/OOMs
  ## while saving, the old checkpoint must stay intact for the next resume.
  let tmpPath = path & ".tmp"
  var f = newFileStream(tmpPath, fmWrite)
  if f == nil:
    raise newException(IOError, "Cannot write: " & tmpPath)
  try:
    f.write(MAGIC_QLORA4)
    writeI32(f, int32(meta.len))
    for k, v in meta.pairs:
      writeStrLP(f, k)
      writeStrLP(f, v)
    writeI32(f, int32(fp32Tensors.len))
    for t in fp32Tensors:
      writeStrLP(f, t.name)
      writeI32(f, int32(t.shape.len))
      for s in t.shape: writeI32(f, int32(s))
      writeI64(f, int64(t.floatValues.len))
      if t.floatValues.len > 0:
        f.writeData(unsafeAddr t.floatValues[0], t.floatValues.len * 4)
    writeI32(f, int32(quantTensors.len))
    for (name, qt) in quantTensors:
      writeStrLP(f, name)
      writeQuantTensorRaw(f, qt)
  finally:
    f.close()
  # POSIX/macOS: rename is atomic at the pathname level -> readers see the old checkpoint
  # or the complete new one, never a half-written file.
  if cRename(tmpPath.cstring, path.cstring) != 0:
    try: removeFile(tmpPath)
    except OSError: discard
    raise newException(IOError, "Cannot commit the checkpoint atomically: " & tmpPath & " -> " & path)

proc isQLora4Bin*(path: string): bool =
  var f = newFileStream(path, fmRead)
  if f == nil: return false
  defer: f.close()
  var magic = newString(8)
  if f.readData(addr magic[0], 8) != 8: return false
  result = magic == MAGIC_QLORA4

proc loadQLora4BinDequant*(path: string): tuple[meta: Table[string, string],
                                                 tensors: seq[RawBinTensor]] =
  ## Loads NIMBIN4 and dequantizes every newBlock to fp32 (the adapters are already fp32).
  ## The result is compatible with the old loadRawBin / loadCheckpointLoraBin.
  var f = newFileStream(path, fmRead)
  if f == nil:
    raise newException(IOError, "Cannot read: " & path)
  defer: f.close()
  var magic = newString(8)
  if f.readData(addr magic[0], 8) != 8 or magic != MAGIC_QLORA4:
    raise newException(IOError, "Not NIMBIN4: " & path)

  proc readI32Local(): int32 =
    var x: int32 = 0
    if f.readData(addr x, 4) != 4:
      raise newException(IOError, "EOF int32")
    result = x
  proc readI64Local(): int64 =
    var x: int64 = 0
    if f.readData(addr x, 8) != 8:
      raise newException(IOError, "EOF int64")
    result = x
  proc readStrLocal(): string =
    let n = readI32Local().int
    result = newString(n)
    if n > 0 and f.readData(addr result[0], n) != n:
      raise newException(IOError, "EOF string")

  result.meta = initTable[string, string]()
  let nMeta = readI32Local().int
  for i in 0 ..< nMeta:
    let k = readStrLocal()
    let v = readStrLocal()
    result.meta[k] = v

  let nFp32 = readI32Local().int
  for i in 0 ..< nFp32:
    var t: RawBinTensor
    t.name = readStrLocal()
    let nd = readI32Local().int
    t.shape = newSeq[int](nd)
    for d in 0 ..< nd: t.shape[d] = readI32Local().int
    let ne = readI64Local().int
    t.data = newSeq[float32](ne)
    if ne > 0 and f.readData(addr t.data[0], ne * 4) != ne * 4:
      raise newException(IOError, "EOF fp32 data: " & t.name)
    result.tensors.add(t)

  let nQuant = readI32Local().int
  for i in 0 ..< nQuant:
    let name = readStrLocal()
    # Reads a QuantTensor (same layout as quant.nim writeQuantTensor)
    var qt: QuantTensor
    var kindByte: uint8
    if f.readData(addr kindByte, 1) != 1:
      raise newException(IOError, "EOF kind")
    qt.kind = QuantKind(kindByte)
    let ndims = readI32Local().int
    qt.shape = newSeq[int](ndims)
    for d in 0 ..< ndims: qt.shape[d] = readI32Local().int
    let nScales = readI32Local().int
    qt.scale = newSeq[float32](nScales)
    if nScales > 0 and f.readData(addr qt.scale[0], nScales * 4) != nScales * 4:
      raise newException(IOError, "EOF scale")
    qt.zero_point = newSeq[float32](nScales)
    if nScales > 0 and f.readData(addr qt.zero_point[0], nScales * 4) != nScales * 4:
      raise newException(IOError, "EOF zero_point")
    qt.groupSize = readI32Local().int
    let eb = readI32Local().int
    let mb = readI32Local().int
    qt.cf = newCustomFloat(max(eb, 1), mb)
    let dlen = readI32Local().int
    qt.data = newSeq[uint8](dlen)
    if dlen > 0 and f.readData(addr qt.data[0], dlen) != dlen:
      raise newException(IOError, "EOF quant data: " & name)
    let deq = dequantizeTensor(qt)
    result.tensors.add(RawBinTensor(name: name, shape: qt.shape, data: deq))

proc loadQLora4Bin*(path: string): tuple[meta: Table[string, string],
                                          fp32Tensors: seq[RawBinTensor],
                                          quantWeights: Table[string, QuantTensor]] =
  ## CHANGED (RAM bug): loadQLora4BinDequant (above) dequantizes ALL of newBlock
  ## to fp32 RIGHT AT LOAD and keeps it resident for the WHOLE process lifetime - exactly
  ## the problem quant-resident (quantBase/mergeIn/restoreBase in
  ## lora.nim) solved for the base Mamba, only wrongly applied to
  ## newBlock. This function: reads the disk once (same file format),
  ## but does NOT call dequantizeTensor - returns newBlock still as QuantTensors
  ## (compressed) in a separate `quantWeights` table, keyed by the REAL parameter NAME
  ## (the "newBlock::" prefix removed) to match paramList(m)/byName directly.
  ## The caller (loadBinaryLoraForInfer/loadCheckpointLoraBin) only needs to merge this table
  ## into lora.quantBase (the same table as the base .nimq quantResident) -
  ## then mergeIn()/restoreBase() (or forwardStepJambaWithLora for an all-'R'
  ## pattern) DEQUANTIZE ON THE FLY + RELEASE exactly like the base, no
  ## separate code needed. newBlock lives in RAM as fp32 only WHILE
  ## really running forward (mergeIn...restoreBase), no longer from load
  ## until the program exits.
  var f = newFileStream(path, fmRead)
  if f == nil:
    raise newException(IOError, "Cannot read: " & path)
  defer: f.close()
  var magic = newString(8)
  if f.readData(addr magic[0], 8) != 8 or magic != MAGIC_QLORA4:
    raise newException(IOError, "Not NIMBIN4: " & path)

  proc readI32Local(): int32 =
    var x: int32 = 0
    if f.readData(addr x, 4) != 4:
      raise newException(IOError, "EOF int32")
    result = x
  proc readI64Local(): int64 =
    var x: int64 = 0
    if f.readData(addr x, 8) != 8:
      raise newException(IOError, "EOF int64")
    result = x
  proc readStrLocal(): string =
    let n = readI32Local().int
    result = newString(n)
    if n > 0 and f.readData(addr result[0], n) != n:
      raise newException(IOError, "EOF string")

  result.meta = initTable[string, string]()
  let nMeta = readI32Local().int
  for i in 0 ..< nMeta:
    let k = readStrLocal()
    let v = readStrLocal()
    result.meta[k] = v

  let nFp32 = readI32Local().int
  for i in 0 ..< nFp32:
    var t: RawBinTensor
    t.name = readStrLocal()
    let nd = readI32Local().int
    t.shape = newSeq[int](nd)
    for d in 0 ..< nd: t.shape[d] = readI32Local().int
    let ne = readI64Local().int
    t.data = newSeq[float32](ne)
    if ne > 0 and f.readData(addr t.data[0], ne * 4) != ne * 4:
      raise newException(IOError, "EOF fp32 data: " & t.name)
    result.fp32Tensors.add(t)

  result.quantWeights = initTable[string, QuantTensor]()
  let nQuant = readI32Local().int
  for i in 0 ..< nQuant:
    let name = readStrLocal()
    if name in result.quantWeights:
      raise newException(IOError, "NIMBIN4 has a duplicate quant tensor name: " & name)
    # Reads a QuantTensor (same layout as quant.nim writeQuantTensor) - KEPT
    # compressed, NO dequantizeTensor here (unlike loadQLora4BinDequant).
    var qt: QuantTensor
    var kindByte: uint8
    if f.readData(addr kindByte, 1) != 1:
      raise newException(IOError, "EOF kind")
    qt.kind = QuantKind(kindByte)
    let ndims = readI32Local().int
    qt.shape = newSeq[int](ndims)
    for d in 0 ..< ndims: qt.shape[d] = readI32Local().int
    let nScales = readI32Local().int
    qt.scale = newSeq[float32](nScales)
    if nScales > 0 and f.readData(addr qt.scale[0], nScales * 4) != nScales * 4:
      raise newException(IOError, "EOF scale")
    qt.zero_point = newSeq[float32](nScales)
    if nScales > 0 and f.readData(addr qt.zero_point[0], nScales * 4) != nScales * 4:
      raise newException(IOError, "EOF zero_point")
    qt.groupSize = readI32Local().int
    let eb = readI32Local().int
    let mb = readI32Local().int
    qt.cf = newCustomFloat(max(eb, 1), mb)
    let dlen = readI32Local().int
    qt.data = newSeq[uint8](dlen)
    if dlen > 0 and f.readData(addr qt.data[0], dlen) != dlen:
      raise newException(IOError, "EOF quant data: " & name)
    # the name stored in the file is "newBlock::<real name>" (see convert()) - the prefix
    # is removed to match paramList(m)/byName like the base .nimq quantResident.
    let realName = if name.startsWith("newBlock::"): name[10 .. ^1] else: name
    result.quantWeights[realName] = qt

proc humanSize(bytes: int64): string =
  if bytes >= 1_073_741_824: return &"{bytes.float / 1_073_741_824.0:.2f} GB"
  if bytes >= 1_048_576: return &"{bytes.float / 1_048_576.0:.2f} MB"
  if bytes >= 1024: return &"{bytes.float / 1024.0:.1f} KB"
  return $bytes & " B"

proc convert*(src, dst: string, kind: QuantKind = qkInt4Asymmetric, groupSize: int = DefaultGroupSize) =
  if not fileExists(src):
    raise newException(IOError, "Not found: " & src)
  if isQLora4Bin(src):
    echo "[qlora4] The source file is already NIMBIN4 — skipping (already quantized)."
    return

  let overwrite = (absolutePath(src) == absolutePath(dst))
  echo &"[qlora4] Reading the fp32 checkpoint: {src}" & (if overwrite: " (will OVERWRITE the same file)" else: "")
  let srcSz = getFileSize(src)
  let loaded = loadRawBin(src)
  var meta = loaded.meta
  meta["format"] = "nimformers-lora-v1-qlora4"
  meta["quantKind"] = $kind
  meta["quantGroupSize"] = $groupSize

  var fp32Keep: seq[RawBinTensor] = @[]
  var quantList: seq[(string, QuantTensor)] = @[]

  for t in loaded.tensors:
    if t.name.startsWith("adapter::"):
      # Keep A/B in fp32
      fp32Keep.add(t)
    elif t.name.startsWith("newBlock::"):
      if t.data.len == 0:
        echo &"[qlora4] Skipping empty tensor: {t.name}"
        continue
      # FIX: 1-D (bias/gamma/beta) stays raw fp32, only 2-D tensors divisible by 32 are quantized to 4 bits,
      # avoiding the 13-15% RMS quantization error that broke LayerNorm.
      # CHANGED2: attn.* / fuseGate.* are trained fully from 0 with a small lr → values close to the FP4-KM
      # noise; kept raw fp32 (the mixed format still supports FP4-KM for other tensors).
      let bare = if t.name.startsWith("newBlock::"): t.name[10 .. ^1] else: t.name
      let isSensitiveFullTrain =
        bare.contains(".attn.") or bare.contains(".fuseGate.")
      if (not isSensitiveFullTrain) and
         t.shape.len == 2 and t.shape[1] mod 32 == 0:
        let qt = quantizeNewBlock(t.data, t.shape, kind, groupSize)
        quantList.add((t.name, qt))
        let ratio = if t.data.len > 0: (t.data.len * 4).float / max(qt.data.len.float, 1.0) else: 0.0
        echo &"[qlora4]   {t.name}: shape={t.shape}  {humanSize(int64(t.data.len)*4)} -> quant {qt.kind} {humanSize(int64(qt.bytes.len))} (~{ratio:.1f}x)"
      else:
        fp32Keep.add(t)
        let reason = if isSensitiveFullTrain: "attn/fuseGate full-train (fp32)"
                     else: "1D/bias/gamma"
        echo &"[qlora4]   {t.name}: shape={t.shape}  kept as raw FP32 ({reason})"
    else:
      # Kept as is (safe)
      fp32Keep.add(t)

  # Safe overwrite: always writes a temporary file then renames → no half-written file
  let outPath = if overwrite: dst & ".qlora4.tmp" else: dst
  echo &"[qlora4] Writing {outPath} (fp32 adapters={fp32Keep.len}, 4-bit newBlock={quantList.len})"
  saveQLora4Bin(outPath, meta, fp32Keep, quantList)

  if overwrite:
    moveFile(outPath, dst)
    echo &"[qlora4] OVERWROTE {dst}"
  let dstSz = getFileSize(dst)
  echo &"[qlora4] Done. {humanSize(srcSz)} -> {humanSize(dstSz)} (about {(1.0 - dstSz.float/max(srcSz.float,1.0))*100:.0f}% smaller)"
  echo "[qlora4] NIMBIN4 magic — finetune/infer can load it right away."

when isMainModule:
  # Default: overwrite the same file:
  #   ./quantize_lora_bin_4bit jamba_lora.bin
  #   ./quantize_lora_bin_4bit jamba_lora.bin jamba_lora.bin
  let src = if paramCount() >= 1: paramStr(1) else: "jamba_lora.bin"
  let dst = if paramCount() >= 2: paramStr(2) else: src
  var kind = qkInt4Asymmetric
  var gs = DefaultGroupSize
  for i in 3 .. paramCount():
    let a = paramStr(i)
    if a == "--int4": kind = qkInt4
    elif a == "--int4asym": kind = qkInt4Asymmetric
    elif a == "--fp4km": kind = qkFp4KM
    elif a.startsWith("--group="): gs = parseInt(a[8 .. ^1])
  convert(src, dst, kind, gs)

proc loadMappedQLora4Bin*(path: string): tuple[meta: Table[string, string],
    fp32Tensors: seq[RawBinTensor], quantWeights: Table[string, QuantTensor]] =
  var c = WeightCursor(owner: mapWeights(path), pos: 8)
  let magic = mappedView[uint8](c.owner, 0, 8)
  if not equalMem(magic.dataPtr, cstring(MAGIC_QLORA4), 8):
    raise newException(IOError, "Invalid NIMBIN4 checkpoint")
  result.meta = c.readMappedMeta()
  let nr = c.readCount()
  for i in 0 ..< nr: result.fp32Tensors.add c.readMappedRawTensor()
  let nq = c.readCount()
  result.quantWeights = initTable[string, QuantTensor]()
  for i in 0 ..< nq:
    let name = c.readString()
    if name in result.quantWeights: raise newException(IOError, "Duplicate quant tensor: " & name)
    result.quantWeights[name] = c.readMappedQuantTensor()
