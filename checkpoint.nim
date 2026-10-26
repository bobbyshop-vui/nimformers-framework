# checkpoint.nim
# Utility for per‑layer in‑progress checkpoints during training.
#
# The `.bin` checkpoint must only ever contain weights from a *completed*
# epoch. Progress inside a step is therefore written per block into a side
# directory next to the checkpoint, and the `.bin` is only rewritten once the
# epoch commit succeeds. The side directory is deleted on that commit.
#
# This module is intentionally lightweight – it only deals with raw
# `seq[float32]` tensors and uses Nim's `writeBuffer`/`readBuffer` for
# binary I/O.  It does not depend on any other part of the framework.

import std/[os, strutils]

# ─────────────────────────────────────────────────────────────
# On-disk layout, all under `<ckptPath>.inprogress/`:
#
#   cursor              "step <n>\nlayer <i>\n" - last block fully written
#   step-<n>/
#     layer-<i>.bin     payload: "NIMLAYER1" magic + raw float32 tensors
#     layer-<i>.idx     manifest: tensor count, then (name, shape) per tensor
#
# Every file is written to a `.tmp` sibling and then atomically renamed, so a
# crash can never leave a half-written layer file that a later run would read
# as if it were complete.
#
# NOTE: new-block (Attention / fuseGate) weights are already durable in the
# AttnWeightDiskStore and Adam moments in the AdamDiskStore, so only the LoRA
# adapters - which otherwise live in RAM only - are duplicated here.
# ─────────────────────────────────────────────────────────────

type
  LayerTensor* = object
    name*: string
    shape*: seq[int]
    data*: seq[float32]

const ProgressMagic = "NIMLAYER1"
const ProgressMagicLen = ProgressMagic.len

# POSIX rename() replaces the destination in one step. Nim's moveFile() is not
# guaranteed to overwrite on every platform, so bind rename() directly.
proc cRename(oldPath, newPath: cstring): cint {.importc: "rename", header: "<stdio.h>".}

proc atomicReplace(tmpPath, finalPath: string) =
  ## Commit `tmpPath` as `finalPath`. Falls back to remove+move on filesystems
  ## that refuse the rename, so a freshly written file is never lost.
  if cRename(tmpPath.cstring, finalPath.cstring) != 0:
    if fileExists(finalPath): removeFile(finalPath)
    moveFile(tmpPath, finalPath)

proc progressDir*(ckptPath: string): string =
  ## Side directory holding per-layer progress for the checkpoint at
  ## `ckptPath`. Deliberately distinct from `<ckptPath>.resume.inprogress`,
  ## which is a plain marker file owned by the resume-commit protocol.
  result = ckptPath & ".inprogress"

proc stepDir(ckptPath: string, step: int): string =
  result = progressDir(ckptPath) / ("step-" & $step)

proc layerBinPath(ckptPath: string, step, layer: int): string =
  result = stepDir(ckptPath, step) / ("layer-" & $layer & ".bin")

proc layerIdxPath(ckptPath: string, step, layer: int): string =
  result = stepDir(ckptPath, step) / ("layer-" & $layer & ".idx")

proc cursorPath(ckptPath: string): string =
  result = progressDir(ckptPath) / "cursor"

proc removeDirRecursive(d: string) =
  ## os.removeDir() refuses to delete a non-empty directory, so a plain
  ## `removeDir(progressDir)` would silently leave stale step-*/layer-* files
  ## behind forever. Walk the tree bottom-up instead.
  if not dirExists(d): return
  for kind, path in walkDir(d, relative = false):
    if kind == pcDir:
      removeDirRecursive(path)
    else:
      try: removeFile(path)
      except OSError: discard
  try: removeDir(d)
  except OSError: discard

proc clearLayerProgress*(ckptPath: string) =
  ## Remove the whole progress directory. Called once the real `.bin` has been
  ## committed for a completed epoch, and at startup to drop stale files left
  ## behind by an interrupted run.
  removeDirRecursive(progressDir(ckptPath))

proc hasLayerProgress*(ckptPath: string, step: int): bool =
  result = dirExists(stepDir(ckptPath, step))

proc writeCursor(ckptPath: string, step, layer: int) =
  createDir(progressDir(ckptPath))
  let p = cursorPath(ckptPath)
  let tmp = p & ".tmp"
  writeFile(tmp, "step " & $step & "\nlayer " & $layer & "\n")
  atomicReplace(tmp, p)

proc readProgressCursor*(ckptPath: string, stepOut, layerOut: var int): bool =
  ## Returns the (step, layer) of the last block fully written, or false when
  ## no cursor exists / it is unreadable.
  let p = cursorPath(ckptPath)
  if not fileExists(p): return false
  try:
    let parts = readFile(p).splitWhitespace()
    if parts.len < 4: return false
    if parts[0] != "step" or parts[2] != "layer": return false
    stepOut = parseInt(parts[1])
    layerOut = parseInt(parts[3])
    result = true
  except CatchableError:
    result = false

proc saveLayerProgress*(ckptPath: string, step, layer: int,
                        tensors: openArray[LayerTensor]) =
  ## Persist the trainable tensors of one block. `step` is the training step
  ## the block belongs to, `layer` is the block index.
  let d = stepDir(ckptPath, step)
  createDir(d)

  # ---- payload ----
  block payload:
    let finalPath = layerBinPath(ckptPath, step, layer)
    let tmpPath = finalPath & ".tmp"
    var f = open(tmpPath, fmWrite)
    try:
      var magic = ProgressMagic
      discard f.writeBuffer(unsafeAddr magic[0], ProgressMagicLen)
      for t in tensors:
        if t.data.len > 0:
          discard f.writeBuffer(unsafeAddr t.data[0], t.data.len * sizeof(float32))
    finally:
      f.close()
    atomicReplace(tmpPath, finalPath)

  # ---- manifest ----
  block manifest:
    let finalPath = layerIdxPath(ckptPath, step, layer)
    let tmpPath = finalPath & ".tmp"
    var f = open(tmpPath, fmWrite)
    try:
      var n = int32(tensors.len)
      discard f.writeBuffer(addr n, sizeof(n))
      for t in tensors:
        var nl = int32(t.name.len)
        discard f.writeBuffer(addr nl, sizeof(nl))
        if t.name.len > 0:
          discard f.writeBuffer(unsafeAddr t.name[0], t.name.len)
        var nd = int32(t.shape.len)
        discard f.writeBuffer(addr nd, sizeof(nd))
        for dim in t.shape:
          var dv = int32(dim)
          discard f.writeBuffer(addr dv, sizeof(dv))
    finally:
      f.close()
    atomicReplace(tmpPath, finalPath)

  writeCursor(ckptPath, step, layer)

proc loadLayerProgress*(ckptPath: string, step, layer: int): seq[LayerTensor] =
  ## Read back exactly what `saveLayerProgress` wrote, or an empty seq when
  ## the files are missing / truncated / carry the wrong magic.
  let binPath = layerBinPath(ckptPath, step, layer)
  let idxPath = layerIdxPath(ckptPath, step, layer)
  if not fileExists(binPath) or not fileExists(idxPath):
    return newSeq[LayerTensor]()

  var names: seq[string] = @[]
  var shapes: seq[seq[int]] = @[]
  block readManifest:
    var f = open(idxPath, fmRead)
    try:
      var n: int32
      if f.readBuffer(addr n, sizeof(n)) != sizeof(n): return newSeq[LayerTensor]()
      if n < 0: return newSeq[LayerTensor]()
      for _ in 0 ..< int(n):
        var nl: int32
        if f.readBuffer(addr nl, sizeof(nl)) != sizeof(nl): return newSeq[LayerTensor]()
        if nl < 0: return newSeq[LayerTensor]()
        var nm = newString(int(nl))
        if nl > 0 and f.readBuffer(addr nm[0], int(nl)) != int(nl): return newSeq[LayerTensor]()
        names.add(nm)
        var nd: int32
        if f.readBuffer(addr nd, sizeof(nd)) != sizeof(nd): return newSeq[LayerTensor]()
        if nd < 0: return newSeq[LayerTensor]()
        var shp = newSeq[int](int(nd))
        for k in 0 ..< int(nd):
          var dv: int32
          if f.readBuffer(addr dv, sizeof(dv)) != sizeof(dv): return newSeq[LayerTensor]()
          shp[k] = int(dv)
        shapes.add(shp)
    finally:
      f.close()

  var f = open(binPath, fmRead)
  try:
    var magic = newString(ProgressMagicLen)
    if f.readBuffer(addr magic[0], ProgressMagicLen) != ProgressMagicLen:
      return newSeq[LayerTensor]()
    if magic != ProgressMagic: return newSeq[LayerTensor]()
    result = newSeq[LayerTensor](names.len)
    for i in 0 ..< names.len:
      var cnt = 1
      for d in shapes[i]: cnt *= d
      result[i].name = names[i]
      result[i].shape = shapes[i]
      result[i].data = newSeq[float32](cnt)
      if cnt > 0:
        let nBytes = cnt * sizeof(float32)
        if f.readBuffer(addr result[i].data[0], nBytes) != nBytes:
          return newSeq[LayerTensor]()
  finally:
    f.close()
