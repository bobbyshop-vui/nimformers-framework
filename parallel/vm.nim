## Batch independent micro-VM inputs through the native process scheduler.
import std/[hashes, math, os]
import vendor/bybylang/distributed

const VmWorkerMagic* = "NFPVM001"

type VmWorkerHeader* = array[5, int64] # words, floats, threads, tg floats, tg size

proc vmWorkerRequest(words: seq[uint32]; memory: seq[float32];
                     first, count, threads, tgFloats, tgSize: int): string =
  let header: VmWorkerHeader = [words.len.int64, count.int64, threads.int64,
                                tgFloats.int64, tgSize.int64]
  result = newString(VmWorkerMagic.len + sizeof(header) + words.len * 4 + count * 4)
  copyMem(addr result[0], cstring(VmWorkerMagic), VmWorkerMagic.len)
  var offset = VmWorkerMagic.len
  copyMem(addr result[offset], unsafeAddr header[0], sizeof(header))
  offset += sizeof(header)
  copyMem(addr result[offset], unsafeAddr words[0], words.len * 4)
  offset += words.len * 4
  copyMem(addr result[offset], unsafeAddr memory[first], count * 4)

proc resolveVmWorker(path: string): string =
  let requested = if path.len > 0: path else: getEnv("NIMFORMERS_WORKER")
  if requested.len > 0: return requested
  let name = addFileExt("nimformers-worker", ExeExt)
  let library = getEnv("NIMFORMERS_LIB")
  if library.len > 0:
    let sibling = parentDir(library) / name
    if fileExists(sibling): return absolutePath(sibling)
  for dir in [getAppDir(), getCurrentDir()]:
    if fileExists(dir / name): return dir / name
  result = findExe(name)
  if result.len == 0:
    raise newException(IOError, "nimformers-worker not found; build with make build-shared or set NIMFORMERS_WORKER")

proc runVmParallel*(group: ProcessGroup; words: seq[uint32]; memory: seq[float32];
                    itemSize, threadsPerItem: int; tgFloats = 0; tgSize = 0;
                    ratios: openArray[float64] = []; workerPath = ""): seq[float32] =
  ## Addressing is shard-relative. Explicit threadgroups are never split.
  if words.len == 0 or words.len > high(int32).int or itemSize <= 0 or threadsPerItem <= 0 or
      tgFloats < 0 or tgFloats > high(int32).int div 4 or tgSize < 0 or
      tgSize > high(int32).int or memory.len mod itemSize != 0:
    raise newException(ValueError, "invalid parallel micro-VM program or launch geometry")
  let items = memory.len div itemSize
  let groupItems = if tgSize == 0: 1 else: tgSize div gcd(tgSize, threadsPerItem)
  if items mod groupItems != 0:
    raise newException(ValueError, "item count must be divisible by " & $groupItems &
      " for this threadgroup size")
  if items == 0:
    discard group.partition(0, ratios)
    return @[]
  if items > high(int32).int div threadsPerItem or memory.len > high(int32).int:
    raise newException(ValueError, "parallel micro-VM launch exceeds the supported extent")
  let worker = resolveVmWorker(workerPath)
  let workload = "vm:" & $hash(words) & ":" & $itemSize & ":" & $threadsPerItem &
    ":" & $tgSize & ":" & $tgFloats
  var extents: seq[int]
  let jobs = group.runShards(items div groupItems,
    proc(device: ParallelDevice; offset, count: int): ParallelCommand =
      let first = offset * groupItems
      let rows = count * groupItems
      extents.add rows * itemSize
      ParallelCommand(executable: worker, args: @[device.backend],
        input: vmWorkerRequest(words, memory, first * itemSize, rows * itemSize,
          rows * threadsPerItem, tgFloats, tgSize)),
    ratios, workload)
  result = newSeq[float32](memory.len)
  var written = 0
  if jobs.len != extents.len:
    raise newException(IOError, "incomplete micro-VM worker results")
  for i, job in jobs:
    let bytes = job.stdout.len
    if bytes != extents[i] * 4 or extents[i] > result.len - written:
      raise newException(IOError, "invalid micro-VM worker response")
    copyMem(addr result[written], unsafeAddr job.stdout[0], bytes)
    written += extents[i]
  if written != result.len:
    raise newException(IOError, "incomplete micro-VM worker response")
