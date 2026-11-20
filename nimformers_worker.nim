## Private local worker for shared-library and native parallel micro-VM calls.
import std/[os, strutils]
import backend, mamba2_ssm_gpu, parallel_vm
import vendor/bybylang/micro_kernel

proc main() =
  let request = stdin.readAll()
  var header: VmWorkerHeader
  if paramCount() != 1 or paramStr(1) notin ["cpu", "metal", "vulkan"] or
      not request.startsWith(VmWorkerMagic) or request.len < VmWorkerMagic.len + sizeof(header):
    raise newException(ValueError, "invalid micro-VM worker request")
  var offset = VmWorkerMagic.len
  copyMem(addr header[0], unsafeAddr request[offset], sizeof(header))
  offset += sizeof(header)
  let (words, values, threads, tgFloats, tgSize) = (header[0], header[1], header[2], header[3], header[4])
  let available = (request.len - offset) div 4
  if words <= 0 or values <= 0 or words > available or values > available - words or
      (words + values) * 4 != request.len - offset or
      threads <= 0 or threads > high(int32) or tgFloats < 0 or tgFloats > high(int32) div 4 or
      tgSize < 0 or tgSize > high(int32):
    raise newException(ValueError, "invalid micro-VM worker payload")
  var prog: VmProgram
  prog.words.setLen(int(words))
  copyMem(addr prog.words[0], unsafeAddr request[offset], int(words) * 4)
  offset += int(words) * 4
  var memory = newSeq[float32](int(values))
  copyMem(addr memory[0], unsafeAddr request[offset], int(values) * 4)
  let ctx = newBackend(paramStr(1))
  backend.setForbidCpuFallback(true)
  defer: closeBackend(ctx)
  if not runMicroVm(ctx, prog, memory, int(threads), max(4, int(tgFloats) * 4),
                   tgSize = int(tgSize)):
    raise newException(IOError, "micro-VM dispatch failed")
  if memory.len != values or stdout.writeBuffer(addr memory[0], memory.len * 4) != memory.len * 4:
    raise newException(IOError, "incomplete micro-VM worker output")

when isMainModule:
  try:
    main()
  except CatchableError as error:
    stderr.writeLine("nimformers-worker: " & error.msg)
    quit(1)
