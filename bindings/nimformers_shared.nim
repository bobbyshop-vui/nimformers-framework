## Small Nim bindings for calling the installed Nimformers C ABI dynamically.
## Set NIMFORMERS_LIB to override the library path.

import std/[dynlib, json, os, strformat]
from vendor/bybylang/probe import apiHooked

type
  ContextHandle* = distinct pointer
  NimformersSharedError* = object of CatchableError
  ContextCreateNameProc = proc(backendName: cstring): pointer {.cdecl.}
  ContextDestroyProc = proc(handle: pointer): cint {.cdecl.}
  ContextBackendNameProc = proc(handle: pointer): cstring {.cdecl.}
  AddProc = proc(ctx: pointer; a, b, outData: ptr cfloat; n: int64): cint {.cdecl.}
  LastErrorProc = proc(): cstring {.cdecl.}
  AbiVersionProc = proc(): uint32 {.cdecl.}
  FreeStringProc = proc(value: cstring) {.cdecl.}
  ParallelCreateProc = proc(devices, ratios: cstring): pointer {.cdecl.}
  ParallelDestroyProc = proc(handle: pointer): cint {.cdecl.}
  ParallelPartitionProc = proc(handle: pointer; length: int64; ratios: cstring): cstring {.cdecl.}
  ParallelGenerateProc = proc(handle: pointer; request: cstring): cstring {.cdecl.}
  ParallelVmRunProc = proc(handle: pointer; words: ptr uint32; wordsLen: int64;
    memory: ptr cfloat; memoryLen, itemSize, threadsPerItem, tgFloats, tgSize: int64;
    ratios, workerPath: cstring; output: ptr cfloat): cint {.cdecl.}

const defaultLibrary = when defined(windows): "nimformers.dll"
  elif defined(macosx): "libnimformers.dylib"
  else: "libnimformers.so"
const defaultWorker = when defined(windows): "nimformers-worker.exe"
  else: "nimformers-worker"

proc libraryCandidates(): seq[string] =
  let packageRoot = currentSourcePath().parentDir.parentDir
  result = @[
    packageRoot / "lib" / defaultLibrary,
    packageRoot / defaultLibrary,
    getAppDir() / "lib" / defaultLibrary,
    getAppDir() / defaultLibrary,
    getCurrentDir() / "lib" / defaultLibrary,
    getCurrentDir() / defaultLibrary,
  ]

proc findLibrary*(explicitPath = ""): string  {.apiHooked.}=
  ## Find the platform library in the Nimble package or return an explicit path.
  ## `NIMFORMERS_LIB` takes precedence over package-relative lookup.
  let requested = if explicitPath.len > 0: explicitPath else: getEnv("NIMFORMERS_LIB")
  if requested.len > 0:
    let path = absolutePath(requested)
    if fileExists(path):
      return path
    raise newException(NimformersSharedError,
      fmt"Nimformers shared library not found at {path}")
  for candidate in libraryCandidates():
    if fileExists(candidate):
      return absolutePath(candidate)
  raise newException(NimformersSharedError,
    fmt"cannot find {defaultLibrary} beside the installed Nimble package; " &
      "build/install the shared library or set NIMFORMERS_LIB to its full path")

var libraryHandle: LibHandle
var contextCreateNameImpl: ContextCreateNameProc
var contextDestroyImpl: ContextDestroyProc
var contextBackendNameImpl: ContextBackendNameProc
var addImpl: AddProc
var lastErrorImpl: LastErrorProc
var abiVersionImpl: AbiVersionProc
var freeStringImpl: FreeStringProc
var parallelCreateImpl: ParallelCreateProc
var parallelDestroyImpl: ParallelDestroyProc
var parallelPartitionImpl: ParallelPartitionProc
var parallelGenerateImpl: ParallelGenerateProc
var parallelVmRunImpl: ParallelVmRunProc

proc loadLibrary() =
  if libraryHandle != nil:
    return
  let configured = getEnv("NIMFORMERS_LIB")
  if configured.len > 0:
    libraryHandle = loadLib(findLibrary(configured))
  else:
    for candidate in libraryCandidates():
      libraryHandle = loadLib(candidate)
      if libraryHandle != nil:
        break
    if libraryHandle == nil:
      # Also permit libraries installed in the platform's loader search path.
      libraryHandle = loadLib(defaultLibrary)
  if libraryHandle == nil:
    var resolved = "not found"
    try:
      resolved = findLibrary()
    except NimformersSharedError:
      discard
    raise newException(NimformersSharedError,
      fmt"cannot load {defaultLibrary} (searched {resolved}); set NIMFORMERS_LIB to its full path")
  contextCreateNameImpl = cast[ContextCreateNameProc](symAddr(libraryHandle, "nimf_context_create_name"))
  contextDestroyImpl = cast[ContextDestroyProc](symAddr(libraryHandle, "nimf_context_destroy"))
  contextBackendNameImpl = cast[ContextBackendNameProc](symAddr(libraryHandle, "nimf_context_backend_name"))
  addImpl = cast[AddProc](symAddr(libraryHandle, "nimf_add"))
  lastErrorImpl = cast[LastErrorProc](symAddr(libraryHandle, "nimf_last_error"))
  abiVersionImpl = cast[AbiVersionProc](symAddr(libraryHandle, "nimf_abi_version"))
  freeStringImpl = cast[FreeStringProc](symAddr(libraryHandle, "nimf_free_string"))
  parallelCreateImpl = cast[ParallelCreateProc](symAddr(libraryHandle, "nimf_parallel_create"))
  parallelDestroyImpl = cast[ParallelDestroyProc](symAddr(libraryHandle, "nimf_parallel_destroy"))
  parallelPartitionImpl = cast[ParallelPartitionProc](symAddr(libraryHandle, "nimf_parallel_partition"))
  parallelGenerateImpl = cast[ParallelGenerateProc](symAddr(libraryHandle, "nimf_parallel_generate"))
  parallelVmRunImpl = cast[ParallelVmRunProc](symAddr(libraryHandle, "nimf_parallel_vm_run"))
  if contextCreateNameImpl == nil or contextDestroyImpl == nil or
      contextBackendNameImpl == nil or addImpl == nil or lastErrorImpl == nil or
      abiVersionImpl == nil:
    unloadLib(libraryHandle)
    libraryHandle = nil
    raise newException(NimformersSharedError,
      "the loaded library does not export the required Nimformers C ABI v2 symbols")

proc checkStatus(status: cint) =
  if status != 0:
    let message = if lastErrorImpl == nil: "unknown Nimformers error" else: $lastErrorImpl()
    raise newException(NimformersSharedError, message)

proc abiVersion*(): uint32  {.apiHooked.}=
  loadLibrary()
  result = abiVersionImpl()

proc newContext*(backend = "auto"): ContextHandle  {.apiHooked.}=
  loadLibrary()
  let handle = contextCreateNameImpl(backend.cstring)
  if handle == nil:
    raise newException(NimformersSharedError, $lastErrorImpl())
  result = ContextHandle(handle)

proc close*(ctx: var ContextHandle)  {.apiHooked.}=
  if cast[pointer](ctx) == nil:
    return
  loadLibrary()
  checkStatus(contextDestroyImpl(cast[pointer](ctx)))
  ctx = ContextHandle(nil)

proc backendName*(ctx: ContextHandle): string  {.apiHooked.}=
  loadLibrary()
  let name = contextBackendNameImpl(cast[pointer](ctx))
  if name == nil:
    raise newException(NimformersSharedError, $lastErrorImpl())
  result = $name

proc add*(ctx: ContextHandle; a, b: openArray[float32]): seq[float32]  {.apiHooked.}=
  if a.len != b.len:
    raise newException(NimformersSharedError, "add inputs must have equal lengths")
  loadLibrary()
  result = newSeq[float32](a.len)
  if a.len == 0:
    return
  checkStatus(addImpl(cast[pointer](ctx), unsafeAddr a[0], unsafeAddr b[0],
    addr result[0], int64(a.len)))

proc lastError*(): string  {.apiHooked.}=
  loadLibrary()
  result = $lastErrorImpl()

type ParallelGroupHandle* = distinct pointer

proc ratioJson(ratios: openArray[float64]): string =
  if ratios.len == 0:
    return ""
  var values = newJArray()
  for ratio in ratios:
    values.add(%ratio)
  result = $values

proc newProcessGroup*(devices: openArray[string]; ratios: openArray[float64] = []): ParallelGroupHandle  {.apiHooked.}=
  loadLibrary()
  if parallelCreateImpl == nil or parallelDestroyImpl == nil or
      parallelPartitionImpl == nil or parallelVmRunImpl == nil or freeStringImpl == nil:
    raise newException(NimformersSharedError,
      "the loaded library does not export the parallel C ABI")
  var deviceValues = newJArray()
  for device in devices:
    deviceValues.add(%device)
  let deviceText = $deviceValues
  let ratioText = ratioJson(ratios)
  let handle = parallelCreateImpl(deviceText.cstring,
    (if ratioText.len == 0: nil else: ratioText.cstring))
  if handle == nil:
    raise newException(NimformersSharedError, $lastErrorImpl())
  result = ParallelGroupHandle(handle)

proc close*(group: var ParallelGroupHandle)  {.apiHooked.}=
  if cast[pointer](group) == nil:
    return
  loadLibrary()
  checkStatus(parallelDestroyImpl(cast[pointer](group)))
  group = ParallelGroupHandle(nil)

proc partition*(group: ParallelGroupHandle; length: int;
    ratios: openArray[float64] = []): seq[int]  {.apiHooked.}=
  loadLibrary()
  let ratioText = ratioJson(ratios)
  let encoded = parallelPartitionImpl(cast[pointer](group), int64(length),
    (if ratioText.len == 0: nil else: ratioText.cstring))
  if encoded == nil:
    raise newException(NimformersSharedError, $lastErrorImpl())
  try:
    for value in parseJson($encoded):
      result.add(value.getInt())
  finally:
    freeStringImpl(encoded)

proc vmRun*(group: ParallelGroupHandle; words: openArray[uint32];
    memory: openArray[float32]; itemSize, threadsPerItem: int;
    tgFloats = 0; tgSize = 0; ratios: openArray[float64] = [];
    workerPath = ""): seq[float32]  {.apiHooked.}=
  if words.len == 0:
    raise newException(NimformersSharedError, "micro-VM bytecode must not be empty")
  if itemSize <= 0 or threadsPerItem <= 0:
    raise newException(NimformersSharedError, "itemSize and threadsPerItem must be positive")
  loadLibrary()
  if parallelVmRunImpl == nil:
    raise newException(NimformersSharedError,
      "the loaded library does not export nimf_parallel_vm_run")
  result = newSeq[float32](memory.len)
  let ratioText = ratioJson(ratios)
  var workerName = workerPath
  if workerName.len == 0:
    try:
      let sibling = findLibrary().parentDir / defaultWorker
      if fileExists(sibling):
        workerName = sibling
    except NimformersSharedError:
      discard
  checkStatus(parallelVmRunImpl(cast[pointer](group), unsafeAddr words[0],
    int64(words.len), (if memory.len == 0: nil else: unsafeAddr memory[0]),
    int64(memory.len), int64(itemSize), int64(threadsPerItem), int64(tgFloats),
    int64(tgSize), (if ratioText.len == 0: nil else: ratioText.cstring),
    (if workerName.len == 0: nil else: workerName.cstring),
    (if result.len == 0: nil else: addr result[0])))

proc generateBatchImpl(group: ParallelGroupHandle; executable: string;
    args, prompts: openArray[string]; tokenIds: bool; maxTokens: int;
    temperature: float64; ratios: openArray[float64];
    environment: openArray[(string, string)];
    unsetEnvironment: openArray[string]): seq[string] =
  loadLibrary()
  if parallelGenerateImpl == nil:
    raise newException(NimformersSharedError,
      "the loaded library does not export nimf_parallel_generate")
  var request = newJObject()
  request["executable"] = %executable
  request["args"] = %* args
  request["prompts"] = %* prompts
  request["token_ids"] = %tokenIds
  request["max_tokens"] = %maxTokens
  request["temperature"] = %temperature
  if ratios.len > 0:
    var ratioValues = newJArray()
    for ratio in ratios:
      ratioValues.add(%ratio)
    request["ratios"] = ratioValues
  if environment.len > 0 or unsetEnvironment.len > 0:
    var env = newJObject()
    for (key, value) in environment:
      env[key] = %value
    for key in unsetEnvironment:
      env[key] = newJNull()
    request["env"] = env
  let requestText = $request
  let encoded = parallelGenerateImpl(cast[pointer](group), requestText.cstring)
  if encoded == nil:
    raise newException(NimformersSharedError, $lastErrorImpl())
  try:
    for value in parseJson($encoded):
      result.add(value.getStr())
  finally:
    freeStringImpl(encoded)

proc generateBatch*(group: ParallelGroupHandle; executable: string;
    args, prompts: openArray[string]; maxTokens = 128; temperature = 0.8;
    ratios: openArray[float64] = []; environment: openArray[(string, string)] = [];
    unsetEnvironment: openArray[string] = []): seq[string]  {.apiHooked.}=
  ## Run independent text prompts. Results are returned in prompt order.
  generateBatchImpl(group, executable, args, prompts, false, maxTokens,
    temperature, ratios, environment, unsetEnvironment)

proc generateIdsBatch*(group: ParallelGroupHandle; executable: string;
    args, promptIds: openArray[string]; maxTokens = 128; temperature = 0.8;
    ratios: openArray[float64] = []; environment: openArray[(string, string)] = [];
    unsetEnvironment: openArray[string] = []): seq[string]  {.apiHooked.}=
  ## Run independent prompt-token sequences; each result is space-separated IDs.
  generateBatchImpl(group, executable, args, promptIds, true, maxTokens,
    temperature, ratios, environment, unsetEnvironment)
