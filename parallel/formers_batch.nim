## Native independent-prompt inference through the shared process scheduler.
import std/strutils
import vendor/bybylang/distributed

proc normalizedArgs(baseArgs: openArray[string]): seq[string] =
  var i = 0
  while i < baseArgs.len:
    let arg = baseArgs[i]
    if arg in ["--backend", "-p", "--prompt", "-f", "--file", "--prompt-ids"]:
      i += 2
    elif arg.startsWith("--backend=") or arg.startsWith("--prompt=") or
        arg.startsWith("--file=") or arg.startsWith("--prompt-ids="):
      inc i
    else:
      result.add(arg)
      inc i

proc runFormersBatchDetailed*(group: ProcessGroup; executable: string; baseArgs: seq[string];
                              prompts: seq[string]; tokenIds: bool; maxTokens: int;
                              temperature: float64; environment: seq[(string, string)];
                              ratios: seq[float64]; unsetEnvironment: seq[string] = @[]): seq[ParallelResult] =
  ## Run one CLI process per prompt and retain stdout and stderr by input index.
  if group.isNil:
    raise newException(ValueError, "parallel inference requires an initialized process group")
  if executable.len == 0:
    raise newException(ValueError, "formers-cli executable path cannot be empty")
  if maxTokens < 0:
    raise newException(ValueError, "maxTokens must be non-negative")
  let common = normalizedArgs(baseArgs)
  var commands = newSeq[ParallelCommand](prompts.len)
  for i, prompt in prompts:
    var args = common
    let backendArgIndex = args.len + 1
    args.add(["--backend", "", "-n", $maxTokens, "--temp", $temperature])
    if tokenIds:
      args.add(["--prompt-ids", prompt])
    else:
      args.add(["-p", prompt])
    commands[i] = ParallelCommand(executable: executable, args: args,
      environment: environment, unsetEnvironment: unsetEnvironment,
      replaceBackendArg: true, backendArgIndex: backendArgIndex)
  result = group.runCommands(commands, ratios, "formers-prompt-batch")

proc runFormersBatch*(group: ProcessGroup; executable: string; baseArgs: seq[string];
                      prompts: seq[string]; tokenIds: bool; maxTokens: int;
                      temperature: float64; environment: seq[(string, string)];
                      ratios: seq[float64]; unsetEnvironment: seq[string] = @[]): seq[string] =
  ## Run a CLI batch and return stdout in prompt order.
  for item in runFormersBatchDetailed(group, executable, baseArgs, prompts, tokenIds,
      maxTokens, temperature, environment, ratios, unsetEnvironment):
    var output = item.stdout
    while output.endsWith("\n") or output.endsWith("\r"):
      output.setLen(output.len - 1)
    result.add(output)
