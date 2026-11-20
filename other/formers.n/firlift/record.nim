## Record a trace straight into the lifter, with nothing written to the disk.
##
## `recordCommand` starts the command with `BYBY_FIR_PROBE=fd:T,D`: T and D are the write ends of
## two pipes. The core recorder (vendor/bybylang/probe.nim) writes the finished trace text to T when
## it stops, and every exact upload payload to D, deduplicated by content. This process reads both
## pipes until the command ends and returns them as strings.
import std/[os, osproc, posix, strutils]

type Recorded* = object
  trace*, data*: string
  exitCode*: int

proc closeQuietly(fd: cint) =
  if fd >= 0: discard posix.close(fd)

proc recordCommand*(command: seq[string]; codestral = false): Recorded =
  if command.len == 0: raise newException(ValueError, "--run needs a command after it")
  var tracePipe, dataPipe: array[2, cint]
  if pipe(tracePipe) != 0 or pipe(dataPipe) != 0:
    raise newException(IOError, "cannot create the trace pipes: " & osErrorMsg(osLastError()))
  # The command inherits only the write ends; the read ends stay with this process.
  discard fcntl(tracePipe[0], F_SETFD, FD_CLOEXEC)
  discard fcntl(dataPipe[0], F_SETFD, FD_CLOEXEC)
  let fd = "fd:" & $tracePipe[1]
  if codestral:
    putEnv("BYBY_FIR_CODESTRAL", "1")
    putEnv("BYBY_FIR_PROBE", fd)
    putEnv("BYBY_FIR_PROBE_DATA", "0")
  else:
    putEnv("BYBY_FIR_PROBE", fd & "," & $dataPipe[1])
    # Legacy lifts can use captured upload bytes for unnamed inputs.
    if getEnv("BYBY_FIR_PROBE_DATA").len == 0: putEnv("BYBY_FIR_PROBE_DATA", "1")
  var child: Process
  try:
    child = startProcess(command[0], args = command[1 .. ^1],
      options = {poParentStreams, poUsePath})
  except OSError, IOError:
    closeQuietly(tracePipe[0]); closeQuietly(tracePipe[1])
    closeQuietly(dataPipe[0]); closeQuietly(dataPipe[1])
    raise
  # Our copies of the write ends must go, or the reads below never see end of file.
  closeQuietly(tracePipe[1])
  closeQuietly(dataPipe[1])
  delEnv("BYBY_FIR_PROBE")
  var fds = [TPollfd(fd: tracePipe[0], events: POLLIN), TPollfd(fd: dataPipe[0], events: POLLIN)]
  var buffer = newString(1 shl 20)
  var openCount = 2
  while openCount > 0:
    let ready = poll(addr fds[0], 2, 200)
    if ready < 0:
      if errno == EINTR: continue
      raise newException(IOError, "poll on the trace pipes failed: " & osErrorMsg(osLastError()))
    if ready == 0:
      # A command that has exited but whose pipe a grandchild still holds open: stop waiting.
      if not child.running: break
      continue
    for i in 0 .. 1:
      if fds[i].fd < 0 or (fds[i].revents and (POLLIN or POLLHUP or POLLERR)) == 0: continue
      let n = read(fds[i].fd, addr buffer[0], buffer.len)
      if n < 0:
        if errno == EINTR: continue
        raise newException(IOError, "read from the trace pipe failed: " & osErrorMsg(osLastError()))
      if n == 0:
        closeQuietly(fds[i].fd)
        fds[i].fd = -1
        dec openCount
        continue
      template append(target: var string) =
        let at = target.len
        target.setLen(at + n)
        copyMem(addr target[at], addr buffer[0], n)
      if i == 0: append(result.trace) else: append(result.data)
  for i in 0 .. 1: closeQuietly(fds[i].fd)
  result.exitCode = child.waitForExit()
  child.close()
