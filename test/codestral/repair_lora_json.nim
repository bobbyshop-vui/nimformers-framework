import std/[os, streams]

proc repair(input, output: string) =
  var src = newFileStream(input, fmRead)
  var dst = newFileStream(output, fmWrite)
  if src == nil or dst == nil: raise newException(IOError, "cannot open repair files")
  defer:
    src.close()
    dst.close()
  var inString = false
  var escaped = false
  var arrayDepth = 0
  var numberState = 0 # 0=none, 1=inside number, 2=just completed number
  var arrayCount: seq[int] = @[]
  var buf = newString(1024 * 1024)
  while not src.atEnd:
    let n = src.readData(addr buf[0], buf.len)
    for i in 0 ..< n:
      let c = buf[i]
      if inString:
        dst.write(c)
        if escaped: escaped = false
        elif c == '\\': escaped = true
        elif c == '"': inString = false
        continue
      if c == '"':
        inString = true
        numberState = 0
        dst.write(c)
        continue
      if c == '[':
        inc arrayDepth
        arrayCount.add(0)
        numberState = 0
        dst.write(c)
        continue
      if c == ']':
        if arrayDepth > 0:
          dec arrayDepth
          discard arrayCount.pop()
        numberState = 0
        dst.write(c)
        continue
      let startsNumber = c in {'-', '0'..'9'}
      if arrayDepth > 0 and startsNumber and numberState == 2:
        dst.write(',')
        numberState = 1
        inc arrayCount[^1]
        dst.write(c)
      elif startsNumber:
        if numberState == 0:
          numberState = 1
          if arrayDepth > 0: inc arrayCount[^1]
        dst.write(c)
      else:
        if numberState == 1: numberState = 2
        if c == ',': numberState = 0
        dst.write(c)

when isMainModule:
  if paramCount() != 1: quit("usage: repair_lora_json <checkpoint.json>")
  let input = paramStr(1)
  let temporary = input & ".repair.tmp"
  repair(input, temporary)
  moveFile(temporary, input)
