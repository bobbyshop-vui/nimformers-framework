import gpubackend

let backend = gbCpu
let a = sessionUpload(backend, @[2'f32, 3])
let b = sessionUpload(backend, @[99'f32, 99, 1, 2, 3, 4])
var c: SessionHandle
if not sessionMatmulNT(backend, a, b, c, 1, 2, 2, 2):
  quit("CPU sessionMatmulNT rejected a valid buffer offset", 1)
let offsetResult = sessionRead(backend, c, 2)
if offsetResult != @[8'f32, 18]:
  quit("CPU sessionMatmulNT returned " & $offsetResult, 1)
sessionFree(backend, c)

let source = @[5'f32, 7, 11]
var uploaded = sessionUploadPrivatePtr(backend, unsafeAddr source[0],
  source.len * sizeof(float32))
let uploadResult = sessionRead(backend, uploaded, source.len)
if uploadResult != source:
  quit("CPU pointer upload returned " & $uploadResult, 1)

var invalidOutput: SessionHandle
if sessionMatmulNT(backend, a, b, invalidOutput, 1, 2, 2, b.cpuData.len):
  quit("CPU sessionMatmulNT accepted an out-of-range buffer offset", 1)

sessionFree(backend, uploaded)
echo "CPU backend kernel regression tests passed"
