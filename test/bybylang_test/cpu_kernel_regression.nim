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

let q4Weights = residentUploadQ4(backend, @[0x21'u8, 0x43], @[1'f32, 1], @[0'f32, 0],
  2, 2, 2, 1)
let activation = sessionUpload(backend, @[5'f32, 6])
var q4Output: SessionHandle
if not sessionMatmulQ4(backend, activation, q4Weights, q4Output, 1):
  quit("CPU sessionMatmulQ4 rejected resident packed weights", 1)
let q4Result = sessionRead(backend, q4Output, 2)
if q4Result != @[17'f32, 39]:
  quit("CPU resident Q4 matmul returned " & $q4Result, 1)
sessionFree(backend, q4Output)
var q4WeightsToFree = q4Weights
residentFreeQ4(q4WeightsToFree)

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
