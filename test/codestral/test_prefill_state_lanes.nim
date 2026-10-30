## Multi-state-lane scan against a mathematical recurrence, including every
## timestep state, grouped B/C, padded time lanes, and nonzero chunk carry.
import std/[math, strformat]
import mamba2_full_vm, prefill_inference_kernels, vm_interp

for T in [3, 17, 32, 64, 128]:
  for carry in [false, true]:
    let L = prefixLayout(2,T,4,2,128,2,4,8,1e-5'f32,2,false)
    var seed = newSeq[float32](L.fwdEnd)
    for i in 0 ..< L.B*T*L.cDim:
      seed[L.conv+i] = sin(float32(i mod 997)*0.13'f32)*0.4'f32
    for i in 0 ..< L.B*T*L.H:
      seed[L.dt+i] = 0.02'f32 + float32(i mod 13)*0.01'f32
    for h in 0 ..< L.H:
      seed[L.aLog+h] = float32(h)*0.1'f32
      seed[L.dSkip+h] = 0.5'f32
    if carry:
      for b in 0 ..< L.B:
        for i in 0 ..< L.HD*L.N:
          seed[L.states+b*(T+1)*L.HD*L.N+i] = cos(float32(i+b)*0.1'f32)
    var expected = seed
    for b in 0 ..< L.B:
      for t in 0 ..< T:
        for hd in 0 ..< L.HD:
          let h = hd div L.D
          let g = h div (L.H div L.G)
          let row = (b*T+t)*L.cDim
          let dt = seed[L.dt+(b*T+t)*L.H+h]
          let decay = exp(-exp(seed[L.aLog+h])*dt)
          let x = seed[L.conv+row+hd]
          var y = 0'f32
          for n in 0 ..< L.N:
            let prev = L.states+(b*(T+1)+t)*L.HD*L.N+hd*L.N+n
            let next = prev+L.HD*L.N
            expected[next] = expected[prev]*decay + dt*seed[L.conv+row+L.C+g*L.N+n]*x
            y += seed[L.conv+row+L.C+L.GN+g*L.N+n]*expected[next]
          expected[L.yPre+(b*T+t)*L.C+hd] = y+seed[L.dSkip+h]*x
    let tg = m2ScanTgSize(T)
    for lanes in [1,2,4]:
      if tg*4*lanes > 1024: continue
      var actual = seed
      runGrid(buildPrefixParallelScanProgram(L,tg,carry,lanes),actual,
        L.B*L.HD*tg*lanes,tgSize=tg*lanes,tgFloats=4*tg*lanes)
      var error = 0'f32
      for i in L.states ..< L.yPre+L.B*T*L.C:
        doAssert classify(actual[i]) notin {fcNan,fcInf,fcNegInf}
        error = max(error,abs(actual[i]-expected[i]))
      doAssert error < 3e-5'f32, &"T={T} carry={carry} lanes={lanes} error={error}"
    echo &"PASS state lanes T={T} carry={carry}"
