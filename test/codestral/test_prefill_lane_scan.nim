## Register-state lane scan against the mathematical recurrence and the serial
## compact kernel: grouped B/C, multi-chunk time (T > 32), batch > 1 and a
## nonzero carried initial state.
import std/[math, strformat]
import mamba2_full_vm, prefill_inference_kernels, vm_interp

for N in [32, 64, 128]:
  for T in [2, 3, 17, 32, 33, 64, 128]:
    for carry in [false, true]:
      let L = prefixLayout(2,T,4,2,N,2,4,8,1e-5'f32,2,true)
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
            seed[L.states+b*2*L.HD*L.N+i] = cos(float32(i+b)*0.1'f32)
      var expected = seed
      for b in 0 ..< L.B:
        var s = newSeq[float32](L.HD*L.N)
        for i in 0 ..< s.len: s[i] = seed[L.states+b*2*L.HD*L.N+i]
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
              s[hd*L.N+n] = s[hd*L.N+n]*decay + dt*seed[L.conv+row+L.C+g*L.N+n]*x
              y += seed[L.conv+row+L.C+L.GN+g*L.N+n]*s[hd*L.N+n]
            expected[L.yPre+(b*T+t)*L.C+hd] = y+seed[L.dSkip+h]*x
        for i in 0 ..< s.len:
          expected[L.states+b*2*L.HD*L.N+i] = s[i]
          expected[L.states+(b*2+1)*L.HD*L.N+i] = s[i]
      var serial = seed
      runGrid(buildCompactPrefixScanProgram(L),serial,L.B*L.HD,tgSize=1,tgFloats=1)
      var lanes = seed
      runGrid(buildPrefixLaneScanProgram(L),lanes,L.B*L.HD*LaneScanWidth,
        tgSize=LaneScanWidth,tgFloats=laneScanTgBytes(T) div 4)
      var errMath, errSerial = 0'f32
      for i in L.states ..< L.yPre+L.B*T*L.C:
        doAssert classify(lanes[i]) notin {fcNan,fcInf,fcNegInf}
        errMath = max(errMath,abs(lanes[i]-expected[i]))
        errSerial = max(errSerial,abs(lanes[i]-serial[i]))
      doAssert errMath < 3e-5'f32, &"N={N} T={T} carry={carry} math error={errMath}"
      doAssert errSerial < 3e-5'f32, &"N={N} T={T} carry={carry} serial error={errSerial}"
    echo &"PASS lane scan N={N} T={T}"
