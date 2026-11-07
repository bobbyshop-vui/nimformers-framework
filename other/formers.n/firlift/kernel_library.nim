## Micro-kernel library used by the trace lifter.
## Bodies are session_vm emit* lines; `firlift` prepends the `kernel` header.
## The attention body is the generated GQA decode kernel for
## hidden=2048, heads=32, kv_heads=4, head_dim=64, positions=2048.

const
  RmsnormBody* = """loadconstf32 1 0
uconst 0 0
loadconstf32 2 in0.size
forbegin 2
uloadf32 3 0 @in0
mul 3 3 3
add 1 1 3
uaddk 0 0 1
forend
mulconst 1 1 1/in0.size
addconst 1 1 eps
rsqrt 1 1
uconst 0 0
loadconstf32 2 in0.size
forbegin 2
uloadf32 3 0 @in0
mul 3 3 1
uloadf32 4 0 @in1
mul 3 3 4
ustoref32 3 0 @out0
uaddk 0 0 1
forend"""

  AddBody* = """rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
add 1 1 2
ustoref32 1 0 @out0"""

  SiluMulBody* = """rtou 0 0
uloadf32 1 0 @in1
sigmoid 2 1
mul 1 1 2
uloadf32 2 0 @in0
mul 1 1 2
ustoref32 1 0 @out0"""

  SigmoidBody* = """rtou 0 0
uloadf32 1 0 @in0
sigmoid 2 1
ustoref32 2 0 @out0"""

  MulBody* = """rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
mul 3 1 2
ustoref32 3 0 @out0"""

  AttentionBody* = """rtou 0 0
uconst 1 0
uloadf32 1 1 @in3
rtou 2 1
addconst 2 1 1
umulk 3 2 64
umulk 4 0 131072
uadd 3 3 4
umulk 5 0 64
uaddk 6 5 0
uaddk 7 3 0
loadconstf32 3 0
loadconstf32 4 32
forbegin 4
mulconst 5 3 -0.28782313662425574
exp 5 5
mul 5 5 1
loadconstf32 7 1
mulconst 6 5 1
loadconstf32 10 0
cmplt 9 6 10
ifbegin 9
neg 6 6
loadconstf32 7 -1
ifend
mulconst 8 6 0.15915494309189535
addconst 8 8 0.5
rtou 8 8
utor 8 8
mulconst 8 8 6.283185307179586
sub 6 6 8
mul 11 6 6
mulconst 12 6 1
mulconst 14 6 1
loadconstf32 13 1
loadconstf32 15 1
mul 12 12 11
mulconst 12 12 -0.16666666666666666
add 14 14 12
mul 13 13 11
mulconst 13 13 -0.5
add 15 15 13
mul 12 12 11
mulconst 12 12 -0.05
add 14 14 12
mul 13 13 11
mulconst 13 13 -0.08333333333333333
add 15 15 13
mul 12 12 11
mulconst 12 12 -0.023809523809523808
add 14 14 12
mul 13 13 11
mulconst 13 13 -0.03333333333333333
add 15 15 13
mul 12 12 11
mulconst 12 12 -0.013888888888888888
add 14 14 12
mul 13 13 11
mulconst 13 13 -0.017857142857142856
add 15 15 13
mul 12 12 11
mulconst 12 12 -0.00909090909090909
add 14 14 12
mul 13 13 11
mulconst 13 13 -0.011111111111111112
add 15 15 13
mul 12 12 11
mulconst 12 12 -0.00641025641025641
add 14 14 12
mul 13 13 11
mulconst 13 13 -0.007575757575757576
add 15 15 13
mul 12 12 11
mulconst 12 12 -0.004761904761904762
add 14 14 12
mul 13 13 11
mulconst 13 13 -0.005494505494505495
add 15 15 13
mul 12 12 11
mulconst 12 12 -0.003676470588235294
add 14 14 12
mul 13 13 11
mulconst 13 13 -0.004166666666666667
add 15 15 13
mul 12 12 11
mulconst 12 12 -0.0029239766081871343
add 14 14 12
mul 13 13 11
mulconst 13 13 -0.0032679738562091504
add 15 15 13
mul 12 12 11
mulconst 12 12 -0.002380952380952381
add 14 14 12
mul 13 13 11
mulconst 13 13 -0.002631578947368421
add 15 15 13
mul 12 12 11
mulconst 12 12 -0.001976284584980237
add 14 14 12
mul 13 13 11
mulconst 13 13 -0.0021645021645021645
add 15 15 13
mul 14 14 7
uloadf32 5 6 @in1
uloadf32 6 6 @in1+32
mul 7 5 15
mul 8 6 14
sub 7 7 8
ustoref32 7 7 @in4
mul 7 6 15
mul 8 5 14
add 7 7 8
ustoref32 7 7 @in4+32
addconst 3 3 1
uaddk 6 6 1
uaddk 7 7 1
forend
umulk 3 2 64
umulk 4 0 131072
uadd 3 3 4
umulk 5 0 64
loadconstf32 3 64
forbegin 3
uloadf32 4 5 @in2
ustoref32 4 3 @in5
uaddk 5 5 1
uaddk 3 3 1
forend
umulk 2 0 131072
umulk 3 0 131072
umulk 5 0 8
umulk 6 5 64
umulk 7 5 2048
umulk 8 5 64
umulk 9 5 64
loadconstf32 3 8
forbegin 3
uaddk 4 6 0
uaddk 5 9 0
loadconstf32 4 0
loadconstf32 5 32
forbegin 5
mulconst 6 4 -0.28782313662425574
exp 6 6
mul 6 6 1
loadconstf32 8 1
mulconst 7 6 1
loadconstf32 11 0
cmplt 10 7 11
ifbegin 10
neg 7 7
loadconstf32 8 -1
ifend
mulconst 9 7 0.15915494309189535
addconst 9 9 0.5
rtou 10 9
utor 9 10
mulconst 9 9 6.283185307179586
sub 7 7 9
mul 12 7 7
mulconst 13 7 1
mulconst 15 7 1
loadconstf32 14 1
loadconstf32 16 1
mul 13 13 12
mulconst 13 13 -0.16666666666666666
add 15 15 13
mul 14 14 12
mulconst 14 14 -0.5
add 16 16 14
mul 13 13 12
mulconst 13 13 -0.05
add 15 15 13
mul 14 14 12
mulconst 14 14 -0.08333333333333333
add 16 16 14
mul 13 13 12
mulconst 13 13 -0.023809523809523808
add 15 15 13
mul 14 14 12
mulconst 14 14 -0.03333333333333333
add 16 16 14
mul 13 13 12
mulconst 13 13 -0.013888888888888888
add 15 15 13
mul 14 14 12
mulconst 14 14 -0.017857142857142856
add 16 16 14
mul 13 13 12
mulconst 13 13 -0.00909090909090909
add 15 15 13
mul 14 14 12
mulconst 14 14 -0.011111111111111112
add 16 16 14
mul 13 13 12
mulconst 13 13 -0.00641025641025641
add 15 15 13
mul 14 14 12
mulconst 14 14 -0.007575757575757576
add 16 16 14
mul 13 13 12
mulconst 13 13 -0.004761904761904762
add 15 15 13
mul 14 14 12
mulconst 14 14 -0.005494505494505495
add 16 16 14
mul 13 13 12
mulconst 13 13 -0.003676470588235294
add 15 15 13
mul 14 14 12
mulconst 14 14 -0.004166666666666667
add 16 16 14
mul 13 13 12
mulconst 13 13 -0.0029239766081871343
add 15 15 13
mul 14 14 12
mulconst 14 14 -0.0032679738562091504
add 16 16 14
mul 13 13 12
mulconst 13 13 -0.002380952380952381
add 15 15 13
mul 14 14 12
mulconst 14 14 -0.002631578947368421
add 16 16 14
mul 13 13 12
mulconst 13 13 -0.001976284584980237
add 15 15 13
mul 14 14 12
mulconst 14 14 -0.0021645021645021645
add 16 16 14
mul 15 15 8
uloadf32 6 4 @in0
uloadf32 7 4 @in0+32
mul 8 6 16
mul 9 7 15
sub 8 8 9
ustoref32 8 5 @in6+65536
mul 8 7 16
mul 9 6 15
add 8 8 9
ustoref32 8 5 @in6+65568
addconst 4 4 1
uaddk 4 4 1
uaddk 5 5 1
forend
loadconstf32 4 -3e+38
uaddk 4 2 0
uaddk 5 7 0
forbegin 2
uaddk 10 9 0
uaddk 11 4 0
loadconstf32 5 0
loadconstf32 6 64
forbegin 6
uloadf32 7 10 @in6+65536
uloadf32 8 11 @in4
mul 7 7 8
add 5 5 7
uaddk 10 10 1
uaddk 11 11 1
forend
mulconst 5 5 1/sqrt(64)
ustoref32 5 5 @in6
cmplt 6 4 5
ifbegin 6
mulconst 4 5 1
ifend
uaddk 4 4 64
uaddk 5 5 1
forend
loadconstf32 5 0
uaddk 5 7 0
forbegin 2
uloadf32 6 5 @in6
sub 6 6 4
exp 6 6
ustoref32 6 5 @in6
add 5 5 6
uaddk 5 5 1
forend
uaddk 4 3 0
uaddk 10 8 0
loadconstf32 4 64
forbegin 4
loadconstf32 6 0
uaddk 11 4 0
uaddk 5 7 0
forbegin 2
uloadf32 7 5 @in6
uloadf32 8 11 @in5
mul 7 7 8
add 6 6 7
uaddk 11 11 64
uaddk 5 5 1
forend
div 6 6 5
ustoref32 6 10 @out0
uaddk 4 4 1
uaddk 10 10 1
forend
uaddk 6 6 64
uaddk 7 7 2048
uaddk 8 8 64
uaddk 9 9 64
forend"""

  AddRmsnormBody* = """loadconstf32 1 0
uconst 0 0
loadconstf32 2 in0.size
forbegin 2
uloadf32 3 0 @in0
uloadf32 4 0 @in1
add 3 3 4
ustoref32 3 0 @out1
mul 3 3 3
add 1 1 3
uaddk 0 0 1
forend
mulconst 1 1 1/in0.size
addconst 1 1 eps
rsqrt 1 1
uconst 0 0
loadconstf32 2 in0.size
forbegin 2
uloadf32 3 0 @out1
mul 3 3 1
uloadf32 4 0 @in2
mul 3 3 4
ustoref32 3 0 @out0
uaddk 0 0 1
forend"""

  ## Parallel RMSNorm: one 256-lane threadgroup per row. Each lane sums
  ## ceil(cols/256) squares, a threadgroup tree reduces the partials, then the
  ## same lanes scale and write their elements. This mirrors the inference
  ## path's GPU norm instead of a single serial lane.
  RmsnormParBody* = """
rtou 0 0
uconst 1 @in0
loadconstf32 1 0
loadconstf32 4 in0.size
loadconstf32 2 ((in0.size+255)/256)
uaddk 3 0 0
forbegin 2
utor 6 3
cmplt 7 6 4
ifbegin 7
uaddk 8 1 0
uadd 8 8 3
uloadf32 9 8 0
mul 9 9 9
add 1 1 9
ifend
uaddk 3 3 256
forend
loadlid 8
tgstoref32 1 8 0
barriertg
loadconstf32 12 128
cmplt 13 8 12
ifbegin 13
addconst 14 8 128
tgloadf32 10 14 0
tgloadf32 11 8 0
add 11 11 10
tgstoref32 11 8 0
ifend
barriertg
loadconstf32 12 64
cmplt 13 8 12
ifbegin 13
addconst 14 8 64
tgloadf32 10 14 0
tgloadf32 11 8 0
add 11 11 10
tgstoref32 11 8 0
ifend
barriertg
loadconstf32 12 32
cmplt 13 8 12
ifbegin 13
addconst 14 8 32
tgloadf32 10 14 0
tgloadf32 11 8 0
add 11 11 10
tgstoref32 11 8 0
ifend
barriertg
loadconstf32 12 16
cmplt 13 8 12
ifbegin 13
addconst 14 8 16
tgloadf32 10 14 0
tgloadf32 11 8 0
add 11 11 10
tgstoref32 11 8 0
ifend
barriertg
loadconstf32 12 8
cmplt 13 8 12
ifbegin 13
addconst 14 8 8
tgloadf32 10 14 0
tgloadf32 11 8 0
add 11 11 10
tgstoref32 11 8 0
ifend
barriertg
loadconstf32 12 4
cmplt 13 8 12
ifbegin 13
addconst 14 8 4
tgloadf32 10 14 0
tgloadf32 11 8 0
add 11 11 10
tgstoref32 11 8 0
ifend
barriertg
loadconstf32 12 2
cmplt 13 8 12
ifbegin 13
addconst 14 8 2
tgloadf32 10 14 0
tgloadf32 11 8 0
add 11 11 10
tgstoref32 11 8 0
ifend
barriertg
loadconstf32 12 1
cmplt 13 8 12
ifbegin 13
addconst 14 8 1
tgloadf32 10 14 0
tgloadf32 11 8 0
add 11 11 10
tgstoref32 11 8 0
ifend
barriertg
loadconstf32 8 0
tgloadf32 7 8 0
mulconst 7 7 1/in0.size
addconst 7 7 eps
rsqrt 7 7
uconst 4 @in1
uconst 5 @out0
rtou 2 0
loadconstf32 2 ((in0.size+255)/256)
forbegin 2
utor 6 2
cmplt 13 6 4
ifbegin 13
uaddk 8 1 0
uadd 8 8 2
uloadf32 9 8 0
mul 9 9 7
uaddk 10 4 0
uadd 10 10 2
uloadf32 11 10 0
mul 9 9 11
uaddk 10 5 0
uadd 10 10 2
ustoref32 9 10 0
ifend
uaddk 2 2 256
forend"""
