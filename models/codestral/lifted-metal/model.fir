fir 1
model hf
config config.json
runtime graph
layout script
script_abi 1
epsilon layer_norm_epsilon
require feature weights_bf16
sym hidden = hidden_size
sym layers = num_hidden_layers
sym positions = max_position_embeddings
sym vocab = vocab_size
buf token f32 1 input:token
buf b23 f32 1208960
init b23 trace.data 0 4835840 sample1-5146a04ed4a1fd36
weight b24 raw backbone.layers.0.norm.weight tensor [4096]
init b24 checkpoint 595199123 16384 sample1-1ea4692f999e3ab5
buf b27 f32 4096
buf b41 f32 1208960
init b41 trace.data 4835840 4835840 sample1-5146a04ed4a1fd36
weight b42 raw backbone.layers.1.norm.weight tensor [4096]
init b42 checkpoint 653543614 16384 sample1-c9e9dec0ec0f654b
buf b45 f32 4096
buf b59 f32 1208960
init b59 trace.data 9671680 4835840 sample1-5146a04ed4a1fd36
weight b60 raw backbone.layers.2.norm.weight tensor [4096]
init b60 checkpoint 1295333105 16384 sample1-4898aa8bc9faef59
buf b63 f32 4096
buf b77 f32 1208960
init b77 trace.data 14507520 4835840 sample1-5146a04ed4a1fd36
weight b78 raw backbone.layers.3.norm.weight tensor [4096]
init b78 checkpoint 1412121230 16384 sample1-9922475d8ca726cc
buf b81 f32 4096
buf b95 f32 1208960
init b95 trace.data 19343360 4835840 sample1-5146a04ed4a1fd36
weight b96 raw backbone.layers.4.norm.weight tensor [4096]
init b96 checkpoint 1470465721 16384 sample1-3692437bb5e53197
buf b99 f32 4096
buf b113 f32 1208960
init b113 trace.data 24179200 4835840 sample1-5146a04ed4a1fd36
weight b114 raw backbone.layers.5.norm.weight tensor [4096]
init b114 checkpoint 1528810212 16384 sample1-7c74ff01223a87b0
buf b117 f32 4096
buf b131 f32 1208960
init b131 trace.data 29015040 4835840 sample1-5146a04ed4a1fd36
weight b132 raw backbone.layers.6.norm.weight tensor [4096]
init b132 checkpoint 1587154703 16384 sample1-b51523b054524234
buf b135 f32 4096
buf b149 f32 1208960
init b149 trace.data 33850880 4835840 sample1-5146a04ed4a1fd36
weight b150 raw backbone.layers.7.norm.weight tensor [4096]
init b150 checkpoint 1645499194 16384 sample1-21ed689eb83e69e2
buf b153 f32 4096
weight b154 raw newBlock::block7.attn.qkvProj.weight tensor [50331648]
init b154 jamba_parallel_lora.bin.raw 35829894 201326592 sample1-f7717df7032b3554
weight b155 raw capture.sample1-4faf6eecd4d86e85 tensor [12288]
init b155 trace.data 38686720 49152 sample1-4faf6eecd4d86e85
weight b156 raw newBlock::block7.attn.outProj.weight tensor [16777216]
init b156 jamba_parallel_lora.bin.raw 237205752 67108864 sample1-deffe74a292b7e13
weight b157 raw newBlock::block7.attn.outProj.bias tensor [4096]
init b157 jamba_parallel_lora.bin.raw 304314670 16384 sample1-280b7a9880826a87
buf b158 f32 151552
init b158 trace.data 38735872 606208 sample1-2a8f57f2d4439f04
weight b159 raw backbone.layers.7.norm.weight tensor [4096]
init b159 checkpoint 1645499194 16384 sample1-21ed689eb83e69e2
buf b164 f32 4096
weight b165 raw newBlock::block7.fuseGate.weight tensor [33554432]
init b165 jamba_parallel_lora.bin.raw 304331110 134217728 sample1-afcd4b69fade7c32
weight b166 raw capture.sample1-f87fcefcec085c05 tensor [4096]
init b166 trace.data 39342080 16384 sample1-f87fcefcec085c05
buf b167 f32 8192
buf b168 f32 4096
buf b169 f32 4096
buf b170 f32 4096
weight b171 raw capture.sample1-3569daa1a5c21185 tensor [4096]
init b171 trace.data 39358464 16384 sample1-3569daa1a5c21185
buf b185 f32 1208960
init b185 trace.data 39374848 4835840 sample1-5146a04ed4a1fd36
weight b186 raw backbone.layers.8.norm.weight tensor [4096]
init b186 checkpoint 1703843685 16384 sample1-3c657301509842f8
buf b189 f32 4096
buf b203 f32 1208960
init b203 trace.data 44210688 4835840 sample1-5146a04ed4a1fd36
weight b204 raw backbone.layers.9.norm.weight tensor [4096]
init b204 checkpoint 1762188176 16384 sample1-f29150a6542713a6
buf b207 f32 4096
buf b221 f32 1208960
init b221 trace.data 49046528 4835840 sample1-5146a04ed4a1fd36
weight b222 raw backbone.layers.10.norm.weight tensor [4096]
init b222 checkpoint 711888114 16384 sample1-4356b5aba83636b3
buf b225 f32 4096
buf b239 f32 1208960
init b239 trace.data 53882368 4835840 sample1-5146a04ed4a1fd36
weight b240 raw backbone.layers.11.norm.weight tensor [4096]
init b240 checkpoint 770232614 16384 sample1-b1a3e5e757318f0a
buf b243 f32 4096
buf b257 f32 1208960
init b257 trace.data 58718208 4835840 sample1-5146a04ed4a1fd36
weight b258 raw backbone.layers.12.norm.weight tensor [4096]
init b258 checkpoint 828577114 16384 sample1-489d27af6c3bada1
buf b261 f32 4096
buf b275 f32 1208960
init b275 trace.data 63554048 4835840 sample1-5146a04ed4a1fd36
weight b276 raw backbone.layers.13.norm.weight tensor [4096]
init b276 checkpoint 886921614 16384 sample1-7dc2513418a7d045
buf b279 f32 4096
buf b293 f32 1208960
init b293 trace.data 68389888 4835840 sample1-5146a04ed4a1fd36
weight b294 raw backbone.layers.14.norm.weight tensor [4096]
init b294 checkpoint 945266114 16384 sample1-f77524f3582d076e
buf b297 f32 4096
buf b311 f32 1208960
init b311 trace.data 73225728 4835840 sample1-5146a04ed4a1fd36
weight b312 raw backbone.layers.15.norm.weight tensor [4096]
init b312 checkpoint 1003610614 16384 sample1-3622d44dfb756dbc
buf b315 f32 4096
buf b329 f32 1208960
init b329 trace.data 78061568 4835840 sample1-5146a04ed4a1fd36
weight b330 raw backbone.layers.16.norm.weight tensor [4096]
init b330 checkpoint 1061955114 16384 sample1-ba780ecb97508d47
buf b333 f32 4096
buf b347 f32 1208960
init b347 trace.data 82897408 4835840 sample1-5146a04ed4a1fd36
weight b348 raw backbone.layers.17.norm.weight tensor [4096]
init b348 checkpoint 1120299614 16384 sample1-e0caf5e9db7bf88f
buf b351 f32 4096
buf b365 f32 1208960
init b365 trace.data 87733248 4835840 sample1-5146a04ed4a1fd36
weight b366 raw backbone.layers.18.norm.weight tensor [4096]
init b366 checkpoint 1178644114 16384 sample1-73a1f34b75f876c0
buf b369 f32 4096
buf b383 f32 1208960
init b383 trace.data 92569088 4835840 sample1-5146a04ed4a1fd36
weight b384 raw backbone.layers.19.norm.weight tensor [4096]
init b384 checkpoint 1236988614 16384 sample1-ecb3a25d3e717249
buf b387 f32 4096
buf b401 f32 1208960
init b401 trace.data 97404928 4835840 sample1-5146a04ed4a1fd36
weight b402 raw backbone.layers.20.norm.weight tensor [4096]
init b402 checkpoint 1353677605 16384 sample1-c63135d2e9814269
buf b405 f32 4096
buf b419 f32 1208960
init b419 trace.data 102240768 4835840 sample1-5146a04ed4a1fd36
weight b420 raw backbone.layers.21.norm.weight tensor [4096]
init b420 checkpoint 1353776739 16384 sample1-9de2c24468c38618
buf b423 f32 4096
buf b437 f32 1208960
init b437 trace.data 107076608 4835840 sample1-5146a04ed4a1fd36
weight b438 raw backbone.layers.22.norm.weight tensor [4096]
init b438 checkpoint 1878778042 16384 sample1-549260db3d11cd14
buf b441 f32 4096
buf b455 f32 1208960
init b455 trace.data 111912448 4835840 sample1-5146a04ed4a1fd36
weight b456 raw backbone.layers.23.norm.weight tensor [4096]
init b456 checkpoint 1937122542 16384 sample1-4e2847a37a66cd09
buf b459 f32 4096
weight b460 raw newBlock::block23.attn.qkvProj.weight tensor [50331648]
init b460 jamba_parallel_lora.bin.raw 438565333 201326592 sample1-97b83897b6c44568
weight b461 raw capture.sample1-4faf6eecd4d86e85 tensor [12288]
init b461 trace.data 38686720 49152 sample1-4faf6eecd4d86e85
weight b462 raw newBlock::block23.attn.outProj.weight tensor [16777216]
init b462 jamba_parallel_lora.bin.raw 639941193 67108864 sample1-7e19d49350c2a7ec
weight b463 raw newBlock::block23.attn.outProj.bias tensor [4096]
init b463 jamba_parallel_lora.bin.raw 707050112 16384 sample1-65edac365ee73987
buf b464 f32 151552
init b464 trace.data 38735872 606208 sample1-2a8f57f2d4439f04
weight b465 raw backbone.layers.23.norm.weight tensor [4096]
init b465 checkpoint 1937122542 16384 sample1-4e2847a37a66cd09
weight b470 raw newBlock::block23.fuseGate.weight tensor [33554432]
init b470 jamba_parallel_lora.bin.raw 707066553 134217728 sample1-1db1d39401f36067
weight b471 raw capture.sample1-f87fcefcec085c05 tensor [4096]
init b471 trace.data 39342080 16384 sample1-f87fcefcec085c05
buf b472 f32 8192
buf b473 f32 4096
buf b474 f32 4096
buf b475 f32 4096
weight b476 raw capture.sample1-3569daa1a5c21185 tensor [4096]
init b476 trace.data 39358464 16384 sample1-3569daa1a5c21185
buf b490 f32 1208960
init b490 trace.data 116748288 4835840 sample1-5146a04ed4a1fd36
weight b491 raw backbone.layers.24.norm.weight tensor [4096]
init b491 checkpoint 1995467042 16384 sample1-ac11daa1a926cb41
buf b494 f32 4096
buf b508 f32 1208960
init b508 trace.data 121584128 4835840 sample1-5146a04ed4a1fd36
weight b509 raw backbone.layers.25.norm.weight tensor [4096]
init b509 checkpoint 2053811542 16384 sample1-8972cb531ad9dd41
buf b512 f32 4096
buf b526 f32 1208960
init b526 trace.data 126419968 4835840 sample1-5146a04ed4a1fd36
weight b527 raw backbone.layers.26.norm.weight tensor [4096]
init b527 checkpoint 2112156042 16384 sample1-49c0a50002d8b0d7
buf b530 f32 4096
buf b544 f32 1208960
init b544 trace.data 131255808 4835840 sample1-5146a04ed4a1fd36
weight b545 raw backbone.layers.27.norm.weight tensor [4096]
init b545 checkpoint 2170500542 16384 sample1-55ad339dd5d1a9d0
buf b548 f32 4096
buf b562 f32 1208960
init b562 trace.data 136091648 4835840 sample1-5146a04ed4a1fd36
weight b563 raw backbone.layers.28.norm.weight tensor [4096]
init b563 checkpoint 2228845042 16384 sample1-3982671f9230fbe3
buf b566 f32 4096
buf b580 f32 1208960
init b580 trace.data 140927488 4835840 sample1-5146a04ed4a1fd36
weight b581 raw backbone.layers.29.norm.weight tensor [4096]
init b581 checkpoint 2287189542 16384 sample1-0e45cf5a4c56d350
buf b584 f32 4096
buf b598 f32 1208960
init b598 trace.data 145763328 4835840 sample1-5146a04ed4a1fd36
weight b599 raw backbone.layers.30.norm.weight tensor [4096]
init b599 checkpoint 2345534042 16384 sample1-c7880af0912f6453
buf b602 f32 4096
buf b616 f32 1208960
init b616 trace.data 150599168 4835840 sample1-5146a04ed4a1fd36
weight b617 raw backbone.layers.31.norm.weight tensor [4096]
init b617 checkpoint 2403878542 16384 sample1-248c866861965df9
buf b620 f32 4096
buf b634 f32 1208960
init b634 trace.data 155435008 4835840 sample1-5146a04ed4a1fd36
weight b635 raw backbone.layers.32.norm.weight tensor [4096]
init b635 checkpoint 2462223042 16384 sample1-a8bf6a3db4c9a33f
buf b638 f32 4096
buf b652 f32 1208960
init b652 trace.data 160270848 4835840 sample1-5146a04ed4a1fd36
weight b653 raw backbone.layers.33.norm.weight tensor [4096]
init b653 checkpoint 2520567542 16384 sample1-62ebfd0cf8388b7f
buf b656 f32 4096
buf b670 f32 1208960
init b670 trace.data 165106688 4835840 sample1-5146a04ed4a1fd36
weight b671 raw backbone.layers.34.norm.weight tensor [4096]
init b671 checkpoint 2578912042 16384 sample1-b27197e6576e901a
buf b674 f32 4096
buf b688 f32 1208960
init b688 trace.data 169942528 4835840 sample1-5146a04ed4a1fd36
weight b689 raw backbone.layers.35.norm.weight tensor [4096]
init b689 checkpoint 2637256542 16384 sample1-fb19f7b8254aa3b4
buf b692 f32 4096
buf b706 f32 1208960
init b706 trace.data 174778368 4835840 sample1-5146a04ed4a1fd36
weight b707 raw backbone.layers.36.norm.weight tensor [4096]
init b707 checkpoint 2695601042 16384 sample1-c8c162d645727e67
buf b710 f32 4096
buf b724 f32 1208960
init b724 trace.data 179614208 4835840 sample1-5146a04ed4a1fd36
weight b725 raw backbone.layers.37.norm.weight tensor [4096]
init b725 checkpoint 2753945542 16384 sample1-a08d421381aff837
buf b728 f32 4096
buf b742 f32 1208960
init b742 trace.data 184450048 4835840 sample1-5146a04ed4a1fd36
weight b743 raw backbone.layers.38.norm.weight tensor [4096]
init b743 checkpoint 2812290042 16384 sample1-0984060e976522ab
buf b746 f32 4096
buf b760 f32 1208960
init b760 trace.data 189285888 4835840 sample1-5146a04ed4a1fd36
weight b761 raw backbone.layers.39.norm.weight tensor [4096]
init b761 checkpoint 2870634542 16384 sample1-26e89c82f90a5263
buf b764 f32 4096
weight b765 raw newBlock::block39.attn.qkvProj.weight tensor [50331648]
init b765 jamba_parallel_lora.bin.raw 841300777 201326592 sample1-1ca8d8426578f6a8
weight b766 raw capture.sample1-4faf6eecd4d86e85 tensor [12288]
init b766 trace.data 38686720 49152 sample1-4faf6eecd4d86e85
weight b767 raw newBlock::block39.attn.outProj.weight tensor [16777216]
init b767 jamba_parallel_lora.bin.raw 1042676637 67108864 sample1-174c7c11cd5aeac2
weight b768 raw newBlock::block39.attn.outProj.bias tensor [4096]
init b768 jamba_parallel_lora.bin.raw 1109785556 16384 sample1-d5d993a0599f57d9
buf b769 f32 151552
init b769 trace.data 38735872 606208 sample1-2a8f57f2d4439f04
weight b770 raw backbone.layers.39.norm.weight tensor [4096]
init b770 checkpoint 2870634542 16384 sample1-26e89c82f90a5263
weight b775 raw newBlock::block39.fuseGate.weight tensor [33554432]
init b775 jamba_parallel_lora.bin.raw 1109801997 134217728 sample1-91d6c6248f7b425c
weight b776 raw capture.sample1-f87fcefcec085c05 tensor [4096]
init b776 trace.data 39342080 16384 sample1-f87fcefcec085c05
buf b777 f32 8192
buf b778 f32 4096
buf b779 f32 4096
buf b780 f32 4096
weight b781 raw capture.sample1-3569daa1a5c21185 tensor [4096]
init b781 trace.data 39358464 16384 sample1-3569daa1a5c21185
buf b795 f32 1208960
init b795 trace.data 194121728 4835840 sample1-5146a04ed4a1fd36
weight b796 raw backbone.layers.40.norm.weight tensor [4096]
init b796 checkpoint 2928979042 16384 sample1-3a59f634594e1b05
buf b799 f32 4096
buf b813 f32 1208960
init b813 trace.data 198957568 4835840 sample1-5146a04ed4a1fd36
weight b814 raw backbone.layers.41.norm.weight tensor [4096]
init b814 checkpoint 2987323542 16384 sample1-5266e5d590db9454
buf b817 f32 4096
buf b831 f32 1208960
init b831 trace.data 203793408 4835840 sample1-5146a04ed4a1fd36
weight b832 raw backbone.layers.42.norm.weight tensor [4096]
init b832 checkpoint 3045668042 16384 sample1-238c7cf69dadb8b9
buf b835 f32 4096
buf b849 f32 1208960
init b849 trace.data 208629248 4835840 sample1-5146a04ed4a1fd36
weight b850 raw backbone.layers.43.norm.weight tensor [4096]
init b850 checkpoint 3086186665 16384 sample1-da8d67b35829feff
buf b853 f32 4096
buf b867 f32 1208960
init b867 trace.data 213465088 4835840 sample1-5146a04ed4a1fd36
weight b868 raw backbone.layers.44.norm.weight tensor [4096]
init b868 checkpoint 3162357042 16384 sample1-cbc4a60eb850a04d
buf b871 f32 4096
buf b885 f32 1208960
init b885 trace.data 218300928 4835840 sample1-5146a04ed4a1fd36
weight b886 raw backbone.layers.45.norm.weight tensor [4096]
init b886 checkpoint 3220701542 16384 sample1-269d85483893256d
buf b889 f32 4096
buf b903 f32 1208960
init b903 trace.data 223136768 4835840 sample1-5146a04ed4a1fd36
weight b904 raw backbone.layers.46.norm.weight tensor [4096]
init b904 checkpoint 3279046042 16384 sample1-f3c2a72c9a8b0bf7
buf b907 f32 4096
buf b921 f32 1208960
init b921 trace.data 227972608 4835840 sample1-5146a04ed4a1fd36
weight b922 raw backbone.layers.47.norm.weight tensor [4096]
init b922 checkpoint 3337390542 16384 sample1-4ec7fd015867ee31
buf b925 f32 4096
buf b939 f32 1208960
init b939 trace.data 232808448 4835840 sample1-5146a04ed4a1fd36
weight b940 raw backbone.layers.48.norm.weight tensor [4096]
init b940 checkpoint 3395735042 16384 sample1-a1dfbb0d0fbdcc76
buf b943 f32 4096
buf b957 f32 1208960
init b957 trace.data 237644288 4835840 sample1-5146a04ed4a1fd36
weight b958 raw backbone.layers.49.norm.weight tensor [4096]
init b958 checkpoint 3454079542 16384 sample1-bf709b7575086c36
buf b961 f32 4096
buf b975 f32 1208960
init b975 trace.data 242480128 4835840 sample1-5146a04ed4a1fd36
weight b976 raw backbone.layers.50.norm.weight tensor [4096]
init b976 checkpoint 3512424042 16384 sample1-70a38a768c8b8530
buf b979 f32 4096
buf b993 f32 1208960
init b993 trace.data 247315968 4835840 sample1-5146a04ed4a1fd36
weight b994 raw backbone.layers.51.norm.weight tensor [4096]
init b994 checkpoint 3570768542 16384 sample1-0cdce1b44fa683cb
buf b997 f32 4096
buf b1011 f32 1208960
init b1011 trace.data 252151808 4835840 sample1-5146a04ed4a1fd36
weight b1012 raw backbone.layers.52.norm.weight tensor [4096]
init b1012 checkpoint 3629113042 16384 sample1-446b5ea63077b8ef
buf b1015 f32 4096
buf b1029 f32 1208960
init b1029 trace.data 256987648 4835840 sample1-5146a04ed4a1fd36
weight b1030 raw backbone.layers.53.norm.weight tensor [4096]
init b1030 checkpoint 3687457542 16384 sample1-e2bf23969262256d
buf b1033 f32 4096
buf b1047 f32 1208960
init b1047 trace.data 261823488 4835840 sample1-5146a04ed4a1fd36
weight b1048 raw backbone.layers.54.norm.weight tensor [4096]
init b1048 checkpoint 3745802042 16384 sample1-8e4b5deedd1c6b5e
buf b1051 f32 4096
buf b1065 f32 1208960
init b1065 trace.data 266659328 4835840 sample1-5146a04ed4a1fd36
weight b1066 raw backbone.layers.55.norm.weight tensor [4096]
init b1066 checkpoint 3804146542 16384 sample1-ade7e84c4b9078ee
buf b1069 f32 4096
weight b1070 raw newBlock::block55.attn.qkvProj.weight tensor [50331648]
init b1070 jamba_parallel_lora.bin.raw 1244036221 201326592 sample1-bb75ec2e77b2fa22
weight b1071 raw capture.sample1-4faf6eecd4d86e85 tensor [12288]
init b1071 trace.data 38686720 49152 sample1-4faf6eecd4d86e85
weight b1072 raw newBlock::block55.attn.outProj.weight tensor [16777216]
init b1072 jamba_parallel_lora.bin.raw 1445412081 67108864 sample1-f8d7997763d7ef76
weight b1073 raw newBlock::block55.attn.outProj.bias tensor [4096]
init b1073 jamba_parallel_lora.bin.raw 1512521000 16384 sample1-2dbae0fbd0eaf2d2
buf b1074 f32 151552
init b1074 trace.data 38735872 606208 sample1-2a8f57f2d4439f04
weight b1075 raw backbone.layers.55.norm.weight tensor [4096]
init b1075 checkpoint 3804146542 16384 sample1-ade7e84c4b9078ee
weight b1080 raw newBlock::block55.fuseGate.weight tensor [33554432]
init b1080 jamba_parallel_lora.bin.raw 1512537441 134217728 sample1-75d1232413dd3c38
weight b1081 raw capture.sample1-f87fcefcec085c05 tensor [4096]
init b1081 trace.data 39342080 16384 sample1-f87fcefcec085c05
buf b1082 f32 8192
buf b1083 f32 4096
buf b1084 f32 4096
buf b1085 f32 4096
weight b1086 raw capture.sample1-3569daa1a5c21185 tensor [4096]
init b1086 trace.data 39358464 16384 sample1-3569daa1a5c21185
buf b1100 f32 1208960
init b1100 trace.data 271495168 4835840 sample1-5146a04ed4a1fd36
weight b1101 raw backbone.layers.56.norm.weight tensor [4096]
init b1101 checkpoint 3862491042 16384 sample1-744479cc98da1701
buf b1104 f32 4096
buf b1118 f32 1208960
init b1118 trace.data 276331008 4835840 sample1-5146a04ed4a1fd36
weight b1119 raw backbone.layers.57.norm.weight tensor [4096]
init b1119 checkpoint 3920835542 16384 sample1-637f8752ec07d6b8
buf b1122 f32 4096
buf b1136 f32 1208960
init b1136 trace.data 281166848 4835840 sample1-5146a04ed4a1fd36
weight b1137 raw backbone.layers.58.norm.weight tensor [4096]
init b1137 checkpoint 3979180042 16384 sample1-5edda3ad593d7c35
buf b1140 f32 4096
buf b1154 f32 1208960
init b1154 trace.data 286002688 4835840 sample1-5146a04ed4a1fd36
weight b1155 raw backbone.layers.59.norm.weight tensor [4096]
init b1155 checkpoint 4037524542 16384 sample1-2c49f75a6461d4e5
buf b1158 f32 4096
buf b1172 f32 1208960
init b1172 trace.data 290838528 4835840 sample1-5146a04ed4a1fd36
weight b1173 raw backbone.layers.60.norm.weight tensor [4096]
init b1173 checkpoint 4095869042 16384 sample1-561685d0dd3225ce
buf b1176 f32 4096
buf b1190 f32 1208960
init b1190 trace.data 295674368 4835840 sample1-5146a04ed4a1fd36
weight b1191 raw backbone.layers.61.norm.weight tensor [4096]
init b1191 checkpoint 4154213542 16384 sample1-1f6f5ffd285a1b39
buf b1194 f32 4096
buf b1208 f32 1208960
init b1208 trace.data 300510208 4835840 sample1-5146a04ed4a1fd36
weight b1209 raw backbone.layers.62.norm.weight tensor [4096]
init b1209 checkpoint 4212558042 16384 sample1-de4f60dc9cd1e48f
buf b1212 f32 4096
buf b1226 f32 1208960
init b1226 trace.data 305346048 4835840 sample1-5146a04ed4a1fd36
weight b1227 raw backbone.layers.63.norm.weight tensor [4096]
init b1227 checkpoint 4270902542 16384 sample1-360e52f5a4437211
buf b1230 f32 4096
buf b1231 f32 4096
init b1231 trace.data 310181888 16384 sample1-e5fa59de89855c05
buf b1232 f32 4096
buf b1233 f32 4096
buf b1234 f32 32768 output:logits
weight b1235 raw backbone.norm_f.weight tensor [4096]
init b1235 checkpoint 4270918989 16384 sample1-c4a85c6f54efce7b
weight b1237 raw lm_head.weight tensor [17825792]
init b1237 checkpoint 4270935432 71303168 sample1-275b6251c89ac03d
weight b1238 raw capture.sample1-9e95129ac7a570f9 tensor [272]
init b1238 trace.data 310198272 1088 sample1-9e95129ac7a570f9
weight b1247 raw capture.sample1-224b2d83a22863e8 tensor [10096640]
init b1247 trace.data 310199360 40386560 sample1-224b2d83a22863e8
weight b1248 raw capture.sample1-c9b348baf68b6459 tensor [272]
init b1248 trace.data 350585920 1088 sample1-c9b348baf68b6459
weight b1251 raw capture.sample1-2a5794516c1322a2 tensor [4456448]
init b1251 trace.data 350587008 17825792 sample1-2a5794516c1322a2
weight b1252 raw capture.sample1-ae1faf7d0d2effa9 tensor [272]
init b1252 trace.data 368412800 1088 sample1-ae1faf7d0d2effa9
weight b1255 raw capture.sample1-99216da49352704f tensor [10096640]
init b1255 trace.data 368413888 40386560 sample1-99216da49352704f
weight b1256 raw capture.sample1-0e9d0df481637769 tensor [272]
init b1256 trace.data 408800448 1088 sample1-0e9d0df481637769
weight b1259 raw capture.sample1-97560412fbdbe42c tensor [4456448]
init b1259 trace.data 408801536 17825792 sample1-97560412fbdbe42c
weight b1260 raw capture.sample1-c6b7d76c927913b9 tensor [272]
init b1260 trace.data 426627328 1088 sample1-c6b7d76c927913b9
weight b1263 raw capture.sample1-407e6e287f44c70b tensor [10096640]
init b1263 trace.data 426628416 40386560 sample1-407e6e287f44c70b
weight b1264 raw capture.sample1-eb4576131e7f30d9 tensor [272]
init b1264 trace.data 467014976 1088 sample1-eb4576131e7f30d9
weight b1267 raw capture.sample1-115a2b8d71ada0b3 tensor [4456448]
init b1267 trace.data 467016064 17825792 sample1-115a2b8d71ada0b3
weight b1268 raw capture.sample1-7e40aea90cee3289 tensor [272]
init b1268 trace.data 484841856 1088 sample1-7e40aea90cee3289
weight b1271 raw capture.sample1-ac08b0a1f7bd209e tensor [10096640]
init b1271 trace.data 484842944 40386560 sample1-ac08b0a1f7bd209e
weight b1272 raw capture.sample1-c9146b27de7e8b95 tensor [272]
init b1272 trace.data 525229504 1088 sample1-c9146b27de7e8b95
weight b1275 raw capture.sample1-73e81d1713577740 tensor [4456448]
init b1275 trace.data 525230592 17825792 sample1-73e81d1713577740
weight b1276 raw capture.sample1-656920d909ced0f1 tensor [272]
init b1276 trace.data 543056384 1088 sample1-656920d909ced0f1
weight b1279 raw capture.sample1-00868223a6739680 tensor [10096640]
init b1279 trace.data 543057472 40386560 sample1-00868223a6739680
weight b1280 raw capture.sample1-7d9bb82b34e3c15d tensor [272]
init b1280 trace.data 583444032 1088 sample1-7d9bb82b34e3c15d
weight b1283 raw capture.sample1-3d84a9eda44235cd tensor [4456448]
init b1283 trace.data 583445120 17825792 sample1-3d84a9eda44235cd
weight b1284 raw capture.sample1-660e22561c5aaf81 tensor [272]
init b1284 trace.data 601270912 1088 sample1-660e22561c5aaf81
weight b1287 raw capture.sample1-13dbeeabf84e32db tensor [10096640]
init b1287 trace.data 601272000 40386560 sample1-13dbeeabf84e32db
weight b1288 raw capture.sample1-6046a93780b95cfd tensor [272]
init b1288 trace.data 641658560 1088 sample1-6046a93780b95cfd
weight b1291 raw capture.sample1-72a520eae5c2219f tensor [4456448]
init b1291 trace.data 641659648 17825792 sample1-72a520eae5c2219f
weight b1292 raw capture.sample1-dfc7b72dfa3ec1d9 tensor [272]
init b1292 trace.data 659485440 1088 sample1-dfc7b72dfa3ec1d9
weight b1295 raw capture.sample1-028667d3b6113730 tensor [10096640]
init b1295 trace.data 659486528 40386560 sample1-028667d3b6113730
weight b1296 raw capture.sample1-56013fd296fa772d tensor [272]
init b1296 trace.data 699873088 1088 sample1-56013fd296fa772d
weight b1299 raw capture.sample1-1d73a4b02857fa49 tensor [4456448]
init b1299 trace.data 699874176 17825792 sample1-1d73a4b02857fa49
weight b1300 raw capture.sample1-0fea4dda603829cd tensor [272]
init b1300 trace.data 717699968 1088 sample1-0fea4dda603829cd
weight b1303 raw capture.sample1-1f86f68da69c728d tensor [10096640]
init b1303 trace.data 717701056 40386560 sample1-1f86f68da69c728d
weight b1304 raw capture.sample1-e71795e990788391 tensor [272]
init b1304 trace.data 758087616 1088 sample1-e71795e990788391
weight b1307 raw capture.sample1-17c47c2512519f9a tensor [4456448]
init b1307 trace.data 758088704 17825792 sample1-17c47c2512519f9a
weight b1308 raw capture.sample1-c2933bb8a7687919 tensor [272]
init b1308 trace.data 775914496 1088 sample1-c2933bb8a7687919
weight b1311 raw capture.sample1-4f140884d1481b8f tensor [10096640]
init b1311 trace.data 775915584 40386560 sample1-4f140884d1481b8f
weight b1312 raw capture.sample1-e36e1175d9d64421 tensor [272]
init b1312 trace.data 816302144 1088 sample1-e36e1175d9d64421
weight b1315 raw capture.sample1-f8279c3ac43518b7 tensor [4456448]
init b1315 trace.data 816303232 17825792 sample1-f8279c3ac43518b7
weight b1316 raw capture.sample1-f45d2ed25d1bac29 tensor [272]
init b1316 trace.data 834129024 1088 sample1-f45d2ed25d1bac29
weight b1319 raw capture.sample1-b591240cc5dc1f4b tensor [10096640]
init b1319 trace.data 834130112 40386560 sample1-b591240cc5dc1f4b
weight b1320 raw capture.sample1-9a4e3e261f198831 tensor [272]
init b1320 trace.data 874516672 1088 sample1-9a4e3e261f198831
weight b1323 raw capture.sample1-0490c4acf7dd4924 tensor [4456448]
init b1323 trace.data 874517760 17825792 sample1-0490c4acf7dd4924
weight b1324 raw capture.sample1-52151f751275c381 tensor [272]
init b1324 trace.data 892343552 1088 sample1-52151f751275c381
weight b1327 raw capture.sample1-7f0d63dd5864e337 tensor [10096640]
init b1327 trace.data 892344640 40386560 sample1-7f0d63dd5864e337
weight b1328 raw capture.sample1-3d690f95499e20d1 tensor [272]
init b1328 trace.data 932731200 1088 sample1-3d690f95499e20d1
weight b1331 raw capture.sample1-a9c1ed9ce9fe30ff tensor [4456448]
init b1331 trace.data 932732288 17825792 sample1-a9c1ed9ce9fe30ff
weight b1332 raw capture.sample1-872652dece9144d1 tensor [272]
init b1332 trace.data 950558080 1088 sample1-872652dece9144d1
weight b1335 raw capture.sample1-795555553f593822 tensor [10096640]
init b1335 trace.data 950559168 40386560 sample1-795555553f593822
weight b1336 raw capture.sample1-6e00f5dc3bdbb2a9 tensor [272]
init b1336 trace.data 990945728 1088 sample1-6e00f5dc3bdbb2a9
weight b1339 raw capture.sample1-b85e35a7d0407406 tensor [4456448]
init b1339 trace.data 990946816 17825792 sample1-b85e35a7d0407406
weight b1340 raw capture.sample1-39c5bab5cfc13855 tensor [272]
init b1340 trace.data 1008772608 1088 sample1-39c5bab5cfc13855
weight b1343 raw capture.sample1-c0dbe9ebaf58b702 tensor [10096640]
init b1343 trace.data 1008773696 40386560 sample1-c0dbe9ebaf58b702
weight b1344 raw capture.sample1-1835d854bc69f109 tensor [272]
init b1344 trace.data 1049160256 1088 sample1-1835d854bc69f109
weight b1347 raw capture.sample1-c72195908611a3ad tensor [4456448]
init b1347 trace.data 1049161344 17825792 sample1-c72195908611a3ad
weight b1348 raw capture.sample1-7d6268ccc4d63aa5 tensor [272]
init b1348 trace.data 1066987136 1088 sample1-7d6268ccc4d63aa5
weight b1351 raw capture.sample1-f5597608dfb709e9 tensor [10096640]
init b1351 trace.data 1066988224 40386560 sample1-f5597608dfb709e9
weight b1352 raw capture.sample1-a0e14b75dd2dbc31 tensor [272]
init b1352 trace.data 1107374784 1088 sample1-a0e14b75dd2dbc31
weight b1355 raw capture.sample1-a8d3a0e1de0301fa tensor [4456448]
init b1355 trace.data 1107375872 17825792 sample1-a8d3a0e1de0301fa
weight b1356 raw capture.sample1-5f776808c7b06d51 tensor [272]
init b1356 trace.data 1125201664 1088 sample1-5f776808c7b06d51
weight b1359 raw capture.sample1-71205ad8200e5767 tensor [10096640]
init b1359 trace.data 1125202752 40386560 sample1-71205ad8200e5767
weight b1360 raw capture.sample1-4a2346c8b6bb8fa5 tensor [272]
init b1360 trace.data 1165589312 1088 sample1-4a2346c8b6bb8fa5
weight b1363 raw capture.sample1-114ddae6b1dfb85f tensor [4456448]
init b1363 trace.data 1165590400 17825792 sample1-114ddae6b1dfb85f
weight b1364 raw capture.sample1-504d2fcc170b0659 tensor [272]
init b1364 trace.data 1183416192 1088 sample1-504d2fcc170b0659
weight b1367 raw capture.sample1-a0a5e236b9a7f6c5 tensor [10096640]
init b1367 trace.data 1183417280 40386560 sample1-a0a5e236b9a7f6c5
weight b1368 raw capture.sample1-62b3222d0f0a76b1 tensor [272]
init b1368 trace.data 1223803840 1088 sample1-62b3222d0f0a76b1
weight b1371 raw capture.sample1-fa0f7424567bbcbb tensor [4456448]
init b1371 trace.data 1223804928 17825792 sample1-fa0f7424567bbcbb
weight b1372 raw capture.sample1-7ab4e5d86428db09 tensor [272]
init b1372 trace.data 1241630720 1088 sample1-7ab4e5d86428db09
weight b1375 raw capture.sample1-c2e1567c43669c0f tensor [10096640]
init b1375 trace.data 1241631808 40386560 sample1-c2e1567c43669c0f
weight b1376 raw capture.sample1-9e26fe336e7e47b5 tensor [272]
init b1376 trace.data 1282018368 1088 sample1-9e26fe336e7e47b5
weight b1379 raw capture.sample1-a3074b423e4200f5 tensor [4456448]
init b1379 trace.data 1282019456 17825792 sample1-a3074b423e4200f5
weight b1380 raw capture.sample1-eb13b24d3746a5d1 tensor [272]
init b1380 trace.data 1299845248 1088 sample1-eb13b24d3746a5d1
weight b1383 raw capture.sample1-e43ffc72bc81a14c tensor [10096640]
init b1383 trace.data 1299846336 40386560 sample1-e43ffc72bc81a14c
weight b1384 raw capture.sample1-62d272602ba706a5 tensor [272]
init b1384 trace.data 1340232896 1088 sample1-62d272602ba706a5
weight b1387 raw capture.sample1-d0bb60d43f1033fb tensor [4456448]
init b1387 trace.data 1340233984 17825792 sample1-d0bb60d43f1033fb
weight b1388 raw capture.sample1-5b7fbf655bafb2f5 tensor [272]
init b1388 trace.data 1358059776 1088 sample1-5b7fbf655bafb2f5
weight b1391 raw capture.sample1-c346b3a5332b326b tensor [10096640]
init b1391 trace.data 1358060864 40386560 sample1-c346b3a5332b326b
weight b1392 raw capture.sample1-01907ac8663cb479 tensor [272]
init b1392 trace.data 1398447424 1088 sample1-01907ac8663cb479
weight b1395 raw capture.sample1-886ee8aaf1ed0ae9 tensor [4456448]
init b1395 trace.data 1398448512 17825792 sample1-886ee8aaf1ed0ae9
weight b1396 raw capture.sample1-99a6c7310ac87205 tensor [272]
init b1396 trace.data 1416274304 1088 sample1-99a6c7310ac87205
weight b1399 raw capture.sample1-4144d23591f52419 tensor [10096640]
init b1399 trace.data 1416275392 40386560 sample1-4144d23591f52419
weight b1400 raw capture.sample1-9cb7d6b3137e7f59 tensor [272]
init b1400 trace.data 1456661952 1088 sample1-9cb7d6b3137e7f59
weight b1403 raw capture.sample1-a8006a19f806278d tensor [4456448]
init b1403 trace.data 1456663040 17825792 sample1-a8006a19f806278d
weight b1404 raw capture.sample1-ce3d90a8344b7bb9 tensor [272]
init b1404 trace.data 1474488832 1088 sample1-ce3d90a8344b7bb9
weight b1407 raw capture.sample1-406e838b33ec985a tensor [10096640]
init b1407 trace.data 1474489920 40386560 sample1-406e838b33ec985a
weight b1408 raw capture.sample1-fe4d14900565ac89 tensor [272]
init b1408 trace.data 1514876480 1088 sample1-fe4d14900565ac89
weight b1411 raw capture.sample1-60bbfe002d904e81 tensor [4456448]
init b1411 trace.data 1514877568 17825792 sample1-60bbfe002d904e81
weight b1412 raw capture.sample1-c28d05f189310465 tensor [272]
init b1412 trace.data 1532703360 1088 sample1-c28d05f189310465
weight b1415 raw capture.sample1-ed7a704c4a1802d0 tensor [10096640]
init b1415 trace.data 1532704448 40386560 sample1-ed7a704c4a1802d0
weight b1416 raw capture.sample1-397de207d7de0b0d tensor [272]
init b1416 trace.data 1573091008 1088 sample1-397de207d7de0b0d
weight b1419 raw capture.sample1-af635e99e8d2489c tensor [4456448]
init b1419 trace.data 1573092096 17825792 sample1-af635e99e8d2489c
weight b1420 raw capture.sample1-252234507056ff59 tensor [272]
init b1420 trace.data 1590917888 1088 sample1-252234507056ff59
weight b1423 raw capture.sample1-58a36e0101ce1e2d tensor [10096640]
init b1423 trace.data 1590918976 40386560 sample1-58a36e0101ce1e2d
weight b1424 raw capture.sample1-b3b1ded1f8a00d4d tensor [272]
init b1424 trace.data 1631305536 1088 sample1-b3b1ded1f8a00d4d
weight b1427 raw capture.sample1-654b823d122666ea tensor [4456448]
init b1427 trace.data 1631306624 17825792 sample1-654b823d122666ea
weight b1428 raw capture.sample1-435bed173e047e79 tensor [272]
init b1428 trace.data 1649132416 1088 sample1-435bed173e047e79
weight b1431 raw capture.sample1-284f5c0c9404e410 tensor [10096640]
init b1431 trace.data 1649133504 40386560 sample1-284f5c0c9404e410
weight b1432 raw capture.sample1-d426804f1a8a58a9 tensor [272]
init b1432 trace.data 1689520064 1088 sample1-d426804f1a8a58a9
weight b1435 raw capture.sample1-5b565e769c7bd825 tensor [4456448]
init b1435 trace.data 1689521152 17825792 sample1-5b565e769c7bd825
weight b1436 raw capture.sample1-b3e5deb470102ef5 tensor [272]
init b1436 trace.data 1707346944 1088 sample1-b3e5deb470102ef5
weight b1439 raw capture.sample1-fb3496a6d0a17c3a tensor [10096640]
init b1439 trace.data 1707348032 40386560 sample1-fb3496a6d0a17c3a
weight b1440 raw capture.sample1-2751aca335aeff01 tensor [272]
init b1440 trace.data 1747734592 1088 sample1-2751aca335aeff01
weight b1443 raw capture.sample1-c6d95de9acddc617 tensor [4456448]
init b1443 trace.data 1747735680 17825792 sample1-c6d95de9acddc617
weight b1444 raw capture.sample1-9839a340d7adbdd5 tensor [272]
init b1444 trace.data 1765561472 1088 sample1-9839a340d7adbdd5
weight b1447 raw capture.sample1-f8905ce0b95537c3 tensor [10096640]
init b1447 trace.data 1765562560 40386560 sample1-f8905ce0b95537c3
weight b1448 raw capture.sample1-0dc1554e61ae8805 tensor [272]
init b1448 trace.data 1805949120 1088 sample1-0dc1554e61ae8805
weight b1451 raw capture.sample1-1590b2736f8cc51a tensor [4456448]
init b1451 trace.data 1805950208 17825792 sample1-1590b2736f8cc51a
weight b1452 raw capture.sample1-6800ba3ef37e50ed tensor [272]
init b1452 trace.data 1823776000 1088 sample1-6800ba3ef37e50ed
weight b1455 raw capture.sample1-dcdc549b8222e446 tensor [10096640]
init b1455 trace.data 1823777088 40386560 sample1-dcdc549b8222e446
weight b1456 raw capture.sample1-241c4caa091428f9 tensor [272]
init b1456 trace.data 1864163648 1088 sample1-241c4caa091428f9
weight b1459 raw capture.sample1-8ba0b6b9a5bd4a9e tensor [4456448]
init b1459 trace.data 1864164736 17825792 sample1-8ba0b6b9a5bd4a9e
weight b1460 raw capture.sample1-b32e7aa6d722eeb9 tensor [272]
init b1460 trace.data 1881990528 1088 sample1-b32e7aa6d722eeb9
weight b1463 raw capture.sample1-4a890d577698ae24 tensor [10096640]
init b1463 trace.data 1881991616 40386560 sample1-4a890d577698ae24
weight b1464 raw capture.sample1-6822ab8dd2308581 tensor [272]
init b1464 trace.data 1922378176 1088 sample1-6822ab8dd2308581
weight b1467 raw capture.sample1-e70a14aa78fee0e6 tensor [4456448]
init b1467 trace.data 1922379264 17825792 sample1-e70a14aa78fee0e6
weight b1468 raw capture.sample1-6957488461e75411 tensor [272]
init b1468 trace.data 1940205056 1088 sample1-6957488461e75411
weight b1471 raw capture.sample1-974feac4191f6e17 tensor [10096640]
init b1471 trace.data 1940206144 40386560 sample1-974feac4191f6e17
weight b1472 raw capture.sample1-8dae1ef99f754e59 tensor [272]
init b1472 trace.data 1980592704 1088 sample1-8dae1ef99f754e59
weight b1475 raw capture.sample1-fe8409906ed758db tensor [4456448]
init b1475 trace.data 1980593792 17825792 sample1-fe8409906ed758db
weight b1476 raw capture.sample1-f40cf0ac7133b87d tensor [272]
init b1476 trace.data 1998419584 1088 sample1-f40cf0ac7133b87d
weight b1479 raw capture.sample1-60788867c3ea7680 tensor [10096640]
init b1479 trace.data 1998420672 40386560 sample1-60788867c3ea7680
weight b1480 raw capture.sample1-80c0906415079699 tensor [272]
init b1480 trace.data 2038807232 1088 sample1-80c0906415079699
weight b1483 raw capture.sample1-4263c79a95165cc5 tensor [4456448]
init b1483 trace.data 2038808320 17825792 sample1-4263c79a95165cc5
weight b1484 raw capture.sample1-45c90cdea515ac8d tensor [272]
init b1484 trace.data 2056634112 1088 sample1-45c90cdea515ac8d
weight b1487 raw capture.sample1-5f384ff0f0f0fe72 tensor [10096640]
init b1487 trace.data 2056635200 40386560 sample1-5f384ff0f0f0fe72
weight b1488 raw capture.sample1-8e594b9ec3e6e401 tensor [272]
init b1488 trace.data 2097021760 1088 sample1-8e594b9ec3e6e401
weight b1491 raw capture.sample1-b2ec44a6ccddb57d tensor [4456448]
init b1491 trace.data 2097022848 17825792 sample1-b2ec44a6ccddb57d
weight b1492 raw capture.sample1-932ea4aeeeafd1c5 tensor [272]
init b1492 trace.data 2114848640 1088 sample1-932ea4aeeeafd1c5
weight b1495 raw capture.sample1-65ac8001d8a0227d tensor [10096640]
init b1495 trace.data 2114849728 40386560 sample1-65ac8001d8a0227d
weight b1496 raw capture.sample1-a99356858d9ec0f1 tensor [272]
init b1496 trace.data 2155236288 1088 sample1-a99356858d9ec0f1
weight b1499 raw capture.sample1-bdbc9d7587d386d7 tensor [4456448]
init b1499 trace.data 2155237376 17825792 sample1-bdbc9d7587d386d7
weight b1500 raw capture.sample1-f06438d0d08f514d tensor [272]
init b1500 trace.data 2173063168 1088 sample1-f06438d0d08f514d
weight b1503 raw capture.sample1-781203af0c5f5f90 tensor [10096640]
init b1503 trace.data 2173064256 40386560 sample1-781203af0c5f5f90
weight b1504 raw capture.sample1-a3b07d27bace88e9 tensor [272]
init b1504 trace.data 2213450816 1088 sample1-a3b07d27bace88e9
weight b1507 raw capture.sample1-f5f4ba413f119cae tensor [4456448]
init b1507 trace.data 2213451904 17825792 sample1-f5f4ba413f119cae
weight b1508 raw capture.sample1-3056ae3e63070f19 tensor [272]
init b1508 trace.data 2231277696 1088 sample1-3056ae3e63070f19
weight b1511 raw capture.sample1-f6f3f76775da38d0 tensor [10096640]
init b1511 trace.data 2231278784 40386560 sample1-f6f3f76775da38d0
weight b1512 raw capture.sample1-127696ae0a015f9d tensor [272]
init b1512 trace.data 2271665344 1088 sample1-127696ae0a015f9d
weight b1515 raw capture.sample1-88c7c2e4b795f9ea tensor [4456448]
init b1515 trace.data 2271666432 17825792 sample1-88c7c2e4b795f9ea
weight b1516 raw capture.sample1-058e19df3ec82da5 tensor [272]
init b1516 trace.data 2289492224 1088 sample1-058e19df3ec82da5
weight b1519 raw capture.sample1-5c9a0d3b0fd3e891 tensor [10096640]
init b1519 trace.data 2289493312 40386560 sample1-5c9a0d3b0fd3e891
weight b1520 raw capture.sample1-263d429e88bb0fb1 tensor [272]
init b1520 trace.data 2329879872 1088 sample1-263d429e88bb0fb1
weight b1523 raw capture.sample1-85c7802382fba761 tensor [4456448]
init b1523 trace.data 2329880960 17825792 sample1-85c7802382fba761
weight b1524 raw capture.sample1-0112e83252876439 tensor [272]
init b1524 trace.data 2347706752 1088 sample1-0112e83252876439
weight b1527 raw capture.sample1-376dcbbf208900e7 tensor [10096640]
init b1527 trace.data 2347707840 40386560 sample1-376dcbbf208900e7
weight b1528 raw capture.sample1-44a536b916654fc9 tensor [272]
init b1528 trace.data 2388094400 1088 sample1-44a536b916654fc9
weight b1531 raw capture.sample1-dc49c710c3868ede tensor [4456448]
init b1531 trace.data 2388095488 17825792 sample1-dc49c710c3868ede
weight b1532 raw capture.sample1-75aa83f195033579 tensor [272]
init b1532 trace.data 2405921280 1088 sample1-75aa83f195033579
weight b1535 raw capture.sample1-08ce0f7cc0b6fea4 tensor [10096640]
init b1535 trace.data 2405922368 40386560 sample1-08ce0f7cc0b6fea4
weight b1536 raw capture.sample1-53ef87fabc7ec1a9 tensor [272]
init b1536 trace.data 2446308928 1088 sample1-53ef87fabc7ec1a9
weight b1539 raw capture.sample1-62447112ef89bd7b tensor [4456448]
init b1539 trace.data 2446310016 17825792 sample1-62447112ef89bd7b
weight b1540 raw capture.sample1-922b6f5d628e3959 tensor [272]
init b1540 trace.data 2464135808 1088 sample1-922b6f5d628e3959
weight b1543 raw capture.sample1-7e73fe12b561247f tensor [10096640]
init b1543 trace.data 2464136896 40386560 sample1-7e73fe12b561247f
weight b1544 raw capture.sample1-c60f0376f8fa2351 tensor [272]
init b1544 trace.data 2504523456 1088 sample1-c60f0376f8fa2351
weight b1547 raw capture.sample1-92d905c158d48192 tensor [4456448]
init b1547 trace.data 2504524544 17825792 sample1-92d905c158d48192
weight b1548 raw capture.sample1-42aab573628aa919 tensor [272]
init b1548 trace.data 2522350336 1088 sample1-42aab573628aa919
weight b1551 raw capture.sample1-d4123af76d98a5df tensor [10096640]
init b1551 trace.data 2522351424 40386560 sample1-d4123af76d98a5df
weight b1552 raw capture.sample1-d24052f82661ee35 tensor [272]
init b1552 trace.data 2562737984 1088 sample1-d24052f82661ee35
weight b1555 raw capture.sample1-3ef335e3c3ab7ad1 tensor [4456448]
init b1555 trace.data 2562739072 17825792 sample1-3ef335e3c3ab7ad1
weight b1556 raw capture.sample1-a8b8b518762ccba5 tensor [272]
init b1556 trace.data 2580564864 1088 sample1-a8b8b518762ccba5
weight b1559 raw capture.sample1-d87a1ceb0b536e33 tensor [10096640]
init b1559 trace.data 2580565952 40386560 sample1-d87a1ceb0b536e33
weight b1560 raw capture.sample1-112817b7b7d2c081 tensor [272]
init b1560 trace.data 2620952512 1088 sample1-112817b7b7d2c081
weight b1563 raw capture.sample1-ae4541a0380062d8 tensor [4456448]
init b1563 trace.data 2620953600 17825792 sample1-ae4541a0380062d8
weight b1564 raw capture.sample1-1d67fe88fe266f39 tensor [272]
init b1564 trace.data 2638779392 1088 sample1-1d67fe88fe266f39
weight b1567 raw capture.sample1-69c89c406ecfb363 tensor [10096640]
init b1567 trace.data 2638780480 40386560 sample1-69c89c406ecfb363
weight b1568 raw capture.sample1-4d6dd066d060e76d tensor [272]
init b1568 trace.data 2679167040 1088 sample1-4d6dd066d060e76d
weight b1571 raw capture.sample1-8336b0e462d862cf tensor [4456448]
init b1571 trace.data 2679168128 17825792 sample1-8336b0e462d862cf
weight b1572 raw capture.sample1-e4bf6a2c6dd87281 tensor [272]
init b1572 trace.data 2696993920 1088 sample1-e4bf6a2c6dd87281
weight b1575 raw capture.sample1-1b479180cc0f8ba1 tensor [10096640]
init b1575 trace.data 2696995008 40386560 sample1-1b479180cc0f8ba1
weight b1576 raw capture.sample1-6466e1d261fd0929 tensor [272]
init b1576 trace.data 2737381568 1088 sample1-6466e1d261fd0929
weight b1579 raw capture.sample1-9e44f70bdf476d27 tensor [4456448]
init b1579 trace.data 2737382656 17825792 sample1-9e44f70bdf476d27
weight b1580 raw capture.sample1-29977e89c357b92d tensor [272]
init b1580 trace.data 2755208448 1088 sample1-29977e89c357b92d
weight b1583 raw capture.sample1-939ba204c227759f tensor [10096640]
init b1583 trace.data 2755209536 40386560 sample1-939ba204c227759f
weight b1584 raw capture.sample1-2d6276d79ee2ea61 tensor [272]
init b1584 trace.data 2795596096 1088 sample1-2d6276d79ee2ea61
weight b1587 raw capture.sample1-e642435abbdbada8 tensor [4456448]
init b1587 trace.data 2795597184 17825792 sample1-e642435abbdbada8
weight b1588 raw capture.sample1-8db03b9ba6b96e91 tensor [272]
init b1588 trace.data 2813422976 1088 sample1-8db03b9ba6b96e91
weight b1591 raw capture.sample1-102bd79230e96e89 tensor [10096640]
init b1591 trace.data 2813424064 40386560 sample1-102bd79230e96e89
weight b1592 raw capture.sample1-ad7aabf7dd787499 tensor [272]
init b1592 trace.data 2853810624 1088 sample1-ad7aabf7dd787499
weight b1595 raw capture.sample1-bb743473bbd613bb tensor [4456448]
init b1595 trace.data 2853811712 17825792 sample1-bb743473bbd613bb
weight b1596 raw capture.sample1-90962e2569012ddd tensor [272]
init b1596 trace.data 2871637504 1088 sample1-90962e2569012ddd
weight b1599 raw capture.sample1-83acf48c0d791989 tensor [10096640]
init b1599 trace.data 2871638592 40386560 sample1-83acf48c0d791989
weight b1600 raw capture.sample1-04618c13c81f20b1 tensor [272]
init b1600 trace.data 2912025152 1088 sample1-04618c13c81f20b1
weight b1603 raw capture.sample1-7cfbf6f9ca4250fa tensor [4456448]
init b1603 trace.data 2912026240 17825792 sample1-7cfbf6f9ca4250fa
weight b1604 raw capture.sample1-86822f5343f99cb1 tensor [272]
init b1604 trace.data 2929852032 1088 sample1-86822f5343f99cb1
weight b1607 raw capture.sample1-f4ecb9f8e795da0a tensor [10096640]
init b1607 trace.data 2929853120 40386560 sample1-f4ecb9f8e795da0a
weight b1608 raw capture.sample1-7d3b263ec68e6f3d tensor [272]
init b1608 trace.data 2970239680 1088 sample1-7d3b263ec68e6f3d
weight b1611 raw capture.sample1-23b907ccb5644270 tensor [4456448]
init b1611 trace.data 2970240768 17825792 sample1-23b907ccb5644270
weight b1612 raw capture.sample1-69413f917b33f4a9 tensor [272]
init b1612 trace.data 2988066560 1088 sample1-69413f917b33f4a9
weight b1615 raw capture.sample1-ee77a166f1ac4f85 tensor [10096640]
init b1615 trace.data 2988067648 40386560 sample1-ee77a166f1ac4f85
weight b1616 raw capture.sample1-40a5ad5a8ee59f61 tensor [272]
init b1616 trace.data 3028454208 1088 sample1-40a5ad5a8ee59f61
weight b1619 raw capture.sample1-43890a72f9cbba41 tensor [4456448]
init b1619 trace.data 3028455296 17825792 sample1-43890a72f9cbba41
weight b1620 raw capture.sample1-ef641423b2ac2c15 tensor [272]
init b1620 trace.data 3046281088 1088 sample1-ef641423b2ac2c15
weight b1623 raw capture.sample1-4dbfaf83524416b4 tensor [10096640]
init b1623 trace.data 3046282176 40386560 sample1-4dbfaf83524416b4
weight b1624 raw capture.sample1-ab6a86d83a0d56d1 tensor [272]
init b1624 trace.data 3086668736 1088 sample1-ab6a86d83a0d56d1
weight b1627 raw capture.sample1-3b4b9bda4acfc788 tensor [4456448]
init b1627 trace.data 3086669824 17825792 sample1-3b4b9bda4acfc788
weight b1628 raw capture.sample1-fd5a7316ede68991 tensor [272]
init b1628 trace.data 3104495616 1088 sample1-fd5a7316ede68991
weight b1631 raw capture.sample1-a538c1a6f17114bd tensor [10096640]
init b1631 trace.data 3104496704 40386560 sample1-a538c1a6f17114bd
weight b1632 raw capture.sample1-4dbeaa83421a43f9 tensor [272]
init b1632 trace.data 3144883264 1088 sample1-4dbeaa83421a43f9
weight b1635 raw capture.sample1-708f0021796f043f tensor [4456448]
init b1635 trace.data 3144884352 17825792 sample1-708f0021796f043f
weight b1636 raw capture.sample1-12bc3e8742cda665 tensor [272]
init b1636 trace.data 3162710144 1088 sample1-12bc3e8742cda665
weight b1639 raw capture.sample1-1bce799d45499b4d tensor [10096640]
init b1639 trace.data 3162711232 40386560 sample1-1bce799d45499b4d
weight b1640 raw capture.sample1-d402d521d849ff89 tensor [272]
init b1640 trace.data 3203097792 1088 sample1-d402d521d849ff89
weight b1643 raw capture.sample1-e5c00ed00585e339 tensor [4456448]
init b1643 trace.data 3203098880 17825792 sample1-e5c00ed00585e339
weight b1644 raw capture.sample1-6ca8603b5ac09e25 tensor [272]
init b1644 trace.data 3220924672 1088 sample1-6ca8603b5ac09e25
weight b1647 raw capture.sample1-9462e85ffd4acb06 tensor [10096640]
init b1647 trace.data 3220925760 40386560 sample1-9462e85ffd4acb06
weight b1648 raw capture.sample1-cbaf5a8d9aa970bd tensor [272]
init b1648 trace.data 3261312320 1088 sample1-cbaf5a8d9aa970bd
weight b1651 raw capture.sample1-556167d11a418922 tensor [4456448]
init b1651 trace.data 3261313408 17825792 sample1-556167d11a418922
weight b1652 raw capture.sample1-cec84f4dbd36ab69 tensor [272]
init b1652 trace.data 3279139200 1088 sample1-cec84f4dbd36ab69
weight b1655 raw capture.sample1-eb46cb775c6d9e72 tensor [10096640]
init b1655 trace.data 3279140288 40386560 sample1-eb46cb775c6d9e72
weight b1656 raw capture.sample1-3f10c287fcdceb6d tensor [272]
init b1656 trace.data 3319526848 1088 sample1-3f10c287fcdceb6d
weight b1659 raw capture.sample1-c05c5ac52fbd6793 tensor [4456448]
init b1659 trace.data 3319527936 17825792 sample1-c05c5ac52fbd6793
weight b1660 raw capture.sample1-8c63e9a696480219 tensor [272]
init b1660 trace.data 3337353728 1088 sample1-8c63e9a696480219
weight b1663 raw capture.sample1-6b1536a0ba80b8fb tensor [10096640]
init b1663 trace.data 3337354816 40386560 sample1-6b1536a0ba80b8fb
weight b1664 raw capture.sample1-18a8e225bcb08cdd tensor [272]
init b1664 trace.data 3377741376 1088 sample1-18a8e225bcb08cdd
weight b1667 raw capture.sample1-80489c8f982697af tensor [4456448]
init b1667 trace.data 3377742464 17825792 sample1-80489c8f982697af
weight b1668 raw capture.sample1-6c0448147ea7f45d tensor [272]
init b1668 trace.data 3395568256 1088 sample1-6c0448147ea7f45d
weight b1671 raw capture.sample1-5013b854fa3f7615 tensor [10096640]
init b1671 trace.data 3395569344 40386560 sample1-5013b854fa3f7615
weight b1672 raw capture.sample1-7668aa667c184a0d tensor [272]
init b1672 trace.data 3435955904 1088 sample1-7668aa667c184a0d
weight b1675 raw capture.sample1-ca784d12ad2451d5 tensor [4456448]
init b1675 trace.data 3435956992 17825792 sample1-ca784d12ad2451d5
weight b1676 raw capture.sample1-089ddaf87022cf81 tensor [272]
init b1676 trace.data 3453782784 1088 sample1-089ddaf87022cf81
weight b1679 raw capture.sample1-9ea4d6c2f1fd9adf tensor [10096640]
init b1679 trace.data 3453783872 40386560 sample1-9ea4d6c2f1fd9adf
weight b1680 raw capture.sample1-f36b33a27fc1e3cd tensor [272]
init b1680 trace.data 3494170432 1088 sample1-f36b33a27fc1e3cd
weight b1683 raw capture.sample1-8fe08c4cc90ec5e0 tensor [4456448]
init b1683 trace.data 3494171520 17825792 sample1-8fe08c4cc90ec5e0
weight b1684 raw capture.sample1-b6b8c891e4e56799 tensor [272]
init b1684 trace.data 3511997312 1088 sample1-b6b8c891e4e56799
weight b1687 raw capture.sample1-50cd9016d9d268c9 tensor [10096640]
init b1687 trace.data 3511998400 40386560 sample1-50cd9016d9d268c9
weight b1688 raw capture.sample1-c545af58185d9819 tensor [272]
init b1688 trace.data 3552384960 1088 sample1-c545af58185d9819
weight b1691 raw capture.sample1-af20b70dd2951444 tensor [4456448]
init b1691 trace.data 3552386048 17825792 sample1-af20b70dd2951444
weight b1692 raw capture.sample1-00cf90ce9c9b9efd tensor [272]
init b1692 trace.data 3570211840 1088 sample1-00cf90ce9c9b9efd
weight b1695 raw capture.sample1-a99025840d2f3652 tensor [10096640]
init b1695 trace.data 3570212928 40386560 sample1-a99025840d2f3652
weight b1696 raw capture.sample1-6f0facd4e5e2b4ed tensor [272]
init b1696 trace.data 3610599488 1088 sample1-6f0facd4e5e2b4ed
weight b1699 raw capture.sample1-661cb50fd8958410 tensor [4456448]
init b1699 trace.data 3610600576 17825792 sample1-661cb50fd8958410
weight b1700 raw capture.sample1-4e8d2e3105ade219 tensor [272]
init b1700 trace.data 3628426368 1088 sample1-4e8d2e3105ade219
weight b1703 raw capture.sample1-ebc8c39611db1ed9 tensor [10096640]
init b1703 trace.data 3628427456 40386560 sample1-ebc8c39611db1ed9
weight b1704 raw capture.sample1-d3758334b731a179 tensor [272]
init b1704 trace.data 3668814016 1088 sample1-d3758334b731a179
weight b1707 raw capture.sample1-0ca1b1a3a7998ed3 tensor [4456448]
init b1707 trace.data 3668815104 17825792 sample1-0ca1b1a3a7998ed3
weight b1708 raw capture.sample1-252fc8b4e124d0ed tensor [272]
init b1708 trace.data 3686640896 1088 sample1-252fc8b4e124d0ed
weight b1711 raw capture.sample1-51840d8e8e410462 tensor [10096640]
init b1711 trace.data 3686641984 40386560 sample1-51840d8e8e410462
weight b1712 raw capture.sample1-e33af14465df2d75 tensor [272]
init b1712 trace.data 3727028544 1088 sample1-e33af14465df2d75
weight b1715 raw capture.sample1-d1bf5f6f586d7211 tensor [4456448]
init b1715 trace.data 3727029632 17825792 sample1-d1bf5f6f586d7211
weight b1716 raw capture.sample1-b6a4e0558b341e95 tensor [272]
init b1716 trace.data 3744855424 1088 sample1-b6a4e0558b341e95
weight b1719 raw capture.sample1-a17d12bc04d7b069 tensor [10096640]
init b1719 trace.data 3744856512 40386560 sample1-a17d12bc04d7b069
weight b1720 raw capture.sample1-360cc2da6b81fc61 tensor [272]
init b1720 trace.data 3785243072 1088 sample1-360cc2da6b81fc61
weight b1723 raw capture.sample1-db6fe6105278f88d tensor [4456448]
init b1723 trace.data 3785244160 17825792 sample1-db6fe6105278f88d
weight b1724 raw capture.sample1-2bdd81071f53c189 tensor [272]
init b1724 trace.data 3803069952 1088 sample1-2bdd81071f53c189
weight b1727 raw capture.sample1-528ffe59067eda97 tensor [10096640]
init b1727 trace.data 3803071040 40386560 sample1-528ffe59067eda97
weight b1728 raw capture.sample1-cf618942e6959ff5 tensor [272]
init b1728 trace.data 3843457600 1088 sample1-cf618942e6959ff5
weight b1731 raw capture.sample1-4a714980a8484d5a tensor [4456448]
init b1731 trace.data 3843458688 17825792 sample1-4a714980a8484d5a
weight b1732 raw capture.sample1-bd74668bb5f08669 tensor [272]
init b1732 trace.data 3861284480 1088 sample1-bd74668bb5f08669
weight b1735 raw capture.sample1-88c1560ccfd0c7ac tensor [10096640]
init b1735 trace.data 3861285568 40386560 sample1-88c1560ccfd0c7ac
weight b1736 raw capture.sample1-47da8bd8b2d76b49 tensor [272]
init b1736 trace.data 3901672128 1088 sample1-47da8bd8b2d76b49
weight b1739 raw capture.sample1-6f9c673b4d2ab0a1 tensor [4456448]
init b1739 trace.data 3901673216 17825792 sample1-6f9c673b4d2ab0a1
weight b1740 raw capture.sample1-9d23056147e1d391 tensor [272]
init b1740 trace.data 3919499008 1088 sample1-9d23056147e1d391
weight b1743 raw capture.sample1-bde31a692fbc3b7d tensor [10096640]
init b1743 trace.data 3919500096 40386560 sample1-bde31a692fbc3b7d
weight b1744 raw capture.sample1-967ebbea5aba6031 tensor [272]
init b1744 trace.data 3959886656 1088 sample1-967ebbea5aba6031
weight b1747 raw capture.sample1-e19c8a9d5b749be1 tensor [4456448]
init b1747 trace.data 3959887744 17825792 sample1-e19c8a9d5b749be1
weight b1748 raw capture.sample1-92c28039c992a999 tensor [272]
init b1748 trace.data 3977713536 1088 sample1-92c28039c992a999
weight b1751 raw capture.sample1-b7df8c4a30719022 tensor [10096640]
init b1751 trace.data 3977714624 40386560 sample1-b7df8c4a30719022
weight b1752 raw capture.sample1-a55994fc8970eb59 tensor [272]
init b1752 trace.data 4018101184 1088 sample1-a55994fc8970eb59
weight b1755 raw capture.sample1-ca36811d3d76d8cb tensor [4456448]
init b1755 trace.data 4018102272 17825792 sample1-ca36811d3d76d8cb
weight b1756 raw capture.sample1-498236587a353401 tensor [272]
init b1756 trace.data 4035928064 1088 sample1-498236587a353401
weight b1757 raw capture.sample1-7c5fc1aeccb30485 tensor [1048576]
init b1757 trace.data 4035929152 4194304 sample1-7c5fc1aeccb30485
weight b1758 raw capture.sample1-3f62582976a0423d tensor [135168]
init b1758 trace.data 4040123456 540672 sample1-3f62582976a0423d
buf b1762 f32 524288
init b1762 trace.data 4040664128 2097152 sample1-16bfc88318476ba5
buf b1763 f32 524288
init b1763 trace.data 4040664128 2097152 sample1-16bfc88318476ba5
buf b1764 f32 2375680
buf b1765 f32 1048576
init b1765 trace.data 4035929152 4194304 sample1-7c5fc1aeccb30485
buf b1766 f32 524288
buf b1767 f32 11396480
buf b1768 f32 30720
buf b1771 f32 524288
buf b1774 f32 5812224
buf b1777 f32 1572864
buf b1778 f32 524288
init b1778 trace.data 4040664128 2097152 sample1-16bfc88318476ba5
buf b1779 f32 524288
buf b1780 f32 1048576
init b1780 trace.data 4035929152 4194304 sample1-7c5fc1aeccb30485
buf b1781 f32 524288
buf b1782 f32 131072
buf b1786 f32 262144
init b1786 trace.data 4042761280 1048576 sample1-fe4fd2c189a60655
buf b1787 f32 262144
init b1787 trace.data 4042761280 1048576 sample1-fe4fd2c189a60655
buf b1788 f32 1187840
buf b1789 f32 524288
init b1789 trace.data 4040664128 2097152 sample1-16bfc88318476ba5
buf b1790 f32 262144
buf b1791 f32 6792064
buf b1792 f32 30720
buf b1795 f32 262144
buf b1798 f32 3174400
buf b1801 f32 786432
buf b1802 f32 262144
init b1802 trace.data 4042761280 1048576 sample1-fe4fd2c189a60655
buf b1803 f32 262144
buf b1804 f32 524288
init b1804 trace.data 4040664128 2097152 sample1-16bfc88318476ba5
buf b1805 f32 262144
buf b1806 f32 131072
buf b1810 f32 131072
init b1810 trace.data 4043809856 524288 sample1-0a87cda250f6b8fd
buf b1811 f32 131072
init b1811 trace.data 4043809856 524288 sample1-0a87cda250f6b8fd
buf b1812 f32 593920
buf b1813 f32 262144
init b1813 trace.data 4042761280 1048576 sample1-fe4fd2c189a60655
buf b1814 f32 131072
buf b1815 f32 4489856
buf b1816 f32 30720
buf b1819 f32 131072
buf b1822 f32 1855488
buf b1825 f32 393216
buf b1826 f32 131072
init b1826 trace.data 4043809856 524288 sample1-0a87cda250f6b8fd
buf b1827 f32 131072
buf b1828 f32 262144
init b1828 trace.data 4042761280 1048576 sample1-fe4fd2c189a60655
buf b1829 f32 131072
buf b1830 f32 131072
buf b1834 f32 65536
init b1834 trace.data 4044334144 262144 sample1-5dd3ce7cda6945e1
buf b1835 f32 65536
init b1835 trace.data 4044334144 262144 sample1-5dd3ce7cda6945e1
buf b1836 f32 296960
buf b1837 f32 131072
init b1837 trace.data 4043809856 524288 sample1-0a87cda250f6b8fd
buf b1838 f32 65536
buf b1839 f32 3338752
buf b1840 f32 30720
buf b1843 f32 65536
buf b1846 f32 1196032
buf b1849 f32 196608
buf b1850 f32 65536
init b1850 trace.data 4044334144 262144 sample1-5dd3ce7cda6945e1
buf b1851 f32 65536
buf b1852 f32 131072
init b1852 trace.data 4043809856 524288 sample1-0a87cda250f6b8fd
buf b1853 f32 65536
buf b1854 f32 131072
buf b1858 f32 32768
init b1858 trace.data 4044596288 131072 sample1-dac5cfc4a8951937
buf b1859 f32 32768
init b1859 trace.data 4044596288 131072 sample1-dac5cfc4a8951937
buf b1860 f32 148480
buf b1861 f32 65536
init b1861 trace.data 4044334144 262144 sample1-5dd3ce7cda6945e1
buf b1862 f32 32768
buf b1863 f32 2763200
buf b1864 f32 30720
buf b1867 f32 32768
buf b1870 f32 866304
buf b1873 f32 98304
buf b1874 f32 32768
init b1874 trace.data 4044596288 131072 sample1-dac5cfc4a8951937
buf b1875 f32 32768
buf b1876 f32 65536
init b1876 trace.data 4044334144 262144 sample1-5dd3ce7cda6945e1
buf b1877 f32 32768
buf b1878 f32 131072
buf b1882 f32 32768
init b1882 trace.data 4044596288 131072 sample1-dac5cfc4a8951937
buf b1883 f32 32768
init b1883 trace.data 4044596288 131072 sample1-dac5cfc4a8951937
buf b1884 f32 148480
buf b1885 f32 65536
init b1885 trace.data 4044334144 262144 sample1-5dd3ce7cda6945e1
buf b1886 f32 32768
buf b1887 f32 2547368
buf b1888 f32 30720
buf b1891 f32 20480
buf b1894 f32 742656
buf b1897 f32 98304
buf b1898 f32 32768
init b1898 trace.data 4044596288 131072 sample1-dac5cfc4a8951937
buf b1899 f32 32768
buf b1900 f32 65536
init b1900 trace.data 4044334144 262144 sample1-5dd3ce7cda6945e1
buf b1901 f32 32768
buf b1902 f32 131072
buf b1903 f32 4096
buf position f32 1 input:position
view b1774_21118976_2129920 b1774 5279744 532480
view b158_81920_524288 b158 20480 131072
view b464_81920_524288 b464 20480 131072
view b769_81920_524288 b769 20480 131072
view b1074_81920_524288 b1074 20480 131072
view b1798_10567680_2129920 b1798 2641920 532480
view b1822_5292032_2129920 b1822 1323008 532480
view b1836_593920_593920 b1836 148480 148480
view b1835_131072_131072 b1835 32768 32768
view b1838_131072_131072 b1838 32768 32768
view b1837_262144_262144 b1837 65536 65536
view b1846_2654208_2129920 b1846 663552 532480
view b1870_1335296_2129920 b1870 333824 532480
view b1884_296960_296960 b1884 74240 74240
view b1883_65536_65536 b1883 16384 16384
view b1886_65536_65536 b1886 16384 16384
view b1885_131072_131072 b1885 32768 32768
view b1894_840704_2129920 b1894 210176 532480
view b23_4803072_32768 b23 1200768 8192
view b41_4803072_32768 b41 1200768 8192
view b59_4803072_32768 b59 1200768 8192
view b77_4803072_32768 b77 1200768 8192
view b95_4803072_32768 b95 1200768 8192
view b113_4803072_32768 b113 1200768 8192
view b131_4803072_32768 b131 1200768 8192
view b158_49152_557056 b158 12288 139264
view b149_4803072_32768 b149 1200768 8192
view b185_4803072_32768 b185 1200768 8192
view b203_4803072_32768 b203 1200768 8192
view b221_4803072_32768 b221 1200768 8192
view b239_4803072_32768 b239 1200768 8192
view b257_4803072_32768 b257 1200768 8192
view b275_4803072_32768 b275 1200768 8192
view b293_4803072_32768 b293 1200768 8192
view b311_4803072_32768 b311 1200768 8192
view b329_4803072_32768 b329 1200768 8192
view b347_4803072_32768 b347 1200768 8192
view b365_4803072_32768 b365 1200768 8192
view b383_4803072_32768 b383 1200768 8192
view b401_4803072_32768 b401 1200768 8192
view b419_4803072_32768 b419 1200768 8192
view b437_4803072_32768 b437 1200768 8192
view b464_49152_557056 b464 12288 139264
view b455_4803072_32768 b455 1200768 8192
view b490_4803072_32768 b490 1200768 8192
view b508_4803072_32768 b508 1200768 8192
view b526_4803072_32768 b526 1200768 8192
view b544_4803072_32768 b544 1200768 8192
view b562_4803072_32768 b562 1200768 8192
view b580_4803072_32768 b580 1200768 8192
view b598_4803072_32768 b598 1200768 8192
view b616_4803072_32768 b616 1200768 8192
view b634_4803072_32768 b634 1200768 8192
view b652_4803072_32768 b652 1200768 8192
view b670_4803072_32768 b670 1200768 8192
view b688_4803072_32768 b688 1200768 8192
view b706_4803072_32768 b706 1200768 8192
view b724_4803072_32768 b724 1200768 8192
view b742_4803072_32768 b742 1200768 8192
view b769_49152_557056 b769 12288 139264
view b760_4803072_32768 b760 1200768 8192
view b795_4803072_32768 b795 1200768 8192
view b813_4803072_32768 b813 1200768 8192
view b831_4803072_32768 b831 1200768 8192
view b849_4803072_32768 b849 1200768 8192
view b867_4803072_32768 b867 1200768 8192
view b885_4803072_32768 b885 1200768 8192
view b903_4803072_32768 b903 1200768 8192
view b921_4803072_32768 b921 1200768 8192
view b939_4803072_32768 b939 1200768 8192
view b957_4803072_32768 b957 1200768 8192
view b975_4803072_32768 b975 1200768 8192
view b993_4803072_32768 b993 1200768 8192
view b1011_4803072_32768 b1011 1200768 8192
view b1029_4803072_32768 b1029 1200768 8192
view b1047_4803072_32768 b1047 1200768 8192
view b1074_49152_557056 b1074 12288 139264
view b1065_4803072_32768 b1065 1200768 8192
view b1100_4803072_32768 b1100 1200768 8192
view b1118_4803072_32768 b1118 1200768 8192
view b1136_4803072_32768 b1136 1200768 8192
view b1154_4803072_32768 b1154 1200768 8192
view b1172_4803072_32768 b1172 1200768 8192
view b1190_4803072_32768 b1190 1200768 8192
view b1208_4803072_32768 b1208 1200768 8192
view b1226_4803072_32768 b1226 1200768 8192
weight backbone_embeddings_weight_emb f32 backbone.embeddings.weight tensor [32768,4096]
kernel raw0 in=1 out=1 threads=32768 tg=256 tgbytes=4 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 43000000 000c4227 00004290 0000428f 0000422b 43000000 00082223 00000386 00004880 00000486 00002800 00000413 42200000 00000513 00000000 00000c13 40400000 00000d13 40800000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 46200000 001cc770 00000772 0000c590 0000a891 0024e000 00146683 0000c685 00002000 0000c685 ffff2680 0000a786 00000004 00004b2a 00000000 00000d40 00316e70 00000f13 3f800000 0039ef23 00000e72 0000a986 00000003 00016a90 00292983 00012991 00ad6d80 0000ea91 00244000 00254822 00000008 00000073 00000f72 0000c991 00000000 0000ea91 00244000 00254822 00000008 00000073 0000c685 00004880 0000e785 00000001 00016b2a 3f800000 00000041 00148883 0001088e 00252980 00010e25 0001ce24 0001ce2a 3f800000 00000f13 3f800000 0039ee27 00390821 0001088e 00392980 00000073 0000a52a 3f800000 00000041 00000080 00000181 0000018f 0000222a 3f000000 00000313 43000000 000c4227 00004290 0000428f 0000422b 43000000 00082223 00000386 00004880 00000486 00000080 00000413 3f800000 00000513 00000000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 43000000 001cc770 00000772 0000c590 00146683 0000c891 00004800 0000a991 00250900 00250820 00010828 00148683 0000c88e 004d2980 00000073 0000a52a 3f800000 00000041 00000001
endkernel
kernel raw1 in=1 out=1 threads=262144 tg=32 tgbytes=4096 layout=slots slots=mem:in0
words 00000080 0000008f 0000012a 3f000000 00000213 46000000 00082127 00002090 0000038f 0000632b 46000000 000c0423 00008190 0000852a 3f000000 00000213 42800000 0008a527 0000a290 0000468f 0000c62a 3f000000 00000213 41800000 0008c627 0000c390 00000481 00004891 00250800 00010824 00010825 00004991 00250880 00000d86 00140000 00000686 00004000 0008c683 00000f86 00100000 0005ef83 00000886 00200000 00002986 00000080 00250883 00110883 00006b86 00000080 00116b83 0001a585 00000000 00011091 004d6980 00011191 004d69a0 00011291 004d69c0 00011391 004d69e0 00008a8f 00014e2b 42000000 00001913 42000000 00001813 42000000 00014d2a 00000000 00001840 0004a983 00012391 00392980 0000c191 004d2980 00202221 00004224 002caa83 00000413 00000000 00014591 00394980 000a0721 00142c21 000d8c21 0030f020 00014691 00394d80 0040c422 00000004 00014591 003949a0 000a2721 00142c21 000d8c21 0030f120 00014691 00394da0 0044c422 00000004 00014591 003949c0 000a4721 00142c21 000d8c21 0030f220 00014691 00394dc0 0048c422 00000004 00014591 003949e0 000a6721 00142c21 000d8c21 0030f320 00014691 00394de0 004cc422 00000004 0001a461 00000000 0001ad2a 42000000 0000a585 00002800 0000c685 00000080 00000041 00000050 00008a8f 00001a13 42000000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00001940 0001e560 00000000 0014e720 0001ef2a 3f800000 00000041 00008c86 00002800 00358c83 00058c83 00018391 00392980 000d2722 00000007 00008e86 00002000 003dce83 0001c78e 006d6980 00000073 00000050 00001813 42000000 00014d2a 00000000 00001840 0004a983 00012391 00392980 0000c191 004d2980 00202221 00004224 002caa83 00000413 00000000 00014591 00394980 000a0721 00142c21 000d8c21 0030f020 00014691 00394d80 0040c422 00000004 00014591 003949a0 000a2721 00142c21 000d8c21 0030f120 00014691 00394da0 0044c422 00000004 00014591 003949c0 000a4721 00142c21 000d8c21 0030f220 00014691 00394dc0 0048c422 00000004 00014591 003949e0 000a6721 00142c21 000d8c21 0030f320 00014691 00394de0 004cc422 00000004 0001a461 00000000 0001ad2a 42000000 0000a585 00002800 0000c685 00000080 00000041 00000050 00008a8f 00001a13 42000000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00001940 0001e560 00000000 0014e720 0001ef2a 3f800000 00000041 00008c86 00002800 00358c83 00058c83 00018391 003e2980 000d2722 00000007 00008e86 00002000 003dce83 0001c78e 00716980 00000073 00000050 00001813 42000000 00014d2a 00000000 00001840 0004a983 00012391 00392980 0000c191 004d2980 00202221 00004224 002caa83 00000413 00000000 00014591 00394980 000a0721 00142c21 000d8c21 0030f020 00014691 00394d80 0040c422 00000004 00014591 003949a0 000a2721 00142c21 000d8c21 0030f120 00014691 00394da0 0044c422 00000004 00014591 003949c0 000a4721 00142c21 000d8c21 0030f220 00014691 00394dc0 0048c422 00000004 00014591 003949e0 000a6721 00142c21 000d8c21 0030f320 00014691 00394de0 004cc422 00000004 0001a461 00000000 0001ad2a 42000000 0000a585 00002800 0000c685 00000080 00000041 00000050 00008a8f 00001a13 42000000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00001940 0001e560 00000000 0014e720 0001ef2a 3f800000 00000041 00008c86 00002800 00358c83 00058c83 00018391 00432980 000d2722 00000007 00008e86 00002000 003dce83 0001c78e 00756980 00000073 00000050 00001813 42000000 00014d2a 00000000 00001840 0004a983 00012391 00392980 0000c191 004d2980 00202221 00004224 002caa83 00000413 00000000 00014591 00394980 000a0721 00142c21 000d8c21 0030f020 00014691 00394d80 0040c422 00000004 00014591 003949a0 000a2721 00142c21 000d8c21 0030f120 00014691 00394da0 0044c422 00000004 00014591 003949c0 000a4721 00142c21 000d8c21 0030f220 00014691 00394dc0 0048c422 00000004 00014591 003949e0 000a6721 00142c21 000d8c21 0030f320 00014691 00394de0 004cc422 00000004 0001a461 00000000 0001ad2a 42000000 0000a585 00002800 0000c685 00000080 00000041 00000050 00008a8f 00001a13 42000000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00001940 0001e560 00000000 0014e720 0001ef2a 3f800000 00000041 00008c86 00002800 00358c83 00058c83 00018391 00482980 000d2722 00000007 00008e86 00002000 003dce83 0001c78e 00796980 00000073 0001108e 005d6980 0001108e 004d6980 0001118e 005d69a0 0001118e 004d69a0 0001128e 005d69c0 0001128e 004d69c0 0001138e 005d69e0 0001138e 004d69e0 00000001
endkernel
kernel raw2 in=1 out=1 threads=262144 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 0000018f 00000213 41000000 00082127 00002290 00004386 00000008 0000048f 0000658f 00148423 00008390 00000181 00002e8f 00004486 00004880 00004586 00002000 00006686 00000400 0018a583 00000113 00000000 00000313 40800000 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00248b83 00016691 00000000 00014791 006d6980 0000c825 00010824 0001082a 3f800000 00000913 3f800000 00212827 0020c821 0020e721 0001478e 007d6980 001cea21 00282120 0000842a 3f800000 00000041 0001c161 00000000 00000050 00000713 41000000 00000613 43000000 00000740 0019c870 00000872 0019c820 0001c960 00000000 00010a60 00000000 00292920 0001c961 00000000 00000073 00000050 0000c62b 3f000000 00000041 00000b13 00000000 00016d60 00000000 0001ad2b 3a800000 0001ad2a 3727c5ac 0001ad29 00000d8e 008d6980 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00014791 007d6980 0034e721 0001478e 008d6d80 00012891 00250980 0020e721 0001478e 009d6d80 0000842a 3f800000 00000041 00000001
endkernel
kernel raw3 in=2 out=1 threads=10240 tg=64 tgbytes=16 layout=slots slots=mem:in0,input:in1
words 00000090 00000186 00000003 0000018d 00238680 0000218e 00000000 0000018d 0023cf00 0000218e 00000001 0000018d 00241780 0000218e 00000002 00000001
endkernel
kernel raw4 in=2 out=1 threads=1572864 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 46400000 00040227 00004190 00002286 00003000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw5 in=2 out=1 threads=131072 tg=64 tgbytes=16 layout=slots slots=mem:in0,input:in1
words 00000090 00000113 44800000 00040227 00004190 00002286 00000400 0000438f 000c0323 00006290 0000448f 00000513 42000000 00148427 00008390 00006486 00000020 0000858f 0000468f 0014c623 0000c490 00002586 00000400 00008486 00000020 0010a583 000ca583 0000a78d 00000000 0000078e 00000000 00000001
endkernel
kernel raw6 in=1 out=1 threads=1048576 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 42000000 000c4227 00004390 0000648f 0000842b 42000000 00102423 00008490 00008486 00000100 00048583 00006686 00003000 0014c683 00006786 00002000 0014e783 0000c591 00000000 0000a624 00000713 00000000 0014e870 00000872 0000a62a 3f800000 00000073 0000e68e 00180000 00000001
endkernel
kernel raw7 in=1 out=1 threads=131072 tg=32 tgbytes=4096 layout=slots slots=mem:in0
words 00000080 00000481 0000008f 00000113 42000000 00040227 00004790 0000e186 00000020 0000218f 00040123 00002190 0000e28f 00000313 43000000 000c4227 00004290 00004386 00000080 0000638f 0000e58f 000ca323 00006390 00006a86 00000020 00004b86 00100000 00296b83 00116b83 00004586 00180000 0028a583 0004a583 00004f86 00080000 0029ef83 0005ef83 00000886 00000020 00110883 0000e986 00000020 00112983 00011091 00509000 00013491 00508000 00008a8f 00014e2b 42000000 00001913 42000000 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00181000 00016791 00180000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 00280000 00000073 00000050 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00181000 00016791 00180000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 00290000 00000073 00000050 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00181000 00016791 00180000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 002a0000 00000073 00000050 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00181000 00016791 00180000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 002b0000 00000073 00000050 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00181000 00016791 00180000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 002c0000 00000073 00000050 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00181000 00016791 00180000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 002d0000 00000073 00000050 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00181000 00016791 00180000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 002e0000 00000073 00000050 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00181000 00016791 00180000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 002f0000 00000073 0001108e 00509000 00002a8f 00001a13 3f800000 00695b70 00001b72 0001348e 00549000 00000073 00000001
endkernel
kernel raw8 in=2 out=1 threads=524288 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw9 in=3 out=1 threads=524288 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00002386 00002000 00086383 0000048c 00000000 0000058d 00000000 0000648e 00000000 0000658e 00001000 00000001
endkernel
kernel raw10 in=3 out=1 threads=524288 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000191 00000000 0000028d 00000000 0000038c 00000000 00006326 00000413 3f800000 000c8423 000c2121 00104221 00082120 0000018e 00000000 00000001
endkernel
kernel raw11 in=1 out=1 threads=16384 tg=256 tgbytes=4 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 42800000 000c4227 00004290 0000428f 0000422b 42800000 00082223 00000386 00004880 00000486 00002800 00000413 42200000 00000513 00000000 00000c13 40400000 00000d13 40800000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 46200000 001cc770 00000772 0000c590 0000a891 0012c000 00146683 0000c685 00002000 0000c685 ffff2680 0000a786 00000004 00004b2a 00000000 00000d40 00316e70 00000f13 3f800000 0039ef23 00000e72 0000a986 00000003 00016a90 00292983 00012991 00672b80 0000ea91 00122000 00254822 00000008 00000073 00000f72 0000c991 00000000 0000ea91 00122000 00254822 00000008 00000073 0000c685 00004880 0000e785 00000001 00016b2a 3f800000 00000041 00148883 0001088e 00130980 00010e25 0001ce24 0001ce2a 3f800000 00000f13 3f800000 0039ee27 00390821 0001088e 001d0980 00000073 0000a52a 3f800000 00000041 00000080 00000181 0000018f 0000222a 3f000000 00000313 42800000 000c4227 00004290 0000428f 0000422b 42800000 00082223 00000386 00004880 00000486 00000080 00000413 3f800000 00000513 00000000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 43000000 001cc770 00000772 0000c590 00146683 0000c891 00004800 0000a991 0012e900 00250820 00010828 00148683 0000c88e 00270980 00000073 0000a52a 3f800000 00000041 00000001
endkernel
kernel raw12 in=1 out=1 threads=262144 tg=32 tgbytes=4096 layout=slots slots=mem:in0
words 00000080 0000008f 0000012a 3f000000 00000213 46000000 00082127 00002090 0000038f 0000632b 46000000 000c0423 00008190 0000852a 3f000000 00000213 42800000 0008a527 0000a290 0000468f 0000c62a 3f000000 00000213 41800000 0008c627 0000c390 00000481 00004891 0012e800 00010824 00010825 00004991 0012e880 00000d86 000a0000 00000686 00002000 0008c683 00000f86 00080000 0005ef83 00000886 00200000 00002986 00000080 00250883 00110883 00006b86 00000080 00116b83 0001a585 00000000 00011091 00272980 00011191 002729a0 00011291 002729c0 00011391 002729e0 00008a8f 00014e2b 42000000 00001913 42000000 00001813 42000000 00014d2a 00000000 00001840 0004a983 00012391 001d0980 0000c191 00270980 00202221 00004224 002caa83 00000413 00000000 00014591 001d2980 000a0721 00142c21 000d8c21 0030f020 00014691 001d2d80 0040c422 00000004 00014591 001d29a0 000a2721 00142c21 000d8c21 0030f120 00014691 001d2da0 0044c422 00000004 00014591 001d29c0 000a4721 00142c21 000d8c21 0030f220 00014691 001d2dc0 0048c422 00000004 00014591 001d29e0 000a6721 00142c21 000d8c21 0030f320 00014691 001d2de0 004cc422 00000004 0001a461 00000000 0001ad2a 42000000 0000a585 00002800 0000c685 00000080 00000041 00000050 00008a8f 00001a13 42000000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00001940 0001e560 00000000 0014e720 0001ef2a 3f800000 00000041 00008c86 00002800 00358c83 00058c83 00018391 001d0980 000d2722 00000007 00008e86 00002000 003dce83 0001c78e 00472980 00000073 00000050 00001813 42000000 00014d2a 00000000 00001840 0004a983 00012391 001d0980 0000c191 00270980 00202221 00004224 002caa83 00000413 00000000 00014591 001d2980 000a0721 00142c21 000d8c21 0030f020 00014691 001d2d80 0040c422 00000004 00014591 001d29a0 000a2721 00142c21 000d8c21 0030f120 00014691 001d2da0 0044c422 00000004 00014591 001d29c0 000a4721 00142c21 000d8c21 0030f220 00014691 001d2dc0 0048c422 00000004 00014591 001d29e0 000a6721 00142c21 000d8c21 0030f320 00014691 001d2de0 004cc422 00000004 0001a461 00000000 0001ad2a 42000000 0000a585 00002800 0000c685 00000080 00000041 00000050 00008a8f 00001a13 42000000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00001940 0001e560 00000000 0014e720 0001ef2a 3f800000 00000041 00008c86 00002800 00358c83 00058c83 00018391 00220980 000d2722 00000007 00008e86 00002000 003dce83 0001c78e 004b2980 00000073 0001108e 00372980 0001108e 00272980 0001118e 003729a0 0001118e 002729a0 0001128e 003729c0 0001128e 002729c0 0001138e 003729e0 0001138e 002729e0 00000001
endkernel
kernel raw13 in=1 out=1 threads=131072 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 0000018f 00000213 41000000 00082127 00002290 00004386 00000008 0000048f 0000658f 00148423 00008390 00000181 00002e8f 00004486 00004880 00004586 00002000 00006686 00000400 0018a583 00000113 00000000 00000313 40800000 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00248b83 00016691 00000000 00014791 00472980 0000c825 00010824 0001082a 3f800000 00000913 3f800000 00212827 0020c821 0020e721 0001478e 004f2980 001cea21 00282120 0000842a 3f800000 00000041 0001c161 00000000 00000050 00000713 41000000 00000613 43000000 00000740 0019c870 00000872 0019c820 0001c960 00000000 00010a60 00000000 00292920 0001c961 00000000 00000073 00000050 0000c62b 3f000000 00000041 00000b13 00000000 00016d60 00000000 0001ad2b 3a800000 0001ad2a 3727c5ac 0001ad29 00000d8e 00572980 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00014791 004f2980 0034e721 0001478e 00572b80 00012891 0012e980 0020e721 0001478e 005f2b80 0000842a 3f800000 00000041 00000001
endkernel
kernel raw14 in=2 out=1 threads=10240 tg=64 tgbytes=16 layout=slots slots=mem:in0,input:in1
words 00000090 00000186 00000003 0000018d 00116680 0000218e 00000000 0000018d 0011af00 0000218e 00000001 0000018d 0011f780 0000218e 00000002 00000001
endkernel
kernel raw15 in=2 out=1 threads=786432 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 46400000 00040227 00004190 00002286 00003000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw16 in=1 out=1 threads=524288 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 42000000 000c4227 00004390 0000648f 0000842b 42000000 00102423 00008490 00008486 00000100 00048583 00006686 00003000 0014c683 00006786 00002000 0014e783 0000c591 00000000 0000a624 00000713 00000000 0014e870 00000872 0000a62a 3f800000 00000073 0000e68e 000c0000 00000001
endkernel
kernel raw17 in=1 out=1 threads=131072 tg=32 tgbytes=4096 layout=slots slots=mem:in0
words 00000080 00000481 0000008f 00000113 42000000 00040227 00004790 0000e186 00000020 0000218f 00040123 00002190 0000e28f 00000313 43000000 000c4227 00004290 00004386 00000080 0000638f 0000e58f 000ca323 00006390 00006a86 00000020 00004b86 00080000 00296b83 00116b83 00004586 000c0000 0028a583 0004a583 00004f86 00040000 0029ef83 0005ef83 00000886 00000020 00110883 0000e986 00000020 00112983 00011091 00285000 00013491 00284000 00008a8f 00014e2b 42000000 00001913 42000000 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 000c1000 00016791 000c0000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 00140000 00000073 00000050 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 000c1000 00016791 000c0000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 00150000 00000073 00000050 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 000c1000 00016791 000c0000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 00160000 00000073 00000050 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 000c1000 00016791 000c0000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 00170000 00000073 0001108e 00285000 00002a8f 00001a13 3f800000 00695b70 00001b72 0001348e 002c5000 00000073 00000001
endkernel
kernel raw18 in=2 out=1 threads=262144 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw19 in=3 out=1 threads=262144 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00002386 00002000 00086383 0000048c 00000000 0000058d 00000000 0000648e 00000000 0000658e 00001000 00000001
endkernel
kernel raw20 in=3 out=1 threads=262144 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000191 00000000 0000028d 00000000 0000038c 00000000 00006326 00000413 3f800000 000c8423 000c2121 00104221 00082120 0000018e 00000000 00000001
endkernel
kernel raw21 in=1 out=1 threads=8192 tg=256 tgbytes=4 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 42000000 000c4227 00004290 0000428f 0000422b 42000000 00082223 00000386 00004880 00000486 00002800 00000413 42200000 00000513 00000000 00000c13 40400000 00000d13 40800000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 46200000 001cc770 00000772 0000c590 0000a891 0009b000 00146683 0000c685 00002000 0000c685 ffff2680 0000a786 00000004 00004b2a 00000000 00000d40 00316e70 00000f13 3f800000 0039ef23 00000e72 0000a986 00000003 00016a90 00292983 00012991 00440a80 0000ea91 00091000 00254822 00000008 00000073 00000f72 0000c991 00000000 0000ea91 00091000 00254822 00000008 00000073 0000c685 00004880 0000e785 00000001 00016b2a 3f800000 00000041 00148883 0001088e 0009f980 00010e25 0001ce24 0001ce2a 3f800000 00000f13 3f800000 0039ee27 00390821 0001088e 000ef980 00000073 0000a52a 3f800000 00000041 00000080 00000181 0000018f 0000222a 3f000000 00000313 42000000 000c4227 00004290 0000428f 0000422b 42000000 00082223 00000386 00004880 00000486 00000080 00000413 3f800000 00000513 00000000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 43000000 001cc770 00000772 0000c590 00146683 0000c891 00004800 0000a991 0009d900 00250820 00010828 00148683 0000c88e 0013f980 00000073 0000a52a 3f800000 00000041 00000001
endkernel
kernel raw22 in=1 out=1 threads=262144 tg=32 tgbytes=4096 layout=slots slots=mem:in0
words 00000080 0000008f 0000012a 3f000000 00000213 46000000 00082127 00002090 0000038f 0000632b 46000000 000c0423 00008190 0000852a 3f000000 00000213 42800000 0008a527 0000a290 0000468f 0000c62a 3f000000 00000213 41800000 0008c627 0000c390 00000481 00004891 0009d800 00010824 00010825 00004991 0009d880 00000d86 00050000 00000686 00001000 0008c683 00000f86 00040000 0005ef83 00000886 00200000 00002986 00000080 00250883 00110883 00006b86 00000080 00116b83 0001a585 00000000 00011091 00140980 00011191 001409a0 00011291 001409c0 00011391 001409e0 00008a8f 00014e2b 42000000 00001913 42000000 00001813 42000000 00014d2a 00000000 00001840 0004a983 00012391 000ef980 0000c191 0013f980 00202221 00004224 002caa83 00000413 00000000 00014591 000f1980 000a0721 00142c21 000d8c21 0030f020 00014691 000f1d80 0040c422 00000004 00014591 000f19a0 000a2721 00142c21 000d8c21 0030f120 00014691 000f1da0 0044c422 00000004 00014591 000f19c0 000a4721 00142c21 000d8c21 0030f220 00014691 000f1dc0 0048c422 00000004 00014591 000f19e0 000a6721 00142c21 000d8c21 0030f320 00014691 000f1de0 004cc422 00000004 0001a461 00000000 0001ad2a 42000000 0000a585 00002800 0000c685 00000080 00000041 00000050 00008a8f 00001a13 42000000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00001940 0001e560 00000000 0014e720 0001ef2a 3f800000 00000041 00008c86 00002800 00358c83 00058c83 00018391 000ef980 000d2722 00000007 00008e86 00002000 003dce83 0001c78e 00340980 00000073 0001108e 00240980 0001108e 00140980 0001118e 002409a0 0001118e 001409a0 0001128e 002409c0 0001128e 001409c0 0001138e 002409e0 0001138e 001409e0 00000001
endkernel
kernel raw23 in=1 out=1 threads=65536 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 0000018f 00000213 41000000 00082127 00002290 00004386 00000008 0000048f 0000658f 00148423 00008390 00000181 00002e8f 00004486 00004880 00004586 00002000 00006686 00000400 0018a583 00000113 00000000 00000313 40800000 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00248b83 00016691 00000000 00014791 00340980 0000c825 00010824 0001082a 3f800000 00000913 3f800000 00212827 0020c821 0020e721 0001478e 00380980 001cea21 00282120 0000842a 3f800000 00000041 0001c161 00000000 00000050 00000713 41000000 00000613 43000000 00000740 0019c870 00000872 0019c820 0001c960 00000000 00010a60 00000000 00292920 0001c961 00000000 00000073 00000050 0000c62b 3f000000 00000041 00000b13 00000000 00016d60 00000000 0001ad2b 3a800000 0001ad2a 3727c5ac 0001ad29 00000d8e 003c0980 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00014791 00380980 0034e721 0001478e 003c0a80 00012891 0009d980 0020e721 0001478e 00400a80 0000842a 3f800000 00000041 00000001
endkernel
kernel raw24 in=2 out=1 threads=10240 tg=64 tgbytes=16 layout=slots slots=mem:in0,input:in1
words 00000090 00000186 00000003 0000018d 00085680 0000218e 00000000 0000018d 00089f00 0000218e 00000001 0000018d 0008e780 0000218e 00000002 00000001
endkernel
kernel raw25 in=2 out=1 threads=393216 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 46400000 00040227 00004190 00002286 00003000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw26 in=1 out=1 threads=262144 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 42000000 000c4227 00004390 0000648f 0000842b 42000000 00102423 00008490 00008486 00000100 00048583 00006686 00003000 0014c683 00006786 00002000 0014e783 0000c591 00000000 0000a624 00000713 00000000 0014e870 00000872 0000a62a 3f800000 00000073 0000e68e 00060000 00000001
endkernel
kernel raw27 in=1 out=1 threads=131072 tg=32 tgbytes=4096 layout=slots slots=mem:in0
words 00000080 00000481 0000008f 00000113 42000000 00040227 00004790 0000e186 00000020 0000218f 00040123 00002190 0000e28f 00000313 43000000 000c4227 00004290 00004386 00000080 0000638f 0000e58f 000ca323 00006390 00006a86 00000020 00004b86 00040000 00296b83 00116b83 00004586 00060000 0028a583 0004a583 00004f86 00020000 0029ef83 0005ef83 00000886 00000020 00110883 0000e986 00000020 00112983 00011091 00143000 00013491 00142000 00008a8f 00014e2b 42000000 00001913 42000000 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00061000 00016791 00060000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 000a0000 00000073 00000050 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00061000 00016791 00060000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 000b0000 00000073 0001108e 00143000 00002a8f 00001a13 3f800000 00695b70 00001b72 0001348e 00183000 00000073 00000001
endkernel
kernel raw28 in=2 out=1 threads=131072 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw29 in=3 out=1 threads=131072 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00002386 00002000 00086383 0000048c 00000000 0000058d 00000000 0000648e 00000000 0000658e 00001000 00000001
endkernel
kernel raw30 in=3 out=1 threads=131072 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000191 00000000 0000028d 00000000 0000038c 00000000 00006326 00000413 3f800000 000c8423 000c2121 00104221 00082120 0000018e 00000000 00000001
endkernel
kernel raw31 in=4 out=1 threads=296960 tg=32 tgbytes=3136 layout=slots slots=mem:in0,u8:in1,lut:in2,input:in3
words 00000080 00000181 00002885 00000000 00000a13 41100000 00000c13 43880000 00000a40 00010b8f 00316d70 00000d72 0001098c 00000000 00016961 00000000 00000073 00010885 00000020 00000041 00000050 00002388 00000002 00002487 00000003 00000c86 00000002 00000286 00001000 00006689 00000004 00184283 00008689 00000002 00184283 00004785 00000800 00006589 00000005 0018a583 00000386 00000100 00006385 02440000 00002688 00000002 00186383 00006b85 00000080 00000813 00000000 00000913 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000f13 00000000 00001013 00000000 00001113 00000000 00001213 00000000 00001313 00000000 00001413 00000000 00001513 00000000 00001613 00000000 00001713 00000000 00001e13 41800000 00001e40 0000489d 00000000 0000e99d 00000000 00006a8b 00000000 00015c8f 00039c60 00000010 00016a8b 00000000 00015d8f 0003bd60 00000010 00010688 00000000 0000c687 0000000f 0000c48f 00008460 00000000 00708421 00012688 00000000 0000c687 0000000f 0000d88f 00031860 00000000 00771821 00010688 00000008 0000c687 0000000f 0000c58f 0000a560 00000000 0070a521 00012688 00000008 0000c687 0000000f 0000d98f 00033960 00000000 00773921 00010688 00000010 0000c687 0000000f 0000c68f 0000c660 00000000 0070c621 00012688 00000010 0000c687 0000000f 0000da8f 00035a60 00000000 00775a21 00010688 00000018 0000c687 0000000f 0000c78f 0000e760 00000000 0070e721 00012688 00000018 0000c687 0000000f 0000db8f 00037b60 00000000 00777b21 0000a09e 00000000 00008822 00000008 00030922 00000009 0004a822 00000008 00072922 00000009 0008c822 00000008 000b4922 00000009 000ce822 00000008 000f6922 00000009 0000a09e 00001000 00008a22 0000000a 00030b22 0000000b 0004aa22 0000000a 00072b22 0000000b 0008ca22 0000000a 000b4b22 0000000b 000cea22 0000000a 000f6b22 0000000b 0000a09e 00002000 00008c22 0000000c 00030d22 0000000d 0004ac22 0000000c 00072d22 0000000d 0008cc22 0000000c 000b4d22 0000000d 000cec22 0000000c 000f6d22 0000000d 0000a09e 00003000 00008e22 0000000e 00030f22 0000000f 0004ae22 0000000e 00072f22 0000000f 0008ce22 0000000e 000b4f22 0000000f 000cee22 0000000e 000f6f22 0000000f 0000a09e 00004000 00009022 00000010 00031122 00000011 0004b022 00000010 00073122 00000011 0008d022 00000010 000b5122 00000011 000cf022 00000010 000f7122 00000011 0000a09e 00005000 00009222 00000012 00031322 00000013 0004b222 00000012 00073322 00000013 0008d222 00000012 000b5322 00000013 000cf222 00000012 000f7322 00000013 0000a09e 00006000 00009422 00000014 00031522 00000015 0004b422 00000014 00073522 00000015 0008d422 00000014 000b5522 00000015 000cf422 00000014 000f7522 00000015 0000a09e 00007000 00009622 00000016 00031722 00000017 0004b622 00000016 00073722 00000017 0008d622 00000016 000b5722 00000017 000cf622 00000016 000f7722 00000017 00010688 00000004 0000c687 0000000f 0000c48f 00008460 00000000 00708421 00012688 00000004 0000c687 0000000f 0000d88f 00031860 00000000 00771821 00010688 0000000c 0000c687 0000000f 0000c58f 0000a560 00000000 0070a521 00012688 0000000c 0000c687 0000000f 0000d98f 00033960 00000000 00773921 00010688 00000014 0000c687 0000000f 0000c68f 0000c660 00000000 0070c621 00012688 00000014 0000c687 0000000f 0000da8f 00035a60 00000000 00775a21 00010688 0000001c 0000c687 0000000f 0000c78f 0000e760 00000000 0070e721 00012688 0000001c 0000c687 0000000f 0000db8f 00037b60 00000000 00777b21 0000a09e 00000010 00008822 00000008 00030922 00000009 0004a822 00000008 00072922 00000009 0008c822 00000008 000b4922 00000009 000ce822 00000008 000f6922 00000009 0000a09e 00001010 00008a22 0000000a 00030b22 0000000b 0004aa22 0000000a 00072b22 0000000b 0008ca22 0000000a 000b4b22 0000000b 000cea22 0000000a 000f6b22 0000000b 0000a09e 00002010 00008c22 0000000c 00030d22 0000000d 0004ac22 0000000c 00072d22 0000000d 0008cc22 0000000c 000b4d22 0000000d 000cec22 0000000c 000f6d22 0000000d 0000a09e 00003010 00008e22 0000000e 00030f22 0000000f 0004ae22 0000000e 00072f22 0000000f 0008ce22 0000000e 000b4f22 0000000f 000cee22 0000000e 000f6f22 0000000f 0000a09e 00004010 00009022 00000010 00031122 00000011 0004b022 00000010 00073122 00000011 0008d022 00000010 000b5122 00000011 000cf022 00000010 000f7122 00000011 0000a09e 00005010 00009222 00000012 00031322 00000013 0004b222 00000012 00073322 00000013 0008d222 00000012 000b5322 00000013 000cf222 00000012 000f7322 00000013 0000a09e 00006010 00009422 00000014 00031522 00000015 0004b422 00000014 00073522 00000015 0008d422 00000014 000b5522 00000015 000cf422 00000014 000f7522 00000015 0000a09e 00007010 00009622 00000016 00031722 00000017 0004b622 00000016 00073722 00000017 0008d622 00000016 000b5722 00000017 000cf622 00000016 000f7722 00000017 00004285 00000080 0000e785 00000080 00006385 00000008 00016b85 00000008 0000a585 00000100 00000041 0000208f 00000861 00000110 00000961 00000130 00000a61 00000150 00000b61 00000170 00000c61 00000190 00000d61 000001b0 00000e61 000001d0 00000f61 000001f0 00001061 00000210 00001161 00000230 00001261 00000250 00001361 00000270 00001461 00000290 00001561 000002b0 00001661 000002d0 00001761 000002f0 00000050 00000113 40a00000 00000213 41800000 00000140 00080370 00000372 00080420 00000560 00000110 00008660 00000110 0018a520 00000561 00000110 00000560 00000130 00008660 00000130 0018a520 00000561 00000130 00000560 00000150 00008660 00000150 0018a520 00000561 00000150 00000560 00000170 00008660 00000170 0018a520 00000561 00000170 00000560 00000190 00008660 00000190 0018a520 00000561 00000190 00000560 000001b0 00008660 000001b0 0018a520 00000561 000001b0 00000560 000001d0 00008660 000001d0 0018a520 00000561 000001d0 00000560 000001f0 00008660 000001f0 0018a520 00000561 000001f0 00000560 00000210 00008660 00000210 0018a520 00000561 00000210 00000560 00000230 00008660 00000230 0018a520 00000561 00000230 00000560 00000250 00008660 00000250 0018a520 00000561 00000250 00000560 00000270 00008660 00000270 0018a520 00000561 00000270 00000560 00000290 00008660 00000290 0018a520 00000561 00000290 00000560 000002b0 00008660 000002b0 0018a520 00000561 000002b0 00000560 000002d0 00008660 000002d0 0018a520 00000561 000002d0 00000560 000002f0 00008660 000002f0 0018a520 00000561 000002f0 00000073 00000050 0000422b 3f000000 00000041 00000213 3f800000 00080370 00000372 00000560 00000110 0001858e 00000000 00000560 00000130 0001858e 00000001 00000560 00000150 0001858e 00004880 00000560 00000170 0001858e 00004881 00000560 00000190 0001858e 00009100 00000560 000001b0 0001858e 00009101 00000560 000001d0 0001858e 0000d980 00000560 000001f0 0001858e 0000d981 00000560 00000210 0001858e 00012200 00000560 00000230 0001858e 00012201 00000560 00000250 0001858e 00016a80 00000560 00000270 0001858e 00016a81 00000560 00000290 0001858e 0001b300 00000560 000002b0 0001858e 0001b301 00000560 000002d0 0001858e 0001fb80 00000560 000002f0 0001858e 0001fb81 00000073 00000001
endkernel
kernel raw32 in=1 out=1 threads=4096 tg=256 tgbytes=4 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 41800000 000c4227 00004290 0000428f 0000422b 41800000 00082223 00000386 00004880 00000486 00002800 00000413 42200000 00000513 00000000 00000c13 40400000 00000d13 40800000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 46200000 001cc770 00000772 0000c590 0000a891 00052800 00146683 0000c685 00002000 0000c685 ffff2680 0000a786 00000004 00004b2a 00000000 00000d40 00316e70 00000f13 3f800000 0039ef23 00000e72 0000a986 00000003 00016a90 00292983 00012991 00327a00 0000ea91 00048800 00254822 00000008 00000073 00000f72 0000c991 00000000 0000ea91 00048800 00254822 00000008 00000073 0000c685 00004880 0000e785 00000001 00016b2a 3f800000 00000041 00148883 0001088e 00057180 00010e25 0001ce24 0001ce2a 3f800000 00000f13 3f800000 0039ee27 00390821 0001088e 0007f180 00000073 0000a52a 3f800000 00000041 00000080 00000181 0000018f 0000222a 3f000000 00000313 41800000 000c4227 00004290 0000428f 0000422b 41800000 00082223 00000386 00004880 00000486 00000080 00000413 3f800000 00000513 00000000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 43000000 001cc770 00000772 0000c590 00146683 0000c891 00004800 0000a991 00055100 00250820 00010828 00148683 0000c88e 000a7180 00000073 0000a52a 3f800000 00000041 00000001
endkernel
kernel raw33 in=1 out=1 threads=262144 tg=32 tgbytes=2048 layout=slots slots=mem:in0
words 00000080 0000008f 0000012a 3f000000 00000213 46000000 00082127 00002090 0000038f 0000632b 46000000 000c0423 00008190 0000852a 3f000000 00000213 42800000 0008a527 0000a290 0000468f 0000c62a 3f000000 00000213 41800000 0008c627 0000c390 00000481 00004891 00055000 00010824 00010825 00004991 00055080 00000d86 00028000 00000686 00000800 0008c683 00000f86 00020000 0005ef83 00000886 00200000 00002986 00000080 00250883 00110883 00006b86 00000080 00116b83 0001a585 00000000 00011091 000a7980 00011191 000a79a0 00011291 000a79c0 00011391 000a79e0 00008a8f 00014e2b 42000000 00001913 42000000 00001813 41800000 00014d2a 00000000 00001840 0004a983 00012391 0007f180 0000c191 000a7180 00202221 00004224 002caa83 00000413 00000000 00014591 00081180 000a0721 00142c21 000d8c21 0030f020 00014691 00081580 0040c422 00000004 00014591 000811a0 000a2721 00142c21 000d8c21 0030f120 00014691 000815a0 0044c422 00000004 00014591 000811c0 000a4721 00142c21 000d8c21 0030f220 00014691 000815c0 0048c422 00000004 00014591 000811e0 000a6721 00142c21 000d8c21 0030f320 00014691 000815e0 004cc422 00000004 0001a461 00000000 0001ad2a 42000000 0000a585 00002800 0000c685 00000080 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00001940 0001e560 00000000 0014e720 0001ef2a 3f800000 00000041 00008c86 00002800 00358c83 00058c83 00018391 0007f180 000d2722 00000007 00008e86 00002000 003dce83 0001c78e 002a7980 00000073 0001108e 001a7980 0001108e 000a7980 0001118e 001a79a0 0001118e 000a79a0 0001128e 001a79c0 0001128e 000a79c0 0001138e 001a79e0 0001138e 000a79e0 00000001
endkernel
kernel raw34 in=1 out=1 threads=32768 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 0000018f 00000213 41000000 00082127 00002290 00004386 00000008 0000048f 0000658f 00148423 00008390 00000181 00002e8f 00004486 00004880 00004586 00002000 00006686 00000400 0018a583 00000113 00000000 00000313 40800000 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00248b83 00016691 00000000 00014791 002a7980 0000c825 00010824 0001082a 3f800000 00000913 3f800000 00212827 0020c821 0020e721 0001478e 002c7980 001cea21 00282120 0000842a 3f800000 00000041 0001c161 00000000 00000050 00000713 41000000 00000613 43000000 00000740 0019c870 00000872 0019c820 0001c960 00000000 00010a60 00000000 00292920 0001c961 00000000 00000073 00000050 0000c62b 3f000000 00000041 00000b13 00000000 00016d60 00000000 0001ad2b 3a800000 0001ad2a 3727c5ac 0001ad29 00000d8e 002e7980 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00014791 002c7980 0034e721 0001478e 002e7a00 00012891 00055180 0020e721 0001478e 00307a00 0000842a 3f800000 00000041 00000001
endkernel
kernel raw35 in=4 out=1 threads=65536 tg=32 tgbytes=3136 layout=slots slots=mem:in0,u8:in1,lut:in2,input:in3
words 00000080 00000181 00002885 00000000 00000a13 41100000 00000c13 43880000 00000a40 00010b8f 00316d70 00000d72 0001098c 00000000 00016961 00000000 00000073 00010885 00000020 00000041 00000050 00002388 00000002 00002487 00000003 00000c86 00000002 00000286 00002000 00006689 00000004 00184283 00008689 00000002 00184283 00004785 00001000 00006589 00000005 0018a583 00000386 00000200 00006385 01000000 00002688 00000002 00186383 00006b85 00000100 00000813 00000000 00000913 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000f13 00000000 00001013 00000000 00001113 00000000 00001213 00000000 00001313 00000000 00001413 00000000 00001513 00000000 00001613 00000000 00001713 00000000 00001e13 42000000 00001e40 0000489d 00000000 0000e99d 00000000 00006a8b 00000000 00015c8f 00039c60 00000010 00016a8b 00000000 00015d8f 0003bd60 00000010 00010688 00000000 0000c687 0000000f 0000c48f 00008460 00000000 00708421 00012688 00000000 0000c687 0000000f 0000d88f 00031860 00000000 00771821 00010688 00000008 0000c687 0000000f 0000c58f 0000a560 00000000 0070a521 00012688 00000008 0000c687 0000000f 0000d98f 00033960 00000000 00773921 00010688 00000010 0000c687 0000000f 0000c68f 0000c660 00000000 0070c621 00012688 00000010 0000c687 0000000f 0000da8f 00035a60 00000000 00775a21 00010688 00000018 0000c687 0000000f 0000c78f 0000e760 00000000 0070e721 00012688 00000018 0000c687 0000000f 0000db8f 00037b60 00000000 00777b21 0000a09e 00000000 00008822 00000008 00030922 00000009 0004a822 00000008 00072922 00000009 0008c822 00000008 000b4922 00000009 000ce822 00000008 000f6922 00000009 0000a09e 00002000 00008a22 0000000a 00030b22 0000000b 0004aa22 0000000a 00072b22 0000000b 0008ca22 0000000a 000b4b22 0000000b 000cea22 0000000a 000f6b22 0000000b 0000a09e 00004000 00008c22 0000000c 00030d22 0000000d 0004ac22 0000000c 00072d22 0000000d 0008cc22 0000000c 000b4d22 0000000d 000cec22 0000000c 000f6d22 0000000d 0000a09e 00006000 00008e22 0000000e 00030f22 0000000f 0004ae22 0000000e 00072f22 0000000f 0008ce22 0000000e 000b4f22 0000000f 000cee22 0000000e 000f6f22 0000000f 0000a09e 00008000 00009022 00000010 00031122 00000011 0004b022 00000010 00073122 00000011 0008d022 00000010 000b5122 00000011 000cf022 00000010 000f7122 00000011 0000a09e 0000a000 00009222 00000012 00031322 00000013 0004b222 00000012 00073322 00000013 0008d222 00000012 000b5322 00000013 000cf222 00000012 000f7322 00000013 0000a09e 0000c000 00009422 00000014 00031522 00000015 0004b422 00000014 00073522 00000015 0008d422 00000014 000b5522 00000015 000cf422 00000014 000f7522 00000015 0000a09e 0000e000 00009622 00000016 00031722 00000017 0004b622 00000016 00073722 00000017 0008d622 00000016 000b5722 00000017 000cf622 00000016 000f7722 00000017 00010688 00000004 0000c687 0000000f 0000c48f 00008460 00000000 00708421 00012688 00000004 0000c687 0000000f 0000d88f 00031860 00000000 00771821 00010688 0000000c 0000c687 0000000f 0000c58f 0000a560 00000000 0070a521 00012688 0000000c 0000c687 0000000f 0000d98f 00033960 00000000 00773921 00010688 00000014 0000c687 0000000f 0000c68f 0000c660 00000000 0070c621 00012688 00000014 0000c687 0000000f 0000da8f 00035a60 00000000 00775a21 00010688 0000001c 0000c687 0000000f 0000c78f 0000e760 00000000 0070e721 00012688 0000001c 0000c687 0000000f 0000db8f 00037b60 00000000 00777b21 0000a09e 00000010 00008822 00000008 00030922 00000009 0004a822 00000008 00072922 00000009 0008c822 00000008 000b4922 00000009 000ce822 00000008 000f6922 00000009 0000a09e 00002010 00008a22 0000000a 00030b22 0000000b 0004aa22 0000000a 00072b22 0000000b 0008ca22 0000000a 000b4b22 0000000b 000cea22 0000000a 000f6b22 0000000b 0000a09e 00004010 00008c22 0000000c 00030d22 0000000d 0004ac22 0000000c 00072d22 0000000d 0008cc22 0000000c 000b4d22 0000000d 000cec22 0000000c 000f6d22 0000000d 0000a09e 00006010 00008e22 0000000e 00030f22 0000000f 0004ae22 0000000e 00072f22 0000000f 0008ce22 0000000e 000b4f22 0000000f 000cee22 0000000e 000f6f22 0000000f 0000a09e 00008010 00009022 00000010 00031122 00000011 0004b022 00000010 00073122 00000011 0008d022 00000010 000b5122 00000011 000cf022 00000010 000f7122 00000011 0000a09e 0000a010 00009222 00000012 00031322 00000013 0004b222 00000012 00073322 00000013 0008d222 00000012 000b5322 00000013 000cf222 00000012 000f7322 00000013 0000a09e 0000c010 00009422 00000014 00031522 00000015 0004b422 00000014 00073522 00000015 0008d422 00000014 000b5522 00000015 000cf422 00000014 000f7522 00000015 0000a09e 0000e010 00009622 00000016 00031722 00000017 0004b622 00000016 00073722 00000017 0008d622 00000016 000b5722 00000017 000cf622 00000016 000f7722 00000017 00004285 00000080 0000e785 00000080 00006385 00000008 00016b85 00000008 0000a585 00000100 00000041 0000208f 00000861 00000110 00000961 00000130 00000a61 00000150 00000b61 00000170 00000c61 00000190 00000d61 000001b0 00000e61 000001d0 00000f61 000001f0 00001061 00000210 00001161 00000230 00001261 00000250 00001361 00000270 00001461 00000290 00001561 000002b0 00001661 000002d0 00001761 000002f0 00000050 00000113 40a00000 00000213 41800000 00000140 00080370 00000372 00080420 00000560 00000110 00008660 00000110 0018a520 00000561 00000110 00000560 00000130 00008660 00000130 0018a520 00000561 00000130 00000560 00000150 00008660 00000150 0018a520 00000561 00000150 00000560 00000170 00008660 00000170 0018a520 00000561 00000170 00000560 00000190 00008660 00000190 0018a520 00000561 00000190 00000560 000001b0 00008660 000001b0 0018a520 00000561 000001b0 00000560 000001d0 00008660 000001d0 0018a520 00000561 000001d0 00000560 000001f0 00008660 000001f0 0018a520 00000561 000001f0 00000560 00000210 00008660 00000210 0018a520 00000561 00000210 00000560 00000230 00008660 00000230 0018a520 00000561 00000230 00000560 00000250 00008660 00000250 0018a520 00000561 00000250 00000560 00000270 00008660 00000270 0018a520 00000561 00000270 00000560 00000290 00008660 00000290 0018a520 00000561 00000290 00000560 000002b0 00008660 000002b0 0018a520 00000561 000002b0 00000560 000002d0 00008660 000002d0 0018a520 00000561 000002d0 00000560 000002f0 00008660 000002f0 0018a520 00000561 000002f0 00000073 00000050 0000422b 3f000000 00000041 00000213 3f800000 00080370 00000372 00000560 00000110 0001858e 00000000 00000560 00000130 0001858e 00000001 00000560 00000150 0001858e 00001000 00000560 00000170 0001858e 00001001 00000560 00000190 0001858e 00002000 00000560 000001b0 0001858e 00002001 00000560 000001d0 0001858e 00003000 00000560 000001f0 0001858e 00003001 00000560 00000210 0001858e 00004000 00000560 00000230 0001858e 00004001 00000560 00000250 0001858e 00005000 00000560 00000270 0001858e 00005001 00000560 00000290 0001858e 00006000 00000560 000002b0 0001858e 00006001 00000560 000002d0 0001858e 00007000 00000560 000002f0 0001858e 00007001 00000073 00000001
endkernel
kernel raw36 in=2 out=1 threads=10240 tg=64 tgbytes=16 layout=slots slots=mem:in0,input:in1
words 00000090 00000186 00000003 0000018d 0003ce80 0000218e 00000000 0000018d 00041700 0000218e 00000001 0000018d 00045f80 0000218e 00000002 00000001
endkernel
kernel raw37 in=2 out=1 threads=196608 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 46400000 00040227 00004190 00002286 00003000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw38 in=1 out=1 threads=131072 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 42000000 000c4227 00004390 0000648f 0000842b 42000000 00102423 00008490 00008486 00000100 00048583 00006686 00003000 0014c683 00006786 00002000 0014e783 0000c591 00000000 0000a624 00000713 00000000 0014e870 00000872 0000a62a 3f800000 00000073 0000e68e 00030000 00000001
endkernel
kernel raw39 in=1 out=1 threads=131072 tg=32 tgbytes=4096 layout=slots slots=mem:in0
words 00000080 00000481 0000008f 00000113 42000000 00040227 00004790 0000e186 00000020 0000218f 00040123 00002190 0000e28f 00000313 43000000 000c4227 00004290 00004386 00000080 0000638f 0000e58f 000ca323 00006390 00006a86 00000020 00004b86 00020000 00296b83 00116b83 00004586 00030000 0028a583 0004a583 00004f86 00010000 0029ef83 0005ef83 00000886 00000020 00110883 0000e986 00000020 00112983 00011091 000a2000 00013491 000a1000 00008a8f 00014e2b 42000000 00001913 42000000 00001813 41800000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00031000 00016791 00030000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000200 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41800000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000200 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 00050000 00000073 0001108e 000a2000 00002a8f 00001a13 3f800000 00695b70 00001b72 0001348e 000e2000 00000073 00000001
endkernel
kernel raw40 in=2 out=1 threads=65536 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw41 in=3 out=1 threads=65536 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00002386 00002000 00086383 0000048c 00000000 0000058d 00000000 0000648e 00000000 0000658e 00001000 00000001
endkernel
kernel raw42 in=3 out=1 threads=65536 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000191 00000000 0000028d 00000000 0000038c 00000000 00006326 00000413 3f800000 000c8423 000c2121 00104221 00082120 0000018e 00000000 00000001
endkernel
kernel raw43 in=1 out=1 threads=2048 tg=256 tgbytes=4 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 41000000 000c4227 00004290 0000428f 0000422b 41000000 00082223 00000386 00004880 00000486 00002800 00000413 42200000 00000513 00000000 00000c13 40400000 00000d13 40800000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 46200000 001cc770 00000772 0000c590 0000a891 0002e400 00146683 0000c685 00002000 0000c685 ffff2680 0000a786 00000004 00004b2a 00000000 00000d40 00316e70 00000f13 3f800000 0039ef23 00000e72 0000a986 00000003 00016a90 00292983 00012991 0029b1c0 0000ea91 00024400 00254822 00000008 00000073 00000f72 0000c991 00000000 0000ea91 00024400 00254822 00000008 00000073 0000c685 00004880 0000e785 00000001 00016b2a 3f800000 00000041 00148883 0001088e 00032d80 00010e25 0001ce24 0001ce2a 3f800000 00000f13 3f800000 0039ee27 00390821 0001088e 00046d80 00000073 0000a52a 3f800000 00000041 00000080 00000181 0000018f 0000222a 3f000000 00000313 41000000 000c4227 00004290 0000428f 0000422b 41000000 00082223 00000386 00004880 00000486 00000080 00000413 3f800000 00000513 00000000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 43000000 001cc770 00000772 0000c590 00146683 0000c891 00004800 0000a991 00030d00 00250820 00010828 00148683 0000c88e 0005ad80 00000073 0000a52a 3f800000 00000041 00000001
endkernel
kernel raw44 in=1 out=1 threads=262144 tg=32 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 0000008f 0000012a 3f000000 00000213 46000000 00082127 00002090 0000038f 0000632b 46000000 000c0423 00008190 0000852a 3f000000 00000213 42800000 0008a527 0000a290 0000468f 0000c62a 3f000000 00000213 41800000 0008c627 0000c390 00000481 00004891 00030c00 00010824 00010825 00004991 00030c80 00000d86 00014000 00000686 00000400 0008c683 00000f86 00010000 0005ef83 00000886 00200000 00002986 00000080 00250883 00110883 00006b86 00000080 00116b83 0001a585 00000000 00011091 0005b180 00011191 0005b1a0 00011291 0005b1c0 00011391 0005b1e0 00008a8f 00014e2b 42000000 00001913 42000000 00001813 41000000 00014d2a 00000000 00001840 0004a983 00012391 00046d80 0000c191 0005ad80 00202221 00004224 002caa83 00000413 00000000 00014591 00048d80 000a0721 00142c21 000d8c21 0030f020 00014691 00049180 0040c422 00000004 00014591 00048da0 000a2721 00142c21 000d8c21 0030f120 00014691 000491a0 0044c422 00000004 00014591 00048dc0 000a4721 00142c21 000d8c21 0030f220 00014691 000491c0 0048c422 00000004 00014591 00048de0 000a6721 00142c21 000d8c21 0030f320 00014691 000491e0 004cc422 00000004 0001a461 00000000 0001ad2a 42000000 0000a585 00002800 0000c685 00000080 00000041 00000050 00008a8f 00001a13 41000000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00001940 0001e560 00000000 0014e720 0001ef2a 3f800000 00000041 00008c86 00002800 00358c83 00058c83 00018391 00046d80 000d2722 00000007 00008e86 00002000 003dce83 0001c78e 0025b180 00000073 0001108e 0015b180 0001108e 0005b180 0001118e 0015b1a0 0001118e 0005b1a0 0001128e 0015b1c0 0001128e 0005b1c0 0001138e 0015b1e0 0001138e 0005b1e0 00000001
endkernel
kernel raw45 in=1 out=1 threads=16384 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 0000018f 00000213 41000000 00082127 00002290 00004386 00000008 0000048f 0000658f 00148423 00008390 00000181 00002e8f 00004486 00004880 00004586 00002000 00006686 00000400 0018a583 00000113 00000000 00000313 40800000 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00248b83 00016691 00000000 00014791 0025b180 0000c825 00010824 0001082a 3f800000 00000913 3f800000 00212827 0020c821 0020e721 0001478e 0026b180 001cea21 00282120 0000842a 3f800000 00000041 0001c161 00000000 00000050 00000713 41000000 00000613 43000000 00000740 0019c870 00000872 0019c820 0001c960 00000000 00010a60 00000000 00292920 0001c961 00000000 00000073 00000050 0000c62b 3f000000 00000041 00000b13 00000000 00016d60 00000000 0001ad2b 3a800000 0001ad2a 3727c5ac 0001ad29 00000d8e 0027b180 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00014791 0026b180 0034e721 0001478e 0027b1c0 00012891 00030d80 0020e721 0001478e 0028b1c0 0000842a 3f800000 00000041 00000001
endkernel
kernel raw46 in=2 out=1 threads=10240 tg=64 tgbytes=16 layout=slots slots=mem:in0,input:in1
words 00000090 00000186 00000003 0000018d 00018a80 0000218e 00000000 0000018d 0001d300 0000218e 00000001 0000018d 00021b80 0000218e 00000002 00000001
endkernel
kernel raw47 in=3 out=1 threads=393216 tg=32 tgbytes=1024 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000080 00000181 00000286 00001000 00044283 00002385 00000000 00000413 00000000 00000613 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000f13 00000000 00000813 43000000 00000840 0000458c 00000000 0000678d 00000000 001ca422 00000004 0000678d 00001000 001ca622 00000006 0000678d 00002000 001caa22 0000000a 0000678d 00003000 001cab22 0000000b 0000678d 00004000 001cac22 0000000c 0000678d 00005000 001cad22 0000000d 0000678d 00006000 001cae22 0000000e 0000678d 00007000 001caf22 0000000f 00004285 00000020 00006385 00000020 00000041 0000298f 00012461 00000000 00012661 00000020 00012a61 00000040 00012b61 00000060 00012c61 00000080 00012d61 000000a0 00012e61 000000c0 00012f61 000000e0 00000050 00000513 40a00000 00000813 41800000 00000540 00212770 00000772 00212a20 00012460 00000000 00014660 00000000 00188420 00012461 00000000 00012460 00000020 00014660 00000020 00188420 00012461 00000020 00012460 00000040 00014660 00000040 00188420 00012461 00000040 00012460 00000060 00014660 00000060 00188420 00012461 00000060 00012460 00000080 00014660 00000080 00188420 00012461 00000080 00012460 000000a0 00014660 000000a0 00188420 00012461 000000a0 00012460 000000c0 00014660 000000c0 00188420 00012461 000000c0 00012460 000000e0 00014660 000000e0 00188420 00012461 000000e0 00000073 00000050 0001082b 3f000000 00000041 00000813 3f800000 00212770 00000772 00012460 00000000 0000048e 00000000 00012460 00000020 0000048e 00003000 00012460 00000040 0000048e 00006000 00012460 00000060 0000048e 00009000 00012460 00000080 0000048e 0000c000 00012460 000000a0 0000048e 0000f000 00012460 000000c0 0000048e 00012000 00012460 000000e0 0000048e 00015000 00000073 00000001
endkernel
kernel raw48 in=2 out=1 threads=98304 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 46400000 00040227 00004190 00002286 00003000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw49 in=1 out=1 threads=65536 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 42000000 000c4227 00004390 0000648f 0000842b 42000000 00102423 00008490 00008486 00000100 00048583 00006686 00003000 0014c683 00006786 00002000 0014e783 0000c591 00000000 0000a624 00000713 00000000 0014e870 00000872 0000a62a 3f800000 00000073 0000e68e 00018000 00000001
endkernel
kernel raw50 in=1 out=1 threads=131072 tg=32 tgbytes=2048 layout=slots slots=mem:in0
words 00000080 00000481 0000008f 00000113 42000000 00040227 00004790 0000e186 00000020 0000218f 00040123 00002190 0000e28f 00000313 43000000 000c4227 00004290 00004386 00000080 0000638f 0000e58f 000ca323 00006390 00006a86 00000020 00004b86 00010000 00296b83 00116b83 00004586 00018000 0028a583 0004a583 00004f86 00008000 0029ef83 0005ef83 00000886 00000020 00110883 0000e986 00000020 00112983 00011091 00051800 00013491 00050800 00008a8f 00014e2b 42000000 00001913 42000000 00001813 41000000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00019000 00016791 00018000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 00000100 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 41000000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 00000100 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 00028000 00000073 0001108e 00051800 00002a8f 00001a13 3f800000 00695b70 00001b72 0001348e 00091800 00000073 00000001
endkernel
kernel raw51 in=3 out=1 threads=131072 tg=32 tgbytes=1024 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000080 00000181 00000286 00001000 00044283 00002385 00000000 00000413 00000000 00000613 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000f13 00000000 00000813 43000000 00000840 0000458c 00000000 0000678d 00000000 001ca422 00000004 0000678d 00001000 001ca622 00000006 0000678d 00002000 001caa22 0000000a 0000678d 00003000 001cab22 0000000b 0000678d 00004000 001cac22 0000000c 0000678d 00005000 001cad22 0000000d 0000678d 00006000 001cae22 0000000e 0000678d 00007000 001caf22 0000000f 00004285 00000020 00006385 00000020 00000041 0000298f 00012461 00000000 00012661 00000020 00012a61 00000040 00012b61 00000060 00012c61 00000080 00012d61 000000a0 00012e61 000000c0 00012f61 000000e0 00000050 00000513 40a00000 00000813 41800000 00000540 00212770 00000772 00212a20 00012460 00000000 00014660 00000000 00188420 00012461 00000000 00012460 00000020 00014660 00000020 00188420 00012461 00000020 00012460 00000040 00014660 00000040 00188420 00012461 00000040 00012460 00000060 00014660 00000060 00188420 00012461 00000060 00012460 00000080 00014660 00000080 00188420 00012461 00000080 00012460 000000a0 00014660 000000a0 00188420 00012461 000000a0 00012460 000000c0 00014660 000000c0 00188420 00012461 000000c0 00012460 000000e0 00014660 000000e0 00188420 00012461 000000e0 00000073 00000050 0001082b 3f000000 00000041 00000813 3f800000 00212770 00000772 00012460 00000000 0000048e 00000000 00012460 00000020 0000048e 00001000 00012460 00000040 0000048e 00002000 00012460 00000060 0000048e 00003000 00012460 00000080 0000048e 00004000 00012460 000000a0 0000048e 00005000 00012460 000000c0 0000048e 00006000 00012460 000000e0 0000048e 00007000 00000073 00000001
endkernel
kernel raw52 in=2 out=1 threads=32768 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw53 in=3 out=1 threads=32768 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00002386 00002000 00086383 0000048c 00000000 0000058d 00000000 0000648e 00000000 0000658e 00001000 00000001
endkernel
kernel raw54 in=3 out=1 threads=131072 tg=32 tgbytes=1024 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000080 00000181 00000286 00002000 00044283 00002385 00000000 00000413 00000000 00000613 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000f13 00000000 00000813 43800000 00000840 0000458c 00000000 0000678d 00000000 001ca422 00000004 0000678d 00002000 001ca622 00000006 0000678d 00004000 001caa22 0000000a 0000678d 00006000 001cab22 0000000b 0000678d 00008000 001cac22 0000000c 0000678d 0000a000 001cad22 0000000d 0000678d 0000c000 001cae22 0000000e 0000678d 0000e000 001caf22 0000000f 00004285 00000020 00006385 00000020 00000041 0000298f 00012461 00000000 00012661 00000020 00012a61 00000040 00012b61 00000060 00012c61 00000080 00012d61 000000a0 00012e61 000000c0 00012f61 000000e0 00000050 00000513 40a00000 00000813 41800000 00000540 00212770 00000772 00212a20 00012460 00000000 00014660 00000000 00188420 00012461 00000000 00012460 00000020 00014660 00000020 00188420 00012461 00000020 00012460 00000040 00014660 00000040 00188420 00012461 00000040 00012460 00000060 00014660 00000060 00188420 00012461 00000060 00012460 00000080 00014660 00000080 00188420 00012461 00000080 00012460 000000a0 00014660 000000a0 00188420 00012461 000000a0 00012460 000000c0 00014660 000000c0 00188420 00012461 000000c0 00012460 000000e0 00014660 000000e0 00188420 00012461 000000e0 00000073 00000050 0001082b 3f000000 00000041 00000813 3f800000 00212770 00000772 00012460 00000000 0000048e 00000000 00012460 00000020 0000048e 00001000 00012460 00000040 0000048e 00002000 00012460 00000060 0000048e 00003000 00012460 00000080 0000048e 00004000 00012460 000000a0 0000048e 00005000 00012460 000000c0 0000048e 00006000 00012460 000000e0 0000048e 00007000 00000073 00000001
endkernel
kernel raw55 in=3 out=1 threads=32768 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000191 00000000 0000028d 00000000 0000038c 00000000 00006326 00000413 3f800000 000c8423 000c2121 00104221 00082120 0000018e 00000000 00000001
endkernel
kernel raw56 in=4 out=1 threads=593920 tg=32 tgbytes=1600 layout=slots slots=mem:in0,u8:in1,lut:in2,input:in3
words 00000080 00000181 0000268f 00002885 00000000 00000a13 41100000 00000c13 43880000 00000a40 00010b8f 00316d70 00000d72 0001098c 00000000 00016961 00000000 00000073 00010885 00000020 00000041 00000050 00000286 00000800 00002687 0000000f 00184283 00000386 00000080 00006385 02440000 00002488 00000004 00008489 00000002 00002585 00000000 00000413 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000513 43000000 00000540 0000468b 00000000 0010c68a 0000c687 0000000f 0000c18f 00002160 00000000 0000678b 00000000 0000e28f 00004260 00000010 00082121 0000a38d 00000000 000c2422 00000004 0000a38d 00001000 000c2b22 0000000b 0000a38d 00002000 000c2c22 0000000c 0000a38d 00003000 000c2d22 0000000d 00004285 00000010 00006385 00000001 0000a585 00000020 00000041 0000268f 0000c461 00000110 00000050 00000513 40a00000 00000713 41800000 00000540 001cc870 00000872 001cc820 0000c960 00000110 00010a60 00000110 00292920 0000c961 00000110 00000073 00000050 0000e72b 3f000000 00000041 00000713 3f800000 001cc870 00000872 0000c960 00000110 0000098e 00000000 00000073 00000050 0000cb61 00000110 00000050 00000513 40a00000 00000713 41800000 00000540 001cc870 00000872 001cc820 0000c960 00000110 00010a60 00000110 00292920 0000c961 00000110 00000073 00000050 0000e72b 3f000000 00000041 00000713 3f800000 001cc870 00000872 0000c960 00000110 0000098e 00004880 00000073 00000050 0000cc61 00000110 00000050 00000513 40a00000 00000713 41800000 00000540 001cc870 00000872 001cc820 0000c960 00000110 00010a60 00000110 00292920 0000c961 00000110 00000073 00000050 0000e72b 3f000000 00000041 00000713 3f800000 001cc870 00000872 0000c960 00000110 0000098e 00009100 00000073 00000050 0000cd61 00000110 00000050 00000513 40a00000 00000713 41800000 00000540 001cc870 00000872 001cc820 0000c960 00000110 00010a60 00000110 00292920 0000c961 00000110 00000073 00000050 0000e72b 3f000000 00000041 00000713 3f800000 001cc870 00000872 0000c960 00000110 0000098e 0000d980 00000073 00000001
endkernel
kernel raw57 in=4 out=1 threads=148480 tg=32 tgbytes=1600 layout=slots slots=mem:in0,u8:in1,lut:in2,input:in3
words 00000080 00000181 00002885 00000000 00000a13 41100000 00000c13 43880000 00000a40 00010b8f 00316d70 00000d72 0001098c 00000000 00016961 00000000 00000073 00010885 00000020 00000041 00000050 00000286 00000004 00002388 00000002 00002487 00000003 00004586 00000800 00006a89 00000004 0028a583 00008a89 00000002 0028a583 00004686 00000080 0000c685 02440000 000cc683 00006789 00000005 0028e783 00000413 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000513 41800000 00000540 0000a89d 00000000 0000cc8b 00000000 00000813 00000000 0000a99d 00000800 0000cd8b 00000080 00000913 00000000 0000aa9d 00001000 0000ce8b 00000100 00000a13 00000000 0000ab9d 00001800 0000cf8b 00000180 00000b13 00000000 0000e38d 00000000 00010388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000010 00010388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000001 00010388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000011 00010388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000002 00010388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000012 00010388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000003 00010388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000013 00010388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0001828f 00004260 00000010 00204422 00000004 0001a28f 00004260 00000010 00244c22 0000000c 0001c28f 00004260 00000010 00284d22 0000000d 0001e28f 00004260 00000010 002c4e22 0000000e 0000a585 00000080 0000c685 00000008 0000e785 00000100 00000041 000084a0 00018ca0 0001ada0 0001cea0 000003a1 000004a2 000005a3 0000278f 0000618f 00000813 3f800000 00202970 00000972 0000828f 00004461 00000110 00004c61 00000130 00004d61 00000150 00004e61 00000170 00000073 00000050 0000a18f 00000213 42000000 00044227 00000413 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000613 00000000 00000513 40800000 00000540 0008c970 00000972 0000c160 00000110 00048420 0000c160 00000130 00058c20 0000c160 00000150 0005ad20 0000c160 00000170 0005ce20 00000073 0000c62a 3f800000 00000041 00000813 3f800000 0020e970 00000972 0000448e 00000000 00004c8e 00000001 00004d8e 00000002 00004e8e 00000003 00000073 00000001
endkernel
kernel raw58 in=1 out=1 threads=1280 tg=256 tgbytes=4 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 40a00000 000c4227 00004290 0000428f 0000422b 40a00000 00082223 00000386 00004880 00000486 00002800 00000413 42200000 00000513 00000000 00000c13 40400000 00000d13 40800000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 46200000 001cc770 00000772 0000c590 0000a891 00020a80 00146683 0000c685 00002000 0000c685 ffff2680 0000a786 00000004 00004b2a 00000000 00000d40 00316e70 00000f13 3f800000 0039ef23 00000e72 0000a986 00000003 00016a90 00292983 00012991 002666a8 0000ea91 00016a80 00254822 00000008 00000073 00000f72 0000c991 00000000 0000ea91 00016a80 00254822 00000008 00000073 0000c685 00004880 0000e785 00000001 00016b2a 3f800000 00000041 00148883 0001088e 00025400 00010e25 0001ce24 0001ce2a 3f800000 00000f13 3f800000 0039ee27 00390821 0001088e 00031c00 00000073 0000a52a 3f800000 00000041 00000080 00000181 0000018f 0000222a 3f000000 00000313 40a00000 000c4227 00004290 0000428f 0000422b 40a00000 00082223 00000386 00004880 00000486 00000080 00000413 3f800000 00000513 00000000 00000440 0000268f 0000a72b 43800000 001cc620 00000713 43000000 001cc770 00000772 0000c590 00146683 0000c891 00004800 0000a991 00023380 00250820 00010828 00148683 0000c88e 0003e400 00000073 0000a52a 3f800000 00000041 00000001
endkernel
kernel raw59 in=1 out=1 threads=262144 tg=32 tgbytes=640 layout=slots slots=mem:in0
words 00000080 0000008f 0000012a 3f000000 00000213 46000000 00082127 00002090 0000038f 0000632b 46000000 000c0423 00008190 0000852a 3f000000 00000213 42800000 0008a527 0000a290 0000468f 0000c62a 3f000000 00000213 41800000 0008c627 0000c390 00000481 00004891 00023280 00010824 00010825 00004991 00023300 00000d86 0000c800 00000686 00000280 0008c683 00000f86 0000a000 0005ef83 00000886 00200000 00002986 00000080 00250883 00110883 00006b86 00000080 00116b83 0001a585 00000000 00011091 0003e680 00011191 0003e6a0 00011291 0003e6c0 00011391 0003e6e0 00008a8f 00014e2b 42000000 00001913 42000000 00001813 40a00000 00014d2a 00000000 00001840 0004a983 00012391 00031c00 0000c191 0003e400 00202221 00004224 002caa83 00000413 00000000 00014591 00033c00 000a0721 00142c21 000d8c21 0030f020 00014691 00034000 0040c422 00000004 00014591 00033c20 000a2721 00142c21 000d8c21 0030f120 00014691 00034020 0044c422 00000004 00014591 00033c40 000a4721 00142c21 000d8c21 0030f220 00014691 00034040 0048c422 00000004 00014591 00033c60 000a6721 00142c21 000d8c21 0030f320 00014691 00034060 004cc422 00000004 0001a461 00000000 0001ad2a 42000000 0000a585 00002800 0000c685 00000080 00000041 00000050 00008a8f 00001a13 40a00000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00001940 0001e560 00000000 0014e720 0001ef2a 3f800000 00000041 00008c86 00002800 00358c83 00058c83 00018391 00031c00 000d2722 00000007 00008e86 00002000 003dce83 0001c78e 0023e680 00000073 0001108e 0013e680 0001108e 0003e680 0001118e 0013e6a0 0001118e 0003e6a0 0001128e 0013e6c0 0001128e 0003e6c0 0001138e 0013e6e0 0001138e 0003e6e0 00000001
endkernel
kernel raw60 in=1 out=1 threads=10240 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 0000018f 00000213 41000000 00082127 00002290 00004386 00000008 0000048f 0000658f 00148423 00008390 00000181 00002e8f 00004486 00004880 00004586 00002000 00006686 00000400 0018a583 00000113 00000000 00000313 40800000 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00248b83 00016691 00000000 00014791 0023e680 0000c825 00010824 0001082a 3f800000 00000913 3f800000 00212827 0020c821 0020e721 0001478e 00248680 001cea21 00282120 0000842a 3f800000 00000041 0001c161 00000000 00000050 00000713 41000000 00000613 43000000 00000740 0019c870 00000872 0019c820 0001c960 00000000 00010a60 00000000 00292920 0001c961 00000000 00000073 00000050 0000c62b 3f000000 00000041 00000b13 00000000 00016d60 00000000 0001ad2b 3a800000 0001ad2a 3727c5ac 0001ad29 00000d8e 00252680 00000413 00000000 00000340 0000872b 43800000 0038e720 0000e890 0020c983 0020aa83 00014791 00248680 0034e721 0001478e 002526a8 00012891 00023400 0020e721 0001478e 0025c6a8 0000842a 3f800000 00000041 00000001
endkernel
kernel raw61 in=4 out=1 threads=131072 tg=32 tgbytes=1600 layout=slots slots=mem:in0,u8:in1,lut:in2,input:in3
words 00000080 00000181 0000268f 00002885 00000000 00000a13 41100000 00000c13 43880000 00000a40 00010b8f 00316d70 00000d72 0001098c 00000000 00016961 00000000 00000073 00010885 00000020 00000041 00000050 00000286 00001000 00002687 0000000f 00184283 00000386 00000100 00006385 01000000 00002488 00000004 00008489 00000002 00002585 00000000 00000413 00000000 00000b13 00000000 00000c13 00000000 00000d13 00000000 00000513 43800000 00000540 0000468b 00000000 0010c68a 0000c687 0000000f 0000c18f 00002160 00000000 0000678b 00000000 0000e28f 00004260 00000010 00082121 0000a38d 00000000 000c2422 00000004 0000a38d 00002000 000c2b22 0000000b 0000a38d 00004000 000c2c22 0000000c 0000a38d 00006000 000c2d22 0000000d 00004285 00000010 00006385 00000001 0000a585 00000020 00000041 0000268f 0000c461 00000110 00000050 00000513 40a00000 00000713 41800000 00000540 001cc870 00000872 001cc820 0000c960 00000110 00010a60 00000110 00292920 0000c961 00000110 00000073 00000050 0000e72b 3f000000 00000041 00000713 3f800000 001cc870 00000872 0000c960 00000110 0000098e 00000000 00000073 00000050 0000cb61 00000110 00000050 00000513 40a00000 00000713 41800000 00000540 001cc870 00000872 001cc820 0000c960 00000110 00010a60 00000110 00292920 0000c961 00000110 00000073 00000050 0000e72b 3f000000 00000041 00000713 3f800000 001cc870 00000872 0000c960 00000110 0000098e 00001000 00000073 00000050 0000cc61 00000110 00000050 00000513 40a00000 00000713 41800000 00000540 001cc870 00000872 001cc820 0000c960 00000110 00010a60 00000110 00292920 0000c961 00000110 00000073 00000050 0000e72b 3f000000 00000041 00000713 3f800000 001cc870 00000872 0000c960 00000110 0000098e 00002000 00000073 00000050 0000cd61 00000110 00000050 00000513 40a00000 00000713 41800000 00000540 001cc870 00000872 001cc820 0000c960 00000110 00010a60 00000110 00292920 0000c961 00000110 00000073 00000050 0000e72b 3f000000 00000041 00000713 3f800000 001cc870 00000872 0000c960 00000110 0000098e 00003000 00000073 00000001
endkernel
kernel raw62 in=4 out=1 threads=32768 tg=32 tgbytes=1600 layout=slots slots=mem:in0,u8:in1,lut:in2,input:in3
words 00000080 00000181 00002885 00000000 00000a13 41100000 00000c13 43880000 00000a40 00010b8f 00316d70 00000d72 0001098c 00000000 00016961 00000000 00000073 00010885 00000020 00000041 00000050 00000286 00000004 00002388 00000002 00002487 00000003 00004586 00001000 00006a89 00000004 0028a583 00008a89 00000002 0028a583 00004686 00000100 0000c685 01000000 000cc683 00006789 00000005 0028e783 00000413 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000513 42000000 00000540 0000a89d 00000000 0000cc8b 00000000 00000813 00000000 0000a99d 00001000 0000cd8b 00000100 00000913 00000000 0000aa9d 00002000 0000ce8b 00000200 00000a13 00000000 0000ab9d 00003000 0000cf8b 00000300 00000b13 00000000 0000e38d 00000000 00010388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000010 00010388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000001 00010388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000011 00010388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000002 00010388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000012 00010388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000003 00010388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000013 00010388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0001828f 00004260 00000010 00204422 00000004 0001a28f 00004260 00000010 00244c22 0000000c 0001c28f 00004260 00000010 00284d22 0000000d 0001e28f 00004260 00000010 002c4e22 0000000e 0000a585 00000080 0000c685 00000008 0000e785 00000100 00000041 0000278f 0000e461 00000110 0000ec61 00000130 0000ed61 00000150 0000ee61 00000170 00000050 00000513 40a00000 00000813 41800000 00000540 0020e970 00000972 0020ea20 0000e160 00000110 00014260 00000110 00082120 0000e161 00000110 0000e160 00000130 00014260 00000130 00082120 0000e161 00000130 0000e160 00000150 00014260 00000150 00082120 0000e161 00000150 0000e160 00000170 00014260 00000170 00082120 0000e161 00000170 00000073 00000050 0001082b 3f000000 00000041 00000813 3f800000 0020e970 00000972 0000e160 00000110 0000418e 00000000 0000e160 00000130 0000418e 00000001 0000e160 00000150 0000418e 00000002 0000e160 00000170 0000418e 00000003 00000073 00000001
endkernel
kernel raw63 in=2 out=1 threads=10240 tg=64 tgbytes=16 layout=slots slots=mem:in0,input:in1
words 00000090 00000186 00000003 0000018d 0000b100 0000218e 00000000 0000018d 0000f980 0000218e 00000001 0000018d 00014200 0000218e 00000002 00000001
endkernel
kernel raw64 in=3 out=1 threads=393216 tg=32 tgbytes=640 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000080 00000181 00000286 00001000 00044283 00002385 00000000 00000413 00000000 00000613 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000813 43000000 00000840 0000458c 00000000 0000678d 00000000 001ca422 00000004 0000678d 00001000 001ca622 00000006 0000678d 00002000 001caa22 0000000a 0000678d 00003000 001cab22 0000000b 0000678d 00004000 001cac22 0000000c 00004285 00000020 00006385 00000020 00000041 0000298f 00012461 00000000 00012661 00000020 00012a61 00000040 00012b61 00000060 00012c61 00000080 00000050 00000513 40a00000 00000813 41800000 00000540 00212770 00000772 00212a20 00012460 00000000 00014660 00000000 00188420 00012461 00000000 00012460 00000020 00014660 00000020 00188420 00012461 00000020 00012460 00000040 00014660 00000040 00188420 00012461 00000040 00012460 00000060 00014660 00000060 00188420 00012461 00000060 00012460 00000080 00014660 00000080 00188420 00012461 00000080 00000073 00000050 0001082b 3f000000 00000041 00000813 3f800000 00212770 00000772 00012460 00000000 0000048e 00000000 00012460 00000020 0000048e 00003000 00012460 00000040 0000048e 00006000 00012460 00000060 0000048e 00009000 00012460 00000080 0000048e 0000c000 00000073 00000001
endkernel
kernel raw65 in=2 out=1 threads=61440 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 46400000 00040227 00004190 00002286 00003000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw66 in=1 out=1 threads=40960 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000080 00000181 0000018f 0000222a 3f000000 00000313 42000000 000c4227 00004390 0000648f 0000842b 42000000 00102423 00008490 00008486 00000100 00048583 00006686 00003000 0014c683 00006786 00002000 0014e783 0000c591 00000000 0000a624 00000713 00000000 0014e870 00000872 0000a62a 3f800000 00000073 0000e68e 0000f000 00000001
endkernel
kernel raw67 in=1 out=1 threads=131072 tg=32 tgbytes=1280 layout=slots slots=mem:in0
words 00000080 00000481 0000008f 00000113 42000000 00040227 00004790 0000e186 00000020 0000218f 00040123 00002190 0000e28f 00000313 43000000 000c4227 00004290 00004386 00000080 0000638f 0000e58f 000ca323 00006390 00006a86 00000020 00004b86 0000a000 00296b83 00116b83 00004586 0000f000 0028a583 0004a583 00004f86 00005000 0029ef83 0005ef83 00000886 00000020 00110883 0000e986 00000020 00112983 00011091 00033500 00013491 00032500 00008a8f 00014e2b 42000000 00001913 42000000 00001813 40a00000 00014d2a 00000000 00001840 0000a391 00002000 00000413 00000000 00000513 00000000 00016691 00010000 00016791 0000f000 001a9420 000cd022 00000010 0040e422 00000004 0050e522 00000005 0001a461 00000000 0001a561 000000a0 0001ad2a 42000000 00016b85 00002000 0000a585 00003000 00000041 00000050 00008a8f 00001a13 40a00000 00695b70 00001b72 0001cf2a 00000000 00000713 00000000 00000813 00000000 00001940 0001e560 00000000 0014e720 0001e560 000000a0 00150820 0001ef2a 3f800000 00000041 0001082a 358637bd 00000913 3f800000 00212827 0020e721 00008e86 00001000 003dce83 0001c78e 00019000 00000073 0001108e 00033500 00002a8f 00001a13 3f800000 00695b70 00001b72 0001348e 00073500 00000073 00000001
endkernel
kernel raw68 in=3 out=1 threads=131072 tg=32 tgbytes=640 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000080 00000181 00000286 00001000 00044283 00002385 00000000 00000413 00000000 00000613 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000813 43000000 00000840 0000458c 00000000 0000678d 00000000 001ca422 00000004 0000678d 00001000 001ca622 00000006 0000678d 00002000 001caa22 0000000a 0000678d 00003000 001cab22 0000000b 0000678d 00004000 001cac22 0000000c 00004285 00000020 00006385 00000020 00000041 0000298f 00012461 00000000 00012661 00000020 00012a61 00000040 00012b61 00000060 00012c61 00000080 00000050 00000513 40a00000 00000813 41800000 00000540 00212770 00000772 00212a20 00012460 00000000 00014660 00000000 00188420 00012461 00000000 00012460 00000020 00014660 00000020 00188420 00012461 00000020 00012460 00000040 00014660 00000040 00188420 00012461 00000040 00012460 00000060 00014660 00000060 00188420 00012461 00000060 00012460 00000080 00014660 00000080 00188420 00012461 00000080 00000073 00000050 0001082b 3f000000 00000041 00000813 3f800000 00212770 00000772 00012460 00000000 0000048e 00000000 00012460 00000020 0000048e 00001000 00012460 00000040 0000048e 00002000 00012460 00000060 0000048e 00003000 00012460 00000080 0000048e 00004000 00000073 00000001
endkernel
kernel raw69 in=2 out=1 threads=20480 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00000491 00000000 0000458c 00000000 00148420 0000048e 00000000 00000001
endkernel
kernel raw70 in=3 out=1 threads=20480 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000113 45800000 00040227 00004190 00002286 00001000 0000438f 000c0323 00006290 00002386 00002000 00086383 0000048c 00000000 0000058d 00000000 0000648e 00000000 0000658e 00001000 00000001
endkernel
kernel raw71 in=3 out=1 threads=131072 tg=32 tgbytes=640 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000080 00000181 00000286 00002000 00044283 00002385 00000000 00000413 00000000 00000613 00000000 00000a13 00000000 00000b13 00000000 00000c13 00000000 00000813 43800000 00000840 0000458c 00000000 0000678d 00000000 001ca422 00000004 0000678d 00002000 001ca622 00000006 0000678d 00004000 001caa22 0000000a 0000678d 00006000 001cab22 0000000b 0000678d 00008000 001cac22 0000000c 00004285 00000020 00006385 00000020 00000041 0000298f 00012461 00000000 00012661 00000020 00012a61 00000040 00012b61 00000060 00012c61 00000080 00000050 00000513 40a00000 00000813 41800000 00000540 00212770 00000772 00212a20 00012460 00000000 00014660 00000000 00188420 00012461 00000000 00012460 00000020 00014660 00000020 00188420 00012461 00000020 00012460 00000040 00014660 00000040 00188420 00012461 00000040 00012460 00000060 00014660 00000060 00188420 00012461 00000060 00012460 00000080 00014660 00000080 00188420 00012461 00000080 00000073 00000050 0001082b 3f000000 00000041 00000813 3f800000 00212770 00000772 00012460 00000000 0000048e 00000000 00012460 00000020 0000048e 00001000 00012460 00000040 0000048e 00002000 00012460 00000060 0000048e 00003000 00012460 00000080 0000048e 00004000 00000073 00000001
endkernel
kernel raw72 in=3 out=1 threads=20480 tg=64 tgbytes=16 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000090 00000191 00000000 0000028d 00000000 0000038c 00000000 00006326 00000413 3f800000 000c8423 000c2121 00104221 00082120 0000018e 00000000 00000001
endkernel
kernel raw73 in=1 out=1 threads=10240 tg=256 tgbytes=4 layout=slots slots=mem:in0
words 00000113 46000000 00040270 00000272 00000314 0000042a 46910000 00008315 00000073 00000113 46200000 00040270 00000272 0000032a 46000000 00006714 0000062b 40400000 0000012a 47a94000 00002114 0000052b 40800000 00000813 00000000 00000913 40400000 00000940 0020c320 0000632a 47bd4000 00006214 0020a420 0000842a 47328000 00008414 00088122 00000001 0001082a 3f800000 00000041 0000a42a 47328300 00008414 001c8122 00000001 00002325 00006324 0000632a 3f800000 00000513 3f800000 000ca327 000c2121 0000042a 46d10000 00008115 00000813 00000000 00000913 40000000 00000940 0020c320 0000632a 47bd4080 00006214 0020c420 0000842a 47bd4000 00008215 0001082a 3f800000 00000041 0000c32a 47bd4100 00006715 00000073 00000113 43000000 00040270 00000272 0000042a 46900000 00008314 0000052a 47120000 0000a514 00146320 00006328 0000042a 47108000 00008315 00000073 00000001
endkernel
kernel raw74 in=1 out=1 threads=8192 tg=256 tgbytes=4 layout=slots slots=mem:in0
words 00000c2a 498f9400 00018114 00000c2a 49909400 00018314 00002c2a 47110000 00018c14 00018c24 00018c25 00002b2a 47108000 00016b14 00316921 00012924 00000a2a 46d10000 00014a14 00002c2a 47118000 00018c14 00298421 0000622b 43000000 0000062b 43000000 00000713 00000000 00000813 43000000 00000840 00004d2a 47088000 001dad20 0001ad14 00004e2a 470c8000 001dce20 0001ce14 0000cf2a 47f94000 001def20 0001ec14 00258c21 0029ad21 00356c22 0000000c 0001ec15 0031c422 00000004 0000e72a 3f800000 00000041 00000d2a 49919400 0001a415 00000001
endkernel
kernel raw75 in=1 out=1 threads=2048 tg=256 tgbytes=1024 layout=slots slots=mem:in0
words 00000116 00000217 0000422b 44800000 00000313 00000000 00000813 40800000 00000713 00000000 00000840 0000e92b 43800000 00052920 00092920 00012a2a 49919400 00014a14 00012b2a 46910000 00016b14 00016c25 00018c24 00018c2a 3f800000 00000d13 3f800000 0031ac27 00316c21 00314a21 00012d2a 49929400 0001aa15 00294e21 00386320 0000e72a 3f800000 00000041 00002361 00000000 00000050 00000513 41000000 00000413 43000000 00000540 00102670 00000672 00102620 00002a60 00000000 0000cb60 00000000 002d4a20 00002a61 00000000 00000073 00000050 0000842b 3f000000 00000041 00000413 00000000 00008f60 00000000 0001ef2b 3a800000 0001ef2a 3727c5ac 0001ef29 00000813 40800000 00000713 00000000 00000840 0000e92b 43800000 00052920 00092920 00012a2a 49929400 00014a14 00012b2a 47128000 00016b14 003d4a21 002d4a21 00012d2a 49929400 0001aa15 0000e72a 3f800000 00000041 00000001
endkernel
kernel raw76 in=3 out=1 threads=393216 tg=32 tgbytes=128 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000080 00000181 0000268f 00000286 00001000 00044283 00002585 00000000 00000513 43000000 00000540 0000418c 00000000 0000a38d 00000000 000c2422 00000004 00004285 00000020 0000a585 00000020 00000041 0000c461 00000000 00000050 00000513 40a00000 00000713 41800000 00000540 001cc870 00000872 001cc820 0000c960 00000000 00010a60 00000000 00292920 0000c961 00000000 00000073 00000050 0000e72b 3f000000 00000041 00000713 3f800000 001cc870 00000872 0000c960 00000000 0000098e 00000000 00000073 00000001
endkernel
kernel raw77 in=2 out=1 threads=12288 tg=256 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000191 00000000 0000028c 00000000 00082120 0000018e 00000000 00000001
endkernel
kernel raw78 in=1 out=1 threads=4096 tg=32 tgbytes=16 layout=slots slots=mem:in0
words 00000e13 00000000 00000080 00000181 00000286 00000020 00004385 00001000 00004485 00002000 00048483 00008b91 00000000 00004785 00004000 00000886 00000400 00010885 00005000 00050883 00000c13 00000000 00000d13 00000000 00000a13 42000000 00000a40 00004191 00000000 00006291 00000000 0005cf70 003c2321 000c2f23 0001ef24 003c6320 0009cf70 003c4421 00104f23 0001ef24 003c8420 00010591 00000000 002c8522 00000005 0001058e 00000000 00146c22 0000000c 0000e691 00000000 0010c620 00186d22 0000000d 00004285 00000001 00006385 00000001 0000e785 00000001 00010885 00000020 00000041 0001ad2a 358637bd 00358c27 00000986 00000020 00052983 00012985 00003000 00012c8e 00000000 00000050 00000a86 00000020 00054a83 00014b85 00001000 00016191 00000000 0005cf70 003c2321 000c2f23 0001ef24 003c6320 00014c85 00004000 00018691 00000000 000cc620 0001868e 00000000 00000001
endkernel
kernel raw79 in=3 out=1 threads=131072 tg=32 tgbytes=128 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000080 00000181 0000268f 00000286 00001000 00044283 00002585 00000000 00000513 43000000 00000540 0000418c 00000000 0000a38d 00000000 000c2422 00000004 00004285 00000020 0000a585 00000020 00000041 0000c461 00000000 00000050 00000513 40a00000 00000713 41800000 00000540 001cc870 00000872 001cc820 0000c960 00000000 00010a60 00000000 00292920 0000c961 00000000 00000073 00000050 0000e72b 3f000000 00000041 00000713 3f800000 001cc870 00000872 0000c960 00000000 0000098e 00000000 00000073 00000001
endkernel
kernel raw80 in=2 out=1 threads=4096 tg=256 tgbytes=16 layout=slots slots=mem:in0,lut:in1
words 00000090 00000191 00000000 0000028c 00000000 00082120 0000018e 00000000 00000001
endkernel
kernel raw81 in=3 out=1 threads=131072 tg=32 tgbytes=128 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000080 00000181 0000268f 00000286 00002000 00044283 00002585 00000000 00000513 43800000 00000540 0000418c 00000000 0000a38d 00000000 000c2422 00000004 00004285 00000020 0000a585 00000020 00000041 0000c461 00000000 00000050 00000513 40a00000 00000713 41800000 00000540 001cc870 00000872 001cc820 0000c960 00000000 00010a60 00000000 00292920 0000c961 00000000 00000073 00000050 0000e72b 3f000000 00000041 00000713 3f800000 001cc870 00000872 0000c960 00000000 0000098e 00000000 00000073 00000001
endkernel
kernel raw82 in=4 out=1 threads=262144 tg=32 tgbytes=1600 layout=slots slots=mem:in0,u8:in1,lut:in2,input:in3
words 00000080 00000181 00002885 00000000 00000a13 41100000 00000c13 43880000 00000a40 00010b8f 00316d70 00000d72 0001098c 00000000 00016961 00000000 00000073 00010885 00000020 00000041 00000050 00000286 00000004 00002388 00000002 00002487 00000003 00004586 00000800 00006a89 00000004 0028a583 00008a89 00000002 0028a583 00004686 00000080 0000c685 04000000 000cc683 00006789 00000005 0028e783 00000413 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000513 41800000 00000540 0000a89d 00000000 0000cc8b 00000000 00000813 00000000 0000a99d 00000800 0000cd8b 00000080 00000913 00000000 0000aa9d 00001000 0000ce8b 00000100 00000a13 00000000 0000ab9d 00001800 0000cf8b 00000180 00000b13 00000000 0000e38d 00000000 00010388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000000 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000010 00010388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000004 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000001 00010388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000008 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000011 00010388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 0000000c 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000002 00010388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000010 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000012 00010388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000014 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000003 00010388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 00000018 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0000e38d 00000013 00010388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2822 00000008 00012388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2922 00000009 00014388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2a22 0000000a 00016388 0000001c 00006387 0000000f 0000618f 00002160 00000000 000c2b22 0000000b 0001828f 00004260 00000010 00204422 00000004 0001a28f 00004260 00000010 00244c22 0000000c 0001c28f 00004260 00000010 00284d22 0000000d 0001e28f 00004260 00000010 002c4e22 0000000e 0000a585 00000080 0000c685 00000008 0000e785 00000100 00000041 000084a0 00018ca0 0001ada0 0001cea0 000003a1 000004a2 000005a3 0000278f 0000618f 00000813 3f800000 00202970 00000972 0000828f 00004461 00000110 00004c61 00000130 00004d61 00000150 00004e61 00000170 00000073 00000050 0000a18f 00000213 42000000 00044227 00000413 00000000 00000c13 00000000 00000d13 00000000 00000e13 00000000 00000613 00000000 00000513 40800000 00000540 0008c970 00000972 0000c160 00000110 00048420 0000c160 00000130 00058c20 0000c160 00000150 0005ad20 0000c160 00000170 0005ce20 00000073 0000c62a 3f800000 00000041 00000813 3f800000 0020e970 00000972 0000448e 00000000 00004c8e 00000001 00004d8e 00000002 00004e8e 00000003 00000073 00000001
endkernel
phase chunk1 rows=128
node embedding ext fir.embedding token backbone_embeddings_weight_emb -> b1771
node session_CopyInto_-2891 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b23
node session_CopyInto_-2892 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b23
node session_CopyInto_-2893 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b41
node session_CopyInto_-2894 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b41
node session_CopyInto_-2895 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b59
node session_CopyInto_-2896 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b59
node session_CopyInto_-2897 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b77
node session_CopyInto_-2898 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b77
node session_CopyInto_-2899 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b95
node session_CopyInto_-2900 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b95
node session_CopyInto_-2901 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b113
node session_CopyInto_-2902 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b113
node session_CopyInto_-2903 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b131
node session_CopyInto_-2904 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b131
node session_CopyInto_-2905 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b149
node session_CopyInto_-2906 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b149
node session_CopyInto_-2908 session sessionCopyInto:srcOff=0,dstOff=16384,n=135168 b1758 -> b158
node session_CopyInto_-2909 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b185
node session_CopyInto_-2910 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b185
node session_CopyInto_-2911 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b203
node session_CopyInto_-2912 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b203
node session_CopyInto_-2913 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b221
node session_CopyInto_-2914 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b221
node session_CopyInto_-2915 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b239
node session_CopyInto_-2916 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b239
node session_CopyInto_-2917 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b257
node session_CopyInto_-2918 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b257
node session_CopyInto_-2919 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b275
node session_CopyInto_-2920 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b275
node session_CopyInto_-2921 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b293
node session_CopyInto_-2922 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b293
node session_CopyInto_-2923 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b311
node session_CopyInto_-2924 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b311
node session_CopyInto_-2925 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b329
node session_CopyInto_-2926 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b329
node session_CopyInto_-2927 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b347
node session_CopyInto_-2928 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b347
node session_CopyInto_-2929 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b365
node session_CopyInto_-2930 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b365
node session_CopyInto_-2931 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b383
node session_CopyInto_-2932 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b383
node session_CopyInto_-2933 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b401
node session_CopyInto_-2934 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b401
node session_CopyInto_-2935 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b419
node session_CopyInto_-2936 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b419
node session_CopyInto_-2937 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b437
node session_CopyInto_-2938 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b437
node session_CopyInto_-2939 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b455
node session_CopyInto_-2940 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b455
node session_CopyInto_-2941 session sessionCopyInto:srcOff=0,dstOff=16384,n=135168 b1758 -> b464
node session_CopyInto_-2942 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b490
node session_CopyInto_-2943 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b490
node session_CopyInto_-2944 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b508
node session_CopyInto_-2945 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b508
node session_CopyInto_-2946 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b526
node session_CopyInto_-2947 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b526
node session_CopyInto_-2948 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b544
node session_CopyInto_-2949 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b544
node session_CopyInto_-2950 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b562
node session_CopyInto_-2951 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b562
node session_CopyInto_-2952 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b580
node session_CopyInto_-2953 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b580
node session_CopyInto_-2954 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b598
node session_CopyInto_-2955 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b598
node session_CopyInto_-2956 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b616
node session_CopyInto_-2957 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b616
node session_CopyInto_-2958 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b634
node session_CopyInto_-2959 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b634
node session_CopyInto_-2960 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b652
node session_CopyInto_-2961 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b652
node session_CopyInto_-2962 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b670
node session_CopyInto_-2963 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b670
node session_CopyInto_-2964 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b688
node session_CopyInto_-2965 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b688
node session_CopyInto_-2966 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b706
node session_CopyInto_-2967 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b706
node session_CopyInto_-2968 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b724
node session_CopyInto_-2969 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b724
node session_CopyInto_-2970 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b742
node session_CopyInto_-2971 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b742
node session_CopyInto_-2972 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b760
node session_CopyInto_-2973 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b760
node session_CopyInto_-2974 session sessionCopyInto:srcOff=0,dstOff=16384,n=135168 b1758 -> b769
node session_CopyInto_-2975 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b795
node session_CopyInto_-2976 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b795
node session_CopyInto_-2977 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b813
node session_CopyInto_-2978 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b813
node session_CopyInto_-2979 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b831
node session_CopyInto_-2980 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b831
node session_CopyInto_-2981 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b849
node session_CopyInto_-2982 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b849
node session_CopyInto_-2983 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b867
node session_CopyInto_-2984 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b867
node session_CopyInto_-2985 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b885
node session_CopyInto_-2986 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b885
node session_CopyInto_-2987 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b903
node session_CopyInto_-2988 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b903
node session_CopyInto_-2989 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b921
node session_CopyInto_-2990 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b921
node session_CopyInto_-2991 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b939
node session_CopyInto_-2992 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b939
node session_CopyInto_-2993 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b957
node session_CopyInto_-2994 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b957
node session_CopyInto_-2995 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b975
node session_CopyInto_-2996 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b975
node session_CopyInto_-2997 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b993
node session_CopyInto_-2998 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b993
node session_CopyInto_-2999 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1011
node session_CopyInto_-3000 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1011
node session_CopyInto_-3001 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1029
node session_CopyInto_-3002 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1029
node session_CopyInto_-3003 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1047
node session_CopyInto_-3004 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1047
node session_CopyInto_-3005 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1065
node session_CopyInto_-3006 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1065
node session_CopyInto_-3007 session sessionCopyInto:srcOff=0,dstOff=16384,n=135168 b1758 -> b1074
node session_CopyInto_-3008 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1100
node session_CopyInto_-3009 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1100
node session_CopyInto_-3010 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1118
node session_CopyInto_-3011 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1118
node session_CopyInto_-3012 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1136
node session_CopyInto_-3013 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1136
node session_CopyInto_-3014 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1154
node session_CopyInto_-3015 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1154
node session_CopyInto_-3016 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1172
node session_CopyInto_-3017 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1172
node session_CopyInto_-3018 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1190
node session_CopyInto_-3019 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1190
node session_CopyInto_-3020 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1208
node session_CopyInto_-3021 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1208
node session_CopyInto_-3022 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1757 -> b1226
node session_CopyInto_-3023 session sessionCopyInto:srcOff=0,dstOff=127616,n=1048576 b1757 -> b1226
node session_CopyInto_-3051 session sessionCopyInto:srcOff=0,dstOff=0,n=524288 b1771 -> b1762
node session_RmsnormEnc_-3078 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b24 -> b1763
node session_Fp4GemmMetal_-3079 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1247 b1248 -> b1764
node session_CopyInto_-3080 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3081 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b23 -> b1767
node session_CopyInto_-3082 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b23 -> b1767
node session_CopyInto_-3083 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b23 -> b1767
node session_CopyInto_-3084 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b23 -> b1767
node dispatch287 micro raw0 b1767 -> b1767
node dispatch288 micro raw1 b1767 -> b1767
node dispatch289 micro raw2 b1767 -> b1767
node session_CopyInto_-3085 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3086 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1251 b1252 -> b1766
node dispatch290 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3087 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b23
node session_CopyInto_-3088 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b23
node session_VecOp_-3089 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3090 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b42 -> b1763
node session_Fp4GemmMetal_-3091 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1255 b1256 -> b1764
node session_CopyInto_-3092 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3093 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b41 -> b1767
node session_CopyInto_-3094 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b41 -> b1767
node session_CopyInto_-3095 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b41 -> b1767
node session_CopyInto_-3096 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b41 -> b1767
node dispatch291 micro raw0 b1767 -> b1767
node dispatch292 micro raw1 b1767 -> b1767
node dispatch293 micro raw2 b1767 -> b1767
node session_CopyInto_-3097 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3098 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1259 b1260 -> b1766
node dispatch294 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3099 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b41
node session_CopyInto_-3100 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b41
node session_VecOp_-3101 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3102 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b60 -> b1763
node session_Fp4GemmMetal_-3103 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1343 b1344 -> b1764
node session_CopyInto_-3104 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3105 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b59 -> b1767
node session_CopyInto_-3106 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b59 -> b1767
node session_CopyInto_-3107 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b59 -> b1767
node session_CopyInto_-3108 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b59 -> b1767
node dispatch295 micro raw0 b1767 -> b1767
node dispatch296 micro raw1 b1767 -> b1767
node dispatch297 micro raw2 b1767 -> b1767
node session_CopyInto_-3109 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3110 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1347 b1348 -> b1766
node dispatch298 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3111 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b59
node session_CopyInto_-3112 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b59
node session_VecOp_-3113 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3114 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b78 -> b1763
node session_Fp4GemmMetal_-3115 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1431 b1432 -> b1764
node session_CopyInto_-3116 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3117 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b77 -> b1767
node session_CopyInto_-3118 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b77 -> b1767
node session_CopyInto_-3119 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b77 -> b1767
node session_CopyInto_-3120 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b77 -> b1767
node dispatch299 micro raw0 b1767 -> b1767
node dispatch300 micro raw1 b1767 -> b1767
node dispatch301 micro raw2 b1767 -> b1767
node session_CopyInto_-3121 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3122 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1435 b1436 -> b1766
node dispatch302 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3123 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b77
node session_CopyInto_-3124 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b77
node session_VecOp_-3125 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3129 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b96 -> b1763
node session_Fp4GemmMetal_-3130 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1519 b1520 -> b1764
node session_CopyInto_-3131 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3132 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b95 -> b1767
node session_CopyInto_-3133 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b95 -> b1767
node session_CopyInto_-3134 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b95 -> b1767
node session_CopyInto_-3135 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b95 -> b1767
node dispatch303 micro raw0 b1767 -> b1767
node dispatch304 micro raw1 b1767 -> b1767
node dispatch305 micro raw2 b1767 -> b1767
node session_CopyInto_-3136 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3137 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1523 b1524 -> b1766
node dispatch306 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3138 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b95
node session_CopyInto_-3139 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b95
node session_VecOp_-3140 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3141 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b114 -> b1763
node session_Fp4GemmMetal_-3142 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1607 b1608 -> b1764
node session_CopyInto_-3143 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3144 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b113 -> b1767
node session_CopyInto_-3145 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b113 -> b1767
node session_CopyInto_-3146 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b113 -> b1767
node session_CopyInto_-3147 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b113 -> b1767
node dispatch307 micro raw0 b1767 -> b1767
node dispatch308 micro raw1 b1767 -> b1767
node dispatch309 micro raw2 b1767 -> b1767
node session_CopyInto_-3148 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3149 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1611 b1612 -> b1766
node dispatch310 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3150 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b113
node session_CopyInto_-3151 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b113
node session_VecOp_-3152 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3153 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b132 -> b1763
node session_Fp4GemmMetal_-3154 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1695 b1696 -> b1764
node session_CopyInto_-3155 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3156 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b131 -> b1767
node session_CopyInto_-3157 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b131 -> b1767
node session_CopyInto_-3158 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b131 -> b1767
node session_CopyInto_-3159 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b131 -> b1767
node dispatch311 micro raw0 b1767 -> b1767
node dispatch312 micro raw1 b1767 -> b1767
node dispatch313 micro raw2 b1767 -> b1767
node session_CopyInto_-3160 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3161 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1699 b1700 -> b1766
node dispatch314 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3162 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b131
node session_CopyInto_-3163 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b131
node session_VecOp_-3164 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3165 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b150 -> b1763
node session_Fp4GemmMetal_-3166 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1735 b1736 -> b1764
node session_CopyInto_-3167 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3168 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b149 -> b1767
node session_CopyInto_-3169 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b149 -> b1767
node session_CopyInto_-3170 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b149 -> b1767
node session_CopyInto_-3171 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b149 -> b1767
node dispatch315 micro raw0 b1767 -> b1767
node dispatch316 micro raw1 b1767 -> b1767
node dispatch317 micro raw2 b1767 -> b1767
node session_CopyInto_-3172 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3173 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1739 b1740 -> b1766
node dispatch318 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3174 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b149
node session_CopyInto_-3175 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b149
node session_MatmulNTAccelerated_-3176 session sessionMatmulNTAccelerated:M=128,K=4096,N=12288,bOff=0 b1763 b154 -> b1777
node dispatch319 micro raw4 b1777 b155 -> b1777
node session_CopyInto_-3178 session sessionCopyInto:srcOff=0,dstOff=0,n=1572864 b1777 -> b1774
node session_CopyInto_-3179 session sessionCopyInto:srcOff=16384,dstOff=5275648,n=4096 b158 -> b1774
node session_CopyInto_-3180 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b158 -> b1782
node dispatch320 micro raw5 b1774_21118976_2129920 b1782 -> b1774_21118976_2129920
node dispatch321 micro raw6 b1774 -> b1774
node dispatch322 micro raw7 b1774 -> b1774
node session_CopyInto_-3181 session sessionCopyInto:srcOff=2621440,dstOff=0,n=524288 b1774 -> b1778
node session_MatmulNTAccelerated_-3182 session sessionMatmulNTAccelerated:M=128,K=4096,N=4096,bOff=0 b1778 b156 -> b1779
node dispatch323 micro raw8 b1779 b157 -> b1779
node session_CopyInto_-3184 session sessionCopyInto:srcOff=5279744,dstOff=0,n=131072 b1774 -> b1782
node dispatch324 micro raw5 b158_81920_524288 b1782 -> b158_81920_524288
node session_CopyInto_-3185 session sessionCopyInto:srcOff=5541888,dstOff=16384,n=4096 b1774 -> b158
node dispatch325 micro raw9 b1780 b1766 b1779 -> b1780
node session_MatmulNTAccelerated_-3186 session sessionMatmulNTAccelerated:M=128,K=8192,N=4096,bOff=0 b1780 b165 -> b1781
node dispatch326 micro raw8 b1781 b166 -> b1781
node dispatch327 micro raw10 b1766 b1781 b1779 -> b1766
node session_VecOp_-3188 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3192 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b186 -> b1763
node session_Fp4GemmMetal_-3193 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1743 b1744 -> b1764
node session_CopyInto_-3194 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3195 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b185 -> b1767
node session_CopyInto_-3196 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b185 -> b1767
node session_CopyInto_-3197 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b185 -> b1767
node session_CopyInto_-3198 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b185 -> b1767
node dispatch328 micro raw0 b1767 -> b1767
node dispatch329 micro raw1 b1767 -> b1767
node dispatch330 micro raw2 b1767 -> b1767
node session_CopyInto_-3199 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3200 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1747 b1748 -> b1766
node dispatch331 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3201 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b185
node session_CopyInto_-3202 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b185
node session_VecOp_-3203 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3204 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b204 -> b1763
node session_Fp4GemmMetal_-3205 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1751 b1752 -> b1764
node session_CopyInto_-3206 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3207 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b203 -> b1767
node session_CopyInto_-3208 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b203 -> b1767
node session_CopyInto_-3209 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b203 -> b1767
node session_CopyInto_-3210 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b203 -> b1767
node dispatch332 micro raw0 b1767 -> b1767
node dispatch333 micro raw1 b1767 -> b1767
node dispatch334 micro raw2 b1767 -> b1767
node session_CopyInto_-3211 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3212 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1755 b1756 -> b1766
node dispatch335 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3213 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b203
node session_CopyInto_-3214 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b203
node session_VecOp_-3215 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3216 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b222 -> b1763
node session_Fp4GemmMetal_-3217 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1263 b1264 -> b1764
node session_CopyInto_-3218 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3219 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b221 -> b1767
node session_CopyInto_-3220 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b221 -> b1767
node session_CopyInto_-3221 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b221 -> b1767
node session_CopyInto_-3222 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b221 -> b1767
node dispatch336 micro raw0 b1767 -> b1767
node dispatch337 micro raw1 b1767 -> b1767
node dispatch338 micro raw2 b1767 -> b1767
node session_CopyInto_-3223 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3224 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1267 b1268 -> b1766
node dispatch339 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3225 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b221
node session_CopyInto_-3226 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b221
node session_VecOp_-3227 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3228 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b240 -> b1763
node session_Fp4GemmMetal_-3229 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1271 b1272 -> b1764
node session_CopyInto_-3230 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3231 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b239 -> b1767
node session_CopyInto_-3232 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b239 -> b1767
node session_CopyInto_-3233 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b239 -> b1767
node session_CopyInto_-3234 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b239 -> b1767
node dispatch340 micro raw0 b1767 -> b1767
node dispatch341 micro raw1 b1767 -> b1767
node dispatch342 micro raw2 b1767 -> b1767
node session_CopyInto_-3235 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3236 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1275 b1276 -> b1766
node dispatch343 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3237 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b239
node session_CopyInto_-3238 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b239
node session_VecOp_-3239 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3243 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b258 -> b1763
node session_Fp4GemmMetal_-3244 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1279 b1280 -> b1764
node session_CopyInto_-3245 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3246 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b257 -> b1767
node session_CopyInto_-3247 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b257 -> b1767
node session_CopyInto_-3248 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b257 -> b1767
node session_CopyInto_-3249 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b257 -> b1767
node dispatch344 micro raw0 b1767 -> b1767
node dispatch345 micro raw1 b1767 -> b1767
node dispatch346 micro raw2 b1767 -> b1767
node session_CopyInto_-3250 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3251 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1283 b1284 -> b1766
node dispatch347 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3252 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b257
node session_CopyInto_-3253 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b257
node session_VecOp_-3254 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3255 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b276 -> b1763
node session_Fp4GemmMetal_-3256 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1287 b1288 -> b1764
node session_CopyInto_-3257 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3258 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b275 -> b1767
node session_CopyInto_-3259 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b275 -> b1767
node session_CopyInto_-3260 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b275 -> b1767
node session_CopyInto_-3261 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b275 -> b1767
node dispatch348 micro raw0 b1767 -> b1767
node dispatch349 micro raw1 b1767 -> b1767
node dispatch350 micro raw2 b1767 -> b1767
node session_CopyInto_-3262 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3263 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1291 b1292 -> b1766
node dispatch351 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3264 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b275
node session_CopyInto_-3265 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b275
node session_VecOp_-3266 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3267 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b294 -> b1763
node session_Fp4GemmMetal_-3268 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1295 b1296 -> b1764
node session_CopyInto_-3269 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3270 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b293 -> b1767
node session_CopyInto_-3271 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b293 -> b1767
node session_CopyInto_-3272 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b293 -> b1767
node session_CopyInto_-3273 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b293 -> b1767
node dispatch352 micro raw0 b1767 -> b1767
node dispatch353 micro raw1 b1767 -> b1767
node dispatch354 micro raw2 b1767 -> b1767
node session_CopyInto_-3274 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3275 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1299 b1300 -> b1766
node dispatch355 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3276 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b293
node session_CopyInto_-3277 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b293
node session_VecOp_-3278 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3279 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b312 -> b1763
node session_Fp4GemmMetal_-3280 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1303 b1304 -> b1764
node session_CopyInto_-3281 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3282 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b311 -> b1767
node session_CopyInto_-3283 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b311 -> b1767
node session_CopyInto_-3284 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b311 -> b1767
node session_CopyInto_-3285 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b311 -> b1767
node dispatch356 micro raw0 b1767 -> b1767
node dispatch357 micro raw1 b1767 -> b1767
node dispatch358 micro raw2 b1767 -> b1767
node session_CopyInto_-3286 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3287 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1307 b1308 -> b1766
node dispatch359 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3288 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b311
node session_CopyInto_-3289 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b311
node session_VecOp_-3290 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3294 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b330 -> b1763
node session_Fp4GemmMetal_-3295 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1311 b1312 -> b1764
node session_CopyInto_-3296 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3297 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b329 -> b1767
node session_CopyInto_-3298 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b329 -> b1767
node session_CopyInto_-3299 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b329 -> b1767
node session_CopyInto_-3300 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b329 -> b1767
node dispatch360 micro raw0 b1767 -> b1767
node dispatch361 micro raw1 b1767 -> b1767
node dispatch362 micro raw2 b1767 -> b1767
node session_CopyInto_-3301 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3302 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1315 b1316 -> b1766
node dispatch363 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3303 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b329
node session_CopyInto_-3304 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b329
node session_VecOp_-3305 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3306 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b348 -> b1763
node session_Fp4GemmMetal_-3307 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1319 b1320 -> b1764
node session_CopyInto_-3308 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3309 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b347 -> b1767
node session_CopyInto_-3310 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b347 -> b1767
node session_CopyInto_-3311 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b347 -> b1767
node session_CopyInto_-3312 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b347 -> b1767
node dispatch364 micro raw0 b1767 -> b1767
node dispatch365 micro raw1 b1767 -> b1767
node dispatch366 micro raw2 b1767 -> b1767
node session_CopyInto_-3313 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3314 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1323 b1324 -> b1766
node dispatch367 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3315 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b347
node session_CopyInto_-3316 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b347
node session_VecOp_-3317 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3318 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b366 -> b1763
node session_Fp4GemmMetal_-3319 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1327 b1328 -> b1764
node session_CopyInto_-3320 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3321 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b365 -> b1767
node session_CopyInto_-3322 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b365 -> b1767
node session_CopyInto_-3323 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b365 -> b1767
node session_CopyInto_-3324 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b365 -> b1767
node dispatch368 micro raw0 b1767 -> b1767
node dispatch369 micro raw1 b1767 -> b1767
node dispatch370 micro raw2 b1767 -> b1767
node session_CopyInto_-3325 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3326 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1331 b1332 -> b1766
node dispatch371 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3327 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b365
node session_CopyInto_-3328 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b365
node session_VecOp_-3329 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3330 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b384 -> b1763
node session_Fp4GemmMetal_-3331 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1335 b1336 -> b1764
node session_CopyInto_-3332 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3333 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b383 -> b1767
node session_CopyInto_-3334 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b383 -> b1767
node session_CopyInto_-3335 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b383 -> b1767
node session_CopyInto_-3336 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b383 -> b1767
node dispatch372 micro raw0 b1767 -> b1767
node dispatch373 micro raw1 b1767 -> b1767
node dispatch374 micro raw2 b1767 -> b1767
node session_CopyInto_-3337 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3338 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1339 b1340 -> b1766
node dispatch375 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3339 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b383
node session_CopyInto_-3340 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b383
node session_VecOp_-3341 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3345 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b402 -> b1763
node session_Fp4GemmMetal_-3346 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1351 b1352 -> b1764
node session_CopyInto_-3347 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3348 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b401 -> b1767
node session_CopyInto_-3349 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b401 -> b1767
node session_CopyInto_-3350 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b401 -> b1767
node session_CopyInto_-3351 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b401 -> b1767
node dispatch376 micro raw0 b1767 -> b1767
node dispatch377 micro raw1 b1767 -> b1767
node dispatch378 micro raw2 b1767 -> b1767
node session_CopyInto_-3352 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3353 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1355 b1356 -> b1766
node dispatch379 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3354 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b401
node session_CopyInto_-3355 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b401
node session_VecOp_-3356 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3357 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b420 -> b1763
node session_Fp4GemmMetal_-3358 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1359 b1360 -> b1764
node session_CopyInto_-3359 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3360 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b419 -> b1767
node session_CopyInto_-3361 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b419 -> b1767
node session_CopyInto_-3362 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b419 -> b1767
node session_CopyInto_-3363 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b419 -> b1767
node dispatch380 micro raw0 b1767 -> b1767
node dispatch381 micro raw1 b1767 -> b1767
node dispatch382 micro raw2 b1767 -> b1767
node session_CopyInto_-3364 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3365 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1363 b1364 -> b1766
node dispatch383 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3366 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b419
node session_CopyInto_-3367 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b419
node session_VecOp_-3368 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3369 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b438 -> b1763
node session_Fp4GemmMetal_-3370 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1367 b1368 -> b1764
node session_CopyInto_-3371 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3372 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b437 -> b1767
node session_CopyInto_-3373 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b437 -> b1767
node session_CopyInto_-3374 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b437 -> b1767
node session_CopyInto_-3375 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b437 -> b1767
node dispatch384 micro raw0 b1767 -> b1767
node dispatch385 micro raw1 b1767 -> b1767
node dispatch386 micro raw2 b1767 -> b1767
node session_CopyInto_-3376 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3377 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1371 b1372 -> b1766
node dispatch387 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3378 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b437
node session_CopyInto_-3379 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b437
node session_VecOp_-3380 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3381 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b456 -> b1763
node session_Fp4GemmMetal_-3382 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1375 b1376 -> b1764
node session_CopyInto_-3383 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3384 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b455 -> b1767
node session_CopyInto_-3385 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b455 -> b1767
node session_CopyInto_-3386 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b455 -> b1767
node session_CopyInto_-3387 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b455 -> b1767
node dispatch388 micro raw0 b1767 -> b1767
node dispatch389 micro raw1 b1767 -> b1767
node dispatch390 micro raw2 b1767 -> b1767
node session_CopyInto_-3388 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3389 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1379 b1380 -> b1766
node dispatch391 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3390 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b455
node session_CopyInto_-3391 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b455
node session_MatmulNTAccelerated_-3392 session sessionMatmulNTAccelerated:M=128,K=4096,N=12288,bOff=0 b1763 b460 -> b1777
node dispatch392 micro raw4 b1777 b461 -> b1777
node session_CopyInto_-3394 session sessionCopyInto:srcOff=0,dstOff=0,n=1572864 b1777 -> b1774
node session_CopyInto_-3395 session sessionCopyInto:srcOff=16384,dstOff=5275648,n=4096 b464 -> b1774
node session_CopyInto_-3396 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b464 -> b1782
node dispatch393 micro raw5 b1774_21118976_2129920 b1782 -> b1774_21118976_2129920
node dispatch394 micro raw6 b1774 -> b1774
node dispatch395 micro raw7 b1774 -> b1774
node session_CopyInto_-3397 session sessionCopyInto:srcOff=2621440,dstOff=0,n=524288 b1774 -> b1778
node session_MatmulNTAccelerated_-3398 session sessionMatmulNTAccelerated:M=128,K=4096,N=4096,bOff=0 b1778 b462 -> b1779
node dispatch396 micro raw8 b1779 b463 -> b1779
node session_CopyInto_-3400 session sessionCopyInto:srcOff=5279744,dstOff=0,n=131072 b1774 -> b1782
node dispatch397 micro raw5 b464_81920_524288 b1782 -> b464_81920_524288
node session_CopyInto_-3401 session sessionCopyInto:srcOff=5541888,dstOff=16384,n=4096 b1774 -> b464
node dispatch398 micro raw9 b1780 b1766 b1779 -> b1780
node session_MatmulNTAccelerated_-3402 session sessionMatmulNTAccelerated:M=128,K=8192,N=4096,bOff=0 b1780 b470 -> b1781
node dispatch399 micro raw8 b1781 b471 -> b1781
node dispatch400 micro raw10 b1766 b1781 b1779 -> b1766
node session_VecOp_-3404 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3408 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b491 -> b1763
node session_Fp4GemmMetal_-3409 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1383 b1384 -> b1764
node session_CopyInto_-3410 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3411 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b490 -> b1767
node session_CopyInto_-3412 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b490 -> b1767
node session_CopyInto_-3413 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b490 -> b1767
node session_CopyInto_-3414 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b490 -> b1767
node dispatch401 micro raw0 b1767 -> b1767
node dispatch402 micro raw1 b1767 -> b1767
node dispatch403 micro raw2 b1767 -> b1767
node session_CopyInto_-3415 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3416 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1387 b1388 -> b1766
node dispatch404 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3417 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b490
node session_CopyInto_-3418 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b490
node session_VecOp_-3419 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3420 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b509 -> b1763
node session_Fp4GemmMetal_-3421 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1391 b1392 -> b1764
node session_CopyInto_-3422 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3423 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b508 -> b1767
node session_CopyInto_-3424 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b508 -> b1767
node session_CopyInto_-3425 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b508 -> b1767
node session_CopyInto_-3426 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b508 -> b1767
node dispatch405 micro raw0 b1767 -> b1767
node dispatch406 micro raw1 b1767 -> b1767
node dispatch407 micro raw2 b1767 -> b1767
node session_CopyInto_-3427 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3428 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1395 b1396 -> b1766
node dispatch408 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3429 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b508
node session_CopyInto_-3430 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b508
node session_VecOp_-3431 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3432 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b527 -> b1763
node session_Fp4GemmMetal_-3433 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1399 b1400 -> b1764
node session_CopyInto_-3434 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3435 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b526 -> b1767
node session_CopyInto_-3436 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b526 -> b1767
node session_CopyInto_-3437 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b526 -> b1767
node session_CopyInto_-3438 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b526 -> b1767
node dispatch409 micro raw0 b1767 -> b1767
node dispatch410 micro raw1 b1767 -> b1767
node dispatch411 micro raw2 b1767 -> b1767
node session_CopyInto_-3439 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3440 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1403 b1404 -> b1766
node dispatch412 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3441 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b526
node session_CopyInto_-3442 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b526
node session_VecOp_-3443 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3444 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b545 -> b1763
node session_Fp4GemmMetal_-3445 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1407 b1408 -> b1764
node session_CopyInto_-3446 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3447 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b544 -> b1767
node session_CopyInto_-3448 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b544 -> b1767
node session_CopyInto_-3449 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b544 -> b1767
node session_CopyInto_-3450 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b544 -> b1767
node dispatch413 micro raw0 b1767 -> b1767
node dispatch414 micro raw1 b1767 -> b1767
node dispatch415 micro raw2 b1767 -> b1767
node session_CopyInto_-3451 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3452 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1411 b1412 -> b1766
node dispatch416 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3453 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b544
node session_CopyInto_-3454 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b544
node session_VecOp_-3455 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3459 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b563 -> b1763
node session_Fp4GemmMetal_-3460 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1415 b1416 -> b1764
node session_CopyInto_-3461 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3462 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b562 -> b1767
node session_CopyInto_-3463 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b562 -> b1767
node session_CopyInto_-3464 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b562 -> b1767
node session_CopyInto_-3465 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b562 -> b1767
node dispatch417 micro raw0 b1767 -> b1767
node dispatch418 micro raw1 b1767 -> b1767
node dispatch419 micro raw2 b1767 -> b1767
node session_CopyInto_-3466 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3467 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1419 b1420 -> b1766
node dispatch420 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3468 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b562
node session_CopyInto_-3469 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b562
node session_VecOp_-3470 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3471 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b581 -> b1763
node session_Fp4GemmMetal_-3472 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1423 b1424 -> b1764
node session_CopyInto_-3473 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3474 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b580 -> b1767
node session_CopyInto_-3475 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b580 -> b1767
node session_CopyInto_-3476 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b580 -> b1767
node session_CopyInto_-3477 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b580 -> b1767
node dispatch421 micro raw0 b1767 -> b1767
node dispatch422 micro raw1 b1767 -> b1767
node dispatch423 micro raw2 b1767 -> b1767
node session_CopyInto_-3478 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3479 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1427 b1428 -> b1766
node dispatch424 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3480 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b580
node session_CopyInto_-3481 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b580
node session_VecOp_-3482 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3483 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b599 -> b1763
node session_Fp4GemmMetal_-3484 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1439 b1440 -> b1764
node session_CopyInto_-3485 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3486 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b598 -> b1767
node session_CopyInto_-3487 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b598 -> b1767
node session_CopyInto_-3488 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b598 -> b1767
node session_CopyInto_-3489 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b598 -> b1767
node dispatch425 micro raw0 b1767 -> b1767
node dispatch426 micro raw1 b1767 -> b1767
node dispatch427 micro raw2 b1767 -> b1767
node session_CopyInto_-3490 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3491 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1443 b1444 -> b1766
node dispatch428 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3492 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b598
node session_CopyInto_-3493 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b598
node session_VecOp_-3494 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3495 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b617 -> b1763
node session_Fp4GemmMetal_-3496 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1447 b1448 -> b1764
node session_CopyInto_-3497 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3498 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b616 -> b1767
node session_CopyInto_-3499 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b616 -> b1767
node session_CopyInto_-3500 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b616 -> b1767
node session_CopyInto_-3501 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b616 -> b1767
node dispatch429 micro raw0 b1767 -> b1767
node dispatch430 micro raw1 b1767 -> b1767
node dispatch431 micro raw2 b1767 -> b1767
node session_CopyInto_-3502 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3503 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1451 b1452 -> b1766
node dispatch432 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3504 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b616
node session_CopyInto_-3505 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b616
node session_VecOp_-3506 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3510 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b635 -> b1763
node session_Fp4GemmMetal_-3511 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1455 b1456 -> b1764
node session_CopyInto_-3512 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3513 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b634 -> b1767
node session_CopyInto_-3514 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b634 -> b1767
node session_CopyInto_-3515 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b634 -> b1767
node session_CopyInto_-3516 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b634 -> b1767
node dispatch433 micro raw0 b1767 -> b1767
node dispatch434 micro raw1 b1767 -> b1767
node dispatch435 micro raw2 b1767 -> b1767
node session_CopyInto_-3517 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3518 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1459 b1460 -> b1766
node dispatch436 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3519 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b634
node session_CopyInto_-3520 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b634
node session_VecOp_-3521 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3522 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b653 -> b1763
node session_Fp4GemmMetal_-3523 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1463 b1464 -> b1764
node session_CopyInto_-3524 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3525 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b652 -> b1767
node session_CopyInto_-3526 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b652 -> b1767
node session_CopyInto_-3527 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b652 -> b1767
node session_CopyInto_-3528 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b652 -> b1767
node dispatch437 micro raw0 b1767 -> b1767
node dispatch438 micro raw1 b1767 -> b1767
node dispatch439 micro raw2 b1767 -> b1767
node session_CopyInto_-3529 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3530 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1467 b1468 -> b1766
node dispatch440 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3531 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b652
node session_CopyInto_-3532 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b652
node session_VecOp_-3533 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3534 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b671 -> b1763
node session_Fp4GemmMetal_-3535 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1471 b1472 -> b1764
node session_CopyInto_-3536 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3537 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b670 -> b1767
node session_CopyInto_-3538 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b670 -> b1767
node session_CopyInto_-3539 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b670 -> b1767
node session_CopyInto_-3540 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b670 -> b1767
node dispatch441 micro raw0 b1767 -> b1767
node dispatch442 micro raw1 b1767 -> b1767
node dispatch443 micro raw2 b1767 -> b1767
node session_CopyInto_-3541 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3542 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1475 b1476 -> b1766
node dispatch444 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3543 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b670
node session_CopyInto_-3544 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b670
node session_VecOp_-3545 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3546 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b689 -> b1763
node session_Fp4GemmMetal_-3547 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1479 b1480 -> b1764
node session_CopyInto_-3548 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3549 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b688 -> b1767
node session_CopyInto_-3550 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b688 -> b1767
node session_CopyInto_-3551 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b688 -> b1767
node session_CopyInto_-3552 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b688 -> b1767
node dispatch445 micro raw0 b1767 -> b1767
node dispatch446 micro raw1 b1767 -> b1767
node dispatch447 micro raw2 b1767 -> b1767
node session_CopyInto_-3553 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3554 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1483 b1484 -> b1766
node dispatch448 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3555 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b688
node session_CopyInto_-3556 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b688
node session_VecOp_-3557 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3561 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b707 -> b1763
node session_Fp4GemmMetal_-3562 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1487 b1488 -> b1764
node session_CopyInto_-3563 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3564 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b706 -> b1767
node session_CopyInto_-3565 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b706 -> b1767
node session_CopyInto_-3566 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b706 -> b1767
node session_CopyInto_-3567 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b706 -> b1767
node dispatch449 micro raw0 b1767 -> b1767
node dispatch450 micro raw1 b1767 -> b1767
node dispatch451 micro raw2 b1767 -> b1767
node session_CopyInto_-3568 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3569 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1491 b1492 -> b1766
node dispatch452 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3570 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b706
node session_CopyInto_-3571 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b706
node session_VecOp_-3572 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3573 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b725 -> b1763
node session_Fp4GemmMetal_-3574 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1495 b1496 -> b1764
node session_CopyInto_-3575 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3576 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b724 -> b1767
node session_CopyInto_-3577 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b724 -> b1767
node session_CopyInto_-3578 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b724 -> b1767
node session_CopyInto_-3579 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b724 -> b1767
node dispatch453 micro raw0 b1767 -> b1767
node dispatch454 micro raw1 b1767 -> b1767
node dispatch455 micro raw2 b1767 -> b1767
node session_CopyInto_-3580 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3581 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1499 b1500 -> b1766
node dispatch456 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3582 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b724
node session_CopyInto_-3583 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b724
node session_VecOp_-3584 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3585 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b743 -> b1763
node session_Fp4GemmMetal_-3586 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1503 b1504 -> b1764
node session_CopyInto_-3587 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3588 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b742 -> b1767
node session_CopyInto_-3589 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b742 -> b1767
node session_CopyInto_-3590 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b742 -> b1767
node session_CopyInto_-3591 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b742 -> b1767
node dispatch457 micro raw0 b1767 -> b1767
node dispatch458 micro raw1 b1767 -> b1767
node dispatch459 micro raw2 b1767 -> b1767
node session_CopyInto_-3592 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3593 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1507 b1508 -> b1766
node dispatch460 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3594 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b742
node session_CopyInto_-3595 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b742
node session_VecOp_-3596 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3597 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b761 -> b1763
node session_Fp4GemmMetal_-3598 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1511 b1512 -> b1764
node session_CopyInto_-3599 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3600 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b760 -> b1767
node session_CopyInto_-3601 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b760 -> b1767
node session_CopyInto_-3602 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b760 -> b1767
node session_CopyInto_-3603 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b760 -> b1767
node dispatch461 micro raw0 b1767 -> b1767
node dispatch462 micro raw1 b1767 -> b1767
node dispatch463 micro raw2 b1767 -> b1767
node session_CopyInto_-3604 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3605 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1515 b1516 -> b1766
node dispatch464 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3606 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b760
node session_CopyInto_-3607 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b760
node session_MatmulNTAccelerated_-3608 session sessionMatmulNTAccelerated:M=128,K=4096,N=12288,bOff=0 b1763 b765 -> b1777
node dispatch465 micro raw4 b1777 b766 -> b1777
node session_CopyInto_-3610 session sessionCopyInto:srcOff=0,dstOff=0,n=1572864 b1777 -> b1774
node session_CopyInto_-3611 session sessionCopyInto:srcOff=16384,dstOff=5275648,n=4096 b769 -> b1774
node session_CopyInto_-3612 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b769 -> b1782
node dispatch466 micro raw5 b1774_21118976_2129920 b1782 -> b1774_21118976_2129920
node dispatch467 micro raw6 b1774 -> b1774
node dispatch468 micro raw7 b1774 -> b1774
node session_CopyInto_-3613 session sessionCopyInto:srcOff=2621440,dstOff=0,n=524288 b1774 -> b1778
node session_MatmulNTAccelerated_-3614 session sessionMatmulNTAccelerated:M=128,K=4096,N=4096,bOff=0 b1778 b767 -> b1779
node dispatch469 micro raw8 b1779 b768 -> b1779
node session_CopyInto_-3616 session sessionCopyInto:srcOff=5279744,dstOff=0,n=131072 b1774 -> b1782
node dispatch470 micro raw5 b769_81920_524288 b1782 -> b769_81920_524288
node session_CopyInto_-3617 session sessionCopyInto:srcOff=5541888,dstOff=16384,n=4096 b1774 -> b769
node dispatch471 micro raw9 b1780 b1766 b1779 -> b1780
node session_MatmulNTAccelerated_-3618 session sessionMatmulNTAccelerated:M=128,K=8192,N=4096,bOff=0 b1780 b775 -> b1781
node dispatch472 micro raw8 b1781 b776 -> b1781
node dispatch473 micro raw10 b1766 b1781 b1779 -> b1766
node session_VecOp_-3620 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3624 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b796 -> b1763
node session_Fp4GemmMetal_-3625 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1527 b1528 -> b1764
node session_CopyInto_-3626 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3627 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b795 -> b1767
node session_CopyInto_-3628 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b795 -> b1767
node session_CopyInto_-3629 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b795 -> b1767
node session_CopyInto_-3630 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b795 -> b1767
node dispatch474 micro raw0 b1767 -> b1767
node dispatch475 micro raw1 b1767 -> b1767
node dispatch476 micro raw2 b1767 -> b1767
node session_CopyInto_-3631 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3632 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1531 b1532 -> b1766
node dispatch477 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3633 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b795
node session_CopyInto_-3634 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b795
node session_VecOp_-3635 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3636 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b814 -> b1763
node session_Fp4GemmMetal_-3637 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1535 b1536 -> b1764
node session_CopyInto_-3638 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3639 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b813 -> b1767
node session_CopyInto_-3640 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b813 -> b1767
node session_CopyInto_-3641 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b813 -> b1767
node session_CopyInto_-3642 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b813 -> b1767
node dispatch478 micro raw0 b1767 -> b1767
node dispatch479 micro raw1 b1767 -> b1767
node dispatch480 micro raw2 b1767 -> b1767
node session_CopyInto_-3643 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3644 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1539 b1540 -> b1766
node dispatch481 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3645 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b813
node session_CopyInto_-3646 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b813
node session_VecOp_-3647 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3648 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b832 -> b1763
node session_Fp4GemmMetal_-3649 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1543 b1544 -> b1764
node session_CopyInto_-3650 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3651 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b831 -> b1767
node session_CopyInto_-3652 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b831 -> b1767
node session_CopyInto_-3653 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b831 -> b1767
node session_CopyInto_-3654 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b831 -> b1767
node dispatch482 micro raw0 b1767 -> b1767
node dispatch483 micro raw1 b1767 -> b1767
node dispatch484 micro raw2 b1767 -> b1767
node session_CopyInto_-3655 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3656 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1547 b1548 -> b1766
node dispatch485 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3657 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b831
node session_CopyInto_-3658 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b831
node session_VecOp_-3659 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3660 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b850 -> b1763
node session_Fp4GemmMetal_-3661 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1551 b1552 -> b1764
node session_CopyInto_-3662 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3663 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b849 -> b1767
node session_CopyInto_-3664 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b849 -> b1767
node session_CopyInto_-3665 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b849 -> b1767
node session_CopyInto_-3666 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b849 -> b1767
node dispatch486 micro raw0 b1767 -> b1767
node dispatch487 micro raw1 b1767 -> b1767
node dispatch488 micro raw2 b1767 -> b1767
node session_CopyInto_-3667 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3668 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1555 b1556 -> b1766
node dispatch489 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3669 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b849
node session_CopyInto_-3670 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b849
node session_VecOp_-3671 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3675 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b868 -> b1763
node session_Fp4GemmMetal_-3676 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1559 b1560 -> b1764
node session_CopyInto_-3677 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3678 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b867 -> b1767
node session_CopyInto_-3679 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b867 -> b1767
node session_CopyInto_-3680 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b867 -> b1767
node session_CopyInto_-3681 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b867 -> b1767
node dispatch490 micro raw0 b1767 -> b1767
node dispatch491 micro raw1 b1767 -> b1767
node dispatch492 micro raw2 b1767 -> b1767
node session_CopyInto_-3682 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3683 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1563 b1564 -> b1766
node dispatch493 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3684 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b867
node session_CopyInto_-3685 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b867
node session_VecOp_-3686 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3687 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b886 -> b1763
node session_Fp4GemmMetal_-3688 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1567 b1568 -> b1764
node session_CopyInto_-3689 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3690 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b885 -> b1767
node session_CopyInto_-3691 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b885 -> b1767
node session_CopyInto_-3692 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b885 -> b1767
node session_CopyInto_-3693 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b885 -> b1767
node dispatch494 micro raw0 b1767 -> b1767
node dispatch495 micro raw1 b1767 -> b1767
node dispatch496 micro raw2 b1767 -> b1767
node session_CopyInto_-3694 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3695 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1571 b1572 -> b1766
node dispatch497 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3696 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b885
node session_CopyInto_-3697 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b885
node session_VecOp_-3698 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3699 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b904 -> b1763
node session_Fp4GemmMetal_-3700 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1575 b1576 -> b1764
node session_CopyInto_-3701 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3702 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b903 -> b1767
node session_CopyInto_-3703 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b903 -> b1767
node session_CopyInto_-3704 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b903 -> b1767
node session_CopyInto_-3705 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b903 -> b1767
node dispatch498 micro raw0 b1767 -> b1767
node dispatch499 micro raw1 b1767 -> b1767
node dispatch500 micro raw2 b1767 -> b1767
node session_CopyInto_-3706 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3707 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1579 b1580 -> b1766
node dispatch501 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3708 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b903
node session_CopyInto_-3709 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b903
node session_VecOp_-3710 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3711 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b922 -> b1763
node session_Fp4GemmMetal_-3712 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1583 b1584 -> b1764
node session_CopyInto_-3713 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3714 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b921 -> b1767
node session_CopyInto_-3715 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b921 -> b1767
node session_CopyInto_-3716 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b921 -> b1767
node session_CopyInto_-3717 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b921 -> b1767
node dispatch502 micro raw0 b1767 -> b1767
node dispatch503 micro raw1 b1767 -> b1767
node dispatch504 micro raw2 b1767 -> b1767
node session_CopyInto_-3718 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3719 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1587 b1588 -> b1766
node dispatch505 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3720 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b921
node session_CopyInto_-3721 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b921
node session_VecOp_-3722 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3726 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b940 -> b1763
node session_Fp4GemmMetal_-3727 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1591 b1592 -> b1764
node session_CopyInto_-3728 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3729 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b939 -> b1767
node session_CopyInto_-3730 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b939 -> b1767
node session_CopyInto_-3731 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b939 -> b1767
node session_CopyInto_-3732 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b939 -> b1767
node dispatch506 micro raw0 b1767 -> b1767
node dispatch507 micro raw1 b1767 -> b1767
node dispatch508 micro raw2 b1767 -> b1767
node session_CopyInto_-3733 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3734 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1595 b1596 -> b1766
node dispatch509 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3735 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b939
node session_CopyInto_-3736 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b939
node session_VecOp_-3737 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3738 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b958 -> b1763
node session_Fp4GemmMetal_-3739 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1599 b1600 -> b1764
node session_CopyInto_-3740 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3741 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b957 -> b1767
node session_CopyInto_-3742 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b957 -> b1767
node session_CopyInto_-3743 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b957 -> b1767
node session_CopyInto_-3744 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b957 -> b1767
node dispatch510 micro raw0 b1767 -> b1767
node dispatch511 micro raw1 b1767 -> b1767
node dispatch512 micro raw2 b1767 -> b1767
node session_CopyInto_-3745 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3746 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1603 b1604 -> b1766
node dispatch513 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3747 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b957
node session_CopyInto_-3748 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b957
node session_VecOp_-3749 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3750 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b976 -> b1763
node session_Fp4GemmMetal_-3751 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1615 b1616 -> b1764
node session_CopyInto_-3752 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3753 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b975 -> b1767
node session_CopyInto_-3754 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b975 -> b1767
node session_CopyInto_-3755 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b975 -> b1767
node session_CopyInto_-3756 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b975 -> b1767
node dispatch514 micro raw0 b1767 -> b1767
node dispatch515 micro raw1 b1767 -> b1767
node dispatch516 micro raw2 b1767 -> b1767
node session_CopyInto_-3757 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3758 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1619 b1620 -> b1766
node dispatch517 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3759 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b975
node session_CopyInto_-3760 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b975
node session_VecOp_-3761 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3762 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b994 -> b1763
node session_Fp4GemmMetal_-3763 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1623 b1624 -> b1764
node session_CopyInto_-3764 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3765 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b993 -> b1767
node session_CopyInto_-3766 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b993 -> b1767
node session_CopyInto_-3767 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b993 -> b1767
node session_CopyInto_-3768 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b993 -> b1767
node dispatch518 micro raw0 b1767 -> b1767
node dispatch519 micro raw1 b1767 -> b1767
node dispatch520 micro raw2 b1767 -> b1767
node session_CopyInto_-3769 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3770 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1627 b1628 -> b1766
node dispatch521 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3771 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b993
node session_CopyInto_-3772 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b993
node session_VecOp_-3773 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3777 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1012 -> b1763
node session_Fp4GemmMetal_-3778 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1631 b1632 -> b1764
node session_CopyInto_-3779 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3780 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1011 -> b1767
node session_CopyInto_-3781 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1011 -> b1767
node session_CopyInto_-3782 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1011 -> b1767
node session_CopyInto_-3783 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1011 -> b1767
node dispatch522 micro raw0 b1767 -> b1767
node dispatch523 micro raw1 b1767 -> b1767
node dispatch524 micro raw2 b1767 -> b1767
node session_CopyInto_-3784 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3785 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1635 b1636 -> b1766
node dispatch525 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3786 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1011
node session_CopyInto_-3787 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1011
node session_VecOp_-3788 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3789 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1030 -> b1763
node session_Fp4GemmMetal_-3790 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1639 b1640 -> b1764
node session_CopyInto_-3791 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3792 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1029 -> b1767
node session_CopyInto_-3793 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1029 -> b1767
node session_CopyInto_-3794 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1029 -> b1767
node session_CopyInto_-3795 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1029 -> b1767
node dispatch526 micro raw0 b1767 -> b1767
node dispatch527 micro raw1 b1767 -> b1767
node dispatch528 micro raw2 b1767 -> b1767
node session_CopyInto_-3796 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3797 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1643 b1644 -> b1766
node dispatch529 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3798 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1029
node session_CopyInto_-3799 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1029
node session_VecOp_-3800 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3801 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1048 -> b1763
node session_Fp4GemmMetal_-3802 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1647 b1648 -> b1764
node session_CopyInto_-3803 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3804 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1047 -> b1767
node session_CopyInto_-3805 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1047 -> b1767
node session_CopyInto_-3806 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1047 -> b1767
node session_CopyInto_-3807 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1047 -> b1767
node dispatch530 micro raw0 b1767 -> b1767
node dispatch531 micro raw1 b1767 -> b1767
node dispatch532 micro raw2 b1767 -> b1767
node session_CopyInto_-3808 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3809 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1651 b1652 -> b1766
node dispatch533 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3810 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1047
node session_CopyInto_-3811 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1047
node session_VecOp_-3812 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3813 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1066 -> b1763
node session_Fp4GemmMetal_-3814 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1655 b1656 -> b1764
node session_CopyInto_-3815 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3816 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1065 -> b1767
node session_CopyInto_-3817 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1065 -> b1767
node session_CopyInto_-3818 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1065 -> b1767
node session_CopyInto_-3819 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1065 -> b1767
node dispatch534 micro raw0 b1767 -> b1767
node dispatch535 micro raw1 b1767 -> b1767
node dispatch536 micro raw2 b1767 -> b1767
node session_CopyInto_-3820 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3821 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1659 b1660 -> b1766
node dispatch537 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3822 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1065
node session_CopyInto_-3823 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1065
node session_MatmulNTAccelerated_-3824 session sessionMatmulNTAccelerated:M=128,K=4096,N=12288,bOff=0 b1763 b1070 -> b1777
node dispatch538 micro raw4 b1777 b1071 -> b1777
node session_CopyInto_-3826 session sessionCopyInto:srcOff=0,dstOff=0,n=1572864 b1777 -> b1774
node session_CopyInto_-3827 session sessionCopyInto:srcOff=16384,dstOff=5275648,n=4096 b1074 -> b1774
node session_CopyInto_-3828 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b1074 -> b1782
node dispatch539 micro raw5 b1774_21118976_2129920 b1782 -> b1774_21118976_2129920
node dispatch540 micro raw6 b1774 -> b1774
node dispatch541 micro raw7 b1774 -> b1774
node session_CopyInto_-3829 session sessionCopyInto:srcOff=2621440,dstOff=0,n=524288 b1774 -> b1778
node session_MatmulNTAccelerated_-3830 session sessionMatmulNTAccelerated:M=128,K=4096,N=4096,bOff=0 b1778 b1072 -> b1779
node dispatch542 micro raw8 b1779 b1073 -> b1779
node session_CopyInto_-3832 session sessionCopyInto:srcOff=5279744,dstOff=0,n=131072 b1774 -> b1782
node dispatch543 micro raw5 b1074_81920_524288 b1782 -> b1074_81920_524288
node session_CopyInto_-3833 session sessionCopyInto:srcOff=5541888,dstOff=16384,n=4096 b1774 -> b1074
node dispatch544 micro raw9 b1780 b1766 b1779 -> b1780
node session_MatmulNTAccelerated_-3834 session sessionMatmulNTAccelerated:M=128,K=8192,N=4096,bOff=0 b1780 b1080 -> b1781
node dispatch545 micro raw8 b1781 b1081 -> b1781
node dispatch546 micro raw10 b1766 b1781 b1779 -> b1766
node session_VecOp_-3836 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3840 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1101 -> b1763
node session_Fp4GemmMetal_-3841 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1663 b1664 -> b1764
node session_CopyInto_-3842 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3843 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1100 -> b1767
node session_CopyInto_-3844 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1100 -> b1767
node session_CopyInto_-3845 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1100 -> b1767
node session_CopyInto_-3846 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1100 -> b1767
node dispatch547 micro raw0 b1767 -> b1767
node dispatch548 micro raw1 b1767 -> b1767
node dispatch549 micro raw2 b1767 -> b1767
node session_CopyInto_-3847 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3848 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1667 b1668 -> b1766
node dispatch550 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3849 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1100
node session_CopyInto_-3850 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1100
node session_VecOp_-3851 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3852 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1119 -> b1763
node session_Fp4GemmMetal_-3853 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1671 b1672 -> b1764
node session_CopyInto_-3854 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3855 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1118 -> b1767
node session_CopyInto_-3856 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1118 -> b1767
node session_CopyInto_-3857 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1118 -> b1767
node session_CopyInto_-3858 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1118 -> b1767
node dispatch551 micro raw0 b1767 -> b1767
node dispatch552 micro raw1 b1767 -> b1767
node dispatch553 micro raw2 b1767 -> b1767
node session_CopyInto_-3859 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3860 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1675 b1676 -> b1766
node dispatch554 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3861 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1118
node session_CopyInto_-3862 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1118
node session_VecOp_-3863 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3864 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1137 -> b1763
node session_Fp4GemmMetal_-3865 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1679 b1680 -> b1764
node session_CopyInto_-3866 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3867 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1136 -> b1767
node session_CopyInto_-3868 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1136 -> b1767
node session_CopyInto_-3869 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1136 -> b1767
node session_CopyInto_-3870 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1136 -> b1767
node dispatch555 micro raw0 b1767 -> b1767
node dispatch556 micro raw1 b1767 -> b1767
node dispatch557 micro raw2 b1767 -> b1767
node session_CopyInto_-3871 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3872 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1683 b1684 -> b1766
node dispatch558 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3873 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1136
node session_CopyInto_-3874 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1136
node session_VecOp_-3875 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3876 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1155 -> b1763
node session_Fp4GemmMetal_-3877 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1687 b1688 -> b1764
node session_CopyInto_-3878 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3879 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1154 -> b1767
node session_CopyInto_-3880 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1154 -> b1767
node session_CopyInto_-3881 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1154 -> b1767
node session_CopyInto_-3882 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1154 -> b1767
node dispatch559 micro raw0 b1767 -> b1767
node dispatch560 micro raw1 b1767 -> b1767
node dispatch561 micro raw2 b1767 -> b1767
node session_CopyInto_-3883 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3884 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1691 b1692 -> b1766
node dispatch562 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3885 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1154
node session_CopyInto_-3886 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1154
node session_VecOp_-3887 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3891 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1173 -> b1763
node session_Fp4GemmMetal_-3892 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1703 b1704 -> b1764
node session_CopyInto_-3893 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3894 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1172 -> b1767
node session_CopyInto_-3895 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1172 -> b1767
node session_CopyInto_-3896 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1172 -> b1767
node session_CopyInto_-3897 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1172 -> b1767
node dispatch563 micro raw0 b1767 -> b1767
node dispatch564 micro raw1 b1767 -> b1767
node dispatch565 micro raw2 b1767 -> b1767
node session_CopyInto_-3898 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3899 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1707 b1708 -> b1766
node dispatch566 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3900 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1172
node session_CopyInto_-3901 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1172
node session_VecOp_-3902 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3903 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1191 -> b1763
node session_Fp4GemmMetal_-3904 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1711 b1712 -> b1764
node session_CopyInto_-3905 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3906 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1190 -> b1767
node session_CopyInto_-3907 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1190 -> b1767
node session_CopyInto_-3908 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1190 -> b1767
node session_CopyInto_-3909 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1190 -> b1767
node dispatch567 micro raw0 b1767 -> b1767
node dispatch568 micro raw1 b1767 -> b1767
node dispatch569 micro raw2 b1767 -> b1767
node session_CopyInto_-3910 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3911 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1715 b1716 -> b1766
node dispatch570 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3912 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1190
node session_CopyInto_-3913 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1190
node session_VecOp_-3914 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3915 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1209 -> b1763
node session_Fp4GemmMetal_-3916 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1719 b1720 -> b1764
node session_CopyInto_-3917 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3918 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1208 -> b1767
node session_CopyInto_-3919 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1208 -> b1767
node session_CopyInto_-3920 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1208 -> b1767
node session_CopyInto_-3921 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1208 -> b1767
node dispatch571 micro raw0 b1767 -> b1767
node dispatch572 micro raw1 b1767 -> b1767
node dispatch573 micro raw2 b1767 -> b1767
node session_CopyInto_-3922 session sessionCopyInto:srcOff=10317184,dstOff=0,n=1048576 b1767 -> b1765
node session_Fp4GemmMetal_-3923 session sessionFp4GemmMetal:M=128,K=8192,N=4096,simdgroup=0 b1765 b1723 b1724 -> b1766
node dispatch574 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3924 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1208
node session_CopyInto_-3925 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1208
node session_VecOp_-3926 session sessionVecOp:op=add,n=524288 b1762 b1766 -> b1762
node session_RmsnormEnc_-3927 session sessionRmsnormEnc:rows=128,cols=4096,eps=h3727c5ac b1762 b1227 -> b1763
node session_Fp4GemmMetal_-3928 session sessionFp4GemmMetal:M=128,K=4096,N=18560,simdgroup=0 b1763 b1727 b1728 -> b1764
node session_CopyInto_-3929 session sessionCopyInto:srcOff=0,dstOff=0,n=2375680 b1764 -> b1767
node session_CopyInto_-3930 session sessionCopyInto:srcOff=45696,dstOff=2375680,n=51200 b1226 -> b1767
node session_CopyInto_-3931 session sessionCopyInto:srcOff=37120,dstOff=2426880,n=8576 b1226 -> b1767
node session_CopyInto_-3932 session sessionCopyInto:srcOff=127616,dstOff=5073280,n=1048576 b1226 -> b1767
node session_CopyInto_-3933 session sessionCopyInto:srcOff=96896,dstOff=11365760,n=30720 b1226 -> b1767
node dispatch575 micro raw0 b1767 -> b1767
node dispatch576 micro raw1 b1767 -> b1767
node dispatch577 micro raw2 b1767 -> b1767
node dispatch578 micro raw3 b1768 b1764 -> b1768
node session_CopyInto_-3936 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1768 -> b1226
node session_CopyInto_-3937 session sessionCopyInto:srcOff=6121856,dstOff=127616,n=1048576 b1767 -> b1226
phase chunk2 rows=64
node embedding_2 ext fir.embedding token backbone_embeddings_weight_emb -> b1795
node session_CopyInto_-3982 session sessionCopyInto:srcOff=0,dstOff=0,n=262144 b1795 -> b1786
node session_RmsnormEnc_-4009 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b24 -> b1787
node session_Fp4GemmMetal_-4010 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1247 b1248 -> b1788
node session_CopyInto_-4011 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4012 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b23 -> b1791
node session_CopyInto_-4013 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b23 -> b1791
node session_CopyInto_-4014 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b23 -> b1791
node session_CopyInto_-4015 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b23 -> b1791
node dispatch585 micro raw11 b1791 -> b1791
node dispatch586 micro raw12 b1791 -> b1791
node dispatch587 micro raw13 b1791 -> b1791
node session_CopyInto_-4016 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4017 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1251 b1252 -> b1790
node dispatch588 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4018 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b23
node session_CopyInto_-4019 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b23
node session_VecOp_-4020 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4021 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b42 -> b1787
node session_Fp4GemmMetal_-4022 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1255 b1256 -> b1788
node session_CopyInto_-4023 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4024 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b41 -> b1791
node session_CopyInto_-4025 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b41 -> b1791
node session_CopyInto_-4026 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b41 -> b1791
node session_CopyInto_-4027 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b41 -> b1791
node dispatch589 micro raw11 b1791 -> b1791
node dispatch590 micro raw12 b1791 -> b1791
node dispatch591 micro raw13 b1791 -> b1791
node session_CopyInto_-4028 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4029 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1259 b1260 -> b1790
node dispatch592 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4030 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b41
node session_CopyInto_-4031 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b41
node session_VecOp_-4032 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4033 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b60 -> b1787
node session_Fp4GemmMetal_-4034 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1343 b1344 -> b1788
node session_CopyInto_-4035 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4036 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b59 -> b1791
node session_CopyInto_-4037 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b59 -> b1791
node session_CopyInto_-4038 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b59 -> b1791
node session_CopyInto_-4039 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b59 -> b1791
node dispatch593 micro raw11 b1791 -> b1791
node dispatch594 micro raw12 b1791 -> b1791
node dispatch595 micro raw13 b1791 -> b1791
node session_CopyInto_-4040 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4041 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1347 b1348 -> b1790
node dispatch596 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4042 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b59
node session_CopyInto_-4043 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b59
node session_VecOp_-4044 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4045 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b78 -> b1787
node session_Fp4GemmMetal_-4046 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1431 b1432 -> b1788
node session_CopyInto_-4047 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4048 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b77 -> b1791
node session_CopyInto_-4049 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b77 -> b1791
node session_CopyInto_-4050 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b77 -> b1791
node session_CopyInto_-4051 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b77 -> b1791
node dispatch597 micro raw11 b1791 -> b1791
node dispatch598 micro raw12 b1791 -> b1791
node dispatch599 micro raw13 b1791 -> b1791
node session_CopyInto_-4052 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4053 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1435 b1436 -> b1790
node dispatch600 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4054 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b77
node session_CopyInto_-4055 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b77
node session_VecOp_-4056 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4060 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b96 -> b1787
node session_Fp4GemmMetal_-4061 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1519 b1520 -> b1788
node session_CopyInto_-4062 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4063 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b95 -> b1791
node session_CopyInto_-4064 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b95 -> b1791
node session_CopyInto_-4065 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b95 -> b1791
node session_CopyInto_-4066 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b95 -> b1791
node dispatch601 micro raw11 b1791 -> b1791
node dispatch602 micro raw12 b1791 -> b1791
node dispatch603 micro raw13 b1791 -> b1791
node session_CopyInto_-4067 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4068 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1523 b1524 -> b1790
node dispatch604 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4069 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b95
node session_CopyInto_-4070 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b95
node session_VecOp_-4071 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4072 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b114 -> b1787
node session_Fp4GemmMetal_-4073 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1607 b1608 -> b1788
node session_CopyInto_-4074 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4075 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b113 -> b1791
node session_CopyInto_-4076 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b113 -> b1791
node session_CopyInto_-4077 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b113 -> b1791
node session_CopyInto_-4078 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b113 -> b1791
node dispatch605 micro raw11 b1791 -> b1791
node dispatch606 micro raw12 b1791 -> b1791
node dispatch607 micro raw13 b1791 -> b1791
node session_CopyInto_-4079 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4080 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1611 b1612 -> b1790
node dispatch608 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4081 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b113
node session_CopyInto_-4082 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b113
node session_VecOp_-4083 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4084 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b132 -> b1787
node session_Fp4GemmMetal_-4085 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1695 b1696 -> b1788
node session_CopyInto_-4086 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4087 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b131 -> b1791
node session_CopyInto_-4088 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b131 -> b1791
node session_CopyInto_-4089 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b131 -> b1791
node session_CopyInto_-4090 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b131 -> b1791
node dispatch609 micro raw11 b1791 -> b1791
node dispatch610 micro raw12 b1791 -> b1791
node dispatch611 micro raw13 b1791 -> b1791
node session_CopyInto_-4091 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4092 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1699 b1700 -> b1790
node dispatch612 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4093 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b131
node session_CopyInto_-4094 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b131
node session_VecOp_-4095 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4096 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b150 -> b1787
node session_Fp4GemmMetal_-4097 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1735 b1736 -> b1788
node session_CopyInto_-4098 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4099 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b149 -> b1791
node session_CopyInto_-4100 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b149 -> b1791
node session_CopyInto_-4101 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b149 -> b1791
node session_CopyInto_-4102 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b149 -> b1791
node dispatch613 micro raw11 b1791 -> b1791
node dispatch614 micro raw12 b1791 -> b1791
node dispatch615 micro raw13 b1791 -> b1791
node session_CopyInto_-4103 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4104 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1739 b1740 -> b1790
node dispatch616 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4105 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b149
node session_CopyInto_-4106 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b149
node session_MatmulNTAccelerated_-4107 session sessionMatmulNTAccelerated:M=64,K=4096,N=12288,bOff=0 b1787 b154 -> b1801
node dispatch617 micro raw15 b1801 b155 -> b1801
node session_CopyInto_-4109 session sessionCopyInto:srcOff=0,dstOff=0,n=786432 b1801 -> b1798
node session_CopyInto_-4110 session sessionCopyInto:srcOff=16384,dstOff=2637824,n=4096 b158 -> b1798
node session_CopyInto_-4111 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b158 -> b1806
node dispatch618 micro raw5 b1798_10567680_2129920 b1806 -> b1798_10567680_2129920
node dispatch619 micro raw16 b1798 -> b1798
node dispatch620 micro raw17 b1798 -> b1798
node session_CopyInto_-4112 session sessionCopyInto:srcOff=1310720,dstOff=0,n=262144 b1798 -> b1802
node session_MatmulNTAccelerated_-4113 session sessionMatmulNTAccelerated:M=64,K=4096,N=4096,bOff=0 b1802 b156 -> b1803
node dispatch621 micro raw18 b1803 b157 -> b1803
node session_CopyInto_-4115 session sessionCopyInto:srcOff=2641920,dstOff=0,n=131072 b1798 -> b1806
node dispatch622 micro raw5 b158_81920_524288 b1806 -> b158_81920_524288
node session_CopyInto_-4116 session sessionCopyInto:srcOff=2904064,dstOff=16384,n=4096 b1798 -> b158
node dispatch623 micro raw19 b1804 b1790 b1803 -> b1804
node session_MatmulNTAccelerated_-4117 session sessionMatmulNTAccelerated:M=64,K=8192,N=4096,bOff=0 b1804 b165 -> b1805
node dispatch624 micro raw18 b1805 b166 -> b1805
node dispatch625 micro raw20 b1790 b1805 b1803 -> b1790
node session_VecOp_-4119 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4123 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b186 -> b1787
node session_Fp4GemmMetal_-4124 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1743 b1744 -> b1788
node session_CopyInto_-4125 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4126 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b185 -> b1791
node session_CopyInto_-4127 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b185 -> b1791
node session_CopyInto_-4128 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b185 -> b1791
node session_CopyInto_-4129 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b185 -> b1791
node dispatch626 micro raw11 b1791 -> b1791
node dispatch627 micro raw12 b1791 -> b1791
node dispatch628 micro raw13 b1791 -> b1791
node session_CopyInto_-4130 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4131 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1747 b1748 -> b1790
node dispatch629 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4132 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b185
node session_CopyInto_-4133 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b185
node session_VecOp_-4134 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4135 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b204 -> b1787
node session_Fp4GemmMetal_-4136 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1751 b1752 -> b1788
node session_CopyInto_-4137 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4138 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b203 -> b1791
node session_CopyInto_-4139 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b203 -> b1791
node session_CopyInto_-4140 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b203 -> b1791
node session_CopyInto_-4141 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b203 -> b1791
node dispatch630 micro raw11 b1791 -> b1791
node dispatch631 micro raw12 b1791 -> b1791
node dispatch632 micro raw13 b1791 -> b1791
node session_CopyInto_-4142 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4143 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1755 b1756 -> b1790
node dispatch633 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4144 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b203
node session_CopyInto_-4145 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b203
node session_VecOp_-4146 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4147 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b222 -> b1787
node session_Fp4GemmMetal_-4148 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1263 b1264 -> b1788
node session_CopyInto_-4149 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4150 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b221 -> b1791
node session_CopyInto_-4151 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b221 -> b1791
node session_CopyInto_-4152 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b221 -> b1791
node session_CopyInto_-4153 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b221 -> b1791
node dispatch634 micro raw11 b1791 -> b1791
node dispatch635 micro raw12 b1791 -> b1791
node dispatch636 micro raw13 b1791 -> b1791
node session_CopyInto_-4154 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4155 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1267 b1268 -> b1790
node dispatch637 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4156 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b221
node session_CopyInto_-4157 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b221
node session_VecOp_-4158 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4159 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b240 -> b1787
node session_Fp4GemmMetal_-4160 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1271 b1272 -> b1788
node session_CopyInto_-4161 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4162 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b239 -> b1791
node session_CopyInto_-4163 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b239 -> b1791
node session_CopyInto_-4164 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b239 -> b1791
node session_CopyInto_-4165 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b239 -> b1791
node dispatch638 micro raw11 b1791 -> b1791
node dispatch639 micro raw12 b1791 -> b1791
node dispatch640 micro raw13 b1791 -> b1791
node session_CopyInto_-4166 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4167 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1275 b1276 -> b1790
node dispatch641 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4168 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b239
node session_CopyInto_-4169 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b239
node session_VecOp_-4170 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4174 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b258 -> b1787
node session_Fp4GemmMetal_-4175 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1279 b1280 -> b1788
node session_CopyInto_-4176 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4177 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b257 -> b1791
node session_CopyInto_-4178 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b257 -> b1791
node session_CopyInto_-4179 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b257 -> b1791
node session_CopyInto_-4180 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b257 -> b1791
node dispatch642 micro raw11 b1791 -> b1791
node dispatch643 micro raw12 b1791 -> b1791
node dispatch644 micro raw13 b1791 -> b1791
node session_CopyInto_-4181 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4182 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1283 b1284 -> b1790
node dispatch645 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4183 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b257
node session_CopyInto_-4184 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b257
node session_VecOp_-4185 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4186 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b276 -> b1787
node session_Fp4GemmMetal_-4187 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1287 b1288 -> b1788
node session_CopyInto_-4188 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4189 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b275 -> b1791
node session_CopyInto_-4190 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b275 -> b1791
node session_CopyInto_-4191 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b275 -> b1791
node session_CopyInto_-4192 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b275 -> b1791
node dispatch646 micro raw11 b1791 -> b1791
node dispatch647 micro raw12 b1791 -> b1791
node dispatch648 micro raw13 b1791 -> b1791
node session_CopyInto_-4193 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4194 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1291 b1292 -> b1790
node dispatch649 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4195 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b275
node session_CopyInto_-4196 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b275
node session_VecOp_-4197 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4198 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b294 -> b1787
node session_Fp4GemmMetal_-4199 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1295 b1296 -> b1788
node session_CopyInto_-4200 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4201 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b293 -> b1791
node session_CopyInto_-4202 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b293 -> b1791
node session_CopyInto_-4203 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b293 -> b1791
node session_CopyInto_-4204 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b293 -> b1791
node dispatch650 micro raw11 b1791 -> b1791
node dispatch651 micro raw12 b1791 -> b1791
node dispatch652 micro raw13 b1791 -> b1791
node session_CopyInto_-4205 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4206 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1299 b1300 -> b1790
node dispatch653 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4207 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b293
node session_CopyInto_-4208 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b293
node session_VecOp_-4209 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4210 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b312 -> b1787
node session_Fp4GemmMetal_-4211 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1303 b1304 -> b1788
node session_CopyInto_-4212 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4213 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b311 -> b1791
node session_CopyInto_-4214 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b311 -> b1791
node session_CopyInto_-4215 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b311 -> b1791
node session_CopyInto_-4216 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b311 -> b1791
node dispatch654 micro raw11 b1791 -> b1791
node dispatch655 micro raw12 b1791 -> b1791
node dispatch656 micro raw13 b1791 -> b1791
node session_CopyInto_-4217 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4218 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1307 b1308 -> b1790
node dispatch657 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4219 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b311
node session_CopyInto_-4220 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b311
node session_VecOp_-4221 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4225 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b330 -> b1787
node session_Fp4GemmMetal_-4226 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1311 b1312 -> b1788
node session_CopyInto_-4227 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4228 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b329 -> b1791
node session_CopyInto_-4229 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b329 -> b1791
node session_CopyInto_-4230 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b329 -> b1791
node session_CopyInto_-4231 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b329 -> b1791
node dispatch658 micro raw11 b1791 -> b1791
node dispatch659 micro raw12 b1791 -> b1791
node dispatch660 micro raw13 b1791 -> b1791
node session_CopyInto_-4232 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4233 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1315 b1316 -> b1790
node dispatch661 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4234 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b329
node session_CopyInto_-4235 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b329
node session_VecOp_-4236 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4237 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b348 -> b1787
node session_Fp4GemmMetal_-4238 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1319 b1320 -> b1788
node session_CopyInto_-4239 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4240 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b347 -> b1791
node session_CopyInto_-4241 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b347 -> b1791
node session_CopyInto_-4242 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b347 -> b1791
node session_CopyInto_-4243 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b347 -> b1791
node dispatch662 micro raw11 b1791 -> b1791
node dispatch663 micro raw12 b1791 -> b1791
node dispatch664 micro raw13 b1791 -> b1791
node session_CopyInto_-4244 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4245 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1323 b1324 -> b1790
node dispatch665 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4246 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b347
node session_CopyInto_-4247 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b347
node session_VecOp_-4248 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4249 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b366 -> b1787
node session_Fp4GemmMetal_-4250 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1327 b1328 -> b1788
node session_CopyInto_-4251 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4252 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b365 -> b1791
node session_CopyInto_-4253 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b365 -> b1791
node session_CopyInto_-4254 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b365 -> b1791
node session_CopyInto_-4255 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b365 -> b1791
node dispatch666 micro raw11 b1791 -> b1791
node dispatch667 micro raw12 b1791 -> b1791
node dispatch668 micro raw13 b1791 -> b1791
node session_CopyInto_-4256 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4257 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1331 b1332 -> b1790
node dispatch669 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4258 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b365
node session_CopyInto_-4259 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b365
node session_VecOp_-4260 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4261 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b384 -> b1787
node session_Fp4GemmMetal_-4262 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1335 b1336 -> b1788
node session_CopyInto_-4263 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4264 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b383 -> b1791
node session_CopyInto_-4265 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b383 -> b1791
node session_CopyInto_-4266 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b383 -> b1791
node session_CopyInto_-4267 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b383 -> b1791
node dispatch670 micro raw11 b1791 -> b1791
node dispatch671 micro raw12 b1791 -> b1791
node dispatch672 micro raw13 b1791 -> b1791
node session_CopyInto_-4268 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4269 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1339 b1340 -> b1790
node dispatch673 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4270 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b383
node session_CopyInto_-4271 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b383
node session_VecOp_-4272 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4276 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b402 -> b1787
node session_Fp4GemmMetal_-4277 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1351 b1352 -> b1788
node session_CopyInto_-4278 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4279 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b401 -> b1791
node session_CopyInto_-4280 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b401 -> b1791
node session_CopyInto_-4281 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b401 -> b1791
node session_CopyInto_-4282 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b401 -> b1791
node dispatch674 micro raw11 b1791 -> b1791
node dispatch675 micro raw12 b1791 -> b1791
node dispatch676 micro raw13 b1791 -> b1791
node session_CopyInto_-4283 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4284 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1355 b1356 -> b1790
node dispatch677 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4285 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b401
node session_CopyInto_-4286 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b401
node session_VecOp_-4287 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4288 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b420 -> b1787
node session_Fp4GemmMetal_-4289 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1359 b1360 -> b1788
node session_CopyInto_-4290 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4291 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b419 -> b1791
node session_CopyInto_-4292 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b419 -> b1791
node session_CopyInto_-4293 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b419 -> b1791
node session_CopyInto_-4294 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b419 -> b1791
node dispatch678 micro raw11 b1791 -> b1791
node dispatch679 micro raw12 b1791 -> b1791
node dispatch680 micro raw13 b1791 -> b1791
node session_CopyInto_-4295 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4296 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1363 b1364 -> b1790
node dispatch681 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4297 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b419
node session_CopyInto_-4298 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b419
node session_VecOp_-4299 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4300 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b438 -> b1787
node session_Fp4GemmMetal_-4301 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1367 b1368 -> b1788
node session_CopyInto_-4302 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4303 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b437 -> b1791
node session_CopyInto_-4304 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b437 -> b1791
node session_CopyInto_-4305 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b437 -> b1791
node session_CopyInto_-4306 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b437 -> b1791
node dispatch682 micro raw11 b1791 -> b1791
node dispatch683 micro raw12 b1791 -> b1791
node dispatch684 micro raw13 b1791 -> b1791
node session_CopyInto_-4307 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4308 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1371 b1372 -> b1790
node dispatch685 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4309 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b437
node session_CopyInto_-4310 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b437
node session_VecOp_-4311 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4312 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b456 -> b1787
node session_Fp4GemmMetal_-4313 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1375 b1376 -> b1788
node session_CopyInto_-4314 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4315 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b455 -> b1791
node session_CopyInto_-4316 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b455 -> b1791
node session_CopyInto_-4317 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b455 -> b1791
node session_CopyInto_-4318 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b455 -> b1791
node dispatch686 micro raw11 b1791 -> b1791
node dispatch687 micro raw12 b1791 -> b1791
node dispatch688 micro raw13 b1791 -> b1791
node session_CopyInto_-4319 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4320 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1379 b1380 -> b1790
node dispatch689 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4321 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b455
node session_CopyInto_-4322 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b455
node session_MatmulNTAccelerated_-4323 session sessionMatmulNTAccelerated:M=64,K=4096,N=12288,bOff=0 b1787 b460 -> b1801
node dispatch690 micro raw15 b1801 b461 -> b1801
node session_CopyInto_-4325 session sessionCopyInto:srcOff=0,dstOff=0,n=786432 b1801 -> b1798
node session_CopyInto_-4326 session sessionCopyInto:srcOff=16384,dstOff=2637824,n=4096 b464 -> b1798
node session_CopyInto_-4327 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b464 -> b1806
node dispatch691 micro raw5 b1798_10567680_2129920 b1806 -> b1798_10567680_2129920
node dispatch692 micro raw16 b1798 -> b1798
node dispatch693 micro raw17 b1798 -> b1798
node session_CopyInto_-4328 session sessionCopyInto:srcOff=1310720,dstOff=0,n=262144 b1798 -> b1802
node session_MatmulNTAccelerated_-4329 session sessionMatmulNTAccelerated:M=64,K=4096,N=4096,bOff=0 b1802 b462 -> b1803
node dispatch694 micro raw18 b1803 b463 -> b1803
node session_CopyInto_-4331 session sessionCopyInto:srcOff=2641920,dstOff=0,n=131072 b1798 -> b1806
node dispatch695 micro raw5 b464_81920_524288 b1806 -> b464_81920_524288
node session_CopyInto_-4332 session sessionCopyInto:srcOff=2904064,dstOff=16384,n=4096 b1798 -> b464
node dispatch696 micro raw19 b1804 b1790 b1803 -> b1804
node session_MatmulNTAccelerated_-4333 session sessionMatmulNTAccelerated:M=64,K=8192,N=4096,bOff=0 b1804 b470 -> b1805
node dispatch697 micro raw18 b1805 b471 -> b1805
node dispatch698 micro raw20 b1790 b1805 b1803 -> b1790
node session_VecOp_-4335 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4339 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b491 -> b1787
node session_Fp4GemmMetal_-4340 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1383 b1384 -> b1788
node session_CopyInto_-4341 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4342 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b490 -> b1791
node session_CopyInto_-4343 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b490 -> b1791
node session_CopyInto_-4344 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b490 -> b1791
node session_CopyInto_-4345 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b490 -> b1791
node dispatch699 micro raw11 b1791 -> b1791
node dispatch700 micro raw12 b1791 -> b1791
node dispatch701 micro raw13 b1791 -> b1791
node session_CopyInto_-4346 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4347 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1387 b1388 -> b1790
node dispatch702 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4348 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b490
node session_CopyInto_-4349 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b490
node session_VecOp_-4350 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4351 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b509 -> b1787
node session_Fp4GemmMetal_-4352 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1391 b1392 -> b1788
node session_CopyInto_-4353 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4354 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b508 -> b1791
node session_CopyInto_-4355 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b508 -> b1791
node session_CopyInto_-4356 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b508 -> b1791
node session_CopyInto_-4357 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b508 -> b1791
node dispatch703 micro raw11 b1791 -> b1791
node dispatch704 micro raw12 b1791 -> b1791
node dispatch705 micro raw13 b1791 -> b1791
node session_CopyInto_-4358 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4359 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1395 b1396 -> b1790
node dispatch706 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4360 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b508
node session_CopyInto_-4361 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b508
node session_VecOp_-4362 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4363 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b527 -> b1787
node session_Fp4GemmMetal_-4364 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1399 b1400 -> b1788
node session_CopyInto_-4365 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4366 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b526 -> b1791
node session_CopyInto_-4367 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b526 -> b1791
node session_CopyInto_-4368 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b526 -> b1791
node session_CopyInto_-4369 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b526 -> b1791
node dispatch707 micro raw11 b1791 -> b1791
node dispatch708 micro raw12 b1791 -> b1791
node dispatch709 micro raw13 b1791 -> b1791
node session_CopyInto_-4370 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4371 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1403 b1404 -> b1790
node dispatch710 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4372 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b526
node session_CopyInto_-4373 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b526
node session_VecOp_-4374 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4375 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b545 -> b1787
node session_Fp4GemmMetal_-4376 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1407 b1408 -> b1788
node session_CopyInto_-4377 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4378 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b544 -> b1791
node session_CopyInto_-4379 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b544 -> b1791
node session_CopyInto_-4380 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b544 -> b1791
node session_CopyInto_-4381 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b544 -> b1791
node dispatch711 micro raw11 b1791 -> b1791
node dispatch712 micro raw12 b1791 -> b1791
node dispatch713 micro raw13 b1791 -> b1791
node session_CopyInto_-4382 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4383 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1411 b1412 -> b1790
node dispatch714 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4384 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b544
node session_CopyInto_-4385 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b544
node session_VecOp_-4386 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4390 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b563 -> b1787
node session_Fp4GemmMetal_-4391 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1415 b1416 -> b1788
node session_CopyInto_-4392 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4393 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b562 -> b1791
node session_CopyInto_-4394 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b562 -> b1791
node session_CopyInto_-4395 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b562 -> b1791
node session_CopyInto_-4396 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b562 -> b1791
node dispatch715 micro raw11 b1791 -> b1791
node dispatch716 micro raw12 b1791 -> b1791
node dispatch717 micro raw13 b1791 -> b1791
node session_CopyInto_-4397 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4398 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1419 b1420 -> b1790
node dispatch718 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4399 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b562
node session_CopyInto_-4400 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b562
node session_VecOp_-4401 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4402 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b581 -> b1787
node session_Fp4GemmMetal_-4403 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1423 b1424 -> b1788
node session_CopyInto_-4404 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4405 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b580 -> b1791
node session_CopyInto_-4406 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b580 -> b1791
node session_CopyInto_-4407 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b580 -> b1791
node session_CopyInto_-4408 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b580 -> b1791
node dispatch719 micro raw11 b1791 -> b1791
node dispatch720 micro raw12 b1791 -> b1791
node dispatch721 micro raw13 b1791 -> b1791
node session_CopyInto_-4409 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4410 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1427 b1428 -> b1790
node dispatch722 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4411 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b580
node session_CopyInto_-4412 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b580
node session_VecOp_-4413 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4414 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b599 -> b1787
node session_Fp4GemmMetal_-4415 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1439 b1440 -> b1788
node session_CopyInto_-4416 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4417 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b598 -> b1791
node session_CopyInto_-4418 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b598 -> b1791
node session_CopyInto_-4419 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b598 -> b1791
node session_CopyInto_-4420 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b598 -> b1791
node dispatch723 micro raw11 b1791 -> b1791
node dispatch724 micro raw12 b1791 -> b1791
node dispatch725 micro raw13 b1791 -> b1791
node session_CopyInto_-4421 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4422 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1443 b1444 -> b1790
node dispatch726 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4423 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b598
node session_CopyInto_-4424 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b598
node session_VecOp_-4425 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4426 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b617 -> b1787
node session_Fp4GemmMetal_-4427 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1447 b1448 -> b1788
node session_CopyInto_-4428 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4429 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b616 -> b1791
node session_CopyInto_-4430 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b616 -> b1791
node session_CopyInto_-4431 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b616 -> b1791
node session_CopyInto_-4432 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b616 -> b1791
node dispatch727 micro raw11 b1791 -> b1791
node dispatch728 micro raw12 b1791 -> b1791
node dispatch729 micro raw13 b1791 -> b1791
node session_CopyInto_-4433 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4434 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1451 b1452 -> b1790
node dispatch730 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4435 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b616
node session_CopyInto_-4436 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b616
node session_VecOp_-4437 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4441 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b635 -> b1787
node session_Fp4GemmMetal_-4442 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1455 b1456 -> b1788
node session_CopyInto_-4443 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4444 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b634 -> b1791
node session_CopyInto_-4445 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b634 -> b1791
node session_CopyInto_-4446 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b634 -> b1791
node session_CopyInto_-4447 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b634 -> b1791
node dispatch731 micro raw11 b1791 -> b1791
node dispatch732 micro raw12 b1791 -> b1791
node dispatch733 micro raw13 b1791 -> b1791
node session_CopyInto_-4448 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4449 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1459 b1460 -> b1790
node dispatch734 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4450 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b634
node session_CopyInto_-4451 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b634
node session_VecOp_-4452 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4453 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b653 -> b1787
node session_Fp4GemmMetal_-4454 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1463 b1464 -> b1788
node session_CopyInto_-4455 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4456 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b652 -> b1791
node session_CopyInto_-4457 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b652 -> b1791
node session_CopyInto_-4458 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b652 -> b1791
node session_CopyInto_-4459 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b652 -> b1791
node dispatch735 micro raw11 b1791 -> b1791
node dispatch736 micro raw12 b1791 -> b1791
node dispatch737 micro raw13 b1791 -> b1791
node session_CopyInto_-4460 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4461 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1467 b1468 -> b1790
node dispatch738 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4462 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b652
node session_CopyInto_-4463 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b652
node session_VecOp_-4464 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4465 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b671 -> b1787
node session_Fp4GemmMetal_-4466 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1471 b1472 -> b1788
node session_CopyInto_-4467 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4468 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b670 -> b1791
node session_CopyInto_-4469 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b670 -> b1791
node session_CopyInto_-4470 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b670 -> b1791
node session_CopyInto_-4471 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b670 -> b1791
node dispatch739 micro raw11 b1791 -> b1791
node dispatch740 micro raw12 b1791 -> b1791
node dispatch741 micro raw13 b1791 -> b1791
node session_CopyInto_-4472 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4473 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1475 b1476 -> b1790
node dispatch742 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4474 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b670
node session_CopyInto_-4475 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b670
node session_VecOp_-4476 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4477 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b689 -> b1787
node session_Fp4GemmMetal_-4478 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1479 b1480 -> b1788
node session_CopyInto_-4479 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4480 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b688 -> b1791
node session_CopyInto_-4481 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b688 -> b1791
node session_CopyInto_-4482 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b688 -> b1791
node session_CopyInto_-4483 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b688 -> b1791
node dispatch743 micro raw11 b1791 -> b1791
node dispatch744 micro raw12 b1791 -> b1791
node dispatch745 micro raw13 b1791 -> b1791
node session_CopyInto_-4484 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4485 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1483 b1484 -> b1790
node dispatch746 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4486 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b688
node session_CopyInto_-4487 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b688
node session_VecOp_-4488 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4492 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b707 -> b1787
node session_Fp4GemmMetal_-4493 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1487 b1488 -> b1788
node session_CopyInto_-4494 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4495 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b706 -> b1791
node session_CopyInto_-4496 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b706 -> b1791
node session_CopyInto_-4497 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b706 -> b1791
node session_CopyInto_-4498 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b706 -> b1791
node dispatch747 micro raw11 b1791 -> b1791
node dispatch748 micro raw12 b1791 -> b1791
node dispatch749 micro raw13 b1791 -> b1791
node session_CopyInto_-4499 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4500 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1491 b1492 -> b1790
node dispatch750 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4501 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b706
node session_CopyInto_-4502 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b706
node session_VecOp_-4503 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4504 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b725 -> b1787
node session_Fp4GemmMetal_-4505 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1495 b1496 -> b1788
node session_CopyInto_-4506 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4507 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b724 -> b1791
node session_CopyInto_-4508 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b724 -> b1791
node session_CopyInto_-4509 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b724 -> b1791
node session_CopyInto_-4510 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b724 -> b1791
node dispatch751 micro raw11 b1791 -> b1791
node dispatch752 micro raw12 b1791 -> b1791
node dispatch753 micro raw13 b1791 -> b1791
node session_CopyInto_-4511 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4512 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1499 b1500 -> b1790
node dispatch754 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4513 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b724
node session_CopyInto_-4514 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b724
node session_VecOp_-4515 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4516 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b743 -> b1787
node session_Fp4GemmMetal_-4517 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1503 b1504 -> b1788
node session_CopyInto_-4518 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4519 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b742 -> b1791
node session_CopyInto_-4520 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b742 -> b1791
node session_CopyInto_-4521 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b742 -> b1791
node session_CopyInto_-4522 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b742 -> b1791
node dispatch755 micro raw11 b1791 -> b1791
node dispatch756 micro raw12 b1791 -> b1791
node dispatch757 micro raw13 b1791 -> b1791
node session_CopyInto_-4523 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4524 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1507 b1508 -> b1790
node dispatch758 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4525 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b742
node session_CopyInto_-4526 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b742
node session_VecOp_-4527 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4528 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b761 -> b1787
node session_Fp4GemmMetal_-4529 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1511 b1512 -> b1788
node session_CopyInto_-4530 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4531 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b760 -> b1791
node session_CopyInto_-4532 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b760 -> b1791
node session_CopyInto_-4533 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b760 -> b1791
node session_CopyInto_-4534 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b760 -> b1791
node dispatch759 micro raw11 b1791 -> b1791
node dispatch760 micro raw12 b1791 -> b1791
node dispatch761 micro raw13 b1791 -> b1791
node session_CopyInto_-4535 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4536 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1515 b1516 -> b1790
node dispatch762 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4537 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b760
node session_CopyInto_-4538 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b760
node session_MatmulNTAccelerated_-4539 session sessionMatmulNTAccelerated:M=64,K=4096,N=12288,bOff=0 b1787 b765 -> b1801
node dispatch763 micro raw15 b1801 b766 -> b1801
node session_CopyInto_-4541 session sessionCopyInto:srcOff=0,dstOff=0,n=786432 b1801 -> b1798
node session_CopyInto_-4542 session sessionCopyInto:srcOff=16384,dstOff=2637824,n=4096 b769 -> b1798
node session_CopyInto_-4543 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b769 -> b1806
node dispatch764 micro raw5 b1798_10567680_2129920 b1806 -> b1798_10567680_2129920
node dispatch765 micro raw16 b1798 -> b1798
node dispatch766 micro raw17 b1798 -> b1798
node session_CopyInto_-4544 session sessionCopyInto:srcOff=1310720,dstOff=0,n=262144 b1798 -> b1802
node session_MatmulNTAccelerated_-4545 session sessionMatmulNTAccelerated:M=64,K=4096,N=4096,bOff=0 b1802 b767 -> b1803
node dispatch767 micro raw18 b1803 b768 -> b1803
node session_CopyInto_-4547 session sessionCopyInto:srcOff=2641920,dstOff=0,n=131072 b1798 -> b1806
node dispatch768 micro raw5 b769_81920_524288 b1806 -> b769_81920_524288
node session_CopyInto_-4548 session sessionCopyInto:srcOff=2904064,dstOff=16384,n=4096 b1798 -> b769
node dispatch769 micro raw19 b1804 b1790 b1803 -> b1804
node session_MatmulNTAccelerated_-4549 session sessionMatmulNTAccelerated:M=64,K=8192,N=4096,bOff=0 b1804 b775 -> b1805
node dispatch770 micro raw18 b1805 b776 -> b1805
node dispatch771 micro raw20 b1790 b1805 b1803 -> b1790
node session_VecOp_-4551 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4555 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b796 -> b1787
node session_Fp4GemmMetal_-4556 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1527 b1528 -> b1788
node session_CopyInto_-4557 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4558 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b795 -> b1791
node session_CopyInto_-4559 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b795 -> b1791
node session_CopyInto_-4560 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b795 -> b1791
node session_CopyInto_-4561 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b795 -> b1791
node dispatch772 micro raw11 b1791 -> b1791
node dispatch773 micro raw12 b1791 -> b1791
node dispatch774 micro raw13 b1791 -> b1791
node session_CopyInto_-4562 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4563 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1531 b1532 -> b1790
node dispatch775 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4564 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b795
node session_CopyInto_-4565 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b795
node session_VecOp_-4566 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4567 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b814 -> b1787
node session_Fp4GemmMetal_-4568 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1535 b1536 -> b1788
node session_CopyInto_-4569 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4570 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b813 -> b1791
node session_CopyInto_-4571 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b813 -> b1791
node session_CopyInto_-4572 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b813 -> b1791
node session_CopyInto_-4573 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b813 -> b1791
node dispatch776 micro raw11 b1791 -> b1791
node dispatch777 micro raw12 b1791 -> b1791
node dispatch778 micro raw13 b1791 -> b1791
node session_CopyInto_-4574 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4575 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1539 b1540 -> b1790
node dispatch779 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4576 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b813
node session_CopyInto_-4577 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b813
node session_VecOp_-4578 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4579 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b832 -> b1787
node session_Fp4GemmMetal_-4580 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1543 b1544 -> b1788
node session_CopyInto_-4581 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4582 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b831 -> b1791
node session_CopyInto_-4583 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b831 -> b1791
node session_CopyInto_-4584 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b831 -> b1791
node session_CopyInto_-4585 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b831 -> b1791
node dispatch780 micro raw11 b1791 -> b1791
node dispatch781 micro raw12 b1791 -> b1791
node dispatch782 micro raw13 b1791 -> b1791
node session_CopyInto_-4586 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4587 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1547 b1548 -> b1790
node dispatch783 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4588 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b831
node session_CopyInto_-4589 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b831
node session_VecOp_-4590 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4591 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b850 -> b1787
node session_Fp4GemmMetal_-4592 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1551 b1552 -> b1788
node session_CopyInto_-4593 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4594 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b849 -> b1791
node session_CopyInto_-4595 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b849 -> b1791
node session_CopyInto_-4596 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b849 -> b1791
node session_CopyInto_-4597 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b849 -> b1791
node dispatch784 micro raw11 b1791 -> b1791
node dispatch785 micro raw12 b1791 -> b1791
node dispatch786 micro raw13 b1791 -> b1791
node session_CopyInto_-4598 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4599 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1555 b1556 -> b1790
node dispatch787 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4600 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b849
node session_CopyInto_-4601 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b849
node session_VecOp_-4602 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4606 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b868 -> b1787
node session_Fp4GemmMetal_-4607 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1559 b1560 -> b1788
node session_CopyInto_-4608 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4609 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b867 -> b1791
node session_CopyInto_-4610 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b867 -> b1791
node session_CopyInto_-4611 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b867 -> b1791
node session_CopyInto_-4612 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b867 -> b1791
node dispatch788 micro raw11 b1791 -> b1791
node dispatch789 micro raw12 b1791 -> b1791
node dispatch790 micro raw13 b1791 -> b1791
node session_CopyInto_-4613 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4614 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1563 b1564 -> b1790
node dispatch791 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4615 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b867
node session_CopyInto_-4616 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b867
node session_VecOp_-4617 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4618 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b886 -> b1787
node session_Fp4GemmMetal_-4619 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1567 b1568 -> b1788
node session_CopyInto_-4620 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4621 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b885 -> b1791
node session_CopyInto_-4622 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b885 -> b1791
node session_CopyInto_-4623 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b885 -> b1791
node session_CopyInto_-4624 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b885 -> b1791
node dispatch792 micro raw11 b1791 -> b1791
node dispatch793 micro raw12 b1791 -> b1791
node dispatch794 micro raw13 b1791 -> b1791
node session_CopyInto_-4625 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4626 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1571 b1572 -> b1790
node dispatch795 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4627 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b885
node session_CopyInto_-4628 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b885
node session_VecOp_-4629 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4630 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b904 -> b1787
node session_Fp4GemmMetal_-4631 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1575 b1576 -> b1788
node session_CopyInto_-4632 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4633 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b903 -> b1791
node session_CopyInto_-4634 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b903 -> b1791
node session_CopyInto_-4635 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b903 -> b1791
node session_CopyInto_-4636 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b903 -> b1791
node dispatch796 micro raw11 b1791 -> b1791
node dispatch797 micro raw12 b1791 -> b1791
node dispatch798 micro raw13 b1791 -> b1791
node session_CopyInto_-4637 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4638 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1579 b1580 -> b1790
node dispatch799 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4639 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b903
node session_CopyInto_-4640 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b903
node session_VecOp_-4641 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4642 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b922 -> b1787
node session_Fp4GemmMetal_-4643 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1583 b1584 -> b1788
node session_CopyInto_-4644 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4645 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b921 -> b1791
node session_CopyInto_-4646 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b921 -> b1791
node session_CopyInto_-4647 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b921 -> b1791
node session_CopyInto_-4648 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b921 -> b1791
node dispatch800 micro raw11 b1791 -> b1791
node dispatch801 micro raw12 b1791 -> b1791
node dispatch802 micro raw13 b1791 -> b1791
node session_CopyInto_-4649 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4650 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1587 b1588 -> b1790
node dispatch803 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4651 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b921
node session_CopyInto_-4652 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b921
node session_VecOp_-4653 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4657 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b940 -> b1787
node session_Fp4GemmMetal_-4658 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1591 b1592 -> b1788
node session_CopyInto_-4659 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4660 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b939 -> b1791
node session_CopyInto_-4661 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b939 -> b1791
node session_CopyInto_-4662 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b939 -> b1791
node session_CopyInto_-4663 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b939 -> b1791
node dispatch804 micro raw11 b1791 -> b1791
node dispatch805 micro raw12 b1791 -> b1791
node dispatch806 micro raw13 b1791 -> b1791
node session_CopyInto_-4664 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4665 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1595 b1596 -> b1790
node dispatch807 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4666 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b939
node session_CopyInto_-4667 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b939
node session_VecOp_-4668 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4669 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b958 -> b1787
node session_Fp4GemmMetal_-4670 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1599 b1600 -> b1788
node session_CopyInto_-4671 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4672 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b957 -> b1791
node session_CopyInto_-4673 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b957 -> b1791
node session_CopyInto_-4674 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b957 -> b1791
node session_CopyInto_-4675 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b957 -> b1791
node dispatch808 micro raw11 b1791 -> b1791
node dispatch809 micro raw12 b1791 -> b1791
node dispatch810 micro raw13 b1791 -> b1791
node session_CopyInto_-4676 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4677 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1603 b1604 -> b1790
node dispatch811 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4678 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b957
node session_CopyInto_-4679 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b957
node session_VecOp_-4680 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4681 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b976 -> b1787
node session_Fp4GemmMetal_-4682 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1615 b1616 -> b1788
node session_CopyInto_-4683 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4684 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b975 -> b1791
node session_CopyInto_-4685 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b975 -> b1791
node session_CopyInto_-4686 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b975 -> b1791
node session_CopyInto_-4687 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b975 -> b1791
node dispatch812 micro raw11 b1791 -> b1791
node dispatch813 micro raw12 b1791 -> b1791
node dispatch814 micro raw13 b1791 -> b1791
node session_CopyInto_-4688 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4689 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1619 b1620 -> b1790
node dispatch815 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4690 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b975
node session_CopyInto_-4691 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b975
node session_VecOp_-4692 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4693 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b994 -> b1787
node session_Fp4GemmMetal_-4694 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1623 b1624 -> b1788
node session_CopyInto_-4695 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4696 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b993 -> b1791
node session_CopyInto_-4697 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b993 -> b1791
node session_CopyInto_-4698 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b993 -> b1791
node session_CopyInto_-4699 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b993 -> b1791
node dispatch816 micro raw11 b1791 -> b1791
node dispatch817 micro raw12 b1791 -> b1791
node dispatch818 micro raw13 b1791 -> b1791
node session_CopyInto_-4700 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4701 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1627 b1628 -> b1790
node dispatch819 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4702 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b993
node session_CopyInto_-4703 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b993
node session_VecOp_-4704 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4708 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1012 -> b1787
node session_Fp4GemmMetal_-4709 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1631 b1632 -> b1788
node session_CopyInto_-4710 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4711 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1011 -> b1791
node session_CopyInto_-4712 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1011 -> b1791
node session_CopyInto_-4713 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1011 -> b1791
node session_CopyInto_-4714 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1011 -> b1791
node dispatch820 micro raw11 b1791 -> b1791
node dispatch821 micro raw12 b1791 -> b1791
node dispatch822 micro raw13 b1791 -> b1791
node session_CopyInto_-4715 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4716 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1635 b1636 -> b1790
node dispatch823 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4717 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1011
node session_CopyInto_-4718 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1011
node session_VecOp_-4719 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4720 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1030 -> b1787
node session_Fp4GemmMetal_-4721 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1639 b1640 -> b1788
node session_CopyInto_-4722 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4723 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1029 -> b1791
node session_CopyInto_-4724 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1029 -> b1791
node session_CopyInto_-4725 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1029 -> b1791
node session_CopyInto_-4726 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1029 -> b1791
node dispatch824 micro raw11 b1791 -> b1791
node dispatch825 micro raw12 b1791 -> b1791
node dispatch826 micro raw13 b1791 -> b1791
node session_CopyInto_-4727 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4728 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1643 b1644 -> b1790
node dispatch827 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4729 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1029
node session_CopyInto_-4730 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1029
node session_VecOp_-4731 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4732 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1048 -> b1787
node session_Fp4GemmMetal_-4733 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1647 b1648 -> b1788
node session_CopyInto_-4734 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4735 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1047 -> b1791
node session_CopyInto_-4736 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1047 -> b1791
node session_CopyInto_-4737 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1047 -> b1791
node session_CopyInto_-4738 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1047 -> b1791
node dispatch828 micro raw11 b1791 -> b1791
node dispatch829 micro raw12 b1791 -> b1791
node dispatch830 micro raw13 b1791 -> b1791
node session_CopyInto_-4739 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4740 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1651 b1652 -> b1790
node dispatch831 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4741 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1047
node session_CopyInto_-4742 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1047
node session_VecOp_-4743 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4744 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1066 -> b1787
node session_Fp4GemmMetal_-4745 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1655 b1656 -> b1788
node session_CopyInto_-4746 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4747 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1065 -> b1791
node session_CopyInto_-4748 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1065 -> b1791
node session_CopyInto_-4749 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1065 -> b1791
node session_CopyInto_-4750 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1065 -> b1791
node dispatch832 micro raw11 b1791 -> b1791
node dispatch833 micro raw12 b1791 -> b1791
node dispatch834 micro raw13 b1791 -> b1791
node session_CopyInto_-4751 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4752 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1659 b1660 -> b1790
node dispatch835 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4753 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1065
node session_CopyInto_-4754 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1065
node session_MatmulNTAccelerated_-4755 session sessionMatmulNTAccelerated:M=64,K=4096,N=12288,bOff=0 b1787 b1070 -> b1801
node dispatch836 micro raw15 b1801 b1071 -> b1801
node session_CopyInto_-4757 session sessionCopyInto:srcOff=0,dstOff=0,n=786432 b1801 -> b1798
node session_CopyInto_-4758 session sessionCopyInto:srcOff=16384,dstOff=2637824,n=4096 b1074 -> b1798
node session_CopyInto_-4759 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b1074 -> b1806
node dispatch837 micro raw5 b1798_10567680_2129920 b1806 -> b1798_10567680_2129920
node dispatch838 micro raw16 b1798 -> b1798
node dispatch839 micro raw17 b1798 -> b1798
node session_CopyInto_-4760 session sessionCopyInto:srcOff=1310720,dstOff=0,n=262144 b1798 -> b1802
node session_MatmulNTAccelerated_-4761 session sessionMatmulNTAccelerated:M=64,K=4096,N=4096,bOff=0 b1802 b1072 -> b1803
node dispatch840 micro raw18 b1803 b1073 -> b1803
node session_CopyInto_-4763 session sessionCopyInto:srcOff=2641920,dstOff=0,n=131072 b1798 -> b1806
node dispatch841 micro raw5 b1074_81920_524288 b1806 -> b1074_81920_524288
node session_CopyInto_-4764 session sessionCopyInto:srcOff=2904064,dstOff=16384,n=4096 b1798 -> b1074
node dispatch842 micro raw19 b1804 b1790 b1803 -> b1804
node session_MatmulNTAccelerated_-4765 session sessionMatmulNTAccelerated:M=64,K=8192,N=4096,bOff=0 b1804 b1080 -> b1805
node dispatch843 micro raw18 b1805 b1081 -> b1805
node dispatch844 micro raw20 b1790 b1805 b1803 -> b1790
node session_VecOp_-4767 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4771 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1101 -> b1787
node session_Fp4GemmMetal_-4772 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1663 b1664 -> b1788
node session_CopyInto_-4773 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4774 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1100 -> b1791
node session_CopyInto_-4775 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1100 -> b1791
node session_CopyInto_-4776 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1100 -> b1791
node session_CopyInto_-4777 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1100 -> b1791
node dispatch845 micro raw11 b1791 -> b1791
node dispatch846 micro raw12 b1791 -> b1791
node dispatch847 micro raw13 b1791 -> b1791
node session_CopyInto_-4778 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4779 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1667 b1668 -> b1790
node dispatch848 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4780 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1100
node session_CopyInto_-4781 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1100
node session_VecOp_-4782 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4783 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1119 -> b1787
node session_Fp4GemmMetal_-4784 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1671 b1672 -> b1788
node session_CopyInto_-4785 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4786 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1118 -> b1791
node session_CopyInto_-4787 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1118 -> b1791
node session_CopyInto_-4788 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1118 -> b1791
node session_CopyInto_-4789 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1118 -> b1791
node dispatch849 micro raw11 b1791 -> b1791
node dispatch850 micro raw12 b1791 -> b1791
node dispatch851 micro raw13 b1791 -> b1791
node session_CopyInto_-4790 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4791 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1675 b1676 -> b1790
node dispatch852 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4792 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1118
node session_CopyInto_-4793 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1118
node session_VecOp_-4794 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4795 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1137 -> b1787
node session_Fp4GemmMetal_-4796 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1679 b1680 -> b1788
node session_CopyInto_-4797 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4798 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1136 -> b1791
node session_CopyInto_-4799 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1136 -> b1791
node session_CopyInto_-4800 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1136 -> b1791
node session_CopyInto_-4801 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1136 -> b1791
node dispatch853 micro raw11 b1791 -> b1791
node dispatch854 micro raw12 b1791 -> b1791
node dispatch855 micro raw13 b1791 -> b1791
node session_CopyInto_-4802 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4803 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1683 b1684 -> b1790
node dispatch856 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4804 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1136
node session_CopyInto_-4805 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1136
node session_VecOp_-4806 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4807 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1155 -> b1787
node session_Fp4GemmMetal_-4808 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1687 b1688 -> b1788
node session_CopyInto_-4809 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4810 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1154 -> b1791
node session_CopyInto_-4811 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1154 -> b1791
node session_CopyInto_-4812 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1154 -> b1791
node session_CopyInto_-4813 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1154 -> b1791
node dispatch857 micro raw11 b1791 -> b1791
node dispatch858 micro raw12 b1791 -> b1791
node dispatch859 micro raw13 b1791 -> b1791
node session_CopyInto_-4814 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4815 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1691 b1692 -> b1790
node dispatch860 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4816 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1154
node session_CopyInto_-4817 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1154
node session_VecOp_-4818 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4822 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1173 -> b1787
node session_Fp4GemmMetal_-4823 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1703 b1704 -> b1788
node session_CopyInto_-4824 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4825 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1172 -> b1791
node session_CopyInto_-4826 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1172 -> b1791
node session_CopyInto_-4827 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1172 -> b1791
node session_CopyInto_-4828 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1172 -> b1791
node dispatch861 micro raw11 b1791 -> b1791
node dispatch862 micro raw12 b1791 -> b1791
node dispatch863 micro raw13 b1791 -> b1791
node session_CopyInto_-4829 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4830 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1707 b1708 -> b1790
node dispatch864 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4831 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1172
node session_CopyInto_-4832 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1172
node session_VecOp_-4833 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4834 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1191 -> b1787
node session_Fp4GemmMetal_-4835 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1711 b1712 -> b1788
node session_CopyInto_-4836 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4837 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1190 -> b1791
node session_CopyInto_-4838 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1190 -> b1791
node session_CopyInto_-4839 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1190 -> b1791
node session_CopyInto_-4840 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1190 -> b1791
node dispatch865 micro raw11 b1791 -> b1791
node dispatch866 micro raw12 b1791 -> b1791
node dispatch867 micro raw13 b1791 -> b1791
node session_CopyInto_-4841 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4842 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1715 b1716 -> b1790
node dispatch868 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4843 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1190
node session_CopyInto_-4844 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1190
node session_VecOp_-4845 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4846 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1209 -> b1787
node session_Fp4GemmMetal_-4847 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1719 b1720 -> b1788
node session_CopyInto_-4848 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4849 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1208 -> b1791
node session_CopyInto_-4850 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1208 -> b1791
node session_CopyInto_-4851 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1208 -> b1791
node session_CopyInto_-4852 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1208 -> b1791
node dispatch869 micro raw11 b1791 -> b1791
node dispatch870 micro raw12 b1791 -> b1791
node dispatch871 micro raw13 b1791 -> b1791
node session_CopyInto_-4853 session sessionCopyInto:srcOff=6237056,dstOff=0,n=524288 b1791 -> b1789
node session_Fp4GemmMetal_-4854 session sessionFp4GemmMetal:M=64,K=8192,N=4096,simdgroup=0 b1789 b1723 b1724 -> b1790
node dispatch872 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4855 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1208
node session_CopyInto_-4856 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1208
node session_VecOp_-4857 session sessionVecOp:op=add,n=262144 b1786 b1790 -> b1786
node session_RmsnormEnc_-4858 session sessionRmsnormEnc:rows=64,cols=4096,eps=h3727c5ac b1786 b1227 -> b1787
node session_Fp4GemmMetal_-4859 session sessionFp4GemmMetal:M=64,K=4096,N=18560,simdgroup=0 b1787 b1727 b1728 -> b1788
node session_CopyInto_-4860 session sessionCopyInto:srcOff=0,dstOff=0,n=1187840 b1788 -> b1791
node session_CopyInto_-4861 session sessionCopyInto:srcOff=45696,dstOff=1187840,n=51200 b1226 -> b1791
node session_CopyInto_-4862 session sessionCopyInto:srcOff=37120,dstOff=1239040,n=8576 b1226 -> b1791
node session_CopyInto_-4863 session sessionCopyInto:srcOff=127616,dstOff=2566528,n=1048576 b1226 -> b1791
node session_CopyInto_-4864 session sessionCopyInto:srcOff=96896,dstOff=6761344,n=30720 b1226 -> b1791
node dispatch873 micro raw11 b1791 -> b1791
node dispatch874 micro raw12 b1791 -> b1791
node dispatch875 micro raw13 b1791 -> b1791
node dispatch876 micro raw14 b1792 b1788 -> b1792
node session_CopyInto_-4867 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1792 -> b1226
node session_CopyInto_-4868 session sessionCopyInto:srcOff=3615104,dstOff=127616,n=1048576 b1791 -> b1226
phase chunk3 rows=32
node embedding_3 ext fir.embedding token backbone_embeddings_weight_emb -> b1819
node session_CopyInto_-4913 session sessionCopyInto:srcOff=0,dstOff=0,n=131072 b1819 -> b1810
node session_RmsnormEnc_-4940 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b24 -> b1811
node session_Fp4GemmMetal_-4941 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1247 b1248 -> b1812
node session_CopyInto_-4942 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-4943 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b23 -> b1815
node session_CopyInto_-4944 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b23 -> b1815
node session_CopyInto_-4945 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b23 -> b1815
node session_CopyInto_-4946 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b23 -> b1815
node dispatch883 micro raw21 b1815 -> b1815
node dispatch884 micro raw22 b1815 -> b1815
node dispatch885 micro raw23 b1815 -> b1815
node session_CopyInto_-4947 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-4948 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1251 b1252 -> b1814
node dispatch886 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-4949 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b23
node session_CopyInto_-4950 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b23
node session_VecOp_-4951 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-4952 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b42 -> b1811
node session_Fp4GemmMetal_-4953 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1255 b1256 -> b1812
node session_CopyInto_-4954 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-4955 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b41 -> b1815
node session_CopyInto_-4956 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b41 -> b1815
node session_CopyInto_-4957 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b41 -> b1815
node session_CopyInto_-4958 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b41 -> b1815
node dispatch887 micro raw21 b1815 -> b1815
node dispatch888 micro raw22 b1815 -> b1815
node dispatch889 micro raw23 b1815 -> b1815
node session_CopyInto_-4959 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-4960 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1259 b1260 -> b1814
node dispatch890 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-4961 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b41
node session_CopyInto_-4962 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b41
node session_VecOp_-4963 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-4964 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b60 -> b1811
node session_Fp4GemmMetal_-4965 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1343 b1344 -> b1812
node session_CopyInto_-4966 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-4967 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b59 -> b1815
node session_CopyInto_-4968 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b59 -> b1815
node session_CopyInto_-4969 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b59 -> b1815
node session_CopyInto_-4970 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b59 -> b1815
node dispatch891 micro raw21 b1815 -> b1815
node dispatch892 micro raw22 b1815 -> b1815
node dispatch893 micro raw23 b1815 -> b1815
node session_CopyInto_-4971 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-4972 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1347 b1348 -> b1814
node dispatch894 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-4973 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b59
node session_CopyInto_-4974 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b59
node session_VecOp_-4975 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-4976 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b78 -> b1811
node session_Fp4GemmMetal_-4977 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1431 b1432 -> b1812
node session_CopyInto_-4978 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-4979 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b77 -> b1815
node session_CopyInto_-4980 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b77 -> b1815
node session_CopyInto_-4981 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b77 -> b1815
node session_CopyInto_-4982 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b77 -> b1815
node dispatch895 micro raw21 b1815 -> b1815
node dispatch896 micro raw22 b1815 -> b1815
node dispatch897 micro raw23 b1815 -> b1815
node session_CopyInto_-4983 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-4984 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1435 b1436 -> b1814
node dispatch898 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-4985 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b77
node session_CopyInto_-4986 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b77
node session_VecOp_-4987 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-4991 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b96 -> b1811
node session_Fp4GemmMetal_-4992 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1519 b1520 -> b1812
node session_CopyInto_-4993 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-4994 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b95 -> b1815
node session_CopyInto_-4995 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b95 -> b1815
node session_CopyInto_-4996 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b95 -> b1815
node session_CopyInto_-4997 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b95 -> b1815
node dispatch899 micro raw21 b1815 -> b1815
node dispatch900 micro raw22 b1815 -> b1815
node dispatch901 micro raw23 b1815 -> b1815
node session_CopyInto_-4998 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-4999 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1523 b1524 -> b1814
node dispatch902 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5000 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b95
node session_CopyInto_-5001 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b95
node session_VecOp_-5002 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5003 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b114 -> b1811
node session_Fp4GemmMetal_-5004 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1607 b1608 -> b1812
node session_CopyInto_-5005 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5006 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b113 -> b1815
node session_CopyInto_-5007 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b113 -> b1815
node session_CopyInto_-5008 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b113 -> b1815
node session_CopyInto_-5009 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b113 -> b1815
node dispatch903 micro raw21 b1815 -> b1815
node dispatch904 micro raw22 b1815 -> b1815
node dispatch905 micro raw23 b1815 -> b1815
node session_CopyInto_-5010 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5011 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1611 b1612 -> b1814
node dispatch906 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5012 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b113
node session_CopyInto_-5013 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b113
node session_VecOp_-5014 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5015 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b132 -> b1811
node session_Fp4GemmMetal_-5016 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1695 b1696 -> b1812
node session_CopyInto_-5017 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5018 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b131 -> b1815
node session_CopyInto_-5019 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b131 -> b1815
node session_CopyInto_-5020 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b131 -> b1815
node session_CopyInto_-5021 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b131 -> b1815
node dispatch907 micro raw21 b1815 -> b1815
node dispatch908 micro raw22 b1815 -> b1815
node dispatch909 micro raw23 b1815 -> b1815
node session_CopyInto_-5022 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5023 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1699 b1700 -> b1814
node dispatch910 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5024 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b131
node session_CopyInto_-5025 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b131
node session_VecOp_-5026 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5027 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b150 -> b1811
node session_Fp4GemmMetal_-5028 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1735 b1736 -> b1812
node session_CopyInto_-5029 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5030 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b149 -> b1815
node session_CopyInto_-5031 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b149 -> b1815
node session_CopyInto_-5032 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b149 -> b1815
node session_CopyInto_-5033 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b149 -> b1815
node dispatch911 micro raw21 b1815 -> b1815
node dispatch912 micro raw22 b1815 -> b1815
node dispatch913 micro raw23 b1815 -> b1815
node session_CopyInto_-5034 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5035 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1739 b1740 -> b1814
node dispatch914 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5036 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b149
node session_CopyInto_-5037 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b149
node session_MatmulNTAccelerated_-5038 session sessionMatmulNTAccelerated:M=32,K=4096,N=12288,bOff=0 b1811 b154 -> b1825
node dispatch915 micro raw25 b1825 b155 -> b1825
node session_CopyInto_-5040 session sessionCopyInto:srcOff=0,dstOff=0,n=393216 b1825 -> b1822
node session_CopyInto_-5041 session sessionCopyInto:srcOff=16384,dstOff=1318912,n=4096 b158 -> b1822
node session_CopyInto_-5042 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b158 -> b1830
node dispatch916 micro raw5 b1822_5292032_2129920 b1830 -> b1822_5292032_2129920
node dispatch917 micro raw26 b1822 -> b1822
node dispatch918 micro raw27 b1822 -> b1822
node session_CopyInto_-5043 session sessionCopyInto:srcOff=655360,dstOff=0,n=131072 b1822 -> b1826
node session_MatmulNTAccelerated_-5044 session sessionMatmulNTAccelerated:M=32,K=4096,N=4096,bOff=0 b1826 b156 -> b1827
node dispatch919 micro raw28 b1827 b157 -> b1827
node session_CopyInto_-5046 session sessionCopyInto:srcOff=1323008,dstOff=0,n=131072 b1822 -> b1830
node dispatch920 micro raw5 b158_81920_524288 b1830 -> b158_81920_524288
node session_CopyInto_-5047 session sessionCopyInto:srcOff=1585152,dstOff=16384,n=4096 b1822 -> b158
node dispatch921 micro raw29 b1828 b1814 b1827 -> b1828
node session_MatmulNTAccelerated_-5048 session sessionMatmulNTAccelerated:M=32,K=8192,N=4096,bOff=0 b1828 b165 -> b1829
node dispatch922 micro raw28 b1829 b166 -> b1829
node dispatch923 micro raw30 b1814 b1829 b1827 -> b1814
node session_VecOp_-5050 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5054 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b186 -> b1811
node session_Fp4GemmMetal_-5055 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1743 b1744 -> b1812
node session_CopyInto_-5056 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5057 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b185 -> b1815
node session_CopyInto_-5058 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b185 -> b1815
node session_CopyInto_-5059 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b185 -> b1815
node session_CopyInto_-5060 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b185 -> b1815
node dispatch924 micro raw21 b1815 -> b1815
node dispatch925 micro raw22 b1815 -> b1815
node dispatch926 micro raw23 b1815 -> b1815
node session_CopyInto_-5061 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5062 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1747 b1748 -> b1814
node dispatch927 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5063 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b185
node session_CopyInto_-5064 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b185
node session_VecOp_-5065 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5066 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b204 -> b1811
node session_Fp4GemmMetal_-5067 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1751 b1752 -> b1812
node session_CopyInto_-5068 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5069 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b203 -> b1815
node session_CopyInto_-5070 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b203 -> b1815
node session_CopyInto_-5071 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b203 -> b1815
node session_CopyInto_-5072 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b203 -> b1815
node dispatch928 micro raw21 b1815 -> b1815
node dispatch929 micro raw22 b1815 -> b1815
node dispatch930 micro raw23 b1815 -> b1815
node session_CopyInto_-5073 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5074 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1755 b1756 -> b1814
node dispatch931 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5075 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b203
node session_CopyInto_-5076 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b203
node session_VecOp_-5077 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5078 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b222 -> b1811
node session_Fp4GemmMetal_-5079 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1263 b1264 -> b1812
node session_CopyInto_-5080 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5081 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b221 -> b1815
node session_CopyInto_-5082 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b221 -> b1815
node session_CopyInto_-5083 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b221 -> b1815
node session_CopyInto_-5084 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b221 -> b1815
node dispatch932 micro raw21 b1815 -> b1815
node dispatch933 micro raw22 b1815 -> b1815
node dispatch934 micro raw23 b1815 -> b1815
node session_CopyInto_-5085 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5086 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1267 b1268 -> b1814
node dispatch935 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5087 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b221
node session_CopyInto_-5088 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b221
node session_VecOp_-5089 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5090 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b240 -> b1811
node session_Fp4GemmMetal_-5091 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1271 b1272 -> b1812
node session_CopyInto_-5092 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5093 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b239 -> b1815
node session_CopyInto_-5094 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b239 -> b1815
node session_CopyInto_-5095 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b239 -> b1815
node session_CopyInto_-5096 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b239 -> b1815
node dispatch936 micro raw21 b1815 -> b1815
node dispatch937 micro raw22 b1815 -> b1815
node dispatch938 micro raw23 b1815 -> b1815
node session_CopyInto_-5097 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5098 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1275 b1276 -> b1814
node dispatch939 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5099 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b239
node session_CopyInto_-5100 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b239
node session_VecOp_-5101 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5105 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b258 -> b1811
node session_Fp4GemmMetal_-5106 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1279 b1280 -> b1812
node session_CopyInto_-5107 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5108 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b257 -> b1815
node session_CopyInto_-5109 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b257 -> b1815
node session_CopyInto_-5110 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b257 -> b1815
node session_CopyInto_-5111 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b257 -> b1815
node dispatch940 micro raw21 b1815 -> b1815
node dispatch941 micro raw22 b1815 -> b1815
node dispatch942 micro raw23 b1815 -> b1815
node session_CopyInto_-5112 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5113 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1283 b1284 -> b1814
node dispatch943 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5114 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b257
node session_CopyInto_-5115 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b257
node session_VecOp_-5116 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5117 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b276 -> b1811
node session_Fp4GemmMetal_-5118 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1287 b1288 -> b1812
node session_CopyInto_-5119 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5120 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b275 -> b1815
node session_CopyInto_-5121 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b275 -> b1815
node session_CopyInto_-5122 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b275 -> b1815
node session_CopyInto_-5123 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b275 -> b1815
node dispatch944 micro raw21 b1815 -> b1815
node dispatch945 micro raw22 b1815 -> b1815
node dispatch946 micro raw23 b1815 -> b1815
node session_CopyInto_-5124 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5125 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1291 b1292 -> b1814
node dispatch947 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5126 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b275
node session_CopyInto_-5127 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b275
node session_VecOp_-5128 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5129 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b294 -> b1811
node session_Fp4GemmMetal_-5130 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1295 b1296 -> b1812
node session_CopyInto_-5131 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5132 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b293 -> b1815
node session_CopyInto_-5133 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b293 -> b1815
node session_CopyInto_-5134 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b293 -> b1815
node session_CopyInto_-5135 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b293 -> b1815
node dispatch948 micro raw21 b1815 -> b1815
node dispatch949 micro raw22 b1815 -> b1815
node dispatch950 micro raw23 b1815 -> b1815
node session_CopyInto_-5136 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5137 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1299 b1300 -> b1814
node dispatch951 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5138 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b293
node session_CopyInto_-5139 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b293
node session_VecOp_-5140 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5141 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b312 -> b1811
node session_Fp4GemmMetal_-5142 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1303 b1304 -> b1812
node session_CopyInto_-5143 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5144 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b311 -> b1815
node session_CopyInto_-5145 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b311 -> b1815
node session_CopyInto_-5146 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b311 -> b1815
node session_CopyInto_-5147 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b311 -> b1815
node dispatch952 micro raw21 b1815 -> b1815
node dispatch953 micro raw22 b1815 -> b1815
node dispatch954 micro raw23 b1815 -> b1815
node session_CopyInto_-5148 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5149 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1307 b1308 -> b1814
node dispatch955 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5150 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b311
node session_CopyInto_-5151 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b311
node session_VecOp_-5152 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5156 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b330 -> b1811
node session_Fp4GemmMetal_-5157 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1311 b1312 -> b1812
node session_CopyInto_-5158 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5159 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b329 -> b1815
node session_CopyInto_-5160 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b329 -> b1815
node session_CopyInto_-5161 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b329 -> b1815
node session_CopyInto_-5162 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b329 -> b1815
node dispatch956 micro raw21 b1815 -> b1815
node dispatch957 micro raw22 b1815 -> b1815
node dispatch958 micro raw23 b1815 -> b1815
node session_CopyInto_-5163 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5164 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1315 b1316 -> b1814
node dispatch959 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5165 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b329
node session_CopyInto_-5166 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b329
node session_VecOp_-5167 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5168 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b348 -> b1811
node session_Fp4GemmMetal_-5169 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1319 b1320 -> b1812
node session_CopyInto_-5170 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5171 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b347 -> b1815
node session_CopyInto_-5172 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b347 -> b1815
node session_CopyInto_-5173 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b347 -> b1815
node session_CopyInto_-5174 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b347 -> b1815
node dispatch960 micro raw21 b1815 -> b1815
node dispatch961 micro raw22 b1815 -> b1815
node dispatch962 micro raw23 b1815 -> b1815
node session_CopyInto_-5175 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5176 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1323 b1324 -> b1814
node dispatch963 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5177 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b347
node session_CopyInto_-5178 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b347
node session_VecOp_-5179 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5180 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b366 -> b1811
node session_Fp4GemmMetal_-5181 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1327 b1328 -> b1812
node session_CopyInto_-5182 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5183 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b365 -> b1815
node session_CopyInto_-5184 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b365 -> b1815
node session_CopyInto_-5185 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b365 -> b1815
node session_CopyInto_-5186 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b365 -> b1815
node dispatch964 micro raw21 b1815 -> b1815
node dispatch965 micro raw22 b1815 -> b1815
node dispatch966 micro raw23 b1815 -> b1815
node session_CopyInto_-5187 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5188 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1331 b1332 -> b1814
node dispatch967 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5189 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b365
node session_CopyInto_-5190 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b365
node session_VecOp_-5191 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5192 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b384 -> b1811
node session_Fp4GemmMetal_-5193 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1335 b1336 -> b1812
node session_CopyInto_-5194 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5195 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b383 -> b1815
node session_CopyInto_-5196 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b383 -> b1815
node session_CopyInto_-5197 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b383 -> b1815
node session_CopyInto_-5198 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b383 -> b1815
node dispatch968 micro raw21 b1815 -> b1815
node dispatch969 micro raw22 b1815 -> b1815
node dispatch970 micro raw23 b1815 -> b1815
node session_CopyInto_-5199 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5200 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1339 b1340 -> b1814
node dispatch971 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5201 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b383
node session_CopyInto_-5202 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b383
node session_VecOp_-5203 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5207 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b402 -> b1811
node session_Fp4GemmMetal_-5208 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1351 b1352 -> b1812
node session_CopyInto_-5209 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5210 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b401 -> b1815
node session_CopyInto_-5211 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b401 -> b1815
node session_CopyInto_-5212 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b401 -> b1815
node session_CopyInto_-5213 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b401 -> b1815
node dispatch972 micro raw21 b1815 -> b1815
node dispatch973 micro raw22 b1815 -> b1815
node dispatch974 micro raw23 b1815 -> b1815
node session_CopyInto_-5214 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5215 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1355 b1356 -> b1814
node dispatch975 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5216 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b401
node session_CopyInto_-5217 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b401
node session_VecOp_-5218 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5219 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b420 -> b1811
node session_Fp4GemmMetal_-5220 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1359 b1360 -> b1812
node session_CopyInto_-5221 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5222 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b419 -> b1815
node session_CopyInto_-5223 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b419 -> b1815
node session_CopyInto_-5224 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b419 -> b1815
node session_CopyInto_-5225 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b419 -> b1815
node dispatch976 micro raw21 b1815 -> b1815
node dispatch977 micro raw22 b1815 -> b1815
node dispatch978 micro raw23 b1815 -> b1815
node session_CopyInto_-5226 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5227 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1363 b1364 -> b1814
node dispatch979 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5228 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b419
node session_CopyInto_-5229 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b419
node session_VecOp_-5230 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5231 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b438 -> b1811
node session_Fp4GemmMetal_-5232 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1367 b1368 -> b1812
node session_CopyInto_-5233 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5234 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b437 -> b1815
node session_CopyInto_-5235 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b437 -> b1815
node session_CopyInto_-5236 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b437 -> b1815
node session_CopyInto_-5237 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b437 -> b1815
node dispatch980 micro raw21 b1815 -> b1815
node dispatch981 micro raw22 b1815 -> b1815
node dispatch982 micro raw23 b1815 -> b1815
node session_CopyInto_-5238 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5239 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1371 b1372 -> b1814
node dispatch983 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5240 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b437
node session_CopyInto_-5241 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b437
node session_VecOp_-5242 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5243 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b456 -> b1811
node session_Fp4GemmMetal_-5244 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1375 b1376 -> b1812
node session_CopyInto_-5245 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5246 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b455 -> b1815
node session_CopyInto_-5247 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b455 -> b1815
node session_CopyInto_-5248 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b455 -> b1815
node session_CopyInto_-5249 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b455 -> b1815
node dispatch984 micro raw21 b1815 -> b1815
node dispatch985 micro raw22 b1815 -> b1815
node dispatch986 micro raw23 b1815 -> b1815
node session_CopyInto_-5250 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5251 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1379 b1380 -> b1814
node dispatch987 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5252 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b455
node session_CopyInto_-5253 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b455
node session_MatmulNTAccelerated_-5254 session sessionMatmulNTAccelerated:M=32,K=4096,N=12288,bOff=0 b1811 b460 -> b1825
node dispatch988 micro raw25 b1825 b461 -> b1825
node session_CopyInto_-5256 session sessionCopyInto:srcOff=0,dstOff=0,n=393216 b1825 -> b1822
node session_CopyInto_-5257 session sessionCopyInto:srcOff=16384,dstOff=1318912,n=4096 b464 -> b1822
node session_CopyInto_-5258 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b464 -> b1830
node dispatch989 micro raw5 b1822_5292032_2129920 b1830 -> b1822_5292032_2129920
node dispatch990 micro raw26 b1822 -> b1822
node dispatch991 micro raw27 b1822 -> b1822
node session_CopyInto_-5259 session sessionCopyInto:srcOff=655360,dstOff=0,n=131072 b1822 -> b1826
node session_MatmulNTAccelerated_-5260 session sessionMatmulNTAccelerated:M=32,K=4096,N=4096,bOff=0 b1826 b462 -> b1827
node dispatch992 micro raw28 b1827 b463 -> b1827
node session_CopyInto_-5262 session sessionCopyInto:srcOff=1323008,dstOff=0,n=131072 b1822 -> b1830
node dispatch993 micro raw5 b464_81920_524288 b1830 -> b464_81920_524288
node session_CopyInto_-5263 session sessionCopyInto:srcOff=1585152,dstOff=16384,n=4096 b1822 -> b464
node dispatch994 micro raw29 b1828 b1814 b1827 -> b1828
node session_MatmulNTAccelerated_-5264 session sessionMatmulNTAccelerated:M=32,K=8192,N=4096,bOff=0 b1828 b470 -> b1829
node dispatch995 micro raw28 b1829 b471 -> b1829
node dispatch996 micro raw30 b1814 b1829 b1827 -> b1814
node session_VecOp_-5266 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5270 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b491 -> b1811
node session_Fp4GemmMetal_-5271 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1383 b1384 -> b1812
node session_CopyInto_-5272 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5273 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b490 -> b1815
node session_CopyInto_-5274 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b490 -> b1815
node session_CopyInto_-5275 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b490 -> b1815
node session_CopyInto_-5276 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b490 -> b1815
node dispatch997 micro raw21 b1815 -> b1815
node dispatch998 micro raw22 b1815 -> b1815
node dispatch999 micro raw23 b1815 -> b1815
node session_CopyInto_-5277 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5278 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1387 b1388 -> b1814
node dispatch1000 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5279 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b490
node session_CopyInto_-5280 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b490
node session_VecOp_-5281 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5282 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b509 -> b1811
node session_Fp4GemmMetal_-5283 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1391 b1392 -> b1812
node session_CopyInto_-5284 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5285 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b508 -> b1815
node session_CopyInto_-5286 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b508 -> b1815
node session_CopyInto_-5287 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b508 -> b1815
node session_CopyInto_-5288 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b508 -> b1815
node dispatch1001 micro raw21 b1815 -> b1815
node dispatch1002 micro raw22 b1815 -> b1815
node dispatch1003 micro raw23 b1815 -> b1815
node session_CopyInto_-5289 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5290 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1395 b1396 -> b1814
node dispatch1004 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5291 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b508
node session_CopyInto_-5292 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b508
node session_VecOp_-5293 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5294 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b527 -> b1811
node session_Fp4GemmMetal_-5295 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1399 b1400 -> b1812
node session_CopyInto_-5296 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5297 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b526 -> b1815
node session_CopyInto_-5298 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b526 -> b1815
node session_CopyInto_-5299 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b526 -> b1815
node session_CopyInto_-5300 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b526 -> b1815
node dispatch1005 micro raw21 b1815 -> b1815
node dispatch1006 micro raw22 b1815 -> b1815
node dispatch1007 micro raw23 b1815 -> b1815
node session_CopyInto_-5301 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5302 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1403 b1404 -> b1814
node dispatch1008 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5303 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b526
node session_CopyInto_-5304 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b526
node session_VecOp_-5305 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5306 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b545 -> b1811
node session_Fp4GemmMetal_-5307 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1407 b1408 -> b1812
node session_CopyInto_-5308 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5309 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b544 -> b1815
node session_CopyInto_-5310 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b544 -> b1815
node session_CopyInto_-5311 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b544 -> b1815
node session_CopyInto_-5312 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b544 -> b1815
node dispatch1009 micro raw21 b1815 -> b1815
node dispatch1010 micro raw22 b1815 -> b1815
node dispatch1011 micro raw23 b1815 -> b1815
node session_CopyInto_-5313 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5314 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1411 b1412 -> b1814
node dispatch1012 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5315 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b544
node session_CopyInto_-5316 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b544
node session_VecOp_-5317 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5321 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b563 -> b1811
node session_Fp4GemmMetal_-5322 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1415 b1416 -> b1812
node session_CopyInto_-5323 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5324 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b562 -> b1815
node session_CopyInto_-5325 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b562 -> b1815
node session_CopyInto_-5326 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b562 -> b1815
node session_CopyInto_-5327 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b562 -> b1815
node dispatch1013 micro raw21 b1815 -> b1815
node dispatch1014 micro raw22 b1815 -> b1815
node dispatch1015 micro raw23 b1815 -> b1815
node session_CopyInto_-5328 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5329 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1419 b1420 -> b1814
node dispatch1016 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5330 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b562
node session_CopyInto_-5331 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b562
node session_VecOp_-5332 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5333 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b581 -> b1811
node session_Fp4GemmMetal_-5334 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1423 b1424 -> b1812
node session_CopyInto_-5335 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5336 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b580 -> b1815
node session_CopyInto_-5337 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b580 -> b1815
node session_CopyInto_-5338 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b580 -> b1815
node session_CopyInto_-5339 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b580 -> b1815
node dispatch1017 micro raw21 b1815 -> b1815
node dispatch1018 micro raw22 b1815 -> b1815
node dispatch1019 micro raw23 b1815 -> b1815
node session_CopyInto_-5340 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5341 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1427 b1428 -> b1814
node dispatch1020 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5342 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b580
node session_CopyInto_-5343 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b580
node session_VecOp_-5344 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5345 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b599 -> b1811
node session_Fp4GemmMetal_-5346 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1439 b1440 -> b1812
node session_CopyInto_-5347 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5348 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b598 -> b1815
node session_CopyInto_-5349 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b598 -> b1815
node session_CopyInto_-5350 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b598 -> b1815
node session_CopyInto_-5351 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b598 -> b1815
node dispatch1021 micro raw21 b1815 -> b1815
node dispatch1022 micro raw22 b1815 -> b1815
node dispatch1023 micro raw23 b1815 -> b1815
node session_CopyInto_-5352 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5353 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1443 b1444 -> b1814
node dispatch1024 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5354 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b598
node session_CopyInto_-5355 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b598
node session_VecOp_-5356 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5357 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b617 -> b1811
node session_Fp4GemmMetal_-5358 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1447 b1448 -> b1812
node session_CopyInto_-5359 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5360 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b616 -> b1815
node session_CopyInto_-5361 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b616 -> b1815
node session_CopyInto_-5362 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b616 -> b1815
node session_CopyInto_-5363 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b616 -> b1815
node dispatch1025 micro raw21 b1815 -> b1815
node dispatch1026 micro raw22 b1815 -> b1815
node dispatch1027 micro raw23 b1815 -> b1815
node session_CopyInto_-5364 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5365 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1451 b1452 -> b1814
node dispatch1028 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5366 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b616
node session_CopyInto_-5367 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b616
node session_VecOp_-5368 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5372 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b635 -> b1811
node session_Fp4GemmMetal_-5373 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1455 b1456 -> b1812
node session_CopyInto_-5374 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5375 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b634 -> b1815
node session_CopyInto_-5376 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b634 -> b1815
node session_CopyInto_-5377 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b634 -> b1815
node session_CopyInto_-5378 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b634 -> b1815
node dispatch1029 micro raw21 b1815 -> b1815
node dispatch1030 micro raw22 b1815 -> b1815
node dispatch1031 micro raw23 b1815 -> b1815
node session_CopyInto_-5379 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5380 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1459 b1460 -> b1814
node dispatch1032 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5381 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b634
node session_CopyInto_-5382 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b634
node session_VecOp_-5383 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5384 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b653 -> b1811
node session_Fp4GemmMetal_-5385 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1463 b1464 -> b1812
node session_CopyInto_-5386 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5387 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b652 -> b1815
node session_CopyInto_-5388 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b652 -> b1815
node session_CopyInto_-5389 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b652 -> b1815
node session_CopyInto_-5390 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b652 -> b1815
node dispatch1033 micro raw21 b1815 -> b1815
node dispatch1034 micro raw22 b1815 -> b1815
node dispatch1035 micro raw23 b1815 -> b1815
node session_CopyInto_-5391 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5392 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1467 b1468 -> b1814
node dispatch1036 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5393 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b652
node session_CopyInto_-5394 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b652
node session_VecOp_-5395 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5396 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b671 -> b1811
node session_Fp4GemmMetal_-5397 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1471 b1472 -> b1812
node session_CopyInto_-5398 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5399 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b670 -> b1815
node session_CopyInto_-5400 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b670 -> b1815
node session_CopyInto_-5401 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b670 -> b1815
node session_CopyInto_-5402 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b670 -> b1815
node dispatch1037 micro raw21 b1815 -> b1815
node dispatch1038 micro raw22 b1815 -> b1815
node dispatch1039 micro raw23 b1815 -> b1815
node session_CopyInto_-5403 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5404 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1475 b1476 -> b1814
node dispatch1040 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5405 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b670
node session_CopyInto_-5406 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b670
node session_VecOp_-5407 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5408 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b689 -> b1811
node session_Fp4GemmMetal_-5409 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1479 b1480 -> b1812
node session_CopyInto_-5410 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5411 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b688 -> b1815
node session_CopyInto_-5412 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b688 -> b1815
node session_CopyInto_-5413 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b688 -> b1815
node session_CopyInto_-5414 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b688 -> b1815
node dispatch1041 micro raw21 b1815 -> b1815
node dispatch1042 micro raw22 b1815 -> b1815
node dispatch1043 micro raw23 b1815 -> b1815
node session_CopyInto_-5415 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5416 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1483 b1484 -> b1814
node dispatch1044 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5417 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b688
node session_CopyInto_-5418 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b688
node session_VecOp_-5419 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5423 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b707 -> b1811
node session_Fp4GemmMetal_-5424 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1487 b1488 -> b1812
node session_CopyInto_-5425 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5426 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b706 -> b1815
node session_CopyInto_-5427 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b706 -> b1815
node session_CopyInto_-5428 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b706 -> b1815
node session_CopyInto_-5429 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b706 -> b1815
node dispatch1045 micro raw21 b1815 -> b1815
node dispatch1046 micro raw22 b1815 -> b1815
node dispatch1047 micro raw23 b1815 -> b1815
node session_CopyInto_-5430 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5431 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1491 b1492 -> b1814
node dispatch1048 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5432 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b706
node session_CopyInto_-5433 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b706
node session_VecOp_-5434 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5435 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b725 -> b1811
node session_Fp4GemmMetal_-5436 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1495 b1496 -> b1812
node session_CopyInto_-5437 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5438 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b724 -> b1815
node session_CopyInto_-5439 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b724 -> b1815
node session_CopyInto_-5440 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b724 -> b1815
node session_CopyInto_-5441 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b724 -> b1815
node dispatch1049 micro raw21 b1815 -> b1815
node dispatch1050 micro raw22 b1815 -> b1815
node dispatch1051 micro raw23 b1815 -> b1815
node session_CopyInto_-5442 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5443 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1499 b1500 -> b1814
node dispatch1052 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5444 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b724
node session_CopyInto_-5445 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b724
node session_VecOp_-5446 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5447 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b743 -> b1811
node session_Fp4GemmMetal_-5448 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1503 b1504 -> b1812
node session_CopyInto_-5449 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5450 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b742 -> b1815
node session_CopyInto_-5451 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b742 -> b1815
node session_CopyInto_-5452 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b742 -> b1815
node session_CopyInto_-5453 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b742 -> b1815
node dispatch1053 micro raw21 b1815 -> b1815
node dispatch1054 micro raw22 b1815 -> b1815
node dispatch1055 micro raw23 b1815 -> b1815
node session_CopyInto_-5454 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5455 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1507 b1508 -> b1814
node dispatch1056 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5456 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b742
node session_CopyInto_-5457 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b742
node session_VecOp_-5458 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5459 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b761 -> b1811
node session_Fp4GemmMetal_-5460 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1511 b1512 -> b1812
node session_CopyInto_-5461 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5462 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b760 -> b1815
node session_CopyInto_-5463 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b760 -> b1815
node session_CopyInto_-5464 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b760 -> b1815
node session_CopyInto_-5465 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b760 -> b1815
node dispatch1057 micro raw21 b1815 -> b1815
node dispatch1058 micro raw22 b1815 -> b1815
node dispatch1059 micro raw23 b1815 -> b1815
node session_CopyInto_-5466 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5467 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1515 b1516 -> b1814
node dispatch1060 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5468 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b760
node session_CopyInto_-5469 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b760
node session_MatmulNTAccelerated_-5470 session sessionMatmulNTAccelerated:M=32,K=4096,N=12288,bOff=0 b1811 b765 -> b1825
node dispatch1061 micro raw25 b1825 b766 -> b1825
node session_CopyInto_-5472 session sessionCopyInto:srcOff=0,dstOff=0,n=393216 b1825 -> b1822
node session_CopyInto_-5473 session sessionCopyInto:srcOff=16384,dstOff=1318912,n=4096 b769 -> b1822
node session_CopyInto_-5474 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b769 -> b1830
node dispatch1062 micro raw5 b1822_5292032_2129920 b1830 -> b1822_5292032_2129920
node dispatch1063 micro raw26 b1822 -> b1822
node dispatch1064 micro raw27 b1822 -> b1822
node session_CopyInto_-5475 session sessionCopyInto:srcOff=655360,dstOff=0,n=131072 b1822 -> b1826
node session_MatmulNTAccelerated_-5476 session sessionMatmulNTAccelerated:M=32,K=4096,N=4096,bOff=0 b1826 b767 -> b1827
node dispatch1065 micro raw28 b1827 b768 -> b1827
node session_CopyInto_-5478 session sessionCopyInto:srcOff=1323008,dstOff=0,n=131072 b1822 -> b1830
node dispatch1066 micro raw5 b769_81920_524288 b1830 -> b769_81920_524288
node session_CopyInto_-5479 session sessionCopyInto:srcOff=1585152,dstOff=16384,n=4096 b1822 -> b769
node dispatch1067 micro raw29 b1828 b1814 b1827 -> b1828
node session_MatmulNTAccelerated_-5480 session sessionMatmulNTAccelerated:M=32,K=8192,N=4096,bOff=0 b1828 b775 -> b1829
node dispatch1068 micro raw28 b1829 b776 -> b1829
node dispatch1069 micro raw30 b1814 b1829 b1827 -> b1814
node session_VecOp_-5482 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5486 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b796 -> b1811
node session_Fp4GemmMetal_-5487 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1527 b1528 -> b1812
node session_CopyInto_-5488 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5489 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b795 -> b1815
node session_CopyInto_-5490 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b795 -> b1815
node session_CopyInto_-5491 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b795 -> b1815
node session_CopyInto_-5492 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b795 -> b1815
node dispatch1070 micro raw21 b1815 -> b1815
node dispatch1071 micro raw22 b1815 -> b1815
node dispatch1072 micro raw23 b1815 -> b1815
node session_CopyInto_-5493 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5494 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1531 b1532 -> b1814
node dispatch1073 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5495 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b795
node session_CopyInto_-5496 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b795
node session_VecOp_-5497 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5498 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b814 -> b1811
node session_Fp4GemmMetal_-5499 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1535 b1536 -> b1812
node session_CopyInto_-5500 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5501 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b813 -> b1815
node session_CopyInto_-5502 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b813 -> b1815
node session_CopyInto_-5503 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b813 -> b1815
node session_CopyInto_-5504 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b813 -> b1815
node dispatch1074 micro raw21 b1815 -> b1815
node dispatch1075 micro raw22 b1815 -> b1815
node dispatch1076 micro raw23 b1815 -> b1815
node session_CopyInto_-5505 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5506 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1539 b1540 -> b1814
node dispatch1077 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5507 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b813
node session_CopyInto_-5508 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b813
node session_VecOp_-5509 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5510 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b832 -> b1811
node session_Fp4GemmMetal_-5511 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1543 b1544 -> b1812
node session_CopyInto_-5512 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5513 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b831 -> b1815
node session_CopyInto_-5514 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b831 -> b1815
node session_CopyInto_-5515 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b831 -> b1815
node session_CopyInto_-5516 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b831 -> b1815
node dispatch1078 micro raw21 b1815 -> b1815
node dispatch1079 micro raw22 b1815 -> b1815
node dispatch1080 micro raw23 b1815 -> b1815
node session_CopyInto_-5517 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5518 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1547 b1548 -> b1814
node dispatch1081 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5519 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b831
node session_CopyInto_-5520 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b831
node session_VecOp_-5521 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5522 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b850 -> b1811
node session_Fp4GemmMetal_-5523 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1551 b1552 -> b1812
node session_CopyInto_-5524 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5525 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b849 -> b1815
node session_CopyInto_-5526 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b849 -> b1815
node session_CopyInto_-5527 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b849 -> b1815
node session_CopyInto_-5528 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b849 -> b1815
node dispatch1082 micro raw21 b1815 -> b1815
node dispatch1083 micro raw22 b1815 -> b1815
node dispatch1084 micro raw23 b1815 -> b1815
node session_CopyInto_-5529 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5530 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1555 b1556 -> b1814
node dispatch1085 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5531 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b849
node session_CopyInto_-5532 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b849
node session_VecOp_-5533 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5537 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b868 -> b1811
node session_Fp4GemmMetal_-5538 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1559 b1560 -> b1812
node session_CopyInto_-5539 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5540 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b867 -> b1815
node session_CopyInto_-5541 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b867 -> b1815
node session_CopyInto_-5542 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b867 -> b1815
node session_CopyInto_-5543 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b867 -> b1815
node dispatch1086 micro raw21 b1815 -> b1815
node dispatch1087 micro raw22 b1815 -> b1815
node dispatch1088 micro raw23 b1815 -> b1815
node session_CopyInto_-5544 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5545 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1563 b1564 -> b1814
node dispatch1089 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5546 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b867
node session_CopyInto_-5547 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b867
node session_VecOp_-5548 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5549 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b886 -> b1811
node session_Fp4GemmMetal_-5550 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1567 b1568 -> b1812
node session_CopyInto_-5551 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5552 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b885 -> b1815
node session_CopyInto_-5553 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b885 -> b1815
node session_CopyInto_-5554 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b885 -> b1815
node session_CopyInto_-5555 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b885 -> b1815
node dispatch1090 micro raw21 b1815 -> b1815
node dispatch1091 micro raw22 b1815 -> b1815
node dispatch1092 micro raw23 b1815 -> b1815
node session_CopyInto_-5556 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5557 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1571 b1572 -> b1814
node dispatch1093 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5558 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b885
node session_CopyInto_-5559 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b885
node session_VecOp_-5560 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5561 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b904 -> b1811
node session_Fp4GemmMetal_-5562 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1575 b1576 -> b1812
node session_CopyInto_-5563 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5564 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b903 -> b1815
node session_CopyInto_-5565 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b903 -> b1815
node session_CopyInto_-5566 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b903 -> b1815
node session_CopyInto_-5567 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b903 -> b1815
node dispatch1094 micro raw21 b1815 -> b1815
node dispatch1095 micro raw22 b1815 -> b1815
node dispatch1096 micro raw23 b1815 -> b1815
node session_CopyInto_-5568 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5569 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1579 b1580 -> b1814
node dispatch1097 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5570 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b903
node session_CopyInto_-5571 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b903
node session_VecOp_-5572 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5573 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b922 -> b1811
node session_Fp4GemmMetal_-5574 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1583 b1584 -> b1812
node session_CopyInto_-5575 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5576 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b921 -> b1815
node session_CopyInto_-5577 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b921 -> b1815
node session_CopyInto_-5578 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b921 -> b1815
node session_CopyInto_-5579 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b921 -> b1815
node dispatch1098 micro raw21 b1815 -> b1815
node dispatch1099 micro raw22 b1815 -> b1815
node dispatch1100 micro raw23 b1815 -> b1815
node session_CopyInto_-5580 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5581 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1587 b1588 -> b1814
node dispatch1101 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5582 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b921
node session_CopyInto_-5583 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b921
node session_VecOp_-5584 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5588 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b940 -> b1811
node session_Fp4GemmMetal_-5589 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1591 b1592 -> b1812
node session_CopyInto_-5590 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5591 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b939 -> b1815
node session_CopyInto_-5592 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b939 -> b1815
node session_CopyInto_-5593 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b939 -> b1815
node session_CopyInto_-5594 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b939 -> b1815
node dispatch1102 micro raw21 b1815 -> b1815
node dispatch1103 micro raw22 b1815 -> b1815
node dispatch1104 micro raw23 b1815 -> b1815
node session_CopyInto_-5595 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5596 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1595 b1596 -> b1814
node dispatch1105 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5597 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b939
node session_CopyInto_-5598 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b939
node session_VecOp_-5599 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5600 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b958 -> b1811
node session_Fp4GemmMetal_-5601 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1599 b1600 -> b1812
node session_CopyInto_-5602 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5603 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b957 -> b1815
node session_CopyInto_-5604 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b957 -> b1815
node session_CopyInto_-5605 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b957 -> b1815
node session_CopyInto_-5606 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b957 -> b1815
node dispatch1106 micro raw21 b1815 -> b1815
node dispatch1107 micro raw22 b1815 -> b1815
node dispatch1108 micro raw23 b1815 -> b1815
node session_CopyInto_-5607 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5608 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1603 b1604 -> b1814
node dispatch1109 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5609 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b957
node session_CopyInto_-5610 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b957
node session_VecOp_-5611 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5612 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b976 -> b1811
node session_Fp4GemmMetal_-5613 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1615 b1616 -> b1812
node session_CopyInto_-5614 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5615 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b975 -> b1815
node session_CopyInto_-5616 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b975 -> b1815
node session_CopyInto_-5617 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b975 -> b1815
node session_CopyInto_-5618 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b975 -> b1815
node dispatch1110 micro raw21 b1815 -> b1815
node dispatch1111 micro raw22 b1815 -> b1815
node dispatch1112 micro raw23 b1815 -> b1815
node session_CopyInto_-5619 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5620 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1619 b1620 -> b1814
node dispatch1113 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5621 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b975
node session_CopyInto_-5622 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b975
node session_VecOp_-5623 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5624 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b994 -> b1811
node session_Fp4GemmMetal_-5625 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1623 b1624 -> b1812
node session_CopyInto_-5626 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5627 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b993 -> b1815
node session_CopyInto_-5628 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b993 -> b1815
node session_CopyInto_-5629 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b993 -> b1815
node session_CopyInto_-5630 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b993 -> b1815
node dispatch1114 micro raw21 b1815 -> b1815
node dispatch1115 micro raw22 b1815 -> b1815
node dispatch1116 micro raw23 b1815 -> b1815
node session_CopyInto_-5631 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5632 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1627 b1628 -> b1814
node dispatch1117 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5633 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b993
node session_CopyInto_-5634 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b993
node session_VecOp_-5635 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5639 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1012 -> b1811
node session_Fp4GemmMetal_-5640 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1631 b1632 -> b1812
node session_CopyInto_-5641 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5642 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1011 -> b1815
node session_CopyInto_-5643 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1011 -> b1815
node session_CopyInto_-5644 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1011 -> b1815
node session_CopyInto_-5645 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1011 -> b1815
node dispatch1118 micro raw21 b1815 -> b1815
node dispatch1119 micro raw22 b1815 -> b1815
node dispatch1120 micro raw23 b1815 -> b1815
node session_CopyInto_-5646 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5647 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1635 b1636 -> b1814
node dispatch1121 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5648 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1011
node session_CopyInto_-5649 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1011
node session_VecOp_-5650 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5651 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1030 -> b1811
node session_Fp4GemmMetal_-5652 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1639 b1640 -> b1812
node session_CopyInto_-5653 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5654 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1029 -> b1815
node session_CopyInto_-5655 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1029 -> b1815
node session_CopyInto_-5656 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1029 -> b1815
node session_CopyInto_-5657 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1029 -> b1815
node dispatch1122 micro raw21 b1815 -> b1815
node dispatch1123 micro raw22 b1815 -> b1815
node dispatch1124 micro raw23 b1815 -> b1815
node session_CopyInto_-5658 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5659 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1643 b1644 -> b1814
node dispatch1125 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5660 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1029
node session_CopyInto_-5661 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1029
node session_VecOp_-5662 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5663 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1048 -> b1811
node session_Fp4GemmMetal_-5664 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1647 b1648 -> b1812
node session_CopyInto_-5665 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5666 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1047 -> b1815
node session_CopyInto_-5667 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1047 -> b1815
node session_CopyInto_-5668 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1047 -> b1815
node session_CopyInto_-5669 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1047 -> b1815
node dispatch1126 micro raw21 b1815 -> b1815
node dispatch1127 micro raw22 b1815 -> b1815
node dispatch1128 micro raw23 b1815 -> b1815
node session_CopyInto_-5670 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5671 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1651 b1652 -> b1814
node dispatch1129 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5672 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1047
node session_CopyInto_-5673 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1047
node session_VecOp_-5674 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5675 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1066 -> b1811
node session_Fp4GemmMetal_-5676 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1655 b1656 -> b1812
node session_CopyInto_-5677 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5678 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1065 -> b1815
node session_CopyInto_-5679 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1065 -> b1815
node session_CopyInto_-5680 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1065 -> b1815
node session_CopyInto_-5681 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1065 -> b1815
node dispatch1130 micro raw21 b1815 -> b1815
node dispatch1131 micro raw22 b1815 -> b1815
node dispatch1132 micro raw23 b1815 -> b1815
node session_CopyInto_-5682 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5683 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1659 b1660 -> b1814
node dispatch1133 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5684 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1065
node session_CopyInto_-5685 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1065
node session_MatmulNTAccelerated_-5686 session sessionMatmulNTAccelerated:M=32,K=4096,N=12288,bOff=0 b1811 b1070 -> b1825
node dispatch1134 micro raw25 b1825 b1071 -> b1825
node session_CopyInto_-5688 session sessionCopyInto:srcOff=0,dstOff=0,n=393216 b1825 -> b1822
node session_CopyInto_-5689 session sessionCopyInto:srcOff=16384,dstOff=1318912,n=4096 b1074 -> b1822
node session_CopyInto_-5690 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b1074 -> b1830
node dispatch1135 micro raw5 b1822_5292032_2129920 b1830 -> b1822_5292032_2129920
node dispatch1136 micro raw26 b1822 -> b1822
node dispatch1137 micro raw27 b1822 -> b1822
node session_CopyInto_-5691 session sessionCopyInto:srcOff=655360,dstOff=0,n=131072 b1822 -> b1826
node session_MatmulNTAccelerated_-5692 session sessionMatmulNTAccelerated:M=32,K=4096,N=4096,bOff=0 b1826 b1072 -> b1827
node dispatch1138 micro raw28 b1827 b1073 -> b1827
node session_CopyInto_-5694 session sessionCopyInto:srcOff=1323008,dstOff=0,n=131072 b1822 -> b1830
node dispatch1139 micro raw5 b1074_81920_524288 b1830 -> b1074_81920_524288
node session_CopyInto_-5695 session sessionCopyInto:srcOff=1585152,dstOff=16384,n=4096 b1822 -> b1074
node dispatch1140 micro raw29 b1828 b1814 b1827 -> b1828
node session_MatmulNTAccelerated_-5696 session sessionMatmulNTAccelerated:M=32,K=8192,N=4096,bOff=0 b1828 b1080 -> b1829
node dispatch1141 micro raw28 b1829 b1081 -> b1829
node dispatch1142 micro raw30 b1814 b1829 b1827 -> b1814
node session_VecOp_-5698 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5702 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1101 -> b1811
node session_Fp4GemmMetal_-5703 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1663 b1664 -> b1812
node session_CopyInto_-5704 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5705 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1100 -> b1815
node session_CopyInto_-5706 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1100 -> b1815
node session_CopyInto_-5707 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1100 -> b1815
node session_CopyInto_-5708 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1100 -> b1815
node dispatch1143 micro raw21 b1815 -> b1815
node dispatch1144 micro raw22 b1815 -> b1815
node dispatch1145 micro raw23 b1815 -> b1815
node session_CopyInto_-5709 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5710 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1667 b1668 -> b1814
node dispatch1146 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5711 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1100
node session_CopyInto_-5712 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1100
node session_VecOp_-5713 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5714 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1119 -> b1811
node session_Fp4GemmMetal_-5715 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1671 b1672 -> b1812
node session_CopyInto_-5716 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5717 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1118 -> b1815
node session_CopyInto_-5718 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1118 -> b1815
node session_CopyInto_-5719 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1118 -> b1815
node session_CopyInto_-5720 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1118 -> b1815
node dispatch1147 micro raw21 b1815 -> b1815
node dispatch1148 micro raw22 b1815 -> b1815
node dispatch1149 micro raw23 b1815 -> b1815
node session_CopyInto_-5721 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5722 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1675 b1676 -> b1814
node dispatch1150 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5723 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1118
node session_CopyInto_-5724 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1118
node session_VecOp_-5725 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5726 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1137 -> b1811
node session_Fp4GemmMetal_-5727 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1679 b1680 -> b1812
node session_CopyInto_-5728 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5729 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1136 -> b1815
node session_CopyInto_-5730 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1136 -> b1815
node session_CopyInto_-5731 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1136 -> b1815
node session_CopyInto_-5732 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1136 -> b1815
node dispatch1151 micro raw21 b1815 -> b1815
node dispatch1152 micro raw22 b1815 -> b1815
node dispatch1153 micro raw23 b1815 -> b1815
node session_CopyInto_-5733 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5734 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1683 b1684 -> b1814
node dispatch1154 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5735 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1136
node session_CopyInto_-5736 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1136
node session_VecOp_-5737 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5738 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1155 -> b1811
node session_Fp4GemmMetal_-5739 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1687 b1688 -> b1812
node session_CopyInto_-5740 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5741 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1154 -> b1815
node session_CopyInto_-5742 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1154 -> b1815
node session_CopyInto_-5743 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1154 -> b1815
node session_CopyInto_-5744 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1154 -> b1815
node dispatch1155 micro raw21 b1815 -> b1815
node dispatch1156 micro raw22 b1815 -> b1815
node dispatch1157 micro raw23 b1815 -> b1815
node session_CopyInto_-5745 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5746 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1691 b1692 -> b1814
node dispatch1158 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5747 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1154
node session_CopyInto_-5748 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1154
node session_VecOp_-5749 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5753 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1173 -> b1811
node session_Fp4GemmMetal_-5754 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1703 b1704 -> b1812
node session_CopyInto_-5755 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5756 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1172 -> b1815
node session_CopyInto_-5757 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1172 -> b1815
node session_CopyInto_-5758 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1172 -> b1815
node session_CopyInto_-5759 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1172 -> b1815
node dispatch1159 micro raw21 b1815 -> b1815
node dispatch1160 micro raw22 b1815 -> b1815
node dispatch1161 micro raw23 b1815 -> b1815
node session_CopyInto_-5760 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5761 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1707 b1708 -> b1814
node dispatch1162 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5762 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1172
node session_CopyInto_-5763 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1172
node session_VecOp_-5764 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5765 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1191 -> b1811
node session_Fp4GemmMetal_-5766 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1711 b1712 -> b1812
node session_CopyInto_-5767 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5768 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1190 -> b1815
node session_CopyInto_-5769 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1190 -> b1815
node session_CopyInto_-5770 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1190 -> b1815
node session_CopyInto_-5771 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1190 -> b1815
node dispatch1163 micro raw21 b1815 -> b1815
node dispatch1164 micro raw22 b1815 -> b1815
node dispatch1165 micro raw23 b1815 -> b1815
node session_CopyInto_-5772 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5773 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1715 b1716 -> b1814
node dispatch1166 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5774 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1190
node session_CopyInto_-5775 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1190
node session_VecOp_-5776 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5777 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1209 -> b1811
node session_Fp4GemmMetal_-5778 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1719 b1720 -> b1812
node session_CopyInto_-5779 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5780 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1208 -> b1815
node session_CopyInto_-5781 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1208 -> b1815
node session_CopyInto_-5782 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1208 -> b1815
node session_CopyInto_-5783 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1208 -> b1815
node dispatch1167 micro raw21 b1815 -> b1815
node dispatch1168 micro raw22 b1815 -> b1815
node dispatch1169 micro raw23 b1815 -> b1815
node session_CopyInto_-5784 session sessionCopyInto:srcOff=4196992,dstOff=0,n=262144 b1815 -> b1813
node session_Fp4GemmMetal_-5785 session sessionFp4GemmMetal:M=32,K=8192,N=4096,simdgroup=0 b1813 b1723 b1724 -> b1814
node dispatch1170 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5786 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1208
node session_CopyInto_-5787 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1208
node session_VecOp_-5788 session sessionVecOp:op=add,n=131072 b1810 b1814 -> b1810
node session_RmsnormEnc_-5789 session sessionRmsnormEnc:rows=32,cols=4096,eps=h3727c5ac b1810 b1227 -> b1811
node session_Fp4GemmMetal_-5790 session sessionFp4GemmMetal:M=32,K=4096,N=18560,simdgroup=0 b1811 b1727 b1728 -> b1812
node session_CopyInto_-5791 session sessionCopyInto:srcOff=0,dstOff=0,n=593920 b1812 -> b1815
node session_CopyInto_-5792 session sessionCopyInto:srcOff=45696,dstOff=593920,n=51200 b1226 -> b1815
node session_CopyInto_-5793 session sessionCopyInto:srcOff=37120,dstOff=645120,n=8576 b1226 -> b1815
node session_CopyInto_-5794 session sessionCopyInto:srcOff=127616,dstOff=1313152,n=1048576 b1226 -> b1815
node session_CopyInto_-5795 session sessionCopyInto:srcOff=96896,dstOff=4459136,n=30720 b1226 -> b1815
node dispatch1171 micro raw21 b1815 -> b1815
node dispatch1172 micro raw22 b1815 -> b1815
node dispatch1173 micro raw23 b1815 -> b1815
node dispatch1174 micro raw24 b1816 b1812 -> b1816
node session_CopyInto_-5798 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1816 -> b1226
node session_CopyInto_-5799 session sessionCopyInto:srcOff=2361728,dstOff=127616,n=1048576 b1815 -> b1226
phase chunk4 rows=16
node embedding_4 ext fir.embedding token backbone_embeddings_weight_emb -> b1843
node session_CopyInto_-5844 session sessionCopyInto:srcOff=0,dstOff=0,n=65536 b1843 -> b1834
node session_RmsnormEnc_-5871 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b24 -> b1835
node dispatch1181 micro raw31 b1836 b1247 b1248 b1835 -> b1836
node dispatch1182 micro raw31 b1836_593920_593920 b1247 b1248 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-5872 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-5873 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b23 -> b1839
node session_CopyInto_-5874 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b23 -> b1839
node session_CopyInto_-5875 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b23 -> b1839
node session_CopyInto_-5876 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b23 -> b1839
node dispatch1183 micro raw32 b1839 -> b1839
node dispatch1184 micro raw33 b1839 -> b1839
node dispatch1185 micro raw34 b1839 -> b1839
node session_CopyInto_-5877 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1186 micro raw35 b1838 b1251 b1252 b1837 -> b1838
node dispatch1187 micro raw35 b1838_131072_131072 b1251 b1252 b1837_262144_262144 -> b1838_131072_131072
node dispatch1188 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-5878 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b23
node session_CopyInto_-5879 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b23
node session_VecOp_-5880 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-5881 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b42 -> b1835
node dispatch1189 micro raw31 b1836 b1255 b1256 b1835 -> b1836
node dispatch1190 micro raw31 b1836_593920_593920 b1255 b1256 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-5882 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-5883 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b41 -> b1839
node session_CopyInto_-5884 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b41 -> b1839
node session_CopyInto_-5885 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b41 -> b1839
node session_CopyInto_-5886 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b41 -> b1839
node dispatch1191 micro raw32 b1839 -> b1839
node dispatch1192 micro raw33 b1839 -> b1839
node dispatch1193 micro raw34 b1839 -> b1839
node session_CopyInto_-5887 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1194 micro raw35 b1838 b1259 b1260 b1837 -> b1838
node dispatch1195 micro raw35 b1838_131072_131072 b1259 b1260 b1837_262144_262144 -> b1838_131072_131072
node dispatch1196 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-5888 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b41
node session_CopyInto_-5889 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b41
node session_VecOp_-5890 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-5891 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b60 -> b1835
node dispatch1197 micro raw31 b1836 b1343 b1344 b1835 -> b1836
node dispatch1198 micro raw31 b1836_593920_593920 b1343 b1344 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-5892 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-5893 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b59 -> b1839
node session_CopyInto_-5894 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b59 -> b1839
node session_CopyInto_-5895 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b59 -> b1839
node session_CopyInto_-5896 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b59 -> b1839
node dispatch1199 micro raw32 b1839 -> b1839
node dispatch1200 micro raw33 b1839 -> b1839
node dispatch1201 micro raw34 b1839 -> b1839
node session_CopyInto_-5897 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1202 micro raw35 b1838 b1347 b1348 b1837 -> b1838
node dispatch1203 micro raw35 b1838_131072_131072 b1347 b1348 b1837_262144_262144 -> b1838_131072_131072
node dispatch1204 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-5898 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b59
node session_CopyInto_-5899 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b59
node session_VecOp_-5900 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-5901 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b78 -> b1835
node dispatch1205 micro raw31 b1836 b1431 b1432 b1835 -> b1836
node dispatch1206 micro raw31 b1836_593920_593920 b1431 b1432 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-5902 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-5903 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b77 -> b1839
node session_CopyInto_-5904 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b77 -> b1839
node session_CopyInto_-5905 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b77 -> b1839
node session_CopyInto_-5906 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b77 -> b1839
node dispatch1207 micro raw32 b1839 -> b1839
node dispatch1208 micro raw33 b1839 -> b1839
node dispatch1209 micro raw34 b1839 -> b1839
node session_CopyInto_-5907 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1210 micro raw35 b1838 b1435 b1436 b1837 -> b1838
node dispatch1211 micro raw35 b1838_131072_131072 b1435 b1436 b1837_262144_262144 -> b1838_131072_131072
node dispatch1212 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-5908 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b77
node session_CopyInto_-5909 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b77
node session_VecOp_-5910 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-5914 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b96 -> b1835
node dispatch1213 micro raw31 b1836 b1519 b1520 b1835 -> b1836
node dispatch1214 micro raw31 b1836_593920_593920 b1519 b1520 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-5915 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-5916 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b95 -> b1839
node session_CopyInto_-5917 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b95 -> b1839
node session_CopyInto_-5918 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b95 -> b1839
node session_CopyInto_-5919 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b95 -> b1839
node dispatch1215 micro raw32 b1839 -> b1839
node dispatch1216 micro raw33 b1839 -> b1839
node dispatch1217 micro raw34 b1839 -> b1839
node session_CopyInto_-5920 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1218 micro raw35 b1838 b1523 b1524 b1837 -> b1838
node dispatch1219 micro raw35 b1838_131072_131072 b1523 b1524 b1837_262144_262144 -> b1838_131072_131072
node dispatch1220 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-5921 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b95
node session_CopyInto_-5922 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b95
node session_VecOp_-5923 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-5924 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b114 -> b1835
node dispatch1221 micro raw31 b1836 b1607 b1608 b1835 -> b1836
node dispatch1222 micro raw31 b1836_593920_593920 b1607 b1608 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-5925 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-5926 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b113 -> b1839
node session_CopyInto_-5927 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b113 -> b1839
node session_CopyInto_-5928 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b113 -> b1839
node session_CopyInto_-5929 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b113 -> b1839
node dispatch1223 micro raw32 b1839 -> b1839
node dispatch1224 micro raw33 b1839 -> b1839
node dispatch1225 micro raw34 b1839 -> b1839
node session_CopyInto_-5930 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1226 micro raw35 b1838 b1611 b1612 b1837 -> b1838
node dispatch1227 micro raw35 b1838_131072_131072 b1611 b1612 b1837_262144_262144 -> b1838_131072_131072
node dispatch1228 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-5931 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b113
node session_CopyInto_-5932 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b113
node session_VecOp_-5933 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-5934 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b132 -> b1835
node dispatch1229 micro raw31 b1836 b1695 b1696 b1835 -> b1836
node dispatch1230 micro raw31 b1836_593920_593920 b1695 b1696 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-5935 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-5936 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b131 -> b1839
node session_CopyInto_-5937 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b131 -> b1839
node session_CopyInto_-5938 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b131 -> b1839
node session_CopyInto_-5939 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b131 -> b1839
node dispatch1231 micro raw32 b1839 -> b1839
node dispatch1232 micro raw33 b1839 -> b1839
node dispatch1233 micro raw34 b1839 -> b1839
node session_CopyInto_-5940 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1234 micro raw35 b1838 b1699 b1700 b1837 -> b1838
node dispatch1235 micro raw35 b1838_131072_131072 b1699 b1700 b1837_262144_262144 -> b1838_131072_131072
node dispatch1236 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-5941 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b131
node session_CopyInto_-5942 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b131
node session_VecOp_-5943 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-5944 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b150 -> b1835
node dispatch1237 micro raw31 b1836 b1735 b1736 b1835 -> b1836
node dispatch1238 micro raw31 b1836_593920_593920 b1735 b1736 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-5945 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-5946 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b149 -> b1839
node session_CopyInto_-5947 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b149 -> b1839
node session_CopyInto_-5948 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b149 -> b1839
node session_CopyInto_-5949 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b149 -> b1839
node dispatch1239 micro raw32 b1839 -> b1839
node dispatch1240 micro raw33 b1839 -> b1839
node dispatch1241 micro raw34 b1839 -> b1839
node session_CopyInto_-5950 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1242 micro raw35 b1838 b1739 b1740 b1837 -> b1838
node dispatch1243 micro raw35 b1838_131072_131072 b1739 b1740 b1837_262144_262144 -> b1838_131072_131072
node dispatch1244 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-5951 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b149
node session_CopyInto_-5952 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b149
node session_MatmulNTAccelerated_-5953 session sessionMatmulNTAccelerated:M=16,K=4096,N=12288,bOff=0 b1835 b154 -> b1849
node dispatch1245 micro raw37 b1849 b155 -> b1849
node session_CopyInto_-5955 session sessionCopyInto:srcOff=0,dstOff=0,n=196608 b1849 -> b1846
node session_CopyInto_-5956 session sessionCopyInto:srcOff=16384,dstOff=659456,n=4096 b158 -> b1846
node session_CopyInto_-5957 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b158 -> b1854
node dispatch1246 micro raw5 b1846_2654208_2129920 b1854 -> b1846_2654208_2129920
node dispatch1247 micro raw38 b1846 -> b1846
node dispatch1248 micro raw39 b1846 -> b1846
node session_CopyInto_-5958 session sessionCopyInto:srcOff=327680,dstOff=0,n=65536 b1846 -> b1850
node session_MatmulNTAccelerated_-5959 session sessionMatmulNTAccelerated:M=16,K=4096,N=4096,bOff=0 b1850 b156 -> b1851
node dispatch1249 micro raw40 b1851 b157 -> b1851
node session_CopyInto_-5961 session sessionCopyInto:srcOff=663552,dstOff=0,n=131072 b1846 -> b1854
node dispatch1250 micro raw5 b158_81920_524288 b1854 -> b158_81920_524288
node session_CopyInto_-5962 session sessionCopyInto:srcOff=925696,dstOff=16384,n=4096 b1846 -> b158
node dispatch1251 micro raw41 b1852 b1838 b1851 -> b1852
node session_MatmulNTAccelerated_-5963 session sessionMatmulNTAccelerated:M=16,K=8192,N=4096,bOff=0 b1852 b165 -> b1853
node dispatch1252 micro raw40 b1853 b166 -> b1853
node dispatch1253 micro raw42 b1838 b1853 b1851 -> b1838
node session_VecOp_-5965 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-5969 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b186 -> b1835
node dispatch1254 micro raw31 b1836 b1743 b1744 b1835 -> b1836
node dispatch1255 micro raw31 b1836_593920_593920 b1743 b1744 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-5970 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-5971 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b185 -> b1839
node session_CopyInto_-5972 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b185 -> b1839
node session_CopyInto_-5973 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b185 -> b1839
node session_CopyInto_-5974 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b185 -> b1839
node dispatch1256 micro raw32 b1839 -> b1839
node dispatch1257 micro raw33 b1839 -> b1839
node dispatch1258 micro raw34 b1839 -> b1839
node session_CopyInto_-5975 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1259 micro raw35 b1838 b1747 b1748 b1837 -> b1838
node dispatch1260 micro raw35 b1838_131072_131072 b1747 b1748 b1837_262144_262144 -> b1838_131072_131072
node dispatch1261 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-5976 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b185
node session_CopyInto_-5977 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b185
node session_VecOp_-5978 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-5979 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b204 -> b1835
node dispatch1262 micro raw31 b1836 b1751 b1752 b1835 -> b1836
node dispatch1263 micro raw31 b1836_593920_593920 b1751 b1752 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-5980 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-5981 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b203 -> b1839
node session_CopyInto_-5982 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b203 -> b1839
node session_CopyInto_-5983 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b203 -> b1839
node session_CopyInto_-5984 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b203 -> b1839
node dispatch1264 micro raw32 b1839 -> b1839
node dispatch1265 micro raw33 b1839 -> b1839
node dispatch1266 micro raw34 b1839 -> b1839
node session_CopyInto_-5985 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1267 micro raw35 b1838 b1755 b1756 b1837 -> b1838
node dispatch1268 micro raw35 b1838_131072_131072 b1755 b1756 b1837_262144_262144 -> b1838_131072_131072
node dispatch1269 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-5986 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b203
node session_CopyInto_-5987 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b203
node session_VecOp_-5988 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-5989 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b222 -> b1835
node dispatch1270 micro raw31 b1836 b1263 b1264 b1835 -> b1836
node dispatch1271 micro raw31 b1836_593920_593920 b1263 b1264 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-5990 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-5991 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b221 -> b1839
node session_CopyInto_-5992 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b221 -> b1839
node session_CopyInto_-5993 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b221 -> b1839
node session_CopyInto_-5994 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b221 -> b1839
node dispatch1272 micro raw32 b1839 -> b1839
node dispatch1273 micro raw33 b1839 -> b1839
node dispatch1274 micro raw34 b1839 -> b1839
node session_CopyInto_-5995 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1275 micro raw35 b1838 b1267 b1268 b1837 -> b1838
node dispatch1276 micro raw35 b1838_131072_131072 b1267 b1268 b1837_262144_262144 -> b1838_131072_131072
node dispatch1277 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-5996 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b221
node session_CopyInto_-5997 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b221
node session_VecOp_-5998 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-5999 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b240 -> b1835
node dispatch1278 micro raw31 b1836 b1271 b1272 b1835 -> b1836
node dispatch1279 micro raw31 b1836_593920_593920 b1271 b1272 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6000 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6001 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b239 -> b1839
node session_CopyInto_-6002 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b239 -> b1839
node session_CopyInto_-6003 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b239 -> b1839
node session_CopyInto_-6004 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b239 -> b1839
node dispatch1280 micro raw32 b1839 -> b1839
node dispatch1281 micro raw33 b1839 -> b1839
node dispatch1282 micro raw34 b1839 -> b1839
node session_CopyInto_-6005 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1283 micro raw35 b1838 b1275 b1276 b1837 -> b1838
node dispatch1284 micro raw35 b1838_131072_131072 b1275 b1276 b1837_262144_262144 -> b1838_131072_131072
node dispatch1285 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6006 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b239
node session_CopyInto_-6007 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b239
node session_VecOp_-6008 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6012 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b258 -> b1835
node dispatch1286 micro raw31 b1836 b1279 b1280 b1835 -> b1836
node dispatch1287 micro raw31 b1836_593920_593920 b1279 b1280 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6013 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6014 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b257 -> b1839
node session_CopyInto_-6015 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b257 -> b1839
node session_CopyInto_-6016 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b257 -> b1839
node session_CopyInto_-6017 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b257 -> b1839
node dispatch1288 micro raw32 b1839 -> b1839
node dispatch1289 micro raw33 b1839 -> b1839
node dispatch1290 micro raw34 b1839 -> b1839
node session_CopyInto_-6018 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1291 micro raw35 b1838 b1283 b1284 b1837 -> b1838
node dispatch1292 micro raw35 b1838_131072_131072 b1283 b1284 b1837_262144_262144 -> b1838_131072_131072
node dispatch1293 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6019 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b257
node session_CopyInto_-6020 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b257
node session_VecOp_-6021 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6022 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b276 -> b1835
node dispatch1294 micro raw31 b1836 b1287 b1288 b1835 -> b1836
node dispatch1295 micro raw31 b1836_593920_593920 b1287 b1288 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6023 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6024 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b275 -> b1839
node session_CopyInto_-6025 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b275 -> b1839
node session_CopyInto_-6026 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b275 -> b1839
node session_CopyInto_-6027 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b275 -> b1839
node dispatch1296 micro raw32 b1839 -> b1839
node dispatch1297 micro raw33 b1839 -> b1839
node dispatch1298 micro raw34 b1839 -> b1839
node session_CopyInto_-6028 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1299 micro raw35 b1838 b1291 b1292 b1837 -> b1838
node dispatch1300 micro raw35 b1838_131072_131072 b1291 b1292 b1837_262144_262144 -> b1838_131072_131072
node dispatch1301 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6029 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b275
node session_CopyInto_-6030 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b275
node session_VecOp_-6031 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6032 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b294 -> b1835
node dispatch1302 micro raw31 b1836 b1295 b1296 b1835 -> b1836
node dispatch1303 micro raw31 b1836_593920_593920 b1295 b1296 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6033 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6034 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b293 -> b1839
node session_CopyInto_-6035 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b293 -> b1839
node session_CopyInto_-6036 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b293 -> b1839
node session_CopyInto_-6037 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b293 -> b1839
node dispatch1304 micro raw32 b1839 -> b1839
node dispatch1305 micro raw33 b1839 -> b1839
node dispatch1306 micro raw34 b1839 -> b1839
node session_CopyInto_-6038 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1307 micro raw35 b1838 b1299 b1300 b1837 -> b1838
node dispatch1308 micro raw35 b1838_131072_131072 b1299 b1300 b1837_262144_262144 -> b1838_131072_131072
node dispatch1309 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6039 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b293
node session_CopyInto_-6040 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b293
node session_VecOp_-6041 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6042 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b312 -> b1835
node dispatch1310 micro raw31 b1836 b1303 b1304 b1835 -> b1836
node dispatch1311 micro raw31 b1836_593920_593920 b1303 b1304 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6043 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6044 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b311 -> b1839
node session_CopyInto_-6045 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b311 -> b1839
node session_CopyInto_-6046 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b311 -> b1839
node session_CopyInto_-6047 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b311 -> b1839
node dispatch1312 micro raw32 b1839 -> b1839
node dispatch1313 micro raw33 b1839 -> b1839
node dispatch1314 micro raw34 b1839 -> b1839
node session_CopyInto_-6048 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1315 micro raw35 b1838 b1307 b1308 b1837 -> b1838
node dispatch1316 micro raw35 b1838_131072_131072 b1307 b1308 b1837_262144_262144 -> b1838_131072_131072
node dispatch1317 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6049 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b311
node session_CopyInto_-6050 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b311
node session_VecOp_-6051 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6055 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b330 -> b1835
node dispatch1318 micro raw31 b1836 b1311 b1312 b1835 -> b1836
node dispatch1319 micro raw31 b1836_593920_593920 b1311 b1312 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6056 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6057 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b329 -> b1839
node session_CopyInto_-6058 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b329 -> b1839
node session_CopyInto_-6059 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b329 -> b1839
node session_CopyInto_-6060 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b329 -> b1839
node dispatch1320 micro raw32 b1839 -> b1839
node dispatch1321 micro raw33 b1839 -> b1839
node dispatch1322 micro raw34 b1839 -> b1839
node session_CopyInto_-6061 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1323 micro raw35 b1838 b1315 b1316 b1837 -> b1838
node dispatch1324 micro raw35 b1838_131072_131072 b1315 b1316 b1837_262144_262144 -> b1838_131072_131072
node dispatch1325 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6062 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b329
node session_CopyInto_-6063 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b329
node session_VecOp_-6064 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6065 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b348 -> b1835
node dispatch1326 micro raw31 b1836 b1319 b1320 b1835 -> b1836
node dispatch1327 micro raw31 b1836_593920_593920 b1319 b1320 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6066 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6067 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b347 -> b1839
node session_CopyInto_-6068 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b347 -> b1839
node session_CopyInto_-6069 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b347 -> b1839
node session_CopyInto_-6070 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b347 -> b1839
node dispatch1328 micro raw32 b1839 -> b1839
node dispatch1329 micro raw33 b1839 -> b1839
node dispatch1330 micro raw34 b1839 -> b1839
node session_CopyInto_-6071 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1331 micro raw35 b1838 b1323 b1324 b1837 -> b1838
node dispatch1332 micro raw35 b1838_131072_131072 b1323 b1324 b1837_262144_262144 -> b1838_131072_131072
node dispatch1333 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6072 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b347
node session_CopyInto_-6073 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b347
node session_VecOp_-6074 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6075 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b366 -> b1835
node dispatch1334 micro raw31 b1836 b1327 b1328 b1835 -> b1836
node dispatch1335 micro raw31 b1836_593920_593920 b1327 b1328 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6076 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6077 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b365 -> b1839
node session_CopyInto_-6078 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b365 -> b1839
node session_CopyInto_-6079 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b365 -> b1839
node session_CopyInto_-6080 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b365 -> b1839
node dispatch1336 micro raw32 b1839 -> b1839
node dispatch1337 micro raw33 b1839 -> b1839
node dispatch1338 micro raw34 b1839 -> b1839
node session_CopyInto_-6081 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1339 micro raw35 b1838 b1331 b1332 b1837 -> b1838
node dispatch1340 micro raw35 b1838_131072_131072 b1331 b1332 b1837_262144_262144 -> b1838_131072_131072
node dispatch1341 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6082 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b365
node session_CopyInto_-6083 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b365
node session_VecOp_-6084 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6085 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b384 -> b1835
node dispatch1342 micro raw31 b1836 b1335 b1336 b1835 -> b1836
node dispatch1343 micro raw31 b1836_593920_593920 b1335 b1336 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6086 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6087 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b383 -> b1839
node session_CopyInto_-6088 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b383 -> b1839
node session_CopyInto_-6089 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b383 -> b1839
node session_CopyInto_-6090 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b383 -> b1839
node dispatch1344 micro raw32 b1839 -> b1839
node dispatch1345 micro raw33 b1839 -> b1839
node dispatch1346 micro raw34 b1839 -> b1839
node session_CopyInto_-6091 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1347 micro raw35 b1838 b1339 b1340 b1837 -> b1838
node dispatch1348 micro raw35 b1838_131072_131072 b1339 b1340 b1837_262144_262144 -> b1838_131072_131072
node dispatch1349 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6092 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b383
node session_CopyInto_-6093 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b383
node session_VecOp_-6094 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6098 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b402 -> b1835
node dispatch1350 micro raw31 b1836 b1351 b1352 b1835 -> b1836
node dispatch1351 micro raw31 b1836_593920_593920 b1351 b1352 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6099 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6100 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b401 -> b1839
node session_CopyInto_-6101 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b401 -> b1839
node session_CopyInto_-6102 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b401 -> b1839
node session_CopyInto_-6103 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b401 -> b1839
node dispatch1352 micro raw32 b1839 -> b1839
node dispatch1353 micro raw33 b1839 -> b1839
node dispatch1354 micro raw34 b1839 -> b1839
node session_CopyInto_-6104 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1355 micro raw35 b1838 b1355 b1356 b1837 -> b1838
node dispatch1356 micro raw35 b1838_131072_131072 b1355 b1356 b1837_262144_262144 -> b1838_131072_131072
node dispatch1357 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6105 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b401
node session_CopyInto_-6106 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b401
node session_VecOp_-6107 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6108 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b420 -> b1835
node dispatch1358 micro raw31 b1836 b1359 b1360 b1835 -> b1836
node dispatch1359 micro raw31 b1836_593920_593920 b1359 b1360 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6109 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6110 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b419 -> b1839
node session_CopyInto_-6111 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b419 -> b1839
node session_CopyInto_-6112 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b419 -> b1839
node session_CopyInto_-6113 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b419 -> b1839
node dispatch1360 micro raw32 b1839 -> b1839
node dispatch1361 micro raw33 b1839 -> b1839
node dispatch1362 micro raw34 b1839 -> b1839
node session_CopyInto_-6114 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1363 micro raw35 b1838 b1363 b1364 b1837 -> b1838
node dispatch1364 micro raw35 b1838_131072_131072 b1363 b1364 b1837_262144_262144 -> b1838_131072_131072
node dispatch1365 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6115 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b419
node session_CopyInto_-6116 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b419
node session_VecOp_-6117 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6118 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b438 -> b1835
node dispatch1366 micro raw31 b1836 b1367 b1368 b1835 -> b1836
node dispatch1367 micro raw31 b1836_593920_593920 b1367 b1368 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6119 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6120 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b437 -> b1839
node session_CopyInto_-6121 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b437 -> b1839
node session_CopyInto_-6122 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b437 -> b1839
node session_CopyInto_-6123 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b437 -> b1839
node dispatch1368 micro raw32 b1839 -> b1839
node dispatch1369 micro raw33 b1839 -> b1839
node dispatch1370 micro raw34 b1839 -> b1839
node session_CopyInto_-6124 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1371 micro raw35 b1838 b1371 b1372 b1837 -> b1838
node dispatch1372 micro raw35 b1838_131072_131072 b1371 b1372 b1837_262144_262144 -> b1838_131072_131072
node dispatch1373 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6125 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b437
node session_CopyInto_-6126 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b437
node session_VecOp_-6127 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6128 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b456 -> b1835
node dispatch1374 micro raw31 b1836 b1375 b1376 b1835 -> b1836
node dispatch1375 micro raw31 b1836_593920_593920 b1375 b1376 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6129 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6130 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b455 -> b1839
node session_CopyInto_-6131 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b455 -> b1839
node session_CopyInto_-6132 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b455 -> b1839
node session_CopyInto_-6133 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b455 -> b1839
node dispatch1376 micro raw32 b1839 -> b1839
node dispatch1377 micro raw33 b1839 -> b1839
node dispatch1378 micro raw34 b1839 -> b1839
node session_CopyInto_-6134 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1379 micro raw35 b1838 b1379 b1380 b1837 -> b1838
node dispatch1380 micro raw35 b1838_131072_131072 b1379 b1380 b1837_262144_262144 -> b1838_131072_131072
node dispatch1381 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6135 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b455
node session_CopyInto_-6136 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b455
node session_MatmulNTAccelerated_-6137 session sessionMatmulNTAccelerated:M=16,K=4096,N=12288,bOff=0 b1835 b460 -> b1849
node dispatch1382 micro raw37 b1849 b461 -> b1849
node session_CopyInto_-6139 session sessionCopyInto:srcOff=0,dstOff=0,n=196608 b1849 -> b1846
node session_CopyInto_-6140 session sessionCopyInto:srcOff=16384,dstOff=659456,n=4096 b464 -> b1846
node session_CopyInto_-6141 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b464 -> b1854
node dispatch1383 micro raw5 b1846_2654208_2129920 b1854 -> b1846_2654208_2129920
node dispatch1384 micro raw38 b1846 -> b1846
node dispatch1385 micro raw39 b1846 -> b1846
node session_CopyInto_-6142 session sessionCopyInto:srcOff=327680,dstOff=0,n=65536 b1846 -> b1850
node session_MatmulNTAccelerated_-6143 session sessionMatmulNTAccelerated:M=16,K=4096,N=4096,bOff=0 b1850 b462 -> b1851
node dispatch1386 micro raw40 b1851 b463 -> b1851
node session_CopyInto_-6145 session sessionCopyInto:srcOff=663552,dstOff=0,n=131072 b1846 -> b1854
node dispatch1387 micro raw5 b464_81920_524288 b1854 -> b464_81920_524288
node session_CopyInto_-6146 session sessionCopyInto:srcOff=925696,dstOff=16384,n=4096 b1846 -> b464
node dispatch1388 micro raw41 b1852 b1838 b1851 -> b1852
node session_MatmulNTAccelerated_-6147 session sessionMatmulNTAccelerated:M=16,K=8192,N=4096,bOff=0 b1852 b470 -> b1853
node dispatch1389 micro raw40 b1853 b471 -> b1853
node dispatch1390 micro raw42 b1838 b1853 b1851 -> b1838
node session_VecOp_-6149 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6153 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b491 -> b1835
node dispatch1391 micro raw31 b1836 b1383 b1384 b1835 -> b1836
node dispatch1392 micro raw31 b1836_593920_593920 b1383 b1384 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6154 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6155 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b490 -> b1839
node session_CopyInto_-6156 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b490 -> b1839
node session_CopyInto_-6157 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b490 -> b1839
node session_CopyInto_-6158 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b490 -> b1839
node dispatch1393 micro raw32 b1839 -> b1839
node dispatch1394 micro raw33 b1839 -> b1839
node dispatch1395 micro raw34 b1839 -> b1839
node session_CopyInto_-6159 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1396 micro raw35 b1838 b1387 b1388 b1837 -> b1838
node dispatch1397 micro raw35 b1838_131072_131072 b1387 b1388 b1837_262144_262144 -> b1838_131072_131072
node dispatch1398 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6160 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b490
node session_CopyInto_-6161 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b490
node session_VecOp_-6162 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6163 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b509 -> b1835
node dispatch1399 micro raw31 b1836 b1391 b1392 b1835 -> b1836
node dispatch1400 micro raw31 b1836_593920_593920 b1391 b1392 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6164 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6165 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b508 -> b1839
node session_CopyInto_-6166 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b508 -> b1839
node session_CopyInto_-6167 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b508 -> b1839
node session_CopyInto_-6168 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b508 -> b1839
node dispatch1401 micro raw32 b1839 -> b1839
node dispatch1402 micro raw33 b1839 -> b1839
node dispatch1403 micro raw34 b1839 -> b1839
node session_CopyInto_-6169 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1404 micro raw35 b1838 b1395 b1396 b1837 -> b1838
node dispatch1405 micro raw35 b1838_131072_131072 b1395 b1396 b1837_262144_262144 -> b1838_131072_131072
node dispatch1406 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6170 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b508
node session_CopyInto_-6171 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b508
node session_VecOp_-6172 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6173 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b527 -> b1835
node dispatch1407 micro raw31 b1836 b1399 b1400 b1835 -> b1836
node dispatch1408 micro raw31 b1836_593920_593920 b1399 b1400 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6174 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6175 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b526 -> b1839
node session_CopyInto_-6176 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b526 -> b1839
node session_CopyInto_-6177 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b526 -> b1839
node session_CopyInto_-6178 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b526 -> b1839
node dispatch1409 micro raw32 b1839 -> b1839
node dispatch1410 micro raw33 b1839 -> b1839
node dispatch1411 micro raw34 b1839 -> b1839
node session_CopyInto_-6179 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1412 micro raw35 b1838 b1403 b1404 b1837 -> b1838
node dispatch1413 micro raw35 b1838_131072_131072 b1403 b1404 b1837_262144_262144 -> b1838_131072_131072
node dispatch1414 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6180 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b526
node session_CopyInto_-6181 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b526
node session_VecOp_-6182 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6183 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b545 -> b1835
node dispatch1415 micro raw31 b1836 b1407 b1408 b1835 -> b1836
node dispatch1416 micro raw31 b1836_593920_593920 b1407 b1408 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6184 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6185 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b544 -> b1839
node session_CopyInto_-6186 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b544 -> b1839
node session_CopyInto_-6187 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b544 -> b1839
node session_CopyInto_-6188 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b544 -> b1839
node dispatch1417 micro raw32 b1839 -> b1839
node dispatch1418 micro raw33 b1839 -> b1839
node dispatch1419 micro raw34 b1839 -> b1839
node session_CopyInto_-6189 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1420 micro raw35 b1838 b1411 b1412 b1837 -> b1838
node dispatch1421 micro raw35 b1838_131072_131072 b1411 b1412 b1837_262144_262144 -> b1838_131072_131072
node dispatch1422 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6190 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b544
node session_CopyInto_-6191 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b544
node session_VecOp_-6192 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6196 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b563 -> b1835
node dispatch1423 micro raw31 b1836 b1415 b1416 b1835 -> b1836
node dispatch1424 micro raw31 b1836_593920_593920 b1415 b1416 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6197 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6198 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b562 -> b1839
node session_CopyInto_-6199 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b562 -> b1839
node session_CopyInto_-6200 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b562 -> b1839
node session_CopyInto_-6201 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b562 -> b1839
node dispatch1425 micro raw32 b1839 -> b1839
node dispatch1426 micro raw33 b1839 -> b1839
node dispatch1427 micro raw34 b1839 -> b1839
node session_CopyInto_-6202 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1428 micro raw35 b1838 b1419 b1420 b1837 -> b1838
node dispatch1429 micro raw35 b1838_131072_131072 b1419 b1420 b1837_262144_262144 -> b1838_131072_131072
node dispatch1430 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6203 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b562
node session_CopyInto_-6204 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b562
node session_VecOp_-6205 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6206 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b581 -> b1835
node dispatch1431 micro raw31 b1836 b1423 b1424 b1835 -> b1836
node dispatch1432 micro raw31 b1836_593920_593920 b1423 b1424 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6207 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6208 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b580 -> b1839
node session_CopyInto_-6209 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b580 -> b1839
node session_CopyInto_-6210 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b580 -> b1839
node session_CopyInto_-6211 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b580 -> b1839
node dispatch1433 micro raw32 b1839 -> b1839
node dispatch1434 micro raw33 b1839 -> b1839
node dispatch1435 micro raw34 b1839 -> b1839
node session_CopyInto_-6212 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1436 micro raw35 b1838 b1427 b1428 b1837 -> b1838
node dispatch1437 micro raw35 b1838_131072_131072 b1427 b1428 b1837_262144_262144 -> b1838_131072_131072
node dispatch1438 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6213 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b580
node session_CopyInto_-6214 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b580
node session_VecOp_-6215 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6216 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b599 -> b1835
node dispatch1439 micro raw31 b1836 b1439 b1440 b1835 -> b1836
node dispatch1440 micro raw31 b1836_593920_593920 b1439 b1440 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6217 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6218 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b598 -> b1839
node session_CopyInto_-6219 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b598 -> b1839
node session_CopyInto_-6220 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b598 -> b1839
node session_CopyInto_-6221 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b598 -> b1839
node dispatch1441 micro raw32 b1839 -> b1839
node dispatch1442 micro raw33 b1839 -> b1839
node dispatch1443 micro raw34 b1839 -> b1839
node session_CopyInto_-6222 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1444 micro raw35 b1838 b1443 b1444 b1837 -> b1838
node dispatch1445 micro raw35 b1838_131072_131072 b1443 b1444 b1837_262144_262144 -> b1838_131072_131072
node dispatch1446 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6223 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b598
node session_CopyInto_-6224 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b598
node session_VecOp_-6225 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6226 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b617 -> b1835
node dispatch1447 micro raw31 b1836 b1447 b1448 b1835 -> b1836
node dispatch1448 micro raw31 b1836_593920_593920 b1447 b1448 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6227 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6228 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b616 -> b1839
node session_CopyInto_-6229 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b616 -> b1839
node session_CopyInto_-6230 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b616 -> b1839
node session_CopyInto_-6231 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b616 -> b1839
node dispatch1449 micro raw32 b1839 -> b1839
node dispatch1450 micro raw33 b1839 -> b1839
node dispatch1451 micro raw34 b1839 -> b1839
node session_CopyInto_-6232 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1452 micro raw35 b1838 b1451 b1452 b1837 -> b1838
node dispatch1453 micro raw35 b1838_131072_131072 b1451 b1452 b1837_262144_262144 -> b1838_131072_131072
node dispatch1454 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6233 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b616
node session_CopyInto_-6234 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b616
node session_VecOp_-6235 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6239 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b635 -> b1835
node dispatch1455 micro raw31 b1836 b1455 b1456 b1835 -> b1836
node dispatch1456 micro raw31 b1836_593920_593920 b1455 b1456 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6240 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6241 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b634 -> b1839
node session_CopyInto_-6242 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b634 -> b1839
node session_CopyInto_-6243 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b634 -> b1839
node session_CopyInto_-6244 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b634 -> b1839
node dispatch1457 micro raw32 b1839 -> b1839
node dispatch1458 micro raw33 b1839 -> b1839
node dispatch1459 micro raw34 b1839 -> b1839
node session_CopyInto_-6245 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1460 micro raw35 b1838 b1459 b1460 b1837 -> b1838
node dispatch1461 micro raw35 b1838_131072_131072 b1459 b1460 b1837_262144_262144 -> b1838_131072_131072
node dispatch1462 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6246 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b634
node session_CopyInto_-6247 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b634
node session_VecOp_-6248 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6249 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b653 -> b1835
node dispatch1463 micro raw31 b1836 b1463 b1464 b1835 -> b1836
node dispatch1464 micro raw31 b1836_593920_593920 b1463 b1464 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6250 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6251 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b652 -> b1839
node session_CopyInto_-6252 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b652 -> b1839
node session_CopyInto_-6253 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b652 -> b1839
node session_CopyInto_-6254 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b652 -> b1839
node dispatch1465 micro raw32 b1839 -> b1839
node dispatch1466 micro raw33 b1839 -> b1839
node dispatch1467 micro raw34 b1839 -> b1839
node session_CopyInto_-6255 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1468 micro raw35 b1838 b1467 b1468 b1837 -> b1838
node dispatch1469 micro raw35 b1838_131072_131072 b1467 b1468 b1837_262144_262144 -> b1838_131072_131072
node dispatch1470 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6256 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b652
node session_CopyInto_-6257 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b652
node session_VecOp_-6258 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6259 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b671 -> b1835
node dispatch1471 micro raw31 b1836 b1471 b1472 b1835 -> b1836
node dispatch1472 micro raw31 b1836_593920_593920 b1471 b1472 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6260 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6261 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b670 -> b1839
node session_CopyInto_-6262 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b670 -> b1839
node session_CopyInto_-6263 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b670 -> b1839
node session_CopyInto_-6264 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b670 -> b1839
node dispatch1473 micro raw32 b1839 -> b1839
node dispatch1474 micro raw33 b1839 -> b1839
node dispatch1475 micro raw34 b1839 -> b1839
node session_CopyInto_-6265 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1476 micro raw35 b1838 b1475 b1476 b1837 -> b1838
node dispatch1477 micro raw35 b1838_131072_131072 b1475 b1476 b1837_262144_262144 -> b1838_131072_131072
node dispatch1478 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6266 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b670
node session_CopyInto_-6267 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b670
node session_VecOp_-6268 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6269 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b689 -> b1835
node dispatch1479 micro raw31 b1836 b1479 b1480 b1835 -> b1836
node dispatch1480 micro raw31 b1836_593920_593920 b1479 b1480 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6270 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6271 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b688 -> b1839
node session_CopyInto_-6272 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b688 -> b1839
node session_CopyInto_-6273 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b688 -> b1839
node session_CopyInto_-6274 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b688 -> b1839
node dispatch1481 micro raw32 b1839 -> b1839
node dispatch1482 micro raw33 b1839 -> b1839
node dispatch1483 micro raw34 b1839 -> b1839
node session_CopyInto_-6275 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1484 micro raw35 b1838 b1483 b1484 b1837 -> b1838
node dispatch1485 micro raw35 b1838_131072_131072 b1483 b1484 b1837_262144_262144 -> b1838_131072_131072
node dispatch1486 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6276 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b688
node session_CopyInto_-6277 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b688
node session_VecOp_-6278 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6282 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b707 -> b1835
node dispatch1487 micro raw31 b1836 b1487 b1488 b1835 -> b1836
node dispatch1488 micro raw31 b1836_593920_593920 b1487 b1488 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6283 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6284 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b706 -> b1839
node session_CopyInto_-6285 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b706 -> b1839
node session_CopyInto_-6286 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b706 -> b1839
node session_CopyInto_-6287 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b706 -> b1839
node dispatch1489 micro raw32 b1839 -> b1839
node dispatch1490 micro raw33 b1839 -> b1839
node dispatch1491 micro raw34 b1839 -> b1839
node session_CopyInto_-6288 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1492 micro raw35 b1838 b1491 b1492 b1837 -> b1838
node dispatch1493 micro raw35 b1838_131072_131072 b1491 b1492 b1837_262144_262144 -> b1838_131072_131072
node dispatch1494 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6289 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b706
node session_CopyInto_-6290 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b706
node session_VecOp_-6291 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6292 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b725 -> b1835
node dispatch1495 micro raw31 b1836 b1495 b1496 b1835 -> b1836
node dispatch1496 micro raw31 b1836_593920_593920 b1495 b1496 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6293 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6294 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b724 -> b1839
node session_CopyInto_-6295 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b724 -> b1839
node session_CopyInto_-6296 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b724 -> b1839
node session_CopyInto_-6297 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b724 -> b1839
node dispatch1497 micro raw32 b1839 -> b1839
node dispatch1498 micro raw33 b1839 -> b1839
node dispatch1499 micro raw34 b1839 -> b1839
node session_CopyInto_-6298 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1500 micro raw35 b1838 b1499 b1500 b1837 -> b1838
node dispatch1501 micro raw35 b1838_131072_131072 b1499 b1500 b1837_262144_262144 -> b1838_131072_131072
node dispatch1502 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6299 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b724
node session_CopyInto_-6300 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b724
node session_VecOp_-6301 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6302 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b743 -> b1835
node dispatch1503 micro raw31 b1836 b1503 b1504 b1835 -> b1836
node dispatch1504 micro raw31 b1836_593920_593920 b1503 b1504 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6303 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6304 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b742 -> b1839
node session_CopyInto_-6305 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b742 -> b1839
node session_CopyInto_-6306 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b742 -> b1839
node session_CopyInto_-6307 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b742 -> b1839
node dispatch1505 micro raw32 b1839 -> b1839
node dispatch1506 micro raw33 b1839 -> b1839
node dispatch1507 micro raw34 b1839 -> b1839
node session_CopyInto_-6308 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1508 micro raw35 b1838 b1507 b1508 b1837 -> b1838
node dispatch1509 micro raw35 b1838_131072_131072 b1507 b1508 b1837_262144_262144 -> b1838_131072_131072
node dispatch1510 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6309 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b742
node session_CopyInto_-6310 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b742
node session_VecOp_-6311 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6312 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b761 -> b1835
node dispatch1511 micro raw31 b1836 b1511 b1512 b1835 -> b1836
node dispatch1512 micro raw31 b1836_593920_593920 b1511 b1512 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6313 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6314 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b760 -> b1839
node session_CopyInto_-6315 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b760 -> b1839
node session_CopyInto_-6316 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b760 -> b1839
node session_CopyInto_-6317 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b760 -> b1839
node dispatch1513 micro raw32 b1839 -> b1839
node dispatch1514 micro raw33 b1839 -> b1839
node dispatch1515 micro raw34 b1839 -> b1839
node session_CopyInto_-6318 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1516 micro raw35 b1838 b1515 b1516 b1837 -> b1838
node dispatch1517 micro raw35 b1838_131072_131072 b1515 b1516 b1837_262144_262144 -> b1838_131072_131072
node dispatch1518 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6319 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b760
node session_CopyInto_-6320 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b760
node session_MatmulNTAccelerated_-6321 session sessionMatmulNTAccelerated:M=16,K=4096,N=12288,bOff=0 b1835 b765 -> b1849
node dispatch1519 micro raw37 b1849 b766 -> b1849
node session_CopyInto_-6323 session sessionCopyInto:srcOff=0,dstOff=0,n=196608 b1849 -> b1846
node session_CopyInto_-6324 session sessionCopyInto:srcOff=16384,dstOff=659456,n=4096 b769 -> b1846
node session_CopyInto_-6325 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b769 -> b1854
node dispatch1520 micro raw5 b1846_2654208_2129920 b1854 -> b1846_2654208_2129920
node dispatch1521 micro raw38 b1846 -> b1846
node dispatch1522 micro raw39 b1846 -> b1846
node session_CopyInto_-6326 session sessionCopyInto:srcOff=327680,dstOff=0,n=65536 b1846 -> b1850
node session_MatmulNTAccelerated_-6327 session sessionMatmulNTAccelerated:M=16,K=4096,N=4096,bOff=0 b1850 b767 -> b1851
node dispatch1523 micro raw40 b1851 b768 -> b1851
node session_CopyInto_-6329 session sessionCopyInto:srcOff=663552,dstOff=0,n=131072 b1846 -> b1854
node dispatch1524 micro raw5 b769_81920_524288 b1854 -> b769_81920_524288
node session_CopyInto_-6330 session sessionCopyInto:srcOff=925696,dstOff=16384,n=4096 b1846 -> b769
node dispatch1525 micro raw41 b1852 b1838 b1851 -> b1852
node session_MatmulNTAccelerated_-6331 session sessionMatmulNTAccelerated:M=16,K=8192,N=4096,bOff=0 b1852 b775 -> b1853
node dispatch1526 micro raw40 b1853 b776 -> b1853
node dispatch1527 micro raw42 b1838 b1853 b1851 -> b1838
node session_VecOp_-6333 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6337 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b796 -> b1835
node dispatch1528 micro raw31 b1836 b1527 b1528 b1835 -> b1836
node dispatch1529 micro raw31 b1836_593920_593920 b1527 b1528 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6338 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6339 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b795 -> b1839
node session_CopyInto_-6340 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b795 -> b1839
node session_CopyInto_-6341 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b795 -> b1839
node session_CopyInto_-6342 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b795 -> b1839
node dispatch1530 micro raw32 b1839 -> b1839
node dispatch1531 micro raw33 b1839 -> b1839
node dispatch1532 micro raw34 b1839 -> b1839
node session_CopyInto_-6343 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1533 micro raw35 b1838 b1531 b1532 b1837 -> b1838
node dispatch1534 micro raw35 b1838_131072_131072 b1531 b1532 b1837_262144_262144 -> b1838_131072_131072
node dispatch1535 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6344 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b795
node session_CopyInto_-6345 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b795
node session_VecOp_-6346 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6347 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b814 -> b1835
node dispatch1536 micro raw31 b1836 b1535 b1536 b1835 -> b1836
node dispatch1537 micro raw31 b1836_593920_593920 b1535 b1536 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6348 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6349 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b813 -> b1839
node session_CopyInto_-6350 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b813 -> b1839
node session_CopyInto_-6351 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b813 -> b1839
node session_CopyInto_-6352 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b813 -> b1839
node dispatch1538 micro raw32 b1839 -> b1839
node dispatch1539 micro raw33 b1839 -> b1839
node dispatch1540 micro raw34 b1839 -> b1839
node session_CopyInto_-6353 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1541 micro raw35 b1838 b1539 b1540 b1837 -> b1838
node dispatch1542 micro raw35 b1838_131072_131072 b1539 b1540 b1837_262144_262144 -> b1838_131072_131072
node dispatch1543 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6354 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b813
node session_CopyInto_-6355 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b813
node session_VecOp_-6356 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6357 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b832 -> b1835
node dispatch1544 micro raw31 b1836 b1543 b1544 b1835 -> b1836
node dispatch1545 micro raw31 b1836_593920_593920 b1543 b1544 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6358 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6359 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b831 -> b1839
node session_CopyInto_-6360 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b831 -> b1839
node session_CopyInto_-6361 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b831 -> b1839
node session_CopyInto_-6362 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b831 -> b1839
node dispatch1546 micro raw32 b1839 -> b1839
node dispatch1547 micro raw33 b1839 -> b1839
node dispatch1548 micro raw34 b1839 -> b1839
node session_CopyInto_-6363 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1549 micro raw35 b1838 b1547 b1548 b1837 -> b1838
node dispatch1550 micro raw35 b1838_131072_131072 b1547 b1548 b1837_262144_262144 -> b1838_131072_131072
node dispatch1551 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6364 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b831
node session_CopyInto_-6365 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b831
node session_VecOp_-6366 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6367 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b850 -> b1835
node dispatch1552 micro raw31 b1836 b1551 b1552 b1835 -> b1836
node dispatch1553 micro raw31 b1836_593920_593920 b1551 b1552 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6368 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6369 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b849 -> b1839
node session_CopyInto_-6370 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b849 -> b1839
node session_CopyInto_-6371 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b849 -> b1839
node session_CopyInto_-6372 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b849 -> b1839
node dispatch1554 micro raw32 b1839 -> b1839
node dispatch1555 micro raw33 b1839 -> b1839
node dispatch1556 micro raw34 b1839 -> b1839
node session_CopyInto_-6373 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1557 micro raw35 b1838 b1555 b1556 b1837 -> b1838
node dispatch1558 micro raw35 b1838_131072_131072 b1555 b1556 b1837_262144_262144 -> b1838_131072_131072
node dispatch1559 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6374 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b849
node session_CopyInto_-6375 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b849
node session_VecOp_-6376 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6380 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b868 -> b1835
node dispatch1560 micro raw31 b1836 b1559 b1560 b1835 -> b1836
node dispatch1561 micro raw31 b1836_593920_593920 b1559 b1560 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6381 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6382 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b867 -> b1839
node session_CopyInto_-6383 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b867 -> b1839
node session_CopyInto_-6384 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b867 -> b1839
node session_CopyInto_-6385 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b867 -> b1839
node dispatch1562 micro raw32 b1839 -> b1839
node dispatch1563 micro raw33 b1839 -> b1839
node dispatch1564 micro raw34 b1839 -> b1839
node session_CopyInto_-6386 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1565 micro raw35 b1838 b1563 b1564 b1837 -> b1838
node dispatch1566 micro raw35 b1838_131072_131072 b1563 b1564 b1837_262144_262144 -> b1838_131072_131072
node dispatch1567 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6387 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b867
node session_CopyInto_-6388 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b867
node session_VecOp_-6389 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6390 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b886 -> b1835
node dispatch1568 micro raw31 b1836 b1567 b1568 b1835 -> b1836
node dispatch1569 micro raw31 b1836_593920_593920 b1567 b1568 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6391 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6392 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b885 -> b1839
node session_CopyInto_-6393 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b885 -> b1839
node session_CopyInto_-6394 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b885 -> b1839
node session_CopyInto_-6395 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b885 -> b1839
node dispatch1570 micro raw32 b1839 -> b1839
node dispatch1571 micro raw33 b1839 -> b1839
node dispatch1572 micro raw34 b1839 -> b1839
node session_CopyInto_-6396 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1573 micro raw35 b1838 b1571 b1572 b1837 -> b1838
node dispatch1574 micro raw35 b1838_131072_131072 b1571 b1572 b1837_262144_262144 -> b1838_131072_131072
node dispatch1575 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6397 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b885
node session_CopyInto_-6398 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b885
node session_VecOp_-6399 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6400 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b904 -> b1835
node dispatch1576 micro raw31 b1836 b1575 b1576 b1835 -> b1836
node dispatch1577 micro raw31 b1836_593920_593920 b1575 b1576 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6401 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6402 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b903 -> b1839
node session_CopyInto_-6403 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b903 -> b1839
node session_CopyInto_-6404 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b903 -> b1839
node session_CopyInto_-6405 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b903 -> b1839
node dispatch1578 micro raw32 b1839 -> b1839
node dispatch1579 micro raw33 b1839 -> b1839
node dispatch1580 micro raw34 b1839 -> b1839
node session_CopyInto_-6406 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1581 micro raw35 b1838 b1579 b1580 b1837 -> b1838
node dispatch1582 micro raw35 b1838_131072_131072 b1579 b1580 b1837_262144_262144 -> b1838_131072_131072
node dispatch1583 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6407 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b903
node session_CopyInto_-6408 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b903
node session_VecOp_-6409 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6410 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b922 -> b1835
node dispatch1584 micro raw31 b1836 b1583 b1584 b1835 -> b1836
node dispatch1585 micro raw31 b1836_593920_593920 b1583 b1584 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6411 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6412 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b921 -> b1839
node session_CopyInto_-6413 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b921 -> b1839
node session_CopyInto_-6414 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b921 -> b1839
node session_CopyInto_-6415 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b921 -> b1839
node dispatch1586 micro raw32 b1839 -> b1839
node dispatch1587 micro raw33 b1839 -> b1839
node dispatch1588 micro raw34 b1839 -> b1839
node session_CopyInto_-6416 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1589 micro raw35 b1838 b1587 b1588 b1837 -> b1838
node dispatch1590 micro raw35 b1838_131072_131072 b1587 b1588 b1837_262144_262144 -> b1838_131072_131072
node dispatch1591 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6417 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b921
node session_CopyInto_-6418 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b921
node session_VecOp_-6419 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6423 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b940 -> b1835
node dispatch1592 micro raw31 b1836 b1591 b1592 b1835 -> b1836
node dispatch1593 micro raw31 b1836_593920_593920 b1591 b1592 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6424 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6425 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b939 -> b1839
node session_CopyInto_-6426 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b939 -> b1839
node session_CopyInto_-6427 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b939 -> b1839
node session_CopyInto_-6428 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b939 -> b1839
node dispatch1594 micro raw32 b1839 -> b1839
node dispatch1595 micro raw33 b1839 -> b1839
node dispatch1596 micro raw34 b1839 -> b1839
node session_CopyInto_-6429 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1597 micro raw35 b1838 b1595 b1596 b1837 -> b1838
node dispatch1598 micro raw35 b1838_131072_131072 b1595 b1596 b1837_262144_262144 -> b1838_131072_131072
node dispatch1599 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6430 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b939
node session_CopyInto_-6431 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b939
node session_VecOp_-6432 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6433 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b958 -> b1835
node dispatch1600 micro raw31 b1836 b1599 b1600 b1835 -> b1836
node dispatch1601 micro raw31 b1836_593920_593920 b1599 b1600 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6434 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6435 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b957 -> b1839
node session_CopyInto_-6436 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b957 -> b1839
node session_CopyInto_-6437 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b957 -> b1839
node session_CopyInto_-6438 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b957 -> b1839
node dispatch1602 micro raw32 b1839 -> b1839
node dispatch1603 micro raw33 b1839 -> b1839
node dispatch1604 micro raw34 b1839 -> b1839
node session_CopyInto_-6439 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1605 micro raw35 b1838 b1603 b1604 b1837 -> b1838
node dispatch1606 micro raw35 b1838_131072_131072 b1603 b1604 b1837_262144_262144 -> b1838_131072_131072
node dispatch1607 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6440 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b957
node session_CopyInto_-6441 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b957
node session_VecOp_-6442 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6443 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b976 -> b1835
node dispatch1608 micro raw31 b1836 b1615 b1616 b1835 -> b1836
node dispatch1609 micro raw31 b1836_593920_593920 b1615 b1616 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6444 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6445 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b975 -> b1839
node session_CopyInto_-6446 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b975 -> b1839
node session_CopyInto_-6447 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b975 -> b1839
node session_CopyInto_-6448 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b975 -> b1839
node dispatch1610 micro raw32 b1839 -> b1839
node dispatch1611 micro raw33 b1839 -> b1839
node dispatch1612 micro raw34 b1839 -> b1839
node session_CopyInto_-6449 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1613 micro raw35 b1838 b1619 b1620 b1837 -> b1838
node dispatch1614 micro raw35 b1838_131072_131072 b1619 b1620 b1837_262144_262144 -> b1838_131072_131072
node dispatch1615 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6450 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b975
node session_CopyInto_-6451 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b975
node session_VecOp_-6452 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6453 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b994 -> b1835
node dispatch1616 micro raw31 b1836 b1623 b1624 b1835 -> b1836
node dispatch1617 micro raw31 b1836_593920_593920 b1623 b1624 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6454 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6455 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b993 -> b1839
node session_CopyInto_-6456 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b993 -> b1839
node session_CopyInto_-6457 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b993 -> b1839
node session_CopyInto_-6458 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b993 -> b1839
node dispatch1618 micro raw32 b1839 -> b1839
node dispatch1619 micro raw33 b1839 -> b1839
node dispatch1620 micro raw34 b1839 -> b1839
node session_CopyInto_-6459 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1621 micro raw35 b1838 b1627 b1628 b1837 -> b1838
node dispatch1622 micro raw35 b1838_131072_131072 b1627 b1628 b1837_262144_262144 -> b1838_131072_131072
node dispatch1623 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6460 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b993
node session_CopyInto_-6461 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b993
node session_VecOp_-6462 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6466 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1012 -> b1835
node dispatch1624 micro raw31 b1836 b1631 b1632 b1835 -> b1836
node dispatch1625 micro raw31 b1836_593920_593920 b1631 b1632 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6467 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6468 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1011 -> b1839
node session_CopyInto_-6469 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1011 -> b1839
node session_CopyInto_-6470 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1011 -> b1839
node session_CopyInto_-6471 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1011 -> b1839
node dispatch1626 micro raw32 b1839 -> b1839
node dispatch1627 micro raw33 b1839 -> b1839
node dispatch1628 micro raw34 b1839 -> b1839
node session_CopyInto_-6472 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1629 micro raw35 b1838 b1635 b1636 b1837 -> b1838
node dispatch1630 micro raw35 b1838_131072_131072 b1635 b1636 b1837_262144_262144 -> b1838_131072_131072
node dispatch1631 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6473 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1011
node session_CopyInto_-6474 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1011
node session_VecOp_-6475 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6476 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1030 -> b1835
node dispatch1632 micro raw31 b1836 b1639 b1640 b1835 -> b1836
node dispatch1633 micro raw31 b1836_593920_593920 b1639 b1640 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6477 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6478 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1029 -> b1839
node session_CopyInto_-6479 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1029 -> b1839
node session_CopyInto_-6480 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1029 -> b1839
node session_CopyInto_-6481 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1029 -> b1839
node dispatch1634 micro raw32 b1839 -> b1839
node dispatch1635 micro raw33 b1839 -> b1839
node dispatch1636 micro raw34 b1839 -> b1839
node session_CopyInto_-6482 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1637 micro raw35 b1838 b1643 b1644 b1837 -> b1838
node dispatch1638 micro raw35 b1838_131072_131072 b1643 b1644 b1837_262144_262144 -> b1838_131072_131072
node dispatch1639 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6483 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1029
node session_CopyInto_-6484 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1029
node session_VecOp_-6485 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6486 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1048 -> b1835
node dispatch1640 micro raw31 b1836 b1647 b1648 b1835 -> b1836
node dispatch1641 micro raw31 b1836_593920_593920 b1647 b1648 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6487 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6488 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1047 -> b1839
node session_CopyInto_-6489 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1047 -> b1839
node session_CopyInto_-6490 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1047 -> b1839
node session_CopyInto_-6491 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1047 -> b1839
node dispatch1642 micro raw32 b1839 -> b1839
node dispatch1643 micro raw33 b1839 -> b1839
node dispatch1644 micro raw34 b1839 -> b1839
node session_CopyInto_-6492 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1645 micro raw35 b1838 b1651 b1652 b1837 -> b1838
node dispatch1646 micro raw35 b1838_131072_131072 b1651 b1652 b1837_262144_262144 -> b1838_131072_131072
node dispatch1647 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6493 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1047
node session_CopyInto_-6494 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1047
node session_VecOp_-6495 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6496 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1066 -> b1835
node dispatch1648 micro raw31 b1836 b1655 b1656 b1835 -> b1836
node dispatch1649 micro raw31 b1836_593920_593920 b1655 b1656 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6497 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6498 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1065 -> b1839
node session_CopyInto_-6499 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1065 -> b1839
node session_CopyInto_-6500 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1065 -> b1839
node session_CopyInto_-6501 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1065 -> b1839
node dispatch1650 micro raw32 b1839 -> b1839
node dispatch1651 micro raw33 b1839 -> b1839
node dispatch1652 micro raw34 b1839 -> b1839
node session_CopyInto_-6502 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1653 micro raw35 b1838 b1659 b1660 b1837 -> b1838
node dispatch1654 micro raw35 b1838_131072_131072 b1659 b1660 b1837_262144_262144 -> b1838_131072_131072
node dispatch1655 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6503 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1065
node session_CopyInto_-6504 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1065
node session_MatmulNTAccelerated_-6505 session sessionMatmulNTAccelerated:M=16,K=4096,N=12288,bOff=0 b1835 b1070 -> b1849
node dispatch1656 micro raw37 b1849 b1071 -> b1849
node session_CopyInto_-6507 session sessionCopyInto:srcOff=0,dstOff=0,n=196608 b1849 -> b1846
node session_CopyInto_-6508 session sessionCopyInto:srcOff=16384,dstOff=659456,n=4096 b1074 -> b1846
node session_CopyInto_-6509 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b1074 -> b1854
node dispatch1657 micro raw5 b1846_2654208_2129920 b1854 -> b1846_2654208_2129920
node dispatch1658 micro raw38 b1846 -> b1846
node dispatch1659 micro raw39 b1846 -> b1846
node session_CopyInto_-6510 session sessionCopyInto:srcOff=327680,dstOff=0,n=65536 b1846 -> b1850
node session_MatmulNTAccelerated_-6511 session sessionMatmulNTAccelerated:M=16,K=4096,N=4096,bOff=0 b1850 b1072 -> b1851
node dispatch1660 micro raw40 b1851 b1073 -> b1851
node session_CopyInto_-6513 session sessionCopyInto:srcOff=663552,dstOff=0,n=131072 b1846 -> b1854
node dispatch1661 micro raw5 b1074_81920_524288 b1854 -> b1074_81920_524288
node session_CopyInto_-6514 session sessionCopyInto:srcOff=925696,dstOff=16384,n=4096 b1846 -> b1074
node dispatch1662 micro raw41 b1852 b1838 b1851 -> b1852
node session_MatmulNTAccelerated_-6515 session sessionMatmulNTAccelerated:M=16,K=8192,N=4096,bOff=0 b1852 b1080 -> b1853
node dispatch1663 micro raw40 b1853 b1081 -> b1853
node dispatch1664 micro raw42 b1838 b1853 b1851 -> b1838
node session_VecOp_-6517 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6521 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1101 -> b1835
node dispatch1665 micro raw31 b1836 b1663 b1664 b1835 -> b1836
node dispatch1666 micro raw31 b1836_593920_593920 b1663 b1664 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6522 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6523 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1100 -> b1839
node session_CopyInto_-6524 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1100 -> b1839
node session_CopyInto_-6525 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1100 -> b1839
node session_CopyInto_-6526 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1100 -> b1839
node dispatch1667 micro raw32 b1839 -> b1839
node dispatch1668 micro raw33 b1839 -> b1839
node dispatch1669 micro raw34 b1839 -> b1839
node session_CopyInto_-6527 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1670 micro raw35 b1838 b1667 b1668 b1837 -> b1838
node dispatch1671 micro raw35 b1838_131072_131072 b1667 b1668 b1837_262144_262144 -> b1838_131072_131072
node dispatch1672 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6528 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1100
node session_CopyInto_-6529 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1100
node session_VecOp_-6530 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6531 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1119 -> b1835
node dispatch1673 micro raw31 b1836 b1671 b1672 b1835 -> b1836
node dispatch1674 micro raw31 b1836_593920_593920 b1671 b1672 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6532 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6533 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1118 -> b1839
node session_CopyInto_-6534 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1118 -> b1839
node session_CopyInto_-6535 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1118 -> b1839
node session_CopyInto_-6536 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1118 -> b1839
node dispatch1675 micro raw32 b1839 -> b1839
node dispatch1676 micro raw33 b1839 -> b1839
node dispatch1677 micro raw34 b1839 -> b1839
node session_CopyInto_-6537 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1678 micro raw35 b1838 b1675 b1676 b1837 -> b1838
node dispatch1679 micro raw35 b1838_131072_131072 b1675 b1676 b1837_262144_262144 -> b1838_131072_131072
node dispatch1680 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6538 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1118
node session_CopyInto_-6539 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1118
node session_VecOp_-6540 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6541 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1137 -> b1835
node dispatch1681 micro raw31 b1836 b1679 b1680 b1835 -> b1836
node dispatch1682 micro raw31 b1836_593920_593920 b1679 b1680 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6542 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6543 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1136 -> b1839
node session_CopyInto_-6544 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1136 -> b1839
node session_CopyInto_-6545 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1136 -> b1839
node session_CopyInto_-6546 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1136 -> b1839
node dispatch1683 micro raw32 b1839 -> b1839
node dispatch1684 micro raw33 b1839 -> b1839
node dispatch1685 micro raw34 b1839 -> b1839
node session_CopyInto_-6547 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1686 micro raw35 b1838 b1683 b1684 b1837 -> b1838
node dispatch1687 micro raw35 b1838_131072_131072 b1683 b1684 b1837_262144_262144 -> b1838_131072_131072
node dispatch1688 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6548 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1136
node session_CopyInto_-6549 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1136
node session_VecOp_-6550 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6551 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1155 -> b1835
node dispatch1689 micro raw31 b1836 b1687 b1688 b1835 -> b1836
node dispatch1690 micro raw31 b1836_593920_593920 b1687 b1688 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6552 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6553 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1154 -> b1839
node session_CopyInto_-6554 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1154 -> b1839
node session_CopyInto_-6555 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1154 -> b1839
node session_CopyInto_-6556 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1154 -> b1839
node dispatch1691 micro raw32 b1839 -> b1839
node dispatch1692 micro raw33 b1839 -> b1839
node dispatch1693 micro raw34 b1839 -> b1839
node session_CopyInto_-6557 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1694 micro raw35 b1838 b1691 b1692 b1837 -> b1838
node dispatch1695 micro raw35 b1838_131072_131072 b1691 b1692 b1837_262144_262144 -> b1838_131072_131072
node dispatch1696 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6558 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1154
node session_CopyInto_-6559 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1154
node session_VecOp_-6560 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6564 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1173 -> b1835
node dispatch1697 micro raw31 b1836 b1703 b1704 b1835 -> b1836
node dispatch1698 micro raw31 b1836_593920_593920 b1703 b1704 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6565 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6566 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1172 -> b1839
node session_CopyInto_-6567 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1172 -> b1839
node session_CopyInto_-6568 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1172 -> b1839
node session_CopyInto_-6569 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1172 -> b1839
node dispatch1699 micro raw32 b1839 -> b1839
node dispatch1700 micro raw33 b1839 -> b1839
node dispatch1701 micro raw34 b1839 -> b1839
node session_CopyInto_-6570 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1702 micro raw35 b1838 b1707 b1708 b1837 -> b1838
node dispatch1703 micro raw35 b1838_131072_131072 b1707 b1708 b1837_262144_262144 -> b1838_131072_131072
node dispatch1704 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6571 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1172
node session_CopyInto_-6572 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1172
node session_VecOp_-6573 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6574 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1191 -> b1835
node dispatch1705 micro raw31 b1836 b1711 b1712 b1835 -> b1836
node dispatch1706 micro raw31 b1836_593920_593920 b1711 b1712 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6575 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6576 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1190 -> b1839
node session_CopyInto_-6577 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1190 -> b1839
node session_CopyInto_-6578 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1190 -> b1839
node session_CopyInto_-6579 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1190 -> b1839
node dispatch1707 micro raw32 b1839 -> b1839
node dispatch1708 micro raw33 b1839 -> b1839
node dispatch1709 micro raw34 b1839 -> b1839
node session_CopyInto_-6580 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1710 micro raw35 b1838 b1715 b1716 b1837 -> b1838
node dispatch1711 micro raw35 b1838_131072_131072 b1715 b1716 b1837_262144_262144 -> b1838_131072_131072
node dispatch1712 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6581 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1190
node session_CopyInto_-6582 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1190
node session_VecOp_-6583 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6584 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1209 -> b1835
node dispatch1713 micro raw31 b1836 b1719 b1720 b1835 -> b1836
node dispatch1714 micro raw31 b1836_593920_593920 b1719 b1720 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6585 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6586 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1208 -> b1839
node session_CopyInto_-6587 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1208 -> b1839
node session_CopyInto_-6588 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1208 -> b1839
node session_CopyInto_-6589 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1208 -> b1839
node dispatch1715 micro raw32 b1839 -> b1839
node dispatch1716 micro raw33 b1839 -> b1839
node dispatch1717 micro raw34 b1839 -> b1839
node session_CopyInto_-6590 session sessionCopyInto:srcOff=3176960,dstOff=0,n=131072 b1839 -> b1837
node dispatch1718 micro raw35 b1838 b1723 b1724 b1837 -> b1838
node dispatch1719 micro raw35 b1838_131072_131072 b1723 b1724 b1837_262144_262144 -> b1838_131072_131072
node dispatch1720 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6591 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1208
node session_CopyInto_-6592 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1208
node session_VecOp_-6593 session sessionVecOp:op=add,n=65536 b1834 b1838 -> b1834
node session_RmsnormEnc_-6594 session sessionRmsnormEnc:rows=16,cols=4096,eps=h3727c5ac b1834 b1227 -> b1835
node dispatch1721 micro raw31 b1836 b1727 b1728 b1835 -> b1836
node dispatch1722 micro raw31 b1836_593920_593920 b1727 b1728 b1835_131072_131072 -> b1836_593920_593920
node session_CopyInto_-6595 session sessionCopyInto:srcOff=0,dstOff=0,n=296960 b1836 -> b1839
node session_CopyInto_-6596 session sessionCopyInto:srcOff=45696,dstOff=296960,n=51200 b1226 -> b1839
node session_CopyInto_-6597 session sessionCopyInto:srcOff=37120,dstOff=348160,n=8576 b1226 -> b1839
node session_CopyInto_-6598 session sessionCopyInto:srcOff=127616,dstOff=686464,n=1048576 b1226 -> b1839
node session_CopyInto_-6599 session sessionCopyInto:srcOff=96896,dstOff=3308032,n=30720 b1226 -> b1839
node dispatch1723 micro raw32 b1839 -> b1839
node dispatch1724 micro raw33 b1839 -> b1839
node dispatch1725 micro raw34 b1839 -> b1839
node dispatch1728 micro raw36 b1840 b1836 -> b1840
node session_CopyInto_-6601 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1840 -> b1226
node session_CopyInto_-6602 session sessionCopyInto:srcOff=1735040,dstOff=127616,n=1048576 b1839 -> b1226
phase chunk5 rows=8
node embedding_5 ext fir.embedding token backbone_embeddings_weight_emb -> b1867
node session_CopyInto_-6647 session sessionCopyInto:srcOff=0,dstOff=0,n=32768 b1867 -> b1858
node session_RmsnormEnc_-6674 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b24 -> b1859
node dispatch1735 micro raw31 b1860 b1247 b1248 b1859 -> b1860
node session_CopyInto_-6675 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6676 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b23 -> b1863
node session_CopyInto_-6677 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b23 -> b1863
node session_CopyInto_-6678 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b23 -> b1863
node session_CopyInto_-6679 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b23 -> b1863
node dispatch1736 micro raw43 b1863 -> b1863
node dispatch1737 micro raw44 b1863 -> b1863
node dispatch1738 micro raw45 b1863 -> b1863
node session_CopyInto_-6680 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1739 micro raw35 b1862 b1251 b1252 b1861 -> b1862
node dispatch1740 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6681 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b23
node session_CopyInto_-6682 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b23
node session_VecOp_-6683 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6684 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b42 -> b1859
node dispatch1741 micro raw31 b1860 b1255 b1256 b1859 -> b1860
node session_CopyInto_-6685 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6686 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b41 -> b1863
node session_CopyInto_-6687 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b41 -> b1863
node session_CopyInto_-6688 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b41 -> b1863
node session_CopyInto_-6689 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b41 -> b1863
node dispatch1742 micro raw43 b1863 -> b1863
node dispatch1743 micro raw44 b1863 -> b1863
node dispatch1744 micro raw45 b1863 -> b1863
node session_CopyInto_-6690 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1745 micro raw35 b1862 b1259 b1260 b1861 -> b1862
node dispatch1746 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6691 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b41
node session_CopyInto_-6692 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b41
node session_VecOp_-6693 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6694 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b60 -> b1859
node dispatch1747 micro raw31 b1860 b1343 b1344 b1859 -> b1860
node session_CopyInto_-6695 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6696 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b59 -> b1863
node session_CopyInto_-6697 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b59 -> b1863
node session_CopyInto_-6698 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b59 -> b1863
node session_CopyInto_-6699 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b59 -> b1863
node dispatch1748 micro raw43 b1863 -> b1863
node dispatch1749 micro raw44 b1863 -> b1863
node dispatch1750 micro raw45 b1863 -> b1863
node session_CopyInto_-6700 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1751 micro raw35 b1862 b1347 b1348 b1861 -> b1862
node dispatch1752 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6701 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b59
node session_CopyInto_-6702 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b59
node session_VecOp_-6703 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6704 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b78 -> b1859
node dispatch1753 micro raw31 b1860 b1431 b1432 b1859 -> b1860
node session_CopyInto_-6705 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6706 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b77 -> b1863
node session_CopyInto_-6707 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b77 -> b1863
node session_CopyInto_-6708 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b77 -> b1863
node session_CopyInto_-6709 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b77 -> b1863
node dispatch1754 micro raw43 b1863 -> b1863
node dispatch1755 micro raw44 b1863 -> b1863
node dispatch1756 micro raw45 b1863 -> b1863
node session_CopyInto_-6710 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1757 micro raw35 b1862 b1435 b1436 b1861 -> b1862
node dispatch1758 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6711 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b77
node session_CopyInto_-6712 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b77
node session_VecOp_-6713 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6717 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b96 -> b1859
node dispatch1759 micro raw31 b1860 b1519 b1520 b1859 -> b1860
node session_CopyInto_-6718 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6719 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b95 -> b1863
node session_CopyInto_-6720 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b95 -> b1863
node session_CopyInto_-6721 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b95 -> b1863
node session_CopyInto_-6722 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b95 -> b1863
node dispatch1760 micro raw43 b1863 -> b1863
node dispatch1761 micro raw44 b1863 -> b1863
node dispatch1762 micro raw45 b1863 -> b1863
node session_CopyInto_-6723 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1763 micro raw35 b1862 b1523 b1524 b1861 -> b1862
node dispatch1764 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6724 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b95
node session_CopyInto_-6725 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b95
node session_VecOp_-6726 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6727 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b114 -> b1859
node dispatch1765 micro raw31 b1860 b1607 b1608 b1859 -> b1860
node session_CopyInto_-6728 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6729 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b113 -> b1863
node session_CopyInto_-6730 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b113 -> b1863
node session_CopyInto_-6731 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b113 -> b1863
node session_CopyInto_-6732 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b113 -> b1863
node dispatch1766 micro raw43 b1863 -> b1863
node dispatch1767 micro raw44 b1863 -> b1863
node dispatch1768 micro raw45 b1863 -> b1863
node session_CopyInto_-6733 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1769 micro raw35 b1862 b1611 b1612 b1861 -> b1862
node dispatch1770 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6734 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b113
node session_CopyInto_-6735 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b113
node session_VecOp_-6736 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6737 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b132 -> b1859
node dispatch1771 micro raw31 b1860 b1695 b1696 b1859 -> b1860
node session_CopyInto_-6738 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6739 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b131 -> b1863
node session_CopyInto_-6740 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b131 -> b1863
node session_CopyInto_-6741 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b131 -> b1863
node session_CopyInto_-6742 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b131 -> b1863
node dispatch1772 micro raw43 b1863 -> b1863
node dispatch1773 micro raw44 b1863 -> b1863
node dispatch1774 micro raw45 b1863 -> b1863
node session_CopyInto_-6743 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1775 micro raw35 b1862 b1699 b1700 b1861 -> b1862
node dispatch1776 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6744 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b131
node session_CopyInto_-6745 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b131
node session_VecOp_-6746 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6747 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b150 -> b1859
node dispatch1777 micro raw31 b1860 b1735 b1736 b1859 -> b1860
node session_CopyInto_-6748 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6749 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b149 -> b1863
node session_CopyInto_-6750 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b149 -> b1863
node session_CopyInto_-6751 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b149 -> b1863
node session_CopyInto_-6752 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b149 -> b1863
node dispatch1778 micro raw43 b1863 -> b1863
node dispatch1779 micro raw44 b1863 -> b1863
node dispatch1780 micro raw45 b1863 -> b1863
node session_CopyInto_-6753 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1781 micro raw35 b1862 b1739 b1740 b1861 -> b1862
node dispatch1782 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6754 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b149
node session_CopyInto_-6755 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b149
node dispatch1783 micro raw47 b1873 b154 b1859 -> b1873
node dispatch1784 micro raw48 b1873 b155 -> b1873
node session_CopyInto_-6757 session sessionCopyInto:srcOff=0,dstOff=0,n=98304 b1873 -> b1870
node session_CopyInto_-6758 session sessionCopyInto:srcOff=16384,dstOff=329728,n=4096 b158 -> b1870
node session_CopyInto_-6759 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b158 -> b1878
node dispatch1785 micro raw5 b1870_1335296_2129920 b1878 -> b1870_1335296_2129920
node dispatch1786 micro raw49 b1870 -> b1870
node dispatch1787 micro raw50 b1870 -> b1870
node session_CopyInto_-6760 session sessionCopyInto:srcOff=163840,dstOff=0,n=32768 b1870 -> b1874
node dispatch1788 micro raw51 b1875 b156 b1874 -> b1875
node dispatch1789 micro raw52 b1875 b157 -> b1875
node session_CopyInto_-6762 session sessionCopyInto:srcOff=333824,dstOff=0,n=131072 b1870 -> b1878
node dispatch1790 micro raw5 b158_81920_524288 b1878 -> b158_81920_524288
node session_CopyInto_-6763 session sessionCopyInto:srcOff=595968,dstOff=16384,n=4096 b1870 -> b158
node dispatch1791 micro raw53 b1876 b1862 b1875 -> b1876
node dispatch1792 micro raw54 b1877 b165 b1876 -> b1877
node dispatch1793 micro raw52 b1877 b166 -> b1877
node dispatch1794 micro raw55 b1862 b1877 b1875 -> b1862
node session_VecOp_-6765 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6769 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b186 -> b1859
node dispatch1795 micro raw31 b1860 b1743 b1744 b1859 -> b1860
node session_CopyInto_-6770 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6771 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b185 -> b1863
node session_CopyInto_-6772 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b185 -> b1863
node session_CopyInto_-6773 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b185 -> b1863
node session_CopyInto_-6774 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b185 -> b1863
node dispatch1796 micro raw43 b1863 -> b1863
node dispatch1797 micro raw44 b1863 -> b1863
node dispatch1798 micro raw45 b1863 -> b1863
node session_CopyInto_-6775 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1799 micro raw35 b1862 b1747 b1748 b1861 -> b1862
node dispatch1800 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6776 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b185
node session_CopyInto_-6777 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b185
node session_VecOp_-6778 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6779 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b204 -> b1859
node dispatch1801 micro raw31 b1860 b1751 b1752 b1859 -> b1860
node session_CopyInto_-6780 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6781 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b203 -> b1863
node session_CopyInto_-6782 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b203 -> b1863
node session_CopyInto_-6783 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b203 -> b1863
node session_CopyInto_-6784 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b203 -> b1863
node dispatch1802 micro raw43 b1863 -> b1863
node dispatch1803 micro raw44 b1863 -> b1863
node dispatch1804 micro raw45 b1863 -> b1863
node session_CopyInto_-6785 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1805 micro raw35 b1862 b1755 b1756 b1861 -> b1862
node dispatch1806 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6786 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b203
node session_CopyInto_-6787 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b203
node session_VecOp_-6788 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6789 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b222 -> b1859
node dispatch1807 micro raw31 b1860 b1263 b1264 b1859 -> b1860
node session_CopyInto_-6790 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6791 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b221 -> b1863
node session_CopyInto_-6792 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b221 -> b1863
node session_CopyInto_-6793 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b221 -> b1863
node session_CopyInto_-6794 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b221 -> b1863
node dispatch1808 micro raw43 b1863 -> b1863
node dispatch1809 micro raw44 b1863 -> b1863
node dispatch1810 micro raw45 b1863 -> b1863
node session_CopyInto_-6795 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1811 micro raw35 b1862 b1267 b1268 b1861 -> b1862
node dispatch1812 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6796 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b221
node session_CopyInto_-6797 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b221
node session_VecOp_-6798 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6799 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b240 -> b1859
node dispatch1813 micro raw31 b1860 b1271 b1272 b1859 -> b1860
node session_CopyInto_-6800 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6801 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b239 -> b1863
node session_CopyInto_-6802 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b239 -> b1863
node session_CopyInto_-6803 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b239 -> b1863
node session_CopyInto_-6804 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b239 -> b1863
node dispatch1814 micro raw43 b1863 -> b1863
node dispatch1815 micro raw44 b1863 -> b1863
node dispatch1816 micro raw45 b1863 -> b1863
node session_CopyInto_-6805 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1817 micro raw35 b1862 b1275 b1276 b1861 -> b1862
node dispatch1818 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6806 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b239
node session_CopyInto_-6807 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b239
node session_VecOp_-6808 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6812 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b258 -> b1859
node dispatch1819 micro raw31 b1860 b1279 b1280 b1859 -> b1860
node session_CopyInto_-6813 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6814 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b257 -> b1863
node session_CopyInto_-6815 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b257 -> b1863
node session_CopyInto_-6816 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b257 -> b1863
node session_CopyInto_-6817 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b257 -> b1863
node dispatch1820 micro raw43 b1863 -> b1863
node dispatch1821 micro raw44 b1863 -> b1863
node dispatch1822 micro raw45 b1863 -> b1863
node session_CopyInto_-6818 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1823 micro raw35 b1862 b1283 b1284 b1861 -> b1862
node dispatch1824 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6819 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b257
node session_CopyInto_-6820 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b257
node session_VecOp_-6821 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6822 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b276 -> b1859
node dispatch1825 micro raw31 b1860 b1287 b1288 b1859 -> b1860
node session_CopyInto_-6823 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6824 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b275 -> b1863
node session_CopyInto_-6825 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b275 -> b1863
node session_CopyInto_-6826 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b275 -> b1863
node session_CopyInto_-6827 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b275 -> b1863
node dispatch1826 micro raw43 b1863 -> b1863
node dispatch1827 micro raw44 b1863 -> b1863
node dispatch1828 micro raw45 b1863 -> b1863
node session_CopyInto_-6828 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1829 micro raw35 b1862 b1291 b1292 b1861 -> b1862
node dispatch1830 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6829 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b275
node session_CopyInto_-6830 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b275
node session_VecOp_-6831 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6832 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b294 -> b1859
node dispatch1831 micro raw31 b1860 b1295 b1296 b1859 -> b1860
node session_CopyInto_-6833 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6834 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b293 -> b1863
node session_CopyInto_-6835 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b293 -> b1863
node session_CopyInto_-6836 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b293 -> b1863
node session_CopyInto_-6837 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b293 -> b1863
node dispatch1832 micro raw43 b1863 -> b1863
node dispatch1833 micro raw44 b1863 -> b1863
node dispatch1834 micro raw45 b1863 -> b1863
node session_CopyInto_-6838 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1835 micro raw35 b1862 b1299 b1300 b1861 -> b1862
node dispatch1836 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6839 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b293
node session_CopyInto_-6840 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b293
node session_VecOp_-6841 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6842 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b312 -> b1859
node dispatch1837 micro raw31 b1860 b1303 b1304 b1859 -> b1860
node session_CopyInto_-6843 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6844 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b311 -> b1863
node session_CopyInto_-6845 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b311 -> b1863
node session_CopyInto_-6846 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b311 -> b1863
node session_CopyInto_-6847 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b311 -> b1863
node dispatch1838 micro raw43 b1863 -> b1863
node dispatch1839 micro raw44 b1863 -> b1863
node dispatch1840 micro raw45 b1863 -> b1863
node session_CopyInto_-6848 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1841 micro raw35 b1862 b1307 b1308 b1861 -> b1862
node dispatch1842 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6849 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b311
node session_CopyInto_-6850 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b311
node session_VecOp_-6851 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6855 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b330 -> b1859
node dispatch1843 micro raw31 b1860 b1311 b1312 b1859 -> b1860
node session_CopyInto_-6856 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6857 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b329 -> b1863
node session_CopyInto_-6858 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b329 -> b1863
node session_CopyInto_-6859 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b329 -> b1863
node session_CopyInto_-6860 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b329 -> b1863
node dispatch1844 micro raw43 b1863 -> b1863
node dispatch1845 micro raw44 b1863 -> b1863
node dispatch1846 micro raw45 b1863 -> b1863
node session_CopyInto_-6861 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1847 micro raw35 b1862 b1315 b1316 b1861 -> b1862
node dispatch1848 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6862 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b329
node session_CopyInto_-6863 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b329
node session_VecOp_-6864 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6865 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b348 -> b1859
node dispatch1849 micro raw31 b1860 b1319 b1320 b1859 -> b1860
node session_CopyInto_-6866 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6867 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b347 -> b1863
node session_CopyInto_-6868 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b347 -> b1863
node session_CopyInto_-6869 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b347 -> b1863
node session_CopyInto_-6870 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b347 -> b1863
node dispatch1850 micro raw43 b1863 -> b1863
node dispatch1851 micro raw44 b1863 -> b1863
node dispatch1852 micro raw45 b1863 -> b1863
node session_CopyInto_-6871 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1853 micro raw35 b1862 b1323 b1324 b1861 -> b1862
node dispatch1854 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6872 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b347
node session_CopyInto_-6873 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b347
node session_VecOp_-6874 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6875 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b366 -> b1859
node dispatch1855 micro raw31 b1860 b1327 b1328 b1859 -> b1860
node session_CopyInto_-6876 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6877 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b365 -> b1863
node session_CopyInto_-6878 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b365 -> b1863
node session_CopyInto_-6879 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b365 -> b1863
node session_CopyInto_-6880 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b365 -> b1863
node dispatch1856 micro raw43 b1863 -> b1863
node dispatch1857 micro raw44 b1863 -> b1863
node dispatch1858 micro raw45 b1863 -> b1863
node session_CopyInto_-6881 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1859 micro raw35 b1862 b1331 b1332 b1861 -> b1862
node dispatch1860 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6882 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b365
node session_CopyInto_-6883 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b365
node session_VecOp_-6884 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6885 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b384 -> b1859
node dispatch1861 micro raw31 b1860 b1335 b1336 b1859 -> b1860
node session_CopyInto_-6886 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6887 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b383 -> b1863
node session_CopyInto_-6888 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b383 -> b1863
node session_CopyInto_-6889 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b383 -> b1863
node session_CopyInto_-6890 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b383 -> b1863
node dispatch1862 micro raw43 b1863 -> b1863
node dispatch1863 micro raw44 b1863 -> b1863
node dispatch1864 micro raw45 b1863 -> b1863
node session_CopyInto_-6891 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1865 micro raw35 b1862 b1339 b1340 b1861 -> b1862
node dispatch1866 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6892 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b383
node session_CopyInto_-6893 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b383
node session_VecOp_-6894 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6898 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b402 -> b1859
node dispatch1867 micro raw31 b1860 b1351 b1352 b1859 -> b1860
node session_CopyInto_-6899 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6900 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b401 -> b1863
node session_CopyInto_-6901 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b401 -> b1863
node session_CopyInto_-6902 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b401 -> b1863
node session_CopyInto_-6903 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b401 -> b1863
node dispatch1868 micro raw43 b1863 -> b1863
node dispatch1869 micro raw44 b1863 -> b1863
node dispatch1870 micro raw45 b1863 -> b1863
node session_CopyInto_-6904 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1871 micro raw35 b1862 b1355 b1356 b1861 -> b1862
node dispatch1872 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6905 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b401
node session_CopyInto_-6906 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b401
node session_VecOp_-6907 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6908 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b420 -> b1859
node dispatch1873 micro raw31 b1860 b1359 b1360 b1859 -> b1860
node session_CopyInto_-6909 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6910 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b419 -> b1863
node session_CopyInto_-6911 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b419 -> b1863
node session_CopyInto_-6912 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b419 -> b1863
node session_CopyInto_-6913 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b419 -> b1863
node dispatch1874 micro raw43 b1863 -> b1863
node dispatch1875 micro raw44 b1863 -> b1863
node dispatch1876 micro raw45 b1863 -> b1863
node session_CopyInto_-6914 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1877 micro raw35 b1862 b1363 b1364 b1861 -> b1862
node dispatch1878 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6915 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b419
node session_CopyInto_-6916 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b419
node session_VecOp_-6917 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6918 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b438 -> b1859
node dispatch1879 micro raw31 b1860 b1367 b1368 b1859 -> b1860
node session_CopyInto_-6919 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6920 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b437 -> b1863
node session_CopyInto_-6921 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b437 -> b1863
node session_CopyInto_-6922 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b437 -> b1863
node session_CopyInto_-6923 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b437 -> b1863
node dispatch1880 micro raw43 b1863 -> b1863
node dispatch1881 micro raw44 b1863 -> b1863
node dispatch1882 micro raw45 b1863 -> b1863
node session_CopyInto_-6924 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1883 micro raw35 b1862 b1371 b1372 b1861 -> b1862
node dispatch1884 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6925 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b437
node session_CopyInto_-6926 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b437
node session_VecOp_-6927 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6928 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b456 -> b1859
node dispatch1885 micro raw31 b1860 b1375 b1376 b1859 -> b1860
node session_CopyInto_-6929 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6930 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b455 -> b1863
node session_CopyInto_-6931 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b455 -> b1863
node session_CopyInto_-6932 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b455 -> b1863
node session_CopyInto_-6933 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b455 -> b1863
node dispatch1886 micro raw43 b1863 -> b1863
node dispatch1887 micro raw44 b1863 -> b1863
node dispatch1888 micro raw45 b1863 -> b1863
node session_CopyInto_-6934 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1889 micro raw35 b1862 b1379 b1380 b1861 -> b1862
node dispatch1890 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6935 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b455
node session_CopyInto_-6936 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b455
node dispatch1891 micro raw47 b1873 b460 b1859 -> b1873
node dispatch1892 micro raw48 b1873 b461 -> b1873
node session_CopyInto_-6938 session sessionCopyInto:srcOff=0,dstOff=0,n=98304 b1873 -> b1870
node session_CopyInto_-6939 session sessionCopyInto:srcOff=16384,dstOff=329728,n=4096 b464 -> b1870
node session_CopyInto_-6940 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b464 -> b1878
node dispatch1893 micro raw5 b1870_1335296_2129920 b1878 -> b1870_1335296_2129920
node dispatch1894 micro raw49 b1870 -> b1870
node dispatch1895 micro raw50 b1870 -> b1870
node session_CopyInto_-6941 session sessionCopyInto:srcOff=163840,dstOff=0,n=32768 b1870 -> b1874
node dispatch1896 micro raw51 b1875 b462 b1874 -> b1875
node dispatch1897 micro raw52 b1875 b463 -> b1875
node session_CopyInto_-6943 session sessionCopyInto:srcOff=333824,dstOff=0,n=131072 b1870 -> b1878
node dispatch1898 micro raw5 b464_81920_524288 b1878 -> b464_81920_524288
node session_CopyInto_-6944 session sessionCopyInto:srcOff=595968,dstOff=16384,n=4096 b1870 -> b464
node dispatch1899 micro raw53 b1876 b1862 b1875 -> b1876
node dispatch1900 micro raw54 b1877 b470 b1876 -> b1877
node dispatch1901 micro raw52 b1877 b471 -> b1877
node dispatch1902 micro raw55 b1862 b1877 b1875 -> b1862
node session_VecOp_-6946 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6950 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b491 -> b1859
node dispatch1903 micro raw31 b1860 b1383 b1384 b1859 -> b1860
node session_CopyInto_-6951 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6952 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b490 -> b1863
node session_CopyInto_-6953 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b490 -> b1863
node session_CopyInto_-6954 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b490 -> b1863
node session_CopyInto_-6955 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b490 -> b1863
node dispatch1904 micro raw43 b1863 -> b1863
node dispatch1905 micro raw44 b1863 -> b1863
node dispatch1906 micro raw45 b1863 -> b1863
node session_CopyInto_-6956 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1907 micro raw35 b1862 b1387 b1388 b1861 -> b1862
node dispatch1908 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6957 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b490
node session_CopyInto_-6958 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b490
node session_VecOp_-6959 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6960 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b509 -> b1859
node dispatch1909 micro raw31 b1860 b1391 b1392 b1859 -> b1860
node session_CopyInto_-6961 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6962 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b508 -> b1863
node session_CopyInto_-6963 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b508 -> b1863
node session_CopyInto_-6964 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b508 -> b1863
node session_CopyInto_-6965 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b508 -> b1863
node dispatch1910 micro raw43 b1863 -> b1863
node dispatch1911 micro raw44 b1863 -> b1863
node dispatch1912 micro raw45 b1863 -> b1863
node session_CopyInto_-6966 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1913 micro raw35 b1862 b1395 b1396 b1861 -> b1862
node dispatch1914 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6967 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b508
node session_CopyInto_-6968 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b508
node session_VecOp_-6969 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6970 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b527 -> b1859
node dispatch1915 micro raw31 b1860 b1399 b1400 b1859 -> b1860
node session_CopyInto_-6971 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6972 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b526 -> b1863
node session_CopyInto_-6973 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b526 -> b1863
node session_CopyInto_-6974 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b526 -> b1863
node session_CopyInto_-6975 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b526 -> b1863
node dispatch1916 micro raw43 b1863 -> b1863
node dispatch1917 micro raw44 b1863 -> b1863
node dispatch1918 micro raw45 b1863 -> b1863
node session_CopyInto_-6976 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1919 micro raw35 b1862 b1403 b1404 b1861 -> b1862
node dispatch1920 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6977 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b526
node session_CopyInto_-6978 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b526
node session_VecOp_-6979 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6980 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b545 -> b1859
node dispatch1921 micro raw31 b1860 b1407 b1408 b1859 -> b1860
node session_CopyInto_-6981 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6982 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b544 -> b1863
node session_CopyInto_-6983 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b544 -> b1863
node session_CopyInto_-6984 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b544 -> b1863
node session_CopyInto_-6985 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b544 -> b1863
node dispatch1922 micro raw43 b1863 -> b1863
node dispatch1923 micro raw44 b1863 -> b1863
node dispatch1924 micro raw45 b1863 -> b1863
node session_CopyInto_-6986 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1925 micro raw35 b1862 b1411 b1412 b1861 -> b1862
node dispatch1926 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-6987 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b544
node session_CopyInto_-6988 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b544
node session_VecOp_-6989 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-6993 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b563 -> b1859
node dispatch1927 micro raw31 b1860 b1415 b1416 b1859 -> b1860
node session_CopyInto_-6994 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-6995 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b562 -> b1863
node session_CopyInto_-6996 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b562 -> b1863
node session_CopyInto_-6997 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b562 -> b1863
node session_CopyInto_-6998 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b562 -> b1863
node dispatch1928 micro raw43 b1863 -> b1863
node dispatch1929 micro raw44 b1863 -> b1863
node dispatch1930 micro raw45 b1863 -> b1863
node session_CopyInto_-6999 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1931 micro raw35 b1862 b1419 b1420 b1861 -> b1862
node dispatch1932 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7000 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b562
node session_CopyInto_-7001 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b562
node session_VecOp_-7002 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7003 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b581 -> b1859
node dispatch1933 micro raw31 b1860 b1423 b1424 b1859 -> b1860
node session_CopyInto_-7004 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7005 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b580 -> b1863
node session_CopyInto_-7006 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b580 -> b1863
node session_CopyInto_-7007 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b580 -> b1863
node session_CopyInto_-7008 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b580 -> b1863
node dispatch1934 micro raw43 b1863 -> b1863
node dispatch1935 micro raw44 b1863 -> b1863
node dispatch1936 micro raw45 b1863 -> b1863
node session_CopyInto_-7009 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1937 micro raw35 b1862 b1427 b1428 b1861 -> b1862
node dispatch1938 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7010 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b580
node session_CopyInto_-7011 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b580
node session_VecOp_-7012 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7013 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b599 -> b1859
node dispatch1939 micro raw31 b1860 b1439 b1440 b1859 -> b1860
node session_CopyInto_-7014 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7015 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b598 -> b1863
node session_CopyInto_-7016 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b598 -> b1863
node session_CopyInto_-7017 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b598 -> b1863
node session_CopyInto_-7018 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b598 -> b1863
node dispatch1940 micro raw43 b1863 -> b1863
node dispatch1941 micro raw44 b1863 -> b1863
node dispatch1942 micro raw45 b1863 -> b1863
node session_CopyInto_-7019 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1943 micro raw35 b1862 b1443 b1444 b1861 -> b1862
node dispatch1944 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7020 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b598
node session_CopyInto_-7021 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b598
node session_VecOp_-7022 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7023 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b617 -> b1859
node dispatch1945 micro raw31 b1860 b1447 b1448 b1859 -> b1860
node session_CopyInto_-7024 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7025 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b616 -> b1863
node session_CopyInto_-7026 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b616 -> b1863
node session_CopyInto_-7027 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b616 -> b1863
node session_CopyInto_-7028 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b616 -> b1863
node dispatch1946 micro raw43 b1863 -> b1863
node dispatch1947 micro raw44 b1863 -> b1863
node dispatch1948 micro raw45 b1863 -> b1863
node session_CopyInto_-7029 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1949 micro raw35 b1862 b1451 b1452 b1861 -> b1862
node dispatch1950 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7030 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b616
node session_CopyInto_-7031 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b616
node session_VecOp_-7032 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7036 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b635 -> b1859
node dispatch1951 micro raw31 b1860 b1455 b1456 b1859 -> b1860
node session_CopyInto_-7037 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7038 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b634 -> b1863
node session_CopyInto_-7039 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b634 -> b1863
node session_CopyInto_-7040 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b634 -> b1863
node session_CopyInto_-7041 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b634 -> b1863
node dispatch1952 micro raw43 b1863 -> b1863
node dispatch1953 micro raw44 b1863 -> b1863
node dispatch1954 micro raw45 b1863 -> b1863
node session_CopyInto_-7042 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1955 micro raw35 b1862 b1459 b1460 b1861 -> b1862
node dispatch1956 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7043 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b634
node session_CopyInto_-7044 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b634
node session_VecOp_-7045 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7046 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b653 -> b1859
node dispatch1957 micro raw31 b1860 b1463 b1464 b1859 -> b1860
node session_CopyInto_-7047 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7048 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b652 -> b1863
node session_CopyInto_-7049 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b652 -> b1863
node session_CopyInto_-7050 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b652 -> b1863
node session_CopyInto_-7051 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b652 -> b1863
node dispatch1958 micro raw43 b1863 -> b1863
node dispatch1959 micro raw44 b1863 -> b1863
node dispatch1960 micro raw45 b1863 -> b1863
node session_CopyInto_-7052 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1961 micro raw35 b1862 b1467 b1468 b1861 -> b1862
node dispatch1962 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7053 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b652
node session_CopyInto_-7054 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b652
node session_VecOp_-7055 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7056 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b671 -> b1859
node dispatch1963 micro raw31 b1860 b1471 b1472 b1859 -> b1860
node session_CopyInto_-7057 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7058 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b670 -> b1863
node session_CopyInto_-7059 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b670 -> b1863
node session_CopyInto_-7060 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b670 -> b1863
node session_CopyInto_-7061 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b670 -> b1863
node dispatch1964 micro raw43 b1863 -> b1863
node dispatch1965 micro raw44 b1863 -> b1863
node dispatch1966 micro raw45 b1863 -> b1863
node session_CopyInto_-7062 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1967 micro raw35 b1862 b1475 b1476 b1861 -> b1862
node dispatch1968 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7063 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b670
node session_CopyInto_-7064 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b670
node session_VecOp_-7065 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7066 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b689 -> b1859
node dispatch1969 micro raw31 b1860 b1479 b1480 b1859 -> b1860
node session_CopyInto_-7067 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7068 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b688 -> b1863
node session_CopyInto_-7069 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b688 -> b1863
node session_CopyInto_-7070 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b688 -> b1863
node session_CopyInto_-7071 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b688 -> b1863
node dispatch1970 micro raw43 b1863 -> b1863
node dispatch1971 micro raw44 b1863 -> b1863
node dispatch1972 micro raw45 b1863 -> b1863
node session_CopyInto_-7072 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1973 micro raw35 b1862 b1483 b1484 b1861 -> b1862
node dispatch1974 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7073 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b688
node session_CopyInto_-7074 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b688
node session_VecOp_-7075 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7079 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b707 -> b1859
node dispatch1975 micro raw31 b1860 b1487 b1488 b1859 -> b1860
node session_CopyInto_-7080 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7081 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b706 -> b1863
node session_CopyInto_-7082 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b706 -> b1863
node session_CopyInto_-7083 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b706 -> b1863
node session_CopyInto_-7084 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b706 -> b1863
node dispatch1976 micro raw43 b1863 -> b1863
node dispatch1977 micro raw44 b1863 -> b1863
node dispatch1978 micro raw45 b1863 -> b1863
node session_CopyInto_-7085 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1979 micro raw35 b1862 b1491 b1492 b1861 -> b1862
node dispatch1980 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7086 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b706
node session_CopyInto_-7087 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b706
node session_VecOp_-7088 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7089 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b725 -> b1859
node dispatch1981 micro raw31 b1860 b1495 b1496 b1859 -> b1860
node session_CopyInto_-7090 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7091 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b724 -> b1863
node session_CopyInto_-7092 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b724 -> b1863
node session_CopyInto_-7093 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b724 -> b1863
node session_CopyInto_-7094 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b724 -> b1863
node dispatch1982 micro raw43 b1863 -> b1863
node dispatch1983 micro raw44 b1863 -> b1863
node dispatch1984 micro raw45 b1863 -> b1863
node session_CopyInto_-7095 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1985 micro raw35 b1862 b1499 b1500 b1861 -> b1862
node dispatch1986 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7096 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b724
node session_CopyInto_-7097 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b724
node session_VecOp_-7098 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7099 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b743 -> b1859
node dispatch1987 micro raw31 b1860 b1503 b1504 b1859 -> b1860
node session_CopyInto_-7100 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7101 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b742 -> b1863
node session_CopyInto_-7102 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b742 -> b1863
node session_CopyInto_-7103 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b742 -> b1863
node session_CopyInto_-7104 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b742 -> b1863
node dispatch1988 micro raw43 b1863 -> b1863
node dispatch1989 micro raw44 b1863 -> b1863
node dispatch1990 micro raw45 b1863 -> b1863
node session_CopyInto_-7105 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1991 micro raw35 b1862 b1507 b1508 b1861 -> b1862
node dispatch1992 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7106 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b742
node session_CopyInto_-7107 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b742
node session_VecOp_-7108 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7109 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b761 -> b1859
node dispatch1993 micro raw31 b1860 b1511 b1512 b1859 -> b1860
node session_CopyInto_-7110 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7111 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b760 -> b1863
node session_CopyInto_-7112 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b760 -> b1863
node session_CopyInto_-7113 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b760 -> b1863
node session_CopyInto_-7114 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b760 -> b1863
node dispatch1994 micro raw43 b1863 -> b1863
node dispatch1995 micro raw44 b1863 -> b1863
node dispatch1996 micro raw45 b1863 -> b1863
node session_CopyInto_-7115 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch1997 micro raw35 b1862 b1515 b1516 b1861 -> b1862
node dispatch1998 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7116 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b760
node session_CopyInto_-7117 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b760
node dispatch1999 micro raw47 b1873 b765 b1859 -> b1873
node dispatch2000 micro raw48 b1873 b766 -> b1873
node session_CopyInto_-7119 session sessionCopyInto:srcOff=0,dstOff=0,n=98304 b1873 -> b1870
node session_CopyInto_-7120 session sessionCopyInto:srcOff=16384,dstOff=329728,n=4096 b769 -> b1870
node session_CopyInto_-7121 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b769 -> b1878
node dispatch2001 micro raw5 b1870_1335296_2129920 b1878 -> b1870_1335296_2129920
node dispatch2002 micro raw49 b1870 -> b1870
node dispatch2003 micro raw50 b1870 -> b1870
node session_CopyInto_-7122 session sessionCopyInto:srcOff=163840,dstOff=0,n=32768 b1870 -> b1874
node dispatch2004 micro raw51 b1875 b767 b1874 -> b1875
node dispatch2005 micro raw52 b1875 b768 -> b1875
node session_CopyInto_-7124 session sessionCopyInto:srcOff=333824,dstOff=0,n=131072 b1870 -> b1878
node dispatch2006 micro raw5 b769_81920_524288 b1878 -> b769_81920_524288
node session_CopyInto_-7125 session sessionCopyInto:srcOff=595968,dstOff=16384,n=4096 b1870 -> b769
node dispatch2007 micro raw53 b1876 b1862 b1875 -> b1876
node dispatch2008 micro raw54 b1877 b775 b1876 -> b1877
node dispatch2009 micro raw52 b1877 b776 -> b1877
node dispatch2010 micro raw55 b1862 b1877 b1875 -> b1862
node session_VecOp_-7127 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7131 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b796 -> b1859
node dispatch2011 micro raw31 b1860 b1527 b1528 b1859 -> b1860
node session_CopyInto_-7132 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7133 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b795 -> b1863
node session_CopyInto_-7134 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b795 -> b1863
node session_CopyInto_-7135 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b795 -> b1863
node session_CopyInto_-7136 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b795 -> b1863
node dispatch2012 micro raw43 b1863 -> b1863
node dispatch2013 micro raw44 b1863 -> b1863
node dispatch2014 micro raw45 b1863 -> b1863
node session_CopyInto_-7137 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2015 micro raw35 b1862 b1531 b1532 b1861 -> b1862
node dispatch2016 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7138 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b795
node session_CopyInto_-7139 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b795
node session_VecOp_-7140 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7141 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b814 -> b1859
node dispatch2017 micro raw31 b1860 b1535 b1536 b1859 -> b1860
node session_CopyInto_-7142 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7143 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b813 -> b1863
node session_CopyInto_-7144 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b813 -> b1863
node session_CopyInto_-7145 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b813 -> b1863
node session_CopyInto_-7146 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b813 -> b1863
node dispatch2018 micro raw43 b1863 -> b1863
node dispatch2019 micro raw44 b1863 -> b1863
node dispatch2020 micro raw45 b1863 -> b1863
node session_CopyInto_-7147 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2021 micro raw35 b1862 b1539 b1540 b1861 -> b1862
node dispatch2022 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7148 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b813
node session_CopyInto_-7149 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b813
node session_VecOp_-7150 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7151 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b832 -> b1859
node dispatch2023 micro raw31 b1860 b1543 b1544 b1859 -> b1860
node session_CopyInto_-7152 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7153 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b831 -> b1863
node session_CopyInto_-7154 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b831 -> b1863
node session_CopyInto_-7155 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b831 -> b1863
node session_CopyInto_-7156 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b831 -> b1863
node dispatch2024 micro raw43 b1863 -> b1863
node dispatch2025 micro raw44 b1863 -> b1863
node dispatch2026 micro raw45 b1863 -> b1863
node session_CopyInto_-7157 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2027 micro raw35 b1862 b1547 b1548 b1861 -> b1862
node dispatch2028 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7158 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b831
node session_CopyInto_-7159 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b831
node session_VecOp_-7160 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7161 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b850 -> b1859
node dispatch2029 micro raw31 b1860 b1551 b1552 b1859 -> b1860
node session_CopyInto_-7162 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7163 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b849 -> b1863
node session_CopyInto_-7164 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b849 -> b1863
node session_CopyInto_-7165 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b849 -> b1863
node session_CopyInto_-7166 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b849 -> b1863
node dispatch2030 micro raw43 b1863 -> b1863
node dispatch2031 micro raw44 b1863 -> b1863
node dispatch2032 micro raw45 b1863 -> b1863
node session_CopyInto_-7167 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2033 micro raw35 b1862 b1555 b1556 b1861 -> b1862
node dispatch2034 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7168 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b849
node session_CopyInto_-7169 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b849
node session_VecOp_-7170 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7174 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b868 -> b1859
node dispatch2035 micro raw31 b1860 b1559 b1560 b1859 -> b1860
node session_CopyInto_-7175 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7176 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b867 -> b1863
node session_CopyInto_-7177 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b867 -> b1863
node session_CopyInto_-7178 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b867 -> b1863
node session_CopyInto_-7179 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b867 -> b1863
node dispatch2036 micro raw43 b1863 -> b1863
node dispatch2037 micro raw44 b1863 -> b1863
node dispatch2038 micro raw45 b1863 -> b1863
node session_CopyInto_-7180 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2039 micro raw35 b1862 b1563 b1564 b1861 -> b1862
node dispatch2040 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7181 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b867
node session_CopyInto_-7182 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b867
node session_VecOp_-7183 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7184 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b886 -> b1859
node dispatch2041 micro raw31 b1860 b1567 b1568 b1859 -> b1860
node session_CopyInto_-7185 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7186 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b885 -> b1863
node session_CopyInto_-7187 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b885 -> b1863
node session_CopyInto_-7188 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b885 -> b1863
node session_CopyInto_-7189 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b885 -> b1863
node dispatch2042 micro raw43 b1863 -> b1863
node dispatch2043 micro raw44 b1863 -> b1863
node dispatch2044 micro raw45 b1863 -> b1863
node session_CopyInto_-7190 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2045 micro raw35 b1862 b1571 b1572 b1861 -> b1862
node dispatch2046 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7191 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b885
node session_CopyInto_-7192 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b885
node session_VecOp_-7193 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7194 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b904 -> b1859
node dispatch2047 micro raw31 b1860 b1575 b1576 b1859 -> b1860
node session_CopyInto_-7195 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7196 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b903 -> b1863
node session_CopyInto_-7197 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b903 -> b1863
node session_CopyInto_-7198 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b903 -> b1863
node session_CopyInto_-7199 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b903 -> b1863
node dispatch2048 micro raw43 b1863 -> b1863
node dispatch2049 micro raw44 b1863 -> b1863
node dispatch2050 micro raw45 b1863 -> b1863
node session_CopyInto_-7200 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2051 micro raw35 b1862 b1579 b1580 b1861 -> b1862
node dispatch2052 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7201 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b903
node session_CopyInto_-7202 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b903
node session_VecOp_-7203 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7204 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b922 -> b1859
node dispatch2053 micro raw31 b1860 b1583 b1584 b1859 -> b1860
node session_CopyInto_-7205 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7206 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b921 -> b1863
node session_CopyInto_-7207 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b921 -> b1863
node session_CopyInto_-7208 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b921 -> b1863
node session_CopyInto_-7209 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b921 -> b1863
node dispatch2054 micro raw43 b1863 -> b1863
node dispatch2055 micro raw44 b1863 -> b1863
node dispatch2056 micro raw45 b1863 -> b1863
node session_CopyInto_-7210 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2057 micro raw35 b1862 b1587 b1588 b1861 -> b1862
node dispatch2058 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7211 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b921
node session_CopyInto_-7212 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b921
node session_VecOp_-7213 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7217 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b940 -> b1859
node dispatch2059 micro raw31 b1860 b1591 b1592 b1859 -> b1860
node session_CopyInto_-7218 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7219 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b939 -> b1863
node session_CopyInto_-7220 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b939 -> b1863
node session_CopyInto_-7221 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b939 -> b1863
node session_CopyInto_-7222 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b939 -> b1863
node dispatch2060 micro raw43 b1863 -> b1863
node dispatch2061 micro raw44 b1863 -> b1863
node dispatch2062 micro raw45 b1863 -> b1863
node session_CopyInto_-7223 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2063 micro raw35 b1862 b1595 b1596 b1861 -> b1862
node dispatch2064 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7224 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b939
node session_CopyInto_-7225 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b939
node session_VecOp_-7226 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7227 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b958 -> b1859
node dispatch2065 micro raw31 b1860 b1599 b1600 b1859 -> b1860
node session_CopyInto_-7228 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7229 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b957 -> b1863
node session_CopyInto_-7230 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b957 -> b1863
node session_CopyInto_-7231 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b957 -> b1863
node session_CopyInto_-7232 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b957 -> b1863
node dispatch2066 micro raw43 b1863 -> b1863
node dispatch2067 micro raw44 b1863 -> b1863
node dispatch2068 micro raw45 b1863 -> b1863
node session_CopyInto_-7233 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2069 micro raw35 b1862 b1603 b1604 b1861 -> b1862
node dispatch2070 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7234 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b957
node session_CopyInto_-7235 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b957
node session_VecOp_-7236 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7237 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b976 -> b1859
node dispatch2071 micro raw31 b1860 b1615 b1616 b1859 -> b1860
node session_CopyInto_-7238 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7239 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b975 -> b1863
node session_CopyInto_-7240 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b975 -> b1863
node session_CopyInto_-7241 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b975 -> b1863
node session_CopyInto_-7242 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b975 -> b1863
node dispatch2072 micro raw43 b1863 -> b1863
node dispatch2073 micro raw44 b1863 -> b1863
node dispatch2074 micro raw45 b1863 -> b1863
node session_CopyInto_-7243 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2075 micro raw35 b1862 b1619 b1620 b1861 -> b1862
node dispatch2076 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7244 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b975
node session_CopyInto_-7245 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b975
node session_VecOp_-7246 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7247 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b994 -> b1859
node dispatch2077 micro raw31 b1860 b1623 b1624 b1859 -> b1860
node session_CopyInto_-7248 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7249 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b993 -> b1863
node session_CopyInto_-7250 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b993 -> b1863
node session_CopyInto_-7251 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b993 -> b1863
node session_CopyInto_-7252 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b993 -> b1863
node dispatch2078 micro raw43 b1863 -> b1863
node dispatch2079 micro raw44 b1863 -> b1863
node dispatch2080 micro raw45 b1863 -> b1863
node session_CopyInto_-7253 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2081 micro raw35 b1862 b1627 b1628 b1861 -> b1862
node dispatch2082 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7254 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b993
node session_CopyInto_-7255 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b993
node session_VecOp_-7256 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7260 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1012 -> b1859
node dispatch2083 micro raw31 b1860 b1631 b1632 b1859 -> b1860
node session_CopyInto_-7261 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7262 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1011 -> b1863
node session_CopyInto_-7263 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1011 -> b1863
node session_CopyInto_-7264 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1011 -> b1863
node session_CopyInto_-7265 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1011 -> b1863
node dispatch2084 micro raw43 b1863 -> b1863
node dispatch2085 micro raw44 b1863 -> b1863
node dispatch2086 micro raw45 b1863 -> b1863
node session_CopyInto_-7266 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2087 micro raw35 b1862 b1635 b1636 b1861 -> b1862
node dispatch2088 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7267 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1011
node session_CopyInto_-7268 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1011
node session_VecOp_-7269 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7270 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1030 -> b1859
node dispatch2089 micro raw31 b1860 b1639 b1640 b1859 -> b1860
node session_CopyInto_-7271 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7272 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1029 -> b1863
node session_CopyInto_-7273 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1029 -> b1863
node session_CopyInto_-7274 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1029 -> b1863
node session_CopyInto_-7275 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1029 -> b1863
node dispatch2090 micro raw43 b1863 -> b1863
node dispatch2091 micro raw44 b1863 -> b1863
node dispatch2092 micro raw45 b1863 -> b1863
node session_CopyInto_-7276 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2093 micro raw35 b1862 b1643 b1644 b1861 -> b1862
node dispatch2094 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7277 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1029
node session_CopyInto_-7278 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1029
node session_VecOp_-7279 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7280 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1048 -> b1859
node dispatch2095 micro raw31 b1860 b1647 b1648 b1859 -> b1860
node session_CopyInto_-7281 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7282 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1047 -> b1863
node session_CopyInto_-7283 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1047 -> b1863
node session_CopyInto_-7284 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1047 -> b1863
node session_CopyInto_-7285 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1047 -> b1863
node dispatch2096 micro raw43 b1863 -> b1863
node dispatch2097 micro raw44 b1863 -> b1863
node dispatch2098 micro raw45 b1863 -> b1863
node session_CopyInto_-7286 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2099 micro raw35 b1862 b1651 b1652 b1861 -> b1862
node dispatch2100 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7287 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1047
node session_CopyInto_-7288 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1047
node session_VecOp_-7289 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7290 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1066 -> b1859
node dispatch2101 micro raw31 b1860 b1655 b1656 b1859 -> b1860
node session_CopyInto_-7291 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7292 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1065 -> b1863
node session_CopyInto_-7293 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1065 -> b1863
node session_CopyInto_-7294 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1065 -> b1863
node session_CopyInto_-7295 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1065 -> b1863
node dispatch2102 micro raw43 b1863 -> b1863
node dispatch2103 micro raw44 b1863 -> b1863
node dispatch2104 micro raw45 b1863 -> b1863
node session_CopyInto_-7296 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2105 micro raw35 b1862 b1659 b1660 b1861 -> b1862
node dispatch2106 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7297 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1065
node session_CopyInto_-7298 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1065
node dispatch2107 micro raw47 b1873 b1070 b1859 -> b1873
node dispatch2108 micro raw48 b1873 b1071 -> b1873
node session_CopyInto_-7300 session sessionCopyInto:srcOff=0,dstOff=0,n=98304 b1873 -> b1870
node session_CopyInto_-7301 session sessionCopyInto:srcOff=16384,dstOff=329728,n=4096 b1074 -> b1870
node session_CopyInto_-7302 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b1074 -> b1878
node dispatch2109 micro raw5 b1870_1335296_2129920 b1878 -> b1870_1335296_2129920
node dispatch2110 micro raw49 b1870 -> b1870
node dispatch2111 micro raw50 b1870 -> b1870
node session_CopyInto_-7303 session sessionCopyInto:srcOff=163840,dstOff=0,n=32768 b1870 -> b1874
node dispatch2112 micro raw51 b1875 b1072 b1874 -> b1875
node dispatch2113 micro raw52 b1875 b1073 -> b1875
node session_CopyInto_-7305 session sessionCopyInto:srcOff=333824,dstOff=0,n=131072 b1870 -> b1878
node dispatch2114 micro raw5 b1074_81920_524288 b1878 -> b1074_81920_524288
node session_CopyInto_-7306 session sessionCopyInto:srcOff=595968,dstOff=16384,n=4096 b1870 -> b1074
node dispatch2115 micro raw53 b1876 b1862 b1875 -> b1876
node dispatch2116 micro raw54 b1877 b1080 b1876 -> b1877
node dispatch2117 micro raw52 b1877 b1081 -> b1877
node dispatch2118 micro raw55 b1862 b1877 b1875 -> b1862
node session_VecOp_-7308 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7312 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1101 -> b1859
node dispatch2119 micro raw31 b1860 b1663 b1664 b1859 -> b1860
node session_CopyInto_-7313 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7314 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1100 -> b1863
node session_CopyInto_-7315 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1100 -> b1863
node session_CopyInto_-7316 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1100 -> b1863
node session_CopyInto_-7317 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1100 -> b1863
node dispatch2120 micro raw43 b1863 -> b1863
node dispatch2121 micro raw44 b1863 -> b1863
node dispatch2122 micro raw45 b1863 -> b1863
node session_CopyInto_-7318 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2123 micro raw35 b1862 b1667 b1668 b1861 -> b1862
node dispatch2124 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7319 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1100
node session_CopyInto_-7320 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1100
node session_VecOp_-7321 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7322 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1119 -> b1859
node dispatch2125 micro raw31 b1860 b1671 b1672 b1859 -> b1860
node session_CopyInto_-7323 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7324 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1118 -> b1863
node session_CopyInto_-7325 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1118 -> b1863
node session_CopyInto_-7326 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1118 -> b1863
node session_CopyInto_-7327 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1118 -> b1863
node dispatch2126 micro raw43 b1863 -> b1863
node dispatch2127 micro raw44 b1863 -> b1863
node dispatch2128 micro raw45 b1863 -> b1863
node session_CopyInto_-7328 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2129 micro raw35 b1862 b1675 b1676 b1861 -> b1862
node dispatch2130 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7329 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1118
node session_CopyInto_-7330 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1118
node session_VecOp_-7331 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7332 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1137 -> b1859
node dispatch2131 micro raw31 b1860 b1679 b1680 b1859 -> b1860
node session_CopyInto_-7333 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7334 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1136 -> b1863
node session_CopyInto_-7335 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1136 -> b1863
node session_CopyInto_-7336 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1136 -> b1863
node session_CopyInto_-7337 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1136 -> b1863
node dispatch2132 micro raw43 b1863 -> b1863
node dispatch2133 micro raw44 b1863 -> b1863
node dispatch2134 micro raw45 b1863 -> b1863
node session_CopyInto_-7338 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2135 micro raw35 b1862 b1683 b1684 b1861 -> b1862
node dispatch2136 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7339 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1136
node session_CopyInto_-7340 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1136
node session_VecOp_-7341 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7342 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1155 -> b1859
node dispatch2137 micro raw31 b1860 b1687 b1688 b1859 -> b1860
node session_CopyInto_-7343 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7344 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1154 -> b1863
node session_CopyInto_-7345 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1154 -> b1863
node session_CopyInto_-7346 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1154 -> b1863
node session_CopyInto_-7347 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1154 -> b1863
node dispatch2138 micro raw43 b1863 -> b1863
node dispatch2139 micro raw44 b1863 -> b1863
node dispatch2140 micro raw45 b1863 -> b1863
node session_CopyInto_-7348 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2141 micro raw35 b1862 b1691 b1692 b1861 -> b1862
node dispatch2142 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7349 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1154
node session_CopyInto_-7350 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1154
node session_VecOp_-7351 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7355 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1173 -> b1859
node dispatch2143 micro raw31 b1860 b1703 b1704 b1859 -> b1860
node session_CopyInto_-7356 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7357 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1172 -> b1863
node session_CopyInto_-7358 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1172 -> b1863
node session_CopyInto_-7359 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1172 -> b1863
node session_CopyInto_-7360 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1172 -> b1863
node dispatch2144 micro raw43 b1863 -> b1863
node dispatch2145 micro raw44 b1863 -> b1863
node dispatch2146 micro raw45 b1863 -> b1863
node session_CopyInto_-7361 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2147 micro raw35 b1862 b1707 b1708 b1861 -> b1862
node dispatch2148 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7362 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1172
node session_CopyInto_-7363 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1172
node session_VecOp_-7364 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7365 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1191 -> b1859
node dispatch2149 micro raw31 b1860 b1711 b1712 b1859 -> b1860
node session_CopyInto_-7366 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7367 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1190 -> b1863
node session_CopyInto_-7368 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1190 -> b1863
node session_CopyInto_-7369 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1190 -> b1863
node session_CopyInto_-7370 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1190 -> b1863
node dispatch2150 micro raw43 b1863 -> b1863
node dispatch2151 micro raw44 b1863 -> b1863
node dispatch2152 micro raw45 b1863 -> b1863
node session_CopyInto_-7371 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2153 micro raw35 b1862 b1715 b1716 b1861 -> b1862
node dispatch2154 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7372 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1190
node session_CopyInto_-7373 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1190
node session_VecOp_-7374 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7375 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1209 -> b1859
node dispatch2155 micro raw31 b1860 b1719 b1720 b1859 -> b1860
node session_CopyInto_-7376 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7377 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1208 -> b1863
node session_CopyInto_-7378 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1208 -> b1863
node session_CopyInto_-7379 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1208 -> b1863
node session_CopyInto_-7380 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1208 -> b1863
node dispatch2156 micro raw43 b1863 -> b1863
node dispatch2157 micro raw44 b1863 -> b1863
node dispatch2158 micro raw45 b1863 -> b1863
node session_CopyInto_-7381 session sessionCopyInto:srcOff=2666944,dstOff=0,n=65536 b1863 -> b1861
node dispatch2159 micro raw35 b1862 b1723 b1724 b1861 -> b1862
node dispatch2160 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7382 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1208
node session_CopyInto_-7383 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1208
node session_VecOp_-7384 session sessionVecOp:op=add,n=32768 b1858 b1862 -> b1858
node session_RmsnormEnc_-7385 session sessionRmsnormEnc:rows=8,cols=4096,eps=h3727c5ac b1858 b1227 -> b1859
node dispatch2161 micro raw31 b1860 b1727 b1728 b1859 -> b1860
node session_CopyInto_-7386 session sessionCopyInto:srcOff=0,dstOff=0,n=148480 b1860 -> b1863
node session_CopyInto_-7387 session sessionCopyInto:srcOff=45696,dstOff=148480,n=51200 b1226 -> b1863
node session_CopyInto_-7388 session sessionCopyInto:srcOff=37120,dstOff=199680,n=8576 b1226 -> b1863
node session_CopyInto_-7389 session sessionCopyInto:srcOff=127616,dstOff=373120,n=1048576 b1226 -> b1863
node session_CopyInto_-7390 session sessionCopyInto:srcOff=96896,dstOff=2732480,n=30720 b1226 -> b1863
node dispatch2162 micro raw43 b1863 -> b1863
node dispatch2163 micro raw44 b1863 -> b1863
node dispatch2164 micro raw45 b1863 -> b1863
node dispatch2166 micro raw46 b1864 b1860 -> b1864
node session_CopyInto_-7392 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1864 -> b1226
node session_CopyInto_-7393 session sessionCopyInto:srcOff=1421696,dstOff=127616,n=1048576 b1863 -> b1226
phase chunk6 rows=5
node embedding_6 ext fir.embedding token backbone_embeddings_weight_emb -> b1891
node session_CopyInto_-7438 session sessionCopyInto:srcOff=0,dstOff=0,n=20480 b1891 -> b1882
node session_RmsnormEnc_-7465 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b24 -> b1883
node dispatch2173 micro raw56 b1884 b1247 b1248 b1883 -> b1884
node dispatch2174 micro raw57 b1884_296960_296960 b1247 b1248 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7467 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7468 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b23 -> b1887
node session_CopyInto_-7469 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b23 -> b1887
node session_CopyInto_-7470 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b23 -> b1887
node session_CopyInto_-7471 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b23 -> b1887
node dispatch2175 micro raw58 b1887 -> b1887
node dispatch2176 micro raw59 b1887 -> b1887
node dispatch2177 micro raw60 b1887 -> b1887
node session_CopyInto_-7472 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2178 micro raw61 b1886 b1251 b1252 b1885 -> b1886
node dispatch2179 micro raw62 b1886_65536_65536 b1251 b1252 b1885_131072_131072 -> b1886_65536_65536
node dispatch2180 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7474 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b23
node session_CopyInto_-7475 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b23
node session_VecOp_-7476 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7477 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b42 -> b1883
node dispatch2181 micro raw56 b1884 b1255 b1256 b1883 -> b1884
node dispatch2182 micro raw57 b1884_296960_296960 b1255 b1256 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7479 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7480 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b41 -> b1887
node session_CopyInto_-7481 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b41 -> b1887
node session_CopyInto_-7482 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b41 -> b1887
node session_CopyInto_-7483 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b41 -> b1887
node dispatch2183 micro raw58 b1887 -> b1887
node dispatch2184 micro raw59 b1887 -> b1887
node dispatch2185 micro raw60 b1887 -> b1887
node session_CopyInto_-7484 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2186 micro raw61 b1886 b1259 b1260 b1885 -> b1886
node dispatch2187 micro raw62 b1886_65536_65536 b1259 b1260 b1885_131072_131072 -> b1886_65536_65536
node dispatch2188 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7486 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b41
node session_CopyInto_-7487 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b41
node session_VecOp_-7488 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7489 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b60 -> b1883
node dispatch2189 micro raw56 b1884 b1343 b1344 b1883 -> b1884
node dispatch2190 micro raw57 b1884_296960_296960 b1343 b1344 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7491 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7492 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b59 -> b1887
node session_CopyInto_-7493 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b59 -> b1887
node session_CopyInto_-7494 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b59 -> b1887
node session_CopyInto_-7495 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b59 -> b1887
node dispatch2191 micro raw58 b1887 -> b1887
node dispatch2192 micro raw59 b1887 -> b1887
node dispatch2193 micro raw60 b1887 -> b1887
node session_CopyInto_-7496 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2194 micro raw61 b1886 b1347 b1348 b1885 -> b1886
node dispatch2195 micro raw62 b1886_65536_65536 b1347 b1348 b1885_131072_131072 -> b1886_65536_65536
node dispatch2196 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7498 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b59
node session_CopyInto_-7499 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b59
node session_VecOp_-7500 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7501 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b78 -> b1883
node dispatch2197 micro raw56 b1884 b1431 b1432 b1883 -> b1884
node dispatch2198 micro raw57 b1884_296960_296960 b1431 b1432 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7503 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7504 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b77 -> b1887
node session_CopyInto_-7505 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b77 -> b1887
node session_CopyInto_-7506 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b77 -> b1887
node session_CopyInto_-7507 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b77 -> b1887
node dispatch2199 micro raw58 b1887 -> b1887
node dispatch2200 micro raw59 b1887 -> b1887
node dispatch2201 micro raw60 b1887 -> b1887
node session_CopyInto_-7508 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2202 micro raw61 b1886 b1435 b1436 b1885 -> b1886
node dispatch2203 micro raw62 b1886_65536_65536 b1435 b1436 b1885_131072_131072 -> b1886_65536_65536
node dispatch2204 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7510 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b77
node session_CopyInto_-7511 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b77
node session_VecOp_-7512 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7516 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b96 -> b1883
node dispatch2205 micro raw56 b1884 b1519 b1520 b1883 -> b1884
node dispatch2206 micro raw57 b1884_296960_296960 b1519 b1520 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7518 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7519 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b95 -> b1887
node session_CopyInto_-7520 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b95 -> b1887
node session_CopyInto_-7521 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b95 -> b1887
node session_CopyInto_-7522 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b95 -> b1887
node dispatch2207 micro raw58 b1887 -> b1887
node dispatch2208 micro raw59 b1887 -> b1887
node dispatch2209 micro raw60 b1887 -> b1887
node session_CopyInto_-7523 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2210 micro raw61 b1886 b1523 b1524 b1885 -> b1886
node dispatch2211 micro raw62 b1886_65536_65536 b1523 b1524 b1885_131072_131072 -> b1886_65536_65536
node dispatch2212 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7525 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b95
node session_CopyInto_-7526 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b95
node session_VecOp_-7527 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7528 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b114 -> b1883
node dispatch2213 micro raw56 b1884 b1607 b1608 b1883 -> b1884
node dispatch2214 micro raw57 b1884_296960_296960 b1607 b1608 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7530 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7531 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b113 -> b1887
node session_CopyInto_-7532 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b113 -> b1887
node session_CopyInto_-7533 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b113 -> b1887
node session_CopyInto_-7534 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b113 -> b1887
node dispatch2215 micro raw58 b1887 -> b1887
node dispatch2216 micro raw59 b1887 -> b1887
node dispatch2217 micro raw60 b1887 -> b1887
node session_CopyInto_-7535 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2218 micro raw61 b1886 b1611 b1612 b1885 -> b1886
node dispatch2219 micro raw62 b1886_65536_65536 b1611 b1612 b1885_131072_131072 -> b1886_65536_65536
node dispatch2220 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7537 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b113
node session_CopyInto_-7538 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b113
node session_VecOp_-7539 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7540 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b132 -> b1883
node dispatch2221 micro raw56 b1884 b1695 b1696 b1883 -> b1884
node dispatch2222 micro raw57 b1884_296960_296960 b1695 b1696 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7542 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7543 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b131 -> b1887
node session_CopyInto_-7544 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b131 -> b1887
node session_CopyInto_-7545 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b131 -> b1887
node session_CopyInto_-7546 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b131 -> b1887
node dispatch2223 micro raw58 b1887 -> b1887
node dispatch2224 micro raw59 b1887 -> b1887
node dispatch2225 micro raw60 b1887 -> b1887
node session_CopyInto_-7547 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2226 micro raw61 b1886 b1699 b1700 b1885 -> b1886
node dispatch2227 micro raw62 b1886_65536_65536 b1699 b1700 b1885_131072_131072 -> b1886_65536_65536
node dispatch2228 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7549 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b131
node session_CopyInto_-7550 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b131
node session_VecOp_-7551 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7552 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b150 -> b1883
node dispatch2229 micro raw56 b1884 b1735 b1736 b1883 -> b1884
node dispatch2230 micro raw57 b1884_296960_296960 b1735 b1736 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7554 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7555 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b149 -> b1887
node session_CopyInto_-7556 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b149 -> b1887
node session_CopyInto_-7557 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b149 -> b1887
node session_CopyInto_-7558 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b149 -> b1887
node dispatch2231 micro raw58 b1887 -> b1887
node dispatch2232 micro raw59 b1887 -> b1887
node dispatch2233 micro raw60 b1887 -> b1887
node session_CopyInto_-7559 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2234 micro raw61 b1886 b1739 b1740 b1885 -> b1886
node dispatch2235 micro raw62 b1886_65536_65536 b1739 b1740 b1885_131072_131072 -> b1886_65536_65536
node dispatch2236 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7561 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b149
node session_CopyInto_-7562 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b149
node dispatch2237 micro raw64 b1897 b154 b1883 -> b1897
node dispatch2238 micro raw65 b1897 b155 -> b1897
node session_CopyInto_-7563 session sessionCopyInto:srcOff=0,dstOff=0,n=61440 b1897 -> b1894
node session_CopyInto_-7564 session sessionCopyInto:srcOff=16384,dstOff=206080,n=4096 b158 -> b1894
node session_CopyInto_-7565 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b158 -> b1902
node dispatch2239 micro raw5 b1894_840704_2129920 b1902 -> b1894_840704_2129920
node dispatch2240 micro raw66 b1894 -> b1894
node dispatch2241 micro raw67 b1894 -> b1894
node session_CopyInto_-7566 session sessionCopyInto:srcOff=102400,dstOff=0,n=20480 b1894 -> b1898
node dispatch2242 micro raw68 b1899 b156 b1898 -> b1899
node dispatch2243 micro raw69 b1899 b157 -> b1899
node session_CopyInto_-7567 session sessionCopyInto:srcOff=210176,dstOff=0,n=131072 b1894 -> b1902
node dispatch2244 micro raw5 b158_81920_524288 b1902 -> b158_81920_524288
node session_CopyInto_-7568 session sessionCopyInto:srcOff=472320,dstOff=16384,n=4096 b1894 -> b158
node dispatch2245 micro raw70 b1900 b1886 b1899 -> b1900
node dispatch2246 micro raw71 b1901 b165 b1900 -> b1901
node dispatch2247 micro raw69 b1901 b166 -> b1901
node dispatch2248 micro raw72 b1886 b1901 b1899 -> b1886
node session_VecOp_-7569 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7573 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b186 -> b1883
node dispatch2249 micro raw56 b1884 b1743 b1744 b1883 -> b1884
node dispatch2250 micro raw57 b1884_296960_296960 b1743 b1744 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7575 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7576 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b185 -> b1887
node session_CopyInto_-7577 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b185 -> b1887
node session_CopyInto_-7578 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b185 -> b1887
node session_CopyInto_-7579 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b185 -> b1887
node dispatch2251 micro raw58 b1887 -> b1887
node dispatch2252 micro raw59 b1887 -> b1887
node dispatch2253 micro raw60 b1887 -> b1887
node session_CopyInto_-7580 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2254 micro raw61 b1886 b1747 b1748 b1885 -> b1886
node dispatch2255 micro raw62 b1886_65536_65536 b1747 b1748 b1885_131072_131072 -> b1886_65536_65536
node dispatch2256 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7582 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b185
node session_CopyInto_-7583 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b185
node session_VecOp_-7584 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7585 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b204 -> b1883
node dispatch2257 micro raw56 b1884 b1751 b1752 b1883 -> b1884
node dispatch2258 micro raw57 b1884_296960_296960 b1751 b1752 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7587 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7588 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b203 -> b1887
node session_CopyInto_-7589 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b203 -> b1887
node session_CopyInto_-7590 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b203 -> b1887
node session_CopyInto_-7591 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b203 -> b1887
node dispatch2259 micro raw58 b1887 -> b1887
node dispatch2260 micro raw59 b1887 -> b1887
node dispatch2261 micro raw60 b1887 -> b1887
node session_CopyInto_-7592 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2262 micro raw61 b1886 b1755 b1756 b1885 -> b1886
node dispatch2263 micro raw62 b1886_65536_65536 b1755 b1756 b1885_131072_131072 -> b1886_65536_65536
node dispatch2264 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7594 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b203
node session_CopyInto_-7595 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b203
node session_VecOp_-7596 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7597 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b222 -> b1883
node dispatch2265 micro raw56 b1884 b1263 b1264 b1883 -> b1884
node dispatch2266 micro raw57 b1884_296960_296960 b1263 b1264 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7599 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7600 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b221 -> b1887
node session_CopyInto_-7601 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b221 -> b1887
node session_CopyInto_-7602 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b221 -> b1887
node session_CopyInto_-7603 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b221 -> b1887
node dispatch2267 micro raw58 b1887 -> b1887
node dispatch2268 micro raw59 b1887 -> b1887
node dispatch2269 micro raw60 b1887 -> b1887
node session_CopyInto_-7604 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2270 micro raw61 b1886 b1267 b1268 b1885 -> b1886
node dispatch2271 micro raw62 b1886_65536_65536 b1267 b1268 b1885_131072_131072 -> b1886_65536_65536
node dispatch2272 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7606 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b221
node session_CopyInto_-7607 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b221
node session_VecOp_-7608 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7609 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b240 -> b1883
node dispatch2273 micro raw56 b1884 b1271 b1272 b1883 -> b1884
node dispatch2274 micro raw57 b1884_296960_296960 b1271 b1272 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7611 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7612 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b239 -> b1887
node session_CopyInto_-7613 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b239 -> b1887
node session_CopyInto_-7614 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b239 -> b1887
node session_CopyInto_-7615 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b239 -> b1887
node dispatch2275 micro raw58 b1887 -> b1887
node dispatch2276 micro raw59 b1887 -> b1887
node dispatch2277 micro raw60 b1887 -> b1887
node session_CopyInto_-7616 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2278 micro raw61 b1886 b1275 b1276 b1885 -> b1886
node dispatch2279 micro raw62 b1886_65536_65536 b1275 b1276 b1885_131072_131072 -> b1886_65536_65536
node dispatch2280 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7618 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b239
node session_CopyInto_-7619 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b239
node session_VecOp_-7620 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7624 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b258 -> b1883
node dispatch2281 micro raw56 b1884 b1279 b1280 b1883 -> b1884
node dispatch2282 micro raw57 b1884_296960_296960 b1279 b1280 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7626 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7627 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b257 -> b1887
node session_CopyInto_-7628 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b257 -> b1887
node session_CopyInto_-7629 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b257 -> b1887
node session_CopyInto_-7630 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b257 -> b1887
node dispatch2283 micro raw58 b1887 -> b1887
node dispatch2284 micro raw59 b1887 -> b1887
node dispatch2285 micro raw60 b1887 -> b1887
node session_CopyInto_-7631 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2286 micro raw61 b1886 b1283 b1284 b1885 -> b1886
node dispatch2287 micro raw62 b1886_65536_65536 b1283 b1284 b1885_131072_131072 -> b1886_65536_65536
node dispatch2288 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7633 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b257
node session_CopyInto_-7634 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b257
node session_VecOp_-7635 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7636 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b276 -> b1883
node dispatch2289 micro raw56 b1884 b1287 b1288 b1883 -> b1884
node dispatch2290 micro raw57 b1884_296960_296960 b1287 b1288 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7638 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7639 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b275 -> b1887
node session_CopyInto_-7640 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b275 -> b1887
node session_CopyInto_-7641 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b275 -> b1887
node session_CopyInto_-7642 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b275 -> b1887
node dispatch2291 micro raw58 b1887 -> b1887
node dispatch2292 micro raw59 b1887 -> b1887
node dispatch2293 micro raw60 b1887 -> b1887
node session_CopyInto_-7643 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2294 micro raw61 b1886 b1291 b1292 b1885 -> b1886
node dispatch2295 micro raw62 b1886_65536_65536 b1291 b1292 b1885_131072_131072 -> b1886_65536_65536
node dispatch2296 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7645 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b275
node session_CopyInto_-7646 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b275
node session_VecOp_-7647 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7648 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b294 -> b1883
node dispatch2297 micro raw56 b1884 b1295 b1296 b1883 -> b1884
node dispatch2298 micro raw57 b1884_296960_296960 b1295 b1296 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7650 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7651 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b293 -> b1887
node session_CopyInto_-7652 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b293 -> b1887
node session_CopyInto_-7653 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b293 -> b1887
node session_CopyInto_-7654 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b293 -> b1887
node dispatch2299 micro raw58 b1887 -> b1887
node dispatch2300 micro raw59 b1887 -> b1887
node dispatch2301 micro raw60 b1887 -> b1887
node session_CopyInto_-7655 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2302 micro raw61 b1886 b1299 b1300 b1885 -> b1886
node dispatch2303 micro raw62 b1886_65536_65536 b1299 b1300 b1885_131072_131072 -> b1886_65536_65536
node dispatch2304 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7657 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b293
node session_CopyInto_-7658 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b293
node session_VecOp_-7659 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7660 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b312 -> b1883
node dispatch2305 micro raw56 b1884 b1303 b1304 b1883 -> b1884
node dispatch2306 micro raw57 b1884_296960_296960 b1303 b1304 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7662 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7663 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b311 -> b1887
node session_CopyInto_-7664 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b311 -> b1887
node session_CopyInto_-7665 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b311 -> b1887
node session_CopyInto_-7666 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b311 -> b1887
node dispatch2307 micro raw58 b1887 -> b1887
node dispatch2308 micro raw59 b1887 -> b1887
node dispatch2309 micro raw60 b1887 -> b1887
node session_CopyInto_-7667 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2310 micro raw61 b1886 b1307 b1308 b1885 -> b1886
node dispatch2311 micro raw62 b1886_65536_65536 b1307 b1308 b1885_131072_131072 -> b1886_65536_65536
node dispatch2312 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7669 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b311
node session_CopyInto_-7670 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b311
node session_VecOp_-7671 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7675 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b330 -> b1883
node dispatch2313 micro raw56 b1884 b1311 b1312 b1883 -> b1884
node dispatch2314 micro raw57 b1884_296960_296960 b1311 b1312 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7677 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7678 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b329 -> b1887
node session_CopyInto_-7679 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b329 -> b1887
node session_CopyInto_-7680 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b329 -> b1887
node session_CopyInto_-7681 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b329 -> b1887
node dispatch2315 micro raw58 b1887 -> b1887
node dispatch2316 micro raw59 b1887 -> b1887
node dispatch2317 micro raw60 b1887 -> b1887
node session_CopyInto_-7682 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2318 micro raw61 b1886 b1315 b1316 b1885 -> b1886
node dispatch2319 micro raw62 b1886_65536_65536 b1315 b1316 b1885_131072_131072 -> b1886_65536_65536
node dispatch2320 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7684 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b329
node session_CopyInto_-7685 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b329
node session_VecOp_-7686 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7687 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b348 -> b1883
node dispatch2321 micro raw56 b1884 b1319 b1320 b1883 -> b1884
node dispatch2322 micro raw57 b1884_296960_296960 b1319 b1320 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7689 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7690 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b347 -> b1887
node session_CopyInto_-7691 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b347 -> b1887
node session_CopyInto_-7692 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b347 -> b1887
node session_CopyInto_-7693 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b347 -> b1887
node dispatch2323 micro raw58 b1887 -> b1887
node dispatch2324 micro raw59 b1887 -> b1887
node dispatch2325 micro raw60 b1887 -> b1887
node session_CopyInto_-7694 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2326 micro raw61 b1886 b1323 b1324 b1885 -> b1886
node dispatch2327 micro raw62 b1886_65536_65536 b1323 b1324 b1885_131072_131072 -> b1886_65536_65536
node dispatch2328 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7696 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b347
node session_CopyInto_-7697 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b347
node session_VecOp_-7698 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7699 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b366 -> b1883
node dispatch2329 micro raw56 b1884 b1327 b1328 b1883 -> b1884
node dispatch2330 micro raw57 b1884_296960_296960 b1327 b1328 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7701 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7702 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b365 -> b1887
node session_CopyInto_-7703 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b365 -> b1887
node session_CopyInto_-7704 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b365 -> b1887
node session_CopyInto_-7705 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b365 -> b1887
node dispatch2331 micro raw58 b1887 -> b1887
node dispatch2332 micro raw59 b1887 -> b1887
node dispatch2333 micro raw60 b1887 -> b1887
node session_CopyInto_-7706 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2334 micro raw61 b1886 b1331 b1332 b1885 -> b1886
node dispatch2335 micro raw62 b1886_65536_65536 b1331 b1332 b1885_131072_131072 -> b1886_65536_65536
node dispatch2336 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7708 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b365
node session_CopyInto_-7709 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b365
node session_VecOp_-7710 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7711 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b384 -> b1883
node dispatch2337 micro raw56 b1884 b1335 b1336 b1883 -> b1884
node dispatch2338 micro raw57 b1884_296960_296960 b1335 b1336 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7713 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7714 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b383 -> b1887
node session_CopyInto_-7715 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b383 -> b1887
node session_CopyInto_-7716 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b383 -> b1887
node session_CopyInto_-7717 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b383 -> b1887
node dispatch2339 micro raw58 b1887 -> b1887
node dispatch2340 micro raw59 b1887 -> b1887
node dispatch2341 micro raw60 b1887 -> b1887
node session_CopyInto_-7718 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2342 micro raw61 b1886 b1339 b1340 b1885 -> b1886
node dispatch2343 micro raw62 b1886_65536_65536 b1339 b1340 b1885_131072_131072 -> b1886_65536_65536
node dispatch2344 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7720 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b383
node session_CopyInto_-7721 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b383
node session_VecOp_-7722 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7726 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b402 -> b1883
node dispatch2345 micro raw56 b1884 b1351 b1352 b1883 -> b1884
node dispatch2346 micro raw57 b1884_296960_296960 b1351 b1352 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7728 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7729 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b401 -> b1887
node session_CopyInto_-7730 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b401 -> b1887
node session_CopyInto_-7731 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b401 -> b1887
node session_CopyInto_-7732 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b401 -> b1887
node dispatch2347 micro raw58 b1887 -> b1887
node dispatch2348 micro raw59 b1887 -> b1887
node dispatch2349 micro raw60 b1887 -> b1887
node session_CopyInto_-7733 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2350 micro raw61 b1886 b1355 b1356 b1885 -> b1886
node dispatch2351 micro raw62 b1886_65536_65536 b1355 b1356 b1885_131072_131072 -> b1886_65536_65536
node dispatch2352 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7735 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b401
node session_CopyInto_-7736 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b401
node session_VecOp_-7737 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7738 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b420 -> b1883
node dispatch2353 micro raw56 b1884 b1359 b1360 b1883 -> b1884
node dispatch2354 micro raw57 b1884_296960_296960 b1359 b1360 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7740 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7741 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b419 -> b1887
node session_CopyInto_-7742 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b419 -> b1887
node session_CopyInto_-7743 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b419 -> b1887
node session_CopyInto_-7744 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b419 -> b1887
node dispatch2355 micro raw58 b1887 -> b1887
node dispatch2356 micro raw59 b1887 -> b1887
node dispatch2357 micro raw60 b1887 -> b1887
node session_CopyInto_-7745 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2358 micro raw61 b1886 b1363 b1364 b1885 -> b1886
node dispatch2359 micro raw62 b1886_65536_65536 b1363 b1364 b1885_131072_131072 -> b1886_65536_65536
node dispatch2360 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7747 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b419
node session_CopyInto_-7748 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b419
node session_VecOp_-7749 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7750 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b438 -> b1883
node dispatch2361 micro raw56 b1884 b1367 b1368 b1883 -> b1884
node dispatch2362 micro raw57 b1884_296960_296960 b1367 b1368 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7752 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7753 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b437 -> b1887
node session_CopyInto_-7754 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b437 -> b1887
node session_CopyInto_-7755 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b437 -> b1887
node session_CopyInto_-7756 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b437 -> b1887
node dispatch2363 micro raw58 b1887 -> b1887
node dispatch2364 micro raw59 b1887 -> b1887
node dispatch2365 micro raw60 b1887 -> b1887
node session_CopyInto_-7757 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2366 micro raw61 b1886 b1371 b1372 b1885 -> b1886
node dispatch2367 micro raw62 b1886_65536_65536 b1371 b1372 b1885_131072_131072 -> b1886_65536_65536
node dispatch2368 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7759 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b437
node session_CopyInto_-7760 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b437
node session_VecOp_-7761 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7762 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b456 -> b1883
node dispatch2369 micro raw56 b1884 b1375 b1376 b1883 -> b1884
node dispatch2370 micro raw57 b1884_296960_296960 b1375 b1376 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7764 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7765 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b455 -> b1887
node session_CopyInto_-7766 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b455 -> b1887
node session_CopyInto_-7767 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b455 -> b1887
node session_CopyInto_-7768 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b455 -> b1887
node dispatch2371 micro raw58 b1887 -> b1887
node dispatch2372 micro raw59 b1887 -> b1887
node dispatch2373 micro raw60 b1887 -> b1887
node session_CopyInto_-7769 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2374 micro raw61 b1886 b1379 b1380 b1885 -> b1886
node dispatch2375 micro raw62 b1886_65536_65536 b1379 b1380 b1885_131072_131072 -> b1886_65536_65536
node dispatch2376 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7771 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b455
node session_CopyInto_-7772 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b455
node dispatch2377 micro raw64 b1897 b460 b1883 -> b1897
node dispatch2378 micro raw65 b1897 b461 -> b1897
node session_CopyInto_-7773 session sessionCopyInto:srcOff=0,dstOff=0,n=61440 b1897 -> b1894
node session_CopyInto_-7774 session sessionCopyInto:srcOff=16384,dstOff=206080,n=4096 b464 -> b1894
node session_CopyInto_-7775 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b464 -> b1902
node dispatch2379 micro raw5 b1894_840704_2129920 b1902 -> b1894_840704_2129920
node dispatch2380 micro raw66 b1894 -> b1894
node dispatch2381 micro raw67 b1894 -> b1894
node session_CopyInto_-7776 session sessionCopyInto:srcOff=102400,dstOff=0,n=20480 b1894 -> b1898
node dispatch2382 micro raw68 b1899 b462 b1898 -> b1899
node dispatch2383 micro raw69 b1899 b463 -> b1899
node session_CopyInto_-7777 session sessionCopyInto:srcOff=210176,dstOff=0,n=131072 b1894 -> b1902
node dispatch2384 micro raw5 b464_81920_524288 b1902 -> b464_81920_524288
node session_CopyInto_-7778 session sessionCopyInto:srcOff=472320,dstOff=16384,n=4096 b1894 -> b464
node dispatch2385 micro raw70 b1900 b1886 b1899 -> b1900
node dispatch2386 micro raw71 b1901 b470 b1900 -> b1901
node dispatch2387 micro raw69 b1901 b471 -> b1901
node dispatch2388 micro raw72 b1886 b1901 b1899 -> b1886
node session_VecOp_-7779 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7783 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b491 -> b1883
node dispatch2389 micro raw56 b1884 b1383 b1384 b1883 -> b1884
node dispatch2390 micro raw57 b1884_296960_296960 b1383 b1384 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7785 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7786 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b490 -> b1887
node session_CopyInto_-7787 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b490 -> b1887
node session_CopyInto_-7788 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b490 -> b1887
node session_CopyInto_-7789 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b490 -> b1887
node dispatch2391 micro raw58 b1887 -> b1887
node dispatch2392 micro raw59 b1887 -> b1887
node dispatch2393 micro raw60 b1887 -> b1887
node session_CopyInto_-7790 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2394 micro raw61 b1886 b1387 b1388 b1885 -> b1886
node dispatch2395 micro raw62 b1886_65536_65536 b1387 b1388 b1885_131072_131072 -> b1886_65536_65536
node dispatch2396 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7792 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b490
node session_CopyInto_-7793 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b490
node session_VecOp_-7794 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7795 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b509 -> b1883
node dispatch2397 micro raw56 b1884 b1391 b1392 b1883 -> b1884
node dispatch2398 micro raw57 b1884_296960_296960 b1391 b1392 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7797 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7798 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b508 -> b1887
node session_CopyInto_-7799 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b508 -> b1887
node session_CopyInto_-7800 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b508 -> b1887
node session_CopyInto_-7801 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b508 -> b1887
node dispatch2399 micro raw58 b1887 -> b1887
node dispatch2400 micro raw59 b1887 -> b1887
node dispatch2401 micro raw60 b1887 -> b1887
node session_CopyInto_-7802 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2402 micro raw61 b1886 b1395 b1396 b1885 -> b1886
node dispatch2403 micro raw62 b1886_65536_65536 b1395 b1396 b1885_131072_131072 -> b1886_65536_65536
node dispatch2404 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7804 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b508
node session_CopyInto_-7805 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b508
node session_VecOp_-7806 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7807 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b527 -> b1883
node dispatch2405 micro raw56 b1884 b1399 b1400 b1883 -> b1884
node dispatch2406 micro raw57 b1884_296960_296960 b1399 b1400 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7809 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7810 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b526 -> b1887
node session_CopyInto_-7811 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b526 -> b1887
node session_CopyInto_-7812 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b526 -> b1887
node session_CopyInto_-7813 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b526 -> b1887
node dispatch2407 micro raw58 b1887 -> b1887
node dispatch2408 micro raw59 b1887 -> b1887
node dispatch2409 micro raw60 b1887 -> b1887
node session_CopyInto_-7814 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2410 micro raw61 b1886 b1403 b1404 b1885 -> b1886
node dispatch2411 micro raw62 b1886_65536_65536 b1403 b1404 b1885_131072_131072 -> b1886_65536_65536
node dispatch2412 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7816 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b526
node session_CopyInto_-7817 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b526
node session_VecOp_-7818 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7819 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b545 -> b1883
node dispatch2413 micro raw56 b1884 b1407 b1408 b1883 -> b1884
node dispatch2414 micro raw57 b1884_296960_296960 b1407 b1408 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7821 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7822 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b544 -> b1887
node session_CopyInto_-7823 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b544 -> b1887
node session_CopyInto_-7824 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b544 -> b1887
node session_CopyInto_-7825 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b544 -> b1887
node dispatch2415 micro raw58 b1887 -> b1887
node dispatch2416 micro raw59 b1887 -> b1887
node dispatch2417 micro raw60 b1887 -> b1887
node session_CopyInto_-7826 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2418 micro raw61 b1886 b1411 b1412 b1885 -> b1886
node dispatch2419 micro raw62 b1886_65536_65536 b1411 b1412 b1885_131072_131072 -> b1886_65536_65536
node dispatch2420 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7828 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b544
node session_CopyInto_-7829 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b544
node session_VecOp_-7830 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7834 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b563 -> b1883
node dispatch2421 micro raw56 b1884 b1415 b1416 b1883 -> b1884
node dispatch2422 micro raw57 b1884_296960_296960 b1415 b1416 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7836 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7837 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b562 -> b1887
node session_CopyInto_-7838 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b562 -> b1887
node session_CopyInto_-7839 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b562 -> b1887
node session_CopyInto_-7840 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b562 -> b1887
node dispatch2423 micro raw58 b1887 -> b1887
node dispatch2424 micro raw59 b1887 -> b1887
node dispatch2425 micro raw60 b1887 -> b1887
node session_CopyInto_-7841 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2426 micro raw61 b1886 b1419 b1420 b1885 -> b1886
node dispatch2427 micro raw62 b1886_65536_65536 b1419 b1420 b1885_131072_131072 -> b1886_65536_65536
node dispatch2428 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7843 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b562
node session_CopyInto_-7844 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b562
node session_VecOp_-7845 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7846 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b581 -> b1883
node dispatch2429 micro raw56 b1884 b1423 b1424 b1883 -> b1884
node dispatch2430 micro raw57 b1884_296960_296960 b1423 b1424 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7848 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7849 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b580 -> b1887
node session_CopyInto_-7850 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b580 -> b1887
node session_CopyInto_-7851 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b580 -> b1887
node session_CopyInto_-7852 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b580 -> b1887
node dispatch2431 micro raw58 b1887 -> b1887
node dispatch2432 micro raw59 b1887 -> b1887
node dispatch2433 micro raw60 b1887 -> b1887
node session_CopyInto_-7853 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2434 micro raw61 b1886 b1427 b1428 b1885 -> b1886
node dispatch2435 micro raw62 b1886_65536_65536 b1427 b1428 b1885_131072_131072 -> b1886_65536_65536
node dispatch2436 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7855 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b580
node session_CopyInto_-7856 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b580
node session_VecOp_-7857 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7858 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b599 -> b1883
node dispatch2437 micro raw56 b1884 b1439 b1440 b1883 -> b1884
node dispatch2438 micro raw57 b1884_296960_296960 b1439 b1440 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7860 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7861 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b598 -> b1887
node session_CopyInto_-7862 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b598 -> b1887
node session_CopyInto_-7863 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b598 -> b1887
node session_CopyInto_-7864 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b598 -> b1887
node dispatch2439 micro raw58 b1887 -> b1887
node dispatch2440 micro raw59 b1887 -> b1887
node dispatch2441 micro raw60 b1887 -> b1887
node session_CopyInto_-7865 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2442 micro raw61 b1886 b1443 b1444 b1885 -> b1886
node dispatch2443 micro raw62 b1886_65536_65536 b1443 b1444 b1885_131072_131072 -> b1886_65536_65536
node dispatch2444 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7867 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b598
node session_CopyInto_-7868 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b598
node session_VecOp_-7869 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7870 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b617 -> b1883
node dispatch2445 micro raw56 b1884 b1447 b1448 b1883 -> b1884
node dispatch2446 micro raw57 b1884_296960_296960 b1447 b1448 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7872 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7873 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b616 -> b1887
node session_CopyInto_-7874 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b616 -> b1887
node session_CopyInto_-7875 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b616 -> b1887
node session_CopyInto_-7876 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b616 -> b1887
node dispatch2447 micro raw58 b1887 -> b1887
node dispatch2448 micro raw59 b1887 -> b1887
node dispatch2449 micro raw60 b1887 -> b1887
node session_CopyInto_-7877 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2450 micro raw61 b1886 b1451 b1452 b1885 -> b1886
node dispatch2451 micro raw62 b1886_65536_65536 b1451 b1452 b1885_131072_131072 -> b1886_65536_65536
node dispatch2452 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7879 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b616
node session_CopyInto_-7880 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b616
node session_VecOp_-7881 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7885 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b635 -> b1883
node dispatch2453 micro raw56 b1884 b1455 b1456 b1883 -> b1884
node dispatch2454 micro raw57 b1884_296960_296960 b1455 b1456 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7887 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7888 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b634 -> b1887
node session_CopyInto_-7889 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b634 -> b1887
node session_CopyInto_-7890 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b634 -> b1887
node session_CopyInto_-7891 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b634 -> b1887
node dispatch2455 micro raw58 b1887 -> b1887
node dispatch2456 micro raw59 b1887 -> b1887
node dispatch2457 micro raw60 b1887 -> b1887
node session_CopyInto_-7892 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2458 micro raw61 b1886 b1459 b1460 b1885 -> b1886
node dispatch2459 micro raw62 b1886_65536_65536 b1459 b1460 b1885_131072_131072 -> b1886_65536_65536
node dispatch2460 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7894 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b634
node session_CopyInto_-7895 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b634
node session_VecOp_-7896 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7897 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b653 -> b1883
node dispatch2461 micro raw56 b1884 b1463 b1464 b1883 -> b1884
node dispatch2462 micro raw57 b1884_296960_296960 b1463 b1464 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7899 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7900 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b652 -> b1887
node session_CopyInto_-7901 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b652 -> b1887
node session_CopyInto_-7902 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b652 -> b1887
node session_CopyInto_-7903 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b652 -> b1887
node dispatch2463 micro raw58 b1887 -> b1887
node dispatch2464 micro raw59 b1887 -> b1887
node dispatch2465 micro raw60 b1887 -> b1887
node session_CopyInto_-7904 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2466 micro raw61 b1886 b1467 b1468 b1885 -> b1886
node dispatch2467 micro raw62 b1886_65536_65536 b1467 b1468 b1885_131072_131072 -> b1886_65536_65536
node dispatch2468 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7906 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b652
node session_CopyInto_-7907 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b652
node session_VecOp_-7908 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7909 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b671 -> b1883
node dispatch2469 micro raw56 b1884 b1471 b1472 b1883 -> b1884
node dispatch2470 micro raw57 b1884_296960_296960 b1471 b1472 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7911 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7912 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b670 -> b1887
node session_CopyInto_-7913 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b670 -> b1887
node session_CopyInto_-7914 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b670 -> b1887
node session_CopyInto_-7915 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b670 -> b1887
node dispatch2471 micro raw58 b1887 -> b1887
node dispatch2472 micro raw59 b1887 -> b1887
node dispatch2473 micro raw60 b1887 -> b1887
node session_CopyInto_-7916 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2474 micro raw61 b1886 b1475 b1476 b1885 -> b1886
node dispatch2475 micro raw62 b1886_65536_65536 b1475 b1476 b1885_131072_131072 -> b1886_65536_65536
node dispatch2476 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7918 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b670
node session_CopyInto_-7919 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b670
node session_VecOp_-7920 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7921 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b689 -> b1883
node dispatch2477 micro raw56 b1884 b1479 b1480 b1883 -> b1884
node dispatch2478 micro raw57 b1884_296960_296960 b1479 b1480 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7923 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7924 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b688 -> b1887
node session_CopyInto_-7925 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b688 -> b1887
node session_CopyInto_-7926 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b688 -> b1887
node session_CopyInto_-7927 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b688 -> b1887
node dispatch2479 micro raw58 b1887 -> b1887
node dispatch2480 micro raw59 b1887 -> b1887
node dispatch2481 micro raw60 b1887 -> b1887
node session_CopyInto_-7928 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2482 micro raw61 b1886 b1483 b1484 b1885 -> b1886
node dispatch2483 micro raw62 b1886_65536_65536 b1483 b1484 b1885_131072_131072 -> b1886_65536_65536
node dispatch2484 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7930 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b688
node session_CopyInto_-7931 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b688
node session_VecOp_-7932 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7936 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b707 -> b1883
node dispatch2485 micro raw56 b1884 b1487 b1488 b1883 -> b1884
node dispatch2486 micro raw57 b1884_296960_296960 b1487 b1488 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7938 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7939 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b706 -> b1887
node session_CopyInto_-7940 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b706 -> b1887
node session_CopyInto_-7941 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b706 -> b1887
node session_CopyInto_-7942 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b706 -> b1887
node dispatch2487 micro raw58 b1887 -> b1887
node dispatch2488 micro raw59 b1887 -> b1887
node dispatch2489 micro raw60 b1887 -> b1887
node session_CopyInto_-7943 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2490 micro raw61 b1886 b1491 b1492 b1885 -> b1886
node dispatch2491 micro raw62 b1886_65536_65536 b1491 b1492 b1885_131072_131072 -> b1886_65536_65536
node dispatch2492 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7945 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b706
node session_CopyInto_-7946 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b706
node session_VecOp_-7947 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7948 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b725 -> b1883
node dispatch2493 micro raw56 b1884 b1495 b1496 b1883 -> b1884
node dispatch2494 micro raw57 b1884_296960_296960 b1495 b1496 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7950 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7951 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b724 -> b1887
node session_CopyInto_-7952 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b724 -> b1887
node session_CopyInto_-7953 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b724 -> b1887
node session_CopyInto_-7954 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b724 -> b1887
node dispatch2495 micro raw58 b1887 -> b1887
node dispatch2496 micro raw59 b1887 -> b1887
node dispatch2497 micro raw60 b1887 -> b1887
node session_CopyInto_-7955 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2498 micro raw61 b1886 b1499 b1500 b1885 -> b1886
node dispatch2499 micro raw62 b1886_65536_65536 b1499 b1500 b1885_131072_131072 -> b1886_65536_65536
node dispatch2500 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7957 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b724
node session_CopyInto_-7958 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b724
node session_VecOp_-7959 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7960 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b743 -> b1883
node dispatch2501 micro raw56 b1884 b1503 b1504 b1883 -> b1884
node dispatch2502 micro raw57 b1884_296960_296960 b1503 b1504 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7962 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7963 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b742 -> b1887
node session_CopyInto_-7964 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b742 -> b1887
node session_CopyInto_-7965 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b742 -> b1887
node session_CopyInto_-7966 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b742 -> b1887
node dispatch2503 micro raw58 b1887 -> b1887
node dispatch2504 micro raw59 b1887 -> b1887
node dispatch2505 micro raw60 b1887 -> b1887
node session_CopyInto_-7967 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2506 micro raw61 b1886 b1507 b1508 b1885 -> b1886
node dispatch2507 micro raw62 b1886_65536_65536 b1507 b1508 b1885_131072_131072 -> b1886_65536_65536
node dispatch2508 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7969 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b742
node session_CopyInto_-7970 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b742
node session_VecOp_-7971 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7972 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b761 -> b1883
node dispatch2509 micro raw56 b1884 b1511 b1512 b1883 -> b1884
node dispatch2510 micro raw57 b1884_296960_296960 b1511 b1512 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7974 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7975 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b760 -> b1887
node session_CopyInto_-7976 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b760 -> b1887
node session_CopyInto_-7977 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b760 -> b1887
node session_CopyInto_-7978 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b760 -> b1887
node dispatch2511 micro raw58 b1887 -> b1887
node dispatch2512 micro raw59 b1887 -> b1887
node dispatch2513 micro raw60 b1887 -> b1887
node session_CopyInto_-7979 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2514 micro raw61 b1886 b1515 b1516 b1885 -> b1886
node dispatch2515 micro raw62 b1886_65536_65536 b1515 b1516 b1885_131072_131072 -> b1886_65536_65536
node dispatch2516 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-7981 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b760
node session_CopyInto_-7982 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b760
node dispatch2517 micro raw64 b1897 b765 b1883 -> b1897
node dispatch2518 micro raw65 b1897 b766 -> b1897
node session_CopyInto_-7983 session sessionCopyInto:srcOff=0,dstOff=0,n=61440 b1897 -> b1894
node session_CopyInto_-7984 session sessionCopyInto:srcOff=16384,dstOff=206080,n=4096 b769 -> b1894
node session_CopyInto_-7985 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b769 -> b1902
node dispatch2519 micro raw5 b1894_840704_2129920 b1902 -> b1894_840704_2129920
node dispatch2520 micro raw66 b1894 -> b1894
node dispatch2521 micro raw67 b1894 -> b1894
node session_CopyInto_-7986 session sessionCopyInto:srcOff=102400,dstOff=0,n=20480 b1894 -> b1898
node dispatch2522 micro raw68 b1899 b767 b1898 -> b1899
node dispatch2523 micro raw69 b1899 b768 -> b1899
node session_CopyInto_-7987 session sessionCopyInto:srcOff=210176,dstOff=0,n=131072 b1894 -> b1902
node dispatch2524 micro raw5 b769_81920_524288 b1902 -> b769_81920_524288
node session_CopyInto_-7988 session sessionCopyInto:srcOff=472320,dstOff=16384,n=4096 b1894 -> b769
node dispatch2525 micro raw70 b1900 b1886 b1899 -> b1900
node dispatch2526 micro raw71 b1901 b775 b1900 -> b1901
node dispatch2527 micro raw69 b1901 b776 -> b1901
node dispatch2528 micro raw72 b1886 b1901 b1899 -> b1886
node session_VecOp_-7989 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-7993 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b796 -> b1883
node dispatch2529 micro raw56 b1884 b1527 b1528 b1883 -> b1884
node dispatch2530 micro raw57 b1884_296960_296960 b1527 b1528 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-7995 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-7996 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b795 -> b1887
node session_CopyInto_-7997 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b795 -> b1887
node session_CopyInto_-7998 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b795 -> b1887
node session_CopyInto_-7999 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b795 -> b1887
node dispatch2531 micro raw58 b1887 -> b1887
node dispatch2532 micro raw59 b1887 -> b1887
node dispatch2533 micro raw60 b1887 -> b1887
node session_CopyInto_-8000 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2534 micro raw61 b1886 b1531 b1532 b1885 -> b1886
node dispatch2535 micro raw62 b1886_65536_65536 b1531 b1532 b1885_131072_131072 -> b1886_65536_65536
node dispatch2536 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8002 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b795
node session_CopyInto_-8003 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b795
node session_VecOp_-8004 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8005 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b814 -> b1883
node dispatch2537 micro raw56 b1884 b1535 b1536 b1883 -> b1884
node dispatch2538 micro raw57 b1884_296960_296960 b1535 b1536 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8007 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8008 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b813 -> b1887
node session_CopyInto_-8009 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b813 -> b1887
node session_CopyInto_-8010 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b813 -> b1887
node session_CopyInto_-8011 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b813 -> b1887
node dispatch2539 micro raw58 b1887 -> b1887
node dispatch2540 micro raw59 b1887 -> b1887
node dispatch2541 micro raw60 b1887 -> b1887
node session_CopyInto_-8012 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2542 micro raw61 b1886 b1539 b1540 b1885 -> b1886
node dispatch2543 micro raw62 b1886_65536_65536 b1539 b1540 b1885_131072_131072 -> b1886_65536_65536
node dispatch2544 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8014 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b813
node session_CopyInto_-8015 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b813
node session_VecOp_-8016 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8017 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b832 -> b1883
node dispatch2545 micro raw56 b1884 b1543 b1544 b1883 -> b1884
node dispatch2546 micro raw57 b1884_296960_296960 b1543 b1544 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8019 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8020 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b831 -> b1887
node session_CopyInto_-8021 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b831 -> b1887
node session_CopyInto_-8022 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b831 -> b1887
node session_CopyInto_-8023 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b831 -> b1887
node dispatch2547 micro raw58 b1887 -> b1887
node dispatch2548 micro raw59 b1887 -> b1887
node dispatch2549 micro raw60 b1887 -> b1887
node session_CopyInto_-8024 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2550 micro raw61 b1886 b1547 b1548 b1885 -> b1886
node dispatch2551 micro raw62 b1886_65536_65536 b1547 b1548 b1885_131072_131072 -> b1886_65536_65536
node dispatch2552 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8026 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b831
node session_CopyInto_-8027 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b831
node session_VecOp_-8028 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8029 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b850 -> b1883
node dispatch2553 micro raw56 b1884 b1551 b1552 b1883 -> b1884
node dispatch2554 micro raw57 b1884_296960_296960 b1551 b1552 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8031 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8032 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b849 -> b1887
node session_CopyInto_-8033 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b849 -> b1887
node session_CopyInto_-8034 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b849 -> b1887
node session_CopyInto_-8035 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b849 -> b1887
node dispatch2555 micro raw58 b1887 -> b1887
node dispatch2556 micro raw59 b1887 -> b1887
node dispatch2557 micro raw60 b1887 -> b1887
node session_CopyInto_-8036 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2558 micro raw61 b1886 b1555 b1556 b1885 -> b1886
node dispatch2559 micro raw62 b1886_65536_65536 b1555 b1556 b1885_131072_131072 -> b1886_65536_65536
node dispatch2560 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8038 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b849
node session_CopyInto_-8039 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b849
node session_VecOp_-8040 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8044 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b868 -> b1883
node dispatch2561 micro raw56 b1884 b1559 b1560 b1883 -> b1884
node dispatch2562 micro raw57 b1884_296960_296960 b1559 b1560 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8046 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8047 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b867 -> b1887
node session_CopyInto_-8048 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b867 -> b1887
node session_CopyInto_-8049 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b867 -> b1887
node session_CopyInto_-8050 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b867 -> b1887
node dispatch2563 micro raw58 b1887 -> b1887
node dispatch2564 micro raw59 b1887 -> b1887
node dispatch2565 micro raw60 b1887 -> b1887
node session_CopyInto_-8051 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2566 micro raw61 b1886 b1563 b1564 b1885 -> b1886
node dispatch2567 micro raw62 b1886_65536_65536 b1563 b1564 b1885_131072_131072 -> b1886_65536_65536
node dispatch2568 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8053 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b867
node session_CopyInto_-8054 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b867
node session_VecOp_-8055 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8056 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b886 -> b1883
node dispatch2569 micro raw56 b1884 b1567 b1568 b1883 -> b1884
node dispatch2570 micro raw57 b1884_296960_296960 b1567 b1568 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8058 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8059 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b885 -> b1887
node session_CopyInto_-8060 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b885 -> b1887
node session_CopyInto_-8061 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b885 -> b1887
node session_CopyInto_-8062 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b885 -> b1887
node dispatch2571 micro raw58 b1887 -> b1887
node dispatch2572 micro raw59 b1887 -> b1887
node dispatch2573 micro raw60 b1887 -> b1887
node session_CopyInto_-8063 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2574 micro raw61 b1886 b1571 b1572 b1885 -> b1886
node dispatch2575 micro raw62 b1886_65536_65536 b1571 b1572 b1885_131072_131072 -> b1886_65536_65536
node dispatch2576 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8065 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b885
node session_CopyInto_-8066 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b885
node session_VecOp_-8067 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8068 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b904 -> b1883
node dispatch2577 micro raw56 b1884 b1575 b1576 b1883 -> b1884
node dispatch2578 micro raw57 b1884_296960_296960 b1575 b1576 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8070 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8071 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b903 -> b1887
node session_CopyInto_-8072 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b903 -> b1887
node session_CopyInto_-8073 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b903 -> b1887
node session_CopyInto_-8074 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b903 -> b1887
node dispatch2579 micro raw58 b1887 -> b1887
node dispatch2580 micro raw59 b1887 -> b1887
node dispatch2581 micro raw60 b1887 -> b1887
node session_CopyInto_-8075 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2582 micro raw61 b1886 b1579 b1580 b1885 -> b1886
node dispatch2583 micro raw62 b1886_65536_65536 b1579 b1580 b1885_131072_131072 -> b1886_65536_65536
node dispatch2584 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8077 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b903
node session_CopyInto_-8078 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b903
node session_VecOp_-8079 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8080 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b922 -> b1883
node dispatch2585 micro raw56 b1884 b1583 b1584 b1883 -> b1884
node dispatch2586 micro raw57 b1884_296960_296960 b1583 b1584 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8082 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8083 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b921 -> b1887
node session_CopyInto_-8084 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b921 -> b1887
node session_CopyInto_-8085 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b921 -> b1887
node session_CopyInto_-8086 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b921 -> b1887
node dispatch2587 micro raw58 b1887 -> b1887
node dispatch2588 micro raw59 b1887 -> b1887
node dispatch2589 micro raw60 b1887 -> b1887
node session_CopyInto_-8087 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2590 micro raw61 b1886 b1587 b1588 b1885 -> b1886
node dispatch2591 micro raw62 b1886_65536_65536 b1587 b1588 b1885_131072_131072 -> b1886_65536_65536
node dispatch2592 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8089 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b921
node session_CopyInto_-8090 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b921
node session_VecOp_-8091 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8095 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b940 -> b1883
node dispatch2593 micro raw56 b1884 b1591 b1592 b1883 -> b1884
node dispatch2594 micro raw57 b1884_296960_296960 b1591 b1592 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8097 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8098 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b939 -> b1887
node session_CopyInto_-8099 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b939 -> b1887
node session_CopyInto_-8100 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b939 -> b1887
node session_CopyInto_-8101 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b939 -> b1887
node dispatch2595 micro raw58 b1887 -> b1887
node dispatch2596 micro raw59 b1887 -> b1887
node dispatch2597 micro raw60 b1887 -> b1887
node session_CopyInto_-8102 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2598 micro raw61 b1886 b1595 b1596 b1885 -> b1886
node dispatch2599 micro raw62 b1886_65536_65536 b1595 b1596 b1885_131072_131072 -> b1886_65536_65536
node dispatch2600 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8104 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b939
node session_CopyInto_-8105 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b939
node session_VecOp_-8106 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8107 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b958 -> b1883
node dispatch2601 micro raw56 b1884 b1599 b1600 b1883 -> b1884
node dispatch2602 micro raw57 b1884_296960_296960 b1599 b1600 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8109 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8110 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b957 -> b1887
node session_CopyInto_-8111 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b957 -> b1887
node session_CopyInto_-8112 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b957 -> b1887
node session_CopyInto_-8113 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b957 -> b1887
node dispatch2603 micro raw58 b1887 -> b1887
node dispatch2604 micro raw59 b1887 -> b1887
node dispatch2605 micro raw60 b1887 -> b1887
node session_CopyInto_-8114 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2606 micro raw61 b1886 b1603 b1604 b1885 -> b1886
node dispatch2607 micro raw62 b1886_65536_65536 b1603 b1604 b1885_131072_131072 -> b1886_65536_65536
node dispatch2608 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8116 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b957
node session_CopyInto_-8117 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b957
node session_VecOp_-8118 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8119 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b976 -> b1883
node dispatch2609 micro raw56 b1884 b1615 b1616 b1883 -> b1884
node dispatch2610 micro raw57 b1884_296960_296960 b1615 b1616 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8121 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8122 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b975 -> b1887
node session_CopyInto_-8123 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b975 -> b1887
node session_CopyInto_-8124 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b975 -> b1887
node session_CopyInto_-8125 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b975 -> b1887
node dispatch2611 micro raw58 b1887 -> b1887
node dispatch2612 micro raw59 b1887 -> b1887
node dispatch2613 micro raw60 b1887 -> b1887
node session_CopyInto_-8126 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2614 micro raw61 b1886 b1619 b1620 b1885 -> b1886
node dispatch2615 micro raw62 b1886_65536_65536 b1619 b1620 b1885_131072_131072 -> b1886_65536_65536
node dispatch2616 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8128 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b975
node session_CopyInto_-8129 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b975
node session_VecOp_-8130 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8131 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b994 -> b1883
node dispatch2617 micro raw56 b1884 b1623 b1624 b1883 -> b1884
node dispatch2618 micro raw57 b1884_296960_296960 b1623 b1624 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8133 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8134 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b993 -> b1887
node session_CopyInto_-8135 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b993 -> b1887
node session_CopyInto_-8136 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b993 -> b1887
node session_CopyInto_-8137 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b993 -> b1887
node dispatch2619 micro raw58 b1887 -> b1887
node dispatch2620 micro raw59 b1887 -> b1887
node dispatch2621 micro raw60 b1887 -> b1887
node session_CopyInto_-8138 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2622 micro raw61 b1886 b1627 b1628 b1885 -> b1886
node dispatch2623 micro raw62 b1886_65536_65536 b1627 b1628 b1885_131072_131072 -> b1886_65536_65536
node dispatch2624 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8140 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b993
node session_CopyInto_-8141 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b993
node session_VecOp_-8142 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8146 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1012 -> b1883
node dispatch2625 micro raw56 b1884 b1631 b1632 b1883 -> b1884
node dispatch2626 micro raw57 b1884_296960_296960 b1631 b1632 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8148 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8149 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1011 -> b1887
node session_CopyInto_-8150 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1011 -> b1887
node session_CopyInto_-8151 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1011 -> b1887
node session_CopyInto_-8152 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1011 -> b1887
node dispatch2627 micro raw58 b1887 -> b1887
node dispatch2628 micro raw59 b1887 -> b1887
node dispatch2629 micro raw60 b1887 -> b1887
node session_CopyInto_-8153 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2630 micro raw61 b1886 b1635 b1636 b1885 -> b1886
node dispatch2631 micro raw62 b1886_65536_65536 b1635 b1636 b1885_131072_131072 -> b1886_65536_65536
node dispatch2632 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8155 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1011
node session_CopyInto_-8156 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1011
node session_VecOp_-8157 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8158 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1030 -> b1883
node dispatch2633 micro raw56 b1884 b1639 b1640 b1883 -> b1884
node dispatch2634 micro raw57 b1884_296960_296960 b1639 b1640 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8160 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8161 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1029 -> b1887
node session_CopyInto_-8162 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1029 -> b1887
node session_CopyInto_-8163 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1029 -> b1887
node session_CopyInto_-8164 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1029 -> b1887
node dispatch2635 micro raw58 b1887 -> b1887
node dispatch2636 micro raw59 b1887 -> b1887
node dispatch2637 micro raw60 b1887 -> b1887
node session_CopyInto_-8165 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2638 micro raw61 b1886 b1643 b1644 b1885 -> b1886
node dispatch2639 micro raw62 b1886_65536_65536 b1643 b1644 b1885_131072_131072 -> b1886_65536_65536
node dispatch2640 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8167 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1029
node session_CopyInto_-8168 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1029
node session_VecOp_-8169 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8170 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1048 -> b1883
node dispatch2641 micro raw56 b1884 b1647 b1648 b1883 -> b1884
node dispatch2642 micro raw57 b1884_296960_296960 b1647 b1648 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8172 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8173 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1047 -> b1887
node session_CopyInto_-8174 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1047 -> b1887
node session_CopyInto_-8175 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1047 -> b1887
node session_CopyInto_-8176 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1047 -> b1887
node dispatch2643 micro raw58 b1887 -> b1887
node dispatch2644 micro raw59 b1887 -> b1887
node dispatch2645 micro raw60 b1887 -> b1887
node session_CopyInto_-8177 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2646 micro raw61 b1886 b1651 b1652 b1885 -> b1886
node dispatch2647 micro raw62 b1886_65536_65536 b1651 b1652 b1885_131072_131072 -> b1886_65536_65536
node dispatch2648 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8179 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1047
node session_CopyInto_-8180 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1047
node session_VecOp_-8181 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8182 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1066 -> b1883
node dispatch2649 micro raw56 b1884 b1655 b1656 b1883 -> b1884
node dispatch2650 micro raw57 b1884_296960_296960 b1655 b1656 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8184 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8185 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1065 -> b1887
node session_CopyInto_-8186 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1065 -> b1887
node session_CopyInto_-8187 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1065 -> b1887
node session_CopyInto_-8188 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1065 -> b1887
node dispatch2651 micro raw58 b1887 -> b1887
node dispatch2652 micro raw59 b1887 -> b1887
node dispatch2653 micro raw60 b1887 -> b1887
node session_CopyInto_-8189 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2654 micro raw61 b1886 b1659 b1660 b1885 -> b1886
node dispatch2655 micro raw62 b1886_65536_65536 b1659 b1660 b1885_131072_131072 -> b1886_65536_65536
node dispatch2656 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8191 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1065
node session_CopyInto_-8192 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1065
node dispatch2657 micro raw64 b1897 b1070 b1883 -> b1897
node dispatch2658 micro raw65 b1897 b1071 -> b1897
node session_CopyInto_-8193 session sessionCopyInto:srcOff=0,dstOff=0,n=61440 b1897 -> b1894
node session_CopyInto_-8194 session sessionCopyInto:srcOff=16384,dstOff=206080,n=4096 b1074 -> b1894
node session_CopyInto_-8195 session sessionCopyInto:srcOff=20480,dstOff=0,n=131072 b1074 -> b1902
node dispatch2659 micro raw5 b1894_840704_2129920 b1902 -> b1894_840704_2129920
node dispatch2660 micro raw66 b1894 -> b1894
node dispatch2661 micro raw67 b1894 -> b1894
node session_CopyInto_-8196 session sessionCopyInto:srcOff=102400,dstOff=0,n=20480 b1894 -> b1898
node dispatch2662 micro raw68 b1899 b1072 b1898 -> b1899
node dispatch2663 micro raw69 b1899 b1073 -> b1899
node session_CopyInto_-8197 session sessionCopyInto:srcOff=210176,dstOff=0,n=131072 b1894 -> b1902
node dispatch2664 micro raw5 b1074_81920_524288 b1902 -> b1074_81920_524288
node session_CopyInto_-8198 session sessionCopyInto:srcOff=472320,dstOff=16384,n=4096 b1894 -> b1074
node dispatch2665 micro raw70 b1900 b1886 b1899 -> b1900
node dispatch2666 micro raw71 b1901 b1080 b1900 -> b1901
node dispatch2667 micro raw69 b1901 b1081 -> b1901
node dispatch2668 micro raw72 b1886 b1901 b1899 -> b1886
node session_VecOp_-8199 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8203 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1101 -> b1883
node dispatch2669 micro raw56 b1884 b1663 b1664 b1883 -> b1884
node dispatch2670 micro raw57 b1884_296960_296960 b1663 b1664 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8205 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8206 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1100 -> b1887
node session_CopyInto_-8207 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1100 -> b1887
node session_CopyInto_-8208 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1100 -> b1887
node session_CopyInto_-8209 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1100 -> b1887
node dispatch2671 micro raw58 b1887 -> b1887
node dispatch2672 micro raw59 b1887 -> b1887
node dispatch2673 micro raw60 b1887 -> b1887
node session_CopyInto_-8210 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2674 micro raw61 b1886 b1667 b1668 b1885 -> b1886
node dispatch2675 micro raw62 b1886_65536_65536 b1667 b1668 b1885_131072_131072 -> b1886_65536_65536
node dispatch2676 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8212 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1100
node session_CopyInto_-8213 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1100
node session_VecOp_-8214 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8215 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1119 -> b1883
node dispatch2677 micro raw56 b1884 b1671 b1672 b1883 -> b1884
node dispatch2678 micro raw57 b1884_296960_296960 b1671 b1672 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8217 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8218 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1118 -> b1887
node session_CopyInto_-8219 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1118 -> b1887
node session_CopyInto_-8220 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1118 -> b1887
node session_CopyInto_-8221 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1118 -> b1887
node dispatch2679 micro raw58 b1887 -> b1887
node dispatch2680 micro raw59 b1887 -> b1887
node dispatch2681 micro raw60 b1887 -> b1887
node session_CopyInto_-8222 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2682 micro raw61 b1886 b1675 b1676 b1885 -> b1886
node dispatch2683 micro raw62 b1886_65536_65536 b1675 b1676 b1885_131072_131072 -> b1886_65536_65536
node dispatch2684 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8224 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1118
node session_CopyInto_-8225 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1118
node session_VecOp_-8226 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8227 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1137 -> b1883
node dispatch2685 micro raw56 b1884 b1679 b1680 b1883 -> b1884
node dispatch2686 micro raw57 b1884_296960_296960 b1679 b1680 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8229 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8230 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1136 -> b1887
node session_CopyInto_-8231 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1136 -> b1887
node session_CopyInto_-8232 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1136 -> b1887
node session_CopyInto_-8233 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1136 -> b1887
node dispatch2687 micro raw58 b1887 -> b1887
node dispatch2688 micro raw59 b1887 -> b1887
node dispatch2689 micro raw60 b1887 -> b1887
node session_CopyInto_-8234 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2690 micro raw61 b1886 b1683 b1684 b1885 -> b1886
node dispatch2691 micro raw62 b1886_65536_65536 b1683 b1684 b1885_131072_131072 -> b1886_65536_65536
node dispatch2692 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8236 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1136
node session_CopyInto_-8237 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1136
node session_VecOp_-8238 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8239 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1155 -> b1883
node dispatch2693 micro raw56 b1884 b1687 b1688 b1883 -> b1884
node dispatch2694 micro raw57 b1884_296960_296960 b1687 b1688 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8241 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8242 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1154 -> b1887
node session_CopyInto_-8243 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1154 -> b1887
node session_CopyInto_-8244 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1154 -> b1887
node session_CopyInto_-8245 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1154 -> b1887
node dispatch2695 micro raw58 b1887 -> b1887
node dispatch2696 micro raw59 b1887 -> b1887
node dispatch2697 micro raw60 b1887 -> b1887
node session_CopyInto_-8246 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2698 micro raw61 b1886 b1691 b1692 b1885 -> b1886
node dispatch2699 micro raw62 b1886_65536_65536 b1691 b1692 b1885_131072_131072 -> b1886_65536_65536
node dispatch2700 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8248 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1154
node session_CopyInto_-8249 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1154
node session_VecOp_-8250 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8254 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1173 -> b1883
node dispatch2701 micro raw56 b1884 b1703 b1704 b1883 -> b1884
node dispatch2702 micro raw57 b1884_296960_296960 b1703 b1704 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8256 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8257 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1172 -> b1887
node session_CopyInto_-8258 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1172 -> b1887
node session_CopyInto_-8259 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1172 -> b1887
node session_CopyInto_-8260 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1172 -> b1887
node dispatch2703 micro raw58 b1887 -> b1887
node dispatch2704 micro raw59 b1887 -> b1887
node dispatch2705 micro raw60 b1887 -> b1887
node session_CopyInto_-8261 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2706 micro raw61 b1886 b1707 b1708 b1885 -> b1886
node dispatch2707 micro raw62 b1886_65536_65536 b1707 b1708 b1885_131072_131072 -> b1886_65536_65536
node dispatch2708 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8263 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1172
node session_CopyInto_-8264 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1172
node session_VecOp_-8265 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8266 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1191 -> b1883
node dispatch2709 micro raw56 b1884 b1711 b1712 b1883 -> b1884
node dispatch2710 micro raw57 b1884_296960_296960 b1711 b1712 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8268 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8269 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1190 -> b1887
node session_CopyInto_-8270 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1190 -> b1887
node session_CopyInto_-8271 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1190 -> b1887
node session_CopyInto_-8272 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1190 -> b1887
node dispatch2711 micro raw58 b1887 -> b1887
node dispatch2712 micro raw59 b1887 -> b1887
node dispatch2713 micro raw60 b1887 -> b1887
node session_CopyInto_-8273 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2714 micro raw61 b1886 b1715 b1716 b1885 -> b1886
node dispatch2715 micro raw62 b1886_65536_65536 b1715 b1716 b1885_131072_131072 -> b1886_65536_65536
node dispatch2716 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8275 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1190
node session_CopyInto_-8276 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1190
node session_VecOp_-8277 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8278 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1209 -> b1883
node dispatch2717 micro raw56 b1884 b1719 b1720 b1883 -> b1884
node dispatch2718 micro raw57 b1884_296960_296960 b1719 b1720 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8280 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8281 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1208 -> b1887
node session_CopyInto_-8282 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1208 -> b1887
node session_CopyInto_-8283 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1208 -> b1887
node session_CopyInto_-8284 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1208 -> b1887
node dispatch2719 micro raw58 b1887 -> b1887
node dispatch2720 micro raw59 b1887 -> b1887
node dispatch2721 micro raw60 b1887 -> b1887
node session_CopyInto_-8285 session sessionCopyInto:srcOff=2475688,dstOff=0,n=40960 b1887 -> b1885
node dispatch2722 micro raw61 b1886 b1723 b1724 b1885 -> b1886
node dispatch2723 micro raw62 b1886_65536_65536 b1723 b1724 b1885_131072_131072 -> b1886_65536_65536
node dispatch2724 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8287 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1208
node session_CopyInto_-8288 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1208
node session_VecOp_-8289 session sessionVecOp:op=add,n=20480 b1882 b1886 -> b1882
node session_RmsnormEnc_-8290 session sessionRmsnormEnc:rows=5,cols=4096,eps=h3727c5ac b1882 b1227 -> b1883
node dispatch2725 micro raw56 b1884 b1727 b1728 b1883 -> b1884
node dispatch2726 micro raw57 b1884_296960_296960 b1727 b1728 b1883_65536_65536 -> b1884_296960_296960
node session_CopyInto_-8292 session sessionCopyInto:srcOff=0,dstOff=0,n=92800 b1884 -> b1887
node session_CopyInto_-8293 session sessionCopyInto:srcOff=45696,dstOff=92800,n=51200 b1226 -> b1887
node session_CopyInto_-8294 session sessionCopyInto:srcOff=37120,dstOff=144000,n=8576 b1226 -> b1887
node session_CopyInto_-8295 session sessionCopyInto:srcOff=127616,dstOff=255616,n=1048576 b1226 -> b1887
node session_CopyInto_-8296 session sessionCopyInto:srcOff=96896,dstOff=2516648,n=30720 b1226 -> b1887
node dispatch2727 micro raw58 b1887 -> b1887
node dispatch2728 micro raw59 b1887 -> b1887
node dispatch2729 micro raw60 b1887 -> b1887
node dispatch2732 micro raw63 b1888 b1884 -> b1888
node session_CopyInto_-8299 session sessionCopyInto:srcOff=0,dstOff=96896,n=30720 b1888 -> b1226
node session_CopyInto_-8300 session sessionCopyInto:srcOff=1304192,dstOff=127616,n=1048576 b1887 -> b1226
phase decode
node embedding_7 ext fir.embedding token backbone_embeddings_weight_emb -> b1903
node session_CopyInto_-8323 session sessionCopyInto:srcOff=0,dstOff=0,n=4096 b1903 -> b1231
node session_RmsnormEnc_-8324 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b24 -> b1232
node dispatch2733 micro raw57 b23 b1247 b1248 b1232 -> b23
node dispatch2734 micro raw73 b23 -> b23
node dispatch2735 micro raw74 b23 -> b23
node dispatch2736 micro raw75 b23 -> b23
node dispatch2737 micro raw62 b27 b1251 b1252 b23_4803072_32768 -> b27
node session_VecOp_-8325 session sessionVecOp:op=add,n=4096 b1231 b27 -> b1231
node session_RmsnormEnc_-8326 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b42 -> b1232
node dispatch2738 micro raw57 b41 b1255 b1256 b1232 -> b41
node dispatch2739 micro raw73 b41 -> b41
node dispatch2740 micro raw74 b41 -> b41
node dispatch2741 micro raw75 b41 -> b41
node dispatch2742 micro raw62 b45 b1259 b1260 b41_4803072_32768 -> b45
node session_VecOp_-8327 session sessionVecOp:op=add,n=4096 b1231 b45 -> b1231
node session_RmsnormEnc_-8328 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b60 -> b1232
node dispatch2743 micro raw57 b59 b1343 b1344 b1232 -> b59
node dispatch2744 micro raw73 b59 -> b59
node dispatch2745 micro raw74 b59 -> b59
node dispatch2746 micro raw75 b59 -> b59
node dispatch2747 micro raw62 b63 b1347 b1348 b59_4803072_32768 -> b63
node session_VecOp_-8329 session sessionVecOp:op=add,n=4096 b1231 b63 -> b1231
node session_RmsnormEnc_-8330 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b78 -> b1232
node dispatch2748 micro raw57 b77 b1431 b1432 b1232 -> b77
node dispatch2749 micro raw73 b77 -> b77
node dispatch2750 micro raw74 b77 -> b77
node dispatch2751 micro raw75 b77 -> b77
node dispatch2752 micro raw62 b81 b1435 b1436 b77_4803072_32768 -> b81
node session_VecOp_-8331 session sessionVecOp:op=add,n=4096 b1231 b81 -> b1231
node session_RmsnormEnc_-8332 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b96 -> b1232
node dispatch2753 micro raw57 b95 b1519 b1520 b1232 -> b95
node dispatch2754 micro raw73 b95 -> b95
node dispatch2755 micro raw74 b95 -> b95
node dispatch2756 micro raw75 b95 -> b95
node dispatch2757 micro raw62 b99 b1523 b1524 b95_4803072_32768 -> b99
node session_VecOp_-8333 session sessionVecOp:op=add,n=4096 b1231 b99 -> b1231
node session_RmsnormEnc_-8334 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b114 -> b1232
node dispatch2758 micro raw57 b113 b1607 b1608 b1232 -> b113
node dispatch2759 micro raw73 b113 -> b113
node dispatch2760 micro raw74 b113 -> b113
node dispatch2761 micro raw75 b113 -> b113
node dispatch2762 micro raw62 b117 b1611 b1612 b113_4803072_32768 -> b117
node session_VecOp_-8335 session sessionVecOp:op=add,n=4096 b1231 b117 -> b1231
node session_RmsnormEnc_-8336 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b132 -> b1232
node dispatch2763 micro raw57 b131 b1695 b1696 b1232 -> b131
node dispatch2764 micro raw73 b131 -> b131
node dispatch2765 micro raw74 b131 -> b131
node dispatch2766 micro raw75 b131 -> b131
node dispatch2767 micro raw62 b135 b1699 b1700 b131_4803072_32768 -> b135
node session_VecOp_-8337 session sessionVecOp:op=add,n=4096 b1231 b135 -> b1231
node session_RmsnormEnc_-8338 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b159 -> b1232
node dispatch2768 micro raw76 b158 b154 b1232 -> b158
node dispatch2769 micro raw77 b158 b155 -> b158
node dispatch2770 micro raw78 b158 -> b158
node dispatch2771 micro raw79 b164 b156 b158_49152_557056 -> b164
node dispatch2772 micro raw80 b164 b157 -> b164
node dispatch2773 micro raw57 b149 b1735 b1736 b1232 -> b149
node dispatch2774 micro raw73 b149 -> b149
node dispatch2775 micro raw74 b149 -> b149
node dispatch2776 micro raw75 b149 -> b149
node dispatch2777 micro raw62 b153 b1739 b1740 b149_4803072_32768 -> b153
node session_CopyInto_-8339 session sessionCopyInto:srcOff=0,dstOff=0,n=4096 b153 -> b167
node session_CopyInto_-8340 session sessionCopyInto:srcOff=0,dstOff=4096,n=4096 b164 -> b167
node dispatch2778 micro raw81 b168 b165 b167 -> b168
node dispatch2779 micro raw80 b168 b166 -> b168
node session_Activation_-8341 session sessionActivation:op=sigmoid,n=4096,p=h00000000 b168 -> b169
node session_VecOp_-8342 session sessionVecOp:op=sub,n=4096 b171 b169 -> b170
node session_VecOp_-8343 session sessionVecOp:op=mul,n=4096 b169 b153 -> b153
node session_VecOp_-8344 session sessionVecOp:op=mul,n=4096 b170 b164 -> b164
node session_VecOp_-8345 session sessionVecOp:op=add,n=4096 b153 b164 -> b153
node session_VecOp_-8346 session sessionVecOp:op=add,n=4096 b1231 b153 -> b1231
node session_RmsnormEnc_-8347 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b186 -> b1232
node dispatch2780 micro raw57 b185 b1743 b1744 b1232 -> b185
node dispatch2781 micro raw73 b185 -> b185
node dispatch2782 micro raw74 b185 -> b185
node dispatch2783 micro raw75 b185 -> b185
node dispatch2784 micro raw62 b189 b1747 b1748 b185_4803072_32768 -> b189
node session_VecOp_-8348 session sessionVecOp:op=add,n=4096 b1231 b189 -> b1231
node session_RmsnormEnc_-8349 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b204 -> b1232
node dispatch2785 micro raw57 b203 b1751 b1752 b1232 -> b203
node dispatch2786 micro raw73 b203 -> b203
node dispatch2787 micro raw74 b203 -> b203
node dispatch2788 micro raw75 b203 -> b203
node dispatch2789 micro raw62 b207 b1755 b1756 b203_4803072_32768 -> b207
node session_VecOp_-8350 session sessionVecOp:op=add,n=4096 b1231 b207 -> b1231
node session_RmsnormEnc_-8351 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b222 -> b1232
node dispatch2790 micro raw57 b221 b1263 b1264 b1232 -> b221
node dispatch2791 micro raw73 b221 -> b221
node dispatch2792 micro raw74 b221 -> b221
node dispatch2793 micro raw75 b221 -> b221
node dispatch2794 micro raw62 b225 b1267 b1268 b221_4803072_32768 -> b225
node session_VecOp_-8352 session sessionVecOp:op=add,n=4096 b1231 b225 -> b1231
node session_RmsnormEnc_-8353 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b240 -> b1232
node dispatch2795 micro raw57 b239 b1271 b1272 b1232 -> b239
node dispatch2796 micro raw73 b239 -> b239
node dispatch2797 micro raw74 b239 -> b239
node dispatch2798 micro raw75 b239 -> b239
node dispatch2799 micro raw62 b243 b1275 b1276 b239_4803072_32768 -> b243
node session_VecOp_-8354 session sessionVecOp:op=add,n=4096 b1231 b243 -> b1231
node session_RmsnormEnc_-8355 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b258 -> b1232
node dispatch2800 micro raw57 b257 b1279 b1280 b1232 -> b257
node dispatch2801 micro raw73 b257 -> b257
node dispatch2802 micro raw74 b257 -> b257
node dispatch2803 micro raw75 b257 -> b257
node dispatch2804 micro raw62 b261 b1283 b1284 b257_4803072_32768 -> b261
node session_VecOp_-8356 session sessionVecOp:op=add,n=4096 b1231 b261 -> b1231
node session_RmsnormEnc_-8357 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b276 -> b1232
node dispatch2805 micro raw57 b275 b1287 b1288 b1232 -> b275
node dispatch2806 micro raw73 b275 -> b275
node dispatch2807 micro raw74 b275 -> b275
node dispatch2808 micro raw75 b275 -> b275
node dispatch2809 micro raw62 b279 b1291 b1292 b275_4803072_32768 -> b279
node session_VecOp_-8358 session sessionVecOp:op=add,n=4096 b1231 b279 -> b1231
node session_RmsnormEnc_-8359 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b294 -> b1232
node dispatch2810 micro raw57 b293 b1295 b1296 b1232 -> b293
node dispatch2811 micro raw73 b293 -> b293
node dispatch2812 micro raw74 b293 -> b293
node dispatch2813 micro raw75 b293 -> b293
node dispatch2814 micro raw62 b297 b1299 b1300 b293_4803072_32768 -> b297
node session_VecOp_-8360 session sessionVecOp:op=add,n=4096 b1231 b297 -> b1231
node session_RmsnormEnc_-8361 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b312 -> b1232
node dispatch2815 micro raw57 b311 b1303 b1304 b1232 -> b311
node dispatch2816 micro raw73 b311 -> b311
node dispatch2817 micro raw74 b311 -> b311
node dispatch2818 micro raw75 b311 -> b311
node dispatch2819 micro raw62 b315 b1307 b1308 b311_4803072_32768 -> b315
node session_VecOp_-8362 session sessionVecOp:op=add,n=4096 b1231 b315 -> b1231
node session_RmsnormEnc_-8363 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b330 -> b1232
node dispatch2820 micro raw57 b329 b1311 b1312 b1232 -> b329
node dispatch2821 micro raw73 b329 -> b329
node dispatch2822 micro raw74 b329 -> b329
node dispatch2823 micro raw75 b329 -> b329
node dispatch2824 micro raw62 b333 b1315 b1316 b329_4803072_32768 -> b333
node session_VecOp_-8364 session sessionVecOp:op=add,n=4096 b1231 b333 -> b1231
node session_RmsnormEnc_-8365 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b348 -> b1232
node dispatch2825 micro raw57 b347 b1319 b1320 b1232 -> b347
node dispatch2826 micro raw73 b347 -> b347
node dispatch2827 micro raw74 b347 -> b347
node dispatch2828 micro raw75 b347 -> b347
node dispatch2829 micro raw62 b351 b1323 b1324 b347_4803072_32768 -> b351
node session_VecOp_-8366 session sessionVecOp:op=add,n=4096 b1231 b351 -> b1231
node session_RmsnormEnc_-8367 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b366 -> b1232
node dispatch2830 micro raw57 b365 b1327 b1328 b1232 -> b365
node dispatch2831 micro raw73 b365 -> b365
node dispatch2832 micro raw74 b365 -> b365
node dispatch2833 micro raw75 b365 -> b365
node dispatch2834 micro raw62 b369 b1331 b1332 b365_4803072_32768 -> b369
node session_VecOp_-8368 session sessionVecOp:op=add,n=4096 b1231 b369 -> b1231
node session_RmsnormEnc_-8369 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b384 -> b1232
node dispatch2835 micro raw57 b383 b1335 b1336 b1232 -> b383
node dispatch2836 micro raw73 b383 -> b383
node dispatch2837 micro raw74 b383 -> b383
node dispatch2838 micro raw75 b383 -> b383
node dispatch2839 micro raw62 b387 b1339 b1340 b383_4803072_32768 -> b387
node session_VecOp_-8370 session sessionVecOp:op=add,n=4096 b1231 b387 -> b1231
node session_RmsnormEnc_-8371 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b402 -> b1232
node dispatch2840 micro raw57 b401 b1351 b1352 b1232 -> b401
node dispatch2841 micro raw73 b401 -> b401
node dispatch2842 micro raw74 b401 -> b401
node dispatch2843 micro raw75 b401 -> b401
node dispatch2844 micro raw62 b405 b1355 b1356 b401_4803072_32768 -> b405
node session_VecOp_-8372 session sessionVecOp:op=add,n=4096 b1231 b405 -> b1231
node session_RmsnormEnc_-8373 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b420 -> b1232
node dispatch2845 micro raw57 b419 b1359 b1360 b1232 -> b419
node dispatch2846 micro raw73 b419 -> b419
node dispatch2847 micro raw74 b419 -> b419
node dispatch2848 micro raw75 b419 -> b419
node dispatch2849 micro raw62 b423 b1363 b1364 b419_4803072_32768 -> b423
node session_VecOp_-8374 session sessionVecOp:op=add,n=4096 b1231 b423 -> b1231
node session_RmsnormEnc_-8375 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b438 -> b1232
node dispatch2850 micro raw57 b437 b1367 b1368 b1232 -> b437
node dispatch2851 micro raw73 b437 -> b437
node dispatch2852 micro raw74 b437 -> b437
node dispatch2853 micro raw75 b437 -> b437
node dispatch2854 micro raw62 b441 b1371 b1372 b437_4803072_32768 -> b441
node session_VecOp_-8376 session sessionVecOp:op=add,n=4096 b1231 b441 -> b1231
node session_RmsnormEnc_-8377 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b465 -> b1232
node dispatch2855 micro raw76 b464 b460 b1232 -> b464
node dispatch2856 micro raw77 b464 b461 -> b464
node dispatch2857 micro raw78 b464 -> b464
node dispatch2858 micro raw79 b164 b462 b464_49152_557056 -> b164
node dispatch2859 micro raw80 b164 b463 -> b164
node dispatch2860 micro raw57 b455 b1375 b1376 b1232 -> b455
node dispatch2861 micro raw73 b455 -> b455
node dispatch2862 micro raw74 b455 -> b455
node dispatch2863 micro raw75 b455 -> b455
node dispatch2864 micro raw62 b459 b1379 b1380 b455_4803072_32768 -> b459
node session_CopyInto_-8378 session sessionCopyInto:srcOff=0,dstOff=0,n=4096 b459 -> b472
node session_CopyInto_-8379 session sessionCopyInto:srcOff=0,dstOff=4096,n=4096 b164 -> b472
node dispatch2865 micro raw81 b473 b470 b472 -> b473
node dispatch2866 micro raw80 b473 b471 -> b473
node session_Activation_-8380 session sessionActivation:op=sigmoid,n=4096,p=h00000000 b473 -> b474
node session_VecOp_-8381 session sessionVecOp:op=sub,n=4096 b476 b474 -> b475
node session_VecOp_-8382 session sessionVecOp:op=mul,n=4096 b474 b459 -> b459
node session_VecOp_-8383 session sessionVecOp:op=mul,n=4096 b475 b164 -> b164
node session_VecOp_-8384 session sessionVecOp:op=add,n=4096 b459 b164 -> b459
node session_VecOp_-8385 session sessionVecOp:op=add,n=4096 b1231 b459 -> b1231
node session_RmsnormEnc_-8386 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b491 -> b1232
node dispatch2867 micro raw57 b490 b1383 b1384 b1232 -> b490
node dispatch2868 micro raw73 b490 -> b490
node dispatch2869 micro raw74 b490 -> b490
node dispatch2870 micro raw75 b490 -> b490
node dispatch2871 micro raw62 b494 b1387 b1388 b490_4803072_32768 -> b494
node session_VecOp_-8387 session sessionVecOp:op=add,n=4096 b1231 b494 -> b1231
node session_RmsnormEnc_-8388 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b509 -> b1232
node dispatch2872 micro raw57 b508 b1391 b1392 b1232 -> b508
node dispatch2873 micro raw73 b508 -> b508
node dispatch2874 micro raw74 b508 -> b508
node dispatch2875 micro raw75 b508 -> b508
node dispatch2876 micro raw62 b512 b1395 b1396 b508_4803072_32768 -> b512
node session_VecOp_-8389 session sessionVecOp:op=add,n=4096 b1231 b512 -> b1231
node session_RmsnormEnc_-8390 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b527 -> b1232
node dispatch2877 micro raw57 b526 b1399 b1400 b1232 -> b526
node dispatch2878 micro raw73 b526 -> b526
node dispatch2879 micro raw74 b526 -> b526
node dispatch2880 micro raw75 b526 -> b526
node dispatch2881 micro raw62 b530 b1403 b1404 b526_4803072_32768 -> b530
node session_VecOp_-8391 session sessionVecOp:op=add,n=4096 b1231 b530 -> b1231
node session_RmsnormEnc_-8392 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b545 -> b1232
node dispatch2882 micro raw57 b544 b1407 b1408 b1232 -> b544
node dispatch2883 micro raw73 b544 -> b544
node dispatch2884 micro raw74 b544 -> b544
node dispatch2885 micro raw75 b544 -> b544
node dispatch2886 micro raw62 b548 b1411 b1412 b544_4803072_32768 -> b548
node session_VecOp_-8393 session sessionVecOp:op=add,n=4096 b1231 b548 -> b1231
node session_RmsnormEnc_-8394 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b563 -> b1232
node dispatch2887 micro raw57 b562 b1415 b1416 b1232 -> b562
node dispatch2888 micro raw73 b562 -> b562
node dispatch2889 micro raw74 b562 -> b562
node dispatch2890 micro raw75 b562 -> b562
node dispatch2891 micro raw62 b566 b1419 b1420 b562_4803072_32768 -> b566
node session_VecOp_-8395 session sessionVecOp:op=add,n=4096 b1231 b566 -> b1231
node session_RmsnormEnc_-8396 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b581 -> b1232
node dispatch2892 micro raw57 b580 b1423 b1424 b1232 -> b580
node dispatch2893 micro raw73 b580 -> b580
node dispatch2894 micro raw74 b580 -> b580
node dispatch2895 micro raw75 b580 -> b580
node dispatch2896 micro raw62 b584 b1427 b1428 b580_4803072_32768 -> b584
node session_VecOp_-8397 session sessionVecOp:op=add,n=4096 b1231 b584 -> b1231
node session_RmsnormEnc_-8398 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b599 -> b1232
node dispatch2897 micro raw57 b598 b1439 b1440 b1232 -> b598
node dispatch2898 micro raw73 b598 -> b598
node dispatch2899 micro raw74 b598 -> b598
node dispatch2900 micro raw75 b598 -> b598
node dispatch2901 micro raw62 b602 b1443 b1444 b598_4803072_32768 -> b602
node session_VecOp_-8399 session sessionVecOp:op=add,n=4096 b1231 b602 -> b1231
node session_RmsnormEnc_-8400 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b617 -> b1232
node dispatch2902 micro raw57 b616 b1447 b1448 b1232 -> b616
node dispatch2903 micro raw73 b616 -> b616
node dispatch2904 micro raw74 b616 -> b616
node dispatch2905 micro raw75 b616 -> b616
node dispatch2906 micro raw62 b620 b1451 b1452 b616_4803072_32768 -> b620
node session_VecOp_-8401 session sessionVecOp:op=add,n=4096 b1231 b620 -> b1231
node session_RmsnormEnc_-8402 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b635 -> b1232
node dispatch2907 micro raw57 b634 b1455 b1456 b1232 -> b634
node dispatch2908 micro raw73 b634 -> b634
node dispatch2909 micro raw74 b634 -> b634
node dispatch2910 micro raw75 b634 -> b634
node dispatch2911 micro raw62 b638 b1459 b1460 b634_4803072_32768 -> b638
node session_VecOp_-8403 session sessionVecOp:op=add,n=4096 b1231 b638 -> b1231
node session_RmsnormEnc_-8404 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b653 -> b1232
node dispatch2912 micro raw57 b652 b1463 b1464 b1232 -> b652
node dispatch2913 micro raw73 b652 -> b652
node dispatch2914 micro raw74 b652 -> b652
node dispatch2915 micro raw75 b652 -> b652
node dispatch2916 micro raw62 b656 b1467 b1468 b652_4803072_32768 -> b656
node session_VecOp_-8405 session sessionVecOp:op=add,n=4096 b1231 b656 -> b1231
node session_RmsnormEnc_-8406 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b671 -> b1232
node dispatch2917 micro raw57 b670 b1471 b1472 b1232 -> b670
node dispatch2918 micro raw73 b670 -> b670
node dispatch2919 micro raw74 b670 -> b670
node dispatch2920 micro raw75 b670 -> b670
node dispatch2921 micro raw62 b674 b1475 b1476 b670_4803072_32768 -> b674
node session_VecOp_-8407 session sessionVecOp:op=add,n=4096 b1231 b674 -> b1231
node session_RmsnormEnc_-8408 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b689 -> b1232
node dispatch2922 micro raw57 b688 b1479 b1480 b1232 -> b688
node dispatch2923 micro raw73 b688 -> b688
node dispatch2924 micro raw74 b688 -> b688
node dispatch2925 micro raw75 b688 -> b688
node dispatch2926 micro raw62 b692 b1483 b1484 b688_4803072_32768 -> b692
node session_VecOp_-8409 session sessionVecOp:op=add,n=4096 b1231 b692 -> b1231
node session_RmsnormEnc_-8410 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b707 -> b1232
node dispatch2927 micro raw57 b706 b1487 b1488 b1232 -> b706
node dispatch2928 micro raw73 b706 -> b706
node dispatch2929 micro raw74 b706 -> b706
node dispatch2930 micro raw75 b706 -> b706
node dispatch2931 micro raw62 b710 b1491 b1492 b706_4803072_32768 -> b710
node session_VecOp_-8411 session sessionVecOp:op=add,n=4096 b1231 b710 -> b1231
node session_RmsnormEnc_-8412 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b725 -> b1232
node dispatch2932 micro raw57 b724 b1495 b1496 b1232 -> b724
node dispatch2933 micro raw73 b724 -> b724
node dispatch2934 micro raw74 b724 -> b724
node dispatch2935 micro raw75 b724 -> b724
node dispatch2936 micro raw62 b728 b1499 b1500 b724_4803072_32768 -> b728
node session_VecOp_-8413 session sessionVecOp:op=add,n=4096 b1231 b728 -> b1231
node session_RmsnormEnc_-8414 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b743 -> b1232
node dispatch2937 micro raw57 b742 b1503 b1504 b1232 -> b742
node dispatch2938 micro raw73 b742 -> b742
node dispatch2939 micro raw74 b742 -> b742
node dispatch2940 micro raw75 b742 -> b742
node dispatch2941 micro raw62 b746 b1507 b1508 b742_4803072_32768 -> b746
node session_VecOp_-8415 session sessionVecOp:op=add,n=4096 b1231 b746 -> b1231
node session_RmsnormEnc_-8416 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b770 -> b1232
node dispatch2942 micro raw76 b769 b765 b1232 -> b769
node dispatch2943 micro raw77 b769 b766 -> b769
node dispatch2944 micro raw78 b769 -> b769
node dispatch2945 micro raw79 b164 b767 b769_49152_557056 -> b164
node dispatch2946 micro raw80 b164 b768 -> b164
node dispatch2947 micro raw57 b760 b1511 b1512 b1232 -> b760
node dispatch2948 micro raw73 b760 -> b760
node dispatch2949 micro raw74 b760 -> b760
node dispatch2950 micro raw75 b760 -> b760
node dispatch2951 micro raw62 b764 b1515 b1516 b760_4803072_32768 -> b764
node session_CopyInto_-8417 session sessionCopyInto:srcOff=0,dstOff=0,n=4096 b764 -> b777
node session_CopyInto_-8418 session sessionCopyInto:srcOff=0,dstOff=4096,n=4096 b164 -> b777
node dispatch2952 micro raw81 b778 b775 b777 -> b778
node dispatch2953 micro raw80 b778 b776 -> b778
node session_Activation_-8419 session sessionActivation:op=sigmoid,n=4096,p=h00000000 b778 -> b779
node session_VecOp_-8420 session sessionVecOp:op=sub,n=4096 b781 b779 -> b780
node session_VecOp_-8421 session sessionVecOp:op=mul,n=4096 b779 b764 -> b764
node session_VecOp_-8422 session sessionVecOp:op=mul,n=4096 b780 b164 -> b164
node session_VecOp_-8423 session sessionVecOp:op=add,n=4096 b764 b164 -> b764
node session_VecOp_-8424 session sessionVecOp:op=add,n=4096 b1231 b764 -> b1231
node session_RmsnormEnc_-8425 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b796 -> b1232
node dispatch2954 micro raw57 b795 b1527 b1528 b1232 -> b795
node dispatch2955 micro raw73 b795 -> b795
node dispatch2956 micro raw74 b795 -> b795
node dispatch2957 micro raw75 b795 -> b795
node dispatch2958 micro raw62 b799 b1531 b1532 b795_4803072_32768 -> b799
node session_VecOp_-8426 session sessionVecOp:op=add,n=4096 b1231 b799 -> b1231
node session_RmsnormEnc_-8427 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b814 -> b1232
node dispatch2959 micro raw57 b813 b1535 b1536 b1232 -> b813
node dispatch2960 micro raw73 b813 -> b813
node dispatch2961 micro raw74 b813 -> b813
node dispatch2962 micro raw75 b813 -> b813
node dispatch2963 micro raw62 b817 b1539 b1540 b813_4803072_32768 -> b817
node session_VecOp_-8428 session sessionVecOp:op=add,n=4096 b1231 b817 -> b1231
node session_RmsnormEnc_-8429 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b832 -> b1232
node dispatch2964 micro raw57 b831 b1543 b1544 b1232 -> b831
node dispatch2965 micro raw73 b831 -> b831
node dispatch2966 micro raw74 b831 -> b831
node dispatch2967 micro raw75 b831 -> b831
node dispatch2968 micro raw62 b835 b1547 b1548 b831_4803072_32768 -> b835
node session_VecOp_-8430 session sessionVecOp:op=add,n=4096 b1231 b835 -> b1231
node session_RmsnormEnc_-8431 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b850 -> b1232
node dispatch2969 micro raw57 b849 b1551 b1552 b1232 -> b849
node dispatch2970 micro raw73 b849 -> b849
node dispatch2971 micro raw74 b849 -> b849
node dispatch2972 micro raw75 b849 -> b849
node dispatch2973 micro raw62 b853 b1555 b1556 b849_4803072_32768 -> b853
node session_VecOp_-8432 session sessionVecOp:op=add,n=4096 b1231 b853 -> b1231
node session_RmsnormEnc_-8433 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b868 -> b1232
node dispatch2974 micro raw57 b867 b1559 b1560 b1232 -> b867
node dispatch2975 micro raw73 b867 -> b867
node dispatch2976 micro raw74 b867 -> b867
node dispatch2977 micro raw75 b867 -> b867
node dispatch2978 micro raw62 b871 b1563 b1564 b867_4803072_32768 -> b871
node session_VecOp_-8434 session sessionVecOp:op=add,n=4096 b1231 b871 -> b1231
node session_RmsnormEnc_-8435 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b886 -> b1232
node dispatch2979 micro raw57 b885 b1567 b1568 b1232 -> b885
node dispatch2980 micro raw73 b885 -> b885
node dispatch2981 micro raw74 b885 -> b885
node dispatch2982 micro raw75 b885 -> b885
node dispatch2983 micro raw62 b889 b1571 b1572 b885_4803072_32768 -> b889
node session_VecOp_-8436 session sessionVecOp:op=add,n=4096 b1231 b889 -> b1231
node session_RmsnormEnc_-8437 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b904 -> b1232
node dispatch2984 micro raw57 b903 b1575 b1576 b1232 -> b903
node dispatch2985 micro raw73 b903 -> b903
node dispatch2986 micro raw74 b903 -> b903
node dispatch2987 micro raw75 b903 -> b903
node dispatch2988 micro raw62 b907 b1579 b1580 b903_4803072_32768 -> b907
node session_VecOp_-8438 session sessionVecOp:op=add,n=4096 b1231 b907 -> b1231
node session_RmsnormEnc_-8439 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b922 -> b1232
node dispatch2989 micro raw57 b921 b1583 b1584 b1232 -> b921
node dispatch2990 micro raw73 b921 -> b921
node dispatch2991 micro raw74 b921 -> b921
node dispatch2992 micro raw75 b921 -> b921
node dispatch2993 micro raw62 b925 b1587 b1588 b921_4803072_32768 -> b925
node session_VecOp_-8440 session sessionVecOp:op=add,n=4096 b1231 b925 -> b1231
node session_RmsnormEnc_-8441 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b940 -> b1232
node dispatch2994 micro raw57 b939 b1591 b1592 b1232 -> b939
node dispatch2995 micro raw73 b939 -> b939
node dispatch2996 micro raw74 b939 -> b939
node dispatch2997 micro raw75 b939 -> b939
node dispatch2998 micro raw62 b943 b1595 b1596 b939_4803072_32768 -> b943
node session_VecOp_-8442 session sessionVecOp:op=add,n=4096 b1231 b943 -> b1231
node session_RmsnormEnc_-8443 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b958 -> b1232
node dispatch2999 micro raw57 b957 b1599 b1600 b1232 -> b957
node dispatch3000 micro raw73 b957 -> b957
node dispatch3001 micro raw74 b957 -> b957
node dispatch3002 micro raw75 b957 -> b957
node dispatch3003 micro raw62 b961 b1603 b1604 b957_4803072_32768 -> b961
node session_VecOp_-8444 session sessionVecOp:op=add,n=4096 b1231 b961 -> b1231
node session_RmsnormEnc_-8445 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b976 -> b1232
node dispatch3004 micro raw57 b975 b1615 b1616 b1232 -> b975
node dispatch3005 micro raw73 b975 -> b975
node dispatch3006 micro raw74 b975 -> b975
node dispatch3007 micro raw75 b975 -> b975
node dispatch3008 micro raw62 b979 b1619 b1620 b975_4803072_32768 -> b979
node session_VecOp_-8446 session sessionVecOp:op=add,n=4096 b1231 b979 -> b1231
node session_RmsnormEnc_-8447 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b994 -> b1232
node dispatch3009 micro raw57 b993 b1623 b1624 b1232 -> b993
node dispatch3010 micro raw73 b993 -> b993
node dispatch3011 micro raw74 b993 -> b993
node dispatch3012 micro raw75 b993 -> b993
node dispatch3013 micro raw62 b997 b1627 b1628 b993_4803072_32768 -> b997
node session_VecOp_-8448 session sessionVecOp:op=add,n=4096 b1231 b997 -> b1231
node session_RmsnormEnc_-8449 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1012 -> b1232
node dispatch3014 micro raw57 b1011 b1631 b1632 b1232 -> b1011
node dispatch3015 micro raw73 b1011 -> b1011
node dispatch3016 micro raw74 b1011 -> b1011
node dispatch3017 micro raw75 b1011 -> b1011
node dispatch3018 micro raw62 b1015 b1635 b1636 b1011_4803072_32768 -> b1015
node session_VecOp_-8450 session sessionVecOp:op=add,n=4096 b1231 b1015 -> b1231
node session_RmsnormEnc_-8451 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1030 -> b1232
node dispatch3019 micro raw57 b1029 b1639 b1640 b1232 -> b1029
node dispatch3020 micro raw73 b1029 -> b1029
node dispatch3021 micro raw74 b1029 -> b1029
node dispatch3022 micro raw75 b1029 -> b1029
node dispatch3023 micro raw62 b1033 b1643 b1644 b1029_4803072_32768 -> b1033
node session_VecOp_-8452 session sessionVecOp:op=add,n=4096 b1231 b1033 -> b1231
node session_RmsnormEnc_-8453 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1048 -> b1232
node dispatch3024 micro raw57 b1047 b1647 b1648 b1232 -> b1047
node dispatch3025 micro raw73 b1047 -> b1047
node dispatch3026 micro raw74 b1047 -> b1047
node dispatch3027 micro raw75 b1047 -> b1047
node dispatch3028 micro raw62 b1051 b1651 b1652 b1047_4803072_32768 -> b1051
node session_VecOp_-8454 session sessionVecOp:op=add,n=4096 b1231 b1051 -> b1231
node session_RmsnormEnc_-8455 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1075 -> b1232
node dispatch3029 micro raw76 b1074 b1070 b1232 -> b1074
node dispatch3030 micro raw77 b1074 b1071 -> b1074
node dispatch3031 micro raw78 b1074 -> b1074
node dispatch3032 micro raw79 b164 b1072 b1074_49152_557056 -> b164
node dispatch3033 micro raw80 b164 b1073 -> b164
node dispatch3034 micro raw57 b1065 b1655 b1656 b1232 -> b1065
node dispatch3035 micro raw73 b1065 -> b1065
node dispatch3036 micro raw74 b1065 -> b1065
node dispatch3037 micro raw75 b1065 -> b1065
node dispatch3038 micro raw62 b1069 b1659 b1660 b1065_4803072_32768 -> b1069
node session_CopyInto_-8456 session sessionCopyInto:srcOff=0,dstOff=0,n=4096 b1069 -> b1082
node session_CopyInto_-8457 session sessionCopyInto:srcOff=0,dstOff=4096,n=4096 b164 -> b1082
node dispatch3039 micro raw81 b1083 b1080 b1082 -> b1083
node dispatch3040 micro raw80 b1083 b1081 -> b1083
node session_Activation_-8458 session sessionActivation:op=sigmoid,n=4096,p=h00000000 b1083 -> b1084
node session_VecOp_-8459 session sessionVecOp:op=sub,n=4096 b1086 b1084 -> b1085
node session_VecOp_-8460 session sessionVecOp:op=mul,n=4096 b1084 b1069 -> b1069
node session_VecOp_-8461 session sessionVecOp:op=mul,n=4096 b1085 b164 -> b164
node session_VecOp_-8462 session sessionVecOp:op=add,n=4096 b1069 b164 -> b1069
node session_VecOp_-8463 session sessionVecOp:op=add,n=4096 b1231 b1069 -> b1231
node session_RmsnormEnc_-8464 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1101 -> b1232
node dispatch3041 micro raw57 b1100 b1663 b1664 b1232 -> b1100
node dispatch3042 micro raw73 b1100 -> b1100
node dispatch3043 micro raw74 b1100 -> b1100
node dispatch3044 micro raw75 b1100 -> b1100
node dispatch3045 micro raw62 b1104 b1667 b1668 b1100_4803072_32768 -> b1104
node session_VecOp_-8465 session sessionVecOp:op=add,n=4096 b1231 b1104 -> b1231
node session_RmsnormEnc_-8466 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1119 -> b1232
node dispatch3046 micro raw57 b1118 b1671 b1672 b1232 -> b1118
node dispatch3047 micro raw73 b1118 -> b1118
node dispatch3048 micro raw74 b1118 -> b1118
node dispatch3049 micro raw75 b1118 -> b1118
node dispatch3050 micro raw62 b1122 b1675 b1676 b1118_4803072_32768 -> b1122
node session_VecOp_-8467 session sessionVecOp:op=add,n=4096 b1231 b1122 -> b1231
node session_RmsnormEnc_-8468 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1137 -> b1232
node dispatch3051 micro raw57 b1136 b1679 b1680 b1232 -> b1136
node dispatch3052 micro raw73 b1136 -> b1136
node dispatch3053 micro raw74 b1136 -> b1136
node dispatch3054 micro raw75 b1136 -> b1136
node dispatch3055 micro raw62 b1140 b1683 b1684 b1136_4803072_32768 -> b1140
node session_VecOp_-8469 session sessionVecOp:op=add,n=4096 b1231 b1140 -> b1231
node session_RmsnormEnc_-8470 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1155 -> b1232
node dispatch3056 micro raw57 b1154 b1687 b1688 b1232 -> b1154
node dispatch3057 micro raw73 b1154 -> b1154
node dispatch3058 micro raw74 b1154 -> b1154
node dispatch3059 micro raw75 b1154 -> b1154
node dispatch3060 micro raw62 b1158 b1691 b1692 b1154_4803072_32768 -> b1158
node session_VecOp_-8471 session sessionVecOp:op=add,n=4096 b1231 b1158 -> b1231
node session_RmsnormEnc_-8472 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1173 -> b1232
node dispatch3061 micro raw57 b1172 b1703 b1704 b1232 -> b1172
node dispatch3062 micro raw73 b1172 -> b1172
node dispatch3063 micro raw74 b1172 -> b1172
node dispatch3064 micro raw75 b1172 -> b1172
node dispatch3065 micro raw62 b1176 b1707 b1708 b1172_4803072_32768 -> b1176
node session_VecOp_-8473 session sessionVecOp:op=add,n=4096 b1231 b1176 -> b1231
node session_RmsnormEnc_-8474 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1191 -> b1232
node dispatch3066 micro raw57 b1190 b1711 b1712 b1232 -> b1190
node dispatch3067 micro raw73 b1190 -> b1190
node dispatch3068 micro raw74 b1190 -> b1190
node dispatch3069 micro raw75 b1190 -> b1190
node dispatch3070 micro raw62 b1194 b1715 b1716 b1190_4803072_32768 -> b1194
node session_VecOp_-8475 session sessionVecOp:op=add,n=4096 b1231 b1194 -> b1231
node session_RmsnormEnc_-8476 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1209 -> b1232
node dispatch3071 micro raw57 b1208 b1719 b1720 b1232 -> b1208
node dispatch3072 micro raw73 b1208 -> b1208
node dispatch3073 micro raw74 b1208 -> b1208
node dispatch3074 micro raw75 b1208 -> b1208
node dispatch3075 micro raw62 b1212 b1723 b1724 b1208_4803072_32768 -> b1212
node session_VecOp_-8477 session sessionVecOp:op=add,n=4096 b1231 b1212 -> b1231
node session_RmsnormEnc_-8478 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1227 -> b1232
node dispatch3076 micro raw57 b1226 b1727 b1728 b1232 -> b1226
node dispatch3077 micro raw73 b1226 -> b1226
node dispatch3078 micro raw74 b1226 -> b1226
node dispatch3079 micro raw75 b1226 -> b1226
node dispatch3080 micro raw62 b1230 b1731 b1732 b1226_4803072_32768 -> b1230
node session_VecOp_-8479 session sessionVecOp:op=add,n=4096 b1231 b1230 -> b1231
node session_RmsnormEnc_-8480 session sessionRmsnormEnc:rows=1,cols=4096,eps=h3727c5ac b1231 b1235 -> b1233
node dispatch3081 micro raw82 b1234 b1237 b1238 b1233 -> b1234

