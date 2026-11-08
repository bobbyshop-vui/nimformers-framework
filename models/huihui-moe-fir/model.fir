fir 1
model huihui-moe
config config.json
runtime graph
layout script
script_abi 1
epsilon rms_norm_eps
require feature weights_bf16
sym hidden = hidden_size
sym layers = num_hidden_layers
sym positions = max_position_embeddings
sym vocab = vocab_size
buf token f32 1 input:token
buf position f32 1 input:position
weight b0 raw capture.sample1-21bee6bfb9d00bef tensor [155582464]
init b0 trace.data 0 622329856 sample1-21bee6bfb9d00bef
weight b1 raw capture.sample1-21bee6bfb9d00bef tensor [155582464]
init b1 trace.data 0 622329856 sample1-21bee6bfb9d00bef
weight b2 raw capture.sample1-c252d9adf7cfec3c tensor [1024]
init b2 trace.data 622329856 4096 sample1-c252d9adf7cfec3c
weight b3 raw capture.sample1-fcd5ea0e44ebd499 tensor [1024]
init b3 trace.data 622333952 4096 sample1-fcd5ea0e44ebd499
weight b4 raw capture.sample1-2937187d7bf6f69d tensor [1024]
init b4 trace.data 622338048 4096 sample1-2937187d7bf6f69d
weight b5 raw capture.sample1-bebb0f4cdb57d7c9 tensor [2097152]
init b5 trace.data 622342144 8388608 sample1-bebb0f4cdb57d7c9
weight b6 raw capture.sample1-66a8ed5cce38df1d tensor [1048576]
init b6 trace.data 630730752 4194304 sample1-66a8ed5cce38df1d
weight b7 raw capture.sample1-5f093b30914dcab2 tensor [1048576]
init b7 trace.data 634925056 4194304 sample1-5f093b30914dcab2
weight b8 raw capture.sample1-07eb1eec5f022bf8 tensor [2097152]
init b8 trace.data 639119360 8388608 sample1-07eb1eec5f022bf8
weight b9 raw capture.sample1-eab2e208d012f4b9 tensor [128]
init b9 trace.data 647507968 512 sample1-eab2e208d012f4b9
weight b10 raw capture.sample1-30f55a4a6cccbbb8 tensor [128]
init b10 trace.data 647508480 512 sample1-30f55a4a6cccbbb8
weight b11 raw capture.sample1-2cd79d81f0e750d8 tensor [3072]
init b11 trace.data 647508992 12288 sample1-2cd79d81f0e750d8
weight b12 raw capture.sample1-85f4c644ba1b9791 tensor [18874368]
init b12 trace.data 647521280 75497472 sample1-85f4c644ba1b9791
weight b13 raw capture.sample1-0fb19eb4f1f30ff6 tensor [9437184]
init b13 trace.data 723018752 37748736 sample1-0fb19eb4f1f30ff6
weight b14 raw capture.sample1-289be76bb5ba9ccf tensor [1024]
init b14 trace.data 760767488 4096 sample1-289be76bb5ba9ccf
weight b15 raw capture.sample1-f16dca38a9295ee5 tensor [1024]
init b15 trace.data 760771584 4096 sample1-f16dca38a9295ee5
weight b16 raw capture.sample1-3e3399470f56cc23 tensor [2097152]
init b16 trace.data 760775680 8388608 sample1-3e3399470f56cc23
weight b17 raw capture.sample1-32514a25a6093c1c tensor [1048576]
init b17 trace.data 769164288 4194304 sample1-32514a25a6093c1c
weight b18 raw capture.sample1-bb1249a382af4d8c tensor [1048576]
init b18 trace.data 773358592 4194304 sample1-bb1249a382af4d8c
weight b19 raw capture.sample1-67fa61b356882a18 tensor [2097152]
init b19 trace.data 777552896 8388608 sample1-67fa61b356882a18
weight b20 raw capture.sample1-435c76908e91d89b tensor [128]
init b20 trace.data 785941504 512 sample1-435c76908e91d89b
weight b21 raw capture.sample1-3c2b5b461cb6dc40 tensor [128]
init b21 trace.data 785942016 512 sample1-3c2b5b461cb6dc40
weight b22 raw capture.sample1-ae88ab0019c584c4 tensor [3072]
init b22 trace.data 785942528 12288 sample1-ae88ab0019c584c4
weight b23 raw capture.sample1-9ce7c52868f7377a tensor [18874368]
init b23 trace.data 785954816 75497472 sample1-9ce7c52868f7377a
weight b24 raw capture.sample1-0f4faf1001d86364 tensor [9437184]
init b24 trace.data 861452288 37748736 sample1-0f4faf1001d86364
weight b25 raw capture.sample1-f63136c06de70612 tensor [1024]
init b25 trace.data 899201024 4096 sample1-f63136c06de70612
weight b26 raw capture.sample1-de9ace6775caa575 tensor [1024]
init b26 trace.data 899205120 4096 sample1-de9ace6775caa575
weight b27 raw capture.sample1-fe8f8c542661617a tensor [2097152]
init b27 trace.data 899209216 8388608 sample1-fe8f8c542661617a
weight b28 raw capture.sample1-5b5a76cdd69bc1b8 tensor [1048576]
init b28 trace.data 907597824 4194304 sample1-5b5a76cdd69bc1b8
weight b29 raw capture.sample1-13c0c39750efd6f2 tensor [1048576]
init b29 trace.data 911792128 4194304 sample1-13c0c39750efd6f2
weight b30 raw capture.sample1-46b1211349272b33 tensor [2097152]
init b30 trace.data 915986432 8388608 sample1-46b1211349272b33
weight b31 raw capture.sample1-a053f6d1e418f177 tensor [128]
init b31 trace.data 924375040 512 sample1-a053f6d1e418f177
weight b32 raw capture.sample1-2c43ab4d2aef9671 tensor [128]
init b32 trace.data 924375552 512 sample1-2c43ab4d2aef9671
weight b33 raw capture.sample1-f5d8d3e5e2b82765 tensor [3072]
init b33 trace.data 924376064 12288 sample1-f5d8d3e5e2b82765
weight b34 raw capture.sample1-61e20bf2bd8728ff tensor [18874368]
init b34 trace.data 924388352 75497472 sample1-61e20bf2bd8728ff
weight b35 raw capture.sample1-19b1225d5f005812 tensor [9437184]
init b35 trace.data 999885824 37748736 sample1-19b1225d5f005812
weight b36 raw capture.sample1-bc3ec3e1a6966fdb tensor [1024]
init b36 trace.data 1037634560 4096 sample1-bc3ec3e1a6966fdb
weight b37 raw capture.sample1-228bbbcac96895bb tensor [1024]
init b37 trace.data 1037638656 4096 sample1-228bbbcac96895bb
weight b38 raw capture.sample1-5ec23bd292b1d300 tensor [2097152]
init b38 trace.data 1037642752 8388608 sample1-5ec23bd292b1d300
weight b39 raw capture.sample1-fd6bc29bcf821241 tensor [1048576]
init b39 trace.data 1046031360 4194304 sample1-fd6bc29bcf821241
weight b40 raw capture.sample1-b1ddd89cfc3ad6a7 tensor [1048576]
init b40 trace.data 1050225664 4194304 sample1-b1ddd89cfc3ad6a7
weight b41 raw capture.sample1-931ffbc1fbf76db4 tensor [2097152]
init b41 trace.data 1054419968 8388608 sample1-931ffbc1fbf76db4
weight b42 raw capture.sample1-a309ef086aedd3ee tensor [128]
init b42 trace.data 1062808576 512 sample1-a309ef086aedd3ee
weight b43 raw capture.sample1-23ae36cdd9a13af3 tensor [128]
init b43 trace.data 1062809088 512 sample1-23ae36cdd9a13af3
weight b44 raw capture.sample1-47ed3b34c0769746 tensor [3072]
init b44 trace.data 1062809600 12288 sample1-47ed3b34c0769746
weight b45 raw capture.sample1-bbfaa8ff96592230 tensor [18874368]
init b45 trace.data 1062821888 75497472 sample1-bbfaa8ff96592230
weight b46 raw capture.sample1-3b1e3c66c6c1ace5 tensor [9437184]
init b46 trace.data 1138319360 37748736 sample1-3b1e3c66c6c1ace5
weight b47 raw capture.sample1-3fb82757a66f3e98 tensor [1024]
init b47 trace.data 1176068096 4096 sample1-3fb82757a66f3e98
weight b48 raw capture.sample1-5a3fd65d0f5df679 tensor [1024]
init b48 trace.data 1176072192 4096 sample1-5a3fd65d0f5df679
weight b49 raw capture.sample1-18addebcf7226e58 tensor [2097152]
init b49 trace.data 1176076288 8388608 sample1-18addebcf7226e58
weight b50 raw capture.sample1-4ba90613f4cc1dd9 tensor [1048576]
init b50 trace.data 1184464896 4194304 sample1-4ba90613f4cc1dd9
weight b51 raw capture.sample1-54d7bf4121b94feb tensor [1048576]
init b51 trace.data 1188659200 4194304 sample1-54d7bf4121b94feb
weight b52 raw capture.sample1-86d6ca1e186bb498 tensor [2097152]
init b52 trace.data 1192853504 8388608 sample1-86d6ca1e186bb498
weight b53 raw capture.sample1-9f466bf2f355c219 tensor [128]
init b53 trace.data 1201242112 512 sample1-9f466bf2f355c219
weight b54 raw capture.sample1-c31c4d5e1c967c11 tensor [128]
init b54 trace.data 1201242624 512 sample1-c31c4d5e1c967c11
weight b55 raw capture.sample1-5980e13ad7afbd07 tensor [3072]
init b55 trace.data 1201243136 12288 sample1-5980e13ad7afbd07
weight b56 raw capture.sample1-be18a88149463c38 tensor [18874368]
init b56 trace.data 1201255424 75497472 sample1-be18a88149463c38
weight b57 raw capture.sample1-fdeab0d3f240e98e tensor [9437184]
init b57 trace.data 1276752896 37748736 sample1-fdeab0d3f240e98e
weight b58 raw capture.sample1-f76f9cc781b942ca tensor [1024]
init b58 trace.data 1314501632 4096 sample1-f76f9cc781b942ca
weight b59 raw capture.sample1-e5107971850a9b78 tensor [1024]
init b59 trace.data 1314505728 4096 sample1-e5107971850a9b78
weight b60 raw capture.sample1-61ad141708796deb tensor [2097152]
init b60 trace.data 1314509824 8388608 sample1-61ad141708796deb
weight b61 raw capture.sample1-c9e13067dd04cd40 tensor [1048576]
init b61 trace.data 1322898432 4194304 sample1-c9e13067dd04cd40
weight b62 raw capture.sample1-dd7f248bd2eeb10f tensor [1048576]
init b62 trace.data 1327092736 4194304 sample1-dd7f248bd2eeb10f
weight b63 raw capture.sample1-d5cdb6615f7cdba6 tensor [2097152]
init b63 trace.data 1331287040 8388608 sample1-d5cdb6615f7cdba6
weight b64 raw capture.sample1-8f692d95fddb8d36 tensor [128]
init b64 trace.data 1339675648 512 sample1-8f692d95fddb8d36
weight b65 raw capture.sample1-927f545e460057e7 tensor [128]
init b65 trace.data 1339676160 512 sample1-927f545e460057e7
weight b66 raw capture.sample1-b93f0da7689959ee tensor [3072]
init b66 trace.data 1339676672 12288 sample1-b93f0da7689959ee
weight b67 raw capture.sample1-346b1b31dc9cead8 tensor [18874368]
init b67 trace.data 1339688960 75497472 sample1-346b1b31dc9cead8
weight b68 raw capture.sample1-c8129dc0b2c2efe8 tensor [9437184]
init b68 trace.data 1415186432 37748736 sample1-c8129dc0b2c2efe8
weight b69 raw capture.sample1-fa84456b89f58de8 tensor [1024]
init b69 trace.data 1452935168 4096 sample1-fa84456b89f58de8
weight b70 raw capture.sample1-acb9cac90ecf75b0 tensor [1024]
init b70 trace.data 1452939264 4096 sample1-acb9cac90ecf75b0
weight b71 raw capture.sample1-aa906c42fc7aed67 tensor [2097152]
init b71 trace.data 1452943360 8388608 sample1-aa906c42fc7aed67
weight b72 raw capture.sample1-7db413a074b87161 tensor [1048576]
init b72 trace.data 1461331968 4194304 sample1-7db413a074b87161
weight b73 raw capture.sample1-4e3169add6206d9e tensor [1048576]
init b73 trace.data 1465526272 4194304 sample1-4e3169add6206d9e
weight b74 raw capture.sample1-73512b3316f6a4ba tensor [2097152]
init b74 trace.data 1469720576 8388608 sample1-73512b3316f6a4ba
weight b75 raw capture.sample1-6d20401de9cf6d6c tensor [128]
init b75 trace.data 1478109184 512 sample1-6d20401de9cf6d6c
weight b76 raw capture.sample1-e3d33ae8eafaec76 tensor [128]
init b76 trace.data 1478109696 512 sample1-e3d33ae8eafaec76
weight b77 raw capture.sample1-f87dff59e57bbd4c tensor [3072]
init b77 trace.data 1478110208 12288 sample1-f87dff59e57bbd4c
weight b78 raw capture.sample1-fe7b9e6ab37765d4 tensor [18874368]
init b78 trace.data 1478122496 75497472 sample1-fe7b9e6ab37765d4
weight b79 raw capture.sample1-eb8af7ac0c34daee tensor [9437184]
init b79 trace.data 1553619968 37748736 sample1-eb8af7ac0c34daee
weight b80 raw capture.sample1-7e80c66403aaecc5 tensor [1024]
init b80 trace.data 1591368704 4096 sample1-7e80c66403aaecc5
weight b81 raw capture.sample1-6037fa00449d6d0a tensor [1024]
init b81 trace.data 1591372800 4096 sample1-6037fa00449d6d0a
weight b82 raw capture.sample1-2c68f5b66527fd45 tensor [2097152]
init b82 trace.data 1591376896 8388608 sample1-2c68f5b66527fd45
weight b83 raw capture.sample1-36831ad24b9a4187 tensor [1048576]
init b83 trace.data 1599765504 4194304 sample1-36831ad24b9a4187
weight b84 raw capture.sample1-ce24298801da9de1 tensor [1048576]
init b84 trace.data 1603959808 4194304 sample1-ce24298801da9de1
weight b85 raw capture.sample1-fb1795ed9f6c481f tensor [2097152]
init b85 trace.data 1608154112 8388608 sample1-fb1795ed9f6c481f
weight b86 raw capture.sample1-4a6160f8b3f08093 tensor [128]
init b86 trace.data 1616542720 512 sample1-4a6160f8b3f08093
weight b87 raw capture.sample1-7ec444fad50012cb tensor [128]
init b87 trace.data 1616543232 512 sample1-7ec444fad50012cb
weight b88 raw capture.sample1-34a442fdb6201190 tensor [3072]
init b88 trace.data 1616543744 12288 sample1-34a442fdb6201190
weight b89 raw capture.sample1-7553893c4d8cfe19 tensor [18874368]
init b89 trace.data 1616556032 75497472 sample1-7553893c4d8cfe19
weight b90 raw capture.sample1-e598f72ba8f73376 tensor [9437184]
init b90 trace.data 1692053504 37748736 sample1-e598f72ba8f73376
weight b91 raw capture.sample1-75a1640c937f1a4b tensor [1024]
init b91 trace.data 1729802240 4096 sample1-75a1640c937f1a4b
weight b92 raw capture.sample1-ee948f34fea7129a tensor [1024]
init b92 trace.data 1729806336 4096 sample1-ee948f34fea7129a
weight b93 raw capture.sample1-defe2fba189c684b tensor [2097152]
init b93 trace.data 1729810432 8388608 sample1-defe2fba189c684b
weight b94 raw capture.sample1-e6a8a41935747d61 tensor [1048576]
init b94 trace.data 1738199040 4194304 sample1-e6a8a41935747d61
weight b95 raw capture.sample1-57094745998e656f tensor [1048576]
init b95 trace.data 1742393344 4194304 sample1-57094745998e656f
weight b96 raw capture.sample1-6ce0d017c848fe53 tensor [2097152]
init b96 trace.data 1746587648 8388608 sample1-6ce0d017c848fe53
weight b97 raw capture.sample1-a43ffed400133f73 tensor [128]
init b97 trace.data 1754976256 512 sample1-a43ffed400133f73
weight b98 raw capture.sample1-922d9b9ba2f1155e tensor [128]
init b98 trace.data 1754976768 512 sample1-922d9b9ba2f1155e
weight b99 raw capture.sample1-04c7c6615b09fcbc tensor [3072]
init b99 trace.data 1754977280 12288 sample1-04c7c6615b09fcbc
weight b100 raw capture.sample1-4b7171eb1366bf54 tensor [18874368]
init b100 trace.data 1754989568 75497472 sample1-4b7171eb1366bf54
weight b101 raw capture.sample1-ef90c8432ffa2544 tensor [9437184]
init b101 trace.data 1830487040 37748736 sample1-ef90c8432ffa2544
weight b102 raw capture.sample1-1029bb03e7bacd21 tensor [1024]
init b102 trace.data 1868235776 4096 sample1-1029bb03e7bacd21
weight b103 raw capture.sample1-1db697e6c4fbcc47 tensor [1024]
init b103 trace.data 1868239872 4096 sample1-1db697e6c4fbcc47
weight b104 raw capture.sample1-d27dbd66ac55ad45 tensor [2097152]
init b104 trace.data 1868243968 8388608 sample1-d27dbd66ac55ad45
weight b105 raw capture.sample1-1e637658f6365400 tensor [1048576]
init b105 trace.data 1876632576 4194304 sample1-1e637658f6365400
weight b106 raw capture.sample1-464ac1513a06b13a tensor [1048576]
init b106 trace.data 1880826880 4194304 sample1-464ac1513a06b13a
weight b107 raw capture.sample1-889157b76f312ad6 tensor [2097152]
init b107 trace.data 1885021184 8388608 sample1-889157b76f312ad6
weight b108 raw capture.sample1-bc9be1c6c472a13e tensor [128]
init b108 trace.data 1893409792 512 sample1-bc9be1c6c472a13e
weight b109 raw capture.sample1-b8e825e730a91f2e tensor [128]
init b109 trace.data 1893410304 512 sample1-b8e825e730a91f2e
weight b110 raw capture.sample1-9050f957f7bccc3a tensor [3072]
init b110 trace.data 1893410816 12288 sample1-9050f957f7bccc3a
weight b111 raw capture.sample1-7cb0e368b220db25 tensor [18874368]
init b111 trace.data 1893423104 75497472 sample1-7cb0e368b220db25
weight b112 raw capture.sample1-3dc9835b5467e7a4 tensor [9437184]
init b112 trace.data 1968920576 37748736 sample1-3dc9835b5467e7a4
weight b113 raw capture.sample1-9781ac431690c99a tensor [1024]
init b113 trace.data 2006669312 4096 sample1-9781ac431690c99a
weight b114 raw capture.sample1-add91d2af0d22a7a tensor [1024]
init b114 trace.data 2006673408 4096 sample1-add91d2af0d22a7a
weight b115 raw capture.sample1-365d97b20a32c763 tensor [2097152]
init b115 trace.data 2006677504 8388608 sample1-365d97b20a32c763
weight b116 raw capture.sample1-21bf2ebf6dbf1ba5 tensor [1048576]
init b116 trace.data 2015066112 4194304 sample1-21bf2ebf6dbf1ba5
weight b117 raw capture.sample1-bdfe3189cdf25ad3 tensor [1048576]
init b117 trace.data 2019260416 4194304 sample1-bdfe3189cdf25ad3
weight b118 raw capture.sample1-2ec61b0c77d764e7 tensor [2097152]
init b118 trace.data 2023454720 8388608 sample1-2ec61b0c77d764e7
weight b119 raw capture.sample1-922de61144c356d6 tensor [128]
init b119 trace.data 2031843328 512 sample1-922de61144c356d6
weight b120 raw capture.sample1-2ad3fd6683001fa6 tensor [128]
init b120 trace.data 2031843840 512 sample1-2ad3fd6683001fa6
weight b121 raw capture.sample1-7435347883d88824 tensor [3072]
init b121 trace.data 2031844352 12288 sample1-7435347883d88824
weight b122 raw capture.sample1-31713e35bb9ba163 tensor [18874368]
init b122 trace.data 2031856640 75497472 sample1-31713e35bb9ba163
weight b123 raw capture.sample1-ce0c3330d7fe0a52 tensor [9437184]
init b123 trace.data 2107354112 37748736 sample1-ce0c3330d7fe0a52
weight b124 raw capture.sample1-38d99fbe9d2109f8 tensor [1024]
init b124 trace.data 2145102848 4096 sample1-38d99fbe9d2109f8
weight b125 raw capture.sample1-31e26899dcbc4744 tensor [1024]
init b125 trace.data 2145106944 4096 sample1-31e26899dcbc4744
weight b126 raw capture.sample1-cd8d85731e286b89 tensor [2097152]
init b126 trace.data 2145111040 8388608 sample1-cd8d85731e286b89
weight b127 raw capture.sample1-b33c92bd038108da tensor [1048576]
init b127 trace.data 2153499648 4194304 sample1-b33c92bd038108da
weight b128 raw capture.sample1-33d0e1585dba9f31 tensor [1048576]
init b128 trace.data 2157693952 4194304 sample1-33d0e1585dba9f31
weight b129 raw capture.sample1-feff4d9189a74efe tensor [2097152]
init b129 trace.data 2161888256 8388608 sample1-feff4d9189a74efe
weight b130 raw capture.sample1-d097c0d22adbdb1e tensor [128]
init b130 trace.data 2170276864 512 sample1-d097c0d22adbdb1e
weight b131 raw capture.sample1-1d1da6e67f9a7f2e tensor [128]
init b131 trace.data 2170277376 512 sample1-1d1da6e67f9a7f2e
weight b132 raw capture.sample1-a72b73e9455c7d4f tensor [3072]
init b132 trace.data 2170277888 12288 sample1-a72b73e9455c7d4f
weight b133 raw capture.sample1-ec20e65ca6c6e9ba tensor [18874368]
init b133 trace.data 2170290176 75497472 sample1-ec20e65ca6c6e9ba
weight b134 raw capture.sample1-f1b81a3334e8b5c3 tensor [9437184]
init b134 trace.data 2245787648 37748736 sample1-f1b81a3334e8b5c3
weight b135 raw capture.sample1-dda7a52a998f6eb3 tensor [1024]
init b135 trace.data 2283536384 4096 sample1-dda7a52a998f6eb3
weight b136 raw capture.sample1-be5a87c434186c4d tensor [1024]
init b136 trace.data 2283540480 4096 sample1-be5a87c434186c4d
weight b137 raw capture.sample1-d64d90589978a4e4 tensor [2097152]
init b137 trace.data 2283544576 8388608 sample1-d64d90589978a4e4
weight b138 raw capture.sample1-fc4ade7b9050bbdd tensor [1048576]
init b138 trace.data 2291933184 4194304 sample1-fc4ade7b9050bbdd
weight b139 raw capture.sample1-bef358ebd1aa448a tensor [1048576]
init b139 trace.data 2296127488 4194304 sample1-bef358ebd1aa448a
weight b140 raw capture.sample1-66bab41f3eb72f67 tensor [2097152]
init b140 trace.data 2300321792 8388608 sample1-66bab41f3eb72f67
weight b141 raw capture.sample1-59af62ea50e927f9 tensor [128]
init b141 trace.data 2308710400 512 sample1-59af62ea50e927f9
weight b142 raw capture.sample1-0199fa1a5f8ae565 tensor [128]
init b142 trace.data 2308710912 512 sample1-0199fa1a5f8ae565
weight b143 raw capture.sample1-fa761250f2d65ff5 tensor [3072]
init b143 trace.data 2308711424 12288 sample1-fa761250f2d65ff5
weight b144 raw capture.sample1-0c9485affdf2514a tensor [18874368]
init b144 trace.data 2308723712 75497472 sample1-0c9485affdf2514a
weight b145 raw capture.sample1-afb32d92acd11a4e tensor [9437184]
init b145 trace.data 2384221184 37748736 sample1-afb32d92acd11a4e
weight b146 raw capture.sample1-e5fd124c2adf5d46 tensor [1024]
init b146 trace.data 2421969920 4096 sample1-e5fd124c2adf5d46
weight b147 raw capture.sample1-52461e60d88e8ff1 tensor [1024]
init b147 trace.data 2421974016 4096 sample1-52461e60d88e8ff1
weight b148 raw capture.sample1-b86baae61cf896fd tensor [2097152]
init b148 trace.data 2421978112 8388608 sample1-b86baae61cf896fd
weight b149 raw capture.sample1-e3a8701df8578d67 tensor [1048576]
init b149 trace.data 2430366720 4194304 sample1-e3a8701df8578d67
weight b150 raw capture.sample1-0046d89b11c555ef tensor [1048576]
init b150 trace.data 2434561024 4194304 sample1-0046d89b11c555ef
weight b151 raw capture.sample1-72cd8af3a706af3b tensor [2097152]
init b151 trace.data 2438755328 8388608 sample1-72cd8af3a706af3b
weight b152 raw capture.sample1-8342b7027985d453 tensor [128]
init b152 trace.data 2447143936 512 sample1-8342b7027985d453
weight b153 raw capture.sample1-2b3b1f48abe09ea0 tensor [128]
init b153 trace.data 2447144448 512 sample1-2b3b1f48abe09ea0
weight b154 raw capture.sample1-48da86b326163e28 tensor [3072]
init b154 trace.data 2447144960 12288 sample1-48da86b326163e28
weight b155 raw capture.sample1-3a0b3b95819089d4 tensor [18874368]
init b155 trace.data 2447157248 75497472 sample1-3a0b3b95819089d4
weight b156 raw capture.sample1-fdd3026a655ca538 tensor [9437184]
init b156 trace.data 2522654720 37748736 sample1-fdd3026a655ca538
weight b157 raw capture.sample1-a5f4919dec8b040a tensor [1024]
init b157 trace.data 2560403456 4096 sample1-a5f4919dec8b040a
weight b158 raw capture.sample1-f3d4e3039d918edf tensor [1024]
init b158 trace.data 2560407552 4096 sample1-f3d4e3039d918edf
weight b159 raw capture.sample1-6e62ed05fdda4a11 tensor [2097152]
init b159 trace.data 2560411648 8388608 sample1-6e62ed05fdda4a11
weight b160 raw capture.sample1-28c2c31453190301 tensor [1048576]
init b160 trace.data 2568800256 4194304 sample1-28c2c31453190301
weight b161 raw capture.sample1-8e10f524e0eef4a7 tensor [1048576]
init b161 trace.data 2572994560 4194304 sample1-8e10f524e0eef4a7
weight b162 raw capture.sample1-54552d8474d679ab tensor [2097152]
init b162 trace.data 2577188864 8388608 sample1-54552d8474d679ab
weight b163 raw capture.sample1-09b33e7e77563cd8 tensor [128]
init b163 trace.data 2585577472 512 sample1-09b33e7e77563cd8
weight b164 raw capture.sample1-61acb918d147460e tensor [128]
init b164 trace.data 2585577984 512 sample1-61acb918d147460e
weight b165 raw capture.sample1-9ded8b2826065fc5 tensor [3072]
init b165 trace.data 2585578496 12288 sample1-9ded8b2826065fc5
weight b166 raw capture.sample1-29219164896c6551 tensor [18874368]
init b166 trace.data 2585590784 75497472 sample1-29219164896c6551
weight b167 raw capture.sample1-d26633490207339d tensor [9437184]
init b167 trace.data 2661088256 37748736 sample1-d26633490207339d
weight b168 raw capture.sample1-e9736def5e39e27f tensor [1024]
init b168 trace.data 2698836992 4096 sample1-e9736def5e39e27f
weight b169 raw capture.sample1-e996d7afca48a628 tensor [1024]
init b169 trace.data 2698841088 4096 sample1-e996d7afca48a628
weight b170 raw capture.sample1-2a3dbc504551b8dd tensor [2097152]
init b170 trace.data 2698845184 8388608 sample1-2a3dbc504551b8dd
weight b171 raw capture.sample1-7c98845f65c43c6e tensor [1048576]
init b171 trace.data 2707233792 4194304 sample1-7c98845f65c43c6e
weight b172 raw capture.sample1-1d9a0bceef72be03 tensor [1048576]
init b172 trace.data 2711428096 4194304 sample1-1d9a0bceef72be03
weight b173 raw capture.sample1-2654aa6c22dcdec2 tensor [2097152]
init b173 trace.data 2715622400 8388608 sample1-2654aa6c22dcdec2
weight b174 raw capture.sample1-1fb7e10ba5eab817 tensor [128]
init b174 trace.data 2724011008 512 sample1-1fb7e10ba5eab817
weight b175 raw capture.sample1-fc717338e8016a45 tensor [128]
init b175 trace.data 2724011520 512 sample1-fc717338e8016a45
weight b176 raw capture.sample1-22be6e47249acacb tensor [3072]
init b176 trace.data 2724012032 12288 sample1-22be6e47249acacb
weight b177 raw capture.sample1-629cdda9bf1be05e tensor [18874368]
init b177 trace.data 2724024320 75497472 sample1-629cdda9bf1be05e
weight b178 raw capture.sample1-7f65c62e77e41b71 tensor [9437184]
init b178 trace.data 2799521792 37748736 sample1-7f65c62e77e41b71
weight b179 raw capture.sample1-438acbf5feda8c79 tensor [1024]
init b179 trace.data 2837270528 4096 sample1-438acbf5feda8c79
weight b180 raw capture.sample1-0f1b667e8919578a tensor [1024]
init b180 trace.data 2837274624 4096 sample1-0f1b667e8919578a
weight b181 raw capture.sample1-2d9dfdeab409d737 tensor [2097152]
init b181 trace.data 2837278720 8388608 sample1-2d9dfdeab409d737
weight b182 raw capture.sample1-56de80dd5d910649 tensor [1048576]
init b182 trace.data 2845667328 4194304 sample1-56de80dd5d910649
weight b183 raw capture.sample1-5f7bc342e33ba5de tensor [1048576]
init b183 trace.data 2849861632 4194304 sample1-5f7bc342e33ba5de
weight b184 raw capture.sample1-eeb444582f669527 tensor [2097152]
init b184 trace.data 2854055936 8388608 sample1-eeb444582f669527
weight b185 raw capture.sample1-b31fea8182e08005 tensor [128]
init b185 trace.data 2862444544 512 sample1-b31fea8182e08005
weight b186 raw capture.sample1-32133e19c12999ab tensor [128]
init b186 trace.data 2862445056 512 sample1-32133e19c12999ab
weight b187 raw capture.sample1-e9d733fcedaa438c tensor [3072]
init b187 trace.data 2862445568 12288 sample1-e9d733fcedaa438c
weight b188 raw capture.sample1-80ea4b1b91b464ef tensor [18874368]
init b188 trace.data 2862457856 75497472 sample1-80ea4b1b91b464ef
weight b189 raw capture.sample1-a7ccd16ed4150529 tensor [9437184]
init b189 trace.data 2937955328 37748736 sample1-a7ccd16ed4150529
weight b190 raw capture.sample1-6a5b5baf8fd9b234 tensor [1024]
init b190 trace.data 2975704064 4096 sample1-6a5b5baf8fd9b234
weight b191 raw capture.sample1-131090d06e3d1578 tensor [1024]
init b191 trace.data 2975708160 4096 sample1-131090d06e3d1578
weight b192 raw capture.sample1-1f2472b066585dd0 tensor [2097152]
init b192 trace.data 2975712256 8388608 sample1-1f2472b066585dd0
weight b193 raw capture.sample1-b10aa3f57e0057ba tensor [1048576]
init b193 trace.data 2984100864 4194304 sample1-b10aa3f57e0057ba
weight b194 raw capture.sample1-d4f32e5821b50c94 tensor [1048576]
init b194 trace.data 2988295168 4194304 sample1-d4f32e5821b50c94
weight b195 raw capture.sample1-e0a91e9882300211 tensor [2097152]
init b195 trace.data 2992489472 8388608 sample1-e0a91e9882300211
weight b196 raw capture.sample1-f56565745c936f15 tensor [128]
init b196 trace.data 3000878080 512 sample1-f56565745c936f15
weight b197 raw capture.sample1-9eeaf862cd1c7b8b tensor [128]
init b197 trace.data 3000878592 512 sample1-9eeaf862cd1c7b8b
weight b198 raw capture.sample1-6c1eb107576f5fd0 tensor [3072]
init b198 trace.data 3000879104 12288 sample1-6c1eb107576f5fd0
weight b199 raw capture.sample1-b76f09845d143310 tensor [18874368]
init b199 trace.data 3000891392 75497472 sample1-b76f09845d143310
weight b200 raw capture.sample1-435e3b280675a651 tensor [9437184]
init b200 trace.data 3076388864 37748736 sample1-435e3b280675a651
weight b201 raw capture.sample1-5bcfabfd685a504b tensor [1024]
init b201 trace.data 3114137600 4096 sample1-5bcfabfd685a504b
weight b202 raw capture.sample1-a8e252312af31321 tensor [1024]
init b202 trace.data 3114141696 4096 sample1-a8e252312af31321
weight b203 raw capture.sample1-1f3d290a371c7ef5 tensor [2097152]
init b203 trace.data 3114145792 8388608 sample1-1f3d290a371c7ef5
weight b204 raw capture.sample1-2341d4d9a858114d tensor [1048576]
init b204 trace.data 3122534400 4194304 sample1-2341d4d9a858114d
weight b205 raw capture.sample1-aeaa90e7273492a7 tensor [1048576]
init b205 trace.data 3126728704 4194304 sample1-aeaa90e7273492a7
weight b206 raw capture.sample1-eb374a6682adc0f0 tensor [2097152]
init b206 trace.data 3130923008 8388608 sample1-eb374a6682adc0f0
weight b207 raw capture.sample1-09503ac352f28436 tensor [128]
init b207 trace.data 3139311616 512 sample1-09503ac352f28436
weight b208 raw capture.sample1-146ed07b9484711c tensor [128]
init b208 trace.data 3139312128 512 sample1-146ed07b9484711c
weight b209 raw capture.sample1-ebf321f271cf5a8e tensor [3072]
init b209 trace.data 3139312640 12288 sample1-ebf321f271cf5a8e
weight b210 raw capture.sample1-aa74a1757da6f320 tensor [18874368]
init b210 trace.data 3139324928 75497472 sample1-aa74a1757da6f320
weight b211 raw capture.sample1-77cede1bb2796438 tensor [9437184]
init b211 trace.data 3214822400 37748736 sample1-77cede1bb2796438
weight b212 raw capture.sample1-940e0d76d4777b44 tensor [1024]
init b212 trace.data 3252571136 4096 sample1-940e0d76d4777b44
weight b213 raw capture.sample1-c29fd10d1457353f tensor [1024]
init b213 trace.data 3252575232 4096 sample1-c29fd10d1457353f
weight b214 raw capture.sample1-ca5e9758205b7e61 tensor [2097152]
init b214 trace.data 3252579328 8388608 sample1-ca5e9758205b7e61
weight b215 raw capture.sample1-291f1089006e19b3 tensor [1048576]
init b215 trace.data 3260967936 4194304 sample1-291f1089006e19b3
weight b216 raw capture.sample1-fd5a30c977697998 tensor [1048576]
init b216 trace.data 3265162240 4194304 sample1-fd5a30c977697998
weight b217 raw capture.sample1-cc5d8f6f714c6efe tensor [2097152]
init b217 trace.data 3269356544 8388608 sample1-cc5d8f6f714c6efe
weight b218 raw capture.sample1-7f5c426969343fc8 tensor [128]
init b218 trace.data 3277745152 512 sample1-7f5c426969343fc8
weight b219 raw capture.sample1-14222731e0158006 tensor [128]
init b219 trace.data 3277745664 512 sample1-14222731e0158006
weight b220 raw capture.sample1-9d2dbb382b2d1d9e tensor [3072]
init b220 trace.data 3277746176 12288 sample1-9d2dbb382b2d1d9e
weight b221 raw capture.sample1-49ad0cf561509a3a tensor [18874368]
init b221 trace.data 3277758464 75497472 sample1-49ad0cf561509a3a
weight b222 raw capture.sample1-704d670454840839 tensor [9437184]
init b222 trace.data 3353255936 37748736 sample1-704d670454840839
weight b223 raw capture.sample1-f2fe4613181d5a1e tensor [1024]
init b223 trace.data 3391004672 4096 sample1-f2fe4613181d5a1e
weight b224 raw capture.sample1-d3bb7937f091580a tensor [1024]
init b224 trace.data 3391008768 4096 sample1-d3bb7937f091580a
weight b225 raw capture.sample1-d9de93ec5aa3f834 tensor [2097152]
init b225 trace.data 3391012864 8388608 sample1-d9de93ec5aa3f834
weight b226 raw capture.sample1-d4f460ed702472e1 tensor [1048576]
init b226 trace.data 3399401472 4194304 sample1-d4f460ed702472e1
weight b227 raw capture.sample1-f6ec867f15a1ede7 tensor [1048576]
init b227 trace.data 3403595776 4194304 sample1-f6ec867f15a1ede7
weight b228 raw capture.sample1-9c4e257aa41b712b tensor [2097152]
init b228 trace.data 3407790080 8388608 sample1-9c4e257aa41b712b
weight b229 raw capture.sample1-bef3adae19214276 tensor [128]
init b229 trace.data 3416178688 512 sample1-bef3adae19214276
weight b230 raw capture.sample1-5e42f512e45fbf59 tensor [128]
init b230 trace.data 3416179200 512 sample1-5e42f512e45fbf59
weight b231 raw capture.sample1-af2b4de1a8eb705f tensor [3072]
init b231 trace.data 3416179712 12288 sample1-af2b4de1a8eb705f
weight b232 raw capture.sample1-36347e1153ca0751 tensor [18874368]
init b232 trace.data 3416192000 75497472 sample1-36347e1153ca0751
weight b233 raw capture.sample1-9971ec943af201b8 tensor [9437184]
init b233 trace.data 3491689472 37748736 sample1-9971ec943af201b8
weight b234 raw capture.sample1-ac152cba0a086e8f tensor [1024]
init b234 trace.data 3529438208 4096 sample1-ac152cba0a086e8f
weight b235 raw capture.sample1-34a4887a9d4f4797 tensor [1024]
init b235 trace.data 3529442304 4096 sample1-34a4887a9d4f4797
weight b236 raw capture.sample1-208d1a66a69a01c2 tensor [2097152]
init b236 trace.data 3529446400 8388608 sample1-208d1a66a69a01c2
weight b237 raw capture.sample1-9bfa5e6097e8054d tensor [1048576]
init b237 trace.data 3537835008 4194304 sample1-9bfa5e6097e8054d
weight b238 raw capture.sample1-6c9d606a2f1ac409 tensor [1048576]
init b238 trace.data 3542029312 4194304 sample1-6c9d606a2f1ac409
weight b239 raw capture.sample1-9f6156fa476c2811 tensor [2097152]
init b239 trace.data 3546223616 8388608 sample1-9f6156fa476c2811
weight b240 raw capture.sample1-021e829f67576425 tensor [128]
init b240 trace.data 3554612224 512 sample1-021e829f67576425
weight b241 raw capture.sample1-011752578f01300b tensor [128]
init b241 trace.data 3554612736 512 sample1-011752578f01300b
weight b242 raw capture.sample1-0724de0a7eb0187c tensor [3072]
init b242 trace.data 3554613248 12288 sample1-0724de0a7eb0187c
weight b243 raw capture.sample1-033e484ea96c2186 tensor [18874368]
init b243 trace.data 3554625536 75497472 sample1-033e484ea96c2186
weight b244 raw capture.sample1-e09186658f783119 tensor [9437184]
init b244 trace.data 3630123008 37748736 sample1-e09186658f783119
weight b245 raw capture.sample1-c366c9d9bcf4ba0b tensor [1024]
init b245 trace.data 3667871744 4096 sample1-c366c9d9bcf4ba0b
weight b246 raw capture.sample1-0d2540ad9553125b tensor [1024]
init b246 trace.data 3667875840 4096 sample1-0d2540ad9553125b
weight b247 raw capture.sample1-e5fb34cd08ce621a tensor [2097152]
init b247 trace.data 3667879936 8388608 sample1-e5fb34cd08ce621a
weight b248 raw capture.sample1-f2295daa2da52c82 tensor [1048576]
init b248 trace.data 3676268544 4194304 sample1-f2295daa2da52c82
weight b249 raw capture.sample1-4546bc51720ca91a tensor [1048576]
init b249 trace.data 3680462848 4194304 sample1-4546bc51720ca91a
weight b250 raw capture.sample1-e6d96c1b4108980c tensor [2097152]
init b250 trace.data 3684657152 8388608 sample1-e6d96c1b4108980c
weight b251 raw capture.sample1-f6dd15b4e624684a tensor [128]
init b251 trace.data 3693045760 512 sample1-f6dd15b4e624684a
weight b252 raw capture.sample1-a94bd0c43f889fb9 tensor [128]
init b252 trace.data 3693046272 512 sample1-a94bd0c43f889fb9
weight b253 raw capture.sample1-7dff7943addd9601 tensor [3072]
init b253 trace.data 3693046784 12288 sample1-7dff7943addd9601
weight b254 raw capture.sample1-8d7f888ea93cdb96 tensor [18874368]
init b254 trace.data 3693059072 75497472 sample1-8d7f888ea93cdb96
weight b255 raw capture.sample1-e0b90e156cfc7dcc tensor [9437184]
init b255 trace.data 3768556544 37748736 sample1-e0b90e156cfc7dcc
weight b256 raw capture.sample1-d98efabb467cf2fc tensor [1024]
init b256 trace.data 3806305280 4096 sample1-d98efabb467cf2fc
weight b257 raw capture.sample1-0c0e5b1aa74b4b18 tensor [1024]
init b257 trace.data 3806309376 4096 sample1-0c0e5b1aa74b4b18
weight b258 raw capture.sample1-b0704ac1c6bf6123 tensor [2097152]
init b258 trace.data 3806313472 8388608 sample1-b0704ac1c6bf6123
weight b259 raw capture.sample1-8774f98c94c8e75f tensor [1048576]
init b259 trace.data 3814702080 4194304 sample1-8774f98c94c8e75f
weight b260 raw capture.sample1-b3ead83e584a5721 tensor [1048576]
init b260 trace.data 3818896384 4194304 sample1-b3ead83e584a5721
weight b261 raw capture.sample1-53141e8decabef88 tensor [2097152]
init b261 trace.data 3823090688 8388608 sample1-53141e8decabef88
weight b262 raw capture.sample1-60db08b15201b56a tensor [128]
init b262 trace.data 3831479296 512 sample1-60db08b15201b56a
weight b263 raw capture.sample1-d8f5bb00721333b9 tensor [128]
init b263 trace.data 3831479808 512 sample1-d8f5bb00721333b9
weight b264 raw capture.sample1-a975623a96408a04 tensor [3072]
init b264 trace.data 3831480320 12288 sample1-a975623a96408a04
weight b265 raw capture.sample1-68fd0bf176374532 tensor [18874368]
init b265 trace.data 3831492608 75497472 sample1-68fd0bf176374532
weight b266 raw capture.sample1-27be895614d05415 tensor [9437184]
init b266 trace.data 3906990080 37748736 sample1-27be895614d05415
weight b267 raw capture.sample1-c8e7c5d6c8ae6960 tensor [1024]
init b267 trace.data 3944738816 4096 sample1-c8e7c5d6c8ae6960
weight b268 raw capture.sample1-52cbf078b3853bae tensor [1024]
init b268 trace.data 3944742912 4096 sample1-52cbf078b3853bae
weight b269 raw capture.sample1-529b77d894b37842 tensor [2097152]
init b269 trace.data 3944747008 8388608 sample1-529b77d894b37842
weight b270 raw capture.sample1-9f45117a9114f3f0 tensor [1048576]
init b270 trace.data 3953135616 4194304 sample1-9f45117a9114f3f0
weight b271 raw capture.sample1-7665f94fde1f3e1c tensor [1048576]
init b271 trace.data 3957329920 4194304 sample1-7665f94fde1f3e1c
weight b272 raw capture.sample1-f2457ffba9f93e5e tensor [2097152]
init b272 trace.data 3961524224 8388608 sample1-f2457ffba9f93e5e
weight b273 raw capture.sample1-18176ec07367ffdf tensor [128]
init b273 trace.data 3969912832 512 sample1-18176ec07367ffdf
weight b274 raw capture.sample1-c84141cba7fd1d7d tensor [128]
init b274 trace.data 3969913344 512 sample1-c84141cba7fd1d7d
weight b275 raw capture.sample1-6c0af9e0452581c9 tensor [3072]
init b275 trace.data 3969913856 12288 sample1-6c0af9e0452581c9
weight b276 raw capture.sample1-2d887e36a82db2e2 tensor [18874368]
init b276 trace.data 3969926144 75497472 sample1-2d887e36a82db2e2
weight b277 raw capture.sample1-b80b0e68a30f6be9 tensor [9437184]
init b277 trace.data 4045423616 37748736 sample1-b80b0e68a30f6be9
weight b278 raw capture.sample1-29e76d987c1599da tensor [1024]
init b278 trace.data 4083172352 4096 sample1-29e76d987c1599da
weight b279 raw capture.sample1-f5e2496906bc9993 tensor [1024]
init b279 trace.data 4083176448 4096 sample1-f5e2496906bc9993
weight b280 raw capture.sample1-4d83739cb00ddf4f tensor [2097152]
init b280 trace.data 4083180544 8388608 sample1-4d83739cb00ddf4f
weight b281 raw capture.sample1-a84f33bc97d36059 tensor [1048576]
init b281 trace.data 4091569152 4194304 sample1-a84f33bc97d36059
weight b282 raw capture.sample1-3ec46478066dc7ca tensor [1048576]
init b282 trace.data 4095763456 4194304 sample1-3ec46478066dc7ca
weight b283 raw capture.sample1-acd4e130e8808144 tensor [2097152]
init b283 trace.data 4099957760 8388608 sample1-acd4e130e8808144
weight b284 raw capture.sample1-aceeaad33c614ce5 tensor [128]
init b284 trace.data 4108346368 512 sample1-aceeaad33c614ce5
weight b285 raw capture.sample1-fe81f53f65a4782f tensor [128]
init b285 trace.data 4108346880 512 sample1-fe81f53f65a4782f
weight b286 raw capture.sample1-3222e5510facf46f tensor [3072]
init b286 trace.data 4108347392 12288 sample1-3222e5510facf46f
weight b287 raw capture.sample1-ded0f0abf4bd8fb4 tensor [18874368]
init b287 trace.data 4108359680 75497472 sample1-ded0f0abf4bd8fb4
weight b288 raw capture.sample1-eaf0a75d5026b1d8 tensor [9437184]
init b288 trace.data 4183857152 37748736 sample1-eaf0a75d5026b1d8
weight b289 raw capture.sample1-712670a4cd939ef5 tensor [1024]
init b289 trace.data 4221605888 4096 sample1-712670a4cd939ef5
weight b290 raw capture.sample1-1a405c4fa3dc18c6 tensor [1024]
init b290 trace.data 4221609984 4096 sample1-1a405c4fa3dc18c6
weight b291 raw capture.sample1-8741a67510f5809c tensor [2097152]
init b291 trace.data 4221614080 8388608 sample1-8741a67510f5809c
weight b292 raw capture.sample1-5a38f7c0b29fd61e tensor [1048576]
init b292 trace.data 4230002688 4194304 sample1-5a38f7c0b29fd61e
weight b293 raw capture.sample1-e5db78a2a525f3b3 tensor [1048576]
init b293 trace.data 4234196992 4194304 sample1-e5db78a2a525f3b3
weight b294 raw capture.sample1-db7f2534e54ec595 tensor [2097152]
init b294 trace.data 4238391296 8388608 sample1-db7f2534e54ec595
weight b295 raw capture.sample1-e18f9c2697ade488 tensor [128]
init b295 trace.data 4246779904 512 sample1-e18f9c2697ade488
weight b296 raw capture.sample1-a8035775ee61032c tensor [128]
init b296 trace.data 4246780416 512 sample1-a8035775ee61032c
weight b297 raw capture.sample1-e5687b9f724b772d tensor [3072]
init b297 trace.data 4246780928 12288 sample1-e5687b9f724b772d
weight b298 raw capture.sample1-07c636fc57f65891 tensor [18874368]
init b298 trace.data 4246793216 75497472 sample1-07c636fc57f65891
weight b299 raw capture.sample1-a0f74d6821aac6a7 tensor [9437184]
init b299 trace.data 4322290688 37748736 sample1-a0f74d6821aac6a7
weight b300 raw capture.sample1-d5b5504a9d7ed38e tensor [1024]
init b300 trace.data 4360039424 4096 sample1-d5b5504a9d7ed38e
weight b301 raw capture.sample1-36b5cfccaf2262e5 tensor [1024]
init b301 trace.data 4360043520 4096 sample1-36b5cfccaf2262e5
weight b302 raw capture.sample1-0c75f5081757aa7b tensor [2097152]
init b302 trace.data 4360047616 8388608 sample1-0c75f5081757aa7b
weight b303 raw capture.sample1-096b854c24e762ad tensor [1048576]
init b303 trace.data 4368436224 4194304 sample1-096b854c24e762ad
weight b304 raw capture.sample1-00dfad74b1f4c366 tensor [1048576]
init b304 trace.data 4372630528 4194304 sample1-00dfad74b1f4c366
weight b305 raw capture.sample1-ad802ccda91e029d tensor [2097152]
init b305 trace.data 4376824832 8388608 sample1-ad802ccda91e029d
weight b306 raw capture.sample1-e764ff211874b79f tensor [128]
init b306 trace.data 4385213440 512 sample1-e764ff211874b79f
weight b307 raw capture.sample1-c3ae1defb8bfe879 tensor [128]
init b307 trace.data 4385213952 512 sample1-c3ae1defb8bfe879
weight b308 raw capture.sample1-6b8c3e2980a689dc tensor [3072]
init b308 trace.data 4385214464 12288 sample1-6b8c3e2980a689dc
weight b309 raw capture.sample1-bbd096dbfdccf621 tensor [18874368]
init b309 trace.data 4385226752 75497472 sample1-bbd096dbfdccf621
weight b310 raw capture.sample1-f29e86582ed7c722 tensor [9437184]
init b310 trace.data 4460724224 37748736 sample1-f29e86582ed7c722
buf b311 f32 3689728
init b311 trace.data 4498472960 14758912 sample1-6183e9a44d2c9b30
buf b312 f32 151936 output:logits
init b312 trace.data 4513231872 607744 sample1-af83d4ef7fcb83e2
kernel raw0 in=2 out=1 threads=1024 tg=256 tgbytes=16 layout=slots slots=mem:in0,input:in1
words 00000080 00000086 00000100 00000181 00040083 00000282 00000000 00004091 00000000 00000390 00006386 00000400 00006383 0000618d 00000000 0000018e 00000100 00000001
endkernel
kernel raw1 in=2 out=1 threads=256 tg=256 tgbytes=1024 layout=slots slots=mem:in0,input:in1
words 00000080 00000181 00001416 00001513 00000000 00000b13 40800000 00000286 00000400 00044283 00000a13 00000000 00000b40 00004591 00000100 0014aa22 0000000a 00004285 00000100 00000041 00028a61 00000000 00000050 00001013 43000000 00428f70 00000f72 00028d60 00000000 00028e60 00000080 0039ad20 00028d61 00000000 00000073 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002aa60 00000000 00000050 00014a2b 3a800000 00014a2a 358637bd 00014c29 00000181 00000286 00000400 00044283 00002385 00000000 00000b40 00004591 00000100 0000668d 00000000 0030a521 0018a521 0000458e 00000500 00006385 00000100 00004285 00000100 00000041 00000001
endkernel
kernel raw2 in=2 out=1 threads=524288 tg=256 tgbytes=1024 layout=slots slots=mem:in0,input:in1
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000286 00000400 00044283 00002785 00000000 00000a13 00000000 00000b13 00000000 00000813 40800000 00000840 0000e591 00000500 0000448d 00000000 00148a22 0000000a 00004285 00000100 0000e785 00000100 00000041 00028a61 00000000 00000050 00001013 43000000 00428f70 00000f72 00028d60 00000000 00028e60 00000080 0039ad20 00028d61 00000000 00000073 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002aa60 00000000 00000050 005a8d70 00000d72 00000a8e 00000900 00000073 00000001
endkernel
kernel raw3 in=2 out=1 threads=262144 tg=256 tgbytes=1024 layout=slots slots=mem:in0,input:in1
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000286 00000400 00044283 00002785 00000000 00000a13 00000000 00000b13 00000000 00000813 40800000 00000840 0000e591 00000500 0000448d 00000000 00148a22 0000000a 00004285 00000100 0000e785 00000100 00000041 00028a61 00000000 00000050 00001013 43000000 00428f70 00000f72 00028d60 00000000 00028e60 00000080 0039ad20 00028d61 00000000 00000073 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002aa60 00000000 00000050 005a8d70 00000d72 00000a8e 00001100 00000073 00000001
endkernel
kernel raw4 in=2 out=1 threads=262144 tg=256 tgbytes=1024 layout=slots slots=mem:in0,input:in1
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000286 00000400 00044283 00002785 00000000 00000a13 00000000 00000b13 00000000 00000813 40800000 00000840 0000e591 00000500 0000448d 00000000 00148a22 0000000a 00004285 00000100 0000e785 00000100 00000041 00028a61 00000000 00000050 00001013 43000000 00428f70 00000f72 00028d60 00000000 00028e60 00000080 0039ad20 00028d61 00000000 00000073 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002aa60 00000000 00000050 005a8d70 00000d72 00000a8e 00001500 00000073 00000001
endkernel
kernel raw5 in=2 out=1 threads=2048 tg=128 tgbytes=512 layout=slots slots=mem:in0,input:in1
words 00000080 00000181 00001416 00001513 00000000 00000b13 3f800000 00000286 00000080 00044283 00000a13 00000000 00000b40 00004591 00000900 0014aa22 0000000a 00004285 00000080 00000041 00028a61 00000000 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002aa60 00000000 00000050 00014a2b 3c000000 00014a2a 358637bd 00014c29 00000181 00000286 00000080 00044283 00002385 00000000 00000b40 00004591 00000900 0000668d 00000000 0030a521 0018a521 0000458e 00000900 00006385 00000080 00004285 00000080 00000041 00000001
endkernel
kernel raw6 in=2 out=1 threads=1024 tg=128 tgbytes=512 layout=slots slots=mem:in0,input:in1
words 00000080 00000181 00001416 00001513 00000000 00000b13 3f800000 00000286 00000080 00044283 00000a13 00000000 00000b40 00004591 00001100 0014aa22 0000000a 00004285 00000080 00000041 00028a61 00000000 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002aa60 00000000 00000050 00014a2b 3c000000 00014a2a 358637bd 00014c29 00000181 00000286 00000080 00044283 00002385 00000000 00000b40 00004591 00001100 0000668d 00000000 0030a521 0018a521 0000458e 00001100 00006385 00000080 00004285 00000080 00000041 00000001
endkernel
kernel raw7 in=1 out=1 threads=1536 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000188 00000006 00002186 00000080 00000287 0000003f 00082383 00000482 00000000 00008091 00000040 00000590 0000a586 00000040 0008a583 0000a191 00002d00 0000a291 00003d00 00006391 00000900 00006491 00000940 00046521 00088621 0018a523 00048721 00086821 0020e720 0000658e 00000900 0000678e 00000940 00000001
endkernel
kernel raw8 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00004d00 00000001
endkernel
kernel raw9 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00004d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00014d00 0000c391 00014d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw10 in=2 out=1 threads=262144 tg=256 tgbytes=1024 layout=slots slots=mem:in0,input:in1
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000286 00000800 00044283 00002785 00000000 00000a13 00000000 00000b13 00000000 00000813 41000000 00000840 0000e591 00001900 0000448d 00000000 00148a22 0000000a 00004285 00000100 0000e785 00000100 00000041 00028a61 00000000 00000050 00001013 43000000 00428f70 00000f72 00028d60 00000000 00028e60 00000080 0039ad20 00028d61 00000000 00000073 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002aa60 00000000 00000050 005a8d70 00000d72 00000e91 00000100 00394a20 00000a8e 00000100 00000073 00000001
endkernel
kernel raw11 in=2 out=1 threads=768 tg=256 tgbytes=1024 layout=slots slots=mem:in0,input:in1
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000286 00000400 00044283 00002785 00000000 00000a13 00000000 00000b13 00000000 00000813 40800000 00000840 0000e591 00000500 0000448d 00000000 00148a22 0000000a 00004285 00000100 0000e785 00000100 00000041 00028a61 00000000 00000050 00001013 43000000 00428f70 00000f72 00028d60 00000000 00028e60 00000080 0039ad20 00028d61 00000000 00000073 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002aa60 00000000 00000050 005a8d70 00000d72 00000a8e 000000c0 00000073 00000001
endkernel
kernel raw12 in=1 out=1 threads=1 tg=1 tgbytes=16 layout=slots slots=mem:in0
words 00000082 00000000 00000091 000000c0 00000191 000000c1 00000291 000000c2 00040570 00002323 000ca422 00000000 00088670 00000713 40000000 0014e723 001cc822 00000005 0000088e 00000080 00000001
endkernel
kernel raw13 in=2 out=1 threads=786432 tg=256 tgbytes=1024 layout=slots slots=mem:in0,input:in1
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000286 00000400 00044283 00000482 00000000 00008391 00000080 00006590 0000a586 00600000 00144283 00004685 00300000 00002785 00000000 00000a13 00000000 00000b13 00000000 00000813 40800000 00000840 0000e591 00000500 0000448d 00000000 00148a22 0000000a 00004285 00000100 0000c48d 00000000 00148b22 0000000b 0000c685 00000100 0000e785 00000100 00000041 00028a61 00000000 00000050 00001013 43000000 00428f70 00000f72 00028d60 00000000 00028e60 00000080 0039ad20 00028d61 00000000 00000073 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002aa60 00000000 00000050 00028b61 00000000 00000050 00001013 43000000 00428f70 00000f72 00028d60 00000000 00028e60 00000080 0039ad20 00028d61 00000000 00000073 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002ab60 00000000 00000050 00014c26 00314a21 002d4a21 005a8d70 00000d72 00000a8e 00002100 00000073 00000001
endkernel
kernel raw14 in=2 out=1 threads=262144 tg=256 tgbytes=1024 layout=slots slots=mem:in0,input:in1
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000286 00000c00 00044283 00000482 00000000 00008391 00000080 00006590 0000a586 00300000 00144283 00002785 00000000 00000a13 00000000 00000b13 00000000 00000813 41400000 00000840 0000e591 00002100 0000448d 00000000 00148a22 0000000a 00004285 00000100 0000e785 00000100 00000041 00028a61 00000000 00000050 00001013 43000000 00428f70 00000f72 00028d60 00000000 00028e60 00000080 0039ad20 00028d61 00000000 00000073 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002aa60 00000000 00000050 005a8d70 00000d72 00000e91 00000100 00394a20 00000a8e 00000100 00000073 00000001
endkernel
kernel raw15 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00024d00 00000001
endkernel
kernel raw16 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00024d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00034d00 0000c391 00034d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw17 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00044d00 00000001
endkernel
kernel raw18 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00044d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00054d00 0000c391 00054d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw19 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00064d00 00000001
endkernel
kernel raw20 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00064d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00074d00 0000c391 00074d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw21 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00084d00 00000001
endkernel
kernel raw22 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00084d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00094d00 0000c391 00094d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw23 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 000a4d00 00000001
endkernel
kernel raw24 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 000a4d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 000b4d00 0000c391 000b4d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw25 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 000c4d00 00000001
endkernel
kernel raw26 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 000c4d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 000d4d00 0000c391 000d4d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw27 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 000e4d00 00000001
endkernel
kernel raw28 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 000e4d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 000f4d00 0000c391 000f4d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw29 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00104d00 00000001
endkernel
kernel raw30 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00104d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00114d00 0000c391 00114d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw31 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00124d00 00000001
endkernel
kernel raw32 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00124d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00134d00 0000c391 00134d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw33 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00144d00 00000001
endkernel
kernel raw34 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00144d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00154d00 0000c391 00154d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw35 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00164d00 00000001
endkernel
kernel raw36 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00164d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00174d00 0000c391 00174d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw37 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00184d00 00000001
endkernel
kernel raw38 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00184d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00194d00 0000c391 00194d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw39 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 001a4d00 00000001
endkernel
kernel raw40 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 001a4d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 001b4d00 0000c391 001b4d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw41 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 001c4d00 00000001
endkernel
kernel raw42 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 001c4d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 001d4d00 0000c391 001d4d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw43 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 001e4d00 00000001
endkernel
kernel raw44 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 001e4d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 001f4d00 0000c391 001f4d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw45 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00204d00 00000001
endkernel
kernel raw46 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00204d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00214d00 0000c391 00214d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw47 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00224d00 00000001
endkernel
kernel raw48 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00224d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00234d00 0000c391 00234d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw49 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00244d00 00000001
endkernel
kernel raw50 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00244d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00254d00 0000c391 00254d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw51 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00264d00 00000001
endkernel
kernel raw52 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00264d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00274d00 0000c391 00274d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw53 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00284d00 00000001
endkernel
kernel raw54 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00284d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00294d00 0000c391 00294d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw55 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 002a4d00 00000001
endkernel
kernel raw56 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 002a4d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 002b4d00 0000c391 002b4d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw57 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 002c4d00 00000001
endkernel
kernel raw58 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 002c4d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 002d4d00 0000c391 002d4d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw59 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 002e4d00 00000001
endkernel
kernel raw60 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 002e4d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 002f4d00 0000c391 002f4d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw61 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00304d00 00000001
endkernel
kernel raw62 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00304d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00314d00 0000c391 00314d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw63 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00324d00 00000001
endkernel
kernel raw64 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00324d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00334d00 0000c391 00334d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw65 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00344d00 00000001
endkernel
kernel raw66 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00344d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00354d00 0000c391 00354d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw67 in=1 out=1 threads=2048 tg=256 tgbytes=16 layout=slots slots=mem:in0
words 00000080 00000086 00000100 00000981 00240083 00000191 00001100 00000288 0000000a 00004286 00010000 00000388 00000007 00006387 00000007 00006386 00002000 000c4283 00000487 0000007f 00104283 00000582 00000000 0000a091 00000040 00000690 0000c686 00000080 00184283 0000418e 00364d00 00000001
endkernel
kernel raw68 in=1 out=1 threads=1024 tg=64 tgbytes=768 layout=slots slots=mem:in0
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000282 00000000 00004091 00000040 0000172a 3f800000 00000386 00000080 00000488 00000001 00008486 00002000 00002586 00000080 0010a583 00000a13 00000000 00000b13 43000000 00000b40 00006191 00000900 0000a291 00364d00 00082a22 0000000a 00006385 00000001 0000a585 00000001 00000041 00014a2b 3db504f3 005e8c70 0032ce23 00000d13 f149f2ca 00314a21 0035ca22 0000000a 0001582a 00000000 00029861 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041f170 003e1023 00422f22 0000000f 00028f61 00000000 00000073 00000050 0002b860 00000000 00000050 00614a23 00014a24 0001592a 00000000 00029961 00000000 00000050 00001213 42000000 004a9170 00001172 00028f60 00000000 00029060 00000020 0041ef20 00028f61 00000000 00000073 00000050 00001213 41800000 004a9170 00001172 00028f60 00000000 00029060 00000010 0041ef20 00028f61 00000000 00000073 00000050 00001213 41000000 004a9170 00001172 00028f60 00000000 00029060 00000008 0041ef20 00028f61 00000000 00000073 00000050 00001213 40800000 004a9170 00001172 00028f60 00000000 00029060 00000004 0041ef20 00028f61 00000000 00000073 00000050 00001213 40000000 004a9170 00001172 00028f60 00000000 00029060 00000002 0041ef20 00028f61 00000000 00000073 00000050 00001213 3f800000 004a9170 00001172 00028f60 00000000 00029060 00000001 0041ef20 00028f61 00000000 00000073 00000050 0002b960 00000000 00000050 00654a27 00028a61 00000080 00000050 00000080 00000181 00000488 00000001 00008486 00002000 00048683 00000a13 00000000 00000b13 00000000 00001a13 00000000 0002fb2a 00000000 00001b40 00034160 00000080 0000c291 00374d00 0000c391 00374d40 00082a22 0000000a 000c2b22 0000000b 0000c685 00000080 005b5a20 00000041 00000786 00000080 0004e783 0000ea8e 00001900 0000eb8e 00001940 00000001
endkernel
kernel raw69 in=3 out=1 threads=38895616 tg=256 tgbytes=1024 layout=slots slots=mem:in0,lut:in1,input:in2
words 00000080 00000181 00001416 00001513 00000000 00001613 3f800000 00000286 00000400 00044283 00002785 00000000 00000a13 00000000 00000b13 00000000 00000813 40800000 00000840 0000e58c 00000500 0000448d 00000000 00148a22 0000000a 00004285 00000100 0000e785 00000100 00000041 00028a61 00000000 00000050 00001013 43000000 00428f70 00000f72 00028d60 00000000 00028e60 00000080 0039ad20 00028d61 00000000 00000073 00000050 00001013 42800000 00428f70 00000f72 00028d60 00000000 00028e60 00000040 0039ad20 00028d61 00000000 00000073 00000050 00001013 42000000 00428f70 00000f72 00028d60 00000000 00028e60 00000020 0039ad20 00028d61 00000000 00000073 00000050 00001013 41800000 00428f70 00000f72 00028d60 00000000 00028e60 00000010 0039ad20 00028d61 00000000 00000073 00000050 00001013 41000000 00428f70 00000f72 00028d60 00000000 00028e60 00000008 0039ad20 00028d61 00000000 00000073 00000050 00001013 40800000 00428f70 00000f72 00028d60 00000000 00028e60 00000004 0039ad20 00028d61 00000000 00000073 00000050 00001013 40000000 00428f70 00000f72 00028d60 00000000 00028e60 00000002 0039ad20 00028d61 00000000 00000073 00000050 00001013 3f800000 00428f70 00000f72 00028d60 00000000 00028e60 00000001 0039ad20 00028d61 00000000 00000073 00000050 0002aa60 00000000 00000050 005a8d70 00000d72 00000a8e 00000000 00000073 00000001
endkernel
node session_CopyInto_-616 session sessionCopyInto:srcOff=0,dstOff=0,n=1 token -> b311
node session_CopyInto_-617 session sessionCopyInto:srcOff=0,dstOff=64,n=1 position -> b311
node dispatch1339 micro raw0 b311 b0 -> b311
node dispatch1340 micro raw1 b311 b3 -> b311
node dispatch1341 micro raw2 b311 b5 -> b311
node dispatch1342 micro raw3 b311 b6 -> b311
node dispatch1343 micro raw4 b311 b7 -> b311
node dispatch1344 micro raw5 b311 b9 -> b311
node dispatch1345 micro raw6 b311 b10 -> b311
node dispatch1346 micro raw7 b311 -> b311
node dispatch1347 micro raw8 b311 -> b311
node dispatch1348 micro raw9 b311 -> b311
node dispatch1349 micro raw10 b311 b8 -> b311
node dispatch1350 micro raw1 b311 b4 -> b311
node dispatch1351 micro raw11 b311 b11 -> b311
node dispatch1352 micro raw12 b311 -> b311
node dispatch1353 micro raw13 b311 b12 -> b311
node dispatch1354 micro raw14 b311 b13 -> b311
node dispatch1355 micro raw1 b311 b14 -> b311
node dispatch1356 micro raw2 b311 b16 -> b311
node dispatch1357 micro raw3 b311 b17 -> b311
node dispatch1358 micro raw4 b311 b18 -> b311
node dispatch1359 micro raw5 b311 b20 -> b311
node dispatch1360 micro raw6 b311 b21 -> b311
node dispatch1361 micro raw7 b311 -> b311
node dispatch1362 micro raw15 b311 -> b311
node dispatch1363 micro raw16 b311 -> b311
node dispatch1364 micro raw10 b311 b19 -> b311
node dispatch1365 micro raw1 b311 b15 -> b311
node dispatch1366 micro raw11 b311 b22 -> b311
node dispatch1367 micro raw12 b311 -> b311
node dispatch1368 micro raw13 b311 b23 -> b311
node dispatch1369 micro raw14 b311 b24 -> b311
node dispatch1370 micro raw1 b311 b25 -> b311
node dispatch1371 micro raw2 b311 b27 -> b311
node dispatch1372 micro raw3 b311 b28 -> b311
node dispatch1373 micro raw4 b311 b29 -> b311
node dispatch1374 micro raw5 b311 b31 -> b311
node dispatch1375 micro raw6 b311 b32 -> b311
node dispatch1376 micro raw7 b311 -> b311
node dispatch1377 micro raw17 b311 -> b311
node dispatch1378 micro raw18 b311 -> b311
node dispatch1379 micro raw10 b311 b30 -> b311
node dispatch1380 micro raw1 b311 b26 -> b311
node dispatch1381 micro raw11 b311 b33 -> b311
node dispatch1382 micro raw12 b311 -> b311
node dispatch1383 micro raw13 b311 b34 -> b311
node dispatch1384 micro raw14 b311 b35 -> b311
node dispatch1385 micro raw1 b311 b36 -> b311
node dispatch1386 micro raw2 b311 b38 -> b311
node dispatch1387 micro raw3 b311 b39 -> b311
node dispatch1388 micro raw4 b311 b40 -> b311
node dispatch1389 micro raw5 b311 b42 -> b311
node dispatch1390 micro raw6 b311 b43 -> b311
node dispatch1391 micro raw7 b311 -> b311
node dispatch1392 micro raw19 b311 -> b311
node dispatch1393 micro raw20 b311 -> b311
node dispatch1394 micro raw10 b311 b41 -> b311
node dispatch1395 micro raw1 b311 b37 -> b311
node dispatch1396 micro raw11 b311 b44 -> b311
node dispatch1397 micro raw12 b311 -> b311
node dispatch1398 micro raw13 b311 b45 -> b311
node dispatch1399 micro raw14 b311 b46 -> b311
node dispatch1400 micro raw1 b311 b47 -> b311
node dispatch1401 micro raw2 b311 b49 -> b311
node dispatch1402 micro raw3 b311 b50 -> b311
node dispatch1403 micro raw4 b311 b51 -> b311
node dispatch1404 micro raw5 b311 b53 -> b311
node dispatch1405 micro raw6 b311 b54 -> b311
node dispatch1406 micro raw7 b311 -> b311
node dispatch1407 micro raw21 b311 -> b311
node dispatch1408 micro raw22 b311 -> b311
node dispatch1409 micro raw10 b311 b52 -> b311
node dispatch1410 micro raw1 b311 b48 -> b311
node dispatch1411 micro raw11 b311 b55 -> b311
node dispatch1412 micro raw12 b311 -> b311
node dispatch1413 micro raw13 b311 b56 -> b311
node dispatch1414 micro raw14 b311 b57 -> b311
node dispatch1415 micro raw1 b311 b58 -> b311
node dispatch1416 micro raw2 b311 b60 -> b311
node dispatch1417 micro raw3 b311 b61 -> b311
node dispatch1418 micro raw4 b311 b62 -> b311
node dispatch1419 micro raw5 b311 b64 -> b311
node dispatch1420 micro raw6 b311 b65 -> b311
node dispatch1421 micro raw7 b311 -> b311
node dispatch1422 micro raw23 b311 -> b311
node dispatch1423 micro raw24 b311 -> b311
node dispatch1424 micro raw10 b311 b63 -> b311
node dispatch1425 micro raw1 b311 b59 -> b311
node dispatch1426 micro raw11 b311 b66 -> b311
node dispatch1427 micro raw12 b311 -> b311
node dispatch1428 micro raw13 b311 b67 -> b311
node dispatch1429 micro raw14 b311 b68 -> b311
node dispatch1430 micro raw1 b311 b69 -> b311
node dispatch1431 micro raw2 b311 b71 -> b311
node dispatch1432 micro raw3 b311 b72 -> b311
node dispatch1433 micro raw4 b311 b73 -> b311
node dispatch1434 micro raw5 b311 b75 -> b311
node dispatch1435 micro raw6 b311 b76 -> b311
node dispatch1436 micro raw7 b311 -> b311
node dispatch1437 micro raw25 b311 -> b311
node dispatch1438 micro raw26 b311 -> b311
node dispatch1439 micro raw10 b311 b74 -> b311
node dispatch1440 micro raw1 b311 b70 -> b311
node dispatch1441 micro raw11 b311 b77 -> b311
node dispatch1442 micro raw12 b311 -> b311
node dispatch1443 micro raw13 b311 b78 -> b311
node dispatch1444 micro raw14 b311 b79 -> b311
node dispatch1445 micro raw1 b311 b80 -> b311
node dispatch1446 micro raw2 b311 b82 -> b311
node dispatch1447 micro raw3 b311 b83 -> b311
node dispatch1448 micro raw4 b311 b84 -> b311
node dispatch1449 micro raw5 b311 b86 -> b311
node dispatch1450 micro raw6 b311 b87 -> b311
node dispatch1451 micro raw7 b311 -> b311
node dispatch1452 micro raw27 b311 -> b311
node dispatch1453 micro raw28 b311 -> b311
node dispatch1454 micro raw10 b311 b85 -> b311
node dispatch1455 micro raw1 b311 b81 -> b311
node dispatch1456 micro raw11 b311 b88 -> b311
node dispatch1457 micro raw12 b311 -> b311
node dispatch1458 micro raw13 b311 b89 -> b311
node dispatch1459 micro raw14 b311 b90 -> b311
node dispatch1460 micro raw1 b311 b91 -> b311
node dispatch1461 micro raw2 b311 b93 -> b311
node dispatch1462 micro raw3 b311 b94 -> b311
node dispatch1463 micro raw4 b311 b95 -> b311
node dispatch1464 micro raw5 b311 b97 -> b311
node dispatch1465 micro raw6 b311 b98 -> b311
node dispatch1466 micro raw7 b311 -> b311
node dispatch1467 micro raw29 b311 -> b311
node dispatch1468 micro raw30 b311 -> b311
node dispatch1469 micro raw10 b311 b96 -> b311
node dispatch1470 micro raw1 b311 b92 -> b311
node dispatch1471 micro raw11 b311 b99 -> b311
node dispatch1472 micro raw12 b311 -> b311
node dispatch1473 micro raw13 b311 b100 -> b311
node dispatch1474 micro raw14 b311 b101 -> b311
node dispatch1475 micro raw1 b311 b102 -> b311
node dispatch1476 micro raw2 b311 b104 -> b311
node dispatch1477 micro raw3 b311 b105 -> b311
node dispatch1478 micro raw4 b311 b106 -> b311
node dispatch1479 micro raw5 b311 b108 -> b311
node dispatch1480 micro raw6 b311 b109 -> b311
node dispatch1481 micro raw7 b311 -> b311
node dispatch1482 micro raw31 b311 -> b311
node dispatch1483 micro raw32 b311 -> b311
node dispatch1484 micro raw10 b311 b107 -> b311
node dispatch1485 micro raw1 b311 b103 -> b311
node dispatch1486 micro raw11 b311 b110 -> b311
node dispatch1487 micro raw12 b311 -> b311
node dispatch1488 micro raw13 b311 b111 -> b311
node dispatch1489 micro raw14 b311 b112 -> b311
node dispatch1490 micro raw1 b311 b113 -> b311
node dispatch1491 micro raw2 b311 b115 -> b311
node dispatch1492 micro raw3 b311 b116 -> b311
node dispatch1493 micro raw4 b311 b117 -> b311
node dispatch1494 micro raw5 b311 b119 -> b311
node dispatch1495 micro raw6 b311 b120 -> b311
node dispatch1496 micro raw7 b311 -> b311
node dispatch1497 micro raw33 b311 -> b311
node dispatch1498 micro raw34 b311 -> b311
node dispatch1499 micro raw10 b311 b118 -> b311
node dispatch1500 micro raw1 b311 b114 -> b311
node dispatch1501 micro raw11 b311 b121 -> b311
node dispatch1502 micro raw12 b311 -> b311
node dispatch1503 micro raw13 b311 b122 -> b311
node dispatch1504 micro raw14 b311 b123 -> b311
node dispatch1505 micro raw1 b311 b124 -> b311
node dispatch1506 micro raw2 b311 b126 -> b311
node dispatch1507 micro raw3 b311 b127 -> b311
node dispatch1508 micro raw4 b311 b128 -> b311
node dispatch1509 micro raw5 b311 b130 -> b311
node dispatch1510 micro raw6 b311 b131 -> b311
node dispatch1511 micro raw7 b311 -> b311
node dispatch1512 micro raw35 b311 -> b311
node dispatch1513 micro raw36 b311 -> b311
node dispatch1514 micro raw10 b311 b129 -> b311
node dispatch1515 micro raw1 b311 b125 -> b311
node dispatch1516 micro raw11 b311 b132 -> b311
node dispatch1517 micro raw12 b311 -> b311
node dispatch1518 micro raw13 b311 b133 -> b311
node dispatch1519 micro raw14 b311 b134 -> b311
node dispatch1520 micro raw1 b311 b135 -> b311
node dispatch1521 micro raw2 b311 b137 -> b311
node dispatch1522 micro raw3 b311 b138 -> b311
node dispatch1523 micro raw4 b311 b139 -> b311
node dispatch1524 micro raw5 b311 b141 -> b311
node dispatch1525 micro raw6 b311 b142 -> b311
node dispatch1526 micro raw7 b311 -> b311
node dispatch1527 micro raw37 b311 -> b311
node dispatch1528 micro raw38 b311 -> b311
node dispatch1529 micro raw10 b311 b140 -> b311
node dispatch1530 micro raw1 b311 b136 -> b311
node dispatch1531 micro raw11 b311 b143 -> b311
node dispatch1532 micro raw12 b311 -> b311
node dispatch1533 micro raw13 b311 b144 -> b311
node dispatch1534 micro raw14 b311 b145 -> b311
node dispatch1535 micro raw1 b311 b146 -> b311
node dispatch1536 micro raw2 b311 b148 -> b311
node dispatch1537 micro raw3 b311 b149 -> b311
node dispatch1538 micro raw4 b311 b150 -> b311
node dispatch1539 micro raw5 b311 b152 -> b311
node dispatch1540 micro raw6 b311 b153 -> b311
node dispatch1541 micro raw7 b311 -> b311
node dispatch1542 micro raw39 b311 -> b311
node dispatch1543 micro raw40 b311 -> b311
node dispatch1544 micro raw10 b311 b151 -> b311
node dispatch1545 micro raw1 b311 b147 -> b311
node dispatch1546 micro raw11 b311 b154 -> b311
node dispatch1547 micro raw12 b311 -> b311
node dispatch1548 micro raw13 b311 b155 -> b311
node dispatch1549 micro raw14 b311 b156 -> b311
node dispatch1550 micro raw1 b311 b157 -> b311
node dispatch1551 micro raw2 b311 b159 -> b311
node dispatch1552 micro raw3 b311 b160 -> b311
node dispatch1553 micro raw4 b311 b161 -> b311
node dispatch1554 micro raw5 b311 b163 -> b311
node dispatch1555 micro raw6 b311 b164 -> b311
node dispatch1556 micro raw7 b311 -> b311
node dispatch1557 micro raw41 b311 -> b311
node dispatch1558 micro raw42 b311 -> b311
node dispatch1559 micro raw10 b311 b162 -> b311
node dispatch1560 micro raw1 b311 b158 -> b311
node dispatch1561 micro raw11 b311 b165 -> b311
node dispatch1562 micro raw12 b311 -> b311
node dispatch1563 micro raw13 b311 b166 -> b311
node dispatch1564 micro raw14 b311 b167 -> b311
node dispatch1565 micro raw1 b311 b168 -> b311
node dispatch1566 micro raw2 b311 b170 -> b311
node dispatch1567 micro raw3 b311 b171 -> b311
node dispatch1568 micro raw4 b311 b172 -> b311
node dispatch1569 micro raw5 b311 b174 -> b311
node dispatch1570 micro raw6 b311 b175 -> b311
node dispatch1571 micro raw7 b311 -> b311
node dispatch1572 micro raw43 b311 -> b311
node dispatch1573 micro raw44 b311 -> b311
node dispatch1574 micro raw10 b311 b173 -> b311
node dispatch1575 micro raw1 b311 b169 -> b311
node dispatch1576 micro raw11 b311 b176 -> b311
node dispatch1577 micro raw12 b311 -> b311
node dispatch1578 micro raw13 b311 b177 -> b311
node dispatch1579 micro raw14 b311 b178 -> b311
node dispatch1580 micro raw1 b311 b179 -> b311
node dispatch1581 micro raw2 b311 b181 -> b311
node dispatch1582 micro raw3 b311 b182 -> b311
node dispatch1583 micro raw4 b311 b183 -> b311
node dispatch1584 micro raw5 b311 b185 -> b311
node dispatch1585 micro raw6 b311 b186 -> b311
node dispatch1586 micro raw7 b311 -> b311
node dispatch1587 micro raw45 b311 -> b311
node dispatch1588 micro raw46 b311 -> b311
node dispatch1589 micro raw10 b311 b184 -> b311
node dispatch1590 micro raw1 b311 b180 -> b311
node dispatch1591 micro raw11 b311 b187 -> b311
node dispatch1592 micro raw12 b311 -> b311
node dispatch1593 micro raw13 b311 b188 -> b311
node dispatch1594 micro raw14 b311 b189 -> b311
node dispatch1595 micro raw1 b311 b190 -> b311
node dispatch1596 micro raw2 b311 b192 -> b311
node dispatch1597 micro raw3 b311 b193 -> b311
node dispatch1598 micro raw4 b311 b194 -> b311
node dispatch1599 micro raw5 b311 b196 -> b311
node dispatch1600 micro raw6 b311 b197 -> b311
node dispatch1601 micro raw7 b311 -> b311
node dispatch1602 micro raw47 b311 -> b311
node dispatch1603 micro raw48 b311 -> b311
node dispatch1604 micro raw10 b311 b195 -> b311
node dispatch1605 micro raw1 b311 b191 -> b311
node dispatch1606 micro raw11 b311 b198 -> b311
node dispatch1607 micro raw12 b311 -> b311
node dispatch1608 micro raw13 b311 b199 -> b311
node dispatch1609 micro raw14 b311 b200 -> b311
node dispatch1610 micro raw1 b311 b201 -> b311
node dispatch1611 micro raw2 b311 b203 -> b311
node dispatch1612 micro raw3 b311 b204 -> b311
node dispatch1613 micro raw4 b311 b205 -> b311
node dispatch1614 micro raw5 b311 b207 -> b311
node dispatch1615 micro raw6 b311 b208 -> b311
node dispatch1616 micro raw7 b311 -> b311
node dispatch1617 micro raw49 b311 -> b311
node dispatch1618 micro raw50 b311 -> b311
node dispatch1619 micro raw10 b311 b206 -> b311
node dispatch1620 micro raw1 b311 b202 -> b311
node dispatch1621 micro raw11 b311 b209 -> b311
node dispatch1622 micro raw12 b311 -> b311
node dispatch1623 micro raw13 b311 b210 -> b311
node dispatch1624 micro raw14 b311 b211 -> b311
node dispatch1625 micro raw1 b311 b212 -> b311
node dispatch1626 micro raw2 b311 b214 -> b311
node dispatch1627 micro raw3 b311 b215 -> b311
node dispatch1628 micro raw4 b311 b216 -> b311
node dispatch1629 micro raw5 b311 b218 -> b311
node dispatch1630 micro raw6 b311 b219 -> b311
node dispatch1631 micro raw7 b311 -> b311
node dispatch1632 micro raw51 b311 -> b311
node dispatch1633 micro raw52 b311 -> b311
node dispatch1634 micro raw10 b311 b217 -> b311
node dispatch1635 micro raw1 b311 b213 -> b311
node dispatch1636 micro raw11 b311 b220 -> b311
node dispatch1637 micro raw12 b311 -> b311
node dispatch1638 micro raw13 b311 b221 -> b311
node dispatch1639 micro raw14 b311 b222 -> b311
node dispatch1640 micro raw1 b311 b223 -> b311
node dispatch1641 micro raw2 b311 b225 -> b311
node dispatch1642 micro raw3 b311 b226 -> b311
node dispatch1643 micro raw4 b311 b227 -> b311
node dispatch1644 micro raw5 b311 b229 -> b311
node dispatch1645 micro raw6 b311 b230 -> b311
node dispatch1646 micro raw7 b311 -> b311
node dispatch1647 micro raw53 b311 -> b311
node dispatch1648 micro raw54 b311 -> b311
node dispatch1649 micro raw10 b311 b228 -> b311
node dispatch1650 micro raw1 b311 b224 -> b311
node dispatch1651 micro raw11 b311 b231 -> b311
node dispatch1652 micro raw12 b311 -> b311
node dispatch1653 micro raw13 b311 b232 -> b311
node dispatch1654 micro raw14 b311 b233 -> b311
node dispatch1655 micro raw1 b311 b234 -> b311
node dispatch1656 micro raw2 b311 b236 -> b311
node dispatch1657 micro raw3 b311 b237 -> b311
node dispatch1658 micro raw4 b311 b238 -> b311
node dispatch1659 micro raw5 b311 b240 -> b311
node dispatch1660 micro raw6 b311 b241 -> b311
node dispatch1661 micro raw7 b311 -> b311
node dispatch1662 micro raw55 b311 -> b311
node dispatch1663 micro raw56 b311 -> b311
node dispatch1664 micro raw10 b311 b239 -> b311
node dispatch1665 micro raw1 b311 b235 -> b311
node dispatch1666 micro raw11 b311 b242 -> b311
node dispatch1667 micro raw12 b311 -> b311
node dispatch1668 micro raw13 b311 b243 -> b311
node dispatch1669 micro raw14 b311 b244 -> b311
node dispatch1670 micro raw1 b311 b245 -> b311
node dispatch1671 micro raw2 b311 b247 -> b311
node dispatch1672 micro raw3 b311 b248 -> b311
node dispatch1673 micro raw4 b311 b249 -> b311
node dispatch1674 micro raw5 b311 b251 -> b311
node dispatch1675 micro raw6 b311 b252 -> b311
node dispatch1676 micro raw7 b311 -> b311
node dispatch1677 micro raw57 b311 -> b311
node dispatch1678 micro raw58 b311 -> b311
node dispatch1679 micro raw10 b311 b250 -> b311
node dispatch1680 micro raw1 b311 b246 -> b311
node dispatch1681 micro raw11 b311 b253 -> b311
node dispatch1682 micro raw12 b311 -> b311
node dispatch1683 micro raw13 b311 b254 -> b311
node dispatch1684 micro raw14 b311 b255 -> b311
node dispatch1685 micro raw1 b311 b256 -> b311
node dispatch1686 micro raw2 b311 b258 -> b311
node dispatch1687 micro raw3 b311 b259 -> b311
node dispatch1688 micro raw4 b311 b260 -> b311
node dispatch1689 micro raw5 b311 b262 -> b311
node dispatch1690 micro raw6 b311 b263 -> b311
node dispatch1691 micro raw7 b311 -> b311
node dispatch1692 micro raw59 b311 -> b311
node dispatch1693 micro raw60 b311 -> b311
node dispatch1694 micro raw10 b311 b261 -> b311
node dispatch1695 micro raw1 b311 b257 -> b311
node dispatch1696 micro raw11 b311 b264 -> b311
node dispatch1697 micro raw12 b311 -> b311
node dispatch1698 micro raw13 b311 b265 -> b311
node dispatch1699 micro raw14 b311 b266 -> b311
node dispatch1700 micro raw1 b311 b267 -> b311
node dispatch1701 micro raw2 b311 b269 -> b311
node dispatch1702 micro raw3 b311 b270 -> b311
node dispatch1703 micro raw4 b311 b271 -> b311
node dispatch1704 micro raw5 b311 b273 -> b311
node dispatch1705 micro raw6 b311 b274 -> b311
node dispatch1706 micro raw7 b311 -> b311
node dispatch1707 micro raw61 b311 -> b311
node dispatch1708 micro raw62 b311 -> b311
node dispatch1709 micro raw10 b311 b272 -> b311
node dispatch1710 micro raw1 b311 b268 -> b311
node dispatch1711 micro raw11 b311 b275 -> b311
node dispatch1712 micro raw12 b311 -> b311
node dispatch1713 micro raw13 b311 b276 -> b311
node dispatch1714 micro raw14 b311 b277 -> b311
node dispatch1715 micro raw1 b311 b278 -> b311
node dispatch1716 micro raw2 b311 b280 -> b311
node dispatch1717 micro raw3 b311 b281 -> b311
node dispatch1718 micro raw4 b311 b282 -> b311
node dispatch1719 micro raw5 b311 b284 -> b311
node dispatch1720 micro raw6 b311 b285 -> b311
node dispatch1721 micro raw7 b311 -> b311
node dispatch1722 micro raw63 b311 -> b311
node dispatch1723 micro raw64 b311 -> b311
node dispatch1724 micro raw10 b311 b283 -> b311
node dispatch1725 micro raw1 b311 b279 -> b311
node dispatch1726 micro raw11 b311 b286 -> b311
node dispatch1727 micro raw12 b311 -> b311
node dispatch1728 micro raw13 b311 b287 -> b311
node dispatch1729 micro raw14 b311 b288 -> b311
node dispatch1730 micro raw1 b311 b289 -> b311
node dispatch1731 micro raw2 b311 b291 -> b311
node dispatch1732 micro raw3 b311 b292 -> b311
node dispatch1733 micro raw4 b311 b293 -> b311
node dispatch1734 micro raw5 b311 b295 -> b311
node dispatch1735 micro raw6 b311 b296 -> b311
node dispatch1736 micro raw7 b311 -> b311
node dispatch1737 micro raw65 b311 -> b311
node dispatch1738 micro raw66 b311 -> b311
node dispatch1739 micro raw10 b311 b294 -> b311
node dispatch1740 micro raw1 b311 b290 -> b311
node dispatch1741 micro raw11 b311 b297 -> b311
node dispatch1742 micro raw12 b311 -> b311
node dispatch1743 micro raw13 b311 b298 -> b311
node dispatch1744 micro raw14 b311 b299 -> b311
node dispatch1745 micro raw1 b311 b300 -> b311
node dispatch1746 micro raw2 b311 b302 -> b311
node dispatch1747 micro raw3 b311 b303 -> b311
node dispatch1748 micro raw4 b311 b304 -> b311
node dispatch1749 micro raw5 b311 b306 -> b311
node dispatch1750 micro raw6 b311 b307 -> b311
node dispatch1751 micro raw7 b311 -> b311
node dispatch1752 micro raw67 b311 -> b311
node dispatch1753 micro raw68 b311 -> b311
node dispatch1754 micro raw10 b311 b305 -> b311
node dispatch1755 micro raw1 b311 b301 -> b311
node dispatch1756 micro raw11 b311 b308 -> b311
node dispatch1757 micro raw12 b311 -> b311
node dispatch1758 micro raw13 b311 b309 -> b311
node dispatch1759 micro raw14 b311 b310 -> b311
node dispatch1760 micro raw1 b311 b2 -> b311
node dispatch1761 micro raw69 b312 b311 b1 -> b312

