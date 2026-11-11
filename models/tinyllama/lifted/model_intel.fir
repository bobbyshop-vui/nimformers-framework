fir 1
model tinyllama
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
buf t718_embedding f32 131072
buf t722_y f32 131072
buf t724_c f32 131072
buf t726_c f32 16384
buf t728_c f32 16384
buf position f32 1 input:position
buf s510 f32 524288
buf s511 f32 524288
buf t734_o f32 131072
buf t736_c f32 131072
buf t738_c f32 131072
buf t740_y f32 131072
buf t742_c f32 360448
buf t744_c f32 360448
buf t746_y f32 360448
buf t748_c f32 360448
buf t750_c f32 360448
buf t752_c f32 131072
buf t754_c f32 131072
buf t773_y f32 131072
buf t775_c f32 131072
buf t777_c f32 16384
buf t779_c f32 16384
buf s512 f32 524288
buf s513 f32 524288
buf t785_o f32 131072
buf t787_c f32 131072
buf t789_c f32 131072
buf t791_y f32 131072
buf t793_c f32 360448
buf t795_c f32 360448
buf t797_y f32 360448
buf t799_c f32 360448
buf t801_c f32 360448
buf t803_c f32 131072
buf t805_c f32 131072
buf t807_y f32 131072
buf t809_c f32 131072
buf t811_c f32 16384
buf t813_c f32 16384
buf s514 f32 524288
buf s515 f32 524288
buf t819_o f32 131072
buf t821_c f32 131072
buf t823_c f32 131072
buf t825_y f32 131072
buf t827_c f32 360448
buf t829_c f32 360448
buf t831_y f32 360448
buf t833_c f32 360448
buf t835_c f32 360448
buf t837_c f32 131072
buf t839_c f32 131072
buf t841_y f32 131072
buf t843_c f32 131072
buf t845_c f32 16384
buf t847_c f32 16384
buf s516 f32 524288
buf s517 f32 524288
buf t853_o f32 131072
buf t855_c f32 131072
buf t857_c f32 131072
buf t859_y f32 131072
buf t861_c f32 360448
buf t863_c f32 360448
buf t865_y f32 360448
buf t867_c f32 360448
buf t869_c f32 360448
buf t871_c f32 131072
buf t873_c f32 131072
buf t875_y f32 131072
buf t877_c f32 131072
buf t879_c f32 16384
buf t881_c f32 16384
buf s518 f32 524288
buf s519 f32 524288
buf t887_o f32 131072
buf t889_c f32 131072
buf t891_c f32 131072
buf t893_y f32 131072
buf t895_c f32 360448
buf t897_c f32 360448
buf t899_y f32 360448
buf t901_c f32 360448
buf t903_c f32 360448
buf t905_c f32 131072
buf t907_c f32 131072
buf t909_y f32 131072
buf t911_c f32 131072
buf t913_c f32 16384
buf t915_c f32 16384
buf s520 f32 524288
buf s521 f32 524288
buf t921_o f32 131072
buf t923_c f32 131072
buf t925_c f32 131072
buf t927_y f32 131072
buf t929_c f32 360448
buf t931_c f32 360448
buf t933_y f32 360448
buf t935_c f32 360448
buf t937_c f32 360448
buf t939_c f32 131072
buf t941_c f32 131072
buf t943_y f32 131072
buf t945_c f32 131072
buf t947_c f32 16384
buf t949_c f32 16384
buf s522 f32 524288
buf s523 f32 524288
buf t955_o f32 131072
buf t957_c f32 131072
buf t959_c f32 131072
buf t961_y f32 131072
buf t963_c f32 360448
buf t965_c f32 360448
buf t967_y f32 360448
buf t969_c f32 360448
buf t971_c f32 360448
buf t973_c f32 131072
buf t975_c f32 131072
buf t977_y f32 131072
buf t979_c f32 131072
buf t981_c f32 16384
buf t983_c f32 16384
buf s524 f32 524288
buf s525 f32 524288
buf t989_o f32 131072
buf t991_c f32 131072
buf t993_c f32 131072
buf t995_y f32 131072
buf t997_c f32 360448
buf t999_c f32 360448
buf t1001_y f32 360448
buf t1003_c f32 360448
buf t1005_c f32 360448
buf t1007_c f32 131072
buf t1009_c f32 131072
buf t1011_y f32 131072
buf t1013_c f32 131072
buf t1015_c f32 16384
buf t1017_c f32 16384
buf s526 f32 524288
buf s527 f32 524288
buf t1023_o f32 131072
buf t1025_c f32 131072
buf t1027_c f32 131072
buf t1029_y f32 131072
buf t1031_c f32 360448
buf t1033_c f32 360448
buf t1035_y f32 360448
buf t1037_c f32 360448
buf t1039_c f32 360448
buf t1041_c f32 131072
buf t1043_c f32 131072
buf t1167_y f32 131072
buf t1169_c f32 131072
buf t1171_c f32 16384
buf t1173_c f32 16384
buf s528 f32 524288
buf s529 f32 524288
buf t1179_o f32 131072
buf t1181_c f32 131072
buf t1183_c f32 131072
buf t1185_y f32 131072
buf t1187_c f32 360448
buf t1189_c f32 360448
buf t1195_c f32 360448
buf t1197_c f32 131072
buf t1199_c f32 131072
buf t1201_y f32 131072
buf t1203_c f32 131072
buf t1205_c f32 16384
buf t1207_c f32 16384
buf s530 f32 524288
buf s531 f32 524288
buf t1213_o f32 131072
buf t1215_c f32 131072
buf t1217_c f32 131072
buf t1219_y f32 131072
buf t1221_c f32 360448
buf t1223_c f32 360448
buf t1229_c f32 360448
buf t1231_c f32 131072
buf t1233_c f32 131072
buf t1235_y f32 131072
buf t1237_c f32 131072
buf t1239_c f32 16384
buf t1241_c f32 16384
buf s532 f32 524288
buf s533 f32 524288
buf t1247_o f32 131072
buf t1249_c f32 131072
buf t1251_c f32 131072
buf t1253_y f32 131072
buf t1255_c f32 360448
buf t1257_c f32 360448
buf t1263_c f32 360448
buf t1265_c f32 131072
buf t1267_c f32 131072
buf t1269_y f32 131072
buf t1271_c f32 131072
buf t1273_c f32 16384
buf t1275_c f32 16384
buf s534 f32 524288
buf s535 f32 524288
buf t1281_o f32 131072
buf t1283_c f32 131072
buf t1285_c f32 131072
buf t1287_y f32 131072
buf t1289_c f32 360448
buf t1291_c f32 360448
buf t1293_y f32 360448
buf t1295_c f32 360448
buf t1297_c f32 360448
buf t1299_c f32 131072
buf t1301_c f32 131072
buf t1303_y f32 131072
buf t1305_c f32 131072
buf t1307_c f32 16384
buf t1309_c f32 16384
buf s536 f32 524288
buf s537 f32 524288
buf t1315_o f32 131072
buf t1317_c f32 131072
buf t1319_c f32 131072
buf t1321_y f32 131072
buf t1323_c f32 360448
buf t1325_c f32 360448
buf t1327_y f32 360448
buf t1329_c f32 360448
buf t1331_c f32 360448
buf t1333_c f32 131072
buf t1335_c f32 131072
buf t1337_y f32 131072
buf t1339_c f32 131072
buf t1341_c f32 16384
buf t1343_c f32 16384
buf s538 f32 524288
buf s539 f32 524288
buf t1349_o f32 131072
buf t1351_c f32 131072
buf t1353_c f32 131072
buf t1355_y f32 131072
buf t1357_c f32 360448
buf t1359_c f32 360448
buf t1361_y f32 360448
buf t1363_c f32 360448
buf t1365_c f32 360448
buf t1367_c f32 131072
buf t1369_c f32 131072
buf t1371_y f32 131072
buf t1373_c f32 131072
buf t1375_c f32 16384
buf t1377_c f32 16384
buf s540 f32 524288
buf s541 f32 524288
buf t1383_o f32 131072
buf t1385_c f32 131072
buf t1387_c f32 131072
buf t1389_y f32 131072
buf t1391_c f32 360448
buf t1393_c f32 360448
buf t1395_y f32 360448
buf t1397_c f32 360448
buf t1399_c f32 360448
buf t1401_c f32 131072
buf t1403_c f32 131072
buf t1405_y f32 131072
buf t1407_c f32 131072
buf t1409_c f32 16384
buf t1411_c f32 16384
buf s542 f32 524288
buf s543 f32 524288
buf t1417_o f32 131072
buf t1419_c f32 131072
buf t1421_c f32 131072
buf t1423_y f32 131072
buf t1425_c f32 360448
buf t1427_c f32 360448
buf t1429_y f32 360448
buf t1431_c f32 360448
buf t1433_c f32 360448
buf t1435_c f32 131072
buf t1437_c f32 131072
buf t1561_y f32 131072
buf t1563_c f32 131072
buf t1565_c f32 16384
buf t1567_c f32 16384
buf s544 f32 524288
buf s545 f32 524288
buf t1573_o f32 131072
buf t1575_c f32 131072
buf t1577_c f32 131072
buf t1579_y f32 131072
buf t1581_c f32 360448
buf t1583_c f32 360448
buf t1589_c f32 360448
buf t1591_c f32 131072
buf t1593_c f32 131072
buf t1595_y f32 131072
buf t1597_c f32 131072
buf t1599_c f32 16384
buf t1601_c f32 16384
buf s546 f32 524288
buf s547 f32 524288
buf t1607_o f32 131072
buf t1609_c f32 131072
buf t1611_c f32 131072
buf t1613_y f32 131072
buf t1615_c f32 360448
buf t1617_c f32 360448
buf t1623_c f32 360448
buf t1625_c f32 131072
buf t1627_c f32 131072
buf t1629_y f32 131072
buf t1631_c f32 131072
buf t1633_c f32 16384
buf t1635_c f32 16384
buf s548 f32 524288
buf s549 f32 524288
buf t1641_o f32 131072
buf t1643_c f32 131072
buf t1645_c f32 131072
buf t1647_y f32 131072
buf t1649_c f32 360448
buf t1651_c f32 360448
buf t1657_c f32 360448
buf t1659_c f32 131072
buf t1661_c f32 131072
buf t1663_y f32 131072
buf t1665_c f32 131072
buf t1667_c f32 16384
buf t1669_c f32 16384
buf s550 f32 524288
buf s551 f32 524288
buf t1675_o f32 131072
buf t1677_c f32 131072
buf t1679_c f32 131072
buf t1681_y f32 131072
buf t1683_c f32 360448
buf t1685_c f32 360448
buf t1691_c f32 360448
buf t1693_c f32 131072
buf t1695_c f32 131072
buf t1697_y f32 131072
buf t1701_c f32 16384
buf t1703_c f32 16384
buf s552 f32 524288
buf s553 f32 524288
buf t1821_embedding f32 2048
buf t1826_y f32 2048
buf t1827_c f32 2048
buf t1828_c f32 256
buf t1829_c f32 256
buf t1832_o f32 2048
buf t1835_c f32 2048
buf t1837_c f32 2048
buf t1842_y f32 2048
buf t1843_c f32 5632
buf t1844_c f32 5632
buf t1850_c f32 5632
buf t1853_c f32 2048
buf t1855_c f32 2048
buf t1860_y f32 2048
buf t1861_c f32 2048
buf t1862_c f32 256
buf t1863_c f32 256
buf t1866_o f32 2048
buf t1869_c f32 2048
buf t1871_c f32 2048
buf t1876_y f32 2048
buf t1877_c f32 5632
buf t1878_c f32 5632
buf t1884_c f32 5632
buf t1887_c f32 2048
buf t1889_c f32 2048
buf t1894_y f32 2048
buf t1895_c f32 2048
buf t1896_c f32 256
buf t1897_c f32 256
buf t1900_o f32 2048
buf t1903_c f32 2048
buf t1905_c f32 2048
buf t1910_y f32 2048
buf t1911_c f32 5632
buf t1912_c f32 5632
buf t1918_c f32 5632
buf t1921_c f32 2048
buf t1923_c f32 2048
buf t1928_y f32 2048
buf t1929_c f32 2048
buf t1930_c f32 256
buf t1931_c f32 256
buf t1934_o f32 2048
buf t1937_c f32 2048
buf t1939_c f32 2048
buf t1944_y f32 2048
buf t1945_c f32 5632
buf t1946_c f32 5632
buf t1952_c f32 5632
buf t1955_c f32 2048
buf t1957_c f32 2048
buf t1962_y f32 2048
buf t1963_c f32 2048
buf t1964_c f32 256
buf t1965_c f32 256
buf t1968_o f32 2048
buf t1971_c f32 2048
buf t1973_c f32 2048
buf t1978_y f32 2048
buf t1979_c f32 5632
buf t1980_c f32 5632
buf t1986_c f32 5632
buf t1989_c f32 2048
buf t1991_c f32 2048
buf t1996_y f32 2048
buf t1997_c f32 2048
buf t1998_c f32 256
buf t1999_c f32 256
buf t2002_o f32 2048
buf t2005_c f32 2048
buf t2007_c f32 2048
buf t2012_y f32 2048
buf t2013_c f32 5632
buf t2014_c f32 5632
buf t2020_c f32 5632
buf t2023_c f32 2048
buf t2025_c f32 2048
buf t2030_y f32 2048
buf t2031_c f32 2048
buf t2032_c f32 256
buf t2033_c f32 256
buf t2036_o f32 2048
buf t2039_c f32 2048
buf t2041_c f32 2048
buf t2046_y f32 2048
buf t2047_c f32 5632
buf t2048_c f32 5632
buf t2054_c f32 5632
buf t2057_c f32 2048
buf t2059_c f32 2048
buf t2064_y f32 2048
buf t2065_c f32 2048
buf t2066_c f32 256
buf t2067_c f32 256
buf t2070_o f32 2048
buf t2073_c f32 2048
buf t2075_c f32 2048
buf t2080_y f32 2048
buf t2081_c f32 5632
buf t2082_c f32 5632
buf t2088_c f32 5632
buf t2091_c f32 2048
buf t2093_c f32 2048
buf t2098_y f32 2048
buf t2099_c f32 2048
buf t2100_c f32 256
buf t2101_c f32 256
buf t2104_o f32 2048
buf t2107_c f32 2048
buf t2109_c f32 2048
buf t2114_y f32 2048
buf t2115_c f32 5632
buf t2116_c f32 5632
buf t2122_c f32 5632
buf t2125_c f32 2048
buf t2127_c f32 2048
buf t2132_y f32 2048
buf t2133_c f32 2048
buf t2134_c f32 256
buf t2135_c f32 256
buf t2138_o f32 2048
buf t2141_c f32 2048
buf t2143_c f32 2048
buf t2148_y f32 2048
buf t2149_c f32 5632
buf t2150_c f32 5632
buf t2156_c f32 5632
buf t2159_c f32 2048
buf t2161_c f32 2048
buf t2166_y f32 2048
buf t2167_c f32 2048
buf t2168_c f32 256
buf t2169_c f32 256
buf t2172_o f32 2048
buf t2175_c f32 2048
buf t2177_c f32 2048
buf t2182_y f32 2048
buf t2183_c f32 5632
buf t2184_c f32 5632
buf t2190_c f32 5632
buf t2193_c f32 2048
buf t2195_c f32 2048
buf t2200_y f32 2048
buf t2201_c f32 2048
buf t2202_c f32 256
buf t2203_c f32 256
buf t2206_o f32 2048
buf t2209_c f32 2048
buf t2211_c f32 2048
buf t2216_y f32 2048
buf t2217_c f32 5632
buf t2218_c f32 5632
buf t2224_c f32 5632
buf t2227_c f32 2048
buf t2229_c f32 2048
buf t2234_y f32 2048
buf t2235_c f32 2048
buf t2236_c f32 256
buf t2237_c f32 256
buf t2240_o f32 2048
buf t2243_c f32 2048
buf t2245_c f32 2048
buf t2250_y f32 2048
buf t2251_c f32 5632
buf t2252_c f32 5632
buf t2258_c f32 5632
buf t2261_c f32 2048
buf t2263_c f32 2048
buf t2268_y f32 2048
buf t2269_c f32 2048
buf t2270_c f32 256
buf t2271_c f32 256
buf t2274_o f32 2048
buf t2277_c f32 2048
buf t2279_c f32 2048
buf t2284_y f32 2048
buf t2285_c f32 5632
buf t2286_c f32 5632
buf t2292_c f32 5632
buf t2295_c f32 2048
buf t2297_c f32 2048
buf t2302_y f32 2048
buf t2303_c f32 2048
buf t2304_c f32 256
buf t2305_c f32 256
buf t2308_o f32 2048
buf t2311_c f32 2048
buf t2313_c f32 2048
buf t2318_y f32 2048
buf t2319_c f32 5632
buf t2320_c f32 5632
buf t2326_c f32 5632
buf t2329_c f32 2048
buf t2331_c f32 2048
buf t2336_y f32 2048
buf t2337_c f32 2048
buf t2338_c f32 256
buf t2339_c f32 256
buf t2342_o f32 2048
buf t2345_c f32 2048
buf t2347_c f32 2048
buf t2352_y f32 2048
buf t2353_c f32 5632
buf t2354_c f32 5632
buf t2360_c f32 5632
buf t2363_c f32 2048
buf t2365_c f32 2048
buf t2370_y f32 2048
buf t2371_c f32 2048
buf t2372_c f32 256
buf t2373_c f32 256
buf t2376_o f32 2048
buf t2379_c f32 2048
buf t2381_c f32 2048
buf t2386_y f32 2048
buf t2387_c f32 5632
buf t2388_c f32 5632
buf t2394_c f32 5632
buf t2397_c f32 2048
buf t2399_c f32 2048
buf t2404_y f32 2048
buf t2405_c f32 2048
buf t2406_c f32 256
buf t2407_c f32 256
buf t2410_o f32 2048
buf t2413_c f32 2048
buf t2415_c f32 2048
buf t2420_y f32 2048
buf t2421_c f32 5632
buf t2422_c f32 5632
buf t2428_c f32 5632
buf t2431_c f32 2048
buf t2433_c f32 2048
buf t2438_y f32 2048
buf t2439_c f32 2048
buf t2440_c f32 256
buf t2441_c f32 256
buf t2444_o f32 2048
buf t2447_c f32 2048
buf t2449_c f32 2048
buf t2454_y f32 2048
buf t2455_c f32 5632
buf t2456_c f32 5632
buf t2462_c f32 5632
buf t2465_c f32 2048
buf t2467_c f32 2048
buf t2472_y f32 2048
buf t2473_c f32 2048
buf t2474_c f32 256
buf t2475_c f32 256
buf t2478_o f32 2048
buf t2481_c f32 2048
buf t2483_c f32 2048
buf t2488_y f32 2048
buf t2489_c f32 5632
buf t2490_c f32 5632
buf t2496_c f32 5632
buf t2499_c f32 2048
buf t2501_c f32 2048
buf t2506_y f32 2048
buf t2507_c f32 2048
buf t2508_c f32 256
buf t2509_c f32 256
buf t2512_o f32 2048
buf t2515_c f32 2048
buf t2517_c f32 2048
buf t2522_y f32 2048
buf t2523_c f32 5632
buf t2524_c f32 5632
buf t2530_c f32 5632
buf t2533_c f32 2048
buf t2535_c f32 2048
buf t2540_y f32 2048
buf t2541_c f32 2048
buf t2542_c f32 256
buf t2543_c f32 256
buf t2546_o f32 2048
buf t2549_c f32 2048
buf t2551_c f32 2048
buf t2556_y f32 2048
buf t2557_c f32 5632
buf t2558_c f32 5632
buf t2564_c f32 5632
buf t2567_c f32 2048
buf t2569_c f32 2048
buf t2572_y f32 2048
buf t2573_c f32 32000 output:logits
weight model_embed_tokens_weight_emb f32 model.embed_tokens.weight tensor [32000,2048]
weight model_layers_0_input_layernorm_weight f32 model.layers.0.input_layernorm.weight tensor [2048]
weight model_layers_0_self_attn_q_proj_weight_q q4b128 model.layers.0.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_0_self_attn_k_proj_weight_q q4b128 model.layers.0.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_0_self_attn_v_proj_weight_q q4b128 model.layers.0.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_0_self_attn_o_proj_weight_q q4b128 model.layers.0.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_0_post_attention_layernorm_weight f32 model.layers.0.post_attention_layernorm.weight tensor [2048]
weight model_layers_0_mlp_gate_proj_weight_q q4b128 model.layers.0.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_0_mlp_up_proj_weight_q q4b128 model.layers.0.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_0_mlp_down_proj_weight_q q4b128 model.layers.0.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_1_input_layernorm_weight f32 model.layers.1.input_layernorm.weight tensor [2048]
weight model_layers_1_self_attn_q_proj_weight_q q4b128 model.layers.1.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_1_self_attn_k_proj_weight_q q4b128 model.layers.1.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_1_self_attn_v_proj_weight_q q4b128 model.layers.1.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_1_self_attn_o_proj_weight_q q4b128 model.layers.1.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_1_post_attention_layernorm_weight f32 model.layers.1.post_attention_layernorm.weight tensor [2048]
weight model_layers_1_mlp_gate_proj_weight_q q4b128 model.layers.1.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_1_mlp_up_proj_weight_q q4b128 model.layers.1.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_1_mlp_down_proj_weight_q q4b128 model.layers.1.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_2_input_layernorm_weight f32 model.layers.2.input_layernorm.weight tensor [2048]
weight model_layers_2_self_attn_q_proj_weight_q q4b128 model.layers.2.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_2_self_attn_k_proj_weight_q q4b128 model.layers.2.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_2_self_attn_v_proj_weight_q q4b128 model.layers.2.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_2_self_attn_o_proj_weight_q q4b128 model.layers.2.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_2_post_attention_layernorm_weight f32 model.layers.2.post_attention_layernorm.weight tensor [2048]
weight model_layers_2_mlp_gate_proj_weight_q q4b128 model.layers.2.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_2_mlp_up_proj_weight_q q4b128 model.layers.2.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_2_mlp_down_proj_weight_q q4b128 model.layers.2.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_3_input_layernorm_weight f32 model.layers.3.input_layernorm.weight tensor [2048]
weight model_layers_3_self_attn_q_proj_weight_q q4b128 model.layers.3.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_3_self_attn_k_proj_weight_q q4b128 model.layers.3.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_3_self_attn_v_proj_weight_q q4b128 model.layers.3.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_3_self_attn_o_proj_weight_q q4b128 model.layers.3.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_3_post_attention_layernorm_weight f32 model.layers.3.post_attention_layernorm.weight tensor [2048]
weight model_layers_3_mlp_gate_proj_weight_q q4b128 model.layers.3.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_3_mlp_up_proj_weight_q q4b128 model.layers.3.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_3_mlp_down_proj_weight_q q4b128 model.layers.3.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_4_input_layernorm_weight f32 model.layers.4.input_layernorm.weight tensor [2048]
weight model_layers_4_self_attn_q_proj_weight_q q4b128 model.layers.4.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_4_self_attn_k_proj_weight_q q4b128 model.layers.4.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_4_self_attn_v_proj_weight_q q4b128 model.layers.4.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_4_self_attn_o_proj_weight_q q4b128 model.layers.4.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_4_post_attention_layernorm_weight f32 model.layers.4.post_attention_layernorm.weight tensor [2048]
weight model_layers_4_mlp_gate_proj_weight_q q4b128 model.layers.4.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_4_mlp_up_proj_weight_q q4b128 model.layers.4.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_4_mlp_down_proj_weight_q q4b128 model.layers.4.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_5_input_layernorm_weight f32 model.layers.5.input_layernorm.weight tensor [2048]
weight model_layers_5_self_attn_q_proj_weight_q q4b128 model.layers.5.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_5_self_attn_k_proj_weight_q q4b128 model.layers.5.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_5_self_attn_v_proj_weight_q q4b128 model.layers.5.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_5_self_attn_o_proj_weight_q q4b128 model.layers.5.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_5_post_attention_layernorm_weight f32 model.layers.5.post_attention_layernorm.weight tensor [2048]
weight model_layers_5_mlp_gate_proj_weight_q q4b128 model.layers.5.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_5_mlp_up_proj_weight_q q4b128 model.layers.5.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_5_mlp_down_proj_weight_q q4b128 model.layers.5.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_6_input_layernorm_weight f32 model.layers.6.input_layernorm.weight tensor [2048]
weight model_layers_6_self_attn_q_proj_weight_q q4b128 model.layers.6.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_6_self_attn_k_proj_weight_q q4b128 model.layers.6.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_6_self_attn_v_proj_weight_q q4b128 model.layers.6.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_6_self_attn_o_proj_weight_q q4b128 model.layers.6.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_6_post_attention_layernorm_weight f32 model.layers.6.post_attention_layernorm.weight tensor [2048]
weight model_layers_6_mlp_gate_proj_weight_q q4b128 model.layers.6.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_6_mlp_up_proj_weight_q q4b128 model.layers.6.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_6_mlp_down_proj_weight_q q4b128 model.layers.6.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_7_input_layernorm_weight f32 model.layers.7.input_layernorm.weight tensor [2048]
weight model_layers_7_self_attn_q_proj_weight_q q4b128 model.layers.7.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_7_self_attn_k_proj_weight_q q4b128 model.layers.7.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_7_self_attn_v_proj_weight_q q4b128 model.layers.7.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_7_self_attn_o_proj_weight_q q4b128 model.layers.7.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_7_post_attention_layernorm_weight f32 model.layers.7.post_attention_layernorm.weight tensor [2048]
weight model_layers_7_mlp_gate_proj_weight_q q4b128 model.layers.7.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_7_mlp_up_proj_weight_q q4b128 model.layers.7.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_7_mlp_down_proj_weight_q q4b128 model.layers.7.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_8_input_layernorm_weight f32 model.layers.8.input_layernorm.weight tensor [2048]
weight model_layers_8_self_attn_q_proj_weight_q q4b128 model.layers.8.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_8_self_attn_k_proj_weight_q q4b128 model.layers.8.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_8_self_attn_v_proj_weight_q q4b128 model.layers.8.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_8_self_attn_o_proj_weight_q q4b128 model.layers.8.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_8_post_attention_layernorm_weight f32 model.layers.8.post_attention_layernorm.weight tensor [2048]
weight model_layers_8_mlp_gate_proj_weight_q q4b128 model.layers.8.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_8_mlp_up_proj_weight_q q4b128 model.layers.8.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_8_mlp_down_proj_weight_q q4b128 model.layers.8.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_9_input_layernorm_weight f32 model.layers.9.input_layernorm.weight tensor [2048]
weight model_layers_9_self_attn_q_proj_weight_q q4b128 model.layers.9.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_9_self_attn_k_proj_weight_q q4b128 model.layers.9.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_9_self_attn_v_proj_weight_q q4b128 model.layers.9.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_9_self_attn_o_proj_weight_q q4b128 model.layers.9.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_9_post_attention_layernorm_weight f32 model.layers.9.post_attention_layernorm.weight tensor [2048]
weight model_layers_9_mlp_gate_proj_weight_q q4b128 model.layers.9.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_9_mlp_up_proj_weight_q q4b128 model.layers.9.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_9_mlp_down_proj_weight_q q4b128 model.layers.9.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_10_input_layernorm_weight f32 model.layers.10.input_layernorm.weight tensor [2048]
weight model_layers_10_self_attn_q_proj_weight_q q4b128 model.layers.10.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_10_self_attn_k_proj_weight_q q4b128 model.layers.10.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_10_self_attn_v_proj_weight_q q4b128 model.layers.10.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_10_self_attn_o_proj_weight_q q4b128 model.layers.10.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_10_post_attention_layernorm_weight f32 model.layers.10.post_attention_layernorm.weight tensor [2048]
weight model_layers_10_mlp_gate_proj_weight_q q4b128 model.layers.10.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_10_mlp_up_proj_weight_q q4b128 model.layers.10.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_10_mlp_down_proj_weight_q q4b128 model.layers.10.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_11_input_layernorm_weight f32 model.layers.11.input_layernorm.weight tensor [2048]
weight model_layers_11_self_attn_q_proj_weight_q q4b128 model.layers.11.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_11_self_attn_k_proj_weight_q q4b128 model.layers.11.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_11_self_attn_v_proj_weight_q q4b128 model.layers.11.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_11_self_attn_o_proj_weight_q q4b128 model.layers.11.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_11_post_attention_layernorm_weight f32 model.layers.11.post_attention_layernorm.weight tensor [2048]
weight model_layers_11_mlp_gate_proj_weight_q q4b128 model.layers.11.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_11_mlp_up_proj_weight_q q4b128 model.layers.11.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_11_mlp_down_proj_weight_q q4b128 model.layers.11.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_12_input_layernorm_weight f32 model.layers.12.input_layernorm.weight tensor [2048]
weight model_layers_12_self_attn_q_proj_weight_q q4b128 model.layers.12.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_12_self_attn_k_proj_weight_q q4b128 model.layers.12.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_12_self_attn_v_proj_weight_q q4b128 model.layers.12.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_12_self_attn_o_proj_weight_q q4b128 model.layers.12.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_12_post_attention_layernorm_weight f32 model.layers.12.post_attention_layernorm.weight tensor [2048]
weight model_layers_12_mlp_gate_proj_weight_q q4b128 model.layers.12.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_12_mlp_up_proj_weight_q q4b128 model.layers.12.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_12_mlp_down_proj_weight_q q4b128 model.layers.12.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_13_input_layernorm_weight f32 model.layers.13.input_layernorm.weight tensor [2048]
weight model_layers_13_self_attn_q_proj_weight_q q4b128 model.layers.13.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_13_self_attn_k_proj_weight_q q4b128 model.layers.13.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_13_self_attn_v_proj_weight_q q4b128 model.layers.13.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_13_self_attn_o_proj_weight_q q4b128 model.layers.13.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_13_post_attention_layernorm_weight f32 model.layers.13.post_attention_layernorm.weight tensor [2048]
weight model_layers_13_mlp_gate_proj_weight_q q4b128 model.layers.13.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_13_mlp_up_proj_weight_q q4b128 model.layers.13.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_13_mlp_down_proj_weight_q q4b128 model.layers.13.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_14_input_layernorm_weight f32 model.layers.14.input_layernorm.weight tensor [2048]
weight model_layers_14_self_attn_q_proj_weight_q q4b128 model.layers.14.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_14_self_attn_k_proj_weight_q q4b128 model.layers.14.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_14_self_attn_v_proj_weight_q q4b128 model.layers.14.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_14_self_attn_o_proj_weight_q q4b128 model.layers.14.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_14_post_attention_layernorm_weight f32 model.layers.14.post_attention_layernorm.weight tensor [2048]
weight model_layers_14_mlp_gate_proj_weight_q q4b128 model.layers.14.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_14_mlp_up_proj_weight_q q4b128 model.layers.14.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_14_mlp_down_proj_weight_q q4b128 model.layers.14.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_15_input_layernorm_weight f32 model.layers.15.input_layernorm.weight tensor [2048]
weight model_layers_15_self_attn_q_proj_weight_q q4b128 model.layers.15.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_15_self_attn_k_proj_weight_q q4b128 model.layers.15.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_15_self_attn_v_proj_weight_q q4b128 model.layers.15.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_15_self_attn_o_proj_weight_q q4b128 model.layers.15.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_15_post_attention_layernorm_weight f32 model.layers.15.post_attention_layernorm.weight tensor [2048]
weight model_layers_15_mlp_gate_proj_weight_q q4b128 model.layers.15.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_15_mlp_up_proj_weight_q q4b128 model.layers.15.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_15_mlp_down_proj_weight_q q4b128 model.layers.15.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_16_input_layernorm_weight f32 model.layers.16.input_layernorm.weight tensor [2048]
weight model_layers_16_self_attn_q_proj_weight_q q4b128 model.layers.16.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_16_self_attn_k_proj_weight_q q4b128 model.layers.16.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_16_self_attn_v_proj_weight_q q4b128 model.layers.16.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_16_self_attn_o_proj_weight_q q4b128 model.layers.16.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_16_post_attention_layernorm_weight f32 model.layers.16.post_attention_layernorm.weight tensor [2048]
weight model_layers_16_mlp_gate_proj_weight_q q4b128 model.layers.16.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_16_mlp_up_proj_weight_q q4b128 model.layers.16.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_16_mlp_down_proj_weight_q q4b128 model.layers.16.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_17_input_layernorm_weight f32 model.layers.17.input_layernorm.weight tensor [2048]
weight model_layers_17_self_attn_q_proj_weight_q q4b128 model.layers.17.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_17_self_attn_k_proj_weight_q q4b128 model.layers.17.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_17_self_attn_v_proj_weight_q q4b128 model.layers.17.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_17_self_attn_o_proj_weight_q q4b128 model.layers.17.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_17_post_attention_layernorm_weight f32 model.layers.17.post_attention_layernorm.weight tensor [2048]
weight model_layers_17_mlp_gate_proj_weight_q q4b128 model.layers.17.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_17_mlp_up_proj_weight_q q4b128 model.layers.17.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_17_mlp_down_proj_weight_q q4b128 model.layers.17.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_18_input_layernorm_weight f32 model.layers.18.input_layernorm.weight tensor [2048]
weight model_layers_18_self_attn_q_proj_weight_q q4b128 model.layers.18.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_18_self_attn_k_proj_weight_q q4b128 model.layers.18.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_18_self_attn_v_proj_weight_q q4b128 model.layers.18.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_18_self_attn_o_proj_weight_q q4b128 model.layers.18.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_18_post_attention_layernorm_weight f32 model.layers.18.post_attention_layernorm.weight tensor [2048]
weight model_layers_18_mlp_gate_proj_weight_q q4b128 model.layers.18.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_18_mlp_up_proj_weight_q q4b128 model.layers.18.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_18_mlp_down_proj_weight_q q4b128 model.layers.18.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_19_input_layernorm_weight f32 model.layers.19.input_layernorm.weight tensor [2048]
weight model_layers_19_self_attn_q_proj_weight_q q4b128 model.layers.19.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_19_self_attn_k_proj_weight_q q4b128 model.layers.19.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_19_self_attn_v_proj_weight_q q4b128 model.layers.19.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_19_self_attn_o_proj_weight_q q4b128 model.layers.19.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_19_post_attention_layernorm_weight f32 model.layers.19.post_attention_layernorm.weight tensor [2048]
weight model_layers_19_mlp_gate_proj_weight_q q4b128 model.layers.19.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_19_mlp_up_proj_weight_q q4b128 model.layers.19.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_19_mlp_down_proj_weight_q q4b128 model.layers.19.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_20_input_layernorm_weight f32 model.layers.20.input_layernorm.weight tensor [2048]
weight model_layers_20_self_attn_q_proj_weight_q q4b128 model.layers.20.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_20_self_attn_k_proj_weight_q q4b128 model.layers.20.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_20_self_attn_v_proj_weight_q q4b128 model.layers.20.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_20_self_attn_o_proj_weight_q q4b128 model.layers.20.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_20_post_attention_layernorm_weight f32 model.layers.20.post_attention_layernorm.weight tensor [2048]
weight model_layers_20_mlp_gate_proj_weight_q q4b128 model.layers.20.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_20_mlp_up_proj_weight_q q4b128 model.layers.20.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_20_mlp_down_proj_weight_q q4b128 model.layers.20.mlp.down_proj.weight tensor [2048,5632]
weight model_layers_21_input_layernorm_weight f32 model.layers.21.input_layernorm.weight tensor [2048]
weight model_layers_21_self_attn_q_proj_weight_q q4b128 model.layers.21.self_attn.q_proj.weight tensor [2048,2048]
weight model_layers_21_self_attn_k_proj_weight_q q4b128 model.layers.21.self_attn.k_proj.weight tensor [256,2048]
weight model_layers_21_self_attn_v_proj_weight_q q4b128 model.layers.21.self_attn.v_proj.weight tensor [256,2048]
weight model_layers_21_self_attn_o_proj_weight_q q4b128 model.layers.21.self_attn.o_proj.weight tensor [2048,2048]
weight model_layers_21_post_attention_layernorm_weight f32 model.layers.21.post_attention_layernorm.weight tensor [2048]
weight model_layers_21_mlp_gate_proj_weight_q q4b128 model.layers.21.mlp.gate_proj.weight tensor [5632,2048]
weight model_layers_21_mlp_up_proj_weight_q q4b128 model.layers.21.mlp.up_proj.weight tensor [5632,2048]
weight model_layers_21_mlp_down_proj_weight_q q4b128 model.layers.21.mlp.down_proj.weight tensor [2048,5632]
weight model_norm_weight f32 model.norm.weight tensor [2048]
weight lm_head_weight_q q4b128 lm_head.weight tensor [32000,2048]
kernel s.rmsnorm in=2 out=1 threads=16384 tg=256 tgbytes=1024 expect=in0:131072,in1:2048,out0:131072 tag=aten::rms_norm
rtou 0 0
loadconstf32 4 0
loadconstf32 7 2048
utgid 1
ulid 2
loadlid 1
umulk 3 1 2048
uaddk 4 2 0
loadconstf32 8 8
forbegin 8
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
mul 2 2 2
add 4 4 2
ifend 
uaddk 4 4 256
forend 
tgstoref32 4 1 0
barriertg 
loadconstf32 8 128
cmplt 10 1 8
ifbegin 10
addconst 9 1 128
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 64
cmplt 10 1 8
ifbegin 10
addconst 9 1 64
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 32
cmplt 10 1 8
ifbegin 10
addconst 9 1 32
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 16
cmplt 10 1 8
ifbegin 10
addconst 9 1 16
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 8
cmplt 10 1 8
ifbegin 10
addconst 9 1 8
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 4
cmplt 10 1 8
ifbegin 10
addconst 9 1 4
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 2
cmplt 10 1 8
ifbegin 10
addconst 9 1 2
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 1
cmplt 10 1 8
ifbegin 10
addconst 9 1 1
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 9 0
tgloadf32 8 9 0
mulconst 8 8 0.00048828125
addconst 8 8 0.00000999999975
rsqrt 8 8
uaddk 4 2 0
loadconstf32 10 8
forbegin 10
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
mul 2 2 8
uloadf32 3 4 @in1
mul 2 2 3
ustoref32 2 5 @out0
ifend 
uaddk 4 4 256
forend 
endkernel
kernel s.matmul_q4 in=2 out=1 threads=2048 tg=64 tgbytes=16 expect=in0:131072,in1:589824,out0:131072 tag=aten::matmul
rtou 0 0
uconst 1 @in1
umulk 1 1 4
umulk 2 0 1024
uadd 2 2 1
umulk 3 0 16
uconst 4 0
uaddk 5 0 0
loadconstf32 6 64
forbegin 6
uaddk 6 2 0
uaddk 7 3 0
loadconstf32 1 0
loadconstf32 7 16
forbegin 7
uloadf32 2 7 @in1+524288
uloadf32 3 7 @in1+557056
loadconstf32 8 64
forbegin 8
uloadu8 8 6 0
uandk 9 8 15
utor 5 9
sub 5 5 3
mul 5 5 2
uloadf32 4 4 @in0
mul 4 4 5
add 1 1 4
ushrk 9 8 4
utor 5 9
sub 5 5 3
mul 5 5 2
uloadf32 4 4 @in0+1
mul 4 4 5
add 1 1 4
uaddk 4 4 2
uaddk 6 6 1
forend 
uaddk 7 7 1
forend 
ustoref32 1 5 @out0
uaddk 5 5 2048
forend 
endkernel
kernel s.matmul_q4_2 in=2 out=1 threads=256 tg=64 tgbytes=16 expect=in0:131072,in1:73728,out0:16384 tag=aten::matmul
rtou 0 0
uconst 1 @in1
umulk 1 1 4
umulk 2 0 1024
uadd 2 2 1
umulk 3 0 16
uconst 4 0
uaddk 5 0 0
loadconstf32 6 64
forbegin 6
uaddk 6 2 0
uaddk 7 3 0
loadconstf32 1 0
loadconstf32 7 16
forbegin 7
uloadf32 2 7 @in1+65536
uloadf32 3 7 @in1+69632
loadconstf32 8 64
forbegin 8
uloadu8 8 6 0
uandk 9 8 15
utor 5 9
sub 5 5 3
mul 5 5 2
uloadf32 4 4 @in0
mul 4 4 5
add 1 1 4
ushrk 9 8 4
utor 5 9
sub 5 5 3
mul 5 5 2
uloadf32 4 4 @in0+1
mul 4 4 5
add 1 1 4
uaddk 4 4 2
uaddk 6 6 1
forend 
uaddk 7 7 1
forend 
ustoref32 1 5 @out0
uaddk 5 5 256
forend 
endkernel
kernel s.rope in=2 out=1 threads=65536 tg=64 tgbytes=16 expect=in0:131072,in1:1,out0:131072
rtou 0 0
uconst 1 0
uandk 2 0 31
ushrk 3 0 5
umulk 5 3 64
uadd 5 5 2
uloadf32 1 1 @in1
utor 10 3
addconst 10 10 0.5
mulconst 10 10 0.03125
rtou 4 10
utor 9 4
add 1 1 9
utor 9 2
mulconst 9 9 -0.28782313662426
exp 2 9
mul 3 1 2
mulconst 11 3 0.1591549430919
addconst 11 11 0.5
rtou 6 11
utor 11 6
mulconst 13 11 6.28125
sub 12 3 13
mulconst 13 11 0.0019353071795864769
sub 12 12 13
mul 14 12 12
mulconst 15 12 1
mulconst 4 12 1
loadconstf32 16 1
loadconstf32 5 1
mul 15 15 14
mulconst 15 15 -0.16666666666667
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.5
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.05
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.08333333333333
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.02380952380952
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.03333333333333
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.01388888888889
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.01785714285714
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00909090909091
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.01111111111111
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00641025641026
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00757575757576
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.0047619047619
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00549450549451
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00367647058824
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00416666666667
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00292397660819
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00326797385621
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00238095238095
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00263157894737
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00197628458498
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.0021645021645
add 5 5 16
uloadf32 6 5 @in0
uloadf32 7 5 @in0+32
mul 8 6 5
mul 3 7 4
sub 8 8 3
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
ustoref32 8 5 @out0+32
endkernel
kernel s.rope_2 in=2 out=1 threads=8192 tg=64 tgbytes=16 expect=in0:16384,in1:1,out0:16384
rtou 0 0
uconst 1 0
uandk 2 0 31
ushrk 3 0 5
umulk 5 3 64
uadd 5 5 2
uloadf32 1 1 @in1
utor 10 3
addconst 10 10 0.5
mulconst 10 10 0.25
rtou 4 10
utor 9 4
add 1 1 9
utor 9 2
mulconst 9 9 -0.28782313662426
exp 2 9
mul 3 1 2
mulconst 11 3 0.1591549430919
addconst 11 11 0.5
rtou 6 11
utor 11 6
mulconst 13 11 6.28125
sub 12 3 13
mulconst 13 11 0.0019353071795864769
sub 12 12 13
mul 14 12 12
mulconst 15 12 1
mulconst 4 12 1
loadconstf32 16 1
loadconstf32 5 1
mul 15 15 14
mulconst 15 15 -0.16666666666667
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.5
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.05
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.08333333333333
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.02380952380952
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.03333333333333
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.01388888888889
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.01785714285714
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00909090909091
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.01111111111111
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00641025641026
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00757575757576
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.0047619047619
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00549450549451
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00367647058824
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00416666666667
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00292397660819
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00326797385621
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00238095238095
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.00263157894737
add 5 5 16
mul 15 15 14
mulconst 15 15 -0.00197628458498
add 4 4 15
mul 16 16 14
mulconst 16 16 -0.0021645021645
add 5 5 16
uloadf32 6 5 @in0
uloadf32 7 5 @in0+32
mul 8 6 5
mul 3 7 4
sub 8 8 3
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
ustoref32 8 5 @out0+32
endkernel
kernel s.kv_scatter in=3 out=1 threads=16384 tg=64 tgbytes=16 expect=in0:16384,in1:1,in2:524288,out0:524288
rtou 0 0
uconst 1 0
uloadf32 1 1 @in1
rtou 2 1
utor 3 0
addconst 3 3 0.5
mulconst 3 3 0.015625
rtou 3 3
utor 4 0
utor 5 3
mulconst 5 5 64
sub 4 4 5
rtou 6 4
utor 6 3
addconst 6 6 0.5
mulconst 6 6 0.25
rtou 4 6
utor 7 3
utor 8 4
mulconst 8 8 4
sub 7 7 8
rtou 5 7
uadd 7 2 4
umulk 5 5 2048
uadd 7 7 5
umulk 7 7 64
uadd 7 7 6
uloadf32 2 0 @in0
ustoref32 2 7 @in2
endkernel
kernel s.flash_prefill in=4 out=1 threads=262144 tg=128 tgbytes=8704 expect=in0:131072,in1:524288,in2:524288,in3:1,out0:131072 tag=aten::scaled_dot_product_attention
rtou 0 0
loadconstf32 13 0
loadconstf32 14 1
loadconstf32 15 -300000000000000012135895401846682943488.0
uconst 11 0
utgid 1
ulid 3
loadlid 1
uloadf32 2 11 @in3
addconst 2 2 1
ushrk 10 1 5
utor 6 10
add 2 2 6
uandk 2 1 31
ushrk 2 2 3
umulk 4 1 64
umulk 5 2 131072
rtou 9 2
uaddk 9 9 127
ushrk 9 9 7
utor 10 9
mulconst 4 15 1
loadconstf32 5 0
uaddk 6 3 0
loadconstf32 3 0
loadconstf32 16 16
forbegin 16
utor 3 6
cmplt 9 3 2
ifbegin 9
umulk 7 6 64
uadd 7 7 5
uaddk 8 4 0
loadconstf32 6 0
loadconstf32 17 64
forbegin 17
uloadf32 7 8 @in0
uloadf32 8 7 @in1
mul 7 7 8
add 6 6 7
uaddk 8 8 1
uaddk 7 7 1
forend 
mulconst 6 6 0.125
tgstoref32 6 3 0
cmplt 9 4 6
ifbegin 9
mulconst 4 6 1
ifend 
ifend 
uaddk 6 6 128
forend 
tgstoref32 4 1 2048
barriertg 
loadconstf32 16 64
cmplt 18 1 16
ifbegin 18
addconst 17 1 64
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 32
cmplt 18 1 16
ifbegin 18
addconst 17 1 32
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 16
cmplt 18 1 16
ifbegin 18
addconst 17 1 16
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 8
cmplt 18 1 16
ifbegin 18
addconst 17 1 8
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 4
cmplt 18 1 16
ifbegin 18
addconst 17 1 4
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 2
cmplt 18 1 16
ifbegin 18
addconst 17 1 2
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 1
cmplt 18 1 16
ifbegin 18
addconst 17 1 1
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
tgloadf32 11 13 2048
barriertg 
uaddk 6 3 0
loadconstf32 16 16
forbegin 16
utor 3 6
cmplt 9 3 2
ifbegin 9
tgloadf32 6 3 0
sub 6 6 11
exp 6 6
tgstoref32 6 3 0
add 5 5 6
ifend 
uaddk 6 6 128
forend 
tgstoref32 5 1 2048
barriertg 
loadconstf32 16 64
cmplt 18 1 16
ifbegin 18
addconst 17 1 64
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 32
cmplt 18 1 16
ifbegin 18
addconst 17 1 32
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 16
cmplt 18 1 16
ifbegin 18
addconst 17 1 16
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 8
cmplt 18 1 16
ifbegin 18
addconst 17 1 8
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 4
cmplt 18 1 16
ifbegin 18
addconst 17 1 4
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 2
cmplt 18 1 16
ifbegin 18
addconst 17 1 2
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 1
cmplt 18 1 16
ifbegin 18
addconst 17 1 1
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
tgloadf32 12 13 2048
barriertg 
loadconstf32 16 64
cmplt 9 1 16
ifbegin 9
loadconstf32 6 0
loadconstf32 3 0
uaddk 7 5 0
uadd 7 7 3
forbegin 2
tgloadf32 7 3 0
uloadf32 8 7 @in2
fma 6 7 8 6
addconst 3 3 1
uaddk 7 7 64
forend 
div 6 6 12
uadd 10 4 3
ustoref32 6 10 @out0
ifend 
endkernel
kernel s.vec_add in=2 out=1 threads=131072 tg=64 tgbytes=16 expect=in0:131072,in1:131072,out0:131072 tag=aten::add
rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
add 1 1 2
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_3 in=2 out=1 threads=5632 tg=64 tgbytes=16 expect=in0:131072,in1:1622016,out0:360448 tag=aten::matmul
rtou 0 0
uconst 1 @in1
umulk 1 1 4
umulk 2 0 1024
uadd 2 2 1
umulk 3 0 16
uconst 4 0
uaddk 5 0 0
loadconstf32 6 64
forbegin 6
uaddk 6 2 0
uaddk 7 3 0
loadconstf32 1 0
loadconstf32 7 16
forbegin 7
uloadf32 2 7 @in1+1441792
uloadf32 3 7 @in1+1531904
loadconstf32 8 64
forbegin 8
uloadu8 8 6 0
uandk 9 8 15
utor 5 9
sub 5 5 3
mul 5 5 2
uloadf32 4 4 @in0
mul 4 4 5
add 1 1 4
ushrk 9 8 4
utor 5 9
sub 5 5 3
mul 5 5 2
uloadf32 4 4 @in0+1
mul 4 4 5
add 1 1 4
uaddk 4 4 2
uaddk 6 6 1
forend 
uaddk 7 7 1
forend 
ustoref32 1 5 @out0
uaddk 5 5 5632
forend 
endkernel
kernel s.act_sigmoid in=1 out=1 threads=360448 tg=64 tgbytes=16 expect=in0:360448,out0:360448 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 1 1
ustoref32 1 0 @out0
endkernel
kernel s.vec_mul in=2 out=1 threads=360448 tg=64 tgbytes=16 expect=in0:360448,in1:360448,out0:360448 tag=aten::mul
rtou 0 0
uloadf32 1 0 @in0
uloadf32 2 0 @in1
mul 1 1 2
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_4 in=2 out=1 threads=2048 tg=64 tgbytes=16 expect=in0:360448,in1:1622016,out0:131072 tag=aten::matmul
rtou 0 0
uconst 1 @in1
umulk 1 1 4
umulk 2 0 2816
uadd 2 2 1
umulk 3 0 44
uconst 4 0
uaddk 5 0 0
loadconstf32 6 64
forbegin 6
uaddk 6 2 0
uaddk 7 3 0
loadconstf32 1 0
loadconstf32 7 44
forbegin 7
uloadf32 2 7 @in1+1441792
uloadf32 3 7 @in1+1531904
loadconstf32 8 64
forbegin 8
uloadu8 8 6 0
uandk 9 8 15
utor 5 9
sub 5 5 3
mul 5 5 2
uloadf32 4 4 @in0
mul 4 4 5
add 1 1 4
ushrk 9 8 4
utor 5 9
sub 5 5 3
mul 5 5 2
uloadf32 4 4 @in0+1
mul 4 4 5
add 1 1 4
uaddk 4 4 2
uaddk 6 6 1
forend 
uaddk 7 7 1
forend 
ustoref32 1 5 @out0
uaddk 5 5 2048
forend 
endkernel
kernel s.silu_mul in=2 out=1 threads=360448 tg=64 tgbytes=16 expect=in0:360448,in1:360448,out0:360448 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 2 1
mul 1 1 2
uloadf32 3 0 @in1
mul 1 1 3
ustoref32 1 0 @out0
endkernel
kernel s.rmsnorm_2 in=2 out=1 threads=256 tg=256 tgbytes=1024 expect=in0:2048,in1:2048,out0:2048 tag=aten::rms_norm
rtou 0 0
loadconstf32 4 0
loadconstf32 7 2048
utgid 1
ulid 2
loadlid 1
umulk 3 1 2048
uaddk 4 2 0
loadconstf32 8 8
forbegin 8
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
mul 2 2 2
add 4 4 2
ifend 
uaddk 4 4 256
forend 
tgstoref32 4 1 0
barriertg 
loadconstf32 8 128
cmplt 10 1 8
ifbegin 10
addconst 9 1 128
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 64
cmplt 10 1 8
ifbegin 10
addconst 9 1 64
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 32
cmplt 10 1 8
ifbegin 10
addconst 9 1 32
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 16
cmplt 10 1 8
ifbegin 10
addconst 9 1 16
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 8
cmplt 10 1 8
ifbegin 10
addconst 9 1 8
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 4
cmplt 10 1 8
ifbegin 10
addconst 9 1 4
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 2
cmplt 10 1 8
ifbegin 10
addconst 9 1 2
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 1
cmplt 10 1 8
ifbegin 10
addconst 9 1 1
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 9 0
tgloadf32 8 9 0
mulconst 8 8 0.00048828125
addconst 8 8 0.00000999999975
rsqrt 8 8
uaddk 4 2 0
loadconstf32 10 8
forbegin 10
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
mul 2 2 8
uloadf32 3 4 @in1
mul 2 2 3
ustoref32 2 5 @out0
ifend 
uaddk 4 4 256
forend 
endkernel
kernel s.matmul_q4_5 in=2 out=1 threads=256000 tg=64 tgbytes=16 expect=in0:2048,in1:9216000,out0:32000 tag=aten::matmul
rtou 0 0
utgid 1
simdlane 2
simdw 4
simdgid 3
umulk 6 3 4
umulk 5 4 8
uconst 8 @in1
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 2048
loadconstf32 24 32000
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 16
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in0+0
uloadf32 6 7 @in0+1
uloadf32 7 7 @in0+2
uloadf32 8 7 @in0+3
uloadf32 9 7 @in0+4
uloadf32 10 7 @in0+5
uloadf32 11 7 @in0+6
uloadf32 12 7 @in0+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+8192000
uloadf32 19 12 @in1+8704000
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+8192000
uloadf32 19 12 @in1+8704000
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+8192000
uloadf32 19 12 @in1+8704000
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+8192000
uloadf32 19 12 @in1+8704000
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend 
uadd 7 7 5
forend 
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out0
ifend 
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out0
ifend 
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out0
ifend 
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out0
ifend 
ifend 
endkernel
kernel s.matmul_q4_6 in=2 out=1 threads=16384 tg=64 tgbytes=16 expect=in0:2048,in1:589824,out0:2048 tag=aten::matmul
rtou 0 0
utgid 1
simdlane 2
simdw 4
simdgid 3
umulk 6 3 4
umulk 5 4 8
uconst 8 @in1
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 2048
loadconstf32 24 2048
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 16
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in0+0
uloadf32 6 7 @in0+1
uloadf32 7 7 @in0+2
uloadf32 8 7 @in0+3
uloadf32 9 7 @in0+4
uloadf32 10 7 @in0+5
uloadf32 11 7 @in0+6
uloadf32 12 7 @in0+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+524288
uloadf32 19 12 @in1+557056
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend 
uadd 7 7 5
forend 
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out0
ifend 
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out0
ifend 
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out0
ifend 
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out0
ifend 
ifend 
endkernel
kernel s.matmul_q4_7 in=2 out=1 threads=2048 tg=64 tgbytes=16 expect=in0:2048,in1:73728,out0:256 tag=aten::matmul
rtou 0 0
utgid 1
simdlane 2
simdw 4
simdgid 3
umulk 6 3 4
umulk 5 4 8
uconst 8 @in1
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 2048
loadconstf32 24 256
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 16
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in0+0
uloadf32 6 7 @in0+1
uloadf32 7 7 @in0+2
uloadf32 8 7 @in0+3
uloadf32 9 7 @in0+4
uloadf32 10 7 @in0+5
uloadf32 11 7 @in0+6
uloadf32 12 7 @in0+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+65536
uloadf32 19 12 @in1+69632
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+65536
uloadf32 19 12 @in1+69632
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+65536
uloadf32 19 12 @in1+69632
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+65536
uloadf32 19 12 @in1+69632
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend 
uadd 7 7 5
forend 
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out0
ifend 
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out0
ifend 
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out0
ifend 
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out0
ifend 
ifend 
endkernel
kernel s.rope_3 in=2 out=1 threads=1024 tg=64 tgbytes=16 expect=in0:2048,in1:1,out0:2048
rtou 0 0
uconst 1 0
uandk 2 0 31
ushrk 3 0 5
umulk 5 3 64
uadd 5 5 2
uloadf32 1 1 @in1
utor 9 2
mulconst 9 9 -0.28782313662426
exp 2 9
mul 3 1 2
mulconst 10 3 0.1591549430919
addconst 10 10 0.5
rtou 6 10
utor 10 6
mulconst 12 10 6.28125
sub 11 3 12
mulconst 12 10 0.0019353071795864769
sub 11 11 12
mul 13 11 11
mulconst 14 11 1
mulconst 4 11 1
loadconstf32 15 1
loadconstf32 5 1
mul 14 14 13
mulconst 14 14 -0.16666666666667
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.5
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.05
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.08333333333333
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.02380952380952
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.03333333333333
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.01388888888889
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.01785714285714
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00909090909091
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.01111111111111
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00641025641026
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00757575757576
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.0047619047619
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00549450549451
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00367647058824
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00416666666667
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00292397660819
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00326797385621
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00238095238095
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00263157894737
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00197628458498
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.0021645021645
add 5 5 15
uloadf32 6 5 @in0
uloadf32 7 5 @in0+32
mul 8 6 5
mul 3 7 4
sub 8 8 3
mulconst 8 8 0.125
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
mulconst 8 8 0.125
ustoref32 8 5 @out0+32
endkernel
kernel s.rope_4 in=2 out=1 threads=128 tg=64 tgbytes=16 expect=in0:256,in1:1,out0:256
rtou 0 0
uconst 1 0
uandk 2 0 31
ushrk 3 0 5
umulk 5 3 64
uadd 5 5 2
uloadf32 1 1 @in1
utor 9 2
mulconst 9 9 -0.28782313662426
exp 2 9
mul 3 1 2
mulconst 10 3 0.1591549430919
addconst 10 10 0.5
rtou 6 10
utor 10 6
mulconst 12 10 6.28125
sub 11 3 12
mulconst 12 10 0.0019353071795864769
sub 11 11 12
mul 13 11 11
mulconst 14 11 1
mulconst 4 11 1
loadconstf32 15 1
loadconstf32 5 1
mul 14 14 13
mulconst 14 14 -0.16666666666667
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.5
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.05
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.08333333333333
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.02380952380952
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.03333333333333
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.01388888888889
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.01785714285714
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00909090909091
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.01111111111111
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00641025641026
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00757575757576
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.0047619047619
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00549450549451
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00367647058824
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00416666666667
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00292397660819
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00326797385621
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00238095238095
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.00263157894737
add 5 5 15
mul 14 14 13
mulconst 14 14 -0.00197628458498
add 4 4 14
mul 15 15 13
mulconst 15 15 -0.0021645021645
add 5 5 15
uloadf32 6 5 @in0
uloadf32 7 5 @in0+32
mul 8 6 5
mul 3 7 4
sub 8 8 3
ustoref32 8 5 @out0
mul 8 7 5
mul 3 6 4
add 8 8 3
ustoref32 8 5 @out0+32
endkernel
kernel s.kv_scatter_2 in=3 out=1 threads=256 tg=64 tgbytes=16 expect=in0:256,in1:1,in2:524288,out0:524288
rtou 0 0
uconst 1 0
uloadf32 1 1 @in1
rtou 2 1
utor 3 0
addconst 3 3 0.5
mulconst 3 3 0.015625
rtou 3 3
utor 4 0
utor 5 3
mulconst 5 5 64
sub 4 4 5
rtou 6 4
utor 6 3
addconst 6 6 0.5
mulconst 6 6 0.25
rtou 4 6
utor 7 3
utor 8 4
mulconst 8 8 4
sub 7 7 8
rtou 5 7
uadd 7 2 4
umulk 5 5 2048
uadd 7 7 5
umulk 7 7 64
uadd 7 7 6
uloadf32 2 0 @in0
ustoref32 2 7 @in2
endkernel
kernel s.attention in=4 out=1 threads=4096 tg=128 tgbytes=8704 expect=in0:2048,in1:524288,in2:524288,in3:1,out0:2048 tag=aten::scaled_dot_product_attention
rtou 0 0
loadconstf32 13 0
loadconstf32 14 1
loadconstf32 15 -300000000000000012135895401846682943488.0
uconst 11 0
utgid 1
ulid 3
loadlid 1
uloadf32 2 11 @in3
addconst 2 2 1
ushrk 2 1 3
umulk 4 1 64
umulk 5 2 131072
rtou 9 2
uaddk 9 9 127
ushrk 9 9 7
utor 10 9
mulconst 4 15 1
loadconstf32 5 0
uaddk 6 3 0
loadconstf32 3 0
loadconstf32 16 16
forbegin 16
utor 3 6
cmplt 9 3 2
ifbegin 9
umulk 7 6 64
uadd 7 7 5
uaddk 8 4 0
loadconstf32 6 0
loadconstf32 17 64
forbegin 17
uloadf32 7 8 @in0
uloadf32 8 7 @in1
mul 7 7 8
add 6 6 7
uaddk 8 8 1
uaddk 7 7 1
forend 
tgstoref32 6 3 0
cmplt 9 4 6
ifbegin 9
mulconst 4 6 1
ifend 
ifend 
uaddk 6 6 128
forend 
tgstoref32 4 1 2048
barriertg 
loadconstf32 16 64
cmplt 18 1 16
ifbegin 18
addconst 17 1 64
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 32
cmplt 18 1 16
ifbegin 18
addconst 17 1 32
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 16
cmplt 18 1 16
ifbegin 18
addconst 17 1 16
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 8
cmplt 18 1 16
ifbegin 18
addconst 17 1 8
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 4
cmplt 18 1 16
ifbegin 18
addconst 17 1 4
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 2
cmplt 18 1 16
ifbegin 18
addconst 17 1 2
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 1
cmplt 18 1 16
ifbegin 18
addconst 17 1 1
tgloadf32 19 17 2048
tgloadf32 20 1 2048
cmplt 18 20 19
ifbegin 18
mulconst 20 19 1
ifend 
tgstoref32 20 1 2048
ifend 
barriertg 
tgloadf32 11 13 2048
barriertg 
uaddk 6 3 0
loadconstf32 16 16
forbegin 16
utor 3 6
cmplt 9 3 2
ifbegin 9
tgloadf32 6 3 0
sub 6 6 11
exp 6 6
tgstoref32 6 3 0
add 5 5 6
ifend 
uaddk 6 6 128
forend 
tgstoref32 5 1 2048
barriertg 
loadconstf32 16 64
cmplt 18 1 16
ifbegin 18
addconst 17 1 64
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 32
cmplt 18 1 16
ifbegin 18
addconst 17 1 32
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 16
cmplt 18 1 16
ifbegin 18
addconst 17 1 16
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 8
cmplt 18 1 16
ifbegin 18
addconst 17 1 8
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 4
cmplt 18 1 16
ifbegin 18
addconst 17 1 4
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 2
cmplt 18 1 16
ifbegin 18
addconst 17 1 2
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
loadconstf32 16 1
cmplt 18 1 16
ifbegin 18
addconst 17 1 1
tgloadf32 19 17 2048
tgloadf32 20 1 2048
add 20 20 19
tgstoref32 20 1 2048
ifend 
barriertg 
tgloadf32 12 13 2048
barriertg 
loadconstf32 16 64
cmplt 9 1 16
ifbegin 9
loadconstf32 6 0
loadconstf32 3 0
uaddk 7 5 0
uadd 7 7 3
forbegin 2
tgloadf32 7 3 0
uloadf32 8 7 @in2
fma 6 7 8 6
addconst 3 3 1
uaddk 7 7 64
forend 
div 6 6 12
uadd 10 4 3
ustoref32 6 10 @out0
ifend 
endkernel
kernel s.add_rmsnorm in=3 out=2 threads=256 tg=256 tgbytes=1024 expect=in0:2048,in1:2048,in2:2048,out0:2048,out1:2048 tag=aten::add
rtou 0 0
loadconstf32 4 0
loadconstf32 7 2048
utgid 1
ulid 2
loadlid 1
umulk 3 1 2048
uaddk 4 2 0
loadconstf32 8 8
forbegin 8
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @in0
uloadf32 3 5 @in1
add 2 2 3
ustoref32 2 5 @out0
mul 2 2 2
add 4 4 2
ifend 
uaddk 4 4 256
forend 
tgstoref32 4 1 0
barriertg 
loadconstf32 8 128
cmplt 10 1 8
ifbegin 10
addconst 9 1 128
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 64
cmplt 10 1 8
ifbegin 10
addconst 9 1 64
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 32
cmplt 10 1 8
ifbegin 10
addconst 9 1 32
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 16
cmplt 10 1 8
ifbegin 10
addconst 9 1 16
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 8
cmplt 10 1 8
ifbegin 10
addconst 9 1 8
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 4
cmplt 10 1 8
ifbegin 10
addconst 9 1 4
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 2
cmplt 10 1 8
ifbegin 10
addconst 9 1 2
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 8 1
cmplt 10 1 8
ifbegin 10
addconst 9 1 1
tgloadf32 11 9 0
tgloadf32 12 1 0
add 12 12 11
tgstoref32 12 1 0
ifend 
barriertg 
loadconstf32 9 0
tgloadf32 8 9 0
mulconst 8 8 0.00048828125
addconst 8 8 0.00000999999975
rsqrt 8 8
uaddk 4 2 0
loadconstf32 10 8
forbegin 10
utor 6 4
cmplt 5 6 7
ifbegin 5
uadd 5 3 4
uloadf32 2 5 @out0
mul 2 2 8
uloadf32 3 4 @in2
mul 2 2 3
ustoref32 2 5 @out1
ifend 
uaddk 4 4 256
forend 
endkernel
kernel s.matmul_q4_8 in=2 out=1 threads=45056 tg=64 tgbytes=16 expect=in0:2048,in1:1622016,out0:5632 tag=aten::matmul
rtou 0 0
utgid 1
simdlane 2
simdw 4
simdgid 3
umulk 6 3 4
umulk 5 4 8
uconst 8 @in1
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 2048
loadconstf32 24 5632
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 16
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in0+0
uloadf32 6 7 @in0+1
uloadf32 7 7 @in0+2
uloadf32 8 7 @in0+3
uloadf32 9 7 @in0+4
uloadf32 10 7 @in0+5
uloadf32 11 7 @in0+6
uloadf32 12 7 @in0+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 1024
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 16
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend 
uadd 7 7 5
forend 
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out0
ifend 
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out0
ifend 
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out0
ifend 
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out0
ifend 
ifend 
endkernel
kernel s.silu_mul_2 in=2 out=1 threads=5632 tg=64 tgbytes=16 expect=in0:5632,in1:5632,out0:5632 tag=aten::sigmoid
rtou 0 0
uloadf32 1 0 @in0
sigmoid 2 1
mul 1 1 2
uloadf32 3 0 @in1
mul 1 1 3
ustoref32 1 0 @out0
endkernel
kernel s.matmul_q4_9 in=2 out=1 threads=16384 tg=64 tgbytes=16 expect=in0:5632,in1:1622016,out0:2048 tag=aten::matmul
rtou 0 0
utgid 1
simdlane 2
simdw 4
simdgid 3
umulk 6 3 4
umulk 5 4 8
uconst 8 @in1
umulk 8 8 4
umulk 7 2 8
loadconstf32 1 0
loadconstf32 2 0
loadconstf32 3 0
loadconstf32 4 0
loadconstf32 22 5632
loadconstf32 24 2048
loadconstf32 25 1
utor 23 6
cmplt 20 23 24
ifbegin 20
loadconstf32 26 64
forbegin 26
utor 21 7
cmplt 20 21 22
ifbegin 20
uloadf32 5 7 @in0+0
uloadf32 6 7 @in0+1
uloadf32 7 7 @in0+2
uloadf32 8 7 @in0+3
uloadf32 9 7 @in0+4
uloadf32 10 7 @in0+5
uloadf32 11 7 @in0+6
uloadf32 12 7 @in0+7
add 14 5 6
add 15 7 8
add 14 14 15
add 15 9 10
add 16 11 12
add 15 15 16
add 13 14 15
uaddk 16 7 0
ushrk 15 16 7
ushrk 14 16 1
uaddk 13 6 0
umulk 9 13 2816
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 44
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 1 18 16 1
uaddk 13 6 1
umulk 9 13 2816
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 44
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 2 18 16 2
uaddk 13 6 2
umulk 9 13 2816
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 44
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 3 18 16 3
uaddk 13 6 3
umulk 9 13 2816
uadd 9 9 8
uadd 9 9 14
uloadu32 10 9 0
ushrk 11 10 28
utor 17 11
mul 16 12 17
ushrk 11 10 24
uandk 11 11 15
utor 17 11
fma 16 11 17 16
ushrk 11 10 20
uandk 11 11 15
utor 17 11
fma 16 10 17 16
ushrk 11 10 16
uandk 11 11 15
utor 17 11
fma 16 9 17 16
ushrk 11 10 12
uandk 11 11 15
utor 17 11
fma 16 8 17 16
ushrk 11 10 8
uandk 11 11 15
utor 17 11
fma 16 7 17 16
ushrk 11 10 4
uandk 11 11 15
utor 17 11
fma 16 6 17 16
ushrk 11 10 0
uandk 11 11 15
utor 17 11
fma 16 5 17 16
umulk 12 13 44
uadd 12 12 15
uloadf32 18 12 @in1+1441792
uloadf32 19 12 @in1+1531904
mul 19 19 13
sub 16 16 19
fma 4 18 16 4
ifend 
uadd 7 7 5
forend 
utor 21 2
cmplt 20 21 25
simdsum 16 1
ifbegin 20
uaddk 13 6 0
ustoref32 16 13 @out0
ifend 
simdsum 16 2
ifbegin 20
uaddk 13 6 1
ustoref32 16 13 @out0
ifend 
simdsum 16 3
ifbegin 20
uaddk 13 6 2
ustoref32 16 13 @out0
ifend 
simdsum 16 4
ifbegin 20
uaddk 13 6 3
ustoref32 16 13 @out0
ifend 
ifend 
endkernel
phase chunk1 rows=64
node embedding_p1 ext fir.embedding token model_embed_tokens_weight_emb -> t718_embedding
node rmsnorm_0 micro s.rmsnorm t718_embedding model_layers_0_input_layernorm_weight -> t722_y
node matmul_q4_0 micro s.matmul_q4 t722_y model_layers_0_self_attn_q_proj_weight_q -> t724_c
node matmul_q4_1 micro s.matmul_q4_2 t722_y model_layers_0_self_attn_k_proj_weight_q -> t726_c
node matmul_q4_2 micro s.matmul_q4_2 t722_y model_layers_0_self_attn_v_proj_weight_q -> t728_c
node rope_0 micro s.rope t724_c position -> t724_c
node rope_1 micro s.rope_2 t726_c position -> t726_c
node kv_scatter_0 micro s.kv_scatter t726_c position s510 -> s510
node kv_scatter_1 micro s.kv_scatter t728_c position s511 -> s511
node flash_prefill_0 micro s.flash_prefill t724_c s510 s511 position -> t734_o
node matmul_q4_3 micro s.matmul_q4 t734_o model_layers_0_self_attn_o_proj_weight_q -> t736_c
node vec_add_0 micro s.vec_add t718_embedding t736_c -> t738_c
node rmsnorm_1 micro s.rmsnorm t738_c model_layers_0_post_attention_layernorm_weight -> t740_y
node matmul_q4_4 micro s.matmul_q4_3 t740_y model_layers_0_mlp_gate_proj_weight_q -> t742_c
node matmul_q4_5 micro s.matmul_q4_3 t740_y model_layers_0_mlp_up_proj_weight_q -> t744_c
node act_sigmoid_0 micro s.act_sigmoid t742_c -> t746_y
node vec_mul_0 micro s.vec_mul t742_c t746_y -> t748_c
node vec_mul_1 micro s.vec_mul t748_c t744_c -> t750_c
node matmul_q4_6 micro s.matmul_q4_4 t750_c model_layers_0_mlp_down_proj_weight_q -> t752_c
node vec_add_1 micro s.vec_add t738_c t752_c -> t754_c
node rmsnorm_2 micro s.rmsnorm t754_c model_layers_1_input_layernorm_weight -> t773_y
node matmul_q4_7 micro s.matmul_q4 t773_y model_layers_1_self_attn_q_proj_weight_q -> t775_c
node matmul_q4_8 micro s.matmul_q4_2 t773_y model_layers_1_self_attn_k_proj_weight_q -> t777_c
node matmul_q4_9 micro s.matmul_q4_2 t773_y model_layers_1_self_attn_v_proj_weight_q -> t779_c
node rope_2 micro s.rope t775_c position -> t775_c
node rope_3 micro s.rope_2 t777_c position -> t777_c
node kv_scatter_2 micro s.kv_scatter t777_c position s512 -> s512
node kv_scatter_3 micro s.kv_scatter t779_c position s513 -> s513
node flash_prefill_1 micro s.flash_prefill t775_c s512 s513 position -> t785_o
node matmul_q4_10 micro s.matmul_q4 t785_o model_layers_1_self_attn_o_proj_weight_q -> t787_c
node vec_add_2 micro s.vec_add t754_c t787_c -> t789_c
node rmsnorm_3 micro s.rmsnorm t789_c model_layers_1_post_attention_layernorm_weight -> t791_y
node matmul_q4_11 micro s.matmul_q4_3 t791_y model_layers_1_mlp_gate_proj_weight_q -> t793_c
node matmul_q4_12 micro s.matmul_q4_3 t791_y model_layers_1_mlp_up_proj_weight_q -> t795_c
node act_sigmoid_1 micro s.act_sigmoid t793_c -> t797_y
node vec_mul_2 micro s.vec_mul t793_c t797_y -> t799_c
node vec_mul_3 micro s.vec_mul t799_c t795_c -> t801_c
node matmul_q4_13 micro s.matmul_q4_4 t801_c model_layers_1_mlp_down_proj_weight_q -> t803_c
node vec_add_3 micro s.vec_add t789_c t803_c -> t805_c
node rmsnorm_4 micro s.rmsnorm t805_c model_layers_2_input_layernorm_weight -> t807_y
node matmul_q4_14 micro s.matmul_q4 t807_y model_layers_2_self_attn_q_proj_weight_q -> t809_c
node matmul_q4_15 micro s.matmul_q4_2 t807_y model_layers_2_self_attn_k_proj_weight_q -> t811_c
node matmul_q4_16 micro s.matmul_q4_2 t807_y model_layers_2_self_attn_v_proj_weight_q -> t813_c
node rope_4 micro s.rope t809_c position -> t809_c
node rope_5 micro s.rope_2 t811_c position -> t811_c
node kv_scatter_4 micro s.kv_scatter t811_c position s514 -> s514
node kv_scatter_5 micro s.kv_scatter t813_c position s515 -> s515
node flash_prefill_2 micro s.flash_prefill t809_c s514 s515 position -> t819_o
node matmul_q4_17 micro s.matmul_q4 t819_o model_layers_2_self_attn_o_proj_weight_q -> t821_c
node vec_add_4 micro s.vec_add t805_c t821_c -> t823_c
node rmsnorm_5 micro s.rmsnorm t823_c model_layers_2_post_attention_layernorm_weight -> t825_y
node matmul_q4_18 micro s.matmul_q4_3 t825_y model_layers_2_mlp_gate_proj_weight_q -> t827_c
node matmul_q4_19 micro s.matmul_q4_3 t825_y model_layers_2_mlp_up_proj_weight_q -> t829_c
node act_sigmoid_2 micro s.act_sigmoid t827_c -> t831_y
node vec_mul_4 micro s.vec_mul t827_c t831_y -> t833_c
node vec_mul_5 micro s.vec_mul t833_c t829_c -> t835_c
node matmul_q4_20 micro s.matmul_q4_4 t835_c model_layers_2_mlp_down_proj_weight_q -> t837_c
node vec_add_5 micro s.vec_add t823_c t837_c -> t839_c
node rmsnorm_6 micro s.rmsnorm t839_c model_layers_3_input_layernorm_weight -> t841_y
node matmul_q4_21 micro s.matmul_q4 t841_y model_layers_3_self_attn_q_proj_weight_q -> t843_c
node matmul_q4_22 micro s.matmul_q4_2 t841_y model_layers_3_self_attn_k_proj_weight_q -> t845_c
node matmul_q4_23 micro s.matmul_q4_2 t841_y model_layers_3_self_attn_v_proj_weight_q -> t847_c
node rope_6 micro s.rope t843_c position -> t843_c
node rope_7 micro s.rope_2 t845_c position -> t845_c
node kv_scatter_6 micro s.kv_scatter t845_c position s516 -> s516
node kv_scatter_7 micro s.kv_scatter t847_c position s517 -> s517
node flash_prefill_3 micro s.flash_prefill t843_c s516 s517 position -> t853_o
node matmul_q4_24 micro s.matmul_q4 t853_o model_layers_3_self_attn_o_proj_weight_q -> t855_c
node vec_add_6 micro s.vec_add t839_c t855_c -> t857_c
node rmsnorm_7 micro s.rmsnorm t857_c model_layers_3_post_attention_layernorm_weight -> t859_y
node matmul_q4_25 micro s.matmul_q4_3 t859_y model_layers_3_mlp_gate_proj_weight_q -> t861_c
node matmul_q4_26 micro s.matmul_q4_3 t859_y model_layers_3_mlp_up_proj_weight_q -> t863_c
node act_sigmoid_3 micro s.act_sigmoid t861_c -> t865_y
node vec_mul_6 micro s.vec_mul t861_c t865_y -> t867_c
node vec_mul_7 micro s.vec_mul t867_c t863_c -> t869_c
node matmul_q4_27 micro s.matmul_q4_4 t869_c model_layers_3_mlp_down_proj_weight_q -> t871_c
node vec_add_7 micro s.vec_add t857_c t871_c -> t873_c
node rmsnorm_8 micro s.rmsnorm t873_c model_layers_4_input_layernorm_weight -> t875_y
node matmul_q4_28 micro s.matmul_q4 t875_y model_layers_4_self_attn_q_proj_weight_q -> t877_c
node matmul_q4_29 micro s.matmul_q4_2 t875_y model_layers_4_self_attn_k_proj_weight_q -> t879_c
node matmul_q4_30 micro s.matmul_q4_2 t875_y model_layers_4_self_attn_v_proj_weight_q -> t881_c
node rope_8 micro s.rope t877_c position -> t877_c
node rope_9 micro s.rope_2 t879_c position -> t879_c
node kv_scatter_8 micro s.kv_scatter t879_c position s518 -> s518
node kv_scatter_9 micro s.kv_scatter t881_c position s519 -> s519
node flash_prefill_4 micro s.flash_prefill t877_c s518 s519 position -> t887_o
node matmul_q4_31 micro s.matmul_q4 t887_o model_layers_4_self_attn_o_proj_weight_q -> t889_c
node vec_add_8 micro s.vec_add t873_c t889_c -> t891_c
node rmsnorm_9 micro s.rmsnorm t891_c model_layers_4_post_attention_layernorm_weight -> t893_y
node matmul_q4_32 micro s.matmul_q4_3 t893_y model_layers_4_mlp_gate_proj_weight_q -> t895_c
node matmul_q4_33 micro s.matmul_q4_3 t893_y model_layers_4_mlp_up_proj_weight_q -> t897_c
node act_sigmoid_4 micro s.act_sigmoid t895_c -> t899_y
node vec_mul_8 micro s.vec_mul t895_c t899_y -> t901_c
node vec_mul_9 micro s.vec_mul t901_c t897_c -> t903_c
node matmul_q4_34 micro s.matmul_q4_4 t903_c model_layers_4_mlp_down_proj_weight_q -> t905_c
node vec_add_9 micro s.vec_add t891_c t905_c -> t907_c
node rmsnorm_10 micro s.rmsnorm t907_c model_layers_5_input_layernorm_weight -> t909_y
node matmul_q4_35 micro s.matmul_q4 t909_y model_layers_5_self_attn_q_proj_weight_q -> t911_c
node matmul_q4_36 micro s.matmul_q4_2 t909_y model_layers_5_self_attn_k_proj_weight_q -> t913_c
node matmul_q4_37 micro s.matmul_q4_2 t909_y model_layers_5_self_attn_v_proj_weight_q -> t915_c
node rope_10 micro s.rope t911_c position -> t911_c
node rope_11 micro s.rope_2 t913_c position -> t913_c
node kv_scatter_10 micro s.kv_scatter t913_c position s520 -> s520
node kv_scatter_11 micro s.kv_scatter t915_c position s521 -> s521
node flash_prefill_5 micro s.flash_prefill t911_c s520 s521 position -> t921_o
node matmul_q4_38 micro s.matmul_q4 t921_o model_layers_5_self_attn_o_proj_weight_q -> t923_c
node vec_add_10 micro s.vec_add t907_c t923_c -> t925_c
node rmsnorm_11 micro s.rmsnorm t925_c model_layers_5_post_attention_layernorm_weight -> t927_y
node matmul_q4_39 micro s.matmul_q4_3 t927_y model_layers_5_mlp_gate_proj_weight_q -> t929_c
node matmul_q4_40 micro s.matmul_q4_3 t927_y model_layers_5_mlp_up_proj_weight_q -> t931_c
node act_sigmoid_5 micro s.act_sigmoid t929_c -> t933_y
node vec_mul_10 micro s.vec_mul t929_c t933_y -> t935_c
node vec_mul_11 micro s.vec_mul t935_c t931_c -> t937_c
node matmul_q4_41 micro s.matmul_q4_4 t937_c model_layers_5_mlp_down_proj_weight_q -> t939_c
node vec_add_11 micro s.vec_add t925_c t939_c -> t941_c
node rmsnorm_12 micro s.rmsnorm t941_c model_layers_6_input_layernorm_weight -> t943_y
node matmul_q4_42 micro s.matmul_q4 t943_y model_layers_6_self_attn_q_proj_weight_q -> t945_c
node matmul_q4_43 micro s.matmul_q4_2 t943_y model_layers_6_self_attn_k_proj_weight_q -> t947_c
node matmul_q4_44 micro s.matmul_q4_2 t943_y model_layers_6_self_attn_v_proj_weight_q -> t949_c
node rope_12 micro s.rope t945_c position -> t945_c
node rope_13 micro s.rope_2 t947_c position -> t947_c
node kv_scatter_12 micro s.kv_scatter t947_c position s522 -> s522
node kv_scatter_13 micro s.kv_scatter t949_c position s523 -> s523
node flash_prefill_6 micro s.flash_prefill t945_c s522 s523 position -> t955_o
node matmul_q4_45 micro s.matmul_q4 t955_o model_layers_6_self_attn_o_proj_weight_q -> t957_c
node vec_add_12 micro s.vec_add t941_c t957_c -> t959_c
node rmsnorm_13 micro s.rmsnorm t959_c model_layers_6_post_attention_layernorm_weight -> t961_y
node matmul_q4_46 micro s.matmul_q4_3 t961_y model_layers_6_mlp_gate_proj_weight_q -> t963_c
node matmul_q4_47 micro s.matmul_q4_3 t961_y model_layers_6_mlp_up_proj_weight_q -> t965_c
node act_sigmoid_6 micro s.act_sigmoid t963_c -> t967_y
node vec_mul_12 micro s.vec_mul t963_c t967_y -> t969_c
node vec_mul_13 micro s.vec_mul t969_c t965_c -> t971_c
node matmul_q4_48 micro s.matmul_q4_4 t971_c model_layers_6_mlp_down_proj_weight_q -> t973_c
node vec_add_13 micro s.vec_add t959_c t973_c -> t975_c
node rmsnorm_14 micro s.rmsnorm t975_c model_layers_7_input_layernorm_weight -> t977_y
node matmul_q4_49 micro s.matmul_q4 t977_y model_layers_7_self_attn_q_proj_weight_q -> t979_c
node matmul_q4_50 micro s.matmul_q4_2 t977_y model_layers_7_self_attn_k_proj_weight_q -> t981_c
node matmul_q4_51 micro s.matmul_q4_2 t977_y model_layers_7_self_attn_v_proj_weight_q -> t983_c
node rope_14 micro s.rope t979_c position -> t979_c
node rope_15 micro s.rope_2 t981_c position -> t981_c
node kv_scatter_14 micro s.kv_scatter t981_c position s524 -> s524
node kv_scatter_15 micro s.kv_scatter t983_c position s525 -> s525
node flash_prefill_7 micro s.flash_prefill t979_c s524 s525 position -> t989_o
node matmul_q4_52 micro s.matmul_q4 t989_o model_layers_7_self_attn_o_proj_weight_q -> t991_c
node vec_add_14 micro s.vec_add t975_c t991_c -> t993_c
node rmsnorm_15 micro s.rmsnorm t993_c model_layers_7_post_attention_layernorm_weight -> t995_y
node matmul_q4_53 micro s.matmul_q4_3 t995_y model_layers_7_mlp_gate_proj_weight_q -> t997_c
node matmul_q4_54 micro s.matmul_q4_3 t995_y model_layers_7_mlp_up_proj_weight_q -> t999_c
node act_sigmoid_7 micro s.act_sigmoid t997_c -> t1001_y
node vec_mul_14 micro s.vec_mul t997_c t1001_y -> t1003_c
node vec_mul_15 micro s.vec_mul t1003_c t999_c -> t1005_c
node matmul_q4_55 micro s.matmul_q4_4 t1005_c model_layers_7_mlp_down_proj_weight_q -> t1007_c
node vec_add_15 micro s.vec_add t993_c t1007_c -> t1009_c
node rmsnorm_16 micro s.rmsnorm t1009_c model_layers_8_input_layernorm_weight -> t1011_y
node matmul_q4_56 micro s.matmul_q4 t1011_y model_layers_8_self_attn_q_proj_weight_q -> t1013_c
node matmul_q4_57 micro s.matmul_q4_2 t1011_y model_layers_8_self_attn_k_proj_weight_q -> t1015_c
node matmul_q4_58 micro s.matmul_q4_2 t1011_y model_layers_8_self_attn_v_proj_weight_q -> t1017_c
node rope_16 micro s.rope t1013_c position -> t1013_c
node rope_17 micro s.rope_2 t1015_c position -> t1015_c
node kv_scatter_16 micro s.kv_scatter t1015_c position s526 -> s526
node kv_scatter_17 micro s.kv_scatter t1017_c position s527 -> s527
node flash_prefill_8 micro s.flash_prefill t1013_c s526 s527 position -> t1023_o
node matmul_q4_59 micro s.matmul_q4 t1023_o model_layers_8_self_attn_o_proj_weight_q -> t1025_c
node vec_add_16 micro s.vec_add t1009_c t1025_c -> t1027_c
node rmsnorm_17 micro s.rmsnorm t1027_c model_layers_8_post_attention_layernorm_weight -> t1029_y
node matmul_q4_60 micro s.matmul_q4_3 t1029_y model_layers_8_mlp_gate_proj_weight_q -> t1031_c
node matmul_q4_61 micro s.matmul_q4_3 t1029_y model_layers_8_mlp_up_proj_weight_q -> t1033_c
node act_sigmoid_8 micro s.act_sigmoid t1031_c -> t1035_y
node vec_mul_16 micro s.vec_mul t1031_c t1035_y -> t1037_c
node vec_mul_17 micro s.vec_mul t1037_c t1033_c -> t1039_c
node matmul_q4_62 micro s.matmul_q4_4 t1039_c model_layers_8_mlp_down_proj_weight_q -> t1041_c
node vec_add_17 micro s.vec_add t1027_c t1041_c -> t1043_c
node rmsnorm_18 micro s.rmsnorm t1043_c model_layers_9_input_layernorm_weight -> t1167_y
node matmul_q4_63 micro s.matmul_q4 t1167_y model_layers_9_self_attn_q_proj_weight_q -> t1169_c
node matmul_q4_64 micro s.matmul_q4_2 t1167_y model_layers_9_self_attn_k_proj_weight_q -> t1171_c
node matmul_q4_65 micro s.matmul_q4_2 t1167_y model_layers_9_self_attn_v_proj_weight_q -> t1173_c
node rope_18 micro s.rope t1169_c position -> t1169_c
node rope_19 micro s.rope_2 t1171_c position -> t1171_c
node kv_scatter_18 micro s.kv_scatter t1171_c position s528 -> s528
node kv_scatter_19 micro s.kv_scatter t1173_c position s529 -> s529
node flash_prefill_9 micro s.flash_prefill t1169_c s528 s529 position -> t1179_o
node matmul_q4_66 micro s.matmul_q4 t1179_o model_layers_9_self_attn_o_proj_weight_q -> t1181_c
node vec_add_18 micro s.vec_add t1043_c t1181_c -> t1183_c
node rmsnorm_19 micro s.rmsnorm t1183_c model_layers_9_post_attention_layernorm_weight -> t1185_y
node matmul_q4_67 micro s.matmul_q4_3 t1185_y model_layers_9_mlp_gate_proj_weight_q -> t1187_c
node matmul_q4_68 micro s.matmul_q4_3 t1185_y model_layers_9_mlp_up_proj_weight_q -> t1189_c
node silu_mul_0 micro s.silu_mul t1187_c t1189_c -> t1195_c
node matmul_q4_69 micro s.matmul_q4_4 t1195_c model_layers_9_mlp_down_proj_weight_q -> t1197_c
node vec_add_19 micro s.vec_add t1183_c t1197_c -> t1199_c
node rmsnorm_20 micro s.rmsnorm t1199_c model_layers_10_input_layernorm_weight -> t1201_y
node matmul_q4_70 micro s.matmul_q4 t1201_y model_layers_10_self_attn_q_proj_weight_q -> t1203_c
node matmul_q4_71 micro s.matmul_q4_2 t1201_y model_layers_10_self_attn_k_proj_weight_q -> t1205_c
node matmul_q4_72 micro s.matmul_q4_2 t1201_y model_layers_10_self_attn_v_proj_weight_q -> t1207_c
node rope_20 micro s.rope t1203_c position -> t1203_c
node rope_21 micro s.rope_2 t1205_c position -> t1205_c
node kv_scatter_20 micro s.kv_scatter t1205_c position s530 -> s530
node kv_scatter_21 micro s.kv_scatter t1207_c position s531 -> s531
node flash_prefill_10 micro s.flash_prefill t1203_c s530 s531 position -> t1213_o
node matmul_q4_73 micro s.matmul_q4 t1213_o model_layers_10_self_attn_o_proj_weight_q -> t1215_c
node vec_add_20 micro s.vec_add t1199_c t1215_c -> t1217_c
node rmsnorm_21 micro s.rmsnorm t1217_c model_layers_10_post_attention_layernorm_weight -> t1219_y
node matmul_q4_74 micro s.matmul_q4_3 t1219_y model_layers_10_mlp_gate_proj_weight_q -> t1221_c
node matmul_q4_75 micro s.matmul_q4_3 t1219_y model_layers_10_mlp_up_proj_weight_q -> t1223_c
node silu_mul_1 micro s.silu_mul t1221_c t1223_c -> t1229_c
node matmul_q4_76 micro s.matmul_q4_4 t1229_c model_layers_10_mlp_down_proj_weight_q -> t1231_c
node vec_add_21 micro s.vec_add t1217_c t1231_c -> t1233_c
node rmsnorm_22 micro s.rmsnorm t1233_c model_layers_11_input_layernorm_weight -> t1235_y
node matmul_q4_77 micro s.matmul_q4 t1235_y model_layers_11_self_attn_q_proj_weight_q -> t1237_c
node matmul_q4_78 micro s.matmul_q4_2 t1235_y model_layers_11_self_attn_k_proj_weight_q -> t1239_c
node matmul_q4_79 micro s.matmul_q4_2 t1235_y model_layers_11_self_attn_v_proj_weight_q -> t1241_c
node rope_22 micro s.rope t1237_c position -> t1237_c
node rope_23 micro s.rope_2 t1239_c position -> t1239_c
node kv_scatter_22 micro s.kv_scatter t1239_c position s532 -> s532
node kv_scatter_23 micro s.kv_scatter t1241_c position s533 -> s533
node flash_prefill_11 micro s.flash_prefill t1237_c s532 s533 position -> t1247_o
node matmul_q4_80 micro s.matmul_q4 t1247_o model_layers_11_self_attn_o_proj_weight_q -> t1249_c
node vec_add_22 micro s.vec_add t1233_c t1249_c -> t1251_c
node rmsnorm_23 micro s.rmsnorm t1251_c model_layers_11_post_attention_layernorm_weight -> t1253_y
node matmul_q4_81 micro s.matmul_q4_3 t1253_y model_layers_11_mlp_gate_proj_weight_q -> t1255_c
node matmul_q4_82 micro s.matmul_q4_3 t1253_y model_layers_11_mlp_up_proj_weight_q -> t1257_c
node silu_mul_2 micro s.silu_mul t1255_c t1257_c -> t1263_c
node matmul_q4_83 micro s.matmul_q4_4 t1263_c model_layers_11_mlp_down_proj_weight_q -> t1265_c
node vec_add_23 micro s.vec_add t1251_c t1265_c -> t1267_c
node rmsnorm_24 micro s.rmsnorm t1267_c model_layers_12_input_layernorm_weight -> t1269_y
node matmul_q4_84 micro s.matmul_q4 t1269_y model_layers_12_self_attn_q_proj_weight_q -> t1271_c
node matmul_q4_85 micro s.matmul_q4_2 t1269_y model_layers_12_self_attn_k_proj_weight_q -> t1273_c
node matmul_q4_86 micro s.matmul_q4_2 t1269_y model_layers_12_self_attn_v_proj_weight_q -> t1275_c
node rope_24 micro s.rope t1271_c position -> t1271_c
node rope_25 micro s.rope_2 t1273_c position -> t1273_c
node kv_scatter_24 micro s.kv_scatter t1273_c position s534 -> s534
node kv_scatter_25 micro s.kv_scatter t1275_c position s535 -> s535
node flash_prefill_12 micro s.flash_prefill t1271_c s534 s535 position -> t1281_o
node matmul_q4_87 micro s.matmul_q4 t1281_o model_layers_12_self_attn_o_proj_weight_q -> t1283_c
node vec_add_24 micro s.vec_add t1267_c t1283_c -> t1285_c
node rmsnorm_25 micro s.rmsnorm t1285_c model_layers_12_post_attention_layernorm_weight -> t1287_y
node matmul_q4_88 micro s.matmul_q4_3 t1287_y model_layers_12_mlp_gate_proj_weight_q -> t1289_c
node matmul_q4_89 micro s.matmul_q4_3 t1287_y model_layers_12_mlp_up_proj_weight_q -> t1291_c
node act_sigmoid_9 micro s.act_sigmoid t1289_c -> t1293_y
node vec_mul_18 micro s.vec_mul t1289_c t1293_y -> t1295_c
node vec_mul_19 micro s.vec_mul t1295_c t1291_c -> t1297_c
node matmul_q4_90 micro s.matmul_q4_4 t1297_c model_layers_12_mlp_down_proj_weight_q -> t1299_c
node vec_add_25 micro s.vec_add t1285_c t1299_c -> t1301_c
node rmsnorm_26 micro s.rmsnorm t1301_c model_layers_13_input_layernorm_weight -> t1303_y
node matmul_q4_91 micro s.matmul_q4 t1303_y model_layers_13_self_attn_q_proj_weight_q -> t1305_c
node matmul_q4_92 micro s.matmul_q4_2 t1303_y model_layers_13_self_attn_k_proj_weight_q -> t1307_c
node matmul_q4_93 micro s.matmul_q4_2 t1303_y model_layers_13_self_attn_v_proj_weight_q -> t1309_c
node rope_26 micro s.rope t1305_c position -> t1305_c
node rope_27 micro s.rope_2 t1307_c position -> t1307_c
node kv_scatter_26 micro s.kv_scatter t1307_c position s536 -> s536
node kv_scatter_27 micro s.kv_scatter t1309_c position s537 -> s537
node flash_prefill_13 micro s.flash_prefill t1305_c s536 s537 position -> t1315_o
node matmul_q4_94 micro s.matmul_q4 t1315_o model_layers_13_self_attn_o_proj_weight_q -> t1317_c
node vec_add_26 micro s.vec_add t1301_c t1317_c -> t1319_c
node rmsnorm_27 micro s.rmsnorm t1319_c model_layers_13_post_attention_layernorm_weight -> t1321_y
node matmul_q4_95 micro s.matmul_q4_3 t1321_y model_layers_13_mlp_gate_proj_weight_q -> t1323_c
node matmul_q4_96 micro s.matmul_q4_3 t1321_y model_layers_13_mlp_up_proj_weight_q -> t1325_c
node act_sigmoid_10 micro s.act_sigmoid t1323_c -> t1327_y
node vec_mul_20 micro s.vec_mul t1323_c t1327_y -> t1329_c
node vec_mul_21 micro s.vec_mul t1329_c t1325_c -> t1331_c
node matmul_q4_97 micro s.matmul_q4_4 t1331_c model_layers_13_mlp_down_proj_weight_q -> t1333_c
node vec_add_27 micro s.vec_add t1319_c t1333_c -> t1335_c
node rmsnorm_28 micro s.rmsnorm t1335_c model_layers_14_input_layernorm_weight -> t1337_y
node matmul_q4_98 micro s.matmul_q4 t1337_y model_layers_14_self_attn_q_proj_weight_q -> t1339_c
node matmul_q4_99 micro s.matmul_q4_2 t1337_y model_layers_14_self_attn_k_proj_weight_q -> t1341_c
node matmul_q4_100 micro s.matmul_q4_2 t1337_y model_layers_14_self_attn_v_proj_weight_q -> t1343_c
node rope_28 micro s.rope t1339_c position -> t1339_c
node rope_29 micro s.rope_2 t1341_c position -> t1341_c
node kv_scatter_28 micro s.kv_scatter t1341_c position s538 -> s538
node kv_scatter_29 micro s.kv_scatter t1343_c position s539 -> s539
node flash_prefill_14 micro s.flash_prefill t1339_c s538 s539 position -> t1349_o
node matmul_q4_101 micro s.matmul_q4 t1349_o model_layers_14_self_attn_o_proj_weight_q -> t1351_c
node vec_add_28 micro s.vec_add t1335_c t1351_c -> t1353_c
node rmsnorm_29 micro s.rmsnorm t1353_c model_layers_14_post_attention_layernorm_weight -> t1355_y
node matmul_q4_102 micro s.matmul_q4_3 t1355_y model_layers_14_mlp_gate_proj_weight_q -> t1357_c
node matmul_q4_103 micro s.matmul_q4_3 t1355_y model_layers_14_mlp_up_proj_weight_q -> t1359_c
node act_sigmoid_11 micro s.act_sigmoid t1357_c -> t1361_y
node vec_mul_22 micro s.vec_mul t1357_c t1361_y -> t1363_c
node vec_mul_23 micro s.vec_mul t1363_c t1359_c -> t1365_c
node matmul_q4_104 micro s.matmul_q4_4 t1365_c model_layers_14_mlp_down_proj_weight_q -> t1367_c
node vec_add_29 micro s.vec_add t1353_c t1367_c -> t1369_c
node rmsnorm_30 micro s.rmsnorm t1369_c model_layers_15_input_layernorm_weight -> t1371_y
node matmul_q4_105 micro s.matmul_q4 t1371_y model_layers_15_self_attn_q_proj_weight_q -> t1373_c
node matmul_q4_106 micro s.matmul_q4_2 t1371_y model_layers_15_self_attn_k_proj_weight_q -> t1375_c
node matmul_q4_107 micro s.matmul_q4_2 t1371_y model_layers_15_self_attn_v_proj_weight_q -> t1377_c
node rope_30 micro s.rope t1373_c position -> t1373_c
node rope_31 micro s.rope_2 t1375_c position -> t1375_c
node kv_scatter_30 micro s.kv_scatter t1375_c position s540 -> s540
node kv_scatter_31 micro s.kv_scatter t1377_c position s541 -> s541
node flash_prefill_15 micro s.flash_prefill t1373_c s540 s541 position -> t1383_o
node matmul_q4_108 micro s.matmul_q4 t1383_o model_layers_15_self_attn_o_proj_weight_q -> t1385_c
node vec_add_30 micro s.vec_add t1369_c t1385_c -> t1387_c
node rmsnorm_31 micro s.rmsnorm t1387_c model_layers_15_post_attention_layernorm_weight -> t1389_y
node matmul_q4_109 micro s.matmul_q4_3 t1389_y model_layers_15_mlp_gate_proj_weight_q -> t1391_c
node matmul_q4_110 micro s.matmul_q4_3 t1389_y model_layers_15_mlp_up_proj_weight_q -> t1393_c
node act_sigmoid_12 micro s.act_sigmoid t1391_c -> t1395_y
node vec_mul_24 micro s.vec_mul t1391_c t1395_y -> t1397_c
node vec_mul_25 micro s.vec_mul t1397_c t1393_c -> t1399_c
node matmul_q4_111 micro s.matmul_q4_4 t1399_c model_layers_15_mlp_down_proj_weight_q -> t1401_c
node vec_add_31 micro s.vec_add t1387_c t1401_c -> t1403_c
node rmsnorm_32 micro s.rmsnorm t1403_c model_layers_16_input_layernorm_weight -> t1405_y
node matmul_q4_112 micro s.matmul_q4 t1405_y model_layers_16_self_attn_q_proj_weight_q -> t1407_c
node matmul_q4_113 micro s.matmul_q4_2 t1405_y model_layers_16_self_attn_k_proj_weight_q -> t1409_c
node matmul_q4_114 micro s.matmul_q4_2 t1405_y model_layers_16_self_attn_v_proj_weight_q -> t1411_c
node rope_32 micro s.rope t1407_c position -> t1407_c
node rope_33 micro s.rope_2 t1409_c position -> t1409_c
node kv_scatter_32 micro s.kv_scatter t1409_c position s542 -> s542
node kv_scatter_33 micro s.kv_scatter t1411_c position s543 -> s543
node flash_prefill_16 micro s.flash_prefill t1407_c s542 s543 position -> t1417_o
node matmul_q4_115 micro s.matmul_q4 t1417_o model_layers_16_self_attn_o_proj_weight_q -> t1419_c
node vec_add_32 micro s.vec_add t1403_c t1419_c -> t1421_c
node rmsnorm_33 micro s.rmsnorm t1421_c model_layers_16_post_attention_layernorm_weight -> t1423_y
node matmul_q4_116 micro s.matmul_q4_3 t1423_y model_layers_16_mlp_gate_proj_weight_q -> t1425_c
node matmul_q4_117 micro s.matmul_q4_3 t1423_y model_layers_16_mlp_up_proj_weight_q -> t1427_c
node act_sigmoid_13 micro s.act_sigmoid t1425_c -> t1429_y
node vec_mul_26 micro s.vec_mul t1425_c t1429_y -> t1431_c
node vec_mul_27 micro s.vec_mul t1431_c t1427_c -> t1433_c
node matmul_q4_118 micro s.matmul_q4_4 t1433_c model_layers_16_mlp_down_proj_weight_q -> t1435_c
node vec_add_33 micro s.vec_add t1421_c t1435_c -> t1437_c
node rmsnorm_34 micro s.rmsnorm t1437_c model_layers_17_input_layernorm_weight -> t1561_y
node matmul_q4_119 micro s.matmul_q4 t1561_y model_layers_17_self_attn_q_proj_weight_q -> t1563_c
node matmul_q4_120 micro s.matmul_q4_2 t1561_y model_layers_17_self_attn_k_proj_weight_q -> t1565_c
node matmul_q4_121 micro s.matmul_q4_2 t1561_y model_layers_17_self_attn_v_proj_weight_q -> t1567_c
node rope_34 micro s.rope t1563_c position -> t1563_c
node rope_35 micro s.rope_2 t1565_c position -> t1565_c
node kv_scatter_34 micro s.kv_scatter t1565_c position s544 -> s544
node kv_scatter_35 micro s.kv_scatter t1567_c position s545 -> s545
node flash_prefill_17 micro s.flash_prefill t1563_c s544 s545 position -> t1573_o
node matmul_q4_122 micro s.matmul_q4 t1573_o model_layers_17_self_attn_o_proj_weight_q -> t1575_c
node vec_add_34 micro s.vec_add t1437_c t1575_c -> t1577_c
node rmsnorm_35 micro s.rmsnorm t1577_c model_layers_17_post_attention_layernorm_weight -> t1579_y
node matmul_q4_123 micro s.matmul_q4_3 t1579_y model_layers_17_mlp_gate_proj_weight_q -> t1581_c
node matmul_q4_124 micro s.matmul_q4_3 t1579_y model_layers_17_mlp_up_proj_weight_q -> t1583_c
node silu_mul_3 micro s.silu_mul t1581_c t1583_c -> t1589_c
node matmul_q4_125 micro s.matmul_q4_4 t1589_c model_layers_17_mlp_down_proj_weight_q -> t1591_c
node vec_add_35 micro s.vec_add t1577_c t1591_c -> t1593_c
node rmsnorm_36 micro s.rmsnorm t1593_c model_layers_18_input_layernorm_weight -> t1595_y
node matmul_q4_126 micro s.matmul_q4 t1595_y model_layers_18_self_attn_q_proj_weight_q -> t1597_c
node matmul_q4_127 micro s.matmul_q4_2 t1595_y model_layers_18_self_attn_k_proj_weight_q -> t1599_c
node matmul_q4_128 micro s.matmul_q4_2 t1595_y model_layers_18_self_attn_v_proj_weight_q -> t1601_c
node rope_36 micro s.rope t1597_c position -> t1597_c
node rope_37 micro s.rope_2 t1599_c position -> t1599_c
node kv_scatter_36 micro s.kv_scatter t1599_c position s546 -> s546
node kv_scatter_37 micro s.kv_scatter t1601_c position s547 -> s547
node flash_prefill_18 micro s.flash_prefill t1597_c s546 s547 position -> t1607_o
node matmul_q4_129 micro s.matmul_q4 t1607_o model_layers_18_self_attn_o_proj_weight_q -> t1609_c
node vec_add_36 micro s.vec_add t1593_c t1609_c -> t1611_c
node rmsnorm_37 micro s.rmsnorm t1611_c model_layers_18_post_attention_layernorm_weight -> t1613_y
node matmul_q4_130 micro s.matmul_q4_3 t1613_y model_layers_18_mlp_gate_proj_weight_q -> t1615_c
node matmul_q4_131 micro s.matmul_q4_3 t1613_y model_layers_18_mlp_up_proj_weight_q -> t1617_c
node silu_mul_4 micro s.silu_mul t1615_c t1617_c -> t1623_c
node matmul_q4_132 micro s.matmul_q4_4 t1623_c model_layers_18_mlp_down_proj_weight_q -> t1625_c
node vec_add_37 micro s.vec_add t1611_c t1625_c -> t1627_c
node rmsnorm_38 micro s.rmsnorm t1627_c model_layers_19_input_layernorm_weight -> t1629_y
node matmul_q4_133 micro s.matmul_q4 t1629_y model_layers_19_self_attn_q_proj_weight_q -> t1631_c
node matmul_q4_134 micro s.matmul_q4_2 t1629_y model_layers_19_self_attn_k_proj_weight_q -> t1633_c
node matmul_q4_135 micro s.matmul_q4_2 t1629_y model_layers_19_self_attn_v_proj_weight_q -> t1635_c
node rope_38 micro s.rope t1631_c position -> t1631_c
node rope_39 micro s.rope_2 t1633_c position -> t1633_c
node kv_scatter_38 micro s.kv_scatter t1633_c position s548 -> s548
node kv_scatter_39 micro s.kv_scatter t1635_c position s549 -> s549
node flash_prefill_19 micro s.flash_prefill t1631_c s548 s549 position -> t1641_o
node matmul_q4_136 micro s.matmul_q4 t1641_o model_layers_19_self_attn_o_proj_weight_q -> t1643_c
node vec_add_38 micro s.vec_add t1627_c t1643_c -> t1645_c
node rmsnorm_39 micro s.rmsnorm t1645_c model_layers_19_post_attention_layernorm_weight -> t1647_y
node matmul_q4_137 micro s.matmul_q4_3 t1647_y model_layers_19_mlp_gate_proj_weight_q -> t1649_c
node matmul_q4_138 micro s.matmul_q4_3 t1647_y model_layers_19_mlp_up_proj_weight_q -> t1651_c
node silu_mul_5 micro s.silu_mul t1649_c t1651_c -> t1657_c
node matmul_q4_139 micro s.matmul_q4_4 t1657_c model_layers_19_mlp_down_proj_weight_q -> t1659_c
node vec_add_39 micro s.vec_add t1645_c t1659_c -> t1661_c
node rmsnorm_40 micro s.rmsnorm t1661_c model_layers_20_input_layernorm_weight -> t1663_y
node matmul_q4_140 micro s.matmul_q4 t1663_y model_layers_20_self_attn_q_proj_weight_q -> t1665_c
node matmul_q4_141 micro s.matmul_q4_2 t1663_y model_layers_20_self_attn_k_proj_weight_q -> t1667_c
node matmul_q4_142 micro s.matmul_q4_2 t1663_y model_layers_20_self_attn_v_proj_weight_q -> t1669_c
node rope_40 micro s.rope t1665_c position -> t1665_c
node rope_41 micro s.rope_2 t1667_c position -> t1667_c
node kv_scatter_40 micro s.kv_scatter t1667_c position s550 -> s550
node kv_scatter_41 micro s.kv_scatter t1669_c position s551 -> s551
node flash_prefill_20 micro s.flash_prefill t1665_c s550 s551 position -> t1675_o
node matmul_q4_143 micro s.matmul_q4 t1675_o model_layers_20_self_attn_o_proj_weight_q -> t1677_c
node vec_add_40 micro s.vec_add t1661_c t1677_c -> t1679_c
node rmsnorm_41 micro s.rmsnorm t1679_c model_layers_20_post_attention_layernorm_weight -> t1681_y
node matmul_q4_144 micro s.matmul_q4_3 t1681_y model_layers_20_mlp_gate_proj_weight_q -> t1683_c
node matmul_q4_145 micro s.matmul_q4_3 t1681_y model_layers_20_mlp_up_proj_weight_q -> t1685_c
node silu_mul_6 micro s.silu_mul t1683_c t1685_c -> t1691_c
node matmul_q4_146 micro s.matmul_q4_4 t1691_c model_layers_20_mlp_down_proj_weight_q -> t1693_c
node vec_add_41 micro s.vec_add t1679_c t1693_c -> t1695_c
node rmsnorm_42 micro s.rmsnorm t1695_c model_layers_21_input_layernorm_weight -> t1697_y
node matmul_q4_148 micro s.matmul_q4_2 t1697_y model_layers_21_self_attn_k_proj_weight_q -> t1701_c
node matmul_q4_149 micro s.matmul_q4_2 t1697_y model_layers_21_self_attn_v_proj_weight_q -> t1703_c
node rope_43 micro s.rope_2 t1701_c position -> t1701_c
node kv_scatter_42 micro s.kv_scatter t1701_c position s552 -> s552
node kv_scatter_43 micro s.kv_scatter t1703_c position s553 -> s553
phase decode
node embedding ext fir.embedding token model_embed_tokens_weight_emb -> t1821_embedding
node rmsnorm_45 micro s.rmsnorm_2 t1821_embedding model_layers_0_input_layernorm_weight -> t1826_y
node matmul_q4_155 micro s.matmul_q4_6 t1826_y model_layers_0_self_attn_q_proj_weight_q -> t1827_c
node matmul_q4_156 micro s.matmul_q4_7 t1826_y model_layers_0_self_attn_k_proj_weight_q -> t1828_c
node matmul_q4_157 micro s.matmul_q4_7 t1826_y model_layers_0_self_attn_v_proj_weight_q -> t1829_c
node rope_44 micro s.rope_3 t1827_c position -> t1827_c
node rope_45 micro s.rope_4 t1828_c position -> t1828_c
node kv_scatter_44 micro s.kv_scatter_2 t1828_c position s510 -> s510
node kv_scatter_45 micro s.kv_scatter_2 t1829_c position s511 -> s511
node attention_0 micro s.attention t1827_c s510 s511 position -> t1832_o
node matmul_q4_158 micro s.matmul_q4_6 t1832_o model_layers_0_self_attn_o_proj_weight_q -> t1835_c
node add_rmsnorm_0 micro s.add_rmsnorm t1821_embedding t1835_c model_layers_0_post_attention_layernorm_weight -> t1837_c t1842_y
node matmul_q4_159 micro s.matmul_q4_8 t1842_y model_layers_0_mlp_gate_proj_weight_q -> t1843_c
node matmul_q4_160 micro s.matmul_q4_8 t1842_y model_layers_0_mlp_up_proj_weight_q -> t1844_c
node silu_mul_8 micro s.silu_mul_2 t1843_c t1844_c -> t1850_c
node matmul_q4_161 micro s.matmul_q4_9 t1850_c model_layers_0_mlp_down_proj_weight_q -> t1853_c
node add_rmsnorm_1 micro s.add_rmsnorm t1837_c t1853_c model_layers_1_input_layernorm_weight -> t1855_c t1860_y
node matmul_q4_162 micro s.matmul_q4_6 t1860_y model_layers_1_self_attn_q_proj_weight_q -> t1861_c
node matmul_q4_163 micro s.matmul_q4_7 t1860_y model_layers_1_self_attn_k_proj_weight_q -> t1862_c
node matmul_q4_164 micro s.matmul_q4_7 t1860_y model_layers_1_self_attn_v_proj_weight_q -> t1863_c
node rope_46 micro s.rope_3 t1861_c position -> t1861_c
node rope_47 micro s.rope_4 t1862_c position -> t1862_c
node kv_scatter_46 micro s.kv_scatter_2 t1862_c position s512 -> s512
node kv_scatter_47 micro s.kv_scatter_2 t1863_c position s513 -> s513
node attention_1 micro s.attention t1861_c s512 s513 position -> t1866_o
node matmul_q4_165 micro s.matmul_q4_6 t1866_o model_layers_1_self_attn_o_proj_weight_q -> t1869_c
node add_rmsnorm_2 micro s.add_rmsnorm t1855_c t1869_c model_layers_1_post_attention_layernorm_weight -> t1871_c t1876_y
node matmul_q4_166 micro s.matmul_q4_8 t1876_y model_layers_1_mlp_gate_proj_weight_q -> t1877_c
node matmul_q4_167 micro s.matmul_q4_8 t1876_y model_layers_1_mlp_up_proj_weight_q -> t1878_c
node silu_mul_9 micro s.silu_mul_2 t1877_c t1878_c -> t1884_c
node matmul_q4_168 micro s.matmul_q4_9 t1884_c model_layers_1_mlp_down_proj_weight_q -> t1887_c
node add_rmsnorm_3 micro s.add_rmsnorm t1871_c t1887_c model_layers_2_input_layernorm_weight -> t1889_c t1894_y
node matmul_q4_169 micro s.matmul_q4_6 t1894_y model_layers_2_self_attn_q_proj_weight_q -> t1895_c
node matmul_q4_170 micro s.matmul_q4_7 t1894_y model_layers_2_self_attn_k_proj_weight_q -> t1896_c
node matmul_q4_171 micro s.matmul_q4_7 t1894_y model_layers_2_self_attn_v_proj_weight_q -> t1897_c
node rope_48 micro s.rope_3 t1895_c position -> t1895_c
node rope_49 micro s.rope_4 t1896_c position -> t1896_c
node kv_scatter_48 micro s.kv_scatter_2 t1896_c position s514 -> s514
node kv_scatter_49 micro s.kv_scatter_2 t1897_c position s515 -> s515
node attention_2 micro s.attention t1895_c s514 s515 position -> t1900_o
node matmul_q4_172 micro s.matmul_q4_6 t1900_o model_layers_2_self_attn_o_proj_weight_q -> t1903_c
node add_rmsnorm_4 micro s.add_rmsnorm t1889_c t1903_c model_layers_2_post_attention_layernorm_weight -> t1905_c t1910_y
node matmul_q4_173 micro s.matmul_q4_8 t1910_y model_layers_2_mlp_gate_proj_weight_q -> t1911_c
node matmul_q4_174 micro s.matmul_q4_8 t1910_y model_layers_2_mlp_up_proj_weight_q -> t1912_c
node silu_mul_10 micro s.silu_mul_2 t1911_c t1912_c -> t1918_c
node matmul_q4_175 micro s.matmul_q4_9 t1918_c model_layers_2_mlp_down_proj_weight_q -> t1921_c
node add_rmsnorm_5 micro s.add_rmsnorm t1905_c t1921_c model_layers_3_input_layernorm_weight -> t1923_c t1928_y
node matmul_q4_176 micro s.matmul_q4_6 t1928_y model_layers_3_self_attn_q_proj_weight_q -> t1929_c
node matmul_q4_177 micro s.matmul_q4_7 t1928_y model_layers_3_self_attn_k_proj_weight_q -> t1930_c
node matmul_q4_178 micro s.matmul_q4_7 t1928_y model_layers_3_self_attn_v_proj_weight_q -> t1931_c
node rope_50 micro s.rope_3 t1929_c position -> t1929_c
node rope_51 micro s.rope_4 t1930_c position -> t1930_c
node kv_scatter_50 micro s.kv_scatter_2 t1930_c position s516 -> s516
node kv_scatter_51 micro s.kv_scatter_2 t1931_c position s517 -> s517
node attention_3 micro s.attention t1929_c s516 s517 position -> t1934_o
node matmul_q4_179 micro s.matmul_q4_6 t1934_o model_layers_3_self_attn_o_proj_weight_q -> t1937_c
node add_rmsnorm_6 micro s.add_rmsnorm t1923_c t1937_c model_layers_3_post_attention_layernorm_weight -> t1939_c t1944_y
node matmul_q4_180 micro s.matmul_q4_8 t1944_y model_layers_3_mlp_gate_proj_weight_q -> t1945_c
node matmul_q4_181 micro s.matmul_q4_8 t1944_y model_layers_3_mlp_up_proj_weight_q -> t1946_c
node silu_mul_11 micro s.silu_mul_2 t1945_c t1946_c -> t1952_c
node matmul_q4_182 micro s.matmul_q4_9 t1952_c model_layers_3_mlp_down_proj_weight_q -> t1955_c
node add_rmsnorm_7 micro s.add_rmsnorm t1939_c t1955_c model_layers_4_input_layernorm_weight -> t1957_c t1962_y
node matmul_q4_183 micro s.matmul_q4_6 t1962_y model_layers_4_self_attn_q_proj_weight_q -> t1963_c
node matmul_q4_184 micro s.matmul_q4_7 t1962_y model_layers_4_self_attn_k_proj_weight_q -> t1964_c
node matmul_q4_185 micro s.matmul_q4_7 t1962_y model_layers_4_self_attn_v_proj_weight_q -> t1965_c
node rope_52 micro s.rope_3 t1963_c position -> t1963_c
node rope_53 micro s.rope_4 t1964_c position -> t1964_c
node kv_scatter_52 micro s.kv_scatter_2 t1964_c position s518 -> s518
node kv_scatter_53 micro s.kv_scatter_2 t1965_c position s519 -> s519
node attention_4 micro s.attention t1963_c s518 s519 position -> t1968_o
node matmul_q4_186 micro s.matmul_q4_6 t1968_o model_layers_4_self_attn_o_proj_weight_q -> t1971_c
node add_rmsnorm_8 micro s.add_rmsnorm t1957_c t1971_c model_layers_4_post_attention_layernorm_weight -> t1973_c t1978_y
node matmul_q4_187 micro s.matmul_q4_8 t1978_y model_layers_4_mlp_gate_proj_weight_q -> t1979_c
node matmul_q4_188 micro s.matmul_q4_8 t1978_y model_layers_4_mlp_up_proj_weight_q -> t1980_c
node silu_mul_12 micro s.silu_mul_2 t1979_c t1980_c -> t1986_c
node matmul_q4_189 micro s.matmul_q4_9 t1986_c model_layers_4_mlp_down_proj_weight_q -> t1989_c
node add_rmsnorm_9 micro s.add_rmsnorm t1973_c t1989_c model_layers_5_input_layernorm_weight -> t1991_c t1996_y
node matmul_q4_190 micro s.matmul_q4_6 t1996_y model_layers_5_self_attn_q_proj_weight_q -> t1997_c
node matmul_q4_191 micro s.matmul_q4_7 t1996_y model_layers_5_self_attn_k_proj_weight_q -> t1998_c
node matmul_q4_192 micro s.matmul_q4_7 t1996_y model_layers_5_self_attn_v_proj_weight_q -> t1999_c
node rope_54 micro s.rope_3 t1997_c position -> t1997_c
node rope_55 micro s.rope_4 t1998_c position -> t1998_c
node kv_scatter_54 micro s.kv_scatter_2 t1998_c position s520 -> s520
node kv_scatter_55 micro s.kv_scatter_2 t1999_c position s521 -> s521
node attention_5 micro s.attention t1997_c s520 s521 position -> t2002_o
node matmul_q4_193 micro s.matmul_q4_6 t2002_o model_layers_5_self_attn_o_proj_weight_q -> t2005_c
node add_rmsnorm_10 micro s.add_rmsnorm t1991_c t2005_c model_layers_5_post_attention_layernorm_weight -> t2007_c t2012_y
node matmul_q4_194 micro s.matmul_q4_8 t2012_y model_layers_5_mlp_gate_proj_weight_q -> t2013_c
node matmul_q4_195 micro s.matmul_q4_8 t2012_y model_layers_5_mlp_up_proj_weight_q -> t2014_c
node silu_mul_13 micro s.silu_mul_2 t2013_c t2014_c -> t2020_c
node matmul_q4_196 micro s.matmul_q4_9 t2020_c model_layers_5_mlp_down_proj_weight_q -> t2023_c
node add_rmsnorm_11 micro s.add_rmsnorm t2007_c t2023_c model_layers_6_input_layernorm_weight -> t2025_c t2030_y
node matmul_q4_197 micro s.matmul_q4_6 t2030_y model_layers_6_self_attn_q_proj_weight_q -> t2031_c
node matmul_q4_198 micro s.matmul_q4_7 t2030_y model_layers_6_self_attn_k_proj_weight_q -> t2032_c
node matmul_q4_199 micro s.matmul_q4_7 t2030_y model_layers_6_self_attn_v_proj_weight_q -> t2033_c
node rope_56 micro s.rope_3 t2031_c position -> t2031_c
node rope_57 micro s.rope_4 t2032_c position -> t2032_c
node kv_scatter_56 micro s.kv_scatter_2 t2032_c position s522 -> s522
node kv_scatter_57 micro s.kv_scatter_2 t2033_c position s523 -> s523
node attention_6 micro s.attention t2031_c s522 s523 position -> t2036_o
node matmul_q4_200 micro s.matmul_q4_6 t2036_o model_layers_6_self_attn_o_proj_weight_q -> t2039_c
node add_rmsnorm_12 micro s.add_rmsnorm t2025_c t2039_c model_layers_6_post_attention_layernorm_weight -> t2041_c t2046_y
node matmul_q4_201 micro s.matmul_q4_8 t2046_y model_layers_6_mlp_gate_proj_weight_q -> t2047_c
node matmul_q4_202 micro s.matmul_q4_8 t2046_y model_layers_6_mlp_up_proj_weight_q -> t2048_c
node silu_mul_14 micro s.silu_mul_2 t2047_c t2048_c -> t2054_c
node matmul_q4_203 micro s.matmul_q4_9 t2054_c model_layers_6_mlp_down_proj_weight_q -> t2057_c
node add_rmsnorm_13 micro s.add_rmsnorm t2041_c t2057_c model_layers_7_input_layernorm_weight -> t2059_c t2064_y
node matmul_q4_204 micro s.matmul_q4_6 t2064_y model_layers_7_self_attn_q_proj_weight_q -> t2065_c
node matmul_q4_205 micro s.matmul_q4_7 t2064_y model_layers_7_self_attn_k_proj_weight_q -> t2066_c
node matmul_q4_206 micro s.matmul_q4_7 t2064_y model_layers_7_self_attn_v_proj_weight_q -> t2067_c
node rope_58 micro s.rope_3 t2065_c position -> t2065_c
node rope_59 micro s.rope_4 t2066_c position -> t2066_c
node kv_scatter_58 micro s.kv_scatter_2 t2066_c position s524 -> s524
node kv_scatter_59 micro s.kv_scatter_2 t2067_c position s525 -> s525
node attention_7 micro s.attention t2065_c s524 s525 position -> t2070_o
node matmul_q4_207 micro s.matmul_q4_6 t2070_o model_layers_7_self_attn_o_proj_weight_q -> t2073_c
node add_rmsnorm_14 micro s.add_rmsnorm t2059_c t2073_c model_layers_7_post_attention_layernorm_weight -> t2075_c t2080_y
node matmul_q4_208 micro s.matmul_q4_8 t2080_y model_layers_7_mlp_gate_proj_weight_q -> t2081_c
node matmul_q4_209 micro s.matmul_q4_8 t2080_y model_layers_7_mlp_up_proj_weight_q -> t2082_c
node silu_mul_15 micro s.silu_mul_2 t2081_c t2082_c -> t2088_c
node matmul_q4_210 micro s.matmul_q4_9 t2088_c model_layers_7_mlp_down_proj_weight_q -> t2091_c
node add_rmsnorm_15 micro s.add_rmsnorm t2075_c t2091_c model_layers_8_input_layernorm_weight -> t2093_c t2098_y
node matmul_q4_211 micro s.matmul_q4_6 t2098_y model_layers_8_self_attn_q_proj_weight_q -> t2099_c
node matmul_q4_212 micro s.matmul_q4_7 t2098_y model_layers_8_self_attn_k_proj_weight_q -> t2100_c
node matmul_q4_213 micro s.matmul_q4_7 t2098_y model_layers_8_self_attn_v_proj_weight_q -> t2101_c
node rope_60 micro s.rope_3 t2099_c position -> t2099_c
node rope_61 micro s.rope_4 t2100_c position -> t2100_c
node kv_scatter_60 micro s.kv_scatter_2 t2100_c position s526 -> s526
node kv_scatter_61 micro s.kv_scatter_2 t2101_c position s527 -> s527
node attention_8 micro s.attention t2099_c s526 s527 position -> t2104_o
node matmul_q4_214 micro s.matmul_q4_6 t2104_o model_layers_8_self_attn_o_proj_weight_q -> t2107_c
node add_rmsnorm_16 micro s.add_rmsnorm t2093_c t2107_c model_layers_8_post_attention_layernorm_weight -> t2109_c t2114_y
node matmul_q4_215 micro s.matmul_q4_8 t2114_y model_layers_8_mlp_gate_proj_weight_q -> t2115_c
node matmul_q4_216 micro s.matmul_q4_8 t2114_y model_layers_8_mlp_up_proj_weight_q -> t2116_c
node silu_mul_16 micro s.silu_mul_2 t2115_c t2116_c -> t2122_c
node matmul_q4_217 micro s.matmul_q4_9 t2122_c model_layers_8_mlp_down_proj_weight_q -> t2125_c
node add_rmsnorm_17 micro s.add_rmsnorm t2109_c t2125_c model_layers_9_input_layernorm_weight -> t2127_c t2132_y
node matmul_q4_218 micro s.matmul_q4_6 t2132_y model_layers_9_self_attn_q_proj_weight_q -> t2133_c
node matmul_q4_219 micro s.matmul_q4_7 t2132_y model_layers_9_self_attn_k_proj_weight_q -> t2134_c
node matmul_q4_220 micro s.matmul_q4_7 t2132_y model_layers_9_self_attn_v_proj_weight_q -> t2135_c
node rope_62 micro s.rope_3 t2133_c position -> t2133_c
node rope_63 micro s.rope_4 t2134_c position -> t2134_c
node kv_scatter_62 micro s.kv_scatter_2 t2134_c position s528 -> s528
node kv_scatter_63 micro s.kv_scatter_2 t2135_c position s529 -> s529
node attention_9 micro s.attention t2133_c s528 s529 position -> t2138_o
node matmul_q4_221 micro s.matmul_q4_6 t2138_o model_layers_9_self_attn_o_proj_weight_q -> t2141_c
node add_rmsnorm_18 micro s.add_rmsnorm t2127_c t2141_c model_layers_9_post_attention_layernorm_weight -> t2143_c t2148_y
node matmul_q4_222 micro s.matmul_q4_8 t2148_y model_layers_9_mlp_gate_proj_weight_q -> t2149_c
node matmul_q4_223 micro s.matmul_q4_8 t2148_y model_layers_9_mlp_up_proj_weight_q -> t2150_c
node silu_mul_17 micro s.silu_mul_2 t2149_c t2150_c -> t2156_c
node matmul_q4_224 micro s.matmul_q4_9 t2156_c model_layers_9_mlp_down_proj_weight_q -> t2159_c
node add_rmsnorm_19 micro s.add_rmsnorm t2143_c t2159_c model_layers_10_input_layernorm_weight -> t2161_c t2166_y
node matmul_q4_225 micro s.matmul_q4_6 t2166_y model_layers_10_self_attn_q_proj_weight_q -> t2167_c
node matmul_q4_226 micro s.matmul_q4_7 t2166_y model_layers_10_self_attn_k_proj_weight_q -> t2168_c
node matmul_q4_227 micro s.matmul_q4_7 t2166_y model_layers_10_self_attn_v_proj_weight_q -> t2169_c
node rope_64 micro s.rope_3 t2167_c position -> t2167_c
node rope_65 micro s.rope_4 t2168_c position -> t2168_c
node kv_scatter_64 micro s.kv_scatter_2 t2168_c position s530 -> s530
node kv_scatter_65 micro s.kv_scatter_2 t2169_c position s531 -> s531
node attention_10 micro s.attention t2167_c s530 s531 position -> t2172_o
node matmul_q4_228 micro s.matmul_q4_6 t2172_o model_layers_10_self_attn_o_proj_weight_q -> t2175_c
node add_rmsnorm_20 micro s.add_rmsnorm t2161_c t2175_c model_layers_10_post_attention_layernorm_weight -> t2177_c t2182_y
node matmul_q4_229 micro s.matmul_q4_8 t2182_y model_layers_10_mlp_gate_proj_weight_q -> t2183_c
node matmul_q4_230 micro s.matmul_q4_8 t2182_y model_layers_10_mlp_up_proj_weight_q -> t2184_c
node silu_mul_18 micro s.silu_mul_2 t2183_c t2184_c -> t2190_c
node matmul_q4_231 micro s.matmul_q4_9 t2190_c model_layers_10_mlp_down_proj_weight_q -> t2193_c
node add_rmsnorm_21 micro s.add_rmsnorm t2177_c t2193_c model_layers_11_input_layernorm_weight -> t2195_c t2200_y
node matmul_q4_232 micro s.matmul_q4_6 t2200_y model_layers_11_self_attn_q_proj_weight_q -> t2201_c
node matmul_q4_233 micro s.matmul_q4_7 t2200_y model_layers_11_self_attn_k_proj_weight_q -> t2202_c
node matmul_q4_234 micro s.matmul_q4_7 t2200_y model_layers_11_self_attn_v_proj_weight_q -> t2203_c
node rope_66 micro s.rope_3 t2201_c position -> t2201_c
node rope_67 micro s.rope_4 t2202_c position -> t2202_c
node kv_scatter_66 micro s.kv_scatter_2 t2202_c position s532 -> s532
node kv_scatter_67 micro s.kv_scatter_2 t2203_c position s533 -> s533
node attention_11 micro s.attention t2201_c s532 s533 position -> t2206_o
node matmul_q4_235 micro s.matmul_q4_6 t2206_o model_layers_11_self_attn_o_proj_weight_q -> t2209_c
node add_rmsnorm_22 micro s.add_rmsnorm t2195_c t2209_c model_layers_11_post_attention_layernorm_weight -> t2211_c t2216_y
node matmul_q4_236 micro s.matmul_q4_8 t2216_y model_layers_11_mlp_gate_proj_weight_q -> t2217_c
node matmul_q4_237 micro s.matmul_q4_8 t2216_y model_layers_11_mlp_up_proj_weight_q -> t2218_c
node silu_mul_19 micro s.silu_mul_2 t2217_c t2218_c -> t2224_c
node matmul_q4_238 micro s.matmul_q4_9 t2224_c model_layers_11_mlp_down_proj_weight_q -> t2227_c
node add_rmsnorm_23 micro s.add_rmsnorm t2211_c t2227_c model_layers_12_input_layernorm_weight -> t2229_c t2234_y
node matmul_q4_239 micro s.matmul_q4_6 t2234_y model_layers_12_self_attn_q_proj_weight_q -> t2235_c
node matmul_q4_240 micro s.matmul_q4_7 t2234_y model_layers_12_self_attn_k_proj_weight_q -> t2236_c
node matmul_q4_241 micro s.matmul_q4_7 t2234_y model_layers_12_self_attn_v_proj_weight_q -> t2237_c
node rope_68 micro s.rope_3 t2235_c position -> t2235_c
node rope_69 micro s.rope_4 t2236_c position -> t2236_c
node kv_scatter_68 micro s.kv_scatter_2 t2236_c position s534 -> s534
node kv_scatter_69 micro s.kv_scatter_2 t2237_c position s535 -> s535
node attention_12 micro s.attention t2235_c s534 s535 position -> t2240_o
node matmul_q4_242 micro s.matmul_q4_6 t2240_o model_layers_12_self_attn_o_proj_weight_q -> t2243_c
node add_rmsnorm_24 micro s.add_rmsnorm t2229_c t2243_c model_layers_12_post_attention_layernorm_weight -> t2245_c t2250_y
node matmul_q4_243 micro s.matmul_q4_8 t2250_y model_layers_12_mlp_gate_proj_weight_q -> t2251_c
node matmul_q4_244 micro s.matmul_q4_8 t2250_y model_layers_12_mlp_up_proj_weight_q -> t2252_c
node silu_mul_20 micro s.silu_mul_2 t2251_c t2252_c -> t2258_c
node matmul_q4_245 micro s.matmul_q4_9 t2258_c model_layers_12_mlp_down_proj_weight_q -> t2261_c
node add_rmsnorm_25 micro s.add_rmsnorm t2245_c t2261_c model_layers_13_input_layernorm_weight -> t2263_c t2268_y
node matmul_q4_246 micro s.matmul_q4_6 t2268_y model_layers_13_self_attn_q_proj_weight_q -> t2269_c
node matmul_q4_247 micro s.matmul_q4_7 t2268_y model_layers_13_self_attn_k_proj_weight_q -> t2270_c
node matmul_q4_248 micro s.matmul_q4_7 t2268_y model_layers_13_self_attn_v_proj_weight_q -> t2271_c
node rope_70 micro s.rope_3 t2269_c position -> t2269_c
node rope_71 micro s.rope_4 t2270_c position -> t2270_c
node kv_scatter_70 micro s.kv_scatter_2 t2270_c position s536 -> s536
node kv_scatter_71 micro s.kv_scatter_2 t2271_c position s537 -> s537
node attention_13 micro s.attention t2269_c s536 s537 position -> t2274_o
node matmul_q4_249 micro s.matmul_q4_6 t2274_o model_layers_13_self_attn_o_proj_weight_q -> t2277_c
node add_rmsnorm_26 micro s.add_rmsnorm t2263_c t2277_c model_layers_13_post_attention_layernorm_weight -> t2279_c t2284_y
node matmul_q4_250 micro s.matmul_q4_8 t2284_y model_layers_13_mlp_gate_proj_weight_q -> t2285_c
node matmul_q4_251 micro s.matmul_q4_8 t2284_y model_layers_13_mlp_up_proj_weight_q -> t2286_c
node silu_mul_21 micro s.silu_mul_2 t2285_c t2286_c -> t2292_c
node matmul_q4_252 micro s.matmul_q4_9 t2292_c model_layers_13_mlp_down_proj_weight_q -> t2295_c
node add_rmsnorm_27 micro s.add_rmsnorm t2279_c t2295_c model_layers_14_input_layernorm_weight -> t2297_c t2302_y
node matmul_q4_253 micro s.matmul_q4_6 t2302_y model_layers_14_self_attn_q_proj_weight_q -> t2303_c
node matmul_q4_254 micro s.matmul_q4_7 t2302_y model_layers_14_self_attn_k_proj_weight_q -> t2304_c
node matmul_q4_255 micro s.matmul_q4_7 t2302_y model_layers_14_self_attn_v_proj_weight_q -> t2305_c
node rope_72 micro s.rope_3 t2303_c position -> t2303_c
node rope_73 micro s.rope_4 t2304_c position -> t2304_c
node kv_scatter_72 micro s.kv_scatter_2 t2304_c position s538 -> s538
node kv_scatter_73 micro s.kv_scatter_2 t2305_c position s539 -> s539
node attention_14 micro s.attention t2303_c s538 s539 position -> t2308_o
node matmul_q4_256 micro s.matmul_q4_6 t2308_o model_layers_14_self_attn_o_proj_weight_q -> t2311_c
node add_rmsnorm_28 micro s.add_rmsnorm t2297_c t2311_c model_layers_14_post_attention_layernorm_weight -> t2313_c t2318_y
node matmul_q4_257 micro s.matmul_q4_8 t2318_y model_layers_14_mlp_gate_proj_weight_q -> t2319_c
node matmul_q4_258 micro s.matmul_q4_8 t2318_y model_layers_14_mlp_up_proj_weight_q -> t2320_c
node silu_mul_22 micro s.silu_mul_2 t2319_c t2320_c -> t2326_c
node matmul_q4_259 micro s.matmul_q4_9 t2326_c model_layers_14_mlp_down_proj_weight_q -> t2329_c
node add_rmsnorm_29 micro s.add_rmsnorm t2313_c t2329_c model_layers_15_input_layernorm_weight -> t2331_c t2336_y
node matmul_q4_260 micro s.matmul_q4_6 t2336_y model_layers_15_self_attn_q_proj_weight_q -> t2337_c
node matmul_q4_261 micro s.matmul_q4_7 t2336_y model_layers_15_self_attn_k_proj_weight_q -> t2338_c
node matmul_q4_262 micro s.matmul_q4_7 t2336_y model_layers_15_self_attn_v_proj_weight_q -> t2339_c
node rope_74 micro s.rope_3 t2337_c position -> t2337_c
node rope_75 micro s.rope_4 t2338_c position -> t2338_c
node kv_scatter_74 micro s.kv_scatter_2 t2338_c position s540 -> s540
node kv_scatter_75 micro s.kv_scatter_2 t2339_c position s541 -> s541
node attention_15 micro s.attention t2337_c s540 s541 position -> t2342_o
node matmul_q4_263 micro s.matmul_q4_6 t2342_o model_layers_15_self_attn_o_proj_weight_q -> t2345_c
node add_rmsnorm_30 micro s.add_rmsnorm t2331_c t2345_c model_layers_15_post_attention_layernorm_weight -> t2347_c t2352_y
node matmul_q4_264 micro s.matmul_q4_8 t2352_y model_layers_15_mlp_gate_proj_weight_q -> t2353_c
node matmul_q4_265 micro s.matmul_q4_8 t2352_y model_layers_15_mlp_up_proj_weight_q -> t2354_c
node silu_mul_23 micro s.silu_mul_2 t2353_c t2354_c -> t2360_c
node matmul_q4_266 micro s.matmul_q4_9 t2360_c model_layers_15_mlp_down_proj_weight_q -> t2363_c
node add_rmsnorm_31 micro s.add_rmsnorm t2347_c t2363_c model_layers_16_input_layernorm_weight -> t2365_c t2370_y
node matmul_q4_267 micro s.matmul_q4_6 t2370_y model_layers_16_self_attn_q_proj_weight_q -> t2371_c
node matmul_q4_268 micro s.matmul_q4_7 t2370_y model_layers_16_self_attn_k_proj_weight_q -> t2372_c
node matmul_q4_269 micro s.matmul_q4_7 t2370_y model_layers_16_self_attn_v_proj_weight_q -> t2373_c
node rope_76 micro s.rope_3 t2371_c position -> t2371_c
node rope_77 micro s.rope_4 t2372_c position -> t2372_c
node kv_scatter_76 micro s.kv_scatter_2 t2372_c position s542 -> s542
node kv_scatter_77 micro s.kv_scatter_2 t2373_c position s543 -> s543
node attention_16 micro s.attention t2371_c s542 s543 position -> t2376_o
node matmul_q4_270 micro s.matmul_q4_6 t2376_o model_layers_16_self_attn_o_proj_weight_q -> t2379_c
node add_rmsnorm_32 micro s.add_rmsnorm t2365_c t2379_c model_layers_16_post_attention_layernorm_weight -> t2381_c t2386_y
node matmul_q4_271 micro s.matmul_q4_8 t2386_y model_layers_16_mlp_gate_proj_weight_q -> t2387_c
node matmul_q4_272 micro s.matmul_q4_8 t2386_y model_layers_16_mlp_up_proj_weight_q -> t2388_c
node silu_mul_24 micro s.silu_mul_2 t2387_c t2388_c -> t2394_c
node matmul_q4_273 micro s.matmul_q4_9 t2394_c model_layers_16_mlp_down_proj_weight_q -> t2397_c
node add_rmsnorm_33 micro s.add_rmsnorm t2381_c t2397_c model_layers_17_input_layernorm_weight -> t2399_c t2404_y
node matmul_q4_274 micro s.matmul_q4_6 t2404_y model_layers_17_self_attn_q_proj_weight_q -> t2405_c
node matmul_q4_275 micro s.matmul_q4_7 t2404_y model_layers_17_self_attn_k_proj_weight_q -> t2406_c
node matmul_q4_276 micro s.matmul_q4_7 t2404_y model_layers_17_self_attn_v_proj_weight_q -> t2407_c
node rope_78 micro s.rope_3 t2405_c position -> t2405_c
node rope_79 micro s.rope_4 t2406_c position -> t2406_c
node kv_scatter_78 micro s.kv_scatter_2 t2406_c position s544 -> s544
node kv_scatter_79 micro s.kv_scatter_2 t2407_c position s545 -> s545
node attention_17 micro s.attention t2405_c s544 s545 position -> t2410_o
node matmul_q4_277 micro s.matmul_q4_6 t2410_o model_layers_17_self_attn_o_proj_weight_q -> t2413_c
node add_rmsnorm_34 micro s.add_rmsnorm t2399_c t2413_c model_layers_17_post_attention_layernorm_weight -> t2415_c t2420_y
node matmul_q4_278 micro s.matmul_q4_8 t2420_y model_layers_17_mlp_gate_proj_weight_q -> t2421_c
node matmul_q4_279 micro s.matmul_q4_8 t2420_y model_layers_17_mlp_up_proj_weight_q -> t2422_c
node silu_mul_25 micro s.silu_mul_2 t2421_c t2422_c -> t2428_c
node matmul_q4_280 micro s.matmul_q4_9 t2428_c model_layers_17_mlp_down_proj_weight_q -> t2431_c
node add_rmsnorm_35 micro s.add_rmsnorm t2415_c t2431_c model_layers_18_input_layernorm_weight -> t2433_c t2438_y
node matmul_q4_281 micro s.matmul_q4_6 t2438_y model_layers_18_self_attn_q_proj_weight_q -> t2439_c
node matmul_q4_282 micro s.matmul_q4_7 t2438_y model_layers_18_self_attn_k_proj_weight_q -> t2440_c
node matmul_q4_283 micro s.matmul_q4_7 t2438_y model_layers_18_self_attn_v_proj_weight_q -> t2441_c
node rope_80 micro s.rope_3 t2439_c position -> t2439_c
node rope_81 micro s.rope_4 t2440_c position -> t2440_c
node kv_scatter_80 micro s.kv_scatter_2 t2440_c position s546 -> s546
node kv_scatter_81 micro s.kv_scatter_2 t2441_c position s547 -> s547
node attention_18 micro s.attention t2439_c s546 s547 position -> t2444_o
node matmul_q4_284 micro s.matmul_q4_6 t2444_o model_layers_18_self_attn_o_proj_weight_q -> t2447_c
node add_rmsnorm_36 micro s.add_rmsnorm t2433_c t2447_c model_layers_18_post_attention_layernorm_weight -> t2449_c t2454_y
node matmul_q4_285 micro s.matmul_q4_8 t2454_y model_layers_18_mlp_gate_proj_weight_q -> t2455_c
node matmul_q4_286 micro s.matmul_q4_8 t2454_y model_layers_18_mlp_up_proj_weight_q -> t2456_c
node silu_mul_26 micro s.silu_mul_2 t2455_c t2456_c -> t2462_c
node matmul_q4_287 micro s.matmul_q4_9 t2462_c model_layers_18_mlp_down_proj_weight_q -> t2465_c
node add_rmsnorm_37 micro s.add_rmsnorm t2449_c t2465_c model_layers_19_input_layernorm_weight -> t2467_c t2472_y
node matmul_q4_288 micro s.matmul_q4_6 t2472_y model_layers_19_self_attn_q_proj_weight_q -> t2473_c
node matmul_q4_289 micro s.matmul_q4_7 t2472_y model_layers_19_self_attn_k_proj_weight_q -> t2474_c
node matmul_q4_290 micro s.matmul_q4_7 t2472_y model_layers_19_self_attn_v_proj_weight_q -> t2475_c
node rope_82 micro s.rope_3 t2473_c position -> t2473_c
node rope_83 micro s.rope_4 t2474_c position -> t2474_c
node kv_scatter_82 micro s.kv_scatter_2 t2474_c position s548 -> s548
node kv_scatter_83 micro s.kv_scatter_2 t2475_c position s549 -> s549
node attention_19 micro s.attention t2473_c s548 s549 position -> t2478_o
node matmul_q4_291 micro s.matmul_q4_6 t2478_o model_layers_19_self_attn_o_proj_weight_q -> t2481_c
node add_rmsnorm_38 micro s.add_rmsnorm t2467_c t2481_c model_layers_19_post_attention_layernorm_weight -> t2483_c t2488_y
node matmul_q4_292 micro s.matmul_q4_8 t2488_y model_layers_19_mlp_gate_proj_weight_q -> t2489_c
node matmul_q4_293 micro s.matmul_q4_8 t2488_y model_layers_19_mlp_up_proj_weight_q -> t2490_c
node silu_mul_27 micro s.silu_mul_2 t2489_c t2490_c -> t2496_c
node matmul_q4_294 micro s.matmul_q4_9 t2496_c model_layers_19_mlp_down_proj_weight_q -> t2499_c
node add_rmsnorm_39 micro s.add_rmsnorm t2483_c t2499_c model_layers_20_input_layernorm_weight -> t2501_c t2506_y
node matmul_q4_295 micro s.matmul_q4_6 t2506_y model_layers_20_self_attn_q_proj_weight_q -> t2507_c
node matmul_q4_296 micro s.matmul_q4_7 t2506_y model_layers_20_self_attn_k_proj_weight_q -> t2508_c
node matmul_q4_297 micro s.matmul_q4_7 t2506_y model_layers_20_self_attn_v_proj_weight_q -> t2509_c
node rope_84 micro s.rope_3 t2507_c position -> t2507_c
node rope_85 micro s.rope_4 t2508_c position -> t2508_c
node kv_scatter_84 micro s.kv_scatter_2 t2508_c position s550 -> s550
node kv_scatter_85 micro s.kv_scatter_2 t2509_c position s551 -> s551
node attention_20 micro s.attention t2507_c s550 s551 position -> t2512_o
node matmul_q4_298 micro s.matmul_q4_6 t2512_o model_layers_20_self_attn_o_proj_weight_q -> t2515_c
node add_rmsnorm_40 micro s.add_rmsnorm t2501_c t2515_c model_layers_20_post_attention_layernorm_weight -> t2517_c t2522_y
node matmul_q4_299 micro s.matmul_q4_8 t2522_y model_layers_20_mlp_gate_proj_weight_q -> t2523_c
node matmul_q4_300 micro s.matmul_q4_8 t2522_y model_layers_20_mlp_up_proj_weight_q -> t2524_c
node silu_mul_28 micro s.silu_mul_2 t2523_c t2524_c -> t2530_c
node matmul_q4_301 micro s.matmul_q4_9 t2530_c model_layers_20_mlp_down_proj_weight_q -> t2533_c
node add_rmsnorm_41 micro s.add_rmsnorm t2517_c t2533_c model_layers_21_input_layernorm_weight -> t2535_c t2540_y
node matmul_q4_302 micro s.matmul_q4_6 t2540_y model_layers_21_self_attn_q_proj_weight_q -> t2541_c
node matmul_q4_303 micro s.matmul_q4_7 t2540_y model_layers_21_self_attn_k_proj_weight_q -> t2542_c
node matmul_q4_304 micro s.matmul_q4_7 t2540_y model_layers_21_self_attn_v_proj_weight_q -> t2543_c
node rope_86 micro s.rope_3 t2541_c position -> t2541_c
node rope_87 micro s.rope_4 t2542_c position -> t2542_c
node kv_scatter_86 micro s.kv_scatter_2 t2542_c position s552 -> s552
node kv_scatter_87 micro s.kv_scatter_2 t2543_c position s553 -> s553
node attention_21 micro s.attention t2541_c s552 s553 position -> t2546_o
node matmul_q4_305 micro s.matmul_q4_6 t2546_o model_layers_21_self_attn_o_proj_weight_q -> t2549_c
node add_rmsnorm_42 micro s.add_rmsnorm t2535_c t2549_c model_layers_21_post_attention_layernorm_weight -> t2551_c t2556_y
node matmul_q4_306 micro s.matmul_q4_8 t2556_y model_layers_21_mlp_gate_proj_weight_q -> t2557_c
node matmul_q4_307 micro s.matmul_q4_8 t2556_y model_layers_21_mlp_up_proj_weight_q -> t2558_c
node silu_mul_29 micro s.silu_mul_2 t2557_c t2558_c -> t2564_c
node matmul_q4_308 micro s.matmul_q4_9 t2564_c model_layers_21_mlp_down_proj_weight_q -> t2567_c
node add_rmsnorm_43 micro s.add_rmsnorm t2551_c t2567_c model_norm_weight -> t2569_c t2572_y
node matmul_q4_309 micro s.matmul_q4_5 t2572_y lm_head_weight_q -> t2573_c
