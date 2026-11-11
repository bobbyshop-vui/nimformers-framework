# TinyLlama 1.1B: inference benchmark

Runs through formers.n (`nim_inference` transformer blocks) from `model.nimq`, Metal backend
(Apple GPUs use the fused decode graph, the 64x64 and 32x32 simdgroup GEMM tiles and fp16 KV).
Chat mode (`-cnv`, temperature 0, chat template from `tokenizer_config.json`).

## English

### Short chat (18-token prompt, 100 generated tokens)

| Machine | Prompt eval | Decode |
|---|---:|---:|
| MacBook Air M1 | 0.099 s, **181.12 tok/s** | 1.250 s, **79.99 tok/s** (11.2 ms/token) |
| MacBook Pro Intel Iris Plus 640 | 0.727 s, **24.76 tok/s** | 5.438 s, **18.39 tok/s** |

### Long prompt (862-token prompt, 50 generated tokens)

| Machine | Prompt eval | Decode |
|---|---:|---:|
| MacBook Air M1 | 2.666 s, **323.35 tok/s** | 1.228 s, **40.71 tok/s** |
| MacBook Pro Intel Iris Plus 640 | 12.456 s, **69.21 tok/s** | 3.847 s, **13.00 tok/s** |

Intel Iris Plus Graphics 640 with Vulkan (MoltenVK), chat mode, temperature 0.7:
prompt 23 tokens in 1.146 s (20.07 tok/s), 45 generated tokens in 8.132 s (5.53 tok/s).

```sh
other/formers.n/formers-cli --models models/tinyllama -cnv --temp 0 --backend metal -n 100 <<< $'Hello\n/exit'
P=$(yes 'The quick brown fox jumps over the lazy dog.' | head -n 70 | tr '\n' ' ')
printf '%s Summarize.\n/exit\n' "$P" | other/formers.n/formers-cli --models models/tinyllama -cnv --temp 0 --backend metal -n 50
```

## Tiếng Việt

Chạy qua formers.n (khối transformer `nim_inference`) từ `model.nimq`, backend Metal
(GPU Apple dùng đồ thị decode gộp, tile GEMM simdgroup 64x64 và 32x32, KV fp16).
Chế độ chat (`-cnv`, temperature 0, template từ `tokenizer_config.json`).

### Chat ngắn (prompt 18 token, sinh 100 token)

| Máy | Prompt eval | Decode |
|---|---:|---:|
| MacBook Air M1 | 0,099 s, **181,12 tok/s** | 1,250 s, **79,99 tok/s** (11,2 ms/token) |
| MacBook Pro Intel Iris Plus 640 | 0,727 s, **24,76 tok/s** | 5,438 s, **18,39 tok/s** |

### Prompt dài (prompt 862 token, sinh 50 token)

| Máy | Prompt eval | Decode |
|---|---:|---:|
| MacBook Air M1 | 2,666 s, **323,35 tok/s** | 1,228 s, **40,71 tok/s** |
| MacBook Pro Intel Iris Plus 640 | 12,456 s, **69,21 tok/s** | 3,847 s, **13,00 tok/s** |

Intel Iris Plus Graphics 640 với Vulkan (MoltenVK), chế độ chat, temperature 0,7:
prompt 23 token trong 1,146 s (20,07 tok/s), sinh 45 token trong 8,132 s (5,53 tok/s).

```sh
other/formers.n/formers-cli --models models/tinyllama -cnv --temp 0 --backend metal -n 100 <<< $'Hello\n/exit'
P=$(yes 'The quick brown fox jumps over the lazy dog.' | head -n 70 | tr '\n' ' ')
printf '%s Summarize.\n/exit\n' "$P" | other/formers.n/formers-cli --models models/tinyllama -cnv --temp 0 --backend metal -n 50
```
