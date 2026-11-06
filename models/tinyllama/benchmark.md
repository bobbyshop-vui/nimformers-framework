# TinyLlama 1.1B: inference benchmark

Runs through formers.n (`nim_inference` transformer blocks) from `model.nimq`.

## English

### Intel Iris Plus Graphics 640

Measured in chat mode (`-cnv`, temperature 0.7, chat template from `tokenizer_config.json`).


| Backend | Prompt tokens | Prompt eval | Generated tokens | Decode |
|---|---:|---:|---:|---:|
| Metal | 24 | 0.895 s, 26.80 tok/s | 32 | 5.388 s, 5.94 tok/s |
| Vulkan | 23 | 1.146 s, 20.07 tok/s | 45 | 8.132 s, 5.53 tok/s |

```sh
other/formers.n/formers-cli --models models/tinyllama -cnv --temp 0.7 --backend metal
other/formers.n/formers-cli --models models/tinyllama -cnv --temp 0.7 --backend vulkan
```

## Tiếng Việt

### Intel Iris Plus Graphics 640

Đo ở chế độ chat (`-cnv`, temperature 0.7, template từ `tokenizer_config.json`).

| Backend | Token prompt | Prompt eval | Token sinh ra | Decode |
|---|---:|---:|---:|---:|
| Metal | 24 | 0,895 s, 26,80 tok/s | 32 | 5,388 s, 5,94 tok/s |
| Vulkan | 23 | 1,146 s, 20,07 tok/s | 45 | 8,132 s, 5,53 tok/s |

```sh
other/formers.n/formers-cli --models models/tinyllama -cnv --temp 0.7 --backend metal
other/formers.n/formers-cli --models models/tinyllama -cnv --temp 0.7 --backend vulkan
```

