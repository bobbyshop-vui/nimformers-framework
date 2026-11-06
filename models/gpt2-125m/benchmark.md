# GPT-2 125M through FIR: inference benchmark

The model runs from `model.fir` only (`runtime graph`): one GPU session per token, KV cache
resident on the GPU. Q4 is the default (resident Q4 linears); `--fir-f32` keeps the linears
in fp32 as the correctness reference.

## English

### Intel Iris Plus Graphics 640

Measured in chat mode (`-cnv`, temperature 0.8, chat template from `chat_template.jinja`).

| Backend | Weights | Prompt tokens | Prompt eval | Generated tokens | Decode |
|---|---|---:|---:|---:|---:|
| Metal | Q4 | 13 | 0.583 s, 22.31 tok/s | 168 | 19.596 s, 8.57 tok/s |
| Vulkan | Q4 | 15 | 1.048 s, 14.31 tok/s | 35 | 3.071 s, 11.40 tok/s |
| Metal | fp32 | 10 | 0.732 s, 13.67 tok/s | 21 | 1.948 s, 10.78 tok/s |
| Vulkan | fp32 | 11 | 0.955 s, 11.52 tok/s | 51 | 5.974 s, 8.54 tok/s |

```sh
other/formers.n/formers-cli --models models/gpt2-125m -cnv --temp 0.8 --backend metal
other/formers.n/formers-cli --models models/gpt2-125m -cnv --temp 0.8 --backend vulkan
other/formers.n/formers-cli --models models/gpt2-125m -cnv --fir-f32 --temp 0.8 --backend metal
other/formers.n/formers-cli --models models/gpt2-125m -cnv --fir-f32 --temp 0.8 --backend vulkan
```

## Tiếng Việt

### Intel Iris Plus Graphics 640

Đo ở chế độ chat (`-cnv`, temperature 0.8, template từ `chat_template.jinja`). GPT-2 125M

| Backend | Weight | Token prompt | Prompt eval | Token sinh ra | Decode |
|---|---|---:|---:|---:|---:|
| Metal | Q4 | 13 | 0,583 s, 22,31 tok/s | 168 | 19,596 s, 8,57 tok/s |
| Vulkan | Q4 | 15 | 1,048 s, 14,31 tok/s | 35 | 3,071 s, 11,40 tok/s |
| Metal | fp32 | 10 | 0,732 s, 13,67 tok/s | 21 | 1,948 s, 10,78 tok/s |
| Vulkan | fp32 | 11 | 0,955 s, 11,52 tok/s | 51 | 5,974 s, 8,54 tok/s |

```sh
other/formers.n/formers-cli --models models/gpt2-125m -cnv --temp 0.8 --backend metal
other/formers.n/formers-cli --models models/gpt2-125m -cnv --temp 0.8 --backend vulkan
other/formers.n/formers-cli --models models/gpt2-125m -cnv --fir-f32 --temp 0.8 --backend metal
other/formers.n/formers-cli --models models/gpt2-125m -cnv --fir-f32 --temp 0.8 --backend vulkan
```

