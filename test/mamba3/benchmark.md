# Mamba-3 inference benchmark

Model-specific results: [187M](#mamba-3-siso-187m) ·
[1.5B](#mamba-3-siso-15b).

## Mamba-3 SISO 187M

### English

### Intel Iris Plus Graphics 640

Measured on the Metal backend using the Mamba-3 SISO 187M model and the
`RRRRRRRRRRRG` fine-tuned checkpoint (`lora.bin`).

| Metric | Measurement |
|---|---:|
| Prompt | `What is the capital of France?` |
| Prompt tokens | 13 |
| Generated tokens | 32 |
| Prefill | 0.242 s, 53.73 tok/s |
| Decode | 1.251 s, 25.58 tok/s (39.1 ms/token) |
| Prefill + decode | 1.493 s for 45 tokens |
| Backend / GPU | Metal / Intel(R) Iris(TM) Plus Graphics 640 |

The run used greedy decoding (`temperature=0`) and kept the LoRA adapters
separate rather than merging them into the quantized base:

```sh
BYBY_MERGE_LORA=0 \
BYBY_MAX_NEW_TOKENS=32 \
./test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
  'What is the capital of France?' \
  0
```

The binary reported 48 LoRA adapters and the `RRRRRRRRRRRG` topology.

#### Chat (`formers-cli -cnv`)

Measured on the same GPU and backend in formers.n conversation mode (512 generated
tokens, chat template from the model folder).

| Metric | Measurement |
|---|---:|
| Prompt tokens | 121 |
| Generated tokens | 512 |
| Prompt eval | 3.747 s, 32.29 tok/s |
| Decode | 29.544 s, 17.33 tok/s |
| Backend / GPU | Metal / Intel(R) Iris(TM) Plus Graphics 640 |

### Tiếng Việt

#### Intel Iris Plus Graphics 640

Đo trên backend Metal với Mamba-3 SISO 187M và checkpoint đã finetune
`RRRRRRRRRRRG` (`lora.bin`).

| Chỉ số | Kết quả |
|---|---:|
| Prompt | `What is the capital of France?` |
| Số token prompt | 13 |
| Số token sinh | 32 |
| Prefill | 0.242 giây, 53.73 tok/s |
| Decode | 1.251 giây, 25.58 tok/s (39.1 ms/token) |
| Prefill + decode | 1.493 giây cho 45 token |
| Backend / GPU | Metal / Intel(R) Iris(TM) Plus Graphics 640 |

Lượt chạy dùng greedy decoding (`temperature=0`) và giữ các adapter LoRA tách
riêng, không gộp vào trọng số base đã lượng tử hóa:

```sh
BYBY_MERGE_LORA=0 \
BYBY_MAX_NEW_TOKENS=32 \
./test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
  'What is the capital of France?' \
  0
```

Binary ghi nhận 48 LoRA adapter và topology `RRRRRRRRRRRG`.

#### Chat (`formers-cli -cnv`)

Đo trên cùng GPU và backend ở chế độ conversation của formers.n (sinh 512 token,
chat template lấy từ thư mục model).

| Chỉ số | Kết quả |
|---|---:|
| Token prompt | 121 |
| Token sinh ra | 512 |
| Prompt eval | 3,747 giây, 32,29 tok/s |
| Decode | 29,544 giây, 17,33 tok/s |
| Backend / GPU | Metal / Intel(R) Iris(TM) Plus Graphics 640 |

## Mamba-3 SISO 1.5B

### English

On Intel Iris Plus Graphics 640 / Metal,
using the FP4-KM 1.5B base and the 96-adapter LoRA
checkpoint `runs/mamba3-1.5b/lora-g1.bin`, merged in place (23 R + G). Prompt:
`Write a Python function to check if a string is a palindrome`. Temperature
was 0.7. The prompt is 19 tokens including the final token; prefill measured
18 tokens.

- Prefill: 1.822 s, 9.88 tok/s; kernel setup/JIT 0.039 s, outside setup 1.783 s.
- Decode: 64 tokens in 13.084 s, 4.89 tok/s (204.4 ms/token). First token:
  0.240 s, including 0.018 s setup/JIT; 63 later tokens: 4.90 tok/s
  (203.9 ms/token).
- Total: 82 tokens in 14.906 s.
- Process peak RSS reported at the end of LoRA merge: 4631 MB (`ru_maxrss`).
  The log does not report a final peak after generation.

Reproduce the run with:

```sh
BYBY_METAL_JIT_LOG=0 \
BYBY_ONE_DISPATCH=0 \
BYBY_MAX_NEW_TOKENS=64 \
test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-1.5b-gated/lora-g1.bin \
  'Write a Python function to check if a string is a palindrome' \
  0.7
```


### Tiếng Việt

Chạy trên Intel Iris Plus Graphics 640 / Metal,
dùng base 1.5B FP4-KM và checkpoint LoRA
96 adapter `runs/mamba3-1.5b/lora-g1.bin`, đã merge tại chỗ (23 R + G). Prompt:
`Write a Python function to check if a string is a palindrome`. Temperature
0.7. Prompt có 19 token tính cả token cuối; prefill đo 18 token.

- Prefill: 1.822 giây, 9.88 tok/s; setup/JIT kernel 0.039 giây, ngoài setup
  1.783 giây.
- Decode: 64 token trong 13.084 giây, 4.89 tok/s (204.4 ms/token). Token đầu:
  0.240 giây, gồm 0.018 giây setup/JIT; 63 token sau đạt 4.90 tok/s
  (203.9 ms/token).
- Tổng: 82 token trong 14.906 giây.
- Peak RSS của process được báo lúc kết thúc merge LoRA: 4631 MB (`ru_maxrss`).
  Log không báo peak cuối cùng sau generation.

Chạy lại bằng lệnh:

```sh
BYBY_METAL_JIT_LOG=0 \
BYBY_ONE_DISPATCH=0 \
BYBY_MAX_NEW_TOKENS=64 \
test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-1.5b-gated/lora-g1.bin \
  'Write a Python function to check if a string is a palindrome' \
  0.7
```

