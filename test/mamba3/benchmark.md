# Mamba-3 inference benchmark

Model-specific results: [187M](#mamba-3-siso-187m) ·
[1.5B](#mamba-3-siso-15b).

## Mamba-3 SISO 187M

### English

### Intel Iris Plus Graphics 640

Measured on the Metal backend using the Mamba-3 SISO 187M model and the
`RRRRRRRRRRRG` fine-tuned checkpoint (`lora.bin`). Every speed on this page
leaves out the one-off kernel setup/JIT of the process (the cold start), which
the tools time and print separately.

| Metric | Measurement |
|---|---:|
| Prompt | `What is the capital of France?` |
| Prompt tokens | 13 |
| Generated tokens | 32 |
| Prefill | 0.159 s, 81.78 tok/s |
| Decode | 0.764 s, 41.88 tok/s (23.9 ms/token) |
| Decode, tokens 2-32 | 42.34 tok/s (23.6 ms/token) |
| Prefill + decode | 0.923 s for 45 tokens |
| Backend / GPU | Metal / Intel(R) Iris(TM) Plus Graphics 640 |

The run used greedy decoding (`temperature=0`) with the LoRA adapters merged in
place into the quantized base (the default). One run, not an average. The
kernel setup/JIT that `infer_lora_bin` prints on its own lines is a cold-start
cost of about 0.6-1.0 s for this prompt and differs from run to run; a prompt of
a new length compiles its own prefill kernels, so longer prompts pay more (about
2-3 s for the long prompts tried). With `BYBY_M3_PROFILE=1` each decode step is
split into host encode (about 1 ms), GPU wait (mean 21.9 ms, range 17.3-31.9 ms)
and logits read (about 0.2 ms), so decode on this GPU is bound by the GPU, not by
the host.

```sh
BYBY_MAX_NEW_TOKENS=32 \
BYBY_METAL_JIT_ARCHIVE_SAVE=0 \
./test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
  'What is the capital of France?' \
  0
```

`BYBY_METAL_JIT_ARCHIVE_SAVE=0` skips saving the JIT archive at exit, which can
take minutes on Intel. The binary reported 48 LoRA adapters and the
`RRRRRRRRRRRG` topology.

#### Prefill scheduler

The prompt goes through prefill in micro-batches of up to 128 rows
(`BYBY_UBATCH`), and on this GPU each micro-batch is split into command buffers
of about 300 ms (`BYBY_PREFILL_CB_MS`), sized from the measured time per layer.
`BYBY_PREFILL_BENCH=4` times one prompt under each scheduler mode inside one
process (one untimed warm-up round, modes interleaved, median of 4 runs), so the
numbers contain no kernel setup/JIT and no process-to-process noise. Prompt of
176 tokens (the sentence `The quick brown fox jumps over the lazy dog. ` repeated
17 times):

| Mode | Median | Best | Speed (median) |
|---|---:|---:|---:|
| One command buffer per micro-batch | 1.721 s | 1.389 s | 102.3 tok/s |
| Commit every 4 layers (the Apple GPU mode) | 1.668 s | 1.367 s | 105.5 tok/s |
| Command buffers of about 300 ms (default on this GPU) | 1.814 s | 1.488 s | 97.0 tok/s |

For this small model the pacing costs about 5-7% of the prefill speed on this
GPU; its purpose is to keep every command buffer short for larger models, not
speed. The commit-every-4-layers mode is what Apple GPUs use by default; it was
timed here only for comparison and has not been measured on an Apple GPU.

```sh
P=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' {1..17})
BYBY_PREFILL_BENCH=4 BYBY_MAX_NEW_TOKENS=1 \
BYBY_METAL_JIT_ARCHIVE_SAVE=0 \
./test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin "$P" 0
```

#### Chat (`formers-cli -cnv`)

Measured on the same GPU and backend in formers.n conversation mode: three turns
in one session, up to 512 generated tokens per reply, `--temp 0.7`, chat template
from the model folder. `ready: warmup` compiled the prefill and decode kernels
before the first prompt (3.8 s in this run), so these speeds contain no kernel
setup/JIT. Each turn feeds its prompt through the prefill scheduler.

| Turn | Prompt tokens | Prompt eval | Generated tokens | Decode |
|---:|---:|---:|---:|---:|
| 1 | 27 | 0.216 s, 125.07 tok/s | 512 | 12.138 s, 42.18 tok/s |
| 2 | 19 | 0.157 s, 121.16 tok/s | 512 | 12.204 s, 41.95 tok/s |
| 3 | 19 | 0.157 s, 121.13 tok/s | 512 | 12.296 s, 41.64 tok/s |

In another run (`--temp 0`, 96 tokens per reply) a turn of 179 prompt tokens took
1.383 s, 129.42 tok/s. Backend / GPU: Metal / Intel(R) Iris(TM) Plus Graphics 640.
One run each, not averages; decode speed varied between 31 and 43 tok/s across the
runs made for this page.

```sh
printf '%s\n%s\n%s\n/exit\n' \
  'Write a long story about a dragon who learns to write code, with several chapters and lots of detail.' \
  'Continue the story with a long, detailed second chapter.' \
  'Now explain the moral of the whole story in detail.' |
BYBY_METAL_JIT_ARCHIVE_SAVE=0 BYBY_METAL_JIT_LOG=0 \
other/formers.n/formers-cli --models test/mamba3/models \
  -m test/mamba3/models/mamba3_siso_187m.nimq \
  --config test/mamba3/models/mamba3-siso-187m/config.json \
  --tokenizer test/mamba3/models/mamba3-siso-187m \
  --lora test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
  -cnv -n 512 --temp 0.7 -s 7 --backend metal --no-progress
```

#### Lifted graph (`formers-cli --fir`)

The same checkpoint with the LoRA merged, recorded once from the native chat
path and lifted into a standalone `model.fir` (recording and lifting:
[`docs/fir-script-lift.md`](../../docs/fir-script-lift.md)), against the native
runtime on one prompt of 171 tokens (BOS plus the 17-sentence text of the
scheduler table above, no chat template, 64 greedy tokens). Six runs back to
back in the order below (native, graph, graph, native, native, graph), so the
GPU state drifts the same way for both paths. Every speed leaves out the cold
start: the native tool prints its kernel setup/JIT separately, and the graph
runs the prompt and two decode steps once (`ready: warmup`) before the timed
run. The graph prefill counts all 171 prompt tokens (the last one is its decode
step with logits); the native prefill counts the first 170 and times the last
token as its first decode step.

| Run | Path | Prefill | Decode (63 later tokens) |
|---:|---|---:|---:|
| 1 | native | 1.770 s, 96.06 tok/s | 31.97 tok/s (31.3 ms/token) |
| 2 | graph | 1.522 s, 112.32 tok/s | 43.05 tok/s (23.2 ms/token) |
| 3 | graph | 1.426 s, 119.89 tok/s | 35.86 tok/s (27.9 ms/token) |
| 4 | native | 1.648 s, 103.18 tok/s | 35.99 tok/s (27.8 ms/token) |
| 5 | native | 1.533 s, 110.89 tok/s | 35.48 tok/s (28.2 ms/token) |
| 6 | graph | 1.397 s, 122.37 tok/s | 35.98 tok/s (27.8 ms/token) |

Median of three: native 103.18 tok/s prefill and 35.48 tok/s decode, graph
119.89 tok/s prefill and 35.98 tok/s decode. Both paths generated the same
64 tokens (290 bytes of text) in every run. Backend / GPU: Metal /
Intel(R) Iris(TM) Plus Graphics 640.

The graph is not slower than the native runtime. Decode agrees within the
spread of this GPU (31-43 tok/s across the runs made for this page). The three
graph prefills (112-122 tok/s) were faster than the three native ones
(96-111 tok/s); the two paths schedule the prompt differently: the native
one-shot path prefills in micro-batches of up to 128 rows with command buffers
of about 300 ms, while the graph feeds the 128-, 32- and 8-row chunks of the
chat scheduler (its three phases, the chunks behind the chat prompt eval above)
and the last token as a decode step. One process per run, three runs per path,
not averages.

```sh
P=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' {1..17}); P="${P% }"
# native
BYBY_METAL_JIT_ARCHIVE_SAVE=0 BYBY_METAL_JIT_LOG=0 \
other/formers.n/formers-cli --models test/mamba3/models \
  -m test/mamba3/models/mamba3_siso_187m.nimq \
  --config test/mamba3/models/mamba3-siso-187m/config.json \
  --tokenizer test/mamba3/models/mamba3-siso-187m \
  --lora test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
  --raw -p "$P" -n 64 --temp 0 --backend metal --no-progress -v
# lifted graph ($OUT is the directory the lifter wrote)
BYBY_METAL_JIT_ARCHIVE_SAVE=0 BYBY_METAL_JIT_LOG=0 \
other/formers.n/formers-cli --models test/mamba3/models/mamba3-siso-187m \
  -m test/mamba3/models/mamba3_siso_187m.nimq --fir "$OUT/model.fir" \
  --raw --add-special-tokens -p "$P" -n 64 --temp 0 --backend metal --no-progress
```

`-v` makes the native run print its `== Speed stats ==` block; the graph run
prints its own speed lines without it.

### Tiếng Việt

#### Intel Iris Plus Graphics 640

Đo trên backend Metal với Mamba-3 SISO 187M và checkpoint đã finetune
`RRRRRRRRRRRG` (`lora.bin`). Mọi tốc độ trên trang này không tính setup/JIT kernel
một lần của process (khởi động lạnh); các công cụ đo và in riêng phần đó.

| Chỉ số | Kết quả |
|---|---:|
| Prompt | `What is the capital of France?` |
| Số token prompt | 13 |
| Số token sinh | 32 |
| Prefill | 0.159 giây, 81.78 tok/s |
| Decode | 0.764 giây, 41.88 tok/s (23.9 ms/token) |
| Decode, token 2-32 | 42.34 tok/s (23.6 ms/token) |
| Prefill + decode | 0.923 giây cho 45 token |
| Backend / GPU | Metal / Intel(R) Iris(TM) Plus Graphics 640 |

Lượt chạy dùng greedy decoding (`temperature=0`), các adapter LoRA được gộp tại
chỗ vào base đã lượng tử hóa (mặc định). Một lượt, không lấy trung bình. Setup/JIT
kernel mà `infer_lora_bin` in ở dòng riêng là chi phí khởi động lạnh, khoảng
0.6-1.0 giây với prompt này và thay đổi giữa các lượt; prompt có độ dài mới phải
biên dịch bộ kernel prefill riêng nên prompt dài tốn hơn (khoảng 2-3 giây với các
prompt dài đã thử). Với `BYBY_M3_PROFILE=1`, mỗi bước decode được tách thành host
encode (khoảng 1 ms), chờ GPU (trung bình 21.9 ms, khoảng 17.3-31.9 ms) và đọc
logits (khoảng 0.2 ms): decode trên GPU này bị GPU chặn, không phải host.

```sh
BYBY_MAX_NEW_TOKENS=32 \
BYBY_METAL_JIT_ARCHIVE_SAVE=0 \
./test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
  'What is the capital of France?' \
  0
```

`BYBY_METAL_JIT_ARCHIVE_SAVE=0` bỏ bước lưu JIT archive lúc thoát, bước này có thể
mất hàng phút trên Intel. Binary ghi nhận 48 LoRA adapter và topology
`RRRRRRRRRRRG`.

#### Bộ lập lịch prefill

Prompt đi qua prefill theo micro-batch tối đa 128 dòng (`BYBY_UBATCH`); trên GPU
này mỗi micro-batch còn được chia thành các command buffer khoảng 300 ms
(`BYBY_PREFILL_CB_MS`), cỡ theo thời gian đo được của mỗi layer.
`BYBY_PREFILL_BENCH=4` đo cùng một prompt dưới từng chế độ lập lịch trong một
process (một vòng khởi động không tính giờ, các chế độ xen kẽ, trung vị của 4
lượt), nên số liệu không có setup/JIT kernel và không có nhiễu giữa các process.
Prompt 176 token (câu `The quick brown fox jumps over the lazy dog. ` lặp 17
lần):

| Chế độ | Trung vị | Tốt nhất | Tốc độ (trung vị) |
|---|---:|---:|---:|
| Một command buffer cho mỗi micro-batch | 1.721 giây | 1.389 giây | 102.3 tok/s |
| Commit sau mỗi 4 layer (chế độ của GPU Apple) | 1.668 giây | 1.367 giây | 105.5 tok/s |
| Command buffer khoảng 300 ms (mặc định trên GPU này) | 1.814 giây | 1.488 giây | 97.0 tok/s |

Với model nhỏ này, việc chia command buffer làm prefill chậm hơn khoảng 5-7% trên
GPU này; mục đích của nó là giữ mọi command buffer ngắn cho model lớn hơn, không
phải tốc độ. Chế độ commit sau mỗi 4 layer là mặc định của GPU Apple; ở đây chỉ đo
để so sánh và chưa đo trên GPU Apple.

```sh
P=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' {1..17})
BYBY_PREFILL_BENCH=4 BYBY_MAX_NEW_TOKENS=1 \
BYBY_METAL_JIT_ARCHIVE_SAVE=0 \
./test/mamba3/infer_lora_bin \
  test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin "$P" 0
```

#### Chat (`formers-cli -cnv`)

Đo trên cùng GPU và backend ở chế độ conversation của formers.n: ba lượt trong một
phiên, tối đa 512 token mỗi câu trả lời, `--temp 0.7`, chat template lấy từ thư mục
model. `ready: warmup` đã biên dịch kernel prefill và decode trước prompt đầu tiên
(3.8 giây ở lượt này), nên các tốc độ này không có setup/JIT kernel. Mỗi lượt nạp
prompt qua bộ lập lịch prefill.

| Lượt | Token prompt | Prompt eval | Token sinh | Decode |
|---:|---:|---:|---:|---:|
| 1 | 27 | 0.216 giây, 125.07 tok/s | 512 | 12.138 giây, 42.18 tok/s |
| 2 | 19 | 0.157 giây, 121.16 tok/s | 512 | 12.204 giây, 41.95 tok/s |
| 3 | 19 | 0.157 giây, 121.13 tok/s | 512 | 12.296 giây, 41.64 tok/s |

Ở một lượt khác (`--temp 0`, 96 token mỗi câu trả lời), một lượt có 179 token
prompt mất 1.383 giây, 129.42 tok/s. Backend / GPU: Metal / Intel(R) Iris(TM) Plus
Graphics 640. Mỗi mục là một lượt, không lấy trung bình; tốc độ decode dao động
31-43 tok/s giữa các lượt chạy làm cho trang này.

```sh
printf '%s\n%s\n%s\n/exit\n' \
  'Write a long story about a dragon who learns to write code, with several chapters and lots of detail.' \
  'Continue the story with a long, detailed second chapter.' \
  'Now explain the moral of the whole story in detail.' |
BYBY_METAL_JIT_ARCHIVE_SAVE=0 BYBY_METAL_JIT_LOG=0 \
other/formers.n/formers-cli --models test/mamba3/models \
  -m test/mamba3/models/mamba3_siso_187m.nimq \
  --config test/mamba3/models/mamba3-siso-187m/config.json \
  --tokenizer test/mamba3/models/mamba3-siso-187m \
  --lora test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
  -cnv -n 512 --temp 0.7 -s 7 --backend metal --no-progress
```

#### Graph đã lift (`formers-cli --fir`)

Cùng checkpoint đã gộp LoRA, ghi lại một lần từ đường chat native và lift thành
`model.fir` độc lập (cách ghi và lift:
[`docs/fir-script-lift.md`](../../docs/fir-script-lift.md)), so với runtime native
trên một prompt 171 token (BOS cộng đoạn 17 câu của bảng bộ lập lịch ở trên,
không dùng chat template, sinh 64 token greedy). Sáu lượt chạy liên tiếp theo thứ
tự trong bảng (native, graph, graph, native, native, graph) để trạng thái GPU
trôi như nhau cho hai đường. Mọi tốc độ không tính khởi động lạnh: công cụ native
in riêng phần setup/JIT kernel, còn graph chạy prompt và hai bước decode một lần
(`ready: warmup`) trước lượt có tính giờ. Prefill của graph tính cả 171 token
prompt (token cuối là bước decode có logits); prefill của native tính 170 token
đầu và tính giờ token cuối như bước decode đầu tiên.

| Lượt | Đường chạy | Prefill | Decode (63 token sau) |
|---:|---|---:|---:|
| 1 | native | 1.770 giây, 96.06 tok/s | 31.97 tok/s (31.3 ms/token) |
| 2 | graph | 1.522 giây, 112.32 tok/s | 43.05 tok/s (23.2 ms/token) |
| 3 | graph | 1.426 giây, 119.89 tok/s | 35.86 tok/s (27.9 ms/token) |
| 4 | native | 1.648 giây, 103.18 tok/s | 35.99 tok/s (27.8 ms/token) |
| 5 | native | 1.533 giây, 110.89 tok/s | 35.48 tok/s (28.2 ms/token) |
| 6 | graph | 1.397 giây, 122.37 tok/s | 35.98 tok/s (27.8 ms/token) |

Trung vị của ba lượt: native 103.18 tok/s prefill và 35.48 tok/s decode, graph
119.89 tok/s prefill và 35.98 tok/s decode. Cả hai đường sinh cùng 64 token
(290 byte văn bản) ở mọi lượt. Backend / GPU: Metal / Intel(R) Iris(TM) Plus
Graphics 640.

Graph không chậm hơn runtime native. Decode khớp nhau trong biên dao động của GPU
này (31-43 tok/s giữa các lượt chạy làm cho trang này). Ba lượt prefill của graph
(112-122 tok/s) nhanh hơn ba lượt của native (96-111 tok/s); hai đường lập lịch
prompt khác nhau: đường one-shot native prefill theo micro-batch tối đa 128 dòng
với command buffer khoảng 300 ms, còn graph nạp các chunk 128, 32 và 8 dòng của bộ
lập lịch chat (ba phase của nó, cũng là các chunk của prompt eval chat ở trên) và
token cuối như một bước decode. Mỗi lượt là một process, mỗi đường ba lượt, không
lấy trung bình.

```sh
P=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' {1..17}); P="${P% }"
# native
BYBY_METAL_JIT_ARCHIVE_SAVE=0 BYBY_METAL_JIT_LOG=0 \
other/formers.n/formers-cli --models test/mamba3/models \
  -m test/mamba3/models/mamba3_siso_187m.nimq \
  --config test/mamba3/models/mamba3-siso-187m/config.json \
  --tokenizer test/mamba3/models/mamba3-siso-187m \
  --lora test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin \
  --raw -p "$P" -n 64 --temp 0 --backend metal --no-progress -v
# graph đã lift ($OUT là thư mục lifter đã ghi)
BYBY_METAL_JIT_ARCHIVE_SAVE=0 BYBY_METAL_JIT_LOG=0 \
other/formers.n/formers-cli --models test/mamba3/models/mamba3-siso-187m \
  -m test/mamba3/models/mamba3_siso_187m.nimq --fir "$OUT/model.fir" \
  --raw --add-special-tokens -p "$P" -n 64 --temp 0 --backend metal --no-progress
```

`-v` làm lượt native in khối `== Speed stats ==`; lượt graph tự in các dòng tốc độ
của nó mà không cần `-v`.

## Mamba-3 SISO 1.5B

### English

On Intel Iris Plus Graphics 640 / Metal,
using the FP4-KM 1.5B base and the 96-adapter LoRA
checkpoint `runs/mamba3-1.5b/lora-g1.bin`, merged in place (23 R + G). Prompt:
`Write a Python function to check if a string is a palindrome`. Temperature
was 0.7. The prompt is 19 tokens including the final token; prefill measured
18 tokens.

- Prefill: 18 tokens in 1.783 s, 10.10 tok/s (the 0.039 s of kernel setup/JIT
  is not counted).
- Decode: 64 tokens in 13.066 s, 4.90 tok/s (204.2 ms/token); the 0.018 s of
  setup/JIT in the first token is not counted. The 63 tokens after the first:
  4.90 tok/s (203.9 ms/token).
- Total: 82 tokens in 14.906 s of wall time, setup included.
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

- Prefill: 18 token trong 1.783 giây, 10.10 tok/s (không tính 0.039 giây
  setup/JIT kernel).
- Decode: 64 token trong 13.066 giây, 4.90 tok/s (204.2 ms/token); không tính
  0.018 giây setup/JIT ở token đầu. 63 token sau token đầu: 4.90 tok/s
  (203.9 ms/token).
- Tổng: 82 token trong 14.906 giây wall time, đã gồm setup.
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

