# Thực thi song song trên nhiều thiết bị

[English](distributed.md) | [Tiếng Việt](distributed.vi.md)

Bộ điều phối native của BybyLang giao công việc cho các tiến trình con trên
cùng một máy. Nim và `.bybylang` gọi trực tiếp qua `nimformers`. Python gọi
cùng bộ điều phối qua shared library. Trên macOS và Linux, mỗi thiết bị đang
làm việc có một tiến trình riêng tại một thời điểm.

## Chọn thiết bị và tỷ lệ chia

Dùng `cpu`, `vulkan[:ordinal]` hoặc `metal[:ordinal]`. Tên GPU thiếu ordinal
sẽ chọn thiết bị 0 theo thứ tự của backend đó. Chọn một mục cho mỗi thiết bị
vật lý bạn muốn sử dụng.

```nim
import nimformers

let group = initProcessGroup(["cpu", "vulkan:0"], [1.0, 8.0])
echo group.partition(90)  # @[10, 80]
```

Script BybyLang import cùng module Nim.

```bybylang
import nim "nimformers"

group = nimformers.initProcessGroup(@["cpu", "vulkan:0"], @[1.0, 8.0])
print nimformers.partition(group, 90)
```

`runCommands` chia các lệnh độc lập thành các
lát liền nhau và trả stdout, stderr theo thứ tự. `runShards` tạo một lệnh
native cho mỗi lát có dữ liệu. Bộ điều phối gán thiết bị qua biến môi trường
trước khi tiến trình con khởi tạo backend.

Tỷ lệ cấu hình được giữ cố định. Mỗi lần gọi có thể đặt tỷ lệ riêng và đặt 0
để thiết bị đó chờ. Chế độ tự động khởi đầu với GPU:CPU là 8:1 rồi điều chỉnh
qua các lần chạy theo thời gian đo được cho từng workload trong group. Trọng
số CPU tối đa bằng một nửa trọng số GPU nhỏ nhất. `partition` xem trước số
item theo tỷ lệ chỉ định, đã cấu hình hoặc mặc định ban đầu. Với batch ngắn,
sau khi làm tròn có thể có thiết bị không nhận item.

## Chạy batch micro-kernel

```nim
import nimformers

let group = initProcessGroup(["cpu"])
var prog: VmProgram
prog.emitLoadF32(1, 0, 0)
prog.emitAdd(1, 1, 1)
prog.emitStoreF32(1, 0, 0)
prog.emitHalt()
let output = runVmParallel(group, prog.words, @[1'f32, 2, 3, 4],
                           itemSize = 1, threadsPerItem = 1, tgSize = 1)
echo output  # @[2.0, 4.0, 6.0, 8.0]
```

`make build-shared` hoặc `nimble buildShared` build `nimformers-worker`
và shared library. Đặt worker trên PATH, cạnh executable hoặc trong thư mục
hiện tại. Dùng `NIMFORMERS_WORKER` hoặc `workerPath` để chọn đường dẫn cụ thể.

VM nhận một buffer riêng trên từng thiết bị và chỉ số trong mỗi lát bắt đầu
từ 0. Hãy dùng các item hoặc threadgroup độc lập. Với `tgSize > 0`, số item
phải chia hết cho `tgSize / gcd(tgSize, threadsPerItem)` để giữ nguyên ranh
giới threadgroup. Với `tgSize = 0`, dùng chương trình không phụ thuộc vào
cách backend chọn threadgroup.

Python dùng `init_process_group(devices, ratios=None)` từ
`nimformers.distributed`. `group.vm_run(program, memory, threads_per_item=None,
tg_floats=0, tg_size=0, ratios=None)` chia theo trục đầu tiên và trả mảng mới
theo thứ tự ban đầu. Mặc định số thread mỗi item bằng số phần tử float32
trong item đó.

## Chạy batch prompt FIR độc lập

Nim và BybyLang gọi `runFormersBatch` với group, executable `formers-cli`,
tham số model của CLI và các prompt. `formers.Model.generate_batch(...,
group=group)` và `generate_ids_batch` trong Python gọi cùng API native.
[Python API của formers.n](../other/formers.n/README.md#python-api)

Trong Nim, `runFormersBatch` trả stdout của từng prompt theo thứ tự đầu vào.
`runFormersBatchDetailed` trả `ParallelResult` gồm cả `stdout` và `stderr`
theo cùng thứ tự, để CLI hoặc ứng dụng có thể hiện log/throughput của từng
worker khi cần.

Mỗi prompt chạy trọn vẹn trên một thiết bị, sau đó kết quả được trả theo thứ
tự đầu vào. Với graph FIR, truyền `--fir model.fir` cùng các tùy chọn config,
weights và tokenizer của model. Thiết bị nhận prompt cần đủ bộ nhớ chứa model
và hỗ trợ kernel của graph. Codestral FIR resident hiện cần Vulkan. Cơ chế
này phân phối các prompt hoặc dữ liệu VM độc lập trên cùng một máy.

### Chạy batch trực tiếp từ `formers-cli`

Lặp `-p` hoặc `-f` để đưa nhiều prompt vào scheduler native:

```sh
other/formers.n/formers-cli --models test/mamba3/models \
  -m test/mamba3/models/mamba3_siso_187m.nimq \
  --config test/mamba3/models/mamba3-siso-187m/config.json \
  --tokenizer test/mamba3/models/mamba3-siso-187m --no-lora \
  --distributed cpu,metal:0 --distributed-ratios 1,1 \
  -p 'Write one sentence about the ocean.' \
  -p 'Explain why the sky appears blue.' \
  -n 24 --temp 0 --repeat-penalty 1.0 --no-progress
```

CLI in số prompt được gán cho mỗi worker và in kết quả theo thứ tự đầu vào.
Để đối chiếu token, chạy từng prompt trực tiếp với `--backend cpu` hoặc
`--backend metal` cùng model và tham số sinh giống hệt. Mỗi worker vẫn nạp
toàn model; API này phân phối batch prompt độc lập.
