# Hook API trong Nim

[English](forward-hooks.md) | [Tiếng Việt](forward-hooks.vi.md)

Hook của Nimformers bao phủ bốn mặt: giá trị Tensor/model, tham số và kết quả
có kiểu của API Nim đã instrument, sự kiện quan sát trước/sau trên framework
và BybyLang, cùng khả năng thay bytecode micro-kernel đã sinh.

## Thay đổi lời gọi Tensor và model

`registerApiHook` cài hook ở phạm vi process cho model forward và các phép toán
trả về Tensor, kể cả lời gọi lồng nhau:

```nim
import nimformers

proc addInputOffset(api: string; ctx: Backend; inputs: var seq[Tensor]) =
  for input in inputs.mitems:
    for value in input.data.mitems:
      value += 0.1'f32

proc scaleOutput(api: string; ctx: Backend; inputs: openArray[Tensor];
    output: Tensor): Tensor =
  result = output
  for value in result.data.mitems:
    value *= 0.5'f32

let hookId = registerApiHook("addT", before = addInputOffset,
  after = scaleOutput)
defer: discard removeApiHook(hookId)
```

Callback `before` nhận tên API, backend và toàn bộ Tensor đầu vào. Callback có
thể thay hoặc sửa từng Tensor tại chỗ nhưng phải giữ nguyên số lượng đầu vào.
Callback `after` nhận đầu vào và trả về Tensor đầu ra mà nơi gọi sẽ sử dụng.
Các callback dùng calling convention `closure` của Nim nên có thể capture
state. Tên selector không phân biệt chữ hoa/chữ thường.

Dùng `registerApiHook("*")` để hook mọi tên trong `hookApiNames()`. Mỗi lời gọi
lồng nhau có callback riêng. `registerForwardHook` vẫn hỗ trợ callback cũ dạng
`(ctx, input)` / `(ctx, input, output)`; wildcard cũ chỉ chọn forward của model.
`Linear.preForwardHook` cũng tiếp tục hoạt động.

Hook chạy theo thứ tự đăng ký và có phạm vi toàn process. Hãy đăng ký trước khi
chạy inference song song và không sửa registry trong lúc API đang chạy. Hook
không cung cấp gradient backward; nếu hook forward đổi giá trị ảnh hưởng đến
training, cần tự cài backward phù hợp.

## Quan sát API framework và BybyLang

`registerBybyApiHook` quan sát API public trong framework, gồm model/Tensor,
backend, BybyLang session, trình tạo bytecode micro-VM, JIT/lowering và
 distributed helper. Truyền tên proc hoặc `"*"` để nhận sự kiện trước/sau cho
mọi API đã instrument:

```nim
import nimformers

proc inspectApi(api: string; phase: BybyApiHookPhase; metadata: string) =
  echo phase, " ", api, " ", metadata

proc main() =
  let hookId = registerBybyApiHook("*", inspectApi)
  defer: discard removeBybyApiHook(hookId)

main()
```

Metadata tóm tắt scalar và độ dài collection. Sự kiện sau có thêm tóm tắt
output và tham số `var` đã đổi; tensor lớn và device buffer không bị copy vào
callback. Hook này quan sát lời gọi. Dùng `registerBybyApiValueHook` để sửa
tham số và giá trị trả về có kiểu cụ thể ở tầng BybyLang:

```nim
import nimformers

proc rewriteRead(api: string; phase: BybyApiHookPhase;
    call: var BybyApiCall) =
  case phase
  of bybyApiBefore:
    if api == "sessionRead":
      apiArg[int](call.args[2]) = 2
  of bybyApiAfter:
    if api == "sessionRead":
      apiArg[seq[float32]](call.output)[0] = 42'f32

proc main() =
  let hookId = registerBybyApiValueHook("sessionRead", rewriteRead)
  defer: discard removeBybyApiValueHook(hookId)
  var buffer = sessionUpload(gbCpu, @[1'f32, 2'f32])
  defer: sessionFree(gbCpu, buffer)
  let values = sessionRead(gbCpu, buffer, 1)
  doAssert values == @[42'f32, 2'f32]

main()
```

Dùng `apiArg[T]` với đúng kiểu Nim ghi trong `BybyApiArg.typeName` để đọc/sửa
tham số ở pha before và giá trị trả về ở pha after. `call.output` chỉ sẵn sàng
ở pha after. Selector `"*"` áp dụng một callback cho mọi public API Nim đã
instrument; hãy kiểm tra `api` và `typeName` trước khi gọi `apiArg[T]` vì kiểu
tham số thay đổi theo API. Tham số `var` được chép lại vào caller sau hook.
Hook chạy theo thứ tự đăng ký và nên được cài trước inference song song. API
Nim trả `var T` vẫn quan sát được, nhưng hook typed không thay alias trả về để
không phá hợp đồng alias của Nim. `registerApiHook` vẫn là đường chuyên cho
Tensor; micro-kernel hook bên dưới dùng để thay phép tính đã sinh.

## Hook procedure trong module Nim của bạn

Import `nimformers` rồi thêm `{.apiHooked.}` vào từng procedure runtime muốn
đưa vào cùng registry. Nhờ vậy thư viện có thể mở API cho hook mà không cần tự
viết phần callback:

```nim
import nimformers

proc scaleValue*(value: float32): float32 {.apiHooked.} =
  value * 2'f32
```

`registerBybyApiValueHook("*", callback)` sẽ nhận procedure này cùng các API
của framework. Macro chạy lúc biên dịch, template và khai báo C import thô
không có thân Nim runtime để bọc; hãy thêm một wrapper Nim quanh hàm import khi
cần hook.

### Import hook plugin

Plugin có thể là một module Nim thông thường được compile cùng ứng dụng host.
Module có thể đưa proc của nó vào registry bằng `{.apiHooked.}` rồi cài
callback cho API riêng hoặc API framework. Plugin mẫu còn sửa lời gọi
BybyLang cấp thấp `sessionRead`:

```sh
make hook-plugin-test
```

Xem [`token_hook_plugin.nim`](../examples/nim/plugins/token_hook_plugin.nim)
và [`test_hook_plugin.nim`](../test/hooks/test_hook_plugin.nim). Test kiểm tra
hook input và kết quả của API thuộc plugin, sau đó đổi số phần tử đọc và dữ
liệu trả về của `sessionRead`. Đây là plugin module Nim chạy trong cùng
process với host; nó chưa dùng ABI plugin riêng qua `dlopen`/`LoadLibrary`.

Plugin export `install()` và `uninstall(id)`. Khi cài, nó đăng ký typed hook
wildcard; callback lọc tên API trước khi truy cập args theo kiểu. Với
`pluginScale(3)`, hook đổi input thành `4`, proc plugin nhân đôi thành `8`, rồi
hook đổi kết quả thành `18`. Với `sessionRead`, hook tăng số phần tử yêu cầu từ
1 lên 2 rồi sửa giá trị đầu tiên trả về. Import module vào host và cài trước
khi gọi API:

```nim
import nimformers
import token_hook_plugin

let pluginHook = install()
defer: discard uninstall(pluginHook)
doAssert pluginScale(3'f32) == 18'f32
```

Registry hook dùng chung trong toàn process. Hãy gỡ hook ID khi plugin không
còn dùng hoặc được unload; không thêm/gỡ hook trong lúc thread khác đang chạy
các API đã hook.

## Thay micro-kernel BybyLang đã sinh

Có thể viết lại micro-kernel trước khi chạy CPU hoặc hạ xuống GPU, kể cả
program đã tạo và sắp được đưa vào JIT hoặc dispatch. Chọn theo tag kernel;
kernel trong DSL dùng tag `bybylang_<name>`. Gán `VmProgram` khác trong callback
để thay bytecode:

```nim
import nimformers

var replacementProgram: VmProgram # Tạo bằng API emit của session_vm.

proc useReplacement(tag: string; program: var VmProgram) =
  if tag == "bybylang_scale":
    program = replacementProgram

proc main() =
  let hookId = registerMicroKernelHook("bybylang_scale", useReplacement)
  defer: discard removeMicroKernelHook(hookId)
  # Gọi code sử dụng kernel "bybylang_scale" ở đây.

main()
```

Kernel thay thế phải giữ nguyên layout buffer, output size, launch dimensions
và tập instruction mà backend hỗ trợ. Hook chạy trước JIT compilation và
dispatch, nên bytecode thay thế là chương trình được compile và thực thi.
Callback micro-kernel dùng calling convention closure của Nim.
Selector "*" thay mọi micro-kernel đã gắn tag; hãy dùng tag cụ thể nếu kernel
thay thế chỉ phù hợp với một kernel. VmProgram đã được viết lại cho một tag
không bị viết lại lần nữa khi đi qua lowering và dispatch.
