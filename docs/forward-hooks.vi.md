# Forward hook trong Nim

[English](forward-hooks.md) | [Tiếng Việt](forward-hooks.vi.md)

`nimformers` hỗ trợ hook trước và sau cho các API `forward` cấp model. Đăng ký
hook cho một API cụ thể hoặc dùng `ForwardApiAll` (`"*"`) để áp dụng cho tất cả.

```nim
import nimformers

proc addInputOffset(ctx: Backend; input: Tensor): Tensor =
  result = input
  for value in result.data.mitems:
    value += 0.1'f32

proc scaleOutput(ctx: Backend; input, output: Tensor): Tensor =
  result = output
  for value in result.data.mitems:
    value *= 0.5'f32

proc main() =
  let hookId = registerForwardHook("linear", before = addInputOffset,
    after = scaleOutput)
  defer: discard removeForwardHook(hookId)
  echo forwardApiNames()

main()
```

Callback `before(ctx, input)` trả về tensor đầu vào được dùng trong forward.
Callback `after(ctx, input, output)` trả về tensor đầu ra cuối cùng. Hook chạy
theo thứ tự đăng ký; forward lồng nhau cũng gọi hook đã đăng ký cho từng API
thành phần. `forwardApiNames()` trả về các selector được hỗ trợ: `linear`,
`layer_norm`, `causal_self_attention`, `feed_forward`, `transformer_block`,
`embedding`, `nimformer_model_batch` và `nimformer_model`.

Dùng `registerForwardHook("*")` hoặc
`registerForwardHook(ForwardApiAll, ...)` để áp dụng mọi selector. Hook được
đăng ký ở phạm vi toàn process. Hãy đăng ký trước khi inference và tránh thay
đổi danh sách hook khi forward đang chạy. `Linear.preForwardHook` cũ vẫn hoạt
động và chạy sau hook `before` đã đăng ký cho `linear`.

Callback dùng calling convention `nimcall`, vì vậy cần khai báo dưới dạng proc
thường, không capture biến từ closure. Hook không có callback backward tương
ứng; nếu thay đổi output làm thay đổi gradient thì cần tự cài backward phù hợp
trước khi dùng trong quá trình training.
