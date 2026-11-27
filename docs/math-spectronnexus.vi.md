# Math SpectronNexus

[English](math-spectronnexus.md) | [Tiếng Việt](math-spectronnexus.vi.md)

Math SpectronNexus là bộ giải toán rời rạc thử nghiệm trong
`test/math_spectronnexus`. Một chương trình Nim native cung cấp các phép tính
qua CLI và máy chủ Model Context Protocol (MCP) chạy cục bộ. Máy chủ công bố
21 tool. Tùy phép toán, kết quả có thể gồm đáp số chính xác, thông tin kiểm tra,
chứng cứ hoặc chuỗi tool đã chạy.

## Các khả năng

- **Số học chính xác:** số nguyên và phân số lớn tùy ý, biểu thức, ma trận,
  định thức, căn, giai thừa, hệ số nhị thức và hàm số học.
- **Dãy số:** tính số hạng, thử công thức, tìm đa thức, truy hồi tuyến tính hoặc
  holonomic, hàm sinh đại số, tính nhân và phần dư tuần hoàn. Phép tìm kiếm còn
  xét các dãy biến đổi như sai phân và tổng riêng.
- **Số hạng xa:** `research_sequence` lập kế hoạch trả lời tại chỉ số lớn. Nó
  có thể yêu cầu thêm số hạng từ generator, khám phá cấu trúc, tính trực tiếp
  theo modulo hoặc các lũy thừa nguyên tố của modulo, ghép CRT, hoặc dừng khi
  không tìm được đường tính. Kết quả có `answer`, `status`, `direction` và
  `steps`.
- **Trường hữu hạn và đa thức:** hạng ma trận, hệ tuyến tính, ước chung lớn
  nhất và phân tích đa thức, ghép số dư Trung Hoa.
- **Tổ hợp và đồ thị:** số thứ tự tuyến tính của poset, luồng cực đại kèm lát
  cắt nhỏ nhất và số đường đi có trọng số.
- **Nhận dạng và tra cứu:** OEIS ngoại tuyến, nhận dạng hằng số từ chữ số,
  tính chữ số hằng số độ chính xác cao và tìm định lý đã chứng minh trong Lean
  Mathlib.

[README SpectronNexus](../test/math_spectronnexus/README.md) liệt kê đầy đủ
các tool, schema đầu vào, giới hạn, ví dụ CLI, lệnh train và benchmark.
Quy trình dùng với mô hình ngôn ngữ và hướng dẫn chi tiết từng tool nằm trong
[SKILL.md](../test/math_spectronnexus/SKILL.md).

## Quy trình xử lý dãy

Dùng `calculate` để tạo các trường hợp nhỏ. Gửi ít nhất tám số hạng cùng chỉ
số bắt đầu vào `explore_sequence` để thử giả thuyết và tìm cấu trúc. Dùng
`research_sequence` khi cần tính tại chỉ số lớn. Nếu có generator, truyền
`generator`, `start` và `count`. Nếu có sẵn số hạng, truyền `terms` và `offset`.
Đặt `index` cần tính và `modulus` tùy chọn bên trong `question`.

Hãy xem `status` trước khi trình bày kết quả số hạng xa. Cụ thể,
`extrapolation_from_conjectured_direction` nghĩa là solver đã tính theo quy
luật khớp từ các số hạng đã cho và kiểm tra trên các số hạng chưa dùng khi
khớp. Đây là bằng chứng thực nghiệm, chưa phải chứng minh toán học. Cần nêu rõ
giới hạn này trừ khi quy luật đã được chứng minh độc lập. `no_answer` nghĩa là
các đường tính hiện có chưa tìm ra kết quả.

## Chạy cục bộ

Build và chạy máy chủ MCP từ thư mục gốc repository:

```sh
make -C test/math_spectronnexus build
NEXUS="$PWD/test/math_spectronnexus/nimcache/math_spectronnexus"
"$NEXUS" mcp --backend=metal
```

Đăng ký cho Claude Code trong checkout hiện tại bằng lệnh:

```sh
claude mcp add --transport stdio --scope local math-spectronnexus -- \
  "$PWD/test/math_spectronnexus/nimcache/math_spectronnexus" mcp --backend=metal \
  "--program-checkpoint=$PWD/test/math_spectronnexus/models/sequence_program_curriculum.nimq" \
  "--explorer-checkpoint=$PWD/test/math_spectronnexus/models/sequence_explorer.nimq" \
  "--planner-checkpoint=$PWD/test/math_spectronnexus/models/research_planner.nimq"
```

Skill cho repo có tên `/math-spectronnexus`. Kiểm tra kết nối trong Claude Code
bằng `/mcp`. Cấu hình MCP và ý nghĩa các checkpoint cũng có trong
[mục local MCP](../test/math_spectronnexus/README.md#local-mcp).

## Trạng thái thử nghiệm

Solver, sequence explorer và planner được học là mã nghiên cứu thử nghiệm.
Tỷ lệ benchmark chỉ phản ánh các bộ kiểm tra đã nêu, không đảm bảo kết quả cho
mọi đầu vào. Không bắt buộc train mới dùng được tool. Khả năng và giới hạn
chính xác tùy từng phép toán. Xem README đã liên kết để biết giới hạn hiện tại.
