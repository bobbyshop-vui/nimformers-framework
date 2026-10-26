#!/bin/bash
# Benchmark A/B xen kẽ: KHÔNG sparsity (A) vs CÓ sparsity (B), lặp lại N lần,
# xen kẽ thứ tự A/B/A/B/... để nhiệt độ máy trôi dần đều ảnh hưởng NGANG NHAU
# lên cả 2 nhóm (thay vì chạy hết A rồi mới chạy B - lúc đó B luôn "thiệt" vì
# máy đã nóng sẵn từ nhóm A, hoặc ngược lại nếu A chạy trước lúc máy mát).
#
# Cách dùng:
#   chmod +x bench_sparsity_ab.sh
#   ./bench_sparsity_ab.sh
#
# Sau khi chạy xong, dán TOÀN BỘ output cho Claude - đủ dữ liệu để tính
# median/range mỗi nhóm, tách "cải thiện thật do sparsity" khỏi "nhiễu do
# tải máy/nhiệt độ".

N_RUNS=4   # số lần lặp MỖI nhóm (A chạy N_RUNS lần, B chạy N_RUNS lần, xen kẽ)
PROMPT="Write a Python function to return the n-th Fibonacci number."
MAX_TOK=20

echo "=== BENCHMARK A/B XEN KE - $N_RUNS lan/nhom ==="
echo ""

for i in $(seq 1 $N_RUNS); do
  echo "### [A - KHONG sparsity] lan $i/$N_RUNS ###"
  BYBY_MERGE_ALL_LAYERS_SESSION=1 BYBY_MERGE_QKV_ATTN_SESSION=1 \
    ./nim_inference "" "" "" "$PROMPT" $MAX_TOK 2>&1 | grep -E "Prefill:|Decode:|MODE3|Tong:"
  echo ""

  echo "### [B - CO sparsity 20%] lan $i/$N_RUNS ###"
  BYBY_SPARSITY_PCT=20 BYBY_MERGE_ALL_LAYERS_SESSION=1 BYBY_MERGE_QKV_ATTN_SESSION=1 \
    ./nim_inference "" "" "" "$PROMPT" $MAX_TOK 2>&1 | grep -E "Prefill:|Decode:|MODE3|Tong:"
  echo ""
done

echo "=== XONG - dan toan bo output nay cho Claude de so sanh ==="
