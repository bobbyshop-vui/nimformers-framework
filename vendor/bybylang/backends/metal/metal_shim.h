// metal_shim.h - C interface exposed to Nim for the Metal compute backend (macOS only).
#ifndef METAL_SHIM_H
#define METAL_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

// Trả về 1 nếu máy có GPU hỗ trợ Metal, 0 nếu không.
int metal_available(void);

// kernel_src: nguồn Metal Shading Language (đọc từ kernels/vecop_matmul.metal
// lúc Nim compile-time qua staticRead, KHÔNG hardcode trong file .m nữa).
// op: 0=add, 1=sub, 2=mul, 3=div. Trả về 1 nếu chạy thành công, 0 nếu lỗi.
int metal_vecop(const char* kernel_src, int op, const float* a, const float* b, float* c, int n);

// C(m x n) = A(m x k) * B(k x n), row-major. Trả về 1 nếu chạy thành công, 0 nếu lỗi.
int metal_matmul(const char* kernel_src, const float* a, const float* b, float* c, int m, int k, int n);

// Matmul truc tiep tren weight int4-asymmetric DA PACK (khong dequant ra
// fp32 tren CPU truoc). wq: packed bytes [N * ceil(K/2)]. scales/zeros:
// [N * nGroupsPerRow] (nGroupsPerRow = ceil(K/groupSize), groupSize<=0 nghia
// la 1 group/hang). Xem chu thich chi tiet trong vecop_matmul.metal
// (matmul_q4_kernel) truoc khi dung - CHUA CHAY THU TREN GPU THAT.
int metal_matmul_q4(const char* kernel_src, const float* a, const unsigned char* wq,
                     const float* scales, const float* zeros, float* c,
                     int m, int k, int n, int groupSize, int nGroupsPerRow);

// Chạy 2 phép matmul ĐỘC LẬP (a1*b1->c1 và a2*b2->c2) trong CÙNG một command
// buffer / một lần compile pipeline, chỉ commit+wait MỘT LẦN. Dùng khi cần
// dispatch 2 matmul không phụ thuộc lẫn nhau (vd. dW và dX trong backward)
// mà không muốn trả giá overhead của 2 lần commit+wait riêng lẻ.
// Trả về 1 nếu chạy thành công, 0 nếu lỗi.
int metal_matmul2(const char* kernel_src,
                   const float* a1, const float* b1, float* c1, int m1, int k1, int n1,
                   const float* a2, const float* b2, float* c2, int m2, int k2, int n2);

int metal_activation(const char* kernel_src, int op, const float* x, float* y, int n);
int metal_softmax(const char* kernel_src, const float* x, float* y, int rows, int cols);
int metal_layernorm(const char* kernel_src, const float* x, const float* gamma, const float* beta, float* y, int rows, int cols, float eps);
int metal_embedding_lookup(const char* kernel_src, const float* table, const int* indices, float* y, int vocab, int dim, int num_indices);

// op: 0=relu, 1=sigmoid, 2=tanh, 3=apflu (dùng chung bảng mã với metal_activation).
// Gọi activation_backward_kernel trong vecop_matmul.metal (đã có sẵn kernel,
// trước đây CHƯA được bind ra hàm C -> gpubackend.nim không có đường gọi tới
// nên luôn rớt xuống CPU cho apflu backward dù chọn backend metal).
int metal_activation_backward(const char* kernel_src, int op, const float* x, const float* dy, float* dx, int n);

// Gọi layernorm_backward_kernel. dx: rows*cols phần tử, dgamma/dbeta: cols phần tử.
int metal_layernorm_backward(const char* kernel_src, const float* dy, const float* x,
                              const float* gamma, const float* beta,
                              float* dx, float* dgamma, float* dbeta,
                              int rows, int cols, float eps);

// Causal fused attention forward. q/k/v: B*H*S*D phần tử, o: B*H*S*D,
// s_matrix: B*H*S*S. `mask` không dùng trong kernel (causal áp qua vòng lặp
// tj<=ti giống bản CPU tham chiếu) nhưng vẫn nhận vào để chữ ký khớp
// beAttentionFused ở tầng backend.nim.
int metal_attention_fused(const char* kernel_src, const float* q, const float* k, const float* v,
                           float* o, float* s_matrix, int B, int H, int S, int D, float scale);

// Causal fused attention backward. dq/dk/dv: B*H*S*D phần tử MỖI cái, PHẢI
// được caller zero-init trước khi gọi (kernel dùng "+=" / atomic-add trên
// buffer output, không ghi đè hoàn toàn như forward).
int metal_attention_fused_backward(const char* kernel_src, const float* q, const float* k, const float* v,
                                    const float* s_matrix, const float* dy,
                                    float* dq, float* dk, float* dv,
                                    int B, int H, int S, int D, float scale);

// ─────────────────────────────────────────────────────────────────────────
// API resident: nhiều op mã hoá vào CÙNG 1 MTLCommandBuffer (giống metal_matmul2
// đã làm cho 2 matmul), chỉ commit+waitUntilCompleted MỘT LẦN ở
// metal_session_end(). Buffer trả về là opaque handle (con trỏ tới id<MTLBuffer>
// đã CFBridgingRetain) - giữ sống tới khi gọi metal_buffer_free.
// LƯU Ý: CHƯA CHẠY THỬ TRÊN GPU METAL THẬT (môi trường sinh code này là Linux,
// không có macOS/framework Metal để biên dịch+chạy) - viết bám sát 100% idiom
// đã có sẵn và ĐANG CHẠY ĐƯỢC trong metal_vecop/metal_matmul/metal_matmul2 ở
// trên (cùng cách tạo buffer, cùng cách lấy pipeline, cùng cách dispatch), chỉ
// khác ở chỗ dùng chung 1 command buffer thay vì tạo mới mỗi lệnh. Bắt buộc
// phải build + test trên máy Mac thật trước khi dùng cho train thật.
typedef void* MetalBufferHandle;

// Bắt đầu 1 phiên: tạo command buffer + encoder mới, lưu vào biến static.
// Gọi trước chuỗi op resident đầu tiên của 1 layer/1 forward pass.
int metal_session_begin(const char* kernel_src);

// Upload dữ liệu host -> MTLBuffer (memcpy CPU, Apple Silicon là unified memory
// nên KHÔNG có DMA H2D thật như CUDA rời - vẫn tách hàm riêng cho rõ ràng
// và để chạy đúng trên Mac Intel có GPU rời nếu có, vì lúc đó storage mode
// khác đi driver mới thật sự cần transfer).
MetalBufferHandle metal_upload(const float* data, int n);
MetalBufferHandle metal_upload_indices(const int* data, int n);
// Cấp buffer output resident (kernel sẽ ghi đè), CHƯA init dữ liệu.
MetalBufferHandle metal_alloc_scratch(int n);
// Encode thêm 1 op vào command buffer đang mở (KHÔNG commit) - trả về 1/0.
int metal_vecop_enc(int op, MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int n);
int metal_activation_enc(int op, MetalBufferHandle x, MetalBufferHandle y, int n);
int metal_softmax_enc(MetalBufferHandle x, MetalBufferHandle y, int rows, int cols);
int metal_layernorm_enc(MetalBufferHandle x, MetalBufferHandle gamma, MetalBufferHandle beta,
                         MetalBufferHandle y, int rows, int cols, float eps);
int metal_embedding_lookup_enc(MetalBufferHandle table, MetalBufferHandle indices,
                                MetalBufferHandle y, int vocab, int dim, int num_indices);
int metal_matmul_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int m, int k, int n);
// Commit + wait MỘT LẦN cho toàn bộ op đã encode từ lúc metal_session_begin.
// Sau khi gọi hàm này, mọi MetalBufferHandle trong phiên đã có dữ liệu đúng,
// đọc được bằng metal_buffer_read.
int metal_session_end(void);
// Đọc dữ liệu về host (chỉ hợp lệ SAU metal_session_end của phiên tạo ra handle đó).
int metal_buffer_read(MetalBufferHandle h, float* outData, int n);
// Giải phóng ngay khi không còn cần (vd. activation trung gian sau khi dùng
// xong trong backward) - an toàn gọi bất cứ lúc nào, kể cả giữa 1 phiên đang
// mở, vì buffer chỉ thật sự release() sau khi ARC thấy không còn tham chiếu
// nào trong command buffer đang chạy (Metal tự retain buffer khi encode).
void metal_buffer_free(MetalBufferHandle h);

#ifdef __cplusplus
}
#endif

#endif