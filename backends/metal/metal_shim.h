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

#ifdef __cplusplus
}
#endif

#endif
