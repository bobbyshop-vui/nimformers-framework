// cuda_bridge.h
// API C generic để Nim gọi vào CUDA (qua cuda_kernels.cu, biên dịch bằng
// nvcc). Cùng vai trò với metal_bridge.h bên Metal, nhưng matmul dùng
// cuBLAS (cublasSgemm) thay vì kernel tự viết — các phép còn lại
// (add/relu/sigmoid/tanh/softmax/layernorm/embedding_lookup) là kernel CUDA
// tự viết trong cuda_kernels.cu.
//
// Quy ước bộ nhớ: cuda_alloc() trả về con trỏ device (cudaMalloc) — dùng
// cuda_free() để giải phóng. cuda_create_context() trả về 1 handle opaque
// bọc cublasHandle_t, sống suốt vòng đời CudaContext bên Nim — dùng
// cuda_destroy_context() khi không cần nữa.

#ifndef CUDA_BRIDGE_H
#define CUDA_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* CudaBufRef;  // con trỏ device (cudaMalloc)
typedef void* CudaCtxRef;  // opaque: bọc cublasHandle_t

// ── Device / context ─────────────────────────────────────────────────
int        cuda_device_count(void);
CudaCtxRef cuda_create_context(void);
void       cuda_destroy_context(CudaCtxRef ctx);

// ── Buffer (device memory) ───────────────────────────────────────────
CudaBufRef cuda_alloc(size_t bytes);
void       cuda_free(CudaBufRef buf);
void       cuda_upload(CudaBufRef dst, const void* hostSrc, size_t bytes);
void       cuda_download(CudaBufRef src, void* hostDst, size_t bytes);

// ── matmul qua cuBLAS: C[MxN] = A[MxK] * B[KxN], quy ước row-major phía Nim ──
void cuda_matmul(CudaCtxRef ctx, const float* dA, int M, int K,
                  const float* dB, int K2, int N, float* dC);

// ── Kernel CUDA tự viết (tương đương metal_kernels.metal) ────────────
void cuda_add(CudaCtxRef ctx, const float* dA, const float* dB, float* dC, int n);
void cuda_relu(CudaCtxRef ctx, const float* dX, float* dY, int n);
void cuda_sigmoid(CudaCtxRef ctx, const float* dX, float* dY, int n);
void cuda_tanh_act(CudaCtxRef ctx, const float* dX, float* dY, int n);
void cuda_softmax(CudaCtxRef ctx, const float* dX, float* dY, int rows, int cols);
void cuda_layernorm(CudaCtxRef ctx, const float* dX, float* dY,
                     const float* dGamma, const float* dBeta,
                     int rows, int cols, float eps);
void cuda_embedding_lookup(CudaCtxRef ctx, const float* dTable, const int* dIndices,
                            float* dOut, int vocab, int dim, int num);

#ifdef __cplusplus
}
#endif

#endif // CUDA_BRIDGE_H