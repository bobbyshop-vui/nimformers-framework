// cuda_kernels.cu
// Cài đặt CUDA cho cuda_bridge.h. Biên dịch bằng nvcc (KHÔNG dùng clang/gcc
// như metal_bridge.m — xem target `cuda` trong Makefile):
//   nvcc -c cuda_kernels.cu -o cuda_kernels.o
//   nvcc -lib cuda_kernels.o -o libcudakernels.a -lcublas -lcudart
// rồi Nim chỉ link tĩnh vào libcudakernels.a (không {.compile.} trực tiếp
// file .cu vì trình biên dịch C mặc định Nim gọi không phải nvcc).

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include "cuda_bridge.h"

struct CudaCtxImpl {
    cublasHandle_t handle;
};

// ─────────────────────────────────────────────────────────────
// Device / context / buffer
// ─────────────────────────────────────────────────────────────

extern "C" int cuda_device_count(void) {
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) return 0;  // không có driver/GPU CUDA -> coi như 0, không throw
    return n;
}

extern "C" CudaCtxRef cuda_create_context(void) {
    CudaCtxImpl* c = new CudaCtxImpl();
    cublasStatus_t st = cublasCreate(&c->handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuda_create_context: cublasCreate lỗi (status=%d)\n", (int)st);
        delete c;
        return nullptr;
    }
    return (CudaCtxRef)c;
}

extern "C" void cuda_destroy_context(CudaCtxRef ctx) {
    if (!ctx) return;
    CudaCtxImpl* c = (CudaCtxImpl*)ctx;
    cublasDestroy(c->handle);
    delete c;
}

extern "C" CudaBufRef cuda_alloc(size_t bytes) {
    void* p = nullptr;
    cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_alloc(%zu bytes) lỗi: %s\n", bytes, cudaGetErrorString(err));
        return nullptr;
    }
    return p;
}

extern "C" void cuda_free(CudaBufRef buf) {
    if (buf) cudaFree(buf);
}

extern "C" void cuda_upload(CudaBufRef dst, const void* hostSrc, size_t bytes) {
    if (bytes == 0) return;
    cudaMemcpy(dst, hostSrc, bytes, cudaMemcpyHostToDevice);
}

extern "C" void cuda_download(CudaBufRef src, void* hostDst, size_t bytes) {
    if (bytes == 0) return;
    // cudaMemcpy mặc định (stream 0) đồng bộ -> tự chờ mọi kernel trước đó
    // trên stream mặc định xong trước khi copy, không cần
    // cudaDeviceSynchronize() thủ công thêm.
    cudaMemcpy(hostDst, src, bytes, cudaMemcpyDeviceToHost);
}

// ─────────────────────────────────────────────────────────────
// matmul qua cuBLAS
//
// cuBLAS coi mọi ma trận là COLUMN-MAJOR, nhưng phía Nim/host truyền
// A[MxK], B[KxN], C[MxN] theo quy ước ROW-MAJOR (giống metalMatmul).
// Mẹo chuẩn: 1 ma trận row-major MxN == chính ma trận đó column-major NxM
// (tức là A_rowmajor == A^T_colmajor). Nên thay vì tính C = A*B (row-major),
// ta nhờ cuBLAS tính C^T = B^T * A^T (column-major) — nhưng vì
// A_rowmajor đã CHÍNH LÀ A^T ở góc nhìn column-major (không cần transpose
// tường minh gì thêm), lời gọi cublasSgemm dưới đây cho kết quả C đúng
// row-major MxN mà không cần bước transpose riêng nào cả.
// ─────────────────────────────────────────────────────────────
extern "C" void cuda_matmul(CudaCtxRef ctx, const float* dA, int M, int K,
                             const float* dB, int K2, int N, float* dC) {
    CudaCtxImpl* c = (CudaCtxImpl*)ctx;
    const float alpha = 1.0f, beta = 0.0f;
    // Tính (row-major C[MxN]) bằng cách nhờ cuBLAS tính
    // column-major C'[NxM] = B'[NxK] * A'[KxM], với B'=dB, A'=dA xem theo
    // column-major (đúng là B/A row-major nhìn theo layout column-major).
    cublasSgemm(c->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                dB, N,
                dA, K,
                &beta,
                dC, N);
}

// ─────────────────────────────────────────────────────────────
// Kernel CUDA tự viết — tương đương metal_kernels.metal
// ─────────────────────────────────────────────────────────────

__global__ void k_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void k_relu(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__global__ void k_sigmoid(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void k_tanh(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i]);
}

// 1 thread / hàng — đơn giản, đủ dùng cho seqlen/embedDim nhỏ-vừa như
// nimformer.nim đang test; có thể tối ưu bằng shared-memory reduction sau.
__global__ void k_softmax(const float* x, float* y, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* xr = x + (size_t)r * cols;
    float* yr = y + (size_t)r * cols;
    float m = xr[0];
    for (int j = 1; j < cols; j++) if (xr[j] > m) m = xr[j];
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) { float e = expf(xr[j] - m); yr[j] = e; sum += e; }
    for (int j = 0; j < cols; j++) yr[j] /= sum;
}

__global__ void k_layernorm(const float* x, float* y, const float* gamma, const float* beta,
                             int rows, int cols, float eps) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* xr = x + (size_t)r * cols;
    float* yr = y + (size_t)r * cols;
    float mean = 0.0f;
    for (int j = 0; j < cols; j++) mean += xr[j];
    mean /= cols;
    float var = 0.0f;
    for (int j = 0; j < cols; j++) { float d = xr[j] - mean; var += d * d; }
    var /= cols;
    float inv = rsqrtf(var + eps);
    for (int j = 0; j < cols; j++) yr[j] = (xr[j] - mean) * inv * gamma[j] + beta[j];
}

__global__ void k_embedding_lookup(const float* table, const int* indices, float* out,
                                    int dim, int num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // hàng cần tra (0..num)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // cột trong dim
    if (i >= num || j >= dim) return;
    int id = indices[i];
    out[(size_t)i * dim + j] = table[(size_t)id * dim + j];
}

// ─────────────────────────────────────────────────────────────
// Wrapper extern "C" — launch config đơn giản, đủ dùng, chưa tối ưu occupancy
// ─────────────────────────────────────────────────────────────

extern "C" void cuda_add(CudaCtxRef, const float* dA, const float* dB, float* dC, int n) {
    int t = 256, g = (n + t - 1) / t;
    if (g > 0) k_add<<<g, t>>>(dA, dB, dC, n);
}

extern "C" void cuda_relu(CudaCtxRef, const float* dX, float* dY, int n) {
    int t = 256, g = (n + t - 1) / t;
    if (g > 0) k_relu<<<g, t>>>(dX, dY, n);
}

extern "C" void cuda_sigmoid(CudaCtxRef, const float* dX, float* dY, int n) {
    int t = 256, g = (n + t - 1) / t;
    if (g > 0) k_sigmoid<<<g, t>>>(dX, dY, n);
}

extern "C" void cuda_tanh_act(CudaCtxRef, const float* dX, float* dY, int n) {
    int t = 256, g = (n + t - 1) / t;
    if (g > 0) k_tanh<<<g, t>>>(dX, dY, n);
}

extern "C" void cuda_softmax(CudaCtxRef, const float* dX, float* dY, int rows, int cols) {
    int t = 128, g = (rows + t - 1) / t;
    if (g > 0) k_softmax<<<g, t>>>(dX, dY, rows, cols);
}

extern "C" void cuda_layernorm(CudaCtxRef, const float* dX, float* dY,
                                const float* dGamma, const float* dBeta,
                                int rows, int cols, float eps) {
    int t = 128, g = (rows + t - 1) / t;
    if (g > 0) k_layernorm<<<g, t>>>(dX, dY, dGamma, dBeta, rows, cols, eps);
}

extern "C" void cuda_embedding_lookup(CudaCtxRef, const float* dTable, const int* dIndices,
                                       float* dOut, int vocab, int dim, int num) {
    (void)vocab;
    dim3 t(16, 16);
    dim3 g((num + t.x - 1) / t.x, (dim + t.y - 1) / t.y);
    if (num > 0 && dim > 0) k_embedding_lookup<<<g, t>>>(dTable, dIndices, dOut, dim, num);
}
