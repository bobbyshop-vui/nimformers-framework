// rope_kernel.cu
//
// Kernel CUDA còn thiếu trong backend: RoPE (rotate_half, in-place).
// Tương ứng 1-1 với rope_kernel bên Metal (metal/kernels/vecop_matmul.metal)
// và OpenCL (opencl/kernels/vecop_matmul.cl) - cùng công thức, cùng thứ tự
// tham số, cùng cách chia grid (i = trục D/2, h = trục Hn, t = trục T),
// để 3 backend cho ra bit-for-bit cùng kết quả (trong sai số float).
//
// File này chỉ là NGUỒN .cu để biên dịch tay bằng nvcc trên máy có CUDA
// toolchain thật:
//
//     nvcc -ptx -arch=sm_70 rope_kernel.cu -o rope_kernel.ptx
//
// rồi nối nội dung .visible .entry rope_kernel sinh ra vào cuối file PTX
// hiện có (backends/cuda/kernels/vecop.ptx) - KHÔNG thay thế toàn bộ file
// đó, vì các entry point khác (vecop_add, softmax_kernel, attention_fused...)
// đã được viết tay và đã chạy đúng, không đụng vào.
//
// Sau khi có .ptx mới, cuda_driver.nim mới nên gọi getFn("rope_kernel") và
// cuLaunchKernel với grid = (ceilDiv(D/2, blockX), Hn, T) - phần plumbing đó
// KHÔNG nằm trong file này, chỉ tập trung vào bản thân kernel theo đúng yêu
// cầu.

extern "C" __global__ void rope_kernel(
    float* x,             // [T, Hn, D] in-place
    const float* cosT,    // [T, D/2]
    const float* sinT,    // [T, D/2]
    int Hn,
    int D)
{
    // gridDim/blockDim ánh xạ: x = i trong [0, D/2), y = h trong [0, Hn), z = t trong [0, T)
    int half_ = D / 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_) return;

    int h = blockIdx.y;
    int t = blockIdx.z;

    int base = (t * Hn + h) * D;
    float c = cosT[t * half_ + i];
    float s = sinT[t * half_ + i];

    float x1 = x[base + i];
    float x2 = x[base + i + half_];

    x[base + i]         = x1 * c - x2 * s;
    x[base + i + half_] = x2 * c + x1 * s;
}
