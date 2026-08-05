// rmsnorm_kernel.cu
//
// Kernel CUDA còn thiếu trong backend: RMSNorm forward (KHÔNG trừ mean,
// KHÔNG có beta - khác LayerNorm ở đúng 2 điểm này). Thuật toán reduction
// (shared-memory tree reduction, 1 threadblock/row) được dịch TRỰC TIẾP
// 1-1 từ rmsnorm_kernel bên Metal (backends/metal/kernels/vecop_matmul.metal)
// - vốn là bản mirror của layernorm_kernel Metal ĐÃ CHẠY THẬT qua
// beAddLayernormFused - để giữ đúng cùng 1 thuật toán reduction đã có tiền
// lệ hoạt động, không tự sáng tác cách reduction mới.
//
// Công thức: invStd = 1/sqrt(mean(x^2) + eps); y = x * invStd * gamma
// (không trừ mean, không cộng beta - đây LÀ định nghĩa RMSNorm, không phải
// thiếu sót so với LayerNorm).
//
// File này chỉ là NGUỒN .cu để biên dịch tay bằng nvcc trên máy có CUDA
// toolchain thật:
//
//     nvcc -ptx -arch=sm_70 rmsnorm_kernel.cu -o rmsnorm_kernel.ptx
//
// rồi nối nội dung .visible .entry rmsnorm_kernel sinh ra vào cuối file PTX
// hiện có (backends/cuda/kernels/vecop.ptx) - KHÔNG thay thế toàn bộ file
// đó, đúng quy trình đã dùng cho rope_kernel.cu.
//
// Launch convention (khớp cudaRmsnormR trong cuda_driver.nim):
//   grid  = (rows, 1, 1)   -- 1 threadblock xử lý đúng 1 row
//   block = (min(maxThreadsPerBlock, 256), 1, 1)
//
// CHƯA CHẠY TRÊN PHẦN CỨNG THẬT (không có máy CUDA ở môi trường viết code
// này) - PHẢI verify bằng dump_reference_states.py (so hidden state sau
// từng layer với HF reference) trước khi tin kết quả sinh ra từ kernel này.

extern "C" __global__ void rmsnorm_kernel(
    const float* x,      // [rows, cols]
    const float* gamma,  // [cols]
    float* y,            // [rows, cols] output
    int cols,
    float eps)
{
    // Shared mem TĨNH cỡ 512 (= getOptimalBlockSize() lớn nhất có thể, xem
    // cuda_driver.nim) - ĐÚNG dù blockDim.x thực tế nhỏ hơn (128/256) vì chỉ
    // tid < blockDim.x được đọc/ghi. Dùng tĩnh để khớp convention hiện có:
    // mọi cuLaunchKernel trong cuda_driver.nim đều truyền sharedMemBytes=0.
    __shared__ float red[512];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nThreads = blockDim.x;
    long off = (long)row * (long)cols;

    // Bước 1: mỗi thread cộng dồn sum-of-squares cho phần cols của mình
    // (stride = nThreads, giống hệt vòng "for c = tid; c < cols; c += BB_WG" bên Metal)
    float sumSq = 0.0f;
    for (int c = tid; c < cols; c += nThreads) {
        float v = x[off + c];
        sumSq += v * v;
    }
    red[tid] = sumSq;
    __syncthreads();

    // Bước 2: tree reduction trong threadblock (y hệt vòng lặp s >>= 1 bên Metal)
    for (int s = nThreads / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }

    float meanSq = red[0] / (float)cols;
    float invStd = rsqrtf(meanSq + eps);
    __syncthreads();

    // Bước 3: scale + gamma, KHÔNG trừ mean, KHÔNG cộng beta
    for (int c = tid; c < cols; c += nThreads) {
        y[off + c] = x[off + c] * invStd * gamma[c];
    }
}
