// matmul_q4_kernel.cu
//
// Kernel CUDA còn THIẾU trong backend (xem comment "TODO" ở
// gpubackend.nim::gpuMatmulQ4 và ResidentWeightHandle) - hiện CUDA đang phải
// dequant int4 -> fp32 MỘT LẦN lúc load model rồi giữ resident dạng fp32
// (residentUploadQ4 nhánh "else"), tức KHÔNG tiết kiệm được băng thông bộ
// nhớ 4-bit như Metal/OpenCL (weight fp32 nặng gấp 8 lần khi đọc từ VRAM
// mỗi lần matmul). Kernel này thêm đúng khả năng đó cho CUDA.
//
// Công thức GIỐNG HỆT 3 chỗ:
//   - Metal:  matmul_q4_kernel_naive (backends/metal/kernels/vecop_matmul.metal)
//   - OpenCL: matmul_q4_naive        (backends/opencl/kernels/vecop_matmul.cl)
//   - CUDA:   matmul_q4_kernel (file này)
// c[m,n] = sum_k a[m,k] * ((nibble(n,k) - zero_point(n,g)) * scale(n,g))
// wq: [N, K] row-major, nibble-packed 2 giá trị/byte (byte thấp = phần tử
// chẵn, giống dequantizeTensorTransposed() trong quant.nim).
//
// BIÊN DỊCH (không có nvcc trong sandbox này, chạy tay trên máy có CUDA
// toolchain thật):
//     nvcc -ptx -arch=sm_70 matmul_q4_kernel.cu -o matmul_q4_kernel.ptx
// rồi NỐI phần ".visible .entry matmul_q4_kernel" sinh ra vào CUỐI file
// backends/cuda/kernels/vecop.ptx hiện có (KHÔNG thay thế toàn bộ file đó -
// các entry point khác đã chạy đúng, giữ nguyên, chỉ thêm entry mới).
// -arch=sm_70 (Volta) là baseline an toàn thấp nhất còn hỗ trợ Tensor Core;
// hạ xuống sm_60/sm_52 nếu cần chạy trên GPU cũ hơn (kernel này không dùng
// tensor core nên không bắt buộc sm_70, chỉ là baseline hợp lý cho 2020+).
//
// Grid/block đề xuất cho matmul_q4_kernel (bản warp-shuffle, xem
// cudaMatmulQ4R trong cuda_driver.nim):
//   block = (32, 1, 1)                       -- đúng 1 warp/block
//   grid  = (ceilDiv(N, MMQ4_NDST=4), M, 1)
// (M thường = 1 lúc decode - 1 token/lần - nên grid.y = 1, không lãng phí gì)
// matmul_q4_kernel_naive (giữ lại để đối chiếu/rollback) vẫn dùng:
//   block = (16, 16, 1)
//   grid  = (ceilDiv(N, 16), ceilDiv(M, 16), 1)

extern "C" __global__ void matmul_q4_kernel_naive(
    const float* __restrict__ a,        // [M, K]
    const unsigned char* __restrict__ wq, // [N, bytesPerRow] packed nibble
    const float* __restrict__ scales,   // [N, nGroupsPerRow]
    const float* __restrict__ zeros,    // [N, nGroupsPerRow]
    float* __restrict__ c,              // [M, N]
    int M, int K, int N,
    int groupSize, int nGroupsPerRow)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int bytesPerRow = (K + 1) / 2;
    int rowByteOff = n * bytesPerRow;
    int rowGroupOff = n * nGroupsPerRow;

    float sum = 0.0f;
    int curGroup = -1;
    float sc = 0.0f, zp = 0.0f;

    #pragma unroll 4
    for (int k = 0; k < K; k++) {
        int g = groupSize > 0 ? (k / groupSize) : 0;
        if (g != curGroup) {
            curGroup = g;
            sc = scales[rowGroupOff + g];
            zp = zeros[rowGroupOff + g];
        }
        unsigned char byteVal = wq[rowByteOff + (k >> 1)];
        unsigned char nibble = (k & 1) == 0 ? (byteVal & 0x0F) : ((byteVal >> 4) & 0x0F);
        float wval = (((float)nibble) - zp) * sc;
        sum += a[m * K + k] * wval;
    }
    c[m * N + n] = sum;
}

// ============================================================
// SUA (đồng bộ với Metal matmul_q4_kernel / OpenCL matmul_q4_kernel_sg):
// kernel naive ở trên = 1 thread/phần tử output, mỗi thread tự đọc lại
// TOÀN BỘ hàng activation a[m,:] từ global memory - lãng phí băng thông
// gấp N/MMQ4_NDST lần so với kernel dưới đây, và không tận dụng được warp
// reduction phần cứng (mỗi thread cộng dồn K lần tuần tự một mình).
//
// Kernel MỚI theo đúng chiến thuật đã áp dụng cho Metal/OpenCL:
//   1. Mỗi warp (32 lane) xử lý MMQ4_NDST=4 hàng output (n) cùng lúc cho
//      1 hàng activation (m) - dispatch theo (rowBlock = ceil(N/4), M).
//   2. Mỗi lane đọc a[m,k] từ global memory ĐÚNG 1 LẦN (k = lane, lane+32,
//      ...) và dùng lại giá trị đó (aVal) cho cả 4 hàng output.
//   3. Reduction dùng __shfl_down_sync (warp-shuffle phần cứng) thay vì
//      shared memory hay vòng lặp tuần tự - tương đương simd_sum() bên
//      Metal / sub_group_reduce_add() bên OpenCL.
// Bắt buộc dispatch block = (32, 1, 1) - đúng 1 warp/block (xem
// cudaMatmulQ4R trong cuda_driver.nim).
#define MMQ4_NDST 4
#define MMQ4_WARP_SIZE 32

extern "C" __global__ void matmul_q4_kernel(
    const float* __restrict__ a,          // [M, K]
    const unsigned char* __restrict__ wq,  // [N, bytesPerRow] packed nibble
    const float* __restrict__ scales,      // [N, nGroupsPerRow]
    const float* __restrict__ zeros,       // [N, nGroupsPerRow]
    float* __restrict__ c,                 // [M, N]
    int M, int K, int N,
    int groupSize, int nGroupsPerRow)
{
    int m = blockIdx.y;
    if (m >= M) return;
    int rowBlock = blockIdx.x;
    int lane = threadIdx.x; // 0..31

    int bytesPerRow = (K + 1) / 2;
    int n0 = rowBlock * MMQ4_NDST;

    int rowByteOff[MMQ4_NDST];
    int rowGroupOff[MMQ4_NDST];
    float partial[MMQ4_NDST] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int rr = 0; rr < MMQ4_NDST; ++rr) {
        int n = n0 + rr;
        rowByteOff[rr] = n * bytesPerRow;
        rowGroupOff[rr] = n * nGroupsPerRow;
    }

    for (int k = lane; k < K; k += MMQ4_WARP_SIZE) {
        float aVal = a[m * K + k];   // đọc global memory ĐÚNG 1 LẦN
        int g = groupSize > 0 ? (k / groupSize) : 0;
        int byteIdx = k >> 1;
        bool lowNibble = (k & 1) == 0;
        #pragma unroll
        for (int rr = 0; rr < MMQ4_NDST; ++rr) {
            int n = n0 + rr;
            if (n >= N) continue;
            float sc = scales[rowGroupOff[rr] + g];
            float zp = zeros[rowGroupOff[rr] + g];
            unsigned char byteVal = wq[rowByteOff[rr] + byteIdx];
            unsigned char nibble = lowNibble ? (byteVal & 0x0F) : ((byteVal >> 4) & 0x0F);
            float wval = (float(nibble) - zp) * sc;
            partial[rr] += aVal * wval;
        }
    }

    #pragma unroll
    for (int rr = 0; rr < MMQ4_NDST; ++rr) {
        float v = partial[rr];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, offset);
        }
        int n = n0 + rr;
        if (lane == 0 && n < N) {
            c[m * N + n] = v;
        }
    }
}
