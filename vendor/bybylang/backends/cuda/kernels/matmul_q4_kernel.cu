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
// Grid/block đề xuất (xem cudaMatmulQ4R trong cuda_driver.nim):
//   block = (16, 16, 1)
//   grid  = (ceilDiv(N, 16), ceilDiv(M, 16), 1)
// (M thường = 1 lúc decode - 1 token/lần - nên grid.y = 1, không lãng phí gì)

extern "C" __global__ void matmul_q4_kernel(
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
