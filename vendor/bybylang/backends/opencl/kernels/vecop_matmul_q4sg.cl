// vecop_matmul_q4sg.cl
//
// Kernel Q4 matmul dùng SUBGROUP REDUCTION - mirror trực tiếp của:
//   - Metal: matmul_q4_kernel (backends/metal/kernels/vecop_matmul.metal),
//            dùng simd_sum() trên simdgroup 32 lane.
//   - CUDA:  matmul_q4_kernel (backends/cuda/kernels/matmul_q4_kernel.cu),
//            dùng __shfl_down_sync() trên warp 32 lane.
// Đây chính là phần OpenCL còn THIẾU khiến backend OpenCL chậm hơn hẳn
// Metal cho Q4 matmul (linear layer chiếm phần lớn thời gian mỗi token
// lúc decode) - trước đây OpenCL chỉ có matmul_q4_naive (1 work-item/phần
// tử output, tự đọc lại toàn bộ hàng activation và cộng dồn tuần tự K lần
// KHÔNG reduction phần cứng, giống hệt matmul_q4_kernel_naive bên Metal).
//
// Chiến thuật (giống hệt 2 backend kia):
//   1. Mỗi subgroup xử lý MMQ4_NDST=4 hàng output (n) cùng lúc cho 1 hàng
//      activation (m) - dispatch theo (rowBlock = ceil(N/4), M).
//   2. Mỗi work-item đọc a[m,k] từ global memory ĐÚNG 1 LẦN (k = lane,
//      lane+sgSize, ...) và dùng lại giá trị đó (aVal) cho cả 4 hàng -
//      giảm băng thông đọc activation ~4 lần so với bản naive.
//   3. Reduction dùng sub_group_reduce_add() (subgroup shuffle phần cứng),
//      KHÔNG cần local/shared memory, KHÔNG cần barrier.
//
// File này build thành 1 cl_program TÁCH RIÊNG khỏi vecop_matmul.cl (xem
// opencl_api.nim: buildQ4SgProgram()) vì "#pragma OPENCL EXTENSION
// cl_khr_subgroups : enable" có thể khiến build THẤT BẠI trên driver
// không hỗ trợ - tách riêng để lỗi đó (nếu có) không kéo sập luôn các
// kernel khác (vecop/softmax/layernorm/attention...) nằm chung 1 program.
// opencl_api.nim bọc việc build này trong try/except; nếu thất bại (device
// không hỗ trợ subgroup, hoặc driver không hỗ trợ pragma này), tự động
// fallback về matmul_q4_naive - không ảnh hưởng phần còn lại của backend.
//
// Dispatch đề xuất (xem openclSessionMatmulQ4Enc/openclMatmulQ4):
//   global_work_size = (ceilDiv(N, MMQ4_NDST) * sgSize, M)
//   local_work_size  = (sgSize, 1)     -- sgSize lấy từ
//     clGetKernelSubGroupInfo(...CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE...)
//     hoặc mặc định 32 (đúng cho hầu hết GPU Intel/NVIDIA/AMD hiện đại) rồi
//     kiểm tra get_sub_group_size() thật bên trong kernel qua kết quả chạy
//     thử nếu cần chính xác tuyệt đối.
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define MMQ4_NDST 4

__kernel void matmul_q4_kernel_sg(__global const float* a, __global const uchar* wq,
                                   __global const float* scales, __global const float* zeros,
                                   __global float* c,
                                   const int M, const int K, const int N,
                                   const int groupSize, const int nGroupsPerRow) {
    int m = get_global_id(1);
    if (m >= M) return;
    int rowBlock = get_group_id(0);
    int lane = get_sub_group_local_id();
    int sgSize = get_sub_group_size();

    int bytesPerRow = (K + 1) / 2;
    int n0 = rowBlock * MMQ4_NDST;

    int rowByteOff[MMQ4_NDST];
    int rowGroupOff[MMQ4_NDST];
    float partial[MMQ4_NDST] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int rr = 0; rr < MMQ4_NDST; ++rr) {
        int n = n0 + rr;
        rowByteOff[rr] = n * bytesPerRow;
        rowGroupOff[rr] = n * nGroupsPerRow;
    }

    for (int k = lane; k < K; k += sgSize) {
        float aVal = a[m * K + k];   // doc global memory DUNG 1 LAN
        int g = groupSize > 0 ? (k / groupSize) : 0;
        int byteIdx = k >> 1;
        bool lowNibble = (k & 1) == 0;
        for (int rr = 0; rr < MMQ4_NDST; ++rr) {
            int n = n0 + rr;
            if (n >= N) continue;
            float sc = scales[rowGroupOff[rr] + g];
            float zp = zeros[rowGroupOff[rr] + g];
            uchar byteVal = wq[rowByteOff[rr] + byteIdx];
            uchar nibble = lowNibble ? (byteVal & 0x0F) : ((byteVal >> 4) & 0x0F);
            float wval = (((float)nibble) - zp) * sc;
            partial[rr] += aVal * wval;
        }
    }

    for (int rr = 0; rr < MMQ4_NDST; ++rr) {
        float rowSum = sub_group_reduce_add(partial[rr]);
        int n = n0 + rr;
        if (lane == 0 && n < N) {
            c[m * N + n] = rowSum;
        }
    }
}
