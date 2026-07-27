#include <metal_stdlib>
using namespace metal;

// ============================================================
// VECTOR OPERATIONS
// ============================================================
kernel void vecop_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    device const int& op [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (op == 0) c[id] = a[id] + b[id];
    else if (op == 1) c[id] = a[id] - b[id];
    else if (op == 2) c[id] = a[id] * b[id];
    else if (op == 3) c[id] = a[id] / b[id];
}

// ============================================================
// COPY INTO (GPU-side concat) - THÊM MỚI: dst[dstOffset + id] = src[id],
// dùng để ghép nhiều buffer con (vd outHeads của từng KV-head trong
// attention decode) thành 1 buffer liên tục [H*D] NGAY TRÊN GPU, tránh
// phải sessionRead() về CPU rồi sessionUpload() lại chỉ để nối mảng -
// đây là điểm chặn cuối cùng khiến session attention-decode và session
// oProj không gộp được làm 1. dstOffset tính bằng SỐ PHẦN TỬ (không phải
// byte) - metal_copy_into_enc (metal_shim.m) tự nhân sizeof(float) khi
// bind buffer offset.
kernel void copy_into_kernel(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    dst[id] = src[id];
}

// ============================================================
// MATMUL TILED
// ============================================================
// SỬA (thay cho fix "hạ cứng BB_TILE=8 cho MỌI GPU" trước đây):
// Root cause thật sự KHÔNG phải "tile 32 luôn nguy hiểm" mà là
// metal_matmul()/metal_matmul_enc() dispatch groupSize=32x32=1024 threads
// mà KHÔNG kiểm tra pipeline.maxTotalThreadsPerThreadgroup của GPU đang
// chạy trước (khác metal_vecop cùng file đã làm đúng bằng MIN(...)). Hạ
// tile về 8 cho MỌI GPU để né bug đó là đúng an toàn nhưng sai hiệu năng:
// trên Apple Silicon / GPU rời mạnh, threadgroup 1024 threads hoàn toàn
// hợp lệ và tile 32 nhanh hơn tile 8 nhiều lần (data reuse tốt hơn hẳn:
// mỗi phần tử tileA/tileB được dùng lại 32 lần thay vì 8 lần trước khi
// nạp lại từ device memory).
// Fix ĐÚNG: build sẵn 3 biến thể kernel tile 8/16/32 (macro sinh code,
// KHÔNG dùng function constant vì threadgroup array size cần literal rõ
// ràng lúc build MSL, tránh rủi ro không tương thích driver cũ). Phía
// metal_shim.m sẽ tự đo pipeline.maxTotalThreadsPerThreadgroup của TỪNG
// pipeline sau khi tạo (không đoán trước theo tên GPU) rồi chọn variant
// lớn nhất mà threadgroup thật sự cho phép, fallback dần xuống nếu không.
#define BB_MATMUL_VARIANT(TILE, KFUNC)                                       \
kernel void KFUNC(                                                           \
    device const float* a [[buffer(0)]],                                    \
    device const float* b [[buffer(1)]],                                    \
    device float* c [[buffer(2)]],                                          \
    device const int& m [[buffer(3)]],                                      \
    device const int& k [[buffer(4)]],                                      \
    device const int& n [[buffer(5)]],                                      \
    uint2 gid [[thread_position_in_grid]],                                  \
    uint2 tid [[thread_position_in_threadgroup]]                            \
) {                                                                          \
    threadgroup float tileA[TILE][TILE];                                    \
    threadgroup float tileB[TILE][TILE];                                    \
    int row = int(gid.y);                                                   \
    int col = int(gid.x);                                                   \
    int tx = int(tid.x), ty = int(tid.y);                                   \
    float sum = 0.0f;                                                       \
    int numTiles = (k + TILE - 1) / TILE;                                   \
    for (int t = 0; t < numTiles; ++t) {                                    \
        int aCol = t * TILE + tx;                                           \
        int bRow = t * TILE + ty;                                           \
        tileA[ty][tx] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;    \
        tileB[ty][tx] = (bRow < k && col < n) ? b[bRow * n + col] : 0.0f;    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        for (int p = 0; p < TILE; ++p) {                                    \
            sum += tileA[ty][p] * tileB[p][tx];                             \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }                                                                        \
    if (row < m && col < n) {                                               \
        c[row * n + col] = sum;                                             \
    }                                                                        \
}

// 3 biến thể - metal_shim.m chọn 1 trong 3 tuỳ pipeline.maxTotalThreadsPerThreadgroup
// thật sự của GPU đang chạy (8x8=64, 16x16=256, 32x32=1024 threads/threadgroup).
BB_MATMUL_VARIANT(8,  matmul_kernel_t8)
BB_MATMUL_VARIANT(16, matmul_kernel_t16)
BB_MATMUL_VARIANT(32, matmul_kernel_t32)

// Giữ nguyên tên "matmul_kernel" = alias tile 8 để mọi chỗ gọi cũ (nếu còn
// sót nơi nào không qua đường chọn-tile-động) vẫn compile được, KHÔNG BAO
// GIỜ dispatch với groupSize > 8x8 cho tên này.
BB_MATMUL_VARIANT(8, matmul_kernel)

// ============================================================
// MATMUL TILED "NT" (C = A @ B^T, B lưu row-major [N,K] - KHÔNG transpose
// vật lý). THÊM MỚI: dùng cho attention decode - K-cache/V-cache lưu tự
// nhiên theo layout [S,D] (N=S, K=headDim) - loại bỏ hoàn toàn bước
// transpose CPU O(S*D)/token trước đây phải làm để build kBlockT.
// ============================================================
#define BB_MATMUL_NT_VARIANT(TILE, KFUNC)                                    \
kernel void KFUNC(                                                           \
    device const float* a [[buffer(0)]],                                    \
    device const float* b [[buffer(1)]],                                    \
    device float* c [[buffer(2)]],                                          \
    device const int& m [[buffer(3)]],                                      \
    device const int& k [[buffer(4)]],                                      \
    device const int& n [[buffer(5)]],                                      \
    uint2 gid [[thread_position_in_grid]],                                  \
    uint2 tid [[thread_position_in_threadgroup]]                            \
) {                                                                          \
    threadgroup float tileA[TILE][TILE];                                    \
    threadgroup float tileB[TILE][TILE];                                    \
    int tx = int(tid.x), ty = int(tid.y);                                   \
    int rowBase = int(gid.y) - ty;                                          \
    int colBase = int(gid.x) - tx;                                          \
    int row = rowBase + ty;                                                 \
    int col = colBase + tx;                                                 \
    float sum = 0.0f;                                                       \
    int numTiles = (k + TILE - 1) / TILE;                                   \
    for (int t = 0; t < numTiles; ++t) {                                    \
        int aK = t * TILE + tx;                                             \
        int bCol = colBase + ty; /* doc lap voi 'col' - dung ty lam cot khi load B */ \
        int bK = t * TILE + tx;                                             \
        tileA[ty][tx] = (row < m && aK < k) ? a[row * k + aK] : 0.0f;        \
        tileB[ty][tx] = (bCol < n && bK < k) ? b[bCol * k + bK] : 0.0f;      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        for (int p = 0; p < TILE; ++p) {                                    \
            sum += tileA[ty][p] * tileB[tx][p];                             \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }                                                                        \
    if (row < m && col < n) {                                               \
        c[row * n + col] = sum;                                             \
    }                                                                        \
}

BB_MATMUL_NT_VARIANT(8, matmul_nt_kernel)

// ============================================================
// MATMUL2 - DUAL MATMUL
// ============================================================
kernel void matmul2_kernel(
    device const float* a1 [[buffer(0)]],
    device const float* b1 [[buffer(1)]],
    device float* c1 [[buffer(2)]],
    device const int32_t* dims1 [[buffer(3)]],
    device const float* a2 [[buffer(4)]],
    device const float* b2 [[buffer(5)]],
    device float* c2 [[buffer(6)]],
    device const int32_t* dims2 [[buffer(7)]],
    uint2 position [[thread_position_in_grid]]
) {
    int m1 = dims1[0], k1 = dims1[1], n1 = dims1[2];
    int m2 = dims2[0], k2 = dims2[1], n2 = dims2[2];
    
    int row = position.y;
    int col = position.x;
    
    if (row < m1 && col < n1) {
        float sum = 0.0f;
        for (int i = 0; i < k1; ++i) {
            sum += a1[row * k1 + i] * b1[i * n1 + col];
        }
        c1[row * n1 + col] = sum;
    }
    
    if (row < m2 && col < n2) {
        float sum = 0.0f;
        for (int i = 0; i < k2; ++i) {
            sum += a2[row * k2 + i] * b2[i * n2 + col];
        }
        c2[row * n2 + col] = sum;
    }
}

// ============================================================
// ACTIVATIONS
// ============================================================
kernel void activation_kernel(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const int& op [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float val = x[id];
    if (op == 0) {
        y[id] = val > 0.0f ? val : 0.0f;
    } else if (op == 1) {
        y[id] = 1.0f / (1.0f + exp(-val));
    } else if (op == 2) {
        y[id] = tanh(val);
    } else if (op == 3) {
        float alpha = 0.1f;
        float beta = 1.0f;
        y[id] = val > 0.0f ? val * (1.0f + alpha * val) : beta * val * exp(val);
    }
}

// ============================================================
// ACTIVATION BACKWARD
// ============================================================
kernel void activation_backward_kernel(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    device const int& op [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float val = x[id];
    float d = dy[id];
    if (op == 0) {
        dx[id] = val > 0.0f ? d : 0.0f;
    } else if (op == 1) {
        float sig = 1.0f / (1.0f + exp(-val));
        dx[id] = d * sig * (1.0f - sig);
    } else if (op == 2) {
        float t = tanh(val);
        dx[id] = d * (1.0f - t * t);
    } else if (op == 3) {
        float alpha = 0.1f;
        float beta = 1.0f;
        dx[id] = val > 0.0f ? d * (1.0f + 2.0f * alpha * val) : d * beta * exp(val) * (1.0f + val);
    }
}

// ============================================================
// ROPE (rotate_half, in-place) - THÊM MỚI
// ============================================================
// Trước đây RoPE phải: đọc Q/K về CPU, tự build mảng cosFull/sinFull/rotHalf
// kích thước [T,H,D] (lãng phí, lặp giá trị cos/sin H lần), upload lại, làm
// 3 sessionVecOp (mul,mul,add), rồi lại đọc về CPU lần nữa để dùng cho
// attention - tốn 1 session GPU + 2 lượt round-trip CPU RIÊNG chỉ cho RoPE.
// Kernel này áp dụng rotate_half TRỰC TIẾP trong lúc Q/K còn nằm trên GPU
// (ngay sau QKV matmul, TRONG CÙNG session), không cần build mảng lặp lại,
// không cần đọc CPU giữa chừng. cosT/sinT là đoạn [T, D/2] đã cắt sẵn theo
// posOffset ở phía CPU (bảng gốc rope.cosT/sinT không đổi giữa các
// layer/token nên phần cắt cũng rất nhỏ so với Q/K).
kernel void rope_kernel(
    device float* x               [[buffer(0)]],   // [T, Hn, D] in-place
    device const float* cosT      [[buffer(1)]],   // [T, D/2]
    device const float* sinT      [[buffer(2)]],   // [T, D/2]
    device const int& Hn          [[buffer(3)]],
    device const int& D           [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]          // gid.x = i in [0, D/2), gid.y = h in [0, Hn), gid.z = t in [0, T)
) {
    int half_ = D / 2;
    int i = int(gid.x);
    if (i >= half_) return;
    int h = int(gid.y);
    int t = int(gid.z);
    int base = (t * Hn + h) * D;
    float c = cosT[t * half_ + i];
    float s = sinT[t * half_ + i];
    float x1 = x[base + i];
    float x2 = x[base + i + half_];
    x[base + i]         = x1 * c - x2 * s;
    x[base + i + half_] = x2 * c + x1 * s;
}

// ============================================================
// SOFTMAX
// ============================================================
#define BB_WG 256

kernel void softmax_kernel(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const int& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float red[BB_WG];
    uint off = row * cols;

    float m = -INFINITY;
    for (int c = int(tid); c < cols; c += BB_WG) {
        m = max(m, x[off + c]);
    }
    red[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxVal = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (int c = int(tid); c < cols; c += BB_WG) {
        float e = exp(x[off + c] - maxVal);
        y[off + c] = e;
        sum += e;
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float invSum = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int c = int(tid); c < cols; c += BB_WG) {
        y[off + c] *= invSum;
    }
}

// ============================================================
// LAYERNORM FORWARD
// ============================================================
kernel void layernorm_kernel(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* y [[buffer(3)]],
    device const int& cols [[buffer(4)]],
    device const float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float red[BB_WG];
    uint off = row * cols;

    float sum = 0.0f, sumSq = 0.0f;
    for (int c = int(tid); c < cols; c += BB_WG) {
        float v = x[off + c];
        sum += v;
        sumSq += v * v;
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = red[0] / float(cols);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    red[tid] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float meanSq = red[0] / float(cols);
    float var = meanSq - mean * mean;
    float invStd = 1.0f / sqrt(var + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int c = int(tid); c < cols; c += BB_WG) {
        y[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c];
    }
}

// ============================================================
// LAYERNORM BACKWARD
// ============================================================
kernel void layernorm_backward_kernel(
    device const float* dy [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device float* dx [[buffer(4)]],
    device float* dgamma [[buffer(5)]],
    device float* dbeta [[buffer(6)]],
    device const int& rows [[buffer(7)]],
    device const int& cols [[buffer(8)]],
    device const float& eps [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < (uint)cols) {
        float dg = 0.0f;
        float db = 0.0f;
        for (int r = 0; r < rows; ++r) {
            uint off = r * cols;
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) mean += x[off + c];
            mean /= float(cols);
            float var = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float diff = x[off + c] - mean;
                var += diff * diff;
            }
            var /= float(cols);
            float invStd = 1.0f / sqrt(var + eps);
            
            float norm = (x[off + id] - mean) * invStd;
            dg += dy[off + id] * norm;
            db += dy[off + id];
        }
        dgamma[id] = dg;
        dbeta[id] = db;
    }
    
    uint row = id;
    if (row < (uint)rows) {
        uint off = row * cols;
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += x[off + c];
        mean /= float(cols);
        
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float diff = x[off + c] - mean;
            var += diff * diff;
        }
        var /= float(cols);
        float invStd = 1.0f / sqrt(var + eps);
        
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float grad = dy[off + c] * gamma[c] * invStd;
            sum1 += grad;
            sum2 += grad * (x[off + c] - mean);
        }
        sum2 *= (-invStd * invStd / float(cols));
        
        for (int c = 0; c < cols; ++c) {
            float term1 = dy[off + c] * gamma[c] * invStd;
            float term2 = sum1 / float(cols);
            float term3 = (x[off + c] - mean) * sum2 * 2.0f / float(cols);
            dx[off + c] = term1 - term2 + term3;
        }
    }
}

// ============================================================
// EMBEDDING LOOKUP
// ============================================================
kernel void embedding_lookup_kernel(
    device const float* table [[buffer(0)]],
    device const int32_t* indices [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const int& dim [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    int32_t idx = indices[id];
    uint dst_off = id * dim;
    uint src_off = idx * dim;
    for (int d = 0; d < dim; ++d) {
        y[dst_off + d] = table[src_off + d];
    }
}

// ============================================================
// MATMUL TRUC TIEP TREN INT4 ASYMMETRIC (packed, per-group scale/zero_point)
// ============================================================
// SUA (item 3 trong ghi chu hieu nang): TRUOC DAY moi lan Linear.forward()
// phai dequantizeTensorTransposed() TOAN BO weight int4 ra 1 mang fp32 day
// du tren CPU (vd 4096x11008 ~180MB fp32) roi UPLOAD ca mang fp32 do len
// GPU chi de nhan 1 lan roi vut - dung y nhu llama.cpp KHONG lam (ho nhan
// truc tiep tren du lieu quantized). Kernel nay nhan thang tren buffer
// int4 da pack (2 gia tri/byte) + scale/zero_point theo group, KHONG can
// dequant rieng tren CPU va KHONG can upload mang fp32 day du - giam bang
// thong upload ~4-8 lan (dung tinh than "giam bang thong memory" cua
// llama.cpp), giai phong luon phan CPU-time danh cho vong for dequant.
// Layout phai khop CHINH XAC voi quant.nim (quantizeInt4Asymmetric /
// dequantizeTensorTransposed):
//   - wq: [N, K] row-major, nibble-packed 2 gia tri/byte, byte thap = phan
//     tu chan (k%2==0), byte cao = phan tu le (k%2==1).
//   - scales/zeros: neu groupSize>0, moi hang N chia thanh
//     ceil(K/groupSize) group, group[g] dung scales[n*nGroupsPerRow+g].
//   - c[m,n] = sum_k a[m,k] * ((nibble(n,k) - zero(n,g)) * scale(n,g))
//     -- dung 100% cong thuc trong dequantizeTensorTransposed() + beMatmul
//     hien tai, chi khac la khong vat chat hoa wT ra fp32 truoc.
// CHUA CHAY THU TREN GPU METAL THAT (moi truong sinh code la Linux, khong
// co macOS/Metal that de bien dich+chay) - viet bam sat dung cong thuc
// tham chieu CPU (quant.nim) nhung BAT BUOC phai tu build+test tren May
// that truoc khi dung cho sinh token that. Neu ra token rac, day la nghi
// ngo dau tien (sai offset nibble hoac sai thu tu group).
// SUA (theo yêu cầu "sửa cách đi như llama.cpp" - kernel CŨ bên dưới đã đổi
// tên còn matmul_q4_kernel_naive để tham khảo/rollback):
// Kernel CŨ: 1 thread = 1 phần tử output, tự đọc & cộng dồn HẾT K phần tử
// tuần tự (không SIMD reduction), dispatch threadgroup 16x1 = chỉ 16 thread/
// group -> occupancy cực thấp trên GPU thật (Iris có hàng trăm ALU rảnh).
// Kernel MỚI theo đúng chiến thuật kernel_mul_mv_q4_K của llama.cpp/ggml
// Metal backend:
//   1. Mỗi threadgroup = ĐÚNG 1 simdgroup (32 thread), xử lý MMQ4_NDST=4
//      hàng output (n) cùng lúc cho 1 hàng activation (m) - dispatch theo
//      (rowBlock = ceil(N/4), M).
//   2. Activation a[m,:] (K phần tử) được 32 thread NẠP 1 LẦN vào threadgroup
//      shared memory rồi DÙNG LẠI cho cả 4 hàng - trước đây mỗi hàng n tự
//      đọc lại a[] từ device memory (băng thông lãng phí gấp N/4 lần).
//   3. Mỗi hàng: 32 thread chia nhau K phần tử (k = lane, lane+32, ...),
//      cộng dồn partial rồi simd_sum() reduce chỉ trong 1 lệnh phần cứng
//      (thay vì 1 thread loop tuần tự K lần) - đây là phần tăng tốc chính.
// Kết quả: số threadgroup dispatch ra nhiều hơn N/4 lần (đủ để lấp đầy toàn
// bộ EU của GPU dù yếu), mỗi threadgroup làm việc song song thật (32-way)
// thay vì 1 thread gánh hết K.
// SUA BUG NGHIÊM TRỌNG (gây "GPU Timeout Error" thật trên máy Intel Iris):
// Bản trước dùng threadgroup shared memory kích thước K*4 byte để cache cả
// hàng activation - với K = intermediate_size (vd down_proj, K~11008+) thì
// cần ~44KB, VƯỢT giới hạn shared-memory/threadgroup của Iris (thường chỉ
// 16-32KB tuỳ gen). Metal không báo lỗi rõ ràng lúc setThreadgroupMemoryLength
// trên GPU cũ kiểu này - nó treo hẳn GPU (timeout), không phải lỗi biên dịch.
// SỬA: bỏ hẳn threadgroup memory. Đổi thứ tự vòng lặp: k làm vòng NGOÀI (mỗi
// thread đọc a[m,k] từ device memory ĐÚNG 1 LẦN, giữ trong 1 register cục bộ
// `aVal`), rr (MMQ4_NDST hàng) làm vòng TRONG dùng lại chính register đó -
// vẫn đạt mục tiêu "không đọc lại activation N/4 lần" như bản shared-memory,
// nhưng an toàn tuyệt đối vì không cần cấp phát shared memory nào cả.
#define MMQ4_NDST 4
#define MMQ4_SIMD_SIZE 32

kernel void matmul_q4_kernel(
    device const float* a          [[buffer(0)]],   // [M,K]
    device const uchar* wq         [[buffer(1)]],   // [N, ceil(K/2)] packed int4
    device const float* scales     [[buffer(2)]],   // [N * nGroupsPerRow]
    device const float* zeros      [[buffer(3)]],   // [N * nGroupsPerRow]
    device float* c                [[buffer(4)]],   // [M,N]
    device const int& M            [[buffer(5)]],
    device const int& K            [[buffer(6)]],
    device const int& N            [[buffer(7)]],
    device const int& groupSize    [[buffer(8)]],
    device const int& nGroupsPerRow [[buffer(9)]],
    uint3 tgpig [[threadgroup_position_in_grid]],   // (rowBlock, m, 0)
    uint tiisg  [[thread_index_in_simdgroup]]        // 0..31 - BẮT BUỘC dispatch threadgroup size = (32,1,1)
) {
    int m = int(tgpig.y);
    if (m >= M) return;
    int rowBlock = int(tgpig.x);

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

    for (int k = int(tiisg); k < K; k += MMQ4_SIMD_SIZE) {
        float aVal = a[m * K + k];              // đọc device memory ĐÚNG 1 LẦN
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
            float wval = (float(nibble) - zp) * sc;
            partial[rr] += aVal * wval;
        }
    }

    for (int rr = 0; rr < MMQ4_NDST; ++rr) {
        int n = n0 + rr;
        if (n >= N) continue;
        float rowSum = simd_sum(partial[rr]);
        if (tiisg == 0) {
            c[m * N + n] = rowSum;
        }
    }
}

// Kernel gốc (naive) giữ lại để đối chiếu/rollback nếu kernel simd-reduction
// phía trên ra kết quả sai trên máy thật (nghi ngờ đầu tiên: layout nibble
// hoặc offset group sai - xem cùng công thức y hệt, chỉ khác cách reduce).
kernel void matmul_q4_kernel_naive(
    device const float* a          [[buffer(0)]],
    device const uchar* wq         [[buffer(1)]],
    device const float* scales     [[buffer(2)]],
    device const float* zeros      [[buffer(3)]],
    device float* c                [[buffer(4)]],
    device const int& M            [[buffer(5)]],
    device const int& K            [[buffer(6)]],
    device const int& N            [[buffer(7)]],
    device const int& groupSize    [[buffer(8)]],
    device const int& nGroupsPerRow [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int n = int(gid.x);
    int m = int(gid.y);
    if (m >= M || n >= N) return;

    int bytesPerRow = (K + 1) / 2;
    int rowByteOff = n * bytesPerRow;
    int rowGroupOff = n * nGroupsPerRow;

    float sum = 0.0f;
    int curGroup = -1;
    float sc = 0.0f, zp = 0.0f;
    for (int k = 0; k < K; ++k) {
        int g = groupSize > 0 ? (k / groupSize) : 0;
        if (g != curGroup) {
            curGroup = g;
            sc = scales[rowGroupOff + g];
            zp = zeros[rowGroupOff + g];
        }
        uchar byteVal = wq[rowByteOff + (k >> 1)];
        uchar nibble = (k & 1) == 0 ? (byteVal & 0x0F) : ((byteVal >> 4) & 0x0F);
        float wval = (float(nibble) - zp) * sc;
        sum += a[m * K + k] * wval;
    }
    c[m * N + n] = sum;
}

// ============================================================
// ATOMIC ADD FLOAT
// ============================================================
inline void atomicAddFloatDevice(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    float expectedF = as_type<float>(expected);
    while (!atomic_compare_exchange_weak_explicit(
              addr, &expected,
              as_type<uint>(expectedF + val),
              memory_order_relaxed, memory_order_relaxed)) {
        expectedF = as_type<float>(expected);
    }
}

// ============================================================
// ATTENTION FUSED FORWARD
// ============================================================
kernel void attention_fused_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* o [[buffer(3)]],
    device float* s_matrix [[buffer(4)]],
    device const int& B [[buffer(5)]],
    device const int& H [[buffer(6)]],
    device const int& S [[buffer(7)]],
    device const int& D [[buffer(8)]],
    device const float& scale [[buffer(9)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint bh = position.x;
    uint ti = position.y;
    
    if (bh < (uint)(B * H) && ti < (uint)S) {
        uint base_idx = bh * S * D;
        uint base_s = bh * S * S;
        uint row = base_s + ti * S;

        // SUA: bo local array "float scores[256]" (tran bo nho neu S > 256,
        // vi day la mang co dinh tren stack cua thread, khong lien quan gi
        // toi kich thuoc thuc te cua S). Dung thang s_matrix (buffer global,
        // da duoc cap phat dung S*S theo tung lan goi thuc te tu phia Nim)
        // lam bo nho tam -> khong con gioi han cung nao ve S nua.
        float mx = -1e30f;
        for (uint tj = 0; tj <= ti; ++tj) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q[base_idx + ti * D + d] * k[base_idx + tj * D + d];
            }
            float sc = dot * scale;
            s_matrix[row + tj] = sc;
            if (sc > mx) mx = sc;
        }

        float sum_exp = 0.0f;
        for (uint tj = 0; tj <= ti; ++tj) {
            float e = exp(s_matrix[row + tj] - mx);
            s_matrix[row + tj] = e;
            sum_exp += e;
        }

        for (uint tj = 0; tj <= ti; ++tj) {
            s_matrix[row + tj] /= sum_exp;
        }
        for (uint tj = ti + 1; tj < (uint)S; ++tj) {
            s_matrix[row + tj] = 0.0f;
        }

        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (uint tj = 0; tj <= ti; ++tj) {
                acc += s_matrix[row + tj] * v[base_idx + tj * D + d];
            }
            o[base_idx + ti * D + d] = acc;
        }
    }
}

// ============================================================
// ATTENTION FUSED BACKWARD
// ============================================================
kernel void attention_fused_backward_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* s_matrix [[buffer(3)]],
    device const float* dy [[buffer(4)]],
    device float* dq [[buffer(5)]],
    device atomic_uint* dk [[buffer(6)]],
    device atomic_uint* dv [[buffer(7)]],
    device const int& B [[buffer(8)]],
    device const int& H [[buffer(9)]],
    device const int& S [[buffer(10)]],
    device const int& D [[buffer(11)]],
    device const float& scale [[buffer(12)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint bh = position.x;
    uint ti = position.y;
    
    if (bh < (uint)(B * H) && ti < (uint)S) {
        uint base_idx = bh * S * D;
        uint base_s = bh * S * S;
        
        float softmaxW[256];
        for (uint tj = 0; tj <= ti; ++tj) {
            softmaxW[tj] = s_matrix[base_s + ti * S + tj];
        }
        
        float dSoftmax[256];
        for (uint tj = 0; tj <= ti; ++tj) {
            float dotVal = 0.0f;
            for (int d = 0; d < D; ++d) {
                float dyVal = dy[base_idx + ti * D + d];
                atomicAddFloatDevice(&dv[base_idx + tj * D + d], softmaxW[tj] * dyVal);
                dotVal += dyVal * v[base_idx + tj * D + d];
            }
            dSoftmax[tj] = dotVal;
        }
        
        float dotSum = 0.0f;
        for (uint tj = 0; tj <= ti; ++tj) {
            dotSum += softmaxW[tj] * dSoftmax[tj];
        }
        
        for (uint tj = 0; tj <= ti; ++tj) {
            float dScore = softmaxW[tj] * (dSoftmax[tj] - dotSum) * scale;
            for (int d = 0; d < D; ++d) {
                dq[base_idx + ti * D + d] += dScore * k[base_idx + tj * D + d];
                atomicAddFloatDevice(&dk[base_idx + tj * D + d], dScore * q[base_idx + ti * D + d]);
            }
        }
    }
}