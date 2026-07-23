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
// MATMUL TILED
// ============================================================
// SỬA: tăng từ 16 lên 32 - tile 16x16 quá nhỏ để tận dụng iGPU Intel
// (quá nhiều threadgroup nhỏ, mỗi threadgroup ít việc, overhead dispatch
// tương đối cao). Tile 32x32 giảm số threadgroup cần dispatch 4 lần, tăng
// số phép tính trên mỗi lần nạp tileA/tileB vào threadgroup memory (data
// reuse tốt hơn) - gần với hướng đi của llama.cpp (dùng tile lớn / simdgroup
// thay vì tile nhỏ cố định). 32*32*4 byte * 2 tile = 8KB threadgroup memory,
// vẫn nằm trong giới hạn 32KB threadgroup memory của GPU Apple/Intel nên an
// toàn không tràn.
// SUA (fix GPU Timeout tren Intel Iris/iGPU yeu): 32x32=1024 threads/threadgroup
// bi hardcode trong metal_shim.m (metal_matmul) MA KHONG kiem tra
// pipeline.maxTotalThreadsPerThreadgroup truoc khi dispatch (khac voi
// metal_vecop cung file co MIN(...) tu te). Tren iGPU yeu, threadgroup memory
// 8KB (tileA+tileB) + vong lap unroll 32 buoc co the khien compiler gioi han
// maxTotalThreadsPerThreadgroup cua pipeline nay XUONG DUOI 1024 -> dispatch
// vuot gioi han that -> driver Intel treo cung thay vi bao loi API ngay ->
// GPU watchdog kill sau vai giay (dung loi "GPU Timeout" ban gap). Ha xuong
// 8 (64 threads/threadgroup) de an toan tren moi GPU, doi lai groupSize
// trong metal_shim.m (metal_matmul VA metal_matmul_enc) tu MTLSizeMake(32,32,1)
// thanh MTLSizeMake(8,8,1) cho khop.
#define BB_TILE 8

kernel void matmul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    device const int& m [[buffer(3)]],
    device const int& k [[buffer(4)]],
    device const int& n [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    threadgroup float tileA[BB_TILE][BB_TILE];
    threadgroup float tileB[BB_TILE][BB_TILE];

    int row = int(gid.y);
    int col = int(gid.x);
    int tx = int(tid.x), ty = int(tid.y);

    float sum = 0.0f;
    int numTiles = (k + BB_TILE - 1) / BB_TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * BB_TILE + tx;
        int bRow = t * BB_TILE + ty;
        tileA[ty][tx] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;
        tileB[ty][tx] = (bRow < k && col < n) ? b[bRow * n + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int p = 0; p < BB_TILE; ++p) {
            sum += tileA[ty][p] * tileB[p][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

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