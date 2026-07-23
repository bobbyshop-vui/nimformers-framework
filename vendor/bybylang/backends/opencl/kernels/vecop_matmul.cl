#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

// ============================================================
// VECTOR OPERATIONS
// ============================================================
__kernel void vecop_add(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}

__kernel void vecop_sub(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] - b[i];
}

__kernel void vecop_mul(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] * b[i];
}

__kernel void vecop_div(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] / b[i];
}

// ============================================================
// ACTIVATIONS
// ============================================================
__kernel void vecop_relu(__global const float* x, __global float* y) {
    int i = get_global_id(0);
    y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__kernel void vecop_sigmoid(__global const float* x, __global float* y) {
    int i = get_global_id(0);
    y[i] = 1.0f / (1.0f + exp(-x[i]));
}

__kernel void vecop_tanh(__global const float* x, __global float* y) {
    int i = get_global_id(0);
    float ev = exp(x[i]);
    float emv = exp(-x[i]);
    y[i] = (ev - emv) / (ev + emv);
}

__kernel void vecop_apflu(__global const float* x, __global float* y, const float alpha, const float beta) {
    int i = get_global_id(0);
    float val = x[i];
    y[i] = val > 0.0f ? val * (1.0f + alpha * val) : beta * val * exp(val);
}

__kernel void vecop_apflu_backward(__global const float* x, __global const float* dy, __global float* dx, const float alpha, const float beta) {
    int i = get_global_id(0);
    float val = x[i];
    float d = dy[i];
    dx[i] = val > 0.0f ? d * (1.0f + 2.0f * alpha * val) : d * beta * exp(val) * (1.0f + val);
}

// ============================================================
// SOFTMAX
// ============================================================
#define BB_WG 256

__kernel void softmax_kernel(__global const float* x, __global float* y, const int cols) {
    __local float red[BB_WG];
    int r = get_group_id(0);
    int tid = get_local_id(0);
    int off = r * cols;

    float m = -INFINITY;
    for (int c = tid; c < cols; c += BB_WG) {
        float v = x[off + c];
        m = v > m ? v : m;
    }
    red[tid] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmax(red[tid], red[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float maxVal = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float s = 0.0f;
    for (int c = tid; c < cols; c += BB_WG) {
        float e = exp(x[off + c] - maxVal);
        y[off + c] = e;
        s += e;
    }
    red[tid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = BB_WG / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float invSum = 1.0f / red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int c = tid; c < cols; c += BB_WG) {
        y[off + c] *= invSum;
    }
}

// ============================================================
// LAYERNORM
// ============================================================
__kernel void layernorm_kernel(__global const float* x, __global const float* gamma, __global const float* beta, __global float* y, const int cols, const float eps) {
    __local float red[BB_WG];
    int r = get_group_id(0);
    int tid = get_local_id(0);
    int off = r * cols;

    float sum = 0.0f, sumSq = 0.0f;
    for (int c = tid; c < cols; c += BB_WG) {
        float v = x[off + c];
        sum += v;
        sumSq += v * v;
    }
    red[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = red[0] / (float)cols;
    barrier(CLK_LOCAL_MEM_FENCE);

    red[tid] = sumSq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float meanSq = red[0] / (float)cols;
    float varr = meanSq - mean * mean;
    float invStd = 1.0f / sqrt(varr + eps);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int c = tid; c < cols; c += BB_WG) {
        y[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c];
    }
}

// ============================================================
// LAYERNORM BACKWARD
// ============================================================
__kernel void layernorm_backward_kernel(__global const float* dy, __global const float* x, __global const float* gamma, __global const float* beta, __global float* dx, __global float* dgamma, __global float* dbeta, const int rows, const int cols, const float eps) {
    int id = get_global_id(0);
    if (id < cols) {
        float dg = 0.0f;
        float db = 0.0f;
        for (int r = 0; r < rows; ++r) {
            int off = r * cols;
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) mean += x[off + c];
            mean /= (float)cols;
            float varr = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float diff = x[off + c] - mean;
                varr += diff * diff;
            }
            varr /= (float)cols;
            float invStd = 1.0f / sqrt(varr + eps);
            
            float norm = (x[off + id] - mean) * invStd;
            dg += dy[off + id] * norm;
            db += dy[off + id];
        }
        dgamma[id] = dg;
        dbeta[id] = db;
    }
    
    int row = get_global_id(0);
    if (row < rows) {
        int off = row * cols;
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += x[off + c];
        mean /= (float)cols;
        float varr = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float diff = x[off + c] - mean;
            varr += diff * diff;
        }
        varr /= (float)cols;
        float invStd = 1.0f / sqrt(varr + eps);
        
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float grad = dy[off + c] * gamma[c] * invStd;
            sum1 += grad;
            sum2 += grad * (x[off + c] - mean);
        }
        sum2 *= (-invStd * invStd / (float)cols);
        
        for (int c = 0; c < cols; ++c) {
            float term1 = dy[off + c] * gamma[c] * invStd;
            float term2 = sum1 / (float)cols;
            float term3 = (x[off + c] - mean) * sum2 * 2.0f / (float)cols;
            dx[off + c] = term1 - term2 + term3;
        }
    }
}

// ============================================================
// EMBEDDING LOOKUP
// ============================================================
__kernel void embedding_lookup_kernel(__global const float* table, __global const int* indices, __global float* y, const int vocab, const int dim) {
    int i = get_global_id(0);
    int idx = indices[i];
    if (idx >= 0 && idx < vocab) {
        for (int j = 0; j < dim; ++j) {
            y[i * dim + j] = table[idx * dim + j];
        }
    } else {
        for (int j = 0; j < dim; ++j) {
            y[i * dim + j] = 0.0f;
        }
    }
}

// ============================================================
// MATMUL TILED
// ============================================================
// SUA: tang tu 16 len 32 (giong metal_shim.m / vecop_matmul.metal) - tile
// lon hon giam so workgroup phai dispatch va tang data reuse tren local mem
#define BB_TILE 32

__kernel void matmul_naive(__global const float* a, __global const float* b, __global float* c,
                            const int M, const int K, const int N) {
    __local float tileA[BB_TILE][BB_TILE];
    __local float tileB[BB_TILE][BB_TILE];

    int tx = get_local_id(0), ty = get_local_id(1);
    int row = get_group_id(0) * BB_TILE + ty;
    int col = get_group_id(1) * BB_TILE + tx;

    float sum = 0.0f;
    int numTiles = (K + BB_TILE - 1) / BB_TILE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * BB_TILE + tx;
        int bRow = t * BB_TILE + ty;
        tileA[ty][tx] = (row < M && aCol < K) ? a[row * K + aCol] : 0.0f;
        tileB[ty][tx] = (bRow < K && col < N) ? b[bRow * N + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int p = 0; p < BB_TILE; p++) {
            sum += tileA[ty][p] * tileB[p][tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

// ============================================================
// MATMUL TRUC TIEP TREN INT4 ASYMMETRIC (packed, per-group scale/zero_point)
// ============================================================
// SUA: ban sao chinh xac cua matmul_q4_kernel ben Metal (vecop_matmul.metal)
// - xem chu thich chi tiet o do va quant.nim/dequantizeTensorTransposed cho
// dinh nghia layout. Thread (get_global_id(0)=n, get_global_id(1)=m).
__kernel void matmul_q4_naive(__global const float* a, __global const uchar* wq,
                               __global const float* scales, __global const float* zeros,
                               __global float* c,
                               const int M, const int K, const int N,
                               const int groupSize, const int nGroupsPerRow) {
    int n = get_global_id(0);
    int m = get_global_id(1);
    if (m >= M || n >= N) return;

    int bytesPerRow = (K + 1) / 2;
    int rowByteOff = n * bytesPerRow;
    int rowGroupOff = n * nGroupsPerRow;

    float sum = 0.0f;
    int curGroup = -1;
    float sc = 0.0f, zp = 0.0f;
    for (int k = 0; k < K; k++) {
        int g = groupSize > 0 ? (k / groupSize) : 0;
        if (g != curGroup) {
            curGroup = g;
            sc = scales[rowGroupOff + g];
            zp = zeros[rowGroupOff + g];
        }
        uchar byteVal = wq[rowByteOff + (k >> 1)];
        uchar nibble = (k & 1) == 0 ? (byteVal & 0x0F) : ((byteVal >> 4) & 0x0F);
        float wval = (((float)nibble) - zp) * sc;
        sum += a[m * K + k] * wval;
    }
    c[m * N + n] = sum;
}

// ============================================================
// ATOMIC ADD FLOAT
// ============================================================
inline void atomicAddFloatGlobal(volatile __global float* addr, float val) {
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

// ============================================================
// ATTENTION FUSED FORWARD
// ============================================================
__kernel void attention_fused_kernel(__global const float* q, __global const float* k, __global const float* v, __global float* o, __global float* s_matrix, const int B, const int H, const int S, const int D, const float scale) {
    int bh = get_global_id(0);
    int ti = get_global_id(1);
    
    if (bh < (B * H) && ti < S) {
        int base_idx = bh * S * D;
        int base_s = bh * S * S;
        int row = base_s + ti * S;

        // SUA: bo local array "float scores[256]" (tran neu S > 256). Dung
        // thang s_matrix (global, da cap phat dung S*S theo tung lan goi)
        // lam bo nho tam -> khong con gioi han cung nao ve S nua.
        float mx = -1e30f;
        for (int tj = 0; tj <= ti; ++tj) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q[base_idx + ti * D + d] * k[base_idx + tj * D + d];
            }
            float sc = dot * scale;
            s_matrix[row + tj] = sc;
            if (sc > mx) mx = sc;
        }

        float sum_exp = 0.0f;
        for (int tj = 0; tj <= ti; ++tj) {
            float e = exp(s_matrix[row + tj] - mx);
            s_matrix[row + tj] = e;
            sum_exp += e;
        }

        for (int tj = 0; tj <= ti; ++tj) {
            s_matrix[row + tj] /= sum_exp;
        }
        for (int tj = ti + 1; tj < S; ++tj) {
            s_matrix[row + tj] = 0.0f;
        }

        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int tj = 0; tj <= ti; ++tj) {
                acc += s_matrix[row + tj] * v[base_idx + tj * D + d];
            }
            o[base_idx + ti * D + d] = acc;
        }
    }
}

// ============================================================
// ATTENTION FUSED BACKWARD
// ============================================================
__kernel void attention_fused_backward_kernel(__global const float* q, __global const float* k, __global const float* v, __global const float* s_matrix, __global const float* dy, __global float* dq, __global float* dk, __global float* dv, const int B, const int H, const int S, const int D, const float scale) {
    int bh = get_global_id(0);
    int ti = get_global_id(1);
    
    if (bh < (B * H) && ti < S) {
        int base_idx = bh * S * D;
        int base_s = bh * S * S;
        
        float softmaxW[256];
        for (int tj = 0; tj <= ti; ++tj) {
            softmaxW[tj] = s_matrix[base_s + ti * S + tj];
        }
        
        float dSoftmax[256];
        for (int tj = 0; tj <= ti; ++tj) {
            float dotVal = 0.0f;
            for (int d = 0; d < D; ++d) {
                float dyVal = dy[base_idx + ti * D + d];
                atomicAddFloatGlobal(&dv[base_idx + tj * D + d], softmaxW[tj] * dyVal);
                dotVal += dyVal * v[base_idx + tj * D + d];
            }
            dSoftmax[tj] = dotVal;
        }
        
        float dotSum = 0.0f;
        for (int tj = 0; tj <= ti; ++tj) {
            dotSum += softmaxW[tj] * dSoftmax[tj];
        }
        
        for (int tj = 0; tj <= ti; ++tj) {
            float dScore = softmaxW[tj] * (dSoftmax[tj] - dotSum) * scale;
            for (int d = 0; d < D; ++d) {
                dq[base_idx + ti * D + d] += dScore * k[base_idx + tj * D + d];
                atomicAddFloatGlobal(&dk[base_idx + tj * D + d], dScore * q[base_idx + ti * D + d]);
            }
        }
    }
}