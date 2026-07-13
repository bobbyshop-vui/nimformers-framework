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
#define BB_TILE 16

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
        
        float scores[256];
        float mx = -1e30f;
        for (uint tj = 0; tj <= ti; ++tj) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q[base_idx + ti * D + d] * k[base_idx + tj * D + d];
            }
            scores[tj] = dot * scale;
            if (scores[tj] > mx) mx = scores[tj];
        }
        
        float sum_exp = 0.0f;
        for (uint tj = 0; tj <= ti; ++tj) {
            scores[tj] = exp(scores[tj] - mx);
            sum_exp += scores[tj];
        }
        
        for (uint tj = 0; tj <= ti; ++tj) {
            scores[tj] /= sum_exp;
            s_matrix[base_s + ti * S + tj] = scores[tj];
        }
        for (uint tj = ti + 1; tj < (uint)S; ++tj) {
            s_matrix[base_s + ti * S + tj] = 0.0f;
        }
        
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (uint tj = 0; tj <= ti; ++tj) {
                acc += scores[tj] * v[base_idx + tj * D + d];
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