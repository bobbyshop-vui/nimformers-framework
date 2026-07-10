#include <metal_stdlib>
using namespace metal;

kernel void vecop_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] + b[id];
}

kernel void vecop_sub(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] - b[id];
}

kernel void vecop_mul(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] * b[id];
}

kernel void vecop_div(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] / b[id];
}

kernel void vecop_relu(device const float* x [[buffer(0)]],
                       device float* y [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    y[id] = x[id] > 0.0 ? x[id] : 0.0;
}

kernel void vecop_sigmoid(device const float* x [[buffer(0)]],
                          device float* y [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    y[id] = 1.0 / (1.0 + exp(-x[id]));
}

kernel void vecop_tanh(device const float* x [[buffer(0)]],
                       device float* y [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    y[id] = tanh(x[id]);
}

kernel void softmax_kernel(device const float* x [[buffer(0)]],
                           device float* y [[buffer(1)]],
                           constant int& cols [[buffer(2)]],
                           uint r [[thread_position_in_grid]]) {
    int off = r * cols;
    float maxVal = x[off];
    for (int c = 1; c < cols; c++) {
        if (x[off + c] > maxVal) {
            maxVal = x[off + c];
        }
    }
    float sum = 0.0;
    for (int c = 0; c < cols; c++) {
        float e = exp(x[off + c] - maxVal);
        y[off + c] = e;
        sum += e;
    }
    for (int c = 0; c < cols; c++) {
        y[off + c] /= sum;
    }
}

kernel void layernorm_kernel(device const float* x [[buffer(0)]],
                             device const float* gamma [[buffer(1)]],
                             device const float* beta [[buffer(2)]],
                             device float* y [[buffer(3)]],
                             constant int& cols [[buffer(4)]],
                             constant float& eps [[buffer(5)]],
                             uint r [[thread_position_in_grid]]) {
    int off = r * cols;
    float mean = 0.0;
    for (int c = 0; c < cols; c++) {
        mean += x[off + c];
    }
    mean /= (float)cols;
    float varr = 0.0;
    for (int c = 0; c < cols; c++) {
        float diff = x[off + c] - mean;
        varr += diff * diff;
    }
    varr /= (float)cols;
    float invStd = 1.0 / sqrt(varr + eps);
    for (int c = 0; c < cols; c++) {
        y[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c];
    }
}

kernel void embedding_lookup_kernel(device const float* table [[buffer(0)]],
                                    device const int* indices [[buffer(1)]],
                                    device float* y [[buffer(2)]],
                                    constant int& vocab [[buffer(3)]],
                                    constant int& dim [[buffer(4)]],
                                    uint i [[thread_position_in_grid]]) {
    int idx = indices[i];
    if (idx >= 0 && idx < vocab) {
        for (int j = 0; j < dim; j++) {
            y[i * dim + j] = table[idx * dim + j];
        }
    } else {
        for (int j = 0; j < dim; j++) {
            y[i * dim + j] = 0.0;
        }
    }
}

kernel void matmul_naive(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* c [[buffer(2)]],
                           constant int& M [[buffer(3)]],
                           constant int& K [[buffer(4)]],
                           constant int& N [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= (uint)N || gid.y >= (uint)M) return;
    float sum = 0.0;
    for (int p = 0; p < K; p++) {
        sum += a[gid.y * (uint)K + p] * b[(uint)p * (uint)N + gid.x];
    }
    c[gid.y * (uint)N + gid.x] = sum;
}
