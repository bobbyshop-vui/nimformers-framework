#include <metal_stdlib>
using namespace metal;

kernel void add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) c[idx] = a[idx] + b[idx];
}

kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= M || gid.y >= N) return;
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[gid.x * K + k] * B[k * N + gid.y];
    }
    C[gid.x * N + gid.y] = sum;
}

kernel void relu_activation(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) out[idx] = max(x[idx], 0.0f);
}

kernel void sigmoid_activation(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) out[idx] = 1.0f / (1.0f + exp(-x[idx]));
}

kernel void tanh_activation(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) out[idx] = tanh(x[idx]);
}

kernel void softmax(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    uint start = row * cols;
    float max_val = x[start];
    for (uint j = 1; j < cols; j++) max_val = max(max_val, x[start + j]);
    float sum = 0.0;
    for (uint j = 0; j < cols; j++) {
        out[start + j] = exp(x[start + j] - max_val);
        sum += out[start + j];
    }
    for (uint j = 0; j < cols; j++) out[start + j] /= sum;
}

kernel void layernorm(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    uint start = row * cols;
    float mean = 0.0;
    for (uint j = 0; j < cols; j++) mean += x[start + j];
    mean /= cols;
    float var = 0.0;
    for (uint j = 0; j < cols; j++) {
        float diff = x[start + j] - mean;
        var += diff * diff;
    }
    var /= cols;
    float inv_std = 1.0 / sqrt(var + eps);
    for (uint j = 0; j < cols; j++) {
        out[start + j] = (x[start + j] - mean) * inv_std * gamma[j] + beta[j];
    }
}

kernel void embedding_lookup(
    device const float* table [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant uint& num [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= num || gid.y >= dim) return;
    int idx = indices[gid.x];
    if (idx >= 0 && idx < vocab) {
        out[gid.x * dim + gid.y] = table[idx * dim + gid.y];
    } else {
        out[gid.x * dim + gid.y] = 0.0;
    }
}