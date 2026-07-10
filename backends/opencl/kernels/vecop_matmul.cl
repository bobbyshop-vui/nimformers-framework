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

__kernel void softmax_kernel(__global const float* x, __global float* y, const int cols) {
    int r = get_global_id(0);
    int off = r * cols;
    float maxVal = x[off];
    for (int c = 1; c < cols; c++) {
        if (x[off + c] > maxVal) {
            maxVal = x[off + c];
        }
    }
    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        float e = exp(x[off + c] - maxVal);
        y[off + c] = e;
        sum += e;
    }
    for (int c = 0; c < cols; c++) {
        y[off + c] /= sum;
    }
}

__kernel void layernorm_kernel(__global const float* x, __global const float* gamma, __global const float* beta, __global float* y, const int cols, const float eps) {
    int r = get_global_id(0);
    int off = r * cols;
    float mean = 0.0f;
    for (int c = 0; c < cols; c++) {
        mean += x[off + c];
    }
    mean /= (float)cols;
    float varr = 0.0f;
    for (int c = 0; c < cols; c++) {
        float diff = x[off + c] - mean;
        varr += diff * diff;
    }
    varr /= (float)cols;
    float invStd = 1.0f / sqrt(varr + eps);
    for (int c = 0; c < cols; c++) {
        y[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c];
    }
}

__kernel void embedding_lookup_kernel(__global const float* table, __global const int* indices, __global float* y, const int vocab, const int dim) {
    int i = get_global_id(0);
    int idx = indices[i];
    if (idx >= 0 && idx < vocab) {
        for (int j = 0; j < dim; j++) {
            y[i * dim + j] = table[idx * dim + j];
        }
    } else {
        for (int j = 0; j < dim; j++) {
            y[i * dim + j] = 0.0f;
        }
    }
}

__kernel void matmul_naive(__global const float* a, __global const float* b, __global float* c,
                            const int M, const int K, const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int p = 0; p < K; p++) {
        sum += a[row * K + p] * b[p * N + col];
    }
    c[row * N + col] = sum;
}
