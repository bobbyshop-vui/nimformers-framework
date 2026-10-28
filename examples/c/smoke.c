/* Minimal C consumer of the Nimformers C ABI. */
#include "nimformers.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("abi=%u version=%s\n", nimf_abi_version(), nimf_version());

    NimfContext* ctx = nimf_context_create(NIMF_BACKEND_AUTO);
    if (!ctx) {
        fprintf(stderr, "context: %s\n", nimf_last_error());
        return 1;
    }
    printf("backend=%s\n", nimf_context_backend_name(ctx));

    const float a[6] = {1, 2, 3, 4, 5, 6};
    const float b[6] = {10, 20, 30, 40, 50, 60};
    float out[6];
    if (nimf_add(ctx, a, b, out, 6) != 0) {
        fprintf(stderr, "add: %s\n", nimf_last_error());
        return 1;
    }
    for (int i = 0; i < 6; ++i) printf("%.0f ", out[i]);
    printf("\n");

    const int64_t shape[2] = {2, 3};
    NimfTensor* t = nimf_tensor_create_shaped(a, 6, shape, 2);
    if (!t) {
        fprintf(stderr, "tensor: %s\n", nimf_last_error());
        return 1;
    }
    printf("tensor rank=%lld numel=%lld data0=%.0f\n",
           (long long)nimf_tensor_rank(t),
           (long long)nimf_tensor_numel(t),
           (double)nimf_tensor_data(t)[0]);
    nimf_tensor_destroy(t);

    nimf_context_destroy(ctx);
    return 0;
}
