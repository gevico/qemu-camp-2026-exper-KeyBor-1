#include "gpgpu_nn.h"
#include "gpgpu_builtins.h"
#include <stdint.h>

void _start(GPGPULinearReduceArgs *args)
{
    uint32_t tx = gpgpu_thread_id_x();
    uint32_t bx = gpgpu_block_id_x();
    uint32_t bdx = gpgpu_block_dim_x();
    uint32_t out_feature = bx * bdx + tx;
    int32_t *partial = (int32_t *)(uintptr_t)args->partial.data;
    int32_t *bias = (int32_t *)(uintptr_t)args->bias.data;
    int32_t *output = (int32_t *)(uintptr_t)args->output.data;
    int32_t acc = bias[out_feature];

    for (uint32_t i = 0; i < args->in_features; ++i) {
        acc += partial[out_feature * args->in_features + i];
    }
    output[out_feature] = acc;

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
