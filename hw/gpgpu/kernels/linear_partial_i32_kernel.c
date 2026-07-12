#include "gpgpu_nn.h"
#include "gpgpu_builtins.h"
#include <stdint.h>

void _start(GPGPULinearPartialArgs *args)
{
    uint32_t in_feature = gpgpu_thread_id_x();
    uint32_t out_feature = gpgpu_block_id_x();
    uint32_t offset = out_feature * args->in_features + in_feature;
    int32_t *input = (int32_t *)(uintptr_t)args->input.data;
    int32_t *weight = (int32_t *)(uintptr_t)args->weight.data;
    int32_t *partial = (int32_t *)(uintptr_t)args->partial.data;

    partial[offset] = input[in_feature] * weight[offset];

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
