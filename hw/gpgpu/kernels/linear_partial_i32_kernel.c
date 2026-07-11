#include "gpgpu_nn.h"

#include <stdint.h>

#define GPGPU_CORE_CTRL_BASE 0x80000000u

#define CTRL_THREAD_ID_X 0x00u
#define CTRL_BLOCK_ID_X  0x10u

void _start(GPGPULinearPartialArgs *args)
{
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    uint32_t in_feature = ctrl[CTRL_THREAD_ID_X / sizeof(uint32_t)];
    uint32_t out_feature = ctrl[CTRL_BLOCK_ID_X / sizeof(uint32_t)];
    uint32_t offset = out_feature * args->in_features + in_feature;
    int32_t *input = (int32_t *)(uintptr_t)args->input.data;
    int32_t *weight = (int32_t *)(uintptr_t)args->weight.data;
    int32_t *partial = (int32_t *)(uintptr_t)args->partial.data;

    partial[offset] = input[in_feature] * weight[offset];

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
