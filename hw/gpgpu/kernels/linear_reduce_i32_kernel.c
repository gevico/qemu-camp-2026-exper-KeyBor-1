#include "gpgpu_nn.h"

#include <stdint.h>

#define GPGPU_CORE_CTRL_BASE 0x80000000u

#define CTRL_THREAD_ID_X 0x00u
#define CTRL_BLOCK_ID_X  0x10u
#define CTRL_BLOCK_DIM_X 0x20u

void _start(GPGPULinearReduceArgs *args)
{
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    uint32_t tx = ctrl[CTRL_THREAD_ID_X / sizeof(uint32_t)];
    uint32_t bx = ctrl[CTRL_BLOCK_ID_X / sizeof(uint32_t)];
    uint32_t bdx = ctrl[CTRL_BLOCK_DIM_X / sizeof(uint32_t)];
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
