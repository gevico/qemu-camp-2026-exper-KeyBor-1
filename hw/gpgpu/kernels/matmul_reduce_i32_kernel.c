#include "gpgpu_nn.h"

#include <stdint.h>

#define GPGPU_CORE_CTRL_BASE 0x80000000u

#define CTRL_THREAD_ID_X 0x00u
#define CTRL_BLOCK_ID_X  0x10u

void _start(GPGPUMatmulReduceArgs *args)
{
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    uint32_t o = ctrl[CTRL_THREAD_ID_X / sizeof(uint32_t)];
    uint32_t m = ctrl[CTRL_BLOCK_ID_X / sizeof(uint32_t)];
    int32_t *partial = (int32_t *)(uintptr_t)args->partial.data;
    int32_t *c = (int32_t *)(uintptr_t)args->c.data;
    int32_t acc = 0;

    for (uint32_t k = 0; k < args->k; ++k) {
        acc += partial[(m * args->o + o) * args->k + k];
    }
    c[m * args->o + o] = acc;

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
