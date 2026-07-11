#include "gpgpu_nn.h"

#include <stdint.h>

#define GPGPU_CORE_CTRL_BASE 0x80000000u

#define CTRL_THREAD_ID_X 0x00u
#define CTRL_BLOCK_ID_X  0x10u
#define CTRL_BLOCK_ID_Y  0x14u

void _start(GPGPUMatmulPartialArgs *args)
{
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    uint32_t k = ctrl[CTRL_THREAD_ID_X / sizeof(uint32_t)];
    uint32_t m = ctrl[CTRL_BLOCK_ID_X / sizeof(uint32_t)];
    uint32_t o = ctrl[CTRL_BLOCK_ID_Y / sizeof(uint32_t)];
    uint32_t a_offset = m * args->a.stride_n + k * args->a.stride_c;
    uint32_t b_offset = k * args->b.stride_n + o * args->b.stride_c;
    uint32_t partial_offset = (m * args->o + o) * args->k + k;
    int32_t *a = (int32_t *)(uintptr_t)args->a.data;
    int32_t *b = (int32_t *)(uintptr_t)args->b.data;
    int32_t *partial = (int32_t *)(uintptr_t)args->partial.data;

    partial[partial_offset] = a[a_offset] * b[b_offset];

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
