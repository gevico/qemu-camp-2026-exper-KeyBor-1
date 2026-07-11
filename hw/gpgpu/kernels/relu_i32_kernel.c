#include "gpgpu_nn.h"

#include <stdint.h>

#define GPGPU_CORE_CTRL_BASE 0x80000000u

#define CTRL_THREAD_ID_X 0x00u
#define CTRL_BLOCK_ID_X  0x10u
#define CTRL_BLOCK_DIM_X 0x20u

void _start(GPGPUReluArgs *args)
{
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    uint32_t tx = ctrl[CTRL_THREAD_ID_X / sizeof(uint32_t)];
    uint32_t bx = ctrl[CTRL_BLOCK_ID_X / sizeof(uint32_t)];
    uint32_t bdx = ctrl[CTRL_BLOCK_DIM_X / sizeof(uint32_t)];
    uint32_t global_id = bx * bdx + tx;
    uint32_t numel = args->input.n * args->input.c *
                     args->input.h * args->input.w;
    int32_t *input = (int32_t *)(uintptr_t)args->input.data;
    int32_t *output = (int32_t *)(uintptr_t)args->output.data;
    int32_t value;

    if (global_id < numel) {
        value = input[global_id];
        output[global_id] = value < 0 ? 0 : value;
    }

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
