#include "gpgpu_nn.h"

#include <stdint.h>

#define GPGPU_CORE_CTRL_BASE 0x80000000u

#define CTRL_THREAD_ID_X 0x00u
#define CTRL_BLOCK_ID_X  0x10u
#define CTRL_BLOCK_ID_Y  0x14u
#define CTRL_BLOCK_ID_Z  0x18u

void _start(GPGPUMoToNchwArgs *args)
{
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    uint32_t ow = ctrl[CTRL_THREAD_ID_X / sizeof(uint32_t)];
    uint32_t n = ctrl[CTRL_BLOCK_ID_X / sizeof(uint32_t)];
    uint32_t c = ctrl[CTRL_BLOCK_ID_Y / sizeof(uint32_t)];
    uint32_t oh = ctrl[CTRL_BLOCK_ID_Z / sizeof(uint32_t)];
    uint32_t m = n * args->h * args->w + oh * args->w + ow;
    int32_t *input = (int32_t *)(uintptr_t)args->input.data;
    int32_t *output = (int32_t *)(uintptr_t)args->output.data;

    output[n * args->output.stride_n +
           c * args->output.stride_c +
           oh * args->output.stride_h +
           ow * args->output.stride_w] = input[m * args->c + c];

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
