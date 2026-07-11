#include "gpgpu_nn.h"

#include <stdint.h>

#define GPGPU_CORE_CTRL_BASE 0x80000000u

#define CTRL_THREAD_ID_X 0x00u
#define CTRL_THREAD_ID_Y 0x04u
#define CTRL_THREAD_ID_Z 0x08u
#define CTRL_BLOCK_ID_X  0x10u
#define CTRL_BLOCK_ID_Y  0x14u
#define CTRL_BLOCK_ID_Z  0x18u

void _start(GPGPUIm2ColArgs *args)
{
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    uint32_t kw = ctrl[CTRL_THREAD_ID_X / sizeof(uint32_t)];
    uint32_t kh = ctrl[CTRL_THREAD_ID_Y / sizeof(uint32_t)];
    uint32_t ic = ctrl[CTRL_THREAD_ID_Z / sizeof(uint32_t)];
    uint32_t n = ctrl[CTRL_BLOCK_ID_X / sizeof(uint32_t)];
    uint32_t oh = ctrl[CTRL_BLOCK_ID_Y / sizeof(uint32_t)];
    uint32_t ow = ctrl[CTRL_BLOCK_ID_Z / sizeof(uint32_t)];
    uint32_t ih = oh * args->stride_h + kh;
    uint32_t iw = ow * args->stride_w + kw;
    uint32_t m = n * args->out_h * args->out_w + oh * args->out_w + ow;
    uint32_t k = ic * args->kernel_h * args->kernel_w +
                 kh * args->kernel_w + kw;
    int32_t *input = (int32_t *)(uintptr_t)args->input.data;
    int32_t *output = (int32_t *)(uintptr_t)args->output.data;

    output[m * args->output.c + k] =
        input[n * args->input.stride_n +
              ic * args->input.stride_c +
              ih * args->input.stride_h +
              iw * args->input.stride_w];

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
