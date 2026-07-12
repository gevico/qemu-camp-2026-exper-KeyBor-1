#include "gpgpu_nn.h"
#include "gpgpu_builtins.h"
#include <stdint.h>

void _start(GPGPUMoToNchwArgs *args)
{
    uint32_t ow = gpgpu_thread_id_x();
    uint32_t n = gpgpu_block_id_x();
    uint32_t c = gpgpu_block_id_y();
    uint32_t oh = gpgpu_block_id_z();
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
