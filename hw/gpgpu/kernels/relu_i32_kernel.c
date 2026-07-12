#include "gpgpu_nn.h"
#include "gpgpu_builtins.h"
#include <stdint.h>

void _start(GPGPUReluArgs *args)
{
    uint32_t tx = gpgpu_thread_id_x();
    uint32_t bx = gpgpu_block_id_x();
    uint32_t bdx = gpgpu_block_dim_x();
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
