#include "gpgpu_nn.h"
#include "gpgpu_builtins.h"
#include <stdint.h>

void _start(GPGPUMatmulReduceArgs *args)
{
    uint32_t o = gpgpu_thread_id_x();
    uint32_t m = gpgpu_block_id_x();
    int32_t *partial = (int32_t *)(uintptr_t)args->partial.data;
    int32_t *c = (int32_t *)(uintptr_t)args->c.data;
    int32_t *bias = (int32_t *)(uintptr_t)args->bias.data;
    int32_t acc = 0;

    for (uint32_t k = 0; k < args->k; ++k) {
        acc += partial[(m * args->o + o) * args->k + k];
    }
    if (args->output_shift != 0) {
        acc >>= args->output_shift;
    }
    if (args->has_bias) {
        acc += bias[o];
    }
    c[m * args->o + o] = acc;

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
