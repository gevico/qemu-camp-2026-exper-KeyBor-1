#include "gpgpu_nn.h"
#include "gpgpu_builtins.h"
#include <stdint.h>

void _start(GPGPUMatmulPartialArgs *args)
{
    uint32_t k = gpgpu_thread_id_x();
    uint32_t m = gpgpu_block_id_x();
    uint32_t o = gpgpu_block_id_y();
    uint32_t a_offset = m * args->k + k;
    uint32_t b_offset = k * args->o + o;
    uint32_t partial_offset = (m * args->o + o) * args->k + k;
    int32_t *a = (int32_t *)(uintptr_t)args->a.data;
    int32_t *b = (int32_t *)(uintptr_t)args->b.data;
    int32_t *partial = (int32_t *)(uintptr_t)args->partial.data;

    partial[partial_offset] = a[a_offset] * b[b_offset];

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
