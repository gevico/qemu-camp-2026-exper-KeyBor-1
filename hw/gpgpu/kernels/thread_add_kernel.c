#include "gpgpu_builtins.h"
#include <stdint.h>

void _start(uint32_t *args)
{
    volatile uint32_t *out = (volatile uint32_t *)(uintptr_t)args[0];
    uint32_t scale = args[1];
    uint32_t iters = args[2];
    uint32_t tx = gpgpu_thread_id_x();
    uint32_t ty = gpgpu_thread_id_y();
    uint32_t tz = gpgpu_thread_id_z();
    uint32_t bx = gpgpu_block_id_x();
    uint32_t by = gpgpu_block_id_y();
    uint32_t bz = gpgpu_block_id_z();
    uint32_t local_id = tx + (ty << 1) + (tz << 2);
    uint32_t block_id = bx + (by << 1) + (bz << 2);
    uint32_t global_id = local_id + (block_id << 3);

    for (uint32_t i = 0; i < iters; ++i) {
        out[global_id] += global_id * scale;
    }

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
