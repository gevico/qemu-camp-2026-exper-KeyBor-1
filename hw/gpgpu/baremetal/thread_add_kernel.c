#include <stdint.h>

#define GPGPU_CORE_CTRL_BASE 0x80000000u

#define CTRL_THREAD_ID_X 0x00u
#define CTRL_THREAD_ID_Y 0x04u
#define CTRL_THREAD_ID_Z 0x08u
#define CTRL_BLOCK_ID_X  0x10u
#define CTRL_BLOCK_ID_Y  0x14u
#define CTRL_BLOCK_ID_Z  0x18u

static uint32_t ctrl_read32(uint32_t offset)
{
    volatile uint32_t *ptr =
        (volatile uint32_t *)(GPGPU_CORE_CTRL_BASE + offset);

    return *ptr;
}

void _start(uint32_t *args)
{
    volatile uint32_t *out = (volatile uint32_t *)(uintptr_t)args[0];
    uint32_t tx = ctrl_read32(CTRL_THREAD_ID_X);
    uint32_t ty = ctrl_read32(CTRL_THREAD_ID_Y);
    uint32_t tz = ctrl_read32(CTRL_THREAD_ID_Z);
    uint32_t bx = ctrl_read32(CTRL_BLOCK_ID_X);
    uint32_t by = ctrl_read32(CTRL_BLOCK_ID_Y);
    uint32_t bz = ctrl_read32(CTRL_BLOCK_ID_Z);
    uint32_t local_id = tx + (ty << 1) + (tz << 2);
    uint32_t block_id = bx + (by << 1) + (bz << 2);
    uint32_t global_id = local_id + (block_id << 3);

    out[global_id] += global_id;

    __asm__ volatile("ebreak");
    __builtin_unreachable();
}
