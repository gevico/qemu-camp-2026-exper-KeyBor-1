/*
 * Minimal GPGPU runtime implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "gpgpu_runtime.h"
#include "gpgpu_freestanding.h"

#define GPGPU_REG_GLOBAL_CTRL       0x0100
#define GPGPU_REG_GLOBAL_STATUS     0x0104
#define GPGPU_REG_ERROR_STATUS      0x0108
#define GPGPU_REG_KERNEL_ADDR_LO    0x0300
#define GPGPU_REG_KERNEL_ADDR_HI    0x0304
#define GPGPU_REG_KERNEL_ARGS_LO    0x0308
#define GPGPU_REG_KERNEL_ARGS_HI    0x030c
#define GPGPU_REG_GRID_DIM_X        0x0310
#define GPGPU_REG_GRID_DIM_Y        0x0314
#define GPGPU_REG_GRID_DIM_Z        0x0318
#define GPGPU_REG_BLOCK_DIM_X       0x031c
#define GPGPU_REG_BLOCK_DIM_Y       0x0320
#define GPGPU_REG_BLOCK_DIM_Z       0x0324
#define GPGPU_REG_DISPATCH          0x0330

#define GPGPU_CTRL_ENABLE           (1u << 0)
#define GPGPU_STATUS_BUSY           (1u << 1)
#define GPGPU_STATUS_ERROR          (1u << 2)

static uint32_t gpgpu_align_up_u32(uint32_t value, uint32_t align)
{
    return (value + align - 1u) & ~(align - 1u);
}

static int gpgpu_check_dev(const GPGPURuntimeDevice *dev)
{
    if (!dev || !dev->ctrl || !dev->vram || dev->vram_size == 0) {
        return -EINVAL;
    }

    return 0;
}

static int gpgpu_check_range(const GPGPURuntimeDevice *dev, uint32_t offset,
                             size_t size)
{
    int ret = gpgpu_check_dev(dev);

    if (ret < 0) {
        return ret;
    }
    if (offset > dev->vram_size || size > dev->vram_size - offset) {
        return -ERANGE;
    }

    return 0;
}

static void gpgpu_mmio_write32(GPGPURuntimeDevice *dev, uint32_t reg,
                               uint32_t value)
{
    volatile uint32_t *ptr = (volatile uint32_t *)(dev->ctrl + reg);

    *ptr = value;
}

static uint32_t gpgpu_mmio_read32(GPGPURuntimeDevice *dev, uint32_t reg)
{
    volatile uint32_t *ptr = (volatile uint32_t *)(dev->ctrl + reg);

    return *ptr;
}

int gpgpu_runtime_init(GPGPURuntimeDevice *dev, volatile void *ctrl_bar,
                       void *vram_bar, uint32_t vram_size)
{
    if (!dev || !ctrl_bar || !vram_bar || vram_size == 0) {
        return -EINVAL;
    }

    dev->ctrl = (volatile uint8_t *)ctrl_bar;
    dev->vram = (uint8_t *)vram_bar;
    dev->vram_size = vram_size;
    dev->heap_next = 0;

    gpgpu_mmio_write32(dev, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_ENABLE);
    return 0;
}

int gpgpu_malloc(GPGPURuntimeDevice *dev, uint32_t *device_ptr, size_t size)
{
    uint32_t base;
    uint32_t end;
    int ret = gpgpu_check_dev(dev);

    if (ret < 0) {
        return ret;
    }
    if (!device_ptr || size == 0 || size > UINT32_MAX) {
        return -EINVAL;
    }

    base = gpgpu_align_up_u32(dev->heap_next, 16);
    if (size > UINT32_MAX - base) {
        return -ENOMEM;
    }

    end = gpgpu_align_up_u32(base + (uint32_t)size, 16);
    if (end > dev->vram_size || end < base) {
        return -ENOMEM;
    }

    dev->heap_next = end;
    *device_ptr = base;
    return 0;
}

int gpgpu_write(GPGPURuntimeDevice *dev, uint32_t dst, const void *src,
                size_t size)
{
    int ret = gpgpu_check_range(dev, dst, size);

    if (ret < 0) {
        return ret;
    }
    if (!src && size != 0) {
        return -EINVAL;
    }

    memcpy(dev->vram + dst, src, size);
    return 0;
}

int gpgpu_read(GPGPURuntimeDevice *dev, uint32_t src, void *dst, size_t size)
{
    int ret = gpgpu_check_range(dev, src, size);

    if (ret < 0) {
        return ret;
    }
    if (!dst && size != 0) {
        return -EINVAL;
    }

    memcpy(dst, dev->vram + src, size);
    return 0;
}

int gpgpu_upload_kernel(GPGPURuntimeDevice *dev, uint32_t *kernel_addr,
                        const uint32_t *code, size_t num_words)
{
    uint32_t addr;
    size_t size;
    int ret;

    if (!kernel_addr || !code || num_words == 0 ||
        num_words > SIZE_MAX / sizeof(*code)) {
        return -EINVAL;
    }

    size = num_words * sizeof(*code);
    ret = gpgpu_malloc(dev, &addr, size);
    if (ret < 0) {
        return ret;
    }

    ret = gpgpu_write(dev, addr, code, size);
    if (ret < 0) {
        return ret;
    }

    *kernel_addr = addr;
    return 0;
}

int gpgpu_upload_args(GPGPURuntimeDevice *dev, uint32_t *args_addr,
                      const void *args, size_t size)
{
    uint32_t addr;
    int ret;

    if (!args_addr || (!args && size != 0)) {
        return -EINVAL;
    }

    ret = gpgpu_malloc(dev, &addr, size == 0 ? sizeof(uint32_t) : size);
    if (ret < 0) {
        return ret;
    }

    if (size != 0) {
        ret = gpgpu_write(dev, addr, args, size);
        if (ret < 0) {
            return ret;
        }
    }

    *args_addr = addr;
    return 0;
}

int gpgpu_launch(GPGPURuntimeDevice *dev, uint32_t kernel_addr,
                 uint32_t kernel_args, GPGPURuntimeDim3 grid,
                 GPGPURuntimeDim3 block)
{
    int ret = gpgpu_check_dev(dev);
    uint32_t status;

    if (ret < 0) {
        return ret;
    }
    if (grid.x == 0 || grid.y == 0 || grid.z == 0 ||
        block.x == 0 || block.y == 0 || block.z == 0) {
        return -EINVAL;
    }

    gpgpu_mmio_write32(dev, GPGPU_REG_ERROR_STATUS, 0);
    gpgpu_mmio_write32(dev, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_ENABLE);
    gpgpu_mmio_write32(dev, GPGPU_REG_KERNEL_ADDR_LO, kernel_addr);
    gpgpu_mmio_write32(dev, GPGPU_REG_KERNEL_ADDR_HI, 0);
    gpgpu_mmio_write32(dev, GPGPU_REG_KERNEL_ARGS_LO, kernel_args);
    gpgpu_mmio_write32(dev, GPGPU_REG_KERNEL_ARGS_HI, 0);
    gpgpu_mmio_write32(dev, GPGPU_REG_GRID_DIM_X, grid.x);
    gpgpu_mmio_write32(dev, GPGPU_REG_GRID_DIM_Y, grid.y);
    gpgpu_mmio_write32(dev, GPGPU_REG_GRID_DIM_Z, grid.z);
    gpgpu_mmio_write32(dev, GPGPU_REG_BLOCK_DIM_X, block.x);
    gpgpu_mmio_write32(dev, GPGPU_REG_BLOCK_DIM_Y, block.y);
    gpgpu_mmio_write32(dev, GPGPU_REG_BLOCK_DIM_Z, block.z);
    gpgpu_mmio_write32(dev, GPGPU_REG_DISPATCH, 1);

    do {
        status = gpgpu_mmio_read32(dev, GPGPU_REG_GLOBAL_STATUS);
    } while (status & GPGPU_STATUS_BUSY);

    if (status & GPGPU_STATUS_ERROR) {
        return -EIO;
    }

    return 0;
}
