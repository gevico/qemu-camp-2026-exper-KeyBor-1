/*
 * Minimal GPGPU runtime interface
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#ifndef GPGPU_RUNTIME_H
#define GPGPU_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GPGPURuntimeDim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} GPGPURuntimeDim3;

typedef struct GPGPURuntimeDevice {
    volatile uint8_t *ctrl;
    uint8_t *vram;
    uint32_t vram_size;
    uint32_t heap_next;
} GPGPURuntimeDevice;

int gpgpu_runtime_init(GPGPURuntimeDevice *dev, volatile void *ctrl_bar,
                       void *vram_bar, uint32_t vram_size);

int gpgpu_malloc(GPGPURuntimeDevice *dev, uint32_t *device_ptr, size_t size);
int gpgpu_write(GPGPURuntimeDevice *dev, uint32_t dst, const void *src,
                size_t size);
int gpgpu_read(GPGPURuntimeDevice *dev, uint32_t src, void *dst, size_t size);

int gpgpu_upload_kernel(GPGPURuntimeDevice *dev, uint32_t *kernel_addr,
                        const uint32_t *code, size_t num_words);
int gpgpu_upload_args(GPGPURuntimeDevice *dev, uint32_t *args_addr,
                      const void *args, size_t size);

int gpgpu_launch(GPGPURuntimeDevice *dev, uint32_t kernel_addr,
                 uint32_t kernel_args, GPGPURuntimeDim3 grid,
                 GPGPURuntimeDim3 block);

#ifdef __cplusplus
}
#endif

#endif /* GPGPU_RUNTIME_H */
