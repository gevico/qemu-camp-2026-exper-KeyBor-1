/*
 * Minimal bare-metal PCI helper for the GPGPU runtime
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#ifndef GPGPU_PCI_H
#define GPGPU_PCI_H

#include "gpgpu_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPGPU_PCI_VENDOR_ID 0x1234
#define GPGPU_PCI_DEVICE_ID 0x1337

typedef struct GPGPUPciPlatform {
    uintptr_t ecam_base;
    uintptr_t mmio_base;
    size_t mmio_size;
} GPGPUPciPlatform;

typedef struct GPGPUPciDevice {
    uint8_t bus;
    uint8_t device;
    uint8_t function;
    uint16_t vendor_id;
    uint16_t device_id;
    uintptr_t bar0_base;
    uintptr_t bar2_base;
    uint32_t bar0_size;
    uint32_t bar2_size;
} GPGPUPciDevice;

int gpgpu_pci_find(const GPGPUPciPlatform *platform, GPGPUPciDevice *pdev);
int gpgpu_pci_setup_bars(const GPGPUPciPlatform *platform,
                         GPGPUPciDevice *pdev);
int gpgpu_runtime_init_pci(GPGPURuntimeDevice *dev,
                           const GPGPUPciPlatform *platform,
                           GPGPUPciDevice *pdev);

#ifdef __cplusplus
}
#endif

#endif /* GPGPU_PCI_H */
