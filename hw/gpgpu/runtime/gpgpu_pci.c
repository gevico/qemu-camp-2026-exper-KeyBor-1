/*
 * Minimal bare-metal PCI helper for the GPGPU runtime
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "gpgpu_pci.h"
#include "gpgpu_freestanding.h"

#include <stdbool.h>

#define PCI_VENDOR_ID              0x00
#define PCI_DEVICE_ID              0x02
#define PCI_COMMAND                0x04
#define PCI_HEADER_TYPE            0x0e
#define PCI_BAR0                   0x10
#define PCI_BAR2                   0x18

#define PCI_COMMAND_MEMORY         (1u << 1)
#define PCI_COMMAND_MASTER         (1u << 2)

#define PCI_BAR_IO_SPACE           0x1u
#define PCI_BAR_MEM_TYPE_MASK      0x6u
#define PCI_BAR_MEM_TYPE_64        0x4u
#define PCI_BAR_MEM_MASK           0xfffffff0u

#define PCI_MAX_BUS                256
#define PCI_MAX_DEVICE             32
#define PCI_MAX_FUNCTION           8

typedef struct GPGPUPciAllocator {
    uintptr_t next;
    uintptr_t end;
} GPGPUPciAllocator;

static uintptr_t gpgpu_align_up_ptr(uintptr_t value, uintptr_t align)
{
    return (value + align - 1u) & ~(align - 1u);
}

static volatile uint8_t *gpgpu_pci_cfg_ptr(const GPGPUPciPlatform *platform,
                                           uint8_t bus, uint8_t device,
                                           uint8_t function, uint16_t offset)
{
    uintptr_t addr = platform->ecam_base;

    addr += ((uintptr_t)bus << 20);
    addr += ((uintptr_t)device << 15);
    addr += ((uintptr_t)function << 12);
    addr += offset;
    return (volatile uint8_t *)addr;
}

static uint8_t gpgpu_pci_read8(const GPGPUPciPlatform *platform,
                               uint8_t bus, uint8_t device,
                               uint8_t function, uint16_t offset)
{
    return *gpgpu_pci_cfg_ptr(platform, bus, device, function, offset);
}

static uint16_t gpgpu_pci_read16(const GPGPUPciPlatform *platform,
                                 uint8_t bus, uint8_t device,
                                 uint8_t function, uint16_t offset)
{
    volatile uint16_t *ptr = (volatile uint16_t *)
        gpgpu_pci_cfg_ptr(platform, bus, device, function, offset);

    return *ptr;
}

static uint32_t gpgpu_pci_read32(const GPGPUPciPlatform *platform,
                                 uint8_t bus, uint8_t device,
                                 uint8_t function, uint16_t offset)
{
    volatile uint32_t *ptr = (volatile uint32_t *)
        gpgpu_pci_cfg_ptr(platform, bus, device, function, offset);

    return *ptr;
}

static void gpgpu_pci_write16(const GPGPUPciPlatform *platform,
                              uint8_t bus, uint8_t device,
                              uint8_t function, uint16_t offset,
                              uint16_t value)
{
    volatile uint16_t *ptr = (volatile uint16_t *)
        gpgpu_pci_cfg_ptr(platform, bus, device, function, offset);

    *ptr = value;
}

static void gpgpu_pci_write32(const GPGPUPciPlatform *platform,
                              uint8_t bus, uint8_t device,
                              uint8_t function, uint16_t offset,
                              uint32_t value)
{
    volatile uint32_t *ptr = (volatile uint32_t *)
        gpgpu_pci_cfg_ptr(platform, bus, device, function, offset);

    *ptr = value;
}

static int gpgpu_pci_check_platform(const GPGPUPciPlatform *platform)
{
    if (!platform || platform->ecam_base == 0 ||
        platform->mmio_base == 0 || platform->mmio_size == 0) {
        return -EINVAL;
    }

    return 0;
}

static uint64_t gpgpu_pci_bar_size_from_mask(uint64_t mask)
{
    uint64_t addr_mask = mask & ~(uint64_t)0xf;

    if (addr_mask == 0) {
        return 0;
    }

    return (~addr_mask) + 1u;
}

static int gpgpu_pci_probe_bar(const GPGPUPciPlatform *platform,
                               const GPGPUPciDevice *pdev,
                               uint16_t bar_offset, uint32_t *size,
                               bool *is_64bit)
{
    uint32_t old_lo;
    uint32_t old_hi = 0;
    uint32_t mask_lo;
    uint32_t mask_hi = 0;
    uint64_t mask;
    uint64_t bar_size;

    old_lo = gpgpu_pci_read32(platform, pdev->bus, pdev->device,
                              pdev->function, bar_offset);
    if (old_lo & PCI_BAR_IO_SPACE) {
        return -ENOTSUP;
    }
    *is_64bit = (old_lo & PCI_BAR_MEM_TYPE_MASK) == PCI_BAR_MEM_TYPE_64;
    if (*is_64bit) {
        old_hi = gpgpu_pci_read32(platform, pdev->bus, pdev->device,
                                  pdev->function, bar_offset + 4);
    }

    gpgpu_pci_write32(platform, pdev->bus, pdev->device,
                      pdev->function, bar_offset, 0xffffffffu);
    if (*is_64bit) {
        gpgpu_pci_write32(platform, pdev->bus, pdev->device,
                          pdev->function, bar_offset + 4, 0xffffffffu);
    }

    mask_lo = gpgpu_pci_read32(platform, pdev->bus, pdev->device,
                               pdev->function, bar_offset);
    if (*is_64bit) {
        mask_hi = gpgpu_pci_read32(platform, pdev->bus, pdev->device,
                                   pdev->function, bar_offset + 4);
    }

    gpgpu_pci_write32(platform, pdev->bus, pdev->device,
                      pdev->function, bar_offset, old_lo);
    if (*is_64bit) {
        gpgpu_pci_write32(platform, pdev->bus, pdev->device,
                          pdev->function, bar_offset + 4, old_hi);
    }

    mask = ((uint64_t)mask_hi << 32) | (mask_lo & PCI_BAR_MEM_MASK);
    bar_size = gpgpu_pci_bar_size_from_mask(mask);
    if (bar_size == 0 || bar_size > UINT32_MAX) {
        return -ENODEV;
    }

    *size = (uint32_t)bar_size;
    return 0;
}

static void gpgpu_pci_write_bar(const GPGPUPciPlatform *platform,
                                const GPGPUPciDevice *pdev,
                                uint16_t bar_offset, uintptr_t base,
                                bool is_64bit)
{
    uint32_t lo = (uint32_t)(base & PCI_BAR_MEM_MASK);

    gpgpu_pci_write32(platform, pdev->bus, pdev->device,
                      pdev->function, bar_offset, lo);
    if (is_64bit) {
        gpgpu_pci_write32(platform, pdev->bus, pdev->device,
                          pdev->function, bar_offset + 4,
                          (uint32_t)((uint64_t)base >> 32));
    }
}

static int gpgpu_pci_alloc_mmio(GPGPUPciAllocator *alloc, uint32_t size,
                                uintptr_t *base)
{
    uintptr_t aligned;

    if (size == 0 || (size & (size - 1u)) != 0) {
        return -EINVAL;
    }

    aligned = gpgpu_align_up_ptr(alloc->next, size);
    if (aligned > alloc->end || size > alloc->end - aligned) {
        return -ENOMEM;
    }

    alloc->next = aligned + size;
    *base = aligned;
    return 0;
}

int gpgpu_pci_find(const GPGPUPciPlatform *platform, GPGPUPciDevice *pdev)
{
    int ret = gpgpu_pci_check_platform(platform);

    if (ret < 0) {
        return ret;
    }
    if (!pdev) {
        return -EINVAL;
    }

    for (uint16_t bus = 0; bus < PCI_MAX_BUS; ++bus) {
        for (uint8_t dev = 0; dev < PCI_MAX_DEVICE; ++dev) {
            uint8_t header;
            uint8_t funcs;

            if (gpgpu_pci_read16(platform, bus, dev, 0,
                                 PCI_VENDOR_ID) == 0xffffu) {
                continue;
            }

            header = gpgpu_pci_read8(platform, bus, dev, 0,
                                     PCI_HEADER_TYPE);
            funcs = (header & 0x80) ? PCI_MAX_FUNCTION : 1;

            for (uint8_t fn = 0; fn < funcs; ++fn) {
                uint16_t vendor = gpgpu_pci_read16(platform, bus, dev, fn,
                                                   PCI_VENDOR_ID);
                uint16_t device = gpgpu_pci_read16(platform, bus, dev, fn,
                                                   PCI_DEVICE_ID);

                if (vendor == 0xffffu) {
                    continue;
                }
                if (vendor == GPGPU_PCI_VENDOR_ID &&
                    device == GPGPU_PCI_DEVICE_ID) {
                    memset(pdev, 0, sizeof(*pdev));
                    pdev->bus = (uint8_t)bus;
                    pdev->device = dev;
                    pdev->function = fn;
                    pdev->vendor_id = vendor;
                    pdev->device_id = device;
                    return 0;
                }
            }
        }
    }

    return -ENODEV;
}

int gpgpu_pci_setup_bars(const GPGPUPciPlatform *platform,
                         GPGPUPciDevice *pdev)
{
    GPGPUPciAllocator alloc;
    bool bar0_64;
    bool bar2_64;
    uint16_t command;
    int ret = gpgpu_pci_check_platform(platform);

    if (ret < 0) {
        return ret;
    }
    if (!pdev) {
        return -EINVAL;
    }

    alloc.next = platform->mmio_base;
    alloc.end = platform->mmio_base + platform->mmio_size;
    if (alloc.end <= alloc.next) {
        return -EINVAL;
    }

    ret = gpgpu_pci_probe_bar(platform, pdev, PCI_BAR0, &pdev->bar0_size,
                              &bar0_64);
    if (ret < 0) {
        return ret;
    }
    ret = gpgpu_pci_alloc_mmio(&alloc, pdev->bar0_size, &pdev->bar0_base);
    if (ret < 0) {
        return ret;
    }
    gpgpu_pci_write_bar(platform, pdev, PCI_BAR0, pdev->bar0_base, bar0_64);

    ret = gpgpu_pci_probe_bar(platform, pdev, PCI_BAR2, &pdev->bar2_size,
                              &bar2_64);
    if (ret < 0) {
        return ret;
    }
    ret = gpgpu_pci_alloc_mmio(&alloc, pdev->bar2_size, &pdev->bar2_base);
    if (ret < 0) {
        return ret;
    }
    gpgpu_pci_write_bar(platform, pdev, PCI_BAR2, pdev->bar2_base, bar2_64);

    command = gpgpu_pci_read16(platform, pdev->bus, pdev->device,
                               pdev->function, PCI_COMMAND);
    command |= PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER;
    gpgpu_pci_write16(platform, pdev->bus, pdev->device,
                      pdev->function, PCI_COMMAND, command);

    return 0;
}

int gpgpu_runtime_init_pci(GPGPURuntimeDevice *dev,
                           const GPGPUPciPlatform *platform,
                           GPGPUPciDevice *pdev)
{
    int ret;

    if (!dev || !pdev) {
        return -EINVAL;
    }

    ret = gpgpu_pci_find(platform, pdev);
    if (ret < 0) {
        return ret;
    }

    ret = gpgpu_pci_setup_bars(platform, pdev);
    if (ret < 0) {
        return ret;
    }

    return gpgpu_runtime_init(dev, (volatile void *)pdev->bar0_base,
                              (void *)pdev->bar2_base, pdev->bar2_size);
}
