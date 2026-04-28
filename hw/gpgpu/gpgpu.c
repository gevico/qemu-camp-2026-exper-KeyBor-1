/*
 * QEMU Educational GPGPU Device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/units.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qapi/error.h"
#include "hw/pci/pci.h"
#include "hw/pci/msi.h"
#include "hw/pci/msix.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"

#include "gpgpu.h"
#include "gpgpu_core.h"

/* TODO: Implement MMIO control register read */
static uint64_t gpgpu_ctrl_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = GPGPU(opaque);
    // 2. 将 State 转换为通用 PCI 设备指针
    // PCIDevice *pdev = PCI_DEVICE(s);
    // // 3. 获取对应的 Class 指针
    // PCIDeviceClass *pc = PCI_DEVICE_GET_CLASS(pdev);
    uint64_t value = 0;
    switch(addr) {
        case GPGPU_REG_DEV_ID:
            value = 0x47505055;
            break;
        case GPGPU_REG_DEV_VERSION:
            value = 0x00010000;
            break;
        case GPGPU_REG_VRAM_SIZE_LO:
            value = s->vram_size & 0xFFFFFFFF;
            break;
        case GPGPU_REG_VRAM_SIZE_HI:
            value = (s->vram_size >> 32) & 0xFFFFFFFF;
            break;
        case GPGPU_REG_DEV_CAPS:
            value = (s->num_cus & 0xFF) | ((s->warps_per_cu & 0xFF) << 8) | ((s->warp_size & 0xFF) << 16) | (0x0 << 24);
            break;
        case GPGPU_REG_GLOBAL_CTRL:
            value = s->global_ctrl;
            break;
        case GPGPU_REG_GLOBAL_STATUS:
            value = s->global_status;
            break;
        case GPGPU_REG_ERROR_STATUS:
            value = s->error_status;
            break;
        case GPGPU_REG_IRQ_ENABLE:
            value = s->irq_enable;
            break;
        case GPGPU_REG_IRQ_STATUS:
            value = s->irq_status;
            break;
        case GPGPU_REG_KERNEL_ADDR_LO:
            value = s->kernel.kernel_addr & 0xFFFFFFFF;
            break;
        case GPGPU_REG_KERNEL_ADDR_HI:
            value = (s->kernel.kernel_addr >> 32) & 0xFFFFFFFF;
            break;
        case GPGPU_REG_KERNEL_ARGS_LO:
            value = s->kernel.kernel_args & 0xFFFFFFFF;
            break;
        case GPGPU_REG_KERNEL_ARGS_HI:
            value = (s->kernel.kernel_args >> 32) & 0xFFFFFFFF;
            break;
        case GPGPU_REG_GRID_DIM_X:
            value = s->kernel.grid_dim[0];
            break;
        case GPGPU_REG_GRID_DIM_Y:
            value = s->kernel.grid_dim[1];
            break;
        case GPGPU_REG_GRID_DIM_Z:
            value = s->kernel.grid_dim[2];
            break;
        case GPGPU_REG_BLOCK_DIM_X:
            value = s->kernel.block_dim[0];
            break;
        case GPGPU_REG_BLOCK_DIM_Y:
            value = s->kernel.block_dim[1];
            break;
        case GPGPU_REG_BLOCK_DIM_Z:
            value = s->kernel.block_dim[2];
            break;
        case GPGPU_REG_SHARED_MEM_SIZE:
            value = s->kernel.shared_mem_size;
            break;
        case GPGPU_REG_DMA_SRC_LO:
            value = s->dma.src_addr & 0xFFFFFFFF;
            break;
        case GPGPU_REG_DMA_SRC_HI:
            value = (s->dma.src_addr >> 32) & 0xFFFFFFFF;
            break;
        case GPGPU_REG_DMA_DST_LO:
            value = s->dma.dst_addr & 0xFFFFFFFF;
            break;
        case GPGPU_REG_DMA_DST_HI:
            value = (s->dma.dst_addr >> 32) & 0xFFFFFFFF;
            break;
        case GPGPU_REG_DMA_SIZE:
            value = s->dma.size;
            break;
        case GPGPU_REG_DMA_CTRL:
            value = s->dma.ctrl;
            break;
        case GPGPU_REG_DMA_STATUS:
            value = s->dma.status;
            break;
        /* 线程上下文寄存器组 (0x1000 - 0x1FFF): GPU 线程读取自身 ID */
        case GPGPU_REG_THREAD_ID_X:
            value = s->simt.thread_id[0];
            break;
        case GPGPU_REG_THREAD_ID_Y:
            value = s->simt.thread_id[1];
            break;
        case GPGPU_REG_THREAD_ID_Z:
            value = s->simt.thread_id[2];
            break;
        case GPGPU_REG_BLOCK_ID_X:
            value = s->simt.block_id[0];
            break;
        case GPGPU_REG_BLOCK_ID_Y:
            value = s->simt.block_id[1];
            break;
        case GPGPU_REG_BLOCK_ID_Z:
            value = s->simt.block_id[2];
            break;
        case GPGPU_REG_WARP_ID:
            value = s->simt.warp_id;
            break;
        case GPGPU_REG_LANE_ID:
            value = s->simt.lane_id;
            break;
        case GPGPU_REG_THREAD_MASK:
            value = s->simt.thread_mask;
            break;
        default:
            break;
    }
    return value;
}

static void gpgpu_ctrl_update_status(GPGPUState *s)
{
    if(s->global_ctrl & GPGPU_CTRL_RESET) {
        s->global_ctrl = 0;
        s->global_status = GPGPU_STATUS_READY;
        s->error_status = 0;
        s->irq_status = 0;
        memset(&s->kernel, 0, sizeof(GPGPUKernelParams));
        memset(&s->dma, 0, sizeof(GPGPUDMAState));
        memset(&s->simt, 0, sizeof(GPGPUSIMTContext));
        return;
    }
}


static void gpgpu_dispatch_kernel(GPGPUState *s)
{
    bool is_illegal_param = ((s->kernel.kernel_addr > GPGPU_DEFAULT_VRAM_SIZE) || (s->kernel.kernel_args > GPGPU_DEFAULT_VRAM_SIZE) || ((s->kernel.grid_dim[0] <= 0) || (s->kernel.grid_dim[1] <= 0) || (s->kernel.grid_dim[2] <= 0)) ||
                             ((s->kernel.block_dim[0] <= 0) || (s->kernel.block_dim[1] <= 0) || (s->kernel.block_dim[2] <= 0)));
    if(!(s->global_ctrl & GPGPU_CTRL_ENABLE) || (s->global_status & GPGPU_STATUS_BUSY) || is_illegal_param) {
        s->global_status |= GPGPU_STATUS_ERROR;
        return;
    }
    s->global_status |= GPGPU_STATUS_BUSY;

    // uint32_t grid_size = s->kernel.grid_dim[0] * s->kernel.grid_dim[1] * s->kernel.grid_dim[2];
    // 一个block中有多少个线程
    uint32_t block_size = s->kernel.block_dim[0] * s->kernel.block_dim[1] * s->kernel.block_dim[2];
    for(int i = 0; i < s->kernel.grid_dim[0]; ++i) {
        for(int j = 0; j < s->kernel.grid_dim[1]; ++j) {
            for(int k = 0; k < s->kernel.grid_dim[2]; ++k){
                //根据warpsize切分warp个数
                uint32_t warp_num = (block_size + (s->warp_size - 1)) / s->warp_size;
                for(int m = 0; m < warp_num; ++m) {
                    GPGPUWarp warp = {0};
                    const uint32_t block_id[3] = {i, j, k};
                    uint32_t block_id_linear = k * (s->kernel.grid_dim[0] * s->kernel.grid_dim[1]) + i * s->kernel.grid_dim[0] + j;
                    uint32_t thread_num = (s->warp_size > block_size) ? block_size : s->warp_size;
                    for(int n = 0; n < thread_num; ++n)
                        warp.active_mask |= (1 << n); 
                    gpgpu_core_init_warp(&warp, s->kernel.kernel_addr,
                                m * s->warp_size, block_id,
                                thread_num,
                                m, block_id_linear);
                    gpgpu_core_exec_warp(s, &warp, 10);
                }
            }
        }
    }
    s->global_status &= ~GPGPU_STATUS_BUSY;
    s->global_status |= GPGPU_STATUS_READY;
}
/* TODO: Implement MMIO control register write */
static void gpgpu_ctrl_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = GPGPU(opaque);
    switch(addr) {
        case GPGPU_REG_GLOBAL_CTRL:
            s->global_ctrl = val;
            break;
        case GPGPU_REG_ERROR_STATUS:
            s->error_status = val;
            break;
        case GPGPU_REG_IRQ_ENABLE:
            s->irq_enable = val;
            break;
        case GPGPU_REG_KERNEL_ADDR_LO:
            s->kernel.kernel_addr =
                (s->kernel.kernel_addr & 0xFFFFFFFF00000000ULL) |
                (uint32_t)val;
            break;
        case GPGPU_REG_KERNEL_ADDR_HI:
            s->kernel.kernel_addr =
                (s->kernel.kernel_addr & 0x00000000FFFFFFFFULL) |
                ((uint64_t)(uint32_t)val << 32);
            break;
        case GPGPU_REG_KERNEL_ARGS_LO:
            s->kernel.kernel_args =
                (s->kernel.kernel_args & 0xFFFFFFFF00000000ULL) |
                (uint32_t)val;
            break;
        case GPGPU_REG_KERNEL_ARGS_HI:
            s->kernel.kernel_args =
                (s->kernel.kernel_args & 0x00000000FFFFFFFFULL) |
                ((uint64_t)(uint32_t)val << 32);
            break;
        case GPGPU_REG_GRID_DIM_X:
            s->kernel.grid_dim[0] = (uint32_t)val;
            break;
        case GPGPU_REG_GRID_DIM_Y:
            s->kernel.grid_dim[1] = (uint32_t)val;
            break;
        case GPGPU_REG_GRID_DIM_Z:
            s->kernel.grid_dim[2] = (uint32_t)val;
            break;
        case GPGPU_REG_BLOCK_DIM_X:
            s->kernel.block_dim[0] = (uint32_t)val;
            break;
        case GPGPU_REG_BLOCK_DIM_Y:
            s->kernel.block_dim[1] = (uint32_t)val;
            break;
        case GPGPU_REG_BLOCK_DIM_Z:
            s->kernel.block_dim[2] = (uint32_t)val;
            break;
        case GPGPU_REG_SHARED_MEM_SIZE:
            s->kernel.shared_mem_size = (uint32_t)val;
            break;
        case GPGPU_REG_DISPATCH:
            gpgpu_dispatch_kernel(s);
            break;
        case GPGPU_REG_DMA_SRC_LO:
            s->dma.src_addr =
                (s->dma.src_addr & 0xFFFFFFFF00000000ULL) |
                (uint32_t)val;
            break;
        case GPGPU_REG_DMA_SRC_HI:
            s->dma.src_addr =
                (s->dma.src_addr & 0x00000000FFFFFFFFULL) |
                ((uint64_t)(uint32_t)val << 32);
            break;
        case GPGPU_REG_DMA_DST_LO:
            s->dma.dst_addr =
                (s->dma.dst_addr & 0xFFFFFFFF00000000ULL) |
                (uint32_t)val;
            break;
        case GPGPU_REG_DMA_DST_HI:
            s->dma.dst_addr =
                (s->dma.dst_addr & 0x00000000FFFFFFFFULL) |
                ((uint64_t)(uint32_t)val << 32);
            break;
        case GPGPU_REG_DMA_SIZE:
            s->dma.size = (uint32_t)val;
            break;
        case GPGPU_REG_DMA_CTRL:
            s->dma.ctrl = (uint32_t)val;
            break;
        case GPGPU_REG_DMA_STATUS:
            /* DMA_STATUS is read-only. Ignore writes. */
            break;
        /* 线程上下文寄存器组 (0x1000 - 0x1FFF): 只读，GPU 核心运行时由硬件设置 */
        case GPGPU_REG_THREAD_ID_X:
            s->simt.thread_id[0] = (uint32_t)val;
            break;
        case GPGPU_REG_THREAD_ID_Y:
            s->simt.thread_id[1] = (uint32_t)val;
            break;
        case GPGPU_REG_THREAD_ID_Z:
            s->simt.thread_id[2] = (uint32_t)val;
            break;
        case GPGPU_REG_BLOCK_ID_X:
            s->simt.block_id[0] = (uint32_t)val;
            break;
        case GPGPU_REG_BLOCK_ID_Y:
            s->simt.block_id[1] = (uint32_t)val;
            break;
        case GPGPU_REG_BLOCK_ID_Z:
            s->simt.block_id[2] = (uint32_t)val;
            break;
        case GPGPU_REG_WARP_ID:
            s->simt.warp_id = (uint32_t)val;
            break;
        case GPGPU_REG_LANE_ID:
            s->simt.lane_id = (uint32_t)val;
            break;
        case GPGPU_REG_THREAD_MASK:
            s->simt.thread_mask = (uint32_t)val;
            break;
        case GPGPU_REG_BARRIER:
            s->simt.barrier_active = (uint32_t)val;
            break;
        default:
            break;
    }
    gpgpu_ctrl_update_status(s);
}

static const MemoryRegionOps gpgpu_ctrl_ops = {
    .read = gpgpu_ctrl_read,
    .write = gpgpu_ctrl_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* TODO: Implement VRAM read */
static uint64_t gpgpu_vram_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = GPGPU(opaque);
    uint8_t *ptr = s->vram_ptr + addr;
    switch (size) {
    case 1:
        return *(uint8_t *)ptr;
    case 2:
        return lduw_le_p(ptr);  /* 小端读取16位 */
    case 4:
        return ldl_le_p(ptr);   /* 小端读取32位 */
    case 8:
        return ldq_le_p(ptr);   /* 小端读取64位 */
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "%s: invalid size %u\n", __func__, size);
        return 0;
    }
}

/* TODO: Implement VRAM write */
static void gpgpu_vram_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = GPGPU(opaque);
    uint8_t *ptr = s->vram_ptr + addr;
    switch (size) {
    case 1:
        *(uint8_t *)ptr = val;
        break;
    case 2:
        stw_le_p(ptr, val);  /* 小端写入16位 */
        break;
    case 4:
        stl_le_p(ptr, val);  /* 小端写入32位 */
        break;
    case 8:
        stq_le_p(ptr, val);  /* 小端写入64位 */
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "%s: invalid size %u\n", __func__, size);
        break;
    }

}

static const MemoryRegionOps gpgpu_vram_ops = {
    .read = gpgpu_vram_read,
    .write = gpgpu_vram_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static uint64_t gpgpu_doorbell_read(void *opaque, hwaddr addr, unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)size;
    return 0;
}

static void gpgpu_doorbell_write(void *opaque, hwaddr addr, uint64_t val,
                                 unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)val;
    (void)size;
}

static const MemoryRegionOps gpgpu_doorbell_ops = {
    .read = gpgpu_doorbell_read,
    .write = gpgpu_doorbell_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* TODO: Implement DMA completion handler */
static void gpgpu_dma_complete(void *opaque)
{
    (void)opaque;
}

/* TODO: Implement kernel completion handler */
static void gpgpu_kernel_complete(void *opaque)
{
    (void)opaque;
}

static void gpgpu_realize(PCIDevice *pdev, Error **errp)
{
    GPGPUState *s = GPGPU(pdev);
    uint8_t *pci_conf = pdev->config;

    pci_config_set_interrupt_pin(pci_conf, 1);

    s->vram_ptr = g_malloc0(s->vram_size);
    if (!s->vram_ptr) {
        error_setg(errp, "GPGPU: failed to allocate VRAM");
        return;
    }

    /* BAR 0: control registers */
    memory_region_init_io(&s->ctrl_mmio, OBJECT(s), &gpgpu_ctrl_ops, s,
                          "gpgpu-ctrl", GPGPU_CTRL_BAR_SIZE);
    pci_register_bar(pdev, 0,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64,
                     &s->ctrl_mmio);

    /* BAR 2: VRAM */
    memory_region_init_io(&s->vram, OBJECT(s), &gpgpu_vram_ops, s,
                          "gpgpu-vram", s->vram_size);
    pci_register_bar(pdev, 2,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64 |
                     PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->vram);

    /* BAR 4: doorbell registers */
    memory_region_init_io(&s->doorbell_mmio, OBJECT(s), &gpgpu_doorbell_ops, s,
                          "gpgpu-doorbell", GPGPU_DOORBELL_BAR_SIZE);
    pci_register_bar(pdev, 4,
                     PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->doorbell_mmio);

    if (msix_init(pdev, GPGPU_MSIX_VECTORS,
                  &s->ctrl_mmio, 0, 0xFE000,
                  &s->ctrl_mmio, 0, 0xFF000,
                  0, errp)) {
        g_free(s->vram_ptr);
        return;
    }

    msi_init(pdev, 0, 1, true, false, errp);

    s->dma_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL, gpgpu_dma_complete, s);
    s->kernel_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,
                                   gpgpu_kernel_complete, s);

    s->global_status = GPGPU_STATUS_READY;
}

static void gpgpu_exit(PCIDevice *pdev)
{
    GPGPUState *s = GPGPU(pdev);

    timer_free(s->dma_timer);
    timer_free(s->kernel_timer);
    g_free(s->vram_ptr);
    msix_uninit(pdev, &s->ctrl_mmio, &s->ctrl_mmio);
    msi_uninit(pdev);
}

static void gpgpu_reset(DeviceState *dev)
{
    GPGPUState *s = GPGPU(dev);

    s->global_ctrl = 0;
    s->global_status = GPGPU_STATUS_READY;
    s->error_status = 0;
    s->irq_enable = 0;
    s->irq_status = 0;
    memset(&s->kernel, 0, sizeof(s->kernel));
    memset(&s->dma, 0, sizeof(s->dma));
    memset(&s->simt, 0, sizeof(s->simt));
    timer_del(s->dma_timer);
    timer_del(s->kernel_timer);
    if (s->vram_ptr) {
        memset(s->vram_ptr, 0, s->vram_size);
    }
}

static const Property gpgpu_properties[] = {
    DEFINE_PROP_UINT32("num_cus", GPGPUState, num_cus,
                       GPGPU_DEFAULT_NUM_CUS),
    DEFINE_PROP_UINT32("warps_per_cu", GPGPUState, warps_per_cu,
                       GPGPU_DEFAULT_WARPS_PER_CU),
    DEFINE_PROP_UINT32("warp_size", GPGPUState, warp_size,
                       GPGPU_DEFAULT_WARP_SIZE),
    DEFINE_PROP_UINT64("vram_size", GPGPUState, vram_size,
                       GPGPU_DEFAULT_VRAM_SIZE),
};

static const VMStateDescription vmstate_gpgpu = {
    .name = "gpgpu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, GPGPUState),
        VMSTATE_UINT32(global_ctrl, GPGPUState),
        VMSTATE_UINT32(global_status, GPGPUState),
        VMSTATE_UINT32(error_status, GPGPUState),
        VMSTATE_UINT32(irq_enable, GPGPUState),
        VMSTATE_UINT32(irq_status, GPGPUState),
        VMSTATE_END_OF_LIST()
    }
};

static void gpgpu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    pc->realize = gpgpu_realize;
    pc->exit = gpgpu_exit;
    pc->vendor_id = GPGPU_VENDOR_ID;
    pc->device_id = GPGPU_DEVICE_ID;
    pc->revision = GPGPU_REVISION;
    pc->class_id = GPGPU_CLASS_CODE;

    device_class_set_legacy_reset(dc, gpgpu_reset);
    dc->desc = "Educational GPGPU Device";
    dc->vmsd = &vmstate_gpgpu;
    device_class_set_props(dc, gpgpu_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo gpgpu_type_info = {
    .name          = TYPE_GPGPU,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GPGPUState),
    .class_init    = gpgpu_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { }
    },
};

static void gpgpu_register_types(void)
{
    type_register_static(&gpgpu_type_info);
}

type_init(gpgpu_register_types)