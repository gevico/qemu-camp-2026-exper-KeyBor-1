/*
 * QEMU Educational GPGPU - Backend Interface
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * GPGPU 执行后端抽象层。
 * 每个 backend 实现完整的 SIMT 执行管线。
 * gpgpu.c (PCIe 设备模型) 通过此接口委托内核执行。
 */

#ifndef __GPGPU_BACKEND_COMM_H__
#define __GPGPU_BACKEND_COMM_H__

#include "qemu/osdep.h"

/* ---- 前向声明 ---- */
struct gpgpu_backend;
struct gpgpu_backend_dev_iface;

/* ================================================================
 * 1. dispatch 参数 (BAR0 写完后传给后端的语义化参数)
 * ================================================================ */
struct gpgpu_dispatch_params {
    uint64_t kernel_addr;        /* 内核代码在 VRAM 中的偏移 */
    uint64_t kernel_args;        /* 内核参数在 VRAM 中的偏移 */
    uint32_t grid_dim[3];        /* Grid 维度 [X, Y, Z] */
    uint32_t block_dim[3];       /* Block 维度 [X, Y, Z] */
    uint32_t shared_mem_size;    /* 每 Block 共享内存大小 (字节) */
};

/* ================================================================
 * 2. 后端能力位
 * ================================================================ */
enum {
    GPGPU_BACKEND_CAP_BARRIER    = (1u << 0),   /* 支持 barrier 同步 */
    GPGPU_BACKEND_CAP_SHARED_MEM = (1u << 1),   /* 支持 __shared__ 内存 */
    GPGPU_BACKEND_CAP_DIVERGENCE = (1u << 2),   /* 支持分支发散重收敛 */
    GPGPU_BACKEND_CAP_FP8        = (1u << 3),   /* 支持 E4M3/E5M2/E2M1 */
    GPGPU_BACKEND_CAP_BF16       = (1u << 4),   /* 支持 BF16 */
};

/* ================================================================
 * 3. 设备资源接口 (backend 通过它访问 GPGPUState 的资源)
 * ================================================================ */
struct gpgpu_backend_dev_iface {
    /* VRAM */
    uint8_t *(*vram_ptr)(void *dev);
    uint64_t (*vram_size)(void *dev);

    /* 错误上报 */
    void (*set_error)(void *dev, uint32_t error_code);

    /*
     * 可选: 寄存器接管
     * 返回 true 表示后端已处理该 BAR0 偏移的读写。
     * 返回 false 表示走 gpgpu.c 默认路径。
     */
    bool (*reg_read)(void *dev, uint32_t offset, uint32_t *value);
    bool (*reg_write)(void *dev, uint32_t offset, uint32_t value);
};

/* ================================================================
 * 4. Backend 虚函数表
 * ================================================================ */
struct gpgpu_backend_ops {
    const char *name;

    /* 生命周期 */
    int  (*init)(struct gpgpu_backend *be, void *dev);
    void (*destroy)(struct gpgpu_backend *be);

    /* 核心调度: 阻塞直到内核完成，返回 0=成功, <0=失败 */
    int  (*dispatch)(struct gpgpu_backend *be, void *dev,
                     const struct gpgpu_dispatch_params *params);

    /* 能力查询 */
    void (*get_caps)(struct gpgpu_backend *be, uint32_t *caps);
};

/* ================================================================
 * 5. Backend 实例
 * ================================================================ */
struct gpgpu_backend {
    const struct gpgpu_backend_ops *ops;
    struct gpgpu_backend_dev_iface *dev_iface;  /* 设备资源，gpgpu.c 在 init 前赋值 */
    void *priv;     /* 后端私有数据 (如 GPGPUWarp 等) */
};

/* ================================================================
 * 6. Registry API
 * ================================================================ */

/* 注册后端类型 (class_init 时调用一次) */
void gpgpu_backend_registry_init(void);

/* 查表创建实例 (realize 时调用), 返回 NULL 表示未找到 */
struct gpgpu_backend *gpgpu_backend_create(const char *name);

/* 后端 factory 函数 (各后端实现) */
struct gpgpu_backend *gpgpu_backend_rv32_new(void);
struct gpgpu_backend *gpgpu_backend_vortex_new(void);

#endif /* __GPGPU_BACKEND_COMM_H__ */
