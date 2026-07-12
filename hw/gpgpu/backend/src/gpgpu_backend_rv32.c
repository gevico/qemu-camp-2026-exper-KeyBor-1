/*
 * QEMU Educational GPGPU - RV32 Interpreter Backend (thin adapter)
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * 将手写 RV32I/F/M SIMT 解释器包装为 GPGPU backend。
 * 本文件仅负责函数指针赋值，所有逻辑在解释器中实现。
 */

#include "qemu/osdep.h"
#include "gpgpu.h"
#include "../include/gpgpu_backend_comm.h"
#include "../arch/riscv/rvi_interpreter/gpgpu_core.h"

/* ================================================================
 * dispatch - 委托给解释器
 * ================================================================ */

static int rv32_dispatch(struct gpgpu_backend *be, void *dev,
                         const struct gpgpu_dispatch_params *p)
{
    (void)be;
    return gpgpu_core_dispatch_kernel(dev, p);
}

/* ================================================================
 * get_caps
 * ================================================================ */

static void rv32_get_caps(struct gpgpu_backend *be, uint32_t *caps)
{
    (void)be;
    *caps = GPGPU_BACKEND_CAP_FP8 | GPGPU_BACKEND_CAP_BF16;
}

/* ================================================================
 * 生命周期 (当前无私有状态)
 * ================================================================ */

static int rv32_init(struct gpgpu_backend *be, void *dev)
{
    (void)be;
    (void)dev;
    return 0;
}

static void rv32_destroy(struct gpgpu_backend *be)
{
    (void)be;
}

/* ================================================================
 * ops
 * ================================================================ */

static const struct gpgpu_backend_ops rv32_ops = {
    .name     = "rv32",
    .init     = rv32_init,
    .destroy  = rv32_destroy,
    .dispatch = rv32_dispatch,
    .get_caps = rv32_get_caps,
};

/* ================================================================
 * factory (供 registry 调用)
 * ================================================================ */

struct gpgpu_backend *gpgpu_backend_rv32_new(void)
{
    struct gpgpu_backend *be = g_new0(struct gpgpu_backend, 1);
    be->ops = &rv32_ops;
    return be;
}
