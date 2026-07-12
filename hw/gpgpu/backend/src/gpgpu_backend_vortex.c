/*
 * QEMU Educational GPGPU - Vortex Backend
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * Thin adapter: 委托 vortex_simx_wrapper 执行 SIMT 内核。
 * 直接进程内链接 libsimx.so，无 IPC 开销。
 *
 * block_id / block_dim / grid_dim 注入:
 *   vortex 无硬件 CSR 支持这些概念 (证据: docs/vortex-csr-evidence.md)，
 *   因此 dispatch 时将它们写入 vortex RAM 的 SIMT metadata 区
 *   (VORTEX_META_ADDR = 0x7FFFF000)，供 kernel builtins 读取。
 */

#include "qemu/osdep.h"
#include "gpgpu.h"
#include "../include/gpgpu_backend_comm.h"
#include "../arch/riscv/vortex_backend/vortex_simx_wrapper.h"

/* ---- 传输缓冲区大小 ---- */
#define KERNEL_BUF_SIZE    (64 * 1024)   /* 内核代码最大 64KB */
#define ARGS_BUF_SIZE      (4 * 1024)    /* 内核参数最大 4KB */
#define RESULT_BUF_SIZE    (4 * 1024)    /* 结果缓冲区大小 */

/*
 * VRAM <-> vortex RAM 同步: kernel 通过 args 收到的指针是 QEMU VRAM 偏移量,
 * 在 vortex RAM 中以相同偏移量访问。因此执行后需要将 vortex RAM 的堆区域
 * 同步回 QEMU VRAM, 以便 host 侧能读到 kernel 的输出。
 */
#define SYNC_HEAP_SIZE     0x10000  /* 64KB: 覆盖 smoke tests 的堆分配 */

/*
 * Trampoline: vortex emulator.reset() 将 args 地址写入 mscratch CSR，
 * 但不会设置 a0 寄存器。baremetal kernel 的 _start(uint32_t *args) 期望
 * a0 接收参数，所以插入 csrr a0, mscratch 作为桥接指令。
 *
 * RISC-V encoding: csrr a0, mscratch
 *   csr     = 0x340 (MSCRATCH)
 *   rd      = x10   (a0)
 *   funct3  = 0x2   (CSRRS)
 *   opcode  = 0x73  (SYSTEM)
 *   → 0x34002573
 */
#define TRAMPOLINE_INSN   0x34002573u
#define TRAMPOLINE_SIZE   4

/* ---- Vortex 地址常量 ---- */
#define VORTEX_STARTUP_ADDR  0x80000000ULL
#define VORTEX_ARGS_ADDR     0x80100000ULL  /* STARTUP_ADDR + 1MB */

/* ---- SIMT metadata 注入地址 ---- */
#define VORTEX_META_ADDR     0x7FFFF000ULL  /* STARTUP_ADDR 下方 4KB */

/* metadata offset (与 gpgpu_builtins.h 中 VORTEX_META_* 保持一致) */
#define META_BLOCK_ID_X   0x10u
#define META_BLOCK_ID_Y   0x14u
#define META_BLOCK_ID_Z   0x18u
#define META_BLOCK_DIM_X  0x20u
#define META_BLOCK_DIM_Y  0x24u
#define META_BLOCK_DIM_Z  0x28u
#define META_GRID_DIM_X   0x30u
#define META_GRID_DIM_Y   0x34u
#define META_GRID_DIM_Z   0x38u

/* ---- DCR 地址 ---- */
#define VX_DCR_STARTUP_ADDR0  0x001
#define VX_DCR_STARTUP_ADDR1  0x002
#define VX_DCR_STARTUP_ARG0   0x003
#define VX_DCR_STARTUP_ARG1   0x004
#define VX_DCR_MPM_CLASS      0x005

/* ---------------------------------------------------------------- */

typedef struct {
    int placeholder;
} VortexBackendPriv;

/* ---- 向 vortex RAM 写入一个 uint32_t ---- */
static void meta_write_u32(uint64_t addr, uint32_t value)
{
    uint8_t buf[4];
    buf[0] = (uint8_t)(value);
    buf[1] = (uint8_t)(value >> 8);
    buf[2] = (uint8_t)(value >> 16);
    buf[3] = (uint8_t)(value >> 24);
    vortex_wrapper_write_ram(addr, buf, 4);
}

/* ---- 写入 SIMT metadata 的 block_dim 和 grid_dim (per-dispatch 不变) ---- */
static void meta_write_dims(const struct gpgpu_dispatch_params *p)
{
    meta_write_u32(VORTEX_META_ADDR + META_BLOCK_DIM_X, p->block_dim[0]);
    meta_write_u32(VORTEX_META_ADDR + META_BLOCK_DIM_Y, p->block_dim[1]);
    meta_write_u32(VORTEX_META_ADDR + META_BLOCK_DIM_Z, p->block_dim[2]);
    meta_write_u32(VORTEX_META_ADDR + META_GRID_DIM_X,  p->grid_dim[0]);
    meta_write_u32(VORTEX_META_ADDR + META_GRID_DIM_Y,  p->grid_dim[1]);
    meta_write_u32(VORTEX_META_ADDR + META_GRID_DIM_Z,  p->grid_dim[2]);
}

/* ---- 写入 SIMT metadata 的 block_id (per-block 变化) ---- */
static void meta_write_block_id(uint32_t bx, uint32_t by, uint32_t bz)
{
    meta_write_u32(VORTEX_META_ADDR + META_BLOCK_ID_X, bx);
    meta_write_u32(VORTEX_META_ADDR + META_BLOCK_ID_Y, by);
    meta_write_u32(VORTEX_META_ADDR + META_BLOCK_ID_Z, bz);
}

/* ---- 配置 DCR 并执行一次 ---- */
static int vortex_exec_one_block(void)
{
    uint32_t startup_lo, startup_hi;
    uint32_t args_lo, args_hi;

    startup_lo = (uint32_t)(VORTEX_STARTUP_ADDR & 0xffffffff);
    startup_hi = (uint32_t)(VORTEX_STARTUP_ADDR >> 32);
    vortex_wrapper_dcr_write(VX_DCR_STARTUP_ADDR0, startup_lo);
    vortex_wrapper_dcr_write(VX_DCR_STARTUP_ADDR1, startup_hi);

    args_lo = (uint32_t)(VORTEX_ARGS_ADDR & 0xffffffff);
    args_hi = (uint32_t)(VORTEX_ARGS_ADDR >> 32);
    vortex_wrapper_dcr_write(VX_DCR_STARTUP_ARG0, args_lo);
    vortex_wrapper_dcr_write(VX_DCR_STARTUP_ARG1, args_hi);

    vortex_wrapper_dcr_write(VX_DCR_MPM_CLASS, 0);

    return vortex_wrapper_run();
}

/* ---------------------------------------------------------------- */

/*
 * vortex_dispatch — 逐 block 串行执行 SIMT 内核。
 *
 * 已知问题 (2026-07-12):
 *   XLEN=32 修复后 basic_smoke (grid=1x1x1, block=1x1x1) 可通过,
 *   但多 block kernel (如 thread_add grid=2x3x4) 会在 Processor::run()
 *   中挂死, 疑似 vortex 模拟器内部在 multi-block 或特定指令序列下
 *   无法正常退出。需要进一步排查 emulator run loop 的退出条件。
 */
static int vortex_dispatch(struct gpgpu_backend *be, void *dev,
                           const struct gpgpu_dispatch_params *p)
{
    uint8_t *vram       = be->dev_iface->vram_ptr(dev);
    uint64_t vram_size  = be->dev_iface->vram_size(dev);
    uint8_t  buf[KERNEL_BUF_SIZE];
    uint8_t  args[ARGS_BUF_SIZE];
    uint8_t  result[RESULT_BUF_SIZE];
    uint32_t bx, by, bz;
    int      exit_code;

    /* ---- 参数校验 ---- */
    if (!vram) {
        return -1;
    }
    if (p->kernel_addr + KERNEL_BUF_SIZE > vram_size) {
        return -1;
    }
    if (p->kernel_args + ARGS_BUF_SIZE > vram_size) {
        return -1;
    }

    /* 1. 加载 kernel code 并前插 trampoline (csrr a0, mscratch) */
    *(uint32_t *)buf = TRAMPOLINE_INSN;
    memcpy(buf + TRAMPOLINE_SIZE, vram + p->kernel_addr,
           KERNEL_BUF_SIZE - TRAMPOLINE_SIZE);
    vortex_wrapper_load_kernel(buf, KERNEL_BUF_SIZE);

    /* 2. 写入 SIMT metadata 中不变的部分: block_dim, grid_dim */
    meta_write_dims(p);

    /* 3. 遍历所有 block */
    for (bz = 0; bz < p->grid_dim[2]; bz++) {
        for (by = 0; by < p->grid_dim[1]; by++) {
            for (bx = 0; bx < p->grid_dim[0]; bx++) {

                /* 3a. 写入当前 block_id */
                meta_write_block_id(bx, by, bz);

                /* 3b. 加载 kernel args (每 block 重新加载，因为可能被上一 block 修改) */
                memcpy(args, vram + p->kernel_args, ARGS_BUF_SIZE);
                vortex_wrapper_load_args(args, ARGS_BUF_SIZE);

                /* 3c. 配置 DCR 并执行 */
                exit_code = vortex_exec_one_block();
                if (exit_code != 0) {
                    return exit_code;
                }

                /* 3d. 读回结果 (IO_MPM_ADDR 区域的 exit code / debug 输出) */
                vortex_wrapper_read_results(result, RESULT_BUF_SIZE);

                /* 3e. 同步 vortex RAM → QEMU VRAM:
                 *     kernel 写入的数据在 vortex RAM 的低地址堆区域 */
                {
                    uint32_t sync_n = SYNC_HEAP_SIZE;
                    if (sync_n > vram_size) {
                        sync_n = (uint32_t)vram_size;
                    }
                    vortex_wrapper_read_ram(VORTEX_ARGS_ADDR,
                                            vram + p->kernel_args,
                                            ARGS_BUF_SIZE);
                    vortex_wrapper_read_ram(0, vram, sync_n);
                }
            }
        }
    }

    return 0;
}

static void vortex_get_caps(struct gpgpu_backend *be, uint32_t *caps)
{
    (void)be;
    *caps = GPGPU_BACKEND_CAP_BARRIER |
            GPGPU_BACKEND_CAP_SHARED_MEM |
            GPGPU_BACKEND_CAP_DIVERGENCE;
}

/* ---------------------------------------------------------------- */

static int vortex_init(struct gpgpu_backend *be, void *dev)
{
    (void)dev;
    VortexBackendPriv *priv = g_new0(VortexBackendPriv, 1);
    be->priv = priv;

    return vortex_wrapper_init();
}

static void vortex_destroy(struct gpgpu_backend *be)
{
    vortex_wrapper_destroy();
    g_free(be->priv);
    be->priv = NULL;
}

/* ---------------------------------------------------------------- */

static const struct gpgpu_backend_ops vortex_ops = {
    .name     = "vortex",
    .init     = vortex_init,
    .destroy  = vortex_destroy,
    .dispatch = vortex_dispatch,
    .get_caps = vortex_get_caps,
};

/* ---------------------------------------------------------------- */

struct gpgpu_backend *gpgpu_backend_vortex_new(void)
{
    struct gpgpu_backend *be = g_new0(struct gpgpu_backend, 1);
    be->ops = &vortex_ops;
    return be;
}
