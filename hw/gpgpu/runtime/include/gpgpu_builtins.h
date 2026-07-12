/*
 * GPGPU portable kernel builtins
 *
 * 同一份 kernel 源码，通过条件编译区分后端：
 *   - rv32 interpreter: 从 MMIO 0x80000000 读取 thread/block ID
 *   - vortex: 从 CSR 读取 thread_id (0xCC0) / warp_id (0xCC1),
 *     block_id / block_dim / grid_dim 由 backend 注入 RAM 固定地址
 *
 * 证据: docs/vortex-csr-evidence.md
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#ifndef GPGPU_BUILTINS_H
#define GPGPU_BUILTINS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Vortex backend builtins
 *
 * CSR 来源 VX_types.vh:
 *   VX_CSR_THREAD_ID   = 0xCC0  (warp 内线程 ID)
 *   VX_CSR_WARP_ID     = 0xCC1  (core 内 warp ID)
 *   VX_CSR_NUM_THREADS = 0xFC0  (每 warp 线程数)
 *   VX_CSR_NUM_WARPS   = 0xFC1  (每 core warp 数)
 *
 * emulator.cpp:504-512 确认了这些 CSR 的实现。
 *
 * block_id / block_dim / grid_dim: vortex 无硬件 CSR，
 * 由 backend adapter 在执行前写入 SIMT metadata 区域。
 * ================================================================ */
#ifdef GPGPU_BACKEND_VORTEX

/* SIMT metadata 注入地址: STARTUP_ADDR (0x80000000) 下方 4KB */
#define GPGPU_VORTEX_META_BASE 0x7FFFF000u

/* offset 与 rv32 CTRL 寄存器保持一致 */
#define VORTEX_META_BLOCK_ID_X  0x10u
#define VORTEX_META_BLOCK_ID_Y  0x14u
#define VORTEX_META_BLOCK_ID_Z  0x18u
#define VORTEX_META_BLOCK_DIM_X 0x20u
#define VORTEX_META_BLOCK_DIM_Y 0x24u
#define VORTEX_META_BLOCK_DIM_Z 0x28u
#define VORTEX_META_GRID_DIM_X  0x30u
#define VORTEX_META_GRID_DIM_Y  0x34u
#define VORTEX_META_GRID_DIM_Z  0x38u

/* ---- 来自 CSR 的 builtins ---- */

static inline uint32_t gpgpu_thread_id_x(void) {
    uint32_t v;
    __asm__ volatile("csrr %0, 0xCC0" : "=r"(v));
    return v;
}

/*
 * vortex 是 1D 线程模型，无 thread_id_y / thread_id_z。
 * 返回 0 保持与 rv32 接口兼容。
 */
static inline uint32_t gpgpu_thread_id_y(void) {
    (void)0; /* unused */
    return 0;
}

static inline uint32_t gpgpu_thread_id_z(void) {
    (void)0;
    return 0;
}

static inline uint32_t gpgpu_warp_id(void) {
    uint32_t v;
    __asm__ volatile("csrr %0, 0xCC1" : "=r"(v));
    return v;
}

static inline uint32_t gpgpu_num_threads(void) {
    uint32_t v;
    __asm__ volatile("csrr %0, 0xFC0" : "=r"(v));
    return v;
}

static inline uint32_t gpgpu_num_warps(void) {
    uint32_t v;
    __asm__ volatile("csrr %0, 0xFC1" : "=r"(v));
    return v;
}

/* ---- 来自注入 RAM 的 builtins ---- */

static inline uint32_t gpgpu_block_id_x(void) {
    return *(volatile uint32_t *)(GPGPU_VORTEX_META_BASE + VORTEX_META_BLOCK_ID_X);
}

static inline uint32_t gpgpu_block_id_y(void) {
    return *(volatile uint32_t *)(GPGPU_VORTEX_META_BASE + VORTEX_META_BLOCK_ID_Y);
}

static inline uint32_t gpgpu_block_id_z(void) {
    return *(volatile uint32_t *)(GPGPU_VORTEX_META_BASE + VORTEX_META_BLOCK_ID_Z);
}

static inline uint32_t gpgpu_block_dim_x(void) {
    return *(volatile uint32_t *)(GPGPU_VORTEX_META_BASE + VORTEX_META_BLOCK_DIM_X);
}

static inline uint32_t gpgpu_block_dim_y(void) {
    return *(volatile uint32_t *)(GPGPU_VORTEX_META_BASE + VORTEX_META_BLOCK_DIM_Y);
}

static inline uint32_t gpgpu_block_dim_z(void) {
    return *(volatile uint32_t *)(GPGPU_VORTEX_META_BASE + VORTEX_META_BLOCK_DIM_Z);
}

static inline uint32_t gpgpu_grid_dim_x(void) {
    return *(volatile uint32_t *)(GPGPU_VORTEX_META_BASE + VORTEX_META_GRID_DIM_X);
}

static inline uint32_t gpgpu_grid_dim_y(void) {
    return *(volatile uint32_t *)(GPGPU_VORTEX_META_BASE + VORTEX_META_GRID_DIM_Y);
}

static inline uint32_t gpgpu_grid_dim_z(void) {
    return *(volatile uint32_t *)(GPGPU_VORTEX_META_BASE + VORTEX_META_GRID_DIM_Z);
}

/* ================================================================
 * rv32 Interpreter backend builtins (default)
 *
 * 通过 MMIO 读取 0x80000000 + offset 的 CTRL 寄存器。
 * 寄存器映射来源 gpgpu_core.h:104-116.
 * ================================================================ */
#else

#define GPGPU_CORE_CTRL_BASE 0x80000000u

#define CTRL_THREAD_ID_X 0x00u
#define CTRL_THREAD_ID_Y 0x04u
#define CTRL_THREAD_ID_Z 0x08u
#define CTRL_BLOCK_ID_X  0x10u
#define CTRL_BLOCK_ID_Y  0x14u
#define CTRL_BLOCK_ID_Z  0x18u
#define CTRL_BLOCK_DIM_X 0x20u
#define CTRL_BLOCK_DIM_Y 0x24u
#define CTRL_BLOCK_DIM_Z 0x28u
#define CTRL_GRID_DIM_X  0x30u
#define CTRL_GRID_DIM_Y  0x34u
#define CTRL_GRID_DIM_Z  0x38u

static inline uint32_t gpgpu_thread_id_x(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_THREAD_ID_X / sizeof(uint32_t)];
}

static inline uint32_t gpgpu_thread_id_y(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_THREAD_ID_Y / sizeof(uint32_t)];
}

static inline uint32_t gpgpu_thread_id_z(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_THREAD_ID_Z / sizeof(uint32_t)];
}

static inline uint32_t gpgpu_block_id_x(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_BLOCK_ID_X / sizeof(uint32_t)];
}

static inline uint32_t gpgpu_block_id_y(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_BLOCK_ID_Y / sizeof(uint32_t)];
}

static inline uint32_t gpgpu_block_id_z(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_BLOCK_ID_Z / sizeof(uint32_t)];
}

static inline uint32_t gpgpu_block_dim_x(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_BLOCK_DIM_X / sizeof(uint32_t)];
}

static inline uint32_t gpgpu_block_dim_y(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_BLOCK_DIM_Y / sizeof(uint32_t)];
}

static inline uint32_t gpgpu_block_dim_z(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_BLOCK_DIM_Z / sizeof(uint32_t)];
}

static inline uint32_t gpgpu_grid_dim_x(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_GRID_DIM_X / sizeof(uint32_t)];
}

static inline uint32_t gpgpu_grid_dim_y(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_GRID_DIM_Y / sizeof(uint32_t)];
}

static inline uint32_t gpgpu_grid_dim_z(void) {
    volatile uint32_t *ctrl = (volatile uint32_t *)GPGPU_CORE_CTRL_BASE;
    return ctrl[CTRL_GRID_DIM_Z / sizeof(uint32_t)];
}

#endif /* GPGPU_BACKEND_VORTEX */

#ifdef __cplusplus
}
#endif

#endif /* GPGPU_BUILTINS_H */
