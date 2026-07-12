/*
 * QEMU Educational GPGPU - Vortex SimX C Wrapper API
 *
 * extern "C" 包装层，隔离 QEMU C 代码和 Vortex C++ 代码。
 * 单例模式管理 Processor + RAM 生命周期。
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#ifndef __VORTEX_SIMX_WRAPPER_H__
#define __VORTEX_SIMX_WRAPPER_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 生命周期 ---- */
int  vortex_wrapper_init(void);
void vortex_wrapper_destroy(void);

/* ---- 数据传输 ---- */
void vortex_wrapper_load_kernel(const uint8_t *data, uint32_t len);
void vortex_wrapper_load_args(const uint8_t *data, uint32_t len);
void vortex_wrapper_read_results(uint8_t *buf, uint32_t len);
void vortex_wrapper_write_ram(uint64_t addr, const uint8_t *data, uint32_t len);
void vortex_wrapper_read_ram(uint64_t addr, uint8_t *buf, uint32_t len);

/* ---- 配置 ---- */
void vortex_wrapper_dcr_write(uint32_t addr, uint32_t value);

/* ---- 执行 (阻塞) ---- */
int  vortex_wrapper_run(void);

#ifdef __cplusplus
}
#endif

#endif /* __VORTEX_SIMX_WRAPPER_H__ */
