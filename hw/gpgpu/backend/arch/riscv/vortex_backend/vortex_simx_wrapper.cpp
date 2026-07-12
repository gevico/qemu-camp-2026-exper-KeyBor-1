/*
 * QEMU Educational GPGPU - Vortex SimX C++ Wrapper
 *
 * 单例持有 Processor + RAM + Arch，对外暴露 extern "C" API。
 *
 * 常量值对应 VX_config.vh (XLEN=32, NUM_THREADS=4, NUM_WARPS=4, NUM_CORES=1):
 *   MEM_PAGE_SIZE  = 4096
 *   STARTUP_ADDR   = 0x80000000
 *   ARGS_ADDR      = 0x80100000  (STARTUP_ADDR + 1MB)
 *   IO_MPM_ADDR    = 0x00000080  (IO_BASE_ADDR + IO_COUT_SIZE = 0x40 + 64)
 *
 * DCR 偏移来源 VX_types.vh:
 *   VX_DCR_BASE_STARTUP_ADDR0 = 0x001
 *   VX_DCR_BASE_STARTUP_ADDR1 = 0x002
 *   VX_DCR_BASE_STARTUP_ARG0  = 0x003
 *   VX_DCR_BASE_STARTUP_ARG1  = 0x004
 *   VX_DCR_BASE_MPM_CLASS     = 0x005
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "vortex_simx_wrapper.h"

#include "processor.h"
#include "arch.h"
#include "mem.h"
#include "VX_config.h"
#include "VX_types.h"
#include "constants.h"
#include <cstring>

/* ================================================================
 * 地址常量 (与 VX_config.vh XLEN=32 配置一致)
 * ================================================================ */
static constexpr uint64_t kStartupAddr = 0x80000000ULL;
static constexpr uint64_t kArgsAddr    = 0x80100000ULL;  /* STARTUP_ADDR + 1MB */
static constexpr uint64_t kIOMpmAddr   = 0x00000080ULL;  /* IO_BASE_ADDR + IO_COUT_SIZE */

/* ================================================================
 * 全局单例状态
 * ================================================================ */
static vortex::Arch      *g_arch      = nullptr;
static vortex::RAM       *g_ram       = nullptr;
static vortex::Processor *g_processor = nullptr;

/* ================================================================
 * 生命周期
 * ================================================================ */

int vortex_wrapper_init(void)
{
    if (g_processor) {
        return 0;  /* 已初始化 */
    }

    g_arch      = new vortex::Arch(NUM_THREADS, NUM_WARPS, NUM_CORES);
    g_ram       = new vortex::RAM(0, MEM_PAGE_SIZE);
    g_processor = new vortex::Processor(*g_arch);

    g_processor->attach_ram(g_ram);
    return 0;
}

void vortex_wrapper_destroy(void)
{
    delete g_processor;
    g_processor = nullptr;

    delete g_ram;
    g_ram = nullptr;

    delete g_arch;
    g_arch = nullptr;
}

/* ================================================================
 * 数据传输
 * ================================================================ */

void vortex_wrapper_load_kernel(const uint8_t *data, uint32_t len)
{
    if (g_ram && data && len > 0) {
        g_ram->write(data, kStartupAddr, len);
    }
}

void vortex_wrapper_load_args(const uint8_t *data, uint32_t len)
{
    if (g_ram && data && len > 0) {
        g_ram->write(data, kArgsAddr, len);
    }
}

void vortex_wrapper_read_results(uint8_t *buf, uint32_t len)
{
    if (g_ram && buf && len > 0) {
        g_ram->read(buf, kIOMpmAddr, len);
    }
}

void vortex_wrapper_write_ram(uint64_t addr, const uint8_t *data, uint32_t len)
{
    if (g_ram && data && len > 0) {
        g_ram->write(data, addr, len);
    }
}

void vortex_wrapper_read_ram(uint64_t addr, uint8_t *buf, uint32_t len)
{
    if (g_ram && buf && len > 0) {
        g_ram->read(buf, addr, len);
    }
}

/* ================================================================
 * 配置
 * ================================================================ */

void vortex_wrapper_dcr_write(uint32_t addr, uint32_t value)
{
    if (g_processor) {
        g_processor->dcr_write(addr, value);
    }
}

/* ================================================================
 * 执行
 * ================================================================ */

int vortex_wrapper_run(void)
{
    if (!g_processor) {
        return -1;
    }

    /*
     * 初始化 exit_code 为 0 (success)。
     * vortex 裸机内核的 crt0.S 会写入 exit_code 到 IO_MPM_ADDR + 8，
     * 但我们加载的是原始 .text 段没有启动代码，不会写这个地址。
     * 如果内核确实写入了非零 exit_code, 会覆盖此处的 0。
     */
    int init_exit = 0;
    g_ram->write(&init_exit, kIOMpmAddr + 8, sizeof(init_exit));

    g_processor->run();

    int exit_code = 0;
    g_ram->read(&exit_code, kIOMpmAddr + 8, sizeof(exit_code));
    return exit_code;
}
