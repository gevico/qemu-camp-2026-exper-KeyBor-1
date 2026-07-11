#include "qemu_virt_platform.h"
#include "gpgpu_pci.h"

#include <stddef.h>
#include <stdint.h>

#define UART0_BASE 0x10000000UL
#define UART_THR   0x00
#define UART_LSR   0x05
#define UART_LSR_THRE 0x20

#define GPGPU_REG_GLOBAL_STATUS 0x0104
#define GPGPU_REG_ERROR_STATUS  0x0108

#define RV_RD(rd)          ((uint32_t)(rd) << 7)
#define RV_RS1(rs1)        ((uint32_t)(rs1) << 15)
#define RV_RS2(rs2)        ((uint32_t)(rs2) << 20)
#define RV_LUI(rd, imm20)  (((uint32_t)(imm20) << 12) | RV_RD(rd) | 0x37)
#define RV_ADDI(rd, rs1, imm) \
    ((((uint32_t)(imm) & 0xfff) << 20) | RV_RS1(rs1) | RV_RD(rd) | 0x13)
#define RV_SLLI(rd, rs1, shamt) \
    (((uint32_t)(shamt) << 20) | RV_RS1(rs1) | (1u << 12) | RV_RD(rd) | 0x13)
#define RV_ADD(rd, rs1, rs2) \
    (RV_RS2(rs2) | RV_RS1(rs1) | RV_RD(rd) | 0x33)
#define RV_LW(rd, imm, rs1) \
    ((((uint32_t)(imm) & 0xfff) << 20) | RV_RS1(rs1) | (2u << 12) | RV_RD(rd) | 0x03)
#define RV_SW(rs2, imm, rs1) \
    (((((uint32_t)(imm) & 0xfe0) << 20) | RV_RS2(rs2) | RV_RS1(rs1) | \
      (2u << 12) | (((uint32_t)(imm) & 0x1f) << 7) | 0x23))
#define RV_EBREAK          0x00100073u

#include "build/thread_add_kernel.inc"

static void uart_putc(char c)
{
    volatile uint8_t *uart = (volatile uint8_t *)UART0_BASE;

    while ((uart[UART_LSR] & UART_LSR_THRE) == 0) {
    }
    uart[UART_THR] = (uint8_t)c;
}

static void uart_puts(const char *s)
{
    while (*s) {
        if (*s == '\n') {
            uart_putc('\r');
        }
        uart_putc(*s++);
    }
}

static void uart_puthex32(uint32_t value)
{
    static const char digits[] = "0123456789abcdef";

    uart_puts("0x");
    for (int shift = 28; shift >= 0; shift -= 4) {
        uart_putc(digits[(value >> shift) & 0xf]);
    }
}

void *memcpy(void *dst, const void *src, size_t n)
{
    uint8_t *d = dst;
    const uint8_t *s = src;

    while (n--) {
        *d++ = *s++;
    }
    return dst;
}

void *memset(void *dst, int c, size_t n)
{
    uint8_t *d = dst;

    while (n--) {
        *d++ = (uint8_t)c;
    }
    return dst;
}

static void report_ret(const char *op, int ret)
{
    uart_puts(op);
    uart_puts(" failed ret=");
    uart_puthex32((uint32_t)ret);
    uart_puts("\n");
}

static uint32_t ctrl_read32(GPGPURuntimeDevice *dev, uint32_t reg)
{
    volatile uint32_t *ptr = (volatile uint32_t *)(dev->ctrl + reg);

    return *ptr;
}

static void trace_u32(const char *name, uint32_t value)
{
    uart_puts(name);
    uart_puts("=");
    uart_puthex32(value);
    uart_puts("\n");
}

int main(void)
{
    GPGPURuntimeDevice dev;
    GPGPUPciDevice pci_dev;
    GPGPUPciPlatform platform = {
        .ecam_base = GPGPU_PLATFORM_ECAM_BASE,
        .mmio_base = GPGPU_PLATFORM_MMIO_BASE,
        .mmio_size = GPGPU_PLATFORM_MMIO_SIZE,
    };
    uint32_t kernel_addr;
    uint32_t args_addr;
    uint32_t out_addr;
    uint32_t result = 0;
    uint32_t sentinel = 0xdeadbeef;
    uint32_t args[2];
    uint32_t thread_args[3];
    uint32_t thread_results[64];
    uint32_t thread_out_addr;
    uint32_t thread_kernel_addr;
    uint32_t thread_args_addr;
    int thread_ok = 1;
    int ret;

    /*
     * Kernel ABI:
     *   x10/a0 = args VRAM offset
     *   user_args[0] = output VRAM offset
     *   user_args[1] = value
     *
     * RV32 code:
     *   lw x11, 0(x10)
     *   lw x12, 4(x10)
     *   sw x12, 0(x11)
     *   ebreak
     */
    static const uint32_t kernel_code[] = {
        RV_LW(11, 0, 10),
        RV_LW(12, 4, 10),
        RV_SW(12, 0, 11),
        RV_EBREAK,
    };

    uart_puts("gpgpu baremetal smoke start\n");

    ret = gpgpu_runtime_init_pci(&dev, &platform, &pci_dev);
    if (ret < 0) {
        report_ret("gpgpu_runtime_init_pci", ret);
        return ret;
    }

    uart_puts("gpgpu pci found bar0=");
    uart_puthex32((uint32_t)pci_dev.bar0_base);
    uart_puts(" size=");
    uart_puthex32(pci_dev.bar0_size);
    uart_puts(" bar2=");
    uart_puthex32((uint32_t)pci_dev.bar2_base);
    uart_puts(" size=");
    uart_puthex32(pci_dev.bar2_size);
    uart_puts("\n");
    trace_u32("gpgpu pci bdf",
              ((uint32_t)pci_dev.bus << 16) |
              ((uint32_t)pci_dev.device << 8) |
              pci_dev.function);
    trace_u32("gpgpu pci vendor_device",
              ((uint32_t)pci_dev.vendor_id << 16) | pci_dev.device_id);

    ret = gpgpu_malloc(&dev, &out_addr, sizeof(result));
    if (ret < 0) {
        report_ret("gpgpu_malloc(out)", ret);
        return ret;
    }
    trace_u32("gpgpu out_addr", out_addr);
    ret = gpgpu_write(&dev, out_addr, &sentinel, sizeof(sentinel));
    if (ret < 0) {
        report_ret("gpgpu_write(out sentinel)", ret);
        return ret;
    }

    args[0] = out_addr;
    args[1] = 0x1234abcd;

    ret = gpgpu_upload_kernel(&dev, &kernel_addr, kernel_code,
                              sizeof(kernel_code) / sizeof(kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel", ret);
        return ret;
    }
    trace_u32("gpgpu kernel_addr", kernel_addr);

    ret = gpgpu_pack_args(&dev, &args_addr, args,
                          sizeof(args) / sizeof(args[0]));
    if (ret < 0) {
        report_ret("gpgpu_pack_args", ret);
        return ret;
    }
    trace_u32("gpgpu args_addr", args_addr);
    trace_u32("gpgpu user_args[0]", args[0]);
    trace_u32("gpgpu user_args[1]", args[1]);
    ret = gpgpu_read(&dev, out_addr, &result, sizeof(result));
    if (ret < 0) {
        report_ret("gpgpu_read(before)", ret);
        return ret;
    }
    trace_u32("gpgpu result_before", result);

    ret = gpgpu_launch(&dev, kernel_addr, args_addr,
                       (GPGPURuntimeDim3){ 1, 1, 1 },
                       (GPGPURuntimeDim3){ 1, 1, 1 });
    trace_u32("gpgpu global_status", ctrl_read32(&dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu error_status", ctrl_read32(&dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch", ret);
        return ret;
    }

    ret = gpgpu_read(&dev, out_addr, &result, sizeof(result));
    if (ret < 0) {
        report_ret("gpgpu_read", ret);
        return ret;
    }

    uart_puts("gpgpu result=");
    uart_puthex32(result);
    if (result != args[1]) {
        uart_puts(" FAIL expected=");
        uart_puthex32(args[1]);
        uart_puts("\n");
        return 1;
    }
    uart_puts(" PASS\n");

    for (uint32_t i = 0; i < sizeof(thread_results) / sizeof(thread_results[0]); ++i) {
        thread_results[i] = 0;
    }

    ret = gpgpu_malloc(&dev, &thread_out_addr, sizeof(thread_results));
    if (ret < 0) {
        report_ret("gpgpu_malloc(thread_out)", ret);
        return ret;
    }
    ret = gpgpu_write(&dev, thread_out_addr, thread_results,
                      sizeof(thread_results));
    if (ret < 0) {
        report_ret("gpgpu_write(thread_out zero)", ret);
        return ret;
    }
    thread_args[0] = thread_out_addr;
    thread_args[1] = 1;
    thread_args[2] = 1;

    ret = gpgpu_upload_kernel(&dev, &thread_kernel_addr,
                              thread_add_kernel_code,
                              sizeof(thread_add_kernel_code) /
                              sizeof(thread_add_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(thread_add)", ret);
        return ret;
    }
    ret = gpgpu_pack_args(&dev, &thread_args_addr, thread_args,
                          sizeof(thread_args) / sizeof(thread_args[0]));
    if (ret < 0) {
        report_ret("gpgpu_pack_args(thread_add)", ret);
        return ret;
    }

    ret = gpgpu_launch(&dev, thread_kernel_addr, thread_args_addr,
                       (GPGPURuntimeDim3){ 2, 2, 2 },
                       (GPGPURuntimeDim3){ 2, 2, 2 });
    trace_u32("gpgpu thread_add global_status",
              ctrl_read32(&dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu thread_add error_status",
              ctrl_read32(&dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(thread_add)", ret);
        return ret;
    }

    ret = gpgpu_read(&dev, thread_out_addr, thread_results,
                     sizeof(thread_results));
    if (ret < 0) {
        report_ret("gpgpu_read(thread_add)", ret);
        return ret;
    }

    for (uint32_t i = 0; i < 64; ++i) {
        uart_puts("gpgpu thread_add[");
        uart_puthex32(i);
        uart_puts("]=");
        uart_puthex32(thread_results[i]);
        uart_puts("\n");
        if (thread_results[i] != i) {
            thread_ok = 0;
        }
    }

    uart_puts("gpgpu thread_add ");
    uart_puts(thread_ok ? "PASS\n" : "FAIL\n");
    return thread_ok ? 0 : 1;
}
