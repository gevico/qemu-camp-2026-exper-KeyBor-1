#include "qemu_virt_platform.h"
#include "gpgpu_pci.h"
#include "gpgpu_nn.h"

#include <stddef.h>
#include <stdint.h>

#define UART0_BASE 0x10000000UL
#define UART_THR   0x00
#define UART_LSR   0x05
#define UART_LSR_THRE 0x20

#define GPGPU_REG_GLOBAL_STATUS 0x0104
#define GPGPU_REG_ERROR_STATUS  0x0108

#define RELU_TEST_LEN 16
#define LINEAR_IN_FEATURES 4
#define LINEAR_OUT_FEATURES 3

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
#include "build/relu_i32_kernel.inc"
#include "build/linear_i32_kernel.inc"

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
    int32_t relu_input[RELU_TEST_LEN] = {
        -7, -1, 0, 1, 2, -3, 9, -11,
        16, -5, 4, 0, -32, 8, -2, 6,
    };
    int32_t relu_output[RELU_TEST_LEN];
    int32_t relu_expected[RELU_TEST_LEN];
    uint32_t relu_input_addr;
    uint32_t relu_output_addr;
    uint32_t relu_kernel_addr;
    uint32_t relu_args_addr;
    GPGPUReluArgs relu_args;
    int relu_ok = 1;
    int32_t linear_input[LINEAR_IN_FEATURES] = { 2, -1, 3, 4 };
    int32_t linear_weight[LINEAR_OUT_FEATURES * LINEAR_IN_FEATURES] = {
        1, 2, 3, 4,
        -1, 0, 2, 1,
        3, -2, 1, 0,
    };
    int32_t linear_bias[LINEAR_OUT_FEATURES] = { 5, -3, 7 };
    int32_t linear_output[LINEAR_OUT_FEATURES];
    int32_t linear_expected[LINEAR_OUT_FEATURES];
    uint32_t linear_input_addr;
    uint32_t linear_weight_addr;
    uint32_t linear_bias_addr;
    uint32_t linear_output_addr;
    uint32_t linear_kernel_addr;
    uint32_t linear_args_addr;
    GPGPULinearArgs linear_args;
    int linear_ok = 1;
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

    ret = gpgpu_upload_args(&dev, &args_addr, args, sizeof(args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args", ret);
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
    ret = gpgpu_upload_args(&dev, &thread_args_addr, thread_args,
                            sizeof(thread_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(thread_add)", ret);
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
    if (!thread_ok) {
        return 1;
    }

    for (uint32_t i = 0; i < RELU_TEST_LEN; ++i) {
        relu_output[i] = 0x77777777;
        relu_expected[i] = relu_input[i] < 0 ? 0 : relu_input[i];
    }

    ret = gpgpu_malloc(&dev, &relu_input_addr, sizeof(relu_input));
    if (ret < 0) {
        report_ret("gpgpu_malloc(relu_input)", ret);
        return ret;
    }
    ret = gpgpu_malloc(&dev, &relu_output_addr, sizeof(relu_output));
    if (ret < 0) {
        report_ret("gpgpu_malloc(relu_output)", ret);
        return ret;
    }
    ret = gpgpu_write(&dev, relu_input_addr, relu_input, sizeof(relu_input));
    if (ret < 0) {
        report_ret("gpgpu_write(relu_input)", ret);
        return ret;
    }
    ret = gpgpu_write(&dev, relu_output_addr, relu_output, sizeof(relu_output));
    if (ret < 0) {
        report_ret("gpgpu_write(relu_output)", ret);
        return ret;
    }

    relu_args = (GPGPUReluArgs) {
        .input = gpgpu_tensor_make_1d_i32(relu_input_addr, RELU_TEST_LEN),
        .output = gpgpu_tensor_make_1d_i32(relu_output_addr, RELU_TEST_LEN),
    };

    ret = gpgpu_upload_kernel(&dev, &relu_kernel_addr,
                              relu_i32_kernel_code,
                              sizeof(relu_i32_kernel_code) /
                              sizeof(relu_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(relu_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(&dev, &relu_args_addr, &relu_args,
                            sizeof(relu_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(relu_i32)", ret);
        return ret;
    }

    ret = gpgpu_launch(&dev, relu_kernel_addr, relu_args_addr,
                       (GPGPURuntimeDim3){ 2, 1, 1 },
                       (GPGPURuntimeDim3){ 8, 1, 1 });
    trace_u32("gpgpu relu_i32 global_status",
              ctrl_read32(&dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu relu_i32 error_status",
              ctrl_read32(&dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(relu_i32)", ret);
        return ret;
    }

    ret = gpgpu_read(&dev, relu_output_addr, relu_output,
                     sizeof(relu_output));
    if (ret < 0) {
        report_ret("gpgpu_read(relu_i32)", ret);
        return ret;
    }

    for (uint32_t i = 0; i < RELU_TEST_LEN; ++i) {
        uart_puts("gpgpu relu_i32[");
        uart_puthex32(i);
        uart_puts("]=");
        uart_puthex32((uint32_t)relu_output[i]);
        uart_puts("\n");
        if (relu_output[i] != relu_expected[i]) {
            relu_ok = 0;
        }
    }

    uart_puts("gpgpu relu_i32 ");
    uart_puts(relu_ok ? "PASS\n" : "FAIL\n");
    if (!relu_ok) {
        return 1;
    }

    for (uint32_t o = 0; o < LINEAR_OUT_FEATURES; ++o) {
        int32_t acc = linear_bias[o];

        linear_output[o] = 0x55555555;
        for (uint32_t i = 0; i < LINEAR_IN_FEATURES; ++i) {
            acc += linear_input[i] *
                   linear_weight[o * LINEAR_IN_FEATURES + i];
        }
        linear_expected[o] = acc;
    }

    ret = gpgpu_malloc(&dev, &linear_input_addr, sizeof(linear_input));
    if (ret < 0) {
        report_ret("gpgpu_malloc(linear_input)", ret);
        return ret;
    }
    ret = gpgpu_malloc(&dev, &linear_weight_addr, sizeof(linear_weight));
    if (ret < 0) {
        report_ret("gpgpu_malloc(linear_weight)", ret);
        return ret;
    }
    ret = gpgpu_malloc(&dev, &linear_bias_addr, sizeof(linear_bias));
    if (ret < 0) {
        report_ret("gpgpu_malloc(linear_bias)", ret);
        return ret;
    }
    ret = gpgpu_malloc(&dev, &linear_output_addr, sizeof(linear_output));
    if (ret < 0) {
        report_ret("gpgpu_malloc(linear_output)", ret);
        return ret;
    }
    ret = gpgpu_write(&dev, linear_input_addr, linear_input,
                      sizeof(linear_input));
    if (ret < 0) {
        report_ret("gpgpu_write(linear_input)", ret);
        return ret;
    }
    ret = gpgpu_write(&dev, linear_weight_addr, linear_weight,
                      sizeof(linear_weight));
    if (ret < 0) {
        report_ret("gpgpu_write(linear_weight)", ret);
        return ret;
    }
    ret = gpgpu_write(&dev, linear_bias_addr, linear_bias,
                      sizeof(linear_bias));
    if (ret < 0) {
        report_ret("gpgpu_write(linear_bias)", ret);
        return ret;
    }
    ret = gpgpu_write(&dev, linear_output_addr, linear_output,
                      sizeof(linear_output));
    if (ret < 0) {
        report_ret("gpgpu_write(linear_output)", ret);
        return ret;
    }

    linear_args = (GPGPULinearArgs) {
        .input = gpgpu_tensor_make_1d_i32(linear_input_addr,
                                          LINEAR_IN_FEATURES),
        .weight = gpgpu_tensor_make_oi_i32(linear_weight_addr,
                                           LINEAR_OUT_FEATURES,
                                           LINEAR_IN_FEATURES),
        .bias = gpgpu_tensor_make_1d_i32(linear_bias_addr,
                                         LINEAR_OUT_FEATURES),
        .output = gpgpu_tensor_make_1d_i32(linear_output_addr,
                                           LINEAR_OUT_FEATURES),
        .in_features = LINEAR_IN_FEATURES,
        .out_features = LINEAR_OUT_FEATURES,
    };

    ret = gpgpu_upload_kernel(&dev, &linear_kernel_addr,
                              linear_i32_kernel_code,
                              sizeof(linear_i32_kernel_code) /
                              sizeof(linear_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(linear_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(&dev, &linear_args_addr, &linear_args,
                            sizeof(linear_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(linear_i32)", ret);
        return ret;
    }

    ret = gpgpu_launch(&dev, linear_kernel_addr, linear_args_addr,
                       (GPGPURuntimeDim3){ 1, 1, 1 },
                       (GPGPURuntimeDim3){ 4, 1, 1 });
    trace_u32("gpgpu linear_i32 global_status",
              ctrl_read32(&dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu linear_i32 error_status",
              ctrl_read32(&dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(linear_i32)", ret);
        return ret;
    }

    ret = gpgpu_read(&dev, linear_output_addr, linear_output,
                     sizeof(linear_output));
    if (ret < 0) {
        report_ret("gpgpu_read(linear_i32)", ret);
        return ret;
    }

    for (uint32_t i = 0; i < LINEAR_OUT_FEATURES; ++i) {
        uart_puts("gpgpu linear_i32[");
        uart_puthex32(i);
        uart_puts("]=");
        uart_puthex32((uint32_t)linear_output[i]);
        uart_puts("\n");
        if (linear_output[i] != linear_expected[i]) {
            linear_ok = 0;
        }
    }

    uart_puts("gpgpu linear_i32 ");
    uart_puts(linear_ok ? "PASS\n" : "FAIL\n");
    return linear_ok ? 0 : 1;
}
