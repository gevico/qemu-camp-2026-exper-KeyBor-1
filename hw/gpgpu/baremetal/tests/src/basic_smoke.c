#include "test_common.h"

int run_basic_smoke(GPGPURuntimeDevice *dev)
{
    uint32_t kernel_addr;
    uint32_t args_addr;
    uint32_t out_addr;
    uint32_t result = 0;
    uint32_t sentinel = 0xdeadbeef;
    uint32_t args[2];
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

    ret = gpgpu_malloc(dev, &out_addr, sizeof(result));
    if (ret < 0) {
        report_ret("gpgpu_malloc(out)", ret);
        return ret;
    }
    trace_u32("gpgpu out_addr", out_addr);
    ret = gpgpu_write(dev, out_addr, &sentinel, sizeof(sentinel));
    if (ret < 0) {
        report_ret("gpgpu_write(out sentinel)", ret);
        return ret;
    }

    args[0] = out_addr;
    args[1] = 0x1234abcd;

    ret = gpgpu_upload_kernel(dev, &kernel_addr, kernel_code,
                              sizeof(kernel_code) / sizeof(kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel", ret);
        return ret;
    }
    trace_u32("gpgpu kernel_addr", kernel_addr);

    ret = gpgpu_upload_args(dev, &args_addr, args, sizeof(args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args", ret);
        return ret;
    }
    trace_u32("gpgpu args_addr", args_addr);
    trace_u32("gpgpu user_args[0]", args[0]);
    trace_u32("gpgpu user_args[1]", args[1]);
    ret = gpgpu_read(dev, out_addr, &result, sizeof(result));
    if (ret < 0) {
        report_ret("gpgpu_read(before)", ret);
        return ret;
    }
    trace_u32("gpgpu result_before", result);

    ret = gpgpu_launch(dev, kernel_addr, args_addr,
                       (GPGPURuntimeDim3){ 1, 1, 1 },
                       (GPGPURuntimeDim3){ 1, 1, 1 });
    trace_u32("gpgpu global_status", ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu error_status", ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch", ret);
        return ret;
    }

    ret = gpgpu_read(dev, out_addr, &result, sizeof(result));
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

    return 0;
}
