#include "test_common.h"

int run_thread_add_smoke(GPGPURuntimeDevice *dev)
{
    uint32_t thread_args[3];
    uint32_t thread_results[64];
    uint32_t thread_out_addr;
    uint32_t thread_kernel_addr;
    uint32_t thread_args_addr;
    int thread_ok = 1;
    int ret;

    for (uint32_t i = 0; i < sizeof(thread_results) / sizeof(thread_results[0]); ++i) {
        thread_results[i] = 0;
    }

    ret = gpgpu_malloc(dev, &thread_out_addr, sizeof(thread_results));
    if (ret < 0) {
        report_ret("gpgpu_malloc(thread_out)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, thread_out_addr, thread_results,
                      sizeof(thread_results));
    if (ret < 0) {
        report_ret("gpgpu_write(thread_out zero)", ret);
        return ret;
    }
    thread_args[0] = thread_out_addr;
    thread_args[1] = 1;
    thread_args[2] = 1;

    ret = gpgpu_upload_kernel(dev, &thread_kernel_addr,
                              thread_add_kernel_code,
                              sizeof(thread_add_kernel_code) /
                              sizeof(thread_add_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(thread_add)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &thread_args_addr, thread_args,
                            sizeof(thread_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(thread_add)", ret);
        return ret;
    }

    ret = gpgpu_launch(dev, thread_kernel_addr, thread_args_addr,
                       (GPGPURuntimeDim3){ 2, 2, 2 },
                       (GPGPURuntimeDim3){ 2, 2, 2 });
    trace_u32("gpgpu thread_add global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu thread_add error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(thread_add)", ret);
        return ret;
    }

    ret = gpgpu_read(dev, thread_out_addr, thread_results,
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

    return 0;
}
