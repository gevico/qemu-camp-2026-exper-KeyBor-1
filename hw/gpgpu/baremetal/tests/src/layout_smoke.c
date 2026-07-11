#include "test_common.h"

int run_layout_smoke(GPGPURuntimeDevice *dev)
{
    int32_t layout_mo[LAYOUT_N * LAYOUT_H * LAYOUT_W * LAYOUT_C] = {
        10, 20,
        11, 21,
        12, 22,
        13, 23,
    };
    int32_t layout_nchw[LAYOUT_N * LAYOUT_C * LAYOUT_H * LAYOUT_W];
    int32_t layout_expected[LAYOUT_N * LAYOUT_C * LAYOUT_H * LAYOUT_W] = {
        10, 11, 12, 13,
        20, 21, 22, 23,
    };
    uint32_t layout_mo_addr;
    uint32_t layout_nchw_addr;
    uint32_t layout_kernel_addr;
    uint32_t layout_args_addr;
    GPGPUMoToNchwArgs layout_args;
    int layout_ok = 1;
    int ret;

    for (uint32_t i = 0; i < LAYOUT_N * LAYOUT_C * LAYOUT_H * LAYOUT_W; ++i) {
        layout_nchw[i] = 0x77777777;
    }
    ret = gpgpu_malloc(dev, &layout_mo_addr, sizeof(layout_mo));
    if (ret < 0) {
        report_ret("gpgpu_malloc(layout_mo)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &layout_nchw_addr, sizeof(layout_nchw));
    if (ret < 0) {
        report_ret("gpgpu_malloc(layout_nchw)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, layout_mo_addr, layout_mo, sizeof(layout_mo));
    if (ret < 0) {
        report_ret("gpgpu_write(layout_mo)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, layout_nchw_addr, layout_nchw,
                      sizeof(layout_nchw));
    if (ret < 0) {
        report_ret("gpgpu_write(layout_nchw)", ret);
        return ret;
    }
    layout_args = (GPGPUMoToNchwArgs) {
        .input = gpgpu_tensor_make_mo_i32(layout_mo_addr,
                                          LAYOUT_N * LAYOUT_H * LAYOUT_W,
                                          LAYOUT_C),
        .output = gpgpu_tensor_make_nchw_i32(layout_nchw_addr, LAYOUT_N,
                                             LAYOUT_C, LAYOUT_H, LAYOUT_W),
        .n = LAYOUT_N,
        .c = LAYOUT_C,
        .h = LAYOUT_H,
        .w = LAYOUT_W,
    };
    ret = gpgpu_upload_kernel(dev, &layout_kernel_addr,
                              mo_to_nchw_i32_kernel_code,
                              sizeof(mo_to_nchw_i32_kernel_code) /
                              sizeof(mo_to_nchw_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(mo_to_nchw_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &layout_args_addr, &layout_args,
                            sizeof(layout_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(mo_to_nchw_i32)", ret);
        return ret;
    }
    ret = gpgpu_launch(dev, layout_kernel_addr, layout_args_addr,
                       (GPGPURuntimeDim3){ LAYOUT_N, LAYOUT_C, LAYOUT_H },
                       (GPGPURuntimeDim3){ LAYOUT_W, 1, 1 });
    trace_u32("gpgpu mo_to_nchw_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu mo_to_nchw_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(mo_to_nchw_i32)", ret);
        return ret;
    }
    ret = gpgpu_read(dev, layout_nchw_addr, layout_nchw,
                     sizeof(layout_nchw));
    if (ret < 0) {
        report_ret("gpgpu_read(mo_to_nchw_i32)", ret);
        return ret;
    }
    for (uint32_t i = 0; i < LAYOUT_N * LAYOUT_C * LAYOUT_H * LAYOUT_W; ++i) {
        uart_puts("gpgpu mo_to_nchw_i32[");
        uart_puthex32(i);
        uart_puts("]=");
        uart_puthex32((uint32_t)layout_nchw[i]);
        uart_puts("\n");
        if (layout_nchw[i] != layout_expected[i]) {
            layout_ok = 0;
        }
    }
    uart_puts("gpgpu mo_to_nchw_i32 ");
    uart_puts(layout_ok ? "PASS\n" : "FAIL\n");
    if (!layout_ok) {
        return 1;
    }

    return 0;
}
