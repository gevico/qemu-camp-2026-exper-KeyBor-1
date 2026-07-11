#include "test_common.h"

int run_maxpool_smoke(GPGPURuntimeDevice *dev)
{
    int32_t pool_input[POOL_N * POOL_C * POOL_H * POOL_W] = {
        1, 3, 2, 4,
        5, 6, 7, 8,
        9, 1, 4, 2,
        3, 5, 6, 0,
    };
    int32_t pool_output[POOL_N * POOL_C * POOL_OUT_H * POOL_OUT_W];
    int32_t pool_expected[POOL_N * POOL_C * POOL_OUT_H * POOL_OUT_W];
    uint32_t pool_input_addr;
    uint32_t pool_output_addr;
    uint32_t pool_kernel_addr;
    uint32_t pool_args_addr;
    GPGPUMaxPool2DArgs pool_args;
    int pool_ok = 1;
    int ret;

    for (uint32_t oh = 0; oh < POOL_OUT_H; ++oh) {
        for (uint32_t ow = 0; ow < POOL_OUT_W; ++ow) {
            uint32_t ih0 = oh * POOL_STRIDE;
            uint32_t iw0 = ow * POOL_STRIDE;
            int32_t max_value = pool_input[ih0 * POOL_W + iw0];

            pool_output[oh * POOL_OUT_W + ow] = 0x55555555;
            for (uint32_t kh = 0; kh < POOL_KERNEL; ++kh) {
                for (uint32_t kw = 0; kw < POOL_KERNEL; ++kw) {
                    int32_t value =
                        pool_input[(ih0 + kh) * POOL_W + (iw0 + kw)];

                    if (value > max_value) {
                        max_value = value;
                    }
                }
            }
            pool_expected[oh * POOL_OUT_W + ow] = max_value;
        }
    }

    ret = gpgpu_malloc(dev, &pool_input_addr, sizeof(pool_input));
    if (ret < 0) {
        report_ret("gpgpu_malloc(pool_input)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &pool_output_addr, sizeof(pool_output));
    if (ret < 0) {
        report_ret("gpgpu_malloc(pool_output)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, pool_input_addr, pool_input, sizeof(pool_input));
    if (ret < 0) {
        report_ret("gpgpu_write(pool_input)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, pool_output_addr, pool_output,
                      sizeof(pool_output));
    if (ret < 0) {
        report_ret("gpgpu_write(pool_output)", ret);
        return ret;
    }
    pool_args = (GPGPUMaxPool2DArgs) {
        .input = gpgpu_tensor_make_nchw_i32(pool_input_addr, POOL_N, POOL_C,
                                            POOL_H, POOL_W),
        .output = gpgpu_tensor_make_nchw_i32(pool_output_addr, POOL_N, POOL_C,
                                             POOL_OUT_H, POOL_OUT_W),
        .kernel_h = POOL_KERNEL,
        .kernel_w = POOL_KERNEL,
        .pad_h = 0,
        .pad_w = 0,
        .stride_h = POOL_STRIDE,
        .stride_w = POOL_STRIDE,
    };
    ret = gpgpu_upload_kernel(dev, &pool_kernel_addr,
                              maxpool_i32_kernel_code,
                              sizeof(maxpool_i32_kernel_code) /
                              sizeof(maxpool_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(maxpool_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &pool_args_addr, &pool_args,
                            sizeof(pool_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(maxpool_i32)", ret);
        return ret;
    }
    ret = gpgpu_launch(dev, pool_kernel_addr, pool_args_addr,
                       (GPGPURuntimeDim3){ POOL_N, POOL_C, POOL_OUT_H },
                       (GPGPURuntimeDim3){ POOL_OUT_W, 1, 1 });
    trace_u32("gpgpu maxpool_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu maxpool_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(maxpool_i32)", ret);
        return ret;
    }
    ret = gpgpu_read(dev, pool_output_addr, pool_output,
                     sizeof(pool_output));
    if (ret < 0) {
        report_ret("gpgpu_read(maxpool_i32)", ret);
        return ret;
    }
    for (uint32_t i = 0; i < POOL_N * POOL_C * POOL_OUT_H * POOL_OUT_W; ++i) {
        uart_puts("gpgpu maxpool_i32[");
        uart_puthex32(i);
        uart_puts("]=");
        uart_puthex32((uint32_t)pool_output[i]);
        uart_puts("\n");
        if (pool_output[i] != pool_expected[i]) {
            pool_ok = 0;
        }
    }
    uart_puts("gpgpu maxpool_i32 ");
    uart_puts(pool_ok ? "PASS\n" : "FAIL\n");
    if (!pool_ok) {
        return 1;
    }

    return 0;
}
