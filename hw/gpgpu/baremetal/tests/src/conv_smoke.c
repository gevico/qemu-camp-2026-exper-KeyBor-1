#include "test_common.h"

int run_conv_smoke(GPGPURuntimeDevice *dev)
{
    int32_t conv_input[CONV_N * CONV_C * CONV_H * CONV_W] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    int32_t conv_weight[CONV_O * CONV_C * CONV_KH * CONV_KW] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1,
    };
    int32_t conv_col[CONV_M * CONV_K];
    int32_t conv_ko[CONV_K * CONV_O];
    int32_t conv_partial[CONV_M * CONV_O * CONV_K];
    int32_t conv_output[CONV_M * CONV_O];
    int32_t conv_expected[CONV_M * CONV_O];
    uint32_t conv_input_addr;
    uint32_t conv_weight_addr;
    uint32_t conv_col_addr;
    uint32_t conv_ko_addr;
    uint32_t conv_partial_addr;
    uint32_t conv_output_addr;
    uint32_t im2col_kernel_addr;
    uint32_t im2col_args_addr;
    uint32_t oihw_to_ko_kernel_addr;
    uint32_t oihw_to_ko_args_addr;
    uint32_t matmul_partial_kernel_addr;
    uint32_t matmul_reduce_kernel_addr;
    uint32_t conv_matmul_partial_args_addr;
    uint32_t conv_matmul_reduce_args_addr;
    GPGPUIm2ColArgs im2col_args;
    GPGPUOihwToKoArgs oihw_to_ko_args;
    GPGPUMatmulPartialArgs conv_matmul_partial_args;
    GPGPUMatmulReduceArgs conv_matmul_reduce_args;
    int conv_ok = 1;
    int ret;

    for (uint32_t m = 0; m < CONV_M; ++m) {
        uint32_t oh = m / CONV_OUT_W;
        uint32_t ow = m - oh * CONV_OUT_W;
        int32_t acc = 0;

        for (uint32_t kh = 0; kh < CONV_KH; ++kh) {
            for (uint32_t kw = 0; kw < CONV_KW; ++kw) {
                acc += conv_input[(oh + kh) * CONV_W + (ow + kw)] *
                       conv_weight[kh * CONV_KW + kw];
            }
        }
        conv_expected[m] = acc;
    }
    for (uint32_t i = 0; i < CONV_M * CONV_K; ++i) {
        conv_col[i] = 0x11111111;
    }
    for (uint32_t i = 0; i < CONV_K * CONV_O; ++i) {
        conv_ko[i] = 0x22222222;
    }
    for (uint32_t i = 0; i < CONV_M * CONV_O * CONV_K; ++i) {
        conv_partial[i] = 0x33333333;
    }
    for (uint32_t i = 0; i < CONV_M * CONV_O; ++i) {
        conv_output[i] = 0x44444444;
    }

    ret = gpgpu_malloc(dev, &conv_input_addr, sizeof(conv_input));
    if (ret < 0) {
        report_ret("gpgpu_malloc(conv_input)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &conv_weight_addr, sizeof(conv_weight));
    if (ret < 0) {
        report_ret("gpgpu_malloc(conv_weight)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &conv_col_addr, sizeof(conv_col));
    if (ret < 0) {
        report_ret("gpgpu_malloc(conv_col)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &conv_ko_addr, sizeof(conv_ko));
    if (ret < 0) {
        report_ret("gpgpu_malloc(conv_ko)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &conv_partial_addr, sizeof(conv_partial));
    if (ret < 0) {
        report_ret("gpgpu_malloc(conv_partial)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &conv_output_addr, sizeof(conv_output));
    if (ret < 0) {
        report_ret("gpgpu_malloc(conv_output)", ret);
        return ret;
    }

    ret = gpgpu_write(dev, conv_input_addr, conv_input, sizeof(conv_input));
    if (ret < 0) {
        report_ret("gpgpu_write(conv_input)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, conv_weight_addr, conv_weight,
                      sizeof(conv_weight));
    if (ret < 0) {
        report_ret("gpgpu_write(conv_weight)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, conv_col_addr, conv_col, sizeof(conv_col));
    if (ret < 0) {
        report_ret("gpgpu_write(conv_col)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, conv_ko_addr, conv_ko, sizeof(conv_ko));
    if (ret < 0) {
        report_ret("gpgpu_write(conv_ko)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, conv_partial_addr, conv_partial,
                      sizeof(conv_partial));
    if (ret < 0) {
        report_ret("gpgpu_write(conv_partial)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, conv_output_addr, conv_output,
                      sizeof(conv_output));
    if (ret < 0) {
        report_ret("gpgpu_write(conv_output)", ret);
        return ret;
    }

    im2col_args = (GPGPUIm2ColArgs) {
        .input = gpgpu_tensor_make_nchw_i32(conv_input_addr, CONV_N,
                                            CONV_C, CONV_H, CONV_W),
        .output = gpgpu_tensor_make_mk_i32(conv_col_addr, CONV_M, CONV_K),
        .kernel_h = CONV_KH,
        .kernel_w = CONV_KW,
        .pad_h = 0,
        .pad_w = 0,
        .stride_h = 1,
        .stride_w = 1,
        .out_h = CONV_OUT_H,
        .out_w = CONV_OUT_W,
    };
    oihw_to_ko_args = (GPGPUOihwToKoArgs) {
        .input = gpgpu_tensor_make_oihw_i32(conv_weight_addr, CONV_O,
                                            CONV_C, CONV_KH, CONV_KW),
        .output = gpgpu_tensor_make_ko_i32(conv_ko_addr, CONV_K, CONV_O),
        .out_channels = CONV_O,
        .in_channels = CONV_C,
        .kernel_h = CONV_KH,
        .kernel_w = CONV_KW,
    };
    conv_matmul_partial_args = (GPGPUMatmulPartialArgs) {
        .a = gpgpu_tensor_make_mk_i32(conv_col_addr, CONV_M, CONV_K),
        .b = gpgpu_tensor_make_ko_i32(conv_ko_addr, CONV_K, CONV_O),
        .partial = gpgpu_tensor_make_1d_i32(
            conv_partial_addr, CONV_M * CONV_O * CONV_K),
        .m = CONV_M,
        .k = CONV_K,
        .o = CONV_O,
    };
    conv_matmul_reduce_args = (GPGPUMatmulReduceArgs) {
        .partial = gpgpu_tensor_make_1d_i32(
            conv_partial_addr, CONV_M * CONV_O * CONV_K),
        .c = gpgpu_tensor_make_mo_i32(conv_output_addr, CONV_M, CONV_O),
        .m = CONV_M,
        .k = CONV_K,
        .o = CONV_O,
        .output_shift = 0,
        .has_bias = 0,
    };

    ret = gpgpu_upload_kernel(dev, &im2col_kernel_addr,
                              im2col_i32_kernel_code,
                              sizeof(im2col_i32_kernel_code) /
                              sizeof(im2col_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(im2col_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_kernel(dev, &oihw_to_ko_kernel_addr,
                              oihw_to_ko_i32_kernel_code,
                              sizeof(oihw_to_ko_i32_kernel_code) /
                              sizeof(oihw_to_ko_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(oihw_to_ko_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_kernel(dev, &matmul_partial_kernel_addr,
                              matmul_partial_i32_kernel_code,
                              sizeof(matmul_partial_i32_kernel_code) /
                              sizeof(matmul_partial_i32_kernel_code[0]));
    if (ret < 0) return ret;
    ret = gpgpu_upload_kernel(dev, &matmul_reduce_kernel_addr,
                              matmul_reduce_i32_kernel_code,
                              sizeof(matmul_reduce_i32_kernel_code) /
                              sizeof(matmul_reduce_i32_kernel_code[0]));
    if (ret < 0) return ret;
    ret = gpgpu_upload_args(dev, &im2col_args_addr, &im2col_args,
                            sizeof(im2col_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(im2col_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &oihw_to_ko_args_addr, &oihw_to_ko_args,
                            sizeof(oihw_to_ko_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(oihw_to_ko_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &conv_matmul_partial_args_addr,
                            &conv_matmul_partial_args,
                            sizeof(conv_matmul_partial_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(conv_matmul_partial)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &conv_matmul_reduce_args_addr,
                            &conv_matmul_reduce_args,
                            sizeof(conv_matmul_reduce_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(conv_matmul_reduce)", ret);
        return ret;
    }

    ret = gpgpu_launch(dev, im2col_kernel_addr, im2col_args_addr,
                       (GPGPURuntimeDim3){ CONV_N, CONV_OUT_H, CONV_OUT_W },
                       (GPGPURuntimeDim3){ CONV_KW, CONV_KH, CONV_C });
    trace_u32("gpgpu im2col_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu im2col_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(im2col_i32)", ret);
        return ret;
    }
    ret = gpgpu_launch(dev, oihw_to_ko_kernel_addr, oihw_to_ko_args_addr,
                       (GPGPURuntimeDim3){ CONV_O, CONV_C, CONV_KH },
                       (GPGPURuntimeDim3){ CONV_KW, 1, 1 });
    trace_u32("gpgpu oihw_to_ko_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu oihw_to_ko_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(oihw_to_ko_i32)", ret);
        return ret;
    }
    ret = gpgpu_launch(dev, matmul_partial_kernel_addr,
                       conv_matmul_partial_args_addr,
                       (GPGPURuntimeDim3){ CONV_M, CONV_O, 1 },
                       (GPGPURuntimeDim3){ CONV_K, 1, 1 });
    trace_u32("gpgpu conv_matmul_partial global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu conv_matmul_partial error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(conv_matmul_partial)", ret);
        return ret;
    }
    ret = gpgpu_launch(dev, matmul_reduce_kernel_addr,
                       conv_matmul_reduce_args_addr,
                       (GPGPURuntimeDim3){ CONV_M, 1, 1 },
                       (GPGPURuntimeDim3){ CONV_O, 1, 1 });
    trace_u32("gpgpu conv_matmul_reduce global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu conv_matmul_reduce error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(conv_matmul_reduce)", ret);
        return ret;
    }

    ret = gpgpu_read(dev, conv_output_addr, conv_output,
                     sizeof(conv_output));
    if (ret < 0) {
        report_ret("gpgpu_read(conv_lowered_i32)", ret);
        return ret;
    }
    for (uint32_t i = 0; i < CONV_M * CONV_O; ++i) {
        uart_puts("gpgpu conv_lowered_i32[");
        uart_puthex32(i);
        uart_puts("]=");
        uart_puthex32((uint32_t)conv_output[i]);
        uart_puts("\n");
        if (conv_output[i] != conv_expected[i]) {
            conv_ok = 0;
        }
    }
    uart_puts("gpgpu conv_lowered_i32 ");
    uart_puts(conv_ok ? "PASS\n" : "FAIL\n");
    if (!conv_ok) {
        return 1;
    }

    return 0;
}
