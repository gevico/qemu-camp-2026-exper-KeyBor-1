#include "test_common.h"

int run_linear_smoke(GPGPURuntimeDevice *dev)
{
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
    int32_t linear_parallel_output[LINEAR_OUT_FEATURES];
    int32_t linear_partial[LINEAR_OUT_FEATURES * LINEAR_IN_FEATURES];
    uint32_t linear_parallel_output_addr;
    uint32_t linear_partial_addr;
    uint32_t linear_partial_kernel_addr;
    uint32_t linear_partial_args_addr;
    uint32_t linear_reduce_kernel_addr;
    uint32_t linear_reduce_args_addr;
    GPGPULinearPartialArgs linear_partial_args;
    GPGPULinearReduceArgs linear_reduce_args;
    int linear_parallel_ok = 1;
    int ret;

    for (uint32_t o = 0; o < LINEAR_OUT_FEATURES; ++o) {
        int32_t acc = linear_bias[o];

        linear_output[o] = 0x55555555;
        for (uint32_t i = 0; i < LINEAR_IN_FEATURES; ++i) {
            acc += linear_input[i] *
                   linear_weight[o * LINEAR_IN_FEATURES + i];
        }
        linear_expected[o] = acc;
    }

    ret = gpgpu_malloc(dev, &linear_input_addr, sizeof(linear_input));
    if (ret < 0) {
        report_ret("gpgpu_malloc(linear_input)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &linear_weight_addr, sizeof(linear_weight));
    if (ret < 0) {
        report_ret("gpgpu_malloc(linear_weight)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &linear_bias_addr, sizeof(linear_bias));
    if (ret < 0) {
        report_ret("gpgpu_malloc(linear_bias)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &linear_output_addr, sizeof(linear_output));
    if (ret < 0) {
        report_ret("gpgpu_malloc(linear_output)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, linear_input_addr, linear_input,
                      sizeof(linear_input));
    if (ret < 0) {
        report_ret("gpgpu_write(linear_input)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, linear_weight_addr, linear_weight,
                      sizeof(linear_weight));
    if (ret < 0) {
        report_ret("gpgpu_write(linear_weight)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, linear_bias_addr, linear_bias,
                      sizeof(linear_bias));
    if (ret < 0) {
        report_ret("gpgpu_write(linear_bias)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, linear_output_addr, linear_output,
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

    ret = gpgpu_upload_kernel(dev, &linear_kernel_addr,
                              linear_i32_kernel_code,
                              sizeof(linear_i32_kernel_code) /
                              sizeof(linear_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(linear_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &linear_args_addr, &linear_args,
                            sizeof(linear_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(linear_i32)", ret);
        return ret;
    }

    ret = gpgpu_launch(dev, linear_kernel_addr, linear_args_addr,
                       (GPGPURuntimeDim3){ 1, 1, 1 },
                       (GPGPURuntimeDim3){ 3, 1, 1 });
    trace_u32("gpgpu linear_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu linear_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(linear_i32)", ret);
        return ret;
    }

    ret = gpgpu_read(dev, linear_output_addr, linear_output,
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
    if (!linear_ok) {
        return 1;
    }

    for (uint32_t i = 0; i < LINEAR_OUT_FEATURES; ++i) {
        linear_parallel_output[i] = 0x33333333;
    }
    for (uint32_t i = 0; i < LINEAR_OUT_FEATURES * LINEAR_IN_FEATURES; ++i) {
        linear_partial[i] = 0x22222222;
    }

    ret = gpgpu_malloc(dev, &linear_partial_addr, sizeof(linear_partial));
    if (ret < 0) {
        report_ret("gpgpu_malloc(linear_partial)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &linear_parallel_output_addr,
                       sizeof(linear_parallel_output));
    if (ret < 0) {
        report_ret("gpgpu_malloc(linear_parallel_output)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, linear_partial_addr, linear_partial,
                      sizeof(linear_partial));
    if (ret < 0) {
        report_ret("gpgpu_write(linear_partial)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, linear_parallel_output_addr,
                      linear_parallel_output,
                      sizeof(linear_parallel_output));
    if (ret < 0) {
        report_ret("gpgpu_write(linear_parallel_output)", ret);
        return ret;
    }

    linear_partial_args = (GPGPULinearPartialArgs) {
        .input = gpgpu_tensor_make_1d_i32(linear_input_addr,
                                          LINEAR_IN_FEATURES),
        .weight = gpgpu_tensor_make_oi_i32(linear_weight_addr,
                                           LINEAR_OUT_FEATURES,
                                           LINEAR_IN_FEATURES),
        .partial = gpgpu_tensor_make_oi_i32(linear_partial_addr,
                                            LINEAR_OUT_FEATURES,
                                            LINEAR_IN_FEATURES),
        .in_features = LINEAR_IN_FEATURES,
        .out_features = LINEAR_OUT_FEATURES,
    };
    linear_reduce_args = (GPGPULinearReduceArgs) {
        .partial = gpgpu_tensor_make_oi_i32(linear_partial_addr,
                                            LINEAR_OUT_FEATURES,
                                            LINEAR_IN_FEATURES),
        .bias = gpgpu_tensor_make_1d_i32(linear_bias_addr,
                                         LINEAR_OUT_FEATURES),
        .output = gpgpu_tensor_make_1d_i32(linear_parallel_output_addr,
                                           LINEAR_OUT_FEATURES),
        .in_features = LINEAR_IN_FEATURES,
        .out_features = LINEAR_OUT_FEATURES,
    };

    ret = gpgpu_upload_kernel(dev, &linear_partial_kernel_addr,
                              linear_partial_i32_kernel_code,
                              sizeof(linear_partial_i32_kernel_code) /
                              sizeof(linear_partial_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(linear_partial_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_kernel(dev, &linear_reduce_kernel_addr,
                              linear_reduce_i32_kernel_code,
                              sizeof(linear_reduce_i32_kernel_code) /
                              sizeof(linear_reduce_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(linear_reduce_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &linear_partial_args_addr,
                            &linear_partial_args,
                            sizeof(linear_partial_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(linear_partial_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &linear_reduce_args_addr,
                            &linear_reduce_args,
                            sizeof(linear_reduce_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(linear_reduce_i32)", ret);
        return ret;
    }

    ret = gpgpu_launch(dev, linear_partial_kernel_addr,
                       linear_partial_args_addr,
                       (GPGPURuntimeDim3){ LINEAR_OUT_FEATURES, 1, 1 },
                       (GPGPURuntimeDim3){ LINEAR_IN_FEATURES, 1, 1 });
    trace_u32("gpgpu linear_partial_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu linear_partial_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(linear_partial_i32)", ret);
        return ret;
    }

    ret = gpgpu_launch(dev, linear_reduce_kernel_addr,
                       linear_reduce_args_addr,
                       (GPGPURuntimeDim3){ 1, 1, 1 },
                       (GPGPURuntimeDim3){ LINEAR_OUT_FEATURES, 1, 1 });
    trace_u32("gpgpu linear_reduce_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu linear_reduce_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(linear_reduce_i32)", ret);
        return ret;
    }

    ret = gpgpu_read(dev, linear_parallel_output_addr,
                     linear_parallel_output,
                     sizeof(linear_parallel_output));
    if (ret < 0) {
        report_ret("gpgpu_read(linear_parallel_i32)", ret);
        return ret;
    }

    for (uint32_t i = 0; i < LINEAR_OUT_FEATURES; ++i) {
        uart_puts("gpgpu linear_parallel_i32[");
        uart_puthex32(i);
        uart_puts("]=");
        uart_puthex32((uint32_t)linear_parallel_output[i]);
        uart_puts("\n");
        if (linear_parallel_output[i] != linear_expected[i]) {
            linear_parallel_ok = 0;
        }
    }

    uart_puts("gpgpu linear_parallel_i32 ");
    uart_puts(linear_parallel_ok ? "PASS\n" : "FAIL\n");
    if (!linear_parallel_ok) {
        return 1;
    }

    return 0;
}
