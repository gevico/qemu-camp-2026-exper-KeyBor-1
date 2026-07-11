#include "test_common.h"

int run_relu_smoke(GPGPURuntimeDevice *dev)
{
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
    int ret;

    for (uint32_t i = 0; i < RELU_TEST_LEN; ++i) {
        relu_output[i] = 0x77777777;
        relu_expected[i] = relu_input[i] < 0 ? 0 : relu_input[i];
    }

    ret = gpgpu_malloc(dev, &relu_input_addr, sizeof(relu_input));
    if (ret < 0) {
        report_ret("gpgpu_malloc(relu_input)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &relu_output_addr, sizeof(relu_output));
    if (ret < 0) {
        report_ret("gpgpu_malloc(relu_output)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, relu_input_addr, relu_input, sizeof(relu_input));
    if (ret < 0) {
        report_ret("gpgpu_write(relu_input)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, relu_output_addr, relu_output, sizeof(relu_output));
    if (ret < 0) {
        report_ret("gpgpu_write(relu_output)", ret);
        return ret;
    }

    relu_args = (GPGPUReluArgs) {
        .input = gpgpu_tensor_make_1d_i32(relu_input_addr, RELU_TEST_LEN),
        .output = gpgpu_tensor_make_1d_i32(relu_output_addr, RELU_TEST_LEN),
    };

    ret = gpgpu_upload_kernel(dev, &relu_kernel_addr,
                              relu_i32_kernel_code,
                              sizeof(relu_i32_kernel_code) /
                              sizeof(relu_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(relu_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &relu_args_addr, &relu_args,
                            sizeof(relu_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(relu_i32)", ret);
        return ret;
    }

    ret = gpgpu_launch(dev, relu_kernel_addr, relu_args_addr,
                       (GPGPURuntimeDim3){ 2, 1, 1 },
                       (GPGPURuntimeDim3){ 8, 1, 1 });
    trace_u32("gpgpu relu_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu relu_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(relu_i32)", ret);
        return ret;
    }

    ret = gpgpu_read(dev, relu_output_addr, relu_output,
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

    return 0;
}
