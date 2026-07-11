#include "test_common.h"

int run_stack_smoke(GPGPURuntimeDevice *dev)
{
    int32_t stack_input[STACK_TEST_LEN] = { 7, -3, 11, 0 };
    int32_t stack_output[STACK_TEST_LEN];
    int32_t stack_expected[STACK_TEST_LEN];
    uint32_t stack_input_addr;
    uint32_t stack_output_addr;
    uint32_t stack_kernel_addr;
    uint32_t stack_args_addr;
    GPGPUReluArgs stack_args;
    int stack_ok = 1;
    int ret;

    for (uint32_t i = 0; i < STACK_TEST_LEN; ++i) {
        stack_output[i] = 0x22222222;
        stack_expected[i] = stack_input[i] + (int32_t)i + 0x100;
    }
    ret = gpgpu_malloc(dev, &stack_input_addr, sizeof(stack_input));
    if (ret < 0) {
        report_ret("gpgpu_malloc(stack_input)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &stack_output_addr, sizeof(stack_output));
    if (ret < 0) {
        report_ret("gpgpu_malloc(stack_output)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, stack_input_addr, stack_input,
                      sizeof(stack_input));
    if (ret < 0) {
        report_ret("gpgpu_write(stack_input)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, stack_output_addr, stack_output,
                      sizeof(stack_output));
    if (ret < 0) {
        report_ret("gpgpu_write(stack_output)", ret);
        return ret;
    }
    stack_args = (GPGPUReluArgs) {
        .input = gpgpu_tensor_make_1d_i32(stack_input_addr, STACK_TEST_LEN),
        .output = gpgpu_tensor_make_1d_i32(stack_output_addr, STACK_TEST_LEN),
    };
    ret = gpgpu_upload_kernel(dev, &stack_kernel_addr,
                              stack_smoke_kernel_code,
                              sizeof(stack_smoke_kernel_code) /
                              sizeof(stack_smoke_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(stack_smoke)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &stack_args_addr, &stack_args,
                            sizeof(stack_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(stack_smoke)", ret);
        return ret;
    }
    ret = gpgpu_launch(dev, stack_kernel_addr, stack_args_addr,
                       (GPGPURuntimeDim3){ 1, 1, 1 },
                       (GPGPURuntimeDim3){ STACK_TEST_LEN, 1, 1 });
    trace_u32("gpgpu stack_smoke global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu stack_smoke error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(stack_smoke)", ret);
        return ret;
    }
    ret = gpgpu_read(dev, stack_output_addr, stack_output,
                     sizeof(stack_output));
    if (ret < 0) {
        report_ret("gpgpu_read(stack_smoke)", ret);
        return ret;
    }
    for (uint32_t i = 0; i < STACK_TEST_LEN; ++i) {
        uart_puts("gpgpu stack_smoke[");
        uart_puthex32(i);
        uart_puts("]=");
        uart_puthex32((uint32_t)stack_output[i]);
        uart_puts("\n");
        if (stack_output[i] != stack_expected[i]) {
            stack_ok = 0;
        }
    }
    uart_puts("gpgpu stack_smoke ");
    uart_puts(stack_ok ? "PASS\n" : "FAIL\n");
    if (!stack_ok) {
        return 1;
    }

    return 0;
}
