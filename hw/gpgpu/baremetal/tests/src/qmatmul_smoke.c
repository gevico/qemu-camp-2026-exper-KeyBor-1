#include "test_common.h"

int run_qmatmul_smoke(GPGPURuntimeDevice *dev)
{
    int32_t qmatmul_a[QMATMUL_M * QMATMUL_K] = {
        256, 512,
    };
    int32_t qmatmul_b[QMATMUL_K * QMATMUL_O] = {
        256, -256,
        128, 512,
    };
    int32_t qmatmul_bias[QMATMUL_O] = { -256, 128 };
    int32_t qmatmul_c[QMATMUL_M * QMATMUL_O];
    int32_t qmatmul_expected[QMATMUL_M * QMATMUL_O];
    int32_t qmatmul_partial[QMATMUL_M * QMATMUL_O * QMATMUL_K];
    uint32_t qmatmul_a_addr;
    uint32_t qmatmul_b_addr;
    uint32_t qmatmul_bias_addr;
    uint32_t qmatmul_c_addr;
    uint32_t qmatmul_partial_addr;
    uint32_t qmatmul_partial_args_addr;
    uint32_t qmatmul_reduce_args_addr;
    uint32_t matmul_partial_kernel_addr;
    uint32_t matmul_reduce_kernel_addr;
    GPGPUMatmulPartialArgs qmatmul_partial_args;
    GPGPUMatmulReduceArgs qmatmul_reduce_args;
    int qmatmul_ok = 1;
    int ret;

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

    for (uint32_t m = 0; m < QMATMUL_M; ++m) {
        for (uint32_t o = 0; o < QMATMUL_O; ++o) {
            int32_t acc = 0;

            qmatmul_c[m * QMATMUL_O + o] = 0x55555555;
            for (uint32_t k = 0; k < QMATMUL_K; ++k) {
                acc += qmatmul_a[m * QMATMUL_K + k] *
                       qmatmul_b[k * QMATMUL_O + o];
            }
            qmatmul_expected[m * QMATMUL_O + o] =
                (acc >> Q8_SHIFT) + qmatmul_bias[o];
        }
    }
    for (uint32_t i = 0; i < QMATMUL_M * QMATMUL_O * QMATMUL_K; ++i) {
        qmatmul_partial[i] = 0x66666666;
    }
    ret = gpgpu_malloc(dev, &qmatmul_a_addr, sizeof(qmatmul_a));
    if (ret < 0) {
        report_ret("gpgpu_malloc(qmatmul_a)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &qmatmul_b_addr, sizeof(qmatmul_b));
    if (ret < 0) {
        report_ret("gpgpu_malloc(qmatmul_b)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &qmatmul_bias_addr, sizeof(qmatmul_bias));
    if (ret < 0) {
        report_ret("gpgpu_malloc(qmatmul_bias)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &qmatmul_c_addr, sizeof(qmatmul_c));
    if (ret < 0) {
        report_ret("gpgpu_malloc(qmatmul_c)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &qmatmul_partial_addr,
                       sizeof(qmatmul_partial));
    if (ret < 0) {
        report_ret("gpgpu_malloc(qmatmul_partial)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, qmatmul_a_addr, qmatmul_a, sizeof(qmatmul_a));
    if (ret < 0) {
        report_ret("gpgpu_write(qmatmul_a)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, qmatmul_b_addr, qmatmul_b, sizeof(qmatmul_b));
    if (ret < 0) {
        report_ret("gpgpu_write(qmatmul_b)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, qmatmul_bias_addr, qmatmul_bias,
                      sizeof(qmatmul_bias));
    if (ret < 0) {
        report_ret("gpgpu_write(qmatmul_bias)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, qmatmul_c_addr, qmatmul_c, sizeof(qmatmul_c));
    if (ret < 0) {
        report_ret("gpgpu_write(qmatmul_c)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, qmatmul_partial_addr, qmatmul_partial,
                      sizeof(qmatmul_partial));
    if (ret < 0) {
        report_ret("gpgpu_write(qmatmul_partial)", ret);
        return ret;
    }
    qmatmul_partial_args = (GPGPUMatmulPartialArgs) {
        .a = gpgpu_tensor_make_mk_i32(qmatmul_a_addr, QMATMUL_M,
                                      QMATMUL_K),
        .b = gpgpu_tensor_make_ko_i32(qmatmul_b_addr, QMATMUL_K,
                                      QMATMUL_O),
        .partial = gpgpu_tensor_make_1d_i32(
            qmatmul_partial_addr, QMATMUL_M * QMATMUL_O * QMATMUL_K),
        .m = QMATMUL_M,
        .k = QMATMUL_K,
        .o = QMATMUL_O,
    };
    qmatmul_reduce_args = (GPGPUMatmulReduceArgs) {
        .partial = gpgpu_tensor_make_1d_i32(
            qmatmul_partial_addr, QMATMUL_M * QMATMUL_O * QMATMUL_K),
        .c = gpgpu_tensor_make_mo_i32(qmatmul_c_addr, QMATMUL_M,
                                      QMATMUL_O),
        .bias = gpgpu_tensor_make_1d_i32(qmatmul_bias_addr, QMATMUL_O),
        .m = QMATMUL_M,
        .k = QMATMUL_K,
        .o = QMATMUL_O,
        .output_shift = Q8_SHIFT,
        .has_bias = 1,
    };
    ret = gpgpu_upload_args(dev, &qmatmul_partial_args_addr,
                            &qmatmul_partial_args,
                            sizeof(qmatmul_partial_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(qmatmul_partial_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &qmatmul_reduce_args_addr,
                            &qmatmul_reduce_args,
                            sizeof(qmatmul_reduce_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(qmatmul_reduce_i32)", ret);
        return ret;
    }
    ret = gpgpu_launch(dev, matmul_partial_kernel_addr,
                       qmatmul_partial_args_addr,
                       (GPGPURuntimeDim3){ QMATMUL_M, QMATMUL_O, 1 },
                       (GPGPURuntimeDim3){ QMATMUL_K, 1, 1 });
    trace_u32("gpgpu qmatmul_partial_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu qmatmul_partial_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(qmatmul_partial_i32)", ret);
        return ret;
    }
    ret = gpgpu_launch(dev, matmul_reduce_kernel_addr,
                       qmatmul_reduce_args_addr,
                       (GPGPURuntimeDim3){ QMATMUL_M, 1, 1 },
                       (GPGPURuntimeDim3){ QMATMUL_O, 1, 1 });
    trace_u32("gpgpu qmatmul_reduce_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu qmatmul_reduce_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(qmatmul_reduce_i32)", ret);
        return ret;
    }
    ret = gpgpu_read(dev, qmatmul_c_addr, qmatmul_c, sizeof(qmatmul_c));
    if (ret < 0) {
        report_ret("gpgpu_read(qmatmul_i32)", ret);
        return ret;
    }
    for (uint32_t i = 0; i < QMATMUL_M * QMATMUL_O; ++i) {
        uart_puts("gpgpu qmatmul_i32[");
        uart_puthex32(i);
        uart_puts("]=");
        uart_puthex32((uint32_t)qmatmul_c[i]);
        uart_puts("\n");
        if (qmatmul_c[i] != qmatmul_expected[i]) {
            qmatmul_ok = 0;
        }
    }
    uart_puts("gpgpu qmatmul_i32 ");
    uart_puts(qmatmul_ok ? "PASS\n" : "FAIL\n");
    if (!qmatmul_ok) {
        return 1;
    }

    return 0;
}
