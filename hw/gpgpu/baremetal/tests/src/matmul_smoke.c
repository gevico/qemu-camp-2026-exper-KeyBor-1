#include "test_common.h"

int run_matmul_smoke(GPGPURuntimeDevice *dev)
{
    int32_t matmul_a[MATMUL_M * MATMUL_K] = {
        1, 2, 3,
        4, -1, 2,
    };
    int32_t matmul_b[MATMUL_K * MATMUL_O] = {
        1, 2,
        0, -1,
        3, 1,
    };
    int32_t matmul_c[MATMUL_M * MATMUL_O];
    int32_t matmul_expected[MATMUL_M * MATMUL_O];
    int32_t matmul_partial[MATMUL_M * MATMUL_O * MATMUL_K];
    uint32_t matmul_a_addr;
    uint32_t matmul_b_addr;
    uint32_t matmul_c_addr;
    uint32_t matmul_partial_addr;
    uint32_t matmul_partial_kernel_addr;
    uint32_t matmul_partial_args_addr;
    uint32_t matmul_reduce_kernel_addr;
    uint32_t matmul_reduce_args_addr;
    GPGPUMatmulPartialArgs matmul_partial_args;
    GPGPUMatmulReduceArgs matmul_reduce_args;
    int matmul_ok = 1;
    int ret;

    for (uint32_t m = 0; m < MATMUL_M; ++m) {
        for (uint32_t o = 0; o < MATMUL_O; ++o) {
            int32_t acc = 0;

            matmul_c[m * MATMUL_O + o] = 0x11111111;
            for (uint32_t k = 0; k < MATMUL_K; ++k) {
                acc += matmul_a[m * MATMUL_K + k] *
                       matmul_b[k * MATMUL_O + o];
            }
            matmul_expected[m * MATMUL_O + o] = acc;
        }
    }
    for (uint32_t i = 0; i < MATMUL_M * MATMUL_O * MATMUL_K; ++i) {
        matmul_partial[i] = 0x44444444;
    }

    ret = gpgpu_malloc(dev, &matmul_a_addr, sizeof(matmul_a));
    if (ret < 0) {
        report_ret("gpgpu_malloc(matmul_a)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &matmul_b_addr, sizeof(matmul_b));
    if (ret < 0) {
        report_ret("gpgpu_malloc(matmul_b)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &matmul_c_addr, sizeof(matmul_c));
    if (ret < 0) {
        report_ret("gpgpu_malloc(matmul_c)", ret);
        return ret;
    }
    ret = gpgpu_malloc(dev, &matmul_partial_addr, sizeof(matmul_partial));
    if (ret < 0) {
        report_ret("gpgpu_malloc(matmul_partial)", ret);
        return ret;
    }

    ret = gpgpu_write(dev, matmul_a_addr, matmul_a, sizeof(matmul_a));
    if (ret < 0) {
        report_ret("gpgpu_write(matmul_a)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, matmul_b_addr, matmul_b, sizeof(matmul_b));
    if (ret < 0) {
        report_ret("gpgpu_write(matmul_b)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, matmul_c_addr, matmul_c, sizeof(matmul_c));
    if (ret < 0) {
        report_ret("gpgpu_write(matmul_c)", ret);
        return ret;
    }
    ret = gpgpu_write(dev, matmul_partial_addr, matmul_partial,
                      sizeof(matmul_partial));
    if (ret < 0) {
        report_ret("gpgpu_write(matmul_partial)", ret);
        return ret;
    }

    matmul_partial_args = (GPGPUMatmulPartialArgs) {
        .a = gpgpu_tensor_make_mk_i32(matmul_a_addr, MATMUL_M, MATMUL_K),
        .b = gpgpu_tensor_make_ko_i32(matmul_b_addr, MATMUL_K, MATMUL_O),
        .partial = gpgpu_tensor_make_1d_i32(
            matmul_partial_addr, MATMUL_M * MATMUL_O * MATMUL_K),
        .m = MATMUL_M,
        .k = MATMUL_K,
        .o = MATMUL_O,
    };
    matmul_reduce_args = (GPGPUMatmulReduceArgs) {
        .partial = gpgpu_tensor_make_1d_i32(
            matmul_partial_addr, MATMUL_M * MATMUL_O * MATMUL_K),
        .c = gpgpu_tensor_make_mo_i32(matmul_c_addr, MATMUL_M, MATMUL_O),
        .m = MATMUL_M,
        .k = MATMUL_K,
        .o = MATMUL_O,
    };

    ret = gpgpu_upload_kernel(dev, &matmul_partial_kernel_addr,
                              matmul_partial_i32_kernel_code,
                              sizeof(matmul_partial_i32_kernel_code) /
                              sizeof(matmul_partial_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(matmul_partial_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_kernel(dev, &matmul_reduce_kernel_addr,
                              matmul_reduce_i32_kernel_code,
                              sizeof(matmul_reduce_i32_kernel_code) /
                              sizeof(matmul_reduce_i32_kernel_code[0]));
    if (ret < 0) {
        report_ret("gpgpu_upload_kernel(matmul_reduce_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &matmul_partial_args_addr,
                            &matmul_partial_args,
                            sizeof(matmul_partial_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(matmul_partial_i32)", ret);
        return ret;
    }
    ret = gpgpu_upload_args(dev, &matmul_reduce_args_addr,
                            &matmul_reduce_args,
                            sizeof(matmul_reduce_args));
    if (ret < 0) {
        report_ret("gpgpu_upload_args(matmul_reduce_i32)", ret);
        return ret;
    }

    ret = gpgpu_launch(dev, matmul_partial_kernel_addr,
                       matmul_partial_args_addr,
                       (GPGPURuntimeDim3){ MATMUL_M, MATMUL_O, 1 },
                       (GPGPURuntimeDim3){ MATMUL_K, 1, 1 });
    trace_u32("gpgpu matmul_partial_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu matmul_partial_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(matmul_partial_i32)", ret);
        return ret;
    }

    ret = gpgpu_launch(dev, matmul_reduce_kernel_addr,
                       matmul_reduce_args_addr,
                       (GPGPURuntimeDim3){ MATMUL_M, 1, 1 },
                       (GPGPURuntimeDim3){ MATMUL_O, 1, 1 });
    trace_u32("gpgpu matmul_reduce_i32 global_status",
              ctrl_read32(dev, GPGPU_REG_GLOBAL_STATUS));
    trace_u32("gpgpu matmul_reduce_i32 error_status",
              ctrl_read32(dev, GPGPU_REG_ERROR_STATUS));
    if (ret < 0) {
        report_ret("gpgpu_launch(matmul_reduce_i32)", ret);
        return ret;
    }

    ret = gpgpu_read(dev, matmul_c_addr, matmul_c, sizeof(matmul_c));
    if (ret < 0) {
        report_ret("gpgpu_read(matmul_i32)", ret);
        return ret;
    }

    for (uint32_t i = 0; i < MATMUL_M * MATMUL_O; ++i) {
        uart_puts("gpgpu matmul_i32[");
        uart_puthex32(i);
        uart_puts("]=");
        uart_puthex32((uint32_t)matmul_c[i]);
        uart_puts("\n");
        if (matmul_c[i] != matmul_expected[i]) {
            matmul_ok = 0;
        }
    }

    uart_puts("gpgpu matmul_i32 ");
    uart_puts(matmul_ok ? "PASS\n" : "FAIL\n");
    if (!matmul_ok) {
        return 1;
    }

    return 0;
}
