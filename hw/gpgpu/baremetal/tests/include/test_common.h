#ifndef GPGPU_BAREMETAL_TESTS_COMMON_H
#define GPGPU_BAREMETAL_TESTS_COMMON_H

#include "qemu_virt_platform.h"
#include "gpgpu_pci.h"
#include "gpgpu_nn.h"

#include <stddef.h>
#include <stdint.h>

#define GPGPU_REG_GLOBAL_STATUS 0x0104
#define GPGPU_REG_ERROR_STATUS  0x0108

#define STACK_TEST_LEN 4
#define RELU_TEST_LEN 16
#define LINEAR_IN_FEATURES 4
#define LINEAR_OUT_FEATURES 3
#define MATMUL_M 2
#define MATMUL_K 3
#define MATMUL_O 2
#define QMATMUL_M 1
#define QMATMUL_K 2
#define QMATMUL_O 2
#define Q8_SHIFT 8
#define LAYOUT_N 1
#define LAYOUT_C 2
#define LAYOUT_H 2
#define LAYOUT_W 2
#define LENET_N 1
#define LENET_IN_C 1
#define LENET_IN_H 28
#define LENET_IN_W 28
#define LENET_CONV1_O 6
#define LENET_CONV1_KH 5
#define LENET_CONV1_KW 5
#define LENET_CONV1_OUT_H 28
#define LENET_CONV1_OUT_W 28
#define LENET_CONV1_M (LENET_N * LENET_CONV1_OUT_H * LENET_CONV1_OUT_W)
#define LENET_CONV1_K (LENET_IN_C * LENET_CONV1_KH * LENET_CONV1_KW)
#define LENET_POOL1_H 14
#define LENET_POOL1_W 14
#define LENET_CONV2_O 16
#define LENET_CONV2_KH 5
#define LENET_CONV2_KW 5
#define LENET_CONV2_OUT_H 10
#define LENET_CONV2_OUT_W 10
#define LENET_CONV2_M (LENET_N * LENET_CONV2_OUT_H * LENET_CONV2_OUT_W)
#define LENET_CONV2_K (LENET_CONV1_O * LENET_CONV2_KH * LENET_CONV2_KW)
#define LENET_POOL2_H 5
#define LENET_POOL2_W 5
#define LENET_FC1_IN (LENET_CONV2_O * LENET_POOL2_H * LENET_POOL2_W)
#define LENET_FC1_OUT 120
#define LENET_FC2_OUT 84
#define LENET_FC3_OUT 10
#define LENET_NODE_MAX 22
#define CONV_N 1
#define CONV_C 1
#define CONV_H 4
#define CONV_W 4
#define CONV_O 1
#define CONV_KH 3
#define CONV_KW 3
#define CONV_OUT_H 2
#define CONV_OUT_W 2
#define CONV_M (CONV_N * CONV_OUT_H * CONV_OUT_W)
#define CONV_K (CONV_C * CONV_KH * CONV_KW)
#define POOL_N 1
#define POOL_C 1
#define POOL_H 4
#define POOL_W 4
#define POOL_OUT_H 2
#define POOL_OUT_W 2
#define POOL_KERNEL 2
#define POOL_STRIDE 2

#define RV_RD(rd)          ((uint32_t)(rd) << 7)
#define RV_RS1(rs1)        ((uint32_t)(rs1) << 15)
#define RV_RS2(rs2)        ((uint32_t)(rs2) << 20)
#define RV_LUI(rd, imm20)  (((uint32_t)(imm20) << 12) | RV_RD(rd) | 0x37)
#define RV_ADDI(rd, rs1, imm) \
    ((((uint32_t)(imm) & 0xfff) << 20) | RV_RS1(rs1) | RV_RD(rd) | 0x13)
#define RV_SLLI(rd, rs1, shamt) \
    (((uint32_t)(shamt) << 20) | RV_RS1(rs1) | (1u << 12) | RV_RD(rd) | 0x13)
#define RV_ADD(rd, rs1, rs2) \
    (RV_RS2(rs2) | RV_RS1(rs1) | RV_RD(rd) | 0x33)
#define RV_LW(rd, imm, rs1) \
    ((((uint32_t)(imm) & 0xfff) << 20) | RV_RS1(rs1) | (2u << 12) | RV_RD(rd) | 0x03)
#define RV_SW(rs2, imm, rs1) \
    (((((uint32_t)(imm) & 0xfe0) << 20) | RV_RS2(rs2) | RV_RS1(rs1) | \
      (2u << 12) | (((uint32_t)(imm) & 0x1f) << 7) | 0x23))
#define RV_EBREAK          0x00100073u

#include "build/thread_add_kernel.inc"
#include "build/stack_smoke_kernel.inc"
#include "build/relu_i32_kernel.inc"
#include "build/linear_i32_kernel.inc"
#include "build/linear_partial_i32_kernel.inc"
#include "build/linear_reduce_i32_kernel.inc"
#include "build/matmul_partial_i32_kernel.inc"
#include "build/matmul_reduce_i32_kernel.inc"
#include "build/im2col_i32_kernel.inc"
#include "build/oihw_to_ko_i32_kernel.inc"
#include "build/mo_to_nchw_i32_kernel.inc"
#include "build/maxpool_i32_kernel.inc"

void uart_putc(char c);
void uart_puts(const char *s);
void uart_puthex32(uint32_t value);
void report_ret(const char *op, int ret);
uint32_t ctrl_read32(GPGPURuntimeDevice *dev, uint32_t reg);
void trace_u32(const char *name, uint32_t value);
void gpgpu_print_pci_info(const GPGPUPciDevice *pci_dev);

int upload_i32_array(GPGPURuntimeDevice *dev, uint32_t *addr,
                     const int32_t *values, uint32_t count);
int alloc_i32_array(GPGPURuntimeDevice *dev, uint32_t *addr, uint32_t count);
int upload_linear_weight_ko(GPGPURuntimeDevice *dev, uint32_t *addr,
                            const int32_t *weight_oi,
                            uint32_t out_features,
                            uint32_t in_features);
int upload_args_checked(GPGPURuntimeDevice *dev, uint32_t *addr,
                        const void *args, size_t size,
                        const char *name);
void add_node(GPGPUNodeDesc *nodes, uint32_t *num_nodes,
              uint32_t kernel_addr, uint32_t args_addr,
              GPGPURuntimeDim3 grid, GPGPURuntimeDim3 block);
int run_nodes(GPGPURuntimeDevice *dev, GPGPUNodeDesc *nodes,
              uint32_t num_nodes, const char *name);

#endif /* GPGPU_BAREMETAL_TESTS_COMMON_H */
