#ifndef GPGPU_BAREMETAL_TESTS_SMOKE_H
#define GPGPU_BAREMETAL_TESTS_SMOKE_H

#include "test_common.h"

int gpgpu_baremetal_smoke_main(void);
int run_basic_smoke(GPGPURuntimeDevice *dev);
int run_thread_add_smoke(GPGPURuntimeDevice *dev);
int run_stack_smoke(GPGPURuntimeDevice *dev);
int run_relu_smoke(GPGPURuntimeDevice *dev);
int run_linear_smoke(GPGPURuntimeDevice *dev);
int run_matmul_smoke(GPGPURuntimeDevice *dev);
int run_qmatmul_smoke(GPGPURuntimeDevice *dev);
int run_layout_smoke(GPGPURuntimeDevice *dev);
int run_conv_smoke(GPGPURuntimeDevice *dev);
int run_maxpool_smoke(GPGPURuntimeDevice *dev);
int run_lenet_smoke(GPGPURuntimeDevice *dev);

#endif /* GPGPU_BAREMETAL_TESTS_SMOKE_H */
