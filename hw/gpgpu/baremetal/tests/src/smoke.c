#include "smoke.h"

static int run_test(int (*fn)(GPGPURuntimeDevice *dev), GPGPURuntimeDevice *dev)
{
    int ret = fn(dev);

    if (ret < 0) {
        return ret;
    }
    return ret == 0 ? 0 : 1;
}

int gpgpu_baremetal_smoke_main(void)
{
    GPGPURuntimeDevice dev;
    GPGPUPciDevice pci_dev;
    GPGPUPciPlatform platform = {
        .ecam_base = GPGPU_PLATFORM_ECAM_BASE,
        .mmio_base = GPGPU_PLATFORM_MMIO_BASE,
        .mmio_size = GPGPU_PLATFORM_MMIO_SIZE,
    };
    int ret;

    uart_puts("gpgpu baremetal smoke start\n");

    ret = gpgpu_runtime_init_pci(&dev, &platform, &pci_dev);
    if (ret < 0) {
        report_ret("gpgpu_runtime_init_pci", ret);
        return ret;
    }
    gpgpu_print_pci_info(&pci_dev);

    ret = run_test(run_basic_smoke, &dev);
    if (ret) return ret;
    ret = run_test(run_thread_add_smoke, &dev);
    if (ret) return ret;
    ret = run_test(run_stack_smoke, &dev);
    if (ret) return ret;
    ret = run_test(run_relu_smoke, &dev);
    if (ret) return ret;
    ret = run_test(run_linear_smoke, &dev);
    if (ret) return ret;
    ret = run_test(run_matmul_smoke, &dev);
    if (ret) return ret;
    ret = run_test(run_qmatmul_smoke, &dev);
    if (ret) return ret;
    ret = run_test(run_layout_smoke, &dev);
    if (ret) return ret;
    ret = run_test(run_conv_smoke, &dev);
    if (ret) return ret;
    ret = run_test(run_maxpool_smoke, &dev);
    if (ret) return ret;
    return run_test(run_lenet_smoke, &dev);
}
