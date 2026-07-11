#include "test_common.h"

#define UART0_BASE 0x10000000UL
#define UART_THR   0x00
#define UART_LSR   0x05
#define UART_LSR_THRE 0x20

void uart_putc(char c)
{
    volatile uint8_t *uart = (volatile uint8_t *)UART0_BASE;

    while ((uart[UART_LSR] & UART_LSR_THRE) == 0) {
    }
    uart[UART_THR] = (uint8_t)c;
}

void uart_puts(const char *s)
{
    while (*s) {
        if (*s == '\n') {
            uart_putc('\r');
        }
        uart_putc(*s++);
    }
}

void uart_puthex32(uint32_t value)
{
    static const char digits[] = "0123456789abcdef";

    uart_puts("0x");
    for (int shift = 28; shift >= 0; shift -= 4) {
        uart_putc(digits[(value >> shift) & 0xf]);
    }
}

void *memcpy(void *dst, const void *src, size_t n)
{
    uint8_t *d = dst;
    const uint8_t *s = src;

    while (n--) {
        *d++ = *s++;
    }
    return dst;
}

void *memset(void *dst, int c, size_t n)
{
    uint8_t *d = dst;

    while (n--) {
        *d++ = (uint8_t)c;
    }
    return dst;
}

void report_ret(const char *op, int ret)
{
    uart_puts(op);
    uart_puts(" failed ret=");
    uart_puthex32((uint32_t)ret);
    uart_puts("\n");
}

uint32_t ctrl_read32(GPGPURuntimeDevice *dev, uint32_t reg)
{
    volatile uint32_t *ptr = (volatile uint32_t *)(dev->ctrl + reg);

    return *ptr;
}

void trace_u32(const char *name, uint32_t value)
{
    uart_puts(name);
    uart_puts("=");
    uart_puthex32(value);
    uart_puts("\n");
}

int upload_i32_array(GPGPURuntimeDevice *dev, uint32_t *addr,
                            const int32_t *values, uint32_t count)
{
    int ret = gpgpu_malloc(dev, addr, count * sizeof(*values));

    if (ret < 0) {
        return ret;
    }
    return gpgpu_write(dev, *addr, values, count * sizeof(*values));
}

int alloc_i32_array(GPGPURuntimeDevice *dev, uint32_t *addr,
                           uint32_t count)
{
    return gpgpu_malloc(dev, addr, count * sizeof(int32_t));
}

int upload_linear_weight_ko(GPGPURuntimeDevice *dev, uint32_t *addr,
                                   const int32_t *weight_oi,
                                   uint32_t out_features,
                                   uint32_t in_features)
{
    int ret = gpgpu_malloc(dev, addr,
                           out_features * in_features * sizeof(int32_t));

    if (ret < 0) {
        return ret;
    }
    for (uint32_t k = 0; k < in_features; ++k) {
        for (uint32_t o = 0; o < out_features; ++o) {
            int32_t value = weight_oi[o * in_features + k];
            uint32_t offset = (k * out_features + o) * sizeof(value);

            ret = gpgpu_write(dev, *addr + offset, &value, sizeof(value));
            if (ret < 0) {
                return ret;
            }
        }
    }
    return 0;
}

int upload_args_checked(GPGPURuntimeDevice *dev, uint32_t *addr,
                               const void *args, size_t size,
                               const char *name)
{
    int ret = gpgpu_upload_args(dev, addr, args, size);

    if (ret < 0) {
        report_ret(name, ret);
    }
    return ret;
}

void add_node(GPGPUNodeDesc *nodes, uint32_t *num_nodes,
                     uint32_t kernel_addr, uint32_t args_addr,
                     GPGPURuntimeDim3 grid, GPGPURuntimeDim3 block)
{
    nodes[*num_nodes] = (GPGPUNodeDesc) {
        .kernel_addr = kernel_addr,
        .args_addr = args_addr,
        .grid = grid,
        .block = block,
    };
    *num_nodes += 1;
}

int run_nodes(GPGPURuntimeDevice *dev, GPGPUNodeDesc *nodes,
                     uint32_t num_nodes, const char *name)
{
    for (uint32_t i = 0; i < num_nodes; ++i) {
        int ret = gpgpu_launch(dev, nodes[i].kernel_addr, nodes[i].args_addr,
                               nodes[i].grid, nodes[i].block);

        if (ret < 0) {
            uart_puts(name);
            uart_puts(" node failed idx=");
            uart_puthex32(i);
            uart_puts("\n");
            report_ret("gpgpu_launch(network_node)", ret);
            return ret;
        }
    }
    return 0;
}


void gpgpu_print_pci_info(const GPGPUPciDevice *pci_dev)
{
    uart_puts("gpgpu pci found bar0=");
    uart_puthex32((uint32_t)pci_dev->bar0_base);
    uart_puts(" size=");
    uart_puthex32(pci_dev->bar0_size);
    uart_puts(" bar2=");
    uart_puthex32((uint32_t)pci_dev->bar2_base);
    uart_puts(" size=");
    uart_puthex32(pci_dev->bar2_size);
    uart_puts("\n");
    trace_u32("gpgpu pci bdf",
              ((uint32_t)pci_dev->bus << 16) |
              ((uint32_t)pci_dev->device << 8) |
              pci_dev->function);
    trace_u32("gpgpu pci vendor_device",
              ((uint32_t)pci_dev->vendor_id << 16) | pci_dev->device_id);
}
