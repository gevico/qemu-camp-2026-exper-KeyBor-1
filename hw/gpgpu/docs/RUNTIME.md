# Minimal GPGPU Runtime

`runtime/` 是第一版最小 runtime 原型，用于在进入完整 Linux driver / command processor 之前跑通 ABI v0。

## Scope

第一版 runtime 假设调用者已经拿到了：

- BAR0 control register 的 mmap 指针
- BAR2 VRAM 的 mmap 指针
- VRAM size

runtime 不负责 PCI 设备发现和 mmap。当前裸机 PCI 初始化位于
`baremetal/drivers/`；后续进入 Linux 后，可以在 runtime 外面增加
sysfs mmap、UIO、VFIO 或自定义 kernel driver backend。

## API

核心接口定义在 `gpgpu_runtime.h`：

```c
int gpgpu_runtime_init(GPGPURuntimeDevice *dev,
                       volatile void *ctrl_bar,
                       void *vram_bar,
                       uint32_t vram_size);

int gpgpu_malloc(GPGPURuntimeDevice *dev, uint32_t *device_ptr, size_t size);
int gpgpu_write(GPGPURuntimeDevice *dev, uint32_t dst, const void *src,
                size_t size);
int gpgpu_read(GPGPURuntimeDevice *dev, uint32_t src, void *dst, size_t size);

int gpgpu_upload_kernel(GPGPURuntimeDevice *dev, uint32_t *kernel_addr,
                        const uint32_t *code, size_t num_words);
int gpgpu_pack_args(GPGPURuntimeDevice *dev, uint32_t *args_addr,
                    const uint32_t *args, uint32_t num_args);

int gpgpu_launch(GPGPURuntimeDevice *dev, uint32_t kernel_addr,
                 uint32_t kernel_args, GPGPURuntimeDim3 grid,
                 GPGPURuntimeDim3 block);
```

## ABI Mapping

`gpgpu_launch()` implements ABI v0 Launch ABI:

```text
KERNEL_ADDR = kernel_addr
KERNEL_ARGS = kernel_args
GRID_DIM    = grid
BLOCK_DIM   = block
DISPATCH    = 1
```

`gpgpu_upload_kernel()` and `gpgpu_pack_args()` allocate VRAM offsets using the same bump allocator. Kernel code, args, input tensors, and output tensors are all represented as 32-bit VRAM offsets.

`gpgpu_pack_args()` writes only user args. Launch metadata such as grid and
block dimensions is written by `gpgpu_launch()` to control registers, and
kernel code reads it through the device builtin-register address space.

## Limitations

- Allocator is bump-only and does not support free.
- All allocations are 16-byte aligned.
- Kernel user args are packed as 32-bit words:
  `user_args[i]` at `kernel_args + i * 4`.
- `gpgpu_launch()` waits by polling `GLOBAL_STATUS.BUSY` and currently has no timeout.
- Runtime register offsets are duplicated locally for now. They should move to a shared UAPI header when the runtime becomes a real external component.
