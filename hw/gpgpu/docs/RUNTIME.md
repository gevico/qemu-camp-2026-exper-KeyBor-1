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
int gpgpu_upload_args(GPGPURuntimeDevice *dev, uint32_t *args_addr,
                      const void *args, size_t size);

int gpgpu_launch(GPGPURuntimeDevice *dev, uint32_t kernel_addr,
                 uint32_t kernel_args, GPGPURuntimeDim3 grid,
                 GPGPURuntimeDim3 block);
```

CNN/tensor 相关公共结构定义在：

```text
runtime/include/gpgpu_tensor.h
runtime/include/gpgpu_nn.h
```

其中：

```text
GPGPUTensorDesc = data VRAM offset + dtype + layout + shape + stride
GPGPUConv2DArgs / GPGPUReluArgs / GPGPUMaxPool2DArgs / GPGPULinearArgs =
  per-op kernel args layout
GPGPUMatmulPartialArgs / GPGPUMatmulReduceArgs =
  two-stage matrix multiply args layout
GPGPUIm2ColArgs / GPGPUOihwToKoArgs =
  conv lowering args layout
GPGPUNodeDesc / GPGPUNetworkDesc =
  host-side execution plan
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

`gpgpu_upload_kernel()` and `gpgpu_upload_args()` allocate VRAM offsets using
the same bump allocator. Kernel code, args, input tensors, weights, bias, and
output tensors are all represented as 32-bit VRAM offsets.

`gpgpu_upload_args()` writes only the user-provided parameter blob. Launch
metadata such as grid and block dimensions is written by `gpgpu_launch()` to
control registers, and kernel code reads it through the device builtin-register
address space.

For CNN-style kernels, `kernel_args` should point to an op-specific args struct
in VRAM. Tensor data, weights, bias, and output buffers are independent VRAM
allocations; the args struct stores their descriptors and VRAM offsets.

## Limitations

- Allocator is bump-only and does not support free.
- All allocations are 16-byte aligned.
- `gpgpu_upload_args()` uploads the kernel parameter blob. Old-style
  `uint32_t user_args[]` and CNN-style structs such as `GPGPUReluArgs` both
  use this path.
- `gpgpu_launch()` waits by polling `GLOBAL_STATUS.BUSY` and currently has no timeout.
- Runtime register offsets are duplicated locally for now. They should move to a shared UAPI header when the runtime becomes a real external component.
