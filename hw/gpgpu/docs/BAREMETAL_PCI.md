# Bare-Metal PCI Setup

本文档解释 `gpgpu_pci.c` 做的事情，以及裸机程序如何通过 PCI BAR 访问 GPGPU VRAM。

## 1. 两类地址

裸机 PCI 初始化需要区分两类地址：

- **PCI config / ECAM 地址**：CPU 用它读写 PCI config space，发现设备、读写 BAR 寄存器。
- **PCI MMIO window 地址**：平台硬件/host bridge 规定的一段 CPU 物理地址范围，访问这段范围会被转发成 PCIe Memory Read/Write。

平台决定的是：

```text
ECAM base
PCI MMIO window base
PCI MMIO window size
```

软件决定的是：

```text
把某个设备的 BAR 放到 PCI MMIO window 里的哪个地址
```

## 2. BAR 映射流程

`gpgpu_runtime_init_pci()` 做了一个最小流程：

```text
1. 扫描 ECAM config space
2. 查找 vendor/device = 0x1234/0x1337
3. 对 BAR0/BAR2 写全 1，探测 BAR size
4. 从 platform.mmio_base..mmio_base+mmio_size 中分配地址
5. 把分配到的地址写回 BAR0/BAR2
6. 设置 PCI command register 的 Memory Space Enable
7. 调用 gpgpu_runtime_init()
```

完成后：

```text
dev.ctrl = BAR0 base
dev.vram = BAR2 base
```

runtime 访问：

```c
dev.vram + offset
```

就等价于 CPU 对：

```text
BAR2_BASE + offset
```

发起 MMIO 访问。

## 3. 最小使用方式

平台启动代码需要先提供地址：

```c
GPGPUPciPlatform platform = {
    .ecam_base = PLATFORM_PCI_ECAM_BASE,
    .mmio_base = PLATFORM_PCI_MMIO_BASE,
    .mmio_size = PLATFORM_PCI_MMIO_SIZE,
};
```

然后：

```c
GPGPURuntimeDevice dev;
GPGPUPciDevice pci_dev;

ret = gpgpu_runtime_init_pci(&dev, &platform, &pci_dev);
if (ret < 0) {
    /* device not found or BAR allocation failed */
}
```

之后可以使用普通 runtime API：

```c
uint32_t buf;
gpgpu_malloc(&dev, &buf, 4096);
gpgpu_write(&dev, buf, host_data, 4096);
```

## 4. 当前限制

- 只支持 ECAM 风格 PCI config space。
- 只配置 BAR0 和 BAR2。
- BAR 地址从传入的 PCI MMIO window 起始处顺序分配。
- 没有处理 PCI bridge bus range、复杂资源树和冲突检测。
- 没有配置 cache attribute；真实硬件上 BAR MMIO 应映射为 device/uncached 或 write-combining。
- 这是裸机教学后端，不替代 Linux PCI 子系统。
