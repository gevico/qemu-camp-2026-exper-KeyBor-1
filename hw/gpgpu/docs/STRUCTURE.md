# GPGPU Directory Structure

本文档记录 `hw/gpgpu/` 当前目录结构，以及进入 CNN kernel 阶段前各部分的职责边界。

## 目录树

```text
hw/gpgpu/
├── Kconfig
├── meson.build
├── gpgpu.c
├── gpgpu.h
├── gpgpu_core.c
├── gpgpu_core.h
├── gpgpu_riscv.h
├── baremetal/
│   ├── Makefile
│   ├── linker.ld
│   ├── start.S
│   ├── main.c
│   ├── thread_add_kernel.c
│   ├── uart_smoke.c
│   └── qemu_virt_platform.h
├── runtime/
│   ├── gpgpu_runtime.c
│   ├── gpgpu_runtime.h
│   ├── gpgpu_pci.c
│   ├── gpgpu_pci.h
│   └── gpgpu_freestanding.h
├── dev/
│   ├── Dockerfile
│   ├── build-image.zsh
│   └── shell.zsh
└── docs/
    ├── README.md
    ├── ABI.md
    ├── DESIGN_NOTES.md
    ├── SIMT_EXECUTION_NOTES.md
    ├── STRUCTURE.md
    ├── RUNTIME.md
    ├── BAREMETAL_PCI.md
    └── DEV_ENV.md
```

`baremetal/build/` 是生成目录，已经由 `.gitignore` 忽略，不属于源码结构。

## QEMU Device Layer

核心设备模型在：

```text
gpgpu.c
gpgpu.h
```

职责：

- PCI 设备建模。
- BAR0 控制寄存器。
- BAR2 VRAM。
- launch 寄存器处理。
- dispatch kernel，遍历 grid/block/warp。

这层模拟 GPU 的控制面和 work distributor。

## Core / SIMT Layer

核心执行模型在：

```text
gpgpu_core.c
gpgpu_core.h
gpgpu_riscv.h
```

职责：

- 初始化 warp/lane。
- 维护 lane PC/GPR/FPR。
- 串行模拟 SIMT 执行。
- 解释当前支持的 RV32 指令。
- 处理 `GPGPU_CORE_CTRL_*` builtin register load。

进入 CNN 阶段后，这层最需要增强：

- branch/jump。
- `mul` 或更多 RV32M 指令。
- 更完整的 load/store 类型。
- 更明确的错误上报。
- 后续可能加入 divergence/predicate/barrier。

## Runtime Layer

最小 runtime 在：

```text
runtime/
```

职责：

- PCI ECAM 扫描。
- BAR 获取。
- VRAM bump allocator。
- host 到 VRAM 的读写。
- kernel code 上传。
- user args 打包。
- launch MMIO 写入。

当前 ABI 是：

```text
x10 = kernel_args = user_args[0] 地址
gridDim/blockDim = launch MMIO 写入设备状态
threadIdx/blockIdx/blockDim/gridDim = kernel 通过 builtin register 读取
```

runtime 不再把 `gridDim/blockDim` 塞进 kernel args header。

## Baremetal Test Layer

裸机测试在：

```text
baremetal/
```

职责：

- 在 QEMU `virt` 上启动 RISC-V baremetal 程序。
- 初始化 UART 输出。
- 使用 runtime 枚举 GPGPU PCI 设备。
- 上传 kernel。
- launch 并检查结果。

当前有两类测试：

- `main.c`：GPGPU PCI/runtime/kernel smoke。
- `uart_smoke.c`：单独验证 UART 和裸机入口。

`thread_add_kernel.c` 是设备侧 C kernel。Makefile 会把它编译为 RV32 `.text`，再生成 `baremetal/build/thread_add_kernel.inc` 给 `main.c` 上传。

## Dev Environment

容器环境在：

```text
dev/
```

职责：

- 提供 Ubuntu 24.04 开发镜像。
- 安装 QEMU 构建依赖。
- 安装 RISC-V baremetal 工具链。
- 安装 Rust/bindgen 等构建工具。
- 在 Colima/Docker 外部工作区中启动 shell。

## 文档分工

```text
docs/ABI.md
```

记录软件和设备之间的 ABI：launch 寄存器、kernel entry、kernel args、builtin register 地址模型。

```text
docs/SIMT_EXECUTION_NOTES.md
```

记录 SIMT 概念和当前模拟器行为，解释 `threadIdx`、`blockIdx`、global index、`mhartid`、串行模拟等容易混淆的点。

```text
docs/DESIGN_NOTES.md
```

保留更高层设计记录和阶段性方案。

```text
docs/RUNTIME.md
docs/BAREMETAL_PCI.md
docs/DEV_ENV.md
```

分别记录 runtime、裸机 PCI、Docker/Colima 开发环境。

## 进入 CNN 阶段前的维护原则

- 源码保留在 `hw/gpgpu/`，生成物放入 `baremetal/build/`。
- 不提交 ELF、bin、object、反汇编、生成的 include。
- 设备侧 kernel 可以先放在 `baremetal/`，等数量变多后再拆出 `kernels/`。
- 每新增一个 kernel，保留 C 源码和构建规则，不保留构建产物。
- 每次扩展解释器指令，应补一个能触发该指令的 C kernel 或 baremetal smoke。

## CNN 阶段建议拆分

以执行一个最小 CNN 为目标，建议先按以下顺序推进：

```text
1. 扩展 RV32 指令子集，让普通 C kernel 能使用 if/loop/mul。
2. 增加 vector add / elementwise / relu 这类简单 C kernel。
3. 增加 2D tensor indexing smoke，验证 width/height/stride 参数。
4. 增加 naive conv2d kernel，不先追求性能。
5. 增加 host/baremetal 侧 tensor 初始化和 golden result 校验。
6. 再考虑 tiling/shared memory/barrier/优化。
```

当前软件栈的目标应先是功能正确，而不是性能接近真实 GPU。
