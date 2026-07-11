# GPGPU Directory Structure

本文档记录 `hw/gpgpu/` 当前目录结构，以及进入 CNN kernel 阶段前各部分的职责边界。

## 目录树

```text
hw/gpgpu/
├── Kconfig
├── meson.build
├── gpgpu.c
├── gpgpu.h
├── arch/
│   └── riscv/
│       ├── gpgpu_core.c
│       ├── gpgpu_core.h
│       └── gpgpu_riscv.h
├── kernels/
│   ├── thread_add_kernel.c
│   ├── relu_i32_kernel.c
│   ├── linear_i32_kernel.c
│   ├── linear_partial_i32_kernel.c
│   ├── linear_reduce_i32_kernel.c
│   ├── matmul_partial_i32_kernel.c
│   ├── matmul_reduce_i32_kernel.c
│   ├── im2col_i32_kernel.c
│   ├── oihw_to_ko_i32_kernel.c
│   └── maxpool_i32_kernel.c
├── baremetal/
│   ├── Makefile
│   ├── include/
│   │   └── qemu_virt_platform.h
│   ├── scripts/
│   │   ├── linker.ld
│   │   └── start.S
│   ├── src/
│   │   ├── main.c
│   │   └── uart_smoke.c
│   └── drivers/
│       ├── include/
│       │   └── gpgpu_pci.h
│       └── src/
│           └── gpgpu_pci.c
├── runtime/
│   ├── include/
│   │   ├── gpgpu_runtime.h
│   │   ├── gpgpu_tensor.h
│   │   ├── gpgpu_nn.h
│   │   └── gpgpu_freestanding.h
│   └── src/
│       └── gpgpu_runtime.c
├── dev/
│   ├── Dockerfile
│   ├── build-image.zsh
│   └── shell.zsh
└── docs/
    ├── README.md
    ├── ABI.md
    ├── DESIGN_NOTES.md
    ├── SIMT_EXECUTION_NOTES.md
    ├── FAQ.md
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
arch/riscv/gpgpu_core.c
arch/riscv/gpgpu_core.h
arch/riscv/gpgpu_riscv.h
```

职责：

- 初始化 warp/lane。
- 维护 lane PC/GPR/FPR。
- 串行模拟 SIMT 执行。
- 解释当前支持的 RV32 指令。
- 处理 `GPGPU_CORE_CTRL_*` builtin register load。

进入 CNN 阶段后，这层仍需要增强：

- 除 `mul` 外的更多 RV32M 指令。
- 更完整的 load/store 类型。
- 更明确的错误上报。
- 后续可能加入 divergence/predicate/barrier。

## Runtime Layer

最小 runtime 在：

```text
runtime/
```

职责：

- VRAM bump allocator。
- host 到 VRAM 的读写。
- kernel code 上传。
- user args 打包。
- launch MMIO 写入。
- tensor/op/network descriptor 定义。

当前 ABI 是：

```text
x10 = kernel_args = user_args[0] 地址
gridDim/blockDim = launch MMIO 写入设备状态
threadIdx/blockIdx/blockDim/gridDim = kernel 通过 builtin register 读取
```

runtime 不再把 `gridDim/blockDim` 塞进 kernel args header。

PCI ECAM 扫描和 BAR 初始化不属于通用 runtime，当前放在裸机 driver：

```text
baremetal/drivers/
```

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

- `src/main.c`：GPGPU PCI/runtime/kernel smoke。
- `src/uart_smoke.c`：单独验证 UART 和裸机入口。

`kernels/thread_add_kernel.c`、`kernels/relu_i32_kernel.c`、
`kernels/linear_i32_kernel.c`、`kernels/linear_partial_i32_kernel.c` 和
`kernels/linear_reduce_i32_kernel.c`、`kernels/matmul_partial_i32_kernel.c`
和 `kernels/matmul_reduce_i32_kernel.c`、`kernels/im2col_i32_kernel.c`、
`kernels/oihw_to_ko_i32_kernel.c`、`kernels/maxpool_i32_kernel.c` 是设备侧
C kernel。Makefile 会把它们编译为 RV32 `.text`，再生成
`baremetal/build/*_kernel.inc` 给 `baremetal/src/main.c` 上传。

## Device Kernels

设备侧 kernel 在：

```text
kernels/
```

这些 kernel 运行在 GPGPU core 上，不属于裸机 host。当前 baremetal 程序只是把它们编译成 `.text`、上传到 VRAM 并 launch。后续进入 Linux driver 或用户态 runtime 时，也应该复用同一批 kernel 源码。

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
- 设备侧 kernel 放在 `kernels/`，不放在 `baremetal/`。
- 每新增一个 kernel，保留 C 源码和构建规则，不保留构建产物。
- 每次扩展解释器指令，应补一个能触发该指令的 C kernel 或 baremetal smoke。

## CNN 阶段建议拆分

以执行一个最小 CNN 为目标，建议先按以下顺序推进：

```text
1. 增加 vector add / elementwise / relu 这类简单 C kernel。
2. 增加 2D tensor indexing smoke，验证 width/height/stride 参数。
3. 增加 naive conv2d kernel，不先追求性能。
4. 增加 host/baremetal 侧 tensor 初始化和 golden result 校验。
5. 按 kernel 反汇编继续补齐缺失的 RISC-V 指令。
6. 再考虑 tiling/shared memory/barrier/优化。
```

当前软件栈的目标应先是功能正确，而不是性能接近真实 GPU。
