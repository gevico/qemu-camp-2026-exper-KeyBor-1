# GPGPU Docs

本目录集中保存 `hw/gpgpu/` 的设计、运行环境和学习笔记。

## 文档索引

- [STRUCTURE.md](STRUCTURE.md): 目录结构和模块职责。
- [ABI.md](ABI.md): launch ABI、kernel entry ABI、builtin register 地址模型。
- [SIMT_EXECUTION_NOTES.md](SIMT_EXECUTION_NOTES.md): SIMT 执行模型、`threadIdx`/`blockIdx`/global index、当前串行模拟方式。
- [FAQ.md](FAQ.md): 实现过程中出现过的关键疑问和结论。
- [DESIGN_NOTES.md](DESIGN_NOTES.md): 阶段性设计记录。
- [RUNTIME.md](RUNTIME.md): 最小 runtime API 和职责。
- [BAREMETAL_PCI.md](BAREMETAL_PCI.md): QEMU `virt` 裸机 PCI/ECAM/BAR 说明。
- [DEV_ENV.md](DEV_ENV.md): Colima/Docker 开发环境。
- [vortex-csr-evidence.md](vortex-csr-evidence.md): Vortex CSR、thread/warp/block 语义证据。
- [VORTEX_BACKEND_ISSUES.md](VORTEX_BACKEND_ISSUES.md): Vortex backend 接入后的已知问题和后续追踪清单。

## 当前阶段

当前 `hw/gpgpu/` 已经完成裸机 CUDA-like 最小闭环：

```text
baremetal RISC-V -> PCI scan -> BAR0/BAR2 -> runtime -> launch -> SIMT core -> C kernel
```

下一阶段目标是支持 CNN kernel。当前已经具备简单 C loop kernel 所需的基础 branch/jump/mul，后续重点是补充 tensor indexing smoke，并按 naive conv2d 反汇编继续补齐缺失指令。
