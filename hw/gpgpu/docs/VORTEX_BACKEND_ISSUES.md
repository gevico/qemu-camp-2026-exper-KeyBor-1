# Vortex Backend Issue Log

本文档记录 Vortex backend 接入后的已知问题和后续排查方向。

## 背景

当前 Vortex backend 的总体方向是:

1. QEMU `gpgpu` 设备仍然使用统一的 `gpgpu_backend` 抽象。
2. Kernel 源码通过 `runtime/include/gpgpu_builtins.h` 在 rv32 interpreter 和 Vortex backend 之间切换 builtin 实现。
3. Vortex 原生 CSR 提供 `thread_id` / `warp_id` / `num_threads` / `num_warps`。
4. Vortex 没有 CUDA-like `block_id` / `block_dim` / `grid_dim` CSR，因此 backend 在 dispatch 时将这些 SIMT metadata 注入到 Vortex RAM 固定地址。

这个方向可以继续保留，但当前实现还没有完全对齐原有 CUDA-like runtime ABI。

## 待处理问题

### 1. Vortex `thread_id_x` 语义和 CUDA-like `threadIdx.x` 不一致

位置:

- `runtime/include/gpgpu_builtins.h`

当前 Vortex 版 `gpgpu_thread_id_x()` 直接读取 CSR `0xCC0`。根据 Vortex CSR 证据，该值是 **warp 内 thread id**，不是 **block 内 threadIdx.x**。

如果一个 block 内有多个 warp，正确的 block 内 linear thread 应该先组合:

```c
linear = warp_id * num_threads + thread_id_x;
```

然后再根据 `block_dim` 还原 CUDA-like 三维 thread id:

```c
x = linear % block_dim_x;
y = (linear / block_dim_x) % block_dim_y;
z = linear / (block_dim_x * block_dim_y);
```

当前 `gpgpu_thread_id_y()` / `gpgpu_thread_id_z()` 直接返回 0，因此 2D/3D thread kernel 在 Vortex backend 下会产生错误索引。

建议:

- 先在 builtin 层把 Vortex CSR 转换为和 rv32 interpreter 一致的 `threadIdx.{x,y,z}` 语义。
- 如果 Vortex 硬件线程数量小于 `block_dim.x * block_dim.y * block_dim.z`，需要明确是否由软件循环覆盖剩余 thread，或限制每次 launch 的 block size。

### 2. 每个 block 调用一次 `Processor::run()`，但没有明确 reset processor 状态

位置:

- `backend/src/gpgpu_backend_vortex.c`
- `backend/arch/riscv/vortex_backend/vortex_simx_wrapper.cpp`

当前 `vortex_dispatch()` 遍历 grid 中的每个 block，每个 block 都调用一次 `vortex_exec_one_block()`，最终进入 `g_processor->run()`。

但 wrapper 没有在每次 block 执行前显式 reset processor/emulator 状态。需要确认 Vortex 的 `Processor::run()` 是否内部执行 reset。如果没有，第二个 block 可能继承第一个 block 的 PC、warp active mask、CSR 或完成状态，导致 hang、no-op 或错误退出。

建议:

- 查清 Vortex `Processor::run()` / emulator reset 的实际调用链。
- 如果 `run()` 不负责重置状态，考虑每个 block 前显式 reset，或者每个 block 使用新的 Processor 实例。
- 更长期的方向是让一个 Vortex run 处理完整 grid，而不是 QEMU adapter 在外层逐 block 重启模拟器。

### 3. Kernel 长度没有 ABI，当前固定拷贝 64KB 风险较高

位置:

- `backend/src/gpgpu_backend_vortex.c`

当前 Vortex backend 会从 QEMU VRAM 中读取固定 `KERNEL_BUF_SIZE`，并在前面插入 trampoline:

```c
buf[0..3] = csrr a0, mscratch;
buf[4..] = vram[kernel_addr .. kernel_addr + 64KB);
```

这意味着 kernel 后面的 VRAM 内容也会被加载到 Vortex 指令区域。如果 Vortex 退出条件不严格依赖一个明确 trap/exit protocol，就可能执行到非 kernel 数据，导致 hang 或不可预测行为。

建议:

- 在 launch ABI 中记录 `kernel_size`。
- 或者将 kernel 打包为明确的 executable image，并定义 Vortex backend 能识别的结束协议。
- 不要长期依赖固定 64KB 指令拷贝。

### 4. Vortex smoke 当前覆盖范围过大

位置:

- `baremetal/Makefile`

当前 `vortex-smoke` 会生成 Vortex kernel inc，然后覆盖 `build/` 下的普通 rv32 kernel inc，再运行完整裸机 smoke。

这个流程有两个问题:

1. Vortex backend 还没有证明支持完整 2D/3D thread、multi-block、stack、matmul、conv、LeNet 等路径。
2. 通过复制 `.inc` 覆盖共享 build 产物，容易污染后续 rv32 构建或让测试结果依赖 build 目录状态。

建议先拆出 Vortex 专用最小 smoke:

```text
grid=(1,1,1), block=(1,1,1)
grid=(1,1,1), block=(4,1,1)
grid=(1,1,1), block=(8,1,1)
grid=(2,1,1), block=(4,1,1)
```

等这些通过后，再逐步加入 2D/3D、stack、linear、matmul、conv。

构建侧建议:

- Vortex kernel inc 使用独立 build 目录或独立 target。
- 避免覆盖普通 rv32 kernel inc。
- `run-vortex` 不应依赖 build 目录里当前残留的是哪一套 inc。

### 5. Backend capability 当前可能过度声明

位置:

- `backend/src/gpgpu_backend_vortex.c`

当前 Vortex backend 声明:

```c
GPGPU_BACKEND_CAP_BARRIER |
GPGPU_BACKEND_CAP_SHARED_MEM |
GPGPU_BACKEND_CAP_DIVERGENCE
```

但当前 adapter 还没有证明这些能力和现有 runtime ABI 完整对齐。若后续 runtime 根据 caps 选择 kernel，过度声明会导致上层错误选择 Vortex 尚未支持的路径。

建议:

- 在 smoke 覆盖之前，只声明已经验证的能力。
- 每新增一个 capability，都配一个最小测试。

### 6. Vortex RAM 到 QEMU VRAM 的同步范围过窄

位置:

- `backend/src/gpgpu_backend_vortex.c`

当前执行后只同步:

```c
VORTEX_ARGS_ADDR -> kernel_args, 4KB
Vortex RAM [0, 64KB) -> QEMU VRAM [0, 64KB)
```

这对简单 smoke 可能够用，但对后续 CNN/LeNet 不够稳。权重、输入、输出、中间 tensor 很容易超过 64KB，固定同步低地址区域也无法表达真实写入范围。

建议:

- 短期可扩大同步范围用于验证。
- 中期应由 runtime/allocator 提供 buffer 元信息，按实际分配区间同步。
- 长期可考虑 dirty page 或统一内存模型。

### 7. 缺少 timeout / step limit，Vortex hang 会拖死 QEMU 进程

位置:

- `backend/arch/riscv/vortex_backend/vortex_simx_wrapper.cpp`

当前 `vortex_wrapper_run()` 直接调用 `g_processor->run()`，没有 timeout、step limit 或 watchdog。只要 Vortex 内核未触发退出条件，QEMU 进程就会卡死。

建议:

- 调研 Vortex simulator 是否支持 step limit。
- 如果支持，先给 smoke 配一个较小上限。
- 如果不支持，至少在文档中标注当前 Vortex smoke 可能 hang，并将测试拆到单独目标。

## 优先级建议

1. 修正 Vortex builtin 的 `threadIdx.{x,y,z}` 语义。
2. 明确 `Processor::run()` 是否会 reset；若不会，先修每 block 重新执行的问题。
3. 拆出 Vortex 专用最小 smoke，避免直接跑全量 smoke。
4. 给 kernel size / kernel exit protocol 建 ABI。
5. 再逐步验证并开放 backend capabilities。

## 当前判断

Vortex 后端接入方向没有问题，但现在还不能把 Vortex 当作 rv32 interpreter 的透明替代品。最大风险点在于:

- Vortex 硬件线程模型和当前 CUDA-like builtin 语义没有完全映射。
- 当前 adapter 用逐 block 重启模拟器的方式模拟 grid，但 simulator 生命周期和退出条件还没有验证清楚。
- 构建和 smoke 流程还没有把 Vortex 的未完成状态隔离出来。
