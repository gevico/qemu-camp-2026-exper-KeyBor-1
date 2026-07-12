# Vortex CSR / Thread-Block ID 机制证据

本文档记录 vortex GPU 模拟器关于 thread ID、warp ID、block ID 的完整证据。
所有引用均来自 vortex 源代码。

## 1. Vortex GPU CSR 定义 (VX_types.vh:207-216)

vortex 有专用的 GPU CSR，不仅仅是 mhartid：

| CSR 地址 | 名称 | 含义 |
|----------|------|------|
| `0xCC0` | `VX_CSR_THREAD_ID` | warp 内线程 ID (0..NUM_THREADS-1) |
| `0xCC1` | `VX_CSR_WARP_ID` | core 内 warp ID (0..NUM_WARPS-1) |
| `0xCC2` | `VX_CSR_CORE_ID` | core ID (0..NUM_CORES-1) |
| `0xCC3` | `VX_CSR_ACTIVE_WARPS` | 活跃 warp 位掩码 |
| `0xCC4` | `VX_CSR_ACTIVE_THREADS` | 活跃线程位掩码 |
| `0xFC0` | `VX_CSR_NUM_THREADS` | 每 warp 线程数 |
| `0xFC1` | `VX_CSR_NUM_WARPS` | 每 core warp 数 |
| `0xFC2` | `VX_CSR_NUM_CORES` | 总 core 数 |
| `0xF14` | `VX_CSR_MHARTID` | 全局唯一 hart ID |

## 2. mhartid 编码 — emulator.cpp:503

```cpp
case VX_CSR_MHARTID:
    return (core_->id() * arch_.num_warps() + wid) * arch_.num_threads() + tid;
```

**mhartid = (core_id * NUM_WARPS + warp_id) * NUM_THREADS + thread_id**

**关键发现：mhartid 不包含 block_id。** vortex 的 mhartid 编码的是物理硬件位置
(core, warp, thread)，不是逻辑 GPU 网格位置 (block, thread)。

默认配置 (VX_config.vh): NUM_CORES=1, NUM_WARPS=4, NUM_THREADS=4
→ mhartid = (0 * 4 + warp_id) * 4 + thread_id = warp_id * 4 + thread_id

## 3. 专用 GPU CSR 实现 — emulator.cpp:504-512

```cpp
case VX_CSR_THREAD_ID:    return tid;                          // warp 内线程索引
case VX_CSR_WARP_ID:      return wid;                          // warp 索引
case VX_CSR_CORE_ID:      return core_->id();                  // core 索引
case VX_CSR_NUM_THREADS:  return arch_.num_threads();           // 每 warp 线程数
case VX_CSR_NUM_WARPS:    return arch_.num_warps();             // 每 core warp 数
case VX_CSR_NUM_CORES:    return uint32_t(arch_.num_cores()) * arch_.num_clusters();
```

**关键发现：vortex 提供了专用 CSR 来获取 thread_id 和 warp_id，不需要从 mhartid 解码。**

## 4. Vortex 没有硬件 block_id

搜索整个 vortex 代码库，不存在 `VX_CSR_BLOCK_ID` 或任何 block/group ID 的硬件 CSR。

Block ID (`blockIdx`) 是 vortex 运行时 `vx_spawn.c` 纯软件计算的：

```c
// vx_spawn.c:138-148 — 分组模式
for (uint32_t group_id = start_group; group_id < end_group; group_id += group_stride) {
    blockIdx.x = group_id % gridDim_x;
    blockIdx.y = (group_id / gridDim_x) % gridDim_y;
    blockIdx.z = group_id / (gridDim_x * gridDim_y);
    callback((void*)arg);
}
```

`blockIdx` 是 `__thread` 变量，由 runtime 根据 grid/block 维度和 worker offset 计算。

## 5. Vortex 内核启动流程 — emulator.cpp:99-134

```cpp
void Emulator::reset() {
    uint64_t startup_addr = dcrs_.base_dcrs.read(VX_DCR_BASE_STARTUP_ADDR0);
    startup_addr |= (uint64_t(dcrs_.base_dcrs.read(VX_DCR_BASE_STARTUP_ADDR1)) << 32);
    uint64_t startup_arg = dcrs_.base_dcrs.read(VX_DCR_BASE_STARTUP_ARG0);
    startup_arg |= (uint64_t(dcrs_.base_dcrs.read(VX_DCR_BASE_STARTUP_ARG1)) << 32);

    for (auto& warp : warps_) {
        warp.reset(startup_addr);   // 所有 warp PC = startup_addr
    }
    csr_mscratch_ = startup_arg;   // MSCRATCH = 内核参数地址
    active_warps_.set(0);           // 只有 warp 0 初始活跃
    warps_[0].tmask.set(0);         // warp 0 中只有线程 0 初始活跃
}
```

流程:
1. DCR 写入 STARTUP_ADDR → 内核代码地址 (0x80000000)
2. DCR 写入 STARTUP_ARG → 内核参数地址 (0x80100000, STARTUP_ADDR + 1MB)
3. `processor.run()` → 每个 core 的 `emulator.reset()`
4. 所有 warp PC 指向 startup_addr
5. 只有 warp0/thread0 初始活跃（单线程启动）
6. `csr_mscratch_` = startup_arg
7. 内核 `_start` 从 mscratch 读取参数指针，可以调用 `vx_spawn_threads()` 激活更多 warp/线程

## 6. 与 rv32 interpreter 的差异

| 特性 | rv32 interpreter | vortex |
|------|-----------------|--------|
| thread_id | MMIO `0x80000000` + offset | CSR `0xCC0` (VX_CSR_THREAD_ID) |
| warp_id | MMIO 或 mhartid 解码 | CSR `0xCC1` (VX_CSR_WARP_ID) |
| num_threads | 硬编码 GPGPU_WARP_SIZE=32 | CSR `0xFC0` (VX_CSR_NUM_THREADS) |
| num_warps | 动态分配 | CSR `0xFC1` (VX_CSR_NUM_WARPS) |
| block_id | MMIO `0x80000010` | **无硬件支持** |
| block_dim | MMIO `0x80000020` | **无硬件支持** |
| grid_dim | MMIO `0x80000030` | **无硬件支持** |
| mhartid | 编码 block+warp+thread | 编码 core+warp+thread |

## 7. 解决方案: SIMT 元数据注入

由于 vortex 没有 block_id/block_dim/grid_dim 的硬件 CSR，
我们的 backend adapter (`gpgpu_backend_vortex.c`) 在每次 dispatch 时将这些值写入
vortex RAM 的固定地址区域 (VORTEX_SIMT_META_ADDR = 0x7FFFF000, STARTUP_ADDR 下方 4KB)，
供 kernel builtins 读取。

```
vortex RAM 布局:
  0x7FFFF000  SIMT metadata (block_id, block_dim, grid_dim)
  0x80000000  STARTUP_ADDR (kernel code)
  0x80100000  ARGS_ADDR (kernel args)
```

Kernel 使用 CSR 指令获取 thread_id/warp_id，使用 RAM load 获取 block_id/dim。

## 8. 线程维度限制

Vortex 是 1D 线程模型 (只有 thread_id_x)。rv32 interpreter 支持 3D (thread_id_x/y/z)。
对于 vortex 后端，`gpgpu_thread_id_y()` 和 `gpgpu_thread_id_z()` 返回 0。
