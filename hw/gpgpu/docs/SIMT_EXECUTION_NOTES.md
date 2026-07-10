# SIMT Execution Notes

本文整理当前 GPGPU 模拟器中 SIMT 执行模型的理解。重点不是完整复刻 CUDA，而是把当前项目里容易混淆的概念固定下来：软件 launch 提供什么，硬件/模拟器派生什么，kernel 读到的 `threadIdx` 和全局数组索引是什么关系。

## 1. 最终分工

软件 launch 只负责告诉 GPU：

```text
kernel_addr
kernel_args
gridDim.x/y/z
blockDim.x/y/z
```

硬件或当前模拟器负责根据这些信息派生：

```text
blockIdx.x/y/z
warpId
laneId
threadIdx.x/y/z
```

kernel 负责根据这些内建变量计算自己要访问的数据位置：

```c
global_x = blockIdx.x * blockDim.x + threadIdx.x;
global_y = blockIdx.y * blockDim.y + threadIdx.y;
global_z = blockIdx.z * blockDim.z + threadIdx.z;
```

因此当前 ABI 应该是：

```text
x10 = kernel_args = user_args[0] 的地址
gridDim/blockDim = launch MMIO 写入设备寄存器
threadIdx/blockIdx/blockDim/gridDim = kernel 通过 builtin register 读取
```

不需要再把 `gridDim/blockDim` 放进 `kernel_args` header。它们已经在 launch 时写入设备状态，再放一份会形成重复信息源。

## 2. threadIdx 不是 global index

`threadIdx` 的语义是 block 内部坐标。

例如：

```text
blockDim.x = 256
```

那么每个 block 内部：

```text
threadIdx.x = 0..255
```

对于一维大数组，kernel 通常自己计算：

```c
global_i = blockIdx.x * blockDim.x + threadIdx.x;
```

举例：

```text
blockIdx.x = 0, threadIdx.x = 5 -> global_i = 5
blockIdx.x = 1, threadIdx.x = 5 -> global_i = 261
```

所以 lane 最终访问数组时一般用 global index，但这个 global index 是 kernel 根据 `blockIdx + blockDim + threadIdx` 计算出来的，不是硬件直接替换 `threadIdx`。

保留 block-local `threadIdx` 很重要，因为很多 kernel 不只是访问一维数组，还会做 tile、shared memory、barrier、二维/三维坐标计算等操作。

## 3. 二维访问和一维地址

kernel 可以写成二维逻辑：

```c
x = blockIdx.x * blockDim.x + threadIdx.x;
y = blockIdx.y * blockDim.y + threadIdx.y;

idx = y * width + x;
out[idx] = in[idx];
```

如果 C 里写：

```c
c[y][x] = value;
```

编译器最终也会生成类似：

```c
*(base + (y * width + x) * sizeof(element)) = value;
```

所以从 GPU core 视角看，没有真正的“二维内存访问”。内存仍然是一维地址空间，二维/三维只是 kernel 用坐标计算地址的方式。

## 4. 当前模拟器如何生成 threadIdx

当前 dispatch 逻辑遍历：

```text
blockIdx.x/y/z
warpId
laneId
```

每个 warp 有一个 block-local 的 thread 起点：

```c
thread_id_base = warp_id * warp_size;
```

执行某个 lane 时：

```c
thread_id_linear = thread_id_base + lane_id;
```

然后拆成 3D block-local `threadIdx`：

```c
threadIdx.x = thread_id_linear % blockDim.x;
threadIdx.y = (thread_id_linear / blockDim.x) % blockDim.y;
threadIdx.z = thread_id_linear / (blockDim.x * blockDim.y);
```

注意：这里的 `thread_id_linear` 是 block 内部 linear id，不是整个 grid 的 global linear id。

如果需要 global linear id，kernel 可以计算：

```c
local_id =
    threadIdx.z * blockDim.x * blockDim.y +
    threadIdx.y * blockDim.x +
    threadIdx.x;

block_id =
    blockIdx.z * gridDim.x * gridDim.y +
    blockIdx.y * gridDim.x +
    blockIdx.x;

global_id = block_id * (blockDim.x * blockDim.y * blockDim.z) + local_id;
```

## 5. mhartid 的含义

当前 `mhartid` 编码是：

```text
bits [31:13] block_id_linear
bits [12:5]  warp_id
bits [4:0]   lane_id
```

低 5 bit 只能表达 `0..31`，所以它不是完整 block 内 thread id。对于一个 block 内超过 32 个 thread 的情况，第二个 warp 的 lane 仍然是 `0..31`。

因此：

```text
mhartid 低 5 bit = lane_id
完整 threadIdx = 通过 builtin register 读取
```

不要用 `mhartid & 0x1f` 当作完整 `threadIdx.x`。

## 6. builtin register 是什么时候更新的

当前实现中，`GPGPU_CORE_CTRL_THREAD_ID_X` 等 builtin register 的数据源是 `s->simt`。

执行路径是串行模拟：

```text
for each block:
  for each warp:
    while active lanes remain:
      for each lane:
        更新 s->simt 为当前 lane 的上下文
        执行当前 lane 的一条指令
```

所以当前模型是：

```text
一个全局 s->simt
每次执行某个 lane 前，把它改成当前 lane 的 threadIdx/blockIdx/warpId/laneId
kernel 执行 lw GPGPU_CORE_CTRL_* 时，从 s->simt 返回值
```

这只是模拟器简化。真实 GPU 不会这样串行改一个全局寄存器。

更真实的抽象是：

```text
warp/block context:
  blockIdx
  blockDim
  gridDim
  warpId
  active mask
  PC

lane/thread context:
  laneId
  per-lane GPR/FPR
  predicate state
```

kernel 读取 `threadIdx` 时，硬件可能通过 special register read 动态生成，而不一定真的为每个 lane 存一份物理寄存器。

## 7. 当前执行是串行模拟 SIMT

当前模拟器不是一次执行完整 warp 指令，而是 lane by lane 串行解释。

更准确地说，不是：

```text
lane0 从头跑完整个 kernel
lane1 从头跑完整个 kernel
...
```

而是：

```text
lane0 执行第 1 条指令
lane1 执行第 1 条指令
...
lane31 执行第 1 条指令

lane0 执行第 2 条指令
lane1 执行第 2 条指令
...
```

每个 lane 有自己的：

```c
pc
gpr[]
fpr[]
```

遇到 `ebreak` 后，该 lane 从 active mask 中清除。

真实 GPU 更接近：

```text
warp scheduler 取一条 warp 指令
这条指令同时发给 active lanes
每个 lane 用自己的寄存器执行
```

当前模拟器用串行循环来得到相同的结果，但不模拟真实并行时序。

## 8. 软件不会给每个 thread 单独写 ID

真实 GPU 中：

```text
gridDim/blockDim/kernel args 由软件 launch 提供
blockIdx/warpId/threadIdx/laneId 由硬件调度和派生
```

软件不会逐个写：

```text
thread 0 的 threadIdx = ...
thread 1 的 threadIdx = ...
thread 2 的 threadIdx = ...
```

那会太慢，也不符合 GPU 执行模型。

对应到当前模拟器：

```text
gpgpu_launch()
  模拟软件 launch，写 kernel_addr/kernel_args/gridDim/blockDim

gpgpu_dispatch_kernel()
  模拟硬件 work distributor，遍历 block 和 warp

gpgpu_core_exec_warp()
  模拟 warp/lane 执行，生成当前 lane 的 threadIdx
```

## 9. C kernel 的当前限制

现在已经可以用 C 写设备 kernel，再由 RISC-V 编译器编译成 `.text` binary 上传执行。

但当前 core interpreter 只支持很小的 RV32 子集，例如：

```text
lui
addi
slli
andi
add
lw
sw
system/ebreak
部分 fp 指令
```

因此 C kernel 需要避免生成当前不支持的指令，例如：

```text
branch
jal/jalr
mul
auipc
复杂栈访问
函数调用返回
```

当前 `thread_add_kernel.c` 能跑，是因为它被写成：

```text
无循环
无 if
无函数调用返回
用 shift 代替乘法
末尾显式 ebreak
```

后续如果希望写普通 C kernel，就需要逐步补齐解释器的 RISC-V 指令支持。

## 10. 当前应坚持的模型

当前项目里最清晰的模型是：

```text
软件：
  提供 kernel_addr/kernel_args/gridDim/blockDim

模拟硬件 dispatch：
  遍历 blockIdx
  按 blockDim 切 warp
  为每个 lane 派生 block-local threadIdx

kernel：
  读取 builtin register
  自己计算 global index
  用 global index 访问 VRAM 数组
```

不要把 `threadIdx` 改成 global id，也不要把 `gridDim/blockDim` 再塞进 kernel args header。这样才能保持 CUDA-like 的语义，也能让一维、二维、三维 kernel 都有一致的表达方式。
