# GPGPU Learning FAQ

本文整理实现裸机 CUDA-like GPGPU 软件栈过程中反复出现的疑问和最终结论。它不是正式 ABI 规范，而是学习笔记，用来记录容易混淆的点。

## 1. PCI ECAM 地址公式是什么意思？

PCI ECAM 地址计算：

```c
addr = ecam_base
     + (bus << 20)
     + (device << 15)
     + (function << 12)
     + offset;
```

含义：

```text
每个 bus      1MB config space
每个 device   32KB
每个 function 4KB
```

所以一个 bus 里最多：

```text
1MB / 32KB = 32 devices
```

`function` 可以理解为同一个 PCI device slot 下的独立功能单元。一个物理设备可以暴露多个 function，每个 function 都有自己的 PCI config space。

## 2. vendor id / device id 是怎么来的？

baremetal 通过 PCI config space 读取：

```text
offset 0x00: vendor id
offset 0x02: device id
```

当前 GPGPU 设备使用：

```text
vendor = 0x1234
device = 0x1337
```

裸机 runtime 扫描 ECAM，匹配到这个 vendor/device 后，再探测 BAR0/BAR2 并配置 MMIO 地址。

## 3. BAR0 和 GPU builtin register 是一回事吗？

不是。

```text
BAR0:
  host/CPU 通过 PCIe MMIO 访问的控制寄存器空间。
  用于 launch、gridDim/blockDim、kernel_addr、kernel_args 等控制面。

GPGPU_CORE_CTRL_BASE:
  device kernel 通过 lw 读取的内部 builtin register 地址空间。
  用于 threadIdx/blockIdx/blockDim/gridDim 等 kernel 内建变量。
```

当前二者都表现为“读写某个地址”，但路径不同：

```text
CPU -> PCI BAR0 -> QEMU device control registers
GPU kernel -> core internal address decode -> builtin register value
```

## 4. gridDim/blockDim 应该放进 kernel args 吗？

当前 ABI 下不应该。

原因是 `gpgpu_launch()` 已经通过 BAR0 控制寄存器写入：

```text
gridDim
blockDim
kernel_addr
kernel_args
```

设备内部 dispatch 也已经用这些值切分 block/warp。kernel 需要这些值时，通过 builtin register 读取即可。

因此：

```text
x10 = kernel_args = user_args[0] 的地址
kernel args 不包含 gridDim/blockDim header
```

如果再把 gridDim/blockDim 放进 kernel args，会形成重复信息源。

## 5. kernel args 里应该放什么？

kernel args 里应该放：

```text
用户参数
tensor descriptor
op descriptor
input/weight/output 的 VRAM 地址
shape/stride/padding/stride 等算子参数
```

但不应该把大块 tensor 数据本身塞进 args。

推荐模型：

```text
VRAM:
  input data
  weight data
  bias data
  output data
  op args descriptor

x10:
  指向 op args descriptor
```

op args descriptor 里放的是各个 tensor 的地址和描述，而不是真正数据。

## 6. TensorDesc 是什么？为什么需要它？

TensorDesc 是对一块 tensor 数据的描述：

```c
typedef struct {
    uint32_t data;
    uint32_t dtype;
    uint32_t layout;
    uint32_t n, c, h, w;
    uint32_t stride_n;
    uint32_t stride_c;
    uint32_t stride_h;
    uint32_t stride_w;
} GPGPUTensorDesc;
```

当前定义位于：

```text
runtime/include/gpgpu_tensor.h
```

它描述：

```text
数据在哪里
元素类型是什么
layout 是什么
shape 是多少
stride 是多少
```

这和主流实现类似：

```text
PyTorch Tensor: data pointer + sizes + strides + dtype + device
cuDNN: tensor descriptor / filter descriptor / convolution descriptor
```

没有 TensorDesc，kernel 就容易把 shape 写死，例如固定 `28x28`。有 TensorDesc 后，kernel 可以根据参数支持不同形状。

## 7. threadIdx 是全局索引吗？

不是。

`threadIdx` 是 block 内部坐标：

```text
threadIdx.x/y/z: 当前 thread 在当前 block 内的位置
blockIdx.x/y/z: 当前 block 在 grid 内的位置
```

访问大数组时，kernel 自己计算全局索引：

```c
global_x = blockIdx.x * blockDim.x + threadIdx.x;
global_y = blockIdx.y * blockDim.y + threadIdx.y;
global_z = blockIdx.z * blockDim.z + threadIdx.z;
```

一维数组常见写法：

```c
global_i = blockIdx.x * blockDim.x + threadIdx.x;
```

所以：

```text
lane 最终通常用 global index 访问数据
但 global index 是 kernel 算出来的
硬件/ABI 提供的是 block-local threadIdx 和 blockIdx
```

## 8. 为什么不直接让硬件给每个 lane 一个 global id？

可以额外提供 `global_id` builtin，但不应该替代 `threadIdx`。

原因：

- CUDA-like 语义里 `threadIdx` 就是 block-local。
- 二维/三维 tensor 的 global 地址如何拍平取决于 layout、stride、padding。
- block 内共享、tile、barrier 等逻辑需要 block-local thread id。

所以更合理的是：

```text
保留 threadIdx/blockIdx/blockDim/gridDim
kernel 自己计算 global index
未来可以额外加 GLOBAL_LINEAR_ID 作为便利寄存器
```

## 9. mhartid 低 5 bit 是 thread id 吗？

不是完整 thread id。

当前编码：

```text
bits [31:13] block_id_linear
bits [12:5]  warp_id
bits [4:0]   lane_id
```

低 5 bit 只能表达 `0..31`，本质是 lane id。一个 block 超过一个 warp 后，第二个 warp 的 lane 仍然是 `0..31`。

完整 3D `threadIdx` 应该通过 builtin register 读取：

```text
GPGPU_CORE_CTRL_THREAD_ID_X/Y/Z
```

## 10. 当前 CTRL_THREAD_ID_X 是什么时候更新的？

当前模拟器是串行模拟 SIMT。

在执行某个 lane 的一条指令前，模拟器会更新全局临时上下文：

```c
s->simt.thread_id[0] = ...
s->simt.thread_id[1] = ...
s->simt.thread_id[2] = ...
s->simt.block_id[] = ...
s->simt.warp_id = ...
s->simt.lane_id = ...
```

kernel 执行：

```c
lw GPGPU_CORE_CTRL_THREAD_ID_X
```

解释器就从当前 `s->simt` 返回对应值。

真实 GPU 不会这样串行改一个全局结构。真实硬件更像：

```text
warp/block context 保存 blockIdx/blockDim/gridDim/warpId
lane context 或硬件 lane id 派生 threadIdx/laneId
special register read 返回当前 lane 的值
```

## 11. 当前执行是不是串行的？

是，当前是串行模拟 SIMT。

但不是：

```text
lane0 从头跑完整个 kernel
lane1 从头跑完整个 kernel
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

```text
pc
gpr[]
fpr[]
```

真实 GPU 是一条 warp 指令同时发给多个 active lanes。当前模拟器用 lane-by-lane 循环模拟结果，不模拟真实并行时序。

## 12. 软件会给每个 thread 单独写 thread id 吗？

不会。

软件 launch 提供：

```text
kernel_addr
kernel_args
gridDim
blockDim
```

硬件或模拟器派生：

```text
blockIdx
warpId
laneId
threadIdx
```

软件不会逐个写：

```text
thread0.threadIdx = ...
thread1.threadIdx = ...
```

那太慢，也不符合 GPU 执行模型。

## 13. 二维访问 `c[y][x]` 和一维索引有什么区别？

从源码视角，二维访问更符合矩阵/图像语义：

```c
c[y][x] = value;
```

但编译到底层后仍然是地址计算：

```c
base + (y * width + x) * sizeof(element)
```

所以 GPU core 看到的始终是：

```text
算地址
load/store
```

它不知道源码里写的是 `c[y][x]` 还是 `c[y * width + x]`。

## 14. 网络描述需要放进 VRAM 吗？

第一版不需要。

网络描述可以只存在于 host/baremetal 内存中：

```c
for each node in network:
    gpgpu_launch(node.kernel_addr, node.args_addr, node.grid, node.block);
```

当前基础结构位于：

```text
runtime/include/gpgpu_nn.h
```

GPU 只执行当前 kernel，不需要知道完整网络。

后续如果做 command processor，可以考虑把 network/node descriptor 编成 command buffer 放进 VRAM。但第一版不需要。

## 15. 为什么不把整个 CNN 写成一个 kernel？

可以写固定小 demo，但不适合作为通用实现。

关键问题是跨 block 同步。

一次 kernel launch 内部有多个 block：

```text
block0
block1
block2
...
```

普通 CUDA/OpenCL kernel 内部只能做 block 内同步，不能保证所有 block 在某个中间阶段全部完成。

如果在一个 kernel 里写：

```c
conv1();
relu1();
pool1();
```

可能出现：

```text
block0 已经开始 relu1
block7 的 conv1 还没写完
```

这时跨 block 依赖会读到未完成数据。

拆成多个 launch：

```c
launch conv2d;
launch relu;
launch pool;
```

在同一个 stream/顺序执行模型中，launch 边界天然是全局同步：

```text
conv2d 所有 block 完成后
relu 才开始
```

这也是主流 runtime 的基础执行方式。

## 16. Python/CUDA 里写一个网络，底层也是一个 kernel 吗？

通常不是。

Python 里写：

```python
y = model(x)
```

底层一般是多个 op kernel launch：

```text
conv kernel
relu kernel
pool kernel
gemm kernel
```

成熟框架会做 kernel fusion，例如：

```text
conv + bias + relu
```

但 fusion 是优化阶段。基础模型仍然是：

```text
host/framework graph executor
逐个调度 op kernel
```

## 17. 权重应该怎么加载？

第一版裸机里可以先用 C 静态数组表示权重：

```c
static const int32_t conv1_weight[] = { ... };
```

host/baremetal 侧：

```c
gpgpu_malloc(&dev, &weight_addr, sizeof(conv1_weight));
gpgpu_write(&dev, weight_addr, conv1_weight, sizeof(conv1_weight));
```

然后把 `weight_addr` 放进 tensor descriptor：

```c
args.weight.data = weight_addr;
```

kernel args 传的是权重地址和 shape，不是权重数据本身。

## 18. 为什么 kernel 一个 op 一个 op 地执行更适合当前项目？

因为当前模拟器还没有：

```text
跨 block barrier
shared memory
atomic
command processor
stream
成熟调试工具
```

所以最稳妥的 CNN 推理方式是：

```text
host-side network desc
per-op args in VRAM
tensor/weight data in VRAM
per-op kernel launch
每个 launch 作为全局同步点
每层都能做 CPU golden 校验
```

后续再考虑：

```text
conv + relu fusion
command buffer
network descriptor in VRAM
更复杂的 runtime graph executor
```

## 19. Linear 为什么先做 exact-cover，再做两阶段并行化？

当前有两个 linear smoke，用来表达两个不同阶段。

第一阶段是 exact-cover dot product：

```text
out_features = 3
launch threads = 3

thread0 -> output[0]
thread1 -> output[1]
thread2 -> output[2]
```

每个 thread 串行遍历 `in_features`：

```text
output[o] = bias[o] + sum_i input[i] * weight[o][i]
```

这个阶段不启动多余 thread，也不需要 `out_feature < out_features`
的 guard。它验证的是最小覆盖 launch：runtime 精确下发需要的 thread
数量，kernel 直接执行对应输出。

第二阶段把乘法部分全并行化：

```text
grid.x  = out_features
block.x = in_features

blockIdx.x  -> out_feature
threadIdx.x -> in_feature
```

这样每个 thread 只计算一个乘法：

```text
partial[o][i] = input[i] * weight[o][i]
```

对于当前例子：

```text
out_features * in_features = 3 * 4 = 12
```

所以 partial kernel 会启动 12 个 thread 覆盖 12 次乘法。

但完整 Linear 还需要把每一行 partial 加起来：

```text
output[o] = bias[o] + partial[o][0] + ... + partial[o][in_features - 1]
```

当前模拟器还没有 shared memory、barrier、atomic add 或 warp reduction，
所以不能在一个 kernel 中安全完成跨 thread 归约。第一版并行 linear
因此拆成两个 launch：

```text
launch linear_partial_i32
launch linear_reduce_i32
```

两次 launch 之间的边界就是全局同步点。后续加入 block 内同步、atomic
或 warp-level reduction 后，可以把 reduce 阶段融合回单 kernel。

## 20. MK、KO、MO 这些 layout 到底是什么意思？

layout 的作用是告诉 kernel 如何解释一段连续内存。

内存本身只是：

```text
data[0], data[1], data[2], ...
```

layout 说明这些元素应该按什么维度取下标。矩阵乘当前只讨论三个 layout：

```text
MK: A[m][k]，M 行 K 列
KO: B[k][o]，K 行 O 列
MO: C[m][o]，M 行 O 列
```

也就是标准矩阵乘：

```text
C = A * B
C[m][o] = sum_k A[m][k] * B[k][o]
```

如果 `A` 是 `[M, K]`，行主序 offset 是：

```text
A[m][k] -> m * K + k
```

如果 `B` 是 `[K, O]`：

```text
B[k][o] -> k * O + o
```

如果 `C` 是 `[M, O]`：

```text
C[m][o] -> m * O + o
```

所以 `MK/KO/MO` 不是硬件概念，也不是额外数据本身。它们只是约定：

```text
同一段 VRAM 中的连续数字，kernel 应该按几行几列解释。
```

当前 matmul smoke 使用两阶段：

```text
matmul_partial_i32:
  grid.x  = M
  grid.y  = O
  block.x = K

  blockIdx.x  -> m
  blockIdx.y  -> o
  threadIdx.x -> k

  partial[m][o][k] = A[m][k] * B[k][o]

matmul_reduce_i32:
  grid.x  = M
  block.x = O

  blockIdx.x  -> m
  threadIdx.x -> o

  C[m][o] = sum_k partial[m][o][k]
```

后续做 conv lowering 时，会先把卷积输入窗口整理成 `MK`，把卷积权重整理
成 `KO`，再直接复用这套 matmul。
