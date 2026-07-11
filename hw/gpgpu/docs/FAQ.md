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

当前 matmul kernel 就按这些连续行主序公式计算 offset，没有使用
`GPGPUTensorDesc.stride_*`。stride 字段先保留给后续更复杂的
view/transpose/padding。

当前阶段只要规定矩阵必须是连续行主序，就可以只用 `M/K/O` 推算 offset。
不能只用 `M/K/O` 的情况，是逻辑矩阵形状和物理内存跨度不一致：

```text
1. Padding / 对齐

逻辑上每行 K=3：
[1, 2, 3]
[4, 5, 6]

物理上为了对齐，每行占 4 个 slot：
[1, 2, 3, pad]
[4, 5, 6, pad]

这时下一行起点是 m * 4，不是 m * 3。
```

```text
2. Slice / view

从大矩阵 big[10][10] 中取一个逻辑上 [3,4] 的小矩阵。
小矩阵每行只有 4 个有效元素，但下一行在物理内存里仍然跨过 big 的 10 列。
```

```text
3. Transpose view

逻辑上把 B[k][o] 看成 B_T[o][k]，但不真的复制和重排内存。
这时逻辑下标移动和物理内存移动不再等于 row * cols + col。
```

这些情况才需要 stride：

```text
offset = row * stride_row + col * stride_col
```

在做 padding、slice、transpose view 之前，matmul kernel 保持简单：

```text
A[m][k] -> m * K + k
B[k][o] -> k * O + o
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

## 21. Conv2D 为什么可以拆成 im2col、OIHW-to-KO 和 matmul？

直接卷积公式是：

```text
output[n][oc][oh][ow] =
  sum over ic, kh, kw:
    input[n][ic][oh + kh][ow + kw] * weight[oc][ic][kh][kw]
```

矩阵乘公式是：

```text
C[m][o] = sum_k A[m][k] * B[k][o]
```

所以 lowering 的目标是把卷积里的三个归约维度：

```text
ic, kh, kw
```

压平成一个矩阵乘维度：

```text
k = ic * kernel_h * kernel_w + kh * kernel_w + kw
```

第一步 `im2col_i32`：

```text
input NCHW -> A MK
M = N * out_h * out_w
K = in_channels * kernel_h * kernel_w
```

也就是把每个卷积窗口拉平成 A 的一行。

第二步 `oihw_to_ko_i32`：

```text
weight OIHW -> B KO
O = out_channels
K = in_channels * kernel_h * kernel_w
```

也就是把每个卷积核按同样的 `k` 顺序拉平，放成矩阵 B 的列。

第三步复用 matmul：

```text
A MK * B KO = C MO
```

第四步解释输出：

```text
C[m][o] 对应 output[n][oc][oh][ow]
m = n * out_h * out_w + oh * out_w + ow
o = oc
```

当前 smoke 使用 `1x1x4x4` 输入和 `1x1x3x3` 卷积核：

```text
NCHW input  = [1, 1, 4, 4]
OIHW weight = [1, 1, 3, 3]
im2col A    = [4, 9]
KO weight   = [9, 1]
matmul C    = [4, 1]
```

第一版 lowered conv 不加 bias；后续可以增加一个 bias/add kernel，或在
matmul reduce 阶段扩展 bias 输入。

## 22. MaxPool2D 为什么可以直接写 kernel？

MaxPool 没有类似卷积/矩阵乘的归约维度重排需求。每个输出元素只需要在
对应窗口里取最大值：

```text
output[n][c][oh][ow] =
  max input[n][c][oh * stride_h + kh][ow * stride_w + kw]
```

所以第一版 `maxpool_i32` 直接让一个 thread 负责一个 output element，
在窗口内部串行扫描。

## 23. 真实 GPU 有栈吗？为什么我们的 kernel 之前不能用栈？

真实 GPU 不是简单地“没有栈”。更准确的说法是：GPU 编程模型给每个
thread 提供 private/local storage 语义。局部变量通常优先放进寄存器；
当寄存器不够、局部数组太大、发生 spill、函数调用需要调用帧时，编译器和
runtime/硬件会把这些内容映射到该 thread 私有的 local memory。

这不等于每个 lane 永久绑定一大块物理 SRAM，也不等于每次执行都像 CPU
程序一样 `malloc/free` 一段栈。真实实现通常由编译器 metadata、runtime
launch 配置、寄存器文件、local memory 和缓存共同完成。

我们之前不能用栈，是因为 device kernel 用 `riscv64-unknown-elf-gcc`
生成 RISC-V 代码。只要编译器生成：

```asm
addi sp, sp, -16
sw ra, 12(sp)
sw s0, 8(sp)
```

解释器就会按普通 RISC-V 语义访问 `x2/sp` 指向的地址。但当时每个 lane 的
`sp` 初始值是 0，没有合法的 thread-private 栈区，所以写栈会落到非法或
错误的 VRAM 地址。

当前采用的简化模型是：

```text
每次 kernel dispatch:
  total_threads = gridDim * blockDim
  在 VRAM 顶部向下预留 total_threads * 64B
  每个逻辑 thread 分配一个固定 64B stack slot
  lane 初始化时把 x2/sp 设置到自己的 stack slot 顶部
```

也就是：

```text
global_thread_id = block_linear_id * block_size + thread_id_in_block
sp = stack_base + (global_thread_id + 1) * 64
```

这样做的好处是语义清楚：每个逻辑 thread 都有自己的 private stack，
warp/lane 调度顺序不会导致局部变量互相覆盖。代价是内存开销较大，并且
runtime 的普通 VRAM 分配目前还不知道这段 launch-time stack 区，后续要
演进成更正式的 VRAM memory layout 或 command metadata。

最初曾使用 4KB stack slot，适合教学但不适合 LeNet 这类有较多逻辑 thread
的裸机 smoke。例如 conv2 lowering 的 partial matmul 可能有 240000 个逻辑
thread，4KB/thread 会接近 1GB。当前 device kernels 的实际栈帧很小，
所以先收敛到 64B/thread；后续更真实的做法是引入 kernel metadata 中的
stack frame size，或者只为驻留 warp 分配 local stack。

## 24. RISC-V 编译器生成的 `sp` 为什么能和 lane 的栈连起来？

RISC-V 汇编里的 `sp`、`a0`、`a5` 这些名字不是额外的硬件对象，而是通用
寄存器编号的 ABI 别名：

```text
sp = x2
a0 = x10
a5 = x15
```

编译器看到 C 局部变量需要放到栈上时，可能生成：

```asm
addi sp, sp, -16
sw a3, 0(sp)
lw a5, 0(sp)
```

但机器码里不会保存字符串 `"sp"`，只会保存寄存器编号。比如：

```asm
sw a3, 0(sp)
```

解码后核心信息是：

```text
rs1 = 2     // base register = sp/x2
rs2 = 13    // source register = a3/x13
imm = 0
```

我们的解释器执行 store 时使用这些编号访问当前 lane 的寄存器数组：

```c
addr = lane->gpr[dec->rs1].u32 + dec->imm;
gpgpu_core_vram_store_u32(s, addr, lane->gpr[dec->rs2].u32);
```

所以当 `dec->rs1 == 2` 时，本质就是：

```c
addr = lane->gpr[2].u32 + dec->imm;
```

而 `gpgpu_core_init_warp()` 已经在 lane 入口处初始化：

```c
lane->gpr[2].u32 = stack_top;    // sp/x2
lane->gpr[10].u32 = kernel_args; // a0/x10
```

因此完整链路是：

```text
C 局部变量
  -> 编译器生成 sw/lw ...(sp)
  -> sp 是 RISC-V ABI 对 x2 的名字
  -> 机器码里 rs1 = 2
  -> 解释器访问 lane->gpr[2]
  -> lane->gpr[2] 是 dispatch 时设置的 per-thread stack top
  -> 最终读写 VRAM 中该 thread 的私有栈区域
```

所以你的理解是对的：**通用寄存器在机器码和解释器里都是通过寄存器编号访问的**。
汇编里的名字主要服务于人类阅读和 ABI 约定；真正执行时使用的是寄存器编号。

这里要避免把“编号”和“偏移”混在一起：

```text
x2 的 2        = 寄存器编号
lane->gpr[2]   = 用寄存器编号作为数组下标访问模拟寄存器
0(sp) 的 0     = 相对 sp/x2 的地址偏移
```

所以更准确的说法是：RISC-V 指令编码使用 `rs1/rs2/rd` 寄存器编号；
我们的解释器再把这个编号作为 `gpr[]` 下标。`lw/sw` 里的 immediate
才是内存地址偏移。

## 25. Q8.8 定点量化在当前推理链路里怎么理解？

第一版 LeNet 推理先采用 per-tensor 的 Q8.8 定点格式：

```text
real_value ~= int32_value / 256
int32_value = round(real_value * 256)
```

输入像素、权重、bias、activation 都用 `int32` 保存，但数值含义是 Q8.8。
这样做的原因是当前 device kernel 已经支持 `int32` load/store 和乘加，
不用先补完整浮点路径。

这里的“量化”不是为了训练模型，而是为了让 PyTorch 里的 float32 权重和
输入能在当前裸机 GPGPU kernel 上执行。PyTorch 原始推理大概是：

```text
float input
float weight
float bias
float output
```

而当前 RV32 device kernel 路径主要支持：

```text
int32 load/store
int32 multiply
int32 accumulate
```

所以需要把 float 数值映射到整数域。Q8.8 的含义是用低 8 位表示小数：

```text
1.0   -> 256
0.5   -> 128
-0.25 -> -64
```

这样我们仍然可以用整数乘法模拟小数乘法。

两个 Q8.8 数相乘时：

```text
(a_real * 256) * (b_real * 256) = a_real * b_real * 65536
```

乘积变成 Q16.16。多个乘积累加后仍然是 Q16.16，所以输出要回到 Q8.8：

```text
acc_q8_8 = acc_q16_16 >> 8
```

这就是 `GPGPUMatmulReduceArgs.output_shift = 8` 的含义。raw i32 smoke
不做定点缩放，所以 `output_shift = 0`。

Bias 需要和输出 activation 使用同一 scale。对于当前 Q8.8 方案：

```text
bias_q8_8 = round(bias_float * 256)
output_q8_8 = (sum(input_q8_8 * weight_q8_8) >> 8) + bias_q8_8
```

这个方案不是最高精度的量化方案，只是最容易和当前 RV32 int kernel 对齐。
后续如果要提高精度，可以演进到 per-channel weight scale、rounding shift、
saturation/clamp，以及真正的 int8 storage。

当前实现采用非常直接的量化流程：

```text
float32 weight/bias/input
  -> q = round(float32 * 256)
  -> 存成 int32_t
  -> kernel 中按 int32_t 做乘加
  -> matmul reduce 阶段右移 8 位回到 Q8.8
```

需要注意两个限制：

- 现在没有做 saturation，极端值可能溢出。
- 现在没有做 rounding shift，`acc >> 8` 是直接截断。

对 MNIST LeNet 这种小模型，第一版先追求链路跑通和语义清楚。

## 26. SunnyHaze LeNet 的 PyTorch 权重是如何提取和使用的？

SunnyHaze 仓库里的模型结构是固定的：

```text
layer1.0 = Conv2d(1, 6, kernel=5, padding=2)
layer1.3 = Conv2d(6, 16, kernel=5)
layer2.0 = Linear(16 * 5 * 5, 120)
layer2.2 = Linear(120, 84)
layer2.4 = Linear(84, 10)
```

PyTorch 保存的 `model.pt` 里有一个 `state_dict`。从逻辑上看，
`state_dict` 就是“参数名 -> tensor”的映射：

```text
layer1.0.weight -> [6, 1, 5, 5]
layer1.0.bias   -> [6]
layer1.3.weight -> [16, 6, 5, 5]
layer1.3.bias   -> [16]
layer2.0.weight -> [120, 400]
layer2.0.bias   -> [120]
layer2.2.weight -> [84, 120]
layer2.2.bias   -> [84]
layer2.4.weight -> [10, 84]
layer2.4.bias   -> [10]
```

PyTorch 正常加载时会按名字把这些 tensor 放回对应 layer：

```python
model.load_state_dict(state_dict)
```

我们裸机环境没有 PyTorch，也不需要 PyTorch 的模块对象。我们的导出脚本
`tools/export_sunnyhaze_lenet.py` 直接读取 `model.pt` zip 里的 raw
float32 storage。这个模型的 storage 顺序和 tensor 形状固定，对应关系是：

```text
archive/data/0 -> layer1.0.weight -> conv1_weight
archive/data/1 -> layer1.0.bias   -> conv1_bias
archive/data/2 -> layer1.3.weight -> conv2_weight
archive/data/3 -> layer1.3.bias   -> conv2_bias
archive/data/4 -> layer2.0.weight -> fc1_weight
archive/data/5 -> layer2.0.bias   -> fc1_bias
archive/data/6 -> layer2.2.weight -> fc2_weight
archive/data/7 -> layer2.2.bias   -> fc2_bias
archive/data/8 -> layer2.4.weight -> fc3_weight
archive/data/9 -> layer2.4.bias   -> fc3_bias
```

导出时做两件事：

```text
1. 按 tensor 的 shape 读取连续 float32 数组
2. 对每个 float 做 Q8.8 量化，写成 C 头文件里的 int32_t 数组
```

生成的文件是：

```text
assets/lenet/sunnyhaze_lenet_q8_weights.h
```

后续裸机加载时，不是按 PyTorch 的 layer object 加载，而是按我们自己的
runtime 规则上传到 VRAM：

```text
sunnyhaze_lenet_conv1_weight_q8 -> gpgpu_write 到 VRAM
sunnyhaze_lenet_conv1_bias_q8   -> gpgpu_write 到 VRAM
...
```

然后把 VRAM offset 填进 tensor descriptor：

```text
weight.data = conv1_weight_addr
bias.data   = conv1_bias_addr
```

kernel 计算时确实是根据维度和 index 取权重值。例如 conv weight 是 OIHW：

```text
weight[oc][ic][kh][kw]
offset = oc * (IC * KH * KW)
       + ic * (KH * KW)
       + kh * KW
       + kw
```

在 lowering 路径里，我们先用 `oihw_to_ko_i32` 把权重从 OIHW 转成 KO：

```text
k = ic * KH * KW + kh * KW + kw
o = oc
B[k][o] = weight[oc][ic][kh][kw]
```

然后 matmul 使用：

```text
C[m][o] = sum_k A[m][k] * B[k][o]
```

所以可以这样理解：

```text
PyTorch 按 layer 名字管理权重；
我们导出后按 tensor 名字和固定 shape 管理权重；
runtime 上传后按 VRAM offset 管理权重；
kernel 执行时按 layout/维度公式计算 offset，取出具体权重值参与乘加。
```
