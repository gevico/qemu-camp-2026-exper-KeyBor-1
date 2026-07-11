# GPGPU ABI v0

本文档定义当前 built-in GPGPU core 的第一版最小 ABI。目标是先跑通手写 RV32 kernel 和最小 runtime；后续进入 Linux driver、command processor 和更完整软件栈时，可以在此基础上演进。

## 1. 分层

当前 ABI 分为两层：

- **Launch ABI**：runtime/driver 如何告诉 GPU 运行哪个 kernel、参数在哪里、执行规模是多少。
- **Kernel Entry ABI**：dispatch 后，GPU 执行逻辑在 kernel 第一条指令执行前，为每个 lane 准备怎样的初始执行环境。

Launch ABI 是软件和 GPU 控制面之间的协议。第一版通过 MMIO 寄存器表达，后续可以演进为 command packet + doorbell + command processor。

Kernel Entry ABI 是 GPU 执行环境对 kernel 代码的承诺。它描述 kernel 启动时 PC、寄存器、CSR 和内存地址的初始语义。

## 2. Launch ABI

第一版 launch 由 runtime/driver 写控制寄存器完成。

### 2.1 Kernel Code

`KERNEL_ADDR` 是 kernel code 在 VRAM 中的起始 offset。

```text
GPGPU_REG_KERNEL_ADDR_LO
GPGPU_REG_KERNEL_ADDR_HI
```

第一版约定：

```text
kernel_addr = uint32_t VRAM offset
```

虽然寄存器是 64-bit 拆分形式，当前 core interpreter 只执行 32-bit VRAM offset 地址空间。

### 2.2 Kernel Args

`KERNEL_ARGS` 是 kernel 参数区在 VRAM 中的起始 offset。

```text
GPGPU_REG_KERNEL_ARGS_LO
GPGPU_REG_KERNEL_ARGS_HI
```

第一版约定：

```text
kernel_args = uint32_t VRAM offset
```

### 2.3 Execution Shape

执行规模由 grid 和 block 描述：

```text
GPGPU_REG_GRID_DIM_X
GPGPU_REG_GRID_DIM_Y
GPGPU_REG_GRID_DIM_Z

GPGPU_REG_BLOCK_DIM_X
GPGPU_REG_BLOCK_DIM_Y
GPGPU_REG_BLOCK_DIM_Z
```

含义：

```text
grid_dim  = block 数量
block_dim = 每个 block 中的 thread 数量
```

第一版 runtime 和 core 都按 3D 字段传递和拆分。1D kernel 只是 3D 的特例：

```text
grid_dim  = { grid_x, 1, 1 }
block_dim = { block_x, 1, 1 }
```

### 2.4 Dispatch

runtime/driver 写：

```text
GPGPU_REG_DISPATCH
```

触发执行。

第一版 launch 顺序：

```text
1. 上传 kernel code 到 VRAM，得到 kernel_addr
2. 打包 kernel args 到 VRAM，得到 kernel_args
3. 写 KERNEL_ADDR
4. 写 KERNEL_ARGS
5. 写 GRID_DIM
6. 写 BLOCK_DIM
7. 写 DISPATCH
```

## 3. Kernel Entry ABI

dispatch 后，GPU 执行逻辑为每个 lane 初始化执行环境。

### 3.1 Initial PC

每个 lane 的初始 PC：

```text
pc = kernel_addr
```

也就是 launch ABI 中传入的 `KERNEL_ADDR`。

### 3.2 Initial Registers

第一版规定：

```text
x0  = 0
x2  = 当前逻辑 thread 的私有栈顶
x10 = kernel_args
其他 GPR 初始为 0
FPR 初始为 0
```

`x10` 采用 RISC-V ABI 中 `a0` 的编号，用作 kernel 用户参数区入口。
它直接指向 `user_args[0]`，不包含 launch metadata header。

`x2` 采用 RISC-V ABI 中 `sp` 的编号。当前模拟器在每次 dispatch 时从
VRAM 顶部向下预留一段 per-thread stack 区：

```text
total_threads = gridDim.x * gridDim.y * gridDim.z *
                blockDim.x * blockDim.y * blockDim.z

stack_total_size = total_threads * GPGPU_STACK_SIZE_PER_THREAD
stack_base       = vram_size - stack_total_size
```

每个逻辑 thread 获得一个固定大小 stack slot：

```text
global_thread_id = block_linear_id * block_size + thread_linear_id_in_block
sp = stack_base + (global_thread_id + 1) * GPGPU_STACK_SIZE_PER_THREAD
```

RISC-V 栈向低地址增长，所以 `sp` 初始化到该 thread slot 的顶部。
第一版 `GPGPU_STACK_SIZE_PER_THREAD = 4096`。这不是模拟真实 GPU 的物理
栈分配策略，而是给 C 编译器生成的 RISC-V 栈访问提供一个 thread-private
local memory 语义。

kernel 可以从 `x10` 读取用户参数：

```asm
lw x11, 0(x10)      # user_args[0]
lw x12, 4(x10)      # user_args[1]
```

launch metadata 由 `gpgpu_launch()` 通过 BAR0 控制寄存器写入设备状态。
kernel 需要 `threadIdx`、`blockIdx`、`blockDim`、`gridDim` 时，通过
`GPGPU_CORE_CTRL_BASE` 只读 builtin register 地址空间读取。

### 3.3 Thread Identity

每个 lane 的 `mhartid` CSR 表示当前执行身份。

第一版编码：

```text
bits [31:13] block id
bits [12:5]  warp id
bits [4:0]   lane id
```

kernel 可以通过 CSR 指令读取：

```asm
csrrs x5, mhartid, x0
```

当前实现中，`mhartid` 低 5 bit 保存 lane id，只能表达当前 warp 内的
`0..31`。完整 3D `threadIdx` 应通过 builtin register 读取。

### 3.4 Kernel Exit

第一版规定：

```text
ebreak
```

结束当前 lane 的执行。所有 active lane 结束后，warp 完成。

## 4. Kernel Args Layout

第一版参数区只包含 32-bit user args：

```text
kernel_args -> user_args[0]
user_args[i] 位于 kernel_args + i * 4
GPU pointer 是 uint32_t VRAM offset
```

示例：

```text
user_args[0] = output_ptr
user_args[1] = value
```

VRAM 中的布局：

```text
kernel_args + 0x00: output_ptr
kernel_args + 0x04: value
```

kernel 读取：

```asm
lw x11, 0(x10)     # output_ptr
lw x12, 4(x10)     # value
```

不同 kernel 可以定义不同的参数含义。ABI 只规定参数如何排列，不规定每个参数的业务语义。

例如 vector add：

```text
user_args[0] = a_ptr
user_args[1] = b_ptr
user_args[2] = c_ptr
user_args[3] = n
```

例如 tiny conv：

```text
user_args[0] = input_ptr
user_args[1] = weight_ptr
user_args[2] = output_ptr
user_args[3] = width
user_args[4] = height
```

对于 CNN-style op kernel，`kernel_args` 通常指向一个 op-specific args
结构体，而不是简单的 word array。例如 conv2d：

```text
kernel_args -> GPGPUConv2DArgs
```

`GPGPUConv2DArgs` 中包含 input/weight/bias/output 的
`GPGPUTensorDesc`。每个 tensor desc 的 `data` 字段是 VRAM offset，真正的
tensor/weight/bias 数据放在独立 VRAM allocation 中。

例如 ReLU：

```text
kernel_args -> GPGPUReluArgs
GPGPUReluArgs.input.data  -> input tensor VRAM allocation
GPGPUReluArgs.output.data -> output tensor VRAM allocation
```

例如 Linear：

```text
kernel_args -> GPGPULinearArgs
GPGPULinearArgs.input.data  -> 1D input tensor, length in_features
GPGPULinearArgs.weight.data -> OI weight tensor, [out_features, in_features]
GPGPULinearArgs.bias.data   -> 1D bias tensor, length out_features
GPGPULinearArgs.output.data -> 1D output tensor, length out_features
```

并行 Linear 第一版拆成两个 kernel：

```text
kernel_args -> GPGPULinearPartialArgs
GPGPULinearPartialArgs.input.data   -> 1D input tensor
GPGPULinearPartialArgs.weight.data  -> OI weight tensor
GPGPULinearPartialArgs.partial.data -> OI partial tensor

kernel_args -> GPGPULinearReduceArgs
GPGPULinearReduceArgs.partial.data -> OI partial tensor
GPGPULinearReduceArgs.bias.data    -> 1D bias tensor
GPGPULinearReduceArgs.output.data  -> 1D output tensor
```

`linear_partial_i32` 使用：

```text
grid.x  = out_features
block.x = in_features
```

其中 `blockIdx.x` 表示输出行，`threadIdx.x` 表示输入列。

矩阵乘使用三个矩阵 layout：

```text
A = MK = [M, K]
B = KO = [K, O]
C = MO = [M, O]
```

计算公式：

```text
C[m][o] = sum_k A[m][k] * B[k][o]
```

第一版 matmul 也拆成两个 kernel：

```text
kernel_args -> GPGPUMatmulPartialArgs
GPGPUMatmulPartialArgs.a.data       -> MK matrix
GPGPUMatmulPartialArgs.b.data       -> KO matrix
GPGPUMatmulPartialArgs.partial.data -> flat partial scratch

kernel_args -> GPGPUMatmulReduceArgs
GPGPUMatmulReduceArgs.partial.data -> flat partial scratch
GPGPUMatmulReduceArgs.c.data       -> MO matrix
GPGPUMatmulReduceArgs.bias.data    -> optional 1D bias
```

`matmul_reduce_i32` computes:

```text
acc = sum_k partial[(m * O + o) * K + k]
if output_shift != 0:
  acc = acc >> output_shift
if has_bias != 0:
  acc = acc + bias[o]
C[m][o] = acc
```

Raw integer smoke tests use `output_shift = 0` and `has_bias = 0`.
Q8.8 inference typically uses `output_shift = 8` and Q8.8 bias.

`matmul_partial_i32` 使用：

```text
grid.x  = M
grid.y  = O
block.x = K
```

其中 `blockIdx.x` 表示 A/C 的行 `m`，`blockIdx.y` 表示 B/C 的列 `o`，
`threadIdx.x` 表示归约维度 `k`。

当前 matmul kernel 假设矩阵都是连续行主序：

```text
A[m][k] -> m * K + k
B[k][o] -> k * O + o
C[m][o] -> m * O + o
```

`GPGPUTensorDesc` 中的 stride 字段保留给后续 view/transpose/padding
支持，第一版 matmul 不依赖 stride 计算 offset。

第一版 lowered conv2d 不使用 direct conv kernel，而是拆成：

```text
im2col_i32:
  input NCHW -> A MK

oihw_to_ko_i32:
  weight OIHW -> B KO

matmul_partial_i32 + matmul_reduce_i32:
  A MK * B KO -> C MO

mo_to_nchw_i32:
  C MO -> output NCHW
```

其中：

```text
M = N * out_h * out_w
K = in_channels * kernel_h * kernel_w
O = out_channels
```

`im2col_i32` 支持 zero padding：

```text
ih = oh * stride_h + kh - pad_h
iw = ow * stride_w + kw - pad_w
```

当 `(ih, iw)` 落在输入边界外时，对应 lowered matrix 元素写 0。

`mo_to_nchw_i32` 用于把 lowered conv 的 matmul 输出接回 NCHW 后续算子。
matmul 输出的内存顺序是：

```text
MO[m][oc], m = n * H * W + h * W + w
```

转换后写入：

```text
NCHW[n][oc][h][w]
```

第一版 maxpool2d 使用 direct kernel：

```text
kernel_args -> GPGPUMaxPool2DArgs
```

每个 thread 计算一个 NCHW output element，并在对应池化窗口内串行取最大值。

device kernel 的类型解释由 `kernel_addr` 决定：ReLU kernel 把 `x10`
解释为 `GPGPUReluArgs *`，Linear kernel 把 `x10` 解释为
`GPGPULinearArgs *` / `GPGPULinearPartialArgs *` /
`GPGPULinearReduceArgs *`，Matmul kernel 把 `x10` 解释为
`GPGPUMatmulPartialArgs *` / `GPGPUMatmulReduceArgs *`，Im2Col kernel
把 `x10` 解释为 `GPGPUIm2ColArgs *`，OIHW-to-KO kernel 把 `x10` 解释为
`GPGPUOihwToKoArgs *`，MO-to-NCHW kernel 把 `x10` 解释为
`GPGPUMoToNchwArgs *`，MaxPool kernel 把 `x10` 解释为
`GPGPUMaxPool2DArgs *`。第一版不在 args struct 中加入统一的 op type 或
magic header。

第一版约定：

```text
activation layout = NCHW
conv weight layout = OIHW
linear weight layout = OI
matmul layout = MK / KO / MO
dtype = GPGPU_DTYPE_I32
stride 单位 = element，不是 byte
network descriptor = host-side execution plan，不放入 VRAM
```

## 5. Address Model

第一版 device 地址模型：

```text
GPU pointer = VRAM offset
```

`LW` / `SW` 使用寄存器中的地址作为设备地址：

```text
addr = gpr[rs1] + imm
```

地址落在 builtin register 空间时，`LW` 读取设备内部执行上下文：

```text
GPGPU_CORE_CTRL_BASE + 0x00: threadIdx.x
GPGPU_CORE_CTRL_BASE + 0x04: threadIdx.y
GPGPU_CORE_CTRL_BASE + 0x08: threadIdx.z
GPGPU_CORE_CTRL_BASE + 0x10: blockIdx.x
GPGPU_CORE_CTRL_BASE + 0x14: blockIdx.y
GPGPU_CORE_CTRL_BASE + 0x18: blockIdx.z
GPGPU_CORE_CTRL_BASE + 0x20: blockDim.x
GPGPU_CORE_CTRL_BASE + 0x24: blockDim.y
GPGPU_CORE_CTRL_BASE + 0x28: blockDim.z
GPGPU_CORE_CTRL_BASE + 0x30: gridDim.x
GPGPU_CORE_CTRL_BASE + 0x34: gridDim.y
GPGPU_CORE_CTRL_BASE + 0x38: gridDim.z
```

否则地址按 VRAM offset 访问：

```text
VRAM[addr]
```

当前限制：

- 只支持 flat VRAM offset。
- `kernel_addr`、`kernel_args`、tensor pointer 均应位于 VRAM offset 范围内。

## 6. Minimal Runtime Contract

第一版 runtime 可以提供：

```c
uint32_t gpgpu_malloc(size_t size);
void gpgpu_write(uint32_t dst, const void *src, size_t size);
void gpgpu_read(uint32_t src, void *dst, size_t size);
uint32_t gpgpu_upload_kernel(const void *code, size_t size);
uint32_t gpgpu_upload_args(const void *args, size_t size);
void gpgpu_launch(uint32_t kernel_addr,
                  uint32_t kernel_args,
                  dim3 grid,
                  dim3 block);
```

其中：

```text
gpgpu_upload_kernel() 使用 gpgpu_malloc() 分配 VRAM 并写入 kernel code
gpgpu_upload_args() 使用 gpgpu_malloc() 分配 VRAM 并写入参数 blob
gpgpu_launch() 按 Launch ABI 写 MMIO 寄存器并触发 DISPATCH
```

## 7. Future Command Processor

后续加入 command processor 后，Launch ABI 可以从 MMIO register 写入演进为 command packet。

第一版 MMIO launch：

```text
runtime -> MMIO registers -> dispatch
```

后续 CP launch：

```text
runtime -> command packet in VRAM/ring -> doorbell -> command processor -> dispatch
```

两者最终都应配置同样的设备内部状态：

```text
kernel_addr
kernel_args
grid_dim
block_dim
```

因此 command processor 的加入不应改变 Kernel Entry ABI。kernel 仍然可以依赖：

```text
pc = kernel_addr
x10 = kernel_args
mhartid = lane identity
```

## 8. Open Items

后续需要补充：

- 继续补齐除 `mul` 外的 RV32M 指令，以及 FLW/FSW 等浮点 load/store。
- 定义 `max_cycles` 超时行为和错误传播。
