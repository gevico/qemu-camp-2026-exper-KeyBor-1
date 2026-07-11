# GPGPU Core Refactor Notes

本文档记录本轮对 built-in GPGPU core 的基础重构。目标不是扩展完整指令集，而是先降低解释器的技术债，让后续类 CUDA 软件栈、kernel ABI 和更多 RISC-V 指令支持更容易维护。

## 1. 寄存器表示

改进前，lane 的通用寄存器和浮点寄存器都是 `uint32_t` 数组。整数、浮点和 bit pattern 搬运需要在执行函数里临时构造 union，导致代码重复且语义不直观。

改进后，引入 `GPURegister`：

```c
typedef union GPURegister {
    uint32_t u32;
    int32_t i32;
    float f32;
} GPURegister;
```

`GPGPULane` 中的 `gpr[]` 和 `fpr[]` 改为 `GPURegister`。执行代码可以按语义选择 `.u32`、`.i32` 或 `.f32` 视图。例如：

```c
lane->gpr[rd].u32 = lane->gpr[rs1].u32 + imm;
lane->gpr[rd].i32 = safe_fcvt_w_s(f_rs1, rtz);
lane->fpr[rd].f32 = f_rs1 + f_rs2;
```

低精度格式 BF16/FP8/FP4 暂时仍使用 `.u32` 加 mask 访问，而不是 C bitfield。原因是 bitfield 的位布局受编译器、ABI 和 endian 影响，不适合作为硬件寄存器格式的稳定定义。

收益：

- 寄存器访问语义更明确。
- `fmv.w.x` / `fmv.x.w` 可以直接搬运 `.u32` bit pattern。
- 后续增加整数和浮点指令时样板代码更少。

代价：

- 所有寄存器访问都必须显式选择视图，机械替换量较大。

## 2. RISC-V Decode Context

改进前，每个 `exec_*()` 函数都重复从 `instr` 里解析 `rd`、`rs1`、`rs2`、`funct3`、`funct7` 和 immediate。随着 `LW`、branch、jump 等指令增加，这种重复会迅速变成维护负担。

改进后，新增 `arch/riscv/gpgpu_riscv.h`，集中定义：

```c
typedef struct GPGPURiscVDecode {
    uint32_t instr;
    uint32_t opcode;
    uint32_t rd;
    uint32_t rs1;
    uint32_t rs2;
    uint32_t rs3;
    uint32_t funct2;
    uint32_t funct3;
    uint32_t funct7;
    uint32_t csr;
    int32_t imm;
} GPGPURiscVDecode;
```

并提供 RISC-V immediate helper：

```c
gpgpu_riscv_imm_i()
gpgpu_riscv_imm_s()
gpgpu_riscv_imm_b()
gpgpu_riscv_imm_u()
gpgpu_riscv_imm_j()
```

现在解释入口先做 common decode，再按 opcode 选择 immediate 类型，最后把 decode context 传给执行函数。

收益：

- 字段解析集中在 arch 相关文件里。
- 执行函数更接近“指令语义”，不再混入重复 decode。
- 后续补 `BEQ/JAL/JALR/AUIPC/LW` 等指令时更清晰。
- 为未来 warp-level “decode once, execute active lanes” 的结构留出空间。

代价：

- 执行函数签名从 `instr` 改为 `GPGPURiscVDecode *`，短期 diff 较大。

## 3. Warp 切分和 Active Mask

改进前，warp 遍历条件把 active mask 放在 `for` 循环继续条件中：

```c
for (int i = 0; i < 32 && (warp->active_mask & (1 << i)); ++i)
```

如果中间 lane inactive，后续 active lane 会被跳过。现在改为遍历固定 lane 范围，并在循环内部判断 mask：

```c
for (int i = 0; i < 32; ++i) {
    if (!(warp->active_mask & (1u << i))) {
        continue;
    }
}
```

同时，block 内 warp 切分改为显式计算：

```c
thread_id_base = m * warp_size;
remaining = block_size - thread_id_base;
thread_num = MIN(warp_size, remaining);
```

收益：

- 支持非连续 active mask。
- 最后一个不满 warp 的线程数正确。
- `thread_id_base` 语义明确，后续计算 thread id 更方便。

## 4. VRAM 访问封装和 LW

改进前，解释器直接解引用 `s->vram_ptr + addr` 取指或执行 `SW`。这种写法绕过了统一的边界检查和 endian 处理。

改进后，新增 core 侧 helper：

```c
gpgpu_core_vram_check()
gpgpu_core_vram_load_u32()
gpgpu_core_vram_store_u32()
```

取指、`SW` 和新增的 `LW` 都通过 helper 访问 VRAM。访问越界时设置：

```c
GPGPU_ERR_VRAM_FAULT
GPGPU_STATUS_ERROR
```

并使用 QEMU little-endian helper：

```c
ldl_le_p()
stl_le_p()
```

收益：

- 取指和 load/store 访问路径一致。
- 避免 host endian 和非对齐直接解引用问题。
- `LW` 已可从 VRAM offset 读取 32-bit word 到寄存器，为后续 kernel args 和 vector add 做准备。

当前限制：

- 只支持 `LW` / `SW` 的 32-bit word 访问。
- 地址空间暂时仍是 flat VRAM offset，没有实现 core control register load/store 分发。
- `max_cycles`、branch/jump PC 更新、完整异常传播仍待后续处理。

## 5. 后续建议

下一步可以围绕最小类 CUDA 执行链路推进：

1. 补 device address map，把 VRAM 和 `GPGPU_CORE_CTRL_BASE` builtin register 区分开。
2. 补 `ADDI/LW/SW/LUI/ADD/SLLI` 能跑的最小 kernel ABI。
3. 定义 kernel args layout，先跑通手写 RV32 vector add。
4. 再补 branch/jump、`max_cycles` 和基础 smoke test。
