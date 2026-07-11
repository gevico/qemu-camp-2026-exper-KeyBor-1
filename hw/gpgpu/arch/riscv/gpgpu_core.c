/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "../../gpgpu.h"
#include "gpgpu_core.h"
#include "gpgpu_riscv.h"
#include <math.h>

#define MHART_ID(block_id_linear, warp_id, thread_id) \
    (((block_id_linear & 0x7FFFF) << 13) | ((warp_id & 0xFF) << 5) | (thread_id & 0x1F))

/* TODO: Implement warp initialization */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t kernel_args,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    warp->thread_id_base = thread_id_base;
    memcpy(warp->block_id, block_id, sizeof(warp->block_id));
    warp->warp_id = warp_id;
    for (int i = 0; i < num_threads; ++i) {
        warp->lanes[i] = (GPGPULane){
            .gpr = {{0}},
            .fpr = {{0}},
            .pc = pc,
            .mhartid = MHART_ID(block_id_linear, warp_id, i),
            .fcsr = 0,
            .fp_status = {0},
            .active = 1
        };
        warp->lanes[i].gpr[10].u32 = kernel_args;
    }
}

/* TODO: Implement warp execution (RV32I + RV32F interpreter) */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    while (true) {
        if (warp->active_mask == 0) break;
        for (int i = 0; i < 32; ++i) {
            uint32_t thread_id_linear;
            uint32_t block_dim_x;
            uint32_t block_dim_y;
            uint32_t block_xy;

            if(!(warp->active_mask & (1u << i))) continue;
            thread_id_linear = warp->thread_id_base + i;
            block_dim_x = s->kernel.block_dim[0];
            block_dim_y = s->kernel.block_dim[1];
            if (block_dim_x == 0 || block_dim_y == 0) {
                s->error_status |= GPGPU_ERR_INVALID_CMD;
                s->global_status |= GPGPU_STATUS_ERROR;
                return -1;
            }
            block_xy = block_dim_x * block_dim_y;
            if (block_xy == 0) {
                s->error_status |= GPGPU_ERR_INVALID_CMD;
                s->global_status |= GPGPU_STATUS_ERROR;
                return -1;
            }
            s->simt.thread_id[0] = thread_id_linear % block_dim_x;
            s->simt.thread_id[1] = (thread_id_linear / block_dim_x) % block_dim_y;
            s->simt.thread_id[2] = thread_id_linear / block_xy;
            s->simt.warp_id = warp->warp_id;
            memcpy(s->simt.block_id, warp->block_id, sizeof(s->simt.block_id));
            s->simt.lane_id = i;
            int ret = gpgpu_core_exec_kernel(&warp->lanes[i], s);
            if (ret < 0) {
                return -1;
            }
            if (ret == 1) warp->active_mask &= ~(1u << i);
        }
    }
    return 0;
}

// --- 辅助宏与内联函数 ---
static inline bool is_nan_or_inf(float f) {
    uint32_t i = *(uint32_t*)&f;
    return (i & 0x7F800000) == 0x7F800000;
}

// --- 安全的浮点转整数 ---
static inline int32_t safe_fcvt_w_s(float f, bool rtz) {
    if (isnan(f)) return 0;
    if (f >= (float)INT32_MAX) return INT32_MAX;
    if (f <= (float)INT32_MIN) return INT32_MIN;
    return rtz ? (int32_t)f : (int32_t)nearbyintf(f);
}

// --- 浮点寄存器访问 ---
static inline float fpr_get_f32(GPGPULane *lane, int idx)
{
    return lane->fpr[idx].f32;
}

static inline void fpr_set_f32(GPGPULane *lane, int idx, float val)
{
    lane->fpr[idx].f32 = val;
}

static bool gpgpu_core_vram_check(GPGPUState *s, uint32_t addr, size_t size)
{
    if (!s->vram_ptr || addr > s->vram_size || size > s->vram_size - addr) {
        s->error_status |= GPGPU_ERR_VRAM_FAULT;
        s->global_status |= GPGPU_STATUS_ERROR;
        return false;
    }

    return true;
}

static bool gpgpu_core_vram_load_u32(GPGPUState *s, uint32_t addr,
                                     uint32_t *value)
{
    if (!gpgpu_core_vram_check(s, addr, sizeof(*value))) {
        return false;
    }

    *value = ldl_le_p(s->vram_ptr + addr);
    return true;
}

static bool gpgpu_core_vram_load_u16(GPGPUState *s, uint32_t addr,
                                     uint16_t *value)
{
    if (!gpgpu_core_vram_check(s, addr, sizeof(*value))) {
        return false;
    }

    *value = lduw_le_p(s->vram_ptr + addr);
    return true;
}

static bool gpgpu_core_vram_load_u8(GPGPUState *s, uint32_t addr,
                                    uint8_t *value)
{
    if (!gpgpu_core_vram_check(s, addr, sizeof(*value))) {
        return false;
    }

    *value = ldub_p(s->vram_ptr + addr);
    return true;
}

static bool gpgpu_core_vram_store_u32(GPGPUState *s, uint32_t addr,
                                      uint32_t value)
{
    if (!gpgpu_core_vram_check(s, addr, sizeof(value))) {
        return false;
    }

    stl_le_p(s->vram_ptr + addr, value);
    return true;
}

static bool gpgpu_core_vram_store_u16(GPGPUState *s, uint32_t addr,
                                      uint16_t value)
{
    if (!gpgpu_core_vram_check(s, addr, sizeof(value))) {
        return false;
    }

    stw_le_p(s->vram_ptr + addr, value);
    return true;
}

static bool gpgpu_core_vram_store_u8(GPGPUState *s, uint32_t addr,
                                     uint8_t value)
{
    if (!gpgpu_core_vram_check(s, addr, sizeof(value))) {
        return false;
    }

    stb_p(s->vram_ptr + addr, value);
    return true;
}

// ==================== FP8/FP16 转换函数 ====================

// --- BF16 ---
static inline uint16_t fp32_to_bf16(float f) {
    uint32_t i = *(uint32_t*)&f;
    uint32_t lsb = (i >> 16) & 1;
    uint32_t rnd = lsb && ((i & 0xFFFF) || ((i >> 23) & 1));
    return (uint16_t)(((i >> 16) + rnd) & 0xFFFF);
}

static inline float bf16_to_fp32(uint16_t h) {
    uint32_t i = (uint32_t)h << 16;
    return *(float*)&i;
}

// ==================== FP8 转换函数 ====================

// --- E5M2: 1s 5e 2m, bias=15, max=57344 ---
// 编码：exp=0..30 正常值，exp=31 且 mant=0 为 Inf，exp=31 且 mant!=0 为 NaN
// 最大值：exp=30, mant=3 → (1+3/4)*2^(30-15) = 1.75*32768 = 57344
// 最大值编码 = (sign<<7) | (30<<2) | 3 = 0x7B/0xFB
static inline uint8_t fp32_to_e5m2(float f) {
    if (f == 0.0f) return 0;

    uint32_t bits = *(uint32_t*)&f;
    int sign = (bits >> 31) & 1;
    int exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    // Inf → 保持为 Inf (规范要求)
    if (isinf(f)) {
        return sign ? 0xF8 : 0x78;  // s11111_00 (Inf)
    }
    // NaN 或溢出 → 饱和到最大值
    if (isnan(f) || exp > 30) {
        return sign ? 0xFB : 0x7B;
    }
    // 下溢到 0
    if (exp < -15) {
        return sign ? 0x80 : 0x00;
    }

    int e = exp + 15;
    if (e < 0) e = 0;
    uint32_t m = (mant >> 21) & 0x3;

    // 向最近偶数舍入：保留 2 位尾数，舍入位是 mant 的第 20 位
    uint32_t round_bit = (mant >> 20) & 1;
    uint32_t sticky = mant & 0xFFFFF;
    if (round_bit && (sticky || (m & 1))) {
        m++;
        if (m > 3) { m = 0; e++; }
    }

    // 舍入后可能溢出到 e=31，饱和到最大值
    if (e >= 31) return sign ? 0xFB : 0x7B;
    return (sign << 7) | ((e & 0x1F) << 2) | m;
}

static inline float e5m2_to_fp32(uint8_t v) {
    if ((v & 0x7F) == 0) return 0.0f;
    int sign = (v >> 7) & 1;
    int exp  = (v >> 2) & 0x1F;
    int mant = v & 0x3;

    if (exp == 31) {
        if (mant == 0) {
            // Inf
            uint32_t bits = (sign << 31) | 0x7F800000;
            return *(float*)&bits;
        } else {
            // NaN
            uint32_t bits = (sign << 31) | 0x7F800000 | (mant << 21);
            return *(float*)&bits;
        }
    }
    // exp=30, mant=3 是最大值: (1+3/4)*2^(30-15) = 0.875*32768 = 57344
    uint32_t bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 21);
    return *(float*)&bits;
}

// --- E4M3: 1s 4e 3m, bias=7, max=448 ---
// E4M3 无 Inf 表示，超出范围的值饱和到 ±448
// 最大值编码：exp=15(1111), mant=6(110) → 0x7E/0xFE → 448
static inline uint8_t fp32_to_e4m3(float f) {
    if (f == 0.0f) return 0;

    uint32_t bits = *(uint32_t*)&f;
    int sign = (bits >> 31) & 1;
    int exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    // Inf/NaN/溢出 → 饱和到最大值 448 (0x7E/0xFE)
    if (isinf(f) || isnan(f) || exp > 8) {
        return sign ? 0xFE : 0x7E;
    }
    // 下溢到 0
    if (exp < -6) {
        return sign ? 0x80 : 0x00;
    }

    int e = exp + 7;
    if (e < 0) e = 0;
    uint32_t m = (mant >> 20) & 0x7;

    // 向最近偶数舍入：保留 3 位尾数，舍入位是 mant 的第 19 位
    uint32_t round_bit = (mant >> 19) & 1;
    uint32_t sticky = mant & 0x7FFFF;
    if (round_bit && (sticky || (m & 1))) {
        m++;
        if (m > 7) { m = 0; e++; }
    }

    // 检查溢出（舍入可能导致指数溢出到 e=16）
    if (e > 15) return sign ? 0xFE : 0x7E;

    return (sign << 7) | ((e & 0xF) << 3) | m;
}

static inline float e4m3_to_fp32(uint8_t v) {
    if ((v & 0x7F) == 0) return 0.0f;
    int sign = (v >> 7) & 1;
    int exp  = (v >> 3) & 0xF;
    int mant = v & 0x7;
    // E4M3 无 Inf，exp=15 表示最大值 448
    uint32_t bits = (sign << 31) | ((exp - 7 + 127) << 23) | (mant << 20);
    return *(float*)&bits;
}

// --- E2M1: 1s 2e 1m, bias=1, max=6 ---
// 4-bit 编码 = sign(1) + exp_field(2, bias=1) + mant(1)
// 可表示的正数值（无符号编码 3bit → 值）：
//   0→0, 1→0.5, 2→1.0, 3→1.5, 4→2.0, 5→3.0, 6→4.0, 7→6.0
static inline uint8_t fp32_to_e2m1(float f) {
    if (f == 0.0f) return 0;

    uint32_t bits = *(uint32_t*)&f;
    int sign = (bits >> 31) & 1;
    int exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    float abs_f = fabsf(f);

    // NaN/Inf/溢出 → 饱和到 ±6.0 (编码 0x7/0xF)
    if (isnan(f) || isinf(f) || abs_f > 6.0f) {
        return sign ? 0xF : 0x7;
    }
    // 下溢到 0
    if (abs_f < 0.25f) {
        return sign ? 0x8 : 0x0;
    }

    int e = exp + 1;  // bias = 1
    if (e < 0) e = 0;
    if (e > 3) e = 3;
    uint32_t m = (mant >> 22) & 0x1;

    // 向最近偶数舍入：保留 1 位尾数，舍入位是 mant 的第 21 位
    uint32_t round_bit = (mant >> 21) & 1;
    uint32_t sticky = mant & 0x1FFFFF;
    if (round_bit && (sticky || (m & 1))) {
        m++;
        if (m > 1) { m = 0; e++; }
    }

    // 检查溢出（舍入可能导致 e=4）
    if (e > 3) return sign ? 0xF : 0x7;

    return (sign << 3) | ((e & 0x3) << 1) | m;
}

static inline float e2m1_to_fp32(uint8_t v) {
    if ((v & 0x7) == 0) return 0.0f;
    // 查表法：3 位无符号编码到值的映射
    static const float pos_values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    int sign = (v >> 3) & 1;
    float val = pos_values[v & 0x7];
    if (sign) val = -val;
    return val;
}

// ==================== 指令执行子函数 ====================

static void exec_system(const GPGPURiscVDecode *dec, GPGPULane *lane) {
    if (dec->funct3 == 0 && dec->instr == 0x00100073) {
        lane->active = false;
        return;
    }
    if (dec->funct3 == 0x2 && dec->csr == 0xF14) {
        uint32_t old_val = lane->mhartid;
        if (dec->rd != 0) lane->gpr[dec->rd].u32 = old_val;
    }
}

static void exec_op_imm(const GPGPURiscVDecode *dec, GPGPULane *lane) {
    uint32_t rs1 = lane->gpr[dec->rs1].u32;
    uint32_t shamt = dec->imm & 0x1f;

    switch (dec->funct3) {
        case 0x0: lane->gpr[dec->rd].u32 = rs1 + dec->imm; break;          // ADDI
        case 0x1: lane->gpr[dec->rd].u32 = rs1 << shamt; break;            // SLLI
        case 0x2: lane->gpr[dec->rd].u32 = (int32_t)rs1 < dec->imm; break; // SLTI
        case 0x3: lane->gpr[dec->rd].u32 = rs1 < (uint32_t)dec->imm; break; // SLTIU
        case 0x4: lane->gpr[dec->rd].u32 = rs1 ^ dec->imm; break;          // XORI
        case 0x5:
            if ((dec->funct7 & 0x20) != 0) {
                lane->gpr[dec->rd].u32 = (uint32_t)((int32_t)rs1 >> shamt); // SRAI
            } else {
                lane->gpr[dec->rd].u32 = rs1 >> shamt;                    // SRLI
            }
            break;
        case 0x6: lane->gpr[dec->rd].u32 = rs1 | dec->imm; break;          // ORI
        case 0x7: lane->gpr[dec->rd].u32 = rs1 & dec->imm; break;          // ANDI
    }
}

static void exec_op(const GPGPURiscVDecode *dec, GPGPULane *lane) {
    uint32_t rs1 = lane->gpr[dec->rs1].u32;
    uint32_t rs2 = lane->gpr[dec->rs2].u32;
    uint32_t shamt = rs2 & 0x1f;

    if (dec->funct7 == 0x01) {
        switch (dec->funct3) {
        case 0x0:
            lane->gpr[dec->rd].u32 = (uint32_t)((uint64_t)rs1 * rs2); // MUL
            break;
        }
        return;
    }

    switch (dec->funct3) {
    case 0x0:
        if (dec->funct7 == 0x20) {
            lane->gpr[dec->rd].u32 = rs1 - rs2; // SUB
        } else {
            lane->gpr[dec->rd].u32 = rs1 + rs2; // ADD
        }
        break;
    case 0x1:
        lane->gpr[dec->rd].u32 = rs1 << shamt; // SLL
        break;
    case 0x2:
        lane->gpr[dec->rd].u32 = (int32_t)rs1 < (int32_t)rs2; // SLT
        break;
    case 0x3:
        lane->gpr[dec->rd].u32 = rs1 < rs2; // SLTU
        break;
    case 0x4:
        lane->gpr[dec->rd].u32 = rs1 ^ rs2; // XOR
        break;
    case 0x5:
        if (dec->funct7 == 0x20) {
            lane->gpr[dec->rd].u32 = (uint32_t)((int32_t)rs1 >> shamt); // SRA
        } else {
            lane->gpr[dec->rd].u32 = rs1 >> shamt; // SRL
        }
        break;
    case 0x6:
        lane->gpr[dec->rd].u32 = rs1 | rs2; // OR
        break;
    case 0x7:
        lane->gpr[dec->rd].u32 = rs1 & rs2; // AND
        break;
    }
}

static void exec_lui(const GPGPURiscVDecode *dec, GPGPULane *lane) {
    lane->gpr[dec->rd].u32 = dec->imm;
}

static void exec_auipc(const GPGPURiscVDecode *dec, GPGPULane *lane)
{
    lane->gpr[dec->rd].u32 = lane->pc + dec->imm;
}

static bool exec_branch(const GPGPURiscVDecode *dec, GPGPULane *lane)
{
    uint32_t rs1 = lane->gpr[dec->rs1].u32;
    uint32_t rs2 = lane->gpr[dec->rs2].u32;
    bool take = false;

    switch (dec->funct3) {
    case 0x0: take = (rs1 == rs2); break;                    // BEQ
    case 0x1: take = (rs1 != rs2); break;                    // BNE
    case 0x4: take = ((int32_t)rs1 < (int32_t)rs2); break;   // BLT
    case 0x5: take = ((int32_t)rs1 >= (int32_t)rs2); break;  // BGE
    case 0x6: take = (rs1 < rs2); break;                     // BLTU
    case 0x7: take = (rs1 >= rs2); break;                    // BGEU
    }

    if (take) {
        lane->pc += dec->imm;
        return true;
    }
    return false;
}

static bool exec_jal(const GPGPURiscVDecode *dec, GPGPULane *lane)
{
    uint32_t next_pc = lane->pc + 4;

    lane->gpr[dec->rd].u32 = next_pc;
    lane->pc += dec->imm;
    return true;
}

static bool exec_jalr(const GPGPURiscVDecode *dec, GPGPULane *lane)
{
    uint32_t next_pc = lane->pc + 4;
    uint32_t target = (lane->gpr[dec->rs1].u32 + dec->imm) & ~1u;

    lane->gpr[dec->rd].u32 = next_pc;
    lane->pc = target;
    return true;
}

static bool gpgpu_core_ctrl_load_u32(GPGPUState *s, uint32_t addr,
                                     uint32_t *value)
{
    switch (addr) {
    case GPGPU_CORE_CTRL_THREAD_ID_X:
        *value = s->simt.thread_id[0];
        return true;
    case GPGPU_CORE_CTRL_THREAD_ID_Y:
        *value = s->simt.thread_id[1];
        return true;
    case GPGPU_CORE_CTRL_THREAD_ID_Z:
        *value = s->simt.thread_id[2];
        return true;
    case GPGPU_CORE_CTRL_BLOCK_ID_X:
        *value = s->simt.block_id[0];
        return true;
    case GPGPU_CORE_CTRL_BLOCK_ID_Y:
        *value = s->simt.block_id[1];
        return true;
    case GPGPU_CORE_CTRL_BLOCK_ID_Z:
        *value = s->simt.block_id[2];
        return true;
    case GPGPU_CORE_CTRL_BLOCK_DIM_X:
        *value = s->kernel.block_dim[0];
        return true;
    case GPGPU_CORE_CTRL_BLOCK_DIM_Y:
        *value = s->kernel.block_dim[1];
        return true;
    case GPGPU_CORE_CTRL_BLOCK_DIM_Z:
        *value = s->kernel.block_dim[2];
        return true;
    case GPGPU_CORE_CTRL_GRID_DIM_X:
        *value = s->kernel.grid_dim[0];
        return true;
    case GPGPU_CORE_CTRL_GRID_DIM_Y:
        *value = s->kernel.grid_dim[1];
        return true;
    case GPGPU_CORE_CTRL_GRID_DIM_Z:
        *value = s->kernel.grid_dim[2];
        return true;
    default:
        return false;
    }
}

static void exec_load(const GPGPURiscVDecode *dec, GPGPULane *lane, GPGPUState *s) {
    uint32_t addr = lane->gpr[dec->rs1].u32 + dec->imm;
    uint32_t value = 0;
    uint16_t value16;
    uint8_t value8;
    bool loaded = false;

    if (dec->funct3 == 0x2) {
        loaded = gpgpu_core_ctrl_load_u32(s, addr, &value) ||
                 gpgpu_core_vram_load_u32(s, addr, &value); // LW
    } else {
        switch (dec->funct3) {
        case 0x0:
            loaded = gpgpu_core_vram_load_u8(s, addr, &value8); // LB
            value = (uint32_t)(int32_t)(int8_t)value8;
            break;
        case 0x1:
            loaded = gpgpu_core_vram_load_u16(s, addr, &value16); // LH
            value = (uint32_t)(int32_t)(int16_t)value16;
            break;
        case 0x4:
            loaded = gpgpu_core_vram_load_u8(s, addr, &value8); // LBU
            value = value8;
            break;
        case 0x5:
            loaded = gpgpu_core_vram_load_u16(s, addr, &value16); // LHU
            value = value16;
            break;
        }
    }

    if (loaded) {
        lane->gpr[dec->rd].u32 = value;
    }
}

static void exec_store(const GPGPURiscVDecode *dec, GPGPULane *lane, GPGPUState *s) {
    uint32_t addr = lane->gpr[dec->rs1].u32 + dec->imm;
    switch (dec->funct3) {
    case 0x0:
        gpgpu_core_vram_store_u8(s, addr, lane->gpr[dec->rs2].u32); // SB
        break;
    case 0x1:
        gpgpu_core_vram_store_u16(s, addr, lane->gpr[dec->rs2].u32); // SH
        break;
    case 0x2:
        gpgpu_core_vram_store_u32(s, addr, lane->gpr[dec->rs2].u32);
        break;
    }
}


// --- 浮点操作 (RV32F + FP8/FP16 扩展) ---
static void exec_op_fp(const GPGPURiscVDecode *dec, GPGPULane *lane)
{
    if (dec->opcode != 0x53)
        return;

    // 直通 fmv.w.x / fmv.x.w
    if (dec->funct7 == 0x78) {
        lane->fpr[dec->rd].u32 = lane->gpr[dec->rs1].u32;
        return;
    }
    if (dec->funct7 == 0x70 && dec->funct3 == 0x0) {
        lane->gpr[dec->rd].u32 = lane->fpr[dec->rs1].u32;
        return;
    }

    float f_rs1 = fpr_get_f32(lane, dec->rs1);
    float f_rs2 = fpr_get_f32(lane, dec->rs2);

    switch (dec->funct7) {

    case 0x00: // FADD.S
        fpr_set_f32(lane, dec->rd, f_rs1 + f_rs2);
        break;

    case 0x08: // FMUL.S
        fpr_set_f32(lane, dec->rd, f_rs1 * f_rs2);
        break;

    case 0x60: // FCVT.W.S
        if (dec->rs2 == 0)
            lane->gpr[dec->rd].i32 = safe_fcvt_w_s(f_rs1, dec->funct3 == 1);
        break;

    case 0x68: // FCVT.S.W
        if (dec->rs2 == 0)
            fpr_set_f32(lane, dec->rd, (float)lane->gpr[dec->rs1].i32);
        break;

    case 0x22: // BF16
        if (dec->rs2 == 0) {
            uint16_t bf16 = (uint16_t)(lane->fpr[dec->rs1].u32 & 0xFFFF);
            fpr_set_f32(lane, dec->rd, bf16_to_fp32(bf16));
        } else {
            uint16_t bf16 = fp32_to_bf16(f_rs1);
            lane->fpr[dec->rd].u32 = (lane->fpr[dec->rd].u32 & 0xFFFF0000) | bf16;
        }
        break;

    case 0x24: // E4M3 和 E5M2 (用 rs2 区分)
        if (dec->rs2 == 1) {
            // F32 → E4M3
            uint8_t v = fp32_to_e4m3(f_rs1);
            lane->fpr[dec->rd].u32 = (lane->fpr[dec->rd].u32 & 0xFFFFFF00) | v;
        } else if (dec->rs2 == 3) {
            // F32 → E5M2
            uint8_t v = fp32_to_e5m2(f_rs1);
            lane->fpr[dec->rd].u32 = (lane->fpr[dec->rd].u32 & 0xFFFFFF00) | v;
        } else if (dec->rs2 == 0) {
            // E4M3 → F32
            uint8_t v = lane->fpr[dec->rs1].u32 & 0xFF;
            fpr_set_f32(lane, dec->rd, e4m3_to_fp32(v));
        } else if (dec->rs2 == 2) {
            // E5M2 → F32
            uint8_t v = lane->fpr[dec->rs1].u32 & 0xFF;
            fpr_set_f32(lane, dec->rd, e5m2_to_fp32(v));
        }
        break;

    case 0x26: // E2M1
        if (dec->rs2 == 1) {
            // F32 → E2M1
            uint8_t v = fp32_to_e2m1(f_rs1);
            lane->fpr[dec->rd].u32 = (lane->fpr[dec->rd].u32 & 0xFFFFFF00) | v;
        } else {
            // E2M1 → F32 (rs2 == 0)
            uint8_t v = lane->fpr[dec->rs1].u32 & 0xFF;
            fpr_set_f32(lane, dec->rd, e2m1_to_fp32(v));
        }
        break;

    default:
        break;
    }
}

// ==================== 主解释器入口 ====================
static bool instru_interpret(uint32_t instr, GPGPULane *lane, GPGPUState *s) {
    bool pc_updated = false;

    lane->gpr[0].u32 = 0;  // x0 恒为 0

    GPGPURiscVDecode dec = gpgpu_riscv_decode(instr);

    switch (dec.opcode) {
        case 0x17:
            dec.imm = gpgpu_riscv_imm_u(instr);
            exec_auipc(&dec, lane);
            break;  // AUIPC
        case 0x13:
            dec.imm = gpgpu_riscv_imm_i(instr);
            exec_op_imm(&dec, lane);
            break;  // OP-IMM
        case 0x33:
            exec_op(&dec, lane);
            break;  // OP
        case 0x37:
            dec.imm = gpgpu_riscv_imm_u(instr);
            exec_lui(&dec, lane);
            break;  // LUI
        case 0x63:
            dec.imm = gpgpu_riscv_imm_b(instr);
            pc_updated = exec_branch(&dec, lane);
            break;  // BRANCH
        case 0x6f:
            dec.imm = gpgpu_riscv_imm_j(instr);
            pc_updated = exec_jal(&dec, lane);
            break;  // JAL
        case 0x67:
            dec.imm = gpgpu_riscv_imm_i(instr);
            pc_updated = exec_jalr(&dec, lane);
            break;  // JALR
        case 0x03:
            dec.imm = gpgpu_riscv_imm_i(instr);
            exec_load(&dec, lane, s);
            break;  // LOAD
        case 0x23:
            dec.imm = gpgpu_riscv_imm_s(instr);
            exec_store(&dec, lane, s);
            break;  // STORE
        case 0x53:
            exec_op_fp(&dec, lane);
            break;  // OP-FP
        case 0x73:
            exec_system(&dec, lane);
            break;  // SYSTEM
    }

    lane->gpr[0].u32 = 0;  // 锁定 x0
    return pc_updated;
}
/* TODO: Implement kernel dispatch and execution */
int gpgpu_core_exec_kernel(GPGPULane *lane, GPGPUState *s)
{
    uint32_t instr;

    if (!gpgpu_core_vram_load_u32(s, lane->pc, &instr)) {
        return -1;
    }
    if (instr == 0x00100073) return 1;
    if (!instru_interpret(instr, lane, s)) {
        lane->pc += 4;
    }
    return 0;
}
