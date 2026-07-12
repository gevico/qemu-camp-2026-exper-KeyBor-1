/*
 * QEMU GPGPU - RISC-V instruction decode helpers
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#ifndef HW_GPGPU_RISCV_H
#define HW_GPGPU_RISCV_H

#include "qemu/osdep.h"

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

static inline int32_t gpgpu_riscv_sign_extend(uint32_t value, unsigned bits)
{
    return ((int32_t)(value << (32 - bits))) >> (32 - bits);
}

static inline GPGPURiscVDecode gpgpu_riscv_decode(uint32_t instr)
{
    return (GPGPURiscVDecode) {
        .instr = instr,
        .opcode = instr & 0x7f,
        .rd = (instr >> 7) & 0x1f,
        .funct3 = (instr >> 12) & 0x7,
        .rs1 = (instr >> 15) & 0x1f,
        .rs2 = (instr >> 20) & 0x1f,
        .funct7 = (instr >> 25) & 0x7f,
        .funct2 = (instr >> 25) & 0x3,
        .rs3 = (instr >> 27) & 0x1f,
        .csr = (instr >> 20) & 0xfff,
    };
}

static inline int32_t gpgpu_riscv_imm_i(uint32_t instr)
{
    return gpgpu_riscv_sign_extend((instr >> 20) & 0xfff, 12);
}

static inline int32_t gpgpu_riscv_imm_s(uint32_t instr)
{
    uint32_t imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1f);
    return gpgpu_riscv_sign_extend(imm & 0xfff, 12);
}

static inline int32_t gpgpu_riscv_imm_b(uint32_t instr)
{
    uint32_t imm = (((instr >> 31) & 0x1) << 12) |
                   (((instr >> 7) & 0x1) << 11) |
                   (((instr >> 25) & 0x3f) << 5) |
                   (((instr >> 8) & 0xf) << 1);
    return gpgpu_riscv_sign_extend(imm, 13);
}

static inline int32_t gpgpu_riscv_imm_u(uint32_t instr)
{
    return (int32_t)(instr & 0xfffff000u);
}

static inline int32_t gpgpu_riscv_imm_j(uint32_t instr)
{
    uint32_t imm = (((instr >> 31) & 0x1) << 20) |
                   (((instr >> 12) & 0xff) << 12) |
                   (((instr >> 20) & 0x1) << 11) |
                   (((instr >> 21) & 0x3ff) << 1);
    return gpgpu_riscv_sign_extend(imm, 21);
}

#endif /* HW_GPGPU_RISCV_H */
