/*
 * GPGPU neural-network operation descriptors
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#ifndef GPGPU_NN_H
#define GPGPU_NN_H

#include "gpgpu_runtime.h"
#include "gpgpu_tensor.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum GPGPUOpType {
    GPGPU_OP_CONV2D         = 0,
    GPGPU_OP_RELU           = 1,
    GPGPU_OP_MAXPOOL2D      = 2,
    GPGPU_OP_LINEAR         = 3,
    GPGPU_OP_LINEAR_PARTIAL = 4,
    GPGPU_OP_LINEAR_REDUCE  = 5,
    GPGPU_OP_MATMUL_PARTIAL = 6,
    GPGPU_OP_MATMUL_REDUCE  = 7,
    GPGPU_OP_IM2COL         = 8,
    GPGPU_OP_OIHW_TO_KO     = 9,
} GPGPUOpType;

typedef struct GPGPUConv2DArgs {
    GPGPUTensorDesc input;   /* Activation tensor, NCHW. */
    GPGPUTensorDesc weight;  /* Filter tensor, OIHW. */
    GPGPUTensorDesc bias;    /* Optional 1D tensor, length output channels. */
    GPGPUTensorDesc output;  /* Activation tensor, NCHW. */
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t dilation_h;
    uint32_t dilation_w;
    uint32_t groups;
} GPGPUConv2DArgs;

typedef struct GPGPUReluArgs {
    GPGPUTensorDesc input;
    GPGPUTensorDesc output;
} GPGPUReluArgs;

typedef struct GPGPUMaxPool2DArgs {
    GPGPUTensorDesc input;
    GPGPUTensorDesc output;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t stride_h;
    uint32_t stride_w;
} GPGPUMaxPool2DArgs;

typedef struct GPGPULinearArgs {
    GPGPUTensorDesc input;
    GPGPUTensorDesc weight;  /* Weight matrix, OI: [out_features, in_features]. */
    GPGPUTensorDesc bias;
    GPGPUTensorDesc output;
    uint32_t in_features;
    uint32_t out_features;
} GPGPULinearArgs;

typedef struct GPGPULinearPartialArgs {
    GPGPUTensorDesc input;
    GPGPUTensorDesc weight;   /* Weight matrix, OI: [out_features, in_features]. */
    GPGPUTensorDesc partial;  /* Partial products, OI: [out_features, in_features]. */
    uint32_t in_features;
    uint32_t out_features;
} GPGPULinearPartialArgs;

typedef struct GPGPULinearReduceArgs {
    GPGPUTensorDesc partial;  /* Partial products, OI: [out_features, in_features]. */
    GPGPUTensorDesc bias;
    GPGPUTensorDesc output;
    uint32_t in_features;
    uint32_t out_features;
} GPGPULinearReduceArgs;

typedef struct GPGPUMatmulPartialArgs {
    GPGPUTensorDesc a;        /* MK matrix: [m, k]. */
    GPGPUTensorDesc b;        /* KO matrix: [k, o]. */
    GPGPUTensorDesc partial;  /* Flat scratch: [m * o * k]. */
    uint32_t m;
    uint32_t k;
    uint32_t o;
} GPGPUMatmulPartialArgs;

typedef struct GPGPUMatmulReduceArgs {
    GPGPUTensorDesc partial;  /* Flat scratch: [m * o * k]. */
    GPGPUTensorDesc c;        /* MO matrix: [m, o]. */
    GPGPUTensorDesc bias;     /* Optional 1D tensor, length o. */
    uint32_t m;
    uint32_t k;
    uint32_t o;
    uint32_t output_shift;    /* Right shift after reduction. 0 keeps raw i32. */
    uint32_t has_bias;
} GPGPUMatmulReduceArgs;

typedef struct GPGPUIm2ColArgs {
    GPGPUTensorDesc input;   /* NCHW tensor. */
    GPGPUTensorDesc output;  /* MK matrix: [n * out_h * out_w, c * kernel_h * kernel_w]. */
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t out_h;
    uint32_t out_w;
} GPGPUIm2ColArgs;

typedef struct GPGPUOihwToKoArgs {
    GPGPUTensorDesc input;   /* OIHW tensor. */
    GPGPUTensorDesc output;  /* KO matrix: [in_channels * kernel_h * kernel_w, out_channels]. */
    uint32_t out_channels;
    uint32_t in_channels;
    uint32_t kernel_h;
    uint32_t kernel_w;
} GPGPUOihwToKoArgs;

typedef struct GPGPUNodeDesc {
    uint32_t op_type;       /* GPGPUOpType */
    uint32_t kernel_addr;   /* VRAM offset of compiled device kernel code. */
    uint32_t args_addr;     /* VRAM offset of op-specific args struct. */
    GPGPURuntimeDim3 grid;
    GPGPURuntimeDim3 block;
} GPGPUNodeDesc;

typedef struct GPGPUNetworkDesc {
    GPGPUNodeDesc *nodes;   /* Host-side execution plan; not a VRAM pointer. */
    uint32_t num_nodes;
} GPGPUNetworkDesc;

static inline GPGPUConv2DArgs gpgpu_conv2d_args_make(
    GPGPUTensorDesc input,
    GPGPUTensorDesc weight,
    GPGPUTensorDesc bias,
    GPGPUTensorDesc output,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t stride_h,
    uint32_t stride_w)
{
    return (GPGPUConv2DArgs) {
        .input = input,
        .weight = weight,
        .bias = bias,
        .output = output,
        .pad_h = pad_h,
        .pad_w = pad_w,
        .stride_h = stride_h,
        .stride_w = stride_w,
        .dilation_h = 1,
        .dilation_w = 1,
        .groups = 1,
    };
}

#ifdef __cplusplus
}
#endif

#endif /* GPGPU_NN_H */
