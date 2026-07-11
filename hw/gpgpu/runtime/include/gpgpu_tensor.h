/*
 * GPGPU tensor descriptor ABI
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#ifndef GPGPU_TENSOR_H
#define GPGPU_TENSOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum GPGPUDType {
    GPGPU_DTYPE_I32 = 0,
    GPGPU_DTYPE_F32 = 1,
    GPGPU_DTYPE_I8  = 2,
    GPGPU_DTYPE_U8  = 3,
} GPGPUDType;

typedef enum GPGPUTensorLayout {
    GPGPU_LAYOUT_NCHW = 0,
    GPGPU_LAYOUT_NHWC = 1,
    GPGPU_LAYOUT_OIHW = 2,
    GPGPU_LAYOUT_1D   = 3,
    GPGPU_LAYOUT_OI   = 4,
} GPGPUTensorLayout;

typedef struct GPGPUTensorDesc {
    uint32_t data;      /* VRAM offset to element 0 */
    uint32_t dtype;     /* GPGPUDType */
    uint32_t layout;    /* GPGPUTensorLayout */
    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;
    uint32_t stride_n;  /* Strides are measured in elements, not bytes. */
    uint32_t stride_c;
    uint32_t stride_h;
    uint32_t stride_w;
} GPGPUTensorDesc;

static inline GPGPUTensorDesc gpgpu_tensor_make_nchw_i32(uint32_t data,
                                                         uint32_t n,
                                                         uint32_t c,
                                                         uint32_t h,
                                                         uint32_t w)
{
    return (GPGPUTensorDesc) {
        .data = data,
        .dtype = GPGPU_DTYPE_I32,
        .layout = GPGPU_LAYOUT_NCHW,
        .n = n,
        .c = c,
        .h = h,
        .w = w,
        .stride_w = 1,
        .stride_h = w,
        .stride_c = h * w,
        .stride_n = c * h * w,
    };
}

static inline GPGPUTensorDesc gpgpu_tensor_make_oihw_i32(uint32_t data,
                                                         uint32_t out_channels,
                                                         uint32_t in_channels,
                                                         uint32_t kernel_h,
                                                         uint32_t kernel_w)
{
    return (GPGPUTensorDesc) {
        .data = data,
        .dtype = GPGPU_DTYPE_I32,
        .layout = GPGPU_LAYOUT_OIHW,
        .n = out_channels,
        .c = in_channels,
        .h = kernel_h,
        .w = kernel_w,
        .stride_w = 1,
        .stride_h = kernel_w,
        .stride_c = kernel_h * kernel_w,
        .stride_n = in_channels * kernel_h * kernel_w,
    };
}

static inline GPGPUTensorDesc gpgpu_tensor_make_1d_i32(uint32_t data,
                                                       uint32_t elements)
{
    return (GPGPUTensorDesc) {
        .data = data,
        .dtype = GPGPU_DTYPE_I32,
        .layout = GPGPU_LAYOUT_1D,
        .n = 1,
        .c = 1,
        .h = 1,
        .w = elements,
        .stride_w = 1,
        .stride_h = elements,
        .stride_c = elements,
        .stride_n = elements,
    };
}

static inline GPGPUTensorDesc gpgpu_tensor_make_oi_i32(uint32_t data,
                                                       uint32_t out_features,
                                                       uint32_t in_features)
{
    return (GPGPUTensorDesc) {
        .data = data,
        .dtype = GPGPU_DTYPE_I32,
        .layout = GPGPU_LAYOUT_OI,
        .n = out_features,
        .c = in_features,
        .h = 1,
        .w = 1,
        .stride_w = 1,
        .stride_h = 1,
        .stride_c = 1,
        .stride_n = in_features,
    };
}

static inline uint32_t gpgpu_tensor_numel(const GPGPUTensorDesc *tensor)
{
    return tensor->n * tensor->c * tensor->h * tensor->w;
}

static inline uint32_t gpgpu_tensor_offset_nchw(const GPGPUTensorDesc *tensor,
                                                uint32_t n, uint32_t c,
                                                uint32_t h, uint32_t w)
{
    return n * tensor->stride_n +
           c * tensor->stride_c +
           h * tensor->stride_h +
           w * tensor->stride_w;
}

#ifdef __cplusplus
}
#endif

#endif /* GPGPU_TENSOR_H */
